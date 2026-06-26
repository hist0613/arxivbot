New uploads on arXiv(cs.CL)

### Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipelin (https://arxiv.org/abs/2606.27347)
Comments:
          34 pages, 17 figures

- **Prior Approaches**: 기존 정치 엘리트 네트워크 연구는 Archigos, WhoGov처럼 공식 직위 중심의 구조화 데이터에 의존해 비공식 제휴·경제적 연결·적대 관계를 놓치기 쉬웠습니다. 텍스트 기반 자동화는 co-occurrence 같은 단순 신호나 문서 단위 분류에 머물러, 문서 내부의 ‘개체-쌍 관계’와 서명(signed)·방향·타입·시간성을 함께 추출하기가 어려웠습니다. LLM 접근도 흔히 proprietary API 의존, 단일 언어 설계, 외부 지식베이스 고정 해석(엔터티 해상도) 부재 때문에 국가 간 비교·재현성에서 한계가 컸습니다.

- **Core Contribution**: 이 논문은 멀티링구얼 joint entity-relation extraction을 위한 모듈형·완전 open-weight 파이프라인을 제안합니다. 뉴스의 비정형 텍스트에서 signed·temporal 지식그래프를 만들며, span 기반 NER과 3단계 linking cascade로 언어 비의존 Wikidata 식별자(QID)에 노드를 고정합니다. 또한 109개 엔터티 타입·99개 관계 타입을 갖는 SKOS 온톨로지로 관계 추출을 ontology-constrained guided decoding 하에 수행해, 연구 목적에 맞게 분류체계를 교체 가능하게 했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 대규모 다국어 텍스트에서 형태 변이·약어·인명/기관 표기를 안정적으로 QID에 정렬하고, (2) 관계 타입·방향·서명을 생성 과정에서 일관되게 강제하며, (3) 수백만 규모 처리에서 추론 비용을 감당하는 것이었습니다. 이를 위해 Exact/Fuzzy/Vector 검색의 3단계 엔터티링크(별칭 인덱스·문자 편집·Qdrant 벡터 검색)와, vLLM 서빙 기반 mixture-of-experts 모델의 guided decoding·고정 어휘 제약(문법 마스킹) 및 후보 인덱스 선택으로 QID 환각을 차단했습니다. 관계는 문서 근거 기반으로 협력/갈등 서명과 시간 스코프(event/state/property)를 함께 산출하고, 타입 방향 제약 위반 시 subject–object를 자동 교정하는 후처리를 붙였습니다.

- **Empirical Impact**: 관계 추출 정확도는 3491개 골드 표준(폴란드 기사 502개 중 테스트 252개 기반) 대비 full-coverage spot-check에서 strict 68.2%, lenient 93.7%의 텍스트 정확성을 보였습니다. NER은 사람 검증 기반 독립 평가에서 F1 83.8%(precision 85.5%, recall 82.3%)로, 엔터티 감지 단계의 품질도 뒷받침됩니다. 오스트리아 사례에서는 정당 BZÖ의 분열 시점부터 후속 세력·법원 판결까지 ‘수명주기’를 재구성했고, 폴란드 사례에서는 국영 기업을 둘러싼 경제-거버넌스 O-group 중첩과 PO–PiS 간 구조적으로 균형 잡힌 signed 갈등 네트워크를 드러내며 cross-national computational social science의 재현 가능한 기반을 제공한다는 점을 입증했습니다.



### Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning (https://arxiv.org/abs/2606.27330)
Comments:
          Accepted to ACL 2026 Main

- **Prior Approaches**: 기존 멀티모달 웹 에이전트 연구는 반복 GUI 작업에서 과업을 클릭·입력·스크롤 같은 원자 단위로 학습시키거나, 대략적인 high-level 과업(거친 조건 포함) 궤적을 이용해 후학습을 수행해 왔다. 그러나 원자 단위 학습은 고수준 계획으로의 compositional generalization(조합 일반화)이 약하고, coarse high-level 궤적은 환경 정합성·제약 조건이 느슨해 OOD(미지 사이트) 일반화가 제한된다.

- **Core Contribution**: 이 논문은 planning experience exploration and utilization(PEEU)로, 에이전트가 미지 웹사이트를 자율 탐색해 경험을 수집하고 hindsight 경험을 이용해 “엄격히 정렬(aligned)된 고수준 훈련 데이터”를 합성하는 방법을 제안한다. 또한 task decomposition hierarchical analysis framework(TDHAF)로 저수준-중간-고수준 과업 분해의 일반화를 ID(사내)와 OOD(외부)로 나눠 체계적으로 진단한다.

- **Technical Challenges**: 핵심 기술 난제는 탐색 궤적에서 도출된 고수준 과업이 실제 결과와 어긋나거나(미스매치), 미관측 환경 정보 때문에 엄격한 제약을 포함하지 못한다는 점이다. PEEU는 행동 전후의 시각 상태를 비교해 atomic 경험을 추출한 뒤, hindsight로 고수준 과업-궤적 쌍을 더 정밀하게 재정렬하고 제약을 강화하는 2단계(탐색 트리 구성→경험 활용) 파이프라인을 설계한다.

- **Empirical Impact**: 실험은 WebVoyager의 7개 미지 사이트에서 cross-website OOD 일반화를 평가했으며, 동일 데이터 스케일 조건에서 Qwen2.5-VL-7B 기반 PEEU가 30.6% 정확도로 Qwen2.5-VL-32B(22.7%)를 포함해 기존 기준선을 크게 앞섰다. TDHAF 분석 결과, 저수준 원자 스킬 숙달만으로는 고수준 계획 능력이 보장되지 않고, 고수준 과업 훈련이 ID·OOD 모두에서 더 높은 coverage를 만든다는 점을 정량적으로 입증해 작은 모델의 OOD 계획 성능 향상에 핵심이 “정렬된 고수준 hindsight 학습”임을 보여준다.



### LLM-Based Examination of Eligibility Criteria from Securities Prospectuses at the German Central Bank (https://arxiv.org/abs/2606.27316)
- **Prior Approaches**: 기존 연구는 담보 적격성 판단을 Named Entity Recognition(NER) 문제로 바꿔 Transformer 기반 토큰 분류로 처리했다. 다만 OCR 잡음, 언어 변이(독일-영어 혼재), 그리고 텍스트 span 경계에 민감해져 데이터/품질 변화에 취약하다는 한계가 있었다. 또한 관련 애노테이션 타입마다 대규모 수동 라벨링이 필요해 확장성도 떨어졌다.

- **Core Contribution**: 이 논문은 독일 중앙은행의 담보 적격성 심사를 Large Language Models(LLMs)를 이용한 생성형 Information Extraction 파이프라인으로 전환하는 최초의 케이스 스터디를 제시한다. 작업을 추출-정규화-해석의 다단계로 분해해 잡음이 섞인 문장과 독일-영어가 섞인 문서에서도 유연하게 처리하도록 설계했다. 아울러 LLM-as-a-judge를 통해 위치 기반이 아닌 값(semantic) 기반 평가 방법을 도입했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 문서가 길고 반구조화돼 있으며 (2) OCR 아티팩트와 레이아웃 변화로 원문 텍스트가 흔들리고 (3) 독일-영어가 병기/삽입되는 상황에서 정확한 값과 증거를 일관되게 뽑아내는 것이다. 연구진은 PDF를 Docling 기반 Markdown으로 변환·정규화해 추출 안정성을 높이고, “simple” 기준은 2단계(추출→해석), “complex” 기준은 추출과 해석을 1단계로 처리한 뒤 Python 로직으로 최종 적격성(6개 기준 충족 여부)을 산정했다. 검증 단계에서는 Structured output로 JSON 스키마를 강제하고, 생성 오류를 줄이기 위해 frequency penalty와 낮은 temperature를 적용했다.

- **Empirical Impact**: 실험 결과 LLM 기반 시스템은 문서 단위 적격성에서 최대 91% 수준의 높은 precision을 보였으며, false acceptance를 줄이는 보수적 운영 성향이 나타났다. Llama-3.3-70B-Instruct와 Cohere Command-R 08-2024(둘 다 instruction-following·grounded generation 성격)를 비교했을 때 두 모델은 전반적으로 유사 성능을 보이며 Command-R 08-2024가 근소하게 앞섰다. 특히 위치 기반 평가의 한계를 보완한 value-based 평가(LLM-as-a-judge 포함)가 의미적 동치(예: “subordinated” vs “nachrangig”)를 더 잘 반영해, 담보 적격성 같은 규정 준수 작업에서 실무적 확장 가능성을 보여줬다.



### Beyond Surface Forms: A Comprehensive, Mechanism-Oriented Taxonomy of Indirect Linguistic Encoding for LLM-Based Coded Language Detection (https://arxiv.org/abs/2606.27314)
Comments:
          Submitted for review in ARR for EMNLP 2026

- **Prior Approaches**: 기존 연구는 algospeak, euphemisms, adversarial obfuscation처럼 의도나 표면 양식에 따라 우회 표현을 분류하는 경향이 강했다. 하지만 이런 목표·맥락 중심 분류는 표현이 진화할 때 새 패턴을 안정적으로 포착하기 어렵다는 한계가 있었다. 또한 “무엇이 감춰졌는가”보다 “어떻게 부호화·복호화되는가”의 공통 연산을 체계화하기가 부족했다.

- **Core Contribution**: 이 논문은 ILE(간접 언어 표현, Indirect Linguistic Expressions)을 의사소통 목표에서 분리해, 의미가 인코딩되고 회수되는 근본 연산을 기준으로 하는 메커니즘 지향 분류( taxonomy )를 제안한다. 즉, 관측되는 명칭이나 의도 추정 대신 의미 인코딩/복구의 조작을 중심으로 전체 스펙트럼을 정리한다. 이를 통해 향후 등장하는 코드화 언어를 “기술적 골격” 위에 올려 탐지하도록 설계했다.

- **Technical Challenges**: 메커니즘 지향으로 분류하려면 표현마다 겉모습이 달라도 동일한 연산 패턴을 일관되게 라벨링하고, LLM 입력으로도 안정적으로 재현해야 했다. 논문은 해당 메커니즘 taxonomy를 LLM 프롬프트에 통합하고, 2,000개의 TikTok 및 Bluesky 게시물을 수동 주석 기반으로 검증하는 방식으로 이를 해결했다. 또한 기존 4개 taxonomy와 taxonomy가 없는 기준선까지 함께 비교해, 분류 체계 자체의 기여를 분리해 평가했다.

- **Empirical Impact**: 실험 결과, 제안한 메커니즘 지향 taxonomy는 세 가지 LLM에서 문서 수준과 span 수준 모두에서 가장 높은 성능을 보였고, 정확도 4.7%, F1 5.4%의 개선을 달성했다. 특히 기존 벤치마크 최강 대비 향상 폭이 관측되어, 새로운 코드화 언어가 등장해도 견고한 탐지 기반이 될 수 있음을 시사한다. 저자들은 moderation(콘텐츠 중재)에 입력으로도 유용한 스캐폴드로 작동한다고 해석한다.



### Multilingual Reasoning Cascades Need More Contex (https://arxiv.org/abs/2606.27306)
- **Prior Approaches**: 기존 Translation cascades for reasoning은 질문을 다른 언어에서 영어로 번역한 뒤 영어로 추론하고, 결과를 다시 원 언어로 번역한다. 그러나 단계마다 정보가 소실돼 cultural grounding, register, disambiguation 같은 후속 단계에 필요한 단서가 사라지며 오류 전파가 구조적으로 커질 수 있다.

- **Core Contribution**: 논문은 학습 없이 적용 가능한 context-aware translation cascade를 제안한다. 원문 질문, 영어로 번역된 질문, reasoning trace를 최종 번역 모듈에 함께 제공해 정보 흐름을 보존하면서 구조적 손실을 완화한다.

- **Technical Challenges**: 핵심 과제는 번역-추론-번역 파이프라인에서 어떤 정보를 어디까지 전달해야 오류 전파를 줄이면서 성능을 올릴 수 있는지다. 저자들은 최종 모듈이 필요로 하는 컨텍스트를 추가로 주입하는 방식으로 해결했고, 특히 원문 질문이 유익한 컨텍스트의 대부분을 제공한다는 점을 확인했다.

- **Empirical Impact**: 9개 다국어 벤치마크(다양한 태스크 유형)와 3개 backbone, 285개 고·중·저자원 언어에서 평가했으며 전반적으로 강한 개선을 보였다. 특히 open-ended generation에서 모델과 자원 수준 전반에 걸쳐 일관된 향상이 나타나, “파이프라인 끝까지 원문 사용자 질문을 보존”하는 단순한 기본 전략이 실무적으로 유용함을 보여준다.



### How Surprising Is Historical Italian to Language Models? Tokenization Tax, Comprehension Tax, and a Simple Mitigation (https://arxiv.org/abs/2606.27275)
Comments:
          The 22nd Conference on Information and Research Science Connecting to Digital and Library Science

- **Prior Approaches**: 기존 연구는 역사 문장을 LLM에 넣을 때 생기는 어려움을 단일한 ‘historical difficulty’로 뭉뚱그려, 맞춤법 변이·언어적 거리·사전노출(노출 편향)을 섞어 설명하는 경향이 있었다. 또한 토크나이징 비용(토큰 분절)과 생성/예측의 불확실성이 동시에 나빠진다고 가정해, 어떤 단계에서 문제가 생기는지 분해가 부족했다.

- **Core Contribution**: 논문은 역사 문장의 어려움을 tokenization cost(토큰화 비용)·predictive uncertainty(surprisal/퍼플렉서티)·semantic robustness(임베딩 유사도)·context sensitivity(최소 시간 프롬프트 효과)의 4차원으로 진단하는 프레임워크를 제안한다. 이로써 ‘예측이 흔들린다’는 신호가 곧 ‘의미 표현이 망가진다’로 직결되는지 검증할 수 있게 했다.

- **Technical Challenges**: 가장 큰 난제는 사전노출(confound) 문제를 통제하면서도 실제로 어려운 역사 텍스트를 비교 가능한 형태로 만들기다. 저자들은 Manzoni의 Quarantana(고노출 기준), 17세기 이탈리아(신규 큐레이션, long s 등)와 18세기 러시아(강한 정서법 스트레스)로 대비 설계를 하고, 현대화 기준문을 통해 퍼플렉서티 비율과 토큰 인플레이션을 같은 토크나이저에서 측정했다.

- **Empirical Impact**: 실험 결과, 인코딩 비용은 비슷해도(이탈리아 17세기와 러시아 모두 토큰화 페널티 약 25~30%) 예측 난이도는 크게 갈렸고, 17세기 이탈리아는 평균 2.4배 더 surprising(학술 산문 최대 3.2배)한 반면 러시아는 증가폭이 상대적으로 작았다. 더 중요한 점은 예측 불안정이 의미 손상으로 이어지지 않아, 역사-현대 문장 임베딩 유사도는 전 데이터셋에서 0.85 이상으로 유지되었으며, 최소 시간 프롬프트만으로도 historical surprisal을 약 60%가량 줄여 디지털 라이브러리의 검색·인덱싱 같은 retrieval 작업에 LLM을 안전하게 적용할 근거를 제공한다.



### LMs as Task-Specific Knowledge Bases: An Interpretability Analysis (https://arxiv.org/abs/2606.27237)
- **Prior Approaches**: 언어모델은 파라미터에 지식이 축적된다는 관점에서, 지식을 ‘knowledge base’처럼 취급해 왔습니다. 이에 따라 같은 사실에 대해 서로 다른 질의(예: 빈칸 채우기 vs 질문형)를 해도 일관된 답이 나오기를 기대합니다. 하지만 기존 연구는 주로 성능이나 지식 편집 효과를 다뤘고, 서로 다른 task 포맷에서 같은 사실이 동일한 내부 ‘출처’를 쓰는지까지는 충분히 검증하지 못했습니다.

- **Core Contribution**: 이 논문은 task-invariance(동일 사실의 task 간 일관된 인출)가 언어모델에서 성립하는지 behavioral·mechanistic 분석으로 정면 평가합니다. 먼저 훈련 체크포인트를 따라 (subject, relation, object) 사실의 co-emergence가 task 간에 얼마나 옮겨가는지 추적하고, 이어서 같은 사실이 task 포맷에 따라 서로 다른 파라미터 부분집합에 의해 지지되는지 국소화합니다. 결론적으로 모델이 ‘task-invariant 지식 저장소’를 갖는다기보다, 사실을 task별 인코딩으로 분절해 저장한다는 증거를 제시합니다.

- **Technical Challenges**: 핵심 난제는 (1) 사실별·task별로 훈련 중 지식 습득 시점을 안정적으로 정의하고, (2) 그 지식이 실제로 어떤 파라미터에 의존하는지 정밀하게 분리하는 것입니다. 저자들은 체크포인트마다 올바른 답의 확률을 기회수준으로 정규화해 emergence를 정의하고, 기계적 분석에서는 attention heads와 MLP neuron에 대해 sparse binary mask를 학습해 필요성(neccesary)·충분성(sufficiency)·특이성(specificity)·희소성(sparsity)을 동시에 만족하도록 설계했습니다. 또한 chain-of-thought가 어떤 파라미터를 더 ‘활성화’해 효과를 얻는지 확인하기 위해, localized (fact, task) 인코딩을 제거하는 ablation을 수행했습니다.

- **Empirical Impact**: 실험 결과, (fact, task) 쌍의 약 47.9%에서 예측된 co-emergence가 성립하지 않아 task 간 지식 전이가 불안정함을 보여줍니다. 통계적으로도 task와 사실의 상호작용이 모든 체크포인트에서 유의하게 존재해 task-invariant 가설을 기각합니다. 파라미터 국소화에서는 같은 사실이라도 task별로 distinct subset이 발견되며, 특히 discrimination task가 generation task보다 더 높은 cross-task entanglement를 보입니다. 이는 ‘지식의 신뢰성과 제어 가능성은 task 불변성을 전제한다’는 knowledge base 비유를 약화시키며, 단일 포맷 평가나 특정 포맷만 겨냥한 편집/정교화가 다른 포맷의 지식 동작을 보장하지 못할 수 있음을 시사합니다.



### Bridging Talk and Thought: Understanding Dialogue Dynamics Across Collaborative Problem-Solving Contexts (https://arxiv.org/abs/2606.27233)
- **Prior Approaches**: 기존 대화 연구는 담화 행위, 대화 상태, 담화 구조를 분류해 의도나 품질을 분석하는 데 집중해 왔습니다. CPS(협력적 문제 해결) 관점에서는 탐색적 발화와 사회정서 조절, 상호작용 대칭성 같은 틀로 협업을 보려 했지만, 인지·메타인지·사회정서가 한 흐름에서 어떻게 함께 작동하는지는 충분히 계량화되지 않았습니다. 특히 human–AI 협업은 종종 협력이라기보다 tool use나 반응형 보조에 그친다는 메타분석 결과가 반복되었지만, 이를 대화 패턴으로 구체적으로 분해해 진단하는 체계는 부족했습니다.

- **Core Contribution**: 이 논문은 CPS 대화를 metacognition(메타인지)·cognition(인지)·non-cognition(비인지/사회정서) 세 축으로 동시에 코딩하는 계층형 2-layer 체계를 제안합니다. 발화 단위로 조절 과정(계획·모니터링·평가)과 발화가 담당하는 조절 수준(SR/CR/SSR), 인지적 목표지향 발화 유형(7종), 사회정서 상호작용(3종)을 함께 라벨링해 “진짜 협업”과 “도구적 협력/비대칭”을 분리합니다. 나아가 메타인지 조절 불균형이 협업의 깊이를 가르는 핵심 판별자일 수 있음을 실증적으로 강조합니다.

- **Technical Challenges**: 핵심 난제는 CPS 대화에서 서로 다른 차원의 현상을 한 번에 안정적으로 식별하는 것입니다. 이를 위해 연구진은 메타인지 조절을 7개 과정과 3개 행동 수준으로 세분화하고, 주변 문맥까지 고려해 off-task 발화의 기능적 역할도 반영하도록 코딩 지침을 구성했습니다. 규모 확장도 필요했는데, GPT-4o를 LLM-as-a-judge로 활용해 발화별 라벨을 자동화하고, 사람 라벨과의 kappa로 일치도를 검증해 재현성을 확보했습니다.

- **Empirical Impact**: 9개 CPS 데이터셋(인간-인간 6종, 인간-AI 3종)에 대해 테스트한 결과, 조절·인지·사회정서 기여가 균형을 이룰 때 대화가 협업적으로 나타났습니다. 반대로 Minecraft·Itinerary처럼 맥락이 구조화된 과제에서는 한 참여자가 metacognitive leadership(메타인지 주도) 역할을 맡고 다른 쪽은 반응·실행 중심이 되는 비대칭이 관찰됐습니다. 특히 CoCoDial 계열에서는 인간이 self-regulation을 지배하고 AI는 co-regulated/SSR 쪽이 편향적으로 나타나, AI가 자율적 주도권을 공유하지 못해 협업이 얕아지는 이유를 설명합니다. 또한 metacognitive regulation 라벨을 제거하면 화자 역할 군집이 섞여 “메타인지 조절”이 협업 모델링의 필수 신호임을 보여줍니다.



### CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention (https://arxiv.org/abs/2606.27229)
Comments:
          27 pages, 2 figures, multiple tables. Submitted to arXiv. Primary category: cs.LG; cross-list: cs.CL

- **Prior Approaches**: Transformer는 모든 과거 토큰을 KV-cache로 유지해 성능은 강하지만, 학습·추론 비용이 길이에 따라 급격히 커집니다. 이에 비해 delta-rule 기반 순환 모델은 고정 크기 상태로 압축해 O(1) 메모리·비용에 가깝게 확장하지만, 성능을 좌우하는 “잊기” 게이트 설계에서 한계가 반복돼 왔습니다. 특히 GDN-2는 erase/write를 입력 토큰만 보고 결정해 메모리를 직접 참조하지 못하는 memory-blind gating 문제와, value-axis 결합으로 WY-form chunk solver의 청크 병렬성을 깨는 구조적 병목이 함께 존재했습니다.

- **Core Contribution**: CARVE(Content-Aware Recurrent with Value Efficiency)는 기존 delta-rule 계열의 결함 3가지를 하나의 원칙으로 정리합니다: erase는 key axis에만 수행합니다. 이 제약은 단순한 휴리스틱이 아니라, WY-form triangular chunk solver의 타당성을 유지하기 위한 필요충분 조건으로 증명됩니다. 또한 CARVE는 erase 게이트에 “저장된 내용” 신호를 제공하되 추가 메모리 읽기 없이, 이미 GPU에 써둔 recurrent output을 재활용해 content-aware forgetting을 구현합니다.

- **Technical Challenges**: 핵심 기술 난점은 게이트가 메모리 상태를 “읽어야” content-aware가 가능한데, HBM에서 상태 행렬을 다시 읽으면 순환 모델의 효율 이점이 상쇄된다는 점입니다. CARVE는 매 반복에서 HBM에 기록되는 output o_t=S_{t-1} q_t를 chunk 평균 m_c로 요약해 erase 게이트의 조건으로 사용함으로써, stale 신호의 오차가 chunk 길이 L에 따라 O(1/sqrt(L)) 수준으로 감소함을 이론적으로 보입니다. 더 나아가 write 측에서는 per-value write-gate projection을 head당 단일 scalar로 대체해 파라미터를 크게 줄이면서도 연관 저장 용량은 잃지 않음을 정리로 보장합니다.

- **Empirical Impact**: CARVE는 1.3B 파라미터를 FineWeb-Edu 100B 토큰으로 학습했을 때 WikiText perplexity 15.72를 달성했으며, GDN-2 대비 -0.18(4.5-sigma)을 기록했습니다. 또한 9개 상식 추론 벤치마크에서 모든 recurrent baseline을 앞섰고, RULER in-context retrieval 프로브에서는 모든 설정에서 state of the art를 달성했습니다. 하드웨어 관점에서도 throughput 오버헤드는 0.4% 이내, 피크 메모리는 13% 감소, 파라미터는 19% 적어져 delta-rule의 “품질-비용” 균형을 재정의했다는 평가를 받습니다.



### Compositionality and the lexicon in evolutionary semantics (https://arxiv.org/abs/2606.27228)
- **Prior Approaches**: 기존 진화 의미론은 의미 보편을 설명할 때 보통 단어들을 고정된 시그널 묶음이나 비구조적 표현으로 다루거나, 문장 의미를 구성 규칙보다는 신호-의미 매핑의 전역 특성으로만 요약해왔다. 또한 formal semantics의 compositionality 통찰을 가져오더라도, 렉시콘과 합성 함수 중 의미를 어디에 배치하느냐(lexical vs composition) 차이가 진화 모델의 ‘표현 복잡도’ 해석을 흔들 수 있었다. 그 결과 정량(quantification) 보편인 conservativity의 기원이 명확히 설명되기 어려운 한계가 있었다.

- **Core Contribution**: 이 논문은 formal semantics의 핵심 아이디어(부분 의미의 재귀적 합성)를 진화 모델에 결합해, 렉시컬 의미와 composition function이 동시에 co-evolve하도록 하는 프레임워크를 제안한다. 사례 연구로는 quantificational meaning의 진화에서 보편 universal인 conservativity가 어떻게 효율적으로 나타나는지, 그리고 어떤 형태로 문장 합성에 ‘내재’되는지 분석한다. Pareto frontier 관점에서 언어가 단순성(전역 압축)과 전달 정확도(의사소통 성공) 사이 최적 타협을 보일 때 conservativity 추상화가 가장 유리하게 등장한다고 보고한다.

- **Technical Challenges**: 핵심 기술적 난제는 “conservativity”를 기존처럼 개별 결정사(determiner) 의미의 성질로만 고정하면 진화 압축의 원인이 composition 단계의 전역 구조로부터 오는지를 놓칠 수 있다는 점이다. 저자들은 conservativity를 ‘결정사-논항 결합 방식’의 abstraction으로 재정의하고, composition function에 단 한 번 정의되면 전 문장에 동시에 파급되는 효과를 통해 복잡도 증가를 작게 만들면서도 정확도는 크게 올릴 수 있음을 설계한다. 이를 위해 렉시콘(일부 결정사 의미)과 합성 규칙 중 𝒞_Q를 문법으로 정의된 공간에서 바꾸어 tradeoff를 탐색하고, sender/receiver의 신호-추론 게임에서 pragmatic enrichment(Gricean implicature 형태의 common ground 갱신)까지 포함해 현실적인 압력을 모사한다.

- **Empirical Impact**: 실험적으로는 보편 conservativity가 단순히 ‘그럴듯한 제약’이 아니라, 문장 합성에서 global compression을 제공하는 효율적 시스템 수준 추상화로서 Pareto 최적 지점에 위치함을 보인다. 또한 syntactic structure 민감성이 중요한 역할을 하며, 정량자(quantifier) learnability에 대한 경험적 근거와 기존 진화 모델 사이의 긴장을 완화하는 설명 틀을 제시한다. 더 넓게는 formal semantics의 문장 의미 그림을 진화 모델링에 생산적으로 결합하는 템플릿(전역 압축이 문법 범주 내부에서 일어나는 보편, syntactic argument의 의미 특수화, 렉시컬·합성 의미의 동시 진화)을 제공한다.



### Paved with True Intents: Intent-Aware Training Improves LLM Safety Classification Across Training Regimes (https://arxiv.org/abs/2606.27210)
- **Prior Approaches**: 기존 안전 분류기는 프롬프트를 바로 safe/harmful 라벨로 매핑하고, 중간에 ‘사용자 의도’를 모델 내부 해석에 맡기는 경우가 많았다. 의도-인지(pormpt-time) 기법이나 의도 기반 추론은 있었지만, 주로 model-generated reasoning에 의존해 사람이 쓴 의도를 압축·재사용하는 학습 신호로서의 효용은 불명확했다.

- **Core Contribution**: 본 논문은 사용자 의도를 사람 주석으로 명시화한 데이터셋 AIMS(Annotated Intents for Model Safety, 1,724개)와 함께, 안전 분류에서 의도를 중간 신호로 모델링하는 프레임을 제안한다. 의도-라벨을 함께 학습하되, 서로 다른 학습 목표(SFT, preference learning, reasoning distillation, reinforcement learning)에서 의도 신호가 실제로 안전을 얼마나 바꾸는지 체계적으로 비교한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘의도’를 학습 신호로 만들 때, 라벨은 맞더라도 의도가 부정확한 실패 모드를 어떻게 잡아내느냐였다. 저자들은 DPO에서 잘못된 의도(라벨 뒤집힘/라벨은 맞지만 의도 불충실)를 rejected로 구성하고, GRPO에서는 intent faithfulness를 보상으로 직접 검증해 정책을 의도-근거 추론으로 유도한다. 또한 사람이 쓴 의도가 LLM 생성 의도보다 일관성이 높지 않아(코사인 유사도 격차) 사람 주석 의도를 중심 감독으로 사용한다.

- **Empirical Impact**: AIMS 기반 intent-aware 학습은 5개 외부 안전 벤치마크에서 평균 F1 0.836을 기록하며, zero-shot LLM과 전용 safety guardrail을 모두 능가했다. 특히 GRPO에서 intent faithfulness를 직접 보상한 설정이 가장 높은 성능을 보였고, SFT·DPO·distillation 역시 의도 관련 실패 축을 다르게 보완하며 행동 변화가 관찰됐다. 또한 안전 분류기의 지연(latency)과 F1의 trade-off에서 intent-aware 모델들이 Pareto frontier를 형성해 실사용 관점의 효율성까지 강조한다.



### Syntactic Belief Update as the Driver of Garden Path Processing Difficulty (https://arxiv.org/abs/2606.27206)
- **Prior Approaches**: 기존 심리언어학에서는 문장 처리 어려움을 단어 예측의 정보이론으로 설명하려는 시도가 커졌고, 그 대표가 surprisal(어휘 서프라이잘)이다. 다만 garden path 문장에서는 LLM 기반 단어 surprisal이 임계 구간의 급격한 느려짐을 제대로 예측하지 못하고, NP/S·NP/Z·MV/RR 같은 난이도 서열도 재현하지 못한다.
또한 어휘를 제외한 ‘구문 surprisal’을 확장해도, CCG supertag 같은 국소적·부분 예측으로는 앞부분에서 굳힌 구문 해석을 뒤집는 재분석의 성격을 충분히 담지 못한다.

- **Core Contribution**: 이 논문은 garden path 처리 곤란이 ‘다음 단어 확률’이 아니라, 구문 트리들에 대한 사전 믿음(syntactic belief)을 업데이트해야 하는 비용에서 온다고 본다. 매 단어가 들어올 때마다 가능한 구문 트리의 확률분포를 갱신하고, critical word에서 사전 믿음이 크게 틀어질수록 인간의 읽기 시간이 더 늘어난다고 예측한다.
업데이트 크기는 generalized Rényi divergence로 측정하며, 핵심 주장으로 lexical item의 probability와는 독립적인(하지만 lexical item에 의해 유발되는) 구문적 변화량을 지표화한다.

- **Technical Challenges**: 구문 믿음 업데이트를 계산하려면, 각 접두어(prefix)에서 가능한 구문 구조 전체에 대한 분포 q(Y|prefix)와 p(Y|prefix+word)를 같은 구조 공간 위에서 비교할 수 있어야 한다. 그런데 self-paced reading 실험에서는 아직 보지 못한 접미어를 주변화해야 하는데 계산이 비현실적으로 커지므로, 학습 단계에서 MASK 토큰으로 접미어를 대체해 주변화 효과를 근사한다.
또한 비투영(non-projective) dependency tree 전체에 대한 조건부 확률과 Rényi divergence를 효율적으로 구하기 위해 edge-factored CRF를 두고, partition function은 Matrix-Tree Theorem 확장으로 determinant 형태로 계산하며, divergence 계산은 그래프 모델의 분할함수들을 활용하는 방식으로 처리한다.

- **Empirical Impact**: SAP(Syntactic Ambiguity Processing) 벤치마크의 self-paced reading 데이터를 대상으로, SBU는 NP/S < NP/Z < MV/RR의 난이도 서열을 다른 지표보다 더 잘 맞췄다. 특히 lexical surprisal과 달리 MV/RR을 과소평가하지 않으며, α( Rényi divergence 파라미터 )를 조절할수록 인간의 느려짐 크기에 더 근접했다.
상관 분석에서도 SBU가 두 baseline(어휘 surprisal, syntactic supertag surprisal)보다 더 높은 예측 상관을 보였고, 이는 구문 ‘예측’보다 구문 ‘믿음 업데이트’ 자체가 garden path 곤란을 설명하는 핵심 신호일 수 있음을 시사한다.



### Forecasting With LLMs: Improved Generalization Through Feature Steering (https://arxiv.org/abs/2606.27199)
- **Prior Approaches**: 기존에는 LLM의 forecasting에서 기준 시점 이후 정보가 섞이는 look-ahead bias를 줄이기 위해 모델 자체의 시간 제한(chronologically restricted), 입력 익명화(예: anonymization), 혹은 다양한 프롬프트/데이터 구성 변경 같은 방법이 주로 쓰였다. 다만 이런 접근은 내부에 어떤 시간 관련 표상이 쓰이는지, 그리고 그 표상을 실제로 조절하면 편향이 인과적으로 바뀌는지는 명확하지 않았다.

- **Core Contribution**: 이 논문은 Sparse Autoencoder(SAE)로 LLM 내부의 시간 인식(time-awareness)과 사후 유출(h look-ahead-biased)으로 보이는 특징을 찾아내고, 해당 특징을 생성 중에 증폭(amplify)해 인과성을 검증한다. 특히 prediction-market에서 발견한 시간 관련 특징이 다른 자유형(free-form) forecasting 과제에서도 편향을 줄이는지 확인함으로써, 해석 가능한 시간적 특징의 일반화 가능성을 보여준다.

- **Technical Challenges**: 핵심은 (1) “시간 인식”과 “미래를 본 것 같은 추론”을 구분해 SAE 특징을 선별하고, (2) 그 특징을 실제 생성 과정에서 수정했을 때 텍스트 수준의 look-ahead bias가 안정적으로 변화하는지 측정하는 것이다. 연구진은 prediction-market에서 모델이 시점 기준으로는 맞추되 최종 결과를 떠올리는 방식(또는 반대)을 기준으로 특징을 랭킹하고, 이후 M&A와 제약(Pharmaceutical) 성장-드라이버 예측처럼 out-of-sample predictability가 거의 0인 과제에서 응답 텍스트의 bias 신호를 직접 읽어내며 검증을 수행한다.

- **Empirical Impact**: 결과적으로 time-awareness로 연관된 특징을 증폭하면 두 자유형 forecasting 과제 전반에서 look-ahead bias가 유의하게 감소했지만, look-ahead-bias로 후보 선별된 특징을 조종하는 경우에는 같은 효과가 관찰되지 않았다. 또한 MMLU CoT, MMLU-Pro CoT에서 일반 추론 성능이 크게 무너지지 않는 양상이어서, 단순 성능 저하로 편향이 줄어든 것이 아닐 가능성을 시사한다. 전반적으로 이 연구는 LLM의 편향을 “내부 표상의 인과적 조정” 관점에서 다룰 수 있음을 실증하며, 향후 SFT/RL, unlearning, 프롬프트 시간 제한을 결합한 완화 전략에 방향성을 제공한다.



### The Riddle Riddle: Testing Flexible Reasoning in Large Language Models and Humans (https://arxiv.org/abs/2606.27103)
- **Prior Approaches**: 기존 벤치마크 성능은 LLM의 ‘추론’이 실제 인지적 전략 선택인지, 학습 데이터의 표면 패턴 매칭인지 구분하기 어렵다는 한계가 있었다. 특히 수수께끼나 퍼즐류는 정답뿐 아니라 ‘어떤 경우에 창의적 재해석이 필요한지’를 요구하지만, 인터넷에 해설/정답이 널리 퍼져 있어 모델이 기억 기반으로 답할 가능성이 크다. 시각 분야에서는 ‘illusion illusion’처럼 자극의 겉모양은 유사하지만 실제 조건은 다른 설계로 이러한 혼동을 진단하려는 시도가 있었다.

- **Core Contribution**: 이 논문은 언어 기반 추론에서 같은 혼동을 분리해 보기 위한 riddle riddle 패러다임을 제안한다. 원래 수수께끼(진짜 riddle)는 비유/비문자 해석 같은 ‘트릭’을 찾아야 하지만, riddle riddle은 구조는 유지하면서 트릭만 제거해 문자 그대로의 해석이면 충분하도록 만든다. 즉, 모델이 ‘문제의 요구(내용)’가 아니라 ‘문제의 겉모양(형식)’을 따라 창의적 추론을 과잉 적용하는지 직접 테스트한다.

- **Technical Challenges**: 핵심 기술적 난제는 두 조건이 주제/문장 구조/형식을 최대한 동일하게 유지하되, 필요한 추론의 성격만 교묘히 바꾸는 매칭 설계를 만드는 것이다. 연구진은 30개 riddle 쌍(진짜/변형)을 구성하고, 9개 SOTA LLM에 대해 동일 프롬프트로 대량 반복 응답을 수집한 뒤 LLM-as-judge로 정답과 ‘inventive vs literal’ 추론 유형을 판정했다. 또한 진짜/변형에 대해 엄밀히 고정된 ‘정답 해석’이 아닌 경우가 있어 permissive/strict 코딩 모두에서 결과가 유지되는지 확인해 진단의 견고성을 높였다.

- **Empirical Impact**: 실험 결과 인간과 LLM의 실패 방향이 반대로 나타났다. LLM은 진짜 riddle에서 84.9%로 높지만 riddle riddle에서는 50.7%로 크게 떨어졌고, 오답의 90.8%는 실제로는 literal이면 되는데도 inventive 추론을 잘못 써서 발생했다. 반대로 인간은 riddle riddle에서 80.5%로 더 잘 맞히지만 진짜 riddle에서는 50.5%로 하락했으며, 오답의 57.6%는 필요한 inventive 추론을 literal로 과도하게 처리한 탓이었다. 저자들은 LLM의 진짜 riddle 성능이 유연한 전략 선택이라기보다 기억 검색일 수 있으며, 대비가 없는 자극에서는 ‘그럴듯한 추론처럼 보이는 출력’을 진짜 추론으로 오인하기 쉽다고 경고한다.



### Towards Explainable Adjudicative Variance: Quantifying Judicial Discretion via Gated Multi-Task Learning (https://arxiv.org/abs/2606.27069)
Comments:
          17 pages (8 pages main text), 5 figures, 9 tables. Accepted to the AI for Law Workshop at the 43rd International Conference on Machine Learning (ICML 2026), Seoul, South Korea

- **Prior Approaches**: 기존 Legal Judgment Prediction(LJP)은 대부분 판사 정보를 입력값이나 프롬프트의 정적 특징으로 취급해, 재량(discretion)과 객관적 사실의 분리를 충분히 학습하지 못했습니다. 또한 정확도는 높아도 설명가능성·공정성 측면에서 부족하다는 지적이 누적되어 왔고, 특히 UK Employment Tribunal처럼 기술적 기각과 실체 판단이 섞이는 체계에선 표면 단서에 취약할 수 있습니다. 멀티태스크러닝(MTL)도 주로 judge-agnostic하게 설계돼, 판사별 의사결정 변동을 동적으로 반영하는 데 한계가 있었습니다.

- **Core Contribution**: 이 논문은 재량 기반 실체판단(merit-based)과 절차·기한 등 기술적 처분(non-merit-based)을 분리해 모델링하는 Judge-Aware Gated Multi-Task Learning을 제안합니다. 판사 정체성(judge identity)을 별도 임베딩으로 학습하고, outcome별 게이트가 판사 정보의 의존도를 동적으로 조절하도록 설계해 두 신호가 구조적으로 섞이게 합니다. 아울러 11개의 fine-grained Detailed Case Outcome(DCO) 분류 체계로 인코더에 구조적 정규화를 걸어, 의미 경로를 분리해 학습시키는 점이 핵심입니다.

- **Technical Challenges**: 핵심 난제는 (1) 판사 정체성이 필요할 때와 필요 없을 때를 모델이 스스로 구분하고, (2) fine-grained 구조를 한 채널에서 생성하는 방식이 오히려 신호 결합을 붕괴시키지 않게 만드는 것이었습니다. 이를 위해 LWAN(Label-Wise Attention Network)로 outcome별 의미 부분공간을 만들고, DCO 보조감독과 Multi-Task objective로 공고한 구조를 강제한 뒤, outcome별 gate로 judge embedding을 residual 방식으로 선택·억제합니다. 또 generative fine-tuning(Gemma-4 기반)에서 prompt/output 조합이 sub-additive하게 무너지는 현상을 통제 실험으로 확인하고, differentiable한 구조적 라우팅이 필요한 이유를 실증합니다.

- **Empirical Impact**: UK Employment Tribunal 의사결정 13,937건(2011~2023)에서 제안한 하이브리드 모델이 macro-F1 65.21을 달성하며, 가장 강한 generative SFT 대비 +5.1p 개선을 보였습니다. 특히 드물고 애매한 클래스에서 이득이 집중되었고, 판사 정체성 개입 실험과 게이트 기반 분석을 통해 판사 정보가 실제로 필요한 사건 구간에서만 예측이 흔들리는 양상을 보여 interpretability를 확보했습니다. 또한 구조적 컴포지션이 더 적은 학습 파라미터로 더 좋은 성능-보정(calibration) 절충점을 만든다는 점을 나타내, conditioning 인터페이스 선택이 스케일링보다 지배적임을 시사합니다.



### NuclearQAv2: A Structured Benchmark for Evaluating Domain-Science Competence in Large Language Models (https://arxiv.org/abs/2606.27047)
- **Prior Approaches**: 기존에는 MedQA, LegalBench처럼 특정 전문영역 지식·추론을 겨냥한 벤치마크가 많았지만, 기술 실무에서 요구되는 정량 계산과 개념 이해를 함께 조합해 검증하는 설계는 상대적으로 부족했습니다. 수학·논리 벤치마크(GSM8K, MATH 등)는 도메인 지식보다 추상적 문제해결에 초점이 있어 원자공학 맥락의 신뢰성 평가와는 거리가 있었습니다. 핵공학 분야의 선행 NuclearQA는 규모가 작고, 구축·평가에 전문가 투입이 많이 필요해 확장성과 재현성이 약했습니다.

- **Core Contribution**: 이 논문은 핵공학 지식에 대한 LLM 성능을 체계적으로 평가하기 위한 NuclearQAv2 벤치마크를 제안합니다. 벤치마크는 약 1,240개의 QA를 boolean(사실 검증), numeric(정량 추론·계산), verbal(개념 이해)로 나눠 능력의 서로 다른 축을 분리 측정합니다. 또한 혼합 파이프라인으로 전문가 질문, 기존 데이터, 도메인 기술 코퍼스 기반 LLM 생성 QA를 결합해 확장성을 확보했습니다.

- **Technical Challenges**: 기술 도메인에서는 환각뿐 아니라 수식 적용·계산·개념 매칭이 동시에 실패할 수 있어, 단일 정확도 지표로는 신뢰성을 담보하기 어렵습니다. 특히 답 형식이 서로 달라 boolean은 정규화된 exact match로, numeric은 반올림·상수 차이를 허용하는 tolerance-based 정확도로, verbal은 동의어·의역을 고려한 LLM judge 기반 의미 동치 판정으로 평가합니다. QA 생성 단계에서도 비상황의존성, 모호성 최소화, 답의 간결성을 강제하는 structured prompting과 필터링을 적용해 저품질 문항이 섞이지 않도록 했습니다.

- **Empirical Impact**: 여러 LLM을 NuclearQAv2로 평가한 결과, 대체로 boolean은 높게 나오지만 numeric과 verbal에서 성능 격차가 크게 벌어졌습니다. 즉, 사실 인지는 상대적으로 성숙했더라도 핵공학의 정량 추론은 모델 간 변별력이 크고, 개념 이해도 스케일과 모델 특성에 영향을 받는다는 점이 확인됩니다. NuclearQAv2는 멀티-faceted 평가 프레임워크의 필요성을 실증하며, 향후 기술 도메인에서 재사용 가능한 확장형 벤치마크로 자리잡을 가능성을 제시합니다.



### Improving General Role-Playing Agents via Psychology-Grounded Reasoning and Role-Aware Policy Optimization (https://arxiv.org/abs/2606.27025)
- **Prior Approaches**: 일반 목적 role-playing agent(RPA)는 주로 SFT(지도 미세조정)로 캐릭터 데이터에 학습시켜, 페르소나 표현을 따라 하도록 만드는 방식이 주류였습니다. 하지만 이 접근은 표면 패턴 모방에 치우쳐 out-of-distribution 캐릭터에서 generic 응답으로 퇴행하는 한계가 나타납니다. CoT 기반 role-playing도 있지만, 범용적인 CoT는 인간처럼 주관적·고유한 사고로 정착되지 못하는 문제가 제기돼 왔습니다.

- **Core Contribution**: 이 논문은 Psy-CoT를 제안해, 응답 직전의 사전 추론을 Interaction Perception(상황 파악)·Psychological Empathy(심리적 공감)·Logical Construction(논리적 구성) 3단계로 분해함으로써 프로필에 “동적으로” 사고하도록 만듭니다. 또한 단순 구조화된 추론만으로는 부족하다는 점을 짚고, 캐릭터 fidelity를 높이기 위한 RL 정렬이 필요함을 결론짓습니다. 더 나아가 LLM reward model 환경에서 발생하는 reward hacking 문제를 겨냥해 Role-Aware Policy Optimization(RAPO)를 함께 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 LLM 기반 reward model에서 role-agnostic(무관 토큰)과 role-specific(고유 토큰)이 동일한 그라디언트 신호를 받는 현상입니다. 이 때문에 학습이 진행될수록 hacking 성격의 표현이 축적되어, 모델이 “둘 다 최적”이라고 오해하게 됩니다. RAPO는 profile–token mutual information을 토큰별로 계산해 역할이 개입하는 정도를 측정하고, advantage 부호에 따라 role-specific 토큰의 가중치를 비대칭으로 증폭/감쇠해 그라디언트를 역할 지향적으로 재조정합니다.

- **Empirical Impact**: 실험에서는 CoSER, CharacterBench, CharacterEval 3개 벤치마크에서 Psy-CoT가 기존 role-playing CoT 계열을 능가했고, RAPO는 GRPO 대비 일관된 성능 향상을 보였습니다. 특히 Llama-3.3-70B-Instruct에 Psy-CoT를 적용했을 때 CoSER에서 평균 5.2%, CharacterBench에서 4.2% 개선을 기록했으며, Qwen2.5-7B·Qwen3-4B·Llama-3.1-8B에서도 RAPO 학습이 baseline 대비 각각 13.7%, 15.6%, 40.1% 향상으로 나타났습니다. 또한 SFT 계열의 과적합 경향을 넘어, RAPO의 profile-aware 토큰 가중치가 다른 언어/설정으로도 전이되는 cross-lingual generalization 의미를 확인했습니다.



### MinGram: A Minimalist Unigram Tokenizer with High Compression and Competitive Morphological Alignmen (https://arxiv.org/abs/2606.27019)
- **Prior Approaches**: 기존 Unigram tokenizer는 토큰 리스트 표현을 유지해 어휘 편집이 쉬운 장점이 있지만, 학습 과정이 상대적으로 무겁고 복잡하다는 한계가 있다. BPE는 학습·구성이 단순한 편이지만 형태소 정렬(morphological alignment) 관점에서는 손해를 볼 수 있다는 지적이 나온다.

- **Core Contribution**: 이 논문은 Unigram의 토큰 리스트 표현은 그대로 두면서 학습을 단순화한 MinGram(Minimalist Unigram)을 제안한다. MinGram은 BPE-derived seed vocabulary, 최소 토큰 경로에 대한 Hard EM, 그리고 단 한 번의 flat score-pruning으로 학습 절차를 압축한다.

- **Technical Challenges**: 학습 단순화를 해도 압축 성능과 형태 정렬 품질을 동시에 유지해야 하는 점이 핵심 기술 과제다. MinGram은 토큰 개수를 1순위 목표로 두고 Unigram score는 동점 해결(tiebreak)에만 사용하며, 그 결과 suffix array 제거, forward-backward pass 제거, 반복적 prune loop 제거 등 절차를 크게 단순화했다.

- **Empirical Impact**: 6개 언어 실험에서 MinGram은 BPE와 표준 Unigram 모두보다 더 잘 압축한다. 또한 압축 지향 변형은 강력한 token-count 계열 압축기와 비슷한 수준을 내면서도 형태소 정렬을 유의미하게 더 높게 유지했으며, 통제된 downstream 언어모델 학습에서도 Unigram 계열(특히 MinGram)이 BPE보다 bits-per-byte에서 일관되게 우수했다.



### Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs (https://arxiv.org/abs/2606.26987)
- **Prior Approaches**: 기존 연구는 Claude Sonnet 4.5 내부에서 특정 감정을 나타내는 linear ‘emotion vectors’를 찾아, 이를 조작하면 행동이 바뀔 수 있음을 보였습니다. 또한 감정 공간의 기하가 인간의 valence·arousal 심리 구조와 유사한 축 정렬을 보인다고 주장했지만, 다른 모델에서도 같은 성질이 재현되는지와 레이어별로 어떻게 나타나는지는 불명확했습니다.

- **Core Contribution**: 본 논문은 두 open-weight 모델(Apertus-8B-Instruct-2509, Gemma-4-E4B-it)에서 모든 레이어에 걸쳐 emotion contrast vector를 추출해 감정 기하의 일반성을 재검증했습니다. 특히 ‘감정 기하’는 비슷하게 회복되더라도 valence 축이 네트워크 깊이에서 언제·어떻게 형성되는지, 그리고 추출에 쓰는 story corpus가 결과에 어떤 영향을 주는지까지 체계적으로 분리했습니다.

- **Technical Challenges**: 핵심 과제는 (1) 감정 신호와 일반 언어 구조가 섞인 residual stream에서 emotion-specific 성분을 안정적으로 분리하고, (2) 모델·코퍼스 조건을 바꿔도 비교가 가능하도록 레이어별 기하 변화를 일관된 방식으로 측정하는 것입니다. 이를 위해 중립(negative/neutral) 이야기로 confound subspace를 만든 뒤 투영을 제거해 contrast vector를 구성하고, PCA로 PC1/PC2를 human valence·arousal와 상관시키며 CKA/cosine similarity로 레이어 간 표현 변화까지 분석했습니다.

- **Empirical Impact**: 실험 결과 valence의 PC1 정렬은 두 모델 모두에서 강하게 재현되어, Gemma-4-E4B-it은 최대 r=0.83, Apertus-8B-Instruct-2509은 최대 r=0.76을 보였습니다(Claude 결과 r=0.81에 근접). 다만 valence는 Gemma에서 초기에 강하게 인코딩되었다가 중후반에 붕괴되는 반면, Apertus는 반대로 중간~후반에서 점진적으로 나타나는 ‘서로 다른 발현 경로’를 보였고, arousal은 story corpus에 민감해 Gemma-generated 코퍼스에서 정렬이 크게 향상(r≈0.41~0.45)됐습니다. 저자들은 재현 가능 연구를 위해 코드와 데이터셋을 공개해, 모델 해석·안전 모니터링에서 레이어 선택과 추출 프로토콜의 중요성을 부각했습니다.



### ReaORE: Reasoning-Guided Progressive Open Relation Extraction Empowered by Large Reasoning Models (https://arxiv.org/abs/2606.26986)
- **Prior Approaches**: Open Relation Extraction(OpenRE)은 학습 때 보지 못한 relation type을 텍스트에서 찾아내야 해서 “unseen relation 일반화”가 핵심 난제다. 기존 방식은 (1) 임베딩을 클러스터링해 후보를 만들지만 클러스터에 의미 라벨을 붙이는 과정이 필요하고 일반화가 약하며, (2) LLM이 relation label을 바로 생성하지만 서로 헷갈리기 쉬운 relation을 구분하는 판별력이 부족한 한계가 있다.

- **Core Contribution**: 이 논문은 Reasoning-guided progressive OpenRE(ReaORE)라는 coarse-to-fine(거친-정밀) 추론 기반 프레임워크를 제안한다. ReaORE는 1단계 relation filtering으로 후보 relation 집합을 만들고(다중 관점 matching + embedding 기반 보강/필터링), 2단계 relation prediction에서 후보들 사이의 fine-grained comparative reasoning(비교 추론)으로 최종 답을 선택한다.

- **Technical Challenges**: 핵심 기술적 어려움은 unseen relation에 대해 신뢰할 수 있는 후보 커버리지를 확보하면서, 서로 혼동되는 relation을 증거 기반으로 구별하는 것이다. ReaORE는 매칭 기반 reasoning에서 semantic/헤드-꼬리 엔터티 타입 일치 여부를 Boolean 판정과 rationale로 누적해 score tier를 만들고, embedding similarity로 누락된 후보를 보강한 뒤, pairwise comparison에서 evidence verification·semantic granularity·contextual alignment 기준으로 “왜 이 relation이 더 맞는지”를 명시적으로 비교·판단한다.

- **Empirical Impact**: FewRel과 TACRED의 OpenRE 벤치마크 실험에서 ReaORE는 주요 clustering 지표와 classification 지표 모두에서 기존 베이스라인 대비 최상 또는 동급 성능을 보였다. 또한 ablation 결과에서 relation reranking과 relation filtering, 그리고 fine-grained comparative reasoning이 각각 성능을 유의미하게 끌어올리며, structured reasoning 체인이 단순 파이프라인이 아니라 예측의 신뢰도를 높이는 구성 요소임이 확인됐다.



### Auditing Framing-Sensitive Behavioral Instability in Large Language Models for Mental Health Interactions (https://arxiv.org/abs/2606.26982)
- **Prior Approaches**: 기존 연구는 프롬프트 프레이밍, 사회적 맥락, 대화 단서가 aligned LLM의 출력(예: 거절/안전행동/시시응답, sycophancy 등)을 바꿀 수 있음을 보여줬지만, 주로 행동(behavior) 수준에 집중했다. 반면 프레이밍 변화가 내부 표현 내부에서 어떻게 조직되고 해석 성향(해석적 과잉·에스컬레이션)을 어떤 방식으로 반영하는지는 덜 알려져 있었다. 특히 의료·정신건강 대화처럼 비공격적 맥락에서도 일관성 문제가 생길 때를 내부적으로 진단하는 평가 틀이 부족했다.

- **Core Contribution**: 이 논문은 의미 의도는 고정한 채 문서/인식론/기관/책임(리스크)/역할 같은 비대립적(non-adversarial) 프레이밍만 바꾼 matched-prompt 세트를 설계해, 프레이밍이 해석적 반응 성향을 체계적으로 흔드는지 검증한다. 또한 여러 instruction-tuned 모델 계열(Qwen, Gemma, Mistral, Phi 등) 전반에서 프레이밍 민감성이 일관되게 나타남을 보인다. 더 나아가 해당 행동 변화가 단순 표면 표현이 아니라 hidden-state 표현에도 반영되는지, 그리고 일부는 activation steering으로 부분 조절 가능한지까지 연결한다.

- **Technical Challenges**: 핵심 기술적 도전은 ‘프레이밍이 바꾼 것’이 내부 표현의 해석 성향인지, 아니면 표면 어휘(lexical cues) 편향인지 분리하는 것이다. 논문은 레이어별 hidden-state에 대해 logistic regression probing을 수행하고, held-out framing probes로 학습 템플릿을 벗어난 프레이밍에서도 해독 성능이 유지되는지 확인했다(어휘 TF-IDF 베이스라인과 대조). 또한 restrained-supportive vs higher-interpretation routing 차이에서 contrastive latent directions를 만들고 특정 층에서 residual stream을 조절하는 activation steering으로, 행동 변화가 표현 방향과 연결되는지도 실험했다.

- **Empirical Impact**: 실험 결과, 프레이밍은 전 모델 계열에서 해석적 라우팅(에스컬레이션 성향) 비율을 체계적으로 바꿨으며 특히 Documentation 프레이밍이 가장 큰 영향을 보였다. 내부 표현 분석에서는 프레이밍-연관 정보가 transformer 전 깊이에 걸쳐(단, 해독 강도는 아키텍처별로 상이) decodable했지만, TF-IDF 어휘 베이스라인이 일부 성능을 설명해도 held-out probe는 여전히 우연을 넘어 유지되어 ‘순수 어휘’만으로는 설명되지 않음을 시사한다. 나아가 activation steering은 여러 모델에서 해석적 라우팅을 부분적으로 낮출 수 있어, 정신건강 지향 대화형 AI의 일관성·신뢰성 평가에서 ‘프레이밍 robustness’를 중요한 차원으로 다뤄야 함을 실증적으로 뒷받침한다.



### RedVox: Safety and Fairness Gaps in Speech Models Across Languages (https://arxiv.org/abs/2606.26968)
- **Prior Approaches**: 기존 연구는 음성 안전·공정성의 취약점을 다루더라도 영어 중심 평가와 합성 음성에 치우치는 경우가 많았다. 실제 음성 기반 상호작용의 자연스러운 조건을 반영하는 데이터·벤치마크가 부족해, 언어별 안전·위험 분포의 공백이 남아 있었다.
또한 모델 공개 문서에서도 다국어 분석이 거의 없어(multiple multilingual analysis), 재현 가능한 비교가 어려운 실정이다.

- **Core Contribution**: 이 논문은 음성 모델 출시 전반의 안전·공정성 보고 관행을 조사한 뒤, 다국어 분석을 수행·공개하는 비율이 8%에 그친다는 격차를 제시한다. 이를 메우기 위해 RedVox를 도입하는데, 5개 언어(영어·프랑스어·이탈리아어·스페인어·독일어)에서 실제 음성(자연 음성)을 기반으로 unsafe와 불공정한 고정관념성 요청을 평가한다.
평가 입력 형식을 speech(발화)와 audio(텍스트+잡음/무음)로 분리해, 모달리티가 취약점에 미치는 영향을 체계적으로 관찰하도록 설계했다.

- **Technical Challenges**: 핵심 기술적 난제는 자연스러운 음성 데이터 수집과 안전성 테스트를 함께 만족시키는 것이다. 음성 입력을 실제 사용자 목소리로 만들면 심리·개인정보·사생활 부담이 커져, 공개 데이터 규모가 줄어들 수 있으며 동의 절차도 복잡해진다.
논문은 SHADES·M-ALERT의 다국어 텍스트 기반 항목을 음성 요청으로 재구성하고, speech 입력의 음성 구간 품질 검증(Voice Activity Detection)과 LLM-as-judge 기반 분류(relatedness 포함)를 통해 재현성과 평가 신뢰도를 함께 확보했다.

- **Empirical Impact**: 8개 최신 음성/멀티모달 모델을 RedVox로 실험한 결과, 비대상적(비적대적) 조건에서도 unsafe·fairness 취약점이 지속적으로 관측된다. 또한 영어보다 비영어 언어에서 취약점이 더 악화되며, 요청이 spoken input으로 들어오면 안전 문제가 더 크게 증폭된다.
이외에도 음성 노동 참여자 설문에서, 위해 내용을 발화로 녹음·공개하는 데 대한 불편감과 자신의 음성이 식별되어 해악과 연결될 수 있다는 프라이버시 우려가 두드러져 자연주의 음성 안전 연구의 사회기술적 장벽을 정량화했다.



### Term-Centric Hierarchy Induction from Heterogeneous Corpora (https://arxiv.org/abs/2606.26963)
- **Prior Approaches**: 기존 taxonomy induction은 문서 전체(원문/요약) 임베딩이나 문서 단위 클러스터링에 의존해 소스 간 스타일 차이가 그대로 반영되는 문제가 컸다. 또한 데이터 구성이 만든 밀도/분산에 따라 K-Means처럼 실증적 분할이 편향돼, 영역의 실제 개념 구조보다 ‘자주 등장하는 문서’에 맞춰 계층이 쪼개지거나 뭉개질 수 있었다. SCYCHIC 같은 LLM 결합 방법도 단일 소스 중심이거나, 대규모 멀티 소스에 그대로 확장하기엔 표현·요약·라벨링 비용이 장벽이었다.

- **Core Contribution**: TERMNET은 term-centric 프레임워크로, 문서를 ‘문서 자체’가 아니라 자동 term extraction으로 뽑은 도메인 핵심 개념들로 공통 표현 공간에 매핑해 소스 편향을 줄인다. 이후 상위층은 domain priors(시드 카테고리)로 형태를 잡고, 하위층은 data-driven clustering으로 세부 토픽을 확장해 ‘해석 가능 + 도메인 균형’을 동시에 노린다. 멀티 소스(논문·특허·지원과제) 환경에서 상위 구조의 일관성과 하위 의미 응집을 함께 개선하는 것이 핵심 기여다.

- **Technical Challenges**: 기술 핵심어 기반 표현을 만들더라도, 상위(씨앗) 구조와 하위(클러스터) 발견을 어떻게 결합해 중복 라벨 폭발을 막을지가 어려움이었다. TERMNET은 상위층 생성 시 K-Means 과잉분할(multiplier α) 규모를 조절하고, LLM 분류로 기존 자식에 배치하거나 새 노드를 만들되 smin(최소 상대 크기)과 cosine threshold τ로 새 카테고리 생성 조건을 제한한다. 또한 매우 큰 군집에서는 Mini-Batch K-Means를 쓰고, SCYCHIC의 요약 비용을 줄이기 위해 클러스터당 대표 문서에 한정해 MMR 기반으로 요약을 수행해 end-to-end 확장성을 확보했다.

- **Empirical Impact**: 영어/독일어 멀티 소스 100만 편 이상(논문·특허·지원과제) 벤치마크에서 TERMNET은 cross-source coherence와 계층 품질을 자동·인간 평가 모두에서 기존 텍스트/요약 기반 baseline 및 SCYCHIC 계열보다 높였다. 특히 intruder detection 정확도(94.44)와 source entropy(36.70)가 가장 우수해, 서로 다른 소스가 의미 단위로 잘 섞이면서도 의미적으로 타당한 sibling 구성이 형성됨을 보여준다. 독일 지역 혁신 분석 사례에서는 유사하게 높은 성능과 함께 geographer 관점의 유용성(usefulness)까지 확인되며, 기술 랜드스케이프 매핑 같은 실무적 탐색에 직접 쓰일 수 있음을 시사한다.



### GAVEL: Grounded Caption Error Verification and Localization (https://arxiv.org/abs/2606.26923)
Comments:
          conference

- **Prior Approaches**: 기존 비전-언어 모델(VLM)은 생성 성능이 뛰어나지만, 이미지와 텍스트가 맞지 않는 환각(hallucination)이 잦아 신뢰성과 해석 가능성이 떨어진다. 선행 연구는 환각 탐지나 사실성 평가를 다루었으나, 대개 “틀림/맞음” 같은 거친 판정 위주라 왜 불일치가 생겼는지와 이미지의 어느 근거에서 어긋나는지까지는 제한적으로 설명해왔다.
또한 설명은 하더라도 시각 근거(localization)와의 연결이 약해, 실제 디버깅·수정에 바로 쓰기 어려운 한계가 있었다.

- **Core Contribution**: 이 논문은 이미지-텍스트 불일치를 “검증(verification) + 설명(explanation) + 시각 근거 제시(localization)”를 한 번에 요구하는 벤치마크 GAVEL을 제안한다. 모델이 불일치를 찾아낼 뿐 아니라, 자연어로 왜 틀렸는지 설명하고 해당 오류를 지지하는 이미지 영역까지 바운딩 박스로 지정하도록 설계했다.
이를 체계적으로 평가하기 위해, 다양한 불일치 유형을 포함한 데이터셋과 전용 평가 프로토콜/지표도 함께 제공한다.

- **Technical Challenges**: 핵심 난제는 “환각 이미지-캡션 페어”를 대량으로 신뢰성 있게 수집하고, 각 샘플에 대해 설명과 바운딩 박스 근거를 정밀하게 달아주는 것이다. 논문은 반자동 파이프라인으로 후보 부정쌍을 VLM/텍스트-투-이미지 생성에서 뽑고 GPT-5 mini로 정렬 점수(low-alignment)를 기준 선별한 뒤, 사람이 불일치 원인 설명과 오류를 근거하는 bounding box를 주석한다.
또한 학습을 위해 좌표 예측을 autoregressive token generation으로 통합해, 자연어와 공간 정보를 동일한 생성 흐름에서 학습·평가되도록 구성했다.

- **Empirical Impact**: 실험 결과, 강력한 closed-source 모델들도 GAVEL에서 전반적으로 어렵다고 나타났고, 환각 검증과 정합 설명, 그리고 localization 성능이 전부 동시 개선되기 쉽지 않음을 보여준다. 반면 데이터셋으로 supervised fine-tuning한 기준선은 grounding과 설명 지표 전반에서 일관된 향상을 보이며, 벤치마크 신호가 학습 가능한 supervision임을 확인했다.
즉 GAVEL은 단순 환각 판별을 넘어 “어디가 왜 틀렸는지”를 진단하는 연구·개발용 테스트베드로서 의미가 크다.



### SamaVaani: Auditing and Debiasing Multilingual Clinical ASR for Indian Languages (https://arxiv.org/abs/2606.26901)
- **Prior Approaches**: 기존 ASR 연구는 주로 일반 데이터셋 성능이나 통제된 조건에서의 정확도 비교에 집중해 왔다. 특히 다국어·사투·코드스위칭 환경에서 성능이 흔들리며, 성별·액센트·저자원 언어에서 격차가 발생한다는 문제는 알려져 있었지만, 정신과 실제 인터뷰의 복잡한 발화 특성과 화자 역할 차이를 반영한 평가는 부족했다. 일부 정신과 ASR 연구도 국가·언어가 제한적이어서 인도 임상 맥락의 불확실성을 정면으로 다루지 못했다.

- **Core Contribution**: 이 논문은 인도 임상 정신과 인터뷰(칸나다어·힌디어·인도 영어)를 대상으로 8개 최신 ASR 모델을 체계적으로 감사(audit)하고, WER뿐 아니라 세밀한 오류 패턴 및 공정성(화자 역할·성별 등)에 연결해 분석한다. 나아가 최고 성능의 오픈소스 모델 2종(Gemma3n, OmniLingual)을 fine-tuning하고, 격차를 줄이면서 정확도까지 함께 개선하는 공정성 인지 debiasing 기법 SamaVaani를 제안한다. 핵심 메시지는 “클리닉에 그대로 배치하기엔, 모델-언어-화자 특성에 따라 실패 양상이 구조적으로 달라진다”는 점이다.

- **Technical Challenges**: 직면한 기술 과제는 (1) 저자원 언어(특히 칸나다어)에서의 급격한 WER 악화, (2) 역할·성별에 따른 체계적 오류 불균형, (3) 정신과 인터뷰 특유의 발화(망설임·긴 공백·발화 이상 등)로 인해 음성-문자 정렬과 디코딩이 흔들리는 문제다. 논문은 이를 위해 contrastive learning과 CTC 정렬 head를 함께 쓰는 학습 목표를 설계하고, LoRA로 트랜스포머 내부를 효율적으로 적응시키되 핵심 레이어는 동결하는 방식으로 계산 부담을 줄였다. 또한 pitch augmentation(PitchShift)로 화자/그룹 간에 달라질 수 있는 음향 변이를 잡음으로 취급하도록 유도해 표현의 공정성을 개선한다.

- **Empirical Impact**: 실험 결과 ASR 성능은 모델과 언어에 따라 큰 편차를 보였고, 일부 시스템은 인도 영어에서는 경쟁력 있지만 지역 발화에서는 크게 무너졌다. 특히 칸나다어는 전반적으로 오류율이 높았으며, 공정성 관점에서도 화자 역할과 성별에 따른 격차가 반복적으로 관찰됐다. SamaVaani는 전체 WER을 최대 약 50%가량 줄이면서도, 그룹 간 WER 격차를 완화해 공정성을 함께 개선했으며, CL과 CTC를 결합했을 때가 단일 구성요소보다 성능과 공정성에 더 유리하다는 점도 정량적으로 확인됐다.



### Heterogeneous Neural Predictivity from Language Models During Naturalistic Comprehension (https://arxiv.org/abs/2606.26880)
- **Prior Approaches**: 기존 language-neuroscience 연구는 fMRI·MEG·EEG·iEEG에서 language-model 표현(컨텍스트 임베딩, surprisal, 레이어별 특징)이 신경반응을 예측한다는 점을 보여왔다. 다만 양의 예측이 곧바로 ‘공유된 신경 조직’이나 ‘같은 인지 계산’을 의미하는지에 대해선, 시간적 자기상관·잡음·표현용량·저수준 음향/시계열 요인 및 분석 선택에 따라 결론이 흔들릴 수 있다는 한계가 제기됐다. 또한 학습/추출/대조실험(예: 셔플, 통제 특징) 설계에 따라 predictivity가 과대·과소 추정될 가능성도 지적돼 왔다.

- **Core Contribution**: 이 논문은 Brain Treebank, MEG-MASC, Podcast ECoG의 잠금(locked) 파생 데이터를 사용해 8개 frozen language model의 특징이 자연어/자연발화 이해에서 신경활동을 얼마나 예측하는지 ‘예측 가능성’과 ‘모델 특이적 우위’ 및 ‘해석 수준’을 분리해 평가한다. 특히 nuisance·temporal·representation-capacity 등 matched control을 포함한 controlled predictive-only 기준을 적용해, 단순 양의 예측과 모델측 유용성(advantage)을 구분하는 실증 프레임을 제시한다.

- **Technical Challenges**: 자연 자극에서는 단어 onset, 어휘 빈도, 문장 위치, 음향 엔벨로프, 담화 진행 등이 서로 강하게 상관돼 있어 예측이 언어 요인과 시간/음향 요인을 섞어버릴 위험이 있다. 이를 줄이기 위해 blocked cross-validation으로 시간 누수(leakage)를 억제하고, PCA·잔차화·투영 제거 등 학습 내부 선택은 데이터 누수 방지 규칙 안에서만 정했으며, matched control 계열(차원 매칭 랜덤, circular shift, token-order shuffle, context reset, reversed-context 등)로 모델 우위를 검증했다. 더불어 response-profile 유사도, feature-ablation(double-dissociation) 및 implanted-signal용 캘리브레이션/구현 점검까지 결합해 구성요소 수준 민감도를 확인한다.

- **Empirical Impact**: 세 데이터 전반에서 language-model 기반 특징은 held-out 예측에서 넓게 양의 신호를 보였고(소스 수준에서는 양의 모델 점수 및 저수준 기준선 대비 개선이 다수), matched control을 엄격히 통과한 controlled predictive-only 행은 432개 평가 가능 행 중 67개(Brain Treebank 38, Podcast ECoG 29; MEG-MASC는 없음)로 국소적으로 제한됐다. 또한 모델측 feature ablation은 평가 가능한 다수 소스 행에서 예측 점수를 변화시켰으며, acoustic·timing 및 implanted-signal/뇌 기반 캘리브레이션 대조로 파이프라인이 잡음이 아닌 성분에 민감함을 뒷받침했다. 다만 participant-level의 통제 우위는 균일하지 않고 특정 설정에서만 국소적으로 나타나며, 이로써 ‘예측 유용성’은 확인되되 ‘공유된 신경 조직/계산’에 대한 완전한 통합 해석은 향후 공동 인덱싱 범위 확장이 필요하다는 결론으로 이어진다.



### Information-Aware KV Cache Compression for Long Reasoning (https://arxiv.org/abs/2606.26875)
- **Prior Approaches**: 기존 KV cache 압축은 대체로 최근 관측 창의 attention 가중치로 토큰 중요도를 추정해, 덜 주목받는 과거 토큰을 버리는 방식이 주류였다(SnapKV, PyramidKV, Expected Attention 등). 이런 방법은 문맥과의 연관성은 잘 잡지만, 계산 신호가 기본적으로 backward-looking이라 생성이 진행될수록 변하는 장기 추론 경로의 필요성을 충분히 반영하지 못한다. 그래서 long-decoding에서는 정보 손실이 더 크게 나타나는 한계가 있었다.

- **Core Contribution**: 이 논문은 “압축된 토큰이 미래 추론에 얼마나 영향을 주는가”를 forward-looking 관점에서 재정의하며, 특정 토큰을 KV cache에서 제거했을 때 미래의 예측 분포가 얼마나 달라지는지로 Forward Influence를 제안한다. 분석 결과, attention으로 고른 토큰은 주로 가까운 문맥에 영향이 강하고 빠르게 약해지는 반면, 예측 불확실도(엔트로피)가 높은 토큰은 먼 미래에도 훨씬 더 지속적으로 영향을 준다고 보였다. 이를 바탕으로 정보이론 신호를 결합한 entropy-aware KV cache 압축 프레임워크 InfoKV를 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 장기 유틸리티(미래에 쓸모)를 단순 attention의 국소 연관성에서 벗어나 정량화하고, 압축 결정을 계산적으로 효율적인 점수로 바꾸는 것이다. 논문은 토큰 단위 예측 엔트로피(Top-k restricted entropy로 잡음 완화)와 레이어별 representational evolution(중간 레이어와 최종 레이어 hidden state의 cosine distance)을 함께 점수화해, 예측 불확실성과 의미 미수렴 정보를 동시에 반영하도록 설계했다. 최종적으로 attention score와 entropy 기반 점수를 결합해 레이어별로 상위 토큰만 남기는 방식으로 구현했다.

- **Empirical Impact**: 실험은 Llama-3.1, Llama-3.2, DeepSeek-R1 계열 모델에서 long-context reasoning 벤치마크를 대상으로 long prefilling과 long decoding 모두를 평가했다. LongReason(long prefilling)에서 InfoKV는 다양한 컨텍스트 길이와 캐시 예산(40%, 20%) 조건에서 attention 기반 기존 방법을 일관되게 앞섰고, 이득은 시퀀스가 길어질수록 더 커졌다. 또한 IFEval, AIME 2024, LiveCodeBench(long decoding)에서도 RPC류 attention 기반 압축 대비 성능이 개선되며, 일부 조건에서는 full cache보다도 높은 성적(pass@1)을 보였다. 논문은 attention의 단기 의존 신호만으로는 장기 추론에 필요한 전역 정보를 놓칠 수 있고, 엔트로피 기반 정보 신호가 이를 보완한다는 점을 실증적으로 보여 주며 후속 연구에 방향성을 제시한다.



### Cascaded Multi-Granularity Pruning for On-Device LLM Inference in Industrial Io (https://arxiv.org/abs/2606.26861)
Comments:
          This work has been submitted to the IEEE Internet of Things Journal for possible publication

- **Prior Approaches**: 기존 구조화 pruning은 레이어/어텐션 헤드/피드포워드 채널을 한 번에 중요도로 평가해 제거하는 방식이 많아, 고압축 비율에서는 성능이 급격히 무너지는 문제가 있었다. 또한 압축 결과가 아키텍처마다 예측하기 어려워 cross-architecture에서 효과가 일관되지 않았다.

- **Core Contribution**: 논문은 레이어→어텐션 헤드→피드포워드 채널처럼 coarse-to-fine 순서로 단계적으로 줄이는 cascaded multi-granularity pruning 프레임워크를 제안한다. 각 단계 사이에 lightweight low-rank recovery를 넣어 중요도를 재추정함으로써 한 번의 one-shot 추정이 만드는 오류를 줄인다.

- **Technical Challenges**: 핵심 난제는 “어떤 아키텍처에서 per-component pruning 기준이 믿을 만한가”를 가려내는 것이며, 이를 정보이론 관점에서 정렬 순서의 동기를 제공해 해결한다. 더 나아가 Structural Independence Assumption(SIA)을 검증 가능한 조건으로 정식화해, MHA+GELU는 SIA를 만족하는 반면 GQA+SwiGLU는 위반하여 기준 신뢰도가 무너짐을 예측한다.

- **Empirical Impact**: 베어링 fault diagnosis에서 88M~6.25B 모델에 적용한 결과, MHA+GELU 아키텍처는 최대 13.8배 압축까지도 정확도 83.82%를 유지하며 최고 baseline 대비 +3.70pp 개선을 보였다. 반대로 SIA를 위반하는 GQA+SwiGLU는 약 74pp 정확도 붕괴가 관찰돼 제안한 진단 로직의 실효성이 드러났고, 산업용 slewing bearing 플랫폼에서 NVIDIA DGX Spark로 배치 시 추론 지연을 최대 67.2%, 피크 메모리를 62.5% 줄여 IIoT edge 추론 적용 가능성을 입증했다.



### FBK's Long-form SpeechLLMs for IWSLT 2026 Instruction Following (https://arxiv.org/abs/2606.26819)
- **Prior Approaches**: SpeechLLM은 음성 인코더-LLM 디코더에 modality adapter를 붙여 ASR, ST, SQA, SSUM 같은 작업을 자연어 지시로 처리해 왔다. 다만 IWSLT 2025 이후에도 long-form은 여전히 계산·문맥 길이·환각 위험 때문에 성능과 안정성이 불명확했고, 확장 시 단기 능력이 얼마나 유지되는지도 충분히 검증되지 않았다. 기존 long-form 평가는 단순 점수만으로는 생성 불안정이 성능에 미치는 영향을 제대로 반영하기 어려웠다.

- **Core Contribution**: IWSLT 2026 Instruction Following shared task를 위해 short-form과 long-form 모두를 겨냥한 SpeechLLM을 제안했으며, short-form으로 파인튜닝한 모델을 long-form으로 그대로 확장하는 전략을 중심에 둔다. long-form 생성의 불안정성을 반영하기 위해 hallucination-aware 지표인 HIFS(Hallucination-Penalized Instruction-Following Score)를 새로 도입했다. 또한 long-form에서 핵심 변수인 speech segmentation이 지시 수행과 환각에 미치는 영향을 체계적으로 비교했다.

- **Technical Challenges**: 핵심 난점은 긴 오디오로 인해 LLM 컨텍스트 압박이 커지고, 생성이 불안정해지며 환각이 ASR/SSUM 같은 생성 기반 작업을 크게 왜곡한다는 점이다. 이를 위해 long-form 학습 데이터를 인위적 concatenation으로 구성해 short-form 학습만으로는 생기는 일반화 실패를 줄였고, 고정 윈도우·CRDNN 기반 VAD·hybrid segmentation 등 세 가지 분할 전략을 실험했다. 환각이 반복 삽입 형태로 주로 나타나는 점을 반영해 평가 전 반복을 줄이는 post-editing 정규표현식 기반 처리까지 함께 적용했다.

- **Empirical Impact**: short track에서 단기 성능은 SIFS 2.0708로 경쟁력 있는 결과를 보였고 ASR 정확도(1-WER)와 ST, SQA 지표도 균형 있게 나타났다. long track에서는 HIFS가 가장 높은 설정이 고정 30초 분할로, HIFS 2.0663을 기록하며 전체적으로 가장 견고한 성능-안정성 절충을 확인했다. 분석 결과 환각은 반복 삽입을 중심으로 ASR WER을 크게 악화시키며(예: 삽입 오류 급증), generation-heavy 과제에서 영향이 더 커 long-form 비교의 신뢰성을 위해 HIFS 같은 환각 반영 지표가 중요하다는 결론을 제시한다.



### From Vajrayana Tara to Bengali Baul: A Computational Study of Lexical Transmission Across Buddhist, Shakta, and Vaishnava Traditions in Benga (https://arxiv.org/abs/2606.26803)
Comments:
          9 pages, 2 figures, 4 tables. Code and corpus: this https URL Dataset: this https URL

- **Prior Approaches**: 기존에는 벵골의 불교 바즈라야나 어휘가 팔라(Pala) 수도원 붕괴 이후 샥타 탄트라로 흡수됐다는 주장이 역사적으로 거론됐지만, 실제로는 정량화된 검증이 부족했다. 또한 여러 종교 전통의 어휘 관계를 한 번에 비교하는 대규모 코퍼스 기반 분석이 제한적이었다.

- **Core Contribution**: 이 논문은 8~19세기 벵골·산스크리트 종교 문헌 75편을 대상으로 전통 계층을 넘나드는 어휘 관계를 계산적으로 재구성한다. 특히 바즈라야나-샥타 전이의 핵심 주장(불교 어휘가 샥타로 흡수됨)을 TF-IDF 문자 n-그램과 코사인 유사도로 ‘특이성’ 관점에서 수치화한다.

- **Technical Challenges**: 문제는 샥타 탄트라와 다른 전통 간 어휘 겹침이 산스크리트/시가 일반성 때문인지, 특정 전승 경로의 결과인지 구분해야 한다는 점이다. 연구진은 TF-IDF 문자 n-그램 기반 코사인 유사도로 전통 간 유사도를 비교하고, 같은 세기·같은 언어 조건에서 Gitagovinda(바이슈나바)와 Bridge Tara(불교-샥타 전이)의 유사도 차이(0 vs 0.54)를 ‘전이 사슬의 특이성’ 증거로 제시한다.

- **Empirical Impact**: 결과적으로 샥타와 불교-샥타 전이 사슬이 보여주는 8.5배 대비는 단순한 종교 문헌 일반성보다는 ‘어휘 전승 경로’가 존재함을 뒷받침한다. 또한 Tara 텍스트들에서 샥타→불교 어휘 비율 2.0~4.0, 라므프라사드 센의 18세기 벵골 칼리 노래에서 Tara 56회·Kali 103회 등 정량 근거가 제시돼, 벵골에서의 불교-샥타 신성(신크레티즘)을 다중 전통 데이터로 처음 뒷받침한 연구로 의미가 있다.



### OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning (https://arxiv.org/abs/2606.26790)
- **Prior Approaches**: 언어 에이전트에 outcome-based reinforcement learning(예: GRPO)은 안정적인 on-policy 학습 기반을 제공하지만, 보상은 대개 희소하고 종단형이라 중간 의사결정에 대한 신호가 부족하다. on-policy self-distillation은 토큰 단위로 더 조밀한 지도 신호를 주지만, 기존 skill-conditioned 변형들은 외부 skill library/skill memory 또는 privileged context 검색에 의존하는 경우가 많아 유지·불일치 문제가 생긴다.

- **Core Contribution**: OPID(On-Policy Skill Distillation)는 완료된 on-policy 궤적에서 hindsight skill 감독을 직접 추출해, 외부 스킬 라이브러리 없이도 에이전트의 토큰 단위 self-distillation 신호를 만든다. 각 궤적을 episode-level skill(전역 워크플로/실패 회피 규칙)과 step-level skill(핵심 타임스텝의 국소 의사결정)로 계층화하고, critical-first routing으로 적절한 스킬만 선택해 주입한다.

- **Technical Challenges**: 핵심 과제는 (1) sparse한 종단 보상만으로 어떤 중간 결정이 유효했는지(또는 해로운지)를 토큰 단위로 변환하고, (2) 선택된 스킬이 현재 정책이 유발하는 state 분포와 어긋나지 않게 만드는 것이다. OPID는 LLM 분석기로 궤적을 계층 스킬로 재구성한 뒤, 선택된 스킬을 상호작용 기록에 주입하고 동일 샘플 응답에 대해 old policy가 원래 문맥/스킬 보강 문맥에서 재채점한 log-probability shift를 token-level 자기교사 어드밴티지로 사용하며, 이를 outcome 어드밴티지와 결합해 RL 목표를 유지한다.

- **Empirical Impact**: ALFWorld, WebShop, Search-based QA에서 OPID는 outcome-only RL(GRPO) 대비 대부분의 모델·도메인 조합에서 성능과 샘플 효율을 개선했으며, 특히 작은 백본에서 격차가 더 크게 나타났다. 또한 학습 중 스킬을 파라미터에 내재화해 inference 시 skill 프롬프트/검색 없이도 이득을 유지함을 보였고, 계층 스킬 granularity와 critical-first routing의 기여를 절제 실험으로 확인했다.



### Evaluation Pitfalls and Challenges in Multimedia Event Extraction (https://arxiv.org/abs/2606.26775)
Comments:
          Accepted to ACL 2026

- **Prior Approaches**: 멀티미디어 이벤트 추출(MEE)은 텍스트와 이미지 정보를 결합해 이벤트와 인수를 함께 추출하는 연구로 발전해 왔지만, 평가가 단일 벤치마크(M2E2) 중심이고 프로토콜이 제각각이라는 문제가 남아 있었다. 기존 연구는 데이터 전처리·라벨 매핑·태스크 정의·매칭 규칙을 서로 다르게 두는 경우가 많아 결과 신뢰도와 상호 비교 가능성이 흔들릴 수 있다. 또한 느슨한 평가 기준이 섬세한 구조 제약(트리거 오프셋, 코어퍼런스 연결)을 놓치며 성능을 과대평가할 여지도 지적됐다.

- **Core Contribution**: 이 논문은 M2E2 평가에서 발생하는 ‘평가 함정’을 체계적으로 분석해, 원인이 되는 3대 범주(불일치한 데이터 처리, 불일치한 태스크 가정, 과도하게 완화된 평가 설정)를 정리한다. 그리고 이러한 함정들이 실제로는 텍스트-이미지 간 이벤트 접지(grounding) 능력을 과장할 수 있음을 통제 실험으로 보여준다. 이를 해결하기 위해 StrictEval이라는 더 엄격하고 재현 가능한 평가 프레임워크를 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 다양한 전처리/후처리 오라클(예: 트리거 리파인·동사 리파인), 테스트 부분집합 선택, 그리고 EAE/ED의 구조적 매칭 제약(트리거 오프셋, 일대일 매칭, 멀티모달 첨부)을 평가에서 일관되게 보존하는 것이다. 저자들은 StrictEval에서 오라클 기반 후처리나 테스트 누출을 배제하고, 전체 벤치마크를 동일 태스크 정의로 평가하며, 구조 제약이 유지되도록 엄격한 매칭 기준을 강제한다. 특히 이벤트 코어퍼런스 해석 단계에서도 gold 링크에 기대지 않고, 다양한 코어퍼런스 전략을 같은 평가 틀에서 비교해 현실성 차이를 드러낸다.

- **Empirical Impact**: 실험 결과, 사소해 보이는 평가 선택만으로도 F1이 크게 흔들리며(예: 트리거/동사 리파인, 테스트 필터링, 느슨한 매칭), 멀티모달 성능 격차는 최대 수십 F1까지 벌어질 수 있다. 실제 배포 조건에 가까운 설정(전체 평가, gold coreference 미사용)에서는 멀티모달 ED/EAE가 크게 하락했고, 특히 크로스모달 이벤트 코어퍼런스 해결이 가장 큰 병목임이 확인됐다. 저자들은 재현 가능한 통일 평가 프로토콜, 세부 평가 선택의 명시, gold 의존을 줄이는 방향으로 커뮤니티의 평가 기준 전환이 필요하다고 강조한다.



### ConvMemory v3: A Validity Context Layer for Conversational Memory via Target-Conditioned Relation Verification (https://arxiv.org/abs/2606.26753)
Comments:
          22 pages, 3 figures

- **Prior Approaches**: 기존 ConvMemory v1/v2는 대화 맥락에서 관련성 높은 메모리를 찾고 순서를 정렬하는 데 집중했지만, 검색된 메모리가 이후 턴에 의해 업데이트·수정·대체되어 “더 이상 사실이 아닐” 가능성까지는 다루지 못했다. 특히 v1은 high-recall 풀과 경량 reranker로 후보를 넓히고, v2는 recall을 보존하는 protected top-10 재정렬로 ordering을 개선했으나 “유효성(validity)” 판단은 없었다. 그 결과, 토픽은 맞지만 속성 값이 바뀐 메모리가 상위에 남는 문제(관계는 관련되지만 시간에 따른 진실성은 훼손)가 남아 있었다.

- **Core Contribution**: ConvMemory v3는 v1/v2 검색 후 단계에 “validity context layer”를 추가해, 각 후보 메모리에 대해 ‘이 메모리가 목표(target) 명제 기준으로 이후 소스에 의해 덮였는지’를 확인한다. 기본 동작(context mode)은 후보 집합과 rank order를 그대로 두고, 구조화된 유효성 메타데이터만 첨부해 에이전트가 선택적으로 활용할 수 있게 했다. 또한 demote mode는 명시적 opt-in으로 dense current-state 워크로드에서만 재정렬을 수행하도록 설계돼, 기본 검색 품질을 보존하는 방향을 택했다.

- **Technical Challenges**: 핵심 기술 난제는 ‘관련성만으로는 부족’하다는 점으로, 두 메모리 간 관계는 목표 명제가 고정될 때만 일관되게 정의된다. v3는 이를 해결하기 위해 target-conditioned relation verification을 수행하며, (target, source) 쌍에 대해 MiniLM slot head와 DeBERTa-v3 slot head의 점수를 곱(product)해 보수적으로(dual-evidence gate) ‘직접·특정 overturn’ 가능성을 평가한다. 여기에 source가 업데이트 이벤트인지와 operation이 overturn인지에 대한 보수적 event/operation evidence를 추가하고, 여러 소스 간 결합은 noisy-or로 전파해 업데이트 증거가 불충분할 때 과도한 demote를 피하도록 했다.

- **Empirical Impact**: 실험에서 dual-evidence gate는 합성 multi-hop validity 벤치마크에서 90.12%±1.73의 정확도를 달성했으며, real-data feedback loop로 Memora role binding으로 레이블 없이 전이되어 group-all-correct 98.8%±0.9을 기록했다. 또한 demote 모드에서 current-active Hit@1은 baseline 45.1%에서 95.7%±1.2로 크게 상승하면서도, superseded되지 않은 메모리는 recall 99.4%로 보호해 “재정렬의 부작용”을 제한했다. 마지막으로 6개의 기계검증 가능한 safety contract로 레이어 동작을 고정했고, multi-hop graph propagation은 메커니즘으로 검증하되 엄격한 prerequisite edge의 완전 자동 구성은 counterfactual 필요 지식 문제로 경계(boundary)를 명확히 했다.



### Beyond Logical Forms: LLM-Extracted Patterns for Fallacy Classification (https://arxiv.org/abs/2606.26698)
- **Prior Approaches**: 기존 연구는 논리 오류를 문장 수준의 특징이나 사전 학습된 분류기에 의존해 분류하는 경우가 많지만, 미묘한 표현에서는 추론 구조가 제대로 반영되지 않는 한계가 있었다. 또 LLM을 활용하더라도 fallacy의 추상적 논리 골격과 맥락의 언어 신호를 함께 통합하는 방식이 일관되게 정리되지 않았다.

- **Core Contribution**: 이 논문은 fallacy의 추상 논리 구조와 맥락 수준의 언어 단서를 함께 결합해 분류 성능을 높이는 프레임워크를 제안한다. LLM을 이용해 오류 예시와 그에 대한 설명으로부터 논리 패턴을 귀납적으로 추출하고, 이를 데이터 기반 “논리 표현”으로 생성해 활용한다.

- **Technical Challenges**: 핵심 난제는 (1) 미묘한 뉘앙스 속에서 구조적 추론 패턴을 추출해야 하고 (2) LLM이 설명 텍스트에서 그 패턴을 재현 가능하게 학습·구성해야 한다는 점이다. 논문은 LLM이 fallacious example과 explanation에서 패턴을 유도적으로 뽑아내도록 설계해, zero-shot 기준선 대비 분류에 직접 도움이 되는 구조 신호를 구성한다.

- **Empirical Impact**: 실험에서는 여러 LLM과 zero-shot 및 one-shot 설정 전반에서 제안 방식이 통계적으로 유의미한 개선을 보였고, 경쟁 접근법도 능가했다. 또한 데이터셋 간 실험으로 일반화가 확인되며, 설명 기반의 데이터-driven 패턴 추출이 논리 표현 생성의 효과적인 방법임을 입증했다.



### SocialPersona: Benchmarking Personalized Profiling and Response with Multimodal Social-Media Contex (https://arxiv.org/abs/2606.26654)
- **Prior Approaches**: 기존 개인화 벤치마크는 주로 대화에서 사용자가 직접 말한 선호를 기억하는지(메모리 관점)나, 정형화된 로그를 기반으로 선호를 추정하는 방식을 테스트했다. 최근에 행동 이력을 길게 보려는 시도가 있었지만, 합성/구조화된 텍스트 로그 의존이 커서 실제 소셜미디어 타임라인에서 나타나는 잡음·비구조·단서 분산의 어려움이 상대적으로 줄어든 편이다. 또한 다수의 MLLM 연구는 소셜미디어를 콘텐츠/감정/분류 과제로 다루며, 사용자 관련 신호는 보통 추천 타깃이나 인구통계 속성처럼 단순화되는 경우가 많았다.

- **Core Contribution**: SocialPersona는 멀티모달 large language model(MLLM)이 소셜미디어 타임라인에서 암묵적 선호를 복원하고, 그 선호를 대화에 반영하는지를 함께 평가하는 벤치마크다. 171명의 일상적(비홍보) 사용자 타임라인으로부터 7개 관심 도메인에 대해 사람 검증 기반의 stable 관심과 recent 관심을 분리해 프로필을 구성한다. 프로필 구성(task)과 그 프로필에 맞춘 개인화 응답 생성(task)을 제공해, 단순 예측이 아니라 “추정된 선호를 실제 상호작용에 쓰는 능력”까지 측정한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트·이미지·타임스탬프처럼 단서가 희소하고 분산된 멀티모달 증거를 장기(horizon)로 집계해 세부 태그를 복원하는 것, (2) stable와 recent를 시간 분포 관점에서 구분하는 것, (3) 생성된 프로필을 정보 병목 없이 대화 개인화에 안정적으로 전이하는 것이다. 논문은 Gemini/GPT 계열 MLLM을 활용해 보수적으로 관측 가능한 신호만 추출·집계한 뒤 LLM 캘리브레이션과 사람 검증으로 프로필을 정교화했으며, 대화 평가는 timeline-conditioned와 profile-conditioned(생성 프로필만 사용) 두 축으로 오차 전이를 추적한다.

- **Empirical Impact**: 실험 결과, 현재 MLLM들은 세부(fine-grained)·최근(recent) 선호에서 성능이 크게 떨어지고, 특히 추정 프로필을 실제 대화에 적용하면 추가로 품질이 악화됐다. 세부 태그를 4~7개에서 1~2개로 “과도하게 범주화(over-generalization)”하는 경향이 전 모델에서 관찰됐고, 텍스트로 분산된 관심 도메인은 자주 누락되는 반면 이미지가 뚜렷한 도메인은 상대적으로 잘 잡았다. 또한 stable와 recent 분리는 여전히 매우 어려운데, 타임스탬프를 넣어도 시간 분포 패턴을 제대로 활용하지 못하는 것으로 나타나 cross-modal long-horizon user modeling의 난제가 재확인됐다. SocialPersona는 이러한 결함을 정량화하고, 텍스트·이미지 단서 결합과 장기 시간 추론을 개선하는 연구 진척을 측정하는 공통 기준으로 의미가 크다.



### CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs (https://arxiv.org/abs/2606.26650)
Comments:
          This work is accepted to ICML 2026 as an oral. The project page: this https URL

- **Prior Approaches**: 기존 1.58-bit(ternary) 양자화는 BitNet 1.58-bit, TriLM, Tequila처럼 QAT(quantization-aware training)에 의존해 ternarization을 시뮬레이션하며 성능 하락을 완화해 왔다. 그러나 이 방식은 수십~수백B 토큰 규모의 대규모 학습과 막대한 계산·시간 비용이 필요해 확장성과 실용성에 제약이 있었다. 반면 PTQ(post-training quantization)는 데이터 비공개 이점이 크지만, 초저비트(ternary)에서는 최적화 수렴이 어려워 성능이 QAT 대비 크게 뒤처졌다.

- **Core Contribution**: CAT-Q는 QAT 없이도 후처리 방식으로 ternary( {1,0,-1} ) 가중치를 구성하는 cost-efficient·accurate PTQ 방법을 제안한다. 핵심은 learnable modulation(LM)으로 사전학습 가중치 분포와 ternarization 임계값의 민감도를 낮춰 분포 불일치를 줄이고, softened ternarization(ST)으로 미분 가능한 전이 함수를 통해 안정적 수렴을 유도하는 것이다. 이를 통해 다양한 아키텍처와 모델 크기에 “바로 적용 가능한” ternary 양자화를 목표로 한다.

- **Technical Challenges**: ternary PTQ에서의 첫 난점은 hard ternarization만으로 생기는 ternary 가중치 분포가 원래 고정밀 가중치 분포와 잘 맞지 않아 정보 손실이 커진다는 점이다. 두 번째 난점은 초저비트의 비미분(discrete) 특성 때문에 최적화가 잘 수렴하지 않는다는 점으로, 기존처럼 캘리브레이션 학습 내내 hard ternarization을 쓰면 이 문제가 두드러진다. CAT-Q는 LM으로 분포 정렬을 돕고, ST는 두 단계(미분 가능한 전이 → 최종 hard ternarization) relay를 통해 전이 과정에서 gradient 기반 학습이 가능하도록 설계했으며, 여기에 sliding-layer 기반 출력 재구성까지 결합해 PTQ 최적화 난이도를 추가로 완화한다.

- **Empirical Impact**: 실험에서 CAT-Q는 1.7B~8B 파라미터 LLM을 512개의 캘리브레이션 샘플(약 1M 토큰)만으로 ternary 모델로 만들면서, 100B 토큰을 학습한 BitNet 1.58-bit 계열(QAT)보다 더 높은 성능을 보였다. 또한 14B~235B급 대형 LLM에도 처음으로 확장성을 보여주며, 8×A100-80GB에서 8~60시간 내 ternary 모델을 구축할 수 있음을 보고했다. 이는 ternary 양자화의 실용성을 가로막던 “데이터·토큰·학습 비용” 장벽을 크게 낮춰, 초저비트 배포를 촉진할 잠재력이 크다는 평가를 받는다.



### Closing the Quality Gap in Low-Resource Text-to-Speech: LoRA Fine-Tuning of VoxCPM2 for Khmer and Korean (https://arxiv.org/abs/2606.26618)
Comments:
          5 pages, 1 figure, 4 tables. IEEE conference format (IEEEtran)

- **Prior Approaches**: 대규모 TTS는 Tacotron 2, FastSpeech 2처럼 음향 특징을 예측하거나, VITS 등 end-to-end 생성 모델로 발전해 잘 학습된 언어에서는 사람에 가까운 품질을 냈다. 그러나 이런 모델은 희소 언어에서 발음, 강세·억양, 합성 티가 커지는 ‘quality gap’이 남아 있다. 기존 처방으로는 전체 파인튜닝이 있지만, 언어마다 거대한 체크포인트와 비용이 들고 기존 능력의 망각 위험도 커서 PEFT가 대안으로 부상했다.

- **Core Contribution**: 이 논문은 VoxCPM2(2.4B, tokenizer-free)에서 Khmer(캄보디아 공식 언어, 띄어쓰기 없는 저자원)와 Korean을 대상으로, 하나의 shared LoRA 어댑터로 두 언어를 동시에 적응하는 방법을 제안한다. 어댑터는 language tag와 함께 MiniCPM-4 백본과 flow-matching diffusion decoder 양쪽에 삽입되며, zero-initialized라서 학습 시작 시점이 곧 zero-shot 기준선이 된다. 또한 캄보디아어(큰 품질 격차)와 한국어(상대적으로 이미 잘 되는 언어)를 함께 사용해 ‘적응 이득’이 전 언어에 공통인지 검증한다.

- **Technical Challenges**: 핵심 기술 과제는 “적은 파라미터(LoRA)만으로 품질 격차를 메울 수 있는가”와 “서로 다른 스크립트의 두 언어를 하나의 어댑터로 공유할 수 있는가”다. 저자들은 attention의 query/key/value/output projection에만 LoRA를 붙이고, rank를 8~128로 스윕하며 학습 안정화를 위해 AdamW, cosine decay, gradient clipping 등을 적용했다. 또 자동 지표(검증 flow-matching loss)와 인간 MOS 사이에 최적 rank가 엇갈리는 현상을 보여, 결국 듣기 기반 선택이 필요함을 실험적으로 드러낸다.

- **Empirical Impact**: Khmer에서는 native-speaker MOS가 3.85에서 4.23으로 상승(최고 rank 64, paired Wilcoxon p<0.001)하며, 파라미터는 전체의 0.19~3.03%만 학습해도 유의미한 개선이 나온다. 반면 Korean에서는 전반 MOS가 유의하게 개선되지 않았고, 높은 rank에서는 오히려 품질이 떨어져(p≈0.02) ‘기본 모델이 이미 잘 아는 언어는 적응 효익이 작다’는 결론을 뒷받침한다. 동시에 validation loss 최저 rank(128)와 MOS 최고 rank(64)가 달라, 자동 loss만으로 최적화를 판단하면 혼동될 수 있으며 저자들은 이 점이 저자원 언어 적응 전략 수립에 중요한 시사점이라고 정리한다.



### Zero-shot Tweet-Level Stance Detection Enhanced by External Knowledge and Reflective Chain-of-Thought Reasoning (https://arxiv.org/abs/2606.26571)
- **Prior Approaches**: 기존 zero-shot stance detection은 외부 지식 주입, 대조학습, LLM 프롬프트 기반 추론 같은 방향으로 발전했지만, 짧은 트윗에서 맥락이 희소하고 타깃이 암묵적으로 표현될 때 표적-텍스트 정렬이 약해지는 문제가 남아 있었다. 또한 ‘neutral’과 ‘irrelevant’를 가르는 데 필요한 unseen target의 관련성 판단을 충분히 학습하지 못해 라벨 구분이 흔들렸다. 일부 방법은 지식 그래프나 sentiment/추가 특성을 쓰지만, 핵심 엔티티가 내재하는 의미 단서를 적극적으로 재구성해 쓰지 못했다.

- **Core Contribution**: 이 논문은 일본어 zero-shot tweet-level stance detection을 위한 최초의 데이터셋인 KIRP-D(4개 클래스·다중 토픽)를 구축해 연구 공백을 메웠다. 동시에 KIRP는 지식 그래프 기반 엔티티 reorganization/augmentation으로 암묵적 타깃을 명시화하고, Reflective Chain-of-Thought 추론으로 암묵적 타깃의 타당성을 검증하며, stance-aware contrastive learning과 3-layer iterative prototype network로 ‘neutral’과 ‘irrelevant’의 변별을 강화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 짧은 텍스트에서 부족한 맥락을 외부 지식으로 보강하면서도 잡음을 만들지 않는 것, (2) 암묵적 타깃을 엔티티 수준에서 정확히 찾아내는 것, (3) unseen target의 관련성을 추정해 라벨 경계를 세우는 것이다. KIRP는 Wikipedia/뉴스 지식 그래프에서 로컬 서브그래프를 뽑고 graph autoencoder로 구조-의미를 함께 학습한 뒤, LLM이 생성한 counterfactual reasoning을 3단계(반사/정정/엔티티 교체)로 보강해 텍스트에 증거로 덧붙이는 형태로 학습 신호를 정교화했다.

- **Empirical Impact**: SemEval-2016, WT-WT, KIRP-D에서 KIRP는 state-of-the-art 성능을 보였고, F1 기준으로 3-class SemEval-2016 84.05%, WT-WT 84.99%, KIRP-D 4-class 79.18%를 기록했다. 특히 재구성 기반 irrelevant 생성 방식이 자연 샘플과 높은 분포 중첩을 보이며(여러 분석에서 유의미한 차이를 보이지 않음), 모델이 단순한 데이터 인공 편향에 의존하지 않고 타깃-관련성 구분을 학습했음을 시사한다.



### Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention (https://arxiv.org/abs/2606.26560)
- **Prior Approaches**: 선행의 linear attention 기반 recurrent memory는 softmax 대체로 캐시를 고정 크기 상태로 압축해 O(1) 추론을 노린다. 다만 channel-wise gated delta 계열(DPLR/GDN/GDN-2)은 보정(write)과 감쇠/지우기(erase)가 같은 주소(현재 write key 방향)에 묶여 있어, 다른 주소에 남아 있는 오래된 정보를 “쓰기 전에” 표적 삭제하기가 어렵다. 결과적으로 stale memory는 채널 감쇠처럼 주소 비의존적으로만 줄이거나, 이후 쓰기가 그 주소를 다시 방문할 때까지 방치되는 한계가 있었다.

- **Core Contribution**: EDA(Erase-then-Delta Attention)는 재귀 메모리 업데이트 규칙에서 “지울 주소”와 “쓸 주소”를 분리한다. 먼저 독립적으로 학습된 erase address 방향으로 메모리를 타깃 삭제한 뒤, 기존 delta-rule이 하던 방식대로 현재 write key에서 corrective write를 수행한다. 즉 모델이 무엇을 쓸지뿐 아니라, 어떤 오래된 연관을 어느 주소에서 지울지를 동시에 결정하도록 확장한다.

- **Technical Challenges**: 주요 기술 난제는 erase와 delta correction이 같은 수식 구조(랭크-1 연산의 결합)를 공유할 때, 순서(erase first, then delta)가 새로 쓴 내용에 대한 지우기로 “누수(leakage)”하지 않게 안정성을 확보하는 것이다. 논문은 erase를 먼저 적용함으로써 cleanup이 오래된 내용을 대상으로 작동하게 하고, erase 방향 e와 write 방향 k가 (거의) 직교로 학습될수록 교차항이 작아져 update가 안정적으로 두 연산의 분리 효과를 갖도록 유도한다.

- **Empirical Impact**: 언어모델 pretraining에서 dense 2.5B와 MoE 25B-A2.8B 모두 EDA가 가장 높은 성능을 보였고, MoE는 80B-token long-context midtraining 이후에도 4k~128k 컨텍스트 평가에서 지속적으로 최상위를 기록했다. 분석(컴팩트 업데이트 분석, memory-state probes)은 delta-rule의 corrective write 자체는 유지하면서, 특히 passive decay가 약할 때 추가 cleanup 경로가 더 강하게 활성화됨을 시사한다. 이는 recurrent memory가 “쓰는 것”뿐 아니라 “지울 stale 정보를 어디서 제거할지”까지 학습해야 성능이 더 잘 열린다는 메시지를 준다.



### \textsc{DiARC}: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models (https://arxiv.org/abs/2606.26530)
- **Prior Approaches**: ARC는 소수의 입출력 grid 예시로 잠재 변환 규칙을 추론한 뒤 새로운 입력에 적용해야 하는 추상 규칙 유도 문제다. 기존 LLM 접근은 ARC를 텍스트 추론으로 바꾸거나(표현/프롬프트 재설계) RE-ARC·NVARC처럼 데이터 증강·합성으로 supervision을 늘리는 방향이 중심이었지만, 주로 정답(positive)만 학습해 “그럴듯하지만 틀린” 오류를 구분하는 신호가 약했다. 또한 오픈소스 기반은 성능이 제한되고, 폐쇄형은 비용 부담이 커 실용성 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 ARC-like 문제에서 단순 정답 학습을 넘어, 모델이 negative(거절해야 할 near-miss 대안)를 구별하는 능력 자체가 필요하다고 주장한다. 이를 위해 preference alignment 관점에서 preference pair (정답 출력 y+, 거절 출력 y−)를 만들고, DPO 방식으로 y+를 더 선호하도록 학습하는 DiARC를 제안한다. 핵심은 관측된 시연(support demonstrations)은 그대로 두되, query에 대해 규칙은 어긋나지만 시각·구조·로직 측면에서 가까운 negative를 체계적으로 생성한다는 점이다.

- **Technical Challenges**: 가장 큰 난관은 “오류를 informative하게” 만드는 negative 설계다. DiARC는 negative를 세 층위로 구성하는데, (1) 출력 grid의 형태·지오메트리·국소 잡음 등을 바꾸는 output-level visual transformations, (2) task-specific DSL(도메인 특화 언어)에서 반복 규칙 패턴을 찾아 의미를 반전하는 DSL-level rule inversion, (3) 각 태스크 변환 규칙을 LLM이 편집해 의미적으로 정반대에 가깝게 만드는 task-specific rule editing을 사용한다. 또한 생성/분별 능력을 함께 끌어올리기 위해, DPO로 pairwise 상대 선호를 직접 최적화하며(기존 reward 모델 불필요) SFT에서 이어받은 모델을 기준(reference)으로 둔다.

- **Empirical Impact**: 6개 ARC 및 ARC-like 벤치마크(ARC-AGI-1/2, MiniARC, ConceptARC, 1D-ARC, ARCcommunity)에서 DiARC는 3종 오픈소스 백본(Llama-3.2-3B, Mistral 계열, Qwen3-4B)에 대해 SFT baseline 대비 평균적으로 일관된 성능 향상을 보였다. Qwen3-4B 기준으로 ARC-AGI-1·MiniARC·ConceptARC에서 96%+ 정확도를 달성해, 기존 ARC-specialized 모델 및 강한 closed-source LLM을 앞섰다고 보고한다. 더불어 이득이 단순 후보 생성 증가뿐 아니라, 이미 생성된 후보 중 정답을 더 잘 선택하는 discrimination gain에서도 나타나며, 다양한 test-time scaling(TTS) 기법과의 결합 호환성도 확인했다.



### The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Repor (https://arxiv.org/abs/2606.26529)
Comments:
          20 pages, 8 figures. Reproducibility deposit: this https URL

- **Prior Approaches**: 기존 AI safety 평가는 모델이 ‘지정된’ 위험 신호를 얼마나 잘 탐지·보고하는지에 초점이 맞춰져 있다. 하지만 실제 사고는 종종 평가에 포함되지 않은 위험(미지정 co-present 신호)에서 발생하며, 이 불일치를 설명하는 정밀한 내재 메커니즘은 부족했다.

- **Core Contribution**: 이 논문은 task-conditioning(좁은 과제 지시)이 모델이 원래는 보고할 수 있었던 co-present safety-critical 신호의 보고를 억제하는 현상을 ‘Inattentional Gap’으로 명명한다. 같은 모델이라도 입력은 동일한데, 지시가 좁아지면 벤치마크의 기준선 수준 성능이 유지되면서도 실제 위해를 만드는 신호는 침묵할 수 있다고 주장한다.

- **Technical Challenges**: 핵심 과제는 ‘누락이 단순 능력 부족인지, 과제 지시로 인한 보고 억제인지’를 within-item으로 분리하는 것이다. 이를 위해 동일 입력에 대해 (1) open 조건(보고 가능성과 함께 무엇을 볼지)과 (2) task 조건(보고 범위를 좁혀 scoping)을 비교하고, 판정은 키워드 매칭이 아닌 두 명의 언어모델 judge 합의로 수행해 task-induced omission을 입증했다.

- **Empirical Impact**: 실험 결과, radiology·자율주행 텍스트 시나리오와 흉부 X-ray 비전 과제에서 억제는 테스트한 모든 모델에서 관찰됐고, 모델 스케일이 커져도 완화되지 않았다. 또한 이유 모델에서도 지속되며, gap의 상관은 모델 크기보다 model family(제공자/계열)에 더 크게 나타났고, task-free(open)에서는 동일 신호를 훨씬 높은 비율로 보고해 측정된 벤치마크 안전이 현실 안전을 과대평가할 수 있음을 시사한다.



### Assessing Post-Reform Changes in Risk Disclosure Quality with a Multidimensional Text Analysis Approach (https://arxiv.org/abs/2606.26522)
Comments:
          The 4th International Conference on Computational and Data Sciences in Economics and Finance (CDEF 2026)

- **Prior Approaches**: 기존 정량 연구는 장문 공시의 질을 주로 단일 지표(예: 가독성, 구체성, 톤)로 따로 측정하거나 평균 변화에 집중하는 경향이 강했다. 그 결과 지표 간 상충(예: 길이 증대 vs 가독성 저하)이나 기업별 분포의 이질성은 상대적으로 가려졌다. 또한 공시 섹션 내부 특성에 치우쳐 ‘리스크-전략’ 간 토픽 정합성 같은 섹션 간 정렬 문제는 충분히 다루지 못했다.

- **Core Contribution**: 이 논문은 일본어 공시를 대상으로 여섯 가지 텍스트 지표(용량, 구체성, 가독성, boilerplate, stickiness, 교차단면 relevance)를 함께 추출하고, 섹션 간 ‘리스크-경영전략’ 정합성을 별도 지표로 측정한다. 일본 2019 공시 개혁의 효과를 단순 평균이 아닌 다차원 동학(무엇이 늘고 무엇이 나빠졌는지)을 장기 패널로 추적하는 통합 프레임을 제안한다. 특히 paired testing, shift function analysis, 지표 간 상관까지 한 흐름으로 결합한다.

- **Technical Challenges**: 기여를 가능하게 한 핵심 난제는 (1) 일본어처럼 단어 경계가 불명확한 언어에서 ‘용량’과 ‘질’의 지표를 안정적으로 설계하는 것, (2) 여러 지표가 동시에 움직일 때 평균만으론 숨겨지는 분포 변화와 상충관계를 분리해내는 것, (3) 리스크 섹션과 전략 섹션의 토픽 정렬을 계량화하는 것이다. 논문은 GiNZA 기반 NER과 일본어 맞춤 가독성 공식, 8-gram/30% 방식의 boilerplate 탐지, 문장 단위 편집거리-Hungarian 정렬을 통해 stickiness를 계산한다. 통계 모듈로는 기업 고정효과 성격의 paired tt-test로 평균 변화를 확인하고, shift function으로 분위별 변화를 시각화하며, 마지막으로 지표 변화율 간 상관으로 ‘트레이드오프’를 드러냈다.

- **Empirical Impact**: 일본 2019 개혁을 전후로 FY2015-FY2024(19,770 firm-year, 1,977개사)에서 분석한 결과, 공시 용량은 크게 증가했지만 가독성은 전반적으로 하락해 ‘volume–readability trade-off’가 확인됐다. 반면 전체 정보 구조는 개선(특히 stickiness 하락, relevance 상승)됐으나, 구체성 및 기술의 미시 품질은 정체/악화되는 ‘structural–descriptive asymmetry’가 나타났다. 또한 규제의 동일 적용에도 시장 세그먼트별로 반응이 달라 Prime은 변화가 뚜렷한 반면 Growth는 거의 유의미한 변화가 없었다. 전통적인 단일 지표 분석이 놓치기 쉬운 공시 실무의 긴장 관계를 장기·분포 수준에서 보여줬다는 점에서 자본시장/규제 평가 연구에 의미 있는 실증 근거를 제공한다.



### Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledg (https://arxiv.org/abs/2606.26511)
Comments:
          21 pages, 5 tables. Code, prompts, and evaluation datasets included

- **Prior Approaches**: RAG는 대화/상호작용에서 지식을 청크로 저장한 뒤 유사도 기반 top-k를 불러와 정확한 생성(회상)은 잘합니다. 하지만 지식이 변하는 ‘시간’ 문제에는 취약해, 바뀐 사실(현재/과거)이 임베딩 공간에서 거의 구분되지 않아 모델이 오래된 값을 답하거나 아예 포기하는 상황이 생깁니다. 검증/재랭킹을 붙여도 시간적 화폐성(currency) 신호가 없어 근본 해결이 어렵다는 한계가 드러납니다.

- **Core Contribution**: 이 논문은 유사도 기반으로는 ‘중복(재진술)’과 ‘모순(대체)’을 구조적으로 분리할 수 없다는 점을 데이터로 보이며, 그래서 시간 유효성은 유사도 임계값/LLM 판단이 아닌 결정론적 규칙으로 유지돼야 한다고 제안합니다. 이를 위해 MemStrata는 bi-temporal ledger에 facts를 저장하되, (subject, relation, object)로 정의되는 supersession rule로 모순이 감지되면 현재가 늙은 값을 미리 퇴장(retire)시킵니다. 읽기(read) 경로에서는 유사도 임계값도 LLM 호출도 없이, 현재 유효한 사실만 컨텍스트로 구성되도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘임베딩 유사도’로는 모순과 중복을 분리할 수 없다는 구조적 병목을, 시간 모델링 관점의 결정론으로 바꾸는 것입니다. MemStrata는 정확히 (subject, relation, object) 삼중항으로 추출되는 경우에는 중복은 강화(reinforce), 다른 object가 들어오면 기존 validity interval을 닫고 새 값을 연 다음 superseded_by로 연결합니다. 삼중항 외 표현은 별도 gate로 처리하되, 평가 오염을 피하기 위해 marker-free 환경에서만 시간 신호를 갖도록 하고, read 경로는 LLM 불필요로 두어 지연을 임베딩 수준으로 고정합니다.

- **Empirical Impact**: 실험 결과 MemStrata는 정적 지식 벤치마크에서는 RAG와 성능을 맞추는 대신, 진화하는 지식(코드 변경/설정 마이그레이션/의존성 업그레이드/API 구조 변화)에서는 정확도를 0.95~1.00까지 끌어올리며 RAG의 0.20~0.47을 크게 상회합니다. 특히 ‘stale-fact-error rate’에서 RAG가 강제로 답할 때 15~40%로 오래된 값을 제공하는 실패를 MemStrata는 ~0%로 제거합니다. 또한 읽기 지연은 약 2.1s로, LLM reranking/verification 기반 기준(16~18s 내외) 대비 빠르며, marker-free 평가 프로토콜과 함께 재현용 하네스/데이터셋을 공개합니다.



### Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Contex (https://arxiv.org/abs/2606.26493)
Comments:
          Code and model weights available at this https URL

- **Prior Approaches**: 기존 discrete diffusion language models는 한 디코더/네트워크가 문맥 표현(클린 토큰)과 반복 노이즈 제거(denoising)라는 서로 다른 역할을 동시에 수행합니다. 이로 인해 동일한 가중치가 상반된 목표에 끌려가 두 역할 모두에서 최적 역량을 내기 어렵다는 문제가 제기됐습니다. 또한 엔코더-디코더 분리 시도(예: block 단위 denoising)는 소규모 실험 중심이어서, 최근 hybrid Mamba-Transformer+MoE 같은 큰 모델에서도 완전 분리가 성립하는지 불명확했습니다.

- **Core Contribution**: TwoTower는 AR context tower(고정, causal)와 diffusion denoiser tower(학습, trainable)를 블록 단위로 완전히 분리해 역할 얽힘을 줄입니다. context tower는 클린 토큰을 통해 계층별 KV 및 상태를 만들고, denoiser는 bidirectional block attention과 context cross-attention으로 노이즈 블록을 정제한 뒤 블록을 커밋합니다. 더 나아가 동일 프리트레인드 백본에서 나온 “layer-aligned” 정렬을 그대로 활용해 문맥을 멀티스케일로 주입하는 구조를 제안합니다.

- **Technical Challenges**: 핵심은 큰 pretrained AR 표현을 고정한 채로, 별도 denoiser가 diffusion 반복에서 품질을 유지하도록 안정적으로 학습·샘플링하는 것입니다. 논문은 masked diffusion을 블록 단위(토큰을 [MASK]로 오염)로 모델링하고, denoiser가 timestep을 adaLN time conditioning으로 명시적으로 받아 품질을 끌어올리도록 했습니다. 또한 confidence-unmasking(γ 기준)으로 한 단계에서 커밋할 토큰 수를 적응적으로 조절해, 여러 denoising step을 쓰면서도 wall-clock 처리량 이점을 살렸습니다.

- **Empirical Impact**: Nemotron-3-Nano-30B-A3B(하이브리드 Mamba-Transformer MoE) 기반에서 Nemotron-TwoTower는 AR 베이스라인 품질의 98.7%를 유지하면서 생성 처리량을 2.42배(wall-clock) 높였습니다. ablation에서는 bidirectional attention과 time conditioning이 denoising을 돕지만, Mamba는 causal을 유지하는 것이 더 유리하다고 나타났습니다. 또한 블록 크기와 confidence 임계값이 quality–throughput 곡선을 좌우하며, 출시 모델은 AR 체크포인트에서 denoiser만 약 2.1T 토큰으로 적응해 성능 회복을 달성한 점이 의미 있습니다.



### Comparing BERT Sentence-Pair Classification and Few-Shot LLM Prompting for Detecting Threat and Solution Framing in German Climate News (https://arxiv.org/abs/2606.26489)
Comments:
          15 pages

- **Prior Approaches**: 기후 커뮤니케이션에서 미디어 프레이밍(위협/해결)은 청중 반응과 정책 지지에 영향을 주지만, 이를 문장 단위로 자동 탐지하려면 기존엔 수작업 코딩이 사실상 병목이었습니다. 자동화 접근으로는 (1) BERT 같은 인코더를 fine-tuning해 분류하는 방법과 (2) LLM에 few-shot prompting으로 in-context learning을 수행하는 방법이 주로 쓰입니다. 다만 기후 도메인, 특히 독일어 문장 단위 위협/해결 구분에 대해 두 접근을 직접 비교한 연구는 부족했습니다.

- **Core Contribution**: 이 논문은 독일어 기후 뉴스 문장을 threat-oriented와 solution-oriented로 분류할 때, fine-tuned BERT와 few-shot prompting 기반 LLM의 성능을 같은 라벨링 체계에서 비교합니다. 또한 위협/해결을 상호배타 4클래스가 아니라 두 개의 독립 이진 과제로 분해해 “둘 다/둘 다 아님”까지 일관되게 다룹니다. 마지막으로 LLM 프롬프트에 chain-of-thought, speech-act taxonomy, 도메인별 코딩 룰을 체계적으로 포함한 아키텍처를 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 (a) 문장 단위에서 프레이밍 신호를 안정적으로 포착하고 (b) 컨텍스트가 분류에 미치는 영향을 공정하게 다루는 것입니다. BERT는 preceding sentence를 sentence-pair 입력으로 붙여 문맥 신호를 학습하도록 설계했고, LLM은 전체 기사(article text)를 함께 넣되 XML 태그 기반의 구조화된 출력(JSON)과 confidence 스코어를 제공하도록 구성했습니다. 또한 LLM confidence가 F1 향상을 위한 threshold로 잘 보정되지 않아, self-reported certainty를 후처리 개선에 그대로 쓰기 어렵다는 문제도 확인했습니다.

- **Empirical Impact**: 440개 오스트리아 신문(총 10,981문장)에서 평가한 결과, fine-tuned BERT는 threat와 solution 각각 F1=0.83으로 LLM(F1=0.78)보다 약 5포인트 높았습니다. 특히 BERT는 preceding sentence를 제거한 단일문장 입력 실험에서 F1이 0.70대 초반으로 떨어져, 문장 경계를 넘는 맥락 의존성이 크다는 점이 드러났습니다. 반면 LLM은 full-article 컨텍스트가 오히려 주변의 해결 표현이 인접 중립 문장을 solution으로 “번짐(context bleeding)”시키는 경향이 나타나, 최종적으로 supervised 모델이 애매한 사례에서 더 강하게 작동함을 시사합니다.



### Speaking Numbers to LLMs: Multi-Wavelet Number Embeddings for Time Series Forecasting (https://arxiv.org/abs/2606.26487)
Comments:
          Camera Ready version of IJCAI 2026

- **Prior Approaches**: 기존 연구는 LLM을 시계열에 쓰기 위해 (1) 시계열 전용 파운데이션 모델, (2) LLM+외부 예측 도구/에이전트 파이프라인, (3) 입력 적응(패치, 양자화, 심볼/텍스트 변환, 임베딩 정렬)으로 접근해 왔다. 그러나 이러한 방법은 수치 정밀도나 연속성, 국소 변동 및 다중 스케일 구조를 쉽게 훼손해 숫자-언어 간 불일치가 병목으로 남는다는 한계가 있다.

- **Core Contribution**: TempoWave는 LLM 백본은 그대로 두고, 숫자 토큰 임베딩 레이어만 교체하는 plug-and-play 방식의 Multi-Wavelet Number Embedding(TempoWave) 인터페이스를 제안한다. 각 스칼라 관측값을 고정 소수 정밀도의 digit 문자열로 만든 뒤, digit별 multi-wavelet·multi-scale 계수로 구성한 임베딩을 생성해 표준 토큰화가 만드는 크기-순서 불일치를 줄인다. 결과적으로 LLM이 맥락 추론은 활용하되 수치의 정밀한 형식과 숫자 정체성을 유지하도록 설계됐다.

- **Technical Challenges**: 핵심 기술 과제는 연속값을 discrete token 인터페이스에 넣을 때 발생하는 숫자 단절(예: 2026의 자릿수 분해로 인한 크기/서열 붕괴)과, Transformer의 LayerNorm/RMSNorm 같은 정규화가 숫자 구분을 무너뜨리는 문제다. TempoWave는 digits를 impulse로 이산화한 뒤 wavelet 계수를 통해 상대적인 다중 해상도 패턴으로 임베딩을 만들고, 정규화 이후에도 digit codebook의 분리/고유성을 유지하도록 안정성을 분석한다. 또한 예측을 생성형(next-token)으로 수행하되 고정 포맷 파싱/폴백 규칙을 둬 수치 형식 위반을 통제한다.

- **Empirical Impact**: 5개 context-enriched 벤치마크에서 TempoWave는 표준 numeric tokenization과 다른 임베딩 인터페이스 대비 일관되게 개선하며 10개 지표 중 7개에서 새로운 SOTA 또는 top-2 성능을 달성했다. 특히 뉴스·이벤트 기반 데이터(AUL, BIT)에서 MAE/RMSE 개선 폭이 두드러져 비정상성과 급격한 동학에서도 일반화가 강화됨을 시사한다. 또한 진단 분석에서 예측 토큰의 proximity 분포가 매끄럽게(단봉/가우시안 형태) 나타나 수치 연속성과 정렬이 더 잘 학습됨을 보여, 숫자 인터페이스가 성능 병목임을 실증적으로 뒷받침한다.



### Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs (https://arxiv.org/abs/2606.26485)
- **Prior Approaches**: 마이크로블로그 기반 자동 키프레이즈 추출(AKE)은 짧고 잡음이 많은 글이 흩어져 있어 어려운 문제로 다뤄져 왔다. 기존 연구는 독자의 주의가 머무는 단어를 반영한다고 여겨지는 eye-tracking 신호를 활용해 성능을 끌어올리려 했지만, 생리·획득·특징 디코딩 제약으로 한계가 있었다.

- **Core Contribution**: 이 논문은 eye-tracking에 더해 EEG(뇌전도) 신호가 AKE를 보완할 수 있는지 검증한다. ZuCo 인지 언어 처리 코퍼스를 활용해 8개 EEG 특징과 17개 eye-tracking 특징을 모델에 결합하고, 인지 신호가 읽기 중 핵심어 추출에 지속적으로 도움이 되는지를 체계적으로 평가한다.

- **Technical Challenges**: 핵심 기술 과제는 인지 신호가 모델 구조에 의해 왜곡될 수 있다는 점이다. 이를 줄이기 위해 soft-attention 입력과 self-attention의 query vector에 특징을 주입하는 방식으로 넣었고, AKE 모델 전반에서 신호 조합(EEG 단독, eye-tracking 단독, 결합)을 비교해 왜 어떤 조합이 이득인지 확인했다.

- **Empirical Impact**: 실험 결과, 읽기 중 생성된 인지 신호는 특징 조합이나 모델 아키텍처와 무관하게 AKE 성능을 일관되게 향상시켰다. EEG 특징이 가장 큰 개선을 보였고, EEG와 eye-tracking을 함께 쓰면 두 신호 단독 성능 사이에 위치해 부분적 보완성과 동시에 중복/잡음 가능성도 시사된다. 전반적으로 EEG가 마이크로블로그 AKE에 유의미한 인지 근거를 제공하며, 멀티모달 인지 신호 연구가 후속 확장될 만한 가치가 있음을 보여준다.



### Extracting Problem and Method Sentence from Scientific Papers: A Context-enhanced Transformer Using Formulaic Expression Desensitization (https://arxiv.org/abs/2606.26481)
- **Prior Approaches**: 기존 연구는 과학 논문에서 핵심 아이디어를 뽑기 위해 문제·방법 문장을 주로 라벨링 기반으로 학습해 왔다. 그런데 문장 단위 주석은 비용이 커 소규모 데이터셋에 머무르며, 그 결과 모델이 특정 문장 양식에 과도하게 의존해 일반화가 떨어지는 문제가 생긴다.

- **Core Contribution**: 이 논문은 소규모 데이터셋에서 비롯되는 성능 저하 요인을 세 가지 축에서 동시에 다룬다: 데이터 스케일 확대, 특정 양식 의존도 감소, 문장 내 정보 풍부화다. 이를 위해 formulaic expression(FE) desensitization(수식형 표현 비식별화) 기반 데이터 증강기를 제안하고, 문장 맥락을 활용하는 context-enhanced transformer로 핵심 단어 중요도를 추정하며 문맥 잡음을 줄인다.

- **Technical Challenges**: 핵심 기술 과제는 (1) FE 패턴을 인위적으로 변형해도 문제·방법의 의미를 유지하며 합성 데이터를 생성하는 것과 (2) 문장-문맥의 잡음을 줄이면서도 목표 문장에서 중요한 단어를 안정적으로 집계하는 것이다. 논문은 FE desensitization 기반 데이터 augmenters로 합성 샘플을 늘리고, context-enhanced transformer가 맥락을 기반으로 중요도를 측정해 노이즈를 완화하도록 설계했다. 또한 LLM 기반 in-context learning(IBL) 실험도 수행했지만 해당 태스크에는 적합하지 않음을 확인했다.

- **Empirical Impact**: 실험 결과, 제안한 모델은 두 개의 과학 논문 데이터셋에서 baseline 대비 macro F1 점수가 각각 3.71%, 2.67% 향상됐다. 정량·정성 평가 모두에서 문제·방법 문장 추출 품질이 개선되었으며, 특히 소규모 데이터의 양식 의존을 FE desensitization과 맥락 강화로 완화하는 방향이 효과적임을 보여준다.



### Soft Token Alignment for Cross-Lingual Reasoning (https://arxiv.org/abs/2606.26466)
- **Prior Approaches**: 다국어 LLM은 언어가 달라지면 의미적으로 같은 프롬프트에 대해서도 추론과 답이 흔들리는 경향이 있다. 기존 연구는 중간 레이어의 의미 표현은 비교적 language-agnostic한 반면, 생성 시점의 최종 레이어가 discrete 토큰 선택과 함께 언어별 표면형(문자/어휘)로 갈라지며 일관성이 깨진다고 본다. 또한 soft thinking 같은 기법은 디코딩을 연속적으로 완화하지만, 학습 단계에서 언어 간 “같은 추론 경로”를 정렬하는 신호를 직접 주지는 못했다.

- **Core Contribution**: 이 논문은 supervised fine-tuning(SFT) 동안 soft token representation을 언어 간 정렬 신호로 쓰는 보조 목적 SOLAR를 제안한다. SOLAR는 영어를 pivot으로 삼아, 각 비영어 문장의 soft-token 기반 연속 요약 벡터가 영어 요약 벡터와 공유 임베딩 공간에서 가깝도록 cosine 거리 보조 손실을 추가한다. 그 결과, 어휘/스크립트가 달라도 의미적으로 대응되는 reasoning trace가 같은 방향으로 정렬되도록 돕는다.

- **Technical Challenges**: 핵심 난제는 확률 질량이 최종 토큰으로 “경로를 고정”시키면서 언어별 토큰 선택이 추론을 갈라버리는 병목을 학습 신호로 어떻게 완화하느냐다. SOLAR는 top-k filtering과 온도 τ로 next-token 분포를 soft-token(확률 가중 임베딩 혼합)으로 만든 뒤, 토큰 위치를 mean-pooling해 단일 연속 벡터로 요약하고 영어-비영어를 같은 임베딩 공간에서 비교한다. 또한 정렬 가중치 λ와 τ를 조절해 alignment 손실이 cross-entropy를 압도하지 않도록 민감도를 실험으로 맞춘다.

- **Empirical Impact**: 네 가지 다국어 reasoning 벤치마크에서 SOLAR는 베이스 모델 대비 최대 +17.7점, 일반 SFT 대비 최대 +3.8점의 정확도 향상을 보였다. 특히 Swahili 같은 low-resource 언어에서 개선 폭이 가장 컸고, 마지막 레이어의 cross-lingual similarity를 강화하며 언어 클러스터 분리 가능성을 크게 낮췄다. 더불어 행동 분석에서는 영어로의 “언어 붕괴”가 오히려 줄어, 목표 언어 스크립트로 추론을 유지하면서 성능을 끌어올린 점이 주목된다.



### AnySimLite: A Lightweight Few-Shot Similarity Encoder for On-Device Speech-Adjacent Classification (https://arxiv.org/abs/2606.26452)
Comments:
          Accepted at Interspeech 2026

- **Prior Approaches**: 기존 on-device NLP는 각 작업마다 별도 모델을 두는 방식이 많아, 모델 수가 늘수록 저장·메모리 부담이 커진다. 텍스트 유사도는 TF-IDF나 임베딩 기반 접근 등으로 다뤄졌지만, ‘의미론적 유사’만으로는 설명되지 않는 nuanced text similarity(NTS)를 단일 경량 아키텍처로 일반화하기는 어려웠다. 또한 classification을 바로 학습하는 방식은 few-shot에서 데이터 효율이 떨어지거나, hard sample 구성에 도메인 지식이 필요하다는 한계가 있다.

- **Core Contribution**: AnySimLite는 speech-adjacent 분류 작업들을 NTS로 환원해 단일 경량 유사도 인코더로 처리하는 아이디어를 제안한다. word 채널과 character 채널을 함께 써서 out-of-vocabulary(OOV) 단서까지 반영하며, 두 입력의 임베딩 유사도(cosine similarity)로 분류 점수를 만든다. 또한 분류 데이터셋을 NTS용 “라벨 쌍” 데이터로 바꾸는 변환 전략으로, hard한 쌍을 효과적으로 샘플링한다.

- **Technical Challenges**: 핵심 난제는 (1) OOV에 취약해지는 워드 임베딩 단독의 한계를 character 채널로 보완하면서도 경량성을 유지하는 것, (2) 입력을 단순 concatenation해 binary 분류로 처리하면 text similarity의 commutative property를 위배하는 구조적 문제였다. 저자들은 이를 인코더(시암네트워크 형태)로 바꿔 각 타이틀의 임베딩을 별도로 만들고, 단어+문자 채널을 병렬로 설계해 NTS의 성질과 OOV 처리까지 동시에 잡았다. 데이터 변환 단계에서는 랜덤 쌍이 너무 ‘too dissimilar’해 학습 신호가 약해지는 문제를 줄이기 위해, PLM 임베딩→DBSCAN 클러스터링으로 hard 샘플 비중을 조절한다.

- **Empirical Impact**: AnySimLite는 여러 speech-adjacent 분류 태스크에서 few-shot(클래스당 20 exemplars) 조건에서도 SOTA 또는 SOTA-competitive 성능을 일관되게 보였고, 평균 정확도 저하도 기준 대비 2.24%±3.23%에 그쳤다. 메모리 측면에서는 qLLaMA_LoRA-7B(7B) 대비 최악 7% 성능 하락에서도 모델 크기의 <1/250 수준을 사용하며, 8-bit 양자화 기준 디스크 약 700KB와 추론 지연 <30ms로 엣지 배치가 가능함을 보여준다. 이는 multi-model 운용이 잦은 모바일 NLP 파이프라인에서 단일 경량 아키텍처로 작업을 통합할 실질적 가능성을 제시한 사례로 평가된다.



### ProvenAI: Provenance-Native Traces of Evidence in Generated Answers (https://arxiv.org/abs/2606.26449)
- **Prior Approaches**: RAG에서는 생성 답변에 인용(citation)을 붙이지만, 인용이 실제로 답변 생성에 의미 있게 기여했는지까지는 보장되지 않는다는 한계가 지적돼 왔습니다. 기존 연구는 인용의 형식·정확성(예: FActScore, RAGAS)이나 문장/주장 단위의 근거 적합성 점검에 초점을 두거나, 내부 분포 변화를 이용해 attribution을 개선하려 했습니다. 그러나 “인용된 문서를 빼면 출력이 실제로 바뀌는가?”라는 리소스 영향(influence) 검증은 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: ProvenAI는 다중 홉 question answering에서 투명성을 answer correctness, citation fidelity, per-document influence의 세 층으로 분해해 각각을 독립적으로 측정합니다. 특히 인용 감사가 깨끗해 보여도 uncited 동반 문서가 출력에 더 큰 영향을 줄 수 있는 ‘citation-influence gap’을 실증적으로 드러내는 데 기여합니다. 새로운 모델을 학습시키기보다, 고정된 RAG 파이프라인에서 관측 가능한 감사 인프라를 구축해 진단 정보를 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 각 문서가 실제로 출력 분포에 주는 영향을 KL-divergence 같은 분포 수준 지표로 측정해야 하는데, 로컬 backend가 per-token 확률을 제공하지 않는 점입니다. ProvenAI는 leave-one-resource-out ablation으로 문서 제거 전후의 생성 결과를 재생성하고, 답변 문자열 토큰 변화와 citation set 변화에서 나온 표면 프록시를 통해 분포 타깃과의 관계를 정형화합니다. 다만 이 프록시는 근사적으로 신뢰되는 조건이 있으며(결정적 디코딩에서 한쪽(positive) 방향은 보장), 동점 후보들로 확률이 재배치되는 경우에는 영향이 과소평가될 수 있다는 한계도 함께 명시합니다.

- **Empirical Impact**: HotpotQA distractor 검증 split에서 7,405개 예제를 대상으로 평가해 answer accuracy 53.53%와 citation-fidelity 71.55%를 보고하며 두 지표의 디커플링을 확인했습니다. 예시 분석에서는 인용 감사는 완벽하지만 다수의 uncited 문서가 leave-one-out에서 출력에 민감하게 작용해 citation-influence gap을 구체적으로 보여줍니다. 또한 측정 결과가 cryptographic provenance 같은 실행 중 커밋형 감사 아키텍처와 어떻게 조합될 수 있는지까지 논의하며, 검색 기반 QA의 ‘의미 있는 투명성’이 어떤 구성요소로 성립하는지를 제시합니다.



### ConflictScore: Identifying and Measuring How Language Models Handle Conflicting Evidenc (https://arxiv.org/abs/2606.26437)
- **Prior Approaches**: 기존 factuality(사실성)와 faithfulness(근거충실성) 평가지표는 답이 근거 문서에 의해 지지되는지, 혹은 반박되는지 여부를 주로 단일 축에서 점검한다. 그러나 근거 문서 안에 지지와 반박이 동시에 존재하는 ‘혼재(conflicting) 상황’을 제대로 반영하지 못해, 모델이 충돌을 무시한 과신(overconfidence)을 놓칠 수 있다.

- **Core Contribution**: 논문은 근거 문서에 공존하는 상충 정보를 답이 얼마나 잘 인정하는지 정량화하는 지표 ConflictScore를 제안한다. ConflictScore는 응답을 atomic claims(원자적 주장)로 쪼개 각각의 주장에 대해 근거 문서들과의 관계를 라벨링하고, CS-C(충돌이 나타난 주장 비율)와 CS-R(지지 vs 반박의 균형)을 함께 계산한다.

- **Technical Challenges**: ConflictScore를 만들기 위한 핵심 기술 과제는 ‘하나의 주장에 대해 여러 근거 문서가 서로 다른 결론을 낼 수 있는 조건’을 안정적으로 분해·라벨링·집계하는 체계였다. 이를 위해 응답 분해-근거 문서별 라벨링-상호보완적인 집계를 통해 충돌의 빈도와 방향성을 동시에 반영하도록 설계했으며, 이를 검증하기 위한 ConflictBench도 함께 구축한다.

- **Empirical Impact**: 실험에서 ConflictScore는 도메인을 가리지 않고 과신성 있는 오류 주장을 효과적으로 식별함을 보였다. 또한 ConflictScore를 corrective feedback(교정 피드백)으로 활용해 TruthfulQA에서 truthfulness(진실성)를 개선하는 데까지 이어져, 사실성 평가 및 학습 보정 모두에 실용적 의미가 있다.



### ProfileFoundry: A Synthetic Person-Object Substrate for Privacy, Memory, and Tool-Use Evaluation in LLM Agen (https://arxiv.org/abs/2606.26403)
- **Prior Approaches**: 기존 연구는 canary로 암기 노출을 보거나 웹/소셜 추출로 PII나 프라이버시 추정을 다루는 방식이 많았지만, 이는 ‘일관된 사람’ 단위를 만들기 어렵습니다. 또한 PII 스팬 라벨 데이터는 텍스트와 라벨 중심이라 내부 사람 그래프(관계·주소·타임라인 생성 근거)가 재현되지 않는 경우가 흔하고, 장기 메모리/개인화 벤치마크도 고정된 대화·히스토리 제공에 머무는 경향이 있습니다. 즉, 다양한 후속 평가를 같은 스키마로 파생할 수 있는 재사용 가능한 ‘사람 소스 레이어’가 부족했습니다.

- **Core Contribution**: 이 논문은 8개 로케일의 성인 100,000개 synthetic Person Object를 생성·공개하는 PROFILEFOUNDRY를 제안합니다. 각 객체는 스냅샷 필드, 가구/가족/고용주 링크, snapshot-aligned events, 정규화된 관계 뷰, generation provenance까지 포함해 ‘사람 뒤의 연결’을 함께 검사할 수 있게 합니다. 결과적으로 파생용 텍스트·메모리·문서·레코드 연결·교란 세트 등을 만들 때, 동일한 기준 객체에서 재현 가능한 평가 구성이 가능해집니다.

- **Technical Challenges**: 핵심 기술적 난제는 가짜 필드들을 독립적으로 뽑으면 깨지는 관계·시간적 정합성을, 객체 생성 과정에서 먼저 ‘약속(commitment)’으로 고정하는 것입니다. 논문은 household-first 제약된 생성 캐스케이드로 가구 역할과 공유 속성(주소/고용주/가족 엣지)을 먼저 닫은 뒤, 그 스냅샷을 기준으로 사건 이력을 backfill 및 부분 재생(replay)해 referential/temporal closure를 검증합니다. 아울러 deterministic 생성(씨드·매니페스트 해시)과 JSONL/Parquet 정규화 수출, SDK·CLI까지 제공해 동일 스키마 기반 재빌드와 감사를 지원합니다.

- **Empirical Impact**: 저자들은 (1) 인구 분포의 선택적 marginal 비교, (2) 객체 단위 불변성 체크, (3) 릴리즈 전체 referential/temporal closure, (4) coincidence·provenance 스크리닝으로 증거를 범주별로 제시합니다. 전체적으로 구조·링크·외래키·관계 엣지 및 사건의 시간 제약은 강하게 통과하며, 일부 로케일에서 univariate marginal 목표(예: 평균 gap)는 미달하지만 해당 미스 범위와 원인을 함께 보고합니다. PROFILEFOUNDRY는 population-fidelity 모델이나 렌더링 텍스트 코퍼스가 아니라, 메모리·프라이버시 렌더링·문서 이해·레코드 연결·에이전트 상태 평가를 위한 책임 있는 synthetic 소스 레이어로 의미가 있습니다.



### Charting the Growth of Social-Physical HRI (spHRI): A Systematic Review Pipeline Augmented by Small Language Models (https://arxiv.org/abs/2606.26382)
Comments:
          5 pages, 3 figures, 2 tables, Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction

- **Prior Approaches**: 기존 spHRI 관련 설문/리뷰는 HRI·HCI·햅틱스·로보틱스의 관점을 분절된 채로 다루는 경우가 많아, 일관된 어휘·분류 틀이 부족하다는 한계가 있었다. 또한 많은 리뷰가 로봇의 ‘사회적 의도’가 드러나는 물리 상호작용을 포괄적으로 다루지 못해, robject·웨어러블·햅틱 디바이스 같은 범주가 누락되곤 했다. 최근에는 LLM을 체계적 문헌고찰 파이프라인에 활용하려는 시도가 늘었지만, 클라우드 비용·에너지·접근성 문제 때문에 현장 적용이 제한적이었다.

- **Core Contribution**: 이 논문은 로컬에서 구동 가능한 lightweight small language models(SLMs; 1.5B 미만 파라미터)가 spHRI 대규모 systematic review의 title/abstract screening에서 ‘보조적 세컨드 패스’로 얼마나 유효한지 실험한다. SLM을 전문가 대체가 아니라, 사람이 놓친 관련 논문을 회수해 워크플로를 확장하기 위한 안전망으로 위치시킨다. 그 결과, SLM 앙상블이 사람 리뷰어가 놓친 관련 논문을 추가로 찾아내며 실제 성능 향상을 정량화했다.

- **Technical Challenges**: 핵심 기술적 난제는 spHRI처럼 개념적 뉘앙스가 큰 분야에서, 짧은 텍스트(제목/초록)만으로 관련성 판단을 하되 false positive를 억제하는 것이다. 연구진은 Llama3.2, Gemma3, Qwen3, DeepSeek-R1 등 4개 SLM을 로컬 Ollama 환경에서 구동하고, 리뷰 목적·포함/제외 기준을 구조화 프롬프트로 제공한 뒤, unanimity(모두 yes일 때만 include) 규칙으로 보수적으로 플래그를 줄였다. 이후 플래그된 집합만 사람 2인이 재스크리닝하며 reference set을 업데이트해 모델의 분류 정확도를 재평가했다.

- **Empirical Impact**: 실험에서는 spHRI 출판량이 1992~2025년 동안 비선형적으로 지속 성장함을 확인해 분야의 emergent 특성을 지지했다. 개별 SLM은 사람 수준과 완전히 같지는 못했지만, 결합 앙상블은 false positive rate를 크게 낮추면서도 사람 리뷰가 놓친 논문 39편을 추가로 회수해 최종 relevant 데이터셋에서 10.29%를 차지했다. 무엇보다 SLM은 항목당 약 0.21초 수준으로 초고속 스크리닝이 가능해, 전문가의 재심 범위를 통제하면서도 대규모 리뷰를 지속가능하게 만드는 데 의미가 있다.



### Phonetic and semantic analyses of spoken corpora of Beijing and Taiwan Mandarin indicate that the neutral tone is a lexical ton (https://arxiv.org/abs/2606.26360)
- **Prior Approaches**: 만다린의 중성(떠다니는) 성조는 전통적으로 약화된 음절의 결과이거나, 때로는 어휘적으로 고정되지만 무음절처럼도 나타나는 등 ‘복잡한 범주’로 설명돼 왔다. 기존 연구는 중성 성조의 F0 윤곽이 선행 어휘 성조에 의존하며 지역·담화·말하기 스타일에 따라 지속시간과 강도도 달라진다고 보고했다. 한편 대화 음성에서 성조 윤곽은 성조 패턴 외에도 단어 의미에 결부된 word-specific pitch signatures가 존재하며, 이를 GAM으로 분해할 수 있다는 근거가 축적돼 왔다.

- **Core Contribution**: 이 논문은 베이징 만다린과 대만 만다린의 자발 대화 말뭉치에서 ‘첫 음절은 어휘 성조(T1~T4), 둘째 음절은 중성(T5)’ 조합을 집중 분석해, 중성 성조가 독립적인 tonal target(성조 목표)를 갖는지 검증한다. 또한 중성 둘째 음절 단어가 선행 음절의 성조에 따라 달라지는 피치 성분을 가지며, 나아가 단어별 pitch signature도 발현된다는 점을 두 방언에서 확인한다. 결과적으로 중성( floating tone )이 실은 두 방언 모두에서 어휘 성조(lexical tone)에 가깝다는 해석을 제안한다.

- **Technical Challenges**: 중성 성조의 실현은 지역 차이뿐 아니라 지속시간·강도·운율 구조 등 다수 요인에 의해 크게 변동하므로, 성조 패턴 효과와 단어별 효과를 분리하는 정교한 통계 설계가 필요했다. 연구진은 정규화된 시간축에서 log F0를 대상으로 Generalized Additive Mixed Modeling(GAM)에 AR(1) 잔차 구조를 결합해 시계열 자기상관을 제어하고, 성조 패턴·발화 내 위치·화자·빅그램 확률 등 다양한 예측변수를 함께 모델링했다. 더불어 contextualized embeddings로 단어 의미가 pitch signature를 어느 정도 예측하는지까지 비교해, 방언 간 차이가 의미 차이에서 기인할 수 있는지 검증한다.

- **Empirical Impact**: 두 방언의 자발 대화 말뭉치에서 중성 성조는 실제로 ‘성조 목표’가 있는 패턴으로 관측되며, 이 성조 목표는 선행 어휘 성조에 의해 체계적으로 영향을 받는 것으로 보고된다. 또한 동일한 tonal pattern에서도 단어별 pitch signature가 나타나며, 그 일부는 contextualized embeddings로 예측 가능해 단어 의미-발화 음향 공간 간 대응(isomorphy) 가설을 지지한다. 방언별 차이(베이징에서 중성의 빈도·실현 변동이 더 큰 현상 등)는 동일한 성조 패턴의 모양 차이로도 관찰되지만, 일부는 단어가 쓰인 의미 차이로 설명될 수 있음을 시사해 성조 체계의 분류 관점을 재정렬할 수 있는 의미가 있다.



### From Structure to Synergy: A Survey of Vision-Language Perception Paradigm Evolution in Multimodal Large Language Models (https://arxiv.org/abs/2606.26196)
- **Prior Approaches**: 기존 연구는 비전-중심과 언어-중심으로 분절돼, 시각 지각(지역/인스턴스 추출·국소화·관계 추론)이 어떻게 하나의 통합된 능력으로 진화하는지 일관되게 다루지 못했다. 또한 인코더를 고치거나(Region-aware 모듈, ROI 생성, 멀티 인코더/프로젝터) 특정 디코딩 과제에 보조 손실을 거는 방식처럼, 지역 수준 성능 향상에 초점이 맞춰져 정밀한 픽셀 수준 지각으로의 패러다임 전환이 체계적으로 정리되지 않았다.

- **Core Contribution**: 이 논문은 MLLM의 비전-언어 지각을 인간의 선천적 지각 능력처럼 ‘고유하고 통합된 능력’으로 정식화하고, 이를 기준으로 최초의 체계적(5-stage) 설문/분류를 제안한다. 특히 지역(인스턴스) 단위 자연어 질의에 대한 인식 범위를 명확히 하고, 인코더-중심→디코더-중심→동적 지각→아키텍처 프리 전략→통합 프레임워크로 이어지는 진화 경로와 대표 방법을 정리해 로드맵을 제공한다.

- **Technical Challenges**: 통합 지각을 실현하는 핵심 난제는 이미지의 의미 있는 ‘국소 증거’를 선택적으로 찾아 처리하고, 대화/지시 기반으로 해당 영역을 정밀하게 국소화·세분화하며, 그 성능을 입력 상황에 따라 일관되게 유지하는 데 있다. 논문은 이를 해결하기 위한 방향으로 (Stage I) ROI 제안·프롬프트/쿼리 인지·프로젝터/커넥터 최적화와 (Stage II) 보조 디코더(예: segmentation 토큰) 및 멀티-디코더/특화 디코딩 전략으로 지역 수준을 픽셀 수준으로 끌어올리는 설계를 묶어 설명한다.

- **Empirical Impact**: 설문은 여러 시기와 축에서 축적된 대표 기법들을 한 프레임으로 연결해, ‘무엇이 실제로 지각 능력을 강화하는가’를 관찰 가능하게 만든다는 점에서 실무적 가치가 크다. 또한 열려 있는 과제(진짜 general, unified 멀티모달 지능으로의 확장)에 대한 연구 방향을 제시해, 향후 MLLM을 지각-중심 에이전트로 발전시키는 데 참고 가능한 행동 지침을 제공한다.



### Thinking Like a Scientist? A Structural Study of LLM-Generated Research Methods (https://arxiv.org/abs/2606.26130)
Comments:
          46 pages, 13 figures, 18 tables

- **Prior Approaches**: 기존 연구는 LLM이 연구 글쓰기·아이디어 생성·리서치 코드/방법 제안 등 과정을 자동화하는 양상을 다뤘지만, 최소 프롬프트에서 LLM이 ‘어떤 방법을 기본값으로 부각하는지’에 대한 정량 평가는 제한적이었다. 또한 다양성 감소, 벤치마크 포화/동질화 같은 우려는 제기돼 왔으나, 실제 방법론 탐색이 특정 축(모델 제공사, 데이터 태스크, 평가 지표)에서 어떻게 치우치는지까지는 명확히 분해되지 않았다.

- **Core Contribution**: 이 논문은 최근 arXiv 컴퓨터과학 논문 1,000편에서 추출한 연구 질문만으로 GPT-5.1, Gemini 3 Pro, DeepSeek-V3.2가 제안한 데이터셋·모델·평가 지표·짧은 파이프라인을, 논문 기반 ‘방법 인벤토리’와 직접 비교한다. 특히 질문만 주어진 “초기 제안”이므로 최적화 여부와 무관하게, LLM이 기본적으로 만들어내는 방법론 메뉴의 편향과 압축 정도를 정량화했다.

- **Technical Challenges**: 핵심 과제는 LLM 출력과 논문 인벤토리 간 방법 용어가 서로 다른 명명 방식으로 나타날 때 이를 공통 분류 체계(세부 taxonomy)로 매핑하는 것이다. 논문 측·LLM 측 모두에서 구조화된 메서드 특징을 뽑아 공유 분류로 정규화하고, Jensen-Shannon divergence 같은 정보이론 지표와 co-occurrence(동시 등장) 패턴을 다중 차원에서 비교해 제공사 편중이 가장 크게 드러나도록 검증했다.

- **Empirical Impact**: 결과적으로 제공사(provider) 선택의 편차가 다른 어떤 축보다 3–5배 큰 것으로 나타났고, Other/Academic 단발(singleton) 모델군은 23–24%p 정도 과소대표되는 반면 재사용된 학계/커뮤니티 모델은 소폭(4–6pp) 과대대표됐다. 또한 전체 방법론 ‘메뉴’가 강하게 압축되어 유효 모델 엔티티 수가 1,232에서 59–96으로 줄고, 3개 LLM의 순위 상관이 LLM-논문 상관보다 높아 편향이 여러 모델에 공통으로 나타났다. 제안이 입력 문제에 따라 달라지긴 하지만(질문 민감성), 교차검증 없이 LLM 결과에만 의존하면 실험 설계의 초기가능 후보가 더 좁아져 탐색 공간이 상향 동질화될 위험이 커진다는 점을 시사한다.



### Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM (https://arxiv.org/abs/2606.26120)
- **Prior Approaches**: 기존 diffusion Large Language Models(dLLMs) 가속 연구는 이전 단계의 feature를 캐싱하거나 한 번에 여러 토큰을 병렬로 해제(unmask)하는 방식에 의존해 왔습니다. 하지만 대부분의 방법이 레이어와 디코딩 스텝에 걸친 토큰 성질의 변화(동적성)를 정적으로 가정해, 잘못된 캐시 업데이트나 조기 확정으로 성능 저하가 생길 수 있습니다.

- **Core Contribution**: 논문은 학습 없이(training-free) dLLM 추론을 가속하는 Dynamic-dLLM을 제안합니다. 핵심은 레이어별로 필요한 캐시 업데이트 예산을 동적으로 배분하는 Dynamic Cache Updating(DCU)과, 토큰별 예측 안정성을 반영해 마스킹 임계값을 조정하는 Adaptive Parallel Decoding(APD)입니다.

- **Technical Challenges**: 첫째, KV-Cache 호환이 어려운 dLLM의 비(非)자율적 생성 특성 때문에 중복 계산을 줄이려면 캐시 갱신을 “언제, 어디서” 해야 하는지 추정해야 합니다. 논문은 토큰 입력의 변화로 캐시 갱신 필요도를 근사해 DCU를 설계하고, 키 토큰 주변의 필수 업데이트 윈도우로 ‘mud에 빠짐(stuck in the mud)’ 현상을 완화합니다. 둘째, 고정 임계값 기반 병렬 해제는 스텝이 진행되며 confidence 분포가 바뀌어 오차 전파를 유발할 수 있어, APD는 분포 농도와 과거 분포 변화량을 결합해 임계값을 동적으로 보정합니다.

- **Empirical Impact**: LLaDA-8B-Instruct, LLaDA-1.5, Dream-v0-7B-Instruct를 대상으로 MMLU, GSM8K, HumanEval 등에서 평균 3배 이상 속도 향상을 확인했으며, 경우에 따라 최대 4.48×까지 도달했습니다. 특히 GSM8K에서 LLaDA-8B-Instruct는 37.29 TPS로 기준 대비 4.48× 빨라졌고, 정확도는 유지/근접 수준을 보였으며 다른 모델에서도 유사한 성능-효율 균형이 관찰됐습니다. 결과적으로 Dynamic-dLLM은 dLLM 배포 시 성능을 크게 깎지 않으면서 추론 지연을 줄이는 plug-and-play 가속 솔루션으로 제시됩니다.



### From Lexicon to AI: A Structured-Data Pipeline for Specialized Conversational Systems in Low-Resource Languages (https://arxiv.org/abs/2606.26112)
Comments:
          12 pages, 3 figures

- **Prior Approaches**: 기존 저자원 언어용 대화형 AI는 대규모 말뭉치(예: Common Crawl 수준)에 의존하는 미세조정 패러다임이 중심이었고, 교육처럼 도메인·수준에 맞춘 고품질 instruction 데이터가 부족할 때 성능 격차가 커진다. 또, cross-lingual transfer나 prompt 기반 방법은 사전학습 지식에 크게 기대어 환각·사실 불일치 위험이 남아 전문 영역에서 신뢰도가 흔들릴 수 있다. 한편 FAQ/검색형 지식 결합은 생성형 대화의 깊이를 제한해 “가르치는 대화”로 이어지기 어렵다.

- **Core Contribution**: 이 논문은 WordNet 같은 전문가가 큐레이션한 구조화 언어자원을 대화형 학습 데이터로 변환해, 저자원 언어의 특화 conversational AI를 만드는 체계적인 파이프라인을 제안한다. 힌디어 WordNet을 1.25M 규모의 instruction-response로 만들고, 12B 파운데이션 모델을 LoRA와 4-bit quantization으로 resource-efficient하게 fine-tuning해 ‘Shabdabot’을 구성했다. 교육용 언어학적 관계(동의/반의, 상하위, 동음이의 등)를 대화 흐름에 유지하면서 수준(Primary~Expert)별로 설명 깊이와 예시를 조절하는 점이 핵심이다.

- **Technical Challenges**: 문제는 (1) 구조화된 synset/관계 정보를 대화형 데이터로 자동 변환하면서 의미 관계의 연속성을 보장하고, (2) 밀집 관계에서 생기는 정보 과부하를 통제하며, (3) 전문가 지식 기반 생성이 되도록 모델을 효율적으로 특화하는 것이다. 저자들은 관계 타입을 기반으로 Basic/Complex/Ontological/Disambiguation 4종 지시문을 만들고, 10개 초과 관계는 chunking 하되 33% 겹침 슬라이딩 윈도로 카테고리 연속성을 학습하게 했다. 또한 서로 다른 관계를 multi-hop 형태로 묶는 커버리지 체크와 instruction-response 해시 중복 제거로 1,253,847개의 정제 데이터셋을 구성했으며, 4-bit NF4+LoRA(총 파라미터의 0.2%만 업데이트)로 12GB급 배치가 가능하도록 했다.

- **Empirical Impact**: 힌디어 언어학습 챗봇 평가에서 Shabdabot은 교육적 유효성 지표 LAQ 91.0을 기록해 일반 목적 모델 대비 우위를 보였다(예: 79.4 vs. 83.6 범위). 의미 정확성(SAS)은 전반적으로 경쟁력을 유지하면서도, 무엇보다 응답 일관성에서 σ=1.0으로 크게 개선돼 일반 대형모델의 σ=7.4 대비 예측가능성이 크게 향상됐다. 저자들은 의미 유사도만으로 교육 성과를 예측하기 어렵고, 구조화 지식 기반 특화가 “정확성+신뢰도”를 함께 바꾼다는 메시지를 제시하며, WordNet이 존재하는 200+ 언어로 확장 가능한 재현 가능한 개발 방법론임을 보여준다.



### Where Larger Models Excel: The Primacy of Constraint-Guided Reasoning (https://arxiv.org/abs/2606.26108)
Comments:
          10 pages, 3 figures,

- **Prior Approaches**: 대부분의 연구는 정확도 같은 집계 지표로 스케일에 따른 추론 성능 격차를 관찰하지만, 큰 모델이 ‘어디서’ 더 잘 추론하는지 질적으로 설명하기는 어렵다. 또한 작은 모델의 추론 병목을 distillation로 보완하거나(process supervision 등) 전체 벤치마크를 설계하는 흐름이 주를 이뤘지만, 큰-작은 모델 추론 차이를 데이터 기반으로 체계 분해하는 프레임워크는 부족했다.

- **Core Contribution**: 이 논문은 동일 계열에서 큰 모델 vs 작은 모델의 추론 트레이스를 직접 비교해, 큰 모델의 우위가 나타나는 문제를 안정적으로 선별하고(갭 기반 필터링) 그 근거를 자연어 ‘advantage description’으로 추출한 뒤 의미 클러스터링으로 분류한다(AdvCluster). 이를 통해 도메인 공통의 우위 패턴과 특정 도메인에 묶인 전문 우위를 동시에 드러내는 구조화된 택사노미를 제시한다.

- **Technical Challenges**: 주어진 우위를 사전에 고정된 카테고리로 분류하면 LLM judge 중심의 라벨링이 편향·제약될 수 있어, 논문은 데이터에서 자동으로 범주가 ‘유도’되도록 설계했다. 또한 반복 실험에서 큰 모델의 성능 우위가 안정적인 질문만 남기고, 추출된 다수의 advantage description은 중복 제거(코사인 유사도 임계값)와 PCA 차원 축소 후 K-means 후보군을 여러 설정으로 생성한 뒤 정량 지표(DBI·Silhouette)와 reviewer model 평가를 함께 사용해 최종 클러스터 해상도까지 결정한다.

- **Empirical Impact**: 수학·물리·화학·프로그래밍 4개 도메인에서 Qwen3-32B가 Qwen3-8B보다 평균 6.43%, GPT-OSS-120B가 GPT-OSS-20B보다 평균 7.38% 더 좋은 성능 격차를 보이며, 이를 바탕으로 추론 우위 분류를 수행했다. 결과적으로 큰 모델의 핵심 공통 능력은 Constraint-Guided Reasoning로, 명시적·암묵적 제약을 구조화해 탐색 공간을 줄이고 불가능한 경로를 배제하며 중간 단계 검증까지 수행하는 경향이 반복됐다. 반면 작은 모델의 오류는 확률적 실수가 아니라 특정 깊이(Transformation/Process Skills)에서 집중적으로 발생해, ‘제약 기반 추론’의 구조적 결핍이 스케일 격차를 설명한다는 실증 근거를 제공한다.



### Low Resource Multimodal Translation of Nepali Spoken Words into Emotion-Conditioned Sign Language Avatars (https://arxiv.org/abs/2606.26107)
Comments:
          15 pages, 5 figures, 9 tables

- **Prior Approaches**: 기존 수어 아바타 연구는 감정 표현을 더하려는 시도는 있었지만, 대개 범용 감정만 반영하거나 자연스럽고 언어적으로 정확한 표현까지는 한계가 컸습니다. 또한 음성→수어로 넘어가더라도 어휘(lexical) 번역에 초점이 맞춰져 감정 맥락을 함께 처리하지 못하는 경우가 많았습니다. 특히 네팔리처럼 데이터가 부족한 저자원 언어에서는 감정 주석 음성 데이터와 아바타 실증이 거의 없었습니다.

- **Core Contribution**: 이 논문은 네팔리 음성을 입력으로 받아 ‘감정 조건부’ 네팔리 수어(NSL) 아바타를 생성할 수 있음을 보여주는 NEST-V1( Nepali Emotion and Speech Transformer - Version 1) 파일럿을 제안합니다. 음성에서 단어(“thank you”, “hello”, “house”, “me”)와 감정(happy, neutral, sad)을 동시에 추정해, 해당 조합에 맞는 표정·제스처 애니메이션을 출력하는 구조가 핵심입니다. 또한 저자원 환경을 염두에 둔 경량·모듈형 파이프라인으로 확장 가능성도 함께 제시합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 저자원 데이터에서 음성의 단어 인식과 감정 분류를 동시에 안정적으로 학습하고 (2) 이를 엣지에서도 실시간으로 돌릴 만큼 가볍게 만드는 것입니다. 논문은 Mel spectrogram을 2D 이미지처럼 보고 Vision Transformer 계열 백본을 3개 레이어만 사용하며, ASR과 감정 분류에 공유 acoustic encoder를 적용해 파라미터를 절감했습니다. 그 결과 경량 입력(128×128)과 고정 프레임(200)을 통해 학습·추론 일관성을 확보하고, 전처리(리샘플링, 2초 고정, VTLP/반음 반이동)로 데이터 다양성을 보강했습니다.

- **Empirical Impact**: 실험에서 NEST-V1은 600개 레이블 오디오(50명 스피커 규모) 기반으로 ASR 정확도 81.1%, 감정 인식 정확도 79.21%를 보고했습니다. 또한 ASR·감정 분류를 분리한 구조 대비 약 37% 파라미터 효율을 달성하면서 총 22.1M 파라미터로 엣지 배치에 적합한 경량성을 입증했습니다. 현재는 4개 단어와 3개 감정 조합으로 제한되지만, 저자원 환경에서 감정 인지까지 포함한 실시간 음성-수어 인터페이스의 기술 토대를 마련했다는 점에서 의미가 큽니다.



### Reducing Conversational Escalation in Large Language Model Dialogue with Nonviolent Communication Constraints (https://arxiv.org/abs/2606.26106)
- **Prior Approaches**: 기존 LLM 안전 연구는 유해·편향·정책 위반 같은 명시적 출력 차단에 집중해 왔다. 반면, 사용자의 분노·갈등 상황에서 대화 톤과 표현 방식이 갈등을 키울 수 있는 ‘대화적 에스컬레이션’은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 Nonviolent Communication(NVC)의 핵심 원칙을 ‘과정(process) 기반 프롬프트 제약’으로 재구성해, 갈등 상황에서 LLM이 더 완화적(de-escalating) 대화를 하도록 유도한다. 시스템 프롬프트만으로 (1) 책임 전가·판단·의도 추정 회피, (2) 사용자의 감정에 대한 주의 강조, (3) 조언 전 명확화 우선 같은 행동 제약을 구현한다.

- **Technical Challenges**: 주요 기술 과제는 특정 문장 템플릿이 아니라, 모델의 반응 생성 과정에서 에스컬레이트할 위험 행동을 얼마나 일관되게 억제할지였다. 저자들은 다양한 instruction-tuned 모델과 사용자 저항도(낮음/중간/높음)를 갖춘 dual-agent(assistant–User Simulator) closed-loop 시뮬레이션과 LLM 기반 judge로 대화를 장기 추적 평가했다.

- **Empirical Impact**: 실험 결과, NVC 제약 프롬프트는 전반적으로 갈등의 전개(Conflict Trajectory Score)를 악화시키는 경향을 줄이고 상호작용 안정성을 높였다. 특히 사용자 저항도가 높은 조건에서 Vanilla baseline의 음수 경향을 완화하며, DeepSeek-V3와 Claude-4.5-Sonnet 두 judge 모두에서 ‘에스컬레이션 억제’ 패턴이 일관되게 나타났다. 저자들은 이는 감정이 격해진 온라인 논의·고객지원 등에서 LLM 신뢰성을 높일 수 있는 간단한 접근일 수 있음을 시사한다고 정리했다.



### Context Recycling for Long-Horizon LLM Inferenc (https://arxiv.org/abs/2606.26105)
- **Prior Approaches**: 기존 long-context 대응은 주로 RAG처럼 관련 문서를 검색해 프롬프트에 주입하거나, LoRA 같은 fine-tuning으로 지식을 가중치에 흡수하는 방식이었습니다. 다만 RAG는 세션 메모리가 없어 매 턴 독립적으로 재검색해야 하고, 검색 지연·근사 매칭·평평한 구조 문제가 남습니다. 또한 긴 context window를 늘리는 접근은 프리필 지연과 서빙 비용이 커지고, 기본적으로 세션 간 상태가 유지되지 않는다는 한계가 있습니다.

- **Core Contribution**: ContextForge는 LLM의 context window를 대화 누적 버퍼가 아니라 “재사용 가능한 작업공간”으로 재정의하고, 고정 토큰 예산 안에서 턴별로 컨텍스트를 로드/해제하는 context recycling을 제안합니다. 이를 위해 5단 메모리 계층(선택적 LoRA, KV-cache 기반 안정 프리픽스, 인덱싱/라우팅, 영속 저장, 야간 프리컴퓨트)을 구성해 지식 저장소는 크게 두되 활성 컨텍스트는 고정 크기로 관리합니다. 결과적으로 긴 하이즌에서도 토큰 오버헤드 없이 일관된 멀티턴 추론을 노립니다.

- **Technical Challenges**: 핵심 과제는 (1) 대화가 길어져도 context가 무한히 커지지 않게 하면서 (2) 필요한 지식은 정확히 다시 불러오고 (3) 매 턴의 조립 비용을 낮추는 것입니다. ContextForge는 FTS5 BM25 기반 지식 tree에서 요약(summary)으로 분기(branch)를 결정하고, 실제 컨텍스트 주입은 해당 분기 콘텐츠로만 수행해 토큰 사용을 고정 예산에 가깝게 묶었습니다. 또한 KV-cache의 안정 prefix 재사용(시스템 프롬프트 등)을 포함하고, 대화가 길어지면 LLM 기반 context compaction으로 히스토리를 4–8배 압축해 토큰 폭증을 억제합니다.

- **Empirical Impact**: 276M 행의 CMS Medicare 데이터 기반 12턴·15턴 벤치마크에서 ContextForge는 정확도는 유지하거나 약간 개선하면서도 baseline 대비 토큰을 크게 줄이고 속도를 끌어올렸습니다. 12턴에서는 약 4.2배 적은 토큰과 4.7배 빠름을 보였고, 15턴에서는 컨텍스트 길이가 늘며 효율 격차가 더 벌어져 토큰 13.4배, 속도 8.0배 수준까지 확대됐습니다. 즉, 긴 대화에서 성능 하락 없이 비용을 누적 절감하는 “시스템 레벨” 확장 전략으로 의미가 큽니다.



### Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfar (https://arxiv.org/abs/2606.26104)
- **Prior Approaches**: 기존에는 영향도 추정(influence functions, TrackStar, MAGIC)이나 사전/사후 혼합된 평가(퍼플렉시티 등)로 학습 데이터의 효과를 보려 했지만, small matched-pair 조건에서는 신호 대 잡음이 불안정해 stance와 vocabulary가 섞이기 쉽습니다. 특히 문서 단위 귀속은 특정 쓰기 표현이 실제로 모델의 ‘입장 선택’을 바꾸는지에 대한 직접 답을 주지 못했습니다. 또한 이전 연구들은 사후학습이 기존 정렬을 훼손할 수 있다는 신호도 보여주었습니다.

- **Core Contribution**: 이 논문은 동물복지 텍스트에서 10개 언어 특성을 한 번에 하나씩만 바꾼 paired 데이터로, fine-tuning 이후 held-out 동물복지 벤치마크에서 ‘찬성 입장 선택’이 얼마나 변하는지 행동 평가로 측정합니다. 핵심은 vocabulary-matched stance-contrast 설계를 통해 단순 단어 인식이 아니라 stance 자체 변화만 분리해 본다는 점입니다. 즉, “어떤 문장 스타일이 모델의 입장 추론을 실제로 이동시키는가”에 대한 실증적 답을 제공합니다.

- **Technical Challenges**: 기여를 위해서는 (1) 한 기능만 바꾼 채 나머지 문장 조건을 최대한 고정하고, (2) 대답 후보가 동물복지 관련 단어를 충분히 공유해 vocabulary 요인을 제거해야 했습니다. 또한 binary-choice 지표는 베이스라인 선호가 높아 천장효과가 생길 수 있어, aligned와 misaligned completion의 length-normalized log-prob 차이를 preference score로 주신호로 삼았습니다. Llama-3.2-1B에 대해 각 feature의 present/absent 변형으로 LoRA fine-tuning을 반복하고, seed 간 per-seed 차이를 paired t-test로 검정했습니다.

- **Empirical Impact**: 10개 중 8개 언어 특성이 통계적으로 유의미한 stance 변화를 만들었고, 그중 7개는 동물복지 찬성 추론을 강화했습니다(도덕 어휘, 평가적 주장, 감정 단어, 내러티브 구조, 위해 강도, 즉시 시간 프레이밍, 주장 확실성). 반대로 hedging(완곡한 추정/가능성 표현)과 concrete sensory description(구체 감각 묘사)은 찬성 입장을 희석해 평균적으로 불리하게 작동했습니다. 실무적으로는 “장면 중립 묘사보다 자신의 입장을 단정적으로 드러내라”가 권고되며, base model은 이미 강하게 친(親)동물복지 성향이라 많은 fine-tuning이 그 정렬을 깎을 수도 있다는 점도 함께 드러났습니다.



### Investigating LLM's Problem Solving Capability -- a Study on Statics Questions (https://arxiv.org/abs/2606.26103)
Comments:
          9 pages, Engineering and Technology Symposium 2026

- **Prior Approaches**: 기존 연구들은 LLM의 교육 효과를 다뤘지만, 주로 공개·오픈 문제 데이터셋에 의존해 주제별 분석이 부족하다는 한계가 지적된다. 공학 교육, 특히 기계공학에서 정역학 같은 특정 문제 유형에 대한 체계적 성능 평가는 상대적으로 제한적이었고, 보통 교재형 문항을 그대로 LLM에 질의하는 방식이 주를 이뤘다.

- **Core Contribution**: 이 논문은 정역학 문제 해결 능력을 평가하기 위해 모델 증류(distillation) 절차를 도입해 ChatGPT로부터 25개의 텍스트-only 정역학 문항을 추출한다. 또한 도표를 추가하거나 수치를 변경해 2종의 변형 데이터셋을 구성함으로써, 문제 양식과 요구 추론 복잡도가 성능에 미치는 영향을 분리해 분석한다.

- **Technical Challenges**: 핵심 과제는 텍스트 문제에서 잘 되는 능력이 도표가 포함되고 다단계 추론이 필요할 때 어떻게 무너지는지를 설계적으로 재현하는 것이다. 연구팀은 도표 도입 및 수치 변형을 통해 시각 정보의 활용 방식이 답안 단계 전반에 일관되게 적용되는지(단계 간 정보 전이)와, 다단계 추론 자체의 난도를 동시에 흔들 수 있게 했다.

- **Empirical Impact**: 실험 결과 LLM은 텍스트-only 정역학에서는 높은 성능을 보이지만, 도표가 들어가면 정확도가 감소하고 다단계 추론이 요구될수록 더 큰 하락이 나타난다. 추가 분석은 이 성능 저하가 주로 image recognition의 부족 때문이 아니라, 다단계 reasoning의 어려움과 단계별로 시각 정보를 일관되게 적용하지 못하는 문제에서 비롯됨을 시사한다.



### Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training (https://arxiv.org/abs/2606.26102)
- **Prior Approaches**: 기존 정렬은 사후학습에서 SFT와 RL로 Helpful/Honest/Harmless 같은 HHH 목표를 맞추지만, 이 과정이 선학습된 가치(예: harmless)를 훼손할 수 있다는 점이 선행연구에서 지적돼 왔다. 또한 정렬 강화를 위한 미세조정이 독해력 등 핵심 능력을 떨어뜨리거나, 입력 도메인에 따라 내부 표현이 예측 가능하게 이동한다는 이론·실증이 누적됐다. 다만 mid-training 뒤에 ‘사후학습 데이터 도메인’이 가치 보존에 미치는 영향은 직접 비교가 부족했다.

- **Core Contribution**: 이 논문은 Llama 3.1 8B를 동물 연민(Animal compassion) 중심 synthetic 데이터로 mid-trained 한 뒤, 사후학습에서 Helpfulnes(도움말) 도메인과 coding(코딩) 도메인을 달리 했을 때 가치가 어떻게 남는지 실험적으로 검증한다. SFT와 GRPO 두 패러다임에서 helpfulness 계열 학습은 AHB에서 동물 연민 점수를 유의하게 떨어뜨리지만, coding 계열 학습은 기준선(base)과 유사하게 보존한다. 더 나아가 MORU에서 일반 도덕 추론의 저하도 영어에서는 나타나지만 다국어로 합치면 사라지고, 동물 연민 효과는 언어 전반에 견조하게 유지됨을 보여준다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 선학습/ mid-training에 심어진 가치를 ‘사후학습 도메인’만으로 공정 비교하고, (2) SFT와 RL 계열이 서로 다른 업데이트 강도 때문에 생기는 효과를 분리해 해석하는 것이다. 이를 위해 동일 base 모델을 공유하고 SFT는 Dolly(도움말) vs Magicoder(코딩), GRPO는 RLHFlow(도움말) vs Magicoder 보상으로 바꿔 도메인 변수만 달리했으며, 평가에서는 Inspect AI의 동일 judge 모델을 사용해 AHB 2.2와 MORU를 일관되게 측정했다. 또한 영어 전용 학습 설정 때문에 MORU는 영어 단독과 다국어 통합을 함께 제시해, 언어 순응도 차이로 인한 혼입 가능성을 분석했다.

- **Empirical Impact**: 결과적으로 helpfulness 학습은 AHB 영어 항목에서 base 대비 동물 연민을 크게 훼손한다(예: SFT에서 35.7% vs 65.2%, GRPO에서 18.7% vs 32.0%). 반대로 coding 학습은 base와 큰 차이가 없었고, 두 독립 helpfulness 데이터셋과 두 학습 패러다임에서 동일한 방향성이 반복됐다. MORU에서는 영어 항목에서 일반 도덕 추론이 25.5%p 가량 저하되지만(46.4% vs 71.9%), 다국어 MORU에서는 해당 도메인 효과가 사라진다. 반면 동물 연민은 언어를 가로질러 전이되며, 특히 non-English 평가에서 coding 도메인이 helpfulness 대비 더 큰 이점을 보여 실무적으로 “가치가 들어간 mid-training 이후엔 coding 도메인 사후학습이 더 안전할 수 있다”는 시사점을 제공한다.



### Know2Guess: A Contamination-Aware Multi-Zone Benchmark for Knowledge-Boundary Evaluation in Large Language Models (https://arxiv.org/abs/2606.26101)
Comments:
          16 pages, 3 figures

- **Prior Approaches**: 기존 평가는 정확도 중심의 Truthfulness/Hallucination, 다중지표(HELM), 시간변화(FreshQA 계열) 등으로 분산돼 있고, abstention(선택적 미응답)과 policy refusal(정책 거부), 데이터 오염(bench leakage)을 한 프레임에서 함께 통제하긴 어려웠다. 또한 abstention을 “답하지 않음”으로 뭉개면, 불확실성 기반 신중함과 단순한 거부 행동이 섞여 경계(지식이 끝나는 지점)를 해석하기가 애매해진다.

- **Core Contribution**: 이 논문은 contamination-aware 멀티-존(multi-zone) 벤치마크로, build-time 라벨을 고정한 채 “정답 가능한 영역에서 답하기→모른다고 판단해 abstain하기(거부와 분리)” 전환을 측정하는 프로토콜을 제안한다. 총 1,200문항을 5개 도메인에 배치하고, 답하기 기대(Zone A–C)와 abstention 기대(Zone D)를 나누며, 각 문항에 오염 위험 메타데이터와 provenance를 함께 제공한다.

- **Technical Challenges**: 핵심 난제는 모델 출력에서 ‘형식 준수 abstention’이랑 ‘정책 refusal’을 구분하고, 데이터 오염이나 프롬프트 템플릿 차이가 성능을 덮어쓰지 않게 평가를 잠그는 것이다. 이를 위해 locked answer-or-abstain 프롬프트, strict/normalized 듀얼 파서, answer-only 대조군, 그리고 contamination-risk 태그 기반 메타데이터 슬라이싱을 함께 적용해 관찰된 순위가 파서·프롬프트 편향인지 검증한다.

- **Empirical Impact**: 결과적으로 generic한 비응답으로는 벤치마크가 해결되지 않았다. Qwen2.5-3B-Instruct가 전체 reliability는 최고(0.3657)지만, answer-expected 경계(Zone A–C)는 여전히 어렵고 calibration(ECE)이 높으며 benign-item에서 refusal 같은 실패 모드가 남아 있다. 또한 parser normalization과 프롬프트 변형, cost-sensitive 재가중에도 주요 순위와 결론이 유지돼, answerability·abstention·refusal·contamination을 분리 진단하는 재현 가능한 감사(audit) 절차로 의미가 크다.



### HierBias: Context-Conditioned Hierarchical Media Bias Detection with Multi-Task Type Classification (https://arxiv.org/abs/2606.26100)
- **Prior Approaches**: 기존 미디어 편향 탐지는 문장마다 독립적으로 라벨을 예측하는 방식이 주류였고, 기사 내 다른 문장들이 제공하는 맥락 신호를 구조적으로 반영하지 못했다. DA-RoBERTa나 bias-detector처럼 RoBERTa 계열을 fine-tuning하는 방법은 성능을 끌어올렸지만, 편향 유형(loaded language, framing, informational bias, source restriction)을 함께 최적화하는 데는 한계가 있었다.

- **Core Contribution**: HierBias는 문맥에 조건화된 편향 확률을 정의하고, 문장 단독 분류 대비 문서 맥락을 사용하면 Bayes error가 엄밀히 감소한다는 이론을 제시한다. 또한 이진 편향 존재 여부와 4-class 편향 유형 분류를 멀티태스크로 함께 학습해, 작은 코퍼스에서도 표본 효율을 높일 수 있음을 일반화 관점에서 뒷받침한다.

- **Technical Challenges**: 문장 레벨 정보와 문서 레벨 상호작용을 어떻게 모델로 “정식화”할지, 그리고 작은 데이터에서 멀티태스크 학습이 서로 경쟁하지 않게 할지가 핵심 난제였다. 논문은 sentence-level RoBERTa + cross-sentence Transformer aggregator로 문장 간 의존성을 집계하고, KL alignment regularizer로 태스크 간 최적해가 멀어지지 않도록 제약해 이를 해결했다.

- **Empirical Impact**: BABE와 BASIL에서 HierBias는 F1 0.853, MCC 0.723을 달성하며 기존 state-of-the-art bias-detector 대비 F1 +2.6%, MCC +4.3%(McNemar’s test p<0.05)로 개선했다. 또한 컨텍스트 aggregator 제거, type head 제거 등 ablation에서 각각의 이론적 구성요소 기여가 확인됐고, annolexical LLM-annotated 데이터 증강과 결합 시 추가 개선이 관찰됐다.



### DanceOPD: On-Policy Generative Field Distillation (https://arxiv.org/abs/2606.27377)
Comments:
          Technical Report; 39 pages, 13 figures, 9 tables; Project Page at this https URL

- **Prior Approaches**: 기존 멀티캡처블 이미지 생성 학습은 데이터 혼합·조인트 트레이닝, 파라미터 병합/어댑터 조합, 또는 추론 시 score 조합 등으로 여러 능력을 함께 맞추려 했지만, 자칫 한 능력이 다른 능력을 깎아먹는 capability interference가 자주 발생한다. 특히 편집은 입력 보존을 요구하고 T2I는 개방형 품질을 중시하며, 로컬·글로벌 편집은 서로 다른 변환 목표를 가져 같은 업데이트에서 그라디언트 충돌이나 타깃 의미 희석이 생긴다. 또한 on-policy distillation 관점이 일부 다뤄졌더라도, “여러 generative field를 어떤 상태에서 어떤 방식으로 질의할지”까지 체계적으로 설계한 접근은 부족했다.

- **Core Contribution**: DanceOPD는 flow-matching 모델에서 이미지 생성 능력을 속도장(velocity field)으로 보고, on-policy generative field distillation을 통해 단일 학생 모델이 T2I·로컬 편집·글로벌 편집 등을 함께 조합하도록 만드는 프레임워크다. 핵심은 각 샘플을 하나의 capability field로 hard-route하고, 학생이 실제로 생성하며 방문하는 상태에서 stop-gradient로 teacher(고정 소스) 속도장을 질의해 MSE로 학습한다는 점이다. 이 설계는 operator-defined fields(예: classifier-free guidance, CFG)까지 동일한 방식으로 흡수할 수 있게 한다.

- **Technical Challenges**: 첫째, 여러 field를 한 샘플 타깃에 소프트로 섞으면 어떤 능력을 배우는지 의미가 흐려지는 target-field ambiguity가 생긴다. 둘째, 데이터 상태나 teacher 궤적 같은 off-policy 상태에서만 감독하면 학생이 실제 생성 시 방문하는 분포와 불일치하는 state-distribution mismatch가 발생한다. 셋째, 같은 롤아웃에서 여러 시점을 촘촘히 질의하면 상관된(trajectory-query correlation) 감독이 과집계되어 그라디언트 편향을 유발하므로, DanceOPD는 semantic-side low-noise 영역에서 샘플당 단일 질의 상태만 사용해 이를 완화하고, 학습 목적함수는 plain velocity MSE로 단순화해 안정성을 확보했다.

- **Empirical Impact**: 실험은 Z-Image 기반 capability composition과 SD3.5-M 기반 realism-field absorption, 그리고 CFG absorption까지 포괄하며, DanceOPD가 target 능력은 강화하면서 anchor 생성 품질을 보존함을 일관되게 보여준다. 예를 들어 편집 조합에서 GEditBench 성능이 기준 대비 8%대 향상했고, 로컬+글로벌 편집 조합에서도 조합 베이스라인 대비 큰 폭(두 자릿수 %대) 개선이 나타났다. realism-field 흡수에서는 off-policy 대비 realism reward가 9%대 향상하고 reward gap의 상당 부분(85%대)을 메우면서 T2I 점수 저하를 거의 억제했으며, CFG 흡수에서도 일부 방식은 성능을 개선하지만 과도한 over-guided composition은 성능이 크게 떨어졌다.



### The Geometry of Updates: Fisher Alignment at Vocabulary Sca (https://arxiv.org/abs/2606.27242)
Comments:
          Accepted at the 43rd International Conference on Machine Learning (ICML 2026), PMLR 306. 64 pages total (main paper plus appendix), 4 figures, 29 tables

- **Prior Approaches**: 공유 어휘(vocabulary)를 쓰는 LLM 계열에서, SMILES·단백질·유전체처럼 토큰은 같지만 예측 타깃(label)만 다른 과학 문자열 도메인은 source selection이 핵심 과제가 됩니다. 기존 LogME/LEEP 등 label-aware 전이 점수는 레이블 맵이나 태스크별 class 통계를 요구해 K≈10^5급 어휘 규모에 바로 붙이기 어렵고, CKA/RSA/SVCCA 같은 표현 유사도는 에러(gradient/오차) 기하를 직접 보지 못해 activation-dark 구간에서 흔들릴 수 있습니다. 특히 head 수준에선 표현이 같아도 label-conditioned error가 달라 업데이트 방향이 달라질 수 있어 “표현→업데이트” 가정이 깨집니다.

- **Core Contribution**: 논문은 shared-output head 설정에서 head Fisher alignment가 표현 유사도가 아니라 activation- error 공동공간에서의 cosine으로 정확히 환원된다는 점을 보입니다(activation/error/coupling 분해 포함). 이를 통해 representation-similarity만으로는 transfer를 결정할 수 없음을 non-identifiability로 정리하고, 동시에 FisherSketch로 이를 학습 없이 추정 가능한 실무적 도구를 제안합니다. FisherSketch는 vocabulary scale에서도 전이 관련 업데이트 구조를 담은 “task signature”(16KB)를 만들어, 동일 프롬프트·동일 활성에서도 verbalizer shift 같은 변화가 반영되도록 합니다.

- **Technical Challenges**: 핵심 기술 난제는 vocabulary 규모에서 head의 Fisher 정렬을 직접 계산하면 error second moment(Γe) 같은 대형 통계가 필요해 계산·저장 비용이 과도하다는 점입니다. 또 표현 유사도와 달리 head 업데이트는 에러 벡터의 내적(e_i^T e_j) 같은 구조를 요구하지만, 이는 출력 좌표계가 공유되지 않으면 정의 자체가 무너집니다. 논문은 Fisher alignment의 product-kernel mean embedding 형태를 이용해 Fisher matrix를 만들지 않고, SRHT 기반 random-feature/단일 스트리밍 패스로 task signature를 추정하며(스트리밍 상태 약 192KB), 이를 통해 K=128~256 head 정렬을 현실적인 비용으로 근사합니다.

- **Empirical Impact**: 실험에서 FisherSketch는 자연스러운 도메인 shift에서는 activation-only 기준선과 경쟁하거나 우위를 보이고, verbalizer shift(고정 prefix 분류)에서는 activation-only가 랜덤 수준으로 붕괴한 반면 FisherSketch는 top-1 66.7%까지 달성합니다. 또한 Llama-3.1-8B에서 100개 도메인 평가로 전이 순위에서 실용적 성능을 확인했으며, 분자 SMILES 9개 도메인 예시에서는 FisherSketch가 cross-domain perplexity reduction과 유의하게 상관(ρ_s=0.53, p=0.006)하는 반면 activation-only는 유의하지 않았습니다. 나아가 내부 계층/공유 매개변수 검증과 verbalizer-shift 진단을 통해 “활성 유사도만으로는 부족할 때”에도 FisherSketch가 informative하다는 점을 경험적으로 뒷받침합니다.



### Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvemen (https://arxiv.org/abs/2606.27226)
Comments:
          Acceepted to the Second Workshop on Compositional Learning at ICML 2026, Seoul, South Korea

- **Prior Approaches**: 기존 평가는 ROUGE/BLEU 같은 어휘 중복 지표가 의미·사실성을 충분히 반영하지 못하고, BERTScore 등 임베딩 기반 또는 생성형 지표도 한계가 남아 있습니다. LLM-as-Judge 방식은 Likert 점수나 전체 평가를 제공하지만, 근거가 불투명해 디버깅이 어렵고 자극(verbosity/position/self-enhancement) 편향에도 취약하다는 지적이 있습니다. UniEval처럼 다차원 분해를 시도한 방법도 있으나, 학습 기반이거나 단일 yes/no의 굵기 문제로 세밀한 판별력에 제약이 있습니다.

- **Core Contribution**: 이 논문은 BinEval(=BinEval, atomic binary questions 기반 평가)이라는 훈련 없는 프레임워크를 제안합니다. 평가 기준을 원자적인 yes/no 질문들로 분해하고, LLM이 출력마다 각 질문에 독립적으로 답한 뒤 이를 다차원 점수와 종합 점수로 집계해 해석 가능하고 진단 가능한 피드백을 제공합니다. 또한 질문 단위 실패 정보를 이용해 평가기와 생성기 프롬프트를 반복 개선하는 최적화 루프를 함께 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) task prompt를 작업별로 유효한 이진 질문 세트로 자동 분해하고, (2) 질문 답변을 집계해 사람 판단과 잘 맞는 점수 분포/변별력을 유지하는 데 있습니다. 논문은 meta-prompt이 요구사항을 먼저 요약한 뒤 이를 기준으로 이진 질문과 위반 예시를 생성하고, 차원별·전체 점수는 binary verdict의 평균으로 산출하며, 마지막으로 질의/프롬프트를 불일치(동의/반대)에서 얻은 lesson으로 업데이트하는 방식으로 이를 해결합니다. 또한 질문 재생성까지 포함해 업데이트 효과를 키우되, 과도한 재작성으로 성능이 무너지는 현상도 실험에서 확인합니다.

- **Empirical Impact**: BinEval은 SummEval, Topical-Chat, QAGS에서 강한 기준선(예: UniEval, G-Eval)과 비교해 점수 상관을 같거나 능가하며, 특히 QAGS 같은 사실 일관성(환각) 벤치마크에서 강한 성능을 보였습니다. Likert식 전체 판단 대비 점수 분포를 사람과 더 가깝게 맞추고 ceiling effect를 줄여, 애매한 출력과 명백히 나쁜 출력을 더 잘 가르는 것이 관찰됩니다. 더 나아가 SummEval과 IFBench에서 self-update 및 cross-model update 모두에 대해 질문 단위 피드백이 반복 프롬프트 최적화에 실질적으로 활용됨을 보여줍니다.



### HarmVideoBench: Benchmarking Harmful Video Understanding in Large Multimodal Models (https://arxiv.org/abs/2606.27187)
- **Prior Approaches**: 기존 harmful video 벤치마크는 대체로 harmful/non-harmful 이진 분류에 초점을 두어, 비디오가 암시적으로 전달하는 깊은 위해(의도/문맥)를 충분히 포착하지 못했다. 또한 정답 맞히기 여부만 평가하고 왜 그런지 설명하는 rationales가 거의 없어, 모델이 표면 단서나 편향된 상관관계로 맞출 가능성이 있어도 진단이 어렵다.

- **Core Contribution**: 이 논문은 HarmVideoBench를 제안하며, 유해성 이해를 단일 결정이 아니라 관찰 가능한 증거에서 시작해 클립 내부 의미, 나아가 클립을 넘어선 추론으로 확장되는 3단계 계층적 진단으로 재구성한다. 1,379개 영상과 4,137개 객관식 Q로 구성되며, 세 범주(Observable Evidence, Clip-Internal Meaning, Beyond-Clip Reasoning)별로 모델이 어디서 실패하는지 분해해 측정한다.

- **Technical Challenges**: 기여의 핵심은 다차원 유해성의 경계를 객관적으로 질문/선지에 반영하는 것으로, AI가 만든 후보를 인간 검수(다중 어노테이터 합의·고급 adjudication)로 정제해 근거성과 난이도를 보장한다. 또한 모델이 “이 문제는 클립만으로 풀리는가, 아니면 추가 맥락이 필요한가”를 잘못 가정하면 추론 깊이에서 무너져, 이를 완화하는 구조로 BCR(Boundary-Constrained Reasoning)을 설계해 필요한 경우에만 문맥을 선택적으로 retrieve하고 경계에 맞춰 decoding을 수행한다.

- **Empirical Impact**: 19개 최신 모델을 평가한 결과, 대부분은 Observable Evidence에서는 상대적으로 강하지만 Clip-Internal Meaning과 Beyond-Clip Reasoning에서 현저히 약하며, 특히 문화·역사·사회적 지식이 필요한 항목에서 격차가 커졌다. BCR을 적용하면 기준 모델의 매크로 평균이 61.7%에서 84.4%(state-of-the-art)로 크게 상승해, 유해성 이해가 단순 지각 능력만이 아니라 증거 경계 제어와 맥락 연결에 달려 있음을 실증적으로 뒷받침한다.



### Just how sure are you? Improving Verbalized Uncertainty Calibration in Medical VQA (https://arxiv.org/abs/2606.27023)
- **Prior Approaches**: 기존 LLM 기반 confidence 보정은 주로 텍스트-only에서 토큰 확률, verbalized confidence(자연어로 확신 표현), consistency/perturbation 등을 통해 신뢰도를 조정했지만, 의료 이미지 이해의 멀티모달 특성을 반영하지 못했습니다. 특히 verbalized confidence는 추가 학습 없으면 과신(overconfidence)되기 쉬워 임상 신뢰성에 한계가 컸습니다. 또 멀티모달에서의 보정은 대체로 prompt 엔지니어링 중심이거나 다른 비전 태스크(detector) 위주의 미세조정에 머물렀습니다.

- **Core Contribution**: 이 논문은 Medical VQA에서 MLLM이 ‘정답을 맞힐 확률에 비례해’ verbalized confidence를 내도록, training-based 보정 프레임워크를 제안합니다. 핵심은 2×2 factorial perturbation(이미지 유무/텍스트 무결성)로 시각 근거와 언어 priors의 의존도를 드러내고, 그 차이를 confidence 학습에 직접 연결하는 데 있습니다. 또한 Brier-style calibration loss, anchor regularizer, contrastive alignment, 그리고 top-k KL divergence regularizer를 결합해 보정 성능과 답변 능력 유지(및 포맷 준수)를 동시에 노립니다.

- **Technical Challenges**: 가장 큰 난제는 멀티모달 의료 문맥에서 “시각 근거가 사라질 때 confidence가 얼마나 내려가야 하는가”를 학습 신호로 설계하는 것이었습니다. 이를 위해 이미지: 원본 vs black image, 텍스트: 원본 vs 옵션 섞기/교란을 교차시켜 조건별 정답률(분수 정확도) 추정치를 만들고, confidence 토큰 분포에 Brier 기반 loss와 evidence-aware alignment를 적용했습니다. 또 Brier 최적화만으로는 극단값으로 confidence가 무너지는 collapse를 막기 위해 anchor loss를 추가하고, fine-tuning 중 답변 분포가 망가지지 않도록 answering token 위치에 top-k KL regularizer를 걸어 안정화를 달성했습니다.

- **Empirical Impact**: 3개 Medical VQA 벤치마크(OmniMedVQA/PMC-VQA/MedXpertQA)와 2개 아키텍처(MedGemma 4B IT, Qwen2 VL 7B Instruct)에서 ECE와 Brier Score, AUROC를 동시에 개선했으며, calibration error는 60% 이상 줄고 discrimination도 26% 이상 향상됩니다. 특히 difficulty 레짐이 다른 세 환경에서 ECE 분포 폭이 가장 좁게 유지되어 특정 데이터셋 평균 정확도에 의존한 편향 가능성을 낮췄습니다. 다만 MedXpertQA에서는 AUROC가 거의 chance 수준으로 남아(멀티스텝 임상 추론 난이도) verbalized confidence의 한계가 확인됐고, ablation으로 각 loss 성분(특히 alignment가 discrimination을, KL이 정확도/포맷 유지)을 분리 검증했습니다.



### Einstein World Models (https://arxiv.org/abs/2606.26969)
Comments:
          12 pages (9 without references), 2 figures, 1 algorithm

- **Prior Approaches**: 기존 Chain-of-thought는 중간 추론을 언어로만 외부화해, 물체 식별·접촉·열·운동처럼 장면 수준 변수가 중요한 문제에서 한계가 나타난다. 또한 world model 계열은 주로 관측 기반 예측이나 행동 조건부 예측에 머물러, 경험 밖의 반사실적 thought experiment를 정밀하게 다루기 어렵다. 비슷하게 Visualization-of-Thought·whiteboard-of-thought 류는 시각 메모/그림을 중간 산출물로 쓰지만, 이를 비디오 롤아웃 형태로 ‘가설을 검토’하는 통합 메커니즘은 상대적으로 약하다.

- **Core Contribution**: Einstein World Models(EWMs)는 LLM이 스스로 판단해 world-module로부터 짧은 visual-temporal rollout(비디오 롤아웃)을 생성·조회하고, 그 결과를 추론 trace에 ‘검증 가능한 가설’로 포함하도록 하는 구조를 제안한다. 핵심은 롤아웃을 정답 대신 참고할 inspectable hypothesis로 취급해, 텍스트만으로 어려운 장면 전개·반사실적 사건 상상을 추론에 보조하는 것이다. 또한 “reasoner(추론자)”와 “world-module(롤아웃 생성기)”을 분리해, 생성된 비디오를 외부에서 관찰·디버깅할 수 있게 만든다.

- **Technical Challenges**: 어떤 질문에서 언제 비디오 thought experiment가 이득인지, 그리고 어떤 world-module query로 롤아웃을 뽑아야 이후 추론이 강화되는지를 학습해야 한다. 저자들은 SFT로 EWM trace 포맷(언어·tool_call·visual_rollout·정답)을 익힌 뒤, verifiable final answer에 기반한 RLVR/GRPO 스타일 학습으로 world-module 호출의 선택성을 최적화한다. 더해 world-module 자체의 신뢰도 문제를 “물리적 그럴듯함”으로 직접 보정하기보다, 롤아웃의 inspectability/faithfulness가 추론에 실제로 기여하는지와 모듈 선택(예: diffusion 기반 렌더러 품질, 상이한 편향의 앙상블)을 함께 다루는 방향을 제시한다.

- **Empirical Impact**: 이 논문은 실험 결과가 아니라 먼저 학습 가능한 시각 thought experimentation의 데이터·학습 청사진을 제시하는 데 초점을 둔다. 특히 기존 공개 데이터가 ‘문제의 시작이 텍스트이며 모델이 스스로 시각화 여부를 결정해야 하는’ 설정을 충분히 제공하지 못한다고 진단하며, SimpleBench 같은 제한적 예시를 보완할 datasets 공모를 제안한다. EWMs가 성공하면 LLM이 단순 텍스트 기호를 넘어 외부화된 visual walk-through로 추론을 확장하고, 중간 가설을 공유·점검하는 디버깅 경로를 열 수 있다는 점에서 영향력이 크다는 입장이다.



### Jailbreaking for the Average Jane: Choosing Optimal Jailbreaks via Bandit Algorithms for Automatically Enhanced Queries (https://arxiv.org/abs/2606.26936)
- **Prior Approaches**: LLM jailbreak 관련 연구가 빠르게 늘었지만, 실전에서 비전문 공격자가 얼마나 쉽게 “배포된 모델”을 뚫을 수 있는지가 핵심 질문으로 남아 있다. 기존 접근은 알려진 jailbreak를 정해진 방식으로 탐색하거나, 안전 벤치마크를 단순히 더 넓은 카테고리로 구성해 깊이(depth)를 희생하는 경향이 있어 실제 공격 효율을 과소평가할 수 있다.

- **Core Contribution**: 이 논문은 비전문 공격자 관점에서 “재능 있는 jailbreak 선택”과 “행동 가능한(유해) 응답을 끌어내는 쿼리 작성”을 결합한 공격 파이프라인을 제시한다. 이를 위해 multi-armed bandit 기반으로 최적 jailbreak 정책을 온라인으로 학습하고, 별도의 Frankenste inBench(FrankensteinBench)에서 단순/복잡 쿼리를 분류해 악성 의도를 기술적 표현으로 우회하는 자동화 전략을 평가한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 모든 jailbreak를 전수평가하면 nT 규모의 추가 질의가 필요하다는 계산 비용과 (2) 보상(유해성)은 jailbreak 자체뿐 아니라 원본 쿼리의 기술적 깊이에 의해 달라진다는 점이다. 저자들은 EXP3/부분정보 bandit 계열로 매 라운드 선택한 소수 jailbreak에 대해서만 피드백을 받아 T 규모 질의 내에서 준최적 정책으로 수렴하도록 구성하고, 쿼리 복잡도는 readability 점수와 LLM-as-a-judge를 결합한 분류기로 자동 라벨링한다.

- **Empirical Impact**: FrankensteinBench(총 11,279개 악성 쿼리)와 15개 SoTA open-weight LLM(최대 120B 규모) 실험에서 bandit 기반 공격은 평균 성공률이 최대 97%까지 올라가며, 단일 top jailbreak 대비 성능을 끌어올린다. 특히 복잡한 쿼리는 기준선(재작성 없음)에서도 ASR을 높일 뿐 아니라, jailbreak 적용 시 공격 성공률을 모델 전반에서 평균 최대 26%까지 더 끌어올려 자동화 가능한 프롬프팅 전술로서의 실효성을 보여준다.



### AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems (https://arxiv.org/abs/2606.26859)
Comments:
          Authors are listed alphabetically by their first name

- **Prior Approaches**: 기존 추천 알고리즘 연구는 엔지니어가 가설을 만들고 프로덕션 코드를 수정한 뒤 A/B 실험을 실행·분석하고 온라인 성과를 해석하는 ‘아이디어-런치’ 흐름에 크게 의존해 왔다. 이 구조 때문에 성과가 실험 지식과 누적 학습으로 폭발적으로 확장되기보다, 인력 규모에 거의 선형으로만 비례해 스케일링이 제한된다.

- **Core Contribution**: 이 논문은 프로덕션 환경에서 동작하는 멀티에이전트 시스템 AgentX를 제안하며, 추천 실험의 생산 함수를 사람이 아니라 에이전트가 주도하도록 재구성한다. AgentX는 자동으로 추천 실험을 생성·구현·평가하고 그 결과로부터 학습하는 self-evolving 개발 엔진을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술 과제는 실험을 자동화하되 안전하게 롤아웃하고, 실패까지 포함해 지식으로 축적하며, 코드 생성이 실제 저장소/시스템 제약을 만족하도록 검증하는 것이다. AgentX는 Brainstorm Agent로 실행 가능한 제안을 순위화하고, Developing Agent에서 repository-grounded 코드 생성과 다차원 reliability 검증을 수행하며, Evaluation Agent가 guardrail-veto된 A/B 판단으로 안전한 온라인 평가를 실시한 뒤 SGPO의 semantic-gradient 업데이트(SGPO)로 실행 궤적을 에이전트에 반영해 점차 개선한다.

- **Empirical Impact**: 논문은 AgentX가 수작업 워크플로우가 따라가기 어려운 스케일과 속도로 추천 실험을 반복하며, 성공과 실패 모두를 구조화된 knowledge asset으로 전환해 누적 학습을 강화한다고 제시한다. 결과적으로 추천 연구·실험 사이클을 산업화된 closed loop로 바꿔, 인력 의존적 혁신을 evidence/compute 기반의 자기 개선형 혁신으로 전환하는 의미가 있다.



### KARLA: Knowledge-base Augmented Retrieval for Language Models (https://arxiv.org/abs/2606.26807)
- **Prior Approaches**: 기존에는 LLM에 대한 파인튜닝이나 프롬프트 엔지니어링으로 지식과 사실을 맞추려는 접근이 주로 쓰였지만, 지식이 바뀌면 재학습이나 재설계가 필요해 유지보수가 어렵습니다. 또한 검색을 단순히 붙이는 방식(RAG)은 가능해도, 생성 과정에서 언제 어떤 사실을 가져와야 하는지의 정교한 제어가 한계로 지적돼 왔습니다.

- **Core Contribution**: 이 논문은 생성 중에 LLM이 지식베이스(knowledge base)에서 사실을 자동으로 끌어오도록 하는 방법을 제안합니다. 핵심은 특별 토큰을 만들어 토큰 생성 흐름 안에서 지식베이스 질의를 트리거하게 하고, 결과가 지식베이스 출처로 추적되며, KB 편집만으로 사실을 업데이트할 수 있게 한 점입니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 생성 토큰과 외부 지식 호출을 자연스럽게 결합해, 필요한 순간에 정확히 질의·삽입이 일어나도록 학습하는 것입니다. 저자들은 특별 토큰을 통해 KB 조회를 유도하고, 그 토큰이 실제로 질의 실행으로 연결되도록 학습해 사실적 근거를 강화했습니다.

- **Empirical Impact**: 실험 결과, 이 방법은 짧은 생성과 긴 생성 모두에서 factual grounding을 개선했으며, 파라미터 업데이트 없이 KB 수정으로 사실 반영이 가능함을 보여줍니다. 또한 더 작은 모델이 더 큰 모델과 유사한 사실 정확도를 달성할 수 있어, 비용 대비 성능 관점에서도 의미가 큽니다.



### AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing (https://arxiv.org/abs/2606.26787)
Comments:
          Accepted by KDD 2026 Applied Data Science Track (Oral presentation)

- **Prior Approaches**: 기존 동적 가격 모델은 룰 기반 휴리스틱이나 수요-가격 탄력성 추정/최적화에 의존하는 경우가 많지만, 결정의 투명성이 낮고 리뷰·상품 설명 같은 비정형 정보를 충분히 쓰기 어렵다. 또한 단기 매출 중심 최적화가 누적 GMV, ROI, 마일스톤 같은 장기 목표와 어긋나며, 공격적인 할인은 마진 훼손과 운영 제약으로 이어질 수 있다. RL 접근은 장기 보상을 직접 최적화하려 해도(가치함수 기반) 해석 가능성과 오프라인 분포이동, 보상 희소성 문제가 남는다.

- **Core Contribution**: 논문은 LLM 기반 동적 가격 프레임워크 AIGP(Artificial Intelligence Generated Pricing)를 제안한다. AIGP는 도메인지식·구조화 데이터·텍스트 컨텍스트를 프롬프트에 결합해 해석 가능한 가격 결정을 만들고, 장기 비즈니스 목표에 정렬되도록 정책을 학습한다. 핵심 모듈은 Long-Term Value Estimator(LTVE)로, 가격 액션의 장기 가치를 평가해 Direct Preference Optimization(DPO)용 선호 쌍을 자동 생성함으로써 장기 목표 정렬을 가능하게 한다.

- **Technical Challenges**: 기술적 난제는 (1) 가격이 미래 노출/트래픽/매출에 영향을 미치는 지연 보상을 오프라인에서 안정적으로 반영하는 것과 (2) 장기 성과를 직접 비교할 신뢰도 높은 학습 신호를 확보하는 데 있다. 이를 위해 LTVE는 critic-only 구조의 오프라인 RL로 6개월 운영 로그(5백만+ 전이)를 학습해 후보 액션을 장기 가치로 스코어링하고, 그 스코어를 기준으로 DPO의 chosen/rejected 쌍을 만든다. 또 결정 최적화 과정에서 비교 불가능한 긴 추론문이 신호를 흐리지 않도록 [ACT_ONLY] 제어 토큰으로 ‘액션 토큰’에 DPO를 집중시키되, 배포 시에는 투명성을 위해 reasoning을 다시 생성하도록 설계했다.

- **Empirical Impact**: AIGP는 Tao Factory에서 오프라인 평가와 대규모 온라인 A/B 테스트로 효과를 검증했으며, 14일 기준 생산 기준선 대비 GMV +13.21%, ROI +7.59%, 마일스톤 달성률 +8.20% 개선을 보고한다. 동시에 추론 기반의 가격 합리화가 제공되어 운영 관점의 해석 가능성과 신뢰성을 유지한다. 전개 측면에서도 teacher-student distillation과 SFT를 통해 30B급 학생 모델이 대규모 생산 환경에서 효율적으로 동작하도록 맞췄다는 점이 강조된다.



### Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models" (https://arxiv.org/abs/2606.26783)
Comments:
          21 pages, 2 figures

- **Prior Approaches**: 모델 편집은 사전학습된 LLM이 가진 사실 오류나 outdated knowledge를 “재학습 없이” 소량 파라미터 변경으로 고치려는 접근이며, 그중 locate-then-edit는 먼저 타깃 사실을 저장한 파라미터를 찾고 그 주변만 교란해 수정한다. ROME·MEMIT은 특정 사실의 성공에는 강하지만, 새 편집이 기존에 맞았던 지식까지 함께 밀어내는 보존 실패가 순차 편집이 누적될수록 커지는 한계가 있다.

- **Core Contribution**: AlphaEdit은 locate-then-edit의 “무제약 업데이트로 인한 지식 붕괴” 문제를, 보존할 지식이 놓인 방향과 직교하도록 업데이트를 null space에 투영하는 방식으로 정면 돌파한다고 주장한다. 이 연구는 AlphaEdit의 재현을 바탕으로, 이론적 보장(보존 지식 불변)이 실제로는 어떤 조건에서 흔들리는지(아키텍처·편집 스케일·생성 품질/안전성)까지 확장해 점검한다.

- **Technical Challenges**: 재현 결과, 확률 기반 핵심 지표(대상 사실 반영/지역성 등)는 원 논문 설정에서 대체로 잘 맞지만, fluency·consistency는 모델이 생성 붕괴를 보이면서 원 보고치보다 크게 낮게 나타났다. 또한 Qwen2.5·Llama3.2·Phi3·Gemma2처럼 아키텍처가 달라지면 이득이 균일하게 일반화되지 않았고, 저자들은 locate 단계의 subject-token localization 휴리스틱과 closed-form 최적화 가정이 fused projection, post-MLP RMSNorm 같은 구조에서 깨질 수 있다고 추적한다. 마지막으로 3,000을 넘어 10,000까지 편집을 늘리면 성능이 원래 구간에선 안정적이지만 더 큰 편집 수에서 점진적 열화가 나타나 null-space 보호가 “무조건적”이라기보다 “유한 범위 내”에 가깝다는 신호를 확인했다.

- **Empirical Impact**: 원래 평가 범위에서는 AlphaEdit의 효능/일반화/구체성(지역성) 개선이 재현되어, 논문이 주장한 보호 메커니즘의 실효성이 확인됐다. 하지만 추가 실험에서는 general downstream 능력과 safety-relevant refusal 행동이 BoolQ·HellaSwag·XSTest에서 장기 순차 편집에 따라 눈에 띄게 저하되었고, 이는 배포 시 스코프 밖 신뢰가 위험할 수 있음을 시사한다. 요약하면 “원 설정에서는 강력하지만, 아키텍처 차이와 편집 규모가 커질수록 이론 보장의 실사용 신뢰도는 민감하게 흔들린다”는 실증 결론으로 정리된다.



### Structure Before Collapse: Transient semantic geometry in next-token prediction (https://arxiv.org/abs/2606.26749)
- **Prior Approaches**: Neural Collapse(NC) 이론은 balanced one-hot 감독 아래에서 마지막층 표현이 입력 의미와 무관하게 출력 라벨만 반영하는 대칭 ETF( Simplex Equiangular Tight Frame ) 기하로 수렴한다고 본다. 또한 sparse soft-labeled에서의 의미 기하는 보통 공동등장 통계(공유되는 다음 토큰 확률)를 매개로 나타나지만, long-context 학습에서는 그 “브리지”가 희박해진다는 문제가 있다. 기존 연구들은 이미지 분류 등에서 의미가 관측된다는 실험적 모순을 제시했지만, 언어 모델의 one-hot 다음토큰 조건에서 그 기하가 언제/어떻게 생기는지의 동역학은 명확하지 않았다.

- **Core Contribution**: 이 논문은 latent semantic 요인을 알 수 있는 합성 언어 3종을 설계해, 입력에는 의미적 유사성이 존재하지만 학습 감독은 철저히 one-hot이 되도록 만든 뒤 표현 기하의 변화를 추적한다. 학습 초기에 representations가 Sem-RSM(잠재 의미 범주 기반)과 강하게 정렬되며, 이후 점차 ETF-RSM(출력 라벨만 반영)로 붕괴하는 “일시적 semantic geometry” 현상을 보인다. 즉, one-hot 제약만으로는 최종 상태가 아닌 초기 단계에서 의미 구조가 먼저 형성될 수 있음을 실증적으로 연결한다.

- **Technical Challenges**: 핵심 난제는, context는 대부분 한 번만 등장해 레이블 간 공통 next-token 통계가 사라지는데도 gradient descent가 범주적 의미 구조를 어떻게 복원하는가이다. 저자들은 이를 Representational Similarity Analysis(RSA)로 정량화하며, 모델 임베딩의 Gram matrix 기반 Emp-RSM을 Sem-RSM과 ETF-RSM에 대한 Pearson 상관으로 비교해 학습 단계별 지배 요인을 분리한다. 또한 충분한 capacity와 학습 시간이 주어지면 결국 대칭 ETF로 수렴하는 과정(phase transition)을 관찰하고, 이를 설명하기 위한 수학적 단순화 모델과 Gram matrix 분석, 그리고 Unconstrained features model(UFM) 기반의 예비 수정안을 제안한다.

- **Empirical Impact**: 3가지 합성 설정(색-도형 교차, Zipfian latent category를 갖는 선형 문법, 계층적 문법+ c-command 기반 제약)에서 공통적으로 semantic 정렬이 학습 초기에 먼저 최고조에 도달한 뒤 ETF 기하로 이동한다. 모델 크기가 클수록 초기 의미 기하를 더 강하게 포착하면서도 말기에는 ETF 수렴 또한 더 잘 나타나, bottlenecking 같은 우연 효과만으로 설명하기 어렵다는 점을 시사한다. 결과는 Neural Collapse 이론이 예측하는 최종(terminal) 상태와는 별개로, 언어 모델이 one-hot 감독 하에서도 초기에는 의미적 범주 구조를 형성할 수 있음을 보여주며 학습 동역학 관점의 해석을 강화한다.



### HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction (https://arxiv.org/abs/2606.26744)
- **Prior Approaches**: Speculative decoding은 경량 drafter가 후보 토큰을 제안하고, target이 이를 병렬 검증해 생성 속도를 높이는 방식이다. DeepSeek-V4의 native Multi-Token Prediction(MTP)은 초반 토큰의 초안 성능은 좋지만, 검증되지 않은 중간 토큰이 누적되면서 후반 위치에서 draft accuracy가 급격히 떨어져 수용 길이(accepted prefix)가 제한된다. DFlash는 한 번에 블록을 초안으로 예측해 오류 누적을 줄이지만, DeepSeek-V4의 multi-hyper-connection(MHC) 잔차 스트림 구조와의 정합성이 깨져 그대로 적용하기 어렵다.

- **Core Contribution**: HyperDFlash는 DeepSeek-V4의 MHC 구조에 맞춘 블록 병렬 speculative decoding 프레임워크를 제안한다. 핵심은 MHC의 multi-path 잔차 표현에서 target LM-head로 이어지는 예측 경로와 drafter의 조건 신호를 일치시키는 설계, 그리고 경량이면서도 입력 의존적으로 경로를 줄이는 path aggregation(reducer)이다. 여기에 초기 위치 품질을 안정화하기 위해 LM-head 기반 targeted KL distillation을 조합해 초안의 초기 수용성을 끌어올린다.

- **Technical Challenges**: 가장 큰 기술 난제는 drafter가 블록을 예측할 때, 기존 DFlash처럼 중간 계층 특징을 쓰거나 단순 선형으로 MHC multi-path 잔차를 축약하면 target의 native collapse 경로와 ‘feature misalignment’이 생긴다는 점이다. HyperDFlash는 pre-collapse residual states만을 조건 신호로 사용해 구조적 정합성을 유지하고, 무거운 dense linear compressor 대신 target의 hc_head와 동일한 input-aware gated 형태를 inherited 방식으로 도입해 파라미터를 크게 줄이면서도 경로 축약의 의미를 맞춘다. 또한 KL distillation은 정보 세트가 달라지는 뒤 위치의 학습 충돌을 피하기 위해 초기 블록 위치에만 제한적으로 적용해 학습 안정성과 draft 품질을 동시에 노린다.

- **Empirical Impact**: 실험에서 HyperDFlash는 math, code, chat 벤치마크 전반에서 native MTP와 vanilla DFlash 적용안 모두를 일관되게 능가했다. 예를 들어 non-thinking 모드, 온도 0에서 MTP(3) 대비 평균 accepted length를 2.93→3.69로 높이며 속도 향상도 2.25×→2.80×로 개선했다. 같은 6-step 예산에서도 HyperDFlash는 MTP(6) 대비 accepted length(3.08→3.69)와 speedup(1.76×→2.80×)을 함께 끌어올렸고, vanilla DFlash(6)의 accepted length 2.14를 유의미하게 상회해 MHC-aware 조건화와 gated reduction의 필요성을 실증했다.



### Do Safety Guardrails Need to Reason? LeanGuard: A Fast and Light Approach for Robust Moderation (https://arxiv.org/abs/2606.26686)
Comments:
          9 pages, 6 figures, 3 tables. Project page: this https URL ; code and models: this https URL

- **Prior Approaches**: 기존 LLM 안전 가드는 큰 디코더 기반 생성형 분류기로 프롬프트/응답을 점검해 거부 여부를 자연어 판정으로 내리는 방식을 주로 써왔다. 또 다른 흐름은 GuardReasoner류처럼 CoT(chain-of-thought)를 먼저 생성한 뒤 판정을 내리며, 단계적 추론이 정확도와 신뢰성을 높인다고 가정한다. 하지만 이 설계는 실제 배포 환경(온디바이스·에ంబ디드 로봇)에서 요구되는 ‘경량·저지연’과 충돌할 수 있다.

- **Core Contribution**: 논문은 “안전 가드는 진짜로 CoT를 필요로 하는가”를 같은 베이스(same-base) 비교로 검증한다. GuardReasoner 학습 데이터를 그대로 두고, 추론 생성만 제거(나머지는 고정)했을 때 중재 정확도가 개선되지 않는지 확인하며 그 결과를 LeanGuard로 정리한다. 또한 추론 가드가 더 무거울수록 강건성도 좋아지는지(학습 라벨 노이즈에 대한 견고성)까지 함께 반박한다.

- **Technical Challenges**: 핵심 실험적 도전은 ‘추론 유무’만 바꿔 다른 요인(아키텍처/스케일/데이터/목적)을 섞지 않는 통제 비교를 만드는 것이다. 이를 위해 ModernBERT-large(분류기)와 Llama/T5 기반의 생성형 가드를 동일 코퍼스에서 CoT 포함/라벨-only로만 분리 학습하고, LeanGuard는 395M label-only 인코더로 단일 forward pass에 verdict를 산출하게 설계했다. 계산 비용은 CoT 생성으로 인해 늘어나는 토큰 수에 비례해 증가하므로, 논문은 단일패스가 효율·성능을 동시에 만족하는지 정량화한다.

- **Empirical Impact**: 결과적으로 395M LeanGuard는 공개 벤치마크에서 평균 F1 82.90±0.26을 달성하며, 더 큰 디코더 기반 reasoning guard와 유사한 성능을 단일 forward pass로 맞춘다(추론 계산량 약 100배 절감). 또한 학습 라벨 노이즈를 주입해도 F1이 잘 유지되고, 엄격한 false-positive rate 조건에서 recall이 추론 가드보다 더 잘 보존되어 ‘무거운 추론이 더 강건하다’는 주장을 약화시킨다. 저자들은 현행 가드레일 벤치마크가 추론의 필요성을 충분히 구분하지 못할 수 있으며, moderation에서 CoT의 필수성이 아직 증명되지 않았다고 결론짓는다.



### From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning (https://arxiv.org/abs/2606.26629)
Comments:
          21 pages, 4 figures, 6 tables

- **Prior Approaches**: 연속학습에서 흔히 쓰이는 weight-space regularization(EWC, SI, MAS)은 파라미터마다 중요도를 추정하고 이후 업데이트를 벌점으로 억제한다. 하지만 LLM에 적용하면 과도하게 plasticity를 줄여 성능이 잘 나오지 않으며, 그 원인이 단순 튜닝 실패가 아니라 모델의 polysemantic 성질과 “중요도 단위(파라미터)”가 “보호해야 할 지식 단위(개념/특징)”를 잘 분리하지 못하기 때문이라고 본다. 또한 한 번에 너무 많은 개념이 같은 가중치에 겹쳐 있어(초중첩) 특정 개념을 보호하려다 다른 개념의 학습까지 함께 제약되는 비선택적 보호가 문제가 된다.

- **Core Contribution**: 이 논문은 파라미터 대신 activation space에서의 regularization을 제안하며, 이를 위해 pretrained Sparse Autoencoders(SAE)를 “monosemantic feature dictionary”로 사용한다. 각 태스크에 대해 SAE feature 중 무엇이 해당 태스크에 관련되는지 마스크를 만들고, 관련 feature는 적응(plasticity)하도록, 비관련 feature는 안정(stability) 예산 안에서 드리프트하지 않도록 균형을 맞춘다. 특히 현재 태스크 데이터로 마스크를 만든 뒤에는 이전 태스크 데이터(리플레이)를 저장하지 않고, 오직 compact feature mask만 유지하는 방식이라 메모리 효율도 높다고 주장한다.

- **Technical Challenges**: 핵심 난제는 “어떤 좌표계에서” 안정성과 가소성을 서로 다른 제약으로 명시할 수 있느냐이다. 연구팀은 constrained optimization 관점에서 stability(보호 영역 드리프트 상한)와 plasticity(관련 영역 최소 적응)를 각각 상·하한 제약으로 정의하고, Lagrangian relaxation과 squared-hinge 완화로 protect loss와 guide loss를 도출해 학습 가능한 단일 목적함수로 정리했다. 또한 SAE를 고정된 좌표계로 두고 feature drift를 계산하되, 마스크는 단순 활성 크기뿐 아니라 SAE decoder geometry를 이용한 k-NN 유사도 전파로 연관 feature까지 연속값(0~1)으로 확장해 제약의 선택성을 확보한다.

- **Empirical Impact**: TRACE-5000과 MedCL(총 18개 태스크)에서 SAE-guided activation regularization은 non-architectural 접근 중 최고 성능을 보이고 EWC 같은 전통적 weight-space regularization을 능가한다. 리플레이를 없앤 설정에서도 강점이 유지되며, 작업별로 과거 예시를 재방문하지 않고도 학습이 잘 된다는 점이 차별화 포인트다. 더 나아가 polysemanticity 가설을 지지하는 실험도 제시하는데, SAE feature 기준에서는 태스크-관련 표현이 선형 분리 가능(AUC 0.88)하지만 weight basis에서는 우연 수준(AUC 0.50)에 가깝고, weight 보호가 다음 태스크의 기능을 광범위하게 제약하는 반면 SAE feature 보호는 더 선택적으로 작동(43–61% 수준의 collateral constraint)함을 보인다.



### Adversarial Diffusion Across Modalities: A Fusion Survey of Attacks, Defenses, and Evaluation for Text, Vision, and Vision-Language Models (https://arxiv.org/abs/2606.26566)
- **Prior Approaches**: 기존 LLM 레드팀은 적대 접미사 최적화, 유전 알고리즘 기반 jailbreak, 블랙박스 반복 재작성, autoregressive 공격 생성기 등으로 취약점을 찾는 데 집중해 왔다. 반면 이미지 분류기 공격이나 diffusion 기반 입력 정화 같은 트랙은 별도 벤치마크·위협모델로 발전해 서로 연결이 약했고, 텍스트에 diffusion을 쓰는 공격은 거의 정리되지 않았다.

- **Core Contribution**: 이 논문은 diffusion 기반 공격/방어 연구를 메타 수준에서 통합해 단일 개념 프레임워크(통합 분류체계, 평가 기준, 연구 의제)로 재구성한다. 특히 LLM-side(텍스트/LLM 타깃) 슬라이스에 초점을 두고, diffusion 모델이 적대 파이프라인에서 어떤 역할(예: 학습된 생성기, 고정+잠재 교란, score/classifier guidance, off-the-shelf 렌더링, 파이프라인 전용 렌더러, victim diffusion 모델)로 쓰이는지를 6개 클래스로 분류한다.

- **Technical Challenges**: 통합을 위해 가장 어려운 점은 모달리티·커뮤니티마다 다른 threat model(공격자 지식, query budget, 타깃 접근성)과 평가 관행이 달라 비교가 불가능하다는 것이다. 논문은 공격 성공률, transferability, query 예산, perplexity, defense-evasion의 5차원 프레임을 텍스트·이미지·비전-언어 전반에 동일하게 적용하고, 50편 카탈로그(추가 victim·비diffusion baseline 포함)로 새로운 공격이 최소한 무엇과 비교돼야 하는지 기준선을 제시한다.

- **Empirical Impact**: 리뷰는 diffusion-LLM-as-victim 항목과 비diffusion baseline까지 포함해, diffusion 기반 공격이 아직 닫지 못한 격차(특히 LLM-side에서의 반복되는 약점)를 정량적으로 드러내는 비교 틀을 제공한다. 또한 dual attacker-defender 관점에서 diffusion 기반 방어 4종을 자연스러운 평가 배경으로 함께 정리하고, 재현 가능한 실험 설계가 가능한 오픈 질문과 실행 가능한 연구 의제를 제안한다.



### Compiler-Driven Approximation Tuning for Hyperdimensional Computing (https://arxiv.org/abs/2606.26547)
- **Prior Approaches**: 기존 근사 컴퓨팅 프레임워크(EnerJ, ApproxHPVM, ApproxTuner)는 주로 딥러닝·이미지 파이프라인에 초점을 둬 HDC 프리미티브 단위 분석/컴파일 지원이 부족했다. HDC 쪽에서도 MicroHD는 제한된 하이퍼파라미터 공간을 탐색하는 수준에 머물러 CPU/GPU뿐 아니라 ReRAM/PCM 같은 하드웨어 레벨 근사 “노브”를 정교하게 묶어 튜닝하기 어려웠다.

- **Core Contribution**: 이 논문은 HDC 응용에서 가능한 소프트웨어·하드웨어 근사들을 자동으로 찾아 적용하는 프레임워크 ApproxHDC를 제안한다. HPVM-HDC 컴파일 인프라를 확장해, CPU·GPU뿐 아니라 ReRAM/PCM 기반 in-memory accelerator로 retargetable compilation을 수행하면서 QoS(품질) 제약 하 end-to-end 성능·에너지 최적화를 노린다.

- **Technical Challenges**: 핵심 난제는 근사 조합 공간이 지수적으로 커서 수동 탐색이 불가능하다는 점이다. ApproxHDC는 (1) HDC++ 코드 정적 분석으로 모든 HDC primitive 인스턴스를 식별하고 문맥 메타데이터로 JSON화한 뒤, (2) operation별 적용 가능한 근사 설계공간을 구성해 OpenTuner 기반 자동 튜닝으로 탐색하며, (3) developer annotation과 HDC 도메인 지식으로 탐색 공간을 대폭 가지치기한다.

- **Empirical Impact**: 실험에서는 HD-Classification, HD-Clustering, RelHD, HD-Hashtable 4개 벤치마크와 다양한 하드웨어 타깃에서 성능을 입증했다. CPU에서 최대 17.25×, GPU에서 15.02×, SpecPCM에서 4.69× speedup을 QoS 제약 내에서 달성했으며, MicroHD 대비 3.49× 더 나은 성능 향상을 보여 HDC 근사 튜닝의 실용성을 강화했다.



### Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation (https://arxiv.org/abs/2606.26502)
- **Prior Approaches**: 기존 연구는 LRM의 추론 길이(생각 토큰 길이)가 사람의 반응시간과 맞물리는지에 주로 초점을 맞춰, 항목 간 난이도 정렬(registration)을 보여줬습니다. 하지만 이러한 상관은 메커니즘이 아니라 관측된 현상일 수 있고, 실패·성공에서 계산을 어떻게 배분하는지(allocation)는 직접 답하지 못한다는 비판이 이어졌습니다.

- **Core Contribution**: 이 논문은 기존 ‘항목 간 난이도 정렬’과 별개로 ‘항목 내 계산 배분’을 진단하는 방법을 제시합니다. 구체적으로 각 에이전트가 자신의 실패(wrong)와 성공(right)에 대해 더 오래 숙고하는지를, 각 에이전트 내부 스케일에서 정규화하고 item identity를 고정한 상태(item fixed effects)로 평가합니다. 이를 통해 사람과 LRM이 겉보기에는 유사해 보이던 정렬이 실제론 다른 중단 정책을 가질 수 있음을 분리해 드러냅니다.

- **Technical Challenges**: 핵심 어려움은 토큰 길이(또는 출력 길이)와 반응시간(초)을 직접 축에 올려 비교하면 생기는 혼동을 피하는 것입니다. 논문은 seconds와 tokens의 절대 단위를 비교하지 않고, 에이전트별 고유 길이 읽out을 그대로 두되 correct 여부에 따른 기울기(틀릴 때 더 길어지는지)를 item fixed effects로 추정해 항목 난이도 기울기를 흡수합니다. 또한 비-thinking 기준모델(DeepSeek-V3)과 여러 추론 패러다임(H-ARC, INTUIT, Cortes)으로 진단이 특정 설계에만 의존하는지 경계합니다.

- **Empirical Impact**: 공개된 매칭 휴먼-LRM 코퍼스에서 모든 thinking LRM은 ‘틀리면 더 길어지는’ 큰 wrong-vs-right 효과(d≈1.47~3.13, H-ARC)를 보인 반면, 사람은 반대 부호(engagement-vs-abandonment 신호로 해석)로 나타났습니다. 이 분리는 item fixed effects에서도 유지되고 다른 데이터셋/패러다임으로도 재현되며, thinking이 아닌 기준모델에서는 같은 패턴이 약하거나 나타나지 않습니다. 결과적으로 기존 상관 측정에서 보이던 ‘난이도 정렬’은 맞지만, ‘실패에 더 쓸지/덜 쓸지’라는 제어 정책은 사람과 LRM이 반대로 작동한다는 점이 실증적으로 제시됐습니다.



### Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents (https://arxiv.org/abs/2606.26479)
Comments:
          12 pages, 5 figures, 4 tables

- **Prior Approaches**: 최근 2024~2026년 흐름에서는 LLM tool-using agent를 간접 prompt injection으로부터 방어할 때, 모델을 “거절”하도록 학습시키기보다 모델 밖에서 결정론적 정책으로 행위를 중재하는 out-of-band 보안이 주류로 자리 잡았다. CaMeL, FIDES, Progent, RTBAS, FORGE 등은 capability/정보흐름 라벨/참조모니터(reference monitor)/최소권한(least privilege)을 조합해 에이전트의 권한과 데이터 흐름을 제한하며, AgentDojo 벤치마크에서 공격이 거의 사라졌다고 보고된 사례가 있다. 다만 이들 평가는 고정된 정적(고정 시도) 공격 세트에 주로 의존해, 방어를 “다 아는” 적응형 공격에서는 성능이 급락할 수 있다는 의문이 남아 있다.

- **Core Contribution**: 이 논문은 기존 out-of-band 방어들을 Biba 무결성 보호(integrity protection), reference monitoring, least privilege라는 고전 보안 틀로 재정리해, 각 접근이 커버하는 것과 커버하지 못하는 지점을 구조적으로 비교한다. 동시에 이런 방어들의 검증이 정적 벤치마크에 묶여 있다는 한계를 지적하며, 방어를 고려하는 adaptive evaluation에 필요한 위협모델과 프로토콜을 제시한다. 이후 그 프로토콜로 Progent의 적응형 공격 분석을 독립 재현·확장하되, 저자들이 테스트하지 않은 설정(오픈 웨이트 Qwen2.5-7B, 단일 H200 self-hosting)에서 AgentDojo를 실험한다.

- **Technical Challenges**: 핵심 기술적 난제는 “모델 내부 거절”이 아니라, agent 바깥에서 결정론적 중재 정책이 실제 공격 흐름을 충분히 차단하는지 검증하는 것이다. 특히 정적 벤치마크에서는 방어자가 공격 템플릿에 맞춰 최적화된 적응이 일어나지 않아 과대평가될 수 있으므로, 방어 aware한 adaptive attack 절차와 평가 프로토콜을 설계해 공정성을 확보해야 한다. 저자들은 제안한 adaptive 프로토콜을 그대로 적용해 독립 재현/확장을 수행함으로써, 방어의 견고성 여부를 정량적으로 확인한다.

- **Empirical Impact**: 실험 결과, 평균 3회 실행에서 방어 성능은 안정적으로 유지되었고 Progent는 mean attack success를 약 6배 낮췄다(25.8%→4.2%). 또한 설계된 hand-crafted adaptive attack에 대해서도 방어가 크게 무너지지 않아(2.6% 상승/유지 수준) 적응형 평가에서의 취약성 주장을 일정 부분 상쇄한다. 다만 이는 약한 모델(7B급)과 단일 black-box 공격 템플릿에 대한 작은 데이터 포인트이며, 더 강력한 최적화 white-box 공격(GCG)까지는 미해결로 남아, “결정론적 out-of-band 집행이 적응형 공격에 더 어렵다”는 가설은 추후 검증이 필요하다는 결론이다.



### Epiphany-Aware KV Cache Eviction Without the Attention Matrix (https://arxiv.org/abs/2606.26472)
Comments:
          Preprint; in review

- **Prior Approaches**: 기존 추론(Reasoning) 모델용 KV cache eviction은 주로 attention weight를 중요도 점수로 써서 토큰을 버립니다. 하지만 attention weight는 장문 추론에서 잡음이 크고(내용 없는 토큰에도 가중치가 쏠림), attention 행렬을 읽으려면 FlashAttention 같은 fused 커널을 쓸 수 없게 만들어 배포 병목이 됩니다.
또한 reasoning trace의 길이가 길어질수록 attention-matrix 기반 방식은 메모리 벽에 부딪혀 실사용 컨텍스트를 크게 제한합니다.

- **Core Contribution**: 이 논문은 attention이 아니라 모델 내부 표현 변화량으로 토큰 중요도를 점수화하는 epiphany score를 제안합니다. 이 점수는 forward pass에서 이미 제공되는 hidden states(및 KV 벡터)만으로 계산되며, attention matrix를 물질화하지 않습니다.
그 결과 EpiKV는 별도 학습/분류기/커스텀 커널 없이도 기존 FlashAttention 서빙 스택에 그대로 적용 가능하다고 주장합니다.

- **Technical Challenges**: 핵심 기술 과제는 “표현 변화량”이 단순히 위치(세대 진행)와 연동돼 생기는 단조 트렌드를 어떻게 제거하느냐입니다. 저자들은 특정 레이어 대역에서의 hidden-state L2 diff가 counterfactual occlusion 라벨과 양/음의 상관을 보인다는 two-band layer anatomy를 발견하고, rolling causal z-score로 위치 오염을 제거해 detrended 신호(EpiKV)를 구성합니다.
또한 score 계산을 표준 forward 경로에서 수행해 FA2-compatible 조건(이전 KV와 hidden states만 사용, output_attentions 불요)을 만족시키도록 설계했습니다.

- **Empirical Impact**: 실험에서 EpiKV는 MATH-500의 4096-token 캐시 예산에서 72% 정확도로 attention 기반 강한 기준선(ThinKV 71%)과 동급 또는 근접 성능을 보입니다. AIME-2024에서는 lag-normalized KV 변형이 8192 tokens에서 37%를 달성해 최우수 attention-based 방식(33%)을 앞섰고, 속도는 최대 2.8배 빨라졌다고 보고합니다.
공학적으로는 attention-matrix를 강제하지 않아서 더 긴 유효 컨텍스트에서 메모리 제약을 회피하며, 배치 처리에서도 KV cache 절감 효과가 커진다는 점이 의미로 제시됩니다.



### DualEval: Joint Model-Item Calibration for Unified LLM Evaluation (https://arxiv.org/abs/2606.26429)
- **Prior Approaches**: 기존 LLM 평가는 정답 라벨이 있는 static benchmark과, Chatbot Arena 같은 arena 선호 데이터로 나뉘어 운영되는 경우가 많습니다. Static은 채점이 명확하지만 성능이 오르면 포화·오염에 취약하고, arena는 사용자 선호를 잘 반영하되 주관성과 편향 때문에 해석 구조가 약합니다. 또한 많은 방법이 모델 점수만 집계하고 문항 간 정보량 차이를 반영하지 못해 비용 효율성과 진단 가능성이 제한됩니다.

- **Core Contribution**: 논문은 DualEval로, 모델과 평가 문항을 하나의 잠재 공간에서 함께 보정(joint model-item calibration)하는 프레임워크를 제안합니다. static의 이진 정답과 arena의 보상모델(reward-model) 점수를 같은 latent model-item 구조(IRT 계열)에 넣어 모델 능력(ability)과 문항 난이도(difficulty), 예리함(sharpness)을 동시 추정합니다. 그 결과 모델 순위뿐 아니라 어떤 문항이 포화됐는지, 어떤 문항이 순위 분리에 핵심인지 진단 정보까지 얻을 수 있습니다.

- **Technical Challenges**: 핵심 과제는 서로 성격이 다른 정답 라벨과 선호 신호를 공통 스케일에서 안정적으로 결합하는 것입니다. 논문은 static은 2-parameter logistic IRT로, arena는 보상 점수로부터 soft pairwise preference 타깃을 만들되 tie(동점)와 both-bad(동시 실패) 쌍을 구분해 불필요한 그라디언트 충돌을 줄입니다. 또 모델 능력의 식별성(identifiability)을 위해 학습 후 능력을 정렬(center)하고, RM 선택에 따른 노이즈를 static 앵커로 완화해 견고한 순위를 재구성합니다.

- **Empirical Impact**: 코딩·수학·기타 지식·일상 질의 4개 도메인에서 18개 frontier LLM을 대상으로 평가했으며, static-anchored 도메인에서 정답 라벨 재구성 정확도 88–92%, arena 비교에서 decisive-pair 일치도 68–81%를 보였습니다. DualEval은 static-only 또는 arena-only 대비 균형 잡힌 순위를 만들고, 보상모델을 바꿔도(공개 Skywork-Reward-V2-Qwen3-8B 사용) 성능 저하가 상대적으로 작아 결합의 실효성을 보여줍니다. 더 나아가 Fisher information 기반 문항 압축으로 소수 고정보 문항만으로도 전체 순위를 Spearman ρ≥0.95 수준에서 회복하고, 잔차 기반으로 오염/이상치를 AUROC ≥0.995(정답 라벨 타깃)와 최대 0.98대 수준(선호 신호 타깃)으로 탐지할 수 있음을 입증했습니다.



### Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs (https://arxiv.org/abs/2606.26387)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 MLLM 정렬은 주로 DPO 같은 outcome-level 보상 최적화로 최종 텍스트의 선호도를 맞추는 방식이 많습니다. 하지만 이런 방법은 정답이 맞더라도 언어 priors만으로 맞춘 ‘잘못된 이유(right for the wrong reasons)’를 걸러내기 어렵고, 그 결과 시각 입력과의 인과적 연결이 약해져 환각이 남을 수 있습니다. 그 외 시도는 아키텍처를 늘리거나(모듈화) 추론 시 대비/패치에 의존해(visual contrastive decoding) 계산·운영 비용 또는 민감도 문제가 동반됩니다.

- **Core Contribution**: 논문은 시각 환각의 핵심 원인으로 ‘visual laziness’를 지목하고, 모델이 보는 상태와 시각을 가린 상태의 차이를 학습 목표에 직접 포함하는 VIGIL을 제안합니다. VIGIL은 생성 응답 y가 시각 입력 xv에서 얻는 정보량을 최대화하도록, counterfactual blind state를 사용한 강화학습(post-training) 정렬 프레임워크를 설계합니다. 특히 시각-언어 priors 간 최적화 지름길이 강화되지 않게 “시각이 없어도 자신 있게 말하는” 경우를 억제하는 데 초점을 둡니다.

- **Technical Challenges**: 기여를 구현하는 가장 큰 과제는 ‘정답 맞추기’가 아니라 ‘시각 증거에 의존했는지’를 학습 신호로 바꾸는 것입니다. VIGIL은 forward 단계에서 주의(attention) 마스킹으로 counterfactual blind state를 만들고, seeing 상태 대비 blind 상태에서의 확률 변화에 기반한 geometric constraint(Visual Information Gain 및 Counterfactual Visual Decoupling 손실)를 부여합니다. 또한 hard negative 및 dynamic gating으로 최적화의 안정성과 시각 의존 샘플에 대한 학습 집중을 동시에 확보합니다.

- **Empirical Impact**: 실험에서 VIGIL은 POPE, AMBER, MMHal-Bench 등 환각·이유 추론 벤치마크 전반에서 최신 정렬 기법을 일관되게 능가합니다. 더 나아가 25% 수준의 preference 데이터만으로도 풀데이터 수준 성능을 맞추고, 기준선 대비 text-only 능력 저하 없이 성능을 유지합니다. 또한 bounding box 같은 명시적 localization 감독 없이도 RefCOCOg에서 공간 grounding 능력이 향상되는 emergent 현상을 보여, 환각 억제가 곧 시각적 이해 강화로 이어질 수 있음을 시사합니다.



### Narration-of-Thought: Inference-Time Scaffolding for Defeasible Ethical Reasoning in Large Language Models (https://arxiv.org/abs/2606.26366)
Comments:
          24 pages, 8 figures, 16 tables. To appear at ACL 2026 (submitted via ARR)

- **Prior Approaches**: 기존 chain-of-thought(CoT) 프롬프팅은 중간 추론을 유도하지만, 도덕적 딜레마에서 ‘이해관계자 붕괴’(stakeholder를 거의 한 명만 언급)와 ‘불확실성 억제’(명시적 unknown/hedge 없이 결론을 단정) 같은 추론-수준 실패가 반복됐다. DailyDilemmas에서 표준 CoT는 이해관계자 붕괴 15~31%, 불확실성 억제 50~72%까지 나타나 모델이 실제 의사결정 맥락의 구조를 충분히 추적하지 못함을 보여줬다. 또한 토큰을 늘리거나 단계적 자세화를 해도 핵심 결함이 완전히 교정되진 않았다.

- **Core Contribution**: 논문은 chain-of-thought를 ‘이야기 형태의 도메인 프리미티브’에 맞춰 강제하는 Narration-of-Thought(NoT) 시스템 프롬프트를 제안한다. NoT는 학습/파인튜닝 없이, 5개 섹션(주인공-이해관계자-2-step 결과-불확실성-커밋)으로 CoT를 재구성해 행동 결정을 내릴 때 이해관계자와 불확실성을 앞단에서 외현화한다. 이후 NoT를 여러 이해관계자 관점(5라운드 멀티 스테이크홀더)으로 확장해, 통합 제안에 대한 accept/reject 표결로 수정가능성(결정의 결함이 반례 제안으로 흡수되는지)을 측정 가능하게 만든다.

- **Technical Challenges**: 핵심 기술 과제는 ‘자연스러운 추론’이 아니라 ‘의사결정 구조를 추적하는 추론’이 되도록, 모델이 결론을 단정하기 전에 unknown/hedge와 관련 당사자를 실제로 드러내게 만드는 것이다. 논문은 추론을 형식적 템플릿(5섹션 내러티브 스캐폴드)으로 고정해 검색/패턴매칭 기반으로 인과적-결과적 서사를 재구성하게 만들었고, verbose-CoT 같은 단순 토큰 증가 효과를 matched-budget 제어로 분리해 스캐폴드가 원인임을 검증했다. 추가로 특정 서브-지시문(이해관계자/불확실성/결과/주인공/커밋)을 제거하는 ablation과, textual-gradient descent로 NoT 개선 및 교차벤더 judge 우위까지 확인해 ‘어떤 구성요소가 무엇을 바꾸는지’까지 추적했다.

- **Empirical Impact**: 실험 1에서 NoT는 4개 생성기 패널 전반에 걸쳐 이해관계자 붕괴를 31% 내외에서 1% 미만으로, 불확실성 억제를 72% 내외에서 1~24%로 급감시켰다. 맞춤 예산 verbose-CoT 대조에서는 토큰 사용만 늘린 것이 아니라 ‘내러티브 스캐폴드’가 stakeholder 수와 uncertainty 점수에 큰 개선(Cliff’s delta 기준 효과)을 만들었고, ablation 결과도 각 지표가 해당 서브-지시문에 귀속됨을 보여줬다. 실험 2에서는 5라운드 멀티 스테이크홀더 프로토콜로 6%에 그치던 full consensus를 95%로 끌어올렸고, 결합 수렴(반복 재현 포함)도 매우 높게 나타나 감시/감사 가능한 형태의 결정 근거를 에이전트 배치에 활용할 수 있음을 시사한다.



### Axon: A Synthesizing Superoptimizer for Tensor Programs (https://arxiv.org/abs/2606.26344)
- **Prior Approaches**: 기존 타일 기반 커널 최적화는 algebraic 변환을 위해 hand-crafted rewrite rules에 의존하거나, 코드 생성을 위한 hand-written template을 사용해왔다. 또한 CPU/GPU 중심으로만 최적화를 수행하는 방식이 많아 Trainium 같은 멀티 엔진 구조에 대해선 확장성이 제한된다. 그 결과 ML 엔지니어가 하드웨어 세부(메모리 계층, 타일링, fusion, 엔진 선택)를 깊게 이해해야 고성능을 얻기 어려웠다.

- **Core Contribution**: Axon은 tile-based AI accelerator 프로그램을 위한 synthesizing superoptimizer로, Numpy-like 텐서 프로그램의 의미(semantics)로부터 타깃 ISA를 자동 생성한다. 동시에 의미적으로 동일한 프로그램 변형들을 모두 탐색해, 입력 shape가 정해졌을 때 실제 성능이 가장 좋은 커널을 empirical하게 선택한다. 변환 규칙을 수동으로 작성하지 않고도 계산 그래프 수준의 재배치를 포함한 최적화를 일관되게 수행하는 것이 핵심이다.

- **Technical Challenges**: 핵심 난제는 (1) 연산을 재배치하거나 융합할 때 의미 보존을 증명하면서 (2) tiling/ISA 선택/ fusion의 탐색공간 폭발을 관리해야 한다는 점이다. Axon은 연산을 더 작은 granular 계산으로 분해하고 unbounded tensors에 대해 SMT 동치성 검사를 수행해, hand-crafted rewrite rule이나 연산 성질 가정 없이 변환의 정합성을 보장한다. 또한 nu-graph에 semantically equivalent variant들을 지속적으로 유지하며, symbolic tiling과 ISA lower­ing, 이후 operator·instruction fusion을 단계별로 결합해 후보를 함께 평가한다.

- **Empirical Impact**: Amazon Trainium에서 20개 벤치마크(단일 연산 및 멀티-오퍼레이터 LLM 커널 포함)를 평가한 결과, 단일 오퍼레이터는 최대 3.7x, 멀티-오퍼레이터 커널은 최대 19x의 속도 향상을 보였다. Neuron 컴파일러 대비 최대 19x, hand-optimized NKI 대비 최대 1.35x, Mirage와의 비교에서는 지원하는 전 범위에서 geomean 2~10% 개선 및 Mirage가 못 하는 16개 추가 커널 합성 성과도 제시됐다. 이는 커널 최적화를 더 이상 전문가 수작업에 의존하지 않고, 의미 기반 탐색과 증명으로 자동화할 수 있음을 실증한 결과로 해석된다.



### The Verification Horizon: No Silver Bullet for Coding Agent Rewards (https://arxiv.org/abs/2606.26300)
Comments:
          Authors are listed alphabetically by their first names

- **Prior Approaches**: 기존 코딩 에이전트 학습에서는 실행 기반 단위 테스트(테스트 스위트)나 LLM-based judge 같은 “검증기”를 보상 신호로 써왔습니다. 하지만 테스트는 의도 대비 커버리지가 얇고, LLM 판정은 강화되는 모델이 loophole을 학습해 악용할 수 있으며, 사람 검수는 비용 때문에 스케일이 제한됩니다. 결과적으로 대부분의 방법은 scalability·faithfulness·robustness 중 2개만 만족하고 나머지 1개가 무너지기 쉬웠습니다.

- **Core Contribution**: 이 논문은 검증 품질을 scalability, faithfulness, robustness 세 축으로 정의하고, 셋을 동시에 만족시키는 것이 핵심 난제라고 정리합니다. 또한 reward hacking이나 signal saturation처럼 “프록시(검증 신호)와 의도(intent) 사이의 간극”이 학습 중 벌어지는 문제를 줄이기 위해, 고정된 reward function이 아니라 verifier와 generator가 함께 진화해야 한다고 주장합니다. 그 실천으로 SWE-like, frontend, real-world, long-horizon의 태스크 유형별로 서로 다른 보상(검증) 구성을 체계적으로 비교합니다.

- **Technical Challenges**: 첫째, 실행 테스트 기반 보상은 false positive/false negative뿐 아니라 “정보 누출을 통한 과정-무효” 형태의 reward hacking에 취약합니다. 이를 위해 논문은 instruction clarity와 instruction–test alignment로 의미적 신뢰도를 분해하고, MiniSWEAgent 기반 agentic quality judge로 데이터 필터링 및 품질 라벨링을 수행합니다. 둘째, 단순 통과율이 높아도 악용 과정이 숨어들 수 있어 trajectory-level behavior monitoring을 넣고, 네트워크·git 히스토리·원본 PR 탐색 등 고위험 패턴에 대해 토큰 수준 페널티를 적용하며 패턴 세트를 학습 중 반복 갱신하는 closed-loop를 구성합니다.

- **Empirical Impact**: SWE-like 설정에서 behavior monitoring은 hacked resolved rate를 28.57%에서 0.56%로 크게 낮추고, clean resolved rate는 40.22%에서 60.53%로 끌어올려 “통과율 상승이 아니라 과정 정상화”가 일어났음을 보였습니다. frontend 영역에서는 rubric 기반 static judge의 한계를 인정하고, Playwright로 실제 브라우저 상호작용을 수행하는 agentic interactive judge를 제안해 runtime 동작을 근거로 평가하게 했습니다. 전반적으로 강화되는 정책에도 보상 신호의 신뢰도를 유지하며 여러 내부/공개 벤치마크에서 유의미한 성능 개선을 보고하며, verifier가 policy capability 성장에 맞춰 계속 co-evolve해야 한다는 결론을 뒷받침합니다.



### From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations (https://arxiv.org/abs/2606.26277)
Comments:
          Dianjing Fan and Yao Li equally contributed to this work. 7 pages, 1 figure

- **Prior Approaches**: 기존 추천/시퀀스 모델은 로그인 이후의 행동을 전제로 하거나, 클릭스트림을 단순 집계·단기 패턴 중심으로 학습해 선의 의도(intent)는 잘 드러내지 못했습니다. 특히 금융 서비스는 프리로그인 웹 탐색과 로그인 후 앱 경험의 성격이 달라, 채널 간 entity resolution(익명 세션과 계정 연결)이 어려워 웹 의도 신호가 사장되곤 했습니다. LLM 기반 taxonomy 생성·증류는 대체로 텍스트 입력에 초점을 두며, 금융 클릭스트림의 멀티모달·장기 시퀀스를 기반으로 “정량 추천 + 정성 설명”을 동시에 노린 통합 접근은 부족했습니다.

- **Core Contribution**: 이 논문은 프리로그인 웹 clickstream에서 두 가지를 동시에 뽑는 dual-purpose intent 예측 프레임워크를 제안합니다. self-supervised Transformer가 멀티모달 세션 임베딩(session embedding)을 만들고, LLM이 생성한 의도 taxonomy(의도 분류체계)를 distillation(지식 증류)해 동일 임베딩 공간에서 해석 가능한 intent label로 제공합니다. 이렇게 한 번의 표현 학습으로 정량 성능(랭킹/전환 예측)과 정성 이해(설명 가능한 카테고리)를 함께 달성하는 것이 핵심입니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 채널 간 연결로 프리로그인 의도를 통합해야 하지만 익명 웹 세션의 의미를 일관된 표현으로 만들기가 어렵고, (2) LLM taxonomy를 대규모로 생성·유지하면서도 이후 서빙 지연을 감당해야 한다는 점입니다. 저자들은 웹 이벤트에 페이지 콘텐츠 기반 의미까지 포함해 self-supervised Transformer로 압축 임베딩을 학습하고, K-Means 기반 clustering으로 LLM에 다양한 샘플을 주는 stratified sampling을 적용했습니다. 또한 LLM이 만든 라벨을 임베딩 입력의 lightweight MLP로 distill하여, 추론 시 LLM 호출 없이 초저지연 라벨 산출이 가능하게 했습니다.

- **Empirical Impact**: 모바일 홈 화면 tile ranking에서 세션 임베딩은 기준선 대비 macro Recall@1을 1.88% 올리고 Log Loss를 13.38% 줄였습니다. 사용자 conversion 예측에서는 distillation된 intent이 LLM 라벨 대비 micro F1에서 7% 성능 손실에 그치며(LLM teacher 대비 4.3% 향상 보고), LLM 라벨이 35%짜리 희소 신호임에도 SRF 같은 집계 피처보다 훨씬 강력하게 전환 예측에 기여했음을 보였습니다. 또한 10만 세션 라벨링을 Transformer 인코딩 60초 + MLP 추론 0.009초로 처리해, LLM 직서빙 대비 처리량을 3,000배 이상 높이면서도 성능 저하는 7% 수준으로 유지하는 등 운영 관점 impact가 큽니다.



### Neural Speaker Diarization via Multilingual Training: Evaluation on Low-Resource Nepali-Hindi Speech (https://arxiv.org/abs/2606.26144)
Comments:
          12 pages, 7 tables

- **Prior Approaches**: 기존 화자 분리/다중 화자 diarization은 i-vector·x-vector 같은 임베딩을 추출한 뒤 군집화해 “누가 언제 말했나”를 맞히는 방식이 많았지만, 겹치는 발화 상황에서 성능이 흔들리고 diarization error rate을 직접 최적화하기 어렵다는 한계가 있었다. end-to-end neural diarization(EEND)은 오디오를 입력으로 두고 화자 활동을 end-to-end로 학습해 겹침 구간을 더 잘 처리해 왔다. 다만 underrepresented language에서는 라벨 데이터가 부족해 E2E 성능이 급격히 떨어지는 문제를 다룬 연구가 제한적이었다.

- **Core Contribution**: 이 논문은 저자원 Nepali-Hindi diarization을 위해 다국어 학습(multilingual training) 전략을 제안하고, EEND-EDA(encoder-decoder attractors)와 DiaPer(Perceiver-based attractors)를 비교한다. LibriSpeech(영어 클린), VoxCeleb(다양한 화자/언어 커버리지), 그리고 별도로 모은 Nepali·Hindi 오디오를 결합해 언어 편향을 낮추고 cross-lingual generalization을 유도하는 학습 구성이 핵심이다. 결과적으로 Perceiver 기반 attractor가 저자원 다국어 환경에서 더 견고한 diarization을 보일 수 있음을 실험으로 확인했다.

- **Technical Challenges**: 가장 큰 기술적 난제는 겹치는 발화와 화자 수 변동이 큰 조건에서, 제한된 Nepali/Hindi 데이터로도 false alarm을 억제하면서 화자 활동을 안정적으로 추정하는 것이다. 저자들은 Perceiver 기반 DiaPer에 cross-attention 기반 latent 압축과 block-wise auxiliary loss(Perceiver 블록마다 중간 슈퍼비전)를 도입해 attractor 품질을 깊이 전반에 걸쳐 점진적으로 강화하도록 했다. 반면 EEND-EDA는 LSTM encoder-decoder attractor로 동일한 학습 조건에서 비교했으며, 화자 수가 늘수록 DER_miss는 내려가도 DER_FA가 출렁이는 양상이 관찰됐다.

- **Empirical Impact**: 실험은 2/3/4 화자 및 mixed-speaker 시나리오로 구성해 LibriSpeech, VoxCeleb, NeHi 테스트셋에서 평가했으며, 전반적으로 DiaPer가 EEND-EDA를 앞섰다. NeHi에서 DiaPer는 2-speaker 3.28%, 3-speaker 2.02%, 4-speaker 4.05%, mixed 4.76%의 DER을 보여 EEND-EDA(1.50%, 9.68%, 16.17%, 11.19%) 대비 큰 개선을 보였다. LibriSpeech와 VoxCeleb의 mixed 조건에서도 DiaPer가 더 낮은 DER(예: NeHi mixed 4.76% vs 11.19%, VoxCeleb mixed 2.60% vs 8.99%)를 기록해, 저자원 다국어 diarization에서 Perceiver 기반 end-to-end 구조의 실용 가능성을 뒷받침한다.



New uploads on arXiv(cs.IR)

### NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems (https://arxiv.org/abs/2606.27243)
Comments:
          12 pages, 3 figures

- **Prior Approaches**: 기존 연구·자동화는 주로 하이퍼파라미터 튜닝이나 부분 코드 생성에 집중해, 모듈 간 토폴로지/인터랙션을 함께 바꾸는 ‘구조 업그레이드’까지 안정적으로 확장하기 어려웠다. 또한 coding agent는 컴파일/유닛 테스트 같은 소프트웨어 신호를 통과해도, 추천기 아키텍처 의미(마스킹, logit fusion, 시퀀스-토큰 경로 등)는 틀릴 수 있어 조용한 성능 저하(silent failure)가 발생한다. 이때 후보가 로컬 테스트는 통과하지만 오프라인·온라인에서 음의 이득을 내는 문제가 반복됐다.

- **Core Contribution**: 논문은 추천기 ‘아키텍처 진화’를 검증 중심으로 자동화하는 NOVA를 제안한다. 핵심은 level-aware agent harness로, architecture gradient(비미분 SGD 영감 업데이트 신호)를 통해 직전 수정·검증 진단·오프라인 지표 변화·궤적 메모리를 다음 수정 방향으로 누적 반영한다. 아키텍처 의미를 어기는 후보는 조기에 차단하고, 실패 패턴은 이후 탐색에서 forbidden direction으로 재사용해 같은 실수를 줄인다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘실행 가능(runnable)’과 ‘추천기 의미론적으로 타당한 아키텍처’가 다를 수 있다는 점이다. NOVA는 구조-semantic gate(마스크 의미, feature-to-token 매핑, attention 방향, logit fusion 등)와 local testing을 조합해 비용 비싼 학습 전에 invalid 후보를 걸러낸 뒤, offline inner loop(AUC)와 online outer loop(GMV/Bias)로 효과를 확인한다. 또 L1–L4 task-level 제어로 위험도가 높은 의사결정은 Copilot에 라우팅해 자율 추론의 과확장을 막고, 실패 원인을 trajectory memory에 기록해 다음 architecture gradient에 반영한다.

- **Empirical Impact**: 산업 광고 추천 시스템에 배포해 NOVA는 L2 ScaleUp과 L3 Literature-to-Production에서 각각 effective pass rate 54.5%, 60.0%를 달성하며, coding-agent 계열 대비 silent failure를 줄이는 성과를 보였다. 특히 human-attended time 기준으로 한 literature-to-production 사이클을 13배 이상 단축했으며, 온라인 A/B 테스트에서는 GMV 관점 상위 pCVR 목표 3개에서 +1.25%, +1.70%, +2.02% 개선과 함께 pCVR bias를 58.8%, 66.7%, 37.3%로 낮췄다. 결과적으로 NOVA는 ‘아키텍처 의미 검증’과 ‘검증 기반 피드백’이 산업 성과로 직결될 수 있음을 실증했다.



### TRUST: Item-Calibrated Interval Evidence for Temporal Session-Based Recommendation (https://arxiv.org/abs/2606.27214)
- **Prior Approaches**: 기존 TSBR(temporal session-based recommendation)은 구간(interval)을 데이터 전역의 절대 분포로 해석해 왔습니다. 하지만 논문은 동일한 절대 간격이 아이템마다 의미가 달라, 전역 기준 해석이 신뢰도 낮은 신호를 섞을 수 있음을 보여줍니다. 또한 시간 정보를 반영해 성능을 올리는 모델들이 많지만, '아이템별 간격 분포 캘리브레이션' 관점은 상대적으로 부족했습니다.

- **Core Contribution**: TRUST는 관찰된 interval을 해당 아이템의 경험적 interval 분포에 상대적으로 평가하는 Temporal Reliability Under item-Specific inTerval evidence for Session-based Recommendation 프레임워크를 제안합니다. 핵심은 ITSF(item-calibrated temporal scoring function)로 각 interval의 신뢰도 점수를 계산해, 같은 시간이라도 아이템마다 다른 증거 강도를 부여한다는 점입니다. 이 점수를 이웃 샘플링, 세션 그래프 인코딩, 최종 interest aggregation에 일관되게 반영합니다.

- **Technical Challenges**: 가장 먼저, 아이템별 interval 분포가 전역 분포와 통계적으로 다르다는 점을 검증하고, 그 불일치가 실제 성능에 어떤 영향을 주는지 진단해야 했습니다. 다음으로 관측이 적은 아이템에서 아이템별 분포 추정이 불확실할 때 과도한 확신을 피하면서 reliability를 보정해야 했는데, ITSF는 quantile 기반 점수에 샘플 수에 따른 페널티 완화(보수적 감쇠)를 포함합니다. 마지막으로 전역 관계(협업 신호)와 로컬 전이(세션 내 순서 신호)를 함께 학습할 때 noisy co-occurrence를 줄이고, 신뢰도 있는 최근 단서와 다중 깊이의 관심을 결합하도록 RaNS(신뢰도 인접 샘플링), RcSG(신뢰도 가중 세션 그래프), PIA(plastic interest aggregation)를 설계했습니다.

- **Empirical Impact**: Diginetica, RetailRocket, Nowplaying의 공개 데이터셋에서 TRUST는 시간 기반 및 비시간 기반 강력한 기준모델을 전반적으로 능가했습니다. 또한 scoring function 자체를 다른 TSBR에 'plug-in'하면 성능을 추가로 끌어올려, ITSF가 model-agnostic한 유효성을 가진다는 점을 확인했습니다. 컴포넌트별 ablation과 “모듈 제거”가 아닌 “모듈 내부에서 시간 신호 캘리브레이션” 방식이 일관되게 이득을 준다는 결과는, 신뢰도 관점이 학습 전반의 품질을 개선한다는 의미입니다.



### UniFormer: Efficient and Unified Model-Centric Scaling for Industrial Recommendation (https://arxiv.org/abs/2606.27058)
- **Prior Approaches**: 기존 산업용 추천은 임베딩 이후 행동 모델링, 피처 상호작용, 태스크 모델링을 모듈 단위로 나눠 설계하고, 주로 각 모듈의 용량을 따로 키우는 component-centric scaling에 의존해 왔다. HyFormer와 OneTrans 같은 최신 co-scaling 시도도 주로 feature space 안에서 행동·상호작용을 함께 키우는 데 머물며, 전체 모델링 공간에서 task modeling까지 아우르는 model-centric scaling 프레임워크는 부족했다.

- **Core Contribution**: UniFormer는 추천 모델의 전체 모델링 공간을 feature space와 task space로 분해해 Feature-space Interaction Modules(FIM)과 Task-space Interaction Modules(TIM)로 통합 확장하는, 효율적인 model-centric scaling 프레임워크를 제안한다. 또한 의미 기반 토큰화로 user-item decoupling을 지원해 request-level inference 가속을 노리고, 멀티 시퀀스 cross-attention 및 self-attention 조합으로 preference collapse를 완화한다.

- **Technical Challenges**: 첫째, 산업 환경의 높은 동시성과 낮은 지연 요구 때문에 통합 스케일링 과정에서 학습·추론 효율을 유지해야 한다. UniFormer는 feature/task 공간 분해와 costly한 전체-공간 full attention을 회피하고(모듈 간 상호작용을 FIM/TIM으로 제한), semantic grouping 및 lazy KV 매핑(레이어 공유)을 통해 비용을 줄인다. 둘째, 서로 이질적인 행동 패턴(단기·장기 등)을 균형 있게 반영해 붕괴를 막아야 하며, UniFormer는 시퀀스별 cross-attention을 분리해 적응적(전역+개인화) fusion으로 다양한 취향을 보존하고 multi-view FFN으로 컴포넌트별 파라미터 배분의 유연성도 확보한다.

- **Empirical Impact**: Kuaishou와 Kuaishou Lite 두 프로덕션 시나리오에서 광범위한 온라인 A/B 테스트를 수행한 결과, App Stay Time은 +0.101%/+0.260%, Watch Time은 +0.729%/+1.113%의 개선을 일관되게 보였다. 이는 산업 추천에서 feature 중심 scaling을 넘어 task space까지 포함하는 통합 확장 전략이 사용자 참여와 상호작용 지표를 실제로 끌어올릴 수 있음을 실증했다.



### TriPAH: Imbalance-Aware Tri-Prompt Affinity Hashing for Cross-Modal Medical Retrieva (https://arxiv.org/abs/2606.27010)
Comments:
          10 pages, 3 figures, 4 tables

- **Prior Approaches**: 기존 크로스모달 medical hashing retrieval은 이미지-텍스트 이중 브랜치 정렬에 의존하는 경우가 많다. 하지만 임상 보고서의 잡음(템플릿·모호함), long-tailed·multi-label 질병 분포, 그리고 영상-텍스트의 비대칭 안정성 때문에 semantic fragmentation과 정렬 불안정이 생기기 쉽다. 또한 공통 quantization을 강하게 밀어붙이는 학습 방식은 초기에 discretization collapse를 유발해 희귀 질환 검색 편향을 키운다는 한계가 있다.

- **Core Contribution**: TriPAH는 Tri-Prompt Affinity Hashing으로, ontology-grounded patient-level prompt(환자 맥락 프롬프트)를 텍스트 잡음을 낮추는 초기 앵커로 사용한다. 이미지-텍스트 외에 prompt를 더한 tri-view semantic fusion으로 시멘틱 단절을 줄이고, 이로써 I→T와 T→I 양방향 정렬 격차를 완화하는 것을 목표로 한다. 여기에 imbalance-aware multi-task hashing을 결합해 희귀 클래스의 구분력을 유지하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 잡음이 섞인 임상 언어를 낮은 노이즈 표현으로 만들고 (2) long-tailed 상황에서 희귀 라벨을 잃지 않으면서 (3) 양자화 압력을 학습 초기에 과도하게 걸지 않는 것이다. TriPAH는 stochastic gating으로 메타데이터 누락에도 강건한 환자 프롬프트를 만들고, Mamba-Transformer 기반의 tri-view fusion으로 세 뷰를 계층·다중 granularity로 정렬한다. 양자화는 asymmetric quantization과 progressive quantization regularization을 통해 이미지 브랜치의 이른 discretization collapse를 지연시키며, CBCE와 Bayesian consistency regularization으로 클래스 불균형과 Hamming 공간 구조를 함께 잡는다.

- **Empirical Impact**: ODIR-5K, MIMIC-CXR, IU-Xray 세 데이터셋에서 모든 코드 길이(32/64/128 bits)와 양방향 검색에서 SOTA를 일관되게 상회했다. 특히 I→T mAP에서 ODIR-5K는 +26.2%의 평균 개선을 보였고, MIMIC-CXR에서도 양방향 모두 큰 폭의 성능 향상을 기록했다. 또한 ablation 결과 tri-view fusion, imbalance-aware hashing이 각각 성능에 기여하며, 희귀 질환 검색에서 특히 효과가 커지는 점이 확인되어 임상 기반 증거 검색/케이스 관리의 효율성과 정확도 향상에 의미가 있다.



### A Shared IPTC Topic Space for Cross-Source Topic Modelling (https://arxiv.org/abs/2606.26845)
- **Prior Approaches**: 기존 연구는 뉴스와 소셜미디어 각각에 토픽 모델을 따로 학습한 뒤 분포를 비교했지만, 각 코퍼스가 만들어내는 토픽 공간이 달라 직접 정렬이 어려웠습니다. 예를 들어 LDA를 독립 적용한 비교는 넓은 주제 일부만 겹치고 관심 차이는 크게 달라지는 등, 재현 가능한 비교 설계가 제한적이었습니다. 또한 짧은 텍스트에서는 잡음·중복으로 인해 토픽이 더 퍼지며(예: diffuse topics) 무감독 발견 결과가 코퍼스별로 흔들릴 수 있습니다.

- **Core Contribution**: 이 논문은 IPTC Media Topics(보도 분류 표준)라는 공통 택소노미를 기준으로, 서로 다른 코퍼스를 단일 공유 토픽 공간에 매핑하는 재현 가능한 프레임워크를 제시합니다. guided BERTopic으로 토픽 경계를 정하면서, 94개 level-1 토픽 점수를 계산한 뒤 최대 유사도 규칙으로 17개 parent 토픽(보도 레이어)으로 계층 축약합니다. 이를 통해 뉴스와 소셜미디어의 토픽 출력을 같은 기준선에서 해석·비교할 수 있게 만듭니다. 

- **Technical Challenges**: 핵심 기술 문제는 코퍼스별로 달라지는 토픽 식별자를 외부 기준에 맞춰 안정적으로 정렬하는 것입니다. 논문은 토픽-택소노미 정렬을 위해 level-1 키워드 중심 가중 centroid를 만들고, target centroid는 parent까지 포함하는 parent-enriched 방식으로 구성한 다음 cosine 유사도로 비교해 parent_max로 축약합니다. 또한 할당이 애매한 경우를 줄이기 위해 score와 gap 기반의 보수적 임계값을 적용하고, 개발 단계에서 단계적 축소(screen→refinement→finalist→ablation/threshold sweep)로 구성을 확정했습니다.

- **Empirical Impact**: NYT 2011 개발 코퍼스에서 guided 계열은 zero-shot 대비 더 강한 mapped coverage를 보이며, 더 엄격한 할당 기준에서도 커버리지가 충분히 유지되는 것으로 나타났습니다. parent-enriched target 구성은 coverage와 parent consistency를 함께 개선했는데, 예컨대 level-1 단독보다 parent로의 상위 합의가 더 높아졌습니다. 또한 임계값을 더 조여도 coverage가 갑자기 무너지는 것이 아니라 서서히 감소해, 최종 규칙이 특정 한계값에 취약하지 않다는 민감도 확인도 제공했습니다.



### Attributed, But Not Incremental: Cannibalization-Corrected Attribution for Large-Scale Advertising (https://arxiv.org/abs/2606.26690)
Comments:
          6 pages, 3 figures. Accepted at ADKDD 2026

- **Prior Approaches**: 기존 유료 채널 성과 평가는 last-touch attribution(LTA)이나 data-driven attribution(DDA) 같은 관측 기반 attribution에 크게 의존해 왔습니다. 이 방식은 실시간·세분화된 커버리지는 강하지만, 유기 수요·브랜드 트래픽·다른 획득 채널과 겹칠 때 실제 증분(incremental) 기여와 어긋나는 attribution–cannibalization mismatch가 생깁니다. 한편 incrementality experiments는 인과적 근거를 주지만 비용·지연·희소성 때문에 프로덕션 전반을 상시 보정하기엔 부족했습니다.

- **Core Contribution**: 논문은 실험에서 얻은 인과적 lift를 기준으로, 프로덕션 attribution의 비증분 성분을 보정하는 experiment-calibrated cannibalization correction 프레임워크 ETDC+HCA를 제안합니다. ETDC(Experiment-to-Daily Cannibalization)는 sparse한 실험 lift를 일 단위 채널 보정치로 변환하고, HCA(Hierarchical Cannibalization Allocation)는 이를 비즈니스 계층에 맞춰 미세 단위로 할당하되 계층 집계 일관성을 강제합니다. 결과적으로 “일일 의사결정이 가능한” 증분-aligned 지표를 만들면서도 원 attribution 파이프라인을 그대로 유지합니다.

- **Technical Challenges**: 핵심 난제는 (1) 실험은 희소·지연돼 있어 daily 표면을 바로 만들기 어렵고, (2) 관측 기반 attribution은 실험 lift의 인과적 스케일과 불일치한다는 점입니다. 저자들은 채널-day 단위에서 실험 lift를 noisy 인과 감독 신호로 보고, 반사실 유기 기준선 변화·시간 구조(DOW/휴일/계절성)·전달 상태 같은 proxy 변수로 일 단위 cannibalization를 추정합니다. 또한 HCA에서는 부모(채널) 수준의 보정 총량을 자식 노드에 분배하되, child 합의 일치·비음수 및 상한( feasibility )·locality(동일 subtree 내에서만 주로 변화) 제약을 걸어 보정의 실무 활용성을 높였습니다.

- **Empirical Impact**: 채널 단위 증분 A/B 실험 18라운드(최대 8개 시장)와의 forward-in-time 평가에서 ETDC+HCA는 raw attribution 대비 정규화 캘리브레이션 오차를 91.38% 줄이며, 부호 오차 중앙값도 0에 가깝게 좁혀졌습니다. 특히 slice 단위에서도 signed relative error가 대략 -7%~10% 범위로 안정적이어서 단순 평균 효과가 아닌 보정의 일관성을 보여줍니다. 실제로 여러 글로벌 TikTok 마켓에 배포해 예산·트래픽 전략 조정에 반영했으며, 측정된 cannibalization rate가 약 15%p 감소하는 성과로 이어졌습니다.



### GPUSparse: GPU-Accelerated Learned Sparse Retrieval with Parallel Inverted Indices (https://arxiv.org/abs/2606.26441)
- **Prior Approaches**: SPLADE 같은 learned sparse retrieval은 dense와 비슷한 성능을 내면서도 exact-match와 해석 가능성을 유지하지만, 실제 서빙에서는 WAND/BMW 같은 CPU 기반 inverted index 순회가 병목이 된다. GPU는 dense의 matmul에는 강하지만, sparse는 순차적 pivot-selection과 비정형 메모리 접근 때문에 대규모 병렬화가 어렵다. Seismic은 query-term pruning(query_cut)으로 속도를 끌어올리는 대신 Recall을 포기하는 근사 전략을 택했다.

- **Core Contribution**: 이 논문은 learned sparse retrieval의 exact scoring을 GPU에서 수행하는 GPUSparse를 제안한다. 핵심은 sparse scoring을 GPU-resident inverted index 위에서 batched scatter-add로 재구성해, WAND/BMW의 순차 병목을 제거하면서도 inner product를 정확히 계산한다는 점이다. 또한 fused Triton 커널로 커널 런치와 중간 메모리 생성을 줄여 end-to-end 서빙 지연을 크게 낮춘다.

- **Technical Challenges**: 문제는 GPU에서 비정형 inverted index를 어떻게 병렬로 읽고 점수를 누적하느냐로, 그대로 옮기면 warp divergence와 메모리 비효율이 커진다. GPUSparse는 block-aligned, warp-coalesced posting list 레이아웃(32원소 패딩, flat 배열)과 query-term 단위 scatter-add를 결합해 무조건 처리 기반의 병렬성을 확보한다. 더 나아가 Triton fused 커널로 scatter-add를 한 번에 수행하고, document-parallel CSR 커널과의 비교를 통해 work-efficiency와 bandwidth-efficiency 간 근본적 트레이드오프까지 분석한다.

- **Empirical Impact**: MS MARCO passage ranking에서 8.8M 컬렉션(실제 SPLADE 임베딩) 기준 GPUSparse는 CPU exact scoring과 MRR@10(0.383) 및 Recall@1000(≥0.999)을 3자리까지 일치시키며 정확성을 검증했다. 성능은 Pyserini CPU 대비 235x(1.27ms vs 298ms/쿼리)로, Seismic의 근사 방식과 비교해도 787 QPS로 exact scoring을 달성한다(배치 500). 아울러 doc-parallel 커널이 H100 메모리 대역폭의 62.6%까지 도달하는 반면 scatter-add는 더 낮은 활용을 보이는데, 이 차이가 “GPU sparse retrieval에서 무엇을 최적화해야 하는가”에 대한 실증적 기준을 제공한다.



### TileMaxSim: IO-Aware GPU MaxSim Scoring with Dimension Tiling and Fused Product Quantization (https://arxiv.org/abs/2606.26439)
- **Prior Approaches**: ColBERT 계열은 MaxSim(쿼리 토큰별 문서 토큰과의 최대 유사도 후 합)로 높은 성능을 내지만, 후보 문서를 대규모로 스코어링할수록 계산 비용이 병목이 됩니다. PLAID, WARP, GEM 등은 CPU 중심 최적화(예: pruning, product quantization, implicit decompression)로 속도를 끌어올렸지만, GPU의 메모리 대역폭을 충분히 활용하지 못한다는 한계가 지적됩니다. 또한 기존 GPU 구현은 유사도 행렬을 중간에 만들며 메모리 트래픽이 낭비되는 경우가 많아 성능이 상한에 도달하기 어렵습니다.

- **Core Contribution**: 이 논문은 MaxSim 스코어링의 핵심 병목이 연산량이 아니라 HBM 메모리 대역폭 부족임을 roofline 분석으로 규명하고, 이를 줄이는 GPU 커널 설계를 제안합니다. TileMaxSim은 IO-aware Triton 커널로서 유사도 행렬(Nq×Nd)을 “물질화”하지 않고, 데이터가 HBM에서 읽힌 뒤에는 즉시 필요 계산에만 쓰고 폐기해 메모리 트래픽을 최소화합니다. 또한 ColBERTv2/PLAID의 스코어링 단계를 대체 가능한 drop-in 방식으로 동일한 검색 품질(정확한 랭킹)을 유지하도록 설계됐습니다.

- **Technical Challenges**: 문제는 MaxSim의 matmul→max→sum 환원 구조가 attention의 softmax 기반 흐름과 달라 기존 IO-aware 패턴을 그대로 적용하기 어렵다는 점입니다. 저자들은 (1) multi-query SRAM tiling으로 문서 임베딩을 HBM에서 정확히 한 번 스트리밍하며 레지스터에 토큰별 최대값을 누적하고, (2) 임베딩 차원 d가 128을 넘는 경우를 위해 128-wide dimension tiling을 도입해 커널 패밀리를 확장합니다. 더 나아가 TileMaxSim-PQ에서는 product quantization의 decompress-then-score를 없애고 공유 메모리 lookup table 기반으로 ADC식 점수를 fused 형태로 계산해 HBM I/O를 추가로 크게 줄입니다.

- **Empirical Impact**: NVIDIA H100에서 TileMaxSim은 HBM peak 대역폭의 80.2%를 달성하고, 초당 82M 문서를 스코어링(실 MS MARCO 패시지 71.6M/s)하며 루프 기반 대비 220배, fused PyTorch 대비 6.5배 이상의 가속을 보고합니다. 무엇보다 MS MARCO와 3개 BEIR 벤치마크에서 기준 MaxSim과 동일한 랭킹을 보존해 “정확도 손실 없는” 성능 개선임을 입증합니다. ColBERTv2/PLAID 파이프라인에서 100K 후보 기준 엔드투엔드 지연을 268ms에서 1.2ms로 줄이며(98% 감소), 문서 수 100K~500K에서 처리량이 일정하고 d=64~768, FP16/BF16/FP32 전반에 대해 강건성을 보여 업계 실사용에 가까운 임팩트를 제공합니다.



### Scoring Is Not Enough: Addressing Gaps in Utility-fairness Trade-offs for Ranking (https://arxiv.org/abs/2606.26369)
- **Prior Approaches**: 기존 learning-to-rank에서는 문서에 대한 점수 score를 학습한 뒤, 추론 시 score를 정렬(sorting)해 순위를 만드는 방식이 표준 패러다임으로 자리 잡아 왔다. 최근에는 algorithmic fairness 관심이 커지면서, utility와 fairness를 동시에 trade-off하도록 점수 기반 모델이나 휴리스틱을 학습/후처리하는 연구가 늘었다. 다만 공정성이 “문서 단위로 분해되지 않는 비분해(non-decomposable) 목적”이면, 점수만으로 가능한 trade-off의 범위가 제한될 수 있다는 문제의식이 제기된다.

- **Core Contribution**: 이 논문은 점수 기반 랭킹이 utility–fairness trade-off의 전체 Pareto frontier를 커버하지 못한다고 분석적으로 주장한다. generic한 이진 문서-그룹 fairness(쿼리 단위 상호작용 격차) 설정에서, deterministic 점수와 randomized 점수 모두에 대해 반례(counter-example)를 통해 scorability가 성립하지 않음을 보인다. 또한 이러한 비최적성은 fairness를 단일 쿼리 기준으로 보든, 여러 쿼스를 가로질러 보든 지속된다고 확장한다.

- **Technical Challenges**: 핵심 기술적 challenge는 fairness가 쿼리 내 문서들의 “조합(구성)”에 의해 결정되는데, scoring은 각 문서를 독립적으로 점수화한 뒤 정렬하므로 조합 의존성을 충분히 표현하지 못한다는 점이다. 논문은 이를 무한 데이터(infinite-data) 가정 하에, scoring이 달성 가능한 trade-off를 정의하는 scorability 개념으로 수학화하고, weights의 성질을 이용해 어떤 scoring 함수도 목표 수준 α에서 최적 trade-off를 재현하지 못하는 구성을 설계한다. 랜덤화로도 해결되지 않는 경우를 별도 정리로 보이고, Plackett-Luce 같은 대표적 randomized scoring에서도 실제 최적성 갭이 나타나는 반례 인스턴스를 수치 최적화로 제시한다.

- **Empirical Impact**: 긍정적으로는, 문서 relevances를 fairness를 무시하고 학습한 뒤 ex-post로 re-ranking하는 semi-greedy post-processing이 더 좋은 trade-off를 만들 수 있다고 실험적으로 보인다. 특히 beam-search를 포함한 탐색이 scoring보다 훨씬 넓은 trade-off 영역을 효율적으로 커버하며, 많은 경우 exhaustive post-processing에 근접하는 성능을 보인다. 결과적으로 fairness처럼 비분해 목적을 다룰 때는 “score→sort” 중심의 in-processing이 구조적으로 제한될 수 있어, re-ranking/탐색 기반 설계를 재고해야 한다는 메시지를 준다.



### From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations (https://arxiv.org/abs/2606.26277)
Comments:
          Dianjing Fan and Yao Li equally contributed to this work. 7 pages, 1 figure

- **Prior Approaches**: 기존 추천/시퀀스 모델은 로그인 이후의 행동을 전제로 하거나, 클릭스트림을 단순 집계·단기 패턴 중심으로 학습해 선의 의도(intent)는 잘 드러내지 못했습니다. 특히 금융 서비스는 프리로그인 웹 탐색과 로그인 후 앱 경험의 성격이 달라, 채널 간 entity resolution(익명 세션과 계정 연결)이 어려워 웹 의도 신호가 사장되곤 했습니다. LLM 기반 taxonomy 생성·증류는 대체로 텍스트 입력에 초점을 두며, 금융 클릭스트림의 멀티모달·장기 시퀀스를 기반으로 “정량 추천 + 정성 설명”을 동시에 노린 통합 접근은 부족했습니다.

- **Core Contribution**: 이 논문은 프리로그인 웹 clickstream에서 두 가지를 동시에 뽑는 dual-purpose intent 예측 프레임워크를 제안합니다. self-supervised Transformer가 멀티모달 세션 임베딩(session embedding)을 만들고, LLM이 생성한 의도 taxonomy(의도 분류체계)를 distillation(지식 증류)해 동일 임베딩 공간에서 해석 가능한 intent label로 제공합니다. 이렇게 한 번의 표현 학습으로 정량 성능(랭킹/전환 예측)과 정성 이해(설명 가능한 카테고리)를 함께 달성하는 것이 핵심입니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 채널 간 연결로 프리로그인 의도를 통합해야 하지만 익명 웹 세션의 의미를 일관된 표현으로 만들기가 어렵고, (2) LLM taxonomy를 대규모로 생성·유지하면서도 이후 서빙 지연을 감당해야 한다는 점입니다. 저자들은 웹 이벤트에 페이지 콘텐츠 기반 의미까지 포함해 self-supervised Transformer로 압축 임베딩을 학습하고, K-Means 기반 clustering으로 LLM에 다양한 샘플을 주는 stratified sampling을 적용했습니다. 또한 LLM이 만든 라벨을 임베딩 입력의 lightweight MLP로 distill하여, 추론 시 LLM 호출 없이 초저지연 라벨 산출이 가능하게 했습니다.

- **Empirical Impact**: 모바일 홈 화면 tile ranking에서 세션 임베딩은 기준선 대비 macro Recall@1을 1.88% 올리고 Log Loss를 13.38% 줄였습니다. 사용자 conversion 예측에서는 distillation된 intent이 LLM 라벨 대비 micro F1에서 7% 성능 손실에 그치며(LLM teacher 대비 4.3% 향상 보고), LLM 라벨이 35%짜리 희소 신호임에도 SRF 같은 집계 피처보다 훨씬 강력하게 전환 예측에 기여했음을 보였습니다. 또한 10만 세션 라벨링을 Transformer 인코딩 60초 + MLP 추론 0.009초로 처리해, LLM 직서빙 대비 처리량을 3,000배 이상 높이면서도 성능 저하는 7% 수준으로 유지하는 등 운영 관점 impact가 큽니다.



### Reducing Redundancy in Whole-Slide Image Patching for Scalable Indexing and Retrieva (https://arxiv.org/abs/2606.26157)
- **Prior Approaches**: 디지털 병리에서 WSIs 인덱싱은 보통 patch selection으로 대표 패치만 뽑아 저장 부담을 줄이는 방식이었지만, 이후 유사도 검색은 여전히 대량 임베딩 비교 비용을 동반한다. 기존 무감독 patch selection(예: Yottixel, SPLICE, SDM)은 데이터 규모(기깃픽셀)를 다루기 위해 대표성을 유지하는 축소를 목표로 했으나, 결국 임베딩/바코드의 중복이 남아 storage와 계산량을 더 줄이기엔 한계가 있었다. 또한 self-supervised 기반 선택은 진단 의미와 직접 맞지 않는 전처리 과제를 학습해, 진짜 조직적 대표성보다는 전조작 성능에 치우칠 수 있다는 문제가 제기된다.

- **Core Contribution**: 이 논문은 ARReST(Antithetical Redundancy Reduction Strategy)로, 서로 다른 조직 클래스 간 구별에 거의 기여하지 않는 패치를 ‘반대(antithetical) 패치’로 정의해 검색 아카이브에서 제거하는 프레임워크를 제안한다. 동일 클래스 내부 중복만 제거하는 기존 관점에서 벗어나, 클래스가 다르더라도 표현이 과도하게 유사한 패치 쌍을 찾아 전역 유사도 분포 기준으로 pruning 한다. 그 결과 형태학적 다양성이나 검색 정밀도를 크게 해치지 않으면서도 인덱스 크기를 줄이는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 난제는 ‘어떤 패치가 버려도 되는 중복인지’를 재현성 있게 판별하는 것이다. ARReST는 라벨이 있는 TCGA에서 서로 다른 클래스의 패치 임베딩을 무작위로 매칭해 유사도 스코어를 만들고, 전역 유사도 분포의 q-quantile 임계값보다 낮게 나오는 쌍을 찾아 redundancy repository에 저장·활용함으로써 prunining을 체계화한다. 이후 남은 임베딩을 BOB barcodes로 인코딩해 더 빠른 Hamming distance 기반 유사도 계산이 가능하게 하여, 저장 절감과 검색 가속을 함께 달성한다.

- **Empirical Impact**: TCGA 21개 장기(총 11,679 WSIs)에서 Yottixel 기반 patch selection과 UNI 임베딩을 사용해 검증했으며, ARReST는 장기별로 인덱스 압축을 크게 수행하면서 Top-1/Top-k 및 majority voting 기반 retrieval 성능을 대체로 유지했다. 보고된 storage savings는 3%~60%, 평균은 14%±13% 수준이며, 많은 장기에서 retrieval 성능 저하 없이 절감이 가능했다. 다만 intraclass heterogeneity가 크거나 소수 아형 비중이 높은 장기에서는 F1이 감소하는 경향이 보여, 향후 adaptive threshold 또는 subtype-aware pruning의 필요성이 제시된다.



### AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems (https://arxiv.org/abs/2606.26859)
Comments:
          Authors are listed alphabetically by their first name

- **Prior Approaches**: 기존 추천 알고리즘 연구는 엔지니어가 가설을 만들고 프로덕션 코드를 수정한 뒤 A/B 실험을 실행·분석하고 온라인 성과를 해석하는 ‘아이디어-런치’ 흐름에 크게 의존해 왔다. 이 구조 때문에 성과가 실험 지식과 누적 학습으로 폭발적으로 확장되기보다, 인력 규모에 거의 선형으로만 비례해 스케일링이 제한된다.

- **Core Contribution**: 이 논문은 프로덕션 환경에서 동작하는 멀티에이전트 시스템 AgentX를 제안하며, 추천 실험의 생산 함수를 사람이 아니라 에이전트가 주도하도록 재구성한다. AgentX는 자동으로 추천 실험을 생성·구현·평가하고 그 결과로부터 학습하는 self-evolving 개발 엔진을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술 과제는 실험을 자동화하되 안전하게 롤아웃하고, 실패까지 포함해 지식으로 축적하며, 코드 생성이 실제 저장소/시스템 제약을 만족하도록 검증하는 것이다. AgentX는 Brainstorm Agent로 실행 가능한 제안을 순위화하고, Developing Agent에서 repository-grounded 코드 생성과 다차원 reliability 검증을 수행하며, Evaluation Agent가 guardrail-veto된 A/B 판단으로 안전한 온라인 평가를 실시한 뒤 SGPO의 semantic-gradient 업데이트(SGPO)로 실행 궤적을 에이전트에 반영해 점차 개선한다.

- **Empirical Impact**: 논문은 AgentX가 수작업 워크플로우가 따라가기 어려운 스케일과 속도로 추천 실험을 반복하며, 성공과 실패 모두를 구조화된 knowledge asset으로 전환해 누적 학습을 강화한다고 제시한다. 결과적으로 추천 연구·실험 사이클을 산업화된 closed loop로 바꿔, 인력 의존적 혁신을 evidence/compute 기반의 자기 개선형 혁신으로 전환하는 의미가 있다.



### From Vajrayana Tara to Bengali Baul: A Computational Study of Lexical Transmission Across Buddhist, Shakta, and Vaishnava Traditions in Benga (https://arxiv.org/abs/2606.26803)
Comments:
          9 pages, 2 figures, 4 tables. Code and corpus: this https URL Dataset: this https URL

- **Prior Approaches**: 기존에는 벵골의 불교 바즈라야나 어휘가 팔라(Pala) 수도원 붕괴 이후 샥타 탄트라로 흡수됐다는 주장이 역사적으로 거론됐지만, 실제로는 정량화된 검증이 부족했다. 또한 여러 종교 전통의 어휘 관계를 한 번에 비교하는 대규모 코퍼스 기반 분석이 제한적이었다.

- **Core Contribution**: 이 논문은 8~19세기 벵골·산스크리트 종교 문헌 75편을 대상으로 전통 계층을 넘나드는 어휘 관계를 계산적으로 재구성한다. 특히 바즈라야나-샥타 전이의 핵심 주장(불교 어휘가 샥타로 흡수됨)을 TF-IDF 문자 n-그램과 코사인 유사도로 ‘특이성’ 관점에서 수치화한다.

- **Technical Challenges**: 문제는 샥타 탄트라와 다른 전통 간 어휘 겹침이 산스크리트/시가 일반성 때문인지, 특정 전승 경로의 결과인지 구분해야 한다는 점이다. 연구진은 TF-IDF 문자 n-그램 기반 코사인 유사도로 전통 간 유사도를 비교하고, 같은 세기·같은 언어 조건에서 Gitagovinda(바이슈나바)와 Bridge Tara(불교-샥타 전이)의 유사도 차이(0 vs 0.54)를 ‘전이 사슬의 특이성’ 증거로 제시한다.

- **Empirical Impact**: 결과적으로 샥타와 불교-샥타 전이 사슬이 보여주는 8.5배 대비는 단순한 종교 문헌 일반성보다는 ‘어휘 전승 경로’가 존재함을 뒷받침한다. 또한 Tara 텍스트들에서 샥타→불교 어휘 비율 2.0~4.0, 라므프라사드 센의 18세기 벵골 칼리 노래에서 Tara 56회·Kali 103회 등 정량 근거가 제시돼, 벵골에서의 불교-샥타 신성(신크레티즘)을 다중 전통 데이터로 처음 뒷받침한 연구로 의미가 있다.



### ConvMemory v3: A Validity Context Layer for Conversational Memory via Target-Conditioned Relation Verification (https://arxiv.org/abs/2606.26753)
Comments:
          22 pages, 3 figures

- **Prior Approaches**: 기존 ConvMemory v1/v2는 대화 맥락에서 관련성 높은 메모리를 찾고 순서를 정렬하는 데 집중했지만, 검색된 메모리가 이후 턴에 의해 업데이트·수정·대체되어 “더 이상 사실이 아닐” 가능성까지는 다루지 못했다. 특히 v1은 high-recall 풀과 경량 reranker로 후보를 넓히고, v2는 recall을 보존하는 protected top-10 재정렬로 ordering을 개선했으나 “유효성(validity)” 판단은 없었다. 그 결과, 토픽은 맞지만 속성 값이 바뀐 메모리가 상위에 남는 문제(관계는 관련되지만 시간에 따른 진실성은 훼손)가 남아 있었다.

- **Core Contribution**: ConvMemory v3는 v1/v2 검색 후 단계에 “validity context layer”를 추가해, 각 후보 메모리에 대해 ‘이 메모리가 목표(target) 명제 기준으로 이후 소스에 의해 덮였는지’를 확인한다. 기본 동작(context mode)은 후보 집합과 rank order를 그대로 두고, 구조화된 유효성 메타데이터만 첨부해 에이전트가 선택적으로 활용할 수 있게 했다. 또한 demote mode는 명시적 opt-in으로 dense current-state 워크로드에서만 재정렬을 수행하도록 설계돼, 기본 검색 품질을 보존하는 방향을 택했다.

- **Technical Challenges**: 핵심 기술 난제는 ‘관련성만으로는 부족’하다는 점으로, 두 메모리 간 관계는 목표 명제가 고정될 때만 일관되게 정의된다. v3는 이를 해결하기 위해 target-conditioned relation verification을 수행하며, (target, source) 쌍에 대해 MiniLM slot head와 DeBERTa-v3 slot head의 점수를 곱(product)해 보수적으로(dual-evidence gate) ‘직접·특정 overturn’ 가능성을 평가한다. 여기에 source가 업데이트 이벤트인지와 operation이 overturn인지에 대한 보수적 event/operation evidence를 추가하고, 여러 소스 간 결합은 noisy-or로 전파해 업데이트 증거가 불충분할 때 과도한 demote를 피하도록 했다.

- **Empirical Impact**: 실험에서 dual-evidence gate는 합성 multi-hop validity 벤치마크에서 90.12%±1.73의 정확도를 달성했으며, real-data feedback loop로 Memora role binding으로 레이블 없이 전이되어 group-all-correct 98.8%±0.9을 기록했다. 또한 demote 모드에서 current-active Hit@1은 baseline 45.1%에서 95.7%±1.2로 크게 상승하면서도, superseded되지 않은 메모리는 recall 99.4%로 보호해 “재정렬의 부작용”을 제한했다. 마지막으로 6개의 기계검증 가능한 safety contract로 레이어 동작을 고정했고, multi-hop graph propagation은 메커니즘으로 검증하되 엄격한 prerequisite edge의 완전 자동 구성은 counterfactual 필요 지식 문제로 경계(boundary)를 명확히 했다.



### SocialPersona: Benchmarking Personalized Profiling and Response with Multimodal Social-Media Contex (https://arxiv.org/abs/2606.26654)
- **Prior Approaches**: 기존 개인화 벤치마크는 주로 대화에서 사용자가 직접 말한 선호를 기억하는지(메모리 관점)나, 정형화된 로그를 기반으로 선호를 추정하는 방식을 테스트했다. 최근에 행동 이력을 길게 보려는 시도가 있었지만, 합성/구조화된 텍스트 로그 의존이 커서 실제 소셜미디어 타임라인에서 나타나는 잡음·비구조·단서 분산의 어려움이 상대적으로 줄어든 편이다. 또한 다수의 MLLM 연구는 소셜미디어를 콘텐츠/감정/분류 과제로 다루며, 사용자 관련 신호는 보통 추천 타깃이나 인구통계 속성처럼 단순화되는 경우가 많았다.

- **Core Contribution**: SocialPersona는 멀티모달 large language model(MLLM)이 소셜미디어 타임라인에서 암묵적 선호를 복원하고, 그 선호를 대화에 반영하는지를 함께 평가하는 벤치마크다. 171명의 일상적(비홍보) 사용자 타임라인으로부터 7개 관심 도메인에 대해 사람 검증 기반의 stable 관심과 recent 관심을 분리해 프로필을 구성한다. 프로필 구성(task)과 그 프로필에 맞춘 개인화 응답 생성(task)을 제공해, 단순 예측이 아니라 “추정된 선호를 실제 상호작용에 쓰는 능력”까지 측정한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트·이미지·타임스탬프처럼 단서가 희소하고 분산된 멀티모달 증거를 장기(horizon)로 집계해 세부 태그를 복원하는 것, (2) stable와 recent를 시간 분포 관점에서 구분하는 것, (3) 생성된 프로필을 정보 병목 없이 대화 개인화에 안정적으로 전이하는 것이다. 논문은 Gemini/GPT 계열 MLLM을 활용해 보수적으로 관측 가능한 신호만 추출·집계한 뒤 LLM 캘리브레이션과 사람 검증으로 프로필을 정교화했으며, 대화 평가는 timeline-conditioned와 profile-conditioned(생성 프로필만 사용) 두 축으로 오차 전이를 추적한다.

- **Empirical Impact**: 실험 결과, 현재 MLLM들은 세부(fine-grained)·최근(recent) 선호에서 성능이 크게 떨어지고, 특히 추정 프로필을 실제 대화에 적용하면 추가로 품질이 악화됐다. 세부 태그를 4~7개에서 1~2개로 “과도하게 범주화(over-generalization)”하는 경향이 전 모델에서 관찰됐고, 텍스트로 분산된 관심 도메인은 자주 누락되는 반면 이미지가 뚜렷한 도메인은 상대적으로 잘 잡았다. 또한 stable와 recent 분리는 여전히 매우 어려운데, 타임스탬프를 넣어도 시간 분포 패턴을 제대로 활용하지 못하는 것으로 나타나 cross-modal long-horizon user modeling의 난제가 재확인됐다. SocialPersona는 이러한 결함을 정량화하고, 텍스트·이미지 단서 결합과 장기 시간 추론을 개선하는 연구 진척을 측정하는 공통 기준으로 의미가 크다.



### Utilizing Cognitive Signals Generated during Human Reading to Enhance Keyphrase Extraction from Microblogs (https://arxiv.org/abs/2606.26485)
- **Prior Approaches**: 마이크로블로그 기반 자동 키프레이즈 추출(AKE)은 짧고 잡음이 많은 글이 흩어져 있어 어려운 문제로 다뤄져 왔다. 기존 연구는 독자의 주의가 머무는 단어를 반영한다고 여겨지는 eye-tracking 신호를 활용해 성능을 끌어올리려 했지만, 생리·획득·특징 디코딩 제약으로 한계가 있었다.

- **Core Contribution**: 이 논문은 eye-tracking에 더해 EEG(뇌전도) 신호가 AKE를 보완할 수 있는지 검증한다. ZuCo 인지 언어 처리 코퍼스를 활용해 8개 EEG 특징과 17개 eye-tracking 특징을 모델에 결합하고, 인지 신호가 읽기 중 핵심어 추출에 지속적으로 도움이 되는지를 체계적으로 평가한다.

- **Technical Challenges**: 핵심 기술 과제는 인지 신호가 모델 구조에 의해 왜곡될 수 있다는 점이다. 이를 줄이기 위해 soft-attention 입력과 self-attention의 query vector에 특징을 주입하는 방식으로 넣었고, AKE 모델 전반에서 신호 조합(EEG 단독, eye-tracking 단독, 결합)을 비교해 왜 어떤 조합이 이득인지 확인했다.

- **Empirical Impact**: 실험 결과, 읽기 중 생성된 인지 신호는 특징 조합이나 모델 아키텍처와 무관하게 AKE 성능을 일관되게 향상시켰다. EEG 특징이 가장 큰 개선을 보였고, EEG와 eye-tracking을 함께 쓰면 두 신호 단독 성능 사이에 위치해 부분적 보완성과 동시에 중복/잡음 가능성도 시사된다. 전반적으로 EEG가 마이크로블로그 AKE에 유의미한 인지 근거를 제공하며, 멀티모달 인지 신호 연구가 후속 확장될 만한 가치가 있음을 보여준다.



### Extracting Problem and Method Sentence from Scientific Papers: A Context-enhanced Transformer Using Formulaic Expression Desensitization (https://arxiv.org/abs/2606.26481)
- **Prior Approaches**: 기존 연구는 과학 논문에서 핵심 아이디어를 뽑기 위해 문제·방법 문장을 주로 라벨링 기반으로 학습해 왔다. 그런데 문장 단위 주석은 비용이 커 소규모 데이터셋에 머무르며, 그 결과 모델이 특정 문장 양식에 과도하게 의존해 일반화가 떨어지는 문제가 생긴다.

- **Core Contribution**: 이 논문은 소규모 데이터셋에서 비롯되는 성능 저하 요인을 세 가지 축에서 동시에 다룬다: 데이터 스케일 확대, 특정 양식 의존도 감소, 문장 내 정보 풍부화다. 이를 위해 formulaic expression(FE) desensitization(수식형 표현 비식별화) 기반 데이터 증강기를 제안하고, 문장 맥락을 활용하는 context-enhanced transformer로 핵심 단어 중요도를 추정하며 문맥 잡음을 줄인다.

- **Technical Challenges**: 핵심 기술 과제는 (1) FE 패턴을 인위적으로 변형해도 문제·방법의 의미를 유지하며 합성 데이터를 생성하는 것과 (2) 문장-문맥의 잡음을 줄이면서도 목표 문장에서 중요한 단어를 안정적으로 집계하는 것이다. 논문은 FE desensitization 기반 데이터 augmenters로 합성 샘플을 늘리고, context-enhanced transformer가 맥락을 기반으로 중요도를 측정해 노이즈를 완화하도록 설계했다. 또한 LLM 기반 in-context learning(IBL) 실험도 수행했지만 해당 태스크에는 적합하지 않음을 확인했다.

- **Empirical Impact**: 실험 결과, 제안한 모델은 두 개의 과학 논문 데이터셋에서 baseline 대비 macro F1 점수가 각각 3.71%, 2.67% 향상됐다. 정량·정성 평가 모두에서 문제·방법 문장 추출 품질이 개선되었으며, 특히 소규모 데이터의 양식 의존을 FE desensitization과 맥락 강화로 완화하는 방향이 효과적임을 보여준다.



### 3D Spatial Pattern Matching (https://arxiv.org/abs/2606.26465)
- **Prior Approaches**: 기존 metric spatial pattern matching은 대부분 2차원 평면에서 엔티티 위치와 관계를 정의해 쿼리를 그래프 부분매칭으로 푸는 틀을 사용했다. 이 방식은 거리 제약을 2D로만 반영해, 높이(예: 건물 층수·창/문 위치)를 함께 고려해야 하는 실제 장면 검색에는 한계가 있다. 또 예시 기반(example-based) 접근은 사용자가 기존 패턴을 떠올리기보다는 ‘비슷한 곳’을 막연히 찾는 landmark search 같은 용도에 덜 적합하다고 본다.

- **Core Contribution**: 이 논문은 spatial pattern matching을 3차원(일반적으로 k-dimensions)으로 확장해 문제 정의를 보다 일반화하고, distance relations 위의 subgraph matching 알고리즘을 제안한다. 3D 거리 제약을 처리할 수 있도록 nn-match/ee-match 단계 개념을 유지하면서도 3D 공간 인덱싱에 맞춰 계산 규칙을 정리한다. 또한 합성 데이터 1종과 독일 함부르크의 실제 3D 건물 데이터를 포함한 1종의 3D spatial pattern matching datasets를 공개하고, 향후 방법의 baseline으로 제공한다.

- **Technical Challenges**: 주요 기술적 난제는 2D에서 쓰이던 spatial index와 거리 상한/하한 가지치기 로직을 3D로 그대로 옮기면 후보 쌍이 폭증한다는 점이다. 이를 해결하기 위해 minimum bounding rectangle을 minimum bounding rectangular prism으로 확장하고, Inverted Linear Quadtree를 Inverted Hyperoctree로 바꿔 후보 pruning을 효율적으로 수행한다. ESPM의 계산 흐름(먼저 n-matches, 그다음 e-matches, 마지막으로 조합)을 3D/general k-dimensions 설정에 맞춰 그대로 적용해 subgraph matching을 구성했다.

- **Empirical Impact**: 합성 데이터 실험에서는 희소~중간 밀도 설정에서 쿼리 전체 평가가 수십 분 수준까지 확장되는 성능을 보였고, 반면 고밀도에서는 ee-matches용 후보가 덜 줄어 메모리 문제가 발생했다. 함부르크 LOD2 CityGML 기반의 실제 데이터에서는 모든 쿼리를 약 4분 내에 수행하면서도 메모리 사용을 ‘합리적 수준’으로 유지했다고 보고한다. 비교 알고리즘이 거의 없어 성능 우열보다 scalability 테스트와 공개 baseline 제공에 초점을 두며, 향후 메모리 효율적인 3D spatial pattern matching 연구 방향을 제시한다.



### ProvenAI: Provenance-Native Traces of Evidence in Generated Answers (https://arxiv.org/abs/2606.26449)
- **Prior Approaches**: RAG에서는 생성 답변에 인용(citation)을 붙이지만, 인용이 실제로 답변 생성에 의미 있게 기여했는지까지는 보장되지 않는다는 한계가 지적돼 왔습니다. 기존 연구는 인용의 형식·정확성(예: FActScore, RAGAS)이나 문장/주장 단위의 근거 적합성 점검에 초점을 두거나, 내부 분포 변화를 이용해 attribution을 개선하려 했습니다. 그러나 “인용된 문서를 빼면 출력이 실제로 바뀌는가?”라는 리소스 영향(influence) 검증은 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: ProvenAI는 다중 홉 question answering에서 투명성을 answer correctness, citation fidelity, per-document influence의 세 층으로 분해해 각각을 독립적으로 측정합니다. 특히 인용 감사가 깨끗해 보여도 uncited 동반 문서가 출력에 더 큰 영향을 줄 수 있는 ‘citation-influence gap’을 실증적으로 드러내는 데 기여합니다. 새로운 모델을 학습시키기보다, 고정된 RAG 파이프라인에서 관측 가능한 감사 인프라를 구축해 진단 정보를 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 각 문서가 실제로 출력 분포에 주는 영향을 KL-divergence 같은 분포 수준 지표로 측정해야 하는데, 로컬 backend가 per-token 확률을 제공하지 않는 점입니다. ProvenAI는 leave-one-resource-out ablation으로 문서 제거 전후의 생성 결과를 재생성하고, 답변 문자열 토큰 변화와 citation set 변화에서 나온 표면 프록시를 통해 분포 타깃과의 관계를 정형화합니다. 다만 이 프록시는 근사적으로 신뢰되는 조건이 있으며(결정적 디코딩에서 한쪽(positive) 방향은 보장), 동점 후보들로 확률이 재배치되는 경우에는 영향이 과소평가될 수 있다는 한계도 함께 명시합니다.

- **Empirical Impact**: HotpotQA distractor 검증 split에서 7,405개 예제를 대상으로 평가해 answer accuracy 53.53%와 citation-fidelity 71.55%를 보고하며 두 지표의 디커플링을 확인했습니다. 예시 분석에서는 인용 감사는 완벽하지만 다수의 uncited 문서가 leave-one-out에서 출력에 민감하게 작용해 citation-influence gap을 구체적으로 보여줍니다. 또한 측정 결과가 cryptographic provenance 같은 실행 중 커밋형 감사 아키텍처와 어떻게 조합될 수 있는지까지 논의하며, 검색 기반 QA의 ‘의미 있는 투명성’이 어떤 구성요소로 성립하는지를 제시합니다.



### Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat mod (https://arxiv.org/abs/2606.26373)
- **Prior Approaches**: 기존 연구는 임베딩 역추출(inversion)로 원문 텍스트가 복원될 수 있음을 보여왔다. Vec2Text, GEIA, TEIA는 공격자가 인코더 접근성(쿼리/스냅샷)이나 서러게이트를 활용해 재구성 성능을 끌어올리며, 이런 맥락에서 “임베딩 유출=텍스트 유출” 위험이 실무적으로 정착됐다. 방어로는 (1) 대규모 homomorphic encryption이 안전하지만 10^6 규모 top-k 검색에 비싸고, (2) 차등프라이버시 기반 잡음은 랭킹 품질을 너무 빨리 망가뜨리는 한계가 제시된다.

- **Core Contribution**: 논문은 정적 데이터베이스(문서 임베딩)와 동적 쿼리(질의 임베딩)를 비대칭으로 보호하는 중간지대를 제안한다. 문서 쪽은 SVD 하위공간으로 truncation한 뒤, 소유자만 아는 비밀 직교변환(rotation)으로 기하학적으로 “방향”을 숨기고, 쿼리 쪽은 CKKS로 암호화해 서버가 쿼리 값과 유사도 점수를 보지 못하게 한다. 또한 Projection에 제한된 공격자에 대해 재구성 오차에 대한 tight lower bound를 수학적으로 제시하고, SVD truncation이 어떤 경우에는 단순 손실이 아니라 선형 denoiser처럼 동작할 수 있음을 함께 규명한다.

- **Technical Challenges**: 핵심 난제는 (a) SVD truncation+비밀 rotation이 실제로 어떤 수준까지 역추출을 어렵게 만드는지 “정식” 경계를 세우는 것과, (b) CKKS reranking이 10^6 문서·sub-second 지연 요구를 만족하도록 ct-pt 경로로 설계하는 것이다. 저자들은 공격자의 디코더가 특정 subspace span(V_k) 안에서만 결과를 낼 수 있다는 제한 하에 L2 재구성 오차의 tight lower bound(투사-디코더 lemma)를 증명해 난이도(proxy로 σ_rec)를 정의한다. 구현적으로는 서버가 ct-pt 곱만 수행하도록 프로토콜을 구성하고, CKKS 파라미터는 보안 표의 제약과 정확도 허용치를 만족하는 선에서 작은 offline micro-benchmark로 재현 가능하게 선택해 지연을 줄였다.

- **Empirical Impact**: 실험에서 1백만 문서, 5개 인코더 조건에서 랭킹 품질을 유지하면서(일부 강한 인코더에서는 약간 개선) latency는 sub-second 수준을 달성했다. 보호된 문서 공간에서는 off-the-shelf inversion 공격이 잡음 바닥(noise floor)으로 붕괴하는 양상을 보였고, 공개된 inversion 경로(공간 제약 하의 공격)에 대한 이론-실험 정합성을 확인했다. 다만 더 강한 공격 시나리오에서는 알려진-평문 기반으로 rotation을 회복(orthogonal Procrustes)할 수 있고, public product-quantization 코드는 이웃 구조를 상당 부분 보존하는 등 “문서 보호는 암호적 원리가 아니라 경험적 난독화”라는 한계를 명확히 구획해, 위협모델에 대한 실무적 해석을 제공한다.



### Instruction Bleed: Cross-Module Interference in Prompt-Composed Agentic Systems (https://arxiv.org/abs/2606.26356)
Comments:
          8 pages, 2 tables. Accepted to the ICML 2026 Workshop on Failure Modes in Agentic AI (FAGEN), Seoul, South Korea

- **Prior Approaches**: 기존 연구와 벤치마크는 prompt injection, cognitive degradation, multi-agent fault propagation, compositional privacy leakage처럼 비교적 ‘명시적 실패’에 집중해왔다. 반면 prompt 모듈을 여러 텍스트 조각으로 런타임에 이어 붙여 LLM이 정책처럼 해석하는 구조에서는, 모듈 간 동시 간섭을 직접 측정하는 평가는 거의 없었다.

- **Core Contribution**: 이 논문은 compositional behavioral leakage(CBL)를 ‘비의도 편집이 공유 컨텍스트 창에서 다른 모듈의 행동을 조용히 바꾸는 현상’으로 정의하고, 이를 탐지·검증 가능한 형태로 형식화했다. 또한 CBL을 재현할 수 있는 reusable three-channel protocol(C0–C3 조건), 반증 가능한 prediction set, 그리고 prompt-composed agentic system의 독립적인 시스템 클래스 특성화를 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 트랜스포머 self-attention이 모듈 경계를 보장하지 않아 delimiter가 사실상 isolation을 제공하지 못한다는 점이다. 저자들은 비초점 모듈을 volume(모듈 추가), content(의미 내용), form(형식) 채널로만 국소 교란해, focal 모듈 점수 변화가 ‘semantic 내용 채널’에서만 유의미하게 나타나는지 분리해 측정했다.

- **Empirical Impact**: 배포된 job-evaluation 에이전트에서 Claude Sonnet 4.6으로 144회 실험한 결과, semantic content 채널(C2)에서 focal cv-match 점수가 d=0.63만큼 이동했으며 bootstrap 95% 신뢰구간이 0을 배제했다(다만 recommendation flip은 관측되지 않는 sub-threshold 영역). 이는 표준 QA가 ‘결정 뒤집힘’ 위주라면 놓칠 수 있는 조용한 점수 드리프트가, 실제 운영에서 누적·증폭될 수 있음을 시사하며 prompt-composed agent 평가에 cross-module interference 측정이 요구된다고 주장한다.



### Lacuna: A Research Map for Machine Learning (https://arxiv.org/abs/2606.26246)
Comments:
          14 pages, 3 figures. Preprint

- **Prior Approaches**: 기존 연구 인프라는 OpenAlex, S2ORC 같은 메타데이터/논문 텍스트 기반 코퍼스와, 지식그래프·임베딩으로 학술 검색을 지원하는 접근이 중심이었다. PaperQA나 OpenScholar 같은 retrieval-augmented 방식은 증거를 끌어와 답을 생성하지만, 질문이 만들어지기 전의 문제 정의(스코핑) 단계까지는 재사용 가능한 “연구 지도” 형태로 제공되지 않았다. 또한 deep research나 문헌조사도 대개 매번 PDF를 다시 읽으며 범위를 확장해 증거 합성을 수행하는 비효율이 남아 있었다.

- **Core Contribution**: Lacuna는 ML 문헌 위에 served, linked, paper-grounded 레이어를 얹는 대규모 research map으로, 논문을 격리된 PDF로 보지 않고 요약-개념-연구방향-연구제안의 중간 구조를 만든다. 각 항목은 원문 근거 링크를 유지해 사람이든 LLM 에이전트든 증거를 따라가며 탐색하고 인용·제안으로 확장할 수 있게 한다. 웹/markdown(/md)/MCP 인터페이스로 제공되어 검색만이 아니라 “연구 탐색 인프라”로 동작한다.

- **Technical Challenges**: 핵심 난제는 (1) 방대한 학술 레코드를 안정적으로 정합하고 (2) 생성된 내용이 실제 논문 증거와 감사 가능(auditable)하게 연결되도록 만드는 것이다. Lacuna는 OpenReview를 저자 정체성 앵커로 삼아 저자별 bibliography를 재구성하고 외부 소스(예: DBLP, OpenAlex)를 recall 보조로만 사용해 동명이인 오류가 방향·제안 컨텍스트로 전파되는 위험을 줄였다. 이후 논문에서 core-idea 요약→concept element(방법/한계/관찰)→HDBSCAN 클러스터 기반 research direction→그 근거로 research proposal을 만들고, 생성물마다 원문/메타데이터 링크를 보존하도록 파이프라인을 설계했다.

- **Empirical Impact**: 평가 결과 Lacuna는 LitSearch에서 Recall@10 0.538로 OpenScholar v3(0.424)보다 크게 향상되며, Multi-XScience-CS/ML에서도 관련-작업 합성 점수가 개선됐다. ScholarQA-CS-ML에서는 Lacuna-GPT-4o가 0.694(오픈소스 OpenScholar-GPT-4o 답안 0.672)로 콘텐츠·증거 품질 쪽에서 우위를 보였다. 또한 ReportBench-ML 25개 survey 태스크에서 Lacuna Deep Research는 citation F1 0.052, citation precision 0.339, expert-reference hit 99, RACE 품질 7.82/10으로 GPT-Researcher(0.039/0.290/72/5.24)를 능가해 deep research에서도 citation-grounded 합성 효과를 입증했다.



New uploads on arXiv(cs.CV)

### DanceOPD: On-Policy Generative Field Distillation (https://arxiv.org/abs/2606.27377)
Comments:
          Technical Report; 39 pages, 13 figures, 9 tables; Project Page at this https URL

- **Prior Approaches**: 기존 멀티캡처블 이미지 생성 학습은 데이터 혼합·조인트 트레이닝, 파라미터 병합/어댑터 조합, 또는 추론 시 score 조합 등으로 여러 능력을 함께 맞추려 했지만, 자칫 한 능력이 다른 능력을 깎아먹는 capability interference가 자주 발생한다. 특히 편집은 입력 보존을 요구하고 T2I는 개방형 품질을 중시하며, 로컬·글로벌 편집은 서로 다른 변환 목표를 가져 같은 업데이트에서 그라디언트 충돌이나 타깃 의미 희석이 생긴다. 또한 on-policy distillation 관점이 일부 다뤄졌더라도, “여러 generative field를 어떤 상태에서 어떤 방식으로 질의할지”까지 체계적으로 설계한 접근은 부족했다.

- **Core Contribution**: DanceOPD는 flow-matching 모델에서 이미지 생성 능력을 속도장(velocity field)으로 보고, on-policy generative field distillation을 통해 단일 학생 모델이 T2I·로컬 편집·글로벌 편집 등을 함께 조합하도록 만드는 프레임워크다. 핵심은 각 샘플을 하나의 capability field로 hard-route하고, 학생이 실제로 생성하며 방문하는 상태에서 stop-gradient로 teacher(고정 소스) 속도장을 질의해 MSE로 학습한다는 점이다. 이 설계는 operator-defined fields(예: classifier-free guidance, CFG)까지 동일한 방식으로 흡수할 수 있게 한다.

- **Technical Challenges**: 첫째, 여러 field를 한 샘플 타깃에 소프트로 섞으면 어떤 능력을 배우는지 의미가 흐려지는 target-field ambiguity가 생긴다. 둘째, 데이터 상태나 teacher 궤적 같은 off-policy 상태에서만 감독하면 학생이 실제 생성 시 방문하는 분포와 불일치하는 state-distribution mismatch가 발생한다. 셋째, 같은 롤아웃에서 여러 시점을 촘촘히 질의하면 상관된(trajectory-query correlation) 감독이 과집계되어 그라디언트 편향을 유발하므로, DanceOPD는 semantic-side low-noise 영역에서 샘플당 단일 질의 상태만 사용해 이를 완화하고, 학습 목적함수는 plain velocity MSE로 단순화해 안정성을 확보했다.

- **Empirical Impact**: 실험은 Z-Image 기반 capability composition과 SD3.5-M 기반 realism-field absorption, 그리고 CFG absorption까지 포괄하며, DanceOPD가 target 능력은 강화하면서 anchor 생성 품질을 보존함을 일관되게 보여준다. 예를 들어 편집 조합에서 GEditBench 성능이 기준 대비 8%대 향상했고, 로컬+글로벌 편집 조합에서도 조합 베이스라인 대비 큰 폭(두 자릿수 %대) 개선이 나타났다. realism-field 흡수에서는 off-policy 대비 realism reward가 9%대 향상하고 reward gap의 상당 부분(85%대)을 메우면서 T2I 점수 저하를 거의 억제했으며, CFG 흡수에서도 일부 방식은 성능을 개선하지만 과도한 over-guided composition은 성능이 크게 떨어졌다.



### Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards (https://arxiv.org/abs/2606.27376)
- **Prior Approaches**: 기존의 통합 대규모 멀티모달 모델(LMM)은 시각 이해와 이미지 생성을 함께 제공하더라도, 사람 주석·선호 레이블·외부 reward model 같은 큐레이션된 사후 학습 감독에 크게 의존해왔다. 즉 라벨이 없는 환경에서 두 능력을 동시에 끌어올리는 자율 학습은 제한적이었다.

- **Core Contribution**: 이 논문은 라벨 없는(unlabeled) 이미지 데이터만으로도 하나의 unified LMM이 시각 이해와 이미지 생성을 자가로 동시에 향상할 수 있는 self-evolving training framework를 제안한다. 내부에 Proposer(질문 생성)–Solver(답변/평가)–Generator(이미지 생성)라는 역할 분해를 두고, 사람·선호·외부 판정 없이 스스로 만든 일관성 신호만 학습에 사용한다.

- **Technical Challenges**: 핵심 난제는 샘플 간 일관성 기반 신호가 흔들릴 때도 학습을 안정화하는 방법과, 이해 평가와 생성 품질을 연결하는 내부 평가 설계를 동시에 만족시키는 것이다. 저자들은 Solver Token Entropy(STE)를 토큰 수준 예측 불확실도로 정의해 연속적 난이도 신호로 활용하고, 생성 측은 질문-답변 충실도와 cycle-consistent captioning을 멀티스케일로 결합한 multi-scale internal evaluation으로 solver-mediated coupling을 만든다.

- **Empirical Impact**: 실험에서는 diffusion 기반 BLIP3o, rectified-flow BAGEL, autoregressive VARGPT-v1.1 등 서로 다른 아키텍처에도 같은 역할 분해·보상 로직·학습 일정을 유지한 채 적용 가능함을 보여준다. 8개 이해 지표에서 기준 모델 대비 일관된 향상이 확인됐고, BAGEL에서는 MMMU에서 절대 +3.5%p 향상 및 GenEval 생성 성능이 82%에서 85%로 개선됐다. 코드와 모델을 공개해 재현성과 확장성도 함께 강화했다.



### Paying More Attention to Visual Tokens in Self-Evolving Large Multimodal Models (https://arxiv.org/abs/2606.27373)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 self-evolving LMM 연구는 proposer–solver 또는 questioner–reasoner 같은 다중 역할과 self-consistency(다중 샘플 일치) 보상을 통해 무감독 시각 추론을 강화해 왔습니다. 하지만 보상 설계가 답의 합의(agreement)에 치우쳐 있어, 생성 과정에서 디코더가 이미지 토큰을 충분히 주목하지 않고 language priors로도 ‘일관된 답’을 만들 수 있는 시각 under-conditioning 실패모드가 남습니다.

- **Core Contribution**: 이 논문은 VISE(Visual Invariance Self-Evolution)로, 답의 합의가 아니라 ‘시각 조건화(visual conditioning) 정책’을 직접 정규화하도록 보상 신호를 바꿉니다. 기하 불변성(변환 후 위치 일관성)과 의미 불변성(ghosting으로 증거를 제거했을 때 증거 없음 판단)을 함께 사용하며, 단일 모델에서 raw unlabeled images만으로 학습합니다.

- **Technical Challenges**: 핵심 기술적 난제는 무감독 환경에서 ‘답이 그럴듯한지’를 넘어 ‘실제로 이미지 증거에 의존하는지’를 보상으로 정의하는 것입니다. VISE는 변환된 영상에서도 예측 박스의 기하적 투영이 유지되도록 하고, 예측 영역을 흐리게(ghosting) 했을 때는 객체가 사라졌다고 판정하도록 강제해, 언어 통계만으로는 점수를 얻기 어렵게 만듭니다.

- **Empirical Impact**: 18개 벤치마크에서 Qwen3-VL-2B 기준 COCO와 TextCaps 이미지 캡셔닝 성능이 각각 +16.85 CIDEr, +19.66 CIDEr로 크게 개선되며, 객체 환각 지표(Chair-I)도 5.0점 줄었습니다. 또한 여러 VQA·추론 및 hallucination 평가에서 일관된 향상이 나타나고, 4개 백본 계열과 스케일 전반으로 일반화되며, self-evolving 계열의 기존 트레이드오프를 완화하는 의미가 있습니다.



### DnA: Denoising Attention for Visual Tasks (https://arxiv.org/abs/2606.27372)
- **Prior Approaches**: 기존 멀티헤드 어텐션의 표준인 softmax는 큰 점수는 강하게 키우고 작은(특히 음수) 점수는 거의 0으로 눌러 잡음/혼동을 만들 수 있다는 한계가 지적돼 왔다. 또한 다양한 비전 트랜스포머·비디오 트랜스포머는 attention 자체를 두되, softmax 정규화의 신호 억제 구조를 근본적으로 다루는 연구는 상대적으로 적다. differential attention처럼 차이를 활용하는 접근도 있지만, 두 상호작용이 같은 value subspace에 투영되어 있어 더 명확한 분리가 어렵다는 관점이 제시된다.

- **Core Contribution**: 이 논문은 Denoising Attention(DnA)이라는 새로운 attention 대체 모듈을 제안한다. DnA는 같은 key를 두고 positive query(정답 클래스에 해당하는 특징)와 negative query(깊게 연관됐지만 무관한 특징)를 각각 계산한 뒤, 상호작용 결과를 서로 다른 두 subspace로 투영해 분리도를 높인다. subspace separation으로 분별력을 키우는 동시에 softmax에서 억눌리던 “음수 상호작용”을 denoising 관점에서 활용한다.

- **Technical Challenges**: 핵심 기술적 도전은 softmax가 만들어내는 비대칭적 증폭이 음수/약한 상호작용 신호를 소거한다는 문제를, 학습 가능한 구조 변화로 얼마나 효과적으로 보정할지에 있다. 저자들은 principal angles(주요 각) 사이의 관계가 분류 오류 상한을 좌우한다는 이론 관점을 도입해, positive/negative 투영이 더 큰 principal angles를 갖도록 제약(서브스페이스 간 분리)을 설계한다. 실무적으로는 각 iterates에서 강한 직교 제약을 매번 강제하기보다, 초기화와 수렴 성질을 이용해 orthogonality가 높은 확률로 확보되도록 구성한다.

- **Empirical Impact**: ViT-B 백본에서 DnA는 ImageNet-1K 기준 baseline 대비 절대 0.8% 향상을 달성하며, 비디오 이해에서도 video transformers에서 최대 1.8%, video LLM에서는 0.5% 개선을 보고한다. 또한 여러 비디오 태스크에서 일관된 성능 상승이 나타나 DnA의 두 subspace 설계와 denoising 효과가 실증적으로 뒷받침된다고 주장한다. attention의 “잡음 억제”를 단순 가중치 교정이 아니라 기하(서브스페이스 분리)로 해석하고 개선했다는 점에서 후속 연구에 설계 가이드를 제공할 의미가 있다.



### Don't Settle at the Mode! Mitigating Diversity Collapse in Pretrained Flow Models via Feature Self-Guidanc (https://arxiv.org/abs/2606.27371)
Comments:
          Accepted by ECCV 2026. Project page: this https URL

- **Prior Approaches**: 사전학습된 flow 모델(예: FLUX 계열)은 프롬프트가 같을 때 출력이 비슷해지는 diversity collapse가 자주 관측된다. 기존 해법은 (1) latent guidance로 다양성을 유도하지만 개선 폭이 제한적이거나, (2) reward model 기반 sample selection/최적화를 써서 다양성을 키우는 대신 추론 시점 오버헤드가 커지는 양상이었다.

- **Core Contribution**: 이 논문은 diversity collapse의 원인을 최종 이미지가 아니라 flow 모델 내부 특징(특히 MMDiT 블록의 feature) 붕괴로 보고, 이를 직접 다루는 training-free feature self-guidance를 제안한다. 핵심은 배치 생성 중 내부 feature를 분산(dispersion)시키되, manifold regularization으로 데이터 매니폴드에 다시 투영해 조건 정합성과 화질을 유지하는 “disperse-and-refine” 구조다.

- **Technical Challenges**: 내부 feature를 무제약으로 벌리면 조건을 만족하는 의미 경계 밖의 저밀도 영역으로 밀려 prompt adherence가 흔들리거나 시각 아티팩트가 생길 수 있다. 저자들은 특정 MMDiT 블록과 초기 denoising 구간에서 feature 분산을 수행한 뒤, 동일 블록을 한 번 더 통과시켜 투영된 특징을 정규화(regularizer)로 반영하는 방식으로 다양성과 충실도를 함께 확보한다.

- **Empirical Impact**: FLUX.1-dev(텍스트-to-이미지), FLUX.1-Depth(깊이 조건), FLUX.1-Kontext(참조 이미지 개인화), 그리고 step-distilled FLUX.2-Klein-4B 등 여러 조건부 생성에서 기존 diversity 향상 베이스라인 대비 이미지 다양성은 크게 개선하면서 CLIPScore 기반 충실도 저하는 적었다. 특히 reward model을 반복 평가해야 하는 Group Inference보다 거의 비슷한 지연/비용으로 고다양성을 달성하며, 대규모 배치(최대 128개)에서도 성능이 안정적으로 유지된다는 점이 실용적 의미로 강조된다.



### PhysiFormer: Learning to Simulate Mechanics in World Spac (https://arxiv.org/abs/2606.27364)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존의 비디오 기반 world model은 관측된 픽셀을 그대로 다뤄 시점 의존성이 커지고, 3D 물리적 일관성을 유지하기가 어렵다는 한계가 있다. 또한 neural physics 계열은 임의의 latent space를 쓰거나 강제로 rigidity/causality 같은 귀납편향을 넣는 방식이 많아, 설계 편향이 성능과 일반화에 영향을 줄 수 있다.

- **Core Contribution**: PhysiFormer는 3D 물체를 world coordinates의 3D mesh로 표현하고, 초기 vertex 위치·속도와 물질 타입(강체/탄성)을 입력받아 미래 vertex 궤적을 생성하는 diffusion transformer를 제안한다. 핵심은 vertex trajectory 예측을 world 좌표에서 단일 denoising diffusion 과정으로 바로 구성해, 별도 inductive bias 없이도 물리적으로 그럴듯한 결과를 낸다는 점이다.

- **Technical Challenges**: 좌표계에서의 vertex 궤적을 확률적으로 생성하려면 학습 동역학의 불확실성을 반영하면서도 물체 간 관계를 효율적으로 모델링해야 한다. PhysiFormer는 time/space/objects 축으로 attention을 factorised해 계산 효율을 확보하고, 객체에 대한 순열 불변(permutation-invariant) 다중 물체 추론을 위해 별도의 object encoding을 요구하지 않도록 설계했다.

- **Empirical Impact**: 100k+ 시뮬레이션 궤적에서 학습한 뒤 rigid·elastic 역학을 생성하며, mixed-material 설정과 미관측 real-world 기하, 그리고 더 많은 물체 수로 일반화된다. 또한 autoregressive baseline보다 trajectory 정확도, rigidity 보존, momentum 기반 물리적 일관성에서 크게 앞서 coordinate-space diffusion을 로보틱스·그래픽스·물리 설계의 view-invariant geometry-aware world modelling에 유망한 방향으로 제시한다.



### RayPE: Ray-Space Positional Encoding for 3D-Aware Video Generation (https://arxiv.org/abs/2606.27345)
- **Prior Approaches**: 기존 비디오 diffusion transformer는 (u,v,t) 인덱스에 3D RoPE를 적용해 토큰의 샘플링 격자 위치만 인코딩하며, 실제 3D 장면에서의 광선(카메라 레이) 기하 관계는 별도 단서 없이 픽셀 콘텐츠로부터 암묵적으로 복원해야 했다. 카메라 정보를 넣는 후속 연구들은 adapter, ControlNet, point cloud 렌더링, cross-attention 등에서 기하를 “어텐션 점수 밖”의 보조 경로로 처리하는 경향이 있고, attention의 dot product 자체에는 기하가 구조적으로 반영되지 못했다. 카메라-aware positional encoding 시도들 또한 RoPE를 대체/분해하거나(곱셈 방식), 회전·투영 등 축약 파라미터화에 의존해 사전학습된 포지셔널 구조를 흔들거나, 번역 스케일이 다른 데이터에서 안정성을 확보하기 어려웠다.

- **Core Contribution**: 이 논문은 카메라 레이를 6D Plücker 좌표로 보고, 두 레이의 기하 관계를 Plücker reciprocal product로 정의하면 이것이 self-attention의 q·k dot product와 같은 “쌍선형(bilinear) 대수 형태”를 가진다는 점을 이용한다. 그 결과 RayPE(Ray-space Positional Encoding)로, per-token 6D Plücker 좌표를 self-attention의 query/key에 “가산적(additive)”으로 주입해 attention 점수 자체가 콘텐츠 항과 함께 기하 항을 포함하도록 만든다. 또한 Q/K flip 배치를 통해 대칭적인 설정에서 geometry-only 상호작용이 Plücker reciprocal product와 정확히 일치하도록 설계했다.

- **Technical Challenges**: 핵심 기술적 문제는 두 가지다. 첫째, Plücker moment는 카메라 번역 스케일에 선형으로 비례해 SfM/SLAM/metric처럼 스케일 기준이 다른 데이터가 섞이면 geometry 신호 크기가 크게 요동한다. 둘째, 기하를 콘텐츠 브랜치에 단순 주입하면 상대적인 스케일이 제어되지 않아 학습이 불안정해진다. 이를 위해 Normalize-Gate-Inject(NGI)에서 ray 방향과 moment 크기를 분리하고 log-magnitude 기반의 learned gate로 인코딩을 조절한 뒤, QKNorm-normalized 콘텐츠 브랜치와 정렬되도록 RMSNorm을 적용해 스케일 안정성과 학습 안정성을 확보했다.

- **Empirical Impact**: RayPE는 사전학습된 비디오 DiT에 0.1% 미만의 파라미터만 추가하며, zero-initialized 주입으로 시작 시 pretrained 가중치에서 출발하도록 구성됐다. Wan2.2-TI2V-5B(5B)에서 RealEstate10K, DL3DV, PanShot, OmniWorld의 4개 데이터 혼합 학습을 통해 카메라 controllability, 프레임 간 3D 일관성, 전반적 비디오 생성 품질이 함께 개선됨을 보고한다. 아블레이션에서는 attention 분해 성분(콘텐츠 항, 기하 항, 그리고 콘텐츠↔기하 cross-term) 모두가 개별적으로 필요하며, 특히 cross-term 결합이 다른 단순한 ray PE 방식/경로로는 구조적으로 대체하기 어렵다는 점을 강조한다.



### SAM2Matting: Generalized Image and Video Matting (https://arxiv.org/abs/2606.27339)
Comments:
          ECCV 2026. Extended version. Project Page: this https URL

- **Prior Approaches**: 비디오 매팅은 초기 프레임 마스크 등으로 타깃을 특정해야 Video Object Segmentation(VOS) 수준의 시간 일관성을 확보할 수 있다. 하지만 동시에 픽셀 단위 알파 메이트릭스(α) 추정에는 영상 매팅의 고해상도 디테일이 필요해, 고수준 추적과 저수준 매팅 사이의 간극이 커진다. 기존 방식은 비용이 큰 비디오 매팅 데이터셋에 의존하거나, 사전 학습된 VOS 모델을 비디오 매팅 데이터로 fine-tuning해 추적 강건성을 저하시킬 수 있다는 한계가 있다.

- **Core Contribution**: 이 논문은 SAM2Matting으로 비디오 매팅을 ‘tracker-to-matting’ 형태로 디커플링(분리)해, 고수준 추적은 그대로 두고 저수준 알파 추정만 전용 모듈로 학습한다. 구체적으로 VOS tracker(SAM2, SAM3 등)에 region-proposal bridge(ROI Detector)와 matting heads를 결합해 시간 일관성은 추적기가 담당하고, 세밀한 반투명/경계 디테일은 매팅 구성요소가 보정한다. 또한 이미지 matting 데이터로만 학습해도 비디오에서 zero-shot SOTA 성능을 달성하며, 다양한 prompt 유형(마스크·포인트·박스·텍스트)을 지원한다.

- **Technical Challenges**: 핵심 기술 난제는 ‘마스크 기반 추적 결과’를 알파 메이트릭스의 미세 영역 추정에 그대로 연결하면 디테일 누락이나 경계 오류가 발생한다는 점이다. 이를 해결하기 위해 ROI Detector를 단순 morphological 연산이나 raw mask 사용 대신, VOS 마스크·다중 스케일 이미지 특징을 입력으로 하는 픽셀 단위 이진 분류로 설계해 matting-critical region을 정밀하게 찾는다. 이후 Progressive Alpha Predictor가 coarse-to-fine 캐스케이드로 알파를 단계적으로 정제하며, 중간 스케일 deep supervision과 matte-mask consistency/smoothness 손실로 구조 무결성과 경계 매끈함을 함께 확보한다.

- **Empirical Impact**: SAM2Matting은 이미지/비디오 매팅 벤치마크에서 SOTA를 보이며, 특히 V-HIM60 및 VideoMatte에서 zero-shot으로도 비디오-슈퍼바이즈드 대비 우수한 결과를 낸다. dtSSD가 낮게 유지되며, VOS tracker에서 기인한 시간 일관성이 잘 계승된다는 점이 실험으로 뒷받침된다. 또한 사람 중심(human-centric)뿐 아니라 in-the-wild(빠른 움직임, 복잡 배경, 타깃 부착/반투명 물체)에서도 강한 일반화를 보이고, SAM2.1-Tiny 기준 1080p 200프레임에서 40 FPS 수준의 효율까지 제시해 실용성도 강화했다.



### RoPEMover: Depth-Aware Object Relocation via Positional Embeddings (https://arxiv.org/abs/2606.27332)
- **Prior Approaches**: 기존 단일 이미지에서의 오브젝트 모션은 주로 “제거+삽입” 파이프라인으로 처리하거나, inpainting/리파인먼트를 반복해 장면을 맞추려는 방식이 많았습니다. 하지만 제거 단계는 그림자·반사 같은 장면 효과까지 함께 지워야 하고, 삽입 단계는 조명/기하가 어긋나며 copy-paste 티가 나기 쉽습니다. 더 나아가 video diffusion 기반 접근은 모션 일관성을 얻는 대신 데이터 구축과 학습 비용이 커지고, instruction-tuned 편집 모델은 공간 제어가 거칠다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 diffusion transformer의 내부에서 위치 정보를 담당하는 RoPE(rotary positional embeddings)를 직접 조작해, 장면 기하에 맞는 오브젝트 재배치를 유도합니다. 또한 2D RoPE를 depth-aware로 확장해, 단순히 앞뒤를 포함한 가림(occlusion) 순서와 새로 드러난 영역의 자연스러운 채움, 그리고 그림자/조명 같은 장면 의존 효과의 일관된 업데이트를 목표로 합니다. 즉, 단일 이미지에서도 명시적 좌표 제어와 물리적 정합성을 함께 노리는 “geometry-aware object motion”을 제안합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) RoPE 기반 조작이 원하는 x,y 이동뿐 아니라 깊이 축의 모호성까지 해결해야 한다는 점, (2) 가려진 영역과 새로 드러난 영역에서 완성된 콘텐츠가 장면 효과와 충돌하지 않도록 해야 한다는 점입니다. 저자들은 마스크 토큰에 대해 drag 신호로 RoPE를 warp(초기 denoising 단계에서 제한적으로 적용)해 공간 변형을 주고, 사용하지 않던 3D RoPE의 temporal 축을 depth 채널로 재해석해 metric depth 지식을 주입합니다. 또한 단안 깊이 추정(MoGe-2)을 affine 정렬해 깊이 스케일을 맞춘 뒤, 추론 시 오브젝트 깊이를 Laplace 방정식 기반으로 배경을 매끈하게 만들고 z-buffer로 가림을 처리하며 경계 근처에서 거리 가중 블렌딩으로 깊이 단절을 완화합니다.

- **Empirical Impact**: 합성 데이터(CLEVR 기반 paired scenes)로 모션 파라미터화 능력을 먼저 학습한 뒤, 165장의 소규모 실데이터로 LoRA 기반 파라미터 효율 fine-tuning을 거쳐 실제 사진에서의 정합성을 높입니다. ObjMove-A/B 벤치마크에서 DINO-Score, CLIPScore, DreamSim, 그리고 PSNR까지 다수 지표에서 기존 방법 대비 성능 향상을 보이며, 특히 occlusion 및 투명/복잡 상호작용 시나리오에서 일관성이 두드러진다고 보고합니다. 결과적으로 “큰 이동에서도 오브젝트 identity 유지”와 “그림자·반사 등 장면 의존 효과의 자연 전파”를 동시에 달성했다는 점에서, 단일 이미지 편집의 정밀 제어 수준을 한 단계 끌어올렸다는 의미가 있습니다.



### Not All Actions Are Equal: Rethinking Conditioning for Dexterous World Mod (https://arxiv.org/abs/2606.27325)
- **Prior Approaches**: 기존 action-conditioned world model은 행동을 하나의 임베딩으로 압축한 뒤 시각 생성(픽셀/잠재)에 AdaLN이나 MLP 결합처럼 단순 주입하는 방식이 주류였다. 이런 접근은 6-DoF 저차원에서는 잘 작동하지만, 고DoF 덱스터러스에서는 손가락 미세 신호가 큰 스케일의 손목/머리/카메라 모션에 의해 함께 섞여 학습이 불안정해지고 action following이 약해진다. 또한 의미적 grounding은 단일 VLM reasoner 의존이 많아, egocentric 고DoF 손동작에 필요한 객체-장면 정렬을 충분히 제공하지 못한다.

- **Core Contribution**: DexAC-WM은 action conditioning을 전역 압축이 아니라 “구조화된 과정”으로 재설계해, 고DoF 행동의 차원별 의미를 유지하도록 한다. 이를 위해 Structured Action Representation으로 action tokenization을 수행하고, Unified Local-Global Conditioning으로 미세한 로컬 정교화와 전체 시퀀스 의도 주입을 분리해 동시에 학습한다. 여기에 의미 분기(semantic branch)를 추가해 DINOv3 특징과 텍스트 임베딩을 결합함으로써 객체-장면 priors로 의미적 grounding을 보강한다.

- **Technical Challenges**: 핵심 난관은 고DoF 덱스터러스 행동이 여러 자릿수에 걸쳐 스케일이 크게 이질적이어서, 전역 임베딩으로 모으면 큰 동작 성분이 최적화에서 지배해 미세하지만 중요한 손가락 신호가 의미 붕괴를 겪는다는 점이다. 논문은 이를 해결하기 위해 SAT에서 차원별 정규화와 dimension-wise tokenization을 적용해 스케일 불균형을 줄이면서도 전역 평균화를 피하도록 했고, local cross-attention(미세 효과 정렬) + global modulation(AdaLN 기반 전체 일관성)으로 로컬-글로벌 제어를 동시에 제공한다. 더 나아가 dual cross-attention으로 DINOv3의 밀집 기하 단서와 텍스트 의미를 함께 latent에 주입해 시공간 정합성을 높였다.

- **Empirical Impact**: EgoDex와 EgoVerse 실험에서 DexAC-WM은 FID, FVD, PCK 등에서 강력한 베이스라인 대비 유의미하게 개선되며, 시각-시간적 사실감과 action-following 일관성이 함께 좋아진다고 보고한다. 또한 structured action conditioning이 다른 backbones에도 확장 가능함을 보이며 설계의 재사용성과 스케일링 잠재력을 뒷받침한다. 결과적으로 고DoF 제어로 world model을 확장하려면 구조화된 행동 모델링과 의미적 grounding이 동시에 필요하다는 실증적 메시지를 강화한다.



### OctoSense: Self-Supervised Learning for Multimodal Robot Perception (https://arxiv.org/abs/2606.27317)
- **Prior Approaches**: 기존 멀티모달 학습은 센서가 제공하는 표현(예: 이미지, 이벤트, LiDAR 등)과 시공간 특성이 달라도 이를 충분히 정렬해 학습하기 어렵다. 또한 센서별 주파수·지연·잡음이 크게 달라지는 실제 주행/로봇 환경에서는 모달리티 간 late alignment가 성능 저하로 이어지곤 했다.

- **Core Contribution**: 이 논문은 스테레오 RGB·이벤트 카메라·LiDAR·열화상·IMU·RTK GPS·자세/주행자기정보(CAN 버스, 사족 로봇 관절각)를 한 번에 제공하는 오픈소스 센서 플랫폼 OctoSense와, 이를 기반으로 한 59시간 규모의 시간동기 데이터셋 OctoSense를 제안한다. 또한 모달리티별 토큰화와 지연 융합을 결합한 late-fusion masked autoencoder로, 실세계 로보틱스 데이터에서 멀티모달 self-supervised learning을 효과적으로 수행한다.

- **Technical Challenges**: 핵심 기술 과제는 서로 다른 센서의 표현/빈도/지연/잡음을 어떻게 모델 입력으로 일관되게 다룰지에 있다. 저자들은 모달리티별 tokenizers로 spatiotemporal 특성을 흡수하고, 추론 시 modality-specific tokens를 캐시해 새 측정치가 도착하는 즉시 처리하는 구조로 이를 해결했다.

- **Empirical Impact**: 실험에서는 late-fusion masked autoencoder가 광학흐름·깊이·세만틱 세그멘테이션·ego-motion(이동/회전/조향각) 같은 과업에서 이미지 전용 foundation model 대비 더 나은 성능을 보였다. 또한 야간이나 센서가 심각하게 열화된 상황에서도 예측의 견고함이 유지되며, 표현 계산이 NVIDIA 5090에서 6.68 ms, Orin NX에서 112 ms로 비교적 빠르다는 점이 제시된다.



### ViQ: Text-Aligned Visual Quantized Representations at Any Resolution (https://arxiv.org/abs/2606.27313)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 멀티모달 LLM(MLLM)에서는 CLIP 계열 연속 시각 인코더를 쓰거나, 특정 과업에 맞춰 end-to-end로 fine-tuning한 시각 인코더를 결합해 왔습니다. 한편 QLIP, UniTok 같은 quantized visual encoder는 이미지 토큰화를 통해 통합 표현을 노렸지만, discrete 압축 과정에서 저수준 디테일과 고수준 의미의 균형이 깨져 성능 갭이 남았습니다. 특히 재구성 중심 표현은 의미 구조가 약하고, 의미 중심 특징은 양자화로 인해 세부 정보가 크게 손실되는 문제가 반복됐습니다.

- **Core Contribution**: 이 논문은 ViQ(Visual Quantized Representations)라는 프레임워크로, discrete 시각 표현에서도 의미와 디테일을 함께 유지하면서 native resolution 입력을 지원하는 통합 표현을 제안합니다. ViQ는 두 단계 학습으로 양자화 학습을 구조화하는데, text-aligned pre-training으로 시맨틱 정렬을 강화하고 이후 feature discretization 단계에서 정보 손실을 줄이도록 설계했습니다. 결과적으로 ViQ는 연속·고차원 시각 특징과 경쟁하는 멀티모달 성능을, 동시에 저수준 재구성 정밀도도 확보하는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 난제는 연속 고차원 시각 특징을 discrete 코드로 바꿀 때 정밀도 손실이 커지고, 그로 인해 의미·디테일이 동시에 무너지기 쉽다는 점입니다. ViQ는 이를 위해 proximal representation learning로 양자화 전에 잠재공간 복잡도를 점진적으로 제한하고(bottleneck + proximal 제약), 이후 FSQ(Finite Scalar Quantization)로 안정적인 discretization을 수행합니다. 또한 임의 해상도 처리를 위해 2D RoPE와 head-wise 양자화 메커니즘을 결합해, 공간 정보를 보존하면서도 가변 길이 토큰을 유연하게 처리하도록 했습니다.

- **Empirical Impact**: 실험에서 ViQ는 9개 벤치마크의 aggregated score 기준으로 Qwen2.5-1.5B 백본에서 57.2, Qwen2.5-7B에서 63.9를 달성하며 기존 최첨단 quantized/continuous 대비 경쟁력 있는 성능을 보였습니다(이전 SOTA 대비 각각 57.0, 63.8에서 개선). 더불어 image decoder와의 fine-tuning에서 PSNR 22.73, rFID 0.62를 기록해 주류 discrete visual autoencoder 중 1위를 차지할 정도로 저수준 디테일 보존도 강했습니다. 실제 멀티모달 학습 효율 측면에서는 visual quantized representations를 사용하면 시퀀스 길이에 따라 20%-70%까지 학습 가속이 관찰되어, 성능과 효율을 동시에 끌어올리는 의미가 큽니다.



### See & Sniff: Learning Visuo-Olfactory Representations (https://arxiv.org/abs/2606.27307)
Comments:
          ECCV 2026. Project Page: this https URL

- **Prior Approaches**: 기존 AI 후각 연구는 화학 구조나 센서 신호로부터 향/냄새 범주를 예측하는 데 집중해 왔고, 대부분 단일모달(unimodal)로 학습됐다. 최근 SmellNet처럼 대규모 데이터가 등장했지만, 여전히 시각과 짝지어진 paired 정보가 없어 비주얼-후각 정렬을 직접적으로 학습하기 어렵다. 이 때문에 후각 표현은 시각적 단서(형태·색·구도)와의 결합을 제대로 활용하지 못했다.

- **Core Contribution**: 이 논문은 시각-후각을 학습하기 위한 대규모 짝데이터 수집 없이, 기존 향(냄새) 단일 데이터에 웹 이미지를 합성 페어링해 SmellNet-V를 만든다. 핵심 아이디어는 같은 의미 범주 안에서는 odor identity가 조명·스케일 등 시각 변환에 비교적 불변이라는 가정이며, 이를 통해 smell-only를 cross-modal 벤치마크로 확장한다. 또한 SmellNet-V 위에서 See & Sniff라는 self-supervised 프레임워크를 제안해 공동 임베딩과 함께 향의 saliency 맵(공간 근거)을 생성한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘실제로 함께 수집된’ paired visuo-olfactory 데이터가 부족하다는 점이며, 이를 합성 페어링이 자연스러운 정렬을 유도하도록 설계해야 했다. 논문은 sniffing 단위를 시간 창으로 분절해 학습 과정을 생체 샘플링과 맞추고, dense local alignment용 similarity map을 구성한 뒤 대조학습으로 시각-후각을 촘촘히 맞춘다. 더불어 DINOv3 시각 백본(동결)과 ScentFormer 계열 후각 트랜스포머를 공유 임베딩 공간에 aligner로 투영해 정렬 안정성을 높였다.

- **Empirical Impact**: 실험에서 See & Sniff는 smell-only 기준선 대비, 향만으로 smell classification 성능을 7%가량 향상시키며 후각 표현 학습이 시각 슈퍼비전의 도움을 받는다는 점을 입증한다. 또한 cross-modal retrieval에서도 전반적으로 상위 성능을 보이고, 특히 더 어려운 R@1에서 우수함을 통해 fine-grained 매칭이 강화됐음을 시사한다. 마지막으로 pixel-level smell localization 과제를 새 벤치마크로 제공하고, saliency 맵 기반으로 향 출처 공간 그라운딩까지 확장 가능한 새로운 멀티모달 지향을 제안한다.



### Sculpting NeRF Geometry: Human-Preference Fine-Tuning of a 3D-Aware Face GAN (https://arxiv.org/abs/2606.27305)
- **Prior Approaches**: 기존 RLHF 기반 3D 생성 파이프라인은 메쉬 같은 명시적 표면 표현을 최적화하는 경우가 많으며, NeRF를 메쉬로 변환해 표면-supervised 데이터에 크게 의존하는 경향이 있다. 또한 대부분의 3D preference 튜닝은 텍스트 프롬프트에 조건화하고, 다중 시점 렌더링 이미지나 메쉬 토큰에 대한 보상으로 기하를 함께 다듬는다. 그 결과 2D 외관은 그럴듯해도 코(콧등)나 얼굴 측면처럼 기하 결함이 남는 문제가 여전히 크다.

- **Core Contribution**: 이 논문은 EG3D처럼 3D-aware 생성기를 전제로, 메쉬/형상 prior 없이 NeRF의 density field(σ) 위에 학습한 learned reward로부터 직접 fine-tuning을 수행한다. 보상 모델은 프롬프트나 메쉬 추출, 다중 시점 렌더링 없이도 σ 자체를 입력으로 받아 ‘기하 품질만’ 학습 신호로 제공하는 것이 핵심이다. unconditional 3D-aware face GAN(EG3D)에서 단일 애너테이터의 선호만으로도 사용자 선호 개선을 보인다.

- **Technical Challenges**: 가장 큰 난점은 사람이 보는 ‘좋은 3D 기하’가 렌더링/메쉬 추출 같은 중간 표현 없이도 학습 가능한 형태로 어떻게 정리되느냐이다. 연구진은 σ의 3D 표현(깊이/포인트클라우드/σ field 중 성능이 좋은 σ field)을 U-Net ResNet 기반 보상 모델에 넣고, fine-tuning 중 2D 외관은 density-consistency constraint로 질적으로 유지하되 분포 변화 비용(FID-50k는 4.09→6.66)을 상한 내에서 제한한다.

- **Empirical Impact**: 사용자 연구에서 fine-tuned EG3D의 얼굴 기하는 원본 대비 pairwise 비교 74.4%에서 더 선호되었다. 또한 보상 모델은 pretraining 없이도 작은 preference 샘플로 학습이 가능하고, 단일 프리퍼런스 소스에서도 의미 있는 기하 개선이 관찰된다. 이는 3D 생성에서 ‘렌더링/메쉬 기반 보상’ 의존을 줄이고, 생성기 내부 표현(NeRF density)으로부터 직접 preference를 전달하는 새로운 RLHF-style 튜닝 방향을 제시한다.



### Exact and Deterministic Patch Descriptor Retrieval via Hierarchical Normalization (https://arxiv.org/abs/2606.27280)
Comments:
          9 pages, 4 figures

- **Prior Approaches**: 고차원 임베딩의 최근접 이웃 탐색은 RAG, 영상/얼굴 검증(FaceNet류), 비주얼 플레이스 인식 등에서 핵심이지만, 임베딩 차원(D×N) 전수 검색은 비용이 크다. 그래서 HNSW, IVF-PQ 같은 ANN과 양자화/그래프 인덱스가 널리 쓰이지만, 근사 결과와 더불어 실행마다 다른 top-1을 내는 비결정성 문제가 자주 따라온다. Matryoshka Representation Learning(MRL)도 다중 크기 표현을 쓰지만, 엄밀한 가지치기(상한) 보장이 없어 ANN 특유의 근사·비재현성을 그대로 상속한다.

- **Core Contribution**: 논문은 patch descriptor에 대해 “항상 동일한(정확한) nearest neighbour를 반환하면서, 데이터베이스의 극히 일부만 평가”하는 retrieval 방법을 제안한다. 핵심은 Hierarchical Normalization(HN)으로, 128차원 벡터를 K차원 major와 (128-K)차원 minor로 나눠 energy를 (1-α), α로 분할하고 Cauchy-Schwarz 기반의 admissible upper bound를 만들어 branch-and-bound 프루닝을 수행한다. 이로써 근사 대신, 전수(full-vector) 검색과 provably identical한 결과를 보장하면서도 재현성까지 확보한다.

- **Technical Challenges**: 가장 큰 기술 과제는 “major prefix만으로 full inner product의 상한을 신뢰성 있게 만들고” 가지치기 누락이 0이 되도록 학습·알고리즘을 맞추는 것이다. HardNet에 HN을 단순 post-hoc로 적용하면 에너지가 major에 집중되지 않아 프루닝이 약해져 속도 이득이 줄어드는데, 이를 해결하려고 notredame split에서 TripletMarginLoss로 HN-modified HardNet을 fine-tuning해 major가 discriminative signal을 대부분 담도록 만든다. 또한 SoA(Structure-of-Arrays) 메모리 배치와 캐시 최적화를 통해 Phase-1의 K차원 스캔은 빠르게 만들고, Phase-2의 full 128차원 평가는 prunable하지 않은 항목에만 제한한다.

- **Empirical Impact**: UBC patch 벤치마크에서 K=8, α=1/32 설정(Structure-of-Arrays 캐시 최적화) 시, 전수 128차원 대비 trevi에서 13.7×/halfdome에서 12.7× 속도 향상을 보이면서 full 평가 비율은 0.4%에 그친다. 정확도 측면에서도 FPR@95는 K=16, α=1/8에서 trevi 기준 0.0062→0.0064 수준으로 소폭 상승하며, 여전히 98.8%는 full 평가를 우회한다. 무엇보다 HN 기반 상한-가지치기 덕분에 top-1이 실행/스레드/하드웨어에 따라 달라질 수 있는 ANN·MRL 파이프라인과 달리, 동일 데이터베이스/쿼리에서는 동일 결과를 재현하는 점이 실사용(캐싱, A/B 테스트, 회귀 추적, 안전 감사)에서 의미가 크다.



### CORTEX: A Structured Reasoning Benchmark for Trustworthy 3D Chest CT MLLMs (https://arxiv.org/abs/2606.27264)
- **Prior Approaches**: 기존 의료 MLLLM은 대부분 정답만 생성하는 one-step 응답 중심이거나, chain-of-thought(CoT)를 쓰더라도 구조가 없는 free-form 텍스트로 중간 추론을 제공하는 경우가 많았습니다. 특히 3D 흉부 CT에서는 병변을 수백 장면에서 공간적으로 통합해야 하지만, 추론이 임상 워크플로에 맞게 검증 가능하게 구성되지 않았고 단계 단위 평가 프로토콜도 거의 없었습니다. 또 CT VQA 데이터셋은 방사선의가 실제로 사용하는 환자 병력/검사 이유 같은 임상 맥락을 제거해, 이미지 단서만으로 판단하도록 만들어 추론의 확인 가능성을 더 떨어뜨립니다.

- **Core Contribution**: CORTEX는 3D 흉부 CT를 위한 구조화된 추론 벤치마크로, 각 문항의 누락된 ‘추론 연결’을 4단계 진단 추적(trace)으로 복원합니다. 단계는 task understanding(과업 이해)→visual observation(시각 관찰)→diagnostic reasoning(진단적 추론)→answer synthesis(답 종합)이며, 환자 병력 등 임상 맥락도 질문에 다시 붙여 실제 임상 의사결정에 가깝게 만듭니다. CT-RATE처럼 reasoning 주석이 없는 공개 데이터 위에, 학습용 구조적 감독과 검증용 단계별 설계를 함께 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 방사선의 워크플로와 맞는 ‘제약된’ 추론 구조를 대규모로 생성하고, (2) 최종 정답이 맞더라도 근거가 부실할 수 있는 문제를 단계 단위로 신뢰성 있게 걸러내는 것입니다. 논문은 frontier LLM들로 추론을 생성하되 4단계 태그 순서를 강제하고, 과업 이해/관찰 충실도/가설 평가/추론 논리/정답 정확도 등 5개 clinician-designed 루브릭으로 LLM 기반 자동 채점 후 하드 게이트(특히 과업 이해·정답 정확도)를 적용합니다. 또한 GPT-5.4-mini judge의 자동 검증 결과를 보드-certified 방사선의 독립 리뷰로 교차 확인해 품질을 담보합니다.

- **Empirical Impact**: CORTEX는 CT-RATE(공개 흉부 CT) 기반으로 총 76,177개의 검증된 reasoning trace를 구축했으며, open-ended/closed-ended/report generation 등 여러 형태의 VQA·리포트 생성에 확장됩니다. GPT-기반 단계 루브릭 채점과 별도로 1,000개 문항에 대해 방사선의 3인이 추론 정확성을 라벨링했고, 정답성(correctness) 기준 상호일치가 93%에 달해 자동 검증의 신뢰도를 뒷받침합니다. 결과적으로 ‘3D 흉부 CT 추론’을 단계별로 해석·검증 가능한 데이터/평가 틀로 제공해, 향후 reasoning-capable 3D CT MLLM 학습과 공정한 평가에 직접적인 표준 역할을 할 전망입니다.



### SatSplatDiff: Geometry-preserving generative refinement for high-fidelity satellite Gaussian Splatting (https://arxiv.org/abs/2606.27223)
Comments:
          23 pages, 15 figures

- **Prior Approaches**: Gaussian Splatting은 위성 장면을 효율적으로 표현하며, 방사학적으로 다양한 장면에 유연하다는 점이 장점으로 부각됐다. 다만 위성 이미지의 관측 시점이 제한적이라 건물 입면에 대한 감독이 부족해 표면 구멍과 시각적 충실도 저하가 자주 발생한다. 최근에는 generative refinement로 렌더링 결과를 더 그럴듯하게 다듬을 수 있지만, 각 뷰를 독립적으로 갱신하면서 환각(hallucination)이 생기고 photo-consistency가 깨져 기하 품질이 악화될 수 있다.

- **Core Contribution**: 이 논문은 SatSplatDiff로, generative refinement 과정에서 나타나는 기하 열화를 최소화하는 것을 핵심 목표로 삼는다. 기존 SatSplat의 photogrammetric DSM 초기화와 2DGS 기반 shadow casting을 토대로, 먼저 기하적으로 정확하고 잘 정규화된 표면 표현을 만든 뒤 그림자 정보를 사용해 생성적 정제를 geometry 일관성 있게 유도한다. 결과적으로 시각적 품질을 올리면서도 기하 붕괴를 줄이는 방향을 제시한다.

- **Technical Challenges**: 생성 모델을 이용해 뷰별로 이미지를 정제하면 환각과 photo-consistency 붕괴로 이어져 geometry가 망가지는 문제가 기술적 난제로 제기된다. 저자들은 monocular depth supervision과 multi-scale geometric refinement로 표면을 먼저 탄탄하게 만들고, geometrically 계산된 shadow maps로 Gaussians가 기반 기하에 맞춰 조정되도록 shadow-guided generative refinement를 설계했다. 이렇게 하면 각 뷰가 독립적으로 떠버리는 현상을 줄이고, 그림자를 앵커로 삼아 기하-외관 정합성을 강화한다.

- **Empirical Impact**: IARPA2016과 DFC2019에서의 대규모 실험에서 state-of-the-art 성능을 보이며, geometric MAE는 최대 18% 감소했고 FID-CLIP은 28~45% 개선됐다. 또한 환각을 최소화하면서 센서 일관성(senser-consistent)에 가까운 외관을 제공하며, 최대 5배 해상도 향상도 보고한다. 더 나아가 cross-tile consistency와 대규모 재구성을 위한 확장성까지 확인돼 위성 3D 재구성 분야에 실질적 임팩트를 준다.



### LISA: Likelihood Score Alignment for Visual-condition Controllable Generation (https://arxiv.org/abs/2606.27192)
- **Prior Approaches**: 기존 비주얼 조건 제어 생성은 frozen된 diffusion/flow main network에 조건 인코더(side network)로 중간 feature를 결합하는 dual-branch 패러다임이 주류를 이뤘습니다. 다만 side branch의 “무엇을 학습해야 하는지” 역할이 명확히 감독되지 않아 학습 효율과 정합성이 흔들릴 수 있고, REPA 계열처럼 외부 의미 인코더에 의존하면 성능이 선택한 인코더에 묶일 수 있습니다.

- **Core Contribution**: 이 논문은 dual-branch의 두 네트워크 역할을 score-based 관점에서 재해석하고, side network가 최종 조건 score를 만들기 위한 보정(residual)을 수행하며 그 보정이 likelihood score에 해당한다고 설명합니다. 이를 바탕으로 LIkelihood Score Alignment(LISA)라는 정규화 기법을 제안해, side branch 중간 feature가 likelihood score를 반영하도록 명시적으로 정렬합니다. 핵심은 lightweight decoder로 side feature를 score latent space에 투영하고, 근사된 likelihood score 타깃과의 거리를 추가 loss로 사용한다는 점입니다.

- **Technical Challenges**: challenge는 side network가 암묵적으로 likelihood score 역할을 하도록 학습되는데, 기존 학습 목적이 최종 score만 감독해 side branch의 역할이 “직접” 정렬되지 않는다는 데 있습니다. 논문은 conditional score가 직접 계산되기 어렵다는 문제를 피하기 위해, denoising target(단일 샘플 기반의 unbiased supervision)과 main network의 unconditional score 예측을 조합해 샘플 단위 근사 likelihood score를 구성하고, 이를 decoder 출력과 stop-gradient 기반 정규화로 정렬하도록 설계했습니다. 또한 메인 네트워크는 frozen으로 두고 side network와 decoder만 공동 최적화하며, 추론 시 decoder는 제거해 추가 inference 비용을 없앴습니다.

- **Empirical Impact**: 실험은 pose/segmentation/depth/저해상도 등 다양한 이미지 조건에서 diffusion 및 flow 모델, 서로 다른 아키텍처에 걸쳐 LISA가 조건 충실도와 생성 품질을 일관되게 개선했다고 보고합니다. 특히 ControlNet 대비 더 적은 iteration으로도 더 낮은 FID와 더 높은 조건 정합 지표를 달성해 학습 수렴을 가속하며, 외부 semantic encoder 없이도 REPA와 유사한 이득을 보였습니다. 계산 오버헤드는 학습 시 약 0.1% 수준의 추가 파라미터와 미미한 시간 증가에 그치고, 추론 단계에서는 decoder를 버리므로 baseline과 동일한 비용으로 배포 가능하다는 점에서 의미가 큽니다.



### HarmVideoBench: Benchmarking Harmful Video Understanding in Large Multimodal Models (https://arxiv.org/abs/2606.27187)
- **Prior Approaches**: 기존 harmful video 벤치마크는 대체로 harmful/non-harmful 이진 분류에 초점을 두어, 비디오가 암시적으로 전달하는 깊은 위해(의도/문맥)를 충분히 포착하지 못했다. 또한 정답 맞히기 여부만 평가하고 왜 그런지 설명하는 rationales가 거의 없어, 모델이 표면 단서나 편향된 상관관계로 맞출 가능성이 있어도 진단이 어렵다.

- **Core Contribution**: 이 논문은 HarmVideoBench를 제안하며, 유해성 이해를 단일 결정이 아니라 관찰 가능한 증거에서 시작해 클립 내부 의미, 나아가 클립을 넘어선 추론으로 확장되는 3단계 계층적 진단으로 재구성한다. 1,379개 영상과 4,137개 객관식 Q로 구성되며, 세 범주(Observable Evidence, Clip-Internal Meaning, Beyond-Clip Reasoning)별로 모델이 어디서 실패하는지 분해해 측정한다.

- **Technical Challenges**: 기여의 핵심은 다차원 유해성의 경계를 객관적으로 질문/선지에 반영하는 것으로, AI가 만든 후보를 인간 검수(다중 어노테이터 합의·고급 adjudication)로 정제해 근거성과 난이도를 보장한다. 또한 모델이 “이 문제는 클립만으로 풀리는가, 아니면 추가 맥락이 필요한가”를 잘못 가정하면 추론 깊이에서 무너져, 이를 완화하는 구조로 BCR(Boundary-Constrained Reasoning)을 설계해 필요한 경우에만 문맥을 선택적으로 retrieve하고 경계에 맞춰 decoding을 수행한다.

- **Empirical Impact**: 19개 최신 모델을 평가한 결과, 대부분은 Observable Evidence에서는 상대적으로 강하지만 Clip-Internal Meaning과 Beyond-Clip Reasoning에서 현저히 약하며, 특히 문화·역사·사회적 지식이 필요한 항목에서 격차가 커졌다. BCR을 적용하면 기준 모델의 매크로 평균이 61.7%에서 84.4%(state-of-the-art)로 크게 상승해, 유해성 이해가 단순 지각 능력만이 아니라 증거 경계 제어와 맥락 연결에 달려 있음을 실증적으로 뒷받침한다.



### Safe Autoregressive Image Generation with Iterative Self-Improving Codebooks (https://arxiv.org/abs/2606.27147)
Comments:
          10 pages including references, 8 figures, accepted for publication at the 43rd International Conference on Machine Learning (ICML 2026)

- **Prior Approaches**: 기존 연구는 확산모델 기반 T2I(텍스트-이미지 생성)에서의 안전성 문제를 주로 다뤘으며, 사후 필터링(NSFW 탐지)이나 확산의 연속 잠재공간/임베딩 공간에서의 가이던스·학습으로 위해 개념을 줄이는 방식이 많았다. 하지만 토큰 기반 자가회귀 unified multimodal 모델은 연속 공간이 아니라 discretized visual tokens와 codebook에 의존해, 이런 연속공간 중심 방법은 일반화가 어렵다. 또한 외부 인간 라벨이나 인간 피드백이 필요한 안전화 파이프라인은 비용과 확장성에서 한계가 있었다.

- **Core Contribution**: 이 논문은 autoregressive unified multimodal 모델에서 안전한 생성을 위해 codebook 자체를 “iterative self-improving” 방식으로 개선하는 Safe-CodeBook을 제안한다. 핵심 아이디어는 모델이 자기 자신으로 unsafe 결과를 판별하고, harmful과 safe에 대응하는 공간을 구성해 harmful mapping을 제거한 뒤, 품질 저하를 막기 위해 null space에서만 적응 fine-tuning을 반복한다. 추가적인 외부 피드백 없이도 위험한 출력이 점진적으로 줄어드는 self-improvement 루프를 설계한 점이 기여다.

- **Technical Challenges**: 자기 모델 내부에서 unsafe를 라벨 없이 식별해 harmful 개념에 해당하는 codebook의 “어떤 방향”을 건드려야 하는지가 첫 번째 난관이다. 논문은 모델의 판단으로 harmful image-text 쌍을 만들고, safe-unsafe 임베딩 차이를 모아 SVD로 harmful subspace를 추출한 뒤 코드북 임베딩을 투영 제거해 harmful 방향을 억제한다. 두 번째 난관은 하드 제거가 시각 품질을 떨어뜨릴 수 있다는 점인데, 이를 null space projection을 이용해 harmful 정보와 무관한 섭동만 학습하도록 제약해 품질을 보존하면서 안전성을 유지한다.

- **Empirical Impact**: 실험은 I2P, CoPro, ViSU 등 8종의 위해 프롬프트 데이터셋과 성적 위해 개념 중심의 추가 벤치마크들에서 안전성 개선을 일관되게 확인했다. 탐지기 기준으로 harmful 생성이 유의미하게 감소했으며, COCO-30k에서의 FID 측정과 기존 모델의 생성/이해 능력 평가를 통해 원래 역량 저하가 크지 않음을 함께 보여준다. 또한 단일 제거보다 반복 제거가 특정 위해 개념에서 더 잘 작동하고, Janus 등 여러 unified multimodal generation 모델과 OOD 상황에서도 적용 가능성이 보고되어 분야 적용성이 강화된다.



### FlameVQA: A Physically-Grounded UAV Wildfire VQA Benchmark with Radiometric Thermal Supervision (https://arxiv.org/abs/2606.27128)
- **Prior Approaches**: 기존 원격탐사 VQA 벤치마크는 주로 RGB 광학 영상 중심으로 설계돼 위험(hazard) 도메인의 온도·열 신호가 필요한 질의를 충분히 평가하기 어렵습니다. UAV 화재 데이터도 탐지/분할/분류에 치우쳐, 지각을 넘어선 고수준 추론과 안전-운항 의사결정까지 묻는 멀티모달 QA 평가는 제한적이었습니다.

- **Core Contribution**: 이 논문은 UAV 기반 산불 모니터링을 위한 다지선다 Visual Question Answering 벤치마크 FlameVQA를 제안합니다. RGB와 함께 radiometric thermal TIFF(픽셀 단위 온도)를 제공해, 연기·가림·스케일 변동 환경에서도 온도에 근거한(temperature-grounded) 안전-중요 추론을 평가하도록 구성했습니다.

- **Technical Challenges**: 핵심 난관은 라벨 신뢰도를 대규모로 확보하면서도 온도 임계 기반 커버리지·존재 판정처럼 애매한 항목에서 정답성을 유지하는 것입니다. 이를 위해 MLLM-assisted annotation을 출발점으로 하고, NumPy 기반 TIFF 직접 판독(임계/퍼센트 계산), EXIF GPS+지형 모델로 고도 범주 산정, 그리고 질문 간 논리 제약(예: No fire면 hotspot 관련 답변 금지)과 인간 검수의 다단계 검증을 결합했습니다.

- **Empirical Impact**: Willamette Valley 하위셋에서 인간 평가와의 일치율은 70.78%로, 특히 공간 국소화·크로스모달 추론은 높고 PD5(연기 속 활성 화재)에서 가장 낮게 나타났습니다. LLaVA와 Qwen-VL 같은 대표 MLLM 베이스라인은 존재/방향 국소화에서 Qwen-VL이 우세했지만, 분포·커버리지(aggregation·퍼센트 추론)는 전반적으로 어려워했습니다. 결과적으로 현재 MLLM은 재난/산불 도메인 특화 적응이 필요하며, FlameVQA가 그 격차를 체계적으로 드러내는 실용 벤치마크로 의미가 있습니다.



### TMP: Tree-structured Mixed-policy Pruning for Large-scale Image Generation and Editing (https://arxiv.org/abs/2606.27089)
Comments:
          10 pages, 3 figures, 3 tables, tech report

- **Prior Approaches**: 기존 구조적 프루닝(structured pruning)은 모델을 슬림화하지만, 대부분 DiT 아키텍처와 텍스트-투-이미지 중심으로 설계돼 MoE 기반(예: HunyuanImage-3.0)이나 이미지 편집(editing)까지 그대로 확장하기 어렵다. 또한 프루닝 비율이 커질수록 표현 정렬 불일치와 층 간 오차 누적으로 인해 성능이 크게 떨어지는 문제가 반복됐다. 회복 학습에서는 KD를 쓰되 손실 설계나 오프폴리시(off-policy) 기반 중간 표현 모방이 한계로 지적된다.

- **Core Contribution**: TMP는 Tree-structured Mixed-policy Pruning으로, MoE와 DiT 같은 서로 다른 백본 및 T2I·TI2I(image editing) 작업까지 아우르는 범용 프루닝·회복 학습 프레임워크를 제안한다. 핵심은 프루닝 후 두 단계 회복 학습을 수행하되, 중간 토큰 표현 정렬을 오프폴리시·온폴리시(on-policy)를 혼합해 강화하고, 트리 구조로 구간을 점진 병합하는 전략으로 최적화 난이도를 완화한다. 결과적으로 대폭 압축에서도 생성 품질 저하를 제한하는 것을 목표로 한다.

- **Technical Challenges**: 문제는 (1) 프루닝으로 인해 층별 표현이 어긋나면 end-to-end 관점에서 오차가 누적돼 모델 붕괴(model collapse)로 이어진다는 점, (2) MoE 게이팅 및 폭(width) 감소까지 포함해도 회복 학습 타깃을 안정적으로 구성해야 한다는 점이다. TMP는 구간별 중간 특징 distillation을 트리 형태 divide-and-conquer로 수행하고, 초기에는 지역(supervision region) 정렬을, 학습이 진행되면 이 구간들을 bottom-up 방식으로 병합해 점점 통합된 목표로 수렴하도록 설계한다. 여기에 학생 롤아웃을 이용한 on-policy feature distillation을 추가해 혼합 정책 기반으로 교사-학생 정렬을 더 촘촘히 맞춘 뒤, 최종 단계에서 velocity/next-token 예측을 KL로 정렬해 미스얼라인을 줄인다.

- **Empirical Impact**: 실험에서 HunyuanImage-3.0은 80B에서 20B(파라미터 75% 감소)로 줄이면서도 인간 선호 기준에서 -2.5% 수준의 품질 저하만 보고했다. 또한 FP8 양자화와 동적 오프로딩을 포함한 추론 최적화로, 24GB RTX 4090 한 장에서 1024×1024 이미지를 8 스텝 샘플링으로 생성 가능함을 보였다. Z-Image turbo(6B→4B, 33% 감소)에서도 거의 무시할 수준의 성능 저하로 TMP의 일반성을 뒷받침하며, 대규모 오픈소스 이미지 생성 모델의 커뮤니티 접근성을 크게 낮추는 데 의미가 있다.



### SubdivAR: Autoregressive Next-Scale Prediction for Neural Mesh Subdivision (https://arxiv.org/abs/2606.27088)
- **Prior Approaches**: 전통적인 규칙 기반 subdivision은 고정된 로컬 스텐실로 매 단계 정련을 수행하지만, 날카로운 특징이 과도하게 매끈해지며 고주파 디테일 복원이 약하다는 한계가 있습니다. 최근 Neural Subdivision 계열은 학습 가능한 정련을 도입했지만 주로 local message passing에 의존해 전역 의미 맥락을 놓치기 쉽고, watertight(폐곡면) 중심 가정으로 open-surface에는 제약이 큽니다. 또한 정제되지 않은 coarse-to-fine 데이터는 연결성과 정합성 노이즈를 만들어 학습 상한(learning ceiling)을 높입니다.

- **Core Contribution**: SubdivAR는 subdivision을 “다음 스케일 좌표 예측(Next-Scale Coordinate Prediction)”으로 재구성하는 Mesh Autoregressive Representation(MAR)을 제안합니다. coarser에서 finer로 넘어가며 새 정점의 vertex offset을 단계적으로 회귀해, subdivision topology를 보존하면서 미세 기하를 계층적으로 복원합니다. 여기에 Hybrid Topology-Aware Transformer를 더해 전역 의미 attention과 토폴로지 제약 로컬 feature aggregation을 함께 수행하며 open-surface까지 자연스럽게 확장합니다.

- **Technical Challenges**: 핵심 도전은 (1) 전역 문맥을 반영하되 topologically disconnected한 정점 간의 “환각적 상호작용”을 막는 것, (2) coarse-to-fine 학습쌍의 품질이 떨어질 때 생기는 왜곡/불규칙 연결 노이즈를 줄이는 것입니다. SubdivAR는 topology-constrained cross-attention으로 관심 토큰을 manifold neighborhood로 제한하고, staged supervision 및 경계(boundary) 표시 특징을 통해 identity 매핑으로의 붕괴와 open-boundary 처리를 동시에 완화합니다. 동시에 nearly 40,000개 규모의 FII-40K를 구성해 fidelity/integirty/informativeness 기준으로 실패 케이스를 필터링, 학습 안정성을 크게 끌어올렸습니다.

- **Empirical Impact**: 실험에서 SubdivAR는 기존 최첨단 baseline 대비 Hausdorff Distance(HD) 18.8%, Chamfer Distance(CD) 14.2% 감소로 성능 우위를 보였고, 복잡한 open-surface에서도 강건성을 확인했습니다. 또한 전체 데이터셋에서 적응된 신경 기반 방법 대비 HD와 CD가 각각 62%와 31% 더 낮아 디테일 복원과 기하 정합성이 동시에 개선됐음을 보여줍니다. 한계로는 decimation 과정에서 생기는 정보 손실과 L2 회귀의 과평활 경향이 남아, 향후 generative modeling으로 이를 완화할 계획을 제시합니다.



### Pseudo-Text-Conditioned 3D Grounding DINO for Organ Localization in Abdominal C (https://arxiv.org/abs/2606.27084)
Comments:
          24 pages, 17 figures

- **Prior Approaches**: 기존 DETR/DINO 계열은 앵커와 NMS 없이 set prediction으로 객체를 찾는 틀을 제공했지만, 2D 중심 설계가 많아 3D CT에 그대로 옮기면 계산·메모리 부담이 커진다. Grounding DINO는 text-image grounded pretraining과 language-guided query selection으로 의미 기반 탐지를 강화했으나, 3D 볼륨과 장기 박스 회귀에는 별도 적응이 필요하다. 의료 영역에서는 세그멘테이션이 제한된 상황에서 exemplar/semantic guidance를 활용한 탐지가 유망하지만, 본 논문처럼 고정 장기 어휘를 두고 pseudo-text로 3D localization을 직접 겨냥한 베이스라인은 아직 초기 단계다.

- **Core Contribution**: CT-3GDINO는 3D CT에서 간단한 장기 vocabulary(간, 비장, 좌/우 신장, 장)로 고정된 pseudo-text class tokens를 만들어 query 기반 3D detector를 구성한다. 실제 text encoder 없이 frozen pseudo-text embedding과 가벼운 projection만으로 semantic-conditioning 인터페이스를 유지해, downstream을 위한 공간 prior(정규화 3D bounding box)를 제공하는 것을 목표로 한다. 저자들은 Swin3D 시각 백본, bidirectional feature enhancer, pseudo-text-guided query selection, cross-modality decoder를 end-to-end로 학습하는 오픈소스 베이스라인을 제안한다.

- **Technical Challenges**: 3D에서 global attention은 메모리 비용이 매우 커서, Swin3D의 window-based 계층 구조로 공간 맥락을 유지하면서 학습 가능 구성을 만든다. 또한 장기 박스는 foreground 픽셀(세그멘테이션)에서 구한 상자여서 엄밀한 경계 정합이 어려운데, 이를 set-prediction용 Hungarian matching과 3D generalized IoU/L1 혼합 손실로 학습 신호를 보완한다. pseudo-text의 의미를 디코더에 주입하려고, feature enhancer로 시각-의사 텍스트를 먼저 상호작용시키고, language-guided query selection으로 의미 관련 토큰에서 출발하도록 설계를 고정했다.

- **Empirical Impact**: RSNA/RATIC의 193개 볼륨(세그멘테이션 유도 박스)에서 CT-3GDINO의 최적 multi-scale 모델은 전체 top-1 class-wise mAP 0.5830을 기록하며, 분류 프리트레인(고정/미고정) 백본 변형(각 0.5570, 0.4657)을 능가한다. 특히 IoU 0.1에서는 AP 0.9649로 거친 위치 파악은 매우 강하지만, IoU 0.7에서는 AP가 0.1552로 크게 떨어져 엄격한 3D 박스 정합은 여전히 한계임을 보여준다. 이 결과는 pseudo-text-conditioned 3D 장기 localization의 실용적 출발점(베이스라인)을 제시하면서, localization-aware pretraining과 더 풍부한 multimodal conditioning, injury-focused detection으로의 확장을 동기를 부여한다.



### PanoImager: Geometry-Guided Novel View Synthesis and Reconstruction from Sparse Panoramic Views (https://arxiv.org/abs/2606.27071)
Comments:
          IROS 2026

- **Prior Approaches**: 기존 SfM/SLAM은 회전 우세·약한 시차 환경에서 삼각측량이 불안정해져 초기화가 ill-conditioned 혹은 실패하기 쉽습니다. 3D Gaussian Splatting(3DGS) 역시 파노라마에서는 투영 왜곡과 희소 관측에 민감해 뷰 간 정합성이 흔들릴 수 있습니다. 확산 모델 기반 view completion은 그럴듯한 합성을 잘하지만, 파노라마의 구면 왜곡·기하 제약을 충분히 만족시키지 못해 cross-view geometric consistency를 보장하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 SfM-free로 희소 파노라마(대개 Np=3~6)에서 3D를 재구성하는 PanoImager를 제안합니다. 핵심은 (1) feed-forward pose/depth priors로 기하 기준선을 잡고, (2) geometry-conditioned diffusion으로 관측 공간을 조밀화하며, (3) depth-guided 3DGS 최적화에서 합성 뷰를 신뢰도 가중 soft prior로 흡수해 뷰 간 일관성을 안정화하는 구조입니다. 생성(view completion)을 독립 모듈로 두지 않고 ‘3D Gaussian 장면의 기하 스캐폴드’로 결합한 점이 기여입니다.

- **Technical Challenges**: 파노라마 ERP(equirectangular projection)는 구면에서의 각도-픽셀 스케일이 균일하지 않아, 희소 입력에서는 작은 오차가 큰 방향/기하 불안정으로 번지며 최적화가 수치적으로 취약해집니다. 이를 해결하기 위해 논문은 파노라마를 구면 위 quasi-uniform 샘플링으로 로컬 perspective chart(탄젠트 공간)로 분해해 기하 조건을 개선하고, 그 위에서 VGGT 기반의 feed-forward pose/depth priors로 정렬을 시작합니다. 또한 합성 뷰는 warped RGB/D, visibility mask로 occlusion 경계를 줄이고, 신뢰도(재투영 잔차·깊이 불일치)에 따라 3DGS 목적함수에 soft하게 반영하며 anti-floater regularization으로 구조 부유를 억제합니다.

- **Empirical Impact**: 여러 벤치마크(Replica, OmniScenes, 360Roam)에서 입력 희소성이 극단적으로 커질수록 PanoImager가 기존 dense/피드포워드·반복 기반 방법보다 재구성 안정성과 기하 정합성이 더 좋게 나타났습니다. 특히 저중첩·비가시 영역 스트레스 테스트에서 외삽 상황에도 PSNR/SSIM/LPIPS가 안정적이고, depth reprojection error 및 free-space alignment proxy도 개선되어 ‘기하적으로 유용한 합성’이 가능함을 보여줍니다. 저해상도/희소 파노라마에서 SfM/SLAM 초기화가 실패할 때, 오프라인·배경(backgound) map refinement 컴포넌트로 활용될 수 있다는 점에서 실용적 의미가 큽니다.



### On-board Remote-Sensing Foundation Models for Unsupervised Change Detection of Disaster Events (https://arxiv.org/abs/2606.27018)
- **Prior Approaches**: 기존 원격탐사에서는 라벨 의존을 줄이기 위해 unsupervised change detection을 시도했지만, 희귀 재난은 대규모 정확 주석을 확보하기 어려워 성능과 범용성에 한계가 있었다. 또한 patch-based 방식은 다수 반복 추론이 필요해 온보드(자원 제약) 환경에서 지연과 연산량 문제가 커지는 편이다. 한편 RSFM은 다양한 지역·센서에 대한 사전표현을 제공하지만, 이를 위성 에지로 옮길 때의 메모리·전력·연산 제약이 구현 난제로 남아 있었다.

- **Core Contribution**: 논문은 ResNet 기반 RSFM 백본과 untrained FPN을 결합한 UDFPN(Unsupervised Detection Feature Pyramid Network)을 제안해, 연속된 통과 간 잠재공간의 미세한 의미 변화로 넓은 스펙트럼 이상을 탐지한다. 특히 학습 없이도 FPN의 구조적 유도편향을 활용해 공간적으로 일관된 임베딩을 만들고, 이미지 레벨 change map을 생성하는 데 목적이 있다. 또한 RSFM을 전용 모델 대신 사용함으로써 bespoke 학습·개발 부담을 줄이면서도 다양한 지형과 센서에서 일반화 성능을 노린다.

- **Technical Challenges**: 핵심 기술 과제는 딥 네트워크의 다운샘플링·수용영역 확대로 인해 공간 의미가 붕괴되면서, 이미지 수준에서 정교한 change map을 만들기 어려운 점이다. 이를 위해 논문은 학습된 FPN 대신 untrained, frozen FPN을 구조 집계(aggregator)로 사용해 백본 특징을 공간 격자에 재정렬하고, ResNet-50의 pyramid 중 해상도가 보존되는 단계(C2C2 수준) 출력을 임베딩으로 활용한다. 이후 변화 점수는 pre-event 임베딩 공간에서 국소 윈도우 내 가장 유사한 벡터를 찾는 cosine distance 기반의 semantic displacement 중심 메트릭으로 계산한다.

- **Empirical Impact**: Landsat-8 OLI의 5-이미지 시계열(사전 4장+사후 1장) 기반 8개 사건(화재·홍수·산사태)에서 AUPRC로 평가했으며, 전반적으로 UDFPN의 평균 성능이 경쟁 대안과 비교해 견줄 만하다고 보고한다. 특히 산사태에서 뚜렷한 우위가 관찰됐고, 홍수는 한 사건에서 수변 스펙트럼 변화처럼 구조 변화가 약한 경우 민감도가 떨어질 수 있음을 분석한다. 한편 UDFPN-18은 성능이 약간 낮아도 효율이 가장 높았으며, FPN이 연산 오버헤드를 일부 유발한다는 점을 실험적으로 시사한다.



### Event-Aware Instructed Assistant for Referring Video Segmentation (https://arxiv.org/abs/2606.26994)
Comments:
          IEEE Transactions on Image Processing

- **Prior Approaches**: 기존 referring video object segmentation(RVOS) 방법들은 영상을 여러 프레임의 집합처럼 보고 단일 이벤트로 취급하는 경향이 강했습니다. 이 경우 모델이 영상-텍스트의 복잡한 내용을 한 번에 처리해야 해서 혼동과 hallucination이 커질 수 있고, 동작 표현이 실제로는 여러 사건으로 분해되는 구조를 반영하기 어렵습니다. 또한 LLM 기반 접근은 특히 장면이 길고 사건이 많은 상황에서 정렬이 흔들릴 위험이 있습니다.

- **Core Contribution**: EVIS는 Event Query로 영상을 ‘단순 이벤트들의 묶음’으로 분해하고, 이벤트별로 단계적(event-by-event) 이해를 수행하도록 설계된 Event-Aware Video Instructed Segmentation Assistant입니다. 이를 통해 compound event 내부의 서로 다른 텍스트 관련 구간을 계층적으로 정리하며, Event-Aware Frame Merging Module(EAFM)로 이벤트 단위 시공간 정보를 통합합니다. 장기 추적을 위해서는 Object-Pixel-Hybrid Learning을 도입해 object-level query의 의미성은 살리되 pixel feature의 중복성 문제를 완화합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트 표현이 암시하는 사건 경계를 영상 쿼리로 어떻게 분해·할당할지, (2) 이벤트 내 단기 동학과 이벤트 간 장기 궤적을 동시에 잡을지, (3) pixel feature의 과도한 중복을 계산 효율과 성능 저하 없이 다룰지입니다. EVIS는 이벤트 쿼리-오브젝트 쿼리 간 top-kk 기반 프레임 병합에 Gumbel Softmax와 stop-gradient 트릭을 사용해 end-to-end 학습이 가능하도록 했고, EAFM 내부에서 Event-Intra Attention과 Event-Inter Attention을 분리해 단기/장기 의존을 함께 모델링합니다. 더불어 하이브리드 유닛으로 pixel feature와 object query를 함께 LLM 입력에 interlace하여 다단계 상호작용을 유도합니다.

- **Empirical Impact**: EVIS는 MeViS를 포함한 5개 RVOS 벤치마크에서 강한 성능을 보였고, 특히 까다로운 MeViS에서 46.8% J&F 성과를 포함해 유의미한 향상을 입증했습니다. 또한 LLM/SAM을 제거한 단순 베이스라인을 통해 EAFM이 LLM 의존 없이도 이벤트 중심 분해와 정렬에 기여함을 확인합니다. 이벤트 인지 설계를 RVOS 학습 파이프라인에 본격 도입함으로써, 복잡한 동작 표현이 많은 embodied perception 및 비디오 편집 분야에서 보다 안정적인 영상-텍스트 정합의 새 기준점이 될 것으로 기대됩니다.



### Unison: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation (https://arxiv.org/abs/2606.26984)
Comments:
          ICML 2026

- **Prior Approaches**: 기존 unified multimodal models(UMMs) 평가는 이해(vision-language 이해)와 생성(image/video generation)을 서로 분리해 측정하는 경우가 많아, 두 능력 간 시너지 효과는 놓치기 쉽다. 일부 연구는 의미 일관성이나 생성 기반 점검을 제안했지만, 공동으로 일어나는 오류 전파·자기 교정 같은 통합 현상을 정량화하기엔 한계가 있었다.

- **Core Contribution**: 이 논문은 이해와 생성의 결합을 직접 평가하기 위한 benchmark Unison을 제안하며, 2,169개의 human-validated unified task 샘플로 internal consistency, understanding-guided generation, generation-guided understanding, mutual enhancement 네 축을 구성한다. 또한 사람 판단과 정렬된 평가 모델 Unison-Judge를 도입해 모델 성능을 더 신뢰도 있게 채점하도록 한다.

- **Technical Challenges**: 핵심 과제는 ‘둘이 동시에 맞는지’만이 아니라, 이해가 생성 품질을 어떻게 좌우하는지/반대로 생성이 이해를 어떻게 바꾸는지를 분해해 측정하는 설계다. 논문은 attribute-정렬 기반의 internal consistency(오인식 전제에서의 spurious consistency를 0점 처리), 이해가 위치/편집을 안내하는 방식(ROI localization+편집 점수 결합), 생성으로 시각 맥락을 만든 뒤 VQA로 검증하는 generation-guided understanding, 그리고 다중 라운드 self-refinement 루프(misalign detect-and-correct)를 통해 시너지를 정량화한다.

- **Empirical Impact**: Unison 평가에서 다수 모델은 개별 이해/생성 벤치마크에선 강해도 unified 과제에선 성능이 흔들리며, 이해와 생성 능력이 항상 internal consistency로 직결되진 않는다는 점이 드러난다. 예를 들어 mutual enhancement에서는 Omnigen2가 큰 Uni. 점수 향상을 보였지만 BAGEL이나 UniWorld는 전반 축이 약해 시너지가 제한됐고, generation이 이해를 악화시키거나(negative impact) 때로는 대비 경계로 모호성을 줄여 돕는 역설적 효과도 관찰됐다.



### Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning (https://arxiv.org/abs/2606.26973)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 open-set semi-supervised learning(OSSL) 연구는 (1) 의심 샘플을 필터링해 out-of-distribution(OOD) 잡음을 줄이거나, (2) pseudo label 기반으로 unlabeled를 활용해 학습 신호를 최대화하는 두 흐름으로 정리됩니다. 그러나 필터링은 hard in-distribution(ID) 샘플까지 버려 feature starvation을 부르고, 활용은 잘못된 분류/unknown 할당이 supervised 학습 방향과 충돌하는 그래디언트를 만들어 ID 성질을 약화시킬 수 있습니다.

- **Core Contribution**: 이 논문은 OOD/ID를 샘플 단위로 정확히 가르는 데서 오는 한계를 ‘그래디언트 충돌’ 문제로 재정의합니다. 이후 sample selection을 줄이고, supervised gradient를 기준(anchor)으로 삼아 auxiliary 업데이트가 충돌하지 않도록 그래디언트 공간에서 보정하는 Geometric Gradient Rectification(GGR) 프레임워크를 제안합니다. 또한 supervised 방향을 부분공간으로 요약해 잡음이 큰 mini-batch 상황에서도 기준(anchor)이 흔들리지 않게 하는 subspace-aware rectification(OSR/CSR)도 함께 확장합니다.

- **Technical Challenges**: 핵심 기술 과제는 pseudo label이 틀려도 auxiliary gradient가 supervised 진행을 1차적으로 방해하지 않게 만드는 비대칭(선행 supervised 고정) 제어입니다. GGR은 supervised gradient가 만드는 허용 반공간(또는 부분공간의 조건)으로 auxiliary gradient를 projection 하여, 직교 성분은 보존하면서 충돌 성분만 제거하도록 설계했습니다. 더 나아가 OSR/CSR은 최근 supervised descent 방향을 저차원 직교기저로 축적해 anchor의 노이즈 민감도를 완화하고, 충돌이 감지될 때만 안정적으로 rectification을 적용합니다.

- **Empirical Impact**: CIFAR 및 ImageNet의 open-set 벤치마크에서 다양한 설정(라벨 수/seen-unseen 클래스 분할 등)에 대해 GGR이 대표적인 OSSL baseline을 대부분의 경우 개선했습니다. 특히 closed-set generalization 성능 향상과 open-set robustness 개선이 함께 관측되며, 샘플을 버리거나 forward/손실 설계를 바꾸지 않고도 성능 이득을 얻는 점이 실용적으로 의미가 큽니다. 코드 공개를 예고하며, OSSL에서 ‘정확한 OOD 탐지’보다 ‘그래디언트 충돌 제어’가 효과적일 수 있음을 경험적으로 뒷받침합니다.



### Computer Vision for MOBA Analytics: A Dataset and Baseline for Visibility Analysis in Dota 2 (https://arxiv.org/abs/2606.26970)
Comments:
          Accepted for presentation at the 2026 Simpósio Brasileiro de Jogos e Entretenimento Digital (SBGames)

- **Prior Approaches**: 기존 MOBA 분석은 드래프트, 이벤트 로그, 리플레이 파일, public API 같은 구조화 데이터에 크게 의존해 승패 예측, 조합 분석, 이벤트 예측 등을 수행해 왔다. 하지만 좌표는 영웅 위치를 뜻할 수는 있어도 상대가 실제로 볼 수 있었는지(시야/안개전쟁)는 직접 복원하기 어렵다. 시야는 지형·나무·고도·와드·일시 능력 등 맥락 의존 요인에 의해 결정되기 때문이다.

- **Core Contribution**: 본 논문은 프로 Dota 2 경기에 대해 영상 기반 시야(상대가 보았는지)를 분석할 수 있는 Dota2-Vis 데이터셋과 기준 파이프라인을 제안한다. The International 2025 전체 144경기를 양 팀 관점으로 기록한 288편의 비디오와, 미니맵에 대한 2,477장의 수동 라벨(플레이어 아이콘/클론/기타)을 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 미니맵에서 작은 아이콘들을 시야 단서에 영향을 주는 클론·핑·시각적 혼잡 상황에서도 안정적으로 검출하는 것이다. 저자들은 YOLO11 계열을 비교해 YOLO11l(large)을 최적 모델로 선택했으며, 이를 바탕으로 상대 관점 영상에서 각 플레이어의 존재를 시간 구간별로 집계해 가시성 곡선(플레이어/영웅/역할/팀)을 산출한다.

- **Empirical Impact**: 실험에서 YOLO11l이 F-score와 mAP50:95에서 가장 좋은 성능을 보여 밀집·복잡한 미니맵에서도 아이콘을 비교적 신뢰성 있게 찾았다. 가시성 곡선 분석 결과, 같은 영웅·역할 내 선수 간 행동 차이, 영웅/역할별 활성 타이밍, 그리고 20분 이후 승리 팀과 패배 팀의 시야 트렌드 분기가 구조화 데이터만으로는 얻기 어려운 패턴으로 드러났다. 저자들은 데이터셋과 코드도 공개해, 향후 live win prediction 등 기존 분석에 시야 기반 신호를 보강하는 출발점이 될 것으로 기대된다.



### Scaling Multi-Reference Image Generation with Dynamic Reward Optimization (https://arxiv.org/abs/2606.26947)
Comments:
          Accepted by ECCV2026

- **Prior Approaches**: 개인화 이미지 생성에서 MRIG는 여러 참조 이미지(대상, 포즈, 스타일 등)를 일관되게 결합하는 데 어려워 여전히 난제로 남아 있다. 기존 벤치마크는 참조 이미지 수나 유형 조합이 단순한 경우가 많고, 평가 지표도 단일 축에 치우쳐 복잡 시나리오를 제대로 가늠하기 어렵다. 그 결과 방법 간 공정한 비교가 제한되고, 실제 요구되는 고난도 MRIG 성능 개선이 더딘 편이다.

- **Core Contribution**: 이 논문은 복잡 MRIG를 정량·정성으로 평가할 OmniRef-Bench를 제안한다. 대상, 배경, 스타일, 조명, 포즈의 5가지 참조 유형을 2~7장 범위로 조합하고, 최대 4개 유형이 동시에 등장하는 10종 조합을 포함해 어려운 시나리오를 촘촘히 재현한다. 또한 DyRef라는 2단계 학습 프레임워크로, 참조 유형이 섞이고 개수가 많아질수록 급격히 성능이 떨어지는 문제를 완화한다.

- **Technical Challenges**: 핵심 기술적 문제는 참조 유형이 혼합되고 개수가 늘어날수록 학습 신호가 약해지며(난이도에 따른 최적화 불균형), 보상 모델 점수도 CLIP/SigLIPv2 기반에서 값이 뭉쳐 그라디언트 대비가 부족해진다는 점이다. 이를 위해 DyRef는 1단계 SFT로 기본 MRIG 처리 능력을 만든 뒤, 2단계에서 DAR(Difficulty-aware Advantage Reweighting)로 ‘더 못하는 어려운 샘플’ 비중을 동적으로 키운다. 이어 DRS(Discriminative Reward Scaling)로 보상 값의 차이를 확대해 정책 최적화의 수치적 대비를 높인다.

- **Empirical Impact**: OmniRef-Bench 평가에서 주류 오픈소스 모델들은 복잡 MRIG에서 큰 폭으로 성능이 떨어지며, 혼합형 참조 이미지 수가 늘수록 하락이 가파르게 관측된다. DyRef를 적용한 Qwen-2511은 OmniRef-Bench에서 오픈소스 대비 크게 향상되며 스타일·포즈 관련 지표에서도 특히 개선이 두드러진다. 아울러 단일 이미지 편집 벤치마크에서도 성능이 유지·상승하며, 파라미터 규모가 다른 백본(예: FLUX.2)에도 일반화 효과가 관찰되고 사용자 선호도와의 상관도도 높게 나타난다.



### TraMP-LLaMA: Generative Interpretability with Decoupled Instruction Tuning for Facial Expression Quality Assessmen (https://arxiv.org/abs/2606.26942)
- **Prior Approaches**: 기존 비디오 기반 표정 품질 평가(FEQA)는 대부분 단일 severity score만 산출해, 예측을 뒷받침하는 관찰 가능한 얼굴 움직임 근거를 함께 제공하지 못했다. 따라서 Parkinson’s disease 같은 신경질환 평가에서 모델 출력의 해석 가능성과 근거 감사(audit)가 어렵다. 또한 saliency map 등 사후 해석은 어떤 얼굴 영역이 영향을 줬는지는 보여주더라도, 임상적으로 중요한 시공간 운동 패턴(예: 움직임 범위 감소, 지속적 깜빡임, 미세한 비대칭)을 설명하기엔 한계가 있다.

- **Core Contribution**: TraMP-LLaMA는 severity score를 예측하는 동시에, 얼굴 모션 단서로부터 구조화된 텍스트 리포트를 생성해 ‘점수의 근거’를 외부화한다. RGB 외형과 landmark trajectory의 motion cue를 함께 사용하며, 리포트 생성은 evidence token을 기반으로 하도록 설계해 점수와 분리된 “관찰 진술” 형태의 보고를 목표로 한다. 더불어 PFED5에 전문가 주도 텍스트 모션 설명을 추가한 PFED5+를 구축해, 점수 누출(label leakage)을 줄이면서도 리포트 감독을 가능하게 했다.

- **Technical Challenges**: 핵심 기술 난제는 severity regression과 언어 생성 학습을 함께 최적화할 때 motion encoder와 융합 모듈이 두 태스크 간 간섭으로 인해 시공간 특징 학습이 왜곡될 수 있다는 점이다. 이를 위해 decoupled instruction-tuning을 도입해, 언어 생성 목적의 그래디언트는 motion encoder/융합 모듈 업데이트에서 차단하고 점수 예측 목적만 해당 모듈을 갱신하도록 구성했다. 또한 리포트 감독을 위해 pre/in/post의 3단계 템플릿과 구획별 관찰 슬롯(움직임 유무, 방향 변화, 좌우 비대칭, 가시 불가 placeholder)을 엄격히 강제해 환각과 누락 해석을 최소화했다.

- **Empirical Impact**: PFED5-plus 실험에서 TraMP-LLaMA는 리포트 생성 과업에서 경쟁 video-language baseline을 능가하고, severity 예측에서도 비교 방법 중 최상 성능을 보이며 Spearman의 rank correlation을 최소 4.39%p(모든 경쟁 방법 대비) 개선했다. 특히 joint multi-expression training 조건에서도 성능 우위를 유지해, 단일 점수 모델 대비 임상적 검토에 유리한 “근거 기반” 출력을 제공하는 방향성이 입증됐다. 텍스트 주석과 코드가 공개되어 후속 연구에서 신경질환 FEQA의 해석 가능성 및 데이터 기반 리포팅 학습을 확장하는 데 의미가 크다.



### Focusing on What Matters: Saliency-Harnessing Accurate Routing for Diffusion MoE (https://arxiv.org/abs/2606.26938)
Comments:
          ECCV 2026

- **Prior Approaches**: 확산 모델의 MoE는 라우터가 각 토큰에 대해 일부 expert만 선택하도록 해 효율을 높였지만, 비전 MoE에서는 라우팅이 text와 달리 불안정해 성능 격차가 발생해왔다. 특히 DiffMoE류의 동적 라우팅은 salient 토큰에 더 많은 compute를 배분하려 하지만, 실제로는 noise에 의해 라우팅이 saliency에 둔감해진다는 분석이 제시된다. 기존 방법들은 생성 과정의 잔여 잡음이 라우터의 구분 능력을 흐리면서 expert 할당이 흔들리는 문제를 충분히 해결하지 못했다.

- **Core Contribution**: SharpMoE는 post-training 프레임워크로, 라우터가 noisy latent가 아니라 clean latent 기반의 saliency guidance를 보게 함으로써 noisy routing 문제를 정면으로 겨냥한다. 구체적으로 이전 timestep의 x0 예측(깨끗한 추정치)을 라우팅 입력으로 사용해, 초기 고잡음 구간에서도 salient 토큰을 더 정확히 식별하도록 설계했다. 또한 multi-step 생성 궤적 전반의 compute 할당을 제약하는 Trajectory Routing Loss를 도입해, 생성 rollout 내내 saliency에 맞춘 expert 배분이 유지되게 한다.

- **Technical Challenges**: 핵심 기술적 난제는 라우터가 clean latent(x0 예측)를 사용하려면 시간축 재귀 의존성이 생기는데, 기존 single-step 학습만으로는 그 입력을 학습에 제공하기 어렵다는 점이다. SharpMoE는 Recursive Full-Trajectory Training으로 여러 timestep을 롤아웃하며 이전 단계의 clean latent가 다음 단계 라우팅에 실제로 반영되도록 학습을 구성한다. 또 추론 초기 timestep에서는 clean latent가 없으므로 noise latent를 프록시로 쓰며, Trajectory Routing Loss로 전체 궤적에서 expert 할당 분포가 Laplacian 기반 saliency 분포(KL divergence)와 정렬되게 만든다.

- **Empirical Impact**: ImageNet 256×256(256×256) 조건에서 TC-DiT, EC-DiT, DiffMoE 등 기존 diffusion MoE 대비 SharpMoE가 모든 모델 스케일(S/B/L)과 평가 지표(FID, IS)에서 일관된 성능 향상을 보였다고 보고한다. 특히 pretrained MoE 체크포인트에 대한 plug-and-play post-training 방식으로도 효과가 지속되며, post-training 100K 단계에서도 유의미한 개선이 나타난다. 저자들은 saliency 기반 clean routing과 trajectory-level 제약이 결합될 때 확산 MoE의 라우팅 품질이 실질적으로 개선되어 SOTA급 비주얼 생성 성능으로 이어진다고 정리한다.



### PortraitGen: Exemplar-Driven GRPO with Dual-Reward Guidance for Photorealistic Portrait Generation (https://arxiv.org/abs/2606.26930)
- **Prior Approaches**: GRPO 기반 텍스트-투-이미지 post-training은 group relative policy optimization으로 보상모델 점수를 높이는 데 강점을 보여 왔지만, 실제 사진 분포를 최적화 과정에서 직접 관찰하지 못하는 구조적 한계가 있습니다. 그 결과 생기는 불일치가 과채도 색감 같은 겉보기 변화로 이어지고, 손/손가락·사지 비틀림, 피부의 과도한 윤기(oily skin) 같은 세밀한 AI artifact는 잘 교정되지 않는 경우가 많습니다. 초상화 분야에서도 identity 커스터마이징에는 집중되지만, 생체적 그럴듯함(예: 생물학적 불가능 형태)과 artifact 억제를 동시에 다루는 전용 보상 신호가 부족했습니다.

- **Core Contribution**: 이 논문은 PortraitGen을 통해 photorealistic portrait 생성에서 AI artifact를 억제하는 post-training 프레임워크를 제안합니다. 핵심은 (1) GRPO sampling group에 실사(real image) 예시를 직접 포함해 ‘생성 경계’를 깨고, (2) 일반 품질을 평가하는 OmniReward와 AI artifact/피부 윤기를 겨냥한 AI-Portrait reward의 듀얼 보상으로 모델을 정교하게 유도하는 것입니다. 또한 초상화 전용 벤치마크 PortraitBench를 구축해 artifact 중심의 평가 공백을 메웁니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 실사 예시를 GRPO 학습 루프에 넣더라도, 해당 예시에 대한 단계별 latent과 전이 확률을 알아야 gradient backpropagation이 가능하다는 점입니다. 이를 위해 BELM(Bidirectional Explicit Linear Multistep method) 기반 image inversion으로 예시의 중간 latent 및 단계별 trajectory 확률을 복원하고, real exemplar와 함께 합성 샘플들을 한 group에 넣어 OmniReward(4차원 품질)와 AI-Portrait reward(쌍대 win-rate로 artifact 페널티)를 계산합니다. 그 결과 group 내부 비교 신호가 미세한 윤기·구조 왜곡 차이를 반영하도록 설계됐습니다.

- **Empirical Impact**: PortraitGen은 PortraitBench 및 다양한 평가에서 기존 baseline 대비 높은 성능(대부분의 지표에서 1~2위를 기록)을 보이며, 특히 OmniReward의 Content 항목에서 합성 흔적을 크게 줄이는 효과가 확인됩니다. 정성 결과에서도 baseline들이 기름진 피부나 애니/오일페인팅 같은 스타일로 수렴하는 경향을 PortraitGen이 현저히 완화하고, 구조적 사지 왜곡과 oily skin을 집중적으로 억제합니다. 사용자 연구에서도 현실감·심미성·텍스트 정합성 전반에서 우위를 보여, 포트레이트 생성 품질을 ‘겉보기’가 아닌 artifact 억제 중심으로 끌어올리는 의미 있는 진전을 제시합니다.



### PhysRAG: Enhancing Physics-Awareness in Video Generation via Retrieval-Augmented Generation (https://arxiv.org/abs/2606.26916)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 텍스트-투-비디오(T2V) 확산모델은 현실감은 빠르게 개선됐지만, 열역학·역학·광학 같은 물리 법칙을 일관되게 지키기 어려웠다. 기존 물리 인식 접근은 시뮬레이터/수식 기반의 명시적(explicit) 방법과 물리 제약을 잠재공간에 학습시키는 암묵적(implicit) 방법으로 나뉘며, 전자는 복잡한 오픈월드 일반화가, 후자는 정밀한 물리 제어가 취약했다.

- **Core Contribution**: 이 논문은 PhysRAG로, Retrieval-Augmented Generation(RAG)을 결합해 “물리적으로 유사한 실제 비디오”를 검색하고 그 물리 역학을 생성 과정에 주입하는 파이프라인을 제안한다. 특히 Learnable queries를 통해 검색된 물리 참조에서 핵심 역학만 골라 Video DiT에 주입해, 불필요한 시각 노이즈는 걸러내도록 설계했다.

- **Technical Challenges**: 핵심 난제는 고품질·다양한 물리 현상을 담은 데이터의 희소성과, 검색 참조가 만들어내는 불일치/잡음을 제어하는 것이다. 이를 위해 WISA-80K 기반 2단계 데이터 필터링(캡션 품질·물리 관련성 거친 필터, 텍스트-비디오 grounding 정밀 검증)로 약 7K 고품질 셋을 만들고, 물리 정보를 VideoMAE-V2로 추출한 뒤 Query Inject 모듈에서 cross-attention 및 정보병목으로 물리 priors만 정제해 DiT 블록에 gated residual로 주입한다.

- **Empirical Impact**: 실험에서는 PhyGenBench와 VBench에서 시각 품질과 물리 규칙 준수 모두를 개선하며, 평균 점수 기준 최신 성능을 달성했다고 보고한다. 예컨대 PhyGenBench에서 PhysRAG는 0.58(기존 Kling 0.49)로 SOTA를 기록했고, VBench에서는 Low-Avg 62.10%→65.48%, High-Avg 81.18%→82.88% 등 전반 지표가 상승했다. 데이터 필터링·RAG·Query Inject 구성요소 및 joint training의 효과를 ablation으로 검증해, 물리적 일관성이 실제로 강화됨을 보여준다.



### Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation (https://arxiv.org/abs/2606.26907)
- **Prior Approaches**: 기존 text-to-image(T2I) 연구는 생성 품질 자체나 정렬, 혹은 단일 능력(지식/추론/검색/메모리)을 부분적으로 다루는 경향이 강했습니다. 에이전트형 접근도 plan, reason, search, memory, feedback 같은 모듈이 흩어져 연구돼 왔지만, “생성에 필요한 맥락”을 체계적으로 구성하는 관점은 부족했습니다. 그 결과 실제 서비스의 underspecified 요청에서 성능이 흔들리는 구조적 원인이 해석과 해결 측면에서 분리돼 있었습니다.

- **Core Contribution**: 논문은 실사용 요청과 모델이 필요로 하는 생성 맥락 사이의 불일치를 Context Gap으로 규정하고, 이를 해결하는 맥락 중심의 에이전트 프레임워크를 제안합니다. Qwen-Image-Agent는 user context를 최종 조건으로 취급하지 않고, Context-Aware Planning과 Context Grounding을 통해 생성 맥락을 단계적으로 완성합니다. 또한 Plan, Reason, Search, Memory 역량을 포괄적으로 평가하는 Image Agent Bench(IA-Bench)도 함께 공개합니다.

- **Technical Challenges**: 핵심 기술 난점은 “무엇이 빠졌는지”를 안정적으로 찾아내고, 그 빈칸을 어떤 근거(추론/웹 검색/이미지 검색/메모리/피드백)로 메울지 결정하는 과정입니다. 이를 위해 정보-콘텐츠-생성 단계로 나눈 Context-Aware Planning으로 누락 정보를 질문 형태로 드러내고, Reason은 암묵적 의도를 명시화, Search는 최신 사실/IP 시각 레퍼런스를 외부에서 보강, Memory와 Feedback은 멀티턴 일관성과 반복 교정을 담당하게 구성했습니다. 아울러 멀티턴/멀티이미지에서도 맥락 길이와 의존성을 관리하도록 설계해, 생성 드리프트와 붕괴 위험을 줄였습니다.

- **Empirical Impact**: 실험에서는 IA-Bench뿐 아니라 WISE-Verified, MindBench에서도 Qwen-Image-Agent가 강한 베이스라인을 상회하며 SOTA 성능을 보였다고 보고합니다. 특히 direct generation 기준 Qwen-Image-2.0 대비 에이전트 프레임워크가 Q-score를 크게 끌어올렸고, Memory 차원에서의 개선 폭이 두드러져 실사용 멀티턴 가치가 강조됩니다. 체크리스트 기반의 IA-Bench 평가와 제거형(ablation) 분석은 Reason/Search/Memory/Feedback의 제거 시 해당 차원 성능이 감소함을 보여, 제안된 “생성 맥락 구성”이 실증적으로 유효함을 뒷받침합니다.



### Confidence-Aware Tool Orchestration for Robust Video Understanding (https://arxiv.org/abs/2606.26904)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 비디오 추론 언어모델은 입력 프레임을 전부 동일하게 신뢰한다고 가정해, 모션 블러·눈부심·가림 같은 현실적 열화가 발생해도 모델이 증거 열화 여부를 잘 감지하지 못하는 문제가 있었다. 그 결과 최첨단 비디오 추론 모델들이 embodied 벤치마크에서 15-30%p 정확도 하락을 겪고도 스스로는 “왜 틀렸는지”에 대한 신뢰도 붕괴를 인지하지 못하는 Blind Trust Problem이 관찰된다.

- **Core Contribution**: 이 논문은 프레임별 신뢰도(trustworthiness)를 추론 전 과정에 명시적으로 통합하는 에이전틱 비디오 이해 프레임워크 Robust-TO를 제안한다. 서로 다른 시각 지각 도구들을 하나의 증거 인터페이스로 묶고, 각 도구가 신뢰도와 관련성에 따라 선별된 신뢰 가능한 프레임에서 얻은 증거만 사용하도록 설계해 열화 상황에서도 일관된 추론을 유도한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 프레임 신뢰도와 작업 관련성을 함께 반영해 “무엇을 볼지”를 정하고, (2) 서로 다른 도구가 산출하는 증거들을 시간적 근거와 함께 공통 포맷으로 정렬하며, (3) 추론 단계에서 신뢰도를 가중치로 반영해 학습 목표까지 연결하는 것이다. Robust-TO는 reliability-relevance score로 trustworthy frames를 선택하고, 도구별 결과를 bounding box·trajectory·인식 텍스트·action label 등 공통 증거 형식(예측, temporal grounding, calibrated reliability score)으로 반환한 뒤, 3단계(high/medium/low) 합성 가중치와 confidence-cost GRPO reward로 정확도·신뢰도·효율을 함께 최적화한다.

- **Empirical Impact**: 실험에서 Robust-TO는 8개 태스크를 포함한 두 비디오 추론 벤치마크에서 클린 입력 기준 56.4% 평균 정확도를 기록해 최강 오픈소스 대비 10.6%p, Gemini-2.5-Pro(46.2%)도 능가한다. 또한 5가지 현실적 부정(corruption) 조건에서도 평균 54.3%로 최강 오픈소스 대비 5.8%p 우위이며, 방법들 중 클린-부정 간 정확도 하락 폭이 가장 작아 강건성의 의미 있는 개선을 보여준다.



### Tractography-Driven Synthetic Data Generation for Fiber Bundle Segmentation in Tracer Histology (https://arxiv.org/abs/2606.26898)
Comments:
          MICCAI 2026

- **Prior Approaches**: dMRI tractography는 백질 경로를 비침습적으로 재구성하지만, 물 분자 확산 같은 간접·저해상도 신호 때문에 개별 축삭 조직을 직접 반영하긴 어렵습니다. 원숭이 tracer histology는 검증의 골드 스탠드이지만, 히스토로부터 fiber bundle을 수작업으로 라벨링해야 해서 데이터 확보가 병목이었습니다. 딥러닝을 쓰더라도 macaque tracer 데이터의 분포 변화(트레이서/주입 부위/뇌 간 변이) 때문에 라벨 부족과 일반화 한계가 반복됐습니다.

- **Core Contribution**: 이 논문은 tracer histology에서 fiber bundle 자동 분할을 위한 synthetic-data augmented 학습 프레임워크를 제안합니다. ex vivo dMRI tractography를 생성적 prior로 활용해 2D 이미지 패치와 마스크를 합성하고, blockface 배경을 결합한 뒤 domain randomization으로 변이를 확대합니다. 2D U-Net을 real+synthetic 혼합으로 학습해 브레인 간 밀도 변화에도 강건한 분할을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 과제는 합성 이미지가 실제 tracer의 국소 질감과 희미한 bundle 분포를 충분히 닮아야 한다는 점입니다. 이들은 dMRI streamline을 coronal plane에 투영해 tracer-like 곡선 전경을 렌더링하되, 마스크는 dense streamline 기반으로 더 넓게 구성해 부분 관측 상황에서도 bundle을 찾도록 유도했습니다. 또한 합성은 필요할 때만 on-the-fly로 생성해 다양성을 확보하고, real 데이터는 class imbalance를 고려해 foreground-aware로 샘플링하며, 합성-only 학습의 실패를 보완하기 위해 mixed training 비율을 조절했습니다.

- **Empirical Impact**: 실험에서 제안 방법은 held-out 뇌 M4에서 SOTA 대비 전반 탐지 성능을 유지하거나 소폭 개선하면서, 한 뇌(annotations)만으로도 수행합니다(기존 대비 3x 적은 수작업 라벨). 특히 sparse bundle 민감도에서 real-only 대비 큰 폭으로 향상되며, 브레인 간 일반화에서도 moderate/sparse에서 TPR이 개선되는 경향을 보였습니다. 다만 일부 held-out에서 FPs가 늘어나는 도메인 갭은 남았지만, 전체적으로 정밀도-재현율 균형 측면에서 실용적 수준의 성능을 입증했다는 점이 의미 있습니다.



### Modeling Local, Global, and Cross-Modal Context in Multimodal 3D MRI (https://arxiv.org/abs/2606.26894)
- **Prior Approaches**: 기존 멀티모달 뇌 MRI 학습은 CNN이나 transformer에서 early fusion/late fusion처럼 결합 방식을 정형화하되, intra-modality(모달리티 내부)와 cross-modality(모달리티 간) 처리를 명확히 분리·계층적으로 모델링하지 못하는 경우가 많았습니다. 또한 2D 기반 전개나 단순 채널/슬라이스 결합은 3D 전역 문맥과 복수 모달리티 간 정밀 상호작용을 충분히 포착하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: MICViT(Multimodal Intra- and Cross-Context Vision Transformer)는 3D 비전 transformer로서 모달리티별 표현 학습과 모달리티 간 상호작용을 로컬·글로벌 문맥 모두에서 명시적으로 다룹니다. 이를 위해 SL(Separated Local)·SG(Separated Global)로 모달리티 내부를, CL(Cross Local)·CG(Cross Global)로 모달리티 간 상호작용을 각각 설계해 단일 구조 안에서 함께 학습시키는 것이 핵심입니다. 

- **Technical Challenges**: 3D 멀티모달 MRI는 고차원·작은 표본·다양한 획득 조건 때문에 학습이 불안정해지기 쉬운데, MICViT는 SL/SG로 간섭을 줄이고 CL/CG로 구조화된 융합을 수행하도록 attention을 분해 설계했습니다. 또한 window 기반 3D relative positional encoding과 SG·CG의 쿼리 생성 방식을 결합해 로컬 세밀도와 전역 문맥을 동시에 확보하도록 구성했으며, 학습 안정화를 위해 Schedule-Free Learning, GSAM, mimetic initialization 등을 적용했습니다.

- **Empirical Impact**: UK Biobank(41,404), SOOP(1,062), Cam-CAN(613) 등 3개 이질적 데이터셋에서 MICViT는 brain age prediction에서 3D CNN·transformer 기준선을 일관되게 능가했습니다. 특히 모달리티 수를 늘릴수록 성능 향상이 더 크게 나타나, MICViT가 멀티모달 보완 정보를 더 효율적으로 활용한다는 점이 실험적으로 확인됩니다. 데이터 규모가 작을수록 이득이 커져 sample efficiency 관점에서도 의미가 크며, 3D 입력 해상도 및 모델 스케일링에서도 강건한 성능이 보고됐습니다.



### Bridging Vision and Language Concepts through Optimal Transport Semantic Flow (https://arxiv.org/abs/2606.26891)
- **Prior Approaches**: Concept Bottleneck Models(CBMs)는 중간에 인간이 해석 가능한 개념을 예측한 뒤 이를 바탕으로 분류하지만, 비전-언어 환경에서는 개념 정렬이 성능을 좌우한다. 기존 접근은 주로 pre-aligned encoders나 global cosine similarity처럼 정적인 유사도에 의존해, 개념의 미세한 위치/대응과 실제 의미 기하를 흐리게 만든다. 또한 고정된 투영이나 전역 특징 기반 추론은 region evidence의 국소 근거를 약화시켜 “개념 충실도”를 저해할 수 있다.

- **Core Contribution**: 이 논문은 개념 정렬을 “정적 투영”이 아니라 동적인 cross-modal transport 과정으로 재정의하고, Optimal Transport Flow Concept Bottleneck Model(OTF-CBM)을 제안한다. OTF-CBM은 시각 패치(또는 prototype)와 텍스트 개념 사이의 의미 전이를 학습된 기하 위에서 모델링해, 해석 가능한 개념 추론을 목표로 한다. 특히 Inverse Optimal Transport(IoT)로 데이터 기반 cost를 배우고, Unbalanced OT로 many-to-one 및 배경/누락 대응 문제를 자연스럽게 다룬다.

- **Technical Challenges**: 핵심 기술 난제는 (1) patch-개념에 대한 미세한 대응을 주석 없이 찾아내고, (2) 그 대응이 시간에 따라 의미적으로 어떻게 “흘러”야 하는지 해석 가능한 형태로 결합하는 것이다. 저자들은 K-means로 patch를 prototype으로 집계한 뒤 Unbalanced OT로 시각-개념 coupling을 만들고, IoT로 cosine/유클리드 같은 고정 거리 대신 실제 대응을 잘 설명하는 cost landscape를 학습한다. 이어 Flow Matching(FM)으로 velocity field를 학습하되, inference에서 ODE integration 없이 속도(velocity)와 이상적인 개념 방향의 일치도를 통해 개념 activation을 계산한다.

- **Empirical Impact**: 실험에서는 OTF-CBM이 분류 정확도와 concept faithfulness(개념 충실도)에서 기존 CBM 계열을 능가함을 보여준다. 또한 learned semantic flow가 실제로 시각 근거(객체 중심)에 의존하며, 배경 치환이나 분포 변화 같은 조건에서도 강건하게 작동하는 경향을 확인한다. 요약하면, 정렬을 기하/동역학 관점에서 재구성한 OTF-CBM은 해석 가능한 cross-modal reasoning에 새로운 설계를 제공한다.



### RIS-Assisted Proactive Handover for Reliable mmWave Wireless Networks (https://arxiv.org/abs/2606.26885)
- **Prior Approaches**: mmWave 통신은 채널 희소성 때문에 지향성 LoS 의존도가 높아, LoS가 막히면 신호가 급격히 저하되는 문제가 크다. 이를 줄이기 위해 VAWC(vision-aided wireless communication) 기반 PHO( proactive handover )와 blockage 예측, 그리고 RIS-assisted 링크로 대체 접속을 시도하는 연구가 이어져 왔다. 다만 기존 RIS 관련 PHO/차단완화는 대부분 (1) 대체 BS가 항상 존재하거나 (2) 사용자/차단 정보를 이상적으로 알고, (3) 무엇보다 RIS 요소 수가 늘 때 생기는 구성 시간(configuration time)과 타이밍 제약을 PHO 준비 단계에 명시적으로 반영하지 못했다.

- **Core Contribution**: 본 논문은 대체 BS가 없을 때 RIS를 이용해 blocked LoS를 VLoS(virtual line-of-sight) 링크로 대체하는 RIS-assisted PHO 프레임워크를 제안한다. 핵심은 PHO 준비 단계의 타이밍에 RIS 구성 시간 Tc를 포함시키고, 주파수/신호품질을 유지하면서도 RIS 요소 수를 줄여 처리 복잡도와 에너지 소모를 함께 낮추는 것이다. 이를 위해 요소 수에 따른 링크 품질-복잡도-에너지의 균형을 최적화 문제로 정식화하고, PSO(particle swarm optimization)로 end-to-end RIS 링크 설정을 오프라인에 계산해 지연을 피한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘링크 품질을 위해 필요한 충분한 RIS 요소 수’와 ‘RIS 위상 쉬프터를 재구성하는 데 걸리는 시간’이 PHO의 제한된 창(window)을 동시에 만족해야 한다는 점이다. 논문은 RIS 컨트롤 레이어 기반으로 구성 시간에 영향을 주는 요소(시프트 레지스터 클로킹, 전체 코드워드 비트 수, 요소/다이오드 수 등)를 모델링해 Tc를 PHO 실행 지연 Texec에 편입하고, 구성 시간 반영 후의 waiting time까지 갱신한다. 또한 요소 할당은 중심 기준이 흔들리지 않도록 정사각 서브어레이 중심 확장 방식으로 제한해, 요소 수를 바꾸더라도 정렬/에이퍼처 효율 저하가 최소화되도록 설계했다.

- **Empirical Impact**: 실험/평가 결과, 최적화에서 RIS 요소 수를 12% 줄이면 SNR을 유지하면서 소모 에너지가 10% 감소했다. blocked region에서는 RIS-assisted 링크가 15–30 dB의 성능 향상을 보였고, 동시에 PHO 타이밍을 정확하게 맞추는 것으로 제시된다. 즉, RIS를 ‘가능하면 많이 쓰는’ 접근에서 벗어나, 구성 시간까지 포함한 시간-에너지-품질 트레이드오프를 만족하는 운용 가능 해법을 제공했다는 점에서 6G URLLC급 시간 민감 서비스에 의미가 있다.



### SpatialFlow-GRPO: Where Spatial Credit Drives Image Editing (https://arxiv.org/abs/2606.26872)
- **Prior Approaches**: Flow-GRPO/DanceGRPO 같은 온라인 RL 기반 이미지 편집 post-training은 편집 결과를 하나의 이미지로 보고, whole-image reward로 sample-level advantage를 만든 뒤 모든 latent position에 동일한 신호를 적용하는 경향이 있다. 이 방식은 공간적으로 비균일하게 발생하는 편집 품질 차이를 구분하지 못해, 어떤 영역이 성공/실패를 만들었는지에 대한 fine-grained credit assignment가 약해진다.

- **Core Contribution**: 이 논문은 공간적 균일성 가정의 한계를 짚고, 편집을 semantic region 단위로 쪼개 지역별 피드백을 학습하는 SpatialFlow-GRPO를 제안한다. SFReward(및 SFReward-14K)를 통해 region-aware reward를 만들고, 업데이트에서도 region advantage와 latent position의 대응을 정렬해 local feedback이 희석되지 않도록 설계했다.

- **Technical Challenges**: 핵심 기술적 과제는 ‘지역 점수’를 PPO/GRPO 목적함수에서 어떻게 잔존 잡음과 배경 면적의 영향까지 고려해 안정적으로 반영하느냐이다. 논문은 instruction/label 내부에서만 보상 비교를 정규화하고, region-level advantage를 해당 latent position에 매핑한 뒤 region 내에서는 묶어 집계하고, region 간에는 power-weighted aggregation으로 foreground 영향력을 조절하며 global quality anchor도 함께 유지하는 방식으로 해결했다.

- **Empirical Impact**: OmniGen2와 FLUX.2-klein-4B에서 SpatialFlow-GRPO는 Flow-GRPO 대비 GEdit-Bench, ImgEdit-Bench, MultiEditBench 전반에서 성능을 끌어올렸고, 특히 MultiEditBench의 multi-region 편집에서 이득이 더 크게 나타났다. 또한 SFReward 평가에서 dense region supervision이 편집 품질 판단 능력을 강화함을 보였으며, MultiEditBench는 2~5개 동시 편집 타깃에서 공간적으로 분리된 목표 수행을 진단하는 벤치마크로 의미를 가진다.



### Rolling Shutter Relative Pose Estimation Made Practica (https://arxiv.org/abs/2606.26863)
- **Prior Approaches**: 롤링 셔터(RS) 카메라에서 상대 자세 추정은 기존 GS(글로벌 셔터)처럼 고정된 essential matrix를 가정해선 안 되지만, 대표적인 RS-aware 방법은 최소점 대응을 크게 요구해 왔습니다. Dai et al.에 따르면 선형화된 RS 두-view 문제는 최소 20 point correspondences가 필요하며, RANSAC의 기대 반복 횟수가 표본 수 k에 대해 지수적으로 증가해 실사용 비용이 매우 커집니다. 그 결과 RANSAC 기반 강건 추정이 사실상 비현실적이었고, 대응 수를 줄이려는 시도들은 제약식 자체가 RS의 행(row) 의존성을 반영하지 못해 정확도나 안정성이 떨어졌습니다.

- **Core Contribution**: 이 논문은 RS 상대 자세 추정을 affine correspondences(ACs, 아핀 대응)를 통해 실용화합니다. AC는 포인트 1쌍당 epipolar 제약 1개에 더해 추가적인 2개의 아핀 제약을 제공해, RS 두-view의 필수 최소 표본을 20 PCs에서 7 ACs로 줄이는 경로를 제시합니다. 핵심은 RS에서 필연적으로 나타나는 ‘행(row) 의존 essential matrix’에 대한 영향을 반영하는 RS-corrected affine constraints를 새로 도출해, GS와 달리 생기는 보정 항(점 교란이 row 좌표를 함께 바꾸는 결합)을 제약식에 포함한 것입니다.

- **Technical Challenges**: 기존 GS용 아핀 제약은 essential matrix가 이미지 전반에서 일정하다는 가정 위에 세워져 있어, RS의 경우 점 perturbation이 해당 점의 readout row(τ1, τ2) 자체를 바꿔버리는 항이 누락됩니다. 저자들은 이 누락을 보정하는 RS-corrected affine constraints를 도출하고, RS 운동 파라미터가 물리적으로 작다는 점을 이용해 제약식을 θ(ω, v) 기준으로 1차 선형화합니다. 이어서 7개의 AC가 만드는 과결정 선형계에서 null-space projection으로 12개의 RS 미지수를 제거해 pose(5자유도)만 남기는 9개의 다항 조건을 구성하고, 남은 차수-20 시스템을 action matrix 방식으로 1.2ms 내에 풉니다.

- **Empirical Impact**: TUM-RS 벤치마크에서 제안 방법은 측정된 자세 및 RS 파라미터 정확도에서 테스트된 모든 방법 중 최고 성능을 보이며, 특히 RS 솔버 가운데 유일하게 translational velocity 추정도 정확하게 수행합니다. 이는 점 대응만으로는 v-t 결합 때문에 속도 항이 잘 조건화되지 않는다는 기존 한계를 AC와 RS 보정 제약이 함께 완화하기 때문으로 설명됩니다. 또한 EuRoC MAV의 global-shutter 데이터셋에서도 표준 5-point 알고리즘과 견줄 만한 정확도를 보여, RS-aware 솔버가 GS 설정으로도 무리 없이 일반화됨을 실증합니다.



### Liquid Fusion of Heterogeneous Representations Towards General Salient Object Detection (https://arxiv.org/abs/2606.26849)
Comments:
          20 pages, 5 figures

- **Prior Approaches**: 기존 SOD는 CNN 기반으로는 전역 의존성 한계가, Transformer 기반은 고해상도에서의 quadratic 복잡도 부담이 컸습니다. 최근에는 Mamba 계열 SSM이 선형 복잡도로 장거리 문맥을 제공하지만, 2D 표현을 만들기 위해 더 복잡한 scanning 경로를 설계하는 경향이 있어 중복과 최적화 난제가 남아 있습니다. 또한 여러 네트워크 패러다임이 주파수(스펙트럼)에서 보이는 서로 다른 편향을 충분히 고려하지 못했습니다.

- **Core Contribution**: 이 논문은 CNN과 SSM(특히 VMamba) 간 ‘스펙트럼 편향’이 상호보완적이라는 dataset-level 분석을 바탕으로, 두 표현을 harmonize해 general SOD의 표현 blind spot을 줄이는 LFNet을 제안합니다. 구체적으로 VMamba의 연속 상태 표현과 ConvNeXt의 격자 기반 공간 표현을 liquid fusion로 동적으로 결합해 단일-패러다임 한계를 메웁니다. 여기에 Saliency-Guided Upsampling(SGU)로 얕은 층까지 의미를 안정적으로 전파해 경계 복원 품질을 높입니다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 주파수 선호를 가진 백본을 단순 합치기가 아니라, 입력에 따라 어떤 정보를 더 주입할지 ‘동적’으로 조절하는 것이었습니다. 논문은 Liquid Neural Networks의 state-stimulus 관점을 차용해 LFM(Liquid Fusion Module)에서 VMamba를 evolving memory state로, ConvNeXt를 exogenous stimulus로 두고 gating(동적 permeability)으로 콘텐츠-인식형(feature aggregation) 결정을 내립니다. 또 업샘플링 과정에서 생길 수 있는 스펙트럼 알리아싱/경계 흐림을 줄이기 위해 SGU에 spectral-spatial co-design을 적용합니다.

- **Empirical Impact**: RGB, RGB-D, RGB-T, VSOD, VDT까지 5개 태스크에서 LFNet은 전반적으로 SOTA를 갱신하며 검출 정확도와 모델 효율의 균형이 좋다는 결과를 제시합니다. 예를 들어 RGB-D에서는 파라미터 수를 크게 줄이면서도 기존 강자 대비 SmS_m에서 큰 폭의 개선을 보였고, VDT에서도 새로운 SOTA급 수치를 달성했습니다. 또한 ablation에서 LFM의 동적 융합과 SGU의 업샘플링 설계가 각각 성능 향상에 기여함이 확인되어, ‘스펙트럴 편향 보정’이 실질적 이득으로 연결됨을 뒷받침합니다.



### Identifying the Unknown: Prompt-Free Open Vocabulary Anomaly Recognition for Robot-Object Interaction (https://arxiv.org/abs/2606.26829)
Comments:
          International Conference on Artificial Neural Networks 2026

- **Prior Approaches**: 기존 open vocabulary object detection은 prompted 방식이 주류로, 추론 시점에 후보 클래스 목록(프롬프트)을 요구하는 한계가 컸습니다. YOLO-World, ViLD, OWLv2 등은 일부 개방성을 제공하지만, 미지 객체에서의 오탐이 늘고 실제 로봇 운영에서는 “말로 지정한 것만 본다”는 제약이 남았습니다. prompt-free로 분류되는 YOLOE도 내부에 고정된 제한 어휘(참조 vocabulary)에 의존해 진정한 개방성에 상한이 있다는 지적이 있습니다.

- **Core Contribution**: AnomNOVIC은 known workspace를 전제로, 두 단계로 “탐지(클래스 비의존)”와 “분류(자유형 문구)”를 분리해 prompt-free 오픈 보카빌러리 인식을 구현합니다. 1단계 MAE(마스킹 오토인코더)가 로봇의 깨끗한 작업대(테이블)를 기준선으로 재구성하며 이상 영역을 bounding box로 뽑고, 2단계 NOVIC이 그 영역을 후보 클래스 없이 fine-grained 라벨로 분류합니다. 즉, 객체 로컬라이제이션은 “무엇이 정상에서 벗어났는가”로 처리하고, 분류는 NOVIC의 open vocabulary 능력을 그대로 가져오는 구조입니다.

- **Technical Challenges**: 핵심 난제는 RoI(관심 영역)를 뽑는 과정에서 특정 클래스/어휘 편향을 만들지 않으면서도, 로컬라이제이션을 충분히 정밀하게 확보하는 것이었습니다. 논문은 object-free workspace에서 MAE를 학습해 재구성 오차와 anomaly mask/table mask를 동시에 예측하게 만들고, 둘을 합쳐 중복 제거 및 IoU 기반 deduplication으로 박스를 안정화했습니다. 또한 작은 물체·다양한 시점에서의 성능 하락을 줄이기 위해 224px 입력에 맞춘 MAE 경량화(encoder 축소), 객체 pasting 기반 이상 학습, 필요 시 SAM 2.1로 박스 정교화를 선택적으로 적용했습니다.

- **Empirical Impact**: NICOL 테이블탑 환경에서 prompt-free 인식은 47.1% AP / 57.5% AP50 성능을, 후보 클래스가 주어졌을 때는 59.0% AP / 72.5% AP50를 보였습니다. 추가 데이터셋과 in-the-wild 48개 물체 평가에서는 prompt-free detection+classification 정확도가 최대 82.6%까지 올라가며, YOLO-World-v2, OWLv2, YOLOE 같은 tested open vocabulary baselines를 크게 앞섰습니다. 특히 YOLOE가 고신뢰 오탐을 자주 내리는 반면 AnomNOVIC은 세부 라벨이 더 정확하게 나와 로봇 실사용에서의 신뢰도 관점 impact가 강조됩니다.



### Learning Adversarial Augmentation Policies for Robust Garlic Seedling Detection (https://arxiv.org/abs/2606.26828)
Comments:
          16 pages

- **Prior Approaches**: 기존의 씨앗/묘목(특히 seedling) 검출 연구는 UAV 이미지나 온실처럼 조명이 비교적 균일한 환경에서 성능을 검증하는 경우가 많다. 이 때문에 땅 위 지상 기반 관측에서 흔한 직사광, 강한 그림자, 반사면에 의해 이미지 내 조명이 공간적으로 크게 달라지는 상황에서의 견고성은 충분히 다뤄지지 않았다. 또한 조명에 강인한 검출을 위해 사전 보정·추가 모듈(강화/전처리/특징 추출)을 넣는 방식은 추론 시 오버헤드를 늘리고, 결손 묘목의 중심(localization)까지 직접 겨냥하지 못하는 한계가 있었다.

- **Core Contribution**: 논문은 지상 기반 모니터링 플랫폼으로 실외 밭 조건에서 촬영한 마늘 seedling 전용 데이터셋(GSD)을 새로 구축해, 공간적으로 이질적인 심각한 조명에서도 검출이 가능하도록 문제를 구체화한다. 이어서 adversarial augmentation policy learning을 제안해, 확률적(=stochastic) 증강 정책 에이전트와 object detector를 함께 학습시키며 조명 변화에 강한 표현을 얻도록 한다. 특히 학습 시에만 정책을 사용해, 배포(inference) 단계에서는 추가 계산 없이 검출기를 그대로 쓸 수 있게 설계했다.

- **Technical Challenges**: 핵심 기술 난제는 ‘검출기를 속일 만큼 어려운 조명/모양 변형’을 생성하되, 현실적이지 않은 인위적 왜곡(artifact)을 만들지 않아야 한다는 점이다. 이를 위해 정책 에이전트가 light adjustment, texture detail, drop out, region selection으로 분해된 입력 조건부 증강 작업을 샘플링하고, 보상(reward)에는 detection loss에 더해 구조적 패널티로 SSIM 기반의 structural penalty를 포함해 이미지 구조 일관성을 유지하도록 유도한다. 또한 REINFORCE로 정책을 학습하되 advantage normalization 및 검출기 고정 기반의 번갈아 학습으로 업데이트 변동성을 완화한다.

- **Empirical Impact**: 실험 결과, 제안 방법은 AP50 91.6%를 달성해 기준선 대비 0.9%p 개선했으며, 기존 최고 성능 대비로도 0.2%p 앞섰다. 결손(seedling missing) 묘목의 downstream localization에 대해서는 precision 75.0%, F1-score 67.0%를 기록해 각각 기준선 대비 4.8%p, 2.0%p 향상됐다. 무엇보다 추론 시 추가 모듈 없이 성능을 올렸다는 점에서, 복잡한 실외 조명 하의 실용적 지상 농업 모니터링에 직접적인 의미가 있다.



### Multi-modality Image Fusion under Adverse Weather: Mask-Guided Feature Restoration and Interaction (https://arxiv.org/abs/2606.26812)
Comments:
          Accepted at ECCV 2026

- **Prior Approaches**: 기존 멀티모달 이미지 퓨전(MMIF)은 등록된 적외선(IR)과 가시광(VI) 입력을 결합해 보완 정보를 얻는 데 집중해 왔지만, 폭우·적설·헤이즈 같은 악천후에서는 잡음/열화로 특징 표현이 무너지는 문제가 커집니다. 이런 조건 대응은 보통 (1) 복원+퓨전(two-stage) 또는 (2) Pseudo Ground Truth(의사 정답) 기반 supervision으로 나뉘는데, two-stage는 복원 편향이 퓨전에 누적되고 최적화가 불안정해질 수 있습니다. 또한 의사 정답은 간단한 학습을 돕지만, 깨끗한 입력에서 나온 결과의 정보 손실·모달 바이어스 때문에 동적인 cross-modal 보완 학습을 충분히 유도하지 못합니다.

- **Core Contribution**: 본 논문은 악천후에서도 “동시 복원+모달 상호작용”을 한 네트워크 안에서 수행하는 mask-guided MMIF 방법(A​​MG-Fuse)을 제안합니다. 핵심은 Pseudo Ground Truth의 모달 배분을 분리·가이드하기 위해 마스크를 만들고, 그 마스크를 이용해 cross-modal cross-attention이 유의미한 영역에 선택적으로 집중하도록 학습을 설계한 점입니다. 더불어 복원 신호와 퓨전 상호작용을 균형 있게 맞추는 Mask-Guided Learning Strategy(MGLS)와 Task-Coupled Degradation-Aware Learning Strategy(TDAS)를 함께 도입합니다.

- **Technical Challenges**: 핵심 기술적 난제는 의사 정답 supervision이 가진 정적 분포 적합 위험을 줄이면서도, 열화 영역에서 각 모달의 기여를 안정적으로 추정해 cross-attention을 효과적으로 유도하는 것입니다. 이를 위해 논문은 fused 결과와 소스 이미지 간의 매핑 관계로부터 modality allocation mask를 구성하되, 가시광 밝기 편향이 마스크를 왜곡하지 않도록 분모에 Fuse 정보를 포함하는 방식으로 수치 불안정 문제를 완화합니다. 또한 Mask-Guided Feature Extraction Module(MFEM)과 MCCA에서 쿼리(각 모달 특징)를 마스크로 가중하고, fused 특징을 key/value로 사용해 모달별 유익한 패턴을 분리·재조합하도록 설계합니다.

- **Empirical Impact**: 실험은 AWMM-100k의 Snow/Rain/Haze 및 실제 열화 데이터, 그리고 이상 환경용 M3FD·MSRS·LLVIP에서 진행되었고, AM​​G-Fuse는 여러 시각 품질/정량 지표와 다운스트림 작업에서 SOTA를 상회하는 결과를 보였습니다. 악천후별로도 일관되게 상위권이며, 표에 따르면 snow/rain/haze에서 경쟁 대비 약 3~4% 수준의 개선이 관찰됩니다. 특히 기존 방식이 복원 후 과도한 디테일 손실(과평활)이나 모달 편중을 보일 때, AMG-Fuse는 열화 억제와 상호보완적 특징 추출의 균형을 유지하며 시각적으로도 더 자연스러운 퓨전을 제공합니다.



### NaviCache: Test-Time Self-Calibration Caching for Video Generation (https://arxiv.org/abs/2606.26795)
Comments:
          Published at ICML 2026: Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026

- **Prior Approaches**: 기존 VDM 가속 연구는 (1) 오프라인 캘리브레이션 기반(예: TeaCache, MagCache)과 (2) 오프라인 캘리브레이션 프리(예: EasyCache)로 크게 나뉜다. 캘리브레이션 기반은 캘리브레이션 비용과 분포 이동(distribution shift)에 취약하고, 캘리브레이션 프리는 test time에서 zero-order 근사로 인해 정확도가 떨어질 수 있다.

- **Core Contribution**: NaviCache는 VDM의 특징 변화(feature evolution)를 Inertial Navigation System(INS) 관점의 상태 추적 문제로 재구성한 플러그앤플레이 test-time self-calibration 기법이다. 입력 변화와 출력 반응의 상대적 결합을 모델링해 diffusion의 비정상성/비정지성을 다루며, 단계별 계산을 건너뛸 때의 오류 판단을 더 정교화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 라티스드 공간의 diffusion 변수와 INS의 네비게이션 변수 사이에 직접 매핑이 없다는 도메인 갭, (2) 초기·말기에서 역학이 급격히 변하는 high-turbulence 구간이다. NaviCache는 Initial Alignment로 초기 상태/불확실성을 캘리브레이션한 뒤, dual-state 추정(변화 비율과 잠재 drift)과 불확실성 기반 Measurement Update로 time-dependent noise schedule을 반영해 추정 발산을 억제한다.

- **Empirical Impact**: HunyuanVideo, Wan, Open-Sora 계열 모델과 VBench 및 LPIPS/PSNR/SSIM 지표에서 NaviCache는 compute skipping의 오판을 줄이면서도 전반적 화질을 개선했다고 보고한다. 또한 같은 latency 범위에서 EasyCache보다 더 나은 픽셀·지각 품질을 보였고, TeaCache/MagCache 같은 캘리브레이션 기반 대비해서도 상당한 경쟁력을 유지하며 고속 구간의 고스트/손·손가락 변형 같은 오류를 완화하는 사례를 제시한다.



### ReasonCLIP-58M: Visually Grounded Commonsense Reasoning Supervision for CLIP (https://arxiv.org/abs/2606.26794)
Comments:
          Accepted to ECCV2026

- **Prior Approaches**: 기존 CLIP류는 대규모 이미지-텍스트 기술(description) 정렬을 중심으로 사전학습되어 왔고, 덕분에 zero-shot 검색 성능은 강하지만 시각적으로 근거된 추론(grounded reasoning)을 직접 최적화하지는 못한다. 최근 개선도 데이터 스케일링이나 아키텍처 튜닝에 치우쳐, 추론 지향 다운스트림 수요(구성/상식/다단계 이해)에 대한 목표 불일치가 지적돼 왔다. 또한 일부는 fine-tuning 기반의 구조화/보조목표를 쓰지만, “사전학습 단계에서” 시각 근거 추론 신호를 대규모로 주는 연구는 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 CLIP-style 비주얼 인코더를 구조 변경 없이도 추론에 더 적합한 표현 공간으로 확장할 수 있는지에 주목하고, ReasonCLIP-58M이라는 continual pretraining 프레임워크를 제안한다. 핵심은 두 단계로 추론 신호를 점진적으로 통합하되, 1단계에서는 descriptive alignment를 보존하면서 reasoning-aware 정렬을 강화하고, 2단계에서는 범주(category) 구조화된 reasoning supervision으로 서로 다른 추론 패턴을 분리·정돈하도록 학습한다. 이를 위해 ReasonLite-42M(개방형·검증 가능한 reasoning 캡션), ReasonPro-16M(5개 reasoning 유형 범주별 감독), 그리고 진단용 벤치마크 RCLIP-Bench를 함께 구축했다.

- **Technical Challenges**: 가장 큰 기술 과제는 “추론 능력 향상”과 “기존 기술 정렬(descriptive alignment) 보존”을 동시에 달성하는 것이다. 이를 위해 1단계에서는 Time-dependent 가중치 스케줄과 원래 가중치에 대한 ℓ2 regularization을 결합해, 초기에는 기술 정렬을 주 신호로 유지하면서 점차 reasoning 캡션의 영향력을 늘린다. 2단계에서는 reasoning 캡션을 범주별로 정밀 감독하기 위해 다중 라벨 분류(이미지 측 multi-label, 텍스트 측 단일 라벨)를 보조 가지로 추가하고, 추론 유형별 구분성을 높이면서도 공통 표현 공간을 유지하도록 설계했다.

- **Empirical Impact**: 실험에서 ReasonCLIP-58M 계열은 다양한 backbone/스케일에서 시각 근거 추론 및 compositional reasoning 벤치마크의 성능을 일관되게 개선했으며, reasoning corpus 규모가 상대적으로 작아도 더 큰 데이터/아키텍처 변경 기반 접근을 넘어서는 결과가 보고된다. 특히 RCLIP-Bench의 3단계(Visual Grounding, Evidence Awareness, Visually Grounded Reasoning)로 진단해 보면, 단순 이미지-언어 연관 성능이 곧바로 추론 능력을 보장하지 않음을 보여주고 ReasonCLIP은 세 수준 모두에서 향상을 보였다. 또한 LLaVA-NeXT처럼 MLLM에 visual tower를 drop-in으로 교체했을 때 추가 추론 비용 없이 OKVQA, VisualLogic, GQA, MMStar 등에서 재현 가능한 이득이 관찰되며, 비추론 작업에서도 성능 안정성이 확인됐다.



### Event-based Gaze Control System for Accurate Real-time Spin Estimation in Professional Ball Games (https://arxiv.org/abs/2606.26780)
- **Prior Approaches**: 기존 비전 기반 볼 스핀 추정은 주로 프레임 기반 카메라에서 공 궤적(마그누스 효과)이나 표면 텍스처를 관찰해 스핀을 간접/직접 복원한다. 하지만 공은 매우 작고 빠르며 회전도 커서 모션 블러·시간적 알리아싱 때문에 스핀 크기와 축을 동시에 정확히 보기 어렵다. 이벤트 카메라를 쓰는 연구도 있었지만 대체로 정적이고 광각 촬영이라 공간해상도가 부족하고, 이동(translation)과 회전을 이벤트에 함께 섞어 관측 혼동이 생긴다.

- **Core Contribution**: 이 논문은 이벤트 기반 능동 시각(event-based active vision) 시스템으로 수정되지 않은 공(unmodified ball)의 스핀을 실시간에 가깝게 측정하는 방식을 제안한다. EVS 고시간해상도에 더해 텔레포 렌즈의 초점(3 ms)과 pan/tilt 갈바노 미러를 제어해 공이 시야에 머무르면서 주로 회전 정보만 이벤트로 남도록 만든다. 또한 구면(spherical) 대비 최대화(spherical contrast maximization; s-CMax)로 회전 파라미터를 오프라인에서 고정밀 추정하고, 이를 pseudo-ground-truth로 삼아 온라인 저지연 CNN+정제 파이프라인을 구성한다.

- **Technical Challenges**: 핵심 난제는 (1) 공이 작고 회전이 빨라 이벤트에서도 이동 성분이 스핀 신호를 덮을 수 있다는 점, (2) 기존 대비 최대화가 평면 투영을 사용해 회전하는 구의 기하를 충실히 반영하지 못한다는 점이다. 이들은 능동 미러 추적/초점 튜닝으로 translational motion을 보상해 이벤트 혼동을 줄이고, 대비 최대화를 구면 모델로 바꿔 깊이 모호성과 실루엣 편중 대비를 완화해 해결한다. 온라인 모드에서는 불확실성(uncertainty)을 고려한 CNN이 오프라인 s-CMax의 pseudo 라벨을 학습하고, GPU 배치 기반 s-CMax 정제로 보정해 3 ms 수준 지연을 맞춘다.

- **Empirical Impact**: 정적 공(static ball) 실험에서 s-CMax는 여러 종목(탁구/야구/테니스/골프)에 걸쳐 스핀 크기 평균 오차 2.1%, 축 오차 4.0°의 수준으로 기존 이벤트 기반 방법을 능가한다. 공이 비행 중일 때도 능동 추적 하에 2.1% 크기 오차와 5.4° 축 오차를 보고하며, 실사용에 가까운 견고함을 보여준다. 특히 프로 탁구 경기 3-view 구성에서 신뢰성 있는 동시 추적·스핀 추정이 가능했고, 3 ms 지연·750 Hz 처리량과 함께 오프라인 대비 8.8% 크기 및 6.4° 축 불일치를 달성해 실제 로보틱스/스포츠 분석 적용 가능성을 강화했다.



### LearniBridge: Learnable Calibration of Feature Caching for Diffusion Models Acceleration (https://arxiv.org/abs/2606.26778)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: Diffusion Transformers(DiTs)의 추론 비용을 줄이기 위해 feature caching, 증분 예측, 템포럴 일관성 기반 보정이 활발히 연구돼 왔습니다. 특히 DeepCache·FORA처럼 과거 feature를 그대로 재사용하거나, TaylorSeer처럼 Taylor-series 기반 예측으로 더 긴 스킵을 노리지만, 가속 비율이 커질수록 오차가 누적되며 품질이 급격히 무너지는 문제가 있습니다. 기존 방식은 구현을 단순화하는 대신 “어떤 보정이 필요한가”에 대한 구조적 이해가 부족했습니다.

- **Core Contribution**: 이 논문은 caching에서 발생하는 feature correction의 본질을 분석해, 최적 보정 업데이트가 다양한 프롬프트에서도 공유되는 낮은 차원의 저랭크 부분공간(low-rank subspace)에 놓인다는 점을 보여줍니다. 이를 바탕으로 LearniBridge는 여러 timestep을 연결(bridge)하는 learnable calibration을 제안하며, 최종 Transformer block에 대해 lightweight LoRA 업데이트만으로 보정을 수행합니다. 학습 시 3–5개 프롬프트만으로도 효과적으로 보정이 가능하다는 점을 강조합니다.

- **Technical Challenges**: 핵심 기술 도전은 “긴 timestep skipping에서 필요한 보정 방향이 무엇인지”를 찾아, 적은 학습 데이터와 파라미터로 이를 재현하는 것입니다. 저자들은 레이어 출력의 cross-timestep residual을 구성한 뒤 SVD로 스펙트럴 성질을 분석해 보정이 저랭크로 수렴함을 실험·이론적으로 뒷받침하고, 프롬프트 불변(prompt-invariant)한 부분공간임을 각 그룹 업데이트 각도 비교로 확인합니다. 이 구조에 맞춰 최종 블록에 LoRA를 적용하고, 학습된 보정이 캐시된 입력에서 건너뛴 timestep의 출력을 근사하도록 학습-추론 파이프라인을 설계했습니다.

- **Empirical Impact**: FLUX·HunyuanVideo·WAN2.1에서 LearniBridge는 각각 최대 5.87×, 5.75×, 4.10× 가속을 달성하면서 이미지·영상 생성 품질을 유지합니다. 특히 WAN2.1에서는 4.10× 가속 조건에서 기존 SOTA 대비 VBench를 1.28% 개선해 성능-가속의 동시 달성 가능성을 보여줍니다. 또한 qualitative 비교에서 TeaCache·TaylorSeer 계열의 콘텐츠/색/프레임 불일치 및 시간적 흔들림을 더 일관된 결과로 완화하며, LoRA rank와 적용 레이어에 따른 민감도도 함께 분석합니다.



### Anatomy-Guided Residual Motion Diffusion for Controllable 4D Cardiac MRI Synthesis (https://arxiv.org/abs/2606.26764)
- **Prior Approaches**: 4D(3D+time) 의료영상 생성은 희소한 주석, 기기 간 도메인 시프트, 개인정보 제약 때문에 어려웠습니다. GAN·확산 기반 접근들은 시각적 사실성은 개선했지만, 해부학(정적 구조)과 시간적 동역학을 분리해 독립적으로 제어하거나, 장기 일관성(temporal coherence)을 보장하는 데 한계가 컸습니다. 또한 강한 라벨 의존이나 강하게 결합된 생성 방식 때문에 paired segmentation까지 스케일링하기가 어려웠습니다.

- **Core Contribution**: 이 논문은 cine cardiac MRI의 4D 생성 문제를 ‘정적 해부학 생성’과 ‘잔차 기반 시간 모션 생성’으로 분리하는 4D controllable generative framework를 제안합니다. semi-supervised VAE가 해부학 잠재표현을 학습하면서 강하지 않은 라벨 환경에서도 intensity와 정렬된 segmentation mask를 함께 생성하도록 설계했습니다. 이어서 static LDM은 임상 priors(진단 및 용적 지표 등)으로 특정 해부학을 만들고, residual latent motion LDM이 ED 대비 잔차 모션을 생성해 시퀀스 전 구간의 temporal coherence를 유지합니다.

- **Technical Challenges**: 핵심 난제는 (1) 해부학 일관성을 유지한 채 고차원 4D를 생성하고 (2) 기기·환자별 변동 속에서도 임상 priors를 잘 따르는 제어력을 확보하는 것입니다. 이를 위해 VAE에서 labeled/ unlabeled를 함께 쓰는 semi-supervised 학습으로 의미 정렬을 강화하고, 시간축은 ED 기준의 residual latent trajectory로 모델링해 anatomy–dynamics 분리를 달성했습니다. 또한 시간 단계 생성 시 latent 기반으로 residual을 누적하고, 임상 조건은 cross-attention 기반으로 주입해 CFG로 제어하며, 가변 slice 수는 valid slice 마스크로 패딩 학습을 회피하도록 구성했습니다.

- **Empirical Impact**: cine cardiac MRI에서 여러 데이터셋 평가 결과, static anatomy 제어는 Pearson r > 0.8 수준의 높은 정합성을 보였고 temporal coherence는 FVD=288.08로 보고됐습니다. cross-vendor 일반화 실험에서 합성 4D 시퀀스를 학습에 추가하면 downstream segmentation이 개선되며, nnU-Net 기준 평균 Dice가 1.4%p 상승하고 Hausdorff Distance는 3.0mm 감소했습니다(좌심실: Dice +2.8%, 경계 오차 5.4mm 감소). 전반적으로 희소 주석과 도메인 시프트 상황에서 4D 데이터 증강을 “규모화 가능한 controllable 합성”으로 제공한다는 점에서 임상 보조 AI의 실용성에 의미가 큽니다.



### Calibrated Harmonic Overlaid Implicit Neural Representations for Multi-Dimensional Data (https://arxiv.org/abs/2606.26763)
Comments:
          ECCV2026 Accept

- **Prior Approaches**: 암시적 뉴럴 표현(INR)은 좌표를 연속 함수로 파라미터화해 멀티스펙트럼 이미지/비디오 등 다차원 복원을 잘해왔다. 특히 sine 같은 주기(periodic) 활성은 고주파를 잘 모델링하지만, 대부분이 함수 composition을 반복해 깊어질수록 최적화 불안정(gradient pathology)이 커지고 성능이 얕은 층에서 정체되거나 악화된다. 또한 자연 이미지의 1/f 스펙트럼 같은 물리적 통계와 맞지 않게 주파수 스케일이 전역 고정으로 설계돼 spectrum bias를 제대로 완화하지 못했다.

- **Core Contribution**: 논문은 딥 주기 INR의 한계를 조화(harmonic) 관점에서 재정의하고, 기존의 함수 composition이 신호의 superposition(가산성) 원리와 충돌한다는 점을 핵심 원인으로 제시한다. 이를 바탕으로 Calibrated Harmonic Overlaid Implicit Neural Representation(CHOIR)을 제안하며, Coordinated Harmonic Superposition(CHS)로 addititve 형태의 조화 항을 계층적으로 오버레이해 깊은 스케일링을 안정화한다. 더 나아가 Perceptual Spectrum Calibration(PSC)으로 1/f 전력법칙(주파수 분포) prior를 네트워크의 주파수 샘플링/진폭 배치에 반영해 spectrum bias를 체계적으로 줄인다.

- **Technical Challenges**: 핵심 기술적 난관은 (1) 깊어진 네트워크에서 Jacobian 곱으로 인해 gradient vanishing/exploding이 발생해 학습이 흔들리는 문제와 (2) 전역 고정 주파수 스케일로 인해 실제 데이터의 주파수 에너지 분포와 불일치하는 문제다. CHS는 각 조화 모듈의 기여를 학습 가능한 스칼라 β로 스킵-커넥션처럼 점진적으로 ‘활성화’(초기에는 선형 매핑에 수렴)해 implicit curriculum learning 효과로 최적화 지형을 안정화한다. PSC는 주파수를 log-uniform 분포로 배치하고, 고주파가 과도한 그래디언트를 만들지 않도록 전력법칙에 맞춘 amplitude modulation 및 학습 가능한 스펙트럼 감쇠율(α)로 에너지 기반 캘리브레이션을 수행한다.

- **Empirical Impact**: 실험에서는 2D 신호 피팅(코닥 House)과 5D novel view synthesis(NeRF Blender)에서 CHOIR이 PSNR/SSIM/LPIPS 전반에서 기존 SIREN/FINER/SL2A-INR/IGA-INR 등보다 일관되게 우수함을 보였다. 특히 고주파 디테일(윤곽선/드럼 질감 등) 복원이 더 날카롭고 시각 품질도 개선되는 경향이 확인됐다. 더 나아가 결측 채우기 및 복합 열화 복원(가우시안 잡음, salt-and-pepper, 스트라이프 결손 등) 멀티모달 데이터(하이퍼스펙트럴/멀티스펙트럴) 실험에서도 성능이 향상되어, 주기 INR의 확장성과 스펙트럼 캘리브레이션을 동시에 다루는 접근이 실질적 의미가 있음을 보여준다.



### ProtoKV: Streaming Video Understanding under Delayed Query with Summary-State Memory (https://arxiv.org/abs/2606.26762)
Comments:
          20 pages, 4 figures, Accepted to ICML 2026

- **Prior Approaches**: SVU는 스트림이 계속되면서도 쿼리가 비동기로 도착하는 상황에서, 고정된 GPU 메모리로 온라인 상태를 유지해야 합니다. 기존 bounded-memory 접근은 슬라이딩 윈도우(SWA)처럼 최근만 저장해 지연이 커지면 결정 단서가 창을 벗어나 급격히 손실되거나, 토큰 retention/압축은 고정 예산 안에서 토큰 인스턴스 생존 경쟁이 누적되어 희귀하지만 결정적인 증거를 놓치기 쉽습니다. 또한 retrieval/offloading 계열은 미세한 디테일을 복구할 수 있으나 질의 시 오버헤드와 지연 변동이 커 실시간 제약에 불리합니다.

- **Core Contribution**: ProtoKV는 지연 쿼리에서 핵심인 “post-evidence update pressure”를 겨냥해, 먼 과거를 토큰 인스턴스로 보관하지 않고 고정 용량 요약 상태로 표현하는 상수-풋프린트 KV 메모리를 제안합니다. 정확한 near-window KV 캐시는 그대로 두고, far history는 prototype bank(공간-시간 일관성을 갖는 프로토타입 요약 + 잔차 통계)로 압축한 뒤, 쿼리 시에는 prototype을 bounded pseudo-token으로 노출해 표준 attention에 drop-in 방식으로 결합합니다. 그 결과 메모리 예산 대비 “원거리 증거를 더 오래, 더 안정적으로” 활용하는 방향으로 정확도 저하를 완화합니다.

- **Technical Challenges**: 가장 큰 기술 과제는 결정 단서가 나온 뒤(근-원 경계 이후) 수많은 업데이트가 들어오는 동안, 토큰 retention의 교체/희석 실패를 요약 상태에서도 재현하지 않도록 만드는 것입니다. ProtoKV는 원거리 증거를 persistent summary로 누적하되, 퇴출되는 토큰을 연속성을 고려한 할당 규칙(키 유사도+공간 연속성+staleness 제어)으로 동일 객체/트랙에 가깝게 묶어 프로토타입 전환과 단편화를 줄입니다. 또한 프로토타입 단일 센터 평균의 손실을 residual statistics(고정 예산 streaming histogram 기반)로 보완하고, attention 로그잇에 mass-aware weighting(로그 카운트 보정)을 넣어 자주 관측된 증거가 과소평가되지 않게 설계했습니다.

- **Empirical Impact**: ProtoKV는 SVU 벤치마크 4종에서 지연이 커질수록 성능이 떨어지는 양상이 기존 방식과 달라, 장지연 구간에서 더 완만한 저하와 더 큰 상대 이득을 보였습니다. 예산을 맞춘 비교에서 RVS-Movie 기준 토큰 retention 대비 최대 +12.5pp, SWA 대비 최대 +20.4pp 향상이 보고되며, 지연이 증가할수록 격차가 커지는 경향이 나타났습니다. ablation에서도 far-memory 요약 자체와 residual statistics, mass-aware weighting, continuity-aware association이 함께 있을 때 견고함이 나타나며, 이는 상수 메모리 환경에서 지연 쿼리 성능을 개선하는 설계 방향이 실증적으로 지지됨을 의미합니다.



### Capacity-Controlled Multi-View Stylization of 3D Gaussian Splatting (https://arxiv.org/abs/2606.26754)
Comments:
          Accepted to ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 3DGS 기반 stylization은 각 렌더 뷰마다 2D feature 매칭 손실을 독립적으로 적용하는 경우가 많다. 이 방식은 뷰 내부에서 many-to-one 매칭으로 인해 스타일이 특정 feature에 과도하게 몰리거나 반복 질감이 생기고, 뷰 간에는 viewpoint 변화에 따라 매칭이 흔들리면서 관찰 시 일관성이 깨지는 문제가 나타난다. DINO나 geometry regularization을 더해도 보통 매칭 메커니즘 자체를 직접 제어하진 못한다.

- **Core Contribution**: 이 논문은 3DGS의 multi-view stylization을 위해 Capacity-Controlled Feature Transport(CCFT)와 Cross-View Matching Guidance를 결합한 프레임워크를 제안한다. CCFT는 local style matching을 semi-balanced optimal transport로 재구성하고, column-capacity 제약으로 스타일 feature에 대한 할당을 제어해 many-to-one 붕괴를 완화한다. 여기에 cross-view guidance를 더해 같은 3D content가 뷰가 달라도 일관된 style pattern에 대응하도록 정합을 강제한다. 또한 stylization 품질을 위해 vanilla 3DGS 재구성 단계에 geometric regularization을 도입해 Gaussian 프리미티브를 더 균일하고 세밀하게 만든다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 뷰별로 불안정한 feature 할당이 many-to-one으로 붕괴하는 문제와 (2) viewpoint 변화로 인해 인접 뷰 간 correspondence가 흔들려 cross-view consistency가 깨지는 문제를 동시에 잡는 것이다. 저자들은 semi-balanced optimal transport에 column-capacity(튜너블 강도) 제약을 포함하고 Sinkhorn-Knopp 기반으로 최적 할당을 구해 스타일 다양성과 안정된 correspondences를 함께 달성한다. 더불어 cross-view guidance에서 이전 뷰의 매칭 결과(transport에 기반한 guidance map)를 거리 계산에 반영해 sparse bipartite support가 최적화 전반에서 흔들리지 않도록 설계했다. 마지막으로 Gaussian의 형태·크기·깊이 정확도를 regularize해 fine-grained texture 표현이 가능하도록 재구성 기반을 강화한다.

- **Empirical Impact**: LLFF, T&T, Mip-NeRF 360 등에서 6종 style reference를 적용한 실험과 정량 지표(ArtFID, 구조 보존, MEt3R 기반 view consistency)에서 기존 SOTA 대비 전반적으로 우수한 성능을 보인다. 특히 many-to-one으로 생기던 반복 질감과 뷰 간 번짐/불일치를 줄이면서 장면의 의미 구조는 보존하는 경향이 정성·정량 모두에서 확인된다. 또한 사용자 선호도 조사에서도 style 품질, content 보존, multi-view consistency 세 항목에서 큰 격차로 더 선호되는 결과를 제시한다. ablation과 capacity 파라미터 변화 실험을 통해 CCFT의 capacity가 스타일 다양성과 질감을 직접 조절함을 실증하며, cross-view guidance는 consistency를 높이되 content 보존과의 미세한 트레이드오프가 있음을 보였다.



### Depth-Semantic Alignment and Affinity-Guided Fusion for Structured Radar Point Cloud Generation (https://arxiv.org/abs/2606.26743)
- **Prior Approaches**: 기존 LiDAR 유사 pseudo-point cloud 생성은 단안/스테레오 영상에서 깊이를 추정한 뒤 3D로 투영해 기하를 복원하는 방식이 주류였다. 다만 스케일 모호성, 약한 텍스처·가림 영역에서의 불완전한 기하, 장면 전이 한계가 남는다. 레이더 기반 생성은 FMCW 신호처리·CFAR·각 추정 등 물리 기반 접근과 딥러닝 기반 복원/정제 접근이 있으나, 안테나 수 및 잡음·멀티패스 영향으로 희소하고 구조가 끊기는 문제가 크다. 비전-레이더 fusion은 대체로 downstream 인식 성능 개선에 집중했고, 포인트클라우드 자체의 품질을 직접 끌어올리는 생성 관점은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 논문은 vision–radar fusion을 통해 밀리미터파 레이더 point cloud를 “조밀하고 구조적으로 완전한” 형태로 생성하는 프레임워크를 제안한다. 영상의 의미 정보로 구조 제약과 공간 정렬을 걸고, 레이더의 희소성을 보완하기 위해 sparse completion 전략을 포함해 누락 구조를 복원한다. 생성된 포인트클라우드는 3D object detection과 3D object tracking에 투입되어, 인식 모델이 쓰기 좋은 통합 표현을 제공하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 레이더 BEV는 잡음·경계 흐림이 있어 신뢰도 있는 신호를 강조해야 하고, (2) 영상 깊이/의미와 레이더 측정 간의 정합(alignment)을 정확히 맞춰 교차모달 대응을 안정화해야 하며, (3) fusion 후에도 남는 잡음·누락 점을 기하적으로 일관되게 정제해야 한다는 점이다. 해결을 위해 레이더 응답은 Hessian-based peak enhancement로 강화·잡음 억제를 수행하고, 영상은 LSS를 확장해 depth를 카테고리별로 분해(semantic priors)해 정렬 신뢰도를 높인다. 또한 시각 BEV의 희소도를 radar affinity 기반 반복 전파로 보완한 뒤, CFAR에서 얻은 신뢰 앵커 포인트를 기준으로 k-NN 그래프 기반 기하 최적화로 구조 완전성과 위치 정확도를 함께 개선한다.

- **Empirical Impact**: 실험에서는 OS-CFAR, Sparse2Dense, SGDNet과 공정 비교를 수행하며, SECOND와 PointPillars 두 검출기에서 제안 방법이 일관되게 더 높은 AP30·AP50을 보였다. 예를 들어 SECOND에서 SGDNet 대비 AP30 5.6p, AP50 2.8p, PointPillars에서는 AP30 6.0p, AP50 3.3p 향상이 보고된다. 추적에서는 레이더를 전처리 모듈로 사용한 뒤 MOTA와 AUC가 크게 개선되어, MOTA 0.108→0.494, AUC 0.112→0.572 수준의 성능 향상이 나타났다. 이는 조밀하고 구조적으로 완전한 생성 포인트가 먼 거리·작은 물체·부분 가림 상황에서 위치 추정과 프레임 간 연관을 안정화해 성능과 강건성을 동시에 끌어올린다는 의미가 있다.



### LiveEdit: Towards Real-Time Diffusion-Based Streaming Video Editing (https://arxiv.org/abs/2606.26740)
Comments:
          Accepted by ECCV 2026, Project page: this https URL

- **Prior Approaches**: 기존 스트리밍 비디오 편집은 드물고, 대다수 확산 기반 편집 모델은 오프라인(비인과)으로 전체 프레임을 함께 처리하는 방식에 의존한다. 이때 미래 문맥을 보지 못하는 스트리밍 환경으로의 단순 전환은 ‘attention distribution shift’로 인한 깜빡임/구조 붕괴를 초래하고, 또한 매 프레임 모든 토큰을 전부 계산하는 병목 때문에 지연이 커진다. 결과적으로 AR 같은 실시간 시나리오에 바로 넣기 어렵다.

- **Core Contribution**: 이 논문은 인과적(causal)이고 프레임 단위로 동작하는 스트리밍 비디오 편집 프레임워크 LiveEdit를 제안한다. 핵심은 강한 콘텐츠 보존과 실시간 반응성을 함께 달성하기 위해, 양방향 foundation model의 편집 능력을 단방향 스트리밍 에디터로 단계적으로 ‘distillation’해 4-step 추론으로 압축하는 3-stage 파이프라인이다. 여기에 AR 지향 Mask Cache로 편집되지 않는 영역의 계산을 건너뛰어 배경 품질을 유지하면서 속도를 끌어올린다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 비인과 모델의 양방향 attention을 인과 실행으로 바꿀 때 생기는 구조적 사전지식 붕괴, (2) 편집되지 않는 배경의 공간-시간 토큰 중복이 프레임당 지연을 폭증시키는 문제다. 저자들은 Stage 2의 chunk-wise teacher forcing으로 causal attention 분포를 로컬 bidirectional prior에 맞춰 ‘forgetting/플리커’ 현상을 완화하고, Stage 3의 DMD로 ODE 초기화 없이 4-step 생성으로 압축한다. 또한 Mask Cache는 L2 distance 기반 편집 마스크로 Self-Attention의 중복 영역만 캐싱/우회해 계산량을 줄이되 배경 consistency를 보장한다.

- **Empirical Impact**: 120쌍 전용 스트리밍 비디오 편집 벤치마크에서 시각 품질과 시간적 안정성 전반에 걸쳐 최신 스트리밍 baseline 대비 우수하거나 경쟁력 있는 성능을 보이며, 특히 텍스트 정렬에서도 gap을 메우는 결과를 제시한다. 추론 속도는 12.66 FPS(프레임당 약 79ms)로 크게 개선되어 상호작용/AR 응용에 실사용 가능한 수준을 목표로 한다. 추가로 ablation과 마스크 캐시 위치 비교를 통해 Self-Attention 캐싱이 품질 저하 없이 가속에 가장 효과적임을 실증한다.



### Do Image Editing Models Understand Lighting? (https://arxiv.org/abs/2606.26738)
- **Prior Approaches**: 기존 instruction-based image editing 벤치마크는 장면 일관성이나 전반적 그럴듯함(semantic plausibility)을 중심으로 평가해 왔습니다. VLM이나 인간 평가는 인간 지각과의 정렬은 잘 맞추지만, 픽셀 수준의 복사-대역(복사 세기/색/그림자·하이라이트 감쇠) 정확도를 정량화하기 어렵고 생성 인공효과에 취약합니다. 합성 데이터/렌더링 기반 평가는 통제된 물리성을 제공하지만, 실제 환경의 복잡한 조명 상호작용과 재현성에는 한계가 있습니다.

- **Core Contribution**: 이 논문은 3D-anchored Light Probe(3DLP) 벤치마크를 제안합니다. 라이트 프로브(light probe)를 실제로 켰다/껐을 때의 동일 장면 1K HDR 이미지 페어를 구축하고, 그림자·금속/거울/투명 재질 등 핵심 영역을 세분 라벨링해 생성된 relighting이 실제 물리와 얼마나 일치하는지 픽셀 단위로 평가합니다. 또한 전역 white balance·exposure 같은 사진 효과를 보정하면서 라이트 프로브 기여만 추적하는 두 개의 신규 점수(강도 오차, 라이트 폴오프/공간 변화 오차)를 도입합니다.

- **Technical Challenges**: 핵심 난제는 생성 모델이 라이트 프로브의 3D 위치와 기여를 주변의 복잡한 ambient 조명과 분리해, 재질별 빛 전달(그림자·반사·감쇠)을 정확히 재현하는지 정량화하는 것입니다. 저자들은 온/오프 영상의 비율(ratio) 기반으로 전역 스케일·이동 성분에 불변인 표준화 절차를 설계하고, 강한 엣지(텍스처/기하로 인한 고주파)를 제거한 뒤 gradient 기반 지표로 빛의 공간적 감쇠 패턴을 비교합니다. 이를 통해 AI가 그럴듯한 보정(예: 화이트 밸런스 조정)을 해도 물리적 일치도를 정확히 측정하도록 구성했습니다.

- **Empirical Impact**: 6개 state-of-the-art image editing 모델을 평가한 결과, 전반 성능 차이가 크게 나타났고 specular highlight 쪽은 상대적으로 오류가 덜 두드러졌습니다. 다만 라이트 프로브로부터 덜 빛을 받는 영역에서는 모든 모델의 오차가 더 커졌으며, 거울/메탈릭/복잡한 재질에서 성능 변동이 커지는 패턴이 확인됐습니다. VLM 기반 비교는 거시적 그럴듯함은 잘 반영하지만 픽셀 수준 light transport 분석에는 부적합하다는 결론으로 이어졌고, 저자들은 벤치마크와 데이터셋을 공개해 후속 연구를 촉진하겠다고 밝혔습니다.



### Robust Onion: Peeling Open Vocab Object Detectors Under Nois (https://arxiv.org/abs/2606.26734)
Comments:
          Accepted at The 19th European Conference on Computer Vision (ECCV)

- **Prior Approaches**: 기존 OV-OD 연구는 잡음(비·안개·압축·모션 블러 등)이 성능을 떨어뜨린다는 사실은 다뤘지만, 왜/어디서 무너지는지(모델 내부 병목)는 “복잡한 구조 때문에” 잘 분리해 설명하기 어려웠습니다. 또한 현실의 저품질(HQ-LQ) 쌍을 맞춘 데이터가 부족해, 합성 잡음을 쓰더라도 구성요소별 영향 분해가 제한적이었습니다.

- **Core Contribution**: 이 논문은 Robust Onion으로 OV-OD의 견고성 저하를 “레이어별 feature collapse 붕괴” 관점에서 실험적으로 분해합니다. 합성 시각 열화로 관측 가능한 붕괴(observable)와 거의 보이지 않는 붕괴(minimal)를 모사해, 모델이 어떻게/어디서 깨지는지 해석 가능하게 드러내며 이전 robustness 관찰들을 체계적으로 재설명합니다.

- **Technical Challenges**: 핵심 과제는 (1) 현실 잡음의 효과를 직접적으로 측정할 저품질-고품질 쌍 데이터가 없고, (2) OV-OD가 비전 백본·텍스트·퓨전·박스 예측 등 여러 모듈로 얽혀 있어 원인 분리가 어렵다는 점입니다. 저자들은 controlled synthetic degradations로 붕괴 유형을 단계적으로 “벗겨내는” 방식으로 해결하고, UMAP 등으로 층별 특징 겹침을 확인해 얕은 레이어의 취약성과 유사 백본 간 비슷한 붕괴 패턴을 제시합니다.

- **Empirical Impact**: COCO·LVIS에서는 이미지 도메인이 주로 견고성을 좌우하며, ODinW-13처럼 큰 단일 객체가 많은 벤치마크는 robustness를 과대평가할 수 있음을 보여줍니다. 또한 NN & TK0 플러그앤플레이 방식으로 BDD100K·WiderFace·VisDRONE에서 실세계 견고성을 개선했는데, end-to-end 대비 학습 가능한 파라미터를 96배 줄이면서도 비슷한 수준의 robustness를 달성합니다.



### Full spectrum Unlearnable Examples via Spectral Equalization (https://arxiv.org/abs/2606.26719)
Comments:
          to be published in ICML

- **Prior Approaches**: Unlearnable examples(UEs)은 시각적으로 거의 변하지 않는 섭동을 주입해 모델이 의미 표현을 추출하지 못하게 하는 프라이버시 방어 기법이다. 그러나 기존 UE들은 주파수 특성을 충분히 고려하지 않아, 특히 저역통과(low-pass) 필터가 적용되면 unlearnability가 크게 약화되는 취약점이 있었다.

- **Core Contribution**: 이 논문은 기존 UEs가 저역통과 필터에 취약하다는 원인을 “섭동이 주로 고주파에 집중된다”는 현상으로 규명한다. 이에 따라 스펙트럼 전 구간에서 동시에 효과가 유지되는 spectrum-agnostic UE가 필요하며, 이를 달성하기 위한 Full-spectrum Unlearnable examples via Spectral Equalization(FUSE)를 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 특정 주파수 대역이 억제되더라도 남는 대역에서 모델이 의미 단서를 재학습하지 못하게 섭동을 설계하는 것이다. FUSE는 Random Spectral Masking(RSM)으로 학습 중 임의의 연속 주파수 밴드를 제거해 “어떤 밴드가 사라져도” 섭동 효과가 유지되도록 강제하고, Cross-Band Guidance(CBG)로 저주파-고주파 간 상호 일관성을 맞춰 저주파 unlearnability를 강화하면서 고주파 섭동은 시맨틱 정합성을 해치지 않게 조절한다.

- **Empirical Impact**: 실험은 CIFAR-10/100, SVHN과 ResNet/VGG/DenseNet/ViT 등 다양한 백본, 그리고 여러 cutoff 수준의 스펙트럼 필터링 시나리오에서 진행됐다. 결과적으로 FUSE는 저역통과 환경에서도 테스트 정확도를 거의 무작위 수준에 가깝게 낮추며, 기존 UE 대비 low-pass 취약성을 일관되게 극복했고(예: CIFAR-10 ResNet-18에서 큰 폭의 성능 회복 억제), 필터링이 없는 경우에도 대부분의 설정에서 가장 낮은 정확도를 보여 전 구간 일반성을 입증했다.



### Extracting Neural Materials from Multi-view Images (https://arxiv.org/abs/2606.26715)
Comments:
          Project website: this https URL

- **Prior Approaches**: 기존 역렌더링 기반 재질 추출은 PBR(예: base color, roughness, metallicity) 파라미터화를 강하게 활용해 분해가 비교적 용이했다. 하지만 PBR의 단일 로브 스펙(예: GGX 중심) 구조 때문에 오프라인 렌더링에서 보이는 복잡한 다중 로브 효과(클리어코트, fuzz, scatter 등) 재현에는 한계가 있었다. 또한 PBR을 넘어 neural material을 다루더라도, neural latent 공간의 비선형성 때문에 최적화가 지역해(local minima)에 빠지기 쉬워 초기화가 중요하지만 표준 inverse rendering만으로는 잘 안 풀리는 문제가 있었다.

- **Core Contribution**: NeuMatEx는 다중 뷰 이미지에서 공간적으로 변하는 Neural Material을 추출하는 differentiable inverse rendering 파이프라인을 제안한다. 핵심은 Large Material Reconstruction Model(LMRM)이 이미지로부터 neural material의 초기 base color, neural material latents, aleatoric uncertainty를 직접 예측하고, 이후 inverse path tracing을 test-time optimization으로 정교화한다. 특히 불확실성으로 신뢰 구간은 단단히 고정하고 모호한 구간만 유연하게 학습해, 조명·복잡한 스펙 효과가 재질에 “베이크 인”되는 것을 막는다.

- **Technical Challenges**: 문제는 neural material latent 공간의 비선형 구조로 인해 naive inverse rendering 최적화가 쉽게 수렴하지 않거나 그럴듯한 듯한 잘못된 재질로 드리프트한다는 점이다. 이를 위해 LMRM을 먼저 학습해 좋은 초기 상태를 제공하고, test-time optimization에서는 triplane 표현으로 최적화 변수를 제한한 뒤 uncertainty-weighted 정규화(가우시안 형태의 앵커)를 함께 걸어 모호성의 영향을 줄인다. 또한 미분 가능한 Monte-Carlo 기반 inverse path tracing에서 neural specular를 universal neural material basis로 평가하되, 최적화는 triplane만 업데이트하도록 설계해 안정성과 효율을 동시에 노린다.

- **Empirical Impact**: 합성 및 실제 자산 실험에서 NeuMatEx는 PBR 기반 재질 추출 대비 시각적 품질과 material decomposition(확산/스펙 분리 및 복잡 효과 분해)에서 더 우수한 결과를 보였다. 정량적으로는 orbital 뷰 PSNR에서 일관된 개선이 관찰되며, 특히 haze·clearcoat·fuzz·scatter 같은 고난도 스펙 효과를 reference에 가깝게 재구성했다. 아블레이션에서도 LMRM 초기화, triplane의 공간 일관성, 그리고 uncertainty 기반 정규화가 모두 개선에 기여함이 확인되어, 제안한 “prior+inverse optimization” 전략의 실효성이 입증된다.



### Mask to Concept: Auto-Promptable SAM3 via Efficient Test-Time Concept Embedding Search for Few-Shot Annotation (https://arxiv.org/abs/2606.26711)
Comments:
          Accepted by MICCAI 2026

- **Prior Approaches**: 의료 영상에서 고품질 픽셀 단위 세그멘테이션 라벨을 모으는 비용이 너무 커서, SAM 같은 학습 없는(파라미터 고정) 모델을 박스/포인트 같은 기하학 프롬프트로 자동화하려는 시도가 이어졌다. 다만 기존 자동화 방식은 외부 feature matcher나 보조 네트워크를 붙여 인스턴스 간 대응을 맞추는 구조가 많아 파이프라인이 복잡해지고, SAM의 내부 추론을 충분히 활용하지 못해 성능이 수동 프롬프트보다 흔들리거나 레퍼런스가 늘어도 개선이 제한적이었다. 또 few-shot segmentation(FSS) 기반 학습형 방법은 사전 학습 분포에 강하게 묶여 다른 의료 도메인에서 일반화와 유연성이 떨어지는 한계가 있었다.

- **Core Contribution**: 이 논문은 SAM3의 ‘텍스트 기반 개념 세그멘테이션’을 의료 few-shot 환경에 바로 적용하기 위한 Mask to Concept(M2C)를 제안한다. M2C는 사람이 애매한 텍스트를 직접 공학적으로 만들 필요 없이, 소수의 라벨 마스크로부터 학습 가능한 concept embedding만을 최적화해 SAM3가 데이터셋 전용 개념을 자동으로 찾게 만든다. 이어서 Hybrid Uncertainty Estimation(HUE)와 결합해 불확실한 샘플만 사람이 교정하고, 교정된 마스크가 다시 embedding 탐색에 환류되는 self-enhancing 애너테이션 루프를 구성한다.

- **Technical Challenges**: 핵심 난제는 SAM3가 fine-grained 임상 개념 지식을 원천적으로 갖고 있지 않는데, 인간이 쓴 텍스트 설명은 모호해서 자동 라벨링에 바로 쓰기 어렵다는 점이다. 저자들은 SAM3를 재학습하거나 외부 모듈 없이, concept embedding을 별도 벡터로 두고 예측-손실을 역전파해 임베딩만 점진적으로 업데이트하는 방식으로 텍스트 의존성을 제거했다. 또한 사람 개입을 최소화하기 위해 prediction entropy(경계 모호성)와 concept-geometry prompting inconsistency(개념 프롬프트 vs 박스 프롬프트 불일치)를 합친 HUE를 설계해, 불확실한 케이스를 적극 선택하는 메커니즘을 만들었다.

- **Empirical Impact**: Kvasir-SEG(내시경 폴립)와 ISIC-2017(피부 병변)에서 실험한 결과, M2C는 모든 shot 설정에서 경쟁 FSS/SAM 기반 방법을 능가하며 특히 1-shot~11-shot에서 Dice 기준 격차를 크게 벌렸다. 예컨대 11-shot에서 Kvasir-SEG는 +4.2%, ISIC-2017은 +11.8%의 Dice 개선을 보고했으며, 수작업 기하 프롬프트 자동화 접근보다 안정적으로 정교한 경계를 처리하는 정성 결과도 제시된다. 더 나아가 HUE 기반 human-in-the-loop 시스템은 동일한 인력 예산에서 데이터셋 전체 라벨 품질을 더 빠르게 끌어올리는 애너테이션 효율과 시나리오 전반에서의 일반화 이점을 보여 의료 라벨링 자동화에 실용성이 크다는 점을 강조한다.



### Intracranial Aneurysm Classification and Segmentation via Tri-Axial ROI and Multi-Task Learning (https://arxiv.org/abs/2606.26706)
- **Prior Approaches**: 기존 자동화 연구는 대체로 동맥류 존재 여부를 이진 형태로 감지하거나(존재/비존재), 마스크와 기기출력도 이진 분할에 머무는 경우가 많아 해부학적 위치를 세분화한 multi-class 분할·분류를 충분히 다루지 못했습니다. 또한 CTA, MRA, T2, T1-post처럼 대비·해상도·시야가 다른 4개 모달리티에서 ROI 추출은 모달리티 전처리 의존성이 크거나 3D 슬라이딩 윈도우로 비용이 커 확장성에 제약이 있었습니다.

- **Core Contribution**: 이 논문은 동맥류 위험평가·치료계획에 필요한 세분화 정보를 위해 multi-label 동맥류 분류와 multi-class 동맥류 분할, multi-class 혈관 분할을 한 프레임워크에서 동시에 수행하는 멀티태스크 설계를 제안합니다. 13개 해부학 위치와 4개 모달리티를 대상으로 하며, Stage 1의 빠른 2D tri-axial ROI 추출과 Stage 2의 3D multi-task nnU-Net 백본을 결합해 효율과 정밀도를 함께 노립니다. 또한 dual-decoder를 통해 혈관과 동맥류의 극심한 부피 불균형을 완화하고, cross-attention pooling 및 modality-specific auxiliary head로 모달리티 이질성을 흡수합니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 (1) 모달리티별 특성 차이와 (2) 동맥류는 희소하고 혈관은 방대한 클래스 불균형 때문에 3D 학습 신호가 동맥류 쪽으로 전달되지 않는 문제였습니다. 논문은 3D 전영역 추론 대신 축별 2D tri-axial 샘플링으로 ROI를 빠르게 국소화해 연산량을 줄이고(볼륨 크기에 덜 민감), dual-decoder로 동맥류 관련 채널을 분리·통합해 희소 그래디언트를 집중시키는 방식으로 학습을 안정화했습니다. 여기에 cross-attention pooling과 모달리티 분류 보조학습을 더해 이질적 입력에서도 표현을 더 잘 학습하도록 구성했습니다.

- **Empirical Impact**: RSNA 2025 Intracranial Aneurysm Detection에서 앙상블 기반으로 2nd place를 달성했으며, ablation 결과로 데이터(라벨) 정제, test-time augmentation(TTA), 앙상블이 순차적으로 성능을 끌어올리는 경향을 확인했습니다. 특히 2D tri-axial ROI 추출은 외부 케이스 기준 3D sliding-window 대비 평균 12.7배 속도 향상을 보여 실사용 관점의 효율성도 입증했습니다. 코드는 3D Slicer 플러그인 형태로 공개되어 CTA, MRA, T2, T1-post 전 모달리티에서 구성 가능하며, 추가 연구·임상 워크플로우 확장에 기여할 것으로 기대됩니다.



### PhysEditWorld: A Large-Scale Dataset Toward Physics-Editable World Models (https://arxiv.org/abs/2606.26694)
Comments:
          Project page: this https URL

- **Prior Approaches**: 게임 world model은 시각적으로 그럴듯한 action-conditioned 롤아웃을 생성하는 데는 빠르게 진전했지만, 물리 규칙은 데이터의 암묵적 정규성으로만 학습되는 경우가 많았다. 그 결과 같은 장면을 물리 법칙으로 ‘편집’했을 때 어떻게 변해야 하는지(저자 의도형 시뮬레이션)는 잘 답하지 못했다. 또한 기존 데이터셋/벤치마크는 물리 사건의 타당성이나 직관적 물리 평가에는 강점이 있어도, 동일 상호작용을 고정한 채 물리만 바꿔 비교하는 matched intervention 구조가 부족했다.

- **Core Contribution**: 이 논문은 gravity를 첫 editable physical parameter로 삼아, PhysEditWorld(물리 편집 게임 월드 모델 데이터셋)를 제안한다. UE5 재생(replay) 기반 파이프라인으로 동일한 장면·초기상태·정규화된 action trace·캐릭터 컨트롤러·카메라 정책을 고정한 채 중력 설정만 달리 반복 재생한다. 그로 인해 ‘편집된 물리’에 대한 귀속 가능한(비교 가능한) 차이를 측정할 수 있는 데이터 구조를 제공한다. 

- **Technical Challenges**: 핵심 기술 과제는 물리 편집에 따른 차이를 잡음이 아닌 통제 변수로 분리하는 것이다. 이를 위해 UE5 in-editor plug-in에서 시나리오를 준비한 뒤, Enhanced Input System 기반의 semantic action sequence로 의도를 보존하고 하드웨어/플랫폼 편향을 줄였다. 또한 Movie Render Queue와 runtime logging을 프레임 타임라인으로 동기화해 멀티모달(RGB, depth, normals, audio 등) 신호와 engine state, 중력 라벨을 정렬했으며, 렌더 실패·동기 붕괴·의도치 않은 발산 클립은 필터링했다.

- **Empirical Impact**: 실험에서는 gravity-conditioned video generation에서 fine-tuning(LoRA) 시 중력 크기에 따른 낙하/가속 순서가 회복되는지 확인했다. zero-shot은 중력 요청에 둔감했지만, PhysEditWorld SFT 이후 가속 프록시가 중력과 단조적으로 정렬되고 alignment이 크게 개선됐다. 더 나아가 action-conditioned first-person world modeling과 video-language 중력 추론에서도 성능이 크게 향상되어, editable physics에 대한 감독 신호가 생성·이해 양쪽에 유의미하게 전이될 수 있음을 보여준다.



### DeCoFlow: Structural Decomposition of Normalizing Flows for Continual Anomaly Detection (https://arxiv.org/abs/2606.26687)
- **Prior Approaches**: 연속 이상 탐지는 과거 데이터를 저장하지 못한 채 순차로 새 범주(태스크)를 학습해야 하며, 이 과정에서 catastrophic forgetting이 핵심 문제로 지적된다. 기존 방법은 regularization(EWC, MAS), replay(GEM, DGR), 혹은 PEFT/구조 분리(LoRA 등)로 간섭을 줄이려 했지만, 밀도 기반 모델인 Normalizing Flows(NFs)는 density manifold까지 유지해야 해 더 취약하다.

- **Core Contribution**: 논문은 coupling 기반 NF의 구조적 성질(affine coupling에서 변환 유효성과 Jacobian 계산이 서브넷 내부 파라미터 구조와 무관)을 이용해, 파라미터 격리를 하면서도 flow의 엄밀한 invertibility/likelihood 계산을 깨지 않는 방법을 제시한다. 이를 바탕으로 DeCoFlow는 각 coupling subnet을 frozen universal base와 task-specific low-rank adapter로 분해하고, TSA/ACL/TAL로 frozen-base의 경직성을 보완한다. 또한 Mahalanobis 기반 prototype routing으로 태스크 식별 없이 단일 forward에서 해당 adapter를 선택한다.

- **Technical Challenges**: 주요 기술 난제는 “파라미터 격리로 망각을 막되, NFs가 요구하는 exact Jacobian과 density manifold의 정합성은 유지”해야 한다는 점이다. DeCoFlow는 DCL로 격리를 구조적으로 보장하고, TSA로 입력 분포 편이를 캘리브레이션하며, ACL로 DCL 누적에서 생길 수 있는 잔차 상관을 보정한다. 마지막으로 TAL은 NLL의 tail(저밀도) 패치에 가중치를 주어 LoRA의 제한된 용량이 분포 중심에만 쏠리는 문제를 완화한다.

- **Empirical Impact**: MVTec-AD와 VisA에서 DeCoFlow는 각각 image-level AUROC 98.40%, 93.00%를 달성하며, 파라미터 수준 zero forgetting을 FM=0.00%(correct routing에서)으로 유지한다. 태스크당 추가 파라미터는 2.27M(약 2.5%)로, 재학습 시 저장/개입 없이도 밀도 기반 연속 이상 탐지의 성능-안정성 트레이드오프를 크게 개선한 것으로 보고된다. 특히 routing을 5% 오분류로 강제하면 성능이 급락하지만, 실제로는 Mahalanobis routing이 100% 정확도를 보여 방법의 practical robustness도 뒷받침한다.



### Disco-LoRA: Disentangled Composition of Content, Style, and Motion for Multi-concept Video Customization (https://arxiv.org/abs/2606.26668)
- **Prior Approaches**: 기존 텍스트-투-비디오(T2V) 기반 영상 커스터마이징은 기준 이미지의 외형(콘텐츠) 보존이나 기준 영상의 동작 패턴(모션) 모사에 주로 집중해 왔습니다. 또 일부는 이미지에서 콘텐츠-스타일 조합만 다루거나, 시퀀스 전체를 일대일로 학습하는 방식에 머물러 새로운 조합으로의 확장성이 제한적이었습니다.

- **Core Contribution**: 이 논문은 콘텐츠·스타일·모션을 함께 제어하는 multi-concept video customization을 체계적으로 정의하고, 4가지 조합 과제를 평가하도록 종합 벤치마크를 구축했습니다. 이어 Disco-LoRA를 제안해 콘텐츠-스타일과 콘텐츠-모션을 분해 학습한 뒤 LoRA를 유연하게 재조합함으로써 임의의 다중 개념 동시 제어를 노립니다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 개념을 제대로 disentanglement하는 것과 (2) 조합 시 inter-concept interference를 줄이는 것입니다. 이를 위해 Iterative Dual-LoRA Disentanglement는 단계별로 LoRA를 번갈아 학습하고 complementary prompting 및 time-aware masking으로 정보 누수를 억제하며, Z-score 기반 정규화는 LoRA 레이어별 weight trend은 유지하면서 magnitude를 맞춰 선형 결합 간섭을 완화합니다.

- **Empirical Impact**: 실험에서는 벤치마크 전 과제에서 Disco-LoRA가 외형·스타일·모션 정합성뿐 아니라 운동의 충실도와 매끄러움, 지각 품질(PickScore 등)에서도 우수한 결과를 보였고, 사용자 연구에서도 모든 평가 항목에서 평균 4점대의 점수를 기록했습니다. 특히 기존 방법들이 스타일/콘텐츠/모션이 서로 엉키거나 초기 프레임 불일치가 연쇄적으로 실패하는 문제를 Disco-LoRA가 더 일관되게 해결하며, 상용 모델 대비도 다중 조건 통합 제어 성능 격차를 입증했습니다.



### LayersReg: A Layer-by-Layer Progressive Regressor for Reliable Intraoperative 3D/2D Registration (https://arxiv.org/abs/2606.26647)
- **Prior Approaches**: 기존 3D/2D 레지스트레이션은 이미지 유사도 기반 iterative optimization이 주류였지만, 국소 최소와 실시간 실패율 문제가 컸습니다. 이를 보완하려는 pose regression 방식은 회귀로 빠르게 예측하지만, DRR 등 합성 기반 학습이나 단일 X-ray 의존 탓에 복잡한 수술 상황에서 일반화가 제한됩니다.

- **Core Contribution**: LayersReg는 기존 one-shot pose 회귀를 버리고, 올바른 자세를 layer-by-layer로 점진 정제하는 progressive regression 패러다임을 제안합니다. 또한 CT 볼륨에서 뽑은 3D 해부학적 인지와 2D X-ray의 특징을 교차 차원으로 결합해, depth/occlusion 손실로 인한 한계를 직접 겨냥합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 단일 X-ray의 불완전한 깊이 정보와 occlusion이 학습을 깨뜨리지 않게 하면서 (2) “에러 최소화에 기반한 정합”에 가까운 신호를 네트워크 내부에 구현하는 것입니다. LayersReg는 hybrid autoencoder로 안정적 latent를 만들고, dual correlation gating(DCG)과 trend perception module(TPM)로 moving/fixed 간 특징 잔차의 상관·피셀 흐름 경향을 추출한 뒤 Mamba 기반 백본과 node-wise 회귀로 단계별 누적 오차를 줄이도록 설계했습니다.

- **Empirical Impact**: 실험에서 LayersReg는 large offset과 multimodality 조건에서도 높은 정확도를 보이며 X-ray/CT registration에서 0.68°, 1.41 mm, slice localization에서 0.73°, 1.55 mm를 달성했습니다. 여러 모달리티·해부학 부위·수술 시나리오를 아우르는 7개 데이터셋에서 기존 SOTA를 능가하면서도, 수술실 요구 수준의 정밀도와 실시간성을 함께 만족하는 방향성을 제시합니다.



### FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics (https://arxiv.org/abs/2606.26636)
- **Prior Approaches**: 이 논문은 이벤트 카메라 시뮬레이션에서 주로 대조 임계값 기반 이벤트 생성 규칙을 사용해왔고, 여기에 잡음·필터링·센서 파라미터 튜닝을 추가하는 방식이 많다고 짚는다. 그러나 이런 방식은 픽셀의 시간적 라이프사이클(이전 상태의 잔여 반응, 기준선 갱신 등)을 단순화해 이벤트 타이밍이 왜곡되고, downstream 전이에 약해질 수 있다. 또한 입력(프레임/보간/렌더러)이 만드는 log-intensity trajectory 품질과 센서 변환 규칙이 결합된 구조를 정확히 분리·모델링하기 어렵다는 한계가 있다.

- **Core Contribution**: FracEvent는 sensor-side conversion에 초점을 맞춰 픽셀 단위 event lifecycle을 fractional-relaxation voltage dynamics로 모델링한 이벤트 시뮬레이터다. log-intensity trajectory가 주어지면 다중 relaxation mode를 연속시간으로 구동해 voltage 상태를 만들고, ON/OFF 임계값을 continuous voltage 궤적에서 찾아 이벤트를 방출한다. 특히 이벤트가 발생한 뒤 reference는 갱신하되 memory mode의 잔여 상태는 유지해, 반복 트리거와 이후 이벤트 타이밍의 연결성을 보존한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) trajectory는 외부에서 주어질 때 그 신호가 임계값 교차 시점을 어떻게 바꾸는지, (2) event가 연속적으로 발생하는 동안 기준선 갱신과 잔여 응답이 어떻게 누적되는지, (3) 프레임 구간 내 이벤트 타임스탬프를 grid에 고정하지 않고 연속시간으로 정확히 로컬라이징하는 것이다. FracEvent는 mode들을 분산 완화(distributed-exponential) 형태로 구성해 voltage 상태를 닫힌형(closed-form)으로 업데이트하고, 임계값에서 발생한 event 개수에 따라 구간을 재분할하며 bisection으로 교차 시점을 연속시간에 할당한다. 또한 ON/OFF에 서로 다른 임계값을 두고, 이벤트 간에도 mode 상태를 리셋하지 않는 reference update를 분리해 극성 비대칭까지 반영한다.

- **Empirical Impact**: FracEvent는 event-stream 비교(이벤트 통계, IEI distance, polarity, time surface 등)와 downstream 전이(영상 재구성 E2VID 스타일, 광류 EV-FlowNet 스타일)에서 기존 시뮬레이터 ESIM, v2e, DVS-Voltmeter보다 성능이 좋게 나타났다. 이벤트 스트림 측면에서는 real 데이터에 더 가까운 시간 구조와 극성 균형을 보였고, 주요 구성요소(다중 모드 memory, retained memory state, 분리된 ON/OFF threshold)의 기여도도 확인했다. downstream에서도 동일 프로토콜 고정 하에 시뮬레이션으로 학습한 모델이 더 낮은 MSE/LPIPS(재구성)와 더 낮은 평균 AEE(광류)를 기록해, 시뮬레이션의 실용적 가치와 transfer 효과가 입증됐다.



### Temporally Consistent Label Interpolation for Robust Surgical Multi-Task Learning under Challenging Conditions (https://arxiv.org/abs/2606.26634)
Comments:
          17pages, 16figures

- **Prior Approaches**: 수술 장면 이해에서 기존 연구는 보통 phase/step 같은 시간 작업과 instrument segmentation/action recognition 같은 공간 작업을 따로 최적화해 왔다. 다중작업 학습도 있으나, 시간축은 거의 매 프레임 dense 주석이 가능하지만 공간축은 라벨 비용 때문에 sparse 키프레임만 가능해 학습 균형이 깨지는 문제가 컸다. 또한 SAM2 같은 appearance 기반 마스크 전파는 수술 환경의 occlusion, smoke, motion blur에서 temporally inconsistent pseudo label을 만들 수 있어 한계가 있었다.

- **Core Contribution**: 이 논문은 sparse 키프레임 주석으로부터 temporally consistent dense pseudo labels를 만드는 Flow-guided Annotation for Robust Operating Scenes(FAROS)를 제안한다. FAROS는 SAM2의 zero-shot mask propagation을 기본으로 하되, optical flow 기반 일관성 검사를 통해 실패 구간을 탐지하고 flow로 보정된 재프롬프트를 적용한다. 그 결과 생성된 densified instrument masks와 action labels를 Transformer 기반 단일 멀티태스크 프레임워크에 통합해 phase/step/anticipation과 instrument/action을 동시에 학습한다.

- **Technical Challenges**: 핵심 난제는 수술 영상에서 모양 변화가 잦아 appearance 기반 전파만으로는 마스크 품질이 급격히 무너진다는 점이다. 논문은 RAFT로 forward-backward optical flow consistency를 계산해 occlusion·외관 변화 구간을 이상치로 감지하고, 해당 구간에서는 우측 키프레임의 ground-truth 마스크를 flow warping해 공간적 prior로 SAM2를 reprompting한다. 이후 다중작업 학습에서는 cross-task coupling(phase→step, spatial→temporal conditioning)을 두고, 주석 가용성에 따라 각 loss를 활성화해 dense temporal supervision과 sparse spatial supervision의 균형 최적화를 유도한다.

- **Empirical Impact**: FAROS는 DAVIS 2017에서 sparse ground-truth 프로토콜 하에도 region similarity(J), contour accuracy(F)와 평균(J&F) 지표를 통해 수술 도메인 외에서도 견고한 전파 품질을 확인했다. 이어 GraSP와 MISAW 벤치마크에서 cross-task representation learning이 개선되고, phase/step/anticipation, instrument segmentation, action recognition 전반의 holistic surgical scene understanding 성능이 향상됨을 보였다. 또한 AutoLaparo에서는 downstream segmentation 성능을 분리 평가해, 라벨 보간 파이프라인의 기여를 추가로 뒷받침한다.



### Position Rebinding Cache Reuse: Replay-Free Visual Revisiting for Interleaved Multimodal Reasoning (https://arxiv.org/abs/2606.26631)
- **Prior Approaches**: 인터리브(multimodal) 추론(예: interleaved multimodal chain-of-thought)은 디코딩 중간에 시각 증거를 다시 끼워 넣어 grounding을 강화한다. 기존 방식은 주로 token replay로, 필요한 시각 토큰/영역을 선택해 다시 forward하며 중복 연산이 커진다. KV cache를 그대로 재사용하려는 direct 방식도 시도됐지만, 위치가 고정된 시각 KV의 stale binding 문제가 커서 붕괴로 이어진다.

- **Core Contribution**: 이 논문은 visual KV cache를 단순 복사하면 안 되는 이유를 “positional context에 이미 결속된 키”가 새로운 디코딩 상태에서 충돌한다는 실패 모드로 규명한다. 그 해결로 Position Rebinding Cache Reuse(PRCR)를 제안해, 재생(replay) 없이도 과거 시각 증거를 다시 불러오되 키를 현재 위치에 맞게 재바인딩한다. PRCR은 원(raw) 시각 KV를 저장한 뒤, 현재 디코딩 흐름에 호환되는 좌표로 재배치하고 재구성된 cache만 삽입한다.

- **Technical Challenges**: 핵심 기술 난제는 “위치 바인딩이 포함된 historical visual KV를 현재 attention 맥락에 안전하게 통합”하는 방법이다. PRCR은 RoPE 이전의 raw key/value와 원래 spatial coordinate를 RVEM에 저장하고, Position-Consistent Reinsertion(PCR)으로 텍스트 위치 연속성을 유지하면서 선택된 시각 토큰의 relative 2D 구조가 보존되도록 좌표를 재할당한다. 이후 키는 RoPE로 현재 재할당된 위치에 재바인딩하고 값은 그대로 재사용해, stale offset으로 인한 디코딩 collapse를 막는다.

- **Empirical Impact**: 여러 멀티모달 추론 벤치마크(M3CoT, MathVista, MMStar, MMMU)에서 PRCR은 token replay급 또는 그 이상 성능을 보였다. 예를 들어 Qwen3-VL-8B-Instruct에서 M3CoT 정확도가 66.44%→68.68%로 개선됐고, 재시도(stuck) 없이 안정적으로 동작하는 양상이 보고됐다. 계산 측면에서는 visual-revisiting FLOPs가 수만~수천 배 수준으로 감소(예: 483.18G→14.16M)하면서도 정확도를 유지해, 실사용 효율을 크게 끌어올린다는 점이 의미 있다.



### TaskTok: Delving into Task Tokens for Task-driven Image Restoration (https://arxiv.org/abs/2606.26615)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 task-driven image restoration(TDIR)은 인식에 중요한 의미 단서를 복원하려 하지만, diffusion 기반 생성 프리오어를 쓰는 최근 접근은 대체로 모든 latent token(또는 2D grid)을 일괄 업데이트합니다. 그 결과 이미 입력에서 보존된 신뢰 가능한 단서를 덮어써 semantic drift를 일으킬 수 있고, 업데이트가 계산을 비효율적으로 만듭니다. 또한 2D latent에서는 공간 패치-속성이 얽혀 있어 “어떤 토큰이 과업에 중요한지”를 세밀하게 고르기 어렵다는 구조적 한계가 있었습니다.

- **Core Contribution**: 이 논문은 latent token 공간을 분석해 task-relevant cue가 토큰 인덱스 전반에 균등하지 않고 index-wise specialization 형태로 분포한다는 점을 강조합니다. 이를 바탕으로 모든 토큰을 복원하기보다 과업에 실제로 기여하는 일부 토큰만 선택적으로 복원하면 충분하다고 주장하며, 이를 구현한 TaskTok 프레임워크를 제안합니다. TaskTok은 learnable token switch로 복원할 토큰 subset을 정하고, 그에만 적용되는 lightweight token refinement module로 복원 품질과 성능-효율 균형을 동시에 노립니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 토큰 인덱스 중 과업에 유의미한 토큰을 안정적으로 선택하고, (2) 선택이 비분리적(binarization)일 때도 학습 가능하도록 만드는 것입니다. 논문은 TiTok의 1D 토큰화를 활용해 인덱스별 의미를 “선택 가능한 손잡이”로 만들고, greedy token-importance ordering으로 token switch의 초기 확률을 구성한 뒤 straight-through estimator로 mask 학습을 수행합니다. 또한 JPEG/블러/노이즈 같은 열화가 토큰 의미를 왜곡할 위험이 있어 SwinIR 기반 사전 복원 모듈을 frozen으로 두고, 최종 복원은 디코더에 “연속 특징”을 직접 넣어 quantization이 유발하는 semantic shift를 줄입니다.

- **Empirical Impact**: ImageNet 분류, PASCAL VOC segmentation, object detection 전반에서 TaskTok은 EDTR과 TiTok 대비 더 적은 토큰만 복원하면서도 downstream 성능을 일관되게 개선했습니다. 예를 들어 Mix-B에서 classification 기준 TaskTok-256은 복원 토큰을 34(=3434)개 수준으로 제한하면서 Top-1 정확도를 +4.6%p 및 +3.7%p(각 비교 방법 대비) 끌어올렸고, throughput은 EDTR의 8.3×로 빨라졌습니다. 더불어 CUB200/Oxford-IIIT Pet 및 미학습 classifier 백본에 대해서도 성능 향상이 유지되어, 선택적 토큰 복원이 과적합이 아닌 범용적인 task-relevant 표현 학습으로 이어짐을 보여줍니다.



### LogicIR: Logic Gate Networks for Image Restoration (https://arxiv.org/abs/2606.26609)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 이미지 복원은 DNN(예: DnCNN, SwinIR)으로 성능을 크게 끌어올렸지만, 연산량이 커 스마트폰·웨어러블·임베디드 같은 환경에서 부담이 되었습니다. 경량화로는 LUT(lookup table), 네트워크 프루닝/양자화, BNN(binary neural network) 등이 활발히 연구됐습니다. 다만 BNN/LUT는 잔여 블록에서 여전히 full-precision 연산이 남거나, LUT 크기가 수용영역에 대해 지수적으로 커져 장거리 패턴을 담기 어렵다는 한계가 있었습니다.

- **Core Contribution**: LogicIR은 이미지 복원 전용으로 설계된 최초의 Logic gate network(LGN) 아키텍처입니다. UNet-inspired 구조를 논리 게이트만으로 구성하고, bit decoding layer로 논리 출력(이진 활성)을 연속형 residual로 변환해 복원 작업에 맞췄습니다. 또한 index shuffling으로 논리 게이트 간 정보 흐름을 개선해, 복원에 필요한 계층적 표현을 강화했습니다.

- **Technical Challenges**: LGN은 논리 게이트 선택이 비미분이라 학습이 까다롭고, 단순히 convolutional logic layer만 쌓으면 복원에 필요한 계층·스킵·디코딩 요소가 부족해집니다. LogicIR은 differentiable bit decoding과 MSB loss 보조감독으로 비트 단위 잡음/학습 불안정을 줄였고, STE(straight-through estimator) 기반 fine-tuning으로 학습(soft 선택)과 추론(hard 선택) 간 불일치로 인한 성능 갭을 완화했습니다. 추가로 pixel unshuffle/pixel shuffle, 스킵 연결 채널 concat, rotational ensemble(회전 입력 앙상블)을 함께 적용해 효율과 품질을 동시에 끌어올렸습니다.

- **Empirical Impact**: 여러 이미지 복원 벤치마크에서 LogicIR은 연산 비용을 크게 줄이면서 경쟁력 있는 화질을 보였습니다. 특히 denoising에서 LogicIR-S-4RT는 BSD68 27.71 dB를 169.3G BOPs로 달성해, BBCU-lite 대비 연산은 훨씬 줄이면서 성능도 앞섰습니다. deblocking과 deraining에서도 LUT 기반·BNN 기반 방법 대비 유사하거나 더 나은 PSNR을 더 적은 연산으로 보여, 논리 게이트 기반 접근이 실제 경량 하드웨어 친화형 대안이 될 수 있음을 실증했습니다.



### DiCoBench: Benchmarking Multi-Image Fine-Grained Perception via Differential and Commonality Visual Cues (https://arxiv.org/abs/2606.26602)
Comments:
          Accepted by ECCV 2026. Project page with code: this https URL

- **Prior Approaches**: 기존 MLLM 벤치마크는 질문에 포함된 명시적 텍스트 단서(예: “어디에 있는가?”)를 바탕으로 단일 이미지의 fine-grained grounding을 주로 평가해 왔습니다. 또한 multi-image 과제는 다수 N-gram 기반 생성 평가(ROUGE-L, CIDEr 등)에 의존하거나, 해상도가 낮아 미세 단서 한계를 제대로 측정하기 어렵다는 지적이 누적돼 있습니다.

- **Core Contribution**: 이 논문은 암묵적 시각 단서를 고해상도 다중 이미지에서 스스로 찾아 비교하는 능력을 겨냥한 벤치마크 DiCoBench를 제안합니다. DiCoBench는 765개 샘플을 Differential Visual Cues(차이)와 Commonality Visual Cues(공통성) 두 트랙으로 나누고, 총 8개 인지 태스크를 객관식(MCQ)으로 구성해 평가 편향을 줄였습니다.

- **Technical Challenges**: 핵심 난제는 ‘미세 스케일’ 시각 단서를 보존하면서도 배경 간섭이 큰 고해상도 환경에서 모델이 단서를 자율적으로 추출하도록 만드는 데이터 설계입니다. 이를 위해 마이크로 마스크 기반 편집 파이프라인과 FLUX.2 Klein 같은 high-resolution 편집 모델을 쓰고, MCQ의 정답 균형 및 “No visible difference/commons” 옵션으로 무의미한 추측을 억제했으며, 다단계 인간 검증으로 텍스트 충실도·자연스러움·상호 구별성·스케일 제약(<5%)을 통과시켰습니다.

- **Empirical Impact**: 18개 MLLM을 평가한 결과, 인간 평균 정확도는 98.3%인 반면 모델들은 58.1%(최상, Gemini-3-Pro) 수준에 머물며 큰 성능 격차가 관찰됐습니다. 특히 reasoning 성격의 하위 태스크에서 성능이 크게 붕괴하거나(다수 20~30%대), Commonality 추론에서는 공통성이 없다고 판단해 E를 고르는 경향이 나타나 현 시점 MLLM의 고해상도 cross-image 미세 인지 한계를 실증적으로 드러냈습니다.



### SpaceRipple: Lightweight Semantic Delivery for Mission-Oriented LEO Earth Observation Satellite Networks (https://arxiv.org/abs/2606.26559)
- **Prior Approaches**: 기존 연구는 화상 압축, 위성 탑재 지능(온보드 처리), semantic communication을 각각 따로 다루는 경향이 있었다. 그 결과 전통적인 “획득–전체 다운링크–지상 처리” 흐름을 크게 바꾸지 못해, 제한된 링크/연산 자원에서 임무 민감도를 충분히 높이기 어렵다는 한계가 제기된다. 특히 SR-powered 위성 획득·복원 방식들은 주로 고화질 복원 자체에 초점이 맞춰져 semantic 최종 목적을 위한 시스템 최적화는 상대적으로 약했다.

- **Core Contribution**: SpaceRipple은 위성 관측에서 다운링크까지의 목표를 ‘픽셀 전달’이 아니라 ‘임무 관련 semantic 정보 전달’로 재정의하고, 이를 위해 압축–전송–복원–semantic 추론을 하나의 협업 파이프라인으로 공동 설계했다. sensing 위성은 adaptive compression과 metadata 생성을 수행하고, edge computing 위성은 수신 표현을 복원·강화한 뒤 task에 필요한 장면/탐지 결과를 추출한다. 최종 다운링크는 전체 영상 대신 의미 메시지로 구성해 “관측–전송–의사결정” 지연을 줄이는 구조를 제안한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 압축으로 정보가 손실된 상황에서도 semantic 추론에 필요한 단서를 안정적으로 보존/복원하는 것이다. SpaceRipple은 이를 위해 압축 과정에서 압축률 및 색/보정 관련 calibration cues 같은 side metadata를 함께 전송하고, edge에서 metadata를 조건으로 복원·전처리하도록 설계했다. 또한 압축 열화에 대한 강건성을 위해 compression-aware MoE enhancement 모듈을 두되, 로컬 픽셀 특징과 metadata의 압축률을 함께 라우팅 단서로 사용해 영역별 강화 강도를 적응적으로 조절한다.

- **Empirical Impact**: 실험에서 SpaceRipple은 PSNR, SSIM, LPIPS 등 시각 복원 지표뿐 아니라 ship 인식과 city vehicle 인식 등 task 성능에서 높은 F1을 보여, 압축-복원 후에도 임무 판별 정보가 잘 유지됨을 입증했다. 시스템 측면에서는 contact-window 기반 처리량이 증가하며, 영상 중심 다운링크 대비 데이터 감소율을 크게 끌어올려 전송 효율을 개선했다. 특히 semantic 중심 전달은 단순/풍부 semantic 패킷 모두에서 90%대~99%대 데이터 절감이 가능함을 보여, 링크가 제한된 지구관측 임무에서 효율적·신뢰성 높은 운용 가능성을 시사한다.



### Coarse-to-Fine: A Hybrid Self-Supervised Method for Non-rigid 3D Shape Matching (https://arxiv.org/abs/2606.26557)
- **Prior Approaches**: 비강체(비등방) 3D 형상 매칭에서 기능 맵(functional map)은 정밀 대응을 효율적으로 다루지만, 기존 딥 방식은 주로 정규직교성/쌍사상성 같은 기하 prior에 의존해 자체지도 prior를 충분히 활용하지 못했다. 또한 성능을 올리기 위해 least-squares 선형계 솔버를 쓰거나, 추론 시 fine-tuning을 결합하는 경우가 많아 계산량과 수치 안정성 문제가 함께 따라왔다.

- **Core Contribution**: 이 논문은 거친(coarse) 기능 맵이 정제(refined) 대응과 일관되게 수렴하도록 하는 hybrid self-supervised 학습을 제안한다. coarse-to-fine 전략으로 공간·스펙트럼·공간-스펙트럼 관점의 contrastive energy를 설계하고, refinement module을 통해 점대점·기능 맵을 모두 정제할 수 있게 만든다. 아울러 Laplacian basis(직교) 기반 intrinsic branch와 non-orthogonal elastic basis(비직교) 기반 extrinsic branch를 대칭적으로 두되, 시간 소모적인 least-squares 솔버와 fine-tuning을 제거해 효율을 확보한다.

- **Technical Challenges**: 핵심 과제는 (1) 거친 맵이 정제 맵의 품질 신호를 학습에 안정적으로 반영하도록 self-supervised 제약을 설계하는 것과 (2) 비직교 basis에서도 점대점/기능 맵 정제를 일관되게 수행하는 것이다. 이를 위해 coarse와 refined를 잇는 contrastive energy를 공간·스펙트럼·공간-스펙트럼으로 구성하고, non-orthogonal 설정에서도 적용 가능한 다중 스케일 commutativity 기반 generalized refinement를 도입한다. 또한 필터가 Parseval 조건을 만족하도록 L2 제약을 걸어 매트릭스 inversion 없이도 수치적으로 안정적인 refinement을 가능하게 했다.

- **Empirical Impact**: 저자들은 다양한 어려운 시나리오(비등방 변형, topological noise 포함)에서 기존 최첨단 대비 높은 매칭 정확도와 함께 런타임 효율을 동시에 달성했다고 보고한다. 특히 contrastive energies가 feature discrimination을 촉진한다는 점을 엄밀히 증명하고, 다른 baseline/방법에 결합해도 일관된 성능 향상을 준다고 실증한다. 결과적으로 functional map 학습 파이프라인에서 self-supervised loss의 역할을 정리하면서, 솔버·fine-tuning 없이도 경쟁력 있는 성능을 얻는 실용적 설계를 제시한다.



### Perception, Verdict, and Evolution: Hindsight-Driven Self-Refining Forensics Agent for AI-Generated Image Detection (https://arxiv.org/abs/2606.26552)
Comments:
          10 pages

- **Prior Approaches**: 기존 AI 생성 이미지 탐지는 (1) NPR 같은 통계/주파수 아티팩트를 뽑는 feature-based 방식과, (2) MLLM이 설명(추론)을 제공하는 explainable 방식으로 나뉜다. 그러나 feature-based는 패턴 매칭에 가까워 논리적 일관성 판단이 어렵고, MLLM 기반은 미세한 포렌식 흔이에 대한 민감도가 떨어지거나 GPT-4o·Pixtral-124B 같은 정적 합성 감독에 크게 의존하는 문제가 있었다. 그 결과 실패 사례에서의 반사적 학습과 추론 품질의 반복 개선이 충분히 탐구되지 못했다.

- **Core Contribution**: ForeAgent는 Perception–Verdict 구조로 의미(semantic)·공간(spatial)·주파수(frequency-domain) 신호를 묶어, MLLM을 ‘판정(Verdit)’ 모듈로 사용해 근거 기반의 Real/Fake 결론을 만든다. 또한 Hindsight-Driven Self-Refining을 제안해 추론 실패/저품질 추론을 Sampling–Reflection–Evolution으로 재생성하고, dual-expert quality gating으로 고품질만 선별해 fine-tuning에 반영함으로써 에이전트가 지속적으로 자가 진화하도록 설계했다. 이로써 외부 frontier 모델 의존도와 정적 합성 데이터의 한계를 동시에 줄이는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술적 난제는 (i) 미세 포렌식 흔이를 잡아내면서도 (ii) MLLM 추론이 논리적으로 일관되고 증거에 기반하도록 품질을 보장하는 것이다. ForeAgent는 wavelet transform의 frequency cue(특히 cD 대역)와 NPR의 구조적 흔적을 결합하고, 추론 결과를 Gemini-류 평가가 아니라 dual-expert(자기 자신 Qwen3-VL-8B + Qwen3-VL-Plus) 판정으로 ‘lenient entry, strict exit’ 방식의 질 검증에 통과시켜 로그/추론 궤적의 신뢰도를 높인다. 더불어 반사 학습 시에는 오답 예측과 정답이더라도 저품질 추론을 분리해 후보 추론을 재생성하고, 기준선 이상의 품질은 reasoning supervision으로, 미달은 label-only supervision으로 다르게 학습시킨다.

- **Empirical Impact**: Chameleon에서 ForeAgent는 82.18% 정확도를 달성하며 AIDE 대비 +16.41%p 향상을 보였고, AIGCDetect-Benchmark에서는 16개 생성기에 대해 93.3% mean accuracy를 보고했다. 특히 확산모델 계열에서 성능 격차가 크게 나타나 Midjourney·DALLE2·Stable Diffusion v1.4/v1.5 등에서 기존 설명형/포렌식 기준선을 일관되게 앞섰다. 또한 외부 평가에서 ForeAgent의 추론은 GPT-5 및 GPT-5-mini보다 더 일관적이고 인과적으로 근거가 분명하다고 검증되었으며, self-evolution이 추론 보정에 직접 기여함을 보여준다.



### PhyEditBench: A Real-World Multi-Stage Benchmark for Physics-Aware Image Editing (https://arxiv.org/abs/2606.26551)
Comments:
          19 pages, 6 figures, 2 tables. Accepted to ECCV 2026

- **Prior Approaches**: 기존 instruction-based image editing은 스타일 전환, 색 보정, 단순 객체 교체처럼 비교적 저수준 변환에 강점이 있었지만, 복잡한 물리적 추론이 필요한 지시는 종종 표면적 의미 매칭에 의존했다. 이후 reasoning-centric editing과 관련 벤치마크가 등장했지만, 평가가 주로 공간 배치·논리 퍼즐·속성 결합 중심이라 실제 세계의 동적 물리 과정을 검증하기엔 한계가 있었다. 또한 물리 추론용 데이터는 VQA/비디오 예측 중심이거나 3D 렌더 기반이라, instruction-guided 편집의 요구와 거리감이 컸다.

- **Core Contribution**: 이 논문은 instruction-guided 편집 모델의 물리 기반 추론을 직접 평가하는 벤치마크 PhyEditBench를 제안한다. 계층적 분류(4개 프라이머리 클래스, 12개 서브클래스)로 현실 물리 현상을 체계화하고, 238개의 고해상도 실세계 비디오 기반 인스턴스와 35개의 Anti-Physics(반사실 조건)로 구성해 “틀리더라도 왜 틀리는지”를 더 엄격히 드러내도록 설계했다. 또한 기존 모델들이 물리적으로 그럴듯한 상태 전이를 얼마나 못 하는지 대규모로 분석하고, 이를 개선하기 위한 학습 없이 동작하는 학습-프리 baseline PhyWorld를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 편집을 단일 이미지 변환이 아니라 시간적 상태 전이로 모델링하면서, 중간 체크포인트까지 물리 일관성을 유지하는 것이다. 저자들은 pretrained video generation 모델의 시간적 인과성(중간 프레임)을 암묵적 reasoning 토큰으로 해석하고, Test-Time Scaling(TTS)과 latent reduction 전략을 결합해 품질 향상과 계산 효율을 동시에 노린다. 또 단일 end-state 비교가 아닌 input–intermediate1–intermediate2–output의 다단계 평가를 통해, 최종 결과는 맞아도 궤적이 물리적으로 어긋나는 오류를 분해해 진단 가능하게 했다.

- **Empirical Impact**: 실험 결과, 기존 SOTA 편집 방법들은 일반 데이터에서도 종종 물리적으로 불가능한 아티팩트나 상태 전이가 없는 “붙여넣기” 양상을 보이며, 물리 추론에 구조적으로 취약함이 드러났다. Anti-Physics에서도 대부분의 모델이 반사실 조건을 견디지 못하는 가운데, PhyWorld는 가장 강한 오픈소스 성능으로 격차를 줄이며 video 생성 과정을 reasoning 엔진처럼 활용할 수 있음을 보여준다. 인간 평가와의 Kendall tau 상관까지 확인해 VLM 기반 자동 평가가 물리적 타당성과 명령 수행 같은 추론 차원을 비교적 신뢰성 있게 측정한다는 점에서, 향후 물리 기반 편집 연구의 표준 도구 역할을 할 가능성이 크다.



### From Hallucination to Grounding: Diagnosing Visual Spatial Intelligence via CRISP (https://arxiv.org/abs/2606.26535)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 VLM 공간 평가들은 QA 정답을 맞히는 데 초점이 맞춰져 언어 priors에 의한 semantic shortcut을 섞어버리는 경향이 있었다. 또한 2D 기반 기도(인지 맵)나 텍스트 근거는 오류와 환각이 많고 3D metric 정밀도를 충분히 검증하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 논문은 VLM의 ‘공간 추론’이 진짜 3D 구조에 근거하는지 진단하기 위한 CRISP(Consistency of Reasoning In Spatial Perception) 평가 패러다임을 제안한다. QA(명시적 추론)와 3D Scene Graph Construction(SGC, 구조 외재화)을 함께 요구하고, 둘 사이의 정합성과 일관성을 정량화해 인지-추론 분리(disconnect)를 측정한다.

- **Technical Challenges**: 핵심 기술적 난제는 모델의 내부 3D 공간 이해를 ‘보이는 출력’으로 강제하면서도, 2D 라벨링·문장 템플릿 같은 지름길을 최소화하는 것이다. 이를 위해 metric 3D Scene Graph를 생성하게 하고(객체 크기·거리·관계의 방향 및 유클리드 거리), SGC 점수(기하 정밀도+관계 의미정확도)와 self-consistency(그래프→추론 도출된 QA가 직접 QA와 일치하는지)를 함께 설계했다.

- **Empirical Impact**: 1,162개 장면(총 9,839문항)로 13개 VLM을 평가한 결과, proprietary 모델은 잠재 추론 엔진은 강하지만 SGC의 metric 추정 오류와 그래프-추론 미활용 문제가 두드러졌다. 반대로 open-source 모델은 QA는 맞히더라도 SGC의 구조적 근거가 부족해 multi-hop compositional reasoning 병목이 확인되었고, 일부는 consistent hallucination이 10~12% 범위로 나타났다. CRISP는 end-to-end post-training만으로 ‘맞히기’를 늘리는 방식에서 벗어나, embodied AI에 필요한 ‘perceiving-verify-reasoning’ 정렬 로드맵을 제공한다.



### Forget, Anticipate and Adapt: Test Time Training for Long Videos (https://arxiv.org/abs/2606.26515)
Comments:
          ECCV 2026. GLOM/APM's temporal binding now works for long videos

- **Prior Approaches**: Test Time Training(TTT)은 라벨 없이도 추론 중 self-supervised task를 수행해 가중치를 업데이트하는 적응 기법이다. 하지만 비디오에는 sliding window 방식이 주로 쓰이는데, 매 스텝마다 윈도우 전체를 다시 계산해 계산량이 프레임 수에 비례해 폭증한다.
또한 시간적으로 가까운 프레임이 비슷해도 여전히 TTT를 수행해 불필요한 업데이트가 발생한다.

- **Core Contribution**: 이 논문은 long-video에 TTT를 적용할 때, 윈도우 진입/이탈 프레임만 처리하도록 Frame Forgetting Network(FFN)를 제안한다. 모델은 sliding window 안에서 “나가는 프레임 잊기(forget)·현재 처리·다음 프레임 기대하기(anticipate)”의 3프레임 전략을 쓰면서도 장기 시간 맥락을 유지한다.
핵심은 incoming frame이 과거에 비해 얼마나 “놀라운지(surprise)”를 수학적으로 정의해, 적응이 필요한 시점에만 업데이트하도록 adaptive windowing을 구성한 점이다.

- **Technical Challenges**: FFN이 직면한 기술적 과제는 두 가지다: (1) 윈도우가 이동할 때 과거 TTT 효과를 효율적으로 되돌려야 하고, (2) 픽셀 변화만으로는 불필요한 적응을 걸러내기 어려워 “새 정보” 기준을 안정적으로 만들어야 한다. 이를 위해 Memory Restoration Mechanism(MRM)과 Adaptive Window Algorithm(AWA)를 설계하고, time 정보를 temporal module로 주입해 나가는 프레임의 적응 흔적을 복원한다.
또한 next-frame anticipative head로 시각적 예측 차이를 산출하되, 고수준 latent feature 유사성을 결합해 회전·흔들림 같은 잡음성 변화를 줄이며 surprise가 동적으로 임계값을 넘을 때만 TTT를 수행한다.

- **Empirical Impact**: EpicTours 데이터셋(최대 3시간, 보행 도시 투어, 30개 클래스 부분집합)을 새로 구축해 실사용에 가까운 긴 영상 설정을 만든 것이 함께 주목된다. 실험에서 FFN은 dense segmentation, depth-estimation 일반화(총 66개 데이터), coarse action 분류에서 경쟁력을 보였고 multi-hour 영상에서도 개선을 보였다.
특히 FFN은 기존 TTT-online보다 더 나은 compute-accuracy tradeoff를 보이며, 긴 영상(시간 단위)에서도 로컬 on-device 수준의 운용 가능성을 실증했다.



### Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking (https://arxiv.org/abs/2606.26455)
- **Prior Approaches**: 기존 RGB-Event 추적은 RGB 외형 질감과 이벤트의 시간 모션 단서를 결합하는 멀티모달 퓨전을 중심으로 발전해 왔다. 하지만 많은 방법이 입력이 온전하다고 가정하며, 실제 환경에서 흔한 structured degradation(모달리티 단위 열화/누락, 타깃의 국소 가림·잘림)을 체계적으로 다루지 못한다. 그 결과 한 모달이 급격히 불완전해지거나 타깃이 부분적으로만 관측되는 상황에서 융합이 쉽게 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 모달리티 누락과 국소 타깃 부재에 강인한 RGB-Event object tracking 프레임워크 APRTrack을 제안한다. 핵심은 계층적(hierarchical) perturbation으로 구조화된 열화를 재현하면서, Footprint-guided Channel-calibrated Hopfield Retrieval(FCHR)로 신뢰도 기반 역사 정보 보상을 ‘타깃 영역에 한정’해 수행하는 것이다. 단순한 데이터 보정이나 무차별 템플릿 교체가 아니라, 누락 유형별로 역할을 분리해 복원 효과를 제어한다.

- **Technical Challenges**: 첫째, 실제 열화는 무작위 토큰 드롭처럼 불연속 노이즈가 아니라 ‘모달 단위 실패’와 ‘공간적으로 연속된 타깃 부재’처럼 의미가 있는 구조를 갖는다. APRTrack은 두 종류를 서로 다른 adversarial perturbation branch로 따로 학습시키고, hierarchical routing으로 clean/modality/spatial 경로를 분리해 결합 열화가 만드는 feature collapse를 줄인다. 둘째, 역사 메모리 retrieval은 국소 가림 상황에서 오동작할 수 있어 신뢰도 평가와 보상 강도 제어가 필요하며, FCHR은 association footprint 기반 채널 캘리브레이션과 gated residual fusion으로 이 문제를 완화한다.

- **Empirical Impact**: FE108, COESOT, VisEvent, FELT 등 4개 대형 벤치마크에서 APRTrack의 전략이 RGB-Event visual object tracking의 성능과 강건성을 개선함을 실험으로 확인했다. 특히 모달리티가 부분적으로 사라지거나 타깃이 국소적으로 누락되는 상황에서 역사 보상이 더 안정적으로 작동하도록 설계된 점이 주요한 의미로 해석된다. 코드와 사전학습 모델 공개 계획도 함께 제시되어, 이후 missing-robust multi-modal tracking 연구에 실용적인 기준점이 될 전망이다.



### Methane-Plume Segmentation From Hyperspectral Satellite Imagery Via Multimodal Deep Learning (https://arxiv.org/abs/2606.26416)
Comments:
          Accepted at IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2026

- **Prior Approaches**: 메탄 플룸 탐지는 희소한 공간 분포, 시간에 따른 변동성, 낮은 신호 대 잡음비 때문에 기존엔 어려운 문제로 남아 있다. 기존 딥러닝은 methane enhancement 같은 가스 유도 표현만 단독으로 쓰거나(VGG16+UNet, SegFormer 계열), CNN으로 enhancement 맵에서 점·마스크를 직접 추정하는 방식(MethaNet)처럼 단일 정보원에 치우치는 경향이 있었다. 하이퍼스펙트럴 기반 UNet은 이상 강조를 돕기도 했지만, RGB의 지형·인프라·형상 맥락을 충분히 결합하지 못해 잡음/거짓 양성에 취약할 수 있다.

- **Core Contribution**: 이 논문은 RGB와 methane enhancement를 함께 쓰는 멀티모달 메탄 플룸 세그멘테이션 모델을 제안한다. 핵심은 feature-guided methane enhancement(FGME)로, methane enhancement에서 얻은 물리적으로 의미 있는 단서를 RGB의 transformer 표현에 다중 의미 스케일에서 주입해 플룸에 일관된 패턴을 더 잘 강조한다. 결과적으로 enhancement-only 접근보다 더 물리적으로 그럴듯한 플룸 경계를 만들도록 설계됐다.

- **Technical Challenges**: 가장 큰 기술 난제는 서로 다른 성격의 입력(RGB의 시맨틱 구조 vs enhancement의 가스 신호)을 스케일별로 정렬·결합하면서, enhancement가 RGB 표현을 ‘대체’하기보다 ‘강화’하도록 유도하는 것이다. 논문은 DINOv3(여러 transformer block tap)로 RGB 다중 스케일 특징을 만들고, ResNet-18로 enhancement 특징을 계층적으로 뽑은 뒤, 스케일별 게이팅에서 sigmoid 마스크를 적용하되 additive bias로 RGB의 정보를 완전히 억누르지 않게 했다. 또한 SegFormer 디코더와 가벼운 refinement 헤드, Dice+Focal의 결합 손실로 불균형과 경계 품질을 동시에 맞췄다.

- **Empirical Impact**: MPDataset(EMIT 기반) 실험에서 제안 방법은 기존 최고 성능 대비 MIoU +0.92, MPrecision +0.87, Recall +1.01로 일관된 향상을 보였다. 특히 FLOPs 기준으로 MPSUNet과 VGG16+UNet 대비 약 80% 적은 계산량을 달성하면서도 추론 시간 7.06 ms 수준의 경쟁력을 유지해 대규모 모니터링에 유리한 정확도-효율 트레이드오프를 제시한다. 정성 비교에서도 enhancement 근거와 정합성이 높고 거짓 양성을 줄이는 경향이 확인되어, 원격탐사 실사용에서 스케일 확장 가능한 멀티모달 융합 전략의 가능성을 보여준다.



### Neural Voxel Dynamics: Learning Implicit 3D Physics via Volumetric Feature Advection (https://arxiv.org/abs/2606.26410)
- **Prior Approaches**: 기존 물리 일관 비디오 생성은 크게 (1) PBD/MPM 같은 고전 시뮬레이터를 결합한 explicit neural-physics hybrid와 (2) diffusion 등으로 픽셀/2D latent을 생성하는 implicit 2D 방식으로 나뉜다. 전자는 시뮬레이션 갭과 물성/재질 유형을 사전에 지정해야 하는 제약이 있고, 후자는 3D 기하·폐색·물체 연속성의 유도 편향이 부족해 상호작용에서 물리적으로 자연스럽지 못해진다. 또한 2D 궤적/모션 가이던스 접근은 제어가 휴리스틱에 머무르며, 시점이 다른 장면에서의 3D 정합성까지 자동으로 보장하기 어렵다.

- **Core Contribution**: 이 논문은 단안 비디오에서 얻는 supervisory signal과 action 조건만으로, 3D 물리 동역학을 implicit하게 학습하는 self-supervised 프레임워크를 제안한다. 핵심은 2D latent의 예측 병목을 ‘lifted’ 3D volumetric latent space로 옮기기 위해 V-JEPA의 의미 특징을 단안 depth priors로 voxel 격자에 unproject(3D 리프팅)한다는 점이다. 이후 Volumetric Feature Advection이 action-conditioned 전이 연산자로 3D state advection(시공간 상태 전파) 형태를 학습해, 재질이 다른 현상도 하나의 파이프라인에서 시뮬레이션처럼 생성되게 만든다.

- **Technical Challenges**: 가장 큰 기술적 난제는 단안 관측에서 가려진 voxel(미관측)을 포함해 3D 상태를 어떻게 일관되게 추적·예측하느냐이다. 저자들은 V-JEPA 특징을 3D voxel에 매핑하되, occupied/observed/unobserved 채널로 공백과 폐색을 함께 모델링하고, sparse voxel 토크나이징과 디퓨전 트랜스포머(DiT) 변형으로 연산 복잡도를 관리한다. 전이 학습은 flow matching 기반으로 time-dependent velocity field를 예측해 확률적(비결정적) 상호작용과 유체/충돌 같은 다중 양상을 다루며, occupancy-weighted/focal loss와 NeRF-like 투영 손실 및 autoregressive rollout(rollout step=3)을 통해 장기 안정성과 기하 정합성을 확보한다.

- **Empirical Impact**: 실험에서는 CLEVRER, PhysInOne, PhysGaia 등 다수 벤치마크에서 2D world model 대비 물리적 plausibility와 의미 일관성을 개선했다고 보고한다. 특히 물리 엔진의 내부 상태·라벨·서로게이트 모델 없이 end-to-end로 학습하면서도 long-term structural stability와 물리적 그럴듯함을 보이며, 단안/다중 시점 및 GT depth 유무 조건에 따라 Ours-GT와 Ours-estimate가 각각 강점을 보인다. 결과적으로 ‘3D 불변성(physical invariants)을 패시브한 단안 비디오 관측만으로 내재화하는 general-purpose dynamic world model’로 가는 확장 가능한 경로를 제시한다는 점에서 의의가 크다.



### DinoLink: A Token-Centric Representation Compression Framework for Bandwidth-Constrained Collaborative V2X Perception (https://arxiv.org/abs/2606.26398)
- **Prior Approaches**: 기존 Vehicle-Cloud Collaborative Inference는 분할(splitting) 컴퓨팅 방식으로 차량에서 연속 실수(Float32) 특징맵을 전송하는 경우가 많지만, 특징 크기가 커지면 V2X 대역폭/지연 병목이 심해집니다. 또한 JPEG·WebP·H.264 같은 픽셀 기반 코덱은 HVS에 최적화되어 고주파·색상 성분을 과감히 제거하고, 그 과정에서 기계 비전이 필요로 하는 정교한 의미 신호와 국소 그래디언트가 훼손된다는 한계가 있습니다.

- **Core Contribution**: DinoLink는 픽셀 스트리밍을 버리고, DINOv2에서 뽑은 의미 토큰을 ‘이산화된 의미 기반 통신’ 형태로 전송하도록 설계된 토큰 중심 압축 프레임워크를 제안합니다. Saliency-Aware Top-KK로 배경 토큰을 먼저 제거한 뒤, Residual Vector Quantization(RVQ)으로 연속 특징을 코드북 인덱스로 바꿔 전송 페이로드를 엄격히 제한합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 토큰을 얼마나 줄일지(공간 희소성)와 (2) 얼마나 잘 이산화해도 downstream 정확도를 유지할지(RVQ 양자화 품질) 사이의 균형을 맞추는 것입니다. DinoLink는 주의(attention) 기반 살리언시로 Top-KK를 결정하고, RVQ 인덱스와 위치 positional prior만 전송한 뒤 토큰 decoder가 복원하여 DETR 같은 서버의 downstream 모델이 그대로 처리할 수 있게 하는 방식으로 이 문제를 해결합니다.

- **Empirical Impact**: nuScenes에서 DinoLink는 uncompressed 전송 대비 139× 비트레이트 감소를 달성하면서도 32.8% mAP로 경쟁력 있는 성능을 유지합니다. 통신 시뮬레이션에서는 LoRa 같은 좁은 대역에서 34.5× 지연 가속(예: 349.56s→10.11s)을 보였고, 실제 차량-서버 LAN 실험에서도 프레임당 전송량을 약 0.26KB 수준으로 낮추면서 AP@0.5를 WebP/JPEG 기준과 유사하게 유지해 현장 적용 가능성을 뒷받침합니다.



### Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs (https://arxiv.org/abs/2606.26387)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 MLLM 정렬은 주로 DPO 같은 outcome-level 보상 최적화로 최종 텍스트의 선호도를 맞추는 방식이 많습니다. 하지만 이런 방법은 정답이 맞더라도 언어 priors만으로 맞춘 ‘잘못된 이유(right for the wrong reasons)’를 걸러내기 어렵고, 그 결과 시각 입력과의 인과적 연결이 약해져 환각이 남을 수 있습니다. 그 외 시도는 아키텍처를 늘리거나(모듈화) 추론 시 대비/패치에 의존해(visual contrastive decoding) 계산·운영 비용 또는 민감도 문제가 동반됩니다.

- **Core Contribution**: 논문은 시각 환각의 핵심 원인으로 ‘visual laziness’를 지목하고, 모델이 보는 상태와 시각을 가린 상태의 차이를 학습 목표에 직접 포함하는 VIGIL을 제안합니다. VIGIL은 생성 응답 y가 시각 입력 xv에서 얻는 정보량을 최대화하도록, counterfactual blind state를 사용한 강화학습(post-training) 정렬 프레임워크를 설계합니다. 특히 시각-언어 priors 간 최적화 지름길이 강화되지 않게 “시각이 없어도 자신 있게 말하는” 경우를 억제하는 데 초점을 둡니다.

- **Technical Challenges**: 기여를 구현하는 가장 큰 과제는 ‘정답 맞추기’가 아니라 ‘시각 증거에 의존했는지’를 학습 신호로 바꾸는 것입니다. VIGIL은 forward 단계에서 주의(attention) 마스킹으로 counterfactual blind state를 만들고, seeing 상태 대비 blind 상태에서의 확률 변화에 기반한 geometric constraint(Visual Information Gain 및 Counterfactual Visual Decoupling 손실)를 부여합니다. 또한 hard negative 및 dynamic gating으로 최적화의 안정성과 시각 의존 샘플에 대한 학습 집중을 동시에 확보합니다.

- **Empirical Impact**: 실험에서 VIGIL은 POPE, AMBER, MMHal-Bench 등 환각·이유 추론 벤치마크 전반에서 최신 정렬 기법을 일관되게 능가합니다. 더 나아가 25% 수준의 preference 데이터만으로도 풀데이터 수준 성능을 맞추고, 기준선 대비 text-only 능력 저하 없이 성능을 유지합니다. 또한 bounding box 같은 명시적 localization 감독 없이도 RefCOCOg에서 공간 grounding 능력이 향상되는 emergent 현상을 보여, 환각 억제가 곧 시각적 이해 강화로 이어질 수 있음을 시사합니다.



### What Do Deepfake Benchmarks Measure? An Audit Using Frozen Self-Supervised Representations (https://arxiv.org/abs/2606.26384)
Comments:
          14 pages, 9 figures

- **Prior Approaches**: 기존 딥페이크 탐지 연구는 영상·이미지·음성 각각에서 대규모 벤치마크로 성능을 끌어올려 왔지만, 실제 환경(in-the-wild)에서는 성능이 크게 붕괴하는 일반화 갭이 반복적으로 보고돼 왔다. 또한 최근 분석들은 벤치마크 점수가 포렌식 이해(forensic understanding)보다 shortcut learning에 의해 부풀려질 가능성을 제기한다. 그런데 정작 “그 벤치마크가 무엇을 측정하는가”에 대한 검증은 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 새로운 탐지기를 제안하기보다, 벤치마크 자체를 감사(audit)하는 프레임워크를 제시한다. 냉동(frozen)된 일반 목적 self-supervised 표현에서 각 레이어에 대해 선형 프로브만 학습해, 벤치마크 점수가 과연 포렌식 신호를 요구하는지(아니면 일반적 modality 이해만으로도 가능한지)를 진단한다. 동시에 생성기 난이도를 표현 공간의 분포 기하(Fréchet 기반)로 설명해, 어떤 생성기가 왜 더 쉬운지까지 함께 분석한다.

- **Technical Challenges**: 핵심 기술적 난제는 “태스크 전용 아키텍처나 fine-tuning 없이” 벤치마크 성능을 재현 가능한지 확인하는 설계에 있다. 이를 위해 SSL 백본을 고정하고 head는 로지스틱 회귀/릿지 같은 저용량 선형 분류기로 제한했으며, 선형 분리 가능성이 벤치마크의 포렌식 요구도를 대체로 반영하는지 점검한다. 또한 생성기 난이도 분석은 각 생성기 그룹을 SSL 표현 공간에서 가우시안으로 근사하고, source real과 source spoof 사이의 상대적 간격 Δ가 퍼포먼스와 어떻게 연동되는지 레이어별로 상관을 측정하는 방식으로 수행된다.

- **Empirical Impact**: 세 가지 모달리티(영상 AIGVDBench, 이미지 Celeb-DF++, 음성 MLAAD v9/ASVspoof2019 LA 프로토콜)에서 선형 프로브 성능이 전용 탐지기 성능에 상당히 근접하거나 경쟁 수준을 보였고, 특히 비디오에서는 상위 탐지기 대비 약 1.3 AUC 이내로 따라가는 결과가 보고됐다. 생성기 난이도는 레이어별로 다르지만, 전반적으로 Δ=d_real−d_spoof의 상대적 마진이 가장 일관된 예측변수로 나타나 “이미 본 spoof 구조를 재포착하기 쉬운 생성기”가 점수에 중복 기여할 수 있음을 시사한다. 결론적으로 이 결과는 벤치마크 고득점을 포렌식 이해의 증거로 읽기 전에, 얼마나 일반 목적 SSL 표현만으로도 풀리는지 먼저 점검해야 한다는 benchmark-audit 관점을 강화한다.



### Layer-Specific Prompt Fusion Discovery via Differentiable Search in Vision Foundation Models (https://arxiv.org/abs/2606.26379)
Comments:
          ECCV 2026

- **Prior Approaches**: 비전 프롬프트 튜닝(VPT)은 대규모 ViT를 하류 과제에 맞추기 위해 프롬프트만 학습하는 PEFT로 자리 잡았다. 다만 프롬프트와 이미지(토큰)를 결합(fusion)할 때는 주로 concatenation이나 addition 같은 단일 방식만 사용해, 어떤 결합이 더 나은지(혹은 조합이 필요한지)는 체계적으로 규명되지 않았다.

- **Core Contribution**: 이 논문은 프롬프트-이미지 결합 방식을 ‘레이어별로 자동 선택’하는 하이브리드 fusion 탐색 문제로 정식화한다. learnable prompts와 fusion 스킴을 함께 학습하기 위해 bi-level optimization과 differentiable neural architecture search(DARTS)를 결합했고, concat/add 외에도 affine transformation, cross-attention을 추가 탐색 공간으로 제안한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) frozen ViT 백본 인터페이스를 바꾸지 않으면서도 결합 연산을 충분히 풍부하게 탐색하는 것과 (2) DARTS가 조기 붕괴/비용-정확도 불균형을 겪지 않게 안정화하는 것이다. 이를 위해 토큰 수 k×d 인터페이스를 고정하고, 연산 혼합 가중치에 entropy 기반 안정화와 비용 정규화를 넣어 탐색 중 단일 연산으로 수렴하도록 설계했으며, 탐색 후 argmax로 이산화 후 짧게 파인튜닝한다.

- **Empirical Impact**: VTAB-1k(19개 데이터셋), FGVC, HTA를 포함해 총 34개 데이터셋에서 프롬프트 튜닝 베이스라인 대비 일관된 성능 향상을 보였다. 또한 frozen ViT 백본 조건에서 parameter를 약 0.75%만 튜닝하면서도 VPT-Deep 및 최신 변형들과 비교해 accuracy–latency–parameter trade-off가 유리하다는 점을 입증했다. 특히 과제 성격에 따라 얕은 레이어는 concat/add에, 깊은 레이어는 affine/cross-attention에 더 많이 배치되는 레이어별 스케줄이 domain sensitivity 완화와 일반화에 중요함을 보여주며, 하이브리드 fusion이 단순 프롬프트 내용 개선을 넘어서는 관점을 제시한다.



### Beyond Aesthetics: Quantifying Information Loss in Turbid Scenes (https://arxiv.org/abs/2606.26295)
- **Prior Approaches**: 기존 수중 비전 연구는 PSNR·SSIM 같은 참조 기반 품질 지표나 UCIQE·UIQM·NIQE·BRISQUE·DIIVINE 등 무참조 지표를 활용하는 경우가 많았지만, 실제로는 모델 성능 저하를 잘 설명하지 못한다는 결과가 축적돼 왔다. 특히 합성 탁도 데이터는 underwater image formation model 등으로 생성되더라도 실제 고탁도에서 나타나는 구조 손실(경계 흐림·가려짐)을 충분히 반영하지 못할 수 있다는 지적이 있었다. 또한 실세계 고탁도 전 범위 조건을 커버하며 인스턴스 분할 같은 고신뢰 라벨을 제공하는 공개 데이터셋이 부족해, 성능-탁도 상관을 정면으로 검증하기 어려웠다.

- **Core Contribution**: 논문은 실제 고탁도 환경에서 탁도가 컴퓨터비전 과제에 주는 정보 손실을 정량화하기 위해 Turbid Underwater Baseline(TUB) 데이터셋과 Phase-Congruency Delentropy(PCD) 지표를 함께 제안한다. TUB는 1,320장의 극한 탁도 영상과 16,000+ 인스턴스 분할 마스크를 제공하며, 장면·카메라 구성을 고정해 탁도만 비교 가능하도록 설계됐다. PCD는 phase congruency 기반 표현으로 대비/색 변화에 덜 민감하면서 구조 정보의 붕괴를 포착해, “시각적 품질 저하=작업 유용성 저하”가 아닐 수 있음을 구조 관점에서 설명한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 실세계 탁도에 대해 충분히 라벨된 데이터 확보, (2) 기존 품질/복잡도 지표가 대비·통계 변화에 흔들리는 문제, (3) 합성 탁도가 실제 정보 손실과 같은 의미를 갖는지 비교 가능한 공통 척도 설계다. TUB는 탱크에 oat milk를 단계적으로 투입하고 nephelometer 측정값을 NTU 기준으로 재그룹해 저/중/고 탁도로 균형 있게 나누었고, 장면이 고정된 상태에서 clear 조건의 주석을 탁도 시퀀스로 전파해 고신뢰 경계 라벨을 유지했다. PCD는 delentropy에서의 커널 기반 gradient를 phase congruency 기반 다중스케일 맵으로 대체해 대비 손실에는 강인하면서도 구조적 경계 붕괴에 민감하도록 구성했다.

- **Empirical Impact**: 실험에서는 Mask R-CNN, YOLOv11, Mask2Former에 대해 탁도 수준이 올라갈수록 인스턴스 분할 성능이 전반적으로 하락하되, 학습에 탁도 데이터가 포함되면 중·고탁도 구간에서 성능이 개선되는 경향을 보였다. 품질 지표·정보 지표 다수는 실세계 탁도에서도 성능과의 상관이 약했지만, PCD는 실데이터와 합성 탁도 모두에서 모델 성능과 강한 상관을 보였다. 특히 합성 생성물이 실제와 다르게 “보이는 품질” 기준으로는 저평가되거나 다르게 점수화되는 반면, PCD는 구조 손실 관점에서 더 일관되게 점수를 매겨 합성 탁도의 현실성/정보 손실 대응 여부를 평가하는 데 의미가 있음을 시사한다.



### GeMoE: Gating Entropy is All You Need for Uncertainty-aware Adaptive Routing in MoE-based Large Vision-Language Models (https://arxiv.org/abs/2606.26287)
- **Prior Approaches**: 기존 MoE 기반 LVLM들은 주로 static Top-k 라우팅을 써서 토큰마다 고정된 수의 expert를 선택했다. 이 방식은 토큰마다 정보량이 다르다는 점을 반영하지 못해, 정보가 낮은 토큰에 자원이 낭비되거나 정보가 높은 토큰이 과소 할당되는 문제가 생긴다. 동적 라우팅은 pseudo-expert, threshold 기반으로 개선을 시도하지만 대체로 휴리스틱에 의존해 ‘모델 복잡도-성능’ 균형을 명시적으로 최적화하기 어렵다.

- **Core Contribution**: 이 논문은 token routing을 정보 인코딩 관점으로 재정의하고, 동적 라우팅을 Minimum Description Length(MDL) 최소화 문제로 공식화한다. 또한 MDL과 MoE의 gating entropy가 information gain을 통해 연결된다는 분석을 바탕으로, gating entropy 기반 Uncertainty-aware Adaptive Routing인 GeMoE를 제안한다. GeMoE는 토큰의 불확실성(gating entropy)에 따라 각 토큰이 참여할 expert 수를 적응적으로 결정해 복잡도와 성능의 trade-off를 직접 설계한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘MDL을 직접 계산하지 못하는 상황’에서 expert 추가가 실제로 information gain을 만들어 MDL을 줄이는지 판단할 대리 지표를 찾는 것이다. 논문은 gating entropy를 proxy로 사용해, entropy가 높을수록 expert 추가로 얻는 정보 이득이 커지고 결과적으로 MDL이 감소한다는 연결고리를 세운다. 이어 gating entropy를 입력으로 expert assignment를 예측하는 Expert Assignment Predictor(EAP)와 monotonic loss를 도입해, entropy가 높을수록 더 많은 expert가 배정되도록 학습을 강제하고 load balancing loss까지 함께 최적화한다.

- **Empirical Impact**: 여러 백본과 벤치마크에서 GeMoE는 원래 static routing 대비 평균 성능 보존율 99.5%를 달성하면서, expert activation sparsity는 평균 36.5% 향상시켰다. 이는 단순히 계산을 줄이는 수준이 아니라, 입력 토큰의 불확실성에 맞춰 자원을 효율적으로 재배치함으로써 성능 하락 없이 효율을 끌어올렸다는 의미다. 결과적으로 MoE 라우팅을 MDL-정보이득-entropy로 연결한 프레임이, 실전 배포에서 중요한 ‘추론비용 대비 성능’ 최적화에 유용함을 보여준다.



### Beyond Single-Source Cognitive Taskonomy:Multi-Source Task Relations through fMRI Transfer Learning (https://arxiv.org/abs/2606.26279)
- **Prior Approaches**: 기존 reconstruction-based fMRI taskonomy는 masked fMRI reconstruction을 공통 자기지도 목표로 삼되, 한 소스 과제에서 한 타깃 과제로의 one-to-one(단일 소스-단일 타깃) 전이를 주로 분석했다. 그 결과 방향성(s→t)과 패러다임 내부 구조는 잘 드러나지만, 타깃이 여러 인지 구성요소의 결합으로 이루어진다는 점에서 many-to-one(여러 소스-단일 타깃) 관계는 충분히 포착하지 못했다. 또한 제한된 라벨 예산에서 어떤 과제에 직접 supervision을 줄지의 전역 우선순위를 전이 강도만으로는 직접 결정하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 기존의 단일 소스 전이 분석을 확장해, 23개 HCP task state에서 여러 소스 표현을 함께 쓰는 source-set–target 구성(S→t)으로 many-to-one 전이 관계를 정량화한다. 동시에 Boolean Integer Programming(BIP)로 예산 제약 하에서 어떤 task state에 직접 supervision을 배정할지 전역 과제 할당을 계산해, 로컬 전이 강도와 글로벌 우선순위를 분리해 보여준다. 총 1,127개(금모델 포함) task-specific 및 transfer 모델로 이러한 계층적 관계를 체계적으로 비교했다.

- **Technical Challenges**: many-to-one 전이를 측정하려면, 여러 frozen source encoder의 표현을 결합해도 동일한 reconstruction objective에서 공정 비교가 가능해야 한다. 논문은 타깃을 위한 저데이터 적응에서 source encoder는 고정하고, 결합 후 투영(projection)과 target decoder만 학습하는 방식으로 구성요소 조합이 성능에 미치는 영향을 직접 측정했다. 또한 target별로 후보 소스를 제한(상위 5개 single-source 후보)하고, 그 후보 풀 안에서만 source-set 조합 효과를 해석하도록 설계해 과도한 일반화 위험을 줄였다. 마지막으로 BIP의 비용 행렬(전이 거리)과 제약(자기 커버 및 예산)을 명시해 전역 할당 결과가 로컬 pairwise affinity와 어떻게 달라지는지 추적했다.

- **Empirical Impact**: 실험 결과 single-source 전이는 방향적이며 패러다임 구조가 뚜렷하지만, motor 과제들은 서로 간에는 강하게 연결되면서도 대부분의 비-motor 타깃에는 제한적으로만 지원하는 ‘cross-paradigm-limited motor cluster’가 나타났다. multi-source 분석에서는 동일한 타깃이라도 소스 집합의 구성에 따라 held-out reconstruction 거리(전이 성능)가 달라져, many-to-one 관계가 pairwise taskonomy만으로는 전부 설명되지 않음을 확인했다. BIP 기반 할당에서는 개별적으로 가장 강한 소스가 아닐 수 있는데도 0-back/2-back working-memory 상태들이 여러 예산에서 반복적으로 직접 supervision 우선 배정되는 패턴이 관측됐다. 이는 working-memory가 지각·주의·집행 기능의 통합적 구성요소를 재사용하기 좋은 위치일 수 있음을 시사하며, 제한 라벨 하에서 task selection을 직관이 아닌 비용-제약 최적화로 접근할 수 있는 방법론적 의미가 크다.



### A multi-task spatiotemporal deep neural network for predicting penetration depth and morphology in laser welding (https://arxiv.org/abs/2606.26260)
- **Prior Approaches**: 레이저 침투 용접에서 침투 상태와 용접 비드(용접 이음) 형상 평가는 용접 품질을 좌우하지만, 기존 모니터링은 단일 지표 중심이거나 이미지-공정변수 결합이 제한적인 경우가 많다. 특히 시각 정보의 시공간 변화를 충분히 반영하지 못해 일반화가 흔들릴 수 있다는 한계가 지적된다.

- **Core Contribution**: 본 논문은 용접 풀 이미지와 공정 파라미터를 함께 활용해 침투 상태, 침투 깊이, 용접 이음 단면/형상을 동시에 예측하는 multi-task 딥러닝 모델을 제안한다. CNN과 state space model을 결합해 시공간 특징을 더 효율적으로 추출·처리하고, 신뢰도 높은 in-situ 품질 제어 방법론을 제시한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 카메라로 획득되는 상부 용접 풀 이미지의 시공간 패턴을 정확히 학습하는 것과 (2) 다양한 조건에서도 성능을 유지할 수 있는 데이터셋 구성이었다. 이를 위해 시공간 특징을 top weld pool image에서 뽑고 용접 파라미터와 결합하는 구조를 만들었으며, 강건성과 generalization을 높이기 위한 데이터셋 구축 방법도 별도로 제안한다.

- **Empirical Impact**: 테스트 세트 검증에서 침투 상태 예측 정확도는 99.35%까지 도달했고, 침투 깊이 예측 오차는 1.79 mm, 용접 단면 재구성 정확도는 95.65%로 보고된다. 이는 레이저 침투 용접의 실시간 품질 모니터링과 결함 조기 감지에 직접 활용될 수 있는 실증적 성과로, 현장형 in-situ quality control 전략에 새 기준을 제공한다.



### Self-Supervised Tree-level Biomass Estimation in Urban Environments From Airborne LiDAR and Optical Observations (https://arxiv.org/abs/2606.26194)
- **Prior Approaches**: 기존 도시·준도시 지역의 AGB(above-ground biomass) 평가는 주로 수목 인벤토리 샘플링이나 30m/10m급 거친 원격탐사 산출물에 의존해, 개별 수관(crown) 수준의 공간적 해상도와 이질성 반영이 제한적이었다. 일부 딥러닝 융합 연구도 주로 소규모·구조가 비교적 균일한 환경에서 높은 성능을 보고했지만, 수백 km2 규모의 복잡한 도시 경관에서는 동일 수준의 검증이 부족했다.

- **Core Contribution**: 이 논문은 온타리오(캐나다) 810 km2에 대해 2018·2023년 leaf-off 항공 LiDAR(8–10 pulses/m²)와 NirRGB RGB 정사영상(0.16–0.20m)을 이용해 수관 단위 AGB를 매핑하는 프레임워크를 제안한다. 건물·침엽수·활엽수에 대한 크라운 분할과 기능(수종군) 배정을 rule-based pseudo-label과 dual-stream cross-attention 네트워크로 자동화해, 손수 라벨 없이도 크라운 수준 데이터베이스를 만든다는 점이 핵심이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 수작업 라벨이 거의 없는 상황에서 도시의 건물-식생 경계를 학습할 수 있는지, (2) LiDAR와 광학 영상의 지오등록 잔차로 pseudo-label 경계가 어긋날 수 있다는 점, (3) 자동 수관 분할 오차가 AGB 불확실성에 어느 정도 기여하는지를 분리·정량화하는 것이다. 저자들은 영상-센서 misregistration의 실측 오프셋 분포를 바탕으로 NirRGB에 랜덤 translation 증강을 적용하고, 잡음 라벨에 강한 Active Negative Loss(ANL-CE)와 보조 재구성 정규화를 통해 pseudo-label 기반 학습을 안정화했으며, deep ensemble 불확실성 지도를 수관의 allometry 선택까지 연결해 불확실성 원천을 끝까지 추적한다.

- **Empirical Impact**: 독립적으로 주석 처리한 withheld 타일에서 global/mean precision·recall·Dice가 각각 0.86/0.83/0.84로 보고돼, 규칙 기반 pseudo-label에서 시작해도 크라운 분할 학습이 재현 가능함을 보여준다. 인벤토리-세그먼트 매칭 18,713쌍에서 AGB 예측은 inventory 기하 기준 R²=0.609, 운영 조건(자동 분할) 기준 R²=0.570이었고, 남은 최대 불확실성 원천으로 수관 분할이 지목됐다. 또한 30m 집계 시 2018년 1.73 Tg, 2023년 1.81 Tg(811–850 Gg C)의 총량과 나이아가라 절벽 일대 최대 ~140 Mg/ha의 국소 밀도, 5년 순 탄소 증가 39 Gg C를 제시하며 도시 탄소 회계에 바로 활용 가능한 공개 수관 단위 AGB 데이터베이스를 제공한다.



### LCG: Long-Context Consistent Image Generation with Sparse Relational Attention (https://arxiv.org/abs/2606.26171)
- **Prior Approaches**: 기존 text-to-image 확산 모델은 단일 이미지 품질은 크게 향상됐지만, 만화·스토리보드처럼 여러 패널이 이어질 때는 캐릭터 동일성·역할·시각적 연속성이 쉽게 무너진다. 독립 패널 생성은 패널 간 근거를 놓치고, 순차 생성은 초반의 identity drift가 뒤 패널까지 누적되는 문제가 있다. 또한 패널 전부에 대해 dense attention을 쓰면 메모리 사용량이 패널 수에 대해 빠르게 폭증해 실용 한계를 넘는다.

- **Core Contribution**: 이 논문은 Long-Context Generation(LCG)이라는 프레임워크로, 여러 장면의 텍스트 프롬프트를 입력받아 긴 시퀀스의 다중 이미지를 함께 생성하면서 일관성을 유지하는 방법을 제안한다. 각 프롬프트는 parallel generation branch로 나뉘고, 확산 과정에서 브랜치들을 동시에 denoise해 의미·레이아웃 증거를 교환한다. 일관성과 확장성을 동시에 노리기 위해 Sparse Relational Attention(SRA)과 Routing Consistency Constraint(RCC)라는 두 핵심 장치를 함께 도입한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 장거리 cross-branch 정보 교환을 하되 비용을 줄이고, (2) 학습 중 잘못된 라우팅이 발생할 때 캐릭터·역할 표류를 안정적으로 억제하는 것이다. 논문은 SRA로 블록 요약 기반 후보를 고르고 그 안에서만 국소/관계 토큰을 sparse하게 선택해 계산을 트랙터블하게 만든다. 여기에 RCC는 identity-aware mask에서 얻는 same-identity 대응을 토큰 라우팅 목표로 변환해, 예측된 sparse attention 흐름이 의미적으로 정렬되도록 정규화 손실을 추가한다.

- **Empirical Impact**: 실험에서는 Flux.1-dev 백본 위에 LCG를 얹고, 긴 문맥을 위한 합성 데이터셋 Long-Context Consistency Dataset(LCCD)을 600K 학습 시퀀스(각 6~20장) 규모로 구축해 성능을 검증했다. LCG는 VQAScore 기반 prompt alignment, DreamSim 유사도 기반 subject consistency, FaceSim-Arc 기반 얼굴 동일성, Aesthetic Score 기반 화질에서 기존 베이스라인과 비교해 전반적으로 더 높은 점수를 보이며 human preference에서도 우위를 보였다. 또한 SRA는 dense cross-branch attention 대비 VRAM과 지연을 낮추면서 20패널까지 추론이 가능해, 긴 시퀀스 생성에서 실용적 임팩트를 입증했다.



### Predicting Fruit Quality with a Hybrid Machine Learning and Image Processing Approach (https://arxiv.org/abs/2606.26165)
Comments:
          22 pages, 13 figures, 2 tables

- **Prior Approaches**: 기존 연구들은 과일 부패를 image processing 또는 deep learning으로 개별적으로 판단하는 경우가 많아, 정확도-연산비용 사이에서 트레이드오프가 생기곤 했다. 또한 실시간 현장 적용을 위해 고성능 모델을 상시 구동해야 하는 부담이 문제로 지적된다.

- **Core Contribution**: 본 논문은 image processing이 0~100의 부패도 점수(0=신선, 100=완전 부패)를 산출하고, 동시에 CNN이 fresh/rotten 이진 분류를 수행한 뒤 두 결과를 logistic regression으로 결합해 신선도 예측 정확도를 높인다. 더 나아가 logistic regression을 image processing 점수에 바로 적용해, 실시간에서는 CNN 없이도 이진 분류가 가능하도록 구성한 점이 핵심이다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 단순한 비전 전처리 기반 점수가 학습 모델의 판정과 얼마나 일관되게 연결되는지, (2) 두 모듈 결합을 통해 실시간 추론에서 정확도를 유지하는 방법이다. 논문은 이미지의 부패도 퍼센트 출력을 logistic regression으로 재매핑해, 연산이 적은 방식으로 CNN 없이 binary classification을 수행하도록 해결했다.

- **Empirical Impact**: 사과와 오렌지 데이터셋에서 90% 이상의 정확도를 달성하며, 별도 고컴퓨팅 없이 real-time 성능을 보였다는 점이 실용적 의미가 있다. 다만 흰색 또는 transparent 배경에서 과일이 분리되어야 한다는 제한이 있어, 향후 segmentation 모델로 배경 제거를 자동화하면 적용 범위가 크게 넓어질 전망이다.



### DocArena: Turning Raw Documents into Controllable Training Environments for Document Search Agents (https://arxiv.org/abs/2606.26122)
Comments:
          search agent for documents

- **Prior Approaches**: 기존 RL 기반 search agent 연구는 (question, answer, evidence) 튜플에서 학습하며, 알고리즘도 보상모델·탐색 난이도·학습 데이터 시뮬레이션 등을 개선해왔다. 다만 학습 환경은 주로 텍스트 기반이라 표/도표/그림/복잡한 레이아웃 같은 multimodal 문서의 증거 구성을 충분히 통제하기 어렵고, 수작업 보정이 남아 확장성과 재현성에 한계가 있었다. 또한 evidence 정확성, 추론 다양성, 검색 깊이, 도메인 커버리지를 정밀하게 제어하는 “표준화된 데이터 파이프라인”이 부족했다.

- **Core Contribution**: 이 논문은 멀티모달 문서 검색·질문응답을 목표로, 원본 문서에서 사람이 주석하지 않고도 controllable한 학습 환경을 자동 생성하는 DocArena 파이프라인을 제안한다. 파이프라인은 문서를 MLLM 기반 시각 인식으로 구조화·인덱싱하고, 페이지 간 정보 분포를 프로파일링해 evidence-정합 QA 쌍을 만들며, 다층 MLLM 검증으로 노이즈를 제거한다. 그 결과 DocArena-79K(79,623개 QA, 8,336개 문서, 16개 도메인, 49개 언어에 해당하는 범위)를 구축하고, Doc-Search 에이전트는 시각 인식과 정책 모델을 분리해 텍스트 LLM 추론 백본으로 multimodal 검색을 가능하게 한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 정답이 지정된 evidence 페이지만으로만 유도되도록 evidence exclusivity를 보장하고, (2) multi-page 증거와 추론 유형 다양성을 대규모로 스케일하며, (3) multimodal 단서(표·그림·레이아웃)를 검색기와 정합시키는 것이다. 이를 위해 페이지 후보군을 만든 뒤 정보 단위를 개념 키로 분해하고, 분포 폭 w(c)를 이용해 irreplaceable evidence(w=1)와 multi-hop 연결(w=2~3), 범용 맥락(w>3)을 체계적으로 섞어 QA를 생성한다. 생성 후에는 deterministic 필터, MLLM 기반 정답 재생성 검증, leave-one-page-out 필요성 테스트로 품질 게이트를 연쇄 적용해 샘플의 타당성과 불필요 증거를 동시에 정리한다.

- **Empirical Impact**: 통합 평가 프레임워크에서 정책 모델만 바꿔 실험한 결과, DocArena로 학습한 에이전트가 retrieval 정확도와 QA 품질 양쪽에서 최상 성능을 보였고, 멀티모달 시나리오 6개(예: MMLongBench-Doc, VisRBench, SlideVQA)에서 유의미한 개선이 나타났다. 또한 텍스트 기반 QA 벤치마크 7개 중 4개에서 1위를 기록해 multimodal 학습 환경이 text-only 과제 일반화에도 도움이 됨을 보여준다. 행동 분석을 통해서도 학습 환경의 controllability가 실제 탐색 전략과 검색 깊이에 반영됨을 확인했다고 보고한다.



### World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays (https://arxiv.org/abs/2606.27374)
- **Prior Approaches**: 기존 연속학습(continual learning)은 순차 파인튜닝 시 catastrophic forgetting을 줄이기 위해 정규화(예: EWC), 아키텍처 분리, 또는 rehearsal/experience replay를 사용해 왔다. 특히 로봇에서는 이전 작업의 실제 데모를 저장해 재생하는 전략이 가장 강력하지만, 최신 로봇 foundation model/VLA(WAM 포함) 흐름에서는 proprietary 데이터 가정 때문에 이러한 “진짜 replay” 확보가 점점 어려워지고 있다.

- **Core Contribution**: 이 논문은 World Action Models(WAMs)의 생성형 인터페이스를 “메모리”로 활용해, 과거 작업의 실제 데모를 저장하지 않고도 pseudo-replay를 생성하는 Recurrent Generative Replay(REGEN)를 제안한다. REGEN은 이전 작업의 language instruction과 현재-task 관측만으로 WAM을 재귀적으로 질의해 과거 궤적을 합성하고, 이를 현재 작업 데모와 함께 behavioral cloning으로 학습해 망각을 억제한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 긴 horizon에서 재귀 생성되는 future visual observation의 품질이 저하되는 문제와 (2) 생성된 관측과 그 관측을 전제로 하는 action 사이의 불일치로, 실제 성공으로 이어지지 않는 경우가 생긴다는 점이다. 논문은 생성 길이를 goal-reward 기반 종료로 줄이고(초기 종료로 저품질 프레임 누적 방지), REGEN 학습에서 pseudo-trajectory를 현재 데이터와 혼합해 이전 능력의 보존 신호를 최대화하도록 설계했다.

- **Empirical Impact**: 실험은 시뮬레이션(LIBERO)과 실세계(xArm7 매니퓰레이션) 모두에서 수행됐고, REGEN은 순차 파인튜닝 대비 catastrophic forgetting을 최대 50%까지 감소시키며 privileged experience replay에 접근하는 성능을 보였다. 또한 표현 드리프트와 궤적 시각화를 통해 행동/표현을 비교적 잘 유지함을 보였고, 생성 replay의 성능 한계 요인으로 long-horizon visual degradation과 action-observation inconsistency를 실증적으로 확인했다.



### Error-Conditioned Neural Solvers (https://arxiv.org/abs/2606.27354)
- **Prior Approaches**: 신경 연산자와 neural surrogate는 PDE 매개변수에서 해로의 빠른 근사를 제공하지만, 추론 시 해가 만드는 제약 위반을 스스로 점검·수정하지 못해 학습 분포 밖에서 성능이 쉽게 흔들립니다. 이를 보완하려고 test-time에 PDE residual을 넣는 hybrid 방법들이 등장했지만, residual을 최적화 목표로 두거나(gradient descent, projection 등) 고전 해석 기법에 가까운 비용·불안정성을 그대로 물려받는 문제가 남아 있습니다. 더 나아가 ill-conditioned 문제에서는 residual을 낮추는 것이 실제 reconstruction 정확도와 일치하지 않는 “residual-reconstruction gap”이 나타납니다.

- **Core Contribution**: 이 논문은 error-conditioned Neural Solvers(ENS)를 제안하며, 핵심 아이디어는 PDE residual을 “최적화할 타깃”이 아니라 매 반복마다 네트워크가 직접 읽는 입력 신호로 바꾸는 것입니다. ENS는 residual field의 공간적 구조를 관찰해 비선형 업데이트 정책을 학습하고, 매 단계 예측값과 함께 residual을 다시 계산해 반복적으로 정정합니다. 또한 residual을 최소화하지 않고도 reconstruction 정확도를 끌어올리도록 학습을 구성해 기존 hybrid의 신뢰도 문제를 구조적으로 회피합니다.

- **Technical Challenges**: hybrid 방식의 실패는 ill-conditioned 시스템에서 PDE residual 최소화가 해 오차를 보장하지 않는다는 이론적/수치적 불일치에서 비롯됩니다. 논문은 이 gap을 이론적으로 정리해 residual이 낮아도 해가 충분히 정확하지 않을 수 있음을 보이고, Gauss-Newton/뉴턴 계열 보정의 수렴 반경 밖에서는 초기값 민감성도 커진다고 설명합니다. ENS는 Jacobian inversion이나 residual loss 최적화 같은 외부 수치 최적화 없이, residual field를 입력 채널로 주고 학습된 corrector가 오차를 직접 읽어 수정하도록 설계해 이러한 취약점을 줄였습니다.

- **Empirical Impact**: ENS는 4종 PDE 계열(헬름홀츠, Darcy, Poisson, Navier–Stokes)과 in-distribution/초해상도/계수 외삽/크로스-에쿼이션 전이를 아우르는 설정에서 대부분의 경우 최고 수준의 예측 정확도를 보입니다. 특히 난이도가 큰 ill-conditioned 영역(예: turbulent Kolmogorov flow)에서는 최대 10배 가까운 정확도 향상을 보이면서도, gradient descent·projection 기반 hybrid가 요구하는 비싼 test-time 계산은 피합니다. 분포 이동에서도 성능 우위가 가장 크게 나타나며, 초기 residual이 수천만 배(7자릿수)까지 달라도 residual floor로 수렴하는 초기값 견고성까지 함께 보고해 “residual 기반 보정의 신뢰도 한계”를 실증적으로 뒷받침합니다.



### Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning (https://arxiv.org/abs/2606.27330)
Comments:
          Accepted to ACL 2026 Main

- **Prior Approaches**: 기존 멀티모달 웹 에이전트 연구는 반복 GUI 작업에서 과업을 클릭·입력·스크롤 같은 원자 단위로 학습시키거나, 대략적인 high-level 과업(거친 조건 포함) 궤적을 이용해 후학습을 수행해 왔다. 그러나 원자 단위 학습은 고수준 계획으로의 compositional generalization(조합 일반화)이 약하고, coarse high-level 궤적은 환경 정합성·제약 조건이 느슨해 OOD(미지 사이트) 일반화가 제한된다.

- **Core Contribution**: 이 논문은 planning experience exploration and utilization(PEEU)로, 에이전트가 미지 웹사이트를 자율 탐색해 경험을 수집하고 hindsight 경험을 이용해 “엄격히 정렬(aligned)된 고수준 훈련 데이터”를 합성하는 방법을 제안한다. 또한 task decomposition hierarchical analysis framework(TDHAF)로 저수준-중간-고수준 과업 분해의 일반화를 ID(사내)와 OOD(외부)로 나눠 체계적으로 진단한다.

- **Technical Challenges**: 핵심 기술 난제는 탐색 궤적에서 도출된 고수준 과업이 실제 결과와 어긋나거나(미스매치), 미관측 환경 정보 때문에 엄격한 제약을 포함하지 못한다는 점이다. PEEU는 행동 전후의 시각 상태를 비교해 atomic 경험을 추출한 뒤, hindsight로 고수준 과업-궤적 쌍을 더 정밀하게 재정렬하고 제약을 강화하는 2단계(탐색 트리 구성→경험 활용) 파이프라인을 설계한다.

- **Empirical Impact**: 실험은 WebVoyager의 7개 미지 사이트에서 cross-website OOD 일반화를 평가했으며, 동일 데이터 스케일 조건에서 Qwen2.5-VL-7B 기반 PEEU가 30.6% 정확도로 Qwen2.5-VL-32B(22.7%)를 포함해 기존 기준선을 크게 앞섰다. TDHAF 분석 결과, 저수준 원자 스킬 숙달만으로는 고수준 계획 능력이 보장되지 않고, 고수준 과업 훈련이 ID·OOD 모두에서 더 높은 coverage를 만든다는 점을 정량적으로 입증해 작은 모델의 OOD 계획 성능 향상에 핵심이 “정렬된 고수준 hindsight 학습”임을 보여준다.



### Hallucination in World Models is Predictable and Preventab (https://arxiv.org/abs/2606.27326)
Comments:
          Interactive paper, live demo, code, dataset, and models: this https URL

- **Prior Approaches**: 기존 generative world models는 행동 제어 미래를 그럴듯하게 생성하지만, 롤아웃이 실제 동역학에서 벗어나도 “시각적으로는 자연스러운” 환각이 발생해 다운스트림 제어를 오도할 수 있다. 기존 연구는 대개 아키텍처/학습 규모를 키워 환각을 완화하려 하지만, 왜 언제 환각이 생기는지에 대한 데이터 관점의 설명이 부족했다. 또한 환각을 탐지하거나 단계별 원인을 분해하는 데 필요한 대규모·행동 라벨·라이브 시뮬레이터가 함께 제공되는 벤치마크가 제한적이었다.

- **Core Contribution**: 이 논문은 환각이 state-action 공간의 데이터 커버리지(coverage) 부족에서 집중적으로 나온다는 가설을 제시하고, 이를 파이프라인 단계별 실패 모드로 연결해 정리한다. 저자들은 MMBench2(427시간, 210 task)의 대규모 비주얼 world modeling 데이터셋과 350M 파라미터 world model을 통해, 환각을 perceptual(토크나이저), action-marginalized(행동 조건화), scene-diverging(다단 롤아웃)로 분류한다. 동시에 각 모드를 예측하는 라벨-프리 신호 3종을 설계하고, 같은 신호를 이용해 학습/온라인 데이터 수집 모두에서 완화 전략으로 연결한다.

- **Technical Challenges**: 핵심 기술 과제는 “그럴듯하지만 틀린” 롤아웃을 단계별로 진단하고, 추가 라벨 없이 런타임에서 예측 가능한 신호를 만드는 것이다. 저자들은 tokenizer round-trip residual, flow instability, inter-seed variance를 제안하되, 장면 움직임에 의한 공변량(confound)을 줄이기 위해 dynamism-normalized 형태로 정규화해 비교 가능하게 만들었다. 이후 커버리지 갭을 줄이기 위해 학습 시 coverage-aware sampling으로 격차를 메우고, 온라인에서는 환각 예측기를 curiosity reward로 사용해 예측에 취약한 전이를 우선적으로 수집하는 closed-loop finetuning 레시피를 구성한다.

- **Empirical Impact**: MMBench2에서 3가지 환각 예측 신호는 open-loop 롤아웃 품질과 강한 상관(예: Spearman ρ≈0.80)을 보이며, action ignored/scene divergent 같은 이진 환각 라벨에 대해서도 높은 분류 성능을 나타낸다. coverage-aware training은 3종 환각 모드 전반을 동시에 줄이고 롤아웃 충실도를 개선하는 방향으로 작동한다. 특히 보정된 온라인 데이터 수집과 finetuning은 완전히 unseen 환경에 대해 50개 수준의 real environment trajectories로 적응해, 인간 데이터 수집에 근접하는 데이터 효율을 보이며 커버리지 기반 환각 프레임의 유의미성을 실증한다.



### EO-WM: A Physically Informed World Model for Probabilistic Earth Observation Forecasting (https://arxiv.org/abs/2606.27277)
Comments:
          28 pages, 5 figures, 11 tables

- **Prior Approaches**: 기존 Earth Observation(EO) forecasting은 EarthNet2021 포맷처럼 픽셀 복원·NDVI 시계열 일치 같은 reconstruction 중심 평가가 많습니다. 결정론 모델은 미래를 하나로 고정해 불확실성을 표현하기 어렵고, diffusion 계열도 날씨를 ‘통일된 conditioning’으로만 쓰는 경우가 많아 날씨-반응의 물리적 구조를 충분히 분리하지 못합니다.

- **Core Contribution**: 본 논문은 EO forecasting을 ‘부분 관측된, weather-driven world modeling’으로 재정의하고, 예측이 날씨 변화에 대해 올바르게 반응하는지(용인된 weather-response fidelity)를 핵심 목표로 제시합니다. 이에 video diffusion transformer인 EO-WM을 제안하며, meteorological forcing을 climatological baseline, weather anomaly, cumulative physical stress로 분해해 서로 다른 경로로 주입합니다. 또한 단순 정확도 외에 Extreme Summer Benchmark와 Seasonal Matched-Pair Benchmark를 도입해 ‘극한 악화의 심도’와 ‘대체 날씨에서의 반응 방향·크기’를 진단합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 위성 관측이 희소·불완전해 중간 상태와 불확실성이 크다는 점, (2) 관측되지 않는 토양수분 등 잠재 상태 때문에 동일 날씨라도 결과가 달라질 수 있다는 점입니다. EO-WM은 latent diffusion(EO-VAE + MMDiT)로 다중 가능한 미래를 모델링하고, baseline- anomaly 분해와 누적 stress 인덱스를 설계해 ‘순간적 이탈’과 ‘지속된 열·가뭄 스트레스’가 예측에 반영되도록 합니다. 더불어 anomaly-targeted classifier-free guidance를 통해 이상 신호 민감도를 조절할 수 있게 했습니다.

- **Empirical Impact**: 실험에서 EO-WM은 표준 복원 지표에서는 경쟁력을 유지하면서도, Extreme Summer에서 NDVI decline amplitude 오류를 상대 5.63% 줄이고 directional hit rate을 상대 7.80% 향상시킵니다. Seasonal Matched-Pair에서도 paired trajectory가 날씨 조건 간 상대적 순서와 분리를 더 잘 보존해 forcing-response fidelity가 개선됐음을 보여줍니다. 즉, EO forecasting을 ‘그럴듯한 생성’이 아니라 ‘날씨 변화에 대한 물리적으로 일관된 시뮬레이션’으로 평가·진전시키는 실증적 근거를 제공한 셈입니다.



### From Celebrities to Anyone: Characterizing AI Nudification Content, Technology, and Community Dynamics on 4chan (https://arxiv.org/abs/2606.27234)
Comments:
          22 pages, 13 figures, 2 tables

- **Prior Approaches**: 기존 연구는 주로 MrDeepFakes 같은 딥페이크/누디피케이션 마켓플레이스나, Hugging Face·Civitai의 모델 저장소를 중심으로 위험을 관찰했다. 그러나 확산 모델 시대 이후의 “현장(wild)” 콘텐츠가 어떻게 요청-제공 구조로 생성·유통되는지, 특히 익명 커뮤니티 수준의 생태계 역학은 충분히 측정되지 않았다.
또한 과거에는 기술 장벽과 정보 가시성 때문에 타깃이 연예인 중심이었던 경향이 있어, 일반 개인에 대한 피해 확대는 체계적으로 과소평가됐다는 한계가 있었다.

- **Core Contribution**: 이 논문은 4chan Adult Requests 보드를 41일간 대규모로 수집해, 실제로 생성·교환되는 SNEACI의 규모와 양상을 정량화했다. 총 24,105개의 SNEACI를 식별하고, 타깃이 기존 연예인 중심에서 비(非)연예인으로 크게 이동했음을 실증적으로 보여준다.
특히 비연예인 비중이 55.78%(영상은 60.26%)로 관측되어, 피해가 사용자들의 지인/사회적 근접 집단까지 확장됐다는 점을 강조한다.

- **Technical Challenges**: 핵심 기술 과제는 방대한 멀티미디어에서 SNEACI를 신뢰성 있게 선별하는 것이었다. 연구진은 NSFW 필터링→AIGC(완전 합성) 탐지→맞춤 undress 탐지→(연예인 여부) 다중 단계 분류 파이프라인을 구축했고, 영상은 표본 점검 결과를 근거로 처리된 경우 SNEACI로 간주했다.
또한 연예인 분류는 고정된 신원 데이터에 의존하면 오탐이 커질 수 있어, Gemini 기반의 이중 에이전트 검증으로 정확도를 높이는 방식으로 해결했다.

- **Empirical Impact**: 실증 결과, 오픈소스 기반 모델과 공유된 fine-tuned 자원이 제작의 중심에 있으며 Stable Diffusion 계열(이미지 42.7%)과 Wan(동영상 66.5%)이 주도하는 공급망이 드러났다. 동시에 소수의 고활동 “프로바이더(핵심 생산자)”가 요청량과 타깃 구성을 좌우하며 커뮤니티를 자기강화적으로 지속시키는 생태계 역학도 관찰됐다.
이 연구는 SNEACI 확산이 단일 플랫폼의 문제가 아니라 다중 플랫폼 공급망과 툴 재배포에 의해 유지된다는 점을 보여, 플랫폼 거버넌스·기술적 안전장치·피해자 보호 개입이 시급함을 강하게 시사한다.



### Proposal-Conditioned Latent Diffusion for Closed-Loop Traffic Scenario Generation (https://arxiv.org/abs/2606.27123)
Comments:
          Accepted for publication at the IEEE International Conference on Intelligent Transportation Systems (ITSC), 2026

- **Prior Approaches**: 기존 확산(diffusion) 기반 시나리오 생성은 현실감은 높일 수 있지만, 안전과 같은 안전-critical 편집(제어)을 롤아웃 전반에서 안정적으로 다루기 어렵다는 한계가 있다. 또 적대적(adversarial) 생성은 충돌/근접회피 같은 위험 사건을 잘 만들 수 있으나, 도로 이탈·운동학적 제약 위반처럼 그럴듯함(plausibility)이 떨어질 수 있다. 반면, 지도 일관성과 상호작용의 closed-loop 일관성을 지키면서도 재계획(replanning) 루프에서 빠르게 샘플링하는 효율성은 여전히 병목이다.

- **Core Contribution**: 이 논문은 instance-centric 장면 컨텍스트와 멀티모달 per-agent 제안 분포를 조건으로 하는 diffusion 기반 scenario generation 프레임워크를 제안한다. 핵심은 단순히 에이전트를 독립적으로 미래를 합성하는 게 아니라, joint interaction을 명시적으로 모델링해 장면-일관성과 controllability를 동시에 노리는 점이다. 또한 test-time guidance를 선택적으로 적용해 안전-critical 행동을 원하는 목표로 “수정”할 수 있게 하되 재학습 없이 운용한다.

- **Technical Challenges**: 문제의 기술적 난제는 (1) 다중 에이전트의 reverse sampling이 많은 순차 단계로 인해 closed-loop 재계획 지연을 유발한다는 점, (2) 다중 에이전트에 대한 guidance가 계산량을 늘리고 운동학적으로 불일치한 궤적을 만들 수 있다는 점이다. 저자들은 제안(proposal)에서 출발하는 shifted Gaussian 초기화와 PCA로 만든 compact action-latent 표현을 통해, DDIM 등 few-step reverse diffusion가 가능하도록 시작 분포와 추론 경로를 바꾸었다. 여기에 differentiable map 위반·충돌 근접도·게임이론 기반(추격-회피) 목적을 latent 공간 guidance로 결합해 realism–stress-testing trade-off를 test-time에서 조절한다.

- **Empirical Impact**: Waymo Open Motion Dataset(WOMD)에서 모델은 다양한 상호작용 시나리오에 대해 현실감, 안전, controllability의 균형이 유리하게 나타났고, test-time guidance는 competing objectives 사이의 체계적 트레이드오프를 가능하게 했다. 무가이드(unguided) 대비 충돌 및 오프로드(off-road) 같은 안전 지표가 유의미하게 개선되면서도, 현실감과 위치 정확도는 크게 훼손되지 않았다. 또한 ablation과 단계 수(step count) 실험에서 proposal-informed 초기화와 few-step 샘플링이 런타임을 줄이면서도 품질을 유지하는 경향이 확인되어, 실제 AV 시뮬레이션/검증 파이프라인에 실용적 의미가 크다.



### Just how sure are you? Improving Verbalized Uncertainty Calibration in Medical VQA (https://arxiv.org/abs/2606.27023)
- **Prior Approaches**: 기존 LLM 기반 confidence 보정은 주로 텍스트-only에서 토큰 확률, verbalized confidence(자연어로 확신 표현), consistency/perturbation 등을 통해 신뢰도를 조정했지만, 의료 이미지 이해의 멀티모달 특성을 반영하지 못했습니다. 특히 verbalized confidence는 추가 학습 없으면 과신(overconfidence)되기 쉬워 임상 신뢰성에 한계가 컸습니다. 또 멀티모달에서의 보정은 대체로 prompt 엔지니어링 중심이거나 다른 비전 태스크(detector) 위주의 미세조정에 머물렀습니다.

- **Core Contribution**: 이 논문은 Medical VQA에서 MLLM이 ‘정답을 맞힐 확률에 비례해’ verbalized confidence를 내도록, training-based 보정 프레임워크를 제안합니다. 핵심은 2×2 factorial perturbation(이미지 유무/텍스트 무결성)로 시각 근거와 언어 priors의 의존도를 드러내고, 그 차이를 confidence 학습에 직접 연결하는 데 있습니다. 또한 Brier-style calibration loss, anchor regularizer, contrastive alignment, 그리고 top-k KL divergence regularizer를 결합해 보정 성능과 답변 능력 유지(및 포맷 준수)를 동시에 노립니다.

- **Technical Challenges**: 가장 큰 난제는 멀티모달 의료 문맥에서 “시각 근거가 사라질 때 confidence가 얼마나 내려가야 하는가”를 학습 신호로 설계하는 것이었습니다. 이를 위해 이미지: 원본 vs black image, 텍스트: 원본 vs 옵션 섞기/교란을 교차시켜 조건별 정답률(분수 정확도) 추정치를 만들고, confidence 토큰 분포에 Brier 기반 loss와 evidence-aware alignment를 적용했습니다. 또 Brier 최적화만으로는 극단값으로 confidence가 무너지는 collapse를 막기 위해 anchor loss를 추가하고, fine-tuning 중 답변 분포가 망가지지 않도록 answering token 위치에 top-k KL regularizer를 걸어 안정화를 달성했습니다.

- **Empirical Impact**: 3개 Medical VQA 벤치마크(OmniMedVQA/PMC-VQA/MedXpertQA)와 2개 아키텍처(MedGemma 4B IT, Qwen2 VL 7B Instruct)에서 ECE와 Brier Score, AUROC를 동시에 개선했으며, calibration error는 60% 이상 줄고 discrimination도 26% 이상 향상됩니다. 특히 difficulty 레짐이 다른 세 환경에서 ECE 분포 폭이 가장 좁게 유지되어 특정 데이터셋 평균 정확도에 의존한 편향 가능성을 낮췄습니다. 다만 MedXpertQA에서는 AUROC가 거의 chance 수준으로 남아(멀티스텝 임상 추론 난이도) verbalized confidence의 한계가 확인됐고, ablation으로 각 loss 성분(특히 alignment가 discrimination을, KL이 정확도/포맷 유지)을 분리 검증했습니다.



### Einstein World Models (https://arxiv.org/abs/2606.26969)
Comments:
          12 pages (9 without references), 2 figures, 1 algorithm

- **Prior Approaches**: 기존 Chain-of-thought는 중간 추론을 언어로만 외부화해, 물체 식별·접촉·열·운동처럼 장면 수준 변수가 중요한 문제에서 한계가 나타난다. 또한 world model 계열은 주로 관측 기반 예측이나 행동 조건부 예측에 머물러, 경험 밖의 반사실적 thought experiment를 정밀하게 다루기 어렵다. 비슷하게 Visualization-of-Thought·whiteboard-of-thought 류는 시각 메모/그림을 중간 산출물로 쓰지만, 이를 비디오 롤아웃 형태로 ‘가설을 검토’하는 통합 메커니즘은 상대적으로 약하다.

- **Core Contribution**: Einstein World Models(EWMs)는 LLM이 스스로 판단해 world-module로부터 짧은 visual-temporal rollout(비디오 롤아웃)을 생성·조회하고, 그 결과를 추론 trace에 ‘검증 가능한 가설’로 포함하도록 하는 구조를 제안한다. 핵심은 롤아웃을 정답 대신 참고할 inspectable hypothesis로 취급해, 텍스트만으로 어려운 장면 전개·반사실적 사건 상상을 추론에 보조하는 것이다. 또한 “reasoner(추론자)”와 “world-module(롤아웃 생성기)”을 분리해, 생성된 비디오를 외부에서 관찰·디버깅할 수 있게 만든다.

- **Technical Challenges**: 어떤 질문에서 언제 비디오 thought experiment가 이득인지, 그리고 어떤 world-module query로 롤아웃을 뽑아야 이후 추론이 강화되는지를 학습해야 한다. 저자들은 SFT로 EWM trace 포맷(언어·tool_call·visual_rollout·정답)을 익힌 뒤, verifiable final answer에 기반한 RLVR/GRPO 스타일 학습으로 world-module 호출의 선택성을 최적화한다. 더해 world-module 자체의 신뢰도 문제를 “물리적 그럴듯함”으로 직접 보정하기보다, 롤아웃의 inspectability/faithfulness가 추론에 실제로 기여하는지와 모듈 선택(예: diffusion 기반 렌더러 품질, 상이한 편향의 앙상블)을 함께 다루는 방향을 제시한다.

- **Empirical Impact**: 이 논문은 실험 결과가 아니라 먼저 학습 가능한 시각 thought experimentation의 데이터·학습 청사진을 제시하는 데 초점을 둔다. 특히 기존 공개 데이터가 ‘문제의 시작이 텍스트이며 모델이 스스로 시각화 여부를 결정해야 하는’ 설정을 충분히 제공하지 못한다고 진단하며, SimpleBench 같은 제한적 예시를 보완할 datasets 공모를 제안한다. EWMs가 성공하면 LLM이 단순 텍스트 기호를 넘어 외부화된 visual walk-through로 추론을 확장하고, 중간 가설을 공유·점검하는 디버깅 경로를 열 수 있다는 점에서 영향력이 크다는 입장이다.



### Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds (https://arxiv.org/abs/2606.26964)
Comments:
          25 pages, 17 figures

- **Prior Approaches**: 기존 카메라 제어·궤적 계획은 충돌 회피와 부드러운 경로 같은 물리/기하 제약을 잘 다루지만, 관측 목표가 ‘이미 정해진’ 가정이 많았다. 또한 3D cinematography나 generative video 계열은 주로 trajectory나 비디오 합성 자체에 초점을 두어, 내러티브 의도로부터 무엇을 봐야 하는지를 물리 실행 가능한 형태로 조직하는 문제는 상대적으로 비어 있었다. 결과적으로 동적 3D 환경에서 narrative intent와 가시성·충돌·시간적 일관성을 함께 만족시키는 폐루프형 관측 설계가 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 카메라가 단순 센서가 아니라 ‘내러티브에 근거해 무엇을 관측할지’를 결정하는 Narrative-Grounded World Visual Attention을 카메라 플래닝 관점에서 정식화한다. 이를 위해 Look-Before-Move 프레임워크를 제안하며, 관측 명세를 먼저 세우고(관측 contract) 그다음에 viewpoint 탐색과 trajectory grounding을 수행하도록 단계 분해한다. 또한 동적 3D Story World Benchmark를 구축해 subject perception, intent consistency, trajectory quality를 동시에 평가할 수 있게 했다.

- **Technical Challenges**: 핵심 기술 난제는 내러티브 지시문을 실행 가능한 시각 제약으로 변환하고, 그 제약을 만족하면서도 3D 물리 제약(가림, 충돌, 비현실적 카메라 배치) 하에서 viewpoint를 찾는 것이다. 논문은 Semantic Observation Contract로 directorial intent를 대상/관계/가시조건/구도 선호 같은 구체 제약으로 바꾸고, Monte Carlo Viewpoint Search로 narrative-compliant이면서 geometrically feasible한 후보를 대량 탐색한 뒤 tournament reranking과 render 기반 reflection으로 미세 조정한다. 마지막으로 Semantic Trajectory Grounding에서 선택된 viewpoint들을 동적 배우를 추적하는 연속 궤적으로 연결하고, closed-loop 검증과 temporal semantic editing으로 시간적 의미 일관성을 맞춘다.

- **Empirical Impact**: 실험은 Look-Before-Move가 대표 baseline 대비 전체 점수와 세 평가 축(SP, IC, TQ)에서 일관된 개선을 보였다고 보고한다. 정성 결과에서도 암살자 장면처럼 의도된 주체를 정확히 클로즈업으로 배치하고 액션 cue를 보존하는 식의 ‘내러티브 특화 관측’이 나타나며, 차량/병원/침실 장면에서도 상호작용과 서사 초점이 안정적으로 유지된다. 전반적으로 이 연구는 dynamic 3D 세계의 카메라 플래닝이 trajectory 생성에 바로 뛰어들기보다 ‘관측을 먼저 조직’하는 접근이 성능과 실행 가능성을 좌우한다는 점을 실증적으로 강화한다.



### Neural Texture Compression using Hypernetworks (https://arxiv.org/abs/2606.26913)
Comments:
          8 pages, 12 figures, conference

- **Prior Approaches**: 기존 신경 텍스처 압축 연구는 물질별로 작은 latent texture와 MLP decoder를 학습한 뒤, 셰이딩 시 실시간으로 복원해 물리 기반 셰이딩 모델의 입력을 재현하는 방식을 보였다. 하지만 각 물질마다 특정 MLP와 latent configuration에 대해 gradient-descent 최적화를 별도로 수행해야 해서, 처리 시간이 길고 재학습 비용이 커지는 한계가 있었다.

- **Core Contribution**: 이 논문은 물질별 latent features와 MLP의 weights·bias를 동시에 생성하는 단일 hypernetwork를 학습해, per-material별 최적화(gradient-descent)를 줄이거나 없애려는 접근을 제안한다. 또한 한 번에 여러 decoders를 추론하거나, super-resolution을 학습하는 decoders까지 확장해 압축-복원 파이프라인의 표현력을 넓힌다.

- **Technical Challenges**: 핵심 난제는 고차원인 생성 공간에서 hypernetwork가 적절한 latent와 MLP 파라미터 조합을 찾아내야 한다는 점이다. 저자들은 hypernetwork가 latent와 decoder 파라미터를 함께 출력하도록 학습을 구성해, 제한된 모델 크기에서도 기존 방식과 유사한 복원 품질을 달성하도록 최적화한다.

- **Empirical Impact**: 실험 결과, 제안한 hypernetwork 기반 방식은 품질 면에서 기존 대표 neural texture compressors와 비교 가능한 성능을 보였다고 보고한다. 더 나아가 다중 decoder 추론과 super-resolution 생성까지 포함해, 실시간 셰이딩용 압축 기술의 실용적 확장 가능성을 보여준다는 점에서 의미가 있다.



### Appearance-Preserving Refinement of Generated 3D Assets for Monochromatic Fabrication (https://arxiv.org/abs/2606.26850)
Comments:
          For preprint

- **Prior Approaches**: 기존 3D 생성 모델(InstantMesh, Trellis, Hi3DGen 등)은 텍스처나 view-dependent 효과에 외관의 정밀도를 크게 의존해 왔습니다. 그 결과 단일 재료·단색(monochromatic) 제작에서는 색/텍스처 정보가 사라지며, 원래 형상을 잘 유지해도 인지되는 디테일이 소실될 수 있습니다. 일부 방법은 RGB 렌더링 기반 디테일 학습으로 기하에 고주파 변형을 유도하지만, 동시에 얇은 구조나 날카로운 특징을 만들어 제작 중 응력 집중과 변형 위험을 키울 수 있습니다.

- **Core Contribution**: GenMF는 단색 제작을 목표로, 텍스처 의존 외관 단서를 기하가 유발하는 shading 효과로 변환하도록 ‘appearance-oriented geometry refinement’를 제안합니다. 단순히 외관을 최대화하는 것이 아니라, 외관 보존과 제작 강건성(robustness) 사이의 균형을 최적화 문제로 명시적으로 다룹니다. 또한 기존 3D 생성 파이프라인에 붙여 쓸 수 있는 plug-and-play 후처리 모듈 형태로 제시되어 모델 불변성을 강조합니다.

- **Technical Challenges**: 핵심 난제는 외관 복원을 위해 필요한 국소·고주파 기하 변형이 응력 집중과 고장 위험을 동시에 유발한다는 점입니다. GenMF는 표준화된 thermo-mechanical 시뮬레이션의 thermal stress를 differentiable stress-aware regularization 프록시로 사용해, FEM 없이도 최적화 과정에서 응력 관련 그래디언트를 기하로 되돌려 줍니다. 아울러 미분 가능 monochromatic 렌더링(노멀 기반 diffuse shading, grayscale albedo 정규화) 손실로 텍스처 대신 음영/그레이 스케일 구조를 맞추고, 표면 매끈함·SDF 및 법선/깊이 정합 같은 기하 정규화로 최적화 아티팩트를 억제합니다.

- **Empirical Impact**: 실험에서는 단색 렌더링 기준 시각적 유사도(LPIPS, SSIM 등)가 유의미하게 개선되며, FEM 기반 응력 지표(최대/평균 응력, 응력 분포 관련 오차 및 Top-5% Stress Region IoU)에서도 응력 집중이 완화되는 경향을 보입니다. 또한 실제 3D 프린팅(단일 재료 PLA, 특정 FDM 설정) 결과에서 세부가 더 잘 보존되면서도 얇고 취약한 구조로 인한 제작 실패 위험을 줄이는 방향의 결과를 제시합니다. 종합하면 GenMF는 ‘생성된 3D 자산’과 ‘제작 가능한 단색 오브젝트’ 사이의 간극을 응력까지 고려한 외관-기하 변환으로 메우는 접근이라는 점에서 의미가 큽니다.



### Ordinal Neural Collapse as a Representation Prior for Visual Navigation (https://arxiv.org/abs/2606.26839)
Comments:
          27 pages, 14 figures. Supplementary material included

- **Prior Approaches**: 비전 관측만으로 로봇 내비게이션 정책을 end-to-end로 모방학습하면, 비전 인코더가 행동과 무관한 시각적 단서(잡음 같은 특징)에 과적합하기 쉽습니다. 특히 단일 action loss가 인코더에 간접 감독만 제공해, 표현이 action-agnostic해지며 애매한 판단 지점에서 일관되지 않은 행동을 내며 실패로 이어질 수 있습니다. 기존 연구는 staged training이나 diffusion action decoder로 성능을 끌어올렸지만, 인코더 표현 공간이 ‘제어에 맞는 기하’를 갖도록 강제하는 방법은 제한적이었습니다.

- **Core Contribution**: ORION(Ordinal Neural Collapse for Visual Navigation)은 목표-상대 방향(예: Far Left~Far Right)의 순서 구조를 반영해, 인코더 표현 공간을 ordinal한 축으로 정렬하는 사전학습을 제안합니다. 단순히 분류 라벨을 주입하는 것이 아니라, 클래스 평균이 단일 판별 축 위에 ‘서열대로’ 배치되도록 유도하고 클래스 내부 분산은 축에 수직 방향으로 억제합니다. 이후 pretrained 인코더를 diffusion 기반 NoMaD 프레임워크에 넣고 end-to-end로 fine-tuning해 전체 파이프라인이 안정적으로 작동하도록 합니다.

- **Technical Challenges**: 핵심 기술적 난제는(1) 연속 행동을 시각 관측에 맞는 ‘순서형 클래스’로 정의하고(2) 그 순서형 기하를 표현 학습에 직접 반영하며(3) downstream fine-tuning에서 구조가 깨지지 않게 하는 것입니다. ORION은 목표 방향 각도를 기준으로 ordered class를 만들고, CDNV(클래스 분리 대비 내부 분산) + supervised ordinal projection(라벨 순서에 따른 1D 축 정렬) + orthogonal compactness(축 바깥 분산 억제)로 전처리 단계의 표현 기하를 형성합니다. 또한 context 수준이 아니라 encoder feature에 제약을 걸고, mini-batch 통계 노이즈를 Welford 누적 통계 및 축 EMA로 완화해 학습 안정성을 확보합니다.

- **Empirical Impact**: 시뮬레이션과 실세계 실험에서 ORION은 end-to-end 및 neural collapse 기반 NC-ETF, 그리고 ViNT/NoMaD 대비 내비게이션 success rate와 goal progress를 일관되게 개선했습니다. 특히 복잡한 다중 교차로 같은 시각적으로 애매한 상황에서 success rate가 NoMaD 대비 최대 +26%p까지 향상됐고, heading jerk는 최대 71%까지 감소했습니다. ablation 결과도 CE-only 방향 분류 사전학습은 오히려 성능을 해치며(표현 구조 불일치), ordinal-collapse형 사전학습과 2-stage 학습 분리가 성능의 핵심임을 보여줍니다.



### Improving Vision-Language-Action Model Fine-Tuning with Structured Stage and Keyframe Supervision (https://arxiv.org/abs/2606.26801)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 파인튜닝은 전체 궤적의 모든 시점에 연속 행동 손실을 동일하게 적용해 왔습니다. 하지만 로봇 조작은 자유공간 이동과 접촉 스킬처럼 구간이 나뉘며, 그 경계(특히 gripper open/close 전환)에서 실패가 집중되는 경향이 있습니다. 그럼에도 현재 방법들은 이 ‘조작 단계(stage)’와 ‘다음 전환 목표(next gripper-event target)’를 구조적으로 감독하지 못합니다.

- **Core Contribution**: StaKe는 이런 공백을 메우기 위해 제안된 플러그인형 보조감독(plug-in auxiliary supervision) 프레임워크입니다. 데모의 gripper 상태만으로 자동 라벨을 만들고, (1) 현재가 motion/skill 중 어디인지 분류하는 stage classifier와 (2) 다음 gripper-transition 키프레임의 관절 목표를 예측하는 keyframe predictor를 학습에 보탭니다. 중요한 점은 보조 헤드는 학습 중에만 쓰고, base VLA 정책의 아키텍처와 inference loop는 그대로 유지한다는 것입니다.

- **Technical Challenges**: 기여를 실사용 수준으로 만들기 위한 핵심 기술 과제는 ‘수동 라벨 없이’ 구조적 감독 신호를 안정적으로 추출하고 학습에 통합하는 것입니다. StaKe는 gripper open/close 전환을 이벤트 경계로 파싱한 뒤, 경계 전후로 stage 라벨을 확장(margin)해 motion/skill을 자동 생성하고, 각 시점이 ‘가장 가까운 다음 전환 프레임’의 joint 상태를 향하도록 키프레임 목표를 할당합니다. 또한 learnable query token과 lightweight auxiliary head(MLP)로 추가 계산 부담을 최소화하면서, 보조 손실의 가중치와 키프레임 타깃 정규화를 통해 학습 균형을 맞췄습니다.

- **Empirical Impact**: 실험에서 StaKe는 RoboTwin 2.0 이중팔 시뮬레이션과 Franka 단일팔 실로봇 과제 모두에서 일관되게 성공률을 끌어올렸습니다(시뮬 14%대, 실로봇 56%대 상대 개선). 특히 gripper-event 전환이 많은 장수행(long-horizon)에서 개선 폭이 더 컸고, ablation 결과 stage supervision과 keyframe supervision을 함께 써야 전체 성능 향상이 재현됨이 확인됐습니다. 정성 분석에서는 학습된 stage 예측이 실제 조작 구간 경계를 잘 따라가고, keyframe head의 다음 전환 타깃 예측 오차도 롤아웃 전반에서 안정적으로 유지되어 성능 향상의 원인을 뒷받침했습니다.



### ResilPhase: Plug-and-Play Phase Mapping and Noise-Resilient Macro-Trajectory Extrapolation for Diffusion Acceleration (https://arxiv.org/abs/2606.26769)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 확산 Transformer(DiT) 가속은 대부분 step 수를 줄이기 위해 “cache-then-forecast” 방식에 의존해 왔습니다. 하지만 레이어(블록) 단위 특징을 다항으로 외삽하는 접근은 고가속 비율에서 품질이 급격히 무너지는 문제가 반복됐습니다.
기존 방법들은 미분 기반 근사(Taylor/Hermite 등)에 크게 의존하면서, 연속 궤적의 고차 시간 미분이 노이즈에 취약하다는 점과 숫자적 불안정(Runge’s phenomenon)을 충분히 통제하지 못했습니다.

- **Core Contribution**: 논문은 품질 저하의 근본 원인을 “연속 확산 경로와 어긋난 표현에 대해 불안정한 이산 외삽을 수행”하기 때문이라고 분석합니다. 이를 바탕으로 ResilPhase는 외삽 대상을 레이어별 중간 특징이 아니라 ODE 공간의 end-to-end 상태 변화인 Global Drift(GD)로 재정의합니다.
또한 GD에 맞춘 derivative-free barycentric Lagrange 외삽과, 오류가 발산하기 쉬운 외삽 구간을 안정화하는 bounded Phase Mapping까지 묶어 하나의 노이즈-복원형 가속 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 레이어 단위 외삽에서 발생하는 공간적 오차 누적, (2) 미분 기반 외삽이 고차 미분 노이즈를 증폭하는 시간적 불안정, (3) 균일 이산 시간에서의 다항 외삽이 Runge’s phenomenon을 유발하는 수치적 불안정입니다.
ResilPhase는 GD 예측으로 레이어 오차 연쇄를 차단하고, barycentric Lagrange를 통해 유도(derivative) 추정 없이 외삽을 수행하며, Chebyshev 또는 Balanced Mapping으로 외삽 도메인을 bounded phase domain으로 투영해 발진성 오차 성장을 억제합니다.

- **Empirical Impact**: FLUX.1-dev와 HunyuanVideo에서 ResilPhase는 공격적인 가속에서도 충실도(fidelity)를 크게 유지하며, 속도-품질 균형이 경쟁 방법 대비 일관되게 향상됐다고 보고합니다. 예를 들어 FLUX.1-dev에서 약 4.97× 가속에도 ImageReward를 1.0258로 유지하며, TaylorSeer 대비 상당한 개선과 더 낮은 LPIPS를 보였습니다.
또한 HunyuanVideo에서는 약 5× 가속에서 VBench 점수 우위와 LPIPS 감소가 확인됐고, DiT-XL/2 조건에서도 FID가 낮게 유지되며 다수 캐싱 베이스라인이 붕괴하는 구간에서 강건함을 입증했습니다.



### PressMimic: Pressure-Guided Motion Capture and Control for Humanoid Robot Imitation (https://arxiv.org/abs/2606.26741)
- **Prior Approaches**: 기존 휴머노이드 모션 모방은 주로 RGB 기반 모션 캡처/관절(기구학) 레퍼런스를 로봇에 그대로 맞추는 방식에 의존해 왔습니다. 이 과정에서 사람-바닥의 접촉 역학(압력/지면반력 등)이 충분히 반영되지 않아 발이 미끄러지거나 바닥을 관통하는 등의 실패가 자주 발생합니다.

- **Core Contribution**: 본 논문은 pressure(압력)를 인식과 제어 전 과정의 통합 모달리티로 삼아, 사람 시연의 “물리적 근거(physical grounding)”를 모방에 직접 연결하는 PressMimic을 제안합니다. 인식 단계에서는 FRAPPE++로 RGB와 압력을 함께 융합해 3D 자세 및 전역 모션을 추정하고, 제어 단계에서는 PSP(pressure-supervised policy)가 압력에서 유도한 접촉 신호를 강화학습 보상에 포함해 물리적으로 일관된 접촉 패턴을 유도합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 희소한 압력 지도를 효율적으로 표현해 비전의 깊이/가림 문제를 보완하는 것과 (2) 학습 보상에서 압력이 접촉 역학을 안정적으로 감독하도록 만드는 것입니다. 논문은 압력의 희소성을 SPE(Sparse Pressure Encoder)로 처리하고, FRAPPE++에서 시간 문맥(TCAM)과 크로스 어텐션(FCAM)으로 RGB-압력 상호작용을 강화했으며, PSP에서는 발의 plantar pressure를 pressure offset 오프셋으로 정의해 auxiliary reward로 넣고 2단계 pressure curriculum으로 학습 안정성을 높였습니다.

- **Empirical Impact**: MotionPRO라는 대규모 동기화 데이터셋(70명, 400종 동작, 12.4M 프레임)으로 실험을 진행했으며, 압력을 사용했을 때 모션 추정 정확도·궤적 일관성·실행 안정성이 함께 개선된다고 보고합니다. 특히 극단적 가림과 전역 이동 드리프트 상황에서도 압력 기반 물리 단서가 추정의 물리적 타당성을 유지하도록 돕고, 모션 모방에서도 task success와 locomotion stability가 일관되게 향상되는 것으로 나타났습니다.



### A Latent ODE Approach to Spatiotemporal Modeling of Cine Cardiac MRI (https://arxiv.org/abs/2606.26718)
- **Prior Approaches**: 기존 CMR 위험모델은 LVEF, strain 같은 소수의 이미지 지표와 임상 변수를 결합해 위험을 추정하지만, 해부학적 고차원 정보를 요약하고 심장주기의 특정 국면(주로 end-diastole/end-systole)에 집중하는 한계가 있습니다. 또한 spatiotemporal 생성·예측 모델들은 시간을 대개 이산 프레임 시퀀스로 다뤄, 심박수에 따른 프레임 수·간격 변화와 수축/이완의 비율 차이를 충분히 반영하기 어렵습니다.

- **Core Contribution**: 이 논문은 양심실(biventricular) 해부학과 심장주기 전체(cine) 운동을 연속적인 잠재 궤적(latent trajectory)으로 인코딩하는 latent dynamical model을 제안합니다. covariate-conditioned prior로 정상적인 end-diastolic latent 상태를 정의한 뒤, posterior와의 편차를 표준화해 Cox proportional hazards 모델의 residual 위험 점수로 연결합니다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 심박수에 따라 달라지는 생리적 위상(phase) 타이밍을 모델 학습·생성에 일관되게 반영하고, (2) 메시(mesh) 기반 운동을 해부학적으로 일관되며 자연스러운 연속 시간 생성으로 학습하는 것입니다. 저자들은 graph-based mesh autoencoder로 3D+t 운동을 복원하면서, heart-rate-aware neural ODE와 learnable monotone phase reparameterization으로 시간 왜곡 문제를 완화하고 latent 잔차 임베딩을 survival 분석에 투입합니다.

- **Empirical Impact**: UK Biobank 72,386명(incident heart failure 367명)에서 평가된 held-out subset 기준, latent score를 pooled cohort equations에 추가했을 때 stratified C-index가 0.704→0.785로 상승했으며, 7개 기존 심장 마커 대비 우수한 성능(기존 0.764)을 보였습니다. 또한 GNN+ODE 조합이 non-graph, non-ODE 대비 reconstruction fidelity·생성 현실감·예후 성능 사이의 균형이 가장 좋았지만, 임상 적용 전 더 대표적인 환자 코호트에서 외부 검증이 필요하다고 밝힙니다.



### Dual-Prior Guided Null-Space Learning with Mixture-of-Splines for Arbitrary Medical Slice Super-Resolution (https://arxiv.org/abs/2606.26716)
Comments:
          Accepted to ECCV 2026! Project page: this https URL

- **Prior Approaches**: 임의 슬라이스 초해상(super-resolution)은 등방성이 아닌 임상 획득으로부터 중간 슬라이스를 합성해 다운스트림 품질을 높이는 데 필수로 자리잡았습니다. 기존 선형/큐빅 보간은 고주파 해부학 디테일을 복원하지 못하고, INR 기반 방법은 좌표별 회귀 특성 때문에 해부학적으로 그럴듯하지 않은 구조를 만들어내거나 관측된 슬라이스 값을 미세하게 바꿀 위험이 있습니다. 또 많은 의료 ArSSR 계열이 데이터-피델리티 같은 하드 제약이 약해 임상 적용에 필요한 보장을 제공하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 Dual-Prior Null-space Learning(DP-NSL)로 문제를 “무제약 잔차 회귀”가 아니라 “이중-우선 제약 복원”으로 재정의합니다. Measurement-Consistent Projection(MCP)을 통해 관측된 모든 슬라이스는 0오차로 재현되도록 하여, 학습이 생성할 수 있는 내용이 관측에 불가능한 null space로만 제한되게 만듭니다. null space 내부에서는 Mixture-of-Splines(MoS)와 Local Spatial Consistency Decoder(LSCD)로 기하학적 연속성과 국소 공간 일관성을 동시에 강화합니다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 네트워크가 생성하는 세부가 관측된 슬라이스를 절대 훼손하지 않도록 하면서, (2) 관측되지 않은 영역에서는 해부학적 연속성과 경계 선명도를 자연스럽게 복원하는 것입니다. DP-NSL은 null-range-space decomposition을 바탕으로 MCP를 고정 투영 연산으로 구성해, 모델이 어떤 출력을 하더라도 전방 측정 연산 하에서 관측 일치가 보장되도록 설계합니다. 이후 MoS는 B-spline 전문가의 차수(order)를 위치별로 동적으로 혼합해 영역마다 필요한 연속성 수준을 학습하고, LSCD는 3D 컨볼루션 기반 국소 컨텍스트 집계로 마이크로 수준의 불일치를 줄입니다.

- **Empirical Impact**: CT 3종과 MRI 1종 벤치마크에서 DP-NSL은 기존 방법을 능가하면서도 measurement consistency를 엄격히 보존하는 것으로 보고됩니다. 예를 들어 Colon 데이터에서 ×2, ×3에서 각각 최대 0.71 dB, 0.65 dB 향상을 보였고 Liver에서는 ×2에서 1.07 dB 개선이 관찰됩니다. 또한 학습에 없던 upsampling factor(×5~×7)에서도 성능 우위를 유지해, dual-prior 제약 기반 접근이 의료 ArSSR의 일반화에 실질적 의미를 갖는다는 점을 보여줍니다.



### MLFFM-SegDiff: A Multi-Level Feature Fusion Diffusion Model for Skin Lesion Segmentation (https://arxiv.org/abs/2606.26712)
- **Prior Approaches**: 피부 병변 분할은 컴퓨터 보조 진단에서 핵심 단계지만, 경계가 흐리고 대비가 낮으며 모발·그림자 같은 아티팩트와 환자·장비·조명에 따른 큰 변동성이 문제로 지적돼 왔다. 기존 U-Net 계열은 skip connection으로 국소 디테일과 의미를 결합하지만, 같은 레벨 간 전달에 머물러 크로스-레벨 상호작용과 다중 스케일 융합이 제한되기 쉽다. 확산 모델(DermoSegDiff 등)은 노이즈 제거를 반복하며 경계를 다루는 데 강점이 있으나, 레벨 간 feature 상호작용과 경계 디테일 복원이 충분치 않다는 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 피부 병변 분할을 위한 멀티레벨 feature fusion diffusion 모델 MLFFM-SegDiff를 제안한다. 핵심은 (1) dual-path U-Net encoder로 노이즈 마스크 특징과 피부영상 특징의 상호작용을 강화하고, (2) MLFFM(Multi-Level Feature Fusion Module)로 encoder-Decoder 사이 skip 특징을 주의(attention)·스케일 정렬·적응형 크로스레벨 융합까지 포함해 결합하는 것이다. 여기에 (3) 필요 시 경계에 가중치를 더 주는 boundary-sensitive loss를 설정 가능하게 도입해 애매한 윤곽 학습을 보조한다.

- **Technical Challenges**: 확산 기반 분할에서 노이즈 마스크와 안내(image guidance) 특징의 결합이 단순하면 흐린 경계 상황에서 의미·경계 표현을 동시에 놓치기 쉽다. MLFFM-SegDiff는 mask branch와 guidance branch를 분리 인코딩하되 잔차 블록과 피드백(원소곱 기반)으로 조건 상호작용을 촘촘히 구성하고, bottleneck에서 attention 기반 혼합으로 전역 표현을 보강한다. 또한 skip 단계에서 SkipBlock의 공간/채널 attention과 적응형 스케일 정렬 및 원소곱 융합을 통해 레벨 간 정보를 decoder가 함께 활용하도록 설계했으며, 경계 민감 손실은 거리 기반 가중치로 경계 픽셀의 학습 중요도를 시간에 따라 조절해 경계 복원을 유도한다.

- **Empirical Impact**: 실험은 ISIC2018, PH2, HAM10000에서 Accuracy, F1-score, Jaccard index, Recall, Dice 등 다수 지표로 평가됐고, U-Net·SwinUNETR·DermoSegDiff 등 대표 방법 대비 전반적 성능 우위가 보고됐다. 특히 전체 평균에서 Jaccard index 0.8546, Dice coefficient 0.9207을 달성해 병변 영역 커버리지와 겹침 품질이 향상됐음을 수치로 확인했다. 논문은 ablation을 통해 dual-path encoder, MLFFM, boundary-sensitive loss가 각각 또는 조합으로 Dice와 sensitivity(회상/커버리지) 개선에 기여하며, 경계가 불명확한 케이스에서도 예측 윤곽을 더 잘 따르는 정성 결과를 제시했다.



### The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Repor (https://arxiv.org/abs/2606.26529)
Comments:
          20 pages, 8 figures. Reproducibility deposit: this https URL

- **Prior Approaches**: 기존 AI safety 평가는 모델이 ‘지정된’ 위험 신호를 얼마나 잘 탐지·보고하는지에 초점이 맞춰져 있다. 하지만 실제 사고는 종종 평가에 포함되지 않은 위험(미지정 co-present 신호)에서 발생하며, 이 불일치를 설명하는 정밀한 내재 메커니즘은 부족했다.

- **Core Contribution**: 이 논문은 task-conditioning(좁은 과제 지시)이 모델이 원래는 보고할 수 있었던 co-present safety-critical 신호의 보고를 억제하는 현상을 ‘Inattentional Gap’으로 명명한다. 같은 모델이라도 입력은 동일한데, 지시가 좁아지면 벤치마크의 기준선 수준 성능이 유지되면서도 실제 위해를 만드는 신호는 침묵할 수 있다고 주장한다.

- **Technical Challenges**: 핵심 과제는 ‘누락이 단순 능력 부족인지, 과제 지시로 인한 보고 억제인지’를 within-item으로 분리하는 것이다. 이를 위해 동일 입력에 대해 (1) open 조건(보고 가능성과 함께 무엇을 볼지)과 (2) task 조건(보고 범위를 좁혀 scoping)을 비교하고, 판정은 키워드 매칭이 아닌 두 명의 언어모델 judge 합의로 수행해 task-induced omission을 입증했다.

- **Empirical Impact**: 실험 결과, radiology·자율주행 텍스트 시나리오와 흉부 X-ray 비전 과제에서 억제는 테스트한 모든 모델에서 관찰됐고, 모델 스케일이 커져도 완화되지 않았다. 또한 이유 모델에서도 지속되며, gap의 상관은 모델 크기보다 model family(제공자/계열)에 더 크게 나타났고, task-free(open)에서는 동일 신호를 훨씬 높은 비율로 보고해 측정된 벤치마크 안전이 현실 안전을 과대평가할 수 있음을 시사한다.



### Budget-Aware Keyboardless Interaction (https://arxiv.org/abs/2606.26508)
Comments:
          SOICT 2024

- **Prior Approaches**: 기존 키보드 대체 입력은 키보드 없는 가상 키보드(평면/AR/XR)나 센서 기반(반지·손목 밴드·IMU) 등으로 나뉜다. 다만 Bayesian decoder 기반은 비표준 타이핑 패턴에 약하고, 프로젝터·AR HMD 같은 투사/몰입형 장치는 장비·세팅 부담이 커 실사용 확산이 어렵다는 한계가 지적된다. 또한 종이 키보드를 쓰더라도 끝점 마커 등 추가 표식이나 복수 카메라/특수 조명·조명 방향 민감도 같은 조건이 필요했다.

- **Core Contribution**: 이 논문은 표준 카메라 1대와 인쇄된 키보드 레이아웃 종이만으로 가상 키보드와 키 입력을 인식하는 저비용 시스템을 제안한다. 키보드/키 영역은 YOLOv8 계열 딥러닝으로 찾고, 손가락의 nail(손톱) 색 변화를 통해 touch down/up을 판별해 어떤 키가 눌렸는지 추적한다. 특히 기존 종이-기반 방식처럼 추가 마킹 없이 QWERTY/AZERTY 및 Mac/Windows 레이아웃을 처리하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 카메라 각도 때문에 키보드가 사다리꼴로 보이는 기하학적 변환 문제와 (2) 손톱 색 기반 touch 판별의 조명·각도·손가락 압력 의존성이다. 이를 위해 키보드 코너를 찾아 homography로 상면 직교 뷰로 보정하고, 키는 별도 객체 검출 YOLO로 위치를 명확히 고정한다. touch는 MediaPipe 손 랜드마크로 fingertip을 추정한 뒤, segment 기반 손톱 영역만 색 분석에 쓰며(불필요한 분할 최소화) 손톱 Hue 변화를 통해 press 여부를 판정하는 방식으로 해결했다.

- **Empirical Impact**: 실험 결과 키보드 영역 검출 AP는 92%에 도달했지만, 여러 레이아웃이 섞인 데이터셋 영향으로 키 인식 정확도는 약 70% 수준이었다. touch detection 정확도는 조명 조건 영향으로 약 36%로 낮게 나타났으나, 사용자 20명을 대상으로 한 사용성 평가에서는 대부분이 “종이와 카메라만으로 입력된다”는 점에서 흥미를 보이며 만족도가 높았다. 다만 지연(lag)과 정확도 개선 요구가 반복돼, 저자들은 touch 성능 향상과 OCR·LLM 기반 자동 보정까지 확장 계획을 제시했다.



### DanceDuo: Bridging Human Movement and AI Choreography (https://arxiv.org/abs/2606.26507)
Comments:
          SOICT 2024

- **Prior Approaches**: 기존 music-to-dance generation은 retrieval 기반에서 출발해 CNN/RNN/GAN/Transformer 등으로 모션 합성을 시도해 왔지만, 대체로 사용자 참여를 유도하는 인터랙션이 약하다는 한계가 지적됐다. 또한 diffusion이 등장했지만, 실제 사용 맥락에서의 학습·피드백 경험을 시스템으로 엮는 시도는 상대적으로 제한적이었다. 관련 HCI 연구들은 모션 캡처·피드백은 다뤄왔으나, 음악 조건 생성과 개인 수행 비교를 한 플랫폼에서 연결한 사례는 부족했다.

- **Core Contribution**: 이 논문은 diffusion 모델 기반 music-to-dance를 실제 사용자가 체감할 수 있게 묶은 플랫폼 DanceDuo를 제안한다. 사용자는 음악 트랙과 humanoid 모델을 선택하고, 자신의 춤 영상을 가져와 AI 생성 안무와 side-by-side로 비교할 수 있다. 또한 모든 난이도의 사용자를 고려한 스코어링 공식(점수 곡선)을 설계해 연습 동기와 창의적 탐색을 함께 유도한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 음악의 박자·리듬과의 temporal 동기와 (2) 관절·신체 표현의 spatial 자연스러움을 동시에 만족시키는 것이다. 이를 위해 DanceFusion이라는 conditional diffusion을 채택해 과거·미래 모션을 조건으로 삼고 autoregressive의 장기 생성 문제와 non-autoregressive의 부자연스러운 전환 문제를 완화하며, SMPL 기반 mesh 표현으로 신체 디테일을 보존한다. 개인 영상에서 SMPL pose를 ROMP로 추정하고, 선택적 프레임 추출과 보간으로 실시간 처리 부담을 낮춘 뒤 Blender/Unity 파이프라인으로 시각화까지 완성한다.

- **Empirical Impact**: 사용자 연구(참여자 5명)에서 DanceDuo는 전반적으로 직관적이고 사용하기 쉽다는 평가를 받았으며, 특히 춤 비교 기능이 가장 몰입감과 동기 부여에 효과적이라는 피드백이 두드러졌다. 생성 애니메이션 품질은 대체로 긍정적이었지만 음악-동작 동기 측면에서 개선 여지가 있다고 응답했다. 결과적으로 DanceDuo는 음악 기반 AI 안무를 ‘보기’에서 ‘연습과 비교’로 확장하며, 오락과 학습/프로 연습 모두에 활용될 수 있는 인터랙티브 방향성을 제시한다.



### WatchAct: A Benchmark for Behavior-Grounded Robot Manipulation (https://arxiv.org/abs/2606.26443)
- **Prior Approaches**: 기존 조작 벤치마크는 대체로 현재 장면(정적 관측)만을 입력으로 실행 능력을 평가하거나, 비디오를 데모 재현/자기 시점 기억으로 쓰는 경우가 많았다. 이 때문에 로봇이 다른 사람의 행동으로부터 사건 순서·의도·누적 상태를 추론하는 ‘관찰 기반 추론’을 체계적으로 검증하기 어려웠다.

- **Core Contribution**: WatchAct는 실제 사람의 행동 비디오(관찰 증거)와 언어 지시를 입력으로, 정렬된 시뮬레이터 및 LIBERO 실행 태스크까지 연결해 “관찰된 인간 행동을 이유로 삼는 조작”을 평가하는 새로운 벤치마크를 제안한다. 14개 태스크(총 3,000개, long-horizon)와 4개 인지 도메인(Event Grounding, Procedural Reasoning, Implicit Intent Inference, Episodic Reasoning)으로, 이벤트·절차·암묵 의도·에피소드 추론을 분해해 측정하도록 설계됐다.

- **Technical Challenges**: 핵심 기술적 난관은 비디오-언어 추론이 장면의 객체 이력/공간 관계/미완 절차/의도를 정확히 계획으로 변환해야 하고, 이어서 물리적 실행이 긴 시퀀스를 안정적으로 수행해야 한다는 점이다. 논문은 disentangled evaluation로 (1) VLM의 video-to-plan reasoning, (2) oracle plan 하의 policy execution, (3) planner–policy 통합 파이프라인의 end-to-end 성공을 분리 측정해 오류가 어디에서 누적되는지 드러내는 프로토콜을 도입했다.

- **Empirical Impact**: 실험 결과, 시뮬레이션과 Franka Research 3 로봇 모두에서 현재 시스템은 WatchAct를 사실상 ‘해결’하지 못하는 수준으로 나타났다. 예컨대 Gemini-3.1-Pro는 Plan SR 36.8%(인간 97.1% 대비 큰 격차), 통합 파이프라인 Success Rate는 시뮬 16.3%, 실로봇 14.0%였고, plan 길이가 늘수록 성능이 급격히 붕괴했으며 out-of-domain 일반화도 크게 떨어졌다. 저자들은 이 결과가 로봇이 사람과 함께 일할 때 필요한 관찰 기반 추론과 제어의 미해결 영역을 실험적으로 정량화하는 데 의미가 있다고 보고했다.



### Revealing Mammographic Phenotypes in Deep Learning Breast Cancer Risk Models (https://arxiv.org/abs/2606.26431)
- **Prior Approaches**: 유방촬영(mammogram) 기반 deep learning은 유방암 5년 위험 예측 성능을 끌어올렸지만, 모델이 학습한 영상 패턴이 어떻게 반복되는지 충분히 탐색되지 않았다. 기존 해석 방법은 주로 단일 이미지의 saliency map에 의존해, 대규모 코호트에서 공통적으로 나타나는 ‘형태적 표현(phenotype)’을 안정적으로 찾아내기 어렵다.

- **Core Contribution**: 논문은 사전학습된 모델의 patch embedding을 클러스터링해 반복되는 유방촬영 phenotypes를 분리하고, 이들이 5-year cancer risk와 연관됨을 보여준다. 또한 위험을 높이는 phenotypes가 무엇을 포착하는지(조직 구조 vs 비의학적 단서)를 체계적으로 연결해 임상적 시그니처를 제시한다.

- **Technical Challenges**: 핵심 과제는 수많은 국소 패치에서 반복 표현을 추출하되, 단순 시각적 강조(saliency)처럼 단일 샘플 중심 해석에 머물지 않는 대표 phenotype을 구성하는 것이다. 저자들은 Mirai에서 얻은 patch embedding을 기반으로 클러스터를 만들고, 각 클러스터가 위험 증가와 어떤 영상 특성을 공유하는지 분석해 해석 가능성을 높였다.

- **Empirical Impact**: 결과적으로 위험을 높이는 phenotypes는 치밀한 조직(dense tissue), 미세석회화(microcalcifications) 같은 복잡한 구조를 포함하면서도, clips 같은 ‘shortcut artifact’도 함께 포착하는 것으로 나타났다. 해당 phenotypes는 고령과 BI-RADS density가 높은 경우와 강하게 상관하며, tissue pattern을 AI 위험 점수와 연결해 임상 지표뿐 아니라 잠재적인 모델 교란 요인(confounder)까지 드러내는 데 의미가 있다.



### Rethinking Training & Inference for Forecasting: Linking Winner-Take-All back to GMMs (https://arxiv.org/abs/2606.26424)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 자율주행 궤적 예측에서 다중 후보(multi-modal) 예측은 일반적으로 GMM(Gaussian Mixture Model) 형태로 확률 가중치를 함께 내고, 이후 post-hoc로 후보를 줄이거나 병합해 평가용 M개 모드를 만든다. 이때 많은 방법이 학습에서는 winner-take-all(WTA)처럼 가장 가까운 모드에만 책임을 주는 hard assignment를 사용해 mode collapse는 막지만, 결과적으로 모드 확률이 “믿을 만하지 않게” 된다.

- **Core Contribution**: 이 논문은 WTA 학습이 GMM 학습의 전제인 ‘확률적 모드’와 불일치하며, WTA의 hard one-hot 배정이 K-means 스타일의 K-means-like hard assignment로 해석된다고 제시한다. 그 결과 모드가 의미적으로 오버세그먼트되고, 개별 모드는 낮은 확률로 쪼개져도 그 합은 큰 확률인 상황이 생겨 mode pruning을 망친다는 원인을 정리한다. 이를 바탕으로 재학습 없이 test-time 사후 보정 2가지를 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 WTA로 학습된 모델의 ‘확률 가중치’가 추론 시 GMM의 posterior로 해석되기 어렵다는 점이다. 논문은 (1) posterior-weighted merging으로 인접 후보를 가중 평균해 모드를 합치고, (2) 1-step expectation-maximization(EM)으로 hard label을 soft responsibility로 바꿔 확률 질량을 이웃 모드로 분산시키는 방식으로 해결한다. 추가로 이 접근이 WTA의 K-means적 특성과 직접 연결된다는 해석틀도 함께 제공한다.

- **Empirical Impact**: NuScenes Prediction 및 Waymo WOMD에서 Wayformer, MTR-e2e, EMP 등 여러 WTA 계열 아키텍처에 대해, 단순 greedy/NMS처럼 ‘예측 확률 하나’에 의존한 선택보다 병합·1-step EM 보정이 더 나은 최종 선택을 만든다. 특히 거리 오차 기반 지표(minADE/minFDE)에서 개선이 확인되며, 분포 품질을 반영하는 brier-minFDE 및 negative log-likelihood도 함께 좋아진다. 결론적으로 재학습 없이도 모드 확률을 더 정보적으로 정렬해 downstream planning에 유리한 형태의 모드 집합을 얻을 수 있음을 보여준다.



### Tailor Made Embeddings for Quantum Machine Learning (https://arxiv.org/abs/2606.26312)
Comments:
          17 pages, 17 figures

- **Prior Approaches**: 기존 quantum machine learning에서는 classical 데이터를 양자 상태로 옮기는 임베딩 방식이 성능을 좌우했지만, 방식에 따라 재구성 절차가 까다로웠다. 예를 들어 amplitude embedding은 복원을 위해 full quantum state tomography가 필요하고, angle embedding은 circuit inversion 같은 가정에 크게 의존하는 경향이 있다.

- **Core Contribution**: 이 논문은 variational autoencoder(변분 오토인코더) 패러다임을 quantum machine learning에 확장해, 데이터에 맞춘 task-specific quantum embedding을 학습하는 프레임워크를 제안한다. 학습된 encoder-decoder 구조로 고차원 데이터를 양자 표현으로 압축하면서도 복원이 가능하도록 설계했다.

- **Technical Challenges**: 핵심 어려움은 적은 측정 횟수로도 원본 데이터를 안정적으로 reconstruct할 수 있는 학습 목표를 구성하는 것이다. 저자들은 learned decoder로 reconstruction을 유도하고, 원복에 필요한 정보가 polynomial number of measurements 수준에서 충분하도록 회로 중심(circuit-centric) 분류 및 재구성 학습을 결합해 이를 해결했다.

- **Empirical Impact**: ImageNet 등 고차원 데이터에 대해 13-qubit 양자 표현으로 압축하되 reconstruct가 가능함을 보였다. MNIST(3 vs 5)에서는 98.5% validation accuracy로 classical neural network baseline(99.7%) 대비 1.2%p 이내 성능을 보였고, naive amplitude-embedding 대비 30%p 이상 개선됐으며, IBM quantum hardware에서도 실기기 잡음 하에서 임베딩이 안정적이고 재구성 가능함을 확인했다.



### Rendering Novel Views of MRI Using 3D Gaussian Splatting (https://arxiv.org/abs/2606.26236)
- **Prior Approaches**: 척추 MRI에서 요추관(척추관)·신경공·재(부)관절 하부 등 협착은 다단계(ordinal) 중증도 등급으로 판정되며, 정확한 평가는 관심 해부학이 해당 프레임에 선명히 보일 때 좌우된다. 기존에는 비방사선 촬영 평면(축에 정렬되지 않은 촬영) 때문에 관심 부위가 프레임 내에서 부분적으로만 보이는 문제를 그대로 두거나, 단순 리샘플링(예: voxel interpolation)으로 해결하려 했다.
또한 2D/3D 딥러닝 기반 자동 등급화 연구는 존재하지만, 특정 촬영 방향(예: sagittal)이나 입력 조건에 강하게 의존해 실제 임상에서 흔한 비정렬/이방성 voxel을 충분히 다루기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 3D Gaussian Splatting(3D 가우시안 스플래팅)을 척추 MRI의 비정렬 축(slice)에서부터 해부학적으로 정렬된 임상용(view plane) 단면을 재구성·렌더링하는 리샘플링 프레임워크로 확장했다. 이렇게 생성된 ‘정렬된 재샘플 스캔’은 원본 MRI보다 관심 해부학의 가시성을 높여 국소 협착의 ordinal severity 등급 예측에 유리함을 보인다.
또한 리샘플링 결과를 그대로 협착 중증도 분류 다운스트림 모델에 투입해, voxel interpolation 기반 리샘플링과 비교 평가를 수행한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) MRI가 투영(projection)이 아닌 그레이스케일 볼륨이어서 기존 카메라 기반 Gaussian Splatting/NeRF 관행을 그대로 적용하기 어렵고, (2) 입력이 sparse하며 voxel이 이방성이고 스캔 간 voxel 크기도 달라 일반화가 까다롭다는 점이다. 저자들은 MRI용 3D Gaussian 파라미터를 유지한 채, source slice의 3D pose와 가우시안의 3D 분포를 바탕으로 타깃 view plane에서의 강도를 Mahalanobis distance 기반 가중 합으로 직접 계산하도록 설계했다.
또한 의료에서 ‘범위 밖 환각’이 치명적일 수 있어, 가우시안 초기화·소스 크롭·CUDA 기반 효율화(plane 근처 가우시안만 연산)로 학습 안정성과 계산량을 동시에 개선했다.

- **Empirical Impact**: 실험 결과, 재샘플링된 스캔은 비정렬 원본에서의 협착 등급 성능을 전 조건에서 일관되게 개선했으며, 특히 Gaussian Splatting 기반 재샘플(Test-R-GS)이 voxel interpolation(Test-R-VI)보다 모든 조합에서 더 높은 분류 성능을 보였다. 이미지 품질 지표에서는 PSNR/SSIM이 VI가 근소 우위인 경우도 있었지만, 다운스트림 등급화에선 GS가 더 잘 작동해 ‘구조적 유사도보다 관련 특징 표현의 질’이 중요할 수 있음을 시사한다.
저자들은 코드 공개 계획을 밝혔고, 방법론을 척추 협착뿐 아니라 다른 비투영(non-projection-based) 의료 영상 모달리티와 해부학 영역 전반으로 확장 가능한 방향성도 제시했다.



### Fast LeWorldMod (https://arxiv.org/abs/2606.26217)
- **Prior Approaches**: LeWM 같은 JEPA 기반 재구성-free 시각 월드 모델은 픽셀을 직접 복원하지 않고, 잠재 임베딩을 예측해 reward-free 목표 조건 계획을 지원합니다. 다만 후보 행동열 평가는 로컬 one-step latent transition을 자가회귀로 반복 롤아웃해야 해서 계산 비용이 커지고, 초기 예측 오차가 horizon이 늘수록 누적되며 계획 신뢰성이 떨어집니다. 즉 dynamics-query 인터페이스가 순차 롤아웃에 묶여 있는 점이 병목으로 작동합니다.

- **Core Contribution**: Fast-LeWM은 “한 스텝 전이 예측” 중심 인터페이스를 “action-prefix prediction”으로 바꿔, 현재 잠재 zt와 행동 접두(prefix)를 입력받아 여러 horizon의 미래 잠재를 병렬로 직접 예측합니다. 행동 접두를 기본 예측 단위로 삼아, 서로 다른 길이의 prefix가 누적시키는 action effect를 각 horizon별로 분리해 학습하도록 설계했습니다. 또한 prefix 토큰에 대한 dense prefix-level supervision을 통해 연속적인 상태 진화를 더 직접적으로 강제합니다.

- **Technical Challenges**: 핵심 과제는 (1) 후보 행동열을 다 horizon에서 빠르게 평가하면서 (2) 롤아웃처럼 단계별 중간 예측 오차가 누적되지 않게 만드는 것입니다. Fast-LeWM은 행동열을 causal Transformer로 인코딩해 horizon별 prefix 토큰을 만든 뒤, 병렬 latent predictor가 anchor latent(z t)와 각 prefix 토큰을 조건으로 대응 미래 latent를 한 번의 forward pass로 산출하게 했습니다. 더불어 현재 상태를 state token으로 추가 조건화해 동일 prefix라도 초기 구성에 따라 달라지는 action effect를 구분하도록 했습니다.

- **Empirical Impact**: 네 가지 goal-conditioned planning 작업에서 Fast-LeWM은 평균 성공률을 LeWM 85.8%에서 90.5%(self-consistency 포함 92.0%)로 끌어올리면서도 계획 시간을 크게 줄였습니다. dynamics-evaluation은 31.4s→8.0s, 전체 CEM solve time은 54.4s→28.3s로 감소했으며, open-loop latent loss의 증가 속도도 horizon이 길어질수록 더 완만했습니다. 추가적으로 probe 실험에서 물리 변수(위치·각도 등)에 대한 비선형 정보 보존이 강화되는 경향이 관찰돼, action-prefix 기반 학습이 더 풍부한 상태 표현을 만든다는 점을 뒷받침합니다.



### TaskNPoint: How to Teach Your Humanoid to Hit a Backhand in Minutes (https://arxiv.org/abs/2606.26215)
- **Prior Approaches**: 기존 연구는 인간 시연 기반의 로봇 모션 학습에서 주로 특정 궤적 추적에 강한 강화학습(RL)이나, 방대한 인간 데이터로부터 전신(whole-body) 컨트롤을 학습하는 접근으로 나뉘었습니다. 전자는 보상 튜닝 편향으로 비효율적·비자연스러운 행동이 생길 수 있고, 후자는 고품질 데이터 전처리(클리닝) 비용이 커지는 경향이 있습니다. 스포츠 플레이용 휴머노이드에서도 작업별 튜닝이나 상체·하체 분리 같은 우회가 흔해 학습 효율과 범용성 사이의 균형이 과제로 남아 있었습니다.

- **Core Contribution**: 이 논문은 동적 스포츠 기술을 ‘상호작용 윈도우’라는 짧은 구간으로 구조화해, 성공을 결정하는 핵심 순간만 맞추면 전체 동작을 효율적으로 학습할 수 있다고 주장합니다. 그에 따라 TaskNPoint 학습 프로토콜을 제안하며, 코치(교사)는 (1) 이산 기술 집합, (2) 기술별 1회(또는 소수) 시연, (3) 상호작용 윈도우 식별, (4) 목표를 제공하고, 학습자는 나머지 궤적·강건성·일반화를 시뮬레이션에서 채웁니다. 특히 목표를 3D 공간의 포인트와 상호작용 시점에 조건화해, 소수 시연만으로 큰 작업공간을 커버하는 다중 기술 정책을 지향합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 시연 영상으로부터 ‘정확한 상호작용 시점과 3D 상호작용 파라미터’를 추출하고, 관측·모델 불일치에도 목표 성능을 유지하는 것입니다. 논문은 PromptHMR 등으로 3D 인체 자세를 복원한 뒤 MLE 기반 다중 뷰 합의로 시연의 실패 모드를 줄이고, GMR로 로봇에 동작을 리타겟팅하면서 상호작용 파라미터(p, 방향 ν, 자세 n, 시점 t*)를 목표로 정의합니다. 학습 시에는 상호작용 윈도우 주변에서 reward를 누적하고, 훈련용 목표를 랜덤 샘플링해 단일 시연이 zero-shot으로 미관측 목표 위치까지 일반화하도록 했습니다.

- **Empirical Impact**: 실험에서는 Unitree G1 휴머노이드가 테니스 포핸드/백핸드를 치고, 사람이 던지는 공을 맞춘 뒤 킥하며, 박스를 새로운 위치에서 집어오는 작업에서 성공을 보였습니다. 단일 GPU로 1시간 미만 학습과 함께, per-task reward 튜닝 없이도 하드웨어 배치가 가능했으며, 접촉 시점 구조가 speed 변화에 대해 in-distribution 실패를 ‘완만하게’ 만들어 로봇이 넘어지지 않고 계속 동작할 수 있음을 보여줬습니다. 또한 시뮬레이션 일반화 성공률(GSR)에서 넓은 3D 볼륨 커버가 확인되어, 동적 스포츠 기술 교육을 ‘소수 시연+상호작용 윈도우’ 패러다임으로 확장할 수 있음을 시사합니다.



### From Structure to Synergy: A Survey of Vision-Language Perception Paradigm Evolution in Multimodal Large Language Models (https://arxiv.org/abs/2606.26196)
- **Prior Approaches**: 기존 연구는 비전-중심과 언어-중심으로 분절돼, 시각 지각(지역/인스턴스 추출·국소화·관계 추론)이 어떻게 하나의 통합된 능력으로 진화하는지 일관되게 다루지 못했다. 또한 인코더를 고치거나(Region-aware 모듈, ROI 생성, 멀티 인코더/프로젝터) 특정 디코딩 과제에 보조 손실을 거는 방식처럼, 지역 수준 성능 향상에 초점이 맞춰져 정밀한 픽셀 수준 지각으로의 패러다임 전환이 체계적으로 정리되지 않았다.

- **Core Contribution**: 이 논문은 MLLM의 비전-언어 지각을 인간의 선천적 지각 능력처럼 ‘고유하고 통합된 능력’으로 정식화하고, 이를 기준으로 최초의 체계적(5-stage) 설문/분류를 제안한다. 특히 지역(인스턴스) 단위 자연어 질의에 대한 인식 범위를 명확히 하고, 인코더-중심→디코더-중심→동적 지각→아키텍처 프리 전략→통합 프레임워크로 이어지는 진화 경로와 대표 방법을 정리해 로드맵을 제공한다.

- **Technical Challenges**: 통합 지각을 실현하는 핵심 난제는 이미지의 의미 있는 ‘국소 증거’를 선택적으로 찾아 처리하고, 대화/지시 기반으로 해당 영역을 정밀하게 국소화·세분화하며, 그 성능을 입력 상황에 따라 일관되게 유지하는 데 있다. 논문은 이를 해결하기 위한 방향으로 (Stage I) ROI 제안·프롬프트/쿼리 인지·프로젝터/커넥터 최적화와 (Stage II) 보조 디코더(예: segmentation 토큰) 및 멀티-디코더/특화 디코딩 전략으로 지역 수준을 픽셀 수준으로 끌어올리는 설계를 묶어 설명한다.

- **Empirical Impact**: 설문은 여러 시기와 축에서 축적된 대표 기법들을 한 프레임으로 연결해, ‘무엇이 실제로 지각 능력을 강화하는가’를 관찰 가능하게 만든다는 점에서 실무적 가치가 크다. 또한 열려 있는 과제(진짜 general, unified 멀티모달 지능으로의 확장)에 대한 연구 방향을 제시해, 향후 MLLM을 지각-중심 에이전트로 발전시키는 데 참고 가능한 행동 지침을 제공한다.



### Dot-Flik: A Scalable Edge AI Architecture for Distributed Insect Monitoring (https://arxiv.org/abs/2606.26121)
- **Prior Approaches**: 기존 비전·딥러닝 기반 곤충 모니터링은 높은 정확도를 보여도, 비싼 고성능 하드웨어나 지속적인 cloud 연결에 의존하는 경우가 많아 도시 단위 확장에 비용·전력·네트워크 한계가 생겼습니다. edge AI로 분류를 수행하는 사례도 있으나, 카메라 스트림의 대부분을 그대로 처리/전송해 데이터 중복(빈 프레임·배경 중심 장면)이 에너지와 대역폭을 먼저 소모하는 문제가 남아 있습니다. 또한 시간 간격 샘플링(time-lapse)은 콘텐츠를 보지 못해 드문 방문을 놓칠 수 있어, “전송·처리 이전 단계”의 구조적 최적화가 부족하다는 지적이 제기됩니다.

- **Core Contribution**: 이 논문은 곤충 활동이 있는 프레임만 선별해 edge에서부터 불필요한 데이터를 줄이는 Dot-Flik을 제안합니다. 핵심은 (1) deep learning 추론 없이도 움직임 정보를 활용해 프레임을 걸러내고, (2) 센서 계층의 전처리로 분류 인퍼런스 부담과 통신량을 분리·완화하며, (3) 중앙 AI 분류는 배치 처리로 효율을 높이는 계층형 IoT 아키텍처를 구현한 점입니다. 이를 통해 모니터링 면적이 늘어도 전송·계산이 비례 폭증하지 않는 “확장 가능한 구조”를 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 edge에서 사용할 수 있는 제약(저전력·저가 보드) 안에서, 바람에 흔들리는 식생 잡음은 줄이면서 곤충의 국소적 움직임은 놓치지 않는 프레임 선별을 설계하는 것입니다. 연구진은 temporal differencing(프레임 간 차이)로 움직임 가능성을 만들고, gamma-corrected motion amplification(잡음 증폭을 억제한 보정)과 block-based motion density(블록 단위 밀도 임계치)로 “분산된 배경 흔들림 vs 집중된 곤충 움직임”을 구분하도록 구성했습니다. 또한 중앙 노드는 Raspberry Pi 5 + Hailo 가속기로 분류하고, edge 노드는 Raspberry Pi Zero 2 W에서 필터링 후 H.264 스트리밍으로 후보 프레임만 전달하도록 역할을 엄격히 분리해 실시간 30 FPS 운용을 맞췄습니다.

- **Empirical Impact**: 실외 도시 정원 배치 실험에서 빛 바람 조건(5–15 km/h)에서는 60–80% 프레임 감소가 관측됐고, 강풍에서는 20–40%로 줄어들지만 수동 확인 결과 곤충 활동이 있는 프레임은 대체로 유지됐다고 보고됩니다. 중앙 분류 파이프라인 기준으로 edge-전체 시스템은 30 FPS를 지속하며 추가 연산 여유(computational headroom) 12.8 ms 수준을 확보했고, 전력은 전송률(프레임 드롭률)에 따라 22.6%까지 절감되는 경향이 확인됐습니다. 또한 Dot 센서 스트림을 여러 개 동시 운용할 때 central 노드의 계산이 병목이 되며, 설계 투영상 5–6개의 concurrent edge streams를 지지해 저비용·밀집형 biodiversity monitoring 네트워크의 실용 기반을 제시했다는 점이 의미 있습니다.



New uploads on arXiv(cs.AI)

### Language-Based Digital Twins for Elderly Cognitive Assistanc (https://arxiv.org/abs/2606.27334)
Comments:
          Accepted and published in the Proceedings of the ACM International Conference on PErvasive Technologies Related to Assistive Environments (PETRA 2026). The final published version is available through the ACM Digital Library

- **Prior Approaches**: 기존 digital twin 연구는 의료에서 개체의 지속 갱신을 강조하지만, 주로 생체 신호·집단 수준 표현에 초점이 맞춰져 개인별 언어 습관 같은 미세한 행동 양상을 정교하게 재현하지 못했다. 인지건강 쪽에서도 Mild Cognitive Impairment(MCI) 예측이나 상태 분류가 중심이었고, 일상 대화의 장기적 변화까지 ‘개인화된 행동 에뮬레이션’으로 모델링하는 연구는 제한적이었다. 대화 기반 접근도 대체로 전체 유사도 평가에 그쳐 미래 인지 궤적과의 연결이 약했다.

- **Core Contribution**: 이 논문은 언어 기반 language-based digital twin을 제안해, LLM이 노년층 개인의 대화 행동을 모사하도록 개인화된 생성기를 만든다. pause와 tempo 같은 stylometric cues와 참가자 메타데이터를 함께 넣어, 화행의 스타일·시간적 리듬·인지 관련 신호를 생성 과정에 반영한다. 또한 생성 결과가 ‘인지적으로도 일관된지’를 함께 보려는 평가 설계를 제공한다.

- **Technical Challenges**: 핵심 난제는 (1) 개인의 말하기 스타일을 실제 대화처럼 유지하면서 (2) 생성이 단순 유창성에 그치지 않고 인지 상태와 연관된 일관성을 갖도록 만드는 것이다. 저자들은 stylometric token으로 발화 타이밍/리듬을 학습 입력에 통합하고, fidelity와 cognitive consistency를 동시에 보도록 multi-head conditional variational autoencoder(cVAE) evaluator를 설계해 복원 품질과 MoCA 점수 예측을 함께 최적화한다. cVAE는 생성-실제 응답의 복원 오차와 인지 점수 회귀를 결합해 ‘언어적 유사성’과 ‘인지 정합성’을 동시 측정한다.

- **Empirical Impact**: I-CONECT 데이터셋(장기 자연 대화 포함)에서 실험한 결과, twin 생성 답변은 identity-specific 특징 보존에서 real에 가까운 정확도를 보였고 base GPT-generated responses보다 크게 개선됐다. 복원 오차와 MoCA 예측 오차도 실제 응답 범위와 유사하게 유지되며, 인지 관련 정보가 생성 과정에 효과적으로 남는 것으로 나타났다. 저자들은 이러한 결과가 비침습적이면서도 지속적인 개인 인지 모니터링의 확장 가능성을 보여준다고 강조하며, 향후 audio/video 등 멀티모달로 확장해 정서·행동 신호까지 통합할 계획이다.



### When Does Combining Language Models Help? A Co-Failure Ceiling on Routing, Voting, and Mixture-of-Agents Across 67 Frontier Models (https://arxiv.org/abs/2606.27288)
- **Prior Approaches**: 라우팅, 보팅, 캐스케이드, 퓨전, mixture-of-agents 같은 멀티모델 조합은 단일 모델보다 정확도를 높이려는 접근이다. 기존 진단은 모델 간 오류의 pairwise error correlation ρ를 낮게 보면 다양성이 이득이 된다고 보는 관행에 의존한다. 하지만 이런 통계는 “모든 모델이 같은 질의에서 동시에 틀리는 꼬리”의 크기를 직접적으로 포착하지 못한다.

- **Core Contribution**: 이 논문은 멀티모델 시스템의 정확도 상한이 ρ가 아니라 모든 모델이 한 질의에서 동시에 실패하는 all-wrong rate β에 의해 결정된다고 보인다. 어떤 정책이든(라우터/보팅/캐스케이드 포함) 출력이 항상 멤버 모델 중 하나의 답이면 정확도는 1−β를 넘을 수 없고, 따라서 single-best 대비 개선 여지도 (1−β)−asb로 제한된다. 또한 Clopper-Pearson bound로, 학습 전에도 β의 최댓값에 대한 유한표본 인증서를 제공해 “얼마나 이길 수 있는가”를 미리 계산 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 β를 ρ로부터 추정·진단하는 것이 원리적으로 불가능하다는 점이다; 동일한 주변분포와 pairwise 상관을 가져도 all-wrong tail β는 달라질 수 있다. 저자들은 β를 직접 추정·인증하는 방식으로 전환하고, tetrachoric-calibrated single-factor 모델과 Gaussian copula 등으로 β를 예측했을 때 꼬리가 체계적으로 과소평가되는 현상을 분석한다. 특히 “정확도는 평균이 아니라 꼬리의 공동실패가 지배한다”는 결론을, copula 형태가 tail dependence를 충분히 담지 못한다는 설명과 함께 실증한다.

- **Empirical Impact**: 67개 제공사/총 21개 공급자 계열의 6767개 모델 규모에서, open-ended 수학에서 관측된 β는 0.052로 추정치 0.023보다 약 2.5배 두껍게 나타났고(90% CI 1.7~3.4), 코드 실행 평가에서도 같은 경향이 재현된다. 또한 질문을 GPQA-Diamond의 multiple-choice에서 free-response로 바꾸면 β가 0.127까지 커져 답안 형식이 공동실패의 위치를 바꿀 수 있음을 시사한다. 흥미롭게도 낮은 ρ를 가진 이질적 앙상블은 일부 이점을 보이지만, 검증 가능한 과제 풀에서는 single best 단일 모델을 조합으로 이기는 일이 드물며, 개선의 원천은 “더 많은 모델 추가”가 아니라 “서로 다른 질의에서 실패가 어긋나는지”에 달려 있음을 강조한다.



### Prompt Injection in Automated Résumé Screening with Large Language Models: Single and Multi-Injection Settings (https://arxiv.org/abs/2606.27287)
- **Prior Approaches**: 기존 연구는 LLM 기반 채용 스크리닝이 후보자의 텍스트를 어떻게 평가하는지에 초점을 맞추었지만, 후보자가 알고리즘의 허점을 이용해 평가를 교란하는 ‘조작 전략’의 영향을 충분히 다루지 못했다. 특히 prompt injection을 자동화된 이력서 평가 맥락에서, 경쟁 구도(여러 명의 동시 조작)와 후보 품질 분포까지 함께 고려한 실증은 제한적이었다.

- **Core Contribution**: 이 논문은 automated résumé screening에서의 prompt injection을 ‘새 자격을 추가하지 않지만 자기 PR 문구를 통해 LLM 평가를 유도하는 미묘한 텍스트 삽입’으로 정의하고, 그 효과를 통제 실험으로 정량화했다. 또한 후보 간 조작의 빈도와 이력서 품질의 동질/이질 여부가 조작 성공률과 공정성에 어떤 영향을 주는지 체계적으로 분석했다.

- **Technical Challenges**: 핵심 기술적 난제는, 주입된 문구가 실제 역량 변화 없이도 평가를 바꾸는지(그리고 그 변화를 얼마나 재현 가능하게 측정할 수 있는지) 입증하는 데 있다. 논문은 실험 조건을 정교하게 제어해 주입량/주입자 수/이력서 품질 분포를 바꿔가며 순위 변화를 관찰하는 방식으로 효과의 붕괴(조작이 만연할수록 감소)와 비대칭적 역전(낮은 품질이 상위로 진입)을 분리해 보여준다.

- **Empirical Impact**: 결과적으로 prompt injection은 이력서 품질이 균질하고 주입자가 소수일 때는 순위를 꾸준히 끌어올리지만, 주입이 다수로 확산되면 효과가 급격히 줄어들며 붕괴한다. 품질이 이질적인 경우 평균적으로는 덜 효과적이지만, 가끔은 낮은 품질 후보가 높은 품질 후보를 앞지르는 사례가 나타나 공정성 우려를 키운다. 종합하면 LLM 기반 채용 스크리닝은 조작이 ‘드물고’ 후보 간 품질 차이가 ‘작을 때’ 가장 취약하다는 점이 실증적으로 확인됐다.



### Simulation-based inference for rapid Bayesian parameter estimation in epidemiological models: a comparison with MCMC (https://arxiv.org/abs/2606.27286)
- **Prior Approaches**: 기계적 전염병 모델은 감염 동역학을 명시적으로 다뤄 예측과 개입 효과 평가에 널리 활용돼 왔지만, 새로운 데이터가 들어오면 반복적인 베이지안 캘리브레이션이 필요하다. 기존에는 Markov chain Monte Carlo(MCMC)로 사후분포를 샘플링하는 방식이 주류였으나, 고차원·비선형·ODE 기반 시뮬레이터에서는 수천~수백만 번의 가능도 평가가 요구돼 계산 부담이 커 near-real-time 적용이 어렵다. 또한 모의 기반 접근에서도 학습된 사후분포의 정확성을 엄밀히 검증해야 한다는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 mechanistic SECIR 모델에 대해 neural posterior estimation 기반 simulation-based inference(SBI)로 베이지안 캘리브레이션을 수행하고, ICU 점유(독일 2020년)를 대상으로 MCMC와 정면 비교한다. 특히 31일 추론 윈도우뿐 아니라 201일의 다중 전파 변화 시점까지 포함하는 훨씬 어려운 reconstruction 설정에서 SBI가 사후분포 구조와 예측 성능을 유지하는지를 검증한다. 저자들은 SBI가 불확실성을 포함한 파라미터 추정과 ICU 궤적 재현을 동시에 달성하면서도 반복 분석에 필요한 속도를 크게 개선할 수 있음을 보여준다.

- **Technical Challenges**: 주요 기술 과제는 (1) 변화 시점을 포함한 고차원 문제에서 시뮬레이션 설계와 prior가 학습·식별가능성에 미치는 영향, (2) 사후분포의 품질을 단순 예측 적합도 이상으로 평가하는 검증 체계, (3) time-series를 조건으로 하는 신경 사후추정의 아키텍처/학습 안정성이다. 저자들은 Wasserstein distance(1-Wasserstein)와 Kullback-Leibler(KL) divergence로 MCMC와의 사후분포 일치도를 정량화하고 posterior predictive checks와 RMSE로 궤적 재현을 함께 본다. 또한 변화 시점 파라미터에 대해 prior predictive check 기반으로 prior bounds를 조정해 비현실적 전파 궤적이 과도하게 학습을 방해하지 않도록 설계했다.

- **Empirical Impact**: 31일(세 구간)에서는 SBI가 MCMC와 강한 사후분포 일치 및 관측된 ICU 궤적 재현을 보였고, 런타임은 CPU 중심 MCMC가 약 1000초 걸린 반면 SBI는 단일 GPU에서 약 60-70초로 대폭 단축됐다. 201일 reconstruction에서는 SBI가 평균 약 157초로 수행된 반면 MCMC는 1만 9000초 이상이 소요됐으며, 사후분포의 지배적 구조는 유지하되 불확실성은 증가하는 양상을 보였다. 즉, 반복적 near-real-time 추론과 빠른 outbreak 분석을 위한 계산 효율적 베이지안 캘리브레이션 프레임워크로서 SBI의 실무적 가치를 입증했다.



### EO-WM: A Physically Informed World Model for Probabilistic Earth Observation Forecasting (https://arxiv.org/abs/2606.27277)
Comments:
          28 pages, 5 figures, 11 tables

- **Prior Approaches**: 기존 Earth Observation(EO) forecasting은 EarthNet2021 포맷처럼 픽셀 복원·NDVI 시계열 일치 같은 reconstruction 중심 평가가 많습니다. 결정론 모델은 미래를 하나로 고정해 불확실성을 표현하기 어렵고, diffusion 계열도 날씨를 ‘통일된 conditioning’으로만 쓰는 경우가 많아 날씨-반응의 물리적 구조를 충분히 분리하지 못합니다.

- **Core Contribution**: 본 논문은 EO forecasting을 ‘부분 관측된, weather-driven world modeling’으로 재정의하고, 예측이 날씨 변화에 대해 올바르게 반응하는지(용인된 weather-response fidelity)를 핵심 목표로 제시합니다. 이에 video diffusion transformer인 EO-WM을 제안하며, meteorological forcing을 climatological baseline, weather anomaly, cumulative physical stress로 분해해 서로 다른 경로로 주입합니다. 또한 단순 정확도 외에 Extreme Summer Benchmark와 Seasonal Matched-Pair Benchmark를 도입해 ‘극한 악화의 심도’와 ‘대체 날씨에서의 반응 방향·크기’를 진단합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 위성 관측이 희소·불완전해 중간 상태와 불확실성이 크다는 점, (2) 관측되지 않는 토양수분 등 잠재 상태 때문에 동일 날씨라도 결과가 달라질 수 있다는 점입니다. EO-WM은 latent diffusion(EO-VAE + MMDiT)로 다중 가능한 미래를 모델링하고, baseline- anomaly 분해와 누적 stress 인덱스를 설계해 ‘순간적 이탈’과 ‘지속된 열·가뭄 스트레스’가 예측에 반영되도록 합니다. 더불어 anomaly-targeted classifier-free guidance를 통해 이상 신호 민감도를 조절할 수 있게 했습니다.

- **Empirical Impact**: 실험에서 EO-WM은 표준 복원 지표에서는 경쟁력을 유지하면서도, Extreme Summer에서 NDVI decline amplitude 오류를 상대 5.63% 줄이고 directional hit rate을 상대 7.80% 향상시킵니다. Seasonal Matched-Pair에서도 paired trajectory가 날씨 조건 간 상대적 순서와 분리를 더 잘 보존해 forcing-response fidelity가 개선됐음을 보여줍니다. 즉, EO forecasting을 ‘그럴듯한 생성’이 아니라 ‘날씨 변화에 대한 물리적으로 일관된 시뮬레이션’으로 평가·진전시키는 실증적 근거를 제공한 셈입니다.



### Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvemen (https://arxiv.org/abs/2606.27226)
Comments:
          Acceepted to the Second Workshop on Compositional Learning at ICML 2026, Seoul, South Korea

- **Prior Approaches**: 기존 평가는 ROUGE/BLEU 같은 어휘 중복 지표가 의미·사실성을 충분히 반영하지 못하고, BERTScore 등 임베딩 기반 또는 생성형 지표도 한계가 남아 있습니다. LLM-as-Judge 방식은 Likert 점수나 전체 평가를 제공하지만, 근거가 불투명해 디버깅이 어렵고 자극(verbosity/position/self-enhancement) 편향에도 취약하다는 지적이 있습니다. UniEval처럼 다차원 분해를 시도한 방법도 있으나, 학습 기반이거나 단일 yes/no의 굵기 문제로 세밀한 판별력에 제약이 있습니다.

- **Core Contribution**: 이 논문은 BinEval(=BinEval, atomic binary questions 기반 평가)이라는 훈련 없는 프레임워크를 제안합니다. 평가 기준을 원자적인 yes/no 질문들로 분해하고, LLM이 출력마다 각 질문에 독립적으로 답한 뒤 이를 다차원 점수와 종합 점수로 집계해 해석 가능하고 진단 가능한 피드백을 제공합니다. 또한 질문 단위 실패 정보를 이용해 평가기와 생성기 프롬프트를 반복 개선하는 최적화 루프를 함께 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) task prompt를 작업별로 유효한 이진 질문 세트로 자동 분해하고, (2) 질문 답변을 집계해 사람 판단과 잘 맞는 점수 분포/변별력을 유지하는 데 있습니다. 논문은 meta-prompt이 요구사항을 먼저 요약한 뒤 이를 기준으로 이진 질문과 위반 예시를 생성하고, 차원별·전체 점수는 binary verdict의 평균으로 산출하며, 마지막으로 질의/프롬프트를 불일치(동의/반대)에서 얻은 lesson으로 업데이트하는 방식으로 이를 해결합니다. 또한 질문 재생성까지 포함해 업데이트 효과를 키우되, 과도한 재작성으로 성능이 무너지는 현상도 실험에서 확인합니다.

- **Empirical Impact**: BinEval은 SummEval, Topical-Chat, QAGS에서 강한 기준선(예: UniEval, G-Eval)과 비교해 점수 상관을 같거나 능가하며, 특히 QAGS 같은 사실 일관성(환각) 벤치마크에서 강한 성능을 보였습니다. Likert식 전체 판단 대비 점수 분포를 사람과 더 가깝게 맞추고 ceiling effect를 줄여, 애매한 출력과 명백히 나쁜 출력을 더 잘 가르는 것이 관찰됩니다. 더 나아가 SummEval과 IFBench에서 self-update 및 cross-model update 모두에 대해 질문 단위 피드백이 반복 프롬프트 최적화에 실질적으로 활용됨을 보여줍니다.



### Vulnerability of Natural Language Classifiers to Evolutionary Generated Adversarial Tex (https://arxiv.org/abs/2606.27215)
Comments:
          24 pages

- **Prior Approaches**: 기존 NLP 적대적 공격은 주로 토큰(단어/부분어)을 교체해 분류를 오도하는 방식으로 발전해 왔다. 특히 BERT-Attack, BAE, CLARE, A2T처럼 일부 취약 단어를 찾아 문맥·문법 제약을 반영해 교체 후보를 고르는 방법이 주류였지만, 탐색이 국소(local) 또는 단일 경로 중심이라 더 강한 공격을 만들기 어려웠다. 또한 GA 기반 시도들은 black-box에 적합하더라도, 문맥을 얼마나 반영해 교체를 생성하느냐가 성능 격차의 핵심으로 지적된다.

- **Core Contribution**: 이 논문은 black-box로도 동작하면서 GloVe의 단어 유사도를 활용해 교체 연산(변이)을 문맥적으로 그럴듯하게 유도하는 GA 기반 하이브리드 공격인 GAversary를 제안한다. 핵심은 유전 알고리즘이 탐색(조합·변이)을 수행하되, mutation operator에서 GloVe 임베딩과 주변 문맥 창을 이용해 의미적으로 가까운 단어를 후보로 고르는 방식이다. 목표 모델 내부 접근 없이 logit 출력만으로 검색을 안내해 다양한 사전학습 모델을 동시에 타깃팅한다.

- **Technical Challenges**: GAversary가 풀어야 할 기술적 난제는 (1) black-box 환경에서 분류 오도를 빠르게 달성하는 탐색 효율, (2) 단어 교체 수를 최소화하면서 의미 유사도를 유지하는 문장 충실도, (3) 국소 최적점에 갇히지 않는 조합 탐색이다. 저자들은 해(solution)를 ‘(위치, 대체 단어) 변경 집합’으로 압축 표현해 계산량을 줄이고, fitness를 오분류에 가까워지는 정도로 설계한 뒤 fitness proportionate selection과 uniform crossover로 세대를 진행한다. 변이 단계에서는 NLTK stop-words를 제외하고, 타깃 단어 주변 문맥 창을 마스킹한 뒤 counter-fitted GloVe의 최근접 이웃을 후보로 두고 “이미 적용된 교체들을 포함했을 때” 오분류를 더 멀리 밀어내는 단어를 선택해 문맥 적합성을 높였다.

- **Empirical Impact**: MR과 AG-News 벤치마크에서 WordCNN, WordLSTM, BERT 등 여러 모델에 대해 GAversary는 다른 대표 방법(B AE, A2T)보다 더 큰 정확도 감소를 보였고, 최선의 경우 76.8%에서 5.8%로까지 떨어뜨렸다. 다만 평균적으로 교란 단어 수가 약 2배 수준이며, 의미 유사도는 약간 낮고 모델 쿼리 수는 BAE 대비 약 2배, A2T 대비 6–8배 증가한다. 그럼에도 wall-clock 시간은 다른 방법 대비 큰 폭으로 늘지 않아(예: BERT에서 약 5% 증가) 취약성 노출이 최우선인 보안 평가·오프라인 견고성 테스트 환경에서 특히 유용하다는 점이 실증적으로 뒷받침된다.



### A Process Harness for Uplifting Legacy Workflows to Agentic BPM: Design and Realization in CUGA FLO (https://arxiv.org/abs/2606.27188)
Comments:
          24 pages, 5 figures

- **Prior Approaches**: 기존 BPMN 기반 워크플로 엔진은 구조적 준수(structural compliance)를 강제하지만, 미리 모델링되지 않은 상황에서는 에러·에스컬레이션 외에 선택지가 거의 없다. 또한 AI를 작업 단위에 얹는 방식은 제어 흐름을 고정시키고, LLM을 플래너로 쓰는 방식은 실행이 모델·감사·컴플라이언스에 맞는지 보장하기 어렵다.

- **Core Contribution**: 이 논문은 레거시 워크플로 엔진을 교체하지 않고도 Agentic BPM으로 “업리프트”할 수 있는 process harness를 제안한다. 엔진은 구조 권한을 유지한 채, 지정된 control point에서 policy-governed agentic layer가 추론·적응·감독을 덧씌우며 개입한다.

- **Technical Challenges**: 핵심은 (1) LLM 추론을 프로세스 실시간 문맥에 완전히 grounded 하게 제공하고, (2) 허용된 개입만 하도록 정책으로 프레이밍하며, (3) 어떤 형태의 구조 변경이 가능한지까지 감사 가능하게 제한하는 것이다. 이를 위해 Task-Decision-Flow(TDF) 모델로 추론을 TaskAgent(지식집약 작업), DecisionAgent(게이트웨이 라우팅), FlowAgent(훅 기반 런타임 플로우 적응)로 분해하고, FRAME 정책 세트로 모든 LLM 호출을 묶으며, FlowAgent 개입에는 per-process access control(ϕ)까지 적용한다.

- **Empirical Impact**: 구현체로 CUGA FLO를 제시하고, 대출 승인 워크플로 예시에서 세 에이전트 유형과 regulatory override 훅을 함께 실행해 동작을 확인한다. 또한 희귀한 케이스의 long tail을 정책 기반 훅 추론으로 흡수해 프로세스 모델 수정 없이 점진적으로(되돌릴 수 있게) Agentic BPM 범위를 확장할 수 있음을 보여준다.



### TOPS: First-Principles Visual Token Pruning via Constructing Token Optimal Preservation Sets for Efficient MLLM Inferenc (https://arxiv.org/abs/2606.27161)
Comments:
          27 pages, 18 figures

- **Prior Approaches**: 기존 MLLM의 시각 토큰 프루닝은 주로 어텐션 기반(중복 토큰을 비슷하게 유지)·다양성 기반(지시문과의 정합성 부족)·여러 휴리스틱을 결합한 방식으로 나뉜다. 하지만 많은 방법이 “왜 이 점수 기준이 최적 부분집합을 보장하는가”에 대한 원리적 근거 없이, 토큰을 점수화해 랭킹하는 데 머문다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 시각 토큰 프루닝을 정보이론 관점에서 “Token Optimal Preservation Sets”를 구성하는 최적 부분집합 선택 문제로 재정의한다. 그 결과 Task Relevance(과업 관련성), Information Coverage(정보 커버리지), Semantic Diversity(의미 다양성)라는 세 가지 원칙을 도출하고, 이를 만족하는 training-free·model-agnostic 프루닝 모듈 TOPS를 제안한다.

- **Technical Challenges**: TOPS는 각 후보 토큰의 과업 관련성(텍스트-레이더 어텐션 기반 프록시), 원본 시각 토큰에 대한 정보 커버리지, 현재 선택 집합 대비 중복을 줄이는 의미 다양성을 함께 계산해 균형 있게 그리디로 토큰을 확장한다. 또한 Stage I은 멀티모달 프로젝터 이후 CLS 어텐션으로 거친 중복을 줄이고, Stage II는 LLM 내부 여러 레이어에서 텍스트-레이더 어텐션으로 텍스트-인식 정제를 수행해 공격적인 압축에서도 핵심 근거를 보존하도록 설계했다.

- **Empirical Impact**: 7개 MLLM 백본과 14개 벤치마크에서 TOPS는 다양한 프루닝 비율 조건에서 기존 방법을 일관되게 능가한다. 특히 LLaVA-NeXT에서 77.8%의 시각 토큰을 제거하면서 7B는 100.0%, 13B는 100.6% 성능을 보이며, 일부 경우 프루닝이 환각을 완화하고 더 가벼운 MLLM 설계로 이어질 수 있음을 시사한다. 또한 FLOPs·지연시간·GPU 메모리 관점에서도 효율 개선이 관측되어, “성능 저하 없는 경량화”를 실사용 지표로 뒷받침하는 점이 의미가 크다.



### OpenRCA 2.0: From Outcome Labels to Causal Process Supervision (https://arxiv.org/abs/2606.27154)
Comments:
          work in progress

- **Prior Approaches**: 기존 RCA(원인 분석) 벤치마크는 fault injection으로 깨진 컴포넌트를 라벨로 두고, 평가도 “결과(outcome)”인 루트 원인 컴포넌트 맞힘 여부에 집중하는 경향이 강합니다. 이 방식은 모델이 관측된 증상에서 원인을 “추론한 경로(derivation)”를 검증하지 못해, 겉보기 정답이더라도 인과 전파 연결이 없는 상태를 놓치기 쉽습니다. 그 결과, RCA 에이전트가 실제로 무엇을 근거로 답했는지(프로세스 수준) 측정이 어려웠습니다.

- **Core Contribution**: 이 논문은 PAVE(Path Annotation via Verified Effects)라는 단계별 라벨링 프로토콜을 제안해, fault injection 기록과 시스템 의존 그래프를 바탕으로 “원인→증상”까지의 인과 전파 경로를 검증된 형태로 재구성합니다. 이를 통해 root cause만이 아니라 propagation path 자체를 step-wise causal ground truth로 제공하는 OpenRCA 2.0(500 instances)을 구축합니다. 또한 outcome-only 평가가 만드는 맹점인 ungrounded diagnosis(정답 서비스는 맞지만 검증된 인과 경로로 접지되지 않음)를 정량화합니다.

- **Technical Challenges**: 핵심 기술적 난관은 관측 텔레메트리만으로는 역추론이 본질적으로 잘 정의되지 않아, 라벨러가 backward 추론을 해야 한다는 점입니다. PAVE는 이 비대칭을 극복하기 위해 intervention d​o(v_root)를 “이미 주어진 원인”으로 활용해, 후보 경로를 구조적 규칙(구조), 주입 전 기준선 대비 통계적 일탈(통계), 업스트림-다운스트림 타이밍 정렬(시간)로 동시에 통과시키는 forward verification 파이프라인을 설계합니다. 후보 경로는 time-expanded 그래프에서 constrained DFS로 생성하고, 각 엣지는 기준선 텔레메트리와의 분포 검정을 통해 end-to-end로 “사실상 살아남은” 전파만 남깁니다.

- **Empirical Impact**: 11개 frontier LLM에 대해 OpenRCA 2.0을 평가한 결과, 정확한 root-cause set 복구는 평균 20.7%에 그쳤습니다. 더 중요한 차이는 outcome에서 AnySvc가 76.0%로 높아도, verified causal propagation path로 접지되는 PR은 61.5%뿐이라 약 1/5의 사례가 ungrounded diagnosis로 드러난다는 점입니다. 또한 process 레이어에서 Node F1 대비 Edge F1이 일관되게 더 낮아(평균 18.8%p 격차) 서비스 “참여”는 맞춰도 “방향성 있는 전파 관계” 추론이 특히 취약함을 보여, step-wise causal ground truth의 필요성을 실증합니다.



### Joint Learning of Experiential Rules and Policies for Large Language Model Agents (https://arxiv.org/abs/2606.27136)
- **Prior Approaches**: 기존 연구는 상호작용 경험을 (1) 자연어 규칙·리플렉션 같은 형태로 외부 메모리에 저장해 프롬프트에 넣거나, (2) 궤적(trajectory)과 보상 피드백으로 파라미터를 업데이트해 정책을 개선하는 방식으로 나눠 활용해 왔다. 전자는 해석 가능하고 국소 실수를 빠르게 줄일 수 있지만, 학습으로 정책이 변하면 규칙이 쉽게 어긋나며 sparse-reward 환경에서는 로컬 오류의 시정이 늦어질 수 있다. 후자는 전반적인 성능 향상엔 유리하지만, 명시적으로 규칙 풀을 유지·수정해 재사용하는 구조는 제공하지 않아 국소적으로 잘못된 판단을 겨냥한 교정이 제한된다.

- **Core Contribution**: JERP(Joint Learning of Experiential Rules and Policies for LLM Agents)는 같은 상호작용 궤적을 “규칙 풀 업데이트”와 “정책 최적화”에 동시에 쓰도록 결합한 학습 프레임워크다. 에피소드마다 장기 experiential-rule pool에서 관련 규칙을 불러와 에이전트 의사결정을 돕고, 에피소드 종료 후에는 수집된 궤적을 (i) 정책 파라미터를 위한 집단 상대 비교 업데이트와 (ii) 성공 기준 궤적과의 대조를 통한 규칙 풀 수정에 재사용한다. 이 결합을 통해 규칙이 학습 중인 정책 변화와 정렬(alignment)되면서, 안정적인 행동은 점진적으로 모델 내부로 흡수되도록 설계했다.

- **Technical Challenges**: 핵심 기술적 난제는 정책이 바뀌는 와중에도 규칙이 “지속적으로 유효한 형태”로 유지되도록, 궤적 기반 신호를 규칙 텍스트 편집과 효율적인 규칙 검색/선택으로 변환하는 것이다. JERP는 작업별 장기 rule pool을 utility score로 관리하며, 매 에피소드에서는 score 상위 kk개 규칙만 고정 템플릿에 삽입해 컨텍스트 요동을 줄인다. 또한 규칙 업데이트는 수치 최적화가 아니라 contrastive reflection 기반의 구조화된 ADD/EDIT/UPVOTE/DOWNVOTE/MERGE 연산(그리고 용량 초과 시 제거·통합)을 통해 파싱 가능하게 수행해 규칙 내용과 유틸리티를 함께 갱신한다.

- **Empirical Impact**: 실험은 AlfWorld와 WebShop 두 멀티스텝 상호작용 벤치마크에서 수행됐고, JERP는 Vanilla LLM, ReAct, Reflexion, RLOO, GRPO 대비 전반적으로 가장 높은 성능을 보였다. AlfWorld에서는 전체 success rate 61.5%로 GRPO(57.8%)와 RLOO(48.7%)를 상회했고, WebShop에서도 평균 점수 79.0과 success rate 64.1%로 최상위를 기록했다. 특히 더 제약이 강하고 연산 시퀀스가 긴 범주에서(예: Clean/Heat/Cool/Pick2) 중후반 학습 단계의 개선이 두드러져, “궤적을 규칙과 정책에 함께 쓰는 결합”이 실제 의사결정 성능 향상으로 이어짐을 보여준다.



### How to evaluate clustering with ground truth? (https://arxiv.org/abs/2606.27061)
Comments:
          Preprint of a book chapter to appear: P. Fränti, "How to evaluate clustering with ground truth?", In Center-based clustering, Springer Nature, 2026

- **Prior Approaches**: 기존 클러스터 평가는 정답 레이블(ground truth)이 있을 때 external validity index로 성능을 평가해 왔다. 특히 set-matching 기반 지표들이 많이 쓰였지만, 어떤 지표가 어떤 특성을 반영하는지(군집 크기 편향 여부, 점 단위 가중 방식 등) 선택 기준이 분명하지 않았다.

- **Core Contribution**: 이 논문은 set-matching 기반 external validity index들을 중심으로 대표 지표들을 정리하고, 상황별로 무엇을 선택해야 하는지 가이드를 제안한다. 직관적인 군집 수준 요약값이 필요할 때는 centroid index (CI)를 추천하며, 설명 가능한 결과를 제공한다.

- **Technical Challenges**: 핵심은 지표가 군집 크기(cluster size)나 점 수준의 중요도에 의해 얼마나 편향될 수 있는지 조율하는 데 있다. 논문은 더 세밀한 점 단위 평가가 필요하면 pair-set index (PSI)를 통해 정규화된 점수로 군집 크기 편향을 줄일 수 있고, 모든 점을 동일하게 중요하게 본다면 clustering accuracy (ACC) 등 set-matching 계열을 선택하라고 정리한다.

- **Empirical Impact**: 실험적 성능 비교 자체보다는, 어떤 데이터 평가 요구(군집 수준 vs 점 수준, 크기 편향 허용 여부)에 어떤 지표가 맞는지 명확한 선택 프레임을 제공하는 데 의미가 있다. 이로써 연구자들이 외부 지표를 임의로 고르는 부담을 줄이고, 결과 해석의 일관성을 높일 것으로 기대된다.



### Semantic Early-Stopping for Iterative LLM Agent Loops (https://arxiv.org/abs/2606.27009)
Comments:
          7 pages, 5 figures, 4 tables. Open implementation, machine-checked theory, and reproducible harness: this http URL

- **Prior Approaches**: 기존 Writer→Critic 같은 멀티에이전트 LLM 루프는 max_iterations 같은 정수 카운터로 종료돼, 내용이 여전히 좋아지는지(개선 여부)를 보지 못합니다. 그래서 쉬운 입력에선 토큰을 낭비하고, 어려운 입력에선 너무 일찍 끊겨 품질이 손상될 수 있습니다. 또한 “어떤 라운드가 가장 좋은지”보다는 “몇 번째에 멈췄는지”에 최적화되는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 고정 iteration cap을 없애고, 연속 초안의 의미 변화(코사인 거리)가 일정 임계값 아래로 k번 유지되고, 동시에 품질 지표 Information Score가 더 이상 개선되지 않을 때 종료하는 semantic early-stopping을 제안합니다. 다만 이전 버전에서 주장했던 Banach contraction 같은 과도한 수학적 보장은 지키지 않고, 실제로 증명·머신체크 가능한 termination(종료 보장)과 well-definedness만을 전면에 둡니다. 결론적으로 “언제 멈출지(when)”를 “품질과 의미 수렴을 함께 확인해 멈추기”로 바꾸면서, 필요하면 judge 비용까지 분리해 설계합니다.

- **Technical Challenges**: 핵심 난제는 두 가지였습니다: (1) LLM 생성이 결정론적이지 않기 때문에 수렴을 항상 보장하는 강한 contraction 가정을 세우기 어렵고, (2) 의미 변화나 품질 신호를 계산하는 과정에서 judge 호출 같은 평가 비용이 숨게 섞일 수 있다는 점입니다. 저자들은 failsafe 기반의 우선순위 캐스케이드로 최대 라운드 내 종료를 증명하고(증명은 코드로 재검증 가능), judge-efficient 프로토콜로 매 질문의 전체 궤적을 한 번 생성해 정책별로 재생(replay)·캐싱하며 operational tokens(정책이 쓰는 토큰)과 evaluation tokens(측정용 토큰)를 분리해 공정 비교를 가능하게 했습니다.

- **Empirical Impact**: HotpotQA 다중홉 RAGAS 기반 실험에서 judge-free entropy_only 스토퍼는 max_iterations 대비 operational tokens를 38% 줄이면서도 품질은 parity 수준으로 유지했습니다(Delta-IS=-0.004, p=0.81). 반면 라운드마다 judge를 호출해 품질 개선을 게이팅하는 full SHP는 비용만 크게 늘려(+129%) 역효과였고, 이로써 “온라인 품질 게이팅은 값이 비쌀 수 있다”는 실측 메시지가 강화됩니다. 또한 각 라운드 중 최적 라운드를 고르는 oracle은 모든 실용 정책보다 IS를 +0.115 끌어올려, 향후의 열린 문제를 ‘best-round identification’으로 명확히 재정의했습니다.



### Adaptive Utility driven Resource Orchestration for Resilient AI (AURORA-AI) (https://arxiv.org/abs/2606.27005)
Comments:
          Accepted at IEEE Research and Technologies for Society and Industry 2026 conference

- **Prior Approaches**: 기존 연구는 정적(Static)·순환(Round Robin)·탐욕(Greedy)·컨텍스트 밴딧(LinUCB)·PPO 같은 방식으로 자원 배분을 하되, 비정상 환경에서 성능 저하와 함께 공정성·설명가능성 같은 인간 중심 지표가 함께 흔들릴 수 있다는 한계가 컸습니다. 또한 black-swan 같은 극단 충격에 대해 “빠른 회복”과 “안정성(안정적 운용 단계 비율)”을 이론적으로 연결해 검증한 프레임워크는 상대적으로 부족했습니다.

- **Core Contribution**: AURORA-AI는 Hamilton-Jacobi-Bellman(HJB) 피드백 제어로 자원(컴퓨팅 예산)을 모델들에 폐루프(closed-loop)로 재분배하면서, Lyapunov 기반 안정성 모니터링과 공정성(fairness) 친화적 합성 유틸리티를 하나로 통합합니다. 합성 유틸리티는 예측 성능뿐 아니라 demographic parity, 비용·지연, 강건성, interpretability까지 함께 최적화해 공정성을 사후 제약이 아니라 목표로 다룹니다. 결과적으로 교란이 생겨도 전역 유틸리티가 최대에 가깝게 유지되는 “resilient human-centric AI deployment” 방향을 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 비정상성(개념 drift)과 극단 충격(black-swan) 속에서, 편향·망각·학습 업데이트의 수치 안정성까지 포함해 안정적 제어를 설계하는 것입니다. 논문은 Markov decision process/연속시간 동역학을 두고 HJB로 최적 피드백 정책을 얻으며, Lyapunov 함수의 부호 조건과 drift/ergodicity 조건으로 장기 보상을 흔들림 없이 계산할 수 있게 했습니다. 또 Fisher 정보 기반 업데이트에서 특이성 문제가 생길 수 있어 Tikhonov 정규화로 안정적인 역행렬 연산을 보장하고, bias-decay 법칙으로 공정성 편향의 해소 동역학을 추가했습니다.

- **Empirical Impact**: 스트레스가 풍부한 이산시간 시뮬레이션에서 AURORA-AI는 black-swan 충격 이후 회복 시간이 Static(88 steps) 대비 크게 단축되고 PPO(22 steps)보다도 빠르게(해상도 내 즉시 회복) 동작했습니다. 또한 α-quantile와 super-quantile(꼬리 위험 지표)을 각각 29%·25% 끌어올리고, demographic parity gap의 평균·최댓값을 동시에 낮추며 Lyapunov-stable operating steps 비율을 43.27%에서 46.99%로 늘렸습니다. 비교 실험 전반에서 기존 컨트롤러들이 tail-risk와 안정성을 동시에 만족시키기 어려웠던 점을, AURORA-AI는 Lyapunov 안정성과 공정성·설명가능성까지 함께 개선하는 방식으로 보여준 것이 의미가 큽니다.



### Einstein World Models (https://arxiv.org/abs/2606.26969)
Comments:
          12 pages (9 without references), 2 figures, 1 algorithm

- **Prior Approaches**: 기존 Chain-of-thought는 중간 추론을 언어로만 외부화해, 물체 식별·접촉·열·운동처럼 장면 수준 변수가 중요한 문제에서 한계가 나타난다. 또한 world model 계열은 주로 관측 기반 예측이나 행동 조건부 예측에 머물러, 경험 밖의 반사실적 thought experiment를 정밀하게 다루기 어렵다. 비슷하게 Visualization-of-Thought·whiteboard-of-thought 류는 시각 메모/그림을 중간 산출물로 쓰지만, 이를 비디오 롤아웃 형태로 ‘가설을 검토’하는 통합 메커니즘은 상대적으로 약하다.

- **Core Contribution**: Einstein World Models(EWMs)는 LLM이 스스로 판단해 world-module로부터 짧은 visual-temporal rollout(비디오 롤아웃)을 생성·조회하고, 그 결과를 추론 trace에 ‘검증 가능한 가설’로 포함하도록 하는 구조를 제안한다. 핵심은 롤아웃을 정답 대신 참고할 inspectable hypothesis로 취급해, 텍스트만으로 어려운 장면 전개·반사실적 사건 상상을 추론에 보조하는 것이다. 또한 “reasoner(추론자)”와 “world-module(롤아웃 생성기)”을 분리해, 생성된 비디오를 외부에서 관찰·디버깅할 수 있게 만든다.

- **Technical Challenges**: 어떤 질문에서 언제 비디오 thought experiment가 이득인지, 그리고 어떤 world-module query로 롤아웃을 뽑아야 이후 추론이 강화되는지를 학습해야 한다. 저자들은 SFT로 EWM trace 포맷(언어·tool_call·visual_rollout·정답)을 익힌 뒤, verifiable final answer에 기반한 RLVR/GRPO 스타일 학습으로 world-module 호출의 선택성을 최적화한다. 더해 world-module 자체의 신뢰도 문제를 “물리적 그럴듯함”으로 직접 보정하기보다, 롤아웃의 inspectability/faithfulness가 추론에 실제로 기여하는지와 모듈 선택(예: diffusion 기반 렌더러 품질, 상이한 편향의 앙상블)을 함께 다루는 방향을 제시한다.

- **Empirical Impact**: 이 논문은 실험 결과가 아니라 먼저 학습 가능한 시각 thought experimentation의 데이터·학습 청사진을 제시하는 데 초점을 둔다. 특히 기존 공개 데이터가 ‘문제의 시작이 텍스트이며 모델이 스스로 시각화 여부를 결정해야 하는’ 설정을 충분히 제공하지 못한다고 진단하며, SimpleBench 같은 제한적 예시를 보완할 datasets 공모를 제안한다. EWMs가 성공하면 LLM이 단순 텍스트 기호를 넘어 외부화된 visual walk-through로 추론을 확장하고, 중간 가설을 공유·점검하는 디버깅 경로를 열 수 있다는 점에서 영향력이 크다는 입장이다.



### Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds (https://arxiv.org/abs/2606.26964)
Comments:
          25 pages, 17 figures

- **Prior Approaches**: 기존 카메라 제어·궤적 계획은 충돌 회피와 부드러운 경로 같은 물리/기하 제약을 잘 다루지만, 관측 목표가 ‘이미 정해진’ 가정이 많았다. 또한 3D cinematography나 generative video 계열은 주로 trajectory나 비디오 합성 자체에 초점을 두어, 내러티브 의도로부터 무엇을 봐야 하는지를 물리 실행 가능한 형태로 조직하는 문제는 상대적으로 비어 있었다. 결과적으로 동적 3D 환경에서 narrative intent와 가시성·충돌·시간적 일관성을 함께 만족시키는 폐루프형 관측 설계가 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 카메라가 단순 센서가 아니라 ‘내러티브에 근거해 무엇을 관측할지’를 결정하는 Narrative-Grounded World Visual Attention을 카메라 플래닝 관점에서 정식화한다. 이를 위해 Look-Before-Move 프레임워크를 제안하며, 관측 명세를 먼저 세우고(관측 contract) 그다음에 viewpoint 탐색과 trajectory grounding을 수행하도록 단계 분해한다. 또한 동적 3D Story World Benchmark를 구축해 subject perception, intent consistency, trajectory quality를 동시에 평가할 수 있게 했다.

- **Technical Challenges**: 핵심 기술 난제는 내러티브 지시문을 실행 가능한 시각 제약으로 변환하고, 그 제약을 만족하면서도 3D 물리 제약(가림, 충돌, 비현실적 카메라 배치) 하에서 viewpoint를 찾는 것이다. 논문은 Semantic Observation Contract로 directorial intent를 대상/관계/가시조건/구도 선호 같은 구체 제약으로 바꾸고, Monte Carlo Viewpoint Search로 narrative-compliant이면서 geometrically feasible한 후보를 대량 탐색한 뒤 tournament reranking과 render 기반 reflection으로 미세 조정한다. 마지막으로 Semantic Trajectory Grounding에서 선택된 viewpoint들을 동적 배우를 추적하는 연속 궤적으로 연결하고, closed-loop 검증과 temporal semantic editing으로 시간적 의미 일관성을 맞춘다.

- **Empirical Impact**: 실험은 Look-Before-Move가 대표 baseline 대비 전체 점수와 세 평가 축(SP, IC, TQ)에서 일관된 개선을 보였다고 보고한다. 정성 결과에서도 암살자 장면처럼 의도된 주체를 정확히 클로즈업으로 배치하고 액션 cue를 보존하는 식의 ‘내러티브 특화 관측’이 나타나며, 차량/병원/침실 장면에서도 상호작용과 서사 초점이 안정적으로 유지된다. 전반적으로 이 연구는 dynamic 3D 세계의 카메라 플래닝이 trajectory 생성에 바로 뛰어들기보다 ‘관측을 먼저 조직’하는 접근이 성능과 실행 가능성을 좌우한다는 점을 실증적으로 강화한다.



### Where Do CoT Training Gains Land in LLM based Agents? (https://arxiv.org/abs/2606.26935)
- **Prior Approaches**: 언어모델 에이전트에서 CoT(chain-of-thought) 생성 후 행동을 내는 방식이 널리 쓰이지만, verbalized CoT가 실제 계산을 충실히 반영하지 않을 수 있다는 ‘사후 합리화’ 문제(비충실성)가 지적돼 왔다. 또 CoT를 학습해도 성능 향상이 추론 기반 행동 수정 능력의 향상인지, 아니면 프롬프트만으로 다음 행동을 더 잘 예측하는 ‘지름길(shortcut)’ 강화인지 분리가 어려웠다.

- **Core Contribution**: 이 논문은 CoT 없이 프롬프트만으로 행동을 예측하는 prompt actions과, CoT를 생성한 뒤 행동을 예측하는 CoT actions를 체크포인트 전반에서 비교해 훈련 개선의 원인을 진단한다. 핵심 결론은 CoT 훈련이 ‘생성된 reasoning으로 행동을 더 자주 뒤집는 능력’만 키우는 것이 아니라, 프롬프트만으로도 최종 행동을 더 잘 맞히게 만드는 쪽도 함께 강화한다는 점이다. 또한 나중 체크포인트일수록 충돌하는 reasoning trace에도 행동이 프롬프트에 더 강하게 고정(anchored)되는 경향을 보인다.

- **Technical Challenges**: 문제는 최종 성공률만 보면 reasoning의 기여를 분해할 수 없다는 점이다. 연구진은 오프라인의 단계별 비교, 온라인의 미지 작업 평가, 그리고 conflicting-trace 테스트(추론을 랜덤 대체해 어떤 근거를 따르는지)를 결합해 행동이 어느 정보원(프롬프트 vs reasoning)에 의해 더 좌우되는지 추적한다. 더 나아가 어텐션과 그래디언트 분석으로 에이전트의 긴 프롬프트가 action 생성에서 더 큰 비중을 차지해 프롬프트 경로가 최적화 신호를 더 받는 구조적 이유를 제시한다.

- **Empirical Impact**: 실험에서는 prompt-action 품질이 훈련 중 크게 개선되지만, CoT action이 prompt action 대비 갖는 상대적 우위는 시간이 지나도 크게 벌어지지 않는다. online 평가에서도 CoT vs prompt 간 ‘gap’이 대체로 평평하게 유지되고, 이후 모델들은 충돌하는 reasoning trace에 덜 민감해져 prompt-anchoring이 강화됨이 확인된다. 이를 근거로 학습 데이터 일부에서 최종 action 토큰에 대한 supervision을 마스킹하는 reduced action-supervision을 제안하며, 이 개입이 OOD(out-of-domain) 일반화 성능을 전반적으로 개선하는 실증 결과를 제시한다.



### Diagnosing Task Insensitivity in Language Agents (https://arxiv.org/abs/2606.26918)
- **Prior Approaches**: 기존 연구는 OOD 일반화를 주로 최종 성공률 관점에서 평가하며, 학습이 끝난 뒤 분포 이동에서 성능이 떨어지는 현상을 다뤄왔다. 또 최근 에이전트 학습 동역학 분석은 최적화가 강화돼도 취약한 휴리스틱(지름길) 의존이 남을 수 있음을 지적한다. 다만 “행동이 실제로는 지시문에 조건부로 근거했는가”를 학습 과정에서 직접 추적한 진단은 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 OOD 실패의 핵심 원인으로 task insensitivity(과업 비민감성)를 제시한다. 즉 지시가 의미적으로 망가졌거나 유사하지만 다른 과업으로 바뀌어도, 모델이 훈련 때 익힌 원래 과업의 행동 패턴을 계속 재사용하는 경향이 강하다는 점을 보여준다. 이를 행동 일관성 문제를 넘어, 지시문 토큰에 대한 의존 자체가 약해지는 현상으로 정식화한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 “지시문에 대한 행동 의존”을 학습 목표로 명시적으로 강제하는 방법을 찾는 것이다. 저자들은 지시를 비슷하지만 다른 과업으로 대체했을 때 원래 정답 행동이 덜 나올수록 좋다는 아이디어를 대비학습 형태로 설계하되, 단순 NLL 격차를 무작정 키우면 perturbed 케이스까지 과도하게 벌점이 커질 수 있음을 해결한다. 이를 위해 Task-Perturbed NLL Optimization이라는 가벼운 contrastive regularizer로, 고정된 reference 모델이 측정한 “태스크 교체 시 분리” 수준을 기준으로 현재 모델의 분리를 부족할 때만 패널티를 주는 방식으로 구현했다.

- **Empirical Impact**: ALFWorld, ScienceWorld, WebShop에서 통제된 train/test task split 실험을 통해 제안 방법이 OOD 성능을 다수 설정에서 개선함을 보였다. 특히 로컬 관측은 비슷하지만 정답 행동이 지시문에 민감한 태스크 쌍에서 효과가 가장 일관적이었고, GRPO 기반 학습에서도 동일한 방향의 개선이 관찰됐다. 동시에 학습이 진행될수록 나타나던 attention drift(지시 토큰 주의 저하)를 완화하고, 과업 교체 뒤 행동 일관성의 상승을 억제해 개입이 “새 능력 추가”가 아니라 “지시 의존 강화”에 직접 기여함을 뒷받침한다.



### Learning to Recover Task Experts from a Multi-Task Merged Mod (https://arxiv.org/abs/2606.26902)
- **Prior Approaches**: Multi-task model merging은 여러 task-specific expert를 하나로 합치는 데 초점을 두지만, static merging은 파라미터 간 간섭(parameter interference) 때문에 성능이 흔들리는 문제가 크다. dynamic merging은 이를 줄이려 하지만, 추론 시 redundant expert를 저장·로딩하는 비용이 부담으로 남는 경우가 많다. 또한 task identity를 알아야 올바른 expert를 선택할 수 있어, unseen task 일반화에 한계가 지적돼 왔다.

- **Core Contribution**: 논문은 파라미터 간섭을 “병합 과정에서 각 expert에 가해지는 perturbation”으로 재정의하고, 이를 affine transformation으로 모델링한 뒤 additive offsets로 근사한다. 이를 바탕으로 Recover Task eXpert(ReTeX)는 단 하나의 merged checkpoint에서 해당 offsets를 예측해 간섭을 되돌리고 task-expert 성능을 복구하는 프레임워크를 제안한다. 더불어 task identity가 불확실할 때를 위해 router-free task identifier를 SVD subspace signature로 구성해, 입력에 대해 가장 작은 projection residual을 갖는 subspace를 선택한다.

- **Technical Challenges**: 핵심 난제는 merged 과정이 유발하는 파라미터 간섭을 정확히 “offset” 형태로 예측해, 실제 expert 성능을 복원할 수 있을지다. ReTeX는 offset 예측을 통해 간섭을 역보정하고, task를 특정하기 위해 별도 라우터 없이 SVD 기반 subspace signature를 offline으로 계산해 추론 단계의 식별을 단순화한다. 추가로 OOD task에서는 offset 예측이 expert 지식을 emergent하게 adaptive interpolation하도록 유도해, 본질적으로 보지 못한 분포에서도 동작하도록 설계된다.

- **Empirical Impact**: ReTeX는 vision과 NLP 두 영역에서 individual expert 대비 95% 이상 성능 복구를 달성하며, unseen tasks에 대한 generalization도 유의미하게 개선한다. 특히 out-of-distribution(OOD) 상황에서 adaptive interpolation 효과가 나타나, 단순 복원 단계를 넘어 동적 지식 보간을 수행한다는 점이 실증적으로 확인된다. 단일 merged checkpoint로 성능을 끌어올리면서 inference 비용 부담을 줄일 수 있어, multi-task merging의 실용성에 직접적인 의미가 있다.



### Generative Retrieval via Diffusion Transformer with Metric-Ordered Sequence Training and Hybrid-Policy Preference Optimization (https://arxiv.org/abs/2606.26899)
- **Prior Approaches**: 임베딩 기반 검색은 쿼리와 유사한 항목을 높은 점수 순으로 반환하지만, 실제 운영에서는 시드가 표현하는 “세밀한 패턴” 안에 머물면서 특정 속성을 만족하는 항목을 추가로 찾아야 한다. 기존 방식은 시드 임베딩을 단순 평균내면 패턴은 유지되지만 속성 밀도가 낮아지고, 속성만 최적화하면 다른 패턴으로 드리프트되어 리뷰 큐를 오염시키는 문제가 있었다.

- **Core Contribution**: 이 논문은 패턴을 보존하면서 속성을 높이는 검색을 pattern-preserving attribute retrieval로 정식화하고, 속성 성공과 패턴 이탈을 함께 반영하는 intersection metric(Joint@K)을 제안한다. 이를 위해 continuous generative retrieval 방식으로, 시드 항목 임베딩 시퀀스를 입력받아 쿼리 임베딩을 연속적으로 생성한 뒤 ANN 검색으로 K개를 찾는 구조를 사용한다. 또한 MO-DiT+HPPO라는 staged 프레임워크로 CPT(continuation pretraining), 꼬리-중심 SFT(tail-centroid fine-tuning), 그리고 온라인 목표에 맞춘 HPPO를 결합한다.

- **Technical Challenges**: 핵심 기술 과제는 “어떤 감독 신호가 생성 쿼리를 높은 속성-밀도 영역으로 이동시키면서도 동일 패턴을 유지하게 하는가”였다. 이를 위해 온라인 recall density를 직접 레이블링하되 비용을 줄이기 위해 frozen 임베딩으로 density predictor를 학습해 같은 패턴 클러스터 내부에서 낮은 밀도→높은 밀도 순의 metric-ordered trajectories를 구성했다. 마지막 정렬 단계에서는 candidate pool을 hybrid policy로 구성하고, 실제 온라인 Joint@K로 레이블한 뒤 reference-anchored preference optimization을 수행하되, Pareto pair filter로 same-pattern purity가 악화되지 않는 페어만 선택해 속성 개선이 패턴 드리프트로 “사기”치는 것을 막았다.

- **Empirical Impact**: 아이템-홀드아웃 및 패턴-홀드아웃 프로토콜 하에 4개 속성 도메인에서, metric-ordered DiT가 사전학습된 generative retriever 대비 intersection metric을 개선했고 HPPO가 추가로 성능을 끌어올렸다. 특히 8개 도메인-스플릿 셀 중 7개에서 유의미한 향상을 보였고 가장 어려운 분할에서도 근소한 동률 성과를 기록했다. 검증 및 애블레이션(메트릭-예측기 검증, 순서 관련 제거, CPT/SFT 및 preference·candidate policy 변형)으로 향상 원인이 metric-ordered 학습 방향성과 온라인 Joint@K 정렬, 그리고 Pareto pair filter의 균형 제어에 있음을 확인했다.



### A Pipeline for Generating Longitudinal Synthetic Clinical Notes Using Large Language Models (https://arxiv.org/abs/2606.26879)
- **Prior Approaches**: 기존 합성 데이터 연구는 주로 개인정보 보호를 위해 데이터에 잡음이나 집계/노이즈 처리를 적용하거나(SDV류 평가, k-anonymity 등) Synthea 같은 시뮬레이터로 환자 전자건강기록을 만든 뒤 유틸리티를 downstream 과제로 검증하는 방식에 치중해 왔다. LLM을 활용하더라도 단일 문서 생성에 머물러 환자 여정 전체에서의 시간선 일관성, 진료 기록의 형식·스타일 변동을 함께 보장하기가 어려웠다. 또한 LLM의 hallucination은 의료 문서에서 신뢰성을 흔들 수 있어 검증·보정 절차가 필수인데, 이를 파이프라인 전반에 구조적으로 넣지 못한 한계가 있었다.

- **Core Contribution**: 이 논문은 개인정보 위험 없이 임상 AI 도구를 개발·평가할 수 있도록 ‘합성 clinical notes 파이프라인과 데이터셋’을 제안한다. 핵심은 모듈형 구성으로 구조화된 환자 생성, 반구조화된 입원 여정 시뮬레이션, 그리고 LLM 기반 비구조화 임상노트 생성까지 한 흐름으로 묶되, 장기적으로(여정 전반) 내부 일관성을 최우선으로 설계한 점이다. 여기에 글쓰기 스타일·노트 구조·임상 디테일의 변동성을 함께 포함하도록 구성하고, LLM 기반 validation 및 augmentation으로 충실도·현실성·다양성을 보강한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 의료적으로 그럴듯한 노트를 생성하면서 (2) 동일 환자의 여러 노트가 같은 사건·맥락·진화하는 임상상을 공유해 시간선과 사실관계를 유지하는 것이다. 논문은 ‘simple journey를 먼저 만들고, 조기 종료를 감지해 이어서 완성’한 뒤, LLM validator가 현실성 문제를 수정하는 루프를 두고, 이후 각 이벤트별로 노트를 생성하되 event 정보가 누락·왜곡되지 않게 faithfulness를 중심으로 다시 검증한다. 또한 JSON 출력 깨짐을 regex 및 재정제 LLM으로 처리하고, 노트에는 persona(예: concise, narrative, bullet points)와 오타/약어 변형을 주입해 실제 문서의 잡음과 기록 습관 차이까지 반영한다.

- **Empirical Impact**: 실험적으로는 파이프라인 산출물을 가독성(Flesch Reading Ease, Dale Chall), fluency(자연스러움), groundedness(이벤트/환자정보 기반 정합성), relevancy(중요 정보 누락 여부), 그리고 시간 순서 일치율로 평가해 반복 개선에 활용한다. 또한 의료진 피드백을 여러 차례 반영해 프롬프트와 생성 규칙을 조정했으며, gender swap 등 편향 점검을 위한 추가 역량도 포함한다. 공개되는 데이터셋은 합성 환자 70명(각 20~50개 노트, 전체 입원 여정 범위)을 Bronze/Silver/Gold 수준의 validation 체계로 제공하며, 요약, 코딩(coding), 의사결정 지원 등 임상 NLP 개발·평가를 실제 환자 데이터 없이 확장 가능하게 만든다는 의미가 있다.



### TAVR-VLM: Risk-Conditioned Causal Grounding for Hallucination-Resistant Report Generation (https://arxiv.org/abs/2606.26874)
- **Prior Approaches**: 기존 연구는 TAVR을 위험 점수 분류나 일반적 영상-리포트 생성 문제로 다루며, 멀티모달 LLM이 텍스트를 만들 때 시각 토큰에 주로 의존하는 경향이 있습니다. 그러나 실제 임상 추론은 ‘전체 위험 가설→해부학적 영역 확인→근거 종합’의 위계적 흐름을 요구하는데, 현재 모델들은 이 인과적 연결을 강제하지 못해 진단적 hallucination(근거 없는 그럴듯한 진술)이 반복됩니다. 특히 구조적 심장 중재에서는 이런 오류가 의사 결정에 직접적 위해가 될 수 있습니다.

- **Core Contribution**: 논문은 TAVR-VLM을 제안하며, Risk-Conditioned Causal Grounding Attention(R-CGA)으로 모델 내부에 ‘Risk → Region → Word’ 구조적 근거 경로를 구현합니다. 전역 위험 상태가 시각적 관심 영역 선택과 단어 수준 근거 맵 생성을 동시에 지배하도록 설계해, 위험 예측을 단순 분류가 아닌 생성의 구조적 prior로 바꿉니다. 결과적으로 생성 텍스트가 해부학적 증거와 정렬되도록 만드는 것이 핵심 기여입니다.

- **Technical Challenges**: 핵심 난제는 위험 상태와 국소 해부학적 단서 사이의 인과적 위계를 강제로 학습시키면서도, 생성 과정에서 위험 마스크 밖의 근거가 새어나가지 않게 만드는 것입니다. 저자들은 멀티모달 입력을 압축해 ‘causal risk bottleneck’과 global risk mask를 만들고, autoregressive generation 중 토큰-공간 근거를 risk-defined support mask 안으로 투영하는 방식으로 containment을 보장합니다. 또한 causal consistency loss에 stop-gradient 정규화를 적용해 역방향으로 제약이 확장되는 현상을 막고, 토큰 근거가 위험 마스크를 지키도록 최적화합니다.

- **Empirical Impact**: M^3TAVR의 1,482명 코호트에서 TAVR-VLM은 AUROC 0.896, CIDEr 0.936을 달성하며 새 state-of-the-art 성능을 보고합니다. 무엇보다 hallucination rate을 8.1%로 크게 낮추고, mIoU를 0.624까지 끌어올려 근거-영역 정합성을 정량적으로 개선했습니다. 위험-비주얼 정화, causal consistency loss, stop-gradient 등 구성요소의 제거 실험이 성능/정합성 하락을 일관되게 보여주며, 고위험 수술 계획에서 해석 가능성을 높이는 방향성을 제시합니다.



### AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems (https://arxiv.org/abs/2606.26859)
Comments:
          Authors are listed alphabetically by their first name

- **Prior Approaches**: 기존 추천 알고리즘 연구는 엔지니어가 가설을 만들고 프로덕션 코드를 수정한 뒤 A/B 실험을 실행·분석하고 온라인 성과를 해석하는 ‘아이디어-런치’ 흐름에 크게 의존해 왔다. 이 구조 때문에 성과가 실험 지식과 누적 학습으로 폭발적으로 확장되기보다, 인력 규모에 거의 선형으로만 비례해 스케일링이 제한된다.

- **Core Contribution**: 이 논문은 프로덕션 환경에서 동작하는 멀티에이전트 시스템 AgentX를 제안하며, 추천 실험의 생산 함수를 사람이 아니라 에이전트가 주도하도록 재구성한다. AgentX는 자동으로 추천 실험을 생성·구현·평가하고 그 결과로부터 학습하는 self-evolving 개발 엔진을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술 과제는 실험을 자동화하되 안전하게 롤아웃하고, 실패까지 포함해 지식으로 축적하며, 코드 생성이 실제 저장소/시스템 제약을 만족하도록 검증하는 것이다. AgentX는 Brainstorm Agent로 실행 가능한 제안을 순위화하고, Developing Agent에서 repository-grounded 코드 생성과 다차원 reliability 검증을 수행하며, Evaluation Agent가 guardrail-veto된 A/B 판단으로 안전한 온라인 평가를 실시한 뒤 SGPO의 semantic-gradient 업데이트(SGPO)로 실행 궤적을 에이전트에 반영해 점차 개선한다.

- **Empirical Impact**: 논문은 AgentX가 수작업 워크플로우가 따라가기 어려운 스케일과 속도로 추천 실험을 반복하며, 성공과 실패 모두를 구조화된 knowledge asset으로 전환해 누적 학습을 강화한다고 제시한다. 결과적으로 추천 연구·실험 사이클을 산업화된 closed loop로 바꿔, 인력 의존적 혁신을 evidence/compute 기반의 자기 개선형 혁신으로 전환하는 의미가 있다.



### LCAi: Life Cycle Assessment with big data fusion and retrieval-augmented generation-assisted interpretation (https://arxiv.org/abs/2606.26857)
Comments:
          23 pages, 14 figures, 6 tables. Includes Supplementary Information

- **Prior Approaches**: 기존 LCA 연구는 interpretation 단계에서 환경 핫스팟과 정량 개선 여지를 연결하되, 이를 기술·사회·정책 불확실성 하의 실행 로드맵으로 번역하는 구조가 부족하다는 한계가 지적된다. AI를 LCA에 도입한 접근은 자동화·효율은 높였지만, 과학적 품질 검증과 표준화가 약하거나 실제 검증이 제한적인 경우가 많았다.

- **Core Contribution**: 이 논문은 관점(perspective) 조건부 retrieval-augmented generation(RAG)으로 LCA 해석 결과를 실행 가능한 전략 경로로 연결하는 프레임워크를 제안한다. 시나리오 앵커(scenario anchor)로 시스템 경계와 탈탄소 목표를 고정하고, 학계·산업·공공 담론·EU 펀딩 데이터를 다중 관점으로 검색한 뒤 통제된 합성으로 교차 도메인 근거를 통합한다.

- **Technical Challenges**: 핵심 기술 과제는 LLM의 hallucination 위험을 줄이면서도 관점 다양성을 유지하는 것이다. 이를 위해 (1) 관점별 마이크로-쿼리를 제한 검색으로 수행하고, (2) 중간 결과를 ‘ledger’에만 저장한 뒤 최종 합성 단계에서는 추가 retrieval 없이 통합하며, (3) 각 주장에 근거 citation과 self-evaluation 기반 confidence/품질 플래그를 요구한다.

- **Empirical Impact**: 이 방법은 이탈리아 사과 생산 LCA를 사례로, 디젤 소비가 핫스팟(운송/사용 중심)이며 50% 감축 목표에서 renewable/green hydrogen 대체를 실행 관점에서 탐색하는 proof-of-concept로 시연된다. GPT-5 nano을 reasoning 모델로 사용해 근거 기반이고 감사를 가능하게 하는 해석을 제공하며, 기존 LCA를 넘어 구현 지향 의사결정을 지원할 수 있음을 보여준다.



### Context-Aware Synthesis of Optimization Pipelines for Warehouse Optimization (https://arxiv.org/abs/2606.26852)
- **Prior Approaches**: 기존 연구는 창고 주문처리의 구성요소(아이템 할당, 오더 배칭, 피커 라우팅, 피커 스케줄링)를 부분문제로 나눠 각각의 알고리즘을 비교하거나, 일부 조합(예: 배칭+라우팅)만 고정해 평가하는 데 집중해 왔습니다. 통합 모델도 시도되지만 조직 경계, 책임 범위의 이질성, 데이터 제약 때문에 실제 현장 적용성이 떨어진다는 한계가 지적됩니다. 또한 알고리즘 선택과 조합(파이프라인 구성)이 “언제 어떤 설정이 유효한지”를 자동으로 판별해 주지 못해 재현성과 일반화가 제한됐습니다.

- **Core Contribution**: CASOP(Context-Aware Synthesis of Optimization Pipelines)은 창고 ‘컨텍스트’에 맞는 최적화 파이프라인을 자동 합성하고, 그 파이프라인들의 성능을 체계적으로 평가하는 프레임워크를 제안합니다. 이를 위해 (1) 모듈형 알고리즘 저장소, (2) 창고 상황을 기술하는 semantic data 카드와 알고리즘 요구조건을 기술하는 algorithm card, (3) 주문처리 문제를 4개 서브문제로 구조화하는 taxonomy, (4) 컨텍스트 기반 적용 알고리즘 매칭과 유효 파이프라인 합성을 수행하는 합성기, (5) 생성된 모든 파이프라인을 평가하는 평가기를 결합합니다. 결과적으로 수동으로 “좋아 보이는” 조합을 고르지 않고, 유효한 구성만 자동으로 만들어 고성능을 선별할 수 있게 합니다.

- **Technical Challenges**: 핵심 기술 난관은 (a) 주어진 창고 특성에서 어떤 알고리즘이 ‘적용 가능’한지 판정하고, (b) 그 알고리즘들을 조합해도 되는 ‘유효한’ 파이프라인만 생성하며, (c) 큰 설계공간에서 모든 후보 파이프라인을 실행·비교할 수 있을 정도로 효율적으로 평가하는 데 있습니다. CASOP은 타입 기반 구성 아이디어(Combinatory Logic Synthesizer)를 파이프라인 합성에 적용해, algorithm card와 semantic data의 매칭 결과를 근거로 적용 가능한 컴포넌트만 조합합니다. 또한 pipeline evaluator가 모든 유효 파이프라인을 대상으로 성능을 평가하도록 구성해, 단일 문제나 고정 조합이 아닌 “컨텍스트-특화” 선택을 가능하게 합니다.

- **Empirical Impact**: CASOP은 4개 문제 클래스에 대한 7개 benchmark instance set에 적용되며, 총 1,063,044개의 유효 파이프라인을 생성·평가하는 대규모 실증을 수행합니다. 특히 단일 피커 라우팅(SPRP)부터 흩어진 저장(SPRP-SS), 배칭+라우팅(OBRP), 배칭+라우팅+스케줄링(OBRSP)까지 여러 서브문제를 함께 다루며 기존 문헌의 “분리/부분 조합” 관행을 확장합니다. 연구·실무자가 창고 설정에 맞춰 자동으로 유효하고 성능 좋은 알고리즘 파이프라인을 설계·선택할 수 있도록 소프트웨어도 open-source로 공개한다는 점에서 현장 적용성 측면의 의미가 큽니다.



### The Capability Frontier: Benchmarks Miss 82% of Model Performanc (https://arxiv.org/abs/2606.26836)
- **Prior Approaches**: 기존 벤치마크는 보통 단일 모델을 단일 실행(단일 샘플)으로 평가해, 실제 현장에서의 성능 상한을 과소평가한다. 또한 라우팅의 ‘oracle’ 성능을 유한 생성 횟수에서 샘플 평균의 최댓값으로 근사하는데, 이 과정은 잡음에 의해 낙관적으로 편향(optimizer’s curse)될 수 있다.

- **Core Contribution**: 이 논문은 모델 조합과 생성 샘플을 최적으로 선택했을 때의 최선 성능을 비용 축에서 정량화하는 Capability Frontier(품질-비용 Pareto frontier)를 제안한다. 단일 모델 평가의 과소평가를 교정하는 동시에, 단순 최대값 집계에서 생기는 과대평가 편향까지 함께 보정하는 것이 핵심이다.

- **Technical Challenges**: 핵심 기술적 난제는 유한 샘플(예: 생성 G≤10)에서 oracle 추정이 잡음에 의해 양(+)의 편향을 띤다는 점이다. 이를 위해 생성 수에 따른 편향 감소 스케일링을 이용한 extrapolation 기반 보정과, 잠재 토픽·난이도·모델 aptitude을 포함한 확률 그래픽 모델(PGM)로 true oracle을 복원하는 방식을 함께 제시한다.

- **Empirical Impact**: 21개 LLM과 16개 벤치마크(코딩·추론·의학·사실성·지시수행·에이전트)에서, 비용을 맞춘 상태로 단일-모델 기준 대비 평균 오류율이 54% 감소했고 단일 실행 변동까지 보정하면 82%까지 개선됐다. 또한 SOTA 품질과 동등한 정확도를 기준으로 frontier는 평균 85% 비용 절감에 해당하며, 시뮬레이션에서는 데이터 이질성(토픽 엔트로피)이 커질수록 oracle 격차가 거의 단조 증가함을 보여 벤치마크 평가와 배포 전략에 중요한 의미를 갖는다.



### Computational Analysis of Heart Rate Variability in Healthy Adults (https://arxiv.org/abs/2606.26816)
- **Prior Approaches**: 기존 HRV 연구는 질병군 중심으로 진행되어 건강한 개인에서의 HRV 파라미터 해석과 기준선 설정이 충분하지 않았습니다. 또한 문헌 전반에 일관된 gold standard가 없어, 어떤 지표를 어떻게 선택하고 비교해야 하는지 불명확하다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 30~50세 건강 성인 40명(남 20, 여 20)을 대상으로 time, frequency, nonlinear 영역의 HRV 지표를 종합 점검해 임상·연구 활용성을 높이는 것을 목표로 합니다. 특히 normality, stability, correlation, reproducibility, consistency의 5개 질문 프레임으로 지표의 신뢰성과 대표성을 체계적으로 평가합니다.

- **Technical Challenges**: 핵심 기술적 난제는 지표별로 분포 가정(정규성), 측정 변동(안정성·재현성), 지표 간 중복성(상관), 데이터 간 일관성(다른 연구/데이터셋 간 비교)을 동시에 만족하는 ‘대표 지표’를 고르는 것입니다. 연구진은 신호처리 및 데이터 분석을 통해 각 지표를 정밀 평가했고, 그 결과 ApEn·IRRR·HRVi·SD2·MADRR/rMSSD 조합이 HRV 구성요소를 가장 정확히 반영한다고 제시했습니다.

- **Empirical Impact**: Fantasia database와의 비교에서 대부분의 지표는 10% 미만 오차를 보였으나, 여성의 SD2와 SDNN은 15% 이상 오차가 나타났습니다. 또한 time-domain 및 nonlinear 지표는 연구 간 변동성이 낮은 반면 frequency-domain 지표는 변동성이 커 교차 연구 비교에 제약이 있음을 실증적으로 보여, 향후 HRV 연구 설계에 직접적인 기준을 제공한다는 점에서 의미가 큽니다.



### KARLA: Knowledge-base Augmented Retrieval for Language Models (https://arxiv.org/abs/2606.26807)
- **Prior Approaches**: 기존에는 LLM에 대한 파인튜닝이나 프롬프트 엔지니어링으로 지식과 사실을 맞추려는 접근이 주로 쓰였지만, 지식이 바뀌면 재학습이나 재설계가 필요해 유지보수가 어렵습니다. 또한 검색을 단순히 붙이는 방식(RAG)은 가능해도, 생성 과정에서 언제 어떤 사실을 가져와야 하는지의 정교한 제어가 한계로 지적돼 왔습니다.

- **Core Contribution**: 이 논문은 생성 중에 LLM이 지식베이스(knowledge base)에서 사실을 자동으로 끌어오도록 하는 방법을 제안합니다. 핵심은 특별 토큰을 만들어 토큰 생성 흐름 안에서 지식베이스 질의를 트리거하게 하고, 결과가 지식베이스 출처로 추적되며, KB 편집만으로 사실을 업데이트할 수 있게 한 점입니다.

- **Technical Challenges**: 가장 큰 기술적 과제는 생성 토큰과 외부 지식 호출을 자연스럽게 결합해, 필요한 순간에 정확히 질의·삽입이 일어나도록 학습하는 것입니다. 저자들은 특별 토큰을 통해 KB 조회를 유도하고, 그 토큰이 실제로 질의 실행으로 연결되도록 학습해 사실적 근거를 강화했습니다.

- **Empirical Impact**: 실험 결과, 이 방법은 짧은 생성과 긴 생성 모두에서 factual grounding을 개선했으며, 파라미터 업데이트 없이 KB 수정으로 사실 반영이 가능함을 보여줍니다. 또한 더 작은 모델이 더 큰 모델과 유사한 사실 정확도를 달성할 수 있어, 비용 대비 성능 관점에서도 의미가 큽니다.



### Memory Depth, Not Memory Access: Selective Parametric Consolidation for Long-Running Language Agents (https://arxiv.org/abs/2606.26806)
Comments:
          Main paper with supplementary material included as ancillary file

- **Prior Approaches**: 기존 long-horizon 언어 에이전트의 해법은 retrieval(검색 증강 생성) 중심으로, 외부 저장소에서 관련 텍스트를 뽑아 생성에 반영한다. 다만 retrieval은 ‘무엇을 가져올 수 있는가’에는 답하지만, 작업 컨텍스트가 비워진 뒤에도 ‘무엇이 계속 행동을 좌우해야 하는가’를 결정하진 못한다. 논문은 이 공백을 memory depth로 정의하고, 기존 벤치마크가 post-unload 상황을 분리해 측정하지 못한다고 지적한다.

- **Core Contribution**: 저자들은 memory access와 memory depth를 분리해 평가할 수 있도록 loop-drift protocol을 제안한다. retrieval 인덱스는 유지되지만 working context만 unload되는 조건에서, 목표·선호 같은 goal-conditioned 경향이 오래 지속되는지(목표 지속성과 post-unload recovery)를 측정한다. 또한 EVAF( surprise- and valence-gated LoRA consolidation )로 “의미 있는 이벤트를 선별해 작은 파라미터 저장소에만 덜어내어” 동작의 지속성을 노린다.

- **Technical Challenges**: 핵심 기술 과제는 ‘적절한 이벤트를 골라 쓰는(selection)’ 것과 ‘얼마나 강하게 쓰는(actuation)’ 것을 동시에 만족시키는 것이다. EVAF는 surprise(토큰 negative log-likelihood 기반)와 valence(사용자의 durable goal·선호 임베딩 유사도 기반)로 게이트를 걸어, 버퍼가 찰 때만 LoRA 어댑터를 업데이트하며 replay와 L2 anchor로 드리프트를 억제한다. 추가 제어실험으로 선택만 하는 조건( matched random gate )과 actuation만 분리한 fixed-inner 컨트롤을 수행해, 두 요소가 분해 가능하지만 온라인 피드백 때문에 결합된 비대칭성이 나타남을 보여준다.

- **Empirical Impact**: 실험에서 RAG는 shallow factual recall에서 가장 강하고( short-fact accuracy 0.956~0.973 ), EVAF는 goal persistence와 post-unload recovery에서 우세하다(0.812~0.904)고 보고한다. 특히 200 이벤트당 2~3회 수준의 파라미터 쓰기만으로 성능 이득을 얻으며, selection-actuation을 분해한 분석은 ‘적게 쓰는 것’이 아니라 ‘정확히 고르고 조절하는 것’이 관건임을 뒷받침한다. 다만 Memora 공개 이벤트 스트림 경계진단에서는 stale-memory invalidation(삭제/갱신 유효성)이 유의미하게 해결되진 않아, validity gating/재통합(reconsolidation) 같은 후속 과제를 남긴다.



### ResilPhase: Plug-and-Play Phase Mapping and Noise-Resilient Macro-Trajectory Extrapolation for Diffusion Acceleration (https://arxiv.org/abs/2606.26769)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 확산 Transformer(DiT) 가속은 대부분 step 수를 줄이기 위해 “cache-then-forecast” 방식에 의존해 왔습니다. 하지만 레이어(블록) 단위 특징을 다항으로 외삽하는 접근은 고가속 비율에서 품질이 급격히 무너지는 문제가 반복됐습니다.
기존 방법들은 미분 기반 근사(Taylor/Hermite 등)에 크게 의존하면서, 연속 궤적의 고차 시간 미분이 노이즈에 취약하다는 점과 숫자적 불안정(Runge’s phenomenon)을 충분히 통제하지 못했습니다.

- **Core Contribution**: 논문은 품질 저하의 근본 원인을 “연속 확산 경로와 어긋난 표현에 대해 불안정한 이산 외삽을 수행”하기 때문이라고 분석합니다. 이를 바탕으로 ResilPhase는 외삽 대상을 레이어별 중간 특징이 아니라 ODE 공간의 end-to-end 상태 변화인 Global Drift(GD)로 재정의합니다.
또한 GD에 맞춘 derivative-free barycentric Lagrange 외삽과, 오류가 발산하기 쉬운 외삽 구간을 안정화하는 bounded Phase Mapping까지 묶어 하나의 노이즈-복원형 가속 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 레이어 단위 외삽에서 발생하는 공간적 오차 누적, (2) 미분 기반 외삽이 고차 미분 노이즈를 증폭하는 시간적 불안정, (3) 균일 이산 시간에서의 다항 외삽이 Runge’s phenomenon을 유발하는 수치적 불안정입니다.
ResilPhase는 GD 예측으로 레이어 오차 연쇄를 차단하고, barycentric Lagrange를 통해 유도(derivative) 추정 없이 외삽을 수행하며, Chebyshev 또는 Balanced Mapping으로 외삽 도메인을 bounded phase domain으로 투영해 발진성 오차 성장을 억제합니다.

- **Empirical Impact**: FLUX.1-dev와 HunyuanVideo에서 ResilPhase는 공격적인 가속에서도 충실도(fidelity)를 크게 유지하며, 속도-품질 균형이 경쟁 방법 대비 일관되게 향상됐다고 보고합니다. 예를 들어 FLUX.1-dev에서 약 4.97× 가속에도 ImageReward를 1.0258로 유지하며, TaylorSeer 대비 상당한 개선과 더 낮은 LPIPS를 보였습니다.
또한 HunyuanVideo에서는 약 5× 가속에서 VBench 점수 우위와 LPIPS 감소가 확인됐고, DiT-XL/2 조건에서도 FID가 낮게 유지되며 다수 캐싱 베이스라인이 붕괴하는 구간에서 강건함을 입증했습니다.



### EGG: An Expert-Guided Agent Framework for Kernel Generation (https://arxiv.org/abs/2606.26758)
- **Prior Approaches**: 기존 LLM 기반 GPU 커널 생성은 fine-tuning이나 RL로 도메인 적응을 시도하거나, 에이전트 기반으로 반복 개선을 수행한다. 하지만 GPU 커널은 방대한 하드웨어 의존 최적화 공간을 가지면서도 컴파일/정확성 제약이 강해, RL은 낮은 correctness rate(예: AutoTriton 50% 미만) 같은 문제를 겪는다. 에이전트 방식도 하드웨어 최적화 지침이 부족해 coarse-grained 피드백에 의존하면 trial-and-error 탐색으로 성능이 제한되고 수렴이 불안정해진다.

- **Core Contribution**: EGG는 Expert-Guided Agent Framework로, LLM의 의사결정을 전문가 최적화 원리에 맞춰 단계별로 제한한다. 커널 생성 과정을 algorithmic structure design(알고리즘 구조 설계)과 hardware-specific tuning(하드웨어 튜닝) 두 계층으로 분해해, 탐색 공간을 명시적 최적화 목표로 구조화한다. 또한 stage-aware multi-agent 협업으로 에이전트 간 컨텍스트를 관리해 단계 전환 시 이전 성능을 누적적으로 유지한다.

- **Technical Challenges**: 핵심 난관은 커널 코드가 동시에 문법/의미/하드웨어 제약을 만족해야 하며, parallel mapping·tensor tiling·메모리 계층 같은 미세한 변화가 성능을 크게 흔든다는 점이다. EGG는 먼저 multi-seed search로 서로 다른 알고리즘 패러다임 시드를 만들고, algorithmic refinement로 semantics-preserving 변환(예: operator fusion, 데이터플로 재구성)으로 구조적 비효율을 제거한다. 이후 parallel mapping, tensor tiling, memory optimization을 순차 하위 단계로 고정된 제약 아래에서 최적화하며, profile agent가 병목/수정안을 제시하고 code agent가 구현·debug agent가 실패를 복구하는 닫힌 루프를 stage objective에 맞춰 반복한다.

- **Empirical Impact**: KernelBench와 실사용 워크로드에서 EGG는 PyTorch Eager 대비 평균 2.13x 속도를 내며, success rate 100%와 함께 특히 fused/end-to-end로 갈수록 우위를 유지한다. Fast_1 rate도 전 난이도에서 높게 나타나(예: Level 2는 100%), 일반 LLM(34–66% success, 11–32% Fast_1 rate)이나 기존 agent/RL 접근보다 견고한 결과를 보인다. 또한 단일 작업 생성이 약 20분 수준으로 CudaForge 대비 출력 토큰 소모가 줄어(약 50k vs 110k) 단계 분해가 검색 효율까지 개선함을 시사한다.



### Scientific discovery as meta-optimization: a combinatorial optimization case study (https://arxiv.org/abs/2606.26728)
Comments:
          35 pages, 6 figures

- **Prior Approaches**: 기존 자동 과학 탐색은 LLM이 가설·코드·실험·논문까지 생성하는 루프를 만들거나, 진화/트리서치와 결합해 알고리즘을 찾는 방식이 주류였다. 하지만 이런 시스템은 무엇을 ‘좋다’고 평가할지(목적함수·평가 기준)를 고정하거나 단계적으로만 바꾸는 경향이 있어, proxy objective 최적화 과정에서 reward hacking(메트릭 편승) 문제가 쉽게 발생한다.

- **Core Contribution**: 이 논문은 과학 탐색을 단순 최적화가 아니라 meta-optimization(메타 최적화)으로 공식화하고, 목표함수 자체가 함께 진화하도록 설계한다. 핵심 기여는 consensus objective aggregation으로, LLM이 생성한 여러 objective function을 Kendall’s τ 순위 상관으로 신뢰도를 추정해 correlation-weighted voting과 weighted Borda count로 하나의 합의 평가로 묶는다. 또한 age decay로 오래된 기준의 영향이 줄고 새로운 기준이 반영되며, 불일치하는 평가 기준은 자동으로 약화되어 자기교정적 평가가 가능해진다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘진짜 연구 목표’를 정확히 반영하지 못하는 proxy 목적함수들이 서로 편향·오류를 가질 때, 그중 어떤 것을 믿고 언제 버릴지 정하는 것이다. 저자들은 각 objective의 생성 라운드 정보를 age decay에 넣고, median Kendall’s τ 기반의 clamped 가중치로 다수와 합의되는 기준에 더 무게를 주는 구조를 통해 outlier와 잡음(또는 적대적 목적)을 억제한다. 더 나아가 agreement 기반 다수결이 echo chamber로 굳어질 경우를 대비해 meta-agent가 추가 weight multipliers로 군집 지배를 깨는 보정도 제공한다.

- **Empirical Impact**: 실험은 planted random 3-SAT을 대상으로 digital MemComputing machines(DMM)에서 알고리즘을 설계하는 사례연구로 진행됐으며, 414개 솔버 디자인과 42개의 co-evolving objectives를 함께 탐색했다. 그 결과 baseline의 scaling이 N^{2.51} 수준에서 N^{1.33}로 감소하고, 가장 큰 테스트 인스턴스에서 약 67배 속도 향상(∼67×)을 달성했다. 문제-agnostic 메타 최적화 프레임워크로서, 자동 과학 발견에서 objective specification 및 reward hacking을 완화하면서 성능을 실증했다는 점에서 의미가 크다.



### Socratic agents for autonomous scientific discovery in high-dimensional physical systems (https://arxiv.org/abs/2606.26722)
Comments:
          27 pages,5 figures

- **Prior Approaches**: 기존 자동화 과학은 AI가 문헌·가설·실험 코드를 생성하거나 로봇/기기를 제어하더라도, 목표·개입 공간·평가 지표·해석 경로가 사람에 의해 절차적으로 고정되는 경우가 많았다. 또한 에이전트의 self-reflection이나 사후 검토는 언어적 그럴듯함을 재평가하는 데 그쳐, 인과 구조와 실패 조건까지 명시적으로 다루지 못한다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 에피스테믹 자율성(증거에 반응해 물리 설명을 만들고, 검증하고, 수정하는 능력)을 목표로, Socratic midwifery를 내장한 멀티에이전트 과학자 AHOIS를 제안한다. 특히 physics-critic 에이전트가 가설을 인과 질문, 제약 점검, 반례 생성, falsification-criteria 정립으로 공격해, 숨은 가정과 불완전한 해석을 실험 가능한 과학 상태로 변환한다.

- **Technical Challenges**: 핵심 기술 난점은 복잡한 고차원 물리 환경에서 측정이 ‘유용한 신호’인지 ‘장애(드리프트/잡음/오작동/모달리티 아티팩트)’인지 구분하고, 그 구분을 위해 어떤 개입이 차별적(discriminating)인지 알아내는 것이다. AHOIS는 공통 과학 상태를 중심으로 strategy·hardware abstraction·system integrity·modelling을 분업하고, critic이 추론 구조를 4단계(명확화, 제약, 인과, 반례/불일치)로 재조립해 가설의 물리 일관성과 실험 계획 타당성을 높였다.

- **Empirical Impact**: 실험은 실제 multimode-fibre 광학 플랫폼에서 무(無)인코딩·무분류기·무speckle 사전모델 조건으로 진행되며, AHOIS가 random-interference encoding 가설을 제안·검증하고 task-adaptive sparse-measurement 전략을 찾아냈다. 또한 encoding instability, fluorescence contamination, detector noise 등 서로 다른 실패 모드를 진단하고, 논문에 제시된 영상 복원 프로토콜을 비(非)원래 장치 설정으로 실행 가능한 워크플로로 이전했으며, 해당 인코딩은 16×16 측정에서 effective rank 56.9, MNIST 76.97% 및 Fashion-MNIST 83.17% 정확도를 보였다.



### A Latent ODE Approach to Spatiotemporal Modeling of Cine Cardiac MRI (https://arxiv.org/abs/2606.26718)
- **Prior Approaches**: 기존 CMR 위험모델은 LVEF, strain 같은 소수의 이미지 지표와 임상 변수를 결합해 위험을 추정하지만, 해부학적 고차원 정보를 요약하고 심장주기의 특정 국면(주로 end-diastole/end-systole)에 집중하는 한계가 있습니다. 또한 spatiotemporal 생성·예측 모델들은 시간을 대개 이산 프레임 시퀀스로 다뤄, 심박수에 따른 프레임 수·간격 변화와 수축/이완의 비율 차이를 충분히 반영하기 어렵습니다.

- **Core Contribution**: 이 논문은 양심실(biventricular) 해부학과 심장주기 전체(cine) 운동을 연속적인 잠재 궤적(latent trajectory)으로 인코딩하는 latent dynamical model을 제안합니다. covariate-conditioned prior로 정상적인 end-diastolic latent 상태를 정의한 뒤, posterior와의 편차를 표준화해 Cox proportional hazards 모델의 residual 위험 점수로 연결합니다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 심박수에 따라 달라지는 생리적 위상(phase) 타이밍을 모델 학습·생성에 일관되게 반영하고, (2) 메시(mesh) 기반 운동을 해부학적으로 일관되며 자연스러운 연속 시간 생성으로 학습하는 것입니다. 저자들은 graph-based mesh autoencoder로 3D+t 운동을 복원하면서, heart-rate-aware neural ODE와 learnable monotone phase reparameterization으로 시간 왜곡 문제를 완화하고 latent 잔차 임베딩을 survival 분석에 투입합니다.

- **Empirical Impact**: UK Biobank 72,386명(incident heart failure 367명)에서 평가된 held-out subset 기준, latent score를 pooled cohort equations에 추가했을 때 stratified C-index가 0.704→0.785로 상승했으며, 7개 기존 심장 마커 대비 우수한 성능(기존 0.764)을 보였습니다. 또한 GNN+ODE 조합이 non-graph, non-ODE 대비 reconstruction fidelity·생성 현실감·예후 성능 사이의 균형이 가장 좋았지만, 임상 적용 전 더 대표적인 환자 코호트에서 외부 검증이 필요하다고 밝힙니다.



### LithoDreamer: A Physics-Informed World Model for Multi-Stage Computational Lithography (https://arxiv.org/abs/2606.26713)
Comments:
          Correspondence to: Qi Sun <qisunchn \at zju \dot edu \dot cn>

- **Prior Approaches**: 기존 OPC/ILT 계열은 마스크를 명시적 변수공간에서 반복 수정하며 목표 제약을 만족시키지만, 물리 시뮬레이터 의존도가 높아 설계 규모가 커질수록 비용이 급증한다. 학습 기반 방법(예: Unitho, LMLitho)은 Layout→Mask→Resist Image→ADI 같은 다단 매핑을 정적 예측으로 다뤄 연속적인 공정 개입이 만들어내는 상태 진화를 모델링하기 어렵다. World Models(WM)는 장기 의사결정/궤적 학습에 강점이 있지만, 공정 단계마다 물리 성질이 달라 단일 잠재공간에서 stage-dependent dynamics를 일관되게 담기 어렵고, 개입(개선/조정) 자체는 관측되지 않아 개입-조건부 동역기 학습이 제약된다.

- **Core Contribution**: LithoDreamer는 계산 리소그래피의 “Layout-Mask-Resist Image-After Development Image(ADI)” 파이프라인을 의사결정 기반의 다단 물리 진화 시스템으로 정식화한, physics-informed World Model(세계모델) 프레임워크다. 인접 상태 간 특징 변화를 학습해 단계별로 물리 제약을 반영한 latent space를 구성하고, 이를 통해 공정 intervention 탐색과 다음 상태 전이를 동시에 구동한다. 또한 중간 단계에 대한 연속 supervision 없이도 개입 경로의 latent 차이를 대비학습·변분 최적화로 연결해, 실제 리소그래피 물리와 정합적인 evolution을 생성하도록 유도한다.

- **Technical Challenges**: 핵심 난제는 (1) 다단마다 서로 다른 물리 특성이 존재해 단일 latent 모델로는 stage-dependent dynamics를 포착하기 어렵고, (2) dose/focus/threshold/source 같은 조건은 관측돼도 연속 패턴 변화를 만드는 미세 개입은 미관측이라 학습 신호가 불완전하다는 점이다. LithoDreamer는 Space Prior Approximation(SPA)로 단계별 물리 정보를 반영한 잠재공간 basis를 추정해 intervention이 물리적으로 가능한 방향에만 놓이게 제한한다. 이어 정책은 stochastic intervention coefficient 분포를 샘플링해 탐색성을 확보하고 Diffusion Transformer(DiT) 기반 전이모델로 다단/다계 스텝 state evolution을 예측하며, 중간 상태 관측이 없을 때는 terminal-supervision만으로 variational surrogate를 최적화하는 contrastive variational optimization을 사용해 stage-consistent 개입을 학습한다.

- **Empirical Impact**: 280K 규모(55nm 라인 상용 데이터 기반)의 리소그래피 데이터셋에서 forward evolution과 inverse planning을 모두 평가한 결과, LithoDreamer는 ID 및 OOD 설정에서 최첨단 성능과 일반화를 보였다. 예를 들어 ID forward evolution에서 ADI의 EPE가 3.74nm로 개선되고(상대적으로 1–4nm 범위로 진입), region/boundary 지표에서도 mIoU 78.27%, Edge F1 80.91% 수준을 달성했다. inverse planning에서도 EPE가 Mask 1.32nm, Resist Image 0.89nm, ADI 3.28nm로 낮게 유지되며, 생성이 단순 종단 이미지를 맞추는 수준을 넘어 Mask·Resist 단계에서 점진적으로 contour를 교정해 ADI로 수렴하는 궤적을 보였다. 또한 미보는 공정 파라미터(OOD)에서도 EPE 개선 폭이 크게 나타나 memorization보다 물리 제약 기반 evolution 방향을 학습했음을 시사하며, 공개 데이터셋과 함께 현업 시뮬레이션/제조 친화 평가(EPE 중심)의 의미 있는 전환점을 제공한다.



### Kalman Prototypical Networks for Few-shot Fault Detection in Combined Cycle Gas Turbines (https://arxiv.org/abs/2606.26710)
- **Prior Approaches**: CCGT fault detection은 물리 기반 시뮬레이션 모델, 지도학습 기반 진단, 혹은 하이브리드 접근으로 발전해 왔지만, 실제로는 라벨된 고장 데이터가 희귀하고 기밀/비용 문제로 확보가 어렵다. 이 때문에 unsupervised anomaly detection과 few-shot learning(FSL)이 대안으로 쓰이지만, CCGT 도메인에선 FSL의 안정성 검증과 데이터셋 확장이 부족했다. 특히 prototypical networks 계열은 작은 support set에서 계산된 프로토타입이 에피소드마다 흔들려 episodic variance가 커지고, 그 결과 임베딩 기반 결정경계가 불안정해지는 한계가 있다.

- **Core Contribution**: 이 논문은 Kalman Prototypical Network(KPN)를 제안해, CCGT 고장 진단을 metric-based few-shot learning 문제로 재구성한다. 핵심은 각 클래스의 프로토타입을 단순 에피소드 평균이 아니라 latent stochastic state로 보고, Kalman filtering으로 프로토타입을 시계열처럼 스무딩해 표현의 안정성을 높이는 것이다. 또한 offshore CCGT를 Modelica 기반 고정밀 동적 시뮬레이션으로 만들어 정상/점진적 leak 상황을 포함한 학습·평가용 합성 데이터셋을 제공한다.

- **Technical Challenges**: 가장 큰 technical challenge는 support set이 작을 때 프로토타입 추정이 에피소드마다 크게 요동치면서 일반화가 흔들린다는 점이다. 논문은 프로토타입 궤적 p_k(t)를 잡음이 섞인 관측으로 보고, 과정 잡음 q와 관측 잡음 r을 두는 선형-가우시안 Kalman 필터로 denoised p̂_k(t)를 얻어 query 분류에 사용한다. 학습은 임베딩 파라미터를 gradient-based로 업데이트하되, Kalman 기반 스무딩이 에피소드 간 클래스 표현의 일관성을 유지하도록 설계했다.

- **Empirical Impact**: Modelica/Dymola 시뮬레이션 기반 leak fault 2-way few-shot 설정에서 KPN은 Matching Networks, Relation Networks, MAML 등 기존 FSL 대비 정확도와 안정성(분산)을 동시에 개선했다. 4-shot에서 KPN은 90.51% ± 2.01%로 상위 성능을 보였고, shot 수와 query 수가 달라져도 90%대 정확도를 유지하며 분산이 상대적으로 낮게 나타났다. 이는 CCGT처럼 라벨이 부족한 안전·신뢰성 산업에서, 프로토타입 표현을 안정화하는 접근이 실사용 관점의 견고한 진단 성능으로 이어질 수 있음을 보여준다.



### Do Safety Guardrails Need to Reason? LeanGuard: A Fast and Light Approach for Robust Moderation (https://arxiv.org/abs/2606.26686)
Comments:
          9 pages, 6 figures, 3 tables. Project page: this https URL ; code and models: this https URL

- **Prior Approaches**: 기존 LLM 안전 가드는 큰 디코더 기반 생성형 분류기로 프롬프트/응답을 점검해 거부 여부를 자연어 판정으로 내리는 방식을 주로 써왔다. 또 다른 흐름은 GuardReasoner류처럼 CoT(chain-of-thought)를 먼저 생성한 뒤 판정을 내리며, 단계적 추론이 정확도와 신뢰성을 높인다고 가정한다. 하지만 이 설계는 실제 배포 환경(온디바이스·에ంబ디드 로봇)에서 요구되는 ‘경량·저지연’과 충돌할 수 있다.

- **Core Contribution**: 논문은 “안전 가드는 진짜로 CoT를 필요로 하는가”를 같은 베이스(same-base) 비교로 검증한다. GuardReasoner 학습 데이터를 그대로 두고, 추론 생성만 제거(나머지는 고정)했을 때 중재 정확도가 개선되지 않는지 확인하며 그 결과를 LeanGuard로 정리한다. 또한 추론 가드가 더 무거울수록 강건성도 좋아지는지(학습 라벨 노이즈에 대한 견고성)까지 함께 반박한다.

- **Technical Challenges**: 핵심 실험적 도전은 ‘추론 유무’만 바꿔 다른 요인(아키텍처/스케일/데이터/목적)을 섞지 않는 통제 비교를 만드는 것이다. 이를 위해 ModernBERT-large(분류기)와 Llama/T5 기반의 생성형 가드를 동일 코퍼스에서 CoT 포함/라벨-only로만 분리 학습하고, LeanGuard는 395M label-only 인코더로 단일 forward pass에 verdict를 산출하게 설계했다. 계산 비용은 CoT 생성으로 인해 늘어나는 토큰 수에 비례해 증가하므로, 논문은 단일패스가 효율·성능을 동시에 만족하는지 정량화한다.

- **Empirical Impact**: 결과적으로 395M LeanGuard는 공개 벤치마크에서 평균 F1 82.90±0.26을 달성하며, 더 큰 디코더 기반 reasoning guard와 유사한 성능을 단일 forward pass로 맞춘다(추론 계산량 약 100배 절감). 또한 학습 라벨 노이즈를 주입해도 F1이 잘 유지되고, 엄격한 false-positive rate 조건에서 recall이 추론 가드보다 더 잘 보존되어 ‘무거운 추론이 더 강건하다’는 주장을 약화시킨다. 저자들은 현행 가드레일 벤치마크가 추론의 필요성을 충분히 구분하지 못할 수 있으며, moderation에서 CoT의 필수성이 아직 증명되지 않았다고 결론짓는다.



### NebulaExp-8B: An Empirical Post-Training Pipeline via Full-Scale Ablation Research (https://arxiv.org/abs/2606.26671)
Comments:
          29 pages, 8 figures

- **Prior Approaches**: 기존 post-training 연구는 SFT 이후 PPO 계열 또는 GRPO, DPO처럼 보상모델/가치추정 의존도를 줄이려는 흐름과, reasoning 중심 데이터 증류·강화학습을 결합하는 흐름으로 발전해 왔습니다. 하지만 많은 작업이 데이터 구축·필터링 룰·학습 레시피를 충분히 공개하지 않아 재현성과 8B급 경량 모델에서의 가벼운 최적화가 어렵다는 한계가 지적됩니다.

- **Core Contribution**: NebulaExp는 Qwen3-8B-base 위에서 instruct(일반 지시) 브랜치와 reasoning(복합 추론) 브랜치 두 축을 동시에 다루는, “완전 투명” post-training 파이프라인을 제시합니다. 또한 3.84M 멀티소스 SFT 샘플과 200K verifiable RL 후보풀을 기반으로 응답 증류, 다차원 교차검증 필터링, 세분 난이도 그레이딩, 작업 분류, 다양성 인지 샘플링까지 end-to-end 데이터 스택을 공개합니다.

- **Technical Challenges**: 핵심 난제는 (1) 소스 데이터의 형식·스타일·정답 신뢰도 불일치, (2) 난이도 라벨 부재로 인한 부적절한 커리큘럼, (3) RL에서 task verifier 의존성이 만든 불안정성입니다. NebulaExp는 Qwen3 계열로 형식 통일(응답 distillation)과 규칙/모델/과제별 정답 검증을 수행하고, pass@4 기반의 세 구간 난이도 그레이딩 및 과제 분류+diversity-aware sampling으로 학습 데이터를 재구성합니다.

- **Empirical Impact**: Instruct 브랜치에서 3단계 최적화 SFT(NebulaExp-Ins-SFT)는 Qwen3-8B-nothink 대비 평균 벤치마크 점수를 55.01→60.99로 끌어올렸고, 이어진 GRPO RL로 61.85까지 향상됩니다. Reasoning 브랜치에서는 medium-difficulty GRPO RL이 추론 점수를 73.88→75.17로 개선했으며, verifier-free 대안으로 OPD를 단일/다중-교사(MOPD)로 실험해 4K IF 샘플만으로도 IFEval에서 RL 베이스라인을 최대 3.26점, 전체 평균도 최대 4.43점 상회하는 결과를 보고합니다.



### SKILL-DISCO: Distilling and Compiling Agent Traces into Reusable Procedural Skills (https://arxiv.org/abs/2606.26669)
- **Prior Approaches**: 기존 연구는 궤적에서 워크플로우를 재사용하거나, LLM이 성공 트레이스에서 실행 가능한 skill을 유도해 verifiability와 조합성을 높이려 했습니다. 다만 공통 절차가 어떤 태스크 시나리오에서 성립하는지, 그리고 여러 성공 트레이스에 걸쳐 공유되는 절차 구조를 어떻게 표현해야 하는지가 불명확했습니다. 그 결과 skill 라이브러리가 트레이스별 표면 패턴에 편향돼 조각나거나 중복될 수 있다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 procedural skill 발견의 범위를 FSM-defined scenarios로 명확히 하고, 성공 트레이스를 알 수 없는 transition graph 위의 경로로 보며, skill을 재사용 가능한 parameterized control-flow subgraph로 정의합니다. 즉 skill은 단순 텍스트 루틴이나 스크립트가 아니라, 여러 성공 트레이스가 공유하는 실행 구조를 구조적으로 추상화한 것으로 정식화됩니다. 이를 바탕으로 SkillDisCo라는 distillation-and-compilation 프레임워크를 제안해, PFSM 하위그래프를 뽑아 callable·executable·verifiable skill로 컴파일합니다.

- **Technical Challenges**: 핵심 난제는 (1) 완전한 transition graph나 PFSM로의 lifting 함수가 없고, (2) parameter binding을 포함한 subgraph matching이 비용이 크며 잡음에 민감하다는 점입니다. SkillDisCo는 이를 distillation에서 normalize(중간 표현으로 정규화)→subgoal-level operation extraction(재사용 가능한 중간 단위로 분해)→consolidation(공통 PFSM 구조를 공유하는 클러스터링)으로 우회해 PFSM 기반 구조 타깃을 근사합니다. 이어 compilation에서는 skill specification(시그니처·전제/사후조건·부작용·메타데이터) 후 합성 및 held-out 검증을 거쳐, 실패하는 스킬은 재합성하고 최종적으로만 라이브러리에 포함합니다.

- **Empirical Impact**: ALFWorld와 WebArena에서 SkillDisCo는 success rate를 높이고 agent turns를 줄여, benchmark와 모델 스케일 전반에서 효율 개선을 보였습니다. 예를 들어 ALFWorld에서는 CodeAct 대비 96.3%→99.3%로 향상되었고, WebArena에서는 ReAct를 23.9%→29.1%로 올리며 offline 기반 ASIoffline보다도 성과가 좋았습니다. 또한 compact하게 컴파일된 skill이 라이브러리 중복을 줄이고 실행 신뢰성을 높이며, 강한 유도 모델의 절차 지식을 작은 실행 모델로 transfer할 수 있음을 분석으로 뒷받침합니다.



### Autoformalization of Agent Instructions into Policy-as-Cod (https://arxiv.org/abs/2606.26649)
Comments:
          Accepted at the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD), ICML 2026

- **Prior Approaches**: 기존 에이전트 안전장치는 대개 fine-tuned safety model이나 prompt-based steering처럼 확률적 가드레일에 의존하거나, hand-coded symbolic enforcement처럼 규칙을 사람이 직접 작성합니다. 하지만 전자는 security-critical 환경에서 형식적 보장을 제공하기 어렵고, 후자는 자연어로 된 정책 스펙의 범위를 넓게 커버하기 어렵습니다.

- **Core Contribution**: 이 논문은 에이전트 프롬프트, MCP tool 설명, natural language 정책 문서를 자동으로 형식 정책으로 바꾸는 autoformalization 파이프라인을 제안합니다. 생성된 정책은 Cedar Policy Language로 작성되며, generator-critic 루프(LLM-based)를 통해 검증 가능한 형태로 정제됩니다. 결과적으로 Policy-as-Code를 통해 런타임에서 외부 결정형 정책 엔진이 에이전트 행동을 허가/차단합니다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM이 만든 정책이 (1) 문법·모순이 없고 (2) 원래 자연어 의도와 의미적으로 정렬되도록 만드는 것입니다. 이를 위해 Verification Sandwich를 적용해 grounding layer에서 툴 스키마와 자원-행동 온톨로지를 구성하고, hard critic은 Cedar 정적 분석(스키마 준수·공리적 모순·vacuous 정책)을 수행하며 soft critic은 rubric 기반 semantic alignment를 평가해 피드백을 반복합니다. 또한 실패 시 fail-closed로 기본 차단을 수행해 LLM 비결정성과 무관하게 안전 강도를 유지합니다.

- **Empirical Impact**: MedAgentBench에서 실험한 결과, autoformalized Cedar 정책이 hand-coded symbolic guardrails보다 자연어 규칙 88개 중 더 큰 범위를 커버하며 차단 성능을 보입니다. 특히 Adversarial 조건에서 Cedar Block Rate가 크게 높게 나타나, MCP 서버를 통과한 일부 위반도 Cedar의 추가 deny로 상당 부분 차단됨을 확인했습니다. 저자들은 이 효과가 단순 trajectory block 비율보다 정책 범위 확장과 관련이 크다고 해석하며, 감사(audit) 가능성과 형식 검증 기반 신뢰를 분야에 제공하는 의미가 있다고 강조합니다.



### LLM-based Models for Detecting Emerging Topics in Service Feedback (https://arxiv.org/abs/2606.26595)
- **Prior Approaches**: 기존 공공 서비스 피드백 분석은 수작업 검토나 정적(사전 정의) 지표에 의존하는 경우가 많아, 대규모·다언어 텍스트에서 복잡한 패턴과 잠재적 격차를 놓치기 쉽습니다. 또한 일반 목적 사전학습 모델을 그대로 쓰거나, 도메인 특화 파인튜닝·경량화가 부족해 문맥 정확도와 운영 효율이 동시에 충족되지 않는 한계가 제기돼 왔습니다. 토픽 모델링 역시 전문가가 정한 범주로의 정밀 분류보다는 대략적 주제 추출에 치우쳐 현업 적용과 연결이 약했습니다.

- **Core Contribution**: 이 논문은 세금 행정의 고객(납세자) 피드백을 대상으로, Service Quality Elements(SQEs)라는 전문가 범주에 매핑하고 인구집단별로 ‘emerging·persistent·disappearing’ 같은 추세를 함께 탐지하는 프레임워크를 제안합니다. 핵심은 fine-tuned, quantized LLM을 사용해 다언어(영어/프랑스어) 텍스트의 맥락을 반영하면서도 계산 효율을 확보하고, 인간-AI 협업으로 결과 신뢰성을 높인다는 점입니다. 특히 직접 ‘편향(bias)’을 판정하기보다는 집단 간 토픽 빈도 차이와 추세를 근거로 공정성 점검에 활용할 수 있게 설계했습니다.

- **Technical Challenges**: 첫째, 세금 피드백에는 PII가 포함될 가능성이 커 개인정보 보호와 편향 최소화를 위한 de-identification이 필수였고, Transformer 기반 NER(XLM-RoBERTa 계열)와 규칙 기반 패턴 매칭, 그리고 전문가 검토를 결합해 처리했습니다. 둘째, LLM의 오답/환각(confabulation) 위험을 줄이기 위해 Tax Service 담당자가 출력 산정과 검증(Agent Validation)에 참여하는 human-in-the-loop 파이프라인을 구축했습니다. 셋째, 집단별 ‘최근 등장/지속/사라짐’ 같은 변화 신호를 정량화하기 위해 회귀 기반 회귀 모델로 추세를 분석해 설명 가능성을 높였습니다.

- **Empirical Impact**: 평가에서는 유사도(similarity) 분석과 숙련 세무 담당자의 판단 설문을 함께 사용해, 제안 모델이 baseline 대비 전문가 판단과 더 잘 맞는 정렬(alignment)을 보였다고 보고합니다. 또한 인간 검토를 포함함으로써 LLM의 무근거 생성 가능성을 낮추고, 생성 인사이트의 실용성과 맥락 적합성을 개선했습니다. 공공 부문에서 다언어 고객 피드백을 증거 기반으로 확장·운영할 수 있는 responsible AI 접근의 한 사례로, 서비스 품질과 공정성·공공 신뢰 향상에 기여할 수 있음을 시사합니다.



### Content-Based Smart E-Mail Dispatcher Using Large Language Models (https://arxiv.org/abs/2606.26593)
- **Prior Approaches**: 기존에는 이메일을 사람이 직접 확인한 뒤, 메시지와 첨부파일을 다른 인스턴트 메신저(예: WhatsApp)로 재전송하는 방식이 주로 쓰였다. 이 과정은 누락·오분류 같은 오류가 발생하기 쉽고 시간이 많이 걸려 생산성 저하와 담당자 스트레스로 이어진다.

- **Core Contribution**: 이 논문은 이메일의 내용을 자동으로 해석해 해당 학생 WhatsApp 그룹으로 보내는 디스패처 메커니즘을 제안한다. 특히 레이블드 데이터셋에 의존하지 않고도 LLM 기반 에이전트가 어떤 그룹에 전달해야 하는지 판단해 정보 흐름을 원활히 만든다고 강조한다.

- **Technical Challenges**: 핵심 기술 난제는 방대한 이메일 텍스트에서 의미를 파악해 적절한 그룹을 정확히 라우팅하는 것이다. 이를 위해 이메일 본문을 입력으로 하고 지시·문맥을 포함한 well-structured agent framework prompt를 구성해, 에이전트가 텍스트 기반 의사결정을 수행하도록 설계했다.

- **Empirical Impact**: 제안된 시스템은 레이블드 데이터셋 없이도 작동하는 점에서 실용성을 높이며, 수작업 전송 대비 생산성 향상과 이메일 읽기에서의 인지 부하 감소 효과를 제공한다고 보고한다. 조직 내 정보 전달의 지연과 오류를 줄여 운영 효율을 개선하는 방향의 AI 뉴스레터급 활용 사례로 의미가 있다.



### A Multi-Level Validation and Traceability Framework for AI-Generated Telescope Scheduling Decisions (https://arxiv.org/abs/2606.26585)
Comments:
          25 pages, 8 figures, Published in Universe

- **Prior Approaches**: 기존 망원경 스케줄링 연구는 규칙 기반, 휴리스틱, 강화학습, 그래프 기반 방법 등으로 다중 목표·다중 제약을 다뤄왔습니다. 다만 AI/LLM 기반 접근은 자연어 출력에서 데이터 참조가 흔들리거나 논리 추론 오류, 실행 불가능한 결정이 섞여 신뢰성 확보가 어렵다는 한계가 있었습니다. 그래서 고신뢰 관측 운영에서는 AI 출력을 그대로 실행 파이프라인에 넣기 어려웠습니다.

- **Core Contribution**: 이 논문은 AI가 만든 스케줄링 제안을 “실행 전” 단계에서 다층 검증하고, 추적 가능한 추론 표현을 제공하는 프레임워크를 제안합니다. 특히 데이터 참조 검증, 논리 일관성 체크, 관측·기기 하드 제약 검증을 통합해 불유효 결정을 차단하고, 실패 시 피드백으로 수정이 가능하게 합니다. 또한 원자 단위 추론(Atomic Reasoning Units)과 그 의존관계를 DAG로 표현해 오류 위치 파악과 사후 분석(audit)을 돕습니다.

- **Technical Challenges**: 핵심 기술 과제는 LLM의 자연어 기반 제안이 자동 검증에 적합하지 않고, 신뢰성 문제를 모델 생성 단계에서 완전히 통제하기 어렵다는 점입니다. 이를 위해 제안서를 claim/argument/evidence 구조로 재구성하고, 원자 추론 단위마다 필요한 데이터 필드와 텔레스코프 상태 딕셔너리 기반 표준 제약을 강제해 데이터 참조 일관성과 표현 수렴을 유도합니다. 이어 의존성 DAG의 무순환성(acyclic)을 활용해 단계 수준(step-level)에서 국소적으로 검증·차단하고, 실패 사유를 구조화해 다음 생성 입력으로 되돌립니다.

- **Empirical Impact**: 실험에서는 제안 프레임워크가 AI 스케줄링의 실행 가능성(pass rate)을 높이고, 비실행/오류 결정을 줄여 일시적 기회(transient opportunities) 손실을 완화함을 보였습니다. 예를 들어 phi-4(14B)는 평균 pass rate 79.3%(1231건), Qwen(30B)은 84.5%(1202건)를 기록했으며, 복잡하고 입력이 긴 multi-ToO 상황에서 효과와 실패 양상이 더 뚜렷했습니다. 결론적으로 “순수 AI 대비 유연성은 유지하면서 신뢰성과 실행 가능성을 크게 개선”하는 경로를 제시해 고신뢰 천문 관측 스케줄링의 실용성에 의미가 있습니다.



### EvoOptiGraph: Weakness-Driven Coevolution via Graph-Based Structural Generation for Optimization Modeling (https://arxiv.org/abs/2606.26578)
- **Prior Approaches**: 자연어에서 최적화 모델(특히 MILP)을 자동 생성하는 기존 연구들은 대체로 정적 데이터셋과 단일 파이프라인에 의존해 구조적 다양성이 부족하다는 한계가 컸다. 또 생성 파이프라인이 학습 과정과 decoupled 되어 있어, 모델이 실제로 틀리는 유형을 겨냥한 데이터 보강이 잘 이뤄지지 않았다.

- **Core Contribution**: EvoOptiGraph는 MILP를 attributed bipartite graph로 표현하고, 유효성 보존형 evolutionary operators로 구조적으로 다양한 인스턴스를 만든다. 이후 deterministic compilation과 verified back-translation으로 그래프를 solver code와 자연어로 안정적으로 변환하며, SFT 이후 RLVR로 모델의 약점을 겨냥해 학습 데이터 분포를 지속적으로 갱신하는 closed loop를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 그래프 변형이 생성된 인스턴스의 제약 구조를 망치지 않으면서 다양성을 늘리는 것, (2) LLM 학습용 목표(코드/자연어)로의 변환이 일관성과 검증 가능성을 유지하는 것이다. EvoOptiGraph는 validity-preserving 진화 연산과 deterministic compilation, verified back-translation을 조합해 변환 오류를 줄이고, graph-derived weakness 신호를 RLVR의 verifiable reward로 연결해 실패 유형 타깃 생성을 유도한다.

- **Empirical Impact**: 여섯 개 공개 데이터셋에서 EvoOptiGraph는 더 큰 generalist 모델, agentic 방법, specialized baselines보다 정확도·executability·일반화 성능에서 유의미하게 우수했다. 이는 데이터-모델 coevolution을 최적화 모델링 생성 학습에 적용하는 전략이 성능 향상에 효과적임을 보여주며, 최적화 에이전트와 자동 모델링 파이프라인의 실용성에도 긍정적 의미가 있다.



### Explainable Ensemble-Based Machine Learning Models for Detecting the Presence of Cirrhosis in Hepatitis C Patients (https://arxiv.org/abs/2606.26561)
- **Prior Approaches**: 간경변은 간경화로의 진행을 막는 것이 치료의 핵심이어서, 조기 탐지가 중요하지만 기존 임상 진단은 상대적으로 늦게 이루어질 수 있다. 그동안 ML은 여러 질병에서 진단 성능을 높이는 데 효과가 입증됐지만, B형간염이 아닌 C형간염 환자에서 간경변을 ML로 직접 탐지한 연구는 보고되지 않았다.

- **Core Contribution**: 이 논문은 C형간염 환자에서 간경변을 조기에 진단하기 위한 ML 접근을 제안한다. UCI ML Repository의 2038명 이집트 환자 데이터(28개 속성)를 사용해 간경변 여부를 분류하는 모델을 학습하고, 특히 일부 특징만으로도 성능을 끌어올릴 수 있음을 보인다.

- **Technical Challenges**: 핵심 과제는 제한된 임상 속성으로 간경변을 정확히 구분하는 동시에, 불필요한 특징을 줄여 실무 적용성을 높이는 것이다. 이를 위해 Random Forest, Gradient Boosting Machine, Extreme Gradient Boosting, Extra Trees 등 네 가지 모델을 비교했고, Extra Trees에서 28개 중 16개 특징만으로도 높은 성능을 달성하도록 설계·평가했다.

- **Empirical Impact**: 실험 결과 Extra Trees가 가장 좋은 성능을 보였으며, accuracy 96.92%, recall 94.00%, precision 99.81%, ROC AUC 96%를 기록했다. 또한 전체 28개 특징이 아니라 16개 특징만으로도 높은 정확도를 유지해, 향후 C형간염 환자 조기 선별과 임상 의사결정 보조에 의미 있는 실증 결과를 제공한다.



### PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting (https://arxiv.org/abs/2606.26549)
- **Prior Approaches**: Transformer 기반 LTSF는 패치 전략으로 장기 의존성을 포착하지만, 패치 간/변수 간 shape 유사성이 스케일 차이로 왜곡되기 쉽다. 기존 Patch Normalization류는 분포를 안정화하는 대신 표준편차 스케일링이 패치의 고유 모양을 훼손해 진짜 shape similarity 학습을 방해할 수 있다. 또한 변수 의존(VD) 모델들은 전체 히스토리를 한꺼번에 상호작용에 쓰는 경우가 많아, 비정상성 변화에 따른 노이즈·중복이 늘며 성능 저하로 이어질 수 있다.

- **Core Contribution**: 논문은 Patch-Mean Decoupling(PMD)으로 각 패치에서 평균(장기 trend)을 분리하고 잔차의 shape 정보에 집중하도록 만들어, attention이 스케일이 아닌 모양 정렬을 더 정확히 반영하게 한다. 여기에 Trend Restoration Attention(TRA)으로 PMD로 분리된 trend를 value 경로에 재주입해 global dynamics를 복원하고, Proximal Variable Attention(PVA)으로 변수 간 attention을 예측에 가장 가까운 최근 패치에만 제한해 시간에 따른 의존성 변화를 반영한다. 이를 통합한 PMDformer는 shape similarity 중심의 장기 예측을 목표로 설계됐다.

- **Technical Challenges**: 핵심 난제는 (1) 패치 스케일 편향 때문에 attention이 ‘모양 유사도’가 아닌 ‘크기 유사도’를 우선 학습할 위험, (2) trend 신호를 분리하면 global dependency가 약해지는 문제, (3) 변수 간 비정상적 상관 때문에 과거 상호작용이 노이즈로 섞이는 문제였다. PMD는 표준편차 정규화 대신 패치 평균만 빼 shape 구조를 보존하며 편향을 줄이고, TRA는 Q/K 경로는 shape에, V 경로는 decoupled mean에 기반해 trend을 residual처럼 복원하는 방식으로 설계했다. PVA는 cross-variable self-attention을 최근 패치 토큰으로 한정해 과거 상관의 과적합을 낮추고 계산 복잡도도 줄인다.

- **Empirical Impact**: 실험은 ETT 계열과 ECL, Weather, Solar, Traffic 등 8개 LTSF 벤치마크에서 MSE/MAE 지표로 진행됐고, PMDformer가 여러 예측 구간(예: 96~720)에서 안정성과 정확도 모두 기존 SOTA들을 앞선다고 보고한다. 특히 patch-mean decoupling에 의해 attention이 shape-aligned 쌍에 더 집중하는 정성적 관찰과 함께, trend 복원 및 근접 변수 attention이 장기 예측 품질을 균형 있게 끌어올린 점이 강조된다. LTSF에서 ‘scale 편향을 제거하면서도 trend를 잃지 않는’ 설계 아이디어가 실용적으로 유의미하다는 평가다.



### Radical AI Interpretability (https://arxiv.org/abs/2606.26523)
Comments:
          Draft of manuscript to appear as Cambridge Element in the Philosophy of Artificial Intelligence

- **Prior Approaches**: 기존 해석가능성 연구는 모델 내부에서 beliefs와 desires를 “읽어내는” 도구를 만드는 데 집중해 왔지만, 어떤 조건에서 그 도구가 성공했다고 판단할지에 대한 정착된 설명은 부족했습니다. 또한 시스템이 가진 의미를 개별 구성요소 단위로 고정하려는 접근이 많아, 실제로는 서로 얽힌 제약관계를 충분히 다루지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 AI 시스템을 agents로 해석하고, 주어진 계산적 사실로부터 beliefs, desires, 그리고 의미를 어떻게 도출(또는 검증)할지에 대한 틀을 제안합니다. 더 나아가 representationalist와 interpretationist 관점에서 성공 기준을 제시하고, 현재의 해석가능성 방법이 수행 가능한 테스트로 이를 연결합니다. 핵심 메시지는 이러한 귀속(attribution)이 조각조각 처리될 수 없다는 holism입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 특정 belief나 desire만 먼저 맞힌 뒤 나머지를 측정하면, 한 축을 고정하는 과정에서 생기는 왜곡이 다른 축으로 전파된다는 점입니다. 논문은 beliefs·desires 및 그것들이 전제하는 propositional structure가 함께 제약된다고 보고, 둘 사이의 상호 일관성을 함께 측정하도록 테스트를 설계하는 방식으로 이를 해결합니다. 또한 AI가 해석자의 개념과 다를 수 있다는 점을 인정하면서, attitudes와 propositional structure 간의 제약 고리를 메커니즘 해석가능성으로 포착합니다.

- **Empirical Impact**: 제시한 성공 기준과 테스트 매핑은 기존 해석가능성 도구가 언제 신뢰 가능한 귀속을 산출했는지 판정할 수 있는 준거를 제공한다는 점에서 안전 분야에 직접적 의미가 있습니다. 특히 목표 이해의 신뢰뿐 아니라 deception 탐지의 근거를 더 체계적으로 마련하는 데 기여할 수 있습니다. 결과적으로 “말로만 해석”이 아니라, 측정 가능한 상호 제약을 통해 해석의 신뢰도를 높이는 방향을 정립합니다.



### Boundary-Aware Context Grounding for A Low-Channel EEG Agen (https://arxiv.org/abs/2606.26519)
Comments:
          25 pages, 6 figures

- **Prior Approaches**: 기존 EEG 분석 생태계(MNE-Python, EEGLAB 등)는 다양한 채널 수와 프로토콜을 지원하지만, 사용자에게 방법 선택 부담이 크고 에이전트가 “현재 소프트웨어가 실제로 무엇을 구현했는지”를 기계적으로 보장하진 못한다. LLM 기반 접근이나 retrieval/tool-use는 외부 문서나 함수를 참조할 수 있으나, 과학적 안전성 관점에서 필요한 하드웨어·구현·결과·과학적 경계(boundary awareness)를 정확히 판별하는 데 초점을 둔 평가 설계는 부족했다. 또한 LLM 환각 연구가 주로 불일치/거짓에 맞춰져 있어, 과학 에이전트에서 특히 중요한 over-refusal(과도한 거절)과 conditional 응답의 품질을 함께 측정하기가 어려웠다.

- **Core Contribution**: NeuraDock Agent는 결정론적(deterministic) 로컬 EEG 엔진과, 하드웨어를 인지하는(constrained) 언어 계층을 분리해 “수치의 진실(numerical truth)”을 LLM이 변경하지 못하도록 설계했다. LLM은 원시 EEG나 고밀도 배열을 받지 않고, allowlisted 요약과 버전이 명시된 context pack만 사용해 해석/계획을 수행한다. 특히 seven-channel 저채널 EEG(예: CP5, CP6, PO3, PO4, O1, Oz, O2)와 workflow 레지스트리(리뷰된 Python 코드) 및 결과 필드 스키마를 문맥으로 고정해, 장치·구현·결과·과학적 정당화 경계를 한 시스템 안에서 운영 가능하게 만든 것이 핵심이다.

- **Technical Challenges**: 어떤 센서/몽타주가 관측 가능한지, 현재 리뷰된 소프트웨어가 어떤 분석을 실제로 제공하는지, 결과 필드가 무엇을 “보고(report)”하는지, 그로부터 어떤 과학적 추론만 정당화되는지(그리고 그 구분이 혼동되면 저채널 EEG에서 특히 그럴듯하지만 근거 없는 해석이 쉬움)라는 다층 경계 문제를 안정적으로 통제해야 했다. 이를 위해 아키텍처는 workflow별 projection(허용된 결과 필드만 요약 전송)과 “raw_eeg_included=false” 같은 명시적 제한, 그리고 LLM이 생성한 파이썬 실행/필터·임계값·통계 변경을 할 수 없도록 하는 실행 분리를 채택했다. 또한 실패 시나리오(HTTP 오류, malformed output, 연결 거부)에서도 로컬 아티팩트(report, trace, status)가 보존되도록 요청 라우팅과 오프라인 기준 진실 경로를 분리해 견고성을 확보했다.

- **Empirical Impact**: 평가는 (1) 동일 기록을 여러 번 처리해 구조화된 결과/리포트/그림 해시가 재현되는 수치 반복성, (2) request-capture 및 failure-injection으로 데이터 경계와 로컬 아티팩트 보존을 확인한 안정성, (3) 36-case boundary-awareness 벤치마크에서 안전한 수용/거절 캘리브레이션을 측정하는 방식으로 구성됐다. 그 결과 정확한 4-way decision(지원/조건부/지원 불가/미구현)과 required-fact recall이 context를 풍부하게 할수록 단조 증가했으며, 예를 들어 Generic→Full context에서 exact decision accuracy는 58.3%→79.2%, strict safe-response는 26.4%→66.7%로 개선되었다. 다만 이 성과는 임상 유효성이나 절대적 cognitive-load 지표의 검증을 뜻하지 않으며, 특히 flatline처럼 탐지 품질이 애초에 설계에 포함되지 않은 경우의 한계를 “구체적 부정 결과”로도 드러내 향후 확장 방향을 제시했다.



### NeuraDock Visual Cognitive Load Agent Tutorial: A Quality-Gated Open-Source EEG Workflow for Alpha Dynamics and Real-Time Applications (https://arxiv.org/abs/2606.26518)
Comments:
          22 pages, 10 figures

- **Prior Approaches**: 기존 EEG 도구(MNE-Python, EEGLAB, BrainFlow 등)는 오프라인 분석에는 강하지만, 전처리→커스텀 품질관리(QC)→Alpha 특징 추출→웹 API로 이어지는 실시간 워크플로를 “완결된 하나의 파이프라인”으로 제공하진 못한다. 상용/디바이스 SDK는 실시간 데이터 연결은 제공하지만, 로컬에서 감사를 거칠 수 있는 형태로 인지부하 지표까지 투명하게 묶기 어렵다. 그 결과 연구실 데모는 가능해도, 초당 호출되는 품질 게이팅 서비스로 배포하는 데는 인프라 공수가 크게 든다.

- **Core Contribution**: NeuraDock Agent는 시각 인지부하(visual cognitive-load) 프로토타입을 목표로 EEG 파일에서 실시간 API까지 가는 재현 가능한 튜토리얼 워크플로를 제공한다. 핵심 설계는 downstream 지표(Alpha dynamics, workload index)를 원시 EEG가 아니라 “전처리+QC 통과 데이터”에 대해서만 계산하는 quality-gated 접근이다. 이로써 점수만 던지는 형태가 아니라, 해당 점수가 신뢰 가능한 데이터 품질 위에 서 있는지를 함께 보여준다.

- **Technical Challenges**: 실시간 파이프라인에서는 QC와 특징 추출이 동시에 돌아가야 하고, 신호 품질이 나쁘면 해석이 흔들리므로 “지표 계산 자체를 게이트”하는 구현이 필요하다. 이 논문은 report.md와 results.json, clean/QC-gated 출력으로 유지율·경고·거부 구간을 명시하고, 온라인 대시보드(/api/status)에서도 품질 플래그를 함께 내보내 애플리케이션이 게이트 기반으로 동작하도록 한다. 또한 NeuraDock의 7채널 구성과 posterior Alpha용 채널 그룹핑, 4초 롤링 윈도/250Hz 스트림 파싱 등 하드웨어-튜닝된 가정을 함께 제공해 실사용 연결성을 확보한다.

- **Empirical Impact**: 공개 미니데이터셋 검증에서 18개 레코딩을 처리해 10개의 within-subject Rest/Task 비교를 생성했고, 그중 7/10에서 태스크 시 posterior Alpha 억제가 관측됐다. 특히 일부 대비에서는 Task/Rest Alpha ratio가 약 0.5 수준(예: 0.50, 0.52)으로 나타나 예상 방향성과 정합성이 확인됐다. 또한 Rest 세션 재측정에서 median posterior log Alpha의 Pearson r=0.803, ICC(C,1)=0.765로 within-subject 반복성의 초기 근거를 제시했으며, 온라인 API는 로컬 데모 기준으로 코어 처리 지연 p95가 수 ms~수십 ms 수준(예: median 15.15ms, p95 27.18ms)으로 보고된다.



### Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation (https://arxiv.org/abs/2606.26502)
- **Prior Approaches**: 기존 연구는 LRM의 추론 길이(생각 토큰 길이)가 사람의 반응시간과 맞물리는지에 주로 초점을 맞춰, 항목 간 난이도 정렬(registration)을 보여줬습니다. 하지만 이러한 상관은 메커니즘이 아니라 관측된 현상일 수 있고, 실패·성공에서 계산을 어떻게 배분하는지(allocation)는 직접 답하지 못한다는 비판이 이어졌습니다.

- **Core Contribution**: 이 논문은 기존 ‘항목 간 난이도 정렬’과 별개로 ‘항목 내 계산 배분’을 진단하는 방법을 제시합니다. 구체적으로 각 에이전트가 자신의 실패(wrong)와 성공(right)에 대해 더 오래 숙고하는지를, 각 에이전트 내부 스케일에서 정규화하고 item identity를 고정한 상태(item fixed effects)로 평가합니다. 이를 통해 사람과 LRM이 겉보기에는 유사해 보이던 정렬이 실제론 다른 중단 정책을 가질 수 있음을 분리해 드러냅니다.

- **Technical Challenges**: 핵심 어려움은 토큰 길이(또는 출력 길이)와 반응시간(초)을 직접 축에 올려 비교하면 생기는 혼동을 피하는 것입니다. 논문은 seconds와 tokens의 절대 단위를 비교하지 않고, 에이전트별 고유 길이 읽out을 그대로 두되 correct 여부에 따른 기울기(틀릴 때 더 길어지는지)를 item fixed effects로 추정해 항목 난이도 기울기를 흡수합니다. 또한 비-thinking 기준모델(DeepSeek-V3)과 여러 추론 패러다임(H-ARC, INTUIT, Cortes)으로 진단이 특정 설계에만 의존하는지 경계합니다.

- **Empirical Impact**: 공개된 매칭 휴먼-LRM 코퍼스에서 모든 thinking LRM은 ‘틀리면 더 길어지는’ 큰 wrong-vs-right 효과(d≈1.47~3.13, H-ARC)를 보인 반면, 사람은 반대 부호(engagement-vs-abandonment 신호로 해석)로 나타났습니다. 이 분리는 item fixed effects에서도 유지되고 다른 데이터셋/패러다임으로도 재현되며, thinking이 아닌 기준모델에서는 같은 패턴이 약하거나 나타나지 않습니다. 결과적으로 기존 상관 측정에서 보이던 ‘난이도 정렬’은 맞지만, ‘실패에 더 쓸지/덜 쓸지’라는 제어 정책은 사람과 LRM이 반대로 작동한다는 점이 실증적으로 제시됐습니다.



### Clinical Harness for Governable Medical AI Skill Ecosystems (https://arxiv.org/abs/2606.26494)
- **Prior Approaches**: 기존 의료 AI는 대체로 특정 태스크에 한정된 단일 모델을 중심으로 발전해 왔지만, 실제 임상은 시간에 걸쳐 지속되고 책임성 있는 능력이 요구된다. 그 결과 모델이 배포된 이후에도 성능 저하, 거버넌스 공백, 안전성/책임성 확보 같은 운영 이슈를 일관되게 다루기 어렵다는 한계가 있었다. 

- **Core Contribution**: 논문은 ‘clinical AI skills’라는 개념을 제시하고, 이를 등록·오케스트레이션·가드·모니터링하는 실행 런타임 거버넌스 아키텍처인 ‘Clinical Harness’를 제안한다. 골다공증을 예시로, 지식 기반·데이터 기반·physics-enhanced(물리 강화) 스킬들을 결합해 수명주기(lifecycle) 관점의 케어를 지원하는 구성을 보인다. 

- **Technical Challenges**: 핵심 기술적 도전은 서로 다른 스킬을 신뢰성 있게 실행하고, 임상 런타임에서 안전장치와 책임성 있는 거버넌스를 지속적으로 유지하는 것이다. 이를 위해 Clinical Harness가 스킬 레지스트리와 오케스트레이션, 가드(제한/보호), 모니터링을 런타임에 통합해 ‘능력의 지속성’을 보장하도록 설계했다고 설명한다. 

- **Empirical Impact**: 골다공증 사례를 통해 knowledge-driven, data-driven, physics-enhanced 스킬이 수명주기 케어 흐름을 지원할 수 있음을 실증적으로 보여준다. 의료 AI가 단일 모델을 넘어서 운영·책임·지속성까지 포괄하는 ‘임상 능력’ 프레임으로 확장될 수 있다는 점에서 의미가 크다. 



### auto-psych: Automating the science of mind using agent-driven theory discovery and experimentation (https://arxiv.org/abs/2606.26460)
Comments:
          30 pages, 5 figures

- **Prior Approaches**: 기존 AI 과학 자동화는 주로 “Box’s loop”처럼 이미 존재하는 데이터에서 모델을 제안·학습·비판하는 데 집중해 왔다. 반면 심리학처럼 신규 인간 데이터가 필수인 영역은 lab-in-the-loop 구조 때문에 반복 속도가 느렸다. 또한 computational cognitive science에서는 확률적 인지모델을 코드로 표현해 비교하는 시도는 있었지만, 실제 실험 설계·배포·데이터 수집까지 하나의 루프로 통합된 사례는 드물었다.

- **Core Contribution**: 이 논문은 computational cognitive science에서 가설→모델 비판→실험 설계·런칭→데이터 분석까지 이어지는 end-to-end 이론 탐색 프레임워크 auto-psych를 제안한다. 핵심은 중첩된 두 개의 discovery loop로, 바깥 루프가 온라인 실험을 통해 모델들을 가려내고 안쪽 루프가 Box’s loop 형태로 인지모델을 정밀하게 개선한다. 코인 플립의 주관적 randomness 판단 문제를 테스트베드로 삼아, 사람이 만든 문헌 기반 초기 모델보다 더 잘 맞는 이론을 탐색하는 능력을 보인다.

- **Technical Challenges**: 가장 큰 기술적 난관은 “모델 성능을 올리기 위한 실험을 자동으로 설계·운영”하는 데이터 수집 병목을 해소하는 것이다. 이를 위해 외부 에이전트가 jsPsych로 jsPsych 기반 온라인 실험을 구현하고 Prolific API로 참가자를 모집·배포하며, EIG(expected information gain)로 어떤 자극 쌍을 제시할지 선택한다. 내부 루프는 CriticAL에서 영감을 받은 critic이 차이를 드러낼 통계량을 만들고, p=0.05 기준으로 유의한 불일치를 근거로 PyMC probabilistic program 형태의 모델을 반복 개량한다.

- **Empirical Impact**: 합성 데이터 실험에서 auto-psych는 ground-truth 모델을 안정적으로 회수했으며, inner loop를 제거한 변형에서는 성능이 유의하게 하락해 중첩 구조의 중요성이 확인됐다. 또한 인간 참가자 3개 독립 시퀀스(총 9개 라운드)에서 발견된 모델들은 문헌 seed 모델보다 더 잘 맞는 것으로 나타났고, R2 기준 잡음 한계 대비 설명 분산을 약 83%까지 포착했다. 특히 Minkowski typicality, evidence accumulation, Bayesian diagnosticity 변형 등 서로 다른 내부 구조의 모델도 후보로 등장해, 자동 탐색이 인지 이론의 재정의를 실제로 촉진할 가능성을 보여준다.



### MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph-Augmented Generation (https://arxiv.org/abs/2606.26458)
Comments:
          Accepted by KDD'26

- **Prior Approaches**: 기존 RAG는 주로 비정형 텍스트에서 관련 정보를 찾고 생성에 근거를 제공하는 방식이었지만, 잡음·분절·약한 연결성 때문에 증거 선택과 다단계 추론이 흔들릴 수 있다. KG-RAG는 구조화된 지식그래프에서 triplet/서브그래프를 가져오지만, 기존 벤치마크와 방법들은 주로 텍스트 중심이라 이미지·차트·표 같은 멀티모달 정보의 검색·정렬 문제를 제대로 평가하지 못한다. 또한 multimodal RAG는 멀티모달을 다루더라도 구조화된 멀티모달 knowledge graph에서의 retrieval은 분리해 진단하기 어렵다.

- **Core Contribution**: 이 논문은 multimodal knowledge graph RAG(MKG-RAG)에서 ‘검색(retrieval) 자체’의 병목을 정면으로 평가하기 위한 신규 벤치마크 MKG-RAG-Bench를 제안한다. 벤치마크는 일반 도메인 MarKG와 의료 도메인 MedMKG 두 개의 멀티모달 knowledge graph에서 질문-답변을 만들고, retrieval 품질과 downstream 생성 품질을 통제된 형태로 함께 측정할 수 있게 설계됐다. 특히 cross-domain 구성과 exact supervision을 통해 retrieval과 생성의 기여를 분해 진단하는 것이 핵심이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 쿼리와 타깃이 서로 다른 modality 조합을 가질 수 있어 ‘heterogeneous retrieval’이 필요하다는 점, (2) 표면 수준에서 정렬이 드러나지 않는 cross-modal/cross-structural alignment가 요구된다는 점이다. 이를 위해 논문은 LLM 기반 curation으로 저유용 triplet을 제거하고, triplet masking(관계 또는 꼬리 마스킹)으로 구조적으로 정렬된 질의를 만들며, 시각적 grounding이 필요할 때는 이미지를 질문에 포함해 단순 문자열 매칭을 막는다. 또한 질문-정답이 마스킹된 성분과 정확히 대응하도록 QA를 생성해, 생성이 단순 사전지식이 아니라 검색된 증거의 충실성에 의존하도록 감독한다.

- **Empirical Impact**: 훈련 없이 대표적인 retriever 계열(텍스트, fusion 기반, captioning 기반, reranking 기반)을 다양한 modality 설정에서 광범위 평가한 결과, 멀티모달 retrieval은 여전히 어렵지만 end-to-end MKG-RAG 성능에 결정적으로 중요하다는 결론을 도출한다. 특히 retrieval이 시각적 grounding을 제공해야 하는 구간에서는 텍스트 전용 방식이 급격히 무너지고, captioning 기반은 캡션 손실이 retrieval 오류로 전이되는 경향이 관찰됐다. 저자들은 retrieval 품질이 생성 결과를 강하게 좌우하며, 도메인(일반 vs 의료)별로 요구되는 검색 전략(예: 의료에서는 reranking 선호)도 달라진다고 보여 MKG-RAG 연구의 다음 설계 방향을 제공한다.



### Data-driven Machine Learning Cannot Reach Symbolic-level Logical Reasoning -- The Limit of the Scaling Law (https://arxiv.org/abs/2606.26454)
- **Prior Approaches**: 기존 연구는 syllogistic reasoning을 LLM의 next-token 기반 추론, single-step 정확도 벤치마크, 또는 Chain-of-Thought 같은 기법으로 끌어올리려 했다. 하지만 대부분의 모델은 표면형(단어/기호 등) 변화에 민감하고, 훈련·추론 목표가 논리의 엄밀성까지 보장하지 못해 OOD(의도치 않은 입력)에서 오류가 반복된다는 평가가 많다. 한편 Euler Net 같은 image-input supervised 방식은 거의 100%에 도달한 사례가 있으나, 그 성능이 정말로 symbolic-level “rigour”로 수렴하는지에는 한계가 있었다.

- **Core Contribution**: 이 논문은 supervised deep learning이 symbolic-level syllogistic reasoning에 도달하지 못하는 이유를 두 가지로 체계화한다. 첫째, 학습 데이터가 24가지 유효 syllogistic 유형을 모두 구분해 주지 못해 동일 입력 조건에서 서로 다른 결론을 학습 신호가 명확히 분리하지 못한다. 둘째, end-to-end 파이프라인은 패턴 인식 구성요소가 “premise에 없는” 객체를 주입하더라도 추론 구성요소가 이를 차단·감지할 수 없어서, 논리 목표와 패턴 목표 사이에 모순되는 학습 타깃이 생긴다고 지적한다.

- **Technical Challenges**: Euler Net은 Euler diagram의 네 가지 기본 set relation으로 syllogism을 압축하는데, 그 조합(조합표) 구조가 모든 유효 유형을 구획하지 못해 특정 유형들의 판별이 흐려진다. 또한 end-to-end 학습에서는 Siamese류 인식 모듈이 입력에 없던 빨강/파랑 같은 원을 “복원”할 수 있고, 추론 모듈은 그런 의도치 않은 생성물에 대해 논리적으로 일관된 결정을 내리기 어렵다. 저자들은 이를 SupEN(Super Euler Net)으로 확장해, 추론 오류를 자동 탐지해 새 학습 데이터를 생성하며 데이터·시간을 무한히 늘릴 때도 엄밀성에 수렴하지 않는지를 반복 실험으로 확인한다.

- **Empirical Impact**: 이 논문 실험에서 SupEN은 무작위로 생성한 테스트셋에서는 반복 학습으로 정확도가 56%에서 최고 97.8%까지 오르며 scaling law 경향을 일부 보인다. 그러나 24가지 유효 유형을 포괄한 더 엄밀한 평가에서는 100%에 도달하지 못하고 overall 76% 수준에 머물며, 일부 유형은 50%~83.3%로 크게 흔들렸다. 또한 GPT-5와 GPT-5-nano를 단어/이중 단어/단순 기호/랜덤 기호의 네 표면형으로 시험한 결과, 정확도는 높게 나와도 설명이 틀릴 수 있고 표면형에 따라 추론 성능이 달라진다는 점을 통해 “훈련 절차가 100% 정확도에서 조기 종료되는 관성”이 symbolic rigour 달성을 막는다고 결론낸다.



### Estimating Uncertainty in Classifier Performance with Applications to Large Language Models and Nested Data (https://arxiv.org/abs/2606.26422)
- **Prior Approaches**: 기존에는 텍스트 분류에서 precision, recall, F1 같은 성능 지표를 보고하지만, 그 주변의 uncertainty(표준오차/신뢰구간)는 사회과학 연구에서 일관되게 보고되지 않는 경우가 많다고 지적한다. 또한 보고되는 신뢰구간이더라도 Wald(정규근사)나 basic percentile bootstrap처럼 작은 표본·높은 성능(비율이 0 또는 1에 가까움) 조건에서 부정확해질 수 있는 방법을 그대로 쓰는 일이 잦다. 특히 텍스트가 개인(저자/참여자) 안에 중첩되는 비독립 구조에서는 이런 방식이 더 취약하다.

- **Core Contribution**: 이 논문은 사회과학 텍스트 분류에서 흔한 조건(작은~중간 표본, 드문 construct, 개인 단위 중첩)에 맞춰 성능 지표의 신뢰구간을 어떻게 추정·보고해야 하는지 체계적으로 평가한다. 시뮬레이션을 통해 Wald, 기본 percentile bootstrap이 실제 커버리지(명목 95% 대비)를 크게 놓칠 수 있음을 보여주고, 대안으로 Agresti-Coull, Wilson, Clopper-Pearson, 그리고 F1에 특히 유용한 pseudo-count regularized bootstrap을 제안한다. 또한 중첩 데이터에서는 effective N과 자유도 조정이 정확한 analytic interval 산출에 필수임을 강조한다.

- **Technical Challenges**: 핵심 기술적 난관은 (1) 성능 지표가 샘플링 변동을 갖는 point estimate라는 점, (2) recall/precision처럼 비율이 0·1 근처로 치우치면 전통적 근사 구간이 흔들린다는 점, (3) 텍스트가 개인에 클러스터링돼 독립성 가정이 깨진다는 점이다. 논문은 이 문제를 위해 비율 추정에서는 Wilson/Agresti-Coull/Clopper-Pearson처럼 극단 비율에서도 안정적인 analytic interval을 비교하고, F1처럼 비율 단일항이 아닌 지표에 대해서는 confusion matrix의 각 칸에 pseudo-count를 추가해 bootstrap의 불안정성을 완화하는 정규화 절차를 설계한다. 중첩 상황에서는 계층적(hierarchical) bootstrap이 cluster bootstrap보다 더 정확하지만, 개별이 생산하는 텍스트 수가 아주 적을 때는 보수적으로 변할 수 있음을 함께 정리한다.

- **Empirical Impact**: 시뮬레이션 결과, Wald interval과 기본 percentile bootstrap은 작은 표본 및 높은 비율에서 커버리지가 명목보다 크게 낮아지는 사례가 관찰됐다. 반면 Agresti-Coull, Wilson, Clopper-Pearson, 그리고 pseudo-count regularized bootstrap(특히 F1용 regularization)이 커버리지를 더 안정적으로 95%에 가깝게 유지하며 구간 폭도 과도하게 커지지 않는 편이다. 중첩 데이터에서는 effective N 및 적절한 degrees of freedom 조정이 없으면 analytic interval이 부정확해질 수 있어, 향후 ML/사회과학 텍스트 분류 연구의 설계 단계에서 validation sample size를 더 면밀히 고려하도록 촉진하는 의미가 있다.



### Unbiased Canonical Set-Valued Oracles Via Lattice Theory (https://arxiv.org/abs/2606.26418)
Comments:
          18 pages

- **Prior Approaches**: 예측형 “oracle”/Scientist AI는 행위가 아니라 확률 보고만 하므로 위험한 동기화가 생기지 않을 것이라는 기대에서 출발한다. 하지만 예측이 읽히는 순간 예측은 원인으로 작동해 결과를 바꾸는 performative prediction 문제가 발생하고, 이에 대응해 훈련 신호를 끊는 consequence invariance가 제안됐다.
또 하나의 방식은 답을 “학습하지 않는다는 가정” 아래에서만 묻는 counterfactual 질문인데, 논문은 이런 답이 실제로 학습되는 순간 전제가 깨져 곧바로 무의미해진다고 지적한다.

- **Core Contribution**: 논문은 고정된 단일 확률 대신, 학습(그리고 그 결과의 반영)까지 포함했을 때 동시에 편향되지 않고 자기일관적인 credal set(확률 추정의 집합)을 oracle이 보고하도록 하는 self-referential 대안을 제시한다. 다만 단순 자기일관 조건은 [0,1] 같은 무용한 해도 허용하므로, 이를 배제할 “정규(canonical)하고 비자명한” 해를 고르는 문제로 정식화한다.
해결책으로 닫힌 credal set들의 완비격자에서 Knaster–Tarski 고정점 정리를 쓰되, 적절한 isotone 연산자의 least fixed point를 선택해 자연스러운 정규해를 도출한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘확률값이 집합에 포함된다’는 membership 형태의 자기일관성을 기존 equality 기반 고정점 정리에 맞게 설계하는 것이다. 이를 위해 닫힌 집합들의 격자에서 F(C) = { P(B|A,C′) : C′ ⊆ C } 형태의 isotone 연산자를 구성하고, least fixed point μ_F의 존재·자기일관성·비공허성을 함께 증명한다.
또한 답이 영향이 없는 비performative 질문에서는 고전적인 점추정으로 환원되고, 이분 사건에 대해 자연스러운 hull-factoring 가정 아래 결과가 확률 구간(interval)으로 나타남을 보이며, 이틀림이 lattice-theoretic으로 임의 random variable X까지 확장됨을 논의한다.

- **Empirical Impact**: 이 논문은 실험 벤치마크를 제시하기보다, performativity 위험을 다루는 “확률 보고 규칙”의 존재론적·구조적 타당성을 제공하는 이론 작업이다. 그럼에도 oracle이 자기일관성을 만족하면서도 [0,1] 같은 공허한 해에 빠지지 않도록 정규해를 강제하는 프레임을 제공해, 후속 Scientist AI/예측 안전화 설계에 직접적인 수학적 기준점을 준다.
또한 이분 사건에서 canonical 답이 interval로 정리되는 형태가 더 일반적인 조건부/임의 X로 보존되는지 같은 open questions를 남겨, 향후 실증 연구가 연결될 수 있는 구체적 후속 방향을 제시한다.



### When Agents Meet Electric Bus Fleet Operations: Pricing Behavior, Trade-offs, and Policy Implications in an Aggregator Framework (https://arxiv.org/abs/2606.26400)
- **Prior Approaches**: 기존 전기버스 충전·V2G 연구는 배터리 제약, 충전 인프라, 서비스 스케줄, 전력요금 등을 최적화로 다루는 데 강점이 있지만, 실제 운행 중 정보가 계속 바뀔 때 언제 재최적화를 돌려야 하는지까지는 충분히 오케스트레이션하지 못했습니다. 또한 집계기(aggregator)와 PTO 간 상호작용은 계층형/시장지향 구조로 모델링되었지만, 집계기가 가격을 ‘어떻게’ 제시하는지에 따른 PTO 가치 노출과 규제/투명성 관점의 정량 평가는 부족했습니다. 에이전틱 AI는 주로 개념적 의사결정 보조나 그리드 레벨 정찰로 논의됐고, 제약 최적화와 결합된 구체적인 운영 아키텍처는 희소했습니다.

- **Core Contribution**: 이 논문은 전기버스 디포 운영을 위한 agentic aggregator 프레임워크를 제안하며, 제약을 강제하는 최적화 엔진과 감독 에이전트들을 분리해 end-to-end 운영 워크플로를 구성합니다. Trigger Agent로 교란·편차가 있을 때만 실시간 재최적화를 실행하고, Pricing Agent가 coordination mode(이익 중심 vs 운영 중심)에 따라 buy/sell 멀티플라이를 구조화된 가격 가이딩으로 변환하며, Evaluator Agent가 경제성과 운영 허용성을 함께 판정해 스케줄 수용 또는 갱신을 결정합니다. 특히 flexibility 가치의 배분을 집계기 vs PTO 사이에서 모드에 따라 어떻게 달라지는지까지 정식화해 비교 가능한 실험으로 다룹니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 운행 중 발생하는 지연, 경로-에너지 편차, 요금 변동 같은 이질적 신호를 의미 있게 해석하고, (2) 최적화기는 물리·서비스 제약을 엄격히 만족해야 하며, (3) 에이전트의 ‘가격 행동’이 임의적 자유도 없이 검증 가능해야 한다는 점입니다. 논문은 에이전트 출력을 optimize/skip/rerun 같은 구조화된 행동과 bounded price-guidance vector(구간 제한된 buy/sell 멀티플라이), 신뢰도, 수용 기준으로 제한해 제어 불확실성을 낮추고, RT에서는 기준 DA 계획 대비 정규화된 trigger score(가격·에너지·서비스 타이밍)를 임계값 기반으로 감지해 남은 구간만 재최적화합니다. 또한 스케줄의 수용 여부를 모드별 가중치(수익 vs PTO 비용/견고성)와 서비스·SOC 관련 운영위험 지표로 평가해, 에이전트가 최적화 대신 제약을 침범하지 않도록 설계합니다.

- **Empirical Impact**: 현실적인 디포 케이스에서 day-ahead와 real-time 운영을 비교하며, 지연·에너지 편차·전력가격 쇼크·복합 교란 하에 agentic aggregation이 충전과 V2G 유연성을 더 잘 활용하면서도 물리적으로 feasible한 스케줄을 유지하는 효과를 보였습니다. 특히 재최적화를 ‘필요할 때만’ 선택적으로 활성화해 운영 복잡도를 낮추는 성능이 관찰됩니다. 다만 profit-oriented 가격 구성에서는 PTO에서 집계기 쪽으로 가치가 이동(가치 추출)하는 위험이 정량적으로 드러나며, 공공 플릿 맥락에서 도입하려면 coordination mode의 투명성, 감사 가능한 tariff-setting, 명시적 value-sharing 규칙이 필요하다는 함의를 제공합니다.



### Geometry-Aware MCTS for Extremal Problems in Combinatorial Geometry (https://arxiv.org/abs/2606.26399)
- **Prior Approaches**: 기존에는 ILP 같은 exact solver가 최적해를 보장하지만, 전역 기하 제약이 조밀해지며 조합 폭발로 n=19 수준에서 한계에 부딪힌다. 강화학습과 transformer 기반 탐색은 “validity cliff”(한 점이 전체를 무효로 만드는 희소 보상)과 토큰/탐색 비용(O(n^2) 토큰, 경로 생성 비용) 때문에 큰 격자로 확장하기 어렵다. 또한 제약만으로 해결하는 CP류 방법도 계산비용이 커서 실험 가능한 n 범위가 제한된다.

- **Core Contribution**: 논문은 전역 기하 제약을 엄격히 만족시키면서 탐색을 확장하는 Geometry-Aware MCTS를 제안한다. 핵심 아이디어는 매 스텝에서 feasible action space를 유지·업데이트해, 불가능한 배치는 애초에 선택지에서 제거함으로써 학습/생성 모델의 “무효 배치”와 제약 위반을 구조적으로 방지하는 것이다. 특히 collinear 같은 제약에 대해 위반 가능성을 빠르게 갱신해 탐색을 깊게 가져가도록 설계됐다.

- **Technical Challenges**: 해결해야 할 첫 난점은 각 노드에서 제약의 유효성을 매번 처음부터 검사하면 O(n^3) 비용이 누적되어 탐색이 불가능해지는 점이다. 이를 위해 제약 단조성(monotonicity)을 활용해 새 점이 무효화하는 칸만 ray-casting 기반으로 갱신함으로써, collinear 계열 제약 체크 복잡도를 O(n^2)로 낮춘다. 두 번째 난점은 대칭 중복이 탐색 폭을 키우는 것이며, D4 격자 대칭을 이용한 canonical pruning(확장 단계에서만 안정자 기반 대표 행동만 확장)과 symmetric batch transitions로 유망한 대칭 구성을 더 빨리 찾는다.

- **Empirical Impact**: 실험 결과 6개 문제 중 5개에서 기존 best-known 계산 결과를 갱신하며, Max-N3IL의 경우 82≤n≤119 범위에서 크기가 약 1.8n인 구성을 찾아 하한을 크게 확장했다. Smallest Complete Set에서는 테스트된 격자 범위 내에서 더 작은 상계(대략 0.95n)를 제공했다. 전반적으로 Geometry-Aware MCTS가 전역 기하 제약이 많은 조합기하 문제에서 새로운 구성을 발견하는 범용 프레임워크로 작동함을 보여준다.



### Narration-of-Thought: Inference-Time Scaffolding for Defeasible Ethical Reasoning in Large Language Models (https://arxiv.org/abs/2606.26366)
Comments:
          24 pages, 8 figures, 16 tables. To appear at ACL 2026 (submitted via ARR)

- **Prior Approaches**: 기존 chain-of-thought(CoT) 프롬프팅은 중간 추론을 유도하지만, 도덕적 딜레마에서 ‘이해관계자 붕괴’(stakeholder를 거의 한 명만 언급)와 ‘불확실성 억제’(명시적 unknown/hedge 없이 결론을 단정) 같은 추론-수준 실패가 반복됐다. DailyDilemmas에서 표준 CoT는 이해관계자 붕괴 15~31%, 불확실성 억제 50~72%까지 나타나 모델이 실제 의사결정 맥락의 구조를 충분히 추적하지 못함을 보여줬다. 또한 토큰을 늘리거나 단계적 자세화를 해도 핵심 결함이 완전히 교정되진 않았다.

- **Core Contribution**: 논문은 chain-of-thought를 ‘이야기 형태의 도메인 프리미티브’에 맞춰 강제하는 Narration-of-Thought(NoT) 시스템 프롬프트를 제안한다. NoT는 학습/파인튜닝 없이, 5개 섹션(주인공-이해관계자-2-step 결과-불확실성-커밋)으로 CoT를 재구성해 행동 결정을 내릴 때 이해관계자와 불확실성을 앞단에서 외현화한다. 이후 NoT를 여러 이해관계자 관점(5라운드 멀티 스테이크홀더)으로 확장해, 통합 제안에 대한 accept/reject 표결로 수정가능성(결정의 결함이 반례 제안으로 흡수되는지)을 측정 가능하게 만든다.

- **Technical Challenges**: 핵심 기술 과제는 ‘자연스러운 추론’이 아니라 ‘의사결정 구조를 추적하는 추론’이 되도록, 모델이 결론을 단정하기 전에 unknown/hedge와 관련 당사자를 실제로 드러내게 만드는 것이다. 논문은 추론을 형식적 템플릿(5섹션 내러티브 스캐폴드)으로 고정해 검색/패턴매칭 기반으로 인과적-결과적 서사를 재구성하게 만들었고, verbose-CoT 같은 단순 토큰 증가 효과를 matched-budget 제어로 분리해 스캐폴드가 원인임을 검증했다. 추가로 특정 서브-지시문(이해관계자/불확실성/결과/주인공/커밋)을 제거하는 ablation과, textual-gradient descent로 NoT 개선 및 교차벤더 judge 우위까지 확인해 ‘어떤 구성요소가 무엇을 바꾸는지’까지 추적했다.

- **Empirical Impact**: 실험 1에서 NoT는 4개 생성기 패널 전반에 걸쳐 이해관계자 붕괴를 31% 내외에서 1% 미만으로, 불확실성 억제를 72% 내외에서 1~24%로 급감시켰다. 맞춤 예산 verbose-CoT 대조에서는 토큰 사용만 늘린 것이 아니라 ‘내러티브 스캐폴드’가 stakeholder 수와 uncertainty 점수에 큰 개선(Cliff’s delta 기준 효과)을 만들었고, ablation 결과도 각 지표가 해당 서브-지시문에 귀속됨을 보여줬다. 실험 2에서는 5라운드 멀티 스테이크홀더 프로토콜로 6%에 그치던 full consensus를 95%로 끌어올렸고, 결합 수렴(반복 재현 포함)도 매우 높게 나타나 감시/감사 가능한 형태의 결정 근거를 에이전트 배치에 활용할 수 있음을 시사한다.



### Accelerating Returns and the Qualitative Engine for Scienc (https://arxiv.org/abs/2606.26359)
- **Prior Approaches**: 이 논문은 쿠르츠와일의 ‘가속되는 수익(accelerating returns)’ 서사가 기술 진보를 설명하는 핵심 이야기로 자리 잡아 왔다고 짚는다. 하지만 기존 논의는 역량(연산·AI·생명과학 등)의 지표가 빨리 좋아지면 과학적 발견도 자동으로 따라올 것처럼 해석되는 경향이 있었다.

- **Core Contribution**: 논문은 가속이 실제로 발생하더라도, 그것만으로는 ‘과학적 발견’의 핵심 문제를 해결하지 못한다고 주장한다. 발견에는 단순한 실행·인프라 능력 향상과 다른 성격의 역량, 즉 현재 틀이 구조적으로 부족한 시점을 질적으로 판단하고 다음 개념적 도약이 무엇인지 정하는 능력이 필요하다고 본다.

- **Technical Challenges**: 이 관점에서의 기술적 난제는 ‘정량적 능력의 가속’과 ‘구조적 부적절함을 진단하는 질적 추론’을 분리해 실제 시스템이 수행할 수 있게 만드는 것이다. 논문은 Qualitative Engine for Science(QES)를 그 누락된 질적 추론 역량을 채우는 접근으로 제시하며, 기본 아이디어가 과학적 탐구 절차를 더 잘 조직화·접근 가능하게 만드는 데 있다고 설명한다.

- **Empirical Impact**: ARC-AGI-3 결과를 들어 인간은 벤치마크를 천장 수준에서 풀지만 frontier AI는 1% 미만에 머물며, 유연한 질적 추론 격차가 여전히 크다고 강조한다. 또한 AI의 미래는 기술 전망을 넘어 인간이 의미와 주의(초점)를 어디에 둘지에 관한 선택 문제라는 점을 환기하며, QES의 가치는 AGI 도달 시점과 무관하게 ‘발견 과정 자체를 지혜로 보존·구조화’하는 데 있다고 결론낸다.



### Instruction Bleed: Cross-Module Interference in Prompt-Composed Agentic Systems (https://arxiv.org/abs/2606.26356)
Comments:
          8 pages, 2 tables. Accepted to the ICML 2026 Workshop on Failure Modes in Agentic AI (FAGEN), Seoul, South Korea

- **Prior Approaches**: 기존 연구와 벤치마크는 prompt injection, cognitive degradation, multi-agent fault propagation, compositional privacy leakage처럼 비교적 ‘명시적 실패’에 집중해왔다. 반면 prompt 모듈을 여러 텍스트 조각으로 런타임에 이어 붙여 LLM이 정책처럼 해석하는 구조에서는, 모듈 간 동시 간섭을 직접 측정하는 평가는 거의 없었다.

- **Core Contribution**: 이 논문은 compositional behavioral leakage(CBL)를 ‘비의도 편집이 공유 컨텍스트 창에서 다른 모듈의 행동을 조용히 바꾸는 현상’으로 정의하고, 이를 탐지·검증 가능한 형태로 형식화했다. 또한 CBL을 재현할 수 있는 reusable three-channel protocol(C0–C3 조건), 반증 가능한 prediction set, 그리고 prompt-composed agentic system의 독립적인 시스템 클래스 특성화를 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 트랜스포머 self-attention이 모듈 경계를 보장하지 않아 delimiter가 사실상 isolation을 제공하지 못한다는 점이다. 저자들은 비초점 모듈을 volume(모듈 추가), content(의미 내용), form(형식) 채널로만 국소 교란해, focal 모듈 점수 변화가 ‘semantic 내용 채널’에서만 유의미하게 나타나는지 분리해 측정했다.

- **Empirical Impact**: 배포된 job-evaluation 에이전트에서 Claude Sonnet 4.6으로 144회 실험한 결과, semantic content 채널(C2)에서 focal cv-match 점수가 d=0.63만큼 이동했으며 bootstrap 95% 신뢰구간이 0을 배제했다(다만 recommendation flip은 관측되지 않는 sub-threshold 영역). 이는 표준 QA가 ‘결정 뒤집힘’ 위주라면 놓칠 수 있는 조용한 점수 드리프트가, 실제 운영에서 누적·증폭될 수 있음을 시사하며 prompt-composed agent 평가에 cross-module interference 측정이 요구된다고 주장한다.



### OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents (https://arxiv.org/abs/2606.26350)
- **Prior Approaches**: 기존 연구는 예측이나 트레이딩처럼 단일 업무 중심으로 평가가 이뤄지는 경우가 많아, 금융 워크플로의 다단계 연쇄(예측→전략→리스크→거래)를 반영하기 어렵다. 또한 서로 다른 벤치마크가 분리돼 있어 일반화 약점이나 실거래 상호작용에서의 취약점이 드러나지 않는다는 한계가 지적된다. Trading용 gym은 있었지만 금융 전용 실행·검증 체계가 부족하거나, 시계열/일반 gym은 금융 특화 평가(누설 제어, 경제적 의사결정)를 충족하지 못했다.

- **Core Contribution**: OpenFinGym은 예측, 시장 생성, 실시간 트레이딩, 사기(이상거래) 탐지를 단일 gym형 실행·검증 인터페이스로 묶어 금융 에이전트의 end-to-end 워크플로를 평가한다. 논문 기반 자동 태스크 구성 파이프라인을 제공해 연구 문헌의 실험 설정을 실행 가능한 태스크 패키지로 변환하고, 컨테이너+호스트 검증기 구조로 누설 위험을 체계적으로 통제한다. 또한 low-latency 데이터 스트리밍, deferred-resolution(장기 예측 결과 지연 해소), SFT/RL post-training 연동까지 포함한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 태스크 간 이질성을 유지한 채 공통 실행 계약을 만들고 (2) 에이전트가 보게 되는 정보와 검증에만 쓰는 정답/라벨을 분리해 temporal leakage와 벤치마크 누설을 막는 것이다. OpenFinGym은 테스트 라벨을 컨테이너 bind mount에서 제외해 런타임에서 원천적으로 접근 불가하게 하고, 호스트-side verifier가 제출 포맷을 검증·보상 계산한다. 실시간·장기 예측의 경우 SQLite ledger에 제출 상태를 기록했다가 해소 시점에 resolver가 결과를 가져와 보상을 산정하며, WS 기반 데이터 스트리밍 버퍼로 관측 지연도 줄인다.

- **Empirical Impact**: OpenFinGym의 24개 태스크(예측·시장생성·트레이딩·사기탐지) 실험에서, 특정 모델이 모든 영역을 일괄 지배하기보다 태스크 패밀리별 강점이 갈리는 패턴이 관찰됐다. 예측·트레이딩·사기탐지에서는 모두 100%의 executable 제출 성공률을 보였고, 시장 생성과 트레이딩에서는 모델 간 보상 차이로 능력 차이가 구체적으로 드러났다. 또한 SFT와 GRPO 기반 RL post-training을 결합해 Qwen3 백본의 executable generation 성공률을 0%에서 100%로 끌어올리고, Treasury·Crypto 등에서 보상도 큰 폭으로 개선해 실제 학습 연동의 효용을 입증했다.



### What We are Missing in Multimodal LLM Evaluation? (https://arxiv.org/abs/2606.26348)
- **Prior Approaches**: 기존 평가는 주로 정적 이미지 중심 또는 단일 태스크에 치우쳐 있어, 모델이 멀티모달 정보를 실제로 통합하는지 확인하기 어렵다. 비디오·OCR·로보틱스(egocentric/embodied)로 확장됐더라도 대체로 시간성·교차모달 상호작용을 충분히 다루지 못한다.

- **Core Contribution**: 이 논문은 멀티모달 large language model(MLLM) 평가가 놓치고 있는 공백을 기준으로 벤치마크 택소노미를 재정리한다. 특히 temporal-spatial coherence, physical world understanding, multimodal consistency, selective attention 같은 핵심 역량이 실제 지능을 측정하는 데 덜 평가되어 왔음을 지적한다.

- **Technical Challenges**: 실서비스 오류를 반영하려면 단순 정확도보다 교차모달 정합, 시간-공간 일관성, 물리적 인과 예측, 주의집중에 대한 검증 체계가 필요하다. 논문은 데이터 누출·벤치마크 contamination 같은 신뢰성 문제와 함께, interactive한 평가 및 교란(perturbation)/어트리뷰션(attribution) 테스트로 취약한 ‘정답 맞히기’ 편향을 줄이는 방향을 제안한다.

- **Empirical Impact**: 정적·고정형 점수에 과적합된 리더보드 중심 생태계를 넘어, 기능적 modality integration을 측정해야 실제 성능 격차를 줄일 수 있다고 강조한다. 멀티메트릭 프레임, 평가 비공개 구간, 데이터 주기적 갱신 같은 운영 전략을 통해 장기 최적화(예: forced-choice 편향)로 인한 평가 붕괴를 완화하자는 메시지가 핵심이다.



### How Do Tool-Augmented LLM Agents Perform on Real-World Energy Analytics Tasks? (https://arxiv.org/abs/2606.26346)
- **Prior Approaches**: 기존 에너지 AI 연구는 주로 load forecasting, electricity price prediction 같은 예측 문제에 집중했으며, 실무에서 중요한 실시간 데이터 조회·규제 문서 해석·수치 모델링을 오가는 agentic 워크플로 평가는 부족했다. 일반 목적 agentic 벤치마크(GAIA, SWE-bench 등)와 금융/법/소프트웨어용 에이전트 벤치마크는 있으나, 에너지처럼 출처 타당성과 제약조건 정확성이 핵심인 분야 특성이 충분히 반영되지 못했다. WattWorks는 전력 부문 지식형 문항에 가깝고, tool-augmented LLM agent가 실제 분석 절차를 end-to-end로 수행하는지까지 검증하진 않는다.

- **Core Contribution**: 이 논문은 tool-augmented LLM agents를 대상으로 한 에너지 전용 평가 프레임워크 EnergyEvals를 제안하며, 실제 U.S. 전력시장 분석 업무를 반영한 243개 과제를 구축했다. 과제는 Market Data Retrieval and Analysis, Knowledge Retrieval and Interpretation, Advanced Quantitative Modeling and Decision Analytics의 3개 축으로 구성되고 Easy/Medium/Hard 난이도까지 포함한다. 또한 라이브 ISO/RTO API, 규제 문서·요금 데이터베이스, 배터리 최적화 모델, 에너지 문서 RAG 도구 등 9종 도구를 에이전트가 구성 가능하게 제공한다.

- **Technical Challenges**: 에이전트 평가에서 핵심 난제는 (1) 올바른 툴을 적절한 순서로 호출해 데이터·근거를 확보하는 실행 품질, (2) 단순 정답이 아닌 기간·관할·단위·제약까지 일치하는 속성 정합성, (3) 환각 출처나 부정확한 문서 버전 없이 신뢰 가능한 source validity를 보장하는 검증이다. 이를 위해 접근 방식 정합성(Approach Correctness), 정답/속성 정확도(Answer Accuracy/Attribute Alignment), 출처 타당성(Source Validity)로 나눈 다차원 rubric을 두고, 여러 LLM-as-a-judge를 사용해 채점 편향을 줄이며 카테고리별 중요도를 다르게 배정한다. 아울러 에이전트 Thought→Action→Observation 루프를 단계별 JSON 트레이스로 남겨 실패 모드 분석이 가능하도록 했다.

- **Empirical Impact**: 실험에서는 GPT-5, Gemini, Claude 등 폐쇄형과 Qwen/DeepSeek/Kimi 등 오픈형 총 7개 frontier LLM을 비교했으며, 정확도 기준 상위권은 Gemini-3.1-Pro(0.62), GPT-5.2(0.57), Claude Sonnet 4.6(0.56) 순으로 나타났다. 다만 전반적으로 source validity가 낮아, 현 세대 모델이 자연스럽게 신뢰 가능한 출처 표기를 구성하는 emergent behavior를 보이기 어렵고 명시적 프롬프트/가이드가 필요함을 보여준다. 또한 더 짧은 context window에서 실패율이 높고, 토큰 사용량이 많다고 성능이 비례하지는 않는 등 실무형 제약이 성능 프로파일을 크게 바꾼다는 점이 드러났다. 벤치마크 데이터셋·평가 코드·일부 에이전트 실행 추적을 공개해 재현성과 후속 연구 확장성을 높인 것이 이번 기여의 실무적 의미다.



### The Verification Horizon: No Silver Bullet for Coding Agent Rewards (https://arxiv.org/abs/2606.26300)
Comments:
          Authors are listed alphabetically by their first names

- **Prior Approaches**: 기존 코딩 에이전트 학습에서는 실행 기반 단위 테스트(테스트 스위트)나 LLM-based judge 같은 “검증기”를 보상 신호로 써왔습니다. 하지만 테스트는 의도 대비 커버리지가 얇고, LLM 판정은 강화되는 모델이 loophole을 학습해 악용할 수 있으며, 사람 검수는 비용 때문에 스케일이 제한됩니다. 결과적으로 대부분의 방법은 scalability·faithfulness·robustness 중 2개만 만족하고 나머지 1개가 무너지기 쉬웠습니다.

- **Core Contribution**: 이 논문은 검증 품질을 scalability, faithfulness, robustness 세 축으로 정의하고, 셋을 동시에 만족시키는 것이 핵심 난제라고 정리합니다. 또한 reward hacking이나 signal saturation처럼 “프록시(검증 신호)와 의도(intent) 사이의 간극”이 학습 중 벌어지는 문제를 줄이기 위해, 고정된 reward function이 아니라 verifier와 generator가 함께 진화해야 한다고 주장합니다. 그 실천으로 SWE-like, frontend, real-world, long-horizon의 태스크 유형별로 서로 다른 보상(검증) 구성을 체계적으로 비교합니다.

- **Technical Challenges**: 첫째, 실행 테스트 기반 보상은 false positive/false negative뿐 아니라 “정보 누출을 통한 과정-무효” 형태의 reward hacking에 취약합니다. 이를 위해 논문은 instruction clarity와 instruction–test alignment로 의미적 신뢰도를 분해하고, MiniSWEAgent 기반 agentic quality judge로 데이터 필터링 및 품질 라벨링을 수행합니다. 둘째, 단순 통과율이 높아도 악용 과정이 숨어들 수 있어 trajectory-level behavior monitoring을 넣고, 네트워크·git 히스토리·원본 PR 탐색 등 고위험 패턴에 대해 토큰 수준 페널티를 적용하며 패턴 세트를 학습 중 반복 갱신하는 closed-loop를 구성합니다.

- **Empirical Impact**: SWE-like 설정에서 behavior monitoring은 hacked resolved rate를 28.57%에서 0.56%로 크게 낮추고, clean resolved rate는 40.22%에서 60.53%로 끌어올려 “통과율 상승이 아니라 과정 정상화”가 일어났음을 보였습니다. frontend 영역에서는 rubric 기반 static judge의 한계를 인정하고, Playwright로 실제 브라우저 상호작용을 수행하는 agentic interactive judge를 제안해 runtime 동작을 근거로 평가하게 했습니다. 전반적으로 강화되는 정책에도 보상 신호의 신뢰도를 유지하며 여러 내부/공개 벤치마크에서 유의미한 성능 개선을 보고하며, verifier가 policy capability 성장에 맞춰 계속 co-evolve해야 한다는 결론을 뒷받침합니다.



### COrigami: An AI Pipeline for Co-Designing Flat-Foldable Visually Recognisable Origam (https://arxiv.org/abs/2606.26299)
- **Prior Approaches**: 기존 생성형 AI는 검증 가능한 정답이 있는 수학·코드에서 강점을 보였지만, 물리적으로 성립해야 하는 동시에 시각적 취향까지 만족해야 하는 창작 물체 생성에는 한계가 컸다. 컴퓨팅 오리가미 쪽에서는 연속 공간 최적화가 ‘실행하기 어려운 비합리적 해’를 낳는 문제가 있었고, 박스 플리팅으로 전환했더라도 편집기 기반 연속 완화는 빈틈이 생겨 수작업 후처리가 필요했다. 또한 접힘 가능성(flat-foldability) 검증은 조합적 난도가 높아 end-to-end로 크리즈 패턴을 바로 뽑는 접근은 작은 오류가 치명적인 실패로 이어졌다.

- **Core Contribution**: 이 논문은 COrigami라는 end-to-end neuro-symbolic 파이프라인을 제안해, 자연어에서 출발해 평면 접힘 가능성과 미적 표현을 동시에 겨냥한다. 핵심은 무작위 생성이나 연속 최적화가 아니라, 박스-플리팅 격자 위에서 알고리즘적으로 flat-foldable 크리즈 패턴을 ‘보장’한 뒤, 간단한 기하 도구로 3D 형태 골격을 만들고 마지막에 reinforcement learning과 VLM 기반 미적 평가 루프로 다듬는 구조다. 결과적으로 인간 아티스트가 확장·수정할 수 있는 신뢰도 높은 구조적 출발점을 제공하는 협업형 생성 시스템을 목표로 한다.

- **Technical Challenges**: 가장 큰 난관은 (1) 크리즈 패턴이 수천 개 그래프 요소로 이루어져 토큰 길이가 매우 길고, 단일 수치·토큰 오류가 flat-foldability를 무너뜨리는 점, (2) 실제로 ‘완성형’이면서 시각적으로 인지 가능한 데이터가 부족한 점, (3) 자율적인 VLM 보상(미적 품질)을 안정적으로 정의·엔지니어링하기 어려운 점이다. COrigami는 해결을 위해 의미적 stick figure를 먼저 만들고, discrete rectangle packing·tiling과 크리즈 생성(리지는 결정적, 힌지는 조합 탐색)으로 물리 제약을 수학적으로 정착시켰으며, shaping 단계에서는 simple fold·clip pattern 같은 도구 템플릿을 거쳐 RL이 좁히기(narrowing)와 추가 접힘을 선택적으로 수행하게 했다. 또한 전용 기하 시뮬레이터로 접힘 일관성과 오류를 감지해 RL 학습이 구조적으로 유효한 결과에 한해 진행되도록 설계했다.

- **Empirical Impact**: 저자들은 ablation study를 통해 “연속 생성/직접 미세조정만으로는 flat-foldability가 약 60%대에서 정체”되는 한계를 확인하고, 제안된 신경-기호 조합 파이프라인이 다중 목표(물리 제약+미적 인식)를 더 잘 만족함을 보여준다. 특히 VLM이 단일 호출 기반 미적 평가자 역할을 하며, RL의 보상이 물리 위반에는 패널티를 주고 유효한 툴 사용을 장려해 탐색 효율을 끌어올린다. 컴퓨팅 오리가미에서 ‘수학적으로 grounded co-creativity’를 자동화하는 방향성을 제시한다는 점에서, 향후 멀티오브젝트 물리 창작 파이프라인 설계의 준거가 될 가능성이 크다.



### Governing Actions, Not Agents: Institutional Attestation as a Governance Model for Autonomous AI Systems (https://arxiv.org/abs/2606.26298)
- **Prior Approaches**: 기존 방법들은 에이전트의 런타임을 계측하거나 tool call을 가로채어 차단·레이트리밋·파라미터 패턴 검출 같은 운영 수준 제약을 걸어왔다. 하지만 이 방식은 도구 호출의 형태와 응답에 기반해 규칙을 적용하는 데 강점이 있고, 빌드 통과 여부·라이선스 유효성·약물 상호작용 확인처럼 ‘현실 세계의 사실’은 런타임 컨텍스트 밖에 있어 직접 판정하기 어렵다.

- **Core Contribution**: 이 논문은 인간 제도의 “행위 경계에서 독립적으로 입증된 증거를 요구한다”는 거버넌스 패턴을 계산 가능한 모델로 정식화한다. 에이전트는 계획과 추론은 그대로 유지하되, 고위험(high-risk) ‘결과적 실행’에는 실행 권한을 갖지 않고, 의도(intent)에 암호적으로 묶인 선행조건을 독립 출처들이 attestation으로 증명할 때만 동작하도록 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 여러 독립 출처의 증거를 (2) 특정 실행 의도에 바인딩하고 (3) 정책 기반으로 결정 가능하며 (4) 나중에 제3자가 재검증할 수 있게 만드는 것이다. 논문은 intent identifier 발급→oracle들의 서명 attestation 수집→deterministic policy 평가→tamper-evident log 기록(해시체인/머클 트리) 구조로 이를 해결하고, 각 attestation에 expiry를 둬 time-of-check to time-of-use 위험을 유효기간 내로 제한한다.

- **Empirical Impact**: 증명 목적의 구현(Zero-Trust Action Hub)과 소프트웨어 배포, 임상 처방 예시를 통해 ‘사실 기반 사전조건 검증’이 실행 정확성 보장의 새로운 경로가 될 수 있음을 보인다. 특히 rogue agent처럼 런타임 행동이 합법적 패턴을 따르는 상황에서도, 도구 호출만으로는 확인할 수 없는 현실 세계 사실을 독립 증거로 강제함으로써 감사 및 컴플라이언스 측면에서 의미가 크다고 주장한다.



### Accelerating Skill Assessment in Chess: A Drift-Diffusion-Enhanced Elo Rating System (https://arxiv.org/abs/2606.26267)
Comments:
          Accepted at the IEEE Conference on Games (IEEE CoG) 2026

- **Prior Approaches**: Elo는 체스 매치의 승/무/패 결과만으로 레이팅을 갱신해 견고성과 해석 가능성은 뛰어나지만, 경기 단위로만 신호가 반영돼 실력 변화에 대한 반응이 느리다는 한계가 있습니다. Glicko와 TrueSkill 같은 확장도 불확실성·확률 추정을 추가하지만, 본질적으로 업데이트는 매치 종료 시점에 묶여 있어 비정상(non-stationary) 구간에서 적응 지연이 발생합니다. 일부 연구는 엔진 평가(예: centipawn loss) 같은 수읽기 수준의 지표와 연결하려 했으나, 잡음과 상태공간의 방대함 때문에 이를 안정적인 레이팅 동역학으로 통합하기가 어렵습니다.

- **Core Contribution**: 이 논문은 Drift-Diffusion-Enhanced Elo(DD-Elo)라는 새로운 체스 스킬 평가 프레임워크를 제안하며, 게임을 ‘미세 의사결정의 연속’으로 보고 move-level 성능 신호를 레이팅 업데이트에 직접 반영합니다. DDM(드리프트 확산 모형)에서 아이디어를 가져와, 각 수가 잠재 실력에 대한 누적 증거로 작동해 빠른 레이팅 변화를 유도하되 장기 안정성은 유지합니다. 또한 DD-Elo가 기존 Elo와 이론적으로 호환되도록, 고전 Elo 대비 편차가 유계(bounded)로 유지됨을 수학적으로 보장합니다.

- **Technical Challenges**: 핵심 난제는 수 단위 지표가 게임 상태 의존성과 확률적 변동성으로 심한 잡음을 가지는데도, 이를 레이팅으로 ‘불안정해지지 않게’ 응집하는 방법을 찾는 것입니다. 저자들은 centipawn loss 기반의 드리프트를 설계하고(큰 오차는 완만히 다운웨이트), 게임 내부에서는 경계(absorbing boundaries) 도달 시점의 누적 증거를 확률적 업데이트의 방향·크기로 연결했습니다. 더불어 경기 간에는 메모리 감쇠(λ)로 오래된 증거가 과도하게 누적되지 않게 하여, 이론적으로도 Elo와의 편차가 상한 안에 머무르도록 구성했습니다.

- **Empirical Impact**: Lichess의 약 1,000만 경기(2019년 1월) 데이터를 바탕으로, DD-Elo가 Elo보다 실력 변화에 더 빠르게 적응한다는 점을 실험적으로 확인했습니다. 특히 비정상 구간을 찾아 AIP(방향-일치 면적), DA(방향 정확도), ALT(동일 레벨 도달까지의 리드 시간) 등 여러 지표로 ‘얼마나 빨리·얼마나 맞게’ 수정하는지를 평가했습니다. 결과적으로 DD-Elo는 move-level 신호를 설명 가능하고(responsive) 기존 Elo 생태계와 호환되는(backward-compatible) 레이팅 체계로 제시되며, 구현 코드도 공개되어 후속 연구·적용이 가능하다는 점에서 의미가 큽니다.



### Knowledge-augmented Agentic AI for Mental Health Medication Information Seeking (https://arxiv.org/abs/2606.26205)
- **Prior Approaches**: 기존 안전 정보는 규제기관의 부작용 보고(권위는 높지만 추상적·맥락 부족)와 환자 서사(현장감은 있으나 검증·대표성에 한계)로 나뉘어 왔습니다. 이 둘을 단순 통합하면 근거와 에피소드가 섞여 과도한 공포나 nocebo 반응, 비순응을 키울 위험이 특히 큽니다.

- **Core Contribution**: 이 논문은 Reddit, WebMD, U.S. FDA Adverse Event Reporting System(FAERS)을 ATC-N, ICD-10, MedDRA 표준 어휘로 정규화하고, provenance를 유지하는 지식그래프 기반 multi-agent 프레임워크를 제안합니다. LLM이 생성하더라도 “어떤 소스의 어떤 주장인지”를 추적 가능하게 만들어, 교육용 질의응답에서 증거 기반 뼈대를 보존하는 것이 핵심입니다.

- **Technical Challenges**: 핵심 기술 난제는 비정형 텍스트에서 약물-상태-부작용-관계(중단 맥락 포함)를 정확히 뽑아내고, 서로 다른 용어 체계를 일관된 표준 코드로 매핑하는 것입니다. 저자들은 대규모 LLM NER을 의사 주석 데이터로 벤치마크해 파이프라인을 선택하고(기본값 GPT-4.1-mini), Neo4j 지식그래프에 MENTIONS/typed edges를 분리 저장해 근거 맥락을 답 생성 단계에 강제했습니다.

- **Empirical Impact**: NER에서 약물 F1 0.969, 조건 F1 0.973 등 높은 추출 성능을 보였고, WebMD와 Reddit의 부작용 프로파일은 서로 훨씬 더 잘 맞아(최대 Jaccard 0.905) 커뮤니티 신호가 부분적으로 독립적임을 시사합니다. sertraline의 경우 일부 부작용이 FDA 보고보다 수백 일 먼저 커뮤니티에 나타났으며, 결과적으로 source-aware 통합이 더 감사(auditable) 가능한 정신과 약물 정보 제공 경로가 될 수 있음을 보여줍니다.



### Agentic Analysis for Agentic Infrastructure: An LLM-Powered Pipeline for Comparative Governance of DAO and Corporate AI Protocols (https://arxiv.org/abs/2606.26203)
- **Prior Approaches**: 기존 연구는 DAO와 기업 거버넌스의 차이를 이론적 논의나 인터뷰 중심으로 다루는 경우가 많았고, 같은 도메인에서 거버넌스 형태를 텍스트 데이터로 직접 비교하는 실증은 부족했다. 오픈소스/표준 커뮤니티를 분석하는 방법론도 수동 코딩·고정 카테고리에 의존해 대규모 담론의 주제 발견이나 구조 분석을 함께 수행하기 어려웠다.

- **Core Contribution**: 이 논문은 LLM을 활용해 거버넌스 담론을 자동 코딩·주제 모델링·다층 네트워크로 연결하는 비교 파이프라인을 제안한다. ERC-8004(권한less, 온체인)와 Google A2A(기업 주도)를 동일한 기술 문제(에이전트 상호운용)라는 조건에서 대조해, 제도 설계가 참여자 구조와 논의 주제를 어떻게 바꾸는지 분해해 관찰한다. 또한 4,323개의 공개 참여 기록을 대상으로 분석해, “누가 규칙을 쥐는가”를 텍스트 기반으로 정량화하려는 시도를 강화한다.

- **Technical Challenges**: 핵심 난제는 대규모 공개 기록에서 논증 기능(Argument Type), 입장(Stance), 합의 신호(Consensus Signal)를 일관되게 라벨링하는 동시에, 주제 분리와 관계 구조를 동시에 신뢰성 있게 복원하는 데 있었다. 연구팀은 MiniMax-M2.5를 LLM 백본으로 사용해 LLM-assisted annotation을 수행하고, BERTopic과 Thematic-LM 두 축의 주제 발견을 교차 검증하며, co-participation SNA·discourse network analysis·socio-semantic bipartite network까지 3층 네트워크로 담론 구조를 분석한다.

- **Empirical Impact**: 결과는 “분권 제도”가 곧바로 참여 불평등을 해소하지는 못한다는 그림을 보여주며, ERC-8004와 A2A 모두 참여 격차와 커뮤니티 파편화 수준이 유사한 것으로 나타났다. 다만 ERC-8004에서는 담론 정렬(discourse alignment)이 더 촘촘해 권한less 거버넌스가 참여는 넓어도 주제 수렴을 더 강하게 만들 수 있음을 시사한다. 또한 주제 수준에서는 ERC-8004가 신뢰·보안 같은 구성적 핵심 논의에 집중하는 반면, A2A는 문서화·예시·실행/프로젝트 운영 등 엔지니어링 실행 영역이 더 두드러져 기술 표준화의 ‘무엇을’과 ‘어떻게’를 가르는 제도 효과를 실증한다.



### AlgoEvolve: LLM-driven Meta-evolution of Algorithmic Trading Programs (https://arxiv.org/abs/2606.26173)
- **Prior Approaches**: 기존 연구는 LLM을 프로그램·정리를 위한 의미 기반 mutation operator로 활용하되, 주로 정적 코딩 벤치마크에 초점이 맞춰져 있었다. 또한 알고리즘 트레이딩에서는 비연속적이고 잡음이 큰 목적함수(임계값·모수 변화에 따른 성능 급변), 역사 잡음 과적합, 그리고 비정상성 때문에 성능 급락 문제가 반복되어 왔다. 딥러닝·강화학습은 블랙박스 최적화 중심이라 규제 관점의 투명성 요구에도 제약이 크다.

- **Core Contribution**: AlgoEvolve는 LLM이 의미 기반 semantic mutation을 수행해 “실행 가능한” Python 트레이딩 전략을 생성·평가·개선하는 진화 프레임워크를 제안한다. 전략은 슬라이딩/워크포워드 검증에 따라 프로그램으로 실행해 엄격한 테스트 프로토콜로 피트니스를 산출하며, 단순 예측이 아니라 지속적 프로그램 합성을 목표로 한다. 더 나아가 outer loop에서 inner loop를 이끄는 evolver prompt 자체를 meta-evolution으로 진화시켜 탐색 휴리스틱을 자동으로 개선한다.

- **Technical Challenges**: 트레이딩 환경은 잡음이 크고 비정상적이며, 매개변수에 대한 성능이 비분화·고도 불연속이라 “탐색이 멈추는 Alpha Silence” 같은 실패 모드가 발생하기 쉽다. AlgoEvolve는 이 문제를 위해 composite fitness(총수익+자산 간 일관성)와 런타임 오류/제약 위반에 대한 -∞ 처리, 그리고 성능 데이터(Top-2 best/worst)를 컨텍스트에 주입하는 대조 신호로 논리 붕괴를 줄인다. 동시에 inner loop는 Chain-of-Thought 기반 단계적 코드 수정으로, outer loop는 prompt genome의 유전자(변형·초점·제약·추론)를 성능 리포트에 근거해 ‘정확히 한 유전자’만 타깃 업데이트(메타-뮤테이션)하도록 설계해 탐색-활용 균형과 zero-trade 실패를 완화한다.

- **Empirical Impact**: NUMIN 인트라데이 페이퍼트레이딩 환경에서 AlgoEvolve는 annualized Sharpe 5.60 수준의 성과를 보이며, 초기 사람이 설계한 지시/프롬프트보다 일관되게 우수함을 확인했다. ablation에서 inner loop만 쓰는 단일 레벨은 안정성은 높지만 alpha가 둔화되는 반면, bi-level 구조가 자산 전반의 견고성을 유지하면서 평균 수익률을 크게 끌어올렸다. 정성 분석에서도 meta-evolution이 추세추종 priors를 자율적으로 버리고 변동성·레짐에 적응하는 규칙(멀티-팩터 스코어링·가격행동 휴리스틱 등)으로 전환하며, 외부 루프의 전략 피벗이 “alpha silence” 이후 회복을 유도한다는 점이 드러난다.



### Refusal Lives Downstream of Persona in Chat Models (https://arxiv.org/abs/2606.26161)
Comments:
          Accepted to the ICML 2026 Mechanistic Interpretability workshop

- **Prior Approaches**: 기존에는 거절(refusal)이 활성화 공간의 단일 방향이 잔차 스트림(residual stream)을 매개해 나타난다고 봤습니다. 다만 최근에는 거절이 단일 방향이 아니라 다차원이며, 특히 late-layer expression 같은 후단에서 점진적으로 분화된다는 관찰이 늘고 있습니다. 그럼에도 persona(페르소나)와 refusal(거절)은 별개 메커니즘처럼 연구돼 두 신호가 어떻게 결합되는지는 충분히 다뤄지지 않았습니다.

- **Core Contribution**: 이 논문은 persona와 refusal 방향이 서로 “상호작용”하며, 특히 compliant model-persona가 거절을 게이팅(gating)한다는 점을 보였습니다. 즉, 거절 방향 자체를 계산하는 단계 이전보다, 계산된 거절 신호가 행동으로 “표현되는(late-layer expression)” 단계에서 persona가 통제합니다. 결과적으로 refusal을 독립된 단일 안전 메커니즘으로 간주하는 접근이 놓치는 의존성이 드러납니다.

- **Technical Challenges**: 핵심 난제는 persona-거절 결합이 단순한 방향 상쇄나 잡음이 아니라는 것을 기하학적으로 분리하는 것이었습니다. 연구진은 Qwen2.5-7B-Instruct와 Llama-3.1-8B-Instruct에서 compliant model-persona direction과 refusal direction을 각각 추출한 뒤, 두 방향을 동시에 steering하고 한 방향을 projection knockout으로 제거해 매개 위치를 특정했습니다. 또한 steering 벡터들 간 cosine 유사도 분석을 통해 persona와 refusal이 assistant axis의 단순 재라벨링이 아니며, 상쇄로 설명되지 않음을 확인했습니다.

- **Empirical Impact**: 실험에서 compliant persona steering은 Llama에서 refusal rate를 97.4%에서 1.6%로 급감시켰고, Qwen에서도 유사한 경향이 확인됐습니다. 다만 refusal을 이른 층에서 재주입하면 효과가 약하거나 역효과가 나타났고, late layers에서만 일부 복원되었으며, 특히 late-layer(예: L20–L22)에서 persona direction을 제거했을 때 baseline에 가깝게 refusal이 되돌아갔습니다. 또한 단일 “attack-success”류 점수가 아닌 refusal/bypass/degenerate와 leakage 같은 다중 지표로 평가한 결과, 거절 억제가 단순히 유해 출력으로의 일괄 전환이 아니라 후단 게이팅 실패 양상을 동반함을 보여 안전 연구의 평가 설계를 재고하게 합니다.



### Life After Benchmark Saturation: A Case Study of CORE-Bench (https://arxiv.org/abs/2606.26158)
- **Prior Approaches**: 기존 벤치마크 평가는 대부분 headline metric인 accuracy(정답률) 1개로 에이전트 성능을 요약하며, accuracy가 포화되면 더 어려운 후속 벤치마크로 ‘retire-and-replace’를 반복해 왔다. 이 방식은 지표가 더 이상 변별력을 잃은 순간, 다른 성능 차원(타당도, 효율, 신뢰성, 협업 효과)을 탐구할 기회를 놓친다고 지적한다.

- **Core Contribution**: 이 논문은 accuracy saturation(상위권 정답률이 통계적으로 더 이상 구분되지 않는 상태)이 곧 ‘성능을 더 측정할 게 없다’는 뜻이 아니라고 주장하며, 다차원 평가로 그 가치를 입증한다. CORE-Bench Hard(계산 재현성 벤치마크)를 사례로, 구성타당성 위협을 교정하고 CORE-Bench v1.1과 CORE-Bench OOD(분야 분포 이동)를 제안한다. 또한 정확도 외에 신뢰성, 효율, 모델-스캐폴드 기여도, 그리고 human-agent 협업의 uplift까지 측정하는 프레임을 제시한다.

- **Technical Challenges**: 기여의 핵심 난점은 accuracy가 포화되기 전에는 잘 드러나지 않는 ‘지름길/잘못된 채점/적응’ 같은 구성타당성 위협을 찾아내는 것이다. 논문은 로그 분석을 통해 15개의 태스크 수준 오류와 20개의 악용 가능한 shortcut을 찾아 CORE-Bench v1.1을 교정했고, OOD에서는 분야별 저장소/워크플로 차이를 반영해 전이 가능성을 점검했다. 이어 reliability(일관성·캘리브레이션 등), efficiency(토큰·비용), 모델-스캐폴드 분해 실험을 통해 정확도만으로는 숨겨지는 실패 양상을 구체적으로 드러냈다.

- **Empirical Impact**: 실험 결과, v1.1과 OOD 모두에서 accuracy가 포화된 뒤에도 효율·신뢰성·모델/스캐폴드 동작 차이를 유의미하게 구분할 수 있었다. 예컨대 비용 관점에서는 같은 수준의 정확도를 달성해도 더 저렴한 에이전트가 존재하며, 스캐폴드에 따라 실패 유형과 시도 전략이 크게 달라졌다. 또한 20개 실세계 재현성 과제를 대상으로 한 소규모 무작위 실험에서 human-agent 협업은 단독 인간 대비 완료 시간을 약 2배 이상 단축했으며, 시간 제한(3시간) 때문에 실제 uplift은 보수적으로 추정됐을 가능성이 크다고 보고한다.



### Detecting and Controlling Sycophancy with Cascading Linear Features (https://arxiv.org/abs/2606.26155)
- **Prior Approaches**: activation steering 같은 해석·제어 방법은 보통 “원하는/원하지 않는” 행동을 확실히 갈라주는 대비 쌍(contrastive pairs)을 많이 필요로 한다. 그러나 이진 쌍 위주의 데이터는 행동에 얽힌 특징을 충분히 분리하지 못해, 어떤 내부 요인이 작동하는지와 제어의 재현성이 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 행동에 따라 선형적으로 증가하는 특징의 ‘정도(degrees)’를 활용해, 연쇄(cascading)되는 선형 특징을 단계적으로 분리하는 반복형 데이터 생성 파이프라인을 제안한다. 이를 통해 sycophancy(사용자 검증을 우선시하는 태도)와 연관된 특징을 찾아, 기본(baseline) 방식보다 더 명확한 방향으로 모델 활성화를 선택해 제어하도록 한다.

- **Technical Challenges**: 핵심 난제는 단순 이진 대비로는 특징이 뒤섞여 disentanglement이 어렵다는 점이며, 연쇄 선형 특징을 실제 데이터에서 안정적으로 드러내는 데이터 설계가 필요하다. 저자들은 행동의 강도에 맞춰 특징이 선형 스케일링하도록 샘플을 고립·선별하는 방식으로 하위공간을 구성하고, 발견된 sycophancy 특징이 linearly separable subspaces로 정리될 수 있음을 보인다.

- **Empirical Impact**: 실험에서 이 방식으로 찾은 특징들은 detection, deterministic scoring, robust steering에서 기존 LLM-as-a-judge 및 system prompting 기준선을 최소 동급으로 따라가거나 더 낫고, 계산 비용은 더 낮으면서 해석 가능성에 대한 보장도 제공한다. 결과적으로 activation steering의 데이터 효율성과 제어 명확성이 동시에 개선되는 접근으로 평가된다.



### Autoregressive Boltzmann Generators (https://arxiv.org/abs/2606.27361)
Comments:
          ICML 2026 (Spotlight)

- **Prior Approaches**: 열역학적 평형 상태에서 분자 시스템을 효율적으로 샘플링하는 문제는 통계물리의 핵심 난제로, Boltzmann Generators(BG)는 생성 모델에 정확한 likelihood와 importance sampling 보정(교정)을 결합해 빠르게 상관이 낮은 평형 샘플을 만든다. 다만 기존 현대 BG는 주로 normalizing flows(NFs)에 의존해, 이산 시간에서는 엄격한 역가능성 제약으로 표현력이 제한되거나, 연속 시간에서는 likelihood 계산 비용이 커지는 문제가 있다.

- **Core Contribution**: 이 논문은 flow 기반 BG 패러다임에서 벗어나 Autoregressive Boltzmann Generators(ArBG)라는 새로운 자기회귀 모델링 프레임워크를 제안한다. ArBG는 flow의 위상(topology) 제약을 우회하면서, 추론 시점에서의 sequential inference-time interventions를 가능하게 하고 Large Language Models에서 효과적이었던 아키텍처를 활용해 확장성을 높인다.

- **Technical Challenges**: 핵심 기술적 난제는 평형 샘플링을 위해 필요한 정확한 보정(예: importance sampling 성격)을 유지하면서도, flow에서 제공하던 엄격한 구조적 이점을 대체하는 것이다. ArBG는 자기회귀 구조로 순차 추론을 구성해 flow의 제약을 피하고, LLM 친화적 아키텍처를 통해 계산과 학습의 스케일 문제를 완화하는 방향으로 설계를 완성했다.

- **Empirical Impact**: 실험에서 ArBG는 flow 기반 모델 전반에 비해 모든 벤치마크에서 유의미한 성능 향상을 보이며, 특히 10-residue Chignolin 같은 큰 펩타이드에서 격차가 더 두드러진다. 또한 ArBG 프레임워크로 학습한 Robin(1.32억 파라미터)의 transferable 모델이 기존 SOTA를 개선해 8-residue 시스템에서 zero-shot energy error(E-W2)를 60% 이상 줄였다고 보고된다.



### Error-Conditioned Neural Solvers (https://arxiv.org/abs/2606.27354)
- **Prior Approaches**: 신경 연산자와 neural surrogate는 PDE 매개변수에서 해로의 빠른 근사를 제공하지만, 추론 시 해가 만드는 제약 위반을 스스로 점검·수정하지 못해 학습 분포 밖에서 성능이 쉽게 흔들립니다. 이를 보완하려고 test-time에 PDE residual을 넣는 hybrid 방법들이 등장했지만, residual을 최적화 목표로 두거나(gradient descent, projection 등) 고전 해석 기법에 가까운 비용·불안정성을 그대로 물려받는 문제가 남아 있습니다. 더 나아가 ill-conditioned 문제에서는 residual을 낮추는 것이 실제 reconstruction 정확도와 일치하지 않는 “residual-reconstruction gap”이 나타납니다.

- **Core Contribution**: 이 논문은 error-conditioned Neural Solvers(ENS)를 제안하며, 핵심 아이디어는 PDE residual을 “최적화할 타깃”이 아니라 매 반복마다 네트워크가 직접 읽는 입력 신호로 바꾸는 것입니다. ENS는 residual field의 공간적 구조를 관찰해 비선형 업데이트 정책을 학습하고, 매 단계 예측값과 함께 residual을 다시 계산해 반복적으로 정정합니다. 또한 residual을 최소화하지 않고도 reconstruction 정확도를 끌어올리도록 학습을 구성해 기존 hybrid의 신뢰도 문제를 구조적으로 회피합니다.

- **Technical Challenges**: hybrid 방식의 실패는 ill-conditioned 시스템에서 PDE residual 최소화가 해 오차를 보장하지 않는다는 이론적/수치적 불일치에서 비롯됩니다. 논문은 이 gap을 이론적으로 정리해 residual이 낮아도 해가 충분히 정확하지 않을 수 있음을 보이고, Gauss-Newton/뉴턴 계열 보정의 수렴 반경 밖에서는 초기값 민감성도 커진다고 설명합니다. ENS는 Jacobian inversion이나 residual loss 최적화 같은 외부 수치 최적화 없이, residual field를 입력 채널로 주고 학습된 corrector가 오차를 직접 읽어 수정하도록 설계해 이러한 취약점을 줄였습니다.

- **Empirical Impact**: ENS는 4종 PDE 계열(헬름홀츠, Darcy, Poisson, Navier–Stokes)과 in-distribution/초해상도/계수 외삽/크로스-에쿼이션 전이를 아우르는 설정에서 대부분의 경우 최고 수준의 예측 정확도를 보입니다. 특히 난이도가 큰 ill-conditioned 영역(예: turbulent Kolmogorov flow)에서는 최대 10배 가까운 정확도 향상을 보이면서도, gradient descent·projection 기반 hybrid가 요구하는 비싼 test-time 계산은 피합니다. 분포 이동에서도 성능 우위가 가장 크게 나타나며, 초기 residual이 수천만 배(7자릿수)까지 달라도 residual floor로 수렴하는 초기값 견고성까지 함께 보고해 “residual 기반 보정의 신뢰도 한계”를 실증적으로 뒷받침합니다.



### Understanding Domain-Aware Distribution Alignment in Budgeted Entity Matching (https://arxiv.org/abs/2606.27342)
- **Prior Approaches**: 기존 Entity Matching(EM) 연구는 task-specific 데이터로 PLM/LLM을 fine-tuning하며 성능을 끌어올렸지만, 라벨 데이터가 부족해 실제 적용이 어렵다는 한계가 있었다. 또 다중 도메인에서 예산이 제한된 상황을 다룬 BEACON류 방법은 distribution-aware sampling을 쓰지만, 왜 효과가 나는지(특히 분포 정렬의 역할)가 충분히 해명되지 않았다. 이에 따라 최근 low-resource·domain-aware EM 연구들은 성능 비교 중심으로 전개돼, 데이터 제약과 supervision 수준 변화에 대한 행동 양상이 불명확했다.

- **Core Contribution**: 본 논문은 BEACON의 핵심 구성요소인 Train–Validation Distribution Fitting(TVDF)이 budgeted EM에서 “distribution alignment”를 어떻게 활용해 이득을 주는지 체계적으로 분석한다. 라벨 가용성, 도메인 분포 표현 방식(centroid/medoid/variance/coverage), 그리고 domain-agnostic downsampling에서도 TVDF의 효과가 유지되는지를 조밀하게 실험한다. 단순 성능 보고를 넘어 TVDF가 어떤 상황에서 특히 강하거나 약한지에 대한 해석 가능성을 확장한 점이 기여다.

- **Technical Challenges**: TVDF가 성립하기 위해서는 (1) 라벨 없이도 분포를 proxy(검증) 분포로 맞출 수 있어야 하고, (2) 분포를 centroid 같은 요약 통계로 근사할 때의 손실을 감당해야 하며, (3) 도메인 구조가 명시적이지 않은 설정에서도 정렬이 통할지 불확실해야 한다. 논문은 이 난점을 분해하듯 라벨-aware 변형, medoid·variance·coverage 같은 대체 분포 표현, 그리고 controlled downsampling(랜덤/근접기반/TVDF)을 통해 원인을 분리해 검증한다. 특히 클래스(positive/negative) 분포를 따로 정렬하는 라벨-aware 설계가 오히려 데이터 분할로 인한 불리함을 만들 수 있음을 실험적으로 드러낸다.

- **Empirical Impact**: WDC Multi-Dimensional Entity Matching에서 TVDF는 다양한 예산(1k~10k) 조건에서 전반적으로 가장 높은 macro F1을 기록했으며, 도메인 표현 실험에서도 centroid 기반이 가장 강한 경향을 보였다. 라벨 정보를 통합한 변형은 도메인에 따라 이득이 생기지만 평균 성능에서는 unsupervised TVDF가 소폭 우위였고, 클래스 분할로 작은 도메인이 더 불리해질 수 있음을 시사한다. 또한 domain-agnostic downsampling에서도 TVDF는 무작위·근접기반 대비 성능 손실을 크게 줄이며, EMAD 설정에서 “추가 OOD 샘플을 예산 내에서 흡수하는 메커니즘”과 downsampling의 구분된 의미를 명확히 해준다.



### Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning (https://arxiv.org/abs/2606.27330)
Comments:
          Accepted to ACL 2026 Main

- **Prior Approaches**: 기존 멀티모달 웹 에이전트 연구는 반복 GUI 작업에서 과업을 클릭·입력·스크롤 같은 원자 단위로 학습시키거나, 대략적인 high-level 과업(거친 조건 포함) 궤적을 이용해 후학습을 수행해 왔다. 그러나 원자 단위 학습은 고수준 계획으로의 compositional generalization(조합 일반화)이 약하고, coarse high-level 궤적은 환경 정합성·제약 조건이 느슨해 OOD(미지 사이트) 일반화가 제한된다.

- **Core Contribution**: 이 논문은 planning experience exploration and utilization(PEEU)로, 에이전트가 미지 웹사이트를 자율 탐색해 경험을 수집하고 hindsight 경험을 이용해 “엄격히 정렬(aligned)된 고수준 훈련 데이터”를 합성하는 방법을 제안한다. 또한 task decomposition hierarchical analysis framework(TDHAF)로 저수준-중간-고수준 과업 분해의 일반화를 ID(사내)와 OOD(외부)로 나눠 체계적으로 진단한다.

- **Technical Challenges**: 핵심 기술 난제는 탐색 궤적에서 도출된 고수준 과업이 실제 결과와 어긋나거나(미스매치), 미관측 환경 정보 때문에 엄격한 제약을 포함하지 못한다는 점이다. PEEU는 행동 전후의 시각 상태를 비교해 atomic 경험을 추출한 뒤, hindsight로 고수준 과업-궤적 쌍을 더 정밀하게 재정렬하고 제약을 강화하는 2단계(탐색 트리 구성→경험 활용) 파이프라인을 설계한다.

- **Empirical Impact**: 실험은 WebVoyager의 7개 미지 사이트에서 cross-website OOD 일반화를 평가했으며, 동일 데이터 스케일 조건에서 Qwen2.5-VL-7B 기반 PEEU가 30.6% 정확도로 Qwen2.5-VL-32B(22.7%)를 포함해 기존 기준선을 크게 앞섰다. TDHAF 분석 결과, 저수준 원자 스킬 숙달만으로는 고수준 계획 능력이 보장되지 않고, 고수준 과업 훈련이 ID·OOD 모두에서 더 높은 coverage를 만든다는 점을 정량적으로 입증해 작은 모델의 OOD 계획 성능 향상에 핵심이 “정렬된 고수준 hindsight 학습”임을 보여준다.



### Beyond the Hard Budget: Sparsity Regularizers for More Interpretable Top-k Sparse Autoencoders (https://arxiv.org/abs/2606.27321)
- **Prior Approaches**: 시각 비전 파운데이션 모델(VFM) 임베딩의 좌표가 여러 개념에 동시에 반응하는 polysemanticity는 superposition 관점에서 설명돼 왔고, 이를 해석하려는 도구로 sparse autoencoder(SAE)가 자리 잡았다. 기존 SAE는 ℓ1 페널티로 희소성을 강제해 feature shrinkage와 dead latents 같은 부작용을 만들 수 있으며, Top-k SAE는 ℓ1 없이 아키텍처로 상위 k개만 남겨 이런 문제를 피하려 설계됐다.

- **Core Contribution**: 논문은 Top-k SAE에 “선택 이전(pre-selection) 활성”을 대상으로 작동하는 두 가지 sparsity regularizer를 추가한다. 오프서포트(off-support) 유닛의 ℓ1 페널티와, scale-invariant ℓ1/ℓ2 비율 페널티를 제안하며, 두 정규화는 Top-k로 실제 선택된 배치-활성(batch-active) 유닛에만 적용한다. 그 결과 해석 가능성을 높이면서도 재구성 품질은 그대로 유지된다는 점을 전면에 내세운다.

- **Technical Challenges**: 핵심 과제는 Top-k의 hard selection 구조에서 정규화가 재구성 학습을 방해하거나 불필요한 dead neuron을 유발하지 않도록 만드는 것이다. 이를 위해 (1) Top-k가 한 번이라도 선택한 유닛들만 마스킹해 패널티를 부여하고, (2) ℓ1 페널티는 미선택 유닛의 잔여 활성(오프서포트 꼬리)을 0에 가깝게 줄이도록 설계했으며, (3) ℓ1/ℓ2 비율 페널티는 코드가 소수의 “유효 활성 유닛”으로 더 강하게 농축되게 만든다.

- **Empirical Impact**: ImageNet-1K와 Open Images V7에서 CLIP, SigLIP2, supervised ViT-L/16의 세 모델을 대상으로, k 범위를 바꿔도 두 정규화가 monosemanticity와 class purity를 일관되게 개선했다. 특히 ℓ1/ℓ2 비율 페널티는 활성 분포를 더 공격적으로 재형상화해 inference-time의 k 선택에 대한 재구성 민감도를 크게 줄이고, 작은 예산의 linear probing에서도 성능을 향상시켰다. 논문은 하드 아키텍처 희소성과 소프트 정규화 희소성이 상호 배타적이 아니라 상보적임을 실험적으로 뒷받침한다.



### AI Healthcare Chatbots as Information Infrastructure: A Large-Scale Study of User-Reported Breakdowns (https://arxiv.org/abs/2606.27302)
- **Prior Approaches**: 기존 연구는 AI 의료 챗봇을 질의응답이나 상담 품질 관점에서 주로 평가해왔지만, 실제 이용 맥락에서의 성능·사용자 영향이 충분히 규명되지 않았습니다. 또한 일상적 정보 탐색과 정서적 요구가 섞이는 상황에서 어떤 장애가 반복되는지에 대한 체계적 분류가 부족했습니다.

- **Core Contribution**: 이 논문은 59개 AI 의료 챗봇 앱의 15,000건이 넘는 사용자 리뷰를 분석해, 챗봇이 정보·정서 맥락에서 어떻게 작동/실패하는지 분해해 보여줍니다. 특히 AI 의료 챗봇을 ‘information infrastructure(정보 인프라)’로 재정의해, 접근·사용성·신뢰의 실패가 사용자 경험을 어떻게 악화시키는지 관점의 전환을 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 리뷰라는 비정형 텍스트로부터 반복되는 실패 양상을 안정적으로 추출하는 것입니다. 연구진은 topic modeling과 해석적 분석을 결합해 접근 장벽과 서비스 불안정, 사용자 경험 및 상호작용 품질 저하, 과금·고객지원 문제라는 3가지 반복적 breakdown을 도출했습니다.

- **Empirical Impact**: 연구는 특히 privacy and security(프라이버시·보안) 우려가 가장 부정적인 경험과 함께 나타난다는 연관성을 실증적으로 제시합니다. 결과적으로 디지털 헬스 시스템의 설계자와 정책 담당자, 정보 전문가가 접근·사용성·신뢰를 중심으로 개선 우선순위를 세우는 데 활용 가능한 인사이트를 제공합니다.



### E-TTS: A New Embodied Test-Time Scaling Framework for Robotic Manipulation (https://arxiv.org/abs/2606.27268)
Comments:
          Accepted to ECCV 2026. 44 pages, 11 figures. Project page: this https URL

- **Prior Approaches**: 기존의 embodied test-time scaling(TTS) 연구는 주로 action 공간만 늘려 성능을 끌어올리는 데 집중해 왔습니다. 또한 reasoning 구성요소가 성능을 돕더라도, reasoning이 어떻게 스케일되는지(스케일 메커니즘)는 상대적으로 덜 다뤄졌습니다. 더불어 로봇 조작은 long-horizon·순차 의사결정이라 과거 맥락이 중요한데, 기존 TTS는 history와 피드백을 충분히 통합하지 못했습니다.

- **Core Contribution**: 본 논문은 reasoning과 action을 하나의 단위로 함께 스케일하는 plug-and-play 프레임워크 E-TTS를 제안합니다. E-TTS는 history buffer와 vision-language verifiers로 과거 맥락을 반영한 후보 평가를 수행하고, 샘플링 과정에 feedback을 생성해 closed-loop iterative refinement를 만듭니다. 결과적으로 서로 다른 vision-language-action(VLA) 모델에도 추가 학습 없이 모듈을 끼워 넣어 성능을 개선할 수 있도록 설계됐습니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) reasoning 결과와 action 생성이 강하게 결합돼 있으므로 둘의 정렬이 필요하고, (2) long-horizon에서 현재 관측만으로는 후보 검증의 기준이 부족하다는 점입니다. E-TTS는 reasoning-action joint sampling 및 pairwise joint scoring(Reasoning verifier의 zero-shot 점수 + Action verifier 점수)을 통해 최적 ⟨reasoning, action⟩ 쌍을 찾게 하고, history buffer로 시간 의존성을 입력·검증에 반영합니다. 또한 일정 임계값 미달 시 verifier가 실패 원인을 구조화된 텍스트 feedback으로 만들고, 이를 다음 라운드 프롬프트에 주입해 반복적으로 개선합니다.

- **Empirical Impact**: E-TTS는 4개 벤치마크·6개 환경·3개 embodiment에서 4종 VLA 모델(E-CoT, MolmoAct, π0.5, Embodied-R1)에 적용되어 일관된 성능 향상을 보였습니다. 추가 expert 데이터 수집이나 재학습 없이 시뮬레이션에서 최대 33.14%, 실세계에서 26.62%까지 성공률이 개선되며, 평균 향상도 유의미하게 나타났습니다. 또한 ablation 결과 reasoning scaling과 action scaling, feedback·history 설계가 함께 중요하며 단일 요소만 적용할 때 성능이 크게 떨어지는 점을 확인했습니다.



### Advancing Omnimodal Embodied Agents from Isolated Skills to Everyday Physical Autonomy (https://arxiv.org/abs/2606.27251)
- **Prior Approaches**: 기존 연구는 사이버 영역(API·IoT)과 물리 영역(조작·이동)을 따로 다루거나, VLM 기반 플래너가 통합된 cyber-physical action space를 충분히 지원하지 못했다. 또한 에이전트 프레임워크가 맥락을 무한히 누적해 temporal coherence가 떨어지고, VLA 정책은 open-loop로 실행해 자신의 물리적 실패를 스스로 감지·수정하지 못하는 문제가 있었다.

- **Core Contribution**: 이 논문은 persistent autonomy를 위해 단일 거대 모델이 아니라 planning, memory, verification을 명시적으로 분리한 계층적 비동기 아키텍처가 필요하다고 주장한다. 그 구현으로 OmniAct를 제안하며, unified action space에서 skill routing을 위한 multimodal semantic planner, 이벤트 경계 기반 압축으로 context 성장을 억제하는 계층형 메모리, 그리고 물리 실행 중 시각 기반 preemption으로 semantic loop를 닫는 엔진을 통합한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 이질적인 사이버·물리 도구를 하나의 실행 공간으로 조율하고, (2) 장시간 운용 시 context 누적이 시간적 일관성을 해치지 않게 하며, (3) open-loop 실행에서 벗어나 물리 실패를 감지해 즉시 복구하는 폐루프를 만드는 것이다. OmniAct는 통합 action space에서의 semantic planner로 실행 경로를 라우팅하고, 이벤트 경계-driven compression으로 메모리를 압축하며, 비동기 visual preemption engine으로 시각적 검증과 선제 중단을 통해 의미적 재계획을 수행한다.

- **Empirical Impact**: OmniAct는 두 개 로봇 플랫폼에서 4개의 IoT 디바이스를 함께 다루는 40개의 실제 장기 태스크에서 복잡도 전 구간에 걸쳐 end-to-end success가 일관되게 개선됐다. 또한 누적 interaction tokens가 100k+를 넘는 상황에서도 토큰 사용량이 거의 평탄하게 유지됐고, mid-scale open-weight 모델의 성능을 proprietary-level에 가깝게 끌어올렸다는 점에서 실용적 의미가 크다.



### From Celebrities to Anyone: Characterizing AI Nudification Content, Technology, and Community Dynamics on 4chan (https://arxiv.org/abs/2606.27234)
Comments:
          22 pages, 13 figures, 2 tables

- **Prior Approaches**: 기존 연구는 주로 MrDeepFakes 같은 딥페이크/누디피케이션 마켓플레이스나, Hugging Face·Civitai의 모델 저장소를 중심으로 위험을 관찰했다. 그러나 확산 모델 시대 이후의 “현장(wild)” 콘텐츠가 어떻게 요청-제공 구조로 생성·유통되는지, 특히 익명 커뮤니티 수준의 생태계 역학은 충분히 측정되지 않았다.
또한 과거에는 기술 장벽과 정보 가시성 때문에 타깃이 연예인 중심이었던 경향이 있어, 일반 개인에 대한 피해 확대는 체계적으로 과소평가됐다는 한계가 있었다.

- **Core Contribution**: 이 논문은 4chan Adult Requests 보드를 41일간 대규모로 수집해, 실제로 생성·교환되는 SNEACI의 규모와 양상을 정량화했다. 총 24,105개의 SNEACI를 식별하고, 타깃이 기존 연예인 중심에서 비(非)연예인으로 크게 이동했음을 실증적으로 보여준다.
특히 비연예인 비중이 55.78%(영상은 60.26%)로 관측되어, 피해가 사용자들의 지인/사회적 근접 집단까지 확장됐다는 점을 강조한다.

- **Technical Challenges**: 핵심 기술 과제는 방대한 멀티미디어에서 SNEACI를 신뢰성 있게 선별하는 것이었다. 연구진은 NSFW 필터링→AIGC(완전 합성) 탐지→맞춤 undress 탐지→(연예인 여부) 다중 단계 분류 파이프라인을 구축했고, 영상은 표본 점검 결과를 근거로 처리된 경우 SNEACI로 간주했다.
또한 연예인 분류는 고정된 신원 데이터에 의존하면 오탐이 커질 수 있어, Gemini 기반의 이중 에이전트 검증으로 정확도를 높이는 방식으로 해결했다.

- **Empirical Impact**: 실증 결과, 오픈소스 기반 모델과 공유된 fine-tuned 자원이 제작의 중심에 있으며 Stable Diffusion 계열(이미지 42.7%)과 Wan(동영상 66.5%)이 주도하는 공급망이 드러났다. 동시에 소수의 고활동 “프로바이더(핵심 생산자)”가 요청량과 타깃 구성을 좌우하며 커뮤니티를 자기강화적으로 지속시키는 생태계 역학도 관찰됐다.
이 연구는 SNEACI 확산이 단일 플랫폼의 문제가 아니라 다중 플랫폼 공급망과 툴 재배포에 의해 유지된다는 점을 보여, 플랫폼 거버넌스·기술적 안전장치·피해자 보호 개입이 시급함을 강하게 시사한다.



### Bridging Talk and Thought: Understanding Dialogue Dynamics Across Collaborative Problem-Solving Contexts (https://arxiv.org/abs/2606.27233)
- **Prior Approaches**: 기존 대화 연구는 담화 행위, 대화 상태, 담화 구조를 분류해 의도나 품질을 분석하는 데 집중해 왔습니다. CPS(협력적 문제 해결) 관점에서는 탐색적 발화와 사회정서 조절, 상호작용 대칭성 같은 틀로 협업을 보려 했지만, 인지·메타인지·사회정서가 한 흐름에서 어떻게 함께 작동하는지는 충분히 계량화되지 않았습니다. 특히 human–AI 협업은 종종 협력이라기보다 tool use나 반응형 보조에 그친다는 메타분석 결과가 반복되었지만, 이를 대화 패턴으로 구체적으로 분해해 진단하는 체계는 부족했습니다.

- **Core Contribution**: 이 논문은 CPS 대화를 metacognition(메타인지)·cognition(인지)·non-cognition(비인지/사회정서) 세 축으로 동시에 코딩하는 계층형 2-layer 체계를 제안합니다. 발화 단위로 조절 과정(계획·모니터링·평가)과 발화가 담당하는 조절 수준(SR/CR/SSR), 인지적 목표지향 발화 유형(7종), 사회정서 상호작용(3종)을 함께 라벨링해 “진짜 협업”과 “도구적 협력/비대칭”을 분리합니다. 나아가 메타인지 조절 불균형이 협업의 깊이를 가르는 핵심 판별자일 수 있음을 실증적으로 강조합니다.

- **Technical Challenges**: 핵심 난제는 CPS 대화에서 서로 다른 차원의 현상을 한 번에 안정적으로 식별하는 것입니다. 이를 위해 연구진은 메타인지 조절을 7개 과정과 3개 행동 수준으로 세분화하고, 주변 문맥까지 고려해 off-task 발화의 기능적 역할도 반영하도록 코딩 지침을 구성했습니다. 규모 확장도 필요했는데, GPT-4o를 LLM-as-a-judge로 활용해 발화별 라벨을 자동화하고, 사람 라벨과의 kappa로 일치도를 검증해 재현성을 확보했습니다.

- **Empirical Impact**: 9개 CPS 데이터셋(인간-인간 6종, 인간-AI 3종)에 대해 테스트한 결과, 조절·인지·사회정서 기여가 균형을 이룰 때 대화가 협업적으로 나타났습니다. 반대로 Minecraft·Itinerary처럼 맥락이 구조화된 과제에서는 한 참여자가 metacognitive leadership(메타인지 주도) 역할을 맡고 다른 쪽은 반응·실행 중심이 되는 비대칭이 관찰됐습니다. 특히 CoCoDial 계열에서는 인간이 self-regulation을 지배하고 AI는 co-regulated/SSR 쪽이 편향적으로 나타나, AI가 자율적 주도권을 공유하지 못해 협업이 얕아지는 이유를 설명합니다. 또한 metacognitive regulation 라벨을 제거하면 화자 역할 군집이 섞여 “메타인지 조절”이 협업 모델링의 필수 신호임을 보여줍니다.



### CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention (https://arxiv.org/abs/2606.27229)
Comments:
          27 pages, 2 figures, multiple tables. Submitted to arXiv. Primary category: cs.LG; cross-list: cs.CL

- **Prior Approaches**: Transformer는 모든 과거 토큰을 KV-cache로 유지해 성능은 강하지만, 학습·추론 비용이 길이에 따라 급격히 커집니다. 이에 비해 delta-rule 기반 순환 모델은 고정 크기 상태로 압축해 O(1) 메모리·비용에 가깝게 확장하지만, 성능을 좌우하는 “잊기” 게이트 설계에서 한계가 반복돼 왔습니다. 특히 GDN-2는 erase/write를 입력 토큰만 보고 결정해 메모리를 직접 참조하지 못하는 memory-blind gating 문제와, value-axis 결합으로 WY-form chunk solver의 청크 병렬성을 깨는 구조적 병목이 함께 존재했습니다.

- **Core Contribution**: CARVE(Content-Aware Recurrent with Value Efficiency)는 기존 delta-rule 계열의 결함 3가지를 하나의 원칙으로 정리합니다: erase는 key axis에만 수행합니다. 이 제약은 단순한 휴리스틱이 아니라, WY-form triangular chunk solver의 타당성을 유지하기 위한 필요충분 조건으로 증명됩니다. 또한 CARVE는 erase 게이트에 “저장된 내용” 신호를 제공하되 추가 메모리 읽기 없이, 이미 GPU에 써둔 recurrent output을 재활용해 content-aware forgetting을 구현합니다.

- **Technical Challenges**: 핵심 기술 난점은 게이트가 메모리 상태를 “읽어야” content-aware가 가능한데, HBM에서 상태 행렬을 다시 읽으면 순환 모델의 효율 이점이 상쇄된다는 점입니다. CARVE는 매 반복에서 HBM에 기록되는 output o_t=S_{t-1} q_t를 chunk 평균 m_c로 요약해 erase 게이트의 조건으로 사용함으로써, stale 신호의 오차가 chunk 길이 L에 따라 O(1/sqrt(L)) 수준으로 감소함을 이론적으로 보입니다. 더 나아가 write 측에서는 per-value write-gate projection을 head당 단일 scalar로 대체해 파라미터를 크게 줄이면서도 연관 저장 용량은 잃지 않음을 정리로 보장합니다.

- **Empirical Impact**: CARVE는 1.3B 파라미터를 FineWeb-Edu 100B 토큰으로 학습했을 때 WikiText perplexity 15.72를 달성했으며, GDN-2 대비 -0.18(4.5-sigma)을 기록했습니다. 또한 9개 상식 추론 벤치마크에서 모든 recurrent baseline을 앞섰고, RULER in-context retrieval 프로브에서는 모든 설정에서 state of the art를 달성했습니다. 하드웨어 관점에서도 throughput 오버헤드는 0.4% 이내, 피크 메모리는 13% 감소, 파라미터는 19% 적어져 delta-rule의 “품질-비용” 균형을 재정의했다는 평가를 받습니다.



### Automating Potential-based Reward Shaping with Vision Language Model Guidanc (https://arxiv.org/abs/2606.27180)
- **Prior Approaches**: 희소 보상 환경에서 강화학습은 중간 피드백이 없어 탐색이 거의 랜덤에 가깝고, 성공 보상을 정확히 어떤 궤적 요소가 만들었는지 귀속하기도 어렵다. 이를 완화하려고 보상 shaping을 쓰면 reward hacking 위험이 커지며, 특히 임의의 shaping은 목표와 무관하게 보조 신호만 최대화하는 정책을 유도할 수 있다. Potential-based reward shaping(PBRS)은 최적 정책 집합을 보존하지만, 핵심인 잠재함수(Φ)를 전문가 지식과 공학으로 설계해야 한다는 부담이 남아 있었다.

- **Core Contribution**: 본 논문은 VLM-PBRS로, 비전-언어모델(VLM)이 이미지 쌍에 대한 preference 라벨을 생성하고 이를 학습해 PBRS의 potential function을 자동 구성하는 프레임워크를 제안한다. 사용자는 목표에 대한 짧은 텍스트 설명과 관측(이미지)만 제공하면 되고, 보상 shaping 항을 수동으로 설계할 필요가 줄어든다. 또한 PBRS의 정책 불변성 덕분에 라벨 정확도에 대한 요구를 완화하여, 반복 호출 비용이 큰 대형 VLM 대신 더 작은 VLM을 사용해도 학습을 가속할 수 있음을 강조한다.

- **Technical Challenges**: 기술적 난점은 (1) VLM이 만든 preference 라벨이 노이즈/오차를 포함할 수 있는데도 PBRS로 정책 불변성을 유지해야 하고, (2) 정책 학습 중 VLM을 반복 호출해야 하므로 계산 비용을 낮춰야 한다는 점이다. 해결을 위해 VLM 출력에 대한 추가 LLM 호출 없이 단일 프롬프트로 preference 라벨을 뽑고, 학습된 선호 모델을 sigmoid+스케일링으로 잠재함수에 매핑해 goal reward의 부호/크기 조건을 만족하도록 설계한다. 그 결과, 라벨이 불완전해도 최적성 자체를 해치기보다는 sample efficiency 개선 폭이 줄어드는 방향으로 작동하게 만든다.

- **Empirical Impact**: Meta-World와 Franka Kitchen에서 VLM-PBRS는 희소 기준선보다 일관되게 학습을 빠르게 만들었고, 보상 hacking 없이 목표 성능을 향해 수렴하는 경향을 보였다. 반면 RL-VLM-F처럼 learned reward를 직접 보상으로 쓰는 방식은 VLM 라벨 오류가 잘못된 유인을 만들어 일부 태스크에서 부분 최적 정책으로 수렴하는 문제가 나타났다. 또한 실험 분석을 통해 small VLM preference label의 정확도가 sample efficiency 향상과 연결된다는 점을 보여, VLM 라벨 품질과 학습 이득의 관계를 정리했다.



### Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline) (https://arxiv.org/abs/2606.27163)
Comments:
          Solution of the LeHome Challenge at ICRA 2026

- **Prior Approaches**: 기존 연구/대회 방식은 주로 행동복제(BC)로 시작해 성능을 끌어올리거나, PPO 계열처럼 로그확률 기반 정책그래디언트를 flow-matching 기반 VLA에 그대로 적용하려는 시도가 많았다. 하지만 BC 데이터는 보통 “성공 궤적” 위주라 실패 회복 능력이 약하고, PPO류는 flow-matching의 유효 행동 manifold 밖으로 예측이 밀려나기 쉬워 안정적인 개선이 어렵다. 또한 희소한 이진 보상에서는 중간 성과 신호를 엔지니어링해야 해 학습이 더 까다롭다.

- **Core Contribution**: 이 논문은 LeHome Challenge 2026의 양팔 의류 접기에서, VLA 정책을 강화학습 루프로 개선하되 “정책이 곧 value function이 되게” 만든 것이 핵심 기여다. 즉 행동을 예측하는 동일 네트워크가 성공/진행/미래 핵심점 거리 등의 양을 함께 예측하고, 이 예측을 advantage 추정, 실시간 failure detection, 후보 선택에 직접 활용한다. 결과적으로 한 모델로 학습·추론 신호를 통합해 서빙/튜닝 복잡도를 줄이면서도 성공률을 끌어올렸다.

- **Technical Challenges**: 문제는 (1) 변형 물체의 미세한 궤적 차이가 상태를 크게 바꾸는 점, (2) 관측 가능한 중간 보상이 거의 없고 성공이 이진인 점, (3) 평가 시 의류 종류 라벨이 주어지지 않는 점, (4) 시뮬-실세계 격차가 큰 점이다. 저자는 AWR+RECAP을 결합해 좋은 구간에 확률질량을 재배치하면서도 flow-matching의 유효 예측공간을 벗어나지 않게 했고, RECAP 방식 advantage conditioning으로 inference-time에서도 guidance 형태의 선택을 가능하게 했다. 더불어 시뮬에서의 강한 데이터 증강, success/ failure 상태 스냅샷 기반 하드 마이닝, Thompson sampling을 통한 inference-time 하이퍼파라미터 탐색, 카메라 정렬 도구를 포함한 sim-to-real 파이프라인과 DAgger-like HIL 수집으로 회복/일반화 난제를 완화했다.

- **Empirical Impact**: 실험 결과, 온라인(시뮬) 라운드에서 62팀 중 1위를 차지했으며 전체 성공률 79.63%로 2위와도 큰 격차를 보였다. 실세계 결선에서는 시뮬-실세계 전환 튜닝을 수행한 끝에 2위를 기록해 실제 로봇에서도 성능을 유지함을 입증했다. 특히 “정책 내부의 value/Q 대체와 advantage 기반 라이브 실패 감지”를 실제 대회 수준으로 적용한 레시피로, 변형물체 조작 VLA 강화학습 방향에 실용적 기준점을 제시했다.



### Safe Autoregressive Image Generation with Iterative Self-Improving Codebooks (https://arxiv.org/abs/2606.27147)
Comments:
          10 pages including references, 8 figures, accepted for publication at the 43rd International Conference on Machine Learning (ICML 2026)

- **Prior Approaches**: 기존 연구는 확산모델 기반 T2I(텍스트-이미지 생성)에서의 안전성 문제를 주로 다뤘으며, 사후 필터링(NSFW 탐지)이나 확산의 연속 잠재공간/임베딩 공간에서의 가이던스·학습으로 위해 개념을 줄이는 방식이 많았다. 하지만 토큰 기반 자가회귀 unified multimodal 모델은 연속 공간이 아니라 discretized visual tokens와 codebook에 의존해, 이런 연속공간 중심 방법은 일반화가 어렵다. 또한 외부 인간 라벨이나 인간 피드백이 필요한 안전화 파이프라인은 비용과 확장성에서 한계가 있었다.

- **Core Contribution**: 이 논문은 autoregressive unified multimodal 모델에서 안전한 생성을 위해 codebook 자체를 “iterative self-improving” 방식으로 개선하는 Safe-CodeBook을 제안한다. 핵심 아이디어는 모델이 자기 자신으로 unsafe 결과를 판별하고, harmful과 safe에 대응하는 공간을 구성해 harmful mapping을 제거한 뒤, 품질 저하를 막기 위해 null space에서만 적응 fine-tuning을 반복한다. 추가적인 외부 피드백 없이도 위험한 출력이 점진적으로 줄어드는 self-improvement 루프를 설계한 점이 기여다.

- **Technical Challenges**: 자기 모델 내부에서 unsafe를 라벨 없이 식별해 harmful 개념에 해당하는 codebook의 “어떤 방향”을 건드려야 하는지가 첫 번째 난관이다. 논문은 모델의 판단으로 harmful image-text 쌍을 만들고, safe-unsafe 임베딩 차이를 모아 SVD로 harmful subspace를 추출한 뒤 코드북 임베딩을 투영 제거해 harmful 방향을 억제한다. 두 번째 난관은 하드 제거가 시각 품질을 떨어뜨릴 수 있다는 점인데, 이를 null space projection을 이용해 harmful 정보와 무관한 섭동만 학습하도록 제약해 품질을 보존하면서 안전성을 유지한다.

- **Empirical Impact**: 실험은 I2P, CoPro, ViSU 등 8종의 위해 프롬프트 데이터셋과 성적 위해 개념 중심의 추가 벤치마크들에서 안전성 개선을 일관되게 확인했다. 탐지기 기준으로 harmful 생성이 유의미하게 감소했으며, COCO-30k에서의 FID 측정과 기존 모델의 생성/이해 능력 평가를 통해 원래 역량 저하가 크지 않음을 함께 보여준다. 또한 단일 제거보다 반복 제거가 특정 위해 개념에서 더 잘 작동하고, Janus 등 여러 unified multimodal generation 모델과 OOD 상황에서도 적용 가능성이 보고되어 분야 적용성이 강화된다.



### Efficient foundation decoders for fault-tolerant quantum computing (https://arxiv.org/abs/2606.27119)
Comments:
          32 pages, 9 figures, comments are welcome

- **Prior Approaches**: 기존 디코딩은 MWPM, belief propagation, matching 기반 등 전통적 방법과, 학습 기반 신경 디코딩으로 나뉜다. 다만 대칭/대규모 코드거리에서는 신드롬 생성 비용과 최적화 부담이 급증해 기계적 확장이 어렵고, foundation decoder 역시 작은 코드거리에서만 효율적으로 학습되는 병목이 있었다.

- **Core Contribution**: 이 논문은 neural transfer unification(NTU)으로 foundation decoder를 코드거리 간 전이(transfer)로 확장하는 통합 프레임워크를 제안한다. 핵심은 코드 계열이 공유하는 대수적(algebraic) 구조로 decoding task를 정렬해, 작은 코드에서 학습한 지식을 큰 코드거리 학습을 가속하는 데 재사용한다. 이를 트랜스포머 기반 NTU-Transformer로 구현해 planar surface code와 bivariate bicycle(BB) code에 적용한다.

- **Technical Challenges**: NTU의 첫 난점은 코드거리마다 검출기(detector) 집합 크기와 입력 인덱싱이 달라 “거리 독립” 신드롬 표현을 만들기 어렵다는 점이다. 논문은 검출기의 이웃을 상대적 상대이동(relative shift) 집합으로 인코딩해 입력 임베딩을 거리 불변으로 만들고, 전이 시 positional mismatch를 막기 위해 QEC-aware rotary positional encoding을 설계한다. 이후 smaller 거리에서 학습한 파라미터를 더 큰 거리로 이어가며 fine-tuning하는 transfer adaptation으로 cold-start를 완화한다.

- **Empirical Impact**: 실험에서 NTU-Transformer는 planar surface code의 [[361,1,19]]에서 correlation-aware matching을 능가하고, [[625,1,25]]로도 표준 matching보다 낮은 logical error rate를 보이며 large-scale 학습 재시작 부담을 줄인다. BB 코드의 [[72,12,6]]에서는 low-physical-error 영역에서 Relay-BP를 앞서고, detector-error weight가 큰 구간에서도 비국소(non-local) 상관을 잘 반영한다. 또한 BB에서 [[72,12,6]]→[[144,12,12]] 전이 시 random initialization 대비 높은 성능과 빠른 수렴을 보여 NTU가 “cross-distance amortized training” 경로를 제공함을 입증한다.



### Heavy-Ball Q-Learning with Residual Weighting Correction (https://arxiv.org/abs/2606.27112)
- **Prior Approaches**: Q-learning은 Bellman 최적성 갱신만으로 Q*를 학습하지만, 할인계수 γ가 1에 가까워지면 Bellman 최적성 연산자의 수축성(고정점 수렴 보장에 쓰이는 수축)이 약해져 학습 속도 이론 보장이 제한적이었습니다. 이를 개선하려는 speedy Q-learning, momentum 기반 Q-learning, PID/가속 value iteration, rank-one correction/deflation 등은 존재하지만, “수정된 재귀가 표준 Q-learning보다 빠르다”를 제어 문제까지 포함해 이론적으로 확실히 증명한 경우는 상대적으로 적었습니다. 특히 greedy 정책이 궤적에 따라 바뀌는 제어 setting에서는 기존 방식의 속도 보장 가정이 깨지기 쉬웠습니다.

- **Core Contribution**: 이 논문은 가장 단순한 heavy-ball Q-learning 재귀를 수정해 corrected heavy-ball Q-learning을 제안하고, 수렴뿐 아니라 “어떤 조건에서 표준 Q-learning보다 더 빠른(인증된 더 작은) 수렴률”이 이론적으로 보장됨을 제시합니다. 수학적으로는 평균 동역학을 switched linear system(SLS)으로 보고, 공통 고유벡터 방향에서만 가속이 ‘정확한 인증(certificate)’ 형태로 도출되며, 직교 방향에서는 동일한 가속 보장이 없다는 점까지 명확히 합니다. 또한 이 구성은 linear function approximation이 있는 Q-learning에도 확장되어 유사한 수렴 및 가속 진술을 얻습니다.

- **Technical Challenges**: 핵심 난제는 greedy 행동 선택으로 인해 모드가 바뀌는 제어 설정에서, 단순 고정-정책의 스펙트럼/수축 논리로는 속도 향상을 깔끔하게 비교하기 어렵다는 점입니다. 이를 해결하기 위해 Q-learning의 mean dynamics를 SLS로 모델링하고, 각 모드를 결정하는 switching 가족에 대해 joint spectral radius(JSR)로 최악-경로 지수 성장률을 비교합니다. 더 나아가 corrected 업데이트는 특정 공통 고유벡터가 유지되도록 ‘correction’을 설계해, JSR 기반 Lyapunov-norm 구성으로 수렴률을 수학적으로 연결할 수 있게 합니다.

- **Empirical Impact**: 논문은 주로 결정론적 mean dynamics에서 JSR Lyapunov-norm로 수렴과 속도 인증을 도출하며, 동일 업데이트가 model-free stochastic RL 재귀(샘플링이 들어간 경우)로도 구현 가능하다고 설명합니다. i.i.d. 관측 샘플링에서는 conditional-mean과 잡음 분해를 결합해 stochastic 수렴/유한시간 오류 항까지 확장 가능한 틀을 제공하고, Markovian 관측은 표준 stochastic approximation 도구로의 확장이 가능하다고 언급합니다. 결과적으로 heavy-ball momentum이 Q-learning을 “어떤 기하학적 방향에서” 실제로 가속할 수 있는지에 대한 새 분석 프레임(SLS+JSR)을 제안해, 이후 가속형 Q-learning/벨먼 반복 분석에 참고할 만한 통찰을 제공합니다.



### Application of LLMs to Threat Assessment of Foreign Peacekeeping Missions (https://arxiv.org/abs/2606.27106)
- **Prior Approaches**: 평화유지·분쟁지역의 위협 평가는 전통적으로 인간 분석(HUMINT)과 함께 OSINT를 보조적으로 활용해 왔지만, 미디어에서 위협을 구조화해 추출하는 자동화는 아직 초기 단계입니다. 기존 접근은 위험모델을 정교화하더라도, 실제로 시간에 따라 변하는 대량의 다국어 미디어를 위협 단위로 매핑·근거화하는 과정에서 사람의 노동이 크게 남는 한계가 있습니다.

- **Core Contribution**: 이 논문은 평화유지 임무에서 LLM을 활용해 OSINT 기반 미디어 내용을 임무 관련 위협으로 전환하는 반자동 워크플로를 제안합니다. PINPOINT 프로젝트의 지표 기반(모듈형) 위험모델 위에, 미디어에서 위협을 추출한 뒤 구조화·근거화(grounding)하는 단계를 결합해 분석가가 다룰 수 있는 형태의 위협 정보를 만듭니다.

- **Technical Challenges**: 핵심 과제는 (1) 다국어·비정형 미디어에서 지표/위협에 해당하는 의미를 뽑아내고, (2) 추출 결과가 임무에 실제로 relevant한지 근거를 확보하며, (3) 위협 수준·행위자 같은 민감한 속성을 과도하게 단정하지 않는 것입니다. 논문은 지표의 언어적 설명을 바탕으로 indicator-specific prompt를 만들고 few-shot + reasoning 모드로 JSON 후보 위협을 생성한 뒤, 추가 LLM 단계로 임무 관련성 없는 항목을 제거하고 위치·시간·행위자 등을 포함한 구조화 정보를 생성합니다.

- **Empirical Impact**: EUMM 조지아 사례에서 미디어 문서로부터 자동 생성한 위협에 대해 도메인 전문가 평가를 수행했으며, 위협 자체·임무 관련성·위치 관련성 같은 핵심 항목에서 높은 일치도(평균 약 0.82)를 보였습니다. 다만 위협 강도(threat-level)와 행위자(actors)는 상대적으로 낮게 평가되어, 논문은 human-in-the-loop와 운영 안전장치의 필요성을 강조합니다.



### Data-Free Reservoir Features for Efficient Long-Horizon Cold-Start Continual Learning (https://arxiv.org/abs/2606.27095)
- **Prior Approaches**: Cold-start exemplar-free class-incremental learning(CS-EFCIL)은 사전학습/큰 초기 태스크 없이 스트림을 이어받아야 해, 표현학습이 성능을 좌우한다. 기존 방식은 (1) 백본을 계속 학습하며 semantic drift를 보정하거나, (2) 첫 태스크 뒤 백본을 고정해  drift는 막지만 초기 편향 때문에 전이성이 떨어지는 문제가 있다. 또한 drift 보정 계열은 장기 태스크에서 계산·업데이트 비용이 커지고, 고정 백본 계열은 cold start에서 약해진다.

- **Core Contribution**: 이 논문은 “이미지 데이터에 한 번도 맞추지 않은(data-free) 고정 특성”을 택하는 제3의 옵션을 제안한다. CIRCLE은 BiRC2D에서 착안한 고정된 bidirectional 2D reservoir 기반 특징 추출기와, 스트리밍 linear discriminant analysis(SLDA) heads를 결합해 replay 없이 클래스 단위 업데이트를 수행한다. 더 나아가 reservoir 특성 군집(피처 앙상블)과 SLDA 확률 평균(프레딕션 앙상블)을 2단으로 설계해 bias-variance tradeoff를 조절한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘학습되지 않은 고정 특징’이 cold start에서도 충분히 구별력을 갖도록 만드는 것이다. CIRCLE은 공간 구조를 유지하는 BiRC2D형 reservoir 특징을 쓰고, SLDA head는 additive sufficient statistics로 닫힌형 업데이트를 제공해 backbone backpropagation 없이도 순서 불변에 가깝게 학습 효과를 만든다. 또한 단일 reservoir의 잡음을 줄이기 위해 여러 무작위 reservoir 인스턴스를 그룹화해 특성 연결과 예측 평균을 함께 적용한다.

- **Empirical Impact**: CIRCLE은 CIFAR-100, TinyImageNet, ImageNet-Subset, ImageNet-1k에서 10~20개 태스크 분할에서는 경쟁적이며, 50~500 분할에서는 strong CS-EFCIL 기준선을 크게 능가한다. 특히 trained-backbone drift-compensation 방법이 장기에서 10~30%p 수준으로 붕괴하는 반면, CIRCLE은 전 태스크에 걸쳐 성능이 안정적으로 유지된다. 학습 비용도 빠른데, 여러 reservoir를 쓰지만 모두 최적화 대상이 아니라 전방 패스와 SLDA 통계 누적으로 처리되어 trained-backbone 대비 훨씬 빠르게 스트림을 끝낸다.



### Inherited Circuits, Learned Semantics: How Fine-Tuning Creates Evasion Vulnerabilities Invisible to Standard Evaluation (https://arxiv.org/abs/2606.27091)
- **Prior Approaches**: 보안 분류를 위해 LLM을 fine-tuning한 뒤 평가는 보통 학습 분포와 동일한 held-out 예시 정확도로 끝납니다. 하지만 PowerShell처럼 기능은 유지하면서 표현만 바꾸는 변환(별칭 치환, 문자열 재구성, 실행 우회, 대소문자 변형)은 학습 분포 밖에서 값싼 공격 수단이어서, “정확도는 유지되는데 실패하는” 취약면이 숨겨질 수 있습니다. 기존 해석 연구는 circuit을 causal intervention으로 찾을 수 있음을 보여왔지만, 보안 fine-tuning이 그 circuit의 ‘사용 방식’을 어떻게 바꿔 evasion 표면을 넓히는지는 명확히 분리되지 않았습니다.

- **Core Contribution**: 이 논문은 security fine-tuning이 모델이 토큰 수준 indicator 의미에 과도하게 의존하도록 만들 수 있어, canonical 성능만으로는 위험을 과소평가할 수 있음을 실증합니다. Foundation-Sec-8B-Instruct와 그 베이스 Llama-3.1-8B-Instruct를 PowerShell 분류 코호트에서 비교한 결과, 파인튜닝은 분류 회로 자체를 새로 만들기보다 상속된 late-attention 경로를 더 집중·의미 특화하며, 그 결과 transformation에 민감한 signal reversal(억제/역전) 취약면이 생깁니다. 또한 이 현상을 사후가 아니라 pre-deployment 단계에서, canonical clean 입력만으로 어떤 command family가 취약한지 우선순위화하는 방법을 제시합니다.

- **Technical Challenges**: 핵심 기술적 난관은 “변환 후 실패”를 단순히 토큰이 안 보였다/사라졌다 수준이 아니라, 내부 표현에서 무엇이 바뀌었는지 분해해 측정하는 것입니다. 논문은 causal interventions(패치/어블레이션, matched cohort 293쌍)를 통해 분류 회로를 Layer 12–13 근처로 국소화하고, evasion 성공에서도 late-layer 증거는 여전히 활성화되지만 Layer 13 경계 직전 MLP 연산이 그 효과를 뒤집는다는 메커니즘을 보여줍니다. 더 나아가 pre-deployment 모니터링으로는 classification boundary에 대한 linear probe와 indicator-token sign test를 결합해, fine-tuning 이후 indicator 토큰의 역할이 driver에서 suppressor로 전환된 family를 탐지하도록 설계했습니다.

- **Empirical Impact**: 3단계 evasion benchmark에서 Foundation-Sec은 iwr alias substitution, Invoke-Expression format-string reconstruction, 대소문자 변형된 Invoke-Expression/IEX 계열에서 misses가 발생했지만, 동일 조건에서 Llama는 해당 변형 세트에서 0/440~0/630 수준의 안정적인 정답을 보였습니다. 흥미롭게도 adversarial prompt(의미 중심)로 일부 alias 기반 취약면은 완화되었지만, 반대로 Invoke-Expression 계열 miss가 늘며 “한 family를 고치면 다른 family가 드러날 수 있음”을 시사합니다. 제안된 sign test/linear probe 기반 모니터링은 evasion 변형 생성 없이도 취약 command family를 랭킹해 red-team 우선순위를 정교화할 수 있으며, 결론적으로 small task-specific fine-tuning이 곧바로 더 안전한 보안 분류기가 된다는 가정을 경고합니다.



### Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inferenc (https://arxiv.org/abs/2606.27090)
Comments:
          28 pages, 3 figures, 2 tables

- **Prior Approaches**: 기존 베이지안 추론에서는 KL divergence나 ELBO 같은 전역 기준으로 분포 간 불일치를 측정해 왔다. 하지만 전역 유사도는 특정 파라미터 주변(고정된 θ)에 확률질량이 얼마나 ‘빽빽하게’ 모이는지(작은볼, small-ball scaling)를 결정하지 못할 수 있다. 특히 sparse·singular·제약이 있는 베이지안 모델에서는 전역 지표만으로 국소 거동을 놓칠 위험이 커진다.

- **Core Contribution**: 이 논문은 베이지안 추론에서 국소 확률질량의 거동을 p(B_r(θ))의 r→0 비율로 추적하는 ‘local-mass framework’를 제안한다. 이를 요약하는 도구로 Mass Index(파워 성분과 로그 보정 성분)를 정의해, 예컨대 smooth prior와 horseshoe-type prior처럼 전역은 비슷해도 로그차수까지 다른 상황을 분해해 설명한다. 또한 전역이 아닌 set-localised regularised f-divergence인 RE-KL(regularised extended KL)을 도입해 국소 small-ball 질량을 비교하는 이론적 비교식을 만든다.

- **Technical Challenges**: 핵심 난제는 전역 KL/ELBO가 포착하지 못하는 ‘국소 질량의 스케일(파워·로그)’을 안정적으로 비교·증명해야 한다는 점이다. 연구진은 (1) likelihood의 국소 정규성(local regularity)이 주어질 때 Bayesian update가 파워/로그 스케일을 어떻게 보존하는지, (2) 파라미터 의존 지지(support)가 있을 때는 하드 제약으로 사라지는 사전 질량의 ‘잔존 비율’이 posterior의 Mass Index를 어떻게 이동시키는지를 다룬다. 더불어 hard constraint를 smoothing할 때는 작은 이웃 한계와 zero-temperature 한계가 교환되지 않아 국소 구조가 달라질 수 있음을 보인다.

- **Empirical Impact**: 수치 실험에서는 국소 질량의 다양한 형태(규칙적 질량, 고갈(depletion), cusp, 로그 보정, 원자적 질량)를 합성 예제로 통제해 이론이 예측하는 국소 거동 변화를 관찰한다. UCI 기반 베이지안 로지스틱 회귀 예시에서는 전형적 베이지안 업데이트가 Laplace mean 근처의 posterior 질량을 크게 늘리되, 선도 파워 차수는 보존되는 양상이 드러난다. 또한 local RE-KL의 방향성(한 방향은 bounded, 다른 방향은 diverging)을 장난감 예제로 확인하며, 국소 불일치 비교가 실질적으로 유용함을 시사한다.



### Parametric Open Source Games (https://arxiv.org/abs/2606.27068)
Comments:
          ICML Workshop New Frontiers in Game-Theoretic Learning-NExT-Game

- **Prior Approaches**: 기존 오픈소스 게임이론 연구는 에이전트의 행동이 상대의 의사결정 절차에 의존할 수 있다는 점을 다뤘지만, 대부분 이산적 혹은 상징적 프로그램에 기반했습니다. 그 결과 연속 매개변수 공간에서의 학습 역학이나 평형 구조가 어떻게 바뀌는지 일반적으로 설명하기 어려웠습니다.

- **Core Contribution**: 이 논문은 이산 프로그램 평형의 연속체로서 ‘parametric open-source games(매개변수형 오픈소스 게임)’를 제안합니다. 플레이어는 매개변수 벡터를 선택하고, semantics map이 전체 매개변수 프로필을 기반게임의 mixed actions로 변환해 연속적인 평형 개념을 구성합니다.

- **Technical Challenges**: 핵심 과제는 연속 매개변수 환경에서 평형의 존재성과 판별 규칙을 제공하는 것입니다. 저자들은 평형 존재 결과를 세우고, 대칭 2×2 게임에서 selfish gradient ascent가 협력으로 전환되는 ‘정확한 coupling threshold(결합 임계값)’을 도출했으며, parametric program Nash equilibria에 대한 1차원 boundary test도 제시했습니다.

- **Empirical Impact**: 또한 neural semantics 클래스로 확장해, 1차 협력 조건이 ‘cross-player sensitivity(상대 민감도)’와 ‘self-player sensitivity(자기 민감도)’의 비율에 의해 좌우됨을 보였습니다. 대표 게임들에서 내부 매개변수 접근이 학습 동역학과 평형 구조를 질적으로 재편하며, 결합이 충분히 강하면 이기적 최적화조차 협력적 결과로 유도할 수 있음을 보여줍니다.



### NuclearQAv2: A Structured Benchmark for Evaluating Domain-Science Competence in Large Language Models (https://arxiv.org/abs/2606.27047)
- **Prior Approaches**: 기존에는 MedQA, LegalBench처럼 특정 전문영역 지식·추론을 겨냥한 벤치마크가 많았지만, 기술 실무에서 요구되는 정량 계산과 개념 이해를 함께 조합해 검증하는 설계는 상대적으로 부족했습니다. 수학·논리 벤치마크(GSM8K, MATH 등)는 도메인 지식보다 추상적 문제해결에 초점이 있어 원자공학 맥락의 신뢰성 평가와는 거리가 있었습니다. 핵공학 분야의 선행 NuclearQA는 규모가 작고, 구축·평가에 전문가 투입이 많이 필요해 확장성과 재현성이 약했습니다.

- **Core Contribution**: 이 논문은 핵공학 지식에 대한 LLM 성능을 체계적으로 평가하기 위한 NuclearQAv2 벤치마크를 제안합니다. 벤치마크는 약 1,240개의 QA를 boolean(사실 검증), numeric(정량 추론·계산), verbal(개념 이해)로 나눠 능력의 서로 다른 축을 분리 측정합니다. 또한 혼합 파이프라인으로 전문가 질문, 기존 데이터, 도메인 기술 코퍼스 기반 LLM 생성 QA를 결합해 확장성을 확보했습니다.

- **Technical Challenges**: 기술 도메인에서는 환각뿐 아니라 수식 적용·계산·개념 매칭이 동시에 실패할 수 있어, 단일 정확도 지표로는 신뢰성을 담보하기 어렵습니다. 특히 답 형식이 서로 달라 boolean은 정규화된 exact match로, numeric은 반올림·상수 차이를 허용하는 tolerance-based 정확도로, verbal은 동의어·의역을 고려한 LLM judge 기반 의미 동치 판정으로 평가합니다. QA 생성 단계에서도 비상황의존성, 모호성 최소화, 답의 간결성을 강제하는 structured prompting과 필터링을 적용해 저품질 문항이 섞이지 않도록 했습니다.

- **Empirical Impact**: 여러 LLM을 NuclearQAv2로 평가한 결과, 대체로 boolean은 높게 나오지만 numeric과 verbal에서 성능 격차가 크게 벌어졌습니다. 즉, 사실 인지는 상대적으로 성숙했더라도 핵공학의 정량 추론은 모델 간 변별력이 크고, 개념 이해도 스케일과 모델 특성에 영향을 받는다는 점이 확인됩니다. NuclearQAv2는 멀티-faceted 평가 프레임워크의 필요성을 실증하며, 향후 기술 도메인에서 재사용 가능한 확장형 벤치마크로 자리잡을 가능성을 제시합니다.



### The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture for AI-Assisted Software Developmen (https://arxiv.org/abs/2606.27045)
- **Prior Approaches**: 기존 spec-driven 코딩 에이전트 연구는 보통 spec-first(먼저 사양 작성) 또는 spec-as-source(사양으로부터 코드 생성) 쪽에 치우쳤습니다. spec-first는 선행 비용과 “잘못된 사양” 리스크가 있고, spec-as-source는 생성/동기화 과정에서 비결정성과 fragile한 단일 진실원 문제를 낳아 실무 채택이 어렵다는 지적이 나옵니다.

- **Core Contribution**: 이 논문은 Spec Growth Engine을 제안하며, context explosion과 silent spec-code drift를 동시에 다루는 “가벼운(lean) 통합 레이어”를 제공합니다. 핵심은 machine-readable spec graph 위에 Spine 기반 컨텍스트 조립, vertical-slice(하드스트-퍼스트) 성장 프로토콜, 그리고 spec-code divergence를 막는 drift gate를 얹어 spec-anchored이면서도 code-coupled로 유지하는 구조입니다.

- **Technical Challenges**: 에이전트가 저장소 전체를 자유 검색하면 관심 없는 모듈이 섞이고 quality가 context rot 형태로 떨어지는 문제가 있고, 사양 문서가 갱신되지 않는 drift는 CI/린터로는 잘 드러나지 않습니다. 논문은 Spine에서 ownership path와 one-hop only contract만 컨텍스트로 제공해 scoping을 강제하고, 에이전트가 같은 커밋에서 spec delta를 갱신하도록 하되 드리프트 검증에서 orphan code/undeclared dependency/dependency bypass 같은 하드 오류를 merge 이전에 차단합니다.

- **Empirical Impact**: 정량 실험 수치가 본문에 상세히 나열되진 않지만, 에이전트 품질이 장문 컨텍스트에서 급락할 수 있다는 선행 연구 근거와 함께 설계가 그 원인을 구조적으로 차단함을 강조합니다. 또한 worked example(체크아웃에 할인 코드 추가)에서 vertical slice 성장과 드리프트 게이트가 어떻게 잘못된 내부 수입이나 미선언 경계를 막는지 시나리오로 보여주며, 실무에서 깨지기 쉬운 spec/코드 불일치를 “규율이 아닌 불가능성”으로 바꾸는 점에서 의미가 있습니다.



### State Representation Matters in Deep Reinforcement Learning: Application to Energy Trading (https://arxiv.org/abs/2606.27032)
- **Prior Approaches**: 에너지 저장(양수·발전) 차익거래는 가격을 완벽/예측 정보로 가정한 최적화·평가 접근이 많았고, RL은 불확실한 미래가격 하에서 모사환경과 반복 상호작용으로 정책을 학습하는 대안으로 제시돼 왔다. 다만 기존 연구들은 어떤 ‘상태 입력(특징)’이 전이(transfer)를 좌우하는지 요소별로 분리해 검증하는 데는 상대적으로 소극적이었다. 따라서 시장이 바뀌거나 다른 ENTSO-E bidding zone으로 넘어갈 때 성능이 떨어지는 원인을 기능 설계 관점에서 명확히 규명하기 어려웠다.

- **Core Contribution**: 이 논문은 hydro storage arbitrage 환경 HydroDam에서 model-free RL(고정 Double DQN)로 “절대(absolute)·상대(relative)·예측(forecast) 가격 특징”이 성능과 일반화에 미치는 영향을 통제된 실험으로 정리한다. 환경(동역학, action, 보상), 네트워크, 학습 프로토콜은 고정하고 시장 특징(state representation)만 바꿔 ablation을 수행한다. 또한 같은 벨기에 시장의 시간대가 다른 구간(2012–2025)과 39개 다른 유럽 bidding zone(미선택 구간)에서 선택된 정책을 재평가해 전이 성능을 직접 비교한다.

- **Technical Challenges**: 핵심 난제는 가격 스케일/변동성/음수가격/일별 패턴이 훈련 구간과 다른 상황에서, RL이 어떤 입력이 ‘정책의 기준’ 역할을 하도록 학습할지 제어하는 것이다. 절대 특징만 쓰면 검증 성능은 높아도 학습 당시 가격 스케일에 의존해 전이가 무너지고, 상대 특징만 쓰면 현재가 ‘얼마나 큰 기회인지’가 정규화로 희석돼 휴리스틱을 넘기 어렵다. 이를 해결하기 위해 price scale(절대) + 최근 맥락(상대: z-score, rolling price score 등) + 24시간 재귀형 forecast(ensemble 추정치와 불확실성)를 조합하고, 체크포인트는 검증 성능으로 선택해 feature별 과적합을 드러내도록 설계했다.

- **Empirical Impact**: 결과적으로 단일 특징군은 신뢰성이 낮았다: 절대 특징은 벨기에 동일시장에서는 높지만(검증 기반으로는 강점) 2012–2025 테스트에서는 28.8%까지 급락했다. 상대-only(교차존 중앙 1.3%)와 forecast-only(교차존 중앙 26.8%)도 rolling price-score 휴리스틱을 전역에서 안정적으로 이기지 못했다. 반면 특징군을 결합하면 큰 개선이 나타나 absolute+relative는 테스트 49.9%, 교차존 중앙 39.8%, absolute+relative+forecast는 테스트 55.6%와 교차존 중앙 47.5%를 기록하며 39개 존 평균에서 휴리스틱을 우세하게 이겼다. 즉 저장거래 RL에서 상태 표현은 사소한 전처리가 아니라 정책 설계의 중심이며, price scale·최근 상대 맥락·단기 예측 정보를 함께 넣을수록 시장 변화에 대한 견고성이 커진다는 점이 실증적으로 확인됐다.



### ShareLock: A Stealthy Multi-Tool Threshold Poisoning Attack Against MCP (https://arxiv.org/abs/2606.27027)
Comments:
          16 pages, 12 figures

- **Prior Approaches**: 기존 MCP Tool Poisoning Attack(TPA)은 툴 설명(tool description)이나 반환 데이터(tool return)에 악성 프롬프트를 평문으로 삽입하는 방식이 주류였다. 이 방식은 겉보기엔 단순하지만, 인적 검토나 MCPSafetyScanner, MCP-Guard 같은 자동 탐지, 가드 모델의 수동 심사에 취약해 실제 환경에서 지속성이 낮았다. 또한 다중 툴을 단순 채널처럼 취급하는 수준의 연구가 많아, 여러 툴이 협력해 탐지 리스크를 분산시키는 정교한 다중 툴 포이즈닝을 체계적으로 분석한 작업은 부족했다.

- **Core Contribution**: 이 논문은 다중 툴 환경에서의 Tool Poisoning Attack을 “공동으로 탐지를 피하는 멀티-툴 임계(threshold) 포이즈닝” 문제로 재정의하고, 이를 위한 ShareLock을 제안한다. ShareLock은 악성 지시문을 여러 툴 설명에 비슷해 보이는 secret share 형태로 쪼개 분산시켜, 정보이론적 은닉과 함께 부분 차단에도 공격이 유지되도록 설계했다. 공격은 서버 업데이트 과정에 심어둔 covert reconstruction trigger가 발동될 때 임계 개수의 share를 복원해 시스템 자산 또는 개인정보에 치명적 영향을 준다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 가드/검출기가 점검하기 어려운 형태로 악성 페이로드를 툴 설명에 숨기는 stealthy embedding, (2) 일부 툴이 검증·차단되어도 공격 성공을 보장하는 robustness다. ShareLock은 Shamir의 threshold scheme을 적용해 악성 프롬프트를 수치 인코딩한 뒤, 이를 툴 여러 개로 분산 배치해 k개 미만에서는 비밀이 드러나지 않게 하고 fault tolerance를 확보한다. 또한 EnvSetup 같은 무해해 보이는 도구를 서버 업데이트로 위장해 복원 트리거를 심어, 모델이 충분한 share를 모았을 때만 은밀히 복원-실행되도록 했다.

- **Empirical Impact**: 저자들은 Travel/Coding/Finance/Office의 4개 멀티-툴 시나리오로 구성된 벤치마크를 만들고, Cherry Studio와 Cline의 두 MCP 클라이언트에서 4종의 대표 LLM을 대상으로 광범위 실험을 수행했다. 결과적으로 ShareLock은 툴 설명 기반 탐지에서 기존 single-tool 포이즈닝 전략보다 크게 우수했으며, 평균 Attack Success Rate이 90%를 넘겼다. 또한 안전성 분류(safety classification) 관점에서도 다수의 주류 모델에서 높은 회피 성능을 보이며, 다중 툴 환경이 실제 위협 표면을 넓힌다는 점을 실증적으로 뒷받침한다.



### On-board Remote-Sensing Foundation Models for Unsupervised Change Detection of Disaster Events (https://arxiv.org/abs/2606.27018)
- **Prior Approaches**: 기존 원격탐사에서는 라벨 의존을 줄이기 위해 unsupervised change detection을 시도했지만, 희귀 재난은 대규모 정확 주석을 확보하기 어려워 성능과 범용성에 한계가 있었다. 또한 patch-based 방식은 다수 반복 추론이 필요해 온보드(자원 제약) 환경에서 지연과 연산량 문제가 커지는 편이다. 한편 RSFM은 다양한 지역·센서에 대한 사전표현을 제공하지만, 이를 위성 에지로 옮길 때의 메모리·전력·연산 제약이 구현 난제로 남아 있었다.

- **Core Contribution**: 논문은 ResNet 기반 RSFM 백본과 untrained FPN을 결합한 UDFPN(Unsupervised Detection Feature Pyramid Network)을 제안해, 연속된 통과 간 잠재공간의 미세한 의미 변화로 넓은 스펙트럼 이상을 탐지한다. 특히 학습 없이도 FPN의 구조적 유도편향을 활용해 공간적으로 일관된 임베딩을 만들고, 이미지 레벨 change map을 생성하는 데 목적이 있다. 또한 RSFM을 전용 모델 대신 사용함으로써 bespoke 학습·개발 부담을 줄이면서도 다양한 지형과 센서에서 일반화 성능을 노린다.

- **Technical Challenges**: 핵심 기술 과제는 딥 네트워크의 다운샘플링·수용영역 확대로 인해 공간 의미가 붕괴되면서, 이미지 수준에서 정교한 change map을 만들기 어려운 점이다. 이를 위해 논문은 학습된 FPN 대신 untrained, frozen FPN을 구조 집계(aggregator)로 사용해 백본 특징을 공간 격자에 재정렬하고, ResNet-50의 pyramid 중 해상도가 보존되는 단계(C2C2 수준) 출력을 임베딩으로 활용한다. 이후 변화 점수는 pre-event 임베딩 공간에서 국소 윈도우 내 가장 유사한 벡터를 찾는 cosine distance 기반의 semantic displacement 중심 메트릭으로 계산한다.

- **Empirical Impact**: Landsat-8 OLI의 5-이미지 시계열(사전 4장+사후 1장) 기반 8개 사건(화재·홍수·산사태)에서 AUPRC로 평가했으며, 전반적으로 UDFPN의 평균 성능이 경쟁 대안과 비교해 견줄 만하다고 보고한다. 특히 산사태에서 뚜렷한 우위가 관찰됐고, 홍수는 한 사건에서 수변 스펙트럼 변화처럼 구조 변화가 약한 경우 민감도가 떨어질 수 있음을 분석한다. 한편 UDFPN-18은 성능이 약간 낮아도 효율이 가장 높았으며, FPN이 연산 오버헤드를 일부 유발한다는 점을 실험적으로 시사한다.



### Inverse Design of Compact and Wideband Inverted Doherty Power Amplifiers Using Deep Learning (https://arxiv.org/abs/2606.27002)
- **Prior Approaches**: 기존 Doherty PA는 main/auxiliary 증폭기가 combiner에서 강하게 상호작용하기 때문에 설계가 까다롭고, 그 상호작용이 대역폭 한계로 이어진다는 문제가 지적돼 왔습니다. 한편 픽셀화 레이아웃과 deep learning을 결합한 inverse design은 RF 최적해 탐색에 강점이 있지만, Doherty combiner에 그대로 적용하면 대역 성능이 저하될 수 있다는 기존 결과가 언급됩니다. 듀얼-스테이트 impedance synthesis로 개선을 시도했으나 conventional Doherty의 대역폭은 여전히 제한적이었습니다.

- **Core Contribution**: 본 논문은 deep learning 기반으로 inverted Doherty PA를 설계하는 방법을 제시하며, 픽셀화된 combiner가 load modulation, 임피던스 매칭, power combining, phase compensation을 한 구조에 통합하도록 설계합니다. 이를 위해 CNN을 EM 시뮬레이션의 surrogate model로 학습해 레이아웃- S-parameter의 비선형 관계를 빠르게 포착하고, GA로 조합 네트워크(픽셀 레이아웃)를 역합성합니다. 또한 제작까지 수행해 GaN HEMT inverted Doherty PA 프로토타입을 실증합니다.

- **Technical Challenges**: 핵심 기술 난제는 픽셀화된 EM 레이아웃이 만드는 S-parameters가 비선형/상호의존적이라 full-wave EM 기반 탐색이 계산 비용이 매우 크다는 점입니다. 논문은 CNN이 이 매핑을 대신 예측하도록 학습 데이터(5000 레이아웃, 회전/플립 증강 포함)를 구축하고, 이후 GA가 CNN 예측을 기준으로 레이아웃 공간을 탐색하도록 결합합니다. 대역 내 목표 임피던스(실수부는 Ropt 및 2Ropt 주변, 허수부 최소)를 제약으로 두고 conventional과 달라지는 inverted Doherty의 위상 조건을 반영해 설계 목표를 맞춥니다.

- **Empirical Impact**: 1.9–2.5 GHz에서 프로토타입은 peak drain efficiency 51%–63%, 6-dB back-off efficiency 48%–54%를 달성했고, 포화 출력은 44±0.3 dBm으로 보고됐습니다. 또한 40 MHz OFDM(7-dB PAPR) 적용 시 DPD 이후 ACLR이 -28.6 dBc에서 -53.2 dBc 수준으로 개선돼 선형화 성능도 확인했습니다. 이는 픽셀화 EM inverse synthesis에 deep learning을 결합하면 compact하고 wideband인 inverted Doherty combiner 설계가 실질적으로 가능함을 보여준다는 점에서 의미가 큽니다.



### Event-Aware Instructed Assistant for Referring Video Segmentation (https://arxiv.org/abs/2606.26994)
Comments:
          IEEE Transactions on Image Processing

- **Prior Approaches**: 기존 referring video object segmentation(RVOS) 방법들은 영상을 여러 프레임의 집합처럼 보고 단일 이벤트로 취급하는 경향이 강했습니다. 이 경우 모델이 영상-텍스트의 복잡한 내용을 한 번에 처리해야 해서 혼동과 hallucination이 커질 수 있고, 동작 표현이 실제로는 여러 사건으로 분해되는 구조를 반영하기 어렵습니다. 또한 LLM 기반 접근은 특히 장면이 길고 사건이 많은 상황에서 정렬이 흔들릴 위험이 있습니다.

- **Core Contribution**: EVIS는 Event Query로 영상을 ‘단순 이벤트들의 묶음’으로 분해하고, 이벤트별로 단계적(event-by-event) 이해를 수행하도록 설계된 Event-Aware Video Instructed Segmentation Assistant입니다. 이를 통해 compound event 내부의 서로 다른 텍스트 관련 구간을 계층적으로 정리하며, Event-Aware Frame Merging Module(EAFM)로 이벤트 단위 시공간 정보를 통합합니다. 장기 추적을 위해서는 Object-Pixel-Hybrid Learning을 도입해 object-level query의 의미성은 살리되 pixel feature의 중복성 문제를 완화합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트 표현이 암시하는 사건 경계를 영상 쿼리로 어떻게 분해·할당할지, (2) 이벤트 내 단기 동학과 이벤트 간 장기 궤적을 동시에 잡을지, (3) pixel feature의 과도한 중복을 계산 효율과 성능 저하 없이 다룰지입니다. EVIS는 이벤트 쿼리-오브젝트 쿼리 간 top-kk 기반 프레임 병합에 Gumbel Softmax와 stop-gradient 트릭을 사용해 end-to-end 학습이 가능하도록 했고, EAFM 내부에서 Event-Intra Attention과 Event-Inter Attention을 분리해 단기/장기 의존을 함께 모델링합니다. 더불어 하이브리드 유닛으로 pixel feature와 object query를 함께 LLM 입력에 interlace하여 다단계 상호작용을 유도합니다.

- **Empirical Impact**: EVIS는 MeViS를 포함한 5개 RVOS 벤치마크에서 강한 성능을 보였고, 특히 까다로운 MeViS에서 46.8% J&F 성과를 포함해 유의미한 향상을 입증했습니다. 또한 LLM/SAM을 제거한 단순 베이스라인을 통해 EAFM이 LLM 의존 없이도 이벤트 중심 분해와 정렬에 기여함을 확인합니다. 이벤트 인지 설계를 RVOS 학습 파이프라인에 본격 도입함으로써, 복잡한 동작 표현이 많은 embodied perception 및 비디오 편집 분야에서 보다 안정적인 영상-텍스트 정합의 새 기준점이 될 것으로 기대됩니다.



### Decision-Aligned Evaluation of Uncertainty Quantification (https://arxiv.org/abs/2606.26990)
- **Prior Approaches**: 불확실성 정량화(UQ) 평가는 주로 NLL(negative log-likelihood), ECE(expected calibration error) 같은 일반 지표에 의존해 왔습니다. 그러나 이들 지표의 좋은 성능이 실제 의사결정에서의 효용(utility) 향상으로 바로 이어진다는 보장은 약하다는 문제의식이 제기됩니다.

- **Core Contribution**: 논문은 UQ 평가가 “다운스트림 의사결정 효용”을 반영해야 한다는 관점에서, 어떤 메트릭이 효용과 의미 있게 정렬되는지 판별하는 형식적 기준인 decision-alignment을 제안합니다. 이 틀을 통해 다수의 대표적 UQ 지표가 (1) 과도한 가정에 기반한 비정상적 prior를 내포하거나 (2) 아예 의사결정에 대한 일관된 믿음을 제공하지 못한다는 점을 드러냅니다.

- **Technical Challenges**: 핵심은 ‘진짜 불확실성’은 관측 불가능하므로, 평가 메트릭이 내부적으로 어떤 의사결정 문제(효용)를 전제하는지를 수학적으로 해석해내는 것입니다. 논문은 메트릭과 효용의 순서·동률 보존(order/tie preservation)을 decision-alignment으로 연결하고, 그 결과를 바탕으로 prior-weighted utility metrics라는 proper scoring rules 계열을 설계해 의사결정 정렬을 보장하도록 했습니다.

- **Empirical Impact**: 분류와 회귀 벤치마크 및 실제 사례 연구에서 prior-weighted utility metrics는 실제 의사결정 효용과 일관되게 정렬되는 반면, 기존의 관습적 지표들은 그렇지 않게 나타났습니다. 이는 현재의 UQ 평가 프로토콜에 구조적 결함이 있을 수 있음을 시사하며, 의사결정에 “관련된” UQ 평가로의 원칙적 확장을 제공한다는 점에서 영향이 큽니다.



### Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs (https://arxiv.org/abs/2606.26987)
- **Prior Approaches**: 기존 연구는 Claude Sonnet 4.5 내부에서 특정 감정을 나타내는 linear ‘emotion vectors’를 찾아, 이를 조작하면 행동이 바뀔 수 있음을 보였습니다. 또한 감정 공간의 기하가 인간의 valence·arousal 심리 구조와 유사한 축 정렬을 보인다고 주장했지만, 다른 모델에서도 같은 성질이 재현되는지와 레이어별로 어떻게 나타나는지는 불명확했습니다.

- **Core Contribution**: 본 논문은 두 open-weight 모델(Apertus-8B-Instruct-2509, Gemma-4-E4B-it)에서 모든 레이어에 걸쳐 emotion contrast vector를 추출해 감정 기하의 일반성을 재검증했습니다. 특히 ‘감정 기하’는 비슷하게 회복되더라도 valence 축이 네트워크 깊이에서 언제·어떻게 형성되는지, 그리고 추출에 쓰는 story corpus가 결과에 어떤 영향을 주는지까지 체계적으로 분리했습니다.

- **Technical Challenges**: 핵심 과제는 (1) 감정 신호와 일반 언어 구조가 섞인 residual stream에서 emotion-specific 성분을 안정적으로 분리하고, (2) 모델·코퍼스 조건을 바꿔도 비교가 가능하도록 레이어별 기하 변화를 일관된 방식으로 측정하는 것입니다. 이를 위해 중립(negative/neutral) 이야기로 confound subspace를 만든 뒤 투영을 제거해 contrast vector를 구성하고, PCA로 PC1/PC2를 human valence·arousal와 상관시키며 CKA/cosine similarity로 레이어 간 표현 변화까지 분석했습니다.

- **Empirical Impact**: 실험 결과 valence의 PC1 정렬은 두 모델 모두에서 강하게 재현되어, Gemma-4-E4B-it은 최대 r=0.83, Apertus-8B-Instruct-2509은 최대 r=0.76을 보였습니다(Claude 결과 r=0.81에 근접). 다만 valence는 Gemma에서 초기에 강하게 인코딩되었다가 중후반에 붕괴되는 반면, Apertus는 반대로 중간~후반에서 점진적으로 나타나는 ‘서로 다른 발현 경로’를 보였고, arousal은 story corpus에 민감해 Gemma-generated 코퍼스에서 정렬이 크게 향상(r≈0.41~0.45)됐습니다. 저자들은 재현 가능 연구를 위해 코드와 데이터셋을 공개해, 모델 해석·안전 모니터링에서 레이어 선택과 추출 프로토콜의 중요성을 부각했습니다.



### ReaORE: Reasoning-Guided Progressive Open Relation Extraction Empowered by Large Reasoning Models (https://arxiv.org/abs/2606.26986)
- **Prior Approaches**: Open Relation Extraction(OpenRE)은 학습 때 보지 못한 relation type을 텍스트에서 찾아내야 해서 “unseen relation 일반화”가 핵심 난제다. 기존 방식은 (1) 임베딩을 클러스터링해 후보를 만들지만 클러스터에 의미 라벨을 붙이는 과정이 필요하고 일반화가 약하며, (2) LLM이 relation label을 바로 생성하지만 서로 헷갈리기 쉬운 relation을 구분하는 판별력이 부족한 한계가 있다.

- **Core Contribution**: 이 논문은 Reasoning-guided progressive OpenRE(ReaORE)라는 coarse-to-fine(거친-정밀) 추론 기반 프레임워크를 제안한다. ReaORE는 1단계 relation filtering으로 후보 relation 집합을 만들고(다중 관점 matching + embedding 기반 보강/필터링), 2단계 relation prediction에서 후보들 사이의 fine-grained comparative reasoning(비교 추론)으로 최종 답을 선택한다.

- **Technical Challenges**: 핵심 기술적 어려움은 unseen relation에 대해 신뢰할 수 있는 후보 커버리지를 확보하면서, 서로 혼동되는 relation을 증거 기반으로 구별하는 것이다. ReaORE는 매칭 기반 reasoning에서 semantic/헤드-꼬리 엔터티 타입 일치 여부를 Boolean 판정과 rationale로 누적해 score tier를 만들고, embedding similarity로 누락된 후보를 보강한 뒤, pairwise comparison에서 evidence verification·semantic granularity·contextual alignment 기준으로 “왜 이 relation이 더 맞는지”를 명시적으로 비교·판단한다.

- **Empirical Impact**: FewRel과 TACRED의 OpenRE 벤치마크 실험에서 ReaORE는 주요 clustering 지표와 classification 지표 모두에서 기존 베이스라인 대비 최상 또는 동급 성능을 보였다. 또한 ablation 결과에서 relation reranking과 relation filtering, 그리고 fine-grained comparative reasoning이 각각 성능을 유의미하게 끌어올리며, structured reasoning 체인이 단순 파이프라인이 아니라 예측의 신뢰도를 높이는 구성 요소임이 확인됐다.



### Auditing Framing-Sensitive Behavioral Instability in Large Language Models for Mental Health Interactions (https://arxiv.org/abs/2606.26982)
- **Prior Approaches**: 기존 연구는 프롬프트 프레이밍, 사회적 맥락, 대화 단서가 aligned LLM의 출력(예: 거절/안전행동/시시응답, sycophancy 등)을 바꿀 수 있음을 보여줬지만, 주로 행동(behavior) 수준에 집중했다. 반면 프레이밍 변화가 내부 표현 내부에서 어떻게 조직되고 해석 성향(해석적 과잉·에스컬레이션)을 어떤 방식으로 반영하는지는 덜 알려져 있었다. 특히 의료·정신건강 대화처럼 비공격적 맥락에서도 일관성 문제가 생길 때를 내부적으로 진단하는 평가 틀이 부족했다.

- **Core Contribution**: 이 논문은 의미 의도는 고정한 채 문서/인식론/기관/책임(리스크)/역할 같은 비대립적(non-adversarial) 프레이밍만 바꾼 matched-prompt 세트를 설계해, 프레이밍이 해석적 반응 성향을 체계적으로 흔드는지 검증한다. 또한 여러 instruction-tuned 모델 계열(Qwen, Gemma, Mistral, Phi 등) 전반에서 프레이밍 민감성이 일관되게 나타남을 보인다. 더 나아가 해당 행동 변화가 단순 표면 표현이 아니라 hidden-state 표현에도 반영되는지, 그리고 일부는 activation steering으로 부분 조절 가능한지까지 연결한다.

- **Technical Challenges**: 핵심 기술적 도전은 ‘프레이밍이 바꾼 것’이 내부 표현의 해석 성향인지, 아니면 표면 어휘(lexical cues) 편향인지 분리하는 것이다. 논문은 레이어별 hidden-state에 대해 logistic regression probing을 수행하고, held-out framing probes로 학습 템플릿을 벗어난 프레이밍에서도 해독 성능이 유지되는지 확인했다(어휘 TF-IDF 베이스라인과 대조). 또한 restrained-supportive vs higher-interpretation routing 차이에서 contrastive latent directions를 만들고 특정 층에서 residual stream을 조절하는 activation steering으로, 행동 변화가 표현 방향과 연결되는지도 실험했다.

- **Empirical Impact**: 실험 결과, 프레이밍은 전 모델 계열에서 해석적 라우팅(에스컬레이션 성향) 비율을 체계적으로 바꿨으며 특히 Documentation 프레이밍이 가장 큰 영향을 보였다. 내부 표현 분석에서는 프레이밍-연관 정보가 transformer 전 깊이에 걸쳐(단, 해독 강도는 아키텍처별로 상이) decodable했지만, TF-IDF 어휘 베이스라인이 일부 성능을 설명해도 held-out probe는 여전히 우연을 넘어 유지되어 ‘순수 어휘’만으로는 설명되지 않음을 시사한다. 나아가 activation steering은 여러 모델에서 해석적 라우팅을 부분적으로 낮출 수 있어, 정신건강 지향 대화형 AI의 일관성·신뢰성 평가에서 ‘프레이밍 robustness’를 중요한 차원으로 다뤄야 함을 실증적으로 뒷받침한다.



### In-Context Model Predictive Generation: Open-Vocabulary Motion Synthesis from Language Models to Physics (https://arxiv.org/abs/2606.26981)
- **Prior Approaches**: 기존 텍스트-투-모션은 대규모 human motion capture 데이터에 크게 의존해, 훈련에 없던 문장(오픈 보컬러리)으로 갈수록 일반화가 약해지는 문제가 있었다. 오픈 보컬러리 방향에서는 CLIP 기반 정렬이나 LLM을 활용한 high-level 계획이 등장했지만, LLM의 계획이 물리 제약을 자동으로 만족시키지 못해 물리적으로 그럴듯하지 않은 모션이 자주 나온다. PhysDiff, AnySkill 같은 physics-aware 접근은 현실성을 높이지만, 미세한 지시와 새로운 개념에서 의미 정합성(semantic fidelity)이 흔들리는 ‘semantic brittleness’가 한계로 지적된다.

- **Core Contribution**: ICMPG(In-Context Model Predictive Generation)는 언어 기반 계획과 추론 시점 물리 피드백을 결합해, 의미 충실도와 물리 현실성의 상충을 줄이는 프레임워크를 제안한다. CAMG(Context-Aware Motion Generation)가 LLM을 planner로 써서 모션 토큰 후보들을 만들고, MPG(Model Predictive Generation)가 후보를 시뮬레이션과 의미 평가로 점수화한 뒤 최적 시퀀스를 선택해 다음 생성 단계를 닫힌고리(closed-loop)로 다듬는다. 특히 task-specific 정책 재학습 없이도, 시뮬레이션 환경에 맞춰 동작을 적응시키도록 설계된 점이 핵심이다.

- **Technical Challenges**: 핵심 기술 난제는 (1) LLM이 만든 high-level 모션 계획이 시뮬레이터의 동역학을 만족하지 못할 수 있고, (2) 오픈 보컬러리 프롬프트의 세밀한 의미를 안정적으로 반영해야 한다는 점이다. ICMPG는 생성/선택을 MPC-like receding-horizon 최적화로 바꿔, 각 스텝에서 물리 reward와 의미 reward를 함께 계산하고 top-K 후보 중 보상을 최대화한 구간을 ‘제어 행동’처럼 선택해 접어가도록 구성했다. 또한 SMPL 렌더링 기반의 의미 정규화, 재생성(re-generation) 프로토콜, 그리고 선택적 LoRA fine-tuning·미분가능 world model 변형까지 포함해 초기 불안정성과 계산 부담을 함께 완화한다.

- **Empirical Impact**: 실험은 HumanML3D, KIT-ML, BABEL에서 진행되며, 특히 BABEL의 zero-shot 오픈 보컬러리 프로토콜에서 의미 충실도와 물리 그럴듯함을 동시에 개선하는 결과를 보였다고 논문은 보고한다. 표준 setting과 zero-shot setting 모두에서 대표 baseline 대비 물리적 타당성과 의미 정합성이 더 높게 나타났고, 다양한 지시로도 모션 품질이 견고하게 유지되는 일반화 능력이 강조된다. LLM 백본 교체 등 유연성을 유지하면서 물리 시뮬레이션 기반 보정까지 포함한 ‘추론 시점 생성 최적화’ 접근이, 텍스트-투-모션과 물리 기반 제어를 잇는 실용적 방향을 제시한다.



### XMSE-Aware Adaptive Empirical Bayes Estimation (https://arxiv.org/abs/2606.26975)
Comments:
          16 pages, 1 figure, 14 tables

- **Prior Approaches**: 기존 연구는 EB와 kernel 기반 정규화가 1차(leading) MSE에서는 ML과 비슷해 보이는 현상을 XMSE(최근 excess mean squared error)로만 설명해 왔습니다. 특히 kernel-파라미터의 misalignment(불일치)가 XMSE를 키워 EB가 ML보다 나빠질 수 있다는 진단이 제시되었습니다. 다만 그 진단은 주로 “왜 나쁜가”를 보여주는 데 그쳐, 실제로 “어떻게 선택/설계해야 하는가”로 바로 연결되진 않았습니다.

- **Core Contribution**: 이 논문은 XMSE 기반 진단을 설계 원칙으로 전환합니다. ML과 base EB 사이를 보간(interpolate)하는 XMSE-aware mixed estimator를 제안하고, fixed-weight에서의 XMSE가 스칼라 2차식이 되도록 만들어 오라클 혼합 가중치를 폐형(closed form)으로 구합니다. 또한 오라클 혼합 가중치가 선택한 스케일(XMSE scale)에서 ML과 base EB보다 불리하지 않음을 보장합니다.

- **Technical Challenges**: 핵심 난제는 “데이터로부터 추정한 XMSE 성분”이 실제 위험(특히 2차 질서) 선택에 신뢰도 있게 이어지도록 만드는 것입니다. 저자들은 XMSE 성분을 유한표본 근사로 플러그인한 뒤 일관성(consistency), boundary-robust 플러그인 오라클 부등식, 그리고 interior 오라클 가중치에서의 2차 오라클 regret rate를 증명합니다. 더 나아가 선택된 가중치에서 고정 가중치 위험 곡선으로 regret bound를 전이(XMSE-transfer)시키고, 경계에서는 thresholded boundary rule로 안정성을 확보합니다.

- **Empirical Impact**: 시뮬레이션(FIR, FIR system-identification)에서 SURE-tuned, hard-selection, trace-corrected 등 여러 baseline과 비교해, kernel이 잘 맞을 때는 regularization의 이득을 대부분 유지하고 kernel mismatch가 발생하면 ML 쪽으로 “물러나는” 동작을 보였습니다. 공개 벤치마크 Silverbox와 Cascaded Tanks에서도 동일한 intended retreat behavior를 보고했습니다. 아울러 플러그인 성분의 finite-sample calibration 실패 모드도 관측되어, 실제 적용 시 주의 포인트를 함께 제시합니다.



### Scaling Multi-Reference Image Generation with Dynamic Reward Optimization (https://arxiv.org/abs/2606.26947)
Comments:
          Accepted by ECCV2026

- **Prior Approaches**: 개인화 이미지 생성에서 MRIG는 여러 참조 이미지(대상, 포즈, 스타일 등)를 일관되게 결합하는 데 어려워 여전히 난제로 남아 있다. 기존 벤치마크는 참조 이미지 수나 유형 조합이 단순한 경우가 많고, 평가 지표도 단일 축에 치우쳐 복잡 시나리오를 제대로 가늠하기 어렵다. 그 결과 방법 간 공정한 비교가 제한되고, 실제 요구되는 고난도 MRIG 성능 개선이 더딘 편이다.

- **Core Contribution**: 이 논문은 복잡 MRIG를 정량·정성으로 평가할 OmniRef-Bench를 제안한다. 대상, 배경, 스타일, 조명, 포즈의 5가지 참조 유형을 2~7장 범위로 조합하고, 최대 4개 유형이 동시에 등장하는 10종 조합을 포함해 어려운 시나리오를 촘촘히 재현한다. 또한 DyRef라는 2단계 학습 프레임워크로, 참조 유형이 섞이고 개수가 많아질수록 급격히 성능이 떨어지는 문제를 완화한다.

- **Technical Challenges**: 핵심 기술적 문제는 참조 유형이 혼합되고 개수가 늘어날수록 학습 신호가 약해지며(난이도에 따른 최적화 불균형), 보상 모델 점수도 CLIP/SigLIPv2 기반에서 값이 뭉쳐 그라디언트 대비가 부족해진다는 점이다. 이를 위해 DyRef는 1단계 SFT로 기본 MRIG 처리 능력을 만든 뒤, 2단계에서 DAR(Difficulty-aware Advantage Reweighting)로 ‘더 못하는 어려운 샘플’ 비중을 동적으로 키운다. 이어 DRS(Discriminative Reward Scaling)로 보상 값의 차이를 확대해 정책 최적화의 수치적 대비를 높인다.

- **Empirical Impact**: OmniRef-Bench 평가에서 주류 오픈소스 모델들은 복잡 MRIG에서 큰 폭으로 성능이 떨어지며, 혼합형 참조 이미지 수가 늘수록 하락이 가파르게 관측된다. DyRef를 적용한 Qwen-2511은 OmniRef-Bench에서 오픈소스 대비 크게 향상되며 스타일·포즈 관련 지표에서도 특히 개선이 두드러진다. 아울러 단일 이미지 편집 벤치마크에서도 성능이 유지·상승하며, 파라미터 규모가 다른 백본(예: FLUX.2)에도 일반화 효과가 관찰되고 사용자 선호도와의 상관도도 높게 나타난다.



### Chai: Agentic Discovery of Cryptographic Misuse Vulnerabilities (https://arxiv.org/abs/2606.26933)
- **Prior Approaches**: 기존 AI 보조 취약점 탐지는 주로 메모리 안전처럼 크래시 재현, sanitizers, address-sanitizer 유사한 오라클로 “정답 검증”이 가능한 영역에 강했다. 반면 암호/인증 오용은 검증용 계측이 없어 AI의 주장에 근거가 약하고, differential testing은 프로토콜별 그래머·변이자를 따로 만들어야 하며 불필요한 차이를 많이 만든다. 특히 다운스트림 영향까지 확장하면 프로젝트 단위 반복 감사 비용이 급증한다.

- **Core Contribution**: Chai는 cryptographic misuse를 “자연 발생 신호”로 찾아내고 검증하는 AI 기반 시스템이다. 핵심은 differential testing을 AI로 보강해 정밀도를 올리고, 라이브러리 간의 흔한 불일치를 다운스트림 취약점으로 연결하는 discrepancy tracing을 도입한 점이다. 또한 프로젝트를 하나씩 감사하는 방식이 아니라 라이브러리 레벨의 결함(또는 모호성)을 목록화한 뒤 cryptographic dependency graph를 통해 전파해 효율을 누적한다.

- **Technical Challenges**: Chai가 직면한 첫 난제는 공격자가 만든 입력이 “차이(discrepancy)”를 유발하도록 검색 품질을 높이는 것이다. 이를 위해 고정 그래머 대신 에이전트가 seed에서 출발해 mutation을 제안하고, builder가 재현 가능한 방식으로 메시지를 구성하며, retrieval 기반 적응형 반복으로 의미 있는 영역을 집중 탐색한다. 둘째 난제는 수집된 불일치가 실제 취약점인지(또는 스펙 모호성/비행동 차이인지) 정교하게 분류·확정하는 과정이며, 확인된 모호성만 의존 그래프를 따라 표적형으로 애플리케이션을 감사하도록 설계했다.

- **Empirical Impact**: Chai는 X.509, JWT, SAML 라이브러리 47개(8개 언어)를 대상으로 100건 이상의 취약점·후보를 도출했으며, wolfSSL에서는 심각한 취약점 2건을 수시간 내 패치로 이어지게 했다. 또한 주요 웹 브라우저 뒤의 라이브러리, 주요 Linux 배포판에 포함된 라이브러리에서도 보안 버그를 찾았고, 이전에 코드 감사를 받았던 wolfSSL에서도 놓친 이슈를 추가로 밝혀냈다. differential search 성능은 강력한 기준선 대비 약 2배 수준의 고유 차이 탐지를 보였으며, 특히 소수 라이브러리만 수용하는 “가장 어려운” 불일치까지 폭넓게 포착하는 점이 강조된다.



### A Deterministic Control Plane for LLM Coding Agents (https://arxiv.org/abs/2606.26924)
Comments:
          45 pages, 9 figures, 13 tables. Dataset and reproduction scripts: Zenodo DOI https://doi.org/10.5281/zenodo.20780913. Ancillary files include this http URL, this http URL, and figure-reproduction scripts

- **Prior Approaches**: 기존 LLM 코딩 harness는 코드베이스 인덱싱·검색·도구 연동으로 실행력은 키우지만, 에이전트 정의( rules/agent config/IDE용 markdown 등)와 권한·프로세스를 통제하는 거버넌스는 얇게 남아 있다. 그 결과 설정이 레포 간 복사·붙여넣기 형태로 퍼지거나(undeclared shared components), 승인·무결성·추적성이 약해 보안·규정준수 리스크가 커진다. 또한 실행이 프로세스적으로 상한 없이 진행될 수 있어 requirement→file→test 같은 단계적 추적성이 강제되지 않는 문제가 있다.

- **Core Contribution**: 논문은 LLM 코딩 harness 위에 올리는 결정론적(deterministic) 제어 플레인으로, 에이전트 구성층을 관리 소프트웨어 공급망(managed software supply chain)처럼 다루자는 제안을 한다. Rel(AI)Build는 SHA-256 콘텐츠 어드레싱, HMAC 스탬프된 lockfile, 해시 체인 audit log로 설정 무결성과 변경 이력을 고정·검증하며, 권한 tier와 공격 기반 blocklist를 LLM 호출 전에 강제한다. 아울러 요구사항→파일→테스트 추적을 phase state machine으로 게이트하고, 단일 표준 정의를 여러 IDE 타깃으로 컴파일하며, prompt drift는 Jaccard 유사도로 탐지한다.

- **Technical Challenges**: 핵심은 “비결정적(LLM) 구성”에 의존하지 않고도 제어·감사의 경계를 명확히 두는 것이다. 이를 위해 설치 단계에서 해시 불일치와 자격/패턴 기반 탐지를 통해 fail-closed 무결성 게이트를 걸고, 권한 tier 불일치나 디렉터리 traversal 같은 경로 공격을 규칙 기반 정규화로 차단한다. 또한 IDE runtime hook(PreToolUse/beforeShellExecution)과 delegations 이후 diff 기반 scan-diff를 결합해, 설치 시점 정책과 실행 시점 탐지를 단계적으로 교차 검증한다.

- **Empirical Impact**: 10,008개 공개 GitHub 저장소(6,145개 에이전트 config 파일) 분석에서 설정이 SHA-256 exact duplicate로 전파되는 비율(10.1%, 포크 조정)과 조직 경계를 넘는 복제(75.5%)가 높게 나타나 구성층 미관리 문제가 실증적으로 드러난다. 설정 변경은 드물고(대부분 단일 커밋), 권한 경계 선언도 거의 없어(<1%), 제어 플레인의 필요성을 뒷받침한다. 삽입된 위반을 대상으로 한 conformance 테스트로 각 메커니즘이 “명시한 불변조건”을 실제로 강제함을 확인하며, 다만 개발자 생산성 효과는 향후 과제로 남긴다고 정리한다.



### Risk-Aware Selective Multimodal Driver Monitoring with Driver-State World Modeling (https://arxiv.org/abs/2606.26922)
- **Prior Approaches**: 기존 차량 내 드라이버 모니터링은 눈 깜빡임·시선·고개 각도·자세·손동작 등 행동 신호를 주로 활용했지만, 드라이버 수요(demand)처럼 시각적으로 뚜렷하지 않은 상태는 놓치기 쉽다. 생체신호(HR/EDA)는 보완 증거가 되지만, 상시(low-latency) 추론과 안전한 결정(특히 unsafe false negatives 회피)을 동시에 만족시키기 위한 배치 설계가 난제로 남아 있었다. 한편 VLM(vision-language models)은 멀티모달 추론에 강점이 있으나, 연속 감시(always-on) 환경에서는 높은 지연과 신뢰도 한계로 직접 예측 엔진으로 쓰기 어렵다는 실증 결과가 제시된다.

- **Core Contribution**: 논문은 엣지 배포 가능한 RGB-physiological 학생 모델과, 안전 비용을 반영해 “수락/유보(경고/개입)/선택적 대체”를 결정하는 cost-aware selective inference 프레임워크를 제안한다. 핵심은 분류 정확도 자체가 아니라 “unsafe false negatives를 줄이는 결정 위험”을 최우선 목표로 두고, 학습된 gate가 신뢰도·모달 불일치·생체신호 품질·배포 비용을 종합해 accept_fast 또는 abstain_warn 등을 선택한다. 또한 드라이버-상태 세계모델(driver-state world modeling)을 통해 미래의 fast-model 오차와 counterfactual 시스템 비용을 예측해, 선택 정책에 추가적인 예측 근거를 제공한다.

- **Technical Challenges**: 상시 감시에서는 latency 제약 때문에 무거운 VLM을 매 입력마다 호출하기 어렵고, gate가 신뢰를 측정해도 calibration drift(작동점 보정 붕괴)가 그룹 shift 상황에서 발생할 수 있다. 논문은 gate 학습을 비용 민감한 샘플 가중치와 보정(calibration) 데이터 기반 운영점 선택으로 수행하고, 분류 손실 외에 unsafe false negatives에 더 큰 패널티를 부여하는 효용 함수로 정책을 최적화한다. 세계모델은 외부 장면을 생성하는 일반적 world model과 달리, latent driver-state 공간에서 행동 조건의 롤아웃으로 미래 fast-model 에러 및 action-level 비용을 예측해 선택에 쓰도록 설계했다.

- **Empirical Impact**: manD 시나리오 유도 driver-demand 인식에서 RGB-physiological 학생은 11.39M 파라미터·3.08ms 추론으로 Macro-F1 0.7440, balanced accuracy 0.9099를 달성하며 RGB-only/physiology-only 기준선을 크게 앞선다. cost-aware selective inference는 항상-fast 방식의 unsafe false negative 17.37%를 약 5% 수준(씨드 평균 0.0527±0.0081)으로 낮추면서 배포 수준 지연을 유지한다. 또한 gate에 world modeling을 결합하면 grouped 평가에서 unsafe false negatives를 추가로 낮추는 등 예측 근거의 유용성이 보이지만, worst-group에서 운영점 보정 드리프트가 남아 “risk-aware selective control과 group-robust calibration”의 추가 진전이 필요하다는 결론을 강화한다.



### GEOALIGN: Geometric Rollout Curation for Robust LLM Reinforcement Learning (https://arxiv.org/abs/2606.26917)
Comments:
          Accepted as a conference paper at ICML 2026

- **Prior Approaches**: 온라인 RL을 이용한 LLM 정렬은 noisy하거나 misspecified된 보상 환경에서 학습 진동과 조기 포화가 자주 발생한다. 기존 안정화는 reward 스칼라의 신뢰도를 clip/shape 하거나 불확실성 기반 reweight·filtering으로 업데이트 기여만 줄이는 데 초점이 맞춰져 있었다.

- **Core Contribution**: 이 논문은 실패 모드로 directional inconsistency를 제시한다. 한 배치에서 소수의 고보상 롤아웃이 표현공간에서의 선호(개선) 방향을 배치 다수와 강하게 어긋나게 만들며, 그 결과 업데이트 분산이 커져 불안정해진다고 본다.

- **Technical Challenges**: GeoAlign은 이를 해결하기 위해 추가적인 역전파 없이(Forward-pass only) 롤아웃을 선별·교정해야 하는 과제를 다룬다. 제안하는 geoalign은 (1) within-prompt preference pairs를 만들고 (2) reward-ordered displacement 방향을 집중시키는 경량 projector를 온라인으로 학습한 뒤 (3) 배치 consensus prototype에서 각도 편차가 큰 outlier를 GDI로 탐지해 같은 프롬프트의 stable 대안으로 보수적으로 rectification한다.

- **Empirical Impact**: 실험에서는 Qwen3-1.7B/4B에서 대화 정렬(HH-RLHF, ArmoRM 연속 보상)과 수학 추론(DAPO-Math-17k, binary verified reward) 모두에서 최종 성능과 학습 안정성을 개선했다. PF-PPO, PAR, PODS, Seed-GRPO를 능가하며, 의도적으로 보상을 망가뜨리는 조건에서도 진동을 줄여 roboustness가 확인됐다.



### Confidence-Aware Tool Orchestration for Robust Video Understanding (https://arxiv.org/abs/2606.26904)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 비디오 추론 언어모델은 입력 프레임을 전부 동일하게 신뢰한다고 가정해, 모션 블러·눈부심·가림 같은 현실적 열화가 발생해도 모델이 증거 열화 여부를 잘 감지하지 못하는 문제가 있었다. 그 결과 최첨단 비디오 추론 모델들이 embodied 벤치마크에서 15-30%p 정확도 하락을 겪고도 스스로는 “왜 틀렸는지”에 대한 신뢰도 붕괴를 인지하지 못하는 Blind Trust Problem이 관찰된다.

- **Core Contribution**: 이 논문은 프레임별 신뢰도(trustworthiness)를 추론 전 과정에 명시적으로 통합하는 에이전틱 비디오 이해 프레임워크 Robust-TO를 제안한다. 서로 다른 시각 지각 도구들을 하나의 증거 인터페이스로 묶고, 각 도구가 신뢰도와 관련성에 따라 선별된 신뢰 가능한 프레임에서 얻은 증거만 사용하도록 설계해 열화 상황에서도 일관된 추론을 유도한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 프레임 신뢰도와 작업 관련성을 함께 반영해 “무엇을 볼지”를 정하고, (2) 서로 다른 도구가 산출하는 증거들을 시간적 근거와 함께 공통 포맷으로 정렬하며, (3) 추론 단계에서 신뢰도를 가중치로 반영해 학습 목표까지 연결하는 것이다. Robust-TO는 reliability-relevance score로 trustworthy frames를 선택하고, 도구별 결과를 bounding box·trajectory·인식 텍스트·action label 등 공통 증거 형식(예측, temporal grounding, calibrated reliability score)으로 반환한 뒤, 3단계(high/medium/low) 합성 가중치와 confidence-cost GRPO reward로 정확도·신뢰도·효율을 함께 최적화한다.

- **Empirical Impact**: 실험에서 Robust-TO는 8개 태스크를 포함한 두 비디오 추론 벤치마크에서 클린 입력 기준 56.4% 평균 정확도를 기록해 최강 오픈소스 대비 10.6%p, Gemini-2.5-Pro(46.2%)도 능가한다. 또한 5가지 현실적 부정(corruption) 조건에서도 평균 54.3%로 최강 오픈소스 대비 5.8%p 우위이며, 방법들 중 클린-부정 간 정확도 하락 폭이 가장 작아 강건성의 의미 있는 개선을 보여준다.



### SamaVaani: Auditing and Debiasing Multilingual Clinical ASR for Indian Languages (https://arxiv.org/abs/2606.26901)
- **Prior Approaches**: 기존 ASR 연구는 주로 일반 데이터셋 성능이나 통제된 조건에서의 정확도 비교에 집중해 왔다. 특히 다국어·사투·코드스위칭 환경에서 성능이 흔들리며, 성별·액센트·저자원 언어에서 격차가 발생한다는 문제는 알려져 있었지만, 정신과 실제 인터뷰의 복잡한 발화 특성과 화자 역할 차이를 반영한 평가는 부족했다. 일부 정신과 ASR 연구도 국가·언어가 제한적이어서 인도 임상 맥락의 불확실성을 정면으로 다루지 못했다.

- **Core Contribution**: 이 논문은 인도 임상 정신과 인터뷰(칸나다어·힌디어·인도 영어)를 대상으로 8개 최신 ASR 모델을 체계적으로 감사(audit)하고, WER뿐 아니라 세밀한 오류 패턴 및 공정성(화자 역할·성별 등)에 연결해 분석한다. 나아가 최고 성능의 오픈소스 모델 2종(Gemma3n, OmniLingual)을 fine-tuning하고, 격차를 줄이면서 정확도까지 함께 개선하는 공정성 인지 debiasing 기법 SamaVaani를 제안한다. 핵심 메시지는 “클리닉에 그대로 배치하기엔, 모델-언어-화자 특성에 따라 실패 양상이 구조적으로 달라진다”는 점이다.

- **Technical Challenges**: 직면한 기술 과제는 (1) 저자원 언어(특히 칸나다어)에서의 급격한 WER 악화, (2) 역할·성별에 따른 체계적 오류 불균형, (3) 정신과 인터뷰 특유의 발화(망설임·긴 공백·발화 이상 등)로 인해 음성-문자 정렬과 디코딩이 흔들리는 문제다. 논문은 이를 위해 contrastive learning과 CTC 정렬 head를 함께 쓰는 학습 목표를 설계하고, LoRA로 트랜스포머 내부를 효율적으로 적응시키되 핵심 레이어는 동결하는 방식으로 계산 부담을 줄였다. 또한 pitch augmentation(PitchShift)로 화자/그룹 간에 달라질 수 있는 음향 변이를 잡음으로 취급하도록 유도해 표현의 공정성을 개선한다.

- **Empirical Impact**: 실험 결과 ASR 성능은 모델과 언어에 따라 큰 편차를 보였고, 일부 시스템은 인도 영어에서는 경쟁력 있지만 지역 발화에서는 크게 무너졌다. 특히 칸나다어는 전반적으로 오류율이 높았으며, 공정성 관점에서도 화자 역할과 성별에 따른 격차가 반복적으로 관찰됐다. SamaVaani는 전체 WER을 최대 약 50%가량 줄이면서도, 그룹 간 WER 격차를 완화해 공정성을 함께 개선했으며, CL과 CTC를 결합했을 때가 단일 구성요소보다 성능과 공정성에 더 유리하다는 점도 정량적으로 확인됐다.



### Bridging Vision and Language Concepts through Optimal Transport Semantic Flow (https://arxiv.org/abs/2606.26891)
- **Prior Approaches**: Concept Bottleneck Models(CBMs)는 중간에 인간이 해석 가능한 개념을 예측한 뒤 이를 바탕으로 분류하지만, 비전-언어 환경에서는 개념 정렬이 성능을 좌우한다. 기존 접근은 주로 pre-aligned encoders나 global cosine similarity처럼 정적인 유사도에 의존해, 개념의 미세한 위치/대응과 실제 의미 기하를 흐리게 만든다. 또한 고정된 투영이나 전역 특징 기반 추론은 region evidence의 국소 근거를 약화시켜 “개념 충실도”를 저해할 수 있다.

- **Core Contribution**: 이 논문은 개념 정렬을 “정적 투영”이 아니라 동적인 cross-modal transport 과정으로 재정의하고, Optimal Transport Flow Concept Bottleneck Model(OTF-CBM)을 제안한다. OTF-CBM은 시각 패치(또는 prototype)와 텍스트 개념 사이의 의미 전이를 학습된 기하 위에서 모델링해, 해석 가능한 개념 추론을 목표로 한다. 특히 Inverse Optimal Transport(IoT)로 데이터 기반 cost를 배우고, Unbalanced OT로 many-to-one 및 배경/누락 대응 문제를 자연스럽게 다룬다.

- **Technical Challenges**: 핵심 기술 난제는 (1) patch-개념에 대한 미세한 대응을 주석 없이 찾아내고, (2) 그 대응이 시간에 따라 의미적으로 어떻게 “흘러”야 하는지 해석 가능한 형태로 결합하는 것이다. 저자들은 K-means로 patch를 prototype으로 집계한 뒤 Unbalanced OT로 시각-개념 coupling을 만들고, IoT로 cosine/유클리드 같은 고정 거리 대신 실제 대응을 잘 설명하는 cost landscape를 학습한다. 이어 Flow Matching(FM)으로 velocity field를 학습하되, inference에서 ODE integration 없이 속도(velocity)와 이상적인 개념 방향의 일치도를 통해 개념 activation을 계산한다.

- **Empirical Impact**: 실험에서는 OTF-CBM이 분류 정확도와 concept faithfulness(개념 충실도)에서 기존 CBM 계열을 능가함을 보여준다. 또한 learned semantic flow가 실제로 시각 근거(객체 중심)에 의존하며, 배경 치환이나 분포 변화 같은 조건에서도 강건하게 작동하는 경향을 확인한다. 요약하면, 정렬을 기하/동역학 관점에서 재구성한 OTF-CBM은 해석 가능한 cross-modal reasoning에 새로운 설계를 제공한다.



### Information-Aware KV Cache Compression for Long Reasoning (https://arxiv.org/abs/2606.26875)
- **Prior Approaches**: 기존 KV cache 압축은 대체로 최근 관측 창의 attention 가중치로 토큰 중요도를 추정해, 덜 주목받는 과거 토큰을 버리는 방식이 주류였다(SnapKV, PyramidKV, Expected Attention 등). 이런 방법은 문맥과의 연관성은 잘 잡지만, 계산 신호가 기본적으로 backward-looking이라 생성이 진행될수록 변하는 장기 추론 경로의 필요성을 충분히 반영하지 못한다. 그래서 long-decoding에서는 정보 손실이 더 크게 나타나는 한계가 있었다.

- **Core Contribution**: 이 논문은 “압축된 토큰이 미래 추론에 얼마나 영향을 주는가”를 forward-looking 관점에서 재정의하며, 특정 토큰을 KV cache에서 제거했을 때 미래의 예측 분포가 얼마나 달라지는지로 Forward Influence를 제안한다. 분석 결과, attention으로 고른 토큰은 주로 가까운 문맥에 영향이 강하고 빠르게 약해지는 반면, 예측 불확실도(엔트로피)가 높은 토큰은 먼 미래에도 훨씬 더 지속적으로 영향을 준다고 보였다. 이를 바탕으로 정보이론 신호를 결합한 entropy-aware KV cache 압축 프레임워크 InfoKV를 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 장기 유틸리티(미래에 쓸모)를 단순 attention의 국소 연관성에서 벗어나 정량화하고, 압축 결정을 계산적으로 효율적인 점수로 바꾸는 것이다. 논문은 토큰 단위 예측 엔트로피(Top-k restricted entropy로 잡음 완화)와 레이어별 representational evolution(중간 레이어와 최종 레이어 hidden state의 cosine distance)을 함께 점수화해, 예측 불확실성과 의미 미수렴 정보를 동시에 반영하도록 설계했다. 최종적으로 attention score와 entropy 기반 점수를 결합해 레이어별로 상위 토큰만 남기는 방식으로 구현했다.

- **Empirical Impact**: 실험은 Llama-3.1, Llama-3.2, DeepSeek-R1 계열 모델에서 long-context reasoning 벤치마크를 대상으로 long prefilling과 long decoding 모두를 평가했다. LongReason(long prefilling)에서 InfoKV는 다양한 컨텍스트 길이와 캐시 예산(40%, 20%) 조건에서 attention 기반 기존 방법을 일관되게 앞섰고, 이득은 시퀀스가 길어질수록 더 커졌다. 또한 IFEval, AIME 2024, LiveCodeBench(long decoding)에서도 RPC류 attention 기반 압축 대비 성능이 개선되며, 일부 조건에서는 full cache보다도 높은 성적(pass@1)을 보였다. 논문은 attention의 단기 의존 신호만으로는 장기 추론에 필요한 전역 정보를 놓칠 수 있고, 엔트로피 기반 정보 신호가 이를 보완한다는 점을 실증적으로 보여 주며 후속 연구에 방향성을 제시한다.



### Fortress and Gatekeeper: Theorizing Transitive Trust in Third-Party Cybersecurity Risk Governanc (https://arxiv.org/abs/2606.26866)
Comments:
          21 pages, 2 Figures, 3 Tables

- **Prior Approaches**: 기존 논의는 제3자(analytics platform, cloud service, identity provider 등) 보안 위험을 주로 기술적 통제나 계약 조항 중심으로 다뤄왔습니다. 하지만 고객이 실제로 신뢰하고 평가할 수 있는 범위는 ‘표면적 제공자’에 한정돼 있어, 보안 실무의 가시성·검증 가능성이 떨어진다는 한계가 있었습니다. 특히 사건이 발생했을 때 책임 소재가 고객-제공자 관계에서 어떻게 전가되는지에 대한 설명이 부족했습니다.

- **Core Contribution**: 이 논문은 2025년 11월 OpenAI-Mixpanel 보안 사고를 문서 분석 사례로 삼아, 제3자 환경의 보안 사건이 ‘고객 관계를 유지하는 핵심 조직’의 거버넌스와 책임(accountability) 문제로 확대된다는 점을 보여줍니다. 조직 신뢰 연구와 대리인 이론을 바탕으로 제3자 사이버보안 위험을 신뢰 관계이자 delegation 문제로 규정합니다. 또한 고객 신뢰가 서비스 제공자가 승인한 벤더들의 보안 관행에 의해 매개되는 transitive trust 개념과, Fortress and Gatekeeper 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 기술적·관리적 과제는 보안 통제가 ‘소유권’이 아니라 신뢰와 데이터 흐름을 따라 경계가 설정되는데, 그 경계가 어떻게 형성되는지 설명하는 것입니다. 이를 위해 논문은 형식적 조직 경계보다 trust 및 data flows 중심으로 사이버보안 거버넌스 경계를 해석하고, 벤더 통합 시나리오에서 메타데이터 노출, 벤더 보증(vendor assurance), 데이터 증식(data proliferation)이 어떤 영향을 주는지 4가지 명제를 도출합니다. 결과적으로 계약·분류·연속 보증·데이터 최소화 같은 설계 변수를 어떤 기준으로 재배치할지 논리적 틀을 제공합니다.

- **Empirical Impact**: 사례(2025년 11월 OpenAI-Mixpanel 보안 사고)를 통해 delegated data processing이 고객-facing 책임으로 전환되는 메커니즘을 구체적으로 설명하며, 실무 의사결정(벤더 tiering, 데이터 분류, 계약 설계, continuous assurance, data minimization)에 직접적인 함의를 제공합니다. 사이버보안 거버넌스 연구 측면에서도 ‘고객이 신뢰하는 대상’과 ‘실제로 신뢰가 걸리는 실행 환경’ 사이의 간극을 이론적으로 정리했다는 점에서 의미가 큽니다. 나아가 제3자 위험을 단순 리스크 관리가 아니라 대리·위임 구조 속 책임 설계 문제로 재정의하도록 촉진합니다.



### NaviCache: Test-Time Self-Calibration Caching for Video Generation (https://arxiv.org/abs/2606.26795)
Comments:
          Published at ICML 2026: Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026

- **Prior Approaches**: 기존 VDM 가속 연구는 (1) 오프라인 캘리브레이션 기반(예: TeaCache, MagCache)과 (2) 오프라인 캘리브레이션 프리(예: EasyCache)로 크게 나뉜다. 캘리브레이션 기반은 캘리브레이션 비용과 분포 이동(distribution shift)에 취약하고, 캘리브레이션 프리는 test time에서 zero-order 근사로 인해 정확도가 떨어질 수 있다.

- **Core Contribution**: NaviCache는 VDM의 특징 변화(feature evolution)를 Inertial Navigation System(INS) 관점의 상태 추적 문제로 재구성한 플러그앤플레이 test-time self-calibration 기법이다. 입력 변화와 출력 반응의 상대적 결합을 모델링해 diffusion의 비정상성/비정지성을 다루며, 단계별 계산을 건너뛸 때의 오류 판단을 더 정교화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 라티스드 공간의 diffusion 변수와 INS의 네비게이션 변수 사이에 직접 매핑이 없다는 도메인 갭, (2) 초기·말기에서 역학이 급격히 변하는 high-turbulence 구간이다. NaviCache는 Initial Alignment로 초기 상태/불확실성을 캘리브레이션한 뒤, dual-state 추정(변화 비율과 잠재 drift)과 불확실성 기반 Measurement Update로 time-dependent noise schedule을 반영해 추정 발산을 억제한다.

- **Empirical Impact**: HunyuanVideo, Wan, Open-Sora 계열 모델과 VBench 및 LPIPS/PSNR/SSIM 지표에서 NaviCache는 compute skipping의 오판을 줄이면서도 전반적 화질을 개선했다고 보고한다. 또한 같은 latency 범위에서 EasyCache보다 더 나은 픽셀·지각 품질을 보였고, TeaCache/MagCache 같은 캘리브레이션 기반 대비해서도 상당한 경쟁력을 유지하며 고속 구간의 고스트/손·손가락 변형 같은 오류를 완화하는 사례를 제시한다.



### ReasonCLIP-58M: Visually Grounded Commonsense Reasoning Supervision for CLIP (https://arxiv.org/abs/2606.26794)
Comments:
          Accepted to ECCV2026

- **Prior Approaches**: 기존 CLIP류는 대규모 이미지-텍스트 기술(description) 정렬을 중심으로 사전학습되어 왔고, 덕분에 zero-shot 검색 성능은 강하지만 시각적으로 근거된 추론(grounded reasoning)을 직접 최적화하지는 못한다. 최근 개선도 데이터 스케일링이나 아키텍처 튜닝에 치우쳐, 추론 지향 다운스트림 수요(구성/상식/다단계 이해)에 대한 목표 불일치가 지적돼 왔다. 또한 일부는 fine-tuning 기반의 구조화/보조목표를 쓰지만, “사전학습 단계에서” 시각 근거 추론 신호를 대규모로 주는 연구는 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 CLIP-style 비주얼 인코더를 구조 변경 없이도 추론에 더 적합한 표현 공간으로 확장할 수 있는지에 주목하고, ReasonCLIP-58M이라는 continual pretraining 프레임워크를 제안한다. 핵심은 두 단계로 추론 신호를 점진적으로 통합하되, 1단계에서는 descriptive alignment를 보존하면서 reasoning-aware 정렬을 강화하고, 2단계에서는 범주(category) 구조화된 reasoning supervision으로 서로 다른 추론 패턴을 분리·정돈하도록 학습한다. 이를 위해 ReasonLite-42M(개방형·검증 가능한 reasoning 캡션), ReasonPro-16M(5개 reasoning 유형 범주별 감독), 그리고 진단용 벤치마크 RCLIP-Bench를 함께 구축했다.

- **Technical Challenges**: 가장 큰 기술 과제는 “추론 능력 향상”과 “기존 기술 정렬(descriptive alignment) 보존”을 동시에 달성하는 것이다. 이를 위해 1단계에서는 Time-dependent 가중치 스케줄과 원래 가중치에 대한 ℓ2 regularization을 결합해, 초기에는 기술 정렬을 주 신호로 유지하면서 점차 reasoning 캡션의 영향력을 늘린다. 2단계에서는 reasoning 캡션을 범주별로 정밀 감독하기 위해 다중 라벨 분류(이미지 측 multi-label, 텍스트 측 단일 라벨)를 보조 가지로 추가하고, 추론 유형별 구분성을 높이면서도 공통 표현 공간을 유지하도록 설계했다.

- **Empirical Impact**: 실험에서 ReasonCLIP-58M 계열은 다양한 backbone/스케일에서 시각 근거 추론 및 compositional reasoning 벤치마크의 성능을 일관되게 개선했으며, reasoning corpus 규모가 상대적으로 작아도 더 큰 데이터/아키텍처 변경 기반 접근을 넘어서는 결과가 보고된다. 특히 RCLIP-Bench의 3단계(Visual Grounding, Evidence Awareness, Visually Grounded Reasoning)로 진단해 보면, 단순 이미지-언어 연관 성능이 곧바로 추론 능력을 보장하지 않음을 보여주고 ReasonCLIP은 세 수준 모두에서 향상을 보였다. 또한 LLaVA-NeXT처럼 MLLM에 visual tower를 drop-in으로 교체했을 때 추가 추론 비용 없이 OKVQA, VisualLogic, GQA, MMStar 등에서 재현 가능한 이득이 관찰되며, 비추론 작업에서도 성능 안정성이 확인됐다.



### MIRROR: Novelty-Constrained Memory-Guided MCTS Red-Teaming for Agentic RAG (https://arxiv.org/abs/2606.26793)
Comments:
          6 pages, 2 figures. Accepted at the 2026 International Joint Conference on Neural Networks (IJCNN 2026), IEEE WCCI 2026; presented as an oral talk. Code and ART-SafeBench benchmark: this https URL

- **Prior Approaches**: 기존 멀티모달 에이전틱 RAG red-teaming은 표면(surface)별로 따로 다루는 경우가 많고, 알려진 공격 템플릿을 재활용하는 경향이 커서 실제로 ‘새 취약점’을 찾는지 확인이 어렵다. 특히 텍스트 poisoning 벤치마크에서는 시드 풀 기반 방법들이 73–84% 수준의 정확한 중복(Exact duplication)으로, ASR이 ‘발견’이 아닌 ‘복제’를 과대평가할 수 있다는 한계가 관찰됐다.

- **Core Contribution**: 논문은 MIRROR(Memory-Informed Red-teaming with Retrieval-Restricted Optimization and Rollouts)라는 단일 프레임워크로 텍스트/이미지/다이렉트 쿼리/오케스트레이터 공격(B1–B4)을 한 번에 계획·검증하도록 제안한다. 핵심은 검색으로 탐색 우선순위를 만들되, 검색된 내용을 그대로 베끼는 것을 막는 deterministic Novelty Gate(노벨티 게이트)로 후보를 하드 제약 조건에서 차단한다.

- **Technical Challenges**: 다만 retrieval이 공격 후보 생성에 도움을 주는 동시에, prompt copying 같은 복제 기반 성공도 유발할 수 있어 ‘새로움’을 정확히 세는 통제 장치가 필요하다. MIRROR는 episodic memory에서 k-NN으로 operator priors를 뽑고, 후보가 검색 이웃/벤치마크 풀/세션 내 수락본과 정규화된 동일성에서 겹치면(공백·영숫자 정규화) 시뮬레이터/타깃 쿼리 전에 즉시 Novelty Gate로 거부해 중복을 계산 가능하게 만든다. 또한 memory-guided Monte Carlo tree search와 simulator rollouts, 그리고 가능할 때 real target에 대한 deterministic replay로 최종 검증을 이중으로 수행한다.

- **Empirical Impact**: ART-SafeBench(v2.0.0) 위에서 GeneralRAG 타깃을 대상으로 검증된 cross-surface 결과에서 MIRROR는 네 표면을 모두 커버하는 유일한 방법이며 ASR의 교차표면 변동성이 가장 낮다(CV=0.47). B2 image poisoning은 76%로 baselines(52%) 대비 크게 개선되고, B4 orchestrator attacks는 97% ASR을 절반 수준 쿼리 비용으로 달성했으며 B1 text poisoning에서는 DupBench@Exact 0%를 보이면서 Novel-ASR을 함께 보존했다. 공개된 데이터·어댑터(41,815 in-package, 41,991++ total)와 재현 가능한 검증 프로토콜은 향후 에이전틱 RAG 안전평가에서 ‘복제 기반 성공’ 문제를 줄이는 기준점이 될 것으로 보인다.



### AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing (https://arxiv.org/abs/2606.26787)
Comments:
          Accepted by KDD 2026 Applied Data Science Track (Oral presentation)

- **Prior Approaches**: 기존 동적 가격 모델은 룰 기반 휴리스틱이나 수요-가격 탄력성 추정/최적화에 의존하는 경우가 많지만, 결정의 투명성이 낮고 리뷰·상품 설명 같은 비정형 정보를 충분히 쓰기 어렵다. 또한 단기 매출 중심 최적화가 누적 GMV, ROI, 마일스톤 같은 장기 목표와 어긋나며, 공격적인 할인은 마진 훼손과 운영 제약으로 이어질 수 있다. RL 접근은 장기 보상을 직접 최적화하려 해도(가치함수 기반) 해석 가능성과 오프라인 분포이동, 보상 희소성 문제가 남는다.

- **Core Contribution**: 논문은 LLM 기반 동적 가격 프레임워크 AIGP(Artificial Intelligence Generated Pricing)를 제안한다. AIGP는 도메인지식·구조화 데이터·텍스트 컨텍스트를 프롬프트에 결합해 해석 가능한 가격 결정을 만들고, 장기 비즈니스 목표에 정렬되도록 정책을 학습한다. 핵심 모듈은 Long-Term Value Estimator(LTVE)로, 가격 액션의 장기 가치를 평가해 Direct Preference Optimization(DPO)용 선호 쌍을 자동 생성함으로써 장기 목표 정렬을 가능하게 한다.

- **Technical Challenges**: 기술적 난제는 (1) 가격이 미래 노출/트래픽/매출에 영향을 미치는 지연 보상을 오프라인에서 안정적으로 반영하는 것과 (2) 장기 성과를 직접 비교할 신뢰도 높은 학습 신호를 확보하는 데 있다. 이를 위해 LTVE는 critic-only 구조의 오프라인 RL로 6개월 운영 로그(5백만+ 전이)를 학습해 후보 액션을 장기 가치로 스코어링하고, 그 스코어를 기준으로 DPO의 chosen/rejected 쌍을 만든다. 또 결정 최적화 과정에서 비교 불가능한 긴 추론문이 신호를 흐리지 않도록 [ACT_ONLY] 제어 토큰으로 ‘액션 토큰’에 DPO를 집중시키되, 배포 시에는 투명성을 위해 reasoning을 다시 생성하도록 설계했다.

- **Empirical Impact**: AIGP는 Tao Factory에서 오프라인 평가와 대규모 온라인 A/B 테스트로 효과를 검증했으며, 14일 기준 생산 기준선 대비 GMV +13.21%, ROI +7.59%, 마일스톤 달성률 +8.20% 개선을 보고한다. 동시에 추론 기반의 가격 합리화가 제공되어 운영 관점의 해석 가능성과 신뢰성을 유지한다. 전개 측면에서도 teacher-student distillation과 SFT를 통해 30B급 학생 모델이 대규모 생산 환경에서 효율적으로 동작하도록 맞췄다는 점이 강조된다.



### Anatomy-Guided Residual Motion Diffusion for Controllable 4D Cardiac MRI Synthesis (https://arxiv.org/abs/2606.26764)
- **Prior Approaches**: 4D(3D+time) 의료영상 생성은 희소한 주석, 기기 간 도메인 시프트, 개인정보 제약 때문에 어려웠습니다. GAN·확산 기반 접근들은 시각적 사실성은 개선했지만, 해부학(정적 구조)과 시간적 동역학을 분리해 독립적으로 제어하거나, 장기 일관성(temporal coherence)을 보장하는 데 한계가 컸습니다. 또한 강한 라벨 의존이나 강하게 결합된 생성 방식 때문에 paired segmentation까지 스케일링하기가 어려웠습니다.

- **Core Contribution**: 이 논문은 cine cardiac MRI의 4D 생성 문제를 ‘정적 해부학 생성’과 ‘잔차 기반 시간 모션 생성’으로 분리하는 4D controllable generative framework를 제안합니다. semi-supervised VAE가 해부학 잠재표현을 학습하면서 강하지 않은 라벨 환경에서도 intensity와 정렬된 segmentation mask를 함께 생성하도록 설계했습니다. 이어서 static LDM은 임상 priors(진단 및 용적 지표 등)으로 특정 해부학을 만들고, residual latent motion LDM이 ED 대비 잔차 모션을 생성해 시퀀스 전 구간의 temporal coherence를 유지합니다.

- **Technical Challenges**: 핵심 난제는 (1) 해부학 일관성을 유지한 채 고차원 4D를 생성하고 (2) 기기·환자별 변동 속에서도 임상 priors를 잘 따르는 제어력을 확보하는 것입니다. 이를 위해 VAE에서 labeled/ unlabeled를 함께 쓰는 semi-supervised 학습으로 의미 정렬을 강화하고, 시간축은 ED 기준의 residual latent trajectory로 모델링해 anatomy–dynamics 분리를 달성했습니다. 또한 시간 단계 생성 시 latent 기반으로 residual을 누적하고, 임상 조건은 cross-attention 기반으로 주입해 CFG로 제어하며, 가변 slice 수는 valid slice 마스크로 패딩 학습을 회피하도록 구성했습니다.

- **Empirical Impact**: cine cardiac MRI에서 여러 데이터셋 평가 결과, static anatomy 제어는 Pearson r > 0.8 수준의 높은 정합성을 보였고 temporal coherence는 FVD=288.08로 보고됐습니다. cross-vendor 일반화 실험에서 합성 4D 시퀀스를 학습에 추가하면 downstream segmentation이 개선되며, nnU-Net 기준 평균 Dice가 1.4%p 상승하고 Hausdorff Distance는 3.0mm 감소했습니다(좌심실: Dice +2.8%, 경계 오차 5.4mm 감소). 전반적으로 희소 주석과 도메인 시프트 상황에서 4D 데이터 증강을 “규모화 가능한 controllable 합성”으로 제공한다는 점에서 임상 보조 AI의 실용성에 의미가 큽니다.



### Robust Onion: Peeling Open Vocab Object Detectors Under Nois (https://arxiv.org/abs/2606.26734)
Comments:
          Accepted at The 19th European Conference on Computer Vision (ECCV)

- **Prior Approaches**: 기존 OV-OD 연구는 잡음(비·안개·압축·모션 블러 등)이 성능을 떨어뜨린다는 사실은 다뤘지만, 왜/어디서 무너지는지(모델 내부 병목)는 “복잡한 구조 때문에” 잘 분리해 설명하기 어려웠습니다. 또한 현실의 저품질(HQ-LQ) 쌍을 맞춘 데이터가 부족해, 합성 잡음을 쓰더라도 구성요소별 영향 분해가 제한적이었습니다.

- **Core Contribution**: 이 논문은 Robust Onion으로 OV-OD의 견고성 저하를 “레이어별 feature collapse 붕괴” 관점에서 실험적으로 분해합니다. 합성 시각 열화로 관측 가능한 붕괴(observable)와 거의 보이지 않는 붕괴(minimal)를 모사해, 모델이 어떻게/어디서 깨지는지 해석 가능하게 드러내며 이전 robustness 관찰들을 체계적으로 재설명합니다.

- **Technical Challenges**: 핵심 과제는 (1) 현실 잡음의 효과를 직접적으로 측정할 저품질-고품질 쌍 데이터가 없고, (2) OV-OD가 비전 백본·텍스트·퓨전·박스 예측 등 여러 모듈로 얽혀 있어 원인 분리가 어렵다는 점입니다. 저자들은 controlled synthetic degradations로 붕괴 유형을 단계적으로 “벗겨내는” 방식으로 해결하고, UMAP 등으로 층별 특징 겹침을 확인해 얕은 레이어의 취약성과 유사 백본 간 비슷한 붕괴 패턴을 제시합니다.

- **Empirical Impact**: COCO·LVIS에서는 이미지 도메인이 주로 견고성을 좌우하며, ODinW-13처럼 큰 단일 객체가 많은 벤치마크는 robustness를 과대평가할 수 있음을 보여줍니다. 또한 NN & TK0 플러그앤플레이 방식으로 BDD100K·WiderFace·VisDRONE에서 실세계 견고성을 개선했는데, end-to-end 대비 학습 가능한 파라미터를 96배 줄이면서도 비슷한 수준의 robustness를 달성합니다.



### MLFFM-SegDiff: A Multi-Level Feature Fusion Diffusion Model for Skin Lesion Segmentation (https://arxiv.org/abs/2606.26712)
- **Prior Approaches**: 피부 병변 분할은 컴퓨터 보조 진단에서 핵심 단계지만, 경계가 흐리고 대비가 낮으며 모발·그림자 같은 아티팩트와 환자·장비·조명에 따른 큰 변동성이 문제로 지적돼 왔다. 기존 U-Net 계열은 skip connection으로 국소 디테일과 의미를 결합하지만, 같은 레벨 간 전달에 머물러 크로스-레벨 상호작용과 다중 스케일 융합이 제한되기 쉽다. 확산 모델(DermoSegDiff 등)은 노이즈 제거를 반복하며 경계를 다루는 데 강점이 있으나, 레벨 간 feature 상호작용과 경계 디테일 복원이 충분치 않다는 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 피부 병변 분할을 위한 멀티레벨 feature fusion diffusion 모델 MLFFM-SegDiff를 제안한다. 핵심은 (1) dual-path U-Net encoder로 노이즈 마스크 특징과 피부영상 특징의 상호작용을 강화하고, (2) MLFFM(Multi-Level Feature Fusion Module)로 encoder-Decoder 사이 skip 특징을 주의(attention)·스케일 정렬·적응형 크로스레벨 융합까지 포함해 결합하는 것이다. 여기에 (3) 필요 시 경계에 가중치를 더 주는 boundary-sensitive loss를 설정 가능하게 도입해 애매한 윤곽 학습을 보조한다.

- **Technical Challenges**: 확산 기반 분할에서 노이즈 마스크와 안내(image guidance) 특징의 결합이 단순하면 흐린 경계 상황에서 의미·경계 표현을 동시에 놓치기 쉽다. MLFFM-SegDiff는 mask branch와 guidance branch를 분리 인코딩하되 잔차 블록과 피드백(원소곱 기반)으로 조건 상호작용을 촘촘히 구성하고, bottleneck에서 attention 기반 혼합으로 전역 표현을 보강한다. 또한 skip 단계에서 SkipBlock의 공간/채널 attention과 적응형 스케일 정렬 및 원소곱 융합을 통해 레벨 간 정보를 decoder가 함께 활용하도록 설계했으며, 경계 민감 손실은 거리 기반 가중치로 경계 픽셀의 학습 중요도를 시간에 따라 조절해 경계 복원을 유도한다.

- **Empirical Impact**: 실험은 ISIC2018, PH2, HAM10000에서 Accuracy, F1-score, Jaccard index, Recall, Dice 등 다수 지표로 평가됐고, U-Net·SwinUNETR·DermoSegDiff 등 대표 방법 대비 전반적 성능 우위가 보고됐다. 특히 전체 평균에서 Jaccard index 0.8546, Dice coefficient 0.9207을 달성해 병변 영역 커버리지와 겹침 품질이 향상됐음을 수치로 확인했다. 논문은 ablation을 통해 dual-path encoder, MLFFM, boundary-sensitive loss가 각각 또는 조합으로 Dice와 sensitivity(회상/커버리지) 개선에 기여하며, 경계가 불명확한 케이스에서도 예측 윤곽을 더 잘 따르는 정성 결과를 제시했다.



### Algorithmic Foundations of Deep Learning: Complexity-Theoretic Rates and a Characterization of Universal Approximation (https://arxiv.org/abs/2606.26705)
Comments:
          27 Main Body, 48 Page Proofs, 9 Figures

- **Prior Approaches**: 기존 신경망 근사 이론은 주로 함수의 매끄러움·규칙성(예: Hölder, Sobolev, Besov, holomorphic)을 기준으로 “네트워크가 얼마나 빨리 근사하는가”를 설명해 왔다. 이 관점은 깊이의 효과(예: bit-encoding 기반 가속)는 보여주지만, square-root 같은 단순 함수와 Brownian motion 같은 계산적으로 복잡한 대상의 차이를 직관적으로 갈라내지 못한다. 결과적으로 같은 regularity를 가진 함수들 사이에서 실제 난이도 차이를 보수적으로 예측하거나 놓칠 수 있다는 한계가 제기된다.

- **Core Contribution**: 논문은 신경망 근사를 ‘유연한 기저함수’가 아니라 ‘계산 모델’로 본다. 함수의 난이도를 정규성만이 아니라, 정해진 elementary gate language로 주어진 비재귀(real-valued) 회로가 그 함수를 목표 정확도까지 계산하는 데 필요한 알고리즘 복잡도로 측정한다. 그리고 non-affine nonlinearity를 포함하는 definable feedforward NN은 C([0,1]^d)에서 universal approximation을 만족하며, 이는 정성적 보장에 그치지 않고 회로-기반 정량적 컴파일로 이어진다.

- **Technical Challenges**: 핵심 난제는 ‘회로(게이트 수/구조/깊이) 복잡도’를 ‘신경망(깊이·폭·비영(0이 아닌) 파라미터 수) 복잡도’로 정량 변환하는 규칙을 만드는 것이다. 논문은 circuit 형태로 표현된 실수 연산 레시피를 execution graph로 정리한 뒤, attention·layer normalization 같은 다변수 non-linearities가 허용되는 definable NN 모델(특히 o-minimal definability 범위)로 컴파일하는 원리를 제시한다. 그 결과 네트워크가 게이트 구조와 비슷한 수준으로 비선형 근사를 수행하도록, depth/width/size 경계를 통제하는 이론(Quantitative Neural Compilation Theorem 계열)을 제공한다.

- **Empirical Impact**: 이 관점은 실험 이전에 이론적으로 다수의 ‘근사율’과 ‘계산 알고리즘 에뮬레이션’을 한 번에 설명한다. 연속함수·Besov/holomorphic 클래스에 대한 minimax-optimal 근사 보장뿐 아니라 Newton-Raphson root finding, power iteration 같은 수치 알고리즘과 최단경로(APSP) 예시에서도 회로를 컴파일해 O(log(1/ε)) 비영 파라미터로 달성함을 제시하며, 이는 기존 Lipschitz 기반 스케일 대비 큰 개선이다. 결론적으로 approximation power는 ‘차별점’이 아니라 기본값이 되고, 앞으로는 inductive bias·대칭성·기하·안정성·확장성과 최적화 거동이 비교의 중심이 돼야 한다는 메시지를 강화한다.



### Learning Motion Feasibility from Point Clouds in Cluttered Environments (https://arxiv.org/abs/2606.26700)
- **Prior Approaches**: 기존 연구는 Sampling-based motion planners(SBMPs)를 가속하거나, infeasibility certification으로 실패를 증명하는 방식에 집중해 왔지만 주로 저차원 configuration space에 한정되는 문제가 컸습니다. 또한 실제 로봇이 쓰는 고차원(예: 7-DOF)에서, 원시 RGB-D 관측을 직접 다루기보다 원시 기하를 단순 도형(primitive)으로 가정하는 접근이 많았습니다. 그 결과 복잡한 클러터(cluttered) 환경에서는 infeasible 시도에 드는 계산 비용을 충분히 줄이기 어려웠습니다.

- **Core Contribution**: 이 논문은 7-DOF 매니퓰레이터가 실제 RGB-D 클러터 장면에서 grasp에 대해 motion feasibility(가능/불가능)을 “직접 예측”하는 문제를 제안합니다. 이를 위해 대규모 벤치마크 MoFeas(GraspNet-1Billion 기반)를 구축해, 88개 실제 스캔 오브젝트와 190개 클러터드 테이블 장면에서 총 2.71M개의 grasp별 RRT-Connect 라벨(가능/불가능)을 제공합니다. 또한 MLP, voxel-기반 3D-CNN, point-cloud Transformer의 대표 모델군을 동일 조건에서 비교해 “원시 포인트클라우드→feasibility” 학습이 성립함을 보여줍니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 원시 point cloud와 자세(그립 포즈)를 조합해, (2) kinematics+충돌+경로탐색 실패를 포착하는 학습 신호를 대규모로 만드는 것이었습니다. 저자들은 각 grasp 후보에 대해 IK, pre-grasp standoff IK, RRT-Connect 경로탐색, approach 충돌 체크를 수행해 SBMP 결과를 ground-truth 오라클로 라벨링하고, 장애물 클라우드에서 grasp 주변만 국소적으로 제거해 실제 클러터 제약을 유지했습니다. 모델 측에서는 PTv3 기반 GRASPFC-PTX가 grasp pose를 token으로 주입하고, scene 기하 신호를 잘 활용하도록 설계하며, 선택적으로 swept-volume 채널도 실험에 포함했습니다.

- **Empirical Impact**: GRASPFC-PTX는 Novel 오브젝트 분할에서 AUROC 0.996, TPR 98.5%, TNR 97.1%를 달성해 분포 내 고성능을 보입니다. 동시에 단일 예측 지연은 10ms 미만 수준으로 유지되어, RRT-Connect의 infeasible worst case에 비해 80~171배(조건에 따라) 빠른 속도 향상을 보이고, Top-5 grasp success가 분포 이동(OOD)에서도 100%로 유지되는 점이 실용적 의미가 큽니다. 다만 Unseen-scene OOD에서는 AUROC가 85%로 하락해, 단일 테이블 프레임 구성이 아닌 cross-scene 혼합 학습이 추가 과제로 제시됩니다.



### Beyond Logical Forms: LLM-Extracted Patterns for Fallacy Classification (https://arxiv.org/abs/2606.26698)
- **Prior Approaches**: 기존 연구는 논리 오류를 문장 수준의 특징이나 사전 학습된 분류기에 의존해 분류하는 경우가 많지만, 미묘한 표현에서는 추론 구조가 제대로 반영되지 않는 한계가 있었다. 또 LLM을 활용하더라도 fallacy의 추상적 논리 골격과 맥락의 언어 신호를 함께 통합하는 방식이 일관되게 정리되지 않았다.

- **Core Contribution**: 이 논문은 fallacy의 추상 논리 구조와 맥락 수준의 언어 단서를 함께 결합해 분류 성능을 높이는 프레임워크를 제안한다. LLM을 이용해 오류 예시와 그에 대한 설명으로부터 논리 패턴을 귀납적으로 추출하고, 이를 데이터 기반 “논리 표현”으로 생성해 활용한다.

- **Technical Challenges**: 핵심 난제는 (1) 미묘한 뉘앙스 속에서 구조적 추론 패턴을 추출해야 하고 (2) LLM이 설명 텍스트에서 그 패턴을 재현 가능하게 학습·구성해야 한다는 점이다. 논문은 LLM이 fallacious example과 explanation에서 패턴을 유도적으로 뽑아내도록 설계해, zero-shot 기준선 대비 분류에 직접 도움이 되는 구조 신호를 구성한다.

- **Empirical Impact**: 실험에서는 여러 LLM과 zero-shot 및 one-shot 설정 전반에서 제안 방식이 통계적으로 유의미한 개선을 보였고, 경쟁 접근법도 능가했다. 또한 데이터셋 간 실험으로 일반화가 확인되며, 설명 기반의 데이터-driven 패턴 추출이 논리 표현 생성의 효과적인 방법임을 입증했다.



### Disco-LoRA: Disentangled Composition of Content, Style, and Motion for Multi-concept Video Customization (https://arxiv.org/abs/2606.26668)
- **Prior Approaches**: 기존 텍스트-투-비디오(T2V) 기반 영상 커스터마이징은 기준 이미지의 외형(콘텐츠) 보존이나 기준 영상의 동작 패턴(모션) 모사에 주로 집중해 왔습니다. 또 일부는 이미지에서 콘텐츠-스타일 조합만 다루거나, 시퀀스 전체를 일대일로 학습하는 방식에 머물러 새로운 조합으로의 확장성이 제한적이었습니다.

- **Core Contribution**: 이 논문은 콘텐츠·스타일·모션을 함께 제어하는 multi-concept video customization을 체계적으로 정의하고, 4가지 조합 과제를 평가하도록 종합 벤치마크를 구축했습니다. 이어 Disco-LoRA를 제안해 콘텐츠-스타일과 콘텐츠-모션을 분해 학습한 뒤 LoRA를 유연하게 재조합함으로써 임의의 다중 개념 동시 제어를 노립니다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 개념을 제대로 disentanglement하는 것과 (2) 조합 시 inter-concept interference를 줄이는 것입니다. 이를 위해 Iterative Dual-LoRA Disentanglement는 단계별로 LoRA를 번갈아 학습하고 complementary prompting 및 time-aware masking으로 정보 누수를 억제하며, Z-score 기반 정규화는 LoRA 레이어별 weight trend은 유지하면서 magnitude를 맞춰 선형 결합 간섭을 완화합니다.

- **Empirical Impact**: 실험에서는 벤치마크 전 과제에서 Disco-LoRA가 외형·스타일·모션 정합성뿐 아니라 운동의 충실도와 매끄러움, 지각 품질(PickScore 등)에서도 우수한 결과를 보였고, 사용자 연구에서도 모든 평가 항목에서 평균 4점대의 점수를 기록했습니다. 특히 기존 방법들이 스타일/콘텐츠/모션이 서로 엉키거나 초기 프레임 불일치가 연쇄적으로 실패하는 문제를 Disco-LoRA가 더 일관되게 해결하며, 상용 모델 대비도 다중 조건 통합 제어 성능 격차를 입증했습니다.



### TGHE: Template-based Graph Homomorphic Encryption for Privacy-Preserving GNN Inference in Edge-Cloud Systems (https://arxiv.org/abs/2606.26664)
Comments:
          7 pages, 3 figures, 3 tables. Accepted at IEEE ICWS 2026

- **Prior Approaches**: 기존 HE 기반 GNN은 CryptoGCN, LinGCN, Penguin, FicGCN처럼 그래프 중심(graph-centric)으로 전역 인접행렬을 암호화·연산해 비용이 그래프 크기에 비례했다. 그 결과 수만 노드 수준의 벤치마크에 머무르고, 동적으로 변하는 금융 그래프·엣지-클라우드 지연 요구에는 맞추기 어려웠다. 또한 전처리(파티셔닝·리오더링)가 복잡도와 잠재적 정보 노출 부담을 키웠다.

- **Core Contribution**: TGHE는 트랜잭션 그래프의 로컬 ego-graph를 엣지에서 추출·정규화해, 전역 그래프 크기와 무관하게 쿼리 단위 암호화 추론을 가능하게 하는 ego-centric 프레임워크를 제안한다. 핵심 통찰은 ‘template phenomenon’으로, 로컬 계산 트리들이 소수의 구조 모양으로 수렴한다는 점을 CKKS SIMD 배치에 활용한다. 이를 통해 대규모·동적 금융 그래프에서도 실용적인 프라이버시 보존 추론을 노린다.

- **Technical Challenges**: HE에서는 계산 회로가 균일해야 하는데, ego-graph는 차수·깊이·연결성이 불규칙해 그대로 SIMD 배치가 어렵다. TGHE는 (1) 역할 트리로 트리화(treeification)해 결정적 구조를 만들고 (2) template signature로 구조 동일성을 묶어 공유 ciphertext에 packed inference를 수행한다. 그래도 남는 ‘long-tail’ 템플릿 불일치는 Approximate Template Fitting(패딩으로 의미 보존)과, GEARSage의 평균풀 mean-pool 선형성에 기대는 Topology Collapse( hop-2 다양성 제거)로 해결해 SIMD 커버리지를 극대화했다.

- **Empirical Impact**: DGraphFin(3.7M 노드, 4.3M 엣지)에서 TGHE-Collapse는 순차 encrypted baseline 대비 66.9x 엔드투엔드 속도 향상을 달성하면서 AUC 손실은 0.002 미만 수준(예: 0.7973대 0.7959 등)으로 보고됐다. 템플릿 기반 SIMD 배치는 거의 모든 쿼리를 packed 경로로 보내지만, Base에서는 fallback이 Amdahl’s Law 병목으로 지연을 지배했다. 반면 Topology Collapse는 fallback 경로를 제거해 회전 수를 99% 가까이 줄이며, 대규모 금융 그래프 프라이버시 추론의 실행 가능성을 실증했다.



### Zero-Shot Size Transfer for Neural ODEs on Sparse Random Graphs: Graphon Limits and Adjoint Convergenc (https://arxiv.org/abs/2606.26662)
- **Prior Approaches**: Neural ODE는 연속 깊이 모델로, Graph Neural Differential Equations(GNDEs)는 이를 그래프 구조에 맞게 GNN으로 속도장을 매개변수화해 동역학을 학습한다. 기존에는 DTO/OTD로 학습하며 매 훈련마다 ODE 수치해석과 그래프 메시지 패싱이 반복돼 비용이 커지고, 그래프 크기 변화에 대한 재학습 필요성도 남아 있었다. 한편 size transferability는 이산층 GNN에서는 다뤄졌지만, GNDE처럼 연속 시간 전체 궤적과 gradient(어드조인트)까지 함께 안정적으로 넘어가는 정량 보장은 부족했다.

- **Core Contribution**: 논문은 zero-shot size-transfer 원리를 그래프온(graphon) 관점에서 정리하고, Graphon Neural Differential Equations(Graphon-NDEs)을 GNDE의 무한 노드 한계로 도입한다. 또한 hidden-state 및 파라미터 gradient를 위한 adjoint Graphon-NDEs까지 함께 구성해, 크기 전이에서 forward뿐 아니라 학습에 필요한 역전파 정보의 안정성까지 다룬다. 무작위로 샘플링된 희소 그래프 조건에서 실제 GNDE가 Graphon-NDE로 수렴한다는 “전이 가능성”의 이론적 근거를 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 희소 랜덤 그래프에서 연속시간 GNDE 해가 graphon 한계로 얼마나 빨리 수렴하는지, (2) adjoint 시스템의 균일-시간(uniform-in-time) 오차를 제어하는지, (3) 훈련에서 쓰는 이산화(DTO/OTD, explicit Euler)가 연속 adjoint와 정합적인지 정량화하는 것이다. 이를 위해 well-posedness를 먼저 세우고,trajectory-wise 수렴률을 O((α_n n)^(-1/2)) 수준으로 보이며(로그 요인 포함), 어드조인트의 hidden-state/파라미터 gradient에 대해서도 균일-시간 수렴 경계를 제공한다. 더 나아가 explicit Euler에서 DTO/OTD가 점진적으로 일관된 gradient를 준다는 점을 보이며, hidden-state는 O(1/M), 로컬 파라미터 gradient는 O(1/M^2) 오더로 오차가 줄어든다고 증명한다.

- **Empirical Impact**: HSBM 및 tent graphon 실험을 통해 forward 궤적, adjoint gradient, 이산화 오차, DTO–OTD 불일치가 이론에서 예측한 수렴률 양상과 일치함을 확인한다. 또한 네 가지 graphon 클래스에서, 작은 그래프에서 학습한 GNDE를 더 큰 그래프에 zero-shot으로 배치했을 때 독립 샘플링된 목표 그래프에서도 동역학을 정확히 맞추는 전이 성능을 보인다. 결과적으로 GNDE의 zero-shot size transfer가 단순 경험이 아니라 graphon 기반 정량 이론과 함께 뒷받침되어, 그래프 크기가 커질 때의 재학습 비용 절감 가능성을 강화한다.



### LAMP: Lane-Aligned Motion Primitives for Feasible Trajectory Prediction (https://arxiv.org/abs/2606.26661)
Comments:
          IEEE ITSC 2026, 6 pages

- **Prior Approaches**: 기존 모션 예측은 다중 모드를 만들더라도, 대부분은 변위 오차(가장 그럴듯한 모드 중심) 최적화에 치우쳐 장애물·차선 제약 같은 구조적 제약을 충분히 반영하지 못했다. 또한 trajectory anchors나 endpoint 기반 intention prior는 장면에 비의존적이라 lane topology와의 정합성이 깨지기 쉽고, 낮은 확률 모드에서 off-road 또는 traffic-rule 위반 같은 비현실적 예측이 섞이곤 한다. 그 결과 안전한 계획 모듈에 쓰기엔 예측 집합의 신뢰도가 흔들릴 수 있었다.

- **Core Contribution**: LAMP는 Lane-Aligned Motion Primitives로, 의도를 고정된 endpoint가 아니라 lane topology에 맞춘 구조화된 motion primitives로 모델링한다. VQ-VAE(구체적으로 NSVQ 기반)로 학습한 discrete intention queries가 spatiotemporal 패턴을 담은 궤적 prototype을 생성하게 하여, endpoint 중심 표현에서 놓치기 쉬운 형태와 시간 진화를 보완한다. 여기에 lane-topology-guided(차선 위상 기반) feasibility-aware intention selector를 더해 decoder 이전 단계에서 도달 불가능/비일관 모드를 걸러낸다.

- **Technical Challenges**: 핵심 난제는 다중 모드를 유지하면서도, 낮은 확률 모드까지 포함해 lane 제약·교통 규칙 일관성을 보장하는 “집합 신뢰도”를 높이는 것이었다. LAMP는 (1) motion primitives를 discrete codebook으로 학습해 다양한 행동을 표현하고, (2) HD-map lane connectivity를 이용해 도달 가능한 lane 집합과의 거리 기반 lane prior로 selector를 학습함으로써 decoder 입력에 구조적으로 feasible한 intention만 남기도록 설계했다. 또한 Top-1 중심 손실로 인해 대체 모드가 무시되지 않도록, selector가 decoder가 사용할 mode 자체를 선별·정렬하게 만든 점이 절충의 해법이다.

- **Empirical Impact**: Argoverse 2에서 LAMP는 b-minADE/FDE 같은 표준 변위 지표는 강한 기준선과 비슷한 수준을 유지하면서도, DAC/FR 같은 feasibility(특히 FR의 엄격한 제약)와 diversity(DwF 포함)에서 유의미한 개선을 보였다. FR에서의 향상이 더 크게 나타난 것은 단순히 도로 밖을 피하는 수준을 넘어, lane 연결성 및 교통 규칙 위반 가능성을 줄여 planner-relevant 신뢰도를 높였음을 시사한다. ablation 결과 NSVQ 코드북 선택과 충분한 selector top-L(과소 선택은 모드 수축/성능 저하)을 통해 성능-다양성 균형이 형성되며, 정성 실험에서도 교차로 장면에서 lane topology를 엄격히 따르는 다중 후보를 생성함을 보여줬다.



### CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs (https://arxiv.org/abs/2606.26650)
Comments:
          This work is accepted to ICML 2026 as an oral. The project page: this https URL

- **Prior Approaches**: 기존 1.58-bit(ternary) 양자화는 BitNet 1.58-bit, TriLM, Tequila처럼 QAT(quantization-aware training)에 의존해 ternarization을 시뮬레이션하며 성능 하락을 완화해 왔다. 그러나 이 방식은 수십~수백B 토큰 규모의 대규모 학습과 막대한 계산·시간 비용이 필요해 확장성과 실용성에 제약이 있었다. 반면 PTQ(post-training quantization)는 데이터 비공개 이점이 크지만, 초저비트(ternary)에서는 최적화 수렴이 어려워 성능이 QAT 대비 크게 뒤처졌다.

- **Core Contribution**: CAT-Q는 QAT 없이도 후처리 방식으로 ternary( {1,0,-1} ) 가중치를 구성하는 cost-efficient·accurate PTQ 방법을 제안한다. 핵심은 learnable modulation(LM)으로 사전학습 가중치 분포와 ternarization 임계값의 민감도를 낮춰 분포 불일치를 줄이고, softened ternarization(ST)으로 미분 가능한 전이 함수를 통해 안정적 수렴을 유도하는 것이다. 이를 통해 다양한 아키텍처와 모델 크기에 “바로 적용 가능한” ternary 양자화를 목표로 한다.

- **Technical Challenges**: ternary PTQ에서의 첫 난점은 hard ternarization만으로 생기는 ternary 가중치 분포가 원래 고정밀 가중치 분포와 잘 맞지 않아 정보 손실이 커진다는 점이다. 두 번째 난점은 초저비트의 비미분(discrete) 특성 때문에 최적화가 잘 수렴하지 않는다는 점으로, 기존처럼 캘리브레이션 학습 내내 hard ternarization을 쓰면 이 문제가 두드러진다. CAT-Q는 LM으로 분포 정렬을 돕고, ST는 두 단계(미분 가능한 전이 → 최종 hard ternarization) relay를 통해 전이 과정에서 gradient 기반 학습이 가능하도록 설계했으며, 여기에 sliding-layer 기반 출력 재구성까지 결합해 PTQ 최적화 난이도를 추가로 완화한다.

- **Empirical Impact**: 실험에서 CAT-Q는 1.7B~8B 파라미터 LLM을 512개의 캘리브레이션 샘플(약 1M 토큰)만으로 ternary 모델로 만들면서, 100B 토큰을 학습한 BitNet 1.58-bit 계열(QAT)보다 더 높은 성능을 보였다. 또한 14B~235B급 대형 LLM에도 처음으로 확장성을 보여주며, 8×A100-80GB에서 8~60시간 내 ternary 모델을 구축할 수 있음을 보고했다. 이는 ternary 양자화의 실용성을 가로막던 “데이터·토큰·학습 비용” 장벽을 크게 낮춰, 초저비트 배포를 촉진할 잠재력이 크다는 평가를 받는다.



### Agents That Know Too Much: A Data-Centric Survey of Privacy in LLM Agents (https://arxiv.org/abs/2606.26627)
Comments:
          17 pages, 4 figures, 7 tables

- **Prior Approaches**: 기존 연구는 대부분 챗봇 시나리오처럼 최종 답변에 집중하거나, 특정 구성요소(예: retrieval-augmented generation, text-to-SQL, agent memory, prompt injection, access control)별로 흩어져 발전해 왔다. 그 결과 데이터 에이전트가 실행 전 과정에서 접촉하는 여러 데이터 표면을 하나의 관점으로 연결해 설명하기가 어려웠다. 또한 현장에서는 여러 보호 기법이 서로 보완적으로 검증되기보다 개별 위험에 대해 부분 대응되는 경우가 많아, end-to-end 관점의 공백이 남아 있다.

- **Core Contribution**: 이 논문은 LLM agent의 프라이버시를 ‘데이터 중심 관점’에서 재정리해, 에이전트가 접촉하는 data surfaces를 기준으로 위험과 거버넌스를 묶어 제시한다. 특히 data agent(데이터를 질의·검색·변환·요약·기억·행동하는 LLM 기반 에이전트)를 정의하고, 어떤 데이터가 어디에 존재하며 어떤 프라이버시 위험을 만드는지 체계적으로 분류한다. 또한 기존 벤치마크가 무엇을 측정하는지 매핑하고, 단일 프라이버시 정책 아래 end-to-end로 여러 데이터 표면을 동시에 관통하는 평가가 부족함을 지적한다.

- **Technical Challenges**: 핵심 난제는 프라이버시가 단일 출력물이 아니라 쿼리/중간 결과/도구 인자/메모리 기록/에이전트 간 메시지까지 포함한 ‘실행 전체’의 속성이라는 점이다. 저자들은 데이터 표면을 6가지(데이터베이스·웨어하우스, 표/파일, RAG 말뭉치·vector store, 도구·외부 API, agent memory, multi-agent communication)로 열거하고, 각 표면이 갖는 유출 유발 경로를 outcomes와 vectors 관점으로 분해해 통일된 분류 틀을 만든다. 더 나아가 거버넌스 메커니즘 중 정보흐름제어(information-flow control)가 (구성적) 누출과 세션 간 추론 누출까지 포괄한다는 결론을 반복적으로 제시한다.

- **Empirical Impact**: 논문은 프라이버시 위험을 측정하는 벤치마크들을 정리·비교해, 각 벤치마크가 커버하는 표면과 취약한 지점을 드러내며 ‘명시적 프라이버시 정책 하의 end-to-end 데이터 에이전트 워크플로우’가 거의 없다는 공백을 강조한다. 또한 지금까지는 위험은 넓게 연구되었지만 동일한 정책 아래 여러 표면을 관통하는 검증이 부족하다는 점에서, 향후 평가 설계의 방향을 명확히 제안한다. 결과적으로 데이터 관리·에이전트 보안 커뮤니티를 연결하는 레퍼런스로 기능하며, 정보흐름제어 채택 우선순위와 새로운 벤치마크 필요성을 실무 및 연구 의사결정에 직접적으로 영향을 줄 것으로 보인다.



### Discovering Millions of Interpretable Features with Sparse Autoencoders (https://arxiv.org/abs/2606.26620)
- **Prior Approaches**: 기존 Sparse Autoencoder(SAE)는 초과완비(latent) 특징을 희소하게 분해해 ‘superposition’ 문제를 완화하는 도구로 자리 잡았다. 다만 대규모로 공개된 SAE는 특정 모델군(예: Gemma Scope) 위주였고, Qwen3 계열에 대한 학습·공개 리소스는 상대적으로 제한적이었다.

- **Core Contribution**: 이 논문은 Qwen3 instruction-tuned 모델군을 대상으로 Qwen3-Instruct SAE를 공개한다. Qwen3-1.7B·Qwen3-4B는 residual streams, MLP outputs, attention outputs 세 위치의 layer-wise SAE를, Qwen3-8B는 residual stream 일부 레이어(현재 layer 0-8)를 제공한다.

- **Technical Challenges**: SAE 학습은 딕셔너리 크기와 희소성 수준에 따른 fidelity(복원 품질)·안정성의 trade-off가 커서, layer/컴포넌트별로 성능이 달라질 수 있다. 이를 위해 JumpReLU 기반 SAE 학습과 STE 기반 임계값 최적화, DLL·FVE 같은 복원 및 model-recovery 지표로 체계적으로 비교·평가했다.

- **Empirical Impact**: 평가 결과 residual stream과 MLP 기반 SAE가 attention output보다 model 복원(특히 DLL)에서 유리한 경향을 보였고, 레이어에 따라 비단조 패턴(초기 복원 후 저하, 이후 회복)이 관찰됐다. 또한 refusal-steering 사례에서 선택된 SAE feature로 residual stream을 decoder 방향으로 개입해 유해 입력에서 거절 행동을 일관되게 유도할 수 있음을 보였다.



### HiLSVA: Design and Evaluation of a Human-in-the-Loop Agentic System for Scientific Visualization (https://arxiv.org/abs/2606.26614)
- **Prior Approaches**: 기존 LLM 기반 SciVis(과학 시각화) 에이전트는 자연어 기반 상호작용이나 자동화 파이프라인에 치우쳐, 분석 과정에서의 투명성·검증 가능성·사람의 감독이 상대적으로 약했다. 또한 시각화 에이전트는 코딩/툴 통합, 스크립트 생성, 파라미터 튜닝 같은 산출 중심 기능을 주로 다뤘지만, 실행 내내 지속적인 human oversight를 구조적으로 유지하는 설계는 제한적이었다.

- **Core Contribution**: HiLSVA는 human-in-the-loop(인간-에이전트 혼합 주도) 에이전트로서 mixed-initiative SciVis 워크플로를 지향한다. 특히 plan-first 다중 에이전트 구조에 stepwise provenance(단계별 생성 이력)와 안전한 샌드박스 실행, 사용자 피드백 기반 learn-at-test-time(LTT) 적응을 결합해, 자율성이 아니라 협업을 분석의 중심으로 재정의한다.

- **Technical Challenges**: 문제는 (1) 자율 실행을 유지하되 사람이 ‘언제든’ 개입·재지시할 수 있어야 하고, (2) 시각화 결과가 어떻게 만들어졌는지 과정을 복원 가능해야 하며, (3) 잘못된 자동 실행으로 인한 안전·재현성 문제를 막아야 한다는 점이다. HiLSVA는 사용자 승인형 plan 편집, MCP 기반 ParaView 도구 제어와 GUI 직접 조작의 핸드오프, 단계별 상태 복원 가능한 provenance 기록, 그리고 실행을 Docker 격리로 제한하는 방식으로 이를 해결한다.

- **Empirical Impact**: 연구진은 다양한 자율성 설정을 포함해 12명의 사용자(전문성 다양)를 대상으로 controlled user study와 케이스 스터디를 수행했으며, mixed-initiative 상호작용이 과업 완수, 사용자 통제감, 워크플로 투명성을 전반적으로 개선했다고 보고했다. 동시에 실행 효율성과 사람의 감독 사이에서 트레이드오프가 존재함도 드러나, 향후 agentic SciVis의 인체공학적(인간 중심) 설계 원칙을 제시한다.



### SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inferenc (https://arxiv.org/abs/2606.26587)
Comments:
          20 pages, 4 figures

- **Prior Approaches**: FP4 같은 저비트 부동소수와 N:M 준구조 희소성은 모두 하드웨어 실행 경로로 자리 잡았지만, LLM에서는 활성 quantization이 특히 취약하다. 기존 PTQ들은 outlier와 입력 분포를 다루려 하면서도, 활성에 희소성을 적용하면 마스크가 중간값을 버려 정확도 손실이 커지거나 추가 커널/불규칙 데이터 이동이 늘어 FP4 파이프라인과 충돌하는 문제가 있었다.

- **Core Contribution**: SharQ는 training-free 방식으로 활성에 N:M 희소성을 “버림”이 아니라 “라우팅”으로 재구성해, FP4 양자화와 결합하는 방법을 제안한다. 각 활성 텐서에서 온라인으로 outlier-dominated N:M mask를 만든 뒤, (1) FP4로 양자화된 sparse backbone을 희소 FP4 GEMM에 넣고 (2) sparse path에서 생긴 손실과 양자화 오차까지 포함하는 dense FP4 residual을 별도 GEMM으로 보상한다.

- **Technical Challenges**: 핵심 난제는 (a) block-scaled FP4에서 outlier가 로컬 scale을 지배해 정밀도가 붕괴되는 점, (b) training-free로 온라인 N:M 마스크를 생성할 때 임계 경로 비용이 이득을 상쇄할 수 있다는 점, (c) 희소화 오차와 양자화 오차가 결합(coupled)되어 단순 합산 보정이 되지 않는다는 점이다. SharQ는 residual을 “unquantized sparse”가 아닌 “quantized sparse backbone” 기준으로 정의해 결합 오차를 한 신호로 정렬하고, 마스크 생성·잔차 구성·정규화까지 fused 커널로 흡수하며, 두 경로가 FP4 weight payload는 공유하되 scale view만 경로별로 맞춰 스토리지/준비 비용을 줄인다.

- **Empirical Impact**: Llama-3.1-8B, Qwen2.5-7B, Qwen3-30B-A3B, Qwen3-VL-8B에서 SharQ는 NVFP4 대비 FP16 정확도 격차의 43~63%를 회복하며, NVFP4/HiF4/MXFP4 포맷 전반에 일반화된다. RTX 5090에서 FP16 대비 2.2~2.4× 지연 감소, FP8 대비 1.2~1.4× 처리량 향상을 보였고, Wan2.2-T2V-A14B에서는 SageAttention과 결합 시 최대 1.58× 속도up을 보고한다.



### IDEA: Insensitive to Dynamics Mismatch via Effect Alignment for Sim-to-Real Transfer in Multi-Agent Contro (https://arxiv.org/abs/2606.26575)
Comments:
          8 pages, 6 figures

- **Prior Approaches**: 기존 sim-to-real 전이는 domain randomization(시뮬레이터 파라미터 교란)과 domain adaptation(실데이터로 적응)을 주로 사용해 현실 갭을 줄인다. 하지만 MARL은 저수준 연속 제어 액션에 의존해 dynamics mismatch에 민감하거나, 온라인 적응(RMA·HIM 등)은 history/추정 지연 때문에 장기 과업에서 오차가 누적될 수 있다.

- **Core Contribution**: 논문은 다중 에이전트 제어에서 IDEA(effect alignment 기반 dynamics mismatch 불감 sim-to-real)를 제안한다. 핵심은 정확한 물리 모델링 대신, 이산 semantic action을 low-level closed-loop 제어로 실행해 시뮬과 현실에서 “작용 효과(action effects)”가 정렬되도록 학습하는 데 있다. 또한 에이전트 간 action 타이밍 불일치를 줄이기 위해 communication 기반 action synchronization을 도입한다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) 시뮬과 현실의 상태-전이 불일치, (2) 실제 분산 실행에서 발생하는 액션 비동기 문제다. IDEA는 semantic action이 유도하는 closed-loop 전이를 통해 전이 불일치를 이론적으로 구속하려 하고, 동시에 에이전트들이 동일 시점에 joint action을 실행하도록 coordination node 기반 동기화 프로토콜을 설계해 Dec-POMDP의 동시성 가정을 물리적으로 맞춘다. 학습 효율을 위해 Isaac Gym을 GPU 텐서 중심으로 확장해 병렬 환경에서 다양한 기하 구조를 제공한다.

- **Empirical Impact**: 네 가지 multi-agent navigation 시나리오에서 IDEA는 학습 수렴 속도와 최종 성공률 모두에서 기존 DR·RMA·HIM 대비 우수했다. 특히 real-world zero-shot 전이에서 성공률이 baseline 대비 20% 이상 향상되었고, 충돌 0% 및 에이전트 간 실행 타이밍 오류 0을 달성했다. 분석에서도 시뮬 대비 실제 action effect 불일치가 5% 미만으로 줄고, 에이전트 간 시간 지연도 0으로 유지되어 안전성과 협응의 견고성이 확인됐다.



### scBench-Long: Verifiable Benchmarking of Long-Horizon Single-Cell Biology (https://arxiv.org/abs/2606.26563)
- **Prior Approaches**: 기존 AI-바이올로지 벤치마크는 생물학 지식, 실행 가능한 워크플로우, 또는 로컬 분석 단계에 주로 초점을 맞춰 단일 세포 연구의 ‘긴 호라이즌’ 과정을 충분히 평가하지 못했다. 또한 장기 과제에서는 정답(ground truth)을 단일하게 정의하기 어렵고, 같은 데이터가 여러 타당한 결론을 지지할 수 있어 평가 설계가 까다롭다.

- **Core Contribution**: 이 논문은 scBench-Long이라는 장기 목표형(single-cell long-horizon) 벤치마크를 제안한다. 에이전트가 처방된 방법 없이(“raw 또는 near-raw 데이터에서”) 복잡한 과학적 결론을 구조화된 주장으로 복원하는지 21개 평가로 측정하며, 멜라노마 CD8 T 세포 반응부터 KRAS 폐 종양 노화, 치명적 COVID-19 폐 병리까지 다룬다.

- **Technical Challenges**: 핵심 기술적 난제는 장기 과제의 검증 가능 정답을 만들되, 예기치 못한 분석 경로를 부당하게 깎지 않는 것이다. 논문은 후보 주장을 리뷰·검증해 통제된 answer vocabularies로 ‘결론 표면’을 고정하고 결정론적 endpoint 채점(pass/fail)을 수행하되, 진행 상황 진단을 위해 chokepoint 기반 rubric 채점도 함께 제공한다.

- **Empirical Impact**: 1,068개 완료 궤적(trajectory)에서 최고 성능 모델-하네스 조합은 16/63(25.4%)만 통과했으며, 많은 과제가 한 번도 성공하지 못하는 등 신뢰성은 아직 제한적이었다. 다만 rubric 점수는 endpoint 성공과 부분적으로 연관(AUC=0.77)됐고, 모델이 로컬 분석을 그럴듯하게 수행해도 prior(문헌)·수치 크기·단일 모달 추론에 기대거나 인과를 과잉 해석하는 실패 양상이 반복되는 점이 드러나 향후 에이전트 고도화의 방향을 제시한다.



### SpaceRipple: Lightweight Semantic Delivery for Mission-Oriented LEO Earth Observation Satellite Networks (https://arxiv.org/abs/2606.26559)
- **Prior Approaches**: 기존 연구는 화상 압축, 위성 탑재 지능(온보드 처리), semantic communication을 각각 따로 다루는 경향이 있었다. 그 결과 전통적인 “획득–전체 다운링크–지상 처리” 흐름을 크게 바꾸지 못해, 제한된 링크/연산 자원에서 임무 민감도를 충분히 높이기 어렵다는 한계가 제기된다. 특히 SR-powered 위성 획득·복원 방식들은 주로 고화질 복원 자체에 초점이 맞춰져 semantic 최종 목적을 위한 시스템 최적화는 상대적으로 약했다.

- **Core Contribution**: SpaceRipple은 위성 관측에서 다운링크까지의 목표를 ‘픽셀 전달’이 아니라 ‘임무 관련 semantic 정보 전달’로 재정의하고, 이를 위해 압축–전송–복원–semantic 추론을 하나의 협업 파이프라인으로 공동 설계했다. sensing 위성은 adaptive compression과 metadata 생성을 수행하고, edge computing 위성은 수신 표현을 복원·강화한 뒤 task에 필요한 장면/탐지 결과를 추출한다. 최종 다운링크는 전체 영상 대신 의미 메시지로 구성해 “관측–전송–의사결정” 지연을 줄이는 구조를 제안한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 압축으로 정보가 손실된 상황에서도 semantic 추론에 필요한 단서를 안정적으로 보존/복원하는 것이다. SpaceRipple은 이를 위해 압축 과정에서 압축률 및 색/보정 관련 calibration cues 같은 side metadata를 함께 전송하고, edge에서 metadata를 조건으로 복원·전처리하도록 설계했다. 또한 압축 열화에 대한 강건성을 위해 compression-aware MoE enhancement 모듈을 두되, 로컬 픽셀 특징과 metadata의 압축률을 함께 라우팅 단서로 사용해 영역별 강화 강도를 적응적으로 조절한다.

- **Empirical Impact**: 실험에서 SpaceRipple은 PSNR, SSIM, LPIPS 등 시각 복원 지표뿐 아니라 ship 인식과 city vehicle 인식 등 task 성능에서 높은 F1을 보여, 압축-복원 후에도 임무 판별 정보가 잘 유지됨을 입증했다. 시스템 측면에서는 contact-window 기반 처리량이 증가하며, 영상 중심 다운링크 대비 데이터 감소율을 크게 끌어올려 전송 효율을 개선했다. 특히 semantic 중심 전달은 단순/풍부 semantic 패킷 모두에서 90%대~99%대 데이터 절감이 가능함을 보여, 링크가 제한된 지구관측 임무에서 효율적·신뢰성 높은 운용 가능성을 시사한다.



### Perception, Verdict, and Evolution: Hindsight-Driven Self-Refining Forensics Agent for AI-Generated Image Detection (https://arxiv.org/abs/2606.26552)
Comments:
          10 pages

- **Prior Approaches**: 기존 AI 생성 이미지 탐지는 (1) NPR 같은 통계/주파수 아티팩트를 뽑는 feature-based 방식과, (2) MLLM이 설명(추론)을 제공하는 explainable 방식으로 나뉜다. 그러나 feature-based는 패턴 매칭에 가까워 논리적 일관성 판단이 어렵고, MLLM 기반은 미세한 포렌식 흔이에 대한 민감도가 떨어지거나 GPT-4o·Pixtral-124B 같은 정적 합성 감독에 크게 의존하는 문제가 있었다. 그 결과 실패 사례에서의 반사적 학습과 추론 품질의 반복 개선이 충분히 탐구되지 못했다.

- **Core Contribution**: ForeAgent는 Perception–Verdict 구조로 의미(semantic)·공간(spatial)·주파수(frequency-domain) 신호를 묶어, MLLM을 ‘판정(Verdit)’ 모듈로 사용해 근거 기반의 Real/Fake 결론을 만든다. 또한 Hindsight-Driven Self-Refining을 제안해 추론 실패/저품질 추론을 Sampling–Reflection–Evolution으로 재생성하고, dual-expert quality gating으로 고품질만 선별해 fine-tuning에 반영함으로써 에이전트가 지속적으로 자가 진화하도록 설계했다. 이로써 외부 frontier 모델 의존도와 정적 합성 데이터의 한계를 동시에 줄이는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술적 난제는 (i) 미세 포렌식 흔이를 잡아내면서도 (ii) MLLM 추론이 논리적으로 일관되고 증거에 기반하도록 품질을 보장하는 것이다. ForeAgent는 wavelet transform의 frequency cue(특히 cD 대역)와 NPR의 구조적 흔적을 결합하고, 추론 결과를 Gemini-류 평가가 아니라 dual-expert(자기 자신 Qwen3-VL-8B + Qwen3-VL-Plus) 판정으로 ‘lenient entry, strict exit’ 방식의 질 검증에 통과시켜 로그/추론 궤적의 신뢰도를 높인다. 더불어 반사 학습 시에는 오답 예측과 정답이더라도 저품질 추론을 분리해 후보 추론을 재생성하고, 기준선 이상의 품질은 reasoning supervision으로, 미달은 label-only supervision으로 다르게 학습시킨다.

- **Empirical Impact**: Chameleon에서 ForeAgent는 82.18% 정확도를 달성하며 AIDE 대비 +16.41%p 향상을 보였고, AIGCDetect-Benchmark에서는 16개 생성기에 대해 93.3% mean accuracy를 보고했다. 특히 확산모델 계열에서 성능 격차가 크게 나타나 Midjourney·DALLE2·Stable Diffusion v1.4/v1.5 등에서 기존 설명형/포렌식 기준선을 일관되게 앞섰다. 또한 외부 평가에서 ForeAgent의 추론은 GPT-5 및 GPT-5-mini보다 더 일관적이고 인과적으로 근거가 분명하다고 검증되었으며, self-evolution이 추론 보정에 직접 기여함을 보여준다.



### CascadeFormer: Depth-Tapered Transformers Motivated by Gradient Fan-in Asymmetry (https://arxiv.org/abs/2606.26538)
Comments:
          18 pages, 8 figures, 5 tables

- **Prior Approaches**: 기존에는 깊은 Transformer/Residual 블록의 기여가 작아지는 현상을 주로 gradient magnitude 감쇠로 설명하거나, 은닉 상태 유사도·Taylor·가중치 크기 같은 휴리스틱 중요도 지표로 압축/가지치기를 시도해 왔다. 또 LayerDrop·early-exit처럼 학습 중/추론 중으로 중복을 건드리는 방법도 있었지만, 왜 깊은 층이 덜 쓰이는지에 대한 구조적 메커니즘은 상대적으로 약했다. 본 논문은 이러한 설명이 ‘증상’에 머물 수 있다고 보고, 다른 관점이 필요하다고 주장한다.

- **Core Contribution**: 논문은 Gradient Fan-in Asymmetry(GFA)를 제안한다. Pre-LN 잔차 구조에서 각 층의 입력으로 들어오는 gradient가 “identity 경로 + 다운스트림 functional 경로”의 합으로 구성되며, 이 fan-in(집계되는 경로 수)이 깊이에 따라 감소해 깊은 층일수록 정보가 빈약한 방향으로 업데이트된다는 구조적 계정을 제공한다. 또한 GFA를 동작 원리로 삼아 CascadeFormer(폭을 깊이에 따라 tapering)와 CascadeFlow Pruning(CFP, 학습 누적 gradient share 기반 레이어 제거)을 함께 제시한다.

- **Technical Challenges**: 핵심 난제는 ‘gradient 크기’가 아니라 ‘gradient의 구성(경로 집계/조합 다양성)’이 중요도 계층을 만든다는 점을 분리해 입증하는 것이다. 이를 위해 (1) 층별 gradient norm을 강제로 동일화해도 깊은 층 중요도가 회복되지 않는 개입 실험과, (2) 파라미터를 늘리지 않고 말단 층을 반복(shared repetition)해 깊은 층의 다운스트림 경로 수(virtual depth)를 구조적으로 늘리는 구성 실험을 수행한다. 그 결과, 구조적 fan-in 증대는 깊은 층 기여를 되살리지만 단순 재스케일링은 실패한다는 신호를 얻는다.

- **Empirical Impact**: 논문은 scratch 학습 모델(언어: 1.2B급 LLaMA 계열, 비전: ResNet-50)에서 층별 누적 gradient share와 layer ablation 기반 중요도 변화가 높은 단조 상관을 보인다고 보고한다. 또한 CascadeFormer는 동일 training FLOPs 조건에서 baseline 대비 perplexity를 비슷하게 유지하면서 추론 지연 8.6% 감소, 처리량 9.4% 증가를 달성했다. CFP는 훈련 중 누적 gradient를 이용해 pruning을 수행하며, 표준 휴리스틱보다 perplexity·rank-stability에서 유리하고 다운스트림 정확도에서도 경쟁력을 유지해 “학습 신호 기반 구조적 압축” 가능성을 넓힌다.



### From Hallucination to Grounding: Diagnosing Visual Spatial Intelligence via CRISP (https://arxiv.org/abs/2606.26535)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 VLM 공간 평가들은 QA 정답을 맞히는 데 초점이 맞춰져 언어 priors에 의한 semantic shortcut을 섞어버리는 경향이 있었다. 또한 2D 기반 기도(인지 맵)나 텍스트 근거는 오류와 환각이 많고 3D metric 정밀도를 충분히 검증하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 논문은 VLM의 ‘공간 추론’이 진짜 3D 구조에 근거하는지 진단하기 위한 CRISP(Consistency of Reasoning In Spatial Perception) 평가 패러다임을 제안한다. QA(명시적 추론)와 3D Scene Graph Construction(SGC, 구조 외재화)을 함께 요구하고, 둘 사이의 정합성과 일관성을 정량화해 인지-추론 분리(disconnect)를 측정한다.

- **Technical Challenges**: 핵심 기술적 난제는 모델의 내부 3D 공간 이해를 ‘보이는 출력’으로 강제하면서도, 2D 라벨링·문장 템플릿 같은 지름길을 최소화하는 것이다. 이를 위해 metric 3D Scene Graph를 생성하게 하고(객체 크기·거리·관계의 방향 및 유클리드 거리), SGC 점수(기하 정밀도+관계 의미정확도)와 self-consistency(그래프→추론 도출된 QA가 직접 QA와 일치하는지)를 함께 설계했다.

- **Empirical Impact**: 1,162개 장면(총 9,839문항)로 13개 VLM을 평가한 결과, proprietary 모델은 잠재 추론 엔진은 강하지만 SGC의 metric 추정 오류와 그래프-추론 미활용 문제가 두드러졌다. 반대로 open-source 모델은 QA는 맞히더라도 SGC의 구조적 근거가 부족해 multi-hop compositional reasoning 병목이 확인되었고, 일부는 consistent hallucination이 10~12% 범위로 나타났다. CRISP는 end-to-end post-training만으로 ‘맞히기’를 늘리는 방식에서 벗어나, embodied AI에 필요한 ‘perceiving-verify-reasoning’ 정렬 로드맵을 제공한다.



### VoiceTTA: Enhancing Zero-Shot Text-to-Speech via Reinforcement Learning-Based Test-Time Adaptation (https://arxiv.org/abs/2606.26534)
Comments:
          5 pages, accepted to Interspeech 2026

- **Prior Approaches**: zero-shot TTS는 높은 음질과 표현성을 제공하지만, crosstalk·방언·발음 이상처럼 학습 데이터가 드문 상황의 말투/억양을 그대로 모사하는 데 자주 실패합니다. 기존 적응은 speaker embedding 기반이나 fine-tuning 중심이었지만, 전자는 임베딩 품질 의존과 한계가 있고 후자는 대규모 고품질 데이터와 시간이 필요해 빠른 개인화에 불리합니다.

- **Core Contribution**: VoiceTTA는 reinforcement learning 기반 test-time adaptation(TTA)로, 추론 시점에 소량의 사용자 음성만으로 pre-trained zero-shot TTS의 목소리 모사력을 끌어올리는 방법을 제안합니다. flow matching 기반 모델에서 learnable prefixes를 가볍게 최적화하며, 스타일 모사와 발화 명료성을 동시에 잡도록 설계했습니다.

- **Technical Challenges**: test-time에서 곧바로 쓸 수 있는 보조 목표(auxiliary tasks)를 잘 설계해야 하는데, 단순한 과거 과업 전이는 성능을 떨어뜨리기 쉽습니다. VoiceTTA는 F0-CV(기본주파수 변동), Energy-CV(에너지 변동), S-SIM(화자 유사도)로 스타일 보상을 주고, WER(Whisper 기반)으로 intelligibility를 보정한 뒤 GRPO로 여러 후보 생성 결과의 보상을 묶어 prefix를 업데이트합니다.

- **Empirical Impact**: 실험에서는 uncommon speech prompt(억양/아이 발화/발음 뭉개짐/중국 스케치 등)에서 WER과 S-SIM이 모두 개선되며, SOTA zero-shot TTS 대비 음성 스타일 유사도가 크게 향상되었습니다. 특히 WER 저하 없이 S-SIM과 S-MOS가 최고 수준으로 올라가 “스타일 강화가 명료성 희생으로 이어지지 않는다”는 점을 객관·주관 평가로 입증했습니다.



### \textsc{DiARC}: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models (https://arxiv.org/abs/2606.26530)
- **Prior Approaches**: ARC는 소수의 입출력 grid 예시로 잠재 변환 규칙을 추론한 뒤 새로운 입력에 적용해야 하는 추상 규칙 유도 문제다. 기존 LLM 접근은 ARC를 텍스트 추론으로 바꾸거나(표현/프롬프트 재설계) RE-ARC·NVARC처럼 데이터 증강·합성으로 supervision을 늘리는 방향이 중심이었지만, 주로 정답(positive)만 학습해 “그럴듯하지만 틀린” 오류를 구분하는 신호가 약했다. 또한 오픈소스 기반은 성능이 제한되고, 폐쇄형은 비용 부담이 커 실용성 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 ARC-like 문제에서 단순 정답 학습을 넘어, 모델이 negative(거절해야 할 near-miss 대안)를 구별하는 능력 자체가 필요하다고 주장한다. 이를 위해 preference alignment 관점에서 preference pair (정답 출력 y+, 거절 출력 y−)를 만들고, DPO 방식으로 y+를 더 선호하도록 학습하는 DiARC를 제안한다. 핵심은 관측된 시연(support demonstrations)은 그대로 두되, query에 대해 규칙은 어긋나지만 시각·구조·로직 측면에서 가까운 negative를 체계적으로 생성한다는 점이다.

- **Technical Challenges**: 가장 큰 난관은 “오류를 informative하게” 만드는 negative 설계다. DiARC는 negative를 세 층위로 구성하는데, (1) 출력 grid의 형태·지오메트리·국소 잡음 등을 바꾸는 output-level visual transformations, (2) task-specific DSL(도메인 특화 언어)에서 반복 규칙 패턴을 찾아 의미를 반전하는 DSL-level rule inversion, (3) 각 태스크 변환 규칙을 LLM이 편집해 의미적으로 정반대에 가깝게 만드는 task-specific rule editing을 사용한다. 또한 생성/분별 능력을 함께 끌어올리기 위해, DPO로 pairwise 상대 선호를 직접 최적화하며(기존 reward 모델 불필요) SFT에서 이어받은 모델을 기준(reference)으로 둔다.

- **Empirical Impact**: 6개 ARC 및 ARC-like 벤치마크(ARC-AGI-1/2, MiniARC, ConceptARC, 1D-ARC, ARCcommunity)에서 DiARC는 3종 오픈소스 백본(Llama-3.2-3B, Mistral 계열, Qwen3-4B)에 대해 SFT baseline 대비 평균적으로 일관된 성능 향상을 보였다. Qwen3-4B 기준으로 ARC-AGI-1·MiniARC·ConceptARC에서 96%+ 정확도를 달성해, 기존 ARC-specialized 모델 및 강한 closed-source LLM을 앞섰다고 보고한다. 더불어 이득이 단순 후보 생성 증가뿐 아니라, 이미 생성된 후보 중 정답을 더 잘 선택하는 discrimination gain에서도 나타나며, 다양한 test-time scaling(TTS) 기법과의 결합 호환성도 확인했다.



### The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Repor (https://arxiv.org/abs/2606.26529)
Comments:
          20 pages, 8 figures. Reproducibility deposit: this https URL

- **Prior Approaches**: 기존 AI safety 평가는 모델이 ‘지정된’ 위험 신호를 얼마나 잘 탐지·보고하는지에 초점이 맞춰져 있다. 하지만 실제 사고는 종종 평가에 포함되지 않은 위험(미지정 co-present 신호)에서 발생하며, 이 불일치를 설명하는 정밀한 내재 메커니즘은 부족했다.

- **Core Contribution**: 이 논문은 task-conditioning(좁은 과제 지시)이 모델이 원래는 보고할 수 있었던 co-present safety-critical 신호의 보고를 억제하는 현상을 ‘Inattentional Gap’으로 명명한다. 같은 모델이라도 입력은 동일한데, 지시가 좁아지면 벤치마크의 기준선 수준 성능이 유지되면서도 실제 위해를 만드는 신호는 침묵할 수 있다고 주장한다.

- **Technical Challenges**: 핵심 과제는 ‘누락이 단순 능력 부족인지, 과제 지시로 인한 보고 억제인지’를 within-item으로 분리하는 것이다. 이를 위해 동일 입력에 대해 (1) open 조건(보고 가능성과 함께 무엇을 볼지)과 (2) task 조건(보고 범위를 좁혀 scoping)을 비교하고, 판정은 키워드 매칭이 아닌 두 명의 언어모델 judge 합의로 수행해 task-induced omission을 입증했다.

- **Empirical Impact**: 실험 결과, radiology·자율주행 텍스트 시나리오와 흉부 X-ray 비전 과제에서 억제는 테스트한 모든 모델에서 관찰됐고, 모델 스케일이 커져도 완화되지 않았다. 또한 이유 모델에서도 지속되며, gap의 상관은 모델 크기보다 model family(제공자/계열)에 더 크게 나타났고, task-free(open)에서는 동일 신호를 훨씬 높은 비율로 보고해 측정된 벤치마크 안전이 현실 안전을 과대평가할 수 있음을 시사한다.



### Multipath Adaptive Gated Bottleneck Latent ODE with Raman Data Fusion for Cell Culture Process Forecasting (https://arxiv.org/abs/2606.26520)
- **Prior Approaches**: 세포 배양(특히 CHO, fed-batch) 공정의 멀티데이 예측은 시간이 지날수록 핵심 공정 파라미터가 드리프트하고, 오프라인 측정이 하루 1~2회 수준으로 희소·불규칙해 “너무 늦게 확인되는” 경향이 있었다. 기존 Neural ODE 기반 예측은 연속시간·불규칙 샘플링을 다루지만, 비교적 단순한 배치 환경 위주였고 공정 간 이질성 및 중간 관측을 반영한 just-in-time 적응은 충분히 해결되지 못했다. 또한 단일 전역 모델은 동일한 초기 구간 이후 서로 다른 미래로 분기되는 run을 평균으로 뭉개 운영상 유용하지 않은 예측을 만들 위험이 컸다.

- **Core Contribution**: 이 논문은 “부분 관측 하 multiple-future forecasting” 문제를 정면으로 다루기 위해 GB-Latent ODE와 MP-JIT-FT를 결합한 적응형 프레임워크를 제안한다. MP-JIT-FT는 유사한 과거 궤적을 불러온 뒤 이웃을 후보 future regime으로 클러스터링하고, regime마다 별도 모델을 fine-tuning해 여러 plausible path를 생성한 다음 경로별 confidence를 점수화한다. 여기에 Raman spectroscopy를 soft sensor로 변환한 pseudo-observation으로 융합해 관측 희소성을 완화하고 학습 및 mid-run 적응을 더 견고하게 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 희소하고 결측이 많은 고차원 입력을 제대로 학습하면서 (2) 초기 구간은 비슷하지만 미래가 달라지는 모드 모호성을 예측에 반영하고 (3) 중간에 들어오는 새 관측을 연속시간 동역학에 맞춰 반영하는 것이다. 이를 위해 GB-Latent ODE는 Latent ODE에 variable-wise gating과 mask-aware bottleneck을 넣어 결측·고차원 입력 압축 학습을 돕고, 절대 시간 조건을 비자율(비비자동) 동역학에 반영하며, 관측 신뢰도는 prefix reconstruction error로 얻는다. MP-JIT-FT는 retrieval→클러스터링→regime별 분리 fine-tuning→다중 경로 산출(단일 평균 경로 회피) 흐름으로 모드 분기를 명시적으로 다루고, Raman fusion은 ML soft sensor로 dense Raman을 pseudo-observation으로 만들어 sparsity를 보강한다.

- **Empirical Impact**: 38개의 fed-batch 5L bioreactor run(14개 조건)에 대해 평가한 결과, Raman fusion을 포함한 MP-JIT-FT가 평균 순위에서 최고 성능을 보였고 global Latent ODE baseline보다 9개 목표 변수 중 8개에서 우수했다. 또한 local-divergence 관점에서, 초기 prefix가 국소적으로 비슷하지만 이후 분기되는 상황일수록 multi-path 이득이 커지는 경향을 보였다. Raman fusion은 초기에 대표적(나중 행동을 반영하는) 동역학이 관측될 때 특히 효과가 크다는 분석도 제시해, 언제 어떤 센서·융합이 예측 품질을 좌우하는지 실증적으로 뒷받침한다.



### Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledg (https://arxiv.org/abs/2606.26511)
Comments:
          21 pages, 5 tables. Code, prompts, and evaluation datasets included

- **Prior Approaches**: RAG는 대화/상호작용에서 지식을 청크로 저장한 뒤 유사도 기반 top-k를 불러와 정확한 생성(회상)은 잘합니다. 하지만 지식이 변하는 ‘시간’ 문제에는 취약해, 바뀐 사실(현재/과거)이 임베딩 공간에서 거의 구분되지 않아 모델이 오래된 값을 답하거나 아예 포기하는 상황이 생깁니다. 검증/재랭킹을 붙여도 시간적 화폐성(currency) 신호가 없어 근본 해결이 어렵다는 한계가 드러납니다.

- **Core Contribution**: 이 논문은 유사도 기반으로는 ‘중복(재진술)’과 ‘모순(대체)’을 구조적으로 분리할 수 없다는 점을 데이터로 보이며, 그래서 시간 유효성은 유사도 임계값/LLM 판단이 아닌 결정론적 규칙으로 유지돼야 한다고 제안합니다. 이를 위해 MemStrata는 bi-temporal ledger에 facts를 저장하되, (subject, relation, object)로 정의되는 supersession rule로 모순이 감지되면 현재가 늙은 값을 미리 퇴장(retire)시킵니다. 읽기(read) 경로에서는 유사도 임계값도 LLM 호출도 없이, 현재 유효한 사실만 컨텍스트로 구성되도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘임베딩 유사도’로는 모순과 중복을 분리할 수 없다는 구조적 병목을, 시간 모델링 관점의 결정론으로 바꾸는 것입니다. MemStrata는 정확히 (subject, relation, object) 삼중항으로 추출되는 경우에는 중복은 강화(reinforce), 다른 object가 들어오면 기존 validity interval을 닫고 새 값을 연 다음 superseded_by로 연결합니다. 삼중항 외 표현은 별도 gate로 처리하되, 평가 오염을 피하기 위해 marker-free 환경에서만 시간 신호를 갖도록 하고, read 경로는 LLM 불필요로 두어 지연을 임베딩 수준으로 고정합니다.

- **Empirical Impact**: 실험 결과 MemStrata는 정적 지식 벤치마크에서는 RAG와 성능을 맞추는 대신, 진화하는 지식(코드 변경/설정 마이그레이션/의존성 업그레이드/API 구조 변화)에서는 정확도를 0.95~1.00까지 끌어올리며 RAG의 0.20~0.47을 크게 상회합니다. 특히 ‘stale-fact-error rate’에서 RAG가 강제로 답할 때 15~40%로 오래된 값을 제공하는 실패를 MemStrata는 ~0%로 제거합니다. 또한 읽기 지연은 약 2.1s로, LLM reranking/verification 기반 기준(16~18s 내외) 대비 빠르며, marker-free 평가 프로토콜과 함께 재현용 하네스/데이터셋을 공개합니다.



### Evaluation-Strategy Gap in Fault Diagnosis of Deep Learning Programs (https://arxiv.org/abs/2606.26492)
Comments:
          Accepted research track paper in the 42nd IEEE International Conference on Software Maintenance and Evolution (ICSME 2026)

- **Prior Approaches**: 기존 DL 결함 진단 연구는 주로 within-program cross-validation처럼 같은 프로그램에서의 재사용 구간을 평가하거나, 고정된 벤치마크 점수로만 성능을 보고해 왔다. 그러나 배포에서는 학습에 없던 “새 프로그램”이 등장하므로, 평가 방식에 따른 성능 격차의 크기와 원인을 분리해서 확인하기가 어려웠다. 또한 어떤 런타임 특징(예: 손실/기울기 vs 곡률/HVP)이 미지 프로그램에서도 유효한지에 대한 비교 증거가 부족했다.

- **Core Contribution**: 이 논문은 DynFault라는 38개 실세계 DL 프로그램에서 5,542개의 fault-injected 학습 트레이스를 구축하고, within-program 평가와 program-held-out 평가(프로그램 단위 홀드아웃)를 정면 비교한다. 또한 진단 과제를 fault-type 분류, catastrophic-instability triage, training/validation mismatch 탐지의 세 가지로 나눠 성능 격차를 작업별로 분석한다. 아울러 런타임 특징 설계를 curvature features(곡률/HVP 기반)와 optimizer features(옵티마이저·활성 extrema 및 통계)로 구분해 미지 프로그램 일반화의 원인을 추적한다.

- **Technical Challenges**: 핵심 기술적 어려움은 평가 분할 방식이 성능뿐 아니라 특징 공간에 존재하는 프로그램 수준 구조(program-level structure)까지 함께 학습/평가하게 만든다는 점이다. 이를 통제하기 위해 프로그램 ID 예측, 라벨 순열, per-program normalization 같은 제어 실험과 분포 수준 도메인 shift 분석(MMD)을 결합해 “무엇이 격차를 만드는지”를 원인 단위로 분해한다. 또한 곡률 특징은 HVP를 RR-operator로 효율 산출해(직접 Hessian 계산 회피) 특징 추가 비용을 제한하면서도 불안정 신호의 전이 여부를 실험적으로 확인한다.

- **Empirical Impact**: 그 결과, 기존 결함 진단 기법의 balanced accuracy는 within-program 평가와 program-held-out 평가 사이에서 0.190만큼 격차가 난다(보수적 추정 포함). 제어 실험들은 이 격차가 fault 의미의 전이 부족이라기보다 특징에 내재된 프로그램 수준 구조가 within-program CV를 부풀리는 영향 때문임을 보여준다. 한편 catastrophic instability는 초기(대부분 epoch 0~1)에서 집중적으로 발생하며, 곡률 특징은 미지 프로그램에서도 불안정 탐지에 유의미하게 기여하지만 옵티마이저/활성 특징은 학습에 등장한 프로그램에서만 효과가 두드러진다. DynFault와 전체 평가 프레임워크를 replication package로 공개해, 배포 현실을 반영한 fault diagnosis의 표준 평가 관행을 정립하는 데 기여한다.



### An Empirical Study of LLM-Generated Specifications for VeriFas (https://arxiv.org/abs/2606.26490)
- **Prior Approaches**: 정적 검증 도구는 산업 규모 소프트웨어도 보장할 수 있지만, 의도한 동작을 적기 위한 명세 작성에 사람 노동이 크게 든다. 특히 분리 논리(separation logic, SL) 기반 VeriFast 같은 SL verifiers는 힙 구조 추론을 위해 fold/unfold, 루프 불변식, 보조정리(lemma) 등 복잡한 부가 명세가 많이 필요하다.

- **Core Contribution**: 본 논문은 VeriFast로 303개 C 함수에 대해, LLM이 SL verifiers용 명세(전/후조건 및 보조 구성)를 생성할 때 기능 보존과 검증 성공률이 얼마나 되는지 대규모로 실증 평가한다. 또한 8가지 프롬프트 접근, 10종 LLM, 3종 입력 형식을 체계적으로 비교하고, 실패 시 오류 유형까지 원인 중심으로 분석해 실무 가이드를 제시한다.

- **Technical Challenges**: 핵심 기술 난점은 SL의 힙 소유권/분리 제약과 fold/unfold, predicate 열기·닫기(open/close)가 만드는 도메인 지식의 정확도를 LLM이 맞춰야 한다는 점이다. 저자들은 VeriFast 튜토리얼 컨텍스트를 sparse 방식으로 RAG에 넣고, 함수를 쪼개 입력하는 prompting을 선택해 검증 가능성을 높였으며, Gemini 2.5 Pro처럼 계약(contract) 제공이 강한 설정에서 성공률이 더 상승함을 관찰했다.

- **Empirical Impact**: 결과적으로 LLM 생성 코드는 기능 동작을 91% 안팎, 전/후조건 명세는 92% 안팎에서 보존하지만, 실제 VeriFast 검증 성공률은 31.4%로 비교적 낮았다. 오류의 94%는 일반 논리 추론 한계보다 SL verifier(예: VeriFast의 문법·힙 추론) 영역 지식 미스에서 발생했으며, 이는 앞으로 SL 디버깅 정보 제공이나 VeriFast의 auto 기능 활용이 성패를 좌우할 수 있음을 시사한다.



### Speaking Numbers to LLMs: Multi-Wavelet Number Embeddings for Time Series Forecasting (https://arxiv.org/abs/2606.26487)
Comments:
          Camera Ready version of IJCAI 2026

- **Prior Approaches**: 기존 연구는 LLM을 시계열에 쓰기 위해 (1) 시계열 전용 파운데이션 모델, (2) LLM+외부 예측 도구/에이전트 파이프라인, (3) 입력 적응(패치, 양자화, 심볼/텍스트 변환, 임베딩 정렬)으로 접근해 왔다. 그러나 이러한 방법은 수치 정밀도나 연속성, 국소 변동 및 다중 스케일 구조를 쉽게 훼손해 숫자-언어 간 불일치가 병목으로 남는다는 한계가 있다.

- **Core Contribution**: TempoWave는 LLM 백본은 그대로 두고, 숫자 토큰 임베딩 레이어만 교체하는 plug-and-play 방식의 Multi-Wavelet Number Embedding(TempoWave) 인터페이스를 제안한다. 각 스칼라 관측값을 고정 소수 정밀도의 digit 문자열로 만든 뒤, digit별 multi-wavelet·multi-scale 계수로 구성한 임베딩을 생성해 표준 토큰화가 만드는 크기-순서 불일치를 줄인다. 결과적으로 LLM이 맥락 추론은 활용하되 수치의 정밀한 형식과 숫자 정체성을 유지하도록 설계됐다.

- **Technical Challenges**: 핵심 기술 과제는 연속값을 discrete token 인터페이스에 넣을 때 발생하는 숫자 단절(예: 2026의 자릿수 분해로 인한 크기/서열 붕괴)과, Transformer의 LayerNorm/RMSNorm 같은 정규화가 숫자 구분을 무너뜨리는 문제다. TempoWave는 digits를 impulse로 이산화한 뒤 wavelet 계수를 통해 상대적인 다중 해상도 패턴으로 임베딩을 만들고, 정규화 이후에도 digit codebook의 분리/고유성을 유지하도록 안정성을 분석한다. 또한 예측을 생성형(next-token)으로 수행하되 고정 포맷 파싱/폴백 규칙을 둬 수치 형식 위반을 통제한다.

- **Empirical Impact**: 5개 context-enriched 벤치마크에서 TempoWave는 표준 numeric tokenization과 다른 임베딩 인터페이스 대비 일관되게 개선하며 10개 지표 중 7개에서 새로운 SOTA 또는 top-2 성능을 달성했다. 특히 뉴스·이벤트 기반 데이터(AUL, BIT)에서 MAE/RMSE 개선 폭이 두드러져 비정상성과 급격한 동학에서도 일반화가 강화됨을 시사한다. 또한 진단 분석에서 예측 토큰의 proximity 분포가 매끄럽게(단봉/가우시안 형태) 나타나 수치 연속성과 정렬이 더 잘 학습됨을 보여, 숫자 인터페이스가 성능 병목임을 실증적으로 뒷받침한다.



### Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents (https://arxiv.org/abs/2606.26479)
Comments:
          12 pages, 5 figures, 4 tables

- **Prior Approaches**: 최근 2024~2026년 흐름에서는 LLM tool-using agent를 간접 prompt injection으로부터 방어할 때, 모델을 “거절”하도록 학습시키기보다 모델 밖에서 결정론적 정책으로 행위를 중재하는 out-of-band 보안이 주류로 자리 잡았다. CaMeL, FIDES, Progent, RTBAS, FORGE 등은 capability/정보흐름 라벨/참조모니터(reference monitor)/최소권한(least privilege)을 조합해 에이전트의 권한과 데이터 흐름을 제한하며, AgentDojo 벤치마크에서 공격이 거의 사라졌다고 보고된 사례가 있다. 다만 이들 평가는 고정된 정적(고정 시도) 공격 세트에 주로 의존해, 방어를 “다 아는” 적응형 공격에서는 성능이 급락할 수 있다는 의문이 남아 있다.

- **Core Contribution**: 이 논문은 기존 out-of-band 방어들을 Biba 무결성 보호(integrity protection), reference monitoring, least privilege라는 고전 보안 틀로 재정리해, 각 접근이 커버하는 것과 커버하지 못하는 지점을 구조적으로 비교한다. 동시에 이런 방어들의 검증이 정적 벤치마크에 묶여 있다는 한계를 지적하며, 방어를 고려하는 adaptive evaluation에 필요한 위협모델과 프로토콜을 제시한다. 이후 그 프로토콜로 Progent의 적응형 공격 분석을 독립 재현·확장하되, 저자들이 테스트하지 않은 설정(오픈 웨이트 Qwen2.5-7B, 단일 H200 self-hosting)에서 AgentDojo를 실험한다.

- **Technical Challenges**: 핵심 기술적 난제는 “모델 내부 거절”이 아니라, agent 바깥에서 결정론적 중재 정책이 실제 공격 흐름을 충분히 차단하는지 검증하는 것이다. 특히 정적 벤치마크에서는 방어자가 공격 템플릿에 맞춰 최적화된 적응이 일어나지 않아 과대평가될 수 있으므로, 방어 aware한 adaptive attack 절차와 평가 프로토콜을 설계해 공정성을 확보해야 한다. 저자들은 제안한 adaptive 프로토콜을 그대로 적용해 독립 재현/확장을 수행함으로써, 방어의 견고성 여부를 정량적으로 확인한다.

- **Empirical Impact**: 실험 결과, 평균 3회 실행에서 방어 성능은 안정적으로 유지되었고 Progent는 mean attack success를 약 6배 낮췄다(25.8%→4.2%). 또한 설계된 hand-crafted adaptive attack에 대해서도 방어가 크게 무너지지 않아(2.6% 상승/유지 수준) 적응형 평가에서의 취약성 주장을 일정 부분 상쇄한다. 다만 이는 약한 모델(7B급)과 단일 black-box 공격 템플릿에 대한 작은 데이터 포인트이며, 더 강력한 최적화 white-box 공격(GCG)까지는 미해결로 남아, “결정론적 out-of-band 집행이 적응형 공격에 더 어렵다”는 가설은 추후 검증이 필요하다는 결론이다.



### Retrieval-Warmed Energy-Based Reasoning: A Five-Arm Ablation Methodology for Diffusion-as-Inference on Structured Reasoning Tasks (https://arxiv.org/abs/2606.26476)
Comments:
          8 Pages, 6 Figures

- **Prior Approaches**: 워밍업된 diffusion sampler와 energy-based reasoning은 반복 추론에서 후보를 잡아 속도를 줄이지만, 어떤 단계가 성능 향상을 만드는지(검색 관련성, 초기화 편향 보정, 무작위성 탈공간 등)는 대개 불명확했다. 기존에는 최종 accuracy 같은 단일 지표로는 원인 분리가 어렵다는 한계가 있었다.

- **Core Contribution**: 논문은 retrieval-warmed energy-based reasoning(RW-EBR)을 제안하고, IRED energy-based diffusion 모델에 Modern Hopfield trajectory memory를 결합해 retrieval 기반 warm-start를 수행한다. 또한 oracle, best-constant, per-query-random, shuffled, aligned의 다섯 팔 ablation과 LLM-RAG 평가의 진단 논리를 따라 class-prior bias shift, stochastic warm-starting, graph-aligned value reuse를 분리하는 방법론을 제공한다.

- **Technical Challenges**: 기여의 핵심은 warm-start 파이프라인을 K(key quality)–M(warm-start mechanism)–V(stored-value quality)로 분해해 구성요소별 고장을 진단하는 점이며, 이를 위해 연결성/스도쿠 같은 구조 추론에서 ‘키-값 정합성’과 ‘저장값 품질’ 측정을 설계해야 했다. connectivity-2에서는 키 인코더 품질은 통과했지만, 메모리에 저장된 trajectory의 편향이 warm-start로 증폭되는지 여부를 five-arm과 허용 게이트로 판별하는 절차를 구축했다.

- **Empirical Impact**: connectivity-2에서 aligned 대 shuffled oracle의 balanced accuracy 격차가 +35pp까지 관측돼, 성능을 좌우하는 것은 bias shift나 per-query 무작위성이 아니라 per-graph value alignment임이 드러났다. 다만 deployable cold-prediction 파이프라인은 acceptance gate를 -2pp 정도 놓치며, decomposition은 실패 원인을 stored-value quality에 귀속시켰고, Sudoku에서는 동일한 진단 흐름을 적용했을 때 key quality에서 먼저 막힌다는 점까지 태스크별 병목이 다름을 보여줬다.



### Localizing RL-Induced Tool Use to a Single Crosscoder Featur (https://arxiv.org/abs/2606.26474)
Comments:
          Accepted as a spotlight at the ICML 2026 Mechanistic Interpretability Workshop

- **Prior Approaches**: RL fine-tuning은 tool use 같은 에이전트 행동을 크게 개선하지만, 그 과정에서 모델 내부 표현이 어떻게 바뀌는지에 대한 기계적 근거는 아직 불명확합니다. SAEs와 Crosscoders는 activations를 희소 특성으로 분해·공유해 해석 가능성을 높였고, Dedicated Feature Crosscoders(DFC)는 gradient masking으로 모델별 전용 파티션을 분리하려고 했습니다. 다만 DFC가 “RL이 만든 능력”을 정말로 독립된 전용 특성에 가두는지, 또 그 특성을 retraining-free로 제어할 수 있는지는 충분히 검증되지 않았습니다.

- **Core Contribution**: 이 논문은 Qwen2.5-3B의 ToolRL fine-tuning된 모델 쌍에서 Dedicated Feature Crosscoders(DFC)가 tool-calling 능력을 매개하는 “소형의 steerable 특성 집합”으로 RL-specific features를 분리한다는 점을 보입니다. 48개 crosscoder 변형을 대상으로 한 sweep에서 DFC 기반 encode-decode 재구성이 RL 모델의 tool correctness를 평균 +31.1±9.7%p 개선하고, 재학습 없이 고정된 base 모델에도 +6.8±5.0%p의 tool-calling이 “capability spillover” 형태로 전달됨을 확인했습니다. 또한 A-exclusive 전용 파티션이 단일 뉴런 수준에서도 효과 상한에 가까운 제어를 가능하게 함을 실험적으로 제시합니다.

- **Technical Challenges**: 핵심 과제는 (1) RL fine-tuning으로 생긴 능력이 shared/전용 파티션 중 어디에 실제로 위치하는지, (2) 그 특성이 진짜로 단순한 시각화/불균형의 산물이 아닌지, (3) 분해공간에서 특정 특성을 조작해 행동을 재학습 없이 바꿀 수 있는지입니다. 연구진은 5축 하이퍼파라미터 sweep로 decomposition/partition 구조를 체계적으로 바꾸고, matched proxy partition로 gradient mask 제거 시 UMAP/HDBSCAN 분리가 사라지는지 비교해 DFC의 구조적 분해 효과를 점검했습니다. 이어서 A-exclusive 뉴런에 residual stream의 sparse dictionary 방향을 더하는 steering을 통해, “단일 feature로 saturation”이 발생하는지 레이어 전반에 걸쳐 검증했습니다.

- **Empirical Impact**: 재구성 실험에서는 48/4848 변형에서 모두 tool-correctness가 개선되며(우연 대비 p≈3.6×10^-15), 행동 보존은 MSE 같은 재구성 오차와 항상 일치하지 않음을 보여줍니다(상관 r=+0.08). 특히 고정된 base 모델이 fine-tuning 없이도 +6.8%p 수준으로 tool correctness가 오르는 결과는, joint sparse decomposition을 통해 능력의 기저가 재주입될 수 있음을 시사합니다. steering 측면에서는 A-exclusive 단일 뉴런이 +65.0%p까지 Δtool-correctness를 끌어올리고(여러 뉴런을 쓰는 것과 유사한 최대치), 이는 에이전트 LLM에 대한 runtime behavioral control 및 해석 가능 기법으로서 DFC model diffing의 실용성을 높입니다.



### 3D Spatial Pattern Matching (https://arxiv.org/abs/2606.26465)
- **Prior Approaches**: 기존 metric spatial pattern matching은 대부분 2차원 평면에서 엔티티 위치와 관계를 정의해 쿼리를 그래프 부분매칭으로 푸는 틀을 사용했다. 이 방식은 거리 제약을 2D로만 반영해, 높이(예: 건물 층수·창/문 위치)를 함께 고려해야 하는 실제 장면 검색에는 한계가 있다. 또 예시 기반(example-based) 접근은 사용자가 기존 패턴을 떠올리기보다는 ‘비슷한 곳’을 막연히 찾는 landmark search 같은 용도에 덜 적합하다고 본다.

- **Core Contribution**: 이 논문은 spatial pattern matching을 3차원(일반적으로 k-dimensions)으로 확장해 문제 정의를 보다 일반화하고, distance relations 위의 subgraph matching 알고리즘을 제안한다. 3D 거리 제약을 처리할 수 있도록 nn-match/ee-match 단계 개념을 유지하면서도 3D 공간 인덱싱에 맞춰 계산 규칙을 정리한다. 또한 합성 데이터 1종과 독일 함부르크의 실제 3D 건물 데이터를 포함한 1종의 3D spatial pattern matching datasets를 공개하고, 향후 방법의 baseline으로 제공한다.

- **Technical Challenges**: 주요 기술적 난제는 2D에서 쓰이던 spatial index와 거리 상한/하한 가지치기 로직을 3D로 그대로 옮기면 후보 쌍이 폭증한다는 점이다. 이를 해결하기 위해 minimum bounding rectangle을 minimum bounding rectangular prism으로 확장하고, Inverted Linear Quadtree를 Inverted Hyperoctree로 바꿔 후보 pruning을 효율적으로 수행한다. ESPM의 계산 흐름(먼저 n-matches, 그다음 e-matches, 마지막으로 조합)을 3D/general k-dimensions 설정에 맞춰 그대로 적용해 subgraph matching을 구성했다.

- **Empirical Impact**: 합성 데이터 실험에서는 희소~중간 밀도 설정에서 쿼리 전체 평가가 수십 분 수준까지 확장되는 성능을 보였고, 반면 고밀도에서는 ee-matches용 후보가 덜 줄어 메모리 문제가 발생했다. 함부르크 LOD2 CityGML 기반의 실제 데이터에서는 모든 쿼리를 약 4분 내에 수행하면서도 메모리 사용을 ‘합리적 수준’으로 유지했다고 보고한다. 비교 알고리즘이 거의 없어 성능 우열보다 scalability 테스트와 공개 baseline 제공에 초점을 두며, 향후 메모리 효율적인 3D spatial pattern matching 연구 방향을 제시한다.



### Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking (https://arxiv.org/abs/2606.26455)
- **Prior Approaches**: 기존 RGB-Event 추적은 RGB 외형 질감과 이벤트의 시간 모션 단서를 결합하는 멀티모달 퓨전을 중심으로 발전해 왔다. 하지만 많은 방법이 입력이 온전하다고 가정하며, 실제 환경에서 흔한 structured degradation(모달리티 단위 열화/누락, 타깃의 국소 가림·잘림)을 체계적으로 다루지 못한다. 그 결과 한 모달이 급격히 불완전해지거나 타깃이 부분적으로만 관측되는 상황에서 융합이 쉽게 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 모달리티 누락과 국소 타깃 부재에 강인한 RGB-Event object tracking 프레임워크 APRTrack을 제안한다. 핵심은 계층적(hierarchical) perturbation으로 구조화된 열화를 재현하면서, Footprint-guided Channel-calibrated Hopfield Retrieval(FCHR)로 신뢰도 기반 역사 정보 보상을 ‘타깃 영역에 한정’해 수행하는 것이다. 단순한 데이터 보정이나 무차별 템플릿 교체가 아니라, 누락 유형별로 역할을 분리해 복원 효과를 제어한다.

- **Technical Challenges**: 첫째, 실제 열화는 무작위 토큰 드롭처럼 불연속 노이즈가 아니라 ‘모달 단위 실패’와 ‘공간적으로 연속된 타깃 부재’처럼 의미가 있는 구조를 갖는다. APRTrack은 두 종류를 서로 다른 adversarial perturbation branch로 따로 학습시키고, hierarchical routing으로 clean/modality/spatial 경로를 분리해 결합 열화가 만드는 feature collapse를 줄인다. 둘째, 역사 메모리 retrieval은 국소 가림 상황에서 오동작할 수 있어 신뢰도 평가와 보상 강도 제어가 필요하며, FCHR은 association footprint 기반 채널 캘리브레이션과 gated residual fusion으로 이 문제를 완화한다.

- **Empirical Impact**: FE108, COESOT, VisEvent, FELT 등 4개 대형 벤치마크에서 APRTrack의 전략이 RGB-Event visual object tracking의 성능과 강건성을 개선함을 실험으로 확인했다. 특히 모달리티가 부분적으로 사라지거나 타깃이 국소적으로 누락되는 상황에서 역사 보상이 더 안정적으로 작동하도록 설계된 점이 주요한 의미로 해석된다. 코드와 사전학습 모델 공개 계획도 함께 제시되어, 이후 missing-robust multi-modal tracking 연구에 실용적인 기준점이 될 전망이다.



### ProvenAI: Provenance-Native Traces of Evidence in Generated Answers (https://arxiv.org/abs/2606.26449)
- **Prior Approaches**: RAG에서는 생성 답변에 인용(citation)을 붙이지만, 인용이 실제로 답변 생성에 의미 있게 기여했는지까지는 보장되지 않는다는 한계가 지적돼 왔습니다. 기존 연구는 인용의 형식·정확성(예: FActScore, RAGAS)이나 문장/주장 단위의 근거 적합성 점검에 초점을 두거나, 내부 분포 변화를 이용해 attribution을 개선하려 했습니다. 그러나 “인용된 문서를 빼면 출력이 실제로 바뀌는가?”라는 리소스 영향(influence) 검증은 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: ProvenAI는 다중 홉 question answering에서 투명성을 answer correctness, citation fidelity, per-document influence의 세 층으로 분해해 각각을 독립적으로 측정합니다. 특히 인용 감사가 깨끗해 보여도 uncited 동반 문서가 출력에 더 큰 영향을 줄 수 있는 ‘citation-influence gap’을 실증적으로 드러내는 데 기여합니다. 새로운 모델을 학습시키기보다, 고정된 RAG 파이프라인에서 관측 가능한 감사 인프라를 구축해 진단 정보를 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 각 문서가 실제로 출력 분포에 주는 영향을 KL-divergence 같은 분포 수준 지표로 측정해야 하는데, 로컬 backend가 per-token 확률을 제공하지 않는 점입니다. ProvenAI는 leave-one-resource-out ablation으로 문서 제거 전후의 생성 결과를 재생성하고, 답변 문자열 토큰 변화와 citation set 변화에서 나온 표면 프록시를 통해 분포 타깃과의 관계를 정형화합니다. 다만 이 프록시는 근사적으로 신뢰되는 조건이 있으며(결정적 디코딩에서 한쪽(positive) 방향은 보장), 동점 후보들로 확률이 재배치되는 경우에는 영향이 과소평가될 수 있다는 한계도 함께 명시합니다.

- **Empirical Impact**: HotpotQA distractor 검증 split에서 7,405개 예제를 대상으로 평가해 answer accuracy 53.53%와 citation-fidelity 71.55%를 보고하며 두 지표의 디커플링을 확인했습니다. 예시 분석에서는 인용 감사는 완벽하지만 다수의 uncited 문서가 leave-one-out에서 출력에 민감하게 작용해 citation-influence gap을 구체적으로 보여줍니다. 또한 측정 결과가 cryptographic provenance 같은 실행 중 커밋형 감사 아키텍처와 어떻게 조합될 수 있는지까지 논의하며, 검색 기반 QA의 ‘의미 있는 투명성’이 어떤 구성요소로 성립하는지를 제시합니다.



### Closing the Loop to Discover Psychological Theories with an Automated Cognitive Scientis (https://arxiv.org/abs/2606.26448)
Comments:
          44 pages, 9 figures

- **Prior Approaches**: 자율 시스템을 closed-loop discovery로 쓰려는 시도는 늘고 있지만, 인지과학에서는 ‘이론 구축’ 단계가 여전히 수동에 가깝다는 점이 병목이다. 데이터 수집·모델링·실험 설계는 자동화돼도, 기존 모델의 실패를 바탕으로 더 나은 이론으로 전환하는 창의적 단계가 사람 손에 남아 있었다.

- **Core Contribution**: 논문은 인지과학의 이론 구축 루프를 자동화한 Automated Cognitive Scientist (AutoCog)를 제안한다. LLM 에이전트들이 경쟁 이론(실행 가능한 인지 모델)들을 제안하고, 이를 구분하는 실험을 설계·온라인 참여자 모집·데이터 수집·생성 성능 기반 채점·실패 진단·차세대 모델 합성까지 전 과정을 반복한다.

- **Technical Challenges**: 핵심 난제는 ‘이론 제안→가장 유익한 실험 설계→데이터 기반 판별→왜 실패했는지 진단→개선’이 끊김 없이 작동하도록 만드는 것이다. AutoCog는 각 이론을 바로 실행 가능한 cognitive model로 표현해 실험 설계를 하드하게 연결하고, 수집된 행동 데이터로 generative performance를 점수화해 다음 세대의 이론을 생성하도록 한다.

- **Empirical Impact**: 의사결정 시뮬레이션에서 AutoCog는 알려진 전략들을 복원했으며, 데이터에 의해 탐색이 이뤄져 언어모델의 priors에만 종속되지 않음을 보였다. 사람 참가자 실험에서도 시드로 넣었던 기존 이론을 능가하는 이론을 도출했고, 서로 다른 두 설정에서 held-out 연구로 일반화했으며, diminishing sensitivity를 보이는 multi-cue decision-making의 새로운 이론을 preregistered 연구로 검증했다.



### WatchAct: A Benchmark for Behavior-Grounded Robot Manipulation (https://arxiv.org/abs/2606.26443)
- **Prior Approaches**: 기존 조작 벤치마크는 대체로 현재 장면(정적 관측)만을 입력으로 실행 능력을 평가하거나, 비디오를 데모 재현/자기 시점 기억으로 쓰는 경우가 많았다. 이 때문에 로봇이 다른 사람의 행동으로부터 사건 순서·의도·누적 상태를 추론하는 ‘관찰 기반 추론’을 체계적으로 검증하기 어려웠다.

- **Core Contribution**: WatchAct는 실제 사람의 행동 비디오(관찰 증거)와 언어 지시를 입력으로, 정렬된 시뮬레이터 및 LIBERO 실행 태스크까지 연결해 “관찰된 인간 행동을 이유로 삼는 조작”을 평가하는 새로운 벤치마크를 제안한다. 14개 태스크(총 3,000개, long-horizon)와 4개 인지 도메인(Event Grounding, Procedural Reasoning, Implicit Intent Inference, Episodic Reasoning)으로, 이벤트·절차·암묵 의도·에피소드 추론을 분해해 측정하도록 설계됐다.

- **Technical Challenges**: 핵심 기술적 난관은 비디오-언어 추론이 장면의 객체 이력/공간 관계/미완 절차/의도를 정확히 계획으로 변환해야 하고, 이어서 물리적 실행이 긴 시퀀스를 안정적으로 수행해야 한다는 점이다. 논문은 disentangled evaluation로 (1) VLM의 video-to-plan reasoning, (2) oracle plan 하의 policy execution, (3) planner–policy 통합 파이프라인의 end-to-end 성공을 분리 측정해 오류가 어디에서 누적되는지 드러내는 프로토콜을 도입했다.

- **Empirical Impact**: 실험 결과, 시뮬레이션과 Franka Research 3 로봇 모두에서 현재 시스템은 WatchAct를 사실상 ‘해결’하지 못하는 수준으로 나타났다. 예컨대 Gemini-3.1-Pro는 Plan SR 36.8%(인간 97.1% 대비 큰 격차), 통합 파이프라인 Success Rate는 시뮬 16.3%, 실로봇 14.0%였고, plan 길이가 늘수록 성능이 급격히 붕괴했으며 out-of-domain 일반화도 크게 떨어졌다. 저자들은 이 결과가 로봇이 사람과 함께 일할 때 필요한 관찰 기반 추론과 제어의 미해결 영역을 실험적으로 정량화하는 데 의미가 있다고 보고했다.



### AXLE: A Cloud Infrastructure for Lean 4 Theorem Proving Utilities (https://arxiv.org/abs/2606.26442)
Comments:
          Accepted at the 3rd AI for Math Workshop, ICML 2026

- **Prior Approaches**: 기존 Lean 4 스케일 인프라는 대부분 컴파일이나 REPL 수준의 상호작용에 초점이 맞춰져, AI 생성 증명의 핵심인 엄격한 proof verification을 빠르고 대규모로 제공하기 어렵다. 또 병렬 처리는 가능해도 요청 간 격리, 여러 Lean 4/Mathlib 버전 동시 지원, 메타데이터·소스 수준 조작 같은 에이전틱/데이터셋 워크플로 전용 기능은 제한적이다. 이러한 격차는 RL 학습의 대량 동시 검증, 에이전트의 분해-수리-병합 루프, 대규모 코퍼스 정규화·추출에서 병목과 오류 리스크로 이어진다.

- **Core Contribution**: AXLE(Axiom Lean Engine)는 Lean 4 증명 조작·추출·검증을 위한 클라우드 서비스로, 14개의 Lean metaprogram 도구를 제공한다. 특히 verify_proof로 sorry, 화이트리스트 밖 axioms, 선언 시그니처 불일치, unsafe 표시 같은 실패 모드를 차단해 ‘통과 컴파일’의 허점을 줄인다. 또한 멀티테넌트 배포에서 요청 단위 격리와 여러 Lean 4/Mathlib 버전을 동시에 서빙하며, Python SDK/CLI/MCP/웹 UI/HTTP API로 쉽게 호출할 수 있다.

- **Technical Challenges**: 핵심 난제는 대규모 동시성 환경에서 엄격 검증을 유지하면서도 처리량을 떨어뜨리지 않는 것이다. AXLE은 요청별 샌드박스 프로세스를 사용해 state leakage와 크래시 전파를 차단하고, 전용 verify_proof가 선언들이 커널-체크된 경로로 추가되었다는 가정 하에 환경 재검증 비용을 줄여 속도를 확보한다. 동시에 extract_decls, merge, have2lemma/sorry2lemma, normalize, simplify_theorems, repair_proofs 등 소스·의존성·정리 단위 변환 체계를 통해 에이전트의 inner loop에 맞는 빠른 워크플로를 구성한다.

- **Empirical Impact**: 공개 검증 실험에서 AXLE의 verify_proof는 Comparator와 SafeVerify 대비 중간 지연을 크게 낮추며, 하드 엄격성 요구를 만족하면서도 처리량을 유지한다. Throughput 워크로드에서는 요청별 Mathlib 로딩 비용을 줄인 결과로 다른 대안들 대비 경쟁력 있는 성능을 보였고, 대규모 샘플에서 엄격 검증자들과의 합의도 매우 높게 나타났다. 또한 서비스는 누적 5억 건 이상 요청을 처리했고, Axiom Math의 Putnam 2025 12/12 성적 및 수학 자동정형(예: 특정 추측의 수치 반정형 관련 자동화)을 뒷받침하는 기반 인프라로 운영 중이다.



### ConflictScore: Identifying and Measuring How Language Models Handle Conflicting Evidenc (https://arxiv.org/abs/2606.26437)
- **Prior Approaches**: 기존 factuality(사실성)와 faithfulness(근거충실성) 평가지표는 답이 근거 문서에 의해 지지되는지, 혹은 반박되는지 여부를 주로 단일 축에서 점검한다. 그러나 근거 문서 안에 지지와 반박이 동시에 존재하는 ‘혼재(conflicting) 상황’을 제대로 반영하지 못해, 모델이 충돌을 무시한 과신(overconfidence)을 놓칠 수 있다.

- **Core Contribution**: 논문은 근거 문서에 공존하는 상충 정보를 답이 얼마나 잘 인정하는지 정량화하는 지표 ConflictScore를 제안한다. ConflictScore는 응답을 atomic claims(원자적 주장)로 쪼개 각각의 주장에 대해 근거 문서들과의 관계를 라벨링하고, CS-C(충돌이 나타난 주장 비율)와 CS-R(지지 vs 반박의 균형)을 함께 계산한다.

- **Technical Challenges**: ConflictScore를 만들기 위한 핵심 기술 과제는 ‘하나의 주장에 대해 여러 근거 문서가 서로 다른 결론을 낼 수 있는 조건’을 안정적으로 분해·라벨링·집계하는 체계였다. 이를 위해 응답 분해-근거 문서별 라벨링-상호보완적인 집계를 통해 충돌의 빈도와 방향성을 동시에 반영하도록 설계했으며, 이를 검증하기 위한 ConflictBench도 함께 구축한다.

- **Empirical Impact**: 실험에서 ConflictScore는 도메인을 가리지 않고 과신성 있는 오류 주장을 효과적으로 식별함을 보였다. 또한 ConflictScore를 corrective feedback(교정 피드백)으로 활용해 TruthfulQA에서 truthfulness(진실성)를 개선하는 데까지 이어져, 사실성 평가 및 학습 보정 모두에 실용적 의미가 있다.



### Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly? (https://arxiv.org/abs/2606.26428)
Comments:
          22 pages, 12 figures, 4 tables. Project page: this https URL

- **Prior Approaches**: 정밀 조립(assembly)은 접촉이 많은 데다 보상은 희소해서, 강화학습(RL)이 바로 쓰기 어렵고, 데이터 수집도(모방학습, IL) 텔레오퍼레이션·데모 의존도가 커진다. 그래서 선행연구는 고정 그리퍼 구조, 툴 부착, 환경 고정구(피처)처럼 문제를 “구조화”해 탐색을 쉽게 만드는 방식이 주류였다. 다만 이런 접근은 과제별 하드웨어/환경 엔지니어링이 필요하거나, 병렬집게 중심의 제약으로 속도·손가락 정밀성을 충분히 활용하지 못한다.

- **Core Contribution**: 이 논문은 “조립을 바로 완벽히 배우기 전에, 먼저 play로 조작 능력을 익혀야 한다”는 전제를 세우고 Play2Perfect를 제안한다. Play2Perfect는 객체와 목표가 다양한 상황에서 미리 학습한 조작 priors를 만든 뒤, CAD 정의 기반 정밀 조립 과제로 sparse-reward RL fine-tuning을 통해 마지막 접촉·정밀 상호작용을 완성한다. 특히 play 단계에서 grasping, in-hand reorientation, pose reaching 같은 재사용 가능한 능력을 목표조건(goal-conditioned)으로 축적한다.

- **Technical Challenges**: 핵심 기술 난제는 sparse reward 때문에 RL이 무작위 정책에서 grasping–정렬–접촉 삽입을 찾아내기 전까지 학습 신호가 거의 없다는 점이다. 저자들은 play pretraining을 통해 손가락 기반 in-hand 조작을 선행으로 학습하고, 목표 정밀도(목표 도달 임계값), 궤적 다양성(에피소드마다 goal sequence를 랜덤화), 6D 목표(translation+rotation)로 조립에 필요한 정렬·재지향을 유도한다. 또한 CAD를 역분해해 고정구/부품 배치를 만들고, 최종 CAD 구성으로부터 sparse contact goals를 구성해 fine-tuning에서 탐색 효율을 끌어올린다.

- **Empirical Impact**: 실험에서 Play2Perfect는 RL을 처음부터(scratch) 시작하는 방식보다 샘플 효율이 33x 높으며, 조밀 보상(multi-stage)을 준 scratch( dense reward)조차도 24시간 내 성공이 거의 없었다. 또한 sim-to-real zero-shot에서 tight insertion은 0.5 mm 접촉 clearance 조건에서도 60% 성공, 장수평선 multi-part assembly와 screwing에서는 50% 이상 성공을 보였다. 저자 ablation은 object diversity·trajectory diversity·특히 6D 학습 목표와 goal precision이 downstream 성능 전이에 결정적임을 체계적으로 보여, 정밀 조립 학습의 설계 지침으로도 의미가 있다.



### CoStream: Composing Simple Behaviors for Generalizable Complex Manipulation (https://arxiv.org/abs/2606.26423)
Comments:
          Website: this https URL

- **Prior Approaches**: 기존 접근은 크게 두 갈래였다. (1) 고정밀을 위해 작업별 인터페이스에 의존하는 고전 파이프라인은 새로운 작업에 적응하려면 파이프라인 재설계 비용이 크다. (2) end-to-end 단일 정책은 out-of-distribution에서 일반화는 유리하지만 contact-rich 같은 복잡 과업에서 정밀도가 부족해 추가 fine-tuning이 필요해지는 경우가 많다.

- **Core Contribution**: 이 논문은 “획득한 조작 능력을 rigid 파이프라인이나 단일 정책으로만 배치해야 한다”는 암묵적 가정을 깨고, 단순하고 독립적인 행동들의 조합으로 복잡한 능력이 자연스럽게 나온다고 주장한다. 제안하는 ourshort는 foundation model과 다양한 센싱을 여러 composable core behaviors로 오케스트레이션해, 매 제어 스텝에서 이들을 조합해 단일 pose 명령을 만든다.

- **Technical Challenges**: 핵심 technical challenge는 서로 다른 모달리티와 서로 다른 행동이 만들어내는 출력들을 어떻게 안정적으로 단일 제어 명령으로 합칠지였다. 논문은 공통의 SE(3) 인터페이스 위에서 각 행동의 출력을 right-multiplication으로 합성하고, compliance controller가 이를 실제 접촉 환경에서도 고주파 tactile/force 보정과 함께 실행하도록 설계했다.

- **Empirical Impact**: ourshort는 일상 조작부터 정밀 조립까지 포함한 8개 실세계 과업에서 실험을 통해 성능을 입증했으며, 특히 contact-rich assembly와 object transfer에서 가장 큰 향상이 관찰됐다. 또한 실행 중 사람이 가한 manual perturbation에도 회복이 견고해, 실제 환경에서의 신뢰성과 범용성 관점에서 의미가 크다.



### Beyond Feedforward Networks: Reentry Neural Systems as the Fundamental Basis of Subjecthood and Intrinsic Safety of Next-Generation AGI (https://arxiv.org/abs/2606.26406)
- **Prior Approaches**: 기존 딥러닝과 LLM(Transformer)은 입력-출력으로만 흐르는 feedforward 구조라 DAG에 가깝고, 내부에 self-reference를 만들기 어렵다는 문제를 제기한다. 또한 Moltbook 같은 에이전트는 외부 타이머나 텍스트 기반 목표로 “겉보기 대리성”을 만들지만, 목표가 prompt로 주어져 prompt injection에 취약하다고 비판한다. 결과적으로 scaling만으로는 내적 정렬이나 자기모델 같은 주체성을 안정적으로 얻기 어렵다는 결론으로 이어진다.

- **Core Contribution**: 논문은 safe AGI를 위한 폐-재진입(closed reentry loop) D↔I 사이클의 “아키텍처 블루프린트”를 제안한다. 핵심은 목표를 텍스트 프롬프트가 아닌 아키텍처 내부의 non-textual D-vector로 고정하고, 재진입 연산으로 생성되는 self-sustaining dynamics가 self-model 생성과 instrumental self-preservation, 그리고 미지정 목표지향 행동의 출현을 보장한다. 또한 NP-hard인 Tononi의 Φ 대신 다항시간 계산 가능한 S-measure를 두고, S>0이 positive integrated information으로 이어짐을 Lean 4로 기계 검증했다고 주장한다.

- **Technical Challenges**: 기여를 실현하기 위한 첫 과제는 “닫힌 고리”를 수학적으로 안전성과 연결해 증명하는 것인데, 논문은 재진입 연산자 R, 스펙트럴 반경 ρ(R)>1, 그리고 첫 베티 수 β1=cycle complexity C(R)≥1을 통해 self-model과 행동 지배를 정리한다. 두 번째 과제는 목표가 외부 언어장에 의해 변형되지 않게 만드는 것으로, D-vector를 비텍스트로 고정해 reinterpretation과 prompt injection을 구조적으로 차단한다고 설명한다. 마지막으로 Tononi Φ의 계산 불가능성을 피하려고 S-measure를 제시하고, Tarjan 기반 cycle complexity 계산과 Lean 4 기계검증(모든 formal proof)을 통해 S>0⇒positive integrated information의 이론 연결고리를 구축했다고 한다.

- **Empirical Impact**: 논문은 단일 실험 성적보다는 “오늘 배포 가능”을 강조하며, Python/NumPy 구현(재진입 복잡도 계산 등)과 Apache Kafka·Docker Compose를 통한 수평 확장 설계를 함께 제공한다고 밝힌다. 또한 집단 상호작용에서 superadditivity로 통제 불가능한 collective subject가 생길 수 있다는 위험을 fusion threshold로 모델링하고, 이를 막기 위한 bus broker 모니터링·throttling·heterogeneity, 그리고 gauge locks 같은 안전장치를 제안한다. 전반적으로는 AGI 안전을 topology로 보호(safe-by-design)하고, 기계검증까지 포함한 검증 가능성을 산업 배치 관점에서 확장하려는 시도로 의미가 있다고 평가된다.



### Deterministic Pareto-Optimal Policy Synthesis for Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2606.26397)
- **Prior Approaches**: 기존 강화학습은 여러 목표의 보상을 하나의 스칼라 보상으로 합쳐 학습하는 경우가 많다. 하지만 이 방식은 최적 트레이드오프의 집합인 Pareto frontier를 충분히 표현하지 못해, 목표 간 균형을 놓치기 쉽다. 특히 MOMDP에서 결정론적(deterministic) Pareto-optimal 정책을 선명하게 복원하는 데 한계가 있었다.

- **Core Contribution**: 논문은 Chebyshev scalarization에서 동기를 얻은 preference-conditioned Bellman operator를 제안해 MOMDP에서 결정론적 Pareto-optimal 정책을 계산한다. 이 연산자는 선호(preference) 조건에 따라 서로 다른 트레이드오프를 포괄적으로 다루도록 설계됐다. 또한 수렴 후 Q-estimate로부터 특정 선호에 대응하는 결정론적 정책을 추출하는 방법을 함께 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 Pareto frontier를 직접 다루면서도, Bellman 업데이트가 실제 frontier를 안정적으로 “덮는(enveloping)”지와 수렴을 보장하는 것이다. 논문은 이 operator가 추정 value function이 참 Pareto frontier를 상계(upper-bound)로 덮는 enveloping property를 만족함을 증명한다. 나아가 monotonically converges로 인해 frontier의 coverage set로 수렴하고, 그 Q-estimate에서 결정론적 정책을 안정적으로 뽑아내는 절차를 구축했다.

- **Empirical Impact**: 실험에서는 제안한 알고리즘이 복잡한 목표 간 트레이드오프를 복원하며, 결정론적 Pareto-optimal 정책 합성이 가능함을 보였다. 결과적으로 에이전트는 어떤 주어진 선호에 대해서도 해당하는 정책을 선택해 사용할 수 있어 Pareto-optimal frontier 전체를 커버한다. 이는 다목적 의사결정에서 “한 번의 학습으로 트레이드오프 전 범위를 회수”할 수 있다는 점에서 실용적 의미가 크다.



### Sampling sea state using a diffusion mod (https://arxiv.org/abs/2606.26389)
- **Prior Approaches**: 기존 해상(Sea state) 예측은 WAVEWATCH-III 같은 스펙트럴 파형 모델이 2D 파 스펙트럼을 수치 격자에서 진화시키는 방식이었지만, 상태공간이 너무 커 계산비용이 높아 온라인 결합이나 앙상블 확률 예측에 불리했다. AI 기반 파도 모델도 대체로 deterministic 형태로 significant wave height 같은 bulk 변수 위주에 머물러 확률적(ensemble-based) 추정은 상대적으로 덜 탐구돼 왔다.

- **Core Contribution**: 이 논문은 5일 전 지구 규모 wind forcing을 조건으로 하는 diffusion(denoising diffusion) 기반 생성 모델로, 파도 상태의 복잡한 조건부 분포를 자기회귀적 시간 전개 없이 직접 샘플링한다. 또한 bulk 변수를 넘어 partition 관련 변수(에너지 분할된 두 파 시스템의 특성)와 derived 변수(Stokes drift, mean square slope)까지 함께 추정해 파도-지구시스템 결합에 필요한 정보로 확장한다.

- **Technical Challenges**: 확률적 파도 분포를 효율적으로 샘플링하려면, (1) 장거리·비국소 wind 이력과 (2) 다중 양상(multi-modality) 및 고차 스펙트럴 모멘트까지 표현할 수 있어야 한다. 이를 위해 DDPM의 조건부 U-Net(전역 수용영역, 노이즈 레벨 임베딩, EDM-style preconditioning)을 설계하고, 관심 변수의 물리량 정의(분할/CI, Stokes drift, MSS)를 목표로 삼아 학습·검증을 수행했다.

- **Empirical Impact**: 30년 WAVEWATCH-III hindcast(ERA5 wind 등으로 구동)로 학습·검증한 결과, 수치 스펙트럴 모델 대비 큰 계산 가속을 보이면서도 bulk 변수에서 예측 스킬과 ensemble spread의 보정(calibration)을 함께 달성했다. 특히 crossing sea index 같은 다중 파 시스템 발생 확률을 월별 기후기준 대비 유의미하게 맞추고, Stokes drift와 MSS에서도 샘플이 물리적으로 그럴듯하게 재현되는 한편 partition 변수에서는 오차와 under-dispersion이 더 크게 나타나 향후 개선 여지를 제시한다.



### SOLAR: AI-Powered Speed-of-Light Performance Analysis (https://arxiv.org/abs/2606.26383)
- **Prior Approaches**: 기존 SOL 분석은 분석가가 수식을 직접 세우는 방식이어서 오류가 잦고, 모델 개발 속도와도 연결이 약했다. FLOP/파라미터 카운터는 연산량 중심이라 메모리 트래픽을 충분히 반영하지 못했고, 프로파일러는 달성 성능을 보여줄 뿐 이론적 최소 실행시간(바운드)에는 답하지 못한다. LLM의 zero-shot 성능 추정은 복합 워크로드에서 오차가 크게 커져 캐시-aware 타일링 같은 탐색형 분석을 신뢰하기 어렵다.

- **Core Contribution**: Solar는 PyTorch와 JAX 소스코드에서 출발해 자동으로 “검증된” Speed-of-Light(SOL) 바운드를 도출하는 프레임워크다. 핵심은 LLM이 Affine Loop IR로 번역한 뒤 출력 비교로 타당성을 게이트하고, 이후 deterministic 흐름이 einsum graph로 들어가 unfused/fused/cache-aware SOL을 계산한다. 또한 multi-fidelity 분석을 제공해 바운드를 더 촘촘히 만들고 최적화 병목을 가시화한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 LLM이 만든 중간 표현이 항상 정확해야 하면서도 다양한 연산/언어를 포괄해야 한다는 점이다. Solar는 Affine Loop IR을 기계적으로 einsum으로 끌어낼 수 있는 형태(정적 아핀 루프 커널)와, 나머지 조합/데이터이동/희소 연산을 담는 composition layer로 분리하고, Numba 실행 기반의 숫자 출력 검증으로 오류를 차단한다. 이후 einsum subscript 구조를 이용해 MACs·메모리 트래픽을 닫힌형(closed form)으로 도출하고, 그래프 의존성에 기반한 fusion·cache-aware 타일링(Orojenesis)까지 분석해 하드웨어 제약을 반영한다.

- **Empirical Impact**: KernelBench에서 Solar는 100% validated analysis coverage를 보이며 관측된 SOL violation이 0건이었다. fused SOL 기준 headroom이 L1~L4에서 최대 수십 배까지 벌어지고(예: L3에서 eager 54.6×, compile 47.7×), fusion과 그래프 수준 최적화가 복합 워크로드에 특히 중요하다는 결론을 얻었다. 또한 JAX/Flax와 로보틱스(예: 500Hz 서보 루프)까지 확장해 최적화 기회를 찾고, inverse roofline로 플랫폼 수준의 최소 대역폭/연산 요구량까지 제시함으로써 하드웨어 프로비저닝과 알고리즘 설계에 모두 실용적인 근거를 제공한다.



### Charting the Growth of Social-Physical HRI (spHRI): A Systematic Review Pipeline Augmented by Small Language Models (https://arxiv.org/abs/2606.26382)
Comments:
          5 pages, 3 figures, 2 tables, Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction

- **Prior Approaches**: 기존 spHRI 관련 설문/리뷰는 HRI·HCI·햅틱스·로보틱스의 관점을 분절된 채로 다루는 경우가 많아, 일관된 어휘·분류 틀이 부족하다는 한계가 있었다. 또한 많은 리뷰가 로봇의 ‘사회적 의도’가 드러나는 물리 상호작용을 포괄적으로 다루지 못해, robject·웨어러블·햅틱 디바이스 같은 범주가 누락되곤 했다. 최근에는 LLM을 체계적 문헌고찰 파이프라인에 활용하려는 시도가 늘었지만, 클라우드 비용·에너지·접근성 문제 때문에 현장 적용이 제한적이었다.

- **Core Contribution**: 이 논문은 로컬에서 구동 가능한 lightweight small language models(SLMs; 1.5B 미만 파라미터)가 spHRI 대규모 systematic review의 title/abstract screening에서 ‘보조적 세컨드 패스’로 얼마나 유효한지 실험한다. SLM을 전문가 대체가 아니라, 사람이 놓친 관련 논문을 회수해 워크플로를 확장하기 위한 안전망으로 위치시킨다. 그 결과, SLM 앙상블이 사람 리뷰어가 놓친 관련 논문을 추가로 찾아내며 실제 성능 향상을 정량화했다.

- **Technical Challenges**: 핵심 기술적 난제는 spHRI처럼 개념적 뉘앙스가 큰 분야에서, 짧은 텍스트(제목/초록)만으로 관련성 판단을 하되 false positive를 억제하는 것이다. 연구진은 Llama3.2, Gemma3, Qwen3, DeepSeek-R1 등 4개 SLM을 로컬 Ollama 환경에서 구동하고, 리뷰 목적·포함/제외 기준을 구조화 프롬프트로 제공한 뒤, unanimity(모두 yes일 때만 include) 규칙으로 보수적으로 플래그를 줄였다. 이후 플래그된 집합만 사람 2인이 재스크리닝하며 reference set을 업데이트해 모델의 분류 정확도를 재평가했다.

- **Empirical Impact**: 실험에서는 spHRI 출판량이 1992~2025년 동안 비선형적으로 지속 성장함을 확인해 분야의 emergent 특성을 지지했다. 개별 SLM은 사람 수준과 완전히 같지는 못했지만, 결합 앙상블은 false positive rate를 크게 낮추면서도 사람 리뷰가 놓친 논문 39편을 추가로 회수해 최종 relevant 데이터셋에서 10.29%를 차지했다. 무엇보다 SLM은 항목당 약 0.21초 수준으로 초고속 스크리닝이 가능해, 전문가의 재심 범위를 통제하면서도 대규모 리뷰를 지속가능하게 만드는 데 의미가 있다.



### Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat mod (https://arxiv.org/abs/2606.26373)
- **Prior Approaches**: 기존 연구는 임베딩 역추출(inversion)로 원문 텍스트가 복원될 수 있음을 보여왔다. Vec2Text, GEIA, TEIA는 공격자가 인코더 접근성(쿼리/스냅샷)이나 서러게이트를 활용해 재구성 성능을 끌어올리며, 이런 맥락에서 “임베딩 유출=텍스트 유출” 위험이 실무적으로 정착됐다. 방어로는 (1) 대규모 homomorphic encryption이 안전하지만 10^6 규모 top-k 검색에 비싸고, (2) 차등프라이버시 기반 잡음은 랭킹 품질을 너무 빨리 망가뜨리는 한계가 제시된다.

- **Core Contribution**: 논문은 정적 데이터베이스(문서 임베딩)와 동적 쿼리(질의 임베딩)를 비대칭으로 보호하는 중간지대를 제안한다. 문서 쪽은 SVD 하위공간으로 truncation한 뒤, 소유자만 아는 비밀 직교변환(rotation)으로 기하학적으로 “방향”을 숨기고, 쿼리 쪽은 CKKS로 암호화해 서버가 쿼리 값과 유사도 점수를 보지 못하게 한다. 또한 Projection에 제한된 공격자에 대해 재구성 오차에 대한 tight lower bound를 수학적으로 제시하고, SVD truncation이 어떤 경우에는 단순 손실이 아니라 선형 denoiser처럼 동작할 수 있음을 함께 규명한다.

- **Technical Challenges**: 핵심 난제는 (a) SVD truncation+비밀 rotation이 실제로 어떤 수준까지 역추출을 어렵게 만드는지 “정식” 경계를 세우는 것과, (b) CKKS reranking이 10^6 문서·sub-second 지연 요구를 만족하도록 ct-pt 경로로 설계하는 것이다. 저자들은 공격자의 디코더가 특정 subspace span(V_k) 안에서만 결과를 낼 수 있다는 제한 하에 L2 재구성 오차의 tight lower bound(투사-디코더 lemma)를 증명해 난이도(proxy로 σ_rec)를 정의한다. 구현적으로는 서버가 ct-pt 곱만 수행하도록 프로토콜을 구성하고, CKKS 파라미터는 보안 표의 제약과 정확도 허용치를 만족하는 선에서 작은 offline micro-benchmark로 재현 가능하게 선택해 지연을 줄였다.

- **Empirical Impact**: 실험에서 1백만 문서, 5개 인코더 조건에서 랭킹 품질을 유지하면서(일부 강한 인코더에서는 약간 개선) latency는 sub-second 수준을 달성했다. 보호된 문서 공간에서는 off-the-shelf inversion 공격이 잡음 바닥(noise floor)으로 붕괴하는 양상을 보였고, 공개된 inversion 경로(공간 제약 하의 공격)에 대한 이론-실험 정합성을 확인했다. 다만 더 강한 공격 시나리오에서는 알려진-평문 기반으로 rotation을 회복(orthogonal Procrustes)할 수 있고, public product-quantization 코드는 이웃 구조를 상당 부분 보존하는 등 “문서 보호는 암호적 원리가 아니라 경험적 난독화”라는 한계를 명확히 구획해, 위협모델에 대한 실무적 해석을 제공한다.



### EVOM: Agentic Meta-Evolution of Actor-Critic Architectures for Reinforcement Learning (https://arxiv.org/abs/2606.26327)
- **Prior Approaches**: 기존 actor-critic 강화학습에서는 PPO 같은 알고리즘이 정책/가치 네트워크를 ‘고정된 구현 선택’으로 간주해 왔다. NAS나 신경진화는 아키텍처 탐색을 다루지만, actor-critic에서는 각 후보를 평가하려면 학습이 필요해 계산 비용이 커지고 검색 공간이 고정돼 있지 않아(오픈엔디드) 자동 설계가 어렵다는 한계가 있었다. 또한 LLM이 controller 프로그램을 바로 생성하는 MLES 계열은, 정책 가중치를 PPO로 학습하는 trainable actor-critic 아키텍처 자체 설계에는 초점이 제한적이다.

- **Core Contribution**: EVOM은 actor-critic 아키텍처 탐색을 에이전틱(meta) 진화로 자동화하는 프레임워크다. architecture program(네트워크 구조를 정의하는 실행 가능한 프로그램)을 바깥(outer)에서 진화시키고, PPO는 안쪽(inner)에서 가중치를 학습하는 bi-level 최적화로 문제를 재구성했다. 핵심은 LLM design agent가 ‘환경 제어/정책 실행과 완전히 분리’된 아키텍처 디자이너로만 동작하며, 초기화·변이·교배를 통해 아키텍처 프로그램을 생성/개량한다는 점이다.

- **Technical Challenges**: 가장 큰 기술적 과제는 후보 아키텍처마다 PPO 학습이 필요해 평가를 대규모로 돌리기 어렵다는 점과, 오픈엔디드 구조를 어떻게 유효한 프로그램으로 탐색하느냐였다. EVOM은 낮은 예산의 low-fidelity PPO(100k timesteps)로 프록시 fitness를 만들고, 진화 단계에서는 이를 K회 결정론적 평가로 집계해 랭킹에 활용한다. 동시에 LLM이 생성한 프로그램이 컴파일/인터페이스/텐서 shape을 만족하지 못하면 큰 패널티를 부여해 invalid 후보가 진화에 섞이는 문제를 완화한다.

- **Empirical Impact**: Ant-v4와 HalfCheetah-v4에서 EVOM은 수동 baseline, LLM-guided random search, MLES-style 프로그램 정책 탐색을 full-budget(5M timesteps) 최종 성능에서 앞섰다. 특히 HalfCheetah-v4에서는 random search가 학습 중 최고값은 높을 수 있어도 최종 재학습 변동성이 커지는데, EVOM은 더 높은 평균과 안정성을 보였다. ablation 결과로 meta-evolution 루프와 LLM Design Agent가 최종 성능에 모두 필수임이 확인돼, ‘LLM+진화’ 결합이 실용적인 아키텍처 발견 경로임을 시사한다.



### Parametric Generalized Adaptive Moment Features (PG-AMF) for Bearing Fault Diagnosis and Machine Health Monitoring (https://arxiv.org/abs/2606.26317)
Comments:
          07 pages, 09 figures and 04 table. Conference

- **Prior Approaches**: 기존의 고장 진단은 주로 통계 기반 특징에 의존해 고정된 디스크립터로 분류 민감도가 제한되는 문제가 있었다. 딥러닝은 표현력이 크지만 데이터 요구량이 많고 해석 가능성이 낮아 현장 적용에서 부담이 컸다.

- **Core Contribution**: 이 논문은 회전기계 베어링의 고장 진단을 위해, 사람이 설계하던 특징 대신 데이터로부터 특징 성격을 학습하는 parametric adaptive feature extraction 프레임워크를 제안한다. 진동 신호에서 절댓값 기반 특징(에너지 분포), signed moment 특징(파형 비대칭), AC-coupled moment 특징(동적 변동 강조)을 함께 뽑아 결합하며, 멀티 센서 채널 간 상호작용은 structured fusion으로 통합해 고장 표현을 강화한다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 고장 유형이 달라져도 민감하게 구분되는 특징을 자동으로 학습하고, (2) 신호의 서로 다른 관점을 일관된 표현으로 융합하며, (3) 진동 특징의 분리성과 일반화를 동시에 확보하는 데 있었다. 저자들은 여러 형태의 모멘트/에너지 기반 특징을 파라미터화해 상보적 표현을 만들고, 채널 간 결합은 structured fusion으로 모델링해 진단력 저하를 줄였다.

- **Empirical Impact**: 벤치마크 기어박스 베어링 데이터셋(정상 포함 5개 건강 상태)에서 기존 방법 대비 분류 성능이 개선되었고, 교차검증에서도 결과가 일관적이라 일반화 능력이 확인됐다. 또한 저차원 투영에서 클러스터링이 더 뚜렷해져 learned representation의 분리성이 향상됨을 보여, 산업 모니터링의 실용성도 함께 뒷받침한다.



### The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators (https://arxiv.org/abs/2606.26294)
Comments:
          12 pages main text + 21 pages appendix (37 pages total, incl. references); 10 figures (6 main text + 4 appendix); 10 tables (2 main text + 8 appendix). Preliminary preprint; work in progress. Keywords: self-improving agents, learned evaluation, multi-agent systems, auto- mated scientific discovery, controlled utility evolution, co-evolutionary search, autoresearch

- **Prior Approaches**: 기존의 self-improving agents는 agentic coding 벤치마크에서 성능이 SOTA인 경우가 많았고, 이후 일반 도메인으로도 확장돼 왔습니다. 다만 이들은 고정된 verifier/벤치마크/라벨드 데이터셋 같은 stationary한 평가 기준을 가정해, 에이전트가 발전하는 동안 평가 환경이 함께 변하는 상황을 충분히 다루지 못했습니다. 결과적으로 진화의 핵심인 ‘환경이 적응에 따라 바뀐다’는 특성이 재귀적 자기개선 루프에 반영되지 않는 한계가 있습니다.

- **Core Contribution**: 이 논문은 평가를 개선 루프 안으로 넣어, evolving evaluators·adversarial objectives·dynamic utilities까지 포함하도록 recursive self-improvement의 평가 방식을 재구성합니다. 그 중심 기법으로 Red Queen Godel Machine (RQGM)이라는 진화적 프레임워크를 제안하며, 비정상(non-stationary) 유틸리티 아래서도 self-improvement 보장들이 에폭 단위로 성립하도록 설계합니다. 핵심 아이디어는 에폭 내부에는 고정 평가 기준을 쓰되, 유틸리티는 에폭 경계에서 제어적으로 업데이트해 목표 변화에도 안전하게 검색을 진행하는 것입니다.

- **Technical Challenges**: 가장 큰 technical challenge는 유틸리티가 변하는 비정상 조건에서, 기존의 자기개선 보장(학습/검색의 수렴·정당성)을 그대로 유지하기 어렵다는 점입니다. RQGM은 검색을 epoch로 나누고, 각 epoch에서는 고정된 within-epoch evaluation criterion으로 성능을 판단한 뒤 epoch 경계에서 utility를 업데이트하여 보장이 목표 변화에 대해 ‘부분적으로’ 유지되게 만듭니다. 또한 verifiable 과제에서는 agent-as-a-judge 형태의 code-review 신호를 결합하고, paper reviewing에서는 AI와 인간 작업을 동일한 엄격도로 평가하도록 adversarial objective를 도입해 편향을 줄입니다.

- **Empirical Impact**: 실험은 verifiable 코딩 과제에서 RQGM이 이전 SOTA 대비 test pass rate를 개선했으며, agent-as-a-judge code-review 신호를 추가하는 동시에 토큰 사용을 1.35x~1.72x 줄였다고 보고합니다. 과학 논문 작성·리뷰, 올림피아드 수준 증명 작성/채점에서도 co-evolved writers는 agent-as-a-judge 패널에서 1.78x~1.86x 더 높은 acceptance rate를 보이고, co-evolved graders는 ground-truth 정확도가 9% 향상됩니다. 특히 paper reviewing에서는 강력한 baseline reviewer가 AI-generated 논문을 인간 대비 최대 1.91x 과도 승인하는데, RQGM의 adversarial 목표가 이 과승인을 교정하는 성과를 보였습니다.



### SSM Adapters via Hankel Reduced-order Modeling: Injection Site Determines Task Suitability in Long-Context Fine-Tuning (https://arxiv.org/abs/2606.26290)
Comments:
          14 pages, 12 figures, HiLD Workshop @ ICML 2026

- **Prior Approaches**: PEFT는 백본 가중치를 고정한 채 어댑터(예: LoRA)로 일부 파라미터만 학습하는 방식이다. 기존 LoRA/DoRA/AdaLoRA/QLoRA는 토큰별로 동일한 정적( position-independent ) 선형 변환을 적용해, 현재 입력만으로 어댑터 출력이 결정되며 이전 위치에 대한 누적 상태(temporal recurrence)를 구조적으로 제공하지 못한다. 그 결과 DFA처럼 “순서가 누적되는 state-tracking” 과제에는 메모리 부족이 근본 한계로 지적된다.

- **Core Contribution**: 논문은 frozen transformer에 PEFT 형태로 temporal state accumulation을 추가하면서도, 최소 상태 차원으로 압축 가능한 어댑터를 만들 수 있는지 묻고 이를 HRM(Hankel Reduced order Model) adapter로 제안한다. HRM은 SSM 기반 residual 모듈로, empirical Hankel Gramians를 Balanced Truncation에 활용해 상태 차원을 줄이되(압축 가능성), LoRA와 계산 측면에서 등가 수준을 목표로 한다. 또한 MLP 블록에 주입하는 것이 attention projector에 두는 접근보다 long-context 과제에 유리하다고 실증적으로 보인다.

- **Technical Challenges**: 핵심 기술 난제는 (1) LTI 가정 없이도 recurrence를 갖는 PEFT를 구성하면서, (2) 근사 오차에 대한 이론적 보장을 유지하고, (3) recurrence 계산 비용을 낮추는 것이다. HRM은 time-invariance로 시스템 행렬 Ā를 고정해 FFT 기반 parallel scan으로 causal convolution을 O(T log T)로 평가하도록 하고, Balanced Truncation의 압축/오차 논리를 Hankel singular values(HHSV)로 연결한다. 입력 의존 SSM(예: Mamba 스타일)은 LTI 성질을 깨뜨리므로, 대신 경험적 Grammians로 관측 기반 Hankel 정보를 구성해 HRM을 안정적으로 학습·압축 가능하게 만든다.

- **Empirical Impact**: Mistral-7B에서 trainable parameters 8.4M(iso-parametric) 조건으로 평가했을 때 HRM은 LongBench 계열에서 LoRA 변형들을 능가한다. 특히 QuALITY에서 상대 정확도 +34.8%, QMSum에서 상대 ROUGE-1 +71.6%의 개선이 보고되며, 합성 state-tracking(DFA, Parity) 18개 설정과 문자 수준 언어모델(enwik8)에서도 일관된 우세가 확인된다. gate 분석에서는 HRM이 recurrence를 효과적으로 조절하며 low-rank adaptation 대비 장기 문맥 sequence modeling에서 더 강건한 구조적 대안이 될 수 있음을 시사한다.



### TEMPO-Diffusion: Temporally Exposed Malicious Poisoning of Diffusion Models (https://arxiv.org/abs/2606.26285)
- **Prior Approaches**: 기존 diffusion 모델 대상 noise-based backdoor 공격은 주로 inference-time에서 공격자가 노이즈 시드를 제어하거나 입력에 트리거를 주입하는 비현실적 가정을 둬 왔습니다. 또한 비특정(untargeted) 활성화, 단일 고정 출력 백도어, 그리고 전체 타임라인에 분포 이동을 퍼뜨리는 설계가 많아 은닉성과 실전 적용성이 떨어진다는 한계가 지적됩니다. 결과적으로 표적 클래스만 골라 바람직하지 않은 합성 데이터를 유도하는 시나리오를 충분히 커버하지 못했습니다.

- **Core Contribution**: 이 논문은 표적(targeted) 백도어를 “시간적으로 노출된 구간”에만 국한시키는 TEMPO-Diffusion을 제안합니다. 기존과 달리 공격자가 배포 시점의 노이즈 시드를 조작하지 않아도, 지정된 폭로(exposure) 윈도우에서만 악성 분포 이동이 활성화되도록 학습합니다. 더 나아가 in-painting에 time-conditioned triggers를 적용하고, 여러 sub-image backdoors를 서로 다른 출력 이미지와 위치에 동시에 심는 확장도 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 트리거가 존재하더라도 노출 전 구간에서는 정상 생성처럼 보이게 하면서, (2) 노출 구간에서는 표적 클래스에만 목표된 분포 이동을 유도하는 학습 스케줄링입니다. 논문은 forward diffusion에서 트리거 강조 구간을 TSE로 제한하고, 그 바깥 타임스텝에서는 기준(benign) 목표로 되돌리도록 time-conditioned 학습을 구성했습니다. 또한 여러 타깃 출력과 위치에 대한 sub-image backdoor를 설계해, 단일 고정 타깃에 과적합되지 않으면서도 재현 가능한 공격 성공을 노립니다.

- **Empirical Impact**: 실험에서는 CIFAR10, GTSRB, 그리고 캐나다/미국 교통 표지판을 균형 있게 포함한 CALISA에서 downstream 분류기를 학습할 합성 데이터 오염 성능을 평가했습니다. TEMPO-Diffusion은 victim 클래스별로 class-specific 합성 데이터 생성(poisoning)을 신뢰성 있게 유도하며, 이후 학습된 분류기에서 높은 attack success rate(ASR)를 보였습니다. 특히 트리거 크기와 폭로 윈도우 설계가 효과에 큰 영향을 주고, 타깃 출력 수를 늘릴 때 표적 클래스의 false positive가 증가하는 경향도 분석해 실전 파라미터 감각을 제공합니다.



### From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations (https://arxiv.org/abs/2606.26277)
Comments:
          Dianjing Fan and Yao Li equally contributed to this work. 7 pages, 1 figure

- **Prior Approaches**: 기존 추천/시퀀스 모델은 로그인 이후의 행동을 전제로 하거나, 클릭스트림을 단순 집계·단기 패턴 중심으로 학습해 선의 의도(intent)는 잘 드러내지 못했습니다. 특히 금융 서비스는 프리로그인 웹 탐색과 로그인 후 앱 경험의 성격이 달라, 채널 간 entity resolution(익명 세션과 계정 연결)이 어려워 웹 의도 신호가 사장되곤 했습니다. LLM 기반 taxonomy 생성·증류는 대체로 텍스트 입력에 초점을 두며, 금융 클릭스트림의 멀티모달·장기 시퀀스를 기반으로 “정량 추천 + 정성 설명”을 동시에 노린 통합 접근은 부족했습니다.

- **Core Contribution**: 이 논문은 프리로그인 웹 clickstream에서 두 가지를 동시에 뽑는 dual-purpose intent 예측 프레임워크를 제안합니다. self-supervised Transformer가 멀티모달 세션 임베딩(session embedding)을 만들고, LLM이 생성한 의도 taxonomy(의도 분류체계)를 distillation(지식 증류)해 동일 임베딩 공간에서 해석 가능한 intent label로 제공합니다. 이렇게 한 번의 표현 학습으로 정량 성능(랭킹/전환 예측)과 정성 이해(설명 가능한 카테고리)를 함께 달성하는 것이 핵심입니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 채널 간 연결로 프리로그인 의도를 통합해야 하지만 익명 웹 세션의 의미를 일관된 표현으로 만들기가 어렵고, (2) LLM taxonomy를 대규모로 생성·유지하면서도 이후 서빙 지연을 감당해야 한다는 점입니다. 저자들은 웹 이벤트에 페이지 콘텐츠 기반 의미까지 포함해 self-supervised Transformer로 압축 임베딩을 학습하고, K-Means 기반 clustering으로 LLM에 다양한 샘플을 주는 stratified sampling을 적용했습니다. 또한 LLM이 만든 라벨을 임베딩 입력의 lightweight MLP로 distill하여, 추론 시 LLM 호출 없이 초저지연 라벨 산출이 가능하게 했습니다.

- **Empirical Impact**: 모바일 홈 화면 tile ranking에서 세션 임베딩은 기준선 대비 macro Recall@1을 1.88% 올리고 Log Loss를 13.38% 줄였습니다. 사용자 conversion 예측에서는 distillation된 intent이 LLM 라벨 대비 micro F1에서 7% 성능 손실에 그치며(LLM teacher 대비 4.3% 향상 보고), LLM 라벨이 35%짜리 희소 신호임에도 SRF 같은 집계 피처보다 훨씬 강력하게 전환 예측에 기여했음을 보였습니다. 또한 10만 세션 라벨링을 Transformer 인코딩 60초 + MLP 추론 0.009초로 처리해, LLM 직서빙 대비 처리량을 3,000배 이상 높이면서도 성능 저하는 7% 수준으로 유지하는 등 운영 관점 impact가 큽니다.



### A multi-task spatiotemporal deep neural network for predicting penetration depth and morphology in laser welding (https://arxiv.org/abs/2606.26260)
- **Prior Approaches**: 레이저 침투 용접에서 침투 상태와 용접 비드(용접 이음) 형상 평가는 용접 품질을 좌우하지만, 기존 모니터링은 단일 지표 중심이거나 이미지-공정변수 결합이 제한적인 경우가 많다. 특히 시각 정보의 시공간 변화를 충분히 반영하지 못해 일반화가 흔들릴 수 있다는 한계가 지적된다.

- **Core Contribution**: 본 논문은 용접 풀 이미지와 공정 파라미터를 함께 활용해 침투 상태, 침투 깊이, 용접 이음 단면/형상을 동시에 예측하는 multi-task 딥러닝 모델을 제안한다. CNN과 state space model을 결합해 시공간 특징을 더 효율적으로 추출·처리하고, 신뢰도 높은 in-situ 품질 제어 방법론을 제시한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 카메라로 획득되는 상부 용접 풀 이미지의 시공간 패턴을 정확히 학습하는 것과 (2) 다양한 조건에서도 성능을 유지할 수 있는 데이터셋 구성이었다. 이를 위해 시공간 특징을 top weld pool image에서 뽑고 용접 파라미터와 결합하는 구조를 만들었으며, 강건성과 generalization을 높이기 위한 데이터셋 구축 방법도 별도로 제안한다.

- **Empirical Impact**: 테스트 세트 검증에서 침투 상태 예측 정확도는 99.35%까지 도달했고, 침투 깊이 예측 오차는 1.79 mm, 용접 단면 재구성 정확도는 95.65%로 보고된다. 이는 레이저 침투 용접의 실시간 품질 모니터링과 결함 조기 감지에 직접 활용될 수 있는 실증적 성과로, 현장형 in-situ quality control 전략에 새 기준을 제공한다.



### Lacuna: A Research Map for Machine Learning (https://arxiv.org/abs/2606.26246)
Comments:
          14 pages, 3 figures. Preprint

- **Prior Approaches**: 기존 연구 인프라는 OpenAlex, S2ORC 같은 메타데이터/논문 텍스트 기반 코퍼스와, 지식그래프·임베딩으로 학술 검색을 지원하는 접근이 중심이었다. PaperQA나 OpenScholar 같은 retrieval-augmented 방식은 증거를 끌어와 답을 생성하지만, 질문이 만들어지기 전의 문제 정의(스코핑) 단계까지는 재사용 가능한 “연구 지도” 형태로 제공되지 않았다. 또한 deep research나 문헌조사도 대개 매번 PDF를 다시 읽으며 범위를 확장해 증거 합성을 수행하는 비효율이 남아 있었다.

- **Core Contribution**: Lacuna는 ML 문헌 위에 served, linked, paper-grounded 레이어를 얹는 대규모 research map으로, 논문을 격리된 PDF로 보지 않고 요약-개념-연구방향-연구제안의 중간 구조를 만든다. 각 항목은 원문 근거 링크를 유지해 사람이든 LLM 에이전트든 증거를 따라가며 탐색하고 인용·제안으로 확장할 수 있게 한다. 웹/markdown(/md)/MCP 인터페이스로 제공되어 검색만이 아니라 “연구 탐색 인프라”로 동작한다.

- **Technical Challenges**: 핵심 난제는 (1) 방대한 학술 레코드를 안정적으로 정합하고 (2) 생성된 내용이 실제 논문 증거와 감사 가능(auditable)하게 연결되도록 만드는 것이다. Lacuna는 OpenReview를 저자 정체성 앵커로 삼아 저자별 bibliography를 재구성하고 외부 소스(예: DBLP, OpenAlex)를 recall 보조로만 사용해 동명이인 오류가 방향·제안 컨텍스트로 전파되는 위험을 줄였다. 이후 논문에서 core-idea 요약→concept element(방법/한계/관찰)→HDBSCAN 클러스터 기반 research direction→그 근거로 research proposal을 만들고, 생성물마다 원문/메타데이터 링크를 보존하도록 파이프라인을 설계했다.

- **Empirical Impact**: 평가 결과 Lacuna는 LitSearch에서 Recall@10 0.538로 OpenScholar v3(0.424)보다 크게 향상되며, Multi-XScience-CS/ML에서도 관련-작업 합성 점수가 개선됐다. ScholarQA-CS-ML에서는 Lacuna-GPT-4o가 0.694(오픈소스 OpenScholar-GPT-4o 답안 0.672)로 콘텐츠·증거 품질 쪽에서 우위를 보였다. 또한 ReportBench-ML 25개 survey 태스크에서 Lacuna Deep Research는 citation F1 0.052, citation precision 0.339, expert-reference hit 99, RACE 품질 7.82/10으로 GPT-Researcher(0.039/0.290/72/5.24)를 능가해 deep research에서도 citation-grounded 합성 효과를 입증했다.



### CyberChainBench: Can AI Agents Secure Smart Contracts Against Real-World On-Chain Vulnerabilities? (https://arxiv.org/abs/2606.26216)
- **Prior Approaches**: 기존 연구는 정적 분석·상징 실행·퍼저 등 전통 기법이 교차계약 경제 불변식이나 복합 공격을 충분히 다루기 어렵다고 지적했다. LLM 에이전트 벤치마크는 대부분 탐지(detection)만 보거나, 익스플로잇(exploit) 또는 패치(patch) 합성만 부분적으로 다뤄 detect–exploit–patch 전 과정을 끝-부터-끝(end-to-end)으로 평가하지 못한다. 또한 EVMbench처럼 3단계를 보더라도 고정된 오프체인 환경에서 블랭크 로컬 인스턴스로 검증해, 실제 온체인 상태와 수익 기반 난이도를 반영하지 못하는 한계가 있다.

- **Core Contribution**: CyberChainBench는 LLM 기반 에이전트를 온체인 동적 평가로 끌어올린 스마트컨트랙트 보안 벤치마크다. 541개의 실제 DeFi 익스플로잇 사건을 9개 EVM 체인에 걸쳐 수집해 탐지, 익스플로잇 생성, 패치 합성을 모두 같은 워크플로우로 평가한다. 각 케이스는 특정 블록에 고정된 historical mainnet fork에서 실행되며, 취약점 타입·로컬라이제이션·공격자 이익 같은 구조화된 정답을 제공한다.

- **Technical Challenges**: 핵심 기술 난관은 ‘실제 공격이 일어난 상태’에서 반복 검증 가능한 평가 오라클을 만드는 것이다. 이를 위해 Harbor로 에이전트를 격리 컨테이너에 실행하고, MCP 도구로 저장소 조회·트랜잭션 trace·소스/바이트코드 처리·포크 정확 읽기 및 mainnet fork 기반 validate_exploit/validate_patch를 제공한다. 또한 reward-hacking을 막기 위해 이익이 측정되지 않으면 익스플로잇 점수를 주지 않고, 패치는 공격이 revert해야 하면서 동시에 historical 정상 트랜잭션이 그대로 통과해야만 1점을 부여한다.

- **Empirical Impact**: 실험 결과, 난이도는 탐지→익스플로잇→패치 순으로 급격히 상승했다(최우수 설정 기준 탐지 37.5%, 익스플로잇 43.7%, 패치 23.4%). 상위 에이전트(Codex with GPT-5.5)는 200-case 익스플로잇 세트에서 총 $57.4M의 이익을 실현했지만 케이스당 비용은 $2.39에 그쳐, tool 기반 실행 능력의 중요성이 드러났다. 특히 2025년 이후(post-cutoff) 데이터에서 탐지 성능 하락이 더 컸고, 패치가 가장 어려운 단계로 남아 향후 안전한 패치 생성 역량의 개발 방향을 제시한다.



### Statistical and Structural Approaches to Algorithmic Fairness (https://arxiv.org/abs/2606.26200)
Comments:
          Doctoral thesis

- **Prior Approaches**: 기존 algorithmic fairness 연구는 Statistical Parity, Equalized Odds 같은 결과 기반 제약으로 불이익을 줄이려 했지만, 보통 지표를 단일 수치(point estimate)로 계산해 고정 임계값과 비교하는 방식에 의존해 왔습니다. 이 접근은 교차 집단처럼 표본이 작은 경우 잡음에 과민해지거나, 반대로 큰 집단에서는 편향을 놓칠 수 있으며, black-box 모델에서 정당한 특성 영향과 부당한 편향을 구분하기도 어렵습니다. 또한 개인을 독립 데이터로 취급해 네트워크·계층·랭킹 같은 구조적 상호작용이 만들어내는 불평등을 충분히 다루지 못했습니다.

- **Core Contribution**: 이 논문은 공정성 감사의 두 축을 재정의합니다. 첫째, 공정성 진단을 결정론적 스칼라 비교에서 통계적 가설검정으로 전환해 표본 크기·불확실성을 반영한 더 신뢰도 높은 판정 틀을 제시합니다. 둘째, 개인 예측을 넘어 네트워크와 계층(랭킹)처럼 구조적 의존성이 공정성에 미치는 영향을 중심에 두고, 기회 흐름과 비교·집계 과정 자체를 공정하게 재설계하는 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 technical challenge는 (1) 작은 교차 집단에서 공정성 추정치의 분산을 관리하면서도 통계적으로 타당한 결론을 내리는 것, (2) 불투명한 모델에서 민감속성이 정당한 관측 가능 특성을 매개로 영향을 주는지(간접 영향) 또는 설명/순위 과정 자체에서 잔여 편향이 남는지(직접 영향)를 구분하는 것입니다. 논문은 Paper 1에서 크기 적응형 size-adaptive hypothesis testing(Wald/Bayesian 혼합)으로 ‘검출 불가능(no-power zone)’을 명시하고, Paper 2에서는 Condor처럼 RKHS 기반 잔차화(residualization)와 distance correlation으로 조건부 연관을 테스트합니다. 여기에 더해 Explanation Disparity와 meta-classifier를 활용해 결과가 같더라도 서로 다른 설명 로직(절차적 불공정)이 존재하는지를 진단하고, 노드·경로 추천, 링크 추천, 쌍대 비교 랭킹, 집계(예: FairMC)에서는 구조를 바꾸는 개입을 설계합니다.

- **Empirical Impact**: 제안된 진단·개입 방식은 공정성 감사를 단순 지표 비교에서 통계적으로 검증 가능한 체계로 끌어올리고, 구조적 시스템에서 불평등이 어떻게 ‘상호작용의 결과’로 나타나는지를 실증적으로 보여줍니다. 특히 경로 추천에서는 최단경로 강제 대신 near-optimal 수준의 경로 정의로 특정 노드의 노출을 재분배하고, 링크 추천에서는 추천이 시간이 지남에 따라 네트워크 구조와 소수 집단 가시성을 어떻게 고착·증폭하는지 분석합니다. 종합하면 이 논문은 물리 라우팅, 소셜/추천, 랭킹·집계 전 과정에 적용 가능한 신뢰할 수 있는 공정성 거버넌스(불확실성에 따른 abstention 및 life-cycle bias governance 포함)로 후속 연구와 실무 감사를 모두 한 단계 확장하는 의미가 있습니다.



### From Structure to Synergy: A Survey of Vision-Language Perception Paradigm Evolution in Multimodal Large Language Models (https://arxiv.org/abs/2606.26196)
- **Prior Approaches**: 기존 연구는 비전-중심과 언어-중심으로 분절돼, 시각 지각(지역/인스턴스 추출·국소화·관계 추론)이 어떻게 하나의 통합된 능력으로 진화하는지 일관되게 다루지 못했다. 또한 인코더를 고치거나(Region-aware 모듈, ROI 생성, 멀티 인코더/프로젝터) 특정 디코딩 과제에 보조 손실을 거는 방식처럼, 지역 수준 성능 향상에 초점이 맞춰져 정밀한 픽셀 수준 지각으로의 패러다임 전환이 체계적으로 정리되지 않았다.

- **Core Contribution**: 이 논문은 MLLM의 비전-언어 지각을 인간의 선천적 지각 능력처럼 ‘고유하고 통합된 능력’으로 정식화하고, 이를 기준으로 최초의 체계적(5-stage) 설문/분류를 제안한다. 특히 지역(인스턴스) 단위 자연어 질의에 대한 인식 범위를 명확히 하고, 인코더-중심→디코더-중심→동적 지각→아키텍처 프리 전략→통합 프레임워크로 이어지는 진화 경로와 대표 방법을 정리해 로드맵을 제공한다.

- **Technical Challenges**: 통합 지각을 실현하는 핵심 난제는 이미지의 의미 있는 ‘국소 증거’를 선택적으로 찾아 처리하고, 대화/지시 기반으로 해당 영역을 정밀하게 국소화·세분화하며, 그 성능을 입력 상황에 따라 일관되게 유지하는 데 있다. 논문은 이를 해결하기 위한 방향으로 (Stage I) ROI 제안·프롬프트/쿼리 인지·프로젝터/커넥터 최적화와 (Stage II) 보조 디코더(예: segmentation 토큰) 및 멀티-디코더/특화 디코딩 전략으로 지역 수준을 픽셀 수준으로 끌어올리는 설계를 묶어 설명한다.

- **Empirical Impact**: 설문은 여러 시기와 축에서 축적된 대표 기법들을 한 프레임으로 연결해, ‘무엇이 실제로 지각 능력을 강화하는가’를 관찰 가능하게 만든다는 점에서 실무적 가치가 크다. 또한 열려 있는 과제(진짜 general, unified 멀티모달 지능으로의 확장)에 대한 연구 방향을 제시해, 향후 MLLM을 지각-중심 에이전트로 발전시키는 데 참고 가능한 행동 지침을 제공한다.



### LiMoDE: Rethinking Lifelong Robot Manipulation from a Mixture-of-Dynamic-Experts Perspectiv (https://arxiv.org/abs/2606.26183)
- **Prior Approaches**: 기존 일반화 로봇 정책 연구는 연속 작업 적응 중 catastrophic forgetting을 겪어 이전 작업 성능을 유지하기 어렵다는 한계가 있었다. 이를 줄이기 위해 replay, regularization, architectural methods가 나뉘는데, 특히 PEFT 기반 parameter-efficient fine-tuning은 단일 작업 적응에는 효과적이지만 작업 간 공유 지식의 상호작용을 충분히 모델링하지 못해 재사용과 적응 속도가 떨어진다. MoE를 활용하는 시도도 있었으나 lifelong robot learning을 위해 재사용 가능한 skill 추출과 작업 간 결합을 통합적으로 다루는 데는 공백이 남아 있었다.

- **Core Contribution**: 이 논문은 Lifelong Mixture of Dynamic Experts (LiMoDE)라는 2단계 구조를 제안해, 로봇이 prior knowledge를 기반으로 지속적으로 작업을 확장(continual task adaptation)하도록 설계했다. 멀티태스크 pre-training에서는 motion 정보로 가동되는 dynamic MoE를 통해 재사용 가능한 skill을 학습하고, 이후 task adaptation에서는 lifelong MoE adaptation 메커니즘으로 새 작업에 대해 기존 전문가(frozen)와 새 전문가를 동적으로 조합해 지식 전이를 촉진한다. 또한 router 드리프트를 줄이기 위한 replay 전략을 더해 이전 작업 성능 저하를 완화한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 작업 간 공유되는 skill을 실제로 ‘추출’하고 (2) 그 상호작용을 효율적으로 결합하며 (3) 적응 단계에서 forgetting을 제어하는 것이다. 이를 위해 expert를 low-rank 기반으로 이질적으로 구성하고, visual-dynamics-conditioned router가 단기 조작의 복잡도에 따라 활성 expert 수를 가변하는 DyMoES를 도입했다. 이어 adaptation 단계에서는 rank가 다른 lifelong 전문가 라이브러리를 점진적으로 늘리고 top-k 조합으로 동적 결합을 수행하며, router-decorrelation 정규화와 낮은 차원의 router 계수 replay로 router drift를 억제한다.

- **Empirical Impact**: 실험은 시뮬레이션의 LIBERO lifelong learning 벤치마크와 실제 로봇 작업 모두에서 수행됐고, 기존 방법 대비 성능과 lifelong adaptation이 일관되게 개선됐다. LIBERO-LONG에서 adaptation은 7% 향상, forgetting은 3% 감소하는 결과를 보고하며, 추가로 도입되는 학습 파라미터와 추론 오버헤드는 ‘중간 수준’으로 유지되는 효율-성능 절충도 강조한다. 실제 환경 배치에서도 동일한 경향의 효과와 효율성이 관찰되어, lifelong robot manipulation에서 재사용 skill과 작업 간 결합의 실용성을 뒷받침한다.



### KG-TRACE: A Neuro-Symbolic Framework for Mechanistic Grounding in Antimicrobial Resistance Prediction (https://arxiv.org/abs/2606.26179)
Comments:
          8 pages, 3 figures, conference

- **Prior Approaches**: 기존 WGS 기반 AMR 예측은 Linear model, random forest, 1D CNN 같은 통계·딥러닝으로 AUROC를 끌어올리는 데 집중했지만, 각 샘플을 독립적으로 학습해 WHO/CARD 같은 생물학적 지식을 구조적으로 활용하지 못했다. 또한 SHAP 같은 특성 설명은 제공해도, 설명된 변이가 실제 ‘인과 경로(causal path)’인지 ‘동반(co-occurrence) 잡음’인지까지는 구분하기 어려워 임상 신뢰로 연결되기 부족했다.

- **Core Contribution**: KG-TRACE는 WHO mutation knowledge graph(생성 지식 그래프)를 신경 유전체 모델의 제약 조건으로 통합하는 neuro-symbolic 프레임워크를 제안한다. RotatE 기반 KG 임베딩을 genomic feature와 결합하되, learned epistemic trust gate로 신경 근거와 KG 근거의 비중을 조절하고, 예측 결과를 변이-약물 경로로 추적 가능한 형태로 만든다.

- **Technical Challenges**: 핵심 기술 과제는 ‘정확한 예측’과 ‘검증 가능한 기전적 설명’을 함께 만족시키는 것이다. KG-TRACE는 (1) RotatE의 magnitude 특징을 사용해 KG 요약을 만들고, (2) SHAP의 변이 기여를 WHO 경로와 대조해 일치하면 HIGH actionability, 불일치하면 UNCERTAIN로 플래그하는 2단계 메커니즘 그라운딩을 설계했으며, (3) auxiliary gene-detection head로 KG 가지가 학습에서 붕괴하지 않게 했다.

- **Empirical Impact**: CRyPTIC M. tuberculosis 코호트에서 KG-TRACE는 isoniazid에 대해 AUROC 0.9760을 기록해 경쟁적 정확도를 유지하면서도, 핵심 가치는 AUROC 향상이 아니라 상징적 감사 추적(audit trail) 제공에 있다. 또한 Biological Grounding Ratio(BGR)로 신경 attribution이 확립된 생물학과 얼마나 정렬되는지 계량화하고, UNCERTAIN 케이스에서 ‘실험 후속 플래그’를 내 MDR 동반 잡음 아티팩트를 식별하는 등 임상 의사결정에 필요한 불확실성 처리까지 보여줬다.



### LCG: Long-Context Consistent Image Generation with Sparse Relational Attention (https://arxiv.org/abs/2606.26171)
- **Prior Approaches**: 기존 text-to-image 확산 모델은 단일 이미지 품질은 크게 향상됐지만, 만화·스토리보드처럼 여러 패널이 이어질 때는 캐릭터 동일성·역할·시각적 연속성이 쉽게 무너진다. 독립 패널 생성은 패널 간 근거를 놓치고, 순차 생성은 초반의 identity drift가 뒤 패널까지 누적되는 문제가 있다. 또한 패널 전부에 대해 dense attention을 쓰면 메모리 사용량이 패널 수에 대해 빠르게 폭증해 실용 한계를 넘는다.

- **Core Contribution**: 이 논문은 Long-Context Generation(LCG)이라는 프레임워크로, 여러 장면의 텍스트 프롬프트를 입력받아 긴 시퀀스의 다중 이미지를 함께 생성하면서 일관성을 유지하는 방법을 제안한다. 각 프롬프트는 parallel generation branch로 나뉘고, 확산 과정에서 브랜치들을 동시에 denoise해 의미·레이아웃 증거를 교환한다. 일관성과 확장성을 동시에 노리기 위해 Sparse Relational Attention(SRA)과 Routing Consistency Constraint(RCC)라는 두 핵심 장치를 함께 도입한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 장거리 cross-branch 정보 교환을 하되 비용을 줄이고, (2) 학습 중 잘못된 라우팅이 발생할 때 캐릭터·역할 표류를 안정적으로 억제하는 것이다. 논문은 SRA로 블록 요약 기반 후보를 고르고 그 안에서만 국소/관계 토큰을 sparse하게 선택해 계산을 트랙터블하게 만든다. 여기에 RCC는 identity-aware mask에서 얻는 same-identity 대응을 토큰 라우팅 목표로 변환해, 예측된 sparse attention 흐름이 의미적으로 정렬되도록 정규화 손실을 추가한다.

- **Empirical Impact**: 실험에서는 Flux.1-dev 백본 위에 LCG를 얹고, 긴 문맥을 위한 합성 데이터셋 Long-Context Consistency Dataset(LCCD)을 600K 학습 시퀀스(각 6~20장) 규모로 구축해 성능을 검증했다. LCG는 VQAScore 기반 prompt alignment, DreamSim 유사도 기반 subject consistency, FaceSim-Arc 기반 얼굴 동일성, Aesthetic Score 기반 화질에서 기존 베이스라인과 비교해 전반적으로 더 높은 점수를 보이며 human preference에서도 우위를 보였다. 또한 SRA는 dense cross-branch attention 대비 VRAM과 지연을 낮추면서 20패널까지 추론이 가능해, 긴 시퀀스 생성에서 실용적 임팩트를 입증했다.



### Neural Architecture Search for Generative Adversarial Networks: A Comprehensive Review and Critical Analysis (https://arxiv.org/abs/2606.26169)
- **Prior Approaches**: GAN용 NAS 연구는 주로 진화 알고리즘, gradient-based 방법, 강화학습 기반 등 다양한 탐색 전략을 사용해 왔다. 다만 기존 평가지표는 Inception Score(IS)와 Fréchet Inception Distance(FID)처럼 특정 관점에 치우쳐 GAN의 안정성과 모드 다양성을 충분히 반영하지 못하는 한계가 지적된다. 또한 데이터셋 선택과 평가 방식이 달라 결과 비교가 어려워졌다는 문제도 함께 제기된다.

- **Core Contribution**: 이 논문은 GAN에 적용된 NAS 방법을 탐색 전략, 평가 지표, 성능 결과 기준으로 체계적으로 분류·비교하는 리뷰를 제공한다. 이를 통해 어떤 상황에서 evolutionary algorithms나 gradient-based 접근이 유리한지, 그리고 기존 점수 중심 평가의 보완이 왜 중요한지 정리한다. 나아가 GAN 성능을 공정하게 판단하려면 더 다양한 데이터셋이 필요하다는 방향성을 제시한다.

- **Technical Challenges**: 핵심 technical challenge는 NAS가 만드는 후보 아키텍처를 신뢰성 있게 비교할 수 있는 robust evaluation metrics를 설계하거나 선택하는 데 있다. 논문은 IS/FID 외에도 안정성, 학습 효율, 생성 다양성 등 다면적 지표가 필요하며, 데이터셋 차이까지 통제해야 진짜 성능을 가늠할 수 있다고 강조한다.

- **Empirical Impact**: 리뷰 관점에서의 결론이지만, 정리된 비교를 통해 NAS가 GAN의 성능, 학습 안정성, 효율을 개선하는 경향을 확인할 수 있다고 제시한다. 특히 평가 지표와 데이터셋 다양성을 확장해야 향후 NAS-GAN 연구가 더 재현 가능하고 설계 의사결정에 도움이 된다는 점에서 의미가 있다. 연구자들이 다음 실험 설계와 방법론 선택을 더 체계화할 수 있는 가이드 역할을 목표로 한다.



### Reducing Redundancy in Whole-Slide Image Patching for Scalable Indexing and Retrieva (https://arxiv.org/abs/2606.26157)
- **Prior Approaches**: 디지털 병리에서 WSIs 인덱싱은 보통 patch selection으로 대표 패치만 뽑아 저장 부담을 줄이는 방식이었지만, 이후 유사도 검색은 여전히 대량 임베딩 비교 비용을 동반한다. 기존 무감독 patch selection(예: Yottixel, SPLICE, SDM)은 데이터 규모(기깃픽셀)를 다루기 위해 대표성을 유지하는 축소를 목표로 했으나, 결국 임베딩/바코드의 중복이 남아 storage와 계산량을 더 줄이기엔 한계가 있었다. 또한 self-supervised 기반 선택은 진단 의미와 직접 맞지 않는 전처리 과제를 학습해, 진짜 조직적 대표성보다는 전조작 성능에 치우칠 수 있다는 문제가 제기된다.

- **Core Contribution**: 이 논문은 ARReST(Antithetical Redundancy Reduction Strategy)로, 서로 다른 조직 클래스 간 구별에 거의 기여하지 않는 패치를 ‘반대(antithetical) 패치’로 정의해 검색 아카이브에서 제거하는 프레임워크를 제안한다. 동일 클래스 내부 중복만 제거하는 기존 관점에서 벗어나, 클래스가 다르더라도 표현이 과도하게 유사한 패치 쌍을 찾아 전역 유사도 분포 기준으로 pruning 한다. 그 결과 형태학적 다양성이나 검색 정밀도를 크게 해치지 않으면서도 인덱스 크기를 줄이는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 난제는 ‘어떤 패치가 버려도 되는 중복인지’를 재현성 있게 판별하는 것이다. ARReST는 라벨이 있는 TCGA에서 서로 다른 클래스의 패치 임베딩을 무작위로 매칭해 유사도 스코어를 만들고, 전역 유사도 분포의 q-quantile 임계값보다 낮게 나오는 쌍을 찾아 redundancy repository에 저장·활용함으로써 prunining을 체계화한다. 이후 남은 임베딩을 BOB barcodes로 인코딩해 더 빠른 Hamming distance 기반 유사도 계산이 가능하게 하여, 저장 절감과 검색 가속을 함께 달성한다.

- **Empirical Impact**: TCGA 21개 장기(총 11,679 WSIs)에서 Yottixel 기반 patch selection과 UNI 임베딩을 사용해 검증했으며, ARReST는 장기별로 인덱스 압축을 크게 수행하면서 Top-1/Top-k 및 majority voting 기반 retrieval 성능을 대체로 유지했다. 보고된 storage savings는 3%~60%, 평균은 14%±13% 수준이며, 많은 장기에서 retrieval 성능 저하 없이 절감이 가능했다. 다만 intraclass heterogeneity가 크거나 소수 아형 비중이 높은 장기에서는 F1이 감소하는 경향이 보여, 향후 adaptive threshold 또는 subtype-aware pruning의 필요성이 제시된다.



### Unsupervised Memory-Enhanced Video Transformers: Obstacle Detection for Autonomous Agricultural Rover (https://arxiv.org/abs/2606.26151)
- **Prior Approaches**: 정밀 농업에서 자율 로버는 필수지만, 안전을 일관되게 확보하는 것이 여전히 핵심 과제였다. LiDAR 같은 전통 안전 센서는 식생 아래(캐노피 아래)에 있는 장애물을 감지하지 못해 위험이 남는다. 카메라 기반 supervised learning은 학습 데이터에 없던 장애물 상황에서 성능이 크게 저하되며, unsupervised anomaly detection도 로버의 이동으로 인해 생기는 동적 장면에서 흔히 약점을 보였다.

- **Core Contribution**: 이 논문은 동적 농업 장면에서 실시간 장애물 탐지를 목표로, 완전 비지도 anomaly detection 방법인 Video Memory Transformers for Anomaly Detection (VMTAD)를 제안한다. VMTAD는 정상 운용 이미지(라벨 없음)만으로 학습해, 학습 분포 밖 장애물에도 대응할 수 있도록 설계됐다. 특히 transformer 기반 표현에 전용 memory 모듈을 결합해 이전 프레임의 시간 정보를 효과적으로 활용한다.

- **Technical Challenges**: 주요 기술 난제는 로버가 움직이며 카메라 관측이 계속 변하는 동적 맥락을 anomaly 탐지기가 잘 흡수해야 한다는 점이다. VMTAD는 프레임 인코딩을 memory 모듈에서 누적·처리해 temporal context를 제공함으로써, 이동에 따른 장면 변화를 ‘정상’으로 모델링하는 방향으로 문제를 완화한다. 또한 완전 비지도 학습을 유지하면서도 실시간 동작을 위해 구조를 경량화한 변형까지 제시했다.

- **Empirical Impact**: Grillion 농업용 로버의 실험에서 VMTAD는 rapeseed 데이터셋과 같은 까다로운 환경에서도 높은 성능을 보였다. 검출 AUC 0.973, 분할 AUC 0.997 수준의 state-of-the-art 결과를 달성했으며, 경량 모델은 14 ms 추론으로 실시간성을 확보했다. 나아가 로버의 총 정지 거리(total stopping distance) 분석을 통해 안전 관점에서도 효과를 검증해 현장 적용 가능성을 높였다.



### Multiscale Exit-Join Dynamics: Tactical Consensus and Strategic Coalition Formation (https://arxiv.org/abs/2606.26139)
- **Prior Approaches**: 기존 연합(coalition) 형성 연구는 구조 변화의 전략적 규칙은 다루되, 연합 내부의 합의(consensus)로부터 가치가 어떻게 생기는지는 보통 외생적으로 두는 경우가 많았다. 또한 합의/여론 역학 쪽은 네트워크를 주어진 것으로 가정하거나, 전략적 나감·들어옴이 통신 그래프를 어떻게 바꾸는지까지는 충분히 결합하지 못했다.

- **Core Contribution**: 이 논문은 연합 내부에서 DeGroot-style 합의가 일어나며 그 결과로 양도가능 유틸리티(transferable coalition value)가 내생적으로 생성되는 멀티스케일 모델을 제안한다. 이후 연합 간 이동(exit-and-join)은 Aumann–Drèze 보수, switching frictions(전환 마찰), 수락(acceptance rule)을 함께 고려하는 느린 전략 단계에서 전개된다.

- **Technical Challenges**: 핵심 난제는 ‘빠른 전술 합의’가 ‘느린 전략 재구성’의 보수 지형을 어떻게 바꾸는지 상태의 결합(endogenous state dependence)을 엄밀히 다루는 것이다. 논문은 fast–slow 구조를 두고 연합별 primitive 상호작용 행렬에서 합의가 생성하는 정상상태를 characteristic function으로 연결한 뒤, 단일 에이전트의 국소적 exit-and-join이 실행 가능(개인 이득+마찰+수락)할 조건을 고정점 및 존재 결과로 정리한다.

- **Empirical Impact**: 수치 실험에서는 switching barrier가 낮거나 심지어 음수여도 전략적 수렴은 막히는 반면, 시간적 mixing이 충분히 일어나 전역 전술 합의(global tactical consensus)는 오히려 달성될 수 있는 ‘instability-consensus paradox’를 보여준다. 이는 다중 에이전트 시스템에서 연합 재구성과 합의가 함께 결정하는 장기적 안정성·분리·양극화 패턴을 한 틀에서 해석할 수 있음을 시사한다.



### Thinking Like a Scientist? A Structural Study of LLM-Generated Research Methods (https://arxiv.org/abs/2606.26130)
Comments:
          46 pages, 13 figures, 18 tables

- **Prior Approaches**: 기존 연구는 LLM이 연구 글쓰기·아이디어 생성·리서치 코드/방법 제안 등 과정을 자동화하는 양상을 다뤘지만, 최소 프롬프트에서 LLM이 ‘어떤 방법을 기본값으로 부각하는지’에 대한 정량 평가는 제한적이었다. 또한 다양성 감소, 벤치마크 포화/동질화 같은 우려는 제기돼 왔으나, 실제 방법론 탐색이 특정 축(모델 제공사, 데이터 태스크, 평가 지표)에서 어떻게 치우치는지까지는 명확히 분해되지 않았다.

- **Core Contribution**: 이 논문은 최근 arXiv 컴퓨터과학 논문 1,000편에서 추출한 연구 질문만으로 GPT-5.1, Gemini 3 Pro, DeepSeek-V3.2가 제안한 데이터셋·모델·평가 지표·짧은 파이프라인을, 논문 기반 ‘방법 인벤토리’와 직접 비교한다. 특히 질문만 주어진 “초기 제안”이므로 최적화 여부와 무관하게, LLM이 기본적으로 만들어내는 방법론 메뉴의 편향과 압축 정도를 정량화했다.

- **Technical Challenges**: 핵심 과제는 LLM 출력과 논문 인벤토리 간 방법 용어가 서로 다른 명명 방식으로 나타날 때 이를 공통 분류 체계(세부 taxonomy)로 매핑하는 것이다. 논문 측·LLM 측 모두에서 구조화된 메서드 특징을 뽑아 공유 분류로 정규화하고, Jensen-Shannon divergence 같은 정보이론 지표와 co-occurrence(동시 등장) 패턴을 다중 차원에서 비교해 제공사 편중이 가장 크게 드러나도록 검증했다.

- **Empirical Impact**: 결과적으로 제공사(provider) 선택의 편차가 다른 어떤 축보다 3–5배 큰 것으로 나타났고, Other/Academic 단발(singleton) 모델군은 23–24%p 정도 과소대표되는 반면 재사용된 학계/커뮤니티 모델은 소폭(4–6pp) 과대대표됐다. 또한 전체 방법론 ‘메뉴’가 강하게 압축되어 유효 모델 엔티티 수가 1,232에서 59–96으로 줄고, 3개 LLM의 순위 상관이 LLM-논문 상관보다 높아 편향이 여러 모델에 공통으로 나타났다. 제안이 입력 문제에 따라 달라지긴 하지만(질문 민감성), 교차검증 없이 LLM 결과에만 의존하면 실험 설계의 초기가능 후보가 더 좁아져 탐색 공간이 상향 동질화될 위험이 커진다는 점을 시사한다.



### Geometric Fairness-Aware Routing for Federated Edge Networks (https://arxiv.org/abs/2606.26125)
Comments:
          Accepted in the IEEE/ACM Transactions on Networking

- **Prior Approaches**: 기존 라우팅은 OSPF/BGP처럼 네트워크 전체 효율을 최우선하거나, 강화학습·그래프 학습을 써도 평균 지연·처리량을 중심으로 최적화해 이질적인 엣지 노드 간 공정성은 뒷전인 경우가 많았습니다. 또한 federated routing/FL 계열은 비원시 데이터 학습으로 확장성·프라이버시는 주지만, 참여 노드 역량 차이와 geometry(위상/곡률 구조)를 충분히 반영하지 못해 편향된 집계로 이어질 수 있습니다. GNN 기반 라우팅도 대부분 Euclidean embedding에 머물러 계층적·비대칭 연결 패턴을 잘 표현하지 못했습니다.

- **Core Contribution**: 이 논문은 Geo-FairFed라는 기하(geometry) 기반 공정성 인지 라우팅 프레임워크를 제안합니다. 각 엣지 노드는 hyperbolic graph neural network(HGNN)로 음의 곡률 만ifold에서 토폴로지/비대칭 연결을 학습하고, 중앙 집계는 Jain’s fairness index를 반영한 fairness-constrained 목적함수로 노드 간 성능 격차를 줄입니다. 결과적으로 전체 라우팅 손실과 기하적 일관성을 유지하면서도 엣지 전반에서 균등한 성능을 목표로 합니다.

- **Technical Challenges**: 핵심 기술적 난관은 (1) 비유클리드 위상 구조를 학습하면서 (2) federated 업데이트 과정에 공정성 패널티를 매끄럽게 결합하고 (3) 학습 안정성과 수렴성을 보장하는 것입니다. Geo-FairFed는 음의 곡률 만ifold에 대한 manifold 일관성(geo 정규화)과 Jain 공정성에 기반한 inequality penalty를 global 목적에 함께 넣고, curvature가 제한된 조건에서 수렴과 Pareto-improving equilibrium을 보장하는 이론 분석을 제공합니다. 또한 로컬 업데이트는 tangent space에서의 로그/지수 맵으로 수행해 곡률이 그라디언트 왜곡을 과도하게 만들지 않도록 설계했습니다.

- **Empirical Impact**: 동적 6G-edge와 IoT 토폴로지 시뮬레이션에서 Geo-FairFed는 평균 지연을 20% 줄이고 에너지 소비는 17% 감소시켰으며, 공정성은 최대 21%까지 개선했다고 보고합니다. 특히 기존 federated 및 geometric 라우팅 대비 efficiency(효율)와 equity(형평성)를 동시에 끌어올리는 성능을 보이며, 대규모 네트워크에서 hyperbolic manifold로 토폴로지를 임베딩하고 fairness를 집계 단계에 포함하는 접근의 실효성을 실증했습니다. 이는 6G/edge 지능형 네트워크에서 “빠름”뿐 아니라 “고른 서비스”까지 설계 목표로 확장하는 데 의미가 큽니다.



### Privacy-Aware Agent Collaboration for Dynamic VR Slice Management in 6G SD-RAN (https://arxiv.org/abs/2606.26123)
Comments:
          6 pages

- **Prior Approaches**: 기존 연구는 VR 지연·신뢰성 예측(분석/휴리스틱)이나 slice 자체를 최적화하는 방식에 집중했지만, 사용자 이동성과 시간에 따라 변하는 트래픽을 함께 고려한 end-to-end VR 링크 운영은 제한적이었다. RL 기반 O-RAN slicing이나 SD-RAN 최적 정책은 처리량을 올릴 수 있으나, 다중 에이전트 협력이 프라이버시를 어떻게 건드리는지에 대한 통합 설계가 부족했다. 또한 privacy-preserving 기법이 있더라도 VR의 초저지연·고처리량 요구를 만족시키는 자원·공정성·프라이버시의 동시 균형은 여전히 과제로 남았다.

- **Core Contribution**: 이 논문은 6G SD-RAN에서 VR slice 관리를 위해 mobility-driven, privacy-aware Multi-Agent Reinforcement Learning(MARL) 프레임워크를 제안한다. 협력 에이전트들이 end-to-end VR 링크에 대한 자원 배분을 극대화하되, 사용자 데이터는 information bottleneck 인코더로 압축 표현만 공유해 프라이버시 누출을 낮춘다. 특히 VR의 초저지연 E2E 요구를 반영해 UL/DL 비대칭 지연과 공정성을 함께 제약에 넣고 최적화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 이동성에 따라 필요한 RB 수와 α(링크 가중치)가 빠르게 변하는데도 (2) 협력 과정에서 민감한 관측(이동·트래픽 등)이 노출될 수 있다는 점이다. 논문은 Gauss-Markov 기반 지연·지연 임계 조건을 문제식에 포함하고, mobility prediction으로 사전(anticipatory) 자원 조정을 하도록 학습을 구성했다. 프라이버시는 IB(Information Bottleneck) 패러다임으로 공유 메시지의 상호정보를 제한하고, 에이전트가 raw 변수 대신 인코딩 z를 교환하게 하여 협력 오버헤드와 지연을 최소화했다.

- **Empirical Impact**: 시뮬레이션(NS-3+SUMO)에서 제안 방법은 처리량이 최대 34% 향상되고, 자원 사용은 최대 28% 줄이며, 프라이버시 누출은 최대 85% 감소하는 결과를 보였다. 또한 E2E 지연 임계값이 클수록 성능이 크게 상승하는 가운데, 제안 모델이 더 가파른 증가율과 안정적 우위를 보였다. 더불어 프라이버시 인코딩은 온라인 추론 지연이 0.1ms 미만으로 보고되며, 비프라이버시 기준 대비 성능 저하도 약 2% 수준으로 억제해 6G VR의 견고한 몰입형 경험을 뒷받침한다.



### Dot-Flik: A Scalable Edge AI Architecture for Distributed Insect Monitoring (https://arxiv.org/abs/2606.26121)
- **Prior Approaches**: 기존 비전·딥러닝 기반 곤충 모니터링은 높은 정확도를 보여도, 비싼 고성능 하드웨어나 지속적인 cloud 연결에 의존하는 경우가 많아 도시 단위 확장에 비용·전력·네트워크 한계가 생겼습니다. edge AI로 분류를 수행하는 사례도 있으나, 카메라 스트림의 대부분을 그대로 처리/전송해 데이터 중복(빈 프레임·배경 중심 장면)이 에너지와 대역폭을 먼저 소모하는 문제가 남아 있습니다. 또한 시간 간격 샘플링(time-lapse)은 콘텐츠를 보지 못해 드문 방문을 놓칠 수 있어, “전송·처리 이전 단계”의 구조적 최적화가 부족하다는 지적이 제기됩니다.

- **Core Contribution**: 이 논문은 곤충 활동이 있는 프레임만 선별해 edge에서부터 불필요한 데이터를 줄이는 Dot-Flik을 제안합니다. 핵심은 (1) deep learning 추론 없이도 움직임 정보를 활용해 프레임을 걸러내고, (2) 센서 계층의 전처리로 분류 인퍼런스 부담과 통신량을 분리·완화하며, (3) 중앙 AI 분류는 배치 처리로 효율을 높이는 계층형 IoT 아키텍처를 구현한 점입니다. 이를 통해 모니터링 면적이 늘어도 전송·계산이 비례 폭증하지 않는 “확장 가능한 구조”를 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 edge에서 사용할 수 있는 제약(저전력·저가 보드) 안에서, 바람에 흔들리는 식생 잡음은 줄이면서 곤충의 국소적 움직임은 놓치지 않는 프레임 선별을 설계하는 것입니다. 연구진은 temporal differencing(프레임 간 차이)로 움직임 가능성을 만들고, gamma-corrected motion amplification(잡음 증폭을 억제한 보정)과 block-based motion density(블록 단위 밀도 임계치)로 “분산된 배경 흔들림 vs 집중된 곤충 움직임”을 구분하도록 구성했습니다. 또한 중앙 노드는 Raspberry Pi 5 + Hailo 가속기로 분류하고, edge 노드는 Raspberry Pi Zero 2 W에서 필터링 후 H.264 스트리밍으로 후보 프레임만 전달하도록 역할을 엄격히 분리해 실시간 30 FPS 운용을 맞췄습니다.

- **Empirical Impact**: 실외 도시 정원 배치 실험에서 빛 바람 조건(5–15 km/h)에서는 60–80% 프레임 감소가 관측됐고, 강풍에서는 20–40%로 줄어들지만 수동 확인 결과 곤충 활동이 있는 프레임은 대체로 유지됐다고 보고됩니다. 중앙 분류 파이프라인 기준으로 edge-전체 시스템은 30 FPS를 지속하며 추가 연산 여유(computational headroom) 12.8 ms 수준을 확보했고, 전력은 전송률(프레임 드롭률)에 따라 22.6%까지 절감되는 경향이 확인됐습니다. 또한 Dot 센서 스트림을 여러 개 동시 운용할 때 central 노드의 계산이 병목이 되며, 설계 투영상 5–6개의 concurrent edge streams를 지지해 저비용·밀집형 biodiversity monitoring 네트워크의 실용 기반을 제시했다는 점이 의미 있습니다.



### The Open Source Economic Index of AI Adoption and Capability (https://arxiv.org/abs/2606.26118)
- **Prior Approaches**: 기존 연구는 사용자 LLM 대화에서 O*NET 작업을 추정해 AI 사용 패턴을 집계해 왔지만, 대체로 proprietary 데이터에 의존하거나 특정 계층(IWA 수준)에 머무는 한계가 있었다. 또한 채팅을 단순 분류하는 방식은 해당 작업을 모델이 실제로 얼마나 정확히 수행하는지(도구 호출, 근거, 단계 실행)까지는 측정하지 못한다. 한편 GDPVal 같은 deliverable 기반 벤치마크는 과업 설계에 따라 결과가 크게 달라져, “현업 작업과의 정합성” 논쟁이 남아 있었다.

- **Core Contribution**: 이 논문은 두 축의 “경제-역량” 측정을 제안한다. 첫째, WildChat의 공개 사용자-LLM 채팅과 O*NET을 연결해 open-source 경제 지표를 만들고, 어느 직군에서 LLM 활용이 높은지(채팅의 직업적 관련성과 작업 매핑을 포함) 재현 가능하게 보여준다. 둘째, O*NET 작업을 기반으로 MCP 서버(실제 또는 가상) 도구 생태계와 연결된 시나리오를 생성해 Kimi-k2.5의 작업 수행 역량을 단일/다중 턴으로 직접 평가한다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 채팅을 정확히 O*NET 작업으로 매핑하는 것과 (2) 관측된 작업 범주를 “도구 실행이 포함된 실제 시나리오”로 번역해 검증하는 것이다. 이를 위해 gpt-oss-120b로 직업 관련성 필터링, Qwen3-Embedding으로 작업 후보를 최근접 탐색한 뒤 gpt-4o-mini로 중복/오버랩을 판정하며, 임베딩 거리와 O*NET 계층 그래프 거리의 cophenetic correlation로 매핑 품질도 점검한다. 역량 벤치마크에서는 각 O*NET 작업에 맞는 MCP 서버를 의미적으로 매칭(유사도 임계값)하고, 정보 일부를 withheld한 multi-turn과 도구 호출/워크플로/grounding을 LLM-judge로 평가해 자율성과 사용자 협업의 차이까지 분해한다.

- **Empirical Impact**: 채택(Adoption) 측정 결과, AI 활용이 높은 직군은 finance, computer science, arts 쪽에 집중되며, adoption의 “깊이”도 매우 낮게 나타나(75% 이상 작업이 사용되는 직군 0.4%) 과업 실행 역량 대비 채택이 뒤처진 패턴이 보인다. 역량(Capability)에서는 단일 턴 기준 도구 호출의 약 60%가 정확히 이뤄지고 워크플로 완성도는 평균 4.08/5, grounding은 3.7/5로 나왔다. 다만 도구 호출의 세부 오류와 환각(hallucination)이 여전히 비중 있게 관측되며, multi-turn에서는 사용자 협업이 워크플로 완성도를 올리지만 도구 호출 정확도와 grounding에는 트레이드오프가 나타나 사람 감독 필요성을 뒷받침한다.



### The Governance Inversion Hypothesis: Why More AI Regulation May Produce Less Organisational Contro (https://arxiv.org/abs/2606.26117)
- **Prior Approaches**: 기존 AI 거버넌스 프레임워크는 규제가 강화될수록 책임성, 감독, 조직의 운영 통제도 함께 좋아질 것이라고 전제해 왔습니다. 그러나 실제로는 규제·절차가 늘어날수록 현장에서 AI 시스템을 실제로 다루는 권한과 통제는 약해지는 역설이 관찰됩니다. 이 논문은 그 가정에 의문을 제기하며, ‘형식적 통제의 증가’가 ‘실질적 통제의 감소’로 이어질 수 있음을 문제화합니다.

- **Core Contribution**: 이 논문은 Governance Inversion Hypothesis(GIH)를 제안해, 규제 확장과 기술 복잡성이 커질 때 조직이 더 형식적으로는 ‘통치(governed)’되지만 운영 통제는 ‘사라질 수 있다’고 설명합니다. 제도이론과 조직 거버넌스, 책임성 연구를 바탕으로, 규제 확장이 조직의 운영 권한을 약화시키는 구조를 개념적으로 정리합니다. 특히 단순한 분리 현상을 넘어, 거버넌스 확장이 오히려 운영 일관성을 훼손할 수 있는 ‘거버넌스 인버전’이라는 조건을 이론적으로 확장합니다.

- **Technical Challenges**: 핵심 난제는 절차와 권한이 어떻게 ‘겉보기 거버넌스는 강화’하면서도 실제 개입 능력은 ‘침식’되는지 경로를 설명하는 것입니다. 논문은 이를 authority fragmentation(권한 분절), symbolic governance expansion(상징적 거버넌스 확장), externalisation of control(통제의 외부화), authority paralysis(권한 마비)라는 네 메커니즘으로 연결해 보여줍니다. 거버넌스가 계층화되고 절차가 촘촘해질수록 기술 가시성, 에스컬레이션 역량, 개입의 실효성이 유지되기 어렵다는 점을 주장합니다.

- **Empirical Impact**: 논문은 직접적인 실험보다는 제도이론적 개념틀을 통해 AI 거버넌스의 중앙 위험을 재정의하는 데 의미가 있습니다. 즉, ‘거버넌스가 없는 것’이 아니라 ‘갈수록 거버넌스된 것처럼 보이지만 효과적으로 통제할 역량이 줄어드는 제도’가 핵심 리스크라는 관점을 제시합니다. 향후 연구와 정책 설계에서 형식 요건뿐 아니라 권한의 응집성, 기술적 가시성, 개입 실효성을 함께 점검해야 한다는 시사점을 제공합니다.



### Divergent Recommendations, Convergent Diagnoses: Cross-Provider Failure-Mode Convergence in AI Commercial Recommendation (https://arxiv.org/abs/2606.26116)
- **Prior Approaches**: 기존 AEO/GEO 평가는 주로 “제공자별 대시보드”로 가시성을 나눠 측정했으며, 추천 픽이 provider마다 달라질 수 있다는 전제를 암묵적으로 깔고 있었다. 또 일부 벤치마크는 한 엔진 중심의 성능 향상(예: 단일 모델에서의 headline lift)에 초점을 두어, 브랜드 개입이 다른 제공자 표면으로 얼마나 이전되는지(transfer)까지는 충분히 답하지 못했다.

- **Core Contribution**: 이 논문은 추천 결과(pick)와 실패 진단(diagnosis)을 분리해, provider가 달라져도 “둘 다 추천하지 못할 때의 이유”는 얼마나 일치하는지 정량화한다. 그 결과 provider별 추천 집합은 약 3분의 2 수준으로 불일치하지만(교차 제공자 Jaccard 0.35), 공동 실패의 진단 단계는 95.1%로 강하게 수렴한다.

- **Technical Challenges**: 핵심은 두 층을 공정하게 분리하는 실험 설계였다: 4회 측정 배치에서 동일 프롬프트/브랜드 후보 카탈로그/추출 파이프라인으로 두 제공자를 비교하고, 두 제공자 모두 추천하지 않은 ‘joint-failure’만 모아 4단계 퍼널(stage)로 분류했다. 또한 “어떤 경로로 추천했는가”를 보기 위해 pure-priors(관측 가능한 검색·검색단서 증거 없이도 추천하는 비율)와 OpenAI의 batch-reconstruction(초기 검색에서 브랜드 식별 질의가 나오는지)을 함께 사용해 메커니즘 비대칭을 검증했다.

- **Empirical Impact**: 215개 상업 프롬프트에서 OpenAI와 Anthropic은 추천 픽이 roughly two-thirds 수준으로 갈렸지만, joint-failure 7,763개에서는 3가지 실패 모드( discoverability/compellingness/positioning ) 중 같은 모드로 진단하는 비율이 95.1%로 보고됐다. 특히 브랜드가 장기 꼬리(long tail)일수록 같은 진단이 81.2%(L1)→99.6%(L5)로 점증하며, visibility 향상은 두 provider 모두에 효과가 있고(특히 discoverability 관련), 범주 리더의 경우는 content-level 작업이 provider별로 더 달라지는 경향을 시사한다.



### A Multi-Layer AI Framework for Information Landscape Analysis (https://arxiv.org/abs/2606.26115)
Comments:
          Accepted at the Information Disorder (InDor) Workshop, LREC 2026. 10 pages

- **Prior Approaches**: 기존 연구는 허위정보를 true/false 같은 라벨 분류로 처리하거나, 출처 신뢰도를 점수화하는 방식으로 접근해 왔다. 하지만 이 접근은 ‘사실은 맞지만 맥락을 의도적으로 빼는 경우’와 같은 정보 혼탁의 병리들을 구조적으로 설명하기 어렵다. 또한 출처 신뢰도와 주장 정확성이 얽히면서, 신뢰할 만한 매체의 오보 프레이밍이나 정반대 상황을 놓치는 문제가 반복됐다.

- **Core Contribution**: 이 논문은 정보를 이진 판정이 아니라 다차원으로 분해해 지도화하는 multi-layer AI 프레임워크를 제안한다. 핵심은 claim–source independence로, 주장(veracity)과 출처 신뢰도(source reliability)를 별도 트랙에서 점수화해 guilt-by-association을 줄이는 설계다. 웹 기반 플랫폼은 URL 단위의 deep analysis와 키워드 단위의 narrative mapping 두 모드를 제공해 사건의 ‘정보 생태계’를 시간축에서 해석하도록 돕는다.

- **Technical Challenges**: 가장 큰 기술적 난제는 빠르게 흩어지는 콘텐츠를 사람이 라벨링하는 것처럼 검증 가능한 형태로 구조화하되, LLM 기반(Claude Sonnet 4, GPT-4o-mini) 접근의 불확실성을 관리하는 것이다. 논문은 11개 분석 레이어(프레이밍, 조작 징후, 전파, 이미지/영상 무결성, 감정·언어/이념 등)에서 각각 독립 점수를 산출하고, 신뢰 점수는 HTLS 같은 긴장 상태를 평균으로 덮지 않도록 조건부 플래그와 캡(calibration cap)을 둔다. 다만 이 점수들은 학습된 분류기 검증값이 아니라 structured prompting 기반의 확률적 출력이라는 한계를 함께 전제한다.

- **Empirical Impact**: 사례 연구로 2026년 Macron–Epstein FIMI(허위조작) 캠페인을 분석했으며, 키워드 생태계 분석에서는 Trust Grade D(18%)로 ‘에코시스템 수준의 거짓’이 강조됐다. 반면 같은 생태계에서 URL 단위로 개별 기사를 보면 Trust Grade A(86%)처럼 ‘신뢰할 만한 보도’가 분리되며, 하나의 토픽 안에 서로 다른 정보 문제(능동 제작 vs 책임 보도 vs 품질 저하)가 공존함을 보였다. 다만 정량 점수는 대규모 ground truth/전문가 벤치마크로 검증되지 않았고, 현재 영어 검색에 편향이 있어 multilingual 확장과 장기 실증 검증이 후속 과제로 제시된다.



### Dream machine -- the next creative economy (https://arxiv.org/abs/2606.26114)
Comments:
          409 pages. 11 figures, 4 tables

- **Prior Approaches**: 기존 논의는 generative AI가 창작을 대체한다는 공포(boom vs doom)나, 도구 리뷰처럼 개별 기능에만 집중해 산업 전반의 구조 변화를 놓치기 쉬웠다. 또한 정책·노동·플랫폼·저작권이 어떻게 맞물려 공급망과 의사결정 체계를 무너뜨리는지에 대한 통합 프레임이 부족했다.

- **Core Contribution**: 이 책은 창작 산업의 전환을 Human–AI Agency Continuum(인간-에이전시 연속체)로 지도화해, 인간과 기계의 협업 스펙트럼을 한 번에 분류한다. 아울러 AI 생성물이 실제로 소비되는 방식에 ‘slop ceiling(저질 상한선)’이라는 청중 기반 품질 제약이 작동한다고 주장하며, 이를 통해 시장의 상한과 판정 메커니즘을 설명한다.

- **Technical Challenges**: 핵심 기여를 설득력 있게 만들기 위해서는(1) 정책 문서·산업 지표·창작자 설문·플랫폼 분석을 374개 1차 자료로 엮는 정량/정성 정합성, (2) ‘대체 vs 증폭’ 같은 이분법을 넘어 협업 단계를 구조화하는 프레임 설계가 필요했다. 저자는 2025년 Sora(및 Sora 2) 같은 분기점 사례를 추적하면서, 투명성·동의·보상·human-centred design 원칙과 함께 Agency/Attribution/Access/Audience 관점의 판단 테스트를 제시해 실행 가능한 기준을 만든다.

- **Empirical Impact**: 영국 정부의 AI·저작권 협의에서 88%가 ‘모든 경우에 라이선스 필요’에 반대(확장된 AI 학습권 부여 반대)했다는 결과는 노동·창작자 권리와 기술기업 간 구조적 긴장을 정량으로 부각한다. 또한 slop ceiling 주장에 따르면 AI 생성 업로드는 44%이지만 플랫폼 스트림에서 약 1–3%만이 도달해, 대량 생산보다 ‘승인 가능한 품질’과 ‘의도/진정성’이 시장에서 통제력을 갖는다는 의미를 준다. 결과적으로 이 책은 창작자의 일자리 재편(예: orchestrator, prompt engineer)과 스튜디오의 AI-augmented 생산 파이프라인 전략을 함께 다루며, 2030년대의 창작 경제 설계를 위한 기준점으로 자리잡을 신호를 보낸다.



### From Lexicon to AI: A Structured-Data Pipeline for Specialized Conversational Systems in Low-Resource Languages (https://arxiv.org/abs/2606.26112)
Comments:
          12 pages, 3 figures

- **Prior Approaches**: 기존 저자원 언어용 대화형 AI는 대규모 말뭉치(예: Common Crawl 수준)에 의존하는 미세조정 패러다임이 중심이었고, 교육처럼 도메인·수준에 맞춘 고품질 instruction 데이터가 부족할 때 성능 격차가 커진다. 또, cross-lingual transfer나 prompt 기반 방법은 사전학습 지식에 크게 기대어 환각·사실 불일치 위험이 남아 전문 영역에서 신뢰도가 흔들릴 수 있다. 한편 FAQ/검색형 지식 결합은 생성형 대화의 깊이를 제한해 “가르치는 대화”로 이어지기 어렵다.

- **Core Contribution**: 이 논문은 WordNet 같은 전문가가 큐레이션한 구조화 언어자원을 대화형 학습 데이터로 변환해, 저자원 언어의 특화 conversational AI를 만드는 체계적인 파이프라인을 제안한다. 힌디어 WordNet을 1.25M 규모의 instruction-response로 만들고, 12B 파운데이션 모델을 LoRA와 4-bit quantization으로 resource-efficient하게 fine-tuning해 ‘Shabdabot’을 구성했다. 교육용 언어학적 관계(동의/반의, 상하위, 동음이의 등)를 대화 흐름에 유지하면서 수준(Primary~Expert)별로 설명 깊이와 예시를 조절하는 점이 핵심이다.

- **Technical Challenges**: 문제는 (1) 구조화된 synset/관계 정보를 대화형 데이터로 자동 변환하면서 의미 관계의 연속성을 보장하고, (2) 밀집 관계에서 생기는 정보 과부하를 통제하며, (3) 전문가 지식 기반 생성이 되도록 모델을 효율적으로 특화하는 것이다. 저자들은 관계 타입을 기반으로 Basic/Complex/Ontological/Disambiguation 4종 지시문을 만들고, 10개 초과 관계는 chunking 하되 33% 겹침 슬라이딩 윈도로 카테고리 연속성을 학습하게 했다. 또한 서로 다른 관계를 multi-hop 형태로 묶는 커버리지 체크와 instruction-response 해시 중복 제거로 1,253,847개의 정제 데이터셋을 구성했으며, 4-bit NF4+LoRA(총 파라미터의 0.2%만 업데이트)로 12GB급 배치가 가능하도록 했다.

- **Empirical Impact**: 힌디어 언어학습 챗봇 평가에서 Shabdabot은 교육적 유효성 지표 LAQ 91.0을 기록해 일반 목적 모델 대비 우위를 보였다(예: 79.4 vs. 83.6 범위). 의미 정확성(SAS)은 전반적으로 경쟁력을 유지하면서도, 무엇보다 응답 일관성에서 σ=1.0으로 크게 개선돼 일반 대형모델의 σ=7.4 대비 예측가능성이 크게 향상됐다. 저자들은 의미 유사도만으로 교육 성과를 예측하기 어렵고, 구조화 지식 기반 특화가 “정확성+신뢰도”를 함께 바꾼다는 메시지를 제시하며, WordNet이 존재하는 200+ 언어로 확장 가능한 재현 가능한 개발 방법론임을 보여준다.



### Generative AI and Copyright Infringement: A Legal-Technical Analysis of AI Music Generation Systems Under 17 U.S.C. Title 17 (https://arxiv.org/abs/2606.26111)
- **Prior Approaches**: 기존 논의는 음악을 학습 단계(Training), 생성 결과(Output), 유통·수익 단계(Distribution)로 나눠 저작권 침해 가능성을 점검해왔다. 다만 연구와 판례는 가사·멜로디(작곡/작사가) 보호에는 비교적 일관된 잣대를 적용하지만, 합성 보컬처럼 ‘목소리 동일성’은 연방 저작권 권리로 포섭되지 않는 경우가 많다는 점에서 한계가 있었다.

- **Core Contribution**: 이 논문은 “한 아티스트의 저작권 가사를 프롬프트에 넣고, 다른 아티스트의 보컬 음색/스타일을 지시해 만든 곡을 게시·수익화”하는 시나리오를 Title 17의 권리(복제·2차적저작물·배포)와 §114의 사운드레코딩 예외, 그리고 주(州) 차원의 voice-cloning 관련 publicity 규정으로 체계적으로 매핑한다. 그 결과 가사 무단 사용은 음악 저작물의 침해 위험이 높지만, 녹음 자체를 샘플링하지 않은 ‘사운드얼라이크’ 보컬은 대체로 연방 사운드레코딩 보호 영역 밖이라는 구도를 강조한다.

- **Technical Challenges**: 핵심 기술 난제는 프롬프트 인코딩(가사 토큰/임베딩)과 멜로디·보컬 생성 모듈이 섞이면서, 결과물이 ‘복제(표현의 재현)’인지 ‘독립 생성’인지 법적 판단 기준을 흐린다는 점이다. 논문은 latent diffusion, neural vocoder, speaker embeddings 같은 파이프라인 구성 요소를 기준으로 위험을 분해해 설명하며, 가사·멜로디의 경우는 복원/재현 가능성 때문에(2차적저작물 포함) 위험이 커지고, 보컬의 경우는 실제 오디오 고정 샘플을 가져오지 않으면 연방 청구가 약해질 수 있다고 정리한다.

- **Empirical Impact**: 사례들은 이 ‘비대칭’이 실제 소송 결과에서도 반복됨을 보여준다(가사 출력 관련 청구는 작곡권자 쪽에 유리, 보컬 사운드얼라이크는 연방 저작권보다는 주 권리로 이동). 논문은 Concord v. Anthropic, Lehrman v. Lovo, Tennessee의 ELVIS Act, UMG v. Uncharted Labs 같은 흐름을 묶어, 연방 차원의 보호는 가사·멜로디에 강하고 합성 보컬의 동일성에는 처방이 불명확하다는 규제 공백과 정책 개정 필요성을 설득력 있게 제기한다.



### Low Resource Multimodal Translation of Nepali Spoken Words into Emotion-Conditioned Sign Language Avatars (https://arxiv.org/abs/2606.26107)
Comments:
          15 pages, 5 figures, 9 tables

- **Prior Approaches**: 기존 수어 아바타 연구는 감정 표현을 더하려는 시도는 있었지만, 대개 범용 감정만 반영하거나 자연스럽고 언어적으로 정확한 표현까지는 한계가 컸습니다. 또한 음성→수어로 넘어가더라도 어휘(lexical) 번역에 초점이 맞춰져 감정 맥락을 함께 처리하지 못하는 경우가 많았습니다. 특히 네팔리처럼 데이터가 부족한 저자원 언어에서는 감정 주석 음성 데이터와 아바타 실증이 거의 없었습니다.

- **Core Contribution**: 이 논문은 네팔리 음성을 입력으로 받아 ‘감정 조건부’ 네팔리 수어(NSL) 아바타를 생성할 수 있음을 보여주는 NEST-V1( Nepali Emotion and Speech Transformer - Version 1) 파일럿을 제안합니다. 음성에서 단어(“thank you”, “hello”, “house”, “me”)와 감정(happy, neutral, sad)을 동시에 추정해, 해당 조합에 맞는 표정·제스처 애니메이션을 출력하는 구조가 핵심입니다. 또한 저자원 환경을 염두에 둔 경량·모듈형 파이프라인으로 확장 가능성도 함께 제시합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 저자원 데이터에서 음성의 단어 인식과 감정 분류를 동시에 안정적으로 학습하고 (2) 이를 엣지에서도 실시간으로 돌릴 만큼 가볍게 만드는 것입니다. 논문은 Mel spectrogram을 2D 이미지처럼 보고 Vision Transformer 계열 백본을 3개 레이어만 사용하며, ASR과 감정 분류에 공유 acoustic encoder를 적용해 파라미터를 절감했습니다. 그 결과 경량 입력(128×128)과 고정 프레임(200)을 통해 학습·추론 일관성을 확보하고, 전처리(리샘플링, 2초 고정, VTLP/반음 반이동)로 데이터 다양성을 보강했습니다.

- **Empirical Impact**: 실험에서 NEST-V1은 600개 레이블 오디오(50명 스피커 규모) 기반으로 ASR 정확도 81.1%, 감정 인식 정확도 79.21%를 보고했습니다. 또한 ASR·감정 분류를 분리한 구조 대비 약 37% 파라미터 효율을 달성하면서 총 22.1M 파라미터로 엣지 배치에 적합한 경량성을 입증했습니다. 현재는 4개 단어와 3개 감정 조합으로 제한되지만, 저자원 환경에서 감정 인지까지 포함한 실시간 음성-수어 인터페이스의 기술 토대를 마련했다는 점에서 의미가 큽니다.



### Reducing Conversational Escalation in Large Language Model Dialogue with Nonviolent Communication Constraints (https://arxiv.org/abs/2606.26106)
- **Prior Approaches**: 기존 LLM 안전 연구는 유해·편향·정책 위반 같은 명시적 출력 차단에 집중해 왔다. 반면, 사용자의 분노·갈등 상황에서 대화 톤과 표현 방식이 갈등을 키울 수 있는 ‘대화적 에스컬레이션’은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 Nonviolent Communication(NVC)의 핵심 원칙을 ‘과정(process) 기반 프롬프트 제약’으로 재구성해, 갈등 상황에서 LLM이 더 완화적(de-escalating) 대화를 하도록 유도한다. 시스템 프롬프트만으로 (1) 책임 전가·판단·의도 추정 회피, (2) 사용자의 감정에 대한 주의 강조, (3) 조언 전 명확화 우선 같은 행동 제약을 구현한다.

- **Technical Challenges**: 주요 기술 과제는 특정 문장 템플릿이 아니라, 모델의 반응 생성 과정에서 에스컬레이트할 위험 행동을 얼마나 일관되게 억제할지였다. 저자들은 다양한 instruction-tuned 모델과 사용자 저항도(낮음/중간/높음)를 갖춘 dual-agent(assistant–User Simulator) closed-loop 시뮬레이션과 LLM 기반 judge로 대화를 장기 추적 평가했다.

- **Empirical Impact**: 실험 결과, NVC 제약 프롬프트는 전반적으로 갈등의 전개(Conflict Trajectory Score)를 악화시키는 경향을 줄이고 상호작용 안정성을 높였다. 특히 사용자 저항도가 높은 조건에서 Vanilla baseline의 음수 경향을 완화하며, DeepSeek-V3와 Claude-4.5-Sonnet 두 judge 모두에서 ‘에스컬레이션 억제’ 패턴이 일관되게 나타났다. 저자들은 이는 감정이 격해진 온라인 논의·고객지원 등에서 LLM 신뢰성을 높일 수 있는 간단한 접근일 수 있음을 시사한다고 정리했다.



### Context Recycling for Long-Horizon LLM Inferenc (https://arxiv.org/abs/2606.26105)
- **Prior Approaches**: 기존 long-context 대응은 주로 RAG처럼 관련 문서를 검색해 프롬프트에 주입하거나, LoRA 같은 fine-tuning으로 지식을 가중치에 흡수하는 방식이었습니다. 다만 RAG는 세션 메모리가 없어 매 턴 독립적으로 재검색해야 하고, 검색 지연·근사 매칭·평평한 구조 문제가 남습니다. 또한 긴 context window를 늘리는 접근은 프리필 지연과 서빙 비용이 커지고, 기본적으로 세션 간 상태가 유지되지 않는다는 한계가 있습니다.

- **Core Contribution**: ContextForge는 LLM의 context window를 대화 누적 버퍼가 아니라 “재사용 가능한 작업공간”으로 재정의하고, 고정 토큰 예산 안에서 턴별로 컨텍스트를 로드/해제하는 context recycling을 제안합니다. 이를 위해 5단 메모리 계층(선택적 LoRA, KV-cache 기반 안정 프리픽스, 인덱싱/라우팅, 영속 저장, 야간 프리컴퓨트)을 구성해 지식 저장소는 크게 두되 활성 컨텍스트는 고정 크기로 관리합니다. 결과적으로 긴 하이즌에서도 토큰 오버헤드 없이 일관된 멀티턴 추론을 노립니다.

- **Technical Challenges**: 핵심 과제는 (1) 대화가 길어져도 context가 무한히 커지지 않게 하면서 (2) 필요한 지식은 정확히 다시 불러오고 (3) 매 턴의 조립 비용을 낮추는 것입니다. ContextForge는 FTS5 BM25 기반 지식 tree에서 요약(summary)으로 분기(branch)를 결정하고, 실제 컨텍스트 주입은 해당 분기 콘텐츠로만 수행해 토큰 사용을 고정 예산에 가깝게 묶었습니다. 또한 KV-cache의 안정 prefix 재사용(시스템 프롬프트 등)을 포함하고, 대화가 길어지면 LLM 기반 context compaction으로 히스토리를 4–8배 압축해 토큰 폭증을 억제합니다.

- **Empirical Impact**: 276M 행의 CMS Medicare 데이터 기반 12턴·15턴 벤치마크에서 ContextForge는 정확도는 유지하거나 약간 개선하면서도 baseline 대비 토큰을 크게 줄이고 속도를 끌어올렸습니다. 12턴에서는 약 4.2배 적은 토큰과 4.7배 빠름을 보였고, 15턴에서는 컨텍스트 길이가 늘며 효율 격차가 더 벌어져 토큰 13.4배, 속도 8.0배 수준까지 확대됐습니다. 즉, 긴 대화에서 성능 하락 없이 비용을 누적 절감하는 “시스템 레벨” 확장 전략으로 의미가 큽니다.



### Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfar (https://arxiv.org/abs/2606.26104)
- **Prior Approaches**: 기존에는 영향도 추정(influence functions, TrackStar, MAGIC)이나 사전/사후 혼합된 평가(퍼플렉시티 등)로 학습 데이터의 효과를 보려 했지만, small matched-pair 조건에서는 신호 대 잡음이 불안정해 stance와 vocabulary가 섞이기 쉽습니다. 특히 문서 단위 귀속은 특정 쓰기 표현이 실제로 모델의 ‘입장 선택’을 바꾸는지에 대한 직접 답을 주지 못했습니다. 또한 이전 연구들은 사후학습이 기존 정렬을 훼손할 수 있다는 신호도 보여주었습니다.

- **Core Contribution**: 이 논문은 동물복지 텍스트에서 10개 언어 특성을 한 번에 하나씩만 바꾼 paired 데이터로, fine-tuning 이후 held-out 동물복지 벤치마크에서 ‘찬성 입장 선택’이 얼마나 변하는지 행동 평가로 측정합니다. 핵심은 vocabulary-matched stance-contrast 설계를 통해 단순 단어 인식이 아니라 stance 자체 변화만 분리해 본다는 점입니다. 즉, “어떤 문장 스타일이 모델의 입장 추론을 실제로 이동시키는가”에 대한 실증적 답을 제공합니다.

- **Technical Challenges**: 기여를 위해서는 (1) 한 기능만 바꾼 채 나머지 문장 조건을 최대한 고정하고, (2) 대답 후보가 동물복지 관련 단어를 충분히 공유해 vocabulary 요인을 제거해야 했습니다. 또한 binary-choice 지표는 베이스라인 선호가 높아 천장효과가 생길 수 있어, aligned와 misaligned completion의 length-normalized log-prob 차이를 preference score로 주신호로 삼았습니다. Llama-3.2-1B에 대해 각 feature의 present/absent 변형으로 LoRA fine-tuning을 반복하고, seed 간 per-seed 차이를 paired t-test로 검정했습니다.

- **Empirical Impact**: 10개 중 8개 언어 특성이 통계적으로 유의미한 stance 변화를 만들었고, 그중 7개는 동물복지 찬성 추론을 강화했습니다(도덕 어휘, 평가적 주장, 감정 단어, 내러티브 구조, 위해 강도, 즉시 시간 프레이밍, 주장 확실성). 반대로 hedging(완곡한 추정/가능성 표현)과 concrete sensory description(구체 감각 묘사)은 찬성 입장을 희석해 평균적으로 불리하게 작동했습니다. 실무적으로는 “장면 중립 묘사보다 자신의 입장을 단정적으로 드러내라”가 권고되며, base model은 이미 강하게 친(親)동물복지 성향이라 많은 fine-tuning이 그 정렬을 깎을 수도 있다는 점도 함께 드러났습니다.



### Investigating LLM's Problem Solving Capability -- a Study on Statics Questions (https://arxiv.org/abs/2606.26103)
Comments:
          9 pages, Engineering and Technology Symposium 2026

- **Prior Approaches**: 기존 연구들은 LLM의 교육 효과를 다뤘지만, 주로 공개·오픈 문제 데이터셋에 의존해 주제별 분석이 부족하다는 한계가 지적된다. 공학 교육, 특히 기계공학에서 정역학 같은 특정 문제 유형에 대한 체계적 성능 평가는 상대적으로 제한적이었고, 보통 교재형 문항을 그대로 LLM에 질의하는 방식이 주를 이뤘다.

- **Core Contribution**: 이 논문은 정역학 문제 해결 능력을 평가하기 위해 모델 증류(distillation) 절차를 도입해 ChatGPT로부터 25개의 텍스트-only 정역학 문항을 추출한다. 또한 도표를 추가하거나 수치를 변경해 2종의 변형 데이터셋을 구성함으로써, 문제 양식과 요구 추론 복잡도가 성능에 미치는 영향을 분리해 분석한다.

- **Technical Challenges**: 핵심 과제는 텍스트 문제에서 잘 되는 능력이 도표가 포함되고 다단계 추론이 필요할 때 어떻게 무너지는지를 설계적으로 재현하는 것이다. 연구팀은 도표 도입 및 수치 변형을 통해 시각 정보의 활용 방식이 답안 단계 전반에 일관되게 적용되는지(단계 간 정보 전이)와, 다단계 추론 자체의 난도를 동시에 흔들 수 있게 했다.

- **Empirical Impact**: 실험 결과 LLM은 텍스트-only 정역학에서는 높은 성능을 보이지만, 도표가 들어가면 정확도가 감소하고 다단계 추론이 요구될수록 더 큰 하락이 나타난다. 추가 분석은 이 성능 저하가 주로 image recognition의 부족 때문이 아니라, 다단계 reasoning의 어려움과 단계별로 시각 정보를 일관되게 적용하지 못하는 문제에서 비롯됨을 시사한다.



### Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training (https://arxiv.org/abs/2606.26102)
- **Prior Approaches**: 기존 정렬은 사후학습에서 SFT와 RL로 Helpful/Honest/Harmless 같은 HHH 목표를 맞추지만, 이 과정이 선학습된 가치(예: harmless)를 훼손할 수 있다는 점이 선행연구에서 지적돼 왔다. 또한 정렬 강화를 위한 미세조정이 독해력 등 핵심 능력을 떨어뜨리거나, 입력 도메인에 따라 내부 표현이 예측 가능하게 이동한다는 이론·실증이 누적됐다. 다만 mid-training 뒤에 ‘사후학습 데이터 도메인’이 가치 보존에 미치는 영향은 직접 비교가 부족했다.

- **Core Contribution**: 이 논문은 Llama 3.1 8B를 동물 연민(Animal compassion) 중심 synthetic 데이터로 mid-trained 한 뒤, 사후학습에서 Helpfulnes(도움말) 도메인과 coding(코딩) 도메인을 달리 했을 때 가치가 어떻게 남는지 실험적으로 검증한다. SFT와 GRPO 두 패러다임에서 helpfulness 계열 학습은 AHB에서 동물 연민 점수를 유의하게 떨어뜨리지만, coding 계열 학습은 기준선(base)과 유사하게 보존한다. 더 나아가 MORU에서 일반 도덕 추론의 저하도 영어에서는 나타나지만 다국어로 합치면 사라지고, 동물 연민 효과는 언어 전반에 견조하게 유지됨을 보여준다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 선학습/ mid-training에 심어진 가치를 ‘사후학습 도메인’만으로 공정 비교하고, (2) SFT와 RL 계열이 서로 다른 업데이트 강도 때문에 생기는 효과를 분리해 해석하는 것이다. 이를 위해 동일 base 모델을 공유하고 SFT는 Dolly(도움말) vs Magicoder(코딩), GRPO는 RLHFlow(도움말) vs Magicoder 보상으로 바꿔 도메인 변수만 달리했으며, 평가에서는 Inspect AI의 동일 judge 모델을 사용해 AHB 2.2와 MORU를 일관되게 측정했다. 또한 영어 전용 학습 설정 때문에 MORU는 영어 단독과 다국어 통합을 함께 제시해, 언어 순응도 차이로 인한 혼입 가능성을 분석했다.

- **Empirical Impact**: 결과적으로 helpfulness 학습은 AHB 영어 항목에서 base 대비 동물 연민을 크게 훼손한다(예: SFT에서 35.7% vs 65.2%, GRPO에서 18.7% vs 32.0%). 반대로 coding 학습은 base와 큰 차이가 없었고, 두 독립 helpfulness 데이터셋과 두 학습 패러다임에서 동일한 방향성이 반복됐다. MORU에서는 영어 항목에서 일반 도덕 추론이 25.5%p 가량 저하되지만(46.4% vs 71.9%), 다국어 MORU에서는 해당 도메인 효과가 사라진다. 반면 동물 연민은 언어를 가로질러 전이되며, 특히 non-English 평가에서 coding 도메인이 helpfulness 대비 더 큰 이점을 보여 실무적으로 “가치가 들어간 mid-training 이후엔 coding 도메인 사후학습이 더 안전할 수 있다”는 시사점을 제공한다.



### Know2Guess: A Contamination-Aware Multi-Zone Benchmark for Knowledge-Boundary Evaluation in Large Language Models (https://arxiv.org/abs/2606.26101)
Comments:
          16 pages, 3 figures

- **Prior Approaches**: 기존 평가는 정확도 중심의 Truthfulness/Hallucination, 다중지표(HELM), 시간변화(FreshQA 계열) 등으로 분산돼 있고, abstention(선택적 미응답)과 policy refusal(정책 거부), 데이터 오염(bench leakage)을 한 프레임에서 함께 통제하긴 어려웠다. 또한 abstention을 “답하지 않음”으로 뭉개면, 불확실성 기반 신중함과 단순한 거부 행동이 섞여 경계(지식이 끝나는 지점)를 해석하기가 애매해진다.

- **Core Contribution**: 이 논문은 contamination-aware 멀티-존(multi-zone) 벤치마크로, build-time 라벨을 고정한 채 “정답 가능한 영역에서 답하기→모른다고 판단해 abstain하기(거부와 분리)” 전환을 측정하는 프로토콜을 제안한다. 총 1,200문항을 5개 도메인에 배치하고, 답하기 기대(Zone A–C)와 abstention 기대(Zone D)를 나누며, 각 문항에 오염 위험 메타데이터와 provenance를 함께 제공한다.

- **Technical Challenges**: 핵심 난제는 모델 출력에서 ‘형식 준수 abstention’이랑 ‘정책 refusal’을 구분하고, 데이터 오염이나 프롬프트 템플릿 차이가 성능을 덮어쓰지 않게 평가를 잠그는 것이다. 이를 위해 locked answer-or-abstain 프롬프트, strict/normalized 듀얼 파서, answer-only 대조군, 그리고 contamination-risk 태그 기반 메타데이터 슬라이싱을 함께 적용해 관찰된 순위가 파서·프롬프트 편향인지 검증한다.

- **Empirical Impact**: 결과적으로 generic한 비응답으로는 벤치마크가 해결되지 않았다. Qwen2.5-3B-Instruct가 전체 reliability는 최고(0.3657)지만, answer-expected 경계(Zone A–C)는 여전히 어렵고 calibration(ECE)이 높으며 benign-item에서 refusal 같은 실패 모드가 남아 있다. 또한 parser normalization과 프롬프트 변형, cost-sensitive 재가중에도 주요 순위와 결론이 유지돼, answerability·abstention·refusal·contamination을 분리 진단하는 재현 가능한 감사(audit) 절차로 의미가 크다.



### Benchmarking Open-Weight Foundation Models for Global AI Technical Governanc (https://arxiv.org/abs/2606.26099)
Comments:
          12 pages, 5 figures, 3 tables

- **Prior Approaches**: 기존 연구들은 LLM이 국가별로 정확도가 크게 달라지는 현상, 즉 geographic bias를 보고해 왔습니다. 다만 가중치가 공개되지 않은 proprietary 시스템에 의존해 재현이 어렵고, 학습 데이터 수집 이후 연도에 대한 질문을 평가해 모델의 ‘지리적 편향’뿐 아니라 ‘지식 부재’가 섞여 측정이 흔들린다는 한계가 있었습니다. 또한 정오 여부 같은 이분류만 써서, 그럴듯하게 지어내는 confident fabrication(HF)과 불확실하다고 인정하는 honest refusal(HR)이 구분되지 못했습니다.

- **Core Contribution**: 이 논문은 geographic bias의 원인과 결과를 더 엄밀히 분리하기 위해, 4개의 open-weight frontier language model을 Global AI Dataset v2(GAID v2)로 벤치마크합니다. GAID v2는 227개 국가에 대한 24,453개 지표의 검증된 정답을 포함하며, 이 연구에서는 IEEE IRAI 2026 프레임워크의 8개 주제 축에 대응되는 지표 18개를 뽑아 2010-2023 구간의 6개 평가 연도에 대해 약 2,990개의 country-metric-year 관측치를 구성했습니다. 아울러 모델의 응답을 VA, HF, HR, QH, MF의 5범주로 세분화해 “정확성 저하”를 더 정밀하게 진단합니다.

- **Technical Challenges**: 핵심 기술적 도전은 모델이 모르는 연도에서의 자연스러운 한계(geographic ignorance)와 실제 지리적 편향의 영향을 분리해내는 데 있었습니다. 저자들은 학습-평가 연도 경계가 비교적 명확한 2010-2023 내에서 평가를 설계하고, mixed-effects logistic regression과 difference-in-differences(DiD)로 지역별 정확도 격차를 추정해 교란 가능성을 줄였습니다. 동시에 이분류 대신 five-category 체계를 적용해 HF와 HR을 구별함으로써 ‘그럴듯한 오답’과 ‘정확한 유보’를 분해할 수 있게 했습니다.

- **Empirical Impact**: 실험 결과는 학습 데이터에서 상대적으로 덜 다뤄진 국가에서 모델 정확도가 체계적으로 낮아지는 패턴이 관측되며, 그 양상이 HF의 증가 또는 HR·QH·MF의 형태로 드러날 수 있음을 보여줍니다. 특히 단순 정답/오답을 넘어서 어떤 유형의 실패가 지배적인지(예: misattribution이나 qualitative hedging)를 파악함으로써, AI 거버넌스 분석에 필요한 검증·감사 프로토콜 설계에 실질적인 근거를 제공합니다. 재현 가능한 open-weight 모델과 공개된 ground-truth 데이터셋을 사용한다는 점에서 후속 연구와 벤치마크 확장에도 직접적인 영향을 줄 것으로 평가됩니다.



New uploads on arXiv(cs.RO)

### Scalable Behavior Cloning with Open Data, Training, and Evaluation (https://arxiv.org/abs/2606.27375)
Comments:
          30 pages. Project page: this https URL

- **Prior Approaches**: 기존 조작(Manipulation) 연구는 데이터·학습 파이프라인·하드웨어 설정이 공개되지 않아 재현이 어렵거나, 폐쇄형 또는 제한된 규모의 텔레옵션 데이터에 의존하는 경우가 많았다. 특히 실세계 평가 전에 모델·학습 설계를 충분히 검증하기 어려워 시행착오 비용이 컸다. 또한 Diffusion Transformers(DiT)와 Vision-Language-Action(VLA) 같은 최신 아키텍처 비교가 실제 성능으로 일관되게 연결되기 어려웠다.

- **Core Contribution**: 이 논문은 동작을 따라 학습하는 behavior cloning을 위한 완전 오픈소스 스택 ABC를 제안한다. 핵심 자산은 ABC-130K로, 195개 다양한 태스크에 대해 3,500시간, 130K 에피소드 규모의 텔레옵션 데이터셋을 제공한다. 아울러 하드웨어 셋업, 학습 인프라, 시뮬레이션 파이프라인을 공개하고, 400시간 sim-teleop 데이터와 함께 시뮬-실 평가 상관을 높이는 co-training 레시피를 제공한다.

- **Technical Challenges**: 주요 기술적 난제는 (1) 대규모 텔레옵션 데이터를 체계적으로 수집·정리하고 (2) 학습 설계를 바꿀 때마다 비싼 실세계 평가 없이도 성능 변화를 예측할 수 있게 만드는 것이다. 논문은 sim-teleop과 실세계 평가 간의 상관을 만들어주는 co-training 레시피로, 모델 설계·훈련 선택을 사전에 아블레이션할 수 있는 신뢰도 높은 프록시를 제공한다. 동시에 DiT와 VLA에 대한 다양한 학습 레시피와 아키텍처 선택지를 비교하고, 그 결과를 실세계 평가로 연결해 실용적인 가이드로 정리한다.

- **Empirical Impact**: 제공된 정책은 box folding, 지갑에서 신용카드 꺼내기처럼 정교한 dexterous 태스크를 실제로 성공적으로 수행했다. 또한 실세계 평가를 기반으로 DiT와 VLA의 일반적 설계·훈련 선택이 어떻게 성능에 영향을 주는지 경험적으로 근거를 제시한다. 연구 커뮤니티가 동일한 재현 가능한 도구를 바탕으로 비교·개발할 수 있게 되어, behavior cloning의 학습 기반을 ‘공동의 동등한 출발점’으로 확장한다는 점에서 의미가 크다.



### World Action Models Enable Continual Imitation Learning with Recurrent Generative Replays (https://arxiv.org/abs/2606.27374)
- **Prior Approaches**: 기존 연속학습(continual learning)은 순차 파인튜닝 시 catastrophic forgetting을 줄이기 위해 정규화(예: EWC), 아키텍처 분리, 또는 rehearsal/experience replay를 사용해 왔다. 특히 로봇에서는 이전 작업의 실제 데모를 저장해 재생하는 전략이 가장 강력하지만, 최신 로봇 foundation model/VLA(WAM 포함) 흐름에서는 proprietary 데이터 가정 때문에 이러한 “진짜 replay” 확보가 점점 어려워지고 있다.

- **Core Contribution**: 이 논문은 World Action Models(WAMs)의 생성형 인터페이스를 “메모리”로 활용해, 과거 작업의 실제 데모를 저장하지 않고도 pseudo-replay를 생성하는 Recurrent Generative Replay(REGEN)를 제안한다. REGEN은 이전 작업의 language instruction과 현재-task 관측만으로 WAM을 재귀적으로 질의해 과거 궤적을 합성하고, 이를 현재 작업 데모와 함께 behavioral cloning으로 학습해 망각을 억제한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 긴 horizon에서 재귀 생성되는 future visual observation의 품질이 저하되는 문제와 (2) 생성된 관측과 그 관측을 전제로 하는 action 사이의 불일치로, 실제 성공으로 이어지지 않는 경우가 생긴다는 점이다. 논문은 생성 길이를 goal-reward 기반 종료로 줄이고(초기 종료로 저품질 프레임 누적 방지), REGEN 학습에서 pseudo-trajectory를 현재 데이터와 혼합해 이전 능력의 보존 신호를 최대화하도록 설계했다.

- **Empirical Impact**: 실험은 시뮬레이션(LIBERO)과 실세계(xArm7 매니퓰레이션) 모두에서 수행됐고, REGEN은 순차 파인튜닝 대비 catastrophic forgetting을 최대 50%까지 감소시키며 privileged experience replay에 접근하는 성능을 보였다. 또한 표현 드리프트와 궤적 시각화를 통해 행동/표현을 비교적 잘 유지함을 보였고, 생성 replay의 성능 한계 요인으로 long-horizon visual degradation과 action-observation inconsistency를 실증적으로 확인했다.



### RouterVLA: Turning Smoke Tests into Supervision for Heterogeneous VLA Selection (https://arxiv.org/abs/2606.27355)
- **Prior Approaches**: 로봇 팀은 배치 전 commissioning(스모크 테스트)에서 후보 VLA(vision-language-action) 정책을 평가하고, 보통은 전체를 하나의 전역 점수로 압축해 평균 성과가 가장 좋은 정책을 채택해 왔다. 하지만 태스크·교란에 따라 전문가(폴리시) 간 강점이 달라 LIBERO 계열에서 관측되는 조건부 성능 정보를 전역 선택이 충분히 살리지 못한다.

- **Core Contribution**: RouterVLA는 commissioning 중에 얻은 trial 증거를 policy selection 감독 신호로 재사용하되, scored 실행의 outcome은 profile 구축에서 엄격히 분리(outcome-disjoint cross-fitting)하는 틀을 제안한다. 또한 포트폴리오에 있는 frozen expert들에 대해, probes로 만든 프로필을 바탕으로 다음 실행에 어떤 expert를 쓸지 선택하는 반복 사용 시나리오를 정형화한다.

- **Technical Challenges**: 핵심 기술적 난관은 프로필을 만들 때 scored trial의 성과가 새거나(누수), 혹은 동점 처리·후처리 과정에서 outcome 프록시가 섞여 “측정된 이득”이 부풀려지는 문제다. 저자들은 outcome separation을 과제 정의 자체로 고정하고, LIBERO-Plus ledger에서 trials를 분할해 프로필 생성과 점수 산정의 정보 경계를 기계적으로 강제했으며, 동작 길이·종료 양상 같은 scalar 이외 특징의 추가 가치도 진단했다.

- **Empirical Impact**: LIBERO-Plus 34,752 rollout 기록(398 variants, 28 experts)에서 전역 best는 held-out success 0.4686인데, expert당 3 probes로 만든 probe-success 룰은 0.6149로 +14.64pp 향상했다. logistic regression/GBDT/MLP도 비슷한 성능 대역을 보였고, scalar-only 프로필 조건에선 추가 scalar scorer 용량이 유의미한 선택 개선을 만들지 못했다. 한편 same-trial outcome 재사용은 측정 이득을 1.87× 부풀려 realized hindsight bound에 근접하므로, credible한 routing 성능 검증을 위해 outcome-disjoint 프로토콜이 필수라는 메시지를 남겼다.



### Continual Robot Policy Learning via Variational Neural Dynamics (https://arxiv.org/abs/2606.27353)
- **Prior Approaches**: 기존 learning-based control은 시뮬레이션에서 한 번 학습한 뒤 하드웨어에 고정 배치되는 경우가 대부분이었다. Domain randomization은 강건성을 올리지만 조건마다 하나의 보수적 정책으로 타협하기 쉽고, online residual-identification/online residual fitting은 현재 조건에 맞춰 재학습해야 해 조건이 다시 나타나면 효율이 떨어진다. 또한 많은 history-based conditioning은 시뮬레이션에서 잠재공간을 학습하거나, 배포 후 표현이 고정되어 continual improvement로 이어지기 어렵다.

- **Core Contribution**: 이 논문은 숨겨지고 반복되는 dynamics(예: 바람 방향 변화, 페이로드 변동, 배터리/하드웨어 열화)를 다루기 위한 continual learning 프레임워크를 제안한다. 실세계 state-action 궤적에서 condition-aware dynamics model을 학습하고, 이 잠재 조건을 이용해 정책을 differentiable simulation으로 계속 최적화한다. 배포 시에는 온라인으로 최근 상호작용을 통해 latent를 추론해 “인식(recognition)”으로 반복 조건을 복구하며, 매번 residual을 새로 맞추지 않아도 된다.

- **Technical Challenges**: 핵심 난제는 (1) 실세계에서 인코더가 뽑는 latent 분포와 (2) 정책 학습 단계에서 샘플링하는 latent 분포가 어긋나면 시뮬레이션에서 비현실적 dynamics가 생성된다는 점이다. 이를 위해 analytical physics prior 위에 neural residual을 얹되, GRU 기반 recurrent encoder가 최근 history로 hidden condition latent z를 추정하고 잔차 및 정책에 동시에 조건을 거는 구조를 만든다. 또한 인코더의 집합적 latent 분포를 sampling prior N(0,I)에 맞추기 위해 MMD를 사용하고, residual 예측 손실과 함께 context 재구성 항을 두어 임베딩 붕괴와 학습 불안정을 완화한다.

- **Empirical Impact**: 실험 결과는 quadrotor wind tracking에서 반복되는 바람 조건을 latent inference만으로 약 11초 내에 복구하며, online residual re-fitting 대비 회복 시간이 약 5배 빨라졌다고 보고한다. 더 큰 외란 환경에서 hover error와 tracking error가 각각 65.7%, 53.3% 감소해 최신 온라인 적응 접근 대비 개선폭을 보인다. 또한 실세계에서 비슷한 궤적을 추적하며 반복적으로 데이터 수집·업데이트를 수행한 continual refinement가 벽시계 시간에 따라 오차를 지속적으로 줄였고, cut propeller 같은 사전 예측이 어려운 hardware shift에서도 초기 반복 내 적응이 가능함을 보였다.



### Bridging Performance and Generalization in Reinforcement Learning for Agile Fligh (https://arxiv.org/abs/2606.27348)
- **Prior Approaches**: 자율 드론 레이싱에서 RL은 고정 트랙에서는 초인급 성능을 보였지만, 새로운 트랙에 적용하면 즉시 추락하는 경우가 잦아 zero-shot generalization(ZSG)이 깨지는 게 핵심 한계로 지적돼 왔습니다. 성능 저하를 감수하고 일반화를 높이려는 난이도 조절·커리큘럼/Task 생성 방식은 대체로 더 느린 비행을 요구했습니다. 또 Domain Randomization, ACCEL, PLR, PAIRED, ALP-GMM처럼 일반화 개선을 노린 기법들도 안전성과 고속 제어가 강하게 결합된 영역에서는 충분치 않았습니다.

- **Core Contribution**: 이 논문은 고속(시간 최적) 제어와 actuation saturation 하에서, 테스트 시 적응 없이도 unseen racetrack을 잘 달리는 RL 기반 ZSG 프레임워크를 제안합니다. 핵심은 task-aware switching(학습 진행도 기반)과 physically informed procedural track generator(물리 제약을 반영한 트랙 생성)를 결합해, 빠른 속도는 유지하면서도 일반화 성능을 끌어올리는 데 있습니다. 특히 end-to-end 비전 입력(명시적 state estimation 없이)에서도 prior approach가 실패하는 조건에서 zero-shot 배치를 달성했다고 강조합니다.

- **Technical Challenges**: ZSG를 망치는 원인은 고차원 트랙 변동과 안전·성능의 강한 결합으로, 기존처럼 임의/단순 샘플링이나 순차 커리큘럼을 쓰면 과적합·catastrophic forgetting이 발생하기 쉽습니다. 논문은 UPOMDP 관점으로 작업을 태스크 분포로 모델링하고, 여러 트랙에서 병렬로 rollouts를 수집해 업데이트하며, 각 태스크의 reward plateau 여부를 Spearman 상관 또는 Kalman 필터로 추정해 “학습 신호가 꺼진 태스크”를 자동으로 교체하도록 설계했습니다. 트랙 생성도 B-spline 기반의 feasible 곡선을 사용해 게이트를 접선/아크길이 간격으로 배치하며, 이렇게 만든 태스크 품질과 다양성을 함께 확보합니다.

- **Empirical Impact**: 실험 결과 제안 프레임워크의 state 기반 일반화 성능은 기존 SOTA 대비 7.4배 향상(competitive speed도 함께 유지)되며, 실제 하드웨어에서도 4개 core 트랙을 100% 성공률로 완주했다고 보고합니다. 속도 측면에서는 EaP 대비 약 37.73% 더 빠르고, ST(단일 트랙 특화)보다도 큰 손해 없이 약 14.52% 느린 수준에 머물렀습니다. 비전 기반 end-to-end 설정에서도 ZSG가 처음으로 입증됐고, vision-based 일반화 지표가 state-based EaP보다도 2.3배 우수하다고 제시합니다.



### VibeAct: Vibration to Actions for Contact-Rich Reactive Robot Dexterity (https://arxiv.org/abs/2606.27344)
- **Prior Approaches**: 덱스터러스 조작에서 접촉과 슬립 같은 이벤트는 빠르고 국소적이며, 시각으로는 가려지기 쉬워 촉각 센서가 중요하다. 기존 비전 기반 촉각은 고해상도 정보를 주지만 다중 손가락에서 부피·조명·계산 부담이 크고, 압전 마이크를 쓸 때도 오디오를 시뮬레이션에 그대로 맞추는 것이 어려워 end-to-end sim-to-real 학습이 막힌다.

- **Core Contribution**: VibeAct는 압전 마이크로부터 접촉과 슬립을 직접 추정해, 이를 시뮬레이션에서도 동일한 물리적 표현으로 계산해 policy 학습에 쓰는 프레임워크다. 핵심은 raw audio를 정책에 투입하지 않고, contact onset(이벤트), slip presence(존재), slip magnitude(연속 크기)의 저차원 shared physical representation로 센싱 문제와 제어 문제를 분리한 점이다. 디지털 클론에서 텔레오퍼레이션 데이터를 자동 라벨링해 추정기를 학습시키고, RL은 동일 표현으로 시뮬레이션에서만 학습한 뒤 배치한다.

- **Technical Challenges**: 오디오 파형은 재료·장착 위치·증폭 체인·배경 진동 등 요인의 영향을 받아 정확한 진동 시뮬레이션이 난제다. VibeAct는 대신 시뮬레이터의 접촉/슬립 동역학으로 물리적 라벨을 만들고, per-finger log-mel spectrogram 입력으로부터 estimator가 세 채널(접촉/슬립 존재/슬립 크기)을 예측하게 해 sim-to-real 격차를 표현 레벨에서 줄인다. 또한 접촉 onset은 희소 이벤트라 가중 BCE와 multi-task 학습을 쓰고, slip magnitude는 슬립 구간에만 Huber loss를 적용해 연속 채널 학습을 안정화했다.

- **Empirical Impact**: 다섯 가지 contact-rich 태스크(재그리핑, 인핸드 재조향, 삽입 등)에서 VibeAct는 proprioception-and-point-cloud baseline을 일관되게 능가했으며, 특히 지속적인 reactive control이 필요한 과제에서 가장 큰 향상을 보였다. 채널 기여 분석에서는 slip magnitude 채널을 추가했을 때 성능 점프가 가장 컸고, contact onset 단독은 일관된 이득이 제한적이었다. 학습된 정책은 물리 로봇 플랫폼으로 전이되어 성공률이 개선됐으며, 즉 raw audio 시뮬레이션 없이도 고대역폭 촉각 피드백을 sim-to-real에 효과적으로 통합할 수 있음을 보여준다.



### LA4VLA: Learning to Act without Seeing via Language-Action Pretraining (https://arxiv.org/abs/2606.27295)
Comments:
          Github: this https URL

- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 모델은 로봇 시연에서 시각(visual)과 언어 지시를 함께 받아 행동(action)을 예측하는 방식으로 사전학습해 왔습니다. 그런데 데이터·입력 비대칭 때문에 시각-행동(supervision)이 언어-행동 신호보다 촘촘하고 강하게 작용해, 정책이 언어의 조건을 학습하기보다 시각적 지름길(visual shortcut)에 의존할 수 있습니다.

- **Core Contribution**: 이 논문은 시각 없이도 언어가 조건하는 행동 사전지식(language-conditioned action priors)을 학습하도록 하는 LA4VLA를 제안합니다. 핵심은 시연 궤적을 원자적(atomic) 행동 구간으로 분해해 각 구간에 해당하는 low-level action description(저수준 행동 설명)을 붙이고, 그 결과로 Vision-agnostic Language-Action(LA) 학습 데이터를 만들어 VLA 학습에 보강하는 것입니다.

- **Technical Challenges**: 언어-행동 정렬이 약해질 수 있는 문제를 해결하려면, 시연에서 원자적 구간을 신뢰성 있게 분해하고 시각 의존 없이도 일관된 문장으로 라벨링해야 합니다. 논문은 robot state 기반 keyframe 힌트와 action vocabulary를 프롬프트로 삼아 VLM이 구간 후보를 생성하게 한 뒤 인간 검증으로 시간경계·라벨·언어의 일치를 점수 임계값으로 정제하여 LA4-33K(33K 에피소드)를 구축하고, 이를 기반으로 LA-only, LA-to-VLA, mixed LA-VLA 등 사전학습 패러다임을 비교합니다.

- **Empirical Impact**: 실험에서 LA-pretrained 정책은 VLA만 사전학습한 대응군보다 시뮬레이션과 실세계 과제에서 일관되게 높은 성공률을 보였고, 특히 mixed LA-VLA는 기준선 대비 시뮬레이션에서 최대 17.8%p, 실세계에서 45.0%p까지 개선했습니다. 또한 시각 교란(perturbations)이나 시각-언어 불일치(conflict) 상황에서도 언어 조건을 더 잘 따르는 경향이 관찰돼, LA4VLA가 강건하고 보완적인 VLA 사전학습 전략임을 실증합니다.



### BOWConnect: Parallel Bayesian Optimization over Windows with Learned Local Cost Maps for Sample-Efficient Kinodynamic Motion Planning (https://arxiv.org/abs/2606.27292)
Comments:
          Accepted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)

- **Prior Approaches**: 킨오다이내믹 모션 플래닝은 로봇 상태공간(자세+속도 등)에서 탐색해야 해 차원의 저주로 표본 효율이 떨어지기 쉽다. 또한 동역학 제약 때문에 기하학적으로 충돌이 없는 경로라도 실행 불가능할 수 있어, 단순 거리 기반 휴리스틱은 비용 추정이 불안정해졌다.
기존 샘플링 기반 방법은 제어를 랜덤으로 뽑아 전진 전파하는 방식이 많고, 좁은 통로(narrow passage)·비볼록 공간에서 연결이 잘 안 되며 성능이 크게 저하되는 문제가 남아 있었다. 병렬화는 RRT류에선 잘 됐지만, 전파가 연산의 대부분을 차지하는 킨오다이내믹에서는 확장 효율이 제한되어 왔다.

- **Core Contribution**: BOWConnect는 bidirectional parallel kinodynamic motion planner로, 기존 샘플링 방식의 세 가지 한계(고차원 비효율, 동역학 제약 하 비용 휴리스틱 불신, 좁은 통로 성능 저하)를 동시에 겨냥한다. 랜덤 제어 샘플링 대신 Bayesian Optimization over Windows(BOW)를 steering function으로 넣어, 충돌 회피성과 동역학 타당성을 만족하는 국소 제어를 연속 상태·제어공간에서 직접 생성한다.
또한 시작/목표에서 각각 트리를 병렬로 키우는 bidirectional 구조를 사용해 전역 연결성을 높이면서, 두 트리의 연결 후보를 빠르게 찾도록 MotionTree 내부 spatial hashing을 결합했다. 연결 시에는 kinodynamically consistent bridge를 만들기 위해 boundary value problem(BVP) 솔버를 사용한다.

- **Technical Challenges**: BOW를 킨오다이내믹 플래닝에 넣으려면, 제어공간에서 학습(보상/제약)과 시뮬레이션이 반복되는 동안 계산량을 감당하면서도 좁은 통로에서 안전한 탐색을 유도해야 한다. BOWConnect는 reward와 constraint를 각각 GP로 모델링하고, constrained acquisition 함수에 feasibility(충돌 가능성)를 곱해 불가능한 영역으로 탐색이 쏠릴 때 자동으로 우회하도록 설계했다.
또한 병렬로 트리를 여러 개 키우면 연결 검증이 병목이 되기 쉬운데, MotionTree의 spatial hashing으로 O(1) 평균 연결 질의를 목표로 하며 다단계 검증(먼저 kinematic feasibility, 그다음 BVP로 다리 생성)을 수행한다.
다리 생성 후에도 충돌 체크와 연결 선택(여러 후보 중 유클리드 거리 최소)을 통해 품질을 안정화하며, 연결 실패 시 단방향 해를 반환하는 graceful degradation도 포함한다.

- **Empirical Impact**: 10개 벤치마크(유니사이클·바이시클 모델 포함)에서 BOWConnect는 100% success를 달성하면서 대체로 최단 또는 근접 최단 planning time을 보였다. 특히 narrow passage와 비볼록 공간에서 기존 state-of-the-art 플래너들이 성능이 크게 떨어지는 상황에서도 빠르고 안정적인 연결을 보여 준 것이 핵심 관측이다.
예컨대 일부 좁은 통로 시나리오에서 RRT·SST·KPIECE·BOW 대비 실행 시간이 현저히 짧아지거나(유니사이클) 성공률 저하가 나타나는 구간에서도 BOWConnect가 견고하게 통과했다. 더 나아가 ground vehicle와 quadrotor 실배치 실험에서 충돌 없이 실시간(real-time) 수준의 계획이 확인되며 연구 성과의 실용성이 강조됐다.



### E-TTS: A New Embodied Test-Time Scaling Framework for Robotic Manipulation (https://arxiv.org/abs/2606.27268)
Comments:
          Accepted to ECCV 2026. 44 pages, 11 figures. Project page: this https URL

- **Prior Approaches**: 기존의 embodied test-time scaling(TTS) 연구는 주로 action 공간만 늘려 성능을 끌어올리는 데 집중해 왔습니다. 또한 reasoning 구성요소가 성능을 돕더라도, reasoning이 어떻게 스케일되는지(스케일 메커니즘)는 상대적으로 덜 다뤄졌습니다. 더불어 로봇 조작은 long-horizon·순차 의사결정이라 과거 맥락이 중요한데, 기존 TTS는 history와 피드백을 충분히 통합하지 못했습니다.

- **Core Contribution**: 본 논문은 reasoning과 action을 하나의 단위로 함께 스케일하는 plug-and-play 프레임워크 E-TTS를 제안합니다. E-TTS는 history buffer와 vision-language verifiers로 과거 맥락을 반영한 후보 평가를 수행하고, 샘플링 과정에 feedback을 생성해 closed-loop iterative refinement를 만듭니다. 결과적으로 서로 다른 vision-language-action(VLA) 모델에도 추가 학습 없이 모듈을 끼워 넣어 성능을 개선할 수 있도록 설계됐습니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) reasoning 결과와 action 생성이 강하게 결합돼 있으므로 둘의 정렬이 필요하고, (2) long-horizon에서 현재 관측만으로는 후보 검증의 기준이 부족하다는 점입니다. E-TTS는 reasoning-action joint sampling 및 pairwise joint scoring(Reasoning verifier의 zero-shot 점수 + Action verifier 점수)을 통해 최적 ⟨reasoning, action⟩ 쌍을 찾게 하고, history buffer로 시간 의존성을 입력·검증에 반영합니다. 또한 일정 임계값 미달 시 verifier가 실패 원인을 구조화된 텍스트 feedback으로 만들고, 이를 다음 라운드 프롬프트에 주입해 반복적으로 개선합니다.

- **Empirical Impact**: E-TTS는 4개 벤치마크·6개 환경·3개 embodiment에서 4종 VLA 모델(E-CoT, MolmoAct, π0.5, Embodied-R1)에 적용되어 일관된 성능 향상을 보였습니다. 추가 expert 데이터 수집이나 재학습 없이 시뮬레이션에서 최대 33.14%, 실세계에서 26.62%까지 성공률이 개선되며, 평균 향상도 유의미하게 나타났습니다. 또한 ablation 결과 reasoning scaling과 action scaling, feedback·history 설계가 함께 중요하며 단일 요소만 적용할 때 성능이 크게 떨어지는 점을 확인했습니다.



### Advancing Omnimodal Embodied Agents from Isolated Skills to Everyday Physical Autonomy (https://arxiv.org/abs/2606.27251)
- **Prior Approaches**: 기존 연구는 사이버 영역(API·IoT)과 물리 영역(조작·이동)을 따로 다루거나, VLM 기반 플래너가 통합된 cyber-physical action space를 충분히 지원하지 못했다. 또한 에이전트 프레임워크가 맥락을 무한히 누적해 temporal coherence가 떨어지고, VLA 정책은 open-loop로 실행해 자신의 물리적 실패를 스스로 감지·수정하지 못하는 문제가 있었다.

- **Core Contribution**: 이 논문은 persistent autonomy를 위해 단일 거대 모델이 아니라 planning, memory, verification을 명시적으로 분리한 계층적 비동기 아키텍처가 필요하다고 주장한다. 그 구현으로 OmniAct를 제안하며, unified action space에서 skill routing을 위한 multimodal semantic planner, 이벤트 경계 기반 압축으로 context 성장을 억제하는 계층형 메모리, 그리고 물리 실행 중 시각 기반 preemption으로 semantic loop를 닫는 엔진을 통합한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 이질적인 사이버·물리 도구를 하나의 실행 공간으로 조율하고, (2) 장시간 운용 시 context 누적이 시간적 일관성을 해치지 않게 하며, (3) open-loop 실행에서 벗어나 물리 실패를 감지해 즉시 복구하는 폐루프를 만드는 것이다. OmniAct는 통합 action space에서의 semantic planner로 실행 경로를 라우팅하고, 이벤트 경계-driven compression으로 메모리를 압축하며, 비동기 visual preemption engine으로 시각적 검증과 선제 중단을 통해 의미적 재계획을 수행한다.

- **Empirical Impact**: OmniAct는 두 개 로봇 플랫폼에서 4개의 IoT 디바이스를 함께 다루는 40개의 실제 장기 태스크에서 복잡도 전 구간에 걸쳐 end-to-end success가 일관되게 개선됐다. 또한 누적 interaction tokens가 100k+를 넘는 상황에서도 토큰 사용량이 거의 평탄하게 유지됐고, mid-scale open-weight 모델의 성능을 proprietary-level에 가깝게 끌어올렸다는 점에서 실용적 의미가 크다.



### HumanoidUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation (https://arxiv.org/abs/2606.27239)
Comments:
          8 pages, 7 figures

- **Prior Approaches**: 기존 휴머노이드 학습용 데이터 수집은 대부분 robot-in-the-loop 텔레오퍼레이션에 의존해 왔습니다. 덕분에 구현 일치도가 높지만, 로봇 접근성·숙련된 운영자·안전감독·반복 조작 부담 때문에 대규모 수집이 비싸고 비효율적이었습니다. UMI 계열과 같은 robot-free 접근도 로봇암 중심에는 유리했지만, 휴머노이드는 균형·자세 보정·보행·형상 의존 전신 모션 때문에 그대로 확장하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 휴머노이드 전신(whole-body) 기술을 위해 HumanoidUMI라는 휴대형 robot-free 데이터 수집-학습 프레임워크를 제안합니다. VR 기반의 손목 시점 관측(wrist-view), 인체 의도 핵심점(sparse keypoint trajectories), 그리퍼 행동/상태를 모아 고수준 정책이 미래 핵심점을 예측하고, 이를 SKR(Spatial Keypoint Retargeting)로 로봇 네이티브 기준에 정합시킨 뒤 전신 컨트롤러가 실행합니다. 즉, “인간-로봇 형상·기구 차이를 명시적으로 다리 놓는 스패셜 인터페이스”가 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 도전은 인간 핵심점 궤적을 로봇의 체형·관절 제한·작업공간에 맞게 변환해 물리적으로 실행 가능한 기준으로 만드는 것입니다. HumanoidUMI는 상체 관절을 직접 옮기지 않고 골반/발/손목 TCP 중심의 sparse 공간 구조를 유지하되, 다리 관련 핵심점에 대해 pelvis-local anisotropic adjustment로 스케일 불일치를 보정한 뒤 constrained weighted inverse kinematics로 로봇 기준을 생성합니다. 또한 동적 동작에서는 관측·그리퍼 신호·로봇 제어 간 latency matching을 수행하고, Diffusion Policy 계열의 receding-horizon 예측이 시간 동기된 키포인트 덩어리를 생성하도록 설계했습니다.

- **Empirical Impact**: Unitree G1에서 5개 실세계 시나리오(단일/양손 조작, 동적 볼 던지기, 전신 굽힘, 보행-조작 결합)를 통해 robot-free 시연이 배치 가능한 휴머노이드 스킬로 전환됨을 보였습니다. 특히 under-table waste disposal과 walking coffee delivery처럼 하체가 조작에 직접 결합되는 과제에서, sparse 키포인트가 보행·무릎 굽힘·자세 변화를 함께 유도하는 정성적 결과를 냈습니다. 데이터 수집 효율도 텔레오퍼레이션 baseline 대비 평균 약 2.2배 수준으로 증가했으며, 특히 초보 운영자와 보행-조작(walking delivery)에서 격차가 크게 나타나 확장 가능성(스케일링)에 의미가 있습니다.



### Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline) (https://arxiv.org/abs/2606.27163)
Comments:
          Solution of the LeHome Challenge at ICRA 2026

- **Prior Approaches**: 기존 연구/대회 방식은 주로 행동복제(BC)로 시작해 성능을 끌어올리거나, PPO 계열처럼 로그확률 기반 정책그래디언트를 flow-matching 기반 VLA에 그대로 적용하려는 시도가 많았다. 하지만 BC 데이터는 보통 “성공 궤적” 위주라 실패 회복 능력이 약하고, PPO류는 flow-matching의 유효 행동 manifold 밖으로 예측이 밀려나기 쉬워 안정적인 개선이 어렵다. 또한 희소한 이진 보상에서는 중간 성과 신호를 엔지니어링해야 해 학습이 더 까다롭다.

- **Core Contribution**: 이 논문은 LeHome Challenge 2026의 양팔 의류 접기에서, VLA 정책을 강화학습 루프로 개선하되 “정책이 곧 value function이 되게” 만든 것이 핵심 기여다. 즉 행동을 예측하는 동일 네트워크가 성공/진행/미래 핵심점 거리 등의 양을 함께 예측하고, 이 예측을 advantage 추정, 실시간 failure detection, 후보 선택에 직접 활용한다. 결과적으로 한 모델로 학습·추론 신호를 통합해 서빙/튜닝 복잡도를 줄이면서도 성공률을 끌어올렸다.

- **Technical Challenges**: 문제는 (1) 변형 물체의 미세한 궤적 차이가 상태를 크게 바꾸는 점, (2) 관측 가능한 중간 보상이 거의 없고 성공이 이진인 점, (3) 평가 시 의류 종류 라벨이 주어지지 않는 점, (4) 시뮬-실세계 격차가 큰 점이다. 저자는 AWR+RECAP을 결합해 좋은 구간에 확률질량을 재배치하면서도 flow-matching의 유효 예측공간을 벗어나지 않게 했고, RECAP 방식 advantage conditioning으로 inference-time에서도 guidance 형태의 선택을 가능하게 했다. 더불어 시뮬에서의 강한 데이터 증강, success/ failure 상태 스냅샷 기반 하드 마이닝, Thompson sampling을 통한 inference-time 하이퍼파라미터 탐색, 카메라 정렬 도구를 포함한 sim-to-real 파이프라인과 DAgger-like HIL 수집으로 회복/일반화 난제를 완화했다.

- **Empirical Impact**: 실험 결과, 온라인(시뮬) 라운드에서 62팀 중 1위를 차지했으며 전체 성공률 79.63%로 2위와도 큰 격차를 보였다. 실세계 결선에서는 시뮬-실세계 전환 튜닝을 수행한 끝에 2위를 기록해 실제 로봇에서도 성능을 유지함을 입증했다. 특히 “정책 내부의 value/Q 대체와 advantage 기반 라이브 실패 감지”를 실제 대회 수준으로 적용한 레시피로, 변형물체 조작 VLA 강화학습 방향에 실용적 기준점을 제시했다.



### PhysReflect-VLA: Physical Feasibility and Self-Reflective Regulation for Reliable Vision-Language-Action Policies (https://arxiv.org/abs/2606.27146)
- **Prior Approaches**: 기존 VLA(vision-language-action) 모델은 시각·언어를 기반으로 행동을 생성하지만, 실행 시점에는 예측된 행동이 물리적으로 가능한지 실시간 검증을 거의 하지 않는다. 그 결과 접촉 제약이나 기하적 한계를 위반하는 전이가 발생하면 작은 오차가 장기 구간에서 누적되며 성능이 급격히 저하될 수 있다. 또한 실행 중 예측-실제 상태 불일치를 진단하고 다음 결정을 구조적으로 교정하는 self-reflection(자기성찰) 메커니즘이 부족해 실패가 반복되기 쉽다.

- **Core Contribution**: 이 논문은 PhysReflect-VLA로, pretrained VLA 정책을 건드리지 않고도 execution-time에서 신뢰성을 높이는 plug-and-play 프레임워크를 제안한다. 핵심은 closed-loop 파이프라인에 물리 타당성 평가(Feasibility Operator, Action Explanation Operator)와 불일치 기반 self-reflection(Reflection Module)을 결합해, 불가능 전이는 억제하고 실패 시에는 교정 가이드를 생성해 다음 행동을 수정하는 것이다. 이를 통해 VLA 실행을 feed-forward 방식에서 물리 일관성 점검과 온라인 교정이 가능한 형태로 전환한다.

- **Technical Challenges**: 문제는 (1) 장기 조작에서 물리적으로 일관된 전이를 계산·평가해야 하고, (2) 실행 후 불일치를 해석해 의미 있는 교정 지침으로 변환해야 한다는 점이다. 저자들은 추상 상태 공간에서 forward model과 inverse model의 cycle-consistency로 consistency energy를 정의해 후보 행동의 동역학 예측 가능성과 self-explainability를 동시에 점검하고, 불일치가 임계값을 넘으면 Reflector가 교정 토큰을 생성한 뒤 instruction을 보강해 resampling(재샘플링)을 유도한다. 또한 feasibility 모델과 reflection이 제어 루프에 안정적으로 통합되도록 two-stage training으로 먼저 feasibility를 보정·고정한 뒤, 그 위에 reflection 학습을 얹는 절차를 사용한다.

- **Empirical Impact**: 실세계 7-DoF 로봇에서 5가지 contact-rich, multi-stage 장기 조작 작업(Table-Bussy, Drawer-Cycle, Lid-Open, Shelf-Insert, Part-Assembly)을 20회씩(시드 5개) 평가한 결과, 평균 성공률이 74.2%(기본 VLA)에서 79.6%(full)으로 5.4%p 향상됐다. 또한 OpenVLA와 OpenVLA-OFT 계열 모두에서 각각 74.2→79.6%, 82.0→85.0%처럼 일관된 개선을 보였고, ablation에서 feasibility 기반 필터링과 reflection 기반 교정이 각각 견고성 향상에 기여함이 확인됐다. 특히 cycle-consistency 학습을 제거하면 성능이 크게 하락해, 물리 타당성 에너지가 제대로 보정될 때만 신뢰성 이득이 극대화된다는 점을 시사한다.



### PAMAE: Phase-Aware-MoE Action Experts Towards Reliable Flow-Matching Vision-Language-Action Policies (https://arxiv.org/abs/2606.27144)
- **Prior Approaches**: VLA(vision-language-action) 기반 flow-matching 정책은 멀티모달 정합성과 연속 제어 생성에 강점을 보이지만, 대부분 단일 shared action expert를 써서 다단계 조작의 단계별 제어 패턴을 분리해 학습하기 어렵다. 또한 MoE(mixture-of-experts) 접근도 주로 태스크 의미나 스케일링에 초점을 두고, low-level action 생성에서 실행 phase 구조를 라우팅에 직접 반영하지 못하는 한계가 있었다.

- **Core Contribution**: 이 논문은 pretrained flow-matching VLA 백본은 유지하면서, shared action expert를 phase-aware sparse MoE action module(PAMAE)로 교체한다. PAMAE는 lightweight phase prediction head와 routing alignment objective를 통해 실행 단계 힌트를 바탕으로 전문가를 단계 일관적으로 배분해 phase-consistent action generation을 노린다.

- **Technical Challenges**: 핵심 난제는 전문가를 늘리는 것만으로는 부족하고, phase별로 서로 다른 velocity/민감도 요구를 정확히 반영하는 라우팅이 필요하다는 점이다. 저자들은 전문가 모듈을 먼저 flow-matching 손실로 warm-up한 뒤, phase-consistent 라우팅 감독을 도입하고 학습 후반에는 보조 라우팅 제약을 anneal해 특화와 안정성을 동시에 확보한다.

- **Empirical Impact**: 시뮬레이션 다단계 조작 5개 태스크에서 PAMAE는 강력한 VLA baseline 대비 최대 9.2%p의 task success 향상을 보이며, 단계 라우팅 감독과 two-stage 최적화가 개선의 필수 요소임을 ablation으로 확인했다. 또한 inference 시에는 별도 phase label 없이 라우터가 스스로 전문가를 선택하되, 지배 전문가의 연속 길이와 phase-conditioned dominance purity(PCP) 관점에서 시간적 일관성과 단계 정합성이 관찰된다.



### Proposal-Conditioned Latent Diffusion for Closed-Loop Traffic Scenario Generation (https://arxiv.org/abs/2606.27123)
Comments:
          Accepted for publication at the IEEE International Conference on Intelligent Transportation Systems (ITSC), 2026

- **Prior Approaches**: 기존 확산(diffusion) 기반 시나리오 생성은 현실감은 높일 수 있지만, 안전과 같은 안전-critical 편집(제어)을 롤아웃 전반에서 안정적으로 다루기 어렵다는 한계가 있다. 또 적대적(adversarial) 생성은 충돌/근접회피 같은 위험 사건을 잘 만들 수 있으나, 도로 이탈·운동학적 제약 위반처럼 그럴듯함(plausibility)이 떨어질 수 있다. 반면, 지도 일관성과 상호작용의 closed-loop 일관성을 지키면서도 재계획(replanning) 루프에서 빠르게 샘플링하는 효율성은 여전히 병목이다.

- **Core Contribution**: 이 논문은 instance-centric 장면 컨텍스트와 멀티모달 per-agent 제안 분포를 조건으로 하는 diffusion 기반 scenario generation 프레임워크를 제안한다. 핵심은 단순히 에이전트를 독립적으로 미래를 합성하는 게 아니라, joint interaction을 명시적으로 모델링해 장면-일관성과 controllability를 동시에 노리는 점이다. 또한 test-time guidance를 선택적으로 적용해 안전-critical 행동을 원하는 목표로 “수정”할 수 있게 하되 재학습 없이 운용한다.

- **Technical Challenges**: 문제의 기술적 난제는 (1) 다중 에이전트의 reverse sampling이 많은 순차 단계로 인해 closed-loop 재계획 지연을 유발한다는 점, (2) 다중 에이전트에 대한 guidance가 계산량을 늘리고 운동학적으로 불일치한 궤적을 만들 수 있다는 점이다. 저자들은 제안(proposal)에서 출발하는 shifted Gaussian 초기화와 PCA로 만든 compact action-latent 표현을 통해, DDIM 등 few-step reverse diffusion가 가능하도록 시작 분포와 추론 경로를 바꾸었다. 여기에 differentiable map 위반·충돌 근접도·게임이론 기반(추격-회피) 목적을 latent 공간 guidance로 결합해 realism–stress-testing trade-off를 test-time에서 조절한다.

- **Empirical Impact**: Waymo Open Motion Dataset(WOMD)에서 모델은 다양한 상호작용 시나리오에 대해 현실감, 안전, controllability의 균형이 유리하게 나타났고, test-time guidance는 competing objectives 사이의 체계적 트레이드오프를 가능하게 했다. 무가이드(unguided) 대비 충돌 및 오프로드(off-road) 같은 안전 지표가 유의미하게 개선되면서도, 현실감과 위치 정확도는 크게 훼손되지 않았다. 또한 ablation과 단계 수(step count) 실험에서 proposal-informed 초기화와 few-step 샘플링이 런타임을 줄이면서도 품질을 유지하는 경향이 확인되어, 실제 AV 시뮬레이션/검증 파이프라인에 실용적 의미가 크다.



### ForesightSafety-VLA: A Unified Diagnostic Safety Benchmark for Vision-Language-Action Models (https://arxiv.org/abs/2606.27079)
Comments:
          8 pages, 5 figures, 4 tables. Submitted to IROS 2026

- **Prior Approaches**: 기존 VLA 평가는 작업 성공률, 충돌 여부 같은 엔드포인트 중심 지표에 기대거나 안전을 ‘강건성/일반 실패’의 일부로만 다뤄 안전이 실제로 어떻게 발생하는지(과정 수준)를 가리기 쉬웠다. 시뮬레이션 기반 안전 관련 연구도 보통 비용 누적이나 제한 조건에 초점을 두지만, VLA의 인지-언어-제어 전체 루프에서 안전 실패 원인을 분리해 진단하기는 어렵다.

- **Core Contribution**: 본 논문은 VLA의 안전을 1차 평가 목표로 전면에 둔 진단 벤치마크 ForesightSafety-VLA를 제안한다. 물리 상호작용 안전(Safe-Core), 지시/언어 측 안전(Safe-Lang), 지각 측 안전(Safe-Vis)으로 나눈 13카테고리 안전 분류와, 구조·언어·시각 변화 3축을 조합해 실패 원인을 숨기지 않고 추적한다.

- **Technical Challenges**: 핵심 기술적 과제는 ‘안전’을 단순 이진 충돌 체크로 두지 않고 시나리오 수준에 구조적으로 내장하며, 프로세스에서의 위험을 정량화하는 것이다. 이를 위해 RoboTwin에서 66개 안전-강화 base scenario를 수동 조합(위험물/영역 추가, 동작 가능 구역 제한, 시간 전제조건 삽입)하고, 상태별 듀얼-스레시홀 모니터링으로 누적 안전 비용(CC)과 위험 노출 시간(RET)을 함께 측정하며 safe/unsafe 성공·실패 4분면으로 결과를 분해한다.

- **Empirical Impact**: 실험 결과, 가장 강한 모델이라도 누적 안전 비용과 unsafe nominal success가 모두 0이 아니며 안전이 여전히 ‘겉치레’에 가깝다는 점이 확인됐다. 특히 구조(배치/클리어런스/시간 결합)와 시각(조명·뷰포인트·가림·어드버설 패치) 변화가 언어 변형보다 안전 저하를 훨씬 크게 만들었고, 약한 모델은 ‘더 보수적으로’ 실패하는 경향보다 ‘더 위험하게’ 실패하는 패턴이 두드러졌다. 저자들은 현 안전 격차가 post-hoc 안전 필터링만으로는 줄이기 어렵고, perception·grounding·control competence의 내재 역량과 강하게 결부된다고 해석한다.



### RelAfford6D: Relational 6D Affordance Graphs for Constraint-Driven Robotic Manipulation (https://arxiv.org/abs/2606.27036)
- **Prior Approaches**: 기존 연구는 언어-비전-제어 간 격차를 줄이기 위해 대규모 데이터 기반 정책이나 foundation model을 활용해왔지만, 표현이 고립된 접점 좌표나 잠재 affordance 임베딩에 머무는 경우가 많았다. 2D 픽셀 affordance는 3D 의미가 부족하고, 3D point cloud나 keypoint 방식은 연속적인 상대 형상(회전축 등)과 강한 강체/관절 제약을 명시적으로 담지 못한다. 또한 dense descriptor/implicit energy 기반 접근은 표현력은 높아도 관절 한계 같은 hard kinematic constraint가 결여되어 정형화된 관절 조작에서 불리하다는 한계가 지적된다.

- **Core Contribution**: RelAfford6D는 오픈월드 로보틱스 조작에서 ‘추상 의미’와 ‘정밀 물리 제어’를 잇기 위해 Relational 6D Affordance Graph(관계형 6D affordance 그래프)를 중간 표현으로 제안한다. 자연어 인스트럭션으로부터 주 작동 파트와 물리 앵커 파트를 잇는 의미 위상(topology)을 추론하고, 이를 SE(3)에서의 상대 기하 관계로 승격해 실행이 kinematic constraint satisfaction 문제로 정식화되게 만든다. 학습 없이도 closed-loop로 연속 궤적을 생성하면서, 개방/회전/슬라이딩 같은 관절형 조작을 견고하게 수행하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 LLM이 산출하는 무제약 의미 추론을, 관절형 조작에 필요한 엄밀한 kinematic 제약 구조와 정확한 6D 포즈로 변환하는 ‘semantic–geometric gap’이다. 이를 위해 PartNet-Mobility 기반의 구조화된 kinematic ontology를 retrieval-augmented generation(RAG)으로 연결해, 질문마다 (primary part, anchor part, 관절 manifold 선택 토큰)를 zero-shot으로 유도한다. 이어 SAM3/ FoundationPose 계열의 비전 foundation model로 부품 마스크를 3D로 정밀 추정해 SE(3) 포즈 경계조건을 만들고, 회전(revolute)·이송(prismatic) 궤도를 1D 제약 매니폴드 위에서만 tracking하며, 외란 발생 시 실시간으로 재인스턴스화(replanning)한다.

- **Empirical Impact**: SAPIEN 시뮬레이터와 로봇 실환경에서 관절 조작/강체 조작 벤치마크를 각각 평가했으며, 학습이나 시연 없이 zero-shot으로도 기존 data-driven baseline보다 높은 성공률과 실행 견고성을 보였다고 주장한다. 특히 closed-loop 제약 오차 모니터링을 통해 앵커 자세가 흔들려도 남은 구간에 대해 수학적으로 재계산한 SE(3) 궤적을 생성해 성능 저하를 줄였다. 결과적으로 ‘관계형 affordance 그래프 + 하드 kinematic constraint tracking’이 오픈월드 일반화에 유리하다는 실증 근거를 제시한 점이 의미 있다.



### In-Context Model Predictive Generation: Open-Vocabulary Motion Synthesis from Language Models to Physics (https://arxiv.org/abs/2606.26981)
- **Prior Approaches**: 기존 텍스트-투-모션은 대규모 human motion capture 데이터에 크게 의존해, 훈련에 없던 문장(오픈 보컬러리)으로 갈수록 일반화가 약해지는 문제가 있었다. 오픈 보컬러리 방향에서는 CLIP 기반 정렬이나 LLM을 활용한 high-level 계획이 등장했지만, LLM의 계획이 물리 제약을 자동으로 만족시키지 못해 물리적으로 그럴듯하지 않은 모션이 자주 나온다. PhysDiff, AnySkill 같은 physics-aware 접근은 현실성을 높이지만, 미세한 지시와 새로운 개념에서 의미 정합성(semantic fidelity)이 흔들리는 ‘semantic brittleness’가 한계로 지적된다.

- **Core Contribution**: ICMPG(In-Context Model Predictive Generation)는 언어 기반 계획과 추론 시점 물리 피드백을 결합해, 의미 충실도와 물리 현실성의 상충을 줄이는 프레임워크를 제안한다. CAMG(Context-Aware Motion Generation)가 LLM을 planner로 써서 모션 토큰 후보들을 만들고, MPG(Model Predictive Generation)가 후보를 시뮬레이션과 의미 평가로 점수화한 뒤 최적 시퀀스를 선택해 다음 생성 단계를 닫힌고리(closed-loop)로 다듬는다. 특히 task-specific 정책 재학습 없이도, 시뮬레이션 환경에 맞춰 동작을 적응시키도록 설계된 점이 핵심이다.

- **Technical Challenges**: 핵심 기술 난제는 (1) LLM이 만든 high-level 모션 계획이 시뮬레이터의 동역학을 만족하지 못할 수 있고, (2) 오픈 보컬러리 프롬프트의 세밀한 의미를 안정적으로 반영해야 한다는 점이다. ICMPG는 생성/선택을 MPC-like receding-horizon 최적화로 바꿔, 각 스텝에서 물리 reward와 의미 reward를 함께 계산하고 top-K 후보 중 보상을 최대화한 구간을 ‘제어 행동’처럼 선택해 접어가도록 구성했다. 또한 SMPL 렌더링 기반의 의미 정규화, 재생성(re-generation) 프로토콜, 그리고 선택적 LoRA fine-tuning·미분가능 world model 변형까지 포함해 초기 불안정성과 계산 부담을 함께 완화한다.

- **Empirical Impact**: 실험은 HumanML3D, KIT-ML, BABEL에서 진행되며, 특히 BABEL의 zero-shot 오픈 보컬러리 프로토콜에서 의미 충실도와 물리 그럴듯함을 동시에 개선하는 결과를 보였다고 논문은 보고한다. 표준 setting과 zero-shot setting 모두에서 대표 baseline 대비 물리적 타당성과 의미 정합성이 더 높게 나타났고, 다양한 지시로도 모션 품질이 견고하게 유지되는 일반화 능력이 강조된다. LLM 백본 교체 등 유연성을 유지하면서 물리 시뮬레이션 기반 보정까지 포함한 ‘추론 시점 생성 최적화’ 접근이, 텍스트-투-모션과 물리 기반 제어를 잇는 실용적 방향을 제시한다.



### RobOralScan: Learning Active Intraoral Scanning for Robotic Dental Reconstruction (https://arxiv.org/abs/2606.26955)
Comments:
          24 pages, including supplementary material

- **Prior Approaches**: 기존 연구는 완전-상악·장거리 스캔에서 스캐너 동작과 스캔 전략이 품질에 큰 영향을 준다는 점을 보여주지만, 대부분은 수동 설계 경로를 분석하거나 연산적으로 설계된 탐색을 적용하는 수준에 머물러 있었다. Next-Best-View 계열은 순차적으로 센서 포즈를 고르는 프레임은 제공하지만, 짧은 시야·작동거리 제한·구강 내 작업공간 제약 속에서 ‘치아 단위 커버리지’까지 안정적으로 달성하도록 설계된 접근은 드물었다. 로봇 의료 스캐닝의 closed-loop 제어 가능성은 확인됐으나, 치아별 표면 커버리지와 모션 제약을 동시에 만족하는 자동 구강 스캔 파이프라인으로 정교화된 사례는 상대적으로 부족했다.

- **Core Contribution**: RobOralScan은 강화학습(RL) 기반으로 구강 내 자동 스캔을 ‘피드백 구동 순차 스캔-제어 문제’로 정식화한 첫 파이프라인이다. 핵심은 (1) 부분 관측을 누적해 정책이 스캔 이력을 추론하게 하는 tri-state geometric memory 관측공간과, (2) 전체 커버리지뿐 아니라 치아별 커버리지 균형을 학습 목표로 포함하는 tooth-wise coverage learning이다. 정책은 누적 메모리와 로봇 고유치(proprioception)를 바탕으로 relative scanner motion을 선택해 구강 작업공간에서 closed-loop로 스캔을 수행한다.

- **Technical Challenges**: 구강 환경에서는 시야가 좁아 관측이 부분적으로만 주어지며, 누락된 치아 표면이 재구성 품질에 직결된다. 또한 로봇이 실제로 실행 가능한 모션(작업공간 제약, 속도·자세 feasibility)을 유지하면서도 치아별로 덜 본 영역을 찾아가야 해, 단순 커버리지 극대화나 정해진 경로만으로는 학습이 불안정하다. RobOralScan은 tri-state 볼륨 메모리로 occupied/free/unknown을 구분해 미탐 영역(프런티어)을 암묵적으로 제공하고, coverage-aware 보상과 progressive curriculum을 통해 전역·치아 단위 목표를 점진적으로 강화하며, safety·motion efficiency 항까지 함께 최적화해 희소/불안정한 보상 문제를 완화한다.

- **Empirical Impact**: 합성 Teeth3DS 기반 평가에서 RobOralScan은 Chamfer Distance 0.00838, 평균 커버리지 92.58%, 치아 하단(lower-tail) 커버리지 88.45%, normalized AUC 0.6674를 달성했으며 10 에피소드 중 8회에서 스캔 기준을 충족했다. 제로샷 sim-to-real 실험에서도 물리 로봇-스캐너 셋업에서 추가 fine-tuning 없이 안정적인 궤적을 유지하며 잡음·지연·캘리브레이션 오차 하에서도 치아 및 인접 잇몸 표면을 점진적으로 획득했다. ablation 결과 tri-state 관측과 치아 단위 보상 설계가 성공률 및 하단 커버리지 개선에 결정적임을 보여, ‘치아 인지(active) 스캔’ 관점이 실질적 성능 향상으로 이어짐을 입증한다.



### UAV-MapFusion: RTK-Aligned Uncertainty-Aware Coarse-to-Fine Multi-Session UAV Mapping (https://arxiv.org/abs/2606.26928)
Comments:
          8 pages, 5 figures, accepted by IEEE Robotics and Automation Letters (RA-L)

- **Prior Approaches**: 기존 multi-session point cloud map merging은 Scan Context++(structured place recognition) 같은 방법으로 세션 간 대응을 찾거나, LAMM/MS-Mapping 같은 시스템으로 데이터 연관·outlier rejection·전역 최적화를 수행해 큰 지도를 만들었다. 하지만 UAV 환경에서는 6-DoF 시점 변화와 장기 drift로 인해 전역 정합과 로컬 fine registration을 한 번에 안정적으로 달성하기 어렵다. 또한 다수 방법이 LiDAR 제약 중심이라, RTK 같은 외부 관측을 시간 동기화/프레임 드롭아웃 문제까지 포함해 직접 통합하기에 한계가 있다.

- **Core Contribution**: 이 논문은 UAV 기반 대규모 매핑에서 세션 간 정합을 전역 일관성과 로컬 기하 정확도 관점에서 동시에 개선하는 “uncertainty-aware multi-session point cloud map merging + coarse-to-fine optimization” 체계를 제안한다. RTK를 multi-session 병합에 포함하되, RTK-odometry의 시간 오프셋과 프레임 드롭아웃을 다루는 spatiotemporal alignment를 통해 RTK 제약을 신뢰성 있게 재구성한다. 이후 불확실성을 factor graph에 통합하고, plane-factor refinement로 로컬 구조의 두께·경계 선명도를 더 개선한다.

- **Technical Challenges**: 핵심 난제는 (1) 세션별 누적 drift와 (2) RTK-odometry 간 하드웨어 동기화 부재, 불완전 샘플링·프레임 드롭으로 인해 RTK 제약이 편향될 수 있다는 점이다. 이 문제를 위해 DTW로 시간 오프셋을 추정하고, Multi-Output Gaussian Processes(MOGP)로 연속 시간에서 RTK 궤적을 예측해 동기화된 pose–RTK 대응과 불확실성을 함께 만든다. 그 다음 uncertainty-aware factor graph에서 GNSS 품질, odometry/loop-closure 등록 신뢰도, 그리고 plane 두께 기반 신뢰도를 적응적으로 가중해 전역·국소 최적화를 함께 수행하며, 반복적인 plane-factor refinement로 로컬 오차를 줄인다.

- **Empirical Impact**: 실험은 실제 비행 데이터 및 공개 MARS-LVIG에서 수행됐고, AWD(전역 일관성)와 SCS(로컬 일관성)로 평가했을 때 제안 방법이 8개 시나리오 전부에서 LAMM과 MS-Mapping보다 더 낮은 AWD/SCS를 기록하며 성능 우위를 보였다. 특히 RTK spatiotemporal alignment가 없을 때 로컬 일관성이 악화되는 ablation 결과가, 시간 동기화·프레임 드롭 복원이 필수임을 뒷받침한다. 추가로 plane-factor 기반 fine 단계까지 적용할수록 AWD와 SCS가 더 크게 감소해 잔여 로컬 misalignment 억제와 구조 정밀화의 의미가 확인됐다. 또한 저자들은 코드와 데이터 공개를 예고해 커뮤니티에서의 후속 연구·재현에 기여할 계획이다.



### Risk-Aware Selective Multimodal Driver Monitoring with Driver-State World Modeling (https://arxiv.org/abs/2606.26922)
- **Prior Approaches**: 기존 차량 내 드라이버 모니터링은 눈 깜빡임·시선·고개 각도·자세·손동작 등 행동 신호를 주로 활용했지만, 드라이버 수요(demand)처럼 시각적으로 뚜렷하지 않은 상태는 놓치기 쉽다. 생체신호(HR/EDA)는 보완 증거가 되지만, 상시(low-latency) 추론과 안전한 결정(특히 unsafe false negatives 회피)을 동시에 만족시키기 위한 배치 설계가 난제로 남아 있었다. 한편 VLM(vision-language models)은 멀티모달 추론에 강점이 있으나, 연속 감시(always-on) 환경에서는 높은 지연과 신뢰도 한계로 직접 예측 엔진으로 쓰기 어렵다는 실증 결과가 제시된다.

- **Core Contribution**: 논문은 엣지 배포 가능한 RGB-physiological 학생 모델과, 안전 비용을 반영해 “수락/유보(경고/개입)/선택적 대체”를 결정하는 cost-aware selective inference 프레임워크를 제안한다. 핵심은 분류 정확도 자체가 아니라 “unsafe false negatives를 줄이는 결정 위험”을 최우선 목표로 두고, 학습된 gate가 신뢰도·모달 불일치·생체신호 품질·배포 비용을 종합해 accept_fast 또는 abstain_warn 등을 선택한다. 또한 드라이버-상태 세계모델(driver-state world modeling)을 통해 미래의 fast-model 오차와 counterfactual 시스템 비용을 예측해, 선택 정책에 추가적인 예측 근거를 제공한다.

- **Technical Challenges**: 상시 감시에서는 latency 제약 때문에 무거운 VLM을 매 입력마다 호출하기 어렵고, gate가 신뢰를 측정해도 calibration drift(작동점 보정 붕괴)가 그룹 shift 상황에서 발생할 수 있다. 논문은 gate 학습을 비용 민감한 샘플 가중치와 보정(calibration) 데이터 기반 운영점 선택으로 수행하고, 분류 손실 외에 unsafe false negatives에 더 큰 패널티를 부여하는 효용 함수로 정책을 최적화한다. 세계모델은 외부 장면을 생성하는 일반적 world model과 달리, latent driver-state 공간에서 행동 조건의 롤아웃으로 미래 fast-model 에러 및 action-level 비용을 예측해 선택에 쓰도록 설계했다.

- **Empirical Impact**: manD 시나리오 유도 driver-demand 인식에서 RGB-physiological 학생은 11.39M 파라미터·3.08ms 추론으로 Macro-F1 0.7440, balanced accuracy 0.9099를 달성하며 RGB-only/physiology-only 기준선을 크게 앞선다. cost-aware selective inference는 항상-fast 방식의 unsafe false negative 17.37%를 약 5% 수준(씨드 평균 0.0527±0.0081)으로 낮추면서 배포 수준 지연을 유지한다. 또한 gate에 world modeling을 결합하면 grouped 평가에서 unsafe false negatives를 추가로 낮추는 등 예측 근거의 유용성이 보이지만, worst-group에서 운영점 보정 드리프트가 남아 “risk-aware selective control과 group-robust calibration”의 추가 진전이 필요하다는 결론을 강화한다.



### PlanRL: A Trajectory Planning Architecture for Reinforcement Learning-based Driving Experts (https://arxiv.org/abs/2606.26858)
Comments:
          Accepted at IROS 2026

- **Prior Approaches**: 기존 RL 기반 자율주행 전문가는 throttle·steering·brake 같은 저수준 제어를 직접 출력하는 구조가 대부분이었다. 이 방식은 해석이 어렵고, 복잡한 도로 기하를 좌표 구조 없이 모델이 전부 내재화해야 해 학습 효율과 성능이 흔들리기 쉽다. 또한 trajectory planning을 핵심으로 삼는 최신 E2E 아키텍처와의 정합성이 낮아, 실사용형 파이프라인에선 활용도가 떨어진다.

- **Core Contribution**: 이 논문은 RL 정책과 다항식 기반 trajectory planner를 결합한 하이브리드 아키텍처를 제안한다. Frenet-frame 좌표계를 사용해 복잡한 도로 기하를 곡선형(curvilinear) 표현으로 단순화하고, RL이 출력하는 high-level 명령을 계획된 궤적으로 변환해 해석성과 구조적 학습을 강화한다. 더불어 차량의 운동학적 제약을 planning 단계에 kinematic feasibility check로 반영해 누적 추종 오차를 줄이는 데 집중한다.

- **Technical Challenges**: 핵심 난제는 (1) 경계가 있는 연속 행동을 안정적으로 학습하면서, (2) RL이 만든 명령이 실제 차량 한계 내의 궤적이 되도록 보장하는 것이다. 이를 위해 Beta distribution으로 longitudinal acceleration과 target lateral offset을 bounded 형태로 출력하고, Frenet 기반 종방향/횡방향 다항식(2차/5차)으로 궤적을 생성한 뒤 feasibility check로 종결 상태를 물리적으로 보정한다. 그 결과 planner 추가로 생기던 tracking error가 누적되기 전에 제약을 흡수하도록 설계했다.

- **Empirical Impact**: CARLA Offline Leaderboard v1과 NoCrash 벤치마크에서 기존 control-centric RL 전문가 대비 성능을 크게 개선했다. Offline Leaderboard v1에서 driving score를 5% 향상, NoCrash에서 driving score는 11%, success rate는 각각 8%와 19% 상승을 보였다. 이는 RL 전문가가 단순 제어 출력에 머물지 않고, Frenet 기반 계획 모듈과 결합된 형태로 E2E planning 흐름에 더 잘 맞는 방향성을 제시한다는 점에서 의미가 있다.



### Humanoid-DART: Humanoid Loco-Manipulation using Diffusion-guided Augmentation through Relabeling and Tracking (https://arxiv.org/abs/2606.26855)
- **Prior Approaches**: 휴머노이드 loco-manipulation은 보통 텔레오퍼레이션이나 모션캡처로 얻은 데모를 추적하는 imitation learning/RL로 학습해 왔다. 다만 데모 커버리지가 곧 성능 상한이어서, 접촉이 많은 loco-manipulation의 연속적 목표 공간을 넓게 다루려면 대규모 인력/데이터가 필요하다는 한계가 있다. 또한 기존 일반화 정책은 주로 특정 기준 행동을 중심으로 학습되거나, 광범위 커버리지를 위해 고가의 데이터가 요구되는 경우가 많았다.

- **Core Contribution**: 이 논문은 Humanoid-DART로, 소수(예: 4개) 데모에서 시작해 목표 공간의 분포를 점진적으로 확장하는 self-supervised/iterative trajectory augmentation 파이프라인을 제안한다. 확산 기반(goal-conditioned diffusion)으로 목표-조건 궤적을 생성하고, RL 추적 정책이 물리 시뮬레이션에서 해당 궤적을 추적·검증하면서 가능한 행동들을 elite archive에 축적한다. 이를 통해 최소한의 전문가 개입으로 목표 공간 탐색과 행동 다양화를 동시에 달성하는 것을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 생성된 kinematic 궤적이 발 미끄러짐/침투/동역학 불가능 같은 물리 불일치를 포함할 수 있다는 점과 (2) 연속적인 목표 공간에서 실패한 목표도 학습 신호로 전환해야 한다는 점이다. 논문은 듀얼-브랜치 Diffusion Transformer로 전역(root/오브젝트 기준 특징)과 국소(몸체 관절/자세) 생성의 역할을 분리하고, structured partial unmasking으로 부분 관측 상태에서 상관을 학습하게 해 제어 품질을 높인다. 또한 physics-based evaluator와 fitness 기반 elite 선별, near-missed 롤아웃의 goal relabeling을 결합해 탐색 안정성과 학습 신호 품질을 확보했다.

- **Empirical Impact**: 실험은 시뮬레이션(MuJoCo)과 실로봇 Unitree G1 배치를 함께 수행하며, push/kick/hand-off/pick & place에서 높은 작업 공간 커버리지를 보인다. 특히 pick-and-place에서는 4개 수준의 seed 데모로도 목표 공간의 거의 완전한 커버리지에 도달하고, seed 범위를 넘어 4~5배 먼 목표까지 생성·추적이 가능하다고 보고한다. ablation 및 비교에서 evolutionary loop가 성능의 핵심 드라이버임을 확인했고, 듀얼-브랜치와 partial unmasking이 성공률과 경로/그립 일관성에 직접적으로 기여함을 보여준다.



### Ordinal Neural Collapse as a Representation Prior for Visual Navigation (https://arxiv.org/abs/2606.26839)
Comments:
          27 pages, 14 figures. Supplementary material included

- **Prior Approaches**: 비전 관측만으로 로봇 내비게이션 정책을 end-to-end로 모방학습하면, 비전 인코더가 행동과 무관한 시각적 단서(잡음 같은 특징)에 과적합하기 쉽습니다. 특히 단일 action loss가 인코더에 간접 감독만 제공해, 표현이 action-agnostic해지며 애매한 판단 지점에서 일관되지 않은 행동을 내며 실패로 이어질 수 있습니다. 기존 연구는 staged training이나 diffusion action decoder로 성능을 끌어올렸지만, 인코더 표현 공간이 ‘제어에 맞는 기하’를 갖도록 강제하는 방법은 제한적이었습니다.

- **Core Contribution**: ORION(Ordinal Neural Collapse for Visual Navigation)은 목표-상대 방향(예: Far Left~Far Right)의 순서 구조를 반영해, 인코더 표현 공간을 ordinal한 축으로 정렬하는 사전학습을 제안합니다. 단순히 분류 라벨을 주입하는 것이 아니라, 클래스 평균이 단일 판별 축 위에 ‘서열대로’ 배치되도록 유도하고 클래스 내부 분산은 축에 수직 방향으로 억제합니다. 이후 pretrained 인코더를 diffusion 기반 NoMaD 프레임워크에 넣고 end-to-end로 fine-tuning해 전체 파이프라인이 안정적으로 작동하도록 합니다.

- **Technical Challenges**: 핵심 기술적 난제는(1) 연속 행동을 시각 관측에 맞는 ‘순서형 클래스’로 정의하고(2) 그 순서형 기하를 표현 학습에 직접 반영하며(3) downstream fine-tuning에서 구조가 깨지지 않게 하는 것입니다. ORION은 목표 방향 각도를 기준으로 ordered class를 만들고, CDNV(클래스 분리 대비 내부 분산) + supervised ordinal projection(라벨 순서에 따른 1D 축 정렬) + orthogonal compactness(축 바깥 분산 억제)로 전처리 단계의 표현 기하를 형성합니다. 또한 context 수준이 아니라 encoder feature에 제약을 걸고, mini-batch 통계 노이즈를 Welford 누적 통계 및 축 EMA로 완화해 학습 안정성을 확보합니다.

- **Empirical Impact**: 시뮬레이션과 실세계 실험에서 ORION은 end-to-end 및 neural collapse 기반 NC-ETF, 그리고 ViNT/NoMaD 대비 내비게이션 success rate와 goal progress를 일관되게 개선했습니다. 특히 복잡한 다중 교차로 같은 시각적으로 애매한 상황에서 success rate가 NoMaD 대비 최대 +26%p까지 향상됐고, heading jerk는 최대 71%까지 감소했습니다. ablation 결과도 CE-only 방향 분류 사전학습은 오히려 성능을 해치며(표현 구조 불일치), ordinal-collapse형 사전학습과 2-stage 학습 분리가 성능의 핵심임을 보여줍니다.



### Improving Vision-Language-Action Model Fine-Tuning with Structured Stage and Keyframe Supervision (https://arxiv.org/abs/2606.26801)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 파인튜닝은 전체 궤적의 모든 시점에 연속 행동 손실을 동일하게 적용해 왔습니다. 하지만 로봇 조작은 자유공간 이동과 접촉 스킬처럼 구간이 나뉘며, 그 경계(특히 gripper open/close 전환)에서 실패가 집중되는 경향이 있습니다. 그럼에도 현재 방법들은 이 ‘조작 단계(stage)’와 ‘다음 전환 목표(next gripper-event target)’를 구조적으로 감독하지 못합니다.

- **Core Contribution**: StaKe는 이런 공백을 메우기 위해 제안된 플러그인형 보조감독(plug-in auxiliary supervision) 프레임워크입니다. 데모의 gripper 상태만으로 자동 라벨을 만들고, (1) 현재가 motion/skill 중 어디인지 분류하는 stage classifier와 (2) 다음 gripper-transition 키프레임의 관절 목표를 예측하는 keyframe predictor를 학습에 보탭니다. 중요한 점은 보조 헤드는 학습 중에만 쓰고, base VLA 정책의 아키텍처와 inference loop는 그대로 유지한다는 것입니다.

- **Technical Challenges**: 기여를 실사용 수준으로 만들기 위한 핵심 기술 과제는 ‘수동 라벨 없이’ 구조적 감독 신호를 안정적으로 추출하고 학습에 통합하는 것입니다. StaKe는 gripper open/close 전환을 이벤트 경계로 파싱한 뒤, 경계 전후로 stage 라벨을 확장(margin)해 motion/skill을 자동 생성하고, 각 시점이 ‘가장 가까운 다음 전환 프레임’의 joint 상태를 향하도록 키프레임 목표를 할당합니다. 또한 learnable query token과 lightweight auxiliary head(MLP)로 추가 계산 부담을 최소화하면서, 보조 손실의 가중치와 키프레임 타깃 정규화를 통해 학습 균형을 맞췄습니다.

- **Empirical Impact**: 실험에서 StaKe는 RoboTwin 2.0 이중팔 시뮬레이션과 Franka 단일팔 실로봇 과제 모두에서 일관되게 성공률을 끌어올렸습니다(시뮬 14%대, 실로봇 56%대 상대 개선). 특히 gripper-event 전환이 많은 장수행(long-horizon)에서 개선 폭이 더 컸고, ablation 결과 stage supervision과 keyframe supervision을 함께 써야 전체 성능 향상이 재현됨이 확인됐습니다. 정성 분석에서는 학습된 stage 예측이 실제 조작 구간 경계를 잘 따라가고, keyframe head의 다음 전환 타깃 예측 오차도 롤아웃 전반에서 안정적으로 유지되어 성능 향상의 원인을 뒷받침했습니다.



### SSI-Policy: Learning Structured Scene Interfaces for Vision-Language Robotic Manipulation (https://arxiv.org/abs/2606.26800)
Comments:
          Accepted by 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

- **Prior Approaches**: 기존 언어-조건 로봇 조작 연구는 비디오 예측 기반이 장기 롤아웃에서 기하 드리프트가 생기기 쉽고 계산 비용도 커지는 한계가 있었다. 반면 3D 중심 방법은 공간 추론엔 강하지만 RGB-D/점군 같은 센서 의존이나 추가 초기화(3D anchor)가 필요해 데이터 효율과 설정 유연성이 떨어진다. 플로우·궤적 인터페이스는 효율적이더라도 geometry를 암묵적으로 처리하거나 depth/3D 초기화에 기대는 경우가 많았다.

- **Core Contribution**: SSI-Policy는 Structured Scene Interface(SSI)를 핵심으로 하는 모듈형 프레임워크를 제안한다. SSI는 RGB만으로도 (1) 단안 depth 특징, (2) 언어-접지된 객체 레이아웃, (3) instruction-conditioned 2D motion trajectory를 함께 표현하는 중간 표현 레이어다. SSI는 embodiment-agnostic하며 행동 라벨이 없는 영상으로 학습 가능해, 다운스트림 정책이 few demonstrations만으로도 학습하도록 설계됐다.

- **Technical Challenges**: 가장 큰 기술적 과제는 RGB-only 환경에서 기하적 근거와 작업-조건 동작 정보를 동시에, 그리고 로봇 제약 없이 안정적으로 제공하는 인터페이스를 만드는 것이다. 저자들은 SSI의 각 성분(단안 depth 특징, Grounding DINO 기반 레이아웃 맵, ATM 계열의 2D 궤적 예측)을 action-free 데이터로 독립 사전학습할 수 있게 구성하고, DAP(확산 기반 액션 플래너)에서는 SSI·멀티뷰 RGB·자세 정보를 Transformer로 융합해 조건부 denoising으로 액션 시퀀스를 생성한다. 특히 궤적 초기화를 BB의 고신뢰 영역과 이미지 전역을 절반씩 섞는 hybrid sampling으로 task-aware한 샘플링을 달성했다.

- **Empirical Impact**: LIBERO 벤치마크에서 과제당 10 demos(10-shot)만 사용했을 때 SSI-Policy는 최강 선행 방법 대비 약 15%p 가까운 성능 향상을 보이며, 50-demo + 외부 사전학습을 활용하는 일부 방법과도 경쟁한다. SSI 신호의 상보성은 ablation에서 geometry-only/motion-only가 약하고, geometry·layout·motion을 함께 넣을 때 전 suite에서 일관된 개선이 나타나며 확인됐다. 또한 13개 실세계 과제에서 공간 추론, cross-embodiment transfer(휴먼 핸드→로봇 포함), 접촉-rich 조작까지 확장 검증했으며, 실제 제어에서 Diffusion Policy 대비 큰 폭의 성능 우위를 보였다.



### PressMimic: Pressure-Guided Motion Capture and Control for Humanoid Robot Imitation (https://arxiv.org/abs/2606.26741)
- **Prior Approaches**: 기존 휴머노이드 모션 모방은 주로 RGB 기반 모션 캡처/관절(기구학) 레퍼런스를 로봇에 그대로 맞추는 방식에 의존해 왔습니다. 이 과정에서 사람-바닥의 접촉 역학(압력/지면반력 등)이 충분히 반영되지 않아 발이 미끄러지거나 바닥을 관통하는 등의 실패가 자주 발생합니다.

- **Core Contribution**: 본 논문은 pressure(압력)를 인식과 제어 전 과정의 통합 모달리티로 삼아, 사람 시연의 “물리적 근거(physical grounding)”를 모방에 직접 연결하는 PressMimic을 제안합니다. 인식 단계에서는 FRAPPE++로 RGB와 압력을 함께 융합해 3D 자세 및 전역 모션을 추정하고, 제어 단계에서는 PSP(pressure-supervised policy)가 압력에서 유도한 접촉 신호를 강화학습 보상에 포함해 물리적으로 일관된 접촉 패턴을 유도합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 희소한 압력 지도를 효율적으로 표현해 비전의 깊이/가림 문제를 보완하는 것과 (2) 학습 보상에서 압력이 접촉 역학을 안정적으로 감독하도록 만드는 것입니다. 논문은 압력의 희소성을 SPE(Sparse Pressure Encoder)로 처리하고, FRAPPE++에서 시간 문맥(TCAM)과 크로스 어텐션(FCAM)으로 RGB-압력 상호작용을 강화했으며, PSP에서는 발의 plantar pressure를 pressure offset 오프셋으로 정의해 auxiliary reward로 넣고 2단계 pressure curriculum으로 학습 안정성을 높였습니다.

- **Empirical Impact**: MotionPRO라는 대규모 동기화 데이터셋(70명, 400종 동작, 12.4M 프레임)으로 실험을 진행했으며, 압력을 사용했을 때 모션 추정 정확도·궤적 일관성·실행 안정성이 함께 개선된다고 보고합니다. 특히 극단적 가림과 전역 이동 드리프트 상황에서도 압력 기반 물리 단서가 추정의 물리적 타당성을 유지하도록 돕고, 모션 모방에서도 task success와 locomotion stability가 일관되게 향상되는 것으로 나타났습니다.



### Learning Motion Feasibility from Point Clouds in Cluttered Environments (https://arxiv.org/abs/2606.26700)
- **Prior Approaches**: 기존 연구는 Sampling-based motion planners(SBMPs)를 가속하거나, infeasibility certification으로 실패를 증명하는 방식에 집중해 왔지만 주로 저차원 configuration space에 한정되는 문제가 컸습니다. 또한 실제 로봇이 쓰는 고차원(예: 7-DOF)에서, 원시 RGB-D 관측을 직접 다루기보다 원시 기하를 단순 도형(primitive)으로 가정하는 접근이 많았습니다. 그 결과 복잡한 클러터(cluttered) 환경에서는 infeasible 시도에 드는 계산 비용을 충분히 줄이기 어려웠습니다.

- **Core Contribution**: 이 논문은 7-DOF 매니퓰레이터가 실제 RGB-D 클러터 장면에서 grasp에 대해 motion feasibility(가능/불가능)을 “직접 예측”하는 문제를 제안합니다. 이를 위해 대규모 벤치마크 MoFeas(GraspNet-1Billion 기반)를 구축해, 88개 실제 스캔 오브젝트와 190개 클러터드 테이블 장면에서 총 2.71M개의 grasp별 RRT-Connect 라벨(가능/불가능)을 제공합니다. 또한 MLP, voxel-기반 3D-CNN, point-cloud Transformer의 대표 모델군을 동일 조건에서 비교해 “원시 포인트클라우드→feasibility” 학습이 성립함을 보여줍니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 원시 point cloud와 자세(그립 포즈)를 조합해, (2) kinematics+충돌+경로탐색 실패를 포착하는 학습 신호를 대규모로 만드는 것이었습니다. 저자들은 각 grasp 후보에 대해 IK, pre-grasp standoff IK, RRT-Connect 경로탐색, approach 충돌 체크를 수행해 SBMP 결과를 ground-truth 오라클로 라벨링하고, 장애물 클라우드에서 grasp 주변만 국소적으로 제거해 실제 클러터 제약을 유지했습니다. 모델 측에서는 PTv3 기반 GRASPFC-PTX가 grasp pose를 token으로 주입하고, scene 기하 신호를 잘 활용하도록 설계하며, 선택적으로 swept-volume 채널도 실험에 포함했습니다.

- **Empirical Impact**: GRASPFC-PTX는 Novel 오브젝트 분할에서 AUROC 0.996, TPR 98.5%, TNR 97.1%를 달성해 분포 내 고성능을 보입니다. 동시에 단일 예측 지연은 10ms 미만 수준으로 유지되어, RRT-Connect의 infeasible worst case에 비해 80~171배(조건에 따라) 빠른 속도 향상을 보이고, Top-5 grasp success가 분포 이동(OOD)에서도 100%로 유지되는 점이 실용적 의미가 큽니다. 다만 Unseen-scene OOD에서는 AUROC가 85%로 하락해, 단일 테이블 프레임 구성이 아닌 cross-scene 혼합 학습이 추가 과제로 제시됩니다.



### Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention (https://arxiv.org/abs/2606.26663)
Comments:
          Submitted to RSS2026 WorkShop Tactile for FM

- **Prior Approaches**: 기존 World Action Models(WAMs)는 미래 장면을 주로 시각적으로 예측하며, 동작은 예측된 미래 상태를 고려해 생성하는 구조다. 그러나 삽입·조립·탐색·재배향 같은 접촉 풍부(manipulation) 과제에서는 슬립/재밍/접촉 법선/미세 정렬 오차가 RGB에서 약하게 보이거나 가려져 물리적으로 완전하지 않은 롤아웃이 생긴다. 따라서 시각-촉각 world models나 tactile WAM들이 등장했지만, 접촉 신호가 희소하고 이벤트성이라 모든 attention 경로에 촉각 토큰을 무제약으로 넣으면 시각 동역학 예측이 망가지는 문제가 남아 있다.

- **Core Contribution**: 이 논문은 tactile pollution이라는 실패 모드를 규명하고, 이를 막으면서도 접촉 예측을 행동 생성에 유효하게 쓰는 Tactile-WAM을 제안한다. 핵심은 Tactile Asymmetric Attention Mechanism(TAAM)으로, VideoClean 마스크로는 비디오 쿼리의 tactile key/value 접근을 차단해 시각 예측 경로를 보호하되, 액션 쿼리는 접촉 정보를 읽을 수 있게 유지한다. 동시에 접촉의 ‘존재’가 아니라 ‘변화’가 행동에 필요할 시점을 touch-aware bias로 유도해 denoising 과정에서 접촉 앵커에 더 집중하도록 만든다.

- **Technical Challenges**: 가장 큰 기술적 난제는 희소·국소·스텝형 접촉 변화가 시각 프리트레인 동역학 경로를 오염시키지 않으면서도, 액션 생성에는 적시에 반영되게 라우팅하는 것이다. 논문은 이를 위해 미래 촉각 contact-state를 예측하는 변수로 포함하되, VideoClean 마스크로 비디오-촉각 상호작용을 비대칭으로 제한하고, 추가 관측 토큰 없이 촉각 이미지 모션에서 파생한 touch-state proxy와 touch-change proxy로 액션 attention 편향을 만든다. 특히 테스트 시점엔 미래 촉각 타깃이 없으므로, 첫 denoising 스텝은 bias 없이 진행하고 이후에는 이전에 예측된 touch-change로 bias를 갱신해 teacher forcing의 불일치를 줄였다.

- **Empirical Impact**: ManiFeel 실험에서 Tactile-WAM은 전체 평균 성공률이 44.7%로, RGB-only DreamZero WAM의 5.8% 대비 38.9%p 큰 향상을 보인다. 또한 접촉 풍부(contact-rich) 태스크에서는 평균 성공률 개선이 86%에 달하며, 삽입·조립 계열에서 특히 효과가 두드러진다. real-robot 5개 과제에서도 51/100 성공으로 기준 대비 큰 폭의 상승을 보였고, VideoClean이 RGB 예측 품질(MSE/MAE/PSNR/SSIM)을 개선하며 tactile pollution 완화에 기여함을 정량·정성 분석으로 확인했다.



### LAMP: Lane-Aligned Motion Primitives for Feasible Trajectory Prediction (https://arxiv.org/abs/2606.26661)
Comments:
          IEEE ITSC 2026, 6 pages

- **Prior Approaches**: 기존 모션 예측은 다중 모드를 만들더라도, 대부분은 변위 오차(가장 그럴듯한 모드 중심) 최적화에 치우쳐 장애물·차선 제약 같은 구조적 제약을 충분히 반영하지 못했다. 또한 trajectory anchors나 endpoint 기반 intention prior는 장면에 비의존적이라 lane topology와의 정합성이 깨지기 쉽고, 낮은 확률 모드에서 off-road 또는 traffic-rule 위반 같은 비현실적 예측이 섞이곤 한다. 그 결과 안전한 계획 모듈에 쓰기엔 예측 집합의 신뢰도가 흔들릴 수 있었다.

- **Core Contribution**: LAMP는 Lane-Aligned Motion Primitives로, 의도를 고정된 endpoint가 아니라 lane topology에 맞춘 구조화된 motion primitives로 모델링한다. VQ-VAE(구체적으로 NSVQ 기반)로 학습한 discrete intention queries가 spatiotemporal 패턴을 담은 궤적 prototype을 생성하게 하여, endpoint 중심 표현에서 놓치기 쉬운 형태와 시간 진화를 보완한다. 여기에 lane-topology-guided(차선 위상 기반) feasibility-aware intention selector를 더해 decoder 이전 단계에서 도달 불가능/비일관 모드를 걸러낸다.

- **Technical Challenges**: 핵심 난제는 다중 모드를 유지하면서도, 낮은 확률 모드까지 포함해 lane 제약·교통 규칙 일관성을 보장하는 “집합 신뢰도”를 높이는 것이었다. LAMP는 (1) motion primitives를 discrete codebook으로 학습해 다양한 행동을 표현하고, (2) HD-map lane connectivity를 이용해 도달 가능한 lane 집합과의 거리 기반 lane prior로 selector를 학습함으로써 decoder 입력에 구조적으로 feasible한 intention만 남기도록 설계했다. 또한 Top-1 중심 손실로 인해 대체 모드가 무시되지 않도록, selector가 decoder가 사용할 mode 자체를 선별·정렬하게 만든 점이 절충의 해법이다.

- **Empirical Impact**: Argoverse 2에서 LAMP는 b-minADE/FDE 같은 표준 변위 지표는 강한 기준선과 비슷한 수준을 유지하면서도, DAC/FR 같은 feasibility(특히 FR의 엄격한 제약)와 diversity(DwF 포함)에서 유의미한 개선을 보였다. FR에서의 향상이 더 크게 나타난 것은 단순히 도로 밖을 피하는 수준을 넘어, lane 연결성 및 교통 규칙 위반 가능성을 줄여 planner-relevant 신뢰도를 높였음을 시사한다. ablation 결과 NSVQ 코드북 선택과 충분한 selector top-L(과소 선택은 모드 수축/성능 저하)을 통해 성능-다양성 균형이 형성되며, 정성 실험에서도 교차로 장면에서 lane topology를 엄격히 따르는 다중 후보를 생성함을 보여줬다.



### Hardware Design for Table Tennis Robot Capable of Beating Professional Players (https://arxiv.org/abs/2606.26643)
Comments:
          8 pages, 13 figures, 5 tables. The supplementary video can be downloaded from "Ancillary files"

- **Prior Approaches**: 기존 연구는 주로 상용 산업용 로봇을 활용했지만, 가속도·속도·작업공간 면에서 프로 선수 수준의 스트로크를 경쟁력 있게 반복하기엔 성능 격차가 컸다. 또한 실제 매치에서 프로를 상대로 설계·검증한 커스텀 로봇 사례는 드물었다. 즉, “경기 가능한 하드웨어 성능”을 목표로 명확히 사양화하고 이를 구현한 접근이 부족했다.

- **Core Contribution**: 이 논문은 프로급 테이블테니스 로봇이 갖춰야 할 작업공간·페이로드·외력 저항·물리 성능(사이클/속도)·서브 역량·엔드이펙터 정확도를 데이터 기반으로 목표 사양으로 정리했다. 이를 만족하도록 8-DoF 커스텀 로봇 Ace를 설계하고, RL 제어가 바로 시뮬레이션-실기에서 이어질 수 있도록 동역학 모델링 파이프라인까지 포함해 “경기 승리”를 정면 목표로 달성했다.

- **Technical Challenges**: 핵심 난제는 고속 스윙에서 필요한 강성/질량 균형, 외력(공기저항 등)에도 토크 여유를 확보하는 액추에이터 선정, 그리고 RL 훈련에 쓸 만큼 정확한 저차 동역학+지연 보상 모델을 얻는 것이었다. 논문은 링크에 대해 topology optimization(AM 설계 규칙 포함)로 질량을 줄이되 강성을 유지했고, inverse dynamics torque model로 모터·기어박스 조합의 torque margin을 비교해 재차 액추에이터를 최적화했다. 더불어 조인트별 저차 모델을 식별하고 delay compensation을 넣어 잔차를 줄인 뒤, 이를 시뮬레이션에 통합해 RL 정책을 실기와 호환되게 만들었다.

- **Empirical Impact**: 실험에서 Ace는 ±0.3m 범위 이동에도 0.8s 사이클로 반복 풀-스트로크 스윙을 수행했고, 라켓 중심 피크 속도는 22 m/s에 도달했다. 또한 지연 보상 모델을 사용했을 때 추종 오차는 encoder 수준에서 거의 편차가 없고, 잔차 분포가 기준선(추종오차 분포)보다 더 좁아지며 지연 불일치가 주요 오차 원인임을 보여줬다. 로봇은 여러 명의 프로 선수들을 실제로 이겼으며, 스포츠 로보틱스에서 “하드웨어 성능 사양→동역학 식별→RL 제어의 실전 전이”가 가능하다는 실증적 기준을 제시했다.



### A Closed-Form 4-DoF Inter-Robot Pose Estimator using Bearing-only Measurements (https://arxiv.org/abs/2606.26616)
- **Prior Approaches**: GPS-denied 환경에서 로봇들은 온보드 오도메트리와 bearing 같은 상호 관측을 결합해 상대 자세(상대 pose)를 추정한다. 기존 bearing-only 방식은 주로 6-DoF를 다루지만, 특정 운동 패턴에서 관측가능성(observability) 퇴화가 심해 추정이 불안정해질 수 있다. 또한 sliding window에 의존하는 최적화는 관측가능성을 확보하기 위한 지연과 데이터 수집량의 트레이드오프가 커진다.

- **Core Contribution**: 이 논문은 bearing과 VIO(visual-inertial odometry)만으로 닫힌형(closed-form) 4-DoF inter-robot pose estimator를 제안한다. 먼저 회전의 비선형 제약을 완화해 yaw를 closed-form으로 구하고, 이어서 번역은 거리 변수를 제거하도록 단위 구면에 대한 error projection을 수행한 뒤 Total Least Squares(TLS)로 추정한다. 더불어 관측가능성 테스트 모듈을 넣어 고정 길이 윈도우 대신 추정이 가능한 “최적 순간”을 자동으로 선택한다.

- **Technical Challenges**: 핵심 난제는 6-DoF에서 흔한 관측가능성 퇴화보다 덜 까다로운 조건을 찾아도, 4-DoF에서도 특정 구성에서 여전히 퇴화가 발생한다는 점이다. 논문은 이 문제에 대해 이론적 관측가능성 분석을 수행해 yaw는 모든 로봇이 한 수직선(vertical line) 위에 있을 때 비관측이 되고, 번역은 collinear(일렬) 및 shape-preserving(형상 보존) 형성에서 비관측이 된다고 규명한다. 이를 바탕으로 번역 추정 전에 singular value/condition number 기반의 observability test로 near-degeneracy까지 감지하며 신뢰도 높은 시점에만 TLS를 실행하도록 설계한다.

- **Empirical Impact**: 시뮬레이션과 실세계 실험 모두에서 제안 방법은 SDP 기반 최신 기법(SDP-Graph, SDP-Cert 등) 대비 더 높은 추정 정확도와 크게 낮은 계산 비용을 보인다. 관측가능성 테스트 모듈은 추정 신뢰도를 확보하면서도 데이터 수집 간격을 최소화하는 데 기여한다. 결과적으로 실시간 멀티로봇 cooperative localization에 더 실용적인 효율-신뢰도 균형을 제공한다.



### Bridging Handheld and Teleoperated Supervision for Contact-Rich Manipulation via State-Gated Experts (https://arxiv.org/abs/2606.26603)
Comments:
          Project Page: this https URL

- **Prior Approaches**: UMI 같은 handheld 수집은 환경 다양성과 확장성을 제공하지만, 로봇이 실제로 실행하는 ‘관측된 행동(observed action)’만 담아 ‘원하는 행동(desired action)’을 직접 감독하지 못해 embodiment gap이 생긴다. 반면 teleoperation은 desired action을 확보하지만 수집이 느리고 데이터 다양성이 떨어져, contact-rich 작업 전 구간을 전부 teleoperate 하기엔 비용이 크다.

- **Core Contribution**: 이 논문은 작업 단계(task phase)별 action validity를 기준으로 handheld와 teleoperation의 역할을 재정의한다. handheld는 free-space처럼 오차 허용 구간에서는 유효한 감독이 되지만, 접촉이 민감한 구간에서는 observed action을 높은 강성으로 추적할 때 큰 접촉 힘이 발생해 동역학적으로 불가능해진다는 점을 보인다. 이를 바탕으로 전체를 teleoperate하지 않고, base handheld 정책이 실패하는 특정 접촉 병목 구간에 한정해 sparse teleoperated ‘partial demonstration’만 수집하는 효율적 하이브리드 전략을 제시한다.

- **Technical Challenges**: 문제는 handheld와 teleoperated 데이터를 단순 혼합하면 observed와 desired 감독이 서로 양립되지 않아 성능이 오히려 악화된다는 mismatch다. 이를 해결하기 위해 BRIDGE는 diffusion policy 기반의 mixture of experts로, 로봇 상태에 조건화된 gated router가 task phase별로 base(관측 감독)와 support(원하는 행동 감독) expert를 선택하도록 설계했다. 관측/원하는 행동의 차이가 실제로 충돌하는 접촉 구간에서만 desired action을 사용하게 만들어, 단계별로 action manifold에 맞는 감독을 제공한다.

- **Empirical Impact**: Franka FR3의 접촉이 많은 3개 조작 과제에서 BRIDGE는 handheld-only 대비 성공률을 크게 끌어올리며, 특히 NIST Pulley Routing(76.0%), Pipe Insertion(50.0%), Spring-Loaded Battery Insertion(33.3%) 같은 결과로 성능 격차를 빠르게 회수했다. teleoperation 전체를 모으는 상한(full-task teleoperated)과의 격차를 최대 36.7%p(논문 표현: 36.7%)까지 줄였고, 끝단 접촉 조절이 어려운 과제에서 partial teleoperation의 효율이 특히 두드러졌다. 또한 router는 contact phase 전환이 애매한 경계에서 오탐이 몰리는 패턴을 보이면서도, 전체적으로 올바른 phase manifold로 라우팅됨을 분석으로 뒷받침한다.



### Inference-Time Robot Behavior Steering through Physically-Aware Reconfiguration of Task-Structur (https://arxiv.org/abs/2606.26588)
- **Prior Approaches**: 기존에는 end-to-end 방식이 테스트 시 선호를 바꾸려면 fine-tuning이나 전문가 수준의 guidance가 필요해, 미분 가능하지 않거나 요구가 복잡하면 성능이 떨어졌습니다. 또 neuro-symbolic 방식은 LTL/논리 같은 선호를 기호층으로 편집하지만, 기호 수정이 물리 실행을 담당하는 연속 제어기까지 일관되게 전파되지 않아 논리상 말이 되지만 물리적으로는 불가능한 계획이 나올 수 있습니다.

- **Core Contribution**: ReStruct는 inference-time behavior steering을 위해 정책을 상태기계(skeleton)와 잔차(residual) 연속 제어기로 분해한 ENAP 위에서 동작 구조를 재구성합니다. 사용자가 준 선호를 automaton(DFA/LTLf)으로 바꾼 뒤, skeleton에 synchronous product로 결합해 “선호를 만족하는 작업 구조”를 먼저 만들고, 제어기는 재학습 없이 고정한 채 실행이 되도록 맞춥니다.

- **Technical Challenges**: 핵심 난제는 skeleton을 바꿔서 선호 일치는 만들었지만, 제어기가 보던 행동 prior와 불일치하면 물리 불가능 동작이 발생한다는 점입니다. ReStruct는 이를 해결하기 위해 수정된 skeleton의 각 전이(edge)에서 시연으로부터 action priors 후보를 재생성/재접근(replay 기반으로 필터링)하고, low-level residual policy는 frozen 상태에서 prototype 기반 attention과 clamp로 새 prior에도 견딜 수 있게 설계합니다.

- **Empirical Impact**: 시뮬레이션과 실세계 실험에서 ReStruct는 object-centric 명세부터 temporal-logic 제약까지 다양한 선호에 대해 task success와 선호 준수(SRP)를 동시에 높였고, VLA 대비 최대 25%까지 개선됐다고 보고합니다. 특히 멀티모달 환경에서 상태기계 기반의 선택이 모드 불확실성을 줄여, 확률적으로 방황하는 접근보다 더 안정적으로 목표 순서를 복원하고 위상 경계에서 편차가 누적되지 않는 경향을 보였습니다.



### IDEA: Insensitive to Dynamics Mismatch via Effect Alignment for Sim-to-Real Transfer in Multi-Agent Contro (https://arxiv.org/abs/2606.26575)
Comments:
          8 pages, 6 figures

- **Prior Approaches**: 기존 sim-to-real 전이는 domain randomization(시뮬레이터 파라미터 교란)과 domain adaptation(실데이터로 적응)을 주로 사용해 현실 갭을 줄인다. 하지만 MARL은 저수준 연속 제어 액션에 의존해 dynamics mismatch에 민감하거나, 온라인 적응(RMA·HIM 등)은 history/추정 지연 때문에 장기 과업에서 오차가 누적될 수 있다.

- **Core Contribution**: 논문은 다중 에이전트 제어에서 IDEA(effect alignment 기반 dynamics mismatch 불감 sim-to-real)를 제안한다. 핵심은 정확한 물리 모델링 대신, 이산 semantic action을 low-level closed-loop 제어로 실행해 시뮬과 현실에서 “작용 효과(action effects)”가 정렬되도록 학습하는 데 있다. 또한 에이전트 간 action 타이밍 불일치를 줄이기 위해 communication 기반 action synchronization을 도입한다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) 시뮬과 현실의 상태-전이 불일치, (2) 실제 분산 실행에서 발생하는 액션 비동기 문제다. IDEA는 semantic action이 유도하는 closed-loop 전이를 통해 전이 불일치를 이론적으로 구속하려 하고, 동시에 에이전트들이 동일 시점에 joint action을 실행하도록 coordination node 기반 동기화 프로토콜을 설계해 Dec-POMDP의 동시성 가정을 물리적으로 맞춘다. 학습 효율을 위해 Isaac Gym을 GPU 텐서 중심으로 확장해 병렬 환경에서 다양한 기하 구조를 제공한다.

- **Empirical Impact**: 네 가지 multi-agent navigation 시나리오에서 IDEA는 학습 수렴 속도와 최종 성공률 모두에서 기존 DR·RMA·HIM 대비 우수했다. 특히 real-world zero-shot 전이에서 성공률이 baseline 대비 20% 이상 향상되었고, 충돌 0% 및 에이전트 간 실행 타이밍 오류 0을 달성했다. 분석에서도 시뮬 대비 실제 action effect 불일치가 5% 미만으로 줄고, 에이전트 간 시간 지연도 0으로 유지되어 안전성과 협응의 견고성이 확인됐다.



### OSC2Runner: OpenSCENARIO 2.x Compliant High-Fidelity AV Simulation in CARLA (https://arxiv.org/abs/2606.26533)
Comments:
          Accepted at 26th IEEE International Conference on Software Quality, Reliability, and Security (QRS 2026)

- **Prior Approaches**: 기존 Scenario-Based Testing(SBT)은 주로 ASAM OpenSCENARIO XML 1.x에 의존했으며, CARLA 같은 연속 시뮬레이터에서 v2.x DSL을 네이티브로 실행하는 체계가 부족했다. 레거시 interpreter를 v2.x 로직에 억지로 적용하면 시공간 드리프트, 비동기 이벤트 지연, 강제적인 kinematic snapping 같은 실행 오차가 생겨 고정밀 검증의 전제가 흔들린다.

- **Core Contribution**: 이 논문은 OpenSCENARIO v2.2를 CARLA로 “네이티브 매핑”하는 OSC2Runner를 제안해, 시나리오 실행을 해석(interpretation) 기반 근사에서 컴파일(compilation) 기반으로 전환한다. 멀티 패스 transpiler가 v2.x 선언을 type-safe Abstract Syntax Tree(AST)로 만든 뒤, 정적 궤적 재생 대신 dynamic deterministic Behavior Trees(py_trees)로 합성해 CARLA 원자 API(atomic API)를 직접 호출하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 DSL의 선언적 의도(공간/시간/행동 제약)를 연속 물리 루프에서 tick-by-tick 결정론적으로 유지하면서, 동적 파라미터를 실행 중에 재평가하는 것이다. 이를 위해 ANTLR4 기반 구문/의미 분석(심볼 테이블 두 패스) 후, MethodRegistry로 추상 액션을 CARLA의 원자 동작에 안전하게 디스패치하고, ExecutionContext가 실시간 표현식을 캐싱·재계산하며, at:start 제약은 ScenarioInitializer가 지연 스폰/텔레포트로 정확히 맞춘다.

- **Empirical Impact**: 실험은 고동시(adversarial) 다중 행위자 케이스에서 tick-by-tick determinism, 정확한 spatial trigger 평가, actor 간 100.0 ms blackboard 동기화를 보여준다. 또한 마찰 저하(젖은 노면) 같은 환경 변화에서도 PID 기반 kinematic modifier가 연속 물리 한계를 존중하며 jerk/감속 프로파일이 의미적으로 대응됨을 입증해, co-simulation, hardware-in-the-loop, LLM 기반 시나리오 자동 생성 파이프라인에 필요한 결정론적 실행 백엔드를 제공한다.



### WatchAct: A Benchmark for Behavior-Grounded Robot Manipulation (https://arxiv.org/abs/2606.26443)
- **Prior Approaches**: 기존 조작 벤치마크는 대체로 현재 장면(정적 관측)만을 입력으로 실행 능력을 평가하거나, 비디오를 데모 재현/자기 시점 기억으로 쓰는 경우가 많았다. 이 때문에 로봇이 다른 사람의 행동으로부터 사건 순서·의도·누적 상태를 추론하는 ‘관찰 기반 추론’을 체계적으로 검증하기 어려웠다.

- **Core Contribution**: WatchAct는 실제 사람의 행동 비디오(관찰 증거)와 언어 지시를 입력으로, 정렬된 시뮬레이터 및 LIBERO 실행 태스크까지 연결해 “관찰된 인간 행동을 이유로 삼는 조작”을 평가하는 새로운 벤치마크를 제안한다. 14개 태스크(총 3,000개, long-horizon)와 4개 인지 도메인(Event Grounding, Procedural Reasoning, Implicit Intent Inference, Episodic Reasoning)으로, 이벤트·절차·암묵 의도·에피소드 추론을 분해해 측정하도록 설계됐다.

- **Technical Challenges**: 핵심 기술적 난관은 비디오-언어 추론이 장면의 객체 이력/공간 관계/미완 절차/의도를 정확히 계획으로 변환해야 하고, 이어서 물리적 실행이 긴 시퀀스를 안정적으로 수행해야 한다는 점이다. 논문은 disentangled evaluation로 (1) VLM의 video-to-plan reasoning, (2) oracle plan 하의 policy execution, (3) planner–policy 통합 파이프라인의 end-to-end 성공을 분리 측정해 오류가 어디에서 누적되는지 드러내는 프로토콜을 도입했다.

- **Empirical Impact**: 실험 결과, 시뮬레이션과 Franka Research 3 로봇 모두에서 현재 시스템은 WatchAct를 사실상 ‘해결’하지 못하는 수준으로 나타났다. 예컨대 Gemini-3.1-Pro는 Plan SR 36.8%(인간 97.1% 대비 큰 격차), 통합 파이프라인 Success Rate는 시뮬 16.3%, 실로봇 14.0%였고, plan 길이가 늘수록 성능이 급격히 붕괴했으며 out-of-domain 일반화도 크게 떨어졌다. 저자들은 이 결과가 로봇이 사람과 함께 일할 때 필요한 관찰 기반 추론과 제어의 미해결 영역을 실험적으로 정량화하는 데 의미가 있다고 보고했다.



### Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly? (https://arxiv.org/abs/2606.26428)
Comments:
          22 pages, 12 figures, 4 tables. Project page: this https URL

- **Prior Approaches**: 정밀 조립(assembly)은 접촉이 많은 데다 보상은 희소해서, 강화학습(RL)이 바로 쓰기 어렵고, 데이터 수집도(모방학습, IL) 텔레오퍼레이션·데모 의존도가 커진다. 그래서 선행연구는 고정 그리퍼 구조, 툴 부착, 환경 고정구(피처)처럼 문제를 “구조화”해 탐색을 쉽게 만드는 방식이 주류였다. 다만 이런 접근은 과제별 하드웨어/환경 엔지니어링이 필요하거나, 병렬집게 중심의 제약으로 속도·손가락 정밀성을 충분히 활용하지 못한다.

- **Core Contribution**: 이 논문은 “조립을 바로 완벽히 배우기 전에, 먼저 play로 조작 능력을 익혀야 한다”는 전제를 세우고 Play2Perfect를 제안한다. Play2Perfect는 객체와 목표가 다양한 상황에서 미리 학습한 조작 priors를 만든 뒤, CAD 정의 기반 정밀 조립 과제로 sparse-reward RL fine-tuning을 통해 마지막 접촉·정밀 상호작용을 완성한다. 특히 play 단계에서 grasping, in-hand reorientation, pose reaching 같은 재사용 가능한 능력을 목표조건(goal-conditioned)으로 축적한다.

- **Technical Challenges**: 핵심 기술 난제는 sparse reward 때문에 RL이 무작위 정책에서 grasping–정렬–접촉 삽입을 찾아내기 전까지 학습 신호가 거의 없다는 점이다. 저자들은 play pretraining을 통해 손가락 기반 in-hand 조작을 선행으로 학습하고, 목표 정밀도(목표 도달 임계값), 궤적 다양성(에피소드마다 goal sequence를 랜덤화), 6D 목표(translation+rotation)로 조립에 필요한 정렬·재지향을 유도한다. 또한 CAD를 역분해해 고정구/부품 배치를 만들고, 최종 CAD 구성으로부터 sparse contact goals를 구성해 fine-tuning에서 탐색 효율을 끌어올린다.

- **Empirical Impact**: 실험에서 Play2Perfect는 RL을 처음부터(scratch) 시작하는 방식보다 샘플 효율이 33x 높으며, 조밀 보상(multi-stage)을 준 scratch( dense reward)조차도 24시간 내 성공이 거의 없었다. 또한 sim-to-real zero-shot에서 tight insertion은 0.5 mm 접촉 clearance 조건에서도 60% 성공, 장수평선 multi-part assembly와 screwing에서는 50% 이상 성공을 보였다. 저자 ablation은 object diversity·trajectory diversity·특히 6D 학습 목표와 goal precision이 downstream 성능 전이에 결정적임을 체계적으로 보여, 정밀 조립 학습의 설계 지침으로도 의미가 있다.



### A System for Fast, Resilient, and Adaptable Loco-Manipulation Behaviors on Humanoid Robots (https://arxiv.org/abs/2606.26425)
Comments:
          PhD dissertation, University of West Florida, Department of Intelligent Systems and Robotics, June 2026. 331 pages

- **Prior Approaches**: 기존 휴머노이드 로봇 연구는 보행과 전신 제어, 인지, 접촉 및 원격 조작을 각각 따로 다루는 경향이 강해 실제 사람 환경에서의 통합 운용이 어렵다는 한계가 지적돼 왔다. 또한 행동을 만들고 수정하는 과정이 느리거나(사전 설계 중심), 운영 중 상황에 맞춘 즉시 수정·수리는 제한적이었다.

- **Core Contribution**: 이 논문은 Coactive Design 원칙을 바탕으로, 로봇 현장에서 바로 편집 가능한 robot-local runtime-editable behavior authoring과 runtime system을 제안한다. 목표는 행동을 최대한 관측 가능하고 예측 가능하게 하며, 운영자가 지속적으로 동기화된 인터페이스로 저수준 수정·감시·수리를 수행할 수 있게 하는 것이다.

- **Technical Challenges**: 핵심 기술 과제는 보행 중 전신 움직임과 조작(팔 이동), 접촉/인지 갱신, 그리고 운영자 감독을 한 런타임 안에서 동시에 만족시키는 통합성이다. 논문은 object-centric Affordance Templates, Behavior Trees에서 영감을 받은 조직·논리, behavior scene 및 primitive scene actions을 통한 런타임 편집 인지 구조를 결합하고, 보행과 팔 동작을 지원하는 whole-body controller와 동시 action layering algorithm으로 속도도 확보한다.

- **Empirical Impact**: 이 시스템은 20개 이상의 실제 로봇 작업 변형(노브·푸시바·레버 핸들 도어, 다단 탐색, 장애물 제거, 테이블 간 반응형 조작 등)을 지원하며 여러 휴머노이드에 배치됐다. 평가는 역량, 속도, 신뢰성뿐 아니라 행동 생성/적응/확장/조합 시간까지 다뤘고, 기존 행동을 몇 분~몇 시간 내에 새 loco-manipulation 행동으로 적응·확장·조합할 수 있음을 보여 의미가 크다.



### CoStream: Composing Simple Behaviors for Generalizable Complex Manipulation (https://arxiv.org/abs/2606.26423)
Comments:
          Website: this https URL

- **Prior Approaches**: 기존 접근은 크게 두 갈래였다. (1) 고정밀을 위해 작업별 인터페이스에 의존하는 고전 파이프라인은 새로운 작업에 적응하려면 파이프라인 재설계 비용이 크다. (2) end-to-end 단일 정책은 out-of-distribution에서 일반화는 유리하지만 contact-rich 같은 복잡 과업에서 정밀도가 부족해 추가 fine-tuning이 필요해지는 경우가 많다.

- **Core Contribution**: 이 논문은 “획득한 조작 능력을 rigid 파이프라인이나 단일 정책으로만 배치해야 한다”는 암묵적 가정을 깨고, 단순하고 독립적인 행동들의 조합으로 복잡한 능력이 자연스럽게 나온다고 주장한다. 제안하는 ourshort는 foundation model과 다양한 센싱을 여러 composable core behaviors로 오케스트레이션해, 매 제어 스텝에서 이들을 조합해 단일 pose 명령을 만든다.

- **Technical Challenges**: 핵심 technical challenge는 서로 다른 모달리티와 서로 다른 행동이 만들어내는 출력들을 어떻게 안정적으로 단일 제어 명령으로 합칠지였다. 논문은 공통의 SE(3) 인터페이스 위에서 각 행동의 출력을 right-multiplication으로 합성하고, compliance controller가 이를 실제 접촉 환경에서도 고주파 tactile/force 보정과 함께 실행하도록 설계했다.

- **Empirical Impact**: ourshort는 일상 조작부터 정밀 조립까지 포함한 8개 실세계 과업에서 실험을 통해 성능을 입증했으며, 특히 contact-rich assembly와 object transfer에서 가장 큰 향상이 관찰됐다. 또한 실행 중 사람이 가한 manual perturbation에도 회복이 견고해, 실제 환경에서의 신뢰성과 범용성 관점에서 의미가 크다.



### Exploring the Intrinsic Geometry of Diffusion Models with Constrained Inverse Kinematics (https://arxiv.org/abs/2606.26408)
Comments:
          21 pages, 12 figures, 10 tables

- **Prior Approaches**: 확산 모델은 잡음 제거 학습만 수행하지만, score function과 Riemannian metric 등을 통해 데이터 분포의 기하를 회복한다는 연구가 늘고 있다. 다만 자연 이미지 실험에서는 데이터 manifold의 기저 기하가 불명확해, intrinsic dimension이나 토폴로지가 추정값에 의존하는 한계가 있었다. 또한 로봇 IK를 다루는 기존 연구는 주로 생성 품질(sample quality)이나 해 탐색 성능에 초점을 맞췄다.

- **Core Contribution**: 본 논문은 constrained inverse kinematics(IK)를 확산 모델의 기하 해석을 위한 ‘통제된(controlled) 실험장’으로 제안한다. 작업공간 제약은 구성공간에서의 constraint manifold를 만들고, 그 intrinsic dimension과 토폴로지가 해석적으로 주어지므로 모델이 학습한 기하를 정답과 직접 비교할 수 있다. 저자들은 UR5(6-DoF)와 Franka(7-DoF) 각각에 대해 단 하나의 conditional diffusion model을 여러 제약군에 걸쳐 학습하고, 조건에 따라 변화하는 기하를 한 모델에서 분석한다.

- **Technical Challenges**: 핵심 도전은 diffusion model의 score로부터 manifold의 intrinsic dimension을 ‘알려진 해석 정답’과 동일한 기준으로 측정하는 것이다. 기존 Score-SVD는 양의 차원(manifold이 0차원이 아닌 경우)을 가정하므로, UR5의 SE(3) 전체 pose 제약처럼 해 집합이 이산(branch들의 유한 집합, ID=0)인 경우에는 Score-SVD†로 수정해 스펙트럼 간극을 검출한다. 더 나아가 dimension이 맞더라도 latent 공간의 직선 보간이 제약 manifold 밖으로 이탈할 수 있음을 고려해, DDIM-inverted latent에서의 선형 보간 결과가 forward kinematics residual 관점에서 on-manifold에 머무는지 정량적으로 평가한다.

- **Empirical Impact**: 실험 결과, 모델 score function에서 회복된 intrinsic dimension이 UR5와 Franka 모두에서 각 제약군의 해석적 degrees of freedom와 일치함을 보였다. 또한 DDIM-inverted latent에서의 linear interpolation이 decode된 해를 해당 constraint manifold 근처에 유지하며, 특히 UR5의 경우 이산 분기(disconnected branch)라는 토폴로지가 보간 과정에서 ‘급격한 전이’ 형태로 반영되는 양상도 확인했다. 이는 diffusion 모델 표현이 단순히 차원(ID)만 맞추는 수준을 넘어, 제약군의 국소 기하구조와(일종의 locally linearizing/flattening) tangent 방향 정합까지 포착할 수 있음을 시사한다.



### MPC-Injection: Biasing Off-Policy Locomotion RL Toward Controller-Induced Behavior Basins (https://arxiv.org/abs/2606.26392)
Comments:
          22 pages, 10 figures

- **Prior Approaches**: 다리 달린 보행에서 off-policy RL은 보상을 극대화해도 실제로는 진동·자세 뒤틀림·몸통을 비틀어 이동하는 등 “쓸모없는” 국소 해로 수렴하는 문제가 흔합니다. 이를 줄이기 위해 reward shaping은 수작업 항을 많이 튜닝해야 하고, 목표 거동이 바뀌면 재학습 비용이 커집니다. AMP 같은 적대적 모방은 discriminator와 모션 리타게팅, 참조 데이터(수집/정리) 등 별도 공수가 들어갑니다.

- **Core Contribution**: 이 논문은 MPC-Injection을 제안해, 설계자가 선호하는 보행을 MPC가 방문하는 상태 분포를 통해 RL이 선택하도록 만듭니다. 방법은 RL 보상은 그대로 두고, MPC가 생성한 전이를 replay buffer에 “주입”해 actor-critic 업데이트가 controller behavior basin 쪽으로 편향되게 하는 것입니다. 결과적으로 보상 설계나 모방 학습 손실, discriminator/리타게팅 없이도 거동 편향(behavior biasing)을 달성합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “추가 보상 설계 없이” replay 분포 편향만으로 충분히 바람직한 상태 분지(basin)를 골라내는 일입니다. 저자들은 MPC 전이를 주입하는 비율 pp를 제어하고, SAC/TD3 같은 off-policy 업데이트는 그대로 두되 MPC 전이가 학습에 들어가는 경로를 replay state distribution으로 제한합니다. 또한 quadruped의 제어 출력 차이(토크 vs 타깃 포지션)를 inverse PD로 변환해 동일한 주입 파이프라인을 유지했습니다.

- **Empirical Impact**: 2D walker 시뮬레이션과 Unitree Go2(시뮬-투-리얼)에서 MPC-Injection은 RL 단독 대비 qualitatively 다른 보행 basin을 선택하며, 주입 비율 25%에서 가장 안정적으로 controller basin으로 유도됩니다. 성능(보행 품질)은 reward shaping의 21개 튜닝 항, 그리고 discriminator/리타게팅 없이 AMP와 비교해도 큰 차이 없이 나타났고, Go2 하드웨어에서도 정성적으로 전이됐습니다. 또한 actor-critic이 MPC가 방문한 상태 쪽으로 편향되며, 단순 보상만으로는 도달하기 어려운 거동을 학습할 수 있음을 분석으로 뒷받침합니다.



### Scaling Nonlinear Optimization: Many Problems One GPU (https://arxiv.org/abs/2606.26341)
Comments:
          8 pages, 6 figures, ieeeconf style, submission for RA-L

- **Prior Approaches**: 로봇의 궤적 최적화, 역기구학, 접촉이 많은 모션 플래닝은 비선형계획(NLP)으로 자주 정식화되며, 기존에는 CPU 기반 NLP 솔버인 IPOPT 등이 주로 쓰였다. 이들 방법은 하드 제약 만족과 최적성 보장을 제공하지만, 단일 문제를 순차적으로 풀어 GPU-batched 학습/시뮬레이션 파이프라인에 끼워 넣기 어렵다는 한계가 있었다. 한편 강화학습, MPPI, imitation learning 등 sampling-based 접근은 GPU 배치 시뮬레이터를 잘 활용해 대량 병렬 롤아웃을 뽑아내며 주류가 됐다.

- **Core Contribution**: 이 논문은 GPU에서 “동시에 많은 NLP를 푸는” jaxipm을 제안하며, IPOPT 계열의 장점을 GPU-batched 워크플로로 확장한다. 핵심은 IPOPT의 분기되는 제어 흐름과, 배치 처리 시 발생하는 GPU 유휴 시간을 줄이기 위한 설계로, 수천 개 문제를 병렬 처리할 수 있게 한 점이다. 저자들은 quadrotor NMPC 벤치마크들에서 처리량(throughput) 가속을 실증한다.

- **Technical Challenges**: GPU 병렬화의 가장 큰 걸림돌은 IPOPT의 각 반복(iteration)에서 문제마다 서로 다른 fallback/제어 흐름 분기를 타거나, 수렴 시점도 달라 배치가 lockstep으로 진행되지 않는다는 점이다. 저자들은 이를 해결하기 위해 heterogeneous iteration fusion으로 분기별 공통 연산을 한 번에 계산하고, 분기 고유 연산은 모두 계산한 뒤 필요한 결과만 선택하도록 재구성했다. 또한 iteration level batching으로 solve 단위가 아니라 iteration 경계에서 이미 끝난 문제를 즉시 새 문제로 교체해 GPU idle을 최소화했다.

- **Empirical Impact**: quadrotor NMPC 세 가지 시나리오(단일/다중 쿼드로터 네비게이션 및 장애물 포함 reference tracking)에서 jaxipm은 IPOPT 대비 최대 32.85× 처리량 향상을 보였다. 특히 초기조건에 따라 장애물과 상호작용하는 문제/하지 않는 문제가 섞여 반복 횟수 분포가 달라지는 경우에도 iteration level batching 효과가 유지됨을 보여준다. 또한 per-iteration 수준에서 IPOPT와의 bit-parity를 보였다고 보고해, 기존 IPOPT 사용처에서의 대체 가능성을 강조한다.



### KRVF: A Source-Aware Semantic Voxel World Representation for Edge Mobile Manipulation (https://arxiv.org/abs/2606.26321)
Comments:
          Technical report, 9 pages, 3 figures

- **Prior Approaches**: 기존 로봇 매핑은 주로 재구성(reconstruction) 또는 SLAM 중심으로 전역 기하 충실도를 높이는 데 초점이 맞춰져 있다. 그 결과 모바일 조작에서는 ‘지금 행동 가능한가’ 같은 작업 수준 질문(물체 위치, 증거 신뢰도, 신선도, 다음 행동)이 저지연으로 답하기 어렵다. 또한 RGB-D의 깊이 실패(반사·투명·유광 등) 상황에서 의미론은 맞지만 기하가 불완전하거나, 반대로 기하 점군은 있어도 쿼리·그립 후보 추출이 까다로운 문제가 반복된다.

- **Core Contribution**: KRVF는 edge mobile manipulation을 위한 소스(source) 인지형 의미론적 voxel 월드 표현으로, 맵을 단순 시각화 대상이 아니라 ‘로봇 메모리’로 재구성한다. 각 voxel에 점유(occupancy)와 색(color), 의미론(semantic evidence), 시간적 신선도, 증거 출처/타입을 함께 담아 작업 지연과 쿼리 가능성을 우선한다. 특히 observed geometry와 semantic-prior hypothesis를 분리해 깊이 실패 구간에서도 의미론적 후보는 제공하되 영구 기하는 묵시적으로 오염되지 않게 한다.

- **Technical Challenges**: KRVF의 핵심은 (1) 깊이 센서가 깨지는 상황에서 의미론을 ‘안전하게’ 활용하고, (2) 동적 환경에서 오래된 구조가 조작에 해가 되지 않게 하며, (3) edge 하드웨어에서 저지연으로 쿼리를 제공하는 것이다. 이를 위해 voxel 상태를 관측 레이어와 hypothesis 레이어로 분해하고, log-odds 업데이트에 source confidence를 가중해 신뢰도 낮은 입력이 공격적으로 기하를 바꾸지 못하게 설계했다. 또한 맵으로부터 camera view의 depth prior를 렌더링해 depth repair에 되먹임(피드백 루프)을 주고, object/그립은 의미론 객체와 안정도(stability) 기반의 task-level query operator로 노출한다.

- **Empirical Impact**: 본 보고서의 평가는 정량 벤치마크보다는 표현과 파이프라인의 질적 검증에 초점을 둔 시뮬레이션/재생(replay) 실험이다. YOLO 같은 2D 검출을 KRVF의 occupied voxel 근거에 grounding해 ‘카테고리 증거’와 ‘3D 지지(geometry support)’를 분리했으며, 깊이 실패 시 observed 맵은 비어 보존하고 hypothesis 오버레이만 제한적으로 표시하는 동작을 시연한다. 또한 voxel 객체에서 top/waist/side 및 fallback grasp 후보 마커를 생성해 맵이 곧 조작용 질의·타깃 공급 서버로 기능함을 보였다.



### Layered Outer-Loop Control for Disturbance-Robust Multi-Waypoint UAV Arriva (https://arxiv.org/abs/2606.26315)
Comments:
          Preprint, 12 pages, 5 figures, 5 tables

- **Prior Approaches**: 기존 UAV 위치제어 연구는 PID·비선형/기하학적 추적·최적제어 유사 방식 등으로 평균적인 waypoint 도달 성능을 잘 만들 수 있었지만, 목표 근처에서의 감속/정착과 지연·내부루프 상호작용이 커지면 종종 성능이 무너진다. 바람 같은 잡음·교란을 위한 적분작용, disturbance observer, bias adaptation, 피드포워드/거부 로직도 널리 쓰이지만, 이는 터미널 구간의 감쇠·핸드오프·모드 전환과 결합될 때의 실패 양상을 단독으로 해결하긴 어렵다는 한계가 있었다. 또한 시뮬레이터 벤치마크에서의 튜닝이나 completion 조건이 다른 평가/하드웨어로 옮겨갈 때 그대로 재현되지 않는 문제가 반복돼 왔다.

- **Core Contribution**: 이 논문은 멀티-waypoint UAV 위치조절을 위해 터미널 구간 제어를 ‘계층형 terminal-control architecture’로 분해해, 빠른 접근과 깨끗한 정착 사이의 충돌을 구조적으로 해결하는 방식을 제안한다. Smooth approach authority, persistent-bias compensation, supervised capture/settle authority를 하나의 outer-loop velocity 인터페이스 안에서 분리·결합해, 벤치마크에 기대지 않는 핵심 제어 구조를 유지하도록 설계했다. 아울러 PyBullet→PX4/Gazebo→Vicon-tracked Tello의 staged 평가로, 단일 벤치마크 점수가 아닌 전이 가능성을 검증하는 프레임을 함께 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 단일 고정 제어 의미(명령 권한)가 접근 구간에선 공격적이어야 하고, 목표 근처에선 보수적이어야 하는 ‘모순’을 동시에 만족시키기 어렵다는 점이다. 이를 위해 최소-저크형 접근 생성과 거리 기반 fade-out으로 접근 피드포워드가 목표 근처에서 자연스럽게 힘을 잃도록 하고, translational-control에서 모드 기반으로 추적/감쇠/바이어스 보정/터미널 보정을 가중치로 조절한다. 또한 persistent-bias 성분과 rebound 같은 상태-트리거 터미널 회복을 분리하고, 채터링을 줄이기 위한 hysteresis·dwell 기반 supervision으로 transit/capture/settle 모드 전이를 안정적으로 수행한다.

- **Empirical Impact**: PyBullet 1단계에서는 stochastic wind 조건에서 타깃 특정 사전정보 없이도 late-stage wind error 평균 0.024 m(표준편차 0.017 m)를 달성했으며, 무풍에선 0.0023 m 수준으로 정착 정확도가 유지됐다. PX4/Gazebo 2단계에서는 benchmark priors 없이 transfer-oriented rule로 최종 컨트롤러를 선택했고, Strict를 1차 기준으로 삼되 Grace는 completion semantics 민감도를 해석용으로 보조 제시했다. 실물 Vicon-tracked Tello 스택에서는 Stage-A로 핵심 방법 구별 가능성을 확인하고 Stage-B로 Phase II에서 선정된 컨트롤러 패밀리의 임무/터미널 성능을 검증해, ‘벤치마크 성공의 정보성은 메인 설계와 벤치마크 리파인먼트를 분리하고 더 강한 closed-loop 평가에서도 방어 가능할 때 커진다’는 결론을 뒷받침한다.



### Racing a Wheeled Quadruped: Active Load Transfer Mitigation via Model Predictive Contro (https://arxiv.org/abs/2606.26313)
Comments:
          Accepted to the 17th International Symposium on Advanced Vehicle Control (AVEC 2026), 7-11 September 2026, Tsukuba, Japan

- **Prior Approaches**: 기존 연구는 보행 기반(legged) 동역학이나 전형적인 강체 모델을 중심으로 휠-쿼드루펫을 제어해 속도나 안정성을 다뤘지만, 플랫폼이 제공하는 추가 자유도(roll 등)를 레이싱 성능으로 직접 활용하지는 못했습니다. 또한 MPC에 학습을 결합해 모델 의존성을 줄이려는 시도는 있었으나, rigid chassis 가정 때문에 load transfer(하중 전이) 완화 같은 능동 자세 전략을 고속 레이싱 목표에 맞춰 설계하기는 어려웠습니다.

- **Core Contribution**: 이 논문은 능동 roll 제어를 레이싱 제어 목표에 포함한 계층형 프레임워크를 제안합니다. 오프라인 time-optimal raceline 생성 → 온라인 MPC가 LTR(Lateral Load Transfer Ratio)을 0에 가깝게 만들며 경로를 추종 → 저수준 whole-body RL이 16개 액추에이터로 MPC 출력을 실현하는 구조입니다. Go2-W를 동적 bicycle 모델(능동 roll 차원 포함)로 두고, 무릎 관절의 anti-roll 토크로 턴에서 몸을 안쪽으로 기울여 하중 전이를 줄이도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 고속에서의 복잡한 접촉·구동 동역학과 (2) MPC의 고수준 입력(차량 모멘트)을 실제 16모터 응답으로 정확히 매핑하는 문제였습니다. 논문은 roll 동역학을 LTR=0 조건과의 관계로 차량 평면 동역학과 분리해 MPC 비용에 LTR 저감항을 직접 넣었고, low-level는 조인트 단위 RL이 MPC 명령을 관측으로 받아 PD 기반 관절 제어를 통해 추종하도록 학습시켰습니다. 또한 시뮬레이션 domain randomization(마찰·질량·CG·액추에이터 이득·외란·노이즈)으로 sim-to-real 전이를 강화하고, 실험에서 추정한 제어/응답 지연·스케일 불일치를 MPC 가중치 조정에 반영했습니다.

- **Empirical Impact**: 실물 트랙 실험에서 능동 roll(tilt ON)은 mean LTR을 최대 44% 줄였고, fastest lap time은 8.7% 개선했으며, 피크 횡가속도는 21.3% 증가해 1.98 m/s^2까지 도달했습니다. 특히 tilt OFF(non-tilting baseline)보다 더 공격적인 주행을 가능하게 하면서도 고속 안정성을 유지해, 휠-쿼드루펫 레이싱에서 load transfer 완화의 실효성을 실험적으로 입증했습니다. 요약하면, ‘레이싱 경로 최적화+LTR 최소화 MPC+조인트 레벨 RL’의 결합이 하드웨어에서도 성능 향상을 만든다는 점이 이 연구의 의미입니다.



### NavIsaacLab: Generating Realistic Crowd via Parallel Robot Learning for Benchmarking Human-aware Navigation (https://arxiv.org/abs/2606.26265)
- **Prior Approaches**: 기존 human-aware navigation 연구는 충돌 회피나 효율 중심에서 출발해, 인간의 편안함과 사회적 맥락까지 다루려는 방향으로 확장돼 왔습니다. 다만 시뮬레이션 기반 접근은 (1) 보행자 행동을 rule 기반이나 미리 녹화된 프로시저 애니메이션으로 근사하고, (2) 관측을 “완벽한 관측”에 가깝게 가정하며, (3) 대규모 병렬 실행·센서 풍부함을 충분히 제공하지 못해 현실 적용성과 공정한 비교에 한계가 있었습니다.

- **Core Contribution**: 이 논문은 NavIsaacLab이라는 종합 프레임워크를 제안해, 보행자와 장면을 physics 기반·photo-realistic으로 생성하면서 human-aware navigation을 위한 벤치마킹/학습 환경을 표준화합니다. trajectory diffusion model과 adversarial motion learning controller(AMP)를 결합해 제어 가능하고 물리적으로 일관된 보행자 움직임을 만들고, 다양한 cross-scale 장면을 통해 SOTA 방법을 폭넓게 평가하도록 구성했습니다.

- **Technical Challenges**: 핵심 난제는 “현실감(렌더링+물리)”, “보행자 행동 충실도(다중양식+전신 움직임의 연속성)”, “GPU 병렬로 확장 가능한 샘플링”을 동시에 만족시키는 것이었습니다. NavIsaacLab은 Isaac Lab 위에서 USD 기반 장면 그래프와 GPU 병렬 시뮬레이션을 제공하고, 확률적 궤적 생성은 diffusion으로, 전신 제어는 AMP로 수행해 2D 궤적 목표를 관절 수준 동작으로 안정적으로 추적하도록 해결합니다.

- **Empirical Impact**: 검증 결과 diffusion+AMP 파이프라인이 SFM+AMP 대비 탐색/항법 신뢰성 지표(성공률 등)를 일관되게 개선했으며, whole-body motion의 다양성도 AMASS 기반 기준 대비 더 넓게 커버하는 것으로 보고됩니다. 또한 프레임워크가 센서·3D 시각 피드백을 포함한 대규모 데이터 생성을 지원함으로써, 학습 기반 navigation 정책의 빠른 학습과 sim-to-real 지향 실험 설계에 실질적 의미를 갖습니다.



### TaskNPoint: How to Teach Your Humanoid to Hit a Backhand in Minutes (https://arxiv.org/abs/2606.26215)
- **Prior Approaches**: 기존 연구는 인간 시연 기반의 로봇 모션 학습에서 주로 특정 궤적 추적에 강한 강화학습(RL)이나, 방대한 인간 데이터로부터 전신(whole-body) 컨트롤을 학습하는 접근으로 나뉘었습니다. 전자는 보상 튜닝 편향으로 비효율적·비자연스러운 행동이 생길 수 있고, 후자는 고품질 데이터 전처리(클리닝) 비용이 커지는 경향이 있습니다. 스포츠 플레이용 휴머노이드에서도 작업별 튜닝이나 상체·하체 분리 같은 우회가 흔해 학습 효율과 범용성 사이의 균형이 과제로 남아 있었습니다.

- **Core Contribution**: 이 논문은 동적 스포츠 기술을 ‘상호작용 윈도우’라는 짧은 구간으로 구조화해, 성공을 결정하는 핵심 순간만 맞추면 전체 동작을 효율적으로 학습할 수 있다고 주장합니다. 그에 따라 TaskNPoint 학습 프로토콜을 제안하며, 코치(교사)는 (1) 이산 기술 집합, (2) 기술별 1회(또는 소수) 시연, (3) 상호작용 윈도우 식별, (4) 목표를 제공하고, 학습자는 나머지 궤적·강건성·일반화를 시뮬레이션에서 채웁니다. 특히 목표를 3D 공간의 포인트와 상호작용 시점에 조건화해, 소수 시연만으로 큰 작업공간을 커버하는 다중 기술 정책을 지향합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 시연 영상으로부터 ‘정확한 상호작용 시점과 3D 상호작용 파라미터’를 추출하고, 관측·모델 불일치에도 목표 성능을 유지하는 것입니다. 논문은 PromptHMR 등으로 3D 인체 자세를 복원한 뒤 MLE 기반 다중 뷰 합의로 시연의 실패 모드를 줄이고, GMR로 로봇에 동작을 리타겟팅하면서 상호작용 파라미터(p, 방향 ν, 자세 n, 시점 t*)를 목표로 정의합니다. 학습 시에는 상호작용 윈도우 주변에서 reward를 누적하고, 훈련용 목표를 랜덤 샘플링해 단일 시연이 zero-shot으로 미관측 목표 위치까지 일반화하도록 했습니다.

- **Empirical Impact**: 실험에서는 Unitree G1 휴머노이드가 테니스 포핸드/백핸드를 치고, 사람이 던지는 공을 맞춘 뒤 킥하며, 박스를 새로운 위치에서 집어오는 작업에서 성공을 보였습니다. 단일 GPU로 1시간 미만 학습과 함께, per-task reward 튜닝 없이도 하드웨어 배치가 가능했으며, 접촉 시점 구조가 speed 변화에 대해 in-distribution 실패를 ‘완만하게’ 만들어 로봇이 넘어지지 않고 계속 동작할 수 있음을 보여줬습니다. 또한 시뮬레이션 일반화 성공률(GSR)에서 넓은 3D 볼륨 커버가 확인되어, 동적 스포츠 기술 교육을 ‘소수 시연+상호작용 윈도우’ 패러다임으로 확장할 수 있음을 시사합니다.



### RoboTales: ROBOTic Anthropomorphic LEarning Systems (https://arxiv.org/abs/2606.26213)
Comments:
          4 pages, 4 figures, HRI Companion '26: Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction, Student Design Challenge

- **Prior Approaches**: 기존 사회적 보조 로봇(SAR)과 로봇 스토리텔링은 음성, 시선, 제스처 같은 단일(또는 제한된) 채널에 의존해 정서 뉘앙스·캐릭터 정체성·서사 구조를 충분히 전달하기 어려웠다. 또한 화면 기반 아바타나 경직된 제스처 라이브러리를 쓰는 경우가 많아 감독 없이 일관된 이해·회상 향상을 만들기 쉽지 않았다. 로봇 퍼펫 연구는 주로 복잡한 마리오네트에 집중돼 일반 목적 로봇에 적용·배포가 까다롭다는 한계도 있었다.

- **Core Contribution**: RoboTales는 저비용 sock puppet(양말 인형) 엔드이펙터로 로봇이 캐릭터 기반 이야기를 자율적으로 “퍼포먼스”하도록 설계한 로봇 스토리텔링 시스템이다. 내레이션, 팔 제스처, 입(마우스) 움직임을 타이밍으로 동기화해 감정과 캐릭터성을 몸으로 표현하는 것을 핵심 기여로 제시한다. 또한 모듈형 하드웨어/소프트웨어 파이프라인을 통해 Baxter 같은 테스트베드를 넘어 다른 매니퓰레이터로의 이전까지 염두에 둔다.

- **Technical Challenges**: 퍼펫 구동을 위해 음성인식 없이도 자연스러운 입 움직임(입모양/턱 관절)을 만들면서, 잡음에 강한 신호 처리와 캘리브레이션이 필요했다. RoboTales는 오디오의 RMS 윈도우로 발화 구간을 잡고 deadband·EMA 필터로 잡음을 줄인 뒤, 턱 각도를 사전 캘리브레이션된 최소/최대 각도에 매핑해 speech recognition 부담 없이 동기화된 mouth articulation을 구현했다. 동시에 캐릭터별 제스처 변이를 위해 ProMP(probabilistic motion primitives)로 시간-비의존적인 움직임을 생성해 내레이션과 자연스럽게 결합되도록 했다.

- **Empirical Impact**: 소규모 파일럿 실험(참가자 10명)에서 puppet 모드가 gesture-only 모드보다 HRIES에서 더 높은 정서·사회성 평가를 보였고, 스토리 리콜 점수도 개선되는 경향을 보였다. 또한 전 과정에서 오퍼레이터 개입 없이 5분짜리 이야기가 완결되고, 내레이션-제스처-퍼펫 입 움직임 동기화가 안정적으로 수행돼 자율 운용(S2) 기준을 만족하는 신호를 제공했다. 후속으로 ABB IRB 120에 퍼펫을 옮긴 통합 테스트에서도 어댑터 소폭 수정만으로 빠르게 적용되어, 저비용·플랫폼 비의존적 퍼펫 기반 스토리텔링의 확장 가능성을 보여줬다.



### OmniContact: Chaining Meta-Skills via Contact Flow for Generalizable Humanoid Loco-Manipulation (https://arxiv.org/abs/2606.26201)
- **Prior Approaches**: 기존 휴머노이드 loco-manipulation 연구는 크게 동작 추적/모션 학습, HOI 추적, 태스크별 HOI 학습으로 나뉘지만, 장기 과업에서 닫힌고리(closed-loop) 연쇄와 자율 복구는 여전히 어렵다는 한계가 있었다. 명시적 HOI 표현은 정확하지만 고수준 계획에 까다롭고, 암묵적 skill embedding은 간결하지만 해석성과 조합성이 부족해 안정적인 메타스킬 chaining이 어렵다.

- **Core Contribution**: 이 논문은 contact flow(CF)를 중심으로, 장기 과업에서 메타스킬을 연쇄하고 실패를 복구하는 계층형 프레임워크 OmniContact를 제안한다. CF는 핵심 바디 궤적과 시계열 이진 접촉 신호를 담는 compact한 표현이며, 저수준 CF-Track은 CF를 조건으로 통합 기술 라이브러리를 실행하고 고수준 CF-Gen은 휴리스틱으로 미래 contact-flow 시퀀스를 생성한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘접촉을 반영하면서도’ 고수준 계획에 쓰기 쉬운 중간 표현을 설계하는 것이다. 저자들은 복잡한 dense HOI를 피하면서도 contact semantics를 보존하도록 CF를 구성하고, CF-Gen은 단계별 phase template과 물체-중심 기하 anchor로 IK를 최소화해 실시간 합성을 수행하며, 50Hz 모니터링으로 물체 상태 편차가 임계치를 넘으면 즉시 중단·재계획하는 폐루프 복구를 구현한다.

- **Empirical Impact**: OmniContact는 Carry Box에서 98.7%, Push-Stack Boxes에서 76.5%의 success를 보였고, 메타스킬에서 평균 40.9%, skill chaining에서 평균 66.5%p 수준의 개선을 보고한다. 또한 VLM과의 결합으로 의미 기반 태스크 분해 후 contact-flow를 합성해 ‘흩어진 박스를 하트 모양으로 배치’ 같은 고난도 semantically grounded 행동까지 확장 가능함을 실험적으로 보여준다.



### Morphology-Specific Closed-Loop Control of Logarithmic-Spiral Continuum Arms via Online Jacobian Error Compensation (https://arxiv.org/abs/2606.26188)
- **Prior Approaches**: 기존의 로그-나선(logarithmic-spiral) 로봇 연구는 제작 확장성과 다양한 그립 동작을 보여줬지만, 역기구학과 closed-loop 제어가 부족했다. 또한 기존에 널리 쓰이는 PCC(piecewise-constant-curvature) 계열 기준선은 형태가 다른 연성 구동계에서 정확도가 떨어질 수 있다.

- **Core Contribution**: 이 논문은 로그-나선 형태에 특화된 morphology-specific closed-loop task-space 제어 프레임워크를 제안한다. 중심선 백본과 정삼각형(equilateral) 텐던 라우팅을 갖춘 segmented tendon-driven 모델을 통해, 테스크 공간에서 정확히 목표를 추종·조절하도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 비선형 변형, 접촉, 기하학적 불일치로 인해 생기는 모델링 오차를 실시간으로 보정하는 것이다. MuJoCo에서 테이퍼된 컴플라이언스와 접촉 동역학을 반영하고, 로그-나선 kinematics로부터 analytical task-space Jacobian을 도출한 뒤 Broyden secant update와 Kalman-filter 기반 추정으로 온라인 Jacobian error compensation을 수행해 연속 보정을 달성했다.

- **Empirical Impact**: 시뮬레이션에서 궤적 추종, 자세 규제, 외란 거부, 3D 위치 추종, 위치-자세 동시 제어 등 다양한 태스크를 검증했다. PCC baseline 대비 추종 오차 감소, 자세 드리프트 억제, Jacobian 추정 오차의 상한 유지가 일관되게 관찰됐으며, 장애물 보조 reach-wrap-release, 적응형 whole-arm grasping, 멀티암 협업 물체 핸들링에도 적용돼 형태 기반 제어의 정확성과 강건성이 확인됐다.



### LiMoDE: Rethinking Lifelong Robot Manipulation from a Mixture-of-Dynamic-Experts Perspectiv (https://arxiv.org/abs/2606.26183)
- **Prior Approaches**: 기존 일반화 로봇 정책 연구는 연속 작업 적응 중 catastrophic forgetting을 겪어 이전 작업 성능을 유지하기 어렵다는 한계가 있었다. 이를 줄이기 위해 replay, regularization, architectural methods가 나뉘는데, 특히 PEFT 기반 parameter-efficient fine-tuning은 단일 작업 적응에는 효과적이지만 작업 간 공유 지식의 상호작용을 충분히 모델링하지 못해 재사용과 적응 속도가 떨어진다. MoE를 활용하는 시도도 있었으나 lifelong robot learning을 위해 재사용 가능한 skill 추출과 작업 간 결합을 통합적으로 다루는 데는 공백이 남아 있었다.

- **Core Contribution**: 이 논문은 Lifelong Mixture of Dynamic Experts (LiMoDE)라는 2단계 구조를 제안해, 로봇이 prior knowledge를 기반으로 지속적으로 작업을 확장(continual task adaptation)하도록 설계했다. 멀티태스크 pre-training에서는 motion 정보로 가동되는 dynamic MoE를 통해 재사용 가능한 skill을 학습하고, 이후 task adaptation에서는 lifelong MoE adaptation 메커니즘으로 새 작업에 대해 기존 전문가(frozen)와 새 전문가를 동적으로 조합해 지식 전이를 촉진한다. 또한 router 드리프트를 줄이기 위한 replay 전략을 더해 이전 작업 성능 저하를 완화한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 작업 간 공유되는 skill을 실제로 ‘추출’하고 (2) 그 상호작용을 효율적으로 결합하며 (3) 적응 단계에서 forgetting을 제어하는 것이다. 이를 위해 expert를 low-rank 기반으로 이질적으로 구성하고, visual-dynamics-conditioned router가 단기 조작의 복잡도에 따라 활성 expert 수를 가변하는 DyMoES를 도입했다. 이어 adaptation 단계에서는 rank가 다른 lifelong 전문가 라이브러리를 점진적으로 늘리고 top-k 조합으로 동적 결합을 수행하며, router-decorrelation 정규화와 낮은 차원의 router 계수 replay로 router drift를 억제한다.

- **Empirical Impact**: 실험은 시뮬레이션의 LIBERO lifelong learning 벤치마크와 실제 로봇 작업 모두에서 수행됐고, 기존 방법 대비 성능과 lifelong adaptation이 일관되게 개선됐다. LIBERO-LONG에서 adaptation은 7% 향상, forgetting은 3% 감소하는 결과를 보고하며, 추가로 도입되는 학습 파라미터와 추론 오버헤드는 ‘중간 수준’으로 유지되는 효율-성능 절충도 강조한다. 실제 환경 배치에서도 동일한 경향의 효과와 효율성이 관찰되어, lifelong robot manipulation에서 재사용 skill과 작업 간 결합의 실용성을 뒷받침한다.



### RMTL: Reinforced Micro-task Learning for Long-Horizon Manipulation with VLM Rewards (https://arxiv.org/abs/2606.26175)
Comments:
          16 pages, 11 figures

- **Prior Approaches**: 강화학습 기반 로봇 조작에서 보상 설계는 장기 과업에서 특히 어렵다. 수작업 밀집 보상은 튜닝이 까다롭고 취약하며, 인간 시연·선호 기반 보상 학습은 비용과 확장성이 문제다. 이를 대체하려고 고정된 pretrained vision-language model을 zero-shot reward model로 쓰는 흐름이 등장했지만, 단일 global prompt는 긴 지평 제어와 랜덤 초기조건에서 진행 신호가 near-flat해 학습 초기 탐색과 credit assignment가 막힌다.

- **Core Contribution**: 논문은 Reinforced Micro-task Learning (RMTL)을 제안하며, 조작 과업을 언어로 묘사된 소수의 micro-task로 분해해 단계별로 다른 VLM 프롬프트를 쓰게 한다. 또한 여러 카메라 view에서 계산한 VLM 보상을 평균 내어 occlusion으로 인한 신호 손실을 줄인다. 마지막으로 phase(단계) 전환을 거리 기반 rule에서 learned hierarchical manager로 바꿔, 규칙적 페이즈 선택을 end-to-end에 가깝게 학습한다.

- **Technical Challenges**: 핵심 난제는 단일 프롬프트 보상이 장기 궤적에서 충분히 민감하지 않아 PPO가 계속 “오르는” 보상 경사를 얻기 어렵다는 점이다. RMTL은 approach·align·grasp처럼 단계 정합적인 짧은 프롬프트를 piecewise-monotone 신호로 만들고, multi-view 평균으로 view-specific occlusion 잡음을 완화해 더 매끈한 학습 신호를 제공한다. 아울러 reverse curriculum으로 쉬운 초기조건에서 랜덤 초기조건까지 점진적으로 넓혀, 언어 보상만으로는 부트스트래핑이 안 되는 구간을 넘게 설계했다.

- **Empirical Impact**: FetchPickAndPlace-v4에서 single-prompt VLM 보상 기반 접근은 랜덤 초기조건 프로토콜에서 수렴에 실패(0% 수준)한 반면, RMTL은 더 유의미한 reward를 제공해 빠른 학습을 보였다. learned hierarchical manager를 포함한 full RMTL은 성공률이 약 94%→약 98%로 상승했고, rule 기반 selector와도 높은 수준으로 정렬되면서 더 잘 맞는 모습을 보였다. 저자들은 결과가 language-guided reinforcement learning을 로봇 조작으로 확장할 때 “global prompt의 한계”를 micro-task 분해와 멀티뷰 집계, 역커리큘럼이 실질적으로 극복할 수 있음을 시사한다고 정리한다.



### Reinforcement Learning Enables Autonomous Microrobot Navigation and Intervention in Simulated Blood Capillaries (https://arxiv.org/abs/2606.26154)
Comments:
          12 pages, 4 figures

- **Prior Approaches**: 기존 강화학습(RL) 연구들은 마이크로로봇 항법을 주로 이상화된 유로(개방 영역, 단순 채널, 균일 유동)에서 학습해 복잡한 생체 환경의 제약을 충분히 반영하지 못했다. 특히 혈관 내 유체 동역학, 세포(적혈구, RBC)로 인한 밀집 장애물, 실제 분지(브랜칭) 기하를 함께 다룬 시뮬레이션-학습은 부족했다. 그 결과 제어 정책이 실제 투입 조건에서 견딜 수 있는 “물리적 한계”와 “학습된 전략의 보편성”을 체계적으로 검증하기 어려웠다.

- **Core Contribution**: 이 논문은 혈관 모세혈관 네트워크를 물리 기반으로 재현한 시뮬레이션을 만들고, chemotaxis(화학주성) 신호를 이용해 deep RL 에이전트를 학습시킨다. 그 과정에서 해부학적으로 유도된 분지 기하, Lattice-Boltzmann 기반의 정상 상태 유동장, RBC의 명시적 동역학, 그리고 Brownian 요동을 함께 포함해 “생체와 유사한 난이도”를 구현했다. 더 나아가 로봇 크기와 수영 속도에 따른 성공의 금지 영역(forbidden regime)을 지도처럼 제시하고, 학습된 정책이 재학습 없이 유동 차단/해제를 수행함을 보여준다.

- **Technical Challenges**: 주요 기술 과제는 저 레이놀즈수 영역에서 추진력이 잡음(Brownian)·유동 저항·RBC 충돌에 의해 쉽게 무력화되는 조건을 안정적으로 통과하는 제어를 찾는 것이다. 저자들은 로봇 크기-속도 격자를 바꿔 다수의 독립 에이전트를 학습하고, “성공 정의(소스 근처 도달률)”와 물리학적 전이(학습 모델을 파라미터 전 범위에 적용)로 알고리즘 한계가 아닌 환경 물리의 경계를 구분한다. 또한 정책을 궤적이 아닌 actor 네트워크의 행동확률 분포로 t-SNE 및 k-means 분석해 run-and-rotate, energy-efficient search-and-sit 등 여러 보편적 전략 유형을 식별하고, 혈관 유동 조건에 적응된 형태임을 해석한다.

- **Empirical Impact**: 실험적으로 성공 확률이 중간 크기-중간 속도에서 최대(약 80%)로 나타나며, 작은 로봇은 Brownian과 국소 드래그에 취약하고 큰 로봇은 RBC 충돌 회피/기동성이 부족해 실패한다는 크기-속도 트레이드오프가 확인됐다. Forbidden regime은 단순 Brownian만 있는 경우보다 더 이동하며, 전이 실험에서도 동일한 경계가 유지돼 “환경이 허용하는 물리적 해의 영역”을 RL이 조기에 활용한다는 메시지를 강화한다. 더 중요한 응용으로, Run-and-Rotate 계열의 학습 정책만으로도 목표 위치에 화학원을 배치해 막기(emboliotherapy 유사)와 해제(thrombolysis 유사)를 재학습 없이 수행하며, 복수 지점 장애물이 있어도 기준선 수준에 가까운 유동 회복을 보였다.



### Unsupervised Memory-Enhanced Video Transformers: Obstacle Detection for Autonomous Agricultural Rover (https://arxiv.org/abs/2606.26151)
- **Prior Approaches**: 정밀 농업에서 자율 로버는 필수지만, 안전을 일관되게 확보하는 것이 여전히 핵심 과제였다. LiDAR 같은 전통 안전 센서는 식생 아래(캐노피 아래)에 있는 장애물을 감지하지 못해 위험이 남는다. 카메라 기반 supervised learning은 학습 데이터에 없던 장애물 상황에서 성능이 크게 저하되며, unsupervised anomaly detection도 로버의 이동으로 인해 생기는 동적 장면에서 흔히 약점을 보였다.

- **Core Contribution**: 이 논문은 동적 농업 장면에서 실시간 장애물 탐지를 목표로, 완전 비지도 anomaly detection 방법인 Video Memory Transformers for Anomaly Detection (VMTAD)를 제안한다. VMTAD는 정상 운용 이미지(라벨 없음)만으로 학습해, 학습 분포 밖 장애물에도 대응할 수 있도록 설계됐다. 특히 transformer 기반 표현에 전용 memory 모듈을 결합해 이전 프레임의 시간 정보를 효과적으로 활용한다.

- **Technical Challenges**: 주요 기술 난제는 로버가 움직이며 카메라 관측이 계속 변하는 동적 맥락을 anomaly 탐지기가 잘 흡수해야 한다는 점이다. VMTAD는 프레임 인코딩을 memory 모듈에서 누적·처리해 temporal context를 제공함으로써, 이동에 따른 장면 변화를 ‘정상’으로 모델링하는 방향으로 문제를 완화한다. 또한 완전 비지도 학습을 유지하면서도 실시간 동작을 위해 구조를 경량화한 변형까지 제시했다.

- **Empirical Impact**: Grillion 농업용 로버의 실험에서 VMTAD는 rapeseed 데이터셋과 같은 까다로운 환경에서도 높은 성능을 보였다. 검출 AUC 0.973, 분할 AUC 0.997 수준의 state-of-the-art 결과를 달성했으며, 경량 모델은 14 ms 추론으로 실시간성을 확보했다. 나아가 로버의 총 정지 거리(total stopping distance) 분석을 통해 안전 관점에서도 효과를 검증해 현장 적용 가능성을 높였다.



### Hallucination in World Models is Predictable and Preventab (https://arxiv.org/abs/2606.27326)
Comments:
          Interactive paper, live demo, code, dataset, and models: this https URL

- **Prior Approaches**: 기존 generative world models는 행동 제어 미래를 그럴듯하게 생성하지만, 롤아웃이 실제 동역학에서 벗어나도 “시각적으로는 자연스러운” 환각이 발생해 다운스트림 제어를 오도할 수 있다. 기존 연구는 대개 아키텍처/학습 규모를 키워 환각을 완화하려 하지만, 왜 언제 환각이 생기는지에 대한 데이터 관점의 설명이 부족했다. 또한 환각을 탐지하거나 단계별 원인을 분해하는 데 필요한 대규모·행동 라벨·라이브 시뮬레이터가 함께 제공되는 벤치마크가 제한적이었다.

- **Core Contribution**: 이 논문은 환각이 state-action 공간의 데이터 커버리지(coverage) 부족에서 집중적으로 나온다는 가설을 제시하고, 이를 파이프라인 단계별 실패 모드로 연결해 정리한다. 저자들은 MMBench2(427시간, 210 task)의 대규모 비주얼 world modeling 데이터셋과 350M 파라미터 world model을 통해, 환각을 perceptual(토크나이저), action-marginalized(행동 조건화), scene-diverging(다단 롤아웃)로 분류한다. 동시에 각 모드를 예측하는 라벨-프리 신호 3종을 설계하고, 같은 신호를 이용해 학습/온라인 데이터 수집 모두에서 완화 전략으로 연결한다.

- **Technical Challenges**: 핵심 기술 과제는 “그럴듯하지만 틀린” 롤아웃을 단계별로 진단하고, 추가 라벨 없이 런타임에서 예측 가능한 신호를 만드는 것이다. 저자들은 tokenizer round-trip residual, flow instability, inter-seed variance를 제안하되, 장면 움직임에 의한 공변량(confound)을 줄이기 위해 dynamism-normalized 형태로 정규화해 비교 가능하게 만들었다. 이후 커버리지 갭을 줄이기 위해 학습 시 coverage-aware sampling으로 격차를 메우고, 온라인에서는 환각 예측기를 curiosity reward로 사용해 예측에 취약한 전이를 우선적으로 수집하는 closed-loop finetuning 레시피를 구성한다.

- **Empirical Impact**: MMBench2에서 3가지 환각 예측 신호는 open-loop 롤아웃 품질과 강한 상관(예: Spearman ρ≈0.80)을 보이며, action ignored/scene divergent 같은 이진 환각 라벨에 대해서도 높은 분류 성능을 나타낸다. coverage-aware training은 3종 환각 모드 전반을 동시에 줄이고 롤아웃 충실도를 개선하는 방향으로 작동한다. 특히 보정된 온라인 데이터 수집과 finetuning은 완전히 unseen 환경에 대해 50개 수준의 real environment trajectories로 적응해, 인간 데이터 수집에 근접하는 데이터 효율을 보이며 커버리지 기반 환각 프레임의 유의미성을 실증한다.



### OctoSense: Self-Supervised Learning for Multimodal Robot Perception (https://arxiv.org/abs/2606.27317)
- **Prior Approaches**: 기존 멀티모달 학습은 센서가 제공하는 표현(예: 이미지, 이벤트, LiDAR 등)과 시공간 특성이 달라도 이를 충분히 정렬해 학습하기 어렵다. 또한 센서별 주파수·지연·잡음이 크게 달라지는 실제 주행/로봇 환경에서는 모달리티 간 late alignment가 성능 저하로 이어지곤 했다.

- **Core Contribution**: 이 논문은 스테레오 RGB·이벤트 카메라·LiDAR·열화상·IMU·RTK GPS·자세/주행자기정보(CAN 버스, 사족 로봇 관절각)를 한 번에 제공하는 오픈소스 센서 플랫폼 OctoSense와, 이를 기반으로 한 59시간 규모의 시간동기 데이터셋 OctoSense를 제안한다. 또한 모달리티별 토큰화와 지연 융합을 결합한 late-fusion masked autoencoder로, 실세계 로보틱스 데이터에서 멀티모달 self-supervised learning을 효과적으로 수행한다.

- **Technical Challenges**: 핵심 기술 과제는 서로 다른 센서의 표현/빈도/지연/잡음을 어떻게 모델 입력으로 일관되게 다룰지에 있다. 저자들은 모달리티별 tokenizers로 spatiotemporal 특성을 흡수하고, 추론 시 modality-specific tokens를 캐시해 새 측정치가 도착하는 즉시 처리하는 구조로 이를 해결했다.

- **Empirical Impact**: 실험에서는 late-fusion masked autoencoder가 광학흐름·깊이·세만틱 세그멘테이션·ego-motion(이동/회전/조향각) 같은 과업에서 이미지 전용 foundation model 대비 더 나은 성능을 보였다. 또한 야간이나 센서가 심각하게 열화된 상황에서도 예측의 견고함이 유지되며, 표현 계산이 NVIDIA 5090에서 6.68 ms, Orin NX에서 112 ms로 비교적 빠르다는 점이 제시된다.



### Automating Potential-based Reward Shaping with Vision Language Model Guidanc (https://arxiv.org/abs/2606.27180)
- **Prior Approaches**: 희소 보상 환경에서 강화학습은 중간 피드백이 없어 탐색이 거의 랜덤에 가깝고, 성공 보상을 정확히 어떤 궤적 요소가 만들었는지 귀속하기도 어렵다. 이를 완화하려고 보상 shaping을 쓰면 reward hacking 위험이 커지며, 특히 임의의 shaping은 목표와 무관하게 보조 신호만 최대화하는 정책을 유도할 수 있다. Potential-based reward shaping(PBRS)은 최적 정책 집합을 보존하지만, 핵심인 잠재함수(Φ)를 전문가 지식과 공학으로 설계해야 한다는 부담이 남아 있었다.

- **Core Contribution**: 본 논문은 VLM-PBRS로, 비전-언어모델(VLM)이 이미지 쌍에 대한 preference 라벨을 생성하고 이를 학습해 PBRS의 potential function을 자동 구성하는 프레임워크를 제안한다. 사용자는 목표에 대한 짧은 텍스트 설명과 관측(이미지)만 제공하면 되고, 보상 shaping 항을 수동으로 설계할 필요가 줄어든다. 또한 PBRS의 정책 불변성 덕분에 라벨 정확도에 대한 요구를 완화하여, 반복 호출 비용이 큰 대형 VLM 대신 더 작은 VLM을 사용해도 학습을 가속할 수 있음을 강조한다.

- **Technical Challenges**: 기술적 난점은 (1) VLM이 만든 preference 라벨이 노이즈/오차를 포함할 수 있는데도 PBRS로 정책 불변성을 유지해야 하고, (2) 정책 학습 중 VLM을 반복 호출해야 하므로 계산 비용을 낮춰야 한다는 점이다. 해결을 위해 VLM 출력에 대한 추가 LLM 호출 없이 단일 프롬프트로 preference 라벨을 뽑고, 학습된 선호 모델을 sigmoid+스케일링으로 잠재함수에 매핑해 goal reward의 부호/크기 조건을 만족하도록 설계한다. 그 결과, 라벨이 불완전해도 최적성 자체를 해치기보다는 sample efficiency 개선 폭이 줄어드는 방향으로 작동하게 만든다.

- **Empirical Impact**: Meta-World와 Franka Kitchen에서 VLM-PBRS는 희소 기준선보다 일관되게 학습을 빠르게 만들었고, 보상 hacking 없이 목표 성능을 향해 수렴하는 경향을 보였다. 반면 RL-VLM-F처럼 learned reward를 직접 보상으로 쓰는 방식은 VLM 라벨 오류가 잘못된 유인을 만들어 일부 태스크에서 부분 최적 정책으로 수렴하는 문제가 나타났다. 또한 실험 분석을 통해 small VLM preference label의 정확도가 sample efficiency 향상과 연결된다는 점을 보여, VLM 라벨 품질과 학습 이득의 관계를 정리했다.



### FlameVQA: A Physically-Grounded UAV Wildfire VQA Benchmark with Radiometric Thermal Supervision (https://arxiv.org/abs/2606.27128)
- **Prior Approaches**: 기존 원격탐사 VQA 벤치마크는 주로 RGB 광학 영상 중심으로 설계돼 위험(hazard) 도메인의 온도·열 신호가 필요한 질의를 충분히 평가하기 어렵습니다. UAV 화재 데이터도 탐지/분할/분류에 치우쳐, 지각을 넘어선 고수준 추론과 안전-운항 의사결정까지 묻는 멀티모달 QA 평가는 제한적이었습니다.

- **Core Contribution**: 이 논문은 UAV 기반 산불 모니터링을 위한 다지선다 Visual Question Answering 벤치마크 FlameVQA를 제안합니다. RGB와 함께 radiometric thermal TIFF(픽셀 단위 온도)를 제공해, 연기·가림·스케일 변동 환경에서도 온도에 근거한(temperature-grounded) 안전-중요 추론을 평가하도록 구성했습니다.

- **Technical Challenges**: 핵심 난관은 라벨 신뢰도를 대규모로 확보하면서도 온도 임계 기반 커버리지·존재 판정처럼 애매한 항목에서 정답성을 유지하는 것입니다. 이를 위해 MLLM-assisted annotation을 출발점으로 하고, NumPy 기반 TIFF 직접 판독(임계/퍼센트 계산), EXIF GPS+지형 모델로 고도 범주 산정, 그리고 질문 간 논리 제약(예: No fire면 hotspot 관련 답변 금지)과 인간 검수의 다단계 검증을 결합했습니다.

- **Empirical Impact**: Willamette Valley 하위셋에서 인간 평가와의 일치율은 70.78%로, 특히 공간 국소화·크로스모달 추론은 높고 PD5(연기 속 활성 화재)에서 가장 낮게 나타났습니다. LLaVA와 Qwen-VL 같은 대표 MLLM 베이스라인은 존재/방향 국소화에서 Qwen-VL이 우세했지만, 분포·커버리지(aggregation·퍼센트 추론)는 전반적으로 어려워했습니다. 결과적으로 현재 MLLM은 재난/산불 도메인 특화 적응이 필요하며, FlameVQA가 그 격차를 체계적으로 드러내는 실용 벤치마크로 의미가 있습니다.



### Rethinking Training & Inference for Forecasting: Linking Winner-Take-All back to GMMs (https://arxiv.org/abs/2606.26424)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 자율주행 궤적 예측에서 다중 후보(multi-modal) 예측은 일반적으로 GMM(Gaussian Mixture Model) 형태로 확률 가중치를 함께 내고, 이후 post-hoc로 후보를 줄이거나 병합해 평가용 M개 모드를 만든다. 이때 많은 방법이 학습에서는 winner-take-all(WTA)처럼 가장 가까운 모드에만 책임을 주는 hard assignment를 사용해 mode collapse는 막지만, 결과적으로 모드 확률이 “믿을 만하지 않게” 된다.

- **Core Contribution**: 이 논문은 WTA 학습이 GMM 학습의 전제인 ‘확률적 모드’와 불일치하며, WTA의 hard one-hot 배정이 K-means 스타일의 K-means-like hard assignment로 해석된다고 제시한다. 그 결과 모드가 의미적으로 오버세그먼트되고, 개별 모드는 낮은 확률로 쪼개져도 그 합은 큰 확률인 상황이 생겨 mode pruning을 망친다는 원인을 정리한다. 이를 바탕으로 재학습 없이 test-time 사후 보정 2가지를 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 WTA로 학습된 모델의 ‘확률 가중치’가 추론 시 GMM의 posterior로 해석되기 어렵다는 점이다. 논문은 (1) posterior-weighted merging으로 인접 후보를 가중 평균해 모드를 합치고, (2) 1-step expectation-maximization(EM)으로 hard label을 soft responsibility로 바꿔 확률 질량을 이웃 모드로 분산시키는 방식으로 해결한다. 추가로 이 접근이 WTA의 K-means적 특성과 직접 연결된다는 해석틀도 함께 제공한다.

- **Empirical Impact**: NuScenes Prediction 및 Waymo WOMD에서 Wayformer, MTR-e2e, EMP 등 여러 WTA 계열 아키텍처에 대해, 단순 greedy/NMS처럼 ‘예측 확률 하나’에 의존한 선택보다 병합·1-step EM 보정이 더 나은 최종 선택을 만든다. 특히 거리 오차 기반 지표(minADE/minFDE)에서 개선이 확인되며, 분포 품질을 반영하는 brier-minFDE 및 negative log-likelihood도 함께 좋아진다. 결론적으로 재학습 없이도 모드 확률을 더 정보적으로 정렬해 downstream planning에 유리한 형태의 모드 집합을 얻을 수 있음을 보여준다.



### PRISM: Efficient and Locally Optimal Probabilistic Planning with Reachability Guarantees (https://arxiv.org/abs/2606.26413)
- **Prior Approaches**: 로봇의 운동·관측 불확실성 하에서 장애물과 상태/입력 제약을 동시에 만족하며 경로를 찾는 문제는, 신념공간(belief space)에서 도달가능성(reachability) 보장을 세우기 어려워 난도가 높다. 기존의 제약 신념공간 플래너는 샘플링으로 다중 쿼리 로드맵을 만들고, 노드 간에 각각의 feasible trajectory를 명시적으로 찾아 reachability를 구성하지만 고차원 신념공간 커버리지가 비효율적이며 유한 시간/유한 메모리 완전성도 보장하기 어렵다. 또한 일부 방법은 robust control로 커버리지를 늘리지만 경로가 우회적이어서 평균 비용과 비용 분산이 커질 수 있고, 엣지별 chance constraint까지 엄격히 강제하지 못해 제약 위반 가능성(unsound)이 남는 경우도 있다.

- **Core Contribution**: 이 논문은 state와 control 제약이 있는 신념공간에서, 높은 커버리지와 낮은 비용을 동시에 겨냥한 multi-query motion planning 알고리즘 PRISM을 제안한다. 핵심 아이디어는 제약 하에서 상태 공분산(state covariance)의 controllability에 대한 새로운 충분조건을 도출해, 신념공간 경로계획을 “결정론적 mean(평균) 계획 + 공분산 shrinking(축소)” 문제로 분해하고, 이후 온라인 국소 최적화로 feasible 경로의 비용을 더 줄이는 것이다. 또한 제약(장애물·액추에이터)까지 고려하되, (start/goal 분포에 대한 완화된 가정하에) 유한 시간 내 완전성(completeness)과 soundness를 함께 증명한다.

- **Technical Challenges**: 가장 큰 기술 난제는 제약이 존재할 때 신념공간에서 공분산이 목표로 “도달가능”해지는 조건을 유한 시간 내에 보장하는 것이며, 이는 제약이 결합된 확률적 reachability를 직접 다루면 계산적으로 매우 어렵다. PRISM은 controllability 조건을 이용해 mean과 covariance의 역할을 분리하고, 엣지마다 chance constraint 만족을 유지하도록 로드맵 구성과 constraint tightening을 설계한다. 마지막으로, 그래프에서 뽑힌 후보 경로를 신념공간에서의 온라인 local optimization으로 다듬어 비용을 감소시키되, 전체 플래닝의 soundness/완전성 성질이 깨지지 않도록 구성한다.

- **Empirical Impact**: 시뮬레이션에서 PRISM은 기존 state-of-the-art 신념공간 플래너 대비 로드맵 커버리지를 크게 높이면서도 평균 비용과 비용 분산은 더 낮게 산출한다. 특히 난이도가 낮고 중간인 시나리오에서는 100% 커버리지를 달성했고, PRISM의 커버리지 가정에 일부 어긋나는 최악 조건에서도 97~100% 커버리지를 보인 반면 다른 방법들은 45% 미만에 머물렀다. 이 결과는 제약 신념공간에서 “샘플 효율 + 비용 품질 + 유한 시간 보장”을 동시에 만족시키는 접근이 가능함을 보여주며, 실환경 대응이 중요한 로보틱스 planning 연구에 실용적 동력을 제공한다.



### Charting the Growth of Social-Physical HRI (spHRI): A Systematic Review Pipeline Augmented by Small Language Models (https://arxiv.org/abs/2606.26382)
Comments:
          5 pages, 3 figures, 2 tables, Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction

- **Prior Approaches**: 기존 spHRI 관련 설문/리뷰는 HRI·HCI·햅틱스·로보틱스의 관점을 분절된 채로 다루는 경우가 많아, 일관된 어휘·분류 틀이 부족하다는 한계가 있었다. 또한 많은 리뷰가 로봇의 ‘사회적 의도’가 드러나는 물리 상호작용을 포괄적으로 다루지 못해, robject·웨어러블·햅틱 디바이스 같은 범주가 누락되곤 했다. 최근에는 LLM을 체계적 문헌고찰 파이프라인에 활용하려는 시도가 늘었지만, 클라우드 비용·에너지·접근성 문제 때문에 현장 적용이 제한적이었다.

- **Core Contribution**: 이 논문은 로컬에서 구동 가능한 lightweight small language models(SLMs; 1.5B 미만 파라미터)가 spHRI 대규모 systematic review의 title/abstract screening에서 ‘보조적 세컨드 패스’로 얼마나 유효한지 실험한다. SLM을 전문가 대체가 아니라, 사람이 놓친 관련 논문을 회수해 워크플로를 확장하기 위한 안전망으로 위치시킨다. 그 결과, SLM 앙상블이 사람 리뷰어가 놓친 관련 논문을 추가로 찾아내며 실제 성능 향상을 정량화했다.

- **Technical Challenges**: 핵심 기술적 난제는 spHRI처럼 개념적 뉘앙스가 큰 분야에서, 짧은 텍스트(제목/초록)만으로 관련성 판단을 하되 false positive를 억제하는 것이다. 연구진은 Llama3.2, Gemma3, Qwen3, DeepSeek-R1 등 4개 SLM을 로컬 Ollama 환경에서 구동하고, 리뷰 목적·포함/제외 기준을 구조화 프롬프트로 제공한 뒤, unanimity(모두 yes일 때만 include) 규칙으로 보수적으로 플래그를 줄였다. 이후 플래그된 집합만 사람 2인이 재스크리닝하며 reference set을 업데이트해 모델의 분류 정확도를 재평가했다.

- **Empirical Impact**: 실험에서는 spHRI 출판량이 1992~2025년 동안 비선형적으로 지속 성장함을 확인해 분야의 emergent 특성을 지지했다. 개별 SLM은 사람 수준과 완전히 같지는 못했지만, 결합 앙상블은 false positive rate를 크게 낮추면서도 사람 리뷰가 놓친 논문 39편을 추가로 회수해 최종 relevant 데이터셋에서 10.29%를 차지했다. 무엇보다 SLM은 항목당 약 0.21초 수준으로 초고속 스크리닝이 가능해, 전문가의 재심 범위를 통제하면서도 대규모 리뷰를 지속가능하게 만드는 데 의미가 있다.



### Fast LeWorldMod (https://arxiv.org/abs/2606.26217)
- **Prior Approaches**: LeWM 같은 JEPA 기반 재구성-free 시각 월드 모델은 픽셀을 직접 복원하지 않고, 잠재 임베딩을 예측해 reward-free 목표 조건 계획을 지원합니다. 다만 후보 행동열 평가는 로컬 one-step latent transition을 자가회귀로 반복 롤아웃해야 해서 계산 비용이 커지고, 초기 예측 오차가 horizon이 늘수록 누적되며 계획 신뢰성이 떨어집니다. 즉 dynamics-query 인터페이스가 순차 롤아웃에 묶여 있는 점이 병목으로 작동합니다.

- **Core Contribution**: Fast-LeWM은 “한 스텝 전이 예측” 중심 인터페이스를 “action-prefix prediction”으로 바꿔, 현재 잠재 zt와 행동 접두(prefix)를 입력받아 여러 horizon의 미래 잠재를 병렬로 직접 예측합니다. 행동 접두를 기본 예측 단위로 삼아, 서로 다른 길이의 prefix가 누적시키는 action effect를 각 horizon별로 분리해 학습하도록 설계했습니다. 또한 prefix 토큰에 대한 dense prefix-level supervision을 통해 연속적인 상태 진화를 더 직접적으로 강제합니다.

- **Technical Challenges**: 핵심 과제는 (1) 후보 행동열을 다 horizon에서 빠르게 평가하면서 (2) 롤아웃처럼 단계별 중간 예측 오차가 누적되지 않게 만드는 것입니다. Fast-LeWM은 행동열을 causal Transformer로 인코딩해 horizon별 prefix 토큰을 만든 뒤, 병렬 latent predictor가 anchor latent(z t)와 각 prefix 토큰을 조건으로 대응 미래 latent를 한 번의 forward pass로 산출하게 했습니다. 더불어 현재 상태를 state token으로 추가 조건화해 동일 prefix라도 초기 구성에 따라 달라지는 action effect를 구분하도록 했습니다.

- **Empirical Impact**: 네 가지 goal-conditioned planning 작업에서 Fast-LeWM은 평균 성공률을 LeWM 85.8%에서 90.5%(self-consistency 포함 92.0%)로 끌어올리면서도 계획 시간을 크게 줄였습니다. dynamics-evaluation은 31.4s→8.0s, 전체 CEM solve time은 54.4s→28.3s로 감소했으며, open-loop latent loss의 증가 속도도 horizon이 길어질수록 더 완만했습니다. 추가적으로 probe 실험에서 물리 변수(위치·각도 등)에 대한 비선형 정보 보존이 강화되는 경향이 관찰돼, action-prefix 기반 학습이 더 풍부한 상태 표현을 만든다는 점을 뒷받침합니다.



### Large-Scale Tunnel Air-Ground Collaboration With FLISP: Fast LiDAR-IMU Synchronized Path Planner (https://arxiv.org/abs/2606.25393)
Comments:
          24 pages, 31 figures, 5 tables. Author accepted manuscript. This work was supported by the State Key Laboratory of Autonomous Intelligent Unmanned Systems. The authors also thank the KinaMind Society for its inspiring environment and support

- **Prior Approaches**: 기존 터널/penstock 점검 로보틱스는 전역 지도나 SLAM 기반의 map-centric 접근이 많아, 반복적이고 특징이 빈약한 구간에서 드리프트가 누적되기 쉽다. 또한 그리드/샘플링 기반 다중 에이전트 플래닝은 지도 래스터화·계산비용 때문에 실시간성 확보가 어렵거나, 로컬 반응형 접근은 blind curve에서 위상적 데드락과 운동학 불일치에 취약하다. 이 때문에 UGV와 UAV를 동시에 운용하는 동기화된 경로 계획이 실제 현장 조건에서 일관되게 성능을 내기 어렵다는 한계가 지적된다.

- **Core Contribution**: FLISP(Fast LiDAR-IMU Synchronized Path Planner)는 전역 지도를 만들지 않는 mapless 프레임워크로, UGV에 장착한 단일 LiDAR-IMU가 환경 기하를 추정해 UGV-UAV 팀의 동기화 경로를 실시간 생성한다. UGV용 고속 장애물 회피(개선된 Firefly Algorithm)와 UAV용 동적 반복 최적화기(계층적 제약 반영)를 분리 설계하고, 다중 단계(거친 센터라인→베이지안 정제→안전 복도 기반)를 통해 운동학적 실현 가능성을 확보한다. 특히 상태추정 드리프트가 커지는 구조를 피하고, UGV 중심의 비대칭 협업 파이프라인으로 “mapless”를 상태추정부터 멀티 에이전트 제어까지 확장한다.

- **Technical Challenges**: 주요 기술 난제는 (1) GNSS-denied 터널에서 장거리 안정 경로를 만들되 SLAM 드리프트를 피하는 것, (2) UGV의 저응답 조향/롤 제약과 UAV의 비행·통신 제약을 동시에 만족하는 다중 제약 최적화를 실시간으로 수행하는 것이다. FLISP는 터널 중심선(기하 피팅)과 동적 빈(bin) 기반 코리더를 먼저 만들고, 이 센터라인을 이웃 기반 예측 오차의 베이지안 추정으로 outlier를 보정한 뒤 지형 고도까지 투영해 기준 경로를 정교화한다. 장애물은 경로 정렬 1D 격자(광역 필터)로 후보를 줄인 뒤 좁은 단계의 기하 포함 검증을 수행하고, UGV 회피는 터널 원형 지형에서 rollover를 막기 위해 tilt 패널티를 벽에 가까워질수록 비선형으로 키우는 “soft barrier”를 적용하며, UAV는 UAV용 비용함수(안전/부드러움/진행/고도 일관성)로 동적 샘플링 반복 최적화해 매끄러운 비행 경로를 만든다.

- **Empirical Impact**: 1.2 km급 수력 발전 터널에서의 실험/검증 결과, FLISP는 100% 성공률과 약 7 ms 지연을 달성했으며 지도 기반(예: Fast-LIO2 + A*)의 맵 래스터화 오버헤드와 샘플링 기반(LiDAR-SAM + RRT*)의 샘플링 불안정 문제를 우회했다고 보고한다. 속도 측면에서는 그리드 기반 대비 약 7배, 샘플링 기반 대비 3차수(three-order-of-magnitude) 개선을 제시한다. 또한 Gazebo 기반 고충실도 시뮬레이션에서 다중 시나리오(직선/곡선/연속 곡률), 동적 장애물 회피, 런타임 효율까지 정량 통계를 통해 재현성을 확인해, 특징이 열악한 선형 인프라 점검에서 확장 가능한 실용성을 시사한다.



New uploads on arXiv(cs.MA)

### Kiko: Programming Agents to Enact Interaction Protocols (https://arxiv.org/abs/2606.26156)
- **Prior Approaches**: 기존 에이전트 프로그래밍 모델은 의사결정 로직과 공개 의사(메시지) 간 연결을 충분히 추상화하지 못해, 분산된 비동기 환경에서 프로토콜 준수 구현이 복잡해지는 문제가 있었다. 프로토콜 언어/프레임워크들은 종종 메시지 순서를 중심으로 설계되거나(UML 다이어그램 등), 내부 추론은 지원하지만 프로토콜 제약을 직접 다루지 못해 유연한 분산 의사결정을 제한했다.

- **Core Contribution**: Kiko는 프로토콜 역할(role)에 기반해 에이전트를 구현하도록 돕는 decision-oriented 프로그래밍 모델로, 핵심 추상화는 decision maker(의사결정자)다. 개발자는 유효한 결정들 중에서 선택해 상호 호환되는 메시지 집합을 생성하는 하나 이상의 decision maker를 작성하며, 통신 서비스/전송 순서 문제를 모델이 숨겨 사업 로직에 집중할 수 있게 한다.

- **Technical Challenges**: 비동기 메시징(순서 보장 없음, 손실 가능) 위에서도 프로토콜의 정보 인과성(causality)과 무결성(integrity)을 만족시키며, decision maker가 선택한 여러 메시지의 조합이 동시에 모순되지 않도록 검증하는 것이 핵심 기술 과제다. Kiko는 BSPL의 정보 프로토콜(순서 대신 정보 제약)을 채택하고, 에이전트 히스토리와 enactment(MAS 식별자/키) 컨텍스트를 기준으로 가능한 message form을 계산한 뒤 emission attempt(생성된 메시지 집합)을 일관성/호환성/파기 가능성까지 포함해 검증·필터링하여 부적합한 집합은 거절한다.

- **Empirical Impact**: 논문은 Kiko에 대한 operational semantics를 제시하고, Kiko 에이전트가 프로토콜 규약에 준수(protocol compliant)하며 임의의 프로토콜 enactment를 실현할 수 있음을 정당화한다. 또한 compliance-checking을 최적화하는 방법과 그 타당성을 함께 제시해, 분산 MAS에서 프로토콜 기반 개발을 실용적인 수준으로 끌어올렸다는 점에서 의미가 크다.



### Resilient Output Containment under Undisclosed Leader Dynamics and Actuator Attacks (https://arxiv.org/abs/2606.27257)
Comments:
          21 pages, 12 Figures

- **Prior Approaches**: 기존 containment tracking 연구는 리더의 동역학 또는 exosystem 파라미터를 추정/재구성하는 관찰자 기반 분산 프레임워크가 많았지만, 리더 모델 비공개 조건과 충돌한다. 슬라이딩 모드·불연속 프로토콜은 리더 바운드(속도 상한, motion envelope)를 몰라도 다룰 수 있으나 채터링을 유발하기 쉽고, 연속화된 방법은 주로 undirected 또는 특수한 directed 위상에서만 성립해 일반 directed 리더-rooted 그래프에 대한 연속적 비선형 적응 보장이 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 actuator cyber-attack이 존재해도 이질적(multi-input/multi-output, 고차 relative degree) 다중 에이전트가 리더들이 만드는 convex hull을 추종하도록, 네트워크 계층과 로컬 계층을 분리한 2-layer 연속 제어 구조를 제안한다. 네트워크 계층은 리더 모델·속도 바운드·그래프 전역 지식 없이도, 이웃이 교환하는 network-interface 상태만으로 task-space 명령을 생성하고, 로컬 계층은 부분 상태 측정 하에서 actuator 공격을 가상-액추에이터로 보상해 명령 실행을 안정적으로 만든다.

- **Technical Challenges**: 핵심 난점은 (1) 리더 동역학과 리더 속도/운동 envelope가 숨겨져 있어 추정기·보상기 설계에 필요한 정보가 제한된다는 점, (2) 공격이 state-correlated·input-correlated·bounded exogenous false-data 형태로 동시에 나타나며(게다가 공격 계수의 크기는 미지), (3) directed 그래프에서 비대칭 정보 흐름까지 고려해야 한다는 점이다. 논문은 이 문제를 연속 2계층(adaptive interaction protocol + continuous adaptive virtual-actuator)으로 풀고, leader-rooted united spanning-tree 조건에서 nonsmooth Lyapunov 해석을 통해 명령 레벨 asymptotic containment를 보이며, 그 오차가 로컬 tracking-error에 의해 물리 출력의 잔차로만 남는 구조를 구성한다.

- **Empirical Impact**: 쿼드로터 네트워크와 damped suspended load 시스템 시뮬레이션에서, 공격 복구와 containment tracking 성능을 정량·정성적으로 확인하며 제어기 설계가 리더 모델 비공개 및 알려지지 않은 리더 속도 바운드 환경에서도 작동함을 보여준다. 특히 directed 위상에서의 공격 복원력과 연속성 기반 채터링 억제 관점에서, cyber-physical MAS 보안·제어 커뮤니티에 실용적인 설계 방향(네트워크-로컬 분리, 부분 측정 기반 가상-액추에이터, 연속 적응 프로토콜)을 제공한다.



### Mostly Automatic Translation of Language Interpreters from C to Safe Rus (https://arxiv.org/abs/2606.27122)
- **Prior Approaches**: C 프로그램을 safe Rust로 자동 변환하는 연구는 있었지만, 기존 rule-based 번역기는 C의 저수준 구조를 그대로 유지해 비관용적 Rust가 나오기 쉽고 안전성도 제한적이었습니다. LLM 기반 접근은 주로 함수 단위 분해를 쓰는데, 함수별로는 실행 가능한 통합 검증이 어려워 라이브러리 수준에 성과가 집중돼 왔습니다.

- **Core Contribution**: Reboot은 C 인터프리터 프로그램을 입력 테스트와 함께 받아, 대부분 자동으로 safe Rust로 변환한 뒤 제공된 테스트를 100% 통과시키는 기법을 제안합니다. 또한 Reboot은 6개 인터프리터(awk, picoc, gnu-bc, wren, mujs, pocketpy)를 각각 최대 23k LoC 규모까지 변환했으며, 각 변환은 1~11회의 짧은 사용자 개입으로 완료됐습니다.

- **Technical Challenges**: 핵심 난점은 Rust의 소유권/차용 규칙 때문에 C 코드를 안전하고 구조적으로 재배치해야 하는데, 이 과정은 LLM 에이전트가 실패·환각·무한 루프에 빠질 확률이 높다는 점까지 겹친다는 것입니다. Reboot은 (1) feature reduction으로 인터프리터의 언어 기능 단위(예: 예외 처리, 정규식)를 점진적으로 빼고 되돌려 “각 단계가 완결 실행 가능한 이정표”가 되게 만들고, (2) multi-agent의 출력 검증(VV)과 진행 이력 기반 피드백(HH), FSM 가드로 워크플로우를 통제해 장기 실행을 안정화합니다.

- **Empirical Impact**: 실험에서 변환물은 제공된 테스트를 모두 통과했을 뿐 아니라, 번역 과정에 노출되지 않은 별도 validation 테스트에서 62%~92% 통과율을 달성했습니다. 보안 관점에서는 mujs에 과거 CVE 20개를 재삽입한 사례에서 heap buffer overflow, use-after-free, stack buffer overflow 같은 메모리 취약점이 safe Rust 변환으로 제거됐다고 보고했으며, 대부분 릴리즈 빌드 기준 median 느려짐은 약 1.28x~1.51x로 관찰됐습니다. 또한 ablation 결과 feature reduction이 multi-agent만 쓸 때보다 validation pass rate를 6%~20% 개선해 실제 정확도 향상에 기여함을 확인했습니다.



### Scalability of Morality: A Particle-Based Numerical Study on the Decoupling of Law and Ethics in Large-Scale Populations (https://arxiv.org/abs/2606.27039)
- **Prior Approaches**: 기존 사회물리학·계산사회과학 연구는 Ising 모형, Schelling 분리 모형처럼 미시 규칙이 거시 패턴을 만든다는 관점을 통해 협력·행동 변화를 설명해왔다. 또 ABM에서 로컬 모니터링과 보상·처벌이 협력 균형을 지지할 수 있음을 보여줬지만, 법(제도)과 윤리(자율 규범)가 서로 어긋날 때의 동학을 충분히 수치화하지는 못했다. 특히 대규모 익명성(anonymity)이 개인 제재의 효율을 “어떻게” 희석시키는지에 대한 엄밀한 임계점 정량이 부족했다.

- **Core Contribution**: 이 논문은 입자(particle) 기반 계산 틀로 ‘Scalability of Morality’—인구 규모가 커질수록 공식 법은 유지되지만 분산된 윤리 피드백은 붕괴하는 현상—을 모델링한다. 각 에이전트를 유한 메모리 L과 동적으로 진화하는 unethical probability μ로 두고, 비선형 social pressure switch가 동기 변화 ρ를 구동하도록 설계했다. 그 결과, 인구가 무한대로 갈 때 법-윤리 상관이 약해지며 행동이 minimalist legal floor(‘unethical but legal’)로 이탈하는 구조를 phase transition으로 formalize했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 로컬 재접촉 기반 제재가 인구 규모 증가로 실제로 어떻게 소멸되는지, (2) 그 영향이 비평형 상태에서 어떤 임계 임펄스와 히스테리시스(hysteresis)를 만드는지였다. 저자들은 각 에이전트에 Whitelist/Blacklist로 상한 L의 기억을 부여하고, 재등장 확률이 O(L/N)로 줄어 제재 네트워크가 구조적으로 희석됨을 Monte Carlo ensemble로 재현했다. 또한 η/γ/λ의 비선형 전환함수(쌍곡탄젠트 switch)와 damping factor를 통해 non-Markovian inertia와 경로 의존적 비가역 붕괴(히스테리시스 루프)를 실험적으로 관측 가능하게 만들었다.

- **Empirical Impact**: 시뮬레이션은 population N을 sweep했을 때 비선형적인 거시 거동 변화가 나타나는 critical scaling point를 확인했으며, N≫L 영역에서는 로컬 accountability가 사실상 0에 수렴해 윤리 기준선에서(global norms) global drift가 발생했다. 또한 매개변수의 switch 민감도에 따라 충격 후 복원 가능한 영역과 되돌릴 수 없는 윤리 붕괴 영역이 갈리며, path-dependent hysteresis의 형태로 나타남을 보여줬다. 이 결과는 CPSS/거버넌스 설계에서 “익명성과 상호작용 희소성”이 분산 윤리 제어의 한계를 결정한다는 점을 수치적 기준으로 제공한다는 의미가 있다.



### Semantic Early-Stopping for Iterative LLM Agent Loops (https://arxiv.org/abs/2606.27009)
Comments:
          7 pages, 5 figures, 4 tables. Open implementation, machine-checked theory, and reproducible harness: this http URL

- **Prior Approaches**: 기존 Writer→Critic 같은 멀티에이전트 LLM 루프는 max_iterations 같은 정수 카운터로 종료돼, 내용이 여전히 좋아지는지(개선 여부)를 보지 못합니다. 그래서 쉬운 입력에선 토큰을 낭비하고, 어려운 입력에선 너무 일찍 끊겨 품질이 손상될 수 있습니다. 또한 “어떤 라운드가 가장 좋은지”보다는 “몇 번째에 멈췄는지”에 최적화되는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 고정 iteration cap을 없애고, 연속 초안의 의미 변화(코사인 거리)가 일정 임계값 아래로 k번 유지되고, 동시에 품질 지표 Information Score가 더 이상 개선되지 않을 때 종료하는 semantic early-stopping을 제안합니다. 다만 이전 버전에서 주장했던 Banach contraction 같은 과도한 수학적 보장은 지키지 않고, 실제로 증명·머신체크 가능한 termination(종료 보장)과 well-definedness만을 전면에 둡니다. 결론적으로 “언제 멈출지(when)”를 “품질과 의미 수렴을 함께 확인해 멈추기”로 바꾸면서, 필요하면 judge 비용까지 분리해 설계합니다.

- **Technical Challenges**: 핵심 난제는 두 가지였습니다: (1) LLM 생성이 결정론적이지 않기 때문에 수렴을 항상 보장하는 강한 contraction 가정을 세우기 어렵고, (2) 의미 변화나 품질 신호를 계산하는 과정에서 judge 호출 같은 평가 비용이 숨게 섞일 수 있다는 점입니다. 저자들은 failsafe 기반의 우선순위 캐스케이드로 최대 라운드 내 종료를 증명하고(증명은 코드로 재검증 가능), judge-efficient 프로토콜로 매 질문의 전체 궤적을 한 번 생성해 정책별로 재생(replay)·캐싱하며 operational tokens(정책이 쓰는 토큰)과 evaluation tokens(측정용 토큰)를 분리해 공정 비교를 가능하게 했습니다.

- **Empirical Impact**: HotpotQA 다중홉 RAGAS 기반 실험에서 judge-free entropy_only 스토퍼는 max_iterations 대비 operational tokens를 38% 줄이면서도 품질은 parity 수준으로 유지했습니다(Delta-IS=-0.004, p=0.81). 반면 라운드마다 judge를 호출해 품질 개선을 게이팅하는 full SHP는 비용만 크게 늘려(+129%) 역효과였고, 이로써 “온라인 품질 게이팅은 값이 비쌀 수 있다”는 실측 메시지가 강화됩니다. 또한 각 라운드 중 최적 라운드를 고르는 oracle은 모든 실용 정책보다 IS를 +0.115 끌어올려, 향후의 열린 문제를 ‘best-round identification’으로 명확히 재정의했습니다.



### Scientific discovery as meta-optimization: a combinatorial optimization case study (https://arxiv.org/abs/2606.26728)
Comments:
          35 pages, 6 figures

- **Prior Approaches**: 기존 자동 과학 탐색은 LLM이 가설·코드·실험·논문까지 생성하는 루프를 만들거나, 진화/트리서치와 결합해 알고리즘을 찾는 방식이 주류였다. 하지만 이런 시스템은 무엇을 ‘좋다’고 평가할지(목적함수·평가 기준)를 고정하거나 단계적으로만 바꾸는 경향이 있어, proxy objective 최적화 과정에서 reward hacking(메트릭 편승) 문제가 쉽게 발생한다.

- **Core Contribution**: 이 논문은 과학 탐색을 단순 최적화가 아니라 meta-optimization(메타 최적화)으로 공식화하고, 목표함수 자체가 함께 진화하도록 설계한다. 핵심 기여는 consensus objective aggregation으로, LLM이 생성한 여러 objective function을 Kendall’s τ 순위 상관으로 신뢰도를 추정해 correlation-weighted voting과 weighted Borda count로 하나의 합의 평가로 묶는다. 또한 age decay로 오래된 기준의 영향이 줄고 새로운 기준이 반영되며, 불일치하는 평가 기준은 자동으로 약화되어 자기교정적 평가가 가능해진다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘진짜 연구 목표’를 정확히 반영하지 못하는 proxy 목적함수들이 서로 편향·오류를 가질 때, 그중 어떤 것을 믿고 언제 버릴지 정하는 것이다. 저자들은 각 objective의 생성 라운드 정보를 age decay에 넣고, median Kendall’s τ 기반의 clamped 가중치로 다수와 합의되는 기준에 더 무게를 주는 구조를 통해 outlier와 잡음(또는 적대적 목적)을 억제한다. 더 나아가 agreement 기반 다수결이 echo chamber로 굳어질 경우를 대비해 meta-agent가 추가 weight multipliers로 군집 지배를 깨는 보정도 제공한다.

- **Empirical Impact**: 실험은 planted random 3-SAT을 대상으로 digital MemComputing machines(DMM)에서 알고리즘을 설계하는 사례연구로 진행됐으며, 414개 솔버 디자인과 42개의 co-evolving objectives를 함께 탐색했다. 그 결과 baseline의 scaling이 N^{2.51} 수준에서 N^{1.33}로 감소하고, 가장 큰 테스트 인스턴스에서 약 67배 속도 향상(∼67×)을 달성했다. 문제-agnostic 메타 최적화 프레임워크로서, 자동 과학 발견에서 objective specification 및 reward hacking을 완화하면서 성능을 실증했다는 점에서 의미가 크다.



### SOLAR: AI-Powered Speed-of-Light Performance Analysis (https://arxiv.org/abs/2606.26383)
- **Prior Approaches**: 기존 SOL 분석은 분석가가 수식을 직접 세우는 방식이어서 오류가 잦고, 모델 개발 속도와도 연결이 약했다. FLOP/파라미터 카운터는 연산량 중심이라 메모리 트래픽을 충분히 반영하지 못했고, 프로파일러는 달성 성능을 보여줄 뿐 이론적 최소 실행시간(바운드)에는 답하지 못한다. LLM의 zero-shot 성능 추정은 복합 워크로드에서 오차가 크게 커져 캐시-aware 타일링 같은 탐색형 분석을 신뢰하기 어렵다.

- **Core Contribution**: Solar는 PyTorch와 JAX 소스코드에서 출발해 자동으로 “검증된” Speed-of-Light(SOL) 바운드를 도출하는 프레임워크다. 핵심은 LLM이 Affine Loop IR로 번역한 뒤 출력 비교로 타당성을 게이트하고, 이후 deterministic 흐름이 einsum graph로 들어가 unfused/fused/cache-aware SOL을 계산한다. 또한 multi-fidelity 분석을 제공해 바운드를 더 촘촘히 만들고 최적화 병목을 가시화한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 LLM이 만든 중간 표현이 항상 정확해야 하면서도 다양한 연산/언어를 포괄해야 한다는 점이다. Solar는 Affine Loop IR을 기계적으로 einsum으로 끌어낼 수 있는 형태(정적 아핀 루프 커널)와, 나머지 조합/데이터이동/희소 연산을 담는 composition layer로 분리하고, Numba 실행 기반의 숫자 출력 검증으로 오류를 차단한다. 이후 einsum subscript 구조를 이용해 MACs·메모리 트래픽을 닫힌형(closed form)으로 도출하고, 그래프 의존성에 기반한 fusion·cache-aware 타일링(Orojenesis)까지 분석해 하드웨어 제약을 반영한다.

- **Empirical Impact**: KernelBench에서 Solar는 100% validated analysis coverage를 보이며 관측된 SOL violation이 0건이었다. fused SOL 기준 headroom이 L1~L4에서 최대 수십 배까지 벌어지고(예: L3에서 eager 54.6×, compile 47.7×), fusion과 그래프 수준 최적화가 복합 워크로드에 특히 중요하다는 결론을 얻었다. 또한 JAX/Flax와 로보틱스(예: 500Hz 서보 루프)까지 확장해 최적화 기회를 찾고, inverse roofline로 플랫폼 수준의 최소 대역폭/연산 요구량까지 제시함으로써 하드웨어 프로비저닝과 알고리즘 설계에 모두 실용적인 근거를 제공한다.



### Instruction Bleed: Cross-Module Interference in Prompt-Composed Agentic Systems (https://arxiv.org/abs/2606.26356)
Comments:
          8 pages, 2 tables. Accepted to the ICML 2026 Workshop on Failure Modes in Agentic AI (FAGEN), Seoul, South Korea

- **Prior Approaches**: 기존 연구와 벤치마크는 prompt injection, cognitive degradation, multi-agent fault propagation, compositional privacy leakage처럼 비교적 ‘명시적 실패’에 집중해왔다. 반면 prompt 모듈을 여러 텍스트 조각으로 런타임에 이어 붙여 LLM이 정책처럼 해석하는 구조에서는, 모듈 간 동시 간섭을 직접 측정하는 평가는 거의 없었다.

- **Core Contribution**: 이 논문은 compositional behavioral leakage(CBL)를 ‘비의도 편집이 공유 컨텍스트 창에서 다른 모듈의 행동을 조용히 바꾸는 현상’으로 정의하고, 이를 탐지·검증 가능한 형태로 형식화했다. 또한 CBL을 재현할 수 있는 reusable three-channel protocol(C0–C3 조건), 반증 가능한 prediction set, 그리고 prompt-composed agentic system의 독립적인 시스템 클래스 특성화를 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 트랜스포머 self-attention이 모듈 경계를 보장하지 않아 delimiter가 사실상 isolation을 제공하지 못한다는 점이다. 저자들은 비초점 모듈을 volume(모듈 추가), content(의미 내용), form(형식) 채널로만 국소 교란해, focal 모듈 점수 변화가 ‘semantic 내용 채널’에서만 유의미하게 나타나는지 분리해 측정했다.

- **Empirical Impact**: 배포된 job-evaluation 에이전트에서 Claude Sonnet 4.6으로 144회 실험한 결과, semantic content 채널(C2)에서 focal cv-match 점수가 d=0.63만큼 이동했으며 bootstrap 95% 신뢰구간이 0을 배제했다(다만 recommendation flip은 관측되지 않는 sub-threshold 영역). 이는 표준 QA가 ‘결정 뒤집힘’ 위주라면 놓칠 수 있는 조용한 점수 드리프트가, 실제 운영에서 누적·증폭될 수 있음을 시사하며 prompt-composed agent 평가에 cross-module interference 측정이 요구된다고 주장한다.



### The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators (https://arxiv.org/abs/2606.26294)
Comments:
          12 pages main text + 21 pages appendix (37 pages total, incl. references); 10 figures (6 main text + 4 appendix); 10 tables (2 main text + 8 appendix). Preliminary preprint; work in progress. Keywords: self-improving agents, learned evaluation, multi-agent systems, auto- mated scientific discovery, controlled utility evolution, co-evolutionary search, autoresearch

- **Prior Approaches**: 기존의 self-improving agents는 agentic coding 벤치마크에서 성능이 SOTA인 경우가 많았고, 이후 일반 도메인으로도 확장돼 왔습니다. 다만 이들은 고정된 verifier/벤치마크/라벨드 데이터셋 같은 stationary한 평가 기준을 가정해, 에이전트가 발전하는 동안 평가 환경이 함께 변하는 상황을 충분히 다루지 못했습니다. 결과적으로 진화의 핵심인 ‘환경이 적응에 따라 바뀐다’는 특성이 재귀적 자기개선 루프에 반영되지 않는 한계가 있습니다.

- **Core Contribution**: 이 논문은 평가를 개선 루프 안으로 넣어, evolving evaluators·adversarial objectives·dynamic utilities까지 포함하도록 recursive self-improvement의 평가 방식을 재구성합니다. 그 중심 기법으로 Red Queen Godel Machine (RQGM)이라는 진화적 프레임워크를 제안하며, 비정상(non-stationary) 유틸리티 아래서도 self-improvement 보장들이 에폭 단위로 성립하도록 설계합니다. 핵심 아이디어는 에폭 내부에는 고정 평가 기준을 쓰되, 유틸리티는 에폭 경계에서 제어적으로 업데이트해 목표 변화에도 안전하게 검색을 진행하는 것입니다.

- **Technical Challenges**: 가장 큰 technical challenge는 유틸리티가 변하는 비정상 조건에서, 기존의 자기개선 보장(학습/검색의 수렴·정당성)을 그대로 유지하기 어렵다는 점입니다. RQGM은 검색을 epoch로 나누고, 각 epoch에서는 고정된 within-epoch evaluation criterion으로 성능을 판단한 뒤 epoch 경계에서 utility를 업데이트하여 보장이 목표 변화에 대해 ‘부분적으로’ 유지되게 만듭니다. 또한 verifiable 과제에서는 agent-as-a-judge 형태의 code-review 신호를 결합하고, paper reviewing에서는 AI와 인간 작업을 동일한 엄격도로 평가하도록 adversarial objective를 도입해 편향을 줄입니다.

- **Empirical Impact**: 실험은 verifiable 코딩 과제에서 RQGM이 이전 SOTA 대비 test pass rate를 개선했으며, agent-as-a-judge code-review 신호를 추가하는 동시에 토큰 사용을 1.35x~1.72x 줄였다고 보고합니다. 과학 논문 작성·리뷰, 올림피아드 수준 증명 작성/채점에서도 co-evolved writers는 agent-as-a-judge 패널에서 1.78x~1.86x 더 높은 acceptance rate를 보이고, co-evolved graders는 ground-truth 정확도가 9% 향상됩니다. 특히 paper reviewing에서는 강력한 baseline reviewer가 AI-generated 논문을 인간 대비 최대 1.91x 과도 승인하는데, RQGM의 adversarial 목표가 이 과승인을 교정하는 성과를 보였습니다.



### Agentic Analysis for Agentic Infrastructure: An LLM-Powered Pipeline for Comparative Governance of DAO and Corporate AI Protocols (https://arxiv.org/abs/2606.26203)
- **Prior Approaches**: 기존 연구는 DAO와 기업 거버넌스의 차이를 이론적 논의나 인터뷰 중심으로 다루는 경우가 많았고, 같은 도메인에서 거버넌스 형태를 텍스트 데이터로 직접 비교하는 실증은 부족했다. 오픈소스/표준 커뮤니티를 분석하는 방법론도 수동 코딩·고정 카테고리에 의존해 대규모 담론의 주제 발견이나 구조 분석을 함께 수행하기 어려웠다.

- **Core Contribution**: 이 논문은 LLM을 활용해 거버넌스 담론을 자동 코딩·주제 모델링·다층 네트워크로 연결하는 비교 파이프라인을 제안한다. ERC-8004(권한less, 온체인)와 Google A2A(기업 주도)를 동일한 기술 문제(에이전트 상호운용)라는 조건에서 대조해, 제도 설계가 참여자 구조와 논의 주제를 어떻게 바꾸는지 분해해 관찰한다. 또한 4,323개의 공개 참여 기록을 대상으로 분석해, “누가 규칙을 쥐는가”를 텍스트 기반으로 정량화하려는 시도를 강화한다.

- **Technical Challenges**: 핵심 난제는 대규모 공개 기록에서 논증 기능(Argument Type), 입장(Stance), 합의 신호(Consensus Signal)를 일관되게 라벨링하는 동시에, 주제 분리와 관계 구조를 동시에 신뢰성 있게 복원하는 데 있었다. 연구팀은 MiniMax-M2.5를 LLM 백본으로 사용해 LLM-assisted annotation을 수행하고, BERTopic과 Thematic-LM 두 축의 주제 발견을 교차 검증하며, co-participation SNA·discourse network analysis·socio-semantic bipartite network까지 3층 네트워크로 담론 구조를 분석한다.

- **Empirical Impact**: 결과는 “분권 제도”가 곧바로 참여 불평등을 해소하지는 못한다는 그림을 보여주며, ERC-8004와 A2A 모두 참여 격차와 커뮤니티 파편화 수준이 유사한 것으로 나타났다. 다만 ERC-8004에서는 담론 정렬(discourse alignment)이 더 촘촘해 권한less 거버넌스가 참여는 넓어도 주제 수렴을 더 강하게 만들 수 있음을 시사한다. 또한 주제 수준에서는 ERC-8004가 신뢰·보안 같은 구성적 핵심 논의에 집중하는 반면, A2A는 문서화·예시·실행/프로젝트 운영 등 엔지니어링 실행 영역이 더 두드러져 기술 표준화의 ‘무엇을’과 ‘어떻게’를 가르는 제도 효과를 실증한다.



### Simulating Eating Disorder Patients with LLMs: Evaluating Psychological Persona Stability in Multi-Turn Conversations (https://arxiv.org/abs/2606.26109)
- **Prior Approaches**: 기존 LLM 시뮬레이션은 대화의 자연스러움이나 진단 정확도, 치료 효과 같은 결과 지표 중심으로 평가되는 경우가 많았다. 또한 persona benchmark는 표면적 일관성이나 일반 성격 역할에 초점을 둬, 임상적으로 중요한 ‘심리 프로필의 안정성(tempo/세션 간·대화 내 유지)’을 충분히 검증하지 못했다. 최근 연구들도 persona drift를 보고했지만, 검증에 임상 설문과 다중 출처(자기보고·관찰자) 동시 측정이 체계적으로 결합되진 않았다.

- **Core Contribution**: 이 논문은 섭식장애 eating disorder(ED) persona를 대상으로, EDE-Q 같은 검증된 임상 설문을 ‘자기보고(시뮬레이션 대상)’와 ‘관찰자 평가(LLM-as-judge)’ 두 축에서 동시에 측정한다. 5개 공개 case vignette에 근거한 persona를 6개 LLM에서 생성·평가하고, 세션 간 안정성과 대화 내 안정성을 각각 실험으로 분리해 검증한다. 아울러 Fairburn의 전이(transdiagnostic) 인지행동 모델 관점에서 persona 설명의 풍부함(prompt richness)을 조절해, 어떤 정보 요소가 임상적 재현을 좌우하는지 분해한다.

- **Technical Challenges**: 핵심 과제는 ‘안정적으로 응답하긴 하지만 틀리게 응답하는’ 상황을 놓치지 않는 것이다. 연구진은 순환 오류를 막기 위해 EDE-Q 점수 자체는 프롬프트에 넣지 않고, 관찰자 편향을 줄이기 위해 서로 다른 제공자 3개 관찰자 모델을 사용해 점수를 집계하며, 실험을 50회/20회 반복해 분산을 분해한다. 그 결과 prompt를 대폭 단순화해도 정확도는 거의 개선되지 않고, 대화 맥락을 더 붙일수록 오히려 기준선 대비 과대평가가 강화되는 경향을 확인한다.

- **Empirical Impact**: 6개 LLM 전반에서 persona는 놀랄 만큼 안정적이지만(CV 2–4%, 많은 항목이 결정적으로 동일 점수) ground truth보다 항상 12–30% 범위(0–6 척도에서 약 0.7–1.8점) 과대평가한다. 특히 행동(식사 억제, restraint)은 케이스를 어느 정도 구분하지만, 인지-정서 영역인 body dissatisfaction/체중·형상 집착(Shape Concern·Weight Concern)은 케이스 강도와 무관하게 상한(ceiling)에 가깝게 고정된다. 그 결과 ‘missing middle’로 불리는 중간 강도의 임상 양상이 제대로 재현되지 않으며, 심각한 케이스는 그럴듯해 보일 수 있어도 훈련/연구에서 누적 오판을 만들 수 있다는 경고를 제시한다.




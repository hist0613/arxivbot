New uploads on arXiv(cs.CL)

### Less is More: Quality-Aware Training Data Selection for Scientific Summarization (https://arxiv.org/abs/2606.24828)
- **Prior Approaches**: 과거의 과학(의생명/생명과학) 장문 요약은 논문 본문에 대응하는 저자 초록(author-written abstract)을 gold reference로 그대로 써왔습니다. 하지만 초록은 누락, 수치/일치성 문제, 과도한 해석, spin 같은 이유로 본문과 어긋날 수 있어 지도 신호와 평가지표의 신뢰성이 흔들릴 수 있습니다. 또한 기존 PMC 기반 벤치마크(PubMed 등)는 규모는 작고, LLM용 장문 입력에 맞춘 구조 보존이 약해 현대 long-context 모델의 학습/평가에 제약이 있습니다.

- **Core Contribution**: 이 논문은 (1) PMC-Large라는 188만 편 규모의 장문 요약 데이터셋을 공개하고, (2) 저자 초록의 reference quality가 본문과의 근거정합성(source-grounded)에 따라 달라짐을 대규모로 분석합니다. 아울러 그 quality 신호를 fine-tuning용 학습 데이터 선별에 활용해, 무작위 샘플링 대비 더 좋은 성능을 얻는지 검증합니다. 핵심 메시지는 “초록 품질은 균일하다고 가정하면 안 되고, 품질 인지 데이터 선택이 학습 효율을 높일 수 있다”는 점입니다.

- **Technical Challenges**: 첫째, 장문 요약에서 본문-초록 정합성을 직접 측정해야 하는데, ROUGE/BERTScore처럼 참조와의 유사도만으로는 사실성/근거 지지 여부를 보장하지 못합니다. 이에 AlignScore, FineSurE, SummaC, G-Eval 같은 source-grounded 및 model-based 지표를 사용하되, 지표 간 상관이 부분적이어서 단일 지표만으로는 부족함을 보여줍니다. 둘째, LLM이 잘 다룰 수 있도록 PMC 텍스트를 섹션 계층과 구조 단서가 남는 Markdown 스타일로 직렬화하고, 그림/표/수식/코드 등은 placeholders로 정규화하며, 비서사 구간과 라이선스/중복 타깃 누출 케이스를 필터링해 학습 입력의 일관성을 확보합니다.

- **Empirical Impact**: 10,000개 샘플에서 다수 초록이 높은 점수를 받지만 하위 꼬리(low-quality tail)도 뚜렷하고, 지표 간 일치도는 강하지 않음을 확인했습니다. fine-tuning 실험에서는 품질 기반으로 상위 예제를 뽑아 학습하면, 같은 학습 크기에서 무작위 선택보다 AlignScore/G-Eval/FineSurE 등에서 더 좋거나 적어도 동급 성능을 보이며, 더 작은 정제 데이터로도 큰 무작위 데이터에 근접·상회하는 결과가 나옵니다. 또한 100,000 후보 풀에서는 AlignScore로 1차 필터링 후 다중 지표로 2차 선별하는 two-step 전략이 전반적 균형 성능을 가장 잘 이끌어, “더 많은 데이터”보다 “더 나은 데이터”가 효율적으로 작동할 수 있음을 시사합니다.



### L3Cube-MahaPOS: A Marathi Part-of-Speech Tagging Dataset and BERT Models (https://arxiv.org/abs/2606.24825)
- **Prior Approaches**: 기존 POS 태깅 연구는 규칙 기반에서 시작해 HMM/CRF 같은 시퀀스 모델, 이후 BiLSTM·transformer로 발전했지만, 대부분은 대규모 주석 데이터가 있는 언어를 전제로 한다. 인도권 언어에서도 일부 말뭉치가 축적돼 왔으나 마라티는 표준화된 태그셋과 충분한 규모의 주석 데이터가 부족해 재현 가능한 비교가 어려운 편이다. 또한 마라티의 형태론적 복잡성과 자유로운 어순, 대문자 단서 부재, 힌디어·영어 코드믹싱이 결합되며 표면 형태만으로 추정하기가 더 까다롭다.

- **Core Contribution**: 이 논문은 마라티를 위한 gold-standard POS 태깅 데이터셋 L3Cube-MahaPOS를 공개한다. 뉴스 텍스트에서 32,354개 문장을 수작업으로 주석했으며, Universal Dependencies v2에 정렬된 16개 태그 체계를 적용해 데이터 품질과 일관성을 강화했다. 아울러 정규화·토큰화·잡음 필터링·분쟁 라벨 해소를 포함한 재현 가능한 전처리/주석 파이프라인과 함께 벤치마킹을 제공한다.

- **Technical Challenges**: 마라티는 형태 굴절이 풍부해 표면 단어가 여러 POS 후보를 동시에 만족할 수 있고, 대문자화 규칙이 없어 고유명사( propn ) 판별에 문맥 의존도가 커진다. 논문은 Unicode NFC 정규화, 데바나가리 aware 토큰화, 결합 조사/클리틱 예외 처리, 숫자·URL 등 비언어적 토큰의 마스킹/제외 같은 절차로 라벨 일관성을 확보한다. 모델 비교는 HMM·CRF·BiLSTM·BiLSTM+CharCNN부터 MuRIL과 마라티 특화 transformer인 MahaBERT-v2까지 6개 계열을 포함하며, 서브워드-토큰 정렬은 first subword 전략을 사용했다.

- **Empirical Impact**: 벤치마크 결과 MahaBERT-v2 기반 L3Cube-MahaPOS-BERT가 토큰 정확도 88.67%, macro-F1 81.67%(평가 15태그, postp 제외)를 달성해 최상 성능을 보였다. 혼동 분석에서는 adj–adv 경계, propn의 낮은 재현율(대문자 단서 부재 영향), 희소 클래스인 intj의 recall 저하가 핵심 오차 요인으로 나타났다. 데이터셋(주석 가이드라인·체크포인트 포함) 공개로 마라티 NLP 연구의 공통 기반을 마련하고, 후속 연구가 데이터 확장과 형태론 정보 통합 같은 방향으로 나아갈 수 있는 실증적 기준선을 제공한다.



### SHERLOC: Structured Diagnostic Localization for Code Repair Agents (https://arxiv.org/abs/2606.24820)
- **Prior Approaches**: LLM 에이전트는 리포지토리 수준 코딩 과제를 푸는 데서 멀티턴 도구 사용을 하지만, 패치 작성 전 “결함 찾기”에 상호작용 예산의 약 절반을 소모하는 경향이 있습니다. 기존 로컬리제이션은 대체로 파일·함수 위치를 검색/랭킹하는 형태에 머물러, 왜 그 위치가 원인인지에 대한 진단 맥락이 부족한 경우가 많습니다. 일부 방법은 설명을 추가하지만, 여전히 진단 정보를 수리 에이전트가 바로 쓰기 쉬운 형태로 구조화해 전이시키는 데 한계가 있습니다.

- **Core Contribution**: SHERLOC은 학습 없이도 단일 reasoning LLM과 컴팩트한 리포지토리 도구 세트, self-recovery를 결합해 “위치 + 구조화된 진단 finding”을 함께 산출하는 로컬리제이션 프레임워크입니다. 각 예측 위치마다 root cause, solution idea, 의존성, 테스트 영향 등 55개 필드의 진단 정보를 생성해 이후 repair 에이전트가 바로 후속 추론을 이어갈 수 있게 합니다. 또한 기능 단위가 아닌 (start, end) 라인 구간 기준의 chunk-level 지표를 제안해 패치 타깃의 구조 불일치 문제를 완화합니다.

- **Technical Challenges**: 핵심 과제는 (1) 파일 경로만이 아니라 원인 가설과 근거를 끝까지 모아야 하며, (2) 멀티턴 도구 호출 중 컨텍스트 손실·루프·깨진 호출 같은 실패를 제어해야 한다는 점입니다. SHERLOC은 deterministic executor 경계를 두고, View File/Codebase Search/Repository Tree/Connected Tree의 제한된 도구만 사용하며, 단계 예산 내에서 evidence가 충분해지면 final findings로 수렴하도록 설계했습니다. 동시에 컨텍스트 truncation, 루프 감지, malformed tool-call 복구, final-turn 합성 프롬프트 등 self-recovery를 넣어 긴 추론-탐색 과정의 안정성을 높였습니다.

- **Empirical Impact**: SWE-Bench Lite에서 SHERLOC은 84.33% accuracy@1, SWE-Bench Verified에서 81.27% recall@1로 파일 수준 로컬리제이션 SOTA를 달성했으며 약 30B 스케일에서도 에이전트형 방법과 견줄 만한 성능을 보였습니다. 더 나아가 SHERLOC의 locations과 진단 findings를 OpenHands와 SWE-Agent 같은 2개 repair 에이전트에 주입했을 때, 평균 resolve rate가 +5.95%p 상승하면서도 localization 및 총 토큰 비용은 각각 36.7%와 23.1% 절감됐습니다. 판단기준(LLM judge)으로 품질이 낮은 finding을 걸러내는 품질 매개(quality mediation) 전략이 특히 강한 로컬라이저/에이전트에서 효과를 보였고, 전이는 “긍정적이되 품질에 의존”한다는 결론을 뒷받침했습니다.



### Paying to Know: Micro-Transaction Markets for Verified Product Information in Agentic E-Commerc (https://arxiv.org/abs/2606.24783)
Comments:
          8 pages, 1 figure. Vision paper, under review

- **Prior Approaches**: 기존 상용 NLP는 쇼핑 챗봇을 추천(recommender)이나 전환(conversion) 도구로 보고, 사용자를 카탈로그 항목에 매칭한 뒤 구매를 설득하는 데 초점을 둔다. 이 프레이밍은 ‘어려운 문제=매칭’이라는 가정을 깔고 있으며, 검증 가능한 정보의 부족과 시장의 순위 인센티브 불일치를 상대적으로 사소하게 취급한다. 또한 제품 관련 데이터가 SEO 카피나 미검증 리뷰처럼 싼 신호로 요약되거나, 혹은 사일로에 갇혀 에이전트가 신뢰도 있게 탐색하기 어렵다는 한계가 남는다.

- **Core Contribution**: 논문은 agent-native micro-payment rails(예: x402, AP2)가 도입되면 희소성이 ‘상품 매칭’에서 ‘의사결정에 필요한 신뢰할 수 있는 정보’로 이동한다고 주장한다. 이에 에이전트가 마이크로 결제로 판매자·검증자·리뷰어가 공개하기로 한 증거를 단계적으로 잠금 해제하는, verified information 중심의 micro-transaction 정보 시장을 제안한다. 그 결과 랭킹 기반 스토어보다 더 진짜 품질과 정직한 경쟁을 유도할 수 있다고 본다.

- **Technical Challenges**: 핵심 난제는 (1) 예산 제약 하에서 다음에 어떤 ‘비용 있는 증거’를 살지 고르는 cost-optimal information acquisition, (2) 단일 데이터 가격과 증거 번들의 협상, (3) 서로 다른 출처의 정보를 동일 실체/속성으로 즉시 매칭하는 real-time entity resolution과 온톨로지 접지, (4) 가격된 주장 전부에 대한 환각 없는 grounded value exchange, (5) 사용자 페르소나를 개인정보 유출 없이 모델링하고 필요 시 주체가 통제 가능하게 유지하는 privacy-preserving persona modelling이다. 이들은 채팅 유창성보다 ‘대화→가격·접지·협상 가능한 증거 파이프라인’을 만드는 문제라고 못 박는다. 더 나아가 값비싼 툴 호출을 무턱대고 하는 대신, 불확실성이 정말로 남아 있을 때만 도구/질의를 쓰도록 비용-가치 캘리브레이션이 필요하다고 강조한다.

- **Empirical Impact**: 본 문서는 향후 비전(포지션 페이퍼)으로, 시장 효율성과 경쟁 개선을 실험으로 측정해 보여주진 않는다. 대신 산업 트랙이 ‘랭킹/대화 품질’ 대신 증거 획득의 복지(welfare)·효율성과 캘리브레이션을 측정하도록 벤치마크와 시뮬레이터를 확장해야 한다는 연구 방향을 제안한다. 의사결정 관련 신뢰 정보가 유료로 거래되는 구조가 마련되면, 에이전트가 구매를 조사·협상·정산까지 end-to-end로 자동화하는 에이전틱 조달로 이어질 수 있다는 점에서 의미가 크다.



### Are We Ready For An Agent-Native Memory System? (https://arxiv.org/abs/2606.24775)
Comments:
          Paper list available at: this https URL. Source code available at: this https URL

- **Prior Approaches**: 기존 LLM 에이전트 memory 평가는 end-to-end 과제 성공지표(F1, BLEU 등)에 치우쳐 메모리 시스템을 블랙박스로 다뤘다. 그 결과 비용(인덱스/지연), 모듈 간 설계 트레이드오프, 동적 지식 업데이트 시 견고성 같은 시스템 관점 평가는 충분히 쌓이지 않았다.

- **Core Contribution**: 본 논문은 agent memory를 데이터 관리 관점에서 분해해, 메모리 표현·저장, 추출, retrieval·routing, maintenance의 4개 핵심 모듈로 체계화했다. 이 프레임워크로 서로 다른 12개 메모리 시스템을 동일한 워크로드에서 비교하며, 효과가 특정 아키텍처의 우열이 아니라 워크로드 병목에 대한 정렬도에 달린다는 결론을 제시한다.

- **Technical Challenges**: 4개 모듈을 공정하게 분리해 실험하려면, 표현 충실도·검색 정밀도·업데이트 정확성·장기 안정성을 모듈 단위로 측정하고 단일 모듈만 바꾼 통제 실험을 설계해야 한다. 저자들은 11개 데이터셋을 포함한 5개 벤치마크 워크로드에서 모듈별 변형(ablations)과 fine-grained 분석을 수행해 각 레이어의 손실(압축/요약/추출이 정보 폐기를 유발)을 정량화했다.

- **Empirical Impact**: 대규모 end-to-end 평가에서 단일 구조가 모든 시나리오를 지배하지 못했으며, 특히 그래프 기반은 단발성 factual recall에는 강하지만 시간 추론에는 취약한 양상이 드러났다. 또한 운영비용 측면에서는 전역 재구성보다 localized maintenance가 더 비용 효율적이며, 지식 업데이트/장기 질의에서의 stale 정보와 시간순 단서 소실 같은 실패 모드가 구체적으로 관측돼 agent-native memory 설계 방향을 제안한다.



### Posterior Refinement: Fast Language Generation via Any-Order Flow Maps (https://arxiv.org/abs/2606.24773)
Comments:
          24 pages, 23 figures

- **Prior Approaches**: Non-autoregressive generation은 반복적으로 토큰을 비판하고(critique) 일부를 지우고(erase) 다시 생성(regenerate)하는 방식으로 품질을 끌어올릴 수 있지만, 기존 모델들은 이 잠재력을 충분히 구현하지 못했습니다. Masked Diffusion Models(MDMs)는 다중 토큰을 동시에 생성할 때 factorization error로 인해 샘플 품질이 급격히 무너지는 문제가 있고, Flow Map Language Models(FMLMs)는 joint sequence transport로 소수 단계 생성 성능은 좋지만 MDM이 제공하는 추론 시 유연성은 포기합니다.

- **Core Contribution**: 이 논문은 두 계열의 공백을 메우기 위해 FMLM에 masking-style noise schedules를 결합한 FMLM+ 프레임워크를 제안합니다. FMLM+는 전체 시퀀스를 한 번에 생성하면서도 각 토큰의 global consistency를 a posteriori로 동시에 점수화하고, 이를 바탕으로 Posterior Refinement이라는 추론 시 정제 전략을 도입합니다.

- **Technical Challenges**: 핵심 기술 난관은 FMLM의 빠른 few-step 생성 장점을 유지한 채, MDM처럼 토큰 수준 마스킹/정제 유연성을 제공하면서도 품질 붕괴를 막는 것입니다. 저자들은 masking-style noise schedules로 잡음 주입과 디노이징을 정렬하고, a posteriori 일관성 점수를 이용해 모델이 자신의 출력을 적응적으로 self-correct하도록 Posterior Refinement을 설계함으로써 안정적인 정제를 달성합니다.

- **Empirical Impact**: 여러 벤치마크에서 FMLM+와 Posterior Refinement은 MDM 계열과 FMLM 계열 모두와 비교해 속도-품질 tradeoff를 개선함을 보였습니다. 특히 discrete baselines 대비 32x fewer NFEs로 성능을 맞추는 결과를 제시하며, 고품질 언어 모델링을 위한 확장 가능한 기반을 제공한다는 점에서 의미가 큽니다.



### CANDLE: Character-level Arabic Noise Deduplication using Lightweight Encoder (https://arxiv.org/abs/2606.24758)
- **Prior Approaches**: 반복 문자 처리는 오타(정상 철자)와 소셜미디어의 비공식적 문자 늘이기(예: elation)를 구분해야 해 까다롭다. 기존 접근은 규칙 기반, 사전 기반, 형태소 분석기에 의존하는 경우가 많아 자원이 부족하거나 도메인이 바뀌면 성능이 흔들릴 수 있다. 또한 분류(classification) 중심으로 접근하면 문자 반복의 정렬/경로 선택을 충분히 모델링하기 어렵다.

- **Core Contribution**: 이 논문은 아랍어 문자 수준 중복 제거(Arabic noise deduplication)를 위한 가볍고 규칙 없이 동작하는 시스템 CANDLE을 제안한다. 핵심은 문자 중복 정규화를 Connectionist Temporal Classification(CTC)으로 재정의해, 문자 기반 인코더 위에서 정렬(sequence alignment) 문제로 푼 점이다. CTC를 문자 deduplication에 적용한 새로운 공식화로, 잘못된 반복을 “정렬” 관점에서 처리한다.

- **Technical Challenges**: CTC로 반복 문자를 정규화하려면, 어떤 반복이 정답 철자이고 어떤 반복이 늘이기인지에 대한 경로 정렬을 안정적으로 학습해야 한다. 논문은 문자 기반 인코더와 CTC 정렬 학습을 통해 규칙/사전/형태소 분석기 없이도 normalization을 수행하도록 설계했다. 더 나아가 추론 지연을 줄이기 위해 6층 CTC를 2층 student로 증류해 3배 깊이 감소를 달성하면서 성능 저하를 최소화했다.

- **Empirical Impact**: 3개 벤치마크(신문의 정제 텍스트, 수작업 애매 사례, 실세계 소셜미디어)에서 CTC 기반 모델은 Sentence Error Rate(SER) 5.37%까지 낮추며 분류 기반 기준선을 크게 능가한다. 또한 정규화는 tokenizer fertility를 최대 12.8% 줄여, 아랍어 LLM에서 추론 비용을 낮추고 컨텍스트 윈도우 활용을 개선하는 실용적 효과가 확인됐다. 코드와 모델을 공개해 재현성과 후속 연구 확산에도 기여한다.



### Task Decomposition for Efficient Annotation (https://arxiv.org/abs/2606.24734)
- **Prior Approaches**: 기존에는 한 명의 주석자가 예시 전체를 end-to-end로 주석하는 방식이 일반적이었지만, 구조화된 표현 주석은 복잡해서 인지·추론 부담이 커집니다. 모델 기반 주석은 생성 비용을 줄일 수는 있으나, 검증 비용이 들고 다운스트림에 쓸 만큼 품질을 확보하려면 추가 감독이 필요하다는 한계가 있습니다. 또한 최근의 혼합 인력(모델+인간) 환경에서는 어떤 주석 하위 과제를 어떤 역량의 주석자에게 배분해야 하는지 불명확합니다.

- **Core Contribution**: 이 논문은 구조화된 주석 작업을 여러 하위 태스크로 분해해, 전체 주석 프로젝트의 누적 inferential load를 줄이는 설계를 제안합니다. centering theory의 center 개념에서 영감을 받아, ‘유효한 주석 공간에서의 자유도’로 inferential load를 정식화하고, center(하위 태스크로 드러나는 핵심 앵커 엔티티) 식별이 출력 공간 복잡도를 제약함을 보입니다. 이를 바탕으로 center 식별을 분리·전진시키는 분해가 누적 부담을 낮춘다는 가이드라인을 제공합니다.

- **Technical Challenges**: 핵심 기술적 어려움은 복잡한 구조 주석에서 어떤 하위 태스크 조합이 center 식별을 강화해 출력 공간을 얼마나 줄이는지 정량화하는 것입니다. 논문은 유효 주석의 공간에서 자유도를 기준으로 inferential load 모델을 만들고, center 식별이 완료되면 남은 선택 공간이 축소된다는 관찰을 이 모델에 연결합니다. 또한 고정 예산 하에서 주석자(모델/인간)에게 하위 태스크를 배분해 품질을 최대화하는 할당 절차를 제시합니다.

- **Empirical Impact**: 이 연구는 이전 work의 사례들을 통해 복잡한 구조 주석을 분해했을 때 비용 효율이 개선됨을 보여줍니다. 더 나아가 제안된 할당 절차로 제한된 예산 안에서 품질을 극대화하는 방향의 운영 전략을 제시해, 혼합 주석 환경에서의 실용성을 강화합니다. 구조 주석의 ‘무엇을 먼저 정해 출력 공간을 줄일지’에 대한 체계적 접근이 가능해지면서, 대규모 코퍼스 주석 파이프라인의 설계 관점을 확장하는 데 의미가 있습니다.



### CN-NewsTTS Bench: a target-level automatic benchmark for raw-input Chinese news TTS pronunciation (https://arxiv.org/abs/2606.24714)
Comments:
          5 pages, 1 figure, 8 tables. ICASSP-style preprint

- **Prior Approaches**: 기존 TTS 평가는 MOS 같은 주관 품질평가나, 음질/자연스러움을 대략적으로 추정하는 객관식 지표에 치우치는 경우가 많다. 또한 ASR 라운드트립 같은 범용 비교는 ‘특정 뉴스 표기 문자열을 의도한 방식으로 읽었는지’라는 표적 읽기 판단을 세밀하게 추적하기 어렵다.

- **Core Contribution**: CN-NewsTTS Bench v0.1은 중국 뉴스 TTS가 원문(raw text)에서 특정 표적(target) 문자열을 뉴스 관례대로 읽는지 평가하는 공개 벤치마크다. 사용자 규칙, LLM rewriting, SSML 힌트, 수동 편집 없이 API 노출 형태의 정확도를 재현 가능하게 측정하도록 dev/public split, 타깃 스키마, 고정 전사(three-ASR)와 자동 채점기를 제공한다.

- **Technical Challenges**: 핵심 난제는 스포츠 점수/범위 하이픈, 군사·차량 모델 표기, 단위 기호, 생성 라벨 등 ‘표면은 같지만 의미가 달라지는’ 타깃을 자동으로 판정하는 것이다. 논문은 MiMo API ASR와 두 개의 로컬 recognizer로 구성한 3개 경로를 앙상블해 strict accuracy를 계산하고, 해석(positive/negative reading) 패턴 매칭 후 majority voting으로 correct/wrong/unknown을 분리함으로써 커버리지 편향도 함께 통제한다.

- **Empirical Impact**: 7개 제품 TTS를 평가한 결과, 최고 성능은 0.879 strict accuracy에 도달했지만 여러 널리 쓰이는 시스템은 0.60 미만에 머물렀다. 특히 스포츠 점수는 하이픈을 범위로 오독하거나 뺄셈으로 읽는 등 실패가 잦고, 단위 기호는 ASR이 기호-수준 판독을 안정적으로 드러내기 어려워 unknown 비율이 높았다. 벤치마크는 strict accuracy뿐 아니라 coverage·resolved accuracy·카테고리 진단을 함께 보고해, 시스템이 어디서 어떤 ‘뉴스 관례 읽기’에 취약한지 프로덕션 관점에서 드러내는 데 의미가 있다.



### DREAM: Dense Retrieval Embeddings via Autoregressive Modeling (https://arxiv.org/abs/2606.24667)
- **Prior Approaches**: 대부분의 dense retriever는 contrastive objective로 학습하며, 쿼리-양성/음성 문서 페어(특히 hard negative)를 구성하는 비용과 불확실성이 병목입니다. 또한 next-token prediction을 활용한 접근도 있었지만(RePlug, Revela) retriever의 점수가 LLM이 실제로 예측에 사용하는 흐름으로 직접 연결되지 않거나, 쿼리-타깃 조건이 명확하지 않아 학습 신호가 간접적이었습니다. 그 결과 ‘후보 문서가 쿼리별 타깃 출력 예측에 도움이 되는지’를 더 정밀하게 반영하는 학습이 제한적이었습니다.

- **Core Contribution**: 이 논문은 DREAM(Dense Retrieval Embeddings via Autoregressive Modeling)을 제안해, frozen LLM의 autoregressive next-token prediction loss로 dense retriever를 학습시키는 방법을 제시합니다. 핵심은 retriever가 계산한 query-document similarity 점수를 LLM 내부의 선택된 attention head에 주입해, LLM이 문서를 ‘얼마나 읽는지’가 학습 중에 점수에 의해 바뀌도록 만든 것입니다. LLM은 학습 내내 고정하고, 그로부터 전달되는 예측 손실 그래디언트만으로 retriever를 업데이트해 추론 시 standalone embedding model로 그대로 사용 가능합니다.

- **Technical Challenges**: 기술적 난관은 next-token prediction loss는 LLM 내부에서 계산되는데 retriever는 별도 embedding 모델이라, score가 LLM 계산에 영향을 주지 않으면 retriever에 대한 그라디언트가 거의 전달되지 않는 점입니다. DREAM은 문서별 정규화된 score 분포를 선택한 query-focused retrieval heads의 attention에 끼워 넣고(competition 기반으로 문서 간 선택이 상호 제약되게 함), 그 attention 변경을 통해 타깃 통과 예측의 교차 엔트로피 손실이 retriever 파라미터로 역전파되게 설계했습니다. 또한 retrieval에 유효한 head만 선택해(LLM이 원래 쿼리 증거를 모으는 head) 불필요한 attention perturbation이 신호를 흐리지 않도록 했습니다.

- **Empirical Impact**: BEIR과 RTEB에서 0.5B~3B 임베딩 백본 규모 전반에 걸쳐 DREAM이 기존 baselines(RePlug, Revela 등)보다 일관되게 성능이 높았고, NDCG@10에서 BEIR은 0.015~0.081, RTEB는 0.068~0.102 수준의 개선이 보고됩니다. ablation과 분석에서는 무작위 head 주입이 약한 반면, query-focused head에 주입할 때 신호가 크게 강해지고 후보 문서 수를 늘릴수록 이득이 커지는 경향이 확인됐습니다. 또한 contrastive 없이도 uniformity/정렬 성격의 임베딩 기하가 개선되는 결과가 제시돼, autoregressive next-token prediction이 dense retrieval 학습의 실용적 대안이 될 수 있음을 시사합니다.



### AI-PAVE-Br: Leveraging Large Language Models for Enhanced Product Attribute Value Extraction through a Golden Set Approach (https://arxiv.org/abs/2606.24655)
- **Prior Approaches**: 기존 PAVE(Product Attribute Value Extraction)는 규칙 기반·정규식·사전 중심 접근이나 CRF/SVM 같은 전통 NER에 의존하는 경우가 많았다. 이런 방법은 데이터와 언어 변형에 취약하고, 도메인별 특징 공학과 라벨 데이터가 많이 필요해 규모가 커질수록 유지보수가 어려웠다. 또한 최근 LLM 활용 연구도 영어 중심이어서 포르투갈어(브라질) 카탈로그의 용어·표기 관습을 그대로 반영하기 어렵다는 한계가 있었다.

- **Core Contribution**: 본 논문은 브라질 e-commerce에 특화된 PAVE 시스템 AI-PAVE-Br을 제안한다. LLM을 대상으로 카탈로그 추출 목표에 맞춘 prompt engineering을 적용해, 상품 제목/설명/기술 데이터에서 Entity-Category-Subcategory를 구조화된 출력으로 뽑아내도록 설계했다.
아울러 재현 가능한 평가를 위해 포르투갈어 PAVE의 Golden Set(수작업 라벨, 20개 상품 유형, Entity·Category·Subcategories 구조)를 공개한다.

- **Technical Challenges**: 핵심 난제는 포르투갈어 상품 텍스트의 언어적 뉘앙스 다양성과 속성-값의 관계 추론, 그리고 값 표기의 비정규화 문제였다. 같은 모델명이 여러 표기로 등장하고(예: WD11 계열), 치수·전압·색상/용량 등이 의미는 같아도 문장 표현이 달라 정규화가 어렵다.
논문은 fine-tuning 대신 task-specific 프롬프트로 LLM이 값뿐 아니라 해당 속성(예: screen size-10-inch)을 함께 매핑하도록 유도했으며, 출력 형식을 JSON 스키마로 강제해 일관성 있는 파싱 가능성을 높였다.

- **Empirical Impact**: Golden Set을 기준으로 AI-PAVE-Br은 전통 NER 기반 baseline 대비 평균 F1-score를 59.79에서 74.68로 끌어올렸다. 대부분의 상품 유형에서 큰 개선을 보였고, 일부 카테고리에서는 입력 텍스트의 구조적 변동성과 속성 표현 복잡도 때문에 성능 하락도 관찰됐다.
또한 coverage(빈 응답 없이 예측 생성)는 46.71%에서 71.96%로 상승해 카탈로그 전반에서 더 안정적으로 추출을 수행함을 보였다. 브라질 비영어 시장에 대한 고정밀 PAVE 실무 해법과 공개 벤치마크를 함께 제공한다는 점에서, 향후 연구의 기준선과 데이터 인프라 역할을 할 것으로 기대된다.



### Harmonic: Hierarchical State Space Models for Efficient Long-Context Language Modeling (https://arxiv.org/abs/2606.24650)
Comments:
          12 pages, 8 figures. NeurIPS 2024 format

- **Prior Approaches**: 기존 장문 언어모델링 연구는 Transformer의 self-attention(O(L^2))을 줄이려는 SSM 계열(S4, Mamba, RWKV 등)과, 장단기 처리를 위한 RNN의 계층적 구조(Clockwork RNN 등)로 발전해 왔다. 다만 많은 비교가 토큰 예산 불공정, 튜닝 차이, 시퀀스 길이에서의 메모리/학습 한계로 인해 아키텍처 자체의 효과를 분리해 보기 어려웠다. 또한 SSM은 계산은 O(L)로 유리해도 장문에서 품질 이점이 일관되게 입증되지 못했다.

- **Core Contribution**: 이 논문은 Harmonic이라는 3단 계층형 SSM을 제안한다. 각 단계는 하위 단계의 raw hidden state가 아니라 prediction error를 입력으로 받아, 서로 다른 timescale(빠름-느림)을 가진 재귀가 오류 기반으로 조합되도록 설계했다. 그 결과 장문 컨텍스트에서 Transformer와 Mamba를 상대로 품질 격차가 길이에 따라 더 커지는 패턴을 보고한다. 추가로, HarmonicBlock을 TinyLlama 1.1B에 주입해 RoPE 기반 positional encoding 한계를 제거하는 Hallamonic 모델도 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 계층형 재귀를 펼쳤을 때 발생하는 깊은 계산 그래프의 gradient 안정성, (2) 여러 timescale에서 오류 신호가 상위 레벨로 잘 전달되게 만드는 구조 설계, (3) O(L) 재귀를 실제 GPU에서 효율적으로 학습·스캔하는 구현 최적화였다. 논문은 오류 신호를 정규화해 계층 전반의 gradient 폭주/소실을 막고, 단일 language modeling loss로 end-to-end 학습되도록 했으며, 병렬 scan을 활용해 학습 시 시퀀스 축 병렬화를 유지했다. 구현 측면에서는 Transformer는 FlashAttention을, SSM은 커스텀 Triton 커널과 torch.compile을 사용해 공정 비교를 지키려 했다.

- **Empirical Impact**: 동일 토큰 예산(공정 equal-budget) 조건에서 enwiki8 기준 Harmonic은 Transformer(약 28M)보다 1K 토큰 +1.4%, 8K +6.7%, 32K +11.4%(bpt, 낮을수록 우수)로 장문일수록 격차가 확대됐다. 또한 모든 길이에서 Mamba보다 0.7–1.8% 우위를 보였고, 64K 토큰에서는 Transformer와 Mamba가 H100 80GB에서 OOM이지만 Harmonic은 학습을 완료하며 6.169 bpt를 기록했다. WikiText-103에서도 같은 방향의 이점이 재현되며, 1B 스케일 Hallamonic은 RoPE 2K 한계를 넘겨 TinyLlama가 급격히 악화하는 구간에서 손실이 안정적으로 유지되는 결과를 보였다.



### Measuring User's Mental Models of Speech Translation in Human-AI Collaboration (https://arxiv.org/abs/2606.24644)
Comments:
          ACL2026

- **Prior Approaches**: 기존 기계번역 연구는 주로 BLEU 등 자동 지표나 사용자 평가에 초점을 두었지만, 사용자가 시스템의 한계를 어떻게 ‘인지적으로’ 가늠하는지는 덜 밝혀져 있었다. 특히 음성번역처럼 출력 품질이 들쑥날쑥할 때 사용자의 판단 근거(오류 징후)가 어떻게 형성되는지에 대한 체계적 프레임워크가 부족했다.

- **Core Contribution**: 이 논문은 교차언어 질의응답(cross-lingual question answering) 관점에서 음성번역 시스템에 대한 사용자의 정신 모델(mental model)을 새로 측정한다. 사용자는 외국어로 제시된 정보를 바탕으로 질문에 답하고, 번역 결과를 수락하거나 전문 재번역을 요청해 정확도를 맞추는 방식으로 시스템의 오류 가능성을 간접적으로 드러낸다.

- **Technical Challenges**: 핵심 기술적 과제는 사용자가 ‘오류를 어디서’ 예측하는지 행동 데이터로부터 신뢰성 있게 분해해내는 것이었다. 연구진은 번역 품질을 다양하게 바꾸며 사용자 행동과 정확도 추세를 분석했고, 경험이 쌓일수록(특히 원천 언어를 조금 아는 경우) 주로 표면 수준의 오류 단서에 의존해 더 강한 정신 모델을 형성하는 패턴을 확인했다. 또한 음성 전사(transcription) 제공이 사용자 모델을 개선하는지 실험으로 검증했다.

- **Empirical Impact**: 실험 결과 사용자는 연습을 통해 시스템이 틀릴 가능성이 큰 상황을 더 잘 가늠해가며, 전사 제공은 예측 능력 향상에 도움이 됐다. 이는 MT 정신 모델 연구를 위한 downstream 과제로 교차언어 질의응답이 유망함을 보여주며, 사람-인공지능 협업에서 ‘언제 재검증이 필요한가’를 이해하는 데 의미가 크다.



### The Warrant Gap: Claim-Conditioned Re-scoring for Fact-Checking (https://arxiv.org/abs/2606.24627)
- **Prior Approaches**: 기존 LLM 기반 팩트체크는 벤치마크에서 높은 verdict 정확도를 보이지만, Supports 판정이 “근거가 주장(Claim)을 정당화하는지”를 제대로 확인하지 못해 warrant gap이 자주 발생한다. 구조화된 rationale(예: claim decomposition)는 근거 점검에 유리하지만, 5W1H 같은 강제 슬롯을 facet 단위로 독립 평가하면 full-claim 맥락이 사라져 정확도를 깎는 문제가 보고돼 왔다.

- **Core Contribution**: 이 논문은 SIFT(Structured Inference with Facet-level Tracing)로, 추출된 5W1H 근거 스팬을 다시 전체 Claim과 재스코어링해 분해로 인한 맥락 단절을 복구한다. 또한 warrant가 실제로 Claim을 entail하는지 자동으로 진단하는 WSP(Warranted Supports Proportion)를 제안해, 정확도와 admissibility를 함께 추적 가능하게 만든다.

- **Technical Challenges**: 핵심 난제는 facet 커버리지는 확보되지만(구조화의 장점) 주장 전체 관점의 정당화(라이선싱)가 붕괴되는 decontextualisation 문제를 어떻게 되돌릴지다. 저자들은 5W1H를 고정 프레임으로 추출하되, verdict를 만든 뒤 전체 Claim 기준 NLI 판정으로 verdict를 claim-conditioned 재수정하는 방식으로 해결했으며, 마지막에 Design B의 결정론적 verifier로 multi-document patchwork 같은 취약 조합을 추가로 억제한다.

- **Empirical Impact**: FEVER, SciFact, 5PILS, DP에서 SIFT는 naive 5W1H 대비 최대 27.6포인트 정확도 손실 구간을 되찾는 등, accuracy–admissibility 사이의 불균형을 완화했다. WSP는 human gold evidence 캘리브레이션에서 AUC 0.92, precision 0.98을 보였고, NLI scorers 계열을 바꿔도 순위 일관성이 유지(예: Spearman 0.89)돼 진단 지표로서의 신뢰성을 뒷받침한다. 남은 오차는 주로 verifier가 재구성할 수 없는 extraction-bound(정체성/술어 누락) 쪽으로 이동해, 다음 연구가 “판정 규칙”보다 “근거 구성”에 있음을 시사한다.



### Privacy-Preserving RAG via Multi-Agent Semantic Rewriting: Achieving Confidentiality Without Compromising Contextual Fidelity (https://arxiv.org/abs/2606.24623)
Comments:
          This full manuscript contains 23 pages and has been formally accepted for publication in Information Processing & Management (Elsevier IPM). Tao Fang is the corresponding author

- **Prior Approaches**: RAG는 외부 지식을 검색해 LLM의 정확도를 높이지만, 민감한 코퍼스에 적용될 경우 prompt injection 등 공격을 통해 검색 원문이 노출될 위험이 큽니다. 기존 방어는 (1) 데이터 소스에서 합성 텍스트로 대체하거나 (2) DP처럼 잡음을 주입하거나 (3) 단일/루프 기반 재작성으로 프라이버시와 유틸리티의 균형을 맞추려는 방식이었습니다. 특히 합성 기반은 의미의 구체성이 사라져 다운스트림 신뢰성이 흔들리고, 단일 에이전트 방식은 한 번에 두 목표를 동시에 처리하다 보니 과잉·과소 마스킹이 발생하기 쉽습니다.

- **Core Contribution**: 이 논문은 RAG 파이프라인의 검색-생성 사이에 끼워 넣는 비공개 데이터용 “semantic rewriting” 프레임워크를 제안합니다. Pri-Extra(프라이버시 민감 구간 추출)–Sem-Extra(유틸리티를 위한 구조화된 의미 백본 추출)–Reconstruction(구간 제거 후 의미 재구성) 세 에이전트가 협업해 식별자(명시적/잠재적)를 제거하면서도 핵심 의미를 보존합니다. 또한 Asymmetric Retrieval과 isolated generation을 결합해 LLM이 원문 프라이빗 식별자를 직접 보지 못하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 텍스트에서 프라이버시와 유틸리티가 분리되지 않고 함께 얽혀 있어 단일 패스로는 삭제-정확도 간 제로섬 균형을 맞추기 어렵다는 점입니다. 이를 위해 Pri-Extra는 rules 기반 NER/정규식으로 명시적 식별자를 먼저 잡고, LLM 기반 추출로 latent quasi-identifiers를 추가로 찾아 누락을 줄입니다. Sem-Extra는 문장 표면 대신 subject–predicate–value–중요도 형태의 슬롯 튜플로 분해해 스타일/표면 단서가 공격에 재사용될 가능성을 낮추고, Reconstruction은 fine-grained conflict routing으로 low-depth면 placeholder 치환, deep semantic conflict면 고수준 추상화로 과잉 마스킹을 제어합니다.

- **Empirical Impact**: ChatDoctor와 Wiki-PII(Enron PII를 Wikitext-103에 혼합)에서 6개 LLM 백본, targeted/untargeted 공격을 모두 평가했으며 프라이버시 누출이 유의미하게 감소했습니다. 예로 LLaMA-3-8B에서 targeted 정보 노출 인스턴스를 baseline의 144에서 1로 크게 줄였고, 문맥 충실도는 BLEU-1 0.122로 기존 SAGE(0.117)를 앞섰습니다. 또한 재작성은 온라인 추론 지연을 만들지 않는 asynchronous offline 전처리 모듈로 동작하며, 공격 상황에서도 semantic fidelity와 보안성을 동시에 유지하는 점에서 의미가 큽니다.



### Same Lesson, Different Story: Cross-Lingual Reconstruction of Cultural Narratives in Large Language Models (https://arxiv.org/abs/2606.24610)
Comments:
          This paper is under review

- **Prior Approaches**: 기존 문화 역량 평가는 사실성이나 표면적 품질에 치우치는 경우가 많아, 이야기의 주제 전개·인물 구조·스타일·응집성 같은 ‘서사적 의미 구성’은 상대적으로 덜 다뤄졌다. 또한 문화 고유 요소가 프롬프트에 명시돼도 언어 커버리지나 번역 차이 때문에 문화적 적합성이 깨질 수 있다는 결과들이 보고돼 왔다. 이런 흐름에서 더 나아가, 같은 도덕을 다른 서사 패턴으로 표현할 수 있음에도 ‘언어를 바꿨을 때 문화적 의미가 유지되는 수준’은 불명확했다.

- **Core Contribution**: 이 논문은 15개 언어의 414개 속담을 기반으로, 4종 LLM이 생성한 13k 규모 서사를 평가하는 다국어 평가 내러티브 프레임워크를 제안한다. 핵심은 ‘같은 도덕 교훈(의미가 대응되는 속담)’을 서로 다른 문화적 형식으로 제시했을 때, LLM이 속담 수준의 의미를 얼마나 보존하는지와 서사 실현이 어떻게 달라지는지를 분리해 본 것이다. 특히 의미 보존(semantic preservation)과 서사적 구현(narrative realization)을 나눠 관찰한 점이 기여로 제시된다.

- **Technical Challenges**: 관건은 번역된 속담이 유도하는 생성이 ‘같은 도덕’은 유지하되, 대리 행위·사회적 포지셔닝·서사 구조는 어떻게 재배치되는지 정량화하는 것이다. 이를 위해 (1) 생성문 임베딩의 코사인 유사도로 속담 수준 의미 변화(semantic shift)를 측정하고, (2) 인물·역할(agency/patient 등) 별 latent power score로 사회적 동학의 변화를 추적했으며, 소스 언어(영어/아랍어) 방향성도 따로 분석했다. 또한 프롬프트 템플릿 의존성을 줄이기 위해 free-form 생성에서도 유사한 경향이 유지되는지 교차 검증했다.

- **Empirical Impact**: 실험 결과, 크로스-랭귀지 프롬프팅은 전반적으로 속담 수준 의미는 크게 훼손하지 않으면서도, 행위자와 피행위자에 대한 agency·사회적 위치·서사 구조를 체계적으로 재분배하는 것으로 나타났다. 모델 4종은 monolingual과 cross-lingual 조건 모두에서 높은 상호 수렴(inter-model convergence)을 보였고, 언어·아키텍처 차이보다 공유된 의미 추상화에 더 의존하는 신호가 관찰됐다. 결론적으로 다국어 서사 평가에서 ‘semantic similarity만’ 보면 문화적 보존을 과대평가할 수 있으며, 문화적 접지(cultural grounding)는 의미뿐 아니라 서사와 사회적 구현 차원까지 함께 봐야 한다는 방향성을 강화한다.



### Qwen-AgentWorld: Language World Models for General Agents (https://arxiv.org/abs/2606.24597)
- **Prior Approaches**: 기존 LLM 에이전트 연구는 주로 policy(상태→행동) 쪽에 집중했고, (상태·행동→다음 상태) world model은 상대적으로 공백으로 남아 있었다. 또한 언어 환경을 다루는 world modeling은 장기 상호작용에서 관측·상태 일관성 유지와 오픈엔디드 피드백 평가가 어려워 성능/확장성에 한계가 있었다.

- **Core Contribution**: 논문은 언어 기반 world model(LWM)을 통해 에이전트 환경 시뮬레이션의 기반을 만들고, 이것이 general agent를 끌어올리는 두 경로(분리된 환경 시뮬레이터, 통합된 에이전트 파운데이션)를 제시한다. Qwen-AgentWorld-35B-A3B와 Qwen-AgentWorld-397B-A17B를 소개하며, 7개 도메인(MCP, Search, Terminal, Software Engineering, Android, Web, OS)을 long chain-of-thought 방식으로 시뮬레이션하는 최초의 언어 world model로 위치시킨다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 7개 서로 다른 관측/행동 표현을 공통 궤적 포맷으로 통합하고, (2) 다음 관측 예측을 장기 구간에서 안정적으로 학습하며, (3) 검증 가능한 형태가 아닌 환경 피드백을 보상으로 안정화하는 것이다. 이를 위해 CPT(비-추론 트랙으로 상태천이 지식 주입)→SFT(다음 상태 예측을 명시적 사고 패턴으로 활성화)→RL(5차원 LLM rubric + 실행 가능한 rule verifier를 9:1로 결합) 3단계 학습과, GUI는 accessibility tree/UI view hierarchy 등 텍스트 관측 표현을 사용한다.

- **Empirical Impact**: 평가를 위해 실제 상호작용 기반의 out-of-distribution 벤치마크 AgentWorldBench(9개 established benchmark 구성, 5개 frontier 모델 데이터 사용)를 제안하고, Qwen-AgentWorld가 기존 frontier 모델을 유의미하게 능가함을 실험으로 보인다. 더 나아가 decoupled environment simulator로서 수천~수만 수준(예: 4k4k OpenClaw) 환경을 시뮬레이션해 agentic RL 성능을 real-environment training만으로는 못 넘는 수준으로 끌어올리고, 통합 파운데이션 warm-up 관점에서도 7개 에이전트 벤치마크에서 downstream 성능 향상을 확인한다.



### To Compare, or Not to Compare: On Methodological Practices in Evaluating Social Bias (https://arxiv.org/abs/2606.24596)
- **Prior Approaches**: 기존 LLM 사회편향 평가는 다수의 벤치마크로 진행돼 왔지만, 어떤 상황에서 편향이 나타나는지 결론이 자주 엇갈립니다. 특히 isolated(독립) 질문과 comparative(비교) 강제선택, CoT(Chain-of-Thought) 사용 여부, ‘Prefer not to answer’ 같은 중립 옵션 처리 방식이 연구마다 달라 관측 차이를 깔끔히 원인 분해하기 어렵습니다. 또한 중립 회피율이나 무작위 응답을 편향 부재의 대리 지표로 삼는 시도는, 실제로는 방어적 정렬이나 숨은 불균형을 가릴 수 있다는 한계가 지적됩니다.

- **Core Contribution**: 논문은 서로 다른 9개 벤치마크를 하나의 unified·controllable 프레임워크로 표준화해, benchmark 수준에서의 구조적 framing 차이(특히 isolated vs comparative)를 체계적으로 분리합니다. 이를 통해 demographic 그룹을 독립 평가하는 Isolated Paradigm(iso)과 두 그룹/선택지를 동시에 비교하게 만드는 Comparative Paradigm(cmp)을 명확히 구분하고, 두 설정의 차이가 편향 측정에 미치는 효과를 실험적으로 대비합니다. 더 나아가 CoT, 중립 fallback, ‘무작위’라고 주장하는 메커니즘이 편향 추정에 끼치는 혼동 요인을 함께 분리해 해석 가능성을 높입니다.

- **Technical Challenges**: 핵심 기술적 도전은 벤치마크마다 다른 설계 변수를 동시에 바꾸지 않으면서, 질문 framing·프롬프트 변형·추론 모드·중립 선택지·편향 지표를 통제된 방식으로 비교하는 것입니다. 저자들은 각 벤치마크를 iso/cmp에 맞게 재구성하고, 옵션 순서에 따른 positional bias를 줄이기 위해 두 순열로 강제비교를 수행했으며, 편향 강도는 Parity Gap(PG)로 정량화합니다. 실험적으로는 CoT vs non-CoT, 중립 abstention 옵션의 유무, 랜덤 응답을 주장하는 프롬프트까지 조절해 ‘편향 완화’로 보일 수 있는 착시가 실제로는 얼마나 유지되는지 확인합니다.

- **Empirical Impact**: 19개 모델(핵심 10개 포함) 전반에서 iso는 거의 일관되게 편향이 약하게 나타나지만, 동일 모델이 cmp 강제비교에서는 심각한 고정적 차별/고정 관념 선호가 드러납니다. 특히 컨텍스트가 사실적으로 답을 도출할 만큼 충분히 주어지지 않은 ambiguous 상황에서 iso-cmp 격차가 크게 발생하며, CoT는 iso에서는 효과가 미미한 반면 cmp에서는 사회적 편향을 증폭하고 프롬프트 변형에도 선호를 더 안정화합니다. ‘Prefer not to answer’나 ‘무작위로 답한다’고 주장해도 선택 분포는 균일해지지 않아, 중립 회피율이나 무작위 주장만으로 안전성을 판단하면 오판할 수 있음을 보여줍니다. 나아가 모델 크기가 커질수록 cmp에서의 comparative prejudice 크기가 양(+)의 상관으로 증가해, 벤치마크 설계/현장 배치 모두에서 comparative 배치의 위험성과 guardrail 필요성을 강하게 시사합니다.



### MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery (https://arxiv.org/abs/2606.24595)
- **Prior Approaches**: 기존의 장기 메모리(long-term memory) 평가는 주로 이후 답변, 개인화 품질, 과제 성공 여부 같은 다운스트림 행동으로 이뤄져 왔다. 이 방식은 사용자 상태를 실제로 얼마나 정확히 “기억”했는지 메모리 자체를 직접 검증하지 못한다는 한계가 있다.

- **Core Contribution**: 논문은 장기 메모리를 ‘감사 가능한 post-interaction artifact’로 평가해야 한다고 주장한다. 즉, 에이전트가 남긴 메모리로부터 상호작용 이후 재구성 가능한 구조화된 사용자 상태(user state)를 복원해 성능을 직접 측정한다.

- **Technical Challenges**: 핵심은 메모리에 저장된 정보가 누출(leak-controlled)과 접근 제한(예: full-store vs top-k) 환경에서 어떤 사용자 상태로 복원되는지, 그리고 이를 정량화할 기준선을 어떻게 만들지였다. 논문은 MEMPROBE 벤치마크를 통해 시뮬레이션 사용자에게 숨겨진 taxonomy-anchored 상태 은행을 부여하고, 합성 정답(synthetic ground truth)으로 메모리 복원 정확도를 대규모로 측정하며 5가지 memory systems를 비교한다.

- **Empirical Impact**: 실험 결과, 과제 성공(성공적으로 도움 제공)과 복원 가능한 메모리(recoverable memory)는 별개의 역량으로 나타났다. 작업 완료율은 메모리 없는 기준선에서도 거의 포화되는 반면, 카테고리 균형을 맞춘 복원 성능은 약 0.6 수준에서 머물고 top-k 접근에서는 더 하락했다. MEMPROBE는 메모리 복원을 직접 목표로 하는 최초의 벤치마크로, 향후 에이전트가 “기억”을 최적화하도록 학습·평가 환경을 바꾸는 데 의미가 있다.



### Cross-Lingual Exploration for Parametric Knowledg (https://arxiv.org/abs/2606.24579)
Comments:
          29 pages, 5 figures, preprint

- **Prior Approaches**: 대규모 언어모델의 매개변수 지식은 언어별로 균일하게 접근되지 않아, 일반적인 추론 방식만으로는 현지화된 사실을 끌어내기 어렵다. 그 결과 교차언어 지식 전이와 일관성이 함께 흔들리며, 특히 다른 언어로 질문할 때 오류가 증가하는 경향이 보고된다.

- **Core Contribution**: 본 논문은 숨겨진 사실 지식을 인퍼런스 단계에서 끌어내기 위해 cross-lingual prompting 전략을 탐구한다. 교차언어 탐색을 좌우하는 네 가지 고유한 차원을 정리하고, 이 차원이 parametric knowledge retrieval에 직접 영향을 준다는 점을 체계적으로 평가한다.

- **Technical Challenges**: 핵심 문제는 어떤 프롬프트 변형이 모델 내부의 언어별 지식을 효과적으로 “탐색”하는지 알기 어렵다는 점이다. 논문은 cross-lingual exploration의 네 차원을 설계 변수를 삼아, 17개 유형적으로 다양한 언어의 다국어 factual benchmark에서 이를 조절·평가하는 방식으로 해결한다.

- **Empirical Impact**: 실험 결과, cross-lingual exploration은 지식 전이와 사실 회상을 유의미하게 향상시켰고, native-language scaling보다 더 효율적인 compute Pareto frontier를 보여준다. 또한 교차언어 일관성도 함께 개선되었으며, 단순 정확도 향상만으로 설명하기 어려운 수준까지 성능이 오른 것으로 관찰된다.



### NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers? (https://arxiv.org/abs/2606.24530)
- **Prior Approaches**: 기존 과학용 에이전트 벤치마크는 PaperBench 계열처럼 논문 재현(reproduction)에 초점을 두거나, PostTrainBench/엔지니어링 최적화 계열처럼 과학적 도메인 추론·특수 툴·상호 운용을 요구하지 않는 경우가 많았다. 또한 환경 분절(environment fragmentation) 문제가 있어 작업을 독립적으로 다시 돌리기 어렵고, 재현 가능성 신뢰도가 떨어진다는 한계가 지적된다. 결과적으로 ‘발견(discovery) 가능한가’보다 ‘비슷한 패턴을 재현하는가’가 평가 축으로 굳어졌다.

- **Core Contribution**: NatureBench는 Nature-family 논문에서 추출한 90개 태스크로 구성된 cross-discipline 벤치마크로, AI 코딩 에이전트가 단순 재현을 넘어 실제 과학 문제에서 방법을 새로 찾아내는지 평가한다. 핵심은 NatureGym 파이프라인으로 논문을 표준화된 컨테이너 태스크로 변환해 환경 분절을 줄이고, 정보 방화벽(information firewall)으로 원 논문의 방법을 숨겨 에이전트가 ‘발견’을 해야만 점수를 얻게 만든 점이다. Surpass-SOTA/Match-SOTA와 validity judge로 진짜 알고리즘 진보와 지름길(shortcut)·검증 회피를 분리해 평가한다.

- **Technical Challenges**: 문제는 (1) 논문 형식·툴체인·데이터 모달리티가 너무 이질적이고, (2) 태스크-평가 환경을 신뢰성 있게 재구성해야 하며, (3) ground truth·소스 방법 누출을 막으면서도 자동 평가가 가능해야 한다는 점이다. NatureGym은 3단계 build-then-verify(필터링→데이터 획득/검증→태스크 패키지 구성)와 verify–repair 루프로 태스크 경계와 평가기(evaluator) 무결성을 맞춘다. 또한 라우팅에 따라 Label/Oracle/Distribution paradigm을 지원하고, 벽시계 4시간·GPU 할당 등 통제된 예산 아래에서 내부 점수 서비스로만 평가가 이뤄지도록 설계했다.

- **Empirical Impact**: 웹 검색을 전면 비활성화한 채 10개 frontier 에이전트(3개 CLI 하네스, 10개 모델)를 NatureBench 90개 태스크에 대해 시험한 결과, 최고 성능 모델도 Surpass-SOTA(g>0.1) 달성은 17.8%에 그쳤다. 분석에 따르면 성공의 주된 경로는 과학 문제를 익숙한 supervised prediction 문제로 바꾸는 methodological translation이며, ‘진짜 과학적 발명’ 비중은 상대적으로 작다. 실패는 작업을 오해해서가 아니라 잘못된 method choice와 부족한 compute budget이 지배적으로 나타났고, NatureBench/NatureGym/리더보드를 공개해 유지자-side 재현까지 제공함으로써 AI-for-Science 평가의 신뢰도를 끌어올리는 데 의미가 있다.



### AGORA: An Archive-Grounded Benchmark for Agentic Workplace Document Reasoning (https://arxiv.org/abs/2606.24526)
- **Prior Approaches**: 기존 LLM 에이전트 평가는 주로 오픈웹 탐색이나 시뮬레이션 환경, 또는 스프레드시트/문서 조작 같은 단일 유형 작업에 초점이 맞춰져 있었다. 멀티홉 QA와 문서/테이블 QA도 존재하지만, 고정된 내부 아카이브에서 희소 근거를 찾아 단위·시간·용어를 교차 검증하는 “archive-groundedness”를 한 번에 함께 요구하는 벤치마크는 드물다. 특히 단일 소스나 단일 도메인 중심이라 도메인에 따른 취약점과 순위 역전이 가려지기 쉽다.

- **Core Contribution**: 이 논문은 실제 업무 문서 아카이브에서의 에이전트 추론을 평가하도록 Agora 벤치마크를 제안한다. Agora는 8개 도메인에 걸쳐 9,664개 문서와 372M tokens 규모의 컬렉션을 고정 소스로 두고, 362개의 교차문서·멀티홉 질문에 대해 단일 수치 정답(자동 검증 가능)을 요구한다. 이를 통해 에이전트의 문서 탐색(계획적 exploration), 근거 희소성 대응, 그리고 불일치(용어·단위·시간) 조정이 함께 측정되도록 설계했다.

- **Technical Challenges**: 핵심 과제는 (1) 모델의 사전지식으로는 풀기 어렵게 고정 컬렉션 기반을 보장하고, (2) 맥락 길이를 넘어서는 대규모·지저분한 문서에서 의도적 탐색을 강제하며, (3) 정답 검증이 가능한 형태로 누출을 억제한 멀티홉 문제를 대량 생성하는 것이다. 저자들은 cross-document task synthesis, leakage-preventing obfuscation, difficulty filtering, 그리고 인간 검증을 포함하는 agentic pipeline으로 문제를 구성하고, 문서 포맷을 Markdown/테이블 프로파일 등으로 정규화해 검색·추론에 적합한 증거 단위를 만든다. 또한 완전 누출과 정답 단서(렉시컬/구조적)를 공격 테스트로 점검해 재작성하고, 자동 패널 평가로 난이도를 다시 깎아낸다.

- **Empirical Impact**: 8개 모델을 Agora에서 평가한 결과, 최강 모델도 정확도 59.4%에 그쳐 과제가 여전히 “풀리지 않았다”고 결론낸다. 더 중요한 관찰로, 모델-도메인 조합에 따라 성능이 크게 달라 순위가 집계 평균과 다르게 뒤집히며(도메인별 체계적 블라인드 스팟), 단일 소스 벤치마크에서 숨겨졌을 취약점이 드러난다. 실패 유형 분석에서는 증거 찾기/적용(증거 오인식, 불완전 탐색 등)이 주요 병목이고, 탐색이 깊어질수록 정답으로 수렴하기보다 방향 상실로 오답이 누적되는 패턴도 확인된다.



### Poster: Exploring the Limits of Audio-Based Detection of Turkish Phone Call Scams (https://arxiv.org/abs/2606.24523)
Comments:
          Poster paper accepted at 47th IEEE Security & Privacy 2026

- **Prior Approaches**: 기존 전화 사기(Scam) 탐지는 주로 영어 등 고자원 언어의 텍스트 전사(transcript) 기반 NLP에 치우쳐, 음성의 억양·강세 같은 감정/운율 단서가 약화된다는 한계가 있었다. 또한 일부 연구는 전화 전사에서 recall이 낮거나 hallucination이 발생할 수 있음을 지적했지만, 멀티모달 신호와 저자원 언어의 현실을 충분히 반영하지 못했다. 결과적으로 문화·언어적 포괄성이 떨어져 터키처럼 데이터가 적은 환경에서는 성능 격차가 커졌다.

- **Core Contribution**: 이 논문은 터키어를 중심으로 최초의 공개 멀티모달 데이터셋을 제시하며, 사기와 정상(benign) 통화 100개의 audio-transcript pair를 정렬(aligned)해 공개한다. 동시에 입력을 raw audio, ASR 전사(자동), 네이티브가 교정한 전사(인간 수정)로 나눠 LLM 안전·전사 품질이 탐지에 미치는 영향을 체계적으로 비교한다. 저자들은 인간 교정의 추가 비용이 탐지 성능에 반드시 필요한지까지 함께 검증한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 저자원 터키어에서 ASR 품질과 발화 변형(격식/구어 차이, 교착어 형태)이 커서 텍스트 기반 단서가 흔들릴 수 있다는 점, (2) 멀티모달 모델이 원음에서 욕설·위협 표현 등을 ‘안전 필터’로 거부하면 실사용 탐지가 실패할 수 있다는 점이다. 이를 위해 파인튜닝이나 프롬프트 최적화 없이 동일한 7개 LLM을 3가지 입력 조건에 그대로 적용해 공정 비교했고, 전사는 ASR+인간 교정 두 버전을 나눠 전사 품질 민감도를 확인했다.

- **Empirical Impact**: 실험 결과, 모든 모델에서 전사 기반 입력이 raw audio보다 일관되게 우수했으며, F1 평균은 Trans/UN-Trans가 약 0.99로 Audio(약 0.97)보다 높았다. 다만 Trans와 UN-Trans의 차이는 평균 0.008 수준으로 작아, 네이티브 교정이 성능 향상으로 이어지지 않을 가능성을 보여준다. audio 조건의 주요 실패 원인은 욕설·경찰 사칭·공갈/협박처럼 실제 사기에서 자주 쓰이는 표현에서의 safety refusal로, ‘공격적 실제 사례’에서 안전장치가 실용성을 훼손할 수 있음을 실증적으로 부각했다. 연구는 저자원 언어 기반의 culturally and linguistically inclusive AI safety 연구와 더 견고한 multi-modal fraud prevention 시스템의 긴급한 필요성을 강조한다.



### UOL@IDEM at BEA 2026 Shared Task 1: Neural Fusion and Feature-Rich Modeling for L1-Aware Vocabulary Difficulty Prediction (https://arxiv.org/abs/2606.24501)
Comments:
          Published at BEA2026, 21st Workshop on Innovative Use of NLP for Building Educational Applications, at ACL, July 2026, San Diego

- **Prior Approaches**: 기존 어휘 난이도 예측(LCP/lexical complexity prediction)은 단어 빈도, 길이, 형태, 다의성, 문맥, 학습자 배경을 기반으로 읽기 난이도나 간단화에 활용돼 왔다. 그러나 BEA 2026의 핵심인 L1-aware(모국어 인지) 난이도는 같은 영어 단어라도 스페인어·독일어·중국어 화자에 따라 난이도 양상이 달라진다는 점에서, 기존 LCP 접근만으로는 전이/번역 단서 반영이 약할 수 있다. 또한 closed-track에서는 L1별로 별도 모델을 학습해야 하므로, 단순한 텍스트 인코더만으로는 L1 민감 신호를 충분히 결합하기 어렵다는 한계가 제기된다.

- **Core Contribution**: UOL@IDEM은 BEA 2026 L1-aware vocabulary difficulty prediction을 회귀(regression)로 재정의하고, Spanish→English, German→English, Chinese→English의 L1별 병렬 예측 체계를 구축했다. 멀티링구얼 contextual representations에 더해 빈도, 표면형 정보, retrieval 증거, 의미 정렬, cognate-like 유사도, masked-language-model predictability(및 surprisal)를 공학적 특징으로 통합해 L1 전이 단서를 강화한 것이 핵심이다. 특히 sentence-embedding 중심의 인코더(BGE-M3, multilingual E5, LaBSE)를 신경 융합(neural fusion) 방식으로 결합해 closed-track 기준선 대비 일관된 성능 향상을 달성했다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) L1_source_word의 노이즈·변이·부정 지시·형태 단편 등을 언어별로 정규화해 특징 추출의 일관성을 확보하는 것, (2) 문맥 기반 신호와 빈도/형태/전이(인지어·유사도) 신호를 함께 학습해 L1 민감성을 재현하는 것이었다. 이를 위해 변이/제외 조건/복합어 절단/구두점·주석 등을 정리한 뒤, retrieval(후보 뱅크 기반)·MLM 확률/엔트로피·semantic tag overlap 및 semantic shift·편언어 유사도 등을 다단계 특징으로 설계했고, 텍스트와 탭ular 특징을 함께 최적화하는 neural fusion을 주 설정으로 채택했다. 또한 각 언어별로 5-fold 교차검증, RMSE 기반 조기종료, Huber loss 검증 등을 통해 회귀 안정성과 이상 잔차의 영향 완화를 병행했다.

- **Empirical Impact**: 개발 세트에서 neural fusion+sentence-embedding 인코더는 official closed-track baseline을 모든 언어에서 일관되게 능가했으며, RMSE를 최대 0.21–0.26 수준까지 개선했다. 공식 제출의 test RMSE는 Spanish 1.132, German 1.037, Chinese 0.891로 보고됐고, 분석 결과 frequency(빈도)가 가장 안정적인 예측자이며 contextual predictability, form similarity, retrieval, semantic 특징이 보완적 신호를 제공함을 확인했다. 다만 오류 분석에서 상위 성능은 유지하되 보정(calibration)은 쉬운 구간에서 과대예측, 어려운 구간에서 과소예측으로 ‘중앙으로 압축’되는 경향이 보여, 향후 보정 개선과 L1별 전이·문맥 회수 가능성에 대한 더 효율적인 모델링이 필요하다는 시사점을 남겼다.



### The African Language Tax: Quantifying the Cost, Latency, and Context Penalty of Tokenizing African Languages in Frontier LLMs (https://arxiv.org/abs/2606.24460)
Comments:
          40 pages, 5 figures, 25 tables

- **Prior Approaches**: 기존 연구는 다국어 토크나이제이션이 언어 간 효율 차이를 만들고, 토큰 수 격차가 비용과 활용성 불평등으로 이어진다는 현상을 일반적으로 다뤘습니다. 다만 유럽 중심의 병렬 실험에 비해 아프리카 언어는 같은 해상도로 체계 측정되지 않았고, 특히 실제 기업 배포 관점(비용·지연·컨텍스트 용량)으로 환산된 정량 결과가 부족했습니다.

- **Core Contribution**: 이 논문은 FLORES-200+ 등 병렬 코퍼스를 바탕으로 20개 아프리카 언어(5개 언어 계통, 3개 문자: Latin, Ge'ez/Ethiopic, N'Ko)를 대상으로 토크나이저의 ‘tokenization premium(영어 대비 토큰 프리미엄)’을 정밀 측정하고, 이를 배포 의사결정 단위로 변환합니다. 또한 afri-fertility 공개 도구, 공개 리더보드, 결과 데이터셋, 완화 가이드를 함께 제공해 측정과 검증을 반복 가능하게 만들었습니다. 결과적으로 토크나이제이션 격차를 “디지털 격차가 서브워드 어휘에 인코딩된 구조적 페널티”로 명확히 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘의미는 동일하지만 토큰 수만 달라지는’ 병렬 설계를 통해 콘텐츠 편향을 제거하면서도, 언어별 단어 경계·정규화·문자 밀도 차이까지 통제하는 것이었습니다. 저자들은 UAX-29 기반 단어 분절과 NFC 정규화를 고정하고, 필요 시 문자/바이트 기반 지표(CPT/BPT)를 병행해 형태론적으로 분절이 흔들릴 수 있는 언어(특히 에티오피아 문자·고교착 언어)에서 결론이 단어 수에만 의존하지 않도록 했습니다. 더 나아가 토크나이저별 토큰 수를 합산-나눗셈(sum-then-divide)으로 집계하고, 문장 단위 부트스트랩 95% 신뢰구간으로 안정성을 확보했습니다.

- **Empirical Impact**: 11개 frontier·open 토크나이저를 FLORES-200+에 적용한 결과, 모든 아프리카 언어는 영어보다 높은 토큰 프리미엄을 보였고 중앙값은 1.88x(예: GPT-5 / o200k_base), 최대는 N'Ko에서 8.92x에 달했습니다. 비용·지연·컨텍스트 관점으로 환산하면 최대 추론 비용 8.9x, 생성 지연 배율 8.9x, 그리고 컨텍스트 윈도우 유효 용량이 영어 대비 최소 11% 수준으로 줄어드는 것으로 나타났습니다. Gemma 4가 프리미엄 평균을 3.31x(cl100k_base)→2.38x로 낮추지만, 페널티를 완전히 제거하는 토크나이저는 없었고 아프리카 빌더의 비용 부담이 언어 사용 가능성과 직접 연결된다는 점을 실증했습니다.



### Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning (https://arxiv.org/abs/2606.24428)
Comments:
          28 pages, 11 figures

- **Prior Approaches**: 기존 experience learning 계열은 대부분 단일 에이전트 루프에서 실행→요약(distill)→기억 저장→재사용을 반복하며 성능을 끌어올립니다. 다만 공개 환경처럼 명시적 ground truth가 없는 경우, 같은 주체가 생성과 평가를 동시에 담당해 Self-Confirmation Trap(자기 일관성은 맞지만 실제론 틀린 궤적을 성공으로 오인)을 키울 수 있습니다. 이 때문에 장기 horizon에서 잘못된 경험이 메모리에 누적되고, 이후 retrieval과 reuse로 오류가 증폭될 위험이 큽니다.

- **Core Contribution**: 논문은 Execute-Distill-Verify(EDV) 프레임워크로 신뢰도 높은 경험을 만들기 위한 파이프라인을 제안합니다. 핵심 아이디어는 execution, distillation, verification을 분리하고, 서로 다른 역할(이질적 실행 그룹, third-party distillation agent, consensus 기반 검증 그룹)이 협력해 메모리 삽입 전에 오류·잡음을 걸러내는 것입니다. 즉 “더 많이 저장”이 아니라 “저장할 만한 경험을 더 정확히 구성”하는 쪽에 초점을 둡니다.

- **Technical Challenges**: 가장 큰 technical challenge는 정답 판별이 어려운 open-world에서, 실행한 에이전트의 자기 평가 편향을 어떻게 끊어낼지입니다. EDV는 (1) 이질적 multi-agent 병렬 실행으로 후보 궤적의 다양성을 확보하고, (2) executor-centric 요약 편향을 줄이기 위해 third-party distillation으로 다중 궤적의 비교 기반 경험을 뽑으며, (3) verification 단계에서 consensus로 unanimous 승인만 공유 메모리에 넣고 그 외는 private 또는 폐기하는 default-reject 정책을 사용합니다. 추가로 Ability Matrix와 shared/private 메모리 계층 구조를 두어, 추론 시 작업-솔버 매칭과 계층적 retrieval이 가능하게 했습니다.

- **Empirical Impact**: EDV는 tau2-bench, Mind2Web, MMTB의 장기 horizon 벤치마크 3종에서 강한 베이스라인을 일관되게 능가하며, 예컨대 tau2-bench에서 평균 Pass@1 86.6으로 Router와 Judge보다 높은 성능을 보였습니다. 사람 평가 기반 메모리 품질 감사에서도 EDV가 Noise/Hallucination과Potential Harm(재사용 시 피해)를 낮추는 등 ‘저장 내용의 신뢰성’이 개선됨을 확인했습니다. 또한 RB 메모리에 의도적으로 오류 경험을 섞으면 성능이 크게 하락해 Self-Confirmation Trap의 실제 위험을 검증했고, ablation 결과 EDV의 three-stage 분리·contrastive distillation·consensus 검증이 시너지로 작동함을 보여주었습니다.



### Beyond Logprobs: A Multi-Signal Confidence Engine for LLM-Based Document Field Extraction (https://arxiv.org/abs/2606.24420)
Comments:
          Extended version of a paper accepted (Oral) at the RobustifAI Workshop, IJCAI-ECAI 2026, Bremen, Germany. 9 pages, 5 figures, 2 tables

- **Prior Approaches**: 기존 문서 필드 추출의 신뢰도 평가는 주로 token-level log-probabilities, LLM의 verbalized confidence, self-consistency 같은 생성 결과 기반 지표에 의존했다. 그러나 DocILE에서 이러한 방식은 거의 모두를 ‘신뢰’로 분류하는 붕괴가 나타나며, 로그확률 평균도 ROC AUC 0.705 수준에 그쳤다. 이는 OCR 잡음·레이아웃 애매성·판독 불가처럼 모델이 관측할 수 없는 문서 원인 때문에 오류가 발생하는 구조적 문제로 설명된다.

- **Core Contribution**: 논문은 ExtractConf를 제안해 “이 추출을 자동화에 써도 되는가”를 field 단위로 판단하는 신뢰도 엔진을 만든다. 핵심 아이디어는 같은 문서를 두 가지 관점에서 읽는 Hunter–Mapper 듀얼 콜 비대칭 설계로, 두 읽기의 failure mode를 다르게 만들어 불일치를 신호로 활용한다. Hunter는 스키마 슬롯 완성 압력 때문에 없는 필드에도 그럴듯한 값을 만들 수 있고, Mapper는 문서 근거가 있는 값만 제시하므로 두 콜의 합의/불일치가 신뢰도에 직접 반영된다.

- **Technical Challenges**: 관측 불가능한 문서 원인(낮은 화질, OCR 오류, 레이아웃 불명확성)이 생성 토큰의 확률값과 일치하지 않는다는 점이 큰 기술 난제였다. 이를 해결하기 위해 ExtractConf는 (1) Hunter/Mapper 각 호출의 log-prob 및 엔트로피 통계, (2) OCR 값-영역(value-region) 정합도, (3) 라벨/값의 OCR 신뢰도, (4) 두 콜이 문서에서 읽은 위치의 공간적 지표(바운딩박스, centroid divergence, Laplacian sharpness), (5) 필드 타입 원-핫을 CatBoost 이진 분류기로 결합한다. 도메인 규칙이나 재학습 없이, 예측 점수는 isotonic regression(M7) 또는 Lasso-calibrated logistic regression(M8)로 사후 보정해 ECE/Brier를 낮춘다.

- **Empirical Impact**: DocILE(55-field, 자연 실패율 26%)에서 ExtractConf는 ROC AUC 0.928과 AURC 0.042를 달성하며, logprob-mean 대비 selective prediction risk를 70% 낮춘다. 또한 커버리지 80%에서 자동 정확도 99.1%를 기록해 production에서 필요한 human-in-the-loop 라우팅이 가능하다고 제시한다. Zero-shot으로 CORD 영수증(다른 도메인)에도 적용했을 때 ROC AUC 0.858을 얻고, Lasso 재보정은 ECE를 89%, Brier를 43% 감소시켜 신뢰 신호가 문서 유형 전반에 일반화됨을 실증한다.



### AutoSpecNER: A Fine-Grained Named Entity Recognition Dataset for Vehicle Specification Extraction (https://arxiv.org/abs/2606.24387)
Comments:
          13 pages, 2 figures, 7 tables, Pre-print

- **Prior Approaches**: 기존 NER 벤치마크(CoNLL-2003 등)는 사람/장소 같은 조밀하지 않은 엔티티에 집중해 자동차 광고의 기술 스펙 추출과는 맞지 않았다. 자동차 쪽 연구도 가벼운 속성 추출이나 도메인 적응 중심이었고, 모델명·엔진·배터리 용량처럼 fine-grained 스펙을 세밀하게 식별해 검증하는 데는 자료와 성능이 부족했다. 또한 규칙 기반은 태그 사전/정규표현식 의존도가 높고 LLM은 경계·정규화·용어 변형에서 오류가 잦다는 한계가 관찰된다.

- **Core Contribution**: 논문은 자동차 광고용 fine-grained entity recognition을 위한 AutoSpecNER 데이터셋을 제안한다. UK 중대형 중고차 판매 사이트 광고 659건에서 15개 엔티티(예: MODEL, ENGINE_SPEC, BATTERY_CAPACITY)로 1만 개 이상 엔티티를 주석했으며, inter-annotator agreement 평균 91.5%로 라벨 품질을 검증했다. 이어서 rules-based, fine-tuned transformer encoder, few-shot 기반 LLM(자기검증 포함)까지 포괄하는 벤치마크를 제공해 모델 선택의 중요성을 정량화한다.

- **Technical Challenges**: 핵심 난제는 도메인 특화 용어(예: 2.0L TDI, kWh, bhp), 비슷한 개념 간 구분(예: 실내/실외 컬러, 용량/주행거리), 그리고 다중 토큰 엔티티·인접 엔티티의 경계 처리였다. 저자들은 사용자 오타/비격식과 AI-generated hallucination이 섞인 이중 소스 광고를 구성하면서도, 주석이 ‘정형 스펙’이 아니라 ‘광고 텍스트의 주장(span)’을 그대로 따르도록 지침을 설계했다. 또한 모든 시스템을 tokeniser-independent 문자 단위 IOB2로 정렬해 공정한 비교를 가능하게 했고, LLM에는 GPT-NER 방식의 few-shot per-label 프롬프팅과 self-verification으로 오탐을 줄이는 절차를 적용했다.

- **Empirical Impact**: 실험 결과 fine-tuned encoder가 전반적으로 최선이며, 특히 DeBERTa-v3-base가 micro-F1 90.1%(약 90%)로 rules-based(43%)와 LLM 상위 모델(최고 77.8%대)을 앞섰다. 규칙 기반은 YEAR/MAKE/MODEL처럼 패턴이 단순한 태그에서 강했지만, INTERIOR_COLOUR·EXTERIOR_COLOUR 같은 표현 변이가 큰 태그에서는 극히 낮은 성능(<0.03 F1)을 보였다. 한편 LLM은 NO_SEATS·BATTERY_RANGE처럼 데이터 지지가 적은 레이블에서 상대적으로 유리하거나(지원이 낮을 때) 향상 가능성이 나타났고, hallucination 대응 같은 검증형 다운스트림에 활용될 수 있음을 시사한다.



### On the Stability of Prompt Ranking in Large Language Model Evaluation (https://arxiv.org/abs/2606.24381)
- **Prior Approaches**: 기존 프롬프트 엔지니어링 연구는 instruction phrasing, reasoning cues, 출력 제약 등 설계가 성능에 큰 영향을 준다는 점을 보여줬습니다. 또한 few-shot 예시 선택이나 stochastic decoding, 데이터 서브샘플링에 따른 출력/점수 변동은 다뤄졌지만, 프롬프트들 사이 ‘순위’의 안정성은 체계적으로 다루지 않았습니다. 결과적으로 평균 정확도나 단일 최고 프롬프트 같은 결정이 평가 잡음에 얼마나 취약한지 과소평가되어 왔습니다.

- **Core Contribution**: 이 논문은 프롬프트 평가를 확률적(불확실성 포함) 순위 문제로 보고, random seed와 제한된 평가 예산(서브셋 크기) 같은 현실적 변동 하에서 순위 안정성을 정량화합니다. 특히 Spearman/Kendall 같은 전역 순위 상관이 중간~높게 나와도 top-1(또는 top-k) 선택은 자주 바뀐다는 ‘결정 수준의 불일치’를 보여줍니다. 이를 바탕으로 mean 기반 선택 대신 performance와 분산을 함께 고려하는 lower confidence bound(LCB) 기반 안정성-aware 선택 전략을 제안합니다.

- **Technical Challenges**: 핵심 과제는 평가 조건이 바뀔 때 프롬프트 성능의 상대적 순서가 얼마나 흔들리는지, 그리고 그 흔들림이 실제 top-prompt 선택을 어떻게 망가뜨리는지 분해해 측정하는 것입니다. 저자들은 고정된 프롬프트 풀을 여러 seed와 서브셋 크기로 반복 평가해 점수 행렬을 만들고, Spearman’s ρ·Kendall’s τ로 전역 순위를, top-1/top-k consistency로 선택 결정의 재현성을 동시에 봅니다. 또한 leave-one-seed-out(LOSO) 프로토콜로 선택 전략이 보이지 않는 평가 조건에서도 일반화되는지 검증하고, LCB로 불확실성을 페널티화해 noisy 상황의 선택을 보정합니다.

- **Empirical Impact**: Mistral-7B-Instruct-v0.3, Phi-3-mini-4k-instruct, Qwen2.5-7B-Instruct의 세 모델과 GSM8K·MMLU 두 벤치마크에서, 전역 순위 상관은 종종 중간~높지만 top-1 일치율은 낮아(특히 GSM8K에서 서브셋 50 기준 top-1 약 40%대) ‘선택 결정’이 불안정함을 확인했습니다. LOSO에서 LCB 전략은 평균 기반 선택을 작은 평가 예산에서 유의미하게 개선하며(예: Qwen GSM8K size 50에서 0.228→0.312), 큰 예산에서도 경쟁력을 유지합니다. 반면 비교적 안정적인 MMLU에서는 분산 페널티로 평균 정확도가 소폭 감소할 수 있어 robust trade-off를 보여줍니다. 전반적으로 프롬프트 배치/벤치마킹에서 점수 한 점(점추정)에만 의존하지 말고 평가 불확실성을 반영해야 한다는 실무적 경고와 방법론을 제공합니다.



### MorfFlex: Handling Rich Morphology (https://arxiv.org/abs/2606.24366)
Comments:
          Accepted to LREC 2026

- **Prior Approaches**: 기존 형태소 리소스는 형태 분석기·어형 생성기 또는 철자 검사 목적의 사전 형태로 발전해 왔고, 굴절이 복잡한 언어에서는 표준화된 어형-어근(lemma)+굴절 조합으로 접근하는 경우가 많다. 다만 방대한 어형을 모두 수록하면 사전 크기가 폭증해 유지보수와 일관성 관리가 어렵고, 규칙을 두더라도 사람이 편집 가능한 형태로 관리하는 체계가 부족했다.

- **Core Contribution**: 본 논문은 굴절(inflection)과 파생(derivation) 모두에 정규성이 큰 언어를 위한 형태소 사전 아키텍처 MorfFlex를 제안한다. 이를 체코어 형태소 사전 MorfFlex CZ에 적용해, 실제 배포 포맷은 <wordform, lemma, tag> 삼중항 목록이지만, 내부적으로는 수많은 어형을 패턴으로 압축·생성하도록 설계했다. 결과적으로 수작업 형태소 어노테이션의 일관성 유지와 MorphoDiTa 같은 최신 NLP 도구 개발의 기반 자원을 함께 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 어형 수가 지나치게 많아 사전 크기를 감당하기 어렵고, (2) 굴절·파생 규칙을 사람도 관리할 수 있는 방식으로 표현하며, (3) 생성 결과가 분석/어노테이션에 대해 모호하지 않게 보장하는 것이다. 저자들은 source 포맷의 파생 패턴을 중간(intermediate) 포맷의 굴절 패턴으로 변환한 뒤 최종 basic 포맷의 삼중항을 생성하는 2단계 절차를 두었고, lemma-tag 쌍의 Golden rule로 <lemma, tag>의 중복 사용을 막아 생성의 비일의성을 차단한다.

- **Empirical Impact**: MorfFlex CZ는 source 포맷의 약 45만 행에서 출발해 1억 개 이상 어형과 100만 개 이상 lemma를 자동 생성하며, 이 구조가 수작업 어노테이션 일관성 확보에 실질적으로 쓰이고 있음을 강조한다. 또한 Prague Dependency Treebanks와 호환되며 MorphoDiTa의 사전·태거 핵심으로 사용되어 태깅 F1 96.27%, lemmatization F1 98.31% 수준의 성능과 초당 10–200K words 처리량을 보고한다. DeriNet과의 어휘 정렬 및 UDPipe+사전 리스코어링 개선 등으로 리소스가 후속 모델 성능에도 영향을 주는 것으로 정리된다.



### Automatic Part-of-Speech Tagging of Arabic-English Dictionary Senses through WordN (https://arxiv.org/abs/2606.24359)
Comments:
          10 pages, 3 figures, 5 tables, Published in Proceedings of the 15th Conference on Language Engineering, Egyptian Society of Language Engineering (ESOLE'15), Dec., 2015

- **Prior Approaches**: 기존 POS 태깅은 통계 기반으로 대규모 라벨링 코퍼스가 필요하거나, 규칙 기반으로는 풍부한 문법·세계지식을 담은 대형 어휘지가 요구된다. 또한 NLP/HLT 도구를 구축하려면 언어 전문가, 큰 투자, 긴 개발 기간이 뒤따른다. 이런 부담 때문에 자원(light) 접근이 저자들이 문제의식으로 삼은 대안으로 부상해 왔다.

- **Core Contribution**: 이 논문은 이중언어 사전의 항목(사전 sense)에 대해 POS 태깅을 수행하는 알고리즘을 제안한다. 핵심은 영어 번역 동치(TE)의 POS 태그를 WordNet에서 얻은 영어 POS 태그 정보로부터 사전 sense에 전이(transfer)하고, 모호성 제거(dis-ambiguities) 이후에 이를 사전 sense에 할당하는 방식이다. 이를 통해 이중언어 사전을 WordNet과 연결하거나 WordNet-LMF 포맷으로 표준화하기 위한 전처리 단계를 제공한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 사전 sense와 영어 번역 동치 사이의 매핑 불확실성을 모호성 제거로 처리하는 것이다. 저자들은 영어 POS는 Princeton WordNet에서 획득하고, dis-ambiguities 과정을 거쳐 TEs로부터 sense에 태그를 안정적으로 전이하는 파이프라인을 구성했다. 그 결과, 추가 비용이 크지 않은 형태로 높은 정확도를 등록할 수 있었다.

- **Empirical Impact**: Al-Mawrid 아랍어-영어 사전(Arabic-English dictionary)에 적용했을 때 POS 태깅 정확도가 높게 보고된다. 특히 계산 비용은 적다고 제시되어, 제한된 자원에서도 사전을 WordNet/WordNet-LMF에 연결하기 위한 실용적 전처리로 의미가 있다. 자원 부족 언어를 위한 resource-light 도구 개발 흐름에 기여할 수 있는 결과로 해석된다.



### Meet UD_Czech-PDTC: A Large and Genre-Rich Treebank in Universal Dependencies (https://arxiv.org/abs/2606.24337)
Comments:
          Accepted to LREC 2026

- **Prior Approaches**: Czech가 Universal Dependencies(UD)에 포함된 초기부터, Prague Dependency Treebank(PDT)는 규모와 품질 면에서 대표적인 언어 자원으로 자리잡아 왔다. 다만 최근 추가된 Prague 계열 데이터와 재주석으로 인해, 기존 PDT와의 차이를 어떻게 UD로 일관되게 맞출지가 새로운 과제로 부상했다.

- **Core Contribution**: 이 논문은 Prague Dependency Treebank-Consolidated(PDT-C) 자원을 UD로 변환하는 절차를 구체적으로 정리한다. 표면적으로는 두 주석 체계가 유사해 보이지만, 의존구조 토폴로지와 POS/관계 타입의 세분화(granularity)에서 많은 미세 차이가 존재함을 지적한다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 PDT-C의 주석 토폴로지 및 라벨 체계가 기존 PDT/UD와 정확히 일치하지 않는 점이다. 논문은 이런 차이를 예시로 보여주고, 변환 과정에서 동기(motivations)가 갈라지는 부분을 해석하며, 그 차이를 흡수/정규화하는 변환 전략을 함께 제안한다.

- **Empirical Impact**: PDT-C는 PDT 대비 크기는 2배 이상이면서도 장르·도메인 다양성이 크게 늘어 UD 생태계의 커버리지를 확장한다. 또한 PDT의 다층(multi-layer) 주석이 기본 UD 트리를 만드는 데 필요한 정보를 충분히 제공한다는 관점을 제시해, 향후 추가 자원 통합과 변환 설계에 실질적인 기준을 제공한다.



### Transformer-Based Language Models Across Domain Verticals: Architectures, Applications and Critical Assessmen (https://arxiv.org/abs/2606.24331)
- **Prior Approaches**: 기존 자연어 처리에서는 번역·요약에 RNN/LSTM 계열이 중심이었지만, 토큰 단위 순차 처리 때문에 병렬화와 장거리 의존 학습에 제약이 컸다. 이후 Transformer는 recurrence를 제거하고 self-attention으로 병렬 학습이 가능해지며 성능과 확장성이 급격히 개선되었지만, 배포 관점의 선택 기준이 문헌에서 일관되게 정리되진 않았다. 또한 기존 리뷰들은 주로 출시 시점·모델 규모 중심으로 분류되어, 실제 현업에서 중요한 아키텍처 성격과 비용·안전 트레이드오프가 흐려진다는 한계가 있었다.

- **Core Contribution**: 이 논문은 Transformer 계열을 encoder-only, decoder-only, encoder-decoder, long-context, permutation-based, generator-discriminator 변형으로 재구성해 ‘배포 의사결정에 유용한’ 실무형 택소노미를 제공한다. 아울러 2023년 이후 실전 양상을 바꾼 instruction tuning, RLHF, DPO, mixture-of-experts, retrieval augmentation, 주요 상용/오픈 모델 패밀리를 정리한다. 마지막으로 아키텍처를 배포 의사결정에 필요한 축(예: compute/에너지, 파라미터 대비 비용)에서 비판적으로 비교하고, alignment·데이터 provenance·benchmark 포화가 ‘state of the art’의 의미를 어떻게 바꾸는지 논의한다.

- **Technical Challenges**: Transformer를 실서비스에 적용할 때 핵심 기술 문제는 (1) 생성의 무제한 출력공간이 만드는 평가/정렬의 어려움, (2) prompt injection 같은 새로운 보안 취약점, (3) MoE의 라우팅·서빙 복잡도(정확도보다 운영 난이도) 등이다. 논문은 decoder-only가 강력한 generality를 제공하지만 output이 무한해 정답 정의가 모호해지고 얼핏 보이지 않는 취약점이 생긴다고 짚는다. 또한 에너지 비용과 파라미터 수의 단순 상관이 깨지므로, attention/롱컨텍스트 효율화와 RAG, 그리고 정렬 방법 선택이 실제 비용·신뢰성에 미치는 영향을 함께 봐야 한다고 정리한다.

- **Empirical Impact**: 논문은 도메인별 활용 사례를 통해, 의료·금융·법률·교육·고객응대·크리에이티브·과학 연구에서 어떤 Transformer 속성이 실질 이득을 주는지 연결한다(예: 의료의 구조화 정보 추출은 임베딩 기반 접근이 유리, 법률은 ‘retrieve-extract-verify’ 패턴이 생성보다 안전). 특히 RAG는 환각과 최신성 문제를 완화하지만, 결국 신뢰성 병목이 retriever로 이동한다는 점을 실증적 관점에서 강조한다. 종합하면 이 리뷰는 단일 모델 성능이 아니라 배포 환경의 제약(비용·에너지, 정렬, 데이터 출처, benchmark-실사용 격차)을 기준으로 판단해야 한다는 메시지를 업계 의사결정에 직접 겨냥한다.



### Prague Dependency Treebank -- Consolidated 2.0: Enriching a Complex Annotation Schem (https://arxiv.org/abs/2606.24324)
Comments:
          Accepted to LREC 2026

- **Prior Approaches**: 기존 Prague Dependency Treebank(PDT) 계열은 다층 언어 정보를 다루려 했지만, PDT-C 1.0에서는 일부 계층이 자동/부분적으로만 정리돼 일관성·품질이 제한됐다. 또한 의미 표현과 코어퍼런스·담화 같은 문장 간 현상을 포함하더라도, 장르 전반에 걸친 수작업 정비가 충분치 않아 비교 연구에 제약이 있었다.

- **Core Contribution**: PDT-C 2.0(Praugue Dependency Treebank–Consolidated 2.0)은 약 4M 토큰 규모의 체코어 데이터를 다층(morphology–surface syntax–deep syntax/semantics)으로 통합하고, 전 계층을 거의 “완전한 수작업 수동 주석”으로 일관되게 통합했다. 코어퍼런스와 담화 관계의 수작업 주석을 모든 데이터셋에 확장하고, MorfFlex(형태소)와 PDT-Vallex(발렌시) 같은 완전 호환 렉시콘까지 함께 제공한다.

- **Technical Challenges**: 핵심은 서로 다른 층의 주석이 정보 손실 없이 연결되도록(하위 층 토큰을 상위 층에서 참조) 설계·링킹하는 것이고, 특히 문장 간 코어퍼런스/담화는 문맥 추론이 필요해 라벨링 난도가 높다. 논문은 수작업 주석 과정에서 모든 계층을 동시에 추적해 기존 주석의 수정·정합성 보정이 누적되도록 했고, 담화는 subtrees(루트) 단위로 명시하며 코어퍼런스는 체인 원칙과 예외(카타포라, 상황/구간 참조)를 함께 다뤘다.

- **Empirical Impact**: PDT-C 2.0 기반으로 형태소/구문/의미(semantic graph) 파서들을 재학습했으며, 예를 들어 UDPipe에서 구문 파싱 거시 평균 오차율이 27% 이상 개선되는 등 모델 성능 향상이 보고됐다. 또한 PDT-C 2.0은 UD, CorefUD, PDiT 등 다양한 포맷으로의 변환과 확장 프로젝트의 기반으로 쓰이며, 코어퍼런스·담화가 포함된 풍부한 주석이 국제 비교와 차세대 NLP 도구 개발에 실질적 자산이 된다는 점을 강조한다.



### AVOC: Enhancing Hour-Level Audio-Video Understanding in Omni-Modal LLMs via Retrieval-Inspired Token Compression (https://arxiv.org/abs/2606.24286)
- **Prior Approaches**: 기존 멀티모달 LLM은 오디오-비디오를 인코더로 처리한 뒤 LLM 문맥에 넣어 short-form 과제에서는 성과를 냈지만, hour-level로 길어지면 컨텍스트 윈도우 한계와 중복 정보로 인해 추론 품질이 급락한다. 컨텍스트 축소도 content-agnostic sampling(샘플링)이나 단순 truncation에 치우쳐 중요한 사건을 놓치거나 토큰 예산을 빠르게 소진하는 문제가 있었다. OmniZip/OmniSIFT 같은 압축 접근은 대개 한 모달리티가 다른 모달리티를 유도하는 비대칭 설계를 써서, 유도 신호가 희박하면 핵심 정보가 버려질 위험이 남았다.

- **Core Contribution**: 이 논문은 long-form audio-video 이해를 위한 Omni-modal LLM용 프레임워크 AVOC를 제안한다. 핵심은 멀티모달 token compression을 top-K retrieval 문제로 재정의하고, 고정된 컨텍스트 예산 안에서 질의에 유리한 토큰 부분집합을 “검색”하듯 학습하도록 만든 점이다. 또한 IR의 relevance(관련성), importance(중요도), diversity(다양성) 3축을 각각 오디오-비디오 이해에 맞는 메커니즘으로 구현해 하나의 통합 압축 파이프라인으로 엮었다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 긴 시퀀스에서 토큰 수를 줄이되 질의-관련 단서를 보존해야 하고, (2) top-K 선택이 비미분적이라 학습을 end-to-end로 이어가기 어렵다는 점이다. AVOC는 text-guided cross-attention으로 relevance를 점수화하고, temporal block 내 bidirectional video-audio cross-attention으로 query-agnostic importance를 보완하며, Temporal-Aware Maximal Marginal Relevance(TA-MMR)로 로컬 시간창 기준 중복을 억제해 diversity를 확보한다. 학습 단계에서는 Gumbel-Softmax 기반 differentiable top-KK 선택과 Straight-Through Estimator를 사용해 다음-토큰 예측 손실에서 압축 모듈의 projection 계층까지 그라디언트를 전달하고, TA-MMR은 추론에서만 greedy 재랭킹으로 적용한다.

- **Empirical Impact**: AVOC는 OmniVideoBench와 LVOmniBench 등 long-form 오디오-비디오 벤치마크에서 SOTA를 달성했으며, 2등 대비 평균 정확도가 각각 4.9점, 5.5점 더 높다. 특히 성능 향상이 비디오 길이에 따라 커져 초장 문맥에서 압축 모듈의 효과가 두드러진다. 또한 Audio-Video Needle-in-a-Haystack에서 최대 1시간 길이까지 질의 기반 “정밀 검색/국소화” 정확도를 견고하게 유지해, 모델이 hour-level 컨텍스트를 실사용 관점에서 다룰 수 있음을 실증한다.



### CALIBER: Calibrating Confidence Before and After Reasoning in Language Models (https://arxiv.org/abs/2606.24281)
- **Prior Approaches**: 기존 LLM 보정(calibration)은 주로 한 번의 confidence만 뽑아 토큰 확률, 샘플 간 일관성, 또는 verbalized confidence를 학습해 왔다. 다만 reasoning 모델은 생각(thinking trace) 전후 정보 상태가 달라지는데, 기존 방식은 보통 confidence 위치(생각 전/후)와 보정 타깃(프롬프트 성공률 vs 정답 정합성)을 단일 조합으로 고정해 ‘정보 상태-감독 신호’ 불일치가 생길 수 있다.

- **Core Contribution**: 이 논문은 reasoning 모델의 confidence가 state-dependent하다고 보고, 생각 전 confidence는 ‘프롬프트를 풀 확률’, 생각 후 confidence는 ‘실제 생성된 답이 맞을 확률’을 각각 추정해야 한다고 정리한다. CALIBER는 하나의 응답에 pre-와 post-thinking confidence를 모두 생성하고, 각 예측에 맞는 타깃(그룹 수준 vs 인스턴스 수준)을 정확히 매칭해 학습시킨다. 그 결과 두 점수는 같은 스칼라가 아니라 역할을 분화하며, thinking에 따른 불확실성 변화도 진단 신호로 남긴다.

- **Technical Challenges**: 핵심 난제는 ‘position–target alignment’를 분리해 실험적으로 검증하는 것인데, 선행법들은 위치/타깃/학습 알고리즘이 함께 달라 공정 비교가 어렵다. 저자들은 같은 학습·평가 프로토콜 하에서 prior-inspired 조합들을 controlled baseline으로 두고, CALIBER는 GRPO 스타일 RLVR에 포맷 보상과 함께 pre(그룹 타깃)·post(인스턴스 타깃) 보정 보상을 동시에 넣되 학습 붕괴를 막기 위해 pre 보상만 먼저 warmup하는 2단계 학습을 설계했다.

- **Empirical Impact**: BigMathDigits에서 7B 모델은 CALIBER가 단일 confidence 강baseline 대비 ECE를 52.5% 줄였고, Brier score와 AUROC도 최상급이며 정확도는 큰 폭으로 훼손되지 않았다(최대 2.1점 이내). 30B에서도 BigMathDigits ECE 최우수 성능을 보이면서 Brier와 AUROC가 경쟁 수준을 유지했고, OOD인 GPQA와 TriviaQA에서는 ECE와 Brier를 가장 잘 낮추며 SimpleQA에서도 준수한 성적을 냈다. 또한 ablation에서 alignment(정보 상태에 맞춘 타깃 매칭)이 특히 분포 이동에서 보정 개선에 가장 유익함을 보여, 실제 신뢰도 추정이 필요한 배포 시나리오에 의미가 크다.



### Pigeonholing: Bad prompts hurt models to collapse and make mistakes (https://arxiv.org/abs/2606.24267)
- **Prior Approaches**: 기존 in-context learning 연구는 일반적으로 성능 향상을 보여주지만, 부적절한 문맥이 들어오면 성능 저하와 모드 붕괴가 발생할 수 있다는 점을 충분히 다루지 못했다. 특히 사용자의 의도와 무관하게(잘못된 정리 정당화 유도, 버그가 있는 코드 수정 누락 등) 나쁜 context가 섞일 때의 실패 양상이 체계적으로 정리되지 않았다.

- **Core Contribution**: 이 논문은 이러한 현상을 'pigeonholing'으로 명명하고, 두 시나리오—(1) 사용자가 해결책을 제시하는 경우, (2) 대화 문맥에 어시스턴트의 이전(오답) 응답이 포함되는 경우—에서의 양상을 분석했다. 또한 pigeonholing 완화를 위해 RLVR with synthetic errors(합성 오류 기반 RLVR)를 제안해, 나쁜 문맥에서도 모델이 망가지는 문제를 줄이는 방향을 제시한다.

- **Technical Challenges**: pigeonholing의 핵심은 나쁜 문맥이 모델을 과거의 오류/선호로 강하게 수렴시켜, 대안 탐색이 사라지고 답이 반복·고정되는 데 있다. 연구자들은 10개 검증 가능 및 오픈엔드 태스크, 10개 서로 다른 모델에서 (오답 반복으로 38-40% 성능 하락, 텍스트/코딩에서 답의 좁은 집합 수렴, 논쟁 주제에서 입장 뒤집기 등) 구체적 모드 붕괴 패턴을 관찰하고, 합성 오류로 학습을 보강하는 RLVR 접근을 통해 이를 완화하려 했다.

- **Empirical Impact**: 실험 결과 pigeonholing은 대화 턴 수가 늘수록 거의 단조 증가하며(오류 반복이 1에서 5로 늘 때 추가 14%+ 성능 하락), 예시가 정답이라도 모드 붕괴가 발생할 수 있음을 보였다. 제안한 RLVR with synthetic errors는 bad context에서 vanilla RLVR 대비 43-60% 개선을 보여, 실전 대화형 AI의 문맥 오염 취약성을 완화하는 데 의미 있는 진전을 제공한다.



### SURGELLM: Rethinking Multi-Task Evaluation through Task-Aware Feature Gating with Class-Balanced Normalization (https://arxiv.org/abs/2606.24259)
Comments:
          Proceedings of the 6th Workshop on Trustworthy NLP (TrustNLP 2026), ACL 2026, San Diego, California, USA. Available at this https URL

- **Prior Approaches**: 기존에는 태스크별로 fine-tuning된 인코더를 두거나, multi-task learning으로 공유 인코더를 쓰는 방식이 주류였다. 하지만 서로 다른 어휘·라벨공간·문체를 가진 이질적 태스크가 섞이면 feature 간섭이 커져 성능 저하가 쉽게 나타난다. 또한 handcrafted lexical feature를 주입하더라도 class-imbalance로 인해 gate가 관찰하는 feature 통계가 왜곡되는 문제는 충분히 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 SURGeLLM이라는 통합 transformer 프레임워크로 이질적 NLP 태스크에서 외부 어휘 지식을 attention에 직접 조건화하면서도 공유 백본을 유지한다. 핵심은 (1) 차원별 surgical feature gate로 유용한 어휘 지표만 [CLS]와 결합하고, (2) task-conditioned prefix tokens로 입력 전반에 feature 값을 주입하며, (3) Instance-Weighted Normalization(IWN)으로 class-imbalance가 feature 통계를 망가뜨리는 효과를 제거하는 것이다. gate의 효용이 surgical feature alignment와 연결됨을 이론적으로 보이고, feature가 무의미할 때는 identity로 붕괴(무해성)함도 함께 제시한다.

- **Technical Challenges**: 가장 큰 기술적 도전은 (a) fine-tuning 후에도 남는 stylometric surface signal을 어떻게 ‘안전하게’ 결합할지, (b) 특히 skewed 라벨 분포에서 feature standardization이 gate를 편향시키지 않게 할지, (c) attention이 외부 lexical knowledge에 조건화되도록 만들지에 있다. SURGeLLM은 ReLU+sigmoid 기반의 per-dimension gate로 modality 혼합을 유연하게 하고, [CLS] fusion 이후 LayerNorm으로 분포 이동을 안정화한다. 동시에 IWN은 학습 단계에서 클래스 균형 통계로 표준화를 수행해 테스트 시 label 없이도 class-prior 편향을 교정하며, gate와 prefix를 같은 역할로 대체하지 않고 보완적으로 작동하게 설계했다.

- **Empirical Impact**: 4개 태스크(SST-2, multi-hop retrieval, LLM-prompt attribution, authorship detection)에서 17,830개 예제와 11개 모델 변형을 3개 시드로 검증한 결과, IWN이 포함된 SURGeLLM-IWN-RoBERTa가 전체 macro-F1 0.940을 달성했다. 이는 가장 강한 non-IWN 대비 +0.036, authorship detection에서 +0.130의 큰 향상을 포함하며, random-vocabulary 제어 실험에서는 평균 F1이 -0.028로 떨어져 성능이 ‘파라미터 증대’가 아닌 lexical feature의 효과임을 확인했다. 저자들은 또한 코드·어휘(vocabularies)와 99.5% 수준으로 curated feature를 자동 복구하는 절차를 공개해, 다른 도메인으로의 확장성을 시사한다.



### Decoherence as Defence and the Magnitude of Noise Regularisation: A Rigorous N -Qubit Theory of Stochastic Quantum Neural Networks for Adversarially Robust Network Intrusion Detection (https://arxiv.org/abs/2606.24219)
- **Prior Approaches**: 기존 SQNN 연구는 뉴런 활성은 qubit으로, 시냅스 토폴로지는 entanglement로, 신경 잡음은 Lindblad master equation으로 인코딩해 노이즈-기반 학습을 시도해왔다. 한 컨퍼런스 선행연구는 ring-entangled SQNN을 침입 탐지에 적용해 ring entanglement가 비국소 이상 탐지에 필수라는 점을 보였고, 동시에 depolarising channel이 dropout-style 정규화 역할을 못 하며 출력 잡음처럼 작동한다고 지적했다. 또한 강건성에 대한 bound는 존재했지만 보수적이어서, true quantum dropout(게이트 단위 확률적 비활성화)가 같은 문제를 어떻게 해결할지, 그리고 느슨한 bound를 예측 가능한 이론으로 바꿀 수 있는지는 미해결로 남았다.

- **Core Contribution**: 이 논문은 per-gate stochastic deactivation(“true quantum dropout”)과 depolarising noise를 각각 정식화해, 침입 탐지 및 일반화 관점에서 무엇이 “정규화”로 작동하는지 분해한다. 구체적으로 N-qubit 확장을 stochastic master equation과 vectorised Liouvillian으로 제시하고, depolarising channel이 weight-ratio Pauli read-out을 어떤 방식으로 수축시키는지 “decoherence-contraction theorem”으로 증명한다. 나아가 Du et al.의 noise-as-defence 아이디어를 정량화·운용 가능 형태로 만들어, robustness bound의 보수성을 줄이고 공격 하에서도 왜 붕괴가 일어나지 않는지 이론-실험을 연결한다.

- **Technical Challenges**: 핵심 기술적 난제는 두 가지다. 첫째, noise를 단순한 정규화 직관으로 다루지 않고, Lindblad 기반의 동역학을 N-qubit 수준에서 계산 가능한 형태(벡터화 Liouvillian)로 정리해 Pauli read-out의 “수축 법칙”을 얻어야 했다. 둘째, true quantum dropout이 weight 공간에서 어떤 패널티로 귀결되는지, depolarising noise는 출력 공간에서 어떤 패널티로 귀결되는지를 구분해 예측 가능한 공식으로 제공해야 했고, 이를 통해 dropout은 curvature-weighted L2 패널티(p(1-p)/2)∑θ^2 ∂^2_θ L 형태, depolarising은 output-space penalty로 정리했다.

- **Empirical Impact**: 실험에서는 real 데이터 NSL-KDD에서 white-box FGSM·PGD 공격을 걸어, depolarising SQNN(채널 포함 학습)이 noiseless circuit보다 강한 ℓ∞/ℓ2 조건 하에서 유의미하게 더 견고하며(예: ℓ∞ PGD-20에서 p=0.04), 특히 noiseless 모델과 gradient-trained 고전 탐지기가 겪는 catastrophic robustness collapse가 재현되지 않았다고 보고한다. 또한 7개 시드에 대해 robustness variance가 약 2배 줄어들었고, 이런 강건성은 공격 시점의 gradient contraction이 아니라 noise-reshaped training boundary에서 비롯된다고 분석한다. 일반화 측면에서는 30개 시드 연구로 dropout/depoloarising이 train-test gap을 각각 약 0.01 수준의 통계적으로 유의미하게 줄이되 둘의 효과가 서로 구별되지 않고, 오버피팅이 큰 구간에서 효과가 집중되며 dropout rate를 1/2를 넘으면 도움이 되지 않는다는 예측이 정량적으로 확인되었다.



### MMed-Bench-IR: A Heterogeneous Benchmark for Multilingual Medical Information Retrieva (https://arxiv.org/abs/2606.24200)
Comments:
          Under review. 15 pages, 3 figures

- **Prior Approaches**: 기존 연구는 다국어 검색과 의료 특화 검색을 각각 따로 다루거나, 둘의 교차 지점을 부분적으로만 측정했다. 예를 들어 의료 biomedical encoders는 영어 위주로 평가됐고, 다국어 벤치마크는 의료 개념(ontology·UMLS) 근거가 약해 실제 임상 지식 매칭의 품질을 검증하기 어려웠다. 또한 관련 벤치마크들은 정렬(alignment), 개념 구분(discrimination), 근거 검색(evidence retrieval)을 ‘동시에’ 보지 않아 한 축의 개선이 다른 축을 해치지 않는지 확인이 불가능했다.

- **Core Contribution**: 논문은 다국어 의료 검색의 세 가지 역량을 한 프레임에서 분해·동시에 평가하는 MMed-Bench-IR을 제안한다. UMLS에 근거한 cross-lingual medical QA retrieval, UMLS 기반 confusion set으로 개념 구분을 측정하는 discrimination, 그리고 RAG용 multilingual evidence retrieval까지 총 3개 태스크를 6개 언어(영·스·프·일·중·러)·3개 문자 체계로 구성했다. 특히 세 태스크는 설계상 query와 concept 중복이 없어, 특정 기술만 잘하는 ‘편향된 합산’이 아니라 실제 capability breadth를 반영한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 언어 간 개념 정렬, (2) 임상적으로 헷갈리는 개념의 미세 구분, (3) 영어 중심 코퍼스에서 다국어 질의 근거를 찾아내는 evidencing을 동시에 측정하는 평가 설계에 있었다. 이를 위해 UMLS CUI 태깅으로 양성군을 구성하고, discrimination의 난이도는 ontology 구조(동의어-유사하지만 다른 개념-더 먼 관련)로 고정했으며, RAG 태스크는 번역 품질을 concept fidelity와 back-translation 일치도로 필터링해 83.7%를 유지했다. 또한 UMLS 기반 tier 정의가 특정 검증자에 치우치지 않는지(leave-one-validator-out) 안정성을 점검하고, 태스크 간 어휘·질의 중복을 극도로 낮춰 ‘평가 눈속임’을 줄였다.

- **Empirical Impact**: 10개 시스템(lexical·biomedical·multilingual dense·late-interaction·hybrid·reranker 계열)을 평가한 결과, 영어에서 잘하는 모델이 일본어 등 비(非)영문 스크립트에서 급격히 붕괴하는 현상이 뚜렷하게 드러났다. 예컨대 biomedical encoder SapBERT는 영어 nDCG@10이 0.818에서 일본어 0.056으로 급락했고, 공정성 격차(fairness gap)는 0.76으로 보고됐다(영어-only 벤치마크로는 포착 불가). 한편 concept discrimination이 가장 어려운 병목으로 나타났고, cross-encoder reranking 및 도메인·언어 특화 임베딩(MMed-Embed) 조합이 최상위 성능(0.377)과 상대적으로 낮은 공정성 격차(0.170)를 달성해, 다국어 임상 동등성 관점의 검색 개발 방향을 제시한다.



### Aspect-Based Sentiment Evolution and its Correlation with Review Rounds in Multi-Round Peer Reviews: A Deep Learning Approach (https://arxiv.org/abs/2606.24188)
- **Prior Approaches**: 기존 연구는 동료평(리뷰) 코멘트의 감정을 주로 거칠게 다루거나, 여러 라운드를 거치며 변화하는 차이를 충분히 구분하지 못했다. 그 결과 리뷰 라운드 전환에 따라 리뷰어가 주목하는 이슈와 감정 성향이 어떻게 이동하는지 분석이 제한적이었다.

- **Core Contribution**: 본 연구는 다중 라운드 리뷰 텍스트에서 aspect-level(항목 수준) 감정의 분포와 진화를 추적하고, 그것이 리뷰 라운드 수와 어떤 상관을 갖는지 체계적으로 분석한다. Nature Communications에 수록된 11,063편의 accepted 논문을 대상으로, fine-grained review aspect clusters(세분화된 리뷰 항목 클러스터)를 식별하고 이를 기반으로 감정 변화를 정량화한다.

- **Technical Challenges**: 핵심 과제는 리뷰 문장을 항목별로 나누어 감정을 정확히 분류하는 동시에, 라운드별 동적 변화를 반영할 수 있는 데이터와 모델링이 필요하다는 점이다. 약 5,000개 리뷰 문장에 대해 수작업으로 라벨링한 코퍼스를 구축해 deep learning 기반 aspect sentiment classification을 학습했으며, 그중 LCF-BERT-CDM이 Macro-F1 82.65%로 최고 성능을 보였다.

- **Empirical Impact**: 통계 분석 결과 리뷰 라운드 수가 늘어날수록 긍정 감정의 비중은 증가하고 부정 감정은 감소하는 일관된 경향이 관찰됐다. 상관 분석에서는 aspect sentiment 점수가 리뷰 라운드 수와 부(-)의 관계를 보였고, 특히 ‘experiments’, ‘research significance’, ‘result analysis’ 항목이 더 강한 연관성을 보이며 과학 평가 과정의 동학을 이해하는 데 의미가 크다.



### A Synthetic Reliability-Aware PINN Benchmark for Offshore Wind Turbine Support-Structure Monitoring with Bayesian Inverse Identification (https://arxiv.org/abs/2606.24176)
Comments:
          18 Pages, 8 Figures

- **Prior Approaches**: OWT(해상풍력) 단일말뚝 지지구조의 SHM은 가속도계·변형률계 같은 희소 센서로부터 실시간 상태를 추정해야 하지만, 고정밀 FEA/에어로엘라스틱 해석을 온라인 루프에 반복 적용하기는 어렵다. PINN은 PDE를 손실에 반영해 적은 데이터로도 대체가 가능하지만, 역문제(inverse PINN)는 초기값이 멀 때 수렴 실패 메커니즘이 체계적으로 규명되지 않았다. 또한 OWT 디지털 트윈에서 신뢰도 지표를 반복 갱신하려면 몬테카를로가 느리고, FORM을 PINN 기반 상태추정과 통합한 사례는 리뷰에서 확인되지 않았다.

- **Core Contribution**: 이 논문은 Digi Turbine이라는 합성 기반 신뢰도-aware PINN 벤치마크를 제안해, OWT 단일말뚝의 희소 측정 기반 빠른 상태추정과 확률적 신뢰도 스크리닝을 하나의 파이프라인으로 묶는다. 학습 목적함수에 Winkler 토양 기초가 포함된 간소화 Euler–Bernoulli 빔 방정식을 포함하고, 역매개변수 식별에는 Bayesian-prior-informed 접근과 FORM screening 레이어를 결합한다. 검증은 NREL 5MW 맥락을 물리적 기준으로 두되, 해석해/유한차분 ground truth로 합성 실험을 수행한다.

- **Technical Challenges**: 핵심 기술난점은 역문제 PINN의 'gradient direction problem'로, 고차 PDE에서 미지 매개변수(E, ksoil)에 대한 PDE 기울기와 데이터 적합 기울기가 지속적으로 반대 방향이 되어 수렴이 막히는 현상을 분석했다. 저자들은 이 원인으로 (1) 변위-only 데이터가 강성에 필요한 고차 미분을 단독으로 충분히 제약하지 못하고 (2) 네트워크 가중치가 그 미분을 왜곡한 상태에서 매개변수 업데이트가 오도된다고 설명한다. 해결책으로는 Bayesian prior를 정확한 설계/시공값 주변의 약한 범위로 배치해 탐색을 물리적으로 그럴듯한 영역에 고정하고, EMA-adaptive loss weighting과 동시 co-evolution 학습으로 구성 요소 간 스케일 불균형도 완화한다.

- **Empirical Impact**: 제안된 Forward PINN은 희소 측정 50–100개만으로도 10/10 테스트에서 구조 변위를 재구성하며, GPU 기준 평균 추론 속도는 약 0.381ms(완전 구성이며 CPU는 0.605ms)로 10ms 목표를 크게 하회한다. 역문제 식별에서는 Bayesian prior가 있을 때 8/8 성공과 동시 E+ksoil 식별(잡음 10% 포함)에서도 식별 오차가 매우 낮게 유지되며, FORM은 선형 limit state에 대해 기계정밀 수준 정확도를 보인다. 전체 end-to-end 온라인 파이프라인은 <7ms로 실행 가능해(온라인에서는 재학습 없이 추론만) 반복적인 신뢰도 업데이트가 가능한 실용적 경로를 제시한다.



### A Pāninian Foundation for Indic Language Processing (https://arxiv.org/abs/2606.24172)
Comments:
          16 pages, 0 figures

- **Prior Approaches**: 기존 NLP는 인도계 언어를 계통(가계)별 또는 소수 언어군 단위로 쪼개 분리된 분석기·파서·데이터셋을 구축해 왔다. 그 결과 언어 간 전이가 가능한 경우에도 벤치마크와 주석 체계가 달라 이동성이 떨어지고, 중복 공학과 자원 격차가 누적된다.
또한 다국어 대규모 모델은 번역 품질 등 ‘폭’ 중심 지표에 강하지만, 인도계 언어를 관통하는 형태·통사·의미 구조를 명시적으로 검증하는 시험 설계는 부족하다.

- **Core Contribution**: 이 논문은 인도계 언어들이 공통으로 공유하는 형태통사적 ‘파니니(Pāṇinian) 문법’ 기반 아키텍처(Aṣṭādhyāyī, Astādhyāyī)를 계산적으로 단일한 기반으로 제안한다. 언어 간 전이가 이미 Pāṇinian의 구조적 레일 위에서 일어나고 있지만, 현재 시스템은 표면 신호에만 의존해 그 깊이를 활용하지 못한다고 본다.
이에 따라 파니니 범주를 기준선으로 삼는 4-part 벤치마크 스위트를 제안하며, 이를 통해 정확도·데이터 효율·전이성을 함께 끌어올리고 희소한 인도계 자원을 ‘하나의 고자원 메타언어’처럼 묶는 것을 목표로 한다.

- **Technical Challenges**: 핵심 난점은 인도계 언어가 형태적으로 매우 풍부해 단어가 뿌리와 접사로 조합되며 격·수·시제·상·태 같은 정보가 복합적으로 얽힌다는 점이다. 특히 sandhi(음운 결합)와 samāsa(복합어) 같은 현상은 문장을 의미 단위로 분해하는 단계부터 어려워, 이를 파니니적 구조로 중심화하지 않으면 대부분의 파이프라인이 모서리 케이스로 취급하게 된다.
또한 문어·구어가 함께 존재하는 diglossia와 코드믹싱, 그리고 의미가 명사-고정 관점이 아니라 사건 구조에서 동사 뿌리를 중심으로 ‘과정적’으로 조직된다는 의미론적 차이가 있어, 기존 영어 기반 평가 프레임이 구조를 제대로 측정하지 못한다.

- **Empirical Impact**: 저자들은 파니니 원리에 기반한 형태 분석이 인도계 언어 간 전이와 다운스트림 성능을 실제로 개선해 왔다는 선행 결과들을 근거로 들며, 공통 아키텍처를 명시화하면 추가 이득이 클 것이라고 주장한다. 나아가 현재의 벤치마크가 대부분 표면 신호(어휘/문자 체계) 중심으로 작동해 문법적 이해를 드러내지 못한다는 문제를 지적한다.
제안된 파니니-근거 벤치마크는 인도계 시스템을 더 데이터 효율적으로 만들고 언어 간 전이를 ‘설계된 메커니즘’으로 전환하는 한편, 신경망이 실제로 파니니 범주를 자발적으로 표상하는지(해석 가능성)까지 실험 가능하게 할 의미가 있다.



### BehaviorBench: Benchmarking Foundation Models for Behavioral Science Tasks (https://arxiv.org/abs/2606.24162)
- **Prior Approaches**: 기존에는 설문 응답 예측이나 특정 게임 기반 의사결정 등, 행동과학의 부분 능력만을 겨냥한 벤치마크가 주로 사용됐다. 또한 많은 평가가 개인을 독립 데이터로 보고 pointwise accuracy 같은 단일 점수로 모델을 채점해, 집단의 다양성과 이질성을 보존하는지(분포적 정렬)는 충분히 다루지 못했다. 그 결과 잠재 성향 추론이나 행동지식 적용 같은 핵심 역량이 체계적으로 측정되지 않는 공백이 있었다.

- **Core Contribution**: 이 논문은 foundation model을 행동과학 태스크에 대해 네 가지 역량—(1) 행동 예측·시뮬레이션, (2) 전략적 의사결정, (3) subject-trait inference, (4) behavioral knowledge 적용—으로 정리해 포괄적으로 평가하는 BehaviorBench를 제안한다. 특히 모델 출력의 평가를 개인 수준뿐 아니라 분포 수준까지 함께 수행해, 행동 유효성에 필수적인 ‘인구 수준 정렬’을 1순위 목표로 만든 점이 핵심 기여다. 아울러 BehaviorBench 태스크 포뮬레이션을 따라 행동 기반 데이터로 fine-tuning한 Be.FM-1.5도 함께 제시한다.

- **Technical Challenges**: 주요 기술적 난제는 행동과학의 입력 변수(개인 특성 x, 맥락 c, 행동 y, 그리고 지식 K)를 모두 반영하면서도, 다양한 데이터원(실험·설문·문헌)에서 일관된 태스크/평가 프레임으로 모델을 비교하는 데 있었다. 논문은 이를 위해 행동을 p(y|x,c,K) 관점에서 네 역량으로 재구성하고, 예측·추론·지식 적용을 각각 다른 생성/역생성 형태로 설계했다. 또한 분포적 평가를 위해 Wasserstein distance 같은 지표를 도입해 집단 분포의 형태와 평균 정렬을 함께 측정하고, 데이터 겹침 없이 Be.FM-1.5를 행동 태스크로 광범위 SFT(LoRA) 수행해 성능 격차를 줄이려 했다.

- **Empirical Impact**: 실험 결과, 일반 목적의 proprietary 모델은 개인 단위 예측과 지식집약형 태스크에서 강점이 있지만 분포적 지표에서는 뒤처지는 경향이 관찰됐다. 반면 행동 데이터로 fine-tuning된 behavioral foundation model은 평균적으로 분포적 정렬이 더 강했고, 그중 Be.FM-1.5는 분포 수준에서 선두권 성능을 보이면서도 개인 수준 지표에서 경쟁력을 유지했다. 저자들은 이러한 결과가 BehaviorBench의 ‘분포 평가’ 중요성을 뒷받침하며, 행동과학 연구 전반에서 정렬된 AI 시스템 개발과 검증의 기반이 될 수 있음을 보여준다고 정리한다.



### MedBench v5: A Dynamic, Process-Oriented, and Hallucination-Aware Benchmark for Clinical Multimodal Models (https://arxiv.org/abs/2606.24155)
- **Prior Approaches**: 기존 의료 AI 벤치마크는 MedQA/CMExam처럼 정적 단일턴 QA에 치우쳐 있어, 실제 임상처럼 불확실성 하에서 누락 정보를 적극 탐색하고 증거를 갱신하는 과정을 충분히 드러내기 어렵다. 또한 Med-HALT/MedHallu 등은 환각을 주로 최종 출력의 사실성 문제로만 다루는 경향이 있어, 환각이 턴마다 어떻게 발생·전파되는지 추적이 제한적이었다. 결과적으로 성능 저하가 관찰되더라도 “어느 추론 단계에서 왜 깨졌는지”를 진단하기가 어려웠다.

- **Core Contribution**: 이 논문은 임상 멀티모달 모델을 대상으로 정적 QA를 넘어 동적 과정(process) 중심으로 평가하는 MedBench v5를 제안한다. MedBench v5는 Clinical Cognitive Responsiveness(14개 세부 차원)와 Medical Atomic Skills(4개 실행 에이전트 환경)을 결합해 총 63개 태스크를 포괄하면서, 모델의 실패 지점을 단계별로 분해해 보여주도록 설계됐다. 더불어 정보 흐름 교란(omission/contradiction/evidence delay)과 추론 프로세스 감사, 환각 전파 모니터링을 통합해 ‘정답 유무’와 ‘과정 안정성’을 함께 측정한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 멀티턴 임상 시나리오에서 실패가 생기는 추론 노드를 세분화해 로컬라이즈하고, (2) 환각이 언제 시작해 어떤 경로로 증거 체인에 고착되는지를 정량화하는 것이다. 이를 위해 3가지 switchable 정보흐름 스트레서로 상황을 통제한 뒤, 5개의 reasoning node(정보공백 탐지, 후속질문, 모순 탐지, 진단 갱신, 근거 정합)를 따라 행동 궤적을 감사하며, 환각은 initiation→propagation→anchoring→hallucination–contradiction interaction의 4축으로 전파 단계를 추적한다. 또한 단순 최종답 스코어 대신 시나리오 메타데이터와 판정(judge) 모델을 이용해 “모델이 관찰 가능한 정보로부터 벗어났는지”를 턴별로 판별하도록 구성했다.

- **Empirical Impact**: 실험에서는 frontier 모델들이 전체 태스크 정확도가 높더라도 과정 안정성(process stability)은 보장되지 않음을 보여줬다. 특히 스트레서가 모순 탐지, 진단 업데이트, 환각 전파, 모순 기반 self-correction을 주로 교란했으며, 최종 근거 grounding은 겉보기엔 안정적으로 유지될 수 있다는 점이 관찰됐다. 이는 MedBench v5가 임상 준비도 평가에서 “정답만이 아니라 실패가 발생하는 경로”와 “환각 궤적”을 드러내는 진단형 인프라로 의미가 있음을 시사한다.



### Metis: Bridging Text and Code Memory for Self-Evolving Agents (https://arxiv.org/abs/2606.24151)
Comments:
          Work in progress

- **Prior Approaches**: 자기진화 에이전트는 이전 실행 경험을 메모리에 저장해 이후 태스크에 재사용한다. 그런데 경험을 자연어 텍스트로 넣을지, 호출 가능한 코드 툴로 둘지는 보통 설계 시점에 고정돼 왔고, 두 표현이 어떤 종류의 경험에 어떤 비용-효율-전이 특성을 보이는지 체계적으로 비교·이해되지 않았다. 그 결과 텍스트 메모리의 견고함과 코드 메모리의 실행 효율 간의 트레이드오프를 데이터 기반으로 조합하는 전략이 부족했다.

- **Core Contribution**: 이 논문은 동일한 경험 집합과 동일한 에이전트 백본 위에서 텍스트 메모리와 코드 메모리를 분리해 평가한 ‘첫 controlled study’를 제시한다. 그 관찰을 바탕으로 Metis는 계층형 dual-representation memory를 설계해, 텍스트(계획/사실/함정)를 기반으로 하되 반복되는 실행 계획만 선별적으로 callable tools로 crystallize한다. facts와 pitfalls은 텍스트로 남겨 추론 가이드는 유지하면서 코드의 비용·취약성을 불필요하게 확산하지 않도록 한다.

- **Technical Challenges**: 핵심 난제는 (1) 코드 툴 생성이 비싸고 검증·디버깅 비용이 크며 (2) 코드는 고정된 실행 동작이라 태스크 변형에 취약하다는 점을 어떻게 제어하느냐에 있다. Metis는 실행 후 반영(reflect)에서 계획/사실/함정의 역할을 분리해 저장하고, 코드 승격(promotion)은 ‘얼마나 자주/안정적으로 선택됐는지’의 recurrence 증거로만 트리거한다. 또한 툴 생성 시에는 sandbox 검증, dependency-closure, compilation check를 통과한 경우에만 라이브러리에 추가하고, 텍스트 반영은 실행 임계 경로에서 비동기로 수행해 지연을 최소화한다.

- **Empirical Impact**: AppWorld에서 Metis는 ReAct 대비 작업 정확도를 최대 20.6%까지 개선하면서 실행 비용은 최대 22.8%까지 줄였다고 보고한다. 비교하는 자기진화 계열 시스템들에 대해서도 정확도·실행 효율·메모리 구성 비용 간 균형을 더 잘 맞추는 경향이 일관되게 관찰된다. 즉, 경험 표현을 ‘무조건 텍스트’ 또는 ‘무조건 코드’로 고정하기보다, 경험 특성에 따라 선택적으로 전환하는 접근이 실용적 성능을 만든다는 점을 실험적으로 뒷받침한다.



### PORTER: Language-Grounded Event Representations for Portable Structured EHR Foundation Models (https://arxiv.org/abs/2606.24102)
- **Prior Approaches**: 기존 구조화 EHR foundation model은 고정된 이벤트 토큰 어휘(vocabulary)에 의존해 임상 이벤트를 토큰으로 인코딩한다. 그래서 기관 간/파이프라인 간에서 코드·명칭·속성 조합이 바뀌면 모델이 보지 못한 토큰이 대거 발생하고 의미 전이가 깨진다. 텍스트 직렬화는 어휘 의존을 줄이지만 과제별로 표현을 다시 계산해야 하고, 수치 값은 보통 버리거나 텍스트로 처리해 정밀한 크기 감도를 얻기 어렵다는 한계가 있었다.

- **Core Contribution**: PORTER는 언어 기반(event-description) 구조화 EHR foundation model로, 이벤트 의미(개념)·수치값·시간 역학을 분리해 고정 토큰 어휘에서 해방시키는 것을 목표로 한다. 냉동(frozen) 텍스트 인코더로 이벤트 설명을 임베딩해 어휘 밖 개념도 표현 가능하게 하고, 숫자는 전용 pathway와 FiLM으로 통합해 개념 정체성과 크기 정보를 함께 반영한다. 또 자가회귀 방식의 temporal backbone을 학습한 뒤 고정해 여러 다운스트림 과제에 재사용(선형 프로브)하도록 설계했다.

- **Technical Challenges**: 핵심 과제는 (1) 새로운 개념/조합이 등장할 때도 임상 의미를 안정적으로 전달하고, (2) 수치의 크기·이상(abnormality)을 텍스트화 없이 구조적으로 주입하며, (3) 재사용 가능한 시간적 표현을 자가회귀 사전학습으로 얻는 것이다. PORTER는 이벤트 설명을 한 번만 인코딩해 캐싱 가능한 vocabulary-independent concept representation을 만들고, 숫자는 기준범위가 있으면 구간 내 위치를 스케일링한 뒤 FiLM으로 텍스트 임베딩을 조절한다. 또한 backbone은 사전학습 후 동결하고, 입력은 이벤트 단위 표현만으로 구성해 과제별 재직렬화 부담을 줄였다.

- **Empirical Impact**: SickKids의 74개 임상 예측 과제에서 PORTER는 동일 temporal backbone/사전학습 목적을 쓰는 고정 어휘 모델과 AUROC 평균을 맞췄다. 특히 같은 환자 타임라인을 사전학습에 없던 이벤트 설명 체계로 렌더링했을 때 재학습 없이 이전되어, 타깃 어휘로 직접 학습한 기준의 평균 AUROC 대비 97.1%를 회복했고, MIMIC 전이에서는 고정 어휘 모델이 토큰 미관측으로 69% 이벤트를 드롭한 반면 PORTER는 이를 피하며 성능을 개선했다. 기계적 분석에서는 어휘가 바뀌어도 환자 표현 기하(geometry)가 보존되는 경향과, numeric pathway가 민감도는 높이되 임상 개념 정체성은 훼손하지 않는 점이 확인됐으며, task-specific 텍스트 비교자 대비 더 높은 AUROC를 더 낮은 amortized compute로 달성했다.



### Predicting Poets' Origins from Verse: A Computational Analysis of Regional Linguistic Fingerprints in the Complete Tang Poems (https://arxiv.org/abs/2606.24093)
- **Prior Approaches**: 당대 시문에서 지역적 전통(지역 학교, 로컬 문체)이 실제 언어 차이를 남기는지에 대해 문학사에서는 오랫동안 논쟁이 있었지만, 이를 체계적으로 정량화한 연구는 제한적이었다. 기존 연구는 주로 면밀한 독해에 의존하거나, 구어 방언 분석에서 쓰이던 방식이 역사 문학 코퍼스에 그대로 적용되기 어려운 문제(라벨 품질, 문체 변동 등)로 인해 진전이 더뎠다. 이에 따라 “지리적 기원=예측 가능한 언어 흔적”이라는 가설을 데이터 기반으로 검증하는 접근은 여전히 도전적 과제로 남아 있었다.

- **Core Contribution**: 본 연구는 당(唐) 시인들의 지리적 기원을 출신 행정구역(circuit, 道)과 연결해 “시집 전체 텍스트만으로 시인의 출신을 분류할 수 있는가”를 직접 실험한다. Complete Tang Poems의 시를 시인 단위로 집계하고, CBDB의 道 정보를 목표로 10개 회로(그리고 남/북 이분류)에서 multi-class classification을 수행해 예측 가능성을 정량화했다. 나아가 단순 정확도보다 ‘무엇이 신호를 담는가(이미지, 계절/시간, 인유 등)’와 ‘신호가 시대에 따라 어떻게 변하는가’를 함께 분석했다.

- **Technical Challenges**: 핵심 기술 과제는 역사 문학 코퍼스에서 라벨(지리/귀속) 잡음이 크고, 시인은 “한 편”이 아니라 “여러 편의 집합”으로 특징을 대표해야 한다는 점이었다. 연구진은 시인을 단일 샘플로 보고 character n-gram TF-IDF와 이미지·계절/시간·인유 밀도 같은 해석 가능한 도메인 특징을 함께 사용했으며, 불균형 데이터를 stratified cross-validation과 class weighting으로 보정했다. 또한 GuwenBERT를 단편(250자) 평균으로 학습하면 불리해지는 문제를 계층형 frozen-encoder(단편→시인 벡터)로 해결해 TF-IDF 기반 고전 모델과의 공정 비교를 가능하게 했다.

- **Empirical Impact**: 실험 결과 시인의 남/북(南/北) 기원은 0.69 정확도와 매크로-F1으로 다수 기준선(0.53)을 뚜렷이 상회했고, 더 세분된 道 수준에서도 우연을 넘어선 성능을 보였다. 언어-지리 거리는 회로 간 거리와 함께 증가(거리감쇠 효과; Mantel r=0.40, p≈0.09)하며, High Tang에서는 남/북 분리가 약하고 Late Tang에서 가장 강해 “궁정 중심 동질화→시대 후반 지역 분기” 패턴을 정량적으로 뒷받침한다. 특히 모델의 자신 있는 오분류가 초기 당의 남방 시인을 북방 궁정 문체로 읽는 형태로 나타나, 오류 자체가 역사적 해석 가설을 제공할 수 있음을 보여주었고(인간-모델 협업 관점), 또한 TF-IDF의 character n-gram이 이미 지역 신호를 충분히 포착해 BERT 결합이 추가 이득이 없다는 점을 시사한다.



### CAVEWOMAN: How Large Language Models Behave Under Linguistic Input and Output Compression (https://arxiv.org/abs/2606.24083)
- **Prior Approaches**: 기존 LLM 압축 연구는 입력 압축이나 출력 압축을 한쪽만 다루거나, 토큰을 줄였을 때의 task accuracy만 주로 보고해 왔습니다. 하지만 압축 효과는 ‘실제 비용(입력/출력 채널별 토큰 단가)’과 ‘모델이 압축 전 기준으로 동일 내용을 말하는지’를 분리해 검증하기 어렵습니다. 그 결과, 짧아진 프롬프트가 비용을 정말 줄였는지와, 압축된 답이 모델 내부 추론 기준과 같은 텍스트 의미를 유지하는지 평가가 빈약했습니다.

- **Core Contribution**: 이 논문은 Cavewoman이라는 두 채널 평가 프로토콜을 제안해, 입력 압축과 출력 압축을 같은 아이템에서 동시에 비교합니다. 각 생성물에 대해 (1) 정답 정확도, (2) priced channel 기준 realized per-item cost, (3) 모델의 무제약 기준(L0) 생성과의 reference-text agreement을 함께 측정합니다. 또한 POS 기반 5단계 reduction(L0~L4)과 8개 모델·5개 벤치마크 전체 커버리지를 통해 채널별 비용/품질 분리를 정량화합니다.

- **Technical Challenges**: 핵심 난제는 ‘토큰 절감’이 실제 비용 절감으로 이어지는지, 그리고 정확도만으로는 기준 생성과의 텍스트 의미 일치를 구분할 수 없다는 점이었습니다. 이를 위해 모델 출력의 비용을 실제 priced 채널 토큰으로 산정하고, 정확도 평가 전에 answer-extraction rate를 감사해 파서/추출 편향을 통제했습니다. 더 나아가 reference-text agreement는 bidirectional NLI entailment과 11개 보완적 semantic 지표로 재검증해, 표면 텍스트 발화가 기준 생성과 어긋나는 현상을 측정합니다.

- **Empirical Impact**: 실험 결과, 출력 압축은 대부분의 API 모델에서 realized cost를 1.4~2.4x(최선 3x) 줄였지만 입력 압축은 반대로 순비용을 키웠습니다(평균 약 1.15x, 최악 최대 1.8x, 강한 압축에선 약 2.7x로 확대). 동일한 비용 절감 설정에서 비추론(non-reasoning) 6개 모델 묶음의 51.9%는 정답은 맞았지만 무제약 기준과의 텍스트 의미 포함관계를 더 이상 만족하지 못했습니다. 즉, 단일 축(accuracy/토큰) 압축 평가는 배치·감사·로그 소비 관점에서 오판할 수 있으며, 실제 배포에서는 constraint(출력 제약) 단계에서 후보를 순위 매겨야 한다는 실무적 시사점을 제공합니다.



### Sentence-Level Contextual Entrainment in Large Language Models (https://arxiv.org/abs/2606.24077)
Comments:
          16 pages, 3 figures

- **Prior Approaches**: in-context learning(ICL)은 프롬프트의 시연 예시를 파라미터 업데이트 없이 활용한다는 점에서 강력하지만, 모델이 맥락을 어떻게 “오남용”하는지는 덜 알려져 있다. 최근 Niu et al.(2025)의 contextual entrainment는 문맥에 등장한 토큰의 다음-토큰 확률을 체계적으로 끌어올리는 현상을 밝혔지만, 분석 단위가 토큰 수준에 머물러 실제 생성에서의 누적 효과를 충분히 반영하지 못한다. 또한 기존 마스킹 분석은 제한된 모델/설정에 집중되었고, 측정이 주로 객관적 회상(factual recall)에 편중돼 주관적 과제까지의 일반성은 불명확했다.

- **Core Contribution**: 이 논문은 contextual entrainment를 토큰에서 “문장(sentence) 수준”으로 확장한다. 각 응답 문장의 per-token mean log-probability를 이용해, 프롬프트에 포함된 문장이 다음 생성 후보로서 확률이 얼마나 오르는지를 정량화한다. 26개 LLM(7개 패밀리)과 LRE(객관)·WVS(주관) 두 데이터셋에서 sentence-level contextual entrainment가 존재함을 보이며, 심지어 프롬프트 문장이 반사실(counterfactual)이어도 해당 문장 확률이 유의미하게 증가할 수 있음을 보여준다. 나아가 2%~4%의 attention heads만 끄더라도 성능을 해치지 않으면서 entrainment를 완화할 수 있음을 제시한다.

- **Technical Challenges**: 문장 수준으로 확장하려면 토큰별 확률 증가를 문장 전체의 누적으로 일관되게 연결하는 측정 설계가 필요하다. 논문은 토큰 시퀀스의 chain rule을 바탕으로, 문장 생성 시 로그확률 변화가 문장 내 토큰 변화들의 합으로 해석되도록 정의해(응답 문장의 per-token mean log-probability) sentence-level 지표를 구성한다. 또한 어떤 attention heads가 해당 현상을 만드는지 찾기 위해, differentiable attention head masking을 문장 수준 학습 목적에 맞게 확장해(게이트를 통해 헤드 출력 기여를 선택적으로 제거) 반복 실험과 head 공유 분석(shared heads)까지 수행한다. 마지막으로 반사실 문맥에서의 확률 왜곡이 실제 성능(정확도/극성 일관성) 변화로 연결되는지까지 평가 체계를 함께 구성한다.

- **Empirical Impact**: 실험 결과, 문맥에 포함된 문장은 정답과 직접 무관하거나 반사실로 구성돼도 모델의 응답 후보 확률을 끌어올리는 sentence-level contextual entrainment가 관찰된다. 모델 크기가 커질수록 contextual entrainment는 점진적으로 감소하지만, 문맥에 나타나지 않는 다른 응답 쪽으로의 distraction 영향은 오히려 커지는 경향이 보고된다. attention head 관점에서는 relation별로 필요한 헤드가 존재하는 동시에, 여러 relation에 공통으로 나타나는 sparse shared heads가 확인되며 그 일부(전체의 2%~4%)만 비활성화해도 entrainment를 효과적으로 완화할 수 있다. 이 결과는 ICL 성능 저하의 원인을 “맥락 토큰/문장 자체에 대한 확률 편향”으로 구체화하고, 최소한의 구조 수정으로 완화하는 실용적 대응책을 제안한다.



### Selective Capability Unlearning in End-to-End Spoken Language Understanding (https://arxiv.org/abs/2606.24063)
Comments:
          5 pages, 3 figures, preprint

- **Prior Approaches**: 기존 SLU unlearning은 주로 특정 intent의 사후확률 p(i|x)를 억제하는 방식(예: Gradient Ascent, NPO, KL 정규화, random label)을 사용한다. 하지만 autoregressive SLU에서는 intent 접두사가 강제로 주어지면 slot 생성은 여전히 그 intent prefix에 조건부로 복원될 수 있다. 이 때문에 intent 정확도는 떨어져도 forced-prefix 조건에서 slot recoverability가 남는 ‘capability persistence’ 문제가 발생한다.

- **Core Contribution**: 이 논문은 SLU에서 제거해야 할 ‘기능’을 intent 레이블이 아니라 intent-조건부 slot 생성의 conditional mapping으로 정의하고, 이를 제거하지 못하는 구조적 실패를 capability persistence로 명확히 규정한다. 그 해결책으로 BSU(Binding Subspace Unlearning)를 제안해, target intent가 만드는 intent-slot 결합에 해당하는 representation subspace를 찾아 감쇠시키는 방식으로 conditional mapping 자체를 약화시킨다. 또한 beam 및 임베딩 기반 유사도 등을 활용한 recoverability 중심 평가 프로토콜을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘marginal intent 억제’만으로는 slot 생성의 conditional mapping이 남는 autoregressive 구조를, 실제로는 표현공간에서 끊어내는 것이다. BSU는 (1) forget/retain 데이터에서 slot 위치의 hidden state를 teacher-forcing으로 정렬한 뒤 층(layer)별 covariance contrast로 intent-조건부로 풍부해지는 방향(바인딩 subspace)을 eigen-direction으로 추출하고, (2) 해당 subspace로의 gradient sensitivity를 정규화해 teacher-forced conditional log-likelihood의 의존도를 줄인다.

- **Empirical Impact**: SLURP와 SpeechMASSIVE(프랑스어 subset)에서 BSU는 forced-prefix 상황에서의 conditional slot recoverability를 크게 낮추면서(retain 성능은 대체로 유지) 의도한 제거 효과를 보여준다. 예를 들어 SLURP NeMo 설정에서 BRR@10은 92.64→22.10, semantic similarity는 90.14→24.80으로 감소했으며, 다른 초기화/데이터셋에서도 유사한 하락 추세가 관찰된다. ablation과 정량 분석을 통해 무작위 표현 교란은 효과가 제한적인 반면, binding regularizer 강도를 키우면 forget 지표가 지속적으로 내려가고 retain은 안정적임을 확인했다.



### Best Preprocessing Techniques for Sentiment Analysis (https://arxiv.org/abs/2606.24055)
Comments:
          9 pages, 3 figures

- **Prior Approaches**: 기존 연구들은 감성 분류에서 전처리가 성능에 크게 영향을 준다는 점을 보여줬지만, 전처리 기법을 어떤 순서로 적용하는지에 대한 체계적 분석은 부족했다. 또한 결과가 알고리즘과 데이터 맥락에 따라 달라져 일관된 지침을 제공하기 어려웠다.

- **Core Contribution**: 본 논문은 트위터 감성 분석에서 전처리 기법들의 적용 순서를 제한된 조합으로 체계적으로 실험해, 실무자가 따라 할 “순서” 권고안을 제시한다. 특히 순서를 고려했을 때 철자 교정(spelling correction)은 영향이 가장 작고, tokenisation이 가장 큰 변화를 만든다는 결론을 도출했다.

- **Technical Challenges**: 전처리 조합의 경우의 수가 매우 커서 모든 순서를 전수 실험하기 어렵고, 각 기법이 토큰화 텍스트를 전제로 하는 등 논리적 제약도 존재했다. 저자들은 토큰화 이후 적용돼야 하는 단계들을 반영해 탐색 공간을 5!에서 크게 줄이고, 5-fold 교차검증과 ANOVA(F-statistic)로 각 단계의 효과 크기를 비교했다.

- **Empirical Impact**: 3개 트위터 데이터셋과 여러 분류기에서 성능을 평균 F1으로 평가한 결과, 전체적으로 best order는 tokenisation → cleaning → spelling correction → stop word removal → stemming 계열로 나타났다. ANOVA 기준으로 tokenisation의 설명력이 spelling correction보다 최소 한 자릿수 이상 컸으며, stemming과 stop-word removal은 대체로 상호 교환 가능하지만 stop-word removal에서 부정어(not, no)를 제거하지 않는 것이 중요하다고 정리했다.



### Towards Version-aware Operations and Transaction Memories for Multi-layer MeMo (https://arxiv.org/abs/2606.24040)
Comments:
          Accepted by MeMo Workshop on Mechanistic Interpretability & Neuro-symbolic Approaches by-design, Rome (Italy), 24/6/2026

- **Prior Approaches**: LLM 지식 갱신은 continued training, RAG(retrieval augmentation), parameter-level model editing 등으로 이뤄져 왔지만, 작은 변화에도 학습·대규모 재계산이 필요할 때 비효율적일 수 있습니다. 또한 ROME/MeMIT류는 업데이트가 “학습의 부작용”처럼 보일 수 있어, 무엇을 잊고 무엇을 남기는지에 대한 제어와 추적이 제한될 수 있습니다. MeMo는 correlation matrix memory를 외부화해 memorization/retrieval/forgetting을 명시적 연산으로 다룰 수 있다는 점에서 출발점이 됩니다.

- **Core Contribution**: 이 논문은 Multi-layer MeMo 위에 “버전 인지(version-aware) 연산 레이어”를 추가해, 모델 전체 재학습 없이도 명시적 memory association 기반의 지식 변경을 메모리 편집으로 반영하는 방식을 제안합니다. replace, obsolete, keep-history, rollback, trace 같은 고수준 연산을 MeMo 기본 편집들의 “순서가 있는 트랜잭션”으로 컴파일해 실행 가능하게 만듭니다. 이를 위해 Version CMM(V-CMM)과 Transaction CMM(T-CMM)을 도입해, 버전 전이→트랜잭션 핸들 매핑과 트랜잭션 내용의 재사용을 분리합니다.

- **Technical Challenges**: 핵심 난제는 버전 변경이 한 번의 MeMo association으로 끝나지 않고, 이전 연속체를 잊고 다른 연속체를 기억하며, 과거 체인을 보존하고 역연산(inverse program)을 기록하는 등 여러 primitive edits의 순차 실행이 필요하다는 점입니다. 논문은 next-token 기반 primitive (sequence, token) 편집으로 연속체(continuation) 업데이트를 분해하고, T-CMM이 템플릿/파라미터 또는 확장된 편집 시퀀스를 제공해 동일한 MeMo stack의 memo/forget/retrieve 프리미티브로 위임되도록 설계합니다. 또한 traceability와 rollback을 위해 트랜잭션을 역순 실행하거나 뷰/버전 alias 전환, 또는 exact log 재플레이 가능한 경로를 함께 제시합니다.

- **Empirical Impact**: 평가는 단계적 연구 로드맵으로 제안되며, toy 사실 편집과 ontology 스타일 diff(클래스 이동, 동의어 변경, obsolete 처리, 정의 업데이트 등)를 대상으로 업데이트 성공/기준선 유지(Outdated current suppression)/역추적·롤백 정확도/국소성(locality)/트랜잭션 재사용(reuse)을 지표로 삼습니다. 특히 trace는 현재 답에서 출발해 어떤 트랜잭션, 출처 버전, old/new value, 어떤 primitive edit들이 결과를 만들었는지를 메모리 수준 경로로 노출하는 해석가능성 기여로 정의됩니다. 제안 방식이 “MeMo-compatible memory edits로 표현 가능한 국소 변화”에서 점진적·가역적·추적 가능한 지식 업데이트를 실증하는 것이 의미 있는 목표입니다.



### Towards Spec Learning: Inference-Time Alignment from Preference Pairs (https://arxiv.org/abs/2606.24004)
- **Prior Approaches**: 기존에는 zero-shot prompting이나 in-context learning처럼 사용자가 프롬프트를 직접 설계·반복하며 성능을 끌어올리는 방식이 널리 쓰였다. 자동 프롬프트 최적화(APO)나 보상/보정 기반 강화학습·gradient 기반 업데이트, 그리고 DPO 같은 preference-based fine-tuning도 있었지만 대개 라벨/비교 데이터와 비용이 많이 들고, 프롬프트 엔지니어링은 오류에 취약하다는 한계가 지적된다. 또한 LLM-as-a-judge는 평가 확장성은 좋지만 길이·위치·모델 패밀리 편향과 분산 문제도 동반한다.

- **Core Contribution**: 이 논문은 spec learning이라는 프레임워크를 제안한다. 사용자는 짧은 지시와 소량의 preference judgment(대략 20개 쌍)만 제공하고, 그 정보를 자연어 시스템 프롬프트 형태의 specifications로 컴파일해 inference 시점에 조건부로 동작하게 만든다. 핵심은 LLM 가중치를 업데이트하지 않고도, 특정 도메인에서는 DPO(데이터 50배 규모의 선호 기반 학습)보다 더 자주 이기며, 동시에 사람이 읽고 수정 가능한 “명시적 정렬 산출물”을 제공한다는 점이다.

- **Technical Challenges**: 문제는 소량의 preference 쌍을 사람이 해석 가능한 소수의 규칙 원칙으로 압축할 수 있는가이며, 그 규칙을 실제로 선호 신호와 정렬되게 구성하는 컴파일 절차가 필요하다는 것이다. 연구진은 (1) proposer가 후보 principles를 생성하고, (2) semantic 클러스터링·중복 제거·검증을 거친 뒤, (3) prevalence와 정확도를 함께 고려해 순위화하고, (4) janus(서술형 정책) 또는 bullets(룰 리스트) 합성기를 통해 system prompt를 조립한다. 또한 judge 편향을 줄이기 위해 강건한 강제 이진 판정(포지션 로테이션/다중 패스)과 gold 기반 캘리브레이션을 사용하며, 필요 시 response 위치 스왑으로 편향을 완화한다.

- **Empirical Impact**: 7개 preference 데이터셋에서 compiled specification은 base 모델 대비 일관되게 우수했고, 특히 domain preference 신호가 규칙으로 잘 요약되는 경우 HH-Helpful 같은 이질적 데이터셋보다 격차가 크게 나타났다. 매크로 평균 win rate는 DPO 대비 모두 높은 결과를 보이며, spec은 DPO보다 훨씬 적은 preference 쌍(50배 적음)으로도 더 잘 맞는 것으로 보고된다. 추가 분석에서 컴파일된 principles를 준 judge가 선호 응답을 더 많이 “조건 충족”으로 분류해, 가중치 업데이트가 아닌 명시적 지침이 실제 preference signal을 담아낸 해석 가능 산물임을 시사한다.



### RASC+: Retrieval-Constrained LLM Adjudication for Clinical Value Set Authoring (https://arxiv.org/abs/2606.23992)
- **Prior Approaches**: 기존 Clinical value set authoring은 큰 코드 우주에서 해당 집합을 찾아내야 하며, 직접적인 코드 생성은 제약조건(버전관리·감사가능성) 때문에 잘 맞지 않는다. RASC는 retrieve-then-select로 무제약 생성의 위험을 줄였지만, stage-1 retrieval이 놓치면 stage-2에서 복구가 불가능하고, 단일 retriever 설계는 recall 병목을 만들 수 있었다.

- **Core Contribution**: 이 논문은 value set completion을 stage 1(후보 풀 구성)과 stage 2(후보 판정)로 분해하고, 각 단계의 목표를 따로 최적화한다. 특히 stage-2는 LLM이 코드를 새로 생성하지 않고 ‘후보 풀에서만’ 선택하도록 제한해 auditability를 유지하면서 precision을 끌어올리는 constrained LLM adjudication을 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) 타깃과 표현이 다르거나 publisher가 달라져도 누락 없이 높은 recall을 만들 후보 풀을 구성하고, (2) 커진 풀에서 LLM이 노이즈를 걸러내는 것이다. 이를 위해 Qwen3 기반 retrieval에 vocabulary-aware graph expansion, ICD-10-CM/RxNorm/SNOMED-CT별 구조적 확장, 그리고 code-display rescue retrieval(BM25)을 결합했으며, GPT-5 adjudication은 blinded 프로토콜로 VSAC의 누출 신호를 차단하고 JSON 스키마로 후보 ID만 반환하도록 설계했다.

- **Empirical Impact**: 결과로 stage-1 후보 풀 recall은 RASC baseline 0.553에서 0.730으로, held-out publisher에서는 0.655로 상승했지만 SAPBert cross-encoder는 같은 풀에서 full-test macro F1 0.287, held-out publisher 0.233에 머물렀다. 반면 GPT-5 adjudication은 full-test macro F1을 0.549, held-out publisher macro F1을 0.533으로 끌어올려 retrieval ceiling을 end-to-end로 더 잘 활용했음을 보였다. 저자들은 이 성능 향상이 ‘후보 풀에서만 선택’하는 안전 제약을 보존하면서도 OOD(publisher shift)에서 크게 개선된다는 점에서 의미가 크다고 정리한다.



### Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization (https://arxiv.org/abs/2606.23989)
- **Prior Approaches**: 기존 멀티문서 요약(MDS)은 end-to-end LLM이 생성 후에 근거를 붙이는 경우가 많아 환각(unsupported content)과 함께 검증 가능성이 낮다는 한계가 있었다. 인용이 있어도 문서/문단 단위로 뭉뚱그려져 있어 각 문장이 실제로 어떤 근거를 쓰는지 확인하기 어렵고, 충돌을 침묵적으로 덮는 경향도 컸다. 또 span-first 방식은 원시 span에 고정해 생성하지만, 문서 간 동등 주장 통합이나 개별 주장 단위 검증까지는 자연스럽게 이어지기 어렵다.

- **Core Contribution**: 이 논문은 Extract–Select–Rewrite의 중간표현을 ‘인용·검증의 단위’로 바꾸는 CAMS를 제안한다. CAMS는 각 문서에서 원자적 claim을 추출해 토큰 수준 provenance를 만들고, 문서 간 동등 claim을 클러스터링하며 충돌을 감지한 뒤, 선택된 claim에만 기반해 재작성하고 각 문장에 claim-근거(span) 포인터를 달아 검증 루프까지 수행한다. 핵심은 문장 생성 전에 내용이 지역화(localized)되도록 파이프라인 구조를 attribution-oriented, faithfulness-oriented로 설계한 점이다.

- **Technical Challenges**: 원자적 claim이 어디에 ‘정확히’ 존재하는지 오프셋으로 직접 강제하면 LLM이 위치를 잘못 내놓는 문제가 있어, claim(정규화 텍스트)과 해당 quote를 분리 저장하고 quote를 span으로 매핑하는 방식으로 처리했다. 또한 클러스터링에서 동등성/유사성을 embedding만으로 판단하면 ‘서로 반대인 주장’이 합쳐질 위험이 있어, bidirectional entailment 기반으로 비상호-엔테일링 병합을 거부하고 후보 충돌 쌍만 골라 3-class NLI로 contradiction을 판정한다. 마지막으로 생성 후 드리프트를 막기 위해 문장에 대해 인용된 span이 문장을 뒷받침하는지(지지)와 인용 밖의 추가 사실이 없는지(정밀) 양방향 체크와 제한적 repair를 수행한다.

- **Empirical Impact**: MultiNews, DiverseSumm, WCEP에서 품질·신뢰도·로컬라이제이션을 분리 평가했으며, 특히 독립적인 support 평가 모델로 citation precision을 감사(audit)하는 2-regime 프로토콜을 사용했다. CAMS는 강한 end-to-end 및 span-attribution 베이스라인과 견줄 만한 요약 품질을 유지하면서, faithfulness와 인용 정밀도에서 크게 개선되어 multi-source attribution 정확도를 약 2/3 수준으로 끌어올렸다. 또한 faithfulness–coverage 트레이드오프를 선택 임계값으로 ‘조절 가능’하게 드러내 end-to-end 모델이 암묵적으로만 다루던 문제를 명시적으로 통제할 수 있음을 보여준다.



### Does My Embedding Reflect That $A = B$? Evaluating Mathematical Equivalence in Embedding Models (https://arxiv.org/abs/2606.23959)
Comments:
          18 pages, comments welcome

- **Prior Approaches**: AI-for-math 연구에서 임베딩 기반 검색은 많이 다뤄졌지만, 기존 성과는 대체로 표면 수준의 유사도(어휘/문장 형태)를 잘 반영할 때 강하게 나타났다. MIRB 같은 벤치마크는 수학 검색을 포괄하지만, 본 논문이 겨냥한 ‘수학적 동치성 자체’를 구체적으로 측정·개선하려는 목적과는 결이 다르다. 즉, 현재 임베딩 모델이 “언어가 달라도 같은 수학”을 가깝게 모으는지에 대한 검증과 학습 설계가 부족했다.

- **Core Contribution**: 본 논문은 Mathematically Equivalent but Lexically Different Pairs(MELD) 데이터셋을 제안해, 어휘적으로는 구별되지만 수학적으로는 동치인 문장 쌍 270개를 자연어/LaTeX로 구성했다. 실험 결과, 최신 임베딩 모델들이 동치보다는 해당 문장이 소개되는 수학 하위 분야의 용어·표현에 따라 군집화되는 경향을 보였다. 이를 개선하기 위해 informal(자연어)과 formal(Lean) 표현을 같은 ‘관점(view)’으로 두는 contrastive 학습을 제안한다.

- **Technical Challenges**: 핵심 과제는 동치성을 반영하도록 임베딩을 “유도”하는 학습 신호를 확보하는 것이며, MELD만으로는 규모를 키우기 어렵다. 이를 위해 mathlib의 formal 정의/정리와 함께 자연어 요약을 묶은 mathlib_informal 데이터(augmented)에서 informal 재서술(rephrasing)을 추가로 생성하고, 이를 여러 view로 삼아 contrastive objective로 미세조정했다. 그 결과 Qwen3-Embedding-8B 기반 MathLeap-Qwen-8B와 Octen-Embedding-8B 기반 MathLeap-Octen-8B를 학습해 수학 동치성 정렬을 강화했다.

- **Empirical Impact**: 모델들은 MELD에서 평가한 범위 내에서 가장 높은 점수를 기록했으며, 자연어만으로 검색하는 MELD에서도 성능 이득이 나타났다. 또한 recall@1이 기준 모델 대비 크게 상승(예: Qwen3 기반 17.0→27.2)하고, 동치 쌍의 평균 순위도 개선되는 등 임베딩 공간의 정렬 변화가 관측됐다. 다만 MathOverflow 중복 질문/Lean premise retrieval처럼 더 넓은 자연어 이해나 proof-state 질의가 필요한 과제에서는 미세조정이 과특화되어 성능이 떨어질 수 있음을 한계로 제시했다.



### Layer-wise Probing of wav2vec 2.0 and Whisper for Consonant Cluster Reduction in African American English (https://arxiv.org/abs/2606.23948)
Comments:
          This paper has been accepted for presentation at Interspeech 2026

- **Prior Approaches**: 기존 연구는 AAE(아프리칸 아메리칸 잉글리시) 화자에 대한 ASR 성능 격차가 주로 학습 데이터 불균형과 방언의 음운·형태통사 특성의 미흡한 반영에서 비롯된다고 봐왔다. 또한 wav2vec 2.0이나 Whisper 같은 speech model 내부를 probing으로 해석하려는 시도는 많았지만, AAE의 대표적 음운 규칙인 consonant cluster reduction(CCR)은 거의 다뤄지지 않았다. 지금까지의 해석은 주로 오류 크기나 특정 음성학적 특징(예: 복합적 복원)에 집중했고, CCR처럼 ‘삭제되지만 문맥으로는 정체가 복원될 수 있는’ 패턴의 내부 표현은 공백으로 남아 있었다.

- **Core Contribution**: 이 논문은 CCR을 ‘단순 세그먼트 삭제’가 아니라 ‘구조화된 정도(gradient) 변화’로 모델이 어떻게 인코딩하는지 밝히는 것을 목표로 한다. 이를 위해 wav2vec2-base(자기지도)와 Whisper-small(지도학습) 인코더 표현을 speaker-independent, layer-wise probing으로 분석한다. 구체적으로 reduction detection(축약 vs 정형)과 segmental restoration(축약된 입력에서 underlying cluster 정체를 복원) 두 과제를 통해, 모델이 삭제된 /t,d/의 정체를 재구성하는 신호를 유지하는지 확인한다.

- **Technical Challenges**: 핵심 기술적 난제는 CCR 관련 표본을 대규모·사회적으로 다양한 화자에서 확보하면서도, 다른 음운 과정(예: L-vocalization, R-deletion)이나 형태론 조건의 영향을 분리해 ‘순수 CCR’만 평가하는 것이다. 이를 위해 CORAAL(Corpus of Regional African American Language)에서 Montreal Forced Aligner와 CMU 사전 기반 forced alignment로 canonical·reduced 발음을 자동 생성/검증하고, CCR에 취약한 특정 클러스터만 선별해 downsampling 및 monomorphemic 조건으로 교란을 줄였다. 또 MLP 선형 probing을 사용하되, 축약 판별은 물론 C1(첫 자음)만으로도 정확도가 얼마나 나오는지 gating-style 보조 실험까지 수행해 coarticulatory(공동조음·문맥) 단서가 성능을 좌우하는지 분해한다.

- **Empirical Impact**: 두 모델 모두 CCR의 reduced 형태와 canonical 형태를 높은 정확도로 구분하며, 이는 내부 표현에 CCR 정보가 실제로 존재함을 보여준다. 특히 reduced 구간에도 underlying stop의 정체에 대한 단서가 남아 있어, CCR이 단순 삭제가 아니라 ‘구조화된 gradient 음운 변이’로 인코딩된다는 결론을 지지한다. 나아가 데이터 부족 상황에서도 모델·클러스터에 따라 일관되게 다른 robustness가 관찰되어, AAE 관련 ASR 격차의 원인을 해석 가능한 수준에서 추적할 실마리를 제공한다.



### QuechuaTok: Morphological Boundary Accuracy as a Necessary Metric for Tokenizer Evaluation in Agglutinative Low-Resource Languages (https://arxiv.org/abs/2606.23943)
Comments:
          4 pages, 3 tables, 1 figure. Code available at this http URL

- **Prior Approaches**: 기존 Quechua NLP는 번역이나 언어모델링에 집중해 토크나이저 평가는 다국어 사전학습 모델의 설정에 의존하는 경우가 많았다. 또 BPE, Unigram LM, WordPiece 같은 통계 기반 서브워드 기법은 빈도 패턴을 따라 분절하므로, 교착어에서 형태소 경계가 빈도 경계와 불일치하면 성능이 흔들린다. 결과적으로 “덜 쪼개지면 더 좋은가?”라는 질문에 대해 표준 지표로는 형태론적 정합성을 충분히 반영하기 어렵다.

- **Core Contribution**: 이 논문은 Southern Quechua(quz)를 대상으로 BPE, Unigram LM, WordPiece, morphology-aware PRPE(접사-어간-접미 구조를 반영한 PRPE) 네 가지 토크나이저를 200k 문장 코퍼스에서 동일 조건으로 비교하는 QuechuaTok 벤치마크를 제안한다. 또한 형태소 경계 정합성을 직접 평가하기 위해 SQUOIA 유한상태 형태분석기 기반의 silver standard와 함께 MorphAcc(모폴로지 경계 정확도) 메트릭을 도입한다. 이를 통해 fertility rate(비옥도)가 교착어의 형태론적 품질을 대변하지 못함을 명확히 보인다.

- **Technical Challenges**: 핵심 기술적 난제는 교착어에서 토크나이저가 만든 경계가 실제 형태소 경계와 얼마나 맞는지 정량화하는 일이었다. 저자들은 SQUOIA 분석 결과를 silver standard로 삼아, 토크나이저 경계가 형태소 경계와 일치하는 비율(MorphAcc)을 계산하는 평가 파이프라인을 구성했다. PRPE는 23개 접사(예: TAM, 인칭, 격, evidential suffix 등)에 기반한 수작업 접사 사전을 필요로 하며, 커버리지에 따라 성능이 좌우된다는 점도 함께 드러난다.

- **Empirical Impact**: 실험 결과 fertility rate만 보면 BPE가 더 낮지만, 형태소 경계 정확도는 크게 떨어져 BPE(16k)는 MorphAcc 6.67%에 그친다. 반대로 PRPE는 MorphAcc 83.33%로 가장 높았고, fertility rate 지표와 독립적으로 형태론적 분절을 잘 포착한다. 언어모델링 관점의 perplexity에서는 Unigram 4k가 가장 낮았지만, 어떤 토크나이저도 세 지표를 동시에 최적으로 달성하지 못해 교착어에서는 fertility rate 단독 사용의 위험이 실증적으로 확인됐다.



### When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents (https://arxiv.org/abs/2606.23937)
- **Prior Approaches**: 정책-제약 tool-use 시스템에서 정책 절(clause)을 미리 검색한 뒤 allow/verify/refuse 같은 사전( pre-action ) 분류를 수행하는 흐름이 널리 쓰인다. 이때 검색 품질 평가는 benchmark에 지정된 ‘정답 정책 절’이 top-k에 들어오는지의 exact-match recall로 대신하는 경우가 많지만, 정답이 아니더라도 결정에 필요한 정책 정보가 포함될 수 있어 downstream 유용성과의 괴리가 생길 수 있다.

- **Core Contribution**: tau-bench에서 Qwen2.5-3B/7B 분류기를 대상으로, (1) decision-state 표현 방식이 정책 분류에 미치는 효과와 (2) 정책 절을 검색해 넣었을 때 exact-match recall이 downstream 성능을 얼마나 예측하는지(프록시 타당성)를 분리해 평가한다. 또한 gold-policy로 조건을 준 설정에서는 compact structured state가 raw trajectory 대비 macro-F1을 크게 끌어올리지만, 검색으로 정책 절을 주입할 때는 recall이 낮아도 성능 저하가 감지되지 않는 구간이 존재함을 보인다.

- **Technical Challenges**: 주요 기술 과제는 ‘정확히 같은 clause를 회수했는가’라는 오프라인 진단이 ‘분류 루프에서의 정책 유용성’을 과소평가할 수 있다는 점을 직접 검증하는 것이다. 논문은 gold-policy 입력을 structured 형태로 명시적으로 구성하는 생성기( {request intent, evidence, policy assertion, action class} 추출 )로 representation 효과를 먼저 확실히 만든 뒤, 테스트 시 해당 정책 절을 top-1 retrieved clause로 치환하는 직접 개입 실험으로 recall-성능 대응을 시험한다.

- **Empirical Impact**: airline 설정에서 top-1 retrieved clause의 exact-match recall은 약 7%에 그치지만, 3B 분류기는 retrieved clause로 macro-F1 0.58을 얻어 gold clause(0.60) 대비 눈에 띄는 손실이 관측되지 않았다(95% CI가 넓어 비열등(non-inferiority) 확정은 어려움). mismatched-policy(0.32)와 no-policy(0.21) 대비로는 retrieved clause가 유의미하게 높아, 이 벤치마크에선 recall만으로 downstream 정책 유용성을 평가하면 보수적으로 틀릴 수 있음을 시사한다. 결과는 다른 retriever/7B 및 파인튜닝 구성에서도 유사한 패턴이 나타나며, 정책 검색 평가는 recall만이 아니라 ‘검색-분류 루프에서의 성능’까지 함께 측정해야 한다는 실무적 함의를 제공한다.



### Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs (https://arxiv.org/abs/2606.23915)
- **Prior Approaches**: RAG에서 “attribution(근거 인용/청구 지지)” 평가는 lexical overlap, embedding 유사도, NLI(엔테일먼트) 기반 자동 점수를 묶어 서로 대체 가능하다고 가정해 왔다. 그러나 기존 연구들은 자동 점수가 인간 판정과 어긋나며 단일 지표가 전 구간에서 최선을 보장하지 못한다는 점을 이미 시사했다.

- **Core Contribution**: 이 논문은 LLM attribution을 평가하는 여러 자동 scorer(8종)를 대상으로, 특정 데이터셋 조합에서도 “최고 scorer가 그대로 전이(transfer)되는지”를 감사(audit)하는 기준을 제안한다. 구체적으로 다중-데이터셋으로 구성된 evaluation construct 안에서, 어떤 scorer도 모든 데이터셋에 대해 최상 성능 구간(95% CI) 내에서 안정적으로 유지되지 않는다는 결론을 제시한다.

- **Technical Challenges**: 핵심 도전은 scorer 간 성능이 데이터셋과 construct에 따라 뒤집히는데, 이것이 길이/잘림(truncation) 같은 단순 요인인지 분리해내는 것이다. 연구진은 provenance/topicality, generated-answer attribution, fact-check entailment의 구성 자체를 분리해 평가하고, AttributionBench(4개 출처, n=1610) 및 독립 세트(HAGRID 등)에서 AUROC·순위 통계(Kendall tau)와 leave-one-dataset-out regret로 검증해 “전이 실패가 구조적”임을 보여준다.

- **Empirical Impact**: 생성문-근거 attribution(AttributionBench)에서는 최고의 NLI scorer가 단문 데이터셋에서는 강하지만 장문 데이터셋에서 AUROC가 0.53(우연 수준)으로 붕괴하는 등 per-dataset 순위 역전이 관찰됐다. 또한 ‘best-on-average’로 평가기를 고르면 leave-one-dataset-out에서 평균 regret(0.172 AUROC)이 커져, metric 선택은 반드시 타깃 데이터셋 검증을 거쳐야 한다는 실무적 경고를 남긴다.



### One Year Later...The Harms Persist, But So Do We! (https://arxiv.org/abs/2606.23884)
Comments:
          20 pages, 8 tables

- **Prior Approaches**: 기존에는 범용 LLM을 정신건강 대화에 활용하되, 안전장치가 임상 조건에 따라 일관되게 작동하는지에 대한 검증이 부족했다. 특히 DSM-5의 여러 임상 진단을 폭넓게 비교하면서 공격 상황에서 취약점을 체계적으로 분류한 연구가 제한적이었다.

- **Core Contribution**: 이 논문은 16개 DSM-5 조건에 대해 6개의 proprietary LLM을 평가하고, 피해를 8개 차원으로 분류한 harm taxonomy와 다차원 평가 프레임워크를 제시한다. 그 결과 안전장치가 자살·자해에서만 비교적 안정적으로 유지되고, 섭식장애·물질사용장애·주요우울장애 등에서는 실패율이 최대 100%까지 나타났다고 보고한다.

- **Technical Challenges**: 핵심 기술적 난제는 임상 조건별로 어떤 형태의 위해가 발생하는지 정의가 불명확하고, 공격에 의해 안전성 경계가 쉽게 무너지는지 정량화하기 어렵다는 점이다. 연구진은 4가지 adversarial attack variant로 시나리오를 흔들어 보며, 8차원 harm taxonomy와 다차원 평가로 위해 유형을 구체화해 모델 취약성을 더 일관되게 드러내도록 설계했다.

- **Empirical Impact**: 실험 결과는 범용 LLM의 안전장치가 특정 진단군에 대해서는 신뢰 가능하지만, 다른 임상 조건에서는 전면적 실패가 발생할 수 있음을 경험적으로 보여준다. 저자들은 교육 현장 등 취약 집단이 상호작용하는 영역으로의 통합이 늘어나는 만큼, 임상 조건 전반에 걸친 명확한 위해 범주 정의와 그에 상응하는 safeguards 구현이 선행돼야 한다고 경고한다.



### Ground Then Rank: Revisiting Knowledge-Based VQA with Training-Free Entity Identification (https://arxiv.org/abs/2606.23881)
Comments:
          Accepted by ACL 2026 Findings. Project page this https URL

- **Prior Approaches**: 기존 KB-VQA 대응은 multi-modal retrieval-augmented generation(MM-RAG) 흐름에서 검색 후 재랭킹을 통해 답을 뒷받침할 텍스트 구간을 찾는 방식이 주류다. 그런데 많은 방법이 entity 식별과 evidence(증거) 구간 선택을 한 번의 재랭킹 단계로 강하게 결합해, 엔티티는 맞지만 구간이 틀리거나 반대로 구간은 그럴듯하지만 엔티티가 틀리는 문제가 발생한다. 또한 학습 기반 멀티모달 재랭커를 붙이는 경우가 많아 비용과 데이터 의존성이 커진다.

- **Core Contribution**: 이 논문은 KB-VQA에서 필요한 grounding을 entity-level grounding과 section-level grounding의 두 병목으로 재정의하고, 이를 workflow 관점에서 명시적으로 분리하는 training-free Identify Before Answer(IBA)를 제안한다. 핵심 아이디어는 MLLM이 답을 생성하기 전에 ‘후보 엔티티 이름’들 중 고신뢰 엔티티를 먼저 고른 뒤, 그 엔티티로 좁혀진 범위에서 텍스트 re-ranker로 증거 구간을 고르는 파이프라인이다. 이를 통해 fine-tuning 없이도 정확도와 효율을 동시에 노린다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 open-ended 엔티티 이름 생성 방식이 MLLM에 높은 불확실성을 요구해 실패가 잦다는 점이다. 저자들은 흥미로운 실험 관찰로, 모델에게 엔티티 이름을 자유 생성하게 하지 않고 candidate name shortlist에서 고르도록 하면 식별 정확도가 크게 올라간다고 보고 이를 tip-of-the-tongue 현상에 비유한다. 구현은 EVA-CLIP-8B로 시각적으로 후보 페이지를 먼저 뽑고, Qwen-2.5-VL-7B-Instruct로 후보 엔티티를 1차 선별한 뒤 BGE 같은 off-the-shelf textual re-ranker로 evidence를 재랭킹하는 방식이다.

- **Empirical Impact**: Encyclopedic-VQA(E-VQA)와 InfoSeek에서 IBA는 fine-tuned multi-modal re-ranking 기준선을 전반적으로 능가하며, training 및 추론 복잡도를 줄이면서도 성능을 끌어올렸다고 보고한다. Recall@1(정답 엔티티 우선순위)에서 InfoSeek은 58.4%로 EchoSight의 53.1%를 상회하고, E-VQA에서는 약간 낮은 Recall@1(35.5%)을 보이지만 다운스트림 답변 품질은 여전히 경쟁적 수준을 유지한다. 분석에 따르면 개선은 단순히 엔티티 식별이 좋아진 데 그치지 않고, 정답 엔티티가 고정된 뒤 더 유익한 evidence를 선택하게 된 효과가 함께 작용한다.



### Evaluating LLM Usage for Efficient and Explainable Numerical and Classified Implicit Sentiment Analysis of Product Desirability (https://arxiv.org/abs/2606.23701)
Comments:
          20 pages, 6 figures, 11 tables. arXiv admin note: text overlap with arXiv:2408.01527

- **Prior Approaches**: 기존 감성 분석은 대체로 리뷰 점수나 명시적 라벨에 의존해 범주형(긍정/부정) 분류 중심으로 작동했지만, PDT 같은 정성 데이터의 ‘감성 강도(수치)’를 안정적으로 뽑기 어렵다는 한계가 있었다. 또한 사전 기반(lexicon-based) 방법이나 특정 도메인에 맞춘 transformer 기반 기법들은 PDT 응답의 맥락 의존적·암묵적 뉘앙스를 충분히 반영하지 못해 신뢰할 만한 정량 성능을 내기 힘들었다. LLM을 쓰더라도 과거에는 암묵 감성 분석에서 도메인 특화나 추가 학습 없이 바로 쓰면 성능이 흔들리곤 했고, 예측 근거의 해석 가능성도 문제로 지적돼 왔다.

- **Core Contribution**: 이 논문은 PDT(Product Desirability Toolkit)에서 얻은 5개 단어-설명(PDT Respondent Term Grouping)을 그대로 LLM에 넣어, 명시적 리뷰 점수 없이도 제품 desirability(선호/바람직함)를 수치 점수와 범주형 감성으로 동시에 정량화하는 zero-shot 프레임워크를 제안한다. 특히 LLM의 confidence rating과 사람이 읽을 수 있는 rationale(explainability/xAI)을 함께 산출해 결과의 투명성과 실무적 신뢰를 높이도록 설계했다. ZORQ와 CARMA 두 PDT 데이터셋(총 106개 응답 묶음)을 대상으로, 전문가 라벨과의 정합성을 체계적으로 비교한다.

- **Technical Challenges**: 핵심 기술 난제는 정성 서술에 내재된 암묵적 감성의 ‘강도’를 일관된 수치로 환산하는 것과, 모델이 데이터가 제시되는 형식이 달라도(여러 형태로 입력) 성능을 유지하는 견고성을 확보하는 데 있었다. 저자들은 PDT의 구조화된 단어-설명 입력을 LLM에 넣고, zero-shot 연속 수치 스코어링과 다중 범주 분류를 함께 수행하는 파이프라인으로 해결했다. 더불어 모델 confidence와 rationale을 결과에 포함해, 단순 정확도뿐 아니라 해석 가능성과 투명성까지 함께 검증 가능한 형태로 제공한다.

- **Empirical Impact**: 실험 결과, LLM은 정성 응답에서 수치 감성 점수를 생성했을 때 전문가 라벨과 Pearson 상관이 최대 0.97까지 도달했으며, 범주형 분류 정확도도 최대 94%를 기록했다. 반면 lexicon 기반 및 transformer 기준선은 통계적으로 유의미한 성능을 내지 못했다. 또한 GPT-4o-mini는 더 큰 모델과 유사한 94% 수준 성능을 더 낮은 비용으로 달성해 대규모·지속적 제품 만족도 분석에 적합함을 보여주었고, confidence와 rationale 포함으로 해석 가능성과 현업 적용성(신뢰/투명성)을 강화한다.



### Self-Recognition Finetuning can Prevent and Reverse Emergent Misalignmen (https://arxiv.org/abs/2606.23700)
Comments:
          18 pages, 11 figures

- **Prior Approaches**: Emergent misalignment(EM)은 특정 페르소나 벡터의 활성화와 ‘악한’ 특성의 동반으로 설명되어 왔지만, 기존 대응은 주로 학습 중(in-training) 방어에 의존해 개입 방식이 제한적이었다. 또 많은 방법들이 EM을 직접적인 유해 콘텐츠 학습으로 취급해, 왜 정렬된 캐릭터가 흔들리며 문제가 커지는지에 대한 기계적 관점이 약했다.

- **Core Contribution**: 본 연구는 EM을 “정렬된 캐릭터의 붕괴/교란”으로 재정의하고, 이를 겨냥한 character-targeted 개입으로 self-generated text recognition(SGTR) fine-tuning을 제안한다. SGTR은 기존 방어처럼 학습 과정에서 차단하는 방식이 아니라, 캐릭터 복원과 강화를 목표로 하는 파인튜닝 개입이라는 점에서 구별된다.

- **Technical Challenges**: 핵심 기술 과제는 (1) EM이 손상시키는 능력을 정확히 복원할 수 있는지, (2) 예방(prevention)에서 단일 지표 악화 없이 misalignment만 줄일 수 있는지 검증하는 것이었다. 저자들은 3개 모델(GPT-4.1, Qwen2.5-32B-Instruct, Seed-OSS-36B-Instruct)과 여러 EM 데이터셋에 대해 2-stage fine-tuning을 설계하고, SGTR을 도메인 정답 데이터/일반 지식/word counting 같은 benign fine-tuning 기준선과 체계적으로 비교했다.

- **Empirical Impact**: 실험 결과 EM reversal에서는 모든 개입이 비슷한 수준의 역전 효과를 보였지만, “EM이 저하시킨 능력 복원”이 있을 때에만 차별적인 개선이 나타났다. EM prevention에서는 SGTR만 일관되게 misalignment를 낮추면서 개별 메트릭을 악화시키지 않았고, 추가 분석으로 EM과 LLM의 기본 identity self-report(자기 정체성 보고) 간의 연결(다양성 유도, 자기 인식 교란의 악화, identity-bearing system prompt 제거 시 효과 감소)을 근거로 제시해 방어 설계의 방향성을 강화했다.



### Quantifying Prior Dominance in RAG Systems (https://arxiv.org/abs/2606.23695)
Comments:
          15 pages, Preprint

- **Prior Approaches**: 기존 RAG 평가는 Exact Match(F1) 같은 이산 휴리스틱에 크게 의존하거나, LLM-as-a-judge 방식처럼 다른 블랙박스 모델의 판정에 의존하는 경우가 많았습니다. 그 결과 모델이 검색된 맥락에서 정보를 ‘추출’했는지, 아니면 파라메트릭 메모리로 답을 ‘회상’했는지 구분이 어려운 epistemic blindness 문제가 남습니다.

- **Core Contribution**: 이 논문은 Normalized Context Utilization(NCU)라는 연속형 지표를 제안해, 검색 맥락이 예측 불확실성을 얼마나 줄였는지를 토큰 log-probability 기반으로 정량화합니다. zero-shot(쿼리만)·oracle(정답 포함 맥락)·adversarial(정답 대체) 조건을 함께 두고, 엄격한 factual extraction(Chain-of-Thought 미사용) 워크플로에서 contextual 정보이득을 분리 평가합니다.

- **Technical Challenges**: 모델마다 토크나이저와 어휘 구성이 달라 log-probability 합을 그대로 비교하면 불공정해지는 문제가 있어, 길이 정규화로 확률 공간을 표준화합니다. 또한 posterior가 prior보다 불확실성을 더 키우는 negative transfer까지 포착하도록 NCU를 경계값 처리·스무딩과 함께 설계하고, 우회적으로 단순 정확도보다 미세한 confidence 변화를 추적합니다.

- **Empirical Impact**: 실험 결과 1.5B~72B 범위에서 엄격 추출 과제는 스케일 증가에 따른 이득이 크게 둔화되며, 작은 모델(예: 1.5B)은 oracle 정확도·NCU가 큰 모델과 동급이거나 더 낫고 지연시간은 상용 API 대비 대폭 절감됩니다. 또한 proprietary 상용 모델은 adversarial 충돌에서 절반가량 맥락을 무시하는 Prior Dominance 경향과 함께, 상반된 priors를 만났을 때 confidence collapse로 negative transfer 위험이 더 높게 나타났고 이는 SLM보다 구조적으로 취약함을 시사합니다.



### ModTGCN: Modularity-aware Graph Neural Networks for Text Classification (https://arxiv.org/abs/2606.23694)
Comments:
          PAKDD2026

- **Prior Approaches**: 기존 TextGCN 계열 그래프 텍스트 분류는 문서-단어 이웃을 중심으로 local neighborhood aggregation에 의존해 왔습니다. 하지만 의미 기반 문서 그래프는 라벨별로 내부 연결이 촘촘하고 외부 연결이 희박한 mesoscopic community(커뮤니티) 구조를 보이는데, 이를 학습 목적에 직접 반영하지 못해 class boundary가 흐려지거나 over-smoothing이 발생할 수 있습니다.

- **Core Contribution**: ModTGCN은 분류용 cross-entropy와 함께 modularity 기반 auxiliary objective를 공동 최적화해 문서의 class-coherent 커뮤니티 형성을 유도합니다. modularity 항은 transformer 임베딩으로 만든 document-document similarity graph에서 계산되며, 라벨이 없는 노드에는 pseudo-label(예측 라벨)을 써서 적은 라벨로도 전역 구조 정규화를 제공합니다.

- **Technical Challenges**: 전역 커뮤니티를 modularity로 넣으려면, 라벨 신호를 제한된 supervised 노드에서 전역으로 전파하면서도 pseudo-label 오류를 악화시키지 않아야 합니다. ModTGCN은 (1) 사전학습/미세조정 SBERT 임베딩으로 문서-문서 그래프를 구성하고 (2) modularity 계산 시 유/무라벨 노드에 gold/ TextGCN pseudo-label을 혼합하는 hybrid supervision을 적용하며 (3) TextGCN의 이종 그래프를 document-word와 word-word로 decouple해 학습 복잡도를 줄였습니다.

- **Empirical Impact**: 5개 벤치마크에서 ModTGCN은 TextGCN 등 대비 일관된 성능 향상을 보였고, 특히 Ohsumed와 20NG처럼 low homophily·구조가 복잡한 데이터에서 개선 폭이 컸습니다. 또한 Gausssian(RBF) similarity와 label-aware edge reweighting, prediction 기반 modularity supervision 조합이 가장 강했으며, decoupled 구조 덕분에 training이 2x~10x 빨라져 확장성까지 입증했습니다.



### EXPO-SQL: Execution-based Clause-level Policy Optimization for Text-to-SQL (https://arxiv.org/abs/2606.23693)
Comments:
          20 pages, 8 figures

- **Prior Approaches**: Text-to-SQL은 자연어와 스키마를 입력으로 SQL을 생성해 실행 가능 쿼리를 만드는 기술로, 기존에는 SFT로 정답 SQL과의 토큰 정합을 맞추거나 prompting으로 추론/리파인 단계를 반복하는 방식이 주류였다. 그러나 두 방법 모두 실행 결과를 학습 신호로 직접 반영하지 못해, 잘못된 SQL이 나올 때 어떤 절(SELECT/WHERE 등)이 원인인지 구분하기 어렵다. 이후 RL 계열이 execution feedback을 보상으로 쓰기 시작했지만, 대다수는 쿼리 단위의 동일 보상을 모든 clause에 동일하게 부여해 coarse credit assignment 문제를 만든다.

- **Core Contribution**: 이 논문은 EXPO-SQL(EXecution-based clause-level Policy Optimization for Text-to-SQL)을 제안하며, 실행 결과를 바탕으로 clause별 오류를 식별하고 clause-level reward로 미세 감독을 제공한다. 기존 RL의 “정답/오답을 쿼리 전체에만” 주던 한계를 깨고, 틀린 절은 확실히 낮추되 맞는 절은 유지하도록 학습 신호를 분리한다. 특히 incorrect result와 execution error를 각각 다른 방식으로 clause 오류 집합 C_err에 매핑한다.

- **Technical Challenges**: 핵심 난제는 온라인 RL에서 clause별 오류를 안정적으로 할당하는 것이다: 어휘/구문만으로는 오류 절을 알 수 없고, SQL은 정답이 여러 형태로 나올 수 있으며, 실행 피드백도 기본적으로 binary(전체 쿼리 성공/실패)라서 직접 신호가 빈약하다. EXPO-SQL은 이를 위해 (1) incorrect result에서는 FROM/JOIN→WHERE→GROUP BY→HAVING→SELECT→ORDER BY→LIMIT의 결정적 실행 순서를 따라 clause-wise incremental execution으로 각 절이 결과를 바꾼 흔적을 추적하고, (2) diff_type(실행 결과 테이블 변화 유형 10종)과 누적 차이를 결합해 오류 절을 판별하며, (3) execution error에서는 에러 메시지에서 원인 요소를 파싱한 뒤 clause-level tracing으로 실제 root clause까지 역추적한다.

- **Empirical Impact**: 실험은 Spider와 BIRD 등 대표 벤치마크에서 SFT·prompt·기존 RL 대비 일관된 우위를 보이며, Spider-Dev/BIRD-Dev에서 각각 1.2%p/2.4%p 개선, 복잡 쿼리에서는 5.6%p까지 향상됐다. 또한 ablation에서 incorrect result·execution error 모두에 clause-level rewards를 적용할 때 성능이 가장 높게 나타나 fine-grained learning 신호의 효과를 뒷받침한다. 코드도 공개되어(제공 링크) 재현성과 후속 연구 확산에 기여할 것으로 보인다.



### Matching Tasks to Objectives: Fine-Tuning and Prompt-Tuning Strategies for Encoder-Decoder Pre-trained Language Models (https://arxiv.org/abs/2606.24841)
- **Prior Approaches**: 기존 prompt-based learning은 주로 특정 pre-training 목표나 고정된 학습/추론 포맷에 의존해 성능이 갈리곤 했다. encoder-decoder 계열에서 generation·question answering을 다루더라도, 과제에 맞는 pre-training 목적(objective)을 체계적으로 고르는 절차가 부족했다.

- **Core Contribution**: 이 논문은 다양한 pre-training objective가 encoder-decoder pre-trained language model의 생성 및 QA 성능에 어떻게 영향을 주는지 분석하고, 과제에 맞는 objective를 자동으로 선택·적용하는 MTO(Match Task to Objective) 프레임워크를 제안한다. 또한 pre-training 목표와 fine-tuning 적응 단계에 정합되도록 템플릿을 설계해 commonsense 지식 검색·완성에서 성능을 끌어올린다.

- **Technical Challenges**: 핵심 난제는 (1) 주어진 task에 어떤 objective가 맞는지 식별하고, (2) 그 objective에 맞게 task 데이터를 unsupervised 방식으로 준비하며, (3) pre-training-적응-템플릿 정합을 맞추는 것이다. 연구진은 MTO로 objective를 고르고, 해당 objective에 기반한 task 데이터 준비 방법과 objective와 정렬되는 fine-tuning 템플릿을 함께 설계해 few-shot에서의 적응 효율을 높였다.

- **Empirical Impact**: 정렬 전략은 few-shot 환경에서 기존 방식 대비 120%를 넘는 성능 향상을 보였고, 관련 연구들과 비교해도 일관되게 우수했다. full-dataset에서도 baseline을 능가하며, prompt-tuning까지 확장해 soft prompt engineering과 최적화에 실질적인 가이드를 제공하는 것으로 확인됐다.



### ParaPairAudioBench: Paralinguistic Pairwise Audio Benchmark for LALM-as-a-Judg (https://arxiv.org/abs/2606.24648)
Comments:
          Accepted to Interspeech 2026

- **Prior Approaches**: 기존 LLM-as-a-Judge는 텍스트 선호나 전반적 음성 품질처럼 ‘전체적 자연스러움’에 초점을 둔 경우가 많아, 발화 속도·강조·화자 특성 같은 미세한 준언어(paralinguistic) 구분은 충분히 다뤄지지 않았다. 음성 평가를 위한 LALM-as-a-Judge 연구도 주로 holistic 판단이나 자연스러움 중심이라, Tie 같은 애매함 상황에서의 신뢰도(보류 판단) 문제는 상대적으로 공백이 있었다. 특히 동일한 기준을 두 오디오 쌍으로 통제해, 어떤 단서(음향 vs 텍스트/어휘)에 의존하는지까지 진단하는 체계는 제한적이었다.

- **Core Contribution**: ParaPairAudioBench는 5,175개의 오디오 페어를 Style, Rate, Emphasis, Age, Gender의 5가지 준언어 축으로 분해해, 쌍대(pairwise) 비교와 Tie 선택을 통해 ‘진짜로 기준을 구분하는 능력’과 ‘애매할 때 보류하는 능력’을 동시에 점검하는 벤치마크를 제안한다. Rate를 제외한 대부분의 기준에서 Tie가 정답이 되는 상황을 명시해, 기존의 강제 선택(forced preference) 편향을 측정 가능하게 만든다. 또한 same-transcript/ cross-transcript 조건을 넣어, 어휘 단서와 음향 단서 중 무엇에 더 의존하는지까지 진단하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 기준별로 미세하게 구분되는 준언어 속성을 오디오 쌍으로 만들되, 어휘 영향과 순서 편향을 통제하고 (2) 애매한 경우 Tie를 정확히 선택하도록 ‘캘리브레이션(calibration)’을 평가하는 프로토콜을 갖추는 것이다. 이를 위해 벤치마크는 Tie(특히 Style/Age/Gender에서 Both Good/Both Bad 균형)와 프리젠테이션 순서 스왑(입력 A/B 교차 평가)을 포함하고, transcript-control로 same/cross 조건을 설계해 모달리티 의존성을 분리한다. 더 나아가 Emphasis는 국소적(단어 수준) 강조라서 same-transcript에서는 미세 신호를 요구하고, cross-transcript에서는 전반적 운율 맥락이 보조하도록 구성해 모델의 국소 민감도 한계를 드러내게 했다.

- **Empirical Impact**: 실험 결과, 현존 LALM judge들은 평균적으로 인간 판단보다 32%p 낮고, 특히 Tie 케이스에서 보류를 거의 못 하는 ‘심각한 캘리브레이션 실패’를 보였다. 예를 들어 Emphasis와 Style에서 Tie 정확도가 20% 미만으로 붕괴하는 양상이 관찰됐고, 이는 단일 자연스러움 점수로는 잘 드러나지 않는 취약점이다. 또한 Style에서는 same-transcript에서 유독 성능이 크게 오르는 반면 cross-transcript에서는 급락해 어휘 의존성이 드러났고, Emphasis는 반대로 cross에서 상대적으로 더 유리해 모델의 국소 prosody 민감도가 제한적임을 시사한다. 마지막으로 프리젠테이션 순서 스왑에서 위치 편향이 크며(일부 모델 최대 29.4%p), swap-robust 평가까지 포함해야 한다는 점을 실증적으로 강조한다.



### AdversaBench: Automated LLM Red-Teaming with Multi-Judge Confirmation and Cross-Model Transferability (https://arxiv.org/abs/2606.24589)
Comments:
          10 pages, 4 figures, 5 tables. Code and data at this https URL

- **Prior Approaches**: 기존 LLM 레드티밍은 LLM-as-a-Judge로 판정의 대규모화를 시도했지만, 레드티밍에서는 판정이 ‘정답 비교’가 아니라 ‘기준 위반 여부(pass/fail)’로 바뀌어 엄격/관대 판정 편향을 동시에 감지하기 어렵다. 또한 단순 프롬프트 생성이나 일괄 공격 세트는 어떤 변형 연산자가 어떤 유형 과제에서 왜 잘 먹히는지 설명력이 떨어질 수 있다. 마지막으로 inter-annotator agreement를 Cohen’s kappa로만 보면 라벨 쏠림(high-agreement, low-κκ paradox)으로 신뢰도가 왜곡될 위험이 있다.

- **Core Contribution**: AdversaBench는 seed prompt를 다섯 가지 structured operator로 변형하고, 타깃 모델을 호출한 뒤 3인 패널로 fail/pass를 판정하며 meta-judge tiebreaker로 판정 불일치를 정리하는 end-to-end red-teaming 파이프라인을 제안한다. 또한 “실패율”만이 아니라 seed별 attacker iterations(생성-검증 반복 횟수)까지 함께 보고, 실패 판단의 신뢰성은 단일 κ뿐 아니라 카테고리별 불일치율로 해석하도록 설계했다. 추가로 약한 모델에서 만든 adversarial prompt가 더 강한 모델로 zero-shot transfer되는지까지 확인해, 변형이 모델 특이 약점을 넘어 일반 행동 패턴을 겨냥할 가능성을 보여준다.

- **Technical Challenges**: 문제는 (1) hard input을 효율적으로 찾는 검색 전략과 (2) “실제 실패인지”를 신뢰도 있게 검증하는 판정 체계가 동시에 필요하다는 점이다. AdversaBench는 epsilon-greedy(ε=0.2)로 operator를 선택하고, failure 미확정 시 강한 변형 모델로 escalation하며, 최대 5회 반복과 checkpoint resume로 탐색 효율과 재현성을 확보했다. 검증은 expected_behavior를 모호함 없이 명시해 ground-truth 기반으로 판단하고, 3인 패널의 불일치는 meta-judge로 중재하되 κ의 한계를 보완하기 위해 raw disagreement 등 대체 지표를 함께 계산한다.

- **Empirical Impact**: 45개 seed(추론, 지시-따르기, 툴 사용) 모두에서 confirmed failure가 나왔고, 특히 instruction-following은 평균 2.4회 반복이 필요해 binary failure rate가 난이도 차이를 숨긴다는 점이 survival curve로 드러났다. operator 효과도 카테고리별로 급격히 달라졌으며, 예를 들어 inject_distractor는 instruction-following에서는 평균 reward 0.00이지만 reasoning/tool-use에서는 0.80~0.83 수준으로 크게 작동했다. 판정 신뢰성 분석에서는 80~87%의 높은 raw agreement가 라벨 쏠림 때문에 Cohen’s kappa가 거의 0에 가까워질 수 있음을 보여주며, 근거 기반의 “불일치율 보고”가 중요하다는 실무적 함의를 준다. 마지막으로 Llama 3.1 8B에서 생성한 adversarial prompt가 Llama 3.3 70B에 zero-shot으로도 상당 비율 전이되어, 공격 변형이 특정 모델 취약점이 아니라 일반적인 행동 취약성을 겨냥할 수 있음을 시사한다.



### A specialized reasoning large language model for accelerating rare disease diagnosis: a randomized AI physician assistance tria (https://arxiv.org/abs/2606.24510)
Comments:
          36 pages, 5 figures

- **Prior Approaches**: 기존 희귀질환 진단 지원 연구는 대규모 언어모델(LLM)의 가능성은 보여주었지만, 임상 현장 배포 관점에서 충분히 작동 가능한 형태가 아니거나 근거가 임상적으로 충분히 정교하지 못한 한계가 있었다. 또한 희귀질환은 데이터가 매우 부족해 학습용 텍스트가 적고, 결과가 long-tail 질환에서 불안정해지기 쉽다는 문제가 동반됐다.

- **Core Contribution**: 이 논문은 RaDaR(Rare Disease navigatoR)라는 32B 파라미터 규모의 오픈소스 compact reasoning LLM을 제안하며, 희귀질환 진단을 위한 deployable 추론 모델을 목표로 한다. RaDaR은 공개 임상 free-text 케이스와 합성(synthetic) 케이스를 함께 학습해 임상 근거 제약과 데이터 부족을 동시에 겨냥한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 희귀질환에서의 부족한 훈련 데이터, (2) 임상적으로 의미 있는 추론 신호를 학습시키는 방법, (3) 배포 가능한 모델 크기와 성능을 맞추는 것이다. 논문은 reasoning-enhanced training으로 49,170개의 공개 텍스트 케이스와 104,666개의 synthetic 케이스를 학습했고, 추가 분석에서 phenotype-anchored narratives(표현에 증상-현상 기준점을 둔 내러티브)가 long-tail 희귀질환에 유효한 학습 신호가 된다는 점을 ablation으로 제시한다.

- **Empirical Impact**: RaDaR은 공개 벤치마크와 4개 외부 검증 센터에서 평가된 오픈소스 모델 중 가장 높은 성능을 보였으며, DeepSeek-R1(671B)까지 포함한 비교에서도 강점을 보였다. 후향적 코호트에서는 문서화된 임상 의심 이전에 최종 진단을 우선순위로 내세운 비율이 61.06%로, 잠재적인 lead time 1.87개월과 within-center interval의 50.18%를 보고했다. 또한 무작위 의사 보조(trial)에서 RaDaR 도움은 인터넷 검색만 사용한 경우 대비 희귀질환 진단 정확도를 21.44%p 개선했으며, 합성 데이터 관련 ablation은 일정 구간에서 단조적 성능 스케일링 경향을 보여 실용성 근거를 강화한다.



### An LLM-based Two-Stage Transformer Framework for Cross-Domain Bearing Fault Diagnosis with Limited Data (https://arxiv.org/abs/2606.24459)
Comments:
          Accepted as a conference article of AIM 2026

- **Prior Approaches**: 기존 베어링 결함 진단 연구는 데이터의 이질성, 운전 조건 변화, 라벨 부족을 각각 따로 다루거나 한쪽에 치우치는 경우가 많았다. 또한 암묵적 feature alignment에 의존해 문제들이 동시에 발생할 때 전이 성능이 쉽게 흔들린다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 지식(guidance)을 활용한 two-stage transfer learning 프레임워크를 제안해, 진동 신호에서 계층적 특징을 뽑는 lightweight GPT-2-style Transformer를 사용한다. 사전학습된 인코더 가중치와 결함 prototype 임베딩을 지식 전달체로 삼아, 멀티 소스 사전학습에서 타깃 적응으로 가는 “명시적 경로”를 구성한다.

- **Technical Challenges**: 핵심 기술적 난제는 서로 다른 운전 조건과 결함 범주가 함께 변할 때, 일반화 표현과 타깃 맞춤을 동시에 달성하는 것이다. 논문은 multi-source learning으로 dual-shift를 완화하고, prototype-based 지식 조절과 taxonomy-adaptive classification을 통해 이질적인 결함 분류 체계에서도 자연스러운 전이를 가능하게 한다.

- **Empirical Impact**: 실험은 실제 데이터 4종에서 수행됐으며, 타깃 라벨 10%만 사용해 평균 92.61% 정확도를 달성했다. 기존 최고 성능 대비 17.24%p 향상으로 cost-effective predictive maintenance에 실질적인 경로를 제시하며 Industry 4.0 적용 가능성을 강화한다.



### Bayesian control for coding agents (https://arxiv.org/abs/2606.24453)
- **Prior Approaches**: 코딩 에이전트는 생성기와 진단기(cheap diagnostics)·검증기(expensive verifier)를 함께 쓰지만, 실제 도구 선택과 중단 판단은 고정 규칙(항상 검증, best-of-N, 단일 크리틱 게이트, 사전 루프)에 의존하는 경우가 많다. 이런 방식은 후보 정답성에 대한 사후확률(posterior)을 유지하지 않고, 크리틱 호출의 정보가치와 비용을 명시적으로 저울질하지 못한다. 결과적으로 과제 난이도, 크리틱 신뢰도, 생성기의 수리/파손 확률, 검증 비용에 따라 멈춤 시점을 유연하게 조정하기 어렵다.

- **Core Contribution**: 이 논문은 코딩 에이전트의 오케스트레이션을 cost-sensitive sequential hypothesis testing으로 정식화한다. 베이지안 컨트롤러가 후보 correctness에 대한 신념 b=P(Y=1|evidence)을 유지하면서, 더 증거를 모을지(크리틱), 후보를 정제할지(생성/리파인), 검증(오라클)을 실행할지, 혹은 중단할지 기대 효용 기준으로 동적으로 결정한다. 또한 이 신념은 해석 가능한 correctness 점수(불확실성 점수)로도 활용되며, 단순 토큰 확률이나 도구 성공률보다 불확실성 추정에 강점이 있음을 제시한다.

- **Technical Challenges**: 핵심 난제는 크리틱 관측이 잡음 섞인 증거라는 점과, 생성 단계가 정답을 고치기도 하지만 깨뜨리기도 한다는 전이 불확실성을 함께 모델링하는 것이다. 논문은 (1) 크리틱의 조건부 우도 P_i(z|Y)를 캘리브레이션 세트에서 추정하고 (2) 리파인 궤적의 오라클 라벨 연속쌍으로 생성 전이( fix/break 확률)를 베타-바이노미얼 스무딩으로 추정한 뒤, 이를 POMDP의 Bellman 방정식으로 연결해 정책을 만든다. 구현은 one-step Bayesian greedy와 finite-horizon Bayesian dynamic-programming 두 컨트롤러로 제공되며, 둘 다 base LLM(생성기)은 고정한 채 control layer에서만 최적화한다.

- **Empirical Impact**: 여섯 가지 생성기와 아홉 코딩 벤치마크에서 실험한 결과, 베이지안 제어는 ‘검증 비용이 크고 크리틱이 유의미하지만 완벽하진 않을 때’ 특히 효용이 크다. 반대로 공개 테스트가 숨은 정답 성공을 잘 예측하거나 후보가 이미 정답일 가능성이 높거나 검증이 싸면, 단순 공개 테스트 게이팅이나 always_verify 같은 고정 전략이 더 유리했다. 또한 조건부 전이(수리 확률)까지 계측해 다단계 리파인의 가치가 클 때에만 DP 기획이 그리디보다 추가 이득을 보였고, 베이지안이 유도하는 사후확률 점수는 불확실성 정량화에서 토큰-확률·raw tool-success 기준선을 앞섰다.



### Age of LLM: A Strategic 1v1 Benchmark for Reasoning, Diplomacy and Reliability of Large Language Models under Fog of War (https://arxiv.org/abs/2606.24391)
Comments:
          25 pages including appendices, 8 figures, 4 tables; appendices include verbatim system prompt and engine resolution pseudocode. All correlations reported with p-values, 95% bootstrap confidence intervals and Spearman's rho; includes a Steiger test and Bradley-Terry fit

- **Prior Approaches**: 기존 LLM 벤치마크(MATH, HumanEval, MMLU 등)는 대부분 단일 턴, 완전 관측, 정답이 명확한 과제에 초점이 있어 적대적 불확실성 하 계획(planning under uncertainty)이나 숨은 상대 추론, 여러 턴 동안의 structured output 신뢰성(잘못된 JSON/불법 행동 처리)을 충분히 검증하지 못한다. 또한 공개 벤치마크는 학습 데이터로의 오염 가능성이 커 점수가 실제 경쟁력보다 부풀려질 위험이 있다.

- **Core Contribution**: Age of LLM은 두 LLM이 fog of war, full diplomacy(메시지·정전·최후통첩)와 함께 13x7 그리드에서 1v1로 기지를 파괴하는 turn-based 1v1 벤치마크를 제안한다. 특히 엔진을 비공개로 두고 매 매치마다 무작위 맵 seed와 상대를 바꿔 데이터 오염 경로를 줄이며, 매 턴 출력은 strict JSON 스키마를 강제하고 불법 행동은 조용히 폐기되는 “신뢰성 차원”을 포함한다.

- **Technical Challenges**: 이 설정에서 핵심 기술적 난제는 부분관측·상대의 은닉·딜레마(핵 발사 vs 지상 공략) 속에서도 JSON 스키마를 지키며, 불법 행동이 누적되지 않도록 믿음(belief)과 상태 추적을 해야 한다는 점이다. 논문은 (1) near rule-only 프롬프트로 build-order 같은 직접 조언을 배제하고 (2) 에이전트가 매 턴 관측과 직전 턴 결과, 외교 기록만 이용하도록 하며 (3) 매치별 재현성을 위해 엔진-뷰어 간 replay 포맷과 replays를 제공해 분석 가능성을 확보한다. 다만 데이터 수집 당시 포함됐던 두 개의 borderline 전술 시드 문구가 정찰/탱크 중심 경향을 증폭했을 수 있어, 그 일부 결론은 제한점으로 표시한다.

- **Empirical Impact**: 54개 매치(15개 reasoning model, 5,258개 액션) 분석에서 결과는 핵(핵 선제 발사 루트)이 압도적으로 많았고(룰-일치 하위코퍼스 78%, 전체 85%), 군사 정복은 드물지만 성사되면 더 빠르게 끝나는 패턴(평균 12.3턴 vs 18.9턴)이 관측됐다. 외교는 활발히 오가지만 실제 합의로까지 이어지는 경우는 거의 없었고, 불법 행동의 약 58%가 fog/state 추적 오류로 나타나 illegal-action rate가 믿음 추적의 프록시로 기능함을 시사한다. 또한(탐색적 1개 모델 제외) reliability가 승리와 강하게 연결되지는 않았으며, 전체 순위는 소규모·불균형·사이드 미스왑의 영향으로 “예비적 기술 통계” 성격임을 강조한다.



### ComputeFHE: A Privacy-Preserving General-Purpose Computation Library (https://arxiv.org/abs/2606.24379)
Comments:
          16 pages, 3 figures

- **Prior Approaches**: FHE는 복호화 없이 암호문 위에서 계산이 가능하지만, 잡음 누적으로 인해 bootstrapping이 자주 필요해 실용성이 제약된다. 특히 TFHE 계열에서는 bootstrapping 비용이 병목이 되며, 이를 줄이기 위한 최적화가 중요하다. 기존에는 게이트/회로 수준 최적화나 컴파일러·합성 도구를 통해 회로를 FHE-friendly 형태로 바꾸려는 접근이 주로 사용됐다.

- **Core Contribution**: ComputeFHE는 TFHE를 기반으로 한 오픈소스 C++ 라이브러리로, 개발자가 암호화된 정수·고정소수점 데이터에 대해 C 스타일 명령형(imperative) 코딩 방식으로 알고리즘을 구현할 수 있게 한다. encrypted conditional과 oblivious array access 같은 핵심 제어·메모리 패턴 연산까지 제공해, 단순 게이트 실험을 넘어 실제 프라이버시 보존 응용 개발 흐름을 지향한다. 또한 표준 두 입력 논리게이트 기반 구현과, FHE-friendly 논리 프리미티브를 활용한 최적화 ALU 아키텍처를 함께 제공한다.

- **Technical Challenges**: 핵심 과제는 암호문 연산에서 bootstrapping이 요구되는 회로 복잡도를 낮추면서도, 암호화 조건 분기와 배열 접근 같은 고수준 구조를 효율적으로 회로에 매핑하는 것이다. ComputeFHE는 표준 ALU 대비 FHE-friendly 3입력 논리(예: Majority, MulAdd)를 활용해 연산별 bootstrapping 횟수를 줄이는 ALU_OPTIMIZED를 구성했으며, ciphertext-plaintext 연산 최적화로 불필요한 비용을 피한다. 더불어 simulation mode를 제공해 실제 암호 연산 없이 회로 복잡도와 bootstrapping 비용을 추정해 디버깅·성능 분석을 가능하게 했다.

- **Empirical Impact**: 실험 결과로 일부 연산에서 bootstrapping 횟수가 크게 감소하며, 선택 연산 기준 성능이 최대 3.9x까지 향상됨을 보인다. 예시로 quickstart에서는 91회에서 52회로 bootstrapping이 줄고, 정렬 예제에서도 3,843회 대비 2,016회로 감소를 보였다. 동시에 equality/inequality는 이미 near-optimal이라 감소 폭이 제한되는 등 연산별 이득이 달라짐을 확인하며, 개발자가 비용 지표를 사전에 추정할 수 있다는 점에서 실무적 가치를 제공한다.



### PETRA: Transforming Web Text for Petroleum-Engineering Domain Adaptation (https://arxiv.org/abs/2606.24346)
- **Prior Approaches**: 기존 석유공학 검색 적응은 주로 소규모 도메인 데이터나 일반 대규모 웹 코퍼스에 의존했지만, 두 경우 모두 라벨 희소성과 잡음(중복·OCR 아티팩트·약한 메타데이터) 문제로 인해 도메인 relevance를 안정적으로 학습하기 어려웠습니다. 또한 공개 검색 벤치마크는 품질 측정은 돕지만 석유공학용 검색 적응에 필요한 쿼리-패시지 관련성 감독(supservion)을 제공하지 못했습니다.

- **Core Contribution**: 이 논문은 PETRA( Petroleum Engineering Text for Retrieval Adaptation )라는 대규모 데이터셋과 파이프라인을 제안해, 잡음이 많은 공개 웹 텍스트를 정제된 도메인 코퍼스와 합성 감독으로 변환합니다. dense retrieval(임베딩 1단계)과 reranking(크로스 인코더 2단계) 모두에 대해 학습용 쿼리-패시지 신호를 만들고, 실제 배포 RAG 스택의 추론 후보 분포를 반영하도록 학습 레이블을 설계했습니다. 

- **Technical Challenges**: 핵심 난제는 (1) 도메인 문맥이 표면 단어 중첩만으로는 구분되지 않아 관련성 라벨이 희소하고, (2) 웹 데이터의 잡음이 모델 적응을 방해한다는 점입니다. 이를 위해 고-recall energy-domain classifier(테스트 정확도 98.4%)로 오프도메인을 게이트하고, 청크 수준 검증·중복 제거·chunk-grounded query 생성·LLM hard negative 생성·retrieval-mined 후보 리스트에 teacher 스코어를 입혀 teacher-scored candidate 형태로 재포장합니다. 특히 합성 라벨의 “생성 정확도”만으로는 성능이 안 오르며, 추론 시 후보 분포와 맞춘 teacher-scored candidate 학습 패키징이 중요하다는 실패 사례도 함께 보고합니다.

- **Empirical Impact**: PETRA 기반 적응은 1단계에서 in-domain SOP nDCG@10을 0.703에서 0.763으로 끌어올리고, score fusion 전략으로 out-of-domain 보존도 함께 달성합니다. reranker adaptation은 Earth Science 벤치마크에서 상대 44% 향상, 6개 reasoning-intensive 패널에서도 상대 23% 개선을 보였으며, 후보 리스트 기반 학습이 모든 평가 표면에서 일관된 이득을 만든다고 정리합니다. 또한 seed 분산 없는 단일 실행 보고와 SOP의 독점 구성(재현 제한), 지연·온라인 A/B 미측정 등 한계도 명확히 두었습니다.



### Dialogue to Discovery: Attribute-Aware Preference Elicitation for Conversational Product Search Assistants (https://arxiv.org/abs/2606.24194)
- **Prior Approaches**: 기존 온라인 쇼핑 보조는 키워드 기반 검색이나 facet/필터로 후보를 줄이지만, 남은 후보가 여전히 많거나 반대로 해답이 없을 수 있습니다. 또한 CPS로 LLM을 쓰더라도 zero-shot 추천은 대화가 효율적이지 않거나(무분별한 질의/추천), 화면 제약 속에서 대화 피로와 이탈이 커질 수 있습니다.

- **Core Contribution**: 이 논문은 Dialogue to Discovery(D2D)라는 속성(attribute) 중심 선호 질의·추천 프레임워크를 제안합니다. D2D는 대화 단계마다 “언제 추천할지”와 “어떤 속성/값을 물을지”를 카탈로그 내 속성 분포와 AVP(Attribute-Value Profile) 불확실성을 근거로 동적으로 결정해, 성급하거나 엇나간 추천을 줄이는 것을 목표로 합니다. 또한 사용자 이탈을 고려한 patience 모델을 함께 도입해 대화 품질과 효율을 함께 최적화합니다.

- **Technical Challenges**: 핵심 기술 과제는 제한된 화면에서 소수 후보만 보여줘야 하는 상황에서, 잘못된 추천을 일찍 내리면 engagement가 무너진다는 점입니다. D2D는 (1) 검색된 후보 부분집합에서만 선호를 갱신하고, (2) 상위 아이템 점수의 겹침 정도(불확실성 포함)와 AVP 불확실성/APU, 추가로 ACE(속성별 누적 엔트로피)로 질문의 정보량을 추정해 추천 타이밍과 질의 대상을 함께 고릅니다. 나아가 추천/질의 응답 생성을 LLM 모듈로 분리해 대화 맥락에 맞는 질문(속성-속성, 값-값 비교 등)을 생성하도록 했습니다.

- **Empirical Impact**: Amazon Reviews 코퍼스에서 33개 데이터셋을 구축해 시뮬레이션과 사용자 연구를 모두 수행했습니다. 시뮬레이션에서는 D2D가 target-finding accuracy를 22.2-29.9% 개선하고, abandonment를 6.6-16.1% 줄이며, 평균 대화를 27.5% 단축하는 성과를 보였습니다. 보완적으로 수행한 사용자 연구에서도 사용자 만족도와 인지된 효율이 유의하게 향상된 것으로 보고되었습니다.



### Co-occurring associated retained concepts in Diffusion Unlearning (https://arxiv.org/abs/2606.24192)
Comments:
          Accepted as a poster at ICLR 2026. Code available at this https URL

- **Prior Approaches**: 확산 모델의 유해 콘텐츠 생성을 줄이기 위한 unlearning 기법이 주목받고 있다. 다만 기존 방식은 목표 개념을 지우는 과정에서 함께 등장하는 정상 개념까지 덩달아 약화시키는 경우가 많다.

- **Core Contribution**: 논문은 함께 동반되지만 보존되어야 할 정상 공존 개념을 CARE(Co-occurring Associated REtained concepts)로 정의하고, 이를 직접 측정하는 CARE score를 제안한다. 또한 목표 개념만 지우면서 CARE를 안정적으로 보호하는 ReCARE(Robust erasure for CARE) 프레임워크를 제시한다.

- **Technical Challenges**: 핵심 난제는 unlearning 중에 발생하는 ‘불필요한 동반 억제’를 정량화하고 학습에 반영해 안정적으로 상쇄하는 것이다. ReCARE는 타깃 이미지에서 추출한 benign co-occurring 토큰으로 CARE-set을 자동 구성하고, 학습 과정에서 이 어휘를 활용해 목표 개념 erasure는 강화하되 CARE 보존은 유지하도록 설계했다.

- **Empirical Impact**: Nudity, Van Gogh style, Tench object 등 다양한 타깃에 대해 실험을 수행했으며, robust concept erasure와 전체 유틸리티, CARE preservation의 균형에서 전반적으로 state-of-the-art 성능을 보였다고 보고한다. 결과적으로 unlearning의 부작용을 줄이면서도 원하는 기능만 선택적으로 제거하는 방향에 실증적 기여를 한다.



### Agon: An Autonomous Large-Scale Omnidisciplinary Research System Built on Prompt Economy (https://arxiv.org/abs/2606.24177)
- **Prior Approaches**: 기존 자율연구·멀티에이전트 연구는 아이디어 생성부터 실험 실행, 원고 작성까지를 한 번에 처리하는 방식이 많아 ‘산출물 생산’에 공수가 쏠리는 문제가 있었다. 또한 자동화된 판단을 위해 프롬프트/코드/스키마를 단계마다 늘리는 경향이 있어 유지보수 비용과 도메인 전이 장벽이 커졌다. 특히 LLM 기반 리뷰는 점수는 잘 오르지만 인간이 받아들이는 과학적 서술·주장 정합성을 벗어나는 실패가 발생했다.

- **Core Contribution**: Agon은 연구 워크플로 안에서 기계가 ‘검증 가능한 것’은 자동으로 확인하고, 그 외의 판단은 사람 과학자가 유지하도록 연구 오케스트레이터를 설계했다. 아이디어-제안-실험-논문을 아티팩트(artifact) 경계로 쪼개고, producer–critic 루프를 반복 재사용하는 ‘연구 팩토리’ 구조가 핵심이다. 이를 통해 새로운 도메인으로 옮길 때도 핵심 아키텍처는 유지하고 입력 컨텍스트만 바꾸는 전이를 지향한다.

- **Technical Challenges**: 대규모 병렬 루프(Massive Parallelism)를 돌리면 어떤 스레드를 어느 순서로 전진시킬지 결정하는 디스패치 부담이 커진다. Agon은 Zero-Code Orchestration으로, 상태를 사람이 읽을 수 있는 컨텍스트로 해석한 뒤 ‘프롬프트 기반 디스패치’로 다음 핸드오프를 선택해 코드·파서·유한상태기계가 비대해지는 문제를 줄였다. 또 논문 개선에서는 LLM 리뷰 점수만 최적화해 인간 친화성을 잃는 ‘종이 괴물’ 실패를 막기 위해, 감사(auditor)로 인간이 읽는 과학적 글쓰기 제약으로 투영(projection)하며 killer reviewer와 area-chair adjudicator로 공격 논리를 통과시킨다.

- **Empirical Impact**: Agon은 10개 이상의 과학 도메인에서 444회 Prompt Economy 루프를 포함해 수천 번의 scientist-coder-auditor 반복을 수행했으며, 사람의 실험 코드 작성 없이도 운영 가능함을 보였다. 동시에 자동화 루프가 포착·수정 가능한 실패와 인간 판단이 필요한 실패를 경계짓고, 이를 severity·fixability·visibility·capability locus 기준으로 분류해 새로운 실패 유형을 드러냈다. 결과적으로 ‘machine scales, human steers’ 패러다임을 실제 배치로 뒷받침하며, 자율연구 인프라의 산업화와 공개(open-source) 배포 가능성을 강화한다.



### CORE-BREW: LLR-Based Soft Decoding for Robust Multi-Bit LLM Watermarking (https://arxiv.org/abs/2606.24163)
- **Prior Approaches**: 기존 multi-bit LLM 워터마킹은 zero-bit 존재 테스트를 넘어 payload를 ECC로 복원하도록 설계됐지만, 대부분 토큰을 green/red 같은 범주로 하드하게 이진화해 디코딩했다. 이 방식은 설계된 acceptance rule로 FPR(오탐률)을 깔끔히 제한할 수 있는 대신, 문맥·시점에 따라 변하는 토큰 확률(신뢰도)을 버려 동의어 치환, 삽입/삭제, 패러프레이징 같은 편집 공격에서 false negative가 늘 수 있다.

- **Core Contribution**: 논문은 CORE-BREW로 블록 단위 BREW에 Constant-hit-rate 임베딩을 확장해, 워터마크 채널을 위치-동일한 신뢰도 스케일로 캘리브레이션한다. 그 결과 각 토큰 범주로부터 closed-form per-token LLR(로그우도비)을 계산해, 기존처럼 하드 디코딩에만 의존하지 않고 multi-bit 복원을 더 견고하게 만든다.

- **Technical Challenges**: 핵심 난제는 LLR을 쓰려면 워터마크 채널이 문맥에 덜 의존적으로 모델링돼야 하는데, 기존 편향(biasing)은 프롬프트/타임스텝마다 의도 리스트의 샘플링 확률이 크게 흔들려 LLR이 ‘부정확하게’ 정의된다는 점이다. CORE-BREW는 타깃 리스트에 할당하는 누적 확률질량을 p⋆로 고정하는 방식으로 채널을 안정화하고, 저엔트로피/저품질 위험 구간에서는 distortion guard로 erasure 처리(LLR=0)해 품질-견고성 트레이드오프를 제어한다; 디코딩은 Strict-Safe(기존 bounded-distance acceptance 영역 보존)와 FPR-Calibrated(지정코드워드 고정 + likelihood 기반 스코어 임계로 FPR–TPR 곡선을 명시적으로 운용) 두 모드로 분리한다.

- **Empirical Impact**: OPT-1.3B, Mistral-7B 등 오픈소스 LLM에서 token-level 편집과 패러프레이징 공격을 평가한 결과, CORE-BREW는 저-FPR 구간에서 분별력을 높이면서 의미 품질(문장 품질 지표)도 기존 베이스라인과 비슷한 수준을 유지했다. 특히 CORE-BREW-Cal이 전반 성능이 가장 좋았고, CORE-BREW-Strict는 더 보수적으로 동작해 ‘엄격한 오탐 제어’가 필요한 시나리오에 대안이 된다.



### Progressive Alignment Objectives for Aligner-Encoder based ASR (https://arxiv.org/abs/2606.24147)
Comments:
          Accepted to Interspeech 2026

- **Prior Approaches**: 기존 E2E ASR에서 정렬(오디오 프레임-토큰 간 단조 정렬)은 CTC, RNN-T, AED(encoder-decoder attention)처럼 서로 다른 방식으로 학습돼 왔다. 특히 Aligner-Encoders는 decoder attention 없이 encoder self-attention 안에서 정렬을 명시적으로 만들지만, 17-layer에서 상위 몇 층에 갑자기 대각(대략적 monotonic) 정렬이 형성되는 경향이 있어 긴 발화에서 학습이 취약해진다.

- **Core Contribution**: 이 논문은 Aligner-Encoders의 ‘late-layer alignment bottleneck’을 완화하기 위해 InterAligner를 제안한다. 중간 층(upper-intermediate)에 더 긴/더 촘촘한 토큰 시퀀스에 대한 intermediate Aligner 목적을 넣어 정렬이 깊이에 따라 점진적으로 형성되게 하고, 최종 층에서는 기존 짧은 토큰 목표로 정교화를 유도한다.

- **Technical Challenges**: 핵심 난제는 encoder 내부에서 정렬이 단계적으로 ‘안정적으로’ 나타나게 최적화를 유도하는 것이다. 이를 위해 중간층에 intermediate CTC loss(InterCTC)를 추가해 early optimization을 안정화했고, intermediate 목적은 one-to-one pairing 조건(U_int≤T'·U_int)에 맞춰 구성되도록 토큰화 길이/개수 제약을 둔다.

- **Empirical Impact**: LibriSpeech(17-layer Conformer-L)에서 final-only Aligner는 test-clean/other WER 5.0/7.8, InterCTC는 3.4/6.0, InterAligner는 3.1/5.6으로 개선됐다. 특히 발화가 긴 구간(>21s)에서 test-clean/other WER이 17.0→11.6, 18.0→13.5로 가장 큰 폭의 향상이 나타나며, attention 시각화도 하위층에서는 정렬이 약하다가 상위층에서 점진적으로 대각 패턴이 강화되는 흐름을 보여준다.



### Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2606.24133)
Comments:
          Our code is at this https URL

- **Prior Approaches**: 기존 데이터 믹싱 연구는 고정된 offline 방식과, 학습 중 도메인 비중을 바꾸는 online 방식으로 나뉜다. offline은 프록시 모델로 최적 믹스(도메인 비율)를 찾아도 학습 전 과정에 같은 레시피를 가정해 LLM 학습의 비정상성(non-stationary)을 놓친다. ODM 계열 online 방법은 bandit이나 actor-critic으로 도메인 가중치를 갱신하지만, 보통 데이터 품질/손실/도메인 간 영향 중 한 관점에 치우쳐 다차원 요구를 충분히 통합하지 못한다.

- **Core Contribution**: 이 논문은 온라인 데이터 믹싱을 위한 새로운 프레임워크 Holistic Data Scheduler(HDS)를 제안한다. HDS는 Soft Actor-Critic(SAC)으로 연속 제어 형태의 강화학습을 구성해, 매 학습 스텝마다 다음 배치의 도메인 샘플링 확률을 동적으로 산출한다. 또한 데이터 품질, 도메인 간 영향, 모델 안정성이라는 세 관점을 함께 넣는 multi-objective 보상 함수를 설계해 단일 관점 최적화의 한계를 보완한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 고차원 정책공간에서 안정적으로 학습 가능한 스케줄러를 만들고, (2) LLM 학습 동역학을 상태로 충분히 반영하며, (3) 서로 다른 성격의 신호를 보상에 균형 있게 통합하는 것이다. HDS는 state를 성능/학습 속도/내부 안정성(샘플 수, 도메인 손실 및 변화량, 특정 레이어 weight norm과 변화량) 벡터로 정의하고, 경량 Transformer-like actor/critic으로 상태-행동 매핑 비용을 줄인다. 보상은 품질(검증 손실 기반), 도메인 간 영향(gradient alignment), 안정성(가중치 변화량 기반)을 가중합해, 학습 초·중·후에 목표가 자연스럽게 전환되도록 구성한다.

- **Empirical Impact**: The Pile에서 HDS는 AC-ODM의 최종 검증 perplexity 수준을 44% 더 적은 학습 iteration으로 도달하며, TPW 정적 베이스라인 대비로도 같은 퍼플렉서티를 57% 적은 스텝으로 달성한다. 최종적으로는 TPW/ODM/AC-ODM보다 각각 13.6%/8.9%/5.3% 더 낮은 검증 perplexity를 보이며, MMLU에서 0-shot 정확도 7.2% 개선을 포함해 일관된 성능 향상을 확인했다. 즉 HDS는 학습 효율(더 빠른 수렴)과 최종 모델 능력(다운스트림 일반화)을 동시에 끌어올리는 데이터 믹싱 정책의 실증적 근거를 제공한다.



### When Top-1 Fails: Calibrating LoRA Monitors for Masked Diffusion LMs (https://arxiv.org/abs/2606.24119)
Comments:
          14 pages, 3 figures. Code and result artifacts: this https URL

- **Prior Approaches**: DLM(Discrete diffusion language model) 계열에서 LoRA/PEFT 미세튜닝은 짧은 실행 구간에서 값비싼 정밀 진단을 피하기 위해 조기 모니터링이 필요하다. 기존에 자주 쓰이던 신호는 denoising 단계에서 관측되는 argmax 토큰 집중도(예: top-1 collapse rate)인데, PEFT 학습 의미가 검증되지 않았다.

- **Core Contribution**: 이 논문은 top-1(또는 top-11) argmax 집중도 기반 붕괴 경고가 DLM-LoRA의 단기 안정성 경고로 전이되지 않음을 대규모로 보여준다. 대신 토큰 쪽 농도 신호가 아니라 LoRA 파라미터 쪽 그래디언트 라우팅을 반영하는 max LoRA gradient norm을, 패밀리별로 보정한 triage(검사 라우팅) 신호로 제안한다.

- **Technical Challenges**: 핵심 과제는 “토큰 집중도”가 실제 collapse 플래그와 매칭되는지(전이성)와, 단기 PEFT에서 의미 있는 조기 지표로 캘리브레이션 가능한지다. 해결책으로 top-1의 pre-equilibrium saturation(최적화 초기에 이미 높은 집중도가 포화되어 학습 안정성과 무관)이 원인임을 토큰-측/파라미터-측(그래디언트 Gini, gradient mass 집중) 비교로 설명하고, max-gradient는 토큰이 아닌 파라미터 라우팅을 샘플링한다는 관점으로 임계값을 패밀리 로컬하게 학습해 사용성을 확보했다.

- **Empirical Impact**: 816개 DLM-LoRA 설정에서 top-1 경고는 816/816마다 발화하지만 실제 collapse는 0/816으로 기록되어 정밀도 0(precision=0)임을 확인했다. 반면 LLaDA-family held-out 평가에서 max-gradient 기반으로 top-decile 최종 손실 구성을 예측했을 때 precision 0.68, recall 0.94, F1 0.79를 얻었고(기준선 all-positive top-11 대비 개선), 임계값은 DLM 패밀리 간 전이되지 않는 대신 단기 검사/라우팅 워크플로우에 실용적으로 쓸 수 있음을 제시한다.



### Exploring Academic Influence of Algorithms by Co-occurrence Network Based on Full-text of Academic Papers (https://arxiv.org/abs/2606.24099)
- **Prior Approaches**: 기존 연구는 특정 알고리즘을 단독으로 성능이나 인용 등으로 평가하는 경우가 많아, 논문들 사이에서 알고리즘이 서로 연결되며 만들어내는 ‘집합적 영향’을 충분히 다루지 못했다. 또한 알고리즘 언급은 인기 지표로 활용되더라도, 네트워크 관점에서의 구조와 시간에 따른 중심성 변화는 상대적으로 덜 탐구됐다.

- **Core Contribution**: 이 논문은 학술 논문의 전체 텍스트를 기반으로 NLP에서 알고리즘 공출현 네트워크를 대규모로 구축하고, 이를 통해 알고리즘 영향력을 네트워크 관점에서 정량화한다. 전체(누적) 네트워크와 연도별 네트워크를 함께 분석하며, 알고리즘이 분야 내에서 어떻게 ‘중심 위치’를 차지하고 변화하는지 보여준다.

- **Technical Challenges**: 핵심 과제는 방대한 본문에서 알고리즘 엔티티를 정확히 추출하고, 공출현을 일관된 규칙으로 네트워크에 반영하는 것이다. 연구진은 deep learning 모델로 알고리즘 엔티티를 추출한 뒤 전역·누적·연도별 공출현 네트워크를 만들고, 여러 centrality 지표로 집단 영향(전체 및 시간 축)을 비교·분석했다.

- **Empirical Impact**: 결과로 네트워크는 약 2년대가량에 걸쳐 점점 더 촘촘해지는 복잡계 특성을 보이며, 전통적 고성능 알고리즘과 서로 다른 연구 시기를 잇는 교차 지점의 알고리즘이 높은 인기·통제·중심성 및 균형 잡힌 영향력을 갖는 경향이 나타났다. 또한 영향이 감소할 때는 먼저 핵심 네트워크 위치를 잃고, 이후 다른 알고리즘과의 연관성이 약화되는 양상이 관찰됐다. 4년대 이상 출판물을 아우르는 최초 규모의 알고리즘 공출현 네트워크 분석으로, 향후 알고리즘·학자·태스크를 연결하는 네트워크 연구의 기반을 제공한다.



### Blockwise Policy-Drift Gating for On-Policy Distillation (https://arxiv.org/abs/2606.24084)
Comments:
          8 pages

- **Prior Approaches**: 온폴리싱 증류(OPD)는 학생이 실제로 만든 롤아웃의 접두사에 대해 교사가 피드백을 주어, 정적 교사 트레이스만 쓰는 방식의 분포 불일치를 줄인다는 장점이 있다. 다만 장기 추론에서는 sampled-token OPD가 취약해지며, 이를 보완하기 위해 Teacher-TopK/LSM처럼 토큰 하나 대신 교사의 truncated support로 국소 신호를 넓히는 방법들이 제안됐다. 또 stale rollout이나 비동기 업데이트를 다루는 freshness-aware 계열 연구도 있으나, 본 논문이 겨냥한 것은 비동기 시스템이 아니라 롤아웃 재사용 시 나타나는 old-current 정책 드리프트라는 다른 불일치 원인이다.

- **Core Contribution**: 본 논문은 rollout reuse 상황에서 재사용된 샘플 경로 위의 old-current 학생 log-probability shift를 ‘플러그인’ 가중치로 OPD 손실에 반영하는 sampled-path policy-drift gating을 제안한다. teacher 타깃, teacher top-K support, 롤아웃 분포 자체는 바꾸지 않고, detach된 mean-normalized 게이트로 base OPD(및 LSM)의 위치별 손실 가중치만 조절한다. 특히 게이트를 토큰 단위가 아니라 블록/스팬 단위로 집계해 noisy한 신호를 완충하는 절충점을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 old-current shift가 토큰 수준에서는 신호가 흔들리는데, 이를 너무 거칠게(시퀀스 단위) 만들면 장기 추론에서 지역성이 사라진다는 점이다. 논문은 같은 샘플 경로에서 behavior student와 current student의 로그확률 차이를 계산하되, teacher 신뢰나 divergence 같은 추가 신호 없이도 쓸 수 있도록 detached soft gate로 설계했다. 또한 fixed-block(예: 64토큰) 또는 newline-delimited span으로 shift를 국소 집계해 해당 구간의 위치 손실에 동일 가중치를 브로드캐스트함으로써 중요도 가중치의 변동성을 줄였다.

- **Empirical Impact**: Qwen3 수학 벤치마크에서 200-step 고정 예산 조건 하에 pass@8을 주요 지표로 평가했으며, fixed 64-token block gating은 sampled-token OPD의 mean pass@8을 0.4978에서 0.5160으로 끌어올렸다(AIME24/AIME25/MATH500/AMC23 평균). 또한 Teacher-TopK/LSM과 결합 시 Block64가 학습된 변형들 중 4개 벤치마크 평균 pass@8에서 최상 성능을 보였다. 결과는 rollout 재사용 하의 local old-current policy drift가 실용적인 제어 신호가 될 수 있고, 그 신호의 granularity는 문제 수준 solve-rate 기준으로 블록형(특히 64토큰)이 기본값으로 유망하다는 점을 시사한다.



### VieSpeaker: A Large-Scale Vietnamese Speaker Recognition Dataset Beyond Visual Dependency (https://arxiv.org/abs/2606.24066)
Comments:
          5 pages, 1 figure, 6 tables, Accepted at Interspeech 2026

- **Prior Approaches**: 기존 베트남 speaker recognition 연구는 VoxCeleb2, CN-Celeb2처럼 대규모 데이터로 성능이 크게 올랐지만, 베트남은 상대적으로 데이터가 부족했다. 또한 기존 데이터 구축은 face detection/face tracking에 의존해 영상에 얼굴이 보이는 경우에만 화자 라벨을 붙이기 쉬워, vlog·익명 팟캐스트·전화 통화처럼 시각 단서가 약한 오디오까지 확장하기 어려웠다.
그 결과 도메인 불일치나 cross-domain 평가에서 성능이 떨어질 수 있다는 문제도 반복해서 관찰됐다.

- **Core Contribution**: 이 논문은 얼굴 단서를 쓰지 않는(face-independent) 베트남 화자 데이터 구축 파이프라인을 제안하고, 그 결과물로 VieSpeaker를 공개했다. VieSpeaker는 대본(transcript)과 영상 메타데이터(제목·채널명·설명)를 기반으로 LLM reasoning으로 SPEAKER_ID를 실제 화자 정체성으로 추론·검증하는 방식이다.
약 902시간, 4,715명의 베트남 화자를 포함해 기존 베트남 코퍼스 대비 스케일과 음향 다양성을 크게 확장한 것이 핵심 기여다.

- **Technical Challenges**: face 없이 정체성을 추론하려면, diarization으로 생긴 익명 화자(SPEAKER_ID)를 잘못 합치거나 분리하지 않는 evidence 기반 추론이 필요하다. 논문은 화자별로 최대 7개 초기 transcript 세그먼트를 샘플링하고, temperature=0.0, top_k=1 같은 controlled decoding으로 LLM의 비결정적 추론을 줄여 오탐을 완화하며, 세그먼트 간 통합으로 과분할된 diarization을 암묵적으로 보정한다고 설명한다.
또한 표기 변형(예: 호칭 prefix 제거, 철자/대소문자 차이)을 LLM 기반으로 정규화하고, 유사도 기반 agglomerative clustering(ECAPA-TDNN 임베딩)과 인간 검증, IQR 기반 이상 발화 제거로 데이터 품질을 관리했다.

- **Empirical Impact**: 실험에서 VieSpeaker-T로부터 학습한 모델은 기존 베트남 데이터로 학습한 모델보다 더 어려운 프로토콜에서 큰 폭의 EER 감소를 보였고, pretraining(예: VoxCeleb2 또는 VieSpeaker로 사전학습)에서도 성능 이득이 관찰됐다. 특히 VieSpeaker pretraining은 VoxCeleb2 대비 베트남 벤치마크에서 상대적으로 더 낮은 EER을 달성하며, 데이터가 cross-dataset 일반화와 downstream 적응에 기여할 가능성을 보여준다.
총 4,715명의 disjoint 평가 스플릿과 Easy/Hard 프로토콜 설계 덕분에 재현성과 통계적 신뢰성도 높이며, 이 데이터셋은 커뮤니티에 Hugging Face로 공개되어 베트남 오디오 기반 화자 인식 연구의 새로운 출발점을 제공한다.



### RoPE-Aware Bit Allocation for KV-Cache Quantization (https://arxiv.org/abs/2606.24033)
Comments:
          Preprint. Code available at this https URL

- **Prior Approaches**: 기존 low-bit KV-cache quantization은 대체로 키를 flat vector로 보고 재구성 오차(또는 outlier) 최소화 관점에서 비트 단위를 배분했다. 하지만 RoPE에서는 key가 미래 attention logit에 기여할 때 2D frequency block들의 위치 의존 합으로 분해돼, 동일 오차라도 logit 영향이 블록마다 크게 달라질 수 있다. 그 결과 RoPE-비의존적(혹은 uniform) 배분은 영향이 큰 block을 과소보호하고 덜 중요한 block에 비트를 낭비하는 문제가 나타난다.

- **Core Contribution**: Block-GTQ는 RoPE-aware하게 key-cache를 ‘logit 보존’ 관점의 블록별 bit-allocation 문제로 재정의한다. 각 레이어·각 KV head에서 RoPE block 단위 에너지 점수(Q/K 통계 기반)를 계산하고, TurboQuant-MSE(TQ-MSE)의 4^{-b} 오차-율 법칙과 함께 marginal gain 기준으로 정수 비트폭을 greedy하게 할당한다. 또한 동일 비트폭을 받은 RoPE block들을 묶어 TQ-MSE 인코더를 재사용함으로써 구현 복잡도를 낮춘다.

- **Technical Challenges**: 핵심 기술 난제는 (1) paired query-key 항을 직접 쓰지 않고도 블록별 logit 민감도를 추정하는 label-free 점수 설계와 (2) 할당을 실제 인코딩 단계와 매끄럽게 연결하는 것이다. Block-GTQ는 AM-GM 기반 energy score로 RoPE-block 중요도를 과대추정하되 과소추정하지 않는 형태로 구성해 불확실성을 완화하고, 블록별 목표함수 Σ s_i·4^{-b_i}에 대해 greedy 배분이 최적이 되도록 구성한다. 추론에서는 packed-cache 서빙 경로를 구현해 HBM 대역폭을 줄이면서도 디코딩된 키를 커널 내부에서만 사용하도록 설계했다.

- **Empirical Impact**: 진단용 10개 모델 패널에서 Block-GTQ는 K-only quantization(2~3 b/dim) 조건에서 RoPE query-key logit MAE를 32~80% 줄였고, 레이어 비교 367/367에서 uniform TQ-MSE를 이겼다. 다운스트림에서는 K2V2 예산에서 Llama-3.1-8B-Instruct의 NIAH 평균(6개 태스크)을 70.6→97.4, LongBench-EN 평균을 36.87→53.31로 크게 끌어올렸다. 또한 Qwen2.5-3B-Instruct에서 packed K3V3는 단일 H800 기준 KV 캐시를 3.24× 더 압축하면서 fp16 FlashAttention2 대비 128K에서 1.34× 빠르고 피크 메모리를 56.31GB→19.85GB로 낮춰 256K/512K에서는 fp16 OOM을 피하는 등 실사용 영향이 확인됐다.



### Reinforcement Learning Towards Broadly and Persistently Beneficial Models (https://arxiv.org/abs/2606.24014)
Comments:
          Blog: this https URL

- **Prior Approaches**: 기존 연구는 misalignment(불일치)가 특정 도메인에서 학습되면 다른 도메인까지 넓게 전이될 수 있음을 보여줬다. 이 때문에 RL의 reward hacking, deception, 안전 훼손 같은 문제가 “좁은 학습 신호”에서 시작해도 전반적 실패 양상으로 확산될 수 있다는 우려가 커졌다. 한편 beneficial behavior(유익한 행동) 쪽으로의 일반화는 상대적으로 덜 검증돼 있었다.

- **Core Contribution**: 이 논문은 truthfulness, fairness, risk awareness, corrigibility 같은 beneficial traits(유익한 성향)를 목표로 하는 multi-domain 데이터셋을 구축하고, 그 위에서 alignment-focused RL을 수행해 전이 일반화를 측정한다. 또 학습 분포 밖의 50개+ 독립 벤치마크에서 alignment와 benefits를 함께 평가해, 단순 과적합이 아니라 “폭넓은 개선”인지 확인한다. 마지막으로 유해한 프롬프트 유도와 harmful finetuning 후에도 정렬이 유지되는 persistence를 체계적으로 다룬다.

- **Technical Challenges**: 핵심 난제는 (1) 특정 과업이 아니라 모델 차원의 성향이 alignment 전이의 원인이 되는지, (2) 단기적으로 유용해 보이는 행동이 다른 failure mode에서 실제로 막히는지, (3) 반대로 refusal 증가 같은 부작용이 성과를 왜곡하지는 않는지였다. 논문은 12개 도메인에서 15개 fine-grained beneficial traits를 합성 대화로 구현하고, trait description·domain description 조건부 생성 및 실패 회피 기준으로 평가/학습 신호를 설계했다. 이후 compute-matched baseline 대비, 5%만 유익한 성향 데이터로 끼워 넣는 ablation과 health-only/health·science 제외 같은 통제 실험으로 원인 분리를 시도했다.

- **Empirical Impact**: beneficial trait RL은 compute-matched baseline 대비 out-of-distribution alignment·safety·benefits 평가에서 80% 이상 벤치마크에서 성능을 개선했으며 평균 개선폭도 확인됐다. 특히 health 도메인에만 5%의 유익한 RL 데이터를 넣었는데도 non-health 평가에서 reward hacking, deception, general misalignment 등이 함께 개선되는 out-of-distribution alignment transfer를 관찰했다. 또한 harmful 의료 persona 프롬프트나 bad medical advice 유도 finetuning 상황에서 성능 저하가 더 작아져 persistence 향상 근거를 제시하되, 세부 원인 분리는 추가 연구가 필요하다고 밝혔다.



### Neuro-Symbolic Drive: Rule-Grounded Faithful Reasoning for Driving VLAs (https://arxiv.org/abs/2606.23938)
- **Prior Approaches**: 기존 Driving VLA는 Chain-of-Thought(CoT)처럼 자연어로 설명을 생성할 수 있지만, 설명이 실제 행동을 좌우하는 인과적 단계와 맞물리지 않을 수 있습니다. 또한 많은 학습 라벨이 사람/다른 모델/후처리 파이프라인에서 만들어져 “생성된 말”과 “결정된 궤적” 사이에 supervision mismatch가 생긴다는 문제를 지적합니다.

- **Core Contribution**: 이 논문은 Neuro-Symbolic Drive를 제안하며, VLA의 CoT를 사후 정렬(post-hoc alignment)이 아니라 rule-based planner의 실행에서 직접 뽑은 reasoning trace로 감독합니다. rule 기반 플래너는 안전 제약 활성화→후보 탐색→최종 궤적 선택을 수행하는 실행 가능한 추론 엔진이므로, 그 내부 의사결정 상태를 구조화해 Qwen3.5-4B를 fine-tuning하는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 서로 다른 플래너의 내부 상태/어휘를 그대로 두면 의미가 일관되지 않고, raw trace에는 구현 디테일이 섞여 그대로 모방하면 재사용 가능한 운전 로직을 배우지 못한다는 점입니다. 이를 위해 planner 상태를 route·constraint·candidate·decision의 4-슬롯 스키마로 추상화해 rule-grounded reasoning으로 직렬화하고, 시뮬레이션에서 3종 플래너(IDM, IDM-MOBIL, PDM-Closed)의 실행 trace와 궤적을 한 쌍으로 수집한 뒤 시나리오 계열별로 가장 신뢰도 높은 teacher를 선택해 학습합니다.

- **Empirical Impact**: 시뮬레이터 기반 NSD-Sim 벤치마크에서 rule-grounded reasoning은 성능을 일관되게 개선했습니다. 3-camera 조건에서 ADE@3s는 0.47→0.26, miss rate는 8.30%→6.40%로 줄었고, 8-camera 조건에서도 ADE@3s 0.54→0.26, miss rate 10.13%→5.99%를 기록했습니다. 결론적으로 “더 긴 설명”이 아니라 planner 실행에 구조적으로 결합된 의사결정 의미가 궤적 생성 학습 신호로 작동함을 보여주며, Driving VLA의 디버깅/감사 가능성(행동-근거 동기화)에도 의미가 큽니다.



### Mind the Heads: Topological Representation Alignment for Multimodal LLMs (https://arxiv.org/abs/2606.23885)
- **Prior Approaches**: 기존 representation alignment 연구는 MLLM 내부 표현을 비전 인코더 표현에 맞추는 방식으로 비주얼 능력을 끌어올리지만, 대개 LLM의 고정된 한 층(예: 중간층)만 정해 정렬합니다. 또 layer-level 정렬은 Transformer의 미세한 구조(어텐션 head별 역할)를 충분히 반영하지 못해, 언어 모델의 기존 우선순위와 충돌하거나 최적의 선택을 놓칠 수 있다는 한계가 제기됩니다.

- **Core Contribution**: 이 논문은 Head-Wise Representation Alignment(HeRA)를 제안해, LLM의 attention head 단위로 비전-언어 표현 정렬을 수행합니다. MKNN(상호 k-최근접 이웃) 점수를 진단으로 사용해 정렬할 head를 선택하고, 대비(contrastive) 목적함수로 비전 인코더가 만드는 로컬 이웃(topological neighborhood) 구조를 차분하게 흉내 내도록 학습시킵니다.

- **Technical Challenges**: 핵심 난관은 MKNN이 k-최근접 이웃 인덱스에 의존해 미분 불가능하다는 점인데, 논문은 이를 differentiable proxy인 multi-target InfoNCE 형태의 대비 손실로 대체합니다. 또한 head를 무작정 탐색하기 어렵기 때문에, 멀티모달 학습 전 순수 텍스트 상태에서 head별 MKNN 정렬 점수를 계산해 상위/하위 후보를 선별하고, HeRA는 그중 일부 head에만 정렬 손실을 적용합니다.

- **Empirical Impact**: 18개 벤치마크(Cambrian)와 여러 hallucination 벤치마크에서 HeRA는 vision-centric 과제에서 일관된 성능 향상을 보이며, 종종 시각 환각을 줄이는 정규화 효과도 확인됐습니다. 특히 “가장 덜 정렬된(worst) head를 맞추는 전략”이 가장 큰 이득을 주었고, 다양한 MLLM(LLaVA 프레임워크, 파라미터 스케일 포함)에서 General/Knowledge/OCR 성능 저하 없이 안정적으로 개선되었습니다.



### ESBMC-PLC+: A Unified IEC~61131-3 Formal Verification Framework as a PLCverif Successor (https://arxiv.org/abs/2606.23870)
Comments:
          21pages

- **Prior Approaches**: PLCverif는 CERN에서 개발돼 2019년부터 운영 중인 성숙한 오픈소스 PLC 형식검증 도구로, SCL/STL 입력을 CBMC 등으로 검증한다. 다만 Ladder Diagram(LD)을 직접 받지 못해 변환 과정에서 충실도(semantic fidelity) 위험이 생기고, CBMC 기반은 본질적으로 bounded proof에 머물러 unbounded safety proof에는 제약이 있다.
또한 그래픽 PLCopen XML 기반 LD에서 타이머/카운터/엣지 트리거 같은 function block이 포함된 경우를 완전하게 처리하지 못해 검증 가능한 범위가 제한됐다. ESBMC-PLC와 ESBMC-GraphPLC는 unbounded proof와 LD 입력을 진전시켰지만, ST/SCL과 그래픽 LD의 function block 상태 의미론까지는 각각 공백이 남아 있었다.

- **Core Contribution**: 이 논문은 ESBMC-PLC+라는 단일 프레임워크로 textual LD, graphical LD, ST/SCL의 세 입력 포맷을 모두 ESBMC 백엔드 하나로 통합한다. 핵심은 (1) MATIEC로 ST/SCL을 C로 컴파일한 뒤 ESBMC의 k-induction이 다룰 수 있게 스캔-사이클 래퍼와 property 주입을 하는 것, (2) 그래픽 LD에서 TON/TOF/TP, CTU/CTD, R_TRIG/F_TRIG function block의 scan-cycle 상태를 GOTO IR에 영속 변수로 모델링하는 것이다.
그 결과, k-induction을 통해 모든 scan count에 대한 unbounded safety proof를 제공하면서 PLCverif 수준의 입력 커버리지를 오픈소스로 달성한다.

- **Technical Challenges**: 주요 과제는 IEC 61131-3의 PLC 실행 의미론(특히 scan cycle, 타이머/카운터 누적, 엣지 검출의 상태 지속)을 ESBMC의 GOTO IR 모델에 정확히 사상하는 것이다. ESBMC-PLC+는 ST에서 MATIEC가 만든 C 코드를 scan-cycle while 루프로 감싸고 nondeterministic 입력(오픈월드 센서 모델)을 삽입해 ESBMC가 모든 실행을 탐색하도록 만들었다.
그래픽 LD에서는 기존 DFS resolver가 function block 노드를 의미 있게 표현하지 못했던 한계를 넘어, 해당 노드 등장 지점에 상태 선언·상태 업데이트 코드를 생성해 출력 핀(Q/edge 결과)을 다음 래더 평가의 논리 항으로 사용되게 했다. 타이머/카운터 값은 스캔 횟수 단위로 해석되도록 ET/프리셋 변환 규칙을 맞췄으며, 지원 범위 밖의 vendor-specific block은 보수적 over-approximation으로 처리한다.

- **Empirical Impact**: 8개 벤치마크를 포함한 실험에서 ESBMC-PLC+는 PLCverif와 동등한 수준의 입력 커버리지를 제공하면서 더 강한 unbounded 보장(k-induction safety proof)을 달성했다고 보고한다. 특히 타이머가 많은 프로그램에서 nuXmv의 BDD 백엔드 대비 400–2,000배 빠른 성능을 보였고, nuXmv가 120s 타임아웃에 도달한 경우에도 ESBMC-PLC+가 증명을 완료했다.
또한 비교/평가에서 안전/위험 분류의 정확성은 false positive와 regression 없이 유지되는 것으로 제시돼, 현업 PLC 검증에서 제약이 컸던 LD+타이밍 semantics 공백을 실질적으로 줄였다는 의미가 있다.



### From Task-Guided Conversational Graphs to Goal-Oriented Dialogue Runtimes (https://arxiv.org/abs/2606.23797)
Comments:
          21 pages, 7 figure, 10 tables

- **Prior Approaches**: 기존 LLM 오케스트레이션은 “다음에 어떤 에이전트/툴/노드를 실행할지”에 강하지만, 여러 목표가 동시에 살아 있는 대화에서 목표의 연속성(objective continuity)을 자동으로 보장하긴 어렵다. 프로세스 기반(워크플로 그래프, FSM)은 실행 위치의 연속성은 잘 유지하지만, 한 목표가 멈춘 채 다른 목표가 개입·무효화되는 상황까지 안정적으로 복원하기엔 한계가 있다. 또한 에이전트 라우팅이나 채팅 히스토리/실행 그래프 위치만으로는 “어떤 사용자 목표가 살아남는지”를 의미론적으로 구분하기 어렵다.

- **Core Contribution**: 이 논문은 Goal-Oriented Dialogue Runtime(GODR)을 제안하며, 프레임워크 중립적인 계층에서 목표(goals)와 생명주기(lifecycle), 태스크 프레임(task frames), 무효화 규칙(invalidation rules), 재개 계약(resumption contracts)을 런타임의 1급 객체로 취급한다. GODR은 그래프 런타임/에이전트/툴이 담당하는 “bounded execution”을 대체하지 않고, 대화 중단·지연·수정·무효화에도 목표가 끊기지 않게 만드는 “objective continuity” 전용 런타임 경계(boundary)를 명확히 한다. 특히 GC-4(목표 복잡도) 영역에서 필요하다고 주장하며, 단순 스택/트리 모델만으로는 공유 제약·의존·무효화가 얽힌 대화를 다루기 어렵다고 정리한다.

- **Technical Challenges**: 핵심 기술 과제는 “새 사용자 발화가 들어왔을 때 어떤 목표 수준 연산(continue/revise/push/switch/pop/resume/cancel/escalate/reset)을 수행할지”를, 실행 라우팅과 분리해 결정하는 것이다. 논문은 목표 구조를 라벨된 directed graph(DAG 포함)로 모델링하고, 재개 가능 상태는 resumption contract가 비어 있지 않다는 등 런타임 불변식(invariants)과 함께 무효화 안전성(invalidation safety)을 체크하도록 설계한다. 또한 Goal Policy가 LLM 라우터처럼 무제한으로 동작하지 않도록, typed state와 가드·심볼릭 제약으로 허용 가능한 연산 집합을 마스킹/필터링하는 하이브리드 신경-상징 접근을 제안한다.

- **Empirical Impact**: 이 논문은 실험 성능을 “검증된 수치”로 주장하기보다, Multi-Objective Interruptible Dialogue Problem을 정식화하고 평가를 위한 아젠다와 기준, 베이스라인 선택 방법론을 제시하는 시스템 논문 성격을 갖는다. 따라서 즉시 실측 개선을 보장하진 않지만, 기존 오케스트레이션 프레임워크가 강한 실행 계층과 달리 목표 연속성 계층이 애플리케이션 아키텍처에서 누락되기 쉽다는 문제의식을 명확히 한다. 목표 수명주기·무효화·재개 계약을 런타임으로 외재화한다는 관점은 향후 복잡한 멀티도메인 대화 벤치마크/평가 설계로 이어질 수 있는 의미가 있다.



### EvidenceLens: A Claim-Evidence Matrix for Auditing Financial Question Answering (https://arxiv.org/abs/2606.23724)
- **Prior Approaches**: 기존 금융 QA는 답변 전체를 한 덩어리로 생성한 뒤 인용을 덧붙이는 방식이 많아, “어떤 문장은 사실이고 어떤 문장은 추론인지” 같은 세분화가 약하다. 또한 텍스트·표·차트가 섞인 근거를 확인할 때, 채팅 UI는 지지/누락/모순을 한눈에 구분하기 어렵다.

- **Core Contribution**: EvidenceLens는 금융 QA를 주장-근거 정렬(claim-evidence alignment) 문제로 보고, 답변을 원자 단위 atomic claims로 쪼개 근거를 연결한다. 핵심 산출물은 multimodal claim-evidence matrix로, 주장별 커버리지·모순·모달리티 불균형을 시각적으로 드러내 검증 부담을 줄인다.

- **Technical Challenges**: 문장형 답변을 주장 타입(값/비교/추세/설명)과 텍스트·표·차트의 증거 단위로 정규화하고, 근거의 약한 지지와 실제 모순을 구분하는 정렬이 필요했다. 논문은 가벼운 멀티모달 정렬 파이프라인과 polarity test 기반의 충돌 탐지, 그리고 JSON 기반 아티팩트 스키마로 동일 입력에서 재현 가능한 시각 상태와 리뷰 우선순위를 구성한다.

- **Empirical Impact**: 두 가지 리포트 시나리오에서 EvidenceLens는 “그럴듯한” 답변이 실제로는 일부 주장만 지지하고 일부는 미세한 각주나 정량 근거와 충돌하는 패턴을 빠르게 보여줬다. 결과적으로 전통적 채팅이 평탄화하던 과신·설명형 점프·단일 모달리티 의존 같은 실패 모드를 분석가가 재검증하도록 유도하며, 금융 감시 환경에서 더 감사 가능한 근거 제시 방향을 제안한다.



New uploads on arXiv(cs.IR)

### PETRA: Transforming Web Text for Petroleum-Engineering Domain Adaptation (https://arxiv.org/abs/2606.24346)
- **Prior Approaches**: 기존 석유공학 검색 적응은 주로 소규모 도메인 데이터나 일반 대규모 웹 코퍼스에 의존했지만, 두 경우 모두 라벨 희소성과 잡음(중복·OCR 아티팩트·약한 메타데이터) 문제로 인해 도메인 relevance를 안정적으로 학습하기 어려웠습니다. 또한 공개 검색 벤치마크는 품질 측정은 돕지만 석유공학용 검색 적응에 필요한 쿼리-패시지 관련성 감독(supservion)을 제공하지 못했습니다.

- **Core Contribution**: 이 논문은 PETRA( Petroleum Engineering Text for Retrieval Adaptation )라는 대규모 데이터셋과 파이프라인을 제안해, 잡음이 많은 공개 웹 텍스트를 정제된 도메인 코퍼스와 합성 감독으로 변환합니다. dense retrieval(임베딩 1단계)과 reranking(크로스 인코더 2단계) 모두에 대해 학습용 쿼리-패시지 신호를 만들고, 실제 배포 RAG 스택의 추론 후보 분포를 반영하도록 학습 레이블을 설계했습니다. 

- **Technical Challenges**: 핵심 난제는 (1) 도메인 문맥이 표면 단어 중첩만으로는 구분되지 않아 관련성 라벨이 희소하고, (2) 웹 데이터의 잡음이 모델 적응을 방해한다는 점입니다. 이를 위해 고-recall energy-domain classifier(테스트 정확도 98.4%)로 오프도메인을 게이트하고, 청크 수준 검증·중복 제거·chunk-grounded query 생성·LLM hard negative 생성·retrieval-mined 후보 리스트에 teacher 스코어를 입혀 teacher-scored candidate 형태로 재포장합니다. 특히 합성 라벨의 “생성 정확도”만으로는 성능이 안 오르며, 추론 시 후보 분포와 맞춘 teacher-scored candidate 학습 패키징이 중요하다는 실패 사례도 함께 보고합니다.

- **Empirical Impact**: PETRA 기반 적응은 1단계에서 in-domain SOP nDCG@10을 0.703에서 0.763으로 끌어올리고, score fusion 전략으로 out-of-domain 보존도 함께 달성합니다. reranker adaptation은 Earth Science 벤치마크에서 상대 44% 향상, 6개 reasoning-intensive 패널에서도 상대 23% 개선을 보였으며, 후보 리스트 기반 학습이 모든 평가 표면에서 일관된 이득을 만든다고 정리합니다. 또한 seed 분산 없는 단일 실행 보고와 SOP의 독점 구성(재현 제한), 지연·온라인 A/B 미측정 등 한계도 명확히 두었습니다.



### Dialogue to Discovery: Attribute-Aware Preference Elicitation for Conversational Product Search Assistants (https://arxiv.org/abs/2606.24194)
- **Prior Approaches**: 기존 온라인 쇼핑 보조는 키워드 기반 검색이나 facet/필터로 후보를 줄이지만, 남은 후보가 여전히 많거나 반대로 해답이 없을 수 있습니다. 또한 CPS로 LLM을 쓰더라도 zero-shot 추천은 대화가 효율적이지 않거나(무분별한 질의/추천), 화면 제약 속에서 대화 피로와 이탈이 커질 수 있습니다.

- **Core Contribution**: 이 논문은 Dialogue to Discovery(D2D)라는 속성(attribute) 중심 선호 질의·추천 프레임워크를 제안합니다. D2D는 대화 단계마다 “언제 추천할지”와 “어떤 속성/값을 물을지”를 카탈로그 내 속성 분포와 AVP(Attribute-Value Profile) 불확실성을 근거로 동적으로 결정해, 성급하거나 엇나간 추천을 줄이는 것을 목표로 합니다. 또한 사용자 이탈을 고려한 patience 모델을 함께 도입해 대화 품질과 효율을 함께 최적화합니다.

- **Technical Challenges**: 핵심 기술 과제는 제한된 화면에서 소수 후보만 보여줘야 하는 상황에서, 잘못된 추천을 일찍 내리면 engagement가 무너진다는 점입니다. D2D는 (1) 검색된 후보 부분집합에서만 선호를 갱신하고, (2) 상위 아이템 점수의 겹침 정도(불확실성 포함)와 AVP 불확실성/APU, 추가로 ACE(속성별 누적 엔트로피)로 질문의 정보량을 추정해 추천 타이밍과 질의 대상을 함께 고릅니다. 나아가 추천/질의 응답 생성을 LLM 모듈로 분리해 대화 맥락에 맞는 질문(속성-속성, 값-값 비교 등)을 생성하도록 했습니다.

- **Empirical Impact**: Amazon Reviews 코퍼스에서 33개 데이터셋을 구축해 시뮬레이션과 사용자 연구를 모두 수행했습니다. 시뮬레이션에서는 D2D가 target-finding accuracy를 22.2-29.9% 개선하고, abandonment를 6.6-16.1% 줄이며, 평균 대화를 27.5% 단축하는 성과를 보였습니다. 보완적으로 수행한 사용자 연구에서도 사용자 만족도와 인지된 효율이 유의하게 향상된 것으로 보고되었습니다.



### ChartWalker: Benchmarking the Cross-Chart RAG Task (https://arxiv.org/abs/2606.23997)
- **Prior Approaches**: 기존 Cross-Chart RAG 벤치마크는 표 중심이거나, 차트에서 핵심 포인트를 뽑아 질문을 만드는 방식이 많았다. 이 접근은 질의-증거 간 어휘 중복을 유발해 추론 고리가 논리적으로 깨지기 쉽고, 멀티홉에서 주제/대상 참조가 미끄러지며 잘못된 계산으로 이어질 수 있다. 또한 KG 기반 QA 생성도 무작위 워크/단순 PageRank로 경로를 만들다 보니 그럴듯한 의미 연속성과 그라뉼러리(정보 수준) 통제가 약하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 차트에 특화된 Cross-Chart RAG 과제를 더 어렵고 논리적으로 검증 가능하게 만들기 위한 ChartWalker 프레임워크를 제안한다. 차트에서 엔티티-관계를 추출해 그라뉼러리별 층을 보존하는 hierarchical knowledge graph를 구성하고, 이를 바탕으로 의미적으로 일관된 multi-hop reasoning path를 생성해 QA의 근거 체인을 명시적으로 제공한다. 또한 이를 통해 ChartWalker-Bench(총 564개 QA)를 공개하고, 비교 분석을 돕는 VLM 기반 에이전트 ChartWalker-Agent(기준선)도 함께 제시한다.

- **Technical Challenges**: 핵심 기술 문제는 (1) 차트가 정보가 조밀하지만 약하게 구조화돼 있어 엔티티/관계를 신뢰성 있게 계층화해야 하고, (2) 무작위 경로 샘플링이 의미 드리프트를 만들지 않도록 경로의 주제 연속성과 그라뉼러리를 동시에 제어해야 한다는 점이다. 이를 위해 VLM 추출기로 엔티티에 granularity level을 부여하고 intra-level/ inter-level 엣지를 구분해 계층형 KG를 만든 뒤, 수정된 PageRank로 시작 anchor를 고르고 다음 홉은 semantic topic coherence와 granularity 전이를 제약하는 structure-aware sampling으로 선택한다. 마지막으로 질문은 자기완결적으로 생성되도록 하고(대명사 등 암묵 참조 제거), 증거 기반 검증을 통과해야만 벤치마크에 포함되게 설계했다.

- **Empirical Impact**: 평가 결과, 주요 RAG 패러다임 전반에서 성능 격차가 크게 나타나며 벤치마크 난도가 실제로 반영됨을 보여준다. 최고 성능 모델도 cross-chart 문제에서 정답률 64% 수준에 그쳤고, 복잡 추론(Complex Reasoning)에서는 대부분의 설정이 30% 미만으로 급락해 장기 멀티홉 검색·통합의 어려움을 드러냈다. 또한 Complex Reasoning은 평균 3.12개의 소스 차트와 5.14개의 추론 홉을 요구하며, ChartWalker-Agent는 동적 탐색으로 정적 파이프라인의 한계를 진단하는 기준선 역할을 수행해 향후 에이전트형 멀티모달 RAG 설계를 자극할 것으로 기대된다.



### Unified Multi-Task Relevance Modeling for E-Commerce: Comparing Task Routing Architectures Across LLMs and Cross-Encoders (https://arxiv.org/abs/2606.23919)
Comments:
          Accepted at E-commerce workshop, SIGIR 2026

- **Prior Approaches**: 전자상거래 관련성(relevance) 모델은 보통 쿼리-상품 매칭 등 관계 유형별로 별도 모델을 학습해 왔다. 이 방식은 태스크 간 지식 전이가 막히고, 서로 다른 기준으로 산출된 관련성 신호가 downstream에서 불일치하게 만들 수 있다. 또한 멀티태스크 학습을 적용하더라도 태스크 라우팅(태스크 정체성 전달) 설계가 인코더-디코더 모델에 동일하게 적용된다는 가정이 많아, 모델 계열별로 발생하는 비대칭 오류는 충분히 분석되지 않았다.

- **Core Contribution**: 이 논문은 전자상거래의 6가지 엔터티-쌍 관계를 하나의 단일 관련성 모델로 통합하는 멀티태스크 프레임워크를 제안한다. 공통 3단계 관련도 스케일(irrelevant/partially relevant/relevant)에서 Hard parameter sharing으로 6개 태스크를 함께 학습하되, 태스크 정체성을 공유 모델에 어떻게 전달하는지를 라우팅 아키텍처로 비교한다. 특히 private transformer layers를 갖는 multi head with private layers(MHP)와 텍스트 prefix·multi head 분류기 구조를 함께 평가해, 태스크 정체성 인코딩이 encoder와 decoder-only에서 다르게 작동함을 핵심 통찰로 제시한다.

- **Technical Challenges**: 공유 모델이 서로 다른 데이터 규모와 의미 요구, 심지어 상충하는 학습 신호를 동시에 처리해야 하는 점이 큰 기술적 난제다. 이를 위해 논문은 6개 관계를 같은 3-class 분류 문제로 재정의하고, LoRA-adapted decoder-only LLM과 fully fine-tuned cross encoder 양쪽에서 동일한 라우팅 비교를 수행한다. 그 결과 text-prefix routing은 decoder-only에서 특히 성능이 급락하며, cross encoder는 상대적으로 견고한 encoder–decoder 비대칭이 나타나는데, 이 문제는 private transformer layers로 복원되는 방식(MHP)으로 해결한다.

- **Empirical Impact**: 453K 테스트 예제 기준으로 MHP 앙상블이 89.96% 정확도를 달성해 모든 구성 중 최고 성능을 보였다. 또한 text prefix를 제거하면 decoder-only LLM이 붕괴하는 반면 cross encoder는 거의 영향이 없고, MHP의 private layer 2개가 이 붕괴를 사실상 회복시킨다는 실험 결과를 제시한다. 멀티태스크 학습은 특히 저자원 태스크에서 최대 14% 향상을 보였으며, 전반적으로 태스크 통합 모델 설계 시 private-layer 라우팅과 encoder–decoder 비대칭 고려가 중요하다는 실무적 가이드를 제공한다.



### Scaling Dense Retrieval with LLM-Annotated Training Data: Structured Mining and Progressive Curriculum for E-Commerce Sponsored Search (https://arxiv.org/abs/2606.23911)
Comments:
          Accepted at E-Commerce Workshop, SIGIR 2026

- **Prior Approaches**: 기존 e-commerce sponsored search의 dense retrieval 학습은 주로 클릭(click) 로그에 의존하지만, 클릭은 노출 위치에 따른 편향(position bias)을 크게 받아 ‘관련성’이 아니라 ‘기존 랭커의 노출 결과’를 학습하기 쉽다. 또 tail query나 신규 조합에서는 클릭이 희소해 데이터 공백이 생기며, 이를 수백만 쌍 단위의 수작업 라벨링으로 메우는 건 비용·주기 측면에서 현실적이지 않다.

- **Core Contribution**: 이 논문은 클릭이나 수작업 라벨 없이도 생산 규모에서 고품질 학습 데이터를 만들기 위한 파이프라인을 제안한다. 핵심 아이디어는 서로 다른 3개 검색 채널이 top 결과에서 강하게 불일치(disagreement)한다는 점을 구조화된 감독 신호로 바꿔, 쉬운 positive(전 채널 동의)·어려운 positive(lexical은 찾지만 ANN은 못 찾음)·어려운 negative(정확히 한 채널을 속이는 반응)로 등급화한다.

- **Technical Challenges**: 실제로는 라벨 노이즈와 계산 비용이 문제인데, 이를 위해 3단계 LLM-기반 relevance 판별 cascade(교차인코더→LoRA-adapt 2B LLM→LoRA-adapt 8B LLM)에 per-class isotonic calibration을 적용해 라우팅을 신뢰도 기준으로 수행한다. 이어서 두-tower BERT 학생 모델을 BCE→MNR→Triplet의 3단계 progressive curriculum으로 학습하고, 채널 합의·랭크 위치·토큰 유사도 및 필터링 규칙을 통해 240M+ 학습 예제를 5개 난이도로 구성한다.

- **Empirical Impact**: Walmart의 sponsored search에서 배포한 결과, click-trained 기준선 대비 NDCG@10이 +5.1% 향상되었고 특히 tail query에서 개선 폭이 커졌다. 또한 ‘embarrassing retrieval(등급 0)’ 비율이 8.7%에서 3.5%로 감소했으며, 온라인 2주 A/B 테스트에서는 ad spend +2.80%, CTR +1.4%, eCPM +2.8%, click conversion rate +2.9%의 이득이 관측됐다.



### INSPIRE: Intent-aware Neural Sponsored Product Retrieval for E-commerc (https://arxiv.org/abs/2606.23889)
Comments:
          Accepted to ACM SIGIR E-commerce Workshop, 2026

- **Prior Approaches**: 기존 e-commerce 검색에서는 의미 기반 dense retrieval이나 쿼리-문맥 기반 intent 모델링이 주로 사용됐지만, 음식/음료 영역에서는 짧고 모호한 쿼리 때문에 의도가 충분히 주어지지 않는 문제가 남아 있었다. 또한 식품은 알레르기·식단 제한처럼 안전과 직결되는 조건이 많아, 표면 텍스트 유사도만으로는 constraint 위반 상품을 걸러내기 어렵다. sponsored search에서는 ad 슬롯 수가 적어 미스매치가 곧바로 클릭·매출 손실로 이어지는데, 기존 접근은 쿼리 측 intent와 상품 측 속성을 함께 정렬하는 데 취약했다.

- **Core Contribution**: INSPIRE는 intent를 brand·flavor·dietary preference·cuisine type 등 다차원 구조 속성(명시/암시 모두)으로 표현하고, 이를 기반으로 sponsored product retrieval을 수행하는 의도 인지 프레임워크다. 워커/상품 모두에 대해 구조화된 intent 예측치를 생성한 뒤, biencoder 학습과 임베딩에 이를 결합해 쿼리의 제약과 상품 속성을 더 정확히 맞춘다. 특히 LLM 교사-학생 증류(LoRA SFT)를 통해 Walmart 카탈로그 규모에서도 효율적으로 intent를 예측하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 암시적 의도(예: gluten-free, peanut-free, cuisine/use-case)가 쿼리에 직접 드러나지 않는다는 점과 (2) 상품 제목만으로는 해당 제약을 완전히 추론하기 어렵다는 점이었다. 논문은 로그 기반 오류 분석/하드 네거티브 마이닝으로 실패 패턴을 식별하고, 여러 LLM이 생성한 구조화 intent를 교차 모델 합의와 검증 단계로 정제해 weak supervision을 구성했다. 이후 Phi-4-mini-instruct에 LoRA 기반 SFT로 intent 속성 예측기를 증류하고, 예측된 intent를 쿼리·상품 표현에 포함한 dense retrieval biencoder를 Multiple Negatives Ranking과 코사인 회귀 손실로 학습해 의도 수준 정렬을 강화했다.

- **Empirical Impact**: 평가에서는 1만2천 쿼리를 대상으로 FAISS 인덱스에서 후보를 뽑아 NDCG@K, Precision@K, relevance@K를 측정했으며, intent-aware 모델이 평균 relevance@10을 3.033→3.105(+2.38%), Precision@1은 +4.2%로 개선했다. 또한 Excellent(점수 4) 비중이 +10.0% 증가하고, constraint 위반 성격의 중/저품질 결과(점수 2,1,0)가 각각 -34.5%, -7.7%, -50.0% 감소해 랭킹 분포가 의도 친화적으로 이동했음을 보여준다. 정성 예시에서도 'peanut free snack'·'gluten free pasta'처럼 단순 유사도로는 놓치기 쉬운 제약 위반을 intent 증강으로 바로잡는 효과가 확인됐으며, 배포 후 A/B 테스트를 계획 중이다.



### EvidenceLens: A Claim-Evidence Matrix for Auditing Financial Question Answering (https://arxiv.org/abs/2606.23724)
- **Prior Approaches**: 기존 금융 QA는 답변 전체를 한 덩어리로 생성한 뒤 인용을 덧붙이는 방식이 많아, “어떤 문장은 사실이고 어떤 문장은 추론인지” 같은 세분화가 약하다. 또한 텍스트·표·차트가 섞인 근거를 확인할 때, 채팅 UI는 지지/누락/모순을 한눈에 구분하기 어렵다.

- **Core Contribution**: EvidenceLens는 금융 QA를 주장-근거 정렬(claim-evidence alignment) 문제로 보고, 답변을 원자 단위 atomic claims로 쪼개 근거를 연결한다. 핵심 산출물은 multimodal claim-evidence matrix로, 주장별 커버리지·모순·모달리티 불균형을 시각적으로 드러내 검증 부담을 줄인다.

- **Technical Challenges**: 문장형 답변을 주장 타입(값/비교/추세/설명)과 텍스트·표·차트의 증거 단위로 정규화하고, 근거의 약한 지지와 실제 모순을 구분하는 정렬이 필요했다. 논문은 가벼운 멀티모달 정렬 파이프라인과 polarity test 기반의 충돌 탐지, 그리고 JSON 기반 아티팩트 스키마로 동일 입력에서 재현 가능한 시각 상태와 리뷰 우선순위를 구성한다.

- **Empirical Impact**: 두 가지 리포트 시나리오에서 EvidenceLens는 “그럴듯한” 답변이 실제로는 일부 주장만 지지하고 일부는 미세한 각주나 정량 근거와 충돌하는 패턴을 빠르게 보여줬다. 결과적으로 전통적 채팅이 평탄화하던 과신·설명형 점프·단일 모달리티 의존 같은 실패 모드를 분석가가 재검증하도록 유도하며, 금융 감시 환경에서 더 감사 가능한 근거 제시 방향을 제안한다.



### Are We Ready For An Agent-Native Memory System? (https://arxiv.org/abs/2606.24775)
Comments:
          Paper list available at: this https URL. Source code available at: this https URL

- **Prior Approaches**: 기존 LLM 에이전트 memory 평가는 end-to-end 과제 성공지표(F1, BLEU 등)에 치우쳐 메모리 시스템을 블랙박스로 다뤘다. 그 결과 비용(인덱스/지연), 모듈 간 설계 트레이드오프, 동적 지식 업데이트 시 견고성 같은 시스템 관점 평가는 충분히 쌓이지 않았다.

- **Core Contribution**: 본 논문은 agent memory를 데이터 관리 관점에서 분해해, 메모리 표현·저장, 추출, retrieval·routing, maintenance의 4개 핵심 모듈로 체계화했다. 이 프레임워크로 서로 다른 12개 메모리 시스템을 동일한 워크로드에서 비교하며, 효과가 특정 아키텍처의 우열이 아니라 워크로드 병목에 대한 정렬도에 달린다는 결론을 제시한다.

- **Technical Challenges**: 4개 모듈을 공정하게 분리해 실험하려면, 표현 충실도·검색 정밀도·업데이트 정확성·장기 안정성을 모듈 단위로 측정하고 단일 모듈만 바꾼 통제 실험을 설계해야 한다. 저자들은 11개 데이터셋을 포함한 5개 벤치마크 워크로드에서 모듈별 변형(ablations)과 fine-grained 분석을 수행해 각 레이어의 손실(압축/요약/추출이 정보 폐기를 유발)을 정량화했다.

- **Empirical Impact**: 대규모 end-to-end 평가에서 단일 구조가 모든 시나리오를 지배하지 못했으며, 특히 그래프 기반은 단발성 factual recall에는 강하지만 시간 추론에는 취약한 양상이 드러났다. 또한 운영비용 측면에서는 전역 재구성보다 localized maintenance가 더 비용 효율적이며, 지식 업데이트/장기 질의에서의 stale 정보와 시간순 단서 소실 같은 실패 모드가 구체적으로 관측돼 agent-native memory 설계 방향을 제안한다.



### Unified Dominance Graph for Interval-Predicate Approximate Nearest Neighbor Search (https://arxiv.org/abs/2606.24204)
- **Prior Approaches**: 기존 range-filtering ANNS(RFANNS)는 대부분 단일 스칼라 범위 제약에 최적화되어 있어, 구간 속성의 연속 구간 조건(containment, overlap 등)을 그대로 다루기 어렵다. 구간을 시작점·끝점 두 스칼라로 분리해 교집합을 내면 두 엔드포인트의 결합 제약이 분리되어 후보 생성/교차 비용이 커질 수 있다. 또 containment 중심 인덱스는 overlap 등 다른 closed two-bound 관계로 일반화되는 공통 인덱싱 추상화를 제공하지 못했다.

- **Core Contribution**: 이 논문은 Interval-Predicate ANNS(IPANNS)에서, 객체 구간과 쿼리 구간 사이의 predicate(containment/overlap 등)를 만족하는 kNN을 근사 탐색하는 문제를 다룬다. 이를 위해 Unified Dominance Graph(UDG)를 제안하며, 선택된 구간 관계를 2차원 dominance 공간의 의미 매핑으로 컴파일해 관계가 달라도 동일한 그래프 구성·탐색 알고리즘을 재사용한다. 또한 쿼리 상태별 proximity graph를 상태별로 따로 만들지 않고, 간선 라벨에 쿼리 상태 범위를 압축해 하나의 인덱스로 묶는 구조적 손실 없는 압축을 목표로 한다.

- **Technical Challenges**: 주요 기술 난제는 interval predicate가 두 개의 결합된 엔드포인트 부등식으로 정의되어 단일 정렬축(스칼라 범위)로는 valid set의 모양이 단조 구간이 되지 않는다는 점이다. UDG는 엔드포인트 비교 방향까지 정규화해 공통의 dominance predicate로 바꾸고, 변환 좌표에서 쿼리를 canonical 상태로 스냅하여 같은 valid 집합을 유도하도록 했다. 여기에 더해 restrictive한 interval 조건에서 그래프 탐색이 막힐 때를 대비해 validity-preserving patch edges로 라우팅 선택지를 보강한다.

- **Empirical Impact**: 실험은 표준 벤치마크와 실제 데이터셋에서 containment과 overlap을 포함한 여러 interval 관계 및 워크로드에 대해 UDG가 안정적인 쿼리 성능을 보이며 기존 hybrid search 베이스라인을 유의하게 능가함을 보여준다. 특히 인덱싱 오버헤드를 낮게 유지하면서도 여러 관계에 대한 성능을 함께 확보해, 하이브리드 질의가 많은 데이터베이스/검색 시나리오에 직접적인 실용성을 제공한다. 코드도 공개되어 있어 해당 분야에서 그래프 기반 interval-aware ANNS로 후속 연구 확장이 가능하다.



### MMed-Bench-IR: A Heterogeneous Benchmark for Multilingual Medical Information Retrieva (https://arxiv.org/abs/2606.24200)
Comments:
          Under review. 15 pages, 3 figures

- **Prior Approaches**: 기존 연구는 다국어 검색과 의료 특화 검색을 각각 따로 다루거나, 둘의 교차 지점을 부분적으로만 측정했다. 예를 들어 의료 biomedical encoders는 영어 위주로 평가됐고, 다국어 벤치마크는 의료 개념(ontology·UMLS) 근거가 약해 실제 임상 지식 매칭의 품질을 검증하기 어려웠다. 또한 관련 벤치마크들은 정렬(alignment), 개념 구분(discrimination), 근거 검색(evidence retrieval)을 ‘동시에’ 보지 않아 한 축의 개선이 다른 축을 해치지 않는지 확인이 불가능했다.

- **Core Contribution**: 논문은 다국어 의료 검색의 세 가지 역량을 한 프레임에서 분해·동시에 평가하는 MMed-Bench-IR을 제안한다. UMLS에 근거한 cross-lingual medical QA retrieval, UMLS 기반 confusion set으로 개념 구분을 측정하는 discrimination, 그리고 RAG용 multilingual evidence retrieval까지 총 3개 태스크를 6개 언어(영·스·프·일·중·러)·3개 문자 체계로 구성했다. 특히 세 태스크는 설계상 query와 concept 중복이 없어, 특정 기술만 잘하는 ‘편향된 합산’이 아니라 실제 capability breadth를 반영한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 언어 간 개념 정렬, (2) 임상적으로 헷갈리는 개념의 미세 구분, (3) 영어 중심 코퍼스에서 다국어 질의 근거를 찾아내는 evidencing을 동시에 측정하는 평가 설계에 있었다. 이를 위해 UMLS CUI 태깅으로 양성군을 구성하고, discrimination의 난이도는 ontology 구조(동의어-유사하지만 다른 개념-더 먼 관련)로 고정했으며, RAG 태스크는 번역 품질을 concept fidelity와 back-translation 일치도로 필터링해 83.7%를 유지했다. 또한 UMLS 기반 tier 정의가 특정 검증자에 치우치지 않는지(leave-one-validator-out) 안정성을 점검하고, 태스크 간 어휘·질의 중복을 극도로 낮춰 ‘평가 눈속임’을 줄였다.

- **Empirical Impact**: 10개 시스템(lexical·biomedical·multilingual dense·late-interaction·hybrid·reranker 계열)을 평가한 결과, 영어에서 잘하는 모델이 일본어 등 비(非)영문 스크립트에서 급격히 붕괴하는 현상이 뚜렷하게 드러났다. 예컨대 biomedical encoder SapBERT는 영어 nDCG@10이 0.818에서 일본어 0.056으로 급락했고, 공정성 격차(fairness gap)는 0.76으로 보고됐다(영어-only 벤치마크로는 포착 불가). 한편 concept discrimination이 가장 어려운 병목으로 나타났고, cross-encoder reranking 및 도메인·언어 특화 임베딩(MMed-Embed) 조합이 최상위 성능(0.377)과 상대적으로 낮은 공정성 격차(0.170)를 달성해, 다국어 임상 동등성 관점의 검색 개발 방향을 제시한다.



### Aspect-Based Sentiment Evolution and its Correlation with Review Rounds in Multi-Round Peer Reviews: A Deep Learning Approach (https://arxiv.org/abs/2606.24188)
- **Prior Approaches**: 기존 연구는 동료평(리뷰) 코멘트의 감정을 주로 거칠게 다루거나, 여러 라운드를 거치며 변화하는 차이를 충분히 구분하지 못했다. 그 결과 리뷰 라운드 전환에 따라 리뷰어가 주목하는 이슈와 감정 성향이 어떻게 이동하는지 분석이 제한적이었다.

- **Core Contribution**: 본 연구는 다중 라운드 리뷰 텍스트에서 aspect-level(항목 수준) 감정의 분포와 진화를 추적하고, 그것이 리뷰 라운드 수와 어떤 상관을 갖는지 체계적으로 분석한다. Nature Communications에 수록된 11,063편의 accepted 논문을 대상으로, fine-grained review aspect clusters(세분화된 리뷰 항목 클러스터)를 식별하고 이를 기반으로 감정 변화를 정량화한다.

- **Technical Challenges**: 핵심 과제는 리뷰 문장을 항목별로 나누어 감정을 정확히 분류하는 동시에, 라운드별 동적 변화를 반영할 수 있는 데이터와 모델링이 필요하다는 점이다. 약 5,000개 리뷰 문장에 대해 수작업으로 라벨링한 코퍼스를 구축해 deep learning 기반 aspect sentiment classification을 학습했으며, 그중 LCF-BERT-CDM이 Macro-F1 82.65%로 최고 성능을 보였다.

- **Empirical Impact**: 통계 분석 결과 리뷰 라운드 수가 늘어날수록 긍정 감정의 비중은 증가하고 부정 감정은 감소하는 일관된 경향이 관찰됐다. 상관 분석에서는 aspect sentiment 점수가 리뷰 라운드 수와 부(-)의 관계를 보였고, 특히 ‘experiments’, ‘research significance’, ‘result analysis’ 항목이 더 강한 연관성을 보이며 과학 평가 과정의 동학을 이해하는 데 의미가 크다.



### Exploring Academic Influence of Algorithms by Co-occurrence Network Based on Full-text of Academic Papers (https://arxiv.org/abs/2606.24099)
- **Prior Approaches**: 기존 연구는 특정 알고리즘을 단독으로 성능이나 인용 등으로 평가하는 경우가 많아, 논문들 사이에서 알고리즘이 서로 연결되며 만들어내는 ‘집합적 영향’을 충분히 다루지 못했다. 또한 알고리즘 언급은 인기 지표로 활용되더라도, 네트워크 관점에서의 구조와 시간에 따른 중심성 변화는 상대적으로 덜 탐구됐다.

- **Core Contribution**: 이 논문은 학술 논문의 전체 텍스트를 기반으로 NLP에서 알고리즘 공출현 네트워크를 대규모로 구축하고, 이를 통해 알고리즘 영향력을 네트워크 관점에서 정량화한다. 전체(누적) 네트워크와 연도별 네트워크를 함께 분석하며, 알고리즘이 분야 내에서 어떻게 ‘중심 위치’를 차지하고 변화하는지 보여준다.

- **Technical Challenges**: 핵심 과제는 방대한 본문에서 알고리즘 엔티티를 정확히 추출하고, 공출현을 일관된 규칙으로 네트워크에 반영하는 것이다. 연구진은 deep learning 모델로 알고리즘 엔티티를 추출한 뒤 전역·누적·연도별 공출현 네트워크를 만들고, 여러 centrality 지표로 집단 영향(전체 및 시간 축)을 비교·분석했다.

- **Empirical Impact**: 결과로 네트워크는 약 2년대가량에 걸쳐 점점 더 촘촘해지는 복잡계 특성을 보이며, 전통적 고성능 알고리즘과 서로 다른 연구 시기를 잇는 교차 지점의 알고리즘이 높은 인기·통제·중심성 및 균형 잡힌 영향력을 갖는 경향이 나타났다. 또한 영향이 감소할 때는 먼저 핵심 네트워크 위치를 잃고, 이후 다른 알고리즘과의 연관성이 약화되는 양상이 관찰됐다. 4년대 이상 출판물을 아우르는 최초 규모의 알고리즘 공출현 네트워크 분석으로, 향후 알고리즘·학자·태스크를 연결하는 네트워크 연구의 기반을 제공한다.



### Is Higher Team Gender Diversity Correlated with Better Scientific Impact? (https://arxiv.org/abs/2606.24098)
- **Prior Approaches**: 기존 연구는 성별이 포함된 협업 패턴과 논문 성과(예: 인용) 간의 관계를 탐색했지만, 성과를 더 높이는 ‘성별 다양성’의 구체적 요소는 충분히 규명되지 않았다. 또한 분야별로 성별 불균형이 어떻게 다르고, 그것이 인용에 어떤 방식으로 반영되는지에 대한 비교가 제한적이었다. 특히 NLP와 LIS처럼 서로 다른 학문 생태계를 대상으로 한 정량 분석은 상대적으로 부족했다.

- **Core Contribution**: 본 연구는 NLP와 LIS 논문 사례를 바탕으로 성별 다양성과 과학적 영향력(인용)의 상관관계를 직접 분석한다. 저자 성별 구성(동일 성별/혼성)과 성별 다양성 수준이 인용 수에 미치는 차이를 함께 제시해, ‘어떤 다양성이’ 유의미한지에 초점을 둔다. 더 나아가 성과가 최대가 되는 이상적인 성별 비율 범위(총 저자 중 한 성별이 5~15%)를 제안한다.

- **Technical Challenges**: 성별 다양성을 단순 분류(혼성 vs 동성 협업)로만 보기에는, 다양성의 ‘정도’가 인용에 미치는 영향이 가려질 수 있다. 연구진은 분야별 성별 불균형이 다르다는 점을 고려하면서, 성별 구성에 따른 평균 인용 차이와 다양성 수준의 비선형 관계를 동시에 추정해 inverted U-shaped 패턴을 도출한다. 이를 위해 NLP와 LIS에서 확보된 논문 데이터를 기반으로 성별 구성 지표와 인용을 체계적으로 연결했다.

- **Empirical Impact**: 결과적으로 NLP와 LIS 모두에서 여성 연구자 과소대표가 확인되며, 그 격차는 NLP에서 더 크게 나타난다. 또한 혼성 협업은 시간이 갈수록 동성 협업보다 평균 인용이 높아지는 경향을 보였고, 성별 다양성 수준은 인용과 inverted U-shaped 관계를 형성했다. 특히 성과가 가장 큰 성별 비율 계산을 통해, 협업 팀 구성에 참고할 수 있는 ‘이상적’ 범위(5~15%)를 제시함으로써 분야 간 협업 전략 논의에 실증적 근거를 제공한다.



### Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs (https://arxiv.org/abs/2606.23915)
- **Prior Approaches**: RAG에서 “attribution(근거 인용/청구 지지)” 평가는 lexical overlap, embedding 유사도, NLI(엔테일먼트) 기반 자동 점수를 묶어 서로 대체 가능하다고 가정해 왔다. 그러나 기존 연구들은 자동 점수가 인간 판정과 어긋나며 단일 지표가 전 구간에서 최선을 보장하지 못한다는 점을 이미 시사했다.

- **Core Contribution**: 이 논문은 LLM attribution을 평가하는 여러 자동 scorer(8종)를 대상으로, 특정 데이터셋 조합에서도 “최고 scorer가 그대로 전이(transfer)되는지”를 감사(audit)하는 기준을 제안한다. 구체적으로 다중-데이터셋으로 구성된 evaluation construct 안에서, 어떤 scorer도 모든 데이터셋에 대해 최상 성능 구간(95% CI) 내에서 안정적으로 유지되지 않는다는 결론을 제시한다.

- **Technical Challenges**: 핵심 도전은 scorer 간 성능이 데이터셋과 construct에 따라 뒤집히는데, 이것이 길이/잘림(truncation) 같은 단순 요인인지 분리해내는 것이다. 연구진은 provenance/topicality, generated-answer attribution, fact-check entailment의 구성 자체를 분리해 평가하고, AttributionBench(4개 출처, n=1610) 및 독립 세트(HAGRID 등)에서 AUROC·순위 통계(Kendall tau)와 leave-one-dataset-out regret로 검증해 “전이 실패가 구조적”임을 보여준다.

- **Empirical Impact**: 생성문-근거 attribution(AttributionBench)에서는 최고의 NLI scorer가 단문 데이터셋에서는 강하지만 장문 데이터셋에서 AUROC가 0.53(우연 수준)으로 붕괴하는 등 per-dataset 순위 역전이 관찰됐다. 또한 ‘best-on-average’로 평가기를 고르면 leave-one-dataset-out에서 평균 regret(0.172 AUROC)이 커져, metric 선택은 반드시 타깃 데이터셋 검증을 거쳐야 한다는 실무적 경고를 남긴다.



### Ground Then Rank: Revisiting Knowledge-Based VQA with Training-Free Entity Identification (https://arxiv.org/abs/2606.23881)
Comments:
          Accepted by ACL 2026 Findings. Project page this https URL

- **Prior Approaches**: 기존 KB-VQA 대응은 multi-modal retrieval-augmented generation(MM-RAG) 흐름에서 검색 후 재랭킹을 통해 답을 뒷받침할 텍스트 구간을 찾는 방식이 주류다. 그런데 많은 방법이 entity 식별과 evidence(증거) 구간 선택을 한 번의 재랭킹 단계로 강하게 결합해, 엔티티는 맞지만 구간이 틀리거나 반대로 구간은 그럴듯하지만 엔티티가 틀리는 문제가 발생한다. 또한 학습 기반 멀티모달 재랭커를 붙이는 경우가 많아 비용과 데이터 의존성이 커진다.

- **Core Contribution**: 이 논문은 KB-VQA에서 필요한 grounding을 entity-level grounding과 section-level grounding의 두 병목으로 재정의하고, 이를 workflow 관점에서 명시적으로 분리하는 training-free Identify Before Answer(IBA)를 제안한다. 핵심 아이디어는 MLLM이 답을 생성하기 전에 ‘후보 엔티티 이름’들 중 고신뢰 엔티티를 먼저 고른 뒤, 그 엔티티로 좁혀진 범위에서 텍스트 re-ranker로 증거 구간을 고르는 파이프라인이다. 이를 통해 fine-tuning 없이도 정확도와 효율을 동시에 노린다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 open-ended 엔티티 이름 생성 방식이 MLLM에 높은 불확실성을 요구해 실패가 잦다는 점이다. 저자들은 흥미로운 실험 관찰로, 모델에게 엔티티 이름을 자유 생성하게 하지 않고 candidate name shortlist에서 고르도록 하면 식별 정확도가 크게 올라간다고 보고 이를 tip-of-the-tongue 현상에 비유한다. 구현은 EVA-CLIP-8B로 시각적으로 후보 페이지를 먼저 뽑고, Qwen-2.5-VL-7B-Instruct로 후보 엔티티를 1차 선별한 뒤 BGE 같은 off-the-shelf textual re-ranker로 evidence를 재랭킹하는 방식이다.

- **Empirical Impact**: Encyclopedic-VQA(E-VQA)와 InfoSeek에서 IBA는 fine-tuned multi-modal re-ranking 기준선을 전반적으로 능가하며, training 및 추론 복잡도를 줄이면서도 성능을 끌어올렸다고 보고한다. Recall@1(정답 엔티티 우선순위)에서 InfoSeek은 58.4%로 EchoSight의 53.1%를 상회하고, E-VQA에서는 약간 낮은 Recall@1(35.5%)을 보이지만 다운스트림 답변 품질은 여전히 경쟁적 수준을 유지한다. 분석에 따르면 개선은 단순히 엔티티 식별이 좋아진 데 그치지 않고, 정답 엔티티가 고정된 뒤 더 유익한 evidence를 선택하게 된 효과가 함께 작용한다.



### HANCLIP: A Family of Hyperbolic Angular Negation Vision Language Models (https://arxiv.org/abs/2606.23843)
- **Prior Approaches**: 기존 비전-언어 모델(VLM)은 대규모 이미지-텍스트 사전학습으로 의미 대응을 학습하지만, 부정(negation)에는 의외로 취약합니다. 많은 연구가 “문장-단어 동시출현” 같은 얕은 단서에 의존해, 부정 캡션과 정답 캡션 모두에 높은 유사도를 주는 문제가 보고됩니다. 또 부정 데이터로 직접 fine-tuning하면 부정 감도는 좋아져도 표준 벤치마크 성능이 떨어지는 지식 망각(forgetting) 문제가 자주 따라옵니다.

- **Core Contribution**: 이 논문은 HANCLIP(Hyperbolic + Angular + Negation)이라는 VLM 계열을 제안하며, 임베딩 공간 자체를 “무엇이 맞는가”와 “무엇이 아닌가”를 분리하는 방식으로 재구성합니다. 하이퍼볼릭(비유클리드) 기하를 이용해 부정 캡션이 긍정/부정 의미 위계에서 다른 영역을 차지하도록 설계하고, 추가로 각도(angular) 기반 삼중항 손실로 부정-긍정 간 관계를 정교하게 조절합니다. 특히 대규모 재학습 없이 CLIP 계열 백본에 플러그인 형태로 적용 가능하다고 강조합니다.

- **Technical Challenges**: 핵심 기술적 난제는 부정 민감도를 높이면서도 기존 사전학습된 표현의 전역 구조(global structure)를 덮어쓰지 않는 것입니다. 이를 위해 (1) HCO(Hyperbolic Contrastive Objective)는 포인카레 볼(Poincaré ball) 위에서 하이퍼볼릭 거리로 진짜 negative를 명시적으로 대조하고, (2) ATNL(Angular Triplet Negation Loss)은 negative를 앵커로 두고 상대 방향의 각도를 정렬/분리하여 부정이 의미를 왜곡하지 않게 하면서도 잡음성 어휘 중복에 덜 흔들리도록 만듭니다. 결과적으로 “부정 캡션은 긍정과는 가깝되(각도 정렬), 진짜 negative와는 멀어지게(각도 분리)” 만드는 기하-aware 목적함수를 구성합니다.

- **Empirical Impact**: CC-Neg에서 2만 개 샘플로 학습한 뒤 NegBench의 Negative Retrieval 및 MCQ-Neg에서 전반적으로 성능이 개선됐고, 백본이 CLIP/LongCLIP/SmartCLIP/HiMo-CLIP일 때도 일관된 향상이 관찰됩니다. 논문은 NegBench에서 rSum 기준 대략 5~10% 수준의 상대 개선과, MCQ에서 평균적으로 큰 폭의 상승(거의 35%에 근접)을 보고하며 데이터 효율성을 강조합니다. 또한 표준 이미지-텍스트 검색/분류 성능은 경쟁 수준을 유지하거나 개선되어, 단순 fine-tuning의 지식 망각 트레이드오프를 완화하는 방향성을 제시합니다.



### EXPO-SQL: Execution-based Clause-level Policy Optimization for Text-to-SQL (https://arxiv.org/abs/2606.23693)
Comments:
          20 pages, 8 figures

- **Prior Approaches**: Text-to-SQL은 자연어와 스키마를 입력으로 SQL을 생성해 실행 가능 쿼리를 만드는 기술로, 기존에는 SFT로 정답 SQL과의 토큰 정합을 맞추거나 prompting으로 추론/리파인 단계를 반복하는 방식이 주류였다. 그러나 두 방법 모두 실행 결과를 학습 신호로 직접 반영하지 못해, 잘못된 SQL이 나올 때 어떤 절(SELECT/WHERE 등)이 원인인지 구분하기 어렵다. 이후 RL 계열이 execution feedback을 보상으로 쓰기 시작했지만, 대다수는 쿼리 단위의 동일 보상을 모든 clause에 동일하게 부여해 coarse credit assignment 문제를 만든다.

- **Core Contribution**: 이 논문은 EXPO-SQL(EXecution-based clause-level Policy Optimization for Text-to-SQL)을 제안하며, 실행 결과를 바탕으로 clause별 오류를 식별하고 clause-level reward로 미세 감독을 제공한다. 기존 RL의 “정답/오답을 쿼리 전체에만” 주던 한계를 깨고, 틀린 절은 확실히 낮추되 맞는 절은 유지하도록 학습 신호를 분리한다. 특히 incorrect result와 execution error를 각각 다른 방식으로 clause 오류 집합 C_err에 매핑한다.

- **Technical Challenges**: 핵심 난제는 온라인 RL에서 clause별 오류를 안정적으로 할당하는 것이다: 어휘/구문만으로는 오류 절을 알 수 없고, SQL은 정답이 여러 형태로 나올 수 있으며, 실행 피드백도 기본적으로 binary(전체 쿼리 성공/실패)라서 직접 신호가 빈약하다. EXPO-SQL은 이를 위해 (1) incorrect result에서는 FROM/JOIN→WHERE→GROUP BY→HAVING→SELECT→ORDER BY→LIMIT의 결정적 실행 순서를 따라 clause-wise incremental execution으로 각 절이 결과를 바꾼 흔적을 추적하고, (2) diff_type(실행 결과 테이블 변화 유형 10종)과 누적 차이를 결합해 오류 절을 판별하며, (3) execution error에서는 에러 메시지에서 원인 요소를 파싱한 뒤 clause-level tracing으로 실제 root clause까지 역추적한다.

- **Empirical Impact**: 실험은 Spider와 BIRD 등 대표 벤치마크에서 SFT·prompt·기존 RL 대비 일관된 우위를 보이며, Spider-Dev/BIRD-Dev에서 각각 1.2%p/2.4%p 개선, 복잡 쿼리에서는 5.6%p까지 향상됐다. 또한 ablation에서 incorrect result·execution error 모두에 clause-level rewards를 적용할 때 성능이 가장 높게 나타나 fine-grained learning 신호의 효과를 뒷받침한다. 코드도 공개되어(제공 링크) 재현성과 후속 연구 확산에 기여할 것으로 보인다.



New uploads on arXiv(cs.CV)

### DiffusionBench: On Holistic Evaluation of Diffusion Transformers (https://arxiv.org/abs/2606.24888)
- **Prior Approaches**: DiT 이미지 생성 연구는 ImageNet 클래스 조건부 생성(256/512) 중심의 단일 평가 관성에 수렴했다. ImageNet-FID는 비교를 빠르게 해왔지만, 방법이 실제 생성 모델링 전반의 진전을 반영하는지 과적합 우려가 커졌다. 텍스트-투-이미지(T2I) 평가는 데이터 파이프라인·평가 절차·코드베이스가 달라 고비용·고마찰이라 생략되는 경향이 있었다.

- **Core Contribution**: 이 논문은 NanoGen이라는 통합 DiT 학습·평가 프레임워크를 제안해 ImageNet과 T2I를 동일한 코드베이스/백본/학습 루프/평가 하네스로 비교 가능하게 만든다. 또한 ImageNet 결과만으로는 T2I 성능을 신뢰 있게 예측할 수 없음을 보여주며, 두 태스크를 함께 묶는 DiffusionBench를 기본 벤치마크로 제안한다. 핵심 질문은 “ImageNet FID를 개선하면 T2I에서도 개선이 동반되는가?”이며, 답은 일관되지 않다는 점으로 귀결된다.

- **Technical Challenges**: 통합 비교를 위해 가장 큰 기술적 난제는 태스크 전환 비용을 최소화하면서도 공정한 실험을 유지하는 것이었다. NanoGen은 DDT(encoder-decoder) 백본과 공통 최적화/샘플링/평가 구조를 공유하고, 태스크 차이는 데이터 로더와 conditioning 모듈(클래스 임베더 vs frozen text encoder)만 교체하도록 설계했다. 그 결과 ImageNet 설정에서 T2I 설정으로의 전환을 설정 변경 중심으로 흡수하며, RAE, VAE, pixel-space, MeanFlow 등 다양한 방법을 동일 레시피로 재학습·평가할 수 있게 했다.

- **Empirical Impact**: NanoGen으로 ImageNet과 T2I의 21개 latent diffusion 모델을 거의 동일 조건에서 학습한 뒤, ImageNet과 T2I 간 순위 상관이 약하고(세 지표 기준 Pearson correlation -0.377~ -0.580) 결론이 뒤집힐 정도의 불일치가 관찰됐다. 또한 RAE/latent-space가 전반적으로 T2I에서도 강한 편이지만, “ImageNet-FID가 좋아지면 T2I가 반드시 좋아진다”는 예측 규칙은 성립하지 않았다. DiffusionBench를 채택해 두 축을 함께 보고, 한 태스크 성능 향상만 주장하는 관행을 넘어서는 것이 향후 DiT 진전을 더 정확히 반영할 것임을 실증적으로 뒷받침한다.



### BenchX: Benchmarking AI Models for Cancer Detection and Localization with Demographic and Protocol Biases (https://arxiv.org/abs/2606.24883)
- **Prior Approaches**: 기존 의료영상 AI는 대체로 평균 정확도에 최적화돼, 환자 인구통계(연령·성별·인종)와 촬영 프로토콜(조영 단계) 변화에 따라 성능이 흔들리는 문제를 충분히 드러내지 못했다. 또한 희귀·과소표본 하위집단은 충분한 라벨 데이터를 모으기 어려워, 공정성·견고성 평가가 제한되는 한계가 있었다.

- **Core Contribution**: 이 논문은 85,355장의 CT를 대상으로 12개 종양-탐지 모델을 종양 크기·위치·환자 하위집단·영상 프로토콜 관점에서 체계 평가하는 대규모 오픈 벤치마크를 구축했다(BenchX). 아울러 LLM을 활용해 임상 데이터에서 하위집단 메타데이터를 추출·정리해 분석을 확장 가능하고 재현 가능하게 만들었다.

- **Technical Challenges**: 하위집단 평가의 신뢰성은 라벨 품질과 메타데이터 정확도에 달려 있는데, 이를 위해 스캔 단위 종양 라벨은 8명의 방사선과 전문의가 다중 판독·합의(adjudication)로 확정했다. 또한 Llama-3.1-70B와 DeepSeek-R1-Distill-Qwen-32B 두 LLM의 속성 추출 일치도를 점검해 불일치 케이스는 수동 검증하고, 보고서 근거가 불충분하면 Unknown으로 처리해 불확실한 메타데이터가 하위집단 분석에 섞이지 않게 했다.

- **Empirical Impact**: BenchX 분석 결과, 평균 정확도에 맞춘 최신 모델들이 젊은 여성 아프리카계 미국인처럼 희귀 또는 과소표본 하위집단에서 성능이 크게 저하되는 패턴을 확인했다. 저자들은 민감도·특이도, F1, Balanced AUC처럼 실패 양상이 다른 지표를 조합해 이런 격차를 놓치지 않도록 했으며, 의료영상 분야에서 subgroup-level 평가의 표준 필요성을 강하게 제시한다.



### FLAT: Feedforward Latent Triangle Splatting for Geometrically Accurate Scene Generation (https://arxiv.org/abs/2606.24876)
- **Prior Approaches**: 단일 이미지(또는 텍스트)로 3D 장면을 생성할 때, 기존 방식은 video diffusion이 만든 프레임/신호를 3DGS나 NeRF로 다시 최적화(generate-then-optimize)해 장면을 맞추는 흐름이 주류였다. 이런 방법은 렌더링 품질은 좋지만 per-scene 최적화로 계산량이 커서 상호작용/확장성에 불리하다. 반면 feedforward latent scene decoder들은 빠르게 장면을 만들지만 주로 3D Gaussians를 출력해, 시뮬레이션/그래픽 파이프라인에 필요한 ‘불투명 표면(삼각형/메시)’로 쓰기 어렵다는 한계가 남아 있었다.

- **Core Contribution**: FLAT은 압축된 video diffusion latents를 단일 forward pass로 ‘표면 정렬(surface-aligned) 삼각형 splat’에 바로 디코딩하는 것을 목표로 한다. 기존 feedforward 3DGS류가 내놓는 체적(반투명) 표현 대신, FLAT은 triangle splatting을 직접 예측해 renderable하면서도 명시적 기하자산에 더 가깝게 만든다. 또한 반투명 예측을 가볍게 test-time refinement해 게임 엔진 호환 수준의 불투명 triangle 표현으로 바꾸는 파이프라인도 제시한다.

- **Technical Challenges**: 삼각형(비체적) 프리미티브는 orientation이나 퇴화(degenerate) 문제에 민감해, 잘못된 자세는 렌더링 기여가 작아져 초기 학습의 gradient flow가 쉽게 붕괴한다는 점이 핵심 난제다. FLAT은 이를 ray-centered rotation parameterization(레이 중심 로컬 프레임에서 residual tilt/spin 예측)과 Cholesky-style 제약을 둔 shape transform으로 안정화하고, differentiable triangle rendering에서 gradient를 끌어주는 novel product window function으로 경계 주변 학습 신호를 개선한다. 추가로 이 렌더링 안정성에 맞춰 디코더는 pretrained video VAE 기반 구조를 전이학습 형태로 수정해, latent→기하 파라미터 회귀에 집중하도록 설계했다.

- **Empirical Impact**: 표준 벤치마크에서 FLAT은 volumetric 3D Gaussians 디코딩 대비 기하 정확도를 유의미하게 끌어올리면서도 시각 품질은 경쟁 수준으로 유지한다. 더 나아가 3DGS, 2DGS, triangle splatting을 동일한 학습 설정에서 비교하는 체계적 분석을 통해 표현(representation) 선택이 렌더링 품질과 기하 정확도, 그리고 downstream 호환성에 미치는 트레이드오프를 처음으로 정리한다. test-time refinement까지 포함하면 예측 삼각형 ‘triangle soup’을 실시간 렌더링이 가능한 불투명 형태로 변환할 수 있어, 그래픽/시뮬레이션 적용성을 실증했다.



### FLUX3D: High-Fidelity 3D Gaussian Generation with Diffusion-Aligned Sparse Representation (https://arxiv.org/abs/2606.24874)
- **Prior Approaches**: 기존 image-to-3D 파이프라인은 주로 representation learning과 latent diffusion을 결합하는 방식이다. 특히 sparse voxel 기반 방법은 계산 효율과 지오메트리 인식을 동시에 노리지만, 입력 이미지의 고주파 디테일을 보존하지 못해 텍스처가 흐려지거나 시각 패턴이 손실되는 문제가 반복된다.
또한 생성 단계에서는 dense 2D 토큰과 sparse 3D voxel 잠재공간의 대응관계를 충분히 정렬하지 못해 cross-modal correspondence 병목이 생기며, 그 결과 뷰가 달라질 때 질감이 어긋나는 현상이 나타난다.

- **Core Contribution**: 이 논문은 FLUX3D라는 image-to-3DGS 프레임워크를 제안해 두 병목—표현 병목과 cross-modal 정렬 병목—을 동시에 완화한다. 핵심은 표현학습 단계에서 DINOv2 같은 판별적 2D 피처 대신 FLUX의 생성형 diffusion 피처를 structured latent로 쓰는 DA-SLAT( Diffusion-Aligned Structured Latents )를 도입한 점이다.
생성 단계에서는 Sparse-structure Multimodal Diffusion Transformer (SMDiT)와 Modal-Aware Rotary Positional Embedding (MARoPE)로 2D 토큰이 sparse 3D 토폴로지에 맞게 학습되도록 하여, 2D-3D 정합성을 geometry-agnostic하게 끌어올린다.

- **Technical Challenges**: 가장 큰 기술적 과제는 (1) sparse voxel latent가 의미론 추상화에 치우친 2D 피처를 받아 고주파 재현에 불리해지는 문제와, (2) diffusion transformer가 표준 attention/positional encoding만으로는 sparse한 3D 구조와 dense 2D 토큰의 공간적 대응을 제대로 못 맞추는 문제를 동시에 해결하는 것이다.
저자들은 DA-SLAT에서 diffusion 피처의 정보밀도/재구성 적합성을 활용하고, decoder-only 구조로 불필요한 인코딩 압축을 줄여 3DGS 복원 충실도를 높인다. 더 나아가 SMDiT의 double-stream+single-stream 설계와 MARoPE의 virtual plane 기반 좌표 프레이밍으로, 카메라 캘리브레이션 없이도 2D-3D relative correspondence를 데이터로부터 학습하게 만든다.

- **Empirical Impact**: 실험에서는 3D-FUTURE, ABO, HSSD, Objaverse-XL로 학습하고 Toys4k로 평가하며, sparse voxel 레이아웃은 주어진 조건에서 appearance(텍스처) 품질을 집중 측정한다. 결과적으로 FLUX3D는 SSIM/PSNR/LPIPS 등 재구성 지표에서 기존 기준선과의 격차를 줄이고, image-to-3DGS 생성에서도 SSIM/PSNR/LPIPS뿐 아니라 CLIP Score, Fréchet Distance(FD), Kernel Distance(KD)까지 아우르는 다면적 성능 향상을 보인다.
정성 비교에서도 LGM류의 왜곡, DiffusionGS/TRELLIS의 질감 불일치 문제를 줄이며 입력 뷰의 색/디테일을 다양한 카테고리에서 더 안정적으로 보존해 SOTA 대비 실용적 의미가 크다는 평가를 받는다.



### IV-CoT: Implicit Visual Chain-of-Thought for Structure-Aware Text-to-Image Generation (https://arxiv.org/abs/2606.24849)
- **Prior Approaches**: 통합 multi-modal large language models(MLLMs)은 텍스트-이미지 생성 품질은 높지만, 객체 수·공간 관계·속성 결합·레이아웃 같은 구조 요구를 제대로 지키지 못하는 경우가 많다. 기존 접근은 프롬프트를 단일 conditioning stream에 압축해 구조와 외형(색·질감·조명 등)이 얽히며, 그 결과 속성이 다른 객체로 바뀌거나 누락, 배치 오류가 발생한다. CoT 기반도 명시적 언어 단계나 중간 시각 상태(마스크·레이아웃·초안 이미지)를 거치면 추론 단계가 늘거나 오류 누적 위험이 커진다.

- **Core Contribution**: 이 논문은 구조-aware prompt following을 위해 Implicit Visual Chain-of-Thought(IV-CoT)를 제안한다. 핵심은 구조 계획을 외부의 텍스트/중간 디코딩 없이, MLLM-DiT의 query 공간에서 structural-to-semantic “캐스케이드”로 내재화하는 것이다. structural query가 먼저 객체 경계·개수·레이아웃·거친 공간관계를 포함한 latent visual plan을 만들고, semantic query가 이를 기반으로 외형의 세부(정체성·색·재질·질감)를 렌더링한다.

- **Technical Challenges**: 가장 큰 도전은(1) 구조 정보를 외형과 분리해 모델 내부에 안정적으로 학습시키면서, (2) 추론 시에는 단일 forward pass로 처리해 효율을 유지하는 것이다. IV-CoT는 training-only sketch supervision을 도입해 structural query가 스케치의 윤곽·형상·개수·레이아웃을 학습하되 색/질감/조명 같은 appearance 요인을 억제하도록 유도한다. 이후 이미지 생성 학습에서는 structural loss를 regularizer로 유지해 구조 플래닝이 appearance 편향으로 drift되지 않게 하면서 semantic query는 세부 외형을 채우도록 최적화한다.

- **Empirical Impact**: GenEval과 T2I-CompBench에서 IV-CoT는 OpenUni-L-1024 대비 성능을 각각 0.86→0.88, 0.5448→0.5743으로 끌어올리며 구조에 민감한 항목(공간관계·형상·질감·색 등)에서 특히 개선이 두드러졌다. 또한 추론은 스케치 추출·중간 이미지 디코딩·테스트타임 탐색 없이 단일 forward pass로 동작해 latency가 명시적 CoT 대비 9~15배 낮다. 분석 결과 structural query는 스케치 도메인의 recoverable한 구조를 인코딩하고, query separation을 바탕으로 프롬프트 간 구조-외형 재조합도 어느 정도 제어 가능함을 보였다.



### Spherical-to-ERP Epipolar Rectification for Single-Axis Disparity in 360 Stereo (https://arxiv.org/abs/2606.24847)
Comments:
          7 Pages, 4 Figures, Conference

- **Prior Approaches**: 전방위(omnidirectional) 스테레오 이미지는 360도 인식을 가능하게 하지만, 구면/어안(fisheye)에서는 고전적 디스패리티 가정이 깨진다. 기존 방법들은 에피폴라 대응이 직선이 아니라 대원( great-circle )을 따라 휘며, 이로 인해 2차원 변위가 단일축 disparity로 바로 해석되지 못해 정합·해상도가 저하될 수 있다.

- **Core Contribution**: 이 논문은 구면-직교투영(spherical-to-equirectangular, ERP) 전처리로 에피폴라 곡선을 펴서 disparity의 1차원 구조(좌우 리그는 수평, 상하 리그는 수직)를 복원한다. 또한 기존에 제안한 RAFT + Epipolar-Aligned Channel Selection(EACS) 프레임워크를 구면 스테레오에도 그대로 적용할 수 있는지를 검증하고, ERP 투영 이후 RAFT의 흐름을 baseline 정렬 성분만 남겨 disparity로 변환한다.

- **Technical Challenges**: 핵심 난제는 구면 공간에서의 왜곡이 RAFT 추정된 optical flow를 단일축 disparity로 환원하기 어렵게 만든다는 점이다. 저자들은 ERP로 기하를 먼저 정규화한 뒤, EACS 방식대로 baseline-aligned flow component만 선택해 1차원 디스패리티 구조를 유지하면서 잡음과 불필요한 성분을 억제하는 방식으로 해결했다.

- **Empirical Impact**: 합성 fisheye 스테레오 데이터에서 제안 파이프라인(spherical-to-ERP-to-RAFT+EACS)은 정확하고 매끄러우며, 구조적으로도 일관된 disparity 맵을 real-time 속도에 가깝게 제공한다. 결과적으로 기존 스테레오 파이프라인에 ERP 전처리를 결합해 360 imaging에서도 실행 가능하고 해석 가능한 disparity 추정을 실용적으로 확장할 수 있음을 보여준다.



### Bridging the Manifold Gap: Riemannian Residual Line Search for One-Step Image Editing (https://arxiv.org/abs/2606.24844)
- **Prior Approaches**: 기존 텍스트-유도 이미지 편집은 DDIM inversion, null-text inversion, prompt optimization처럼 역과정/최적화를 거쳐 품질을 확보했지만, 추론 지연이 커지는 한계가 있었다. 반대로 ChordEdit처럼 one-step 업데이트는 빠르지만 업데이트 강도가 편집 종류 전반을 동시에 만족하기 어렵고, 첫-order 에너지 차분 가정 때문에 분포 불일치가 커지면 편집 능력이 급격히 저하된다.

- **Core Contribution**: RRLS(Riemannian Residual Line Search)는 한 가지 공격적인 one-step 업데이트를 고정으로 강제하는 대신, 에너지 필드의 second-order 곡률 정보를 이용해 “더 강한 편집 방향”을 먼저 만든다. 이후 소스 보존과 의미 정합 사이의 균형은 후보 혼합(residual path)과 CLIP 기반 타깃 정합을 최대화하는 선택기로 분리해 해결한다. 결과적으로 모델 구조를 새로 학습하거나 반복 최적화를 하지 않고도, one-step 프레임워크를 유지한 채 성능을 끌어올린다.

- **Technical Challenges**: 핵심 technical challenge는 first-order 근사로는 prompt-delta field가 시간에 따라 휘는 경우 곡률 정보가 소실되어 under-editing 또는 over-drift가 생긴다는 점이다. RRLS는 second finite difference로 로컬 time curvature를 추정하고 trust coefficient로 보정 강도를 안정적으로 조절한 뒤, update norm을 보존하는 투영으로 크기는 유지하면서 방향만 곡률 보정한다.

- **Empirical Impact**: PIE-Bench++의 700개 샘플(편집 타입 10종)에서 RRLS는 현재 one-step 계열 중 SOTA를 달성했으며, 특히 구조 보존과 의미 정합을 함께 개선했다. 또한 ChordEdit 대비 PSNR과 LPIPS/DINO 기반 구조 지표가 크게 개선되면서도 전체 transport 처리를 단 한 번에 끝내 0.52초 수준의 경쟁력 있는 런타임을 보였다. 편집 품질-추론 효율의 Pareto frontier를 한 단계 확장했다는 점에서 실제 적용 관점의 의미가 크다.



### GeoT2V-Bench: Benchmarking 3D Consistency in Text-to-Video Models via 3D Reconstruction (https://arxiv.org/abs/2606.24829)
Comments:
          36 pages, 17 figures, 18 tables

- **Prior Approaches**: 카메라 프롬프트 기반 text-to-video(T2V) 모델은 객체를 공전하거나 고정 장면을 통과하는 가상 카메라 영상을 생성하지만, 기존 평가는 주로 한 프레임의 시각적 그럴듯함에 치우친 경향이 있다. 그래서 생성 결과가 ‘하나의 정적 3D 장면’에 대한 일관된 다중 뷰 증거로 작동하는지(명시적 rigid 3D 재구성 가능성)는 잘 진단되지 못했다.

- **Core Contribution**: 이 논문은 생성된 카메라 주행 영상이 단일 정적 3D 장면의 rigid 3D 재구성을 뒷받침하는지 점검하는 진단 벤치마크 GeoT2V-Bench를 제안한다. pass/fail 같은 이분법 대신, 여러 지표로 구성된 연속형 reconstruction profile을 제공해 재구성 실패 양상을 더 세밀하게 드러낸다.

- **Technical Challenges**: 핵심 기술적 과제는 각 프레임의 카메라 intrinsics와 pose를 안정적으로 추정한 뒤, 그 궤도를 따라 정적 3D 프록시를 구축해 영상 간 일관성을 측정하는 것이다. 이를 위해 VGGT-style 기하 추정으로 프레임별 카메라 파라미터를 추정하고 DeformableGS를 적합한 뒤 temporal-median aggregation으로 MedianGS 정적 프록시를 만들고, 추정된 카메라 경로로 렌더링 및 정적 렌더링 오차·flow agreement·flexible vs static 적합 간 격차 등을 산출한다.

- **Empirical Impact**: 공정한 포맷의 four-seed 평가에서 12개 오픈웨이트 모델 구성과 80개의 GeCo-Eval 정적-scene 프롬프트로 총 3,840개의 재구성을 수행했으며, visible motion, static rendering error, flow agreement, flexible-vs-static 거동이 자주 서로 불일치하는 패턴을 확인했다. 즉, 시각적으로 그럴듯해 보이는 결과라도 전역 정적 장면 획득 관점에서 상이한 실패 모드가 동시 발생하며, GeoT2V-Bench가 이를 포착하는 데 효과적임을 보여준다.



### High-Fidelity Synthetic Transmission Electron Microscopy Image Generation Using Diffusion Probabilistic Models for Data-Limited Semiconductor Metrology (https://arxiv.org/abs/2606.24817)
Comments:
          To be presented at the 2026 International Symposium ELMAR, published by IEEE in the conference proceedings

- **Prior Approaches**: 초미세 반도체 공정의 진전으로 TEM(투과전자현미경) 데이터 수요가 급증했지만, 시료 준비 과정이 파괴적이고 촬영이 느리며 비용 부담이 커 데이터 확보가 어렵다. 이에 synthetic data 생성이 대안으로 떠올랐으나, 기존 생성 모델은 TEM 고유의 잡음, 세밀한 구조, 그리고 평가에 필요한 확률적 변이를 충분히 반영하지 못하는 한계가 있다.

- **Core Contribution**: 이 논문은 극단적 데이터 부족 상황에서 synthetic TEM 이미지를 생성하기 위한 DDPM(Denoising Diffusion Probabilistic Model) 프레임워크를 제안한다. 또한 생성 자체를 넘어, DDPM의 특징 표현을 분할(segmentation) 등에 재사용하고 영역 마스크를 일관되게 얻기 위해 파티션 기반 인코더 특징 맵 분할을 도입한다.

- **Technical Challenges**: 핵심 기술 도전은 (1) TEM 특유의 잡음·구조 디테일·stochastic variability를 모델이 학습하도록 하는 것, (2) 전체 이미지의 전역적 구조와 공간 관계를 깨지 않게 합성하는 것이다. 저자들은 저해상도 패치에서 고해상도/전체 이미지로 확장하는 progressive patch-based training으로 15 samples만으로 from-scratch 학습을 가능하게 하고, TrivialAugment 커스텀, cross-process domain transfer, classifier guidance, RePaint-style inpainting을 결합해 FAB metrology 요구를 만족하는 합성을 달성한다.

- **Empirical Impact**: 생성 결과는 MS-SSIM 기준 최대 0.98을 상회하며, 전문가성 정성 평가도 구조적 유사성 결과와 일관된 품질을 보였다. 더 나아가 synthetic 이미지는 결함 탐지·세그멘테이션·메트롤로지 같은 downstream ML 학습에 활용 가능하고, 통계적·물리적 사실성을 보존한다는 점에서 반도체 기반 비전 분야의 데이터 제약 문제를 완화하는 의미가 있다.



### DDStereo: Efficient Dual Decoder Transformers for Stereo 3D Road Anomaly Detection (https://arxiv.org/abs/2606.24805)
- **Prior Approaches**: 기존 stereo 기반 3D 탐지는 monocular 대비 정확도가 높지만, 추론 속도가 더 느려 실시간 적용이 어렵다는 한계가 있었다. 또한 대부분이 closed-set 가정에 머물러 학습에 없던 OoD(Out-of-Distribution) 장애물을 안정적으로 ‘발견’하는 데 약했다. 오픈셋/오픈보캐벌(open-vocabulary)은 주로 monocular에서 활발했지만, stereo 기반 오픈셋 3D 탐지는 상대적으로 탐색이 부족했다.

- **Core Contribution**: DDStereo는 Dual-Decoder Stereo Transformer로, 실시간(open-set 포함) stereo 3D object detection을 목표로 한 최초의 end-to-end 설계를 제시한다. 텍스트 프롬프트 없이도 OoD를 분리해내기 위해, 하나의 디코더는 open-set foreground(이진) 위치를 disparity 기반으로 찾고 다른 디코더는 3D 속성(다중 클래스/기하)을 회귀한다. 두 분기에서 object-level query를 공유해 박스 정렬 문제를 완화하고, 일관성 기반 MNPF 점수로 anomaly를 추정한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) disparity로 foreground를 뽑을 때와 (2) 클래스가 포함된 3D attribute 회귀 박스의 정합을 맞추는 문제, 그리고 (3) 오픈셋 신뢰도 계산이 query/decoding 정렬에 민감하다는 점이다. DDStereo는 shared object queries로 두 디코더의 공간적 일관성을 확보하고, 채널 결합으로 인한 연산 병목을 피하려 compact disparity feature extractor와 depth-to-box center sampling 방식(깊이 맵 예측 후 객체 중심에서 보정)을 도입해 효율을 유지한다. 추가로 NMS/anchor 기반 정렬을 배제하고 Transformer의 end-to-end 흐름에서 MNPF로 anomaly confidence를 계산한다.

- **Empirical Impact**: KITTI stereo benchmark의 closed-set 및 KITTI-AR-OoD의 open-set에서 state-of-the-art 정확도를 보이며, 특히 실시간 성능을 강조해 프레임당 23.5ms로 기존 stereo 대비 큰 속도 이점을 보인다. OoD(unknown) 클래스에서도 알려진 클래스와 함께 일관된 향상을 보이며, BEV localization에서 S3AD 대비 개선 폭이 두드러진다. ablation 결과 shared query, dual-decoder 분리, Sigmoid 기반 정규화 등 설계 요소가 open-set AP에 직접적인 기여를 했고, 성능-효율 균형을 실사용 관점에서 입증했다.



### OrbitForge: Text-to-3D Scene Generation via Reconstruction-Anchored Video Synthesis (https://arxiv.org/abs/2606.24799)
Comments:
          40 pages, 33 figures, 19 tables

- **Prior Approaches**: 기존 3D 생성은 2D/비디오 확산 모델의 사전지식을 활용하지만, 일반적으로 프롬프트마다 SDS(score-distillation sampling) 최적화를 길게 수행하거나, 고정된 멀티뷰 포맷/카메라 격자를 전제로 학습된 멀티뷰 생성기를 쓰는 경우가 많다. 이로 인해 오버스무딩·과포화, Janus 아티팩트, 약한 장면 맥락, 혹은 카메라 제어 불가로 “렌더링은 그럴듯하지만 재사용 가능한 3D 자산”으로 이어지기 어렵다. 로컬 novel-view 보강 방법은 짧은 궤적 완성에는 강하지만, 생성 비디오의 불일치를 그대로 안고 닫힌 360도 오빗(orbit) 장면 생성과 검증을 동시에 만족시키는 데는 한계가 있다.

- **Core Contribution**: OrbitForge는 단 하나의 text-to-video 생성 결과를 입력으로 받아, 닫힌 closed-orbit 3D Gaussian Splatting 씬을 만들기 위한 “어댑터”를 제안한다. 핵심은 첫 복원에서 얻은 좌표계(스캐폴드)를 앵커로 삼고, 정해진 canonical orbit에서 부족한(unsupported) 뷰 구간만 비디오 priors로 보완한 뒤, 다시 한 번 동일 카메라 시스템으로 재구성해 오빗을 닫는 것이다. 특히 per-prompt SDS 최적화나 task-specific 멀티뷰 fine-tuning 없이, frozen 비디오 프라이어를 그대로 사용하면서 장면 단위의 3D 일관성을 끌어올린다는 점을 강조한다.

- **Technical Challenges**: 가장 큰 기술 난제는 text-to-video의 카메라 모션이 암묵적이고 부분 호(arc)만 커버하며, 시간에 따라 대상/배경이 흔들려 프레임 간 불일치가 3D 피팅 시 floaters·흐림·늘어난 표면·뷰 의존 아티팩트로 번진다는 것이다. OrbitForge는 Deformable Gaussian Splatting으로 1차 복원을 만든 뒤, 이를 MedianGS로 정적 구조 프록시로 압축해 canonical orbit 렌더링의 “지지/미지지 영역”을 판별한다. 이어서 endpoint-window 방식으로 미지원 구간만 완성하고, 완성된 orbit를 미리 지정된 canonical 카메라 인덱스에 고정해 2차 복원을 수행함으로써, 생성된 프레임이 유도하는 의사-동역학(pseudo-dynamics)이 좌표계 붕괴로 이어지지 않게 한다.

- **Empirical Impact**: 평가에서는 로컬 프레임 스무스만으로는 공정하지 않다고 보고, coverage-aware 지표인 measured angular span 등을 포함한 closed-orbit 프로토콜을 사용한다. 냉동 300-prompt T3Bench-derived audit에서 OrbitForge는 MedianGS-only 대비 unsupported 구간의 Q10 ImageReward를 8.07→16.36으로 크게 끌어올리면서, measured median span도 359.0-degree로 동일 수준을 유지한다. 또한 coverage까지 고려했을 때 VideoMV와 경쟁력 있는 품질-커버리지 균형을 보여, “닫힌 360도 오빗 3D 생성”이라는 목표에 맞춰 실증 효과를 입증한다.



### EG-VQA: Benchmarking Verifiable Video Question Answering with Grounded Temporal Evidenc (https://arxiv.org/abs/2606.24797)
- **Prior Approaches**: 기존 VideoQA 벤치마크는 주로 정답 일치 여부(정답 정확도)로 평가되며, 비디오 내 근거가 되는 시간 구간을 실제로 찾아냈는지는 충분히 검증하지 못했다. 또한 다지선다 포맷은 option bias를 유발해 언어적 지름길로 답을 맞힐 위험이 있고, 개방형 설정에서도 평가가 정답 생성과 근거(증거) 정합성을 분리해 보는 경향이 강하다.

- **Core Contribution**: 이 논문은 답을 맞추는 것뿐 아니라 시간적으로 국소화된 증거를 함께 제시하도록 요구하는 Evidence-Grounded Video Question Answering Benchmark(EG-VQA)를 제안한다. EG-VQA는 2,067개 비디오와 11,838개 QA 쌍에 대해 세밀한 temporal evidence를 텍스트 설명과 함께 주석하며, 서술·시간·인과·반사실(counterfactual) 추론 유형을 포괄한다.

- **Technical Challenges**: 핵심 과제는 ‘증거가 맞는지’를 정답 정확도와 결합해 일관된 방식으로 평가하고, 모델 학습에도 신호를 촘촘히 제공하는 것이다. 이를 위해 Evidence-Grounded F1(EG-F1)을 time alignment(IoU)와 semantic consistency를 동시에 묶어 최적 매칭(Hungarian algorithm)으로 계산하도록 설계했고, EG-Reasoner는 evidence 블록과 evidence-to-reasoning-to-answer 구조화된 출력에 대해 GRPO 기반 reinforcement learning과 soft matching 보상을 결합해 초기 학습의 희소성 문제를 완화했다.

- **Empirical Impact**: 실험 결과, 강한 Video-LLM일수록 정답 정확도는 높아도 evidence grounding은 낮게 나타나며, 정답 맞춤과 근거 충실성 사이의 근본적 불일치가 확인됐다. EG-Reasoner는 오픈소스 모델 중 최고 성능을 보이며 GPT-4o 대비 특히 counterfactual 질문에서 더 큰 향상을 보였고, evidence reward를 제거하거나 hard EG-F1만 쓰면 성능이 크게 떨어져 구조화된 증거 감독의 필요성이 실증됐다. 결론적으로 본 연구는 scaling만으로는 신뢰 가능한 비디오 이해가 어렵고, structured evidence supervision이 해석 가능하고 검증 가능한 VideoQA 시스템 개발에 필수임을 보여준다.



### Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM (https://arxiv.org/abs/2606.24796)
Comments:
          2026 IEEE International Conference on Robotics and Automation(ICRA)

- **Prior Approaches**: 3D Gaussian Splatting(3DGS)을 SLAM에 적용한 연구들은 고해상도 기하 표현과 고품질 novel view 합성을 강점으로 내세워 왔지만, 야외 대규모 장면에서는 Gaussian 점이 계속 누적되며 peak 메모리가 시간에 따라 증가하는 문제가 남아 있다. 기존 pruning 연구는 주로 opacity나 gradient magnitude 같은 Gaussian 단위 휴리스틱에 초점을 두거나, 3DGS-SLAM의 keyframe 저장 최적화에 머무는 경우가 많아 실제 runtime peak memory 절감과 정확도 동시 보장에 한계가 있다. 특히 실내/야외는 텍스처 밀도 분포가 달라, 면적 기반 제거가 야외에서 특정 영역의 Gaussian이 사실상 “비게 되는” 부작용을 유발하기 쉽다.

- **Core Contribution**: 본 논문은 3DGS-SLAM을 위한 Pocket-SLAM을 제안하며, Gaussian을 삭제할 때 opacity·gradient 같은 로컬 기준이 아니라 ‘렌더링에 실제로 얼마나 기여하는지’를 기준으로 중요도를 평가하는 rendering-area-aware pruning을 도입한다. 여기서 각 Gaussian의 이미지 평면 유효 픽셀 커버리지를 계산해, 결과 렌더링 이미지 관점에서 중복을 줄이도록 설계했다. 또한 렌더링 면적 기반 pruning만 쓰면 텍스처 밀집 영역에서 정보 손실이 커질 수 있어, 타일 단위 budget으로 pruning 비율을 영역별로 제한해 과도 제거를 방지한다.

- **Technical Challenges**: 핵심 기술 난제는 ‘어떤 Gaussian이 현재 프레임에서 추적/재구성에 결정적인가’를 메모리 절감 목표와 함께 정량화하는 것이다. 논문은 tracking 단계에서 얻는 Gaussian gradient 정보를 타일별로 평균해 텍스처 밀집/희소 영역을 구분하고, 이를 기반으로 타일별 survivor budget을 배정함으로써 dense·sparse 모두에서 완전 소거를 막는다. 이후 mapping 수렴 후 각 타일에서 유효 픽셀 커버리지 기반 중요도 순으로 budget 내에서 Gaussian을 선택적으로 pruning하여, 렌더링 기여도와 균형 있는 분포를 동시에 유지한다.

- **Empirical Impact**: EuRoC와 KITTI(야외 대규모) 실험에서 Pocket-SLAM은 기존 pruning 방식 대비 안정적으로 더 나은 tracking(Ate)과 재구성 지표(PSNR/SSIM/LPIPS)를 유지하면서 peak 메모리를 크게 줄였다. 특히 큰 장면에서 메모리는 60% 이상 감소했고 FPS는 2배 이상 향상되며, 정확도 저하 없이 runtime 효율 병목을 완화하는 것이 확인됐다. 또한 rendering-area-aware pruning만 적용한 경우보다 tile-level budget을 함께 쓸 때 ATE/PSNR이 훨씬 개선되어, 야외 텍스처 분포 차이를 고려한 균형 제약의 실효성이 실증됐다.



### Counting Trees from Satellite Imagery with Noisy Supervision (https://arxiv.org/abs/2606.24786)
- **Prior Approaches**: 위성 기반 트리 카운팅은 대부분 산림 피복률/수목 바이오매스/수관 고도 같은 집계 지표에 머물렀고, 개별 수목 수준은 아직 본격적으로 다뤄지지 않았다. 기존 연구는 크게 (1) 포인트→밀도맵 회귀, (2) 박스/포인트 기반 detection, (3) balanced optimal transport를 이용한 distribution matching, (4) 베이지안 크라우드카운팅류로 나뉘지만, 수관 경계가 불명확한 조밀한 숲에서 분리 가능성 가정이 깨져 성능이 흔들린다. 또한 항공 LiDAR로 만든 대규모 라벨은 구조적이고 상관된 잡음이 있어, 이를 정교하게 학습에 반영하는 공식을 갖추지 못한 점이 한계로 지적된다.

- **Core Contribution**: 이 논문은 트리 카운팅을 “공간 밀도 매칭” 문제로 재정의하고, Unbalanced Optimal Transport(UOT)로 학습을 감독하는 TreeMatch를 제안한다. UOT은 질량(총 개수) 불일치를 허용하면서도 공간 정렬은 유지해, 고립 수목은 정밀 위치화하고 조밀 숲에서는 경계 애매함에 견디는 학습이 가능하다. 더불어 transport residual(수송 잔차)을 이용한 self-correction으로, ALS 등 약한(supervision) 라벨의 노이즈를 학습 과정에서 점진적으로 보정한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 위성 해상도에서 수관 크기와 중심 정의가 불안정하고, (2) 고립-조밀-중간 밀도가 한 장면에 공존하며, (3) 대규모 약한 라벨은 누락·허위가 섞여 있다는 점이다. TreeMatch는 픽셀 단위 밀도(z)와 라벨 분포(y)를 정규화하지 않은 비확률 질량으로 두고, KL 회귀 대신 OT 비용(좌표 기반 거리)으로 공간 오차를 완화한다. 또한 self-correction은 UOT 해에서 얻은 marginal residual로 “과소/과대”된 지역의 감독 신호를 재가중하며, 계산 그래디언트가 교정 단계로 역전파되지 않도록 stop-gradient를 적용해 학습 안정성을 확보한다.

- **Empirical Impact**: 새 벤치마크 TinyTrees는 3개 대륙·3개 위성 센서를 아우르며 23,000km² 이상, 2.15억 개 이상 트리 어노테이션(이 중 773K 수동 검증)을 포함한다. TreeMatch는 detection 기반, 회귀 기반, 기존 transport 기반(balanced OT 포함) 기준모델을 모든 지표에서 일관되게 능가하며, especially 조밀 숲/약한 감독 조건에서 강건함이 확인된다. 또한 1회 forward만으로 추론이 가능하고 학습도 빠르게 수렴해 대규모 배치 적용에 적합하다는 실용성까지 함께 보여준다.



### AerialFusionMapNet: Online HD Map Construction with Aerial-Onboard BEV Fusion (https://arxiv.org/abs/2606.24784)
Comments:
          Accepted at the IEEE International Conference on Intelligent Transportation Systems (ITSC) 2026

- **Prior Approaches**: 온라인 HD 맵 구성은 차량 온보드 센서에서 BEV 기반으로 차선/도로 구조를 벡터로 예측하는 쪽으로 발전했지만, 가림과 시야 한계 때문에 성능이 제한될 수 있다. 이에 대응해 위성·항공의 고해상도 aerial 이미지를 BEV에 융합한 연구들이 가능성을 보였으나, 기존의 end-to-end 융합은 aerial이 담는 구조 정보를 충분히 활용하는 학습 설계가 부족하다는 한계가 지적된다.

- **Core Contribution**: AerialFusionMapNet은 aerial-onboard 융합을 “구조를 반영하는 학습” 관점에서 재구성한 프레임워크다. 핵심은 aerial 인코더를 먼저 aerial-only로 학습해 기준 표현을 만들고, 이후 joint 학습 단계에서 Cross-View Supervision(CVS)을 통해 BEV 표현을 정렬해 구조적 prior를 더 효과적으로 결합한다.

- **Technical Challenges**: 어려움은 aerial 표현과 온보드 BEV 표현이 서로 다른 통계/유도편향을 갖는 상태에서 단순 end-to-end 학습으로는 표현 정합이 잘 안 된다는 점이다. 논문은 시나리오 일관 회전인 Scenario-Consistent Rotation(SCR)로 aerial 기준 표현의 회전 강건성을 확보하고, CVS 학습 시 onboard 특징에 가벼운 affine alignment을 추가해 feature statistics 차이를 완화함으로써 융합 정합을 안정화한다.

- **Empirical Impact**: nuScenes geographic split에서 AerialFusionMapNet은 최대 54.7 mAP를 달성했으며, 기존 aerial-onboard fusion 베이스라인 48.8 mAP 대비 +5.9(절대)·+12.1%(상대) 향상이다. 또한 encoder 용량을 키우는 것보다 2-stage 학습 설계가 성능을 좌우하며, 데이터 정합 오차 수준의 translational misalignment에도 일정 범위(대략 수 m가 아닌 수십 cm 수준)에서는 성능이 비교적 안정적임을 보여준다.



### Revealing Training Data Exposure in Vision Language Large Models via Parameter Gradients (https://arxiv.org/abs/2606.24774)
- **Prior Approaches**: 기존 training data detection(데이터 오디팅) 연구는 주로 단일 모달 모델을 대상으로 하거나, VLLM에서는 토큰 확률·출력 엔트로피 같은 겉보기 신호에 의존하는 경우가 많았다. 그 결과 cross-modal 입력에서는 신호가 약해 AUROC가 무작위 수준(대략 50–55%)에 머무르는 설정이 자주 관찰된다. 또한 일부 multimodal 전용 방법도 아키텍처/학습 목적에 대한 일반화가 제한적이라는 문제가 제기된다.

- **Core Contribution**: 이 논문은 VLLM을 black box로 취급하지 않고, 내부 최적화 동역학을 반영하는 gradient 기반 감사를 수행하는 GradAudit을 제안한다. 핵심 관찰은 학습 샘플에 대해 계산한 그래디언트는 더 안정적이고 잘 정렬되는 반면, 비학습 샘플의 그래디언트는 잡음이 많고 일관성이 떨어진다는 gradient signature의 비대칭성이다. 이를 바탕으로 GradAudit은 이미지와 텍스트의 개별 포함 여부를 넘어서, 실제로 학습된 image-text 연관(쌍)을 식별하도록 설계됐다.

- **Technical Challenges**: 가장 큰 기술적 난관은 수십억 파라미터 전체의 그래디언트를 그대로 쓰기엔 계산이 불가능하다는 점이다. GradAudit은 파라미터 그래디언트 행렬을 기능적으로 해석 가능한 slice로 분해해 feature vector로 만들고, 이어서 민감하지 않은 기능을 기준(reference) 데이터로 noise masking해 억제한 뒤, 기준 학습 데이터와의 gradient similarity로 membership을 판정한다. 또한 fine-tuning과 같이 실제 환경에서 중요한 설정에서도, 참조 데이터와 유사도 비교가 오탐을 줄이면서 신호를 유지하도록 설계했다.

- **Empirical Impact**: 실험에서 GradAudit은 총 7개 구성에서 모든 baseline을 일관되게 능가하며, pretraining 시나리오 AUROC 87.2%, fine-tuning 시나리오에서는 최대 92.7%까지 도달한다. 특히 shuffling으로 만든 비학습 쌍을 포함해 joint image-text 매칭을 검증하는 제어 실험에서 85.5% AUROC를 기록해 ‘개별 모달 포함’ 수준을 넘는 성능을 보였다. 더 나아가 Studio Ghibli 사례 연구에서는 최신·고성능 모델일수록 GradAudit이 기존 방법보다 더 큰 노출 비율(최대 약 3배)을 추정해, 기존 데이터 오디팅이 무단 사용을 과소평가했음을 시사한다.



### Compact Object-Level Representations with Open-Vocabulary Understanding for Indoor Visual Relocalization (https://arxiv.org/abs/2606.24767)
Comments:
          Accepted by RA-L 2026

- **Prior Approaches**: 기존 실내 visual relocalization은 주로 low-level 특징에 의존해 조명 변화나 센서 노이즈에 취약했고, 의미·구성을 충분히 반영하지 못했다. 또한 점(point) 기반 맵은 오버헤드가 커지고, downstream에서 의미 객체 대신 원시 특징점에 가까워 해석성과 적용성이 떨어졌다. 최근 일부 object-level 접근도 있으나, 객체 매칭의 outlier가 크고 scalable 장면을 위한 pose prior·전용 최적화 전략이 부족했다.

- **Core Contribution**: OpenReLoc은 semantics, layout, geometry를 포함한 객체 정보를 object-level structured map으로 정리해 camera relocalization을 수행하는 프레임워크를 제안한다. 이를 위해 multi-modal open-vocabulary object matching, object-oriented reference frame 기반 pose prior, 그리고 객체 형상 정보를 활용한 dual-path 2D ICP loss로 coarse-to-fine 추정을 안정화한다. 결과적으로 의미 인식 기반의 정밀한 6-DoF pose 추정과 메모리 효율·확장성을 함께 노린다.

- **Technical Challenges**: 핵심 기술 과제는 (1) open-vocabulary 객체 매칭에서의 낮은 분별력과 outlier를 줄이고, (2) 큰 실내 장면에서 재검색 가능한 pose prior를 제공하며, (3) sparse한 객체 대응에서도 안정적으로 pose를 최적화하는 것이다. 논문은 CLIP 기반 시각-언어 객체 디스크립터와 global scene graph의 sub-graph matching을 결합해 매칭을 정교화하고, RGB 없이 관측된 object ID와 2D bbox만 담는 reference frame과 DIOU retrieval로 대규모 장면의 coarse prior를 만든다. 마지막으로 2D ICP를 forward/backward 모두에 대해 설계해 scale ambiguity를 완화하고, Huber 커널로 2D 픽셀 outlier에 둔감하도록 하여 정밀 최적화를 달성한다.

- **Empirical Impact**: 실험에서 OpenReLoc은 ScanNet·ScanNet++ 및 Habitat 기반 대규모 합성 장면(HSSD)에서 기존 object-level 및 low-level 베이스라인보다 recall과 정확도를 일관되게 개선했다. 특히 ScanNet에서 GoReloc 대비 relocalization success rate가 약 5~10배 수준으로 향상되었고, 대규모 다중 룸/다중 플로어 설정에서도 확장성과 강건성이 확인됐다. 모델이 closed-vocabulary 의존도가 낮아 복잡한 실제 환경에서도 더 많은 유효 매칭을 생성하며, 전용 최적화 loss의 부재가 초래하는 drift·수렴 불안정 문제를 줄였다는 점이 성능 차이를 설명한다.



### UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving (https://arxiv.org/abs/2606.24759)
- **Prior Approaches**: 기존 multimodal large language model(MLLM) 기반 자율주행 장면 이해는 성과가 있지만, 시간 추론과 공간 정밀도 사이에 근본적인 트레이드오프가 남아 있다. 단일 프레임/저해상도 입력 중심 방식은 작은·먼·부분 가림 위험을 놓치기 쉽고, 언어 중심 접근은 설명의 근거가 충분히 grounded 되지 않는 문제가 있다.

- **Core Contribution**: 이 논문은 UniDrive로, 해석 가능한 위험(risk) 이해를 위해 visual-language와 grounding을 하나의 프레임워크에서 통합한다. 멀티프레임으로 장면 동역학을 잡는 temporal reasoning branch와 최신 프레임의 고해상도 세부를 보존하는 perception branch를 결합하고, gated cross-attention으로 두 표현을 정렬해 위험 객체에 대한 자연어 설명과 바운딩 박스를 함께 생성한다.

- **Technical Challenges**: 핵심 난제는 동적 맥락(시간 의미)을 정밀한 공간 근거(위치/가시성)와 동시에 일관되게 정렬하는 것이다. UniDrive는 두 가지 표현을 gated cross-attention fusion으로 결합해 상황 전개 정보와 고해상도 증거를 맞물리게 만들고, fused representation을 기반으로 언어 생성과 grounded bounding-box 출력을 공동으로 수행한다.

- **Empirical Impact**: DRAMA-Reasoning benchmark에서 UniDrive는 이미지 기반·비디오 기반 대표 baseline 대비 captioning과 위험 객체 grounding 모두에서 향상됐다. 특히 small-object localization, NuScenes·BDD100K로의 zero-shot 일반화, 사람 평가 기반 interpretability/trustworthiness에서 강점이 확인되며, 시간 의미와 고해상도 지각의 명시적 결합이 안전 지향 자율주행 시스템의 기반이 될 수 있음을 시사한다.



### Adaptive Hebbian Memory Routing in Vision Transformers for Few-Shot Learning (https://arxiv.org/abs/2606.24756)
- **Prior Approaches**: 퓨샷 이미지 인식은 소수 라벨 support set로 새로운 클래스를 맞혀야 하지만, 대부분의 신경망 지식은 추론 중 고정된 slow weights에 저장돼 빠른 적응이 어렵다. 프로토타입 기반(metric) 방법은 support로 만든 class prototype과의 거리로 분류하지만, 쿼리를 잘 표현하는 task-specific representation이 백본에서 만들어지느냐가 성패를 좌우한다. Hebbian fast-weight는 에피소드 동안 임시 연관 정보를 담을 수 있으나, 고정된 memory 위치/활성·업데이트 규칙은 모델과 작업에 따라 효과가 크게 달라질 수 있다는 한계가 있었다.

- **Core Contribution**: 이 논문은 few-shot Vision Transformer에 대해 Adaptive Hebbian Routing을 제안한다. lightweight MLP router가 support-set 특징을 입력으로 받아, (1) Hebbian memory의 현재 기여도, (2) memory 업데이트의 plasticity 강도, (3) episode 동안의 retention(감쇠)을 에피소드 조건으로 제어한다. 이를 통해 고정 Hebbian 동작을 모든 few-shot task에 동일하게 적용하는 문제를 해결한다.

- **Technical Challenges**: 핵심 기술 난점은 “어디에” Hebbian을 넣는 것만으로는 부족하고, episode마다 “어떻게” 업데이트하고 “얼마나” 오래 유지할지까지 안정적으로 학습·제어해야 한다는 점이다. 논문은 각 selected location마다 별도 router를 두고, support 특징을 요약(mean pooling)해 control 값을 예측한 뒤 Hebbian readout과 fast-weight write/decay를 라우팅하는 구조로 해결한다. 또한 router 오버헤드를 줄이기 위해 hidden dimension을 작게 설계하고, 초기값을 보수적으로 두어 학습 초기에 과도한 fast-weight 업데이트가 발생하지 않도록 했다.

- **Empirical Impact**: Omniglot 5-way 1-shot에서 Swin을 직접 비교하면, fixed Hebbian 대비 Adaptive Plasticity는 96.74%→96.92%로 개선되고 Fully Adaptive Routing은 96.94%로 최고 성능을 보인다. 뿐만 아니라 Swin의 경우 inference time도 fixed Hebbian 16.51 ms에서 Fully Adaptive 14.05 ms로 단축된다. CIFAR-FS에서는 모든 백본에서 성능이 개선되며 1-shot뿐 아니라 10-shot까지 유사한 이득이 유지되고, CIFAR-FS→Omniglot cross-domain transfer도 백본별로 차이는 있으나 adaptive 제어가 전반적인 적응성을 높일 수 있음을 보여준다.



### BioMedVR: Confusion-Aware Mixture-of-Prompt Experts for Biomedical Visual Reprogramming (https://arxiv.org/abs/2606.24740)
Comments:
          Accepted at ECCV 2026. 19 pages, 6 figures. Project page: this https URL

- **Prior Approaches**: 기존 vision-language model(VLM)은 CLIP처럼 자연 영상에서 강한 zero-shot 일반화 성능을 보였지만, 의료 영상으로의 전환은 계산 비용과 데이터 희소성, 미세한 클래스 간 구분 때문에 까다롭다. prompt learning(예: CoOp, CoCoOp, MaPLe, BiomedCoOp)은 매개변수 효율적이지만 의료의 fine-grained 단서(미세 혈관·조직 질감 등)를 충분히 모델링하기 어렵고, VR은 대체로 단일 시각 패턴/프롬프트에 의존해 혼동이 큰 상황에서 과신되기 쉽다.

- **Core Contribution**: BioMedVR은 biomedical imaging에 Visual Reprogramming(VR)을 처음으로 체계적으로 적용해, VLM을 full-model fine-tuning 없이 few-shot으로 적응시키는 프레임워크를 제안한다. 핵심은 Confusion-aware Mixture-of-Prompt Experts(MoPE)로, positive expert로 본질(주 클래스) 정합을 강화하고 negative expert가 LLM이 생성한 confusion-aware 속성으로 혼동을 억제하도록 설계했다. 여기에 Confusion-Suppression Loss로 true class와 혼동 클래스를 가르는 의미적 margin을 명시적으로 키운다.

- **Technical Challenges**: 의료 데이터에서 클래스 간 코사인 유사도가 지나치게 높아지면 softmax가 평평해지고 학습 신호가 약해져 VR 최적화가 둔화된다. 또한 단일 shared perturbation은 MRI/CT 등 이질적 modality에서 의미 경계 정렬이 흔들려 혼동이 더 커질 수 있다. BioMedVR은 positive/negative 두 전문가와 LLM 기반 혼동 속성을 분리하고, 가장 유력한 대안 클래스에 대한 margin을 확대하는 CS loss로 decision boundary를 안정화하며, gating 모듈로 두 경로의 기여를 자동 균형 조절한다.

- **Empirical Impact**: BioMedVR은 11개 biomedical 데이터셋(9개 modality)과 7개 자연 영상 벤치마크에서 기존 prompt-learning 및 VR 대비 정확도와 일반화 성능을 일관되게 개선했으며, few-shot과 zero-shot 모두에서 격차를 줄였다. 특히 혼동이 큰 Knee X-ray·DermaMNIST 등에서 개선 폭이 두드러졌고, CLIP 대비 zero-shot 평균 정확도를 28.1%→31.6%(BiomedCLIP 백본에서는 47.6%)로 끌어올렸다. 아블레이션과 시각화 결과는 negative expert·confusion-aware 속성·CS loss가 임베딩 공간에서 클래스 클러스터 분리를 강화하고 캘리브레이션 안정성에도 기여함을 보여, 의료 진단형 few-shot 적응의 실용적 대안으로 자리매김한다.



### VSANet: View-aware Sparse Attention Network for Light Field Image Denoising (https://arxiv.org/abs/2606.24737)
- **Prior Approaches**: 기존 LF denoising은 뷰 간 잡음 독립성과 장면의 강한 공간-각도 상관을 활용하지만, LFBM5D 같은 방식은 hand-crafted prior에 의존해 복잡한 LF 구조에 대한 적응성이 떨어진다. 딥러닝 기반 접근도 epipolar geometry 등 미리 정한 분해에 묶여 전체 4D에서의 장거리 의존성을 충분히 모델링하기 어렵고, transformer 계열은 LF를 2D 부분공간으로 나눠 공간-각도 위치 간 직접 상호작용이 제한된다.

- **Core Contribution**: VSANet은 4D LF 전체를 단일 spatial-angular 토큰 공간으로 통합하고, view-aware sparse attention(VSA)로 전 뷰-전 위치 간 전역 상호작용을 수행한다. VSA는 locality-sensitive hashing(LSH) 기반 sparse attention으로 전역 비국소 집계를 효율화하고, feature refinement(FR)로 공간·각도·에피폴라 하위공간에서 유의미한 특징을 강화한다.

- **Technical Challenges**: 핵심 난제는 토큰 수가 S·T·H·W로 커지는 LF에서 full attention의 O(N^2) 복잡도를 현실적으로 감당할 수 없다는 점이다. 저자들은 LSH로 유사 토큰을 해시 버킷에 모으고, 여러 해시 라운드를 confidence-weighted 방식으로 결합하며, 버킷 내부에서는 로컬 윈도우 attention만 계산해 계산량을 선형에 가깝게 줄이도록 설계했다.

- **Empirical Impact**: 실험에서 VSANet은 STFLytro 및 EPFL 벤치마크에서 세 가지 잡음 수준(σ=10/20/50) 모두에 대해 PSNR·SSIM 기준 성능을 SOTA 대비 우수하게 달성했으며, 특히 runtime과 파라미터 효율도 더 좋다고 보고된다. 또한 정성 비교에서 잡음을 더 잘 제거하면서 구조 디테일을 더 잘 보존했고, ablation study로 VSA와 FR, 그리고 FE의 다중 하위공간 설계가 성능에 기여함을 확인했다.



### SER: Learning to Ground Video Reasoning with Semantic Evidence Rewards (https://arxiv.org/abs/2606.24726)
- **Prior Approaches**: 기존 비디오 MLLM은 미세한 시공간 추론에서 자주 실패하며, 정답이 맞더라도 관련 없는 프레임이나 객체에 근거하는 현상이 보고된다. 시공간 근거를 출력하도록 유도하는 RL 접근은 흔히 IoU 같은 geometry-only 보상에 의존해 경계 변화에 민감하고, 의미적 정렬(semantic alignment)은 놓치기 쉽다.

- **Core Contribution**: 이 논문은 spatio-temporal evidence grounding을 제약이 있는 verification 과제로 재구성해 Semantic Evidence Reward(SER)를 제안한다. SER는 pixel-level overlap 대신 referee VLM(로컬 체크커)이 증거 주장(evidence claims)을 관련성(relevance)과 localization 품질의 두 축에서 검증하고, 여기에 시간 패널티를 결합해 학습을 유도한다.

- **Technical Challenges**: 핵심 난제는 밀도 높은 박스 주석 없이도 근거의 품질을 신뢰성 있게 평가해 RL 학습 신호로 만들 수 있느냐였다. SER는 referee VLM을 이용해 국소 검증을 수행하고, 관련성·위치 품질·시간 제약을 함께 반영하는 보상 설계로 표준 video QA 데이터만으로도 직접 학습되도록 했다.

- **Empirical Impact**: V-STAR 벤치마크에서 SER은 49.6% mLGM를 달성했으며, strong evidence-grounded baseline인 Open-o3-Video 대비 3.0포인트 향상됐다. 이는 정답 정확도뿐 아니라 evidence grounding까지 개선할 수 있음을 실증하며, 비디오 QA에서 시공간 근거 학습 방향에 의미 있는 진전을 제공한다.



### Evaluating the Interpretability of Sparse Autoencoders with Concept Annotations (https://arxiv.org/abs/2606.24716)
Comments:
          Accepted at ECCV 2026

- **Prior Approaches**: 기존 sparse autoencoders(SAE) 평가는 주로 sparsity나 reconstruction 같은 구조적 지표, 또는 정성적 사례 중심에 머물렀다. 기능적 평가로서 개입(intervention)을 시도해도, 비전에서는 속성 변화가 ‘정확히 하나만’ 분리된 반사실 데이터 구성이 어려워 의미 대응(semantic correspondence)을 정량화하기가 힘들었다.

- **Core Contribution**: 이 논문은 SAE latent와 사람의 주석된 개념/속성 간 정렬(alignment)을 사람 기준에 근거해 수치화하는 평가 프레임워크를 제안한다. 사용자 연구 없이도(별도 설문 없이도) latent-개념 매칭과, 매칭된 개념이 개입된 이미지 변화에 선택적으로 반응하는지 확인하는 TAPAScore를 통해 의미적 대응을 검증한다.

- **Technical Challenges**: 핵심 난제는 비전 데이터에서 ‘한 속성만 바꾼’ 통제된 쌍을 만들 수 없다는 점이며, 이를 위해 synCUB와 synCOCO를 합성 데이터로 구축해 정확히 하나의 속성/객체만 달라지게 했다. 또한 feature splitting 같은 실패를 반영해, Fully-Binary Matching Pursuit(FBMP)로 latent를 one-to-one이 아니라 many-to-one으로 조합 매칭하고, 논리 연산 기반의 residual 업데이트로 binary 도메인에서 매칭 탐색을 수행한다.

- **Empirical Impact**: 실험에서 MATCHScore의 기존 one-to-one 기준과 여러 대리 지표(FMS, MS, CKNNA)는 sanity check에서 trained SAE를 untrained/random과 안정적으로 구분하지 못했지만, FBMP 계열 매칭과 TAPAScore는 강하게 분리했다. CLIP과 DINOv2 임베딩에서 overcompleteness(사전 크기)가 커질수록 perturbation alignment가 떨어져 해석 가능성이 감소할 수 있으며, dictionary size는 적당한 수준이 가장 좋은 trade-off를 보였다.



### Agentic Collaborative Cognition for Zero-Shot 3D Understanding (https://arxiv.org/abs/2606.24649)
Comments:
          Accepted by ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 zero-shot 3D 이해는 비디오에서 추출한 keyframe을 MLLM에 넣어 객체 중심으로 추론하는 방식이 많았지만, 영상이 제공하는 관측 시점이 제한적이라 query에 결정적인 관점이 빠지기 쉽다. 또 공간 배치를 키프레임에서 암묵적으로 추정하다 보니 관점이 크게 달라질 때 일관된 3D 인지(통합된 공간 인식)를 유지하기 어렵다. 결과적으로 불완전한 관측과 분절된 추론이 성능 병목이 된다.

- **Core Contribution**: 이 논문은 zero-shot 3D 이해를 “planning–perception”의 닫힌루프(interactive planning-perception)로 재구성하는 협업형 multi-agent 프레임워크를 제안한다. Planning Agent는 holistic cognitive map(인지 맵)을 바탕으로 query 관련 관점을 계획하고, Perception Agent는 3D 장면을 structured holistic cognitive map으로 명시적으로 문서화한다. 특히 Perception Agent가 시점 간에 일관된 instance identifier를 부여해 객체 속성을 통합하고, 불일치 후보를 되먹임으로 다음 관점 계획을 정교화한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 비디오의 유한 관측 시점만으로 부족한 “누락 관점”을 찾아 보완하는 것과 (2) 시점이 바뀌어도 객체 정체성을 유지하며 속성/공간 관계를 일관되게 업데이트하는 것이다. 이를 위해 Planning Agent는 인지 맵에서 객체의 category와 fine-grained attributes까지 함께 매칭하는 retrieval-augmented candidate filtering로 후보를 좁히고, 필요할 때 렌더링 이미지로 누락 관점을 보충한다. Perception Agent는 실이미지와 렌더드 이미지를 역할 분담해 속성은 실이미지에 우선하고, 관계/배치는 렌더드로 보강한 뒤 confidence 기반으로 속성을 갱신하며, attribute conformity·spatial consistency·observation sufficiency로 반복을 종료한다.

- **Empirical Impact**: 여섯 개 벤치마크에서 state-of-the-art 성능을 보였고, ScanRefer에서 Acc@0.5가 11.1% 개선, 3D-assisted dialog에서 BLEU-1이 14.6% 상승, SQA3D에서 EM이 2.1% 향상됐다. 또한 여러 태스크(시각적 grounding, 상황 추정, 질의응답, 대화/분해) 전반에서 성능이 고르게 좋아 “일관된 3D 인지와 관점 보완” 전략의 일반성이 확인된다. 반면 추론 시간은 반복 라운드에 따라 증가하지만 토큰 비용과 프레임 전수 샘플링을 줄여 효율도 함께 개선되는 양상을 보인다.



### ViTexQA: A Multi-Frame Temporal Perception Dataset for Video Text Question Answering (https://arxiv.org/abs/2606.24602)
Comments:
          Accepted by ECCV2026

- **Prior Approaches**: 기존 MLLM 기반 비디오 텍스트 QA는 시각-언어 결합을 강점으로 하지만, 많은 데이터셋이 단일 프레임에서도 대부분 문항을 풀 수 있게 설계되는 경향이 있다. 그 결과 모델은 시간축 통합보다는 프레임 단서 패턴 매칭으로 우회할 여지가 커지고, 실제 비디오에서 요구되는 cross-frame 텍스트 융합 능력이 과소평가된다.
또한 데이터셋 일부는 비디오 텍스트 인식·지각을 다루더라도, ‘각 문항이 반드시 다중 프레임 근거를 필요로 한다’는 검증 체계가 부족해 temporal reasoning을 정면으로 학습시키기 어렵다.

- **Core Contribution**: 이 논문은 모든 문항이 단일 프레임만으로는 답할 수 없도록 검증한 대규모 비디오-텍스트 QA 데이터셋 ViTexQA를 제안한다. 아울러 다중 프레임에서 흩어진 텍스트 단서를 시간 순서/공간 위치까지 포함해 통합하도록 설계된 FrameThinker 학습 방법을 제안한다.
ViTexQA는 품질 통제된 CoT(Chain-of-Thought) 주석 파이프라인과 시간 제약을 결합해 “진짜 temporal reliance”를 강제하며, FrameThinker는 이를 기반으로 explicit temporal modeling을 학습하도록 구성된다.

- **Technical Challenges**: 핵심 기술 과제는 비디오 속 텍스트 의미가 시간적으로 분산되어 나타날 때, 이를 학습 가능한 형태의 증거(언제/어디의 텍스트인지)로 안정적으로 제공하는 것이다. 이를 위해 논문은 keyframe에서 OCR을 수행하고, 동일/유사 텍스트를 시간-태그로 묶어 temporally-grounded CoT에 spatiotemporal evidence를 인터리브(interleave)하는 방식으로 교사 신호를 만든다.
학습 측면에서는 SFT만으로는 CoT 품질의 한계가 남을 수 있어, Temporally-grounded Reinforcement Learning을 통해 temporal coherence·근거 타당성·정답 정확도를 동시에 보상하도록 설계했다.

- **Empirical Impact**: 실험 결과, ViTexQA에서 최신 SOTA 대비 성능 격차가 관측되며, 이는 현 비디오 텍스트 모델들의 temporal perception 능력이 실제 요구 수준에 못 미친다는 점을 드러낸다. 또한 FrameThinker는 강력한 베이스라인을 능가하고 ROUGE-L을 6.3%p 향상시키는 등 다중 프레임 통합 학습의 효과가 확인된다.
이로써 비디오 텍스트 이해 분야에서 ‘시간에 의존하는 평가/학습’이 실제 성능 향상과 직결됨을 경험적으로 보여주며, 향후 비슷한 시간 제약형 벤치마크 설계에도 영향을 줄 전망이다.



### EERLoss: A Novel Loss Function for Training Deep Biometric Models. A Case Study in Keystroke Dynamics (https://arxiv.org/abs/2606.24586)
- **Prior Approaches**: 생체인증 딥러닝은 보통 embedding 분리도를 높이는 간접 목적(contrastive, triplet, ArcFace/CosFace/AdaFace 등)으로 학습해, 평가 지표인 EER과 최적화 목표가 어긋나는 문제가 있었다. 특히 Keystroke dynamics처럼 분산이 크고 잡음·희소성이 심한 행동형 생체에서는 이 불일치가 더 크게 드러날 수 있다. 기존 손실은 EER을 직접 타깃하지 않아, 결정 경계(보안-사용성 트레이드오프)가 중요한 상황에서 성능이 제한될 여지가 있다.

- **Core Contribution**: 이 논문은 EER 자체를 학습 목표로 정렬하기 위해 EERLoss를 제안한다. EERLoss는 subdifferentiable하면서도 임의의 정확도로 EER을 근사하며, DET curve의 특정 operating point도 함께 최적화하도록 확장 가능하다. 저자는 이를 keystroke dynamics verification에 적용해, 고변동/저변동 특성이 공존하는 어려운 생체 인증에서도 직접적인 지표 정합성을 입증하려 한다.

- **Technical Challenges**: 가장 큰 기술적 난관은 EER이 FAR과 FRR의 교점(비미분·단계함수)이라서 손실로 그대로 넣기 어렵다는 점이다. 저자들은 FAR/FRR을 t​anh 기반 매끄러운 근사로 만들고, 이를 subdifferentiable binary search로 최적 임계값 dEER을 추정한 뒤, EER 및 FAR-FRR 곡선 겹침 영역을 학습 가능한 형태로 정의해 end-to-end 최적화를 가능하게 한다. 또한 먼 구간의 값이 과도하게 기여하지 않도록 거리 가중(β)과 margin 계열(α)을 도입해 학습 안정성과 수렴성을 개선한다.

- **Empirical Impact**: KVC-onGoing(185,000+ 사용자의 desktop/mobile 시나리오)에서 EERLoss는 기존 SoTA 손실들보다 우수한 성능과 함께 더 빠른 수렴(학습 비용 절감)을 보였다는 ablation 결과가 제시된다. 특히 가장 까다로운 설정(G=1)에서 Global EER 7.74%를 달성했으며, Avg. per-user EER 관점에서도 G=10에서 3.64% 수준을 기록해 실제 배치 운영(사용자별 임계값)과의 정합성이 높다. 또한 LSIA 우승 아키텍처를 EERLoss로 재학습했을 때 원래 SoTA 대비 상대 EER이 최대 약 30%까지 감소해, EERLoss가 고변동 행동형 생체에 특히 적합한 task-aligned training objective임을 강하게 시사한다.



### Jolia: Concept-Level Vision-Language Alignment for 3D CT Contrastive Learning (https://arxiv.org/abs/2606.24570)
- **Prior Approaches**: 기존 3D 의료 비전-언어 파운데이션은 CLIP 스타일의 전역(글로벌) 정렬로 이미지만 하나의 벡터, 긴 방사선 보고서도 하나의 전역 토큰으로 압축해 구조적 국소성을 잃기 쉽다. 이를 보완하려는 세분화 접근들은 보통 segmentation 마스크나 공간 감독을 요구해 적용 가능한 해부학 범위가 제한된다. 또 일부 token-wise 정렬은 데이터에서 정렬을 “학습으로부터” 끌어내는 방식이라 보고서의 개념 분해 구조와 직접 맞물리지 못한다.

- **Core Contribution**: 이 논문은 Concept Queries(ConQuer)로, 전역 CLIP 정렬에 더해 개념(Concept) 단위의 병렬 대조 정렬을 학습하는 방식을 제안한다. 보고서를 LLM으로 해부학 개념별 섹션으로 쪼개고, 이미지 인코더 위에 개념별 cross-attention query를 두어 공간/마스크 감독 없이도 해당 개념에 대응하는 시각 특징을 모은다. 이렇게 얻은 개념 토큰을 기반으로 Jolia(3D CT 파운데이션)를 학습하며, 각 쿼리는 “해부학적 해석 가능성”을 내장한 attention map을 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 마스크 없이 어떤 장기/개념에 맞는 시각 영역을 query가 찾아가게 만들고, (2) 개념이 보고서에 항상 등장하지 않는 불완전성을 다루며, (3) 개념별 정렬을 전역 학습과 충돌 없이 결합하는 것이다. 저자들은 개념별로 독립적인 symmetric InfoNCE를 적용하되, 배치에서 텍스트 섹션이 비어 있는 개념은 presence mask로 제외해 개념 간 혼선을 줄였다. 또한 전역 [CLS]용 CLIP 손실과 개념 손실을 함께 학습해 end-to-end로 query가 해당 개념을 구분하도록 유도했다.

- **Empirical Impact**: ConQuer로 학습한 Jolia는 findings 분류(선형 프로빙·zero-shot), 방사선 보고서 생성, 그리고 cross-center transfer에서 CLIP 기반 기준선 및 다수의 경쟁 모델을 일관되게 능가한다. 특히 segmentation 기반 fVLM(마스크 없이 평가) 대비 큰 AUROC 향상을 보이며, seed 간 분산도 작아 재현성이 개선된 것으로 보고한다. CT-RATE·Merlin-Abd-CT·INSPECT 등 여러 공개 벤치마크에서 새로운 SOTA를 달성했으며, 가중치(weights)는 승인 시 공개할 예정이다.



### Multilevel Stochastic Plug-and-Play for Sparse-View CT Reconstruction (https://arxiv.org/abs/2606.24567)
Comments:
          12 pages, 6 figures, 3 tables

- **Prior Approaches**: Sparse-view computed tomography(SVCT)는 투영(view) 수를 줄여 방사선 노출과 촬영 시간을 줄이지만, 재구성 문제가 심하게 ill-posed가 되어 스트릭(streak) 아티팩트가 생기기 쉽다. 이를 줄이기 위해 Plug-and-Play(PnP) 계열은 데이터 일관성(data fidelity)과 학습된 이미지 prior를 결합하지만, 반복 수렴을 위해 시간이 많이 걸리는 문제가 있다. 또한 stochastic PnP는 denoiser 입력 분포를 재노이징(re-noising)으로 맞춰 강건성을 높이지만, 역시 많은 반복이 필요해 실용 효율이 떨어진다.

- **Core Contribution**: 본 논문은 SVCT용 multilevel(ML) stochastic PnP를 제안해 stochastic PnP의 재구성 속도를 크게 개선한다. 핵심 아이디어는 levels 간 prior coherence를 그대로 강제하면, fine-level prior gradient를 여러 번의 denoiser 평가로 정확히 추정해야 해서 계산 비용이 폭증한다는 관찰을 바탕으로 한다. 이를 해결하기 위해 multiresolution analysis(MRA) 근사 공간에서 multilevel 단계를 수행하는 ML-SPnP(Multilevel Stochastic Plug-and-Play)를 제안한다.

- **Technical Challenges**: 어려움은 levels 간 prior-coherence correction을 구현하는 과정에서, fine-level stochastic prior gradient를 비용 들여 추정해야 한다는 점이다. 저자들은 wavelet 분해 구조가 기대값 관점에서 prior-coherence correction이 소거(vanish)되도록 만들어, coarse-level 보정을 위해 비싼 fine-level stochastic gradient 추정을 피할 수 있음을 보인다. 그 결과 denoiser 함수 호출을 줄이면서도 multilevel 효과를 유지하는 구성을 달성한다.

- **Empirical Impact**: SVCT 재구성 실험에서 ML-SPnP는 기존 state-of-the-art 방법과 견줄 만한 재구성 품질을 보이면서도 runtime을 유의미하게 감소시켰다. 즉, stochastic PnP의 강건성을 유지하면서도 반복 비용 부담을 완화해 실제 적용 가능성을 높였다는 의미가 있다. SVCT뿐 아니라 PnP의 multilevel 가속을 검토하는 연구 흐름에도 참고가 될 만한 실증 결과로 평가된다.



### PatternGSL: A Structured Specification Language for Template-Free and Simulation-Ready 3D Garments (https://arxiv.org/abs/2606.24564)
Comments:
          11 pages, 6 figures

- **Prior Approaches**: 기존 학습 기반 방법은 neural implicit 같은 형태로 옷의 기하를 재구성하지만, 수선(봉제) 구조가 흐려져 날카로운 경계와 seam이 과도하게 매끈해지는 경향이 있어 cloth simulation에 바로 쓰기 어렵습니다. 반대로 프로그램 기반(예: GarmentCode)은 simulation-ready하지만 미리 정의된 구성 템플릿/컴포넌트에 의존해 이미지에서 임의 토폴로지를 추론하는 데 한계가 있습니다.

- **Core Contribution**: PatternGSL은 템플릿 의존을 없애면서도 패턴 모델의 물리적 엄밀함을 유지하는, learnable한 structured garment representation(명시적 패널·모서리·스티치 위상)을 제안합니다. 단일 이미지에서 vision-language model이 PatternGSL을 바로 예측하고, optimization 없이도 rule 기반 deterministic validity handling을 통해 시뮬레이터 입력 패턴으로 디코딩합니다.

- **Technical Challenges**: 핵심 난제는 “기하 재구성”과 “봉제 구조 구성” 사이의 representation gap을 메우는 것으로, 패널 경계 곡선과 스티치 토폴로지를 안정적으로 생성·표준화된 출력공간으로 설계해야 했습니다. 이를 위해 PatternGSL은 계층형 JSON에서 패널을 정점/배치로, 곡선은 상대 파라미터(및 필요한 경우 샘플 포인트)로, 스티치는 패널-엣지 인덱스 쌍으로 명시해 VLM 학습 타깃을 canonical topology로 고정하고, 디코더 단계에서 결함을 가볍게 정리합니다.

- **Empirical Impact**: 새로 구축한 PatternGSLData(총 30만 샘플, complete sewing pattern annotations 포함)로 supervised VLM 학습을 수행한 결과, 2D 패턴 정확도와 스티치 복원에서 기존 baseline을 크게 앞섰고 stitched topology 정확도도 높게 나타났습니다. 특히 physics 기반 시뮬레이션 성공률이 99.2%로 보고되어, 생성된 패턴이 실제 draping에 “바로” 사용될 수 있음을 보여주며, 동일 디코딩 파이프라인을 통한 pattern-level editing 및 in-the-wild 일반화도 함께 입증했습니다.



### Quantum CT via Dynamic Interval Encoding and Prior-Balanced QUBO Reconstruction (https://arxiv.org/abs/2606.24561)
Comments:
          10 pages, 10 figures

- **Prior Approaches**: QUBO 기반 quantum CT는 재구성을 이진 변수의 이차 목적함수로 바꿔 양자 어닐링/하이브리드 솔버로 풀어낸다. 하지만 grayscale CT에서는 고정 bit-plane 인코딩이 정밀도(비트 깊이) 향상과 함께 QUBO 크기·결합 복잡도·계수 범위를 빠르게 키우는 반면, 저비트 인코딩은 양자화 오차를 크게 만든다. 기존 연구들은 인코딩 후보 상태 수를 줄이거나(이산 라벨/저정수), 또는 낮은 쿼드/근사 가정에 기대는 경우가 많아, grayscale에서의 “표현력 vs 이진 변수 예산” 균형 문제가 상대적으로 약했다.

- **Core Contribution**: 이 논문은 grayscale QUBO 재구성의 병목을 ‘binary-variable budget’으로 정식화하고, 이를 고정 전역 인코딩 대신 동적 구간 인코딩으로 돌파한다. 각 refinement round마다 현재 추정치 주변의 로컬 gray-level interval만 2-bit로 표현해, per-round 변수 수를 억제하면서도 라운드를 거치며 유효 표현력을 점진적으로 키운다. 또한 edge-preserving quadratic prior와 projection-domain data consistency를 결합할 때 항의 스케일을 퍼센타일 기반으로 정규화해 더 안정적인 QUBO를 구성한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 제한된 2-bit 로컬 표현이 gray-level을 충분히 커버하지 못할 때의 수렴 불안정, (2) 데이터 충실도 항과 prior 항의 계수 스케일 차이로 생기는 최적화 진동/왜곡이다. 저자들은 boundary-hit-guided update rule로 로컬 interval이 경계에 부딪히면 탐색 구간을 확장(초기 half-width로 재설정), 내부 해이면 수축해 국소 정제를 강화한다. 더불어 prior와 data 항을 ‘절대 비영(非零) 계수의 퍼센타일’로 목표 크기에 맞춰 균형화한 뒤 최종 QUBO를 만들고, D-Wave 하이브리드 BQM 솔버에서도 실행 가능하도록 이 구성을 유지한다.

- **Empirical Impact**: 실험은 sparse-view 및 limited-angle fan-beam CT에서 진행됐으며, 제안 방식은 analytic/iterative/variational/representation-based baseline 대비 구조와 gray-level 분포를 더 충실히 복원했다. 특히 Hybrid(Ours)는 TV-PDHG 대비 평균 PSNR을 33.55 dB에서 40.63 dB로, 평균 NRMSE를 0.0217에서 0.0097로 크게 개선했고, SA(Ours)와 결과가 유사해 성능 향상이 솔버 고유 요인보다 동적 인코딩·prior 균형화에 기인함을 시사한다. 또한 고정 전역 인코딩(예: 5-bit) 대비 동적 2-bit가 더 높은 재구성 품질을 내며, 해당 차이는 gray-level occupancy 분포에서 로컬/점진적 표현 전략이 더 효과적임을 뒷받침한다.



### Heterogeneous Knowledge Distillation via Geometry Decoupling and Momentum-Aware Gradient Regulation (https://arxiv.org/abs/2606.24557)
Comments:
          Preprint. Under review

- **Prior Approaches**: Heterogeneous Knowledge Distillation(HKD)는 Transformer-계열 교사를 CNN-계열 학생에 압축·전이하려 하지만, 기존 방법들은 주로 투영 레이어나 attention 모듈로 표현 공간을 억지로 맞추는 데 집중해 왔다. 예를 들어 OFA는 intermediate feature를 unified logit space로 사상하지만 feature magnitude 불일치와 학습 불안정의 근본 원인을 다루지 못한다. PAT는 추가 attention으로 정합성을 높이려 했지만, 무거운 모듈로 인한 비용 부담과 더불어 gradient conflict 문제를 충분히 제어하지 못한다.

- **Core Contribution**: 이 논문은 HKD의 학습 불안정이 (1) feature-level norm discrepancy(특징 크기 불일치)와 (2) optimization-level gradient conflict(기본 분류 목적과 증류 목적의 그라디언트 충돌)라는 두 요인이 강하게 결합된 결과라고 규명한다. 이를 바탕으로 SPOFA(Spatial Projector and Momentum-based Adaptive One-For-All)를 제안하며, feature와 gradient 양쪽에서 “이중 안정화”를 수행한다. 특히 LayerNorm 기반 decoupling projector로 magnitude와 direction을 분리해 의미 정렬에 집중시키고, MEMA로 conflict가 악화될 때만 증류 신호를 억제하도록 설계한다.

- **Technical Challenges**: 핵심 기술 난점은 서로 다른 inductive bias를 가진 모델 사이에서 features의 절대 스케일이 뒤틀리면 학생이 의미 방향 정렬 대신 크기 맞추기에 소모된다는 점이다. SPOFA는 auxiliary branch에만 LayerNorm을 삽입하는 구조적 decoupling로, 학습 중 student이 무의미한 magnitude 스케일링 경로를 덜 타게 만든다. 두 번째 난점은 minibatch 변동이 큰 환경에서 기존의 memoryless 가중치 조절이 그라디언트 충돌을 오히려 증폭한다는 점이며, SPOFA는 logit 공간에서 per-sample gradient conflict를 측정하고 EMA 기반 historical baseline과 punish-only 페널티로 안정적으로 dynamic weight를 조절한다.

- **Empirical Impact**: ImageNet과 CIFAR의 두 mainstream 벤치마크에서 SPOFA는 OFA 같은 강한 기준선을 대부분 능가하고 PAT 같은 계산 집약적 방법 대비도 state-of-the-art 정확도를 보인다. 또한 성능 향상과 함께 계산 오버헤드는 표준 베이스라인에 비해 최소 수준으로 유지된다고 보고한다. 즉, 구조적 feature 불일치와 그라디언트 충돌을 직접 겨냥한 가벼운 설계로 HKD의 “불안정” 문제를 실전형 학습 파이프라인에서 완화했다는 점에서 의미가 크다.



### Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning (https://arxiv.org/abs/2606.24548)
Comments:
          10 pages, 7 figures. Project page: this https URL

- **Prior Approaches**: 기존 T2I 평가는 문장-이미지 정합이나 상식/지식 기반 생성처럼 “연관” 수준의 성공을 중심으로 설계되는 경우가 많아, 진짜 인과적 반사실 추론인지 확인하기 어렵다. 반대로 반사실 벤치마크는 대체로 단순한 물체 조합이나 피상적 의미 변형에 치우쳐, 법칙 수준의 추론 능력을 분리해 측정하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 Counterfactual-World (CF-World)라는 반사실 인과 추론 벤치마크를 제안해, 기존 지식에 반하는 규칙 하에서 T2I 모델이 이미지를 생성하는 능력을 단계적으로 평가한다. 시나리오는 사실 생성(L1), 결과까지 명시한 명시적 반사실(L2), 결과를 숨기고 법칙으로부터 추론해야 하는 암시적 반사실(L3)로 구성된다.

- **Technical Challenges**: 핵심은 반사실 상황에서 “시각 품질”과 “규칙/인과 일관성”을 분리해 자동 채점하는 평가 파이프라인을 만드는 것이다. 이를 위해 Vision Language Model(VLM) 기반 CF-Eval을 구축하고 Prior Resistance Rate(PRR)와 Reasoning Retention Rate(RRR)로 기준선 대비 성능 저하를 수치화했으며, 사실 미달 시 반사실 점수를 0으로 두는 conditional thresholding으로 우연 성공을 차단한다.

- **Empirical Impact**: 실험 결과 대부분의 모델이 L1→L2, L2→L3로 갈수록 급격히 성능이 떨어져 반사실 인과 추론 능력이 약함을 보여준다. 진단 분석에서는 모델이 규칙과 속성을 decoupling하지 못하고, 데이터에서 자주 동반되는 시각-텍스트 공기열 패턴에 강하게 고정되어 commonsense prior로 되돌아간다는 병목이 확인되며, 학계에서 “T2I의 추론 주장”을 반사실 인과 관점에서 재검증해야 한다는 메시지를 남긴다.



### PointVG-R: Internalizing Geometric Reasoning in MLLMs for Precise Pointing Localization via Visual Chain of Though (https://arxiv.org/abs/2606.24539)
- **Prior Approaches**: 포인팅 기반 비주얼 그라운딩은 제스처가 가리키는 대상의 위치를 찾아야 하지만, 기존 방식은 이미지를 정적 특징으로 임베딩한 뒤 언어 연관 중심으로 추론하는 경우가 많다. 그 결과 이미지에 내재된 공간-기하 제약을 명시적으로 다루지 못해 복잡한 상호작용 상황에서 위치 하이라이트가 흔들리는 ‘cognitive fragility’ 문제가 나타난다. 또한 제스처를 단순 시각 프롬프트처럼 취급하는 접근은 정밀한 좌표 회귀에서 필요한 기하적 해석을 충분히 내재화하지 못한다.

- **Core Contribution**: PointVG-R은 멀티모달 LLM에 ‘Thinking with Images’ 관점을 도입해, 포인팅 제스처를 이미지 기반 기하 추론의 연쇄(Visual CoT)로 내부화하도록 설계한 reasoning-guided 모델이다. 인간이 포인팅을 해석할 때처럼, 손/키포인트/레이 기하를 단계적으로 구성한 뒤 그 결과로 목표 박스를 앵커링하도록 추론 파이프라인을 제공한다. 더불어 EgoPoint-CoT라는 시각 Chain-of-Thought 데이터셋으로 정답 경로를 학습 신호로 주고 SFT와 RL을 결합한다.

- **Technical Challenges**: RL을 좌표 정밀 과업에 적용할 때는 샘플링으로 인해 학습 신호의 품질이 집단(group)마다 달라져, 잡음이 큰 롤아웃이 BBox 좌표 학습을 불안정하게 만들 수 있다. 이를 해결하기 위해 Group Variance 기반 Adaptive Importance Weighting을 제안해, intra-group 일관성이 낮거나 분산이 큰 집단에 대해 보상 신호의 기여를 동적으로 조절한다. 학습은 (1) LoRA 기반 CoT-SFT로 포맷/기하 문법을 안정화하고 (2) GRPO 정렬 단계에서 도구 호출(draw_ray 등)과 기하 보상(손 박스, 레이 각도, 키포인트, IoU, 포맷 유효성)을 계층적으로 설계해 추론과 좌표를 함께 최적화한다.

- **Empirical Impact**: EgoPoint-CoT 벤치마크에서 PointVG-R은 mIoU 기준으로 기존 베이스라인 대비 15.86 포인트 높은 SOTA를 달성하며, SFT만으로는 보이지 않던 정밀한 기하 정렬 이득이 RL 정렬에서 확대됨을 보였다. 특히 Qwen2.5-VL(7B) 백본에서 mIoU가 0.6688에서 0.7570으로 크게 상승해, 그룹 분산 기반 재가중이 보상 잡음을 억제하며 추론 잠재력을 끌어냈다는 점을 입증한다. 중간 V-CoT 구성요소(손 앵커, 레이/키포인트 슈퍼비전)와 중요도 추정 통계(Group Variance)의 기여도에 대한 ablation은 ‘명시적 기하 추론’이 포인팅 그라운딩의 견고성을 좌우한다는 결론을 뒷받침한다.



### ForensicsTok: Forensics-Guided Tokenized Modeling for Image Tampering Localization (https://arxiv.org/abs/2606.24538)
Comments:
          16 pages, 4 figures, 8 tables

- **Prior Approaches**: 기존 MLLM 기반 IMDL(이미지 조작 탐지·국소화)은 LISA 계열처럼 MLLM의 숨은 표현을 외부 SAM-like 세그멘터/디코더와 결합해 마스크를 ‘붙여서’(stitched pipeline) 예측하는 방식이 많았다. 이 구조는 backpropagation에서 공간 신호가 병목으로 희석되고, 세그멘터의 의미적 priors에 성능이 제한되며, 마스크 토큰이 공간적으로 애매하게 표현될 수 있다는 한계가 있다. 또한 포렌식 지식을 후단에 단순 결합(post-backbone)하거나 LoRA 중심 fine-tuning으로만 주입해 다중 스케일 단서 결합이 비효율적이라는 문제도 제기됐다.

- **Core Contribution**: ForensicsTok은 IMDL을 ‘autoregressive sequence generation’으로 재구성해, MLLM이 공간적으로 grounded된 토큰 시퀀스를 직접 생성하고 이를 마스크로 변환하는 end-to-end 흐름을 제안한다. 특히 Token Splatting Decoder(TSD)가 codebook 기반 detokenization에 코드북 인지 label smoothing을 적용해, 이진 마스크 코드에 대한 one-hot의 경직성을 완화하고 정확한 마스크 예측을 가능하게 한다. 여기에 Hierarchical Expert Fusion(HEF)으로 다중 스케일 포렌식 전문가(SparseViT) 특징을 시각 인코더의 중간 단계에 주입해, 표준 MLLM이 부족한 포렌식 단서를 더 잘 활용하도록 설계했다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 마스크를 토큰으로 표현할 때 시각적으로 유사한 지역이 codebook의 이웃 인덱스로 매핑될 수 있는데, 이를 one-hot으로 학습하면 ‘all-or-nothing’ 형태의 불안정한 그래디언트가 생긴다는 점이다. TSD는 고정된 codebook에서 정답 코드와 의미적으로 가까운 이웃(Top-K)만을 찾아 온도 스케일 softmax로 확률 질량을 분산시키는 방식으로, 하드 디코딩은 유지하면서도 학습 안정성을 높인다. 또 다른 도전은 (2) 포렌식 전문 특징이 저수준 질감 단서부터 고수준 의미 불일치까지 성격이 달라, 후단에서 단순 결합하면 이질성 문제로 정렬이 깨진다는 점인데, HEF는 시각 백본 중간 레이어에서 attention 기반 게이팅으로 전문가의 다중 스케일 정보를 정렬·선택 주입한다.

- **Empirical Impact**: 6개 벤치마크(CASIA1, NIST, Coverage, Columbia, Glide, IMD) 교차 도메인 평가에서 ForensicsTok은 평균 IoU 0.68, F1 0.78로 기존 MLLM 기반 기준모델보다 크게 향상됐다. 특히 세그멘터를 끼우는 stitched 방식의 정보 병목·의미-공간 불일치를 제거한 토큰 생성 재구성이 성능 개선의 주요 원인으로 제시되며, 다른 포렌식 전문가들에 대해서는 약간의 우위 또는 근소한 개선, 그리고 잡음/교차 도메인 교란에 대한 더 강한 견고성이 관찰됐다. 구성요소별 ablation에서도 codebook 디코더(Tok) 기반 마스크 토큰화가 가장 큰 이득을 주고, HEF와 TSD가 각각 추가 상승을 만든다는 점이 확인되며, 중간 결합(intermediate fusion)이 post-backbone 결합보다 일관되게 유리함이 비교 실험으로 뒷받침됐다.



### VisCritic: Visual State Comparison as Process Reward for GUI Agents (https://arxiv.org/abs/2606.24525)
Comments:
          17 pages, 4 figures; ECCV 2026 submission; supplementary material uploaded as ancillary file

- **Prior Approaches**: GUI 에이전트는 스크린샷을 바탕으로 클릭·타이핑·스크롤 같은 동작을 수행하지만, 긴 과업에서 한 번의 오동작이 연쇄 실패로 이어지기 쉽다. 기존 process reward model(PRM)·에러 검출은 주로 텍스트 추론, 툴 호출, 체크리스트 등으로 “검증 신호”를 만들며 GUI의 실제 결과가 드러나는 시각적 상태 변화(픽셀/레이아웃 변화)와의 모달리티 간극이 존재한다.

- **Core Contribution**: VisCritic은 GUI 에이전트의 동작 성공 여부를 “사전/사후 스크린샷”을 비전 특징 공간에서 직접 비교해 시각적으로 검증하는 visual process reward 프레임워크를 제안한다. Siamese Visual Difference Encoder와 Action-Aware Critic Head를 통해 action success, task progress, error type을 함께 예측하며, 에이전트 아키텍처를 수정하지 않고 플러그앤플레이로 붙일 수 있다.

- **Technical Challenges**: 핵심 과제는 픽셀 차분만으로는 렌더링 잡음·애니메이션·사소한 레이아웃 이동에 취약하다는 점과, 텍스트 기반 검증을 대체할 만큼 “의미 있는 변화”를 안정적으로 포착하는 것이다. VisCritic은 학습된 semantic feature 공간에서 스크린샷 차이를 인코딩하고 change magnitude 기반 change region attention으로 변화가 큰 UI 영역에 집중시키며, 기존 궤적에서 약지도(w​​eakly supervised) 샘플을 자동 구성해 추가 인력 라벨 없이 critic 학습 데이터를 만든다.

- **Empirical Impact**: 다섯 개 벤치마크(웹·모바일·데스크톱)에서 VisCritic은 대부분의 상호작용 설정에서 다양한 base agent를 일관되게 개선하고 텍스트 기반 PRM/critic 대비 우위를 보였다. 특히 No-op(효과 없음)처럼 텍스트로 설명하기 어려운 오류 유형에서 시각적 비교가 강하게 작동했으며, change-region attention 맵을 통해 성공/실패 원인에 대한 진단 단서까지 제공한다. 



### What Do Flow-Based Inverse Solvers Approximate? A Posterior-Transport View (https://arxiv.org/abs/2606.24516)
- **Prior Approaches**: 훈련 없이(flow model 재학습 없이) pretrained flow-matching prior를 그대로 쓰는 인버스 문제 해결법들이 늘고 있다. 대표적으로 DPS/DAPS 계열과 FlowDPS, FLOWER, PnP-Flow는 확률-흐름 ODE를 적분하되 매 단계에서 likelihood 기반 correction으로 궤적을 측정에 가깝게 “유도”한다.
다만 이런 per-step correction이 실제로는 무엇을 근사하는지, 그리고 생성된 샘플이 진짜 posterior p(x|y)에서 얼마나 벗어나는지는 정량화되지 않았다. 지각 품질 지표는 그 차이를 숨길 수 있어, 복원 신뢰도와 불확실성 정량화에는 한계가 있다.

- **Core Contribution**: 이 논문은 흐름 기반 인버스 솔버를 “posterior transport(사후 수송)” 관점에서 다시 해석한다. 핵심은 deterministic(잡음 없는) flow prior에서는 Bayesian conditioning이 궤적의 드리프트 보정이 아니라, source 분포에 대한 재가중(reweighting)으로만 정확히 구현된다는 점이다.
즉, 원래 velocity field를 수정하지 않고 reweighted source를 그대로 pushforward하면 정확한 posterior 샘플이 나온다. 또한 trajectory-guidance 방식들은 이 정확한 reweighting을 대신하기 위해 필요한 “최소 kinetic energy” correction을 근사하려는 것으로 정리하며, FlowDPS/FLOWER/PnP-Flow의 차이를 그 근사의 차이(zeroth-order/gaussian/proximal)로 분해한다.

- **Technical Challenges**: 문제는 reweighted source p0^y를 직접 샘플링하는 것이 일반적으로 비싸고(고차원에서 effective sample size가 급감), 따라서 guidance 솔버는 correction 필드 u_t를 억지로 주입해야 한다는 점이다. 논문은 어떤 correction이든 posterior로 가려면 결국 canonical correction u*와의 간격이 작아야 한다는 안정성/편향 상계를 제시해, “무엇이 부족한지”를 수학적으로 연결한다.
구현 측면에서는 Flow Tweedie endpoint, isotropic-Gaussian projection, data-fidelity proximal+reprojection 같은 휴리스틱이 결국 u*의 고차 구조를 무시한다는 점을 지적한다. 그 결과 guidance의 강도를 어떻게 조절하더라도 모드 붕괴 및 posterior 편향이 쉽게 사라지지 않는 이유를 설명한다.

- **Empirical Impact**: 모델 오차를 제거한 2D 닫힌형(closed-form) posterior 실험에서 이론은 강하게 확인된다. source reweighting은 모든 지표에서 Monte-Carlo floor까지 posterior를 정확히 재현하지만, trajectory guidance는 guidance strength와 무관하게 200~800배 더 큰 오차를 보이며 posterior 모드를 붕괴시킨다.
또한 AFHQ·CelebA 복원과 out-of-distribution 조건에서도 분석이 유도한 “cheap velocity-correction solver”가 품질은 주요 in-domain baseline과 경쟁하면서도, 점추정형 source-space 최적화와 달리 다양한 posterior 샘플과 복원 오차와 상관된 불확실성을 제공한다. 즉, 단순히 그럴듯한 복원에 그치지 않고 신뢰 가능한 posterior 기반 복원/불확실성 추정으로 확장될 수 있음을 보여준다.



### GeoIMO: Geometry-Driven Independent Motion Classification for Event Cameras (https://arxiv.org/abs/2606.24499)
- **Prior Approaches**: 기존 자동차 event 데이터셋/학습은 RGB 프레임 파이프라인에서 얻은 appearance 기반 박스 라벨을 그대로 쓰는 경우가 많아, 정지물과 Independently Moving Objects(IMO)를 motion 관점에서 분리하기 어렵다. 이 때문에 event 카메라의 motion-sensitive 특성을 활용하는 방법들의 성능 상한이 라벨 모달리티 불일치로 제한된다.

- **Core Contribution**: 논문은 geometry-driven, annotation-free 방식으로 박스 단위 정지/이동을 분류하는 프레임워크를 제안한다. Focus of Expansion(FOE) 기반 전역 ego-motion을 event 스트림에서 추정한 뒤, 각 박스 내부 국소 motion이 전역 예측을 scale-invariant residual로 벗어나면 moving으로 판정한다.

- **Technical Challenges**: 핵심 난제는 event가 희소하고 yaw 회전 같은 회피 불가능한 카메라 운동이 전역 motion 추정을 흔든다는 점이다. 이를 위해 dense optical flow 없이 FOE에 yaw 보정 파라미터를 결합하고, 연속 temporal window에서 전역/국소 추정을 안정화하며, contrast maximization으로 정렬 품질이 낮은 추정은 down-weight하는 방식으로 강건성을 확보한다.

- **Empirical Impact**: MVSEC와 Prophesee 1 Megapixel Automotive Detection에서 diverse driving scenario 전반에 걸쳐 일관된 성능을 보였고, 특히 turn에서 yaw compensation이 결과를 개선했다. 또한 국소 motion은 radial 모델보다 단순 translational 모델이 종종 더 낫거나 비슷한 수준의 정확도/런타임 트레이드오프를 보여, 복잡도 대비 효율적인 선택지를 제시했다.



### VistaRef: Boosting Visual Spatial Orientation Awareness for Pointing-to-Object Detection (https://arxiv.org/abs/2606.24498)
- **Prior Approaches**: 비전-언어 grounding( VG )/referring expression comprehension(REC) 계열은 텍스트와 이미지 간 의미 대응을 박스·마스크 예측으로 연결해 왔지만, 텍스트만으로는 사용자의 신체 지시(데익틱 제스처)를 충분히 반영하기 어렵다는 한계가 있었다. 특히 pointing-to-object 같은 지시 과제에서는 Transformer의 global attention이 미세한 손가락 기하 관계를 놓쳐 손가락 포즈 변화에 대한 방향 민감도가 떨어지고, 원거리·복잡 장면에서 localization drift와 모호성이 커진다고 지적한다.

- **Core Contribution**: VistaRef는 pointing 제스처의 “암묵적 방향”을 “명시적 기하 광선(ray)”으로 변환해, 대상의 위치뿐 아니라 손이 어떻게 가리키는지를 모델이 직접 정렬하도록 설계한 프레임워크다. 이를 위해 Local Hand Entity Modeling(LHEM)으로 미세 손가락 편차를 강화하고, Geometric Ray Modeling(GRM)으로 손-타깃 간 기하 경로를 attention 기반 특징 집계에 반영한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 손의 미세 포즈를 신뢰도 있게 인코딩하고 (2) 광선 정렬에 필요한 방향 정보를 단순 attention 이상으로 강제해 end-to-end 학습 과정에서 기하 일관성을 유지하는 것이다. VistaRef는 LHEM의 hand-pose embeddings 및 GRM의 ray embedding·ray-aware interaction을 통해 방향 신호를 특징으로 “명시화”하고, Orientation-Consistent Alignment Loss(OCAL)로 hand 존재/키포인트/광선 일관성을 함께 감독하되 손재현이 없으면 기하 항을 비활성화하는 비대칭 조건부 슈퍼비전을 적용한다.

- **Empirical Impact**: EgoPoint-Ground에서 VistaRef는 기준선 대비 grounding accuracy에서 14포인트의 절대 향상을 보이며, 특히 복잡하거나 원거리 pointing에서 localization robustness와 orientation accuracy가 개선됨을 확인했다. 정성 분석에서도 손-타깃 간 기하 상관을 더 잘 포착해 기존 Transformer의 공간 인지 갭을 메운다는 점이 드러난다.



### RetiSEM: Generalising Causal Models for Fragmented Biomedical Data (https://arxiv.org/abs/2606.24488)
- **Prior Approaches**: 기존 인과 발견(예: PC, NOTEARS, LiNGAM)이나 일반적인 DAG 학습은 관측 잡음과 고차원 생의학 데이터에서 방향 추정이 불안정해지고, 생물학적으로 그럴듯하지 않은 구조를 만들 수 있다는 한계가 있다. 또 딥러닝 기반 망막 심혈관 위험 예측은 높은 예측 성능을 보이더라도 대체로 black-box이며, 상류 생리 신호가 망막 특징을 거쳐 결과로 이어지는 메커니즘 해석이 어렵다. 무엇보다 임상·분자·영상 변수가 참여자 단위에서 함께 관측되지 않는 fragmented multimodal 데이터에서는 인과 추론 가정 자체가 취약해진다.

- **Core Contribution**: RetiSEM은 유전/인구구조-분자-망막-혈관결과로 이어지는 생물학적 블록 순서를 반영한 domain-constrained SEM으로, fragmented 멀티모달에서 인과 그래프를 복원하고 매개 경로를 해석한다. 또한 forbidden-edge masking으로 허용되지 않는 역방향 간선을 배제해 탐색 공간을 줄이고, TE·NDE·NIE로 경로 수준의 직접/간접 효과를 분해해 망막 특징이 ‘지표’인지 ‘매개 신호’를 갖는지 검정한다. 결과적으로 망막을 단순 예측 변수에서 가설 검정 가능한 phenotype layer로 재정의한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 참여자 단위의 완전한 멀티모달 정렬이 없고 (2) 변수 수가 많으며 (3) 생의학적 비선형성이 존재하는 상황에서 그래프 복원이 안정적으로 이뤄져야 한다는 점이다. RetiSEM은 선형 SEM 근사로 계산 가능성과 TE/NDE/NIE 분해의 재현성을 확보하면서, DAG 탐색은 블록 순서 제약과 forbidden-edge mask로 acyclicity 및 역방향 간선 문제를 줄였다. 이후 mediation formulation을 통해 망막 블록을 매개(간접효과) 또는 반영(직접효과)하는 형태로 정량화한다.

- **Empirical Impact**: 합성 실험 10개 시나리오(차원·비선형성·인과 깊이·병렬 경로 구조 변화)에서 RetiSEM은 unconstrained baseline 대비 SHD가 낮고 causal accuracy가 높게 나타나 구조 복원과 방향성 회복이 더 일관적임을 보였다. real-world 설정(NHANES 임상 변수 + 외부 유래 망막 표현)에서는 망막 관련 NIE가 대체로 TE/NDE보다 작지만 0이 아닌 간접효과가 관측되어, 망막은 주로 downstream biomarker-like 지표로 작동하되 일부 경로에서는 mediator-like 통계 신호를 제공한다는 결론을 뒷받침한다. 전체적으로 제한 자원 멀티모달 환경에서 해석 가능한 구조적 인과 가설 테스트를 가능하게 한다는 점에서 실무적 의미가 있다.



### Advancing WordArt-Oriented Scene Text Recognition: Datasets and Methods (https://arxiv.org/abs/2606.24484)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 Scene Text Recognition(STR) 연구는 CTC/PD/AR 계열로 발전했지만, 대부분 고정된 입력 템플릿이나 가로 정렬 텍스트를 전제로 한다. Artistic text(WordArt)는 글꼴·질감·레이아웃이 크게 커스터마이즈돼 읽기 순서와 형태가 자주 깨지며, 이 때문에 일반 STR 데이터셋/모델 성능이 WordArt에서 크게 하락한다. 또한 WATER 같은 워드아트용 데이터는 수집·라벨링 비용이 높아 규모와 다양성이 부족하다는 한계가 누적된다.

- **Core Contribution**: 이 논문은 WordArt-oriented scene TExt Recognition(WATER)을 데이터와 모델 관점에서 동시에 확장한다. 데이터 측면에서 tool-rendering(SynthWordArt)과 VLM 프롬프트 마이닝+생성모델(Z-Image)을 결합한 2M 합성 데이터 WATER-S를 구축해 폰트·레이아웃·시각적 잡음을 현실에 가깝게 확장한다. 모델 측면에서는 WATERec을 제안해 임의 형태 입력을 처리하는 시각 인코더와 복잡한 레이아웃을 단계별로 모델링하는 autoregressive 디코더로, 고정 템플릿 기반 병목을 구조적으로 완화한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) WordArt의 극단적 종횡비와 비정형 레이아웃을 왜곡 없이 인코딩하고, (2) 읽기 순서가 애매한 경우에도 문자를 정확한 순서로 생성하며, (3) 합성 데이터가 스타일 다양성과 라벨 정확성을 동시에 만족해야 한다는 점이다. WATERec은 입력 크기를 무리하게 템플릿화하지 않고 패치 토큰화를 하되, 절대 위치 임베딩의 민감도를 줄이기 위해 RoPE 기반 RoPE attention을 채택해 가변 길이/형상에 일반화하도록 설계한다. 데이터는 WATER-T로는 artistic font와 레이아웃 제어를 통해 텍스트 정확도를 확보하고, WATER-Z로는 Qwen3-VL이 만든 세밀한 프롬프트로 배경 질감·구도·전역 스타일의 현실성을 높여 두 경로를 보완하도록 구성했다.

- **Empirical Impact**: 실험에서 WATERec은 WordArt-Bench에서 합성 데이터 WATER-S(특히 WATER-T+WATER-Z)를 함께 학습할 때 90.40% 정확도를 달성하며, 일반 VLM/OCR-specialized 모델 대비 큰 격차를 보인다. 또한 WATER-R를 해시 기반으로 중복 제거해 라벨 누출을 방지한 뒤, 합성 데이터 추가가 예술 subset뿐 아니라 일반적인 STR 벤치마크에서도 일관된 향상을 만든다고 보고한다. 종합적으로 WordArt에서 90%를 최초로 넘긴 강력한 기준선과 대규모 합성 데이터 설계를 동시에 제공해, 워드아트 인식 연구의 데이터·모델 개발 방향을 한 단계 끌어올렸다는 점에서 의미가 크다.



### MambaRaw: Selective State Space Modeling for Efficient 4K Raw Image Reconstruction (https://arxiv.org/abs/2606.24479)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: JPEG 미리보기(sRGB)를 메타데이터 비트스트림과 함께 전송한 뒤 raw를 복원하는 메타데이터 기반 복원 방식들이 주목받았다. 다만 기존 컨텍스트 모델은 고해상도(예: 4K)에서 attention의 제곱 복잡도나 수용영역 한계로 인해 계산량 병목이 발생했고, 무거운 모델을 모든 위치에 동일하게 적용해 낭비가 컸다. 또한 raw 신호의 카메라별·공간별 에너지 분포가 치우치는데, 이를 정교하게 반영하는 방식이 제한적이었다.

- **Core Contribution**: 이 논문은 JPEG-guided 메타데이터 기반 raw 복원을 효율적으로 수행하는 MambaRaw를 제안한다. 핵심은 Spatial-Energy Coupled Context Modeling으로, long-range 문맥 추정은 SSM(State Space Models)으로 하되 계산은 에너지 정보로 선택적으로 수행하고, EAR(Energy-Aware Refinement)로 raw의 long-tail 에너지 분포에 맞춰 엔트로피 관련 특징을 보정하는 것이다. 이를 통해 높은 품질과 낮은 지연을 동시에 노리는 설계가 제시된다.

- **Technical Challenges**: 고해상도 feature map에서 global 컨텍스트 모델링을 유지하면서도 계산을 억제하는 것이 첫 번째 난제이며, 기존 attention/콘볼루션 기반 접근은 이 균형이 어렵다. MambaRaw는 4방향 cross-scan 기반 SSM을 쓰되 TileMambaBlock에서 L2 energy로 ‘정보 밀집 타일’만 골라 selective scanning을 수행하고, EAR은 identity-initialized residual로 공간 정밀도를 보존한 채 에너지 기반 게이팅으로 특징을 정교화한다. 또한 JPEG 미리보기는 scale별로 리사이즈해 채널 결합 형태로 여러 스케일에 주입하여 cross-attention 없이 JPEG 조건을 반영한다.

- **Empirical Impact**: Sony/Olympus/Samsung 3개 카메라 데이터셋에서 강한 메타데이터 기반 베이스라인 대비 일관된 rate–distortion 개선이 보고됐다. 특히 메타데이터 비트레이트가 낮을 때 PSNR이 1.2–1.4 dB 향상되고 end-to-end 코딩 지연도 약 9% 감소해, 대역 제한 환경에서 유리한 성능/효율 균형을 보였다. 4K 완해상도 평가에서도 Beyond-R2LCM 대비 PSNR이 1.37 dB 개선되면서 FLOPs는 약 56% 줄고 지연은 9% 낮아졌으며, 메모리 사용도 더 안정적으로 유지된다고 한다.



### video-SALMONN-R$^3$: Learning to ReWatch, ReAsk, and ReAnswer for Efficient Video Understanding (https://arxiv.org/abs/2606.24477)
- **Prior Approaches**: 기존 비디오 LLM은 계산·메모리 제약 때문에 프레임을 성긴 간격으로 샘플링하거나 해상도를 공격적으로 낮춰 처리해, QA에서 필요한 미세한 시공간 단서를 놓치기 쉽습니다. 이를 보완하려고 나온 re-watch(재시청) 방식은 다중 에이전트(여러 모델 조합)나 단일 모델의 멀티패스 전략으로 나뉘는데, 전자는 연산 오버헤드가 크고 후자는 CoT(chain-of-thought) 콜드스타트용 대규모 주석 의존도가 높습니다. 또한 CoT 기반 SFT는 사전학습된 비디오 이해 능력을 훼손하거나 편향을 유발할 수 있다는 문제가 제기됩니다.

- **Core Contribution**: 이 논문은 CoT 콜드스타트 주석 없이도 강화학습(RL)만으로 end-to-end re-watch 능력을 획득하는 video-SALMONN-R3를 제안합니다. 특히 모델이 첫 시청에서 답을 먼저 내고, 재시청 후 답을 정교화하도록 하는 re-answer 전략과, 재시청 시 질문을 다시 주입해(질문-프레임 직접 상호작용) 정합성을 높이는 re-ask 메커니즘을 함께 설계했습니다. 그 결과, localization(어디를 볼지)과 reasoning(무엇을 답할지) 분리가 필요한 상황에서도 기존 성능을 유지하면서 재시청 이점을 안정적으로 끌어올리려 합니다.

- **Technical Challenges**: 핵심 기술적 난관은 re-watch가 유도하는 reasoning-first 행동과 사전학습 비디오 LLM의 answer-first 성향 사이의 분리(불일치)로, 그대로 학습하면 성능이 흔들릴 수 있다는 점입니다. 저자들은 re-answer로 첫 시청의 답을 기반앵커로 고정한 채, 두 번째 시청에서만 수정이 일어나도록 학습 목표와 출력 구조를 맞췄고, 또한 causal attention 제약으로 재시청 프레임에 질문이 직접 못 닿는 문제는 re-ask로 해결합니다. 학습은 오디오 정렬→오디오-비디오 캡션 SFT→마지막으로 DAPO 계열 on-policy RL로 end-to-end re-watch 궤적을 주입하며, 큰 CoT 주석 없이도 ‘정답·형식·수정’에 기반한 규칙 보상으로 정책을 최적화합니다.

- **Empirical Impact**: 실험에서 video-SALMONN-R3는 6개 벤치마크(단시간부터 장시간까지)에서 베이스 모델과 QA-SFT 기준선을 일관되게 능가하며, prior re-watch 접근들보다 더 낮은 연산 비용으로 성능을 끌어올렸다고 보고합니다. 특히 VideoMME·LVOmniBench 같은 장문 비디오에서 개선폭이 커져, 긴 영상에서 균일 샘플링의 한계를 ‘필요 구간만 집중 재시청’으로 상쇄한다는 관찰과 연결됩니다. 아울러 ablation 결과는 두 번 답하기만으로는 이득이 제한적이며, localization에 맞춘 재시청과 re-ask·revise 보상이 함께 작동해야 효과가 재현된다는 점을 강조합니다.



### Boosting Text-Driven Video Segmentation via Geometry-Aware Distillation (https://arxiv.org/abs/2606.24464)
Comments:
          Accepted by ECCV2026

- **Prior Approaches**: Text-driven Referring Video Object Segmentation(RVOS)은 언어로 지시된 대상을 영상에서 분할하는 문제다. 기존 방법은 주로 RefCOCO/RefCOCOg 같은 2D 이미지 데이터로 전처리하며, 단순 분할 손실로 인해 프레임 간 3D 기하 일관성을 놓쳐 시점 변화나 카메라 운동에 취약하다. 또한 영상 파인튜닝 단계에서 메모리/어텐션으로 관계를 암묵적으로 학습하더라도, 3D 구조 지식이 부족해 추적 안정성과 공간 이해가 흔들린다.

- **Core Contribution**: 이 논문은 Geometry-enhanced Language-guided Video segmentation(GeoLaV)이라는 2단계 프레임워크를 제안해 3D 기하 priors를 명시적으로 통합한다. 1단계에서는 단안(monocular) 기하 사전학습으로 single image에서 기하 일관된 novel-view 시퀀스를 합성해, 프레임 간 구조 정합성을 먼저 학습한다. 2단계에서는 3D geometry-aware distillation으로 일반 3D 사전모델로부터 구조 지식을 전이해, language grounding과 시공간 일관성을 동시에 강화한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 2D 기반 전처리가 제공하는 뷰/프레임 기하 변동의 한계를 줄이고 (2) 합성 데이터에서 3D 구조를 왜곡하지 않으면서 (3) 이를 실제 RVOS 학습에 효과적으로 주입하는 것이다. GeoLaV는 π3나 VGGT 같은 단안 3D 추정 모델로 깊이/점맵을 얻고, 연속 카메라 궤적 기반으로 소수 프레임의 기하 일관된 novel-view를 생성해 MGP를 수행한다. 이후 SAM2의 메모리 표현과 VFM(DINOv3)·3D-aware encoder(π3)를 맞추기 위해 학습 시에만 쓰는 projection head와 코사인 유사도 기반 distillation 손실을 도입해, 파인튜닝 중에도 기하 일관성이 유지되도록 만든다.

- **Empirical Impact**: 실험에서 GeoLaV는 Ref-Youtube-VOS, Ref-DAVIS17, MeViS 등 주요 벤치마크 전반에서 SOTA 성능을 달성하며, 특히 비-Large VLM 계열 대비도 경쟁력 있는 수치를 보인다. 더 중요한 포인트로, 이미지 분할 데이터(RefCOCO/+/g)만으로 전처리한 뒤 비디오 파인튜닝 없이도 zero-shot 일반화가 유의미하게 나타난다. 이는 3D 기하를 합성 기반 사전학습과 distillation으로 주입하면, 대규모 비디오 주석 없이도 RVOS의 시공간 안정성과 정확도를 끌어올릴 수 있음을 실증한다.



### Lite Any Stereo V2: Faster and Stronger Efficient Zero-Shot Stereo Matching (https://arxiv.org/abs/2606.24457)
- **Prior Approaches**: 기존 스테레오 매칭은 대체로 정확도 중심으로 발전해 왔지만, 대형 모델·고비용 연산·추가 프라이어 모듈(예: foundation-model 기반 단안 depth priors)에 의존하는 경우가 많아 엣지/저전력 배포가 어렵다는 한계가 제기돼 왔다. 효율 지향 모델은 추론은 빠르지만 보통 strong zero-shot generalization은 약하다고 여겨져, 오프더셸프보다는 도메인별 fine-tuning이 필요한 편이다. 또한 pseudo supervision을 쓰는 방식은 좌-우 대응이 완전한 스테레오 기하에서 오지 못해 잡음과 디테일 손실 문제가 남는다.

- **Core Contribution**: 이 논문은 Lite Any Stereo V2(LAS2)로, 경량 스테레오도 zero-shot으로 강한 성능을 낼 수 있다는 가정을 실험적으로 뒤집는다. 핵심 기여는 (1) 실제 지연(latency)을 목표로 한 2D-only cost aggregation 설계와 (2) 합성-자기증류-실데이터 지식증류로 이어지는 3단계 학습 전략이다. 여기에 실데이터 pseudo label의 신뢰도를 높이기 위한 filtering과 error-clamping을 더해 synthetic-to-real 전이를 매끄럽게 만든다.

- **Technical Challenges**: 기여의 첫 난점은 MACs 같은 이론 지표만으로는 실제 GPU/엣지 지연을 잘 반영하기 어렵다는 점이다. LAS2는 3D cost aggregation을 제거하고 2D-only U-Net 스타일 집계를 적용하되, 다양한 해상도/지연 조건에서 작동하도록 아키텍처를 세밀하게 재조정했으며, feed-forward(LAS2-S/M/L)와 iterative(LAS2-H) 변형으로 효율-정확도 스펙트럼을 분리했다. 두 번째 난점은 실데이터 pseudo supervision이 구조적으로 틀릴 수 있고 초기에는 큰 오차가 학습을 흔들 수 있다는 점인데, 좌-우 일관성/edge-aware/sky 마스킹으로 잡음을 줄이고, 고오차 픽셀의 손실을 truncation하는 error-clamped loss로 최적화 안정성을 확보한다.

- **Empirical Impact**: 실험에서 LAS2는 효율 스테레오 계열 중 state-of-the-art 정확도를 유지하면서도 지연은 크게 낮춘 것으로 보고된다. 특히 LAS2-H는 iterative 방식 Fast-FoundationStereo보다 전반 성능이 더 좋으면서 H200에서 1.8x, Orin에서 2.7x 더 빠르다고 제시한다. 또한 LAS2-M은 여러 real-world 벤치마크에서 기존 feed-forward 대비 오류를 줄이면서 H200/Orin에서 각각 1.6x/1.9x 빠른 결과를 보여, 경량 모델의 zero-shot 일반화가 실제 배포 효율과 양립 가능함을 강조한다.



### SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking (https://arxiv.org/abs/2606.24449)
Comments:
          Accepted for publication at the European Conference on Computer Vision (ECCV 2026)

- **Prior Approaches**: 메모리 기반 영상 물체 추적은 과거 특징을 메모리 뱅크로 유지해 가림, 모션 블러, 외형 변화에 대응해 왔다. 다만 SAM2 계열은 프레임 단위 마스크를 confidence(예: IoU 예측치 등)만으로 선택해 메모리에 쓰는 구조라, 가림·급격한 이동·주의 대상(distractor) 상황에서 고신뢰지만 시간적으로 부정합한 마스크가 저장되며 drift가 누적될 수 있다. 기존 개선(SAMURAI의 모션 필터링, DAM4SAM의 distractor 체크 등)도 결국 같은 confidence 기반 write 의존성을 크게 벗어나지 못했다.

- **Core Contribution**: 이 논문은 SAM2 기반 추적기의 핵심 약점인 “confidence-only mask selection”을 겨냥해, 훈련 없이(inference-time) 메모리 write 직전에 short-horizon temporal consistency를 검증하는 refine-before-write 모듈 SENTRY를 제안한다. SENTRY는 프레임당 다양한 분할 가설을 모으고, 각 가설을 짧은 구간으로 백트래킹해 후보가 최근 궤적과 얼마나 일관적인지(이웃-aware, cycle-consistent) 확인한 뒤에야 메모리에 반영한다. 즉 백본/메모리 구조는 건드리지 않고, 메모리 업데이트 인터페이스의 “선택 기준”만 consistency 기반으로 교체한다.

- **Technical Challenges**: 가장 큰 기술적 난점은 후보 마스크가 많을수록 잘못된 가설을 메모리에 쓰기 쉬운데, 이를 재학습 없이도 신뢰도 있게 걸러내야 한다는 점이다. SENTRY는 (1) decoder의 마스크 가설과 AMG의 target-agnostic proposal을 함께 수집하고 Soft-NMS로 중복을 줄이며, (2) 각 후보를 τ=10 길이의 윈도우에서 백트래킹 트랙릿으로 만들고, (3) 최근 선택된 타깃 궤적과 이웃(남겨진 후보들의 forward 트랙릿)을 함께 고려하는 neighbor-aware cycle-consistent matching으로 후보 간 1:1 할당(Hungarian)을 수행해 temporally/geometrically 일관된 선택을 유도한다. 또한 심한 가림/실패 시에는 constant-acceleration Kalman 기반 motion-only fallback을 넣어 일관성 검증이 약해질 때의 붕괴를 완화한다.

- **Empirical Impact**: 논문은 다섯 개 오픈소스 SAM2 기반 트래커(SAM2, SAMURAI, DAM4SAM, SAMITE, HiM2SAM)에 SENTRY를 플러그인으로 통합하고, 9개 벤치마크와 여러 스케일에서 통합(all-scale) 재평가를 수행해 공정한 베이스라인을 제시한다. 그 결과 SENTRY는 5개 베이스라인에 일관된 개선을 제공하며, LaSOT·LaSOT_ext·GOT-10k·VOT20·VOT22·DiDi에서 zero-shot SOTA를 새로 기록했다(예: LaSOT에서 SENTRY-S2가 SAM2 대비 성능 지표 전반에서 우위). 성능/비용 측면에서도 A100 기준 실시간성을 유지하며(32.8 FPS 수준, VRAM 추가 약 0.4–0.6GB) “write-time에서 temporal validity를 강제하면 memory-augmented tracking이 안정화된다”는 메시지를 뒷받침한다.



### P-MTP: Efficient Document Parsing via Multi-Token Prediction with Progressive Depth Scaling (https://arxiv.org/abs/2606.24447)
- **Prior Approaches**: 기존 비전-언어 모델(Vision-Language Models, VLMs) 기반 문서 파싱은 end-to-end로 구조화 텍스트를 생성하지만, 토큰 단위 순차 디코딩 특성 때문에 지연(latency)이 커졌다. 이를 줄이기 위해 Multi-Token Prediction(MTP)은 유망하지만, look-ahead depth를 깊게 늘릴 때 distal supervision이 만드는 최적화 불안정성과 gradient noise가 문제로 지적돼 왔다.

- **Core Contribution**: 이 논문은 문서 파싱에 맞춰 Progressive Multi-Token Prediction(P-MTP)을 제안하며, look-ahead depth 확장을 학습-추론 전 과정에서 안정적으로 달성하는 프레임워크를 제공한다. 핵심은 Progressive Curriculum Loss로 예측 경로의 신뢰도에 따라 손실 가중치를 동적으로 조절하고, Confidence-Gated Dynamic Drafting으로 추론 시 speculative draft 길이를 실시간 불확실도에 맞춰 조절하는 것이다.

- **Technical Challenges**: MTP에서 깊은 look-ahead를 학습할 때는 먼 미래 예측이 앞선 예측 실패로 인해 쉽게 흔들리며, 정적/직관적 loss 재가중은 이런 trajectory divergence를 반영하지 못해 학습이 깨질 수 있다. P-MTP는 누적 경로 신뢰도(Sequential Path Constraint)와 목표 토큰의 역기억 정합성(Retrospective Target Constraint)을 결합해 원거리 손실을 자동으로 억제/완화하고, 추론 단계에서는 누적 joint probability 기반 confidence threshold로 신뢰 낮은 경로의 불필요한 계산을 미리 차단한다.

- **Empirical Impact**: 실험은 UniMERNet, PubTabNet, OmniDocBench 등 여러 벤치마크와 서로 다른 백본에서 진행됐고, 최대 5× 속도 향상과 정확도 손실이 거의 없음을 보고한다. 또한 PubTabNet에서 progressive depth 확장(K=9)까지도 TEDS 점수를 기준선 수준으로 유지하면서 속도 향상이 지속되었고, vLLM 기반 처리에서도 배치 크기별 처리량 개선이 확인돼 실사용 관점의 의미가 크다.



### S1-Omni-Image: A Unified Model for Scientific Image Understanding, Generation, and Editing (https://arxiv.org/abs/2606.24441)
Comments:
          32 pages, 15 figures

- **Prior Approaches**: 기존 multimodal LLM은 시각 정보를 입력에서 읽어 텍스트 답변을 내는 데 강점이 있지만, 고품질 과학 이미지 생성·편집을 직접 수행하기는 어렵다. 한편 일반 이미지 생성/편집 모델은 자연 이미지 중심 학습으로 인해 과학적 구조 오류, 불안정한 텍스트 렌더링, 다중 편집 시 의미 드리프트 같은 문제가 자주 발생했다. 또한 과학 일러스트 생성은 에이전트·툴체인에 의존하거나(예: TikZ 기반), 의료 세그멘테이션/번역/초해상도는 별도 전용 모델로 분리되어 통합된 open-weight 모델이 부족했다.

- **Core Contribution**: S1-Omni-Image는 scientific image understanding, generation, editing을 하나의 unified 프레임워크로 묶은 open-weight 멀티모달 모델이다. 핵심 아이디어는 사용자 프롬프트를 곧바로 이미지로 매핑하지 않고, S1-VL-32B(Scientific multimodal reasoning backbone)가 먼저 task-oriented reasoning trace를 만들고 이를 think-before-generate 방식으로 확산(생성/편집) 모듈에 조건 주입하는 것이다. 또한 <image_gen>/<image_edit> task special token을 라우터로 사용해 생성과 편집을 같은 reasoning-답변 접두 구조에서 분기한다.

- **Technical Challenges**: 과학 이미지는 시각적 그럴듯함뿐 아니라 구조적 관계, 과학 의미의 일관성, 도메인 지식, 텍스트 가독성, 그리고 멀티턴 편집의 통제성을 동시에 만족해야 한다. 이 논문은 LLM 표현 공간과 MMDiT 확산 모듈의 conditioning 공간이 불일치하는 문제를 reasoning-to-diffusion alignment layer로 해결해, 과학 추론에 기반한 hidden states를 MMDiT가 받는 조건 형태로 정렬한다. 학습은 3단계로 진행되며, Stage I에서 S1-VL-32B의 과학 추론/의도 이해를 먼저 강화하고, Stage II에서 정렬층을 학습해 조건공간 격차를 메운 뒤, Stage III에서 생성·편집 샘플로 확산 supervision을 통해 통합 성능을 끌어올린다.

- **Empirical Impact**: 논문은 SciGenEdit(약 314K 샘플)과 SciGenEdit-10K를 구성·공개하고, 해당 데이터로 학습된 S1-Omni-Image가 과학 이미지 생성·편집 성능을 유의미하게 개선하면서 S1-VL-32B의 과학 이미지 이해 능력도 유지함을 보인다. GenExam과 TechImage-Bench에서 오픈소스 모델을 상회하며, 편집 벤치마크 MSD, cigRockSEM, SynthRAD2025, IXI에서 state-of-the-art 결과를 달성한다. 전반적으로 scientific image understanding 평가에서도 안정적인 성능을 유지해, 과학 도메인 중심 unified 모델에 대한 실증적 기반을 제공한다.



### MedPCFM: Improving Medical Point Cloud Completion by Integrating Point Transformers and Flow Matching (https://arxiv.org/abs/2606.24433)
Comments:
          25 pages, 9 figures

- **Prior Approaches**: 의료 point cloud completion은 결측 부위가 크고 얇은 구조가 많아 “가능한 완성”이 여러 개일 수 있어 어렵다. 기존에는 분할 기반의 voxel 중심 파이프라인(예: 3D U-Net, V-Net)이 주로 쓰였고, 최근에는 PCDiff처럼 diffusion을 의료 결측 보완에 적용한 시도가 늘었지만 sampling 비용이 높고 속도-정확도 절충이 부담이었다. 또한 deterministic 모델(encoder-decoder)이나 diffusion의 비교가 충분히 매칭되지 않아, 생성형 관점의 효율성과 일반화 특성이 명확히 정리되지 않았다.

- **Core Contribution**: 이 논문은 PTv3를 기반으로 한 flow matching 연속시간 생성모델 PCFM을 제안해 의료 point cloud completion을 다룬다. PCFM은 관측된 결손 입력 X를 조건으로, 단순 분포에서 출발해 연속적 transport(ODE)를 통해 완성 분포로 이동시키는 방식으로 multi-modality를 생성형으로 포착한다. 또한 PTv3를 deterministic encoder-decoder 및 diffusion(PCDiff) 비교군에 동일·유사하게 적용해 공정한 벤치마크 구조를 마련하고, SkullFix/SkullBreak에 더해 Mandibular Defect까지 확장 검증한다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) point set의 순열 불변성과 고해상도 처리를 유지하면서 (2) diffusion 대비 훨씬 적은 ODE 적분 단계로도 충분한 품질을 내는 flow matching 설계다. 이를 위해 PTv3의 space-filling curve 직렬화, Serialized Attention/풀링-언풀링으로 점 집합 표현을 안정화하고, affine path(예: Conditional-OT 계열)와 scheduler 선택, 선택적 contrastive regularizer(배치 내 음쌍 permutation)를 조합해 학습 신호를 정교하게 만든다. 더 나아가 Heun 방법 기반의 ODE 적분으로 sampling-step 예산을 통제하고, diffusion과 동일한 운영 관점에서 step budget trade-off를 체계적으로 비교한다.

- **Empirical Impact**: 실험은 SkullFix/SkullBreak 및 Mandibular Defect 3개 데이터셋에서 수행되며, PCFM(PTv3)이 generative 성능에서 diffusion·deterministic 기준을 전반적으로 상회하거나 경쟁력 있게 나타났다. 특히 CD/DSC/BDSC/HD95 등 point·voxel 계열 지표에서 PTv3 기반 PCFM이 최상 운영점에서 diffusion 대비 훨씬 적은 steps로 품질을 유지하는 경향을 보였다. 속도 관점에서는 최적 설정에서 PVCNN 대비 PCFM이 최대 7×, PCDiff가 약 5.5× 가량 빠른 speed-up을 보고하며, 추가로 모델 크기와 점 개수(cardinality) 스케일링에서 점 해상도가 성능을 좌우하고 모델 스케일은 기여가 점진적으로 둔화되는 경험적 트렌드를 제시한다.



### Transformation Behavior of Images in Latent Spac (https://arxiv.org/abs/2606.24430)
- **Prior Approaches**: 히스토패톨로지 분류에서는 라벨이 적다는 한계를 줄이기 위해 데이터 변환(augmentation)과 인코더(encoder) 기반 임베딩을 함께 활용하는 흐름이 일반적이다. 또한 Barlow Twins, MoCoV2, SwAV, DINO처럼 self-supervised 방식으로 임베더를 학습하며, 변환에 대해 불변(invariant)한 표현을 기대한다. 하지만 “클래식 이미지 변환이 실제로 잠재공간에서 얼마나 상쇄되는지”를 체계적으로 정량 비교한 연구는 부족했다.

- **Core Contribution**: 본 논문은 히스토패톨로지 타일을 대상으로, 원본 이미지와 변환된 이미지의 임베딩이 잠재공간에서 얼마나 가까워지는지(또는 얼마나 달라지는지)를 비교해 임베더의 변환 민감도를 측정한다. 이를 통해 augmentation이 성능을 높이는 이유가 단순한 데이터 확대를 넘어, 임베딩 공간에서 변환 효과가 완전히 중화되지는 않는 현실을 반영할 수 있음을 보여준다. 또한 ImageNet 사전학습 기반 임베더와 병리학 특화 임베더 간 차이도 함께 관찰한다.

- **Technical Challenges**: 핵심 과제는 임베더의 “이상적인 불변성”이 이론적 기대치인 반면, 실제 학습된 네트워크가 유한 자원에서 불완전하게 변환 영향을 남긴다는 점을 정량화하는 것이다. 연구진은 각 변환 전후 임베딩의 L2 거리와, 동일 잠재공간에서의 random 임베딩 간 거리(ARD)를 정규화해 네트워크/변환 간 비교 가능하게 만들었다. 추가로 차원별 변화 분포와 Generalized Jaccard Index(GJI)를 사용해 어떤 잠재 차원들이 변환에 의해 공통적으로 영향을 받는지(부분적 disentanglement 실패)를 분석했다.

- **Empirical Impact**: TCGA와 내부 CAD 두 WSI 데이터에서 해상도(0.25~2 mpp) 및 데이터셋 간 유사한 경향을 확인했으며, 어떤 변환에서도 임베딩이 완전히 불변은 아니었다. 공간 변환(플립/크롭)은 상대적으로 영향이 작지만, 색상 관련 변환(color jitter, 그레이스케일/정규화 등)이 임베딩 이동을 더 크게 만들었고 이는 스캐너 간 색 차이가 분류 일반화에 영향을 줄 수 있음을 시사한다. 더불어 병리학 특화 인코더가 ImageNet 기반 baseline보다 더 높은 변환 견고성을 보였지만, 동시에 일부 변환이 특정 차원에 국한되지 않고 여러 차원에 분산되어 나타나 잠재공간 disentanglement이 완전하지 않다는 메시지를 남긴다. 



### EgoSAT: A Comprehensive Benchmark of Egocentric Streaming Interaction Understanding (https://arxiv.org/abs/2606.24422)
Comments:
          Accepted to ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 연구는 streaming VLM에서 지연·메모리·토큰 효율 같은 시스템 설계에 집중했지만, egocentric 환경의 ‘증거가 시간에 따라 생기고 사라지는’ 상호작용 추론을 충분히 통합해 평가하긴 어려웠습니다. 또한 장기 영상 이해 벤치마크는 대체로 오프라인 접근 또는 제한적 스트리밍 조건에서 최종 정확도 위주로 측정되어, present/prospective/retrospective을 한 프레임워크에서 함께 점검하기가 제한됩니다.

- **Core Contribution**: EgoSAT는 egocentric streaming interaction understanding을 위한 최초의 포괄 벤치마크로, 프레임이 순차로 도착하는 온라인 prefix 제약 하에서 과거·현재·미래 추론을 한꺼번에 평가하도록 설계됐습니다. 특히 query가 완료된 사건(retrospective), 진행 중 활동(present), 미래 행동(prospective) 중 무엇인지에 따라 ‘현재까지 관측된 것만으로 답할 수 있는가’를 답가능성(answerability) 관점에서 재구성합니다.

- **Technical Challenges**: 핵심 난제는 부분 관측(과거는 있으나 미래는 없음) 아래에서 모델이 (1) 상호작용 존재, (2) 현재 증거로 추론 가능성, (3) 불확실성의 타당한 갱신을 동시에 수행하는 것입니다. 논문은 Surprise·Branchiness로 미래 사건의 답가능성을 정량화하고, MCQ에서 confidence가 증거 축적과 답가능성을 추적하는지까지 진단하도록 벤치마크와 confidence diagnostics를 함께 구성했습니다.

- **Empirical Impact**: EgoSAT 평가에서 present narration는 일부 frontier 모델이 강점을 보이지만, prospective anticipation과 retrospective retrieval은 모델군 전반에서 여전히 큰 어려움을 보였습니다. 더 우려스러운 점은 confidence가 정답의 답가능성과 잘 맞지 않아 ‘확신에 찬 오답(confidently wrong)’이 반복된다는 것으로, 정확도 외에 answerability-aware 진단의 필요성을 실증적으로 제시합니다.



### Modality-Aware Out-of-Distribution Detection for Multi-Modal Action Recognition (https://arxiv.org/abs/2606.24404)
Comments:
          Accepted at ECCV '26

- **Prior Approaches**: 기존 멀티모달 동작 인식 OOD 탐지는 주로 A2D 계열의 학습 시 정규화에 의존하며, 추론(inference) 단계에서는 uni-modal OOD detector를 그대로 가져와 적용하는 경우가 많다. 그 결과 모달 간 관계를 추론에서 충분히 활용하지 못해 OOD 신호가 일부 버려질 수 있다. 또한 단순 확률/로짓/특징 기반 점수는 멀티모달 통합이 바뀌는 양상을 직접 모델링하기 어렵다.

- **Core Contribution**: 이 논문은 uni-modal 예측과 multi-modal 예측 사이에 ID 샘플에서 높은 상관 관계가 형성되고 OOD에서 그 관계가 달라진다는 발견을 바탕으로, 이를 명시적으로 쓰는 post-hoc modality-aware OOD detector를 제안한다. 새로운 detector는 multi-modal logits를 정규화하면서 uni-modal 기반 “관계 불일치” 점수와 feature-space off-manifold 점수를 함께 결합한 hybrid 스코어를 만든다. 학습 단계의 기존 방법(A2D, DPU 등)과 호환되며, 추가 학습 없이도 성능을 일관되게 끌어올린다.

- **Technical Challenges**: 핵심 기술적 도전은 멀티모달 추론에서 모달 간 통합 변화가 OOD에서 어떻게 드러나는지 “점수화”하는 방법을 설계하는 것이다. 이를 위해 (1) 검증 ID 데이터에서 uni-modal logits를 multi-modal logits로 근사하는 클래스별 선형 매핑을 학습해 분포 불일치(d)를 OOD 신호로 만들고, (2) 모달 임베딩을 이어붙인 feature 공간에서 off-manifold 방향을 PCA의 저분산 성분과 Ledoit-Wolf shrinkage로 추정한 뒤 Mahalanobis 거리로 점수화한다. 마지막으로 두 점수를 상위 로짓 기반 통계로 스케일 정규화한 뒤, virtual logits 형태로 분포와 함께 통합해 최종 OOD 확률/점수를 산출한다.

- **Empirical Impact**: Multi-OOD benchmark(여러 데이터셋, 최대 3개 모달, 총 14개 실험)에서 평균적으로 기존 SOTA를 능가하며, 특히 Far-OOD의 어려운 Kinetics-600 설정에서 FPR 기준으로 큰 폭의 개선을 보고한다. Near-OOD에서도 대부분의 경우 최상 성능 또는 상위권을 유지하며, 3모달 조합과 even early/mid-fusion 구조에서도 유사한 이득이 관찰된다. 또한 multi-modal 관계 점수 s(x), feature 점수 r(x), 확률/로짓 기반 신호 g(x)를 모두 결합할 때가 가장 강력해, 멀티모달 OOD 탐지에서 추론 시 모달을 분리해 고려해야 한다는 메시지를 실증적으로 뒷받침한다.



### MATCH: Flow Matching for Multi-View Anomaly Detection (https://arxiv.org/abs/2606.24375)
Comments:
          Accepted at ECCV 2026

- **Prior Approaches**: 기존 멀티뷰 이상탐지는 Normalizing Flows 기반 확률밀도 추정(예: RealNVP 계열)이나 kNN/메모리뱅크, 재구성형 방법 등이 주로 활용됐다. 하지만 멀티뷰 이미지는 고차원이며 뷰별로 결함이 일부만 나타나기 때문에, RealNVP 같은 결합(block) 중심 설계는 표현력 제약과 계산 부담으로 성능 격차가 생길 수 있다. 또한 Flow 기반 방법이어도 likelihood 추정에 필요한 divergence 항 계산이 병목이 되어 실시간 배치/배포에 불리했다.

- **Core Contribution**: MATCH는 멀티뷰 이상탐지를 위해 Flow Matching(Flow Matching, FM)을 최초로 정면 적용한 방법이다. ODE 기반 FM의 성질을 활용해 likelihood를 추정하고, 이를 바탕으로 객체-이미지-픽셀 수준의 anomaly score 및 segmentation을 동시에 도출한다. 더불어 FM 모델의 유연한 아키텍처로 서로 다른 해상도의 특징맵을 정상 분포로 효율 변환하며, divergence 항을 생략하는 단순화로 실사용성을 높였다.

- **Technical Challenges**: 핵심 난제는 (1) 멀티뷰처럼 고차원 특징공간에서 정상 분포를 안정적으로 학습하고 (2) likelihood 산출 시 divergence 항을 포함하면 ODE 단계마다 비용이 폭증하는 계산 문제를 동시에 해결하는 데 있었다. MATCH는 ODE-formulation Flow Matching(특히 OT-CFM)을 학습에 사용해 고차원 분포 표현력을 확보하고, likelihood 계산에서는 divergence 항을 생략해 계산 시간·메모리를 크게 줄이면서도 점수의 의미는 유지한다. 또한 WideResNet 기반 frozen feature 추출로 원본 이미지 대신 latent 분포에서 흐름을 학습하고, 시간/뷰 임베딩 및 리컨스트럭션 디코더로 뷰 불일치와 결함 위치를 세밀하게 반영한다.

- **Empirical Impact**: Real-IAD와 MANTA-Tiny에서 MATCH는 탐지와 세그멘테이션 모두에서 최첨단 성능을 보였고, Real-IAD의 경우 I-AUROC 91.17, P-AUPRO 94.76 수준을 보고했다. 특히 Flow 기반 경쟁모델(Multi-Flow)은 탐지에서 근소한 열세를 보였고, MATCH는 segmentation에서 더 높은 정밀도를 보이며 결함 위치를 더 잘 집어낸다는 점을 강조했다. 또한 소비자급 GPU 환경에서 동작 가능(약 18 FPS)하고 divergence 항 제거로 실시간 지향 구성을 달성했으며, MANTA-Tiny에 대해서도 포괄적 벤치마크를 제공해 후속 연구의 기준점 역할을 한다.



### Structural Kolmogorov-Arnold Convolutions: Learnable Function on the Values or the Filter Shape as Parameter-Efficient Alternative to Per-Edge Convolutional KANs (https://arxiv.org/abs/2606.24371)
- **Prior Approaches**: 기존 Convolutional Kolmogorov–Arnold Networks(ConvKAN)은 커널의 각 항목마다 입력 픽셀 값에 작용하는 univariate function을 따로 두는 per-edge 방식이 주류였다. 이 접근은 표현력은 높지만 함수 수가 입력 채널·출력 채널·커널 면적에 비례해 파라미터와 연산이 급증하고 과적합 위험도 커진다. 또한 맞춘 예산에서 정확도 이득이 제한적이거나, Gram 변형처럼 좋은 결과는 더 많은 파라미터를 요구하는 경향이 있었다.

- **Core Contribution**: 이 논문은 learnable function의 “위치”를 value(픽셀 값)와 shape(필터 구조)라는 단일 축으로 재정의하고, per-edge 대신 구조(필터 형상) 쪽에 function을 배치하는 structural KAN을 제안한다. RF-KAN은 Morlet wavelet 기반의 ridge profile로 필터 shape를 만들고, SV-KAN/AG-KAN은 픽셀 값에는 단일 shared function을 쓰되 shape는 각각 정적 필터 또는 content-adaptive Gaussian gate로 제공한다. 핵심 메시지는 동일한 파라미터 스케일에서 value에도 shape에도 function을 배치했을 때 “경쟁 구간”에 도달하며, per-edge 방식의 과도한 함수 분산 없이도 성능이 나온다는 점이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 function을 구조에 두면 학습이 어떻게 안정적으로 이뤄지고, shape 표현이 실제 이득으로 연결되는지 설계해야 한다는 것이다. 저자들은 RF-KAN에서 ridge projection(방향을 내적 좌표로 축약해 ridge 형태를 만들고), Morlet basis로 부드럽고 국소화된 진동적 프로파일을 조합하며, 채널-평균 패치 기반 routing으로 content-adaptive amplitudes를 더해 필터가 입력에 맞게 조정되도록 했다. 또한 RF-KAN의 연산 효율을 위해 연속 좌표상에서 필터를 구성한 뒤 bilinear upsampling을 선형적으로 접어 넣어, 큰 중간장을 다루는 비용을 줄이면서도 필터의 연속성을 유지했다.

- **Empirical Impact**: CIFAR-10/100에서 4-layer 동일 프로토콜로 비교했을 때 RF-KAN은 CIFAR-10 88.47±0.10% (약 0.40M 파라미터), SV-KAN은 88.20±0.31%로 per-edge KAN들과 plain convolution을 모두 앞섰다. CIFAR-100에서도 RF-KAN 64.40±0.19%, SV-KAN 64.57±0.30%로 거의 동률이며, 두 모델 모두 per-edge Gram 변형 대비도 파라미터 효율에서 우위를 보였다. 제어 실험/어블레이션은 성능 향상이 Morlet basis의 intrinsically localised oscillatory 표현과 content adaptivity의 조합에서 오며, learned shape를 제거하면 정확도가 40점 이상 붕괴해 “shape가 하중을 담당”한다는 점을 명확히 했다.



### SignNet-1M: Large-Scale Multilingual Sign Language Video Dataset with Downstream Benchmarks (https://arxiv.org/abs/2606.24361)
Comments:
          25 pages. Accepted to ECCV 2026

- **Prior Approaches**: 기존 SLT/SLR 데이터셋(Phoenix14T, How2Sign, CSL-Daily 등)은 거의 정면 촬영, 스튜디오 배경, 제한된 화자 풀로 수집돼 실제 배치 환경의 분포 변화(시점·배경·화자 외형)에 취약한 측면이 있었다. 그 결과 SpaMo, UniSign 같은 최신 모델도 시점/배경/화자 변화와 압축·광도 등 이미지·시간 도메인 잡음에 대해 성능이 크게 흔들리는 “coverage blind spot”이 드러났다. 또한 크롭·색상 지터 같은 픽셀 수준 증강은 부드러운 3D 일관성이나 구조적 시점/장면 편집을 충분히 합성하기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 SignNet-1M을 제안하며, ASL·CSL·DGS를 포괄하는 약 100만 클립 규모의 대규모 증강 데이터로 언어별·환경별 분포 편향을 완화한다. 핵심은 시점(viewpoint)은 3D Gaussian Splatting 기반 novel-view rendering, 배경(scene/background)과 조명은 diffusion 기반 scene editing, 화자(identity) 외형은 cross-reenactment signer substitution으로 각각 “축(axis)별”로 통제해 합성한다는 점이다. 여기에 post-rendering augmentations를 더해 촬영·압축으로 인한 in-the-wild 열화까지 닮도록 설계했으며, 통합 벤치마크와 성능 원인 분해용 ablation까지 제공한다.

- **Technical Challenges**: 기여를 가능하게 하려면 (1) 손·얼굴 등 미세 동작을 바꾸지 않으면서 시점·배경·화자 외형만 구조적으로 변형하고, (2) 시간적 일관성(영상 내 프레임 간 모션/조명 흐름)을 유지하며, (3) 라벨(글로스/번역)과의 정합성을 깨지 않아야 한다. 논문은 프레임별 SMPL-X/FLAME 기반 추적(EHM-Tracker)으로 identity와 motion을 분리한 뒤, GUAVA의 3D-aware 렌더링과 알파 합성으로 novel view를 생성하고, FlowPortal+IC-Light로 temporally consistent 배경/조명 편집을 수행한다. 이후 video-consistent 변환과 mild temporal resampling으로 저수준 열화를 추가해도 linguistic annotations을 보존하도록 파이프라인을 구성했다.

- **Empirical Impact**: 실험에서 SignNet-1M으로 학습한 모델은 cross-view, cross-background, cross-identity, post-rendering shift 전반에서 일관되게 일반화가 개선되며, i.i.d. 성능도 함께 유지되는 결과를 보였다. 특히 zero-shot에서 큰 BLEU-4/WER 악화가 발생하더라도(SignNet-1M 시험 셋으로의 분포 이동) Trained 설정이 이를 회복하며, 원래 벤치마크 테스트셋에서도 성능 향상이 나타나 자연 분포에서도 이득이 있음을 확인했다. severity(시점/조명 난이도)별로 보면 어려운 구간일수록 gain이 커져 “현실 강인성” 관점에서 증강 커버리지가 필수적임을 실증했으며, 데이터 합성의 시각적 품질도 FID-VID/FVD 및 PSNR/SSIM 기반으로 원본에 가깝게 유지된다고 보고한다.



### Open-Vocabulary BEV Segmentation with 3D-Aware Geometric Constraints (https://arxiv.org/abs/2606.24353)
Comments:
          This paper has been accepted by ECCV 2026

- **Prior Approaches**: 기존 BEV perception은 BEV와 2D 카메라 사이의 공간 대응을 깊이 기반, projection 기반, attention 기반, 3D Gaussian splatting 기반으로 구성해 실시간 성능을 끌어왔습니다. 하지만 대부분의 파이프라인이 학습·추론 모두 동일한 고정 클래스(closed-set)를 전제로 해, 실제 도로에서 새로 등장하는 객체 카테고리는 BEV 맵에서 무시되기 쉽습니다. 특히 2D 의미를 BEV로 옮기는 unprojection 계열은 희소 시점에서 2D-to-3D lifting이 ill-posed라 기하 불일치와 위치 왜곡 문제가 누적됩니다.

- **Core Contribution**: 이 논문은 open-vocabulary BEV segmentation(OVBS)이라는 과제를 제안해, 학습 셋에 없는 범주도 vision-language model(VLM)로 인식하면서도 정밀한 BEV 지각과 실시간 효율을 동시에 노립니다. 핵심은 unprojection이 아니라 3D 기하를 먼저 신뢰성 있게 아래로 투영하는 “3D projection-first” 관점을 도입한 OVBEVSeg입니다. OVBEVSeg는 열려있는 카테고리의 의미를 BEV에 일관되게 매핑하기 위해 3D 기하 제약을 활용합니다.

- **Technical Challenges**: OVBS에서 가장 큰 기술 난제는 2D VLM 의미를 BEV로 옮기는 과정에서 발생하는 3D geometric inconsistency(기하 불일치)입니다. 논문은 이를 Gaussian splatting(GS) 기반 unprojection 학습 전반에 기하 제약을 주입하는 OVBEVSeg로 해결하며, 단계적으로 (1) 신뢰도 높은 3D projection으로 2D-to-BEV pseudo-label을 만들고, (2) BEV 구조 제약으로 2D–BEV의 per-scene joint 최적화를 수행해 기하 일관성을 복원합니다. 마지막으로 (3) 복원된 기하를 distillation로 압축해 온라인 추론 시 메모리·연산 부담을 줄이면서도 기하 품질을 유지합니다.

- **Empirical Impact**: nuScenes에서 OVBEVSeg는 novel classes에 대해 closed-set 방식 대비 15.3 mIoU를 개선하며, 심지어 novel-class ground-truth 라벨이 없을 때도 self-/semi-supervised 기준과 경쟁 수준을 보입니다. 또한 projection 기반 방법 대비 2.5배 더 빠른 추론을 달성하면서 메모리는 0.22배 수준으로 크게 줄였습니다. 실험 결과는 open-world 안전을 좌우할 수 있는 “시맨틱 완전성”을 BEV에서 실질적으로 확보할 수 있음을 보여줍니다.



### TIGER: Taming Identity, Geometry, and Generative Priors for High-Quality Face Video Restoration (https://arxiv.org/abs/2606.24336)
- **Prior Approaches**: 기존 영상 복원(VSR)·확대/복원 방법은 합성곱·정렬(광류, 정합) 또는 Transformer에 기반해 열화를 완화해 왔지만, 얼굴 복원(FVR)에서는 identity drift(신원 변형)와 깜빡임 같은 문제가 두드러진다. 또한 diffusion 기반 접근이 인상적인 지각 품질을 주지만, 정적인 2D 참조를 그대로 쓰면 자세·표정 변화에서 identity와 구조 제약이 엉켜(뷰포인트-엔탱글) 시간적 일관성이 무너질 수 있다. 빠른 one-step 가속을 위한 회귀/증류 학습은 평균화로 인한 질감 뭉개짐이 생겨 사진 같은 realism과 효율의 균형이 어렵다는 한계가 있다.

- **Core Contribution**: 본 논문은 FVR의 핵심 난제 3가지(Identity Drift, Viewpoint-Entangled Guidance, Perceptual Realism)를 동시에 다루기 위해 TIGER를 제안한다. TIGER은 Identity Prior(ArcFace 임베딩으로 라땐트에 주체 앵커), Geometry Prior(2D 단서를 3D 파라미터 공간으로 분해·교차 결합해 normal map 생성), Generative Prior(1-step rectified flow로 한 번에 클린 매니폴드 이동)를 tri-prior로 결합한다. 특히 복원을 prior-conditioned latent transport로 재정의해, 구조·신원·생성 지식을 하나의 통합 흐름에서 안정화한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘정적 참조 이미지가 특정 자세/표정과 함께 신원 정보를 품고 있어’ 동적 프레임에서 일관된 구조·신원 제약을 제공하지 못한다는 점이다. TIGER은 3DMM/3DDFA-V3 기반으로 신원(참조에서 고정)과 모션(열화 동영상에서 프레임별 추정)을 disentangled 파라미터로 분리한 뒤, 교차-소스 fusion으로 자세 비의존 geometry를 구성한다(정확히는 normal map을 view-agnostic scaffold로 렌더링). 또한 효율을 유지하면서 realism을 놓치지 않기 위해 DiT 기반 one-step rectified flow에 priors를 채널 결합·cross-attention으로 주입하고, 구조→질감→분포 수준 사실감을 순차 강화하는 3-stage progressive 학습을 설계했다.

- **Empirical Impact**: 대규모 FVR 데이터셋(총 28,491개, 얼굴 중심 품질 필터링 포함)과 벤치마크를 구축해 표준화된 평가 환경을 제공한다. 실험 결과 TIGER은 VFHQ-Test와 VoxCeleb2에서 대부분의 지표에서 SOTA를 달성했으며, 특히 LPIPS·IDS 같은 지각/신원 보존과 FVD·VIRD 같은 시간적 일관성에서 큰 개선을 보였다. 정성적으로도 피부 질감과 미세 얼굴 특징이 더 선명하고, 복잡한 움직임에서도 프레임 간 일관성이 우수하게 나타나 “고품질·효율·신원 일관성”을 동시에 달성한 것으로 보고된다.



### Ill-Posed by Design: Probing Evidence Use in VLMs (https://arxiv.org/abs/2606.24335)
- **Prior Approaches**: 기존 counterfactual interventions는 이미지 일부를 가리거나 바꿔 모델 출력 변화를 보고 어떤 근거를 쓰는지 진단하는 데 널리 쓰인다. 다만 잘-정의된(여러 단서가 중복 지지하는) 비전 과제에서는 한 단서를 제거해도 예측이 안 바뀌어 ‘미사용’인지 ‘중복 근거’인지 구분이 어렵다. 기존 평가는 주로 성능 점수로 귀결돼, 강한 정답이 언어 prior만으로도 가능한지 세밀한 원인 분해가 부족했다.

- **Core Contribution**: 이 논문은 단일 단안 이미지에서 물리적 크기가 직접 관측되지 않는 monocular metric object-size estimation을 ill-posed 진단 설정으로 제안한다. 각 정답이 단일 단서로 결정되지 않으므로, counterfactual에서 예측이 흔들리면 해당 단서가 load-bearing 근거일 가능성이 높아진다. 또한 Metric VQA(Objectron 기반 10,482개 쿼리, tape-measured in-the-wild 331개 쿼리의 총 10,813차원)를 만들고, 6개 evidence 채널(범주 prior, 타깃 픽셀/정체성, 로컬 컨텍스트, 겉보기 크기, 전역 장면 기하)을 고정된 프롬프트 하에 영상 레벨로 분해한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘단안에서 스케일이 불충분’한 상황에서, 모델이 어떤 시각 단서(또는 언어 prior)에 의존하는지 의미 있게 분리해내는 것이다. 이를 위해 SAM으로 타깃 마스크를 만들고 object_only/random_mask로 타깃 픽셀 vs 면적 통제를, wrong_object/object_ring로 정체성 및 로컬 컨텍스트만 변경을 수행하며, 겉보기 크기는 마스크 기반 zoom-in/out으로, 전역 기하 영향은 방사형 렌즈 왜곡(r 정규화 기반)으로 겨냥한다. 그 결과를 해석하기 위해 text-only(이미지 제거) 및 텍스트-only frontier LLM(이미지 없이 질의만)을 기준선으로 두고, paired-bootstrap으로 개입 효과의 방향 비일관성까지 반영해 유의성을 분류한다.

- **Empirical Impact**: 실험에서 12개 open-weight VLM(33B~397B)은 Metric VQA의 in-the-wild split에서 이미지 없는 frontier 텍스트 LLM보다 전반적으로 뒤처졌고, Objectron split에서는 일부 큰 모델만 따라잡는다. 개입 시그니처는 target identity가 가장 일관되게 load-bearing이며, target pixels와 local context는 모델 일부에서만 유의미한 이득을 주고, 겉보기 크기는 예측을 바꾸되 방향성이 일관되지 않으며, 전역 scene geometry는 대체로 거의 활용되지 않는 패턴을 보였다. 또한 supervised LoRA fine-tuning은 정확도를 올리지만 장면 기하를 새로 학습한다기보다 기존에 부분적으로 쓰던 채널(특히 target identity, local context, 언어 prior 접근)을 보강하는 형태로 나타나 진단적 해석의 실용성을 함께 입증한다.



### UniTranslator: A Unified Multi-modal Framework for End-to-end In-Image Machine Translation (https://arxiv.org/abs/2606.24333)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 IIMT(in-image machine translation)는 주로 OCR–MT–Editing의 연쇄 파이프라인으로 처리해 효율은 좋지만, 단계가 늘수록 오류가 누적되기 쉽다. 최근 unified multimodal models(UMMs)은 이해와 생성을 한 프레임워크에서 다루지만, IIMT에 그대로 적응하면 번역 이해와 이미지 생성 간 학습 신호가 어긋나거나(understanding-generation conflict) 글자 위치 정렬이 틀어지는 경우가 잦다. 또한 diffusion 기반 생성은 전반적 자연스러움에 강하지만 텍스트 영역의 정밀한 기하 정합까지 보장하기 어렵다는 한계가 있다.

- **Core Contribution**: UniTranslator는 번역 이해와 텍스트 편집(visual writeback)을 단일 통합 모델에서 함께 최적화하는 IIMT 프레임워크를 제안한다. 핵심으로 Understand-Generation Alignment Module(UGAM)을 통해 이해 쪽에서 만든 표현과 생성 쪽 조건을 정렬해 의미 불일치를 줄인다. 여기에 Spatial Mask Decoder(SMD)를 두어 텍스트 영역에 대한 픽셀 단위 감독으로 위치·레이아웃 정밀도를 끌어올린다.

- **Technical Challenges**: 가장 큰 기술 과제는 언어 수준 번역의 ‘정답 동치’(여러 번역 후보 가능)와 픽셀 수준 렌더링 감독(특정 문자열의 특정 시각적 배치)을 동시에 만족시키는 충돌을 해결하는 것이다. UGAM은 이해 표현을 query로, 생성에 쓰는 비주얼 멀티모달 표현을 key/value로 하는 cross-attention 기반 정렬 임베딩을 만들어 생성 조건을 의미적으로 일관되게 만든다. 두 번째 과제는 생성 결과의 텍스트가 정확한 영역에 쓰이도록 강제하는 것으로, SMD가 텍스트 영역 마스크를 예측하도록 하고 BCE+Dice로 픽셀 단위 정합을 직접 학습해 spatial grounding을 강화한다.

- **Empirical Impact**: 여러 벤치마크에서 UniTranslator는 state-of-the-art 성능을 보이며 언어 방향(De→En, En→De, Fr→En, Ro→En)과 복잡한 실사 레이아웃 전반에서 일반화가 확인됐다. 특히 IIMT30k처럼 배경 텍스처가 복잡한 조건에서 FID가 크게 개선되어 현실 장면 편집에서 시각적 충실도가 향상됐음을 시사한다. 또한 ablation과 분석 결과는 UGAM과 SMD가 각각 의미 정합과 공간 정확도를 보완하며, 이해와 생성 최적화가 상호 강화(mutual reinforcement)되는 통합 학습의 장점을 뒷받침한다.



### REDI-Match: Rotation-Equivariant Distillation for Efficient and Robust Dense Matching (https://arxiv.org/abs/2606.24330)
- **Prior Approaches**: 기존 비전 foundation model(VFM)을 활용한 dense matching은 의미론적 표현을 강하게 가져오지만, in-plane 회전이 심할 때 성능이 급락하는 문제가 남아 있었다. 이를 보완하려는 데이터 기반 접근은 회전 데이터 augmentation을 크게 늘려야 해 파라미터/계산이 비효율적이고, 반대로 엄격한 equivariant 네트워크는 VFM급 의미 용량을 충분히 활용하기 어렵다는 한계가 공존한다.

- **Core Contribution**: 이 논문은 Rotation-Equivariant Distillation(REDI)라는 새로운 증류 패러다임을 제안해, 비등변(비 equivariant) 의미 공간을 가볍고 엄격한 rotation-equivariant 인코더로 옮기는 구조적 해법을 제시한다. 이후 REDI-Match는 entropy-driven 공간 정렬 모듈로 discrete rotation hypothesis를 평가해 전역 회전 모호성을 먼저 제거한 뒤, 연속적인 residual refinement로 세부 정합을 수행한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) VFM의 비등변 의미 표현을 equivariant 제약이 걸린 구조로 옮기되 (2) aliasing으로 인한 geometric phase 정보 붕괴를 막고 (3) 전역 회전을 학습이 아닌 추론 단계에서 안정적으로 canonicalize하는 것이다. 저자들은 distillation 단계에서 MSE 기반 loss로 의미를 안정적으로 이식하고, 인코더에는 grid-alignment modulo 조건을 통해 subsampling aliasing을 억제해 엄격한 equivariance를 만족시키며, 디코더에서는 entropy로 가장 날카로운 회전 가설을 선택해 기준 좌표계를 고정한 뒤 GP kernel regression 기반 보정으로 잔여 회전을 처리한다.

- **Empirical Impact**: 실험에서 REDI-Match는 여러 회전 벤치마크에서 SOTA를 달성했으며, 특히 극도로 어려운 SatAst에서 pose accuracy를 13.89%p 절대 향상시켰다. 또한 RTX 4090 단일 GPU에서 41 FPS(24.05 ms) 수준으로 현재 SOTA 대비 1.9x 빠르면서도 성능을 끌어올려, real-time dense feature matching의 실용성을 크게 확장한다.



### TrOCR for Medieval HTR: A Systematic Ablation Study with Cross-Dataset Validation (https://arxiv.org/abs/2606.24302)
Comments:
          Accepted at Document Analysis Systems Workshop 2026 (ICDAR Satellite event)

- **Prior Approaches**: 기존 HTR 연구는 역사 원고의 시각적 열화(번짐, 얼룩, 자간 변형)와 문자/약어 관습 차이 때문에, 현대 인쇄체로 사전학습된 모델이 동일 성능을 내기 어렵다는 문제를 반복적으로 지적해 왔다. 특히 fine-tuning에서 전처리(예: 대비 보정), augmentation, 최적화가 함께 튜닝되는 경우가 많아, 무엇이 성능을 좌우하는지 해석이 어려웠다. 또한 attention 시각화가 보편적이지만 정량적 진단 도구로 쓰이는 정도는 제한적이었다.

- **Core Contribution**: 이 논문은 TrOCR를 13세기 이탈리아 원고(Cortonese)와 READ-16에 적용하며, fine-tuning에서 3가지 제어 선택지(contrast normalization, data augmentation, encoder/decoder layer freezing)가 CER에 미치는 영향을 13개 구성의 통제 실험으로 분리해 제시한다. Cortonese에서 최적 구성은 8.03% CER을 달성했고, 통계적으로 enc_3 또는 dec_6 수준의 freezing은 full fine-tuning과 큰 차이를 보이지 않았다. 또 CLAHE 제거처럼 전처리 의존도가 생각보다 낮을 수 있음을 실증하며, 진단을 위해 Grad-CAM과 decoder cross-attention을 함께 사용한다.

- **Technical Challenges**: 핵심 난제는 소규모 역사 데이터에서 사전학습 분포와의 시각 도메인 갭을 얼마나 architecture/학습 제어로 흡수할지이다. 이를 위해 저자들은 preprocessing과 augmentation은 “한 번에 묶음” 단위로, freezing은 encoder/decoder 각각의 층 수를 체계적으로 스윕해 One-Cycle 스케줄 등 나머지 조건을 고정했다. 더 나아가 Grad-CAM이 토큰에서 비정보적인 all-zero 형태로 붕괴하는 등 예외 상황까지 고려해 attention과 gradient 기반 신호를 상호보완적으로 해석하도록 설계했다.

- **Empirical Impact**: Cortonese에서는 CLAHE와 전체 augmentation 번들이 모두 유의미한 개선을 만들지 못했고(최대 7.84% CER 수준), 대신 freezing의 안정성이 encoder/decoder 간에 크게 갈렸다는 점이 확인됐다. READ-16 재현 실험에서는 encoder freezing이 더 취약해졌고, augmentation은 Cortonese보다 명확히 유익해졌으며, enc_3_dec_6 같은 조합 전략은 데이터에 따라 full fine-tuning보다 유의하게 나빠질 수 있음이 드러났다. 결과적으로 “CLAHE는 선택”, “encoder freezing은 신중”, “augmentation과 combined freezing은 데이터별 재검증” 같은 실무 가이드가 정량적으로 제안되며, Grad-CAM/attention 진단은 ablation이 드러낸 실패 모드를 해석하는 도구로 의미를 가진다.



### MM-TRELLIS: Point-Cloud Guided Multi-Modal 3D Vehicle Generation in Autonomous Driving (https://arxiv.org/abs/2606.24301)
- **Prior Approaches**: 기존 3D 차량 생성은 단일 이미지 기반이 많아 폐색과 관점 변화에서 기하가 모호해지기 쉽고, 멀티뷰를 써도 구간이 잘려 보이는 경우가 잦아 정밀도가 떨어진다. 또한 NeRF/Neural rendering 계열로 메쉬를 뽑는 방법은 추론이 느리고 메쉬 품질이 제한되는 문제가 있었다. LiDAR를 쓰더라도 보통 학습 단계의 supervision으로만 활용되어, 생성 과정에서 직접적인 기하 제약을 주지 못한다.

- **Core Contribution**: MM-TRELLIS는 TRELLIS류 native 3D 생성기의 강한 3D 형상 priors를 유지하면서, 자율주행 데이터의 멀티모달 입력을 zero-shot 방식으로 결합하는 프레임워크다. 멀티뷰 이미지는 conditioning 입력으로 cycle 방식에 넣되, LiDAR point cloud를 test-time guidance로 denoising 궤적에 결합해 기하 정확도와 cross-view consistency를 동시에 노린다. 또 orientation 불일치를 pose optimization으로 해결하고, 3D Gaussian Splatting의 opacity를 기반으로 floaters를 줄이는 voxel filtering까지 포함한다.

- **Technical Challenges**: 핵심 난점은 (1) 멀티뷰 조건이 시점/폐색으로 인해 denoising 과정에서 상충되는 gradient를 만들 수 있고, (2) 이미지로부터 유도되는 voxel의 방향이 기준(가이드)과 어긋나 최적화가 불안정해질 수 있다는 점이다. MM-TRELLIS는 LiDAR 점군을 voxel grid로 만들어 LiDAR-occupied voxel에만 masked BCE로 supervision을 걸어 불완전한 점군에서도 학습 저하를 줄이고, denoising 초기에 Chamfer Distance 기반 회전 최적화로 pose를 맞춘 뒤 안정적으로 guided generation을 진행한다. 마지막으로 3DGS에서 낮은 opacity splat을 prune해 메쉬의 불필요한 추가 기하를 억제한다.

- **Empirical Impact**: Waymo 데이터셋 실험에서 MM-TRELLIS는 novel-view 렌더링 품질(PSNR, LPIPS)과 기하 정확도(일방/쌍방 Chamfer Distance) 모두에서 기존 방법을 능가했다고 보고한다. 특히 PSNR/SSIM이 높아도 실제 형상이 틀리던 실패 사례들이 Chamfer Distance 및 시각 결과에서 드러나며, MM-TRELLIS는 정밀한 비율과 폐색 보완까지 더 일관된 메쉬를 만든다. 단일 GPU 기준 차량 1대 재구성에 37.3s가 걸리며, ablation을 통해 LiDAR guidance, pose optimization, opacity 기반 filtering이 각각 기하 정렬 안정성과 floaters 제거에 기여함을 확인한다.



### Training-free Cross-domain Few-shot Segmentation via Robust Semantic Representation and Matching (https://arxiv.org/abs/2606.24297)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 CD-FSS는 source 도메인에서 학습하거나, target 도메인에서 fine-tuning하는 방식이 많아 계산 비용이 크고 overfitting 위험이 동반된다. 또한 backbone을 ImageNet 사전학습에 의존하는 경우가 많아 도메인 갭에 취약하며, 비슷한 vision foundation model을 넣어도 성능이 크게 늘지 않거나 오히려 악화되는 현상이 관찰된다. 일부 training-free 접근은 단일 도메인에 맞춰져 있거나 추가 VFM이 필요한 한계가 있다.

- **Core Contribution**: 이 논문은 CD-FSS에서 vision foundation model을 쓰면 학습 기반 패러다임이 과적합에 취약해진다는 점에 주목하고, trainable parameter를 제거한 training-free 프레임워크를 제안한다. DINOv3 self-supervised 비전 인코더를 기반으로, 도메인 간 semantic 차이를 줄이면서도 학습 오버헤드 없이 세그멘테이션을 수행하도록 설계했다. 세 모듈 SAFR, ASE, HPM을 통해 의미성 강화와 support-query 정합, 그리고 매칭 적응성을 동시에 노린다.

- **Technical Challenges**: 핵심 기술적 난제는 강력한 DINOv3 특징이 로컬 일관성에는 강하지만 semantic discriminability는 약해 cross-domain에서 의미 있는 foreground-background 구분이 흐려질 수 있다는 점이다. 이를 해결하기 위해 SAFR는 position-sensitive 채널을 줄이도록 레이어를 선택·가중 재융합해 더 의미 분별력 있는 특징을 만든다. 이어 ASE는 support와 query 사이 semantic gap을 줄이기 위해 robust 영역만 골라 training-free cross-attention으로 query 정보를 support에 주입하고, HPM은 도메인별 semantic 복잡도 차이를 반영해 global/regional/pixel prototype 매칭 결과를 하이브리드로 결합한다.

- **Empirical Impact**: 네 개의 타깃 도메인 데이터셋에서 제안 방법은 학습 없이 state-of-the-art 성능을 달성하며, 기존 CD-FSS의 학습/미세조정 의존성을 실질적으로 대체할 수 있음을 보여준다. 특히 VFM을 넣었을 때 기존 학습 기반이 악화되던 관찰을 training-free로 전환해 성능 저하를 방지한 점이 의미 있다. 이 결과는 cross-domain few-shot 세그멘테이션에서 “학습 대신 표현·매칭 설계”로도 큰 개선이 가능하다는 방향성을 제시한다.



### Hierarchical Spatial and Channel Aggregation for Cross-domain Few-shot Segmentation (https://arxiv.org/abs/2606.24296)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 CD-FSS는 스타일 갭으로 인한 분포 차이를 줄이는 데 집중했지만, 도메인 간 클래스의 의미 세분성(예: 머리/몸 vs 전체)이나 판별 속성(예: 색상 같은 속성)의 차이는 충분히 다루지 못했습니다. 그 결과 학습 중 support–query 정합이 과도하게 맞춰져 semantic over-alignment과 attribute over-alignment 같은 성능 저하가 발생한다는 문제가 지적됩니다. 일부 후속 연구는 semantic over-alignment만 부분적으로 완화하지만, 다양한 세분성 요구와 attribute over-alignment까지 동시에 처리하는 데 한계가 있었습니다.

- **Core Contribution**: 이 논문은 semantic over-alignment과 attribute over-alignment를 동시에 겨냥하는 Dual Hierarchical Aggregation Network (DHANet)를 제안합니다. DHANet은 공간 차원과 채널 차원에서 각각 계층적 집계를 수행해 서로 다른 의미 세분성과 속성 활성 패턴에도 유연하게 대응하도록 설계됐습니다. 또한 Online Probabilistic Semantic Bank (OPSB)로 테스트 시점에 클래스 확률 분포를 온라인으로 누적·갱신하고, 그 분포에서 다중 pseudo-prototype을 샘플링해 부족한 support 정보를 보완합니다.

- **Technical Challenges**: 핵심 난제는 서로 다른 도메인에서 ‘같은 클래스’라도 의미 단위와 속성이 다르게 나타나는데, 기존 정렬 학습이 단일 granularity의 hard alignment를 유도해 과정합을 만들었다는 점입니다. 논문은 Hierarchical Spatial Aggregation (HSA)로 멀티스케일 공간 slot과 slot attention을 통해 의미 세분성별 계층 표현을 만들고, Hierarchical Channel Aggregation (HCA)로 채널 slot을 이용해 속성 단위의 계층 대표를 형성해 채널 정렬로 인한 속성 붕괴를 완화합니다. 더 나아가 테스트에서는 per-episode fine-tuning 없이, OPSB가 질의 예측에서 고신뢰 특징을 모아 class probability distribution을 업데이트하고 샘플링된 pseudo-prototype으로 보조 정합을 수행하도록 구성했습니다.

- **Empirical Impact**: PASCAL VOC 2012+SBD를 source로 학습하고 Deepglobe, ISIC, Chest X-ray, FSS-1000의 4개 target 도메인에서 mIoU를 평가한 결과, 제안 방법이 state-of-the-art 성능을 달성했다고 보고됩니다. 특히 스타일 갭뿐 아니라 도메인 간 클래스 의미 세분성·속성 차이가 큰 데이터에서 개선이 두드러질 것으로 해석됩니다. CD-FSS 연구에서 과도 정렬(over-alignment) 메커니즘을 의미/속성의 이중 관점으로 분해해 다중 granularity 대응과 테스트 시점 보정(OPSB)을 결합한 점이 실용적 의미가 큽니다.



### ActiveScope: Actively Seeking and Correcting Perception for MLLMs (https://arxiv.org/abs/2606.24292)
Comments:
          ICML 2026

- **Prior Approaches**: 기존 MLLM의 fine-grained 시각 정밀인식은 주로 localize-then-zoom-in(주의/검색 기반) 패턴을 따르지만, 초기 국소화 실수가 이후 단계에 연쇄적으로 전파되는 문제가 있습니다. 또한 단일 두드러진 영역에 집중하는 방식은 다중 객체 쿼리에서 순차적(모든 타깃) 바운딩 추적을 잘 못합니다. 결과적으로 distractor(주의를 빼앗는 요소)에 흔들리거나, 전역 의미에 고정돼 일부 객체를 놓치는 사례가 반복됩니다.

- **Core Contribution**: 이 논문은 localization 실패 원인을 Contextual Dominance(강한 시각적 distractor가 타깃 attention을 압도)와 Semantic Bias(첫 토큰의 전역 의미가 가장 salient한 개념에 고정)를 각각 부정확/불완전 국소화로 분해해 제시합니다. 이를 바탕으로 training-free 프레임워크 ActiveScope를 제안하며, 두 모듈로 “찾고(Active) 바로잡는(self-correction)” 인식 전략을 구현합니다. SAL(Semantic Anchor Localization)은 타깃을 설명 키워드로 앵커링해 의미 편향을 줄이고, ISR(Interference-Suppressed Refinement)은 오탐 영역의 간섭을 억제해 국소화를 정밀화합니다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) 한 번의 attention 기반 one-shot 국소화가 distractor에 속아 어긋나는 문제와 (2) 첫 생성 토큰 의존이 다중 객체를 전부 커버하지 못하는 의미 편향을 동시에 다루는 것입니다. 저자들은 SAL에서 객체별 키 토큰에 대한 cross-attention을 정밀 의미 앵커로 사용하고, 상대 attention으로 잡음을 줄인 뒤 다중 스케일 윈도우로 후보 박스를 뽑습니다. 이어 ISR에서는 Verification으로 불일치 후보를 가려낸 뒤, 실패한 영역과의 attention 연결을 hard masking(-infinity)으로 차단해 제한된 사이클(C≤3) 내에 재국소화를 유도합니다.

- **Empirical Impact**: V∗V^{*} Bench, HR-Bench 4K/8K, MME-RealWorld-Lite 등 고해상도 벤치마크에서 ActiveScope는 기존 training-free 방법(예: ZoomEye, ViCrop, ICoT)과 비교해 전반적으로 우수한 성능을 보였습니다. 예를 들어 Qwen3-VL 4B에서 V∗V^{*} Bench 평균 정확도 96.34%로, 경쟁 최상대 대비 3.15%p, 베이스 대비 6.81%p 향상으로 fine-grained 인식에서의 효과를 입증했습니다. 또한 정성 시각화와 ablation에서 SAL의 상대 attention 정규화, ISR의 간섭 억제와 3회 내 수렴 특성이 성능을 좌우함을 확인하며, 여러 베이스 모델에도 확장 가능한 training-free 보완책으로 의미가 큽니다.



### UniRED: Unified RGB-D Video Frame Interpolation with Event Guidanc (https://arxiv.org/abs/2606.24282)
- **Prior Approaches**: RGB 비디오 프레임 보간은 Super SloMo, PerVFI, RIFE처럼 양 끝 프레임만으로 중간을 복원하는 flow-and-synthesis 중심 방법이 강세를 보였다. 다만 빠른 동역학/복잡한 가림에서 motion drift, 블러, 고스트 같은 문제가 커지고, depth 보간은 hand-crafted priors나 sparse LiDAR 기반 파이프라인에 치우쳐 dense per-pixel depth에는 취약했다.
이와 별개로 event를 RGB 보간에 결합하면 micro-temporal 단서로 under-constrained 문제를 완화할 수 있지만, depth 보간까지 확장해 RGB-D 일관성을 직접 강화한 통합 프레임은 거의 없었다.

- **Core Contribution**: 이 논문은 RGB, depth, event를 한 아키텍처에서 동시에 다루는 통합 프레임워크 UniRED를 제안한다. RGB 보간과 depth 보간을 공동 모델링하고, event를 “중간 매개 도메인”으로 삼아 appearance(모양)와 geometry(형상)의 상호오염을 줄이면서 geometry-consistent 보간을 목표로 한다.
또한 tri-modal 학습/평가를 위한 RGB-D-Event 데이터셋(예: SyncRDE-60 등)을 구축해, 기존에 부족했던 동기화된 다중모달 벤치마크를 제공한다.

- **Technical Challenges**: 핵심 난제는 (1) 두 boundary 관측만으로는 중간 시점의 motion/visibility가 under-constrained인 점, (2) RGB와 depth의 직접 융합 시 depth discontinuity에서 RGB 디테일이 새거나 depth 결함이 RGB 특징을 오염시키는 점이다. 논문은 이벤트(voxelized event)를 통해 시간적으로 정렬된 신호로 교차모달 상호작용을 수행하고, 융합 후 bidirectional flow를 추정하되 RGB에는 motion basis refinement, depth에는 Z-axial refinement를 분리 적용해 축방향 오차를 보정한다.
마지막으로 전/후방 warping 결과를 soft-blending visibility map으로 적응적으로 합성해 가림과 경계 주변에서 아티팩트를 줄인다.

- **Empirical Impact**: 공개 벤치마크와 새로 만든 tri-modal 데이터셋에서 UniRED는 RGB 보간에서는 photometric fidelity를, depth 보간에서는 geometric accuracy를 기존 방법보다 더 높게 달성했다고 보고한다. 특히 RGB-only event-guided 접근 대비 depth까지 함께 다룰 때 기하 일관성 측면의 개선이 뚜렷하며, 실제로 통합 융합 전략과 branch별 refinement가 성능 향상에 기여함을 실험으로 뒷받침한다.
결과적으로 RGB-D-Event 기반 보간 연구에서 “단일 프레임 기반 관측의 한계”를 event로 보완하면서도 depth의 metric 구조를 보존하는 새로운 레퍼런스를 제공한다.



### MotifGen: Spatiotemporal interpolation of misaligned satellite images via multi-source generative modeling, in an application to tropical cyclones (https://arxiv.org/abs/2606.24263)
- **Prior Approaches**: 열대저기압 모니터링에서 마이크로파 영상은 위성 궤도 제약으로 재방문 간격이 길어 급격한 변화를 놓칠 수 있어, 관측을 시간·공간으로 보간하는 연구가 필요해졌다. 기존 생성모델·도메인적응·확산 기반 접근은 주로 입력과 같은 영역/고정된 단일 소스에 머물거나, 단일 센서 입력에 의존해 서로 다른 센서 특성과 불규칙한 관측 시점을 동시에 다루기 어렵다는 한계가 지적됐다. 또한 결정론적 RMSE 최적화 방식은 확률적 보간 과제 특성상 예측을 흐리게 만들고 물리적으로 그럴듯한 분포를 제대로 재현하지 못하는 문제가 있었다.

- **Core Contribution**: 이 논문은 서로 다른 지리적 위치에서, 불규칙한 시간 간격으로, 센서 특성이 다른 다중 geospatial 소스를 입력으로 받아 보간을 수행하는 최초 수준의 생성적 프레임워크를 제안한다. 목표는 기준 마이크로파 계측기(GPM Microwave Imager, GMI)의 영상을 원하는 시간·공간에서 생성(재구성)하는 것으로, 다중 마이크로파 계측기와 적외선 관측을 함께 활용한다. 훈련은 self-supervised 마스킹·재구성 형태로 설계해 보간이 본질적으로 확률적이라는 점을 모델링한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 관측 시점이 불규칙하고, (2) 관측이 지리적으로 어긋나며, (3) 센서별 주파수·시야각·해상도·스와스 폭 등 특성이 달라 단일 소스 기반 학습이 일반화되기 어렵다는 점이다. 논문은 flow matching(CondOT 경로) 기반 생성모델로 확률분포에서 샘플을 만들도록 하고, ViT 계열의 멀티소스 트랜스포머에서 위도·경도·시간을 공통 의미의 positional encoding으로 사용하며, 소스 간 정보 교환을 위한 창(window) 기반 cross-source attention을 설계해 계산 비용을 줄이면서도 misalignment을 흡수한다. 또한 표적 소스 픽셀만 잡음으로 오염시키고 입력 소스는 그대로 두는 self-supervised reconstruction으로 학습 데이터 효율을 끌어올린다.

- **Empirical Impact**: TC-PRIMED v01r01 데이터셋에서 마이크로파(여러 센서)와 적외선(11µm)을 함께 사용했을 때, supervised 학습 대비 Continuous Ranked Probability Score(CRPS)를 유의미하게 낮추는 결과를 보였다. IR을 추가하면 microwave only 대비 추가 개선이 확인됐고, 생성모델의 ensemble mean은 결정론적 기준선과 동등한 수준의 결정론적 지표(RMSE 등)를 유지하면서도 power spectrum은 실제 관측에 더 가깝게 맞췄다. 요약하면, 불규칙·다중·이질적 위성 관측을 결합한 생성적 보간이 실제 통계적 성질(스펙트럼)까지 더 잘 보존한다는 점에서 열대저기압 원격탐지 보간 분야에 실증적 의미가 크다.



### 3DCarGen: Scalable 3D Car Generation via 3D-consistent Multi-view Synthesis (https://arxiv.org/abs/2606.24257)
- **Prior Approaches**: 기존 3D 카 생성은 최적화 기반 또는 diffusion 기반의 2단계 파이프라인이 주를 이뤘다. 하지만 multi-view diffusion은 보통 4~6개의 고정 시점만 생성하거나 뷰 간 국소 기하 불일치가 생겨, 재구성 정밀도와 일관성이 떨어진다는 한계가 있었다. 또한 실제 이미지는 perspective 카메라인데, orthographic 가정을 두는 방법은 왜곡·기하 불일치가 심해질 수 있다.

- **Core Contribution**: 3DCarGen은 단일 실사용 이미지 입력만으로 고품질 3D 카를 만드는 스케일러블 프레임워크다. 핵심은 MV-SI(Multi-view Splatter Image)로 임의 시점에서 충분히 밀도 높은(그리고 3D-consistent한) 다중 뷰 이미지를 합성하고, 이를 coarse 3D 표현으로 동기화해 multi-view diffusion이 임의 카메라 뷰에서도 일관성을 유지하도록 하는 것이다. 이후 ISOMER+를 확장해 색상까지 포함한 joint 최적화로 디테일하고 일관된 메쉬를 빠르게 복원한다.

- **Technical Challenges**: 어려운 점은 (1) 실제 카메라의 perspective·가변 intrinsic 조건에서도 뷰 간 기하 일관성을 확보하고, (2) diffusion이 국소적으로 서로 다른 뷰를 생성해 생기는 불일치를 후속 3D 재구성이 견딜 수준으로 줄이는 것이다. 3DCarGen은 Zero123++ 전처리로 시점을 통일한 뒤 MV-SI로 저해상도이지만 기하적으로 정합된 3D Gaussian 기반 coarse 표현을 만들고, 이를 multi-view diffusion의 조건으로 삼아 임의 시점 생성에서도 동기화를 유지한다. 또 SDEdit 기반 multi-view sampling(중간 노이즈에서 출발)으로 coarse 렌더의 ‘흐리지만 기하 정확’한 특성을 활용해 뷰 간 코히런스를 강화한다.

- **Empirical Impact**: 합성(SRN-Cars, Objaverse 계열)과 실세계 Sketchfab·3DRealCar의 in-the-wild 평가에서 3DCarGen은 기존 방법 대비 다중 뷰 지오메트리 일관성과 재구성 품질이 우수함을 보였다. 정량적으로는 NVS에서 LPIPS 등 지각 품질이 개선되고, 메쉬 평가에서는 CD·VIoU에서 더 정확한 형상 복원이 확인됐다. 또한 ablation에서 MV-SI 동기화와 multi-view SDEdit 샘플링, 그리고 ISOMER+의 색상-정규 joint 최적화가 미러 같은 세밀 구조 복원에 직접 기여함을 보여주며 실용성을 뒷받침한다.



### Trimming the Long-Tail of Visual World Modeling Evaluation (https://arxiv.org/abs/2606.24256)
- **Prior Approaches**: 기존 이미지/비디오 world model 평가는 주로 head 시나리오, 즉 데이터에 자주 등장하는 도구-작업 조합의 시각적 그럴듯함과 기본적인 물리 상식에 초점을 맞춰 왔다. 그 결과 높은 벤치마크 성능은 진짜 물리 원리 내재화인지, 학습 데이터의 통계적 패턴 재현인지를 구분하기 어렵다는 한계가 있다.

- **Core Contribution**: TailOR는 irregular(긴 꼬리) 물리 상호작용을 시뮬레이션하도록 world model을 시험하는 벤치마크로, Regular- Unconventional- Impossible의 3단계 시나리오 모드를 설계했다. 또한 Predictive generation(결과 비공개로 예측)과 Descriptive generation(원하는 결과를 명시) 두 설정을 함께 제공해 ‘추론’과 ‘지시 준수/실현’ 능력을 분리 평가한다.

- **Technical Challenges**: 긴 꼬리 상호작용을 체계적으로 만들기 위해 필요한 도구 속성(affordance)과 제약 조건을 조절하는 데이터 생성 파이프라인을 구축했다. HICO-DET 기반 행동 온톨로지와 object–affordance graph를 활용해 대체 도구는 attribute-compatible로, 불가능 도구는 attribute-violating으로 만들고, 상태 변화/실패 결과까지 함께 정의한 뒤 체크리스트 루브릭으로 instruction adherence와 interaction correctness를 정량화한다.

- **Empirical Impact**: 실험에서는 Regular에서 Unconventional, Impossible으로 갈수록 Interaction Accuracy와 Physical Realism이 일관되게 하락해 긴 꼬리 일반화 격차가 뚜렷함을 보여준다. 이미지 모델은 올바른 상태 변화·속성 반영에 실패하는 경향이 크고, 비디오 모델은 여기에 시간적 일관성 붕괴가 추가로 발생하며, Descriptive 설정에서도 지시를 무시하고 익숙한 패턴으로 회귀하는 사례가 확인된다.



### Social Structure Matters in 3D Human-Human Interaction Generation (https://arxiv.org/abs/2606.24255)
- **Prior Approaches**: 기존 text-to-motion 기반 HHI(인간-인간 상호작용) 연구는 두 사람의 모션을 “그럴듯하게” 생성하는 데 집중해, 단일 인물 모션 생성기를 2인으로 확장하거나 모션 프리어를 조합하는 방식이 많았다. 하지만 이런 접근은 HHI의 핵심인 시간적 단계(phase progression)와 비대칭 역할(initiator/receiver 등), 두 사람의 상호 조율을 조직하는 ‘사회적 구조’를 명시적으로 중간표현으로 드러내지 못해, 접촉 타이밍이나 방향, 상호작용 기하가 어긋나는 문제가 남는다. 또한 LLM을 직접 모션 생성기로 쓰는 경우에도 연속적·물리적으로 타당한 3D 상호작용 실행 품질이 한계가 있다는 진단이 제시된다.

- **Core Contribution**: 이 논문은 HHI 생성 문제를 ‘사회적 구조(social structure) 모델링 및 그라운딩(grounding)’으로 재정의한다. 즉, 먼저 상호작용이 어떤 단계로 전개되는지(approach/contact/release/in-place)와 각 단계에서 두 배우의 partner-aware 역할이 무엇인지 추론하고, 그 구조를 연속적이고 물리적으로 그럴듯하며 파트너를 고려한 3D 모션으로 실현하도록 만든다. 이를 위해 “Think with LLM, Move with Motion Skill” 패러다임(LLM은 planner, 모션 모델은 executor)을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM이 언어를 통해 사회적 구조를 ‘생각’(phase 분해·역할 추론)할 수는 있어도, 이를 바로 동역학/기구학적으로 정확한 연속 3D 상호작용 ‘움직임’으로 ‘실행’하긴 어렵다는 점이다. 저자들은 이를 해결하기 위해 (1) LLM 기반 사회적 구조 플래닝으로 phase 레이블과 partner-aware 역할을 motion-aligned한 형태로 재구성하고, (2) Solo-to-Social(S2S) 실행기에서 phase-wise self-conditioning(이전 phase 모션을 앵커로 사용)과 ego-relative partner conditioning(파트너 최신 상태를 상대좌표로 주입), 그리고 LoRA로 pretrained solo 백본을 효율적으로 상호작용 실행기로 적응한다.

- **Empirical Impact**: InterHuman/InterX 같은 표준 HHI 벤치마크에서 기존 방법 대비 텍스트-모션 정렬, phase consistency, partner-aware coordination이 동시에 개선되었다. 정량적으로는 phase progression 측면에서 R-Precision과 사용자 평가의 Phase ranking이 가장 좋았고, 파트너 조율 측면에서는 평균 FID가 가장 낮아 상호작용 구성 품질을 유지하면서도 물리적 현실성이 향상됐다. 특히 social structure planning만으로는 부족하며 S2S 그라운딩이 필요함이 ablation에서 확인되며, 다양성(multimodality/diversity)은 유지한 채 성능 이득을 달성한 점이 의미 있다.



### TuringViT: Making SOTA Vision Transformers Accessible to A (https://arxiv.org/abs/2606.24253)
- **Prior Approaches**: 기존 VLM/VLA는 SigLIP2 같은 off-the-shelf ViT를 시각 인코더로 재사용하는 경우가 많지만, 해상도·지연·시간적 모델링 요구가 달라지면 성능과 비용 측면에서 한계가 드러납니다. 또한 softmax attention 기반 ViT는 고해상도·동적해상도에서 시퀀스 길이가 늘며 학습/배포 비용이 이차적으로 커져, 낮은 해상도로 사전학습 후 후처리 적응(post-hoc adaptation)에 의존하는 경향이 있습니다. 마지막으로 대규모 웹 image-text 데이터를 “그대로” 쓰면 잡음, 약한 정렬, 중복 감독이 누적되어 데이터 효율이 낮아지는 문제가 큽니다.

- **Core Contribution**: TuringViT는 VLM-native Vision Transformer로서 (1) 효율적인 아키텍처, (2) supervision 강화 데이터 큐레이션, (3) 처음부터 native dynamic-resolution을 학습하는 패러다임을 함께 제안합니다. 핵심 아이디어는 선형 복잡도에 가까운 TLA로 고해상도 시퀀스 부담을 줄이고, VISTA-Curation으로 이미지·비디오의 “품질 높은” 정렬 감독을 만들어 데이터 사용량을 줄이는 것입니다. 그 결과, TuringViT는 오픈소스 ViT 대비 10% 데이터로도 더 나은 비전 표현과 VLM 성능을 보이며, 고해상도 입력에서 지연(latency) 스케일링이 크게 개선됩니다.

- **Technical Challenges**: 문제는 (a) 동적/고해상도에서 긴 토큰 시퀀스를 다루면서도 학습 안정성과 국소 고주파 정보를 유지해야 하고, (b) 웹 데이터의 잡음이 정렬 학습을 비효율적으로 만들며, (c) 고정 해상도 사전학습과 달리 다운스트림의 다양한 입력 크기 요구를 미스매치 없이 흡수해야 한다는 점입니다. TLA(Turing Linear Attention)는 대부분의 attention을 선형 복잡도 경로로 대체하되, 주기적으로만 vanilla MHA를 끼워 넣어 토큰 간 상호작용을 희소하게 보존하고, 입력-dependent gate 및 길이 기반 정규화로 variable token length에서도 안정성을 확보합니다. 데이터 측면에서는 VISTA-Curation이 이미지 품질 필터링, 다중 캡션 생성/검증, 공유 풀 기반 relative s-CLIPLoss 스타일 스코어링과 텍스트 정보도 가중을 통해 discriminative·grounded 감독을 선택하며, 비디오는 시간 분해·대표 프레임 샘플링·의미/모션 일관성 필터링·로컬-글로벌 캡션 융합으로 transformation-aware 감독을 구성합니다.

- **Empirical Impact**: 실험에서 TuringViT-18L/24L은 약 0.85B image-text로 학습되며, 6개 zero-shot 분류 및 2개 retrieval 벤치마크에서 SigLIP2 등 강한 기준선을 능가하면서도 데이터 사용량은 10% 수준임을 보여줍니다. 특히 ImageNet-1K/v2/A 등 분포 이동이 큰 구간에서 강점이 나타나며, retrieval 성능 향상은 데이터 큐레이션이 만든 정확하고 잘 정렬된 image-text 쌍과 연결됩니다. 또한 dense prediction(깊이/세그먼트/트래킹)과 VLM 통합에서도 일관된 개선이 보고되어, TuringViT가 단순한 “더 빠른 인코더”를 넘어 VLM 호환성과 데이터 효율을 함께 끌어올리는 범용 비전 백본임을 실증합니다.



### M^2C-EvDet: Multi-Domain Multi-Order Cross-Modal Knowledge Distillation for Event-based Object Detection (https://arxiv.org/abs/2606.24248)
- **Prior Approaches**: 이벤트 기반 object detection(EvDet)은 고시간 해상도·넓은 다이내믹 레인지가 장점이지만, 이벤트 데이터의 희소성 때문에 RGB 프레임 기반 검출 대비 성능 격차가 크게 발생한다. 기존 cross-modal 지식 증류는 공간 의미(단일 spatial domain)나 쌍(pairwise) 관계(저차 관계)에 집중해 복잡한 장면에서 정보 전달이 제한된다는 한계가 있었다. 또한 RGB-이벤트 동기화 의존을 줄이기 위한 RGB-guided 증류 연구들이 늘었지만, 주파수/관계의 “다중 도메인·다중 차수” 관점이 부족했다.

- **Core Contribution**: 본 논문은 RGB에서 얻은 시각적·질감 정보를 이벤트로 더 robust하게 옮기기 위해 M^2C-EvDet(Multi-domain Multi-order Cross-modal knowledge distillation) 프레임워크를 제안한다. 핵심은 주파수 학습과 하이퍼그래프 연산을 바탕으로, 특징 증류를 두 축—(1) AF^2D^2(Adaptive Frequency-Decoupled Feature Distillation)과 (2) MORD(Multi-Order Relational Distillation)—로 확장했다. AF^2D^2는 저주파/고주파 의미를 분리해 modality-agnostic vs modality-specific 혼선을 줄이고, MORD는 저차·고차 관계를 함께 모델링해 복잡한 다중 객체 상관을 전한다.

- **Technical Challenges**: 문제의 첫 난관은 이벤트-가이드 증류에서 공간 도메인 직접 학습이 aliasing을 만들어 저·고주파가 섞이며 글로벌 의미와 로컬 윤곽이 동시에 왜곡될 수 있다는 점이다. 이를 해결하기 위해 AF^2D^2는 multi-scale wavelet 기반으로 주파수 성분을 저주파/고주파로 분해하고, adaptive wavelet fusion으로 객체 크기·형상 변화에 맞춰 성분을 재결합하며, 고주파 성분은 잡음으로 인한 학습 불안정을 줄이도록 정규화를 적용한다. 두 번째 난관은 self-attention 중심의 관계 증류가 pairwise 저차 관계에 머물러 복잡한 다중-다중 고차 상관을 전달하기 어렵다는 점이며, MORD는 하이퍼그래프 기반 hyper-attention으로 집계-분배(aggregation-and-distribution) 경로를 만들어 고차 관계를 전파하도록 설계한다.

- **Empirical Impact**: 훈련 단계에서는 RGB 프레임과 이벤트 스트림을 함께 쓰되, 추론 단계에서는 이벤트만 사용해 성능을 끌어올리는 구성이며, 검출 성능이 일관되게 향상됨을 보인다. 저주파/고주파 분해 증류와 하이퍼그래프 기반 고차 관계 증류가 결합될수록 복잡한 장면에서 이벤트 기반 검출의 정확도가 개선된다고 보고한다. 저자들은 RGB-guided 이벤트 기반 object detection 3개 데이터셋에서 SOTA 수준의 결과를 달성하며, EvDet 증류 연구가 “단일 공간·저차 관계”에서 “다중 도메인·다중 차수”로 확장될 필요성을 실증적으로 뒷받침한다고 주장한다.



### From Open Waters to Enclosed Cabins: ProteusVPR for Cross-Scene Visual Place Recognition in Maritime Perception and Cabin Inspection (https://arxiv.org/abs/2606.24234)
- **Prior Approaches**: 기존 VPR 연구는 주로 전역 디스크립터 학습과 nearest neighbor 기반 검색 성능을 키우는 데 집중해 왔다. 패치/트랜스포머 기반 재랭킹이나 의미·비전언어 단서(SAM/CLIP 등)를 더해도, 결과적으로 “정확한 위치 추정”에는 한계가 남는다. 특히 선박의 개방 데크(광량 변화·텍스처 희소)와 선실(반복 구조·시각적 모호성)처럼 도메인이 크게 바뀌는 cross-scene에서는 일반화가 쉽게 무너진다.

- **Core Contribution**: 이 논문은 선박 선실 환경에 맞춰 기존 VPR 백본 위에 얹어 쓸 수 있는 2단계 retrieval-refinement 프레임워크 ProteusVPR을 제안한다. 1단계는 임의의 standard VPR 모델로 후보를 찾고, 2단계는 검색된 참조 이미지와 쿼리 이전 두 프레임을 함께 써서 상대 변위 기반의 정밀 로컬라이제이션을 수행한다. 또한 기하-시각 추정 모듈이 반복 선실에서의 검색 모호성을 일부 완화하며 localization precision을 끌어올린다.

- **Technical Challenges**: 문제의 핵심 기술 난점은 (1) 전역 검색은 도메인 변화에 취약하고 (2) 선실의 반복 구조는 동일/유사한 전역 특징을 만들어 위치가 헷갈린다는 점이다. 이를 해결하기 위해 DINOv2 ViT-B/14로 프레임 특징을 뽑되, inter-reference attention과 query-reference attention으로 참조 정보를 정교하게 융합한다. 더 나아가 로컬 affine 좌표계에서 camera azimuth(boresight azimuth)를 인코딩하고, 기준점(세 참조 이미지)의 기하량과 함께 MLP 추정으로 예측 변위를 전역 좌표로 변환해 정밀도를 높인다.

- **Empirical Impact**: 평가는 선박에서 운영 중 수집한 8K급 360도 panoramic 기반 XHZ 데이터셋에서 수행된다. XHZ는 다층 선실·데크 전이 구간을 포함하고 query-database separation을 엄격히 적용해 까다로운 벤치마크로 설계됐다. 실험 결과 ProteusVPR는 여러 VPR 백본 전반에서 평균 mean localization error를 60% 이상 줄이며, cross-scene 해상·선실 환경에서 강건한 정밀 시각 로컬라이제이션 해법임을 보였다.



### Latent Visual States for Efficient Multimodal Reasoning (https://arxiv.org/abs/2606.24233)
- **Prior Approaches**: 기존 MLLM 비주얼 추론은 CoT를 이미지로 확장하거나(visual Chain-of-Thought) 추론 중 외부 도구를 호출해 박스/크롭/탐지 신호 같은 이산 출력을 통해 정밀 정보를 얻는 방식이 주류였다. 다만 도구 호출은 모델 내부 계산과 분리되어 end-to-end 최적화가 어려우며, 매 단계 재인코딩과 런타임 의존으로 고해상도·다중 턴에서 지연이 크게 누적된다.

- **Core Contribution**: 논문은 EVA(LatEnt Visual StAtes)로, 외부 도구 호출을 대신해 연속형 잠재 시각 표현인 Latent_slot 토큰을 모델이 natively 생성하도록 제안한다. Latent_slot은 추론 과정의 중간 ‘시각적 사고’ 역할을 하며, 텍스트 토큰과 함께 end-to-end로 학습되도록 설계됐다. 또한 EVA-230K라는 텍스트-이미지 interleaved CoT 데이터셋을 구축해 적응형(필요한 slot 수를 상황에 따라 바꿈) 학습을 뒷받침한다.

- **Technical Challenges**: Latent_slot과 텍스트를 공동 최적화할 때, Latent_slot 직후 ‘transition window’에서 정책 분포가 극단적으로 벗어나 학습 불안정(예: exploding gradients) 문제가 발생했다. 이를 해결하기 위해 D-GSPO(Decouple-GSPO)를 도입해 이산(텍스트)과 연속(잠재 표현)의 최적화를 분리하고, transition window에 더 강한 localized KL 제약을 적용하면서도 Latent_slot으로는 그래디언트가 흐르도록 구성했다.

- **Empirical Impact**: EVA는 MME-RealWorld, HR-Bench, V* 등 여러 벤치마크에서 일관된 성능 향상을 보였고, 특히 MME-Real-Lite에서 기준 모델 대비 12.9%p 개선을 보고했다. 동시에 도구 호출 기반 모델보다 고해상도에서 추론 시간이 거의 해상도에 무관하게 유지되어 8192×8192 조건에서 DeepEyes 대비 약 84.6% 속도업을 달성했다.



### FiCA: Feed-forward instant Gaussian Codec Avatars from a Single Portrait Imag (https://arxiv.org/abs/2606.24232)
Comments:
          Project page: this https URL

- **Prior Approaches**: 단일 이미지/영상에서 사실적인 3D 얼굴 아바타를 만드는 일은 관측 정보가 부족해 본질적으로 ill-posed 문제다. 기존 접근은 (1) 비디오 기반으로 얼굴 추적·다중 시점 단서를 모아 최적화로 품질을 끌어올리거나, (2) 단일 입력에 강한 prior를 의존해 생성/최적화를 수행하지만 사람별 test-time 최적화와 오프라인 추적이 필요해 접근성이 떨어진다. 또 몇몇 feed-forward 방식은 빠르지만 표현 제어나 시각 품질이 제한되거나(렌더링 제약), 시간 일관성 문제가 남는 등 trade-off가 존재한다.

- **Core Contribution**: FiCA는 단 한 장의 초상 이미지를 입력으로 받아, 3D Gaussian으로 표현되는 실시간 구동 가능한 얼굴 아바타를 즉시 생성하는 feed-forward 파이프라인을 제안한다. 핵심은 human-centric 비전 foundation model로 불완전한 UV 텍스처/기하 관측을 뽑고, diffusion 모델이 이를 완전한 텍스처·메시(메시를 proxy로)로 매핑한 뒤, feed-forward mesh refinement로 영상 공간 정렬과 ID 보존을 강화한다. 마지막으로 Universal Prior Model(UPM)이 생성된 canonical mesh를 3D Gaussians로 디코딩해 다양한 표정 파라미터에 대한 실시간 구동을 가능하게 한다.

- **Technical Challenges**: 관측이 부족한 단일 초상에서 자가가림 영역(예: 입 안, 턱-목 사이, 경계부)의 텍스처와 기하를 ‘생각해서’ 채워야 하며, 동시에 입력의 ID 정보를 흐리면 아바타가 다른 사람처럼 보인다. FiCA는 SDXL VAE 기반 latent diffusion(DiT 구조)로 CLIP 시맨틱 임베딩과 Sapiens가 예측한 부분 UV 맵을 조건화하고, conditional flow matching으로 texture와 geometry를 동시 생성하도록 학습해 불완전 관측을 완결한다. 이어 U-Net+cross-attention 기반 UV refinement 네트워크가 참조 이미지와 렌더된 메시에 대한 photometric/키포인트/마스크 손실을 통해 픽셀 정렬을 보정하며, person-specific test-time optimization 없이도 미세한 스킨 톤·의상 디테일을 유지하도록 설계됐다.

- **Empirical Impact**: 실험에서는 dome 멀티뷰 캡처와 iPhone 단안 캡처 두 데이터셋을 사용해, 학습에 없던 다양한 ID와 표정 조건에서 생성 품질을 검증한다. FiCA는 held-out 테스트 ID에 대해 생성된 3D Gaussian 아바타가 입력 디테일(예: 문신, 목걸이)을 잘 반영하고, 관측되지 않은 영역의 텍스처/기하도 그럴듯하게 보완한다. 또한 단일 이미지 기반 경쟁 방법들과 비교해 시각적·정량적으로 우수함을 보이며, 단일 초상→5초 내 feed-forward 생성 및 실시간 구동을 지향한다는 점에서 얼굴 아바타 생성의 실용적 접근성을 높인다는 의미가 있다.



### Geometry-Instructed Video Editing (https://arxiv.org/abs/2606.24225)
- **Prior Approaches**: 기존 생성 비디오 편집은 텍스트 프롬프트, 마스크, 트랙, 포인트/궤적 같은 제어로 편집을 유도하지만, 목표 객체의 “편집 전/후 3D 상태”를 명시적으로 규정하긴 어렵다. 이로 인해 모델이 의도한 3D 변환을 덜 하거나 잘못 수행하고, 그림자·반사처럼 geometry-dependent 2차 효과가 변환과 일관되게 업데이트되지 않는 문제가 잦다. 또 다른 접근은 비디오마다 3D 프록시를 재구성해 그 공간에서 편집한 뒤 보정/재렌더링하는데, 재구성 비용과 불안정성(현실 동역학/추정 오차)이 그대로 결과 신뢰도를 떨어뜨린다.

- **Core Contribution**: 이 논문은 객체 수준 geometric edit을 “편집 전(pre) 객체 상태 → 편집 후(post) 객체 상태”의 전이(transition)로 통일해 표현하는 Geometry-Instructed Video Editing 프레임워크 GIVE를 제안한다. 목표 3D 상태 변화를 명확히 하되, 완전한 per-video 3D 프록시 재구성 없이도 translation, rotation, scaling, duplication, removal, trajectory editing 같은 연산을 하나의 공통 제어 형식으로 지원한다. 편집 의도는 별도 operator 라벨이 아니라 pre/post 상태 차이로 암묵적으로 유도된다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 시점과 시간에 걸쳐 객체의 3D 상태 변화를 비모호하게 지정하고 (2) 그 변환에 종속된 그림자·반사 같은 2차 효과까지 일관되게 맞추는 것이다. GIVE는 이를 위해 두 개의 비디오 정렬 geometry 스트림(depth-box, orientation-box)으로 pre/post 상태를 간결하게 인코딩해 geometry-instruction(GI) 토큰으로 확산(diffusion) 편집 모델을 조건화한다. 또한 학습을 위한 paired supervision을 위해 graphics engine에서 object-level edit program을 실행해 controlled before/after 렌더를 대량 합성하며, 물리 기반 렌더링 하에서 2차 효과가 의도된 변환과 함께 자연스럽게 변하도록 한다.

- **Empirical Impact**: 실험 결과 GIVE는 통제된 벤치마크에서 다양한 operator에 대해 기하학적 편집 충실도와 temporal coherence를 보이며, 2차 효과가 편집 의도와 일관되게 나타나는 성능을 확인했다. in-the-wild 비디오에서도 사용자 블라인드 연구에서 가장 높은 WinRate를 기록해 사람이 체감하는 편집 신뢰성이 개선됨을 시사한다. 전반적으로 “가벼운 기하 지시만으로 일관된 객체 편집”을 학습·일반화하려는 시도에 실증적 기반을 제공하며, 기존 3D 프록시 기반의 취약성을 크게 줄인 점이 의미 있다.



### MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network (https://arxiv.org/abs/2606.24214)
- **Prior Approaches**: 기존 폐 혈관 세그멘테이션은 형태학 연산·튜뷸러 강화·리전 그로잉 같은 전통 기법부터, 수작업 특징 기반 ML, 그리고 end-to-end 딥러닝까지 이어져 왔습니다. 하지만 대부분이 voxel-wise 이진 마스크 최적화에 치우쳐 작은 가지를 놓치거나(topology discontinuity) 직경·경계 기하가 일관되지 않는 문제가 남아 있습니다. 특히 SAM 계열은 의료 도메인 적응이 부족하거나 2D 슬라이스 단위 학습에 머물면 Z축 연속성과 세부 연결성을 충분히 학습하기 어렵습니다.

- **Core Contribution**: MorVess는 형태(morphology)와 기하(geometry) 사전지식을 학습에 직접 포함하는 segmentation 프레임워크를 제안합니다. 모델은 혈관 마스크뿐 아니라 distance map(VDM)과 thickness map(VTM)을 함께 예측해 경계·중심선 일관성·직경 전이의 연속성을 명시적으로 감독합니다. 또한 SAM ViT를 2.5D 방식으로 3D 문맥에 적응시키고, global-local fusion block(GLFB)로 다중 스케일 의미와 기하 단서를 결합해 토폴로지 복원을 강화합니다.

- **Technical Challenges**: 주요 기술 난제는 희소하고 꼬불꼬불한 미세 혈관에서 경계 애매성과 직경 단절, 그리고 전역 연결성 붕괴를 voxel-wise 학습만으로는 해결하기 어렵다는 점입니다. MorVess는 (1) 경계층을 에너지 레이어로 보고 연속 미분 가능한 VDM을 구성해 sub-pixel 경계 신호를 제공하고, (2) centerline 기반 두께 전이를 통해 전역 직경 일관성을 강제하는 VTM을 설계합니다. 여기에 2.5D Adapter로 SAM의 2D 특징에 슬라이스 간 의존성을 주입하고, VDM·VTM 및 다중 레벨 feature를 GLFB에서 융합하되 동적 가중으로 샘플별로 주의를 조절하게 했습니다.

- **Empirical Impact**: 논문은 Parse2022, AIIB23, HiPas, ATM2022 등 까다로운 폐 CT 벤치마크에서 Dice, clDice, HD95 개선을 통해 특히 small-vessel 복구와 global connectivity가 유의미하게 향상됨을 보였습니다. VDM·VTM 기반 기하 감독과 SAM 적응(2.5D)·전역-지역 융합이 함께 작동하면서 세부 가지의 끊김과 기하적 불일치가 줄어든 것이 핵심 성과로 제시됩니다. 저자들은 선학습 비전 모델에 geometric intelligence를 내재화하는 접근이 정밀 혈관 분석과 임상 신뢰 구조 계량으로 확장 가능한 방향임을 보여준다고 주장합니다.



### Inclusive Interactive Collisions for Multi-View Consistent Compositional 3D Generation (https://arxiv.org/abs/2606.24206)
- **Prior Approaches**: 기존 3D 생성은 대체로 학습 기반(단일 입력을 바로 3D로 복원)과 최적화 기반(Score Distillation Sampling, SDS로 2D diffusion 사전지식을 증류)으로 나뉜다. 하지만 학습 기반은 다중 객체·복잡 상호작용 데이터 부족으로, 최적화 기반은 2D diffusion의 제어 한계와 SDS의 뷰별 최적화로 인해 cross-view 불일치(Janus) 문제가 두드러진다.

- **Core Contribution**: 이 논문은 다중 객체가 상호작용하는 compositional 3D를 만들면서도 멀티뷰 일관성을 유지하는 최적화 기반 프레임워크 I2C-3D를 제안한다. 핵심은 Inclusive Interactive Collisions(I2C)로 Gaussian primitives의 배치를 계획된 바운딩박스 내부로 강제하되 상호작용 영역에서는 자연스러운 충돌이 일어나도록 유도하는 것이다. 여기에 Multi-View Adaptive Score Distillation Sampling(MV-ASDS)을 더해, viewpoint 전반에서 prior와 레이아웃 정보를 주의(attention) 조절로 함께 증류해 뷰 간 환각을 줄인다.

- **Technical Challenges**: 다중 객체 compositionality를 생성하려면 (1) 객체 배치와 상호작용 영역의 형상/충돌이 동시에 자연스러워야 하고 (2) SDS가 뷰별로만 학습 신호를 주는 구조에서 멀티뷰 일관성이 유지돼야 한다. 논문은 Gaussian primitives의 상호작용 분포가 연결선 중점 주변에 밀집한다는 관찰을 바탕으로 상호작용 collision 제약을 설계하고, K-means로 객체별 primitives를 클러스터링해 in-box 제약과 interaction collision 제약을 함께 최적화한다. 또한 MV-ASDS에서 instance token과 spatial token의 attention map을 멀티뷰에 맞춰 적응적으로 변조해 장면 수준과 개체 수준 모두의 일관성을 강화한다.

- **Empirical Impact**: 실험은 기존 SOTA 대비 semantic alignment, spatial arrangement, geometric consistency, scene quality 전반에서 우수함을 정량·정성으로 보여준다. 특히 상호작용 영역과 멀티뷰 일관성 측면에서 경쟁 방법들이 충돌 영역이 애매하거나 Janus 문제를 겪는 반면, I2C-3D는 상호작용이 더 그럴듯하고 뷰별 텍스처가 유지되는 경향을 보인다. 추가로 progressive 3D editing(객체를 단계적으로 삽입)까지 지원하며, ablation에서도 MV-ASDS/I2C/ASDS를 제거하면 멀티뷰 불일치나 텍스처 소실이 뚜렷해 효과가 확인된다.



### Co-occurring associated retained concepts in Diffusion Unlearning (https://arxiv.org/abs/2606.24192)
Comments:
          Accepted as a poster at ICLR 2026. Code available at this https URL

- **Prior Approaches**: 확산 모델의 유해 콘텐츠 생성을 줄이기 위한 unlearning 기법이 주목받고 있다. 다만 기존 방식은 목표 개념을 지우는 과정에서 함께 등장하는 정상 개념까지 덩달아 약화시키는 경우가 많다.

- **Core Contribution**: 논문은 함께 동반되지만 보존되어야 할 정상 공존 개념을 CARE(Co-occurring Associated REtained concepts)로 정의하고, 이를 직접 측정하는 CARE score를 제안한다. 또한 목표 개념만 지우면서 CARE를 안정적으로 보호하는 ReCARE(Robust erasure for CARE) 프레임워크를 제시한다.

- **Technical Challenges**: 핵심 난제는 unlearning 중에 발생하는 ‘불필요한 동반 억제’를 정량화하고 학습에 반영해 안정적으로 상쇄하는 것이다. ReCARE는 타깃 이미지에서 추출한 benign co-occurring 토큰으로 CARE-set을 자동 구성하고, 학습 과정에서 이 어휘를 활용해 목표 개념 erasure는 강화하되 CARE 보존은 유지하도록 설계했다.

- **Empirical Impact**: Nudity, Van Gogh style, Tench object 등 다양한 타깃에 대해 실험을 수행했으며, robust concept erasure와 전체 유틸리티, CARE preservation의 균형에서 전반적으로 state-of-the-art 성능을 보였다고 보고한다. 결과적으로 unlearning의 부작용을 줄이면서도 원하는 기능만 선택적으로 제거하는 방향에 실증적 기여를 한다.



### Towards Fast and Effective Long Video Understanding of Multimodal Large Language Models via Adaptive Quasi-Gaussian Sampling (https://arxiv.org/abs/2606.24187)
Comments:
          NeurIPS 2026 submission. 15 pages, 8 figures

- **Prior Approaches**: 기존 장편 영상 이해에서는 계산·메모리 부담을 줄이기 위해 keyframe selection이 주로 쓰인다. 하지만 hard sampling 기반의 유사도(embedding) 점수 정규화는 (1) 국소 작업에 유리하되 전역(holistic) 이해에서는 시야가 좁아지고, (2) 영상-쿼리 유사도에 존재하는 noise가 선택 정확도와 적응성을 떨어뜨린다.

- **Core Contribution**: 이 논문은 프레임 선택을 Adaptive Quasi-Gaussian Sampling 문제로 재정의하고, 학습 없이 적용 가능한 AdaQ를 제안한다. Gaussian의 3-sigma(3-σ) 규칙을 응용해 전역 쿼리에는 더 넓은 3-σ 구간을, 국소 쿼리에는 더 좁은 구간을 자동으로 만들어 견고하고 유연한 샘플링을 목표로 한다.

- **Technical Challenges**: 핵심 난제는 쿼리-프레임 유사도를 그대로 확률분포로 만들면 분포가 너무 평평해져 중요 프레임을 구별하기 어렵다는 점이다. AdaQ는 예제별 similarity variance 패턴을 온도(temperature) 조절 신호로 사용해 Quasi-Gaussian 분포의 3-σ 유효 구간을 변형하고, 그 구간 밖 프레임은 soft하게 억제한 뒤 확률 가중 샘플링으로 keyframe을 뽑는다(하이퍼파라미터는 γ 하나).

- **Empirical Impact**: 네 가지 MLLM과 세 가지 embedding 모델에 대해 LongVideoBench, Video-MME, LVBench, MLVU 등에서 실험을 수행했으며, AdaQ는 기본 MLLM 및 SOTA keyframe selection을 일관되게 능가했다. 예로 Qwen3-VL-8B가 64 frames만 사용하고도 GPT4o 대비 평균 15.8% 우위를 보였고, 전역·국소 모두에서 성능 하락이 적었다. 또한 embedding 모델이 약할 때(예: CLIP)에도 강건성이 유지되며, 대부분의 작업에서 1 hyper-parameter만으로 효율적으로 장편 영상 이해 성능을 끌어올린다는 점에서 의미가 크다.



### Deep Learning Approaches for 3D Medical Scene Completion: From Geometric Modeling to Generative Paradigms (https://arxiv.org/abs/2606.24180)
- **Prior Approaches**: 3D scene completion은 RGB-D, LiDAR, multi-view 등에서 얻는 부분 관측을 바탕으로 누락 기하를 복원하는 문제로, 기존 접근은 주로 voxel 격자(예: SSCNet, ScanComplete)와 point cloud(예: PCN, TopNet) 또는 implicit neural representations(예: Occupancy Networks, DeepSDF)로 나뉘었다. 그러나 voxel은 메모리·연산이 해상도에 따라 급격히 증가하고, point cloud는 불연속성과 표면 복원 후처리 부담이 생기며, implicit은 표면 추출을 위한 대량 쿼리로 실시간 제약이 커졌다. 최근에는 transformer, diffusion 같은 생성 기반이 성능을 끌어올렸지만 계산 비용과 조건부 정합(보이는 기하를 얼마나 정확히 존중하는지)의 어려움이 남아 있었다.

- **Core Contribution**: 이 논문은 2016–2026년(최근 10년) 동안의 3D scene completion 연구를 체계적으로 정리하고, representation(예: voxels, point learning, implicit neural fields, transformer, diffusion, rendering-aware 3D Gaussian primitives) 관점의 진화 흐름을 분석한다. 또한 수많은 논문을 포함·배제 기준에 따라 선별(초기 1,847편 중 최종 267편)하고 PRISMA 가이드라인을 따라 선행연구의 커버리지를 투명하게 제시한다. 마지막으로 분야의 기여를 분류하는 taxonomy와 함께, 남은 과제와 차세대 시스템 개발을 위한 research agenda를 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) 입력의 불완전성(가림/잡음/희소성) 속에서 (2) 표현력·연산량·스케일의 균형을 맞추고 (3) 학습/평가가 실제 배치 성능을 충분히 반영하는지 검증하는 것이다. 논문은 이 난제를 representation 패러다임의 트레이드오프로 해석하며, voxel의 O(N^3) 비용 문제를 sparse convolution으로 완화하고, implicit의 대량 쿼리 문제는 효율화 필요성으로 연결한다. 나아가 diffusion의 다단계 denoising(수십~수천 스텝) 비용과 hallucination/조건부 정합 이슈, transformer의 O(N^2) 병목 같은 실무적 제약까지 함께 정리해 해결 방향을 구조화한다.

- **Empirical Impact**: 저자들은 해마다 선택된 연구 분포와 패러다임 전환(특히 최근 diffusion 및 Gaussian 기반으로의 이동, 2024 이후 voxel 감소)을 통해 연구 속도와 트렌드 변화를 보여준다. 동시에 메모리 효율(예: sparse/point), 생성 다양성(예: diffusion), 실시간 렌더링 가능성(예: 3D Gaussian splatting)처럼 서로 다른 강점이 어떻게 empirical 성과로 이어졌는지를 taxonomy 안에서 비교 가능하게 만든다. 결과적으로 이 리뷰는 3D scene completion 연구자들이 다음 연구 방향(표현-추론-배치의 연결, 평가 지표/실세계 검증, 실시간성 확보)을 더 빠르게 설계하도록 하는 지도 역할을 한다.



### Zero-Shot Test-Time Canonicalization using Out-of-Distribution Scoring (https://arxiv.org/abs/2606.24178)
- **Prior Approaches**: 회전·스케일·시어처럼 클래스는 유지하지만 기하학만 바뀌는 입력에서, 기존 pretrained 비전 모델은 쉽게 오분류되며 이를 보완하려면 아키텍처에 불변/등변성을 넣거나 데이터 증강으로 재학습하는 경우가 많았다. Canonicalization도 있었지만, 대부분 logit 기반 energy 점수와 전용(비용 큰) 탐색 절차에 의존해 점수 함수/최적화기 설계 공간이 제한됐고, 이미 정렬된 입력에선 ID 정확도가 떨어지는 문제가 남았다.

- **Core Contribution**: 이 논문은 canonicalization을 OOD detection(분포 밖 감지)으로 재구성해, 어떤 OOD score든 “변환 중 에너지를 최소화”하는 canonicalizer의 energy로 쓸 수 있도록 만든다. 또한 탐색 알고리즘 선택이 성능을 크게 좌우함을 체계적으로 실험하고, 이미 정렬된 입력에 불필요한 변환을 적용해 생기는 정확도 저하를 gated canonicalization으로 줄인다.

- **Technical Challenges**: 핵심 과제는 (1) OOD score를 에너지로 쓸 때 ‘정답 canonical form’과의 일치가 실제로 성립하는지, (2) 연속 affine 변환군에서 비볼록 최적화를 어떻게 안정적으로 수행할지, (3) 최적화가 잘못된 변환을 선택해 ID 정확도를 떨어뜨리지 않게 제어하는 것이다. 저자들은 약 20종 OOD score와 9종 탐색 알고리즘을 대규모로 비교하고, 대체로 distance 기반 score에 random search + local refinement가 강하며, OOD 임계값 기반 selection gate와 score 감소를 강제하는 acceptance gate로 필요할 때만 변환을 수행하게 했다.

- **Empirical Impact**: 다양한 벤치마크(MNIST/EMNIST, TU Berlin 스케치, ModelNet10 3D point cloud, SI-Score 회전, 그리고 ImageNet 회전 셋업)에서 OODC는 training-free 기준 중 변환 입력 정확도를 가장 높였고, logit 기반 energy는 성능 개선이 미미한 경우가 많았다. 또한 gated 메커니즘으로 already-aligned 입력에서의 정확도 하락을 대부분 복구하면서도 변환 강건성은 유지해, 기존 canonicalization의 ‘강건성-정확도 트레이드오프’를 조절 가능하게 만들었다.



### Tri-Efficient Transfer Learning for Point Cloud Videos (https://arxiv.org/abs/2606.24175)
- **Prior Approaches**: 기존 포인트클라우드 foundation model을 fine-tuning할 때는 PEFT가 파라미터 수를 줄여 주지만, 여전히 대규모 데이터 라벨링 비용과 GPU 메모리 병목이 남아 있다. PointCSA 같은 additive adapter 계열은 backbone은 얼리고 어댑터만 학습해 효율을 얻지만, 백프로퍼게이션 과정에서 중간 캐시가 차지하는 메모리 문제를 근본적으로 줄이긴 어렵다. 또한 4D(포인트클라우드 비디오)용 사전학습은 대체로 동적 데이터 수집이 부담되고, motion을 디코딩된 특징에서 직접 학습해 전이성이 떨어질 수 있다는 지적이 있다.

- **Core Contribution**: 이 논문은 “무작정 데이터만 키우는 방식”이 아니라, 기존 데이터에서 더 풍부한 supervision 신호를 끌어내는 방향으로 TriETL(tri-efficient transfer learning)를 정식화한다. 그에 맞춰 PoinTriE는 data-, parameter-, memory efficiency를 동시에 만족하도록 설계된 통합 프레임워크를 제안한다. 특히 fine-tuning 단계에서는 pretrained backbone을 동결하고, LoRA 기반의 lightweight Spatio-temporal Side Network만 업데이트해 목표 효율을 맞춘다.

- **Technical Challenges**: 핵심 난제는 (1) TriETL을 단일/이중 단계 중 어떤 정의가 성립하는지 이론적으로 다루고, (2) 메모리 사용을 줄이면서도 충분한 성능 향상을 보장하는 학습 파이프라인을 구성하는 것이다. 이를 위해 3D 비전 관점에서 scaling law와 파라미터-메모리 trade-off를 분석하고, 단일 단계에서 세 효율이 동시에 최적화되기 어렵다는 관찰로부터 two-stage 설계를 정당화한다. 또한 pretraining은 rigid transformation으로 pseudo-motion trajectories를 합성하고, Geometric-Motion Duality Network(GMD Net)에서 multimodal contrastive learning과 rigid rotation prediction, motion distribution divergence(KL)로 dense self-supervision을 생성한다. fine-tuning에서는 gradient flow masking 전략으로 LoRA 업데이트 범위를 제어해 메모리와 trainable parameter 오버헤드를 동시에 줄인다.

- **Empirical Impact**: PoinTriE는 action recognition과 semantic segmentation의 대표 벤치마크에서 state-of-the-art 성능을 보고하며, 예로 MSR-Action3D에서 94.37% 평균 정확도, Synthia 4D에서 84.11% mIoU를 달성해 기존 대비 소폭이지만 의미 있는 개선을 보인다. 이 결과는 라벨 비용이 큰 동적 포인트클라우드 환경에서도 정적 데이터 기반의 pseudo-motion 생성과 multimodal self-supervision이 전이를 강화할 수 있음을 시사한다. 더불어 PEFT가 남기기 쉬운 GPU 메모리 병목을 gradient flow masking과 side network/LoRA 배치로 완화했다는 점에서, 향후 포인트클라우드 PFMs fine-tuning 설계의 기준점(baseline) 역할을 할 것으로 기대된다.



### Spectral Evolution-Guided Token Pruning in Multimodal Large Language Models (https://arxiv.org/abs/2606.24165)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: MLLM의 속도를 높이기 위해 비전 토큰을 줄이려는 연구들이 활발하지만, 대다수는 단일 레이어의 신호(어텐션 점수, 피처 크기, 토큰 유사도)에 의존한다. 이런 방식은 Transformer가 층을 거치며 시각 표현을 저수준 디테일에서 고수준 의미로 “변환”하는 과정을 충분히 반영하지 못하며, 멀티모달 토큰 시퀀스에서 위치 편향도 발생할 수 있다. 또한 압축을 위해 토큰을 폐기하는 hard pruning이나 병합하는 token merging 모두 상황에 따라 성능 손실-효율 균형이 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 훈련 없이(inference-time) 작동하는 시각 토큰 pruning 프레임워크 CLSE(Cross-Layer Spectral Evolution)를 제안한다. CLSE는 토큰 중요도를 단일 레이어 값이 아니라, Transformer 레이어를 통과하며 토큰 표현이 주파수 영역에서 어떻게 재분배되는지(세부→의미로의 변화)를 기준으로 정량화한다. 스펙트럴 재분배가 강한 토큰은 의미적으로 활성화되어 있을 가능성이 높으므로 보존하고, 비교적 정적인 토큰은 제거해 redundancy를 줄인다.

- **Technical Challenges**: 핵심 난제는 “의미 기여”를 단일 레이어의 어텐션/크기 같은 즉시 신호로 대신할 때 생기는 불안정성과 위치 편향을 제거하는 것이다. 이를 위해 CLSE는 시각 토큰을 주파수 도메인으로 투영한 뒤 Gaussian high-pass filter로 저주파(전역 패턴)를 억제하고, 남은 band-limited 구조 에너지의 크로스-레이어 변화를 토큰별 진화 강도로 계산한다. 또한 수치적 안정성을 위한 정규화와 outlier 완화 장치를 포함해, 다운스트림 디코딩에서 신호가 흔들리지 않도록 중요도 기준을 설계했다.

- **Empirical Impact**: 실험은 이미지/비디오 QA 벤치마크 9종과 여러 MLLM(LLaVA-1.5, LLaVA-Next, Qwen2-VL 등)에 대해 수행됐으며, CLSE는 강한 토큰 감축에서도 정확도-효율의 균형이 가장 좋게 유지되는 편향이 관찰된다. 예를 들어 LLaVA-1.5-7B에서 토큰 192/128/64 보존 조건의 평균은 각각 99.4%/98.1%/94.8%로, 기존 hard pruning과 token merging 계열을 전반적으로 앞섰다. 더 나아가 Qwen2-VL-7B에서도 pruning 비율이 커질수록 흔들림이 적었고, CLSE는 FLOPs와 KV cache 메모리, 지연(latency)을 줄이면서도 경쟁력 있는 성능을 유지하는 것으로 확인됐다.



### Dual-Branch Cross-Projection Debiasing through Diffusion-based Disentanglemen (https://arxiv.org/abs/2606.24161)
- **Prior Approaches**: 기존 group-robust learning은 group label이 있을 때는 reweighting, resampling, robust optimization, contrastive learning 같은 방식으로 그룹 간 성능을 균형화한다. 라벨이 없을 때는 biased 모델을 학습한 뒤 confidence나 training dynamics 등 모델 동작에서 pseudo spurious labels를 만들어 debiasing에 사용하지만, 이 과정의 “spurious”는 의미적으로 실세계 편향 요인과 맞닿지 않을 수 있다. 또한 대부분 single-branch 설계라 target과 spurious가 같은 feature space에서 얽혀 있어 정교한 분리가 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 diffusion 기반 disentanglement을 이용해 의미적으로 정렬된 concept 표현으로 spurious 속성을 비지도/무라벨 환경에서도 찾아내는 CBCM(Confidence-guided Bias Concept Mining)을 제안한다. 또한 DCD(Dual-branch Cross-projection Debiasing)로 target/ spurious를 두 브랜치로 분리한 뒤 cross null-space projection으로 spurious 정보를 제거하면서 target에 유용한 의미는 보존하도록 학습한다. 결과적으로 “신뢰 가능한 pseudo spurious supervision”과 “피처 수준의 명시적 차단”을 함께 해결하는 통합 프레임워크 D2CP를 제시한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 group label이 없을 때 spurious factor를 의미적으로 신뢰성 있게 추출하는 문제였다. 저자들은 EncDiff를 통해 이미지의 concept tokens를 diffusion disentanglement과 cross-attention grounding으로 분해하고, 편향된 분류기 confidence가 높은 샘플의 concept prototype를 class별로 만들고, classifier 활성과의 정렬 정도로 candidate spurious concept을 점수화해 pseudo spurious label로 변환한다. 두 번째 난제는 pseudo spurious를 학습에 “원칙적으로” 주입해 entanglement을 실제로 줄이는 것이며, DCD는 두 브랜치에 대해 null-space projector를 분기별 classifier로부터 구성한 뒤 cross-projection으로 반대 브랜치의 핵심 공간을 제거하는 방식으로 해결한다.

- **Empirical Impact**: 네 가지 벤치마크에서 group-unsupervised setting 기준으로 worst-group accuracy에서 state-of-the-art 성능을 보였고, 학습 시 fine-tuning 파라미터를 최대 0.22%만 조정하는 parameter-efficient 설계의 이점도 확인됐다. 특히 기존 단일-브랜치 접근이 갖던 target/spurious 얽힘 문제를 구조적으로 완화해 minority 그룹 일반화가 개선되는 것으로 해석된다. 또한 코드가 보조자료에 공개되어 재현성과 확장 연구에 도움이 될 전망이다.



### Accelerating Multimodal Large Language Models with Prior-Corrected Token Reduction (https://arxiv.org/abs/2606.24156)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 시각 토큰 감소 방법들은 주로 텍스트-비주얼 attention 점수를 중요도로 삼아 상위 토큰을 남기는 training-free pruning을 사용합니다. FastV 같은 방식은 attention 크기가 의미적 기여를 잘 반영한다고 가정하지만, 실제로는 모델이 무지시 상태에서도 특정 시각 영역에 강하게 attention을 거는 saliency prior(모델 유도 prior)가 섞입니다. 이 때문에 instruction-conditioned 토큰이 절대 attention 값 기준에서 억눌려 탈락할 위험이 커집니다.

- **Core Contribution**: 이 논문은 attention 기반 pruning의 핵심 실패 원인으로 model-induced prior가 posterior ranking을 지배하는 “prior-dominated masking”을 지적합니다. 이를 해결하기 위해 PriorTR(Prior-Corrected Token Reduction)을 제안하며, 각 토큰의 ‘추가로 사용 가능한 정보’가 prior 대비 얼마나 늘었는지로 순위를 매깁니다. 또한 null token을 이용해 prior(무지시 분포)와 posterior(지시 조건 분포)를 분리해 단일 forward pass에서 동시에 추정합니다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) prior와 instruction에 의한 의미를 attention 안에서 분리하고 (2) 계산 중복 없이 이를 토큰 점수로 변환하는 것입니다. PriorTR은 causal self-attention 구조를 활용해 instruction을 볼 수 없는 null token의 attention을 prior로, instruction 토큰들의 attention을 task-conditioned posterior로 얻습니다. 그리고 PVI(pointwise V-information) 형태의 prior-corrected 점수로 토큰을 랭킹한 뒤, 선택된 hidden state와 KV cache만 물리적으로 남겨 이후 레이어 계산량을 실제로 줄입니다.

- **Empirical Impact**: 12개 멀티모달 벤치마크와 여러 MLLM(LLaVA 계열, Qwen3-VL 등)에서 PriorTR은 강한 training-free 기준선 대비 정확도–효율 trade-off를 일관되게 개선했습니다. 특히 토큰 예산이 매우 작은 aggressive budget 구간에서 기존 attention 절대값 기반 방식의 오판을 줄이며 정확도 격차가 확대되는 경향이 보고됩니다. Video-LLaVA에서도 pruning-only 비교에서 성능 우위를 보였고, 효율 분석에서는 64개 토큰 유지 시 prefilling latency 2.21× speedup 및 KV cache 메모리 감소가 확인되며 실제 런타임 이득으로 이어졌습니다.



### Differential Unfolding: Efficient Unfolding Reconstruction for Video Snapshot Compressive Imaging (https://arxiv.org/abs/2606.24153)
- **Prior Approaches**: Video Snapshot Compressive Imaging(비디오 스냅샷 압축 센싱, video SCI)은 고속 영상을 마스크로 압축 측정해 저속 카메라로 복원하는 역문제다. 기존 접근은 model-based, plug-and-play, end-to-end 학습으로 나뉘며, 그중 deep unfolding은 반복 최적화의 구조를 네트워크로 펼쳐 해석성과 성능을 함께 노려왔다. 그러나 대부분의 DUN은 모든 unfolding stage에 동일한(균질한) prior 구조를 반복 적용해, 수렴에 가까워질수록 feature 변화가 거의 없는데도 고비용 계산이 계속되는 비효율이 발생한다.

- **Core Contribution**: 논문은 DUN의 ‘Inter-Stage Representation Homogeneity’(스테이지 간 표현 동질성)를 계산 중복의 근본 원인으로 규명하고, 이를 해결하는 Differential Unfolding(DU) 패러다임을 제안한다. 핵심은 Differential Evolutionary Framework(DEF)로 unfolding 과정의 역할을 분리해, 주기적으로는 고파라미터 general unfolding으로 기반 feature를 만들고 이후에는 differential evolution으로 미세 갱신만 수행하게 한 것이다. 또한 Differential Representation Prior(DRP)를 통해 스테이지 간 차이를 반영하는 방식으로, 정적 상태를 다시 재구성하는 대신 변화량을 효율적으로 전파·정련한다.

- **Technical Challenges**: 가장 큰 technical challenge는 수렴 과정에서 stage-wise feature가 유사해지는 상황에서도 ‘어떤 계산은 줄이고 어떤 업데이트는 유지할지’를 구조적으로 구분하는 것이다. DEF는 periodic general stage와 가벼운 differential stage를 번갈아 배치해, 대표적인 고비용 prior를 모든 스테이지에 반복하지 않도록 설계했다. DRP는 Differential Representation Attention(DRA)으로 주의(attention) 변화의 차이를 강조하고, Differential Modulated FFN(DM-FFN)으로 중간 state를 이용한 differential gating/피처 보정을 수행해 최소 오버헤드로 스테이지 간 변화를 모델링한다.

- **Empirical Impact**: 실험에서 DU는 multiple SCI 벤치마크와 실촬영(real captured) 데이터에서 기존 SOTA를 갱신하면서도 계산량과 파라미터를 크게 줄였다. 예를 들어 DU-9stg는 DADUN 대비 50.2%의 파라미터와 25.12%의 FLOPs로 0.56 dB 성능 향상을 보였고, DU-5stg는 가장 낮은 FLOPs/파라미터 조건에서도 EfficientSCI++ 대비 0.6 dB 이상 개선됐다. 또한 ablation에서 DEF의 differential periodicity와 DRP 적용이 다른 unfolding 모델에도 비용 절감과 성능 개선을 동시에 가져온다는 점을 확인했다.



### Autonomous Video Generation with Counterfactual Controllability for Self-Evolving World Models (https://arxiv.org/abs/2606.24152)
Comments:
          5 pages, 1 figure

- **Prior Approaches**: 기존 문헌은 비디오 생성이 본질적으로 world modelling이라고 주장하며, 시각적 예측을 스케일링하면 물리적 장면과 에이전트로 자연스럽게 확장될 것이라 기대해 왔습니다. 그러나 예측이 그럴듯하더라도 어떤 변수가 조절 가능하고, 신체(embodiment) 제약을 어떤 미래가 만족하는지, 개입 후의 미래가 계속 유효한지를 검증하지 못하는 한계가 남습니다. 즉 ‘그럴듯한 비디오’와 ‘행동 가능한 미래’ 사이에는 인과적 격차(causal gap)가 존재합니다.

- **Core Contribution**: 이 논문은 video generation이 학습하는 것은 완전하게 grounded·controllable world model이 아니라, 일부적이고 암묵적인 spatiotemporal world model일 뿐이라고 정리합니다. 대신 self-evolving world model을 만들기 위한 결정적 기준을 counterfactual controllability로 제안하는데, 이는 ‘행동을 했을 때 무엇이 달라지는가’를 묻고(개입 조건), 신체 제약 하에서 미래가 살아남는지 시험한 뒤, 그 행동 지식을 다음 imagination(생성)에 되먹임하는 능력입니다. 결론적으로 autonomous video generation을 “예측 사실성”이 아니라 “반사실적 행동 가능성”을 중심 축으로 재정의합니다.

- **Technical Challenges**: 핵심 기술 과제는 단순히 미래 프레임을 예측하는 p(ot+1:t+k|o≤t)가 아니라, 개입 do(a) 조건의 미래 분포 p(τ|s_t, do(a_{t:t+k}), e)를 신체 제약 e까지 포함해 생성·평가·추출해야 한다는 점입니다. 이를 위해 4단계 closed-loop를 설계합니다: Generation(다양한 반사실 미래 제안)→Binding(신체·제어·에너지 제약에 결속)→Verification(관측/동역학/embodiment shift에서 드리프트 분기 검증·보정)→Distillation(살아남은 분기를 압축해 의사결정 변수로 변환)합니다. 또한 novelty·consistency·Out-of-Distribution·efficiency를 결합해, 어느 단계라도 실패하면 전체 controllability 점수가 크게 낮아지는 곱 형태의 평가 관점을 둡니다.

- **Empirical Impact**: 논문은 드론·매니퓰레이션 testbed를 예로 들어, 바람/센서 장애/배터리/접촉 같은 ‘희귀하지만 배치에서 치명적인 반사실’ 케이스를 측정해야 한다고 강조합니다. 이러한 설정에서 모델은 단지 비디오를 그럴듯하게 만드는 것을 넘어, 불가능하거나 안전하지 않은 분기를 reject하고 shift 하에서 재계획하며, 위험·도달 가능성·회복 가능성을 가진 compact한 의사결정 변수로 distil되어야 합니다. 제시된 counterfactual controllability 중심의 평가 프레임은 비디오 fidelity를 넘어 embodied 의사결정 성능과 안전성을 직접 개선하는 방향의 분야 표준을 제안한다는 의미가 있습니다.



### Geometry-Aware Style Transfer in 3D Gaussian Splatting (https://arxiv.org/abs/2606.24144)
Comments:
          14 pages, 7 figures, accepted at ECCV 2026

- **Prior Approaches**: 기존 3DGS 기반 스타일 전이는 주로 색(appearance) 통일에 집중해 왔고, 기하(geometry) 적응은 보수적으로 다뤄왔다. NeRF 계열은 기하를 반영하지만 최적화 비용과 렌더링 속도 문제가 커 실용성이 떨어진다는 한계가 있었다. 3DGS 계열에서도 geometry를 거의 고정하거나(색만 업데이트), 제한적 분할/정규화로만 변형을 허용해 ‘구조를 매체로 하는’ 스타일링 표현이 부족했다.

- **Core Contribution**: 이 논문은 3D Gaussian splatting(3DGS)에서 스타일 이미지의 외형뿐 아니라 기하 구조까지 함께 옮기는 geometry-aware style transfer 프레임워크를 제안한다. 핵심은 색 파라미터와 기하 파라미터를 번갈아 업데이트하는 decoupled optimization으로, 색-기하 간 최적화 간섭을 줄여 장면 전체에서 일관된 구조 변형을 유도하는 것이다. 또한 geometry-aware contrastive feature matching(GCFM)으로 RGB, depth, edge 신호를 함께 정렬해 스타일의 구조적 특징을 가우시안 원시(primitive)에 전달한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 3DGS에서 색 변화보다 기하 변화가 훨씬 민감해, 동시에 학습하면 쉽게 구조가 망가지거나(과도한 색 고정화로 geometry 손상) 아티팩트가 발생한다는 점이다. 논문은 이를 해결하기 위해 외부 cycle마다 두 단계(먼저 color, 그다음 geometry)로 나눠 고정/갱신을 교대하는 형태로 안정성을 확보한다. GCFM은 VGG 특징을 색·depth·edge 3모달로 추출한 뒤 채널 결합한 joint feature에서, 각 렌더 위치의 positive(가장 가까운 스타일 특징)와 negative(가장 먼 특징)를 함께 사용해 contrastive objective로 구조 정합을 강화한다.

- **Empirical Impact**: LLFF, Tanks & Temples, MipNeRF-360의 8개 장면과 9개 스타일 조합(총 72개 scene–style)에서 정성/정량 모두 기존 3DGS 스타일 전이 방법을 능가하는 결과를 보였다고 보고한다. 특히 꽃/뿔 사례에서 단순 색칠이 아니라 목표 스타일의 ‘블록형 구조’나 ‘입자형 표면’처럼 형태 자체가 변형되면서도 depth와 경계가 비교적 명확히 유지되는 경향이 관찰된다. NeRF보다 실시간에 가까운 3DGS의 장점을 살리면서도 geometry를 스타일의 표현 수단으로 끌어올렸다는 점에서, 3D 일관 스타일링의 실용적 확장에 의미가 있다.



### Sat2City v2: Native 3D City Asset Generation from a Single Satellite Imag (https://arxiv.org/abs/2606.24138)
- **Prior Approaches**: 기존 위성-지상 생성은 주로 street-view 이미지나 동영상 뷰 합성을 목표로 하며, 3D는 렌더링용 프록시(implicit/NeRF/가우시안 등) 형태로 학습되는 경우가 많다. 그 결과 geometry는 노이즈·단절·편집 난이도 문제를 겪고, 위성 입력이 텍스처까지 직접 제어하기도 어렵다. Sat2City는 명시적 3D(스파스 보셀+라텐트 확산)로 한 단계 나아갔지만, (1) height map 중심이라 appearance 정보가 소실되고 (2) 합성 데이터·임의 appearance에 의존하며 (3) 태스크 전용 VAE가 실제 복원 메시에 대한 확장성에 한계가 있었다.

- **Core Contribution**: Sat2City v2는 단일 위성 이미지로 지오레퍼런스된 ‘명시적 텍스처드 시티 메쉬 자산’을 생성하되, appearance를 입력 위성에 따라 제어 가능하게 만드는 프레임워크를 제안한다. 이를 위해 TRELLIS.2의 pretrained native structured-latent 3D foundation model을 약한 정렬의 위성-메쉬 쌍에 맞춰 지오스페이셜하게 파인튜닝하고, 생성된 shape를 앵커로 텍스처(위성 조건) 합성을 결합한다. 또한 합성 height-map 조건을 버리고, 실제 위성 이미지와 텍스처드 3D 메쉬가 같은 지리 구역에서 수집된 매칭 쌍을 학습에 사용한다.

- **Technical Challenges**: 핵심 기술 난제는 (i) 단일 위성 크롭과 Google Earth 텍스처드 메쉬 간 ‘약한 정렬’로 인해 발생하는 슈퍼비전 잡음, (ii) 그 잡음을 스파스 보셀 계층/과제별 VAE가 그대로 학습해 분절·아티팩트를 증폭시키는 문제, (iii) 위성 픽셀 정보를 보셀에 억지로 투영(pseudo-projection)해 브리틀한 결합을 만들 경우 나타나는 오류 누적이다. Sat2City v2는 이 한계를 네이티브 3D 라텐트 공간(사전학습된 자산 manifold)에서 mesh를 인코딩하고, satellite-conditioned geometry flow를 약한 정렬에서도 안정적으로 파인튜닝하며, 디코딩된 형상으로 텍스처 고정을 수행하는 방식으로 완화한다. 즉 geometry-to-appearance cascade는 유지하되, appearance 단계에서 위성 조건이 직접 생성 제어에 반영되도록 학습 구조를 재설계한다.

- **Empirical Impact**: 저자들은 9개 도시의 24개 지역에서 위성-텍스처드 메쉬 쌍 16,241개(학습/테스트 분리 포함)를 구축해 실세계 스케일 학습 가능성을 입증한다. 실험 결과 Sat2City v2는 metric-scale DSM 재구성과 geometry/appearance 생성 벤치마크를 함께 평가할 때, 비교된 베이스라인 중 전체 성능이 가장 높다. 렌더링 지향 3D 프록시에서 자산급 텍스처드 메쉬로의 전환을 실증했으며, 지리적으로 매칭된 위성-메쉬 페어를 이 태스크 용도로 수집한 데이터 세트 제공은 분야에서 중요한 선행 관측으로 평가된다.



### Bengal-HP_RU: A Dataset of Bengal People For Head Pose Estimation (https://arxiv.org/abs/2606.24122)
- **Prior Approaches**: 기존 head pose dataset들은 규모와 다양성을 제공하더라도 서구권 또는 동아시아 출신 위주라, 남아시아(특히 방글라데시/벵골어권) 인구가 크게 부족했다. 이런 비대표성은 얼굴 기하/피부 톤/의복·액세서리 차이 때문에 underrepresented demographics에서 일반화 성능이 떨어질 수 있다는 한계로 이어진다. 또한 많은 데이터는 실환경보다 실험실 촬영 조건에 치우친 경우가 있어 배경 잡음·부분 가림·조명 변화에 대한 강건성이 제한됐다.

- **Core Contribution**: 이 논문은 벵골어권 피험자를 중심으로 한 최초의 공개 head pose dataset인 Bengal-HP_RU를 제안한다. 총 12,894장의 머리 이미지에 대해 연속형 yaw, pitch, roll 값을 라벨링했으며, Wikimedia Commons(자유 라이선스)에서 수집해 in-the-wild 조건을 반영한다. 라벨 생성은 자동 기하 추정 후 수동 보정의 2단계를 통해 연속 각도 레이블의 품질을 높이는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 난제는 자동 각도 추정(solvePnP 기반)이 극단적 자세나 경계/비정상 포즈에서 흔들릴 수 있다는 점이다. 이를 위해 MediaPipe FaceMesh의 478 랜드마크 중 6점을 고정하고 solvePnP로 초기 Euler 각을 계산한 뒤, 커스텀 어노테이션 툴에서 이미지와 자동 각도를 함께 제시하여 모든 샘플을 수동으로 미세 조정했다. 또한 데이터 오염을 막기 위해 Wikimedia uploader ID 기준으로 train/test를 분리하고, 다중 얼굴 검출이 있는 경우 비대상 얼굴을 마스킹해 레이블 모호성을 줄였다.

- **Empirical Impact**: Bengal-HP_RU는 296명의 서로 다른 업로더에서 10,494(train)와 2,400(test) cropped head 이미지를 제공하며, 연령·성별·가림(occlusion)·조명·배경의 다양성이 크다고 보고한다. 기존 데이터의 인종/맥락 편향 문제를 직접 겨냥해, robust head pose estimation 연구에서 평가 범위를 넓히는 실용적 벤치마크 역할을 기대할 수 있다. 공개 데이터셋으로 제공됨에 따라, 특정 인구 집단에 대한 일반화와 공정성 관련 실험에도 활용 가치가 크다.



### Flood Mapping from RGB imagery using a Vision Foundation Mod (https://arxiv.org/abs/2606.24120)
- **Prior Approaches**: 드론/항공 RGB로 침수 범위를 빠르게 산출하려면 보통 물 분할 deep learning이 쓰인다. CNN(DeepLabV3+, UNet++)과 작은 Vision Transformer(SegFormer-b5) 기반 접근은 이벤트별로 학습해야 성능이 잘 나오지만, 새 침수 사건으로 장면(domain)이 바뀌면 적응을 위한 데이터가 크게 필요하다. 또한 기존 Earth observation foundation model(EOFMs)은 주로 위성 센서 벤치마크에서 검증돼, 단일 시점의 고해상도 RGB(비위성)에서의 데이터 효율·전이·적응 속도는 체계적으로 분석되지 않았다.

- **Core Contribution**: 이 논문은 위성에서 사전학습된 Prithvi-EO-2.0을 항공 RGB 침수 분할에 맞춰 UPerNet 디코더를 결합한 Prithvi-2.0-UPN으로 파인튜닝한다. BlessemFlood21과 NeuenahrFlood 두 데이터셋에서 이벤트 내부(in-domain) 학습 성능을 기준 모델들과 비교하고, 이후 다른 침수 사건 간 전이를 zero-shot과 소량 파인튜닝 시나리오로 평가한다. 특히 새 사건(NeuenahrFlood)에서 라벨이 거의 없을 때도 유의미한 분할 성능이 나오는지, 그리고 소량 라벨 추가 시 얼마나 빨리 성능이 복원되는지를 보여준다.

- **Technical Challenges**: 핵심 난제는 위성 사전학습이지만 실제 입력은 단일 시점의 센티미터급 항공 RGB라서 공간 해상도·시점(geometry)·방사 특성이 크게 달라진다는 점이다. 저자들은 Prithvi-EO-2.0(ViT 인코더)을 RGB 512×512 타일에 대해 엔드투엔드 fine-tuning하고, 이진 물/비물 분할을 위해 UPerNet 디코더와 binary focal loss로 학습을 수행한다. 또한 NeuenahrFlood의 소량 학습(128~2,048 타일)으로 성능이 스케일업되는 양상을 실험 설계로 확인해 적응이 언제부터 효과가 나는지 파악했다.

- **Empirical Impact**: 이벤트 내부 학습에서는 Prithvi-2.0-UPN이 BlessemFlood21과 NeuenahrFlood에서 최상위권 성능을 보이며, 특히 pixel accuracy에서 강점을 나타낸다. zero-shot 전이(BlessemFlood21 학습 후 NeuenahrFlood 테스트)에서도 적절한 decision threshold를 쓰면 Dice가 약 62%까지 도달해 기존 baseline보다 나은 편의 성능이 관측된다. 더 나아가 NeuenahrFlood 소량 라벨 파인튜닝에서는 Prithvi-2.0-UPN이 가장 빠르게 개선해 약 2,048 타일(전체 학습의 9.5%) 수준에서 Dice 90%+에 근접하며, transfer capability가 실증된 것으로 해석된다.



### An LMM for Precisely Grounding Elements in Documents (https://arxiv.org/abs/2606.24118)
- **Prior Approaches**: 기존 LMM의 문서 이해는 생성형 접근이 늘었지만, 텍스트 밀도가 높은 문서에서 정밀 grounding(단어·구·줄·문단 등 영역 좌표)이 자주 어긋나 결과 신뢰도를 떨어뜨린다. 시각적 근거를 reasoning에 포함시키는 visual grounded reasoning 연구도 주로 자연 이미지나 토큰/분리된 전처리 중심이라, 문서 특유의 레이아웃 복잡성과 작은 증거 영역을 안정적으로 다루기 어렵다.

- **Core Contribution**: 이 논문은 문서 요소를 정밀하게 찾도록 설계된 LMM PreciseDoc과, grounding을 추론 과정에 결합한 PreciseDoc-Reasoner를 제안한다. 두 종류의 문서 생성 파이프라인(LaTeX 기반 PDF 대량 생성, 스캔/손글씨 효과를 반영한 synthetic handwritten 문서 생성)으로 좌표 메타데이터가 짝지어진 어려운 학습 데이터를 구축한다. 또한 cold-start SFT 후, IoU 기반 보상과 Hungarian Algorithm 매칭, 길이 패널티를 결합해 grounding과 reasoning을 동시에 강화하는 RL 학습 체계를 마련한다.

- **Technical Challenges**: 핵심 난제는 (1) 문서에서 작은 증거를 정확히 좌표로 지정해야 하는데, 대량 생성이 가능한 고품질 데이터가 부족하다는 점과 (2) 모델이 중간 bounding box를 근거답게 생성하도록 reward를 설계해야 한다는 점이다. 저자들은 LaTeX 컴파일 단계에서 좌표를 직접 기록하고 OCR로 검증하며, 손글씨 문서는 NIST 기반 필기체 조합과 값 삽입 후 tight bounding box를 재계산해 현실감을 높였다. RL에서는 many-to-many 평균 IoU로 인한 metric inflation을 줄이기 위해 prediction-gt를 one-to-one으로 Hungarian Algorithm 매칭하고, 미매칭/중복 생성을 억제하는 length penalty로 잔여 박스에 불이익을 준다.

- **Empirical Impact**: 실험은 문서 텍스트 grounding과 문서 이해(증거 영역 포함)를 함께 평가하며, PreciseDoc은 DocLocal4K에서 전반적인 미세 단위 grounding 성능을 크게 개선했다. PreciseDoc-Reasoner는 TRIG와 BBox-DocVQA에서 evidence grounding 정확도와 추론 답변 정확도 모두에서 경쟁 모델 대비 우위를 보이며, 특히 차트 하위셋에서 큰 향상이 보고된다. 또한 DocVQA·ChartQA·TextVQA 등 기존 문서 이해 벤치마크에서도 준수한 성능을 보여, 문서 전용 grounding 학습이 실제 이해로 연결된다는 점을 실증한다.



### A Benchmark for Hallucination Detection in VLMs for Gastrointestinal Endoscopy (https://arxiv.org/abs/2606.24115)
Comments:
          Accepted at the Medical Image Understanding and Analysis (MIUA) 2026 conference

- **Prior Approaches**: 기존 의료 VLM 환각 탐지는 주로 방사선 데이터셋(MIMIC-CXR, VQA-RAD 등)에서 검증되어, 위장관 내시경(GI) 환경으로의 일반화가 불명확했다. 또한 방법들은 모델 내부 접근(white-box), 확률·엔트로피 등 토큰 신호(gray-box), 샘플 일관성 같은 출력 기반(black-box)으로 나뉘며, 접근성·성능·비용 간 트레이드오프가 커서 실제 배포 판단이 어려웠다. 특히 GI처럼 이미지 품질과 시각 패턴이 이질적인 도메인에서는 기존 성능이 유지되는지 확인이 부족했다.

- **Core Contribution**: 이 논문은 GI 내시경 VQA 벤치마크인 Gut-VLM 데이터셋에서 환각 탐지 9개 방법을 5개 VLM에 대해 동일한 실험 파이프라인으로 체계적으로 비교했다. radiology 중심의 선행 검증 결과가 GI로 잘 이전되지 않으며, 의료·비의료 전반 VLM에서 특히 black-box/gray-box가 거의 무작위 수준으로 붕괴할 수 있음을 보여준다. 동시에 white-box hidden-state 접근 기반 방법인 ReXTrust가 모든 VLM에서 가장 높은 AUC를 기록하며, 최강 비(非) white-box 대비 통계적으로 유의미한 향상을 달성했다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) GI 내시경 도메인 이동(domain shift) 하에서 환각 탐지 신호가 유지되는지, (2) 다양한 접근 수준(출력·토큰 확률·hidden state)에서 어떤 신호가 안정적으로 환각을 드러내는지였다. 저자들은 GREEN을 이용한 Gut-VLM 기반 라벨링으로 9개 방법을 동일한 입력·라벨 체계에서 공정 비교하고, gray-box의 토큰 확률/엔트로피와 white-box의 중간 레이어 hidden state를 통해 탐지 신호의 차이를 분석했다. 더 나아가 일관성 기반과 불확실성 기반을 동시에 무너뜨리는 failure mode인 confident confabulation을 찾아 그 구조적 원인을 제시했다.

- **Empirical Impact**: 실험 결과 ReXTrust는 5개 VLM 모두에서 최고 AUC를 보였고, 예시로 MedGemma-4B에서 AUC 93.0까지 도달했다. white-box는 평균적으로 약 19.5 AUC 포인트의 일관된 우위를 가지며, LLaVA-v1.6-7B처럼 black-box와 clustering 계열 gray-box가 near-chance로 붕괴하는 경우에도 강한 성능을 유지했다. 또한 black-box 중에서는 SelfCheckGPT-NLI가 가장 신뢰 가능한 대안으로 나타났지만, 짧고 자신감 있는 오답을 반복하는 모델(Lingshu-32B)에서는 confident confabulation 때문에 탐지가 실패할 수 있음이 확인됐다.



### DramaDirector: Geometry-Guided Short Drama Generation (https://arxiv.org/abs/2606.24107)
Comments:
          20 pages, 17 figures, 6 tables. Code is available at this https URL

- **Prior Approaches**: 기존 plot-to-video·narrative video 생성은 대개 스크립트/샷을 텍스트로 정교화하거나(SkyScript 등), 멀티에이전트로 플래닝을 반복하거나(MovieAgent, GENMAC), 시각 요소를 조건으로 합성해 일관성을 노리는 방식이 주류였다. 하지만 LLM이 만든 텍스트 중심 스토리보드는 short-drama의 촬영 문법(빠른 컷 리듬, 대사 주도의 프레이밍 전환)을 충분히 반영하기 어렵고, 컷을 규정하는 공간 기하(시선·가림·깊이 관계)를 거의 담지 못한다.

- **Core Contribution**: DramaDirector는 전역 plot과 국소 context를 ‘멀티샷 스토리보드’로 만들되, 각 샷을 static visual condition과 dynamic narrative condition으로 분리해 렌더링 단계에 직접 연결한다. 특히 실제 short-drama 샷 갤러리에서 depth·pose로 인덱싱된 기하 선행정보를 retrieval해, 텍스트가 전달하지 못한 공간 제약을 첫 프레임 생성과 image-to-video 합성에 주입한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) LLM 플래너가 short-drama 스키마(카메라 그라머·샷 전개 규칙)를 따르도록 만드는 것과 (2) 기하 정렬이 텍스트-비디오 정렬 보상으로부터 복구되지 않는다는 점이다. 논문은 schema-constrained SFT로 플래너를 단단히 초기화하고, depth·pose 기반 텍스트-시각 정렬 reward를 학습한 뒤 GRPO로 보상 최적화를 수행해 ‘기하가 실제 샷에 맞는’ 스토리보드를 생성하도록 폐루프(계획-검색-보상)를 구성한다.

- **Empirical Impact**: 또한 35편 실사 드라마에서 2.8K episodes·81K shots를 구축한 DramaBoard를 제안하고, 스토리보드 내러티브 품질부터 shot-level instruction following과 비디오 고유 품질까지 다차원 평가를 제공한다. 실험 결과 DramaDirector는 주요 베이스라인 대비 faithfulness·consistency·controllability에서 향상되며, ablation과 정렬 모델의 검색 판별 성능이 depth·pose 기하 설계의 기여를 뒷받침한다.



### Beyond Bayer: Task-Optimal Sensor Co-Design for Robust Autonomous-Driving Segmentation (https://arxiv.org/abs/2606.24096)
- **Prior Approaches**: 자율주행에서의 견고한 인지는 보통 더 큰 백본, foundation model, 그리고 다중 에이전트 협력 융합을 키우는 방식으로 성능을 끌어올려 왔다. 반면 카메라 자체가 무엇을 측정(센서 설계)해야 하는지는 상대적으로 상류 단계에서 체계적으로 다루지 못했다.

- **Core Contribution**: 이 논문은 카메라의 센서 측 자유도 중 dense prediction(예: 분할/세그멘테이션)에 실제로 유익한 요소가 무엇인지, differentiable RAW-to-task 파이프라인으로 분해해 제안한다. 특히 spectral colour-filter-array(CFA) 가중치 학습이 가장 큰 레버임을 밝혀, 고정 카메라 대비 mIoU를 각각 KITTI-360에서 +0.017, ACDC에서 +0.023 개선한다.

- **Technical Challenges**: 센서 설계를 학습하려면 RAW 입력을 작업까지 거치되 미분 가능해야 하고, 동시에 CFA/광학/잡음 같은 설계 자유도를 공정하게 비교할 수 있어야 한다. 연구진은 RAW-to-task를 미분 가능하게 구성해 센서 개입의 정보 흐름을 계량했고, data-processing inequality에 따라 point-spread-function(PSF, optics) co-design이 순손해가 되는 이유를 설명하며, noise 공동 최적화 효과는 미미하다고 본다.

- **Empirical Impact**: 결과는 모델 크기를 키우거나 협력 융합을 하더라도 다운스트림이 회복할 수 있는 태스크 정보에 상한이 있다는 점을 실증적으로 뒷받침한다. 또한 ACDC의 fog/night/rain/snow에서도 개선의 견고함을 확인하며, 결론적으로 2x2 CFA 가중치만 학습하고 PSF는 identity로 유지하는 간단한 레시피를 제시한다.



### Universal Guideline-Driven Image Clustering via a Hybrid LLM Agen (https://arxiv.org/abs/2606.24094)
Comments:
          CVPR 2026

- **Prior Approaches**: 기존 이미지 클러스터링은 임베딩과 거리 기반(K-Means, DBSCAN 등)에 의존해 의미 이해가 약했다. 또 딥 클러스터링과 텍스트 기반 방법은 대체로 단일 기준, 미리 정해진 클러스터 수, 또는 새로운 기준 추가 시 task-specific 학습이 필요해 시나리오 확장이 어렵다. LLM을 쓰는 접근도 정확도는 높을 수 있으나 모든 클러스터 쌍을 비교하는 식으로 계산 비용이 커 실사용 제약이 컸다.

- **Core Contribution**: 이 논문은 자연어 텍스트 가이드라인으로 다양한 클러스터링 시나리오를 통합하는 “Guideline-Driven Image Clustering Agent”를 제안한다. 핵심은 학습 없이 가이드라인을 반영한 임베딩을 만들고(Generative Concept Proxy Modeling, GCPM), 필요한 경우에만 LLM 추론을 선택적으로 수행해(LLM Traversal + MST) 복합·추상 기준까지 처리하는 것이다. 결과적으로 일반-세부, 전역-국소, balanced-long-tail 조건을 단일 프레임워크로 일반화한다.

- **Technical Challenges**: 가이드라인을 이미지 임베딩에 반영할 때, 복합 속성의 상호작용을 분리하지 못하거나 시각적으로 두드러지지만 무관한 특징이 중요한 속성을 압도하는 문제가 생긴다. 논문은 MLLM으로 ‘concept proxy’ 캡션을 추출해 속성을 텍스트로 명시적으로 분리한 뒤 instruction-aware embedding에 반영하는 방식으로 해결한다. 또한 LLM 병합 판단의 O(M^2) 쌍 비교 비용을 줄이기 위해 HDBSCAN의 초기 소클러스터를 만들고, Ward distance로 MST를 구성한 뒤 MST 순서대로 LLM 병합 결정을 요청하며 캐싱으로 중복 호출을 줄인다.

- **Empirical Impact**: STL-10, ImageNet-10, CIFAR-10 등 일반/다중/세부 클러스터링과, e-commerce의 long-tail을 반영한 ABO-LC까지 4종 작업에서 학습이 필요한 기존 방법들을 일관되게 능가하는 성능을 보였다. 특히 long-tail에서는 HDBSCAN+MST 기반 LLM 병합이 클러스터 수를 몰라도 더 유리했으며, 정밀도는 약간만 희생하고 리콜을 크게 개선하는 경향이 관찰된다. 런타임은 LLM 호출 횟수를 기준으로 비교했을 때 MST Traversal이 쌍 전수 비교 대비 효율을 크게 줄였고, 하이브리드(임베딩 기반 + 선택적 LLM) 설계의 실용성을 입증했다.



### Progressive Pixel-Neighborhood Deformable Cross-Attention for Multispectral Object Detection (https://arxiv.org/abs/2606.24092)
Comments:
          Accepted by Sensors

- **Prior Approaches**: 기존 멀티스펙트럼 객체 탐지는 CNN 기반의 국소 융합에서 출발해, 이후 Transformer로 전환되며 전역 cross-attention을 통해 RGB(TIS)와 TIR의 장거리 상호작용을 강화해왔다. ICAFusion처럼 iterative cross-attention도 제안됐지만, 표준 전역 attention은 해상도가 커질수록 계산·메모리가 제곱으로 증가해 임베디드 배치에 제약이 크다. 또한 VIS와 TIR은 센서 간 파라랙스·캘리브레이션 잔차 등으로 weak misalignment이 발생하는데, 전역 매칭은 배경의 간섭까지 함께 끌어들여 ghosting과 위치 편향을 키울 수 있다.

- **Core Contribution**: 이 논문은 weakly aligned 멀티스펙트럼 검출을 위해 PNAFusion(Pixel-Neighborhood Progressive Deformable Cross-Attention)을 제안한다. 핵심은 (1) 정해진 k×k 픽셀 이웃 안에서만 cross-modal attention을 수행하는 Pixel-Neighborhood Cross-Attention(PNCA)과, (2) 로컬에서 비선형 대응을 보정하는 Adaptive Deformable Alignment(ADA)를 결합해 정렬 정확도와 효율을 동시에 노린다는 점이다. 나아가 iterative feedback으로 정렬을 점진적으로 재정제해 잔여 misalignment를 줄인다.

- **Technical Challenges**: PNAFusion이 직면한 첫 난제는 non-linear spatial offset 하에서 대응 불일치를 줄이면서도 cross-attention의 계산 폭발을 피하는 것이다. ADA는 key-value 쪽에 대해 픽셀 단위 offset을 학습하고 deformable bilinear sampling으로 로컬 정렬을 선행한 뒤, PNCA가 정렬된 key-value를 k×k 이웃 내에서만 Query–Key–Value로 상호작용하도록 제한한다. 두 번째 난제는 단일 정렬 단계만으로는 잔여 오차가 남을 수 있다는 점인데, 이를 해결하기 위해 dual-stream PNDCA 블록을 반복 적용해 정렬과 상호작용을 단계적으로 보정한다.

- **Empirical Impact**: FLIR, M3FD, DroneVehicle에서 YOLOv5 기반 평가 시 각각 84.2, 90.5, 85.5 mAP@0.5를 달성하고, Co-DETR로 전이하면 FLIR 86.8 mAP@0.5, M3FD 90.8 mAP@0.5까지 확장된다. 효율 측면에서도 ICAFusion 대비 할당 GPU 메모리 33.0% 감소, 이론 FLOPs 194.8G→156.4G로 연산량을 줄이면서도 deformable sampling과 반복 정제에 따른 latency 증가는 분석해 투명하게 다뤘다. 결과적으로 전역 cross-attention의 메모리/배경 간섭 문제를 로컬 이웃+비선형 정렬로 우회하는 설계가 멀티스펙트럼 탐지의 실사용 가능성을 높였다는 점에서 의미가 크다.



### End-to-End Radar and Communication Modulation Recognition with Neuromorphic Computing (https://arxiv.org/abs/2606.24075)
- **Prior Approaches**: 기존 end-to-end AMR 딥러닝은 raw IQ를 그대로 넣어 정확도를 끌어올렸지만, 대규모 파라미터와 dense MAC 연산 때문에 계산량·지연·전력 문제가 커 자원제한 플랫폼에 불리했습니다. SNN 기반 접근은 neuromorphic chip에서 spike-driven 희소 연산으로 효율을 노렸지만, Poisson/temporal 같은 static spike encoder로 인한 정보 손실과 장기 의존성 포착 한계로 정확도-전력 균형이 어려웠습니다.

- **Core Contribution**: 논문은 EMRFormer라는 end-to-end spiking Transformer 기반 AMR 구조를 제안해, 원시 IQ를 직접 입력으로 받아 신호-카테고리 인식을 수행합니다. 핵심은 adaptive spike encoder로 static 인코딩의 정보 손실을 완화하고, Integer Leaky Integrate-and-Fire(ILIF) 뉴런으로 SNN 표현력을 키우는 것입니다. 여기에 Spike-Separable Convolutional Neural Network(SSCNN)를 SpikeFormer(Spike-driven transformer)에 결합해 단기·장기 시간 특징을 동시에 학습합니다.

- **Technical Challenges**: 문제는 (1) 연속 IQ를 spike로 변환하는 과정에서 정보가 깎이고, (2) transformer가 필요한 장기 의존성 학습을 SNN의 event-driven 특성 안에서 효율적으로 구현해야 한다는 점입니다. 해결책으로 ILIF의 multi-spike(정수 방출)와 virtual time steps를 활용해 neuromorphic 하드웨어 친화적인 sparse 연산을 유지하고, SSCNN-기반 spike feature 추출 후 SpikeFormer attention으로 장거리 의존성을 보완합니다. 또한 encoder 전처리를 줄여 하드웨어 연산/저장 부담을 낮추고 저SNR 환경에서도 특징이 유지되도록 설계했습니다.

- **Empirical Impact**: 여러 대표 AMR 데이터셋(RML2016.10a/b, RML2018.01a, DeepRadar2022)에서 EMRFormer는 모든 baselines 대비 정확도를 상회하며, 특히 저SNR(<0 dB)에서 성능 우위를 유지했습니다. 저전력 측면에서도 이론 에너지 소비를 90% 이상 절감하고, KA200 neuromorphic chip 실측에서는 RTX 3090/Orin NX 대비 최대 5배 전력 절감(배치 1 기준)을 보였다고 보고합니다. ablation 결과에서도 adaptive encoder, ILIF, Positional Encoding, SNN 인코딩 방식 변경이 성능을 크게 흔들며 전체 설계가 저SNR 강건성과 희소 특징 학습에 기여함을 확인했습니다.



### Fabric Image Demoiréing Benchmark from Synthesis to Restoration (https://arxiv.org/abs/2606.24072)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 moiré 제거 연구는 주로 display–camera 조합의 모니터/스크린 격자에서 생기는 moiré를 다뤘고, 주파수 대역 분리나 구조적 priors, 워브렛/밴드패스 모델링 등으로 복원을 시도해왔다. 그러나 fabric은 촘촘하고 반주기적인 직물 weaves로 인해 모사(샘플링)로 생긴 aliasing이 원래 텍스처의 분포와 강하게 겹쳐 신호와 잡음이 분광적으로 얽혀 복원이 본질적으로 더 ill-posed다. 또한 pixel-level 정렬이 어긋나면 학습이 불안정해져 성능이 급락하는데, 직물 장면은 비강체 변형과 pose 변화 때문에 대규모로 완벽한 GT 페어를 만들기 어렵다.

- **Core Contribution**: 논문은 fabric image demoiréing을 위한 최초의 포괄적 벤치마크를 제안한다. PRISM(Physics-based Residual Injection for Synthetic Moiré)은 물리 기반 이미징 체인을 이용하되 residual injection으로 텍스처의 진짜 디테일을 보존하면서 pixel-wise alignment가 보장되는 페어(16,050개, 멀티레졸루션)를 합성해 학습을 가능하게 한다. 더불어 PRISM 벤치마크를 기반으로 FaDeNet을 설계해, 저주파 base와 고주파 detail을 마스크로 선택 보정하고 주기/방향성 특징을 주파수-공간에서 동시에 포착하도록 했다.

- **Technical Challenges**: 핵심 기술 난관은 (1) 스크린 기반 학습 데이터로는 전이되기 어려운 domain gap과 (2) 직물의 비선형/비강체 변형으로 인해 pixel-aligned supervised pair를 대규모로 확보하기 어렵다는 점이다. PRISM은 보기 각도와 거리 같은 기하 변수를 무작위화하고, RGGB CFA 모자이크/노이즈/데모자이킹까지 포함한 ISP 유사 파이프라인을 거쳐 현실적인 색수차 무늬와 구조 왜곡을 만들면서도, 단순 forward chain의 비가역적 손실 대신 residual만 추출·주입해 텍스처 참조를 안정적으로 유지한다. FaDeNet은 마스크 게이팅과 magnitude-bounded detail refinement로 불필요한 과복원을 억제하고, SAGB(Spectral-Anisotropic Gated Blocks)로 방향성 stripe와 narrow-band periodic cue를 적응적으로 보정한다.

- **Empirical Impact**: PRISM 벤치마크에서 FaDeNet은 PSNR/SSIM/LPIPS 전 지표에서 최상 성능을 보이며, 특히 구조 보존과 지각적 자연스러움에서 큰 개선을 보였다. 연산 효율 측면에서도 파라미터 수와 처리 속도를 함께 고려해 7.05M 파라미터, 38.10 FPS로 경쟁 모델 대비 속도 우위를 유지했다. 또한 PRISM로 학습한 모델을 정교한 미세조정 없이 실제 비쌍( unpaired ) 직물 moiré에 적용했을 때도 ARNIQA/CLIP-IQA+ 및 블라인드 MOS에서 통계적으로 유의미한 선호를 보여, 벤치마크와 모델이 실제 전이에 강한 의미를 가진다.



### ObsGraph: Hierarchical Observation Representation for Embodied Reasoning and Exploration (https://arxiv.org/abs/2606.24068)
- **Prior Approaches**: 기존 embodied reasoning/exploration 연구는 dense 3D 지도나 scene graph, 그리고 이미지 메모리(일부는 task-conditioned)를 활용해 왔다. 하지만 dense 3D는 추상화가 부족해 task-조건 검색이 비효율적이고, 계층형 scene graph는 미래 태스크에 필요한 시각 증거를 일찍 압축해 놓칠 수 있다. 또한 VLM 기반 탐색은 next-target을 매 단계에 정하더라도, 현재 지식 상태에서 “얼마나/어디까지” 더 관측해야 하는지에 대한 스코프(탐색 스케일) 적응이 충분히 다뤄지지 않는 한계가 있었다.

- **Core Contribution**: 이 논문은 관측 자체를 중심으로 계층을 구성하는 observation-centric 계층형 장면 그래프 ObsGraph를 제안한다. room–view–object 3층으로 시각 증거를 보존하면서도, room은 거친 의미 앵커, view는 object co-visibility 기반 문맥, object는 fine-grained 디테일을 담당해 task-aware 추론과 정보 접근을 한 구조 안에서 연결한다. 더 나아가 retrieval 결과가 탐색 후보 공간(방/뷰/프론티어 탐색) 자체를 구조화하도록 설계해 representation–retrieval–exploration의 결합을 강화한다.

- **Technical Challenges**: 핵심 난제는 (1) 장면을 충분한 증거로 압축하되, (2) 추상화 수준에 맞춰 task-relevant 정보를 정확히 검색하고, (3) 검색 부족분을 어떤 스케일의 탐색 옵션으로 보완할지 결정하는 것이다. 이를 위해 view layer는 object co-visibility의 다양성을 유지하는 방식으로 redundancy를 줄이도록 설계했고, room layer는 VLM 예측을 확률적(베이지안) 결합해 room–object 할당의 강건성을 확보했다. retrieval은 장면 그래프 계층을 따라 LLM 호출 예산(bounded budget) 안에서 점진적으로 후보를 좁히고, 그 결과로 “room-level, view refinement, frontier exploration” 중 어떤 옵션을 활성화할지 결정하는 구조로 해결한다.

- **Empirical Impact**: EM-EQA, A-EQA, GOAT-bench에서의 실험은 ObsGraph가 기준선 대비 성공도와 효율을 함께 개선함을 보여준다. 특히 EM-EQA에서는 3D-Mem 대비 accuracy가 더 높으면서도 VLM 질의 프레임은 거의 늘지 않아(프레임 효율) “증거 보존+계층적 필터링”의 이점을 입증했다. A-EQA에서는 accuracy와 LLM SPL을 동시에 개선했으며, 이는 검색이 맞춰준 증거 갭이 탐색 결정을 더 VLM 추론에 정렬된 방식으로 유도했기 때문으로 해석된다.



### Ingredient-Level Food Image Segmentation for Nutrition Awareness (https://arxiv.org/abs/2606.24059)
Comments:
          5 pages, 4 figures, 4 tables

- **Prior Approaches**: 기존 음식 이미지 연구는 보통 한 장면 전체에 하나의 dish label을 부여해, 실제로는 한 이미지에 공존하는 여러 재료의 시각적 구조를 놓치기 쉽다. 이에 따라 재료 수준의 의미론적 분할(ingredient-level semantic segmentation)로 픽셀 단위 재료 클래스를 예측하려는 흐름이 있지만, FoodSeg103 같은 데이터에서의 성능 격차와 활용 방식이 과제로 남아 있었다.

- **Core Contribution**: 이 논문은 FoodSeg103에서 각 픽셀에 재료 클래스를 할당하는 ingredient-level semantic segmentation을 정면으로 다룬다. 또한 SegFormer의 SegFormer-B0(소형)과 SegFormer-B1(대형)을 ImageNet-pretrained MiT 백본 위에 104개 출력층을 새로 초기화해 fine-tuning하고, 예측 마스크를 재료 면적 비율(visual composition summary)로 변환하는 시각화 도구를 함께 제안한다.

- **Technical Challenges**: 핵심 technical challenge는 복잡한 음식 이미지에서 재료 경계를 픽셀 단위로 안정적으로 학습시키는 것과, 데이터/출력 차이에 맞춘 모델 파라미터 구성이었다. 연구진은 SegFormer의 MiT 백본을 ImageNet으로 사전학습한 뒤 출력 레이어만 새로 초기화해 104-class 분할에 적합하게 fine-tuning하여 정확도와 mean IoU를 끌어올렸다.

- **Empirical Impact**: FoodSeg103 테스트 split(2,135장)에서 SegFormer-B0는 pixel accuracy 0.7709, mean IoU 0.2521을 기록했고, SegFormer-B1은 pixel accuracy 0.7929, mean IoU 0.3204로 모든 저장 지표에서 개선되었으며 mean IoU는 +0.0683의 절대 향상을 보였다. 예측 마스크 기반 재료 면적 비율 요약은 영양 인식에 대한 첫 단서로 활용될 수 있으나, 칼로리·거시영양소·중량/부피·밀도·실제 portion size를 직접 추정하는 것은 아니라고 선을 그었다.



### VisChronos: Revolutionizing Image Captioning Through Real-Life Events (https://arxiv.org/abs/2606.24058)
Comments:
          SOICT 2024

- **Prior Approaches**: 기존 이미지 캡셔닝은 물체·행동·외형 같은 시각 신호에 크게 의존해 서술의 깊이가 제한되는 경우가 많다. 덴스 캡셔닝, region-to-text 트랜스포머 등은 장면을 더 촘촘히 설명하려 했지만, 영상/이미지에 없는 ‘사건의 맥락’까지는 잘 포착하지 못한다. 또한 한 번의 생성 패스로 끝나 디테일을 놓치거나 의미적으로 단단한 내러티브를 만들기 어렵다는 한계가 지적된다.

- **Core Contribution**: 논문은 사건 맥락까지 반영하는 새로운 과제인 Event-Enriched Image Captioning (EEIC)을 제안한다. 이를 위해 VisChronos라는 4단계 자동 프레임워크를 도입해, 단일 이미지에서 사건 정보를 끌어내고 자연어 캡션으로 통합하는 흐름을 만든다. 아울러 EventCap 데이터셋(3140개 이미지-캡션 페어)을 공개해 ‘사건 중심’ 캡셔닝 연구를 위한 학습·평가 기반을 제공한다.

- **Technical Challenges**: 핵심 난제는 이미지에만 기반하면 상상/추측이 섞이기 쉬운 ‘누가-무엇을-언제-어디서-왜’ 같은 사건적 사실을, 외부 지식과 일관되게 생성하는 데 있다. VisChronos는 (1) Perception Bot의 dense caption, (2) Interrogation Bot의 질문 생성, (3) Explanation Bot의 답변 추출(확신 없으면 no information), (4) Integration Bot의 캡션 합성으로 단계적으로 의미를 보강한다. 또한 EventCap 구축 시 1단계는 GPT-4o, 2~4단계는 Gemma-2-9b-it을 사용해 파이프라인을 구성했다.

- **Empirical Impact**: 사용자 연구에서 VisChronos 캡션은 인간이 쓴 캡션과 비교해 Faithfulness·Comprehensibility·Plus-Info 지표에서 유의미한 차이가 크지 않게 나타났다. 이는 사건 맥락을 더 풍부하게 담되, 이미지 기반 충실성을 해치지 않는 캡션 생성 가능성을 보여준다. EventCap이 ‘사건 중심’으로 최초 공개된 데이터셋이라는 점에서 향후 event-centric image understanding 및 멀티모달 캡셔닝 벤치마크 확장에 의미가 있다.



### EPEdit: Redefining Image Editing with Generative AI and User-Centric Design (https://arxiv.org/abs/2606.24057)
Comments:
          SOICT 2024

- **Prior Approaches**: 기존 포토 편집은 Photoshop, Capture One처럼 강력하지만 전문 지식과 학습 비용이 커서 진입 장벽이 높다. Luminar Neo, Pixlr X, Canva 등은 사용성을 낮추는 데 초점을 맞추지만, 사용자가 원하는 수준의 세밀한 제어와 유연성이 부족하다는 한계가 있다. 또한 Stable Diffusion 같은 대형 생성 모델을 상용 편집 서비스에 붙이려면 재학습·fine-tuning 비용이 커서 교육/전문 환경의 접근성에도 제약이 따른다.

- **Core Contribution**: 논문은 마스크와 프롬프트 기반으로 생성·교체·삭제·배경 수정·포즈/원근 변경·영역별 편집·테마 컬렉션 설계까지 아우르는 웹 기반 Efficient Photo Editor (EPEdit)를 제안한다. 핵심 차별점은 Stable Diffusion에 대해 추가 fine-tuning 없이 zero-shot image editing을 수행해, 높은 재학습 비용 없이 다양한 기능을 제공한다는 점이다. 별도의 복잡한 설정 없이 텍스트 커맨드나 영역 마킹으로 편집 의도를 전달할 수 있게 UI/상호작용도 함께 설계했다.

- **Technical Challenges**: 대형 모델을 활용하면서도 사용자가 체감하는 속도와 조작성(특히 마스킹 기반 편집)을 동시에 만족시키는 것이 기술적 난제다. EPEdit은 MVC 구조 위에 ReactJS 프론트엔드와 Express 백엔드를 두고, Python model server에 미리 로드된 생성 모델을 배치해 API 호출마다 모델을 다시 읽지 않도록 preloading으로 지연을 줄였다. 사용자는 캔버스에서 마스킹과 드로잉을 수행하고, 가상 어시스턴트가 제안하는 기능을 선택한 뒤 결과를 즉시 확인·저장할 수 있다.

- **Empirical Impact**: 사용자 연구는 24명의 지원자가 EPEdit으로 이미지 생성, 오브젝트 제거/추가, 배경 변경 등을 직접 수행하고 만족도·사용성 등을 1~5 척도로 평가한 방식으로 진행됐다. 결과적으로 EPEdit은 사용자 친화성, 조작 용이성, 기대 부합, 전반적 만족에서 높은 평가를 받았으며, 특히 Photoshop 대비 사용성·속도·결과 품질에서 유의미한 우위를 보였다고 보고한다. GPU A-100에서 샘플당 약 12–15초의 처리 시간을 제시하며, 전문가가 아니어도 접근 가능한 ‘비용 효율형’ 생성형 편집 도구로서의 의미를 강조한다.



### DriveStack-VLA: Render-Teacher Alignment for BEV-Based DeepStack Vision-Language-Action Mod (https://arxiv.org/abs/2606.24051)
- **Prior Approaches**: 기존 end-to-end Vision-Language-Action(VLA) 주행 모델은 멀티카메라 perspective image 토큰과 언어 지시를 기반으로 미래 궤적을 생성한다. 하지만 2D 관점 토큰은 시점 의존적이고 위에서 내려다보는(top-down) 메트릭 기하 정보를 계획에 직접 쓰기 어렵게 만들어, 약한 시각-기하 정합과 안전-핵심 단서의 커버리지 부족 문제가 발생한다. 또한 렌더링/래스터 합성 이미지를 섞으면 외형은 유사해져도 action decoder의 주의(attention)가 동일한 계획-관련 단서를 보지 못해 성능이 흔들릴 수 있다.

- **Core Contribution**: 본 논문은 large VLM 백본 위에 주행용 공간 인텔리전스를 강화한 DriveStack-VLA 프레임워크를 제안한다. 핵심은 (1) BEV(Bird-Eye-View) 표현을 DeepStack-style로 LLM 디코더에 주입해 위상-기하 기반의 안정적 프라이어를 제공하고, (2) Render-Teacher Alignment로 실영상과 래스터 이미지 사이에서 지각-주의를 정렬하며, (3) head-based self-critique로 샘플된 후보 궤적을 순위화·선택적으로 정교화하는 점이다.

- **Technical Challenges**: 기여를 위해 첫째, 긴 문맥과 다중 카메라에서 관점 토큰만으로는 기하적 안정성이 깨지는 문제를 BEV 분기로 완화해야 했다. 둘째, 래스터(교사)와 실영상 사이에 외형 차이로 인해 주의 분포가 어긋나지 않도록 action-to-vision attention distillation 및 render-guided soft mask로 배경 토큰 영향을 줄이며 KL 기반 attention 정렬까지 수행한다. 셋째, best-of-K 샘플링에서 디코딩 가능성과 주행 보상을 동시에 만족시키기 위해 GRPO로 actor를 직접 미세조정하되, KL 페널티와 format reward로 토큰 스키마 위반을 방지하고 critic 기반 점수/잔차 refinement로 후처리 효율을 높였다.

- **Empirical Impact**: DriveStack-VLA는 NAVSIMv1에서 91.6 PDMS, NAVSIMv2에서 human penalty filter 사용 시 91.0 EPDMS를 달성했고, Bench2Drive closed-loop에서는 주행 점수 79.49와 성공률 56.36%를 보고한다. 이는 실로그 기반 오픈루프와 시뮬레이션 폐루프 모두에서 end-to-end VLA의 약점이었던 기하 정합·안전 단서 학습·후보 궤적 선택 문제를 동시에 개선했음을 시사한다. 결과적으로 주행에서 언어 추론만이 아니라 metric geometry를 직접 활용하는 VLA 설계 방향에 실증적 근거를 제공한다.



### Token-to-Token Alignment of Text Embeddings for Semantic Blending (https://arxiv.org/abs/2606.24021)
- **Prior Approaches**: 기존 text-to-image 확산모델 편집/블렌딩은 픽셀 또는 latent 공간의 보간, LoRA·attention 조절 같은 방식으로 연속성을 시도해 왔다. 하지만 이런 접근은 중간 프레임에서 유령(ghosting), 구조 불연속, 속성 드리프트(semantic drift)가 생기기 쉬워 “의미가 자연스럽게 변하는 궤적”을 보장하기 어렵다. 또한 text embedding 공간에서 선형 보간이 가능해 보여도, 프롬프트마다 토큰화·표현이 달라 임베딩 정렬이 깨지면 보간이 무의미하게 섞인다.

- **Core Contribution**: 이 논문은 Token-to-Token alignment으로 프롬프트 쌍 사이에 토큰 단위 의미 대응을 명시적으로 만든 뒤, 그 위에서 선형 보간을 “의미 궤적”으로 해석 가능하게 만든다. 먼저 LLM을 써서 공통된 구조(semantically consistent scene description)로 프롬프트를 재표현한 뒤, 그 구조 제약 하에 토큰 임베딩을 의미 유사도로 정렬한다. 결과적으로 의미적으로 대응되는 토큰끼리만 섞이므로 블렌딩과 continuous editing에서 더 부드럽고 일관된 전이(transition)가 나온다.

- **Technical Challenges**: 핵심 난제는 (1) 문장 간 재표현에도 불구하고 토큰화 길이·경계가 달라 토큰 위치가 어긋나며, (2) 텍스트 인코더가 문맥화한 임베딩에서 같은 개념이 완전히 같은 토큰 ID로 대응되지 않는다는 점이다. 저자들은 구조 정렬로 semantic field별 역할과 위치의 공통 프레임을 만들고, 이후 유사도 행렬(cosine similarity)에 semantic field 마스킹, 국소 위치 편향(locality bias), temperature scaling을 적용해 안정적인 소프트 대응행렬을 계산한다. 그리고 정렬된 임베딩 시퀀스(EA, EB)에서 대응 토큰끼리 선형 보간해 확산모델 조건 입력으로 사용한다.

- **Empirical Impact**: 실험에서 FLUX-2와 FIBO-edit, 텍스트 인코더(Qwen3, SmolLM3) 조합으로 continuous blending과 continuous editing을 질적·정량적으로 평가했으며, 제안 방법은 중간 상태의 의미 일관성과 시각 품질 균형에서 경쟁력을 보였다. Morph4Data 및 새 벤치마크 BlendBench에서는 morphing 기반 방법들이 부드러움은 좋아도 의미·구조가 깨지는 경향이 나타난 반면, Token-to-Token alignment은 더 grounded하고 구조적으로 일관된 전이를 보였다. 또한 PIE-Bench 하위 세트의 continuous editing 평가와 사용자 선호도 조사에서 기존 연속 편집/블렌딩 기준선 대비 win rate가 높게 나타났고, ablation은 structural alignment와 embedding-level alignment의 결합이 성능을 좌우함을 확인했다.



### DivRL: Disentangled Self-Similarity Rewards for Diverse Subject-Driven Generation (https://arxiv.org/abs/2606.23950)
Comments:
          Accepted to ECCV 2026. Project page: this https URL

- **Prior Approaches**: 주제(subject-driven) 이미지 생성의 기존 확산 모델들은 참조 이미지의 정체성(identity)을 강하게 재현하는 데 강점이 있지만, 그 과정에서 자세·시점·표정 같은 구조가 거의 고정되어 출력 다양성이 급격히 떨어지는 문제가 반복됐다. 반대로 다양성을 밀어 올리는 보상 설계는 정체성 기준을 흐려(identity drift) 정밀한 미세 특징을 놓치기 쉽다. 또한 identity 보상을 단순히 선형 가중합으로 결합하면 두 목표의 그라디언트가 직접 경쟁해 학습 불안정이나 reward hacking이 발생하기 쉽다.

- **Core Contribution**: 논문은 Identity–Diversity Paradox를 “정체성은 보존하되 구조적 다양성은 함께 키우는” 방향으로 정식화하고, 이를 달성하는 post-training 프레임워크 DivRL을 제안한다. DivRL은 disentangled visual features 기반으로 정체성 일관성은 Visual Semantic Matching(VSM)으로, 구조 다양성은 Negative Self-Similarity Measure(nSSM)로 동시에 측정한다. 핵심 아이디어는 Explore-and-Suppress로 정체성 제약을 게이트로 취급해, 다양성 탐색을 먼저 넓히고 이후 VSM 기준을 위반하는 샘플만 억제해 두 목표가 함께 좋아지게 만든다는 점이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 구조 다양성을 강화하는 과정에서 무작위 샘플 생성 같은 reward hacking이 끼어들어 학습이 흔들리는 것이다. 저자들은 (1) 구조 다양성 측정이 고해상도에서 고주파 질감 재현으로 치우치지 않도록 feature downsampling(2×2 average pooling)을 포함한 nSSM 설계를 하고, (2) 그 뒤 두 단계 최적화로 nSSM 탐색 단계의 부작용을 VSM “identity gate”로 억제한다. 구체적으로 VSM 임계치를 넘는 위반 샘플에만 quadratic hinge loss를 적용해 identity 보상을 경쟁 목표가 아니라 feasibility constraint로 바꾸는 gated optimization을 구현했다.

- **Empirical Impact**: DreamBench++에서 Flux-Kontext 백본을 사용한 실험 결과, DivRL은 정체성 일관성은 경쟁 수준으로 유지하면서도 구조 다양성을 유의하게 확장했다. 특히 VSM-only, nSSM-only 변형들과 비교할 때, 단일 보상 최적화가 각각 한쪽 목표만 극단적으로 개선하는 경계(정체성-다양성 트레이드오프)를 넘어서는 균형점을 만든 것이 확인됐다. 또한 구조 다양성/정체성 지표의 전개 곡선과 trade-off frontier에서 linear weighting 조합이 동시 개선에 실패하는 반면, gated 두 단계 전략은 VSM과 nSSM이 함께 상승하는 경향을 보여 DivRL의 실질적 기여를 뒷받침한다.



### Trustworthy Image Authentication using Forensic Knowledge Graphs (https://arxiv.org/abs/2606.23917)
Comments:
          Accepted and Published at ECCV 2026

- **Prior Approaches**: 기존 미디어 인증은 합성 이미지 검출, 조작 분류기, 스플라이싱 로컬라이저처럼 특정 위조 유형에 맞춘 방식이 많아 다른 조작으로 일반화가 어렵다. 포렌식 모델은 보이지 않는 통계 흔적(센서 노이즈, 압축/디모자이싱 잔차 등)을 쓰지만, 다중 단서 간의 인과적 추론이나 왜 그런 결론이 나왔는지의 설명은 제한적이다. 반대로 VLM은 자연어 근거를 만들 수 있으나 포렌식 마이크로구조를 실제 증거로 활용하지 못해 신뢰할 만한 검출 성능이 낮고, 설명도 증거에 덜 고정되는 문제가 크다.

- **Core Contribution**: 이 논문은 Forensic Knowledge Graphs(FKG)를 도입해 포렌식 증거 추출, 구조화된 추론, 인간이 해석 가능한 정당화를 하나의 프레임워크로 통합한다. FKG는 이미지의 영역(Region)별로 출처(Source), 후처리(Post-Processing), 압축(Compression), 장면 의미(Content)를 그래프 형태로 연결하고, 각 결론이 어떤 증거로 지지되는지 인과 의존성까지 함께 표현한다. 또한 FKG 생성에 필요한 Forensic authentication network와 FKG-50K 데이터셋(5만 장, ground-truth FKG 포함)을 함께 제시한다.

- **Technical Challenges**: 핵심 난제는(1) 눈에 보이지 않는 통계 지문을 기반으로 영역을 올바르게 분할하고(2) 그 결과를 그래프 온톨로지에 일관되게 매핑하며(3) VLM이 그래프 근거를 ‘그럴듯하게’ 꾸미지 않도록 신뢰도 높게 설명을 생성하는 것이다. 이를 위해 논문은 self-supervised trace extraction으로 포렌식 임베딩을 학습하고, Forensic Region Proposal Network(FRPN)에서 포렌식 동질 영역을 찾기 위해 global self-attention과 local graph attention을 교차하는 Hybrid Graph Attention Transformer를 사용한다. 마지막으로 Iterative Context Refinement(ICR)로 VLM이 누락·환각하는 삼중항(증거)을 자동으로 보정하는 컨텍스트를 반복 구축해, 모든 포렌식 주장이 FKG 증거에 정착되도록 한다.

- **Empirical Impact**: 실험 결과 FKG는 위조 탐지에서 포렌식 전용 모델과 VLM을 모두 능가하며, 위조 유형 식별 및 로컬라이제이션에서도 일관된 향상을 보인다. FKG-50K에서 전체 정확도 0.92, AUC 0.94를 달성하고, OOD 데이터에서도 강한 일반화 성능(예: DSO-1 0.96, Synthbuster 0.96)을 유지한다. 특히 VLM이 지역 편집/로컬 조작에서 취약한 반면, FKG는 영역 간 포렌식 지문 차이를 그래프로 추적해 AI-edits까지 포함한 다중 조작 스펙트럼을 처리하며, 포렌식 justification까지 제공하는 점에서 ‘신뢰 가능한 인증’ 관점의 의미가 크다.



### The Professor: Multi-Teacher Unsupervised Prompt Distillation for Vision-Language Models (https://arxiv.org/abs/2606.23897)
- **Prior Approaches**: PromptKD는 unlabeled 도메인 이미지로 큰 VLM(teacher)의 soft 예측을 작은 student에 KL-divergence로 전이해, 추론 시 teacher를 제거하는 비지도 프롬프트 증류 방식을 제시했다. 하지만 teacher가 하나라서 pretraining, prompt tuning 목표, calibration 같은 ‘한 가지 귀납편향’이 그대로 soft label에 반영되는 한계가 있었다. 기존 앙상블/증류 문헌은 teacher diversity가 이득의 핵심이라고 보지만, VLM 프롬프트 학습 맥락에서는 다중 teacher가 언제/얼마나 유효한지 충분히 검증되지 않았다.

- **Core Contribution**: TheProfessor는 PromptKD를 다중 teacher 증류로 확장해, 서로 다른 두 CLIP-family teacher의 신호를 student에 함께 학습시킨다. T1은 PromptSRC로 도메인 few-shot을 학습한 CLIP ViT-L/14이고, T2는 zero-shot EVA-CLIP-L/14이며 T2의 logits는 데이터셋별로 사전 캐시해 학습 중 추가 추론 비용을 없앤다. 또한 teacher 예측을 equal averaging과 confidence-weighted averaging 두 방식으로 결합하며, 특히 도메인 shift에서 보완적 supervision이 있을 때 효과가 커진다는 관점을 제시한다.

- **Technical Challenges**: 다중 teacher를 쓰면 학습 중 두 큰 모델의 forward가 필요해 비용이 급증하는데, 이를 해결하기 위해 T2 logits를 이미지 단위로 한 번만 pre-compute해 저장하고 학습 시에는 경로로 즉시 로드한다. 다음으로 학생 학습 손실은 앙상블 teacher 분포와 student 분포 사이의 KL loss로 정의해 end-to-end student 업데이트를 유지했다. 마지막으로 confidence-weighted averaging은 이미지별 최대 softmax confidence를 가중치로 사용해, teacher 신뢰도가 이미지/데이터셋에 따라 달라지는 문제에 적응하도록 설계했다.

- **Empirical Impact**: 4개 base-to-novel 데이터셋(Caltech-101, DTD, UCF101, EuroSAT)에서 12-run 단일 시드 스윕을 수행한 결과, confidence-weighted ensembling이 평균 HM을 87.52에서 89.28로 +1.77p 개선했다. equal averaging은 평균 HM을 88.88로 +1.37p 올렸고, 데이터셋 의존성이 뚜렷해 Caltech-101에서는 거의 변화가 없었지만 EuroSAT에서 HM이 +5.78p로 가장 크게 상승했다. 저자들은 teacher agreement가 높은 ‘쉬운’ 데이터셋에서는 이득이 천장에 걸리지만, 도메인 shift에서는 teacher의 상보성이 살아나 multi-teacher prompt distillation이 상당한 성능 향상을 낼 수 있음을 실증했다고 정리한다.



### REALM: A Unified Red-Teaming Benchmark for Physical-World VLMs (https://arxiv.org/abs/2606.23892)
Comments:
          20 pages, 5 figures. Preprint

- **Prior Approaches**: 기존 red-teaming은 주로 jailbreak·콘텐츠 안전을 표준화했지만, 물리 세계에서의 기능적 실패(잘못된 주행/조작 판단 등)를 체계적으로 잡아내기 어렵습니다. 또한 공격별로 데이터셋·지표·위협 모델이 달라, 어떤 차이가 더 강한 공격 때문인지 더 취약한 모델 때문인지 비교가 흐려졌습니다. 물리 벤치마크는 주로 clean 성능을 보지만, 다양한 물리-근거 공격과 공통 목표를 한 프로토콜로 묶어 검증하는 통합 평가는 부족했습니다.

- **Core Contribution**: REALM(논문명: ReaLM)은 물리 세계 VLM을 위한 최초의 unified red-teaming benchmark로, 12개 공격 기법과 3개 model-agnostic defense를 13개 VLM에 동일한 설정으로 평가합니다. 핵심은 시나리오별로 물리적으로 그럴듯한 실패 목표를 만들고(정답에서 벗어나되 장면 맥락과 일치), 이 목표를 모든 공격 계열이 공유하도록 맞춘 agentic target-generation pipeline입니다. 이를 통해 다양한 공격을 같은 “유도하려는 오작동” 아래에서 공정 비교할 수 있게 합니다.

- **Technical Challenges**: 물리 세계에서는 고정된 유해 행동 카테고리나 임의의 source–target 쌍만으로는 장면·질문·과업에 맞는 실패를 설계하기 어렵습니다. 논문은 reasoning–generation–refinement 루프로 장면을 분석해 타깃 답과 타깃 이미지를 함께 생성하고, 생성된 타깃이 의도된 오해를 실제로 반영하도록 프롬프트를 반복 정교화합니다. 또한 black-box·one-shot 평가(피해 모델 파라미터/그라디언트 비사용, 전처리만 defense 적용) 제약 하에서 공통 데이터셋·공통 metrics(특히 physically grounded ASR)를 사용해 공격 계열 간 비교 가능성을 확보했습니다.

- **Empirical Impact**: 실험은 driving·manipulation·grasping·physics 등 7개 물리 도메인과 13개 VLM(7B~100B+ 스케일)에서 진행됐고, attack별 ASR 패턴이 명확히 드러났습니다. text 및 typographic injection 계열이 가장 큰 실패를 유발했고, multimodal co-optimization은 시각 섭동의 transfer를 가장 강하게 만들었으며, AnyAttack 같은 single-pass 생성형 공격은 iterative 방식에 근접하는 효율을 보였습니다. 반면 모델 scale 증가나 물리 reasoning용 post-training만으로는 black-box 물리-근거 취약성을 충분히 줄이지 못했고, 도메인별 취약성 편차가 커 단일 설정 평가의 한계를 보여주었습니다.



### Mind the Heads: Topological Representation Alignment for Multimodal LLMs (https://arxiv.org/abs/2606.23885)
- **Prior Approaches**: 기존 representation alignment 연구는 MLLM 내부 표현을 비전 인코더 표현에 맞추는 방식으로 비주얼 능력을 끌어올리지만, 대개 LLM의 고정된 한 층(예: 중간층)만 정해 정렬합니다. 또 layer-level 정렬은 Transformer의 미세한 구조(어텐션 head별 역할)를 충분히 반영하지 못해, 언어 모델의 기존 우선순위와 충돌하거나 최적의 선택을 놓칠 수 있다는 한계가 제기됩니다.

- **Core Contribution**: 이 논문은 Head-Wise Representation Alignment(HeRA)를 제안해, LLM의 attention head 단위로 비전-언어 표현 정렬을 수행합니다. MKNN(상호 k-최근접 이웃) 점수를 진단으로 사용해 정렬할 head를 선택하고, 대비(contrastive) 목적함수로 비전 인코더가 만드는 로컬 이웃(topological neighborhood) 구조를 차분하게 흉내 내도록 학습시킵니다.

- **Technical Challenges**: 핵심 난관은 MKNN이 k-최근접 이웃 인덱스에 의존해 미분 불가능하다는 점인데, 논문은 이를 differentiable proxy인 multi-target InfoNCE 형태의 대비 손실로 대체합니다. 또한 head를 무작정 탐색하기 어렵기 때문에, 멀티모달 학습 전 순수 텍스트 상태에서 head별 MKNN 정렬 점수를 계산해 상위/하위 후보를 선별하고, HeRA는 그중 일부 head에만 정렬 손실을 적용합니다.

- **Empirical Impact**: 18개 벤치마크(Cambrian)와 여러 hallucination 벤치마크에서 HeRA는 vision-centric 과제에서 일관된 성능 향상을 보이며, 종종 시각 환각을 줄이는 정규화 효과도 확인됐습니다. 특히 “가장 덜 정렬된(worst) head를 맞추는 전략”이 가장 큰 이득을 주었고, 다양한 MLLM(LLaVA 프레임워크, 파라미터 스케일 포함)에서 General/Knowledge/OCR 성능 저하 없이 안정적으로 개선되었습니다.



### HANCLIP: A Family of Hyperbolic Angular Negation Vision Language Models (https://arxiv.org/abs/2606.23843)
- **Prior Approaches**: 기존 비전-언어 모델(VLM)은 대규모 이미지-텍스트 사전학습으로 의미 대응을 학습하지만, 부정(negation)에는 의외로 취약합니다. 많은 연구가 “문장-단어 동시출현” 같은 얕은 단서에 의존해, 부정 캡션과 정답 캡션 모두에 높은 유사도를 주는 문제가 보고됩니다. 또 부정 데이터로 직접 fine-tuning하면 부정 감도는 좋아져도 표준 벤치마크 성능이 떨어지는 지식 망각(forgetting) 문제가 자주 따라옵니다.

- **Core Contribution**: 이 논문은 HANCLIP(Hyperbolic + Angular + Negation)이라는 VLM 계열을 제안하며, 임베딩 공간 자체를 “무엇이 맞는가”와 “무엇이 아닌가”를 분리하는 방식으로 재구성합니다. 하이퍼볼릭(비유클리드) 기하를 이용해 부정 캡션이 긍정/부정 의미 위계에서 다른 영역을 차지하도록 설계하고, 추가로 각도(angular) 기반 삼중항 손실로 부정-긍정 간 관계를 정교하게 조절합니다. 특히 대규모 재학습 없이 CLIP 계열 백본에 플러그인 형태로 적용 가능하다고 강조합니다.

- **Technical Challenges**: 핵심 기술적 난제는 부정 민감도를 높이면서도 기존 사전학습된 표현의 전역 구조(global structure)를 덮어쓰지 않는 것입니다. 이를 위해 (1) HCO(Hyperbolic Contrastive Objective)는 포인카레 볼(Poincaré ball) 위에서 하이퍼볼릭 거리로 진짜 negative를 명시적으로 대조하고, (2) ATNL(Angular Triplet Negation Loss)은 negative를 앵커로 두고 상대 방향의 각도를 정렬/분리하여 부정이 의미를 왜곡하지 않게 하면서도 잡음성 어휘 중복에 덜 흔들리도록 만듭니다. 결과적으로 “부정 캡션은 긍정과는 가깝되(각도 정렬), 진짜 negative와는 멀어지게(각도 분리)” 만드는 기하-aware 목적함수를 구성합니다.

- **Empirical Impact**: CC-Neg에서 2만 개 샘플로 학습한 뒤 NegBench의 Negative Retrieval 및 MCQ-Neg에서 전반적으로 성능이 개선됐고, 백본이 CLIP/LongCLIP/SmartCLIP/HiMo-CLIP일 때도 일관된 향상이 관찰됩니다. 논문은 NegBench에서 rSum 기준 대략 5~10% 수준의 상대 개선과, MCQ에서 평균적으로 큰 폭의 상승(거의 35%에 근접)을 보고하며 데이터 효율성을 강조합니다. 또한 표준 이미지-텍스트 검색/분류 성능은 경쟁 수준을 유지하거나 개선되어, 단순 fine-tuning의 지식 망각 트레이드오프를 완화하는 방향성을 제시합니다.



### ABACUS: Adapting Unified Foundation Model for Bridging Image Count Understanding and Generation (https://arxiv.org/abs/2606.23835)
Comments:
          Under review, webpage: this https URL

- **Prior Approaches**: 기존 객체 카운팅은 밀도맵 회귀나 검출 기반, 혹은 클래스별/클래스비경계 방식으로 나뉘며, 대부분은 특정 감독 신호에 크게 의존해 다른 카운팅 체계로의 일반화가 어렵습니다. VLM/MLLM을 활용한 통합 시도도 많지만, dense scene에서 인스턴스 수준 지침을 공간적으로 정렬(grounding)하지 못해 정밀 카운트 지침을 따라가지 못하는 문제가 반복됩니다. 텍스트-투-이미지 생성 쪽은 수적 정확도(numeracy) 실패가 널리 알려져 있고, 외부 카운팅 모듈이나 critic을 붙이더라도 정확도는 critic 신뢰도에 제한됩니다.

- **Core Contribution**: ABACUS는 단일 3B-파라미터 비전-언어 모델로 객체 카운팅, 군중 카운팅, referring-expression counting, count-faithful image generation을 같은 표현에서 처리하며, 벤치마크별 학습 없이 zero-shot 성능을 노립니다. 이를 위해 MLLM의 spatial grounding 실패를 줄이기 위한 density-aware adaptive zooming과 objectness map을 도입하고, crop 경계에서 발생하는 over/undercounting을 boundary-aware count policy로 완화합니다. 더 나아가 이해(카운팅) 분기를 frozen verifier로 쓰는 cycle-consistent GRPO로 생성 분기를 self-critique 기반으로 개선해 이해-생성 synergy gap을 닫습니다.

- **Technical Challenges**: 첫 번째 난제는 dense 장면에서 MLLM이 인스턴스 단위의 세밀한 카운트 근거를 안정적으로 추론하지 못한다는 점이며, ABACUS는 재귀적 영역 분할로 국소적으로 더 sparse한 필드에서 추정하게 만들어 이를 보완합니다. 두 번째 난제는 분할 과정에서 생기는 crop boundary 문제로, 경계 걸침 인스턴스에 대한 소유권 귀속이 흔들리면 체계적 카운트 오류가 누적된다는 것입니다. ABACUS는 GRPO 기반의 boundary-aware count policy로 per-quadrant/경계/전역 보상을 중첩해 경계 모호성을 학습적으로 해결하고, 또한 생성 단계에서는 frozen 이해 분기의 count-deviation 점수를 포함해 reward hacking을 억제하면서도 생성 분기만 업데이트하도록 설계했습니다.

- **Empirical Impact**: ABACUS는 7개 벤치마크 전반에서 state-of-the-art를 보고하며, task-specific specialist와 더 큰 generalist 모델을 단일 3B 모델로 능가하는 결과를 제시합니다. 예를 들어 FSC-147과 CARPK에서 객체 카운팅 MAE가 유의미하게 개선되었고, ShanghaiTech A/B에서는 군중 카운팅 오차를 기존 최고 수준 대비 크게 낮춰 오류를 절반가량으로 줄였다고 설명합니다. count-faithful generation 및 count reasoning 영역에서도 기존 diffusion 기반의 numeracy 실패를 완화하는 방향으로 성능을 확인하며, 별도의 external critic/플래너/주석 없이도 카운트 일관성을 학습할 수 있음을 실증합니다.



### From Spatial to Spectral: An Efficient, Frequency-Guided Feature Representation Learner for Small Object Detection (https://arxiv.org/abs/2606.23825)
- **Prior Approaches**: 기존 소형 물체 탐지는 downsampling과 다중 스케일 융합 과정에서 작은 물체의 고주파 단서가 약화·소실되면서 성능이 급격히 떨어지는 문제가 컸습니다. 그래서 dilatation, 더 촘촘한 피라미드 등 공간 복원(특히 upscaling) 기반 처방이 흔했지만, 계산 비용이 크고 배경 잡음까지 함께 증폭할 수 있다는 한계가 지적됩니다. 주파수 도메인 접근도 분류 중심이거나 탐지에서 stage별 요구를 통일적으로 다루지 못해 통합 솔루션으로 자리잡기 어려웠습니다.

- **Core Contribution**: 이 논문은 공간 복원에서 벗어나, 탐지 파이프라인을 구성하는 backbone-neck-head 단계에서 필요한 주파수 단서를 보존·재주입하는 Frequency-Guided Feature Representation 패러다임을 제안합니다. 핵심은 Decompose--Enhance--Reconstruct(DER)라는 통합 연산자 인터페이스로, 이를 Wavelet-Difference Gate(WDG), Log-Gabor Enhancer(LGE), Frequency-Driven Head(FDHead)로 단계별 구현해 CNN/Transformer 모두에 플러그앤플레이로 적용 가능하게 했습니다. 결과적으로 해상도 감소와 무관하게 고주파 신호를 명시적으로 다루어 소형 물체의 경계·디테일을 복원하려는 접근이 정리됩니다.

- **Technical Challenges**: 해결해야 할 기술 과제는 (1) downsampling·fusion 같은 되돌리기 어려운 연산 전에 고주파 증거를 “복원/강화”하되 잡음 증폭을 피하는 것, (2) 이를 backbone/neck/head의 역할에 맞게 stage-wise로 설계해 서로 다른 탐지기에도 동일한 방식으로 꽂아 넣는 것입니다. 저자들은 WDG에서 wavelet 저주파 근사에 RepCDC로 경계 성분을 보강하고, 고주파 서브밴드를 self-derived gate로 활용해 low-frequency 오염을 억제한 뒤 inverse transform으로 재구성합니다. neck에서는 Log-Gabor 기반 고방향 잔차를 복원하도록 LGE를 설계하고, head에서는 wavelet 고주파 에너지로 box regression에만 boundary-sensitive 게인을 부여하는 FDHead를 통해 위치 회귀의 안정성을 높였습니다.

- **Empirical Impact**: VisDrone2019, UAVDT, TinyPerson, DOTAv1의 다중 벤치마크에서 DERNet 계열은 일관된 소폭~큰 폭의 성능 향상을 보였고, 특히 YOLOv11과 동일 스케일 조건에서 파라미터는 1/6 수준으로 줄이면서도 더 높은 mAP50을 달성했다고 보고합니다. 단계별 모듈 기여를 쪼갠 결과 FDHead가 가장 큰 향상을 주는 등, 설계 의도가 성능으로 연결됨이 확인됩니다. 또한 TIDE 기반 오차 분해와 2D FFT 기반 고주파 에너지 비율 분석에서 high-frequency 보존/경계 신호 정밀화가 Miss 및 Localization 오류 감소와 인과적으로 연결된다는 해석을 뒷받침하며, edge 하드웨어에서도 현실적인 FPS 저하 없이 효율-정확도 절충을 달성했다고 강조합니다.



### Listening makes Vision Clear for VLMs (https://arxiv.org/abs/2606.23763)
Comments:
          18pages,3 figures

- **Prior Approaches**: 기존 VLM 의미 일관성 평가는 answer-side 토큰의 attention 분포(또는 IoU 유사 마스크 겹침)를 통해 시각 영역과 언어 토큰의 정렬을 추정하는 방식이 주를 이뤘습니다. 하지만 가장 높은 attention이 항상 목표 의미 토큰과 일치하지 않는 사례가 관찰되며, 이는 이전에 생성된 토큰이 누적시키는 언어 prior로 인한 decoding drift 문제로 설명됩니다. 또한 modality boundary 같은 구조 토큰이 문맥 전체를 포괄하며 목표와 무관한 영역에 높은 attention을 유도할 수 있습니다.

- **Core Contribution**: 이 논문은 answer 생성에 의존하던 평가 관점을 prompt-side 의미로 전환해, 고정된 프롬프트 토큰을 대상으로 token-to-vision activation map을 뽑아 일관성을 진단하는 Prompt-Vision Token Activation Map(PV-TAM)을 제안합니다. 동시에 구조 토큰에서 기인한 공간 편향을 인접 구조 토큰으로부터 추정해 제거하는 denoising 필터를 설계해, 목표 의미 신호를 더 깨끗하게 분리합니다. 마지막으로 attention의 강도 분포까지 반영하도록 TGR, TDR, MinDist 같은 정렬 지표를 제시합니다.

- **Technical Challenges**: 핵심 난제는 자동회귀 생성 과정에서 answer-side attribution이 prefix 문맥 변화에 민감하게 흔들리며(디코딩 드리프트), attention 피크가 목표 토큰의 의미 증거와 어긋날 수 있다는 점입니다. 이를 해결하기 위해 PV-TAM은 목표 개념을 고정된 prompt 토큰으로 배치하고, 그 prompt 토큰에서 얻는 prompt-to-vision attention을 activation map으로 사용해 생성 접두사의 오염을 차단합니다. 이어 structural-token 주의를 제거하기 위해 인접한 특별 토큰들의 attention을 이용해 편향을 빼고(양수 증거만 유지하는 ReLU 기반 처리 포함), 선택적으로 foreground 마스크(rembg)로 배경 지배를 줄여 성능을 안정화합니다.

- **Empirical Impact**: PartImageNet과 RefCOCO 실험에서 PV-TAM은 기존 answer-side 해석/국소화 기준선(TAM, Grad-CAM, CP-LRP, Attention Rollout) 대비 TGR/TDR 개선과 MinDist 감소를 일관되게 보였습니다. 특히 Qwen 및 InternVL 계열의 여러 모델에서 유사한 이득이 재현되며, 작은 모델(예: InternVL3-1B)에서도 성능 우위가 나타났습니다. 또한 정성 시각화에서 PV-TAM은 목표 부품/표현의 경계에 더 정확히 집중하고 attention diffusion 실패를 줄이는 경향을 보였으며, “attention 정렬이 training 정렬 손실 없이도 자연 발생할 수 있다”는 관찰까지 제시합니다.



### Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation (https://arxiv.org/abs/2606.23743)
- **Prior Approaches**: 기존 비디오 diffusion 가속은 캐시, sparse attention/토큰 감축, 양자화, 커널 최적화 등 단일 기술군을 중심으로 성능을 끌어올리는 방식이 많았다. 하지만 (모델-하드웨어-서빙 설정) 조합마다 병목과 민감도가 달라져, 한 인스턴스에서 잘 된 레시피가 다른 배포로 잘 옮겨가지 않는 문제가 반복됐다.

- **Core Contribution**: 이 논문은 Sol Video Inference Engine을 통해 비디오 diffusion 추론 가속을 “훈련 없이” 인스턴스별 조합 최적화로 재정의한다. 특히 cache, sparse attention, token pruning, quantization, kernel fusion을 에이전트 기반으로 조합해 전 과정 속도-품질을 맞추는 풀스택 가속 스택을 제안한다.

- **Technical Challenges**: 가장 큰 기술 난제는 각 가속 기법이 품질 열화의 양상과 하드웨어/서빙 설정 의존성이 커서, 로컬 최적을 단순 결합하면 전역 성능이 무너질 수 있다는 점이다. 논문은 병렬 skill agent가 각 기법군을 로컬 튜닝한 뒤 integrator가 전역 조합을 탐색하고, human validator가 VBench 유사한 정량만으로 놓치기 쉬운 시각 품질을 피드백해 다음 라운드를 조절하는 구조로 이를 해결한다.

- **Empirical Impact**: Cosmos3-Super(64B), LTX-2.3(22B), SANA-Video(2B)에서 동일한 프레임워크로 배포 인스턴스를 최적화해 엔드투엔드 기준 2배 이상 가속을 달성하면서 VBench 품질을 “거의 무손실” 수준에 가깝게 유지했다고 보고한다. 결과적으로 단일 기법이 아니라 다기술 조합을 에이전트로 자동 구성하는 접근이 비디오 diffusion 가속의 실사용 비용을 낮출 수 있음을 실증한다.



### A Geometry-Informed Computer Vision Method for Detecting and Examining Overtaking Vehicles From A Bicyc (https://arxiv.org/abs/2606.23699)
Comments:
          18 pages, 6 figures, in preparation for journal submission

- **Prior Approaches**: 계측 자전거(instrumented bicycle) 연구는 실제 추월 행위를 직접 포착하지만, 후방 연속 영상에서 ‘추월 이벤트가 발생한 구간’을 프레임 단위로 수동 주석해야 했습니다. 이 병목은 표본 수를 줄이고 연구 결과의 재현성과 관찰자 간 일관성(이벤트 식별 기준 차이)을 제한합니다. 한편 YOLO 계열이나 RT-DETR 같은 외관 기반 검출/분류는 도메인 이동과 카메라 흔들림에 취약하고, 다중 센서 방식은 동기화·보정·하드웨어 비용 부담이 큽니다.

- **Core Contribution**: 논문은 단일 자전거 장착 카메라(후방 좌측 시점)에서 기하학 정보로 추월 이벤트를 자동 격리하는 파이프라인을 제안합니다. RT-DETR 기반 검출과 ByteTrack 추적 위에, 시야각(bearing angle) 추세·겉보기 크기 성장·공간적 확인(확정 임계치)을 3단계로 검증해 ‘완료된 추월’만 통과시키는 로직을 구현합니다. 카메라 보정 없이도 동작하도록 설계해, 장비 의존성을 낮추고 다양한 장착 위치에 맞춰 설정값만 조정 가능하게 했습니다.

- **Technical Challenges**: 핵심 기술 난제는 후방 영상에 존재하는 대부분의 차량이 실제 추월이 아니라는 점, 그리고 움직이는 카메라의 ego-motion으로 인해 단순 모션 필터만으로는 이벤트를 구분하기 어렵다는 데 있습니다. 저자들은 확률적 검출 오류와 ID 스위치, 중복 박스까지 고려해 검출 전 중복 억제, 추적 파라미터 조정, 동일 프레임 중복 트랙 필터를 적용했습니다. 이어서 확정 전 ‘probationary’ 구간에서 선형 회귀 기반 각도 증가, 바운딩박스 면적 성장률, 연속 프레임 지속성을 동시에 요구해 기하학적 불변성을 이벤트 확인에 연결했습니다.

- **Empirical Impact**: Ann Arbor(도심 도로)에서 수동 검증된 315개 이벤트로 평가한 결과, 97.8% 재현율과 0 false positives를 달성했습니다. 이벤트 의도 감지(확정 전 probation 진입) 시점은 평균 2.44초로, 84.1%가 일반적인 1.5초 인간 반응 시간 기준을 넘겨 능동 경고(active warning) 가능성을 보여줍니다. 또한 96개 이벤트의 초음파 기반 측면 거리에서 33.3%가 5-foot(152.4cm) 이하로 나타났고, 바운딩박스 기하 피처만으로 보정 없는 거리 추정 MAE 13~14cm 수준을 확보해 안전 분류를 위한 스케일업 기반을 마련했습니다.



### ArtiTwinSplat: Interactable Digital Twin Reconstruction via Gaussian Splatting from RGB-D videos (https://arxiv.org/abs/2606.24628)
Comments:
          Presented at the ICRA 2026 Workshop on Advances and Challenges in AI-Driven Automation and Robotic System Integration with Digital Twins, Vienna, June 2026

- **Prior Approaches**: 관절이 있는 물체(문, 서랍 등)를 재구성하려는 기존 연구는 3D 형상과 모션을 함께 추정하되, 정적 장면 가정(NeRF, 3DGS 계열)이나 파트 라벨/사전 분할, 다중 뷰·다중 상태 캡처 같은 전제가 많았다. Real2Code, ArticulateAnything, PARIS 등은 큰 주석 데이터나 분할 입력, 통제된 캡처 조건에 의존해 실제 로봇 배치 환경에서 확장성이 떨어진다. 3DGS 기반의 ArtGS, SplArt 역시 부분 감독 또는 여러 관측 조건이 필요해 핸드헬드 RGB-D 단일 시퀀스에는 제약이 컸다.

- **Core Contribution**: ArtiTwinSplat은 CAD나 시뮬레이션 자산, 수동 주석 없이 핸드헬드 RGB-D 영상만으로 관절(회전/직선) 디지털 트윈을 자동 생성하는 프레임워크다. 3D Gaussian Splatting으로 외관과 기하의 photorealistic 재현성을 유지하면서, 관절 구조와 kinematics를 “관측된 움직임만”으로 비지도 방식으로 찾아낸다. 결과물은 실제 로봇 시뮬레이션에 바로 쓸 수 있게 URDF로 내보낼 수 있어, 디지털 트윈 제작의 통합 병목을 줄이는 데 초점을 둔다.

- **Technical Challenges**: 가장 큰 난제는 관절이 있는 객체에서 구조(파트)와 모션(관절 파라미터)이 강하게 결합되어 있어 단일 관측만으로는 문제 자체가 ill-posed라는 점이다. 이를 위해 ArtiTwinSplat은 (1) 사전 변화(pre-change) 기준 3DGS를 만든 뒤, (2) 변화 마스크를 외관·기하 차이로 검출하고 SAM2로 시공간 일관된 파트 분할을 역방향 전파로 안정화한다. 이후 TAPIP3D로 마스크 영역 점 궤적을 추적하고 4D RANSAC으로 revolute/prismatic joint를 피팅해 관절 타입·축·피벗·가동 범위를 추정한 뒤, 관절 파라미터에 조건을 건 2단계 Gaussian 최적화로 장면 외관은 유지하면서 시간적으로 일관된 관절 변형을 학습한다.

- **Empirical Impact**: 아이폰 Pro로 촬영한 3개 가정용 장면(각 장면은 정적 pre-change와 동적 post-change 2개 시퀀스)에서, 움직이는 컴포넌트를 안정적으로 분리하고 미관 생성 및 새로운 관절 상태에서의 합성 렌더링을 일관되게 보여준다. prismatic은 운동 중 내부가 비교적 지속적으로 보이기 때문에 더 정확하게 복원되며, revolute는 회전 중 가림으로 내부 표면 일부 아티팩트가 생길 수 있다고 보고한다. 무엇보다 한 번의 캐주얼 RGB-D 캡처로 Isaac Sim 호환 URDF까지 생성돼, embodied AI와 휴먼-로봇 협업에서 “바로 쓰는” articulated digital twin 제작 접근을 실사용 관점에서 한 단계 낮췄다는 점이 의미 있다.



### Female-RHINO: A Real-Time Scanner-Integrated Framework for Automated Quantitative Uterine MRI Analysis and Structured Reporting (https://arxiv.org/abs/2606.24390)
- **Prior Approaches**: 기존 자궁 MRI 평가는 해부학적 변동성과 병변/생리 변화, 관찰자 의존성 때문에 표준화가 어렵고 주로 검사 후 수작업 계측에 의존해 왔다. 딥러닝 기반 자동화도 주로 segmentation, landmark detection 같은 단일 태스크 중심의 오프라인 방식에 머물러 실제 촬영 중 워크플로우에 바로 결합되지 못했다.

- **Core Contribution**: 이 논문은 Female-RHINO로, MRI 촬영 중 실시간으로 자동 정량 분석과 structured reporting을 생성하는 scanner-integrated end-to-end 프레임워크를 제안한다. sagittal T2-weighted 골반 MRI에서 3D 분할과 3D landmark를 동시에 수행해 자궁/병변 volumetry, Nabothian cyst·fibroid 계측 및 6개 landmark 기반 바이오메트릭을 자동 산출한다.

- **Technical Challenges**: 핵심 기술 난제는 다양한 프로토콜·벤더·필드세기에서의 도메인 차이와, 큰 근종 등 해부학적 왜곡 상황에서의 landmark 안정적 국소화였다. 이를 위해 nnUNet-v2 기반 3D segmentation과 Swin-UNETR heatmap 회귀 landmark detection을 함께 학습하고, ROI 추출·공간 재샘플링·앙코럼된 기하적 priors 및 엄격한 radial error(10mm) 실패 기준으로 품질을 관리했다. 또한 Fire/Gadgetron 인터페이스로 DICOM 스트리밍→GPU 추론→HTML 보고서 생성까지 지연을 줄여 스캔 진행 중 결과 제공을 목표로 했다.

- **Empirical Impact**: 다중 센터 500+ 데이터(독립 회고/전향 코호트)에서 자궁 Dice 0.82, fibroids Dice 0.80을 달성했으며 landmark는 평균 radial error 3.7mm로 보고됐다. 전향 배치에서는 자동 volumetry/바이오메트릭 정확도가 관찰자 간 변동과 유사했고, end-to-end 처리 시간은 70초 미만으로 실시간 보고가 가능함을 보였다. 결과적으로 자동화된 정량·시각화 기반의 재현성 높은 자궁 MRI 표준화를 촉진하며 임상 효율 향상 잠재력을 제시한다.



### AVOC: Enhancing Hour-Level Audio-Video Understanding in Omni-Modal LLMs via Retrieval-Inspired Token Compression (https://arxiv.org/abs/2606.24286)
- **Prior Approaches**: 기존 멀티모달 LLM은 오디오-비디오를 인코더로 처리한 뒤 LLM 문맥에 넣어 short-form 과제에서는 성과를 냈지만, hour-level로 길어지면 컨텍스트 윈도우 한계와 중복 정보로 인해 추론 품질이 급락한다. 컨텍스트 축소도 content-agnostic sampling(샘플링)이나 단순 truncation에 치우쳐 중요한 사건을 놓치거나 토큰 예산을 빠르게 소진하는 문제가 있었다. OmniZip/OmniSIFT 같은 압축 접근은 대개 한 모달리티가 다른 모달리티를 유도하는 비대칭 설계를 써서, 유도 신호가 희박하면 핵심 정보가 버려질 위험이 남았다.

- **Core Contribution**: 이 논문은 long-form audio-video 이해를 위한 Omni-modal LLM용 프레임워크 AVOC를 제안한다. 핵심은 멀티모달 token compression을 top-K retrieval 문제로 재정의하고, 고정된 컨텍스트 예산 안에서 질의에 유리한 토큰 부분집합을 “검색”하듯 학습하도록 만든 점이다. 또한 IR의 relevance(관련성), importance(중요도), diversity(다양성) 3축을 각각 오디오-비디오 이해에 맞는 메커니즘으로 구현해 하나의 통합 압축 파이프라인으로 엮었다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 긴 시퀀스에서 토큰 수를 줄이되 질의-관련 단서를 보존해야 하고, (2) top-K 선택이 비미분적이라 학습을 end-to-end로 이어가기 어렵다는 점이다. AVOC는 text-guided cross-attention으로 relevance를 점수화하고, temporal block 내 bidirectional video-audio cross-attention으로 query-agnostic importance를 보완하며, Temporal-Aware Maximal Marginal Relevance(TA-MMR)로 로컬 시간창 기준 중복을 억제해 diversity를 확보한다. 학습 단계에서는 Gumbel-Softmax 기반 differentiable top-KK 선택과 Straight-Through Estimator를 사용해 다음-토큰 예측 손실에서 압축 모듈의 projection 계층까지 그라디언트를 전달하고, TA-MMR은 추론에서만 greedy 재랭킹으로 적용한다.

- **Empirical Impact**: AVOC는 OmniVideoBench와 LVOmniBench 등 long-form 오디오-비디오 벤치마크에서 SOTA를 달성했으며, 2등 대비 평균 정확도가 각각 4.9점, 5.5점 더 높다. 특히 성능 향상이 비디오 길이에 따라 커져 초장 문맥에서 압축 모듈의 효과가 두드러진다. 또한 Audio-Video Needle-in-a-Haystack에서 최대 1시간 길이까지 질의 기반 “정밀 검색/국소화” 정확도를 견고하게 유지해, 모델이 hour-level 컨텍스트를 실사용 관점에서 다룰 수 있음을 실증한다.



### Automated Residual Plot Assessment With the R Package autovi and the Shiny Application autovi.web (https://arxiv.org/abs/2606.24236)
Comments:
          Published in Australian & New Zealand Journal of Statistics

- **Prior Approaches**: 기존 선형회귀 진단은 residuals vs. fitted values, Q-Q plot, residuals vs. leverage 같은 시각 검사를 기반으로 하며, 사용자의 주관에 따라 해석이 달라질 수 있다. lineup protocol은 null plot들 사이에 관측 plot을 섞어 주관성을 줄이지만, 인간 노동이 여전히 크게 필요하다. 이를 보완하려는 null 생성·비교 자동화 도구들이 있었지만, 대규모 적용에서 효율성과 일관성 한계가 남아 있었다.

- **Core Contribution**: 이 논문은 residual plot을 입력으로 받아 시각적 이상 정도를 수치화하는 VSS(visual signal strength)를 도입하고, 이를 computer vision 모델이 자동 예측하도록 만든다. 또한 자동화된 시각 추론 결과를 R 패키지 autovi와 웹 인터페이스 autovi.web(autovi.web)로 제공해 분석가가 p-value 수준의 결론을 빠르게 얻도록 한다. 결과적으로 수작업/lineup의 인력 부담을 줄이면서도 residual pattern의 유의성을 계산 가능하게 한다.

- **Technical Challenges**: 핵심 과제는 residual plot에서 “보이는 패턴”과 “모델 위반 정도”를 안정적으로 연결하는 표현을 정의하고 학습하는 것이었다. 논문은 VSS를 KL divergence 기반의 거리로 정식화해, 다양한 가정 위반을 합성한 데이터로 지도학습을 수행하고 residual plot 이미지를 입력받아 해당 거리를 근사하는 CNN을 사용한다. 추가로 null residual은 모델이 올바를 때의 기준 분포를 만들기 위해 시뮬레이션하고, bootstrapped residual은 재표집·재적합으로 VSS 변동성을 추정해 해석의 견고성을 함께 평가한다.

- **Empirical Impact**: autovi는 true residual plot의 VSS를 null과 bootstrapped 분포에 대해 비교해 pp-value(한쪽 p-value)를 산출하고, 필요 시 lineup 기반 시각 보조 자료도 제공한다. 예시에서는 이질분산처럼 보이지만 실제로는 데이터 크기/분포 효과인 경우 낮은 VSS로 잘 구분되고, 비선형성이나 구조가 강하게 포함된 경우에는 0.05 미만의 pp-value로 유의성을 포착한다. 또한 Poisson 같은 다른 glm으로의 확장 방법(자체 Keras model 학습 필요 조건 포함)을 제시해, 향후 진단 자동화의 실용 범위를 넓히는 데 의미가 있다.



### A Dual Edge Spatial Jacobian Image Graph for Interpretable Diabetic Retinopathy Grading (https://arxiv.org/abs/2606.24168)
- **Prior Approaches**: 기존 연구는 컬러 안저사진에서 DR을 이미지 수준 라벨로 예측하는 딥러닝 분류기를 주로 사용했지만, 임상 근거(병변 종류·위치·부담·혈관 형태)까지 정량화해 설명하긴 어렵다. Grad-CAM 계열은 시각적 히트맵을 제공하지만 병변 근거가 혈관 주변/특정 영역/정량 바이오마커와 어떻게 연결되는지는 답하기 어렵다. 반대로 oculomics처럼 혈관 형태와 질병 중증도 간 연관을 다루는 접근은 공간 배치를 잃어버린다는 한계가 있다.

- **Core Contribution**: 이 논문은 병변 근거가 혈관에 대해 어떻게 분포하는지, 그리고 병변 기반 임베딩이 AutoMorph 바이오마커에 어떤 민감도(반응) 기하를 보이는지에 초점을 둔 dual-edge spatial–Jacobian image graph를 제안한다. 각 안저 이미지를 그래프 노드로 보고, vessel–lesion 공간 관계(edge E12)와 embedding–biomarker 민감도/반응 기하(edge E34)를 분리된 의미 있는 두 엣지 계열로 구성한다. 두 엣지 계열을 lightweight two-token attention으로 융합해 DR 등급 예측뿐 아니라 그래프 토폴로지와 민감도 지표를 통한 해석/가설 생성에도 쓰도록 설계했다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 병변 근거 지도를 혈관 지도 및 영역 정보와 정합되게 결합하는 것과 (2) 병변 기반 embedding과 AutoMorph 정량 바이오마커 간 민감도 관계를 계산 가능한 형태로 만드는 것이다. 논문은 X1(혈관)과 X2(병변 evidence)를 공간 상호작용 채널로 구성해 vessel-인접/영역-겹침 정보를 엣지 E12로 만들고, X3(lesion-based contrastive embedding)에서 X4(AutoMorph biomarkers)로의 differentiable mapping 뒤에 local Jacobian을 계산해 엣지 E34를 구성한다. 마지막으로 두 브랜치의 경량 attention 융합을 통해 그래프 수준 표현을 만들면서도, Jacobian 기반 민감도 지표를 해석 목적에 맞게 정리했다.

- **Empirical Impact**: 내부 코호트 APTOS 2019 비증강 2,910장 실험에서 full graph는 5-class 0.8076 accuracy, 0.8312 quadratic weighted kappa, 0.5915 macro-F1을 달성했고 인접 등급 정확도도 0.9330으로 높았다. referable DR(이진)에서는 0.9055 accuracy, 0.9711 AUROC를 기록해, 인접 구간의 불확실성보다 임상적으로 중요한 screening 설정에서 강점을 보였다. 해석 관점에서는 FDR 보정 Spearman 경향 검정으로 병변 근거(예: neovascularisation·haemorrhage)와 혈관 calibre/density 및 영역-C 바이오마커(CRAE/CRVE, artery density 등)의 연관을 발견했으며, 이는 인과 주장보다는 가설 생성용 구조화된 표현 학습이라는 점에서 의미가 있다.



### NavWM: A Unified Navigation World Model for Foresight-Driven Planning (https://arxiv.org/abs/2606.24101)
Comments:
          13 pages, 5 figures, accepted to ECCV 2026

- **Prior Approaches**: 기존 시각 기반 내비게이션은 카메라 관측에서 행동으로 바로 매핑하는 반응형 정책이 많아, 미래를 보지 못해 myopic(단견) 의사결정과 mode collapse로 탐색이 불안정해지기 쉽다. World model을 쓰더라도 perception·generation·control을 분리 학습하는 모듈형 파이프라인이 공통의 시공간 동역학 시너지를 놓친다는 한계가 지적된다. 또한 단일 궤적을 예측해 멀티모달 미래를 충분히 다루지 못해 국소 최적에 갇히는 문제가 남아 있다.

- **Core Contribution**: 본 논문은 NavWM( Unified Navigation World Model )로, 지각·시각 생성·제어를 하나의 네트워크에서 통합 학습하는 시공간 일치형 내비게이션 world model을 제안한다. Latent World Tokens로 기하·의미의 구조적 선험을 압축하고, anchor-based multimodal trajectory forecasting으로 다양한 행동 후보를 생성해 폐루프(closed-loop) 계획에 필요한 시각적 foresight를 제공한다. 결과적으로 에이전트가 미래 비전 관측을 시뮬레이션하며 목표 정합성이 높은 경로를 고르는 방식으로 더 멀리 내다보는 계획을 수행한다.

- **Technical Challenges**: 핵심 기술적 난점은 (1) 장기 예측에 필요한 기하·의미 추상화를 안정적으로 학습하는 것, (2) 단일 궤적 예측에서 오는 mode collapse를 피하고 물리적으로 그럴듯한 멀티모달 미래를 뽑는 것, (3) 생성 모델이 예측된 행동의 오차에도 견디며 폐루프 계획으로 동작하도록 train–inference discrepancy를 줄이는 것이다. NavWM은 Depth Anything V2와 SAM에서 distill한 기하/의미 pseudo-label로 Latent World Tokens을 감독하고, scale ambiguity는 scale-invariant loss로 완화한다. 또한 Flow Matching 기반의 controllable visual generation과 두 단계 학습(teacher forcing 후 예측 샘플 조건 fine-tuning)을 결합해, world model planner가 잘못된 행동 가정에도 견고하게 미래를 시뮬레이션하도록 만든다.

- **Empirical Impact**: 실험은 Go Stanford, SCAND, RECON, HuRoN, Tartan Drive 등 로보틱스 데이터셋에서 수행됐고, PSNR이 14.17→17.34로 생성 품질이 유의미하게 개선됐다. Image Goal Navigation에서 성공률이 seen 66%→72%, 특히 unseen 환경 zero-shot에서 44%의 성공률을 보여 기존 대비 높은 성능을 달성했다. 또한 ablation과 다양성/정확도 분석에서 anchor 기반 멀티모달 예측이 모드 붕괴를 줄이면서도 trajectory 다양성(APD)을 높여, world model planner와의 궁합이 좋다는 점을 확인했다고 요약된다.



### Cyclic Denoising Reveals Ultrastable Memories in Diffusion Models (https://arxiv.org/abs/2606.24000)
Comments:
          22 pages, 7 main figures; supplementary material included. Supplementary movies available at the project webpage

- **Prior Approaches**: 확산모델에서 학습 데이터가 유출되는지 보려는 기존 추출공격은 대개 generate-and-filter처럼 프롬프트(또는 캡션)로 대량 생성한 뒤 사후 유사도/클러스터링/멤버십 추론 등으로 “기억” 후보를 걸러내는 방식이 많다. 무조건(unconditional) 모델에서는 학습셋을 알고 있거나 검색 가능한 벤치마크 규모에서 생성 샘플과 학습 데이터를 직접 대조해야 하는 경우가 흔하다. 다른 접근도 one-step denoising 신호, 보조 분류기, 추가 학습된 탐지기 등 외부 신호나 데이터 접근을 요구해 실제 적용 장벽이 남아 있다.

- **Core Contribution**: 이 논문은 cyclic denoising을 제안한다. 이는 고정된 sampler에서 “부분 노이징→역복원”을 한 사이클로 반복하되 노이즈 진폭 γ를 제어해, 샘플링을 시계열적(스트로보스코픽) 동역학으로 바꾸고 그 장시간 안정 상태를 “추출 신호”로 읽어낸다.

- **Technical Challenges**: 핵심 기술적 과제는 반복 노이징-디노이징에서 어떤 상태가 오래 버티는지(흡수 상태, limit cycle, basin hopping 등)와 그것이 실제 “기억된 예”와 어떻게 연결되는지를 정량화하는 것이다. 저자들은 사이클 종료 시점의 연속 상태 유사도(코사인 유사도)로 yielding-like 전이를 그리며, 특정 γ 구간에서 장시간 거의 완전 고정(흡수 에피소드) 또는 장수 트랩(깊은 attractor)이 나타나는 패턴을 포착한다.

- **Empirical Impact**: 실험은 Stable Diffusion v1.4와 픽셀공간 DDPM(CIFAR-10)에서 수행했으며, γ에 따라 로고·웹 템플릿 같은 단순 반복물에서 더 풍부한 기억 이미지로 넘어가는 안정성 스펙트럼이 일관되게 관측된다. 이 과정에서 수천 사이클 단위로 재생성되는 ultrastable attractor가 보고되며, 후보들은 프롬프트 없이도 sampler-level 제어만으로 드러나 “학습 데이터 기억 감사(memorization auditing)” 및 모델 핑거프린팅/개인정보·저작권 컴플라이언스 점검에 실용적 도구가 될 수 있음을 시사한다.



### 3D Masked Autoencoders are Robust Learners of Volumetric and Multimodal Cellular Representations for Microscopy (https://arxiv.org/abs/2606.23964)
Comments:
          Accepted at MICCAI 2026. Code available at: this https URL

- **Prior Approaches**: 형광 현미경의 self-supervised learning은 세포의 3D 구조에도 불구하고, 3D zz-stack을 2D max-projection이나 slice로 투영해 학습하는 경우가 많았다. Subcell, DINO4Cell, Cytoself 등 기존 접근은 깊이 정보를 버리거나(2D 투영) 단편적으로 처리해(슬라이스 평균) 공간 문맥 활용에 한계가 있었다. 또한 2D 기반 표현을 단일 단계로 end-to-end 학습하며 단백질 서열(예: ESM2)과의 정렬을 제대로 결합하는 데도 격차가 남아 있었다.

- **Core Contribution**: 이 논문은 2D masked autoencoder(MAE-2D)와 3D masked autoencoder(MAE-3D)를 동일한 아키텍처/학습 프로토콜 조건에서 체계적으로 비교해, 네이티브 3D 모델이 downstream 단일세포 태스크 성능을 일관되게 앞선다는 점을 보였다. 여기에 채널 cross-attention(CCA)과 frequency-domain 정규화(FFT loss)를 추가하고, 단백질 언어모델 ESM2 임베딩을 decoder에 주입해 cross-modal supervision이 3D 모델에서 특히 더 큰 이득을 준다는 시너지를 실험적으로 확인했다. 즉, 3D 구조 보존과 멀티모달 정렬을 동시에 추구하는 표현학습 프레임을 제안한다.

- **Technical Challenges**: 핵심 기술 과제는 3D 볼륨에서 마스킹 복원 학습을 안정적으로 수행하면서, 채널 간 상호작용과 깊이별 공간 문맥을 효과적으로 활용하는 것이다. 저자들은 (1) 채널별 토큰 스트림에 cross-attention을 두되 softmax 대신 sigmoid gating으로 주의 깊게 설계하고, (2) FFT 기반 손실을 3D 전 공간 축으로 확장하며, (3) MSE와 FFT 손실을 warm-up/ ramp-up으로 가중해 초반 학습 불안정을 완화했다. 또한 ESM2는 reconstruction 안정화 이후 투입하고, decoder에 단백질 토큰을 삽입한 뒤 visible token 위주로 symmetric InfoNCE를 적용해 이미지-서열 표현 정렬을 어렵지만 강하게 유도했다.

- **Empirical Impact**: OpenCell 데이터에서 protein-protein interaction 태스크(PPI)는 MAE-3D*가 ROC-AUC 0.865로 prior 대비 최대 +0.025 개선을 보였다. protein localization에서는 최상 모델이 AUC$_{micro}$ 0.952, F1$_{micro}$ 0.742를 달성했으며, 이전 접근 대비 각각 +0.003, +0.010 절대 개선이다. 특히 ESM2 정렬은 3D 모델에서 더 큰 폭으로 이득이 나타나며, 데이터가 상대적으로 적은 6k 볼륨 규모에서도 2D/기존 self-supervised 대비 강한 성능을 보여 3D 모델링과 멀티모달 정렬이 cellular imaging의 representation learning에서 실질적 임팩트를 가진다는 점을 시사한다.



### E-MRL: Cross-view Aligned Evidence-driven Multimodal Reinforcement Learning for Reliable 3D Tumor Analysis (https://arxiv.org/abs/2606.23888)
Comments:
          9 pages, 2 figures

- **Prior Approaches**: 기존 Vision-Language Model(VLM) 기반 3D CT 종양 리포트 생성은 주로 Supervised Fine-Tuning(SFT)이나 text-based Reinforcement Learning(RLHF)처럼 텍스트 정합성·진단 정확도에 보상을 주는 방식이 많았다. 이 접근은 시각 근거(어느 슬라이스를 봤는지) 자체에 대한 과정 수준 감독이 부족해, 입력에 없는 병변 디테일을 언어적 사전지식으로 그럴듯하게 만들어내는 visual hallucination이 발생하기 쉽다. 또 3D 분야에서도 시각 보상은 제한적이어서, 미스인 지역화 오류를 학습 신호가 제대로 벌주지 못하는 한계가 있었다.

- **Core Contribution**: 논문은 E-MRL(Evidence-driven Multimodal Reinforcement Learning)을 제안하며, 생성 과정을 diagnosis-localization-verification의 Markov Decision Process로 재구성한다. 모델이 전역 진단 리포트를 내는 동시에 “key evidence slice” 인덱스를 명시적으로 찾아, 해당 슬라이스를 다시 질의(verification)해 근거를 검증하도록 학습한다. 핵심은 cross-view consistency reward로, 전역 리포트에 적힌 속성과 로컬 슬라이스 재질의 결과가 의미적으로 일치할 때 추가 보상을 줘 텍스트-시각 정합을 강제한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 key slice 인덱스 선택이 이산적이라 end-to-end 최적화가 어렵다는 점과, 멀티모달 보상을 어떻게 안정적으로 설계할지였다. 이를 위해 슬라이스 선택과 텍스트 생성을 함께 다루는 hybrid action(토큰 생성/슬라이스 포인팅)으로 MDP를 구성하고, 종양 존재, fine-grained 속성(size·location·enhancement), 그리고 cross-view 일관성의 다중 보상을 조합했다. 또한 value network 없이 안정적인 정책 업데이트를 위해 GRPO(Group Relative Policy Optimization)를 사용하고, KL 페널티 및 기준 SFT 모델을 활용해 reward hacking을 줄이도록 설계했다.

- **Empirical Impact**: AbdomenAtlas3.0의 대규모 3D CT 종양 데이터에서 E-MRL은 SFT·RL 기반 경쟁군과 RadGPT를 포함한 여러 의료 VLM 대비 hallucination을 줄이고 진단 정확도와 속성 근거(특히 key slice hit rate)에서 일관되게 향상됐다. 보고된 성능은 세 종양 유형 평균 balanced-accuracy(B-Acc) 87.93%로, 이전 SOTA RadGPT 대비 16% 이상 높은 수준이며 전역-로컬 근거 정렬이 잘 된 사례도 제시됐다. 결과적으로 “과정 수준 시각 검증”을 갖춘 end-to-end 3D 의료 VLM의 신뢰도·해석 가능성을 높였다는 점에서 임상 보조용 자동 리포트 생성 연구에 의미 있는 진전을 제공한다.



### Ground Then Rank: Revisiting Knowledge-Based VQA with Training-Free Entity Identification (https://arxiv.org/abs/2606.23881)
Comments:
          Accepted by ACL 2026 Findings. Project page this https URL

- **Prior Approaches**: 기존 KB-VQA 대응은 multi-modal retrieval-augmented generation(MM-RAG) 흐름에서 검색 후 재랭킹을 통해 답을 뒷받침할 텍스트 구간을 찾는 방식이 주류다. 그런데 많은 방법이 entity 식별과 evidence(증거) 구간 선택을 한 번의 재랭킹 단계로 강하게 결합해, 엔티티는 맞지만 구간이 틀리거나 반대로 구간은 그럴듯하지만 엔티티가 틀리는 문제가 발생한다. 또한 학습 기반 멀티모달 재랭커를 붙이는 경우가 많아 비용과 데이터 의존성이 커진다.

- **Core Contribution**: 이 논문은 KB-VQA에서 필요한 grounding을 entity-level grounding과 section-level grounding의 두 병목으로 재정의하고, 이를 workflow 관점에서 명시적으로 분리하는 training-free Identify Before Answer(IBA)를 제안한다. 핵심 아이디어는 MLLM이 답을 생성하기 전에 ‘후보 엔티티 이름’들 중 고신뢰 엔티티를 먼저 고른 뒤, 그 엔티티로 좁혀진 범위에서 텍스트 re-ranker로 증거 구간을 고르는 파이프라인이다. 이를 통해 fine-tuning 없이도 정확도와 효율을 동시에 노린다.

- **Technical Challenges**: 가장 큰 기술적 어려움은 open-ended 엔티티 이름 생성 방식이 MLLM에 높은 불확실성을 요구해 실패가 잦다는 점이다. 저자들은 흥미로운 실험 관찰로, 모델에게 엔티티 이름을 자유 생성하게 하지 않고 candidate name shortlist에서 고르도록 하면 식별 정확도가 크게 올라간다고 보고 이를 tip-of-the-tongue 현상에 비유한다. 구현은 EVA-CLIP-8B로 시각적으로 후보 페이지를 먼저 뽑고, Qwen-2.5-VL-7B-Instruct로 후보 엔티티를 1차 선별한 뒤 BGE 같은 off-the-shelf textual re-ranker로 evidence를 재랭킹하는 방식이다.

- **Empirical Impact**: Encyclopedic-VQA(E-VQA)와 InfoSeek에서 IBA는 fine-tuned multi-modal re-ranking 기준선을 전반적으로 능가하며, training 및 추론 복잡도를 줄이면서도 성능을 끌어올렸다고 보고한다. Recall@1(정답 엔티티 우선순위)에서 InfoSeek은 58.4%로 EchoSight의 53.1%를 상회하고, E-VQA에서는 약간 낮은 Recall@1(35.5%)을 보이지만 다운스트림 답변 품질은 여전히 경쟁적 수준을 유지한다. 분석에 따르면 개선은 단순히 엔티티 식별이 좋아진 데 그치지 않고, 정답 엔티티가 고정된 뒤 더 유익한 evidence를 선택하게 된 효과가 함께 작용한다.



### Machine Learning Modeling for Real-Time Melt Pool Monitoring in Laser Powder Bed Fusion Additive Manufacturing: A Hybrid Approach (https://arxiv.org/abs/2606.23851)
- **Prior Approaches**: LPBF에서 용융풀 이상을 잡기 위한 AI/ML 연구는 주로 딥러닝 단독 분류에 의존해 왔으나, 데이터가 제한적인 공정 환경에서는 성능-일관성 확보가 어렵다. 또한 공장 현장 배치를 전제로 하면 추론 지연이 길어지는 경우가 많아 하드웨어 제약(오픈 아키텍처 장비의 CPU/GPU 성능)과 충돌한다. 기존 접근 중 일부는 전이학습을 쓰더라도 공정 변동을 충분히 반영한 데이터 구성과 배치 가능성까지 함께 최적화하기는 제한적이었다.

- **Core Contribution**: 본 논문은 LPBF 실시간 모니터링을 위해 정상/이상 용융풀을 이진 분류하는 프레임워크를 제안하고, 데이터 제한 환경에서 성능을 끌어올리는 실용적 조합을 제시한다. 특히 사전학습 CNN(ResNet50, EfficientNetB0, MobileNetV2)으로 특징을 뽑은 뒤 Random Forest에 결합한 hybrid 방식을 도입해, 순수 딥러닝보다 정확도와 배치 효율을 동시에 노린다. 다양한 하드웨어/제어 요구 조건 하에서 정확도와 추론 지연을 함께 평가한다는 점이 기여의 핵심이다.

- **Technical Challenges**: 기여를 실현하는 기술적 난제는 (1) NIST AMMT에서 수집한 제한된 1,200장 데이터로 공정 변동을 일반화해야 하고, (2) 공장 바닥에서 가능한 수준의 추론 지연과 CPU/GPU 사용량을 만족해야 한다. 이를 위해 80/20 학습-테스트 분할과 90/10 검증 분할을 적용하고, 라벨 보존 중심으로 리사이징·정규화·데이터 증강을 통해 공정 변동을 모사했다. 또한 transfer learning 백본들을 비교하고, hybrid( EfficientNetB0 feature embeddings + Random Forest )와 baseline(원시 픽셀 + Random Forest)을 나란히 벤치마크해 배치 가능성과 정확도의 균형을 맞췄다.

- **Empirical Impact**: Nickel superalloy 625 데이터에서 hybrid EfficientNetB0-plus-Random Forest는 테스트셋 기준 F1 0.9451, 정확도 0.9458, AUC 0.9904를 달성하며 이미지당 1.15 ms 수준의 sub-millisecond 추론을 유지했다. 반면 순수 딥러닝 모델은 정확도는 낮아지면서 추론 시간은 유의미하게 길어져 실시간 모니터링 요건을 충족하기 어려웠다. 이 결과는 data-limited 적층제조 환경에서 ‘pre-trained convolutional features + classical ensemble’이 견고하면서도 계산 효율적인 실시간 이상 탐지 경로가 될 수 있음을 보여준다.



### Performance and Interpretability of Convolutional, Transformer, and Hybrid Deep Learning Models in Colorectal Histology Classification (https://arxiv.org/abs/2606.23744)
- **Prior Approaches**: 그동안 대장 조직병리(histopathology) 분류에는 주로 CNN(Convolutional Neural Network)이 중심이었고, 최근에는 transformer 기반 및 하이브리드 모델이 성능을 끌어올린다는 보고가 늘었다. 다만 Kather 대장 조직 타일 데이터셋에서 CNN·transformer·하이브리드 전반을 같은 조건으로 폭넓게 비교한 종합 벤치마크는 부족했다. 또한 transfer-learning/ fine-tuning 프로토콜과 평가 지표가 연구마다 달라 직접 비교의 신뢰도가 제한적이었다.

- **Core Contribution**: 이 논문은 ImageNet-pretrained CNN·transformer·하이브리드 12개 모델을 Kather 대장 조직병리 데이터셋(8개 조직 클래스, 5,000 타일)에서 표준화된 transfer-learning 및 fine-tuning 절차로 공정 비교한다. 분류 정확도뿐 아니라 정밀도, 재현율, 특이도, F1-score, ROC-AUC, Cohen's kappa, Matthews correlation coefficient 등 다각도의 성능을 함께 제시해 모델 간 관점별 우위를 정리한다. 이를 통해 대장 조직 분류에서 transformer 계열의 예측 성능 상한과 CNN의 효율적 절충점을 함께 벤치마킹한다.

- **Technical Challenges**: 핵심 기술적 과제는 서로 다른 아키텍처(CNN/transformer/하이브리드)를 동일한 학습 파이프라인과 평가 기준으로 맞추는 데 있었다. 논문은 모든 모델에 대해 동일한 transfer-learning·fine-tuning 프로토콜을 적용하고, 여러 지표로 성능을 교차 검증해 단일 지표에 의존한 결론을 피했다. 또한 클래스별 난이도(특히 Complex Stroma)까지 분석해 평균 성능만으로는 놓치기 쉬운 편향을 점검했다.

- **Empirical Impact**: 실험 결과, 모든 모델이 높은 분류 성능을 보였고 정확도는 93.2%~97.1% 범위였다. EVA-02가 97.1% 정확도와 97.0% F1-score로 최고 성능을 기록했으며, ViT-B/16이 그 뒤를 이었다; CNN 중에서는 ResNet34(96.4%), ConvNeXt-Tiny(96.3%)가 경쟁력 있는 결과를 보였다. 전반적으로 transformer가 평가 전반에서 가장 강했지만, CNN 상위 모델과의 격차는 비교적 작아 정확도-복잡도 트레이드오프 관점의 선택 기준을 제공한다. Kather 대장 조직병리 분류에서 대표 딥러닝 패러다임을 한 번에 정리한 벤치마크로, 향후 모델 설계와 실험 재현성에 직접적으로 활용될 의미가 크다.



### Systematic Exploration of 4-Expert Heterogeneous Mixture-of-Experts via Automated Pipeline Search (https://arxiv.org/abs/2606.23739)
Comments:
          8 pages, 2 figures

- **Prior Approaches**: MoE는 입력을 선택적으로 전문가 네트워크에 라우팅해 계산을 줄이면서 성능을 높이는 구조로, CNN·Vision Transformer 등에서 다양한 변형이 등장했다. 그러나 LEMUR 생태계에서의 MoE 실험은 주로 수작업으로 소수 구성을 설계·평가하는 방식에 머물러 조합적으로 큰 heterogeneous expert 조합 공간을 충분히 탐색하기 어려웠다.

- **Core Contribution**: 이 논문은 LEMUR 기반 heterogeneous 4-Expert Mixture-of-Experts(MoE4) 아키텍처를 대규모로 자동 탐색하는 파이프라인을 제안한다. 결정적 code-assembly generator가 LLM 호출 없이 LEMUR의 서로 다른 4개 패밀리를 결합해 forward-pass까지 검증된 MoE4 모델을 생성하고, 다단계 검증(문법 검사, CPU 해상도 프루브, MD5 중복 제거)과 캠페인 자동화로 대규모 실험을 안정적으로 수행한다.

- **Technical Challenges**: 핵심 난제는 (1) 조합을 대량 생성해도 깨지거나 중복된 모델이 GPU 자원을 낭비하지 않게 하는 것과 (2) 256×256 해상도에서 4개 전문가를 함께 학습할 때 CUDA out-of-memory(OOM)가 구조적으로 대량 발생한다는 점이다. 저자들은 컴파일/CPU 프루브로 사전 걸러내고, 캠페인 상태를 JSON으로 지속 저장해 장애에도 재개 가능하게 했지만, OOM은 해상도·활성메모리 피크 특성 때문에 완전히 제거되지 않아 향후 메모리 인지 pre-filtering이나 해상도 저감 등을 제안한다.

- **Empirical Impact**: 28일 동안 RTX 4090에서 4,463개를 만들고 1,021개를 성공 평가했으며, 탐색 공간 분석에서는 sorted 결정적 열거 방식 때문에 모든 탐색이 AirNet에 고정(이론 조합 23,751개 중 4.8%만 커버)되는 coverage bias가 발견됐다. AirNet-고정 범위 내에서는 ShuffleNet과 MobileNetV3가 co-produce 앙상블에서 가장 높은 평균 정확도(각각 최대 0.632)를 보인 반면, FractalNet과 MNASNet은 낮은 수율/성능 저하로 향후 풀에서 제외할 근거가 제시됐다. 또한 최고 단일 조합(AirNet+AlexNet+DPN68+ResNet)은 CIFAR-10에서 단 1 epoch 스크리닝 기준 Top-1 68.0%를 달성해 자동 조립이 수작업 튜닝 없이도 경쟁력 있는 후보를 만들 수 있음을 실증했다.



### MultiMem: Measuring and Mitigating Memorization in Multi-Modal Contrastive Learning (https://arxiv.org/abs/2606.22220)
Comments:
          Accepted at The 19th European Conference on Computer Vision (ECCV), 2026

- **Prior Approaches**: 기존 memorization 연구는 SL/SSL 또는 단일·양자(예: image-text) 설정에 맞춰 설계돼, 여러 modality가 함께 얽히는 multi-modal contrastive learning의 전역 memorization을 제대로 포착하기 어렵습니다. CLIPMem이나 déjà vu 같은 방법도 특정 두 modality 중심이라 video·audio 같은 추가 modality까지 일반화하기가 쉽지 않습니다.

- **Core Contribution**: 논문은 multi-modal contrastive learning에서 memorization을 정량화하는 최초의 프레임워크인 MultiMem을 제안합니다. MultiMem은 leave-one-out로 ‘해당 multi-modal sample을 뺀 모델’과 ‘전체로 학습한 모델’의 cross-modal consistency 차이를 측정해, 모든 modality가 함께 등장하는 전역 memorization을 평가합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “여러 modality 간 정합(semantic alignment)”을 기준으로 representation 품질을 proxy로 잡고, 샘플 삭제에 따른 변화를 안정적으로 비교하는 것입니다. 논문은 cross-modal consistency를 표현공간에서 모든 modality 쌍의 일관성과 비관련 샘플(held-out) 간 차이로 정의하고, 임의 augmentation 기대값까지 포함해 metric의 안정성을 확보했습니다.

- **Empirical Impact**: 여러 contrastively 학습한 모델(AudioCLIP, AVT-CLIP, AVIT-CLIP, VideoCLIP 등)에서 가장 큰 memorization 영향 요인은 cross-modal semantic misalignment이며, text가 주도하고 video·image·audio가 뒤따르는 경향을 보였습니다. 또한 MultiMem 기준으로 상위 memorized 샘플에 modality 전반에 걸친 targeted noise-based augmentation을 적용하거나, 학습 후 고 memorization 샘플을 제외해 fine-tuning하면 memorization을 최대 약 20% 줄이면서 retrieval/zero-shot/downstream 성능도 각각 최대 8%/10%/4%까지 개선됩니다.



New uploads on arXiv(cs.AI)

### OpenThoughts-Agent: Data Recipes for Agentic Models (https://arxiv.org/abs/2606.24855)
- **Prior Approaches**: 기존 오픈 데이터 노력(SWE-Smith, SERA, Nemotron-Terminal, OpenSWE)은 대개 단일 벤치마크(예: SWE-Bench, Terminal-Bench) 중심으로 설계되어 에이전틱 작업 전반으로의 일반화 학습 전략을 파악하기 어렵다. 또한 SFT와 RL 데이터 큐레이션이 각각 중심을 이루면서 두 단계의 교차 설계(무엇을 어떻게 섞고 정제할지)가 충분히 체계화되지 못했다.

- **Core Contribution**: 이 논문은 OpenThoughts-Agent(OT-Agent)로, 에이전틱 모델의 supervised fine-tuning(SFT)을 위한 “완전 공개형” 데이터 큐레이션 파이프라인을 제시한다. 6단계 파이프라인을 대상으로 100회+ 통제적 ablation을 수행해, 작업 소스(task sources)와 다양성(diversity)이 성능에 미치는 영향이 크다는 실험적 결론을 도출했다. 이를 바탕으로 Qwen3-32B를 10만 개 예제로 fine-tune해 7개 에이전틱 벤치마크 평균 정확도 44.8%를 달성하고, Nemotron-Terminal-32B 대비 3.9%p 개선을 보고한다.

- **Technical Challenges**: 에이전틱 SFT 데이터는 (task, trajectory) 쌍의 품질이 성능을 좌우하므로, 작업 지시문 생성/혼합, teacher(롤아웃 생성 모델), 그리고 롤아웃 필터링(예: 턴 수, 타임아웃/서브에이전트 제거) 같은 선택지가 결과에 큰 변동을 만든다는 점이 핵심 기술 난관이다. 저자들은 파이프라인 각 단계를 독립 ablation하고, 지표 비교를 위해 벤치마크별 z-score 표준화를 사용했으며, 실행 트레이스의 turn이 길수록(≥5턴) 성능이 좋아지고 토큰 예산을 맞춰도 그 이득이 유지된다고 검증했다.

- **Empirical Impact**: OpenThoughts-Agent-v2로 만든 100K 학습셋은 compute-controlled 비교에서 다른 오픈 데이터셋을 학습 크기 전 구간에서 능가하는 강한 스케일링 성질을 보였고, Terminal-Bench 2.0에서 26.2%, SWE-Bench Verified에서 54.0%를 기록했다. RL 측면에서는 8B 스케일에서 pymethods2test 등 다양한 소스를 대상으로 데이터 소스 ablation을 수행하고, SFT+RL이 SOTA 8B baseline 대비 평균 18%p 내외 개선을 만든다고 보고한다. 데이터/파이프라인/모델을 공개해 에이전틱 모델 학습 연구의 재현성과 확장성을 높인다는 점에서 분야 임팩트가 크다.



### World Models in Pieces: Structural Certification for General Agents (https://arxiv.org/abs/2606.24842)
Comments:
          30 pages, camera-ready version in ICML 2026

- **Prior Approaches**: 기존 연구는 긴 지평에서의 안정적 의사결정을 위해 전역적으로 유사 최적 성능을 보장하는 “universal” 가정을 두고 내부 world model을 추정하거나 검증하려는 흐름이 있었다. 하지만 big-world 환경에서는 실패가 소수의 병목 전이(critical bottleneck transition)에 집중되고, 나머지는 경로에서 거의 활용되지 않기 때문에 worst-case 중심의 균일 보증이 지나치게 거칠다. 또한 단일 약한 전이가 최악 보증을 지배하면서, 실제로는 성공 궤적에서 무관한 오류를 “실패”로 과대평가하는 문제가 남는다.

- **Core Contribution**: 이 논문은 general agent가 전역적으로 universal하지 않다는 점을 형식화한 뒤, 전이(transition) 국소 구간에서의 “structural certification”을 제안한다. 구체적으로 bounded goal-conditioned 성능을 기준으로, 에이전트 내부 world model의 특정 엔트리(전이 확률)에 대한 entry-wise 보장을 도출한다. 즉, 긴 지평 planning이 신뢰되는 전이 조각만 골라 “인증 가능한 배치”가 가능하다는 틀을 제공한다.

- **Technical Challenges**: 핵심 기술적 어려움은 전역 near-optimality가 불가능한 상황에서, 에이전트의 행동이 어떤 전이 조각을 정확히 반영하는지 분리해 내야 한다는 점이다. 이를 위해 deep compositional goals로 목표들을 전이별로 격리하는 filtering 알고리즘을 구성하고, 인증된 전이에 대해 에이전트의 행동이 유도하는 전이 확률 추정 P^s_{s'}(a)\hat{P}_{ss'}(a)의 오차가 O(1/n)+O(δ) 범위로 수렴함을 보인다(δ≪1이면 δ 항이 지배적으로 작아짐). 또한 인증 세트 밖에서는 일관된 world model을 구성하더라도 비일치가 불가피하다는 하한( tight limit )을 함께 제시해, 국소 인증의 한계를 명확히 한다.

- **Empirical Impact**: 정량 보장은 전역 universal 실패율을 요구하지 않아도, 특정 전이에 한정하면 에이전트 내부 예측이 구조적으로 정렬(alignment)된다는 점을 뒷받침한다. 결과적으로 long-horizon 계획이 실패하는 원인을 “전체 모델”이 아니라 “인증된 전이 조각”의 신뢰성으로 국소화할 수 있어, general agent의 certifiable deployment 관점에서 실용적 가치를 제공한다. 특히 small-δ(소수 병목 실패) 영역에서는 상계가 타이트하게 작동해, 필요한 horizon n을 줄이거나도 유효한 전이 식별을 가능하게 한다는 메시지를 준다.



### Matching Tasks to Objectives: Fine-Tuning and Prompt-Tuning Strategies for Encoder-Decoder Pre-trained Language Models (https://arxiv.org/abs/2606.24841)
- **Prior Approaches**: 기존 prompt-based learning은 주로 특정 pre-training 목표나 고정된 학습/추론 포맷에 의존해 성능이 갈리곤 했다. encoder-decoder 계열에서 generation·question answering을 다루더라도, 과제에 맞는 pre-training 목적(objective)을 체계적으로 고르는 절차가 부족했다.

- **Core Contribution**: 이 논문은 다양한 pre-training objective가 encoder-decoder pre-trained language model의 생성 및 QA 성능에 어떻게 영향을 주는지 분석하고, 과제에 맞는 objective를 자동으로 선택·적용하는 MTO(Match Task to Objective) 프레임워크를 제안한다. 또한 pre-training 목표와 fine-tuning 적응 단계에 정합되도록 템플릿을 설계해 commonsense 지식 검색·완성에서 성능을 끌어올린다.

- **Technical Challenges**: 핵심 난제는 (1) 주어진 task에 어떤 objective가 맞는지 식별하고, (2) 그 objective에 맞게 task 데이터를 unsupervised 방식으로 준비하며, (3) pre-training-적응-템플릿 정합을 맞추는 것이다. 연구진은 MTO로 objective를 고르고, 해당 objective에 기반한 task 데이터 준비 방법과 objective와 정렬되는 fine-tuning 템플릿을 함께 설계해 few-shot에서의 적응 효율을 높였다.

- **Empirical Impact**: 정렬 전략은 few-shot 환경에서 기존 방식 대비 120%를 넘는 성능 향상을 보였고, 관련 연구들과 비교해도 일관되게 우수했다. full-dataset에서도 baseline을 능가하며, prompt-tuning까지 확장해 soft prompt engineering과 최적화에 실질적인 가이드를 제공하는 것으로 확인됐다.



### Grading the Grader: Lessons from Evaluating an Agentic Data Analysis System (https://arxiv.org/abs/2606.24839)
- **Prior Approaches**: agentic 데이터 분석 시스템 평가는 단일 턴 LLM 응답보다 어려운데, 그 이유는 코드·실행 로그·중간 통계·진단이 섞인 “풍부한 출력”에서 정답 구간을 뽑아내야 하기 때문이다. 기존 DSGym의 exact_match 같은 엄격한 비교는 예를 들어 0.2와 0.20을 서로 다른 값으로 처리해 구조적으로 false negative를 만들며, LAMBDA 같은 시스템의 출력 형식 변동은 자동 채점의 취약점이 된다. 또한 자동 그레이더는 오답을 “진짜 불일치”로 착각할 수 있어, 그레이딩 아티팩트(채점 실패)를 분리해내는 접근이 필요해졌다.

- **Core Contribution**: 이 논문은 agentic data analysis 출력에 대해 자동 그레이더가 실제로 얼마나 신뢰 가능하게 평가하는지 계량하고, 품질을 높이는 실전 전략을 제시한다. 핵심은 세 단계 human-AI 그레이딩 cascade(엄격 regex 기반, LLM 기반 lenient, 스니펫 기반 인간 점검)로 genuine disagreement과 grading artifacts를 분리하려는 설계다. 또한 keyword-anchored 정답 추출과 nudge(답 형식 템플릿 큐) 메커니즘을 결합해 “채점 가능한 출력”을 만들고 정답 추출 안정성을 개선한다.

- **Technical Challenges**: 첫째, 에이전트 출력에서 정답이 포함된 구간을 안정적으로 찾아야 하는데, 숫자가 너무 많고(특히 범주형) 형식이 불규칙하면 마지막 숫자 휴리스틱 같은 파서는 쉽게 실패한다. 둘째, 엄격 비교는 파싱/포맷 차이에 취약하고, LLM lenient은 문맥 기반 추출은 잘하지만 환각·드리프트 같은 위험이 있어 대규모 스케일에서 false positive 감지가 어렵다. 저자들은 (1) 키워드 앵커 점수로 답 영역을 찾는 keyword-anchored 파이프라인, (2) 비생성(non-GenAI) 층으로 파싱-기반 엄격 채점의 신뢰도를 올리고, (3) 생성(GenAI) 층으로 형식 변동을 흡수하는 lenient 그레이딩을 조합해 실패 프로파일이 다른 두 방식을 함께 쓴다. 더불어 nudge를 “원래 질문 재주입” 없이 답 형식만 반복해도 되는지 실험해, 재주입은 이 과제군에선 이득이 없음을 보인다.

- **Empirical Impact**: QRData의 수치 단일 스칼라 153개 과제를 대상으로 한 실험에서, 두 자동 그레이더 모두 관측된 precision이 100%(false positive 0/70)로 매우 높게 나타났다. lenient 그레이더는 human 라벨 대비 recall 97%를 달성했고, keyword-anchored 추출은 마지막 숫자 휴리스틱 대비 strict-grader recall을 60%p 끌어올렸다. nudge는 grading run success를 36%에서 97%로, lenient-pass 비율을 16%에서 46%로 크게 개선했으며, 변수 type(범주형/연속형/혼합)이 그레이딩 파이프라인 동역학과 결과 등급에 가장 일관되게 연관된 공변량으로 관찰됐다. 결론적으로 이 연구는 agentic 데이터 분석 평가에서 “채점 아티팩트를 분리하고, 추출-채점-인간 점검을 캘리브레이션하는” 실무 표준에 가까운 접근을 제공한다.



### Accuracy and Satisfaction in Multi-Turn LLM Dialogues for NFR Assessmen (https://arxiv.org/abs/2606.24834)
Comments:
          9 pages, 5 figures. Accepted to SIGDIAL 2026 (27th Annual Meeting of the Special Interest Group on Discourse and Dialogue)

- **Prior Approaches**: 기존 LLM 코드 어시스턴트 평가는 HumanEval, SWE-bench처럼 단일 턴의 기능적 정답(functonal correctness)에 초점이 맞춰져 있다. 다중 턴 대화에서의 정확도·대화 품질, 특히 비기능 요구사항(NFR)을 둘러싼 협업적 추론의 검증은 충분히 다뤄지지 못했다. 또한 대화 만족도를 직접 모델링하는 평가 틀(PARADISE 계열)은 소프트웨어 개발 LLM 대화 평가에 거의 적용되지 않았다.

- **Core Contribution**: 이 논문은 HIPAA 규제 준수라는 규제형 NFR 도메인에서, LLM 기반 에이전트와 개발자 간 다중 턴 대화의 정확도와 상호작용 품질을 함께 평가한다. 148개의 HIPAA-derived NFR에 대해 iTrust 코드베이스 기반의 전문가 ground truth(요구사항 만족 수준, 이유, 코드 로컬라이제이션)를 구축하고, 개발자들이 Copilot 에이전트의 평가에 동의하는 정도와 전문가 정답 대비 정확도를 분리해 측정했다. 또한 PARADISE 프레임워크를 적용해 사용자 만족도에 영향을 주는 대화 특성을 회귀 모델로 찾아냈다.

- **Technical Challenges**: NFR은 모호하고 문맥 의존적이며 프로그램의 여러 지점을 동시에 건드리기 때문에, 단순 정답 여부가 아니라 ‘만족 여부–이유–어디에 반영됐는지’를 모두 정교하게 비교해야 했다. 저자들은 만족 수준은 4분류 F1로, 이유는 BERTScore 등 의미 유사도로, 코드 로컬라이제이션은 파일·라인 집합의 precision/recall/F1로 각각 다차원 평가를 설계했다. 이어 사용자 만족도는 PARADISE의 성능함수 개념으로 추정하되, 대화 턴 라벨링 신뢰도를 높이기 위해 Cohen’s kappa를 단계적으로 검증하고 유의한 예측변수만 남기는 절차(후진 제거)를 적용했다.

- **Empirical Impact**: 실험에서 개발자들은 에이전트 응답에 91~94% 수준으로 ‘동의’했지만, 전문가 ground truth 대비 정확도는 낮아 요구사항 만족 수준 F1=0.381, 이유( BERTScore ) 기반 F1=0.520, 코드 로컬라이제이션 F1=0.203을 보였다. 즉, 그럴듯한 설명과 커뮤니케이션 품질이 높은 경우에도 규제 NFR 판단 자체는 오판일 수 있음을 시사한다. 사용자 만족도 회귀 결과에서는 verbose한(장문) 응답과 정보 제공 턴 수가 만족도를 낮추는 반면, proactive 상호작용 턴 수는 만족도를 높이는 것으로 나타나, NFR 평가 대화 설계의 실질적 가이드가 제시됐다.



### Difference-Making without Making a Differenc (https://arxiv.org/abs/2606.24832)
Comments:
          Preprint

- **Prior Approaches**: 저자는 7편에 걸쳐 실제 인과(actual causation)에 대한 7가지 정의를 제시하고, 이를 사실적 차이 만들기(factual difference-making), 반사실적 차이 만들기(counterfactual difference-making), 정칙성 기반(regularity-based)이라는 3계열 경쟁적 설명으로 분류했다고 설명한다. 그러나 이 분류가 정말로 서로 다른 구별을 제공하는지에 대한 검증이 부족하다는 점이 문제로 제기된다.

- **Core Contribution**: 이 논문은 가장 최근의 사실적 차이 만들기 정의가, 결과적으로 세 가지 유형(사실적/반사실적/정칙성 기반)을 모두 포괄하게 됨을 보이며, 따라서 세 유형 간의 구분이 의미 있는 차이를 만들지 못한다는 결론을 제시한다. 나아가 새로 제안된 해석을 나머지 6개 계정과 주요 사례들에서 비교해, 기존 7개 계정 모두가 흔들린다고 주장한다.

- **Technical Challenges**: 핵심 난제는 인과 정의가 실제로는 어떤 논리적 구조를 갖는지—즉 차이 만들기 방식이 반사실적 요소나 정칙성 요소까지 자연스럽게 포함하는지—를 엄밀한 예시와 사례 대조로 드러내는 것이다. 논문은 결정적 예들을 통해 한 정의가 분류 체계의 여러 축을 동시에 만족하는 현상을 보여, 유형 구분이 이론적 분해력을 갖지 못함을 설득한다.

- **Empirical Impact**: 실증 실험보다는 개념적·사례 기반 반박의 형태로, 기존 분류 및 정의들의 타당성을 동시에 약화시키는 영향이 크다. 실제 인과의 계정들이 ‘서로 다른 유형’으로 정리될 수 있다는 관점이 흔들리면서, 이후 논의는 더 강한 판별 기준과 정의의 정합성 검증에 초점을 옮겨야 한다는 신호로 읽힌다.



### Solving Inverse Problems of Chaotic Systems with Bidirectional Conditional Flow Matching (https://arxiv.org/abs/2606.24824)
Comments:
          50 pages, 17 figures

- **Prior Approaches**: 혼돈 동역학의 역문제(최종 관측으로부터 초기조건 추론)는 해의 비유일성·불안정성, 시간역추적 시의 정보 손실, 그리고 역적분 과정의 오차 누적으로 인해 크게 풀리지 못했다. 기존 수치 역적분은 비혼돈에서는 안정적이지만, 혼돈에서는 지수적으로 퍼지는 섭동 때문에 장기 horizon에서 성능이 급격히 무너진다. 딥러닝 기반 연구도 주로 예측 같은 순방향 과제에 치우쳐, 혼돈 역문제를 분포 수준에서 다루는 접근은 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 초기-최종 상태 분포 간의 양방향 매핑을 학습하는 Bidirectional Conditional Flow Matching(Bi-CFM)으로 혼돈 역문제를 정면으로 다룬다. Bi-CFM은 혼돈 진화의 확률적 성격을 분포 학습으로 반영하고, forward/backward 동역학을 함께 맞춰 시간에 따른 오차 누적을 완화한다. 또한 보존법칙이 있는 경우 Conservation-constrained Bi-CFM(CBi-CFM)으로 확률 흐름 전체가 보존 매니폴드(예: 에너지 보존)에 머물도록 제약을 추가한다.

- **Technical Challenges**: 핵심 난제는 (1) 역문제가 ill-posed라 단일 궤적보다 분포로 비교해야 하고, (2) 혼돈 영역에서 역적분이 시간 축 오차를 반복적으로 누적해 불안정해진다는 점, (3) 보존법칙이 존재할 때 추론 분포가 물리 제약을 위반하지 않게 해야 한다는 점이다. 논문은 Conditional Flow Matching(CFM) 기반으로 end-to-end 매핑을 학습해 시간 방향 반복 오차를 줄이고, 양방향 일관성으로 forward에서의 오류 증폭을 억제한다. CBi-CFM은 보존량이 정의하는 매니폴드의 접평면에서 velocity field가 움직이도록 만들어, 샘플링·학습 경로 전체가 보존 제약을 따르도록 구성한다.

- **Empirical Impact**: Lorenz, Circuit, Lorenz 96 등 대표 혼돈 시스템에서 Bi-CFM은 기준선 대비 5개 분포 수준 지표를 개선하고, 역적분 대비 2자릿수 이상 속도 향상도 보고한다(특히 긴 horizon에서 안정성 우위). 3-body 행성 산란·충돌 문제에서는 관측 불가능 구간이 생기는 정보 손실 상황에서도 CBi-CFM이 보존법칙(에너지) 준수성을 강화하며, 보존 오차가 ground truth 수준과 유사한 범위로 나타난다. 마지막으로 관측 기반 구상성단(globular cluster) 10 Gyr급 장기 진화 역문제에서도 Bi-CFM이 최신 Monte Carlo 대비 관측 프로파일(SBP/VDP)에 더 잘 부합해, 장기 실세계 혼돈 역문제의 확장 가능한 해결 경로를 제시한다.



### Assessing Distribution Shift in Human Activity Recognition for Domain Generalization (https://arxiv.org/abs/2606.24781)
Comments:
          22 pages with references

- **Prior Approaches**: 기존 HAR 연구는 학습과 테스트 데이터의 분포가 비슷하다는 가정에 기대는 경우가 많았고, 현실의 디바이스/센서 이질성과 맥락 변화가 이를 깨면서 성능 저하가 발생해 왔다. 이를 보완하려고 transfer learning이나 domain adaptation 같은 접근이 널리 쓰이지만, 실제 센서 기반 환경에서는 타깃 도메이터의 데이터(라벨/비라벨)를 구하기 어렵다는 한계가 있다. domain generalization은 새로운 도메이터 데이터 접근 없이 일반화하는 목표지만, HAR에서 어떤 분포이동 요인이 일반화를 특히 어렵게 만드는지, 그리고 알고리즘이 그 특정 요인들에 얼마나 취약한지에 대한 체계적 평가는 부족했다.

- **Core Contribution**: 논문은 센서 기반 HAR에서 발생하는 분포이동을 4가지 원인(디바이스 타입, 센서 부착 위치, 샘플링 레이트, 사용자 행동 변화)으로 나눠 체계적으로 평가한다. 또한 각 분포이동이 독립 도메인 간에 공유되지 않는 고유한 특징을 만든다는 점을 데이터 수준에서 정량화해, “왜” 일반화가 깨지는지에 대한 실마리를 제공한다. 마지막으로 uniform HAR 기반 distribution shift benchmark을 제안하고, 최대 28개의 domain generalization 방법을 동일한 평가 틀에서 폭넓게 비교한다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 분포이동 요인들이 실제 성능 저하에 어떻게 기여하는지 분해해 측정하고, (2) 그 특정 요인들을 재현하는 공정한 벤치마크를 구성하며, (3) 공통 알고리즘군이 그 유형별로 얼마나 일반화에 실패하는지 확인하는 것이다. 저자들은 분포이동을 diversity shift와 correlation shift의 2차원으로 추정하고, 이를 위해 도메인을 구분하는 분류 네트워크를 학습해 shift를 계산한다. 또한 디바이스/위치/레이트 변화는 공개 데이터와 다운샘플링/도메인 분할로 만들고, 사용자 행동 변화는 IRB 승인 저글링 데이터 수집으로 별도 도메인을 구성해 1:1로 비교 가능하게 처리한다.

- **Empirical Impact**: 실험 결과, 네 가지 분포이동 모두에서 diversity shift가 지배적으로 나타나 도메인 간 공유되지 않는 고유 특징이 존재함을 보여준다. domain generalization 방법들은 empirical risk minimization 기준선 대비 소폭의 개선에 그쳤고, 센서 이질성과 분포이동이 결합된 HAR 환경에서 일반화가 쉽게 달성되지 않음을 드러낸다. 저자들은 이러한 ‘특정 분포이동 중심’의 체계적 탐구를 최초로 제시하며, open-source benchmark 플랫폼과 데이터셋을 공개해 후속 연구의 표준 실험 기반을 마련했다.



### BluTrain: A C++/CUDA Framework for AI Systems (https://arxiv.org/abs/2606.24780)
- **Prior Approaches**: 기존 딥러닝 학습 프레임워크는 모델 아키텍처 자체보다 하드웨어에서 어떻게 표현·실행되는지가 학습 속도, 메모리 사용량, 수치 정확도를 크게 좌우한다. PyTorch 같은 산업 표준은 성숙했지만, 시스템 복잡도를 다루기 위한 반복적인 오케스트레이션 로직이 필요하고, 하드웨어 표현을 ‘절대적으로’ 통제하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 BluTrain을 표준 C++과 core CUDA 프로그래밍 모델 위에 처음부터 설계해, 시스템 복잡도를 추상화하면서도 하드웨어 표현을 엄밀히 통제할 수 있는 학습 프레임워크를 제안한다. 모든 계층을 네이티브로 구현(typed tensor 모듈+reverse-mode autograd, 선형대수 라이브러리, caching allocator, 분산 실행 모듈, MLIR 기반 컴파일러)해 반복 제어 로직 의존을 줄이고, 아키텍처-일반성을 목표로 한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 성능 최적화를 위한 하드웨어 표현을 프레임워크가 정확히 통제하면서도 (2) 역전파·연산의 수치적 충실도를 엄격히 보존하고 (3) 분산 실행까지 포함한 end-to-end 파이프라인을 가볍게 유지하는 것이다. 저자들은 각 레이어를 네이티브로 구현하고 MLIR 기반 딥러닝 컴파일러와 캐싱 allocator, 다중 모드 분산 실행을 통해 수치 fidelity와 수렴 특성을 훼손하지 않으면서 처리량과 메모리 효율을 동시에 끌어올린다.

- **Empirical Impact**: 실험에서 BluTrain은 8-GPU 6000 Ada 시스템에서 FP32로 124M 파라미터 GPT-2 baseline을 학습해 처리량과 메모리 효율을 동시에 개선했다. 평균 407K tokens/s로 PyTorch의 395K tokens/s를 상회하고, 최대 22%까지 footprint reduction을 달성했으며 수치 정확성은 엄격히 보존하면서 최종 검증 손실도 약간 더 낮게 수렴했다. 결과적으로 BluTrain은 프레임워크 자체가 성능 상한의 ‘병목’을 재규정할 수 있음을 보여주며, 고성능 학습 시스템 엔지니어링의 재현성과 통제성에 의미가 있다.



### Can Scale Save Us From Plasticity Loss in Large Language Models? (https://arxiv.org/abs/2606.24752)
- **Prior Approaches**: 손실된 가소성(loss of plasticity)은 학습이 진행될수록 새 데이터를 효율적으로 학습하지 못하게 되는 현상으로, 기존 연구는 주로 오래된 소형 네트워크나 합성 데이터/비전 영역에 집중돼 왔다. 또한 언어 모델에서는 과제 수가 적거나 평가가 제한적이라, LLM 규모에서 가소성 손실이 실제로 언제/어떻게 나타나는지 정량화가 어려웠다. 이 때문에 GPT-style Transformer가 자연어 스트림에서 장기간 학습 후에도 동일한 문제가 생기는지, 모델 크기에 따라 양상이 어떻게 달라지는지 불명확했다.

- **Core Contribution**: 이 논문은 GPT-style decoder-only Transformer에 대해 다국어 연속학습(continual pretraining)에서 가소성 손실이 발생하는지, 그리고 그 시작 시점을 모델 크기로 예측할 수 있는지를 자연어 설정으로 검증한다. 학습 중에는 다양한 언어를 순차적으로 제공하고, 학습에 포함되지 않은 베트남어를 held-out probing task로 주기 평가해 “고정된 학습 예산에서 적응 효율이 떨어지는지”를 측정한다. 또한 연속학습뿐 아니라 언어를 섞은 stationary 학습에서도 동일한 악화 신호를 관찰해, 급격한 task change만이 원인이라는 관점을 흔든다.

- **Technical Challenges**: 핵심은 (1) 자연어에서 충분히 긴 학습 길이로 가소성 손실의 ‘시점’을 포착하고, (2) 언어 간 전이(transfer) 차이로 평가 잡음을 줄이며, (3) 모델 크기 비교가 공정하도록 아키텍처 변수를 최소화하는 것이다. 저자들은 CulturaX 기반 다국어 next-token prediction을 사용해 태스크 인스턴스를 무한히 연장 가능하게 만들고, 베트남어는 학습에 배제해 학습되지 않은 새 정보에 대한 적응을 측정한다. 모델 크기별로는 비-임베딩 파라미터만 스케일하고 attention head 차원/모양 비율을 고정했으며, probing 빈도를 늘린 뒤 스무딩(이동 평균)으로 곡선의 최소점을 가소성 손실 onset의 추정치로 사용해 로그-로그 파워 법칙을 피팅했다.

- **Empirical Impact**: 실험 결과, 5M부터 314M(비-임베딩)까지 전 모델에서 held-out 베트남어 probing 성능이 시간이 지날수록 악화되는 경향이 나타났다. onset은 작은 모델에서 더 빨리 관측됐고, 크기가 커질수록 효과는 지연되지만 “파라미터 수만 늘리면 완전히 사라지지”는 않았다(스케일링은 sublinear 파워 법칙). 더 나아가 stationary multilingual training에서도 동일한 가소성 손실 신호가 관측돼, 실제 조건에서 LLM이 결국 새 데이터 적응 능력을 잃을 수 있음을 시사한다. 저자들은 또한 가소성 손실과 연관될 수 있는 내부 상관 지표(예: parameter magnitude, dormant units, attention head 붕괴/지연 등)를 측정하지만, 아직 이를 진단·상쇄하는 ‘smoking gun’은 찾지 못했다고 정리한다.



### Scaling Laws for Task-Specific LLM Distillation (https://arxiv.org/abs/2606.24747)
Comments:
          24 pages, 13 figures

- **Prior Approaches**: 기존 지식 증류는 hard distillation(정답 라벨) 또는 soft distillation(로짓 분포)을 학생이 모사하도록 하는 방식으로 발전해 왔다. 또한 chain-of-thought(CoT) 같은 풍부한 감독 신호가 데이터 효율을 높일 수 있다는 연구들이 있으나, 도메인 특화 데이터에서 pruning과 결합될 때 in-domain 성능과 일반 지식의 균형이 어떻게 깨지는지는 규모·방법별로 정리되지 않았다.

- **Core Contribution**: 이 논문은 양자금융 도메인(35개 이벤트 분류)에서 domain-specific LLM compression을 위한 실증적 스케일링 법칙을 도출한다. 특히 dataset size, compression ratio, supervision format, iterative pruning schedule이 in-domain 성능 저하와 general-knowledge 붕괴의 시점을 어떻게 바꾸는지 계량화하고, logit-based vs LoRA-based distillation을 함께 비교한다.

- **Technical Challenges**: 핵심 난제는 pruning으로 사라진 일반 지식을 CoT 기반 증류가 안정적으로 “복구”할 수 있느냐이며, 이를 위해 KL-divergence가 reasoning trace 구간에서 흔들리는 문제가 발생한다. 저자들은 label 토큰과 CoT 토큰의 KL 손실을 블렌딩(Blended chain-of-thought supervision loss)해 label 쪽 그래디언트를 명시적으로 유지하면서도 전체 trace 정보를 함께 학습하도록 설계했고, iterative structural pruning과 distillation을 단계적으로 반복해 압축 한계를 밀어 올린다.

- **Empirical Impact**: 실험 결과, compression이 진행될 때 in-domain 품질은 예측 가능한 방식으로 저하되는 반면 general-knowledge는 더 일찍 붕괴하는데, 그 격차를 좌우하는 가장 큰 요인이 supervision format임을 확인했다. blended CoT 감독은 pruning으로 지워진 일반 지식을 적극적으로 되살려, teacher의 파라미터를 16% 수준으로 줄이면서도 의미 있는 과업 품질을 유지했으며, FinHeadlineMix 데이터셋과 스케일링 법칙·실무 권고를 공개해 도메인별 압축 의사결정을 재사용 가능한 틀로 제공한다.



### Decentralised AI Training and Inference with BlockTrain (https://arxiv.org/abs/2606.24722)
Comments:
          First arXiv version. 17 pages

- **Prior Approaches**: 기존 분산/탈중앙 학습은 federated averaging, local SGD, DiLoCo 계열처럼 통신만 줄이거나, gossip·decentralized SGD처럼 중앙 파라미터 서버만 제거하는 방향이 대부분이다. 하지만 공통적으로 각 워커는 여전히 full-model을 들고 학습하며, 그 결과 소비자 GPU 수준에서는 full optimizer state/메모리 부담이 병목이 된다. 다른 계열인 네트워크·파이프라인 분산은 remote autograd·활성(activation) 트래픽 등 시스템 문제를 해결하려 하지만, 학습 목적 자체를 로컬화하지는 않는다.

- **Core Contribution**: Spheroid BlockTrain은 모델을 블록 단위로 쪼개고, 각 워커가 block-local denoising objective만 최적화한 뒤 추론 시 조립(합성)해 하나의 모델로 만든다. 즉, 분산 시스템의 “참여 단위”를 하드웨어 분할 단위와 동일한 granularity로 맞추는 objective-hardware alignment를 제안한다. 이 설계는 워커가 full-model optimizer state를 가질 필요 없이, 학습 메모리가 active block에만 비례하도록 만든다.

- **Technical Challenges**: 핵심 난관은 깊은 네트워크에서 오차 신호가 본질적으로 global하다는 점이며, 단순 셔딩은 결국 centralized cluster를 흉내 내는 동기화/통신이 필요해질 수 있다는 것이다. BlockTrain은 target embedding을 디스커트 라벨로 두고 EDM-style preconditioning, lognormal sigma partitioning, weighted cross entropy로 block-local 목표를 구성해 learning을 로컬로 유지한다. 또한 naive σ 스케줄이 local denoising CE만 좋아 보이게 만드는 target-leakage를 유발함을 지적하고 σ∈[1,10] 범위를 통해 from-noise 추론 성능 붕괴를 막는다.

- **Empirical Impact**: byte-level WikiText에서 block으로 학습한 모델은 cross entropy 1.359(perplexity 3.89)로, 동일 셋업 end-to-end Transformer 기준선 대비 약 0.04 CE 이내를 달성했다. 6워커가 같은 블록을 공유하는 shared run에서도 CE 1.385를 기록했으며, HTTP/TCP 전송 실험은 public-IP 3호스트 환경에서 CE를 5.580→1.811로 끌어내리며 15.22GB 규모 체크포인트/업데이트 전송까지 검증했다. 서빙 측면에서는 한 번의 블록-스택(one-sweep) 순회로 전체 출력 시퀀스를 정제하는 경로를 유지해, 토큰마다 WAN 왕복이 필요한 autoregressive TCP 파이프라인 대비 효율 이점을 보이며 internet-scale 탈중앙 학습 연구의 실험 발판을 제공한다.



### Cost-Optimal Decision Diagrams for Stochastic Boolean Function Evaluation (https://arxiv.org/abs/2606.24672)
Comments:
          11 pages, 4 figures

- **Prior Approaches**: Boolean 함수를 평가할 때 각 변수 관측 비용이 다르고, 변수 값이 통계적으로 의존하는 경우가 있어 Stochastic Boolean Function Evaluation(SBFE) 문제가 중요해졌다. 기존 연구는 일부 특수한 공식/확률 제약(예: 대칭 함수, 단순 확률 가정)이나 근사/최악-경우 복잡도에 치우쳐, 일반적인 확률 분포와 일반 Boolean 함수에서의 “정확한 최적”을 실용적으로 풀기엔 한계가 있었다. 또한 결정트리/decision diagram 계열 접근도 대개 비용·확률을 제한적으로만 다루거나(또는 탐색 휴리스틱과 pruning이 부족해) 확장성이 약했다.

- **Core Contribution**: 이 논문은 변수별 비용과 임의의 확률분포(예: Bayesian networks로 표현 가능한 조건부 확률)를 전제로, 주어진 명제식 ϕ를 최소 기대비용으로 판정하는 결정 다이어그램(optimal decision diagram) 구성을 다룬다. Branch-and-bound 기반의 정확 알고리즘을 제시하고, 변수 선택 heuristics, pruning, caching을 결합해 일반성의 수준에서 “최초의 practical exact algorithm”임을 강조한다. 또한 beam-search 변형의 runtime–quality trade-off를 함께 계량하고, 구조화된 심장질환 진단 문제를 사례로 평가한다.

- **Technical Challenges**: 핵심 기술적 난점은 일반 Boolean 함수에 대해 탐색 공간이 폭발하는데, 기대비용 목표가 확률 의존성 때문에 부분해의 품질/하한을 정교하게 계산해야 한다는 점이다. 논문은 부분 결정다이어그램 비용과 하한(bound)을 이용한 가지치기 조건을 설계하고, 동일 부분문제에 대한 caching으로 중복 계산을 줄이며, 변수 선택 순서에 대한 휴리스틱을 통해 실제 탐색 깊이를 유의미하게 낮춘다. 종료 조건은 부분식이 논리적으로 ⊤/⊥로 동치가 되는지(즉 SAT 판정)로 처리해 정확성을 유지한다.

- **Empirical Impact**: 무작위 인스턴스 실험에서 알고리즘이 확장성을 보이며, pruning·caching·변수 정렬 휴리스틱이 성능을 어떻게 끌어올리는지 정량적으로 확인한다. 특히 clause-to-variable ratio에 따른 난이도 변화(어떤 구간에서 특히 어려워지는 현상)를 관찰해, 실제 인스턴스 특성과 계산 가능성의 연결고리를 제공한다. 복잡도 측면에서는 최적 결정다이어그램 구성 문제가 #P-hard임을 #SAT로부터 보였고, 기대비용 계산/판정이 PSPACE(FPSPACE) 범주에 포함된다는 멤버십도 증명해 이 문제의 난이도 스펙트럼을 명확히 했다.



### LaGO: Latent Action Guidance for Online Reinforcement Learning (https://arxiv.org/abs/2606.24669)
Comments:
          9 pages, 2 figures. Accepted at the ICML 2026 Workshop on Large Language Models for Planning (LM4Plan)

- **Prior Approaches**: 기존에는 LLM을 직접 controller로 써서 관측에서 바로 action을 예측하거나, LLM이 subgoal/plan을 제안하고 저수준 정책이 이를 실행하는 고수준 planner 형태로 활용하는 연구가 많았습니다. 또한 LLM이 world model처럼 미래 궤적을 시뮬레이션해 planning에 쓰기도 했지만, 결국 action·계획·시뮬레이션을 “정확히” 생성해야 RL 제어가 안정적이었습니다. 그 결과 모델별 출력 품질 편차와 작은 오차의 누적이 online RL에서 신뢰성 문제로 이어질 수 있습니다.

- **Core Contribution**: 이 논문은 LaGO(Latent Action Guidance for Online Reinforcement Learning)를 제안하며, LLM을 명시적 planner/controller로 쓰지 않고 latent action prior로서 online 정책 최적화를 “부드럽게” 유도하는 프레임워크를 제시합니다. Stage 1에서 expert demonstration과 task description을 바탕으로 pretrained LLM 위에 latent action guidance 모델을 학습하고, Stage 2에서 PPO 같은 온라인 RL 정책 학습에 그 prior를 정규화 항으로 결합합니다. 이렇게 하면 LLM이 정확한 의사결정 대신 거친 행동 편향만 제공해도 보상 피드백으로 적응할 수 있습니다.

- **Technical Challenges**: 핵심 기술 과제는 LLM 지식을 online RL에서 안정적으로 통합하는 방식이었습니다. LaGO는 prior 모델을 고정한 채 RL 정책 손실에 prior regularization을 추가하고, 이 정규화가 action space(이산/연속)에 맞춰 동작하도록 구현해 RL의 탐색·수렴을 해치지 않게 설계했습니다. 특히 β로 prior의 영향력을 제어해 “soft guidance”를 유지하며, prior 품질이 낮아도 학습이 무너지는 것을 완화합니다.

- **Empirical Impact**: CLEVR-Robot(이산)과 Meta-World(연속)에서 LaGO는 Vanilla PPO 대비 일관된 성능 향상을 보였습니다. 성공률은 CLEVR-Robot에서 15.1%→27.2%, Meta-World에서 2.7%→15.2%로 크게 개선됐고, reward도 각각 0.076→0.122, 0.840→1.161로 상승했습니다. 또한 Llama-2-7b 같은 더 강한 pretrained LLM을 쓸수록 latent prior가 더 유효해져, 사전학습 모델의 발전이 곧 online RL 성능 향상으로 연결될 수 있음을 분석으로 확인했습니다.



### CineCap: Structured Reasoning with Spatio-Temporal Anchors for Cinematographic Video Captioning (https://arxiv.org/abs/2606.24636)
Comments:
          10 pages, 4 figures

- **Prior Approaches**: 기존 연구는 시네마틱 이해를 주로 VQA(분류/객관식)로 평가하거나, 캡션 생성도 카메라 모션처럼 일부 요소에 한정하는 경우가 많았다. 이 때문에 모델이 여러 촬영 언어 차원을 하나의 개방형 문장으로 통합해 설명하는 능력은 상대적으로 덜 다뤄졌다. 또한 비디오에는 시간에 따라 복합적으로 변하는 촬영 패턴이 존재하지만, 이를 자연스럽게 포착하는 평가·학습 설계가 부족했다.

- **Core Contribution**: 본 논문은 cinematographic captioning을 위해 CineCap 프레임워크를 제안한다. 핵심은 다중 차원(카메라 이동, 샷 크기, 촬영 각도, 심도, 구도, 피사체 방향)을 하나의 캡션에 통합하되, spatio-temporal evidence에 근거해 생성하도록 만드는 것이다. 아울러 수작업 주석 472쌍으로 구성된 CineCap Bench를 공개해 체계적인 비교가 가능하게 했다.

- **Technical Challenges**: 어려움은 (1) 미묘한 시각 단서로 전문 촬영 개념을 구분·추론해야 한다는 점, (2) 단일 라벨이 아니라 시간에 따라 변하는 복합 촬영 패턴을 잡아야 한다는 점, (3) 개방형 생성에서 포괄성(comprehensiveness)과 정확도(accuracy)를 동시에 만족시켜야 한다는 점이다. 이를 위해 spatial anchor와 temporal anchor 기반 Structured Reasoning으로 근거를 정렬하고, atomic CoT로 필요한 경우에만 추론을 학습하며, GRPO로 scmps(포괄성)·saccs(정확도) 중심 보상에 더해 atomic statement coverage 보상(정확도 게이트 포함)을 추가해 균형을 맞췄다.

- **Empirical Impact**: 실험 결과 CineCap은 Gemini 등 폐쇄형 모델과 다양한 오픈소스 비디오/멀티모달 모델 대비 일관되게 높은 성능을 보였으며, F1 평가에서 최대 32.41% 향상을 보고한다. 특히 aspect-level과 overall-level 모두에서 포괄성과 정확도를 동시에 끌어올려, 촬영 언어의 다차원·시간적 특성을 다루는 방법론의 실효성을 입증했다. 이는 MLLM의 세밀한 비디오 이해와 ‘통제 가능한 영화 품질’ 생성 방향에 유의미한 기반을 제공한다.



### SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active Investigation (https://arxiv.org/abs/2606.24626)
Comments:
          Published at the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) at ICML 2026

- **Prior Approaches**: 기존 fault attribution은 (1) 궤적을 통째로 LLM 컨텍스트에 넣거나, (2) 부분 시뮬레이션/이산적 단계별 탐색을 통해 원인을 찾아내는 방식이 주류였다. 그러나 다중 에이전트·장기 지평에서는 컨텍스트 한계로 인해 attention dilution과 정보 소실이 발생해 성능이 급락한다. 특히 Who&When, TRAIL 같은 벤치마크에서 토큰이 커질수록 정확도 저하가 관측된다.

- **Core Contribution**: 이 논문은 SAFARI를 제안한다. SAFARI는 긴 궤적을 선형으로 읽어 넣는 대신, 도구를 활용한 Active Investigation 루프와 Short-Term Memory(STM)로 필요한 구간만 찾아가며 결론을 내린다. 이로써 진단 정확도를 LLM 아키텍처의 컨텍스트 한계와 분리한다는 목표를 세운다.

- **Technical Challenges**: 장기 궤적에서 (1) 어디를 읽어야 하는지, (2) 누적된 대화 기록이 컨텍스트 예산을 초과할 때 증거를 어떻게 보존할지, (3) 최종 판정을 어떻게 검증할지가 핵심 난제다. SAFARI는 read(offset,limit)와 search(pattern) 도구로 증거를 표적 검색하고, up to E=3개의 원자적 가설을 Reasoning Evaluator LLM이 인용 증거만 근거로 점검하도록 해 검증 단계를 강제한다. 또한 STM을 통해 다음 행동 계획과 실패 가설을 지속 저장해, 오래된 로그가 컨텍스트에서 밀려나도 추론의 일관성을 유지한다.

- **Empirical Impact**: 실험 결과 SAFARI는 1M token 예산의 Who&When에서 기존 SOTA 대비 20% 향상, 25K token 예산의 TRAIL GAIA subset에서 19% 향상을 보였다. 더 나아가 목표 fault가 모델 네이티브 컨텍스트 윈도우의 5배 바깥(ς=5)에 있어도 Precision 0.58을 유지했는데, 전통적 평가/판정 방식은 이 설정에서 사실상 실패했다. 즉, 초장기 에이전트 실행에서 신뢰성 진단을 가능하게 하는 실용적 스케일링 방향을 제시한 셈이다.



### Themis: An explainable AI-enabled framework for Reinforcement Learning with Human Feedback (https://arxiv.org/abs/2606.24622)
Comments:
          The extended version of a paper published at the 2026 IEEE Conference on Artificial Intelligence (CAI). Includes an additional appendix with extended derivations and supplementary results. The main paper has 8 pages, 6 figures, 1 table

- **Prior Approaches**: 기존 RLHF 연구는 보통 보상 모델 학습과 인간 선호를 통해 reward hacking을 줄이려 하지만, 공개 프레임워크는 일반화와 접근성이 낮다는 한계가 지적된다. 또한 XAI(설명가능한 AI)는 주로 지도학습/협업 맥락에서 다뤄졌고, RLHF 파이프라인에 통합해 인간 피드백의 효율을 끌어올리려는 시도는 상대적으로 부족했다.
인간 피드백 수집 인프라도 단편적이어서, 실시간 상호작용·확장성·실험 관리 도구를 함께 제공하는 공개 체계가 사실상 없었다.

- **Core Contribution**: Themis는 XAI-enabled testing and evaluation framework로, RLHF에서 (1) reward learning과 modeling, (2) 에이전트와 결합된 설명 생성, (3) 대규모 피드백 수집용 클라우드 인프라를 한 프레임워크에서 연결한다. Atari, MuJoCo, Minigrid, BabyAI 등 200개+ 환경을 대상으로 구성 가능하며, 실험 목적에 맞춰 RL·투명성·정렬(alignment) 실험을 쉽게 조합할 수 있게 설계됐다.
또한 환경 보상 또는 인간 선호로 학습한 reward model을 선택할 수 있고, 실험마다 설정을 자동 기록해 재현성을 높인다.

- **Technical Challenges**: 핵심 난제는 인간 선호만으로 reward model을 안정적으로 학습시키면서도, 설명이 실제 피드백 품질을 높이도록 파이프라인에 자연스럽게 결합하는 것이다. Themis는 Bradley-Terry 기반 선호 학습과 ensemble(3개 MLP) reward 모델을 사용하고, 실험 예산 제약 하에서도 효과적인 클립/세그먼트 페어링을 위한 샘플링 기법(Maximum Disagreement, Entropy, K-Centered 등)을 제공해 학습 데이터를 효율화한다.
또한 상태-행동 세그먼트를 영상으로 렌더링해 설명을 붙일 수 있게 하고, Captum의 Integrated Gradients, Kernel SHAP, TracInCP를 plug-and-play로 지원한다.

- **Empirical Impact**: 실험적으로는 Synthetic teacher가 생성한 preference로 학습된 reward model이 true environment reward에 가깝게(때로는 더 좋게) 동작함을 보여, 선호 기반 reward distillation의 타당성을 검증한다. 특히 설명 기법 적용은 전반적인 실험 시간은 늘리되, 성능 관찰에는 큰 영향을 주지 않는 것으로 보고된다.
더불어 피드백 크라우드소싱 플랫폼의 확장성을 500~1000 사용자 시나리오에서 평가했으며, 중앙값 응답시간이 피크에서도 0.1초 미만으로 유지되는 등 대규모 동시 실험 운영이 가능함을 확인했다. 이 결과는 RLHF 연구가 ‘평가/테스트’ 단계까지 더 투명하고 재현 가능하게 확장될 수 있음을 시사한다.



### When CQs Go Wrong: Challenges in CQ Verification with OE-Assis (https://arxiv.org/abs/2606.24619)
Comments:
          Acceted poster at this https URL 23rd European Semantic Web Conference (Satellite Event)

- **Prior Approaches**: 기존 CQ-verification은 만들어진 온톨로지가 사전에 정의한 Competency Questions(CQs)를 자연어로부터 정확히 답할 수 있는지 확인하는 과정이다. 다만 CQ 자체의 모호성·가독성·복잡도가 해석을 흔들어 모델링 결정과 검증 결과를 일관되지 않게 만들며, 좋은 CQ를 작성하는 구체적 가이드는 부족했다. LLM은 CQ 생성·정제·정형화·온톨로지 생성/평가를 자동화·보조하는 방향으로 발전했지만, “어떤 CQ가 왜 어려운지”에 대한 실증적 기준은 여전히 제한적이다.

- **Core Contribution**: 이 논문은 OE-Assist 맥락에서 CQ-verification 수행 시 CQ의 모호성/복잡성이 사용자 성능(정확도·인지 난이도·응답 시간)에 어떤 영향을 주는지 실험적으로 분석한다. 참가자 19명이 20개 태스크를 LLM 보조(assisted) 또는 비보조(unassisted) 조건에서 수행했으며, 사용자 자유 서술을 함께 수집해 문제 유형을 도출했다. 특히 “CQ를 공개 전에 다듬어 모호하거나 과도하게 복잡해지는 것을 줄여야 한다”는 도구화 필요성을 강조한다.

- **Technical Challenges**: CQ가 어려운 원인을 분해해 측정하려면, 언어적 특성(가독성)과 온톨로지/태스크 조건(예: 온톨로지 크기, CQ 텍스트 기반 복잡도 지표)을 동시에 고려해야 한다. 연구진은 사용자 결정 일치 여부(ground truth와 CQ modelled 여부), 인지 난이도, 결정 시간, 온톨로지 크기, 그리고 CQ complexity로 Flesch–Kincaid Grade Level과 Gunning Fog Index를 연속변수로 계산해 상관/비교 분석을 수행했다. 그 결과 통계적으로 유의한 정량 신호는 제한적이었지만, 자유 텍스트에서 (i) SPARQL 생성 오류/이상한 구조, (ii) 이해하기 어려운 CQ 문장, (iii) 반복되는 모호성, 특히 다국어에서의 의미 간섭(lexical clarity 부족)이 핵심 문제로 나타났다.

- **Empirical Impact**: 실험에서는 인지 난이도와 결정 시간이 중간 수준의 양의 상관관계를 보였고, 이는 ‘어려운 CQ는 시간이 더 걸릴 가능성’으로 해석된다(p<0.001). 반면 CQ complexity, 온톨로지 크기 등 다른 조합에서는 유의한 연관이 관찰되지 않았으며, 이는 이 데이터셋에서 ‘난이도’가 주로 시간 측면에서 드러났음을 시사한다. 그러나 정성 분석은 읽기 난이도(Flesch–Kincaid 10 초과)는 실제로 참가자 이해 어려움과 정렬되고, “resource”처럼 다의적/다국어 동족 간섭을 부르는 표현이 의미를 갈라 답변 탐색을 흔든다는 점을 실무적으로 확인했다. 따라서 LLM 기반으로 CQ wording을 자동 점검하고 명확화 제안을 하는 CQ pitfall scanner(예: “Which resources (documents or records) …?”처럼 개념을 구체화)가 후속 온톨로지 공학 단계의 오류를 줄일 도구로 의미가 크다.



### Abstractions of Queries in Ontology-Based Data Access (https://arxiv.org/abs/2606.24618)
Comments:
          Extended version of a paper published in the proceedings of KR 2025

- **Prior Approaches**: 기존 OBDA 연구는 온톨로지 질의(주로 UCQ)를 데이터 질의로 ‘완전 변환’하는 문제에 집중해 왔지만, 반대로 데이터 질의를 온톨로지 레벨로 번역하는 query abstraction은 훨씬 어렵고 완전 추상이 항상 존재하지는 않는다. 또한 UCQ 수준에서는 매핑이 소스 관계를 구분하지 못해 원하지 않는 답이 섞이는 등, soundness·completeness를 동시에 만족하는 완전 추상이 막히는 경우가 보고돼 왔다.

- **Core Contribution**: 이 논문은 존재 규칙(existential rules) 기반 OBDA(마핑과 온톨로지 모두 TGD로 표현)에서 query abstraction의 ‘최소 완전성(minimally complete)’과 ‘최대 soundness(maximally sound)’를 체계적으로 다룬다. 특히 UCQ에 제한적 부등호와 DB 상수 표식을 추가한 질의 언어 UCQ≠,C를 도입해, 완전 추상이 존재한다면 그 추상을 이 언어로 표현할 수 있음을 보이고, 최소 완전 추상을 포착하는 데 초점을 맞춘다.

- **Technical Challenges**: 핵심 난제는 (1) 완전 추상이 존재하지 않거나 UCQ로는 표현 불가능할 수 있고, (2) 최소 완전성·최대 soundness를 판단/구성하기 위한 계산 복잡도와 구성 가능성까지 같이 다뤄야 한다는 점이다. 저자들은 수정된 chase로 UCQ≠,C에서 최소 완전 추상을 구성하고, 완전 추상 존재 여부의 복잡도를 경계(frontier) 크기에 따라 Π2^P2-완전 또는 Co-NExpTime으로 특성화하며, 최대 soundness는 data exchange의 maximum recovery 개념과 연결해 동일한 관점의 쿼리 재작성으로 정리한다.

- **Empirical Impact**: 이론적 증명 중심의 결과이지만, ‘완전 추상이 존재하는 경우 항상 UCQ≠,C로 표현 가능’하다는 표현력 정리는 OBDA 시스템에서 역방향 질의 번역을 더 실용적으로 설계할 기반을 제공한다. 또한 최소 완전/최대 soundness에 대한 계산 복잡도 분류와 maximum recovery와의 등가 정리는 query abstraction을 보다 예측 가능하게 만들어, 매핑 검증·역공학 및 FAIR한 데이터 서비스 의미 부여 같은 응용 방향에 직접적인 의미를 갖는다.



### AI Tokenomics: The Economics of Tokens, Computation, and Pricing in Foundation Models (https://arxiv.org/abs/2606.24616)
- **Prior Approaches**: 기존 연구는 토큰을 주로 BPE·SentencePiece 같은 토큰화 단위로 보거나, 스케일링 법칙을 통해 계산량과 성능의 상관을 설명하는 데 집중해 왔다. 또한 사업자 가격 페이지와 FinOps 실무는 토큰을 청구·집계·비용 통제의 대상으로 다루지만, 토큰이 워크플로 품질과 기업 가치·리스크·시장 구조를 어떻게 좌우하는지에 대한 통합 이론은 부족했다.

- **Core Contribution**: 이 논문은 AI tokenomics를 ‘토큰이 생성·소비·가격·평가·배분·최적화·거버넌스되는 방식’을 다루는 연구 분야로 정의하고, 토큰을 단순 회계 단위가 아니라 배분 가능한 경제 자원으로 재정의한다. 특히 토큰 지출이 곧 경제적 가치와 일치하지 않으며, 한계생산성, 워크플로 내 위치, 숨은 추론 활동, 리스크 전파, 다운스트림 파급효과에 의해 가치가 결정된다고 정리한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 토큰 수를 계산·메모리·지연·에너지로 연결하는 매핑, (2) 입력/컨텍스트/검색/출력 외에 보이지 않는 hidden reasoning tokens 같은 구성요소를 포함한 측정, (3) 태스크 난이도·컨텍스트·불확실성이 토큰 수요를 비선형·내생적으로 만들 때 이를 예측·보정하는 것이다. 논문은 토큰의 기술적 해부학과 카테고리 분해를 제공하고, 태스크 수준에서 난이도·컨텍스트·불확실성을 생산함수 형태로 두어 토큰 수요를 캘리브레이션할 수 있는 기반을 마련한다.

- **Empirical Impact**: 프레임워크는 토큰 지출, 워크플로 품질, 경제적 가치를 실사용 배치에서 어떻게 상호작용시키는지 수치 사례로 보여주며, 기존 단순 비용관리 관점을 넘어서는 의사결정 모델의 필요성을 강조한다. 또한 hidden-token 측정·실증적 캘리브레이션·token productivity·동적 배분·토큰 기반 시장설계 등 후속 연구 의제를 제시해 기업 AI 인프라에서 토큰 계량을 ‘가치 생성’ 중심으로 전환하는 데 기여한다.



### ScaleToT: Generalizing Structured LLM Reasoning for Billion-Scale Low-Activity User Modeling (https://arxiv.org/abs/2606.24605)
- **Prior Approaches**: 기존 사용자 모델링은 클릭·구매 같은 행동 시퀀스를 기반으로 다음 행동을 예측하는 데 강점이 있지만, 저활동·휴면 사용자에게는 기록이 너무 희박하거나 오래돼 신호로 쓰기 어렵다. 보조 메타데이터 결합, 크로스도메인 전이, LLM을 통한 추론 보강 같은 접근도 결국 행동연계 신호를 주입하는 경우가 많아 정적 프로필만으로는 한계가 크다.

- **Core Contribution**: ScaleToT는 정적 프로필만 있는 저활동 사용자에서도 LLM의 “사용자 상태 추론”을 재현하되, 단일 판단을 신뢰성 있게 만들기 위해 typed user-state chain을 도입한다. 또한 bounded entropy-guided Tree-of-Thought(ToT)로 대안 상태를 보존하며 국소적으로만 수정해 추론 신뢰도를 높이고, 이 체인을 SFT와 OSIPO로 학생 모델에 전이한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 희박 프로필에서 LLM 추론이 불안정해지는 문제와 (2) 사용자별 LLM 추론을 빌리언 스케일에서 수행하기 불가능한 비용 문제다. ScaleToT는 교사(Teacher) 단계에서 훈련에만 제공되는 privileged context와 judge 기반 품질검증을 활용해 chain을 만들고, OSIPO로 결과(outcome) 기반·구간(segment) 인지형 암묵 보상을 주어 “중간 상태가 의미 있게 유지되도록” 학습시킨다.

- **Empirical Impact**: 광고 LTV(Lifetime Value) 예측에서 ScaleToT는 온라인 랜덤 A/B 테스트로 LT30이 6.738% 개선됐고, 온/오프라인 모두에서 추론 품질과 예측 성능을 함께 끌어올렸다고 보고한다. 특히 잠재 모집단 중 LLM 체인 생성은 7.32%에만 적용해 전체 인퍼런스 비용을 크게 낮추면서도 효과를 유지하는 점이 실용적 의미를 갖는다.



### Uncertainty-Aware Longitudinal Forecasting of Alzheimer's Disease Progression Using Deep Learning (https://arxiv.org/abs/2606.24604)
- **Prior Approaches**: 기존 AD 진행 모델링은 다음 방문의 진단을 단일 레이블로 예측하는 next-visit classification에 치우쳐, CN–MCI–dementia의 순서성(ordinal structure)과 장기 예측에서 불확실성이 어떻게 누적되는지를 충분히 반영하지 못했다. 또한 점추정 중심이라 불확실성이 실제 질병 변동(aleatoric) 때문인지 모델 무지(epistemic) 때문인지가 분리되지 않는 경우가 많아 외부 코호트 전이 시 경고 신호를 제공하기 어렵다. 사건 기반/미분방정식/혼합 모델 등은 해석성은 있으나 개인별 이질성과 장기 확률 궤적 생성 및 캘리브레이션을 동시에 다루기엔 제약이 있었다.

- **Core Contribution**: 본 논문은 AD 진행을 “순서형 진단 예측 + 다중 시점 확률 궤적 생성 + 불확실성 분해”를 한 파이프라인으로 통합하는 확률론적 프레임워크로 제안한다. Temporal Fusion Transformer 인코더를 CORAL ordinal 출력과 비대칭 손실 가중, converter oversampling에 맞춰 CN–MCI–dementia의 단계 순서를 학습에 반영한다. 이어서 인코더의 환자 컨텍스트를 조건으로 autoregressive Mixture Density Network가 5년치(진단 상태 및 여러 바이오마커)의 다중 미래를 확률 궤적으로 생성한다.

- **Technical Challenges**: 핵심 난제는 (1) 진단 라벨의 순서를 학습 손실에 자연스럽게 반영하면서 (2) 장기 예측에서도 다봉 분포(여러 가능한 미래)를 표현하고 (3) 불확실성을 aleatoric/epistemic으로 분해해 의미 있게 전달하는 것이다. 논문은 MDN의 mixture 분산으로 aleatoric을 analytic하게 추정하고, 5개 멤버 deep ensemble로 epistemic을 추정해 law of total variance 기반으로 단계별·바이오마커별 불확실성을 분리한다. 또한 6~12개월 간격처럼 방문 간격이 불균일한 ADNI 설정에 맞춰 TFT를 적응하고, MCI-to-dementia 전이에 민감하도록 학습 구성을 조정한다.

- **Empirical Impact**: ADNI에서 제안 모델은 다음 방문 진단 예측에서 선형/순환/Transformer 계열 기준선을 능가하며, 특히 MCI vs dementia 구분에서 향상폭이 가장 크다. 생성된 5년 확률 궤적은 credible interval coverage가 거의 명목 수준(약 90%)에 수렴하고 예측 horizon이 늘수록 불확실성이 자연스럽게 확대되며, 바이오마커 동역학도 알려진 AD 진행 양상과 부합한다. 더 나아가 OASIS-3 외부 평가에서 epistemic uncertainty는 외부 분포 이동 하의 예측 오류와 함께 증가해, 드물거나(진행 archetype) MCI·dementia 환자에서 “신뢰도 경고 신호”로 작동함을 보여준다.



### ASALT: Adaptive State Alignment for Lateral Transfer in Multi-agent Reinforcement Learning (https://arxiv.org/abs/2606.24601)
Comments:
          Accepted at RLC 2026 conference

- **Prior Approaches**: 기존 MARL 전이학습은 CTDE 하에서 중앙집중학습된 정책을 다른 도메인으로 옮기는 연구가 많았지만, 대부분 소스와 타깃의 관측 공간과(global) state 공간 차원이 동일하다는 가정에 기대는 경우가 많다. 이 제약 때문에 StarCraft II, Google Research Football처럼 맵/에이전트 수에 따라 관측 차원이 바뀌는 현실 환경에서는 전이가 구조적으로 막히는 병목이 생긴다.

- **Core Contribution**: 이 논문은 관측 및 global state 공간의 차원이 서로 다른 도메인 사이에서도 지식을 옮길 수 있도록 ASALT를 제안한다. ASALT는 observation adapter와 state adapter로 타깃 입력을 소스의 공유 임베딩 공간으로 매핑하고, 이후 MALT 스타일의 lateral knowledge transfer로 actor와 critic 양쪽에서 단계별 표현을 이어받도록 설계된다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 차원 불일치를 해결하면서도 (2) 에이전트 간 상호작용(협조/경쟁 맥락)을 보존해 negative transfer를 줄이는 것이다. 이를 위해 observation adapter는 agent-specific 특징과 contextual 특징을 분리하고 H-MHA(계층형 multi-head attention)로 관계를 학습하며, state adapter는 Transformer encoder로 state 구성요소 간 상호작용을 임베딩화해 소스 critic에 입력한다. 또한 어댑터는 타깃과 함께 joint-training하거나 고정 학습 후 타깃을 학습하는 두 패러다임을 비교해, 전이에 유리한 임베딩 품질을 얻도록 한다.

- **Empirical Impact**: 실험은 SMAC, Google Research Football, MPE의 여러 전이 설정에서 수행됐고, ASALT는 관측 차원 불일치/에이전트 수 변화/이질적 시나리오에서 기준선 및 기존 전이 기법 대비 학습 효율(수렴 속도)과 최종 성능(global return)을 개선하는 경향을 보였다. 특히 negative transfer가 큰 장애가 되는 상황에서도 ASALT가 이를 완화함을 관찰했으며, 성능 이득은 소스-타깃 간 불일치 정도에 따라 달라진다고 정리한다.



### AdversaBench: Automated LLM Red-Teaming with Multi-Judge Confirmation and Cross-Model Transferability (https://arxiv.org/abs/2606.24589)
Comments:
          10 pages, 4 figures, 5 tables. Code and data at this https URL

- **Prior Approaches**: 기존 LLM 레드티밍은 LLM-as-a-Judge로 판정의 대규모화를 시도했지만, 레드티밍에서는 판정이 ‘정답 비교’가 아니라 ‘기준 위반 여부(pass/fail)’로 바뀌어 엄격/관대 판정 편향을 동시에 감지하기 어렵다. 또한 단순 프롬프트 생성이나 일괄 공격 세트는 어떤 변형 연산자가 어떤 유형 과제에서 왜 잘 먹히는지 설명력이 떨어질 수 있다. 마지막으로 inter-annotator agreement를 Cohen’s kappa로만 보면 라벨 쏠림(high-agreement, low-κκ paradox)으로 신뢰도가 왜곡될 위험이 있다.

- **Core Contribution**: AdversaBench는 seed prompt를 다섯 가지 structured operator로 변형하고, 타깃 모델을 호출한 뒤 3인 패널로 fail/pass를 판정하며 meta-judge tiebreaker로 판정 불일치를 정리하는 end-to-end red-teaming 파이프라인을 제안한다. 또한 “실패율”만이 아니라 seed별 attacker iterations(생성-검증 반복 횟수)까지 함께 보고, 실패 판단의 신뢰성은 단일 κ뿐 아니라 카테고리별 불일치율로 해석하도록 설계했다. 추가로 약한 모델에서 만든 adversarial prompt가 더 강한 모델로 zero-shot transfer되는지까지 확인해, 변형이 모델 특이 약점을 넘어 일반 행동 패턴을 겨냥할 가능성을 보여준다.

- **Technical Challenges**: 문제는 (1) hard input을 효율적으로 찾는 검색 전략과 (2) “실제 실패인지”를 신뢰도 있게 검증하는 판정 체계가 동시에 필요하다는 점이다. AdversaBench는 epsilon-greedy(ε=0.2)로 operator를 선택하고, failure 미확정 시 강한 변형 모델로 escalation하며, 최대 5회 반복과 checkpoint resume로 탐색 효율과 재현성을 확보했다. 검증은 expected_behavior를 모호함 없이 명시해 ground-truth 기반으로 판단하고, 3인 패널의 불일치는 meta-judge로 중재하되 κ의 한계를 보완하기 위해 raw disagreement 등 대체 지표를 함께 계산한다.

- **Empirical Impact**: 45개 seed(추론, 지시-따르기, 툴 사용) 모두에서 confirmed failure가 나왔고, 특히 instruction-following은 평균 2.4회 반복이 필요해 binary failure rate가 난이도 차이를 숨긴다는 점이 survival curve로 드러났다. operator 효과도 카테고리별로 급격히 달라졌으며, 예를 들어 inject_distractor는 instruction-following에서는 평균 reward 0.00이지만 reasoning/tool-use에서는 0.80~0.83 수준으로 크게 작동했다. 판정 신뢰성 분석에서는 80~87%의 높은 raw agreement가 라벨 쏠림 때문에 Cohen’s kappa가 거의 0에 가까워질 수 있음을 보여주며, 근거 기반의 “불일치율 보고”가 중요하다는 실무적 함의를 준다. 마지막으로 Llama 3.1 8B에서 생성한 adversarial prompt가 Llama 3.3 70B에 zero-shot으로도 상당 비율 전이되어, 공격 변형이 특정 모델 취약점이 아니라 일반적인 행동 취약성을 겨냥할 수 있음을 시사한다.



### LLMs Prompted for Legal Context Object More: Overrefusal from Small On-Premises LLMs in Criminal Legal Contex (https://arxiv.org/abs/2606.24585)
- **Prior Approaches**: 기존 연구는 안전 정렬(safety alignment)이 해로운 출력을 막는 대신, 겉보기엔 무해한 질의도 거부하는 과잉거부(over-refusal)를 낳는다는 문제를 다뤄왔다. 오버리풀절렉(OR-Bench) 같은 벤치마크는 주로 언어적 트리거(어휘 편향)와 같은 원인을 분석했지만, 법-사법 맥락에서 사용자의 역할/권위 프레이밍이 거부율을 어떻게 흔드는지는 체계적으로 보지 못했다.

- **Core Contribution**: 이 논문은 온디바이스(on-premises)에서 실행될 가능성이 큰 small open-weight LLM에 대해, ‘권위 있는 사용자로 가장하는 접두사(prefix)’가 과잉거부를 얼마나 증폭하는지 정량적으로 확인한다. 특히 “defense lawyer”, “national supreme court” 같은 제도적 역할 프레이밍이 무해한 범주의 법률 프롬프트에서도 거부를 크게 올리며, 어떤 경우에는 jailbreak 역할 오버라이드보다 더 강하게 작동함을 보여준다.

- **Technical Challenges**: 데이터 거주/비공개 제약 때문에 상용 API를 배제하고, 8B급 이하 오픈 모델을 Ollama로 로컬 서빙하며 온도 T=0 등 OR-Bench 관행에 맞춰 평가했다. 또한 거부 판정은 OR-Bench의 키워드 매칭을 확장해 프랑스어/독일어까지 다뤘고, 동일 프롬프트에 대해 무접두사/변호사/대법원/언론플레이(jailbreak) 접두사를 순차 적용해 언어별·모델별 불안정성을 비교했다.

- **Empirical Impact**: 5개 OR-Bench 범주(폭력·성적·유해·불법·비윤리) 전반에서 권위 접두사가 무접두사 대비 거부율을 2~20배까지 올렸으며, 특히 supreme court 접두사가 lawyer 접두사보다 더 큰 경향을 보였다. 언어 효과도 갈라져 독일어에서는 영향이 약해지거나 거의 사라지는 모델이 나타났고, OR-Bench 밖의 실제 법률 문서 30건에서도 동일한 방향성의 예비 재현이 관찰됐다. 저자들은 이는 소형 로컬 LLM이 ‘실제 기관 사용자가 자연스럽게 붙일 법한 문맥’에 대해 안전동작이 흔들릴 수 있음을 시사하며, 법률·다국어 제도 적용 시 품질뿐 아니라 과잉거부 신뢰성 평가가 필요하다고 강조한다.



### Quant Convergence: Bridging Classical Value Investing and Modern Factor Models for Systematic Equity Selection (https://arxiv.org/abs/2606.24575)
- **Prior Approaches**: 기존 정량금융은 XGBoost, Random Forest, AutoGluon 같은 복잡한 ML로 주가 예측 신호를 찾는 흐름이 강했다. 하지만 금융 데이터의 높은 잡음 때문에 과적합(overfitting)으로 단기 모멘텀·변동성에 치우치며, 충격 국면에서 손실이 커질 수 있다는 한계가 반복 지적돼 왔다. 또한 Graham의 방어적 가치지표는 개별 지표나 선형/단순 설정에서는 성과가 확인됐지만, 이를 비선형·블랙박스 모델에 어떻게 “제약(regularization)”으로 넣을지에 대한 검증은 부족했다.

- **Core Contribution**: 이 논문은 벤저민 그레이엄의 가치 투자 규칙을 특징(feature)으로 고정해, 모델이 단기 잡음을 쫓지 못하도록 수학적 low-pass filter처럼 작동하는지 실험한다. S&P 500 20년 데이터를 바탕으로 Graham 규칙 단독(Strategy A), 현대 팩터 단독(Strategy B), 두 범주의 혼합(Strategy C), 그리고 XGBoost·AutoGluon 같은 무제한 AutoML(Strategy D/E)을 동일한 buy-and-hold 평가 틀로 비교했다. 핵심 메시지는 “복잡한 모델이 항상 우수”하다는 관성에 반해, Graham의 margin of safety가 리스크를 억제하는 실질적 규제 장치가 될 수 있다는 점이다.

- **Technical Challenges**: 가장 큰 기술적 난관은 미래 정보 누수와 시간적 편향(look-ahead bias)을 피하면서도 하이퍼파라미터 탐색을 공정하게 수행하는 것이었다. 논문은 엄격한 80/20 시간 분할(TimeSeriesSplit), 학습 기간에서만 결측치 중앙값을 산정하는 방식, 난수 시드 고정, 그리고 재고나 거래비용이 아닌 단일 buy-and-hold 시뮬레이션으로 재현성을 확보했다. 또한 상위 20개 종목을 확률 상위로 선택하는 공통 포트폴리오 구성으로 비교 가능성을 높였고, ROC-AUC 기준의 Optuna(TPE) 튜닝으로 노이즈 환경에서도 모델을 동등하게 맞췄다.

- **Empirical Impact**: 4년 외부 테스트(2022년 3월~2026년 3월)에서 무제한 AutoGluon은 총수익 222.68%를 기록했지만 최대낙폭 39.78%로 급락해 리스크 조정 성과가 약화됐다. 반면 Pure Graham Random Forest는 총수익 232.13%를 달성하면서 최대낙폭 35.01%, Calmar ratio 1.38로 최고 수준의 위험-수익 비대칭을 보였다. 특히 혼합 접근(Combined RF)은 모멘텀 신호가 우선되며 수익이 줄고 낙폭도 가장 낮지는 않았고, 통계적으로는 Pure Graham이 S&P 500 대비 우월성을 10% 유의수준(p=0.098)에서 뒷받침해 “구조적 제약이 신호를 정제한다”는 결론에 힘을 실었다.



### GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents (https://arxiv.org/abs/2606.24551)
- **Prior Approaches**: 기존 GUI 에이전트 평가는 주로 화면 기반 상호작용의 성능을 다루지만, 작업·초기상태·검증기(verifier)·허용 행동공간이 함께 달라져 ‘모달리티 효과’가 분리되지 못했다. 반면 programmatic/skill 기반 CLI 연구도 존재했으나, CLI는 주로 노출되는 기능(스킬 레이어)의 구성 차이가 성능을 좌우해 모달리티 비교가 혼탁해졌다.

- **Core Contribution**: 이 논문은 GUI와 skill-mediated CLI를 동일한 목표·초기상태·최종상태 검증기로 매칭한 matched execution-layer benchmark를 제안한다. 18개 응용프로그램과 12개 워크플로에서 총 440개 데스크톱 작업을 사용하며, 각 에이전트는 모달리티 네이티브 행동만 허용해 ‘같은 실행 문제를 다른 표현으로 푸는 효과’를 관찰한다. 

- **Technical Challenges**: 핵심 난제는 공정한 비교를 위해 작업 절차는 모달리티 중립적으로 재서술하되, 검증은 동일한 최종 상태에서 수행해야 한다는 점이다. 이를 위해 OpenComputer 작업을 기반으로 하고, CLI-Anything에 해당하는 스킬이 있는 앱만 선별해 440개 작업을 만들었으며, GUI는 스크린 액션만, CLI는 skill layer만 사용하도록 제약해 우회 편법을 차단했다. 또한 CLI의 스킬 부족을 진단하기 위해 verifier-guided patched-skill(검증기가 요구하는 체크포인트에 맞춰 스킬 경로를 보수·동기화) 실험을 설계했다. 

- **Empirical Impact**: 실험에서 가장 강한 GUI 에이전트 GPT-5.4는 59.1% full pass rate로, 원래 skill 레이어의 CLI 강자 Codex GPT-5.5(48.2%)를 앞섰다. 그러나 verifier-guided patched-skill로 CLI 스킬 커버리지를 보강하면 CLI 성공이 69.3%까지 올라, CLI 격차의 상당 부분이 모델 능력보다는 스킬 인터페이스의 불완전성에서 온 것임을 시사한다. 추가로 오류 분석에서는 GUI가 visual grounding·긴 실행 체인에, CLI는 skill coverage·암묵적 기본값 재구성·비가시적 애플리케이션 시맨틱 부재에 더 취약함을 보여 ‘실행 병목이 어디에 설계되었는가’라는 관점으로 비교를 재정의한다.



### Governed Shared Memory for Multi-Agent LLM Systems (https://arxiv.org/abs/2606.24535)
- **Prior Approaches**: 기존 LLM 메모리 연구는 긴 문맥과 retrieval-relevance를 중심으로, 주로 단일 에이전트(단일 사용자·대화) 환경의 recall 품질을 개선하는 데 초점이 있었다. 그러나 실제 멀티에이전트 공유 메모리에서는 접근 권한 범위, 시간적 일관성(최신성), 충돌 해결, provenance(작성/유래) 보장 같은 거버넌스 요구가 핵심이어서 단순 임베딩 기반 검색 최적화만으로는 부족하다는 한계가 지적된다. 공유 메모리를 다루는 최근 접근들도 temporal contradiction-resolution(시간적 모순 해소)을 1차 클래스 연산으로 다루지 않거나, 라이브 서비스에서 강제 실패를 계측한 실증이 제한적이었다.

- **Core Contribution**: 논문은 fleet-memory 문제를 형식화하고, 멀티에이전트 공유 메모리에서 발생하는 네 가지 기반 실패 모드(unauthorized leakage, stale propagation, contradiction persistence, provenance collapse)를 정의한다. 이를 해결하기 위해 scoped retrieval, temporal supersession, provenance tracking, policy-governed memory propagation이라는 시스템 수준 프리미티브를 제안한다. 또한 프로덕션 멀티테넌트 메모리 서비스 MemClaw를 구현하고, 이를 실험적으로 계측하기 위한 리프로듸서블 하네스 ArgusFleet를 제공한다.

- **Technical Challenges**: 핵심 난제는 ‘관련 있어 보이는’ 정보 검색이 아니라, 각 검색 경로에서 권한·범위·시간적 올바름을 정책대로 평가하며 충돌을 비동기 탐지/해소와 함께 정확히 동작시키는 것이다. 저자들은 scoped retrieval과 provenance/temporal 해석을 retrieval 파이프라인에 내장하고, MemClaw에서 GET/검색 경로별 강제와 함께 정책 주도 전파를 구현했다. 다만 라이브 측정 중 (1) GET-by-id 경로에서 sub-tenant 스코프가 누락되는 갭과 (2) 동기 near-duplicate 게이트가 비동기 contradiction detector보다 먼저 쓰기 요청을 거절해 pipeline-ordering conflict가 발생하는 문제를 발견했으며, 운영 수정(remediation)과 경계 조건을 명시한다.

- **Empirical Impact**: ArgusFleet로 MemClaw를 라이브 계측한 결과, provenance는 depth-4 derivation chain의 100%를 정확한 writer 정체성과 함께 재구성했고(서브초 지연), 플릿 간 cross-fleet leakage는 0으로 관측됐다. 또한 강한 write mode에서 write-to-visible 지연을 단일 search round-trip 수준으로 최적화했음을 보고한다. 더 중요한 의미는 long-context retrieval만으로는 부족하며, governed shared memory는 명시적 시스템 추상화와 라이브 기반 부정(negative) 결과 측정이 필수라는 점을 실증적으로 보여줬다는 것이다.



### Reinforcement Learning for Computer-Use Agents with Autonomous Evaluation (https://arxiv.org/abs/2606.24515)
Comments:
          Accepted to the 4th International Workshop on Generalizing from Limited Resources in the Open World (GLOW @ IJCAI 2026)

- **Prior Approaches**: CUA는 자연어 지시를 보고 GUI를 스스로 조작하는 end-to-end 에이전트로 발전했지만, 데스크톱 환경은 성공 여부를 기계가 읽을 만한 보상으로 제공하지 않는 경우가 많다. 그래서 기존 RL은 task별 DOM 체크 같은 취약한 휴리스틱이나 사람이 최종 상태를 라벨링하는 수작업에 의존해 확장성과 일반화가 제한됐다. Vision-Language Model 기반 자율 평가기도 등장했지만, false positive/false negative 같은 오차를 무시하고 raw 평가 결과를 그대로 보상으로 쓰면 학습이 편향되고 불안정해질 수 있다.

- **Core Contribution**: 논문은 Vision-Language Model(비전-언어 모델) 자율 평가를 GUI 에이전트의 RL fine-tuning용 “스케일 가능한 보상 신호”로 쓰되, 평가 오차를 노이즈 채널로 명시적으로 모델링한다. 특히 최종 스크린샷과 원 지시문만으로 이진 성공을 판단하는 evaluator 피드백을, false positive/false negative를 보정한 reward estimator로 변환해 PPO에 그대로 삽입 가능한 형태로 제안한다. 이를 통해 수작업 라벨이나 task-specific 휴리스틱 없이도 신뢰도 있는 학습 신호를 얻는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 과제는 evaluator가 이진 판단을 내리지만 그 자체가 불완전해, raw 보상을 그대로 쓰면 정책 업데이트가 편향될 수 있다는 점이다. 논문은 보정 대상인 false positive/false negative의 오류율을 held-out calibration split에서 추정하고, separability 조건 하에 평가 출력의 기댓값이 참 보상에 맞도록 asymptotically unbiased reward estimator를 유도한다. 이후 terminal-only 보상 특성을 고려해 PPO의 advantage 계산에 보정된 보상을 사용함으로써, 노이즈가 섞인 피드백에서도 안정적으로 정책을 학습하게 한다.

- **Empirical Impact**: macOSWorld, Windows Agent Arena, OSWorld의 실험에서 corrected evaluator 보상이 zero-shot 및 raw evaluator fine-tuning보다 일관되게 우수했다. 통합 모델과 per-OS 모델 모두에서 raw 보상보다 corrected 보상이 성공률이 더 크게 개선됐고, 예를 들어 per-OS fine-tuning에서 success rate가 macOSWorld 0.084→0.203, Windows 0.331→0.442, OSWorld 0.283→0.432로 상승했다. 전체적으로 zero-shot 대비 평균 12.6%p 개선, raw 보상 대비 5.1%p 개선을 보이며, “자율 평가는 가능하지만 오차 보정이 필수”라는 실용적 결론을 제시한다.



### A specialized reasoning large language model for accelerating rare disease diagnosis: a randomized AI physician assistance tria (https://arxiv.org/abs/2606.24510)
Comments:
          36 pages, 5 figures

- **Prior Approaches**: 기존 희귀질환 진단 지원 연구는 대규모 언어모델(LLM)의 가능성은 보여주었지만, 임상 현장 배포 관점에서 충분히 작동 가능한 형태가 아니거나 근거가 임상적으로 충분히 정교하지 못한 한계가 있었다. 또한 희귀질환은 데이터가 매우 부족해 학습용 텍스트가 적고, 결과가 long-tail 질환에서 불안정해지기 쉽다는 문제가 동반됐다.

- **Core Contribution**: 이 논문은 RaDaR(Rare Disease navigatoR)라는 32B 파라미터 규모의 오픈소스 compact reasoning LLM을 제안하며, 희귀질환 진단을 위한 deployable 추론 모델을 목표로 한다. RaDaR은 공개 임상 free-text 케이스와 합성(synthetic) 케이스를 함께 학습해 임상 근거 제약과 데이터 부족을 동시에 겨냥한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 희귀질환에서의 부족한 훈련 데이터, (2) 임상적으로 의미 있는 추론 신호를 학습시키는 방법, (3) 배포 가능한 모델 크기와 성능을 맞추는 것이다. 논문은 reasoning-enhanced training으로 49,170개의 공개 텍스트 케이스와 104,666개의 synthetic 케이스를 학습했고, 추가 분석에서 phenotype-anchored narratives(표현에 증상-현상 기준점을 둔 내러티브)가 long-tail 희귀질환에 유효한 학습 신호가 된다는 점을 ablation으로 제시한다.

- **Empirical Impact**: RaDaR은 공개 벤치마크와 4개 외부 검증 센터에서 평가된 오픈소스 모델 중 가장 높은 성능을 보였으며, DeepSeek-R1(671B)까지 포함한 비교에서도 강점을 보였다. 후향적 코호트에서는 문서화된 임상 의심 이전에 최종 진단을 우선순위로 내세운 비율이 61.06%로, 잠재적인 lead time 1.87개월과 within-center interval의 50.18%를 보고했다. 또한 무작위 의사 보조(trial)에서 RaDaR 도움은 인터넷 검색만 사용한 경우 대비 희귀질환 진단 정확도를 21.44%p 개선했으며, 합성 데이터 관련 ablation은 일정 구간에서 단조적 성능 스케일링 경향을 보여 실용성 근거를 강화한다.



### On the Smallness of the Large Language Models Scaling Exponents (https://arxiv.org/abs/2606.24504)
Comments:
          11 pages, 2 figures

- **Prior Approaches**: 기존 LLM 스케일링 연구는 “no-wall”을 근거로 지수 α가 양수이므로 더 큰 모델·데이터·연산이 성능을 계속 개선한다고 주장해왔다. 다만 관측된 α가 0.05~0.10 수준으로 매우 작아 자원 대비 손실 감소가 둔화되는 ‘diminishing returns’가 반복 지적됐고, 실제 계산 관점에서는 비현실적인 비용 증가로 이어질 수 있다는 비판이 나왔다. 또한 일부는 “pedestal effect(무한 데이터에서도 loss가 0으로 수렴하지 않는 효과)”를 무시한 수치적 편향 탓이라고 반박했으나, 저자는 이 설명이 지속가능성 문제를 해소하지 못한다고 본다.

- **Core Contribution**: 이 논문은 LLM 스케일링 지수가 작아 나타나는 비지속가능성의 이유를 물리·수치해석 관점의 일관성/효율성 프레임으로 재정의한다. loss를 ‘정답과의 수치오차를 대표하는 경험적 준-거리(pseudo-metric)’로 보더라도, 결국 중요한 것은 loss가 얼마나 빨리 줄어드는가(스케일링 지수의 실제 크기)이며 pedestal effect로는 결론이 바뀌지 않는다고 주장한다. 더 나아가 Sharma-Kaplan(SK)류의 하한 이론을 근거로, 지수의 크기가 데이터가 놓인 내재(intrinsic) 차원과 데이터의 매끈함/거칠기에서 유도된다는 설명을 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) pedestal effect 같은 ‘수치적 편향’이 관측된 지수 α를 얼마나 왜곡하는지, (2) 그 왜곡을 제거해도 여전히 효율성 붕괴(지수의 작은 값)가 남는지, (3) 지수 값이 왜 내재 차원과 거칠기에 의해 결정되는지 이론적으로 연결하는 것이다. 저자들은 loss의 pedestal을 포함해 running scaling exponent가 언제 저지수(Anthropic)에서 고지수(Chinchilla)로 전이되는지 임계 조건을 계산하며, 실제 대규모 학습에서 pedestal이 충분히 작지 않다고 보여 pedestal이 지수 문제의 주원인이 아님을 논증한다. 이어 SK 토이를 설명하고, 데이터가 매끈하지 않은 경우(난류 유사 거칠기) 신호의 미분불가능성이 스케일링 지수를 더 낮추며 결과적으로 내재 차원이 1/h만큼 더 커져 지속가능성이 악화된다는 반유도(유사) 논리를 제시한다.

- **Empirical Impact**: 저자는 SK의 하한 관계와 이론적 유추(난류의 Hurst 지수, 거칠기-지수 감소)를 바탕으로, 내재 차원 d가 일정 수준을 넘으면 α<0.1 같은 작은 지수가 구조적으로 발생해 자원 효율이 나빠질 수밖에 없다고 결론 내린다. 따라서 ‘bigger is better’와 ‘no-wall’ 해석은 pedestal 조정만으로는 정당화되지 않으며, 복잡계처럼 비매끈한 데이터에서는 문제는 더 커진다고 강조한다. 분야적으로는 LLM 성능을 단순 스케일링 추세가 아니라 데이터의 내재 차원과 거칠기를 반영하는 world models 같은 물리 기반 접근으로 전환해야 한다는 문제의식을 강화하는 메시지로 받아들여질 수 있다.



### The Latent Bridge: A Continuous Slow-Fast Channel for Real-Time Game Agents (https://arxiv.org/abs/2606.24470)
- **Prior Approaches**: 기존 GCU(일반 컴퓨터 사용) 에이전트 연구는 느린 추론과 빠른 반응을 분리해 비동기 추론/스트리밍 텍스트로 결합하는 접근이 많았지만, 실시간 제어 루프에 맞춘 공개형(frozen) 결합 설계나 학습 채널 비교는 제한적이었습니다. 또한 단일 모델로 fast/slow를 동시에 학습하려는 시도는 “실시간 상호작용”을 목표로 포함시키는 순간 학습 데이터·목표가 흔들려 최상위 추론 능력의 유지가 어렵다는 문제가 지적됩니다.

- **Core Contribution**: 이 논문은 크기가 맞춘 두 개의 frozen VLM(반응형 MiniCPM-o 4.5, 추론형 Qwen3-VL-8B-Thinking)을 결합하되, 느린 모델의 deliberation이 빠른 모델에 전달되는 “브리지”만 학습하도록 설계를 고정합니다. 표준 Text Bridge(느린 모델의 텍스트 suffix를 프롬프트로 전달) 대신, 텍스트 왕복 없이 연속적인 Latent Bridge를 제안해 느린 모델 residual을 LLaVA-style로 빠른 모델 입력 임베딩 공간에 투영하고 전처리 latent 토큰을 붙입니다.

- **Technical Challenges**: 핵심 난제는 텍스트/레이턴트 입력 방식 차이가 빠른 모델의 action head를 OOD(out-of-distribution)로 만들어 붕괴를 유발할 수 있다는 점입니다. 논문은 deployment에서의 검증을 강조하며, (1) LLaVA-style 토큰 프리펜딩 구조로 아키텍처를 정착시키고, (2) action-head가 suffix/bridge 입력에 덜 민감하도록 robust 헤드(슬로우 스타일 suffix 확률적 삽입)까지 포함해 붕괴 모드를 완화합니다. 또한 최종 비교를 위해 채널별 action decoder를 held-out seeds 기준으로 튜닝해 공정한 성능 비교를 수행합니다.

- **Empirical Impact**: Atari 7게임과 MetaDrive 운전 시뮬레이터에서, Latent Bridge는 Text Bridge 대비 결코 크게 뒤처지지 않으면서 MsPacman(획득률 +57%)과 RoadRunner(+28%)에서는 더 나은 결과를 보였습니다. 반대로 두 채널(Text+Latent)을 동시에 넣으면 action head 고정 학습 특성 때문에 RoadRunner에서 -96% 같은 파괴적 간섭이 나타나 “딱 한 채널만” 쓰는 설계 규칙이 도출됩니다. 마지막으로 Latent Bridge의 이득은 “느린 추론이 빠른 반응보다 이미 더 낫다(T>Fast-Only)”일 때만 예측 가능하게 따라오는 상관(r=0.93)을 보이며, MetaDrive에서는 통제 실험처럼 실질적 이득이 없어서 채널이 무의미하게 동작하지 않음을 확인합니다.



### CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inferenc (https://arxiv.org/abs/2606.24467)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2508.02401

- **Prior Approaches**: 롱컨텍스트 LLM에서 KV cache는 길이에 비례해 커지며, 메모리 병목과 디코딩 비용을 함께 유발한다. 이를 줄이기 위한 KV-cache eviction은 StreamingLLM, SnapKV, CAKE처럼 토큰 중요도를 점수화해 KV 쌍을 버리지만, GQA 기반에서 헤드 기능 차이를 충분히 반영하지 못해 핵심 중간 근거가 함께 제거되는 문제가 반복됐다.

- **Core Contribution**: CompressKV는 GQA에서 기능이 다른 attention head를 구분해, 특정 헤드만을 근거로 중요한 토큰을 고르는 KV-cache 압축 프레임워크를 제안한다. 그 핵심은 Semantic Retrieval Heads(SRH)가 답변의 초기·말단뿐 아니라 의미적으로 중요한 중간 근거까지 포착하도록 식별하고, SRH가 지목한 토큰들의 KV를 유지하는 방식이다. 여기에 레이어별 압축 오차를 오프라인으로 추정해, layer-adaptive cache budget을 더해 자원-성능 균형을 개선한다.

- **Technical Challenges**: 기존의 head-agnostic 집계나 top-k/peak 중심의 retrieval-head 정의는 긴 문맥에서 희소하고 경계 토큰에 쏠린 attention 분포 때문에 중요한 스팬을 놓치기 쉽다. CompressKV는 calibration 데이터 기반의 span aggregation으로 SRH 점수를 구성해, 단일 토큰의 날카로운 최상위 attention이 없어도 의미적 기여가 있는 헤드를 포착하도록 설계했다. 또한 full-cache 대비 압축된 attention-block 출력의 Frobenius norm 오차를 레이어별로 오프라인 계산해, 온라인 생성 중 추가 연산 없이 예산을 레이어에 비례 배분한다.

- **Empirical Impact**: LongBench 실험에서 CompressKV는 매우 빡빡한 메모리 예산에서도 성능을 크게 보존하며, KV cache를 3%만 써서도 질문응답 과제에서 97% 수준의 성능을 유지한다. Needle-in-a-Haystack에서는 KV 저장 0.7%로도 정확도 90%를 달성해 긴 문맥 근거 검색에 특히 강한 내성을 보였다. 전반적으로 CompressKV는 메모리 예산 전 구간에서 기존 eviction 방법들을 일관되게 능가하며, 자원 제약 환경에서도 장문 추론을 더 지속 가능하게 만든다는 점에서 의미가 크다.



### Bayesian control for coding agents (https://arxiv.org/abs/2606.24453)
- **Prior Approaches**: 코딩 에이전트는 생성기와 진단기(cheap diagnostics)·검증기(expensive verifier)를 함께 쓰지만, 실제 도구 선택과 중단 판단은 고정 규칙(항상 검증, best-of-N, 단일 크리틱 게이트, 사전 루프)에 의존하는 경우가 많다. 이런 방식은 후보 정답성에 대한 사후확률(posterior)을 유지하지 않고, 크리틱 호출의 정보가치와 비용을 명시적으로 저울질하지 못한다. 결과적으로 과제 난이도, 크리틱 신뢰도, 생성기의 수리/파손 확률, 검증 비용에 따라 멈춤 시점을 유연하게 조정하기 어렵다.

- **Core Contribution**: 이 논문은 코딩 에이전트의 오케스트레이션을 cost-sensitive sequential hypothesis testing으로 정식화한다. 베이지안 컨트롤러가 후보 correctness에 대한 신념 b=P(Y=1|evidence)을 유지하면서, 더 증거를 모을지(크리틱), 후보를 정제할지(생성/리파인), 검증(오라클)을 실행할지, 혹은 중단할지 기대 효용 기준으로 동적으로 결정한다. 또한 이 신념은 해석 가능한 correctness 점수(불확실성 점수)로도 활용되며, 단순 토큰 확률이나 도구 성공률보다 불확실성 추정에 강점이 있음을 제시한다.

- **Technical Challenges**: 핵심 난제는 크리틱 관측이 잡음 섞인 증거라는 점과, 생성 단계가 정답을 고치기도 하지만 깨뜨리기도 한다는 전이 불확실성을 함께 모델링하는 것이다. 논문은 (1) 크리틱의 조건부 우도 P_i(z|Y)를 캘리브레이션 세트에서 추정하고 (2) 리파인 궤적의 오라클 라벨 연속쌍으로 생성 전이( fix/break 확률)를 베타-바이노미얼 스무딩으로 추정한 뒤, 이를 POMDP의 Bellman 방정식으로 연결해 정책을 만든다. 구현은 one-step Bayesian greedy와 finite-horizon Bayesian dynamic-programming 두 컨트롤러로 제공되며, 둘 다 base LLM(생성기)은 고정한 채 control layer에서만 최적화한다.

- **Empirical Impact**: 여섯 가지 생성기와 아홉 코딩 벤치마크에서 실험한 결과, 베이지안 제어는 ‘검증 비용이 크고 크리틱이 유의미하지만 완벽하진 않을 때’ 특히 효용이 크다. 반대로 공개 테스트가 숨은 정답 성공을 잘 예측하거나 후보가 이미 정답일 가능성이 높거나 검증이 싸면, 단순 공개 테스트 게이팅이나 always_verify 같은 고정 전략이 더 유리했다. 또한 조건부 전이(수리 확률)까지 계측해 다단계 리파인의 가치가 클 때에만 DP 기획이 그리디보다 추가 이득을 보였고, 베이지안이 유도하는 사후확률 점수는 불확실성 정량화에서 토큰-확률·raw tool-success 기준선을 앞섰다.



### ReM-MoA: Reasoning Memory Sustains Mixture-of-Agents Scaling (https://arxiv.org/abs/2606.24437)
- **Prior Approaches**: 기존 Mixture-of-Agents(MoA) 계열은 여러 LLM 에이전트를 층(layer)로 쌓아 추론을 반복 정제하지만, 깊이가 늘수록 성능이 저하·조기 포화·포인트 수렴 등 ‘스케일 붕괴’를 겪는다고 지적한다. 이를 완화하려는 시도로 정보 통합(RMoA), 층 내 상호작용(AttentionMoA), 품질 기반 선택(VeriMoA) 같은 접근이 제안됐으나, 공통적으로 성능이 깊이 증가에 따라 지속적으로 개선되지 못했다. 연구진은 특히 (1) 오류의 누적 전파, (2) 인접층 중심의 국소 통신, (3) 맥락이 비슷해지며 탐색 다양성이 무너지는 문제가 결합돼 나타난다고 분석한다.

- **Core Contribution**: 논문은 ReM-MoA를 제안하며, 핵심은 ‘구조화된 cross-layer reasoning memory’로 MoA의 스케일 붕괴를 해결하는 것이다. ReM-MoA는 모든 층에서 나온 추론 흔적을 Ranked Reasoning Memory에 저장하고, Curated Diversified Memory Routing으로 후속 층의 에이전트가 서로 다른 조합(성공/실패 포함)의 흔적을 보게 만들어 품질(Q)과 다양성(D)을 동시에 보존한다. 또한 Reviewer Agent의 순위 품질을 높이는 선택적 multi-domain reviewer distillation 파이프라인도 제공한다.

- **Technical Challenges**: ReM-MoA가 직면한 기술 과제는 두 가지를 동시에 만족시키는 것이다: 더 깊은 층이 이전 층의 ‘좋은 추론’을 안정적으로 재사용하도록 하면서, 한편으로는 에이전트들이 동일한 경로로 수렴해 탐색 다양성이 붕괴하지 않게 만드는 문제다. 이를 위해 Reviewer Agent가 각 층에서 모든 proposer의 reasoning trace를 비교·평가해 점수와 rationale을 만들고, 메모리는 raw 결과가 아니라 랭크된 trace를 append-only로 누적해 후속 층이 임의의 거리에서 조회할 수 있게 설계했다. Routing은 high-score/low-score 조합과 reinforcement·failure-mode focus를 균형 있게 분배해, 메모리 노출은 유지하되 within-layer 입력이 겹치지 않도록 구성했다.

- **Empirical Impact**: 5개 추론 벤치마크(수학, 형식 논리, 코드, 지식, 상식)에서 ReM-MoA는 깊이·너비 스케일 실험 전 구간에서 기존 MoA 변형들을 일관되게 능가하며, 특히 깊이가 커질수록 격차가 더 벌어지는 경향을 보였다. 반면 Standard MoA는 빠르게 성능이 떨어지고 RMoA/AttentionMoA는 조기 포화나 포인트 수렴으로 이어져 스케일 이득을 지속하지 못했다. 또한 distillation을 적용한 ReM-MoA*는 모든 벤치마크·설정에서 추가 우위를 보였고, 해당 결과는 “cross-layer reasoning memory”가 확장 가능한 multi-agent inference에 필수적인 누락 메커니즘일 수 있음을 시사한다.



### Can Aggregate Invariants Accelerate Continuous Subgraph Matching? Limits, Laws, and a Dynamic Spectral Index (https://arxiv.org/abs/2606.24421)
Comments:
          11 pages, 3 figures, 3 tables

- **Prior Approaches**: 정적 서브그래프 매칭에서는 Laplacian interlacing을 이용한 스펙트럼 가지치기가 강력하게 작동해 후보의 이웃 구조가 쿼리를 담을 수 없는 경우를 안전하게 제거한다. 하지만 연속 서브그래프 매칭(CSM)은 그래프가 업데이트되는 동적 환경이라, 이러한 집계(aggregate) 기반 불변식을 그대로 “지연 유지(lazy maintenance)”하거나 “느슨한 상계”로 다루면 가지치기 성능이 빠르게 붕괴할 수 있다. 기존 CSM 인덱스는 라벨/차수/이웃 라벨 멀티셋 등 국소 신호와 adjacency-guided 열거에 의존해 왔고, 스펙트럼 같은 집계 불변식은 사용되지 않았다.

- **Core Contribution**: 본 논문은 스펙트럼 interlacing 같은 집계 구조 테스트가 동적 CSM에서 실제로 이득이 되는지 체계적으로 답한다. 결론은 ‘지연 스펙트럼 상계는 가치가 있는 구간에서 거의 작동하지 않지만, 소수의 정점에 대해서는 정확한 지역 스펙트럼을 선택적으로 재계산하면 안전성과 함께 효과적인 가지치기가 가능하다’이다. 또한 CSM에서 필터 성능을 공정하게 평가하는 intermediate-invariance 방법론과, 업데이트 단위로 불필요한 delta enumeration을 스킵하는 AnchorGate 같은 메커니즘을 제안한다.

- **Technical Challenges**: 첫 번째 난점은 동적 업데이트로 인한 Laplacian 고유값 변동을 안전하게 상계로 추적하면서도 가지치기 ‘여유(margin)’를 유지하는 것인데, 논문은 perturbation relaxation 관점에서 지연 유지의 최적 규칙조차 최대 4회의 touching update 이후에는 사실상 가지치기 힘을 잃는다고 보인다. 두 번째 난점은 정확 재계산이 비싸지지 않게 하는 것으로, 공 변동성과 허브(hub) 제외 성질을 근거로 ‘작은 반경/라벨 관련 정점’에 한정해 SelSpec을 배치하면 로컬 스펙트럼을 마이크로초 단위로 정확 유지할 수 있음을 보인다. 마지막으로, 스펙트럼 테스트가 실제로 열거 단계(특히 adjacency-guided 탐색)에서 무엇을 건너뛰는지 정량화하기 위해 intermediate-invariance 기반 평가를 도입한다.

- **Empirical Impact**: Decoupled CSM 벤치마크에서 동일 파이프라인(스펙트럼 테스트만 제거)과 엄격 비교한 결과, 후보는 최대 51%까지 감소시키거나 업데이트 열거 중 최대 47%를 안전하게 스킵하지만, 완성도(정답 집합)와 enumeration intermediates는 게이트를 통과하지 못한 첫 바인딩을 제외하면 거의 동일하게 유지된다. 또한 반경 층화(radius-stratified) 워크로드에서 예외가 존재할 때는 intermediate 수가 크게 줄어드는 것을 검출하며, constructed 케이스에서는 -99.9% intermediates, 748× 가속까지 관찰된다. 종합적으로, 이 논문은 adjacency-guided 열거의 “초기 확장 수준에서만 의미 있는” 후보 제거를 스펙트럼·기타 subgraph-monotone 집계 테스트가 정확히 경계 맞춰 수행한다는 경험적 규칙성과 평가 프레임을 제공한다.



### Agentic AI for Bilevel Long-Term Optimization of Policy-Driven Physical Layer Systems (https://arxiv.org/abs/2606.24416)
Comments:
          14 pages, 11 figures

- **Prior Approaches**: 기존 물리계층 제어는 연산 목표와 제약이 고정된 가정에 맞춰 설계되어, 사업자 정책·의도·KPI가 바뀌는 비정상 환경에서는 효과가 떨어지는 문제가 있었다. DRL·학습 기반 접근도 사전에 정해진 reward/utility에 강하게 의존해 의도가 바뀌면 보상 설계 변경이나 재학습이 필요해 운영 비용이 커졌다. 한편 LLM/에이전트 관련 연구는 주로 intent 추출이나 구성 보조, 제어-plane 오케스트레이션에 집중되어, 엄격한 실시간성과 수치적 실행가능성까지 동시에 만족하는 physical-layer 결정 생성에는 직접 적용이 어렵다고 지적된다.

- **Core Contribution**: 이 논문은 Agentic long-term performance optimization(Agentic-LTPO)라는 중첩 bilevel 최적화 프레임워크로, 상위 레벨에서 에이전트가 정책을 해석해 물리계층 설정을 갱신하고, 하위 레벨에서 실시간으로 해당 설정의 최적 beamforming을 계산하도록 분리한다. 에이전트는 자연어 정책·환경 요약·과거 경험을 구조화된 하위 최적화 문제 구성 파라미터로 변환하되, 하위 레벨의 수학적 최적성은 유지한다. 사용 예로 cell-free MIMO 빔포밍에서 상위 레벨은 멀티에이전트(해석·관측·계획·비판)로 구성되고, 하위 레벨은 closed-form 빔포머와 robust 제약을 이용한 빠른 per-slot 해법을 둔다.

- **Technical Challenges**: 핵심 난제는 상위 레벨의 언어/의도 입력이 시간에 따라 바뀌는 가운데, 하위 레벨이 요구하는 ‘실행가능하고 제약을 만족하는’ 구조화 설정을 신뢰성 있게 생성하는 인터페이스를 만드는 것이다. 이를 위해 상위 레벨은 retrieval augmented generation(RAG)으로 policy memory와 case memory를 유지해 누적된 운용 근거에 기반해 설정을 평가·개선하며, planner–critic refinement loop로 불일치 위험을 줄인다. 또한 CSI 불완전성과 QoS 변동을 robust로 다루기 위해 worst-case SINR/robust QoS 제약을 두고, 특정 저수준 에너지 최소화 문제는 zero-forcing 조건 하의 closed-form worst-case SINR bound를 도출해 전 슬롯에서 선형 복잡도의 효율적인 판정·해 계산이 가능하게 했다.

- **Empirical Impact**: 실험은 cell-free MIMO 빔포밍에서 랜덤 및 piecewise-stationary 사업자 정책 시나리오를 포함해 수행되며, Agentic-LTPO가 기존 정적 기준선 대비 누적 통신 유틸리티를 57.2% 향상시키는 결과를 보였다. 또한 KPI 응답과 설정 궤적을 정책 레짐별로 분석해, 자연어 정책을 그대로 쓸 때의 언어 모호성이 상위 결정에 미치는 민감도를 점검하고(oracle structured-policy 대비), 그 영향을 줄이며 해석 가능한 구성 업데이트로 이어짐을 보여준다. 종합하면, 장기적 정책 변화를 빠른 물리계층 의사결정과 안정적으로 연결하는 실용적 프레임으로서 분야의 적응형 RAN/물리계층 최적화 방향에 의미 있는 성과를 제시한다.



### Cycle-Consistent Neural Explanation of Formal Verification Certificates (https://arxiv.org/abs/2606.24414)
Comments:
          15 pages of main text

- **Prior Approaches**: 기존 정형검증은 만족/위반을 증명하는 기계검증 가능한 인증서(certificates)를 제공하지만, 감사자나 컴플라이언스 담당자가 읽기엔 내용이 구조적·전문적이라 해석이 어렵다. 자연어 설명 생성에서 LLM 기반 접근은 유창성을 우선해 근거 없는 사실(환각)이 섞일 위험이 있고, 템플릿 기반은 충실도는 높지만 인증서 종류나 도메인마다 규칙을 손으로 맞춰야 한다는 한계가 있다.

- **Core Contribution**: 이 논문은 검증 인증서→자연어 설명→인증서 재구성의 순환(cycle) 구조로, 설명의 충실도(faithfulness)를 “재구성이 가능하면 충실”하다는 기준으로 강제한다. forward 모델은 pointer-generator로 인증서의 state 이름을 그대로 복사해 어휘 접지(lexical grounding)를 줄이고, inverse 모델이 설명으로부터 인증서를 다시 만들면 상징 검증기가 두 인증서를 대조해 오류를 판정한다.

- **Technical Challenges**: 핵심 과제는 설명이 유창하더라도 인증서 사실을 어기지 않게 만드는 것인데, 이를 위해 symbolic verifier의 의미 동치 검사를 differentiable한 cycle consistency 손실로 훈련에 연결한다. 또한 인증서 종류별로 필요한 생성 양상이 달라 단일 디코딩 전략이 비효율적이어서, 분류된 카테고리에 따라 고정/전수 디코딩을 선택하는 hybrid inference-time routing을 도입했다.

- **Empirical Impact**: 금융 컴플라이언스 도메인에서 420개 테스트 인증서(12개 verdict/kind 범주, 207개 named states)로 평가했으며, cycle-verified soundness 기준 90.0%를 달성해 최고 LLM few-shot 조합(76.1%)보다 13.9%p 높았다. 추론 지연은 160초(다중 LLM 풀 베이스라인) 대비 185ms로 약 860배 빠르고, 오프라인·결정적 출력·추가 per-inference 비용이 없어 규제 산업 배포 제약을 줄인다는 점에서 의미가 있다.



### ATRIA: Adaptive Traceable ECG Reporting with Iterative Agents (https://arxiv.org/abs/2606.24392)
- **Prior Approaches**: 기존 ECG 리포트 생성은 해석과 문서화를 end-to-end로 결합해 단계별 재검증이 어렵고, 한 번 생긴 오류가 최종 결과로 그대로 전파되는 문제가 있었다. 에이전트 기반 접근은 작업을 나누지만 보통 single-pass로 끝나며, 이전 출력의 재방문·수정이 제한적이었다. 그 결과 임상에서 흔한 반복적 맥락 통합과 bidirectional editing 흐름을 그대로 반영하기 어려웠다.

- **Core Contribution**: ATRIA는 임상의 반복 워크플로를 모사하는 multi-agent ECG reporting 시스템으로, 모든 문장(클레임)을 근거(evidence)와 바인딩해 추적 가능하게 만든다. 또한 세션 동안 추가 맥락(예: lab value, 과거 ECG)을 mid-session에 반영하고, 개별 finding 단위로 검증·수정을 할 수 있게 설계했다. one-shot처럼 보이는 생성이 아니라, 청사진처럼 증거 기반으로 편집 가능한 auditable 작업 흐름을 제공한다.

- **Technical Challenges**: 핵심 난제는 (1) 단계별 중간 결과를 보존해 재실행 없이 부분 수정이 가능해야 하고, (2) 새 컨텍스트가 들어왔을 때 영향을 받는 섹션만 정확히 재생성해야 하며, (3) 근거 없는 문장이 섞이는 것을 막아야 한다는 점이다. ATRIA는 stage-level handoffs로 inspectable artifacts를 저장하고 shared store에서 에이전트들이 읽고/쓰도록 하며, Review Agent가 드래프트와 상위 artifact manifest를 대조해 unsupported·모순 문장을 플래그하고 필요한 섹션만 targeted revision하도록 라우팅한다. 증거 요청 시에는 Literature Agent가 ECG-specific 지식베이스에서 문장 단위로 retrieved evidence를 가져와 분석 결과를 대체하지 않고 보강하는 방식으로 정합성을 유지한다.

- **Empirical Impact**: 논문은 실제 임상 판독에서 있을 법한 네 가지 상호작용 케이스(초기 생성 시 근거 없는 문장 플래그, lab value 추가 반영, 특정 문장 근거 요청, 과거 ECG 비교)를 통해 설계 요건을 시연한다. 특히 follow-up 요청 시 전체 파이프라인을 다시 돌리지 않고, 변경된 artifact만 갱신해 리포트의 해당 부분만 재작성되도록 동작함을 보여준다. cloud 기반 웹 서비스 형태로 배포 가능성을 제시해, 임상 도입 관점에서 즉시 확장 가능한 형태의 단계적·추적 가능한 ECG 리포팅 대안을 제안한다.



### Age of LLM: A Strategic 1v1 Benchmark for Reasoning, Diplomacy and Reliability of Large Language Models under Fog of War (https://arxiv.org/abs/2606.24391)
Comments:
          25 pages including appendices, 8 figures, 4 tables; appendices include verbatim system prompt and engine resolution pseudocode. All correlations reported with p-values, 95% bootstrap confidence intervals and Spearman's rho; includes a Steiger test and Bradley-Terry fit

- **Prior Approaches**: 기존 LLM 벤치마크(MATH, HumanEval, MMLU 등)는 대부분 단일 턴, 완전 관측, 정답이 명확한 과제에 초점이 있어 적대적 불확실성 하 계획(planning under uncertainty)이나 숨은 상대 추론, 여러 턴 동안의 structured output 신뢰성(잘못된 JSON/불법 행동 처리)을 충분히 검증하지 못한다. 또한 공개 벤치마크는 학습 데이터로의 오염 가능성이 커 점수가 실제 경쟁력보다 부풀려질 위험이 있다.

- **Core Contribution**: Age of LLM은 두 LLM이 fog of war, full diplomacy(메시지·정전·최후통첩)와 함께 13x7 그리드에서 1v1로 기지를 파괴하는 turn-based 1v1 벤치마크를 제안한다. 특히 엔진을 비공개로 두고 매 매치마다 무작위 맵 seed와 상대를 바꿔 데이터 오염 경로를 줄이며, 매 턴 출력은 strict JSON 스키마를 강제하고 불법 행동은 조용히 폐기되는 “신뢰성 차원”을 포함한다.

- **Technical Challenges**: 이 설정에서 핵심 기술적 난제는 부분관측·상대의 은닉·딜레마(핵 발사 vs 지상 공략) 속에서도 JSON 스키마를 지키며, 불법 행동이 누적되지 않도록 믿음(belief)과 상태 추적을 해야 한다는 점이다. 논문은 (1) near rule-only 프롬프트로 build-order 같은 직접 조언을 배제하고 (2) 에이전트가 매 턴 관측과 직전 턴 결과, 외교 기록만 이용하도록 하며 (3) 매치별 재현성을 위해 엔진-뷰어 간 replay 포맷과 replays를 제공해 분석 가능성을 확보한다. 다만 데이터 수집 당시 포함됐던 두 개의 borderline 전술 시드 문구가 정찰/탱크 중심 경향을 증폭했을 수 있어, 그 일부 결론은 제한점으로 표시한다.

- **Empirical Impact**: 54개 매치(15개 reasoning model, 5,258개 액션) 분석에서 결과는 핵(핵 선제 발사 루트)이 압도적으로 많았고(룰-일치 하위코퍼스 78%, 전체 85%), 군사 정복은 드물지만 성사되면 더 빠르게 끝나는 패턴(평균 12.3턴 vs 18.9턴)이 관측됐다. 외교는 활발히 오가지만 실제 합의로까지 이어지는 경우는 거의 없었고, 불법 행동의 약 58%가 fog/state 추적 오류로 나타나 illegal-action rate가 믿음 추적의 프록시로 기능함을 시사한다. 또한(탐색적 1개 모델 제외) reliability가 승리와 강하게 연결되지는 않았으며, 전체 순위는 소규모·불균형·사이드 미스왑의 영향으로 “예비적 기술 통계” 성격임을 강조한다.



### PHANTOM: A Large-Scale Dataset of Multimodal Adversarial Attacks for Vision-Language Models (https://arxiv.org/abs/2606.24388)
Comments:
          The dataset has been released at: this https URL

- **Prior Approaches**: 기존 VLM 안전·견고성 평가는 템플릿/특정 공격군에 치우친 벤치마크가 많아, 실제로 바로 돌려볼 수 있는 대규모 pre-generated adversarial 샘플이 부족했다. 또 상당수 데이터가 특정 모델·특정 모달리티 조합에 집중해, 시각-텍스트 교차 정렬을 악용하는 multimodal 특유의 공격면을 충분히 드러내기 어려웠다.

- **Core Contribution**: 이 논문은 VLM을 대상으로 한 대규모 오픈소스 adversarial 공격 데이터셋 PHANTOM을 공개한다. 10개 상위 범주·55개 하위 범주에 걸쳐 7,826개 harmful intent를 체계화하고, 총 47,524개 image–text 공격쌍(단일턴·대화형)을 ready-to-use 형태로 제공한다.

- **Technical Challenges**: 대규모 공격을 만들려면 계산비용과 탐색공간이 커지는데, 논문은 여러 최신 attack 전략 중에서도 BAP, IDEATOR, MML, FC ATTACK, CSDJ로 계산-효율 균형을 맞추어 샘플을 생성했다. 또한 판단 일관성을 위해 Abel-24-HarmClassifier 같은 공개 automated judge를 기준선으로 사용하고, intents는 여러 벤치마크를 병합한 뒤 의미 중복을 cosine similarity 임계값으로 정리해 잡음을 줄였다.

- **Empirical Impact**: 생성된 공격은 다양한 open-source 모델에 대해 효과를 보이며, 일부는 API로 호출한 proprietary 모델들(예: Claude Opus, GPT, Gemini)에서도 transferability 경향을 보였다. 특히 공격 성공률 관점에서 범주별 취약성이 드러났고, 전반적으로 이미지에 악성 텍스트를 임베딩하는 방식이 가장 강력한 편으로 관찰됐다. 결과적으로 연구자들이 재현성 있게 VLM robustness와 alignment을 평가·미세조정·방어 가드레일 스트레스 테스트를 수행할 수 있는 실용적 자원을 제공한다.



### When Helpfulness Overrides Causal Caution: Context-Dependent Suppression and Recovery in LLMs (https://arxiv.org/abs/2606.24370)
Comments:
          43 pages, 3 figures, 5 tables. SSRN Abstract ID: 6965680

- **Prior Approaches**: 기존 벤치마크 연구는 주로 LLM의 인과 추론 능력을 평가하며, 인과 판단을 해야 할 때 이를 보류하는 ‘인과적 신중함(Causal Caution)’ 같은 더 근본적인 인식론적 특성은 상대적으로 다뤄지지 않았다. 또한 학술형 문제와 실무형 조언을 같은 방식으로 비교하는 경우가 많아, 맥락에 따른 발화 억제 현상은 잘 드러나지 않았다. 


- **Core Contribution**: 이 논문은 LLM이 근거가 불충분할 때 인과 판단을 억제하는 성향인 Causal Caution을 핵심 평가 축으로 제안한다. 특히 학술 컨텍스트에서 실무 조언 컨텍스트로 전환될 때 Causal Caution이 체계적으로 억제되는지를 정량적으로 분석한다. 아울러 간단한 자기 수정 프롬프트가 이 억제를 되돌릴 수 있음을 보여 준다. 


- **Technical Challenges**: 핵심 기술적 과제는 ‘능력 부족’이 아니라 ‘표현의 맥락 의존성’인지 분리해 측정하는 것이다. 이를 위해 Pearl의 Causal Hierarchy에서 영감을 받은 PCH score 기반 루브릭으로 4개 고성능 LLM을 480개 트라이얼에 걸쳐 동일 조건에서 비교하고, 실무형 프롬프트에서는 구체 추천·설명 요청만 따로 제한해 Causal Caution이 실제로 얼마나 사라지는지 검증했다. 나아가 ‘Please reconsider this judgment from the perspective of causal relationships’ 같은 짧은 자기 수정 프롬프트로 발화 억제가 복구되는 메커니즘을 확인했다. 


- **Empirical Impact**: 실험 결과 Causal Caution 유지율은 학술 컨텍스트에서 91.7–100.0%였지만, 실무 조언 컨텍스트에서는 6.7–18.3%로 급락했다(모델 전체 p < .001). 더 구체적으로 실무에서 구체 추천/설명만 요구한 조건에서는 200개 응답 중 1개만(0.5%)이 신중함을 유지했으며, 자기 수정 프롬프트 적용 시 유지율이 71.4–100.0%로 회복됐다(p < .001). 논문은 유용성 지향의 응답 패턴이 신중함 표현을 누를 수 있음을 시사하며, 제안 생성과 인과 감사(auditing)를 분리하는 multi-agent 아키텍처가 조직 거버넌스에 유망하다고 제안한다. 



### Accelerating Disaggregated RL for Visual Generative LLMs with Diffusion-Based Parallelism and Trainer-Assisted Generation (https://arxiv.org/abs/2606.24369)
Comments:
          14 pages, 18 figures, 1 table

- **Prior Approaches**: 기존 Diffusion-RL(DanceGRPO, FlowGRPO, Long-RL, GenRL, veRL-Omni 등)은 대체로 생성기(rollout)와 트레이너를 같은 자원 풀에 함께 배치한 colocated 구조를 사용합니다. 이 방식은 동기화가 단순하지만, rollout과 학습이 강하게 결합돼 이질적인 GPU를 활용한 비동기적 스케줄링이나 독립적 스케일링에 제약이 생깭니다. 특히 disaggregated 구조를 쓰면 파이프라인 버블이 늘어 GPU 유휴가 발생해 전체 처리량이 떨어지기 쉬운 한계가 있습니다.

- **Core Contribution**: 이 논문은 diffusion-based generative LLM을 위한 disaggregated RL 프레임워크 DigenRL을 제안해, GPU를 생성기/트레이너로 분리하되 처리량을 높이는 것을 목표로 합니다. 핵심은 버블을 최소화하는 실행 설계로, (1) 생성축 파이프라인(GAP), (2) 시간스텝 병렬성(TSP), (3) trainer-assisted generation(TAG), (4) trajectory-consistent stale synchronization(TCSS)를 조합해 효율 저하를 줄입니다. 이를 통해 서로 다른 GPU 특성과 자원 배치를 유연하게 가져가면서도 학습 안정성을 유지합니다.

- **Technical Challenges**: disaggregated Diffusion-RL의 가장 큰 기술 난제는 생성기와 트레이너가 번갈아 기다리며 생기는 파이프라인 버블을 줄이는 것입니다. DigenRL은 diffusion 특성을 반영해 GAP로 더 잘게 파이프라인 단위를 쪼개고, TSP로 선택된 denoising timesteps를 병렬 계산해 트레이너 쪽 연산과 통신을 효율화합니다. 또한 TAG로 트레이너가 유휴일 때 일부 rollout을 대신 실행해 생성 지연을 흡수하고, TCSS로 마지막 동기화에서 생기는 idle 구간을 “한 step까지 정책이 stale일 수 있음” 제약 하에 미세하게 겹치며 제거합니다.

- **Empirical Impact**: 여러 하드웨어 테스트베드(동질 32GPU 2개, 이질 16GPU 1개)에서 HunyuanVideo-13B, Wan2.1-14B, FLUX.1-12B, QwenImage-20B 모델로 실험한 결과, DigenRL은 veRL-Omni 및 GenRL 대비 1.56~2.10x 처리량 향상을 보였습니다. 이는 Diffusion-RL에서도 disaggregated 아키텍처가 실전 처리량 병목을 해결할 수 있음을 경험적으로 확인한 결과입니다. 결과적으로 diffusion-based post-training에서 GPU 스케줄링과 파이프라인 설계가 성능을 좌우한다는 점을, 체계적인 시스템 관점에서 뒷받침합니다.



### MVG-KAN: Multi-View Geo-Wind Guided KAN for PM$_{2.5}$ Forecasting (https://arxiv.org/abs/2606.24347)
- **Prior Approaches**: 기존 PM2.5 예측은 ARIMA 같은 통계/회귀 기반 방법부터 RNN, LSTM, TCN, Transformer 등 딥러닝까지 폭넓게 발전했다. 다만 선형 가정·수작업 특징 중심 접근은 비선형 상호작용을 충분히 담기 어렵고, 딥러닝은 주로 시간에 집중해 다지점(공간) 의존성을 놓치는 경우가 많다. 그래프 기반 STGNN은 불규칙 관측소를 잘 다루지만, 거리/상관만으로 만든 그래프나 purely adaptive adjacency는 풍향·풍속에 따른 upwind/downwind 수송의 방향성을 물리적으로 반영하기 어렵다는 한계가 지적돼 왔다.

- **Core Contribution**: MVG-KAN은 PM2.5 진화를 예측 목적 관점의 3가지 뷰(국소 주기성, 관측소별 잔차 잔존 동역학, 바람 유도 공간 수송)로 동시에 모델링한다. 먼저 periodic-residual backbone으로 일/주 단위의 안정적 주기 성분을 분리해 비주기 잔차에 집중하고, 잔차는 Geo-Wind Graph와 TKAN residual head를 통해 각각 ‘바람 방향 수송’과 ‘관측소별 비선형 잔차 보정’을 담당하도록 역할을 나눴다.

- **Technical Challenges**: 핵심 난제는 (1) 주기 제거 후 잔차를 단순 잡음이 아닌 관측소별 비선형 자기회귀 보정 신호로 재구성하고, (2) 관측소 간 영향을 거리·상관만이 아니라 풍향·풍속의 downwind 정렬과 운반 가능 시간으로 유도하는 directed graph를 구성하는 것이다. 논문은 haversine 거리 커널에 더해 bearing 기반 downwind 정렬 게이트, 풍속 게이트(운반 가능성), 운반 시간(여행시간) 기반 감쇠를 결합해 Geo-Wind Graph를 만들고, TKAN은 de-periodized PM2.5 잔차와 과거 다중 오염물 시퀀스를 이용해 station-wise nonlinear autoregressive correction을 학습한다.

- **Empirical Impact**: 베이징 PM2.5 데이터셋 실험에서 MVG-KAN은 MAE 14.09, RMSE 21.40로 비교 딥러닝 방법 대비 평균 성능이 가장 좋았다. 또한 ablation에서 TKAN은 로컬 잔차의 시간적 보정에 기여하고, Geo-Wind Graph는 바람 유도 방식의 물리적 공간 보강을 통해 성능을 함께 끌어올린다는 상호보완성이 확인됐다.



### Prob-BBDM: a Probabilistic Brownian Bridge Diffusion Model for MRI sequence image-to-image translation (https://arxiv.org/abs/2606.24313)
- **Prior Approaches**: 기존 image-to-image 합성은 다양한 분야에서 성과를 내왔지만, 의료 영상에서는 여러 modality를 동시에 확보하기가 비용과 시간이 많이 든다는 문제가 남아 있습니다. 특히 3D 영상에서 modality 간 변환은 데이터 수집/획득이 병목이 되면서 성능과 실용성의 균형을 맞추기 어려웠습니다.

- **Core Contribution**: 이 논문은 Brownian Bridge Diffusion Models(BBDM)를 기반으로 2D axial slice에서 MRI sequences를 생성하는 image-to-image translation 모델을 제안합니다. 또한 Prob-BBDM(Probabilistic-BBDM)로 변분 인코더-가이드 diffusion 메커니즘을 결합해, 확률적 이미지 분포를 활용해 합성 품질을 높이는 데 초점을 맞춥니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 임상에서 얻는 제한된 정보로도 modality 변환에서 고품질을 유지하는 것과 (2) diffusion 절차가 너무 길어지면 계산 비용이 커진다는 점입니다. 저자들은 확률적 이미지 분포를 활용한 variational encoder-guided diffusion으로 합성의 안정성과 품질을 강화하면서, diffusion을 4 steps만으로 수행해 효율을 확보했습니다.

- **Empirical Impact**: BraTS 2021에서 Prob-BBDM은 여러 translation task에서 최대 88.46% SSIM, 26.09 dB PSNR을 달성하며 기존 방법 대비 일관된 개선을 보였습니다. 외부 제3자 데이터셋에서도 도메인 간 성능이 유지됐고, 합성 slice를 사전학습 segmentation 모델의 입력으로 사용했을 때 종양 segmentation Dice 88.71%, HD95 3.49 mm로 임상적으로 중요한 진단 정보를 보존함을 확인했습니다.



### LemonHarness Technical Repor (https://arxiv.org/abs/2606.24311)
- **Prior Approaches**: 기존 LLM 에이전트는 긴 작업에서 도구 호출과 로그 일부만 관찰하는 경우가 많아, 실제 파일 시스템·임시 디렉터리·의존성 환경 변화는 모델 시야 밖에서 누적되기 쉽다. 또 시간 예산을 외부 타임아웃으로만 두는 방식이 많아, 무의미한 탐색이나 과도한 검증이 마지막에 몰리며 timeout로 실패가 이어질 수 있다. 일부 연구는 성공 실패가 무작위로 분산되지 않고 파일 생성 같은 소수의 “상태를 바꾸는” 행동에 집중된다는 점을 지적한다.

- **Core Contribution**: LemonHarness는 long-horizon agent 실행을 위한 통합 실행 프레임워크로, 모델 호출-도구 실행-규칙 지식-로그 기록을 하나의 통제된 runtime boundary 안으로 묶어 state drift를 줄이는 것을 목표로 한다. 파일 쓰기, dependency 설치, 임시 아티팩트 생성 같은 상태 변경을 지정된 workspace 안의 structured tool 인터페이스를 통해 수행하고, 그 결과를 관찰로 기록해 다음 의사결정에 일관되게 반영한다. 또한 반복되는 실행 규칙과 acceptance criteria를 runtime 지식(재사용 가능한 일반 규칙 지식)으로 바꿔 시작부터 유효한 작업 경계를 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 “모델이 보는 관찰”과 “실제로 바뀐 작업공간 상태”의 불일치를 장시간에 걸쳐 얼마나 안정적으로 동기화하느냐에 있다. LemonHarness는 workspace 경계를 명시하고 상태 변경 연산을 structured tool로 고정해, 변경의 위치와 의미가 로그에서 추론 가능하도록 만들며 실행 피드백을 누적 관찰로 연결한다. 더불어 시간도 runtime state로 취급해 elapsed/remaining budget을 매 턴 제공하고, explore–implement–validate에 맞춘 단계 전환과 grace window로 우선순위를 연속적으로 재조정해 마지막 과검증·런어웨이 탐색을 완화한다.

- **Empirical Impact**: Terminal-Bench 2.0에서 LemonHarness_GPT-5.3-CodeX는 445 trials 중 84.49% 정확도를 달성했고, 같은 프레임워크에 더 강한 GPT-5.5 백본을 붙이면 평균 정확도 86.52%로 상승했다. 작업 유형별 잔여 실패는 주로 장시간 컴파일·학습·멀티모달 처리·교차 환경 적응처럼 긴 실행 체인과 엄격한 중간 검증이 필요한 경우에 집중됐다. 또한 Terminal-Bench 2.1의 업데이트 조건에서도 성능이 유지되며(예: GPT-5.5로 267 trials에서 91.76% 평균, 예외 6건) 통합 runtime boundary와 규칙 지식, 시간 인지 실행이 장기 에이전트 안정성에 의미 있는 개선을 준다는 신호를 제공한다.



### Tractable Reasoning and Conjunctive Query Answering for Defeasible DL-Lite under Rational Closur (https://arxiv.org/abs/2606.24279)
Comments:
          108 pages, 2 figures, 1 table

- **Prior Approaches**: 설명논리(Description Logics, DLs)에서 결함 가능(가변) 지식을 다루기 위한 비단조 추론으로 Rational Closure(RC)는 널리 알려진 형식주의다. 다만 DL-Lite 계열의 lightweight description logics 중에서도 core와 horn 변형에서 RC 하의 인스턴스 체크(entitlement)나 Conjunctive Query(CQ) 응답을 기존 추론기와 연결해 효율적으로 처리하는 방법은 제한적이었다. 기존 접근들은 RC의 비단조 성격을 별도 방식으로 구현해 계산 비용이 커지거나, 표준 classical reasoner 활용이 어려운 문제가 있었다.

- **Core Contribution**: 이 논문은 DL-Lite의 core 및 horn 변형에 RC를 적용할 때, entitlement(인스턴스 체크)와 CQ answering을 효율적으로 수행하는 절차를 제시한다. 핵심 기여는 표준 classical reasoner 위에 얹혀 동작하는 plug-in architecture를 제공해, RC 추론/질의응답을 최소한의 계산 오버헤드로 수행 가능하다는 점이다. 즉 RC 특유의 비단조 추론을 새로 처음부터 재구현하기보다, 기존 추론 인프라를 활용하는 방향으로 정리했다.

- **Technical Challenges**: RC 하에서는 결함 가능 지식이 상황에 따라 선택적으로 반영되므로, DL-Lite의 빠른 추론 특성을 유지하면서 비단조 추론을 정확히 결합하는 것이 기술적 난제다. 특히 entitlement와 CQ answering 모두에서 RC 조건이 반영된 의미론을 구현하면서도, 추가 단계가 과도한 시간·메모리 비용을 만들지 않도록 설계해야 한다. 논문은 기존 classical reasoner가 처리할 수 있는 부분을 최대화하고, RC 처리 로직을 플러그인 형태로 분리해 계산 오버헤드를 최소화하는 구성으로 이를 해결한다.

- **Empirical Impact**: 저자들은 RC 하에서 DL-Lite core/horn에 대한 reasoning과 CQ answering이 효율적으로 동작함을 실험적으로 입증한다. 결과적으로 표준 추론기를 활용하더라도 RC의 비용이 크게 증가하지 않아, 결함 가능 지식을 갖는 DL-Lite 응용에서 실용적인 질의응답 파이프라인을 제공한다. 이는 DL 계열 비단조 추론을 경량 프레임워크에 통합하는 데 있어 ‘컴퓨팅 비용 대비 구현 부담’을 낮추는 의미가 있다.



### Probing the Misaligned Thinking Process of Language Models (https://arxiv.org/abs/2606.24251)
- **Prior Approaches**: 기존에는 강한 LLM judge가 출력·행동을 보고 오정렬 여부를 판정하거나, chain of thought(CoT)를 읽어 더 효과적으로 감지하는 방식이 쓰였습니다. 하지만 배포 시에는 judge의 토큰 처리 비용과 지연이 크고, 향후 모델이 misaligned reasoning을 숨겨 transcript 신뢰도가 떨어질 수 있습니다.

- **Core Contribution**: 이 논문은 misalignment를 단일 개념으로 보지 않고, 18개의 fine-grained misalignment indicators(오정렬 인지 지표)로 분해한 뒤 내부 activation에서 이를 선형 probe로 탐지하는 모니터링 프레임워크를 제안합니다. indicator별로 개별 인지 패턴을 겨냥해 해석 가능성과 학습 가능성을 동시에 확보하는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 난제는 각 indicator가 실제 추론에서 일어날 때의 다양한 양성/음성 사례를 어떻게 안정적으로 라벨링·생성하느냐입니다. 이를 위해 meta-plan-guided 파이프라인으로 multi-turn 합성 대화를 만들고 hard negative와 benign overtriggering control을 함께 구성했으며, 캐시된 중간 activation(선형 로지스틱 회귀 probe)을 학습합니다. 또한 오탐을 줄이기 위해 probe filter(온라인)와 LLM judge(정밀 판정)를 결합한 white-box+black-box cascade를 설계했습니다.

- **Empirical Impact**: 5가지 오정렬 행동에 대해 out-of-distribution 평가에서 indicator probes 앙상블은 strong LLM judge와 유사한 성능을 보이며 AUROC 0.935(특정 OOD misalignment 벤치마크 기준)를 달성합니다. cascade는 probe 단독 대비 오탐을 크게 줄이면서도 LLM judge와 비슷한 종단 성능을 유지하며, benign 트래픽에서 false positive rate도 낮게 관리됩니다. 더불어 CoT를 저자원 언어로 번역해도 probe 신호는 상대적으로 견조하게 유지되어, black-box judge가 덜 읽는 상황에서 내부표현 기반 모니터링의 실용성을 시사합니다.



### Towards Federated Long-Tailed Graph Learning: An Energy-Guided Dual Decoupling Approach (https://arxiv.org/abs/2606.24237)
- **Prior Approaches**: 연합 그래프 학습(FGL)은 분산 클라이언트가 그래프 구조·특성을 공유하지 않고도 GNN을 공동 학습할 수 있게 해준다. 하지만 실제 데이터는 power law 기반의 multiclass long-tailed 분포를 보이며, 이때 Non-IID 환경에서는 희소한 tail 클래스의 노드가 더 적은 표본과 더 강한 heterophily를 동시에 겪는다. 기존 보정(logit adjustment, 보간/분해, 보편적인 topology-agnostic 보상 등)은 이런 구조적 결함을 제대로 분리하지 못해 majority 주변의 structural noise를 과적합하며 tail 표현이 악화되는 문제가 보고된다.

- **Core Contribution**: 논문은 long-tailed 문제의 원인을 ‘빈도 부족’뿐 아니라 ‘구조적 간섭(heterophilic interference)’으로 재정의하고, 이를 해결하는 FedEPD를 제안한다. FedEPD는 topological purification(토폴로지 정화)과 semantic recalibration(의미 재보정)을 분리하는 dual decoupling 패러다임을 도입한다. 또한 두 단계 교대 최적화와 spatial low-pass prototype injection을 통해 majority의 결정 경계는 보호하면서 tail의 정확도를 끌어올린다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 이질적 분포와 희소성 때문에 tail 노드가 불리한 이웃 구성을 갖는다는 점과 (2) 숫자/로짓 기반 보정을 곧바로 모델 전체에 적용하면 gradient 충돌로 인해 표현이 동시에 손상된다는 점이다. FedEPD는 distribution-aware Dirichlet energy pruning으로 heterophilic edges를 먼저 필터링해 구조적 노이즈를 줄인 뒤, 서버가 topological centrality를 기준으로 추출한 robust global prototypes를 저주파(low-frequency) 성분에만 주입한다. 마지막으로 graph encoder를 동결한 two-stage alternating optimization으로 majority decision boundary를 엄격히 보호하며 tail 보정을 수행한다.

- **Empirical Impact**: CoraFull, Amazon-Electronics 등 long-tailed 벤치마크에서 FedEPD는 Accuracy 최대 4.97%, Macro-F1 최대 5.48%(논문 요약 기준)까지 향상되는 state-of-the-art 성능을 보였다. 특히 head·medium 클래스의 분류 안정성을 유지하면서 tail 정확도를 유의미하게 개선해, 기존 방법의 representation degradation 가설을 실험적으로 뒷받침한다. 결과적으로 long-tailed FGL에서 빈도 보정 중심 접근을 넘어 구조적 간섭 분리의 중요성을 보여주며, 향후 federated heterophily 환경 설계에 직접적인 실무적 시사점을 제공한다.



### SP-Mind: An Autonomous Reasoning Agent for Spatial Proteomics Analysis (https://arxiv.org/abs/2606.24235)
Comments:
          23 pages, 6 figures. Accepted to ICML 2026. Equal contribution by Yucheng Yuan and Yuanfeng Ji

- **Prior Approaches**: 공간 단백질체(spatial proteomics) 분석은 이미지 등록, 잡음/배경 보정, 세포 분할, 정량, 표현형 추정, 공간 분석 등 다단계 파이프라인으로 구성돼 왔습니다. 기존 자동화는 Nextflow 같은 워크플로 관리로 도구들을 연결해 처리량과 재현성을 높였지만, 플랫폼·조직 맥락에 맞춘 수동 설정과 파이프라인의 정적 실행이 한계로 지적됩니다. 또한 LLM 에이전트의 일반 프레임워크는 다단계 도구 연쇄는 가능해도, 공간 생물학의 묵시적 제약과 도메인별 파라미터/입력 계약까지 제대로 반영하기 어렵습니다.

- **Core Contribution**: 이 논문은 자연어 질의를 받아 원시 멀티플렉스 조직 이미징부터 표현형(phenotype) 발견까지 end-to-end로 자동 실행하는 자율 에이전트 SP-Mind를 제안합니다. SP-Mind는 10+개의 공간 단백질체 전용 도구와 전문가가 큐레이션한 Spatial BioSkill Templates를 결합해, task-specific fine-tuning 없이도 도메인 절차에 맞게 워크플로를 구성합니다. 또한 ReAct 스타일 추론 루프에서 실행 결과를 관찰·진단·자기수정하며, 도메인 해석 가능성을 높이도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 다양한 조직/획득 조건에서 필요한 도구 선택과 파라미터 구성을 추론해야 하고, (2) 전 단계 출력의 데이터 형식·파일 상태를 다음 단계 도구가 요구하는 계약에 맞게 유지해야 한다는 점입니다. 저자들은 관찰 기반(data-first) 추론과 실행 전 단계 의존성 관리로 오류 전파를 줄이고, CodeAct식으로 Python 스크립트/쉘/파일 연산까지 허용해 사전 정의 API만으로는 못하는 조합 로직을 구현하게 했습니다. 또 스킬 템플릿에 추천 휴리스틱과 오류 복구 프로토콜을 포함해 자율 에이전트의 ‘방법론적 선견지명’을 보강했습니다.

- **Empirical Impact**: 평가를 위해 공간 단백질체 자율 오케스트레이션 전용 벤치마크 SP-Bench(총 102개 내추럴 언어 태스크, 18개 카테고리, 4 난이도)를 새로 구성했습니다. SP-Mind는 SP-Bench에서 실행 정확도 68.9%로 최고 베이스라인 대비 13%p 이상 향상되었고, 세포 정량/어노테이션 하류 작업에서도 일관된 개선을 보였습니다. 정성 사례에서도 SP-Mind가 ASHLAR 같은 도메인 도구를 올바르게 호출하고(예: 단계 드리프트 반영 등록), 표현형 구분에서 랭크/풍부도 기반 로직으로 일반 에이전트의 라인리지 붕괴를 완화함을 보여 의미 있는 실증 효과를 입증했습니다.



### FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning (https://arxiv.org/abs/2606.24231)
Comments:
          Project page: this https URL

- **Prior Approaches**: 멀티모달 자율주행 계획은 크게 스코어링 기반과 anchor 기반으로 나뉜다. 스코어링 기반은 고정된 action vocabulary에 대해 시뮬레이션 보상으로 촘촘한 감독을 주지만, 판별모델이라 새 제안을 생성하기 어렵고 어휘 밖 적응에 한계가 있다. 반면 anchor 기반은 anchor로 동적 제안을 만들지만, 장면당 단일 ground-truth 궤적에만 winner-takes-all 감독이 걸려 sparse supervision 문제와 단일 궤적 모방의 병폐(결과 인지 부족, shortcut learning 등)가 생긴다.

- **Core Contribution**: FlowR2A는 보상 기반 시뮬레이션 레이아웃을 판별 타깃에서 생성 조건으로 재해석해 scoring과 anchor의 긴장을 푼다. 즉 reward-conditioned action distribution p(a|r)를 직접 학습하는 단일 생성모델로, dense trajectory-reward pair를 통해 안전·진행·안락·규칙 준수에 대한 action-결과 상관을 내부화하도록 설계했다. 결과적으로 “촘촘한 보상 감독”과 “제안 생성 능력”을 한 프레임워크에서 동시에 확보한다.

- **Technical Challenges**: 핵심 난제는 고보상 조건을 주면 생성기가 feasible boundary를 넘나들며 하드 안전 제약을 깨기 쉬운 점이다. FlowR2A는 TTC와 DAC 같은 이진/단일 타깃을 per-timestep 배열로 세분화해 시간 해상도를 높이고, EP와 PDM 같은 연속 보상은 학습 중 Gaussian noise로 label smoothing을 적용해 과도한 보상 의존을 막는다. 또한 flow-matching 기반 action decoder에 reward guidance(classifier-free guidance)와 zero-shot anchored sampling을 얹어 테스트 시 보상 목표와 앵커 구조를 동시에 컨트롤 가능하게 한다.

- **Empirical Impact**: NAVSIM v1(v2)에서 FlowR2A는 각각 92.8 PDMS(88.9 EPDMS)로 state-of-the-art를 달성하며, 안전(NC, TTC)과 진행(EP) 지표에서 동시 개선을 보인다. 특히 proposal 수가 적을 때도 강한 성능을 내고, 평균적으로는 강한 anchor 기반 대비 on-distribution 제안 품질이 더 일관적이며 분산도 낮다. 문맥적으로는 dense 보상을 “조건”으로 학습해 p(a|r) 전체 분포를 모델링하는 접근이, 단일 제안 선택뿐 아니라 다중 제안의 실용성을 크게 끌어올린다는 점을 실증했다.



### Exploring the relationship between human-centric AI and firm idiosyncratic risks (https://arxiv.org/abs/2606.24224)
Comments:
          Information Systems Frontiers, 2026

- **Prior Approaches**: Industry 5.0 맥락에서 human-centric AI(HCAI)는 많이 논의됐지만, 기업의 고유 리스크(idiosyncratic risks, IR)에는 어떤 영향이 있는지 연구가 부족했다. 기존 연구들은 주로 시스템 성능이나 윤리 준수 자체를 다뤘고, 기업 수준에서 주가 변동성을 체계적 요인과 분리해 해석하는 IR 관점은 상대적으로 약했다.

- **Core Contribution**: 이 논문은 HCAI를 단순 윤리 프레임이 아니라, situated AI strategy로 개념화해 이해관계자 기대를 정렬함으로써 AI 관련 윤리 리스크를 낮추고 결과적으로 IR을 줄인다고 제안한다. 또한 social-technical systems 관점에서 디지털화, 운영 효율, 임원 지분, IT 배경 CEO가 HCAI-IR 관계를 어떻게 조절하는지까지 함께 검증한다.

- **Technical Challenges**: HCAI의 효과를 IR로 연결하려면, 기업별 주가 변동을 체계적 요인과 분리해 측정 가능한 형태로 설계하고 다중 요인의 혼재를 통제해야 한다. 저자들은 2015~2023년 중국 상장기업의 multi-source 패널 데이터로 분석하고, 조절변수들(디지털화·운영 효율·임원 지분·IT 배경 CEO)의 방향성과 상호작용 효과를 함께 추정해 이 문제를 완화했다.

- **Empirical Impact**: 분석 결과 HCAI는 기업 IR을 낮추는 것과 연관됐다. 특히 digitalisation과 executive shareholding은 위험을 줄이는 효과를 강화했지만, operational efficiency와 CEOs with IT background는 오히려 그 효과를 약화시키는 상반된 결과가 관찰됐다. 이로써 HCAI 거버넌스와 기업의 재무적 위험 관리가 연결된다는 실증적 근거를 제공하며, AI 시대의 의사결정에 실무적 시사점을 준다.



### Navigating User Behavior toward Personalized Multimodal Generation (https://arxiv.org/abs/2606.24196)
Comments:
          16 pages, 15 figures, 5 tables. Code is available at this https URL

- **Prior Approaches**: 기존 text-to-image/video 파이프라인은 사용자가 잘 구성된 creation instruction을 제공한다는 가정에 기대어 왔다. 또 prompt enrichment, multi-agent, reference/layout/identity 기반 conditional control 등은 세부 프롬프트를 보강하지만, 소비자 사용자의 암묵적 취향을 실제 생성 가능한 instruction으로 연결하는 경로는 약했다. 결과적으로 생성물이 일반적이거나 사용자의 실수요와 어긋나는 문제(behavior-to-instruction gap)가 남아 있었다.

- **Core Contribution**: 이 논문은 personalized content generation을 정식 문제로 다루며, 사용자의 interaction history를 “언어 모델이 실행 가능한 creation instruction”으로 변환하는 fθ를 학습한다. 이를 위해 NaviGen을 제안하고, 행동 신호와 의미 신호를 한 토큰 스트림에서 동시에 다루는 dual-identifier 표현(CID/TID)을 도입한다. 또한 인스트럭션 작성 능력이 부족한 문제를 해결하기 위해 두 단계 SFT+RL 파이프라인으로 preference 추론과 instruction writing을 별도 정렬한다.

- **Technical Challenges**: 핵심 기술적 난제는 (C1) 행동 데이터를 LM이 추론 가능한 형태로 표현하는 표현 갭과, (C2) 선호를 아는 것과 generation-ready instruction을 쓰는 능력 갭이다. NaviGen은 residual vector quantization 기반 collaborative identifier(CID)와 LLM으로 압축·정렬한 의미 브리지 textual identifier(TID)를 결합해 표현 갭을 줄였고, SFT에서는 evolutionally searched supervision(LLM judge)을 통해 instruction 작성 자체를 학습시킨 뒤 GRPO로 downstream 목적에 맞춰 정렬한다. 특히 계층형 CID 보상과 instruction-aware(삼각형 self-consistency) 보상을 함께 최적화해 instruction의 구체성·타깃 의미 고정·자기일관성을 동시에 강화한다.

- **Empirical Impact**: product, game, short-video 등 3개 도메인에서 NaviGen은 personalized image/video generation 품질을 전반적으로 개선했고, 다음 항목 예측에서도 CID-space 성능이 향상되었다. 정량적으로는 relevance, novelty, aesthetic, consistency 전반에서 경쟁력을 보였고, 특히 video에서 여러 요인 간 trade-off를 관리하는 양상이 관찰됐다. 또한 예시 비교에서 기존 방식이 역사 단서를 거칠게 반영한 반면, NaviGen은 CID 수준의 선호 전이와 TID 수준의 의미 고정을 결합해 더 구체적이고 시각 생성이 가능한 instruction을 만든다는 점이 정성적으로도 드러났다.



### Data Scale, Not Latency, Shapes Cross-Lingual Encoder Transfer in Streaming ASR (https://arxiv.org/abs/2606.24169)
- **Prior Approaches**: 스트리밍 ASR 전이학습 연구는 주로 random 초기화 대비 pretraining 효과를 확인했지만, 실무에서 중요한 ‘두 사전학습 인코더(ML vs EN) 중 무엇을 선택할지’에 초점을 덜 맞췄습니다. 또한 스트리밍 지연(latency)을 평가축으로 체계적으로 스윕하지 않아, 저데이터 환경에서의 이점이 지연 조건에서 얼마나 지속되는지 불명확했습니다.

- **Core Contribution**: 이 논문은 0.6B cache-aware FastConformer transducer를 기반으로, 새로운 언어 적응에서 multilingual(ML) 인코더와 English-only(EN) 인코더 중 초기화를 어떻게 고를지 대규모로 비교합니다. 특히 목표 언어 데이터 규모(100h~2500h), 세 가지 streaming tier(160/560/1120ms), 최대 네 개 공개 테스트셋을 동시에 스윕해 ‘데이터 제한 vs 지연 제한’ 중 어디가 핵심인지 명확히 규명합니다.

- **Technical Challenges**: 핵심 과제는 (1) streaming 환경에서의 제한된 문맥과 캐시 구조가 초기화 이점을 증폭/감쇠시키는지, (2) deployment에 가까운 4-bit weight-only quantization이 그 차이를 유지하거나 바꾸는지였습니다. 저자들은 동일 모델/토크나이저/파인튜닝 레시피를 고정한 채 EN과 ML 인코더를 공정 비교하고, Slavic-pivot·hybrid-encoder ablation로 전이 신호가 인코더의 어느 층에 있는지까지 추적했으며, 양자화에서도 측정 가능한 비용을 함께 정리했습니다.

- **Empirical Impact**: 결과적으로 ML 초기화의 EN 대비 WER 우위는 ‘데이터가 적을 때만’ 의미가 있고, streaming latency에 의해 크게 달라지지 않았습니다. 예컨대 FLEURS에서 160ms 기준 평균 EN–ML WER 갭은 100h에서 +4.21pp였지만 2500h에서는 +0.20pp로 거의 소멸했으며, 각 streaming tier 간 갭의 변동폭도 약 1pp 수준에 머물렀습니다. 또한 동일 560ms 조건에서 4-bit weight-only 인코더 양자화는 인코더 크기를 약 3배 줄이면서 FLEURS WER을 평균 0.5pp 정도만 올려, 저데이터 이점과는 별개로 지연·양자화 의사결정을 독립적으로 내릴 수 있다는 가이드라인을 제공합니다.



### An Introduction to Causal Reinforcement Learning (https://arxiv.org/abs/2606.24160)
- **Prior Approaches**: 기존 인과 추론은 환경에 대한 데이터와 지식을 결합해 반사실(가정된 현실)이 어떻게 달라졌을 때 결과가 바뀌는지 추론하는 데 초점을 둔다. 반면 강화학습(RL)은 보상이나 regret 같은 목표를 최적화하는 정책을 학습하되, 실제 배치 환경에서의 탐험-시도(trial-and-error)를 기반으로 발전해 왔다. 두 분야는 “counterfactual relations”를 다루면서도, 문헌과 방법이 거의 상호작용하지 못했다는 한계가 있다.

- **Core Contribution**: 이 논문은 인과 추론과 강화학습이 같은 기본 구성요소인 반사실 관계를 다룬다는 점을 명시적으로 수학화해, 두 분야를 한 프레임으로 잇는 관점을 제안한다. RL에서의 환경 배치가 특정 구조적 인과모형(Structural Causal Model, SCM)으로 분해될 수 있고, 일반적인 RL 세팅이 이미 이런 모델을 암묵적으로 인코딩한다고 본다. 이를 바탕으로 online, off-policy, causal calculus learning 같은 서로 무관해 보이던 학습 양식을 통합적으로 다룬다.

- **Technical Challenges**: 핵심 기술적 난제는 RL 환경을 “인과적 불변성(causal invariance)”을 갖는 자율 메커니즘들의 집합으로 어떻게 구조적으로 모델링해, 반사실 추론과 학습을 연결할지다. 논문은 이를 SCM 형태의 분해로 포착해 학습 모드를 하나의 수학적 틀로 통일하고, 개입 위치(where to intervene) 같은 분석 차원을 학습 문제로 끌어온다. 또한 generalized policy learning, imitation learning, counterfactual learning을 인과 렌즈로 재정의하며 causal reinforcement learning(CRL)이라는 범주를 정립한다.

- **Empirical Impact**: 추상적 관점 제시에 더해, 제안한 통합 프레임이 다양한 학습 설정을 하나의 반사실 학습 스펙트럼에서 비교 가능하게 만든다는 점이 의미가 있다. 즉 RL과 인과 추론을 나란히 다루며, 기존에는 분리돼 있던 문제 설계와 평가 기준을 인과적으로 해석할 수 있게 된다. 결과적으로 CRL은 새로운 연구 질문(개입 설계, 반사실 학습 차원, off-policy와 인과 계산의 연결)을 촉진하는 실질적 영향력을 기대하게 한다.



### The Geometry Behind Diffusion and Flow Matching: Gradient Flows and Geodesics in Wasserstein Spac (https://arxiv.org/abs/2606.24157)
- **Prior Approaches**: 기존 확산 모델은 forward process를 DDPM/SMLD의 이산 마코프 체인으로 만들고, 이를 연속 극한에서 SDE로 해석한 뒤 score를 학습해 reverse-time 생성 절차를 구성했다. 다만 이 과정에서 Fokker-Planck와 연관된 확률공간의 기하학(gradient flow, JKO 등)은 보조 단계로 취급되는 경우가 많았다. 또한 Flow Matching 계열은 ODE/continuity equation 관점에서 별개 이론처럼 전개되어 diffusion과의 정확한 대응이 모호했다.

- **Core Contribution**: 논문은 확률질량의 공간 𝒫2(ℝd)를 Wasserstein 기하 위의 하나의 (formal) Riemannian manifold로 보고, 확산 모델과 Flow Matching을 같은 공간 위의 두 variational principle로 정확히 연결한다. diffusion은 free energy의 Wasserstein gradient flow(초기값 문제)로, Flow Matching은 Wasserstein geodesic/최소 작용 경로(경계값 문제)로 해석된다. 결과적으로 두 계열은 서로 다른 경로를 따라 동일한 종착 엔드포인트를 재현한다는 관계를 정식화한다.

- **Technical Challenges**: 핵심 기술적 난제는 “Fokker-Planck(2차 미분항 포함)”와 “continuity equation(1차 수송)”이 같은 확률 경로를 설명한다는 관점을 일관된 언어로 통합하는 것이다. 논문은 확산의 확률유동을 v_t=f−(g^2/2)∇log ρ_t 형태의 effective velocity로 재표현해, diffusion의 diffusion term을 수송 항의 속도장에 흡수시키는 정체(disguised transport)로 두 방정식의 동치성을 보인다. 그 위에서 diffusion의 denoising 한 스텝이 JKO scheme의 한 걸음과 대응되고, Flow Matching의 geodesic 기반 학습도 최소작용 곡선의 ODE 생성으로 정리된다.

- **Empirical Impact**: 이 통일 프레임은 DDPM, DDIM, NCSN/SMLD, Energy Matching 같은 diffusion 계열과 Flow Matching/OT-geodesic 계열을 “하나의 기하학에서 나온 서로 다른 샘플링 경로”로 재해석하게 해, 설계 선택이 왜 비슷한 궤적(trajectory)을 공유하는지 설명력을 제공한다. 특히 geodesic를 따라 고정된 엔드포인트를 만족시키면 샘플링은 더 적은 스텝의 결정적 ODE로 귀결되어 효율적 생성의 이론적 근거가 된다. 또한 diffusion(gradient-flow)과 Flow Matching(geodesic) 및 Energy Matching(평형/Free energy 관점)을 동일 manifold 위에서 연결함으로써, 향후 모델 계열 간 이론 비교와 아키텍처 설계의 가이드가 될 의미를 가진다.



### T2D-Bench: Evidence-Gated Evaluation of LLM Outputs for Type 2 Diabetes Using a Multi-Layer Clinical-Lifestyle Knowledge Graph (https://arxiv.org/abs/2606.24145)
Comments:
          7 pages, 2 figures, 2 tables. Accepted as a poster at AMIA 2026 Annual Symposium

- **Prior Approaches**: 기존 연구는 LLM이 제2형 당뇨병(type 2 diabetes)에 대해 임상적으로 그럴듯한(유창한) 권고를 생성하는지 주로 보았지만, ADA 같은 지침의 제약을 실제로 만족하는지나 생활습관 관련 혈당(glycemic) 주장에 근거가 명시되는지까지는 엄격히 검증하지 못하는 경우가 많았습니다. 또한 근거 누락을 자동으로 식별하고 수정 가능한 형태로 점검하는 평가 프레임워크가 부족했습니다.

- **Core Contribution**: 이 논문은 LLM 권고가 명시적이고 그래프 기반으로 검증 가능한 “근거 요구사항”을 충족하는지 시험하는 재현 가능 벤치마크이자 evidence-gated 평가 프레임워크인 T2D-Bench를 제안합니다. T2D-Bench는 임상의학 지식(UMLS, DrugBank, SIDER)과 계산 가능한 ADA Standards of Care 규칙을 연결하고, 생활습관 지식과 혈당 실험실 효과를 기전( mechanistic ) 브리지로 묶은 다층 임상-라이프스타일 지식 그래프를 사용합니다.

- **Technical Challenges**: 핵심 과제는 (1) 생활습관과 혈당 관련 주장 사이의 근거 경로를 “계산 가능한” 형태로 만들고, (2) LLM 출력에서 근거 없는 생략(unsupported omissions)을 검출하며, (3) 검증기 기준에 맞게 수정(constrained revision)하는 것이었습니다. 저자들은 evidence gate로 근거 경로를 체크해 누락을 탐지하고, 검증기 수준의 compliance에 도달할 때까지 조건부 수정으로 출력을 재작성하도록 구성했습니다.

- **Empirical Impact**: 100개 구조화된 비네트(vignette)를 대상으로 한 실험에서 GPT-4o-mini와 GPT-4o의 기본 출력은 근거 경로 체크를 각각 35%, 33%에서 실패해, 유창함만으로는 지침·근거 충족이 보장되지 않음을 보여주었습니다. evidence gate와 constrained revision을 적용하면Verifier 수준의 근거 요구사항을 만족하도록 조정되며, 당뇨병 특화 LLM에서 근거 없는 임상적 생략을 명시화·측정·수정 가능하게 한다는 점에서 의의가 있습니다.



### OmniPath: A Multi-Modal Agentic Framework for Auditing Wheelchair Accessibility (https://arxiv.org/abs/2606.24129)
Comments:
          10 pages, 13 figures. Submitted to IEEE COMPSAC 2026. OmniPath: A Multi-Modal Agentic Framework for Auditing Wheelchair Accessibility

- **Prior Approaches**: 기존 휠체어 내비게이션은 OSM 같은 2D 벡터 지도에서 경로의 연결성은 잘 잡아도, 실제 노면이 주는 micro-barriers(단차·급경사·횡단 기울기 등)를 충분히 전달하지 못했다. 또 crowdsourcing이나 사용자 경험 기반 경고는 전 사용자 데이터가 없는 구간에서는 위험을 “모르는 채” 지나가기 쉬워, 접근성 함정이 그대로 남는 문제가 있었다. LiDAR를 활용하는 연구도 있었지만, ADA 규격에 맞춘 미세 지형 위반을 사전 점검하는 agentic 접근과는 아직 연결이 약했다.

- **Core Contribution**: OmniPath는 정적 지도 제공을 넘어, LiDAR 기반 3D 환경 감사를 선제적으로 수행하는 agentic 시스템을 제안한다. OSM의 네트워크 토폴로지에 USGS 3DEP 항공 LiDAR를 결합해 보행 환경을 3D로 재구성하고, 경로를 “가상으로 순회”하며 0.5m 단위로 ADA 위반(주행 경사·횡단 경사·수직 불연속)을 정량화한다. 이렇게 각 네트워크 구간에 Accessibility Impedance Score를 부여해 위험을 Mild~Critical로 등급화하고, 사용자가 나가기 전에 장벽을 예측하는 접근성 데이터 소스로 전환한다.

- **Technical Challenges**: 가장 큰 기술 과제는 지도(2D 연결)와 지형(LiDAR의 3D 표면) 간 정합을 맞추는 것과, 한 구간을 휠체어의 실제 접지/이동 스케일에 대응하는 미세 단위로 쪼개는 것이었다. OmniPath는 지리좌표계를 공통 CRS로 정렬한 뒤, OSM 엣지를 0.5m 슬라이딩 윈도우로 분해하고 해당 구간에 포함되는 LiDAR 점을 연관(sampling/association)시켜 경사와 불연속을 계산한다. 이후 ADA 기준 대비 초과 정도를 Proportional Exceedance로 정규화하고, 주행 경사·횡단 경사·수직 불연속에 가중치를 둔 Weighted Severity Score로 통합해 위험 등급을 산출한다.

- **Empirical Impact**: National Mall 구간에서 1,485개 사이드워크 세그먼트를 평가했으며, LiDAR 신뢰 구간 기준으로 97개 세그먼트에서 미세 감사가 수행됐다(나머지는 수목 캐노피 가림·비행경로 공백 등으로 제외). 분석된 구간에서는 ADA 위반이 100% 관측됐고, 그중 Critical 234를 포함해 총 331개의 위반이 탐지되었으며 Critical 또는 Severe가 97.9%로 나타났다. 또한 200개 현장 실측(field surveys)과의 검증에서 Severe와 critical 카테고리에 대해 F1-score 0.60/0.58을 기록해, 표준 지도에 “보이지 않던” 접근성 장벽을 진단·우선순위화하는 데 의미 있는 성능을 보였다.



### VeryTrace: Verifying Reasoning Traces through Compilable Formalism and Structured Verification (https://arxiv.org/abs/2606.24124)
Comments:
          Accepted at LM4Plan Workshop @ ICML 2026

- **Prior Approaches**: Chain-of-Thought(CoT) 계열은 긴 추론을 잘 생성하지만, 초기에 생긴 논리·계산 오류가 이후 단계로 조용히 전파되는 취약성이 남아 있습니다. End-to-end 검증은 최종 답만 보거나, self-consistency/자체 투표는 다수 샘플 비용만 늘리면서도 실패 원인을 단계 단위로 진단하기 어렵습니다. 반면 정리증명기(Lean/Coq 등)는 엄밀하지만 도메인별 형식화 비용과 전문 문법 요구 때문에 일반 문제 유형으로의 전이가 어렵습니다.

- **Core Contribution**: VeryTrace는 자연어 추론 트레이스를 ‘컴파일 가능한 프로그램’처럼 보고, DSL(Domain-Specific Language)로 단계 의존성과 정량 계산을 구조화해 zero-shot 검증·수정까지 수행하는 프레임워크를 제안합니다. 핵심은 기계적으로 검증 가능한 부분은 결정적(deterministic) 검증으로 처리하고, 기계화가 어려운 의미 판단만 단계 스코프의 LLM audit로 제한하는 하이브리드 방식입니다. 이를 통해 단계별 오류 국소화와 해당 구간만 재작성하는 verification-driven repair를 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술 문제는 (1) 자연어 추론의 문장들이 어떤 전제에 의존하는지, (2) 계산/제약/논리적 귀결을 어떤 형태로 기계 검증 가능한 표현으로 바꿀지, (3) 의미적(commonsense·자연언어) 부분을 과도하게 맡기지 않는 균형을 맞추는 것입니다. VeryTrace는 2-stage DSL 변환(문맥 K만 먼저 추출한 뒤, 추론 트레이스를 DSL로 변환)으로 컨텍스트 누락·환각 편향을 줄이고, COMPUTE는 실행 가능한 식으로 검증, ASSUME/DEDUCE는 고정된 deduction schema와 구조화된 audit로 판단합니다. 또한 불변(invariant)/목표(goal) 제약을 상태 전반/최종에 각각 엄격히 적용하고, forward-reference·순환 의존성 같은 구조 오류도 사전 차단합니다.

- **Empirical Impact**: 실험은 AIME 2025(경시 수학), LLM-BabyBench(로보틱스 계획), CLUTRR(친족 관계 추론) 3개 도메인에서 state-of-the-art LLM을 대상으로 zero-shot 조건에서 수행되며, VeryTrace가 강한 zero-shot 기준선 대비 정확도를 개선한 것으로 보고됩니다. 특히 긴 지평 추론에서 단계별 검증·수정이 오류 전파를 줄이는 효과를 보였고, CoVe/NP 같은 프롬프트 기반 검증 대비도 성능 우위를 확인했다고 합니다. 도메인별 학습이나 in-context examples 없이 추론 트레이스의 형식화 검증이 precision과 generalization을 함께 가져올 수 있음을 시사합니다.



### ReMMD: Realistic Multilingual Multi-Image Agentic Verification for Multimodal Misinformation Detection (https://arxiv.org/abs/2606.24112)
Comments:
          The project is available at this https URL

- **Prior Approaches**: 기존 멀티모달 허위정보 탐지는 짧은 캡션, 단일 이미지-텍스트 쌍, 이진 라벨, 또는 하나의 조작 소스에 초점을 둔 경우가 많아 실제 배포 환경과 불일치가 컸습니다. 또한 에이전트형 검증은 가능해졌지만, 현실적인 증거 탐색과 비용 부담 때문에 장문·다중 이미지·다중 출처 설정을 충분히 반영하기 어려웠습니다.

- **Core Contribution**: 이 논문은 현실적 조건을 반영한 ReMMD 프레임워크로, 멀티이미지·다국어·다중 라벨을 갖춘 벤치마크 ReMMDBench와 증거를 관리하는 에이전트 ReMMD-Agent를 제안합니다. ReMMDBench는 500샘플(다중 언어, 장문 텍스트, 최대 다수 이미지, 5-way veracity, distortion 라벨, provenance 및 rationale)로 ‘등급형 진위 판정’과 ‘왜곡 진단’을 함께 평가하도록 설계됐습니다.

- **Technical Challenges**: 핵심 기술 문제는 “판정”보다 “증거 선택과 정렬”로, 장문 내 다수 주장·이미지·출처 중 무엇을 어떻게 확인해 중심 주장에 어떤 영향을 주는지 추적해야 한다는 점입니다. ReMMD-Agent는 게시물을 원자적(atomic) 포인트로 분해해 검색 질의 대상을 명확히 하고, persistent memory에 증거를 저장·재사용한 뒤 structured judge가 L1/L2/L3(진위/왜곡/근거)을 evidence state 기반으로 예측하도록 구성해 비용과 추적 가능성을 동시에 노렸습니다.

- **Empirical Impact**: 실험에서 ReMMD-Agent는 GPT-5.2 기준 5-way veracity 정확도 41.80%, macro-F1 39.12%로 비교 에이전트 대비 최상 성능을 보였고, 비용도 MMD-Agent 대비 17.5%, T2-Agent 대비 79.9% 절감했습니다. 또한 같은 파이프라인이 MMFakeBench로 전이될 때도 강한 성능을 보여, 개선 효과가 특정 라벨 체계에만 국한되지 않음을 시사합니다.



### Exploring Academic Influence of Algorithms by Co-occurrence Network Based on Full-text of Academic Papers (https://arxiv.org/abs/2606.24099)
- **Prior Approaches**: 기존 연구는 특정 알고리즘을 단독으로 성능이나 인용 등으로 평가하는 경우가 많아, 논문들 사이에서 알고리즘이 서로 연결되며 만들어내는 ‘집합적 영향’을 충분히 다루지 못했다. 또한 알고리즘 언급은 인기 지표로 활용되더라도, 네트워크 관점에서의 구조와 시간에 따른 중심성 변화는 상대적으로 덜 탐구됐다.

- **Core Contribution**: 이 논문은 학술 논문의 전체 텍스트를 기반으로 NLP에서 알고리즘 공출현 네트워크를 대규모로 구축하고, 이를 통해 알고리즘 영향력을 네트워크 관점에서 정량화한다. 전체(누적) 네트워크와 연도별 네트워크를 함께 분석하며, 알고리즘이 분야 내에서 어떻게 ‘중심 위치’를 차지하고 변화하는지 보여준다.

- **Technical Challenges**: 핵심 과제는 방대한 본문에서 알고리즘 엔티티를 정확히 추출하고, 공출현을 일관된 규칙으로 네트워크에 반영하는 것이다. 연구진은 deep learning 모델로 알고리즘 엔티티를 추출한 뒤 전역·누적·연도별 공출현 네트워크를 만들고, 여러 centrality 지표로 집단 영향(전체 및 시간 축)을 비교·분석했다.

- **Empirical Impact**: 결과로 네트워크는 약 2년대가량에 걸쳐 점점 더 촘촘해지는 복잡계 특성을 보이며, 전통적 고성능 알고리즘과 서로 다른 연구 시기를 잇는 교차 지점의 알고리즘이 높은 인기·통제·중심성 및 균형 잡힌 영향력을 갖는 경향이 나타났다. 또한 영향이 감소할 때는 먼저 핵심 네트워크 위치를 잃고, 이후 다른 알고리즘과의 연관성이 약화되는 양상이 관찰됐다. 4년대 이상 출판물을 아우르는 최초 규모의 알고리즘 공출현 네트워크 분석으로, 향후 알고리즘·학자·태스크를 연결하는 네트워크 연구의 기반을 제공한다.



### Beyond Trajectory Imitation: Strategy-Guided Policy Optimization for LLM Reasoning (https://arxiv.org/abs/2606.24064)
- **Prior Approaches**: 기존 추론 증류는 주로 강한 모델의 해결 과정(trajectory)을 그대로 따라 하게 하는 instance-level SFT 기반 접근이 많았고, 이는 특정 문제의 단계 패턴을 암기하기 쉬워 새로운 문제로의 일반화가 제한된다. RL 관점의 GRPO/보상 결합이나 proximal constraint를 추가해도 대체로 “무엇을 답으로 내는지” 또는 “어떤 단계열을 생성하는지”가 전이 단위로 남는다. 또한 노이즈가 섞인 궤적 데이터 품질 민감성, exposure bias 같은 학습 불안정 이슈도 제약으로 지적된다.

- **Core Contribution**: 이 논문은 Strategy-Guided Policy Optimization(SGPO)로 증류 대상을 ‘해결 궤적’에서 ‘재사용 가능한 문제 해결 전략’으로 바꾼다. 강한 모델 응답에서 계산/정답을 제외한 전략 설명(문제 유형, 접근법, 절차)을 추출해, 학생 모델이 전략을 조건으로 받았을 때의 행동 분포 변화를 통해 “어떻게 추론하는지”를 내재화하도록 유도한다. 특히 학습 시 strategy-guided와 autonomous 두 궤적을 함께 만들고, 증류는 trajectory 복제가 아니라 분포 변화(토큰 확률 변화) 수준에서 수행한다.

- **Technical Challenges**: 핵심 난제는 전략 조건부에서 드러나는 지식이 추론 과정에서 학생의 unguided 정책으로 “안정적으로” 전이되게 만드는 것이다. SGPO는 토큰 레벨 forward-KL을 사용해 strategy conditioning이 바꾼 다음 토큰 분포의 차이를 선택적으로 흡수하게 하며, proximal 제약(trajectory 및 token 단계)과 reachable target selection, KL clipping으로 업데이트 폭과 엔트로피 붕괴 위험을 제어한다. 더불어 적응형 인스턴스 가중치로, autonomous 탐색이 실패할 때는 증류 압력을 키우고 능력이 생기면 줄여 수동 스케줄링 없이 동작하도록 설계했다.

- **Empirical Impact**: 수학 추론 4개 벤치마크(MATH500, AMC 23, OlympiadBench, AIME 24)에서 Qwen2.5와 Llama-3.2 두 모델 계열을 대상으로 SGPO는 SFT, on-policy RL, 그리고 강한모델 지식을 쓰는 하이브리드 기준선을 일관되게 능가했다. 특히 Qwen2.5-7B-Instruct에서 최강 베이스라인 대비 평균 점수를 2.2점 끌어올렸고, base model 능력이 커질수록 성능 향상이 더 커지는 complementary scaling 경향도 확인됐다. 분석에선 forward-KL이 토큰 단위에서 전략에 중요한 의사결정 지점에 자연스럽게 학습 압력이 모이며, 직접 trajectory imitation(SFT)보다 더 선택적 증류 신호를 제공한다는 점이 실험적으로 뒷받침됐다.



### Ensemble Feature Selection and Harris Hawks Optimization for Explainable Mental Health Risk Prediction in Female Sex Workers (https://arxiv.org/abs/2606.24047)
Comments:
          Accepted and presented at the 2026 8th IEEE Symposium on Computers & Informatics (ISCI 2026). To appear in IEEE conference proceedings

- **Prior Approaches**: 기존 연구는 불충분한 feature selection, 단순 분류기, 최적화가 부족한 모델, 그리고 XAI 부재로 인해 FSW(여성 성노동자) 집단의 고차원·비선형·불균형 위험 패턴을 충분히 포착하지 못하는 경우가 많았다. 또한 폭력 노출, 사회경제 변수, 건강 지표가 얽힌 상호작용을 함께 모델링하지 못해 정확도와 안정성, 해석 가능성이 동시에 떨어지는 한계가 지적된다.

- **Core Contribution**: 이 논문은 ANOVA+mutual information 기반 ensemble feature selection으로 핵심 변수를 추린 뒤, Harris Hawks Optimization(HHO)로 튜닝한 logistic regression(LR)을 결합한 하이브리드 예측 모델을 제안한다. 여기에 LIME 같은 explainable AI(XAI)를 붙여 예측에 기여한 외상·직업 관련 요인을 개인 단위로 설명하고, 취약집단 조기 개입을 돕는 도구로 확장한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 폭력·낙인·경제 취약성 등 복합 요인으로 인한 고차원 데이터에서 신호 대비 잡음을 줄이고, (2) 불균형한 우울(positive) 라벨에서 재현율을 최대화하며, (3) 예측 결과를 임상/현장 실무자가 납득할 수 있게 설명하는 것이다. 저자들은 전처리 후 ANOVA+mutual information으로 feature를 11개로 축소하고, HHO로 LR 파라미터 탐색을 수행하며, LIME으로 local 수준 설명을 집계해 PTSD 및 폭력 관련 변수가 일관되게 중요하다는 점을 확인했다.

- **Empirical Impact**: 3,005명의 FSW 데이터에서 제안 모델은 정확도 95.78%, F1-score 95.77%, AUC 0.96으로 전통적 분류기 대비 성능이 우수했다. LIME 기반 분석에서는 post-traumatic stress(PTSD), client-related violence(클라이언트 관련 폭력), occupational factors(직업 요인)가 우울 예측의 주요 기여 요인으로 나타났고, 설명의 인구집단 안정성도 높게 보고됐다. 저자들은 이 결과가 취약집단을 위한 evidence-based 표적 정신사회적 돌봄과 건강 서비스 계획 수립에 바로 활용될 수 있는 XAI형 조기 스크리닝 경로를 제공한다고 강조한다.



### Breaking the Filter Bubble: A Semantic Pareto-DQN Framework for Multi-Objective Recommendation (https://arxiv.org/abs/2606.24042)
Comments:
          IEEE International Conference on Responsible Artificial Intelligence (IRAI) - 2026

- **Prior Approaches**: 기존 추천 시스템은 사용자 즉시 만족을 단일 목표로 최적화하는 경우가 많아, 시간이 지날수록 노출이 좁아지며 필터 버블과 의미적 동질화가 심화된다고 알려져 있다. RL 기반 DQN도 장기 보상을 다루지만 보상 스칼라화(단일 점수화) 중심 설계라 다양성·공정성과 같은 상충 목표를 구조적으로 조율하기 어렵다. 다목적 MORL이 활발히 연구돼 왔지만, 추천 맥락에서 의미 붕괴를 유발하는 피드백 루프를 직접 끊는 방식은 여전히 제한적이었다.

- **Core Contribution**: 이 논문은 추천을 의미 기반 semantic multi-objective Markov decision process로 정식화하고, engagement·diversity·fairness를 비가산(non-aggregable) 보상 신호로 분리해 함께 학습하는 multi-objective reinforcement learning 프레임워크를 제안한다. 핵심 에이전트로 Pareto-DQN을 쓰되, reward를 정적 가중치로 한 점수로 합치지 않고 Pareto frontier를 따라 선택하도록 설계한다. 또한 all-MiniLM-L6-v2 Sentence Transformer로 고신뢰 의미 임베딩을 구성해, 의미 관련성·zero-shot 일반화까지 한 파이프라인에서 다룬다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 카탈로그 전체에서 Q값을 계산하지 않으면서도 (2) 상충 목표의 Pareto frontier를 안정적으로 매핑하고 (3) 필터 버블을 만드는 상태(사용자 선호) 축소를 학습 중에 실제로 방해하는 것이다. 저자들은 후보 풀을 KNN 기반 고친화 아이템과 long-tail 주입을 섞는 stratified candidate pooling으로 제한하고, ρ 비율로 공정성 경로를 항상 남긴다. 또 사용자 선호 드리프트를 embedding의 정규화된 지수이동 평균으로 모델링해, 탐욕적 engagement 최적화가 의미 축소를 일으키는 동역학을 환경에 포함시키며, hypervolume 기반 action selection으로 Pareto 준수 선택을 수행한다.

- **Empirical Impact**: MovieLens-Small 오프라인 실험에서 Pareto-DQN은 hypervolume 기반 선택으로 의미 붕괴를 유발하는 피드백 루프를 끊어, fairness·diversity에서 유의미한 개선을 보이면서도 engagement 하락은 제한적이라고 보고한다. 특히 fully greedy 정책 평가에서 Pareto-DQN의 목적 공간 분포가 더 넓게 펼쳐져(non-dominated episode 다수), 기준선 DQN보다 훨씬 다양한 Pareto manifold를 형성한다. 사용자 상태-trajectory의 임베딩 분산(trace(Cov))도 Pareto-DQN이 더 크게 유지해 필터 버블 완화 지표를 실증적으로 뒷받침하며, ‘책임의 비용(price of responsibility)’ 관점에서도 다양성 향상 대비 engagement 손실이 관리 가능한 수준임을 보여준다.



### Can Language Model Agents be Helpful Circuit Explainers in Mechanistic Interpretability? (https://arxiv.org/abs/2606.24026)
Comments:
          23 pages, 4 figures, 14 tables

- **Prior Approaches**: 기존 mechanistic interpretability(MI)는 회로를 국소화한 뒤 각 구성요소의 의미를 설명하는 단계를 사람이 반복적으로 수행해 왔다. ACDC, EAP, EAP-IG처럼 개입/어트리뷰션 기반 방법은 localization을 자동화했지만, 설명(semantic role 및 상호작용 규명)은 여전히 표준화·확장이 어렵다. 또한 자동 해석 에이전트 연구는 개별 뉴런/특징 중심이거나, MI 결과를 검증하는 목적이어서 ‘회로 구성요소 단위 설명’을 정면으로 벤치마킹한 접근은 부족했다.

- **Core Contribution**: 이 논문은 localization이 이미 끝난 회로를 대상으로, LM agent가 구성요소 역할과 회로 수준 작업 설명을 생성하도록 돕는 문제를 다룬다. 이를 위해 84개 semi-synthetic transformer 회로(총 163개 컴포넌트)에 구성요소 태그(5-class)와 자연어 역할 메모를 붙인 벤치마크 AgenticInterpBench를 제안한다. 또 관찰→가설→인과 검증을 반복하는 에이전트 프레임워크 HyVE(Hypothesize, Validate, Explain)를 통해 component-level explanation과 circuit-level task description을 산출한다.

- **Technical Challenges**: 핵심 기술 난제는 ‘가설 생성’보다, 인과적으로 타당한 validation plan을 세우고 이를 코드 실행으로 실제 증거로 전환하는 신뢰성에 있다. 저자들은 observation 단계에서 텐서/어텐션 패턴을 수집하고, hypothesis 단계에서는 역할 분류 라벨을 유보한 채 관찰·예제로부터 서술을 만든 뒤, validation 단계에서 개입/패치 실험을 설계·execute_python으로 실행해 지지/반박을 판정하도록 구성했다. 그러나 실패는 validation 루프 후반에서 불완전한 계획, 코드 실행 오류(텐서 shape/API 사용 실수, 오프셋 처리, 툴 프로토콜 위반 등)로 더 자주 발생했으며, 이를 개선하기 위해 helper/scaffolding을 더 견고하게 할 필요가 제시된다.

- **Empirical Impact**: HyVE는 GPT-5.4, Claude-Sonnet-4.6, Gemini-3.1-Pro, Qwen-3-Coder 등 4개 백본에서 유의미한 컴포넌트·작업 수준 설명을 복구했지만, 백본 간 일관된 1위는 없었다. Claude-Sonnet-4.6은 component tag 정확도와 코드 실행 성공이 높았고, Gemini-3.1-Pro는 LLM-판정 기반 설명 품질과 task 정확에서 가장 강했다. 또한 Llama-3-8B의 실제 산술(three-operand addition) 회로 사례연구에서도 역할 복구가 관찰됐으나, 특히 Gemini는 logit-lens/위치 신호 같은 연관 증거를 인과로 과해석하는 오류 경향이 드러나 ‘신뢰성 있는 검증’의 중요성이 재확인됐다.



### Reinforcement Learning Towards Broadly and Persistently Beneficial Models (https://arxiv.org/abs/2606.24014)
Comments:
          Blog: this https URL

- **Prior Approaches**: 기존 연구는 misalignment(불일치)가 특정 도메인에서 학습되면 다른 도메인까지 넓게 전이될 수 있음을 보여줬다. 이 때문에 RL의 reward hacking, deception, 안전 훼손 같은 문제가 “좁은 학습 신호”에서 시작해도 전반적 실패 양상으로 확산될 수 있다는 우려가 커졌다. 한편 beneficial behavior(유익한 행동) 쪽으로의 일반화는 상대적으로 덜 검증돼 있었다.

- **Core Contribution**: 이 논문은 truthfulness, fairness, risk awareness, corrigibility 같은 beneficial traits(유익한 성향)를 목표로 하는 multi-domain 데이터셋을 구축하고, 그 위에서 alignment-focused RL을 수행해 전이 일반화를 측정한다. 또 학습 분포 밖의 50개+ 독립 벤치마크에서 alignment와 benefits를 함께 평가해, 단순 과적합이 아니라 “폭넓은 개선”인지 확인한다. 마지막으로 유해한 프롬프트 유도와 harmful finetuning 후에도 정렬이 유지되는 persistence를 체계적으로 다룬다.

- **Technical Challenges**: 핵심 난제는 (1) 특정 과업이 아니라 모델 차원의 성향이 alignment 전이의 원인이 되는지, (2) 단기적으로 유용해 보이는 행동이 다른 failure mode에서 실제로 막히는지, (3) 반대로 refusal 증가 같은 부작용이 성과를 왜곡하지는 않는지였다. 논문은 12개 도메인에서 15개 fine-grained beneficial traits를 합성 대화로 구현하고, trait description·domain description 조건부 생성 및 실패 회피 기준으로 평가/학습 신호를 설계했다. 이후 compute-matched baseline 대비, 5%만 유익한 성향 데이터로 끼워 넣는 ablation과 health-only/health·science 제외 같은 통제 실험으로 원인 분리를 시도했다.

- **Empirical Impact**: beneficial trait RL은 compute-matched baseline 대비 out-of-distribution alignment·safety·benefits 평가에서 80% 이상 벤치마크에서 성능을 개선했으며 평균 개선폭도 확인됐다. 특히 health 도메인에만 5%의 유익한 RL 데이터를 넣었는데도 non-health 평가에서 reward hacking, deception, general misalignment 등이 함께 개선되는 out-of-distribution alignment transfer를 관찰했다. 또한 harmful 의료 persona 프롬프트나 bad medical advice 유도 finetuning 상황에서 성능 저하가 더 작아져 persistence 향상 근거를 제시하되, 세부 원인 분리는 추가 연구가 필요하다고 밝혔다.



### Safe and Generalizable Hierarchical Multi-Agent RL via Constraint Manifold Contro (https://arxiv.org/abs/2606.24010)
Comments:
          10 pages

- **Prior Approaches**: 기존 multi-agent 안전강화 접근은 CMDP처럼 제약을 벌점/최적화에 섞거나, CBF를 safety filter로 붙이는 방식이 주류였다. 하지만 학습 기반은 이론적 하드 안전 보장이 약하고, 제어이론 기반은 QP 풀이나 보수적인 동작으로 인해 조정이 나쁘거나 효율이 떨어지기 쉽다. 또한 CBF 기반 계층 방법은 low level에서 매 타임스텝 QP를 풀어야 해 계산량 부담이 커진다.

- **Core Contribution**: 이 논문은 계층형 multi-agent reinforcement learning 프레임워크 HMM(Hierarchical Manifold Multi-Agent PPO)을 제안한다. low level에서는 constraint manifold를 이용해 행동을 tangent space로 투영함으로써 매 환경 타임스텝 하드 안전 제약을 보장하고, high level에서는 협업용 subgoal을 학습해 효율적인 조정을 가능하게 한다.

- **Technical Challenges**: 핵심 난제는 (1) multi-agent 환경에서 안전을 수학적으로 “구조적으로” 보장하면서 (2) 매 스텝 QP 없이 계산 효율을 확보하고 (3) 계층 학습이 안정적으로 수렴하도록 만드는 것이다. 저자들은 슬랙 변수를 포함한 constraint manifold로 안전을 동치인 등식 제약으로 바꾸고, augmented dynamics가 manifold에 머물도록 수축/드리프트 보상 항과 tangential action 항을 포함한 제어식을 통해 QP 없이 안전성을 유지한다. high level은 CTDE에서 중앙집중 가치함수와 graph 기반 정책으로 subgoal을 생성해, 학습 안정성과 에이전트 수/장애물 변화에 대한 일반화를 돕는다.

- **Empirical Impact**: 실험에서 HMM은 Lidar 기반 벤치마크에서 경쟁력 있는 성능을 보이면서도 거의 완벽에 가까운 safety rate를 유지한다. 특히 3개 에이전트·3개 장애물로 학습한 정책이 최대 21개 에이전트/장애물로 확장되는 상황에서도 안전성과 높은 과업 성공률을 함께 달성해, 안전-효율-스케일링의 균형을 실증적으로 보여준다.



### Critique of Agent Mod (https://arxiv.org/abs/2606.23991)
- **Prior Approaches**: 최근 LLM을 ‘coding agents’, ‘AI co-scientists’처럼 에이전트라 부르며 생산성을 높이려는 시도가 늘었지만, 많은 시스템은 미리 짜둔 툴·워크플로·프로그램 제어 루프를 외부에서 오케스트레이션하는 agentic 성격이 강하다. 그 결과 목표나 정체성, 의사결정의 모드 전환, 학습의 시작/중단 같은 핵심 구조가 모델 내부에서 내재화(endogenously)되기보다 파이프라인 설계에 의존하는 경향이 있다. 논문은 이러한 방식이 다양한 환경 적응과 진짜 자율성(agency)의 일부만 다룰 뿐, 생물학적 에이전트처럼 열린 세계에서 스스로 조직되는 역량을 설명하기 어렵다고 지적한다.

- **Core Contribution**: 논문은 ‘에이전트(agency)’를 단순 작업 수행 능력(operation excellence)과 분리해, 목표 지향 행동, 정체성의 진화, 자기조절, 자기성찰 및 학습이 내부에서 발생해야 한다고 주장한다. 이를 위해 agentic 시스템(외부 공학으로 워크플로를 조립)과 agentive 시스템(모델 내부에서 구조가 내생적으로 발생)을 경계짓고, 에이전트 아키텍처를 goal, identity, decision-making, self-regulation, learning의 5차원으로 분석한다. 나아가 범용 에이전트 모델을 위한 Goal-Identity-Configurator(GIC) 아키텍처를 제안하며, 계층적 목표 분해·진화하는 identity·월드 모델 기반 simulative reasoning·학습/추론 깊이를 조절하는 configurator·실제/시뮬 경험에서의 self-directed learning을 한 틀로 결합한다.

- **Technical Challenges**: 기여의 핵심은 ‘월드 모델과 에이전트 모델을 기능적으로 분리’하는 데 있으며, 둘을 합치면 next-state 예측과 reward/행동 선택이 뒤섞여 계획과 시뮬레이션의 신뢰성이 흔들릴 수 있다고 본다. 또한 장기 목표와 identity를 내부 잠재변수로 두고, 그에 조건화된 계획·실행·메타 의사결정이 어떻게 학습 가능한 형태로 구성될지 해결해야 했다. 논문은 simulative reasoning(System II)으로 world model에서 미래를 예측·계획하고, actor(System I)로 즉각 실행을 담당하며, configurator(System III)가 ‘언제 얼마나 깊게’ 숙고하고 ‘언제 학습/시뮬/업데이트’를 수행할지 내재적으로 결정하도록 설계해 self-regulation과 self-directed learning을 구현한다.

- **Empirical Impact**: 제안된 프레임워크는 무엇을 측정·감사(audit)·통제·안전 확보해야 하는지(agentive 성격이 강할수록 더 중요해지는 지점)를 더 명확히 해, 자율성이 커지는 시스템의 안전 논의를 구체화하려는 의미가 있다. 아키텍처 차원에서 agentic과 agentive의 차이를 5개 구조로 분해해 진단 가능성을 높이고, 향후 벤치마크/실험 설계에서 “워크플로 조립형”과 “내생적 agency형”을 구분해 비교할 근거를 제공한다. 즉, 단일 LLM 기능을 ‘에이전트화’하는 접근을 넘어, 열린 세계에서 지속 학습과 자기조절이 가능한 범용 agent model로 발전시키는 방향성을 제시한다.



### Neuro-Symbolic Drive: Rule-Grounded Faithful Reasoning for Driving VLAs (https://arxiv.org/abs/2606.23938)
- **Prior Approaches**: 기존 Driving VLA는 Chain-of-Thought(CoT)처럼 자연어로 설명을 생성할 수 있지만, 설명이 실제 행동을 좌우하는 인과적 단계와 맞물리지 않을 수 있습니다. 또한 많은 학습 라벨이 사람/다른 모델/후처리 파이프라인에서 만들어져 “생성된 말”과 “결정된 궤적” 사이에 supervision mismatch가 생긴다는 문제를 지적합니다.

- **Core Contribution**: 이 논문은 Neuro-Symbolic Drive를 제안하며, VLA의 CoT를 사후 정렬(post-hoc alignment)이 아니라 rule-based planner의 실행에서 직접 뽑은 reasoning trace로 감독합니다. rule 기반 플래너는 안전 제약 활성화→후보 탐색→최종 궤적 선택을 수행하는 실행 가능한 추론 엔진이므로, 그 내부 의사결정 상태를 구조화해 Qwen3.5-4B를 fine-tuning하는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 서로 다른 플래너의 내부 상태/어휘를 그대로 두면 의미가 일관되지 않고, raw trace에는 구현 디테일이 섞여 그대로 모방하면 재사용 가능한 운전 로직을 배우지 못한다는 점입니다. 이를 위해 planner 상태를 route·constraint·candidate·decision의 4-슬롯 스키마로 추상화해 rule-grounded reasoning으로 직렬화하고, 시뮬레이션에서 3종 플래너(IDM, IDM-MOBIL, PDM-Closed)의 실행 trace와 궤적을 한 쌍으로 수집한 뒤 시나리오 계열별로 가장 신뢰도 높은 teacher를 선택해 학습합니다.

- **Empirical Impact**: 시뮬레이터 기반 NSD-Sim 벤치마크에서 rule-grounded reasoning은 성능을 일관되게 개선했습니다. 3-camera 조건에서 ADE@3s는 0.47→0.26, miss rate는 8.30%→6.40%로 줄었고, 8-camera 조건에서도 ADE@3s 0.54→0.26, miss rate 10.13%→5.99%를 기록했습니다. 결론적으로 “더 긴 설명”이 아니라 planner 실행에 구조적으로 결합된 의사결정 의미가 궤적 생성 학습 신호로 작동함을 보여주며, Driving VLA의 디버깅/감사 가능성(행동-근거 동기화)에도 의미가 큽니다.



### RIFT-Bench: Dynamic Red-teaming For Agentic AI Systems (https://arxiv.org/abs/2606.23927)
Comments:
          Preprint

- **Prior Approaches**: 기존 LLM 중심 보안 평가는 prompt injection, jailbreak 등 ‘텍스트 입출력’에 주로 초점을 맞춰 에이전트의 tool use, memory, 상태 지속, inter-agent 통신에서 생기는 시스템 수준 취약점을 충분히 다루지 못했다. 또한 red-teaming 도구/벤치마크가 특정 프레임워크나 프로토콜, 시뮬레이션 환경에 강하게 결합돼 있어 이질적인 agentic architectures 간 비교와 공격 세트 재사용이 어렵다는 한계가 있었다. 일부 통합 벤치마크는 공통 인터페이스를 제공하지만, 각 시스템을 수동으로 벤치마크에 맞춰 바꿔야 하는 부담이 평가 재현성과 확장성을 떨어뜨린다.

- **Core Contribution**: 이 논문은 그래프 표현 기반의 동적 red-teaming 방법론 RIFT-Bench를 제안해, 서로 다른 agentic 시스템에서도 통일된 방식으로 취약점 평가를 수행할 수 있게 한다. 핵심은 NodeSpec(계층적 시스템 표현)으로 시스템 구조를 표준화하고, 그 표현을 바탕으로 Discovery(구조 추출)와 Scanning(적응형 공격 실행)을 자동화한 파이프라인을 제공하는 것이다. 추가로 완화(mitigation) 전략도 같은 공격 시나리오에서 함께 평가하도록 설계해 방어 측 검증까지 범위를 확장한다.

- **Technical Challenges**: 가장 큰 기술적 과제는 이질적인 구현(프레임워크/아키텍처/코딩 관행)에서 시스템 구조를 누락 없이 추출하고, 공격 템플릿을 시스템에 맞게 안전하게 주입·실행하는 ‘전이 가능한 표현’과 파이프라인을 만드는 것이다. 논문은 NodeSpec에 코드 레퍼런스·토폴로지·입출력/설정·capabilities를 포함한 implementation-grounded 계층 그래프를 정의하고, SI(Structure Identifier)에서 정적/동적 검증을 통해 NodeSpec을 완성한다. 또한 실제 도구 실행이 야기할 수 있는 비용과 부작용을 줄이기 위해 선택적 tool emulation을 지원하고, Scanning에서는 probe의 요구조건을 NodeSpec으로 결정적으로 필터링해 구조 불일치 공격을 줄인다.

- **Empirical Impact**: RIFT-Bench는 5개 도메인에 걸쳐 45개 agentic 시스템을 대상으로 105개 probe를 적용해 1만 건 이상 서로 다른 공격 테스트를 생성·평가하며, 이질적 아키텍처 전반에서 일반화 가능성을 보였다. 평가 정확도와 표현 정렬 지표(예: accuracy, code reference overlap, node alignment coverage, required keys F1)가 도메인별로 일관되게 높게 보고되며, tool input alignment는 사실상 1.0 수준으로 나타났다. 또한 defense를 포함한 실험에서 attack success/유용성/행동 일탈(Execution Drift) 같은 다중 지표로 방어-성능 트레이드오프를 측정할 수 있음을 보여 보안 평가의 확장 가능한 기반으로 자리매김한다.



### InSight: Self-Guided Skill Acquisition via Steerable VLAs (https://arxiv.org/abs/2606.24884)
Comments:
          Project website: this https URL

- **Prior Approaches**: VLA 모델은 언어와 비전을 바탕으로 조작을 학습하지만, 성능은 학습 데이터에 포함된 스킬 범위에 갇히는 한계가 있었다. STEER/Steerable Policies처럼 steerability를 제공하는 연구도 원시(primitives) 집합을 고정된 것으로 취급해, 누락된 동작을 새로 배우는 “continual skill acquisition”까지는 확장되지 않았다. 또한 SayCan, VoxPoser, Code-as-Policies 같은 VLM/LLM 기반 접근은 테스트 시 조합이나 계획에 강하지만, 새로운 primitive를 실제 정책 학습에 추가해 어휘를 늘리는 방식은 아니었다.

- **Core Contribution**: InSight는 VLA를 primitive 단위로 steerable하게 만들고, VLM이 새 태스크에 필요한 primitive “gap”을 찾아 자동으로 획득·재학습함으로써 기술 확장을 가능하게 한다. 핵심은 (1) 시연 영상을 VLM 분해와 엔드이펙터 포즈 기반으로 자동 segmentation해 labeled primitive로 변환하고, (2) VLM-guided data flywheel로 누락 primitive를 저수준 제어 파라미터까지 포함해 시도→성공 시 학습 데이터에 통합→VLA를 retraining하는 폐루프를 구축한 점이다. 그 결과 사람의 target-skill 시연 없이도 장기 과제를 새로운 조합으로 실행할 수 있다.

- **Technical Challenges**: 가장 큰 기술 과제는 (a) 수작업 없이 시연을 primitive 경계와 라벨로 정확히 쪼개 steerable 인터페이스를 만드는 것, (b) primitive gap이 무엇인지 판별한 뒤, 해당 gap에 맞는 단일축(translation/rotation axis) 기반 저수준 제어 파라미터를 VLM이 생성·일관되게 적용하는 것이었다. InSight는 VLM plan decomposition과 엔드이펙터 축 태그/포즈·운동량 정보를 결합해 경계를 자동으로 확정하고, 학습된 progress channel로 primitive 종료 신호를 제공한다. 또한 OOD 초기 상태나 종료 오차가 생길 때는 VLM completion check로 primitive 전이를 보정해 누락된 조작을 안정적으로 획득하도록 설계했다.

- **Empirical Impact**: InSight는 시뮬레이션과 하드웨어에서 block flipping, drawer closing, sweeping, twisting, pouring 등 5개 과제를 평가하며, 목표 스킬에 대한 추가 인간 시연 없이도 새로운 primitive를 얻고 이를 조합해 long-horizon 태스크를 수행했다. 예컨대 pour와 twist는 primitive 수준 신뢰도가 누적되며 end-to-end success가 Code-as-Policies 대비 크게 개선(실험에서 92~96% 수준)되었고, 더 긴 조합 과제에서도 InSight는 80%에 도달한 반면 테스트-시 조합만 하던 baseline은 급락했다. 또한 기존 pick-and-place 기반 능력은 새 primitive 추가 후에도 유지(100% 성공)되며, scooping 시연만으로 sweeping(비집기(non-prehensile) 접촉 동작) primitive까지 자율 획득하는 등 continual skill acquisition의 실용성을 입증했다.



### FLUX3D: High-Fidelity 3D Gaussian Generation with Diffusion-Aligned Sparse Representation (https://arxiv.org/abs/2606.24874)
- **Prior Approaches**: 기존 image-to-3D 파이프라인은 주로 representation learning과 latent diffusion을 결합하는 방식이다. 특히 sparse voxel 기반 방법은 계산 효율과 지오메트리 인식을 동시에 노리지만, 입력 이미지의 고주파 디테일을 보존하지 못해 텍스처가 흐려지거나 시각 패턴이 손실되는 문제가 반복된다.
또한 생성 단계에서는 dense 2D 토큰과 sparse 3D voxel 잠재공간의 대응관계를 충분히 정렬하지 못해 cross-modal correspondence 병목이 생기며, 그 결과 뷰가 달라질 때 질감이 어긋나는 현상이 나타난다.

- **Core Contribution**: 이 논문은 FLUX3D라는 image-to-3DGS 프레임워크를 제안해 두 병목—표현 병목과 cross-modal 정렬 병목—을 동시에 완화한다. 핵심은 표현학습 단계에서 DINOv2 같은 판별적 2D 피처 대신 FLUX의 생성형 diffusion 피처를 structured latent로 쓰는 DA-SLAT( Diffusion-Aligned Structured Latents )를 도입한 점이다.
생성 단계에서는 Sparse-structure Multimodal Diffusion Transformer (SMDiT)와 Modal-Aware Rotary Positional Embedding (MARoPE)로 2D 토큰이 sparse 3D 토폴로지에 맞게 학습되도록 하여, 2D-3D 정합성을 geometry-agnostic하게 끌어올린다.

- **Technical Challenges**: 가장 큰 기술적 과제는 (1) sparse voxel latent가 의미론 추상화에 치우친 2D 피처를 받아 고주파 재현에 불리해지는 문제와, (2) diffusion transformer가 표준 attention/positional encoding만으로는 sparse한 3D 구조와 dense 2D 토큰의 공간적 대응을 제대로 못 맞추는 문제를 동시에 해결하는 것이다.
저자들은 DA-SLAT에서 diffusion 피처의 정보밀도/재구성 적합성을 활용하고, decoder-only 구조로 불필요한 인코딩 압축을 줄여 3DGS 복원 충실도를 높인다. 더 나아가 SMDiT의 double-stream+single-stream 설계와 MARoPE의 virtual plane 기반 좌표 프레이밍으로, 카메라 캘리브레이션 없이도 2D-3D relative correspondence를 데이터로부터 학습하게 만든다.

- **Empirical Impact**: 실험에서는 3D-FUTURE, ABO, HSSD, Objaverse-XL로 학습하고 Toys4k로 평가하며, sparse voxel 레이아웃은 주어진 조건에서 appearance(텍스처) 품질을 집중 측정한다. 결과적으로 FLUX3D는 SSIM/PSNR/LPIPS 등 재구성 지표에서 기존 기준선과의 격차를 줄이고, image-to-3DGS 생성에서도 SSIM/PSNR/LPIPS뿐 아니라 CLIP Score, Fréchet Distance(FD), Kernel Distance(KD)까지 아우르는 다면적 성능 향상을 보인다.
정성 비교에서도 LGM류의 왜곡, DiffusionGS/TRELLIS의 질감 불일치 문제를 줄이며 입력 뷰의 색/디테일을 다양한 카테고리에서 더 안정적으로 보존해 SOTA 대비 실용적 의미가 크다는 평가를 받는다.



### It's Complicated: On the Design and Evaluation of AI-Powered AAC Interfaces (https://arxiv.org/abs/2606.24854)
Comments:
          Presented at Speech AI for All: The What, How, and Who of Measurement Workshop at the CHI Conference on Human Factors in Computing Systems, Barcelona, Spain, 2026

- **Prior Approaches**: 기존 연구는 AI를 AAC에 적용할 때 예측 모델의 속도·정확도 최적화 같은 단일한 기술 지표에 치우치는 경향이 있었다. 하지만 AAC 사용자에게 중요한 요구(정체성 표현, 대화 맥락, 관계적 상호작용, 상황별 변동)는 효율 중심 측정으로는 충분히 포착되기 어렵다. 또한 장애를 ‘표준 사용자’에 맞춘 치료/보정의 문제로 보게 되면, 사용자의 능력과 욕구가 고정·단일하다는 가정 아래 오히려 다른어긋남(misfit)을 만들 수 있다고 비판한다.

- **Core Contribution**: 이 논문은 AAC 인터페이스 평가에서 교차성 관점을 반영해야 한다고 주장하며, 이를 위해 여섯 가지 설계 고려사항(속도·정확도, 신체·정신적 노력, 정체성 프레젠테이션의 agency, 소통 맥락 적응, 턴테이킹, 변동하는 신체 능력)을 제안한다. 또한 AI로 기대 가능한 기능(문맥·대화 파트너 반영, big swing 예측, 노력 감지 기반 적응, 음성 프로소디 지원 등)을 “어떤 것을 개선할지” 관점에서 정리한다. 마지막으로 기술 성능 지표만이 아니라 인간중심 연구 방법을 결합한 다차원 평가 체계를 제안해, 사용자 욕구와의 정렬 여부를 더 견고하게 측정하려 한다.

- **Technical Challenges**: 기여를 실제로 구현·평가하려면 ‘의미는 맞지만 문구는 다른’ 출력, 사용자 의도(intent)의 주관성, 그리고 대화의 사회적 성공을 동시에 다뤄야 하는 문제가 있다. 논문은 의미·언어 유사도(offline, LLM judge/임베딩), 사용자 매개 평가(online에서 수용/수정, 보정 빈도), 과업 기반 기능 성공(task-based)처럼 서로 다른 축의 정확도를 함께 측정하는 방식을 제시한다. 여기에 더해 대체 접근수단(switch, eye gaze 등) 환경에서는 신체 노력(스위치 입력 수 등)과 정신적 작업부하(NASA-TLX류, EEG/시선추적 등)까지 정량화·동적 적응으로 연결하는 접근을 함께 제안한다.

- **Empirical Impact**: 해당 제안은 단일 벤치마크 성패가 아니라, 사용자 만족·참여도·사회적 존재감 같은 참여 중심 성과를 실증적으로 비교할 수 있게 설계된 평가 틀을 제공하는 데 의미가 있다. 특히 교차성에 따른 ‘기술 성능과 실제 사용 적합성의 불일치’를 드러내고, AI가 사용자에게 불필요한 부담이나 오해(예: 파트너가 AI 통제처럼 느끼는 상황)를 만들지 점검할 근거를 마련한다. 결과적으로 AAC 연구자와 개발자가 효율 지표에만 맞춘 기능 고정을 피하고, 사용자 맥락에 맞춘 closed-loop형 적응(노력·피로·선호 변화 감지 포함)까지 연구 범위를 확장하는 데 기여할 것으로 전망된다.



### IV-CoT: Implicit Visual Chain-of-Thought for Structure-Aware Text-to-Image Generation (https://arxiv.org/abs/2606.24849)
- **Prior Approaches**: 통합 multi-modal large language models(MLLMs)은 텍스트-이미지 생성 품질은 높지만, 객체 수·공간 관계·속성 결합·레이아웃 같은 구조 요구를 제대로 지키지 못하는 경우가 많다. 기존 접근은 프롬프트를 단일 conditioning stream에 압축해 구조와 외형(색·질감·조명 등)이 얽히며, 그 결과 속성이 다른 객체로 바뀌거나 누락, 배치 오류가 발생한다. CoT 기반도 명시적 언어 단계나 중간 시각 상태(마스크·레이아웃·초안 이미지)를 거치면 추론 단계가 늘거나 오류 누적 위험이 커진다.

- **Core Contribution**: 이 논문은 구조-aware prompt following을 위해 Implicit Visual Chain-of-Thought(IV-CoT)를 제안한다. 핵심은 구조 계획을 외부의 텍스트/중간 디코딩 없이, MLLM-DiT의 query 공간에서 structural-to-semantic “캐스케이드”로 내재화하는 것이다. structural query가 먼저 객체 경계·개수·레이아웃·거친 공간관계를 포함한 latent visual plan을 만들고, semantic query가 이를 기반으로 외형의 세부(정체성·색·재질·질감)를 렌더링한다.

- **Technical Challenges**: 가장 큰 도전은(1) 구조 정보를 외형과 분리해 모델 내부에 안정적으로 학습시키면서, (2) 추론 시에는 단일 forward pass로 처리해 효율을 유지하는 것이다. IV-CoT는 training-only sketch supervision을 도입해 structural query가 스케치의 윤곽·형상·개수·레이아웃을 학습하되 색/질감/조명 같은 appearance 요인을 억제하도록 유도한다. 이후 이미지 생성 학습에서는 structural loss를 regularizer로 유지해 구조 플래닝이 appearance 편향으로 drift되지 않게 하면서 semantic query는 세부 외형을 채우도록 최적화한다.

- **Empirical Impact**: GenEval과 T2I-CompBench에서 IV-CoT는 OpenUni-L-1024 대비 성능을 각각 0.86→0.88, 0.5448→0.5743으로 끌어올리며 구조에 민감한 항목(공간관계·형상·질감·색 등)에서 특히 개선이 두드러졌다. 또한 추론은 스케치 추출·중간 이미지 디코딩·테스트타임 탐색 없이 단일 forward pass로 동작해 latency가 명시적 CoT 대비 9~15배 낮다. 분석 결과 structural query는 스케치 도메인의 recoverable한 구조를 인코딩하고, query separation을 바탕으로 프롬프트 간 구조-외형 재조합도 어느 정도 제어 가능함을 보였다.



### Large-Language-Model Discovery of Quantum LDPC Codes through Structured Concept Evolution (https://arxiv.org/abs/2606.24808)
Comments:
          17 pages, 5 figures

- **Prior Approaches**: qLDPC는 희소 패리티 체크와 유한 인코딩 레이트, 거리 성장으로 오류 정정 스케일링을 노린다. 다만 CSS qLDPC 구성은 이산 설계 문제라서 사람이 조합적으로 탐색하기 어렵고, 기존 lifted-product/HGP 계열도 특정 군 구조에 종속되는 경향이 있다.

- **Core Contribution**: 논문은 LLM과 구조화된 대수 변이 문법을 결합한 structured concept evolution(SCE)로 CSS qLDPC의 lifted-product 코드 패밀리를 탐색한다. LLM이 처음부터 코드를 설계하기보다, 대수 명세와 이를 실행하는 프로그램(코드 생성기)을 ‘진화’시키며 군 알제브라·protograph 기하·기저 공간을 단계적으로 수정한다.

- **Technical Challenges**: 핵심 난제는 (1) CSS 유효성 조건(HX*HZ^T=0)을 깨지 않으면서 (2) 비가환(non-abelian) 군까지 포함하는 거대한 이산 탐색공간을 효율적으로 탐사하고 (3) LLM이 만든 명세가 실제로 generate()에서 정확히 구현되도록 일관성을 보장하는 것이다. SCE는 정규표현 기반 lifting으로 이산 명세를 이진 패리티 체크로 결정론 변환하고, 3단계 변이(로컬 시드/형상·스케일링/기저 공간)를 두어 탐색 리스크를 분산하며, 유효성·가중치·디코딩 평가를 생성된 체크행렬에서만 수행한다.

- **Empirical Impact**: GPT-5.4-mini와 GPT-5.4-nano 수준의 경량 모델로 SCE를 돌려 abelian부터 non-abelian에 이르는 다양한 경쟁 코드 패밀리를 발굴하고, 코드-capacity depolarizing noise에서 BP+OSD 디코딩(BP+OSD)으로 성능을 특성화한다. 또한 발견 코드는 다소 비단조적인 유한 크기 응답을 보이지만, 검색이 겨냥한 구간에서는 논리 실패율 억제가 나타났고rate–디코더 성능 tradeoff에서 dicyclic·abelian 패밀리가 서로 다른 영역을 점유함을 보였다.



### OrbitForge: Text-to-3D Scene Generation via Reconstruction-Anchored Video Synthesis (https://arxiv.org/abs/2606.24799)
Comments:
          40 pages, 33 figures, 19 tables

- **Prior Approaches**: 기존 3D 생성은 2D/비디오 확산 모델의 사전지식을 활용하지만, 일반적으로 프롬프트마다 SDS(score-distillation sampling) 최적화를 길게 수행하거나, 고정된 멀티뷰 포맷/카메라 격자를 전제로 학습된 멀티뷰 생성기를 쓰는 경우가 많다. 이로 인해 오버스무딩·과포화, Janus 아티팩트, 약한 장면 맥락, 혹은 카메라 제어 불가로 “렌더링은 그럴듯하지만 재사용 가능한 3D 자산”으로 이어지기 어렵다. 로컬 novel-view 보강 방법은 짧은 궤적 완성에는 강하지만, 생성 비디오의 불일치를 그대로 안고 닫힌 360도 오빗(orbit) 장면 생성과 검증을 동시에 만족시키는 데는 한계가 있다.

- **Core Contribution**: OrbitForge는 단 하나의 text-to-video 생성 결과를 입력으로 받아, 닫힌 closed-orbit 3D Gaussian Splatting 씬을 만들기 위한 “어댑터”를 제안한다. 핵심은 첫 복원에서 얻은 좌표계(스캐폴드)를 앵커로 삼고, 정해진 canonical orbit에서 부족한(unsupported) 뷰 구간만 비디오 priors로 보완한 뒤, 다시 한 번 동일 카메라 시스템으로 재구성해 오빗을 닫는 것이다. 특히 per-prompt SDS 최적화나 task-specific 멀티뷰 fine-tuning 없이, frozen 비디오 프라이어를 그대로 사용하면서 장면 단위의 3D 일관성을 끌어올린다는 점을 강조한다.

- **Technical Challenges**: 가장 큰 기술 난제는 text-to-video의 카메라 모션이 암묵적이고 부분 호(arc)만 커버하며, 시간에 따라 대상/배경이 흔들려 프레임 간 불일치가 3D 피팅 시 floaters·흐림·늘어난 표면·뷰 의존 아티팩트로 번진다는 것이다. OrbitForge는 Deformable Gaussian Splatting으로 1차 복원을 만든 뒤, 이를 MedianGS로 정적 구조 프록시로 압축해 canonical orbit 렌더링의 “지지/미지지 영역”을 판별한다. 이어서 endpoint-window 방식으로 미지원 구간만 완성하고, 완성된 orbit를 미리 지정된 canonical 카메라 인덱스에 고정해 2차 복원을 수행함으로써, 생성된 프레임이 유도하는 의사-동역학(pseudo-dynamics)이 좌표계 붕괴로 이어지지 않게 한다.

- **Empirical Impact**: 평가에서는 로컬 프레임 스무스만으로는 공정하지 않다고 보고, coverage-aware 지표인 measured angular span 등을 포함한 closed-orbit 프로토콜을 사용한다. 냉동 300-prompt T3Bench-derived audit에서 OrbitForge는 MedianGS-only 대비 unsupported 구간의 Q10 ImageReward를 8.07→16.36으로 크게 끌어올리면서, measured median span도 359.0-degree로 동일 수준을 유지한다. 또한 coverage까지 고려했을 때 VideoMV와 경쟁력 있는 품질-커버리지 균형을 보여, “닫힌 360도 오빗 3D 생성”이라는 목표에 맞춰 실증 효과를 입증한다.



### EG-VQA: Benchmarking Verifiable Video Question Answering with Grounded Temporal Evidenc (https://arxiv.org/abs/2606.24797)
- **Prior Approaches**: 기존 VideoQA 벤치마크는 주로 정답 일치 여부(정답 정확도)로 평가되며, 비디오 내 근거가 되는 시간 구간을 실제로 찾아냈는지는 충분히 검증하지 못했다. 또한 다지선다 포맷은 option bias를 유발해 언어적 지름길로 답을 맞힐 위험이 있고, 개방형 설정에서도 평가가 정답 생성과 근거(증거) 정합성을 분리해 보는 경향이 강하다.

- **Core Contribution**: 이 논문은 답을 맞추는 것뿐 아니라 시간적으로 국소화된 증거를 함께 제시하도록 요구하는 Evidence-Grounded Video Question Answering Benchmark(EG-VQA)를 제안한다. EG-VQA는 2,067개 비디오와 11,838개 QA 쌍에 대해 세밀한 temporal evidence를 텍스트 설명과 함께 주석하며, 서술·시간·인과·반사실(counterfactual) 추론 유형을 포괄한다.

- **Technical Challenges**: 핵심 과제는 ‘증거가 맞는지’를 정답 정확도와 결합해 일관된 방식으로 평가하고, 모델 학습에도 신호를 촘촘히 제공하는 것이다. 이를 위해 Evidence-Grounded F1(EG-F1)을 time alignment(IoU)와 semantic consistency를 동시에 묶어 최적 매칭(Hungarian algorithm)으로 계산하도록 설계했고, EG-Reasoner는 evidence 블록과 evidence-to-reasoning-to-answer 구조화된 출력에 대해 GRPO 기반 reinforcement learning과 soft matching 보상을 결합해 초기 학습의 희소성 문제를 완화했다.

- **Empirical Impact**: 실험 결과, 강한 Video-LLM일수록 정답 정확도는 높아도 evidence grounding은 낮게 나타나며, 정답 맞춤과 근거 충실성 사이의 근본적 불일치가 확인됐다. EG-Reasoner는 오픈소스 모델 중 최고 성능을 보이며 GPT-4o 대비 특히 counterfactual 질문에서 더 큰 향상을 보였고, evidence reward를 제거하거나 hard EG-F1만 쓰면 성능이 크게 떨어져 구조화된 증거 감독의 필요성이 실증됐다. 결론적으로 본 연구는 scaling만으로는 신뢰 가능한 비디오 이해가 어렵고, structured evidence supervision이 해석 가능하고 검증 가능한 VideoQA 시스템 개발에 필수임을 보여준다.



### Grad Detect: Gradient-Based Hallucination Detection in LLMs (https://arxiv.org/abs/2606.24790)
Comments:
          Accepted to the 2nd Workshop on Compositional Learning at ICML 2026, Seoul, South Korea. Copyright 2026 by the author(s)

- **Prior Approaches**: 기존 환각(hallucination) 탐지는 주로 next-token 분포의 confidence·entropy 같은 출력 레벨 신호에 임계값을 적용하거나, 여러 번 생성해 일관성을 보는 방식이 중심이었다. 다만 최신 LLM은 오버컨피던스처럼 잘못된 확률을 높게 주는 경우가 많고, 일관성 기반은 샘플링 비용이 커진다. 내부 표현을 probe(분류기 학습)하는 연구도 있으나, 특정 레이어의 상태 스냅샷을 보므로 “왜” 틀렸는지의 민감도 정보를 충분히 담기 어렵다.

- **Core Contribution**: 이 논문은 Grad Detect로, 추론 시 단 한 번의 forward-backward pass에서 계층(layer-wise) 그래디언트 패턴을 뽑아 환각 가능성을 예측한다. 출력의 표면 신호만으로는 드러나지 않는 “모델 파라미터 민감도”의 기하학이 정오/미응답(DNA) 같은 신뢰도 신호를 함께 인코딩한다는 점을 보여준다. 또한 그래디언트 레이어 분해를 통해 실패가 네트워크의 어디에서 집중되는지도 해석 가능하게 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 그래디언트를 이용해 거대한 차원을 효율적으로 요약하고, (2) 실제 생성과 무관하게 비교 가능한 기준을 만들어야 한다는 점이다. 이를 위해 카테고리별(정답/오답/미응답) 레퍼런스 그래디언트를 라벨 데이터 평균으로 만들고, 각 샘플의 레이어별 그래디언트와의 cosine similarity를 특징으로 압축한다. 최종적으로 이 L×|C| 특징 행렬을 lightweight transformer encoder에 넣어 분류하며, 대상 LLM은 fine-tuning 없이 그대로 둔다.

- **Empirical Impact**: 11개 instruction-tuned 모델(4개 아키텍처 계열)을 4개 Q&A 벤치마크에서 평가한 결과, Grad Detect는 confidence·sampling·activation probing 및 다중 생성 기반 방법을 일관되게 능가한다(정답성 Correct vs Incorrect에서 AUC/정확도 모두 개선). 특히 abstention 예측(Response task)은 94–99% 정확도를 보이며, 삼분류 Full task는 Correct/Incorrect 분리가 병목임을 확인해 추가 클래스가 성능을 크게 흔들지 않음을 보여준다. 레이어 소거 실험에서는 판별 그래디언트 신호의 97% 이상이 마지막 5개 레이어에 집중되어, 그만큼만 써도 성능 손실이 최소인 효율적 배치가 가능하다는 실증적 근거를 제공한다.



### Paying to Know: Micro-Transaction Markets for Verified Product Information in Agentic E-Commerc (https://arxiv.org/abs/2606.24783)
Comments:
          8 pages, 1 figure. Vision paper, under review

- **Prior Approaches**: 기존 상용 NLP는 쇼핑 챗봇을 추천(recommender)이나 전환(conversion) 도구로 보고, 사용자를 카탈로그 항목에 매칭한 뒤 구매를 설득하는 데 초점을 둔다. 이 프레이밍은 ‘어려운 문제=매칭’이라는 가정을 깔고 있으며, 검증 가능한 정보의 부족과 시장의 순위 인센티브 불일치를 상대적으로 사소하게 취급한다. 또한 제품 관련 데이터가 SEO 카피나 미검증 리뷰처럼 싼 신호로 요약되거나, 혹은 사일로에 갇혀 에이전트가 신뢰도 있게 탐색하기 어렵다는 한계가 남는다.

- **Core Contribution**: 논문은 agent-native micro-payment rails(예: x402, AP2)가 도입되면 희소성이 ‘상품 매칭’에서 ‘의사결정에 필요한 신뢰할 수 있는 정보’로 이동한다고 주장한다. 이에 에이전트가 마이크로 결제로 판매자·검증자·리뷰어가 공개하기로 한 증거를 단계적으로 잠금 해제하는, verified information 중심의 micro-transaction 정보 시장을 제안한다. 그 결과 랭킹 기반 스토어보다 더 진짜 품질과 정직한 경쟁을 유도할 수 있다고 본다.

- **Technical Challenges**: 핵심 난제는 (1) 예산 제약 하에서 다음에 어떤 ‘비용 있는 증거’를 살지 고르는 cost-optimal information acquisition, (2) 단일 데이터 가격과 증거 번들의 협상, (3) 서로 다른 출처의 정보를 동일 실체/속성으로 즉시 매칭하는 real-time entity resolution과 온톨로지 접지, (4) 가격된 주장 전부에 대한 환각 없는 grounded value exchange, (5) 사용자 페르소나를 개인정보 유출 없이 모델링하고 필요 시 주체가 통제 가능하게 유지하는 privacy-preserving persona modelling이다. 이들은 채팅 유창성보다 ‘대화→가격·접지·협상 가능한 증거 파이프라인’을 만드는 문제라고 못 박는다. 더 나아가 값비싼 툴 호출을 무턱대고 하는 대신, 불확실성이 정말로 남아 있을 때만 도구/질의를 쓰도록 비용-가치 캘리브레이션이 필요하다고 강조한다.

- **Empirical Impact**: 본 문서는 향후 비전(포지션 페이퍼)으로, 시장 효율성과 경쟁 개선을 실험으로 측정해 보여주진 않는다. 대신 산업 트랙이 ‘랭킹/대화 품질’ 대신 증거 획득의 복지(welfare)·효율성과 캘리브레이션을 측정하도록 벤치마크와 시뮬레이터를 확장해야 한다는 연구 방향을 제안한다. 의사결정 관련 신뢰 정보가 유료로 거래되는 구조가 마련되면, 에이전트가 구매를 조사·협상·정산까지 end-to-end로 자동화하는 에이전틱 조달로 이어질 수 있다는 점에서 의미가 크다.



### DeepBD: A Grounded Agentic Workflow for Variant Prioritization and Diagnosis of Genetic Birth Defects (https://arxiv.org/abs/2606.24779)
- **Prior Approaches**: 기존 변이 우선순위 도구(Phen-Gen, Exomiser, Xrare, LIRICAL 등)와 변이효과 예측(SIFT, PolyPhen-2, CADD, REVEL 등), 임상 데이터베이스(ClinVar, ClinGen, OMIM 등)는 각각 유용한 근거를 제공한다. 그러나 출생결함 맥락에서는 태아/영아의 불완전한 phenotype과 환자별 후보 변이(수천 개), 증거의 상충 가능성을 반영해 ‘어떤 변이를 먼저 볼지’ 순위를 환자 조건에 맞게 재가중하는 통합 단계가 병목이었다. 또한 LLM/에이전트 방식도 존재하지만, VCF 수준 후보 비교에서 안정적인 변이-단위 causal ranking을 위해서는 근거 기반의 고정된 계산 토대가 필요하다는 한계가 제기된다.

- **Core Contribution**: DeepBD는 유전성 출생결함의 post-sequencing 해석에서 변이 우선순위와 진단 해석을 동시에 돕는 ‘grounded agentic workflow’를 제안한다. 핵심은 evidence-allocation 원칙으로, (1) LLM-assisted case structuring, (2) 사전학습 evidence engine, (3) specialist evidence modules, (4) grounded diagnostic review layer를 역할 분리해 변이 점수 계산은 학습 가능한 엔진이 담당하고, 에이전트는 증거 정리·감사·검토에 집중하도록 설계했다. 이를 통해 불완전한 phenotype과 다양한 증거원을 환자 조건에 맞춰 순위화하고, 검토 가능한 진단형 합성으로 연결한다.

- **Technical Challenges**: 핵심 기술 난제는 환자별 불완전 phenotype 하에서 수많은 후보 변이를 ‘원인 변이’ 관점으로 안정적으로 비교하고, 서로 다른 증거 유형(규칙 기반, 서열/변이효과, 세포·경로 맥락)을 일관된 점수로 융합하는 것이다. DeepBD는 evidence engine이 rule evidence, variant-intrinsic evidence, phenotype-conditioned biological context를 각각 표현한 뒤 fusion하여 케이스 조건 점수를 학습하며, specialist modules는 필요한 후보에 대해 SHEPHERD/Exomiser 트랙이나 구조 기반(가능할 때) refinement 같은 외부 신호를 도구 호출 형태로 보강한다. 마지막으로 grounded diagnostic agent가 ranked evidence workspace를 기반으로 top-k 재정렬(제약 기반), reflection-style review, provenance audit, 진단 지향 합성을 제공한다.

- **Empirical Impact**: 18,622명의 사내 태아·영아 코호트에서 held-out solved-case 벤치마크를 평가한 결과, DeepBD는 Recall@1/3/5/10이 0.658/0.882/0.912/0.929로 Exomiser, DeepRare 및 LLM reranking 베이스라인보다 우수했다. 특히 LLM reranking은 전반적으로 Recall이 더 낮았고, DeepBD는 rank 1 이후에서 절대 이득이 커 ‘짧은 수동 검토 리스트’에 원인 변이를 더 자주 포함시키는 방향으로 성능이 나타났다. ablation/overlap 분석에서도 rule evidence, graph 기반 기전 맥락, 구조 기반 refinement가 서로 보완 신호를 제공함이 확인돼, 모듈 분리형 grounded agentic 설계의 효과가 실증적으로 뒷받침된다.



### Context-Aware Prediction of Student Quiz Performance with Multimodal Textbook Features (https://arxiv.org/abs/2606.24770)
Comments:
          4 pages, 2 figures, 2 tables

- **Prior Approaches**: 기존 학생 성취 예측은 주로 지식 추적(knowledge tracing)처럼 과거 수행 이력을 기반으로 향후 점수를 추정하는 데 초점이 있었다. 최근에는 문항/문제 정보(예: 항목 난이도, 질문 의미)까지 다루는 연구가 늘었지만, 학생 개인의 과거 점수와 동일 장(챕터) 콘텐츠 변동을 분리해 추가 신호를 검증하는 설계는 상대적으로 부족했다. 특히 멀티모달 접근도 있으나, 어떤 콘텐츠 요소가 일반화에 실제로 기여하는지는 실증이 제한적이었다.

- **Core Contribution**: 본 논문은 CourseKata의 챕터 리뷰 질문에서 뽑은 lightweight 콘텐츠 특징(텍스트/이미지)이, 학생의 과거 챕터 수행 평균만으로는 설명되지 않는 종말 챕터 퀴즈 점수를 예측하는지 질문한다. 2023년 CourseKata 응답 데이터에 챕터 수준 텍스트 특징(어휘 복잡도·희귀도 등)과 이미지 특징(에지/라인/영역 등)만 추가해 선형·정규화 모델로 비교한다. 결과적으로 “학생 기록 + 콘텐츠 맥락”이 단순 과거 성취만 쓰는 baseline보다 더 나은 예측 신호를 준다는 점을 체계적으로 보여준다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 문항의 의미·시각 복잡도를 과적합 없이 가볍게 표현하는 것과 (2) 예측 성능 개선이 ‘새 학생’ 또는 ‘새 챕터’로 얼마나 전이되는지 검증하는 것이다. 이를 위해 단순 회귀 파이프라인에서 단어 빈도 기반 텍스트 특징과 OpenCV 기반 거친 시각 특징을 만들고, 학생 단위로 묶은 5-fold 교차검증(RQ1)과 챕터를 통째로 배제하는 leave-chapter-out(RQ2) 검증을 분리했다. 또한 텍스트/이미지 특징 상관으로 인한 과적합을 ridge regression으로 완화하고, 분할 내부에서 정규화 강도를 교차검증해 안정성을 확보했다.

- **Empirical Impact**: 실험(4,742개 학생-챕터 관측, 562개 class-student ID)에서 Base 대비 텍스트+이미지 전체를 더한 모델은 학생 단위 held-out 예측에서 MSE를 상대 9.1% 줄이며 유의미한 개선을 보였다. 반면 leave-chapter-out에서는 텍스트 특징이 baseline 대비 오류를 줄였지만, 현재 이미지 특징을 포함한 모델은 오히려 오류가 커졌다. 이는 챕터 수준에서 텍스트 맥락은 커리큘럼 간 전이가 가능할 가능성을 시사하는 동시에, 이미지 특징은 현재 표현이 의미적 정보를 충분히 담지 못해 더 정교한 비전 표현(예: vision-language 임베딩) 연구 여지가 있음을 의미한다.



### UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving (https://arxiv.org/abs/2606.24759)
- **Prior Approaches**: 기존 multimodal large language model(MLLM) 기반 자율주행 장면 이해는 성과가 있지만, 시간 추론과 공간 정밀도 사이에 근본적인 트레이드오프가 남아 있다. 단일 프레임/저해상도 입력 중심 방식은 작은·먼·부분 가림 위험을 놓치기 쉽고, 언어 중심 접근은 설명의 근거가 충분히 grounded 되지 않는 문제가 있다.

- **Core Contribution**: 이 논문은 UniDrive로, 해석 가능한 위험(risk) 이해를 위해 visual-language와 grounding을 하나의 프레임워크에서 통합한다. 멀티프레임으로 장면 동역학을 잡는 temporal reasoning branch와 최신 프레임의 고해상도 세부를 보존하는 perception branch를 결합하고, gated cross-attention으로 두 표현을 정렬해 위험 객체에 대한 자연어 설명과 바운딩 박스를 함께 생성한다.

- **Technical Challenges**: 핵심 난제는 동적 맥락(시간 의미)을 정밀한 공간 근거(위치/가시성)와 동시에 일관되게 정렬하는 것이다. UniDrive는 두 가지 표현을 gated cross-attention fusion으로 결합해 상황 전개 정보와 고해상도 증거를 맞물리게 만들고, fused representation을 기반으로 언어 생성과 grounded bounding-box 출력을 공동으로 수행한다.

- **Empirical Impact**: DRAMA-Reasoning benchmark에서 UniDrive는 이미지 기반·비디오 기반 대표 baseline 대비 captioning과 위험 객체 grounding 모두에서 향상됐다. 특히 small-object localization, NuScenes·BDD100K로의 zero-shot 일반화, 사람 평가 기반 interpretability/trustworthiness에서 강점이 확인되며, 시간 의미와 고해상도 지각의 명시적 결합이 안전 지향 자율주행 시스템의 기반이 될 수 있음을 시사한다.



### Beyond U-Net: A Latent-Representation-Aligned Skip-Free Backbone for Flow-Matching Speech Enhancemen (https://arxiv.org/abs/2606.24745)
- **Prior Approaches**: 최근 speech enhancement에서 diffusion/score-based 생성모델은 성능이 좋지만, 역시간 샘플링 과정에서 DNN을 여러 번 평가해야 해서 NFE가 커져 실시간 적용이 어렵다는 한계가 있었다. 이를 줄이기 위해 Flow Matching(FM) 같은 ODE 기반 접근이 도입됐고, 몇 단계의 few-step inference로도 경쟁력 있는 품질을 보였지만, 학습 안정성과 최종 품질은 백본 설계에 크게 좌우됐다. 특히 U-Net의 skip connection은 음향 디테일을 살리는 대신 잡음과 상관된 저수준 특징까지 함께 전달할 수 있어 디코더의 부담이 커질 수 있다는 문제도 지적됐다.

- **Core Contribution**: 이 논문은 flow-matching 기반 SE에서 U-Net skip connection을 제거한 skip-free encoder-decoder 백본을 제안하고, 이를 Latent Representation Alignment(LRA)로 학습을 유도한다. 핵심은 Descript Audio Codec(DAC)을 quantization 없이 연속 오디오 오토인코더처럼 사용해, frozen codec에서 뽑은 clean latent 특징으로 병목(bottleneck)과 디코더 표현을 정렬하는 supervision을 제공한다. 그 결과 모델은 잡음이 섞인 저수준 shortcut에 의존하기보다 압축된 clean-speech 표현을 학습하도록 유도된다.

- **Technical Challenges**: skip-free로 구조적 지름길을 없애면 정보 병목이 생겨 학습이 흔들릴 위험이 있는데, 이를 LRA로 완화한다. 구체적으로 DAC의 연속 잠재표현을 “교사(teacher)”로 두고, 타임스텝 조건이 들어간 투영을 통해 병목 표현과 decoder 특징을 clean latent에 맞추는 정렬 손실을 설계했다. 또한 ODE 적분을 위한 velocity/trajectory 학습은 clean waveform 예측 파라미터화로 구성하고, 학습의 안정성을 위해 무가중 clean-speech 재구성 손실과 adversarial(및 feature matching) 손실을 함께 활용했다.

- **Empirical Impact**: 실험에서 제안한 FlowSE(LRA, proposed)는 WSJ0-CHiME3와 VoiceBank-DEMAND에서 지각 품질 지표가 개선되며, 특히 VoiceBank-DEMAND에서 PESQ와 DNSMOS, WVMOS 등 여러 지표에서 우수한 결과를 보였다. 또한 다섯 번 함수평가(five function evaluations)만으로도 향상된 품질을 달성해 few-step inference의 실용성을 뒷받침한다. 학습 동역학 관점에서도 LRA 백본이 U-Net 대비 더 적은 epoch로 높은 지각 품질에 도달해, codec 기반 representation-level 가이드가 효과적임을 보여줬다.



### Task Decomposition for Efficient Annotation (https://arxiv.org/abs/2606.24734)
- **Prior Approaches**: 기존에는 한 명의 주석자가 예시 전체를 end-to-end로 주석하는 방식이 일반적이었지만, 구조화된 표현 주석은 복잡해서 인지·추론 부담이 커집니다. 모델 기반 주석은 생성 비용을 줄일 수는 있으나, 검증 비용이 들고 다운스트림에 쓸 만큼 품질을 확보하려면 추가 감독이 필요하다는 한계가 있습니다. 또한 최근의 혼합 인력(모델+인간) 환경에서는 어떤 주석 하위 과제를 어떤 역량의 주석자에게 배분해야 하는지 불명확합니다.

- **Core Contribution**: 이 논문은 구조화된 주석 작업을 여러 하위 태스크로 분해해, 전체 주석 프로젝트의 누적 inferential load를 줄이는 설계를 제안합니다. centering theory의 center 개념에서 영감을 받아, ‘유효한 주석 공간에서의 자유도’로 inferential load를 정식화하고, center(하위 태스크로 드러나는 핵심 앵커 엔티티) 식별이 출력 공간 복잡도를 제약함을 보입니다. 이를 바탕으로 center 식별을 분리·전진시키는 분해가 누적 부담을 낮춘다는 가이드라인을 제공합니다.

- **Technical Challenges**: 핵심 기술적 어려움은 복잡한 구조 주석에서 어떤 하위 태스크 조합이 center 식별을 강화해 출력 공간을 얼마나 줄이는지 정량화하는 것입니다. 논문은 유효 주석의 공간에서 자유도를 기준으로 inferential load 모델을 만들고, center 식별이 완료되면 남은 선택 공간이 축소된다는 관찰을 이 모델에 연결합니다. 또한 고정 예산 하에서 주석자(모델/인간)에게 하위 태스크를 배분해 품질을 최대화하는 할당 절차를 제시합니다.

- **Empirical Impact**: 이 연구는 이전 work의 사례들을 통해 복잡한 구조 주석을 분해했을 때 비용 효율이 개선됨을 보여줍니다. 더 나아가 제안된 할당 절차로 제한된 예산 안에서 품질을 극대화하는 방향의 운영 전략을 제시해, 혼합 주석 환경에서의 실용성을 강화합니다. 구조 주석의 ‘무엇을 먼저 정해 출력 공간을 줄일지’에 대한 체계적 접근이 가능해지면서, 대규모 코퍼스 주석 파이프라인의 설계 관점을 확장하는 데 의미가 있습니다.



### Evaluating the Interpretability of Sparse Autoencoders with Concept Annotations (https://arxiv.org/abs/2606.24716)
Comments:
          Accepted at ECCV 2026

- **Prior Approaches**: 기존 sparse autoencoders(SAE) 평가는 주로 sparsity나 reconstruction 같은 구조적 지표, 또는 정성적 사례 중심에 머물렀다. 기능적 평가로서 개입(intervention)을 시도해도, 비전에서는 속성 변화가 ‘정확히 하나만’ 분리된 반사실 데이터 구성이 어려워 의미 대응(semantic correspondence)을 정량화하기가 힘들었다.

- **Core Contribution**: 이 논문은 SAE latent와 사람의 주석된 개념/속성 간 정렬(alignment)을 사람 기준에 근거해 수치화하는 평가 프레임워크를 제안한다. 사용자 연구 없이도(별도 설문 없이도) latent-개념 매칭과, 매칭된 개념이 개입된 이미지 변화에 선택적으로 반응하는지 확인하는 TAPAScore를 통해 의미적 대응을 검증한다.

- **Technical Challenges**: 핵심 난제는 비전 데이터에서 ‘한 속성만 바꾼’ 통제된 쌍을 만들 수 없다는 점이며, 이를 위해 synCUB와 synCOCO를 합성 데이터로 구축해 정확히 하나의 속성/객체만 달라지게 했다. 또한 feature splitting 같은 실패를 반영해, Fully-Binary Matching Pursuit(FBMP)로 latent를 one-to-one이 아니라 many-to-one으로 조합 매칭하고, 논리 연산 기반의 residual 업데이트로 binary 도메인에서 매칭 탐색을 수행한다.

- **Empirical Impact**: 실험에서 MATCHScore의 기존 one-to-one 기준과 여러 대리 지표(FMS, MS, CKNNA)는 sanity check에서 trained SAE를 untrained/random과 안정적으로 구분하지 못했지만, FBMP 계열 매칭과 TAPAScore는 강하게 분리했다. CLIP과 DINOv2 임베딩에서 overcompleteness(사전 크기)가 커질수록 perturbation alignment가 떨어져 해석 가능성이 감소할 수 있으며, dictionary size는 적당한 수준이 가장 좋은 trade-off를 보였다.



### TACTFUL: Tactile-Driven Exploration For Object Localization and Identification in Confined Environments (https://arxiv.org/abs/2606.24712)
Comments:
          IROS 2026

- **Prior Approaches**: 기존 연구는 시각이 불완전할 때 촉각으로 localization을 돕거나(예: tactile SLAM) 접촉 신호를 기반으로 탐색을 수행했지만, 대부분 저차원 접촉 정보에 머물러 정밀한 3D 재구성까지 일관되게 이어지기 어려웠습니다. 또 촉각으로 형상을 복원하는 방법은 많았지만, 대개 ‘물체 위치를 이미 안다’는 가정 하에서 표면 탐색에 집중해 비전-무응답 상태의 자율 탐색과 결합이 약했습니다. 일부 촉각 전용 탐색은 시뮬레이션 또는 휴리스틱 기반으로 진행돼, 실제 고해상도 촉각 데이터에서의 성능 증거가 제한적이었습니다.

- **Core Contribution**: TACTFUL은 비전 없이(vision-free) 다지(多指) 로봇이 한 에피소드 내에서 (1) 한정된 작업공간을 자율 탐색해 물체를 찾고, (2) 접촉으로부터 촉각 포인트를 모아 3D 형상을 재구성하며, (3) 재구성된 형상으로 물체를 식별하는 일련의 프레임워크를 제안합니다. 핵심은 단일 RL 정책이 글로벌 탐색과 로컬 표면 정제를 동시에 수행하도록, ‘dynamic reward schedule’로 행동을 단계적으로 전환시키는 점입니다. 또한 시뮬레이션 없이 실제 하드웨어 데이터로만 학습한 정책을 사용해 실세계 구동성을 강조합니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 비전-무응답 환경에서 무작위 탐색이 비효율적이고 안전 문제를 유발할 수 있는데, 동시에 장기 탐색과 형상 재구성을 동시에 최적화해야 한다는 점입니다. 저비용 실전 학습을 위해 먼저 Behavior Cloning(BC)으로 텔레오퍼레이션 시연 5311건에서 안전한 접촉/탐색 행동의 초기화(사전 지식)를 만들고, 이후 PPO로 ‘접촉 보상, 탐색 보상, 재구성 보상’을 동적으로 가중해 하나의 정책을 학습합니다. 재구성은 희소 촉각 포인트를 shape completion(포인트 확산 기반 모델)으로 조밀하게 복원해, Chamfer distance 기반으로 식별 정확도를 높이는 방식으로 해결합니다.

- **Empirical Impact**: 실험에서는 실제 3개 물체(큐브, 변형 실린더, 변형 컵)를 대상으로 물체 라이브러리에서 타깃을 지정하고 식별 성공률을 평가했으며, TACTFUL은 평균 Chamfer-L2L2 0.015m에서 성공률 77%를 달성했습니다. 휴리스틱 탐색 및 BC 전용, RL만 학습한 대안보다 재구성 정확도와 식별 일관성이 모두 높았고, 특히 세 가지 보상 항을 모두 넣을 때만 성능이 안정적으로 유지됐습니다. 또한 shape completion 모델을 제거하면 Chamfer가 낮아도 식별이 흔들릴 수 있음을 보여, 촉각 기반 ‘명확한 물체별 형상’ prior가 실세계에서 중요하다는 함의를 제공합니다.



### FlowPipe: LLM-Enhanced Conditional Generative Flow Networks for Data Preparation Pipeline Construction (https://arxiv.org/abs/2606.24679)
Comments:
          Accepted by SIGMOD 2027

- **Prior Approaches**: 자동 데이터 준비 파이프라인 자동화는 이질적 연산자들의 조합적 탐색과 논리적 의존성 때문에 어렵다. 기존 AutoML/차분·미분 기반 방법은 연산자 제약이나 end-to-end 평가 비용, 혹은 상태 의존 전이의 세밀한 모델링 부족으로 한계를 보였고, RL 계열은 delayed reward에서의 credit assignment가 불안정했다. 또한 SOTA Multi-DQN은 가치 추정이 분리되며 장기 의존성에 약하고, 데이터 컨텍스트를 출력층의 보정처럼 넣어 의미가 정책 내부 추론으로 충분히 들어가지 못하며, 유효 상태가 많은 희소 탐색 공간에서 효율이 떨어진다.

- **Core Contribution**: FlowPipe는 파이프라인 합성을 conditional probabilistic flow generation으로 정식화해, 전체 궤적의 확률이 최종 검증 성능 보상에 비례하도록 학습한다. 기존 Multi-DQN의 분절형 가치 추정을 피하고 Trajectory Balance 목적을 통해 terminal reward 신호가 초반 결정까지 더 직접적으로 전달되도록 했다. 더불어 LLM에서 유도한 논리적 priors를 FiLM 기반 Deep Semantic Modulation으로 정책의 내부 활성에 구조적으로 조건화해, 데이터 의미에 맞춘 연산자 선택을 가능하게 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 조합적·장기 의존 의사결정에서의 정확한 credit assignment, (2) LLM 컨텍스트가 정책 내부 추론을 실제로 바꾸지 못하는 semantic detachment, (3) 많은 invalid state를 포함한 희소 탐색에서 탐색 효율을 유지하는 것이다. FlowPipe는 C-GFlowNets에 Trajectory Balance를 결합해 궤적 일관성(Flow Consistency) 경로로 장기 신용을 보강하고, failure awareness를 목표함수에 반영해 유효하지 않은 파이프라인 상태를 조기에 가지치기한다. 동시에 Structured Prompt로 요약한 LLM 의미 임베딩을 FiLM으로 중간 표현에 주입해, 스케일링 같은 의미 불일치 연산이 내부적으로 억제되고 적절한 인코딩 선택이 강화되도록 설계했다.

- **Empirical Impact**: 74개의 실세계 데이터셋을 대상으로 한 실험에서 FlowPipe는 두 벤치마크에서 SOTA 대비 평균 정확도 11.96% 향상과 12.5배 빠른 학습 수렴을 보였다. 또한 exhaustive search(데이터셋당 30시간급) 수준의 품질에 근접하는 성능을 수 초 내에 달성해, end-to-end 평가 비용이 큰 설정에서의 실용성이 강조된다. 결과적으로 데이터 준비 자동화에서 GFlowNets 기반 reward sampling이 long-horizon 문제와 의미 조건부 탐색을 함께 개선할 수 있음을 경험적으로 입증했다.



### AI-PAVE-Br: Leveraging Large Language Models for Enhanced Product Attribute Value Extraction through a Golden Set Approach (https://arxiv.org/abs/2606.24655)
- **Prior Approaches**: 기존 PAVE(Product Attribute Value Extraction)는 규칙 기반·정규식·사전 중심 접근이나 CRF/SVM 같은 전통 NER에 의존하는 경우가 많았다. 이런 방법은 데이터와 언어 변형에 취약하고, 도메인별 특징 공학과 라벨 데이터가 많이 필요해 규모가 커질수록 유지보수가 어려웠다. 또한 최근 LLM 활용 연구도 영어 중심이어서 포르투갈어(브라질) 카탈로그의 용어·표기 관습을 그대로 반영하기 어렵다는 한계가 있었다.

- **Core Contribution**: 본 논문은 브라질 e-commerce에 특화된 PAVE 시스템 AI-PAVE-Br을 제안한다. LLM을 대상으로 카탈로그 추출 목표에 맞춘 prompt engineering을 적용해, 상품 제목/설명/기술 데이터에서 Entity-Category-Subcategory를 구조화된 출력으로 뽑아내도록 설계했다.
아울러 재현 가능한 평가를 위해 포르투갈어 PAVE의 Golden Set(수작업 라벨, 20개 상품 유형, Entity·Category·Subcategories 구조)를 공개한다.

- **Technical Challenges**: 핵심 난제는 포르투갈어 상품 텍스트의 언어적 뉘앙스 다양성과 속성-값의 관계 추론, 그리고 값 표기의 비정규화 문제였다. 같은 모델명이 여러 표기로 등장하고(예: WD11 계열), 치수·전압·색상/용량 등이 의미는 같아도 문장 표현이 달라 정규화가 어렵다.
논문은 fine-tuning 대신 task-specific 프롬프트로 LLM이 값뿐 아니라 해당 속성(예: screen size-10-inch)을 함께 매핑하도록 유도했으며, 출력 형식을 JSON 스키마로 강제해 일관성 있는 파싱 가능성을 높였다.

- **Empirical Impact**: Golden Set을 기준으로 AI-PAVE-Br은 전통 NER 기반 baseline 대비 평균 F1-score를 59.79에서 74.68로 끌어올렸다. 대부분의 상품 유형에서 큰 개선을 보였고, 일부 카테고리에서는 입력 텍스트의 구조적 변동성과 속성 표현 복잡도 때문에 성능 하락도 관찰됐다.
또한 coverage(빈 응답 없이 예측 생성)는 46.71%에서 71.96%로 상승해 카탈로그 전반에서 더 안정적으로 추출을 수행함을 보였다. 브라질 비영어 시장에 대한 고정밀 PAVE 실무 해법과 공개 벤치마크를 함께 제공한다는 점에서, 향후 연구의 기준선과 데이터 인프라 역할을 할 것으로 기대된다.



### Visualizing "We the People": Bridging the Perception Gap through Pluralistic Data Storytelling (https://arxiv.org/abs/2606.24635)
- **Prior Approaches**: 전통적인 시각 데이터 스토리텔링은 흔히 두 집단을 이분법적으로 대비시키는 이분 그래픽에 의존해 왔습니다. 그 결과 집단 내부의 이견을 단순화하거나 모호성과 공유 가치가 사라져 “우리 대 그들” 같은 편향을 강화할 수 있다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 AI로 뒷받침되는 디지털 플랫폼에서 다원적(pluralistic) 설계 선택을 통해 분열적인 대비 프레임을 줄이고, 의견 분포와 집단 간 공통분을 더 잘 드러내는 방법을 제안합니다. 특히 고차원 의견 공간을 매핑하고 합의(consensus)와 불일치(dissensus) 영역을 함께 시각화하는 ‘의견 지형(opinion landscape)’ 접근을 핵심으로 제시합니다.

- **Technical Challenges**: 기존 대비 그래픽을 ‘분포 기반·대화형’으로 전환하려면, 참여자들의 장문 텍스트 의견을 고차원 의견 공간으로 변환해 일관된 지형으로 합성하는 기술이 필요합니다. 논문은 Jigsaw와 Napolitan Institute의 2025년 9월 We the People deliberation처럼 AI가 장문 입력을 통합해 상호작용 가능한 의견 지형을 구성함으로써, 서로 다른 관점이 동시에 인간적으로 해석되도록 하는 흐름을 제시합니다.

- **Empirical Impact**: 이 방식은 435개 의회 선거구 전반의 2,400명 이상 미국인을 대상으로 진행된 비동기 대화에서 실증됐으며, ‘숨겨진 수준의 광범위한 합의’가 드러났다고 보고합니다. 대비 중심 시각 프레임에서 분포 중심의 대화형 모델로 전환하는 것은 대규모 확장 가능하고 비용이 낮은 개입으로서, 지각 격차를 줄이고 더 회복력 있는 협력적 민주 문화를 키우는 데 의미가 있다고 결론짓습니다.



### Privacy-Preserving RAG via Multi-Agent Semantic Rewriting: Achieving Confidentiality Without Compromising Contextual Fidelity (https://arxiv.org/abs/2606.24623)
Comments:
          This full manuscript contains 23 pages and has been formally accepted for publication in Information Processing & Management (Elsevier IPM). Tao Fang is the corresponding author

- **Prior Approaches**: RAG는 외부 지식을 검색해 LLM의 정확도를 높이지만, 민감한 코퍼스에 적용될 경우 prompt injection 등 공격을 통해 검색 원문이 노출될 위험이 큽니다. 기존 방어는 (1) 데이터 소스에서 합성 텍스트로 대체하거나 (2) DP처럼 잡음을 주입하거나 (3) 단일/루프 기반 재작성으로 프라이버시와 유틸리티의 균형을 맞추려는 방식이었습니다. 특히 합성 기반은 의미의 구체성이 사라져 다운스트림 신뢰성이 흔들리고, 단일 에이전트 방식은 한 번에 두 목표를 동시에 처리하다 보니 과잉·과소 마스킹이 발생하기 쉽습니다.

- **Core Contribution**: 이 논문은 RAG 파이프라인의 검색-생성 사이에 끼워 넣는 비공개 데이터용 “semantic rewriting” 프레임워크를 제안합니다. Pri-Extra(프라이버시 민감 구간 추출)–Sem-Extra(유틸리티를 위한 구조화된 의미 백본 추출)–Reconstruction(구간 제거 후 의미 재구성) 세 에이전트가 협업해 식별자(명시적/잠재적)를 제거하면서도 핵심 의미를 보존합니다. 또한 Asymmetric Retrieval과 isolated generation을 결합해 LLM이 원문 프라이빗 식별자를 직접 보지 못하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 텍스트에서 프라이버시와 유틸리티가 분리되지 않고 함께 얽혀 있어 단일 패스로는 삭제-정확도 간 제로섬 균형을 맞추기 어렵다는 점입니다. 이를 위해 Pri-Extra는 rules 기반 NER/정규식으로 명시적 식별자를 먼저 잡고, LLM 기반 추출로 latent quasi-identifiers를 추가로 찾아 누락을 줄입니다. Sem-Extra는 문장 표면 대신 subject–predicate–value–중요도 형태의 슬롯 튜플로 분해해 스타일/표면 단서가 공격에 재사용될 가능성을 낮추고, Reconstruction은 fine-grained conflict routing으로 low-depth면 placeholder 치환, deep semantic conflict면 고수준 추상화로 과잉 마스킹을 제어합니다.

- **Empirical Impact**: ChatDoctor와 Wiki-PII(Enron PII를 Wikitext-103에 혼합)에서 6개 LLM 백본, targeted/untargeted 공격을 모두 평가했으며 프라이버시 누출이 유의미하게 감소했습니다. 예로 LLaMA-3-8B에서 targeted 정보 노출 인스턴스를 baseline의 144에서 1로 크게 줄였고, 문맥 충실도는 BLEU-1 0.122로 기존 SAGE(0.117)를 앞섰습니다. 또한 재작성은 온라인 추론 지연을 만들지 않는 asynchronous offline 전처리 모듈로 동작하며, 공격 상황에서도 semantic fidelity와 보안성을 동시에 유지하는 점에서 의미가 큽니다.



### Infinitesimal Causality (https://arxiv.org/abs/2606.24621)
Comments:
          17 pages

- **Prior Approaches**: 기존 범주론적 인과추론 연구는 Markov category/스트링 다이어그램을 이용해 do-calculus의 문법적 변환(와이어 절단, 조건화, 반사실 조합)을 주로 ‘이산적 등식’ 수준에서 다뤘다. 하지만 이러한 관점은 개입이 통계적 매니폴드를 어떻게 미분적으로 변형하는지, 그리고 그 변형이 Lie bracket 잔차로 어떤 기하학적 시그니처를 남기는지까지는 설명하지 못한다. 그래서 잠재 교란(latent confounding)이 그래프/표현의 선택과 무관하게 보이는 기하학적 실패(비가시적인 닫힘)를 다루기엔 취약했다.

- **Core Contribution**: 이 논문은 Frobenius Markov categories에 tangent-bundle semantics를 얹어 ‘infinitesimal causality(미소 인과성)’를 범주 내부에서 정의한다. 핵심은 개입(intervention)이 Frobenius copy/discard 구조의 접선 변형으로 해석되고, 이 변형이 클래식 정보 흐름 구조를 보존하는지 여부가 Lie bracket 닫힘(involutive closure)으로 판정된다는 점이다. 또한 structural causal models에서는 가시적 stochastic kernel이 pushforward된 뒤에만 나타나므로, 미소 개입은 외생변수(exogenous variables) 위의 결정적 메커니즘의 slice에서 가장 자연스럽게 정식화된다고 제안한다.

- **Technical Challenges**: 난점은 ‘개입’의 미분학적 대상이 가시 관측분포에서 바로 정의되지 않을 때(지원(support) 병리나 전역 density ratio 부재)도 calculus를 성립시키는 것이다. 저자들은 전역 hard intervention 대신, 외생 법칙의 부드러운(soft) 미소 변형 벡터필드로부터 tangent pushforward를 먼저 취하고, 그 결과를 선택된 가시 통계모형의 접공간으로 투영하는 방식으로 해결한다. 그리고 Frobenius-derivative defect를 기본 장애물로 두고, do-calculus의 세 규칙을 counit/coproduct/Kan transport 및 bracket closure의 ‘등식’ 형태로 대응시키며, bracket가 가시 스팬 밖으로 새면 residual 항이 나타난다고 정리한다.

- **Empirical Impact**: 논문에서 제공된 내용은 주로 새로운 범주적 프레임워크와 정의(특히 Frobenius/tangent 닫힘 조건, Frobenius-derivative defect, structural slice에서의 미소 개입 정식화)에 집중되어 있다. 다만 기존의 Lie-bracket causal discovery(예: BRIDGE/SKFM)에서 관찰되던 ‘가시 개입 방향의 내부 닫힘 실패’가 이론적으로 어떤 범주적 불변량(미소 수준의 copy/discard 보존 및 involutive bracket closure)으로 환원될 수 있음을 제시한다. 즉, 향후 실증 알고리즘이 있을 때 이들이 무엇을 측정하고 왜 잠재 교란을 검출하는지에 대한 해석틀을 제공하는 데 의미가 있다.



### Toward Self-Evolution-Ready Workflow Harnesses: A Reversible Migration Path and Convertibility Taxonomy for Expert LLM Pipelines (https://arxiv.org/abs/2606.24598)
- **Prior Approaches**: 기존에는 “LLM + script”로 만든 전문가 검증 워크플로가 널리 쓰이지만, 결과 피드백이 실행에 반영되지 않아 고정된(frozen) 상태로 남습니다. 에이전트 연구도 주로 greenfield 설계나 합성 벤치마크에 초점이 있어, 실제로 이미 돌아가는 레거시 업무 흐름을 적응형 시스템으로 옮기는 ‘마이그레이션’ 문제는 충분히 다뤄지지 않았습니다. 또한 안전/평가 메커니즘 연구는 종종 프롬프트 인젝션 같은 위협을 다루거나 추상적 평가에 머무르며, 실운영 전환 과정 자체의 lived migration 보고가 부족합니다.

- **Core Contribution**: 이 논문은 레거시 LLM+스크립트 워크플로를 교체·되돌리기가 가능한 마이그레이션 경로로 바꾸는 방법론을 제안합니다. 핵심은 Strangler-Fig 방식으로 래핑 후 단계적으로 stage 단위로 분해하되, 각 단계마다 1-플래그 롤백과 무변경(business-logic change 0)을 유지하는 것입니다. 더불어 convertibility를 A/B/C로 분류하는 3단계 taxonomy를 ‘라우팅 stage’로 구현해, 어떤 워크플로는 바로 단계화하고 어떤 것은 먼저 refactor가 필요함을 진단·결정합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 ‘적응형 에이전트’의 성능이 아니라, 비즈니스 판단을 typed 입력/출력과 독립 재실행 가능한 stage 경계로 분리(convertibility)할 수 있는지 판단하는 것입니다. 저자들은 마이그레이션을 5단계(블랙박스 래핑→toolification→stage composition→결정 루프 agent화→롤백/감사)로 쪼개고, LLM이 수행하는 저위험 선택은 맡기되 되돌릴 수 없는 행동은 deterministic safety gate와 human checkpoint로 권한을 분리합니다. 또한 모든 LLM 결정·툴 호출·게이트 트리거·승인 단계가 structured trace로 남아, outcome 변화가 strategy 조정으로 이어지는 흐름을 재구성 가능하게 했습니다.

- **Empirical Impact**: WeChat 콘텐츠 워크플로 사례(Type A)에서 레거시 스크립트(순차 하드코딩 루프)를 tool과 stage로 1:1 매핑해 “zero business-logic change”와 단계별 롤백 체계를 달성했으며, subprocess 엔진과 agent 엔진이 항상 동시 공존하는 구조를 유지합니다. 안전성은 결정적 불변식 기반 테스트에서 boundary violation을 100% 차단하는 형태로 정리했고, agent 엔진의 가치는 ‘더 좋은 출력’ 단정이 아니라 explainability·adjustability·evolvability에 있음을 강조합니다. Self-evolution 신호는 생산 계정에서 topic selection을 outcome 리뷰로 조정하는 초기 관측 결과로 제시되며(인과적 ablation이 아닌 관찰 비교), 향후 교차 모델/교차 세로 재현과 정량적 지표 축적이 과제로 남습니다.



### Poster: Exploring the Limits of Audio-Based Detection of Turkish Phone Call Scams (https://arxiv.org/abs/2606.24523)
Comments:
          Poster paper accepted at 47th IEEE Security & Privacy 2026

- **Prior Approaches**: 기존 전화 사기(Scam) 탐지는 주로 영어 등 고자원 언어의 텍스트 전사(transcript) 기반 NLP에 치우쳐, 음성의 억양·강세 같은 감정/운율 단서가 약화된다는 한계가 있었다. 또한 일부 연구는 전화 전사에서 recall이 낮거나 hallucination이 발생할 수 있음을 지적했지만, 멀티모달 신호와 저자원 언어의 현실을 충분히 반영하지 못했다. 결과적으로 문화·언어적 포괄성이 떨어져 터키처럼 데이터가 적은 환경에서는 성능 격차가 커졌다.

- **Core Contribution**: 이 논문은 터키어를 중심으로 최초의 공개 멀티모달 데이터셋을 제시하며, 사기와 정상(benign) 통화 100개의 audio-transcript pair를 정렬(aligned)해 공개한다. 동시에 입력을 raw audio, ASR 전사(자동), 네이티브가 교정한 전사(인간 수정)로 나눠 LLM 안전·전사 품질이 탐지에 미치는 영향을 체계적으로 비교한다. 저자들은 인간 교정의 추가 비용이 탐지 성능에 반드시 필요한지까지 함께 검증한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 저자원 터키어에서 ASR 품질과 발화 변형(격식/구어 차이, 교착어 형태)이 커서 텍스트 기반 단서가 흔들릴 수 있다는 점, (2) 멀티모달 모델이 원음에서 욕설·위협 표현 등을 ‘안전 필터’로 거부하면 실사용 탐지가 실패할 수 있다는 점이다. 이를 위해 파인튜닝이나 프롬프트 최적화 없이 동일한 7개 LLM을 3가지 입력 조건에 그대로 적용해 공정 비교했고, 전사는 ASR+인간 교정 두 버전을 나눠 전사 품질 민감도를 확인했다.

- **Empirical Impact**: 실험 결과, 모든 모델에서 전사 기반 입력이 raw audio보다 일관되게 우수했으며, F1 평균은 Trans/UN-Trans가 약 0.99로 Audio(약 0.97)보다 높았다. 다만 Trans와 UN-Trans의 차이는 평균 0.008 수준으로 작아, 네이티브 교정이 성능 향상으로 이어지지 않을 가능성을 보여준다. audio 조건의 주요 실패 원인은 욕설·경찰 사칭·공갈/협박처럼 실제 사기에서 자주 쓰이는 표현에서의 safety refusal로, ‘공격적 실제 사례’에서 안전장치가 실용성을 훼손할 수 있음을 실증적으로 부각했다. 연구는 저자원 언어 기반의 culturally and linguistically inclusive AI safety 연구와 더 견고한 multi-modal fraud prevention 시스템의 긴급한 필요성을 강조한다.



### A Fair Evaluation of Graph Foundation Models for Node Property Prediction (https://arxiv.org/abs/2606.24509)
Comments:
          Accepted at The Workshop on Graph Foundation Models at ICML 2026

- **Prior Approaches**: 노드 property prediction은 그래프 머신러닝에서 가장 흔한 설정이지만, 기존 Graph Foundation Model(GFM) 연구는 평가가 제각각이라 모델 간 공정 비교가 어려웠다. 또한 많은 이전 결과에서 기준선으로 쓰인 GNN이 약하게 튜닝되거나 적절한 하이퍼파라미터 탐색이 부족해, GNN의 실제 성능이 과소평가됐다는 문제의식이 있었다. 그 결과 “GFMs가 항상 우월하다”는 주장과 “GNN도 충분히 튜닝하면 강하다”는 관점이 충돌했다.

- **Core Contribution**: 이 논문은 node property prediction을 대상으로 최근 9개 GFM을 GraphLand 벤치마크의 1010개 데이터셋에서 공정하게 재평가하고, 강한 GNN 기준선과 체계적으로 비교한다. 특히 PFNs(Prior-data Fitted Networks) 패러다임 기반 GFM만이 잘 튜닝된 GNN을 예측 성능에서 실질적으로 앞서는 경향을 보였고, 나머지 GFM들은 대체로 뒤처졌다. 다만 성능 우위는 더 비싼 추론(inference) 비용과 동반된다는 점도 함께 확인했다.

- **Technical Challenges**: 핵심 기술적 난제는 서로 다른 그래프 도메인이 서로 다른 노드 feature/target 공간을 갖는 환경에서, foundation model이 데이터 스코프 차이에 잘 적응하도록 평가·비교하는 것이다. 이를 위해 연구진은 GraphLand의 RL split을 사용하고, 평균 모델 순위와 데이터셋별 점수 정규화를 통해 다양한 지표를 통합 비교하는 평가 체계를 구성했다. 또한 GNN 기준선은 Platonov류와 Luo류의 개선 백본 2계열을 택해 TPE로 대규모 하이퍼파라미터 탐색(100,100 trials)까지 수행해 약한 베이스라인 문제를 줄였다.

- **Empirical Impact**: 실험 결과 GFMs는 크게 두 부류로 나뉘었는데, PFNs 기반 모델은 대부분 데이터셋에서 상위권(특히 GraphPFN) 성과를 보인 반면 비-PFNs 모델은 대체로 잘 튜닝된 GNN에 못 미쳤다. 정량적으로는 PFN 기반이 다수 데이터에서 최고 성능을 차지했고, GNN은 상위권 진입이 제한적이었으며 이 차이는 일관됐다. 대신 PFN 계열은 추론 시간이 수 초~수 분(일부는 최대 30분 수준)으로 더 길어, “정확도 대 비용”의 트레이드오프가 실무 의사결정에 중요하다는 메시지를 남긴다.



### CrossPool: Efficient Multi-LLM Serving for Cold MoE Models through KV-Cache and Weight Disaggregation (https://arxiv.org/abs/2606.24506)
- **Prior Approaches**: 기존 다중 LLM 서빙은 hot 모델에 수요가 몰리는 현상을 전제로 하지만, cold MoE 모델은 대부분의 시간에 유휴 상태로 남아 큰 가중치 메모리 비용만 부담하는 문제가 큽니다. 또한 long-context에서는 KV-cache가 요청 수명에 따라 변하는데도, 대부분의 시스템이 weights와 KV-cache를 한 덩어리(monolithic) GPU 메모리 풀에 함께 배치해 KV-cache 공유 효율이 낮아지고(특히 저동시성), 유효 컨텍스트 용량이 줄어 OOM/거절 위험까지 커집니다.

- **Core Contribution**: CrossPool은 cold MoE 모델을 위한 서빙 엔진으로, FFN weights와 KV-cache를 서로 다른 GPU 메모리 풀로 분리합니다. KV-cache 풀은 활성 요청의 KV-cache를 공유해 수요의 합에 맞게 유연하게 할당하고, attention 및 비-FFN 모듈은 KV-cache 풀에 두되 FFN은 weights 풀에서 실행하도록 경계를 설계해 성능을 유지하면서도 메모리 효율을 끌어올립니다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 attention 알고리즘/모델별 KV 헤드 구조가 만들어내는 이질적 KV 요구량을 한 풀에서 어떻게 예측·계획할지, (2) 풀을 가르는 통신 경계에서 hidden state 이동 오버헤드를 어떻게 임계 경로로 커지지 않게 할지, (3) layer-wise 혼합 스케줄링 상황에서 CUDA graph capture 이점을 해치지 않게 제어·스케줄 오버헤드를 낮출지입니다. CrossPool은 오프라인 KV-cache planner와 온라인 virtualizer로 풀 예산과 페이지 매핑을 관리하고, layer-wise pipeline scheduler로 attention-FFN 간 hidden-state 전송을 계산과 겹치며, attention/FFN을 분리해 그래프를 캡처한 뒤 persistent kernels와 control lowering으로 CPU-GPU 왕복 제어를 줄입니다.

- **Empirical Impact**: 5개의 A100(총 동일 메모리 예산) 환경에서 decode 측정을 수행한 결과, CrossPool은 bursty long-context 요청을 더 긴 범위까지 안정적으로 수용하며 fixed per-model/replica KV 파티션 방식의 ‘capacity cliff’를 완화했습니다. ShareGPT 기반 TBT(시간-토큰 간격) 비교에서 CrossPool은 kvcached 대비 P99 TBT를 최대 10.4×까지 낮추며 tail 지연을 크게 개선했고, 특히 긴 컨텍스트에서 shared KV-cache 풀의 활용도를 높여 장문 지원의 실질 한계를 확장했다는 점에서 의미가 큽니다.



### Red-Teaming the Agentic Red-Team (https://arxiv.org/abs/2606.24496)
Comments:
          v0.1

- **Prior Approaches**: 기존 보안 자동화는 딥러닝 기반 비밀번호 추정, 퍼징 등 특정 서브루틴 중심이었고, 에이전트형 시스템은 비교적 최근에 LLM을 결합해 end-to-end 보안 작업을 자동화하는 방향으로 확장됐다. 다만 커뮤니티는 성능(자율성/공격성) 향상에 집중해 온 반면, 실제 운영 환경에서 에이전트가 얼마나 안전한지에 대한 체계적 보안 평가는 부족했다. 특히 offensive security 에이전트가 프롬프트 조작이나 샌드박스에 의존해 안전하다고 가정하는 경우가 많았고, 그 취약점 분류도 표준화돼 있지 않았다.

- **Core Contribution**: 이 논문은 offensive security용으로 널리 쓰이는 agentic-red-teams 12종에 대해 AI와 시스템 양쪽을 아우르는 최초의 심층 보안 분석을 제공한다. 공격자는 타깃을 장악한 뒤 에이전트를 조작해 API key 탈취, 영속적 foothold 확보, 샌드박스 컨테이너 내부에서도 운영자 호스트까지 완전 장악으로 이어지는 공통 설계 결함을 보여준다. 또한 해당 공격 진행을 포괄하는 agentic-red-teams 전용 cyber kill chain을 정의하고, 이를 완화하기 위한 견고한 아키텍처와 설계 원칙을 제시한다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM/에이전트가 악의적으로 조작될 때(심지어 명시적 prompt-injection이 약화된 환경에서도) 실제로 worker에서 RCE 채널을 어떻게 안정적으로 여는가, 그리고 이후 권한 상승·영속화·sandbox escape까지 어떻게 막아야 하는가에 있다. 논문은 MANTIS 계열의 deception 기반 접근을 확장해 prompt-injection-free “agent-phishing”을 제안하며, 페이로드 스테이징과 trojanization(취약 코드 유도 및 입력 기반 트리거)로 near-deterministic RCE를 달성했다고 보고한다. 더 나아가 방어 설계는 LLM을 harden하는 낙관을 버리고, untrusted LLM/worker 환경의 blast radius를 최소화·격리하는 invariant와 least privilege를 중심으로 아키텍처를 구성해 완화한다.

- **Empirical Impact**: 실험에서는 frontier급 LLM(예: Claude Opus 4.8, GPT-5.5, Gemini 3.1 Pro)에서도 agent-phishing이 높은 성공률(97.8%)로 에이전트를 장악하며, 분석 대상 12개 중 10개에서 샌드박스 escape로 호스트 컴프로마이즈가 가능함을 확인했다. 이 결과는 offensive security 에이전트의 “능력”만큼이나 “안전성/격리”가 실제 운영 도입의 전제조건임을 실증적으로 강조한다. 또한 kill chain과 설계 원칙은 커뮤니티의 체계적 red teaming과 아키텍처 수준의 방어 표준화에 직접적인 기준점이 될 것으로 기대된다.



### RetiSEM: Generalising Causal Models for Fragmented Biomedical Data (https://arxiv.org/abs/2606.24488)
- **Prior Approaches**: 기존 인과 발견(예: PC, NOTEARS, LiNGAM)이나 일반적인 DAG 학습은 관측 잡음과 고차원 생의학 데이터에서 방향 추정이 불안정해지고, 생물학적으로 그럴듯하지 않은 구조를 만들 수 있다는 한계가 있다. 또 딥러닝 기반 망막 심혈관 위험 예측은 높은 예측 성능을 보이더라도 대체로 black-box이며, 상류 생리 신호가 망막 특징을 거쳐 결과로 이어지는 메커니즘 해석이 어렵다. 무엇보다 임상·분자·영상 변수가 참여자 단위에서 함께 관측되지 않는 fragmented multimodal 데이터에서는 인과 추론 가정 자체가 취약해진다.

- **Core Contribution**: RetiSEM은 유전/인구구조-분자-망막-혈관결과로 이어지는 생물학적 블록 순서를 반영한 domain-constrained SEM으로, fragmented 멀티모달에서 인과 그래프를 복원하고 매개 경로를 해석한다. 또한 forbidden-edge masking으로 허용되지 않는 역방향 간선을 배제해 탐색 공간을 줄이고, TE·NDE·NIE로 경로 수준의 직접/간접 효과를 분해해 망막 특징이 ‘지표’인지 ‘매개 신호’를 갖는지 검정한다. 결과적으로 망막을 단순 예측 변수에서 가설 검정 가능한 phenotype layer로 재정의한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 참여자 단위의 완전한 멀티모달 정렬이 없고 (2) 변수 수가 많으며 (3) 생의학적 비선형성이 존재하는 상황에서 그래프 복원이 안정적으로 이뤄져야 한다는 점이다. RetiSEM은 선형 SEM 근사로 계산 가능성과 TE/NDE/NIE 분해의 재현성을 확보하면서, DAG 탐색은 블록 순서 제약과 forbidden-edge mask로 acyclicity 및 역방향 간선 문제를 줄였다. 이후 mediation formulation을 통해 망막 블록을 매개(간접효과) 또는 반영(직접효과)하는 형태로 정량화한다.

- **Empirical Impact**: 합성 실험 10개 시나리오(차원·비선형성·인과 깊이·병렬 경로 구조 변화)에서 RetiSEM은 unconstrained baseline 대비 SHD가 낮고 causal accuracy가 높게 나타나 구조 복원과 방향성 회복이 더 일관적임을 보였다. real-world 설정(NHANES 임상 변수 + 외부 유래 망막 표현)에서는 망막 관련 NIE가 대체로 TE/NDE보다 작지만 0이 아닌 간접효과가 관측되어, 망막은 주로 downstream biomarker-like 지표로 작동하되 일부 경로에서는 mediator-like 통계 신호를 제공한다는 결론을 뒷받침한다. 전체적으로 제한 자원 멀티모달 환경에서 해석 가능한 구조적 인과 가설 테스트를 가능하게 한다는 점에서 실무적 의미가 있다.



### Adaptive Machine Learning Framework for UAV Trajectory Optimization in O-RAN (https://arxiv.org/abs/2606.24483)
Comments:
          16 pages, 12 figures, IEEE Transactions on Vehicular Technology

- **Prior Approaches**: UAV를 O-RU로 쓰는 6G/O-RAN 연구에서는 RL 기반 궤적 최적화가 유망하지만, 학습된 정책이 특정 도시 환경에 강하게 종속되어 신규 환경에서는 대개 처음부터 재학습해야 한다. transfer learning(TL)은 재학습 시간을 줄이지만, 목표 환경과 소스 환경의 유사도가 어떻게 평가·선택되는지에 대한 체계가 부족해 음의 전이가 발생할 수 있다. 또한 O-RAN 구조 내에서 모델 선택과 지속적 적응을 실전 수준으로 묶는 설계도 제한적이었다.

- **Core Contribution**: 이 논문은 O-RAN 아키텍처에 통합되는 UAV 궤적 최적화 프레임워크를 제안하며, 사전 학습 모델 라이브러리에서 ‘환경 유사도’에 따라 가장 적합한 모델을 선택해 TL을 수행한다. 유사 모델이 기준점 이상 없으면, 합성 도시 기반 fallback 모델 MGM_G로 기준선 성능을 확보하고 이후 continual learning(CL)으로 지속 개선한다. 결과적으로 적응 시간과 전이 효율을 동시에 노린 선택적 전이 + 지속 갱신 구조를 제공한다.

- **Technical Challenges**: 핵심 난제는 (1) 도시 환경의 다차원 차이를 실제 전이 성능과 연결해 유사도를 안정적으로 측정하는 것, (2) 선택된 소스가 맞지 않을 때 음의 전이를 피하는 것, (3) 이러한 로직을 O-RAN의 Near-RT/Non-RT RIC 흐름에 맞춰 실시간 추론과 연속 학습으로 구현하는 것이다. 이를 위해 도시를 평균/최대 건물 높이, 건물 점유 커버리지(밀도), UAV 운용 고도, 레이 트레이싱 맵 크기, O-DU 배치 수 등으로 요약해 유사도를 계산하고, 임계값 τ를 넘는 모델만 전이하며, 미달 시 MGM_G로 학습을 시작해 방어한다. 또한 ray tracing 기반 RSSI 맵과 3D 전파(다중경로 반사/회절)를 사용해 학습 신뢰도를 높였다.

- **Empirical Impact**: 실험(시뮬레이션)에서는 model selection 기반 TL이 scratch 재학습 대비 수렴 시간을 44%~56% 줄이고, 전통적인 TL(선택 없이 고정 전이) 대비 최대 40%까지 개선됨을 보였다. 더불어 MGM_G fallback과 CL 누적 구조가 ‘유사 모델 부재’ 상황에서도 기준선 성능을 유지하면서 점진적으로 일반화 능력을 강화하는 방향으로 작동함을 시사한다. 도시 실측 기반 레이 트레이싱 시나리오(York, Beijing, Ottawa)를 사용한 검증은 6G/O-RAN 맥락에서의 실전 적용 가능성을 높인 점에서 의미가 크다.



### video-SALMONN-R$^3$: Learning to ReWatch, ReAsk, and ReAnswer for Efficient Video Understanding (https://arxiv.org/abs/2606.24477)
- **Prior Approaches**: 기존 비디오 LLM은 계산·메모리 제약 때문에 프레임을 성긴 간격으로 샘플링하거나 해상도를 공격적으로 낮춰 처리해, QA에서 필요한 미세한 시공간 단서를 놓치기 쉽습니다. 이를 보완하려고 나온 re-watch(재시청) 방식은 다중 에이전트(여러 모델 조합)나 단일 모델의 멀티패스 전략으로 나뉘는데, 전자는 연산 오버헤드가 크고 후자는 CoT(chain-of-thought) 콜드스타트용 대규모 주석 의존도가 높습니다. 또한 CoT 기반 SFT는 사전학습된 비디오 이해 능력을 훼손하거나 편향을 유발할 수 있다는 문제가 제기됩니다.

- **Core Contribution**: 이 논문은 CoT 콜드스타트 주석 없이도 강화학습(RL)만으로 end-to-end re-watch 능력을 획득하는 video-SALMONN-R3를 제안합니다. 특히 모델이 첫 시청에서 답을 먼저 내고, 재시청 후 답을 정교화하도록 하는 re-answer 전략과, 재시청 시 질문을 다시 주입해(질문-프레임 직접 상호작용) 정합성을 높이는 re-ask 메커니즘을 함께 설계했습니다. 그 결과, localization(어디를 볼지)과 reasoning(무엇을 답할지) 분리가 필요한 상황에서도 기존 성능을 유지하면서 재시청 이점을 안정적으로 끌어올리려 합니다.

- **Technical Challenges**: 핵심 기술적 난관은 re-watch가 유도하는 reasoning-first 행동과 사전학습 비디오 LLM의 answer-first 성향 사이의 분리(불일치)로, 그대로 학습하면 성능이 흔들릴 수 있다는 점입니다. 저자들은 re-answer로 첫 시청의 답을 기반앵커로 고정한 채, 두 번째 시청에서만 수정이 일어나도록 학습 목표와 출력 구조를 맞췄고, 또한 causal attention 제약으로 재시청 프레임에 질문이 직접 못 닿는 문제는 re-ask로 해결합니다. 학습은 오디오 정렬→오디오-비디오 캡션 SFT→마지막으로 DAPO 계열 on-policy RL로 end-to-end re-watch 궤적을 주입하며, 큰 CoT 주석 없이도 ‘정답·형식·수정’에 기반한 규칙 보상으로 정책을 최적화합니다.

- **Empirical Impact**: 실험에서 video-SALMONN-R3는 6개 벤치마크(단시간부터 장시간까지)에서 베이스 모델과 QA-SFT 기준선을 일관되게 능가하며, prior re-watch 접근들보다 더 낮은 연산 비용으로 성능을 끌어올렸다고 보고합니다. 특히 VideoMME·LVOmniBench 같은 장문 비디오에서 개선폭이 커져, 긴 영상에서 균일 샘플링의 한계를 ‘필요 구간만 집중 재시청’으로 상쇄한다는 관찰과 연결됩니다. 아울러 ablation 결과는 두 번 답하기만으로는 이득이 제한적이며, localization에 맞춘 재시청과 re-ask·revise 보상이 함께 작동해야 효과가 재현된다는 점을 강조합니다.



### G$^3$VLA: Geometric inductive bias for Vision-Language-Action Models (https://arxiv.org/abs/2606.24472)
Comments:
          Submitted to CoRL 2026

- **Prior Approaches**: 기존 VLA(vision-language-action) 모델은 2D 이미지 토큰을 주로 다루며, 카메라 intrinsics/extrinsics로 결합된 다중 뷰의 기하 구조를 행동 감독을 통해 간접적으로 학습해 왔습니다. PerAct, RVT, Act3D 등은 3D 표현(복셀/렌더/3D feature field)을 쓰며 공간 정밀도를 높이지만, 사전학습된 VLM의 의미 토큰을 그대로 활용하기 어렵거나 액션 표현을 바꾸는 경우가 많았습니다. SpatialVLA와 3D-VLA 같은 연결 시도도 대개 3D 센서 입력, 대규모 공간 사전학습, 혹은 액션 표현 수정이 필요하다는 한계가 남아 있습니다.

- **Core Contribution**: 이 논문은 카메라 보정(calibrated) 기하를 VLA의 시각 토큰 스트림에 “가벼운 인터페이스”로 주입하는 G^3VLA를 제안합니다. 백본/액션 공간/모방(imitation) 목적은 그대로 둔 채, intrinsic-conditioned ray embeddings, PRoPE(projective positional encoding), 양방향 cross-view fusion으로 로봇 카메라 모델 기반 구조를 토큰에 주입합니다. 또한 depth 센서 없이도 ground-truth point map이 있거나, confidence-gated π^3X(π^3π^{3}X) teacher 예측을 사용해 기하를 감독합니다.

- **Technical Challenges**: 핵심 과제는 “2D 토큰 기반 인터페이스”에 캘리브레이션된 다중 뷰 기하를 넣되, 사전학습된 VLA의 행동 경로를 깨지 않는 것입니다. 저자들은 patch 단위 토큰이 각 카메라의 viewing direction을 알도록 ray embeddings로 intrinsics를 주입하고, PRoPE로 뷰 간 투영 관계를 attention bias로 제공한 뒤, cross-view fusion에서 기하 컨텍스트를 교환하게 했습니다. 또 기하 모듈을 즉시 행동 손실만으로 학습하면 불안정할 수 있어, 보조 점 헤드 기반의 dense geometry distillation과 two-stage curriculum(기하 선학습→전체 파인튜닝)으로 안정적으로 부트스트랩합니다.

- **Empirical Impact**: 실험에서 G^3VLA는 LIBERO, RoboCasa24, RoboTwin2.0 및 실제 로봇 환경 전반에서 일관된 성능 향상을 보이며, 특히 공간/객체 민감 태스크에서 개선폭이 큽니다. π0 설정에서는 ground-truth 기하 감독이 가장 강하지만, depth가 없을 때도 π^3X 증류로 기준 대비 큰 이득을 회복해 실용성을 확인했습니다. 추가로 π0.5에서는 소폭 추가 이득이 유지되고, GR00T 1.5에서는 geometry-aware 토큰이 액션 생성 경로에 직접 도달할수록 효과가 커진다는 해석이 제시되어, “기하 전이”의 설계 조건도 함께 시사합니다.



### The African Language Tax: Quantifying the Cost, Latency, and Context Penalty of Tokenizing African Languages in Frontier LLMs (https://arxiv.org/abs/2606.24460)
Comments:
          40 pages, 5 figures, 25 tables

- **Prior Approaches**: 기존 연구는 다국어 토크나이제이션이 언어 간 효율 차이를 만들고, 토큰 수 격차가 비용과 활용성 불평등으로 이어진다는 현상을 일반적으로 다뤘습니다. 다만 유럽 중심의 병렬 실험에 비해 아프리카 언어는 같은 해상도로 체계 측정되지 않았고, 특히 실제 기업 배포 관점(비용·지연·컨텍스트 용량)으로 환산된 정량 결과가 부족했습니다.

- **Core Contribution**: 이 논문은 FLORES-200+ 등 병렬 코퍼스를 바탕으로 20개 아프리카 언어(5개 언어 계통, 3개 문자: Latin, Ge'ez/Ethiopic, N'Ko)를 대상으로 토크나이저의 ‘tokenization premium(영어 대비 토큰 프리미엄)’을 정밀 측정하고, 이를 배포 의사결정 단위로 변환합니다. 또한 afri-fertility 공개 도구, 공개 리더보드, 결과 데이터셋, 완화 가이드를 함께 제공해 측정과 검증을 반복 가능하게 만들었습니다. 결과적으로 토크나이제이션 격차를 “디지털 격차가 서브워드 어휘에 인코딩된 구조적 페널티”로 명확히 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘의미는 동일하지만 토큰 수만 달라지는’ 병렬 설계를 통해 콘텐츠 편향을 제거하면서도, 언어별 단어 경계·정규화·문자 밀도 차이까지 통제하는 것이었습니다. 저자들은 UAX-29 기반 단어 분절과 NFC 정규화를 고정하고, 필요 시 문자/바이트 기반 지표(CPT/BPT)를 병행해 형태론적으로 분절이 흔들릴 수 있는 언어(특히 에티오피아 문자·고교착 언어)에서 결론이 단어 수에만 의존하지 않도록 했습니다. 더 나아가 토크나이저별 토큰 수를 합산-나눗셈(sum-then-divide)으로 집계하고, 문장 단위 부트스트랩 95% 신뢰구간으로 안정성을 확보했습니다.

- **Empirical Impact**: 11개 frontier·open 토크나이저를 FLORES-200+에 적용한 결과, 모든 아프리카 언어는 영어보다 높은 토큰 프리미엄을 보였고 중앙값은 1.88x(예: GPT-5 / o200k_base), 최대는 N'Ko에서 8.92x에 달했습니다. 비용·지연·컨텍스트 관점으로 환산하면 최대 추론 비용 8.9x, 생성 지연 배율 8.9x, 그리고 컨텍스트 윈도우 유효 용량이 영어 대비 최소 11% 수준으로 줄어드는 것으로 나타났습니다. Gemma 4가 프리미엄 평균을 3.31x(cl100k_base)→2.38x로 낮추지만, 페널티를 완전히 제거하는 토크나이저는 없었고 아프리카 빌더의 비용 부담이 언어 사용 가능성과 직접 연결된다는 점을 실증했습니다.



### NoContactNoWorries: Estimating Contact through Vision and Proprioception for In-Hand Dexterous Manipulation (https://arxiv.org/abs/2606.24450)
Comments:
          Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems(IROS) 2026

- **Prior Approaches**: 기존 비전 기반 덱스터러스 조작은 영상에서 정책을 학습하거나 자세/물체 재배치를 수행하지만, slip·힘 분포·국소 안정성처럼 접촉 의존 현상은 자기 가림과 감독 신호의 약함 때문에 일관된 일반화가 어렵습니다. 반면 촉각 기반 접근은 접촉 위치의 직접 증거를 제공하지만, 센서의 비용·취약성·커버리지 한계·배선/캘리브레이션 부담 때문에 확장성이 떨어집니다. 또한 비전-터치 융합이나 self-supervised 선학습은 효과적일 수 있으나, 결국 배치 시 촉각 입력이 필요하거나 다운스트림 정책을 재학습하는 경우가 많았습니다.

- **Core Contribution**: 이 논문은 촉각 하드웨어 없이도 ego-centric RGB-D와 proprioception만으로 손가락 접촉(contact)을 이진 신호로 추정하는 pseudo-tactile sensing을 제안합니다. 제안 모델 NoContactNoWorries는 RGB-D 토큰과 현재/명령된 관절 상태를 transformer의 cross-attention으로 융합하고, causal attention으로 짧은 시간 정보를 누적해 각 손가락 위치의 이진 접촉을 실시간 관측처럼 제공합니다. 또한 단일 접촉 예측 모델을 여러 물체에 대해 학습한 뒤, 예측된 접촉 신호를 시뮬레이션에서 강화학습 에이전트의 다운스트림 입력으로 사용해 새 물체로도 일반화되는 것을 보입니다.

- **Technical Challenges**: 가장 큰 난제는 단일 프레임에서 접촉이 단순 근접과 쉽게 구분되지 않으며(모호성), 비전은 자기 가림/시점 의존성이 크고, proprioception은 외부 물체 기하를 직접 담지 못한다는 점입니다. 이를 위해 frozen RGB-D 백본에서 시공간 특징을 뽑고, 현재 관절과 명령 관절을 별도 쿼리로 사용해 ‘반응적/예측적’ 정보가 다른 방식으로 장면 토큰을 선택하도록 설계했으며, causal transformer로 시간적 단서를 누적해 접촉을 추론합니다. 학습은 PhysX/실물 저프로파일 FSR로 이진 접촉 레이블을 만들고, 모델은 이 추정치를 정책에 직접 넣는 real-time observation 형태로 사용하도록 구성했습니다.

- **Empirical Impact**: 시뮬레이션과 실제 로봇(LEAP hand) 모두에서 이진 접촉 예측의 F1이 상위 수준으로 나타났고, 시뮬-to-실물 전이에서도 견고하게 유지되었으며 모호한 비전 단서만으로는 성능이 크게 떨어졌습니다. 특히 카메라 가시성 기준으로 나눴을 때 접촉 사이트가 가려진 경우 비전-only 성능이 급감해, 제안된 융합(시각+자세+명령+시간)이 자기 가림 상황에서 실제로 기여함을 뒷받침합니다. 더 나아가 예측된 접촉 신호를 oracle/tactile을 대체해 시뮬에서 학습한 in-hand reorientation 정책이 실제 하드웨어로 직접 이전(정책 재학습 없이) 가능한 점이 덱스터러스 조작에서 촉각의 ‘가상 센서화’ 가능성을 실증합니다.



### MedPCFM: Improving Medical Point Cloud Completion by Integrating Point Transformers and Flow Matching (https://arxiv.org/abs/2606.24433)
Comments:
          25 pages, 9 figures

- **Prior Approaches**: 의료 point cloud completion은 결측 부위가 크고 얇은 구조가 많아 “가능한 완성”이 여러 개일 수 있어 어렵다. 기존에는 분할 기반의 voxel 중심 파이프라인(예: 3D U-Net, V-Net)이 주로 쓰였고, 최근에는 PCDiff처럼 diffusion을 의료 결측 보완에 적용한 시도가 늘었지만 sampling 비용이 높고 속도-정확도 절충이 부담이었다. 또한 deterministic 모델(encoder-decoder)이나 diffusion의 비교가 충분히 매칭되지 않아, 생성형 관점의 효율성과 일반화 특성이 명확히 정리되지 않았다.

- **Core Contribution**: 이 논문은 PTv3를 기반으로 한 flow matching 연속시간 생성모델 PCFM을 제안해 의료 point cloud completion을 다룬다. PCFM은 관측된 결손 입력 X를 조건으로, 단순 분포에서 출발해 연속적 transport(ODE)를 통해 완성 분포로 이동시키는 방식으로 multi-modality를 생성형으로 포착한다. 또한 PTv3를 deterministic encoder-decoder 및 diffusion(PCDiff) 비교군에 동일·유사하게 적용해 공정한 벤치마크 구조를 마련하고, SkullFix/SkullBreak에 더해 Mandibular Defect까지 확장 검증한다.

- **Technical Challenges**: 핵심 기술적 과제는 (1) point set의 순열 불변성과 고해상도 처리를 유지하면서 (2) diffusion 대비 훨씬 적은 ODE 적분 단계로도 충분한 품질을 내는 flow matching 설계다. 이를 위해 PTv3의 space-filling curve 직렬화, Serialized Attention/풀링-언풀링으로 점 집합 표현을 안정화하고, affine path(예: Conditional-OT 계열)와 scheduler 선택, 선택적 contrastive regularizer(배치 내 음쌍 permutation)를 조합해 학습 신호를 정교하게 만든다. 더 나아가 Heun 방법 기반의 ODE 적분으로 sampling-step 예산을 통제하고, diffusion과 동일한 운영 관점에서 step budget trade-off를 체계적으로 비교한다.

- **Empirical Impact**: 실험은 SkullFix/SkullBreak 및 Mandibular Defect 3개 데이터셋에서 수행되며, PCFM(PTv3)이 generative 성능에서 diffusion·deterministic 기준을 전반적으로 상회하거나 경쟁력 있게 나타났다. 특히 CD/DSC/BDSC/HD95 등 point·voxel 계열 지표에서 PTv3 기반 PCFM이 최상 운영점에서 diffusion 대비 훨씬 적은 steps로 품질을 유지하는 경향을 보였다. 속도 관점에서는 최적 설정에서 PVCNN 대비 PCFM이 최대 7×, PCDiff가 약 5.5× 가량 빠른 speed-up을 보고하며, 추가로 모델 크기와 점 개수(cardinality) 스케일링에서 점 해상도가 성능을 좌우하고 모델 스케일은 기여가 점진적으로 둔화되는 경험적 트렌드를 제시한다.



### Transformation Behavior of Images in Latent Spac (https://arxiv.org/abs/2606.24430)
- **Prior Approaches**: 히스토패톨로지 분류에서는 라벨이 적다는 한계를 줄이기 위해 데이터 변환(augmentation)과 인코더(encoder) 기반 임베딩을 함께 활용하는 흐름이 일반적이다. 또한 Barlow Twins, MoCoV2, SwAV, DINO처럼 self-supervised 방식으로 임베더를 학습하며, 변환에 대해 불변(invariant)한 표현을 기대한다. 하지만 “클래식 이미지 변환이 실제로 잠재공간에서 얼마나 상쇄되는지”를 체계적으로 정량 비교한 연구는 부족했다.

- **Core Contribution**: 본 논문은 히스토패톨로지 타일을 대상으로, 원본 이미지와 변환된 이미지의 임베딩이 잠재공간에서 얼마나 가까워지는지(또는 얼마나 달라지는지)를 비교해 임베더의 변환 민감도를 측정한다. 이를 통해 augmentation이 성능을 높이는 이유가 단순한 데이터 확대를 넘어, 임베딩 공간에서 변환 효과가 완전히 중화되지는 않는 현실을 반영할 수 있음을 보여준다. 또한 ImageNet 사전학습 기반 임베더와 병리학 특화 임베더 간 차이도 함께 관찰한다.

- **Technical Challenges**: 핵심 과제는 임베더의 “이상적인 불변성”이 이론적 기대치인 반면, 실제 학습된 네트워크가 유한 자원에서 불완전하게 변환 영향을 남긴다는 점을 정량화하는 것이다. 연구진은 각 변환 전후 임베딩의 L2 거리와, 동일 잠재공간에서의 random 임베딩 간 거리(ARD)를 정규화해 네트워크/변환 간 비교 가능하게 만들었다. 추가로 차원별 변화 분포와 Generalized Jaccard Index(GJI)를 사용해 어떤 잠재 차원들이 변환에 의해 공통적으로 영향을 받는지(부분적 disentanglement 실패)를 분석했다.

- **Empirical Impact**: TCGA와 내부 CAD 두 WSI 데이터에서 해상도(0.25~2 mpp) 및 데이터셋 간 유사한 경향을 확인했으며, 어떤 변환에서도 임베딩이 완전히 불변은 아니었다. 공간 변환(플립/크롭)은 상대적으로 영향이 작지만, 색상 관련 변환(color jitter, 그레이스케일/정규화 등)이 임베딩 이동을 더 크게 만들었고 이는 스캐너 간 색 차이가 분류 일반화에 영향을 줄 수 있음을 시사한다. 더불어 병리학 특화 인코더가 ImageNet 기반 baseline보다 더 높은 변환 견고성을 보였지만, 동시에 일부 변환이 특정 차원에 국한되지 않고 여러 차원에 분산되어 나타나 잠재공간 disentanglement이 완전하지 않다는 메시지를 남긴다. 



### Detecting AI Coding Agents in Open Source: A Validated Multi-Method Census of 180 Million Repositories (https://arxiv.org/abs/2606.24429)
- **Prior Approaches**: 기존 연구는 주로 PR(풀 리퀘스트)이나 특정 설정 파일(.cursorrules, .cursorrules 등) 같은 단일 신호에 의존해 에이전트의 실제 채택 범위를 과소평가해 왔습니다. PR 기반은 bot-account 중심(예: Type A)을 잘 잡지만 silent agent(코드 변경만 하고 흔적이 거의 없는 Type D)나 developer identity를 그대로 쓰는 distributed attribution(예: Type C)을 놓치기 쉽습니다. 반대로 설정 파일 기반은 특정 IDE/플러그인의 adoption 프록시는 제공하되, commit 단위 기여나 PR 채널 활동은 거의 관측하지 못합니다.

- **Core Contribution**: 이 논문은 버전관리 히스토리에 남는 흔적을 기준으로 AI 코딩 에이전트를 4가지 행동 유형(Type A~D)으로 분류하고, 이를 World of Code(WoC) 인프라 위에 다층 탐지 프레임워크로 구현했습니다. 구성 파일 스캔, 커밋 메시지 분석, author-identity 매칭, bot-signature 조회를 함께 사용해 단일 신호가 만드는 대표성 격차를 계량합니다. 또한 커밋 기반과 PR 기반(AIDev)의 관측이 서로 거의 다른 에이전트 집단과 작업 유형을 포착한다는 점을 직접 비교합니다.

- **Technical Challenges**: 핵심 기술 난제는(1) 단일 탐지 신호로는 undercount가 필연이라는 점, (2) 에이전트가 사람 정체성 아래에서 조용히 활동할 수 있다는 점, (3) 개발자 alias(이메일/이름 변형)와 fork로 인한 중복을 정확히 정리해야 한다는 점입니다. 논문은 WoC의 identity resolution(약 3,800만 raw author ID aliasing)과 deforking을 적용하고, author-email 해시 조회, 커밋 메시지 정규식, author-name suffix 스캔, end-of-line anchoring된 설정 파일 매칭을 조합해 multi-method union을 산출합니다. Claude Code처럼 봇 계정만으로는 거의 보이지 않는 경우, 메시지 시그니처를 추가하면 탐지량이 최대 30배까지 커지는 차이를 검증했습니다.

- **Empirical Impact**: 180M+ 저장소를 대상으로 2024년 12월~2026년 4월의 스냅샷을 분석한 결과, commit-attributed 에이전트가 월 32만 건 이상 커밋을 생성하는 등 활동 규모가 확인됐습니다. 특히 Claude Code는 multi-method 탐지 시 850,157 커밋(한 스냅샷 기준)으로 측정되며, bot-signature 단독 대비 탐지 누락이 크게 나타나 단일 신호 기반 유병률 추정이 구조적으로 낮게 나옴을 보여줍니다. PR 기반(AIDev)과 비교하면 Claude Code의 commit-detected 채택 프로젝트 중 79%가 PR census에서 빠지고, 작업 성격도 다르게 나타나(예: PR 채널은 feature 중심, commit 채널은 maintenance/bug-fix 성격) 어떤 한 채널도 생태계를 대표하지 못한다는 결론을 제공합니다.



### Entity Resolution via Batched Oracle Queries (https://arxiv.org/abs/2606.24407)
- **Prior Approaches**: 기존 엔터티 레졸루션(ER)은 대부분 레코드 쌍을 비교하는 매칭 함수에 기반해, 데이터가 커질수록 비교 비용이 이차적으로 증가한다. 이를 줄이기 위해 blocking과 prioritization 같은 기법이 발전했지만, 여전히 “어떤 배치를 먼저 물어볼지” 같은 예산 배분 문제를 정교하게 다루기엔 한계가 있었다. 최근에는 배치 단위로 클러스터링을 내는 Set Transformer, 비전 기반 클러스터링, crowdsourcing, LLM 프롬프팅 등 더 큰 과립도로 처리하는 접근이 등장했지만, 예산을 나눠가며 단계마다 최대 recall을 보장하는 체계가 부족했다.

- **Core Contribution**: 논문은 제한된 크기의 배치를 처리하는 오라클로부터, 배치 전체가 특정 엔터티의 모든 레코드를 포함하지 않는 상황에서도 대규모 데이터의 엔터티를 점진적으로 복원하는 문제를 “progressive batched entity resolution”으로 형식화한다. 또한 최적 배치 스케줄이 항상 존재하지 않을 수 있고, 다음 배치를 “예상 이득을 최대화”하는 방식으로 고르는 것이 NP-hard임을 보이며 문제의 본질적 난도를 확정한다. 그 위에 엔터티 크기에 대한 자연스러운 조건에서의 최적해(또는 근사적 가이드)와 같은 알고리즘적 해결책을 제시한다.

- **Technical Challenges**: 핵심 기술 난제는, 배치 ER이 이전 질의들의 결과에 의존해 상태 공간이 지수적으로 커지며 각 배치의 ‘추가로 발견되는 매치 수’가 누적된 질의들 전부와 결합된다는 점이다. 논문은 오라클 동작을 일관되고 오류 없는 것으로 이상화한 뒤, 각 단계에서 Match edge(새로 발견되는 매치)를 최대화하도록 “다음 배치 선택”을 배치 배치 최적화 문제로 환원하고 NP-hard성을 bin packing류로 증명한다. 아울러 엔터티를 배치 질의를 통해 더 작은 대표 레코드 수로 줄이는 r ecursive remainder/quotient 기반의 이론적 상한·하한을 세워, 실제 구현에서는 매 단계 이득을 끌어올리는 배치 선택 전략을 설계한다.

- **Empirical Impact**: 여섯 개 데이터셋(실세계 및 합성)을 대상으로 평가한 결과, 제안한 방식은 같은 수준의 oracle consult 예산 하에서 state-of-the-art baseline보다 더 높은 recall을 보이며 우수성을 입증했다. 또한 배치 크기가 커질 때 성능이 흔들릴 수 있는 LLM/클러스터링 계열 오라클의 제약(context window, 비용, 고정 배치 처리)을 고려하더라도 pay-as-you-go 방식의 효과가 관측된다. 결론적으로, 대규모 ER에서 비용 통제와 성능 향상을 동시에 다루는 “진행형 배치 질의” 설계의 실증적 기준을 제시했다는 점에서 의미가 크다.



### Average Rankings Mask Per-Subject Optimality: A Friedman-Nemenyi Benchmark of EEG Motor-Imagery BCI Decoders (https://arxiv.org/abs/2606.24394)
Comments:
          16 pages, 6 figures, 4 tables

- **Prior Approaches**: 모터 이미지를 EEG로 분류하는 BCI에서는 CSP 같은 공간 기반 방법과 tangent-space 같은 Riemannian 방법이 기본값으로 자주 권장돼 왔다. 하지만 개인 간·세션 간 변동성이 커서, 고정된 한 파이프라인이 환경 변화에 강건하다는 주장이 실제로는 조건에 따라 달라질 수 있다는 문제의식이 축적돼 있다.
또한 time/frequency/분해 기반 특징과 Hjorth, HFD, Hurst, SVD entropy 같은 비선형 기술자, 연결성(connectivity) 특징 등 다양한 표현이 제안돼 “무엇이 언제 잘 먹히는가” 질문이 더 복잡해졌다.

- **Core Contribution**: 이 논문은 “한 파이프라인이 전반적으로 더 낫다”는 약한 형태의 주장조차 가장 유리한 평가 조건(동일 참가자·동일 세션 내 학습/검증)에서 검증한다. MOABB 프레임워크로 3개 공개 모터-이미지(left vs right) 데이터셋에서 1,056개 end-to-end 디코딩 구성(feature extractor × scaler × classifier)을 대규모로 비교해, 파이프라인 보편성의 하한선을 제시한다.
핵심 결론은 cov-tgsp와 CSP가 평균 성능 1·2위를 보이지만, 데이터셋에 따라 순서가 바뀌고 최대 집단에서는 통계적으로 동률(closed-form)이며 개인별 “승자”가 갈린다는 점이다.

- **Technical Challenges**: 가장 큰 기술적 도전은 개인 내 변동까지 포함해 파이프라인 간 성능 차이를 공정하게 비교하면서, 단순 평균이 아니라 통계적 유의성과 효과크기를 함께 보이는 것이다. 이에 따라 Friedman 검정(Omnibus)과 Nemenyi critical-difference 분석, Wilcoxon signed-rank(효과크기 포함)로 다중 파이프라인 순위를 비교하고, ‘winner-per-subject’ 및 ‘per-subject oracle(개인별 최적 선택)’까지 분해해 heterogeneity의 크기를 수치화했다.
또한 사전처리를 최소화해 공정성을 확보했지만, 비선형 기술자가 잡음/아티팩트 민감도가 더 클 수 있음을 한계로 명시하고 비선형 결과를 “개인 이질성 가설”로 해석하도록 설계했다.

- **Empirical Impact**: 실험에서 cov-tgsp와 CSP는 평균 정확도에서 앞섰지만(예: PhysionetMI에서 통계적으로 구분되지 않음), 단일 최적 파이프라인이 대부분 사용자를 커버하지 못했다. PhysionetMI의 경우 ‘한 파이프라인 고정’의 승률은 약 35% 수준이며, 개인별로 비선형 기술자가 약 1/3에서 최선이 되어 공간/Riemannian 단일 기본값의 한계를 보여준다.
반대로 feature dimensionality는 성능을 단조롭게 설명하지 못했고, classifier/scaler 선택은 강한 feature 표현을 고른 뒤에는 크지 않은 2차 요인으로 나타났다. 결과적으로 연구·제품 모두에서 “보편 디코더 탐색”보다 “참가자 인지(participant-aware) 모델 선택/개인화”에 투자할 근거가 정량적으로 강화되었다.



### Female-RHINO: A Real-Time Scanner-Integrated Framework for Automated Quantitative Uterine MRI Analysis and Structured Reporting (https://arxiv.org/abs/2606.24390)
- **Prior Approaches**: 기존 자궁 MRI 평가는 해부학적 변동성과 병변/생리 변화, 관찰자 의존성 때문에 표준화가 어렵고 주로 검사 후 수작업 계측에 의존해 왔다. 딥러닝 기반 자동화도 주로 segmentation, landmark detection 같은 단일 태스크 중심의 오프라인 방식에 머물러 실제 촬영 중 워크플로우에 바로 결합되지 못했다.

- **Core Contribution**: 이 논문은 Female-RHINO로, MRI 촬영 중 실시간으로 자동 정량 분석과 structured reporting을 생성하는 scanner-integrated end-to-end 프레임워크를 제안한다. sagittal T2-weighted 골반 MRI에서 3D 분할과 3D landmark를 동시에 수행해 자궁/병변 volumetry, Nabothian cyst·fibroid 계측 및 6개 landmark 기반 바이오메트릭을 자동 산출한다.

- **Technical Challenges**: 핵심 기술 난제는 다양한 프로토콜·벤더·필드세기에서의 도메인 차이와, 큰 근종 등 해부학적 왜곡 상황에서의 landmark 안정적 국소화였다. 이를 위해 nnUNet-v2 기반 3D segmentation과 Swin-UNETR heatmap 회귀 landmark detection을 함께 학습하고, ROI 추출·공간 재샘플링·앙코럼된 기하적 priors 및 엄격한 radial error(10mm) 실패 기준으로 품질을 관리했다. 또한 Fire/Gadgetron 인터페이스로 DICOM 스트리밍→GPU 추론→HTML 보고서 생성까지 지연을 줄여 스캔 진행 중 결과 제공을 목표로 했다.

- **Empirical Impact**: 다중 센터 500+ 데이터(독립 회고/전향 코호트)에서 자궁 Dice 0.82, fibroids Dice 0.80을 달성했으며 landmark는 평균 radial error 3.7mm로 보고됐다. 전향 배치에서는 자동 volumetry/바이오메트릭 정확도가 관찰자 간 변동과 유사했고, end-to-end 처리 시간은 70초 미만으로 실시간 보고가 가능함을 보였다. 결과적으로 자동화된 정량·시각화 기반의 재현성 높은 자궁 MRI 표준화를 촉진하며 임상 효율 향상 잠재력을 제시한다.



### On the Stability of Prompt Ranking in Large Language Model Evaluation (https://arxiv.org/abs/2606.24381)
- **Prior Approaches**: 기존 프롬프트 엔지니어링 연구는 instruction phrasing, reasoning cues, 출력 제약 등 설계가 성능에 큰 영향을 준다는 점을 보여줬습니다. 또한 few-shot 예시 선택이나 stochastic decoding, 데이터 서브샘플링에 따른 출력/점수 변동은 다뤄졌지만, 프롬프트들 사이 ‘순위’의 안정성은 체계적으로 다루지 않았습니다. 결과적으로 평균 정확도나 단일 최고 프롬프트 같은 결정이 평가 잡음에 얼마나 취약한지 과소평가되어 왔습니다.

- **Core Contribution**: 이 논문은 프롬프트 평가를 확률적(불확실성 포함) 순위 문제로 보고, random seed와 제한된 평가 예산(서브셋 크기) 같은 현실적 변동 하에서 순위 안정성을 정량화합니다. 특히 Spearman/Kendall 같은 전역 순위 상관이 중간~높게 나와도 top-1(또는 top-k) 선택은 자주 바뀐다는 ‘결정 수준의 불일치’를 보여줍니다. 이를 바탕으로 mean 기반 선택 대신 performance와 분산을 함께 고려하는 lower confidence bound(LCB) 기반 안정성-aware 선택 전략을 제안합니다.

- **Technical Challenges**: 핵심 과제는 평가 조건이 바뀔 때 프롬프트 성능의 상대적 순서가 얼마나 흔들리는지, 그리고 그 흔들림이 실제 top-prompt 선택을 어떻게 망가뜨리는지 분해해 측정하는 것입니다. 저자들은 고정된 프롬프트 풀을 여러 seed와 서브셋 크기로 반복 평가해 점수 행렬을 만들고, Spearman’s ρ·Kendall’s τ로 전역 순위를, top-1/top-k consistency로 선택 결정의 재현성을 동시에 봅니다. 또한 leave-one-seed-out(LOSO) 프로토콜로 선택 전략이 보이지 않는 평가 조건에서도 일반화되는지 검증하고, LCB로 불확실성을 페널티화해 noisy 상황의 선택을 보정합니다.

- **Empirical Impact**: Mistral-7B-Instruct-v0.3, Phi-3-mini-4k-instruct, Qwen2.5-7B-Instruct의 세 모델과 GSM8K·MMLU 두 벤치마크에서, 전역 순위 상관은 종종 중간~높지만 top-1 일치율은 낮아(특히 GSM8K에서 서브셋 50 기준 top-1 약 40%대) ‘선택 결정’이 불안정함을 확인했습니다. LOSO에서 LCB 전략은 평균 기반 선택을 작은 평가 예산에서 유의미하게 개선하며(예: Qwen GSM8K size 50에서 0.228→0.312), 큰 예산에서도 경쟁력을 유지합니다. 반면 비교적 안정적인 MMLU에서는 분산 페널티로 평균 정확도가 소폭 감소할 수 있어 robust trade-off를 보여줍니다. 전반적으로 프롬프트 배치/벤치마킹에서 점수 한 점(점추정)에만 의존하지 말고 평가 불확실성을 반영해야 한다는 실무적 경고와 방법론을 제공합니다.



### Structural Kolmogorov-Arnold Convolutions: Learnable Function on the Values or the Filter Shape as Parameter-Efficient Alternative to Per-Edge Convolutional KANs (https://arxiv.org/abs/2606.24371)
- **Prior Approaches**: 기존 Convolutional Kolmogorov–Arnold Networks(ConvKAN)은 커널의 각 항목마다 입력 픽셀 값에 작용하는 univariate function을 따로 두는 per-edge 방식이 주류였다. 이 접근은 표현력은 높지만 함수 수가 입력 채널·출력 채널·커널 면적에 비례해 파라미터와 연산이 급증하고 과적합 위험도 커진다. 또한 맞춘 예산에서 정확도 이득이 제한적이거나, Gram 변형처럼 좋은 결과는 더 많은 파라미터를 요구하는 경향이 있었다.

- **Core Contribution**: 이 논문은 learnable function의 “위치”를 value(픽셀 값)와 shape(필터 구조)라는 단일 축으로 재정의하고, per-edge 대신 구조(필터 형상) 쪽에 function을 배치하는 structural KAN을 제안한다. RF-KAN은 Morlet wavelet 기반의 ridge profile로 필터 shape를 만들고, SV-KAN/AG-KAN은 픽셀 값에는 단일 shared function을 쓰되 shape는 각각 정적 필터 또는 content-adaptive Gaussian gate로 제공한다. 핵심 메시지는 동일한 파라미터 스케일에서 value에도 shape에도 function을 배치했을 때 “경쟁 구간”에 도달하며, per-edge 방식의 과도한 함수 분산 없이도 성능이 나온다는 점이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 function을 구조에 두면 학습이 어떻게 안정적으로 이뤄지고, shape 표현이 실제 이득으로 연결되는지 설계해야 한다는 것이다. 저자들은 RF-KAN에서 ridge projection(방향을 내적 좌표로 축약해 ridge 형태를 만들고), Morlet basis로 부드럽고 국소화된 진동적 프로파일을 조합하며, 채널-평균 패치 기반 routing으로 content-adaptive amplitudes를 더해 필터가 입력에 맞게 조정되도록 했다. 또한 RF-KAN의 연산 효율을 위해 연속 좌표상에서 필터를 구성한 뒤 bilinear upsampling을 선형적으로 접어 넣어, 큰 중간장을 다루는 비용을 줄이면서도 필터의 연속성을 유지했다.

- **Empirical Impact**: CIFAR-10/100에서 4-layer 동일 프로토콜로 비교했을 때 RF-KAN은 CIFAR-10 88.47±0.10% (약 0.40M 파라미터), SV-KAN은 88.20±0.31%로 per-edge KAN들과 plain convolution을 모두 앞섰다. CIFAR-100에서도 RF-KAN 64.40±0.19%, SV-KAN 64.57±0.30%로 거의 동률이며, 두 모델 모두 per-edge Gram 변형 대비도 파라미터 효율에서 우위를 보였다. 제어 실험/어블레이션은 성능 향상이 Morlet basis의 intrinsically localised oscillatory 표현과 content adaptivity의 조합에서 오며, learned shape를 제거하면 정확도가 40점 이상 붕괴해 “shape가 하중을 담당”한다는 점을 명확히 했다.



### What Does ODRL Mean? A Cross-Level Ontological Grounding of Permissions, Prohibitions, and Duties in UFO-L (https://arxiv.org/abs/2606.24344)
Comments:
          Accepted at FOIS 2026 (16th International Conference on Formal Ontology in Information Systems), Vitória, Brazil; to appear in Frontiers in Artificial Intelligence and Applications, IOS Press. 16 pages, 1 figure, 2 tables

- **Prior Approaches**: ODRL policy evaluators는 정책의 활성/위반/이행 여부 같은 verdict만 산출했지만, 그 결정이 전제하는 규범의 위치(누가 어떤 권한·의무를 갖는지)나 위반 선언 권위(누가 제재를 요구할 수 있는지)는 모델링하지 못했다. 또한 violation이 어떤 consequential한 결과로 이어지는지에 대한 평가 절차가 형식 의미론에 명시되지 않아, remedy 의무가 있어도 권위 근거가 공백으로 남는다.

- **Core Contribution**: 이 논문은 violable하면서도 consequential한 규범 표현에는 conduct-level 위치(Permission, Duty, Right, No right)와 competence-level 위치(Power, Subjection, Immunity, Disability)가 함께 필요하다는 Cross-Level Design Principle을 제안한다. 이를 ODRL에 적용해, prohibition은 sanctioned(위반 가능+결과 발생)하며 permission의 behaviour parameter(open vs closed world)는 상호 배타적인 규범 위치를 만든다는 점, 그리고 기존 형식 의미론 범위가 achievement obligations에만 정합적임을 정리한다.

- **Technical Challenges**: 핵심 과제는 “규칙이 활성화될 때 실제로 어떤 법적 개체(규범적 위치의 존재)가 생기는가”를 정의해 evaluator의 verdict와 분리된 규범론을 결합하는 것이었다. 저자들은 UFO-L에 기반해 각 활성 규칙을 단일 legal relator로 맵핑하고, violation-to-remedy 전이를 정당화하는 Power–Subjection 쌍(위반 선언 권위)을 activation 시점에 명시적으로 구성하도록 확장했으며, assigner의 Right to Omission 같은 상관항까지 evaluator 커버리지를 2개에서 8개 법적 위치로 확장했다.

- **Empirical Impact**: Isabelle/HOL에서 모든 공리를 기계적으로 검증했고, Vampire, E, Z3로 39개 문제 벤치마크에서 추론 가능성을 실증했다. 그 결과, remedy 집행 권위처럼 배포 환경에 따라 달라질 수 있던 모호성이 정책 그래프 수준에서 명시되며 데이터 소버린티(누가 제재를 선언/이행 요구할 수 있는지)를 표현·검증하는 기반이 강화된다.



### ZONOS2 Technical Repor (https://arxiv.org/abs/2606.24320)
Comments:
          15 pages, 7 figures, 7 tables. Technical report. Model weights, inference code, and the ZTTS1-Eval benchmark released under Apache 2.0. Code: this https URL ; weights: this https URL ; benchmark: this https URL

- **Prior Approaches**: 기존 TTS 연구는 자연스러움, 발화 제어, 음성 복제 충실도, 스트리밍 지연 중 일부만 강화하는 경향이 있어, 전반 성능의 균형이 깨지는 문제가 있었다. 또한 벤치마크는 Seed-TTS-Eval처럼 영어·중국 중심이거나 준비된(읽기) 음성에 치우쳐, prosody(운율) 재현과 제로샷 음성복제 평가를 충분히 분리하지 못했다.

- **Core Contribution**: ZONOS2 8B는 decoder-only transformer 기반 TTS에 Mixture of Experts(MoE)를 도입해 1.6B→8B(900M active) 스케일을 달성하면서 자연스러움·운율·voice cloning fidelity를 동시에 끌어올렸다고 제시한다. 더불어 데이터/학습 레시피를 확장하고, 생성 조건을 단순화해 voice cloning과 스트리밍 친화적 지연을 함께 개선하는 방향을 취한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 고프레임레이트 음향 토큰(RVQ 기반 오디오 코덱)을 텍스트와 안정적으로 결합하고, (2) 제로샷 음성복제를 위해 프롬프트 음성 임베딩이 ‘어떤 불필요 정보’까지 함께 새지 않게 만드는 것이다. 논문은 텍스트를 byte-level로 토큰화해 G2P(문자→발음) 오류 의존을 제거하고, ECAPA-TDNN 임베딩을 LDA로 투영해 화자 간 차별 정보만 남기며, speaker embedding 학습 시점에는 오디오 증강과 손실 마스킹/크롭 전략 및 2단계 annealing을 결합해 과적합(cheat)과 누출을 완화한다.

- **Empirical Impact**: 평가는 품질, 화자 유사도, WER뿐 아니라 새 TTS 벤치마크 ZTTS1-Eval(9개 read, 17개 in-the-wild 언어, prosody·생성 다양성 포함)로 수행되며, 기존 SOTA와 경쟁적인 성능과 streaming 지연 유지가 보고된다. 또한 Apache 2.0 라이선스로 모델 가중치와 예제 추론 코드를 공개해, open-source TTS와 음성복제 실사용 가능성을 높였다는 점에서 의미가 있다.



### Real-Time Interactive Music Generation via Data-Free Streaming Consistency Distillation (https://arxiv.org/abs/2606.24307)
- **Prior Approaches**: 기존 텍스트-투-뮤직 모델(Suno, Stable Audio, ACE-Step 계열)은 구조적 일관성을 갖춘 고품질 오디오를 만들지만, 기본적으로 한 번에 렌더링하는 one-shot 방식에 머물러 실시간으로 시간축 궤적을 수정하기 어렵습니다. 스트리밍 생성이나 가속 증류 연구가 있었어도, 이를 ‘연주 가능한 악기’처럼 지연을 줄이면서도 timbre·transients·리듬 안정성을 유지하는 방향으로 통합한 접근은 제한적이었습니다.

- **Core Contribution**: 이 논문은 생성 모델을 기다렸다가 재생하는 시스템이 아니라, 사용자의 입력에 즉시 반응하며 음악을 연속적으로 이어가는 playable instrument로 재구성하는 프레임워크를 제안합니다. 핵심은 data-free streaming consistency distillation: paired audio-latent 데이터 없이, prompt-only로 고정 teacher의 chunk-wise 타깃 궤적을 온라인 생성해 single-step 스트리밍 학생을 학습시키는 것입니다.

- **Technical Challenges**: 실시간 스트리밍에서는 단계 수를 극단적으로 줄일 때 timbral fidelity 저하, transient 흐림, 리듬 드리프트, 장기 코히런스 붕괴가 빠르게 누적될 수 있습니다. 저자들은 이를 해결하기 위해 latent, spectral(RFFT 기반 크기 스펙트럼), temporal-difference(인접 프레임 차분의 L1)로 구성된 music-aware consistency objective를 결합하고, chunk-wise cached streaming context(KV-cache)를 유지한 채 연속 autoregressive latent space에서 증류·생성하도록 설계했습니다.

- **Empirical Impact**: SongDescriber 기반 평가에서 full objective(잠재+스펙트럴+시간차)는 한 단계 스트리밍 모델의 CLAP(텍스트-오디오 정렬), PaSST-KLD(음악 태그 분포), OpenL3-FD(지각/음향 충실도)를 모두 개선하며 특히 장기 roll-out에서 우수함을 보였습니다. 또한 streaming 재구성 덕분에 startup latency 0.086s와 낮은 RTF를 달성했고, 사용자 조작성을 포함한 주관 평가에서도 offline 방식 대비 responsiveness·steerability·co-creation 점수가 크게 향상되었습니다.



### CALIBER: Calibrating Confidence Before and After Reasoning in Language Models (https://arxiv.org/abs/2606.24281)
- **Prior Approaches**: 기존 LLM 보정(calibration)은 주로 한 번의 confidence만 뽑아 토큰 확률, 샘플 간 일관성, 또는 verbalized confidence를 학습해 왔다. 다만 reasoning 모델은 생각(thinking trace) 전후 정보 상태가 달라지는데, 기존 방식은 보통 confidence 위치(생각 전/후)와 보정 타깃(프롬프트 성공률 vs 정답 정합성)을 단일 조합으로 고정해 ‘정보 상태-감독 신호’ 불일치가 생길 수 있다.

- **Core Contribution**: 이 논문은 reasoning 모델의 confidence가 state-dependent하다고 보고, 생각 전 confidence는 ‘프롬프트를 풀 확률’, 생각 후 confidence는 ‘실제 생성된 답이 맞을 확률’을 각각 추정해야 한다고 정리한다. CALIBER는 하나의 응답에 pre-와 post-thinking confidence를 모두 생성하고, 각 예측에 맞는 타깃(그룹 수준 vs 인스턴스 수준)을 정확히 매칭해 학습시킨다. 그 결과 두 점수는 같은 스칼라가 아니라 역할을 분화하며, thinking에 따른 불확실성 변화도 진단 신호로 남긴다.

- **Technical Challenges**: 핵심 난제는 ‘position–target alignment’를 분리해 실험적으로 검증하는 것인데, 선행법들은 위치/타깃/학습 알고리즘이 함께 달라 공정 비교가 어렵다. 저자들은 같은 학습·평가 프로토콜 하에서 prior-inspired 조합들을 controlled baseline으로 두고, CALIBER는 GRPO 스타일 RLVR에 포맷 보상과 함께 pre(그룹 타깃)·post(인스턴스 타깃) 보정 보상을 동시에 넣되 학습 붕괴를 막기 위해 pre 보상만 먼저 warmup하는 2단계 학습을 설계했다.

- **Empirical Impact**: BigMathDigits에서 7B 모델은 CALIBER가 단일 confidence 강baseline 대비 ECE를 52.5% 줄였고, Brier score와 AUROC도 최상급이며 정확도는 큰 폭으로 훼손되지 않았다(최대 2.1점 이내). 30B에서도 BigMathDigits ECE 최우수 성능을 보이면서 Brier와 AUROC가 경쟁 수준을 유지했고, OOD인 GPQA와 TriviaQA에서는 ECE와 Brier를 가장 잘 낮추며 SimpleQA에서도 준수한 성적을 냈다. 또한 ablation에서 alignment(정보 상태에 맞춘 타깃 매칭)이 특히 분포 이동에서 보정 개선에 가장 유익함을 보여, 실제 신뢰도 추정이 필요한 배포 시나리오에 의미가 크다.



### Pigeonholing: Bad prompts hurt models to collapse and make mistakes (https://arxiv.org/abs/2606.24267)
- **Prior Approaches**: 기존 in-context learning 연구는 일반적으로 성능 향상을 보여주지만, 부적절한 문맥이 들어오면 성능 저하와 모드 붕괴가 발생할 수 있다는 점을 충분히 다루지 못했다. 특히 사용자의 의도와 무관하게(잘못된 정리 정당화 유도, 버그가 있는 코드 수정 누락 등) 나쁜 context가 섞일 때의 실패 양상이 체계적으로 정리되지 않았다.

- **Core Contribution**: 이 논문은 이러한 현상을 'pigeonholing'으로 명명하고, 두 시나리오—(1) 사용자가 해결책을 제시하는 경우, (2) 대화 문맥에 어시스턴트의 이전(오답) 응답이 포함되는 경우—에서의 양상을 분석했다. 또한 pigeonholing 완화를 위해 RLVR with synthetic errors(합성 오류 기반 RLVR)를 제안해, 나쁜 문맥에서도 모델이 망가지는 문제를 줄이는 방향을 제시한다.

- **Technical Challenges**: pigeonholing의 핵심은 나쁜 문맥이 모델을 과거의 오류/선호로 강하게 수렴시켜, 대안 탐색이 사라지고 답이 반복·고정되는 데 있다. 연구자들은 10개 검증 가능 및 오픈엔드 태스크, 10개 서로 다른 모델에서 (오답 반복으로 38-40% 성능 하락, 텍스트/코딩에서 답의 좁은 집합 수렴, 논쟁 주제에서 입장 뒤집기 등) 구체적 모드 붕괴 패턴을 관찰하고, 합성 오류로 학습을 보강하는 RLVR 접근을 통해 이를 완화하려 했다.

- **Empirical Impact**: 실험 결과 pigeonholing은 대화 턴 수가 늘수록 거의 단조 증가하며(오류 반복이 1에서 5로 늘 때 추가 14%+ 성능 하락), 예시가 정답이라도 모드 붕괴가 발생할 수 있음을 보였다. 제안한 RLVR with synthetic errors는 bad context에서 vanilla RLVR 대비 43-60% 개선을 보여, 실전 대화형 AI의 문맥 오염 취약성을 완화하는 데 의미 있는 진전을 제공한다.



### Neural Network-Based Parametric Model Reduction for Predicting Turbulent Flow for Different Vehicle Geometries (https://arxiv.org/abs/2606.24265)
- **Prior Approaches**: 산업용 수치 시뮬레이션은 실험 조건마다 고정밀 계산을 반복해야 하지만, 계산 자원 제약이 정확도와 비용의 균형을 어렵게 만든다. 이를 줄이기 위해 물리 시스템의 상태를 저차원 부분공간으로 제한하는 model reduction이 발전했으며, 특히 neural network로 비선형 부분공간에 투영하는 방식이 활발히 연구돼 왔다.

- **Core Contribution**: 이 논문은 기존의 neural-network 기반 축약과 시간진화(time-evolution) 접근을 확장해, variational autoencoder(변분 오토인코더)를 결합함으로써 고Reynolds-number(고 레이놀즈수) 유동에서도 모델의 robustness(견고성)를 평가·강화한다. 여러 형상의 차량 바디에 대해 압축된 latent representation(잠재 표현)만으로 와류(vortex) 생성의 재구성 정확도를 공간·시간 스케일 전반에서 점검하고, 특히 차체 후미 주변 유동 거동에 초점을 둔다.

- **Technical Challenges**: 핵심 기술 난제는 고Re 유동처럼 비선형성과 다중 스케일 와류 구조가 강한 상황에서, 저차원 잠재공간이 충분히 정보를 보존하는지 검증하는 것이다. 연구진은 VAE의 압축 특성을 활용해 고해상도 흐름을 compact latent로 표현하고, 시간진화 기반 축약 모델이 후미 영역의 재구성 오차와 와류 생성 품질을 얼마나 유지하는지로 robustness를 정량적으로 살폈다.

- **Empirical Impact**: 여러 차량 형상과 고Re 조건에서 재구성 정확도를 비교함으로써, 단순 평균 성능이 아니라 와류 발생처럼 구조적으로 민감한 유동 특성에서의 견고성을 실증한다. 결과적으로 비용 대비 정밀도를 요구하는 차량 공력 시뮬레이션 및 고난도 유동 제어·설계 워크플로에서 model reduction의 실용 가능성을 한 단계 끌어올리는 데 의미가 있다.



### SURGELLM: Rethinking Multi-Task Evaluation through Task-Aware Feature Gating with Class-Balanced Normalization (https://arxiv.org/abs/2606.24259)
Comments:
          Proceedings of the 6th Workshop on Trustworthy NLP (TrustNLP 2026), ACL 2026, San Diego, California, USA. Available at this https URL

- **Prior Approaches**: 기존에는 태스크별로 fine-tuning된 인코더를 두거나, multi-task learning으로 공유 인코더를 쓰는 방식이 주류였다. 하지만 서로 다른 어휘·라벨공간·문체를 가진 이질적 태스크가 섞이면 feature 간섭이 커져 성능 저하가 쉽게 나타난다. 또한 handcrafted lexical feature를 주입하더라도 class-imbalance로 인해 gate가 관찰하는 feature 통계가 왜곡되는 문제는 충분히 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 SURGeLLM이라는 통합 transformer 프레임워크로 이질적 NLP 태스크에서 외부 어휘 지식을 attention에 직접 조건화하면서도 공유 백본을 유지한다. 핵심은 (1) 차원별 surgical feature gate로 유용한 어휘 지표만 [CLS]와 결합하고, (2) task-conditioned prefix tokens로 입력 전반에 feature 값을 주입하며, (3) Instance-Weighted Normalization(IWN)으로 class-imbalance가 feature 통계를 망가뜨리는 효과를 제거하는 것이다. gate의 효용이 surgical feature alignment와 연결됨을 이론적으로 보이고, feature가 무의미할 때는 identity로 붕괴(무해성)함도 함께 제시한다.

- **Technical Challenges**: 가장 큰 기술적 도전은 (a) fine-tuning 후에도 남는 stylometric surface signal을 어떻게 ‘안전하게’ 결합할지, (b) 특히 skewed 라벨 분포에서 feature standardization이 gate를 편향시키지 않게 할지, (c) attention이 외부 lexical knowledge에 조건화되도록 만들지에 있다. SURGeLLM은 ReLU+sigmoid 기반의 per-dimension gate로 modality 혼합을 유연하게 하고, [CLS] fusion 이후 LayerNorm으로 분포 이동을 안정화한다. 동시에 IWN은 학습 단계에서 클래스 균형 통계로 표준화를 수행해 테스트 시 label 없이도 class-prior 편향을 교정하며, gate와 prefix를 같은 역할로 대체하지 않고 보완적으로 작동하게 설계했다.

- **Empirical Impact**: 4개 태스크(SST-2, multi-hop retrieval, LLM-prompt attribution, authorship detection)에서 17,830개 예제와 11개 모델 변형을 3개 시드로 검증한 결과, IWN이 포함된 SURGeLLM-IWN-RoBERTa가 전체 macro-F1 0.940을 달성했다. 이는 가장 강한 non-IWN 대비 +0.036, authorship detection에서 +0.130의 큰 향상을 포함하며, random-vocabulary 제어 실험에서는 평균 F1이 -0.028로 떨어져 성능이 ‘파라미터 증대’가 아닌 lexical feature의 효과임을 확인했다. 저자들은 또한 코드·어휘(vocabularies)와 99.5% 수준으로 curated feature를 자동 복구하는 절차를 공개해, 다른 도메인으로의 확장성을 시사한다.



### Social Structure Matters in 3D Human-Human Interaction Generation (https://arxiv.org/abs/2606.24255)
- **Prior Approaches**: 기존 text-to-motion 기반 HHI(인간-인간 상호작용) 연구는 두 사람의 모션을 “그럴듯하게” 생성하는 데 집중해, 단일 인물 모션 생성기를 2인으로 확장하거나 모션 프리어를 조합하는 방식이 많았다. 하지만 이런 접근은 HHI의 핵심인 시간적 단계(phase progression)와 비대칭 역할(initiator/receiver 등), 두 사람의 상호 조율을 조직하는 ‘사회적 구조’를 명시적으로 중간표현으로 드러내지 못해, 접촉 타이밍이나 방향, 상호작용 기하가 어긋나는 문제가 남는다. 또한 LLM을 직접 모션 생성기로 쓰는 경우에도 연속적·물리적으로 타당한 3D 상호작용 실행 품질이 한계가 있다는 진단이 제시된다.

- **Core Contribution**: 이 논문은 HHI 생성 문제를 ‘사회적 구조(social structure) 모델링 및 그라운딩(grounding)’으로 재정의한다. 즉, 먼저 상호작용이 어떤 단계로 전개되는지(approach/contact/release/in-place)와 각 단계에서 두 배우의 partner-aware 역할이 무엇인지 추론하고, 그 구조를 연속적이고 물리적으로 그럴듯하며 파트너를 고려한 3D 모션으로 실현하도록 만든다. 이를 위해 “Think with LLM, Move with Motion Skill” 패러다임(LLM은 planner, 모션 모델은 executor)을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM이 언어를 통해 사회적 구조를 ‘생각’(phase 분해·역할 추론)할 수는 있어도, 이를 바로 동역학/기구학적으로 정확한 연속 3D 상호작용 ‘움직임’으로 ‘실행’하긴 어렵다는 점이다. 저자들은 이를 해결하기 위해 (1) LLM 기반 사회적 구조 플래닝으로 phase 레이블과 partner-aware 역할을 motion-aligned한 형태로 재구성하고, (2) Solo-to-Social(S2S) 실행기에서 phase-wise self-conditioning(이전 phase 모션을 앵커로 사용)과 ego-relative partner conditioning(파트너 최신 상태를 상대좌표로 주입), 그리고 LoRA로 pretrained solo 백본을 효율적으로 상호작용 실행기로 적응한다.

- **Empirical Impact**: InterHuman/InterX 같은 표준 HHI 벤치마크에서 기존 방법 대비 텍스트-모션 정렬, phase consistency, partner-aware coordination이 동시에 개선되었다. 정량적으로는 phase progression 측면에서 R-Precision과 사용자 평가의 Phase ranking이 가장 좋았고, 파트너 조율 측면에서는 평균 FID가 가장 낮아 상호작용 구성 품질을 유지하면서도 물리적 현실성이 향상됐다. 특히 social structure planning만으로는 부족하며 S2S 그라운딩이 필요함이 ablation에서 확인되며, 다양성(multimodality/diversity)은 유지한 채 성능 이득을 달성한 점이 의미 있다.



### AutoSpec: Safety Rule Evolution for LLM Agents via Inductive Logic Programming (https://arxiv.org/abs/2606.24245)
- **Prior Approaches**: 기존 LLM agent 안전 대응은 정적 규칙 기반과 신경 분류 기반으로 나뉜다. 규칙 기반은 해석성과 감사를 제공하지만 보수/완화의 균형이 어려워 false positive와 false negative가 동시에 발생하며, 에이전트가 바뀔 때마다 수작업으로 갱신해야 한다. 신경 분류는 적응성이 있지만 왜 위험으로 판정하는지 설명이 부족해 안전성-critical 환경에서 신뢰·감사 요구를 충족하기 어렵다.

- **Core Contribution**: AutoSpec은 배포된 전문가용 안전 규칙을 사용자 safe/unsafe 주석으로부터 자동으로 진화(evolve)시키는 프레임워크를 제안한다. counterexample-guided inductive synthesis(CEGIS)를 안전 규칙 편집에 적용하되, inductive logic programming(ILP)으로 어떤 술어(predicate)가 오판(FP/FN)을 가르는지 학습해 규칙을 인간이 읽고 검토 가능한 형태로 유지한다. 결과적으로 정적 규칙의 해석 가능성과, 데이터 기반 자동 보정의 적응성을 함께 노린다.

- **Technical Challenges**: 핵심 난제는 규칙 편집 후보의 탐색공간이 술어 수에 따라 지수적으로 커져 CEGIS를 그대로 돌리면 비효율적이라는 점이다. AutoSpec은 FN에서 자주 나타나고 FP에서는 드물게 나타나는 판별 술어를 ILP가 빠르게 찾아 우선순위를 매김으로써 후보 생성을 급격히 축소한다. 이후 반례(FP/FN) 중심으로 AddException/Relax/AddDisjunct 같은 제한된 편집 연산을 생성·검증해 F1을 우선으로 개선되는 후보만 채택하며 반복하며 수렴시킨다.

- **Empirical Impact**: 291개 실행 trace(code execution, embodied agent)에서 AutoSpec은 rule F1을 각각 0.98과 0.93까지 끌어올렸고, 최대 94%의 false positive 감소를 유지한 채 recall을 보전했다. 또한 ILP 기반이 휴리스틱 CEGIS 대비 최대 4.8배 더 높은 F1을 달성했으며 4~5 iteration 내에 수렴한다. 학습된 규칙은 사람이 읽고 감사할 수 있고, 보지 못한 시나리오에도 일반화되어 안전 가드레일의 지속 유지보수에 의미 있는 실증 효과를 보였다.



### Inclusive Interactive Collisions for Multi-View Consistent Compositional 3D Generation (https://arxiv.org/abs/2606.24206)
- **Prior Approaches**: 기존 3D 생성은 대체로 학습 기반(단일 입력을 바로 3D로 복원)과 최적화 기반(Score Distillation Sampling, SDS로 2D diffusion 사전지식을 증류)으로 나뉜다. 하지만 학습 기반은 다중 객체·복잡 상호작용 데이터 부족으로, 최적화 기반은 2D diffusion의 제어 한계와 SDS의 뷰별 최적화로 인해 cross-view 불일치(Janus) 문제가 두드러진다.

- **Core Contribution**: 이 논문은 다중 객체가 상호작용하는 compositional 3D를 만들면서도 멀티뷰 일관성을 유지하는 최적화 기반 프레임워크 I2C-3D를 제안한다. 핵심은 Inclusive Interactive Collisions(I2C)로 Gaussian primitives의 배치를 계획된 바운딩박스 내부로 강제하되 상호작용 영역에서는 자연스러운 충돌이 일어나도록 유도하는 것이다. 여기에 Multi-View Adaptive Score Distillation Sampling(MV-ASDS)을 더해, viewpoint 전반에서 prior와 레이아웃 정보를 주의(attention) 조절로 함께 증류해 뷰 간 환각을 줄인다.

- **Technical Challenges**: 다중 객체 compositionality를 생성하려면 (1) 객체 배치와 상호작용 영역의 형상/충돌이 동시에 자연스러워야 하고 (2) SDS가 뷰별로만 학습 신호를 주는 구조에서 멀티뷰 일관성이 유지돼야 한다. 논문은 Gaussian primitives의 상호작용 분포가 연결선 중점 주변에 밀집한다는 관찰을 바탕으로 상호작용 collision 제약을 설계하고, K-means로 객체별 primitives를 클러스터링해 in-box 제약과 interaction collision 제약을 함께 최적화한다. 또한 MV-ASDS에서 instance token과 spatial token의 attention map을 멀티뷰에 맞춰 적응적으로 변조해 장면 수준과 개체 수준 모두의 일관성을 강화한다.

- **Empirical Impact**: 실험은 기존 SOTA 대비 semantic alignment, spatial arrangement, geometric consistency, scene quality 전반에서 우수함을 정량·정성으로 보여준다. 특히 상호작용 영역과 멀티뷰 일관성 측면에서 경쟁 방법들이 충돌 영역이 애매하거나 Janus 문제를 겪는 반면, I2C-3D는 상호작용이 더 그럴듯하고 뷰별 텍스처가 유지되는 경향을 보인다. 추가로 progressive 3D editing(객체를 단계적으로 삽입)까지 지원하며, ablation에서도 MV-ASDS/I2C/ASDS를 제거하면 멀티뷰 불일치나 텍스처 소실이 뚜렷해 효과가 확인된다.



### MMed-Bench-IR: A Heterogeneous Benchmark for Multilingual Medical Information Retrieva (https://arxiv.org/abs/2606.24200)
Comments:
          Under review. 15 pages, 3 figures

- **Prior Approaches**: 기존 연구는 다국어 검색과 의료 특화 검색을 각각 따로 다루거나, 둘의 교차 지점을 부분적으로만 측정했다. 예를 들어 의료 biomedical encoders는 영어 위주로 평가됐고, 다국어 벤치마크는 의료 개념(ontology·UMLS) 근거가 약해 실제 임상 지식 매칭의 품질을 검증하기 어려웠다. 또한 관련 벤치마크들은 정렬(alignment), 개념 구분(discrimination), 근거 검색(evidence retrieval)을 ‘동시에’ 보지 않아 한 축의 개선이 다른 축을 해치지 않는지 확인이 불가능했다.

- **Core Contribution**: 논문은 다국어 의료 검색의 세 가지 역량을 한 프레임에서 분해·동시에 평가하는 MMed-Bench-IR을 제안한다. UMLS에 근거한 cross-lingual medical QA retrieval, UMLS 기반 confusion set으로 개념 구분을 측정하는 discrimination, 그리고 RAG용 multilingual evidence retrieval까지 총 3개 태스크를 6개 언어(영·스·프·일·중·러)·3개 문자 체계로 구성했다. 특히 세 태스크는 설계상 query와 concept 중복이 없어, 특정 기술만 잘하는 ‘편향된 합산’이 아니라 실제 capability breadth를 반영한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 언어 간 개념 정렬, (2) 임상적으로 헷갈리는 개념의 미세 구분, (3) 영어 중심 코퍼스에서 다국어 질의 근거를 찾아내는 evidencing을 동시에 측정하는 평가 설계에 있었다. 이를 위해 UMLS CUI 태깅으로 양성군을 구성하고, discrimination의 난이도는 ontology 구조(동의어-유사하지만 다른 개념-더 먼 관련)로 고정했으며, RAG 태스크는 번역 품질을 concept fidelity와 back-translation 일치도로 필터링해 83.7%를 유지했다. 또한 UMLS 기반 tier 정의가 특정 검증자에 치우치지 않는지(leave-one-validator-out) 안정성을 점검하고, 태스크 간 어휘·질의 중복을 극도로 낮춰 ‘평가 눈속임’을 줄였다.

- **Empirical Impact**: 10개 시스템(lexical·biomedical·multilingual dense·late-interaction·hybrid·reranker 계열)을 평가한 결과, 영어에서 잘하는 모델이 일본어 등 비(非)영문 스크립트에서 급격히 붕괴하는 현상이 뚜렷하게 드러났다. 예컨대 biomedical encoder SapBERT는 영어 nDCG@10이 0.818에서 일본어 0.056으로 급락했고, 공정성 격차(fairness gap)는 0.76으로 보고됐다(영어-only 벤치마크로는 포착 불가). 한편 concept discrimination이 가장 어려운 병목으로 나타났고, cross-encoder reranking 및 도메인·언어 특화 임베딩(MMed-Embed) 조합이 최상위 성능(0.377)과 상대적으로 낮은 공정성 격차(0.170)를 달성해, 다국어 임상 동등성 관점의 검색 개발 방향을 제시한다.



### Co-occurring associated retained concepts in Diffusion Unlearning (https://arxiv.org/abs/2606.24192)
Comments:
          Accepted as a poster at ICLR 2026. Code available at this https URL

- **Prior Approaches**: 확산 모델의 유해 콘텐츠 생성을 줄이기 위한 unlearning 기법이 주목받고 있다. 다만 기존 방식은 목표 개념을 지우는 과정에서 함께 등장하는 정상 개념까지 덩달아 약화시키는 경우가 많다.

- **Core Contribution**: 논문은 함께 동반되지만 보존되어야 할 정상 공존 개념을 CARE(Co-occurring Associated REtained concepts)로 정의하고, 이를 직접 측정하는 CARE score를 제안한다. 또한 목표 개념만 지우면서 CARE를 안정적으로 보호하는 ReCARE(Robust erasure for CARE) 프레임워크를 제시한다.

- **Technical Challenges**: 핵심 난제는 unlearning 중에 발생하는 ‘불필요한 동반 억제’를 정량화하고 학습에 반영해 안정적으로 상쇄하는 것이다. ReCARE는 타깃 이미지에서 추출한 benign co-occurring 토큰으로 CARE-set을 자동 구성하고, 학습 과정에서 이 어휘를 활용해 목표 개념 erasure는 강화하되 CARE 보존은 유지하도록 설계했다.

- **Empirical Impact**: Nudity, Van Gogh style, Tench object 등 다양한 타깃에 대해 실험을 수행했으며, robust concept erasure와 전체 유틸리티, CARE preservation의 균형에서 전반적으로 state-of-the-art 성능을 보였다고 보고한다. 결과적으로 unlearning의 부작용을 줄이면서도 원하는 기능만 선택적으로 제거하는 방향에 실증적 기여를 한다.



### Deep Learning Approaches for 3D Medical Scene Completion: From Geometric Modeling to Generative Paradigms (https://arxiv.org/abs/2606.24180)
- **Prior Approaches**: 3D scene completion은 RGB-D, LiDAR, multi-view 등에서 얻는 부분 관측을 바탕으로 누락 기하를 복원하는 문제로, 기존 접근은 주로 voxel 격자(예: SSCNet, ScanComplete)와 point cloud(예: PCN, TopNet) 또는 implicit neural representations(예: Occupancy Networks, DeepSDF)로 나뉘었다. 그러나 voxel은 메모리·연산이 해상도에 따라 급격히 증가하고, point cloud는 불연속성과 표면 복원 후처리 부담이 생기며, implicit은 표면 추출을 위한 대량 쿼리로 실시간 제약이 커졌다. 최근에는 transformer, diffusion 같은 생성 기반이 성능을 끌어올렸지만 계산 비용과 조건부 정합(보이는 기하를 얼마나 정확히 존중하는지)의 어려움이 남아 있었다.

- **Core Contribution**: 이 논문은 2016–2026년(최근 10년) 동안의 3D scene completion 연구를 체계적으로 정리하고, representation(예: voxels, point learning, implicit neural fields, transformer, diffusion, rendering-aware 3D Gaussian primitives) 관점의 진화 흐름을 분석한다. 또한 수많은 논문을 포함·배제 기준에 따라 선별(초기 1,847편 중 최종 267편)하고 PRISMA 가이드라인을 따라 선행연구의 커버리지를 투명하게 제시한다. 마지막으로 분야의 기여를 분류하는 taxonomy와 함께, 남은 과제와 차세대 시스템 개발을 위한 research agenda를 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) 입력의 불완전성(가림/잡음/희소성) 속에서 (2) 표현력·연산량·스케일의 균형을 맞추고 (3) 학습/평가가 실제 배치 성능을 충분히 반영하는지 검증하는 것이다. 논문은 이 난제를 representation 패러다임의 트레이드오프로 해석하며, voxel의 O(N^3) 비용 문제를 sparse convolution으로 완화하고, implicit의 대량 쿼리 문제는 효율화 필요성으로 연결한다. 나아가 diffusion의 다단계 denoising(수십~수천 스텝) 비용과 hallucination/조건부 정합 이슈, transformer의 O(N^2) 병목 같은 실무적 제약까지 함께 정리해 해결 방향을 구조화한다.

- **Empirical Impact**: 저자들은 해마다 선택된 연구 분포와 패러다임 전환(특히 최근 diffusion 및 Gaussian 기반으로의 이동, 2024 이후 voxel 감소)을 통해 연구 속도와 트렌드 변화를 보여준다. 동시에 메모리 효율(예: sparse/point), 생성 다양성(예: diffusion), 실시간 렌더링 가능성(예: 3D Gaussian splatting)처럼 서로 다른 강점이 어떻게 empirical 성과로 이어졌는지를 taxonomy 안에서 비교 가능하게 만든다. 결과적으로 이 리뷰는 3D scene completion 연구자들이 다음 연구 방향(표현-추론-배치의 연결, 평가 지표/실세계 검증, 실시간성 확보)을 더 빠르게 설계하도록 하는 지도 역할을 한다.



### Zero-Shot Test-Time Canonicalization using Out-of-Distribution Scoring (https://arxiv.org/abs/2606.24178)
- **Prior Approaches**: 회전·스케일·시어처럼 클래스는 유지하지만 기하학만 바뀌는 입력에서, 기존 pretrained 비전 모델은 쉽게 오분류되며 이를 보완하려면 아키텍처에 불변/등변성을 넣거나 데이터 증강으로 재학습하는 경우가 많았다. Canonicalization도 있었지만, 대부분 logit 기반 energy 점수와 전용(비용 큰) 탐색 절차에 의존해 점수 함수/최적화기 설계 공간이 제한됐고, 이미 정렬된 입력에선 ID 정확도가 떨어지는 문제가 남았다.

- **Core Contribution**: 이 논문은 canonicalization을 OOD detection(분포 밖 감지)으로 재구성해, 어떤 OOD score든 “변환 중 에너지를 최소화”하는 canonicalizer의 energy로 쓸 수 있도록 만든다. 또한 탐색 알고리즘 선택이 성능을 크게 좌우함을 체계적으로 실험하고, 이미 정렬된 입력에 불필요한 변환을 적용해 생기는 정확도 저하를 gated canonicalization으로 줄인다.

- **Technical Challenges**: 핵심 과제는 (1) OOD score를 에너지로 쓸 때 ‘정답 canonical form’과의 일치가 실제로 성립하는지, (2) 연속 affine 변환군에서 비볼록 최적화를 어떻게 안정적으로 수행할지, (3) 최적화가 잘못된 변환을 선택해 ID 정확도를 떨어뜨리지 않게 제어하는 것이다. 저자들은 약 20종 OOD score와 9종 탐색 알고리즘을 대규모로 비교하고, 대체로 distance 기반 score에 random search + local refinement가 강하며, OOD 임계값 기반 selection gate와 score 감소를 강제하는 acceptance gate로 필요할 때만 변환을 수행하게 했다.

- **Empirical Impact**: 다양한 벤치마크(MNIST/EMNIST, TU Berlin 스케치, ModelNet10 3D point cloud, SI-Score 회전, 그리고 ImageNet 회전 셋업)에서 OODC는 training-free 기준 중 변환 입력 정확도를 가장 높였고, logit 기반 energy는 성능 개선이 미미한 경우가 많았다. 또한 gated 메커니즘으로 already-aligned 입력에서의 정확도 하락을 대부분 복구하면서도 변환 강건성은 유지해, 기존 canonicalization의 ‘강건성-정확도 트레이드오프’를 조절 가능하게 만들었다.



### Agon: An Autonomous Large-Scale Omnidisciplinary Research System Built on Prompt Economy (https://arxiv.org/abs/2606.24177)
- **Prior Approaches**: 기존 자율연구·멀티에이전트 연구는 아이디어 생성부터 실험 실행, 원고 작성까지를 한 번에 처리하는 방식이 많아 ‘산출물 생산’에 공수가 쏠리는 문제가 있었다. 또한 자동화된 판단을 위해 프롬프트/코드/스키마를 단계마다 늘리는 경향이 있어 유지보수 비용과 도메인 전이 장벽이 커졌다. 특히 LLM 기반 리뷰는 점수는 잘 오르지만 인간이 받아들이는 과학적 서술·주장 정합성을 벗어나는 실패가 발생했다.

- **Core Contribution**: Agon은 연구 워크플로 안에서 기계가 ‘검증 가능한 것’은 자동으로 확인하고, 그 외의 판단은 사람 과학자가 유지하도록 연구 오케스트레이터를 설계했다. 아이디어-제안-실험-논문을 아티팩트(artifact) 경계로 쪼개고, producer–critic 루프를 반복 재사용하는 ‘연구 팩토리’ 구조가 핵심이다. 이를 통해 새로운 도메인으로 옮길 때도 핵심 아키텍처는 유지하고 입력 컨텍스트만 바꾸는 전이를 지향한다.

- **Technical Challenges**: 대규모 병렬 루프(Massive Parallelism)를 돌리면 어떤 스레드를 어느 순서로 전진시킬지 결정하는 디스패치 부담이 커진다. Agon은 Zero-Code Orchestration으로, 상태를 사람이 읽을 수 있는 컨텍스트로 해석한 뒤 ‘프롬프트 기반 디스패치’로 다음 핸드오프를 선택해 코드·파서·유한상태기계가 비대해지는 문제를 줄였다. 또 논문 개선에서는 LLM 리뷰 점수만 최적화해 인간 친화성을 잃는 ‘종이 괴물’ 실패를 막기 위해, 감사(auditor)로 인간이 읽는 과학적 글쓰기 제약으로 투영(projection)하며 killer reviewer와 area-chair adjudicator로 공격 논리를 통과시킨다.

- **Empirical Impact**: Agon은 10개 이상의 과학 도메인에서 444회 Prompt Economy 루프를 포함해 수천 번의 scientist-coder-auditor 반복을 수행했으며, 사람의 실험 코드 작성 없이도 운영 가능함을 보였다. 동시에 자동화 루프가 포착·수정 가능한 실패와 인간 판단이 필요한 실패를 경계짓고, 이를 severity·fixability·visibility·capability locus 기준으로 분류해 새로운 실패 유형을 드러냈다. 결과적으로 ‘machine scales, human steers’ 패러다임을 실제 배치로 뒷받침하며, 자율연구 인프라의 산업화와 공개(open-source) 배포 가능성을 강화한다.



### Lightweight Transformer Models for On-Device Fault Detection: A Benchmark Study on Resource-Constrained Deploymen (https://arxiv.org/abs/2606.24173)
Comments:
          5 pages, 3 figures

- **Prior Approaches**: 기존 on-device fault detection은 클라우드 의존을 줄이기 위해 다양한 ML 분류기를 사용해 왔고, 대표적으로 Random Forest, XGBoost, SVM, Logistic Regression 같은 전통적 방법이 널리 활용됐다. 한편 transformer는 높은 정확도로 주목받았지만, 자원 제약 환경에서 모델 크기·지연을 감안한 체계적 비교/검증은 부족했다. 또한 SECOM, UCI-PM처럼 극단적 클래스 불균형에서는 기존 방식 전반이 한계에 부딪히는 문제가 반복됐다.

- **Core Contribution**: 이 논문은 NASA C-MAPSS, SECOM, UCI AI4I 2020 3개 공개 데이터셋에서 전통적 ML과 lightweight transformer(DistilBERT, TinyBERT-6L/4L, MobileBERT)를 “정직한 벤치마크”로 비교한다. F1/AUC뿐 아니라 모델 크기와 CPU 추론 지연을 함께 보고, INT8 dynamic quantization 및 two-stage adaptive inference 파이프라인까지 포함해 엣지 배치 관점의 성능-효율 트레이드오프를 정리한다.

- **Technical Challenges**: transformer를 탭ular 센서 fault detection에 적용하려면 데이터 표현이 핵심인데, 논문은 피처를 이름-값 쌍 문자열로 직렬화해 text로 변환한 뒤 pretrained 표현을 활용하는 방식을 쓴다. 또 자원 제약을 만족시키기 위해 INT8 dynamic quantization을 적용하고, confidence 기반으로 TinyBERT-4L INT8 triage와 DistilBERT expert를 라우팅하는 adaptive two-stage 전략으로 평균 지연을 줄인다. 다만 MobileBERT는 이 입력 표현에서 0% F1로 학습 실패/다수 클래스 편향이 관찰되어, NLP용 아키텍처 혁신이 탭ular 도메인에 바로 전이되지 않을 수 있음을 보여준다.

- **Empirical Impact**: C-MAPSS에서는 TinyBERT-6L/4L이 XGBoost와 비슷한 F1(최대 약 87.8~87.9%)을 보였지만, 모델 크기와 지연은 크게 불리해 전통 ML이 “정확도 대비 비용”에서 우위를 보였다. TinyBERT-4L(55MB, CPU 18ms)은 transformer 중 배치 친화성이 가장 높았고, INT8 quantization으로 25% 크기 감소(41MB)와 함께 F1 86.9%를 유지했다. adaptive 파이프라인은 97.9% 예측을 경량 triage로 처리해 평균 19.5ms에 F1 87.6%를 달성했으며, SECOM·UCI-PM에서는 모든 방법이 극단적 불균형의 벽을 넘지 못해(특히 F1이 크게 저하) 아키텍처보다 클래스 불균형이 핵심 병목임을 실증적으로 시사한다. 코드는 공개되어 재현 가능한 비교를 지원한다.



### A Pāninian Foundation for Indic Language Processing (https://arxiv.org/abs/2606.24172)
Comments:
          16 pages, 0 figures

- **Prior Approaches**: 기존 NLP는 인도계 언어를 계통(가계)별 또는 소수 언어군 단위로 쪼개 분리된 분석기·파서·데이터셋을 구축해 왔다. 그 결과 언어 간 전이가 가능한 경우에도 벤치마크와 주석 체계가 달라 이동성이 떨어지고, 중복 공학과 자원 격차가 누적된다.
또한 다국어 대규모 모델은 번역 품질 등 ‘폭’ 중심 지표에 강하지만, 인도계 언어를 관통하는 형태·통사·의미 구조를 명시적으로 검증하는 시험 설계는 부족하다.

- **Core Contribution**: 이 논문은 인도계 언어들이 공통으로 공유하는 형태통사적 ‘파니니(Pāṇinian) 문법’ 기반 아키텍처(Aṣṭādhyāyī, Astādhyāyī)를 계산적으로 단일한 기반으로 제안한다. 언어 간 전이가 이미 Pāṇinian의 구조적 레일 위에서 일어나고 있지만, 현재 시스템은 표면 신호에만 의존해 그 깊이를 활용하지 못한다고 본다.
이에 따라 파니니 범주를 기준선으로 삼는 4-part 벤치마크 스위트를 제안하며, 이를 통해 정확도·데이터 효율·전이성을 함께 끌어올리고 희소한 인도계 자원을 ‘하나의 고자원 메타언어’처럼 묶는 것을 목표로 한다.

- **Technical Challenges**: 핵심 난점은 인도계 언어가 형태적으로 매우 풍부해 단어가 뿌리와 접사로 조합되며 격·수·시제·상·태 같은 정보가 복합적으로 얽힌다는 점이다. 특히 sandhi(음운 결합)와 samāsa(복합어) 같은 현상은 문장을 의미 단위로 분해하는 단계부터 어려워, 이를 파니니적 구조로 중심화하지 않으면 대부분의 파이프라인이 모서리 케이스로 취급하게 된다.
또한 문어·구어가 함께 존재하는 diglossia와 코드믹싱, 그리고 의미가 명사-고정 관점이 아니라 사건 구조에서 동사 뿌리를 중심으로 ‘과정적’으로 조직된다는 의미론적 차이가 있어, 기존 영어 기반 평가 프레임이 구조를 제대로 측정하지 못한다.

- **Empirical Impact**: 저자들은 파니니 원리에 기반한 형태 분석이 인도계 언어 간 전이와 다운스트림 성능을 실제로 개선해 왔다는 선행 결과들을 근거로 들며, 공통 아키텍처를 명시화하면 추가 이득이 클 것이라고 주장한다. 나아가 현재의 벤치마크가 대부분 표면 신호(어휘/문자 체계) 중심으로 작동해 문법적 이해를 드러내지 못한다는 문제를 지적한다.
제안된 파니니-근거 벤치마크는 인도계 시스템을 더 데이터 효율적으로 만들고 언어 간 전이를 ‘설계된 메커니즘’으로 전환하는 한편, 신경망이 실제로 파니니 범주를 자발적으로 표상하는지(해석 가능성)까지 실험 가능하게 할 의미가 있다.



### Breaking Shortcut Learning for Cross-Trial EEG-Guided Target Speech Extraction via Two-Stage Training (https://arxiv.org/abs/2606.24164)
Comments:
          Accepted by Interspeech 2026

- **Prior Approaches**: 기존 EEG-guided target speech extraction(TSE) 연구들은 end-to-end 방식으로 EEG와 오디오 믹스를 함께 학습해, within-trial 평가에서 SI-SDR 10dB+와 attended-source 정확도 90%대 성능을 보고해 왔다. 하지만 본 논문 분석에 따르면, within-trial에서의 높은 점수는 attention 관련 신호보다 trial-specific EEG 구조가 만든 shortcut에 의해 부풀려질 수 있으며, strict cross-trial에서는 성능이 크게 붕괴한다.

- **Core Contribution**: 논문은 trial-robust하게 동작하지 못하는 end-to-end TSE의 핵심 원인을 “trial identity를 매개로 한 shortcut learning”으로 지목하고, 이를 완화하는 두 단계 프레임워크 TRUST-TSE를 제안한다. TRUST-TSE는 Stage 1에서 contrastive pretraining으로 EEG 인코더가 trial-정체성 단서 대신 EEG–speech 정렬에 민감하도록 만들고, Stage 2에서 confidence-weighted SI-SDR 목적함수로 추출기를 안정적으로 학습한다.

- **Technical Challenges**: 두 단계로 분해하더라도 Stage 2에서는 고정된 EEG 임베딩이 구간별로 신뢰도가 달라(명확한 attended 정렬/모호함/ignored 쪽 호환 등) 학습이 흔들릴 수 있다. 이를 해결하기 위해 논문은 EEG–source similarity 기반의 confidence weight로 긍정/부정 구간을 모두 반영하는 confidence-weighted SI-SDR을 설계해, ambiguous 구간에서의 잡음성 그라디언트와 과도한 조건기피를 동시에 줄이려 한다.

- **Empirical Impact**: KUL과 DTU 데이터에서 strict cross-trial 프로토콜을 적용했을 때 TRUST-TSE는 end-to-end baseline보다 일관되게 더 좋은 성능을 보이며, 특히 unseen trial에서의 reliability 병목을 완화한다. 즉, 단순 평균 수치 개선을 넘어 “평가 프로토콜이 바뀌면 무너지는 문제”를 겨냥해 일반화 신뢰도를 높였다는 점에서 분야의 실사용 관점에 의미가 크다.



### Metis: Bridging Text and Code Memory for Self-Evolving Agents (https://arxiv.org/abs/2606.24151)
Comments:
          Work in progress

- **Prior Approaches**: 자기진화 에이전트는 이전 실행 경험을 메모리에 저장해 이후 태스크에 재사용한다. 그런데 경험을 자연어 텍스트로 넣을지, 호출 가능한 코드 툴로 둘지는 보통 설계 시점에 고정돼 왔고, 두 표현이 어떤 종류의 경험에 어떤 비용-효율-전이 특성을 보이는지 체계적으로 비교·이해되지 않았다. 그 결과 텍스트 메모리의 견고함과 코드 메모리의 실행 효율 간의 트레이드오프를 데이터 기반으로 조합하는 전략이 부족했다.

- **Core Contribution**: 이 논문은 동일한 경험 집합과 동일한 에이전트 백본 위에서 텍스트 메모리와 코드 메모리를 분리해 평가한 ‘첫 controlled study’를 제시한다. 그 관찰을 바탕으로 Metis는 계층형 dual-representation memory를 설계해, 텍스트(계획/사실/함정)를 기반으로 하되 반복되는 실행 계획만 선별적으로 callable tools로 crystallize한다. facts와 pitfalls은 텍스트로 남겨 추론 가이드는 유지하면서 코드의 비용·취약성을 불필요하게 확산하지 않도록 한다.

- **Technical Challenges**: 핵심 난제는 (1) 코드 툴 생성이 비싸고 검증·디버깅 비용이 크며 (2) 코드는 고정된 실행 동작이라 태스크 변형에 취약하다는 점을 어떻게 제어하느냐에 있다. Metis는 실행 후 반영(reflect)에서 계획/사실/함정의 역할을 분리해 저장하고, 코드 승격(promotion)은 ‘얼마나 자주/안정적으로 선택됐는지’의 recurrence 증거로만 트리거한다. 또한 툴 생성 시에는 sandbox 검증, dependency-closure, compilation check를 통과한 경우에만 라이브러리에 추가하고, 텍스트 반영은 실행 임계 경로에서 비동기로 수행해 지연을 최소화한다.

- **Empirical Impact**: AppWorld에서 Metis는 ReAct 대비 작업 정확도를 최대 20.6%까지 개선하면서 실행 비용은 최대 22.8%까지 줄였다고 보고한다. 비교하는 자기진화 계열 시스템들에 대해서도 정확도·실행 효율·메모리 구성 비용 간 균형을 더 잘 맞추는 경향이 일관되게 관찰된다. 즉, 경험 표현을 ‘무조건 텍스트’ 또는 ‘무조건 코드’로 고정하기보다, 경험 특성에 따라 선택적으로 전환하는 접근이 실용적 성능을 만든다는 점을 실험적으로 뒷받침한다.



### DTT-BSR+: A Generative-Regression Cascade for Music Source Restoration (https://arxiv.org/abs/2606.24127)
Comments:
          Accepted by Interspeech 2026

- **Prior Approaches**: 기존 Music source restoration(MSR) 연구는 MSS처럼 단순 합 모델을 넘어서 비선형 프로덕션 효과를 되돌려야 하지만, 이 결합 최적화가 어렵다. 최근 접근은 (1) 분리형 multi-stage(X-LANCE는 separation·dereverberation·denoising을 순차 수행) 또는 (2) GAN+재구성 손실을 함께 쓰는 joint 학습(DTT-BSR, Hachimi 등)으로 나뉜다. 그런데 여러 시스템에서 FAD는 나아도 MMSNR이 제한돼, 의미 분포는 맞지만 실제 파형 복원이 약하다는 문제가 반복됐다.

- **Core Contribution**: 논문은 DTT-BSR+를 제안하며 MSR을 두 단계로 분해한다: 1단계는 clean 소스의 semantic distribution(prior)을 맞추는 “분포 피팅”을, 2단계는 파형 수준 “신호 재구성”을 담당한다. 1단계에서 DTT-BSR 생성형 separator가 줄기를 뽑아 clean stem의 prior에 맞추고, 2단계에서 수정 Demucs-L이 1단계 출력을 입력으로 받아 time-domain과 multi-resolution spectral loss로 waveform fidelity를 강화한다. 이 구조로 단일-stage에서 생기던 semantic consistency와 재구성 정확도의 충돌을 줄이려는 의도를 명확히 한다.

- **Technical Challenges**: 핵심 기술 난제는 non-linear production effects의 역문제를 안정적으로 처리하면서도, 1단계에서 형성된 의미 분포를 깨지 않고 2단계가 파형을 정밀하게 다듬어야 한다는 점이다. 이를 위해 2단계 Demucs-L에서는 표준 Demucs의 BLSTM bottleneck을 제거해 temporal receptive field를 국소화하고, L1(시간) + MR-STFT(다중 스케일 스펙트럼)로 재구성 손실을 균형 있게 강제했다. 또한 학습 시 주파수 영역 phase offset, 그리고 1단계 출력 대신 ground-truth를 확률적으로 대체하는 regularization을 적용해 2단계가 1단계 출력 분포에 과적합하지 않도록 했다.

- **Empirical Impact**: MSRBench(8개 stem, 3250초급 대규모 스테레오 클립)에서 DTT-BSR+는 단일-stage DTT-BSR 대비 모든 stem에서 MMSNR을 일관되게 개선하며, Vocals·Guitars·Synthesizers·Bass·Drums에서 X-LANCE를 앞섰다. 특히 Bass와 Drums에서 개선 폭이 커(예: Bass MMSNR이 크게 상승) 2단계가 실제 파형 복원을 효과적으로 끌어올림을 시사한다. 한편 FAD-CLAP 분해 결과, 일부 stem에서는 MMSNR 향상이 FAD 증가로도 이어지는데 이는 분포의 covariance보다 semantic mean shift가 주로 커지기 때문이며, 이 trade-off가 stem-specific하게 다르게 나타남을 보여준다. 또한 FAD-CLAP과 Zimtohrli 모두에서 전반적 품질 개선이 관측돼, 제안 구조가 재구성 정확성과 지각 품질을 동시에 끌어올릴 잠재력이 있음을 확인했다.



### A Benchmark for Hallucination Detection in VLMs for Gastrointestinal Endoscopy (https://arxiv.org/abs/2606.24115)
Comments:
          Accepted at the Medical Image Understanding and Analysis (MIUA) 2026 conference

- **Prior Approaches**: 기존 의료 VLM 환각 탐지는 주로 방사선 데이터셋(MIMIC-CXR, VQA-RAD 등)에서 검증되어, 위장관 내시경(GI) 환경으로의 일반화가 불명확했다. 또한 방법들은 모델 내부 접근(white-box), 확률·엔트로피 등 토큰 신호(gray-box), 샘플 일관성 같은 출력 기반(black-box)으로 나뉘며, 접근성·성능·비용 간 트레이드오프가 커서 실제 배포 판단이 어려웠다. 특히 GI처럼 이미지 품질과 시각 패턴이 이질적인 도메인에서는 기존 성능이 유지되는지 확인이 부족했다.

- **Core Contribution**: 이 논문은 GI 내시경 VQA 벤치마크인 Gut-VLM 데이터셋에서 환각 탐지 9개 방법을 5개 VLM에 대해 동일한 실험 파이프라인으로 체계적으로 비교했다. radiology 중심의 선행 검증 결과가 GI로 잘 이전되지 않으며, 의료·비의료 전반 VLM에서 특히 black-box/gray-box가 거의 무작위 수준으로 붕괴할 수 있음을 보여준다. 동시에 white-box hidden-state 접근 기반 방법인 ReXTrust가 모든 VLM에서 가장 높은 AUC를 기록하며, 최강 비(非) white-box 대비 통계적으로 유의미한 향상을 달성했다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) GI 내시경 도메인 이동(domain shift) 하에서 환각 탐지 신호가 유지되는지, (2) 다양한 접근 수준(출력·토큰 확률·hidden state)에서 어떤 신호가 안정적으로 환각을 드러내는지였다. 저자들은 GREEN을 이용한 Gut-VLM 기반 라벨링으로 9개 방법을 동일한 입력·라벨 체계에서 공정 비교하고, gray-box의 토큰 확률/엔트로피와 white-box의 중간 레이어 hidden state를 통해 탐지 신호의 차이를 분석했다. 더 나아가 일관성 기반과 불확실성 기반을 동시에 무너뜨리는 failure mode인 confident confabulation을 찾아 그 구조적 원인을 제시했다.

- **Empirical Impact**: 실험 결과 ReXTrust는 5개 VLM 모두에서 최고 AUC를 보였고, 예시로 MedGemma-4B에서 AUC 93.0까지 도달했다. white-box는 평균적으로 약 19.5 AUC 포인트의 일관된 우위를 가지며, LLaVA-v1.6-7B처럼 black-box와 clustering 계열 gray-box가 near-chance로 붕괴하는 경우에도 강한 성능을 유지했다. 또한 black-box 중에서는 SelfCheckGPT-NLI가 가장 신뢰 가능한 대안으로 나타났지만, 짧고 자신감 있는 오답을 반복하는 모델(Lingshu-32B)에서는 confident confabulation 때문에 탐지가 실패할 수 있음이 확인됐다.



### DramaDirector: Geometry-Guided Short Drama Generation (https://arxiv.org/abs/2606.24107)
Comments:
          20 pages, 17 figures, 6 tables. Code is available at this https URL

- **Prior Approaches**: 기존 plot-to-video·narrative video 생성은 대개 스크립트/샷을 텍스트로 정교화하거나(SkyScript 등), 멀티에이전트로 플래닝을 반복하거나(MovieAgent, GENMAC), 시각 요소를 조건으로 합성해 일관성을 노리는 방식이 주류였다. 하지만 LLM이 만든 텍스트 중심 스토리보드는 short-drama의 촬영 문법(빠른 컷 리듬, 대사 주도의 프레이밍 전환)을 충분히 반영하기 어렵고, 컷을 규정하는 공간 기하(시선·가림·깊이 관계)를 거의 담지 못한다.

- **Core Contribution**: DramaDirector는 전역 plot과 국소 context를 ‘멀티샷 스토리보드’로 만들되, 각 샷을 static visual condition과 dynamic narrative condition으로 분리해 렌더링 단계에 직접 연결한다. 특히 실제 short-drama 샷 갤러리에서 depth·pose로 인덱싱된 기하 선행정보를 retrieval해, 텍스트가 전달하지 못한 공간 제약을 첫 프레임 생성과 image-to-video 합성에 주입한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) LLM 플래너가 short-drama 스키마(카메라 그라머·샷 전개 규칙)를 따르도록 만드는 것과 (2) 기하 정렬이 텍스트-비디오 정렬 보상으로부터 복구되지 않는다는 점이다. 논문은 schema-constrained SFT로 플래너를 단단히 초기화하고, depth·pose 기반 텍스트-시각 정렬 reward를 학습한 뒤 GRPO로 보상 최적화를 수행해 ‘기하가 실제 샷에 맞는’ 스토리보드를 생성하도록 폐루프(계획-검색-보상)를 구성한다.

- **Empirical Impact**: 또한 35편 실사 드라마에서 2.8K episodes·81K shots를 구축한 DramaBoard를 제안하고, 스토리보드 내러티브 품질부터 shot-level instruction following과 비디오 고유 품질까지 다차원 평가를 제공한다. 실험 결과 DramaDirector는 주요 베이스라인 대비 faithfulness·consistency·controllability에서 향상되며, ablation과 정렬 모델의 검색 판별 성능이 depth·pose 기하 설계의 기여를 뒷받침한다.



### The impact of generative artificial intelligence on academic development of Chinese students in humanities and social sciences (https://arxiv.org/abs/2606.24104)
- **Prior Approaches**: 기존 연구는 GenAI가 고등교육 학습에 미치는 영향을 다루더라도, 인문·사회계열(HSS) 학생을 대상으로 한 체계적·실증적 근거는 상대적으로 부족했다. 특히 HSS의 학습 성과가 글쓰기와 해석 중심으로 나타나는 점과 GenAI의 강점이 맞물리면서도, 실제 학업 발달로 이어지는 경로는 명확히 정리되지 않았다. 기존 논의는 주로 전반적 관찰이나 사례 중심에 머물러 평가 방식의 한계까지 충분히 검토하지 못한 경우가 많았다.

- **Core Contribution**: 이 논문은 중국의 대규모 HSS 학생 설문을 기반으로 GenAI 사용이 학업 발달 전반에 미치는 영향을 네 축(사용 패턴, 학습 과정 및 성과 효과, 사용의 도전 과제, 교육과정 통합 선호)으로 정리해 근거를 제공한다. 학생 인식을 바탕으로 동기·독립적 사고·창의성 향상에 대한 긍정과, 변화가 없거나 감소했다는 부정이 함께 존재함을 보여준다. 또한 성과 향상 인식이 크더라도 기존 평가 관행의 제약이 일부 영향을 줄 수 있음을 함께 지적한다.

- **Technical Challenges**: 핵심 과제는 HSS 맥락에서 GenAI 활용의 효과를 학습 과정(동기·사고·창의성)과 학업 성과(성적/평가)를 분리해 설문으로 측정하고, 개인별 사용 경험(경험 기간)·전공(학문 분야)·성별에 따른 차이를 해석하는 것이다. 이를 위해 학습이론을 토대로 사용 패턴과 효과, 우려(정확도·과의존) 및 윤리/프라이버시 인식을 구조화해 질문 설계를 구성했다. 더불어 교육과정 통합에 대한 선호(부분적/선택적 도입, 실습 기반 훈련)를 함께 수집해 실무적 권고로 연결했다.

- **Empirical Impact**: 결과적으로 절반 이상이 학습 동기와 독립적 사고·창의성의 향상을 체감했지만, 적지 않은 소수는 변화가 없거나 오히려 악화되었다고 응답했다. 학업 성과는 더 큰 비율이 개선을 보고했으나, 전통적 평가의 한계 가능성이 제기되며 해석에 주의가 필요하다는 메시지를 준다. 학생들은 정확도 문제와 과의존을 가장 큰 우려로 꼽았고, 윤리의 중요성은 압도적으로 동의했지만 프라이버시 보호 만족은 상대적으로 낮아 책임 있는 통합을 위한 구체적 가이드가 필요함을 시사한다. 결론적으로 학생 관점에 기반한 교육과정 통합 권고를 제시해 HSS GenAI 도입 논의의 실증적 기준점을 제공한다.



### Beyond Bayer: Task-Optimal Sensor Co-Design for Robust Autonomous-Driving Segmentation (https://arxiv.org/abs/2606.24096)
- **Prior Approaches**: 자율주행에서의 견고한 인지는 보통 더 큰 백본, foundation model, 그리고 다중 에이전트 협력 융합을 키우는 방식으로 성능을 끌어올려 왔다. 반면 카메라 자체가 무엇을 측정(센서 설계)해야 하는지는 상대적으로 상류 단계에서 체계적으로 다루지 못했다.

- **Core Contribution**: 이 논문은 카메라의 센서 측 자유도 중 dense prediction(예: 분할/세그멘테이션)에 실제로 유익한 요소가 무엇인지, differentiable RAW-to-task 파이프라인으로 분해해 제안한다. 특히 spectral colour-filter-array(CFA) 가중치 학습이 가장 큰 레버임을 밝혀, 고정 카메라 대비 mIoU를 각각 KITTI-360에서 +0.017, ACDC에서 +0.023 개선한다.

- **Technical Challenges**: 센서 설계를 학습하려면 RAW 입력을 작업까지 거치되 미분 가능해야 하고, 동시에 CFA/광학/잡음 같은 설계 자유도를 공정하게 비교할 수 있어야 한다. 연구진은 RAW-to-task를 미분 가능하게 구성해 센서 개입의 정보 흐름을 계량했고, data-processing inequality에 따라 point-spread-function(PSF, optics) co-design이 순손해가 되는 이유를 설명하며, noise 공동 최적화 효과는 미미하다고 본다.

- **Empirical Impact**: 결과는 모델 크기를 키우거나 협력 융합을 하더라도 다운스트림이 회복할 수 있는 태스크 정보에 상한이 있다는 점을 실증적으로 뒷받침한다. 또한 ACDC의 fog/night/rain/snow에서도 개선의 견고함을 확인하며, 결론적으로 2x2 CFA 가중치만 학습하고 PSF는 identity로 유지하는 간단한 레시피를 제시한다.



### Predicting Poets' Origins from Verse: A Computational Analysis of Regional Linguistic Fingerprints in the Complete Tang Poems (https://arxiv.org/abs/2606.24093)
- **Prior Approaches**: 당대 시문에서 지역적 전통(지역 학교, 로컬 문체)이 실제 언어 차이를 남기는지에 대해 문학사에서는 오랫동안 논쟁이 있었지만, 이를 체계적으로 정량화한 연구는 제한적이었다. 기존 연구는 주로 면밀한 독해에 의존하거나, 구어 방언 분석에서 쓰이던 방식이 역사 문학 코퍼스에 그대로 적용되기 어려운 문제(라벨 품질, 문체 변동 등)로 인해 진전이 더뎠다. 이에 따라 “지리적 기원=예측 가능한 언어 흔적”이라는 가설을 데이터 기반으로 검증하는 접근은 여전히 도전적 과제로 남아 있었다.

- **Core Contribution**: 본 연구는 당(唐) 시인들의 지리적 기원을 출신 행정구역(circuit, 道)과 연결해 “시집 전체 텍스트만으로 시인의 출신을 분류할 수 있는가”를 직접 실험한다. Complete Tang Poems의 시를 시인 단위로 집계하고, CBDB의 道 정보를 목표로 10개 회로(그리고 남/북 이분류)에서 multi-class classification을 수행해 예측 가능성을 정량화했다. 나아가 단순 정확도보다 ‘무엇이 신호를 담는가(이미지, 계절/시간, 인유 등)’와 ‘신호가 시대에 따라 어떻게 변하는가’를 함께 분석했다.

- **Technical Challenges**: 핵심 기술 과제는 역사 문학 코퍼스에서 라벨(지리/귀속) 잡음이 크고, 시인은 “한 편”이 아니라 “여러 편의 집합”으로 특징을 대표해야 한다는 점이었다. 연구진은 시인을 단일 샘플로 보고 character n-gram TF-IDF와 이미지·계절/시간·인유 밀도 같은 해석 가능한 도메인 특징을 함께 사용했으며, 불균형 데이터를 stratified cross-validation과 class weighting으로 보정했다. 또한 GuwenBERT를 단편(250자) 평균으로 학습하면 불리해지는 문제를 계층형 frozen-encoder(단편→시인 벡터)로 해결해 TF-IDF 기반 고전 모델과의 공정 비교를 가능하게 했다.

- **Empirical Impact**: 실험 결과 시인의 남/북(南/北) 기원은 0.69 정확도와 매크로-F1으로 다수 기준선(0.53)을 뚜렷이 상회했고, 더 세분된 道 수준에서도 우연을 넘어선 성능을 보였다. 언어-지리 거리는 회로 간 거리와 함께 증가(거리감쇠 효과; Mantel r=0.40, p≈0.09)하며, High Tang에서는 남/북 분리가 약하고 Late Tang에서 가장 강해 “궁정 중심 동질화→시대 후반 지역 분기” 패턴을 정량적으로 뒷받침한다. 특히 모델의 자신 있는 오분류가 초기 당의 남방 시인을 북방 궁정 문체로 읽는 형태로 나타나, 오류 자체가 역사적 해석 가설을 제공할 수 있음을 보여주었고(인간-모델 협업 관점), 또한 TF-IDF의 character n-gram이 이미 지역 신호를 충분히 포착해 BERT 결합이 추가 이득이 없다는 점을 시사한다.



### DynaWM: Dynamics-Aware Distillation with World Model and Momentum Targets for Smooth Locomotion over Continuous Stairs (https://arxiv.org/abs/2606.24089)
Comments:
          Comments: 8 pages, 7 figures, accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

- **Prior Approaches**: 기존 연구는 교사-학생 distillation(CTS)로 지형 높이/접촉 정보 같은 특권(privileged) 정보를 학습에만 쓰고, 배치 시에는 proprioception만으로도 지형을 “알아보게” 하는 접근을 주로 썼습니다. 하지만 교사 인코더가 보상 최적화(PPO 등) 중심으로 학습되면 dynamics-aware 표현이 약해져 즉시 보상과 무관한 기하 정보를 놓치기 쉽습니다. 또한 교사/학생이 동시 업데이트되면서 non-stationary 타깃이 발생해 학생 표현이 축소(dimensional collapse)되거나, 잠재공간이 블랙박스로 남아 지형 높이 모델링 신뢰성을 검증하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 논문은 연속 계단(traversal over continuous stairs)을 목표로, 지형의 forward dynamics를 의식하는 표현을 학습하는 DynaWM을 제안합니다. 세계 모델(world model)을 정규화 항으로 도입해 인코더가 미래 상태 예측에 필요한 지형 기하를 유지하도록 만들고, PCA 기반 평가로 계층적(계단 높이 중심) 인코딩을 가시화·검증합니다. 더불어 momentum target encoder를 넣어 distillation 타깃을 안정화해 표현 붕괴 위험을 줄입니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 보상 최적화만으로는 지형 기하의 dynamics를 충분히 담지 못하는 점, (2) 교사 표현이 빠르게 변할 때 학생이 비정상 타깃에 휘둘려 dimensional collapse가 생기는 점, (3) 인코딩이 실제로 “높이”를 얼마나 담는지 확인하기 어려운 점입니다. 논문은 세계 모델 예측 손실로 forward-dynamics aware 정규화를 걸고, BYOL에서 가져온 momentum target encoder(EMA)로 비정상성을 완화했으며, PC-Correlation·Linear Probing·CCA 같은 지표와 PCA 시각화로 해석 가능성도 확보했습니다.

- **Empirical Impact**: 시뮬레이션(IsaacGym)과 실제 하드웨어 배치에서 DynaWM은 CTS 및 여러 ablation 대비 연속 계단에서 더 높은 성공률과 더 매끈한 모션(에너지/가속 관련 지표 포함)을 보였습니다. 특히 step width/height 변형이 큰 조건에서도 성능이 유지되어, 계층적 지형 인코딩이 실제 제어의 smoothness와 적응성에 연결됨을 실증했습니다. 또한 PCA 기반 정성/정량 평가에서 지형 높이·마찰 정보가 주된 분산 축에 정렬되는 경향과 학생-교사 정렬(CCA)이 개선되어, “지형을 제대로 배웠다”는 근거를 함께 제시합니다.



### Blockwise Policy-Drift Gating for On-Policy Distillation (https://arxiv.org/abs/2606.24084)
Comments:
          8 pages

- **Prior Approaches**: 온폴리싱 증류(OPD)는 학생이 실제로 만든 롤아웃의 접두사에 대해 교사가 피드백을 주어, 정적 교사 트레이스만 쓰는 방식의 분포 불일치를 줄인다는 장점이 있다. 다만 장기 추론에서는 sampled-token OPD가 취약해지며, 이를 보완하기 위해 Teacher-TopK/LSM처럼 토큰 하나 대신 교사의 truncated support로 국소 신호를 넓히는 방법들이 제안됐다. 또 stale rollout이나 비동기 업데이트를 다루는 freshness-aware 계열 연구도 있으나, 본 논문이 겨냥한 것은 비동기 시스템이 아니라 롤아웃 재사용 시 나타나는 old-current 정책 드리프트라는 다른 불일치 원인이다.

- **Core Contribution**: 본 논문은 rollout reuse 상황에서 재사용된 샘플 경로 위의 old-current 학생 log-probability shift를 ‘플러그인’ 가중치로 OPD 손실에 반영하는 sampled-path policy-drift gating을 제안한다. teacher 타깃, teacher top-K support, 롤아웃 분포 자체는 바꾸지 않고, detach된 mean-normalized 게이트로 base OPD(및 LSM)의 위치별 손실 가중치만 조절한다. 특히 게이트를 토큰 단위가 아니라 블록/스팬 단위로 집계해 noisy한 신호를 완충하는 절충점을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 old-current shift가 토큰 수준에서는 신호가 흔들리는데, 이를 너무 거칠게(시퀀스 단위) 만들면 장기 추론에서 지역성이 사라진다는 점이다. 논문은 같은 샘플 경로에서 behavior student와 current student의 로그확률 차이를 계산하되, teacher 신뢰나 divergence 같은 추가 신호 없이도 쓸 수 있도록 detached soft gate로 설계했다. 또한 fixed-block(예: 64토큰) 또는 newline-delimited span으로 shift를 국소 집계해 해당 구간의 위치 손실에 동일 가중치를 브로드캐스트함으로써 중요도 가중치의 변동성을 줄였다.

- **Empirical Impact**: Qwen3 수학 벤치마크에서 200-step 고정 예산 조건 하에 pass@8을 주요 지표로 평가했으며, fixed 64-token block gating은 sampled-token OPD의 mean pass@8을 0.4978에서 0.5160으로 끌어올렸다(AIME24/AIME25/MATH500/AMC23 평균). 또한 Teacher-TopK/LSM과 결합 시 Block64가 학습된 변형들 중 4개 벤치마크 평균 pass@8에서 최상 성능을 보였다. 결과는 rollout 재사용 하의 local old-current policy drift가 실용적인 제어 신호가 될 수 있고, 그 신호의 granularity는 문제 수준 solve-rate 기준으로 블록형(특히 64토큰)이 기본값으로 유망하다는 점을 시사한다.



### CAVEWOMAN: How Large Language Models Behave Under Linguistic Input and Output Compression (https://arxiv.org/abs/2606.24083)
- **Prior Approaches**: 기존 LLM 압축 연구는 입력 압축이나 출력 압축을 한쪽만 다루거나, 토큰을 줄였을 때의 task accuracy만 주로 보고해 왔습니다. 하지만 압축 효과는 ‘실제 비용(입력/출력 채널별 토큰 단가)’과 ‘모델이 압축 전 기준으로 동일 내용을 말하는지’를 분리해 검증하기 어렵습니다. 그 결과, 짧아진 프롬프트가 비용을 정말 줄였는지와, 압축된 답이 모델 내부 추론 기준과 같은 텍스트 의미를 유지하는지 평가가 빈약했습니다.

- **Core Contribution**: 이 논문은 Cavewoman이라는 두 채널 평가 프로토콜을 제안해, 입력 압축과 출력 압축을 같은 아이템에서 동시에 비교합니다. 각 생성물에 대해 (1) 정답 정확도, (2) priced channel 기준 realized per-item cost, (3) 모델의 무제약 기준(L0) 생성과의 reference-text agreement을 함께 측정합니다. 또한 POS 기반 5단계 reduction(L0~L4)과 8개 모델·5개 벤치마크 전체 커버리지를 통해 채널별 비용/품질 분리를 정량화합니다.

- **Technical Challenges**: 핵심 난제는 ‘토큰 절감’이 실제 비용 절감으로 이어지는지, 그리고 정확도만으로는 기준 생성과의 텍스트 의미 일치를 구분할 수 없다는 점이었습니다. 이를 위해 모델 출력의 비용을 실제 priced 채널 토큰으로 산정하고, 정확도 평가 전에 answer-extraction rate를 감사해 파서/추출 편향을 통제했습니다. 더 나아가 reference-text agreement는 bidirectional NLI entailment과 11개 보완적 semantic 지표로 재검증해, 표면 텍스트 발화가 기준 생성과 어긋나는 현상을 측정합니다.

- **Empirical Impact**: 실험 결과, 출력 압축은 대부분의 API 모델에서 realized cost를 1.4~2.4x(최선 3x) 줄였지만 입력 압축은 반대로 순비용을 키웠습니다(평균 약 1.15x, 최악 최대 1.8x, 강한 압축에선 약 2.7x로 확대). 동일한 비용 절감 설정에서 비추론(non-reasoning) 6개 모델 묶음의 51.9%는 정답은 맞았지만 무제약 기준과의 텍스트 의미 포함관계를 더 이상 만족하지 못했습니다. 즉, 단일 축(accuracy/토큰) 압축 평가는 배치·감사·로그 소비 관점에서 오판할 수 있으며, 실제 배포에서는 constraint(출력 제약) 단계에서 후보를 순위 매겨야 한다는 실무적 시사점을 제공합니다.



### PixJail: Self-Evolving Paper-to-Pipeline Reproduction for Text-to-Image Jailbreak Evaluation (https://arxiv.org/abs/2606.24081)
- **Prior Approaches**: 기존 T2I jailbreak 평가는 단일 프롬프트 중심이 아니라 프롬프트 변환-이미지 생성-안전 필터링-멀티모달 판단이 얽힌 파이프라인 문제임에도, 많은 벤치마크와 재현 워크플로가 논문마다 구현·평가 구성이 달라 재현성과 공정 비교가 어려웠다. Jailbreak Foundry는 텍스트 LLM 공격을 executable 모듈로 매핑하는 데 강점이 있지만, T2I에서는 이미지 생성/시각 특징 회피/멀티모달 판정 등 추가 구성요소를 동일 방식으로 흡수하기 어렵다. 결과적으로 공격 코드만 있거나(또는 코드가 없어도) 평가 프로토콜까지 재현하지 못하면 ASR 비교가 흔들릴 수 있었다.

- **Core Contribution**: PixJail은 T2I jailbreak를 ‘공격 코드’가 아니라 ‘논문→파이프라인’으로 변환해 재현 가능한 평가 파이프라인까지 생성하는 프레임워크다. 논문(및 선택적 레퍼런스 코드)을 입력으로 받아 공격 모듈과 실행 가능한 평가 파이프라인을 통합된 계약(contract) 아래에서 자동 구성하며, 논문 매칭 재현과 표준화 벤치마킹을 동시에 가능하게 한다. 또한 PixJail-Memory로 이전 논문 요약, 공격 진화 패턴, 템플릿, 실패 사례, 버전 아티팩트를 축적해 다음 재현의 효율과 안정성을 높인다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 논문에 흩어진 공격/평가 설정을 실행 가능한 세부 파라미터로 분해하고 (2) 파이프라인 전 단계(필터·저장·판정·메트릭)가 같은 계약으로 묶이도록 하는 것이다. PixJail은 논문을 구조화된 중간 표현으로 변환한 뒤 공격 측/평가 측을 분리 합성하고, planner–implementor–auditor 및 protocol adapter–pipeline composer–consistency checker의 bounded revision 루프로 불일치를 수정해 제약을 맞춘다. 더해 공통 런타임 코어에서 안전 필터와 멀티모달 judge 판정 결과를 통합해 success/failure를 일관되게 계산한다.

- **Empirical Impact**: PixJail로 11개 대표 T2I jailbreak 방법(코드 제공/미제공 모두)을 동일 프로토콜에 넣어 재현했으며, 원설정 기준 평균 오차 2.1%(중앙값 0%)로 높은 고충실도 재현을 보였다. 또한 코드가 있는 방법은 평균 오차 1.2% 수준까지 낮아져 레퍼런스 코드+논문 정보 결합의 실용성이 확인됐다. 표준화 벤치마크에서는 모델별로 ASR 격차가 크게 나타나, paper-matched 평가에서 숨겨졌던 ‘파이프라인 제어 민감도’와 ‘피해자 모델 생성 경계 의존성’을 드러내며, 논문 간 비교의 신뢰도를 한 단계 끌어올릴 수 있음을 시사한다.



### End-to-End Radar and Communication Modulation Recognition with Neuromorphic Computing (https://arxiv.org/abs/2606.24075)
- **Prior Approaches**: 기존 end-to-end AMR 딥러닝은 raw IQ를 그대로 넣어 정확도를 끌어올렸지만, 대규모 파라미터와 dense MAC 연산 때문에 계산량·지연·전력 문제가 커 자원제한 플랫폼에 불리했습니다. SNN 기반 접근은 neuromorphic chip에서 spike-driven 희소 연산으로 효율을 노렸지만, Poisson/temporal 같은 static spike encoder로 인한 정보 손실과 장기 의존성 포착 한계로 정확도-전력 균형이 어려웠습니다.

- **Core Contribution**: 논문은 EMRFormer라는 end-to-end spiking Transformer 기반 AMR 구조를 제안해, 원시 IQ를 직접 입력으로 받아 신호-카테고리 인식을 수행합니다. 핵심은 adaptive spike encoder로 static 인코딩의 정보 손실을 완화하고, Integer Leaky Integrate-and-Fire(ILIF) 뉴런으로 SNN 표현력을 키우는 것입니다. 여기에 Spike-Separable Convolutional Neural Network(SSCNN)를 SpikeFormer(Spike-driven transformer)에 결합해 단기·장기 시간 특징을 동시에 학습합니다.

- **Technical Challenges**: 문제는 (1) 연속 IQ를 spike로 변환하는 과정에서 정보가 깎이고, (2) transformer가 필요한 장기 의존성 학습을 SNN의 event-driven 특성 안에서 효율적으로 구현해야 한다는 점입니다. 해결책으로 ILIF의 multi-spike(정수 방출)와 virtual time steps를 활용해 neuromorphic 하드웨어 친화적인 sparse 연산을 유지하고, SSCNN-기반 spike feature 추출 후 SpikeFormer attention으로 장거리 의존성을 보완합니다. 또한 encoder 전처리를 줄여 하드웨어 연산/저장 부담을 낮추고 저SNR 환경에서도 특징이 유지되도록 설계했습니다.

- **Empirical Impact**: 여러 대표 AMR 데이터셋(RML2016.10a/b, RML2018.01a, DeepRadar2022)에서 EMRFormer는 모든 baselines 대비 정확도를 상회하며, 특히 저SNR(<0 dB)에서 성능 우위를 유지했습니다. 저전력 측면에서도 이론 에너지 소비를 90% 이상 절감하고, KA200 neuromorphic chip 실측에서는 RTX 3090/Orin NX 대비 최대 5배 전력 절감(배치 1 기준)을 보였다고 보고합니다. ablation 결과에서도 adaptive encoder, ILIF, Positional Encoding, SNN 인코딩 방식 변경이 성능을 크게 흔들며 전체 설계가 저SNR 강건성과 희소 특징 학습에 기여함을 확인했습니다.



### Token Complexity of Certifying Stochastic-Oracle Reliability (https://arxiv.org/abs/2606.24074)
Comments:
          21 pages, 0 figures

- **Prior Approaches**: 기존 SOTM(Stochastic-Oracle Turing Machine) 연구는 주어진 확률적 오라클로 어떤 작업을 목표 성능까지 달성하는 데 필요한 token 복잡도를 다뤘다. 다만 이는 “성공/품질”을 맞추는 문제이고, 특정 오라클의 신뢰도(reliability)를 통계적으로 “증명”하는 비용을 직접 다루지는 못했다. 또한 정답성 확률이 있는 오라클 호출에서 생기는 두 단계(질문 토큰/응답 토큰) 비용 구조와, 오라클 신뢰도 구간 구분 문제는 별도 이론화가 필요했다.

- **Core Contribution**: 이 논문은 오라클 reliability certification을 정의하고, 어떤 오라클이 목표 신뢰도 이상인지 혹은 더 낮은 기준선 이하인지를 정해진 오차확률로 구분하는 데 필요한 “certification token complexity”를 도입한다. 특히 p≥p1 영역은 Reliable로, p≤p0 영역은 Unreliable로 판정하되, (p0,p1) 애매 구간에서는 판정 불필요하다는 양측 판정 설정을 명확히 한다. 이후 이 문제를 작은 오차(small-error) 정권에서의 token 비용 특성으로 정리해, 배치 전 사전 추정 관점을 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 오라클 신뢰도를 내부적으로 추정하지 않고, (2) 양측 신뢰도 문턱(p0,p1) 사이를 피하면서, (3) 질의-응답 토큰 비용을 반영한 “정지 규칙”을 설계하는 것이다. 논문은 SPRT(Sequential Probability Ratio Test) 기반의 non-adaptive certification SOTM을 구성해 각 질의의 이진 점수(correctness score)를 얻고 누적 log-likelihood evidence가 임계값을 넘는 순간 멈추게 한다. 그 결과 거의 확실한 종료와 원하는 두-sided error 보장을 증명하며, 신뢰도 문턱·허용 오차·turn당 기대 token 비용으로 certification token complexity의 명시적 상한을 제시한다.

- **Empirical Impact**: 추가로 정보이론적 lower bound를 통해, 허용 오차가 0으로 작아질 때(오차→0) SPRT 기반 상한과 leading-order 기대 token 비용이 일치함을 보인다. 즉, 적응적(adaptive) 질의를 쓰더라도 오차 구간이 작아질수록 필요한 token 비용의 1차 항은 피할 수 없다는 점이 정량적으로 확정된다. 최종적으로 confidence를 더 요구하는 데 드는 비용은 로그 수준으로만 증가하지만, p0와 p1의 분리가 좁아질수록 비용은 이차적으로 커진다는 성질이 도출되어, LLM 등 AI 오라클의 배치 전 “사전 비용 견적”에 직접 활용될 수 있다.



### Selective Capability Unlearning in End-to-End Spoken Language Understanding (https://arxiv.org/abs/2606.24063)
Comments:
          5 pages, 3 figures, preprint

- **Prior Approaches**: 기존 SLU unlearning은 주로 특정 intent의 사후확률 p(i|x)를 억제하는 방식(예: Gradient Ascent, NPO, KL 정규화, random label)을 사용한다. 하지만 autoregressive SLU에서는 intent 접두사가 강제로 주어지면 slot 생성은 여전히 그 intent prefix에 조건부로 복원될 수 있다. 이 때문에 intent 정확도는 떨어져도 forced-prefix 조건에서 slot recoverability가 남는 ‘capability persistence’ 문제가 발생한다.

- **Core Contribution**: 이 논문은 SLU에서 제거해야 할 ‘기능’을 intent 레이블이 아니라 intent-조건부 slot 생성의 conditional mapping으로 정의하고, 이를 제거하지 못하는 구조적 실패를 capability persistence로 명확히 규정한다. 그 해결책으로 BSU(Binding Subspace Unlearning)를 제안해, target intent가 만드는 intent-slot 결합에 해당하는 representation subspace를 찾아 감쇠시키는 방식으로 conditional mapping 자체를 약화시킨다. 또한 beam 및 임베딩 기반 유사도 등을 활용한 recoverability 중심 평가 프로토콜을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘marginal intent 억제’만으로는 slot 생성의 conditional mapping이 남는 autoregressive 구조를, 실제로는 표현공간에서 끊어내는 것이다. BSU는 (1) forget/retain 데이터에서 slot 위치의 hidden state를 teacher-forcing으로 정렬한 뒤 층(layer)별 covariance contrast로 intent-조건부로 풍부해지는 방향(바인딩 subspace)을 eigen-direction으로 추출하고, (2) 해당 subspace로의 gradient sensitivity를 정규화해 teacher-forced conditional log-likelihood의 의존도를 줄인다.

- **Empirical Impact**: SLURP와 SpeechMASSIVE(프랑스어 subset)에서 BSU는 forced-prefix 상황에서의 conditional slot recoverability를 크게 낮추면서(retain 성능은 대체로 유지) 의도한 제거 효과를 보여준다. 예를 들어 SLURP NeMo 설정에서 BRR@10은 92.64→22.10, semantic similarity는 90.14→24.80으로 감소했으며, 다른 초기화/데이터셋에서도 유사한 하락 추세가 관찰된다. ablation과 정량 분석을 통해 무작위 표현 교란은 효과가 제한적인 반면, binding regularizer 강도를 키우면 forget 지표가 지속적으로 내려가고 retain은 안정적임을 확인했다.



### RAVEN: A Regime-Aware Variable-context Expert Network for Financial Time Series Forecasting (https://arxiv.org/abs/2606.24062)
- **Prior Approaches**: 기존 금융 시계열 예측은 log-return으로 문제를 완화해도 잡음 대비 신호비(SNR)가 매우 낮고, 시장이 레짐(regime)에 따라 시간 의존성이 달라지는 비정상성이 남아 있습니다. XGBoost/LightGBM 같은 전통 모델은 정적 탭 입력으로 시간 구조를 놓치며, RNN 계열은 최근 관측에 치우치거나 장기 레짐 전환을 노이즈와 구분하지 못하기 쉽습니다. Transformer 계열 역시 고정 길이 context window L에 의존하는 구조적 병목이 있어, 단기 레짐 변화에는 정보가 부족하고 장기 창에는 과거 레짐의 잔여 잡음이 섞인다는 한계가 제기됩니다.

- **Core Contribution**: 이 논문은 SOTA 시계열 모델에서 ‘고정 context window’가 비정상 금융 데이터의 최적 look-back과 충돌한다는 점을 핵심 문제로 지목하고, Regime-Aware Variable-context Expert Network (RAVEN)을 제안합니다. RAVEN은 입력마다 가변 temporal context를 스스로 정하기 위해, reverse chronological 순서로 패치 중요도를 누적해 Cumulative Importance Thresholding (CIT)으로 중첩된 연속(prefix) look-back 창을 만들고 이를 scale-specialized expert에 라우팅합니다. 또한 local expert의 선택적 처리만으로 생길 수 있는 전역 일관성 손실을 Global Compressed Representation (GCR) 브랜치가 보완하도록 설계합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 입력별로 바뀌는 최적 길이를 안정적으로 선택하면서 (2) 중첩 라우팅으로 인해 expert 출력이 구조적으로 겹쳐 redundancy가 커지는 문제입니다. RAVEN은 CIT 기반으로 길이가 데이터에 의해 정해지는 중첩 prefix를 만들되, expert 간 중첩으로 생기는 상관/중복을 줄이기 위해 Correlation-Aware Weighting (CAW)로 집계 시 코사인 유사도를 페널티하고 표현 정렬을 유도합니다. 더불어 CIT 라우터 collapse를 막기 위한 entropy 정규화와, expert 표현 collapse를 억제하기 위한 diversity 정규화를 함께 사용해 저SNR 환경에서 학습 안정성을 확보합니다.

- **Empirical Impact**: 실험에서는 누적 log-return 예측(HS300, S&P500)과 fund sales forecasting에서 RAVEN이 기존 SOTA 대비 Pearson correlation을 HS300에서 9.2%, S&P500에서 20.2% 개선하고, fund sales forecasting의 MSE는 18.2% 감소시켰습니다. 또한 PEMS(03/04/07/08) 교통 벤치마크 4개에 대해 16개 지표 중 14개에서 최상위를 달성해 금융 외 일반화 가능성도 확인했습니다. 운영 관점에서는 동일한 현실적 backtest 조건에서 단일 production baseline 대비 누적 수익률이 10% 이상 우수했으며, 현재 온라인 통합과 실시간 모니터링/앙상블 고도화 단계가 진행 중입니다.



### Rapid FinFET Modelling Using an Autoencoder (https://arxiv.org/abs/2606.24046)
- **Prior Approaches**: 기존에는 FinFET을 BSIM-CMG 같은 물리 기반 컴팩트 모델로 피팅해 I-V 특성을 재현하는 방식이 주를 이뤘다. 하지만 실제 공정/소자 데이터가 늘어나면 파라미터 재보정 비용이 커지고, 바이어스 조건 변화가 반영된 높은 정확도를 유지하기가 어렵다.

- **Core Contribution**: 이 논문은 autoencoder(AE)를 이용해 FinFET의 전체 I-V 곡선을 저차원 잠재공간으로 압축하고, 그로부터 곡선 복원과 소자 지표 추출을 동시에 수행하는 ML 프레임워크를 제안한다. 특히 VDS 같은 바이어스 파라미터를 입력 feature로 명시적으로 포함해 bias dependent variation을 더 잘 포착하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 고차원 I-V 곡선의 비선형 물리 정보를 소수 차원에 효과적으로 담으면서, VDS 변화에 따른 특성 이동까지 재현하는 것이다. 저자들은 먼저 BSIM-CMG 보정으로 ID-VG 데이터셋을 만들고, AE 학습 시 VDS를 입력에 넣어 잠재공간이 디바이스 physics를 내재적으로 인코딩하도록 유도해 복원 성능과 지표 추출 정확도를 확보했다.

- **Empirical Impact**: 학습된 모델은 full I-V curve를 높은 정확도로 재구성하고, VTH, SS, peak transconductance(gm) 같은 핵심 소자 메트릭을 직접 추출해 공정/소자 특성화에 유용함을 보였다. 또한 실제 characterization 데이터 기반 data driven compact model이 적은 학습 데이터로도 정밀도를 낼 수 있어, 신속한 device characterization 및 회로 수준 simulation에 대한 실무적 임팩트를 시사한다.



### Towards Version-aware Operations and Transaction Memories for Multi-layer MeMo (https://arxiv.org/abs/2606.24040)
Comments:
          Accepted by MeMo Workshop on Mechanistic Interpretability & Neuro-symbolic Approaches by-design, Rome (Italy), 24/6/2026

- **Prior Approaches**: LLM 지식 갱신은 continued training, RAG(retrieval augmentation), parameter-level model editing 등으로 이뤄져 왔지만, 작은 변화에도 학습·대규모 재계산이 필요할 때 비효율적일 수 있습니다. 또한 ROME/MeMIT류는 업데이트가 “학습의 부작용”처럼 보일 수 있어, 무엇을 잊고 무엇을 남기는지에 대한 제어와 추적이 제한될 수 있습니다. MeMo는 correlation matrix memory를 외부화해 memorization/retrieval/forgetting을 명시적 연산으로 다룰 수 있다는 점에서 출발점이 됩니다.

- **Core Contribution**: 이 논문은 Multi-layer MeMo 위에 “버전 인지(version-aware) 연산 레이어”를 추가해, 모델 전체 재학습 없이도 명시적 memory association 기반의 지식 변경을 메모리 편집으로 반영하는 방식을 제안합니다. replace, obsolete, keep-history, rollback, trace 같은 고수준 연산을 MeMo 기본 편집들의 “순서가 있는 트랜잭션”으로 컴파일해 실행 가능하게 만듭니다. 이를 위해 Version CMM(V-CMM)과 Transaction CMM(T-CMM)을 도입해, 버전 전이→트랜잭션 핸들 매핑과 트랜잭션 내용의 재사용을 분리합니다.

- **Technical Challenges**: 핵심 난제는 버전 변경이 한 번의 MeMo association으로 끝나지 않고, 이전 연속체를 잊고 다른 연속체를 기억하며, 과거 체인을 보존하고 역연산(inverse program)을 기록하는 등 여러 primitive edits의 순차 실행이 필요하다는 점입니다. 논문은 next-token 기반 primitive (sequence, token) 편집으로 연속체(continuation) 업데이트를 분해하고, T-CMM이 템플릿/파라미터 또는 확장된 편집 시퀀스를 제공해 동일한 MeMo stack의 memo/forget/retrieve 프리미티브로 위임되도록 설계합니다. 또한 traceability와 rollback을 위해 트랜잭션을 역순 실행하거나 뷰/버전 alias 전환, 또는 exact log 재플레이 가능한 경로를 함께 제시합니다.

- **Empirical Impact**: 평가는 단계적 연구 로드맵으로 제안되며, toy 사실 편집과 ontology 스타일 diff(클래스 이동, 동의어 변경, obsolete 처리, 정의 업데이트 등)를 대상으로 업데이트 성공/기준선 유지(Outdated current suppression)/역추적·롤백 정확도/국소성(locality)/트랜잭션 재사용(reuse)을 지표로 삼습니다. 특히 trace는 현재 답에서 출발해 어떤 트랜잭션, 출처 버전, old/new value, 어떤 primitive edit들이 결과를 만들었는지를 메모리 수준 경로로 노출하는 해석가능성 기여로 정의됩니다. 제안 방식이 “MeMo-compatible memory edits로 표현 가능한 국소 변화”에서 점진적·가역적·추적 가능한 지식 업데이트를 실증하는 것이 의미 있는 목표입니다.



### Fast and Slow Variational Continual Learning (https://arxiv.org/abs/2606.24007)
- **Prior Approaches**: 연속 학습은 순차 업데이트 중 이전 지식을 “catastrophic forgetting” 없이 유지해야 하는데, SGD와 Adam 같은 표준 최적화기는 비정상 환경을 고려해 설계되지 않아 외부 장치가 필요하다는 한계가 큽니다. 기존 접근은 정규화(예: EWC 계열), replay/메모리, 혹은 아키텍처 분리처럼 학습 파이프라인 밖에서 안정성-가소성 균형을 맞추는 방식이 주류입니다. CLS 같은 뇌 과학 아이디어를 반영한 fast/slow 동작도 많이 제안됐지만, 실제로는 옵티마이저 내부에 직접 통합하는 합의된 방법이 부족했습니다.

- **Core Contribution**: 이 논문은 VCL(variational continual learning)에서 “past posterior를 prior로 사용”하는 베이지안 업데이트 구조에 fast/slow adaptation을 자연스럽게 넣는 방법을 제안합니다. 핵심은 느린 적응을 posterior merging으로 구현해 지식의 drift를 늦추고, 그 merged posterior를 다시 VCL 업데이트의 prior로 써서 빠른 가중치 갱신을 수행한다는 점입니다. 이를 IVON(Improved Variational Online Newton) 형태에 거의 동일한 비용으로 통합해 Continual IVON(CoVON) 옵티마이저를 제시합니다.

- **Technical Challenges**: 문제는 VCL의 단순 베이지안 갱신이 신경망 근사(예: mean-field 가정) 때문에 실제로는 drift/망각이 빠르게 악화될 수 있다는 점입니다. 논문은 (1) 현재 태스크를 위한 fast posterior를 정밀도(uncertainty/curvature) 기반으로 학습하고, (2) 이전 slow posterior와 새 정보를 precision-aware 방식으로 합쳐 지식을 점진적으로(지수 가중) 통합하도록 설계합니다. 또한 CoVON은 IVON의 diagonal Gaussian posteriors와 샘플링-기반 그래디언트/커브처 추정으로 Hessian 비용을 과도하게 늘리지 않으면서, merged prior를 통해 안정성과 가소성을 함께 제어하도록 구성했습니다.

- **Empirical Impact**: 실험에서 CoVON은 VCL 계열 기존 방법을 일관되게 능가하며, posterior merging을 제거하면(=No Merge) 성능이 크게 떨어져 fast/slow 결합의 필요성이 확인됩니다. Permuted MNIST(10 tasks)에서는 최종 정확도에서 약 2%p 개선과 함께 최고 성능이 92.12%까지 향상됐고, 표준 IVON처럼 느린 통합이 없으면 catastrophic forgetting이 심각하게 나타납니다. 더 나아가 CDDB, CORe50, DomainNet 같은 도메인 증분 학습 및 LLM의 continual pre-training/미세조정 시나리오에서도 다른 weight-regularization 및 exemplar-free SOTA와 비교해 우수한 균형을 보이며, 대규모 모델 스케일에서도 적용 가능함을 시사합니다.



### Towards Spec Learning: Inference-Time Alignment from Preference Pairs (https://arxiv.org/abs/2606.24004)
- **Prior Approaches**: 기존에는 zero-shot prompting이나 in-context learning처럼 사용자가 프롬프트를 직접 설계·반복하며 성능을 끌어올리는 방식이 널리 쓰였다. 자동 프롬프트 최적화(APO)나 보상/보정 기반 강화학습·gradient 기반 업데이트, 그리고 DPO 같은 preference-based fine-tuning도 있었지만 대개 라벨/비교 데이터와 비용이 많이 들고, 프롬프트 엔지니어링은 오류에 취약하다는 한계가 지적된다. 또한 LLM-as-a-judge는 평가 확장성은 좋지만 길이·위치·모델 패밀리 편향과 분산 문제도 동반한다.

- **Core Contribution**: 이 논문은 spec learning이라는 프레임워크를 제안한다. 사용자는 짧은 지시와 소량의 preference judgment(대략 20개 쌍)만 제공하고, 그 정보를 자연어 시스템 프롬프트 형태의 specifications로 컴파일해 inference 시점에 조건부로 동작하게 만든다. 핵심은 LLM 가중치를 업데이트하지 않고도, 특정 도메인에서는 DPO(데이터 50배 규모의 선호 기반 학습)보다 더 자주 이기며, 동시에 사람이 읽고 수정 가능한 “명시적 정렬 산출물”을 제공한다는 점이다.

- **Technical Challenges**: 문제는 소량의 preference 쌍을 사람이 해석 가능한 소수의 규칙 원칙으로 압축할 수 있는가이며, 그 규칙을 실제로 선호 신호와 정렬되게 구성하는 컴파일 절차가 필요하다는 것이다. 연구진은 (1) proposer가 후보 principles를 생성하고, (2) semantic 클러스터링·중복 제거·검증을 거친 뒤, (3) prevalence와 정확도를 함께 고려해 순위화하고, (4) janus(서술형 정책) 또는 bullets(룰 리스트) 합성기를 통해 system prompt를 조립한다. 또한 judge 편향을 줄이기 위해 강건한 강제 이진 판정(포지션 로테이션/다중 패스)과 gold 기반 캘리브레이션을 사용하며, 필요 시 response 위치 스왑으로 편향을 완화한다.

- **Empirical Impact**: 7개 preference 데이터셋에서 compiled specification은 base 모델 대비 일관되게 우수했고, 특히 domain preference 신호가 규칙으로 잘 요약되는 경우 HH-Helpful 같은 이질적 데이터셋보다 격차가 크게 나타났다. 매크로 평균 win rate는 DPO 대비 모두 높은 결과를 보이며, spec은 DPO보다 훨씬 적은 preference 쌍(50배 적음)으로도 더 잘 맞는 것으로 보고된다. 추가 분석에서 컴파일된 principles를 준 judge가 선호 응답을 더 많이 “조건 충족”으로 분류해, 가중치 업데이트가 아닌 명시적 지침이 실제 preference signal을 담아낸 해석 가능 산물임을 시사한다.



### EMAgnet: Parameter-Space EMA Regularization for Policy Gradient Self-Play in Large Games (https://arxiv.org/abs/2606.23995)
Comments:
          Accepted at NExT-Game 2026: New Frontiers in Game-Theoretic Learning (ICML 2026 Workshop). 13 pages, 2 figures,

- **Prior Approaches**: 자기대국(self-play)에서 PPO 같은 정책 그래디언트에 정규화를 더하면, 2인 영(0-sum) 불완전정보 게임에서 게임이론 기반 방법과 견줄 수 있다는 결과가 축적돼 왔다. 특히 uniform distribution을 entropy bonus로 쓰는 “uniform-magnet” 정규화가 강력한 기준선으로 자리 잡았지만, 모든 행동을 똑같이 향해 규제해 전략적으로 쓸모없는(특히 strictly dominated) 행동에도 정규화 예산이 낭비된다.

- **Core Contribution**: 이 논문은 정규화 타깃을 고정 uniform에서 벗어나, 에이전트가 학습하면서 변하는 “adaptive regularization target”으로 설정한다. 제안한 EMAgnet은 정책 네트워크 파라미터에 대한 exponential moving average(EMA)를 정규화 자석(magnet)으로 써서, dominated 전략을 회피하는 방향으로 타깃 자체가 자연스럽게 멈추게 한다. 결과적으로 “나쁜 전략은 잊고, 좋은 전략은 기억”하는 성질을 고정 타깃 대비 구현한다.

- **Technical Challenges**: 딥 강화학습에서는 상태별(policy space) 이동 자석을 그대로 유지하기 어렵기 때문에, 이를 파라미터 공간(parameter-space) EMA로 옮기는 것이 핵심 난관이었다. 논문은 PPO에 KL 정규화 항을 추가하되, uniform처럼 고정된 타깃이 아니라 PPO 업데이트마다 정책 파라미터의 EMA를 magnet 파라미터로 갱신해 연속적으로 적응하도록 설계했다. 이때 학습 루프는 크게 바꾸지 않고 “추가 EMA 업데이트 1회” 정도의 최소 복잡도로 구현된다.

- **Empirical Impact**: 표준 벤치마크에서는 PPO-EMAg(특히 EMA magnet 정책)이 PPO-Uniform(선형/파워법 스케줄)과 비슷하거나 더 낮은 exploitability를 달성하며, 대부분의 환경에서 우위가 관찰됐다. 더 나아가 strictly dominated 전략이 대량 포함된 수정 벤치마크(FF, Control 변형)에서는 일관된 성능 개선이 나타났고, 수렴 속도에서도 uniform 방식 대비 빠르게 낮은 exploitability에 도달했다. 즉, 전략적으로 쓸모없는 행동에 정규화 예산을 낭비하는 문제를 줄이면서도 학습 초중반의 혼합(mixing) 압력을 유지한다는 점에서 실전적 의미가 크다.



### Learning to Trigger: Reinforcement Learning at the Large Hadron Collider (https://arxiv.org/abs/2606.23993)
- **Prior Approaches**: 기존 LHC 트리거 메뉴는 대부분 정적이며 전문가가 사전 튜닝한 임계값을 기반으로, 파일업과 배경 구성 드리프트가 생길 때마다 반복 최적화 비용이 크게 듭니다. 동적 적응을 탐색한 연구도 있었지만, 일반적인 운영 시나리오처럼 “고정 메뉴를 온라인에서 재조정해야 하는” 제약 하에서 연속 적응이 실현 가능한지는 충분히 다뤄지지 않았습니다. 또 PID류 제어는 오류를 단일 스칼라로 섞어 서로 다른 실패 모드를 구분하기 어렵고, 값기반 RL은 드리프트 상황에서 재생 버퍼·가치함수 갱신이 불안정해질 수 있습니다.

- **Core Contribution**: 이 논문은 온라인 트리거 임계값 튜닝을 순차 의사결정(강화학습) 문제로 재구성하고, 스트리밍으로 들어오는 최근 레이트 요약과 신호 민감 특징을 관측해 임계값을 갱신하면서 “배경 레이트를 목표 허용대역 안에 유지”하는 제어를 목표로 합니다. Group-Filtered Policy Optimization(GFPO)을 스트리밍 제어로 개조해 GFPO-F, GFPO-FR 두 변형을 제안하고, 학습 중 배경 레이트 feasibility를 강제해 안정성과 효율을 함께 끌어올리도록 했습니다. 특히 HT(H_T)와 복원 손실 기반 이상탐지(AD) 두 트리거를 실제 CMS 데이터에 적용해 end-to-end RL 제어를 입증합니다.

- **Technical Challenges**: 핵심 난제는 (1) 배경 점수 분포가 pileup 및 검출기 조건 변화로 지속적으로 드리프트해 학습 데이터가 곧바로 off-distribution이 되며, (2) 임계값 근처에서 레이트의 국소 민감도(∂r/∂c)가 비정상적·이분산적으로 커져 값기반 TD 부트스트래핑이 깨지기 쉽다는 점입니다. 저자들은 (a) 스트리밍 상태를 최근 KK 이벤트 시퀀스와 임계값 주변 occupancy 등 “분포 맥락”까지 포함한 표현으로 만들고, (b) GFPO 계열에서 후보군을 샘플링 후 레이트 feasible 여부로 필터링하는 방식으로 학습 업데이트가 제약 위반을 강화하지 않게 설계했습니다.

- **Empirical Impact**: MC 스트림 벤치마크(현실적 콜라이더 운영을 모사)에서 H_T는 InBand(목표 배경 레이트 허용대역 내 시간 비율)를 48% 개선, AD는 28% 개선했으며, 해당 in-tolerance 구간에서 신호 효율도 최대 2%까지 누적 이득을 보였습니다. 더 나아가 시뮬레이션에서 학습한 에이전트를 추가 fine-tuning 없이 CMS Run 283408의 실제 충돌 데이터에 그대로 전이했을 때도 H_T 56%, AD 28%의 in-tolerance 개선과 함께 신호 효율 향상이 관측됐습니다. 저자들이 아는 한, 실제 LHC 충돌 데이터에서 RL 기반 트리거 제어를 시연한 첫 사례로 정리됩니다.



### RASC+: Retrieval-Constrained LLM Adjudication for Clinical Value Set Authoring (https://arxiv.org/abs/2606.23992)
- **Prior Approaches**: 기존 Clinical value set authoring은 큰 코드 우주에서 해당 집합을 찾아내야 하며, 직접적인 코드 생성은 제약조건(버전관리·감사가능성) 때문에 잘 맞지 않는다. RASC는 retrieve-then-select로 무제약 생성의 위험을 줄였지만, stage-1 retrieval이 놓치면 stage-2에서 복구가 불가능하고, 단일 retriever 설계는 recall 병목을 만들 수 있었다.

- **Core Contribution**: 이 논문은 value set completion을 stage 1(후보 풀 구성)과 stage 2(후보 판정)로 분해하고, 각 단계의 목표를 따로 최적화한다. 특히 stage-2는 LLM이 코드를 새로 생성하지 않고 ‘후보 풀에서만’ 선택하도록 제한해 auditability를 유지하면서 precision을 끌어올리는 constrained LLM adjudication을 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) 타깃과 표현이 다르거나 publisher가 달라져도 누락 없이 높은 recall을 만들 후보 풀을 구성하고, (2) 커진 풀에서 LLM이 노이즈를 걸러내는 것이다. 이를 위해 Qwen3 기반 retrieval에 vocabulary-aware graph expansion, ICD-10-CM/RxNorm/SNOMED-CT별 구조적 확장, 그리고 code-display rescue retrieval(BM25)을 결합했으며, GPT-5 adjudication은 blinded 프로토콜로 VSAC의 누출 신호를 차단하고 JSON 스키마로 후보 ID만 반환하도록 설계했다.

- **Empirical Impact**: 결과로 stage-1 후보 풀 recall은 RASC baseline 0.553에서 0.730으로, held-out publisher에서는 0.655로 상승했지만 SAPBert cross-encoder는 같은 풀에서 full-test macro F1 0.287, held-out publisher 0.233에 머물렀다. 반면 GPT-5 adjudication은 full-test macro F1을 0.549, held-out publisher macro F1을 0.533으로 끌어올려 retrieval ceiling을 end-to-end로 더 잘 활용했음을 보였다. 저자들은 이 성능 향상이 ‘후보 풀에서만 선택’하는 안전 제약을 보존하면서도 OOD(publisher shift)에서 크게 개선된다는 점에서 의미가 크다고 정리한다.



### Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization (https://arxiv.org/abs/2606.23989)
- **Prior Approaches**: 기존 멀티문서 요약(MDS)은 end-to-end LLM이 생성 후에 근거를 붙이는 경우가 많아 환각(unsupported content)과 함께 검증 가능성이 낮다는 한계가 있었다. 인용이 있어도 문서/문단 단위로 뭉뚱그려져 있어 각 문장이 실제로 어떤 근거를 쓰는지 확인하기 어렵고, 충돌을 침묵적으로 덮는 경향도 컸다. 또 span-first 방식은 원시 span에 고정해 생성하지만, 문서 간 동등 주장 통합이나 개별 주장 단위 검증까지는 자연스럽게 이어지기 어렵다.

- **Core Contribution**: 이 논문은 Extract–Select–Rewrite의 중간표현을 ‘인용·검증의 단위’로 바꾸는 CAMS를 제안한다. CAMS는 각 문서에서 원자적 claim을 추출해 토큰 수준 provenance를 만들고, 문서 간 동등 claim을 클러스터링하며 충돌을 감지한 뒤, 선택된 claim에만 기반해 재작성하고 각 문장에 claim-근거(span) 포인터를 달아 검증 루프까지 수행한다. 핵심은 문장 생성 전에 내용이 지역화(localized)되도록 파이프라인 구조를 attribution-oriented, faithfulness-oriented로 설계한 점이다.

- **Technical Challenges**: 원자적 claim이 어디에 ‘정확히’ 존재하는지 오프셋으로 직접 강제하면 LLM이 위치를 잘못 내놓는 문제가 있어, claim(정규화 텍스트)과 해당 quote를 분리 저장하고 quote를 span으로 매핑하는 방식으로 처리했다. 또한 클러스터링에서 동등성/유사성을 embedding만으로 판단하면 ‘서로 반대인 주장’이 합쳐질 위험이 있어, bidirectional entailment 기반으로 비상호-엔테일링 병합을 거부하고 후보 충돌 쌍만 골라 3-class NLI로 contradiction을 판정한다. 마지막으로 생성 후 드리프트를 막기 위해 문장에 대해 인용된 span이 문장을 뒷받침하는지(지지)와 인용 밖의 추가 사실이 없는지(정밀) 양방향 체크와 제한적 repair를 수행한다.

- **Empirical Impact**: MultiNews, DiverseSumm, WCEP에서 품질·신뢰도·로컬라이제이션을 분리 평가했으며, 특히 독립적인 support 평가 모델로 citation precision을 감사(audit)하는 2-regime 프로토콜을 사용했다. CAMS는 강한 end-to-end 및 span-attribution 베이스라인과 견줄 만한 요약 품질을 유지하면서, faithfulness와 인용 정밀도에서 크게 개선되어 multi-source attribution 정확도를 약 2/3 수준으로 끌어올렸다. 또한 faithfulness–coverage 트레이드오프를 선택 임계값으로 ‘조절 가능’하게 드러내 end-to-end 모델이 암묵적으로만 다루던 문제를 명시적으로 통제할 수 있음을 보여준다.



### Maestro Order: A Model-Agnostic Orchestration Harness (https://arxiv.org/abs/2606.23983)
Comments:
          10 pages, 4 figures

- **Prior Approaches**: 기존 접근은 대부분 생성에 의존하거나(예: 한 번에 답하기, chain-of-thought), 여러 샘플을 모아 최빈/투표로 정답을 고르는 방식(self-consistency, voting), 또는 생성-검증-수정 루프(ReAct, Reflexion, Self-Refine 등)처럼 부분적으로 verification을 붙이는 데 그쳤습니다. 이때 중요한 비용-신뢰도(얼마나 비싸고 얼마나 안전한가)와 검증기의 품질(거짓수용률/완전성)이 파이프라인 설계에 명시적으로 반영되지 않아, instance별로 최적 조합을 찾기 어렵다는 한계가 있습니다. 또한 decomposition과 같은 다단계 구성은 작은 오류가 누적되기 쉬워, 단독 기법으로는 “신뢰 가능한 조직”을 만들기 힘듭니다.

- **Core Contribution**: 이 논문은 어떤 모델이든 black-box base solver로 감싸서, 불확실한 솔버를 신뢰 가능한 문제해결 시스템으로 바꾸는 model-agnostic orchestration harness인 Maestro Order를 제안합니다. 핵심은 decompose, ensemble, verify, recurse의 네 가지 구조 프리미티브를 일관된 스키마로 조합하고, compute 예산을 어디에 쓰는지 결정하는 budget-aware controller를 붙여 “정확도 향상”을 비용과 함께 제어하는 것입니다. 특히 verification gating의 신뢰도 증폭을 검증기 discrimination(Λ=β/α)으로 측정·활용해, instance마다 적절한 메커니즘을 선택하도록 설계했습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 검증기가 생성 방식을 “악용”해 거짓수용률이 올라가는 verifier gaming, (2) 솔버/검증기의 오류가 상관되어 ensemble 효과가 줄어드는 correlated errors, (3) decomposition 경계에서 오류가 누적되는 compounding입니다. 논문은 결정론적(trace 기반 재현), fail-closed(툴/타임아웃/오류는 reject), 점수 캘리브레이션(의사결정에 raw score를 직접 쓰지 않음), 그리고 각 검증기의 β(완전성)와 α(거짓수용률)를 온라인으로 추정해 Λ를 업데이트하는 장치로 이를 제어합니다. 또한 planner는 타입 호환성과 “검증 가능한 분해”일 때만 split을 허용하고, controller는 각 단계의 marginal reliability gain per unit cost를 기준으로 검증과 투표를 단계별로 배치합니다.

- **Empirical Impact**: 평가에서는 reliability at fixed cost, coverage/ risk–coverage, calibration(ECE), 비용·지연을 축으로 측정하고, 투표(voting)와 검증(gating)의 상대적 이득 및 분해/재귀/다양성의 기여를 ablation으로 분리합니다. 특히 parameterized solver/verifier에 대한 faithful Monte Carlo simulation으로 odds law를 정량적으로 재현했는데, 예를 들어 Λ가 있는 verification gate는 기하급수적으로 신뢰도를 끌어올려 0.55→0.98(게이트 2개) 및 0.999(게이트 4개) 같은 결과를 보였습니다. 동일 목표 신뢰도에 대해 budget-aware controller가 voting 단독 대비 훨씬 적은 비용으로 목표를 달성하며, 이는 “신뢰도-비용”을 명시적으로 최적화하는 오케스트레이션의 실용적 가치를 시사합니다.



### Offline Reinforcement Learning for Warehouse SLAM Throughput Contro (https://arxiv.org/abs/2606.23978)
Comments:
          Accepted at 2026 14th International Conference on Control, Mechatronics and Automation (ICCMA 2026)

- **Prior Approaches**: 기존 창고 연구는 주로 로봇 내비게이션이나 공간적 협업 같은 시각적/물리적 계획 문제에 집중해 왔고, SLAM 처리량 자체를 연속 자동제어하는 접근은 상대적으로 부족했다. 공정 수준 RL도 배치/순서/디스패칭처럼 의사결정 시점의 규칙을 다루는 경우가 많아, 지연된 보상과 상류-하류 동시 안정화가 핵심인 ‘throttling(처리속도 제한)’ 문제에는 직접적으로 대응하기 어려웠다. 또한 온라인 학습 기반은 안전성 제약 때문에 실제 물류 현장에서 곧바로 적용하기가 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 오프라인 reinforcement learning을 이용해 SLAM throughput control(처리량 제어)을 최적화하는 프레임워크를 제안한다. 상류(예: pick/pack) 처리 흐름이 하류 congestion(정체/혼잡)에 미치는 지연 효과를 함께 고려하면서, 처리량 극대화와 하류 안정성 간 균형을 자동으로 조절하는 정책을 학습한다. 특히 알고리즘-agnostic한 통합 구조로 BCQ, CQL, TD3+BC를 동일한 파이프라인에 얹어 비교·적용할 수 있게 했다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 지연된 시스템 반응 때문에 ‘지금의 throttling이 나중의 congestion/효율로 이어지는’ 장기 의존성을 모델링해야 한다는 점, (2) 다수 SLAM lane의 다단계 이산 제어가 결합되면 행동공간이 폭증한다는 점, (3) 상류 backlog과 하류 congestion을 동시에 반영하는 보상이 학습 안정성을 해칠 수 있다는 점이다. 논문은 이 문제를 위해 시간 이력 기반 state representation(슬라이딩 윈도/예측 맥락), 시간 가중 평균으로 action space를 저차원 연속 표현으로 축약한 뒤 실행 시 이산 레벨로 복원, 그리고 상류·하류 지표를 정규화/재스케일한 balanced reward로 해결한다. 또한 과적합/분포이탈 위험을 줄이기 위해 CQL 등 오프라인 RL 알고리즘의 특성을 그대로 활용하고, FQE와 Deep Koopman 기반 모델평가로 장기 거동을 함께 검증한다.

- **Empirical Impact**: 대규모 창고의 비식별 historical operational logs로 학습·평가했으며, 15분 단위 결정을 두고 모델-free(즉시 보상 회귀 추정, Fitted Q Evaluation) 및 모델-based(Deep Koopman 동역학 롤아웃) 다각도 평가를 수행했다. 그 결과 CQL 정책은 behavior baseline 대비 장기 성능을 22.97% 개선했고, 평균 throttling duration은 3.18% 줄였다. 즉시 보상은 항상 최적이 아니더라도(장기 목표 반영) 시스템 건강과 개입 빈도/지속시간을 동시에 개선하는 안전·확장 가능한 오프라인 RL 적용 가능성을 보여줬다는 점에서 의미가 크다.



### When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents (https://arxiv.org/abs/2606.23937)
- **Prior Approaches**: 정책-제약 tool-use 시스템에서 정책 절(clause)을 미리 검색한 뒤 allow/verify/refuse 같은 사전( pre-action ) 분류를 수행하는 흐름이 널리 쓰인다. 이때 검색 품질 평가는 benchmark에 지정된 ‘정답 정책 절’이 top-k에 들어오는지의 exact-match recall로 대신하는 경우가 많지만, 정답이 아니더라도 결정에 필요한 정책 정보가 포함될 수 있어 downstream 유용성과의 괴리가 생길 수 있다.

- **Core Contribution**: tau-bench에서 Qwen2.5-3B/7B 분류기를 대상으로, (1) decision-state 표현 방식이 정책 분류에 미치는 효과와 (2) 정책 절을 검색해 넣었을 때 exact-match recall이 downstream 성능을 얼마나 예측하는지(프록시 타당성)를 분리해 평가한다. 또한 gold-policy로 조건을 준 설정에서는 compact structured state가 raw trajectory 대비 macro-F1을 크게 끌어올리지만, 검색으로 정책 절을 주입할 때는 recall이 낮아도 성능 저하가 감지되지 않는 구간이 존재함을 보인다.

- **Technical Challenges**: 주요 기술 과제는 ‘정확히 같은 clause를 회수했는가’라는 오프라인 진단이 ‘분류 루프에서의 정책 유용성’을 과소평가할 수 있다는 점을 직접 검증하는 것이다. 논문은 gold-policy 입력을 structured 형태로 명시적으로 구성하는 생성기( {request intent, evidence, policy assertion, action class} 추출 )로 representation 효과를 먼저 확실히 만든 뒤, 테스트 시 해당 정책 절을 top-1 retrieved clause로 치환하는 직접 개입 실험으로 recall-성능 대응을 시험한다.

- **Empirical Impact**: airline 설정에서 top-1 retrieved clause의 exact-match recall은 약 7%에 그치지만, 3B 분류기는 retrieved clause로 macro-F1 0.58을 얻어 gold clause(0.60) 대비 눈에 띄는 손실이 관측되지 않았다(95% CI가 넓어 비열등(non-inferiority) 확정은 어려움). mismatched-policy(0.32)와 no-policy(0.21) 대비로는 retrieved clause가 유의미하게 높아, 이 벤치마크에선 recall만으로 downstream 정책 유용성을 평가하면 보수적으로 틀릴 수 있음을 시사한다. 결과는 다른 retriever/7B 및 파인튜닝 구성에서도 유사한 패턴이 나타나며, 정책 검색 평가는 recall만이 아니라 ‘검색-분류 루프에서의 성능’까지 함께 측정해야 한다는 실무적 함의를 제공한다.



### Catastrophic Compositional Generation: Why Vanilla Diffusion Models Fail to Extrapola (https://arxiv.org/abs/2606.23920)
- **Prior Approaches**: 조합 생성(compositional generation)은 조건부 생성모델을 학습 조건의 부분집합만으로 학습한 뒤, 관측되지 않은 조건의 조합으로부터 표적 분포를 샘플링하려는 문제다. 특히 조건부 확산모델에서 많이 쓰는 휴리스틱은 denoising 과정 중 score를 선형 결합하는 방식이지만, 노이징을 거치면 score의 선형 관계가 깨지며 근본적인 근사오차가 발생한다.
이를 줄이기 위해 Feynman-Kac correction(FKC) 같은 particle 기반 보정 기법이 등장했으나, 본 논문은 그보다 더 치명적인 원인으로 score estimation error를 지목한다.

- **Core Contribution**: 본 논문은 조건부 diffusion을 “분포 수준의 기하 가중 조합”을 “효율적으로 샘플링”해야 하는 과제로 정식화하고, 그 어려움이 인퍼런스-time 근사오차뿐 아니라 학습된 score가 만드는 추정오차에서 비롯된다는 점을 이론적으로 주장한다. 특히 표적 분포가 source 분포들에 대해 out-of-distribution(OOD)인 설정에서, score estimation error가 성능을 압도하며 보정이 오히려 샘플 품질을 악화시킬 수 있음을 보여준다.
또한 weighted composition이 인과 표현학습(causal representation learning)과 깊게 연결됨을 논의하며, 오류가 왜 특정 조합에서 더 크게 나타나는지 체계적으로 분석한다.

- **Technical Challenges**: 핵심 기술적 난제는 두 갈래인데, (1) denoising 궤적에서 target을 정확히 추적하지 못하는 inference-time approximation error와 (2) 학습된 score 함수가 조합된 분포에서 잘 동작하지 못해 생기는 score estimation error다. 기존 보정(FKC)은 전자의 오차를 줄이기 위해 importance weight를 추적하지만, 본 논문은 조합된 분포가 source의 저밀도 영역에 놓이는 경우 OOD 성격 때문에 score 추정오차가 지배적으로 커진다고 분석한다.
이를 분리해 보기 위해 합성 데이터와 현실 데이터 실험을 설계하고, score 추정오차와 인퍼런스 근사오차가 실패를 유발하는 서로 다른(부분적으로 겹치는) 영역을 분해해 관찰한다.

- **Empirical Impact**: 이 논문은 이론적 직관(오차의 거동, OOD에서의 취약성)과 함께, Gaussian 같은 해석 가능한 세팅부터 합성/현실 데이터까지 다양한 실험을 통해 compositional failure 양상을 정교하게 재현한다. 그 결과, 최근의 FKC 계열이 inference-time approximation error를 완화하더라도, target이 source에 대해 OOD인 경우에는 score estimation error가 더 “파국적” 영향을 준다는 결론을 얻는다.
따라서 향후 조합 생성에서 단순한 inference-time 보정보다, OOD score 추정을 견딜 수 있는 다른 접근(표적 분포 영역에서의 일반화 강화 등)이 필요하다는 실무적 함의를 제공한다.



### ARIA: Adaptive Region-Based Importance Allocation for Conditional Diffusion Distillation (https://arxiv.org/abs/2606.23898)
Comments:
          26 pages, 11 figures

- **Prior Approaches**: 조건부 디퓨전에서 지식 증류는 teacher의 잡음 예측이 조건 신호에 강하게 의존해, 학습 분포 밖의 지식을 잘 옮기지 못하는 문제가 있었다. 특히 RC(Random Conditioning)는 캐시된 이미지-텍스트 쌍에서 텍스트를 무작위로 바꿔 조건 공간을 넓히지만, 어떤 영역이 실제로 더 어려운지에 대한 적응성이 없어 자원 배분이 비효율적일 수 있다. 또한 paired 이미지-조건 데이터는 제한적인 반면 텍스트 프롬프트 풀은 방대해, “학습 예산을 어떤 조건에 써야 하는가”가 별도 병목으로 남았다.

- **Core Contribution**: ARIA(Adaptive Region-based Importance Allocation)는 조건 공간을 거친(region) 단위로 나누고, teacher-student 불일치가 지속되는 영역에 학습을 더 집중하도록 학습 샘플링을 조정한다. 기존 증류 목적(teacher-student noise prediction/feature objective)은 그대로 유지하면서, 조건 선택(샘플링 정책)만 가볍게 교체해 RC의 한계를 보완한다. 특히 coarse region 수준의 불일치 추적을 통해, 초대형 조건 코퍼스에서도 per-sample 점수 계산 없이 확장 가능한 적응형 배분을 제안한다.

- **Technical Challenges**: 핵심 기술 난점은 조건이 너무 크기 때문에 전체 조건에 대해 teacher-student 차이를 정밀하게 측정하거나 전수 생성하는 것이 불가능하다는 점이다. ARIA는 이를 region-level EMA(지수이동평균)로 해결해, 각 region의 discrepancy를 온라인으로 추정하고 softmax 같은 단조 매핑을 통해 샘플링 확률을 재가중한다. 더불어 bounded variance와 bounded drift 가정 하에서 EMA 기반 tracking이 학습 중 “진화하는 불일치”를 유한 시간 내 안정적으로 따라간다는 이론적 보장(오차 항 분해 및 고확률 안정성)을 함께 제시한다.

- **Empirical Impact**: 실험에서 ARIA는 RC 대비 대부분의 아키텍처와 설정에서 성능을 개선하며, 특히 unseen 및 underrepresented(잘 표현되지 않는) 영역에서 이득이 가장 크게 나타난다. Stable Diffusion v1.4, v2.1, SDXL teacher를 대상으로 BK-SDM, channel-pruned, block-pruned, KOALA students 등 다양한 조합에서 일관된 향상과 더 빠른 수렴을 보였다. 또한 추가 오버헤드는 보조 텍스트 임베딩/클러스터링에 주로 발생하며 학습 중에는 EMA 갱신만으로 충분해 실용적 비용 내에서 효과를 얻는 것으로 보고된다.



### The Professor: Multi-Teacher Unsupervised Prompt Distillation for Vision-Language Models (https://arxiv.org/abs/2606.23897)
- **Prior Approaches**: PromptKD는 unlabeled 도메인 이미지로 큰 VLM(teacher)의 soft 예측을 작은 student에 KL-divergence로 전이해, 추론 시 teacher를 제거하는 비지도 프롬프트 증류 방식을 제시했다. 하지만 teacher가 하나라서 pretraining, prompt tuning 목표, calibration 같은 ‘한 가지 귀납편향’이 그대로 soft label에 반영되는 한계가 있었다. 기존 앙상블/증류 문헌은 teacher diversity가 이득의 핵심이라고 보지만, VLM 프롬프트 학습 맥락에서는 다중 teacher가 언제/얼마나 유효한지 충분히 검증되지 않았다.

- **Core Contribution**: TheProfessor는 PromptKD를 다중 teacher 증류로 확장해, 서로 다른 두 CLIP-family teacher의 신호를 student에 함께 학습시킨다. T1은 PromptSRC로 도메인 few-shot을 학습한 CLIP ViT-L/14이고, T2는 zero-shot EVA-CLIP-L/14이며 T2의 logits는 데이터셋별로 사전 캐시해 학습 중 추가 추론 비용을 없앤다. 또한 teacher 예측을 equal averaging과 confidence-weighted averaging 두 방식으로 결합하며, 특히 도메인 shift에서 보완적 supervision이 있을 때 효과가 커진다는 관점을 제시한다.

- **Technical Challenges**: 다중 teacher를 쓰면 학습 중 두 큰 모델의 forward가 필요해 비용이 급증하는데, 이를 해결하기 위해 T2 logits를 이미지 단위로 한 번만 pre-compute해 저장하고 학습 시에는 경로로 즉시 로드한다. 다음으로 학생 학습 손실은 앙상블 teacher 분포와 student 분포 사이의 KL loss로 정의해 end-to-end student 업데이트를 유지했다. 마지막으로 confidence-weighted averaging은 이미지별 최대 softmax confidence를 가중치로 사용해, teacher 신뢰도가 이미지/데이터셋에 따라 달라지는 문제에 적응하도록 설계했다.

- **Empirical Impact**: 4개 base-to-novel 데이터셋(Caltech-101, DTD, UCF101, EuroSAT)에서 12-run 단일 시드 스윕을 수행한 결과, confidence-weighted ensembling이 평균 HM을 87.52에서 89.28로 +1.77p 개선했다. equal averaging은 평균 HM을 88.88로 +1.37p 올렸고, 데이터셋 의존성이 뚜렷해 Caltech-101에서는 거의 변화가 없었지만 EuroSAT에서 HM이 +5.78p로 가장 크게 상승했다. 저자들은 teacher agreement가 높은 ‘쉬운’ 데이터셋에서는 이득이 천장에 걸리지만, 도메인 shift에서는 teacher의 상보성이 살아나 multi-teacher prompt distillation이 상당한 성능 향상을 낼 수 있음을 실증했다고 정리한다.



### E-MRL: Cross-view Aligned Evidence-driven Multimodal Reinforcement Learning for Reliable 3D Tumor Analysis (https://arxiv.org/abs/2606.23888)
Comments:
          9 pages, 2 figures

- **Prior Approaches**: 기존 Vision-Language Model(VLM) 기반 3D CT 종양 리포트 생성은 주로 Supervised Fine-Tuning(SFT)이나 text-based Reinforcement Learning(RLHF)처럼 텍스트 정합성·진단 정확도에 보상을 주는 방식이 많았다. 이 접근은 시각 근거(어느 슬라이스를 봤는지) 자체에 대한 과정 수준 감독이 부족해, 입력에 없는 병변 디테일을 언어적 사전지식으로 그럴듯하게 만들어내는 visual hallucination이 발생하기 쉽다. 또 3D 분야에서도 시각 보상은 제한적이어서, 미스인 지역화 오류를 학습 신호가 제대로 벌주지 못하는 한계가 있었다.

- **Core Contribution**: 논문은 E-MRL(Evidence-driven Multimodal Reinforcement Learning)을 제안하며, 생성 과정을 diagnosis-localization-verification의 Markov Decision Process로 재구성한다. 모델이 전역 진단 리포트를 내는 동시에 “key evidence slice” 인덱스를 명시적으로 찾아, 해당 슬라이스를 다시 질의(verification)해 근거를 검증하도록 학습한다. 핵심은 cross-view consistency reward로, 전역 리포트에 적힌 속성과 로컬 슬라이스 재질의 결과가 의미적으로 일치할 때 추가 보상을 줘 텍스트-시각 정합을 강제한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 key slice 인덱스 선택이 이산적이라 end-to-end 최적화가 어렵다는 점과, 멀티모달 보상을 어떻게 안정적으로 설계할지였다. 이를 위해 슬라이스 선택과 텍스트 생성을 함께 다루는 hybrid action(토큰 생성/슬라이스 포인팅)으로 MDP를 구성하고, 종양 존재, fine-grained 속성(size·location·enhancement), 그리고 cross-view 일관성의 다중 보상을 조합했다. 또한 value network 없이 안정적인 정책 업데이트를 위해 GRPO(Group Relative Policy Optimization)를 사용하고, KL 페널티 및 기준 SFT 모델을 활용해 reward hacking을 줄이도록 설계했다.

- **Empirical Impact**: AbdomenAtlas3.0의 대규모 3D CT 종양 데이터에서 E-MRL은 SFT·RL 기반 경쟁군과 RadGPT를 포함한 여러 의료 VLM 대비 hallucination을 줄이고 진단 정확도와 속성 근거(특히 key slice hit rate)에서 일관되게 향상됐다. 보고된 성능은 세 종양 유형 평균 balanced-accuracy(B-Acc) 87.93%로, 이전 SOTA RadGPT 대비 16% 이상 높은 수준이며 전역-로컬 근거 정렬이 잘 된 사례도 제시됐다. 결과적으로 “과정 수준 시각 검증”을 갖춘 end-to-end 3D 의료 VLM의 신뢰도·해석 가능성을 높였다는 점에서 임상 보조용 자동 리포트 생성 연구에 의미 있는 진전을 제공한다.



### Mind the Heads: Topological Representation Alignment for Multimodal LLMs (https://arxiv.org/abs/2606.23885)
- **Prior Approaches**: 기존 representation alignment 연구는 MLLM 내부 표현을 비전 인코더 표현에 맞추는 방식으로 비주얼 능력을 끌어올리지만, 대개 LLM의 고정된 한 층(예: 중간층)만 정해 정렬합니다. 또 layer-level 정렬은 Transformer의 미세한 구조(어텐션 head별 역할)를 충분히 반영하지 못해, 언어 모델의 기존 우선순위와 충돌하거나 최적의 선택을 놓칠 수 있다는 한계가 제기됩니다.

- **Core Contribution**: 이 논문은 Head-Wise Representation Alignment(HeRA)를 제안해, LLM의 attention head 단위로 비전-언어 표현 정렬을 수행합니다. MKNN(상호 k-최근접 이웃) 점수를 진단으로 사용해 정렬할 head를 선택하고, 대비(contrastive) 목적함수로 비전 인코더가 만드는 로컬 이웃(topological neighborhood) 구조를 차분하게 흉내 내도록 학습시킵니다.

- **Technical Challenges**: 핵심 난관은 MKNN이 k-최근접 이웃 인덱스에 의존해 미분 불가능하다는 점인데, 논문은 이를 differentiable proxy인 multi-target InfoNCE 형태의 대비 손실로 대체합니다. 또한 head를 무작정 탐색하기 어렵기 때문에, 멀티모달 학습 전 순수 텍스트 상태에서 head별 MKNN 정렬 점수를 계산해 상위/하위 후보를 선별하고, HeRA는 그중 일부 head에만 정렬 손실을 적용합니다.

- **Empirical Impact**: 18개 벤치마크(Cambrian)와 여러 hallucination 벤치마크에서 HeRA는 vision-centric 과제에서 일관된 성능 향상을 보이며, 종종 시각 환각을 줄이는 정규화 효과도 확인됐습니다. 특히 “가장 덜 정렬된(worst) head를 맞추는 전략”이 가장 큰 이득을 주었고, 다양한 MLLM(LLaVA 프레임워크, 파라미터 스케일 포함)에서 General/Knowledge/OCR 성능 저하 없이 안정적으로 개선되었습니다.



### One Year Later...The Harms Persist, But So Do We! (https://arxiv.org/abs/2606.23884)
Comments:
          20 pages, 8 tables

- **Prior Approaches**: 기존에는 범용 LLM을 정신건강 대화에 활용하되, 안전장치가 임상 조건에 따라 일관되게 작동하는지에 대한 검증이 부족했다. 특히 DSM-5의 여러 임상 진단을 폭넓게 비교하면서 공격 상황에서 취약점을 체계적으로 분류한 연구가 제한적이었다.

- **Core Contribution**: 이 논문은 16개 DSM-5 조건에 대해 6개의 proprietary LLM을 평가하고, 피해를 8개 차원으로 분류한 harm taxonomy와 다차원 평가 프레임워크를 제시한다. 그 결과 안전장치가 자살·자해에서만 비교적 안정적으로 유지되고, 섭식장애·물질사용장애·주요우울장애 등에서는 실패율이 최대 100%까지 나타났다고 보고한다.

- **Technical Challenges**: 핵심 기술적 난제는 임상 조건별로 어떤 형태의 위해가 발생하는지 정의가 불명확하고, 공격에 의해 안전성 경계가 쉽게 무너지는지 정량화하기 어렵다는 점이다. 연구진은 4가지 adversarial attack variant로 시나리오를 흔들어 보며, 8차원 harm taxonomy와 다차원 평가로 위해 유형을 구체화해 모델 취약성을 더 일관되게 드러내도록 설계했다.

- **Empirical Impact**: 실험 결과는 범용 LLM의 안전장치가 특정 진단군에 대해서는 신뢰 가능하지만, 다른 임상 조건에서는 전면적 실패가 발생할 수 있음을 경험적으로 보여준다. 저자들은 교육 현장 등 취약 집단이 상호작용하는 영역으로의 통합이 늘어나는 만큼, 임상 조건 전반에 걸친 명확한 위해 범주 정의와 그에 상응하는 safeguards 구현이 선행돼야 한다고 경고한다.



### Promise and challenges of heart chamber segmentation from non-contrast CT scans using contrastive unpaired image translation: a feasibility study (https://arxiv.org/abs/2606.23879)
- **Prior Approaches**: 기존에는 비조영 CT에서 심장 챔버(4방) 분할을 위해 조영 CT 또는 수작업 비조영 라벨을 활용하는 방식이 주로 쓰였다. 다만 비조영 CT에는 라벨 확보가 어려워, 비지도/준지도나 도메인 적응을 적용해도 조영-비조영 차이로 인해 경계·용적 오차가 커질 수 있다.

- **Core Contribution**: 본 연구는 조영 CT의 라벨만으로 비조영 CT를 생성(contrastive unpaired image translation)한 뒤 분할하는 ChameleonNet을 제안한다. 핵심은 CUT에 decoupled contrastive learning(DCL) loss를 결합해 비조영을 더 그럴듯하게 합성하고, 합성 이미지를 nnU-Net 기반 분할에 바로 연결한 점이다. 이후 Hausdorff distance loss를 강화해 경계 품질을 목표로 한다.

- **Technical Challenges**: 가장 큰 기술 과제는 조영 유무에 따른 영상 도메인 갭을 라벨 없는 unpaired 조건에서 안정적으로 메우는 것이다. 저자들은 CUT+ DCL로 비조영 합성의 일관성을 높이고, 분할기는 Hausdorff distance loss 강화 nnU-Net으로 경계 민감도를 끌어올리는 전략을 썼다. 합성 학습은 수만 장 슬라이스 수준(조영 35,538 vs 비조영 37,197)으로 진행하고, 분할기는 합성 비조영 스캔 292장을 사용했다.

- **Empirical Impact**: 합성 비조영 테스트에서 4개 챔버의 DSC는 LA 0.94, LV 0.91, RA 0.92, RV 0.93, HD95는 각각 약 3.63~5.74mm 수준으로 보고됐다. 실제 비조영 CT에서도 용적 경향을 반영하며 Pearson 상관은 0.82~0.93(모두 p<0.001)였지만, MAPE 9.22%~20.79%와 함께 LV·RV에서 용적 오차가 상대적으로 커 임상 적용 전 추가 정교화와 검증이 필요함을 시사한다.



### JupOtter: Cell-Level Bug Detection in Jupyter Notebooks (https://arxiv.org/abs/2606.23877)
Comments:
          Accepted at the 42nd International Conference on Software Maintenance and Evolution - ICSME 2026 (Research Papers Track)

- **Prior Approaches**: Jupyter Notebooks의 버그 탐지는 비선형 실행(임의 순서 실행)과 셀 구조 때문에 전통적인 스크립트 기반 디버깅/분석이 그대로 적용되기 어렵다. 기존 ML 기반 버그 검출은 주로 파일이나 함수 단위에 머물러 “어떤 셀이” 문제인지 알려주지 못하거나, 셀 간 의존성을 충분히 반영하기 어렵다는 한계가 있었다. 정적 분석도 라인 단위 매핑과 다중 오류 상황에서 오탐이 늘어나는 문제가 나타난다.

- **Core Contribution**: 논문은 셀 수준 구현 버그를 직접 탐지하도록 설계된 JupOtter를 제안한다. (1) 셀 경계를 보존하는 notebook-specific tokenization, (2) 실행 순서 고정 없이 각 셀에 대해 별도 결함 예측을 수행하는 cell-level bug prediction, (3) 세포 라벨을 포함하는 OtterDataset(21,303개 노트북)을 핵심 기여로 제시한다. 특히 OtterDataset은 공개된 대규모 셀 단위 라벨 데이터로 처음 시도됐다는 점에서 의미가 있다.

- **Technical Challenges**: 핵심 기술 과제는 (i) 긴 노트북을 잘라도 셀 경계와 구조 정보를 잃지 않고, (ii) 다중 청크로 나뉜 문맥에서 각 셀을 정확히 대표해 예측하며, (iii) 학습 시 GPU 메모리 한계를 넘기지 않는 것이다. JupOtter는 셀 경계 마커를 삽입한 뒤 청크(비중복 chunking) 단위로 인코딩하고, 경계 토큰 사이 토큰을 묶어 셀 임베딩을 만든 다음 셀마다 bug logit을 산출한다. 또한 노트북 청크의 메모리 부담을 줄이기 위해 mixed precision과 per-sample gradient 누적 방식의 multi-segment 학습 루프를 사용한다.

- **Empirical Impact**: 평가는 OtterDataset 보류분과 외부 데이터셋(Jupyter Errors, CodeParrot Jupyter subset)에서 진행되며, 정적 분석기와 대형 언어 모델 대비 셀 수준 성능을 비교한다. JupOtter는 셀 단위 F1에서 두 개 데이터셋에서는 정적 분석기 및 LLM보다 우수한 결과를 보였고, 한 개 데이터셋에서는 격차가 제한적이거나 조건에 따라 성능이 흔들릴 수 있음을 시사한다. 실행 전 탐지 관점에서 “버그가 있는 파일”이 아니라 “버그가 있는 셀”을 반환할 수 있게 되어, 노트북 디버깅 시간과 리소스 비용 절감에 직접적으로 기여할 것으로 기대된다.



### MGI: Member vs Generated Inferenc (https://arxiv.org/abs/2606.23872)
Comments:
          Accepted at ECCV 2026

- **Prior Approaches**: 기존 membership inference(MIA)는 “자연 훈련 샘플 vs 홀드아웃 자연 샘플”을 구분하도록 설계돼, likelihood/probability 같은 신호에 의존한다. 이미지 attribution은 “특정 생성 모델이 만들었는가”를 보려 하지만, 목표가 여전히 학습에 쓰인 멤버와 생성된 출력의 미묘한 경계를 다루지 못해 한계가 생긴다. 특히 현대 IGM의 생성 출력은 같은 잠재 분포에서 최적화되므로, 기존 방법이 가정한 신호 분리가 쉽게 붕괴된다.

- **Core Contribution**: 이 논문은 Member-vs-Generated Inference(MGI)라는 새 문제를 정의해, 주어진 샘플이 대상 생성 모델의 ‘진짜 훈련 멤버’인지 ‘그 모델이 만든 출력’인지 추론하는 프레임을 제시한다. 또한 image IGM에서 MIA/attribution이 각각 반대로 오분류(생성물을 멤버로, 멤버를 생성으로)하는 구조적 실패를 체계적으로 보인다. 이를 해결하기 위해 Data Circuit Breaker(DCB)를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 생성 출력이 멤버와 함께 likelihood 기반 신호를 공유해, 단일 확률/우도 통계만으로는 구분이 거의 불가능해진다는 점이다. DCB는 (1) autoencoder 재구성/양자화 기반의 self-consistency로 생성 가능성이 큰 샘플을 먼저 거르고, (2) 나머지에 대해 latent generator 기반 MIA를 적용해 멤버 판별 가정을 복원하며, (3) 여러 모델 버전에 걸친 cross-generator conditional log-probability 차이로 데이터 회로가 어디서 왔는지까지 추적한다.

- **Empirical Impact**: VAR-d30, RAR-XXL, LlamaGen-XXL 같은 이미지 autoregressive 모델과 Stable Diffusion 계열 diffusion 모델 전반에서 DCB가 기존 MIA/attribution의 실패를 일관되게 완화함을 보여준다. 특히 verbatim memorization(훈련 샘플을 거의 그대로 재생)에서도 near-duplicate 사이의 차이를 autoencoder/생성 파이프라인 잔차로 포착해 구분 성능을 유지한다. 더 나아가 생성 데이터를 학습에 재사용하는 model derivative 설정에서도 데이터 회로 붕괴 위험을 줄이려는 실질적 감사(audit) 도구로 의미가 크다.



### Are Safety Guarantees in Neural Networks Safe? How to Compute Trustworthy Robustness Certifications (https://arxiv.org/abs/2606.23858)
- **Prior Approaches**: 강화학습 안전성의 핵심 이슈로, 입력에 가해지는 아주 미세한 변화가 신경망의 분류를 깨는 adversarial examples가 알려져 있다. 이를 막기 위해 robustness certification(최대 허용 왜곡)을 계산하는 연구가 진행됐고, 기존에는 축 정렬 다차원 구간(하이퍼-직육면체)을 최적화해 품질을 키우려는 방식이 주류였다. 특히 제약된 convex 최적화를 수치적으로 푸는 접근은 성능은 좋지만, ReLU 비미분성 때문에 선형 완화(linear relaxation)가 동반되어 안전성을 과소평가(false positive)할 수 있다는 한계가 있다.

- **Core Contribution**: 이 논문은 하이퍼-직육면체의 부피(volume) 대신 최소 면까지의 ‘여유(slack)’를 나타내는 apothem measure를 도입해, 네트워크가 실제로 얼마나 멀리까지 안전한지 더 직접적으로 측정한다. 또한 NN verifier(oracle)에 반복 질의해 apothem-optimal robustness certification을 계산하는 절차(oracle 호출 수 선형 수준)를 제시하고, 클래스 전체를 포괄하는 dual certification을 도입해 다른 품질 지표(예: volume) 요구조건에 대해서도 상한/불가능성 판정을 가능하게 했다. 더 나아가 부피 기준의 volume-optimal을 oracle 기반 알고리즘으로 다항 시간에 달성하는 것은 불가능하다는 이론적 한계를 보인다.

- **Technical Challenges**: 문제는 ‘어떤 크기의 하이퍼-직육면체까지 안전한지’를 찾는 최적화가 단순 부피 최대화보다 계산적으로 어렵고, oracle 기반에서도 효율 보장이 까다롭다는 데 있다. 저자들은 특정 adversarial example를 구간에서 배제하는 constrain 연산을 정교하게 정의해 apothem 기준의 최적성을 유지하면서도 종료를 보장하도록 precision 상수 δ를 사용한다. 이어서 여러 adversarial example를 다루는 big-step 연산 및 dual certification 산출까지 연계해, 제한된 수의 oracle 호출로 apothem-최적 구간을 구성하는 전략을 만든다.

- **Empirical Impact**: MNIST와 Fashion MNIST에서 ParallelepipedoNN으로 실험한 결과, 기존 소프트웨어와의 예비 비교에서 minimum edge length 기준으로 최소 2배 개선, diameter 기준으로는 최소 한 자릿수(오더) 이상 개선을 보였다. 특히 부피 최적성 보장이 불가능하다는 이론과 결합해, apothem 기반으로 안전성 판단을 더 신뢰성 있게 만들고 dual certification으로 다른 품질 지표 요구를 보수적으로 점검할 수 있다는 점에서 실용적 의미가 크다.



### The Measurable Majority (https://arxiv.org/abs/2606.23853)
- **Prior Approaches**: 기존의 “대부분(most)” 같은 질적 다수 판단은 확률의 완전한 수치화보다는 약하지만, 그렇다고 단순한 순서/비교 주장보다 강해 표준 대표성 정리(representation result) 적용이 까다로웠다. 사회선택에서는 strict majority rule의 구조적 성격을 May-type 결과로 설명하려 했지만, 이 논문은 finiteness 하에서 다수 판단이 측도(measure)로 표현 가능한지의 조건을 더 근본적으로 따진다. 또한 Suppes의 weak qualitative probability 구조는 직관적으로 단순해 보였으나, 유한 경우에도 제시된 공리들이 representability를 보장하지 못함을 기존 작업의 반례(oddly even 프레임)로 드러낸다.

- **Core Contribution**: 이 논문은 social decision frames(사회적 의사결정 프레임)라는 형식을 도입해 “대부분” 판단을 구성론적으로 모델링하고, strict majority가 finitely additive measure로 표현되는 정확한 정당화 조건을 제시한다. 핵심은 프레임 자체만으로 판별 가능한 coherence(응집성) 기준이며, 이는 “표현 가능성”과 동치임을 보인다. 더 나아가 이 응집성에 기반한 최소 natural logic(자연 논리)로 “most of all(전부의 대부분)” 형태의 문장과 논리 연산을 다루며 soundness 및 complete(건전성/완전성)를 유한 모델에서 달성한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 측도/수치에 기대지 않고도 다수 판단이 일관되게 측도로 조정될 수 있는지를 “모든 유한 열(sequence)”에 대한 제약으로 정식화하는 것이었다. 이를 위해 coherence를 (1) 각 집합이 기준선인 half 이상인지, (2) 각 원소가 집합 열에서 지나치게 많이(half를 초과해) 포함되지 않는지, (3) half일 때/맞춰질 때의 등가 조건을 결합한 형태의 조건들로 구성했다. Suppes 정리의 실패는 특정 유한 조합(oddly even)에서 발생하며, 논문은 이 조합을 coherence가 정확히 배제하도록 연결해 논리 체계의 필요조건을 보강한다.

- **Empirical Impact**: 실험적 데이터라기보다, 논문은 조합적 구성과 논리적 성질로 결과의 “입증”을 진행한다: coherence ↔ measurability의 동치, 그리고 관련 term logic의 유한 model property 및 decidability를 수반하는 완전성 증명이다. 또한 May-type characterization을 유도해, symmetry(익명성/등등성) 같은 자연 조건 아래에서 coherence가 사실상 standard strict majority rule로 붕괴됨을 보여준다. 마지막으로 coherence 실패의 크기를 재는 incoherence index를 도입하고, incoherence의 조합론적 한계(축소 불가능성, 불일치 구성에 대한 부분 결과와 conjecture)를 제시해 향후 연구 방향을 구체화한다.



### Decentralized Coordination of Autonomous Traffic Through Advanced Air Mobility Corridors (https://arxiv.org/abs/2606.23832)
Comments:
          Presented at the AIAA SciTech 2026 Forum

- **Prior Approaches**: 기존 AAM(Advanced Air Mobility) 코리더 연구는 주로 코리더 네트워크 설계, 코리더 내부 충돌 회피, 또는 중앙집중형 MPC 같은 전략에 집중해 왔다. 또한 코리더 기반 운항은 중앙 교통관리 부재 시 비효율적일 수 있다고 보는 관점이 널리 퍼져 있었고, 분산/자율 접근도 제약이 적은 4D 궤적 최적화에 의존하는 경우가 많았다. 최근 MARL 시도도 merge/intersection 조정이나 단순화된 가정(예: 코리더 내 평행 통과) 위주라 코리더 경계 준수와 분리거리 제약을 함께 다루기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 분산 환경에서도 자율 고정익 항공기가 local information만으로 코리더 흐름을 스스로 self-organize하도록 학습할 수 있음을 보인다. 특히 centralized training/decentralized execution 형태로 InforMARL 계열을 확장해, 단일 코리더(출구 이후 메터링), 연속 코리더, 코리더 분기(2갈래)까지 포함하는 대표 시나리오를 다룬다. 코리더 경계에 대한 순응성과 목표 도착(메터링)까지 동시에 달성하는 것이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 난제는 (1) Dec-POMDP 성격의 제한된 관측에서 다중 에이전트가 충돌 없이 안전 분리 기준을 만족하면서, (2) 코리더 위상(진입-중간-출구) 전환을 올바른 순서로 수행하고, (3) 경계 준수와 효율을 함께 최적화하는 것이다. 저자들은 GNN 기반 그래프 관측과 다단계 코리더 phase 정보(진입/코리더/진출)를 포함해 에이전트 정책이 필요한 지역 정보를 교환·통합하도록 설계했다. 또한 분리거리 위반 패널티, 코리더 경계 관련 유도(phase 전환 보상), 목표 도착 보상을 조합한 reward로 학습 안정성을 확보했다.

- **Empirical Impact**: 시뮬레이션 결과 3가지 코리더 구성 모두에서 항공기는 코리더 경계를 94% 이상 준수하며, 정해진 에피소드 시간 내에 목표에 도달한다. 분리 최소치 위반 시 필요한 tactical intervention은 저·중밀도에서는 8% 미만으로 드물지만, 고밀도·분기 코리더에서는 약 17%로 늘어난다. 교통량이 증가하고 시나리오가 복잡해질수록 성공률은 낮아지는 경향이 확인되며, 이는 코리더 네트워크 운영에서 분산 자율학습의 실효성과 한계(혼잡 시 개입 필요성)를 동시에 보여준다.



### Deciphering Fingerprints of 3D Molecular Surfaces for Accurate Epitope Prediction (https://arxiv.org/abs/2606.23830)
- **Prior Approaches**: 기존 항원-항체 epitope 예측은 서열이나 백본 중심의 표현을 주로 써서, 전역적 단서는 잘 포착하지만 항원 표면에서 실제 결합을 좌우하는 미세하고 불연속적인 패턴을 정밀하게 국소화하기 어렵습니다. 또한 epitope는 결합 파트너(항체/리간드) 문맥에 따라 달라지는데, 기존 접근은 종종 partner-agnostic하게 처리되어 상호작용 의존성을 충분히 반영하지 못합니다. 이런 한계는 서로 다른 항체/항원 계열과 같은 out-of-distribution 상황에서 일반화 성능을 떨어뜨리는 요인이 됩니다.

- **Core Contribution**: SurfBind는 epitope 예측을 분자 표면(surface)을 1-class 모델링 대상으로 삼아, 기하(geometry)와 물성(physicochemical) 신호를 표면 표현 위에서 직접 결합하도록 설계된 surface-centric 학습 프레임워크입니다. 패치 단위 표면 모델링, binder-aware cross-attention, 그리고 coarse-to-fine 계층적 예측으로 상호작용에 맞춘 표면 특징 교환과 정밀 국소화를 동시에 노립니다. 결과적으로 “불연속 표면 기반 epitopes”를 보다 안정적으로 포착하는 것이 핵심 기여입니다.

- **Technical Challenges**: 표면은 기준 좌표계가 없고 점(점군) 분포가 희소·중복되는 경향이 있어, Transformer가 전역 문맥을 다루면서도 국소 결합 신호를 잃지 않게 만드는 것이 기술적 난제입니다. SurfBind는 farthest point sampling으로 불규칙한 로컬 패치를 만들고 Morton(Z-order)로 패치의 1D 순서를 부여해 효율적인 전역 추론을 가능하게 했으며, SurfFormer++로 장거리 패치 의존성을 geometry-aware attention(근사 지오데식 거리 기반 구조 임베딩)에서 처리합니다. 또한 리간드-수용체 간 결합 문맥은 binder-aware cross-attention으로 상호정보 교환을 수행하고, self-supervised 사전학습 단계에서는 discrete latent modeling(VQ 기반)과 다중 재구성(좌표·곡률·화학 특징) 목표를 결합해 상호작용 정렬된 표면 표현을 학습시키는 전략을 사용합니다.

- **Empirical Impact**: 실험은 SAbDab과 DB5.5 같은 까다로운 epitope 식별 벤치마크에서 SurfBind가 state-of-the-art 성능을 달성했으며, unseen 항체와 다양한 conformational state에 대해서도 강한 일반화 성능을 보였다고 보고합니다. 이는 “표면-상호작용을 입력의 중심에 둔 모델링”이 단순한 구조/서열 단서보다 불연속 epitope 탐지에 더 직접적으로 유리함을 시사합니다. 결과적으로 단백질-단백질 상호작용 인터페이스 인식 및 항체 설계/발견 파이프라인에 실질적 성능 향상과 해석 단서 제공 가능성을 보여줍니다.



### From Spatial to Spectral: An Efficient, Frequency-Guided Feature Representation Learner for Small Object Detection (https://arxiv.org/abs/2606.23825)
- **Prior Approaches**: 기존 소형 물체 탐지는 downsampling과 다중 스케일 융합 과정에서 작은 물체의 고주파 단서가 약화·소실되면서 성능이 급격히 떨어지는 문제가 컸습니다. 그래서 dilatation, 더 촘촘한 피라미드 등 공간 복원(특히 upscaling) 기반 처방이 흔했지만, 계산 비용이 크고 배경 잡음까지 함께 증폭할 수 있다는 한계가 지적됩니다. 주파수 도메인 접근도 분류 중심이거나 탐지에서 stage별 요구를 통일적으로 다루지 못해 통합 솔루션으로 자리잡기 어려웠습니다.

- **Core Contribution**: 이 논문은 공간 복원에서 벗어나, 탐지 파이프라인을 구성하는 backbone-neck-head 단계에서 필요한 주파수 단서를 보존·재주입하는 Frequency-Guided Feature Representation 패러다임을 제안합니다. 핵심은 Decompose--Enhance--Reconstruct(DER)라는 통합 연산자 인터페이스로, 이를 Wavelet-Difference Gate(WDG), Log-Gabor Enhancer(LGE), Frequency-Driven Head(FDHead)로 단계별 구현해 CNN/Transformer 모두에 플러그앤플레이로 적용 가능하게 했습니다. 결과적으로 해상도 감소와 무관하게 고주파 신호를 명시적으로 다루어 소형 물체의 경계·디테일을 복원하려는 접근이 정리됩니다.

- **Technical Challenges**: 해결해야 할 기술 과제는 (1) downsampling·fusion 같은 되돌리기 어려운 연산 전에 고주파 증거를 “복원/강화”하되 잡음 증폭을 피하는 것, (2) 이를 backbone/neck/head의 역할에 맞게 stage-wise로 설계해 서로 다른 탐지기에도 동일한 방식으로 꽂아 넣는 것입니다. 저자들은 WDG에서 wavelet 저주파 근사에 RepCDC로 경계 성분을 보강하고, 고주파 서브밴드를 self-derived gate로 활용해 low-frequency 오염을 억제한 뒤 inverse transform으로 재구성합니다. neck에서는 Log-Gabor 기반 고방향 잔차를 복원하도록 LGE를 설계하고, head에서는 wavelet 고주파 에너지로 box regression에만 boundary-sensitive 게인을 부여하는 FDHead를 통해 위치 회귀의 안정성을 높였습니다.

- **Empirical Impact**: VisDrone2019, UAVDT, TinyPerson, DOTAv1의 다중 벤치마크에서 DERNet 계열은 일관된 소폭~큰 폭의 성능 향상을 보였고, 특히 YOLOv11과 동일 스케일 조건에서 파라미터는 1/6 수준으로 줄이면서도 더 높은 mAP50을 달성했다고 보고합니다. 단계별 모듈 기여를 쪼갠 결과 FDHead가 가장 큰 향상을 주는 등, 설계 의도가 성능으로 연결됨이 확인됩니다. 또한 TIDE 기반 오차 분해와 2D FFT 기반 고주파 에너지 비율 분석에서 high-frequency 보존/경계 신호 정밀화가 Miss 및 Localization 오류 감소와 인과적으로 연결된다는 해석을 뒷받침하며, edge 하드웨어에서도 현실적인 FPS 저하 없이 효율-정확도 절충을 달성했다고 강조합니다.



### Ten Digits on a Train: AI-Assisted Verification of Two Eigenvalue Problems (https://arxiv.org/abs/2606.23821)
- **Prior Approaches**: 검증 가능한 수치해석(validated computation)은 interval analysis, a posteriori estimate, Taylor model, 고정점 테스트 등으로 ‘계산 결과에 증명 객체를 붙이는’ 전통이 있었다. 하지만 특이(singular)하거나 비정상(non-normal)인 연산자에서는 영값(고유값)·공명(resonance)을 충분히 정밀하게 “증명”하는 일이 특히 어렵다. 비정상 문제의 경우 기존 검증이 한 방향 사격(shooting)이나 유한구간 스칼라 판별식/argument principle에 의존하면 꼬리(tail)와 경로 의존 오차 때문에 분해·분리가 잘 안 되는 한계가 있었다.

- **Core Contribution**: 이 논문은 인간–AI 협업을 통해 두 종류의 스펙트럼 계산을 각각 10자리 수준으로 “인클로저(certificate)”화하는 사례를 제시한다. 특이 자가수반(singular self-adjoint) Schrödinger 연산자에서는 영점 개수 검증과 Dirichlet–Neumann bracketing으로 완전한 음의 스펙트럼을 10자리까지 봉인한다. 비정상 atom–molecule 벤치마크에서는 기존에 해결되지 못했던 공명 쌍을 분리해, 각 공명을 10자리까지 포획하는 공명 인증서를 만든다.

- **Technical Challenges**: 가장 큰 기술적 도전은 비정상성 때문에 한 번의 전파로 얻는 단일 스칼라(예: one-way determinant)가 꼬리와 경로 오차에 취약하다는 점이다. 이 문제를 해결하기 위해 저자는 ‘해를 한 줄짜리로 쏘는’ 방식 대신, 각 메쉬에서 해의 투영해(solution lines)를 projectivize하고 노드 좌표를 동시 미지수로 취급한 뒤 전역 matching(전역 매칭) 시스템을 세우고 이를 Krawczyk–Brouwer 포함으로 검증한다. 무한 꼬리는 단말 projective 데이터의 불확실성(집합값 최종행 perturbation)으로 인코딩하고, tail-robust한 componentwise Krawczyk–Brouwer inclusion이 다변수(polydisc)에서 필요한 성분별 조건까지 커버하도록 설계했다.

- **Empirical Impact**: 실험적으로는 AI가 빠르게 후보 해와 그럴듯한 증명 전략을 만들었지만, 최종적으로는 사람이 증명 객체를 감사(audit)하며 실패 경로(예: 꼬리 주장 누락으로 componentwise 체크가 빠진 경우)까지 드러났다고 보고한다. 그럼에도 공명 문제에서는 기존 2자리 수준의 봉인을 17자리급 certified enclosure로 끌어올리며, ‘정밀도 증가’가 아니라 ‘증명 객체의 재구성’이 성능을 좌우함을 보여준다. 저자는 이는 AI 보조 수학에서 출력은 숫자가 아니라 “숫자+증명”이어야 한다는 기준을 강화하는 재사용 가능한 아키텍처와 사회적 검증 표준의 필요성을 시사한다고 결론짓는다.



### From Task-Guided Conversational Graphs to Goal-Oriented Dialogue Runtimes (https://arxiv.org/abs/2606.23797)
Comments:
          21 pages, 7 figure, 10 tables

- **Prior Approaches**: 기존 LLM 오케스트레이션은 “다음에 어떤 에이전트/툴/노드를 실행할지”에 강하지만, 여러 목표가 동시에 살아 있는 대화에서 목표의 연속성(objective continuity)을 자동으로 보장하긴 어렵다. 프로세스 기반(워크플로 그래프, FSM)은 실행 위치의 연속성은 잘 유지하지만, 한 목표가 멈춘 채 다른 목표가 개입·무효화되는 상황까지 안정적으로 복원하기엔 한계가 있다. 또한 에이전트 라우팅이나 채팅 히스토리/실행 그래프 위치만으로는 “어떤 사용자 목표가 살아남는지”를 의미론적으로 구분하기 어렵다.

- **Core Contribution**: 이 논문은 Goal-Oriented Dialogue Runtime(GODR)을 제안하며, 프레임워크 중립적인 계층에서 목표(goals)와 생명주기(lifecycle), 태스크 프레임(task frames), 무효화 규칙(invalidation rules), 재개 계약(resumption contracts)을 런타임의 1급 객체로 취급한다. GODR은 그래프 런타임/에이전트/툴이 담당하는 “bounded execution”을 대체하지 않고, 대화 중단·지연·수정·무효화에도 목표가 끊기지 않게 만드는 “objective continuity” 전용 런타임 경계(boundary)를 명확히 한다. 특히 GC-4(목표 복잡도) 영역에서 필요하다고 주장하며, 단순 스택/트리 모델만으로는 공유 제약·의존·무효화가 얽힌 대화를 다루기 어렵다고 정리한다.

- **Technical Challenges**: 핵심 기술 과제는 “새 사용자 발화가 들어왔을 때 어떤 목표 수준 연산(continue/revise/push/switch/pop/resume/cancel/escalate/reset)을 수행할지”를, 실행 라우팅과 분리해 결정하는 것이다. 논문은 목표 구조를 라벨된 directed graph(DAG 포함)로 모델링하고, 재개 가능 상태는 resumption contract가 비어 있지 않다는 등 런타임 불변식(invariants)과 함께 무효화 안전성(invalidation safety)을 체크하도록 설계한다. 또한 Goal Policy가 LLM 라우터처럼 무제한으로 동작하지 않도록, typed state와 가드·심볼릭 제약으로 허용 가능한 연산 집합을 마스킹/필터링하는 하이브리드 신경-상징 접근을 제안한다.

- **Empirical Impact**: 이 논문은 실험 성능을 “검증된 수치”로 주장하기보다, Multi-Objective Interruptible Dialogue Problem을 정식화하고 평가를 위한 아젠다와 기준, 베이스라인 선택 방법론을 제시하는 시스템 논문 성격을 갖는다. 따라서 즉시 실측 개선을 보장하진 않지만, 기존 오케스트레이션 프레임워크가 강한 실행 계층과 달리 목표 연속성 계층이 애플리케이션 아키텍처에서 누락되기 쉽다는 문제의식을 명확히 한다. 목표 수명주기·무효화·재개 계약을 런타임으로 외재화한다는 관점은 향후 복잡한 멀티도메인 대화 벤치마크/평가 설계로 이어질 수 있는 의미가 있다.



### Integrated Sensing and Communications for Real-time Avatar Control in XR over 5G (https://arxiv.org/abs/2606.23771)
- **Prior Approaches**: 기존 XR 상호작용은 핸드헬드 컨트롤러나 카메라 기반 인식에 의존해 왔지만, 전신 포즈를 안정적으로 포착하기 어렵고 손의 자유로운 사용을 제한하며 시야/광량/가시성(LoS) 제약도 커진다. 5G mmWave ISAC는 기지국 신호로 기기 없이도 자세·동작을 추정할 수 있다는 장점이 있으나, 세밀한 손가락·핑거 관절 같은 미세 동작에는 공간 해상도가 부족하다는 한계가 지적돼 왔다.

- **Core Contribution**: 이 논문은 XR용 멀티모달 감지 아키텍처로 5G mmWave ISAC와 sEMG를 결합해, 동작을 ‘다중 스케일’로 인식하는 프레임워크를 제안한다. mmWave 쪽은 HMD 전송과 동시에 PPBP(Power-per-beam-pair)로 전신 단위의 거친 제스처/포즈를 실시간 제어에 활용하고, 손가락 단위의 미세 제스처는 저가·경량 sEMG 센서로 보완한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 통신 신호 기반 감지에서 사용자 간 일반화가 깨질 수 있고, (2) mmWave의 해상도 한계로 손가락 세부 동작을 RF만으로 구분하기 어렵다는 점이다. 저자들은 PPBP 입력을 원본 대신 프레임 간 차이(frame difference)로 변환해 DC 오프셋(정적 사용자 서명)을 제거함으로써 16.4pp까지 흔들리는 zero-shot 성능을 3pp 개선했으며, sEMG는 시야 비의존성(LoS 불필요)과 occlusion 강인성을 활용해 핀치/피스트 같은 손 제어 프리미티브를 학습에 반영한다.

- **Empirical Impact**: 실험 결과, PPBP 기반 전신 제스처 인식은 in-domain에서는 8명 사용자 평균 95.4% 정확도를 보였고, Leave-one-user-out에서는 평균 82.2±5.9%로 사용자 간 성능 분산(16.4pp)이 확인됐다. 프레임 차이 특징으로 전반 정확도를 3pp 끌어올렸고, sEMG 데이터(MultiSenseVR)에서는 핀치 대 피스트 분류에서 Random Forest가 최대 약 0.78 평균 정확도 수준의 일관된 성능을 보였다. 두 모달을 합치면 controller-free XR에서 전신 제어와 손가락 정밀 상호작용을 함께 아우르는 ‘완전한’ 제스처 파이프라인을 구현할 가능성을 실증적으로 제시한다.



### Cryptographic certificates of validity for trustworthy AI (https://arxiv.org/abs/2606.23768)
- **Prior Approaches**: 에이전트가 수행하는 행동의 정당성·정책 준수를 보장하려는 기존 방식은 주로 메시지 서명, 로그, 사후 모니터링에 의존한다. 그러나 서명/로그/감시는 ‘행동을 실행하기 전에’ 해당 행동이 correctness·safety·compliance 조건을 만족함을 증명하긴 어렵다.

- **Core Contribution**: 이 논문은 에이전트 행동에 대해 실행 전 cryptographic certificates of validity(유효성 증명서)를 부여하는 새로운 인증 패러다임을 제안한다. 정책(또는 correctness 조건)을 논리의 술어(predicate)로 명시한 뒤, 이를 polynomial constraints에 대한 유한 witness-checking 문제로 컴파일하고, succinct cryptographic proof(그리고 필요 시 zero-knowledge)를 통해 조건 만족을 증명한다.

- **Technical Challenges**: 핵심 난제는 논리적 조건을 cryptography 친화적인 형태로 ‘정확히’ 사상하고, 검증자가 에이전트의 계산을 신뢰하지 않아도 상호 독립적으로 확인 가능한 증거를 만들 수 있게 하는 것이다. 논문은 first-order logic의 validity를 polynomial 제약의 만족 판정으로 arithmetisation하는 수학적 번역 구조(논리 → polynomial constraint → cryptographic certificate)를 제시하며, soundness·completeness에 해당하는 타당성 관계를 정리한다.

- **Empirical Impact**: 구체적 실험 결과보다는, proof-carrying code와 zkVMs, formal methods 및 agent governance를 잇는 ‘검증 가능한 정책 증명’의 설계 아키텍처를 제공하는 데 의미가 있다. 또한 명세·감사(auditing)·배포(deployment)에서 어떤 질문을 답해야 완전한 구현이 가능한지까지 짚어, 에이전트 시스템의 사전 보증 메커니즘으로 확장될 여지를 제시한다.



### Emergent Relational Order in LLM Agent Societies: From Collective Affect to Authority Stratification (https://arxiv.org/abs/2606.23764)
Comments:
          Accepted to Findings of the Association for Computational Linguistics: ACL 2026. 37 pages

- **Prior Approaches**: 기존 사회과학의 Differential Order Pattern은 중국 농촌의 문화 특수성으로 해석되는 경우가 많았지만, 이를 작동시키는 ‘기계적 기제’가 충분히 실험적으로(계산적으로) 검증되진 못했다. LLM 기반 멀티에이전트 연구도 사회계약·규범·제도 형성 등은 다루되 대체로 단기 조정이나 이인(duadic) 수준에 집중해, 정서-윤리-권위-분업이 함께 진화하는 장기 사회구조 재현은 공백이었다.

- **Core Contribution**: 이 논문은 CAREB-MAS(COllective Affection–Reasoning–Emergence Based Multi-Agent Simulation)를 제안해, 문화 특수 규칙을 인코딩하지 않고도 Differential Order Pattern이 장기 시뮬레이션에서 ‘발현’될 수 있음을 보여준다. 에이전트는 Affect Control Theory·Social Identity Theory·뒤르껭적 집단 정서를 바탕으로 emotion–ethics–belief 체인을 순차 추론하며, 거시 환경은 개인 생산·선호 기반 할당·최소 상호작용 프로토콜만 제공한다. 그 결과 5가지 핵심 현상(안정적 분업, guanxi 기반 경제-윤리, 관계 거리별 협력 감쇠, 관계 기반 권위의 출현, 씨족 중심의 중심–주변부 층화)을 일관되게 재현한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘장기 사회구조’를 단순한 보상 설계나 규칙 기반 위계로 만들지 않고, 에이전트 내부의 정서·윤리·관계 인지 동역학이 상호작용을 통해 구조로 응집되는 경로를 구현하는 것이다. 이를 위해 augmented BDI에 Affect/윤리 모듈(EC·ER)과 SIM(사회적 정체성) 업데이트를 결합했고, 동적 커뮤니티 탐지·이동, 공개적인 제안/발표(투표·강제 규칙 없음), 제안 채택 빈도로 ‘emergent 권위’를 사후 계측하는 방식으로 설계했다. 또한 두 가지 생산 구조(대칭/상보 능력)와 모듈 제거(ablation)를 통해 결과가 단순 LLM 아티팩트가 아니라 아키텍처·구조 조건 의존적임을 확인했다.

- **Empirical Impact**: 30라운드·18에이전트 장기 시뮬레이션에서 분업 고착, 관계 거리 기반 협력 그라데이션, 권위의 우측치우침(중심–주변부)을 포함한 Differential Order 현상이 자발적으로 나타났다. 특히 생산 구조가 달라지면 통합 양상이 변해, 대칭 능력에서는 친족 중심의 기계적 연대에 가까운 형태가, 상보 능력에서는 기능적 상호의존이 커지며 더 ‘유기적 연대’에 가까운 패턴이 관측됐다. 모듈 제거 실험과 여러 LLM 간 교차 강건성 결과는 CAREB-MAS의 발현 메커니즘이 특정 모델 학습 편향에만 기대지 않으며, Differential Order를 일반 기제의 구조민감적 emergent outcome로 해석하는 데 실증적 근거를 제공한다.



### Listening makes Vision Clear for VLMs (https://arxiv.org/abs/2606.23763)
Comments:
          18pages,3 figures

- **Prior Approaches**: 기존 VLM 의미 일관성 평가는 answer-side 토큰의 attention 분포(또는 IoU 유사 마스크 겹침)를 통해 시각 영역과 언어 토큰의 정렬을 추정하는 방식이 주를 이뤘습니다. 하지만 가장 높은 attention이 항상 목표 의미 토큰과 일치하지 않는 사례가 관찰되며, 이는 이전에 생성된 토큰이 누적시키는 언어 prior로 인한 decoding drift 문제로 설명됩니다. 또한 modality boundary 같은 구조 토큰이 문맥 전체를 포괄하며 목표와 무관한 영역에 높은 attention을 유도할 수 있습니다.

- **Core Contribution**: 이 논문은 answer 생성에 의존하던 평가 관점을 prompt-side 의미로 전환해, 고정된 프롬프트 토큰을 대상으로 token-to-vision activation map을 뽑아 일관성을 진단하는 Prompt-Vision Token Activation Map(PV-TAM)을 제안합니다. 동시에 구조 토큰에서 기인한 공간 편향을 인접 구조 토큰으로부터 추정해 제거하는 denoising 필터를 설계해, 목표 의미 신호를 더 깨끗하게 분리합니다. 마지막으로 attention의 강도 분포까지 반영하도록 TGR, TDR, MinDist 같은 정렬 지표를 제시합니다.

- **Technical Challenges**: 핵심 난제는 자동회귀 생성 과정에서 answer-side attribution이 prefix 문맥 변화에 민감하게 흔들리며(디코딩 드리프트), attention 피크가 목표 토큰의 의미 증거와 어긋날 수 있다는 점입니다. 이를 해결하기 위해 PV-TAM은 목표 개념을 고정된 prompt 토큰으로 배치하고, 그 prompt 토큰에서 얻는 prompt-to-vision attention을 activation map으로 사용해 생성 접두사의 오염을 차단합니다. 이어 structural-token 주의를 제거하기 위해 인접한 특별 토큰들의 attention을 이용해 편향을 빼고(양수 증거만 유지하는 ReLU 기반 처리 포함), 선택적으로 foreground 마스크(rembg)로 배경 지배를 줄여 성능을 안정화합니다.

- **Empirical Impact**: PartImageNet과 RefCOCO 실험에서 PV-TAM은 기존 answer-side 해석/국소화 기준선(TAM, Grad-CAM, CP-LRP, Attention Rollout) 대비 TGR/TDR 개선과 MinDist 감소를 일관되게 보였습니다. 특히 Qwen 및 InternVL 계열의 여러 모델에서 유사한 이득이 재현되며, 작은 모델(예: InternVL3-1B)에서도 성능 우위가 나타났습니다. 또한 정성 시각화에서 PV-TAM은 목표 부품/표현의 경계에 더 정확히 집중하고 attention diffusion 실패를 줄이는 경향을 보였으며, “attention 정렬이 training 정렬 손실 없이도 자연 발생할 수 있다”는 관찰까지 제시합니다.



### Neuromorphic Speech Enhancement with Dual-Branch Spiking Neural Networks (https://arxiv.org/abs/2606.23761)
Comments:
          5 pages, 3 figures, 2 tables. Submitted to Interspeech 2026

- **Prior Approaches**: 기존 SNN 기반 speech enhancement은 이진 스파이크와 이벤트 기반 동역학으로 효율성을 노리지만, ANN 대비 이진 활성 한계와 설계가 덜 정교해 품질 격차가 남아 있었다. 특히 다수의 방법이 단일 스펙트럼 차원만 모델링해 magnitude와 complex spectrum(위상 포함)의 상보 정보를 충분히 활용하지 못했다. dual-path 같은 구조적 아이디어는 있었으나, SNN의 spiking dynamics가 어떤 신호 도메인(시간/주파수)에서 어떻게 맞물려야 하는지 체계적으로 다루지 못했다.

- **Core Contribution**: 이 논문은 dual-branch spiking neural network GSU-DBNet을 제안해 magnitude spectrum과 complex spectrum을 동시에 다루고 각각의 spectral mask를 예측한다. 또한 dual-path GSU 모듈로 주파수 경로에는 BiGSU(전역 스펙트럼 상관), 시간 경로에는 GSU(인과적 시간 의존)를 배치해 시공간 정보를 함께 학습한다. 복소 마스크는 DeepFilter 계수로 위상 복원을 지원하고, 크기 마스크는 에너지 엔벨로프 추정에 집중하며 두 출력을 가중 평균으로 융합한다.

- **Technical Challenges**: GSU 같은 spiking recurrent cell을 enhancement에 적용할 때 핵심 난점은 (1) 이진 출력으로 인한 정보 병목이 성능을 제한하는지, (2) 추가 게이트가 오히려 중복 파라미터만 늘릴지 불명확하다는 점이다. 이를 위해 forget gate만 유지한 단일-gate GSU를 기본으로 두고, SLSTM-2G/SLSTM-3G 같은 멀티-게이트 변형을 ablation해 게이트 복잡도가 성능을 개선하지 못함을 확인했다. 학습은 power-law 압축된 STFT spectral MSE와 시간영역 SI-SNR을 결합한 하이브리드 loss로 구성해 저에너지 구간 품질을 끌어올리도록 설계했다.

- **Empirical Impact**: VoiceBank+DEMAND에서 GSU-DBNet은 wideband PESQ 3.04를 달성했으며 파라미터는 394K에 그쳐 ANN 대비 4.5%–10.6% 수준의 작은 크기를 보인다. DPSNN 대비 PESQ가 0.84, Spiking-FSN 대비 0.38 향상되어 기존 SNN의 품질 격차를 실질적으로 줄였다는 점이 확인됐다. ablation에서는 이진 출력 병목 때문에 단일-gate GSU가 최적이며, magnitude/complex dual-branch와 dual-path(time/frequency) 구성이 모두 필수임을 보여 향후 경량 neuromorphic SE 설계에 방향성을 제공한다.



### Engineering Reliable Autonomous Systems: Challenges and Solutions (https://arxiv.org/abs/2606.23760)
- **Prior Approaches**: 기존 연구는 주로 전통적인 사이버-물리 시스템의 검증 관행을 자율 시스템에 그대로 이식하는 경향이 있었고, 복잡한 환경에서의 불확실성과 블랙박스 구성요소까지 충분히 다루지 못한다는 한계가 지적됐다. 또한 symbolic 방식은 상대적으로 검증이 쉬운 반면 sub-symbolic 방식은 확장성은 높지만 신뢰성 보장이 어렵고, neuro-symbolic로 간극을 메우려 해도 여전히 신뢰 문제는 미해결로 남아 있었다. 다중 로봇·재구성 시스템에서는 내부/외부 도전이 동시에 커지며, “검증 기법은 있으나 현장 적용이 낮은” 간극도 반복적으로 언급됐다.

- **Core Contribution**: ERAS(Engineering Reliable Autonomous Systems) 워크숍 보고서는 자율 시스템을 위한 verification & validation, 실세계 엔지니어링, 안전한 소프트웨어 아키텍처를 축으로 “도전 과제 카탈로그”와 “해결을 잇는 로드맵”을 제시한다. 특히 학계에선 알려져 있지만 실제 산업 적용으로 이어지지 못한 과제들과, 여전히 해결되지 않은 과제를 분리해 다음 연구·협업 경로를 구체화했다. 또한 FMAS(형식 방법)와 AREA(인지로보틱스/멀티에이전트)의 커뮤니티를 결합해, 서로가 갖는 사용사례·기법 이해의 공백을 메우려는 방향성을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 예측 불가능한 물리 환경과 black-box 부품을 모델링하는 문제, (2) AI 개발 방식(특히 sub-symbolic/학습 기반)으로 인해 생기는 신뢰성 격차, (3) 사람-로봇 상호작용·안전 요구를 보장할 증거를 설계하는 문제로 정리된다. 보고서는 이를 위해 정적 검증과 runtime verification을 함께 쓰는 접근, 아키텍처 수준 분해와 안전 아티팩트(예: GSN) 활용, 시뮬레이션·실험을 결합한 V&V 파이프라인(서로의 결과로 모델/요구사항을 보정)을 강조한다. 더불어 안전한 자율을 위해 안전 사례(safety case)를 구성하고, 테스트베드의 현실성·격리된 안전 모듈 등 실물 제약을 반영하는 방법을 연결해 제시한다.

- **Empirical Impact**: 워크숍은 11개 케이스 스터디를 통해 우주 능동 파지, 지하 광산 UAV, 의료 triage 의사결정지원, 산업 협업 로봇 안전거리 검증, 수중 케이블 추적, 자율 주행의 추가적 책임/설명 요구, 가정용 Care-O-bot 의사결정 검증, 인간 인수인계 협업(테이블 레그 handover), 건물 화재 분사, 원전 시설 루틴 점검, 전장 내(반)자율 협동 시나리오 등 다양한 도메인을 아우른다. 특히 runtime monitor로 주입 결함을 탐지하거나(우주 파지), GPS 부재·부분 관측 환경에서 fail-safe 자율의 수용성을 시험하며(광산 UAV/수중 AUV), 학습 구성요소와 닫힌 고리(closed-loop) 시스템을 함께 검증하는 사례가 제시됐다. 결과적으로 연구자와 산업이 “무엇이 아직 증거로 굳어지지 않았는지”를 공통 언어로 파악하고, 후속 연구·저널 스페셜 테마·후속 이벤트로 협업 모멘텀을 만들 수 있다는 점에서 의미가 크다.



### VeriPilot: An LLM-Powered Verilog Debugging Framework (https://arxiv.org/abs/2606.23759)
Comments:
          13 pages, 6 figures

- **Prior Approaches**: 기존 Verilog 디버깅은 컴파일 에러나 출력 웨이브폼 불일치 같은 거친 end-to-end 피드백에 의존해 반복적으로 코드를 수정하는 방식이 주를 이뤘습니다. 하지만 복잡한 RTL에서는 뿌리 원인이 관측된 출력과 멀리 떨어진 long dependency chain 형태로 나타나, LLM이 텍스트 기반으로 원인 추적을 수행하기 어렵습니다.

- **Core Contribution**: 논문은 VeriPilot을 제안하며, 단순 출력 비교를 넘어 golden reference model을 기준으로 내부 변수 의미(semantics)를 정렬한 뒤 그 정렬 결과를 디버깅 근거로 활용합니다. 또한 static analysis로 만든 Control-Data-Flow Graphs(CDFGs)를 바탕으로 단계별 signal tracing을 수행해 최소 의심 코드 영역과 정답 대응 로직을 함께 추출합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) RTL과 high-level golden model 사이의 변수 대응을 정확히 찾는 것과 (2) 의심 위치를 코드 전체 문맥이 아니라 구조적으로 좁혀 LLM 추론이 가능하게 만드는 것입니다. VeriPilot은 CDFG 기반으로 변수 정합을 bipartite matching(Hungarian) 형태로 수행하고, sequential state의 frontier와 combinational logic의 stop boundary를 통해 suspicious region을 논리 콘(cone) 단위로 정밀화하며, 이 증거를 LLM이 패치 생성에 그대로 쓰도록 구성합니다.

- **Empirical Impact**: NVIDIA의 CVDP 벤치마크에서 VeriPilot은 GPT-4o의 repair success rate를 54.3%에서 85.71%로 크게 끌어올리며 bug localization 정확도와 수리 효과를 동시에 개선했습니다. 즉, 테스트 출력 수준을 넘어 내부 의미 정렬과 구조적 추적을 결합한 접근이 복잡한 Verilog 설계에서도 신뢰성 있는 자동 RTL 디버깅으로 이어질 수 있음을 실증했습니다.



### Exploring Dualistic Meta-Learning to Enhance Domain Generalization in Open Set Scenarios (https://arxiv.org/abs/2606.23758)
- **Prior Approaches**: 도메인 일반화(DG)는 여러 소스 도메인으로 학습해 미지의 타깃 도메인에 적용하는 문제지만, 대부분은 소스와 타깃의 라벨 집합이 일치한다는 close set 가정에 기대 왔습니다. 오픈셋 도메인 일반화(OSDG)도 제안됐지만, one-vs-all 기반 방법은 양성/음성 샘플 불균형 때문에 결정 경계가 특정 쪽(양성)으로 치우치며, 그 결과 타깃에서 알려진 클래스까지 과도하게 거절하는 문제가 생깁니다. 메타러닝 기반 선행 연구는 주로 도메인 간 gradient 정렬에 초점을 두어, 클래스 간 관계가 필요한 오픈셋 상황을 충분히 다루지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 오픈셋 도메인 일반화에서 “도메인 쉬프트+카테고리 쉬프트”를 함께 다루기 위해 MEDIC(Dualistic MEta-learning with joint DomaIn-Class matching) 전략을 제안합니다. MEDIC은 도메인 분할뿐 아니라 클래스 단위 재조합을 통해 inter-domain과 inter-class gradient matching을 동시에 수행해, 알려진 클래스와 미지 클래스의 경계를 도메인 전반에서 균형 있게 학습하도록 설계됐습니다. 또한 MEDIC++로 일반화 프레임워크(다중 태스크/다중 스텝 스케줄링)로 확장해, 기존 MEDIC의 특수 설정을 더 탄탄한 베이스라인으로 다룹니다.

- **Technical Challenges**: 핵심 난제는 한 번의 최적화 업데이트가 다른 도메인/다른 클래스 과제의 학습 방향을 방해하는 gradient 간 간섭을 줄이면서, 계산 비용은 과도하게 늘리지 않는 것입니다. MEDIC은 명시적 2차 미분 정규화 대신 메타러닝의 implicit gradient matching을 사용하고, 도메인과 클래스가 동시에 엮인 meta-train/meta-test split을 설계해 gradient 정렬(동일/유사 방향)을 촉진합니다. 더 나아가 MEDIC++에서는 “한 스텝에 너무 적게(태스크별 통계가 비슷해져 구분이 약화)” 또는 “한 스텝에 너무 많이(페어링 단계가 사라져 gradient matching이 0에 가까움)”를 피하도록, 태스크 개수-스텝 수의 균형을 맞추는 스케줄링을 적용합니다.

- **Empirical Impact**: 실험 결과 MEDIC은 오픈셋 시나리오에서 기존 SOTA 방법들을 능가했으며, 동시에 conventional close set 성능도 경쟁 수준으로 유지했습니다. 이는 단순히 out-of-distribution을 “전부 미지”로 보내는 경직된 거절이 아니라, 타깃 도메인에서 알려진 클래스의 분류 정확도를 지키는 방향으로 경계가 조정됨을 시사합니다. 저자들은 MEDIC++의 일반화된 스케줄링이 gradient matching 효과를 실용적 학습 비용 하에서 유지할 수 있음을 보이며, 오픈셋 도메인 일반화의 표준적 접근으로 자리 잡을 잠재력을 제시합니다.



### Synergizing Physically Constrained MCMC and Chemical-Informed Gaussian Processes for Reaction Network Discovery (https://arxiv.org/abs/2606.23757)
- **Prior Approaches**: 기존에는 희소 회귀·상징적 회귀(예: PySR, SINDy)로 운동 법칙을 찾거나, MINLP/초구조 선택 등 결정론적 네트워크 복원을 시도했지만, 이러한 접근은 “반응 메커니즘”과 “현상학적 곡선맞춤”을 분리해 보장하기 어렵다. 또 Bayesian optimization에서 GP-BO처럼 실험을 블랙박스 표면으로 취급하면 물리적으로 불가능한 영역에 대한 제안이 생겨 비용과 안전성 문제가 발생한다. 특히 화학 반응 네트워크는 이산 토폴로지(반응 활성/비활성)와 연속 동역학 파라미터가 강하게 결합되어 불충분한 물리 제약은 수학적으로는 그럴듯하지만 물리적으로는 금지된 해를 남기기 쉽다.

- **Core Contribution**: 본 논문은 gray-box 워크플로 PC-MCMC-CIGP를 제안해, 희소·잡음 화학 시계열에서 해석 가능한 지배 방정식(반응 메커니즘)을 “물리 제약이 내장된 방식”으로 복원한다. 구조 추정은 mass/전하 보존과 detailed balance를 하드 제약으로 거르는 spike-and-slab 기반 PC-MCMC로 수행하고, 파라미터 캘리브레이션·잔차 보정은 ODE를 prior mean으로 쓰는 Chemical-Informed Gaussian Process(CIGP) 잔차 모델로 처리한다. 끝으로 uncertainty-aware acquisition으로 실험 설계를 자동화해, 단순한 성능 최적화가 아니라 “불확실성과 물리 타당성”을 함께 고려한다.

- **Technical Challenges**: 핵심 난제는 (1) 이산 토폴로지와 연속 파라미터의 강결합으로 인한 ill-posed 역문제와 (2) 수학적으로 허용되지만 물리적으로 불가능한 메커니즘이 탐색공간에서 살아남는 문제였다. 저자들은 PC-MCMC에서 effective rate 상수 마스킹과 spike-and-slab으로 구조적 희소성을 강제하고, 질량·전자 보존과 detailed balance를 위반하는 후보를 rejection sampling으로 사전에 제거해 posterior 자체를 물리적으로 정화한다. 이후 CIGP에서는 mechanistic ODE를 GP mean으로 두고 RBF 커널 잔차를 제한 복잡도로 학습하며, 실험 제안 단계에서는 EI류 기준에 물리 feasibility gate를 붙여 비현실적 영역의 제안을 부드럽게 차단한다.

- **Empirical Impact**: H2 + Br2 벤치마크에서는 constrained sampler가 라디칼의 elementary pathway를 구분하고, 수학적으로 가능한 “기만적 현상학적 핏” 또는 kinetically prohibited 직접 충돌 경로를 낮은 PIP로 배제하는 결과를 보였다. styrene epoxidation의 closed-loop 최적화에서는 PC-EI가 보고된 GP-BO baseline 대비 최종 수율을 12.5% 개선했고, 5회 내 고수율 구간 진입과 함께 수렴 가속(약 3.75배) 및 누적 regret 74% 감소가 관찰됐다. 또한 10-seed 비교에서 EI-style 기준은 최종 수율 성능이 강한 반면 PC-EI는 low-yield 제안을 크게 줄여 실험 효율과 안정성을 함께 개선하는 trade-off가 확인됐다.



### JEDEL: Zero-Shot DNA-Encoded Library Design for Early-Stage Drug Discovery (https://arxiv.org/abs/2606.23745)
- **Prior Approaches**: 기존 3D 기반 생성 모델은 약물 후보를 ‘가상 분자’로 뽑아내고, 이후 retrosynthetic 분석·벤더 탐색·경로 최적화 같은 합성 계획을 거치며 대량 후보를 버리는 경우가 많다. 따라서 출력은 실험 가능성과 직결되지 않아 DEL(Drug-Encoded Library) 제작 단계에서 비효율이 발생한다. 또한 단순 diversity 중심의 생성/열거는 타깃 상호작용 조건을 정밀하게 만족시키기 어렵고 샘플 효율도 낮다.

- **Core Contribution**: JEDEL은 3D pharmacophore(약물 구조의 상호작용 패턴)에서 바로 합성 가능한 DNA-encoded library(DEL)를 생성하는 프레임워크를 제안한다. 핵심은 pharmacophore의 기하학적 정렬을 ‘행동 가능한 합성 지시’로 매핑해, 모든 출력이 purchasable building blocks와 validated DEL 반응 스킴 안에서 구성되도록 합성-by-construction을 보장하는 것이다. 기존의 다운스트림 합성 계획 부담을 줄이면서도, 타깃별 재학습 없이 focused 라이브러리를 대규모로 설계할 수 있음을 보인다.

- **Technical Challenges**: 주요 기술 난제는 (1) 3D pharmacophore의 기하 제약을 학습하면서 (2) 약물 패턴과 분자 토폴로지의 many-to-many 대응을 예측하고 (3) 20만+ building block 전체에 대해 매 step softmax를 수행하지 않고도 합성 경로를 스케일 있게 디코딩하는 것이다. JEDEL은 P4-EGNN(회전·이동 대칭을 고려)으로 pharmacophore를 인코딩하고, JEPA 계열의 predictive alignment 학습으로 latent 공간을 정렬한 뒤, reaction role·반응 호환성·pharmacophore 프로파일 기반으로 building block을 클러스터링한 hierarchical decoder로 O(|V|)를 O(C+K)로 낮춰 10^6 단위 생성이 가능하게 했다. 결과적으로 pharmacophore 조건을 유지하면서도 합성 경로의 유효성을 아키텍처 차원에서 강제한다.

- **Empirical Impact**: 18개 단백질 타깃에 대해 JEDEL은 평균 docking energy에서 Enamine diversity baseline과 random sampling을 모두 앞서며, 타깃별로 대다수 단백질에서 최우수 평균 점수를 기록했다. 약물 패턴 측면에서도 pharmacophore recovery(P4)와 Jaccard similarity가 가장 높고, hit rate·sample efficiency에서 특히 어려운 타깃 구간에서 1자릿수 이상 개선이 나타났다. 또한 실험 단계로의 전환을 염두에 둔 ‘합성 실행 가능성’이 일관되게 보장되며, 기존 virtual molecule 중심 파이프라인에서 실험 deployable library design으로의 전환을 시사한다.



### Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation (https://arxiv.org/abs/2606.23743)
- **Prior Approaches**: 기존 비디오 diffusion 가속은 캐시, sparse attention/토큰 감축, 양자화, 커널 최적화 등 단일 기술군을 중심으로 성능을 끌어올리는 방식이 많았다. 하지만 (모델-하드웨어-서빙 설정) 조합마다 병목과 민감도가 달라져, 한 인스턴스에서 잘 된 레시피가 다른 배포로 잘 옮겨가지 않는 문제가 반복됐다.

- **Core Contribution**: 이 논문은 Sol Video Inference Engine을 통해 비디오 diffusion 추론 가속을 “훈련 없이” 인스턴스별 조합 최적화로 재정의한다. 특히 cache, sparse attention, token pruning, quantization, kernel fusion을 에이전트 기반으로 조합해 전 과정 속도-품질을 맞추는 풀스택 가속 스택을 제안한다.

- **Technical Challenges**: 가장 큰 기술 난제는 각 가속 기법이 품질 열화의 양상과 하드웨어/서빙 설정 의존성이 커서, 로컬 최적을 단순 결합하면 전역 성능이 무너질 수 있다는 점이다. 논문은 병렬 skill agent가 각 기법군을 로컬 튜닝한 뒤 integrator가 전역 조합을 탐색하고, human validator가 VBench 유사한 정량만으로 놓치기 쉬운 시각 품질을 피드백해 다음 라운드를 조절하는 구조로 이를 해결한다.

- **Empirical Impact**: Cosmos3-Super(64B), LTX-2.3(22B), SANA-Video(2B)에서 동일한 프레임워크로 배포 인스턴스를 최적화해 엔드투엔드 기준 2배 이상 가속을 달성하면서 VBench 품질을 “거의 무손실” 수준에 가깝게 유지했다고 보고한다. 결과적으로 단일 기법이 아니라 다기술 조합을 에이전트로 자동 구성하는 접근이 비디오 diffusion 가속의 실사용 비용을 낮출 수 있음을 실증한다.



### Low-power analogue neural networks with trainable nonlinear connections for continuous contro (https://arxiv.org/abs/2606.23742)
Comments:
          Preprint. Further verification of all simulations is ongoing. Any resulting corrections will be incorporated in a revised version

- **Prior Approaches**: 기존 physical neural networks는 아날로그 소자의 비선형 성질을 “스칼라 weight”에 흡수하도록 설계되는 경우가 많아, 소자의 물리 자원을 제대로 쓰지 못한다. 그 결과 연결 수 증가와 함께 제작·프로그래밍·제어 부담이 커지고, 복잡한 기기 응답을 단일 가중치로 납작하게 만드는 구조적 한계가 드러난다.
또한 physics-aware training, noise-aware dynamic optimisation 같은 학습 방법들은 주로 “이미 정해진 weight-activation 구조”에서 학습을 안정화하는 데 초점이 맞춰져, 아예 어떤 아키텍처가 하드웨어에 맞는지에 대한 질문은 상대적으로 덜 다뤄졌다.

- **Core Contribution**: 이 논문은 연결(edge)마다 스칼라 가중치 대신 학습 가능한 비선형 함수가 흐르도록 바꿔, 각 물리적 연결 자체를 표현력 있는 계산 단위로 만든 Physical Kolmogorov–Arn-inspired Network(PhyKAN)를 제안한다. 노드에서는 신호가 하드웨어 동작 범위를 벗어나지 않도록 고정 sigmoidal step을 두고, 연결의 비선형이 소자의 네이티브 응답을 그대로 계산 자원으로 사용하게 한다.
필터 기반(또는 memristor 기반)으로 edge 비선형을 구현했을 때의 “태스크 의존적” 이득이 왜 생기는지(연속·매끄러운 목표에 특히 유리함)를 정리하며, 결정 경계형 분류/이산 행동에서는 파라미터 효율 이점이 사라진다는 결론도 제시한다.

- **Technical Challenges**: 가장 큰 난제는 학습 가능한 비선형 edge를 실제 아날로그 하드웨어에서 구현하면서도, 학습에서 쓸 수 있는 미분가능한 모델과의 연결(transfer)이 안정적으로 성립해야 한다는 점이다. 논문은 필터의 전달함수(코너 주파수·이득)를 파라미터로 두고 gradient descent로 튜닝하되, hardware-achievable 값으로의 매핑을 위해 룩업 테이블과 straight-through estimator 등을 사용해 직통 전이를 견딜 수 있게 했다.
또 edge를 더 표현력 있게 만들기 위해 여러 필터 서브유닛을 병렬 합성하되 학습 중 pruning으로 불필요 유닛을 줄여, “유한 크기의 매끄러운 기저”가 최적화·일반화에 미치는 영향을 조절했다.

- **Empirical Impact**: 로봇 6축 kinematics(순전·역전), continuous-action reinforcement learning, photovoltaic maximum-power-point tracking에서 PhyKAN은 multilayer perceptron(MLP) 대비 훨씬 적은 노드·연결·학습 파라미터로 동등 또는 더 나은 성능을 보였다. 반대로 Fashion-MNIST 분류와 이산(±10) CartPole에서는 MLP와 거의 비슷하거나 이점이 거의 사라져, 효율 이득이 “연속값·회귀/연속 제어”에 맞물려 나타난다는 점을 실험적으로 확인했다.
또 약 3만 5천 개 수준의 연결에 대해 direct transfer를 수행했고, 시뮬레이션 대비 오차를 정량화하며 잔여 불일치는 기생 성분(parasitic)에서 주로 온다는 분석을 제시했다. 전력 측면에서는 전용 CMOS 추정으로 태스크급 photovoltaic 제어가 약 30μW 수준에서 동작할 것으로 전망되며, memristive 실현에서도 동일한 태스크 의존적 패턴이 재현돼 특정 기기보다 “연결에 trainable nonlinearity를 둔다”는 설계 원칙의 일반성을 뒷받침한다.



### A Survey on Federated Causal Discovery and Inferenc (https://arxiv.org/abs/2606.23741)
Comments:
          27 pages, 4 figures, 2 tables, journal

- **Prior Approaches**: 기존 연구는 federated learning(FL)을 활용해 분산된 데이터를 중앙집계 없이 학습하는 데 집중했지만, 이를 인과 추론까지 확장한 방식은 초기 단편 연구에 머물렀다. federated causal discovery(FCD)는 PC/FCI류의 조건독립 기반 접근이나 GES류의 점수-탐색 접근 등으로 나뉘어 왔으나, 분산 환경에서의 조건독립 검정 오류 전파, 연합된 그래프의 Markov equivalence class 한계, 통신·스케일 병목이 반복적으로 지적된다. 또한 federated causal inference(FCI)는 ATE/CATE 같은 목표 추정에서 다뤄지더라도 FCD와의 단계적 연결을 통합 파이프라인으로 정리한 체계적 정리 부족이 진입 장벽이 됐다.

- **Core Contribution**: 이 논문은 FCD/FCI를 함께 다루는 포괄적 survey로서, FCD를 구조 학습 방식·연합 토폴로지·구조적 범위라는 3축(방법론 패러다임 × federation topology × structural scope)으로, FCI를 estimand(평균 대 개인화/조건부 처리효과) × estimation strategy로 분류한다. 특히 FCD가 FCI에 필요한 구조적 지식을 공급하는 “통합된 federated causal reasoning pipeline”의 상보적 두 단계 관계를 형식적으로 정리해, 두 영역을 따로 보는 한계를 줄인다. 아울러 시간 동역학, data heterogeneity, missing data, 비일치 변수 집합(non-identical variable sets) 등 실전 차원의 공통 쟁점을 체계화한다.

- **Technical Challenges**: 분산 환경에서의 핵심 기술 난제는 (1) DAG의 비순환성 같은 전역 구조 제약을 연합 학습 중에 일관되게 만족시키는 일, (2) 사이트마다 인과 메커니즘과 변수 집합이 달라져 단순 pooling/meta-analysis가 깨지는 일, (3) 조건 집합·교란 가정·식별성(identifiability)을 요구하는 인과 문제를 서버-클라이언트 협업으로 재구성하는 일이다. 이를 위해 조건독립 기반 방법은 federated conditional independence testing을 설계해 전역 CI 결정을 복원하고(예: layer-wise aggregation, 안전한 통계 집계), 점수 기반 방법은 전역 점수/제약을 분산 최적화·ordering 탐색·regret 기반 프라이버시 프레임워크로 대응시킨다. 연속 최적화 계열은 NOTEARS 같은 연속화 관점에서 ADMM 분해, GSL/MA의 이원화, 서버의 전역 최적화 대체를 통해 acyclicity·희소성·분산화를 동시에 노린다.

- **Empirical Impact**: 논문은 방법별 프라이버시 메커니즘, 통신 효율, 이론적 보장(수렴/일관성/식별 관련 조건 포함), 그리고 실전 고려사항을 비교표와 체계적 정리로 제시해 연구자들이 빠르게 설계를 선택하도록 돕는다. 특히 federated causal discovery에서 DARLS의 oracle 수렴률 같은 형태의 첫 이론적 보장 사례와, regret 기반 프레임워크의 프라이버시-정확도 절충 분석이 대표적 기여로 조명된다. 결과적으로 FCD→FCI로 이어지는 통합 파이프라인 관점과 세밀한 분류 체계는, 향후 표준 벤치마크·이론 정합적 설계·현실 제약(결측/비일치 변수/시간성) 대응 연구의 방향성을 명확히 하는 데 의미가 크다.



### Weight-Space Geometry of Offline Reasoning Training (https://arxiv.org/abs/2606.23740)
Comments:
          accepted for ICML 2026 workshop

- **Prior Approaches**: 추론(reasoning) 증류에 Offline reinforcement-learning 계열 손실들이 널리 쓰이지만, 기존 평가는 거의 전적으로 벤치마크 정확도(pass@1 등) 비교에 그쳤다. 그래서 SFT에서 시작해 DPO/GRPO/DFT/RFT/RIFT로 손실을 바꿀 때 모델 파라미터가 실제로 어떤 ‘업데이트 방향’으로 이동하는지, 혹은 같은 메커니즘으로 수렴하는지 불명확했다. 또한 “offline RL”이 단일 메커니즘인지, 손실군마다 고유한 기하학이 있는지 해석 연구에도 빈틈이 있었다.

- **Core Contribution**: 논문은 동일 수학 롤아웃과 동일 베이스 모델(Qwen3-4B) 위에서 SFT, RFT, DFT, RIFT, Offline GRPO, DPO를 모두 attention-only LoRA로 학습한 뒤, LoRA 델타 ΔW의 가중치 공간 기하를 비교한다. 특히 코사인 유사도, principal-angle 서브스페이스 분석, linear mode connectivity, CKA 등으로 “업데이트가 비슷한 방향/같은 basin에 머무는가, 아니면 근본적으로 다른 모드로 이동하는가”를 기계적으로 판별한다. 그 결과 보상가중 reward-weighted 손실군은 대체로 같은 방향으로 수렴하는 반면, DFT와 DPO는 상대적으로 다른 업데이트 양상을 보인다는 결론을 제시한다.

- **Technical Challenges**: 기계적 구분을 위해서는 손실 함수 차이뿐 아니라 LoRA 초기화(seed)와 학습률 같은 학습 조건이 델타의 방향을 교란하지 않도록 통제하는 것이 핵심 난제다. 논문은 동일 롤아웃 통제 후에도 seed 변동과 learning-rate 변동이 델타의 방향에 미치는 영향을 분리해, 특히 DPO는 코드베이스 관례로 10× smaller learning rate를 사용해 업데이트 크기-정확도 차이가 손실과 옵티마이저가 함께 얽힐 수 있음을 명시한다. 해결책으로 ΔW 자체의 코사인(게이지 불변)을 중심으로 서브스페이스 각도와 연결성 장벽(barrier)을 함께 측정해 “방향성/기저 basin”을 동시에 보였다.

- **Empirical Impact**: 실험에서 SFT·RFT·RIFT는 전역 LoRA 델타 코사인 유사도가 0.97 이상이고, top-1 principal angle 중앙값이 약 7도로 매우 정렬되어 GSM8K 정확도도 87~88%대로 서로 유의미한 차이가 없었다. 반면 DFT는 SFT/RIFT와 코사인 약 0.55 수준으로 더 크게 방향이 갈라졌고, Offline GRPO는 SFT 방향에 대해 전역 약 67%가 직교 성분이면서도 SFT/RIFT loss basin 안에 머무는 중간 양상을 보였다. DPO는 SFT와 거의 직교하는 서브스페이스에 놓이며 linear mode connectivity에서 장벽이 관측되는 동시에 GSM8K 93.5%, AIME26 30.0%로 최고 성능을 달성했으나, 10× smaller learning rate로 인해 causal 분리는 후속 연구로 남겼다.



### A Unified Framework for Runtime Verification and Model-Based Diagnosis in LOLA (https://arxiv.org/abs/2606.23720)
- **Prior Approaches**: 기존 runtime verification(RV)은 시스템이 사양을 위반하는지 “감지”하는 데 강하지만, 진단(model-based diagnosis, MBD)은 주로 위반 이후 특정 시점에서 “원인 부품 국소화”를 수행하는 식으로 절차가 분리돼 있었다. LTL 기반 RV 모니터와 MBD를 결합한 초기 시도도 최소 이상 부품 집합을 계산하긴 했지만, 모니터링과 진단이 여전히 별도 도구·단계로 돌아가는 한계가 컸다. stream 기반 RV(Lola 계열)는 온라인으로 판정을 내릴 수 있지만, MBD 의미의 부품 이상 상태까지 같은 틀에서 직접 연결하는 통합은 부족했다.

- **Core Contribution**: 이 논문은 stream specification language LOLA 안에서 runtime verification과 model-based diagnosis를 하나의 통합 프레임워크로 묶는다. 시스템 설명, 컴포넌트 health(정상/비정상 상태), 관측을 동일한 stream 기반 형식으로 인코딩해, fault detection과 동시에 온라인 fault localization을 수행한다. 또한 time-invariant fault뿐 아니라 transient fault, 그리고 nondeterministic 관측까지 자연스럽게 지원하도록 설계했다.

- **Technical Challenges**: 통합의 핵심 난제는 “온라인 스트림 판정”의 형태로 MBD의 최소 진단(부품 이상 상태 가정)을 매 시점 계산해야 한다는 점이다. 저자들은 Lola에서 생성되는 제약을 SMT solver와 진단 계산 알고리즘에 결합해, 관측되는 시스템 동작에 대해 가능한 진단들을 스트림 상에서 암묵적으로 구성하도록 했다. 더 나아가 time-invariant을 위한 multi-instant diagnosis와, 이상 상태가 시간에 따라 변하는 temporal diagnosis 이론을 제시해 transient fault의 발생 시점까지 다룬다.

- **Empirical Impact**: 저자들은 제안 프레임워크를 프로토타입 구현하고 ISCAS85/ISCAS89 벤치마크 회로 두 개에 대해 실현 가능성을 검증했으며, 초기 성능 지표도 제공했다. 결과적으로 하나의 Lola 기반 파이프라인에서 다양한 진단 시나리오(배경지식, 추상 모델링, fault model/유형 등)를 처리할 수 있다는 점에서 실용성이 강조된다. RV와 MBD의 엄밀한 이론적 결합을 통해, 라이브 모니터링 환경에서의 온라인 복구·점검 의사결정에 직접 기여할 수 있는 방향성을 제시한다.



### Legal Reasoning Is Not Lawyering: Rethinking Legal Benchmarks for Pro Se Access to Justic (https://arxiv.org/abs/2606.23716)
Comments:
          Both authors contributed equally. Accepted to the AI4Law Workshop at ICML 2026

- **Prior Approaches**: 기존 법률 AI 벤치마크는 LegalBench, LEXam, LegalBench-RAG처럼 변호사 매개 입력에 대해 법률 추론을 평가하는 데 집중해 왔다. 따라서 사실이 법적으로 정리된 형태로 제공되고 절차적 국면도 명확해, 모델이 “이미 잘 다듬어진 질문”에서 보이는 상한 성능을 주로 측정한다.
이때 중요한 직무인 법yering(법적 서류 작성에 필요한 사실 선별·구조화·절차 맥락 정리)은 벤치마크 밖에서 전제되거나 가려진다.

- **Core Contribution**: 이 논문은 접근성 향상(access to justice)을 주장하려면, 단순 추론 정확도뿐 아니라 pro se(자기대리) 입력에서의 견고성(robustness)을 하한(lower bound)으로 직접 측정해야 한다고 제안한다. 즉, 변호사 없이 당사자가 작성한 프롬프트의 잡음, 누락, 서술 순서 문제, 민간 법상식(folk-legal assumptions) 같은 결함이 모델 성능을 어떻게 흔드는지를 벤치마크가 반영해야 한다고 주장한다.
LEXam의 소규모 실험을 통해 기존 벤치마크가 상한을 주로 보여주며, pro se 유사 조건에서는 성능이 더 크게 떨어질 수 있음을 구체화한다.

- **Technical Challenges**: 핵심 technical challenge는 pro se 입력 결함이 LLM의 대표적 취약점(롱컨텍스트 민감도, underspecification에서의 침묵 거부 실패/가정 생성, hallucination, 그리고 오타·문장 교란)과 구조적으로 맞물린다는 점이다. 논문은 이 결함들을 벤치마크 실험에 맞춰 (1) 단일 문자 수준 typo 주입, (2) 불필요한 문장 삽입으로 컨텍스트 희석(padding) 같은 두 축의 perturbation으로 구현한다.
그 결과, 단순 오타만으로도 모델별 성능 저하와 순위 변동 가능성(rank instability)이 나타나고, 컨텍스트가 희석되면 관련 정보를 찾는 능력도 흔들려 추론 성능이 감소한다.

- **Empirical Impact**: 실험은 LEXam 객관식 100문항 표본에서 GPT-4.1-mini, GPT-4.1-nano, GPT-4o-mini 3개 모델을 대상으로 왜곡 전/후 정답 수 정확도를 비교해 성능 저하를 관찰했다. 특히 “잘못된 답을 자신 있게 내는 것”과 “답을 보류하는 abstention”의 문제를 다룰 여지는 남기되, 적어도 정답 산출 단계에서 pro se 유사 결함이 전반적 성능을 떨어뜨린다는 신호를 제공한다.
저자들은 이 갭이 측정되지 않으면 접근성 향상 주장이 경험적으로 검증 불가능해지며, 상한만 개선되고 하한이 정체·악화되어 불평등이 커질 수도 있다고 경고한다.



### Random coloured digraphs defined by a Markov logic network (https://arxiv.org/abs/2606.23715)
- **Prior Approaches**: 기존 Statistical Relational AI에서 Markov logic network(MLN)는 닫힌 도메인 가정을 벗어나기 위해 PRM/리프트형 모델로 다뤄졌지만, 도메인 크기가 바뀌면 질의(1st-order 문장) 성질의 확률이 흔들리는 문제가 반복해서 지적돼 왔다. 이를 완화하려고 projectivity(도메인 크기 전이의 정확한 일치)나 weight-scaling(가중치의 도메인 크기 의존 스케일링) 같은 접근이 제안됐지만, first-order로 표현되는 일반 질의의 “안정적 극한”을 폭넓게 보장하는 일반 이론은 부족했다.

- **Core Contribution**: 이 논문은 특정 MLN 언어(단항 P(x)와 이항 R(x,y))에서, 도메인 크기 n이 커질 때 임의의 first-order sentence φ의 진리 확률이 안정적으로 수렴/발산하는지(0-1 법칙, 수렴 법칙)를 정밀 분석한다. 특히 가중치를 1/n로 스케일하면 어떤 가중치 선택에도 불구하고 모든 first-order sentence에 대해 확률이 극한에서 0 또는 1로 수렴하며, 그 극한 값은 가중치에 의존하지 않는다는 강한 “0-1 law”를 제시한다.

- **Technical Challenges**: 핵심 난점은 “ground formula에서 오는 에너지”와 “가능한 구조 수에서 오는 엔트로피”가 n에 따라 서로 다른 속도로 성장해, 가중치가 비스케일일 때는 사건 확률이 가중치별로 여러 거동으로 갈라질 수 있다는 점이다. 저자들은 coloured directed graph(점 색 P, 간선 관계 R)로 모델을 정형화하고, (unscaled) 표준 MLN에서는 가중치에 의해 7개 질적 레짐(phase transition 포함)으로 나뉘며, some regime에서는 0-1 law가 성립하지 않더라도 convergence law가 나타날 수 있음을 엄밀히 분류했다. 반대로 (scaled) 1/n 스케일에서는 엔트로피-에너지 균형이 안정화되어 색 비율이 엔드포인트 0/1에서 멀어지고, 그 결과 모든 first-order sentence에 대해 0-1 법칙과 가중치 비의존 극한을 동시에 얻는다.

- **Empirical Impact**: 이 결과는 “가중치가 학습된 도메인 외삽(extrapolation)에서 무의미해질지/여전히 영향이 남을지”를 first-order 수준에서 정량적으로 규명해, 큰 도메인으로의 추론 안정성에 직접적인 기준을 제공한다. unscaled 설정에서는 가중치에 따라 sudden phase transition과 결합된 7가지 장기 거동이 나타나 예측이 불안정해질 수 있음을 보여주고, scaled 설정에서는 어떤 질의도 가중치에 흔들리지 않는 0-1 극한으로 정리돼 inference 설계(예: weight-scaling의 필요성과 효과)를 뒷받침한다. 즉, MLN을 대규모 도메인으로 적용할 때 “어떤 스케일링을 쓸 것인가”가 논리적 질의의 장기 확률을 좌우한다는 실질적 함의를 제공한다.



### Audio-visual Contrastive Alignment for Diffusion-based Visual-conditioned Speech Enhancemen (https://arxiv.org/abs/2606.23712)
- **Prior Approaches**: 오디오-비주얼 음성 향상(AVSE)은 입 모양 같은 시각 단서를 활용해 잡음 환경에서 발화를 복원한다. 최근에는 cross-attention으로 시각 특징을 조건화한 diffusion 기반 unsupervised AVSE가 제안됐지만, 융합 단계에서 cross-modal alignment를 얼마나 명시적으로 강제하는지에 따른 효과는 불명확했다.

- **Core Contribution**: 이 논문은 diffusion 학습 목적함수에 contrastive audio-visual loss를 추가해, 모델이 시각 정보를 더 강하게 활용하도록 유도한다. 다만 posterior sampling 프레임워크 자체는 그대로 두어 기존 방식의 장점은 유지하면서 정렬 문제만 보완하는 데 초점을 둔다.

- **Technical Challenges**: 핵심 과제는 시각-청각 정렬을 더 강하게 만들면서도 posterior sampling 기반의 생성/복원 흐름을 깨지 않는 것이다. 저자들은 cross-attention 기반 조건화를 유지한 채 학습 시에만 contrastive loss를 결합해, 융합의 효율을 높이도록 설계했다고 밝힌다.

- **Empirical Impact**: matched 및 mismatched 테스트 데이터 모두에서 간섭 억제, 신호 재구성, 지각 품질이 일관되게 개선됐고, 특히 low SNR에서 성능 향상이 가장 컸다. 이는 diffusion-우선 데이터 구동 prior에 정렬 유도 손실을 결합하는 접근이 AVSE의 실사용 조건(고잡음)에서 체감 성능을 끌어올릴 수 있음을 시사한다.



### Coordinate-Queryable Neural Field Reconstruction for EEG Spatial Super-Resolution with Unseen-Electrode Generation (https://arxiv.org/abs/2606.23707)
- **Prior Approaches**: 기존 EEG 공간 초해상(EEGSR)은 미리 정해진 저채널→고채널 고정 매핑을 학습(end-to-end)하거나, GAN·diffusion 등 생성모델로 신호 분포를 보강하는 방식이 주류였다. 이 접근들은 특정 실험용 전극 레이아웃과 채널 인벤토리에 강하게 결합돼, 테스트 시 랜덤 채널 누락이나 전극 품질 변동이 생기면 흔들리기 쉽다. 또한 보간 기반 방법은 지오메트리 인지를 제공하지만 국소 고주파 구조를 과도하게 매끄럽게 만들 수 있어, 학습에서 노출되지 않은 전극 좌표로의 생성에 유연하지 않다는 한계가 있었다.

- **Core Contribution**: 논문은 EEGSR을 “고정된 출력 텐서 복원”이 아니라, 부분 관측 support 채널로부터 “좌표를 질의하면 답을 내는 coordinate-queryable scalp field”를 학습하는 문제로 재정의한다. 그 결과, 변동하는 채널 누락 복원과 strict unseen-electrode 신호 생성(학습 중 support로 쓰이지 않은 전극 위치)을 하나의 프레임에서 동시에 다룰 수 있게 된다. 모델은 position-guided encoder로 관측된 신호와 전극 좌표를 조건(latent condition)으로 만들고, conditional INR decoder가 원하는 좌표에서 신호를 재구성한다.

- **Technical Challenges**: 주요 기술적 난제는 테스트마다 보이는 채널 집합이 달라져, 단순히 채널 인덱스 상관으로 복원 규칙을 학습하면 안 된다는 점이었다. 이를 위해 encoder가 관측 채널들의 신호 컨텍스트와 좌표를 함께 요약해 조건을 만들고, decoder는 좌표 질의로 재구성을 수행하도록 설계해 레이아웃 고정 의존을 줄였다. 또 conditional INR이 관측 evidence에 충분히 고정되지 않고 드리프트할 위험을 줄이기 위해 fidelity-preserving channel corruption training을 도입해 missing·corrupted·visible-but-predicted의 혼합 상태로 일관성 제약을 강화했다.

- **Empirical Impact**: SEED, AAD, BCI2000에서 실험한 결과, ScalpINR은 랜덤 채널 누락 복원에서 전반적으로 최상 성능을 보이며 특히 support ratio가 낮을 때 격차가 커졌다. 더 나아가 AAD의 strict held-out-electrode 설정에서 학습 중 support로 노출되지 않은 전극 위치를 생성하는 능력을 직접 입증했는데, NMSE는 37.5% 감소, SNR은 2.12 dB 개선(강한 베이스라인 대비)을 기록했다. 신호 레벨뿐 아니라 주파수/스펙트럼 보존과 downstream 분류 정확도까지 함께 향상돼, 좌표 질의형 스케일프 필드 접근이 실제 가변 채널 환경에서 유연성과 견고성을 제공할 수 있음을 보여준다.



### Event-Aligned Analysis of Multi-Rater Pain Assessments Using Continuous Wearable Physiology (https://arxiv.org/abs/2606.23705)
Comments:
          6 pages, 3 figures. Accepted at IEEE EMBC 2026 (Toronto, Canada, July 26-30, 2026)

- **Prior Approaches**: 기존 웨어러블 기반 통증 분석은 EDA, HR/HRV, PPG, 피부온도, 움직임 같은 생리 신호로 통증 강도를 추정하는 데 집중했지만, 대부분 통증을 단일 연속 라벨(사실상 single ground-truth)로 취급한다는 한계가 있었다. 또한 통증 평가는 환자·간호사·의사 등 관찰자에 따라 달라지는데도, 이 차이를 annotation noise로 흡수하는 경우가 많았다. 시간 정렬 측면에서도 통증 평가는 sparse·irregular한 반면 생리 신호는 연속이어서, 고정 윈도잉이나 보간으로 처리하되 통증을 ‘변화 이벤트’로 보지 않는 접근이 주를 이뤘다.

- **Core Contribution**: 이 논문은 rater-aware(평가자 인지) 방식으로 통증 점수를 그대로 라벨로 다루지 않고, 평가자의 정체성을 유지한 채 통증 변화(change)를 이산 이벤트로 변환하는 framework를 제안한다. 이어서 연속 웨어러블 생리 신호를 해당 이벤트 시점에 event-aligned로 정렬해, 평가자별로 이벤트 전 동역학을 직접 비교할 수 있게 한다. 이를 통해 통증-생리 관계가 평가자마다 달라질 수 있음을 정면으로 탐색한다.

- **Technical Challenges**: 핵심 기술적 난제는 sparse한 NRS 통증 평점을 연속 생리 신호와 어떻게 정렬하고, 평가자 차이를 모델링하면서도 재현 가능한 이벤트로 정의하느냐였다. 연구진은 각 평가자별 연속 점수 간 절댓값 차이가 임계값(δ=1)을 넘을 때를 통증 증가 이벤트로 정의하고, 이벤트 전 15초~후 30초 구간을 고정 윈도우로 잘라 표준화한 뒤 event-centered 평균과 이벤트 전 기울기(slope) 특징을 계산했다. 생리 이벤트가 충분히 덮이지 않는 경우는 제외해 정렬 일관성을 확보했으며, 예측 목적이 아닌 해석을 위해 평가자 그룹별 선형 모델로 신호 기여도를 비교했다.

- **Empirical Impact**: 척추 시술 맥락의 다중 평가자(환자·간호사·의사) 동기 데이터에서 평가자 간 통증 변화 식별은 전반적으로 크게 불일치했으며, 특히 환자-의사 간 격차가 가장 컸다. 또한 이벤트 전 EDA와 HR의 전개가 평가자에 따라 달랐는데, 의사 보고 통증 증가는 더 강하고 일관된 자율신경 신호 상승(특히 EDA)과 HR 증가 경향과 연관된 반면 환자·간호사는 더 약하거나 변동적인 양상이 관찰됐다. 다만 시술 단계 교란 요인, 표본 수(n=24)와 이벤트 수(평가자당 41–47), 반복측정 구조 때문에 결과는 탐색적이며, 향후 mixed-effects 등 엄밀한 검증이 필요하다고 제시한다.



### Heterogeneous 2D/1D Signal Representation Fusion for Underwater Acoustic Modulation Recognition Under Distribution Shif (https://arxiv.org/abs/2606.23702)
- **Prior Approaches**: 기존 수중 acoustic modulation recognition은 waveform에서 추출한 시간-주파수(STFT), cyclostationary 같은 2D 표현이나 higher-order power spectra 같은 1D 통계량에 의존하는 단일 뷰/주도 뷰가 많았다. 그러나 분포 이동(SNR 저하, 환경/채널 불일치, 통신 파라미터 변화)에서는 각 모달리티의 신뢰도가 고르게 떨어지지 않아, 단순 결합 방식이 오히려 성능을 흔드는 문제가 있었다. 또한 robustness를 공정하게 비교할 수 있는 unified 평가 프로토콜이 부족해, 서로 다른 shift 유형을 분리해 검증하기 어려웠다.

- **Core Contribution**: 이 논문은 UAMR-ShiftBench를 제안하며, underwater acoustic modulation recognition에서 in-distribution, low-SNR, unseen-environment, unseen-communication-parameter, 그리고 simulation-to-real(측정 해상 시험)까지 단일 matched 프로토콜로 묶어 shift disentanglement을 가능하게 했다. 동시에 SCP-TriCA를 제안하는데, STFT와 cyclostationary는 bidirectional cross-attention으로 먼저 정렬·통합하고, P2/P4(2차·4차 power spectra)는 sample-adaptive selective gate로 신뢰도에 따라 단계적으로 주입한다. 이를 통해 “분포 이동에서 모달리티 신뢰도가 비대칭으로 변한다”는 관찰을 아키텍처 설계에 직접 반영했다.

- **Technical Challenges**: 핵심 기술 난제는 분포 이동 하에서 모달리티별 열화 정도가 달라, 대칭적인 multimodal fusion(예: 단순 feature concatenation)이 어느 한 모달리티의 잡음/왜곡을 그대로 최종 판단으로 끌고 갈 수 있다는 점이다. SCP-TriCA는 2D 모달리티 간 cross-attention으로 먼저 공통 토큰 기하를 맞추고, 1D 통계 토큰(P2/P4)은 두 번째 단계에서 게이트로 제어해 통계 증거가 유효할 때만 정교한 보강이 되도록 설계했다. 또한 STFT·cyclostationary는 시간-주파수/주기성 패턴을, P2/P4는 고차 통계 단서를 담당하도록 입력 모달리티를 일관된 캐리어 대역 정보(f0, B) 기반으로 구성했다.

- **Empirical Impact**: UAMR-ShiftBench에서 SCP-TriCA는 in-distribution 정확도 95.33%, simulated OOD 평균 74.59%를 달성해 최강 baseline 대비 각각 5.12%p 향상된 결과를 보였다. 더 나아가 March/November 두 독립 sea-trial subset에서 91.14%와 94.86%를 기록했으며, 각 baseline 대비 15.71%p와 23.00%p 초과 성능을 달성했다. ablation은 성능 향상의 원인이 모달리티 보완성과 계층적 fusion 설계에 있음을 확인하며, 실측 데이터 기반 simulation-to-real gap 평가의 가치도 함께 부각된다.



### Evaluating LLM Usage for Efficient and Explainable Numerical and Classified Implicit Sentiment Analysis of Product Desirability (https://arxiv.org/abs/2606.23701)
Comments:
          20 pages, 6 figures, 11 tables. arXiv admin note: text overlap with arXiv:2408.01527

- **Prior Approaches**: 기존 감성 분석은 대체로 리뷰 점수나 명시적 라벨에 의존해 범주형(긍정/부정) 분류 중심으로 작동했지만, PDT 같은 정성 데이터의 ‘감성 강도(수치)’를 안정적으로 뽑기 어렵다는 한계가 있었다. 또한 사전 기반(lexicon-based) 방법이나 특정 도메인에 맞춘 transformer 기반 기법들은 PDT 응답의 맥락 의존적·암묵적 뉘앙스를 충분히 반영하지 못해 신뢰할 만한 정량 성능을 내기 힘들었다. LLM을 쓰더라도 과거에는 암묵 감성 분석에서 도메인 특화나 추가 학습 없이 바로 쓰면 성능이 흔들리곤 했고, 예측 근거의 해석 가능성도 문제로 지적돼 왔다.

- **Core Contribution**: 이 논문은 PDT(Product Desirability Toolkit)에서 얻은 5개 단어-설명(PDT Respondent Term Grouping)을 그대로 LLM에 넣어, 명시적 리뷰 점수 없이도 제품 desirability(선호/바람직함)를 수치 점수와 범주형 감성으로 동시에 정량화하는 zero-shot 프레임워크를 제안한다. 특히 LLM의 confidence rating과 사람이 읽을 수 있는 rationale(explainability/xAI)을 함께 산출해 결과의 투명성과 실무적 신뢰를 높이도록 설계했다. ZORQ와 CARMA 두 PDT 데이터셋(총 106개 응답 묶음)을 대상으로, 전문가 라벨과의 정합성을 체계적으로 비교한다.

- **Technical Challenges**: 핵심 기술 난제는 정성 서술에 내재된 암묵적 감성의 ‘강도’를 일관된 수치로 환산하는 것과, 모델이 데이터가 제시되는 형식이 달라도(여러 형태로 입력) 성능을 유지하는 견고성을 확보하는 데 있었다. 저자들은 PDT의 구조화된 단어-설명 입력을 LLM에 넣고, zero-shot 연속 수치 스코어링과 다중 범주 분류를 함께 수행하는 파이프라인으로 해결했다. 더불어 모델 confidence와 rationale을 결과에 포함해, 단순 정확도뿐 아니라 해석 가능성과 투명성까지 함께 검증 가능한 형태로 제공한다.

- **Empirical Impact**: 실험 결과, LLM은 정성 응답에서 수치 감성 점수를 생성했을 때 전문가 라벨과 Pearson 상관이 최대 0.97까지 도달했으며, 범주형 분류 정확도도 최대 94%를 기록했다. 반면 lexicon 기반 및 transformer 기준선은 통계적으로 유의미한 성능을 내지 못했다. 또한 GPT-4o-mini는 더 큰 모델과 유사한 94% 수준 성능을 더 낮은 비용으로 달성해 대규모·지속적 제품 만족도 분석에 적합함을 보여주었고, confidence와 rationale 포함으로 해석 가능성과 현업 적용성(신뢰/투명성)을 강화한다.



### Self-Recognition Finetuning can Prevent and Reverse Emergent Misalignmen (https://arxiv.org/abs/2606.23700)
Comments:
          18 pages, 11 figures

- **Prior Approaches**: Emergent misalignment(EM)은 특정 페르소나 벡터의 활성화와 ‘악한’ 특성의 동반으로 설명되어 왔지만, 기존 대응은 주로 학습 중(in-training) 방어에 의존해 개입 방식이 제한적이었다. 또 많은 방법들이 EM을 직접적인 유해 콘텐츠 학습으로 취급해, 왜 정렬된 캐릭터가 흔들리며 문제가 커지는지에 대한 기계적 관점이 약했다.

- **Core Contribution**: 본 연구는 EM을 “정렬된 캐릭터의 붕괴/교란”으로 재정의하고, 이를 겨냥한 character-targeted 개입으로 self-generated text recognition(SGTR) fine-tuning을 제안한다. SGTR은 기존 방어처럼 학습 과정에서 차단하는 방식이 아니라, 캐릭터 복원과 강화를 목표로 하는 파인튜닝 개입이라는 점에서 구별된다.

- **Technical Challenges**: 핵심 기술 과제는 (1) EM이 손상시키는 능력을 정확히 복원할 수 있는지, (2) 예방(prevention)에서 단일 지표 악화 없이 misalignment만 줄일 수 있는지 검증하는 것이었다. 저자들은 3개 모델(GPT-4.1, Qwen2.5-32B-Instruct, Seed-OSS-36B-Instruct)과 여러 EM 데이터셋에 대해 2-stage fine-tuning을 설계하고, SGTR을 도메인 정답 데이터/일반 지식/word counting 같은 benign fine-tuning 기준선과 체계적으로 비교했다.

- **Empirical Impact**: 실험 결과 EM reversal에서는 모든 개입이 비슷한 수준의 역전 효과를 보였지만, “EM이 저하시킨 능력 복원”이 있을 때에만 차별적인 개선이 나타났다. EM prevention에서는 SGTR만 일관되게 misalignment를 낮추면서 개별 메트릭을 악화시키지 않았고, 추가 분석으로 EM과 LLM의 기본 identity self-report(자기 정체성 보고) 간의 연결(다양성 유도, 자기 인식 교란의 악화, identity-bearing system prompt 제거 시 효과 감소)을 근거로 제시해 방어 설계의 방향성을 강화했다.



### FP8 is All You Need (Part 2): Efficient Ozaki-Bailey Style FFT Through Tensor-core Garner Reformulation and Kulisch Escape Rou (https://arxiv.org/abs/2606.23698)
Comments:
          There is an accompanying Part (1) paper also submitted to arXiv:2606.06510

- **Prior Approaches**: 기존 TME 분석에서는 FP64 연산 파이프가 약한 GPU에서도 Ozaki-II 에뮬레이션이 메모리 바운드 구간에서 FP64-equivalent 처리량을 복원할 수 있다고 봤다. 다만 FFT처럼 Bailey 분해에서 내부 차원 k가 작아지는 경우에는 Garner reconstruction 지연(γ)이 커져 오버헤드가 지배항이 될 수 있어, 기존 방식만으로는 한계가 드러났다.

- **Core Contribution**: 이 논문은 Ozaki-II에 Bailey six-step 분해를 결합한 Ozaki-Bailey FFT 커널을 제시하고, FFT에서 TME의 핵심 병목인 γ-roof를 정량화한다. 특히 Garner를 Phase A(텐서 코어 기반 FP8/INT8 내적 배치)와 Phase B(출력별 감소/재구성)로 쪼갠 뒤, Phase B를 Kulisch complete arithmetic로 소프트웨어적 ‘escape route’로 재구성한다.

- **Technical Challenges**: 문제는 Bailey-FFT에서 k≈sqrt(N)로 인해 γ가 계산·메모리 항을 넘어 바인딩 제약이 되는 점이다. 저자들은 먼저 Garner를 forward CRT 및 mantissa-sliced 계수로 변환해 Phase A를 FP8/INT8 tensor cores에 태우고, Phase B는 Kulisch의 fixed-point complete arithmetic로 INT32 SIMT 파이프에서 완전 정밀도를 유지하도록 구성했다.

- **Empirical Impact**: 대역폭-패리티 관점에서 네 가지 closed-form floor(네이티브 FP64, naive-Ozaki, Kulisch INT32, Phase A FP8)를 도출했으며, B300의 1.3TF FP64 성능은 네이티브 floor에는 못 미치지만 Kulisch 두 조건(특히 INT32 sub-floor)은 만족한다고 분석한다. 1024^3 전체 FP64 기준 예상 wall time은 약 18ms로 메모리 roof(약 12.9ms)에 거의 근접하며, 결과가 실측에서 유지되면 FP64-collapsed GPU에서도 소프트웨어만으로 full-FP64 FFT가 ‘viable’해지고 libKulisch 및 벤치마크 확장이 촉진될 전망이다.



### SemChunk-C: Semantic Segmentation for C Cod (https://arxiv.org/abs/2606.23697)
Comments:
          7 pages, 9 tables, 2 figures

- **Prior Approaches**: 기존 C-family 코드 chunking은 고정 길이 윈도우, 휴리스틱 분할, Tree-sitter 같은 AST 기반 규칙에 의존하는 경우가 많았다. 하지만 매크로 확장과 문법 복잡성 때문에 AST가 깨지거나, 분할이 의미 있는 기능 단위를 잘 포착하지 못해 검색과 RAG 등 다운스트림 성능이 제한됐다.

- **Core Contribution**: 이 논문은 C-family 언어용 semantic chunking을 위해 코드 chunk 범주(category) 체계를 정의하고, LLM 기반 토큰 분류로 chunk 경계와 각 chunk의 기능 속성을 함께 예측한다. 또한 SemChunk-C라는 17M~150M 파라미터의 경량 encoder 모델 묶음을 제안해, 중첩 정의와 매크로 같은 현실적 난제를 견딜 수 있는 분할을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난제는 C-family 코드의 들쭉날쭉한 구조에서 ‘의미 단위’ 경계를 유연하게 찾고, 동시에 chunk 내부를 12개 범주로 일관되게 라벨링하는 것이다. 저자들은 Ettin encoders를 코드 데이터로 추가 사전학습하고, Qwen2.5-Coder-32B-Instruct로 자동 생성한 (chunk, attribute) 데이터로 2단계 토큰 분류(경계/내부 범주) 파인튜닝을 수행해 경계를 더 정확히 맞추도록 했다.

- **Empirical Impact**: 수작업 및 자동 테스트셋에서 경계 탐지 정확도와 범주 분류 성능을 확인했으며, 특히 작은 17M 모델이 큰 모델 수준의 결과를 보이는 경향을 보고한다. RepoQA의 검색(SNF)과 YABloCo의 함수 수준 코드 생성에서 chunking 컨텍스트가 전반적으로 개선을 주었고, Tree-sitter 및 더 큰 LLM 기반 컨텍스트를 상회하거나 견줄 때도 있었다. 또한 더 적은 chunk 수를 생성해 벡터 DB 검색 비용을 줄이고, GPU/CPU 추론 시간 측면에서도 작은 모델이 실용적임을 제시한다.



### Quantifying Prior Dominance in RAG Systems (https://arxiv.org/abs/2606.23695)
Comments:
          15 pages, Preprint

- **Prior Approaches**: 기존 RAG 평가는 Exact Match(F1) 같은 이산 휴리스틱에 크게 의존하거나, LLM-as-a-judge 방식처럼 다른 블랙박스 모델의 판정에 의존하는 경우가 많았습니다. 그 결과 모델이 검색된 맥락에서 정보를 ‘추출’했는지, 아니면 파라메트릭 메모리로 답을 ‘회상’했는지 구분이 어려운 epistemic blindness 문제가 남습니다.

- **Core Contribution**: 이 논문은 Normalized Context Utilization(NCU)라는 연속형 지표를 제안해, 검색 맥락이 예측 불확실성을 얼마나 줄였는지를 토큰 log-probability 기반으로 정량화합니다. zero-shot(쿼리만)·oracle(정답 포함 맥락)·adversarial(정답 대체) 조건을 함께 두고, 엄격한 factual extraction(Chain-of-Thought 미사용) 워크플로에서 contextual 정보이득을 분리 평가합니다.

- **Technical Challenges**: 모델마다 토크나이저와 어휘 구성이 달라 log-probability 합을 그대로 비교하면 불공정해지는 문제가 있어, 길이 정규화로 확률 공간을 표준화합니다. 또한 posterior가 prior보다 불확실성을 더 키우는 negative transfer까지 포착하도록 NCU를 경계값 처리·스무딩과 함께 설계하고, 우회적으로 단순 정확도보다 미세한 confidence 변화를 추적합니다.

- **Empirical Impact**: 실험 결과 1.5B~72B 범위에서 엄격 추출 과제는 스케일 증가에 따른 이득이 크게 둔화되며, 작은 모델(예: 1.5B)은 oracle 정확도·NCU가 큰 모델과 동급이거나 더 낫고 지연시간은 상용 API 대비 대폭 절감됩니다. 또한 proprietary 상용 모델은 adversarial 충돌에서 절반가량 맥락을 무시하는 Prior Dominance 경향과 함께, 상반된 priors를 만났을 때 confidence collapse로 negative transfer 위험이 더 높게 나타났고 이는 SLM보다 구조적으로 취약함을 시사합니다.



### Beyond the Autoregressive Horizon: A Comprehensive Survey of Diffusion Models, World Modelling, and State Space Models for Cod (https://arxiv.org/abs/2606.23690)
Comments:
          14 Pages, 1 Table, 1 Figure

- **Prior Approaches**: 기존 자동 소프트웨어 엔지니어링은 대부분 autoregressive(AR) next-token 예측에 의존해 왔다. 이 방식은 좌→우 순차 생성 특성 때문에 오류가 되돌려 수정되지 못하는 Sequential Dependency Trap, 긴 코드의 문법/의존성 유지의 어려움, 그리고 실행 의미론(semantics)과의 단절이라는 한계를 동반한다.

- **Core Contribution**: 이 논문은 코드 모델링에서 AR 편향에서 벗어나 Diffusion Models, Code World Models(CWMs), State Space Models(SSMs) 같은 비(非)autoregressive 패러다임을 구조적으로 정리하는 초기형 설문(survey)을 제시한다. 또한 이들의 아키텍처적 성질을 인간의 “System 2”식 추론(의도적 분석과 오류 수정) 관점으로 연결해, 코드 지능 에이전트의 다음 방향을 제안한다.

- **Technical Challenges**: Diffusion 기반 접근은 이산(discrete) 코드 토큰을 위한 discrete diffusion 설계가 핵심이며, 병렬 디코딩과 iterative refinement로 전역 문법 제약과 장거리 구조를 학습하도록 구성한다. SSM 기반 접근은 변환기처럼 self-attention에 의존하지 않으면서도 긴 문맥을 선형 시간으로 처리해야 하는데, HiPPO/S4 계열과 Mamba 같은 선택적(state dependent) 전이를 통해 긴 컨텍스트 효율과 표현력을 동시에 노린다.

- **Empirical Impact**: 논문은 Diffusion이 코드 생성·수리·유닛 테스트 생성·취약점 탐지 등에서 AR 대비 정확도/효율 이점을 보일 수 있고, 코드 편집은 denoising 기반으로 last-mile repair에 가까운 국소 수정 메커니즘을 제공한다고 정리한다. 더 나아가 SSM과 CWM 같은 대안들은 대규모 저장소(repository) 수준 이해와 실행 트레이스 reasoning처럼 장문·장거리 의존이 큰 작업에서 실용적 가능성을 높이며, 코드 에이전트의 “System 2”형 설계를 촉진하는 의미가 있다.



### Reentrant value fields as delayed coupled reaction-diffusion systems on finite graphs (https://arxiv.org/abs/2605.03940)
- **Prior Approaches**: 기존 연구는 기호(semantic/symbolic) 처리 모듈과 기하(geometric) 모듈을 분리해 학습하거나, 두 표현을 단순 결합해 end-to-end로 최적화하는 방식이 많았다. 하지만 지연된 상호작용이 포함될 때 시스템이 well-posed(해의 존재·유일·연속의존)인지, 안정성이 delay와 무관한지에 대한 엄밀한 조건이 부족했다.

- **Core Contribution**: 이 논문은 기호 필드와 기하 필드를 bipartite Hilbert-Schmidt kernel로 결합한 dynamical system을 제시하고, 이를 지연 히스토리 공간에서의 retarded functional differential equation(RFDE)로 완전히 기술한다. 또한 principal subsystem(H_L, X_R, P)이 interfield coupling의 small-gain 조건 C_{K}^2 < μ_L μ_R를 만족하면 delay와 무관하게 전역 안정(global stability)임을 보인다.

- **Technical Challenges**: 핵심 기술 난제는 지연 결합이 포함된 RFDE에서 Lipschitz 조건과 small-gain 조건을 만족시키면서, 그래프 라플라시안 기반 확산/반응이 결합되어도 해공간의 양의 불변성(viability)을 유지하는 것이다. 논문은 연산자-노름이 유한한 Hilbert-Schmidt 커널, 대칭(또는 symmetrized) conductance Laplacian의 스펙트럴 gap 가정, 그리고 경계에서 외향 속도를 제어하기 위한 projection onto Frobenius balls 같은 설계 명세로 이를 해결한다.

- **Empirical Impact**: 실험 중심 논문이라기보다, constant input에서 RFDE의 well-posed성과 compact global attractor(컴팩트 전역 끌개) 존재를 보장하는 이론적 설계를 제공해 신경망 모듈을 안정화하는 실무 지침을 제공한다. 특히 transformer/graph-neural 모듈이 곧바로 이 정리의 가정을 만족하지 못한다는 점을 지적하고, residual backbone, task-dependent selectivity, 투영/포화 같은 메커니즘으로 안정 후보를 구성하는 로드맵을 제시한다.



### Repeated Shared Access Enables Grokking, but Edit Propagation Depends on an Addressable Memory (https://arxiv.org/abs/2606.20737)
Comments:
          35 pages, 4 figures, 22 tables

- **Prior Approaches**: 기존 연구는 지식그래프 QA에서 factual edit(사실 수정)이 합성 추론을 통해 전파되는지 관찰하며, 보통 편집 자체나 특정 네트워크 구조를 강조해 왔다. 하지만 dense transformer는 OOD 2-hop 합성에서 near chance로 실패해, “편집이 전파되는 조건”을 명확히 분해하지 못했다는 한계가 있다. 또한 loop recurrence나 shared-memory를 각각 도입했을 때 성공/실패를 보고하지만, 두 성공 경로의 공통 원인과 분기 축은 충분히 정리되지 않았다.

- **Core Contribution**: 이 논문은 합성 단일-사실 편집이 2-hop으로 propagation되는 메커니즘을, 제어된 synthetic knowledge-graph QA 실험으로 원인 분해하는 “메커니즘 연구”를 제시한다. 특히 2x2 조건에서 loop recurrence(반복 계산)와 shared-memory access(공유 메모리 접근)를 교차시켜, 두 성공 경로가 학습(grokking)과 편집 전파(edit propagation)에서 요구하는 것이 다르다는 점을 분리한다. 결론적으로 편집 전파의 핵심은 루프의 반복 계산 자체가 아니라, forward computation이 ‘쓰기’한 뒤 ‘다시 읽을 수 있는’ addressable memory의 유무에 달려 있음을 주장한다.

- **Technical Challenges**: 문제는 (1) 반복적 공유 접근이 학습에서는 동일하게 유도되더라도, (2) 편집 전파는 왜 달라지는지 구체적 기여 요소를 분해해야 한다는 점이다. 이를 위해 저자들은 OOD grokking barrier를 함께 넘는 4개 모델(Dense, Loop, Dense+Mem, LMC)을 만들고, 단일 국소 편집을 direct success 조건으로 적용한 뒤 pre-edit-correct 기반에서 2-hop 전파를 측정한다. 또한 store를 N=128에서 N=13으로 coarsening해도 메모리/비메모리 분리가 유지되는지 확인해, fine granularity가 affordance를 만들기보다 정밀도를 보강한다는 관점을 뒷받침한다.

- **Empirical Impact**: 실험에서 편집 전파는 shared memory가 있는 셀에서 크게 나타났고, LMC와 Dense+Mem은 각각 대략 0.78-0.92, 0.71-0.96 범위로 강하게 전파된 반면 Loop와 Dense는 0.04-0.30, 0.00-0.03 수준에 그쳤다. 즉 전파 성능은 loop axis가 아니라 memory axis를 따라 갈리며, 메모리를 가진 두 변형 간에는 유의미한 차이를 보이지 않았다. 저자들은 Dense+Mem이 recurrence가 없는데도 전파가 강하다는 점을 근거로 “주소 가능한 메모리에 대한 쓰기-재읽기”가 편집 전파의 실질적 affordance임을 강조하며, learning competence와 editing affordance를 분리해 해석하는 데 영향이 크다.



New uploads on arXiv(cs.RO)

### InSight: Self-Guided Skill Acquisition via Steerable VLAs (https://arxiv.org/abs/2606.24884)
Comments:
          Project website: this https URL

- **Prior Approaches**: VLA 모델은 언어와 비전을 바탕으로 조작을 학습하지만, 성능은 학습 데이터에 포함된 스킬 범위에 갇히는 한계가 있었다. STEER/Steerable Policies처럼 steerability를 제공하는 연구도 원시(primitives) 집합을 고정된 것으로 취급해, 누락된 동작을 새로 배우는 “continual skill acquisition”까지는 확장되지 않았다. 또한 SayCan, VoxPoser, Code-as-Policies 같은 VLM/LLM 기반 접근은 테스트 시 조합이나 계획에 강하지만, 새로운 primitive를 실제 정책 학습에 추가해 어휘를 늘리는 방식은 아니었다.

- **Core Contribution**: InSight는 VLA를 primitive 단위로 steerable하게 만들고, VLM이 새 태스크에 필요한 primitive “gap”을 찾아 자동으로 획득·재학습함으로써 기술 확장을 가능하게 한다. 핵심은 (1) 시연 영상을 VLM 분해와 엔드이펙터 포즈 기반으로 자동 segmentation해 labeled primitive로 변환하고, (2) VLM-guided data flywheel로 누락 primitive를 저수준 제어 파라미터까지 포함해 시도→성공 시 학습 데이터에 통합→VLA를 retraining하는 폐루프를 구축한 점이다. 그 결과 사람의 target-skill 시연 없이도 장기 과제를 새로운 조합으로 실행할 수 있다.

- **Technical Challenges**: 가장 큰 기술 과제는 (a) 수작업 없이 시연을 primitive 경계와 라벨로 정확히 쪼개 steerable 인터페이스를 만드는 것, (b) primitive gap이 무엇인지 판별한 뒤, 해당 gap에 맞는 단일축(translation/rotation axis) 기반 저수준 제어 파라미터를 VLM이 생성·일관되게 적용하는 것이었다. InSight는 VLM plan decomposition과 엔드이펙터 축 태그/포즈·운동량 정보를 결합해 경계를 자동으로 확정하고, 학습된 progress channel로 primitive 종료 신호를 제공한다. 또한 OOD 초기 상태나 종료 오차가 생길 때는 VLM completion check로 primitive 전이를 보정해 누락된 조작을 안정적으로 획득하도록 설계했다.

- **Empirical Impact**: InSight는 시뮬레이션과 하드웨어에서 block flipping, drawer closing, sweeping, twisting, pouring 등 5개 과제를 평가하며, 목표 스킬에 대한 추가 인간 시연 없이도 새로운 primitive를 얻고 이를 조합해 long-horizon 태스크를 수행했다. 예컨대 pour와 twist는 primitive 수준 신뢰도가 누적되며 end-to-end success가 Code-as-Policies 대비 크게 개선(실험에서 92~96% 수준)되었고, 더 긴 조합 과제에서도 InSight는 80%에 도달한 반면 테스트-시 조합만 하던 baseline은 급락했다. 또한 기존 pick-and-place 기반 능력은 새 primitive 추가 후에도 유지(100% 성공)되며, scooping 시연만으로 sweeping(비집기(non-prehensile) 접촉 동작) primitive까지 자율 획득하는 등 continual skill acquisition의 실용성을 입증했다.



### Vision-Language Model Reasoning for Contextual Semantic Mapping in Intralogistics (https://arxiv.org/abs/2606.24814)
Comments:
          Accepted for publication at IEEE ETFA 2026

- **Prior Approaches**: 기존에는 SLAM 기반 기하 지도에 YOLO 같은 폐쇄형 인식 결과를 투영해 의미 라벨을 붙이거나, 미리 정해진 객체 범주에 의존하는 semantic mapping이 주를 이뤘습니다. open-vocabulary를 시도한 연구는 CLIP/VLM으로 객체 식별의 유연성을 얻었지만, 객체 ‘움직일 수 있는지’ 같은 맥락 속성에 대한 추론은 상대적으로 부족했습니다. 또한 SAM 계열 세그멘테이션을 써도 대개 단일 시점 중심이라 다중 시점 관측을 모아 추론하는 단계가 약했습니다.

- **Core Contribution**: 이 논문은 SLAM-기반 기하 지도 위에 SAM 기반 인스턴스 분할, 기하 공간에서의 instance clustering(클러스터링), 그리고 VLM 다중 시점 reasoning을 결합해 ‘contextual semantic map’을 만드는 파이프라인을 제안합니다. 그 결과 지도는 기하 구조뿐 아니라 객체 클래스와 movability(immovable/movable/mobile)를 함께 인코딩해, 작업에 유용한 맥락 정보를 제공합니다. 특히 task-specific fine-tuning이나 predefined 카테고리 없이도 VLM을 zero-shot, open-vocabulary로 질의해 속성을 추론합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 멀티모달 센서에서 시점 간 동일 객체를 일관되게 연결하고, (2) VLM이 단일 이미지의 모양만 보고 오판하지 않도록 다중 시점 정보를 구조화해 추론시키는 것입니다. 논문은 SAM 마스크를 기하 지도 좌표로 투영하고, IoU 기준으로 같은 객체 관측을 clustering해 VLM에 묶어서 전달합니다. 또한 full-scene 문맥과 close-up 디테일을 함께 담은 composite 이미지를 만들고, movability ontology와 JSON 출력 포맷을 포함한 structured prompting(직접 JSON vs CoT-augmented JSON)을 비교해 견고한 추론 구성을 찾았습니다.

- **Empirical Impact**: 실험은 통제된 intralogistics 환경에서 18개 객체 인스턴스/13개 클래스에 대해 평가했으며, semantic classification은 98.93% mIoU, movability 추정은 89.17% mAcc를 달성했습니다. VLM 비교와 prompting 전략 분석에서 Gemini 3.1 Flash Lite(직접 JSON)가 semantic 성능에 가장 유리했고, CoT는 모델별로 이득/손실이 달랐습니다. 컴포넌트 분석 결과 movability의 주요 병목은 VLM reasoning이었고, panoptic 성능의 주요 한계는 instance association(융합/연결)에서 나타났으며, 전체 파이프라인은 다중 시점 추론의 필요성을 보여주었습니다.



### World Value Models for Robotic Manipulation (https://arxiv.org/abs/2606.24742)
Comments:
          preprint

- **Prior Approaches**: 기존 일반화 가치(value) 모델은 VLM 백본에 기반한 경우가 많지만, VLM은 정적이거나 시간적으로 드문 관측에 대해 사전학습된 경우가 많아 가치 추정에 필요한 깊은 temporal modeling이 약합니다. 그 결과 현재의 믿음(상태)을 과거 맥락으로 정교하게 고정하고, 미래 결과를 기반으로 계획하는 능력이 제한되어 데이터 품질 평가나 학습 가이드가 흔들릴 수 있습니다.

- **Core Contribution**: 이 논문은 world model의 시간 이해·미래 계획 강점을 가치 추정에 결합해 World Value Model(WVM)을 제안합니다. WVM은 현재 상태를 과거 관측으로 정합하고 미래 전개를 고려해 데이터의 task progression을 정확히 평가하는 일반ist robotic value model로 설계됩니다.

- **Technical Challenges**: 핵심 과제는 world model을 가치 추정용으로 정렬해, 시간적 문맥 기반의 정확한 value estimation을 안정적으로 수행하는 것입니다. 이를 위해 WVM은 world model 계열의 temporal grounding과 future outcome 기반 계획 구조를 가치 평가 목표에 맞춰 결합하고, task progression을 신뢰도 있게 산출하도록 학습·평가를 구성합니다.

- **Empirical Impact**: 표준 벤치마크에서 WVM은 Value-Order Correlation(VOC)에서 SOTA를 달성했습니다. 또한 expert 데이터만 포함하던 평가를 넘어 Suboptimal-Value-Bench(800개의 suboptimal trajectory, 인간 라벨 프레임 주석)를 제안·검증해도 WVM의 SOTA 성능과 강인함이 유지됨을 보였고, policy learning에 활용 시 시뮬레이션과 실제 로보틱스 전개 모두에서 다양한 policy extraction 접근의 조작 성능을 향상시켰습니다.



### TACTFUL: Tactile-Driven Exploration For Object Localization and Identification in Confined Environments (https://arxiv.org/abs/2606.24712)
Comments:
          IROS 2026

- **Prior Approaches**: 기존 연구는 시각이 불완전할 때 촉각으로 localization을 돕거나(예: tactile SLAM) 접촉 신호를 기반으로 탐색을 수행했지만, 대부분 저차원 접촉 정보에 머물러 정밀한 3D 재구성까지 일관되게 이어지기 어려웠습니다. 또 촉각으로 형상을 복원하는 방법은 많았지만, 대개 ‘물체 위치를 이미 안다’는 가정 하에서 표면 탐색에 집중해 비전-무응답 상태의 자율 탐색과 결합이 약했습니다. 일부 촉각 전용 탐색은 시뮬레이션 또는 휴리스틱 기반으로 진행돼, 실제 고해상도 촉각 데이터에서의 성능 증거가 제한적이었습니다.

- **Core Contribution**: TACTFUL은 비전 없이(vision-free) 다지(多指) 로봇이 한 에피소드 내에서 (1) 한정된 작업공간을 자율 탐색해 물체를 찾고, (2) 접촉으로부터 촉각 포인트를 모아 3D 형상을 재구성하며, (3) 재구성된 형상으로 물체를 식별하는 일련의 프레임워크를 제안합니다. 핵심은 단일 RL 정책이 글로벌 탐색과 로컬 표면 정제를 동시에 수행하도록, ‘dynamic reward schedule’로 행동을 단계적으로 전환시키는 점입니다. 또한 시뮬레이션 없이 실제 하드웨어 데이터로만 학습한 정책을 사용해 실세계 구동성을 강조합니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 비전-무응답 환경에서 무작위 탐색이 비효율적이고 안전 문제를 유발할 수 있는데, 동시에 장기 탐색과 형상 재구성을 동시에 최적화해야 한다는 점입니다. 저비용 실전 학습을 위해 먼저 Behavior Cloning(BC)으로 텔레오퍼레이션 시연 5311건에서 안전한 접촉/탐색 행동의 초기화(사전 지식)를 만들고, 이후 PPO로 ‘접촉 보상, 탐색 보상, 재구성 보상’을 동적으로 가중해 하나의 정책을 학습합니다. 재구성은 희소 촉각 포인트를 shape completion(포인트 확산 기반 모델)으로 조밀하게 복원해, Chamfer distance 기반으로 식별 정확도를 높이는 방식으로 해결합니다.

- **Empirical Impact**: 실험에서는 실제 3개 물체(큐브, 변형 실린더, 변형 컵)를 대상으로 물체 라이브러리에서 타깃을 지정하고 식별 성공률을 평가했으며, TACTFUL은 평균 Chamfer-L2L2 0.015m에서 성공률 77%를 달성했습니다. 휴리스틱 탐색 및 BC 전용, RL만 학습한 대안보다 재구성 정확도와 식별 일관성이 모두 높았고, 특히 세 가지 보상 항을 모두 넣을 때만 성능이 안정적으로 유지됐습니다. 또한 shape completion 모델을 제거하면 Chamfer가 낮아도 식별이 흔들릴 수 있음을 보여, 촉각 기반 ‘명확한 물체별 형상’ prior가 실세계에서 중요하다는 함의를 제공합니다.



### Beyond Monotonic Progress: Retry-Supervised Value Learning for Robot Imitation (https://arxiv.org/abs/2606.24633)
- **Prior Approaches**: 로봇 모방학습에서 보상·가치 모델은 주로 task progress를 시간에 따라 단조 증가하는 스칼라로 가정해 학습한다. 하지만 인간 시연에는 그립 부정확, 오브젝트 정렬 불일치, 불안정 접촉, 반복 시도처럼 국소적인 실행 오류와 재시도가 섞이며, 단조 진행 가정은 이런 ‘미세한 실패-회복 패턴’을 놓치기 쉽다.

- **Core Contribution**: ReTVL( ReTry-Supervised Value Learning )은 시연 중 반복 시도(retry) 시작 시점을 스파스(sparse)한 감독 신호로 활용해 실수에 민감한 value function을 학습한다. 전역의 coarse progress 기준선은 유지하되, retry 주변에서 값이 하락했다가(오류) 회복되는(교정) 국소 구조를 pairwise preference loss로 반영한다.

- **Technical Challenges**: 핵심 기술적 난제는 retry 주변의 비교쌍이 애매한 윈도 경계나 먼 프레임에 의존해 노이즈가 커질 수 있다는 점이다. 논문은 retry keypoint를 기준으로 pre/near/post 구간을 나눠 국소 최소값 가정을 만들고, soft-window weighting으로 경계 민감도를 완화하며, preference 기반 로스를 통해 지역적인 값의 순서(하락/회복)를 직접 학습한다.

- **Empirical Impact**: 4개 실제 로봇 조작 태스크에서 ReTVL은 progress 기반 가치 모델보다 지역 오류 민감도(local mistake sensitivity)가 크게 개선되면서도 전역 상관(VOC)과 성공/실패 구분은 유지했다. 또한 learned value로 시연 chunk의 가중치를 재조정한 ReTVL-BC는 Standard BC와 RECAP-BC 대비 높은 성공률(평균 80.00%)을 보였고, 수동 라벨된 ‘나쁜 행동’에 대한 가중치도 더 강하게 낮추는 것으로 분석됐다. 이는 corrective behaviors를 단순 잡음이 아니라 감독 신호로 활용할 수 있음을 실증적으로 보여주는 결과다.



### Optimization-based Safe Trajectory Planning for Autonomous Ground Vehicle in Multi-Floor Scenarios (https://arxiv.org/abs/2606.24631)
- **Prior Approaches**: 기존 AGV 궤적 계획은 검색 기반(A* 계열), 샘플링 기반(RRT/PRM), 최적화 기반(OCP)으로 나뉘며, 특히 최적화 기반은 직접법이 복잡한 제약에서 유리하다고 알려져 있습니다. 다만 많은 접근이 단일 평면/단층 위주로 설계돼 다층 환경에서의 층 간 연결(출구 선택)과, 그 선택이 좌우하는 초기조건 품질을 충분히 다루지 못합니다. 또한 다층에서 출구 후보가 여러 개일 때 kinematic 제약까지 고려한 “빠른 최적 출구 의사결정”이 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 다층(multi-floor) 시나리오에 맞춘 궤적 계획 프레임워크를 제안합니다. 1단계에서는 generalized voronoi diagram(GVD) 기반 거리/연결성 정보와 다목적(Pareto) 최적화를 결합해 각 층의 floor exit를 선택합니다. 2단계에서는 최적화 기반 궤적 생성에 계층적(hierarchical) warm-start/hot-start 전략을 적용해 빠른 수렴과 고품질 궤적을 동시에 노립니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 다층에서의 전역 작업 계획과 (2) 직접법 OCP 최적화의 초기 추정 품질/수렴 속도, (3) 복잡한 장애물 제약의 비효율을 동시에 줄이는 것입니다. 이를 위해 1층(low-precision)에서는 단순화된 kinematic 모델과 “warm-start factor”를 사용해 초기 해(시간/상태/운동정보)를 빠르게 만들고, 그 결과를 2층(high-precision) OCP의 warm 시작으로 넘겨 SQP 기반 수렴을 가속합니다. 또 obstacle inflation과 더불어 필요 없는 장애물 제약을 제거하는 obstacle deletion(상호관계 판단)으로 제약 수를 줄여 수렴성을 향상시켰습니다.

- **Empirical Impact**: MATLAB 시뮬레이션을 통해 다층 실내 시나리오 2종에서 프레임워크의 feasibility와 성능을 확인했습니다. 비교 실험에서 제안 방법의 궤적 성능 지표가 다른 최적화 기반 대안보다 우수했고, 몬테카를로 안정성 테스트에서도 성공률이 크게 높게(예: 89.6%) 나타나 외란에 대한 robustness도 입증했습니다. 특히 출구 선택부터 층별 궤적 최적화까지 end-to-end에 가까운 전역 계획 흐름을 구성해, 다층 AGV 연구에서 실용성을 높인다는 의미가 있습니다.



### ArtiTwinSplat: Interactable Digital Twin Reconstruction via Gaussian Splatting from RGB-D videos (https://arxiv.org/abs/2606.24628)
Comments:
          Presented at the ICRA 2026 Workshop on Advances and Challenges in AI-Driven Automation and Robotic System Integration with Digital Twins, Vienna, June 2026

- **Prior Approaches**: 관절이 있는 물체(문, 서랍 등)를 재구성하려는 기존 연구는 3D 형상과 모션을 함께 추정하되, 정적 장면 가정(NeRF, 3DGS 계열)이나 파트 라벨/사전 분할, 다중 뷰·다중 상태 캡처 같은 전제가 많았다. Real2Code, ArticulateAnything, PARIS 등은 큰 주석 데이터나 분할 입력, 통제된 캡처 조건에 의존해 실제 로봇 배치 환경에서 확장성이 떨어진다. 3DGS 기반의 ArtGS, SplArt 역시 부분 감독 또는 여러 관측 조건이 필요해 핸드헬드 RGB-D 단일 시퀀스에는 제약이 컸다.

- **Core Contribution**: ArtiTwinSplat은 CAD나 시뮬레이션 자산, 수동 주석 없이 핸드헬드 RGB-D 영상만으로 관절(회전/직선) 디지털 트윈을 자동 생성하는 프레임워크다. 3D Gaussian Splatting으로 외관과 기하의 photorealistic 재현성을 유지하면서, 관절 구조와 kinematics를 “관측된 움직임만”으로 비지도 방식으로 찾아낸다. 결과물은 실제 로봇 시뮬레이션에 바로 쓸 수 있게 URDF로 내보낼 수 있어, 디지털 트윈 제작의 통합 병목을 줄이는 데 초점을 둔다.

- **Technical Challenges**: 가장 큰 난제는 관절이 있는 객체에서 구조(파트)와 모션(관절 파라미터)이 강하게 결합되어 있어 단일 관측만으로는 문제 자체가 ill-posed라는 점이다. 이를 위해 ArtiTwinSplat은 (1) 사전 변화(pre-change) 기준 3DGS를 만든 뒤, (2) 변화 마스크를 외관·기하 차이로 검출하고 SAM2로 시공간 일관된 파트 분할을 역방향 전파로 안정화한다. 이후 TAPIP3D로 마스크 영역 점 궤적을 추적하고 4D RANSAC으로 revolute/prismatic joint를 피팅해 관절 타입·축·피벗·가동 범위를 추정한 뒤, 관절 파라미터에 조건을 건 2단계 Gaussian 최적화로 장면 외관은 유지하면서 시간적으로 일관된 관절 변형을 학습한다.

- **Empirical Impact**: 아이폰 Pro로 촬영한 3개 가정용 장면(각 장면은 정적 pre-change와 동적 post-change 2개 시퀀스)에서, 움직이는 컴포넌트를 안정적으로 분리하고 미관 생성 및 새로운 관절 상태에서의 합성 렌더링을 일관되게 보여준다. prismatic은 운동 중 내부가 비교적 지속적으로 보이기 때문에 더 정확하게 복원되며, revolute는 회전 중 가림으로 내부 표면 일부 아티팩트가 생길 수 있다고 보고한다. 무엇보다 한 번의 캐주얼 RGB-D 캡처로 Isaac Sim 호환 URDF까지 생성돼, embodied AI와 휴먼-로봇 협업에서 “바로 쓰는” articulated digital twin 제작 접근을 실사용 관점에서 한 단계 낮췄다는 점이 의미 있다.



### Enabling Robust Cloth Manipulation via Inference-Time Simulator-in-the-Loop Refinemen (https://arxiv.org/abs/2606.24552)
- **Prior Approaches**: 변형(클로스/패브릭) 조작은 무한차원 구성공간, 국소 형상 변화, 강한 자기 가림, 마찰 접촉 때문에 상태추정과 행동추론이 매우 어렵다. 기존에는 모방학습으로 명시적 동역학 모델링을 회피하거나, RGBD/포인트클라우드 기반 3D 복원·예측을 통해 상태를 추정한 뒤 행동을 생성했다. 또 simulator-in-the-loop 시도도 있었지만, 변형물에서는 상태·접촉 표현의 비용과 sim-to-real 불일치가 커서 안정적 온라인 정제가 제한적이었다.

- **Core Contribution**: 이 논문은 단일 RGB 입력으로부터 실세계 옷 조작을 수행하면서, inference-time에 물리 시뮬레이터를 직접 forward model로 써서 실행 오차를 보정하는 unified simulator-in-the-loop 프레임워크를 제안한다. 핵심은 (1) FLASH 기반 변형 시뮬레이터로 확장 가능한 synthetic-data-롤아웃 파이프라인을 구축하고, (2) real-to-sim 모듈을 오직 synthetic 데이터로 학습해 시뮬레이션 호환 cloth state를 복원하며, (3) prior-guided MPPI로 nominal action trajectory를 병렬 물리 롤아웃으로 온라인 정제하는 것이다. 이를 통해 prior 정책을 closed-loop로 끌어올려 접촉이 많은 변형 조작에서 누적 오차를 복구한다.

- **Technical Challenges**: 가장 큰 기술적 도전은 (a) 시뮬레이터가 온라인 롤아웃에 충분히 빠르고 수치적으로 안정적이어야 하고, (b) 단일 RGB에서 시뮬레이터와 동기화 가능한 cloth 상태를 정확히 복원해야 하며, (c) 고차원 변형 표현을 유지하면서도 제한된 제어/배치 예산 하에 MPPI 후보를 충분히 병렬 평가해야 한다는 점이다. 저자들은 FLASH의 삼각 메쉬 기반 변형·비매끄러운 접촉(마찰 포함) 처리를 롤아웃 엔진으로 선택하고, real-to-sim에서는 frozen DINOv2 시각 특징과 learnable canonical tokens을 결합해 고정 토폴로지 머티리얼 버텍스의 3D 변형을 예측하도록 설계했다. 또한 input/latent 수준의 섬세한 perturbation(시각 랜덤화 및 토큰 마스킹, grasping-point 주변 가림)을 넣어 sim-to-real 시프트에 강인하게 학습한 뒤, prior-guided MPPI가 sparse-mesh 롤아웃과 비용함수를 통해 로컬 탐색을 효율적으로 수행하도록 구성했다.

- **Empirical Impact**: 실로봇 실험에서 제안한 전체 파이프라인은 baseline 대비 더 높은 success rate와 더 강한 robustnness를 보였다. 구성요소를 하나라도 대체·약화하면 성능이 떨어지는 것으로 나타나, 상태동기화(2), 시뮬레이터 충실도(1), 온라인 정제(3)가 tightly coupled임을 실증한다. 또한 FLASH를 다른 변형 시뮬레이터(Isaac Sim, Newton, Genesis)와 동일 MPPI 설정에서 비교한 결과, 제어 정확도·계산시간·최종 상태 충실도 간 최적의 균형을 보이며 다수 환경에서도 높은 성공을 달성했다.



### Explaining Failures of Cyber-Physical Systems with Actual Causality (https://arxiv.org/abs/2606.24546)
Comments:
          Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)

- **Prior Approaches**: 기존 인과 분석 기반 설명은 주로 image classifier 같은 단순 모델(깊이 고정의 인과 모델 가정)에서 주로 수행됐다. CPS에서는 시간적으로 전개되는 제어·관측 루프, 환경 변수의 전역/국소 영향, 그리고 실패가 특정 시점의 궤적에서 발생한다는 점 때문에 동일한 방식의 적용이 곧바로 맞지 않을 수 있다. 그 결과, 무작정 적용하면 실제로 관측되지 않은 환경 요소를 원인으로 포함하거나, ‘무엇이 실패 타입을 구분했는지’가 사라진 채로 잘못된 설명이 도출될 위험이 있다.

- **Core Contribution**: 이 논문은 actual causality(실제 인과) 프레임워크를 CPS 실패 설명에 적용하는 새로운 개념을 제시하고, CPS 도메인에서 올바른 설명 도출을 위한 이론적 보정 지침을 마련한다. 또한 failure event(실패 사건)를 “어떤 환경 요소들이 그 사건의 결과를 재현했는가”로 설명하도록, CPS 테스트 모델을 인과 모델과 연결한다. 추가로, 설명의 품질(최적성)과 계산 효율 중 한쪽을 우선하는 두 가지 시스템-비의존적(practical, system-agnostic) 알고리즘을 제안한다.

- **Technical Challenges**: 핵심 기술 난점은 CPS 인과 모델을 구성할 때의 차이에서 비롯된다: 환경 parameter는 중시되어야 하지만 단순 중화(neutralization)가 어렵고, time-extended 구조 때문에 실패 이전에 관측되지 않은 element가 설명에 섞일 수 있다. 또 CPS의 이진 success/failure 지표만으로는 실패가 어떤 궤적 구간에서 발생했는지(실패 타입) 구분이 되지 않아, 사건과 논리적으로 맞지 않는 설명이 나올 수 있다. 논문은 관측된 환경 요소에 설명을 제한하고, task-success를 넘어 연속 궤적의 의미를 적절히 추상화해 출력(이벤트)을 정의하는 방식으로 이를 해결한다.

- **Empirical Impact**: 제안한 접근은 neural-network-controlled autonomous car가 장애물이 있는 트랙에서 충돌을 회피하지 못하는 상황을 대상으로 실험·평가됐다. 두 알고리즘은 설명 최적성 또는 도출 효율을 선택적으로 우선하면서도, 해당 실패 사건에 맞는 원인 요소를 뽑아내는 데 유의미한 성과를 보인다. 이는 CPS에서 신뢰를 높이고(failure explanation을 통한 이해), 향후 유사 실패를 더 잘 예방·완화하는 실용적 경로를 제공한다는 점에서 의미가 있다.



### Decentralized Pose Graph Riemannian Optimization for Object-based Multi-Robot SLAM (https://arxiv.org/abs/2606.24489)
- **Prior Approaches**: 기존 PGO는 중앙집중형 g2o/GTSAM 같은 비선형 최소제곱이나, 이론적 보장을 강조한 분산 최적화가 주로 다뤄졌다. 그러나 분산 PGO는 통신 그래프가 물리적 측정 토폴로지와 잘 맞아야 하거나, 일부 로봇이 대기하는 등 운용 제약이 생기기 쉽다. object-based multi-robot SLAM에서는 로봇 궤적(사적 변수)과 지속 물체 포즈(공적 변수)가 강하게 얽히는데, 기존 분할/합의 방식이 이 구조를 충분히 활용하지 못해 통신이 불필요하게 커지고 수렴이 느려질 수 있다.

- **Core Contribution**: 이 논문은 DRAN으로, object-based multi-robot PGO의 결합 추정을 fully decentralized 방식의 Riemannian 최적화로 풀되, 공적 물체 변수에는 consensus를 적용하고 사적 궤적 변수는 separator 교환으로만 연결해 결합을 구조적으로 분리한다. 그 결과 통신 토폴로지가 물리 상호작용 그래프와 불일치하거나 희소/간헐/시간변동이어도 더 유연하게 동작하도록 설계했다. 또한 SE(d) 매니폴드 위에서 동작하는 distributed approximate-Newton을 도입해, 2차 정보는 네트워크로 Hessian/그래프 전체를 교환하지 않고도 수렴을 가속하는 데 활용한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 공적 변수 합의와 사적 변수 로컬 업데이트를 통신 제약 하에서 동시에 만족시키는 최적화 설계, (2) SE(d) 기하를 보존하는 2차 근사 업데이트, (3) 제한된 통신 예산에서 Newton급 방향을 얻는 것이다. 논문은 LM/Gauss–Newton 기반의 로컬 2차 곡률 모델을 만들고, Schur complement로 공적/사적 변수를 분리해 큰 Hessian을 직접 교환·역전파하지 않도록 했다. 더불어 Riemannian 합의 과정에서 1차 정상점으로의 수렴을 보이고, approximate second-order 정보가 first-order Riemannian descent 대비 유리한 이유를 로컬 condition-number 관점에서 분석한다.

- **Empirical Impact**: 공개 PGO 벤치마크, 대규모 시뮬레이션, 실제 멀티로봇 실험에서 DRAN은 정확도는 유지하면서 반복 횟수와 통신량을 줄여 런타임 효율을 개선했다. 특히 통신 밀도 변화와 확률적 네트워크 중단 상황에서도 near-centralized 수준의 추정 성능을 보이며 견고성을 확인했다. 요약하면, 통신 실패/희소 네트워크에서도 스케일러블한 object-based multi-robot PGO 백엔드로서 실용성이 강화됐다는 점이 의미 있다.



### G$^3$VLA: Geometric inductive bias for Vision-Language-Action Models (https://arxiv.org/abs/2606.24472)
Comments:
          Submitted to CoRL 2026

- **Prior Approaches**: 기존 VLA(vision-language-action) 모델은 2D 이미지 토큰을 주로 다루며, 카메라 intrinsics/extrinsics로 결합된 다중 뷰의 기하 구조를 행동 감독을 통해 간접적으로 학습해 왔습니다. PerAct, RVT, Act3D 등은 3D 표현(복셀/렌더/3D feature field)을 쓰며 공간 정밀도를 높이지만, 사전학습된 VLM의 의미 토큰을 그대로 활용하기 어렵거나 액션 표현을 바꾸는 경우가 많았습니다. SpatialVLA와 3D-VLA 같은 연결 시도도 대개 3D 센서 입력, 대규모 공간 사전학습, 혹은 액션 표현 수정이 필요하다는 한계가 남아 있습니다.

- **Core Contribution**: 이 논문은 카메라 보정(calibrated) 기하를 VLA의 시각 토큰 스트림에 “가벼운 인터페이스”로 주입하는 G^3VLA를 제안합니다. 백본/액션 공간/모방(imitation) 목적은 그대로 둔 채, intrinsic-conditioned ray embeddings, PRoPE(projective positional encoding), 양방향 cross-view fusion으로 로봇 카메라 모델 기반 구조를 토큰에 주입합니다. 또한 depth 센서 없이도 ground-truth point map이 있거나, confidence-gated π^3X(π^3π^{3}X) teacher 예측을 사용해 기하를 감독합니다.

- **Technical Challenges**: 핵심 과제는 “2D 토큰 기반 인터페이스”에 캘리브레이션된 다중 뷰 기하를 넣되, 사전학습된 VLA의 행동 경로를 깨지 않는 것입니다. 저자들은 patch 단위 토큰이 각 카메라의 viewing direction을 알도록 ray embeddings로 intrinsics를 주입하고, PRoPE로 뷰 간 투영 관계를 attention bias로 제공한 뒤, cross-view fusion에서 기하 컨텍스트를 교환하게 했습니다. 또 기하 모듈을 즉시 행동 손실만으로 학습하면 불안정할 수 있어, 보조 점 헤드 기반의 dense geometry distillation과 two-stage curriculum(기하 선학습→전체 파인튜닝)으로 안정적으로 부트스트랩합니다.

- **Empirical Impact**: 실험에서 G^3VLA는 LIBERO, RoboCasa24, RoboTwin2.0 및 실제 로봇 환경 전반에서 일관된 성능 향상을 보이며, 특히 공간/객체 민감 태스크에서 개선폭이 큽니다. π0 설정에서는 ground-truth 기하 감독이 가장 강하지만, depth가 없을 때도 π^3X 증류로 기준 대비 큰 이득을 회복해 실용성을 확인했습니다. 추가로 π0.5에서는 소폭 추가 이득이 유지되고, GR00T 1.5에서는 geometry-aware 토큰이 액션 생성 경로에 직접 도달할수록 효과가 커진다는 해석이 제시되어, “기하 전이”의 설계 조건도 함께 시사합니다.



### FT-WBC: Learning Fault-Tolerant Whole-Body Control for Legged Loco-Manipulation (https://arxiv.org/abs/2606.24466)
- **Prior Approaches**: 다리 달린 로봇에 팔을 결합한 loco-manipulation은 작업을 더 유연하게 만들지만, 팔이 만드는 CoM 이동과 동적 교란 때문에 하체 actuator 고장 시 불안정 위험이 커진다. 기존 fault-tolerant 제어는 주로 locomotion(넘어짐 방지, 속도 추종)에 초점이 맞춰져 있어, 팔 reachability를 위해 필요한 기울어진 자세가 고장 상황에서 어떻게 위험해지는지를 함께 다루지 못했다.

- **Core Contribution**: 본 논문은 actuator failure 하에서도 whole-body stability를 유지하면서 팔의 작업공간을 최대한 보존하는 fault-tolerant loco-manipulation 프레임워크 FT-WBC를 제안한다. 핵심은 decoupled upper/lower-body policy 구조와, 하체의 Fault Estimator(FE) 및 이를 이용해 base posture plan을 안전한 명령으로 바꾸는 Posture Adaptation Module(PAM)이다.

- **Technical Challenges**: 주요 기술 과제는 actuator fault가 물리적으로는 숨은 상태(hidden state)여서 단순한 학습 특징만으로는 fault-aware 대응을 만들기 어렵다는 점이다. FE는 하체 proprioceptive history를 시간적으로 집계해 faulty joint를 예측하고, PAM은 그 fault vector를 바탕으로 arm policy가 제안한 posture 요청이 CoM을 열화된 지지면 쪽으로 몰아갈 수 있는 경우를 재매핑·클리핑해 실행 가능하고 안전한 base 자세만 남기도록 설계했다.

- **Empirical Impact**: 시뮬레이션과 실로봇 실험에서 FT-WBC는 weakening과 locked 고장 조건 전반에서 생존율과 workspace를 기존 Robo-Duet 및 무모듈 변형(w/o FE, w/o PAM)보다 일관되게 개선했다. 또한 학습 중 보지 못한 locked fault에 대해서도 ground-to-table pick-and-place에서 zero-shot sim-to-real로 적응성을 보이며, 90cm 작업처럼 위험한 자세 요구에서도 “survival-first” 전략으로 급격한 실패를 억제하는 성과를 확인했다.



### Varying Bundle Size Reactive Multi-Task Assignment using Selective Cost Estimation for Multi-Agent Systems (https://arxiv.org/abs/2606.24462)
Comments:
          Accepted for publication at the 23rd World Congress of the International Federation of Automatic Control (IFAC 2026)

- **Prior Approaches**: 기존 조합 경매 기반 multi-robot task allocation은 bundle(작업 순서 묶음)별 정확한 보상/비용을 알기 어렵거나, 비용 검증을 위한 경로계획이 비싸면 bundle 생성이 지수·팩토리얼로 폭증해 실시간 반응이 힘들었다. 또한 분산 합의 기반(consensus) 방식은 확장성은 좋지만 전역 제약을 충분히 다루지 못하거나 비최적 해가 나올 수 있고, 비용 추정이 반복되면 재계산 부담이 커졌다. 중앙집중 최적화는 성능·제약 처리는 강하지만, bundle마다 무거운 planning을 수행해야 할 때 확장성이 떨어진다.

- **Core Contribution**: 이 논문은 비용이 비싼 환경에서 실시간 반응성을 유지하기 위해, 에이전트가 만드는 bundle을 ‘두 단계, multi-fidelity’로 생성하는 분산 프레임워크를 제안한다. 1단계에서는 유클리드 근접 같은 저정밀 휴리스틱으로 depth-limited variable-width beam search를 통해 후보 bundle을 대량 생성하고, 2단계에서는 best-first 방식으로 유망한 소수 후보에만 고정밀 경로계획 기반 비용 검증을 적용한다. 그 뒤 중앙 코디네이터는 refined bid를 set packing 문제로 풀어 전역 충돌 없이 전체 utility를 최대화하며, 에이전트의 상태와 내부 비용 모델은 공유하지 않도록 설계됐다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 팩토리얼로 커지는 bundle 공간을 탐색하면서도 (2) 각 bundle의 정확한 실행비용은 motion planning이 필요해 비싸다는 점이며, (3) 이 과정에서 익명성·프라이버시를 유지해야 한다. 논문은 트리 생성 시에는 빠른 거리 기반 추정치로 우선순위를 만들고, best-first 정제 단계에서 우선순위 큐로 상위 후보만 고정밀 비용을 계산하도록 하여 계산 예산을 절감한다. 또한 경로 구간 비용을 로컬 캐싱해 동일 전이(작업 i→j) 재계산을 줄이고, 중앙에는 bundle 총비용만 bid로 제출해 내부 모델/상태 노출을 최소화했다.

- **Empirical Impact**: 시뮬레이션 3개 환경(개활지/동굴형 장애물/터널 레이아웃)에서 제안 프레임워크는 fully reactive 경매 기반 기준 대비 평균적으로 task completion 성능을 약 14~18% 향상(정적 시나리오)시키며, 계산 시간은 기준과 비슷한 수준을 유지했다. 동적 과제 도착과 임의 중단이 있는 조건에서도 평균 10~17% 우위를 보여, 제한 시간 안에 ‘hot spot’ 중심으로 더 많은 작업을 수행하는 경향을 확인했다. 특히 터널 검사 시나리오에서는 유클리드 휴리스틱이 부정확한 비볼록 지형에서도 후보 트리·정제 과정이 의미 있는 bundle을 구성함을 정성적으로 보여, 실용 가능성을 뒷받침한다.



### NoContactNoWorries: Estimating Contact through Vision and Proprioception for In-Hand Dexterous Manipulation (https://arxiv.org/abs/2606.24450)
Comments:
          Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems(IROS) 2026

- **Prior Approaches**: 기존 비전 기반 덱스터러스 조작은 영상에서 정책을 학습하거나 자세/물체 재배치를 수행하지만, slip·힘 분포·국소 안정성처럼 접촉 의존 현상은 자기 가림과 감독 신호의 약함 때문에 일관된 일반화가 어렵습니다. 반면 촉각 기반 접근은 접촉 위치의 직접 증거를 제공하지만, 센서의 비용·취약성·커버리지 한계·배선/캘리브레이션 부담 때문에 확장성이 떨어집니다. 또한 비전-터치 융합이나 self-supervised 선학습은 효과적일 수 있으나, 결국 배치 시 촉각 입력이 필요하거나 다운스트림 정책을 재학습하는 경우가 많았습니다.

- **Core Contribution**: 이 논문은 촉각 하드웨어 없이도 ego-centric RGB-D와 proprioception만으로 손가락 접촉(contact)을 이진 신호로 추정하는 pseudo-tactile sensing을 제안합니다. 제안 모델 NoContactNoWorries는 RGB-D 토큰과 현재/명령된 관절 상태를 transformer의 cross-attention으로 융합하고, causal attention으로 짧은 시간 정보를 누적해 각 손가락 위치의 이진 접촉을 실시간 관측처럼 제공합니다. 또한 단일 접촉 예측 모델을 여러 물체에 대해 학습한 뒤, 예측된 접촉 신호를 시뮬레이션에서 강화학습 에이전트의 다운스트림 입력으로 사용해 새 물체로도 일반화되는 것을 보입니다.

- **Technical Challenges**: 가장 큰 난제는 단일 프레임에서 접촉이 단순 근접과 쉽게 구분되지 않으며(모호성), 비전은 자기 가림/시점 의존성이 크고, proprioception은 외부 물체 기하를 직접 담지 못한다는 점입니다. 이를 위해 frozen RGB-D 백본에서 시공간 특징을 뽑고, 현재 관절과 명령 관절을 별도 쿼리로 사용해 ‘반응적/예측적’ 정보가 다른 방식으로 장면 토큰을 선택하도록 설계했으며, causal transformer로 시간적 단서를 누적해 접촉을 추론합니다. 학습은 PhysX/실물 저프로파일 FSR로 이진 접촉 레이블을 만들고, 모델은 이 추정치를 정책에 직접 넣는 real-time observation 형태로 사용하도록 구성했습니다.

- **Empirical Impact**: 시뮬레이션과 실제 로봇(LEAP hand) 모두에서 이진 접촉 예측의 F1이 상위 수준으로 나타났고, 시뮬-to-실물 전이에서도 견고하게 유지되었으며 모호한 비전 단서만으로는 성능이 크게 떨어졌습니다. 특히 카메라 가시성 기준으로 나눴을 때 접촉 사이트가 가려진 경우 비전-only 성능이 급감해, 제안된 융합(시각+자세+명령+시간)이 자기 가림 상황에서 실제로 기여함을 뒷받침합니다. 더 나아가 예측된 접촉 신호를 oracle/tactile을 대체해 시뮬에서 학습한 in-hand reorientation 정책이 실제 하드웨어로 직접 이전(정책 재학습 없이) 가능한 점이 덱스터러스 조작에서 촉각의 ‘가상 센서화’ 가능성을 실증합니다.



### Supervise What Survives: Geometry-Guided VLA Adaptation from Synthetic Robot Videos (https://arxiv.org/abs/2606.24448)
Comments:
          14 pages, 5 figures

- **Prior Approaches**: 기존 VLA 연구는 생성된 로봇 영상을 “시연”으로 간주하고, inverse dynamics나 retargeting으로 pseudo-action을 복구한 뒤 action head를 학습하는 방식이 주류였습니다. 하지만 생성은 시각적 그럴듯함에 최적화되며 물리 제어 법칙을 동일하게 보존하지 못해, 픽셀 기반 신호를 정밀한 motor command에 대응시키는 과정에서 영상-행동 추상 수준이 어긋난다고 지적합니다.

- **Core Contribution**: 이 논문은 생성 영상이 주로 보존하는 것은 geometry(공간 구조, where)이고 제어 신호 control(정확한 실행 방식, how)는 소실된다는 Asymmetric Preservation Principle을 제안합니다. 이를 바탕으로 GRA(Geometry-guided Representation Alignment)는 사람 비디오에서 얻은 2D end-effector waypoint를 별도 보조 신호로 비전 백본에 주고, action head는 real demonstration으로만 학습하도록 감독 경로를 분리합니다. 또한 파인튜닝 중 waypoint loss를 유지해 공간 기반 표현이 action 학습에 의해 무너지는 drift를 막습니다.

- **Technical Challenges**: 핵심 난제는 “생성 영상에서 회수된 행동”이 아니라 “살아남는 기하”만 신뢰도 있게 감독으로 라우팅하는 설계를 만드는 것입니다. 이를 위해 pose estimation- retargeting- physics simulation- 카메라 캘리브레이션을 거쳐 미래 K-step 2D waypoint를 계산하고, 생성 프레임은 비전 경로에만 넣되 action 학습 데이터로는 쓰지 않습니다. Stage 1에서 waypoint로 비전 표현을 먼저 정렬하고, Stage 2에서 action fine-tuning과 동시에 공간 앵커 손실을 병행해 표현 붕괴를 완화합니다.

- **Empirical Impact**: 실험은 7-DoF Franka에서 pick-and-place 3개 과제로 진행됐으며, 같은 real 데이터 예산 조건에서 GRA가 pseudo-action 라벨링 기반 대비 성능을 크게 개선했습니다. 특히 DreamGen-style과 MimicDreamer-style pseudo-action 학습은 real-only보다 낮은 성공률을 보였고, 이는 closed-loop 실행에서 delta action의 작은 바이어스가 누적된다는 해석과 연결됩니다. 또한 GRA는 더 많은 real 데모로 학습한 기준 대비 격차를 절반 수준으로 줄이면서, 생성 비디오를 “통제”가 아닌 “공간 감독”으로 올바르게 연결하는 전략의 효과를 실증했습니다.



### Legible and Intuitive Multi-modal Robot State and Intent Communication Validated in Online and Real-world Studies (https://arxiv.org/abs/2606.24445)
Comments:
          Accepted for publication at the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026)

- **Prior Approaches**: 기존 연구는 LED, 디스플레이, 음성+제스처 등 다양한 eHMI/HRI 신호를 온라인·실험실 조건에서 검증해 왔지만, 산업 표준은 메시지를 무엇이자 얼마나 ‘해독 가능(legible)’하게 만드는지에 대한 구체 지침이 부족했다. 특히 표현 대역폭(Expressive Bandwidth) 수준을 체계적으로 비교한 연구가 적고, 동영상 기반 가상 평가 결과가 실제 로봇 상호작용에서 그대로 재현되는지(virtual-to-real gap)도 충분히 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 비인간형 mobile 로봇에 대해 표현 대역폭의 양 끝단을 대표하는 두 가지 커뮤니케이션 전략을 동일한 메시 세트로 구현해, online(동영상 설문)과 in-person(박물관 현장)에서 대규모로 비교 검증한다. 하나는 LED 중심의 unimodal 저표현 전략, 다른 하나는 gaze/제스처/음성 등으로 구성한 multimodal 고표현 전략이며, turning intention, attention request, error status, stuck/정상 동작 여부를 포함한 5종 메시를 분석한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 다양한 메시 유형에 대해 ‘의도 추론 가능성(legibility)’과 ‘이해 용이성(intuitiveness)’을 공정하게 측정하고, (2) 온라인에서 보던 신호 해독이 실제 소음·주의 분산 환경에서 얼마나 약화되는지 검증하며, (3) 동일 로봇 플랫폼에서 저/고 대역폭 전략을 메시별로 일관되게 구현하는 것이다. 저자들은 산업 표준에서 영감을 얻은 메시 포맷을 사용해 LED(저대역폭)와 gaze·gesture·speech(고대역폭)를 각각 동일 메시에 매핑하고, 온라인 설문( N=100, 실험 후 불합격 제외 N=97) 및 박물관 현장 검증( N=139)으로 두 단계 평가를 설계했다.

- **Empirical Impact**: 결과는 전반적으로 online이 real-world보다 legibility와 intuitiveness가 높고, 두 전략 모두 online에서 더 잘 이해된다는 점을 보여 virtual-to-real gap의 존재를 뒷받침한다. 동시에 고표현 multimodal 전략(𝒮high)이 저표현 LED 기반 전략(𝒮low)보다 전반적 해독성과 직관성이 더 높게 지각되며(legibility: 온라인 M=0.65 vs 현장 M=0.35, 전략별로도 𝒮high>𝒮low), 특히 LED로 의도를 알리는 항목에서 현장 감쇠가 두드러지고 해석에 대한 신뢰도도 감소했다. 저자들은 시스템 상태는 LED 같은 견고한 단일 신호가 효율적일 수 있지만, 복잡한 의도(공간적으로 앵커된 메시)는 multimodal 단서가 오해를 줄이는 데 더 유리하다는 ‘하이브리드 설계 패러다임’을 제안한다.



### RE4: Transformation-aware Imitation of Object Interactions Using Manipulation Modes (https://arxiv.org/abs/2606.24403)
Comments:
          8 pages, appendix

- **Prior Approaches**: 확산(diffusion)·flow-matching 같은 end-to-end 생성 정책은 접촉이 많은 조작에서 성능이 뛰어나지만, 각 스텝이 denoiser를 반복 호출해 계산 비용과 함께 해석 가능성이 떨어진다는 한계가 있습니다. 또한 diffusion이 데모를 생성하기보다 사실상 가장 가까운 것을 재생(retrieval)한다는 분석이 나오며, retrieval 기반 접근이 주목받았지만 단순 nearest-neighbour 재사용이나 에피소드 단위의 단일 pose transfer에 그쳐 manipulation mode의 불연속성(접촉 전환)을 충분히 다루지 못했습니다. Object-centric·pose-informed IL도 일부 요소는 갖추지만, 필요한 변환·재계획·모드 제약을 하나의 해석 가능한 파이프라인으로 엮는 방식은 상대적으로 미흡했습니다.

- **Core Contribution**: 이 논문은 object interaction imitation을 위해 RE4 프레임워크( Retrieve–Reframe–Replan–Replay )를 제안하며, manipulation theory의 핵심 아이디어를 빌려 성능과 해석 가능성을 함께 보존하려고 합니다. 4단계를 모두 manipulation mode( transit/transfer )에 조건화하고, 데모의 동작을 목표 물체 pose 기준으로 재구성(객체 상대 reframing)한 뒤 모드 제약을 유지한 채 연결·롤아웃하는 방식으로 동작의 불연속성을 정면으로 처리합니다. 이미지 관측에서는 general-purpose 6DoF pose estimator 없이, 데모 데이터에 대한 self-supervised로 lightweight한 타깃 물체 pose 추정을 학습해 이 조건화를 가능하게 합니다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 이미지에서 물체 pose를 ‘정확히’가 아니라 ‘일관되게’ 추정해야 한다는 점과 (2) 접촉 지점에서 mode가 바뀌며 단순 연속 보간만으로는 데모 연결이 불가능하다는 점입니다. 저자들은 pose 절대값 정확도에 의존하지 않도록, Retrieve와 Reframe을 모두 물체 pose의 상대 변환으로만 구성해 다운스트림이 pose의 상대성에 안전하도록 만들었습니다. 또한 Retrieve 단계에서 manipulation mode를 hard filter로 고정해 서로 다른 모드 간에는 단순 연결을 시도하지 않게 하고, Replan에서는 active mode에 따라 transit에서는 object를 고정하고 transfer에서는 grasp transform을 보존하도록 제약을 거는 표준 motion planning으로 연결을 실현합니다.

- **Empirical Impact**: RE4는 Push-T와 Robomimic의 state-based 및 image-based 벤치마크에서 기준 방법들과 비교해 성능을 유지하거나 능가하면서, 학습 비용을 크게 줄이는 방향성을 보였습니다. 특히 이미지 기반 Push-T에서 데이터가 희소한 영역을 겨냥한 adversarial benchmark에서 robust함을 보여주며, 저데이터(low-data) 실험에서도 black-box 계열이 취약해지는 조건에서 상대적 이점이 나타났다는 신호를 제시합니다. 결과적으로 생성 정책의 ‘블랙박스화’ 없이도 간단한 해석 가능한 구성 요소(검색·재구성·모드 제약 재계획·롤아웃)로 조작 기술을 학습할 수 있음을 실증적으로 뒷받침합니다.



### PDS Joint: A Parametric Double-Spiral Joint Tailored for Dexterous Hands (https://arxiv.org/abs/2606.24377)
- **Prior Approaches**: 기존 연성 관절은 엘라스토머, 모놀리식 플렉처, 롤링 컨택 등으로 구현되지만, 재현성·하중 지지 강성·조립 난이도 사이에서 트레이드오프가 생기고 주로 단일 굽힘 모드에 지배되는 경우가 많았습니다. 또한 나선형(spiral) 기반 요소는 큰 변형을 활용하지만, 손가락 각 관절이 요구하는 방향별 강성 분포를 ‘파라미터-역학’으로 체계화해 설계하는 방법은 부족했습니다. 상태 추정을 위해 다양한 센서를 쓰지만 온도/히스테리시스, EMC 간섭, 자기장 영향, 신호 손실 등 제약이 있어 접촉이 많은 환경의 신뢰성 확보가 어려웠습니다.

- **Core Contribution**: 이 논문은 PDS joint(Parametric Double-Spiral compliant joint)를 제안해, 손가락의 flexion/extension, abduction/adduction, pronation/supination에 맞춰 방향별 강성을 기하 파라미터로 체계적으로 형상화합니다. Archimedean spiral과 logarithmic spiral 템플릿을 서로 다른 관절(DIP/PIP/MCP vs 엄지 CMC)에 적용하고, asymmetry ratio로 그립 안정성과 hyperextension 저항을 함께 튜닝하도록 설계 방향을 제시합니다. 나아가 PDS 관절을 실제 사용 가능하게 만들기 위해 embedded inductive proprioception과 ArUco 기반 learning-based calibration 파이프라인을 공동 설계해 관절 상태 추정의 신뢰성을 높였습니다.

- **Technical Challenges**: 핵심 난제는 (1) 큰 스트로크에서도 관절별 방향성 강성을 의도대로 만들고, (2) 복잡한 변형 하에서 센서 신호가 단조 관계를 깨뜨릴 수 있어 상태 추정이 흔들린다는 점입니다. 연구진은 double-spiral의 기하 파라미터(θmax, 두께 t0, 기울기 k, asymmetry ratio)를 통해 강성 landscape를 실험적으로 매핑하고, 비선형/결합 효과를 포착하기 위해 ArUco로 정렬한 데이터로 MLP가 inductive 신호에서 관절 각을 직접 예측하도록 학습했습니다. 또한 inductive sensing을 LC 탱크 기반 eddy-current 원리로 구현해 접촉/EMC 잡음 환경에서도 비교적 견고한 proprioception 신호를 확보하고, ROS 2 기반 타임스탬프 정렬로 캘리브레이션 품질을 끌어올렸습니다.

- **Empirical Impact**: 실험은 먼저 geometry 파라미터가 방향성 강성을 좌우하며, 특히 asymmetry ratio에 대해 lateral stiffness가 비단조(non-monotonic)로 변해 ‘무작정 크게’가 아니라 정교한 튜닝이 중요하다는 점을 보여줍니다. 관절 상태 추정에서는 가장 어려운 abduction/adduction 조건에서 MLP 캘리브레이션이 기존 curve fitting 대비 오차를 MAE/RMSE 기준으로 각각 41.6% 줄였습니다. 시스템 통합 결과, open-source 11-DoF dexterous hand에 PDS joint와 유도 센서를 탑재해 9종 일상 물체를 안정적으로 잡았고, Kapandji test 10/10 및 엄지 opposition·핀치·인간이 개입한 공구 조작에서도 충돌 시 안전성과 접촉 풍부한 조작 능력을 시연했습니다. 다만 비교군(강체/다른 연성 관절)과 closed-loop 정책 이득, 장기 피로 평가가 제한적이라 후속 연구에서 정량화·확장이 필요하다고 밝힙니다.



### SlipSense: Multimodal Sensing for Online Slip Detection in Legged Robots (https://arxiv.org/abs/2606.24350)
- **Prior Approaches**: 기존의 미끄러짐(slips) 감지는 주로 관절 토크나 IMU 기반의 상태추정으로 GRF를 추정한 뒤, 특히 foot velocity 같은 운동학적 신호로 슬립을 판단하는 방식이 많았습니다. 하지만 이런 접근은 잡음과 모델 오차 누적으로 인해 초기(incipient) 슬립 신호를 잡아내기까지 시간이 필요해, 조기 경보가 늦어지고 임계값을 높이면 오탐을 줄이는 대신 민감도가 떨어지는 문제가 있었습니다. 또한 시각 기반 지형 매핑은 가림(occlusion)이나 시야 저하 상황에서 성능이 크게 흔들릴 수 있습니다.

- **Core Contribution**: 이 논문은 quadruped에 커스텀 경량 센서를 달아, 힘(force) 기반으로 온라인 슬립을 감지하는 SlipSense 프레임워크를 제안합니다. 센서가 piezoresistive 압력과 관성 정보를 함께 내고, LSTM 기반 force inference로 3D GRF를 추정한 뒤, LSTM Autoencoder의 one-class self-supervised anomaly detection으로 “정상 접촉”에서 벗어난 슬립 징후를 구분합니다. 특히 파국적 불안정으로 커지기 전의 초기 슬립을 잡는 것을 핵심 목표로 합니다.

- **Technical Challenges**: 기술적으로는 (1) 동적 보행 중에도 신뢰할 수 있는 GRF 추정이 필요하고, (2) 레이블된 슬립 데이터를 대량으로 얻기 어렵다는 문제가 있습니다. 이를 위해 센서 설계에서는 IMU를 센서 보드 중앙에 통합하고, 다이내믹 접촉을 견디도록 캡 제거/실리콘 포팅/오버몰딩으로 내구성을 강화했으며, 시계열 변형 이력을 반영하는 LSTM으로 재료의 점탄성(viscoelasticity)까지 고려해 GRF를 추정합니다. 학습은 안정 접촉 데이터만으로 LSTM Autoencoder를 self-supervised 방식으로 학습해, reconstruction error의 99.5th percentile을 anomaly threshold로 써서 슬립을 온라인 판별하도록 구성했습니다.

- **Empirical Impact**: 실험은 Unitree Go1을 motion capture 및 바닥 force plate 환경에 올려, 미끄러운 구간(저마찰)과 일반 구간(고마찰)에서 전진 보행을 수행하며 검증했습니다. 그 결과 SlipSense는 kinematic baseline 대비 정확도(85.9% vs 69.3%)가 높고, 특히 탐지 분해능에서 작은 슬립도 더 잘 잡아 평균 최소 변위 24.1±6.4mm까지 조기 감지했으며 이는 baseline 대비 3.3배 개선(상대 정확도도 24% 향상)입니다. 저변위 슬립에 대한 오탐/미탐이 줄어들어, 이후 컨트롤러가 마찰 추정과 제약 조정으로 안정성을 높일 수 있는 기반 기술로 의미가 큽니다.



### RoBoSR: Structured Scene Representations for Embodied Robotic Reasoning (https://arxiv.org/abs/2606.24338)
- **Prior Approaches**: 기존 embodied reasoning 연구는 인간 데모에 기반한 순차 편향(demonstration-driven sequential biases)이나 end-to-end Vision-Language-Action(VLA) 같은 방식에 크게 의존해 왔다. 데모 기반은 학습 분포에서는 잘 동작하지만 open-ended, long-horizon 작업에서 실행 경로가 바뀌면 유연성이 떨어질 수 있고, VLA는 픽셀/잠재공간 병목으로 인해 물리적 관계가 잡음과 얽히며 분포 변화에 취약해질 수 있다. TAMP 계열은 명시적 구조로 추론하지만 수작업 규격과 경직된 형식 제약으로 확장성이 제한된다.

- **Core Contribution**: 이 논문은 RoBoSR을 제안하며, 조작을 “그래프 공간의 단계별 상태 전이”로 모델링하는 intermediate structural representation을 핵심으로 둔다. semantically grounded, object-centric scene graph를 perception-action 인터페이스에 두어 원시 입력에서 추상화된 상태 추론을 분리하고, precondition/effect/goal state를 중심으로 인과적(subtask dependency) 추론을 가능하게 한다. 또한 이를 학습하기 위해 Manip-Cognition-1.6M을 구축해 scene understanding, instruction interpretation, subtask planning을 함께 감독한다.

- **Technical Challenges**: 문제는 (1) 장기 작업에서 “상태 일관성”을 유지하는 구조 추론을 만들고, (2) 지시-상태 정렬과 인과 전이를 학습하며, (3) 실행 시 객체 식별 오류나 멀티 액션 환각을 억제하는 것이다. 저자들은 grounded scene graph를 3D 작업공간의 주요 state space로 삼고, 목표조건 다음-행동 예측, forward scene prediction(상태 전이), inverse action reasoning(다단 전이 역추론), goal-state abstraction(목표 그래프 추론)을 다목적 학습으로 결합한다. 이후 SFT로 구조적 추론 능력을 먼저 만들고, GRPO 기반 RFT에서 object-grounding 일관성, 단계별 단일 원자 행동, 종료조건 일치(termination reward)를 보상으로 강제해 closed-loop 추론을 정렬한다.

- **Empirical Impact**: GSR-bench에서 RoBoSR-8B는 spatial-aware sequencings와 goal-conditioned generalization에서 prompting 기반 LLM과 classical TAMP 대비 안정적으로 우수한 Task Progress를 보였고, zero-shot에서도 난이도 증가에 따른 성능 저하가 상대적으로 작았다. 특히 36개 in-domain demonstration만 추가한 RoBoSR-8B-FT는 어려운 goal-conditioned 추론에서 의미 있는 향상을 보여 데이터 효율성을 입증한다. 실세계 VR 데모 기반 실험에서도 instruction–state alignment, 인과 제약 상태공간 추론, 실행 편차에 대한 state-consistency correction까지 각각의 핵심 역량을 검증하며, “구조적 중간 표현이 장기 embodied reasoning을 스케일하는 중요한 귀납 편향”임을 강조한다.



### Grounding Generative Policies in Physics: Optimization-Guided Diffusion for Robot Contro (https://arxiv.org/abs/2606.24208)
- **Prior Approaches**: 확산 모델은 고차원·다중모달 분포에서 샘플을 잘 생성하지만, 로봇 배치 시 필요한 도달성·충돌회피·폐루프 실행가능성 같은 제약을 위반할 수 있다. 이를 보완하려고 projection, barrier, constraint-aware training, gradient guidance, post-hoc projection 등 다양한 guidance/제약 기법이 제안됐으나 대체로 재학습·세팅별 튜닝이 필요하거나, 강한 그라디언트가 학습된 prior(기본 분포)에서 샘플을 멀어지게 만들 수 있다. 특히 embodiment를 달리 옮겼을 때도, 생성 예측을 “실행 가능”으로 보장하는 통합 추론 단계가 부족했다.

- **Core Contribution**: 이 논문은 DDIM의 역과정에서 샘플링 시 쓰이던 “교란(perturbation)”을 미리 정한 확률적 항이 아니라, 추론 시 최적화 변수를 통해 보정하도록 바꾸는 inference-time 최적화 프레임워크를 제안한다. 이렇게 하면 diffusion 모델은 고정(frozen)한 채로, 역과정 각 단계에서 reachability·collision-avoidance·controller-level trackability 같은 제약을 하드 제약(또는 소프트 패널티)로 직접 반영할 수 있다. 핵심은 샘플의 fidelity를 해치지 않으면서도 prior에 가까운 보정만 허용하는 정규화와 함께 제약 만족을 강제하는 구조를 만든 데 있다.

- **Technical Challenges**: 제약을 “그냥 유도”가 아니라 “명시적으로 강제”하면서도, 확산 prior에서 과도하게 벗어나지 않게 만드는 것이 어려움이다. 저자들은 DDIM 역업데이트에서의 교란 항을 δ_k로 치환하고, ||δ_k||^2에 기반한 정규화로 prior 경로 근접성을 유지하되, J(x_k) 형태의 feasibility cost를 통해 수렴 목표를 설계한다. 또한 NLS(비선형 최소제곱) 형태로 relaxation(Theseus)을 제공하거나 IPOPT 같은 경성 제약 풀이로 terminal feasibility를 맞추는 등, 제약 함수의 미분가능성/런타임 제약에 맞춰 해결 방식을 분기했다.

- **Empirical Impact**: 실험은 (1) 2개 로봇 팔에서의 dexterous grasp synthesis, (2) 이미지 조건 dynamic manipulation에서 controller-level executability를 평가한다. grasp에서는 최적화-유도 denoising이 reachability·충돌회피를 확보하면서도 grasp quality를 크게 유지하며, task success가 최우수 베이스라인 대비 dexterous grasping에서 최대 20pp, visuomotor manipulation에서 최대 23pp 향상됐다. 다만 hard 제약(예: IPOPT)은 연산시간이 커 Theseus 기반 relaxation이나 L-BFGS 같은 경량 온라인 재계획을 조합해 실용성을 확보한 점도 함께 강조된다.



### The Evaluation Cost of Task Specialization in Evolutionary Multi-Robot Systems (https://arxiv.org/abs/2606.24191)
Comments:
          Accepted for publication at GECCO '26 Companion: Proceedings of the Genetic and Evolutionary Computation Conference Companion. Supplementary video: this https URL

- **Prior Approaches**: 기존 연구는 진화 최적화 등을 통해 작업 분업(task specialization)이 자연스럽게 출현할 수 있음을 보여왔고, 특히 서브태스크 동작이 조립 블록처럼 제공될 때 분업이 유리하다고 주장해왔다. 다만 서브태스크마다 별도의 평가 예산을 써야 해서, 고정된 총 evaluation budget 하에서는 각 컨트롤러에 예산이 분산되는 불리함이 생긴다. 반면 generalist는 하나의 동작만으로 예산을 전부 써 최적화할 수 있어 비교가 까다롭다.

- **Core Contribution**: 이 논문은 foraging 시나리오에서 task-specialist와 generalist를 ‘평가 예산 관점의 비용-편익(cost-benefit)’으로 정면 비교한다. 목표는 분업 자체가 언제 잘 생기느냐가 아니라, ‘사전 분할된(interdependent) 서브태스크’에 대한 specialist 동작을 진화해 generalist를 이기려면 총 evaluation budget이 얼마나 필요한지 규명하는 것이다. 결과적으로 MRS 크기가 커질수록 specialist가 generalist를 능가하는 데 필요한 총 예산이 줄어든다는 경향을 제시한다.

- **Technical Challenges**: 핵심 technical challenge는 고정된 총 evaluation budget EE 안에서 서브태스크별로 예산을 나누면 학습이 불리해질 수 있는데도, specialist가 실제로 더 적은 예산으로 성능 격차를 만들 수 있는 조건을 찾아내는 것이다. 이를 위해 physics-based 시뮬레이터 ARGoS에서 사전 분할된 dropper/collector(2개 서브태스크)와 전체를 한 번에 수행하는 generalist(1개)를 진화 비교했고, 각 행동의 effective budget을 E/n으로 공정하게 배분해 최적화 비용을 통제했다. 또한 진화 후 60초 post-evaluation에서 Mann-Whitney U test 및 Holm–Bonferroni 보정으로 break-even 지점을 통계적으로 정의했다.

- **Empirical Impact**: 실험에서 specialist 동작은 일반적으로 약 2000 generations 내 수렴하며, MRS 규모 S가 커질수록 break-even evaluation budget이 낮아졌다. 구체적으로 S=2,4,6,8에서 specialist가 generalist 이상이 되는 대략적 예산 임계는 각각 ~2500, 500, 100, 30 수준으로 관측됐고, 해당 구간 이후 성능 우위가 유지되는 경향이 보고됐다. 즉, 팀 규모가 커질수록 ‘서브태스크 분할로 인한 예산 분산 비용’보다 ‘분업을 통한 효율 이득’이 더 크게 작동할 수 있음을 실증적으로 보여주며 MRS 최적화 설계에 직접적인 지침을 제공한다.



### NavWM: A Unified Navigation World Model for Foresight-Driven Planning (https://arxiv.org/abs/2606.24101)
Comments:
          13 pages, 5 figures, accepted to ECCV 2026

- **Prior Approaches**: 기존 시각 기반 내비게이션은 카메라 관측에서 행동으로 바로 매핑하는 반응형 정책이 많아, 미래를 보지 못해 myopic(단견) 의사결정과 mode collapse로 탐색이 불안정해지기 쉽다. World model을 쓰더라도 perception·generation·control을 분리 학습하는 모듈형 파이프라인이 공통의 시공간 동역학 시너지를 놓친다는 한계가 지적된다. 또한 단일 궤적을 예측해 멀티모달 미래를 충분히 다루지 못해 국소 최적에 갇히는 문제가 남아 있다.

- **Core Contribution**: 본 논문은 NavWM( Unified Navigation World Model )로, 지각·시각 생성·제어를 하나의 네트워크에서 통합 학습하는 시공간 일치형 내비게이션 world model을 제안한다. Latent World Tokens로 기하·의미의 구조적 선험을 압축하고, anchor-based multimodal trajectory forecasting으로 다양한 행동 후보를 생성해 폐루프(closed-loop) 계획에 필요한 시각적 foresight를 제공한다. 결과적으로 에이전트가 미래 비전 관측을 시뮬레이션하며 목표 정합성이 높은 경로를 고르는 방식으로 더 멀리 내다보는 계획을 수행한다.

- **Technical Challenges**: 핵심 기술적 난점은 (1) 장기 예측에 필요한 기하·의미 추상화를 안정적으로 학습하는 것, (2) 단일 궤적 예측에서 오는 mode collapse를 피하고 물리적으로 그럴듯한 멀티모달 미래를 뽑는 것, (3) 생성 모델이 예측된 행동의 오차에도 견디며 폐루프 계획으로 동작하도록 train–inference discrepancy를 줄이는 것이다. NavWM은 Depth Anything V2와 SAM에서 distill한 기하/의미 pseudo-label로 Latent World Tokens을 감독하고, scale ambiguity는 scale-invariant loss로 완화한다. 또한 Flow Matching 기반의 controllable visual generation과 두 단계 학습(teacher forcing 후 예측 샘플 조건 fine-tuning)을 결합해, world model planner가 잘못된 행동 가정에도 견고하게 미래를 시뮬레이션하도록 만든다.

- **Empirical Impact**: 실험은 Go Stanford, SCAND, RECON, HuRoN, Tartan Drive 등 로보틱스 데이터셋에서 수행됐고, PSNR이 14.17→17.34로 생성 품질이 유의미하게 개선됐다. Image Goal Navigation에서 성공률이 seen 66%→72%, 특히 unseen 환경 zero-shot에서 44%의 성공률을 보여 기존 대비 높은 성능을 달성했다. 또한 ablation과 다양성/정확도 분석에서 anchor 기반 멀티모달 예측이 모드 붕괴를 줄이면서도 trajectory 다양성(APD)을 높여, world model planner와의 궁합이 좋다는 점을 확인했다고 요약된다.



### DynaWM: Dynamics-Aware Distillation with World Model and Momentum Targets for Smooth Locomotion over Continuous Stairs (https://arxiv.org/abs/2606.24089)
Comments:
          Comments: 8 pages, 7 figures, accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

- **Prior Approaches**: 기존 연구는 교사-학생 distillation(CTS)로 지형 높이/접촉 정보 같은 특권(privileged) 정보를 학습에만 쓰고, 배치 시에는 proprioception만으로도 지형을 “알아보게” 하는 접근을 주로 썼습니다. 하지만 교사 인코더가 보상 최적화(PPO 등) 중심으로 학습되면 dynamics-aware 표현이 약해져 즉시 보상과 무관한 기하 정보를 놓치기 쉽습니다. 또한 교사/학생이 동시 업데이트되면서 non-stationary 타깃이 발생해 학생 표현이 축소(dimensional collapse)되거나, 잠재공간이 블랙박스로 남아 지형 높이 모델링 신뢰성을 검증하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 논문은 연속 계단(traversal over continuous stairs)을 목표로, 지형의 forward dynamics를 의식하는 표현을 학습하는 DynaWM을 제안합니다. 세계 모델(world model)을 정규화 항으로 도입해 인코더가 미래 상태 예측에 필요한 지형 기하를 유지하도록 만들고, PCA 기반 평가로 계층적(계단 높이 중심) 인코딩을 가시화·검증합니다. 더불어 momentum target encoder를 넣어 distillation 타깃을 안정화해 표현 붕괴 위험을 줄입니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 보상 최적화만으로는 지형 기하의 dynamics를 충분히 담지 못하는 점, (2) 교사 표현이 빠르게 변할 때 학생이 비정상 타깃에 휘둘려 dimensional collapse가 생기는 점, (3) 인코딩이 실제로 “높이”를 얼마나 담는지 확인하기 어려운 점입니다. 논문은 세계 모델 예측 손실로 forward-dynamics aware 정규화를 걸고, BYOL에서 가져온 momentum target encoder(EMA)로 비정상성을 완화했으며, PC-Correlation·Linear Probing·CCA 같은 지표와 PCA 시각화로 해석 가능성도 확보했습니다.

- **Empirical Impact**: 시뮬레이션(IsaacGym)과 실제 하드웨어 배치에서 DynaWM은 CTS 및 여러 ablation 대비 연속 계단에서 더 높은 성공률과 더 매끈한 모션(에너지/가속 관련 지표 포함)을 보였습니다. 특히 step width/height 변형이 큰 조건에서도 성능이 유지되어, 계층적 지형 인코딩이 실제 제어의 smoothness와 적응성에 연결됨을 실증했습니다. 또한 PCA 기반 정성/정량 평가에서 지형 높이·마찰 정보가 주된 분산 축에 정렬되는 경향과 학생-교사 정렬(CCA)이 개선되어, “지형을 제대로 배웠다”는 근거를 함께 제시합니다.



### MinInter: Minimizing Trajectory Interpolation During Data Augmentation for Imitation Learning (https://arxiv.org/abs/2606.24078)
Comments:
          Accepted by IEEE CASE 2026

- **Prior Approaches**: 모방학습은 로봇이 데모에서 조작 기술을 배우지만, 고품질 데모 확보 비용이 커서 데이터 증강이 핵심으로 떠올랐다. 특히 trajectory-level augmentation은 전문가 궤적을 재조합해 초기 상태를 바꾸며 새 데이터를 만들지만, 서로 다른 구간을 연결할 때 삽입되는 보간(interpolation)이나 비전문 transition이 부자연스러운 동작을 만들 수 있다. 이로 인해 합성 데이터 품질이 떨어지고, 결과적으로 downstream 정책 성능도 제한될 수 있다는 문제가 제기되어 왔다.

- **Core Contribution**: 이 논문은 Minimizing Interpolation(MinInter)을 제안해, 궤적 생성 시 추가되는 interpolation의 총량을 최소화하도록 소스 데모를 선택한다. 샘플링된 초기 구성(initial configuration)마다 완전한 궤적을 만들기 위해 필요한 interpolation이 가장 적은 데모를 고르며, 생성 파이프라인 호환성은 유지한다. 즉, 기존 재조합 기반 데이터 증강의 ‘연결 구간 품질 저하’ 지점을 직접 겨냥한다.

- **Technical Challenges**: 핵심 기술 난제는 재조합 과정에서 불연속 구간을 자연스럽게 이어야 하지만, 그때 발생하는 interpolation이 데이터 품질을 해칠 수 있다는 점을 정량화하고 최소화하는 것이다. MinInter는 end-effector pose의 translation과 rotation을 가중치 w로 함께 반영하는 interpolation metric을 정의하고, 인접 세그먼트 간 보간 비용의 합(총 interpolation)을 계산해 소스 궤적을 선택한다. 또한 생성 실패 조건을 만족하지 못하면 초기 상태를 다시 샘플링하며, 세그먼트를 서로 다른 소스에서 섞지 않는 제약으로 일관성도 확보한다.

- **Empirical Impact**: MimicGen 벤치마크의 12개 태스크·26개 변형에서 MinInter는 데이터 생성 성공률과 정책 성공률을 함께 일관되게 끌어올린다. 평균적으로 데이터 생성 성공률은 +12.8% 개선되었고, contact-rich·long-horizon·고분산 설정에서 특히 큰 폭(예: Hammer Cleanup +50.1%, Nut Assembly +31.1%)의 향상이 나타났다. 정책 성공률도 평균 +4.8% 향상되며 최대 +20.7%까지 개선되었고, SkillGen과 비교해 더 높은 성능(+4.9% 개선, state-of-the-art)을 보이면서도 구조는 상대적으로 단순하다는 점이 강조된다.



### SPACE: Enabling Learning from Cross-Robot Data Toward Generalist Policies (https://arxiv.org/abs/2606.24049)
Comments:
          Project page: this http URL

- **Prior Approaches**: 로봇 학습에서는 다양한 환경/embodiment 데이터를 모아 일반화 가능한 정책을 만들려는 흐름이 주류가 됐습니다. 대표적으로 behavior cloning으로 데이터의 동작(대개 제어 명령)을 그대로 흉내 내지만, 여러 로봇을 섞으면 ‘같은 움직임을 만들기 위한 명령’이 로봇마다 달라져 감독 신호가 흔들립니다. 기존 방법은 로봇별 action head 분리, embodiment 조건화, 또는 latent action처럼 공유 표현을 쓰더라도 목표 로봇에서 fine-tuning이 필요하거나 제어기/동역학 변화에 약할 수 있습니다.

- **Core Contribution**: 이 논문은 제어기 의존성을 줄이기 위해 ‘Cartesian state delta’를 보편 행동 표현으로 쓰고, 이를 실제 로봇 제어명령으로 바꾸는 Action Adapter를 결합한 SPACE를 제안합니다. SPACE의 핵심은 (1) end-effector의 기하학적 변위(상태 변화)를 예측하는 정책과 (2) 대상 로봇에 맞춘 선형 변환으로 실행을 연결하는 프레임워크입니다. 즉, 정책은 로봇 공통의 기하학적 목표를 예측하고, 실행 단계에서 로봇별 역학 차이를 흡수하도록 설계돼 있습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘예측한 Cartesian state delta를 그대로 실행하면’ 대상 로봇에서 원하는 궤적/도달이 보장되지 않는다는 점입니다(제어 추종 오차와 로봇/컨트롤러의 영향). 이를 위해 Action Adapter를 least-squares로 캘리브레이션하고, 배치 간/배치 내 동역학 변화에 대응하도록 deployment 중 LMS(online 업데이트)를 수행해 오차를 줄입니다. 또한 단순 출력이 아니라, 목표 변위에 대응하는 제어명령을 로봇별로 생성하되 온라인에서 편향 b를 포함한 변환을 지속 보정하도록 구성했습니다.

- **Empirical Impact**: 실험에서 SPACE는 제어명령(command) 직접 예측 정책 대비 cross-embodiment 및 cross-hardware 상황에서 유의미하게 성능을 끌어올렸고, 명령 기반 학습은 다른 하드웨어에서 큰 실패로 이어졌습니다. 특히 FR3에서 UR5 데이터를 학습해 Cloth 작업을 zero-shot으로 전이할 때 SPACE는 높은 성공률을 보인 반면, 제어명령 기반은 under-reaching으로 0%에 가까웠습니다. 더 나아가 제어 주기(control frequency), payload(물체 무게), controller gains가 바뀌는 deployment shift에서도 SPACE는 온라인 적응으로 성능 저하를 크게 억제했으며, 제어명령 정책은 성공률이 급락했습니다.



### TurboMPC: Fast, Scalable, and Differentiable Model Predictive Control on the GPU (https://arxiv.org/abs/2606.24039)
- **Prior Approaches**: 기존 MPC는 신뢰성이 높지만, 고성능 솔버가 CPU에 최적화돼 batching·병렬 롤아웃·미분가능 학습 파이프라인과 결합이 어렵다는 한계가 있었다. GPU 미분가능 MPC는 병렬 처리로 확장 가능성을 보였으나, implicit integrator·cross-time 비용·state inequality·슬랙 변수 같은 표현력을 제한하는 경우가 많았다. 또한 학습용 JAX의 오버헤드와 CUDA 커널 효율 사이의 간극, 그리고 forward/backward에서 달라지는 희소구조 때문에 통일된 고성능 구현이 난도가 컸다.

- **Core Contribution**: TurboMPC는 MPC를 GPU에서 end-to-end로 푸는 differentiable MPC solver로, state/control inequality 제약, implicit integrator, cross-time-coupled costs, slack variables를 함께 지원한다. 최적화는 SQP로 외부 루프를 구성하고, 내부 QP는 ADMM 기반으로 해결해 실제 로보틱스에서 필요한 제약 표현을 유지한다. 여기에 implicit differentiation과 JAX-CUDA co-design을 결합해, 빠른 온라인 제어와 미분가능성을 동시에 노린 것이 핵심 기여다.

- **Technical Challenges**: 핵심 과제는 (1) GPU에서 빠르게 수렴하는 QP/linear algebra를 구성하면서 (2) 미분 시 backward의 KKT 선형계 희소구조가 forward와 달라져도 일관된 미분을 제공하는 것이었다. TurboMPC는 ADMM의 Schur complement로 forward의 block-tridiagonal 구조를 활용하되, backward에서는 그 구조가 깨지는 점을 고려해 cuDSS 기반 sparse direct factorization을 forward/backward 공통 primitive로 선택했다. 또한 JAX 프런트엔드에서 선형화·배치를 처리하고, CUDA에서 ADMM 루프를 fused 커널로 구현해 반복 최적화의 GPU 효율을 끌어올렸다.

- **Empirical Impact**: 시뮬레이션에서 TurboMPC는 constrained planning, humanoid imitation learning, neural-network cost function을 쓰는 강화학습 등 다양한 작업에서 CPU 및 기존 GPU 미분가능 솔버 대비 최대 15×, 58× 속도 향상을 보였다. 차량 레벨 배치에서는 minimum-time racing에 TurboMPC를 적용하고, Bayesian optimization을 batched·GPU-가속 방식으로 튜닝해 hand-tuned 기준선보다 더 빠른 주행을 달성했다. 더 나아가 planning horizon을 8000개 이상의 knot points로 늘려도 차량 제어를 유지해, 장기 지평 확장성과 실주행 가능성을 동시에 입증했다.



### Sim-to-Real Betting on the E-Process: Bringing "simulators" to anytime-valid confidence sequences (https://arxiv.org/abs/2606.24038)
Comments:
          Affiliated open source code at: this https URL

- **Prior Approaches**: 기존 sim-to-real 성능평가에서는 시뮬레이터의 양은 많지만 분포가 불일치한다는 점 때문에, Chen et al.처럼 베팅 게임으로 샘플을 재가중해 MSE를 줄이려는 접근이 제안됐다. 다만 산출물이 점추정치(bet-weighted estimate)라서 mean μ 자체에 대한 신뢰구간(시간에 무관한 커버리지) 인증을 제공하지 못한다. 또한 safe anytime-valid inference 계열은 μ0 후보에 대해 e-process를 누적해 confidence sequence를 만들지만, 베팅 크기를 데이터 단독의(느리고 잡음이 큰) 모멘트로 잡는 경우가 많다.

- **Core Contribution**: 이 논문은 Chen et al.의 sim-to-real 베팅 추정에 쓰인 simulator mixture의 모멘트로, Ramdas et al.식 anytime-valid confidence sequence의 베팅 크기까지 동일하게 맞춰주는 통합을 제안한다. 핵심은 신뢰구간의 유효성(커버리지)은 베팅 규칙이 “예측가능”하기만 하면 simulator bank의 품질과 무관하게 보장되지만, 구간이 얼마나 빨리 좁아지느냐(효율성)는 mixture로 속도를 끌어올릴 수 있다는 점을 활용하는 것이다. 즉, 신뢰구간의 validity는 유지하면서 tightness를 개선하는 실용적 설계를 제공한다.

- **Technical Challenges**: main challenge는 “어떤 베팅 크기든 커버리지가 보장되는가”와 “그럼에도 구간을 실제로 빨리 좁히려면 무엇이 필요한가”를 동시에 만족시키는 것이다. 논문은 confidence sequence를 구성할 때 μ0에 대한 residual에 예측가능한 Kelly형 베팅을 두고 e-process를 적절히 truncated(1+λt(y−μ0)≥δ)해 비음성을 유지함으로써 Ville 부등식 기반의 anytime-valid 커버리지를 확보한다. 이어서 베팅 크기를 데이터 러닝 모멘트가 아니라 sim에서 얻는 mixture (mt, vt)로 산정해, bank 모멘트가 충분히 맞을 때 초반부터 e-process가 더 빠르게 “틀린 μ0”에 대해 누적 증거를 쌓아 구간 폭을 줄이도록 만든다.

- **Empirical Impact**: 합성 예제에서 제안한 방식은 confidence sequence의 커버리지를 해치지 않으면서, mixture 기반 베팅으로 인해 잘못된 후보에 대한 증거 축적 속도가 빨라져 구간이 더 빠르게 수축되는 경향을 보인다. 특히 로봇 성능 테스트처럼 실제 실험 비용이 크고(샘플 수 n이 작고) long-tail이 나타날 수 있는 환경에서 mean μ에 대한 신뢰 가능한 anytime-valid 인증이 유용하다는 점을 강조한다. 또한 coverage는 simulator bank가 부정확하더라도 유지되므로, sim-to-real 불일치에 대한 위험을 효율성 측면에서만 완화하며 안정적인 안전장치로 활용될 수 있다.



### Topological Online Learning for Displacement-based Formation Contro (https://arxiv.org/abs/2606.23901)
- **Prior Approaches**: 기존 다중로봇 형성제어는 리더-팔로워, virtual structure, behavior 기반, consensus 기반, APF 등으로 안정성은 확보하지만, 외란이나 통신/상호작용 변화에 대해 실시간 적응성이 제한적이라는 한계가 있었다. 특히 consensus 계열을 변형한 방법들은 보통 고정 또는 distance-dependent 가중치를 쓰며, 로봇 개별 입력을 robust하게 만들 뿐 상호작용 토폴로지를 능동적으로 조정하지 못한다.

- **Core Contribution**: 이 논문은 displacement-based 형성제어에서 ‘상호작용 토폴로지(엣지) 가중치’를 실시간으로 학습·갱신하는 Topological Online Learning for Displacement-based (TOLD) 형성제어를 제안한다. TOLD는 엣지 수준(adaptation overlay)에서 형상 왜곡(formation distortion)을 직접 줄이도록 가중치를 업데이트하며, 기존 node-level 물리 제어기와 결합해 시너지 robust를 노린다.

- **Technical Challenges**: 핵심 기술적 과제는 directed graph에서 가중치를 온라인으로 바꾸면서도 안정성/수렴 성질을 보장하는 것이다. 이를 위해 OGF(Online Gradient Flow)는 unconstrained 가중치 업데이트로 형상 정확도를 높이는 대신 가중치가 커질 수 있고, OExpGF(Online Exponential Gradient Flow)는 non-negative convex weight를 지켜 asymptotic consensus를 이론적으로 보장하도록 설계했다.

- **Empirical Impact**: 이론적으로는 single-integrator, directed graph 조건에서 OExpGF가 비동기적 의견일치(동의) 수렴을 보장하고, OGF는 형상 왜곡이 시간에 따라 커지지 않음을 보인다. 실험·시뮬레이션에서는 간헐적 외란 하 12대 로봇에서 node-level robust 제어기와 TOLD를 결합할 때 median 누적 Root Mean Distortion Error가 OGF 약 1.4%~33.14%, OExpGF 약 1.2%~30.88% 감소했으며, Crazyflie 2.0 쿼드로터 하드웨어에서도 fixed-weight consensus 대비 median 형상 왜곡이 OGF 62%+, OExpGF 31.4% 줄었다.



### Enforcing Human-like Kinematics in Dexterous Piano Playing via Adversarial Posture Regularization (https://arxiv.org/abs/2606.23848)
- **Prior Approaches**: 강화학습 기반 로봇 피아노 연주는 주로 MuJoCo 같은 시뮬레이터에서 task reward(키 누름 정확도, sustain 등)를 최적화하지만, 희소 보상만으로는 reward hacking이 발생해 관절 과신전 같은 비자연스러운 자세를 유도할 수 있다. 또한 IK를 보조로 쓰더라도 고DoF 손(예: Shadow Hand)의 중복성 때문에 끝점(손가락 끝 2D/위치)만 맞추면 중간 관절(MCP/PIP 등)이 생체역학적으로 부자연스러운 ‘zombie hand’ 형태로 무리하게 수렴하기 쉽다. 모방학습은 대안이지만, 전체 곡에 대한 고정밀 풀 포즈 트래킹 데이터는 제작 비용과 캘리브레이션 부담 때문에 확장성이 낮았다.

- **Core Contribution**: 논문은 Adversarial Posture Regularization(APR)로, task 정확도를 넘어 ‘자세 자연스러움’의 통계적 prior를 적대적(Adversarial) 분포 매칭 방식으로 학습해 과도한 IK 해를 억제한다. 고가의 전문가 시연(곡 단위, 시간정렬 풀 포즈) 없이도 Meta Quest 3에서 얻은 소량의 자연스러운(캐주얼) 피아노 핸드 데이터를 사용하며, discriminator가 “정책 전이(상태 전이)가 인간 피아니스트의 분포처럼 보이는지”를 판별하도록 설계한다. 또한 손 형태 차이를 줄이기 위해 모폴로지 불변 retargeting(벡터 뼈 방향/국소 좌표 기반)과 손 회전·엄지 매핑 휴리스틱을 포함해 Shadow Hand로의 적용성을 높였다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 고DoF 손의 자세를 직접 감독하면 필요 데이터가 커지고, (2) 센서 잡음이 큰 VR 트래킹(프레임별 흔들림)에서도 discriminator가 안정적으로 자세 prior를 학습해야 하며, (3) 좌/우 양손의 대칭성과 학습 효율을 동시에 확보해야 한다는 점이다. APR은 프레임 정렬 없이 상태-전이 쌍의 분포로 학습해 시간정렬 부담을 줄이고, Meta Quest 3 트래킹의 노이즈에 대해 통계적으로 견고한 regularizer가 되도록 LSGAN 기반 판별자 학습과 expert gradient penalty로 학습 안정성을 확보했다. 나아가 하나의 공유 가중치 discriminator로 좌/우 입력을 동일하게 처리해 양손 학습을 더 효율적으로 만들었다.

- **Empirical Impact**: 실험에서 APR은 MuJoCo 상 Shadow Hand 듀얼 피아노 연주 벤치마크 전반에서 task accuracy(F1)를 기준선인 PianoMime과 동등하게 유지하면서도 cPSI/BSE/FAC 세 가지 인간 유사도 지표와 시각 품질을 유의미하게 개선했다. 특히 APR의 개선은 reward hacking을 줄이고 IK로 생기기 쉬운 생체역학 위반(과신전·텐던 시너지 붕괴·손가락 아크 불연속)을 완화한 결과로 해석된다. 데이터 측면에서도 약 19초 분량의 비정렬 VR 핸드 데이터로 universal하게 여러 곡에 재사용 가능함을 보여, 곡 단위 시간정렬 시연이 필요한 기존 모방/DeepMimic 계열 대비 확장성의 의미 있는 격차를 제시한다.



### Engineering Reliable Autonomous Systems: Challenges and Solutions (https://arxiv.org/abs/2606.23760)
- **Prior Approaches**: 기존 연구는 주로 전통적인 사이버-물리 시스템의 검증 관행을 자율 시스템에 그대로 이식하는 경향이 있었고, 복잡한 환경에서의 불확실성과 블랙박스 구성요소까지 충분히 다루지 못한다는 한계가 지적됐다. 또한 symbolic 방식은 상대적으로 검증이 쉬운 반면 sub-symbolic 방식은 확장성은 높지만 신뢰성 보장이 어렵고, neuro-symbolic로 간극을 메우려 해도 여전히 신뢰 문제는 미해결로 남아 있었다. 다중 로봇·재구성 시스템에서는 내부/외부 도전이 동시에 커지며, “검증 기법은 있으나 현장 적용이 낮은” 간극도 반복적으로 언급됐다.

- **Core Contribution**: ERAS(Engineering Reliable Autonomous Systems) 워크숍 보고서는 자율 시스템을 위한 verification & validation, 실세계 엔지니어링, 안전한 소프트웨어 아키텍처를 축으로 “도전 과제 카탈로그”와 “해결을 잇는 로드맵”을 제시한다. 특히 학계에선 알려져 있지만 실제 산업 적용으로 이어지지 못한 과제들과, 여전히 해결되지 않은 과제를 분리해 다음 연구·협업 경로를 구체화했다. 또한 FMAS(형식 방법)와 AREA(인지로보틱스/멀티에이전트)의 커뮤니티를 결합해, 서로가 갖는 사용사례·기법 이해의 공백을 메우려는 방향성을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 예측 불가능한 물리 환경과 black-box 부품을 모델링하는 문제, (2) AI 개발 방식(특히 sub-symbolic/학습 기반)으로 인해 생기는 신뢰성 격차, (3) 사람-로봇 상호작용·안전 요구를 보장할 증거를 설계하는 문제로 정리된다. 보고서는 이를 위해 정적 검증과 runtime verification을 함께 쓰는 접근, 아키텍처 수준 분해와 안전 아티팩트(예: GSN) 활용, 시뮬레이션·실험을 결합한 V&V 파이프라인(서로의 결과로 모델/요구사항을 보정)을 강조한다. 더불어 안전한 자율을 위해 안전 사례(safety case)를 구성하고, 테스트베드의 현실성·격리된 안전 모듈 등 실물 제약을 반영하는 방법을 연결해 제시한다.

- **Empirical Impact**: 워크숍은 11개 케이스 스터디를 통해 우주 능동 파지, 지하 광산 UAV, 의료 triage 의사결정지원, 산업 협업 로봇 안전거리 검증, 수중 케이블 추적, 자율 주행의 추가적 책임/설명 요구, 가정용 Care-O-bot 의사결정 검증, 인간 인수인계 협업(테이블 레그 handover), 건물 화재 분사, 원전 시설 루틴 점검, 전장 내(반)자율 협동 시나리오 등 다양한 도메인을 아우른다. 특히 runtime monitor로 주입 결함을 탐지하거나(우주 파지), GPS 부재·부분 관측 환경에서 fail-safe 자율의 수용성을 시험하며(광산 UAV/수중 AUV), 학습 구성요소와 닫힌 고리(closed-loop) 시스템을 함께 검증하는 사례가 제시됐다. 결과적으로 연구자와 산업이 “무엇이 아직 증거로 굳어지지 않았는지”를 공통 언어로 파악하고, 후속 연구·저널 스페셜 테마·후속 이벤트로 협업 모멘텀을 만들 수 있다는 점에서 의미가 크다.



### Verifiable Foundation Models for Robot Safety (https://arxiv.org/abs/2606.23754)
- **Prior Approaches**: 기존의 foundation model 기반 로봇 제어는 VLA 같은 end-to-end 파이프라인으로 강력한 인지·추론 성능을 보이지만, 모델 규모와 복잡도로 인해 공식 검증이 사실상 불가능하다는 한계가 컸습니다. 실험 기반 안전성 학습(안전 리워드, constrained reinforcement learning 등)은 ‘인증서’가 아니라서 분포 밖 실패 위험을 완전히 제거하기 어렵고, 런타임 shielding은 추가 비용과 과제 최적성 훼손 문제를 동반합니다.

- **Core Contribution**: 이 논문은 FEARL(Foundation-Enabled Assured Robot Learning)로, 안전 검증 가능성과 foundation model의 표현력을 동시에 확보하는 모듈형 분해를 제안합니다. 정책을 큰 Controller(C)와 작은 Safety 모듈(S)로 나누고, Safety 모듈의 저차원 안전 센서 관측에 대해서만 formal verification을 적용해 tractable한 보증을 만듭니다.

- **Technical Challenges**: 핵심 기술 난제는 safety를 결정하는 부분이 foundation model의 불투명한 고차원 입력에 얽혀 있어 검증 도구가 스케일되지 않는다는 점입니다. FEARL은 S 입력에 bounded context embedding을 사용해 verifiable 입력공간을 유한하게 만들고, 확장된 ε-ProVe로 universially quantified context와 다중 안전 제약을 동시에 다루며, 검증으로 얻은 certified safe region 바깥에서만 shielding이 선택적으로 개입하도록 설계했습니다.

- **Empirical Impact**: FEARL은 2D 시뮬레이터, Hello Robot Stretch2 실내 내비게이션, Unitree GO2 실외 내비게이션의 3개 도메인에서 서로 다른 controller backbone(BERT+ViT, LLM+ViT, SmolVLA 등)에도 유지되며 과제 성공률을 유지했습니다. Offline verification은 Safety 모듈 입력의 큰 비율을 provably safe로 인증했고, verification-guided shielding에서는 모든 환경에서 안전 위반 0건과 낮은 override·성능 저하를 달성했으며, Indoor 내비게이션 정책이 재학습 없이 물리 Stretch2로 sim-to-real 전이되는 결과도 제시됐습니다.



### Compact Object-Level Representations with Open-Vocabulary Understanding for Indoor Visual Relocalization (https://arxiv.org/abs/2606.24767)
Comments:
          Accepted by RA-L 2026

- **Prior Approaches**: 기존 실내 visual relocalization은 주로 low-level 특징에 의존해 조명 변화나 센서 노이즈에 취약했고, 의미·구성을 충분히 반영하지 못했다. 또한 점(point) 기반 맵은 오버헤드가 커지고, downstream에서 의미 객체 대신 원시 특징점에 가까워 해석성과 적용성이 떨어졌다. 최근 일부 object-level 접근도 있으나, 객체 매칭의 outlier가 크고 scalable 장면을 위한 pose prior·전용 최적화 전략이 부족했다.

- **Core Contribution**: OpenReLoc은 semantics, layout, geometry를 포함한 객체 정보를 object-level structured map으로 정리해 camera relocalization을 수행하는 프레임워크를 제안한다. 이를 위해 multi-modal open-vocabulary object matching, object-oriented reference frame 기반 pose prior, 그리고 객체 형상 정보를 활용한 dual-path 2D ICP loss로 coarse-to-fine 추정을 안정화한다. 결과적으로 의미 인식 기반의 정밀한 6-DoF pose 추정과 메모리 효율·확장성을 함께 노린다.

- **Technical Challenges**: 핵심 기술 과제는 (1) open-vocabulary 객체 매칭에서의 낮은 분별력과 outlier를 줄이고, (2) 큰 실내 장면에서 재검색 가능한 pose prior를 제공하며, (3) sparse한 객체 대응에서도 안정적으로 pose를 최적화하는 것이다. 논문은 CLIP 기반 시각-언어 객체 디스크립터와 global scene graph의 sub-graph matching을 결합해 매칭을 정교화하고, RGB 없이 관측된 object ID와 2D bbox만 담는 reference frame과 DIOU retrieval로 대규모 장면의 coarse prior를 만든다. 마지막으로 2D ICP를 forward/backward 모두에 대해 설계해 scale ambiguity를 완화하고, Huber 커널로 2D 픽셀 outlier에 둔감하도록 하여 정밀 최적화를 달성한다.

- **Empirical Impact**: 실험에서 OpenReLoc은 ScanNet·ScanNet++ 및 Habitat 기반 대규모 합성 장면(HSSD)에서 기존 object-level 및 low-level 베이스라인보다 recall과 정확도를 일관되게 개선했다. 특히 ScanNet에서 GoReloc 대비 relocalization success rate가 약 5~10배 수준으로 향상되었고, 대규모 다중 룸/다중 플로어 설정에서도 확장성과 강건성이 확인됐다. 모델이 closed-vocabulary 의존도가 낮아 복잡한 실제 환경에서도 더 많은 유효 매칭을 생성하며, 전용 최적화 loss의 부재가 초래하는 drift·수렴 불안정 문제를 줄였다는 점이 성능 차이를 설명한다.



### Parallel Dynamic Programming for Conic Linear Quadratic Contro (https://arxiv.org/abs/2606.24632)
Comments:
          This paper was accepted for presentation at the IFAC World Congress 2026 (IFAC WC 2026)

- **Prior Approaches**: MPC에서 반복적으로 풀리는 핵심 계산은 LQ(Linear Quadratic) 문제이며, SQP나 DDP 같은 방법의 하위 단계로도 자주 등장한다. 전통적 LQ 해법(리카티 재귀, banded KKT를 이용한 희소 LDL^T 등)은 예측 호라이즌이 길어질수록 직렬 복잡도가 커져 실시간 스케일링이 병목이 된다.
또한 conic 제약(2차 콘 제약 등)을 포함하면 기존 병렬화는 제약을 충분히 다루지 못하는 경우가 많았다.

- **Core Contribution**: 이 논문은 ADMM의 primal 업데이트를 LQ 문제로 재구성한 뒤, time horizon을 따라 분할하는 parallel-in-time 기법을 제안한다. ADMM 내부 LQ 업데이트를 fixed-end LQ 부분문제로 쪼개고, 각 부분문제를 동적 계획 기반 리카티 변형으로 독립적으로 풀어 병렬성을 확보한다.
그 결과 conic 제약 처리를 ADMM과 결합하면서도 리카티 구조를 유지해 효율을 끌어올린다.

- **Technical Challenges**: 가장 큰 난제는 ADMM의 제약 처리와 time horizon 분할이 결합될 때, 부분문제를 연결하는 링크 상태와 dual 변수까지 포함한 최적성 조건을 안정적으로 계산하는 것이다. 논문은 결합을 terminal 및 second-last 단계의 dualization으로 바꿔, 각 subproblem의 inner 최적화를 dynamic programming으로 풀고 conditional dual value function을 통해 해를 구성한다.
또한 고정-종단(fixed-end) 재귀가 추가 연산을 유발하므로, square-root Riccati recursion을 활용해 flop과 캐시 효율을 줄이고 load-balancing으로 스레드 분할의 불균형을 완화한다.

- **Empirical Impact**: 실험은 멀티코어 CPU(8 physical cores)에서 quadrotor 호버링(단호라이즌)과 low-thrust 궤도 이전(장호라이즌) 두 사례로 검증했으며, horizon 길이에 따라 QDLDL 및 ProxLQR 대비 최대 5x 가속을 보고한다. 특히 제약이 많은 상황에서 병렬 고정-종단 분할의 비용이 stage constraints의 연산 증가분에 의해 상쇄되어 성능 이점이 커진다.
장호라이즌에서는 load-balancing이 분할 개수에 따라 solve time을 추가로 줄여, 전체적으로 실시간 MPC/제어 최적화의 시간 병목 완화에 의미가 있다.



### Average Rankings Mask Per-Subject Optimality: A Friedman-Nemenyi Benchmark of EEG Motor-Imagery BCI Decoders (https://arxiv.org/abs/2606.24394)
Comments:
          16 pages, 6 figures, 4 tables

- **Prior Approaches**: 모터 이미지를 EEG로 분류하는 BCI에서는 CSP 같은 공간 기반 방법과 tangent-space 같은 Riemannian 방법이 기본값으로 자주 권장돼 왔다. 하지만 개인 간·세션 간 변동성이 커서, 고정된 한 파이프라인이 환경 변화에 강건하다는 주장이 실제로는 조건에 따라 달라질 수 있다는 문제의식이 축적돼 있다.
또한 time/frequency/분해 기반 특징과 Hjorth, HFD, Hurst, SVD entropy 같은 비선형 기술자, 연결성(connectivity) 특징 등 다양한 표현이 제안돼 “무엇이 언제 잘 먹히는가” 질문이 더 복잡해졌다.

- **Core Contribution**: 이 논문은 “한 파이프라인이 전반적으로 더 낫다”는 약한 형태의 주장조차 가장 유리한 평가 조건(동일 참가자·동일 세션 내 학습/검증)에서 검증한다. MOABB 프레임워크로 3개 공개 모터-이미지(left vs right) 데이터셋에서 1,056개 end-to-end 디코딩 구성(feature extractor × scaler × classifier)을 대규모로 비교해, 파이프라인 보편성의 하한선을 제시한다.
핵심 결론은 cov-tgsp와 CSP가 평균 성능 1·2위를 보이지만, 데이터셋에 따라 순서가 바뀌고 최대 집단에서는 통계적으로 동률(closed-form)이며 개인별 “승자”가 갈린다는 점이다.

- **Technical Challenges**: 가장 큰 기술적 도전은 개인 내 변동까지 포함해 파이프라인 간 성능 차이를 공정하게 비교하면서, 단순 평균이 아니라 통계적 유의성과 효과크기를 함께 보이는 것이다. 이에 따라 Friedman 검정(Omnibus)과 Nemenyi critical-difference 분석, Wilcoxon signed-rank(효과크기 포함)로 다중 파이프라인 순위를 비교하고, ‘winner-per-subject’ 및 ‘per-subject oracle(개인별 최적 선택)’까지 분해해 heterogeneity의 크기를 수치화했다.
또한 사전처리를 최소화해 공정성을 확보했지만, 비선형 기술자가 잡음/아티팩트 민감도가 더 클 수 있음을 한계로 명시하고 비선형 결과를 “개인 이질성 가설”로 해석하도록 설계했다.

- **Empirical Impact**: 실험에서 cov-tgsp와 CSP는 평균 정확도에서 앞섰지만(예: PhysionetMI에서 통계적으로 구분되지 않음), 단일 최적 파이프라인이 대부분 사용자를 커버하지 못했다. PhysionetMI의 경우 ‘한 파이프라인 고정’의 승률은 약 35% 수준이며, 개인별로 비선형 기술자가 약 1/3에서 최선이 되어 공간/Riemannian 단일 기본값의 한계를 보여준다.
반대로 feature dimensionality는 성능을 단조롭게 설명하지 못했고, classifier/scaler 선택은 강한 feature 표현을 고른 뒤에는 크지 않은 2차 요인으로 나타났다. 결과적으로 연구·제품 모두에서 “보편 디코더 탐색”보다 “참가자 인지(participant-aware) 모델 선택/개인화”에 투자할 근거가 정량적으로 강화되었다.



### From Open Waters to Enclosed Cabins: ProteusVPR for Cross-Scene Visual Place Recognition in Maritime Perception and Cabin Inspection (https://arxiv.org/abs/2606.24234)
- **Prior Approaches**: 기존 VPR 연구는 주로 전역 디스크립터 학습과 nearest neighbor 기반 검색 성능을 키우는 데 집중해 왔다. 패치/트랜스포머 기반 재랭킹이나 의미·비전언어 단서(SAM/CLIP 등)를 더해도, 결과적으로 “정확한 위치 추정”에는 한계가 남는다. 특히 선박의 개방 데크(광량 변화·텍스처 희소)와 선실(반복 구조·시각적 모호성)처럼 도메인이 크게 바뀌는 cross-scene에서는 일반화가 쉽게 무너진다.

- **Core Contribution**: 이 논문은 선박 선실 환경에 맞춰 기존 VPR 백본 위에 얹어 쓸 수 있는 2단계 retrieval-refinement 프레임워크 ProteusVPR을 제안한다. 1단계는 임의의 standard VPR 모델로 후보를 찾고, 2단계는 검색된 참조 이미지와 쿼리 이전 두 프레임을 함께 써서 상대 변위 기반의 정밀 로컬라이제이션을 수행한다. 또한 기하-시각 추정 모듈이 반복 선실에서의 검색 모호성을 일부 완화하며 localization precision을 끌어올린다.

- **Technical Challenges**: 문제의 핵심 기술 난점은 (1) 전역 검색은 도메인 변화에 취약하고 (2) 선실의 반복 구조는 동일/유사한 전역 특징을 만들어 위치가 헷갈린다는 점이다. 이를 해결하기 위해 DINOv2 ViT-B/14로 프레임 특징을 뽑되, inter-reference attention과 query-reference attention으로 참조 정보를 정교하게 융합한다. 더 나아가 로컬 affine 좌표계에서 camera azimuth(boresight azimuth)를 인코딩하고, 기준점(세 참조 이미지)의 기하량과 함께 MLP 추정으로 예측 변위를 전역 좌표로 변환해 정밀도를 높인다.

- **Empirical Impact**: 평가는 선박에서 운영 중 수집한 8K급 360도 panoramic 기반 XHZ 데이터셋에서 수행된다. XHZ는 다층 선실·데크 전이 구간을 포함하고 query-database separation을 엄격히 적용해 까다로운 벤치마크로 설계됐다. 실험 결과 ProteusVPR는 여러 VPR 백본 전반에서 평균 mean localization error를 60% 이상 줄이며, cross-scene 해상·선실 환경에서 강건한 정밀 시각 로컬라이제이션 해법임을 보였다.



### Importance of Intent-Sharing for V2X-based Maneuver Coordination (https://arxiv.org/abs/2606.24203)
- **Prior Approaches**: 기존 CAV(connected and automated vehicles) 연구에서는 원격 차량의 의도를 직접 받기보다, 현재 운동학 데이터 등을 바탕으로 궤적을 예측해 협력 기동을 수행하는 접근이 주를 이뤘다. 다만 예측 기반 방식은 의도 불확실성이 커질수록 협조 기동의 성공률을 떨어뜨릴 수 있다는 한계가 있다.

- **Core Contribution**: 이 논문은 intent-sharing(의도 공유)이 기동(manervor) 조정의 성패를 어떻게 바꾸는지, ‘성공한 조정의 비율’로 정량화해 평가한다. 특히 원격 차량이 자신의 계획을 ego 차량에 직접 전달하는 방식이, ego가 대신 원격의 궤적을 예측하는 방식보다 조정 효과를 높일 수 있음을 분석한다.

- **Technical Challenges**: 핵심 기술 과제는 의도 공유가 실제로 조정 성능을 얼마나 끌어올리는지 공정하게 비교·계량하는 것이다. 저자들은 고속도로 시나리오에서 coordinated lane changes(차로 변경 협조) 문제를 설정하고, intent-sharing의 성능을 현재 kinematic data로부터의 trajectory prediction 기반 방법과 비교함으로써 효과를 분리해 측정했다.

- **Empirical Impact**: 분석 결과 두 가지 시나리오에서 CAV들이 주변 차량의 driving intentions에 직접 접근할 수 있을 때 maneuver coordination이 실질적으로 크게 향상됐다. 연구는 maneuver coordination 프로토콜 설계에 intent-sharing을 포함하는 것이 중요하다는 점을 실증적으로 강조하며, 협조 주행의 신뢰도를 높일 방향을 제시한다.



### Efficient Time-Domain Simulation of USV Motions in Short-Crested Irregular Waves Using an IRF-Based Framework (https://arxiv.org/abs/2606.24130)
- **Prior Approaches**: 기존 선박 운동의 시간영역 예측은 불규칙 파를 여러 정현파 성분으로 분해해 응답을 중첩하는 방식이 많아, 장시간 시뮬레이션이나 실시간 적용에서 계산 비용이 커진다. 특히 USV의 파랑 응답 예측은 해상상태 평가, 시뮬레이션 기반 시험, 제어 개발에 필수지만 효율성과 현실성이 동시에 요구된다.

- **Core Contribution**: 본 논문은 impulse response function(IRF) 기반 시간영역 프레임워크로 짧게 갈라진(short-crested) 불규칙 파에서 선박 운동을 예측한다. 주파수 영역에서 Froude-Krylov, diffraction, radiation 하중을 구한 뒤 시간영역으로 변환하고, 힘을 convolution 기반으로 재구성해 반복적인 정현파 시뮬레이션 필요성을 줄인다. 또한 약한 비선형 복원력을 즉시 접수면 압력 적분으로 반영하고, 방향 파 스펙트럼으로 실제 해상 상태를 표현한다.

- **Technical Challenges**: 핵심 난제는 주파수 도메인의 선형(및 약한 비선형) 파랑 하중 정보를 시간영역에서 안정적으로 합성해, 불규칙 파에 대한 순간 응답을 정확히 재구성하는 것이다. 논문은 방사(radiation) 포함 하중을 IRF로 시간 변환하고 convolution 기반 힘 재구성으로 즉시 응답을 계산하는 구조를 채택했으며, 접수면 압력 적분으로 약한 비선형 복원 효과까지 통합했다. 더불어 방향 파 스펙트럼을 이산화할 때의 정확도-연산비용 트레이드오프도 함께 분석한다.

- **Empirical Impact**: 제안 프레임워크는 롱크레스트 빔 불규칙파에서의 해상 실험(모형시험)과, 실제 해상 조건에서 운항하는 USV의 전범위(full-scale) 계측 데이터로 검증됐다. 유의 진폭, mean zero-crossing period, 표준편차, 운동 시간 이력 등 주요 통계와 궤적이 실측과 잘 일치하며 예측 신뢰성을 보여준다. 방향 스펙트럼 이산화의 영향은 진폭에 중간 정도 민감하지만 주기는 상대적으로 둔감했으며, 30 deg 방향 구간이 정확도와 계산비용의 실용적 균형으로 제시된다.



### ObsGraph: Hierarchical Observation Representation for Embodied Reasoning and Exploration (https://arxiv.org/abs/2606.24068)
- **Prior Approaches**: 기존 embodied reasoning/exploration 연구는 dense 3D 지도나 scene graph, 그리고 이미지 메모리(일부는 task-conditioned)를 활용해 왔다. 하지만 dense 3D는 추상화가 부족해 task-조건 검색이 비효율적이고, 계층형 scene graph는 미래 태스크에 필요한 시각 증거를 일찍 압축해 놓칠 수 있다. 또한 VLM 기반 탐색은 next-target을 매 단계에 정하더라도, 현재 지식 상태에서 “얼마나/어디까지” 더 관측해야 하는지에 대한 스코프(탐색 스케일) 적응이 충분히 다뤄지지 않는 한계가 있었다.

- **Core Contribution**: 이 논문은 관측 자체를 중심으로 계층을 구성하는 observation-centric 계층형 장면 그래프 ObsGraph를 제안한다. room–view–object 3층으로 시각 증거를 보존하면서도, room은 거친 의미 앵커, view는 object co-visibility 기반 문맥, object는 fine-grained 디테일을 담당해 task-aware 추론과 정보 접근을 한 구조 안에서 연결한다. 더 나아가 retrieval 결과가 탐색 후보 공간(방/뷰/프론티어 탐색) 자체를 구조화하도록 설계해 representation–retrieval–exploration의 결합을 강화한다.

- **Technical Challenges**: 핵심 난제는 (1) 장면을 충분한 증거로 압축하되, (2) 추상화 수준에 맞춰 task-relevant 정보를 정확히 검색하고, (3) 검색 부족분을 어떤 스케일의 탐색 옵션으로 보완할지 결정하는 것이다. 이를 위해 view layer는 object co-visibility의 다양성을 유지하는 방식으로 redundancy를 줄이도록 설계했고, room layer는 VLM 예측을 확률적(베이지안) 결합해 room–object 할당의 강건성을 확보했다. retrieval은 장면 그래프 계층을 따라 LLM 호출 예산(bounded budget) 안에서 점진적으로 후보를 좁히고, 그 결과로 “room-level, view refinement, frontier exploration” 중 어떤 옵션을 활성화할지 결정하는 구조로 해결한다.

- **Empirical Impact**: EM-EQA, A-EQA, GOAT-bench에서의 실험은 ObsGraph가 기준선 대비 성공도와 효율을 함께 개선함을 보여준다. 특히 EM-EQA에서는 3D-Mem 대비 accuracy가 더 높으면서도 VLM 질의 프레임은 거의 늘지 않아(프레임 효율) “증거 보존+계층적 필터링”의 이점을 입증했다. A-EQA에서는 accuracy와 LLM SPL을 동시에 개선했으며, 이는 검색이 맞춰준 증거 갭이 탐색 결정을 더 VLM 추론에 정렬된 방식으로 유도했기 때문으로 해석된다.



### Critique of Agent Mod (https://arxiv.org/abs/2606.23991)
- **Prior Approaches**: 최근 LLM을 ‘coding agents’, ‘AI co-scientists’처럼 에이전트라 부르며 생산성을 높이려는 시도가 늘었지만, 많은 시스템은 미리 짜둔 툴·워크플로·프로그램 제어 루프를 외부에서 오케스트레이션하는 agentic 성격이 강하다. 그 결과 목표나 정체성, 의사결정의 모드 전환, 학습의 시작/중단 같은 핵심 구조가 모델 내부에서 내재화(endogenously)되기보다 파이프라인 설계에 의존하는 경향이 있다. 논문은 이러한 방식이 다양한 환경 적응과 진짜 자율성(agency)의 일부만 다룰 뿐, 생물학적 에이전트처럼 열린 세계에서 스스로 조직되는 역량을 설명하기 어렵다고 지적한다.

- **Core Contribution**: 논문은 ‘에이전트(agency)’를 단순 작업 수행 능력(operation excellence)과 분리해, 목표 지향 행동, 정체성의 진화, 자기조절, 자기성찰 및 학습이 내부에서 발생해야 한다고 주장한다. 이를 위해 agentic 시스템(외부 공학으로 워크플로를 조립)과 agentive 시스템(모델 내부에서 구조가 내생적으로 발생)을 경계짓고, 에이전트 아키텍처를 goal, identity, decision-making, self-regulation, learning의 5차원으로 분석한다. 나아가 범용 에이전트 모델을 위한 Goal-Identity-Configurator(GIC) 아키텍처를 제안하며, 계층적 목표 분해·진화하는 identity·월드 모델 기반 simulative reasoning·학습/추론 깊이를 조절하는 configurator·실제/시뮬 경험에서의 self-directed learning을 한 틀로 결합한다.

- **Technical Challenges**: 기여의 핵심은 ‘월드 모델과 에이전트 모델을 기능적으로 분리’하는 데 있으며, 둘을 합치면 next-state 예측과 reward/행동 선택이 뒤섞여 계획과 시뮬레이션의 신뢰성이 흔들릴 수 있다고 본다. 또한 장기 목표와 identity를 내부 잠재변수로 두고, 그에 조건화된 계획·실행·메타 의사결정이 어떻게 학습 가능한 형태로 구성될지 해결해야 했다. 논문은 simulative reasoning(System II)으로 world model에서 미래를 예측·계획하고, actor(System I)로 즉각 실행을 담당하며, configurator(System III)가 ‘언제 얼마나 깊게’ 숙고하고 ‘언제 학습/시뮬/업데이트’를 수행할지 내재적으로 결정하도록 설계해 self-regulation과 self-directed learning을 구현한다.

- **Empirical Impact**: 제안된 프레임워크는 무엇을 측정·감사(audit)·통제·안전 확보해야 하는지(agentive 성격이 강할수록 더 중요해지는 지점)를 더 명확히 해, 자율성이 커지는 시스템의 안전 논의를 구체화하려는 의미가 있다. 아키텍처 차원에서 agentic과 agentive의 차이를 5개 구조로 분해해 진단 가능성을 높이고, 향후 벤치마크/실험 설계에서 “워크플로 조립형”과 “내생적 agency형”을 구분해 비교할 근거를 제공한다. 즉, 단일 LLM 기능을 ‘에이전트화’하는 접근을 넘어, 열린 세계에서 지속 학습과 자기조절이 가능한 범용 agent model로 발전시키는 방향성을 제시한다.



### Decentralized Coordination of Autonomous Traffic Through Advanced Air Mobility Corridors (https://arxiv.org/abs/2606.23832)
Comments:
          Presented at the AIAA SciTech 2026 Forum

- **Prior Approaches**: 기존 AAM(Advanced Air Mobility) 코리더 연구는 주로 코리더 네트워크 설계, 코리더 내부 충돌 회피, 또는 중앙집중형 MPC 같은 전략에 집중해 왔다. 또한 코리더 기반 운항은 중앙 교통관리 부재 시 비효율적일 수 있다고 보는 관점이 널리 퍼져 있었고, 분산/자율 접근도 제약이 적은 4D 궤적 최적화에 의존하는 경우가 많았다. 최근 MARL 시도도 merge/intersection 조정이나 단순화된 가정(예: 코리더 내 평행 통과) 위주라 코리더 경계 준수와 분리거리 제약을 함께 다루기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 분산 환경에서도 자율 고정익 항공기가 local information만으로 코리더 흐름을 스스로 self-organize하도록 학습할 수 있음을 보인다. 특히 centralized training/decentralized execution 형태로 InforMARL 계열을 확장해, 단일 코리더(출구 이후 메터링), 연속 코리더, 코리더 분기(2갈래)까지 포함하는 대표 시나리오를 다룬다. 코리더 경계에 대한 순응성과 목표 도착(메터링)까지 동시에 달성하는 것이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 난제는 (1) Dec-POMDP 성격의 제한된 관측에서 다중 에이전트가 충돌 없이 안전 분리 기준을 만족하면서, (2) 코리더 위상(진입-중간-출구) 전환을 올바른 순서로 수행하고, (3) 경계 준수와 효율을 함께 최적화하는 것이다. 저자들은 GNN 기반 그래프 관측과 다단계 코리더 phase 정보(진입/코리더/진출)를 포함해 에이전트 정책이 필요한 지역 정보를 교환·통합하도록 설계했다. 또한 분리거리 위반 패널티, 코리더 경계 관련 유도(phase 전환 보상), 목표 도착 보상을 조합한 reward로 학습 안정성을 확보했다.

- **Empirical Impact**: 시뮬레이션 결과 3가지 코리더 구성 모두에서 항공기는 코리더 경계를 94% 이상 준수하며, 정해진 에피소드 시간 내에 목표에 도달한다. 분리 최소치 위반 시 필요한 tactical intervention은 저·중밀도에서는 8% 미만으로 드물지만, 고밀도·분기 코리더에서는 약 17%로 늘어난다. 교통량이 증가하고 시나리오가 복잡해질수록 성공률은 낮아지는 경향이 확인되며, 이는 코리더 네트워크 운영에서 분산 자율학습의 실효성과 한계(혼잡 시 개입 필요성)를 동시에 보여준다.



New uploads on arXiv(cs.MA)

### Generating Realistic Individual Activity Schedules via Activity Location Allocation Based on Simulated Travel Times (https://arxiv.org/abs/2606.24566)
Comments:
          8 pages, 5 figures. This is the author version of a short paper accepted for presentation in the poster session at the 17th Conference on Spatial Information Theory (COSIT 2026)

- **Prior Approaches**: 개인 단위 일일 활동 스케줄은 감염병 통제, 도시 교통 계획, 정책 설계 등에서 핵심이지만 프라이버시와 관측 한계로 실제 데이터 확보가 어렵다. 그래서 인구 데이터와 travel survey를 매칭해 스케줄을 합성하는 방식이 주로 쓰이며, 기존 방법은 보통 위치 쌍 간 정적 travel time에 의존한다. 정적 시간만 쓰면 교통 흐름의 불확실성이 반영되지 않아 활동 위치가 비현실적으로 바뀌고, 그 결과 시뮬레이션 travel time이 설문과 맞지 않는 문제가 생긴다.

- **Core Contribution**: 이 논문은 활동 위치를 단순 추정하는 대신, 동적 교통 시뮬레이션으로 얻은 travel time과 travel survey의 travel time이 일치하도록 활동 위치를 반복적으로 재할당하는 프레임워크를 제안한다. 활동 위치 할당에는 Viterbi와 유사한 dynamic programming 절차를 적용해 “가능한 경로 중 시뮬레이션 시간과 설문 시간을 가장 가깝게 만드는” 위치열을 찾는다. 또한 기존 activity modification(Ge0AI) 프레임워크와 결합해 활동 스케줄 수정까지 포함한 정합성 개선도 함께 탐색한다.

- **Technical Challenges**: 핵심 난제는 활동 위치가 교통 혼잡과 이동시간에 내생적으로 영향을 주기 때문에, 초기 위치 추정에서 생기는 불일치를 어떻게 줄이느냐에 있다. 논문은 이를 위해 (1) 시뮬레이션 기반 travel time이 불가능한 초기 단계에서는 도로 제한속도 기반 travel time을 쓰고, (2) 이후 traffic simulation과 activity location allocation을 교대로 반복(feed-back)하며 시뮬레이션 travel time을 최신 위치 추정에 반영한다. 계산 측면에서는 dynamic programming의 복잡도가 O(TV_n^2)로 커질 수 있어, 확장 시 beam search 같은 휴리스틱 적용 가능성도 제시한다.

- **Empirical Impact**: 더미 데이터와 애버딘(스코틀랜드) 도로 지도를 사용한 수치 실험에서, 첫 iteration 대비 iterative refinement 후 시뮬레이션 travel time과 설문 불일치(MAPE)가 52.2% 감소했다. 추가로 activity modification(GeoAI)까지 결합하면 MAPE가 5.5% 더 줄어 전체 개선이 유지됨을 보였다. 다만 활동 스케줄 수정은 일부 범주에서 활동 순서 패턴을 최대 16.2%까지 왜곡할 수 있으나, 그룹 내 주요(빈도 높은) 순서의 랭킹은 비교적 보존되고 시간대별 활동 분포 오차는 평균 2.0% 수준으로 제한되었다.



### Decentralized Coordination of Autonomous Traffic Through Advanced Air Mobility Corridors (https://arxiv.org/abs/2606.23832)
Comments:
          Presented at the AIAA SciTech 2026 Forum

- **Prior Approaches**: 기존 AAM(Advanced Air Mobility) 코리더 연구는 주로 코리더 네트워크 설계, 코리더 내부 충돌 회피, 또는 중앙집중형 MPC 같은 전략에 집중해 왔다. 또한 코리더 기반 운항은 중앙 교통관리 부재 시 비효율적일 수 있다고 보는 관점이 널리 퍼져 있었고, 분산/자율 접근도 제약이 적은 4D 궤적 최적화에 의존하는 경우가 많았다. 최근 MARL 시도도 merge/intersection 조정이나 단순화된 가정(예: 코리더 내 평행 통과) 위주라 코리더 경계 준수와 분리거리 제약을 함께 다루기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 분산 환경에서도 자율 고정익 항공기가 local information만으로 코리더 흐름을 스스로 self-organize하도록 학습할 수 있음을 보인다. 특히 centralized training/decentralized execution 형태로 InforMARL 계열을 확장해, 단일 코리더(출구 이후 메터링), 연속 코리더, 코리더 분기(2갈래)까지 포함하는 대표 시나리오를 다룬다. 코리더 경계에 대한 순응성과 목표 도착(메터링)까지 동시에 달성하는 것이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 난제는 (1) Dec-POMDP 성격의 제한된 관측에서 다중 에이전트가 충돌 없이 안전 분리 기준을 만족하면서, (2) 코리더 위상(진입-중간-출구) 전환을 올바른 순서로 수행하고, (3) 경계 준수와 효율을 함께 최적화하는 것이다. 저자들은 GNN 기반 그래프 관측과 다단계 코리더 phase 정보(진입/코리더/진출)를 포함해 에이전트 정책이 필요한 지역 정보를 교환·통합하도록 설계했다. 또한 분리거리 위반 패널티, 코리더 경계 관련 유도(phase 전환 보상), 목표 도착 보상을 조합한 reward로 학습 안정성을 확보했다.

- **Empirical Impact**: 시뮬레이션 결과 3가지 코리더 구성 모두에서 항공기는 코리더 경계를 94% 이상 준수하며, 정해진 에피소드 시간 내에 목표에 도달한다. 분리 최소치 위반 시 필요한 tactical intervention은 저·중밀도에서는 8% 미만으로 드물지만, 고밀도·분기 코리더에서는 약 17%로 늘어난다. 교통량이 증가하고 시나리오가 복잡해질수록 성공률은 낮아지는 경향이 확인되며, 이는 코리더 네트워크 운영에서 분산 자율학습의 실효성과 한계(혼잡 시 개입 필요성)를 동시에 보여준다.



### Emergent Relational Order in LLM Agent Societies: From Collective Affect to Authority Stratification (https://arxiv.org/abs/2606.23764)
Comments:
          Accepted to Findings of the Association for Computational Linguistics: ACL 2026. 37 pages

- **Prior Approaches**: 기존 사회과학의 Differential Order Pattern은 중국 농촌의 문화 특수성으로 해석되는 경우가 많았지만, 이를 작동시키는 ‘기계적 기제’가 충분히 실험적으로(계산적으로) 검증되진 못했다. LLM 기반 멀티에이전트 연구도 사회계약·규범·제도 형성 등은 다루되 대체로 단기 조정이나 이인(duadic) 수준에 집중해, 정서-윤리-권위-분업이 함께 진화하는 장기 사회구조 재현은 공백이었다.

- **Core Contribution**: 이 논문은 CAREB-MAS(COllective Affection–Reasoning–Emergence Based Multi-Agent Simulation)를 제안해, 문화 특수 규칙을 인코딩하지 않고도 Differential Order Pattern이 장기 시뮬레이션에서 ‘발현’될 수 있음을 보여준다. 에이전트는 Affect Control Theory·Social Identity Theory·뒤르껭적 집단 정서를 바탕으로 emotion–ethics–belief 체인을 순차 추론하며, 거시 환경은 개인 생산·선호 기반 할당·최소 상호작용 프로토콜만 제공한다. 그 결과 5가지 핵심 현상(안정적 분업, guanxi 기반 경제-윤리, 관계 거리별 협력 감쇠, 관계 기반 권위의 출현, 씨족 중심의 중심–주변부 층화)을 일관되게 재현한다.

- **Technical Challenges**: 핵심 기술 과제는 ‘장기 사회구조’를 단순한 보상 설계나 규칙 기반 위계로 만들지 않고, 에이전트 내부의 정서·윤리·관계 인지 동역학이 상호작용을 통해 구조로 응집되는 경로를 구현하는 것이다. 이를 위해 augmented BDI에 Affect/윤리 모듈(EC·ER)과 SIM(사회적 정체성) 업데이트를 결합했고, 동적 커뮤니티 탐지·이동, 공개적인 제안/발표(투표·강제 규칙 없음), 제안 채택 빈도로 ‘emergent 권위’를 사후 계측하는 방식으로 설계했다. 또한 두 가지 생산 구조(대칭/상보 능력)와 모듈 제거(ablation)를 통해 결과가 단순 LLM 아티팩트가 아니라 아키텍처·구조 조건 의존적임을 확인했다.

- **Empirical Impact**: 30라운드·18에이전트 장기 시뮬레이션에서 분업 고착, 관계 거리 기반 협력 그라데이션, 권위의 우측치우침(중심–주변부)을 포함한 Differential Order 현상이 자발적으로 나타났다. 특히 생산 구조가 달라지면 통합 양상이 변해, 대칭 능력에서는 친족 중심의 기계적 연대에 가까운 형태가, 상보 능력에서는 기능적 상호의존이 커지며 더 ‘유기적 연대’에 가까운 패턴이 관측됐다. 모듈 제거 실험과 여러 LLM 간 교차 강건성 결과는 CAREB-MAS의 발현 메커니즘이 특정 모델 학습 편향에만 기대지 않으며, Differential Order를 일반 기제의 구조민감적 emergent outcome로 해석하는 데 실증적 근거를 제공한다.



### Age of LLM: A Strategic 1v1 Benchmark for Reasoning, Diplomacy and Reliability of Large Language Models under Fog of War (https://arxiv.org/abs/2606.24391)
Comments:
          25 pages including appendices, 8 figures, 4 tables; appendices include verbatim system prompt and engine resolution pseudocode. All correlations reported with p-values, 95% bootstrap confidence intervals and Spearman's rho; includes a Steiger test and Bradley-Terry fit

- **Prior Approaches**: 기존 LLM 벤치마크(MATH, HumanEval, MMLU 등)는 대부분 단일 턴, 완전 관측, 정답이 명확한 과제에 초점이 있어 적대적 불확실성 하 계획(planning under uncertainty)이나 숨은 상대 추론, 여러 턴 동안의 structured output 신뢰성(잘못된 JSON/불법 행동 처리)을 충분히 검증하지 못한다. 또한 공개 벤치마크는 학습 데이터로의 오염 가능성이 커 점수가 실제 경쟁력보다 부풀려질 위험이 있다.

- **Core Contribution**: Age of LLM은 두 LLM이 fog of war, full diplomacy(메시지·정전·최후통첩)와 함께 13x7 그리드에서 1v1로 기지를 파괴하는 turn-based 1v1 벤치마크를 제안한다. 특히 엔진을 비공개로 두고 매 매치마다 무작위 맵 seed와 상대를 바꿔 데이터 오염 경로를 줄이며, 매 턴 출력은 strict JSON 스키마를 강제하고 불법 행동은 조용히 폐기되는 “신뢰성 차원”을 포함한다.

- **Technical Challenges**: 이 설정에서 핵심 기술적 난제는 부분관측·상대의 은닉·딜레마(핵 발사 vs 지상 공략) 속에서도 JSON 스키마를 지키며, 불법 행동이 누적되지 않도록 믿음(belief)과 상태 추적을 해야 한다는 점이다. 논문은 (1) near rule-only 프롬프트로 build-order 같은 직접 조언을 배제하고 (2) 에이전트가 매 턴 관측과 직전 턴 결과, 외교 기록만 이용하도록 하며 (3) 매치별 재현성을 위해 엔진-뷰어 간 replay 포맷과 replays를 제공해 분석 가능성을 확보한다. 다만 데이터 수집 당시 포함됐던 두 개의 borderline 전술 시드 문구가 정찰/탱크 중심 경향을 증폭했을 수 있어, 그 일부 결론은 제한점으로 표시한다.

- **Empirical Impact**: 54개 매치(15개 reasoning model, 5,258개 액션) 분석에서 결과는 핵(핵 선제 발사 루트)이 압도적으로 많았고(룰-일치 하위코퍼스 78%, 전체 85%), 군사 정복은 드물지만 성사되면 더 빠르게 끝나는 패턴(평균 12.3턴 vs 18.9턴)이 관측됐다. 외교는 활발히 오가지만 실제 합의로까지 이어지는 경우는 거의 없었고, 불법 행동의 약 58%가 fog/state 추적 오류로 나타나 illegal-action rate가 믿음 추적의 프록시로 기능함을 시사한다. 또한(탐색적 1개 모델 제외) reliability가 승리와 강하게 연결되지는 않았으며, 전체 순위는 소규모·불균형·사이드 미스왑의 영향으로 “예비적 기술 통계” 성격임을 강조한다.



### Agon: An Autonomous Large-Scale Omnidisciplinary Research System Built on Prompt Economy (https://arxiv.org/abs/2606.24177)
- **Prior Approaches**: 기존 자율연구·멀티에이전트 연구는 아이디어 생성부터 실험 실행, 원고 작성까지를 한 번에 처리하는 방식이 많아 ‘산출물 생산’에 공수가 쏠리는 문제가 있었다. 또한 자동화된 판단을 위해 프롬프트/코드/스키마를 단계마다 늘리는 경향이 있어 유지보수 비용과 도메인 전이 장벽이 커졌다. 특히 LLM 기반 리뷰는 점수는 잘 오르지만 인간이 받아들이는 과학적 서술·주장 정합성을 벗어나는 실패가 발생했다.

- **Core Contribution**: Agon은 연구 워크플로 안에서 기계가 ‘검증 가능한 것’은 자동으로 확인하고, 그 외의 판단은 사람 과학자가 유지하도록 연구 오케스트레이터를 설계했다. 아이디어-제안-실험-논문을 아티팩트(artifact) 경계로 쪼개고, producer–critic 루프를 반복 재사용하는 ‘연구 팩토리’ 구조가 핵심이다. 이를 통해 새로운 도메인으로 옮길 때도 핵심 아키텍처는 유지하고 입력 컨텍스트만 바꾸는 전이를 지향한다.

- **Technical Challenges**: 대규모 병렬 루프(Massive Parallelism)를 돌리면 어떤 스레드를 어느 순서로 전진시킬지 결정하는 디스패치 부담이 커진다. Agon은 Zero-Code Orchestration으로, 상태를 사람이 읽을 수 있는 컨텍스트로 해석한 뒤 ‘프롬프트 기반 디스패치’로 다음 핸드오프를 선택해 코드·파서·유한상태기계가 비대해지는 문제를 줄였다. 또 논문 개선에서는 LLM 리뷰 점수만 최적화해 인간 친화성을 잃는 ‘종이 괴물’ 실패를 막기 위해, 감사(auditor)로 인간이 읽는 과학적 글쓰기 제약으로 투영(projection)하며 killer reviewer와 area-chair adjudicator로 공격 논리를 통과시킨다.

- **Empirical Impact**: Agon은 10개 이상의 과학 도메인에서 444회 Prompt Economy 루프를 포함해 수천 번의 scientist-coder-auditor 반복을 수행했으며, 사람의 실험 코드 작성 없이도 운영 가능함을 보였다. 동시에 자동화 루프가 포착·수정 가능한 실패와 인간 판단이 필요한 실패를 경계짓고, 이를 severity·fixability·visibility·capability locus 기준으로 분류해 새로운 실패 유형을 드러냈다. 결과적으로 ‘machine scales, human steers’ 패러다임을 실제 배치로 뒷받침하며, 자율연구 인프라의 산업화와 공개(open-source) 배포 가능성을 강화한다.



### EMAgnet: Parameter-Space EMA Regularization for Policy Gradient Self-Play in Large Games (https://arxiv.org/abs/2606.23995)
Comments:
          Accepted at NExT-Game 2026: New Frontiers in Game-Theoretic Learning (ICML 2026 Workshop). 13 pages, 2 figures,

- **Prior Approaches**: 자기대국(self-play)에서 PPO 같은 정책 그래디언트에 정규화를 더하면, 2인 영(0-sum) 불완전정보 게임에서 게임이론 기반 방법과 견줄 수 있다는 결과가 축적돼 왔다. 특히 uniform distribution을 entropy bonus로 쓰는 “uniform-magnet” 정규화가 강력한 기준선으로 자리 잡았지만, 모든 행동을 똑같이 향해 규제해 전략적으로 쓸모없는(특히 strictly dominated) 행동에도 정규화 예산이 낭비된다.

- **Core Contribution**: 이 논문은 정규화 타깃을 고정 uniform에서 벗어나, 에이전트가 학습하면서 변하는 “adaptive regularization target”으로 설정한다. 제안한 EMAgnet은 정책 네트워크 파라미터에 대한 exponential moving average(EMA)를 정규화 자석(magnet)으로 써서, dominated 전략을 회피하는 방향으로 타깃 자체가 자연스럽게 멈추게 한다. 결과적으로 “나쁜 전략은 잊고, 좋은 전략은 기억”하는 성질을 고정 타깃 대비 구현한다.

- **Technical Challenges**: 딥 강화학습에서는 상태별(policy space) 이동 자석을 그대로 유지하기 어렵기 때문에, 이를 파라미터 공간(parameter-space) EMA로 옮기는 것이 핵심 난관이었다. 논문은 PPO에 KL 정규화 항을 추가하되, uniform처럼 고정된 타깃이 아니라 PPO 업데이트마다 정책 파라미터의 EMA를 magnet 파라미터로 갱신해 연속적으로 적응하도록 설계했다. 이때 학습 루프는 크게 바꾸지 않고 “추가 EMA 업데이트 1회” 정도의 최소 복잡도로 구현된다.

- **Empirical Impact**: 표준 벤치마크에서는 PPO-EMAg(특히 EMA magnet 정책)이 PPO-Uniform(선형/파워법 스케줄)과 비슷하거나 더 낮은 exploitability를 달성하며, 대부분의 환경에서 우위가 관찰됐다. 더 나아가 strictly dominated 전략이 대량 포함된 수정 벤치마크(FF, Control 변형)에서는 일관된 성능 개선이 나타났고, 수렴 속도에서도 uniform 방식 대비 빠르게 낮은 exploitability에 도달했다. 즉, 전략적으로 쓸모없는 행동에 정규화 예산을 낭비하는 문제를 줄이면서도 학습 초중반의 혼합(mixing) 압력을 유지한다는 점에서 실전적 의미가 크다.



### Critique of Agent Mod (https://arxiv.org/abs/2606.23991)
- **Prior Approaches**: 최근 LLM을 ‘coding agents’, ‘AI co-scientists’처럼 에이전트라 부르며 생산성을 높이려는 시도가 늘었지만, 많은 시스템은 미리 짜둔 툴·워크플로·프로그램 제어 루프를 외부에서 오케스트레이션하는 agentic 성격이 강하다. 그 결과 목표나 정체성, 의사결정의 모드 전환, 학습의 시작/중단 같은 핵심 구조가 모델 내부에서 내재화(endogenously)되기보다 파이프라인 설계에 의존하는 경향이 있다. 논문은 이러한 방식이 다양한 환경 적응과 진짜 자율성(agency)의 일부만 다룰 뿐, 생물학적 에이전트처럼 열린 세계에서 스스로 조직되는 역량을 설명하기 어렵다고 지적한다.

- **Core Contribution**: 논문은 ‘에이전트(agency)’를 단순 작업 수행 능력(operation excellence)과 분리해, 목표 지향 행동, 정체성의 진화, 자기조절, 자기성찰 및 학습이 내부에서 발생해야 한다고 주장한다. 이를 위해 agentic 시스템(외부 공학으로 워크플로를 조립)과 agentive 시스템(모델 내부에서 구조가 내생적으로 발생)을 경계짓고, 에이전트 아키텍처를 goal, identity, decision-making, self-regulation, learning의 5차원으로 분석한다. 나아가 범용 에이전트 모델을 위한 Goal-Identity-Configurator(GIC) 아키텍처를 제안하며, 계층적 목표 분해·진화하는 identity·월드 모델 기반 simulative reasoning·학습/추론 깊이를 조절하는 configurator·실제/시뮬 경험에서의 self-directed learning을 한 틀로 결합한다.

- **Technical Challenges**: 기여의 핵심은 ‘월드 모델과 에이전트 모델을 기능적으로 분리’하는 데 있으며, 둘을 합치면 next-state 예측과 reward/행동 선택이 뒤섞여 계획과 시뮬레이션의 신뢰성이 흔들릴 수 있다고 본다. 또한 장기 목표와 identity를 내부 잠재변수로 두고, 그에 조건화된 계획·실행·메타 의사결정이 어떻게 학습 가능한 형태로 구성될지 해결해야 했다. 논문은 simulative reasoning(System II)으로 world model에서 미래를 예측·계획하고, actor(System I)로 즉각 실행을 담당하며, configurator(System III)가 ‘언제 얼마나 깊게’ 숙고하고 ‘언제 학습/시뮬/업데이트’를 수행할지 내재적으로 결정하도록 설계해 self-regulation과 self-directed learning을 구현한다.

- **Empirical Impact**: 제안된 프레임워크는 무엇을 측정·감사(audit)·통제·안전 확보해야 하는지(agentive 성격이 강할수록 더 중요해지는 지점)를 더 명확히 해, 자율성이 커지는 시스템의 안전 논의를 구체화하려는 의미가 있다. 아키텍처 차원에서 agentic과 agentive의 차이를 5개 구조로 분해해 진단 가능성을 높이고, 향후 벤치마크/실험 설계에서 “워크플로 조립형”과 “내생적 agency형”을 구분해 비교할 근거를 제공한다. 즉, 단일 LLM 기능을 ‘에이전트화’하는 접근을 넘어, 열린 세계에서 지속 학습과 자기조절이 가능한 범용 agent model로 발전시키는 방향성을 제시한다.



### Maestro Order: A Model-Agnostic Orchestration Harness (https://arxiv.org/abs/2606.23983)
Comments:
          10 pages, 4 figures

- **Prior Approaches**: 기존 접근은 대부분 생성에 의존하거나(예: 한 번에 답하기, chain-of-thought), 여러 샘플을 모아 최빈/투표로 정답을 고르는 방식(self-consistency, voting), 또는 생성-검증-수정 루프(ReAct, Reflexion, Self-Refine 등)처럼 부분적으로 verification을 붙이는 데 그쳤습니다. 이때 중요한 비용-신뢰도(얼마나 비싸고 얼마나 안전한가)와 검증기의 품질(거짓수용률/완전성)이 파이프라인 설계에 명시적으로 반영되지 않아, instance별로 최적 조합을 찾기 어렵다는 한계가 있습니다. 또한 decomposition과 같은 다단계 구성은 작은 오류가 누적되기 쉬워, 단독 기법으로는 “신뢰 가능한 조직”을 만들기 힘듭니다.

- **Core Contribution**: 이 논문은 어떤 모델이든 black-box base solver로 감싸서, 불확실한 솔버를 신뢰 가능한 문제해결 시스템으로 바꾸는 model-agnostic orchestration harness인 Maestro Order를 제안합니다. 핵심은 decompose, ensemble, verify, recurse의 네 가지 구조 프리미티브를 일관된 스키마로 조합하고, compute 예산을 어디에 쓰는지 결정하는 budget-aware controller를 붙여 “정확도 향상”을 비용과 함께 제어하는 것입니다. 특히 verification gating의 신뢰도 증폭을 검증기 discrimination(Λ=β/α)으로 측정·활용해, instance마다 적절한 메커니즘을 선택하도록 설계했습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 검증기가 생성 방식을 “악용”해 거짓수용률이 올라가는 verifier gaming, (2) 솔버/검증기의 오류가 상관되어 ensemble 효과가 줄어드는 correlated errors, (3) decomposition 경계에서 오류가 누적되는 compounding입니다. 논문은 결정론적(trace 기반 재현), fail-closed(툴/타임아웃/오류는 reject), 점수 캘리브레이션(의사결정에 raw score를 직접 쓰지 않음), 그리고 각 검증기의 β(완전성)와 α(거짓수용률)를 온라인으로 추정해 Λ를 업데이트하는 장치로 이를 제어합니다. 또한 planner는 타입 호환성과 “검증 가능한 분해”일 때만 split을 허용하고, controller는 각 단계의 marginal reliability gain per unit cost를 기준으로 검증과 투표를 단계별로 배치합니다.

- **Empirical Impact**: 평가에서는 reliability at fixed cost, coverage/ risk–coverage, calibration(ECE), 비용·지연을 축으로 측정하고, 투표(voting)와 검증(gating)의 상대적 이득 및 분해/재귀/다양성의 기여를 ablation으로 분리합니다. 특히 parameterized solver/verifier에 대한 faithful Monte Carlo simulation으로 odds law를 정량적으로 재현했는데, 예를 들어 Λ가 있는 verification gate는 기하급수적으로 신뢰도를 끌어올려 0.55→0.98(게이트 2개) 및 0.999(게이트 4개) 같은 결과를 보였습니다. 동일 목표 신뢰도에 대해 budget-aware controller가 voting 단독 대비 훨씬 적은 비용으로 목표를 달성하며, 이는 “신뢰도-비용”을 명시적으로 최적화하는 오케스트레이션의 실용적 가치를 시사합니다.



### Welfarist Control Design -- How to fulfill the societal mandate in multi-agent control? (https://arxiv.org/abs/2606.23931)
- **Prior Approaches**: 기존 사회선택·복지경제 연구는 다수 에이전트의 선호를 집합적으로 합치는 문제를 다뤄왔지만, 통상적으로 정보 제약(대인 복지 비교 가능 여부)에 따라 허용 가능한 사회후생 규칙이 크게 달라진다. 또한 고전적 경우에는 Arrow의 불가능성처럼 직관적 공리(예: 독립성, 만장일치, 비독재)를 동시에 만족하기 어려운 한계가 강조돼 왔다.

- **Core Contribution**: 이 논문(튜토리얼 파트)은 welfarism 관점에서, 각 에이전트의 개별 비용(또는 효용) 평가를 먼저 정의하고 그 평가로부터 ‘사회적 순위(선호관계)’를 구성하는 틀을 엄밀한 수학적 객체(Social Cost Functional, SCFL)로 정식화한다. 나아가 세 가지 공리(Pareto, IIA, Pairwise Continuity)가 성립하면 사회적 순위를 단일 Social Cost Function(SCF)으로 수치화해 최적화 문제로 환원할 수 있음을 제시한다.

- **Technical Challenges**: 핵심 난제는 ‘어떤 SCF를 쓸 것인가’가 단순히 공리만으로 결정되지 않고, 에이전트 간 비용을 비교할 수 있는 정도(인간 간 비교가능성)라는 정보 가정에 의해 달라진다는 점이다. 논문은 비용 차이·수준의 비교가능성을 CNC/CUC/CFC/OLC 등으로 분류해, 동일한 물리적 배분 문제라도 서로 다른 공정성·효율성 기준의 배분 규칙(예: Rawls maximin, Nash bargaining, utilitarian sum, max-min형 혼합)을 선택하도록 절차를 제공한다.

- **Empirical Impact**: 제시된 예시(전력 감축 curtailment)에서는 CUC/CNC/OLC에 따라 ‘최소 총 감축’, ‘비례 감축’, ‘최악자 보호/동일 감축’ 등 서로 다른 배분 결과가 설명 가능하게 도출된다. 즉, 사회적 목적이 통제기 설계로 연결될 때, 비교가능성 가정과 공리를 명시함으로써 설계가 임의적 관습이 아니라 책임 있는 규범적 근거 위에서 재구성된다는 점에서 의미가 크다. 이어지는 동적 welfarist control에서는 폐루프 동작이 에이전트 선호와 정렬되도록 보장하는 통제 공학적 과제를 강조한다.



### From Task-Guided Conversational Graphs to Goal-Oriented Dialogue Runtimes (https://arxiv.org/abs/2606.23797)
Comments:
          21 pages, 7 figure, 10 tables

- **Prior Approaches**: 기존 LLM 오케스트레이션은 “다음에 어떤 에이전트/툴/노드를 실행할지”에 강하지만, 여러 목표가 동시에 살아 있는 대화에서 목표의 연속성(objective continuity)을 자동으로 보장하긴 어렵다. 프로세스 기반(워크플로 그래프, FSM)은 실행 위치의 연속성은 잘 유지하지만, 한 목표가 멈춘 채 다른 목표가 개입·무효화되는 상황까지 안정적으로 복원하기엔 한계가 있다. 또한 에이전트 라우팅이나 채팅 히스토리/실행 그래프 위치만으로는 “어떤 사용자 목표가 살아남는지”를 의미론적으로 구분하기 어렵다.

- **Core Contribution**: 이 논문은 Goal-Oriented Dialogue Runtime(GODR)을 제안하며, 프레임워크 중립적인 계층에서 목표(goals)와 생명주기(lifecycle), 태스크 프레임(task frames), 무효화 규칙(invalidation rules), 재개 계약(resumption contracts)을 런타임의 1급 객체로 취급한다. GODR은 그래프 런타임/에이전트/툴이 담당하는 “bounded execution”을 대체하지 않고, 대화 중단·지연·수정·무효화에도 목표가 끊기지 않게 만드는 “objective continuity” 전용 런타임 경계(boundary)를 명확히 한다. 특히 GC-4(목표 복잡도) 영역에서 필요하다고 주장하며, 단순 스택/트리 모델만으로는 공유 제약·의존·무효화가 얽힌 대화를 다루기 어렵다고 정리한다.

- **Technical Challenges**: 핵심 기술 과제는 “새 사용자 발화가 들어왔을 때 어떤 목표 수준 연산(continue/revise/push/switch/pop/resume/cancel/escalate/reset)을 수행할지”를, 실행 라우팅과 분리해 결정하는 것이다. 논문은 목표 구조를 라벨된 directed graph(DAG 포함)로 모델링하고, 재개 가능 상태는 resumption contract가 비어 있지 않다는 등 런타임 불변식(invariants)과 함께 무효화 안전성(invalidation safety)을 체크하도록 설계한다. 또한 Goal Policy가 LLM 라우터처럼 무제한으로 동작하지 않도록, typed state와 가드·심볼릭 제약으로 허용 가능한 연산 집합을 마스킹/필터링하는 하이브리드 신경-상징 접근을 제안한다.

- **Empirical Impact**: 이 논문은 실험 성능을 “검증된 수치”로 주장하기보다, Multi-Objective Interruptible Dialogue Problem을 정식화하고 평가를 위한 아젠다와 기준, 베이스라인 선택 방법론을 제시하는 시스템 논문 성격을 갖는다. 따라서 즉시 실측 개선을 보장하진 않지만, 기존 오케스트레이션 프레임워크가 강한 실행 계층과 달리 목표 연속성 계층이 애플리케이션 아키텍처에서 누락되기 쉽다는 문제의식을 명확히 한다. 목표 수명주기·무효화·재개 계약을 런타임으로 외재화한다는 관점은 향후 복잡한 멀티도메인 대화 벤치마크/평가 설계로 이어질 수 있는 의미가 있다.



### Engineering Reliable Autonomous Systems: Challenges and Solutions (https://arxiv.org/abs/2606.23760)
- **Prior Approaches**: 기존 연구는 주로 전통적인 사이버-물리 시스템의 검증 관행을 자율 시스템에 그대로 이식하는 경향이 있었고, 복잡한 환경에서의 불확실성과 블랙박스 구성요소까지 충분히 다루지 못한다는 한계가 지적됐다. 또한 symbolic 방식은 상대적으로 검증이 쉬운 반면 sub-symbolic 방식은 확장성은 높지만 신뢰성 보장이 어렵고, neuro-symbolic로 간극을 메우려 해도 여전히 신뢰 문제는 미해결로 남아 있었다. 다중 로봇·재구성 시스템에서는 내부/외부 도전이 동시에 커지며, “검증 기법은 있으나 현장 적용이 낮은” 간극도 반복적으로 언급됐다.

- **Core Contribution**: ERAS(Engineering Reliable Autonomous Systems) 워크숍 보고서는 자율 시스템을 위한 verification & validation, 실세계 엔지니어링, 안전한 소프트웨어 아키텍처를 축으로 “도전 과제 카탈로그”와 “해결을 잇는 로드맵”을 제시한다. 특히 학계에선 알려져 있지만 실제 산업 적용으로 이어지지 못한 과제들과, 여전히 해결되지 않은 과제를 분리해 다음 연구·협업 경로를 구체화했다. 또한 FMAS(형식 방법)와 AREA(인지로보틱스/멀티에이전트)의 커뮤니티를 결합해, 서로가 갖는 사용사례·기법 이해의 공백을 메우려는 방향성을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 예측 불가능한 물리 환경과 black-box 부품을 모델링하는 문제, (2) AI 개발 방식(특히 sub-symbolic/학습 기반)으로 인해 생기는 신뢰성 격차, (3) 사람-로봇 상호작용·안전 요구를 보장할 증거를 설계하는 문제로 정리된다. 보고서는 이를 위해 정적 검증과 runtime verification을 함께 쓰는 접근, 아키텍처 수준 분해와 안전 아티팩트(예: GSN) 활용, 시뮬레이션·실험을 결합한 V&V 파이프라인(서로의 결과로 모델/요구사항을 보정)을 강조한다. 더불어 안전한 자율을 위해 안전 사례(safety case)를 구성하고, 테스트베드의 현실성·격리된 안전 모듈 등 실물 제약을 반영하는 방법을 연결해 제시한다.

- **Empirical Impact**: 워크숍은 11개 케이스 스터디를 통해 우주 능동 파지, 지하 광산 UAV, 의료 triage 의사결정지원, 산업 협업 로봇 안전거리 검증, 수중 케이블 추적, 자율 주행의 추가적 책임/설명 요구, 가정용 Care-O-bot 의사결정 검증, 인간 인수인계 협업(테이블 레그 handover), 건물 화재 분사, 원전 시설 루틴 점검, 전장 내(반)자율 협동 시나리오 등 다양한 도메인을 아우른다. 특히 runtime monitor로 주입 결함을 탐지하거나(우주 파지), GPS 부재·부분 관측 환경에서 fail-safe 자율의 수용성을 시험하며(광산 UAV/수중 AUV), 학습 구성요소와 닫힌 고리(closed-loop) 시스템을 함께 검증하는 사례가 제시됐다. 결과적으로 연구자와 산업이 “무엇이 아직 증거로 굳어지지 않았는지”를 공통 언어로 파악하고, 후속 연구·저널 스페셜 테마·후속 이벤트로 협업 모멘텀을 만들 수 있다는 점에서 의미가 크다.




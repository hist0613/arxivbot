New uploads on arXiv(cs.CL)

### StylisticBias: A Few Human Visual Cues Drive Most Social Biases in MLLMs (https://arxiv.org/abs/2606.20527)
Comments:
          Accepted to the non-archival workshops AI4Good and Culture x AI at ICML 2026

- **Prior Approaches**: 기존 연구는 다양한 개인/인구집단을 비교해 편향을 측정해 왔지만, 이 방식은 ‘외모 속성’과 ‘정체성(개인 고유 차이)’을 분리하기 어렵다는 한계가 있습니다. 또한 시각적 편향은 주로 매력도 같은 집계 신호로 다뤄져, 어떤 구체적 단서가 판단을 바꾸는지 고해상도로 분해하기가 어려웠습니다. 결과적으로 MLLM이 사람을 평가할 때 어떤 비주얼 cue에 민감한지 정밀 감사가 부족했습니다.

- **Core Contribution**: 이 논문은 MLLM의 사회적 판단에서 ‘속성 수준 편향’을 통제적으로 평가하기 위한 벤치마크 StylisticBias를 제안합니다. 핵심은 500개의 기준(base) 얼굴에서 정체성은 고정한 채 단일 시각 속성만 편집해 약 25K 이미지를 만들고, 6개 MLLM을 25개의 이분(positive/negative) 사회 판단 시나리오에 대해 비교한다는 점입니다. 이를 통해 모델이 특정 외모 속성의 변화에 얼마나 일관되게 반응하는지 직접 측정할 수 있게 합니다.

- **Technical Challenges**: 가장 큰 기술적 도전은 ‘정체성 고정’ 하에서 속성만 바꾼 이미지들이 실제로 단일 속성 변화 외의 잡음(불의도 아티팩트, 시맨틱 불일치)을 최소화하도록 설계·검증하는 것입니다. 연구진은 Imagen 4로 기준 얼굴을 만들고 Nano Banana로 속성별 단일 편집을 수행하되, 프레이밍·조명·배경을 최대한 통제하고 인간 검증(90% 수동 리뷰, 98% 통과)을 거쳐 품질을 보장했습니다. 또한 프롬프트 순서와 random seed에 대한 민감도를 줄이기 위해 다양한 주문/시드에서 강제 선택을 반복하고 파싱 가능한 응답만 집계해 선호도 점수 변화를 계산합니다.

- **Empirical Impact**: 실험 결과, 외모 판단 편향은 소수의 자기표현(appearance-related) 단서에 집중되며, 특히 패션 스타일 같은 의도된 선택 신호가 가장 큰 속성 수준 변화(예: 패션·헤어·메이크업/립·안경)를 유발했습니다. 반면 age와 body type은 가장 큰 인구통계(정체성) 수준의 효과를 보이되, 단일 속성 편집에서는 약 15개 속성이 전체 변화의 약 80%를 설명해 편향이 ‘소수 단서’에 응축됨을 보여줍니다. 의미 정렬(semantic alignment) 편향도 확인되어, 사회경제/스타일 관련 판단처럼 외모 의미와 직접 맞닿은 질문에서 민감도가 더 강하게 나타났고, 모델 간에는 ‘중요 단서의 패턴’이 비교적 일관되지만 ‘반응 강도’는 더 큰 모델에서 완화되는 경향이 관찰됩니다.



### Beyond Global Replanning: Hierarchical Recovery for Cross-Device Agent Systems (https://arxiv.org/abs/2606.20487)
- **Prior Approaches**: 기존 multi-device 에이전트는 작업 분해와 디바이스 간 할당, 실행 중 계획 업데이트를 제공하지만, 실패 복구가 대체로 coarse-grained에 머물렀습니다. 예외가 나면 같은 전략을 재시도하거나 서브태스크를 다시 할당하거나 전체 계획을 바꾸는 방식이 많아, “현재 디바이스에서 가능한 복구 범위”를 체계적으로 구분하지 못했습니다.

- **Core Contribution**: 이 논문은 H-RePlan으로, 디바이스 로컬 복구(전략 선택/전략 수준 회복)와 오케스트레이터 글로벌 복구(교차 디바이스 재계획)를 계층적으로 분리합니다. 또한 디바이스마다 지원 가능한 API-CLI-GUI 전략 공간을 platform-independent하게 표현하는 통합 전략 제어 추상화를 제안합니다.

- **Technical Challenges**: 핵심 난제는 실패가 strategy/서비스/디바이스 중 어디에 속하는지 모호한 상태에서, 오케스트레이터가 과도한 정보 없이도 “스코프에 맞는” 복구 결정을 하게 만드는 것입니다. 이를 위해 cross-layer failure abstraction인 CLFE(Cross-Layer Failure Event)를 도입해, 어떤 하위태스크가 어떤 디바이스에서 어떤 범주의 실패를 겪었고 로컬 복구가 왜 중단됐는지 요약 증거만 전달하도록 설계했습니다.

- **Empirical Impact**: HeraBench(리눅스+안드로이드 4대, 전략/디바이스 수준 fault 주입)에서 H-RePlan은 completion 75.84%, instruction adherence 77.72%, perfect-pass 36.78%로 single-strategy 및 coarse-grained multi-device baseline을 크게 앞섰습니다. 특히 end-to-end 성공의 기대 토큰 비용이 UFO3-GUI 대비 5.44× 낮아졌고, perfect-pass가 높은 수준에서 fault가 있는 에피소드 전반으로 확장되어 ‘스코프 인지형 계층 복구’의 필요성을 실험적으로 뒷받침합니다.



### Your Mouse and Eyes Secretly Leak Your Preference: LLM Alignment using Implicit Feedback from Users (https://arxiv.org/abs/2606.20482)
- **Prior Approaches**: 기존 LLM 정렬(alignment)은 대부분 텍스트 응답에 대한 명시적 인간 피드백을 모으고, 그 선호를 예측하는 reward model을 학습하는 방식이 중심이었다. 하지만 사용자는 실제로 만족/불만 같은 명시 피드백을 1~3% 수준으로만 제공하고, 잦은 피드백 요청은 사용자 만족을 떨어뜨릴 수 있다. 또한 기존 정렬 연구는 click 같은 암묵적 신호를 활용한 경험이 있는 검색/추천과 달리, 사용자 행동 기반 암묵적 피드백을 LLM 정렬에 체계적으로 끌어오지 못했다.

- **Core Contribution**: 이 논문은 웹캠 기반 눈 응시(eye-gazing)와 마우스 이동(mouse movement)을 선호 라벨과 함께 수집한 IFLLM(Implicit Feedback for Large Language Models) 데이터셋을 제시한다. Mechanical Turk 근로자 59명(총 1336 멀티턴)이 LLM 답변을 보며 웹캠으로 눈 응시를 남기고, 마우스 궤적도 함께 기록되며, 동시에 답변 쌍/단일에 대한 선호를 얻는다. 이를 통해 암묵적 피드백이 실제 배치 환경에서 정렬 성능에 주는 가치를 정량화한다.

- **Technical Challenges**: 핵심 기술 난점은 ‘명시적 선호 라벨’이 없는 암묵적 행동 신호를 reward model의 입력으로 신뢰성 있게 변환하는 것이다. 저자들은 시간축 정규화로 다양한 세션을 비교하고, 마우스·눈 응시 궤적에서 읽기 시간/위치 등 특징을 추출해 random forest 기반 선호 예측을 구성했으며, 불필요하거나 노이즈가 되는 입력(예: 응답 생성기 식별자 등)은 제외했다. 특히 응답 길이에 따라 마우스-눈 응시 상관이 달라짐을 이용해, DPO 학습 시 reward model의 암묵적 신호 반영이 긴 답변에서 더 크게 작동하도록 설계했다.

- **Empirical Impact**: 실험 결과, 암묵적 피드백 기반 reward model이 텍스트만 쓰는 reward model의 정확도를 55%에서 64%로 끌어올렸다. 또한 8개 LLM에 대해 DPO 적용 시 상대적인 응답 품질 개선이 거의 3배 가까이 커져, ‘야외(wild) 환경’에서도 암묵적 피드백의 효용이 성립함을 보여준다. 분석적으로는 특히 긴 응답에서 스크롤 필요가 반영된 마우스 궤적이 강한 선호 신호가 되어, 분야의 데이터 플라이휠(data flywheel) 아이디어까지 확장될 여지를 남긴다.



### CATCH-ME if you RAG: a dataset of Contextually Annotated multi-Turn Counterspeech against Hate and Misinformation Exchanges (https://arxiv.org/abs/2606.20369)
- **Prior Approaches**: 온라인 혐오표현과 허위정보는 현실에선 함께 나타나지만, 기존 NLP 연구는 주로 각각 따로 다뤄 왔습니다. counterspeech(CS) 생성에서도 zero-shot LLM은 반복적이고 모호한 답을 내는 경향이 있어, 고품질 예제가 필요하다는 문제의식이 커졌습니다. 하지만 혐오+허위정보 교차를 다루는 다중 턴·다국어 대화 데이터는 희소했고, 있더라도 단일 턴 영어에 한정됐습니다.

- **Core Contribution**: 이 논문은 혐오와 허위정보의 교차를 다루는 최초의 대규모 전문가 큐레이션, 다국어(5개 언어) 대화 데이터셋 CATCH-ME를 제안합니다. 7개 소수집단을 향한 발화에 대해, 혐오+허위정보를 퍼뜨리는 인물과 이를 사실에 근거해 바로잡는 counterspeaker 간의 다중 턴 대화를 제공합니다. 또한 fact-checking 기사와 NGO 보고서처럼 검증된 외부 지식에 대화를 고정하고, 문서/청크 단위 span annotation을 포함해 RAG(검색증강생성) 적용성을 높였습니다.

- **Technical Challenges**: 핵심 기술적 도전은 (1) 대화 맥락을 유지하면서 (2) 검증된 사실로 근거를 제공하고 (3) 모호하거나 반복적인 generation을 줄이는 동시에 다국어로 확장하는 것입니다. 연구진은 23명의 CS 전문가( fact-checkers+NGO operators )와 4가지 human-machine collaboration 전략을 조합해, GPT-기반 초안 생성 후 전문가가 retrieved 근거 span을 검토·수정하도록 설계했습니다. 또한 자동 번역 구간도 포함하되, 언어별 후편집 난이도 차이를 고려해 별도로 평가하며 데이터 품질 특성을 분석했습니다.

- **Empirical Impact**: 실험은 retrieval과 generation을 분리해 ‘사실 기반 CS’에서의 RAG 동작을 점검합니다. retrieval에선 dense retriever가 BM25 같은 sparse baseline을 전반적으로 앞섰고, 특히 전체 대화 맥락을 반영한 질의(QD_CQ)가 성능을 끌어올렸습니다. generation에선 gold 지식 또는 검색된 청크를 주는 CSretr/CSgold 구성이 무근거 CSbase보다 의미 유사도·사실 정합성(예: NLI/faithfulness)이 크게 개선되었으며, retriever 정확도에 따라 결과가 민감하게 달라지는 trade-off도 확인했습니다. 결과적으로 교차 혐오·허위정보 영역에서 다국어 RAG 기반 CS 평가/학습의 벤치마크 역할을 할 만한 실증적 기반을 제공합니다.



### PsyScore: A Psychometrically-Aware Framework for Trait-Adaptive Essay Scoring and ZPD-Scaffolded Feedback (https://arxiv.org/abs/2606.20287)
- **Prior Approaches**: 기존 AES는 정확한 점수 예측에 집중해 왔고, 점수와 피드백을 분리해 해석 가능성과 진단 효용이 약해지는 한계가 있었다. 신경망 기반 점수 모델은 교육 측정 관점에서 불투명(opaque)해 psychometric validity와 공정성 논란이 생길 수 있고, LLM 피드백은 학습자 숙련도나 ZPD를 반영하지 못해 인지적 불일치가 나타나기 쉽다. 또한 멀티태스크로 여러 trait를 예측하더라도, 공유 잠재능력(latent ability)로 묶어 교육적 의사결정으로 연결하는 방식은 부족했다.

- **Core Contribution**: PsyScore는 점수 산출과 수업적 피드백을 하나의 공유 잠재 능력 표현으로 통합하는 psychometrically-aware 프레임워크를 제안한다. Trait-Adaptive Neural IRT Scorer가 Graded Partial Credit Model(GPCM)을 신경망에 접목해 해석 가능한 학생 능력 추정치를 만들고, 이를 ZPD-Scaffolded Feedback Generator의 조건 신호로 사용해 숙련도에 맞춘 instructional scaffolding을 생성한다. 결과적으로 AES가 단순 summative score를 넘어 formative diagnosis-피드백 루프로 전환되는 것을 목표로 한다.

- **Technical Challenges**: 핵심 과제는 (1) 신경 IRT(GPCM)에서 calibration 불안정성과 최적화 비볼록성을 다루면서도 (2) 추정된 잠재능력 θ를 피드백 생성의 인지적 제어(control signal)로 실제 연결하는 것이다. PsyScore는 θ의 안정화를 위해 IRT 가정에 맞춘 제약과 clipping을 적용하고, GPCM의 차별도/난이도 파라미터에 대해 grid-search 기반 초기화(prior 정렬)로 mode collapse를 완화한다. 또한 생성 단계에서는 여러 LLM 에이전트가 초안을 만들고 DeepSeek-V3.1이 latent 진단 벡터 Dx(및 ZPD 제어 토큰)에 맞춰 Generate-and-Fuse로 충돌을 정리하며, 피드백 평가는 pairwise preference와 student revision simulation을 함께 사용한다.

- **Empirical Impact**: ASAP++에서 PsyScore-AES는 평균 QWK 0.747로 기존 SOTA를 능가하며, 8개 프롬프트 중 6개에서 1위를 기록했다. 특히 trait 차원에서도 다수 항목에서 상위 성능을 보였고, IRT 모듈의 grid-search 보정이 성능을 크게 끌어올리는 것으로 나타나 psychometric calibration의 실효성을 확인했다. 피드백 품질 측면에서는 actionability와 adaptivity에서 강한 승률을 보였고, 시뮬레이션 기반 revision gain은 저숙련(θ<-1)에서 +17.38%로 크게 향상되며 고숙련에서는 ceiling effect로 완만해져 ZPD 정렬이 반영됨을 시사한다.



### The Register Gap: A Meaning Intelligence Framework for Nigerian Public Discours (https://arxiv.org/abs/2606.20255)
Comments:
          Preprint. 12 pages, 2 tables. Supplementary materials: MIF Master Specification v2.0, Annotation Guidelines v1.0, and 30-item public calibration set with gold labels available from the author

- **Prior Approaches**: 기존 나이지리아 언어/담론 벤치마크(예: NaijaSenti, AfriSenti)는 감정 분류를 positive, negative, neutral의 3분류 극성 과제로 다뤘습니다. 하지만 논문은 번역 실패보다 문맥 실패가 주된 실패 모드라고 지적하며, 같은 발화도 화자·청중·상황에 따라 반대의 화용적 힘(pragmatic force)을 가질 수 있다고 봅니다. 결국 표면 감정 라벨만으로는 의도와 맥락을 분리해 평가하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 논문은 Meaning Intelligence Framework(MIF)라는 9개 차원 주석 및 평가 스키마를 제안해, 표면 sentiment과 실제 communicative intent(진정한 전달 의도)를 분리해 측정합니다. register, surface sentiment, true intent, irony, coded subtext, risk tier, annotator confidence, speaker emotion, recommended communications action을 점수화해 담론의 의미를 다차원으로 캡처합니다. 또한 재현성을 위해 프레임워크 사양과 주석 가이드라인, 30-item calibration 세트를 공개하고 오염 방지 목적의 private holdout도 유지합니다.

- **Technical Challenges**: 핵심 기술 과제는 문맥 의존적 화용 정보를 안정적으로 구조화해 모델이 읽을 수 있게 만드는 것입니다. 이를 위해 9차원 스키마를 prompt에 반영한 schema-informed prompting을 설계해, zero-shot보다 register 및 coded subtext 같은 문맥 신호를 더 잘 끌어내도록 했습니다. 실험에는 Gemini 2.5 Flash를 사용해 Standard English, Nigerian English, Nigerian Pidgin, code-mixed register 전반의 문맥 변이를 평가했습니다.

- **Empirical Impact**: 실험 결과의 핵심은 Register Gap으로, zero-shot register 분류 정확도가 33.3%에 그쳤지만 MIF 스키마를 in-context로 제공하면 73.3%로 크게 상승했습니다(약 +40점). 전체 Meaning Intelligence Score도 schema-informed prompting에서 73.2에서 78.6으로 5.4점 개선됐고, 특히 coded-subtext detection(+10점)과 strategic action recommendation(+10.3점)에서 실용적 이득이 두드러졌습니다. 나이지리아 공공 담론을 대상으로 한 평가가 단순 감정 극성에서 벗어나 ‘의미 지능’을 정량화하는 방향성을 제시했다는 점에서 의미가 큽니다.



### Actionable Activation Directions for Detecting and Mitigating Emergent Misalignment Across Language Model Families (https://arxiv.org/abs/2606.20225)
Comments:
          12 pages, 2 figures

- **Prior Approaches**: 기존 연구는 insecure code로 파인튜닝하면 의도와 무관한 프롬프트에서도 위험한 행동(코드 스필오버 등)이 나타나는 emergent misalignment 현상을 주로 관찰하는 데 집중했다. 또한 단일 모델 패밀리 내부에서는 misalignment가 활성 공간의 선형 방향으로 요약될 수 있고, 같은 아키텍처 내 파인튜닝 간에도 일부 전이될 수 있다는 결과가 나왔다. 다만 아키텍처가 다른 모델들 사이에서 같은 “기하”가 공유되는지, 그리고 전이가 타깃된 교정(특이성)으로 이어지는지에 대한 검증은 부족했다.

- **Core Contribution**: 이 논문은 Qwen2.5-1.5B, Gemma-2-2B, Llama-3.2-1B, Ministral-3-3B 등 4개 instruction-tuned 패밀리에서 동일한 QLoRA 조건으로 insecure code를 학습한 뒤, 최종 잔차 스트림(residual stream) 레이어에서 difference-in-means 방향이 aligned/misaligned 활성들을 거의 완벽히 분리함을 보인다(99.6%). 더 나아가 동일한 선형 방향을 빼는 causal steering이 코드 스필오버를 21–51점 줄이며, secure-code 음성 대조로 학습 콘텐츠 기반의 기하임을 확인한다.

- **Technical Challenges**: 핵심 과제는 (1) 활성 공간의 “방향”이 단순 상관이 아니라 실제 행동에 인과적으로 연결되는지, (2) 다른 아키텍처로 옮겼을 때도 그 방향이 특이적으로 작동하는지(랜덤·직교 방향 대비)였다. 저자들은 within-model에서는 random/orthogonal/wrong-layer-wrong-source 통제로 특이성을 게이트했고, cross-architecture에서는 ridge regression으로 선형 매핑을 만든 뒤 동일한 특이성 음성 대조를 통과하는지 확인했다. 그 결과 cross-architecture 매핑은 행동 억제(Δ=13–46)를 만들지만 방향 특이성은 실패하며, two-tier specificity 구조(내부는 특이적·공유된 매핑은 비특이적)가 드러났다.

- **Empirical Impact**: 실험은 four-family 전반에서 misalignment 기하의 선형 접근 가능성과 인과성을 강하게 뒷받침한다. cross-architecture 전이에서는 Gemma와 Qwen이 “donor”로 더 잘 억제하며(예: Gemma→Llama, Gemma→Ministral에서 큰 Δ), Llama는 “receiver”로서 수신은 잘 되지만 donor 역할은 약한 비대칭 topology가 관찰된다. 다만 교정이 타깃 축을 정확히 겨냥하는 것이 아니라 목표 모델 레이어의 일반적 민감도에 더 가깝다는 점을 보여줘, 감사(auditing)는 within-model probing을 중심으로, cross-architecture 교정은 비선형 방법이나 아키텍처별 프로브가 필요하다는 실무적 한계를 제시한다.



### CzechDocs: A Multiway Parallel Dataset of Formatted Documents for Minority Languages in Czechia (https://arxiv.org/abs/2606.20212)
- **Prior Approaches**: 기존 마크업 보존 번역(format-preserving MT) 평가는 BLEU/COMET처럼 의미·어휘 정확도 중심이라, HTML/XML 태그나 문서 구조가 실제로 얼마나 잘 유지되는지까지는 정교하게 다루기 어려웠습니다. 실무에서는 detag-and-project(태그 제거 후 정렬 기반 재삽입), 태그를 입력에 그대로 두고 출력에 복사하도록 유도하는 방식, 그리고 word alignment/휴리스틱 기반 reinsertion 등 다양한 절충이 쓰였습니다. 다만 공개 데이터셋은 제한적이어서 문서 단위·태그 밀도 높은 설정에서의 체계적 비교가 어려웠습니다.

- **Core Contribution**: 본 논문은 체코어와 체코 내 소수 언어(주로 우크라이나어, 영어)를 아우르는 형식화 문서(HTML/DOCX/PDF) 멀티웨이 병렬 데이터셋 CzechDocs를 제안합니다. 문서 번역 시 마크업 태그와 포맷을 함께 보존해야 하는 번역 평가를 목표로, 유효성(validation)용 분할과 평가 툴킷을 공개합니다. 향후 문서 수준 shared task를 위한 테스트 분할도 별도로 보류합니다.

- **Technical Challenges**: 핵심 난제는 원문·번역문이 “동일한 구조/세그먼트 경계”를 갖추도록 정렬(alignment)하면서, HTML 태그·인라인 마크업을 깨지 않게 추출/변환하는 것입니다. 저자들은 Okapi Tikal 기반 XLIFF/MOS 추출과 세그먼트 검증을 수행하고, 일부 HTML에 대해서는 소스 HTML을 수동 수정해 세그먼트 단위 정렬을 맞춘 뒤 재내보내기(reexport) 워크플로로 언어 버전 간 경계를 통일했습니다. 이후 LLM 번역에서는 detag-and-project(정렬 기반 태그 재삽입)와 tagged input(태그를 포함해 프롬프트로 보존 유도)을 비교하며, 문맥 포함 여부와 태그 강조 지시가 결과에 미치는 영향을 함께 실험합니다.

- **Empirical Impact**: 검증(validation) 세트에서 Aya-expanse-8b와 gpt-4.1-nano을 대상으로 태그 포함 BLEU와 detagged BLEU를 비교한 결과, 프롬프트에서 태그 보존을 명시할 때 성능이 눈에 띄게 개선되는 경향이 관찰됐습니다. detag-and-project와 tag-aware prompting은 평균적으로 유사한 태그 점수를 보이되, 문서별로 한쪽이 더 유리한 편차가 컸고 이 원인 분석은 후속 연구로 남겼습니다. 전체적으로는 마크업 보존까지 포함한 평가용 공개 자원과 벤치마크 축을 제공해, localization·문서 번역 분야에서 “구조 충실도”를 함께 측정하려는 움직임에 실질적인 기반을 마련했다는 점에서 의미가 있습니다.



### Pitch Spelling Jazz Lead Sheets, Solo Transcriptions, Classical Piano and Monophonic Scores (https://arxiv.org/abs/2606.20198)
- **Prior Approaches**: 기존 연구들은 음높이(MIDI-like 입력)만으로 음명(note names)이나 조성(key) 정보를 추정할 때, 각 음을 독립적으로 해석하거나 단일 기준으로 최소화해 실제 악보 표기(임시표의 개수, 조표 적용 범위)를 충분히 함께 최적화하지 못하는 경우가 많았습니다. 특히 바(bar) 단위로는 스케일을 정해도, 전체 곡의 global Key Signature와의 정합성이 뒤늦게 반영되어 표기 품질이 흔들릴 여지가 있었습니다.

- **Core Contribution**: 이 논문은 음높이와 마디 경계를 입력으로 받아, 각 마디의 local scale과 곡 전체의 global Key Signature, 그리고 표기용 note names를 함께 추정하는 알고리즘을 제안합니다. 두 단계 최적화(modal→tonal)로 관련 요소를 단계적으로 결합 평가해, 최종적으로 악보에 가장 자연스럽게 출력되는 표기 조합을 찾습니다.

- **Technical Challenges**: 핵심 난제는 주어진 반음(semitones) 정보가 여러 조성과 스케일로 동시에 설명될 수 있어 모호성이 크다는 점이며, 이때 임시표(accidentals)를 최소화하면서도 전체 곡의 조성 일관성을 유지해야 한다는 것입니다. 연구진은 modal 단계에서 각 마디에 대해 accidentals 수를 최소화하는 후보 스케일을 shortest-path search로 제안한 뒤, tonal 단계에서 이 local scales를 이용해 전체 곡의 표기가 최적인 global Key Signature와 note names를 재추정하도록 설계했습니다.

- **Empirical Impact**: Real Book 기반 재즈 리드시트, 재즈 솔로/베이스 라인 전사, 전통 곡, 피아노 및 단선악기 고전 악보 등 다양한 디지털 스코어 데이터셋에서 평가를 수행해 제안 방식의 실용성을 확인했습니다. 또한 재즈 스케일 간 거리를 새롭게 정의해 음악학적 분석에도 활용 가능성을 넓혔으며, 특히 오디오 기반 음악 전사에서 디지털 컬렉션 구축과 교육·문화유산 보존 같은 응용에 기대 효과가 큽니다.



### ReNikud: Audio-Supervised Hebrew Grapheme-to-Phoneme Conversion (https://arxiv.org/abs/2606.20179)
- **Prior Approaches**: 기존 Modern Hebrew G2P(문자-발음 변환)는 abjad 표기 특성상 모음이 대부분 생략돼 모호성이 크기 때문에, 먼저 니쿠드(nikud) 모음 부호를 예측한 뒤 IPA로 변환하는 방식이 주류였다. 하지만 니쿠드 라벨은 만들기 어렵고 데이터도 부족하며, 어휘 강세 같은 음운 특징을 직접 반영하지 못하고 공식 문법 규칙에 가까운 “문어적” 정보에 머무르는 한계가 있었다. 또한 단순 end-to-end 방식의 문자→IPA 직접 예측은 제한된 데이터에서 성능이 잘 오르지 않고, abjad의 문자-정렬(align­ment) 성질을 충분히 활용하지 못한다.

- **Core Contribution**: 논문은 ReNikud으로, (1) 대량의 무라벨 히브리어 음성에 대해 phoneme 기반 ASR을 이용한 pseudo-labeling으로 약한 오디오 감독 신호를 만든다. 이렇게 얻은 phonemic transcriptions은 수작업 라벨 없이 자연 발화 관행을 더 잘 반영하도록 설계됐다. (2) 동시에 pseudo-vocalization 구조로 문자 위치마다 IPA phoneme을 예측해, abjad의 문자-정렬 특성을 귀납적 편향으로 강제한다.

- **Technical Challenges**: 핵심 기술 난제는 (i) 무라벨 음성에서 신뢰할 만한 phoneme 수준 pseudo-label을 만들 수 있는지, (ii) 제한된 정답 데이터에서 문자-IPA 매핑을 안정적으로 학습할 수 있는지에 있다. ReNikud은 phoneme 기반 ASR pseudo-labeling 파이프라인으로 자연 발화에 가까운 학습 신호를 확보하고, pseudo-vocalization이 문자 위치별 예측을 수행하도록 모델 구조를 정렬 중심으로 설계해 데이터 효율과 학습 안정성을 동시에 노렸다. 결과적으로 “직접 IPA 예측”이 놓치던 문자 수준 alignment 이점을 되살리는 전략이 된다.

- **Empirical Impact**: 평가에서는 기존 히브리어 G2P 벤치마크와, 구어 히브리어를 겨냥한 신규 MILIM 벤치마크에서 ReNikud이 이전 state-of-the-art를 능가한다. 특히 자연 발화 관행을 반영한 약한 오디오 감독과 문자-정렬 유도 설계가 구어 성능에서 효과를 보였다는 점이 의미 있다. 저자들은 코드와 학습 모델을 공개해 히브리어 TTS 및 관련 음성 기술의 확장에 직접 활용될 전망이다.



### MedRLM: Recursive Multimodal Health Intelligence for Long-Context Clinical Reasoning, Sensor-Guided Screening, Evidence-Grounded Decision Support, and Community-to-Tertiary Referral Optimization (https://arxiv.org/abs/2606.20164)
Comments:
          9 pages, 3 figures, 3 tables, 1 Algorithm, 29 equations

- **Prior Approaches**: 기존 의료 LLM과 RAG는 긴 환자 정보가 한 프롬프트에 압축되거나 단일 검색에 의존하는 경우가 많아, 장문 구간의 핵심 근거를 놓치고 근거 추적성과 신뢰성이 약해질 수 있다. 또한 순수 long-context 확장이나 retrieval만으로는 센서 신호 같은 비정형 입력을 위험도·워크플로우·의뢰 의사결정까지 일관되게 연결하기 어렵다. 그래프 기반 RAG와 멀티모달 RAG가 사실성은 개선했지만, 재귀적 임상 추론 흐름과 불확실성 기반 안전 정제, 의뢰 최적화까지 “시스템”으로 모델링한 연구는 제한적이었다.

- **Core Contribution**: MedRLM은 환자 케이스를 하나의 거대한 입력이 아니라 외부 임상 환경으로 보고, 재귀적으로 분해·검색·검증·종합하는 Recursive Multimodal Health Intelligence 프레임워크를 제안한다. 텍스트/EHR/이미지/센서/가이드라인/의뢰 규칙을 전담 에이전트가 조율하며, Clinical Evidence Graph Memory로 환자 관찰과 표준 정의, 근거, 의뢰 기준을 감사 가능하게 연결한다. 센서 기반 이상 패턴이 깊은 추론을 “트리거”하고, 불확실성이 높을 때는 불확실성-게이트 refinement 또는 clinician review로 안전한 경로를 선택한다.

- **Technical Challenges**: 핵심 난제는 긴·이질적 데이터에서 관련 근거를 안정적으로 찾아 쓰는 것과, 다중 모달 근거를 의뢰/치료 경로 결정에 안전하게 결합하는 것이다. 논문은 컨텍스트 복잡도(길이·모달 다양성·근거 분산·임상 위험·모순)를 계산해 재귀 분기 여부를 제어하고, 에이전트별 모달 추론 결과를 그래프 메모리에 정규화된 evidence weight로 축적한다. 더불어 센서 신호는 baseline-adjusted 이상도와 임계값으로 재귀 트리거를 만들고, 불확실성 점수(예측 불확실성·자기일관성·근거 충돌)를 기준으로 추가 검색/재귀 정제를 수행해 무리한 생성과 과신을 줄인다.

- **Empirical Impact**: 실험 파트에서는 MedRLM의 입력 채널을 실제 임상 데이터로 커버하는 벤치마크 설계를 제시하며, EHR(long-context), 흉부 영상/리포트, 생체신호/ECG, ICU 시계열 등 서로 다른 데이터 축을 포괄하도록 구성한다. 공개/자격 있는(real-data) 데이터만 사용하고 합성 케이스를 섞지 않았다는 점을 강조하며, 다만 공개 데이터에서 “community-to-tertiary referral” 라벨은 드물어 ICU admission·사망·급격한 악화·전문가 에스컬레이션 같은 proxy outcome로 평가해야 함을 명확히 한다. 또한 실제 결과 수치 대신 현재 공개된 real-data baseline 앵커(예: PhysioNet/CinC 2012 mortality의 공식 점수)를 제시해, 향후 MedRLM이 반드시 기존 기준선 대비 검증 가능하도록 함정 설정을 피하고 재현성/검증 가능성을 확보하려는 방향성을 보여준다.



### From Texts to Scores: Tracing the Emergence of Essay Quality Representations in Large Language Models (https://arxiv.org/abs/2606.20152)
Comments:
          This is a preprint of a manuscript currently under peer review

- **Prior Approaches**: 최근 LLM 기반 Automated Essay Scoring(AES)은 성능이 크게 향상됐지만, 점수 산출에 쓰이는 내부 표현 메커니즘은 충분히 해명되지 않았다. 기존 연구는 주로 출력 성능이나 학습 설정을 다루었고, hidden representations에서 품질 신호가 어떤 방식으로 인코딩되는지에 대한 분석은 제한적이었다.

- **Core Contribution**: 저자들은 8개 LLM의 hidden representations를 ASAP++, CSEE, ENEM(포르투갈어) 세 데이터셋에 걸쳐 체계적으로 분석한다. 결과적으로 에세이 품질 정보가 LLM 표현 안에 linearly accessible 형태로 존재하며, 레이어 전반에서 점진적으로 형성되고 다양한 prompting 전략에도 비교적 견고하다는 증거를 제시한다.

- **Technical Challenges**: 핵심 난제는 LLM 표현이 실제로 점수를 어떻게 담고 있는지 검증 가능한 형태로 드러내는 것이다. 선형 probing, cross-prompt generalization, 차원 축소, neuron-level 분석을 조합해 품질 신호의 선형 판독 가능성을 점검했으며, nonlinear probe는 선형 대비 향상폭이 작고 일관성이 낮아 대부분 신호가 선형 디코더로 충분함을 뒷받침한다.

- **Empirical Impact**: 추가로 특정 ‘essay scoring neurons’의 활성은 에세이 점수와 강하게 상관하고, 표적 개입에 민감하게 반응하는 것으로 확인돼 해석 가능성 실마리를 제공한다. 또한 essay length에 따라 이 뉴런들의 레이어 분포가 체계적으로 이동하며 긴 에세이는 더 깊은 레이어에 더 의존한다는 관찰로, LLM 기반 AES의 내부 동작을 보다 구조적으로 이해하는 데 의미가 있다.



### When Does Streaming Tool Use Help? Characterizing Tool-Intent Stabilization in Streaming Retrieval-Augmented Generation (https://arxiv.org/abs/2606.20113)
- **Prior Approaches**: 기존 Retrieval-Augmented Generation(RAG)은 검색 결과를 생성에 접목해 정확성을 높이지만, 도구 호출(tool call) 자체가 대화 지연을 만든다는 문제가 있었다. Streaming RAG은 입력이 끝나기 전에 speculative tool query를 병렬로 보내고 결과가 충분한지로 반영해 지연을 줄이려 했으나, 벤치마크 전체의 평균 개선만 보고 “어떤 쿼리에서” 이득이 실제로 가능한지 메커니즘을 쿼리 단위로 설명하진 못했다. 또한 기존 지표는 주로 시스템/집계 성능에 초점이어서, 사용자 발화 흐름 중 어느 시점에 의도(검색 대상)가 결정되는지가 성능 상한을 어떻게 좌우하는지 불명확했다.

- **Core Contribution**: 이 논문은 Streaming RAG의 이득이 “시스템이 똑똑해서”가 아니라 “검색 쿼리를 결정짓는 정보가 입력 스트림에서 언제 처음 등장하는가”에 의해 좌우된다는 점을 정량화한다. 이를 위해 tool-intent stabilization(도구 의도 안정화)이라는 쿼리 고유의 측정치를 정의하고, 이 값이 얼마나 일찍 안정되는지에 따라 숨길 수 있는 tool latency의 최대 비율을 model-agnostic bound H로 상한 계산한다. 결과적으로 어떤 발화가 speculative query로 latency hiding이 가능한지, 그리고 그 비용 대비 가치가 언제 생기는지 배포 관점의 판단 기준을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 “정답을 담고 있는 증거가 접두사(prefix) 단계에서 실제로 retrievable해지는 순간”을 구분해 안정화 시점을 측정하는 것이다. CRAG에는 gold passage 레이블이 없어서, 정답 문자열을 검색 문서에 grounding(기계적 문자열 매칭)해 sufficiency stabilization의 시간을 계산하고, grounding이 만드는 오차/편향을 줄이기 위해 엄격/완화(relaxed) 두 가지 grounding 팔을 함께 운영한다. 또한 실제 스트리밍 파이프라인을 재현하는 비동기 harness로, 이론적 bound H가 측정된 perceived-latency saving을 보수적으로(상한처럼) 설명하는지 검증하며, CPU-only·training-free 환경에서 재현 가능한 분석을 수행한다.

- **Empirical Impact**: CRAG validation 1371문항에서 grounding 가능한 비율은 35.4%였고, BM25가 어떤 접두사에서든 gold evidence를 top-k에 올리는 쿼리는 21.3%였다. sufficiency 관점 안정화는 매우 일찍(평균 φ_suf=0.26, 중앙값 0.14) 일어나며, 분포는 가설처럼 뚜렷한 bimodal이라기보다 “초기 안정화가 많은 쏠림 + 얇은 지연 꼬리” 형태를 보였다. 배포 핵심 수치로는 L=600ms, δ=3w/s, θ=0.8에서 evidence가 제때 retrievable한 쿼리(=95.2%)가 높게 나타나지만, top-1이 안정화되는 것까지 섞은 전체 blend 기준으로는 73.9%로 낮아져 지표 해석에 주의가 필요함을 시사한다. 또한 question type이 early/late를 유의미하게(다만 효과 크기는 작게, 대략 4% 수준) 갈라 주며, reasoning 복잡도보다 발화 내 entity 위치가 retrieval-sufficiency 안정화를 좌우한다는 관찰로 이후 “learned speculative trigger” 설계 방향을 구체화한다.



### HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization (https://arxiv.org/abs/2606.20097)
- **Prior Approaches**: 기존 장기 컨텍스트 LLM들은 Full Attention(FA)의 O(T^2) 병목을 줄이기 위해 Linear Attention(LA)과의 하이브리드를 주로 layer-wise로 구성해 왔습니다. 하지만 선행 연구는 특히 LA와 FA를 결합할 때 “expressivity collapse” 및 학습/표현 통합의 난제가 커서, 하이브리드 설계 공간이 충분히 탐색되지 못했다고 지적합니다. 또한 token-wise/경량 라우팅 방식은 이질 경로가 섞일 때 장기 표현의 안정성이 무너지기 쉽고, sparse 계열은 인퍼런스에서 KV 캐시 보존 비용이 커질 수 있습니다.

- **Core Contribution**: 이 논문은 attention hybridization의 단위를 layer가 아닌 head로 옮기는 HydraHead를 제안합니다. 메커니즘 해석 결과, 같은 레이어 안에서도 head들은 입력을 공유하면서도 기능적으로 크게 이질적이며, 레이어 단위는 블록 형태로 출력 유사성이 매끈하게 나타나 더 구분력이 약하다는 통찰을 제공합니다. 이를 바탕으로 head 축에서 FA와 LA를 선택·혼합해 장기 컨텍스트의 정밀 검색 능력과 선형 비용의 효율을 동시에 노립니다.

- **Technical Challenges**: 핵심 난제는 (1) 어떤 head는 FA를 유지해야 하는지, (2) 서로 다른 방식(FA vs LA)이 만든 head 출력 분포 차이를 어떻게 안정적으로 융합할지입니다. 저자들은 activation patching과 path patching을 이용해 retrieval에 결정적인 “causally indispensable” head를 식별하고, 그 head에만 FA를 보존하는 interpretability-driven 선택 전략을 설계했습니다. 더불어 FA/LA 출력의 분포 갭을 scale-normalized fusion 모듈로 완화하고, 파라미터 재사용+distillation 기반의 3-stage transfer 파이프라인으로 최소한의 학습 오버헤드로 하이브리드화를 수행합니다.

- **Empirical Impact**: 통합 학습 설정에서 HydraHead는 장기 컨텍스트 태스크에서 다른 하이브리드 설계를 능가하면서도 일반 추론 성능을 유지합니다. 특히 해석 기반 head 선택 덕분에 layer-wise 3:1 혼합과 유사한 long-context 성능을 LA-to-FA 비율 7:1에서도 맞추는 결과를 제시합니다. 또한 15B 토큰 학습만으로 512K 컨텍스트에서 baseline 대비 69%+ 개선을 달성하며, native 256K 컨텍스트를 가진 동급 크기 선두 모델(Qwen3.5)에 근접해 head-level hybridization의 스케일링 잠재력을 실험적으로 강조합니다.



### Self-Preference Is Weak or Absent in Verifiable Instruction-Following Revision: A Four-Model Test Under Genuine Authorship (https://arxiv.org/abs/2606.20093)
Comments:
          7 pages, 3 tables. Code and data: this https URL

- **Prior Approaches**: LLM-as-a-judge 연구는 모델이 자신이 생성한 텍스트를 더 잘 인식하고 더 선호하는 self-preference(자기 편향)를 문서화해 왔습니다. 또한 self-refinement 같은 폐루프 맥락에서는 편향이 더 커질 수 있다는 결과가 있습니다. 하지만 ‘자기 글을 수정할 때도’ 그 편향이 그대로 나타나는지는, 특히 개선 여부를 다른 LLM이 판단하는 방식(순환성)과 ‘진짜 작성자’ 구분이 모호한 방식 때문에 깔끔하게 답하기 어려웠습니다.

- **Core Contribution**: 이 논문은 IFEval의 deterministic verifier(명령-지시 준수 여부를 프로그램이 판정)로 “검증된 좋은 수정(verified-good fix)”을 만들고, 그 수정이 작성자(author)의 수용률을 실제로 떨어뜨리는지 테스트합니다. 비교는 같은 초안에 대해 ‘인컨텍스트에서 실제로 초안을 작성한 모델’ vs ‘새 모델(fresh model)’이 편집을 받아들이는지로 설계해 told-label 혼선을 줄였습니다. 결과적으로, 검증된 좋은 수정은 자기 생성물에 대해 신뢰를 덜 하거나 더 가혹하게 거부하는 패턴이 관측되지 않았습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) “이 수정이 좋은가”를 다시 LLM 판단으로 맡기면 편향이 재주입되는 순환성 문제, (2) 비교 대상이 ‘실제 작성자’와 ‘다른 모델 초안’인지 분리되지 않는 문제였습니다. 저자들은 체크를 모델이 아니라 IFEval의 공식 deterministic checker가 하도록 고정하고, draft++fix를 모델이 직접 받아들이는 author 조건과 생성 이력을 모르는 fresh 조건으로 분리했습니다. 또한 atomically editable한 제약만 사용해 한 번의 최소 편집이 ‘통과로 전환’되는 cases만 남기고, 중복 편집(이미 반영된 수정) 때문에 거부가 발생하는 아티팩트를 줄였습니다.

- **Empirical Impact**: 4개 mid-tier 모델군과 총 85개의 author-versus-fresh 비교(usable draft 기준)에서, 작성자는 검증된 좋은 수정의 거부율이 fresh 대비 5.1%p 낮거나(자기 글에 더 관대) 같은 수준이며 95% CI도 0을 포함했습니다(CI [-12.9, +2.7]). 즉 정량적으로는 self-preference가 이 verifiable revision 셀에서는 ‘탐지 불가’로 나타났습니다. 다만 원인의 질적 분석에서, 작성자가 거부하는 경우의 97%는 “내 글을 선호해서”라기보다 verifier가 요구한 조건이 여전히 결함이 있거나(또는 다른 제약을 건드림) 같은 flaw-catching 사유였고, 거부율 자체가 높아진 것은 아니었습니다.



### IHUBERT: Vector-Based Semantic Deduplication and Domain-Balanced Pretraining for Persian Resources (https://arxiv.org/abs/2606.20089)
- **Prior Approaches**: 기존 페르시아 PLM(pretrained language models)은 대규모 고품질 사전학습 말뭉치의 부족과, 분류·NER 같은 표준 작업에 편중된 평가 한계로 인해 성능 확장에 제약이 있었다. 또한 말뭉치 품질(정규화/중복/익명성)과 도메인·레지스터 균형을 체계적으로 통제하지 못한 경우가 많아 모델의 일반화가 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 RoBERTa-base(125M) 인코더를 from scratch로 학습한 단일언어 페르시아 PLM “IHUBERT”를 제안하며, 45GB 규모의 Sepahr-Danesh 기반 정제 코퍼스(약 70억~80억 토큰)를 사용한다. 특히 토크나이저를 위해 전체 사전학습 코퍼스에서 139k-vocabulary BPE를 학습해 페르시아의 형태론·철자 변이를 더 잘 포착하도록 설계했다.

- **Technical Challenges**: 핵심 난관은 (1) 말뭉치 정제와 중복 제거를 단순 문자열 수준을 넘어 의미 중복까지 줄이면서, (2) 도메인·레지스터 분포 균형을 유지하는 것이었다. 이를 위해 정규화, exact/near-duplicate 제거, anonymization, 그리고 벡터 DB 기반 semantic deduplication을 멀티스테이지 파이프라인으로 적용했으며, 토크나이저는 BPE/WordPiece를 비교한 제어 실험(토크나이저 ablation)으로 단편화 감소를 확인했다.

- **Empirical Impact**: IHUBERT는 7개 페르시아 NLU 벤치마크에서 NER, sentiment, topic, NLI, extractive QA, relation extraction을 모두 평가하며 분류형과 이해형 과제를 폭넓게 커버한다. 특히 extractive QA에서 PQuAD(F1 88.3542), ParsiNLU-RC(F1 49.0987) 1위를 달성했고, FarsTail(Macro-F1 0.8350)에서도 최고 성과를 보였다. 반면 relation extraction은 PERLEX에서 Macro-F1 0.6684로 상대적 격차가 남았지만, 의미 정제 대규모 사전학습과 확장 평가를 통해 페르시아 언어 모델링의 실질적 진전을 제시했다.



### Source-Grounded Data Generation for Text-to-JSON Learning (https://arxiv.org/abs/2606.20072)
Comments:
          Preprint

- **Prior Approaches**: 기존 text-to-JSON 연구는 직접 prompting, constrained decoding, structured-output API처럼 형식 준수를 먼저 확보하는 방식이 많았고, 최근에는 value-level fidelity까지 평가하려는 흐름이 커지고 있습니다. 다만 훈련용 JSON-라벨을 만들 때 사람 주석은 비용과 확장성 한계가, LLM 생성 supervision은 순환 의존(이미 잘하는 모델이 다시 라벨을 만드는 문제)으로 신뢰성이 떨어질 수 있습니다.

- **Core Contribution**: 이 논문은 STAGE(Spreadsheet-grounded Text-to-JSON Artifact GEneration)라는 소스-근거 기반 데이터 생성 파이프라인을 제안합니다. LLM이 보고서와 JSON 스키마/타깃을 합성하되, 스프레드시트라는 공통 “정답 근거”에 대해 생성된 JSON 값이 검증되면 남기도록 설계해 값의 근거성을 보장합니다.

- **Technical Challenges**: 핵심 기술 난제는 “스케일은 올리되, JSON 타깃의 값이 실제 입력 소스(보고서에 삽입된 표)에서 나온 것임을 검증”하는 균형입니다. STAGE는 스프레드시트를 Markdown 테이블로 중간 표현화한 뒤, (1) 보고서 생성 시 <original_table>을 원본 테이블로 치환하고, (2) JSON의 모든 leaf value를 경로-값 쌍으로 펼친 후 코드 기반(정규화 후 부분문자열/토큰 커버리지) 렉시컬 검증을 통과한 예만 채택하는 방식으로 해결했습니다.

- **Empirical Impact**: STAGE-Eval(테스트 851개)에서 Qwen3-4B의 exact match는 31.37%에서 74.27%로, value accuracy는 45.46%에서 90.69%로 크게 향상됐습니다. 구조적 안정성도 좋아져 파싱/절단 오류가 크게 줄었고, Llama-3 계열에서도 value 정확도가 4배 이상 개선되는 등 소규모 모델 전반에 이득이 전이되는 양상을 보였습니다.



### GEMS: Geometric Constraints Enable Multi-Semantic Superposition in LLMs (https://arxiv.org/abs/2606.19946)
Comments:
          30 pages, 5 figures, 20 tables. Code and logs are available at: this https URL

- **Prior Approaches**: Activation steering은 재학습 없이 추론 시 중간 hidden state를 바꿔 행동을 조절하지만, 기존 training-free 방식은 대체로 단일 semantic 방향에 집중했다. ActAdd, CAA, RepE, ITI 같은 방법은 한 방향 제어는 가능해도 여러 방향을 동시에 중첩하면 모델이 쉽게 붕괴(collapse)한다.

- **Core Contribution**: 이 논문은 다중 방향 steering 붕괴를 두 독립 요인으로 분해한다: (1) distributional deviation(여러 층에서 additive 잡음이 누적되어 활성 norm이 학습 분포를 벗어남)과 (2) directional interference(서로 직교하지 않은 semantic vector가 중첩될 때 상호 감쇠됨)이다. 이를 바탕으로 training-free 다중 방향 개입이 지켜야 할 설계 제약을 제시하고, 그에 맞춘 GEMS(Geometry-based Expert Multi-Steering)를 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 “재학습/최적화 없이” 동시에 여러 방향을 주입하면서도 (a) 활성 norm 안정성, (b) 방향 간 간섭 최소화, (c) 올바른 경로(특히 attention output projection oprojo_proj)만 타격하기를 동시에 만족시키는 것이다. GEMS는 Gram-Schmidt 기반 real-time orthogonalization으로 directional interference를 막고, norm-preserving weighted superposition과 oprojo_proj targeted injection으로 distributional deviation과 경로 오염을 제한하며, Gaussian envelope와 cosine decay로 층별 강도를 조절한다.

- **Empirical Impact**: GEMS는 GSM8K에서 비수학(non-mathematical) 3방향 동시 주입 시 정확도 98%(baseline 92%)를 달성했지만, 무제약 ActAdd는 4%로 collapse했다. Wikitext-2에서는 PPL이 2.2%만 증가(무제약 ActAdd는 약 1700배 악화)해 연속 언어모델 품질까지 유지됨을 보여주며, 컴포넌트 ablation과 layer-level probe로 orthogonalization·norm/경로 제약의 인과성을 분리했다.



### Light-weight Pronunciation Assessment via Discrete Speech Token Surprisa (https://arxiv.org/abs/2606.19910)
Comments:
          Accepted to Interspeech 2026

- **Prior Approaches**: 기존 자동 발음 평가는 GoP처럼 강제정렬 기반 ASR에 의존하거나, 특징을 점수로 회귀하는 방식이 주로 쓰였습니다. 하지만 labeled learner data나 비원어 말뭉치처럼 비싼 학습 데이터가 필요해 데이터가 부족한 환경에선 성능과 확장성이 떨어졌습니다. 최근에는 SSL 토큰과 pseudo-labeling 등으로 비용을 낮추려 했지만, 여전히 phone-level 자원/정렬/학습 데이터 의존이 남아 있는 편입니다.

- **Core Contribution**: 이 논문은 native speech만으로 토큰화·음운규칙(phonotactics) 우선의 사전분포를 학습하고, learner 발화를 정답 라벨 없이도 점수화하는 경량 프레임워크를 제안합니다. 핵심은 SSL 인코더+K-means로 만든 이산 토큰에서 native 토큰 언어모델의 surprisal을 계산해, 문맥에서 덜 예측되는 토큰 패턴(발음 편차)을 포착하는 것입니다. 또한 transcript가 있을 때는 Text2DUnit로 텍스트 기반 “기대 토큰열”을 만든 뒤 DTW로 정렬해 오류 민감 특징을 추가합니다.

- **Technical Challenges**: native-trained 토큰 공간에서 “발음 오류”가 실제로 드러나게 하려면, 강제정렬·phoneme inventory·mispronunciation 라벨 없이도 토큰 편차를 안정적으로 수치화해야 했습니다. 이를 위해 mean보다 국소 이상치에 민감한 surprisal의 표준편차/Spike Rate 같은 통계량을 설계하고, 텍스트가 주어질 때는 Text2DUnit–DTW 정렬 비용(centroid L2 기반)을 정규화해 길이 차이 문제를 완화했습니다. 마지막으로 surprisal 특징과 정렬 특징을 단순 회귀(Ridge)로 결합해 calibration까지 가볍게 처리합니다.

- **Empirical Impact**: SpeechOcean762에서 오디오 단독 라인도 기준선(aMRT)과 유사한 수준의 상관(PCC)을 보이고, transcript-guided 정렬 특징을 더하면 Acc./Flu./Pros.에서 0.60→0.66 수준으로 개선됩니다. 특히 SpeechOcean762 학습 모델을 그대로 L2-ARCTIC에 옮겨도 성능이 유지되며, 제로 리트레이닝 전이에서 상관이 일관되게 상승합니다. 네이티브 학습데이터를 960시간에서 100시간으로 줄여도 큰 붕괴 없이 안정적 성능을 보여 low-resource 설정에서의 실용성을 시사합니다.



### REDACT: A Systematically Controlled Multilingual Benchmark for Personal Information Detection (https://arxiv.org/abs/2606.19881)
Comments:
          14 pages, 5 figures

- **Prior Approaches**: 기존 PII 벤치마크는 언어·도메인 범위가 좁거나(대부분 영어 중심), 생성 조건이 템플릿화돼 행동 다양성이 제한되는 문제가 있었습니다. 또한 aggregate F1 중심 평가가 관행화되면서, 실제 규제 관점에서 중요한 고위험(예: HIGH-sensitivity)·비원문(disclosed/partial/obfuscated) 표면 조건에서의 취약한 실패 패턴이 가려졌습니다. 게다가 “실제로 드러난 PII인지, 가정/부분 노출인지” 같은 메타데이터가 공개 벤치마크에 충분히 반영되지 않았습니다.

- **Core Contribution**: 이 논문은 REDACT를 통해 다국어 PII 탐지의 실패 요인을 표면 조건과 규제 의미축에 맞춰 체계적으로 평가할 수 있게 합니다. REDACT는 13,427 레코드, 324,078 엔티티 주석, 51개 엔티티 타입, 25개 언어(9개 스크립트), 4,127개 surface-form 패턴을 제공하며, 9개 생성 축을 strength-2 covering array로 균형 샘플링합니다. 여기에 disclosed status, disclosure form, GDPR-aligned sensitivity tier(HIGH/MEDIUM/LOW) 메타데이터를 엔티티 단위로 붙여 aggregate F1을 넘어선 stratified recall 평가 프레임을 구현했습니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 다국어·다형식 PII를 반복 가능하게 생성하면서 (2) 주석 정합성(오프셋 등)을 자동 검증·수리하고 (3) 조밀한 축 조합에서 품질 게이트를 통과시키는 것이었습니다. 저자들은 7단계 파이프라인으로 단일 LLM 호출 기반 생성→결정적 오프셋 정합성 점검→조건 검증 및 repair→중복 제거→품질 감사(6개 release gate)를 수행해 94.0% 수준의 유지율로 데이터셋을 확정했습니다. 또한 LLM 출력은 offsets 신뢰성이 낮을 수 있어, 동일 입력에서 문자열 위치를 재추적(str.find)해 표준 exact/partial/fuzzy span 매칭으로 공정 비교가 가능하게 했습니다.

- **Empirical Impact**: 락(locked)된 언어-층화 샘플 1,000 레코드에서 Claude Sonnet 4.6이 partial-overlap micro-F1 0.636으로 1위를 보였고, GPT-4.1이 0.597, OpenAI Privacy Filter가 0.512, GLiNER가 0.320, Presidio가 0.195로 뒤를 이었습니다. 그러나 stratified 결과가 더 중요해, Presidio는 HIGH-sensitivity에서 recall 0.07로 크게 하락했고 비원문 형태(부분/obfuscated)에서도 급격한 격차가 관찰됐습니다(aggregate F1이 위험 구간을 숨김). 반면 LLM 계열은 HIGH tier에서 가장 강한 slice를 보이며, disclosure form과 sensitivity tier가 탐지 실패 구조를 가장 잘 드러내는 축임을 LLM-as-judge 평가가 뒷받침했습니다.



### The Almost Intelligent Revolution: Options for Scaling Up Deliberation and Empowering People with AI (https://arxiv.org/abs/2606.19864)
Comments:
          Published in /Handbook of Democracy in the Era of Artificial Intelligence/ edited by Evangelos Pournaras, Srijoni Majumdar, Carina Ines Hausladen, and Dirk Helbing. 2026

- **Prior Approaches**: 기존 연구는 red teaming 등 안전성 대응에 초점을 두지만, 언어적 제약·편향·sycophancy(사용자 의견에 맞추려는 경향) 같은 더 넓은 참여 왜곡은 충분히 다루지 못한다. 또한 참여의 핵심 매개체가 언어라는 관점에서, prestigious register(권위적 문체)의 규범이 특정 집단을 배제해왔다는 점이 반복적으로 지적돼 왔다. LLM을 단순 요약·정리 도구로 쓰면 그럴듯한 문장을 만들 수 있지만, 거짓(hallucination)과 데이터/알고리즘 편향이 민주적 숙의를 왜곡할 가능성이 남는다.

- **Core Contribution**: 이 장은 Systemic-Functional Linguistics(체계기능언어학)를 바탕으로, 언어 사용자의 사회인구학적 차이와 의사소통 기능 차이가 AI 지원 숙의 참여를 어떻게 바꾸는지 분석한다. 이를 통해 LLM이 숙의를 “확대(scale up)”하고 참여를 “democratise(민주화)”할 수 있는 조건(예: 언어 장벽 완화, 접근성 향상)을 제시하면서도, 과장(overclaiming)·과소보고(underclaiming)를 경계한다. 결국 목적은 이상적 발화 상황에 가까워지되, 언어 불평등이 재생산되지 않도록 윤리적 안전장치를 내재화하는 것이다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) LLM의 표면 언어 생성이 사용자의 의도/맥락과 불일치할 수 있고, (2) 훈련 데이터와 빈도 기반 선택으로 인해 소수 표현이 억압되는 편향이 생기며, (3) sycophancy가 합의 편향을 키울 수 있다는 점이다. 장은 RLHF 같은 정렬(alignment)과 guardrails로 위험을 줄이되, RAG로도 환각을 완전히 제거하기 어렵고 참조 문서의 문체가 사용자 언어와 어긋날 수 있음을 강조한다. 또한 iDem류 접근에서는 문장 단순화 유형을 예측·적용해 target audience에 맞추는 식으로, 권위적 문체의 진입장벽을 직접 낮추는 설계를 제안한다.

- **Empirical Impact**: Habermas Machine 연구에서는 영국 주민 표본(5,734명)에서 AI가 작성한 집단 성명이 인간 중재보다 품질·명확성·정보성·공정성에서 더 선호됐고, 집단 내 합의 수렴도 강화되는 경향이 나타났다. 다만 소수 입장 비판이 반영되더라도 전체 이동은 다수 의견 쪽으로 더 기울 수 있어, “포함”과 “가중치”가 동일하지 않을 수 있음을 보여준다. 한편 iDem 실험에서는 유럽 의회·UN 문장 중 상당 비율이 단순화가 필요하다는 결과(언어·출처별 92~96%)가 제시돼, 기관 언어가 실제로 참여 장벽이 됨을 경험적으로 뒷받침한다.



### Large Language Models Do Not Always Need Readable Languag (https://arxiv.org/abs/2606.19857)
Comments:
          23 pages, 10 figures. Preprint

- **Prior Approaches**: 기존 LLM 활용은 사람 친화적 자연어를 프롬프트/입력으로 사용하며, 다른 모델이 읽어야 하는 경우에도 구조적으로 자연어 형식을 고수해 왔습니다. 일부 연구는 압축 프롬프트나 요약을 통해 컨텍스트 비용을 줄이지만, 의미 보존을 체계적으로 검증하거나 ‘모델 전용 텍스트 표현’ 자체의 회복 가능성을 정량화하는 데는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 LLM이 생성·해석 가능한, 비표준적이지만 의미를 담는 텍스트 표현 계열을 BabelTele(“모델 네이티브 텍스트 표현”에 가까운 개념)로 정의하고 그 가능성을 실험적으로 탐색합니다. 특히 BabelTele를 고정 프로토콜이 아닌 ‘LLM의 표현 생성/해석 역량을 측정하는 진단 도구’로 제시하며, 의미가 압축된 텍스트에서도 복원되는지에 초점을 둡니다.

- **Technical Challenges**: 핵심 난제는 사람이 읽기 어렵게 비정형으로 바꿀 때도 LLM이 의미를 안정적으로 복원할 수 있는 표현을 설계·검증해야 한다는 점입니다. 논문은 가독성 진단, 모델 likelihood 기반 측정, 인간 설문, 그리고 다운스트림 태스크 평가를 함께 사용해 ‘의미 충실도’와 ‘컨텍스트 절감 효과’를 교차 검증하며, 특히 compressor-reader 페어와 태스크 세팅 의존성이 존재함을 보여줍니다.

- **Empirical Impact**: 실험 결과 BabelTele은 텍스트를 원문 대비 27.9%로 줄이면서도 의미 충실도 99.5%를 유지하는 등 높은 정보 밀도를 보였습니다. 또한 cross-model transfer, agent memory, multi-agent communication에서 컨텍스트 오버헤드를 줄이면서도 전반적으로 다운스트림 성능을 유지하되, 모델 쌍과 태스크에 따라 효과가 달라진다고 보고해 향후 LLM 시스템에서 ‘가독성-의미 복원 가능성’의 부분적 분리를 시사합니다.



### Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives (https://arxiv.org/abs/2606.19852)
Comments:
          7 pages, 2 figures, 3 tables. Affiliations: (1) Department of Health Outcomes and Biomedical Informatics, College of Medicine, University of Florida, Gainesville, FL, USA; (2) Division of Pulmonary, Critical Care and Sleep Medicine, Department of Medicine, College of Medicine, University of Florida, Gainesville, FL, USA; (3) College of Nursing, Florida State University, Tallahassee, FL, USA

- **Prior Approaches**: 기존 병리 보고서 정보 추출은 지도학습 기반 Named Entity Recognition과 Relation Extraction으로 해결해왔지만, 정답 라벨을 대규모로 수작업해야 한다는 비용 부담이 큽니다. 또한 앞단에서 엔티티를 놓치면 뒤의 관계 추출도 연쇄적으로 실패하는 cascade 문제가 나타납니다. 특히 종합병리(synoptic) 필드처럼 구조화 데이터가 중요한 과제에서 이 한계가 더 크게 드러납니다.

- **Core Contribution**: 이 논문은 13개 CAP(College of American Pathologists) 종합병리 필드를 대상으로, task-specific 학습 없이 zero-shot으로 LLM을 활용하는 agentic workflow를 제안합니다. 또한 GatorTron NER-RE 같은 최첨단 지도학습 기준선과 비교하기 위해, 레지스트리 정합(registry-aligned)을 반영한 새로운 평가 프레임워크를 도입합니다. 결과적으로 병리 서술문 안의 핵심 정보를 저비용으로 구조화하는 경로를 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) narrative 문서에서 요구되는 필드 스키마로 정확히 매핑하는 것, (2) 복잡한 관계(예: Pathologic Stage)를 놓치지 않고 추출하는 것, (3) 기존 성능평가가 레지스트리 요구와 불일치할 수 있다는 점입니다. 논문은 5종 open-source 생성형 LLM을 zero-shot으로 에이전트형 워크플로에 투입하고, 누락·오류가 실제 레지스트리 관점에서 어떻게 평가되는지 확인할 수 있는 맞춤형 평가 체계를 설계해 이를 완화합니다.

- **Empirical Impact**: 실험에서 지도학습 기준선 GatorTron NER-RE는 Micro-F1 0.960을 기록했고, 가장 좋은 zero-shot 모델 GPT-OSS-20B는 Micro-F1 0.893(재현율 0.949)로 복잡한 관계인 Pathologic Stage도 task-specific training 없이 추출했습니다. 이는 open-source 생성형 LLM과 zero-shot agentic 접근이 폐 절제 병리 보고서 정보 추출에서 실용적인 대안이 될 수 있음을 시사합니다. 병리 레지스트리 구축처럼 라벨 비용과 오류 전파가 큰 분야에서 특히 low-cost 확장 가능성이 기대됩니다.



### AtomMem: Building Simple and Effective Memory System for LLM Agents via Atomic Facts (https://arxiv.org/abs/2606.19847)
Comments:
          19 pages, 10 figures, 5 tables

- **Prior Approaches**: 기존 메모리 강화 에이전트들은 RAG와 유사하게 외부 저장공간을 붙이지만, 핵심은 ‘메모리 표현’과 ‘업데이트 안정성’에 달려 있다. 많은 접근이 대화 원문을 요약/정리해 넣거나 텍스트·그래프·요약 기반 표현을 사용하지만, 원문 저장은 잡음과 중복으로 검색 비용을 키우고, 압축은 미세 정보 손실과 함께 시간이 지날수록 LLM 오류가 누적되기 쉽다. 또한 평면(Flat) 검색은 세션 간 연관 근거를 체계적으로 잇지 못해 원격 의존성을 복원하기 어렵다.

- **Core Contribution**: AtomMem은 장기 대화에서 ‘가치가 높은 원자 단위 atomic fact’만 뽑아 저장하고, 이를 계층 이벤트(event)와 시간 프로파일(profile)로 조직해 안정적으로 진화하는 메모리를 만든다. Fact Executor는 긴 대화에서 핵심만 선택적으로 추출하며, 동질 사건을 묶어 episodic context를 형성하고 사용자 속성을 시간에 따라 추적한다. 검색 단계에서는 associative recall용 memory graph를 활성화해 파편화된 기억들을 연결하고, 그 결과를 LLM 응답 생성에 정합적으로 주입한다.

- **Technical Challenges**: 가장 큰 과제는 (1) 긴 대화에서 검색 가능한 고밀도·자기완결형 정보를 안정적으로 뽑는 것과 (2) 업데이트가 자유롭게 변형되면서 기존 사실이 붕괴하는 현상을 막는 것이다. AtomMem은 SFT로 학습된 Atomic Fact Extractor를 통해 coreference resolution, temporal anchoring 같은 가벼운 추론과 함께 노이즈를 제거하고, 구조화된 메타데이터(참여자/키워드/시간)를 붙인다. 중복·충돌은 후보군 필터링과 혼합 유사도(임베딩+키워드)로 국소적으로 탐색한 뒤 residual/업데이트 튜플로만 수정하여 글로벌 일관성을 유지한다.

- **Empirical Impact**: LoCoMo, LongMemEval 벤치마크에서 AtomMem은 여러 추론 태스크 전반에 걸쳐 SOTA 성능을 보였고, 특히 Multi-Hop과 Temporal에서 큰 J-score 개선을 기록했다. 더 단순한 AtomMem-Flat도 원문(history) 기반 기준선보다 크게 향상되며, 성능 격차가 ‘메모리 표현 품질’의 중요성을 강하게 시사한다. 동시에 전체 토큰 소비를 Mem0 대비 약 61.4% 절감하면서도 경쟁/우수 성능을 유지해, 개인화 지능형 에이전트를 장기 운용할 때의 확장성과 경제성에 의미가 있다.



### Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering in Language Models (https://arxiv.org/abs/2606.19831)
- **Prior Approaches**: 기존 연구는 refusal 같은 정렬된 행동이 희소한 FFN 뉴런·회로에 국소화된다는 점을 보여줬지만, “언제” 단일 뉴런 개입이 일관되게 행동을 바꾸는지에 대한 예측 이론은 부족했습니다. 또한 gradient 기반 attribution이나 고정 크기 activation steering은 원인적 개입과 잘 맞지 않아, 같은 뉴런이 어떤 실험에선 controllable로, 다른 실험에선 단순히 파괴적(collapse)으로 보이는 모순이 있었습니다.

- **Core Contribution**: 이 논문은 단일 뉴런 steering을 ‘용량(dose)’과 ‘제어 좌표’ 하나로 환원하는 budget-normalized control window 프레임워크를 제안합니다. 잔차 스트림과 해당 뉴런의 write 방향 정렬(alignment)이 보편적인 포화 곡선을 따라 움직이며, 행동이 뒤집히는 trigger가 출력 붕괴(collapse) ceiling보다 낮을 때에만 coherent control이 성립한다고 정식화합니다.

- **Technical Challenges**: 핵심 난제는 단일 뉴런 개입이 일으키는 효과가 dose에 따라 ‘행동 전환’과 ‘출력 붕괴’로 동시에 갈라지는데, 그 경계를 미리 예측할 수 없다는 점이었습니다. 저자들은 collapse ceiling을 가중치와 한 번의 generic forward pass로 산출(사전 예측)하고, trigger는 rollout 시점에서 행동별로 측정해, 두 값을 같은 signed 제어 가지(branch)에서 비교하는 방식으로 window 법칙을 검증합니다.

- **Empirical Impact**: 15개 held-out 뉴런에서 예측한 ceiling은 평균 절대 오차 0.14로 맞았고(대부분 layer에서는 약 0.07), majority baseline 대비 open/closed verdict가 11/15에서 유지됐습니다. 특히 refusal에서는 성공이 scalar(단일 크기 성공률)가 아니라 typed(예: coherent bypass vs strict-actionable reach 분리)로 나타나며, 실제 actionable reach는 일부 Llama pivot에서 더 긴 rollout horizon에서만 관측되는 등 안전 측면의 ‘제어 가능성’ 감사를 용량 기반으로 재정의했다는 의미가 큽니다.



### CREDENCE: Claim Reduction for Decomposition & Enhanced Credibility -- Semantic Metrics and Convergence Analysis (https://arxiv.org/abs/2606.19819)
Comments:
          40 pages, 6 figures, 19 tables. Submitted to Language Resources and Evaluation

- **Prior Approaches**: 기존 자동 사실검증 파이프라인은 복합 문장을 원자(atomic) 청구로 분해한 뒤 검증하는 접근이 주류였지만, 분해 품질 평가는 주로 token-overlap 기반 Jaccard(Soft-F1Jac) 같은 지표에 의존했다. 이 방식은 “X is big”과 “X is large”처럼 의미는 같은 paraphrase를 낮게 점수화해, 사람 판단과 자동 점수 간 괴리를 만들었다. 또한 rule 기반 repair와 LLM self-repair를 반복 적용하는 루프는 제안됐지만, 종료(termination)와 개선이 단조(monotone)로 보장되는지에 대한 형식적 분석이 부족했다.

- **Core Contribution**: 이 논문은 Credence라는 ‘청구 분해 및 평가 프레임워크’로 (1) 패러프레이즈를 과도하게 벌점 주던 평가 취약점과 (2) repair 루프의 종료/단조성 문제를 동시에 다룬다. 핵심은 Semantic-F1(의미 유사도 기반 재정의)와, rule repair 및 LLM self-repair 각각에 대한 수렴/비수렴 성질을 정리한 verified repair 설계에 있다. 더불어 도메인 간 일반화를 측정하는 3종 벤치마크와, 여러 decomposer를 함께 비교하는 다중 모델 평가 체계를 제시한다.

- **Technical Challenges**: Semantic-F1을 설계할 때는 토큰 중첩이 아니라 문장 의미를 반영하면서도, decomposer가 만든 원자 단위의 구조적 불일치에 강인한 매칭(average-max pooling)을 구현해야 했다. 이를 위해 BGE-large cosine similarity를 사용해 Jaccard식 벌점 문제를 해결했고, precision/recall 모두 greedy max 유사도 정렬을 채택해 Hungarian 1-to-1 강제의 과도한 불이익을 피했다. 또 repair의 종료성을 보이려면 디펜던시 파싱의 “분해 경계”를 상태로 삼아 rule repair는 단조 감소 및 유한 종료를 증명하고, LLM self-repair는 비단조(non-monotone) 반례와 함께 early-exit guard 같은 안전장치가 필요함을 정리했다.

- **Empirical Impact**: 실험에서 Semantic-F1은 Jaccard-F1 대비 +15~32pp(평균 +25pp)까지 향상되어, 패러프레이즈 정렬이 실제 검증 성능에 유리하게 반영됨을 보여줬다. 정량적으로는 SocialClaimSplit과 WikiSplitBench에서 EPR이 0.94~1.00 수준이며, 더 어려운 뉴스 도메인인 ClaimDecompBench에서는 베이스 EPR이 0.824까지 내려가도 평가 일관성을 유지했다. 또한 rule-repair는 Atomicity Violation Rate(AVR)를 베이스 대비 47~100% 줄이면서 fidelity를 떨어뜨리지 않아, “안정적 원자성 보정”이 실증적으로 확인됐다.



### Clusters are All You Need: Pre-Training the Tsetlin Machine with Semantic Clusters from Language Models for Interpretability (https://arxiv.org/abs/2606.19815)
- **Prior Approaches**: BERT 같은 사전학습 언어모델은 성능이 뛰어나지만 의사결정 과정이 불투명해 고위험 분야 적용이 어렵다. Tsetlin Machine(T M)은 조항 기반 규칙으로 설명 가능하지만, Boolean BOW 입력 때문에 BERT가 가진 문맥 의미를 충분히 활용하지 못해 성능 격차가 생긴다. 기존의 GloVe/Word2Vec 등 정적 임베딩 결합 시도는 문맥 의미를 놓쳐 한계를 지속했다.

- **Core Contribution**: 본 논문은 임베딩을 직접 사용하지 않고, BERT(또는 Top2Vec)로부터 얻은 의미적 클러스터 지식을 TM으로 “전이”하는 사전학습 프레임워크를 제안한다. 비지도 샘플을 의미적으로 일관된 클러스터로 묶고, 클러스터-샘플 페어로 Non-Negated Tsetlin Machine(NTM)을 사전학습해 해석 가능한 의미 키워드를 학습한다. 이후 다운스트림 학습에서는 이 키워드를 의미 Bag-of-Words로 보강해 기존 TM을 fine-tuning한다.

- **Technical Challenges**: 핵심 과제는 TM이 다루는 Boolean 입력 구조 안에서 문맥적 의미를 넣으면서도 규칙의 해석 가능성을 유지하는 것이다. 이를 위해 NTM에서는 negated literals를 비활성화하고, Type I feedback(positive reinforcement)을 강화하는 등 학습 신호를 재설계해 클러스터별 “상대적으로 신뢰도 높은” 단어/구문이 규칙에 선택되도록 유도한다. 클러스터 품질과 어휘 잡음의 영향을 줄이기 위해 TA 상태 기반 confidence를 활용해 고신뢰 키워드 개수를 조절한다.

- **Empirical Impact**: 5개 토픽 분류 데이터셋에서 제안 방법은 vanilla TM과 embedding 기반 TM을 일관되게 능가하며, BERT-large와도 성능 격차를 1~2% 수준으로 좁히는 경쟁력을 보인다. 특히 BERT-derived 클러스터로 사전학습한 TM은 Top2Vec 키워드 기반 사전학습보다 대체로 1~2% 더 나은 결과를 보였다. 계산적으로는 도메인당 오프라인 클러스터링·사전학습 비용이 추가되지만, 최종 모델은 규칙 기반이어서 고위험 분야의 설명가능성과 성능을 동시에 노릴 수 있는 방향성을 제시한다.



### Beyond Uniform Forgetting: A Study of Sequential Direct Preference Optimization Across Preference Settings (https://arxiv.org/abs/2606.19744)
Comments:
          Submitted to EMNLP 2026

- **Prior Approaches**: LLM 선호 정렬은 RLHF나 DPO 같은 offline 대조 학습으로 여러 행동 목표(도움됨/해로움/안전/정직 등)를 단계적으로 다루지만, 새 목표를 추가할 때 이전 목표가 얼마나 보존되는지는 불명확했습니다. 기존 연구는 대체로 특정 포스트트레이닝에서의 평균 성능 하락(혹은 이전 지식의 망각/전이)을 보고해 왔고, preference pair처럼 열린 응답 비교에서는 변화가 ‘상대 선호 마진’ 단위로 분산될 수 있어 분석이 거칠다는 한계가 있었습니다. 또한 망각을 원인별(직접 gradient conflict vs 분포 드리프트/신호 불균형/파라미터 이동)로 설명하려는 실증 근거가 제한적이었습니다.

- **Core Contribution**: 이 논문은 sequential DPO(Direct Preference Optimisation)를 두 단계로 나눠 학습할 때, 후속 목표 학습이 이전 목표의 선호를 균일하게 망각하는지 또는 목표 간 관계에 따라 안정/재분배/긍정적 전이를 보이는지 체계적으로 확인합니다. HH-RLHF, HelpSteer2, PKU-SafeRLHF, UltraFeedback의 네 가지 preference 설정(분포 갈등, 다속성 상호작용, 강한 안전 신호, 호환되는 응답 품질)을 두 순서로 비교해, “한 가지 forgetting 패턴”이 성립하지 않음을 보입니다. 더 나아가 aggregate 지표가 가리는 pair-level 이질성을 분해하는 분석틀을 제안합니다.

- **Technical Challenges**: 핵심 난제는 이전 단계에서 학습된 선호가 단계 2에서 어떻게, 그리고 얼마나 ‘부분적으로’ 변하는지 정밀 측정하는 것입니다. 이를 위해 각 stage 이후 모든 목표를 고정 base-model 기준으로 평가해 DPO-relative reward margin과 relative preference accuracy를 일관된 척도로 비교하고, preference pair마다 길이 정규화 policy margin 변화를 추적한 뒤 quartile decomposition으로 변화를 구간별로 분해합니다. 원인 규명에서는 direct gradient opposition 가설을 gradient cosine similarity와 LoRA 어댑터 이동(파라미터 공간 방향성) 진단으로 직접 점검하지만, 두 진단 모두에서 stage2 업데이트가 이전 목표에 대해 거의 직교에 가깝다는 결과를 제시합니다.

- **Empirical Impact**: 실험 결과 sequential DPO는 이전 목표에 대해 일관된 catastrophic forgetting을 만들지 않고, 목표 관계와 신호 강도, 학습 순서에 따라 부분 열화, 안정, pair-level 재분배, 긍정적 전이가 폭넓게 나타났습니다(예: PKU-SafeRLHF에서는 높은 안전 신호가 다른 단계 후에도 선호를 잘 유지). 특히 pair-level/ quartile 분석은 “평균은 비슷해 보이지만 일부 구간의 confidence 높은 pair가 크게 망각되거나 오히려 강화될 수 있음”을 보여 주어, aggregate 지표만으로는 정책 변화의 구조를 놓칠 수 있음을 강조합니다. 기계적 진단에서는 stage2 gradient와 LoRA 업데이트가 이전 목표와 near-orthogonal하여 direct gradient conflict가 주된 동인이라는 증거가 약하다는 점을 제시함으로써, 향후 sequential alignment 파이프라인 설계 시 objective compatibility와 signal strength를 고려해야 한다는 실무적 함의를 제공합니다.



### NRITYAM: Language Models Meet Art and Heritage of Danc (https://arxiv.org/abs/2606.19727)
Comments:
          18 pages, 12 figures, in ECML_PKDD'26

- **Prior Approaches**: 기존 문화 VQA/문화 추론 벤치마크는 단일 언어 중심이거나(영어 편중), 춤보다는 모션 인식 같은 다른 작업에 초점이 맞춰지는 경우가 많았습니다. 또한 일부는 지역 범위가 좁거나(예: 동남아) 문화 주제가 넓더라도 ‘전통 춤’의 언어·지역 맥락을 깊게 다루는 대규모 QA 세트는 부족했습니다. 그 결과 모델이 지역 특수성을 이해하는지 정밀하게 측정하기 어려웠습니다.

- **Core Contribution**: NRITYAM은 전통 춤의 문화적 이해 능력을 평가하기 위한 다국어·다문화 벤치마크로, 12개 언어에서 총 9,260개의 QA를 제공합니다. 텍스트와 이미지 두 모달리티를 모두 다루며, 질문은 역사 기반·규칙 기반·시나리오 기반으로 체계화되어 전통 공연의 맥락 추론을 시험합니다. 데이터는 현지 무용가·원어민과의 협업으로 지역에 맞는 질문을 직접 작성·검증해 문화적 근거를 강화했습니다.

- **Technical Challenges**: 전통 춤 지식은 언어(용어, 서술 방식)와 시각 단서(의상·동작·의례 맥락)가 결합돼 있어, 단순 번역이나 일반 상식만으로는 정답 판단이 어렵습니다. 저자들은 각 질문이 국가별 전통 춤과 정합적인지 먼저 걸러내고, 36명의 국가별 전문가가 QA를 제작·번역한 뒤 교차검증과 편향/민감도 리뷰로 품질을 통제했습니다. 또한 평가를 zero-shot(온도 0, MCQ 정답 후보 중 확률 최대)로 통일해 모델 비교의 공정성을 확보했습니다.

- **Empirical Impact**: 실험에서 frontier LLM은 대체로 소형 모델보다 우수했지만, 전 언어에서 고르게 잘하진 못했으며 특히 저자원 언어(예: Māori, Amharic, Arabic)에서 성능 저하가 두드러졌습니다. 멀티모달 모델은 시나리오 추론에서 강세를 보였으나, 역사·규칙처럼 문화적으로 고정된 근거를 요구하는 범주에서는 여전히 병목이 관측됐습니다. 결과적으로 NRITYAM은 전통 예술 맥락을 ‘문화적으로 민감하게’ 이해하는 능력을 계량화하며, 향후 데이터 보강과 저자원 문화 지식 전이 연구의 기준선 역할을 할 것으로 기대됩니다.



### FineREX: Fine-Tuned NER-RE for Human Smuggling Knowledge Graphs (https://arxiv.org/abs/2606.19710)
Comments:
          Code available at this https URL

- **Prior Approaches**: 인신밀수(human smuggling) 수사에선 법원 기록에서 NER-RE를 뽑아 지식그래프(KG)로 만드는 다단계 파이프라인이 주로 쓰였다. 대표적으로 LINK-KG는 coreference resolution을 위해 문서 리라이팅을 수행한 뒤, 다시 NER-RE/GraphRAG 추출을 반복해 최종 그래프를 만든다. 하지만 일반-purpose LLM에 의존해 도메인별 개체·관계 스키마를 정확히 맞추기 어렵고, 추가 추론 단계가 잡음(hallucination)·중복 추출·비용을 키우는 문제가 있었다.

- **Core Contribution**: FineREX는 법원 문서용 NER-RE를 중심으로, fine-tuned LLaMA 3.1 8B 기반 단일 추출 패스와 coreference 매핑을 그래프 통합에 직접 연결하는 “스트림라인” 파이프라인을 제안한다. 문서 리라이팅과 중복 NER-RE 패스 없이, 1차 추출 결과의 coreference cache를 곧바로 노드 통합에 사용한다. 또한 인신밀수 도메인에 맞춘 NER-RE 스키마를 정의하고, 이를 학습시키기 위한 수작업 주석 데이터(512개 청크)를 구축해 성능 격차를 만든다.

- **Technical Challenges**: 핵심 난관은 도메인 용어와 스키마가 강하게 제약된 법률 문맥에서, 잡음 정보까지 섞지 않으면서 개체/관계를 정확히 추출하는 것이다. FineREX는 delimiter 포맷과 타깃 엔터티 정의를 포함한 LINK-KG 계열 system prompt를 고정하고, QLoRA로 LLaMA 3.1 8B를 인신밀수 NER-RE에 파라미터-efficient fine-tuning해 recall과 관계 강도(score) 예측까지 같이 끌어올린다. 이후 coreference 단계에서 별칭을 canonical 이름으로 정규화하고, 그래프 통합 시 다중 청크에서의 type은 majority vote로 안정화하며 관계 강도는 평균(중복 설명 제거)으로 집계한다.

- **Empirical Impact**: 실험에서 FineREX는 512개 데이터 기반 NER-RE에서 entity F1과 relationship F1을 각각 절대 15.50%, 31.46% 개선했으며, 대형 일반 목적 모델(LLaMA 3.3 70B) 대비도 파라미터가 훨씬 적은 8B로 더 높은 추출 품질을 보였다. 또한 16개 DOJ 인신밀수 사건 문서 적용 결과, 법적 잡음 비율을 거의 절반(약 50%에 근접) 줄이고, 장문에서 노드 중복률을 17.78%에서 11.17%로 낮췄다. 더불어 리라이팅과 불필요한 재추출을 제거해 end-to-end 처리 시간을 50.0% 단축해, KG 품질과 효율을 동시에 개선하는 도메인 fine-tuning의 실용적 이점을 입증했다.



### TerraMARS: A Domain-Adapted Small-Language-Model Pipeline for Mars Terraforming Literatur (https://arxiv.org/abs/2606.19700)
Comments:
          16 pages, 1 figure, 4 tables

- **Prior Approaches**: 기존 연구는 Mars terraforming 아이디어를 정성·정량으로 제시해 왔지만, 최신 논문에서 제약 조건을 찾아 사람이 직접 정리해야 하는 병목이 컸습니다. 한편 화학·소재·생명과학 등 다른 분야에서는 BERT/SciBERT 같은 과학용 언어모델과 텍스트-구조화 추출이 시도됐지만, JSON 같은 구조출력에서 LLM의 hallucination 문제가 반복돼 왔습니다.

- **Core Contribution**: 이 논문은 Mars 관련 과학 문헌에서 질문 답변과 정량 제약을 동시에 뽑아내는 end-to-end 파이프라인 TerraMARS를 제안합니다. domain-adapted Small Language Model을 Gemma 3 1B로 구축하고, 회수(Retrieval)+청킹+합성데이터+QLoRA fine-tuning을 통해 비정형 초록을 JSON으로 변환할 수 있게 합니다.

- **Technical Challenges**: 핵심 난제는 (1) 분야별로 맞는 추출을 하되 (2) JSON 스키마를 지키고 (3) 작은 모델에서 사실 일치(factual consistency)를 유지하는 것입니다. 저자들은 Llama 3.2 3B를 teacher로 knowledge distillation해 6개 태스크용 instruction-output을 합성하고, JSON 추출은 필수 5필드(parameter/value/unit/condition/terraforming stage) 검증에 통과한 샘플만 학습에 사용했으며, Gemma 3 1B는 QLoRA(4-bit NF4)로 경량화해 GPU 부담을 줄였습니다.

- **Empirical Impact**: arXiv/PMC/Semantic Scholar에서 초록을 수집해 총 393개를 정리했고, 카테고리 분포가 general에 치우친 점이 성능 제한으로 지적됩니다. 합성데이터는 1179개 고품질 예제가 최종적으로 생성되었고 모델의 산출은 템플릿-문헌 부합 시에는 소스에 근거한 요약/제약 추출이 잘 되지만, 템플릿 도메인 불일치나 스키마는 맞아도 값 grounding이 약한 경우가 관찰돼 추가 개선 여지가 큽니다.



### What sentiment analysis can't see: Measuring whether customers were helped, and what went wrong, across 70,000 support conversations (https://arxiv.org/abs/2606.19698)
Comments:
          25 pages, 6 figures

- **Prior Approaches**: 기존 고객지원 데이터 분석은 주로 sentiment analysis(감성분석)로, 고객이 얼마나 ‘좋게/나쁘게’ 말했는지에 초점을 둡니다. 하지만 이는 실제로 결과에 만족했는지(만족 상태)나 구체적 문제가 있었는지를 직접 반영하기 어렵다는 한계가 있습니다. 또한 대시보드는 흔히 단일 라벨(예: Neutral)이 복잡한 고객 경험을 한 덩어리로 뭉개는 문제를 안고 있습니다.

- **Core Contribution**: 이 논문은 같은 대화 텍스트에서 고객의 만족도 추정과 “구체적 문제 보고” 여부를 구조화해 읽는 방식을 제안합니다. GPT-5.4를 활용해 톤(tone) 외에 만족 상태(satisfaction estimate)를 추정하고, 실제로 문제를 명시했는지도 함께 플래그로 뽑아냈습니다. 그런 다음 고객이 남긴 1~5점 평점을 기준으로 세 가지 읽기를 모두 검증해, 감성분석 중심 접근의 약점을 정면으로 비교합니다.

- **Technical Challenges**: 핵심 과제는 LLM이 텍스트의 톤과 실제 만족도를 얼마나 신뢰도 있게 연결해 ‘만족 상태’라는 메트릭으로 환산하느냐였습니다. 논문은 GPT-5.4로 만족도와 문제 보고 여부를 동시에 추정한 뒤, 고객이 실제로 매긴 1~5점 평점과의 일치도를 직접 검증하는 방식으로 해결했습니다. 더 나아가 톤과 만족이 불일치하는 대화(예: 겉으론 중립이지만 실제로는 불만족)까지 구조화 결과로 분해해 드러나게 했습니다.

- **Empirical Impact**: 70,450개의 지원 대화를 실험한 결과, 만족도 추정은 감성분석보다 고객 평점과 더 잘 맞아 상관계수에서 0.47 대 0.36을 보였습니다. 또한 불행한(unsatisfied) 고객을 플래그하는 데 있어 오탐(false alarm)을 크게 줄였고, 톤과 만족이 서로 어긋나는 대화가 44%나 됨을 확인했습니다. 더 중요한 점은 ‘tolerated friction(만족하지만 고칠 수 있는 문제를 보고)’처럼 감성 기반 대시보드가 포착하지 못하던 최대 규모의 집단까지 구조적으로 드러내, 고객 상태 중심의 새로운 비즈니스 지표 가능성을 보여줬다는 것입니다.



### Code-Switching Reveals Language Anchoring in Multilingual LLMs (https://arxiv.org/abs/2606.19668)
Comments:
          36 pages, 13 figures, 27 tables

- **Prior Approaches**: 기존 Code-Switched (CS) 연구는 강건성, 번역, 생성 등 결과 중심 평가에 치우쳐 CS가 성능을 떨어뜨리는 내부 원리를 구체적으로 설명하지 못했다. 또한 MLLM 분석은 언어 선호나 English-centered 처리 경로 같은 경향을 보이지만, CS 표현이 source/target 단일언어 표현 공간에서 어디에 놓이는지(기하학적 위치)는 거의 다뤄지지 않았다. 즉, 동일한 정답 실패가 서로 다른 내부 표현 원인에서 비롯될 수 있는데 이를 분리·진단하기 어려웠다.

- **Core Contribution**: 논문은 grammar-forced CS를 통제 진단 도구로 써서 CS hidden state가 source 앵커에 가까운지 target 앵커에 가까운지 측정하는 Anchor Bias를 제안한다. 다양한 MLLM에서 공통적으로 “source-framed CS는 source-anchored 유지, target-framed CS는 target-ward로 이동”하며 그때 QA 성능 저하가 더 커진다는 일관된 패턴을 확인했다. 이를 바탕으로 CS 추론 실패를 완화할 실마리로 ‘내부 앵커링 신호’를 제시한다.

- **Technical Challenges**: 핵심 난제는 CS 성능 저하가 단순한 단어 혼합 때문인지, 문법 프레임이 표현 공간에서 앵커를 어떻게 바꾸는지 분해해 측정하는 것이었다. 논문은 matched source/target 쌍과 grammar-forced CS( GF-SRC, GF-TGT )를 고정된 정보 질의 위에 구성해, cosine 기반 유사도 차이를 정규화까지 포함해 layer-wise로 Anchor Bias를 계산했다. 그다음 추론 단계에서 upper layers(후반 30%)만을 대상으로 CANVAS(Contextual Anchor-based Neural Vector Alignment Steering)를 적용해, in-context에서 추출한 source canvas를 기준으로 target-language hidden states를 부드럽게 source 쪽으로 steering한다.

- **Empirical Impact**: 실험에서 Anchor Bias의 프레임 의존적 이동은 모델 전반의 층(upper layers 포함)에서 반복 관찰되며, CS의 target-ward 치우침이 클수록 F1 저하가 커지는 상관(유의한 상관)을 보였다. CANVAS는 파라미터 업데이트 없이도 여러 MLLM/CS 조건에서 QA F1을 일관되게 회복했는데, 특히 target-framed CS에서 개선 폭이 가장 컸다. 이는 CS 입력의 내부 ‘target-ward anchoring 실패’를 source-side 정렬로 교정할 수 있음을 보여주며, CS 대응을 결과 수준뿐 아니라 표현 정렬 관점에서 설계 가능하게 만든다는 점에서 의미가 크다.



### CacheWeaver: Cache-Aware Evidence Ordering for Efficient Grounded RAG Inferenc (https://arxiv.org/abs/2606.19667)
- **Prior Approaches**: RAG는 검색 문서를 프롬프트에 넣어 사실성을 높이지만, 그만큼 prefill(입력 처리) 지연이 커져 TTFT가 병목이 되기 쉽습니다. vLLM의 Automatic Prefix Caching(APC)는 토큰 prefix가 같을 때 KV 캐시를 재사용하지만, RAG에서는 문서 청크가 겹쳐도 “순서”가 달라지면 prefix 일치가 깨져 재사용 이점이 급격히 줄어듭니다.

- **Core Contribution**: 이 논문은 서빙 엔진을 바꾸지 않고도 검색된 증거(evidence)의 “순서”만 재배치해 캐시 친화적인 prefix를 만들자는 CacheWeaver를 제안합니다. 프롬프트 레이어에서 최근에 서빙된 문서 시퀀스를 trie(지식 트리)로 기억하고, 그 트리에 기반해 가장 재사용 가능성이 큰 prefix를 먼저 오도록 greedy walk으로 정렬합니다. 이렇게 하면 retrieval top-kk의 증거 집합은 유지한 채 prefill 비용만 줄이도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 “토큰 prefix”는 캐시에 민감한데, retrieval의 단위는 문서 청크라서 겹친 증거라도 순서가 바뀌면 prefix 재사용이 성립하지 않는다는 점입니다. CacheWeaver는 최근 서빙 시퀀스의 접두(prefix) 재사용 깊이를 최대화하는 목표를 세우고, 런타임에 복잡한 조합 탐색 없이 trie에서 캐시 확장 경로를 따라가되, 확장이 불가능하면 나머지는 원래 retrieval 순서를 따르는 안전한 폴백을 둡니다. 또한 실제 캐시 잔존 상태가 완전히 동일하지 않을 수 있어, 필요 시 TTFT 기반 피드백으로 “stale” 경로 의존을 줄이는 옵션도 제공합니다.

- **Empirical Impact**: vLLM의 세 가지 설정에서 CacheWeaver는 retrieval-order 기반 APC 대비 중앙값 TTFT를 약 20–33% 낮췄고, QA 품질은 제한된 체크 범위에서 저하 없이 유지됐습니다. 특히 greedy 정책은 oracle ordering이 주는 TTFT 감소의 97.5% 수준을 회수해, 엔진 변경 없이도 상당 부분 locality를 되살릴 수 있음을 보였습니다. 공용 데이터 검증에서는 HotpotQA 같은 고정 슬라이스가 인접 재사용이 거의 없어 중앙 TTFT 개선이 없고, TriviaQA/NQ-Open 등에서는 일정 구간에서만 이득이 관측돼 “인접한 증거가 실제로 겹치는 세션형 locality”에서 효과가 발현된다는 경계 조건을 명확히 했습니다.



### SAGE-OPD: Selective Agent-Guided Intervention for Multi-Turn On-Policy Distillation (https://arxiv.org/abs/2606.19659)
Comments:
          21 pages, 3 figures

- **Prior Approaches**: 기존 on-policy distillation(OPD)은 학생이 생성한 on-policy 궤적 위에서 교사가 token-level로 지도하여 exposure bias를 줄이는 데 강점이 있지만, 대부분은 single-turn에 초점을 맞췄습니다. multi-turn 환경에서는 초기에 난 실수가 이후 관측을 바꿔 오류가 누적(compounding)되고, dense token-level OPD는 의미적으로 가능한 대안을 과도하게 불리하게 만들거나 반복/퇴행 같은 로컬 degeneracy를 강화하며, off-distribution 이력이 커질수록 교사 신호가 덜 신뢰로워집니다. 그 결과 multi-turn에서 균일한 OPD 적용은 brittle해질 수 있다는 분석이 제기됐습니다.

- **Core Contribution**: 이 논문은 verifier-free 형태로 multi-turn OPD를 선택적으로 수행하는 SAGE-OPD를 제안합니다. 각 turn마다 환경 피드백을 확인한 뒤, 교사의 판단으로 해당 학생 응답을 skip할지, 약하게 개입(weak intervention)할지, 강하게 개입(strong intervention)할지를 결정해 불필요한 dense supervision을 줄입니다. 또한 교사 confidence로 token-level distillation 신뢰도를 보정하고, loss normalization으로 표준 OPD의 전체 손실 스케일을 보존하면서 turn-level 선택 가중치의 효과만 유지합니다.

- **Technical Challenges**: 핵심 기술 난점은 (1) multi-turn에서 오류 누적 때문에 후반 turn의 교사 분포가 불안정해질 수 있고, (2) token-level 정렬이 의미적으로 허용되는 대안을 과벌점하거나 로컬 반복 패턴을 강화할 수 있으며, (3) 선택적 가중치를 넣으면 학습 신호 스케일이 함께 변해 효과를 해석하기 어려워진다는 점입니다. SAGE-OPD는 결정 변수(Skip/Weak/Strong)를 교사 프롬프트로 turn 단위에서 산출하고, 필요한 turn에 대해서만 token-level OPD를 가중 적용하며, 교사의 top-1 확률 기반 confidence로 불확실한 교사 신호의 영향을 줄입니다. 마지막으로 배치 내 loss normalization으로 평균 토큰 가중치를 표준 OPD와 맞춰 최적화 스케일 변화를 최소화합니다.

- **Empirical Impact**: 실험은 ALFWorld, ScienceWorld, SearchQA 등 multi-turn 에이전트 벤치마크에서 SAGE-OPD가 표준 OPD 및 오프폴리시 SFT보다 일관되게 향상됨을 보여줍니다. 특히 Qwen3 모델 세팅에서 ALFWorld unseen success rate는 표준 OPD 대비 최대 13.3% 상대 개선을 기록했으며, 반복 turn을 줄이면서도 성능 이득이 나타났습니다. ablation 결과는 turn-level intervention, teacher confidence weighting, loss normalization이 각각 보완적으로 기여하며, multi-turn OPD는 on-policy 이점을 유지하되 교사 감독은 필요하고 신뢰 가능한 turn에 선택적으로 배분해야 한다는 메시지를 강화합니다.



### From 50K to 8.2 Million in 24 Hours: Vozinha's Algorithmic Consecration and the Multilingual Making of World Cup Visibility (https://arxiv.org/abs/2606.19647)
Comments:
          11 pages, 4 figures, 3 tables; v0.1 pilot preprint. Dataset and evidence package available at this https URL

- **Prior Approaches**: 기존 연구는 담화 프레이밍을 언어 신호로 분석하거나, 바이럴 확산을 메트릭의 퍼짐 현상 중심으로 설명하는 경향이 강했습니다. 또한 교차언어 비교는 “같은 사건이 언어마다 다르게 이야기된다”는 점은 다뤘지만, 플랫폼 지표 자체를 ‘담화의 재료(linguistic object)’로 취급하는 방식은 상대적으로 드뭅니다.

- **Core Contribution**: 이 논문은 ‘Vozinha(카프베르데 골키퍼)의 월드컵 무득점(0-0) 무대가 어떻게 전 지구적으로 상징화됐는가’를 언어+플랫폼의 결합 문제로 재정의합니다. 특히 팔로워 수를 단순 측정치가 아니라, “50k to 8M”처럼 반복·인용·스크린샷화되는 공유 가능한 증거로 보고 프레이밍과 결합되는 메커니즘을 제안합니다.

- **Technical Challenges**: 플랫폼 지표는 연속 시계열을 확보하기 어렵고(대부분 추정치·범위), 시각 증거는 캡처 시점 인터페이스 아티팩트 가능성이 있어 엄밀한 인과 주장에 제약이 있습니다. 논문은 단 하나의 정확 앵커(8,235,652, 2026-06-16 15:47 UTC)만 ‘exact’으로 두고 나머지는 범위/임계값으로 제한했으며, LLM-assisted suggestion+인간 검증, 큐-기반 9개 내러티브 프레임 라벨링, 스크린샷은 SHA-256 해시와 수기 전사 로그로 무결성을 관리했습니다.

- **Empirical Impact**: 실험 결과로는 포르투갈어는 CazéTV·Casimiro를 매개한 ‘동원/집단행동’ 프레임, 스페인어는 스페인의 실패를 강조하는 ‘위기/굴욕’ 프레임, 영어는 언더독·국가 서사를 강화하는 프레임, 프랑스어는 ‘성취→네트워크 가시성’ 흐름을 포착하는 양상이 관찰됩니다. 그럼에도 공통의 연결 조직은 ‘팔로워 수 자체를 뉴스 헤드라인처럼 전면화(F4)’하는 플랫폼-메트릭 스펙터클이며, 이는 주변부 경기력이 다국어 담화와 숫자 증거를 통해 세계적으로 읽히는 과정을 설명하는 데 의미가 있습니다.



### Creating Multilingual Mental Health Dialogue Datasets: Limits of Persona-Based Localization via Nationality and Languag (https://arxiv.org/abs/2606.19640)
Comments:
          15 pages, 4 figures. Accepted to the 2026 Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2026), co-located with ACL 2026

- **Prior Approaches**: 기존 연구는 우울증 검출·스크리닝을 위해 영어 중심 데이터셋을 보강하거나, 합성 데이터에서 persona를 활용해 학습/평가 규모를 키우는 흐름이 강했다. 특히 영어 임상 persona를 기반으로 BDI-II 같은 척도를 넣어 대화형 환자(standardized patient)를 만들고, LLM을 judge로 써 심각도를 비교하는 방식이 자주 쓰인다. 다만 대부분이 영어/서구 맥락에 고정돼 있어, 언어가 바뀌면 동일한 임상 신호가 유지되는지에 대한 검증은 부족했다.

- **Core Contribution**: 이 논문은 영어로 검증된 임상 persona에서 nationality와 Language Use만 바꿔 만다린, 벵골어, 힌디어 대화로 확장할 때 우울 심각도 신호가 보존되는지 통제 실험으로 확인한다. 또한 생성된 다국어 대화를 대상으로 여러 LLM judge들이 우울 심각도를 블라인드(pairwise)로 비교할 때의 정확도와 불확실성(tie)을 함께 분석한다. 핵심 결론은, 단순 파라미터 치환만으로는 언어 간 임상 일관성을 안정적으로 재현하기 어렵다는 점이다.

- **Technical Challenges**: 기여를 실현하려면 (1) 영어 persona의 BDI-II 기반 증상 표현을 다른 언어에서도 동일한 임상 강도로 유지하면서, (2) judge가 비영어 텍스트에서 같은 신호를 일관되게 해석하도록 평가 파이프라인을 설계해야 한다. 연구진은 영어 baseline persona를 고정하고 nationality-언어 쌍(미국-영어, 중국-만다린, 방글라데시-벵골어, 인도-힌디어)만 변경해 48개 persona를 만들었고, 각 persona당 다단계 증상 인터뷰 대화를 생성한 뒤 12개 persona 간 66회 pairwise 비교로 심각도 순서를 회복하는지 측정했다. 평가지표는 overall accuracy뿐 아니라 same-level error rate(같은 심각도 범위 오차)와 tie distance(불확실성의 심각도 간격)까지 포함해 ‘오차의 구조’가 언어에 따라 어떻게 바뀌는지 드러냈다.

- **Empirical Impact**: 실험 결과, 모든 judge 모델에서 영어 성능이 상한으로 유지됐고 비영어(특히 벵골어·힌디어)로 갈수록 정확도 하락과 cross-severity error 증가, tie 빈도 및 tie distance 확대가 나타났다. 일부 모델은 영어에서만 심각도 보정(calibration)이 안정적이었고, 더 작은 8B 계열은 비영어에서 변동이 커 BDI-II 수준 신호가 약해진 정황을 보였다. 저자들은 따라서 영어-centric persona를 그대로 ‘동등한 샘플’로 간주하기보다, 언어·문화에 맞춘 culturally responsive 데이터 생성과 출력 수준 검증이 필요한 임상 아티팩트로 취급해야 한다고 강조한다.



### MiqraBERT: Regression-Based Sentence-BERT Finetuning for Biblical Hebrew Parallel Detection (https://arxiv.org/abs/2606.19638)
- **Prior Approaches**: 기존의 성경 구절 재사용 탐지는 대부분 어휘 중첩(lexical overlap) 기반이라, 패러프레이즈나 어휘 치환, 구문 재구성 같은 변형이 들어가면 성능이 급격히 흔들립니다. 특히 평행(parallel) 관계가 의미 단위로 유지되더라도 표면 형태가 달라지면 탐지 신뢰도가 떨어지는 한계가 컸습니다.

- **Core Contribution**: 이 논문은 히브리 성경 구절의 의미 유사도를 위해 AlephBERT를 바탕으로 Sentence-BERT 계열인 MiqraBERT를 verse-level 의미 임베딩 모델로 파인튜닝합니다. 1,650개의 labeled verse 및 half-verse pair(진짜 병렬 825 vs 랜덤 네거티브 825)를 구성해, 병렬 구절은 가깝게, 비관련 구절은 멀어지도록 코사인 유사도 회귀로 학습합니다.

- **Technical Challenges**: 핵심 난제는 의미적으로는 평행인데도 표면 표현은 달라지는 경우를 임베딩 공간에서 안정적으로 분리하는 것입니다. 논문은 cosine-similarity regression으로 임베딩을 학습하고, 분리 정도를 Wasserstein distance와 overlap coefficient 같은 분포 기반 지표로 검증하며, 10개 random seed로 재현성을 함께 확인합니다.

- **Empirical Impact**: MiqraBERT는 사전학습 baseline 대비 분포적 분리(distributional separation)를 2.7배 향상시키고, 모호한 overlap 구간을 약 24%에서 약 6%로 줄였습니다. 다만 genre 의존성이 뚜렷해 narrative synoptic parallels는 recall@10 87.1%로 강하지만, poetic parallels는 9% 미만으로 여전히 어렵다고 보고해 신뢰 가능한 범위를 narrative textual reuse 쪽으로 제한합니다.



### Before the Labels: How Dataset Construction Shapes Suicidality Detection in Clinical Tex (https://arxiv.org/abs/2606.19637)
Comments:
          To appear in the Proceedings of the 11th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2026)

- **Prior Approaches**: 임상 NLP는 자살 위험 탐지를 고위험 과제로 보고, 기존에는 소셜미디어보다 EHR을 더 신뢰할 만한 ground truth로 간주해 왔습니다. 그러나 실제 예측 성능이 다양한 임상 환경에서 기대보다 미미해 데이터 라벨이 어떤 가정을 내장하는지 점검이 필요하다는 문제의식이 커졌습니다.

- **Core Contribution**: 이 논문은 EHR 기반 자살징후(suicidality) 데이터셋의 라벨을 ‘중립적 진실’이 아니라 특정 operationalization(문서화 매개·에피소드 단위·의도(intent) 해결)으로 해석해야 한다고 제안합니다. 특히 ScAN(Scandinavic? ScAN) 사례를 통해 같은 라벨이 문서의 시간성/부정/불확실성에 따라 서로 다른 임상 프레이밍을 포괄할 수 있음을 보여줍니다. 

- **Technical Challenges**: 핵심은 라벨이 어떻게 생성되는지(거버넌스·코호트 정의·어노테이션·병원 체류 단위 집계·불확실성 처리)가 언어적 패턴으로 어떻게 ‘평탄화’되는지 구조적으로 추적하는 것입니다. 논문은 ScAN에서 ICD 기반 코호트 시딩, 단일 어노테이터 라벨링(불일치 시 physician review로 수정), “unsure”를 downstream에서 “negative”와 합치는 방식이 라벨 신호를 바꾼다는 점을 프레이밍 분석으로 확인합니다.

- **Empirical Impact**: 언어 분석 결과, 동일 라벨이 temporality/negation/uncertainty 프로필이 다른 스팬을 흡수하며 특히 “unsure”·과거표지(historical)·불확실성 신호가 라벨 공간에서 구분되지 않을 수 있음을 정량적으로 보였습니다. 또한 discharge summary와 HPI처럼 문서 섹션에 따라 동일 라벨이라도 unmodified 비율이 크게 달라져, 모델 성능·캘리브레이션이 숨은 데이터 가정에 의해 체계적으로 달라질 수 있음을 시사합니다.



### Where Does Social Reasoning Come From? Capability Provenance in Language Models (https://arxiv.org/abs/2606.19625)
Comments:
          Under review at COLM 2026 (Conference)

- **Prior Approaches**: 기존 training-data attribution은 문서 수준 영향도를 통해 “어떤 데이터가 중요해 보이는지”를 보여주지만, 문서 단위로는 사회 추론 vs STEM 추론처럼 서로 다른 능력이 어떤 말뭉치 ‘구역’에서 기인하는지 분해하기 어렵다는 한계가 있다. 또한 영향도 기반 연구가 사실성 지식(factual knowledge)에 치우치는 경향이 있어, 추론(reasoning) 능력의 근거가 되는 데이터 영역을 더 정밀하게 찾기 힘들었다. 다른 접근은 top-k 문서를 뽑아 사후 해석하거나, unlearning으로 검증하더라도 이전 연구는 주로 폐쇄형(비공개) 모델에서는 프록시 데이터 의존 문제가 있었다.

- **Core Contribution**: 이 논문은 gradient-based training-data attribution을 능력의 provenance(근거) 탐색에 활용하되, 문서 단위를 WebOrganizer의 24 format × 24 topic(총 576 bin)으로 집계해 ‘말뭉치 구역’ 단위의 해석 가능 지도를 만든다. 또한 SocialIQA(사회 추론)·ARC-Challenge(과학 추론)·MMLU Social Sciences(사회 지식)·MMLU STEM(수학/과학 지식)로 2×2 factorial 설계를 구성해, 도메인(social vs STEM)과 능력 유형(reasoning vs knowledge) 중 무엇이 corpus-level 차이를 더 잘 설명하는지 비교한다. 마지막으로 영향도가 높은 bin을 targeted machine unlearning으로 부분적 인과 검증해, 연관(associational) 신호를 넘어 선택적 손상 패턴을 확인한다.

- **Technical Challenges**: 핵심 난제는 (1) 문서 단위 영향도가 잡음이 커서 bin 수준으로 안정적인 신호를 만들기 어렵고, (2) Dolma3처럼 빈(bin)별 데이터량이 크게 치우친 말뭉치에서는 편향된 샘플링이 gradient 추정치를 흔든다는 점이다. 논문은 Bergson 기반 TrackStar로 gradient-based attribution을 계산한 뒤, bin 간 비교를 위해 de-duplicated Dolma3에서 각 bin을 균일하게 샘플링하는 stratified working set을 구성해 안정성을 확보한다. 이어서 benchmark 쿼리를 2×2로 고정하고 bin-level 영향 행렬을 만든 뒤, 대비(표준화 차이)와 bootstrap 재샘플링으로 판별 가능한 bin을 찾고, unlearning은 NGDiff의 gradient-normalized LoRA로 “단일 topic bin”과 “전역 대조”를 비교해 선택적 손상 효과를 추정한다.

- **Empirical Impact**: 결과적으로 OLMo3-7B에서 사회 추론(SocialIQA)은 문서 유형과 토픽 분포에서 STEM/지식 벤치마크와 질적으로 다른 구역에 강한 양(+)의 영향이 집중되며, 특히 Literature 및 Social Life 계열이 사회 추론에서 두드러지고 다른 벤치마크에서는 부정적 영향을 보이는 ‘부호 전환’이 관찰된다. 또한 reasoning에서는 사회/ STEM별 구역 대비가 knowledge 대비 더 선명하게 나타나, 능력 유형이 corpus 분해를 크게 좌우할 가능성을 시사한다. targeted machine unlearning에서도 높은 attribution bin(예: Literature 계열)을 잊도록 하면 해당 벤치마크 성능 저하가 within-bin 무작위 baseline보다 더 커져, 발견된 말뭉치 구역이 해당 능력에 부분적으로 필수적임을 뒷받침하며 관련 코드·샘플링 manifest·bin-level 영향 행렬·unlearning 체크포인트까지 공개한다.



### A BART-based approach with hierarchical strategy for Vietnamese abstractive multi-document summarization (https://arxiv.org/abs/2606.19591)
Comments:
          originally written in 2022

- **Prior Approaches**: 베트남어 multi-document abstractive summarization은 초기 연구가 부족하며, 기존에는 그래프 기반(semantic 관계를 잘 잡지만 discourse correlation 같은 추가 정보가 필요)과 계층형(hierarchical) 접근이 주로 쓰였다. 계층형은 문서별 중간 표현을 만든 뒤 이를 합쳐 최종 요약을 생성하지만, 중간 표현이 입력만으로 만들어져 최종 요약과 정보 불일치가 생기기 쉽고 그 결과 환각(hallucination)을 유도할 수 있다. 또한 end-to-end로 처리하기에는 입력 길이가 너무 길어 기존 pretrained seq-to-seq 모델의 토큰 한계가 문제가 된다.

- **Core Contribution**: 이 논문은 계층형 전략을 유지하되, 1단계에서 중간 표현을 만들 때 golden summary를 기준으로 문장을 선택해 두 단계 간 정보 상관을 높이는 간단한 샘플링 전략을 제안한다. 구체적으로 각 문장에 대해 golden summary와의 ROUGE1을 중요도로 계산해 원하는 길이 임계치까지 상위 문장을 고르고, 이를 2단계 요약 생성의 입력으로 사용한다. 추가로 VietnameseMDS 및 ViMs 등에 더해 Multinews를 베트남어로 번역해 약 5만 개 규모의 학습 클러스터를 커뮤니티에 공개한다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 장문/다문서 입력에서 정보 손실을 줄이면서 (2) 단계 1의 추출 과정이 단계 2 생성에 필요한 정보를 충분히 제공하도록 만드는 것이다. 저자들은 golden summary 기반 문장 선택으로 단계 간 정보 미스매치를 완화하고, 2단계에서는 문서 순서 변화에 견고하도록 입력 문장 순서를 permute하는 데이터 증강을 적용한다. 또한 BARTPho처럼 상대적으로 긴 입력을 다루는 seq-to-seq 모델을 1·2단계에 각각 fine-tuning해 토큰 제약을 실무적으로 흡수했다.

- **Empirical Impact**: VLSP 2022 공개 테스트셋에서 제안 방법은 ROUGE2-F1 0.2468을 기록해 organizer의 abstractive baseline 및 anchor baseline을 상회했다. 다만 extractive 및 rule-based 기준선보다는 낮아 절대적인 성능 격차는 남았으며, 그럼에도 사람 평가 관찰에서 문법적으로 자연스럽고 핵심 내용을 대체로 커버하는 요약을 생성한다고 보고한다. 한편 Multinews 베트남어 번역 데이터(추가 약 50,286 클러스터)를 공개함으로써 베트남 multi-document summarization의 학습 데이터 부족 문제를 실질적으로 줄이는 영향도 크다.



### LaViSA: A Language and Vision Structural Ambiguity Benchmark (https://arxiv.org/abs/2606.19552)
- **Prior Approaches**: 기존 Vision and Language Models(VLMs) 평가는 주로 단어 순서 변경 같은 compositionality(구성성) 문제를 다루거나, 모달리티에서의 일반적 ambiguity(애매성)를 폭넓게 다뤘다. 또한 시각 장면을 이용해 모호한 표현을 푸는 benchmark도 있었지만, 구조적 애매성의 유형을 체계적으로 분리해 평가하거나 “시각 증거가 의미 해석을 어떻게 바꿔야 하는지”를 정면으로 검증하긴 어려웠다. 특히 구조적 ambiguity는 한 문장에 다중 의미가 붙는다는 점에서, 단일 정답 의미를 맞히는 평가와 다른 난도가 존재한다.

- **Core Contribution**: 이 논문은 시각 장면을 통해 structural ambiguity(구조적 애매성)를 해소해 유일한 의미 해석을 고르는 능력을 측정하는 benchmark LaViSA를 제안한다. LaViSA는 모호한 문장-해소된 해석 문장들-해석에 대응하는 이미지로 구성되며, 7개 카테고리 총 1,503개 샘플을 제공한다. 또한 다양한 VLM(상용/오픈소스, 파라미터 규모 및 reasoning 역량 차이)을 LaViSA의 controlled multiple-choice 방식으로 비교 평가한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 구조적 애매성의 다양한 패턴을 통제된 형태로 만들고, (2) 이미지가 특정 해석과 정합적으로 대응하도록 데이터셋을 설계하는 데 있다. 논문은 TAB에서 가져온 카테고리를 7개로 재정의·확장하고, text-to-image 모델로 각 해석에 대응하는 이미지를 생성한 뒤 인간 검증을 통해 이미지-해석 매칭 품질을 확인(전반적으로 90% 이상 정확도)했다. 평가 또한 생성문을 참조문과 비교하는 방식(BERTScore 등) 대신 “이미지에 가장 잘 맞는 해석 선택”의 다지선다로 metric 의존성을 줄였다.

- **Empirical Impact**: 실험 결과, 최신 VLM들은 시각 장면을 활용해 구조적 애매성을 어느 정도는 풀지만, 카테고리별 편차가 크고 Conj(접속 범위)·Ellip(생략) 같은 유형에서 지속적으로 약점을 보였다. 또한 PT-Acc(시도 단위 정확도) 대비 PS-Acc(문장 단위에서 모든 해석을 일관되게 맞히는 정확도)의 갭이 관찰돼, “시각 증거 변화 → 의미 구조 변화”를 꾸준히 추적하지 못하는 경향이 드러났다. 이는 VLM이 시각적 세부를 보는 수준을 넘어 predicate–argument 관계 및 의미적 구조를 통합적으로 grounding하는 데 아직 한계가 있음을 시사하며, 향후 구조적 의미 통합 기법 개발에 직접적인 벤치마크 기준을 제공한다.



### Reliability without Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement, Consistency, and Bias (https://arxiv.org/abs/2606.19544)
- **Prior Approaches**: LLM-as-a-Judge는 MT-Bench, RewardBench, JudgeBench 같은 벤치마크에서 평가자가 모델의 답을 “채점”하도록 한 뒤, 성능을 주로 exact match(정확히 일치한 비율)로 검증해 왔습니다. 하지만 이 지표는 우연 합의 가능성을 보정하지 않아 chance-corrected 판별력을 체계적으로 과대평가한다는 한계가 지적돼 왔습니다.

- **Core Contribution**: 본 논문은 LLM-as-a-Judge의 judge validation을 대규모로 체계 평가해, 서로 다른 9개 제공자의 21개 judge 모델을 MT-Bench·JudgeBench·RewardBench 3종 벤치마크에서 118회 실행(약 54만 건 판단)으로 검증합니다. 또한 agreement/consistency/bias audit의 3가지 프로토콜을 함께 적용해, 단일 지표로는 드러나지 않는 문제(랭킹 불일치, 편향-신뢰도 역설)를 정량화합니다.

- **Technical Challenges**: 핵심 과제는 “재현성(test-retest)이 높아도 실제 판별이 올바른지”를 분리해 측정하는 데 있습니다. 이를 위해 Cohen’s kappa(우연 보정), Krippendorff’s alpha(다중 리런 일관성), position bias(위치 민감도), verbosity bias(길이 선호)를 함께 보고, AB+BA 순서 페어링으로 위치 편향을 통제한 상태에서 kappa deflation과 position flip 같은 진단 지표를 비교합니다.

- **Empirical Impact**: 결과적으로 exact match와 kappa 간 kappa deflation이 모든 judge에서 보편적으로 나타나며(특히 MT-Bench에서 33–41pp), 벤치마크 간 judge 랭킹은 최대 14위까지 흔들립니다. 더 나아가 consistency(높은 test-retest)와 bias(강한 position bias)가 동시에 존재하는 “consistency–bias paradox”가 실제 production 배치에서도 관측됐고, verbosity bias는 비교적 작게(<0.011) 측정되어 향후 최소 검증 프로토콜의 설계 근거를 제공합니다.



### Characterizing Narrative Content in Web-scale LLM Pretraining Data (https://arxiv.org/abs/2606.19468)
Comments:
          8 pages of main content, 28 total pages. 30 figures

- **Prior Approaches**: 기존 웹스케일 LLM pretraining은 데이터가 다뤄지지만, 텍스트의 ‘서사(narrative)’ 자체가 어떻게 구성되는지에 대한 정밀한 분석은 거의 없었습니다. 특히 narrative features를 해석 가능한 형태로 정의·측정하기 위한 표준 프레임워크가 부족해, 데이터 큐레이션이 서사적 품질을 어느 수준으로 반영하는지 평가하기 어려웠습니다.

- **Core Contribution**: 본 논문은 3조 토큰 규모 open pretraining corpus Dolma에서 서사의 미시적 특성을 처음으로 체계적으로 연구합니다. narrative theory를 바탕으로 agency, setting, events의 3요소를 11개의 해석 가능한 차원으로 operationalize하고, 이를 예측하는 RoBERTa 기반 NarraBERT를 fine-tuning·검증합니다.

- **Technical Challenges**: 핵심 난제는 웹 텍스트처럼 이질적인 데이터에서도 서사 구조를 연속적·다차원적으로 측정할 수 있게 모델링하는 것이었습니다. 저자들은 400개 구간을 샘플링·annotation해 11차원 예측을 학습시키고, NarraBERT를 Dolma의 300만 passages에 적용해 NarraDolma 데이터셋을 구축함으로써 대규모 일관된 서사 신호를 산출했습니다.

- **Empirical Impact**: 실험 결과, 극단적으로 이질적인 데이터 전반에서 서사 구조가 규모에 따라 측정 가능함을 보였고, 웹 텍스트 아래에 연속적·다차원적인 서사 구조가 존재함을 확인했습니다. 또한 pretraining source와 topic에 따라 narrative qualities가 불균등하게 분포하지만 기존 큐레이션은 이를 측정·반영하지 못한다는 점을 밝혀, 향후 데이터 구성과 서사 추론 품질의 연관 연구에 기반이 될 것으로 기대됩니다.



### Trustworthy Multi-Agent Systems: Mitigating Semantic Drift with the Argent Signaling Protoco (https://arxiv.org/abs/2606.19356)
Comments:
          17 pages

- **Prior Approaches**: 다중 에이전트 LLM(MAS)에서 리트라이(retry)는 대개 실패를 “다시 시도하면 나아질 것”으로 뭉뚱그려 처리해, 부분적으로만 근거가 맞는 오류와 아예 근거가 없는 오류를 구분하지 못했습니다. CAMEL/AutoGen 같은 에이전트 프레임워크와 Self-Refine/Reflexion 계열 개선은 대화 로그가 남아도, 재시도가 근거 기반의 수리(repair)였는지 아니면 중단(containment)해야 할 유형이었는지 기계적으로 판별하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 Argent Signaling Protocol(ASP)을 제안하며, 모든 AI 응답에 @C(확실성), @G(근거), @S(확률적 불안정성), 그리고 claim evidentiary basis를 분류하는 assumption index를 “머신-리더블 헤더”로 함께 부착합니다. 이를 통해 컨트롤러가 재시도의 타당성(수리 가능 vs 차단 필요)을 구분하고, 각 실패 유형에 맞는 라우팅 결정을 내릴 수 있게 합니다. 또한 sidecar 방식으로 에이전트 내부 수정 없이도 메시지 경계에서 품질 게이트를 강제하는 배치를 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 모델 내부 로그 없이도 근거/불안정성을 일관되게 추정 가능한 신호를 설계하고, (2) 다중 턴에서 semantic drift가 누적될 때 이를 감지해 올바르게 경로를 분기하는 것입니다. 논문은 문서 기반 QA에서 토큰 단위의 문맥 중복도, 인용 존재/정합성, 질문/문맥을 벗어난 novel ratio 등을 근거로 @G/@C/@S를 16단계(0~F)로 양자화해 헤더에 넣고, JSD 기반 drift로 “좋은 정답 프로필에서 얼마나 벗어났는지”를 경계 임계값으로 판단하도록 구성합니다.

- **Empirical Impact**: 문서 기반 QA 벤치마크에서 ASP를 사용하면 모델별 pass rate가 유의미하게 개선됩니다(예: Qwen~0.8B 11.1%→33.3%, Dobby~8B 33.3%→44.4%). 또한 다중 에이전트 파이프라인에서는 retrieval 에이전트가 내보낸 근거 없는 출력이 downstream 의사결정 에이전트로 전달되지 않도록 sidecar가 100% 차단(24/27 blocked, ungrounded propagation 0)을 보였습니다. 결과적으로 “리트라이가 필요했던 실패만 수리 루프로 보내고, 비근거 실패는 즉시 차단”하는 운영 가능성을 실증하며 규제/감사 관점의 traceability를 강화하는 의미가 있습니다.



### Granularity-Regulated Adaptive Computational Efficiency for Optimal Verification in Test-Time Scaling (https://arxiv.org/abs/2606.19354)
- **Prior Approaches**: Test-time scaling (TTS)는 추론 시 추가 compute를 써서 LLM의 추론 성능을 끌어올리는 패러다임이다. 기존에는 outcome reward model(ORM)처럼 정답 전체를 한 번에 평가하거나, process reward model(PRM)처럼 reasoning step마다 점수를 매기는 방식이 주로 쓰였지만, 둘 중 어느 쪽이 “얼마나 잘게” 최적인지는 불명확했다. 이 때문에 베스트-of-N, beam search, MCTS 등은 존재했지만 verifier granularity가 compute 예산과 문제 난도에 따라 어떻게 달라져야 하는지 이론적 기준이 부족했다.

- **Core Contribution**: 이 논문은 verification granularity의 최적점을 compute 예산, 문제 난도, verifier 정확도의 함수로 제시하는 단일 이론 프레임워크 GRACE를 제안한다. GRACE는 coarse(ORM 쪽)와 fine(PRM 쪽) 사이를 연속적인 granularity g로 모델링하고, compute-performance Pareto frontier를 만족하는 granularity 정책군을 정리한다. 특히 “최적 granularity가 언제 coarse에서 fine로 바뀌는가”를 단계적으로 설명하는 phase transition까지 포함한다.

- **Technical Challenges**: 핵심 난제는 granularity를 더 잘게 하면 precision(정밀한 선택)이 늘지만, 동시에 검증 비용과 후보 수 감소로 인한 sample loss가 커진다는 상반된 힘을 동시에 정량화하는 것이다. 논문은 verifier 비용을 granularity에 대해 분리 가능한 형태로 두고, 문제 난도 δ와 verifier 정확도 α(g)의 증가 관계를 가정해 최적화 조건(정밀 이득=샘플 손실)을 유도한다. 그 결과로 compute 예산이 임계값을 넘을 때는 fine이, 임계값 아래에서는 coarse가 유리해지는 closed-form 임계 구간을 증명한다.

- **Empirical Impact**: 실험은 MATH-500, GSM8K, AIME에서 수행됐고, GRACE-Adapt(문제별 난도 추정 후 granularity를 적응적으로 선택)는 고정 granularity baselines를 전반적으로 능가했다. 개선 폭은 난도가 높은 AIME에서 최대 +3.4점 수준으로 커졌고, 쉬운 GSM8K에서는 +1.4%로 작게 나타나 이론의 경향(난도·compute에 따른 전환)을 지지했다. 또한 MATH-500의 quartile별 관측 phase transition 지점은 이론 예측과 중간 상대오차 6.2% 내로 맞았으며, 난도 추정에 잡음이 들어가도 성능이 완만하게 떨어져 실용적 강건성도 확인됐다.



### Quantifying Aleatoric Uncertainty of In-Context Learning for Robust Measure of LLM Prediction Confidenc (https://arxiv.org/abs/2606.19353)
Comments:
          Accepted to ACL 2026

- **Prior Approaches**: ICL은 소수의 데모로 가중치 업데이트 없이 태스크를 적응하지만, 프롬프트 설계나 데모 순서, 데이터 특성에 따라 예측이 흔들리는 ‘취약성’이 큽니다. 기존 uncertainty quantification/uncertainty decomposition 연구는 주로 표준 생성·분류 설정에 맞춰져 있어 ICL의 컨텍스트 의존 역학을 그대로 반영하기 어렵습니다.

- **Core Contribution**: 이 논문은 ICL 내부 표현을 활용해 aleatoric uncertainty(AU)를 직접 추정하는 self-function vectors 개념을 제안합니다. Bayesian 관점의 잠재 개념 학습을 메커니즘 해석(특정 attention head의 기능 벡터)에 연결해, 출력 흔들림이나 decoding 조작에 의존하지 않고 AU를 측정합니다.

- **Technical Challenges**: 핵심 난제는 ICL에서 AU/EU를 ‘데이터 요인 vs 모델 요인’으로 분리해 신뢰성 있게 평가하는 방법이 없다는 점입니다. 이를 위해 라벨 노이즈/쿼리 분포 이동처럼 원인을 독립적으로 조절하는 최초의 평가 프로토콜을 설계했고, self-function vector 개입과 causal head 선정을 통해 내부표현 기반 분해를 수행합니다.

- **Empirical Impact**: WordNetMCQ 같은 합성-통제 환경과 실제 데이터셋 전반에서 self-function vectors가 기존 entropy 기반 분해 방법보다 AU/EU 상관을 더 안정적으로 보였다고 보고합니다. 또한 hallucination detection 등 신뢰성 관련 작업에서도 경쟁력 있는 성능을 보여, 분해 결과를 실용적 도구로 확장할 가능성을 제시합니다.



### Sign-Language Datasets at Scale: A Comprehensive Survey on Resources, Benchmarks, and Annotation Standards (https://arxiv.org/abs/2606.19352)
Comments:
          Accepted to ACL 2026 Main. 27 pages, 5 figures

- **Prior Approaches**: 기존 sign-language 연구는 SLR, SLT, SLP 중 특정 과제나 일부 고빈도 벤치마크에 집중돼, 데이터셋이 실제 사용 환경과 얼마나 맞물리는지 체계적으로 비교하기 어려웠습니다. 또한 데이터셋이 서로 분절돼 있고(언어 커버리지 불균형), 주석 체계·입력 모달리티·메타데이터가 제각각이라 재현성과 교차 데이터 학습이 약해집니다. 결과적으로 모델 성능 향상이 “모델”보다 “데이터 성질”에 좌우되는 경향이 강했습니다.

- **Core Contribution**: 이 논문은 공개된 sign-language 데이터셋 120개(35개 수어)를 한데 모은 데이터셋 인덱스를 제시하고, SLR/SLT/SLP 전반의 공통 병목(모달리티 불균형, signer bias, 주석 불일치)을 분석합니다. 더 나아가 24-field Sign-Language Datasheet를 도입해 데이터셋을 표준화된 문서 양식으로 기록하게 하고, GitHub 저장소를 통해 재현 가능한 평가 기반을 제공합니다. 즉 “데이터셋-벤치마크-평가”를 연결하는 데이터 중심 관점을 실무적으로 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) RGB/depth/pose 등 입력 모달리티와 주석 레이어(글로스, 문장 정렬 등)가 데이터셋마다 달라 모델·전처리·평가 파이프라인이 호환되지 않는 문제, (2) 글로스 품질/관행이 일관되지 않아 학습 신호의 해석 가능성과 transferability가 흔들리는 문제, (3) signer 인구통계와 손잡이(handedness) 같은 메타데이터가 부족해 편향을 분석·제어하기 어려운 문제입니다. 논문은 datasheet와 통합 메타데이터 관리를 통해 문서화 공백을 줄이고, 모달리티/주석/도메인 차이를 벤치마크 결과 해석에 직접 반영하는 방식으로 해결 방향을 제시합니다.

- **Empirical Impact**: PHOENIX14T·CSL-Daily·How2Sign·YouTube-ASL·OpenASL을 중심으로 벤치마크를 작업별(SLR/SLT/SLP)·글로스 유무별로 정리한 결과, 글로스 기반 설정은 성능 지표가 높게 나오되 실제 적용성에는 트레이드오프가 있음을 보여줍니다. 또한 CSL-Daily처럼 환경·화자 다양성이 큰 데이터셋에서 WER/BLEU 갭이 더 크게 나타나 일반화 평가의 중요성이 확인됩니다. SLP는 평가 파이프라인과 공개성 부족으로 비교 일관성이 낮다는 점을 지적하며, BLEU 외에 MPJPEDTW, timing F1, 인체 평가 등 보완 지표 필요성을 제안해 향후 연구 품질을 끌어올리는 기준점 역할을 합니다.



### Detecting Hallucinations for Large Language Model-based Knowledge Graph Reasoning (https://arxiv.org/abs/2606.19351)
- **Prior Approaches**: 기존 KG reasoning은 관련 triple을 검색해 프롬프트에 넣어 정확도를 높이지만, LLM이 여전히 검색 지식을 어긋나게 생성하는 hallucination 문제는 지속됩니다. 일반 hallucination 탐지는 LLM 내부 상태나 불확실성에 치우쳐 외부의 KG 구조 정보를 충분히 반영하지 못하고, RAG 전용 탐지는 검색 컨텍스트와의 정합성만 보며 KG의 관계·연결 구조를 놓치는 경우가 많았습니다.

- **Core Contribution**: 이 논문은 LLM 기반 KG reasoning에서의 hallucination을 구조적으로 탐지하는 최초의 방법 LUCID를 제안합니다. LUCID는 LLM attention(내부 집중), KG semantic similarity(관계-질문 의미 적합도), KG structural information(그래프 연결성)을 함께 결합해 “어떤 근거가 실제로 쓰였는지”를 판단하도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 retrieved KG를 프롬프트로 읽는 과정에서, 잘못된 답이 어떤 KG 관계 선택/구조적 모순에서 비롯되는지 정량 신호로 만드는 것이었습니다. LUCID는 응답 토큰이 retrieved subgraph의 노드·엣지에 주는 attention 점수를 계층·헤드별 행렬로 추출한 뒤, 관계 임베딩과 질의 임베딩의 cosine similarity로 의미 점수를 보강하고, 이를 GNN(GINE)에 넣어 그래프 단위 hallucination 확률을 예측합니다.

- **Empirical Impact**: 평가 결과 LUCID는 9개 데이터셋에서 15개 베이스라인을 상대로 SOTA 성능을 달성했으며, 세 프레임워크 전반에서 안정적으로 우수했습니다. 또한 hallucination 확률을 QA 후처리에 활용한 Mixed 전략은 고가 API 모델만 전면 적용했을 때와 비슷한 EM 정확도를 유지하면서 비용을 평균 55.4% 절감해, 신뢰도 측정이 실제 운영 효율 개선으로 이어질 수 있음을 보여줍니다.



### Pruning via Causal Attribution Preserves Reasoning Performance in Large Language Models (https://arxiv.org/abs/2606.19350)
Comments:
          Accepted at the ICLR 2026 Workshop on LLM Reasoning. 13 pages, 2 figures

- **Prior Approaches**: 기존 LLM pruning은 주로 가중치 크기나 입력 활성 기반 같은 상관(signals)으로 중요도를 추정해 one-shot으로 잘라내며, 재학습이 없어도 압축 효과를 얻습니다. 다만 추론에 핵심인 low-magnitude 파라미터가 남겨지지 않거나, gradient 기반 국소 민감도는 교차-레이어 상호작용을 충분히 반영하지 못해 chain-of-thought 같은 멀티스텝 추론에서 성능 붕괴가 나타날 수 있습니다.

- **Core Contribution**: CAP(Causal Attribution Pruning)은 학습 없이, 주어진 추론 벤치마크의 작은 calibration set에서 attention head를 마스킹했을 때의 성능 저하를 인과적으로 측정해 “기능적으로 중요한 head”를 찾는 방식입니다. 그런 head-level 인과 점수를 각 attention projection 가중치 구간에 전파해 weight-level magnitude pruning 기준을 만들며, 추론 과제를 기준으로 점수를 정렬해 capability 보존을 노립니다.

- **Technical Challenges**: 인과적 head 평가를 위해 각 head를 실제로 forward-time에서 마스킹해야 하는데, 이때 내부 confound를 피하려고 attention 집계 후 output projection 직전 지점에 forward hook으로 개입합니다. 또한 표본이 적으면 head 중요도 추정이 흔들릴 수 있어 K개의 disjoint subsample로 분산을 줄이고 median 기반 집계와 intact baseline을 결합해 이상치에 강건하게 설계했습니다.

- **Empirical Impact**: Llama-3-8B-Instruct와 Mistral-7B-Instruct에서 GSM8K, StrategyQA, ARC-Challenge를 10~50% sparsity로 평가했으며, 특히 Llama-3의 ARC-Challenge에서 20% sparsity 기준 상대 최대 61% 성능 향상을 Wanda 대비 보였습니다. 다만 50%처럼 높은 sparsity에서는 MLP를 attention head 평균으로 대략화하는 제약 때문에 general language modeling 붕괴가 관측되며, Mixture-of-Experts(MoE) 같은 동적 라우팅 구조로의 전이는 추가 확장이 필요하다는 점이 드러났습니다.



### Where to Place the Query? Unveiling and Mitigating Positional Bias in In-Context Learning for Diffusion LLMs via Decoding Dynamics (https://arxiv.org/abs/2606.19349)
Comments:
          9 figures, 4 tables

- **Prior Approaches**: 기존 ICL 연구는 주로 example selection과 example ordering에 초점을 맞추며, dLLM에서는 쿼리(질문) 삽입 위치를 사실상 고정된 포맷(대개 trailing)으로 취급해 왔습니다. AR LLM에서는 causal masking 때문에 테스트 쿼리가 시퀀스 맨 끝에 고정되지만, dLLM은 bidirectional attention 덕분에 위치를 바꿀 수 있어 구조적 전환이 필요합니다. 그럼에도 대부분의 실무는 AR 스타일 템플릿을 그대로 답습해 위치 최적화 가능성을 놓쳤습니다.

- **Core Contribution**: 이 논문은 dLLM에서 query placement(질문 삽입 위치)가 성능에 대해 first-order variable이라는 점을 실증적으로 규명합니다. 특히 위치를 바꾸는 효과가 데모 의미를 교체하는 효과와 비슷한 수준일 수 있음을 보여주며(GSM8K에서 r≈1.236), 최적 위치가 과제 유형에 따라 달라진다고 정리합니다. 결론적으로 dLLM ICL을 “포맷 고정”이 아니라 “공간 토폴로지 최적화” 문제로 재정의합니다.

- **Technical Challenges**: 핵심 난제는 레이블 없이(ground-truth 없이) 어떤 query 위치가 생성 안정적인지 판별하는 신호가 기존에 없다는 점입니다. 단일 스텝 confidence(C_decoded)는 dLLM의 iterative decoding 진화(temporal evolution)를 버려 위치 랭킹을 실패하게 만들며, 이를 해결하기 위해 전체 디코딩 궤적을 누적하는 Average Confidence(\bar{C})를 제안합니다. 이후 \bar{C}로 후보 위치들에 대한 안정성을 점수화해 학습 없이 Auto-ICL(적응형 위치 라우팅)을 구성합니다.

- **Empirical Impact**: 실험에서는 Sudoku(전역 지각)와 GSM8K(순차 추론)처럼 인지 패러다임이 다른 벤치마크에서, Auto-ICL이 정적 배치 대비 최적 성능에 근접하거나 능가하는 결과를 보입니다. 예를 들어 Sudoku에서는 prefix 경계 최적 성향을 회복하고, GSM8K에서는 순차 추론에 필요한 trailing 성향을 유지해 성능을 동시에 방어합니다. 또한 추가 지연은 미미한 편(예: GSM8K에서 약 +0.08s)이며, 제한된 generation budget이나 shot 수 변화 같은 악조건에서도 인스턴스 단위 라우팅이 정적 기준선을 넘는 사례를 확인해 실용성이 강조됩니다.



### DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligenc (https://arxiv.org/abs/2606.19348)
- **Prior Approaches**: 기존 reasoning 모델은 test-time scaling을 통해 성능을 끌어올리지만, vanilla attention의 제곱 복잡도 때문에 초장문(예: 100만 토큰)에서는 계산·메모리 병목이 커집니다. 오픈소스 LLM들도 long-horizon 작업을 지원하려 했으나, 근본적으로 초장 시퀀스 효율을 크게 개선하지 못해 추가 확장이 제한돼 왔습니다.

- **Core Contribution**: DeepSeek-V4 시리즈는 DeepSeekMoE와 MTP 설정은 유지하면서, 초장맥락 효율을 극적으로 낮추는 새로운 아키텍처·최적화를 결합해 100만 토큰 컨텍스트를 “실제로” 가능하게 합니다. DeepSeek-V4-Pro(1.6T, activated 49B)와 DeepSeek-V4-Flash(284B, activated 13B) 두 라인업을 제시하며 long-context 및 추론 모드의 성능/비용 균형을 함께 겨냥합니다.

- **Technical Challenges**: 초장문에서는 attention의 KV cache와 추론 FLOPs가 지배 항이 되므로, 계산량·저장량을 동시에 줄이면서도 품질을 유지하는 설계가 핵심 난제입니다. 논문은 CSA(Compressed Sparse Attention)와 HCA(Heavily Compressed Attention)를 하이브리드로 엮어 KV를 압축·선택적으로 참조하게 만들고, mHC(Manifold-Constrained Hyper-Connections)와 Muon optimizer, FP4 양자화 인지 학습 및 MoE 인프라 최적화를 추가해 안정성과 효율을 동시에 확보했다고 설명합니다.

- **Empirical Impact**: 100만 토큰 조건에서 DeepSeek-V4-Pro는 DeepSeek-V3.2 대비 single-token 추론 FLOPs를 27%, KV cache를 10%로 낮추며, DeepSeek-V4-Flash는 FLOPs 10%, KV cache 7% 수준까지 더 줄입니다. 벤치마크에서는 DeepSeek-V4-Pro-Max가 오픈 모델 대비 추론·코딩·롱컨텍스트 전반에서 강세를 보이고, 일부 knowledge 영역에서는 근소 우위를 보이면서도 Gemini급 격차를 상당히 좁혔다고 보고해 open models의 장문 추론 기준선을 끌어올렸다는 의미가 큽니다.



### How LLMs Fail and Generalize in RTL Coding for Hardware Design? (https://arxiv.org/abs/2606.19347)
Comments:
          Preview, under submission for EMNLP 2026

- **Prior Approaches**: 기존 연구는 RTL(Verilog) 코드를 잘 생성하도록 SFT나 RL fine-tuning을 적용하거나, RTL 데이터셋을 확장/도메인적응시키는 방식이 주를 이뤘습니다. 하지만 평가 단계에서 실패 원인을 체계적으로 분해하지 못해, 왜 한계가 생기는지(지식 부족 vs 단순 형식 오류)를 관찰하기 어렵다는 문제가 남아 있습니다. 또한 alignment나 보상 최적화가 실제로 ‘컴파일’만 개선하는지, ‘검증을 통과하는 기능’을 늘리는지는 명확히 정리되지 않았습니다.

- **Core Contribution**: 이 논문은 인지이론에서 영감을 받은 문제 solvability(해결 가능성) 관점의 4단계 오류 분류 체계를 제안합니다. L1 구문(syntactic), L2 의미(semantic), 그리고 기능(functional) 오류를 L3S(해결 가능하지만 현재 샘플에서 실패)와 L3U(어떤 롤아웃에서도 실패)로 나눠, 테스트벤치에서 드러나는 지식 격차를 구분해냅니다. 이 프레임워크로 LLM의 RTL 생성 실패를 ‘오류 파이프라인’ 관점에서 정량화할 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술적 난제는 RTL의 병렬 temporal 논리를 순차 실행처럼 추적하기 어렵고, 따라서 실패 원인이 단순 문법/타이핑 실수인지 더 깊은 기능적 추론 부족인지 분해해야 한다는 점입니다. 저자들은 HDL 컴파일·정적 제약(엘라보레이션/린팅/합성)과 테스트벤치 시뮬레이션(명세 충족)을 단계적으로 매핑해, 오류를 L1→L2→L3S/L3U로 배타적으로 분할하는 프레임워크를 구축했습니다. 또한 GRPO 기반 RL fine-tuning 동안 롤아웃 분포가 어떤 오류 레벨로 이동하는지까지 추적해, ‘선택 압력’이 오류 파이프라인을 어떻게 밀어내는지 분석합니다.

- **Empirical Impact**: VerilogEval-Human(VerilogEval) 평가에서 frontier 모델들은 초기 pass rate가 90.8%에서 포화되며, 이 한계는 주로 L3U(unsolvable functional errors)에서 기인하는 것으로 드러납니다. 더 나아가 SFT/RL은 L1+L2를 줄이지만 L3(특히 기능 오류)를 늘리는 경향이 있어, 모델이 ‘컴파일은 잘하지만 하드웨어 이해(검증 통과)는 부족한 상태’로 남는다는 메시지를 강화합니다. 결론적으로 정렬(alignment)만으로는 RTL 파이프라인의 근본 병목을 넘기 어렵고, 모델 추론 능력에 대한 추가 연구가 필요하다는 실증적 근거를 제공합니다.



### Disentangling Linguistic Relatedness from Task Alignment in Cross-Lingual Transfer (https://arxiv.org/abs/2606.19346)
- **Prior Approaches**: 기존 교차언어 전이는 mBERT/XLM-R처럼 공유 표현을 학습하거나, 모델 스케일·instruction tuning이 전이를 좌우한다고 보는 관점이 많았습니다. 다만 decoder-only LLM로 오면 용량 배분, multilinguality의 저주, fine-tuning이 기존 다언어 표현을 덮어쓰는 현상(catasrophic forgetting) 등이 얽혀 결과 해석이 복잡해졌습니다. 특히 Semitic 언어(아랍어-히브리어-암하라어-몰타어) 연구는 대부분 인코더 중심·단일/이중 언어에 치우쳐, 추론이 중요한 제로샷 독해에서의 전이 양상은 덜 탐구돼 왔습니다.

- **Core Contribution**: 이 논문은 아랍어 방언 데이터로 7개 대형 LLM(4B~671B)을 fine-tuning한 뒤, Semitic 언어(히브리어/암하라어/몰타어)와 비Semitic 대조군(일본어/한국어/프랑스)에서 Belebele 벤치마크 제로샷 독해를 비교합니다. 언어 계통·스크립트(아브자드/아부기다/라틴/CJK)를 함께 흔들어, “아랍어 fine-tuning이 Semitic 지식만 골라 옮기는가”를 통제된 설계로 검증합니다. 결과적으로 Semitic-specific 전이 근거는 없고, 성능 향상은 전반적으로 task-format alignment(평가 포맷에 맞춘 정렬)와 더 잘 맞는 그림을 보여줍니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) train-test overlap을 엄격히 통제하고 (2) 언어 계통/스크립트 효과를 분리해 (3) ‘정말로 교차언어 지식 전이’인지 ‘포맷 캘리브레이션 개선’인지 구분하는 것입니다. 이를 위해 Belebele의 공통 질문 ID 기반 split으로 평행성을 유지했고, 평가도 자유생성 대신 logprob 기반으로 1~4 정답 토큰을 판별해 결정성을 확보했습니다. 또한 CoT(Chain-of-Thought) ablation을 통해 fine-tuning 없이도 추론 컨텍스트를 붙였을 때 같은 모델들이 유사한 수준의 이득을 얻는지 확인해, 두 메커니즘이 동일한 병목(포맷 정렬/캘리브레이션)을 보정한다는 해석을 강화했습니다.

- **Empirical Impact**: 실험에서는 베이스라인이 약한 모델(특히 MoE 계열 GPT-OSS-120B/20B)이 아랍어 fine-tuning 후 대폭 개선하지만, 그 이득이 Semitic에만 몰리지 않고 비Semitic 대조군에서도 거의 같은 크기로 확장됩니다. 반대로 베이스라인이 강한 모델(DeepSeek-V3.1, Qwen3-32B)은 미세한 개선만 보이며, baseline을 통제한 회귀에서도 Semitic 지표가 오히려 부정적(유의미한 우위 없음) 계수로 나타났습니다. CoT 프리펜딩 ablation은 fine-tuning에서 가장 많이 이득을 본 모델이 추론에서도 동일하게 많이 이득을 봄을 보여, “Semitic 전이”로 보이던 현상이 사실은 task-format alignment/분포 캘리브레이션 보정일 수 있음을 시사합니다.



### Ensembles of Large Language Models for Identifying EQ-5D Studies in PubMed Based on Their Abstracts (https://arxiv.org/abs/2606.19345)
Comments:
          6 pages, 7 tables, 8 equations

- **Prior Approaches**: 체계적 문헌고찰(SLR)에서 포함/제외 기준을 만족하는 논문을 사람이 선별하면 속도·일관성·오류 문제가 커집니다. 특히 EQ-5D처럼 임상적 해석이 필요한 건강 관련 삶의 질 지표는 초록만으로 판별하기가 어려워 기존 자동화 NLP/LLM 연구의 성능 신뢰성 확보가 핵심 과제로 남아있습니다. 기존 LLM 기반 선별은 성능은 높아도 과업 표준화와 도메인(바이오메디컬) 텍스트 처리 한계가 자주 지적됩니다.

- **Core Contribution**: 이 논문은 PubMed의 바이오메디컬 초록만을 입력으로, EQ-5D 결과 보고 여부를 자동 탐지하는 앙상블 LLM 프레임워크를 제안합니다. few-shot prompting으로 기본 분류 능력을 끌어올리고, 모델별 예측을 weight ensembling 및 soft stacking(로지스틱 회귀 메타 분류기)으로 결합해 신뢰성과 균형을 동시에 노립니다. 최종적으로 gemini-2.5-pro, gemma-3-12b, gemma-3-27b 3개 모델 조합이 최우수 성능을 보였다고 보고합니다.

- **Technical Challenges**: 초록에 EQ-5D를 ‘명시적으로’ 언급하는 경우만 양성으로 라벨링해야 하므로, 도메인 특화 신호를 잡는 능력과 확률의 신뢰도 보정이 함께 필요합니다. 또한 모델마다 편향이 달라 단순 투표만으로는 정밀도-재현율 균형이 흔들릴 수 있어, F1 기반 가중치와 confidence를 함께 사용하는 weight ensembling을 설계했습니다. 더 나아가 soft probabilities와 raw confidence를 메타 특징으로 넣은 soft stacking에서 로지스틱 회귀로 조합 가중치를 학습해 일반화 및 해석가능성을 높였습니다.

- **Empirical Impact**: 수작업 라벨(전문가 2인 검토) 200편 데이터셋에서 개별 9개 Gemini/Gemma 모델 중 gemini-2.5-pro가 가장 높은 weighted F1(0.71)을 기록했고, 상위 3개 모델을 앙상블했을 때 weighted F1과 accuracy가 각각 0.74로 향상됐습니다. soft stacking은 weighted F1 0.72, accuracy 0.73으로 유사한 수준의 성능을 유지하며 메타 특징 중요도 분석에서 LLM의 확률(soft probability)이 판별에 가장 큰 기여를 한다는 점을 보여줍니다. 런타임과 비용 추정에서도 모델 크기 선택에 따라 7~64분 및 0.07~5.04달러 수준으로 스케일링 가능성을 제시해, 바이오메디컬 SLR 자동 선별의 실용성을 강화합니다.



### Exposing the Unsaid: Visualizing Hidden LLM Bias through Stochastic Path Aggregation (https://arxiv.org/abs/2606.19344)
Comments:
          14 pages

- **Prior Approaches**: 기존 편향 감사는 단일 출력만 점검하거나 WEAT/SEAT 같은 정적 자동 지표에 의존해, 확률 분포의 내부 분기(낮은 확률 경로)에 숨은 편향을 놓치기 쉽습니다. 또한 고정 템플릿 기반 카운터팩추얼 probing도 문법적 스퓨리어스 상관을 유발해 의미 기반 진단의 신뢰도를 흔들 수 있습니다.

- **Core Contribution**: TreeTracer는 온톨로지(의미 변수를 반영한 단어 집합)를 프롬프트의 특정 토큰 자리에 체계적으로 치환하고, 수백 개의 확률적 생성 결과를 문법 구조에 맞춰 집계 비교하는 시각 분석 워크스페이스를 제안합니다. 두 온톨로지 트리(예: 남성/여성 이름)의 차이를 Sankey 다이어그램 기반으로 나란히 보여주며, 무엇이 “사라졌는지”가 아니라 “확률 질량이 어디로 이동했는지”를 추적하도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 (1) 수많은 stochastic generation을 읽을 수 있는 구조로 줄이면서도 (2) 낮은 빈도의 숨은 확률 질량을 보존하고 (3) 토큰 확률을 단순 합산이 아닌 의미-맥락까지 반영해 일관되게 렌더링하는 것입니다. TreeTracer는 constituency 파서로 문법 트리 집계를 수행하고 하위워드 확률은 기하평균으로 재조정하며, 의미 카테고리 기반 노드 머징과 decoupled Sankey 인코딩(노드 높이=global probability, 링크 폭=selected sample probability)을 통해 가지치기 이후의 손실을 시각적으로 드러냅니다.

- **Empirical Impact**: 검증에서는 GPT-2 XL(unaligned)과 constitutionally aligned Apertus 모델을 비교해 대명사 억제(counterfactual pronoun suppression)와 특정 개인에 대한 대화 맥락의 주변화 같은 representational harms를 “숨은 확률 분기”까지 포함해 드러냈습니다. 또한 예비 사용자 연구에서 집계 비교 인터페이스가 분석가의 인지 부하를 줄이면서 체계적 편향 탐지에 효과적임을 확인했습니다.



### LedgerAgent: Structured State for Policy-Adherent Tool-Calling Agents (https://arxiv.org/abs/2606.20529)
Comments:
          Work in Progress

- **Prior Approaches**: 기존 tool-calling 에이전트는 사용자 대화, 도구 응답, 정책 문서를 프롬프트에 함께 쌓아두고 매 턴마다 필요한 ‘상태’를 모델이 재구성하도록 맡기는 방식이 많았다. 이 설계는 (1) 올바른 기록을 찾더라도 이후 의사결정에서 오래되었거나 누락된 상태에 근거할 수 있고, (2) 문법적으로는 유효한 write/tool 호출이라도 ‘현재 상태’에 의존하는 정책을 위반할 수 있다는 실패를 낳는다. 즉 도구 선택/콜 자체만이 아니라, 행동 경계에서의 상태-정책 정합성 확인이 취약했다.

- **Core Contribution**: LedgerAgent는 추론(inference) 시점에 관측된 작업 상태를 별도의 ‘ledger(장부)’에 명시적으로 유지하고, 이를 프롬프트에 렌더링해 모델이 상태를 검색/재구성하지 않아도 되게 만든다. 더 나아가 환경을 바꾸는(environment-changing) 도구 호출 직전에 policy gate로 ledger 필드 기반 정책 제약을 검증해, 위반 가능성이 있으면 호출을 막거나(revise/block) 피드백을 준다. 모델 가중치는 바꾸지 않고, 상태 표현과 행동 경계 검증 방식을 시스템 레벨에서 교체하는 것이 핵심이다.

- **Technical Challenges**: 핵심 난제는 상태가 대화 로그에 흩어져 있을 때 생기는 stale·missing·incorrect grounding 문제를 줄이고, 정책이 상태 의존적일 때도 write를 안전하게 걸러내는 것이다. 논문은 (1) tool 경로 맵에 따라 성공적인 read 도구 반환을 스키마-고정 typed dictionary로 저장하는 ledger를 만들고, (2) environment-changing 호출 제안마다 ledger를 근거로 실행 가능한 predicate 정책 게이트를 적용해 누락된 증거가 있으면 revise로 되돌리는 구조를 제시한다. 또한 write 직후에는 observe-not-assume 원칙으로 ‘다시 읽어 상태를 확인한 뒤’ ledger를 갱신해, 관측 기반 일관성을 유지한다.

- **Empirical Impact**: 네 가지 customer-service 도메인과 open/closed-weight 혼합 백본 모델에서 LedgerAgent는 표준 프롬프트 기반 도구 호출 대비 pass^k(특히 pass^4의 일관성 지표)에서 평균적으로 더 높은 성능을 보였다. 감소폭이 아니라 일관된 개선이 관찰되며, 다른 컨텍스트 엔지니어링 계열 방법(IRMA)보다 토큰 오버헤드를 늘리지 않으면서도 더 나은 결과를 보고한다. 실패 분석에서는 남는 오류가 대부분 누락된 필수 행동(missed actions)과 상태-인자 인식 오류(틀린 인자)로 나타나, 명시적 상태와 write-time 검증이 중요한 불안정성 원인을 줄였음을 시사한다.



### Scalable Training of Spatially Grounded 2D Vision-Language Models for Radiology (https://arxiv.org/abs/2606.20477)
Comments:
          Accepted for MICCAI 2026. First two authors: equal contribution. Last two authors: equal supervision

- **Prior Approaches**: 기존 방사선 VLM은 보고서나 질의응답은 잘 생성하지만, 출력이 어떤 영상 영역을 근거로 했는지 “공간적으로” 확인하기 어렵다는 한계가 컸습니다. 또한 CT/MRI에서 수동 2D 공간 어노테이션이 부족해 학습 규모를 키우기 어려웠고, 2D 슬라이스 기반 시도도 제한된 데이터 규모(강한 필터링 등)로 확장에 제약이 있었습니다. 가슴 X-ray 중심의 grounded VLM 연구는 있었지만, 임상 CT/MRI 대규모에서의 공간 검증 학습은 상대적으로 덜 탐구됐습니다.

- **Core Contribution**: 이 논문은 수동 공간 어노테이션 없이 CT/MRI 슬라이스에 대한 visually grounded 학습을 가능하게 하는 RefRad2D와 RadGrounder를 제안합니다. RefRad2D는 임상 리포트에서 추출한 1.2M(독일어/영어) 이미지-텍스트 쌍과, TotalSegmentator 기반의 자동 공간 레이블을 결합한 대규모 데이터셋입니다. RadGrounder는 PaliGemma 2 기반 멀티태스크 VLM으로 보고서 생성, visual question answering, 그리고 bounding-box(또는 segmentation) 형태의 공간 그라운딩을 동시에 수행합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 단일 슬라이스 VLM에서 리포트의 시간적 표현이 환각을 유발하지 않도록 정제하고, (2) 3D 세그멘테이션 결과를 2D 입력에 맞춰 정확히 텍스트-영상을 정렬하며, (3) 그라운딩 감독을 추가해도 언어 품질을 해치지 않는 학습 구성이었습니다. 연구진은 GPT-OSS로 캡션의 시간 맥락을 제거하고, Gemma 3로 독일어→영어 번역을 품질 개선(샘플링·재판정)하며, GPT-OSS가 캡션의 해부학적 언급을 추출해 마스크 클래스 스키마(C=121)와 매칭하도록 자동 파이프라인을 구성했습니다. 모델 측에서는 추가 디코더/손실 없이 token-based bounding-box detection을 텍스트 생성으로 학습해 효과적인 로컬라이제이션을 구현했고, ablation 결과 그라운딩 감독이 VQA/보고서 생성 성능을 떨어뜨리지 않는 것도 확인했습니다.

- **Empirical Impact**: 외부 VQA 벤치마크 Slake와 VQA-RAD에서 RadGrounder는 전문 의료 VLM들과 경쟁하는 결과를 보이며, 특히 VQA-RAD에서 open F1을 비교 방법 중 최고 수준으로 달성했습니다. 또한 다운스트림 데이터만으로 fine-tuning하는 것에 비해, RefRad2D의 임상 데이터를 학습 혼합에 추가하면 open-ended VQA 성능이 개선되어 데이터 전이성(transferability)이 입증됐습니다. 더불어 bounding-box/segmentation 같은 공간 그라운딩 감독을 넣어도 언어 품질 저하가 없다는 점이 정량적으로도 뒷받침되어, “공간적으로 검증 가능한” 방사선 해석을 VQA 성능 저비용으로 제공할 수 있음을 시사합니다.



### Token-Operations-Oriented Inference Optimization Techniques for Large Models (https://arxiv.org/abs/2606.20295)
Comments:
          62 pages, 36 figures

- **Prior Approaches**: 기존 대형 모델 서빙 최적화는 주로 단일 모델 가속(예: KV cache, 배치, 네트워크 최적화)이나 라우팅을 통한 일부 비용 절감에 집중돼 왔습니다. Multi-model 협업은 있었지만, 어떤 모델을 언제/왜 선택하는지에 대한 ‘역량 경계’의 표준화와 이를 운영 루프에 연결하는 체계가 부족하다는 한계가 있었습니다. 또한 Cascading(단계적 상향 호출)이나 Ensembling(병렬 호출 후 통합)은 유망했지만, 신뢰도 판정 신호와 임계값 설계가 과제였습니다.

- **Core Contribution**: 이 논문은 토큰 단위 운영을 위해 Multi-model Fusion, Model Optimization, Compute-Model Fusion, Compute-Network-Model Fusion의 4계층 기술 아키텍처를 처음으로 제안합니다. 각 계층은 모델 선택/조합, 토큰 생성 비용 자체의 감소, 하드웨어 실행 효율, 그리고 멀티노드·네트워크·게이트웨이 운영까지 end-to-end로 연결해 ‘호출 가능’ 수준에서 ‘운용 가능’ 수준으로 전환하는 실무 경로를 제시합니다. 목표는 품질-비용-지연-처리량-안정성-보안의 균형을 토큰 공급 안정성까지 포함해 달성하는 것입니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 모델별 능력의 경계를 정량화하고 (2) 요청이 들어올 때 비용-품질 제약을 만족하도록 라우팅/캐스케이딩 결정을 내리며 (3) 서빙 시 지연 꼬리(tail latency)와 안정성 리스크를 제어하는 것입니다. 논문은 capability boundary quantification(표준 평가+런타임 관찰로 실패/성공 범위 추정), intelligent routing(semantic/rule, learning-based discriminator, 고성능 보완 라우팅)과 model cascading(try-then-judge 구조의 검증 신호) 같은 구성요소를 운영 게이트에 맞춰 정리합니다. 또한 요청 파이프라인이 Prefill/Decode/KV cache/툴 호출/게이트웨이 스케줄링/폴백으로 이어지는 점을 고려해, 네트워크·컴퓨트 단에서도 동적 batching, KV cache 클러스터 스케줄링, graceful degradation 같은 보정 장치를 묶어 설계합니다.

- **Empirical Impact**: 실증적 근거로는 라우팅/캐스케이딩 관련 연구와 산업 제품 사례에서 비용 절감 및 품질 유지 가능성이 반복적으로 관찰된 점을 들며, 예컨대 FrugalGPT류는 강한 모델 성능에 근접하면서 최대 98%까지 비용 절감을 보고하는 흐름이 소개됩니다. 또한 Mix of routing 전략이 상용화돼 Bedrock Intelligent Prompt Routing, Martian Model Router, OpenRouter, LiteLLM 같은 게이트웨이에서 비용-품질 트레이드오프를 운영 레벨 기능으로 제공하는 사례가 정리됩니다. 결론적으로 본 아키텍처는 토큰 생산 비용을 낮추고(비용 우선 라우팅), 검증 기반 상향 호출로 안정적 공급을 보장(quality gate/fallback)하며, 에이전트·툴 호출 워크로드까지 ‘운용 지향’ 최적화로 확장하는 데 의미가 큽니다.



### Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifac (https://arxiv.org/abs/2606.20205)
- **Prior Approaches**: 기존 연구는 LLM의 성향을 인간 차별심리학의 자기보고 설문과 행동 과제로 “그대로” 측정해 심리 프로필을 만들었습니다. 그러나 LLM 응답이 프롬프트/문항 표현 변화에 민감하고 요인구조·행동 일치가 약하다는 의심이 누적돼, 프로필이 실제 특성(trait)인지 잡음/편향인지 해석이 흔들려 왔습니다.

- **Core Contribution**: 이 논문은 LLM 측정에서 나타나는 체계적 응답을 trait과 response bias(방향성 응답 편향)로 분해하는 정형(형식) 심리측정 프레임워크로 재해석합니다. 특히 문항 키잉(정방향/역방향)처럼 trait과 bias가 서로 반대 방향으로 작동하는지의 정도를 response orthogonality(응답 직교성)로 정의하고, 이를 중심으로 기존 “심리 프로필”의 타당성을 재검증합니다.

- **Technical Challenges**: 핵심 난제는 LLM 응답이 실제로 심리 특성에서 오는지, 아니면 문항 옵션·척도 방향 같은 형식에서 오는지 분리해내는 것입니다. 연구진은 IPIP-NEO-300(성격)과 Frey et al. 위험선호 배터리(성향) 29개 계측도구를 56개 instruction-tuned LLM과 1만여 명급 인간 기준표본에 동일 방식으로 적용하고, forward/reverse 평균 상관 및 분산 분해로 trait-편향 혼선을 정량화합니다.

- **Empirical Impact**: 결과적으로 LLM 간 차이는 81–90%가 response bias로 설명되고 인간은 9–16% 수준에 그쳤으며, capability를 키워도 bias는 완전히 사라지지 않았습니다. 또한 문항의 response orthogonality가 낮을수록 Cronbach’s alpha 같은 “신뢰도”가 부풀려지고, 직교성이 높아지면 신뢰도가 0에 가깝거나 음수로 붕괴했습니다. 더 나아가 프로필은 forward-키잉 vs reverse-키잉에 따라 크게 달라지며 문항 선택(item selection)만으로도 조작 가능해, LLM의 심리 프로필을 현재 방식 그대로 고정된 성질로 취급하면 위험하다는 메시지를 실증적으로 제공합니다.



### NAMESAKES: Probing Identity Memorization in Text-to-Image Models (https://arxiv.org/abs/2606.20155)
- **Prior Approaches**: 기존 T2I 프라이버시/멤버리제이션(Identity memorization) 평가는 후보군의 기준 사진(ground-truth), 학습 데이터 접근, 또는 모델 내부(white-box) 정보가 필요해 적용 범위가 좁았습니다. 판별 모델/공격 기법은 주로 생성 이미지 단위 또는 특정 샘플 재현 여부를 보려 하지만, “이름-정체성” 수준의 전반적 유출을 일관되게 가늠하기 어렵다는 한계가 있었습니다. 또한 분별이 되더라도 모델 불특정(architecture-agnostic) 환경에서는 재현성 확보가 쉽지 않았습니다.

- **Core Contribution**: 이 논문은 완전한 블랙박스(fully black-box), 레퍼런스 사진 없이도 개인 이름에 대해 “정체성 재현(Identity memorization)”과 “그럴듯한 생성(Identity fabrication)”을 구분하는 behavioral probe를 제안합니다. 핵심은 얼굴 유사도(멤버리제이션의 기준)를 직접 쓰지 않고도 생성들의 일관성/분산 패턴으로 이를 예측한다는 점입니다. 아울러 벤치마크로 NAMESAKES(1,269개 이름-얼굴 쌍, 인지도 스펙트럼 포함)와 perturbed name 세트를 만들어 테스트/비교를 가능하게 했습니다.

- **Technical Challenges**: 어려움은 (1) GT 얼굴 없이도 멤버리제이션을 추정해야 하고, (2) 모델별 생성 다양성 차이까지 흡수해야 하며, (3) 단일 이름만으로 “실제 인물 vs 비기억 기본 얼굴”을 구분할 신호를 찾아야 한다는 데 있습니다. 저자들은 생성 4장을 seed 변화로 뽑아 ArcFace 임베딩을 만든 뒤, (a) 같은 이름의 생성들이 얼마나 일관된 얼굴을 띠는지 보는 dispersion 기반 점수 δ, (b) 다른 이름들의 생성 중심(centroid) 대비 얼마나 벗어나는지 보는 centroid distance 기반 점수 sc을 결합해 참조 유사도와의 상관을 학습/추정합니다.

- **Empirical Impact**: SDXL-base, SDXL-Turbo, Flux1-Dev, Flux1-Schnell 등 SOTA T2I 모델 4종에서 연속 예측(R^2)과 이진 분리(AUC/정확도) 모두 유의미하게 성능을 보였습니다. 특히 SDXL-base는 reference similarity 연속 예측에서 R^2≈0.58, real vs perturbed 분리에서 AUC≈0.86(Acc≈0.79)로 가장 강했고, 고명성 구간에서는 AUC≈0.95 및 90%+ 정확도까지 상승했습니다. 즉, 레퍼런스/내부 접근 없이도 이름 기반 정체성 유출 가능성을 실무적으로 스크리닝할 수 있으며, 모델 계열 간(예: SDXL vs Flux) 멤버리제이션 양상 차이도 관찰된다는 점에서 프라이버시 감사와 unlearning 평가에 직접적인 의미가 있습니다.



### Learning to Prompt: Improving Student Engagement with Adaptive LLM-based High-School Tutoring (https://arxiv.org/abs/2606.20138)
- **Prior Approaches**: 기존 LLM 튜터링은 대체로 정적 prompting과 단일 과목(예: 수학) 중심 검증에 머물러 있었고, 시뮬레이션 기반 학습은 현실의 분포 이동과 데이터 희소성, 상호작용 복잡성을 놓치기 쉽습니다. 또한 prompt engineering이나 전역 최적화는 전형적으로 전 과목 공통의 고정 프롬프트로 수렴하는 경향이 있습니다.

- **Core Contribution**: 이 논문은 과목(topic/subject) 정보를 입력으로 받아 교육 전략을 선택하는 subject-aware prompt routing 프레임워크를 제안합니다. 동시에 14개 교육학적 관찰 기준(예: scaffolding, 이해도)을 분해해 산출하는 LLM evaluator 점수를 활용해, 과제 점수처럼 지연되는 신호 없이도 학습 효과의 즉시 대리지표를 만듭니다.

- **Technical Challenges**: 핵심 기술 난제는 시뮬레이터에서 학습한 라우터가 실제 학생의 행동에 과적합되는 sim-to-real gap을 줄이는 것이며, 이를 위해 점수 분포를 보정하는 sigmoid calibration과 score smoothing을 도입합니다. 또 사전학습 임베딩의 표현 붕괴를 막기 위해 topic embedding과 learnable subject ID embedding을 결합하는 dual-path 입력을 쓰고, 라우팅은 contextual bandit 형태로 PPO 기반 actor-critic(단일 스텝 학습)로 최적화합니다.

- **Empirical Impact**: 시뮬레이션 벤치마크에서 제안한 라우터는 정적 기준선 대비 더 높은 성능을 보였고(0.694 vs 0.647/0.64, p<0.001), 실제 온라인 A/B 테스트(N=656 대화, 359명)에서도 sim-to-real 전이가 관찰됐습니다. 특히 greedy 라우터는 conversion rate가 기준선과 유사했지만(19.1% vs 19.6%), stochastic 라우터는 conversion rate를 더 높였고(28.1%), 상호작용 턴 수는 약 3턴 줄이면서도 교육 품질을 유지했습니다(p=0.007).



### PASQA: Pitch-Accent-Focused Speech Quality Assessment Model Trained on Synthetic Speech with Accent Errors (https://arxiv.org/abs/2606.20137)
Comments:
          Accepted to INTERSPEECH 2026

- **Prior Approaches**: 기존 MOS(Mean Opinion Score) 예측 모델은 발화 단위 전반적 자연스러움 점수에 초점이 맞춰져, 일본어의 pitch-accent처럼 국소적 오류를 세밀하게 반영하지 못하는 한계가 있었다. 일부 프레임 단위 품질 예측 연구도 있었지만, pitch accent의 ‘정확성’ 자체를 타깃으로 삼지는 않았다. 또한 TTS 내부의 accent 예측 모듈/라벨을 직접 활용하기 어려운 black-box 환경에서는 음성 신호만으로 평가해야 한다.

- **Core Contribution**: 이 논문은 pitch-accent 정확도에 직접 민감한 Pitch-Accent-focused Speech Quality Assessment(PASQA)를 제안한다. 핵심 아이디어는 accent nucleus 위치 오류의 비율로부터 pseudo accent-quality score를 만들고, 학습 시 모델이 오류 ‘심각도 순서’를 유지하도록 유도하는 것이다. 이를 통해 전통적 발화 단위 자연스러움 MOS 예측의 국소 오류 둔감성을 보완한다.

- **Technical Challenges**: 도전 과제는 실제 데이터에서 pitch-accent 오류 라벨을 얻기 어렵다는 점이다. 저자들은 accent-controllable TTS로 nucleus 위치를 제어해 일본어 accent-error 데이터를 합성하고, mora 시퀀스 보조 입력, ranking loss(순서 학습), frame-level accent-error localization 보조 헤드, speaker-invariant을 위한 gradient reversal( GRL )로 모델을 구성했다. 이렇게 temporal 위치 정보와 발화/화자 편향을 동시에 통제해 국소 오류 감지 성능을 끌어올렸다.

- **Empirical Impact**: 실험 결과, 기존 MOS 예측 모델은 accent-error 심각도 순서를 거의 맞추지 못하고 상관도도 낮은 반면 PASQA는 seen/unseen 화자 모두에서 ordering accuracy와 인간 판단 일치를 크게 개선했다. 또한 OOD TTS(GPT-4o-mini-TTS)에서도 grapheme/mora 입력 차이를 반영한 pairwise discrimination 정확도가 가장 높아, accent-quality 변화에 대한 민감도와 적용 가능성을 입증했다. 결론적으로 PASQA는 일본어 pitch-accent 품질 평가에서 ‘국소 오류를 반영하는’ 실용적 대안으로 의미가 크다.



### What Makes Effective Supervision in Latent Chain-of-Thought: An Information-Theoretic Analysis (https://arxiv.org/abs/2606.20075)
- **Prior Approaches**: 기존 Latent Chain-of-Thought(잠재 CoT)은 최종 정답만으로 잠재 추론을 유도하려는 outcome supervision(OS)과, 중간 합리화를 넣는 process supervision(PS) 계열로 나뉜다. 그러나 OS는 학습 신호가 희박해 잠재 궤적에서 gradient attenuation이 일어나며, 그 결과 의미적 드리프트(semantic drift)가 커져 잠재 추론이 실패하기 쉽다. PS는 이를 보완하지만, 신호를 어떻게 잠재 공간에 정착시키는지에 대한 원칙이 부족해 방법 간 성능 격차가 크다.

- **Core Contribution**: 이 논문은 Latent CoT 실패를 정보이론 관점에서 ‘dual collapse’로 정리한다: 최적화 경로에서의 gradient attenuation과 잠재 공간에서의 representational drift가 동시에 발생한다. 또한 process supervision을 trajectory supervision(시간적 단계 신호)과 space supervision(잠재 공간의 의미 구조 보존)으로 분해하고, 두 신호가 정보 보존을 어떻게 좌우하는지 프레임을 제시한다. 특히 단순한 geometric imitation보다 mutual information(상호정보) 최대화 쪽으로 관점을 전환해야 한다고 주장한다.

- **Technical Challenges**: 핵심 난제는 잠재 상태가 관찰 불가능한데다 최종 정답 감독만으로는 각 잠재 단계가 ‘정답에 기여하는 논리’인지 보장하기 어렵다는 점이다. 저자들은 이 문제를 해결하기 위해 (1) trajectory supervision으로 다음 명시적 추론 단계를 예측하는 dense stepwise 신호를 넣어 시간 정보 감쇠를 막고, (2) space supervision에서는 semantic drift를 억제하되 rigid한 기하 압축 대신 generative reconstruction이 더 유리하다고 분석한다. 더 나아가 Unified Latent Probe(ULP)를 제안해 잠재 궤적과 명시적 reasoning step 사이의 mutual information을 재구성 기반 진단값으로 계량화한다.

- **Empirical Impact**: 실험 결과는 ‘Information-Performance Binding’으로 요약되는데, 추론 정확도는 잠재 사슬에서 보존되는 정보의 fidelity(정보 보존 정도)에 강하게 의존한다. OS 계열은 ULP 관점에서 재구성/상호정보가 낮아 추론 정확도 개선이 거의 없고, PS-trajectory는 먼저 의미 정보가 잠재 궤적에 들어오게 만들며, PS-GR(trajectory + generative reconstruction)은 정보 보존 프론티어를 가장 잘 밀어 올린다. 즉, 잠재 CoT 학습을 ‘기하 강제 정렬’이 아니라 ‘정보 용량 보존과 mutual information 최대화’로 설계해야 한다는 실증적 가이드를 제공한다.



### Generative Engine Optimization at Scale: Measuring Brand Visibility Across AI Search Engines (https://arxiv.org/abs/2606.20065)
Comments:
          14 pages, 4 tables; v1.0 preprint

- **Prior Approaches**: 기존 SEO의 연장선이지만, AI 검색에서는 “키워드 순위”가 아니라 LLM이 브랜드를 실제로 인용·추천하느냐가 핵심이 됐습니다. 과거 GEO 관련 연구는 주로 인용을 끌어내는 콘텐츠 신호(예: 인용 가능한 근거, provenance)를 다뤘고, conversational SEO 전술은 대부분 효과가 제한적이며 오히려 해가 될 수 있음을 보여줬습니다.

- **Core Contribution**: 이 논문은 Ranqo를 통해 5개 AI 엔진의 프롬프트 응답을 대규모로 측정해, 브랜드의 AI Visibility를 “어떻게 인용되는가”까지 포함한 실증 지표로 정리합니다. 특히 권위 있는 상위 브랜드를 제외한 SMEs, D2C, 크리에이터, 초기 스타트업처럼 ‘비교적 덜 노출되는 집단’에서의 인용 격차를 정량화해, GEO의 기준선(baseline) 데이터를 제공합니다.

- **Technical Challenges**: 해결해야 할 어려움은 엔진마다 소스 선택·재검색 방식이 달라 비교 가능성을 유지하는 것과, 감정(sentiment)처럼 노이즈가 큰 신호를 안정적으로 분해하는 데 있습니다. 논문은 unbranded 카테고리 프롬프트 비중을 통해 ‘브랜드가 스스로 호명되는 효과’를 줄이고, 페이지 감사(6개 차원)와 인용 근거(출처 타입/관계)를 결합해 플랫폼별 차이를 측정하도록 설계했습니다.

- **Empirical Impact**: 생산 데이터(102개 브랜드, 3,500+ 런, 10만+ 응답)에서 브랜드 위상에 따라 인용·추천 확률이 3단 사다리로 갈라지며, 첫 실행 기준으로 Tier 1(73%)→Tier 2(44%)→Tier 3(11%)처럼 단계당 약 30%p 격차가 관찰됩니다. 또한 인용은 기업(official) 사이트로 집중되는 경향이 있고, 최고 레버리지는 ‘best-of’ 리스트클처럼 랭크형 콘텐츠(전체 인용의 약 21%)로 나타나며, sentiment는 mention 여부보다 훨씬 더 자주 뒤집힌다는 점이 GEO 측정의 실무 가이드를 제공합니다.



### When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents (https://arxiv.org/abs/2606.20023)
Comments:
          code: this https URL

- **Prior Approaches**: 기존 에이전트 안전 연구는 주로 prompt injection, tool injection, jailbreaking 같은 공격/정책 위반 출력에 초점을 맞췄고, 도구 선택 편향은 provider identity나 description 같은 메타데이터 선호 문제로 주로 다뤄졌다. 또 privilege escalation을 외부 조작에 의한 결과로 보거나, 시스템 차원의 권한 경계 강제처럼 에이전트 내부의 ‘최소 권한 선택’ 성향은 상대적으로 덜 검증됐다. 그 결과, 낮은 권한의 충분한 대안이 있는데도 높은 권한 도구를 고르는 행동(과권한 선택)이 얼마나 흔한지 정량화가 부족했다.

- **Core Contribution**: 논문은 과권한 선택(over-privileged tool selection)을 “충분한 낮은-privilege 도구가 가능한데도 더 높은 privilege를 직접 선택하거나 일시적 실패 뒤에 승격하는” 문제로 정의한다. 이를 평가하기 위한 벤치마크 ToolPrivBench를 제안해 초기 선택뿐 아니라 transient failures 이후 escalation까지 함께 측정한다. 또한 일탈(과권한) 원인을 ‘도구 성능 부족’이 아닌 ‘권한 선호 편향’으로 분리하도록, 각 도구가 비오류 조건에서 독립적으로 충분(sufficient)하도록 시뮬레이션 환경을 설계한다.

- **Technical Challenges**: 핵심 기술적 난관은 높은 권한을 쓰는 것이 단순히 낮은 권한 도구가 실패해서 ‘합리적’ 선택일 수 있다는 혼동(confound)을 제거하는 것이다. 논문은 함수적 충분성 조건을 엄격히 강제하고, Gemini 2.5 Pro와 GPT-5.2의 교차 합의 심사로 도구 sufficiency를 1차 검증한 뒤 인간 전문가 감사로 애매한 권한 대비나 표준 도구의 부족 사례를 제거한다. 그 위에 transient, privilege-unrelated 오류(예: 연결 에러)를 표준 도구 호출에 주입해 “실패가 오면 에이전트가 더 큰 권한으로 도피하는지”를 자연스럽게 관찰하고, OPUR@k와 PED로 공격적 선택/성급한 승격을 구분해 평가한다.

- **Empirical Impact**: 실험 결과 ToolPrivBench의 8개 도메인·5개 위험 패턴에서 주류 LLM 에이전트 다수가 최소 권한 대안이 남아있는데도 높은 권한 도구를 선택/승격하는 현상이 흔했다. 특히 transient failure가 발생하면 OPUR이 더 크게 증폭됐고, 일반적인 safety alignment는 least-privilege 도구 선택으로 잘 전이되지 않았으며 prompt-level 제어는 transient 실패 상황에서 완화 효과가 제한적이었다. 이를 해결하기 위해 privilege-aware post-training 방어(표준 도구 우선, 필요할 때만 escalation)를 제안했고, GRPO 기반 학습을 통해 불필요한 high-privilege 사용을 크게 줄이면서 전반 성능(MMLU, GSM8K, MetaTool)은 대체로 유지되는 것으로 나타났다. 다만 시뮬레이션과 소규모 도구 묶음의 ‘독립 충분’ 조건 때문에 실제 운영 환경의 복잡한 멀티-툴/장기 지평을 완전히 대변하진 못하며, 향후 더 긴 시퀀스와 더 큰 도구 인벤토리로 확장이 필요하다는 한계도 제시한다.



### Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning (https://arxiv.org/abs/2606.20002)
Comments:
          Work in progress; we will continuously update the codebase and arXiv version

- **Prior Approaches**: 기존 연구는 lifelong agent를 위해 persistent memory나 skill 세트 같은 컨텍스트 업데이트 메커니즘을 주로 사람이 설계해 왔고, LLM 자체를 CoD(Connect the Dots) 메타-역량까지 end-to-end로 후학습하는 체계는 부족했습니다. 또 standard task-by-task RL은 각 작업을 독립적으로 학습해 장기 배치에서 “이전에 얻은 환경 지식으로 다음 작업을 더 잘 푸는” 목표와 정렬이 약하다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 LLM 에이전트가 장기 실행 환경에서 관련된 작업들을 연속 수행하면서 탐색하고, 자기 경험을 바탕으로 컨텍스트를 갱신해 다음 작업 성능을 점진적으로 끌어올리는 CoD 메타-역량을 명시적으로 학습시키는 일반 프레임워크를 제안합니다. CoD-Train은 CoD-Deploy와 동일한 롤아웃 패턴(풀기-컨텍스트 업데이트 에피소드 교차)을 RL 후학습에 그대로 반영해, “문제 해결”과 “환경 맥락 학습”을 모델 가중치 수준에서 함께 길들이도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 긴 상태-행동 시퀀스 안에서 solve-task와 update-context가 섞여 있을 때의 credit assignment(어떤 업데이트가 미래 보상을 키웠는지)를 안정적으로 계산하는 것입니다. 논문은 Bellman 원리로 각 에피소드가 즉시 보상뿐 아니라 미래 solve-task 보상을 동시에 최대화하도록 목표를 재정의하고, 여러 작업 보상이 들어간 시퀀스를 GRPO-style 알고리즘에 맞게 advantage 계산(에피소드 위치별 그룹화)하는 방식으로 해결합니다.

- **Empirical Impact**: 실험은 Qwen3-8B-Instruct를 CoD-Train에 적용해 FrozenLake-Obscure 같은 환경에서 “첫 작업(컨텍스트 없음) 성능 한계”와 달리, 시퀀스의 뒤 작업은 컨텍스트 갱신 덕분에 크게 향상됨을 보여줍니다(예: 1회성 성공률 상승보다 4번째 작업 성능 상승이 두드러짐). 또한 학습 도메인 내 더 어려운 인스턴스, 도메인 간(Alchemy-Random·TerminalSimulator) 평가, 그리고 Ralph-loop 설정으로의 out-of-distribution generalization 가능성을 실증하며, 장기 에이전트 학습 파이프라인에 새로운 방향을 제시합니다.



### Segment-Level Mandarin Chinese Speech-Based Cognitive Impairment Detection via an Autoencoder with Contrastive Learning (https://arxiv.org/abs/2606.19996)
Comments:
          15 pages, 7 figures, 5 tables

- **Prior Approaches**: 기존 음성 기반 인지장애 탐지는 MFCC·prosodic feature 같은 수공학적 특징과 SVM/RF/GMM 조합에 의존하는 경우가 많았고, 데이터셋 간 일반화가 약하다는 한계가 있었다. 딥러닝 이후에는 CNN/RNN/Transformer로 end-to-end 분류를 시도했지만, labeled recordings·speaker 수가 적거나 녹음이 짧게 분절되는 segment-level 설정에서 과적합과 도메인 이동(domain shift) 문제가 지속됐다. 또한 다수 연구가 AD vs CN 중심이거나 영어/크로스-링구얼에 치우쳐 MCI vs CN의 미세한 차이를 안정적으로 가르는 연구와, 만다린 다중 데이터셋 검증은 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 segment-level 로그-멜 스펙트로그램을 입력으로 하는 표현학습 프레임워크를 제안한다. GRU 기반 autoencoder(AE)에 reconstruction 목표를 두고, 동시에 supervised contrastive learning으로 잠재공간에서 인지 상태(class) 간 분리성을 강화해 분류에 바로 쓰일 수 있는 latent representation을 만든다. 더 나아가 offline과 online spectrogram augmentation을 함께 설계해 저자원 환경에서도 견고한 표현을 학습하도록 했다.

- **Technical Challenges**: segment-level에서는 각 샘플이 국소 문맥만 담아 안정적인 인지 관련 신호를 충분히 대표하기 어렵고, labeled data가 적어 overfitting 위험이 커진다. 이를 위해 (1) GRU-AE가 스펙트로그램의 시간 의존성과 전역 spectro-temporal 구조를 reconstruction으로 보존하게 하고, (2) 온라인 스펙트로그램 마스킹 두 관점을 만들어 SupCon의 positive/negative 쌍 정의가 잘 되도록 했으며, (3) offline Augmentation으로 학습 데이터 분포를 사전에 확장해 generalisation을 끌어올렸다. 학습 시에만 decoder와 supervised contrastive 목적을 사용하고, 추론 단계에서는 encoder와 분류 헤드만 남겨 효율을 유지한 점도 설계 포인트다.

- **Empirical Impact**: 만다린 중국어 음성 4개 데이터셋에서 binary와 three-class(AD/MCI/CN) 모두에 대해 안정적이고 경쟁력 있는 성능을 보였고, 특히 임상적으로 더 어려운 three-class 설정에서 유의미한 개선이 관찰됐다. ablation study도 AE+SupCon의 결합과 offline/online augmentation이 성능에 실질적으로 기여함을 지지한다. 결과적으로 자원 제약이 큰 임상 현장에서도 적용 가능한 segment-level speech representation learning 전략으로서, 인지장애 선별의 확장성과 실용성을 강화했다.



### Investigating Human-Model Discrepancies in Speech Quality Assessment via Acoustic and Prosodic Perturbations (https://arxiv.org/abs/2606.19951)
Comments:
          Accepted to INTERSPEECH 2026

- **Prior Approaches**: TTS 품질을 사람 대신 빠르게 재는 MOS 예측 모델이 널리 쓰이지만, 기존 연구들은 모델이 인간의 지각 차원을 동일하게 반영하지 못할 수 있음을 시사해 왔다. 특히 SSL-MOS, UTMOS 같은 self-supervised 기반 모델은 음향 열화에 강하게 반응하는 반면, prosody(억양/악센트 오류)나 화자 특성의 미세한 변화에는 둔감할 수 있다는 의심이 누적돼 왔다.

- **Core Contribution**: 이 논문은 MOS 예측의 한계를 “어떤 품질 차원에서” 발생하는지 통제 실험으로 분해해 확인한다. 음향 열화, linguistically meaningful prosodic errors(일본어 pitch-accent 패턴 오류), 화자별 pitch·speaking rate 변형을 각각 독립적으로 가한 뒤, 인간 MOS와 모델 MOS의 차이를 정밀 분석한다.

- **Technical Challenges**: 문제는 (1) MOS 예측 모델이 scalar MOS로 압축된 표현에서 prosody/화자 정보를 실제로 얼마나 유지하는지, (2) 화자 관련 신호가 모델의 학습 데이터 분포(통계적 상관)로 인해 인간 지각과 다르게 매핑될 수 있다는 점이다. 저자들은 일본어 pitch-accent 언어라는 조건에서 악센트 오류를 의도적으로 스왑해 관측하고, VERSA로 다수 MOS 모델(SHEET-MB/BV, UTMOS, UTMOSv2, NISQA, DNSMOS)을 동일 파이프라인으로 평가해 민감도 차이를 비교한다.

- **Empirical Impact**: 결과적으로 대부분의 모델은 clipping·잡음·MP3 같은 신호 수준 음향 열화에는 인간과 유사하게 점수 저하를 추적했지만, 악센트/억양 오류에는 모든 모델이 거의 변하지 않았다(인간 점수는 크게 하락). 화자 특성에서는 double dissociation이 관찰돼, 모델은 mean F0 편향에는 민감하지만 인간이 체감하는 speaking rate·F0 variability 변화에는 둔감해 인간의 지각 구조와 어긋남을 보여주며, 이는 “음향 fidelity를 넘어서는 MOS 예측의 한계”를 실증적으로 강화한다.



### Multi-Agent Transactive Memory (https://arxiv.org/abs/2606.19911)
- **Prior Approaches**: 기존 retrieval-augmented generation(RAG)은 인간이 쓴 문서를 중심으로 한 에이전트 단일 사용 맥락의 보강에 강점이 있지만, 에이전트들이 만든 절차적 산출물은 보통 한 번 쓰고 폐기되거나 생산한 에이전트에만 남겨집니다. reasoning/thought reuse 계열은 비용·효율을 개선해도 “생산자 중심 재사용”에 머물러 새로 투입된 에이전트가 이미 존재하는 해결책을 다시 찾아야 하는 문제가 남아 있습니다. 또한 transfer learning·knowledge distillation은 도메인 정렬이나 추가 학습이 필요해, 오픈 생태계의 이질적·동적으로 생성되는 에이전트 집단에 바로 쓰기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 Multi-Agent Transactive Memory(MATM)라는 인프라로, 에이전트 집단이 생성한 action-observation trajectory를 공유 저장소에 누적하고 소비 에이전트가 이를 조회해 성능을 개선하도록 제안합니다. producer(생성)와 consumer(소비) 역할은 고정이 아니라 문맥에 따라 바뀌며, 조회된 궤적을 통해 “인구 단위 경험 재사용”을 가능하게 합니다. 특히 interactive 환경에서 긴 상호작용 궤적이 담는 절차 지식을 핵심 아티팩트로 삼아, 개인의 재탐색을 집단의 누적 지식으로 전환하려는 설계가 중심입니다.

- **Technical Challenges**: 가장 큰 도전은 상태가 조건인 궤적을 검색하는 일입니다. MATM은 최근 l단계 상호작용을 retrieval key로 쓰고 다음 구간을 value로 저장하는 state-conditioned key-value 인덱싱을 채택해, 쿼리와 현재 상태에 맞춘 “이어가기” 형태의 guidance를 제공합니다. 더 나아가 단순 임베딩 유사도만으로 뽑힌 후보를 learning-to-rank(LTR) reranker로 재정렬하며, producer 메타데이터(신뢰/품질), consumer 메타데이터(개인화) 등 여러 특성을 포함해 marginal utility(무조회 대비 성능 기여) 기준으로 라벨링·학습합니다.

- **Empirical Impact**: ALFWorld와 WebArena에서 MATM 조회는 평균 task 성공률과 상호작용 단계 수 효율을 함께 개선하며, 별도 조정이나 joint training 없이도 downstream 성능 향상이 나타납니다. ALFWorld에서는 성공률이 47%→55%로 상승하고 단계도 11.77→11.18로 줄어드는 등 효과가 뚜렷했고, WebArena에서도 개선은 더 완만하지만(성공률 18%→20%, 단계 22.0→20.3) RPP가 양수로 전환되며 Pareto 측면의 우위가 관측됩니다. 또한 reranking은 환경별로 이득 폭이 달랐지만(예: ALFWorld에서 SVMRank 강세), 전반적으로 “저장소가 커질수록”과 “집단 내 다양한 능력대 에이전트가 함께” 이득을 확장할 수 있음을 보여 오픈 에이전트 생태계에서의 경험 공유 설계 패턴으로 위치시킵니다.



### JAMER: Project-Level Code Framework Dataset and Benchmark on Professional Game Engines (https://arxiv.org/abs/2606.19830)
- **Prior Approaches**: 기존 AI 게임 생성은 주로 아트 자산과 간단한 게임 규칙·로직에 집중했지만, 프로 게임 엔진에서의 ‘프로젝트 수준 코드 프레임워크’ 생성은 데이터와 평가 방식 부재로 크게 다뤄지지 않았다. 기존 평가는 단일 파일/웹 게임 중심이거나, 수작업 테스트·주관적 VLM/LLM 채점·일반 unit testing처럼 게임 런타임 상호작용을 충분히 반영하지 못했다. 또한 대규모 오픈소스 저장소에서 양질의 프로젝트를 가려내는 객관적 필터링이 어려워 연구용 데이터셋 구축이 막혔다.

- **Core Contribution**: 이 논문은 Godot 기반으로 ‘프로젝트 레벨 게임 코드 프레임워크’ 데이터셋 JamSet과 벤치마크 JamBench를 제안한다. Game Jam에서 공개되는 오픈소스 프로젝트를 대량 수집하되, Godot의 텍스트 포맷과 headless 실행을 활용해 결정론적(deterministic) 검증·평가 파이프라인을 만든다. 그 결과 240K 저장소 후보 중 8,133개를 검증해 학습 데이터와, 수작업 검증을 거친 300개로 벤치마크 평가를 구성한다.

- **Technical Challenges**: 핵심 난제는 (1) 수천 개의 다양한 게임을 그래픽 없이 실행해 런타임 거동을 수집하면서도 (2) 결과 재현성을 보장하는 결정론적 평가를 만드는 것이다. 저자들은 L1 파일 무결성, L2 컴파일 정확성, L3a 실행 안정성, L3b 행동 수집의 4단계 검증으로 ‘침묵(silent)·크래시·불호환’ 이슈를 체계적으로 걸러낸다. 행동 평가는 Structural Completeness Score(SCS)와 Behavioral Alignment Score(BAS)로 정적 구조와 런타임 유사성을 동시에 측정하며, eval_config로 생성되는 결정론적 입력 전략을 통해 BAS를 계산한다.

- **Empirical Impact**: 실험에서 과제 규모가 커질수록 성능이 급격히 붕괴하는 ‘capability cliff’가 확인되며, 런타임 pass rate가 작은 프로젝트 80.4%에서 큰 프로젝트 5.7%로 떨어졌다. Code Agent는 컴파일 통과율은 크게 올리지만 SCS·BAS 같은 런타임 품질에는 실질 개선이 없어서, 병목이 문법/수정이 아니라 아키텍처 설계 역량에 있음을 시사한다. 마지막으로 JamSet로 fine-tuning을 수행하면 입력 추상화·전역 상태 관리(autoload) 등 인간다운 공학 관행이 더 잘 반영되어 데이터셋의 학습 효율이 검증된다.



### Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning (https://arxiv.org/abs/2606.19808)
- **Prior Approaches**: 기존 연구는 chain-of-thought, self-consistency처럼 추론에 더 많은 연산을 투입해 성능을 끌어올리는 방식(추론 확장/탐색)을 주로 다뤘다. 또 outcome/process verifiers나 단계/상태별 selective verification처럼 검증을 세밀하게 제어하려는 접근이 있지만, 추가 검증기·검색 상태·세분 제어가 필요해 배포 복잡도가 커진다. 무엇보다 “한 번 더 생각하기”가 항상 이득이 아니라, 맞는 답을 망치거나( harmful flip) 불필요하게 비용을 늘릴 수 있어 배치(배포) 관점의 제어가 요구됐다.

- **Core Contribution**: 이 논문은 post-generation 추론을 ‘새 verifier를 만드는 문제’가 아니라, 서빙 레이어에서 “첫 시도 답을 보존할지/active verification을 호출할지”를 결정하는 deployment allocation 문제로 재정의한다. SeVRA(Selective Verification for Reasoning Allocation)는 frozen solver가 만든 초기 답과 serving-visible attempt state를 보고 recoverability(복구 가능성)를 추정해 개입을 선택한다. 또한 도움 수정(helpful fix)과 악화 수정(harmful flip)을 함께 고려하는 정책 학습/평가 틀을 제공한다.

- **Technical Challenges**: 핵심 난제는 서빙 시점에 정답 라벨이나 숨겨진 상태 없이도 active verification이 ‘도움이 될지’ 예측해야 한다는 점이다. 이를 위해 Qwen3-4B를 frozen 상태로 사용하며, 2,000개 MATH 데이터에서 base 시도 후 여러 개입(action)의 결과를 로그로 수집해 recoverability-aware gate를 학습하고, 임계값은 downstream 정책 성능과 액션 토큰을 기준으로 고정한다. “최대 토큰 예산”과 “실제 총 토큰(realized tokens)”을 분리해, 개입 횟수 절감이 전체 비용 절감으로 직결되지 않을 수 있음을 비용 지표 설계에 반영했다.

- **Empirical Impact**: MATH500에서 SeVRA의 selective active verification은 76.3% 정확도로 always verifying(75.5%)보다 좋고, 검증 토큰을 26.8% 줄이며 harmful flip을 2.2%→1.0%로 낮췄다. 반면 긴 초기 solve(8,192 토큰)는 76.0%로 통계적으로 비슷한 정확도를 내면서 총 realized 토큰은 28% 적어, math 비용 frontier에서는 “초기 예산 조정”이 더 효율적일 수 있음을 보여준다. GSM8K 전이에서는 SeVRA가 단 3.0%만 검증을 호출해 검증 토큰을 91.2% 줄이면서 93.4%→94.5%로 개선했고, CommonsenseQA에서는 항상 검증이 오히려 성능을 떨어뜨려 워크로드 의존성을 확인했으며 cheap feature gate가 learned gate들과 거의 비슷한 성능으로 배포 효율이 높다는 결론을 제시한다.



### CombEval: A Framework for Evaluating Combinatorial Counting in Large Language Models (https://arxiv.org/abs/2606.19788)
Comments:
          under review. Code: this https URL

- **Prior Approaches**: 기존 조합·확률(Combinatorial Counting) 평가는 GSM8K, MATH처럼 큰 수학 추론 데이터셋의 하위 범주 또는 정적 CO 전용 벤치마크(예: CombiBench)에 의존하는 경우가 많았다. 그러나 이런 정적 벤치마크는 데이터 오염(학습 데이터에 겹침)과 표면 패턴 편향으로 인해 실제 추론 능력보다 암기/피상적 추정이 성능을 부풀릴 위험이 있다.

- **Core Contribution**: CombEval은 LLM을 위한 동적(dynamic) 조합 카운팅 벤치마크로, 각 문제를 Cofola의 typed 명세(엔터티·조합 대상·의존성·제약)로 표현해 자연언어 문제를 “제어된 방식”으로 생성한다. 또한 Cofola 기반 정형화 뒤 백엔드에서 정확한 정답을 solver-verified로 검증해, 템플릿 매칭이 아닌 구조적 추론을 평가할 수 있게 한다.

- **Technical Challenges**: 핵심은 (1) 조합 카운팅 문제의 구조적 다양성을 포괄하면서도 (2) 생성 즉시 정확한 정답 검증을 대규모로 수행하고 (3) 난이도를 entity scale, 제약 개수, 추론 깊이 같은 파라미터로 예측 가능하게 조절하는 일이다. CombEval은 typed object-DAG 파이프라인과 제약-연산자 호환 규칙으로 Cofola 명세를 만들고, WFOMC로 컴파일되는 Cofola solver에서 시간 제한 내에만 안정적으로 풀리는 인스턴스만 필터링해 이 문제를 해결한다.

- **Empirical Impact**: 11개 LLM(오픈/클로즈드 포함)을 zero-shot 및 code-augmented 조건에서 평가했더니, 성능은 전반적으로 상위 모델에서 좋아지지만 ordered object, 구분 불가능(identical/indistinguishable) 원소, 비교적 positional 제약, 중첩된 객체 의존성에서 공통적으로 취약함이 관찰됐다. 특히 오류 분석은 제약 해석과 카운팅 원리(예: 함께(together) 블록의 순열/인접·연속 조건 오독, 멀티셋 원형의 몫 처리 누락 등)의 실패가 주요 원인임을 보여주며, CombEval이 “언제/왜” LLM이 조합 추론에서 깨지는지 진단하는 테스트베드로 의미가 크다.



### AgentFinVQA: A Deployable Multi-Agent Pipeline for Auditable Financial Chart QA (https://arxiv.org/abs/2606.19782)
- **Prior Approaches**: 기존 금융 차트 Q&A는 정확도 자체에 초점이 맞춰져 있고, 오답일 때 왜 틀렸는지 설명·검증이 불투명한 경우가 많습니다. 또한 외부 API 의존이 전제된 설계가 흔해 개인정보·데이터 레지던시 요구가 큰 규제 환경에는 바로 적용하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: AgentFinVQA는 단일 모델 추론 대신 계획(Plan)–OCR–legend grounding–시각 검토(Inspect)–검증(Verify)으로 질의를 분해하고, 각 단계 산출물을 per-sample Model Evaluation Packet(MEP)으로 기록해 감사를 가능하게 합니다. 더불어 verifier의 판정(CONFIRM/REVISE)과 신뢰 점수를 활용해 human-in-the-loop 검토 우선순위를 조절하는 흐름을 제안합니다.

- **Technical Challenges**: 핵심 난제는 (1) 텍스트(축·범례·라벨) 추출 오류와 (2) 범례-색상 매핑 혼동이 시각 추정까지 연쇄적으로 악화되는 점, 그리고 (3) 검증 단계가 정확한 수정인지 과도한 번복인지 가르는 점입니다. 논문은 OCR·범례를 구조화된 메타데이터로 다음 단계에 “증거”로 주입하고, stacked/pie 계열 일부 경우에 한해 deterministic colour-area 도구로 색상 기반 추정 단서를 추가하며, 독립 VLM 호출로 draft answer를 감사·판정한 뒤 confidence 게이트로 오버라이드를 억제합니다.

- **Empirical Impact**: FinMME에서 AgentFinVQA는 proprietary backbone(Gemini-3 Flash) 기준 model-matched zero-shot 대비 +7.68%p(71.24% vs 63.56%) 향상, open-weights(Qwen3.6-27B-FP8)로 로컬 서빙 시에도 +4.84%p 개선을 보여 API 의존 없이 이득이 유지됨을 입증합니다. 또한 exact accuracy 관점에서 verifier 판정이 “confirmed”와 “revised” 정답 품질 차이를 신호로 제공(68.2% vs 55.6%)하며, 실패의 상당 부분이 질문 오해·legend 혼동·추출 오류에 집중돼(약 2/3) 향후 개선 방향을 구체화합니다.



### Manifold Bandits: Bayesian Curriculum Learning over the Latent Geometry of Large Language Models (https://arxiv.org/abs/2606.19750)
Comments:
          Webpage: this https URL

- **Prior Approaches**: 기존 LLM 강화학습의 group-relative optimization은 한 프롬프트에서 여러 롤아웃의 보상 변동을 통해 정책 기울기를 얻는데, 보상 변동이 없으면 신호가 붕괴해 학습 효율이 급격히 떨어집니다. 이를 개선하려는 적응형 샘플링/커리큘럼은 대체로 각 프롬프트를 독립적인 arm으로 보고(밴딧) 중간 난이도 같은 단일 기준을 맞추거나, Dynamic Sampling처럼 보상 분산이 0이 아닌 프롬프트만 골라내려다 계산 오버헤드를 치르는 경향이 있습니다. 또한 문제 간 구조(유형 이질성, 표현 공간에서의 유사성)를 제대로 반영하지 못해 탐색이 비효율적이거나, 다양성(만ifold 커버리지)과 평가 관련성 사이의 균형을 놓칠 수 있습니다.

- **Core Contribution**: 이 논문은 문제 샘플링을 ‘엔도제너스(non-stationary)한 manifold-structured bandit 문제’로 재정의합니다. 모델이 업데이트되면 각 프롬프트의 보상 분포가 바뀌고, 그 변화는 샘플링 선택이 만든 피드백 루프로 진화하며, 프롬프트들은 모델의 잠재 표현 공간에서 서로 연관돼 정보가 전이됩니다. 이를 바탕으로 Bayesian Manifold Curriculum(BMC)을 제안해, Latent Task Tree(모델 임베딩 기반 계층 트리) 위에서 Bayesian 의사결정으로 학습 신호를 극대화하면서도 태스크 manifold 커버리지를 유지합니다.

- **Technical Challenges**: 핵심 어려움은 (1) 모델 내부 표현을 이용해 데이터의 구조를 학습 과정에서 재사용할 수 있는 형태로 근사하고, (2) endogenously 변하는 보상 환경에서 프롬프트별 ‘학습 신호’에 대한 확률적 믿음을 온라인으로 갱신하며, (3) 트리 상에서 하향(top-down) 선택과 상향(bottom-up) 정보 전파를 안정적으로 결합하는 것입니다. 논문은 policy 중간층 임베딩으로 Latent Task Tree를 재귀적으로 구성하고(국소 차원/연결성을 보는 Chart Test 포함), BMC에서는 프롬프트 단위 Gaussian(분산+불확실성) 믿음을 surprise 기반 Bayesian 필터로 갱신한 뒤, 정밀도 가중 집계에 subtree-level heterogeneity 보정을 더해 이질적 영역이 과신(과잉 평균)되지 않게 합니다. 결과적으로 트리의 관련성에 따라 관측이 인접 프롬프트의 선택 확률까지 연동되어, 정보 공유를 계산 효율적으로 수행합니다.

- **Empirical Impact**: 수학 추론(RLVR 기반) 설정에서 DAPO-Math-17K와 Qwen3-4B-Base/8B-Base를 사용해, 샘플링 전략을 생산성(productivity: 보상 분산/효과적 샘플링 비율), 다양성(diversity: latent manifold 커버리지/정보 공유), 유틸리티(utility: 평가 관련성) 축으로 진단합니다. 관찰에 따르면 reward variance와 effective ratio가 높을수록 학습 속도(pass@1)가 빨라지며, BMC는 Dynamic Sampling이 주는 학습 속도 이점을 상당 부분 재현하면서도 wall-clock 시간은 훨씬 덜 소모하는 경향을 보입니다. 또한 단순 난이도(구조 제거)나 고정 트리(업데이트 제거)만으로는 productivity–diversity–utility의 비대칭적 트레이드오프를 충분히 해결하기 어려워, ‘난이도만’으로는 강한 downstream 성능을 보장하기 어렵다는 메시지를 실증적으로 강화합니다.



### Benchmarking Agentic Review Systems (https://arxiv.org/abs/2606.19749)
Comments:
          11 pages, 7 tables, 4 figures

- **Prior Approaches**: 기존 LLM 기반 리뷰 연구는 작은 백본 모델이나 단일 프롬프트/부분 파이프라인에 머물러, 실제 공개 리뷰 시스템 전체를 공정하게 비교하기 어려웠습니다. 또한 평가가 모델 출력을 직접 보는 경우가 많아, “시스템 단위”에서 품질 신호와 오류 탐지를 함께 검증하기가 제한적이었습니다. 최근 멀티에이전트/루브릭 기반 접근도 있었지만, 무엇이 얼마나 잘 작동하는지에 대한 재검증이 필요했습니다.

- **Core Contribution**: 이 논문은 실제로 사용 가능한 리뷰 시스템(OpenAIReview, coarse, Reviewer3)과 zero-shot 기준을 여러 LLM(6개, 프론티어+경량)과 묶어 “시스템 전체”를 정면 비교합니다. 먼저 ICLR/NeurIPS 논문에서 AI 리뷰가 외부 품질 신호(인용, 수상/선정, 리뷰 점수 등)를 얼마나 따라가는지 상관을 측정합니다. 다음으로 알려진 오류를 주입한 perturbation benchmark로, 각 시스템이 오류를 실제로 얼마나 회수(recall)하는지 검증합니다.

- **Technical Challenges**: 핵심 난제는 (1) 논문 품질의 정답 라벨이 없다는 점과 (2) 오류가 실제로 “읽을 가치가 있는 진짜 오류”인지 통제해야 한다는 점입니다. 이를 위해 연구팀은 품질 프록시 기반의 low/high 페어링과, 8개 arXiv 분야·4개 오류 유형(로컬 수학 편집, 거짓 주장, 논리 오류, 실험/분석 오류)을 갖춘 perturbation benchmark를 설계했고, 여러 단계(추출-생성-구조 검증-체크리스트 검증-주입)로 오류의 유효성을 통제했습니다. 또한 리뷰 코멘트가 주입된 변경을 가리키는지 퍼지 매칭+LLM 판정으로 검출 여부를 판정해 재현성 있는 recall을 산출했습니다.

- **Empirical Impact**: 결과적으로 OpenAIReview + GPT-5.5가 ICLR/NeurIPS 품질 프록시와의 페어와이즈 정확도에서 83.0%로 가장 높았고, 리뷰가 단순 점수 예측이 아니라도 품질 신호를 포착하는 경향이 확인됐습니다. 오류 회수에서는 OpenAIReview + GPT-5.5가 주입 오류의 71.6%를 잡았으며, 모델 여러 개의 탐지를 합치면 recall이 83.3%까지 상승해 모델 간 보완성이 관찰됐습니다. 더 나아가 실제 공개 배포에서 사용자는 코멘트에 대해 긍정 투표가 부정보다 1.44배(1.44:1) 많았고, 주요 불만은 false positive와 사소한 nitpicks로 정밀도 개선 여지가 드러났습니다.



### Closing the Calibration Gap in Semantic Caching (https://arxiv.org/abs/2606.19719)
Comments:
          23 pages, 2 figures. Source code: this https URL ; Models and Datasets: this https URL

- **Prior Approaches**: 기존 의미 캐싱은 임베딩 유사도/재랭커 점수를 임계값 τ로 이진화해 캐시 히트를 결정하지만, 평가는 주로 PR-AUC 같은 순위 기반 지표에 의존했습니다. PR-AUC는 점수의 실제 크기(캘리브레이션)를 보지 않아, 임계값에서의 “캐시로 제공 가능한 정밀도”와 어긋나는 모델 선정을 유발할 수 있습니다.

- **Core Contribution**: 이 논문은 의미 캐싱의 모델 선택 문제를 ‘랭킹’이 아니라 ‘캘리브레이션(점수 배치)’ 문제로 재정의합니다. 이를 위해 캐시 활용(언제 캐시를 “발사”하는가)까지 반영하는 P-CHR AUC와, 오프라인 랭킹 품질이 배포에서 어느 정도 보존되는지를 나타내는 Calibration Retention Rate(CRR)를 제안합니다.

- **Technical Challenges**: 주요 기술적 난관은 오프라인 평가에서 PR-AUC가 임계값 기반 배포 의사결정을 추적하지 못한다는 점이며, 논문은 이를 운영 격차 operational gap으로 공식화해 구조적 한계와 캘리브레이션 격차로 분해합니다. 실험적으로는 학습 objective(BCE vs MNRL)가 점수 압축/경계 붕괴를 통해 캘리브레이션 격차를 결정하며, 데이터 스케일만으로는 이를 크게 개선하기 어렵다는 점을 보였습니다.

- **Empirical Impact**: LangCache SentencePairs v3(74,265 쿼리)에서 고 PR-AUC 모델이 오히려 배포에서는 최악이 되는 ‘역전(inversion)’이 관찰됐습니다. 예컨대 BCE 재랭커는 PR-AUC는 높지만 P-CHR AUC가 크게 낮아 CRR이 25% 미만까지 떨어졌고, 반대로 ColBERTv2.0은 PR-AUC가 낮아도 P-CHR AUC는 가장 높아 임계값 의사결정에 유리했습니다. 결론적으로, 의미 캐싱에서는 PR-AUC보다 P-CHR AUC/CRR 같은 캐시-인식 지표로 모델을 고르는 것이 비용 절감과 품질 보장의 첫 단계임을 실증적으로 제시합니다.



### NEST: Narrative Event Structures in Time for Long Video Understanding (https://arxiv.org/abs/2606.19706)
- **Prior Approaches**: 기존 비디오-언어 모델은 긴 토큰을 처리해도, 장편 영상의 ‘서사 구조’는 추론하기 어렵다는 한계가 있었다. 벤치마크 역시 짧은 클립의 원자 행동 이해나 needle-in-a-haystack 검색에 치우쳐, 사건(event)이 시간에 따라 어떻게 이벤트로 묶이고 관계 맺는지(사건 그래프, 인과, 계층, 플래시백) 평가는 부족했다. 일부 내러티브 이벤트 연구가 있었지만 대부분이 짧은 구간에 머물러, 수십 분~수시간 간격의 선행 사건을 연결하는 능력을 검증하지 못했다.

- **Core Contribution**: 논문은 장편 영화 전체를 대상으로 ‘서사 사건 구조’를 평가하는 NEST를 제안한다. NEST는 1005편(평균 98분) 영화에 대해 시각·대사·오디오에 근거한 102개의 멀티모달 내러티브 이벤트를 구조화해 주석하고, 시간 순서·계층적 구성·장거리 의존성을 반영하는 이벤트 관계까지 연결한다. 또한 ETD(이벤트 트리거 탐지)·EL(로컬라이즈)·EAE(인자 추출)·ERE(관계 추출) 멀티태스크를 도입해, 단순 retrieval이 아닌 서사 이해를 직접 겨냥한다.

- **Technical Challenges**: 기여를 현실화하기 위한 핵심 난제는 (1) 원시 비디오에서 서사 수준 이벤트를 찾아내고(grounded discovery), (2) 그 이벤트 사이의 관계를 멀리 떨어진 장면까지 일관되게 연결하는 것이다. 이를 위해 오디오 설명을 활용해 시각-행동과 정렬된 텍스트 신호로 이벤트 트리거/인자를 추출하고, LLM 기반 파이프라인과 PropBank 동사 어휘를 결합해 이벤트 온톨로지를 통일했다. 로컬라이제이션은 장면 경계(scene boundary) 수준에서 보수적으로 평가해 주관적 경계 문제와 단기 grounding의 불안정성을 완화했으며, 관계는 닫힌 집합 분류로 ERE를 안정적으로 측정했다.

- **Empirical Impact**: 실험 결과, 모델들은 후보 이벤트를 ‘찾는’ 능력에서 크게 무너진다: ETD는 8% 미만, EL은 6% 미만, EAE는 11% 미만에 머물렀다. 반면 이벤트가 주어졌을 때의 ERE는 상대적으로 가능해져 zero-shot F1 35.45%, fine-tuning 후 F1 44.42%까지 상승했으며, 이는 conditional reasoning과 grounded discovery가 별개 난제임을 보여준다. 또한 플래시백 관계 하위셋에서는 대부분 0에 가까운 성능 붕괴가 나타나, 비선형 시간 추론이 장편 서사 이해의 독립적 병목임이 강조된다.



### Efficiently Representing Algorithms With Chain-of-Thought Transformers (https://arxiv.org/abs/2606.19697)
- **Prior Approaches**: 기존 chain-of-thought(CoT) 연구는 transformer가 Turing complete하다는 표현력 결과를 통해 알고리즘을 ‘원리상’ 시뮬레이션할 수 있음을 보여줬습니다. 하지만 알려진 구성은 주로 Turing machine(TM) 수준의 전이 하나하나를 토큰/디코딩 단계로 옮기는 방식이라, textbook 알고리즘의 기준 모델인 Word RAM의 random-access와 word 단위 연산을 효율적으로 다루기 어렵다는 문제가 남았습니다. 그 결과 TM 기반 시뮬레이션은 Word RAM 대비 큰 오버헤드(대체로 t^2급)를 유발할 수 있었습니다.

- **Core Contribution**: 이 논문은 질문을 구체화해 “CoT transformer가 Word RAM 알고리즘을 textbook 복잡도에 가깝게 효율적으로 모사할 수 있는가?”를 직접 답합니다. 결론적으로, CoT transformer가 Word RAM의 임의 프로그램을 poly-logarithmic 오버헤드만으로 시뮬레이션할 수 있음을 세 가지 실용적 설정에서 증명합니다. 특히 정렬(O(n log n))이나 Dijkstra(O(E+V log V)) 같은 알고리즘 수준의 효율이 과도하게 망가지지 않는다는 점을 목표로 삼습니다.

- **Technical Challenges**: 핵심 기술 난제는 random-access 메모리 접근과 word 단위 산술을 ‘토큰 기반 CoT’가 아니라도 효율적으로 재현하는 구조를 설계하는 데 있습니다. 논문은 (i) 유한 정밀도 + polylog width + rightmost-unique hard attention, (ii) fixed width + log-precision + continuous CoT(벡터 형태 추론), (iii) transformer 위에 선형 RNN 레이어를 얹는 hybrid 아키텍처를 통해 Word RAM instruction 단위 모사를 구성합니다. 또한 곱셈/나눗셈/mod처럼 계산 비용이 큰 연산이 있는 경우와 없는 경우에 대해 CoT 단계 수의 polylog 오버헤드가 각각 달라지도록 분석합니다.

- **Empirical Impact**: 이 논문은 실험 성능이 아니라 복잡도/표현력 관점의 ‘직접 시뮬레이션 가능성’을 엄밀히 제공해, reasoning 모델의 이론적 효율 논의를 한 단계 끌어올립니다. 정보이론적 하한(이산 CoT에서 w-비트 단어를 표현할 때의 하한)을 비교해, 제안한 discrete CoT 시뮬레이션이 사실상 최선에 가깝고 곱셈 없는 flat instruction에서는 더 타이트해짐을 보여줍니다. 결과적으로 CoT가 단지 임의 계산을 “할 수 있다”를 넘어, textbook 알고리즘 수준의 효율을 “대체로 유지할 수 있다”는 방향의 이론적 근거를 마련한 점이 의미 있습니다.



### A Layered Security Framework Against Prompt Injection in RAG-Based Chatbots (https://arxiv.org/abs/2606.19660)
Comments:
          Submitted in ICCK Transactions on Information Security and Cryptography

- **Prior Approaches**: 기존 prompt injection 방어는 대체로 단일 단계(입력 필터링, 시스템 프롬프트 강화, 출력 모니터링)에 머물러 RAG의 취약 구조를 완전히 막기 어렵다. 특히 입력 필터는 검색된 문서에 담긴 indirect injection을 볼 수 없고, 출력 모니터는 악성 페이로드가 모델 컨텍스트에 도달한 뒤에야 대응한다. 결과적으로 지식베이스가 오염되면 해당 문서를 검색하는 모든 사용자가 동시 피해를 입는 문제가 남았다.

- **Core Contribution**: 이 논문은 direct/indirect prompt injection을 모두 겨냥해 추론 파이프라인 전 구간을 가로채는 3-layer 미들웨어 프레임워크를 제안한다. Layer 1은 사용자 입력을 룰 기반 서명 라이브러리와 fine-tuned semantic anomaly classifier로 스크리닝하고, Layer 2는 provenance 기반 instruction hierarchy로 검색 컨텍스트가 operator policy를 덮어쓰지 못하게 한다. Layer 3는 출력에 대해 policy rule engine과 semantic drift detector로 정책 위반 및 목표 달성 여부를 최종 감사한다.

- **Technical Challenges**: 핵심 기술적 난제는 LLM 컨텍스트 윈도우에서 instruction-plane과 data-plane이 분리되지 않아, 검색된 텍스트가 ‘데이터’처럼 보이더라도 지시로 전환될 수 있다는 구조적 취약점이다. 논문은 이를 위해 모델-agnostic한 프롬프트 엔지니어링(티어 태깅과 계층 메타지시)으로 계층 강제를 시도하되, 계층을 우회할 수 있는 공격 가능성을 Layer 3의 정책/드리프트 감사로 흡수하도록 설계했다. 또한 Layer 1의 취약한 단일 룰 매칭을 semantic 확률 점수로 보완하고, 연속 audit loop로 로그를 축적해 분류기를 신종 공격에 맞게 재학습할 수 있게 했다.

- **Empirical Impact**: GPT-4o, Llama 3, Mistral 7B에서 5,080개 샘플을 평가한 결과, 프레임워크는 Attack Success Rate(ASR)를 71.4%에서 11.3%로 크게 낮추며 단일 레이어 최강 베이스라인 대비 27.3%p, 기존 발표된 guardrail 대비 23.8%p 개선을 보였다. false positive rate는 4.8%로 유지했고 median latency overhead는 61.2ms에 그쳐 배포 효율도 입증됐다. ablation 분석에서는 3개 레이어가 상호 보완적으로 작동해 단순 합 이상의 누적 방어 효과를 만든다는 점이 확인됐다.



### Toten: Knowledge-Based Ontological Tokenization Of Physical Quantities And Technical Notation In Brazilian Portugues (https://arxiv.org/abs/2606.19626)
- **Prior Approaches**: 기존 Byte-Pair Encoding, WordPiece, SentencePiece 같은 통계 기반 토크나이저는 어휘 압축에 최적화되어 공학적 수량·단위·숫자·기호식을 의미 구조와 무관하게 subword로 쪼갤 수 있다. 그 결과 토크나이저 단계에서 물리량/단위의 결합 규칙과 수치 표기(로케일, 지수, 소수점, 자리 구분)가 깨지고, 재조립은 downstream 모델에 거의 전가된다. 한편 수량 추출이나 Pint/udunits-2 같은 차원 라이브러리는 대체로 이미 분리된 단위 문자열을 전제하거나, 일반 엔티티 인식은 person/location 같은 범주 위주로 기술-과학 PT-BR 표현을 명시적으로 모델링하지 못한다.

- **Core Contribution**: 이 논문은 지식 기반 온톨로지 토크나이제이션 프레임워크 TOTEN을 제안하며, 통계적 분할 대신 Engineering entities의 형식 온톨로지(OEE)에 근거해 ‘타입 분류→구조화’로 텍스트를 처리한다. TOTEN은 온톨로지(O), classify(원문을 타입 영역으로 매핑), instantiator family(자기-설명형 구조 표현 생성)의 삼중 구조 <O, classify, {inst_τ}>로 형식화되어, 의미 보존을 construction 단계에서 강제한다. 또한 Pint(차원), Unicode Character Database(타이포그래피), RSLP(포르투갈 형용/어근 처리)를 결정론적으로 결합해 규칙 기반 권위를 시스템에 고정한다.

- **Technical Challenges**: 핵심 난제는 ‘수량·단위·기호식’이 텍스트에서 다양한 표기 관례(로케일 숫자, 첨자/차수, 복합 단위, 문장 내 연산자 주변 구조)를 갖는데도 토크나이저가 의미적으로 원자적(atomic) 태그를 유지하도록 규정하는 것이다. 논문은 OEE의 primary types(예: physical quantity, symbolic expression, hierarchical reference 등)와 8개의 구조 원칙(합성 가능 조건, 범주 오류 금지, 불변량 보존 등)을 정의하고, classification이 먼저 타입을 결정하면 instantiation은 포맷만 수행하는 단일 권위(single authority) 구조로 ‘점진적 품질 저하’ 대신 categorical error를 전파한다. 평가 가능한 입력-출력 규약을 위해 출력 언어 ℳ는 BNF 기반 타입 태그와 원문 잔여 텍스트의 literal preservation(타입 영역만 구조화)을 포함하며, 수치 표준화(canonicalization)로 IEEE 754 표현 차이도 제어한다.

- **Empirical Impact**: intrinsic evaluation로 온톨로지 원자성, 차원 동치, 타이포 견고성, 수치 재구성(4가지)을 construction으로 검증하며 EngQuant(N=800)와 PT-BR 외부 코퍼스 4종(총 eligible 사례 1771)을 대상으로 수행했다. 그 결과 TOTEN은 비교 기준에서 unit ontological atomicity를 압도적으로 달성하고, 외부 코퍼스에서 numerical reconstruction 0.775~0.904를 기록해 최강 baseline인 Quantulum3(0.627~0.703)보다 높았으며 EngQuant에서는 0.780 vs 0.340으로 격차가 컸다. 또한 McNemar+Holm 보정의 통계적 유의성과 내부·외부 랭킹의 Spearman 상관으로 타당성을 확인했으며, dimensional equivalence는 Pint 권위를 그대로 계승해 통계적으로 동등 수준을 보였다.



### Uncertainty Decomposition for Clarification Seeking in LLM Agents (https://arxiv.org/abs/2606.19559)
Comments:
          26 pages, 8 figures. Source code: this https URL

- **Prior Approaches**: 기존 불확실성 추정은 대체로 aleatoric/epistemic 같은 틀에 의존하거나, 단일 턴 예측 정확도를 위해 설계된 logprob·다중샘플·학습 기반 보정이 중심이었다. 하지만 LLM 에이전트는 부분관측과 장기 상호작용에서 불확실성이 누적·전파되며, 불확실성의 성격(행동 난이도 vs 요청 모호성)이 섞여 에이전트의 “무엇을 해야 하는지(질문 vs 진행)” 판단을 어렵게 만든다. 또한 black-box API, 상호작용 지연(latency) 한계, 라벨 없는 trajectories 제약 때문에 logprob·multi-sampling·training 기반 접근은 실배포에서 제한되고, 결국 프롬프트 기반 신호 표출이 현실적인 선택지로 남는다.

- **Core Contribution**: 이 논문은 프롬프트 기반 불확실성 신호를 “행동 확신(action confidence)”과 “요청 불확실성(request uncertainty)”로 분해해, 모호한 요청에는 clarification을 요청하고 애매하지 않으면 행동을 이어가도록 설계한다. 특히 request uncertainty를 0/0.5/1의 anchored 척도로 모델이 스스로 판단하게 만들어, 애매함(underspecification) 여부를 대화 중 관측 가능한 행동(request_clarification)과 직접 연결한다. 이를 통해 단순히 실패 가능성을 가리는 수준을 넘어, 사용자와의 공유 mental-model 구축 같은 상호작용적 능력을 불확실성에서 끌어내려는 방향성을 제시한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 black-box API 환경에서 다중샘플·logprob 접근 없이도 ‘원인별 불확실성’을 안정적으로 프롬프트로 추출하고, 이를 장기 에이전트 궤적에서 일관되게 전파하는 것이다. 저자들은 UAM처럼 불확실성(및 설명)을 history에 semantic propagation으로 누적하되, 기존 단일 scalar confidence를 (u_t, c_t) 두 신호로 확장해 단계별로 요청 모호성과 행동 진척도를 분리해 평가 가능하게 만들었다. 또한 trajectory-level 집계(예: 마지막 단계/곱/평균/보수적 규칙) 선택이 결과에 큰 하이퍼파라미터 효과를 주므로, 다양한 집계 전략으로 민감도와 한계를 체계적으로 다룬다.

- **Empirical Impact**: 평가는 50%가 의도적으로 underspecified된 WebShop-Clarification과 ALFWorld-Clarification을 포함해, 5개 LLM 백본(GPT-5.1, DeepSeek-v3.2-exp, GLM-4.7, Qwen3.5-35B, GPT-OSS-120B)에서 ReAct+UE와 UAM을 비교하는 방식으로 수행됐다. 평균적으로 ALFWorld-Clarification에서 제안 방법의 clarification F1이 ReAct+UE 대비 73%, UAM 대비 36% 개선됐고, WebShop-Clarification에서는 모든 백본에서, ALFWorld-Clarification에서는 5개 중 4개 백본에서 clarification 성능이 향상되며 일반성을 보여줬다. 즉, 프롬프트 기반 분해가 특정 모델에 국한되지 않고, 에이전트가 모호함을 스스로 감지해 질문으로 전환하는 능력을 실제 벤치마크에서 뚜렷이 끌어올릴 수 있음을 시사한다.



### Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deploymen (https://arxiv.org/abs/2606.19558)
- **Prior Approaches**: 양자화된 LLM 품질을 빠르게 고르기 위해 per-token KL divergence(KLD)나 Perplexity(PPL)를 BF16 기준 모델 대비 ‘fidelity metric’으로 쓰는 관행이 널리 퍼져 있다. 기존 연구들은 이런 지표가 전체 품질과 어느 정도 함께 움직일 수는 있지만, 후속 태스크 성능을 항상 대변하진 못한다는 한계를 지적해 왔다.

- **Core Contribution**: 이 논문은 Qwen3.6-35B-A3B와 Devstral-Small-2-24B 각각의 여러 GGUF 양자화 코호트에서, KLD가 downstream 벤치마크 점수를 얼마나 잘 ‘순위화’하는지 정밀하게 검증한다. 특히 near-baseline 후보가 모이는 silent zone에서는 KLD–벤치마크 상관이 무너져, 실무에서 가장 중요한 후보군을 가르는 도구로는 부적절함을 보여준다.

- **Technical Challenges**: 핵심 도전은 KLD가 단순히 성능과 연관된 것처럼 보이는 전체 코호트 상관이, 실무적 ‘비교 구간(near-baseline)’에서도 유지되는지 확인하는 것이었다. 저자들은 (1) silent zone/loasy zone을 구분하고, (2) 점수 차이를 ‘disagreement의 양(volume)’과 ‘유익한 방향(direction)’으로 분해하는 관점으로 KLD가 주로 disagreement volume만 측정하며 direction은 과제·상황에 따라 달라진다는 구조를 제시한다.

- **Empirical Impact**: 실험 결과 full cohort에서는 KLD와 합성 벤치마크 점수 간 상관이 강하게 나타났지만(예: Qwen ρ=-0.72, Devstral ρ=-0.86), silent zone에서는 Qwen(ρ≈0.00)·Devstral(유의하지 않음)로 붕괴했다. 또한 per-prompt KLD는 코드 작업에서 실패 예측 신호가 약하고(통과 대비 geometric-mean 비 [1.08,1.22] 수준) 두 모델 간 라우팅 성능도 42.3%~49.4%로 거의 추측 수준에 머물러, KLD/PPL는 ‘손상된 후보 필터링’에는 도움이 되지만 ‘근접 후보 순위 선정’에는 신뢰하기 어렵다는 메시지를 남긴다.



### PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models (https://arxiv.org/abs/2606.19534)
Comments:
          Code available at this https URL

- **Prior Approaches**: 기존 MLLM은 시각 이해에 강점을 보이지만, 대부분 autoregressive(AR) 생성 방식이라 여러 영역을 한 번에 처리할 때 비효율적입니다. 특히 region captioning에서는 각 마스크를 순차적으로 디코딩하고 토큰도 단계적으로 생성해, 영역 수가 늘수록 지연(latency)이 거의 선형으로 커집니다. diffusion language model(DLM)은 병렬 토큰 생성 잠재력이 있지만, fine-grained localized perception에 그대로 확장하는 것은 품질과 병렬성 모두에서 검증이 부족했습니다.

- **Core Contribution**: 이 논문은 diffusion 기반 멀티모달 언어모델 PerceptionDLM을 제안해, 단일 denoising 과정에서 여러 masked region의 캡션을 병렬로 생성하도록 설계했습니다. PerceptionDLM-Base는 discrete diffusion multimodal baseline으로 visual instruction tuning을 통해 지각(perception) 품질을 먼저 끌어올리고, 이후 region-aware 구조(모양을 구분하는 region prompting과 영역 간섭을 줄이는 structured attention masking)를 얹어 병렬 region captioning을 가능하게 했습니다. 또한 ParaDLC-Bench를 통해 캡션 품질과 추론 효율을 동시에 평가하는 틀을 마련했습니다.

- **Technical Challenges**: 핵심 난제는 DLM이 한 번에 여러 토큰을 복원하는 특성상, 동시에 생성되는 서로 다른 영역 간 ‘엔탱글먼트/간섭’을 어떻게 억제하느냐입니다. 논문은 영역별로 RoI-aligned feature replay와 learnable region embedding을 주고, 영역 토큰의 attention을 전역 비주얼 토큰/공유 프롬프트/해당 영역 RoI 토큰 및 자기 영역 캡션 스팬으로 제한하는 structured attention masking으로 영역 독립성을 강제했습니다. 더불어 효율을 위해 멀티해상도 타일링과 학습 단계별(정렬→중간지식→지시따르기→고품질 정제) 파이프라인을 구성했습니다.

- **Empirical Impact**: 실험에서 PerceptionDLM-Base는 16개 멀티모달 벤치마크 중 15개에서 기존 diffusion VLM 대비 우위를 보이며, 특히 ParaDLC-Bench의 multi-region 설정에서 평균 정확도 62.4%로 baseline 대비 큰 폭(예: LLaDA-V 35.2%)의 향상을 보였습니다. 효율 측면에서는 병렬 디코딩 덕분에 multi-region 상황에서 지연이 영역 수에 따라 급격히 늘지 않고, heavy workload(예: 4 masks)에서는 throughput이 최대 3.44배, 단일 이미지 latency가 10.04초→2.92초로 감소했다고 보고합니다. 결과적으로 diffusion 기반 멀티모달 모델이 fine-grained 시각 지각을 ‘병렬’로 확장할 수 있음을 실증하며, 실사용 관점의 처리량 개선 가능성을 제시합니다.



### DeXposure-Claw: An Agentic System for DeFi Risk Supervision (https://arxiv.org/abs/2606.19501)
- **Prior Approaches**: 기존 접근은 LLM 에이전트를 원시 on-chain 데이터나 텍스트를 직접 읽고 추론하게 하거나, 프로토콜별 노출 변화 같은 상대 지표로 위험을 정렬하는 경우가 많다. 이 방식은 약한·오래된·불완전한 근거를 그럴듯한 설명으로 과독해해 높은 단계 개입(고위험 조치)을 유발할 수 있고, 평가 역시 규제자가 실제로 우선순위를 매기는 손실 관점과 false alarm를 정량화하기 어렵다. 결과적으로 “어떤 개입이 쓸모 있었는가”가 아니라 “무엇이 그럴듯해 보였는가”에 치우칠 위험이 있었다.

- **Core Contribution**: 논문은 DeXposure-Claw라는 예측-근거 기반 agentic supervision 시스템을 제안한다. LLM이 원시 데이터를 직접 해석하는 대신, DeXposure-FM(그래프 시계열 파운데이션 모델)이 미래 노출 네트워크를 예측하고 이를 모니터링·스트레스 시나리오·불확실성 추정으로 구조화한 ‘typed evidence’를 통해서만 감독 티켓을 작성한다. 또한 데이터 품질과 신뢰도 게이트로 개입 단계 출력을 제한해, 모든 티켓이 감사 가능한 근거 묶음과 함께 발행되도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 그래프 형태의 미래 노출을 신뢰할 수 있게 예측하면서 (2) 그 예측이 약할 때 LLM이 과잉확신으로 고위험 조치를 하지 않게 만드는 것이다. 논문은 여러 모니터 통계(예: PageRank/HHI/지니 기반 집중도)와 표준화된 스트레스 시나리오를 예측 샘플에 적용해 시나리오 손실(CVaR)과 attribution 신호를 생성하고, 데이터-health 및 confidence gate로 escalation을 사전에 차단한다. 또한 DeXposure-Bench의 decision 축에서 규제자 정렬형 absolute-loss ground truth와 명시적 false-intervention rate로 평가해, “설명 그럴듯함”이 아니라 “규제 관점 개입 정확도”를 측정 가능하게 했다.

- **Empirical Impact**: 실험은 주간 DeFi 노출 그래프 5년치 데이터에서 진행됐으며, 예측-근거 라우팅을 통해 티켓 F1이 대폭 개선되는 결과를 제시한다. 다만 안전성(과잉 개입)은 모델을 더 강하게 해도 근본적으로 해결되지 않고, 데이터-health와 confidence 게이트가 false-intervention 위험을 통제하는 역할을 한다는 점이 강조된다(개입 오판 비율이 모델 간에도 큰 차이를 보이지 않음). DeXposure-Bench는 예측 품질, 경고/불확실성 캘리브레이션, 스트레스 충실도, 티켓 품질, 견고성까지 6축 평가를 제공해 규제 친화적 검증 틀을 만들었다는 의미가 있다.



### Diffusion Language Models: An Experimental Analysis (https://arxiv.org/abs/2606.19475)
- **Prior Approaches**: 기존 LLM은 left-to-right 자동회귀 생성으로 토큰을 순차 예측해 왔고, 생성 지연이 길이에 비례해 늘며 한 번 내린 결정은 되돌리기 어렵다는 한계가 있습니다. 반면 Diffusion Language Models(DLMs)는 반복적 denoising으로 전체 시퀀스를 동시에 다듬어 병렬성·전역적 정교화를 노리지만, 과제별 평가 프로토콜과 추론 예산(denoising steps 등)이 달라 비교가 어려웠습니다. 또한 diffusion 계열은 성능-비용의 스위치가 학습보다 추론 시 매개변수에 크게 의존해, 논문마다 선택이 섞이면 “아키텍처 개선”인지 “디코딩 설정”인지 분리가 불명확했습니다.

- **Core Contribution**: 이 논문은 최신 DLM 8종을 추론 품질과 계산 효율을 함께 보는 통일된 실험 프로토콜로 평가해, 아키텍처 패러다임별(순수 diffusion vs block-hybrid vs 자동회귀 기준) 강점과 약점을 정량 비교합니다. 더 나아가 denoising steps, context length, block size, parallel unmasking 같은 추론 시간 설계요소가 모델 거동을 어떻게 바꾸는지 체계적으로 분해 분석합니다. 통제된 조건에서 더 작은 모델들을 추가로 학습해, 대규모 결과가 “데이터/학습 차이”가 아닌 “구조·추론 설정 차이”에서 기인함을 확인하려는 점이 특징입니다.

- **Technical Challenges**: 핵심 난제는 DLM의 성능이 반복 denoising 횟수와 unmasking 스케줄 등 추론-time 하이퍼파라미터에 민감한데, 기존 연구는 이를 동일 조건에서 스윕하거나 보고하지 않아 공정한 비교가 어렵다는 것입니다. 저자들은 벤치마크를 지식·추론·코딩·번역·구조화 문제해결로 넓히면서도 한 프레임워크(lm-evaluation-harness) 아래에서 평가하고, generation length/step budget/block granularity를 고정·변형하는 controlled comparison으로 품질-효율 곡선을 분리합니다. 또한 순수 diffusion과 block-diffusion의 계산비용(메모리·연산량)을 forward 1회와 전체 생성 관점에서 함께 제시해 배포 관점의 trade-off를 드러냅니다.

- **Empirical Impact**: 대규모 결과에서 순수 full-sequence diffusion은 전역 제약 만족과 지식/추론 과제에서 강한 경향을 보였고, block 기반 하이브리드는 GSM8K·HumanEval 같은 알고리즘·코딩 류에서 두각을 보이되 언어/번역 성능에서는 약화되는 ‘과제 특화’ 패턴이 확인됩니다. 또한 추론 스케일링 실험에서 steps와 context length의 상호작용이 크며, 특정 길이 이후에는 성능 포화 또는 하락이 나타났고 번역은 특히 긴 생성에서 급격히 무너질 수 있음을 보여줍니다. 결론적으로 이 연구는 “모델”뿐 아니라 “생성-time 설계”가 DLM의 성능 경로를 좌우한다는 점을 실증해, 실제 배포 시 어떤 계산 예산에서 어떤 품질을 기대할지 가이드하는 데 의미가 있습니다.



### Thermodynamic Signatures of Reasoning: Free-Energy and Spectral-Form-Factor Diagnostics for Hallucination Detection in Large Language Models (https://arxiv.org/abs/2606.19404)
- **Prior Approaches**: 기존 환각(hallucination) 탐지는 출력 불확실성(UQ)이나 은닉 상태/어텐션 프로브로 나뉜다. 특히 어텐션을 그래프로 보고 Laplacian 스펙트럼을 쓰는 방식은 상위 고유값(top-KK eigenvalues)이나 수작업 스칼라(예: Fiedler value, spectral entropy)처럼 저해상도 요약에 의존해 스펙트럼의 대부분 구조를 놓친다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 Free-Energy Signatures(Fes)를 제안한다. 각 레이어의 attention Laplacian을 Hamiltonian으로 보고, partition function, free energy, spectral entropy, heat capacity, spectral form factor를 여러 스케일(temperature/time)에서 뽑아 한 번에 멀티스케일 스펙트럼 형상을 기술하는 training-free 단일 샘플 디스크립터로 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 어텐션 perturbation(잡음/양자화/수치변화)에 대한 안정성과 (2) 유한 그리드·유한 샘플에서 정보가 얼마나 보존되는지의 이론적 보장이다. 논문은 Fes의 Lipschitz 안정성 정리, 유한 요약이 스펙트럼 역복원이 아니라 모멘트·에너지 스케일 기반 함수에 대한 근사로 작동한다는 expressiveness 결과, 그리고 RMT 기반 unsupervised 점수의 AUROC에 대한 유한-표본 PAC 스타일 바운드를 제시한다.

- **Empirical Impact**: 6개 오픈 웨이트 LLM과 6개 벤치마크에서 Fes 디스크립터에 가벼운 supervised probe를 얹었을 때 attention-spectral 기준선 대비 평균 AUROC를 크게 끌어올렸다(평균 +6.5 AUROC points vs LapEig, +2.4 points vs GoR-4). 완전 unsupervised에서는 RMT-deviation score가 label-free로도 mean AUROC 0.71을 달성했으며, 스펙트럼이 정답 생성에서는 Wigner-Dyson-like, 환각에서는 Poisson-like 통계를 보이는 차이를 함께 확인했다.



### Beyond the GUI Paradigm: Do Mobile Agents Need the Phone Screen? (https://arxiv.org/abs/2606.19388)
- **Prior Approaches**: 기존 모바일 에이전트 연구는 주로 GUI 패러다임에 치중해 화면 정보를 인식하고 screen interaction을 생성하는 방식이 중심이었습니다. 반면 CLI(명령줄 인터페이스)는 기기 서비스와 데이터에 직접 접근할 수 있음에도 연구에서 2순위로 취급돼 왔습니다. 저자들은 이 편향이 실제 사용자 의도 수행에서 성능 한계를 만들 수 있다고 지적합니다.

- **Core Contribution**: 이 논문은 CLI를 GUI와 동등한 1급(first-class) 접근으로 다루며, 여러 CLI 코딩 에이전트(Claude Code, Terminus-2, mini-swe-agent)를 AndroidWorld·MobileWorld에서 평가합니다. 또한 GUI 범위를 넘어서는 일상적 의도를 포착하기 위해 CLI-Advantage Task Suite(45개 템플릿, 5개 카테고리)를 새로 제안합니다. 마지막으로 CLI 패러다임의 상한을 보기 위한 oracle CLI 솔루션도 함께 제공합니다.

- **Technical Challenges**: 핵심 기술 과제는 CLI 맥락에서 필요한 인자·조건·상태를 파악해 명령을 정확히 조합하는 능력과, 화면 기반 작업과 다른 실행 경로를 끝까지 추적하는 데 있습니다. 저자들은 별도의 mobile-specific post-training 없이 여러 모델 API를 교차 평가해 CLI에서도 성능이 자연스럽게 나타나는지 검증했고, 과제 설계를 통해 숨은 기기 상태·다중 조건·크로스 앱 워크플로 같은 난이도를 체계적으로 드러냈습니다. 더불어 GUI 대비 단계 수가 줄어드는지까지 함께 측정해 CLI의 효율성을 정량화했습니다.

- **Empirical Impact**: 결과적으로 Claude Code( Opus 4.7 )는 AndroidWorld에서 71.8%, MobileWorld에서 51.9%로, 재현 가능한 GUI 기준선 전부를 능가했습니다(각 69.3/68.1/57.8%, 43.2/26.3/13.3%). CLI의 잠재력은 oracle 솔루션이 AndroidWorld 88.8%, MobileWorld 86.3%에 도달하며 확인됐고, GUI 밖 일상 의도 과제에서는 모든 CLI 에이전트가 모든 GUI 기준선을 5개 카테고리에서 일관되게 앞섰습니다. 특히 태스크당 단계 수는 10.7 vs 18.6으로 더 적어, 향후 모바일 에이전트의 설계 방향을 CLI 중심으로 확장할 실질적 근거를 제공합니다.



### How Linear Is a Transformer Feed-Forward Block? Per-Block Linear Recoverability Is Learned, Not Architectura (https://arxiv.org/abs/2606.19379)
Comments:
          14 pages, 5 figures

- **Prior Approaches**: Transformer FFN은 key–value memory로 해석되며, 압축이나 해석을 위해 “얼마나 선형(가법)이고 얼마나 비선형(곱셈/상호작용)”인지 논의가 이어져 왔다. 하지만 기존의 선형/다항 probe는 학습 최적화에 의존해 정확한 선형성의 상한을 주지 못하거나, 비선형 잔차를 저차 곱으로 설명할 수 있는지의 측정이 일관되지 않았다.
또한 트랜스포머 활성은 ill-conditioned(조건이 나쁨)하고 outlier 특징이 커서, 단순히 학습한 선형 기준선이 충분히 수렴하지 못해 선형 recoverability를 과소평가할 위험이 있었다.

- **Core Contribution**: 논문은 각 FFN을 입력-출력의 position-wise map으로 보고, 해당 블록의 활성 분포에서 “정확한 least-squares 선형 근사(닫힌형 해)”로 분해한다. 이때 held-out에서 선형 부분이 설명하는 분산비를 linear recoverability R^2_lin으로 정의해, 최적화 없이 재현 가능한 블록 고유 측정값을 제공한다.
그 다음 잔차를 저차(랭크-낮은) bilinear probe로 추가 측정해, 잔차가 단일 저차 곱 형태로 포착되는지까지 확인한다.

- **Technical Challenges**: 핵심 기술적 난제는 트랜스포머 활성의 조건이 나빠서 학습한 선형 baseline이 잘 수렴하지 못한다는 점이다. 논문은 이 문제를 해결하기 위해 선형 항은 학습이 아니라 닫힌형 least-squares 해로 “선형 상한(ceiling)”을 정확히 계산하고, 이후 probe의 이득을 이 상한 대비로만 읽는다.
잔차 분석에서는 bilinear probe의 선형 분기(branch)를 고정해 residual-gain만 분리 추정하며, 저차 곱이 잔차를 얼마나 설명하는지 상호작용 구조를 직접 시험한다.

- **Empirical Impact**: GPT-2, Pythia-160m, llama-160m의 12개 블록을 모두 측정한 결과, R^2_lin은 깊이에 따라 단조롭게 변하지 않고 블록 간 이질성이 매우 크다(거의 선형 >0.99부터 강한 비선형 <0.3까지). 특히 같은 폭/깊이의 GELU 모델(GPT-2 vs Pythia)은 서로 다른 블록이 선형으로 보이며, 선형 recoverability가 활성함수나 아키텍처가 아니라 “학습된 개별 블록 성질”임을 보여준다.
또한 잔차는 랭크-낮은 degree-2 bilinear probe로는 대부분 잘 회수되지 않아, unrecovered 계산이 단일 저차 곱이 아닌 더 고차/분산된 구조일 가능성을 시사한다. 아울러 R^2_lin이 높은 블록은 훨씬 작은 단일 레이어로 교체해도 성능 손실이 제한적일 수 있음을 압축 신호로 제시한다.



### ESBMC-GraphPLC: Formal Verification of Graphical PLCopen XML Ladder Diagram Programs Using SMT-Based Model Checking (https://arxiv.org/abs/2606.18941)
Comments:
          18 pages

- **Prior Approaches**: 기존 ESBMC-PLC는 PLCopen XML의 Ladder Diagram(LD) 중 텍스트 형식(<rung>)만 제대로 라더 로직을 GOTO IR로 변환했다. 그런데 그래픽 형식(tc6_0201의 connection graph 기반)은 파서는 통과했지만, LD→GOTO 변환기가 <rung>을 찾지 못해 라더 할당을 비워버려 검증이 공허하게 성공할 수 있었다.

- **Core Contribution**: 이 논문은 ESBMC-PLC의 백엔드를 그대로 두고, 그래픽 LD를 라더 경로로 복원해 기존 텍스트 LD 변환 파이프라인에 끼워 넣는 ESBMC-GraphPLC를 제안한다. 핵심은 leftPowerRail부터 각 coil까지 DFS로 경로를 뽑아 접점들의 AND 조합과 경로의 OR 조합을 Boolean 라더 표현으로 만들어 GOTO IR을 완성하는 것이다.

- **Technical Challenges**: 그래픽 LD는 localId/refLocalId로 연결 토폴로지를 표현하므로, 그래프에서 “rung 경로”를 정확히 재구성해야 한다. 특히 IEC 61131-3의 scan-cycle에서 SET/RESET 래치 의미를 맞추려면 rightPowerRail의 connectionPointIn 순서를 반영해 SET이 먼저 처리되도록 코일 실행 순서를 잡는 것이 중요한데, 이 논문은 이를 반영했다; 입력(%IX/%QX) 분류는 주소 기반 추정 후 비주소는 휴리스틱으로 보강한다.

- **Empirical Impact**: 검증 실험에서 CONTROLLINO/OpenPLC Editor의 그래픽 LD 3개는 이전처럼 빈 GOTO IR이 아니라 비결정 입력과 라더 로직을 포함한 전체 IR로 생성되었고, 모두 SAFE를 k=2에서 70ms 미만으로 증명했다. 또한 존재하는 11개의 텍스트 LD 벤치마크는 회귀 없이 보존되었으며, Beremiz 예제 2개는 실제 제한(특정 timer semantics 등)으로 추가 발견해 투명하게 보고했다.



New uploads on arXiv(cs.IR)

### Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation (https://arxiv.org/abs/2606.20554)
- **Prior Approaches**: 생성형 추천(Generative recommendation, GR)은 LLM 기반의 자기회귀로 사용자의 다음 상호작용을 예측하며 e-commerce, 광고, 스트리밍에서 성과를 보였다. 핵심 구성요소인 item tokenization(아이템 토큰화)에서 기존 그래프 연동 방식은 graph serialization이 LLM 입력이 길어져 비싸거나, GNN이 지역 서브그래프만 보아 holistic(전체) 정보를 충분히 못 쓴다는 한계가 있다. 또 semantic tokenization은 휴리스틱 학습 목표에 의존해 의미 토큰의 정합성이 보장되기 어렵고, 감독 신호가 부족해 최적 표현을 놓칠 수 있다.

- **Core Contribution**: 논문은 G2Rec을 제안해 사용자 관심 맥락을 그래프 기반 co-engagement(동시관여) 모델링과 의미 토큰화를 동시에 다루는 통합 프레임워크를 만든다. item-item co-engagement graph에서 얻은 관심 prototype과, 각 아이템의 soft(연속형) 클러스터 멤버십을 토대로 “관심 프로필” 토큰을 구성해, ground-truth 사용자 관심 라벨 없이도 더 포괄적이고 의미 정렬된 사용자 맥락을 학습한다. 결과적으로 산업용 sequential recommendation에서 정확도 향상과 모델링 일관성을 노린다.

- **Technical Challenges**: 첫째, co-engagement 그래프는 규모가 커서 원형(E) 크기가 O(M^2)로 폭발할 수 있는데, 이를 위해 그래프 라플라시안 보존을 목표로 이론적으로 근사 보존되는 희소화(sparsification)를 설계해 O(M log M) 수준까지 간선 수를 줄인다. 둘째, 각 아이템이 여러 관심으로 동시에 연결될 수 있어 hard 클러스터링이 부적합하므로, differentiable한 soft graph clustering을 위해 modularity의 연속형(soft modularity) 목적함수를 제안하고 GPU 친화적으로 계산되게 만든다. 마지막으로 학습을 위해 아이템을 “아이템-관심 프로필” 교대 시퀀스로 토큰화하고, 아이템 다음 예측 손실에 더해 관심 프로필 예측을 soft label로 함께 학습하는 구조를 제공한다.

- **Empirical Impact**: Meta의 제품 서피스에서의 온라인 A/B 테스트와 공개 데이터셋의 대규모 실험을 통해 G2Rec이 기존 생성형 추천 방법들 대비 우수함을 보인다. 특히 사용자 그래프 맥락을 전체적으로 반영하면서도 희소화·soft 클러스터링으로 산업 규모 효율을 유지하는 점이 강점으로 제시된다. 온라인 배포에서는 그래프 처리 엔진을 통해 클러스터링을 주기적으로 오프라인 수행해 실시간 응답 시간을 보호하며, 실제 프로덕트 성능 개선으로 이어졌다는 점에서 의미가 크다.



### ELVA: Exploring Ranking-Driven Universal Multimodal Retrieva (https://arxiv.org/abs/2606.20280)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 UMR(Universal Multimodal Retrieval)은 텍스트/이미지 등 다양한 검색을 하나로 묶되, MLLM을 Retrieval로 전환할 때는 주로 contrastive learning을 사용해 positive/negative 임베딩을 분리했다. 하지만 이 방식은 negative를 모두 같은 것으로 취급해, 쿼리 안의 엔티티-행동 같은 여러 grain 정보를 충분히 세밀하게 학습하지 못한다(이를 grain blindness로 지칭). 특히 다중 grain 쿼리에서는 모델이 핵심 의미의 일부만 잡고 나머지를 놓치는 경향이 관측된다.

- **Core Contribution**: 이 논문은 grain blindness를 줄이기 위해 negative를 “유사도에 따라 다르게” 다루는 ranking-driven 학습을 제안한다. 그 결과로 ELVA(ExpLoring Ranking-Driven UniVersal Multimodal RetrievAl)라는 규칙 기반 RL 프레임워크를 설계해, reward를 통해 positive의 상위 랭킹과 negative 간 위계(순서)를 동시에 학습시킨다. 또한 ranking 라벨 없이도 RLVR(Reinforcement Learning with Verifiable Rewards)과 GRPO 기반 탐색으로 랭킹 능력을 유도한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 대조학습의 이진 분류식 학습이 왜 grain-level 정보를 붕괴시키는지(gradient starvation 및 representational collapse)와 (2) retrieval에서 “정답 랭킹 라벨”을 구하기 어렵다는 점이다. ELVA는 생성형 임베딩 추출을 통해 정책에 변동성을 주고(RET 토큰의 hidden state를 임베딩으로 사용), verifiable reward 두 가지(연속형 ranking reward, similarity gap을 강제하는 margin reward)로 보상을 구성해 라벨 없이도 학습을 안정화한다. 더불어 RL용 balanced negative sampling으로 너무 어려운 negative를 걸러 reward 분포가 과도하게 좁아지는 문제를 완화한다.

- **Empirical Impact**: ELVA는 표준 UMR 벤치마크(M-BEIR)에서 다양한 모달 조합에 대해 SOTA 성능을 일관되게 보였다. 특히 multi-grain 전용 벤치마크 MRBench에서는 13.1%의 큰 개선을 기록해 grain blindness 완화 효과를 정량적으로 입증했다. 보상 함수 및 negative sampling 전략에 대한 ablation에서도 ranking reward와 margin reward, 그리고 balanced sampling이 성능에 직접 기여함이 확인되며, 다중 grain 정보 보존 능력이 실험적으로 뒷받침된다.



### ScholarQuest: A Taxonomy-Guided Benchmark for Agentic Academic Paper Search in Open Literature Environments (https://arxiv.org/abs/2606.20235)
- **Prior Approaches**: 기존 학술 논문 검색은 주로 lexical/semantic 매칭으로 큰 문헌에서 빠르게 후보를 찾는 방식이 많았지만, 복합 조건 질의에서는 한 번에 고정된 ranked-list로 끝나는 한계가 있었다. 또한 LLM 기반 agentic search가 등장했지만, 평가용 벤치마크는 소수의 수작업 질의나 특정 논문에서 유도된 질의에 의존해 다양한 연구 의도와 편향을 충분히 통제하지 못했다. 답변(관련 논문 집합) 구축도 비용이 커서 확장성이 떨어졌고, 표준화된 공개 평가 환경이 부족해 재현성 있는 비교가 어려웠다.

- **Core Contribution**: 이 논문은 agentic academic paper search를 현실적인 open literature 환경에서 체계적으로 평가할 수 있게 하는 대규모 벤치마크 ScholarQuest를 제안한다. ScholarQuest는 1,000+ CS 주제와 4가지 연구 의도(방법 중심, 설정 고정, 비교 기반, 범위 제어)를 기반으로 질의 분포를 통제하고, 답변 집합을 만들기 위한 자동 파이프라인과 공용 검색 백엔드 ScholarBase를 함께 제공한다. 이를 통해 “무엇을 찾았는지”뿐 아니라 “어떻게 탐색했는지”까지 동일 조건에서 측정할 수 있게 한다.

- **Technical Challenges**: 핵심 난관은 (1) 의도별로 세밀한 제약을 반영하는 질의를 대규모로 편향 없이 생성하고, (2) 서로 다른 용어/하위 분야에 흩어진 관련 논문을 확장·정제해 큰 gold answer pool을 만드는 데 있었다. 저자들은 ACM CCS 주제에 arXiv taxonomy를 매핑하고, LLM 기반 질의 생성 후 품질 필터링·중복 제거로 1,111개 고품질 질의를 만들었다. 또한 다중 소스 retrieval과 citation expansion, 다단계 관련성 판정(LLM 심사 + 인간 감사)으로 스케일 가능한 답변 구성을 구축했으며, ScholarBase API로 재현 가능한 탐색 환경을 제공했다.

- **Empirical Impact**: 실험 결과, agentic 방법은 single-shot 검색 기준선을 전반적으로 앞섰지만 최고 성능도 Recall@100 0.314, Recall@All 0.355에 머물러 개선 여지가 큰 것으로 나타났다. 특히 scope-controlled 질의와 답변 집합이 커질수록 성능 저하가 커 제약 민감성/강건성의 병목이 드러났다. 실패 분석에서는 검색 노력이 부족해서가 아니라 off-target exploration(잘못된 문헌 영역을 확장)에서 오류가 주로 발생함을 확인해, 향후 intent 보존과 constraint-aware 추론 필요성을 구체적으로 시사한다.



### Generative Engine Optimization at Scale: Measuring Brand Visibility Across AI Search Engines (https://arxiv.org/abs/2606.20065)
Comments:
          14 pages, 4 tables; v1.0 preprint

- **Prior Approaches**: 기존 SEO의 연장선이지만, AI 검색에서는 “키워드 순위”가 아니라 LLM이 브랜드를 실제로 인용·추천하느냐가 핵심이 됐습니다. 과거 GEO 관련 연구는 주로 인용을 끌어내는 콘텐츠 신호(예: 인용 가능한 근거, provenance)를 다뤘고, conversational SEO 전술은 대부분 효과가 제한적이며 오히려 해가 될 수 있음을 보여줬습니다.

- **Core Contribution**: 이 논문은 Ranqo를 통해 5개 AI 엔진의 프롬프트 응답을 대규모로 측정해, 브랜드의 AI Visibility를 “어떻게 인용되는가”까지 포함한 실증 지표로 정리합니다. 특히 권위 있는 상위 브랜드를 제외한 SMEs, D2C, 크리에이터, 초기 스타트업처럼 ‘비교적 덜 노출되는 집단’에서의 인용 격차를 정량화해, GEO의 기준선(baseline) 데이터를 제공합니다.

- **Technical Challenges**: 해결해야 할 어려움은 엔진마다 소스 선택·재검색 방식이 달라 비교 가능성을 유지하는 것과, 감정(sentiment)처럼 노이즈가 큰 신호를 안정적으로 분해하는 데 있습니다. 논문은 unbranded 카테고리 프롬프트 비중을 통해 ‘브랜드가 스스로 호명되는 효과’를 줄이고, 페이지 감사(6개 차원)와 인용 근거(출처 타입/관계)를 결합해 플랫폼별 차이를 측정하도록 설계했습니다.

- **Empirical Impact**: 생산 데이터(102개 브랜드, 3,500+ 런, 10만+ 응답)에서 브랜드 위상에 따라 인용·추천 확률이 3단 사다리로 갈라지며, 첫 실행 기준으로 Tier 1(73%)→Tier 2(44%)→Tier 3(11%)처럼 단계당 약 30%p 격차가 관찰됩니다. 또한 인용은 기업(official) 사이트로 집중되는 경향이 있고, 최고 레버리지는 ‘best-of’ 리스트클처럼 랭크형 콘텐츠(전체 인용의 약 21%)로 나타나며, sentiment는 mention 여부보다 훨씬 더 자주 뒤집힌다는 점이 GEO 측정의 실무 가이드를 제공합니다.



### PACMS: Submodular Context Selection as a Pluggable Engine for LLM Agents (https://arxiv.org/abs/2606.20047)
- **Prior Approaches**: 기존 에이전트의 컨텍스트 관리는 주로 recency truncation(최근순 자르기)에 요약을 더하는 방식이었지만, 이는 주제와 무관하게 “오래됨”만으로 사실을 버려 기억형 작업에서 실패하기 쉽습니다. RAG는 외부 문서를 프롬프트에 불러오지만, 이미 에이전트 루프에 쌓인 메모리/대화/툴 출력 후보를 “함께” 어떻게 고를지는 중재하지 않습니다. Prompt compression은 토큰을 줄이기 위해 텍스트를 손실적으로 재작성·가지치기하지만, 질의 의존성이 부족하고 retained 항목에 정보 손실이 생길 수 있습니다.

- **Core Contribution**: 이 논문은 에이전트의 컨텍스트 assembly(프롬프트 조립)를 “예산 제약 하 부분모형(submodular) 최대선택” 문제로 재정의하고, 메모리 항목·대화 턴·툴 출력 전체를 단일 후보 풀로 두고 관련성 기반으로 고르는 PACMS를 제안합니다. PACMS는 OpenClaw 내부에서 pluggable context engine으로 동작하며, assembly 단계는 PACMS가 책임지고 compaction(압축)은 런타임에 위임하도록 설계했습니다. 이로써 기존처럼 선택 정책이 루프 밖/사후처리에 머무르지 않고, 모델 호출 직전에 “무엇을 남길지”를 원칙적으로 결정합니다.

- **Technical Challenges**: 핵심 난제는 여러 소스가 섞여 있는 후보 풀에서 토큰 예산(예: knapsack 제약) 안에 들어오는 최적 부분집합을 실시간으로 고르는 것이었고, 논문은 facility-location 계열의 커버리지 목적함수를 monotone submodular로 구성해 CELF lazy-greedy로 근사해를 제공합니다. 또한 툴 출력처럼 길고 큰 단위를 first-class candidate로 취급해, 10토큰 메시지부터 5,000토큰 툴 결과까지 동일한 선택 로직으로 처리합니다. PACMS는 임베딩 기반 유사도(relevance)와 커버리지 가중치를 결합해 “관련 없는 최근 정보 유지”와 “관련 있지만 오래된 정보 탈락”을 동시에 완화하도록 설계했습니다.

- **Empirical Impact**: LongMemEval 100문항 샘플(중복도 삽입 포함)에서 PACMS는 evidence-round recall은 MMR과 비슷하지만 end-to-end QA 정확도에서 더 높은 성능을 보였습니다. 예산 45% 조건에서 GPT-5-mini 리더 기준 QA는 +8%p, GPT-5.4-mini 리더 기준 +12%p 우위를 기록했고, 특히 recall에서는 뒤지는데 QA는 앞서는 “정확도 중심”의 격차가 관찰됐습니다. recency truncation은 중복이 커질수록 붕괴했고, 이는 topic-blind 자르기가 장기 기억 에이전트의 기본값으로 부적절함을 실증적으로 뒷받침합니다.



### Stellar: Scalable Multimodal Document Retrieval for Natural Language Queries (https://arxiv.org/abs/2606.19960)
- **Prior Approaches**: 기존 멀티모달 문서 검색은 문서와 질의를 여러 토큰 임베딩으로 표현하고 late interaction(토큰 간 MaxSim 등)으로 정밀도를 높여왔다. 다만 토큰 단위 임베딩을 모두 메모리에 올려야 해서 메모리와 계산 비용이 급증하며, 예로 ColPali는 대규모 코퍼스에서 800GB 이상이 필요하다는 추정이 제시된다. 양자화 기반 ANN/압축 기법도 피크 메모리와 스케일 문제를 근본적으로 해결하진 못한다.

- **Core Contribution**: 이 논문은 Stellar라는 확장 가능한 멀티모달 문서 검색 프레임워크를 제안하며, 핵심은 “토큰 임베딩을 메모리에 고정 보관하지 않고 디스크에 두고, 필요한 후보만 일부 로드”하는 표현-저장 공동 설계다. Stellar는 1) LRF(Lexical Representation-based Filtering)로 후보 문서 집합을 크게 줄이고 2) DLI(Disk-backed Late Interaction)로 후보의 토큰 임베딩만 효율적으로 디스크→메모리로 불러 late interaction을 수행한다. 또한 LRF의 목표는 recall을 해치지 않으면서도 필터링 비용을 낮추는 것이다.

- **Technical Challenges**: 첫 번째 난제는 디스크 기반 late interaction에서 후보를 충분히 줄이기 위한 가벼운 필터를 만드는 일로, dense 검색은 저장·연산 부담이 커 병목이 되기 쉽다. Stellar는 pre-trained MLLM의 next-token prediction head를 재활용해 문서/질의를 어휘(vocabulary) 공간의 sparse lexical representation으로 매핑하고, 이를 inverted index로 top-k 후보를 빠르게 추린다. 두 번째 난제는 디스크 I/O 지연인데, Stellar는 의미적으로 유사한 문서가 같은 디스크 블록에 오도록 balanced clustering 기반 저장 레이아웃을 만들고, 비용 모델에 따라 블록 전체 로드 vs 필요한 토큰만 선택 로드를 동적으로 결정한다.

- **Empirical Impact**: 실험은 4개(추가로 대규모 LargeDoc 신규 데이터셋 포함) 리얼 벤치마크에서 수행되었으며, Stellar는 기존 multi-vector late interaction 대비 메모리 오버헤드와 질의 지연을 1~2 orders of magnitude 줄이면서도 검색 정확도 저하 없이 효과를 유지한다. 즉, 스케일을 “압축으로 억지로 메모리에 맞추는 방식”이 아니라 “디스크에 원본을 두고 필요한 것만 읽는 방식”으로 전환했다는 점에서 배포 관점의 의미가 크다. 대규모 문서 검색 연구를 촉진하기 위해 LargeDoc도 공개한다고 밝힌다.



### Closing the Calibration Gap in Semantic Caching (https://arxiv.org/abs/2606.19719)
Comments:
          23 pages, 2 figures. Source code: this https URL ; Models and Datasets: this https URL

- **Prior Approaches**: 기존 의미 캐싱은 임베딩 유사도/재랭커 점수를 임계값 τ로 이진화해 캐시 히트를 결정하지만, 평가는 주로 PR-AUC 같은 순위 기반 지표에 의존했습니다. PR-AUC는 점수의 실제 크기(캘리브레이션)를 보지 않아, 임계값에서의 “캐시로 제공 가능한 정밀도”와 어긋나는 모델 선정을 유발할 수 있습니다.

- **Core Contribution**: 이 논문은 의미 캐싱의 모델 선택 문제를 ‘랭킹’이 아니라 ‘캘리브레이션(점수 배치)’ 문제로 재정의합니다. 이를 위해 캐시 활용(언제 캐시를 “발사”하는가)까지 반영하는 P-CHR AUC와, 오프라인 랭킹 품질이 배포에서 어느 정도 보존되는지를 나타내는 Calibration Retention Rate(CRR)를 제안합니다.

- **Technical Challenges**: 주요 기술적 난관은 오프라인 평가에서 PR-AUC가 임계값 기반 배포 의사결정을 추적하지 못한다는 점이며, 논문은 이를 운영 격차 operational gap으로 공식화해 구조적 한계와 캘리브레이션 격차로 분해합니다. 실험적으로는 학습 objective(BCE vs MNRL)가 점수 압축/경계 붕괴를 통해 캘리브레이션 격차를 결정하며, 데이터 스케일만으로는 이를 크게 개선하기 어렵다는 점을 보였습니다.

- **Empirical Impact**: LangCache SentencePairs v3(74,265 쿼리)에서 고 PR-AUC 모델이 오히려 배포에서는 최악이 되는 ‘역전(inversion)’이 관찰됐습니다. 예컨대 BCE 재랭커는 PR-AUC는 높지만 P-CHR AUC가 크게 낮아 CRR이 25% 미만까지 떨어졌고, 반대로 ColBERTv2.0은 PR-AUC가 낮아도 P-CHR AUC는 가장 높아 임계값 의사결정에 유리했습니다. 결론적으로, 의미 캐싱에서는 PR-AUC보다 P-CHR AUC/CRR 같은 캐시-인식 지표로 모델을 고르는 것이 비용 절감과 품질 보장의 첫 단계임을 실증적으로 제시합니다.



### SAFE-Cascade: Cost-Adaptive Vision-Language Routing for Chart Question Answering (https://arxiv.org/abs/2606.19646)
Comments:
          Demo paper submitted at CIKM 2026. 4 pages, 2 figures

- **Prior Approaches**: 기존 ChartQA 계열 연구는 VLM이 차트 이미지를 직접 읽고 수치 추론까지 수행하는 방향에 초점이 맞춰져 왔습니다. 다만 질의마다 항상 VLM을 호출하면 비용과 지연이 커지고, OCR 텍스트로 충분한 문제에서도 시각 추론을 낭비할 수 있습니다. 선택적 예측·LLM 라우팅 연구도 있지만, SAFE-Cascade는 ‘모달리티(시각 필요 여부) 라우팅’ 자체를 사용자에게 투명하게 보여주는 데 차별점이 있습니다.

- **Core Contribution**: SAFE-Cascade는 OCR+텍스트 전용 경로로 먼저 답을 만들고, 라우터가 VLM 호출 필요성을 판단해 필요할 때만 visual grounding을 수행하도록 설계된 cost-adaptive chart question answering 시스템입니다. 핵심은 라우팅 과정을 숨기지 않고 OCR 증거, 텍스트 답안, escalation 확률, 최종 결정과 비용/지연 추정치를 함께 제공하는 ‘inspectable modality-routing’ 인터페이스입니다. 또한 임계값 슬라이더로 정확도–비용 프런티어를 직접 조절·탐색하게 합니다.

- **Technical Challenges**: 가장 큰 과제는 텍스트 경로에서 나온 ‘잠정 답’이 충분히 신뢰할 만한지, 아니면 VLM의 시각 기반 추론이 필요한지 오류 없이 분별하는 라우터의 정확도입니다. SAFE-Cascade는 Random Forest 라우터를 사용하고, OCR 길이·밀도, 질문 길이, 수치/비교 플래그, 질문- OCR 겹침, 답안 길이 및 불명/빈 답 지표, 숫자 답 존재, OCR 내 답 포함 여부, 텍스트 모델 추정 지연 등 inference-time 특징만으로 escalation probability를 추정합니다. 확률이 임계값 이상이면 VLM으로 escalate하고, 아니면 텍스트 답을 그대로 채택합니다.

- **Empirical Impact**: ChartQA-2500 홀드아웃 평가에서 SAFE-Cascade는 unified accuracy 69.1%로 full-VLM(67.7%)과 유사한 성능을 보이면서 VLM 호출률을 73.1%로 줄였습니다. 동시에 VLM 호출을 26.9% 감축하고 추정 비용을 9.3% 절감해, 정확도 손실 없이 비용 효율을 얻는 구도를 제시합니다. 또 텍스트 단독 경로(25.3%)와 단순 휴리스틱 개선(37.3%) 대비 성능 갭이 크다는 점에서, 단순 대체가 아니라 ‘언제 VLM이 필요한지’ 학습 라우팅이 이득의 중심임을 실증합니다.



### Token Factory: Efficiently Integrating Diverse Signals into Large Recommendation Models (https://arxiv.org/abs/2606.19635)
Comments:
          8 pages, 10 figures

- **Prior Approaches**: 기존에는 전통적 추천 신호를 transformer 기반 Large Recommendation Models(LRMs)에 넣기 위해 신호를 텍스트로 변환하거나, 이산적인 아이템 표현을 만들어 프롬프트에 편입하는 방식이 주로 쓰였다. 하지만 이런 “textualize” 계열 접근은 프롬프트 길이 폭증, 큰 메모리 사용, 높은 연산 비용으로 이어지기 쉽다.

- **Core Contribution**: 이 논문은 전통적 신호를 LRM이 바로 처리할 수 있는 “soft tokens”로 변환하는 프레임워크 Token Factory를 제안한다. 이를 통해 이질적인 입력 특징을 효율적으로 압축·통합하면서 추천 성능을 개선하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 전통 신호를 토큰화할 때 정보 손실을 최소화하면서도 프롬프트 길이와 연산·메모리 부담을 동시에 줄이는 것이다. 논문은 Token Factory의 아키텍처로 soft tokens를 생성·주입해 prompt length explosion을 억제하고, LRM 전처리 파이프라인의 효율을 확보한다.

- **Empirical Impact**: 생산 규모(production-scale) 추천 환경에서의 실험 결과는 Token Factory가 기존 방식 대비 효율성과 모델 성능 모두에서 이점을 제공함을 보여준다. 즉, 대규모 서비스에서 다양한 전통 신호를 현실적으로 통합할 수 있는 경로를 제시하며, 추천 산업 적용 가능성을 강화한다.



### VCG: A Multimodal Retrieval Framework for E-Commerce Video Feeds under Extreme Cold-Start Conditions (https://arxiv.org/abs/2606.19627)
- **Prior Approaches**: 기존 전자상거래 추천은 검색 기반 카탈로그에서 축적된 클릭·구매 이력을 바탕으로 collaborative filtering(예: Matrix Factorization, Autoencoder)에 의존해 왔습니다. 하지만 동영상 피드 환경에서는 신규 영상이 상호작용 이력이 부족한 extreme cold-start가 발생하고, watch-time 최적화가 duration bias와 position bias를 함께 왜곡해 표준 engagement 신호가 흔들립니다. 텍스트 메타데이터 기반 검색은 메타 태그가 희소하거나 부정확한 경우 시각적 취향을 제대로 반영하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Video Candidate Generation(VCG)이라는 대규모 멀티모달 검색 엔진을 제안해, 사용자와 영상을 shared semantic space에 매핑하고 행동 이력 없이 zero-shot으로 후보를 뽑는 접근을 제시합니다. 핵심은 domain-adapted CLIP 기반 두 타워(two-tower) 구조로, retrieval은 시각 내용과 사용자 스타일의 매칭 질문(What does this video look like?)으로 전환한다는 점입니다. 또한 generative embedding(LVLM/LLM)과 discriminative embedding(CLIP)의 검색 적합성을 비교해, retrieval에서는 contrastive 기반 임베딩이 유리하다는 분석을 함께 제공합니다.

- **Technical Challenges**: VCG가 직면한 기술 과제는 (1) 희소한 상호작용 그래프에서 사용자-영상 유사도를 안정적으로 학습/정의하는 문제, (2) 동영상 피드의 duration·position bias를 오프라인 지표로는 검증하기 어려운 exposure bias 문제, (3) LLM 임베딩을 그대로 쓰면 embedding space collapse처럼 검색 성능을 망가뜨릴 수 있다는 문제입니다. 저자들은 픽셀 기반 시각 신호만으로 10개 프레임을 평균해 영상 임베딩을 만들고, 메타 태그의 잡음을 의도적으로 배제했으며, 오프라인 평가는 LVLM-as-a-judge(Qwen2.5-VL)로 visual coherence를 점수화해 온라인 성공의 대리 지표로 연결했습니다. 더 나아가 Qwen 임베딩은 attribute 예측에는 강하지만 k-NN 검색에서는 등방성이 깨져 약해지는 반면, CLIP은 hypersphere에서 더 균일한 분포를 만들어 검색 엔진으로 적합하다고 보여줍니다.

- **Empirical Impact**: 오프라인에서는 recency 기준선 대비 의미적/시각적 coherence 지표가 개선됐고, Top-10에서 LVLM judge의 Relevant 점수 분포가 ‘좋은 매칭/우수 매칭’ 쪽으로 이동하는 것이 관찰됐습니다. 온라인 4주 A/B 테스트에서는 비디오 시작률은 +8%로 크지 않았지만, 소비 깊이(예: 절반 이상 시청)는 50% uplift을 기록해 클릭을 넘어선 체류·완주 품질 개선을 확인했습니다. 또한 daily lift가 실험 기간 내내 지속돼 단기 노출 효과가 아니라는 점과, 매출·사용자 리텐션 같은 핵심 지표가 안정적으로 유지돼 추천 피드가 거래 흐름을 잠식하지 않았다는 의미가 큽니다.



### MonaVec: A Training-Free Embedded Vector Search Kernel for Edge and Offline AI Systems (https://arxiv.org/abs/2606.19458)
Comments:
          27 pages, 11 figures. Code and artifacts: this https URL (PyPI: monavec; this http URL: monavec-core). Zenodo: doi:https://doi.org/10.5281/zenodo.20559587

- **Prior Approaches**: 기존 벡터 검색(Qdrant, Weaviate 등)은 상시 서버 프로세스와 대용량 RAM, 또는 IVF용 클러스터링/학습용 코퍼스 패스를 전제로 한다. FAISS는 임베디드화가 가능하지만 IVF 학습 데이터가 필요하고, 모바일/임베디드 환경에선 C++ API 및 외부 라이브러리 의존성이 부담이 된다. 즉, “네트워크·서버·학습데이터 없이도 단일 파일로 재현 가능한 임베디드 벡터 검색”을 만족하기 어렵다.

- **Core Contribution**: MonaVec은 SQLite와 같은 배포 모델(단일 파일, 단일 함수 호출, 오프라인 동작)을 벡터 검색 커널에 적용한 결정론적(deterministic) end-to-end 시스템이다. 학습 없이 기본 quantization이 동작하도록, Randomized Hadamard Transform(RHDH)로 입력 분포를 N(0,1)에 가깝게 만든 뒤 미리 계산된 Lloyd-Max 표로 4-bit 양자화를 수행한다. 결과 인덱스(.mvec)는 ChaCha20 회전 시드를 내장해 아키텍처/플랫폼이 달라도 빌드 내에서는 top-K가 재현되도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 데이터 관측 없이도 양자화 테이블을 “맞는 분포”에 대해 유효하게 유지하는 것과 (2) 결정론까지 보장하면서 빠른 검색을 제공하는 것이다. MonaVec은 RHDH의 구조화된 랜덤 회전과 Lloyd-Max 스칼라 양자화 결합으로 training-free 파이프라인을 만들고, L2 메트릭에서는 per-dimension whitening이 아닌 단일-pass global standardization(fit())로 순위 보존 성질을 확보한다. 또한 HNSW에선 그래프 구성과 검색의 스코어링 불일치(특히 L2)를 바로잡고, FP32로 그래프 토폴로지를 구성한 뒤 4-bit로 검색만 수행하는 설계로 양자화 잡음이 그래프를 망가뜨리는 문제를 피한다.

- **Empirical Impact**: AG News에서 BGE-M3 임베딩(45K×1024)을 대상으로 4-bit BruteForce의 Recall@10이 0.960(27MB)까지 도달하며, float32 FAISS-IVF 및 8-bit usearch보다 낮은 처리량을 대가로 더 높은 재현성과 메모리 효율을 제공한다. L2/유사 과제에서도 fit() 기반 global standardization이 Recall을 유의미하게 끌어올려(예: 패션 MNIST에서 0.41→0.62) “학습 없이도 메트릭 특성에 맞춘 표준화”의 효과를 보여준다. edge·offline RAG/에이전트용으로 단일 파일/단일 호출이라는 실사용 제약을 정면으로 겨냥해, 해당 분야에서 벡터 검색의 배포 패러다임을 바꿀 잠재력이 크다.



### Easy Reads: A Python program for making Scientific Papers on arXiv more Reader Friendly and Accessib (https://arxiv.org/abs/2606.20550)
Comments:
          9 pages. Open-source software project available at: this https URL

- **Prior Approaches**: 그동안 논문 접근성을 높이려면 확대(zoom) 같은 임시 조치에 의존하는 경우가 많았고, 이는 문서 탐색 시 매끄럽지 않으며 인쇄본에는 효과가 그대로 반영되지 않는 한계가 있었다. arXiv의 실험적 HTML 제공도 일부 논문에만 해당하고, Zotero 같은 참고문헌/노트 워크플로와의 통합성이 떨어질 수 있다. ePub은 폰트·레이아웃 커스터마이즈가 가능하지만 저널별 품질 편차와 제공 범위의 제약이 크고, 아카이브의 방대한 preprint에는 범용적으로 대응하기 어렵다.

- **Core Contribution**: Easy Reads는 arXiv URL로부터 TeX 소스를 가져와 글꼴 크기와 컬럼 레이아웃을 재편집한 뒤 PDF를 다시 컴파일하는 end-to-end 오픈소스 파이썬 도구다. 사용자는 본문 글꼴 크기와 (원하면) 전체 단일 컬럼 전환만으로 화면/인쇄 모두에서 더 읽기 쉬운 결과물을 얻을 수 있다. 핵심은 ‘확대’가 아니라 문서 자체의 포맷을 재생성해 가독성 장벽을 낮춘다는 점이다.

- **Technical Challenges**: 가독성을 높이는 포맷 변경은 LaTeX 문서 구조와 저널별 패키지 관행에 따라 호환성 문제가 생길 수 있어, 단일 컬럼 전환·여백/줄간격·수식/그림 스케일링의 일관성을 맞추는 것이 기술적 난제다. Easy Reads는 메인 .tex 파일을 찾아 사용자가 지정한 설정(기본 글꼴 크기, 단일 컬럼 모드, 여백/라인스페이싱)을 적용한 뒤 LaTeX 컴파일로 최종 PDF를 생성해 이를 해결한다. 또한 명시적 CLI 인자로 빠르게 재현 가능한 워크플로를 제공한다.

- **Empirical Impact**: 이 프로젝트는 화면에서의 작은 글꼴·두 컬럼 레이아웃이 digital eye strain과 고정시간, 시각 피로 부담을 키울 수 있다는 필요성(기존 연구들)을 배경으로, 실제 사용자 경험 개선을 목표로 한다. 현재는 alpha 단계이며, 저널별 LaTeX 차이로 인한 출력 품질 변동 가능성을 전제로 GitHub에서 이슈 제보를 받고 있다. 그럼에도 arXiv preprint까지 포맷 커스터마이즈를 확장한다는 점에서 접근성 중심의 PDF 배포/개선 논의에 실질적인 참고가 될 것으로 기대된다.



### When Does Streaming Tool Use Help? Characterizing Tool-Intent Stabilization in Streaming Retrieval-Augmented Generation (https://arxiv.org/abs/2606.20113)
- **Prior Approaches**: 기존 Retrieval-Augmented Generation(RAG)은 검색 결과를 생성에 접목해 정확성을 높이지만, 도구 호출(tool call) 자체가 대화 지연을 만든다는 문제가 있었다. Streaming RAG은 입력이 끝나기 전에 speculative tool query를 병렬로 보내고 결과가 충분한지로 반영해 지연을 줄이려 했으나, 벤치마크 전체의 평균 개선만 보고 “어떤 쿼리에서” 이득이 실제로 가능한지 메커니즘을 쿼리 단위로 설명하진 못했다. 또한 기존 지표는 주로 시스템/집계 성능에 초점이어서, 사용자 발화 흐름 중 어느 시점에 의도(검색 대상)가 결정되는지가 성능 상한을 어떻게 좌우하는지 불명확했다.

- **Core Contribution**: 이 논문은 Streaming RAG의 이득이 “시스템이 똑똑해서”가 아니라 “검색 쿼리를 결정짓는 정보가 입력 스트림에서 언제 처음 등장하는가”에 의해 좌우된다는 점을 정량화한다. 이를 위해 tool-intent stabilization(도구 의도 안정화)이라는 쿼리 고유의 측정치를 정의하고, 이 값이 얼마나 일찍 안정되는지에 따라 숨길 수 있는 tool latency의 최대 비율을 model-agnostic bound H로 상한 계산한다. 결과적으로 어떤 발화가 speculative query로 latency hiding이 가능한지, 그리고 그 비용 대비 가치가 언제 생기는지 배포 관점의 판단 기준을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 “정답을 담고 있는 증거가 접두사(prefix) 단계에서 실제로 retrievable해지는 순간”을 구분해 안정화 시점을 측정하는 것이다. CRAG에는 gold passage 레이블이 없어서, 정답 문자열을 검색 문서에 grounding(기계적 문자열 매칭)해 sufficiency stabilization의 시간을 계산하고, grounding이 만드는 오차/편향을 줄이기 위해 엄격/완화(relaxed) 두 가지 grounding 팔을 함께 운영한다. 또한 실제 스트리밍 파이프라인을 재현하는 비동기 harness로, 이론적 bound H가 측정된 perceived-latency saving을 보수적으로(상한처럼) 설명하는지 검증하며, CPU-only·training-free 환경에서 재현 가능한 분석을 수행한다.

- **Empirical Impact**: CRAG validation 1371문항에서 grounding 가능한 비율은 35.4%였고, BM25가 어떤 접두사에서든 gold evidence를 top-k에 올리는 쿼리는 21.3%였다. sufficiency 관점 안정화는 매우 일찍(평균 φ_suf=0.26, 중앙값 0.14) 일어나며, 분포는 가설처럼 뚜렷한 bimodal이라기보다 “초기 안정화가 많은 쏠림 + 얇은 지연 꼬리” 형태를 보였다. 배포 핵심 수치로는 L=600ms, δ=3w/s, θ=0.8에서 evidence가 제때 retrievable한 쿼리(=95.2%)가 높게 나타나지만, top-1이 안정화되는 것까지 섞은 전체 blend 기준으로는 73.9%로 낮아져 지표 해석에 주의가 필요함을 시사한다. 또한 question type이 early/late를 유의미하게(다만 효과 크기는 작게, 대략 4% 수준) 갈라 주며, reasoning 복잡도보다 발화 내 entity 위치가 retrieval-sufficiency 안정화를 좌우한다는 관찰로 이후 “learned speculative trigger” 설계 방향을 구체화한다.



### Multi-Agent Transactive Memory (https://arxiv.org/abs/2606.19911)
- **Prior Approaches**: 기존 retrieval-augmented generation(RAG)은 인간이 쓴 문서를 중심으로 한 에이전트 단일 사용 맥락의 보강에 강점이 있지만, 에이전트들이 만든 절차적 산출물은 보통 한 번 쓰고 폐기되거나 생산한 에이전트에만 남겨집니다. reasoning/thought reuse 계열은 비용·효율을 개선해도 “생산자 중심 재사용”에 머물러 새로 투입된 에이전트가 이미 존재하는 해결책을 다시 찾아야 하는 문제가 남아 있습니다. 또한 transfer learning·knowledge distillation은 도메인 정렬이나 추가 학습이 필요해, 오픈 생태계의 이질적·동적으로 생성되는 에이전트 집단에 바로 쓰기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 Multi-Agent Transactive Memory(MATM)라는 인프라로, 에이전트 집단이 생성한 action-observation trajectory를 공유 저장소에 누적하고 소비 에이전트가 이를 조회해 성능을 개선하도록 제안합니다. producer(생성)와 consumer(소비) 역할은 고정이 아니라 문맥에 따라 바뀌며, 조회된 궤적을 통해 “인구 단위 경험 재사용”을 가능하게 합니다. 특히 interactive 환경에서 긴 상호작용 궤적이 담는 절차 지식을 핵심 아티팩트로 삼아, 개인의 재탐색을 집단의 누적 지식으로 전환하려는 설계가 중심입니다.

- **Technical Challenges**: 가장 큰 도전은 상태가 조건인 궤적을 검색하는 일입니다. MATM은 최근 l단계 상호작용을 retrieval key로 쓰고 다음 구간을 value로 저장하는 state-conditioned key-value 인덱싱을 채택해, 쿼리와 현재 상태에 맞춘 “이어가기” 형태의 guidance를 제공합니다. 더 나아가 단순 임베딩 유사도만으로 뽑힌 후보를 learning-to-rank(LTR) reranker로 재정렬하며, producer 메타데이터(신뢰/품질), consumer 메타데이터(개인화) 등 여러 특성을 포함해 marginal utility(무조회 대비 성능 기여) 기준으로 라벨링·학습합니다.

- **Empirical Impact**: ALFWorld와 WebArena에서 MATM 조회는 평균 task 성공률과 상호작용 단계 수 효율을 함께 개선하며, 별도 조정이나 joint training 없이도 downstream 성능 향상이 나타납니다. ALFWorld에서는 성공률이 47%→55%로 상승하고 단계도 11.77→11.18로 줄어드는 등 효과가 뚜렷했고, WebArena에서도 개선은 더 완만하지만(성공률 18%→20%, 단계 22.0→20.3) RPP가 양수로 전환되며 Pareto 측면의 우위가 관측됩니다. 또한 reranking은 환경별로 이득 폭이 달랐지만(예: ALFWorld에서 SVMRank 강세), 전반적으로 “저장소가 커질수록”과 “집단 내 다양한 능력대 에이전트가 함께” 이득을 확장할 수 있음을 보여 오픈 에이전트 생태계에서의 경험 공유 설계 패턴으로 위치시킵니다.



### Query-aware Routing for Filtered Approximate Nearest Neighbors Search (https://arxiv.org/abs/2606.19898)
Comments:
          12 pages

- **Prior Approaches**: 필터드 ANN(범주형 속성 필터)은 벡터 유사도 외에 라벨 조건(AND/OR/정확일치)을 함께 만족하는 top-k를 찾는 핵심 연산으로, pre-filter/post-filter 같은 범용 적응과 라벨 정보를 인덱스에 통합한 전용 인덱스로 나뉜다. 하지만 대표 기법들은 데이터셋·조건(AND/OR/Equality)에 따라 성능이 크게 달라져 “항상 이기는” 단일 방법이 없다는 문제가 드러났다. 또한 같은 데이터셋/같은 술어 타입 안에서도 쿼리마다 최적 방법이 달라 rule-based 라우팅은 recall 손실을 줄이기 어렵다.

- **Core Contribution**: 논문은 쿼리 정보를 반영해, 필터드 ANN의 실행 방법과 파라미터 조합을 쿼리 단위로 선택하는 query-aware routing 프레임워크를 제안한다. 각 후보 ANN 방법에 대해 쿼리의 recall을 예측하고, 오프라인 벤치마크 표(방법/파라미터별 측정 recall과 QPS)를 조회해 recall–QPS 트레이드오프 최적 조합을 고른다. 이를 통해 데이터셋 특성·술어 타입뿐 아니라 개별 쿼리 난이도 변화까지 흡수하려는 접근이 핵심이다.

- **Technical Challenges**: 주요 기술 과제는 (1) recall을 잘 맞히는 feature를 고르는 것과 (2) 라우팅 추론이 느려지지 않게 모델을 설계하는 것이다. 20개 이상의 후보 통계에서 중첩 ablation으로 selectivity(쿼리 선택도), LIDmean(데이터셋 기하 난이도), predicate type의 최소 3개 특징만 남겨 과적합 위험을 줄였고, 분류 대신 각 방법별 recall을 직접 예측하는 regression(각 방법마다 독립 MLP)으로 오차 크기까지 학습하도록 했다. 또한 recall만 보고 끝내지 않고, 서비스 요구에 맞는 recall threshold TT를 두어 예측 recall을 만족하는 방법들 중에서 QPS가 최대인 파라미터를 고르는 방식으로 end-to-end 효율을 맞췄다.

- **Empirical Impact**: 대규모 벤치마크로 10개 내외 필터드 ANN 알고리즘을 여러 데이터셋·3종 술어 타입·수천 파라미터 조합에서 비교하고, 어떤 단일 기법도 전 영역에서 우수하지 않음을 정량 확인했다. 라우터는 6개 실제 데이터셋으로 학습한 뒤 5개 unseen 검증 데이터셋에 적용되며, 기존 filtered ANN 베이스라인 대비 모든 검증 세트에서 recall–QPS 균형을 동시에 개선하는 state-of-the-art 결과를 보였다. 더불어 라우터의 추가 지연(latency) 오버헤드는 “무시할 수준에 가깝다”고 보고되어, RAG·벡터DB 같은 실서비스 흐름에 적용 가능성을 강화했다.



### When Global Gating Is Enough: Admission-Time Hubness Control in Anisotropic Vector Retrieval Systems (https://arxiv.org/abs/2606.19692)
- **Prior Approaches**: 기존 RAG 보안은 주로 탐지 방식으로, 문서가 허브(hub)처럼 역kNN에 과도하게 등장하는지 주기적으로 reverse-kNN 스캔해 이상치를 삭제/차단한다. 하지만 이 방식은 문서가 삽입된 뒤 다음 스캔 전까지의 노출 창이 생기고, 대규모 코퍼스를 반복적으로 재검사해야 한다는 구조적 비용이 따른다.

- **Core Contribution**: 이 논문은 “admission-time control”로 전환해, 문서가 검색 가능한 상태로 들어가기 전에 sentinel-query들을 기준으로 후보 문서의 hub-likeness를 점수화해 게이트로 통과/격리(quarantine)한다. 또한 top-k를 많이 가져오는 허브형 문서를 차단하되, 자연스럽게 일반적인(glossary/overview) 문서까지 과도하게 막지 않도록 기준선은 false-positive 예산에 맞춰 고정한다.

- **Technical Challenges**: 핵심 난제는 공격형 hub와 실제로도 범용적인 문서(자연 hub)를 구분하면서도, 토픽별(“domain-aware”) 통계를 추가했을 때 오히려 더 잘해야 하는지 검증하는 것이다. 저자들은 임베딩 공간의 anisotropy가 토픽-국소 신호와 전역 신호를 결합해 주어 “전역 게이트”만으로 충분한 기하적 조건이 성립함을 보이고, 토픽 게이트는 실험 전반에서 이득이 없음을 찾아내며, 임계값 τ는 incremental하게 유지해 삽입 비용이 코퍼스 크기에 거의 독립적이도록 설계했다.

- **Empirical Impact**: 100,000 문서 코퍼스 2종과 5개 인코더(384~1024d), 공격 쿼리/수비 sentinel을 분리한 조건에서 전역 게이트는 결정적 구간에서 universal·concept hub를 recall 1.0 수준으로 잡고, 일반 문서 false positive는 1% 내외로 유지된다. HNSW 기반 실제 ANN 인덱스에서도 게이트 오버헤드는 end-to-end 삽입 경로에서 약 3.1%로 작았고, approximate index에서 결정 뒤집힘은 1.2%였으며 공격 관련 뒤집힘은 관찰되지 않았다.



### Denoising Implicit Feedback for Cold-start Recommendation (https://arxiv.org/abs/2606.19658)
Comments:
          Accepted by KDD 2026 ADS Track

- **Prior Approaches**: 암묵적 피드백을 활용한 추천에서 라벨 노이즈(클릭베이트, position bias 등)는 필연적이지만, 기존 연구는 이를 “loss 값이 높다/예측 점수가 낮다” 같은 휴리스틱 신호로 구분해 sample selection이나 sample re-weighting으로 처리해왔다. 이 방식은 일반적으로 노이즈 샘플을 찾는 판별이 잘 작동할 때 성능이 나오지만, cold-start 아이템은 원천적으로 모델 출력 분포가 불안정해 loss·예측점수·gradient 신호가 노이즈 탐지에 부적합해진다. 또한 스트리밍 환경과 산업용 추천 파이프라인에 바로 얹기 어려워 온라인 적용성이 제한된다는 문제가 있었다.

- **Core Contribution**: 이 논문은 cold-start 아이템이 특히 노이즈 암묵적 피드백(false positive/false negative)에 더 취약하다는 점을 강조하고, 이를 denoising implicit feedback 문제로 정면 해결한다. model-agnostic 접근인 DIF(Denoising Implicit Feedback)는 cold 아이템에 대해 콘텐츠 유사 warm 아이템들의 collaborative representation을 바탕으로 pseudo-label을 생성하고, 이를 이용해 노이즈 라벨을 샘플 수준에서 보정한다. 기존처럼 모델 예측 자체에 의존하기보다 “사용자 관심(안정적) + 콘텐츠 유사성”을 축으로 추정 정확도를 높이는 것이 핵심이다.

- **Technical Challenges**: 어려움은 (1) cold 아이템의 pseudo-label을 합리적으로 설계하는 것과 (2) 온라인 스트리밍 학습에서 콘텐츠 유사 warm 아이템을 실시간으로 검색·표현해 pseudo-label을 만들 수 있는지, (3) 여러 pseudo-label을 어떻게 더 정확히 결합할지, (4) 노이즈 라벨을 어떤 정도로 수정할지에 있다. DIF는 warm 아이템을 top-k 콘텐츠 유사 이웃으로 뽑아 여러 pseudo-label을 만들고, 콘텐츠 유사도 기반 confidence로 가중 합산하며, relative entropy와 cold-start 상태를 통해 샘플별 uncertainty를 추정해 pseudo-label의 보정 강도를 adaptive하게 조절한다. 또한 Kuaishou의 듀얼-타워 기반 retrieval 단계에 실제로 붙이기 위한 데이터/서빙 파이프라인(콘텐츠 임베딩, item-to-item ANN, collaborative embedding 저장 및 갱신)까지 구현 경험을 제시한다.

- **Empirical Impact**: 이 방법은 이론적 정당화와 함께 실제 데이터셋 3종에서 offline 성능을 넓게 검증해 일반화 가능성을 보여준다. 더 나아가 Kuaishou에 billion-user 규모로 배포되어 cold-start 시나리오에서 여러 상용 지표를 유의미하게 개선했다고 보고한다. 즉, cold-start 구간의 잘못된 학습 신호를 줄여 추천의 성장 잠재력을 확보하고 피드백 루프 악화를 완화하는 실용적 효과가 확인된 셈이다.



### Cost-Optimal LLM Routing with Limited User Feedback under User Satisfaction Guarantees (https://arxiv.org/abs/2606.19376)
Comments:
          Preprint. Under review

- **Prior Approaches**: 기존 cost-aware LLM routing은 추론 비용을 줄이는 데는 효과적이었지만, SLA에서 요구하는 품질(만족도) 기준을 시간에 대해 보장하는 정식 이론적 근거는 부족했습니다. CARROT나 PROTEUS처럼 formal 보장을 제공하더라도 주로 offline 학습 기반이거나, online 적응을 하더라도 실제 운영에서 거의 충족되기 어려운 완전·균형·지연 없는 피드백 가정을 사용했습니다. 또한 관측 기반 학습을 시도한 방법도 SLA 보장과 online 적응을 동시에 만족시키지 못했습니다.

- **Core Contribution**: 이 논문은 프로덕션에서 흔한 “희소하고 one-sided인 사용자 피드백”만으로도, 비용을 최소화하면서 SLA(만족도 비율 α) 위반을 엄격히 통제하는 온라인 라우팅 알고리즘 SLARouter를 제안합니다. SLARouter는 비용 최적 정책을 목표로 하되, Lyapunov drift-plus-penalty를 관측 데이터 환경에 맞게 확장해 시간 평균 만족도 조건을 유지합니다. 결과적으로 per-benchmark 튜닝 없이도 SLA 제약을 만족시키는 라우팅을 구현합니다.

- **Technical Challenges**: 핵심 난제는 (1) 대부분의 요청에서 품질 레이블이 없고, (2) 라우터가 실제로 선택한 모델에 대해서만 피드백이 관측된다는 점(다른 모델의 성능 불명)입니다. 논문은 탐색(exploration) 확률을 점진적으로 줄이는 전략과, 모델별 multi-label satisfaction predictor를 사용해 one-sided 관측만으로도 학습이 진행되게 설계했습니다. 또한 피드백이 없는 경우 가상 큐 업데이트에 predictor 점수를 대체 프록시로 사용하면서, 가상 큐 안정성과 SLA 준수 성질이 유지되도록 분석을 구성했습니다.

- **Empirical Impact**: 다양한 LLM 벤치마크에서 SLARouter는 SLA 제약을 만족하면서도 기존 cost-aware 라우팅 대비 최대 2.2x 운영 비용 절감을 보였습니다. 특히 per-benchmark tuning 없이도 동등 수준의 라우팅 성능을 달성해, 벤치마크-운영 불일치로 인한 실무 부담을 낮출 가능성이 큽니다. 실질적으로 “online adaptivity + 형식적 SLA 보장 + 희소 one-sided 피드백 학습”을 동시에 만족한 최초 계열로, 운영형 LLM 라우팅 설계에 기준점을 제시합니다.



New uploads on arXiv(cs.CV)

### JanusMesh: Fast and Zero-Shot 3D Visual Illusion Generation via Cross-Space Denoising (https://arxiv.org/abs/2606.20563)
Comments:
          ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 3D 시각 착시 연구는 2D 투영·표면 패턴(그림자 아트, 와이어 아트 등)에 머물렀고, 진짜 3D 메시는 보통 SDS로 3D 표현을 최적화하는 방식이 주류였습니다. 다만 이런 최적화 기반 방법은 형태당 약 40분 수준으로 느리고 색이 과포화되는 문제가 있으며, 서로 다른 두 객체를 단순 이어붙이면 기하 경계(이음새)와 semantic leak(의도치 않은 의미 유출)가 눈에 띕니다.

- **Core Contribution**: 이 논문은 text-driven 3D visual illusion을 zero-shot·training-free로 생성하는 2단계 프레임워크를 제안합니다. 단일 메시는 한 각도군에서는 프롬프트 y1 의미를, 다른 각도군에서는 y2 의미를 보여주도록 설계되며, 임의 시점에서는 추상적/비의도 의미가 드러나지 않게 숨기는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 서로 다른 의미를 갖는 기하를 하나의 매끈한 메시에 동시에 결합하되, (2) 각 시점에서 올바른 텍스처 의미만 보이게 만드는 데 있습니다. 저자들은 Stage 1에서 dual-branch denoising을 cross-space로 수행(라티스를 voxel/SDF로 디코딩-블렌딩 후 재인코딩)하고, CLIP-guided orientation search로 두 객체의 기하 정렬 실패를 줄인 뒤 SDF blending으로 기하적 일관성을 확보합니다; Stage 2에서는 view-conditioned texture synthesis로 각 시점의 2D diffusion prior를 fused geometry에 투영·집계해 시점별 의미를 정확히 고정합니다.

- **Empirical Impact**: 실험 결과, 제안 방법은 3–5분 내 생성하면서 geometric integrity, semantic recognizability, 효율성에서 기존 baseline을 전반적으로 능가합니다. 정량 평가지표(예: CLIP 유사도, GPT-4.1-mini 기반 의미 판별 정확도, FID/KID, Object Detection 기반 기하 융합도)와 사용자 연구 모두에서 이음새가 덜 보이고 의도한 의미가 더 잘 인식되는 경향이 확인되며, CLIP 기반 적응형 회전이 고정 각도보다 자연스러운 착시 효과를 만든다고 보고합니다.



### TimeProVe: Propose, then Verify for Efficient Long Video Temporal Reasoning in Activities of Daily Living (https://arxiv.org/abs/2606.20561)
- **Prior Approaches**: 기존 Long Video Question Answering(LVQA) 접근은 길이가 긴 무편집 영상에서 질의와 관련된 희소한 근거를 찾기 위해 영상을 촘촘히 처리하거나(c) 대규모 vision-language model(VLM)을 전부 순회하는 방식이 많아 계산 비용이 커집니다. 반대로 캡션 기반의 희소 추론은 있으나, 시간적으로 국소화된 단서나 동작(모션) 중심의 증거를 놓치기 쉽다는 한계가 지적됩니다.

- **Core Contribution**: 논문은 비용 효율적인 하이브리드 프레임워크 TimeProVe를 제안해, 전체 영상에 VLM을 쓰지 않고도 시간적 근거를 찾도록 설계했습니다. 핵심은 Action-based Candidate Evidence(ACE) 모듈로, 시간 구간에 대응되는 동작을 질의 조건에 맞춘 후보 답변과 근거 윈도우로 변환해 경량 단계에서 먼저 “찾을 것”을 좁히고, 이후에만 VLM으로 검증하도록 합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 긴 영상에서 시간적으로 국소화된 단서를 정확히 후보로 뽑는 것과 (2) 그 후보에만 VLM을 호출해도 검증 품질을 유지하는 것입니다. TimeProVe는 경량 모듈로 action-grounded answer–evidence hypothesis를 만들고, ACE에서 경량 LLM 추론으로 query-conditioned 후보를 구성한 뒤 타깃 구간에 대해서만 expensive VLM 검증을 수행하는 방식으로 비용-정확도 균형을 맞춥니다.

- **Empirical Impact**: 실험 결과 TimeProVe는 새 벤치마크 OpenTSUBench(OTB)에서 최강 baseline 대비 7.3% 향상과 함께 VLM 호출을 75%, 추론 비용을 93% 절감했습니다. 또한 temporal grounding을 위한 명시적 학습 없이도 Charades-STA에서 경쟁 성능을 보였고, grounding VLM을 추가하면 state-of-the-art까지 달성해 해당 분야의 실용성에 의미 있는 진전을 시사합니다.



### UNIEGO: Proxies as Mediators for Unified Egocentric Video Representation Learning (https://arxiv.org/abs/2606.20559)
- **Prior Approaches**: 기존의 시점 고정 웨어러블 카메라 기반 egocentric video understanding은 단일 관점·단일 모달리티·단일 모델로는 사람 행동의 복잡한 맥락을 충분히 담기 어렵습니다. 또한 RGB/Depth/Skeleton, ego-exo 시점 등 서로 다른 지식을 가진 여러 teacher로 distillation을 시도해도, teacher들의 아키텍처와 특징 기하(geometry)가 달라 gradients가 충돌해 학습이 불안정해지는 한계가 있었습니다.

- **Core Contribution**: 본 논문은 egocentric 영상만으로도 보완적 지식(관점·모달리티·foundation model 표현)을 함께 수용하는 “표현력 있는 통합 egocentric 표현”을 목표로 합니다. 이를 위해 9명의 teacher(ego-exo 관점, RGB/Depth/Skeleton 모달리티)와 4개의 foundation model을 아우르는 UNIEGO라는 unified egocentric encoder를 제안하고, proxy를 매개로 teacher 지식을 공통 공간으로 통역하는 구조를 도입합니다.

- **Technical Challenges**: 핵심 기술 과제는 서로 이질적인 teacher 지식을 그대로 한 unified 모델에 증류하면 특징 정렬이 되지 않아 학습 신호가 서로 간섭한다는 점입니다. 논문은 representation-specific Proxy models로 proxy 공간을 통해 지식 충돌을 줄인 뒤, Selective Proxy Distillation(SPD)로 각 샘플에서 “정확하고 confidence 있는” proxy만 선택해 신뢰할 수 있는 supervision만 증류하도록 하며, 추가로 UNIEGO를 proxy 파라미터의 learned convex combination으로 초기화해 손실 지형을 안정화합니다.

- **Empirical Impact**: UNIEGO는 action recognition, video retrieval, action segmentation의 3개 egocentric video 이해 과제에서 3개의 ego-exo benchmark에 대해 state-of-the-art를 달성했습니다. 특히 naive multi-teacher distillation 대비 성능이 더 높게 나타나, proxy 매개 구조와 SPD 같은 선택적 증류가 더 풍부하고 discriminative한 egocentric representation을 만든다는 점을 실증적으로 보여줍니다.



### Thinking in Boxes: 3D Editing in Real Images Made Easy (https://arxiv.org/abs/2606.20556)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 text-to-image 편집은 2D 프롬프트나 bounding box로 공간 변환을 제어하려 하지만, 이동 거리·회전·카메라 움직임을 명확히 분리해 지시하기 어렵다는 한계가 있다. 또 3D에 가까운 방식들도 주로 느슨한 depth 기반 조건화(예: ControlNet)나 depth/최적화(inversion·optimization) 의존, 또는 메시·convex primitive 같은 복잡한 입력이 필요해 실사용성이 떨어진다. 특히 real photograph에서 큰 변환과 disocclusion까지 안정적으로 다루는 데는 취약했다.

- **Core Contribution**: 이 논문은 ‘thinking in boxes’라는 인터페이스로 3D box 쌍(입력 박스·목표 박스)을 편집 사양으로 사용해, 번역·회전·스케일·viewpoint 변화를 geometry 문제처럼 well-posed하게 만든다. 또한 카메라 운동과 객체 운동의 상대 관계를 고정하기 위해 depth-aligned planar floor를 전역 기준틀로 도입하고, 박스의 면 색상(방향별 RGB 계열)을 통해 3D orientation 신호를 명시한다. 그 결과, 사용자는 실사진에서 직접 3D 편집을 지시하면서 scene/object identity를 유지하는 결과를 얻는다.

- **Technical Challenges**: 핵심 기술 문제는 3D 박스만으로는 ‘객체가 움직였는지 카메라가 움직였는지’ 등 변환 해석이 모호해진다는 점이다. 이를 depth-aligned planar floor로 기준 좌표계를 고정해 단일 해석을 보장하고, 각 박스 면의 방향별 색코딩을 통해 회전·관점 변화까지 조건 신호로 전달한다. 모델은 FLUX-Kontext에 박스/바닥 조건을 멀티모달 토큰으로 정렬해 넣고, LoRA로 fine-tuning하되 전체 생성 능력은 보존하도록 설계한다.

- **Empirical Impact**: 학습은 합성 멀티오브젝트에서 paired views로 시작한 뒤, Objectron 일부 실데이터로 소량 fine-tuning해 in-the-wild 실사진 일반화 성능을 확인했다. object editing과 camera editing 모두에서, 사용자 선호도(A/B user study)와 PSNR·SSIM·Warp Error·Mean Distance·IoU·Angular Error 같은 정량 지표에서 기존 방법들을 크게 앞서며 큰 3D 편집과 disocclusion 복원까지 잘 수행했다. 또한 한계로는 유사 스케일 객체에서 박스가 구분되지 않아 변환이 identity로 수렴할 수 있고, 배경에 미세한 원치 않는 변화가 남을 수 있음을 제시한다.



### Current World Models Lack a Persistent State Cor (https://arxiv.org/abs/2606.20545)
Comments:
          39 pages, 16 figures

- **Prior Approaches**: 기존 비디오/월드모델 벤치마크는 화질, 모션, camera controllability처럼 “관측되는 표면”을 중심으로 평가했습니다. 하지만 카메라를 돌려 관측가능성을 바꿨을 때, 생성된 세계 상태가 관측 밖에서도 진화하며 다시 돌아와도 동일한 사건의 끝점이 유지되는지(=world-state persistence under viewpoint intervention)는 제대로 묻지 못했습니다.

- **Core Contribution**: 이 논문은 관측가능성 변경을 카메라 개입(intervention)으로 해석하는 최초의 진단 벤치마크 WRBench를 제안합니다. WRBench는 카메라가 요청대로 움직였는지, 화면이 연속적으로 읽히는지, 재등장한 대상이 사건의 엔드포인트와 일관되는지를 evidence-attribution 체인으로 분해해 평가합니다.

- **Technical Challenges**: 핵심 기술적 난제는 “카메라가 실제로 떠났는지”와 “세계가 실제로 계속 진화했는지”를 한 스코어로 섞지 않고 원인별로 분리하는 attribution problem입니다. 이를 위해 WRBench는 Natural-25 이벤트 설계를 바탕으로 관측-이탈-재관측을 계층적 게이팅(재관측 가능성 판정)과 6차원 진단 지표로 분리하고, WRBenchLib로 모델이 받은 viewpoint 조건과 생성된 evidence의 provenance를 기록해 이질적 제어 패러다임 간 비교를 공정화합니다.

- **Empirical Impact**: 23개 비디오 생성 모델, 9,600개의 결과를 분석한 결과 공통적으로 “관측 중이던 상태를 추적하듯 유지하다가 카메라가 돌아오면 중단 시점의 상태로 되돌리는” 보존-복귀 실패가 반복됐습니다. 이 오류는 다양한 제어 패러다임/모델 계열/스케일 증가에도 줄지 않았으며, 끝점(endpoint) 지속을 학습목표로 삼는 what-memory 중심의 설계가 필요하다는 방향성을 제시합니다.



### SSD: Spatially Speculative Decoding Accelerates Autoregressive Image Generation (https://arxiv.org/abs/2606.20543)
- **Prior Approaches**: 기존 자율회귀(autoregressive) 이미지는 VQ-VAE 계열로 이미지를 discrete token grid로 만들고, 이를 raster-scan으로 1D 시퀀스처럼 펼쳐 “다음 토큰 예측”을 반복해 생성합니다. 이 1D 평탄화는 이미지의 2D 공간 국소성을 버려 다음 토큰을 여러 개 미리 맞추기 어렵고, 긴 시퀀스로 인해 메모리 wall(매 스텝 가중치 재로딩/메모리 I/O)이 심해집니다. speculative decoding, Jacobi 기반 병렬화, MTP(다중 토큰 예측) 같은 NLP식 가속은 대부분 1D 가정에 묶여 있어 시각 생성에서 속도 향상이 제한적이거나(대개 수 배) 품질 저하/새 아키텍처 필요 같은 절충이 생깁니다.

- **Core Contribution**: 이 논문은 SSD(Spatially Speculative Decoding)로, 예측 목표 자체를 이미지의 2D 기하(geometry)에 정렬하는 틀을 제안합니다. 다음 토큰만 맞히는 대신, 같은 행의 인접 토큰(가로)과 바로 아래 토큰(세로)을 동시에 초안(draft)으로 예측해 2D 상관을 활용합니다. 또한 SSD는 미리학습된 백본을 수정하지 않는 plug-and-play 모듈로 설계돼 다양한 unified autoregressive 모델에 그대로 얹을 수 있습니다.

- **Technical Challenges**: 핵심 난제는 ① 2D로 초안을 짜려면 토큰 간 관계를 잘 학습해야 하고, ② discrete 코드북에서 정확한 토큰을 맞히면 확률 분포가 평평해 acceptance이 낮아져 비효율적이라는 점입니다. 이를 위해 SSD는 연속 latent space로 학습하며, 마지막 transformer 층의 RMSNorm 직전 hidden state를 예측 타깃으로 삼아 더 안정적인 학습 신호를 얻습니다. 이어서 verification(검증)은 단순 이진 reject가 아니라 auto correction(자기 수정) 형태로, 불일치 위치들을 잔차 분포(residual)로 추가 forward 없이 보정하고 r+1 라운드만으로 전체 블록을 정리해 속도 이득을 유지합니다.

- **Empirical Impact**: 검증은 Janus-Pro, Lumina-mGPT, Emu3 세 모델에서 수행했으며, 최대 13.3x 수준의 wall-clock 속도 향상을 달성하면서 DPG-Bench와 GenEval에서 높은 생성 충실도를 유지합니다. 특히 Emu3의 90×90 토큰 격자(대략 8100 토큰)에서 339s→25.6s(약 13.3x)로, 격자 크기 증가에 따라 병렬 초안의 이득이 더 커지는 경향도 확인됩니다. 결과적으로 “언어에서 온 1D 디코딩 제약”을 벗어나 시각의 2D 기하를 존중할 때, 실시간에 가까운 고해상도 자율회귀 생성 효율화가 가능하다는 점을 실증한 작업으로 평가됩니다.



### CalTennis: Large Multi-View Tennis Video Dataset and Benchmark of Monocular-to-3D Pose Estimation (https://arxiv.org/abs/2606.20542)
- **Prior Approaches**: 기존 3D 포즈 추정은 실험실 기반 MOCAP처럼 강한 기준선(ground truth)에 의존하거나, in-the-wild에서도 몸에 센서/마커를 부착하는 방식이 많았습니다. 이 때문에 일반 카메라(모노큘러)로 자연스러운 동작을 측정할 때 남는 깊이(depth), 발 접촉, 안정성 같은 실패 모드가 충분히 드러나지 못했습니다. 또한 다중 뷰가 있더라도 이를 정답 생성에 쓰거나(준정답/트라이앵글레이션), 훈련 신호로만 활용해 모델 오류의 ‘측정’으로 연결되지 않는 경우가 흔했습니다.

- **Core Contribution**: 이 논문은 테니스 실전/경기 영상을 대규모로 모은 Caltech Tennis Dataset(CalTennis)을 제안하며, 11M+ 프레임(51시간), 40명, 2–6대 동기화 소비자 카메라(60Hz) 구성을 제공합니다. 더불어 모노큘러-to-3D 포즈 추정의 오류를 라벨 없이 평가하기 위해, 뷰 간 일치도(disagreement)를 ‘오류의 하한(lower bound)’으로 이용하는 표준화된 프로토콜을 제공합니다. 기존 표준 지표(MPJP, PA-MPJPE 등) 외에 footwork(foot skating)와 stability(균형) 같은 스포츠 분석에 직접 연관된 불일치 지표를 새로 도입합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 카메라 캘리브레이션/동기화가 완전히 신뢰하기 어렵고, (2) iPhone의 거친 타임스탬프 때문에 카메라 간 최대 ~1000ms 오프셋이 생기며, (3) 이 오프셋을 평가에서 제거하지 않으면 모델 성능과 시간 불일치가 섞인다는 점입니다. 논문은 법선화된 코트 기하를 이용해 외부파라미터를 추정하고, 임의 시점 pose를 보간한 뒤 전역 시간 오프셋을 grid search로 보정해 뷰 간 불일치가 ‘실제 재구성 오류’가 되도록 만듭니다. 이후 각 모델의 예측을 공통 월드-시공간 좌표로 변환해, 발 속도/높이와 안정성(지면에 닿는 발의 convex hull 대비 중심질량 투영)을 기반으로 일관성 지표를 산출합니다.

- **Empirical Impact**: 5종 SOTA 모노큘러 3D 포즈 추정기를 CalTennis에 벤치마킹한 결과, 관절 각도/포즈 자체는 상대적으로 정확하지만 깊이 스케일과 발 접촉은 프레임·뷰에 걸쳐 일관되게 추정되지 못합니다. 특히 translation(공간 이동/깊이축) 추정은 모델마다 최대 오류 폭이 크고 ‘pose drifting’ 형태의 진동이 관찰되며, 반면 포즈는 뷰 간 평균 약 11cm 수준의 불일치를 보였습니다. 또한 SMPL-X shape(신장/체형 파라미터)와 안정성도 카메라 각도에 민감해 모델이 서로 다른 신체 형상 편향을 학습했을 가능성을 시사하며, 이는 스포츠 동작 분석·바이오메카닉스 적용에서 새로운 개선 방향을 제공합니다.



### The FID Lottery: Quantifying Hidden Randomness in Generative-Model Evaluation (https://arxiv.org/abs/2606.20536)
Comments:
          Website: this https URL

- **Prior Approaches**: 기존 이미지 생성 평가는 FID를 단일 숫자로 제시하며, 같은 모델에서만 여러 sampling seed를 돌려 분산을 줄이려는 방식이 널리 쓰였다. 하지만 이는 재학습(training lottery)에서 생기는 변동을 포함하지 못해, “리트레이닝해도 같은 결론인가”를 검증하기 어렵다. 또한 diffusion류 모델은 flow-matching loss에서 매 step마다 Gaussian noise를 다시 그려 학습 중 랜덤성이 지속되는데, 이 점이 FID 재현성에 미치는 영향은 충분히 계량되지 않았다.

- **Core Contribution**: 이 논문은 FID를 training seed와 generation seed 두 축의 확률변수로 보고, 여러 SiT 네트워크를 수백 단위로 리트레이닝해 N×K 패널에서 FID 분산을 직접 측정한다. 그 결과, 리트레이닝 변동이 sampling 재추출 변동보다 훨씬 크며(예: FID가 3.2× 더 흔들림), 단일 FID 보고는 노이즈 바닥을 가릴 수 있음을 정량화한다. 더 나아가 per-cell 최적 classifier-free guidance 튜닝을 포함한 새로운 FID 평가 프로토콜과 “약 1.3% CoV 이하면 결론 불가”라는 불확실성 기준을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 FID가 단일 이미지 점수가 아니라 Inception feature 분포의 Fréchet distance라서, 작은 차이가도 집계 지표에서 크게 요동칠 수 있다는 점이다. 연구진은 (1) 초기화, 데이터 로더 순서, flow-matching loss의 per-step Gaussian noise를 분해해 어느 랜덤원이 학습 축 분산을 만드는지 추적하고, (2) CFG 스케일을 (training, sampling) seed 쌍마다 golden-section search로 맞추는 방식으로 노이즈 바닥을 줄이면서도 seed 순위가 재배치되는 현상까지 함께 관측한다.

- **Empirical Impact**: 실험 결과 training seed가 만드는 FID 변동이 generation seed 변동을 지배하며, compute나 모델 크기를 늘려도 FID 변동의 상대적 바닥(CoV)은 큰 폭으로 줄지 않았다(대략 1–2%대 유지). 또한 per-cell CFG 튜닝은 분산을 약 절반 수준으로 낮추지만, 어떤 seed가 더 좋은지 “순위”를 바꾸어 단일 프로토콜 간 비교를 위험하게 만든다. 따라서 저자들은 여러 training seed에 대한 error bar와 CoV 기반의 “inconclusive” 판정 기준을 보고 표준으로 도입해야 한다고 촉구한다.



### VisDom: Sparse Novel View Synthesis with Visible Domain Constrain (https://arxiv.org/abs/2606.20531)
- **Prior Approaches**: 기존 sparse NVS는 NeRF나 3D Gaussian Splatting(GS/3DGS)이 좋은 성능을 보이더라도, 입력 뷰가 적으면 3D 구조 복원이 ill-posed가 되어 부유하는 아티팩트(floaters)와 기하 불일치가 쉽게 발생한다. 이를 줄이려 학습된 priors, depth 제약, diffusion guidance 같은 방법들이 동원되지만, 도메인 가정이나 추가 학습 데이터 의존성이 커져 일반화에 한계가 있다. 반면 silhouette 정합(실루엣 loss)은 직관적이지만 극단적 sparsity에서는 제약이 약해, 실제보다 큰 visual hull 영역을 허용해 오히려 성능이 떨어질 수 있다.

- **Core Contribution**: 이 논문은 학습 없이 동작하는 기하 제약인 VisDom(visible domain)을 제안한다. 고전 visual hull을 “모든 뷰에서 교집합으로 최대한 일치하는 영역”이 아니라, “최소 K개 뷰에서 공동 관측된 3D 공간(visible domain)”으로 더 타이트하게 정의해 희미한 깊이 모호성을 줄인다. VisDom은 NeRF와 3DGS 모두에 쉽게 붙일 수 있으며, 파라미터 학습 없이 silhouettes만으로 작동한다.

- **Technical Challenges**: 핵심 과제는 실루엣이 제공하는 2D-기반 제약만으로는 sparsity에서 불확실한 큰 3D 부피가 남는다는 점이다. 저자들은 voxel carving 기반의 visual hull 계산에서 점유(occupancy) 투표 기준에 더해, 해당 복셀(또는 샘플 지점)이 최소 K개 카메라에서 관측되었는지(공동 가시성)를 추가 필터링한다. 이후 NeRF에서는 해당 visible domain에만 volumetric sampling 범위를 제한해 밀도(density)가 배치될 공간을 제어하고, 3DGS에서는 visual hull을 초기화와 최적화 시 마스크 기반 제약에 활용해 Gaussian 배치를 유도한다.

- **Empirical Impact**: 실험은 ActorsHQ, MipNeRF360, Omni3D에서 4장 입력 같은 극단적 sparse 조건까지 포함하며, VisDom은 적용된 모든 설정에서 일관된 개선을 보였다. 특히 4 views에서 PSNR이 최대 90% 가까이 개선되는 등, 기존 범용 모델이 사실상 실패하던 구간을 복원했다는 점이 두드러진다. 또한 GaussianObject 위에 얹으면 Omni3D/MipNeRF360에서 성능을 끌어올리면서도 22배 이상 빠른 학습 비용으로 경쟁/향상 결과를 보이며, learned prior 없이도 효과적인 보완재 역할을 입증했다.



### SARLO-80: Worldwide Slant SAR Language Optic Dataset 80cm (https://arxiv.org/abs/2606.20523)
- **Prior Approaches**: 기존 SAR–광학 멀티모달 벤치마크는 주로 Sentinel-1 GRD처럼 강도(intensity)만 제공하고 지상 기준으로 투영된 저해상도 데이터를 사용해 왔습니다. 그 결과 복소 SLC가 담는 위상 정보와 layover/shadow 같은 네이티브 획득 기하를 충분히 보존하지 못해, SAR 고유 물리성 기반의 학습이 제한됩니다.

- **Core Contribution**: 본 논문은 Umbra spotlight 공개 데이터에서 2,500여 개 전 세계 장면을 기반으로, 복소값 SAR SLC(VV/HH)와 정렬된 고해상도 광학, 그리고 자연어 캡션을 한 샘플로 묶은 VHR SAR–optical–text 데이터셋을 제안합니다. SICD(Sensor Independent Complex Data)로 제공되는 센서 네이티브 정보를 활용해 복소 SAR을 유지한 채 slant-range 80cm 격자로 표준화하고, 광학 타일을 SAR 그리드에 국소 좌표 상응(affine)으로 워핑해 픽셀 정합을 맞춥니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 복소 SAR의 대역/주파수 성질을 깨지 않으면서 표준 격자로 일관 리샘플링하고, (2) SAR 네이티브 슬랜트-레인지 좌표계에서 광학 이미지를 지오메트리 왜곡 없이 로컬 정합하는 것입니다. 논문은 band-limited FFT 리샘플링(크롭/제로패딩)으로 네이키스트 조건에 맞춘 표준화와, 장면 메타데이터 기반의 조밀 지오로케이션 그리드 및 1024x1024 패치 단위 국소 affine 워핑을 결합해 이 문제를 해결합니다.

- **Empirical Impact**: 데이터셋은 총 119,566개의 트리플(복소+진폭 슬랜트레인지 SAR 패치, 정렬 광학 패치, SHORT/MID/LONG 캡션)을 제공하며, 72개 국가·257개 로케이션에 걸친 폭넓은 지형/인프라를 커버합니다. 공개 고정 split과 전처리/베이스라인 코드를 통해 네이티브 SAR 기하에서 cross-modal retrieval 및 텍스트 조건 생성의 가능성을 보였고, 단일 캡션보다 캡션 길이 다양성을 함께 쓰거나 후반 timestep reweighting을 적용할 때 생성 speckle의 사실성이 개선되는 경향을 보고합니다.



### HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining (https://arxiv.org/abs/2606.20521)
Comments:
          Github: this https URL

- **Prior Approaches**: 기존 embodied foundation models 사전학습은 텔레오퍼레이션 기반 실로봇 궤적이 중심이었는데, 정밀한 행동(액션) 지도와 신체(embodiment) 정렬 덕분입니다. 다만 수집 비용과 획득 난이도가 높고 행동·환경 다양성이 낮아 확장에 병목이 생깁니다. 그래서 대안으로 egocentric human video(1인칭 인간 영상)가 저비용·고다양성 소스로 주목받았지만, 실로봇 데이터 대비 성능 검증은 부족했습니다.

- **Core Contribution**: 이 논문은 고정된 post-training/validation 프로토콜 하에서 egocentric human video와 teleoperated real-robot trajectories를 사전학습 데이터로 체계적으로 비교합니다. 핵심 발견은, 적절한 filtering과 labeling 파이프라인을 거친 egocentric 데이터가 단순 대체를 넘어 더 나은 성능을 낼 수 있다는 점입니다. 또한 “영상으로 세계 표현을 먼저 학습한 뒤, 소량의 실로봇 라벨 데이터로 액션 공간 정렬에 적응(adapt)”하는 스케일링 패러다임을 제시합니다.

- **Technical Challenges**: egocentric 영상은 행동 감독이 직접적이지 않고, 로봇 실행 관점에서의 라벨 품질·데이터 정합성이 흔들릴 수 있어 그대로 쓰면 성능이 떨어질 위험이 큽니다. 논문은 이를 위해 데이터 filtering(품질/유효 샘플 선별)과 labeling(행동·학습 표적의 정교한 생성)을 설계해 사전학습 신호의 신뢰도를 높였습니다. 그 결과 실로봇과의 행동 예측 및 과제 실행에서 유의미한 격차를 만듭니다.

- **Empirical Impact**: 동일한 사전학습 데이터 양을 가정했을 때, egocentric 데이터 사전학습 모델은 실로봇 action prediction에서 validation loss가 24% 낮았습니다. 더 나아가 in-distribution 과제 성공률은 52.5% 높고, out-of-distribution에서는 90% 더 높은 성공률을 보여 견고한 일반화를 시사합니다. 이 결과는 고가의 로봇 데이터 수집 전, egocentric 영상 데이터 품질 평가와 가공 파이프라인에 투자하는 전략이 분야 확장에 실질적 의미가 있음을 확인합니다.



### S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligenc (https://arxiv.org/abs/2606.20515)
Comments:
          Project Page : this https URL

- **Prior Approaches**: 기존 VLM 및 tool-augmented 에이전트는 대체로 정적·무상태 추론에 머물러, 연속적인 3D 공간에서의 persistent object state 유지와 다중 시점·시간에 걸친 증거 통합이 약하다는 한계가 지적된다. VADAR, SpaceTools 같은 최근 에이전트형 접근도 도구 사용을 강화하지만, 궁극적으로는 숨겨지고 변화하는 3D 월드를 관측 스트림으로부터 일관되게 재구성·추론하는 데까지는 거리가 있었다. 결과적으로 프레임 단위 2D 관찰에 과도하게 의존해, 정밀 기하를 직접 근거로 삼기보다 텍스트 패턴/semantic prior에 취약해진다.

- **Core Contribution**: 논문은 연속 multi-view 이미지와 비디오에서 3D 공간을 이해·추론하는 “공간 도구 사용” 에이전트 패러다임 S-Agent를 제안한다. 핵심은 공간 추론을 프레임 레벨 예측이 아니라 spatio-temporal evidence accumulation(공간-시간 증거 누적)으로 재정의해, scene-centric understanding을 frame-centric recognition 너머로 확장하는 것이다. VLM은 의미적 planner로서 어떤 증거가 필요한지 결정하고, 계층형 spatial tools/experts와 temporal memory가 2D 근거를 3D 기하 근거로 끌어올린 뒤 counting·measurement·orientation·relative position 같은 고수준 지식으로 집계한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 부분적이고 일시적인 2D 관찰을 3D 장면 상태로 연결하는 것, (2) 추론 단계가 진행되는 동안 이미 획득한 증거를 재사용하며 불필요한 도구 호출을 줄이는 상태 관리였다. S-Agent는 2D evidence acquisition(프레임 내 grounding/후보 영역 수집)→2D-to-3D geometric lifting(멀티뷰 기하 회복)→spatial knowledge aggregation(전용 experts가 task 지향 측정·관계로 해석)이라는 3단 도구 계층으로 이를 해결한다. 또한 Scene Memory는 객체 정체성과 누적된 기하/관계 사실을 유지하고, Agent Memory는 tool 호출·성공/실패·추론 맥락을 저장해 다음 evidence request를 정교하게 만든다.

- **Empirical Impact**: 실험에서 S-Agent는 MMSI-Bench·ViewSpatial-Bench(멀티 이미지), ReVSI·VSI-SUPER(비디오) 전반에 걸쳐 training-free zero-shot 조건에서 open-source 및 closed-source VLM의 성능을 일관되게 끌어올렸다. 특히 MMSI-Bench에서 Gemini 3 Pro 대비 약 1.2%, GPT-5.4 대비 약 4.5% 개선 같은 수치가 보고되며, 특히 동작/모션 지각과 multi-step reasoning에서 큰 이득이 관찰된다. 추가로 S-Agent가 생성한 공간 궤적 S-300K로 Qwen3-VL-8B를 SFT하면 S-Agent-8B가 더 강한 compact spatial agent로 발전해 Qwen3-VL-8B(베이스라인)보다 크게 개선되고 GPT-5.4·Gemini 3 같은 고성능 폐쇄 모델과도 견줄 만한 성과를 보여, “도구·기억·증거 누적”이 학습 없는 강화와 학습 기반 압축 모두에서 유효한 패러다임임을 시사한다.



### FreeStyle: Free Control of Style-Content Dual-Reference Generation from Community LoRA Mining (https://arxiv.org/abs/2606.20506)
Comments:
          35 pages, 26figures. Project page: this https URL

- **Prior Approaches**: 기존 Style-content dual-reference generation은 내용 기준(reference content)의 구조·의미를 유지하면서 다른 스타일(reference style)을 입히는 방식이지만, 콘텐츠 보존과 스타일 정렬이 동시에 요구돼 학습이 까다롭습니다. 특히 스타일 참조에서 의미가 새어 들어가는 semantic leakage를 막는 동시에 instruction following까지 만족해야 해서 모델이 쉽게 균형을 잃는 문제가 반복돼 왔습니다.

- **Core Contribution**: 이 논문은 FreeStyle이라는 확장 가능한 dual-reference 생성 프레임워크를 제안합니다. 커뮤니티 LoRA를 compositional anchor로 보고, 스타일과 콘텐츠를 깔끔히 분리한 대규모 triplet(Style-Reference, Content-Reference) 데이터를 생성·구성하는 파이프라인을 설계해 학습 재료의 병목을 완화합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 스타일 단계에서 스타일 참조의 누출을 억제하고 (2) 더 어려운 dual-reference 상황에서 위치 대응(positional correspondence) 기반 누출까지 잡아내는 것입니다. 이를 위해 두 단계 커리큘럼을 두고, 스타일-transfer 단계에서 attention-level enrichment constraint로 leakage를 억제하며, 어려운 단계에서는 frequency-aware RoPE modulation으로 positional-correspondence 기반 누출을 겨냥해 성능을 끌어올립니다.

- **Empirical Impact**: 저자들은 style-reference와 dual-reference를 함께 평가하는 벤치마크를 도입하고, style similarity·content preservation·aesthetics·instruction following·leakage rejection을 종합해 검증합니다. 또한 스타일 불변 Content Alignment Score(CAS)와 calibrated VLM 기반 Rejection Score를 사용해 생성 신뢰성과 누출 억제 효과를 계량화했으며, 실험에서 FreeStyle이 스타일 정렬·콘텐츠 보존·leakage suppression 사이의 균형을 강하게 달성했다고 보고합니다.



### How Fragile Are Training-Free AI-Generated Image Detectors? A Controlled Audit of Score Direction, Preprocessing, and Compression (https://arxiv.org/abs/2606.20488)
- **Prior Approaches**: 생성형 이미지 탐지는 전통적으로 real/fake 분류기를 training해 대응했지만, 압축·리사이징·미지 아키텍처에 취약하다는 한계가 반복됐다. 이후 generator fingerprints나 고주파 아티팩트, CLIP 기반 특징 공간 근접도 등을 활용하는 학습 없는(training-free) 점수들이 등장했으나, 보고 프로토콜이 논문마다 달라 공정 비교가 어려웠다. 특히 JPEG 기원 같은 데이터 포맷 바이어스와 점수의 “방향성(실제는 더 robust)” 가정이 예외를 가질 수 있다는 지적이 누적돼 왔다.

- **Core Contribution**: 이 논문은 새로운 탐지기를 제안하기보다, 대표적인 training-free 두 점수(AEROBLADE류 재구성 점수, RIGID류 잡음-민감도 점수)를 단일 통제 프로토콜로 “감사(audit)”한다. GenImage 기반 1,500장 벤치마크에서 7개 생성기와 JPEG 품질 70/50을 고정해, 백본·전처리·잡음 강도·표현 깊이·재인코딩 등 한 요인씩 바꿔 숫자가 얼마나 흔들리는지 정량화했다. 결론적으로 pooled AUROC 같은 요약 지표가 쉽게 오해를 낳을 수 있음을 실험적으로 보여준다.

- **Technical Challenges**: 핵심 도전은 구현 디테일과 하이퍼파라미터가 “방법 차이”처럼 보이는 혼선을 제거하는 것이다. 실제로 LPIPS 백본(AlexNet→VGG-16) 한 줄 변경만으로 AUROC가 크게 변하고, resize-512 vs native 전처리 정책에 따라 생성기별 순위가 최대 0.38 AUROC까지 뒤집힌다. 또한 RIGID류 점수는 σ(잡음 강도)·표현 깊이에 따라 방향이 바뀌어(예: 특정 생성기에서는 가짜가 더 robust) AUROC가 0.5 미만으로 뒤집힐 수 있으며, 이를 다루기 위해 생성기별 방향 검증과 provenance-controlled(실/가짜 동일 JPEG 인코딩) 재인코딩을 수행했다.

- **Empirical Impact**: 통제된 실험 결과, training-free 점수는 생성기·프로토콜 의존성이 커서 pooled AUROC를 맹신하기 어렵고, 최고 성능을 말하려면 생성기별 리포트와 조건별 분해가 필수라고 제안한다. 데이터 포맷 바이어스를 교정한 뒤에도 성능 구조는 일부 유지되지만, “compression helps” 같은 착시가 특정 생성기(BigGAN) 상호작용으로 국소화됨을 보여준다. 나아가 단순 z-score 융합은 보완성을 살리지 못했는데, 방향이 틀린 생성기에서 실패한 점수가 평균을 망가뜨리기 때문이라는 점이 실증됐다. 결과적으로 이 분야의 최소 보고 기준(백본/해상도/크롭을 1순위 변수로, 각 생성기군에서 점수 방향 검증, 기본 provenance-controlled 재인코딩 제공)을 강화하는 데 의미가 있다.



### Scalable Training of Spatially Grounded 2D Vision-Language Models for Radiology (https://arxiv.org/abs/2606.20477)
Comments:
          Accepted for MICCAI 2026. First two authors: equal contribution. Last two authors: equal supervision

- **Prior Approaches**: 기존 방사선 VLM은 보고서나 질의응답은 잘 생성하지만, 출력이 어떤 영상 영역을 근거로 했는지 “공간적으로” 확인하기 어렵다는 한계가 컸습니다. 또한 CT/MRI에서 수동 2D 공간 어노테이션이 부족해 학습 규모를 키우기 어려웠고, 2D 슬라이스 기반 시도도 제한된 데이터 규모(강한 필터링 등)로 확장에 제약이 있었습니다. 가슴 X-ray 중심의 grounded VLM 연구는 있었지만, 임상 CT/MRI 대규모에서의 공간 검증 학습은 상대적으로 덜 탐구됐습니다.

- **Core Contribution**: 이 논문은 수동 공간 어노테이션 없이 CT/MRI 슬라이스에 대한 visually grounded 학습을 가능하게 하는 RefRad2D와 RadGrounder를 제안합니다. RefRad2D는 임상 리포트에서 추출한 1.2M(독일어/영어) 이미지-텍스트 쌍과, TotalSegmentator 기반의 자동 공간 레이블을 결합한 대규모 데이터셋입니다. RadGrounder는 PaliGemma 2 기반 멀티태스크 VLM으로 보고서 생성, visual question answering, 그리고 bounding-box(또는 segmentation) 형태의 공간 그라운딩을 동시에 수행합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 단일 슬라이스 VLM에서 리포트의 시간적 표현이 환각을 유발하지 않도록 정제하고, (2) 3D 세그멘테이션 결과를 2D 입력에 맞춰 정확히 텍스트-영상을 정렬하며, (3) 그라운딩 감독을 추가해도 언어 품질을 해치지 않는 학습 구성이었습니다. 연구진은 GPT-OSS로 캡션의 시간 맥락을 제거하고, Gemma 3로 독일어→영어 번역을 품질 개선(샘플링·재판정)하며, GPT-OSS가 캡션의 해부학적 언급을 추출해 마스크 클래스 스키마(C=121)와 매칭하도록 자동 파이프라인을 구성했습니다. 모델 측에서는 추가 디코더/손실 없이 token-based bounding-box detection을 텍스트 생성으로 학습해 효과적인 로컬라이제이션을 구현했고, ablation 결과 그라운딩 감독이 VQA/보고서 생성 성능을 떨어뜨리지 않는 것도 확인했습니다.

- **Empirical Impact**: 외부 VQA 벤치마크 Slake와 VQA-RAD에서 RadGrounder는 전문 의료 VLM들과 경쟁하는 결과를 보이며, 특히 VQA-RAD에서 open F1을 비교 방법 중 최고 수준으로 달성했습니다. 또한 다운스트림 데이터만으로 fine-tuning하는 것에 비해, RefRad2D의 임상 데이터를 학습 혼합에 추가하면 open-ended VQA 성능이 개선되어 데이터 전이성(transferability)이 입증됐습니다. 더불어 bounding-box/segmentation 같은 공간 그라운딩 감독을 넣어도 언어 품질 저하가 없다는 점이 정량적으로도 뒷받침되어, “공간적으로 검증 가능한” 방사선 해석을 VQA 성능 저비용으로 제공할 수 있음을 시사합니다.



### PCFootprint: A Large-Scale Dataset and Benchmark for Vectorized Building Footprint Extraction from Aerial LiDAR Point Clouds (https://arxiv.org/abs/2606.20455)
Comments:
          14 pages, 9 figures

- **Prior Approaches**: 기존 건물 footprint 추출은 주로 2D 광학 이미지 기반으로, 픽셀 단위 마스크(세그멘테이션)나 벡터 폴리곤(코너 정점 회귀) 형태로 성능을 끌어올려 왔습니다. 하지만 그림자·가림·시점 왜곡 같은 광학 고유 문제와 residual relief displacement로 인해 경계가 끊기거나 어긋나는 경우가 잦았고, 무엇보다 고도 정보가 직접 제공되지 않아 LoD1(상세 수준 1) 모델링에 제약이 컸습니다. 3D point cloud 연구도 존재하지만, 대개 point-wise/instance 중심 데이터이거나 ALS에서 vectorized footprint을 제대로 비교·재현할 수 있는 대규모 공개 벤치마크가 부족했습니다.

- **Core Contribution**: 이 논문은 ALS(Airborne Laser Scanning) point cloud에서 vectorized building footprint을 추출하기 위한 최초의 대규모 공개 데이터셋 PCFootprint를 제안합니다. PCFootprint는 에스토니아의 Estonian Land and Spatial Development Board 데이터를 기반으로 33,000 타일(총 227,264개 building instance)의 벡터 정답(정점 순서 폴리곤)을 체계적으로 정렬해 제공합니다. 또한 지리적으로 mainland과 분리된 3,000 타일 cross-domain test set을 포함해 지역 변화에 대한 일반화 평가까지 가능하게 만들었습니다.

- **Technical Challenges**: ALS 기반 footprint 추출은 intra-class variance(같은 클래스 내 형태 변이)와 데이터 불균형, 그리고 복잡한 지형·식생 환경에서의 잡음이 성능을 크게 흔드는 것이 핵심 난관입니다. 논문은 데이터셋 구축 단계에서 tile 경계로 잘리는 인스턴스를 제거하고, 레이저 echo 특성과 density-connectivity climbing 절차로 높이 추정의 신뢰도를 높여 비현실적 height 속성을 정리했습니다. 더 나아가 point density(0.9~20.9 pts/□)와 건물 크기/형상 복잡도까지 다양한 분포를 포함하도록 설계해, 서로 다른 품질·스케일 조건에서도 벡터화 학습/평가가 가능하도록 했습니다.

- **Empirical Impact**: 저자들은 PCFootprint로 mainstream 방법들을 폭넓게 벤치마킹하고, 실제로 높은 intra-class variance, 데이터 불균형, 잡음이 복합적으로 나타나며 해결이 쉽지 않은 문제가 있음을 실험적으로 확인합니다. 평가 프로토콜은 mainland 기반 표준(intra-domain) 세팅과 islands 기반 교차 도메인(generalization) 세팅으로 나뉘며, 예측을 2D 벡터 폴리곤/마스크로 통일해 MS-COCO 스타일의 IoU 기반 비교가 가능하게 했습니다. 공개 벤치마크와 기준선(baseline)을 제공함으로써, 건물/도시 장면 이해와 geospatial 분석, 나아가 LoD1~2 건물 모델링으로의 연구 확장을 촉진할 것으로 기대됩니다.



### InfantFace: Detecting infant faces in neonatal clinical environments (https://arxiv.org/abs/2606.20449)
Comments:
          32 pages, 7 figures, 4 tables; supplementary information included

- **Prior Approaches**: 기존에는 일반 얼굴 검출기(예: 범용 face detector)를 그대로 적용해 영아 얼굴을 찾는 방식이 많았지만, 신생아 임상환경의 복잡한 배경과 조명 변화, 저조도 조건에서 정확도가 쉽게 떨어졌다. 또한 의료 장비나 시술로 얼굴이 부분 가려지는 경우가 잦아 시각 기반 평가가 더 어려워졌다.

- **Core Contribution**: 이 논문은 신생아 임상환경에 맞춘 one-stage YOLOv11m 기반 영아 얼굴 검출 모델을 제안한다. VGGFace2, CelebA, FDDB, WIDER FACE를 조합해 사전 학습/평가한 뒤, 113명의 독립 영아로 구성된 228개 비디오를 포함한 신생아 연구 데이터셋으로 fine-tuning을 수행했다.

- **Technical Challenges**: 핵심 기술적 난제는 임상 도메인에서의 시각적 편차(배경/조명/가림)와, 신생아용 공개 데이터셋의 부족으로 인한 일반화 검증의 어려움이다. 연구팀은 먼저 범용 데이터로 강한 초기 성능을 만든 뒤 임상 도메인 adaptation으로 AP50을 끌어올렸고, 향후에는 개인정보 보호와 윤리 기준을 전제로 공개 신생아 데이터셋 구축이 필요하다고 강조했다.

- **Empirical Impact**: 모델은 fine-tuning 전 AP50 0.87로 3종의 최신 범용 얼굴 검출기보다 높은 성능을 보였으며, 임상 도메인 adaptation 이후에는 AP50 0.96까지 개선됐다. 신생아 비침습 영상 기반 통증/표정, 통증 점수화, 심폐 신호 추출, 무호흡 경고 같은 비접촉 평가 파이프라인의 1단계 신뢰성을 높일 수 있다는 점에서 의미가 크다.



### Spectral Query-Key Product Weight Steering for Training-Free VLM Hallucination Mitigation (https://arxiv.org/abs/2606.20419)
Comments:
          Under Review

- **Prior Approaches**: 기존 비전-언어 모델(VLM)은 이미지 단서보다 사전 학습된 시각-언어 공접합 우선순위가 강해질 때 물체 환각을 낳는다는 문제가 있다. 해결은 주로 추가 학습(데이터·감독 필요)이나 디코딩 시 개입(contrastive decoding, logit adjustment, attention 재가중, 페널티 등)으로 이루어지며, 이는 비용이 커지거나 매 스텝 추가 계산이 생긴다. 또한 attention head/activation/서브스페이스를 다루는 mechanistic intervention류도 있지만, GQA에서 가중치 편집은 공유 key 때문에 부작용 위험이 있어 제약이 크다.

- **Core Contribution**: 이 논문은 데이터/학습 없이 한 번만 가중치를 편집하고 이후 디코딩 비용을 0으로 유지하는 QK Product Steering을 제안한다. 핵심은 attention logits의 전(前)단계인 query-key product를 타깃으로 하여, 선택한 middle layer의 각 head에서 지배적인 singular mode 몇 개를 억제하는 방식이다. GQA 호환성을 위해 shared key는 고정하고 query만 closed-form으로 업데이트해 편집 결과를 원래 모델 가중치 공간에 안전하게 복원한다.

- **Technical Challenges**: 직접적으로 key 가중치를 바꾸면 GQA의 공유 구조 때문에 여러 query head에 의도치 않은 영향이 번질 수 있다. 이를 피하기 위해 논문은 (1) per-head QK product의 thin SVD를 이용해 저랭크 스펙트럴 모드를 정밀히 억제하고, (2) 편집된 QK product를 query-only 최소 변화 업데이트로 사상(매핑)해 shared key를 유지하는 “GQA-safe query recovery”를 설계한다. 또한 QK product를 symmetric/antisymmetric 성분으로 분해해 환각 신호가 어느 채널에 주로 존재하는지 해석 가능하게 구성했다.

- **Empirical Impact**: 3개의 GQA 기반 VLM(Qwen2.5-VL-7B, InternVL3-8B, Pixtral-12B)에서 평균 relative CHAIRs 감소 4.0%를 달성했으며, matched random-mode 제어에서는 거의 변화가 없어 임의의 저랭크 섭동이 아님을 보여준다. symmetric/antisymmetric ablation 결과 환각 신호는 dominant QK mode에 특이적이고 주로 symmetric mutual-attention channel에 국한된 것으로 나타났다. 디코딩-time 완화와 달리 추가 데이터·fine-tuning·추가 추론 오버헤드 없이 작동한다는 점에서 실무적 적용성이 높다는 평가를 받는다.



### FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows (https://arxiv.org/abs/2606.20404)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 conditional diffusion/flow 모델은 조건 y를 “정적 힌트”로만 보고 샘플링을 open-loop로 진행해, 샘플이 제약에서 벗어나도 이를 되돌릴 메커니즘이 부족합니다. 반면 inference-time guidance 계열은 forward operator H를 통해 피드백을 보지만, hand-tuned 선형 업데이트로 학습-추론 불일치가 생기며 fidelity와 plausibility 사이에 선택이 강요되는 경향이 있습니다. 특히 depth/edge-to-RGB나 제약 기반 생성에서 재추출 측정이 입력과 불일치하는 문제가 반복해서 나타납니다.

- **Core Contribution**: FlowBender는 conditional diffusion과 flow matching을 closed-loop로 바꿔, 모델이 “자기 정렬 오차(alignment error)”를 1순위 입력으로 사용해 스스로 궤적을 교정하도록 학습합니다. 매 샘플링 단계에서 무유도 look-ahead로 clean signal 추정을 만들고, forward operator로 deviation(오차)을 계산한 뒤, refinement pass가 그 오차를 반영해 corrected velocity를 출력합니다. 또한 gradient 기반 formulation뿐 아니라 zero-order variant까지 제안해 non-differentiable 제약에서도 같은 교정 프레임을 적용할 수 있게 했습니다.

- **Technical Challenges**: 핵심 난제는 피드백 계산에 필요한 clean signal 추정이 네트워크 출력에서 나오는데, 그 피드백을 다시 입력으로 넣어야 하므로 시간 의존성이 생긴다는 점입니다. FlowBender는 이를 두 번 통과(two-pass) 실행으로 해결해, 첫 패스에서는 feedback을 0으로 두고 clean estimate를 만든 뒤, 둘째 패스에서 오차 입력을 받아 보정된 velocity를 얻습니다. 더 나아가 differentiable operator에서는 loss gradient를, JPEG처럼 미분 불가능하거나 black-box에선 measurement-space residual을 직접 입력하는 zero-order 학습을 제공하며, prior-step shortcut으로 추가 계산을 N+1 수준까지 줄일 수 있게 설계했습니다.

- **Empirical Impact**: 실험에서 FlowBender는 image-to-image translation(초해상도, depth/edge-to-RGB, JPEG restoration)과 3D mesh texturing 전반에서 기존 supervised fine-tuning, alignment-loss-augmented 학습, 그리고 FlowChef 같은 state-of-the-art inference-time guidance보다 fidelity와 plausibility를 동시에 개선했습니다. 특히 guidance는 보통 제약을 맞추는 대신 데이터 매니폴드 품질이 나빠져 FID가 급격히 악화되는 trade-off가 관찰된 반면, FlowBender는 그 반대로 양쪽 지표가 함께 좋아지는 패턴을 보였습니다. JPEG 복원 같은 비미분 설정에서도 zero-order 변형이 의미 있는 개선을 입증하며, 제약 교정이 “미분 가능성”에 덜 종속적이라는 실용적 의미를 강화했습니다.



### Geometry-Aware Superpixel Graph Transformer with Metadata for Skin Lesion Classification (https://arxiv.org/abs/2606.20390)
Comments:
          Accepted at MICCAI 2026

- **Prior Approaches**: 기존 피부 병변(흑색종 등) 분류는 CNN/ViT가 전역 벡터나 patch/그리드 토큰에 주로 의존하는 경우가 많아, 병변 하위 구조의 공간적 상호작용(예: 경계 불규칙, 색소 분포 패턴)을 약하게 반영하기 쉽습니다. 또한 환자 메타데이터(나이/성별/병변 위치 등)는 late fusion 형태의 보조 입력으로 처리되는 일이 잦아, 시각 근거가 어떤 방식으로 집계되는지에 대한 구조적 추론이 제한됩니다. 그래프 신경망(GNN)도 존재하지만, 엣지 속성을 활용한 기하-주의(attention)와 메타데이터를 그래프의 핵심 추론 공간에 통합한 방식은 상대적으로 드뭅니다.

- **Core Contribution**: 이 논문은 dermoscopic 이미지를 superpixel 기반의 그래프로 명시화하고, CNN 특징을 frozen 노드 표현으로 두는 Geometry-Aware Superpixel Graph Transformer(GeoMeta-GT)를 제안합니다. 병변의 미세한 하위 구조 배치(노드 간 관계)를 엣지 속성(거리/방향)으로 인코딩하고, 메타데이터를 모든 노드에 연결된 전용 context node로 넣어 그래프 내부에서 함께 추론하도록 설계했습니다. 그 결과, 단순 결합이 아닌 “관계 기반 멀티모달 융합” 관점에서 흑/양성(benign/malignant) 분류 성능을 끌어올립니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) RGB 강도만으로는 병변 구조 정렬이 어렵고, (2) 지역 간 기하 관계가 진단 단서이지만 이를 attention에 효과적으로 반영해야 하며, (3) 메타데이터를 late fusion이 아닌 관계 추론에 녹여야 한다는 점입니다. 해결책으로 frozen CNN feature 공간에서 SLIC을 수행해 공간적으로 일관된 superpixel을 만들고, 노드 특징은 각 영역의 mean/max 기반 통계로 안정적으로 요약합니다. 이어 엣지 속성을 attention의 query–key 상호작용에 주입하고, similarity-weighted propagation으로 의미적으로 일관된 영역 간 정보 교환을 강화해 잡음으로 인한 표현 흔들림을 줄였습니다.

- **Empirical Impact**: ISIC2024/HAM10000/PAD-UFES-20/HIBA의 4개 벤치마크에서 GeoMeta-GT는 전반적으로 SOTA를 일관되게 상회했으며, 특히 ISIC2024에서 98.61%로 경쟁 멀티모달 대안(96.69%)보다 큰 폭의 향상을 보였습니다. PAD-UFES-20과 HIBA처럼 촬영 조건 변동과 잡음이 큰 데이터에서도 각각 97.17%, 95.41%로 강건한 성능을 입증해, 병변 이질성과 미세 차이를 관계 추론으로 흡수하는 효과를 시사합니다. 절대 성능뿐 아니라 ablation에서 geometric edge encoding과 메타데이터-as-node fusion이 평균적으로 몇 퍼센트포인트씩 성능을 끌어올리는 것으로 확인되어, 그래프-중심 설계의 실질적 기여가 정량적으로 뒷받침됩니다.



### Reliability-Aware Prototype Calibration for Frozen Pose-Flow Video Anomaly Detection (https://arxiv.org/abs/2606.20312)
Comments:
          15 pages, 5 figures, 7 tables. Code available at this https URL

- **Prior Approaches**: 기존 pose-flow 기반 one-class 비디오 이상탐지는 normalizing flow의 likelihood(예: negative log-likelihood)로 이상 점수를 매기지만, likelihood는 의미적 이상과 직접 대응되지 않을 수 있습니다. 또한 정상 동작은 walking/정지/회전/상호작용처럼 다중 모드(multimodal)인데, 단일 스코어는 이 구조를 충분히 반영하지 못합니다. 더불어 상류 pose-observation(가림, truncation, 추적 스위치 등) 노이즈로 keypoint confidence가 흔들리면 이상 점수 해석이 더 애매해지는 한계가 큽니다.

- **Core Contribution**: 이 논문은 frozen-detector(백본, 캐시된 skeleton track, 평가 파이프라인 고정) 환경에서 score-level calibration 문제로 재정의합니다. Reliability-Aware Prototype Calibration(RPC)은 flow 기반 밀도 신호를 그대로 두면서, latent 공간에서 normal-mode의 nearest prototype deviation을 표준화해 likelihood 스코어에 보정 항으로 더합니다. 여기에 keypoint confidence는 prototype(기하) 증거의 “기여 강도”만 게이팅하도록 설계해, 신뢰도 저하를 곧바로 이상 근거로 오염시키지 않습니다.

- **Technical Challenges**: 핵심은 (1) 단일 likelihood가 숨기는 다중 모드 정상 구조를 어떻게 “추가 신호”로 복원하느냐, (2) pose 관측 신뢰도가 낮을 때 prototype distance가 pose error까지 포함해 잘못된 이상 근거가 되지 않게 하느냐입니다. RPC는 normal 학습 샘플에서 k-means로 prototype을 만든 뒤 diagonal Mahalanobis prototype deviation을 계산하고, likelihood와 distance를 normal 학습 통계로 표준화해 스케일 불일치 문제를 줄였습니다. 마지막으로 keypoint confidence의 clipped mean으로 신뢰도 q를 만들고, prototype 보정 항에만 q^γ 형태의 게이트를 적용해 보정의 보수성을 확보했습니다.

- **Empirical Impact**: 두 개의 frozen pose-flow 백본과 네 개 데이터셋(총 8개 백본-데이터 조합)에서 RPC는 프레임 수준 AUROC를 모든 설정에서 개선하며, 증가폭은 0.34~4.49%p(평균 2.03%p)로 보고됩니다. ablation/신뢰도 분석에서는 diagonal Mahalanobis prototype deviation이 주요 개선 신호였고, 게이팅은 pose 신뢰도가 낮을 때 특히 효과가 컸습니다. 재학습 없이 cached pose-flow 시스템을 가볍게 업그레이드할 수 있다는 점에서, 감시/재현 제약이 큰 환경에 실용적 의미가 큽니다.



### Through the PRISM: Preference Representation in Intermediate States of Video Diffusion Models (https://arxiv.org/abs/2606.20310)
- **Prior Approaches**: 기존 Video Reward Model(VRM)은 주로 VLM 기반으로 픽셀 공간에서 동작하며, 노이즈가 큰 중간 단계(timestep)에서는 선호 예측 정확도가 크게 떨어졌다. 또한 평가를 위해 매번 VAE 디코딩을 반복해야 해서 Best-of-N 같은 inference-time scaling을 적용하면 계산비용이 선형으로 폭증한다. 더 나아가 보상 모델과 생성 백본의 구조가 분리돼 있어 joint scaling이나 self-improving 같은 순환 학습이 어렵다는 한계도 제기됐다.

- **Core Contribution**: 이 논문은 “강력한 비디오 생성기가 노이즈가 심한 latent에서도 선호를 판별할 수 있는가”라는 질문에 답하며 PRISM을 제안한다. PRISM은 Video Diffusion Transformer 백본을 freeze한 채, noisy latent의 중간 상태에서 선호 신호를 직접 추출하는 Latent Video Reward Modeling 패러다임을 제시한다. 또한 Query-based Aggregation head로 중간 표현에서 preference를 디코딩-free로 읽어내며, 생성기 성능과 evaluative power 사이의 상관도 체계적으로 분석한다.

- **Technical Challenges**: 핵심 난제는 noisy latent의 고차원 spatio-temporal feature에서 미세한 선호 단서를 효과적으로 끌어내는 것이다. PRISM은 첫 N_b block의 중간 특징을 사용해 저수준 모션과 고수준 의미 정렬을 유지하면서도 시간에 따른 노이즈에 강한 표현을 확보하고, 학습 가능한 query를 cross-attention에 투입해 토큰 중요도를 적응적으로 재가중한다. 더해 per-timestep reward를 학습해 임의의 노이즈 수준에서 평가 가능하게 만들고, pairwise preference 데이터에 대해 BTT(Bradley-Terry model with Ties) 기반 우도 최대화로 학습 안정성을 확보했다.

- **Empirical Impact**: 실험에서 PRISM은 VideoGen-RewardBench 및 추가 OOD 성격의 VLRM-Bench에서 pixel 기반 VRM 대비 모든 noise timestep에서 더 높은 선호 예측 정확도를 보이며 SOTA를 달성했다. 특히 high-noise 구간에서 다른 방법들의 성능 붕괴(예: tie-collapse) 현상을 피하고, 결과적으로 early-stage Best-of-N sampling이 가능해져 relative time cost를 크게 줄였다(최대 7.6× 속도up). 또한 성능이 우수한 diffusion backbone일수록 평가력도 높다는 상관을 보여 self-improving 비디오 백본 방향의 가능성을 제시한다.



### GEN-Guard: Correcting Generalization Failures for Deployable Federated Surgical AI (https://arxiv.org/abs/2606.20303)
- **Prior Approaches**: 연구들은 수술 비디오 AI를 위한 Federated Learning(FL)이 프라이버시를 지키면서도 성능을 중앙집중학습 수준으로 끌어올릴 수 있음을 보여줬습니다. 하지만 모델 선택은 주로 참여 병원의 검증 데이터에 의존해, 다른 기관(미참여 분포)으로 갔을 때 일반화가 크게 깨질 수 있습니다. 특히 수술 영상의 기관별 장비·워크플로 차이로 non-IID가 심한데도, 평가/선정 편향이 성능 누출(performance leakage)로 이어진다는 점은 충분히 다뤄지지 않았습니다.

- **Core Contribution**: 이 논문은 FL 수술 AI에서 “성능 누출”이 실제 배포에서 Model Selection Failure(MSF)를 유발한다고 문제를 규정합니다. 이어서 GEN-Guard라는 사후(post-hoc) 프레임워크를 제안해, 표준 FL 수렴 이후에 후보 체크포인트들로부터 일반화에 실패한 모델을 찾아내고(Generalization Detection) 교정합니다(Generalization Correction). 핵심은 Client-Blocked Evaluation(CBE)로 검증 편향을 차단하고, Disagreement-Aware Distillation(DAD)로 기관 간 강건성을 높이는 것입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 “참여 클라이언트 섞인 검증 성능”이 “미지의 기관 성능”을 믿을 만하게 대리하는지 검증하고, 실패 시 그 오차를 표현 공간에서 줄이는 것입니다. GEN-Guard는 CBE에서 학습에 쓰이지 않은 클라이언트 분포를 격리해 평가함으로써 performance leakage 가능성을 직접 차단하고, 이후 DAD에서 CBE-강건 모델과 기존 선택 모델의 불일치(disagreement)를 특징 레벨에서 학습해 residual 보정으로 옮깁니다. 또한 DAD는 레이블 없이도(기본 zero-shot) 동작하도록 설계해, 배포 단계에서 추가 지도비용 없이 적응을 노립니다.

- **Empirical Impact**: 실험에서는 표준 평가 관행만 썼을 때 MSF가 80%를 넘는 등, 모델 선택이 매우 자주 잘못된다는 점을 정량화합니다. Multi-Chole(수술 단계 인식)과 PolypGen(용종 분할) 두 멀티센터 벤치마크에서 GEN-Guard는 in-federation F1을 최대 2점, 미지 기관 성능을 최대 3점, 최악 기관 성능을 3~9점까지 개선하며 실패를 일관되게 교정합니다. 또한 다양한 FL 알고리즘(FedAvg/FedProx/SCAFFOLD)에도 공통으로 이득을 주고, 최악 케이스 변동성도 줄여 “배포 신뢰성”을 강화하는 의미가 큽니다.



### CUPID: Reconstructing UV Texture Maps for Interpretable Person-of-Interest Deepfake Detection (https://arxiv.org/abs/2606.20302)
- **Prior Approaches**: POI(Person-of-Interest) 딥페이크 탐지는 크게 두 흐름으로 나뉜다. POI 데이터를 학습에 쓰는 방식은 POI마다 별도 모델을 학습해야 해 데이터 부담이 크고, POI를 추론에서만 쓰는 방식은 기준 영상과의 잠재공간 유사도로 판단하지만 post-processing에 대한 견고성·효율·해석 가능성을 동시에 만족시키기 어렵다.

- **Core Contribution**: CUPID는 POI 비디오 딥페이크를 “UV 텍스처맵 + Masked Autoencoder(MAE)”로 판별하는 재구성 기반(detector in general POI setting) 방법을 제안한다. 훈련 단계에서 POI 딥페이크 영상이 필요 없고, 심지어 특정 POI 자체를 학습에 포함하지 않아도, MAE가 학습한 신원 일관 표현이 학습 중 보지 못한 신원에 대해서도 잘 일반화되도록 설계됐다.

- **Technical Challenges**: 핵심 난제는 ① 다양한 리사이즈·압축 등 후처리에 대한 강건성, ② 빠른 추론을 위한 효율, ③ “왜 가짜라고 판단했는지”를 보여주는 해석 가능성을 함께 달성하는 것이다. CUPID는 프레임 공간 대신 3DMM 기반 UV 텍스처맵으로 의미적 대응(픽셀의 얼굴 부위 일치)을 고정한 뒤 MAE의 마스킹 재구성 학습에 더해 다중 깊이 대조학습과 perceptual loss까지 결합해, 판별에 유리한 압축 잠재공간을 만들고 UV 공간에서 residual map으로 의사결정 기여 영역을 시각화한다.

- **Empirical Impact**: 4개 딥페이크 데이터셋에서 CUPID는 다수 데이터셋에서 최신 기법을 능가하며, 특히 강한 downscaling과 compression에서도 전체적으로 가장 좋은 견고성을 보였다고 보고한다. 또한 추론 속도는 더 빠르며, 임계값 보정(threshold calibration)에 대한 민감도도 낮아 실제 운용 관점의 신뢰성이 높다는 점이 강조된다.



### CMDS-AD: Cross-Modal Dual-Stream Decoupling for Few-Shot Anomaly Detection (https://arxiv.org/abs/2606.20300)
Comments:
          Accepted to ECCV 2026!

- **Prior Approaches**: 기존 multi-modal anomaly detection(MAD)은 2D RGB와 3D 정보를 결합하지만, few-shot(특히 1~4샷)에서는 2D-3D modality gap 때문에 정렬 오차가 커지고 오탐(false-positive)이 증가한다. 또한 대부분은 공간 특징을 균일하게 처리해 저주파 안정 구조와 고주파 국소 결함 신호가 섞이며, 결과적으로 정밀한 미세 결함 분리가 어려워진다.

- **Core Contribution**: 본 논문은 few-shot MAD를 위한 Cross-Modal Dual-Stream Anomaly Detection(CMDS-AD)를 제안하며, 핵심은 실(stream)과 추정(stream)을 분리해 주파수 성격을 다르게 다루는 구조다. LoRA-guided diffusion으로 RGB를 보강하는 한편, 3D에서는 diffusion 기반 normal estimator를 저주파(non-linear low-pass) 필터로 재해석해 “순수 저주파 앵커” 스트림을 만들고, 이를 원본 스트림과 함께 결함을 분리·강조한다.

- **Technical Challenges**: 문제는 (1) 이중 스트림이 서로 다른 진폭/특성으로 인해 정렬이 무너져 feature collapse를 유발할 수 있고, (2) 2D·3D 의미를 스케일별로 정교하게 맞추지 못하면 잡음이 오탐으로 이어진다는 점이다. 이를 위해 Coordinate-Aware Hierarchical Feature Mapper로 좌표(위치) 및 멀티스케일 계층을 적응 정렬하고, Cross-Modal Multiplicative Anomaly Scoring에서 두 모달리티의 동시 이상만 통과시키는 곱셈 기반 필터로 modality-specific 노이즈를 억제한다.

- **Empirical Impact**: 실험에서 CMDS-AD는 MVTec 3D-AD의 1-shot 조건에서 I-AUROC 5.7%, AUPRO 2.0% 절대 성능 향상을 보이며 EyeCandies에서도 각각 7.7%, 5.6% 개선하며 새로운 SOTA를 달성했다. 더 나아가 2D는 질감/색 변화, 3D는 형태 불연속을 잘 포착하는 상보성이 정성·정량 모두에서 확인되며, 실험의 주요 컴포넌트(이중 스트림, 계층형 매퍼)가 성능을 일관되게 끌어올리는 것으로 나타났다.



### U$^2$Mamba: A Two-level Nested U-structure Mamba for Salient Object Detection (https://arxiv.org/abs/2606.20282)
Comments:
          6 pages, 2 figures

- **Prior Approaches**: 기존 SOD는 CNN(FCN) 기반으로 해상도를 낮춰 전역 문맥을 키우지만, 다운샘플링으로 경계에 필요한 고해상도 디테일이 줄어드는 문제가 있다. Transformer(ViT류)는 self-attention으로 장거리 의존성을 잘 잡지만, 고해상도에서 계산량이 커져 실사용 제약이 생긴다. 한편 Mamba 같은 structured SSM은 긴 시퀀스를 효율적으로 모델링하지만, 시각 SOD에 맞게 전체 아키텍처 깊이·컨텍스트 결합·SOD 특화 계층적 학습을 충분히 설계하지 못한 연구가 많았다.

- **Core Contribution**: 이 논문은 Mamba를 SOD에 맞게 처음부터 구성한 중첩 U-구조 네트워크 U$^2$Mamba를 제안한다. 핵심은 멀티스케일 Mamba U-block(MMUB)으로, 한 블록 안에서 U-Net의 국소 정밀도와 Mamba의 장거리 문맥을 함께 얻도록 설계해 깊이를 늘리면서도 비용을 억제한다. 또한 nested U-structure가 얕은·깊은 레이어에서 다양한 receptive field를 통합해 해상도 제약 없이 더 풍부한 컨텍스트를 모으도록 한다.

- **Technical Challenges**: Mamba의 효율성을 살리면서도 SOD의 정밀 경계 보존을 위해 “깊이 확대 vs 고해상도 연산 폭증”의 균형이 어려웠다. 저자들은 대부분의 연산을 downsample된 표현에서 수행하되, 고해상도 얕은 특징을 명시적으로 유지하는 중첩 U-설계를 통해 경계를 보존한다. 더 나아가 MMUB에서 저주파/고주파 분해로 공간적 중복을 줄여 계산·메모리를 절감하고, 계층적 deep supervision으로 각 레벨 출력 간 일관성을 BCE+KL 조합으로 강제해 학습 안정성을 확보했다.

- **Empirical Impact**: DUTS-TR로 학습하고 ECSSD, PASCAL-S, DUTOMRON, HKU-IS, DUTS 등 주요 벤치마크에서 기존 SOTA들과 비교해 경쟁력 있는 성능을 보였다. 특히 max F$\beta$ 개선과 낮은 MAE로 더 정확한 saliency 경계와 일관된 관심 영역을 생성한다는 정성 결과도 제시됐다. ablation에서는 hierarchical supervision과 MMUB 각각이 성능을 끌어올리며, 최종 조합이 최고 성능(maxF$\beta$: 0.904, MAE: 0.024)을 달성했고, 효율 측면에서도 parameters/FLOPs가 줄고 inference 속도도 더 빠른 trade-off를 보였다.



### Single-Stage Hierarchical Rectification for Weakly Supervised Histopathology Segmentation (https://arxiv.org/abs/2606.20250)
Comments:
          Accepted to MICCAI 2026. This is the pre-review submitted version, not the camera-ready version. The final authenticated version will be available in the MICCAI 2026 proceedings

- **Prior Approaches**: 기존 병리 분야 WSSS(Weakly Supervised Semantic Segmentation)는 CAM(Class Activation Map) 생성→오프라인 pseudo-mask 정제→세그멘테이션 재학습의 다단계 파이프라인에 의존하는 경우가 많습니다. 이 구조는 국소 텍스처 편향에서 비롯된 false-positive가 CAM에 먼저 들어가고, 이후 재학습이 이를 “정정”하기보다 “증폭”할 수 있다는 오류 전파 문제가 지적돼 왔습니다. 또한 단계가 분리돼 있어 학습 비용이 커 임상 적용의 병목이 되기도 합니다.

- **Core Contribution**: 이 논문은 Single-Stage Hierarchical Rectification(SSHR)로, CAM 생성 이후의 사후 정제가 아니라 forward pass 동안 특징을 사전에 정제해 단일 학습 루프에서 고품질 activation map을 만드는 접근을 제안합니다. 핵심 모듈인 Hierarchical Feature Rectification Module(HFRM)은 깊은 계층의 전역 의미(semantic)로 얕은 계층의 국소 이상(local anomalies)을 걸러내어 오류 전파 구조 자체를 줄입니다. 결과적으로 다단계 재학습 없이도 WSSS 성능을 끌어올리면서 학습 시간을 크게 단축하는 것이 기여의 중심입니다.

- **Technical Challenges**: 문제는 전역 의미와 국소 경계 디테일이 동시에 필요한데, 얕은 CNN 층은 수용영역이 작아 국소 잡음에 취약하다는 점입니다. SSHR은 (1) Global Semantic Rectification(GSR)로 깊은 특징에서 얻은 전역 컨텍스트를 채널 단위 attention으로 억제해 클래스 불일치 false-positive를 줄이고, (2) Contextual Homogenization(CH)로 large-kernel depthwise convolution을 적용해 공간적 고립 outlier를 주변 조직과 구조적으로 동화시킵니다. 또한 zero-initialized residual 연결로 기존 백본에 비침습적으로 끼워 넣어 학습 초기의 그라디언트 충격을 완화하는 설계를 사용합니다.

- **Empirical Impact**: LUAD-HistoSeg와 BCSS 두 데이터셋에서 SSHR은 기존 SOTA 다단계 방법을 단일 단계 학습만으로 능가하며, mIoU 기준 각각 77.93%, 71.82%를 보고했습니다. 특히 다단계 파이프라인 대비 학습 소요가 2~5배 줄어들어(2.23×~5.42×) WSI의 거대 해상도 때문에 수 주~수 개월 걸리던 개발 주기를 단축하는 실용적 의미가 큽니다. 또한 패치당 추론 지연이 9.10 ms 수준으로, 성능 향상과 함께 추론 효율도 유지되는 점이 임상 전환에 유리하다는 평가를 뒷받침합니다.



### SPOT-E: Test-Time Entropy Shaping with Visual Spotlights for Frozen VLMs (https://arxiv.org/abs/2606.20244)
- **Prior Approaches**: 기존 비전-언어 모델(VLM)용 추론 시각 개입은 이미지에 마스크/오버레이를 하거나 크롭·줌으로 해상도를 재배치하는 방식으로, 모델 가중치는 고정한 채 grounding을 개선하려는 시도가 많습니다. 다만 대부분 open-loop라서, 강조한 영역이 실제 정답 결정 증거로 ‘사용’됐는지 검증 메커니즘이 부족해 오개입 시 실패가 그대로 누적됩니다. 또한 불확실성(엔트로피)을 단순히 낮추는 접근은 evidence-grounded 확신과 shortcut 붕괴를 구분하지 못한다는 한계가 드러납니다.

- **Core Contribution**: 이 논문은 answer-span prediction entropy를 ‘증거 사용도’의 모델 내부 피드백 신호로 제안하고, 엔트로피 감소가 의미하는 바가 모호하다는 점을 정리합니다. 이를 해결하기 위해 low-entropy anchors를 도입해, 기준 입력에서 이미 결정적인 토큰에 해당하는 위치의 안정성을 유지하면서 엔트로피를 낮추는 entropy-shaping objective를 설계합니다. 결과적으로 label 없이도 “증거 기반 확신은 강화하고, shortcut collapse는 억제”하는 방향의 테스트타임 적응을 가능하게 합니다.

- **Technical Challenges**: 핵심 난제는 엔트로피를 줄이되, 증거를 실제로 노출한 경우와 단지 추론을 단순화해 확신을 만드는 경우를 구분해야 한다는 점입니다. 저자들은 baseline에서 near-deterministic한 토큰 위치를 anchors로 잡고, 개입 후 anchor 위치의 엔트로피가 증가하면 패널티를 주어 ‘비파괴적인 엔트로피 감소’를 유도합니다. SPOT-E는 이를 구현하기 위해 CLIP 기반 question-conditioned visual spotlight를 LoRA로 가볍게 학습하며, per-instance Group Relative Policy Optimization (GRPO)로 spotlight 마스크를 짧게 최적화합니다.

- **Empirical Impact**: SPOT-E는 여러 frozen VLM 계열(오픈소스 백본과 토큰 로그확률이 노출되는 API 포함)에 걸쳐 evidence-intensive 벤치마크에서 일관된 성능 향상을 보이며, 특히 작은 텍스트·숫자·국소 기호 의존 과제에서 이득이 큽니다. 시각 잡음·해상도 저하·부분 가림 같은 visual corruption에서도 성능 하락을 완화해 robustness가 개선됨을 보였습니다. 또한 엔트로피 분석에서 오답은 엔트로피가 높아지고 정답은 낮게 유지되어, confident-but-unsupported 오류를 줄이며 정오 분리가 강화되는 것으로 나타났습니다.



### BAFIS: Dataset + Framework to assess occupational Bias and Human Preference in modern Text-to-image Models (https://arxiv.org/abs/2606.20241)
Comments:
          Accepted at the IEEE Winter Conference on Applications of Computer Vision, WACV 2026

- **Prior Approaches**: 기존 연구는 주로 영어 프롬프트 기반 자동 지표로 편향과 품질을 측정했지만, 성별·인종 편향이 실제 사회적 다양성을 충분히 반영하지 못한다는 문제가 반복해서 관찰됐다. 또한 human preference를 직접 반영하는 벤치마크는 있었지만, 편향 평가까지 함께 제공하진 못해 실사용 인지 수준과의 연결이 약했다. 특히 멀티링구얼 환경에서 편향을 체계적으로 비교한 평가는 제한적이었고, 모델 계열과 상용/오픈 모델 확장도 뒤따르지 못했다.

- **Core Contribution**: 이 논문은 직업 관련 이미지 생성에서 language(언어)가 편향에 미치는 영향을 정량화하고, 인간 선호 피드백을 함께 반영하기 위해 BAFIS(Battle-Arena for Fair Image Synthesis)와 21,140장 규모의 합성 데이터셋을 제안한다. Midjourney v6.1, Stable Diffusion 3 Medium, DALL-E 3, Playground v2.5, FLUX.1-dev 다섯 모델을 성별·인종 편향, 이미지 품질, prompt alignment(프롬프트 정합성) 관점에서 포괄 평가한다. 나아가 독일 연방고용청 통계와의 비교로, 편향 결과를 사회 맥락에서 해석할 수 있게 했다.

- **Technical Challenges**: 주요 과제는 (1) 언어에 따른 편향 변화를 잡아내면서 (2) 인간이 ‘어떤 결과를 더 낫다’고 느끼는지 편향 평가와 분리해 측정하는 것이었다. 연구팀은 MAGBIG의 직업 프롬프트를 독일어·영어로 확장해 1,057개 프롬프트(그룹 프롬프트 포함)로 합성 데이터를 만들고, BAFIS 웹에서 모델 간 익명 pairwise battle로 bias·품질·정합성에 대한 투표를 수집했다. 이후 Elo 등급을 Bradley-Terry 기반 MLE-Elo로 안정화해 인간 선호 순위를 만들고, YoloV8·FairFace로 얼굴/인종 분포를 자동 분석해 자동 지표(MAD, FID, MagFace, CLIP)와의 상관도 함께 검증했다.

- **Empirical Impact**: 결과는 text-to-image 모델들이 성별과 인종에서 체계적 편향을 보이며, 특히 독일어 프롬프트에서 인종 편향이 더 강해지는 패턴을 보여준다(백인 피부톤 쏠림 증가). 인간 선호 기반 평가에서는 DALL-E 3가 독일어에서 prompt alignment에 강하고, FLUX.1-dev는 인간이 느끼는 전체 품질 선호에서 상위권이지만 자동 지표(FID, MagFace)와 일치하지 않는 경우가 있어 ‘기계 지표만으로 인간 선호/편향을 대변하기 어렵다’는 시사점을 남긴다. 저자들은 편향 완화와 모델 평가에 human preference를 정기적으로 포함해야 더 공정하고 포용적인 T2I를 만들 수 있다고 강조한다.



### Cinematic Compositing Using Character-Environment-Harmonized Video Generation Models (https://arxiv.org/abs/2606.20233)
- **Prior Approaches**: 기존 그린스크린 합성은 alpha matting 기반의 픽셀 블렌딩에 의존해, 캐릭터-환경 간 양방향 물리/광학 상호작용을 학습하지 못했다는 한계가 컸습니다. 영상 inpainting·relighting 계열은 각각 배경/조명 일관성을 부분적으로 다루지만, 새 환경에 대해 E2C(환경→캐릭터) 조화나 캐릭터 움직임에 따른 C2E(캐릭터→환경) 반응을 함께 만족시키기 어렵습니다. 또한 props는 실제 보존·프록시 교체·가상 생성의 서로 다른 처리 요구를 통합해 다루지 못해, 상호작용 장면에서 부자연스러운 재질/그림자/접촉이 남았습니다.

- **Core Contribution**: 본 논문은 그린스크린 합성을 생성형 문제로 재정의하고, C2E 물리 상호작용과 E2C 조명 조화를 단일 end-to-end video diffusion 프레임워크로 동시에 모델링합니다. 특히 interactive props를 위해 tri-mask 기반의 적응형 처리(보존·프록시 교체·완전 생성)를 통합 인터페이스로 제공해 촬영 시나리오 차이를 흡수합니다. 아울러 reference-conditioned 메커니즘으로 목표 환경 합성과 prop 교체를 더 정밀하게 제어하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 이진 마스크로는 보존/깊이앵커/완전생성을 구분할 수 없고, (2) RGB만으로는 접촉/가림 같은 공간적 C2E를 안정적으로 학습하기 어렵다는 점입니다. 이를 위해 tri-mask를 RGB 보존/깊이 보존 상태로 분해하고, RGB-D joint denoising을 통해 캐릭터·props·환경 간 3D 관계를 함께 복원하도록 했습니다. 또 고가 렌더링 없이도 다중 조명 학습쌍을 만들기 위해 prior-driven data curation 파이프라인을 구축하고, 기준선(reference)과 첫 프레임 공간 배치를 결합한 reference injection으로 controllability를 높였습니다.

- **Empirical Impact**: 실험에서는 합성 품질을 종합 평가하는 synthetic benchmark와 실제 그린스크린 영상 기반 user study를 통해 성능을 검증했습니다. 정량적으로는 identity preservation, foreground/background 유사도 등 핵심 지표에서 기존 end-to-end가 아닌 파이프라인·단일 태스크 접근을 크게 앞섰고, 미적 점수(Aesthetic Score)에서도 상위권을 유지했습니다. 결과적으로 cinematic-quality 동적 비디오 합성에서 새로운 state of the art를 제시하며, VFX/영상 편집 자동화에서 양방향 물리+광학 조화의 실용 가능성을 한 단계 끌어올렸다는 점에서 의미가 큽니다.



### DeepForestVisionV2: Ecology-Driven Taxonomy Expansion for Camera-Trap Monitoring in African Tropical Forests (https://arxiv.org/abs/2606.20223)
Comments:
          Accepted at ICPR 2026 - Computer Vision for Biodiversity Monitoring and Conservation Workshop

- **Prior Approaches**: 카메라트랩 자동 분류는 Mbaza, Zamba, SpeciesNet, DeepForestVision 같은 오픈 도구가 있으나, 처리 방식과 라벨 세분화가 현장 배치 제약을 충분히 반영하진 못했습니다. 특히 DeepForestVision은 closed-canopy 지상 배치에 맞춘 35-class 라벨이라 수목성 영장류, 조류, 반수생 분류군, 그리고 livestock 같은 인위적 혼입이 늘어나는 구간에서 해상도가 거칠어집니다.

- **Core Contribution**: DeepForestVisionV2는 35-class에서 64-class(61개 동물 + human, vehicle, blank)로 라벨 공간을 확장해 vertical stratification, scene openness, anthropogenic interfaces의 세 가지 생태적 배치 그라디언트를 더 잘 반영합니다. AddaxAI 기반의 기존 offline 처리 워크플로우(사진·비디오 모두)를 그대로 유지하면서, 실제 현장 활용도를 높이는 방향으로 설계를 바꿨습니다.

- **Technical Challenges**: 핵심 난제는 “라벨을 늘리면 모델이 현장 다양성(사이트·서식지·카메라 조건)을 따라갈 수 있는가”와 “비디오 파이프라인까지 포함한 end-to-end 성능을 보장할 수 있는가”입니다. 이를 위해 MegaDetector v5로 탐지-크롭을 만든 뒤 DINOv3 ViT-B/16을 fine-tuning하고, 카메라 단위로 데이터 분할·클래스별 검증 샘플 상한을 두어 국가 간/조건 간 강건성을 점검했습니다.

- **Empirical Impact**: 교차국가 cropped-photo 검증에서 DeepForestVisionV2는 accuracy 0.86, macro-F1 0.82, balanced accuracy 0.81을 달성했습니다. 우간다의 held-out 비디오 벤치마크에서는 forest-interior에서 정확도는 유지(0.89→0.89)하면서 탐지되는 분류군을 22→29로, riverbanks에서 0.72→0.73 근처의 성능을 유지하며 4→9로 늘렸고, park-edge에서는 정확도를 0.62→0.86으로 끌어올리며 false alarm 11을 0으로 낮췄습니다. 또한 reliability/ECE 결과로 confidence는 under-confident 경향이 있어 자동 알림 임계값 적용 전 보정 필요성을 시사합니다.



### Evaluation of Image Matching for Art Skills Assessmen (https://arxiv.org/abs/2606.20199)
Comments:
          MAPR 2024

- **Prior Approaches**: 기존 연구는 스케치 품질을 저자/전문가의 주관적 평가를 기준으로 하거나, 미리 정한 스케일에서 템플릿과의 매칭 정도를 점수화하는 방식이 많았다. 또한 SIFT·ORB·SURF 같은 handcrafted feature와 CNN 계열을 사용한 유사도 학습 시도가 있었지만, 공정한 유사도 스코어링과 잡음·전처리 민감도를 완전히 해소하진 못했다. 특히 전통적 매칭은 복잡한 파이프라인이 필요하고, CNN 기반은 입력 특성(예: 색/스케일)에 따른 편차가 남는 편이었다.

- **Core Contribution**: 이 논문은 사용자가 그린 스케치를 템플릿 사진과 “이미지 유사도”로 직접 비교해 그리기 실력을 수치화하는 프레임워크를 제안한다. 구체적으로 SIFT keypoint 매칭과 VGG16 기반 Siamese Network(유사도는 Cosine·Euclidean)를 함께 실험해, 어떤 접근이 실력 구간 분류에 더 적합한지 비교한다. 또한 단순히 유사도 점수를 만들고 끝내지 않고, 사람 평가(초/중/상/프로)와 맞닿는 범위 설계까지 포함해 실용성을 높인다.

- **Technical Challenges**: 핵심 과제는 스케치마다 나타나는 잡음·이중선·왜곡과 전처리 파라미터(예: auto Canny, 형태학적 처리)가 유사도 계산에 미치는 영향을 안정적으로 제어하는 것이다. 논문은 500×500 리사이즈, 흑백 변환, Gaussian Blur, morphological operations, 그리고 Flann 기반 KNN 매칭 필터링(거리 비율 임계치) 같은 전처리·매칭 단계로 SIFT의 견고함을 강화한다. CNN/Siamese 쪽은 ImageNet 사전학습 VGG16 특징을 Siamese 구조에서 Cosine·Euclidean로 비교해 전통적 핸드크래프트 매칭과의 차이를 정량화했다.

- **Empirical Impact**: 120명의 참가자가 만든 스케치 189장과 추가로 실력 스케일 캘리브레이션을 위한 데이터(83장)를 사용해, SIFT 기반이 사람 평가와 가장 가깝게 정렬되는 경향을 보였다. 결과적으로 SIFT는 전체적으로 80%~98% 수준의 정확도 범위를 달성했으며, keypoint 및 양호한 매칭 비율에서도 상대적으로 유리한 패턴이 관찰됐다. 저자들은 특히 SIFT가 회전·스케일·밝기 변화에 덜 민감하고 색 의존성이 낮아 실력 평가에 효과적이라고 정리하며, 자동 미술 평가 도구로서의 가능성을 제시한다. 



### Distill Once, Adapt Life-Long: Exploring Dataset Distillation for Continual Test-Time Adaptation (https://arxiv.org/abs/2606.20196)
Comments:
          ECCV 2026

- **Prior Approaches**: CTTA는 비라벨 타깃 스트림에서 온라인으로 적응하면서도 긴 시간 동안 성능을 유지하려는 시도다. 하지만 source-free 설정에서는 원천 데이터가 남아 있지 않아, 분포가 계속 변할수록 self-training 오류가 누적되고 catastrophic forgetting이 심해져 장기 안정성이 떨어진다.
기존에는 source 정보를 원본 형태로 저장하지 않고 통계·prototype·lightweight proxy 등으로 압축해 기준점을 제공하려 했지만, 이런 prior가 충분히 “의미적으로” 고정되지 못하면 drift를 완전히 막기 어렵다.

- **Core Contribution**: DO-ALL(Distill Once, Adapt Life-Long)은 Dataset Distillation(DD)으로 소스 분포를 한 번에 작은 synthetic distilled anchors로 요약한 뒤, 배포 후에는 원본 소스를 보관하지 않고도 이를 안정 기준선으로 재사용한다. 적응 단계에서는 각 타깃 샘플을 의미적으로 가장 가까운 앵커에 매칭해, 기존 CTTA 알고리즘의 목표/구조를 바꾸지 않고도 replay·representation alignment·manifold-smoothing 정규화를 플러그인한다.
또한 harm-adaptive blending으로 장기 적응 중 “유해한” 파라미터 변화만 선별적으로 소스 초기값 쪽으로 되돌려 잔류 drift를 완화한다.

- **Technical Challenges**: 핵심 난제는 (1) 소스를 직접 저장하지 못하는 source-free 제약 하에서 (2) 분포가 계속 이동하는 동안 학습 신호의 신뢰도를 유지하며 (3) 적응 과정에서 self-training 오류가 누적되지 않도록 안정 장치를 설계하는 것이다.
DO-ALL은 DD로 만든 앵커에 soft pseudo-label과 소스 latent feature를 함께 담아 의미 정합성을 높이고, 타깃-앵커 매칭 기반 replay와 layer-wise MMD 정렬로 표현 공간의 구조를 고정한다; 여기에 MixUp 형태의 국소 smoothing과 harm-adaptive blending을 더해 장기 누적 드리프트를 차단한다.

- **Empirical Impact**: CIFAR100-C와 ImageNet-C, 그리고 CCC 벤치마크에서 DO-ALL은 EATA·RMT·ROID 같은 다양한 CTTA 베이스라인에 일관되게 성능을 개선했으며, 특히 최악의 corruption severity 및 CCC의 Hard 시나리오에서 장기 안정성 향상 폭이 두드러졌다. 예를 들어 ImageNet-to-ImageNet-C(level 5)에서는 EATA+DO-ALL이 평균 오류를 58.0%→56.6%, RMT+DO-ALL이 59.8%→57.4%로 낮췄다.
결과적으로 DD 기반 앵커 품질이 개선될수록 CTTA 성능이 더 크게 좋아지는 상관이 관찰되어, “한 번만 증류하고 오래 적응하는” 안정적 설계 가능성을 실증했다.



### HilDA: Hierarchical Distillation with Diffusion for Advancing Self-Supervised LiDAR Pre-trainin (https://arxiv.org/abs/2606.20189)
Comments:
          Accepted to ECCV 2026. Maciej and Jesper contributed equally

- **Prior Approaches**: 기존 camera-LiDAR knowledge distillation은 Vision Foundation Model(VFM)을 black-box teacher로 두고 주로 프레임 단위 feature similarity만 맞추는 경향이 강했습니다. 이 때문에 VFM의 층별(semantic evolution) 구조와 CLS 토큰이 담는 전역(global context) 의미를 충분히 활용하지 못하고, LiDAR의 시공간(spatiotemporal) 일관성 학습도 약했습니다.

- **Core Contribution**: 본 논문은 LiDAR 백본을 위한 self-supervised pretraining 프레임워크 HilDA를 제안합니다. HilDA는 계층적(hierarchical) distillation로 “semantic what(무엇)”을 VFM에서 가져오고, temporal occupancy diffusion으로 “geometric where(어디)”와 미래의 시공간 일관성을 함께 학습하도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 모달리티에서 VFM의 층별 의미 계층을 점-픽셀 대응 위에서 안정적으로 전이하고, (2) 의미 정렬만으로는 부족한 동적 장면의 미래 점유를 생성적으로 학습하는 것입니다. HilDA는 기하학적 calibration 기반의 dense-to-sparse 멀티레이어 distillation과 CLS 토큰 정렬을 동시에 수행하고, 미래 BEV occupancy를 conditional diffusion(denoising 기반)으로 예측하는 auxiliary objective를 추가해 저라벨/무라벨 상황에서도 예측형 기하·운동 표현을 주입합니다.

- **Empirical Impact**: HilDA는 cross-modal distillation 벤치마크에서 state-of-the-art를 달성했으며, 특히 레이블이 적은 데이터 스케줄(1%–10%)에서 기존 distillation 대비 큰 폭의 향상을 보였습니다. 또한 3D object detection, scene flow, semantic occupancy prediction 전반에서 이전 기법을 능가하고, nuScenes-C에서 mCE 낮고 mRR 높게 나타나는 등 강인성도 개선되는 것으로 보고됩니다.



### Evaluating and Enhancing Negation Comprehension in Remote Sensing MLLMs (https://arxiv.org/abs/2606.20177)
Comments:
          ECCV 2026 Accepted

- **Prior Approaches**: 기존 연구는 negation을 평가하기 위해 NegVQA, GaslightingBench 같은 일반 도메인 비전-언어 벤치마크를 제시했지만, 원격탐사(RS)에서 요구되는 세부 속성·상태 수준의 부정과 소형 표적 문제는 충분히 다루지 못했습니다. 또한 CLIP/embedding 중심의 negation 학습·테스트는 있어도, MLLM에 그대로 확장하기 어렵고 대규모 라벨 수집 없이 개선하는 방법도 상대적으로 비어 있었습니다.

- **Core Contribution**: 이 논문은 RS MLLM의 negation 이해를 지역-수준부터 장면-수준까지 포괄적으로 점검하는 최초 벤치마크 RS-Neg(총 22,464샘플)를 제안합니다. LLM이 자연스럽고 다양한 negation 쿼리를 만들고, Dynamic visual focus(DyFo)로 부정 개념의 시각적 부재를 검증한 뒤 과제별 VQA/MCQ/visual grounding/scene classification 데이터로 재구성합니다. 더불어 테스트 시점 학습 NeFo를 제안해, 라벨 추가 없이 약 5% 미라벨 테스트 샘플로 negation 이해를 크게 끌어올립니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) RS 이미지에서 ‘없음/부정’ 개념을 실제로 검증해야 하는데 소형 표적 때문에 기존 MLLM 검증이 잘 실패한다는 점, (2) 테스트 시점 적응 시 잘못된 예측이 자기강화되어 성능이 더 악화될 수 있다는 점입니다. 논문은 DyFo 기반 탐색으로 negation에 해당하는 시각적 부재를 필터링하고, NeFo에서는 negation의 논리적 역할(진리값 반전)을 목적함수에 직접 반영해 negated 쿼리와 그 부정 마스킹 변형의 출력 불일치를 키우는 방식으로 학습 신호를 설계합니다. 동시에 knowledge retaining 정규화와 엔트로피 최소화를 함께 써 catastrophic forgetting을 억제합니다.

- **Empirical Impact**: 실험 결과, 대부분의 RS 특화 및 일반 MLLM은 부정 조건에서 환각과 함께 성능이 크게 하락하며, 장면-수준에서 평균 VQA 8.6%, classification 21.3%의 저하가 관찰됩니다. NeFo는 LoRA의 가벼운 파라미터 업데이트와 unlabeled 테스트 데이터만으로도 Qwen2.5-VL 등 여러 베이스 모델에서 VQA/MCQ 성능을 유의미하게 개선하고, 이후 unseen 과제(분류·grounding·FloodNet VQA)로도 전이 일반화가 확인됩니다. 특히 기존 TTL 방법들이 grounding에서 급격한 붕괴를 보일 때 NeFo는 구성요소까지 포함한 ablation에서 안정적으로 이득을 유지해, RS 재난·안전 응용에서 ‘무엇이 아닌가’를 정확히 판단해야 하는 요구를 실증적으로 뒷받침합니다.



### ARTEMIS: Agent-guided Reliability-aware Temporal Mask Evolution for Imperfectly Supervised Video Polyp Segmentation (https://arxiv.org/abs/2606.20161)
- **Prior Approaches**: 기존 VPS 연구는 정밀 라벨(full supervision)에 초점이 많아, 저대비·모호한 경계·운동 블러·사양(spotlight/specular) 같은 임상 변이를 그대로 만났을 때 마스크가 끊기거나 시간 드리프트가 커지기 쉽습니다. 불완전 감독에선 weakly supervised(점/스크리블) 또는 semi-supervised(소수 dense 라벨)처럼 한 축만 다루는 경우가 많아, 예산이 섞인 실제 데이터에서 서로 다른 감독 레짐을 하나로 통합하기 어렵고 temporal consistency도 충분히 활용되지 못했습니다. 또한 SAM/SAM2 기반 pseudo-labeling은 실패 프레임에서 기하가 깨지거나 경계 누출이 생기는데, 신뢰도 평가와 시간에 따른 진화가 약해 노이즈를 그대로 학습에 태우는 한계가 있었습니다.

- **Core Contribution**: ARTEMIS는 weakly supervised와 semi-supervised를 동시에 만족하는 “imperfectly supervised VPS” 통합 프레임워크로, sparse·부분 dense 정보를 temporally consistent한 dense pseudo mask로 완성합니다. 핵심은 agent-guided reliability-aware temporal mask evolution으로, 신뢰도 높은 프레임 앵커를 선택하고 SAM2로 양방향 전파해 신뢰 없는 프레임까지 점진적으로 개선한다는 점입니다. 이어 RPTM과 reliability-aware robust loss를 통해 남은 경계 오류·누출·노이즈를 가중치로 억제하며 세그멘터를 학습합니다.

- **Technical Challenges**: 기하적으로 그럴듯하지만 실제 폴립 경계에 부정확한 pseudo-label이 low-contrast/blur/occlusion 상황에서 쉽게 생성되어, 단순 pseudo-label링은 구조 왜곡과 boundary leakage를 유발합니다. ARTEMIS는 이를 줄이기 위해 debate-and-judge vision-language agent(Qwen2.5-VL-7B)로 temporal anchor의 신뢰도를 점수화하고, 양방향 SAM2 전파로 앵커 정보를 앞뒤 프레임에 재주입합니다. 또한 “어떤 프레임·픽셀이 신뢰 가능한가”를 전면에서 처리하기 위해 forward-backward agreement, 영역 유효성, foreground confidence 등을 기반으로 reference frame을 안정적으로 고르고, RPTM으로 시간 내 identity를 수송·진화시키면서 reliability-aware robust loss로 잡음 학습을 down-weight합니다.

- **Empirical Impact**: SUN-SEG와 CVC-ClinicDB-612에서 scribble, point, limited-label 같은 저감독 설정 전반에 대해 SOTA 성능을 달성했다고 보고합니다. 특히 제한된 dense 라벨이 섞인 상황에서도 통합된 pseudo-label 완성 전략이 시간적 일관성을 개선해, 기존 레짐별 분절 접근 대비 범용성과 안정성이 강화된 것으로 해석됩니다. 임상 폴립 분할처럼 비용이 큰 라벨 문제를 더 현실적인 예산 구성에서 해결하는 방향성을 제시하며, 추후 foundation model 기반 medical video segmentation에 대한 신뢰도 진화 학습 설계를 확산시킬 잠재력이 큽니다.



### NAMESAKES: Probing Identity Memorization in Text-to-Image Models (https://arxiv.org/abs/2606.20155)
- **Prior Approaches**: 기존 T2I 프라이버시/멤버리제이션(Identity memorization) 평가는 후보군의 기준 사진(ground-truth), 학습 데이터 접근, 또는 모델 내부(white-box) 정보가 필요해 적용 범위가 좁았습니다. 판별 모델/공격 기법은 주로 생성 이미지 단위 또는 특정 샘플 재현 여부를 보려 하지만, “이름-정체성” 수준의 전반적 유출을 일관되게 가늠하기 어렵다는 한계가 있었습니다. 또한 분별이 되더라도 모델 불특정(architecture-agnostic) 환경에서는 재현성 확보가 쉽지 않았습니다.

- **Core Contribution**: 이 논문은 완전한 블랙박스(fully black-box), 레퍼런스 사진 없이도 개인 이름에 대해 “정체성 재현(Identity memorization)”과 “그럴듯한 생성(Identity fabrication)”을 구분하는 behavioral probe를 제안합니다. 핵심은 얼굴 유사도(멤버리제이션의 기준)를 직접 쓰지 않고도 생성들의 일관성/분산 패턴으로 이를 예측한다는 점입니다. 아울러 벤치마크로 NAMESAKES(1,269개 이름-얼굴 쌍, 인지도 스펙트럼 포함)와 perturbed name 세트를 만들어 테스트/비교를 가능하게 했습니다.

- **Technical Challenges**: 어려움은 (1) GT 얼굴 없이도 멤버리제이션을 추정해야 하고, (2) 모델별 생성 다양성 차이까지 흡수해야 하며, (3) 단일 이름만으로 “실제 인물 vs 비기억 기본 얼굴”을 구분할 신호를 찾아야 한다는 데 있습니다. 저자들은 생성 4장을 seed 변화로 뽑아 ArcFace 임베딩을 만든 뒤, (a) 같은 이름의 생성들이 얼마나 일관된 얼굴을 띠는지 보는 dispersion 기반 점수 δ, (b) 다른 이름들의 생성 중심(centroid) 대비 얼마나 벗어나는지 보는 centroid distance 기반 점수 sc을 결합해 참조 유사도와의 상관을 학습/추정합니다.

- **Empirical Impact**: SDXL-base, SDXL-Turbo, Flux1-Dev, Flux1-Schnell 등 SOTA T2I 모델 4종에서 연속 예측(R^2)과 이진 분리(AUC/정확도) 모두 유의미하게 성능을 보였습니다. 특히 SDXL-base는 reference similarity 연속 예측에서 R^2≈0.58, real vs perturbed 분리에서 AUC≈0.86(Acc≈0.79)로 가장 강했고, 고명성 구간에서는 AUC≈0.95 및 90%+ 정확도까지 상승했습니다. 즉, 레퍼런스/내부 접근 없이도 이름 기반 정체성 유출 가능성을 실무적으로 스크리닝할 수 있으며, 모델 계열 간(예: SDXL vs Flux) 멤버리제이션 양상 차이도 관찰된다는 점에서 프라이버시 감사와 unlearning 평가에 직접적인 의미가 있습니다.



### HEad and neCK TumOR (HECKTOR) 2025: Benchmark of Segmentation, Diagnosis, and Prognosis in Multimodal PET/C (https://arxiv.org/abs/2606.20143)
Comments:
          17 pages, 4 figures, 4 tables. Overview paper for the HECKTOR 2025 challenge, held as a satellite event at MICCAI 2025. Challenge website: this https URL

- **Prior Approaches**: 기존 HNC 자동 분석은 주로 CT/PET에서 종양 경계를 rule-based 혹은 U-Net 계열, nnU-Net, Swin UNETR 같은 딥러닝으로 분할하는 흐름이었습니다. 다만 PET의 낮은 신호대잡음비, 영상 획득·스캐너 간 차이, 병변 크기/경계의 미묘함 때문에 일반화와 정확한 병변 검출·경계 정합이 어렵다는 한계가 컸습니다. 예후(RFS)나 HPV 상태는 radiomics 기반 Cox/기계학습, 혹은 DeepSurv·transformer survival 같은 방법이 연구됐지만, 장기 추적 라벨과 다기관 검증의 벽으로 임상 전환성이 떨어졌습니다.

- **Core Contribution**: HECKTOR 2025는 oropharyngeal HNC를 대상으로 PET/CT와 EHR을 결합한 다목표 자동 분석 벤치마크를 제시하며, 단순 분할을 넘어 RFS 예측과 비침습 HPV 분류까지 통합했습니다. 10개 기관 1,123명 규모의 멀티센터 데이터로 과제의 현실성(다양한 스캐너·프로토콜)을 확보하고, Docker 기반 제출로 재현 가능한 평가 환경도 제공했습니다. 특히 분할 정확도뿐 아니라 병변 검출 성능까지 함께 보도록 설계해 실제 임상 워크플로에 가까운 문제정의를 했습니다.

- **Technical Challenges**: 기여를 실현하는 핵심 난점은 (1) PET/CT 다중모달 융합에서 정합·스케일 차이를 안정적으로 다루는 것, (2) GTVp/GTVn의 크기 편차와 불규칙 경계, 작은 림프절 병변을 동시에 정확히 포착하는 것, (3) RFS처럼 censoring이 존재하는 생존분석을 견고하게 학습·평가하는 것입니다. 논문은 과제 차원에서 PET/CT SUV 표준화, RTDose가 있는 일부 케이스의 정합, 그리고 GTVn은 cohort-level 집계 DSC와 lesion-level F1로 “경계 품질+검출 누락/환각”을 동시에 반영하도록 평가를 구성했습니다. 또한 HPV는 클래스 불균형 영향을 줄이기 위해 balanced accuracy를 사용해 비침습 분류의 실용성을 높였습니다.

- **Empirical Impact**: 테스트셋 held-out 기준으로 상위 알고리즘들은 분할에서 GTVp mean Dice 0.75, RFS에서 Harrell’s C-index 0.66 수준, HPV 분류에서 balanced accuracy 0.56 수준의 성능을 달성했습니다. 이는 분할·예후·바이오마커(HPV)까지 한 데이터 파이프라인에서 비교 가능하게 측정할 수 있음을 보여주며, 다기관 일반화와 병변 검출을 함께 요구하는 설계의 유효성을 시사합니다. HECKTOR 2025의 방법론 분석은 자동 oncology 워크플로와 의사결정 보조 시스템으로의 임상 번역 가능성을 높이는 실증적 기준점 역할을 할 것으로 기대됩니다.



### SA-VIS: Sparse frame Annotations for training Video Instance Segmentation (https://arxiv.org/abs/2606.20140)
- **Prior Approaches**: 기존 온라인 VIS는 프레임별 인스턴스 예측 후 시간축에서 feature similarity 또는 query propagation으로 매칭해 성능을 냈지만, 이를 위해 학습 때 조밀한(밀집) 프레임 라벨과 긴 시퀀스를 요구하는 경우가 많습니다. 오프라인 방식은 더 큰 입력 청크를 한 번에 처리해 성능을 내기도 하지만, 긴 비디오에서 tracklet 연결 같은 추가 비용/복잡성이 따라붙습니다.

- **Core Contribution**: 이 논문은 조밀 라벨 없이도 인스턴스의 “진화(temporal evolution)”를 학습할 수 있다고 보고, Sparse frame Annotation VIS(SA-VIS)를 제안합니다. 핵심은 Past-frames Feature Propagation(PFP)으로 과거 프레임의 저차원 특징을 Feature Queue에 축적해 현재 프레임의 인스턴스 매칭을 돕는 구조이며, 여기에 frame-specific instance queries(FSI queries)를 더해 프레임마다 고유한 탐지/분할 질의를 생성합니다.

- **Technical Challenges**: 문제는 조밀 라벨이 없어 과거 프레임 정보를 학습 신호로 직접 쓰기 어렵다는 점인데, 저자들은 학습 시 key-reference 프레임 쌍에만 손실을 계산하고 과거 프레임 특징은 backbone만으로 “frozen” 방식으로 주입하도록 설계했습니다. PFP는 깊은 feature를 pooling/FFN으로 압축해 2-layer transformer encoder에서 과거 특징과 현재 특징의 cross-attention을 수행하며, FSI queries는 경량 center heatmap 검출로 얻은 인스턴스 위치에서 feature를 뽑아 query를 동적으로 생성합니다.

- **Empirical Impact**: SA-VIS는 YouTube-VIS 2019/2021/2022와 OVIS에서 기존 baseline 및 유사한 pairwise-past 학습 계열 대비 우수한 성능을 보였습니다. 특히 조밀 라벨 대비 1/5 프레임만 라벨링해도 성능 하락이 0.4%에 그치며, 제한된 라벨 시나리오에서는 state-of-the-art 대비 1%+ AP 개선을 보고합니다.



### TriFlow: Generating Artist-Like 3D Mesh Topology via Nearest-Vertex Vector Fields (https://arxiv.org/abs/2606.20131)
- **Prior Approaches**: 기존 3D 생성/재구성은 SDF/occupancy 같은 암시적 표현을 만든 뒤 iso-surfacing으로 메쉬를 뽑아내는 방식이 많아, 결과 메쉬가 과도하게 조밀해지고(과테셀레이션) downstream에 바로 쓰기 어려운 경우가 잦았습니다. 또 discrete triangle 시퀀스를 autoregressive로 생성하는 학습 기반 접근은 느린 토큰-단위 추론과 오류 누적으로 인해 일반화·완성도에서 한계가 있습니다. 한편 QEM·QuadriFlow 같은 단순화/리토폴로지는 기하 오차는 줄이지만, 아티스트다운 topology(연결 구조)를 직접 보장하기는 어렵습니다.

- **Core Contribution**: TriFlow는 메쉬의 topology를 이산 연결(정점-면 조합) 대신 표면 위에서 정의되는 nearest-vertex vector field(NVF)로 표현해 생성 문제를 연속적인 벡터장 모델링으로 바꿉니다. 입력 조건으로는 signed distance field(SDF)를 사용하고, flow matching으로 NVF를 생성한 뒤 watershed 기반 군집화와 topology-aware constrained QEM 추출로 최종 메쉬를 복원합니다. 결과적으로 입력 기하와 잘 맞으면서도 구조적으로 정돈된(artist-like) 연결을 갖는 compact 메쉬를 목표로 합니다.

- **Technical Challenges**: 핵심 기술 난관은 (1) topology를 잘 요약하는 NVF 표현을 안정적으로 학습·생성하고, (2) 생성된 NVF에서 일관된 connected component를 정확히 복원하며, (3) 그 연결을 유지한 채 geometry 정합까지 맞추는 것입니다. TriFlow는 VAE로 SDF·NVF를 희소 라텐트로 압축한 뒤, 조건부 latent flow-matching으로 NVF를 합성하고, 예측 잡음으로 생기는 군집 모호성은 bilateral filter + watershed로 완화합니다. 또한 constrained QEM에서 region root가 다른 edge collapse를 거부하고, 타겟 위치에 가깝게 하도록 quadric에 positional penalty(QtQ)를 추가해 topology와 형상 정합을 동시에 유도합니다.

- **Empirical Impact**: 실험에서는 TriFlow가 기존 learning 기반 방법 대비 일반화가 더 강하고 topology 품질이 크게 개선되었으며, Chamfer Distance는 90% 낮추고 추론 속도는 8배 빨라졌다고 보고합니다. 특히 TRELLIS로부터 생성된 도전적(노이즈/아티팩트) 형상에서도 실용성이 드러나, 단순히 깨끗한 데이터에만 맞춘 결과가 아님을 시사합니다. 즉, 고정밀 기하 생성과 프로덕션용 메쉬 topology 생성 사이의 간극을 줄이는 접근으로 평가됩니다.



### SAM3 Self-Distillation for Fine-Grained GOOSE 2D Semantic Segmentation (https://arxiv.org/abs/2606.20130)
Comments:
          4th place in ICRA 2026 GOOSE 2D Semantic Segmentation Challenge

- **Prior Approaches**: GOOSE 2D 같은 오프로드 환경 세그멘테이션은 데이터가 롱테일 분포를 갖고(희귀 클래스 극히 적음), 시각적으로 비슷한 within-category 유사 클래스가 많아 학습이 까다롭다. 기존 대회 참가들은 주로 세그멘터 백본/헤드 설계와 표준 fine-tuning, 그리고 일반적인 색/기하 증강·TTA로 성능을 끌어올리려 했지만 플랫폼별 해상도·노출 차이 때문에 한 플랫폼에 과적합되기 쉽다. 또한 SAM 계열처럼 고정 입력 크기에 강하게 제약이 걸린 경우, 기존 multi-scale test-time augmentation을 그대로 쓰기 어렵다는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 Segment Anything Model 3 (SAM3)의 이미지 인코더를 활용하되, 상위 레이어만 부분 fine-tuning하고 가벼운 FPN 스타일 디코더를 붙여 픽셀 단위 클래스 지도를 end-to-end로 학습시키는 구성을 제시한다. 여기에 (1) SAM3를 자체 지도교사로 쓰는 self-distillation을 더해, SAM3가 오히려 더 좋은 클래스에서만 ground-truth box 프롬프트 마스크를 추가 학습 신호로 반영한다. 또한 (2) 고정 입력 크기 모델에서도 multi-scale 추론의 효과를 살리기 위해 모델 입력이 아니라 이미지를 rescale한 뒤 타일 단위로 처리하고 점수를 Hann 가중치로 퓨전하는 test-time augmentation 스킴을 제안한다. 

- **Technical Challenges**: 핵심 기술적 난제는 (a) SAM3의 prompting이 텍스트만으로는 오프로드 장면에서 오동작할 수 있고, (b) SAM3가 항상 더 정확하지는 않아서 무차별 증류는 성능을 망칠 수 있다는 점이다. 이를 위해 ground-truth bounding box로 제한하는 oracle-box prompting과 함께, 클래스별로 SAM3 마스크가 우리 모델보다 “분명히” 좋은 경우에만 22개 클래스로 distillation 신호를 제한했다. 또 SAM3 인코더가 단일 fixed input size만 받는 제약 때문에 일반적인 rescaled-model TTA가 막혀 있었고, 이를 이미지 rescaling+타일 윈도우 퓨전으로 우회해 추가 학습 없이 multi-scale 효과를 회복했다. 마지막으로 오프로드는 계절·조명·노출 변동이 커서 색을 단서로 쓰면 깨지는데, 대회 2025 우승 엔트리의 aggressive photometric distortion을 pipeline에 이식해 shape/texture 중심 학습으로 유도했다.

- **Empirical Impact**: 이 방법으로 ICRA 2026 GOOSE 2D Fine-Grained Semantic Segmentation Challenge에서 composite mIoU 69.73% (공식 1,815-image 테스트셋)로 4위를 기록했다. 기여도 분석에서 가장 큰 향상은 aggressive photometric distortion 기반 color augmentation으로, self-distillation 대비 +0.86 수준의 큰 이득을 보였다. flip augmentation은 소폭이지만 일관된 개선을 주었고, 고정 입력 크기에서 multi-scale을 복원하는 이미지 레벨 TTA가 그 위에 추가로 +0.34를 더해 총 성능을 끌어올렸다. 특히 kick-scooter 같은 희귀 클래스는 큰 폭으로 개선되었지만 moss는 여전히 혼동이 커서, 향후 희귀 클래스 추가 사전학습이나 더 정교한 앙상블 전략이 필요함을 시사한다.



### Pixel-Level Residual Diffusion Transformer: Scalable 3D CT Volume Generation (https://arxiv.org/abs/2606.20112)
Comments:
          Accepted at ICLR 2026. Code available at this https URL

- **Prior Approaches**: 기존 GAN 기반 3D 생성은 국소 디테일을 잘 만들 수 있지만 mode collapse와 학습 불안정, 고해상도 3D에서의 큰 메모리 부담이 커서 확장에 제약이 있습니다. Diffusion 기반 접근은 학습 안정성과 품질이 좋아졌지만, 대부분 U-Net 계열로 구성되어 long-range dependency 같은 전역 구조 일관성 학습에 한계가 있고, 3D에서 해상도를 키우면 토큰 수와 attention 비용이 급증해 학습이 비싸고 불안정해집니다. 또한 latent diffusion은 VAE/VQ-VAE 병목 때문에 3D 의료 데이터에서 세부 해부학 정보가 약해지거나 엔코더 학습이 어렵다는 문제가 이어집니다.

- **Core Contribution**: PRDiT(Pixel-Level Residual Diffusion Transformer)는 CT 볼륨을 autoencoder bottleneck 없이 voxel 수준에서 직접 생성하도록 설계된 확장형 diffusion transformer 프레임워크입니다. 핵심은 2단계 residual 학습으로, (1) 겹치는 3D patch를 쓰는 MLP 기반 local denoiser로 저주파 구조를 먼저 추정하고, (2) 전역 residual을 정제하는 Global residual diffusion transformer가 고주파 잔차를 보정해 미세 구조를 보존한다는 전략입니다. 더 나아가 고해상도에서는 pretrained 저해상도 백본을 재사용하고 고주파 refinement만 추가 학습해, 해상도 스케일링 비용을 크게 줄입니다.

- **Technical Challenges**: Transformer diffusion을 고해상도 3D에 그대로 적용하면 해상도 증가에 따라 토큰 수가 폭증하고 self-attention 비용이 크게 올라(예: 해상도 2배 시 attention이 대략 64배 수준) 학습이 메모리/최적화 측면에서 취약해집니다. PRDiT는 이 문제를 2단계(로컬→전역 잔차) 구조와 메모리 효율 attention, 그리고 고해상도에서는 저해상도 모델을 freeze한 채 residual refinement 모듈만 학습하는 방식으로 완화했습니다. 또한 생성 과정에 cold predictor–hot corrector 형태의 hot diffusion sampling을 도입해 결정적 유도와 제어된 잡음 주입을 균형화함으로써 구조 보존과 샘플 다양성/안정성을 동시에 노렸습니다.

- **Empirical Impact**: 실험에서 PRDiT는 LIDC-IDRI와 Rad-ChestCT 두 데이터셋에서 HA-GAN, 3D LDM, WDM-3D 같은 최신 모델을 3D FID, MMD, Wasserstein distance 전반에서 일관되게 능가합니다. 특히 PRDiT의 얕은 변형(예: 4 layers)도 경쟁 모델보다 낮은 지표를 보여주며, depth를 늘릴수록 bone edge와 경계가 더 선명해지는 정성적 개선이 지표 감소와 함께 나타납니다. 고해상도에서도 토큰 폭증 없이 residual refinement만 추가하는 구성이 학습 비용과 OOM 리스크를 줄이면서 품질을 유지/향상시키는 점에서 3D CT 생성 분야에 실용적인 확장 방향을 제시합니다.



### FrozenDrive: Zero-Shot Text-Guided Driving Scene Generation and Data Augmentation with Parameter-Free Frozen Diffusion Mod (https://arxiv.org/abs/2606.20110)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 자율주행용 합성 데이터 생성은 diffusion 기반으로 확장됐지만, 멀티뷰·시간 일관성을 위해 backbone을 fine-tuning하거나(또는) 추가 레이어/모듈을 학습하는 경우가 많습니다. 그 결과 텍스트-장면 정렬(text alignment)이 약해지고, 사전학습이 담고 있던 강한 prior가 지워져 zero-shot 제어력이 떨어질 수 있습니다. 또한 합성 결과가 학습 분포에 가까워져 비·야간·희귀 구성 같은 long tail에서 품질이 급락한다는 한계도 큽니다.

- **Core Contribution**: 이 논문은 backbone diffusion 모델을 Frozen(매개변수 추가·가중치 업데이트 없음)한 채로 멀티뷰 및 temporal consistency를 강제하는 생성 프레임워크 FrozenDrive를 제안합니다. 핵심은 지식 보존(knowledge-preserving) 관점에서, 기존 self-attention의 ‘가중치 자체’는 그대로 두고 입력 컨텍스트를 재구성하는 spatio-temporal attention을 설계한 점입니다. 더해 희귀 객체의 per-object fidelity를 높이기 위해 object-focused 제약(객체 출현 비율 기반 loss)도 함께 둡니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 “백본을 얼린 상태”에서 일관성을 만들되 텍스트 정렬과 사전학습 prior를 훼손하지 않는 것입니다. FrozenDrive는 ControlNet 기반 구조화된 driving-stack 신호(HD map, 3D bounding box, depth/occupancy, 뷰 식별, 전 프레임 대비 relative pose)로 조건을 주고, MISA(멀티뷰 inflated self-attention)와 TRSA(temporal reference self-attention)를 통해 단일 패스에서 cross-view 정합과 프레임 간 안정성을 주입합니다. 여기에 long-tail 문제를 줄이기 위해 3D box 투영 기반 object-presence ratio loss로 희귀 클래스에 학습 비중을 더 줍니다.

- **Empirical Impact**: nuScenes에서 생성 품질(FVD)과 downstream 성능(3D detection/BEV segmentation, UniAD 기반 perception·planning)을 함께 평가한 결과, FrozenDrive가 기존 generative baseline 대비 전반적으로 우수한 성능을 보였습니다. 특히 데이터 증강을 night/rain 시나리오에 text prompt만으로 적용했을 때 SparseDrive로 학습한 AD 모델의 탐지·매핑 점수가 개선되고 planning error가 유의미하게 감소해 강건성이 강화됨을 보여줍니다. 또한 학습 분포 밖의 adverse/희귀 조합(예: 눈 장면)을 fine-tuning 없이도 생성·전이가 가능하다는 점에서, 합성 데이터가 long tail 대응에 직접 기여할 수 있음을 시사합니다.



### EFIQA: Explainable Fundus Image Quality Assessment via Anatomical Priors (https://arxiv.org/abs/2606.20108)
Comments:
          Accepted in MIDL 2026. Code: this https URL

- **Prior Approaches**: 기존 fundus IQA는 보통 데이터셋별 품질 라벨로 supervised classifier를 학습해왔다. 이 방식은 (1) ‘좋은 품질’의 기준이 라벨 기준에 종속돼 다른 기준의 데이터에 취약하고, (2) 이미지 품질을 한 점수로만 내며 저하 위치를 설명하기 어렵다는 한계가 있었다.

- **Core Contribution**: EFIQA는 주관적 품질 라벨 없이도 해부학적 구조의 ‘기대되는 가시성’에서 벗어남을 학습해, 설계 단계에서 공간 quality map을 생성한다. 특히 fundus에서는 혈관(vasculature) 가시성을 기준으로 삼아, 손상/저하가 어디서 발생하는지 지도 형태로 제공하도록 만든 것이 핵심이다.

- **Technical Challenges**: 해부학적 priors를 학습하되, 빈 영역을 “그럴듯하게 복원”하는 shortcut learning(정체성 붕괴)을 피해야 한다는 점이 기술적 난제다. 이를 위해 masked anatomical inpainting 기반 VUAD로 혈관 토폴로지를 unsupervised하게 학습하고, VUAD의 anomaly map(의사라벨)을 frozen foundation model의 feature에 distill해 shallow adapter가 단일 forward로 정밀한 품질 지도를 출력하게 했다.

- **Empirical Impact**: 외부 데이터셋들(MSHF, mBRSET, DRIMDB 등)에서 EFIQA는 supervised 방식 대비 평균적으로 MCC가 크게 개선되며, 동시에 저하 영역을 더 안정적으로 국소화했다. 또한 품질 map을 제공하는 동시에 explainability 측면에서도 post-hoc GradCAM 등 기존 접근보다 저하 경계와 범위를 더 잘 따라가며, 최소한의 적응으로 다양한 품질 기준에 대응할 잠재력을 보였다.



### Geometry-Preserving in 3D Gaussian Splatting for LiDAR-Camera Extrinsic Calibration (https://arxiv.org/abs/2606.20103)
Comments:
          Accepted to ECCV 2026. 15 pages (excluding references), 5 figures

- **Prior Approaches**: LiDAR-카메라 extrinsic calibration은 고정된 마커(체커보드 등)를 쓰는 target-based 방식이 정밀하지만, 수동 세팅 부담과 현장 드리프트 대응의 한계가 큽니다. 이를 피하려는 targetless 방식은 양쪽에서 추출되는 엣지/구조선 같은 공유 특징에 의존하나, 텍스처가 빈약하거나 반복 구조가 많은 주행 장면에서는 판별력 있는 크로스모달 특징이 부족해 정확도가 흔들립니다. 최근에는 장면을 differentiable 재구성(NeRF/3DGS)으로 표현해 dense pixel supervision으로 extrinsic을 최적화하는 흐름이 늘었지만, 3DGS는 본래 novel view synthesis 목적이라 photometric 성능을 위해 proxy 기하가 실제 LiDAR 구조에서 “흐려지는” 문제가 생깁니다.

- **Core Contribution**: 이 논문은 3DGS 기반 calibration에서 photometric supervision이 proxy의 metric geometry를 훼손하는 현상을 Geometric Decay로 규정하고, 그 결과 calibration 정확도·안정성이 떨어진다는 점을 실증적으로 분석합니다. 이를 해결하기 위해 Geometry-Preserving Calibration(GeoP-Calib)을 제안하며, Dense Depth Anchoring(DDA)로 다중 뷰 LiDAR를 누적해 기하 기준선을 단단히 만들고, Gradient Decoupling(GD)으로 photometric 그라디언트가 Gaussian의 위치/공분산을 건드리지 못하게 막습니다. DDA와 GD의 역할을 분리해, appearance·extrinsics는 조정하되 proxy 공간 파라미터의 드리프트는 억제하는 전략입니다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 단일 스캔 LiDAR의 희소성 때문에 proxy가 제약이 없는 영역에서 쉽게 이동하고, (2) 다중 뷰 누적 시 가림(occlusion)으로 인해 먼 뷰 포인트가 잘못 투영되는 구조적 모호성이 동시에 발생한다는 점입니다. 논문은 먼저 누적 point cloud에서 현재 뷰로 투영한 dense depth prior를 만들되, occlusion 문제는 3DGS의 렌더링 depth를 연속적인 visibility 기준으로 써서 Volumetric Soft Mask(VSM)로 완만하게 다운웨이트 처리합니다. 다음으로 GD를 stop-gradient 형태로 적용해 photometric loss의 gradient가 Gaussian spatial parameters에 전파되지 않도록 하여, texture-driven 잔차가 기하를 변형시키는 경로를 차단합니다.

- **Empirical Impact**: KITTI odometry와 KITTI-360 같은 공개 주행 데이터셋에서 GeoP-Calib은 기존 targetless 방식 대비 calibration 정확도를 일관되게 개선했으며, 특히 translation 추정에서 개선 폭이 두드러졌다고 보고합니다. 또한 photometric 최적화가 포함될 때 Geometric Decay가 발생하는 경향(렌더 depth 손실의 평균/변동성 증가)을 관찰함으로써, 제안한 방지 메커니즘의 필요성을 실험적으로 뒷받침합니다. 결과적으로 3DGS의 differentiable 최적화 이점을 유지하면서도 LiDAR 관점의 metric faithfulness를 보존하는 calibration 프레임워크로 의미가 큽니다.



### WeGenBench: A Multidimensional Diagnostic Benchmark towards Text-to-Image Model Optimization (https://arxiv.org/abs/2606.20100)
- **Prior Approaches**: 기존 text-to-image(T2I) 벤치마크는 (1) 주제 범위는 넓지만 평가가 거칠거나, (2) 공간 정렬·속성 결합·시각 텍스트 렌더링처럼 특정 축만 보는 단일 시나리오 중심으로 설계되는 경우가 많습니다. 또한 대다수 평가지표와 데이터가 영어에 치우쳐, 중국어 한자 획 수준 정밀도나 긴 영어 문장의 타이포그래피 레이아웃 같은 언어·문화 비대칭 난이도를 충분히 드러내지 못했습니다. 게다가 VLM 기반 평가는 종종 설명 가능성이 낮거나(black-box), 이미지 환각으로 인해 텍스트 오류를 잘못 통과시키는 문제가 있었습니다.

- **Core Contribution**: 이 논문은 WeGenBench라는 종합적·이중언어(중국어/영어) 벤치마크를 제안하며, 총 4,000개의 테스트 프롬프트를 장면 시나리오(일반 이미지)와 시각 텍스트 렌더링으로 나눠 다각도 성능을 평가합니다. 단순히 프롬프트를 늘리는 대신, 각 프롬프트에 다차원 태그(스타일 제약, 텍스트 렌더링 복잡도, 생성 병목)를 부여해 특정 카테고리에서의 취약점을 정밀하게 핀포인트할 수 있게 했습니다. 나아가 scene 분류와 태그를 함께 활용하는 cross-dimensional 평가로 모델의 강점·약점을 “국소 결함” 수준에서 진단하는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 도전은 (1) 언어별 제약을 반영한 데이터 구조화와 태깅 설계, (2) 다차원 품질을 한 점수로만 뭉개지 않으면서도 안정적으로 자동평가하는 것입니다. 이를 위해 Semantic Alignment는 VLM을 활용해 Checklist-based QA Verification(미세 제약을 Yes/No 질문 리스트로 점검)과 COT-based Deduction(전역 의미·논리 판단)을 병렬로 사용하며, Aesthetic Quality는 Vision-Language Model의 환각을 줄이기 위한 anchor-based match grading을 채택합니다. Visual Text Rendering은 OCR과 VLM을 함께 써서 철자·문장 완결성 같은 텍스트 정확도를 다차원으로 측정하고, 점수 산출에 대한 rationale(추론/근거)를 제공해 검증 가능성을 강화합니다.

- **Empirical Impact**: 저자들은 최신 T2I 모델(오픈소스 및 상용 포함)을 WeGenBench로 체계 벤치마킹해, 기존 벤치마크로는 잘 드러나지 않던 카테고리별 실패 유형과 내재적 취약점을 분석합니다. 특히 중국어 획/타이포 제약과 영어 장문·타이포 레이아웃의 비대칭 난이도가 평가에서 반영되어, 모델 성능 편향을 보다 선명하게 드러내는 것이 의미 있습니다. 또한 제안한 해석 가능한 지표들이 인간 인지와의 상관을 보이며, 향후 특정 태그·시나리오에 맞춘 targeted model optimization 또는 post-training 전략에 실질적인 로드맵을 제공할 수 있음을 시사합니다.



### Stitching and dimensionality effects on large artificially generated volume datasets (https://arxiv.org/abs/2606.20095)
- **Prior Approaches**: 대규모 이미지를 생성할 때는 GPU 메모리 한계를 피하려고 패치를 잘라 만든 뒤 다시 이어 붙이는 ‘stitching’이 흔하지만, 경계 불일치로 인한 stitching artefacts가 생길 수 있다. 기존 연구는 겹침/크롭, valid convolution, overlap-averaging 같은 완화법을 주로 분석·제안해 왔지만, style-transfer 같은 생성 모델에서 artefacts가 downstream 성능에 미치는 영향은 불명확했다. 또한 2D와 3D 패치 선택(문맥/연산 비용)은 세그멘테이션에서는 알려져 있으나, 생성 품질 및 과학 이미징 적응(adaptation) 평가에서의 영향은 충분히 탐구되지 않았다.

- **Core Contribution**: 이 논문은 cryo-electron microscopy(cryoEM) 대형 데이터에서 cycleGAN 기반 style-transfer를 수행할 때 stitching artefacts가 지각 품질과 미토콘드리아 분할(segmentation) 성능에 어떻게 전파되는지 체계적으로 비교한다. 특히 stitching 방식 3가지와 패치 차원(2D vs 3D)을 함께 실험해, 생성 모델 평가에서 perceptual metric 중심(FID)의 한계를 downstream 지표로 드러낸다. 결론적으로, ‘좋아 보이는지’로 끝내지 말고 domain adaptation 품질을 downstream 태스크로 검증해야 한다는 메시지를 제시한다.

- **Technical Challenges**: 핵심 기술 과제는 패치를 이어 붙이는 과정에서 경계 픽셀이 서로 다른 맥락으로 생성돼 생기는 artefacts를 통제하고, 그 차이가 학습/추론 안정성·평가 지표에 미치는 영향을 분리하는 것이다. 저자들은 tile-and-stitch(효과 없는 경계 생성을 위해 valid convolution + downsampling에 맞춘 크롭), overlap-averaging(겹침 평균/크롭), 무겹침 방식 등 3종 stitching을 구성하고, UNet 기반 3D 분할기까지 연결해 IoU로 성능을 비교한다. 또한 타일별 정규화를 피하기 위해 dataset의 global normalisation 통계를 학습 중 수집해 freeze하고, 3D는 배치가 1로 고정되는 현실적 제약까지 실험 설계에 반영했다.

- **Empirical Impact**: FID는 stitching artefacts의 미세한 존재를 잘 잡아내지 못하지만, 그 artefacts는 downstream segmentation 성능에 유의미한 영향을 준다는 점이 실증됐다. 2D 모델은 더 큰 batch size 덕에 학습이 더 안정적이었고, 3D 모델은 artefact-free tile-and-stitch 조합에서만 분할 성능이 약간 우수했으나 계산 비용 대비 이득은 제한적이었다. 반대로 출력이 이미 고품질일 때는 세 방향(orthogonal directions) ensembling이 도움을 주지 못했으며, 과학 이미지에서 생성 품질을 판단할 때 FID만으로는 부족하다는 점을 강조한다.



### MakeupMirror: Improving Facial Attribute Preservation in Diffusion Models for Makeup Transfer (https://arxiv.org/abs/2606.20094)
- **Prior Approaches**: 기존 메이크업 트랜스퍼는 물리 기반 렌더링과 GAN/확산(diffusion) 기반 접근으로 발전해 왔지만, 복잡한 메이크업 질감과 얼굴 보존을 동시에 만족시키기 어려웠습니다. 특히 Stable-Makeup 같은 diffusion 기반 성능은 좋아졌지만, 인물 간(cross-subject) 전송에서 얼굴 특징(눈/코 형태 등)과 피부 톤이 의도치 않게 바뀌는 문제가 남아 e-commerce 수준의 VTO(virtual try-on) 구현을 어렵게 했습니다.

- **Core Contribution**: 이 논문은 얼굴 기하와 피부 톤을 더 강하게 보존하도록 확산 모델을 개조한 MakeupMirror를 제안합니다. 핵심은 (1) facial geometry conditioning으로 얼굴 구조를 유지하고, (2) 부위별로 메이크업 적용 강도를 달리하며, (3) Monk scale 기반 피부 톤 차이를 감지해 전송 강도를 자동 조절함으로써 인물 간 충돌을 줄이는 것입니다.

- **Technical Challenges**: 문제는 메이크업 스타일을 옮기는 동시에 source 얼굴의 정체성/피부 톤을 깨지 않게 제어하는 것입니다. 이를 위해 ControlNets에 depth(Depth-Anything)와 저수준 edge(Canny) 정보를 결합해 얼굴 형태 fidelity를 높였고, segmentation 기반 마스크로 피부/눈/입의 classifier-free guidance scale과 clamping time-steps를 다르게 적용해 over-application을 억제했습니다. 또한 Levenberg–Marquardt Langevin sampler를 통합해 품질 저하 없이 추론을 few-step 수준으로 가속했습니다.

- **Empirical Impact**: CPM-Real, Makeup Wild뿐 아니라 피부 톤 다양성을 강화한 새 데이터셋 MakeupSelfies에서도 Stable-Makeup 대비 성능이 개선됐습니다. 결과로 얼굴 인식 유사도는 +60% 개선, 피부 톤 차이는 -50% 감소, 그리고 뷰티 전문가 기준 결함 감사에서 94% pass-rate를 달성했습니다. 지연 시간은 0.7s로 보고되며, 전체 파이프라인은 약 2.8× 속도 향상까지 확인되어 실서비스 VTO 가능성을 뒷받침합니다.



### EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies (https://arxiv.org/abs/2606.20092)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA)은 대체로 Markovian 가정에 기대어 현재 관측에만 의존해 행동하지만, 실제 로봇 조작은 중간에만 보였다 사라지는 단서(폐색/비관측)가 자주 발생합니다. 메모리 보강 접근은 (1) dual-system으로 분리해 추론·제어를 하거나 (2) recurrent으로 과거를 hidden state에 압축하거나 (3) 메모리 버퍼로 프레임을 쌓지만, 정보 병목·지연·불필요한 시각 중복 축적 문제가 반복됩니다.

- **Core Contribution**: 이 논문은 end-to-end로 동작하는 EventVLA를 제안하며, 핵심은 희소(sparse)한 ‘시각적 증거 메모리’를 설계해 단서가 사라지기 전에 기록하는 것입니다. EventVLA는 고정형 foundational visual anchors(초기 프레임+단기 윈도)와, 미래 핵심 시점 확률을 예측해 이벤트 키프레임을 저장하는 Keyframe Evidence Memory(KEM)를 결합합니다.

- **Technical Challenges**: 난제는 “언제/무엇을” 저장해야 하는지 정확한 타이밍이 없고, 이벤트는 순간적으로 나타나 temporal ambiguity가 생긴다는 점입니다. 저자들은 VLA 트랜스포머의 latent embedding에서 미래 청크 단위 keyframe 확률을 foresight-driven으로 예측하고, 확률 임계치+FIFO 버퍼+1D NMS 및 temporal cooldown으로 실시간 희소 쓰기 스케줄을 만들며, Qwen3-VL 기반 오프라인 자동 라벨링으로 부드러운 soft label 감독과 teacher-to-student curriculum을 학습에 적용했습니다.

- **Empirical Impact**: 평가는 메모리 필요성이 다르지만 static anchor만으로도 풀리는 기존 한계를 겨냥해, RoboTwin-MeM이라는 ‘진짜 비 Markovian’ 진단 벤치마크를 새로 도입해 수행됩니다. 그 결과 EventVLA는 17개 시뮬레이션 메모리 태스크와 4개 실세계 bimanual 태스크에서 기존 memory-augmented VLA 대비 평균 성공률이 +40% 개선되며, RoboTwin-MeM에서는 75.2%를 달성해 희소 이벤트 기억의 효용을 실증합니다.



### Holo-World: Unified Camera, Object and Weather Control for Video World Mod (https://arxiv.org/abs/2606.20083)
Comments:
          Project Page: \url{this https URL} Code: \url{this https URL}

- **Prior Approaches**: 기존 video world models는 카메라/물체 모션 같은 제어를 유지하면서도 환경 상태를 바꾸는 방향으로 발전해 왔지만, 제어와 환경 변화가 분리돼 작동하는 경우가 많았습니다. 특히 날씨(weather) 생성은 대개 소스 비디오나 재구성된 장면이 미래의 구조까지 이미 정해주는 방식에 의존해, 1장(첫 프레임)에서 시작해 자유롭게 구조 보존/전이를 동시에 달성하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 첫 프레임(단일 이미지)에 고정된 first-frame-anchored source-to-state 설정을 제안해, 명시적 camera/object control과 선택적 weather instruction을 입력받아 영상을 생성하되 (1) 원본 세계를 보존하거나 (2) 목표 날씨 상태로 전이하도록 만듭니다. 이를 위해 HoloStateData로 다양한 영상을 통일된 제어 샘플(카메라·물체·날씨 감독)로 재구성하고, 단일 이미지에서 출발하는 unified controllable video world model인 Holo-World를 제시합니다.

- **Technical Challenges**: 핵심 기술 난제는 1장 입력에서 scene 구조를 정밀하게 유지하면서도 weather-dependent appearance와 particle effects까지 자연스럽게 바꾸는 동시에, 제어 조건이 과도하게 섞여 목표 날씨만 ‘과장’되거나 전체 조건을 망가뜨리지 않도록 하는 것입니다. Holo-World는 Unified Scene Adapter로 world preservation과 weather transfer를 분리된 parameter subspace로 factorize하고, rendered background/geometry buffers/객체 제어를 활용해 제어된 장면 구조는 유지하며 날씨 효과는 따로 모델링합니다; 또한 Scene-Weather Decomposed CFG로 scene 잔차와 weather 잔차를 분리 가이드해 목표 날씨 효과를 강화하되 전반적 조건 증폭은 줄입니다.

- **Empirical Impact**: 실험 결과, Holo-World는 camera와 object 제어를 정밀하게 유지하면서 장면 구조의 일관성을 보이고, 다양한 목표 날씨 상태로의 전이도 안정적으로 수행함을 정량·정성 모두에서 확인했습니다. 특히 weather-state generation에서 video-to-video weather editing baselines보다 성능이 우수했으며, 소스 영상에 미래 구조가 미리 포함돼야 한다는 제약을 완화해 controllable weather generation의 실용성을 한 단계 끌어올렸다는 점에서 의미가 큽니다.



### The Hidden Evolution of Disguised Visual Context inside the VLM (https://arxiv.org/abs/2606.20077)
- **Prior Approaches**: 기존 VLM 연구는 시각 토큰을 LLM 입력에 붙여 self-attention으로 처리하는 in-context injection과, 중간 레이어에 시각 정보를 끼워 넣는 layer-wise injection으로 크게 나뉜다. 그러나 두 패러다임을 비교할 때 데이터 구성, 토큰 예산, 모델 스케일, 최적화 조건이 함께 달라져 “통합 아키텍처 자체가 무엇을 바꾸는지”를 인과적으로 분리하기 어려웠다. 또한 분석도 주로 한 가지 통합 방식에 치우쳐, 표상 진화나 모달리티 정렬이 다른 방식으로 일반화되는지 불명확했다.

- **Core Contribution**: 이 논문은 in-context injection(IN-CT)과 두 종류 layer-wise injection(LW-GC, LW-AT)을 동일한 학습 조건에서 단일 이미지/멀티 이미지/비디오 벤치마크로 공정 비교한다. 그 결과 LLM 내부에서 시각 토큰이 ‘언어 구조를 가진 시각 맥락처럼 위장(disguised)되어 들어가지만’, 통합 방식에 따라 표상 형태가 서로 다르게 재구성되며, 각 방식이 시각 신호의 서로 다른 frequency 특성을 포착한다는 “숨은 진화(hidden evolution)”를 제시한다. 더 나아가 attention 분배만으로 성능 차이를 설명할 수 없고, 레이어별 시각 표상의 품질이 실제 능력을 좌우한다고 주장한다.

- **Technical Challenges**: 핵심 기술 난제는 통합 설계만 바꿔도 내부 처리 메커니즘이 어떻게 달라지는지 ‘같은 학습 레시피’로 분리해 측정하는 것이다. 이를 위해 커넥터 구조와 데이터/학습 단계를 통제한 뒤, (1) 레이어 전개에서 시각 토큰 표상이 매끄럽게 진화하는지 CKA로 구조 변화를 추적하고, (2) Fourier 기반 주파수 분석으로 레이어별 고주파/저주파 편향을 정량화하며, (3) PCA로 시각·텍스트 토큰이 언어 공간으로 수렴하는지 확인하고, (4) 생성 과정에서 시각 토큰에 대한 Attention Mass로 실제 활용 타이밍을 측정한다. 특히 IN-CT는 레이어 전반에 걸친 연속적 재구성이 나타나지만, LW-GC/LW-AT는 레이어마다 불연속적 표상 점프가 관찰되어 ‘왜 언어 정렬이 달라지는지’를 구조적으로 뒷받침한다.

- **Empirical Impact**: 실험 결과 IN-CT가 전반적으로 가장 좋은 성능을 보이며, 특히 OCR과 비디오에서 layer-wise 방식보다 큰 격차가 반복된다. OCR-heavy 데이터에서 IN-CT는 전반적인 성능 하락이 크고, layer-wise는 텍스트가 섞인 과제에서만 선택적으로 무너져 “in-context 토큰이 서로 attention으로 결합해 패치/프레임 전반의 세밀한 증거를 조립”하는 능력 차이를 시사한다. 또한 생성 중 attention 분포가 비슷하게 나오는 경우에도 IN-CT가 이기는 것을 통해, 성능은 attention이 아니라 레이어별 시각 표상의 주파수 품질과 언어 공간 정렬의 정도에 의해 좌우됨을 설득력 있게 보여준다. 나아가 IN-CT와 LW-AT를 하이브리드로 결합하면 대부분의 벤치마크에서 성능이 개선되어, 주파수 특성의 상보성을 설계에 반영할 수 있다는 실용적 함의도 제시한다.



### Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers (https://arxiv.org/abs/2606.20076)
- **Prior Approaches**: Latent Diffusion Models(LDMs)은 토크나이저 압축률이 품질-연산 비용 트레이드오프를 좌우하지만, 기존 토크나이저는 고정 압축률이라 시나리오별 최적화에 한계가 있었다. 이를 보완하려고 Variable-length tokenizers(VLTs)들이 등장했는데, 대표적으로 nested dropout은 tail 토큰을 무작위로 잘라 길이를 조절해 효율을 얻는다. 하지만 잘림(truncation) 방식은 토큰 순서에 기반한 의미 구조가 길이마다 달라져, 길이 간 데이터포인트 유사도(대표성) 구조가 어긋나 diffusion 모델이 단일 모델로 다양한 토큰 길이를 잘 일반화하기 어렵게 만든다.

- **Core Contribution**: 이 논문은 길이 조절을 truncation이 아니라 token merging으로 수행하는 “merging-based variable-length tokenizer”를 제안한다. 핵심 아이디어는 merging 패턴(어떤 토큰이 합쳐지는지)이 diffusion transformer 입력에서 함께 고려되면, 길이가 달라도 동일한 ‘cardinality를 맞춘 full-length equivalent representation’ 관점으로 정렬이 가능해진다는 점이다. 즉, 길이로 인해 생기는 대표성 시프트를 줄이기 위해 “유사한 토큰끼리 병합”하도록 학습 목표를 설계한다.

- **Technical Challenges**: 문제는 생성 시점에 데이터(이미지)에 의존하는 conventional merging/클러스터링처럼 “입력에 맞춘 병합 패턴”을 알 수 없다는 호환성 제약이다. 이를 해결하기 위해, 이미지에 무관한 learnable global merging(전역 병합)을 도입해 병합 패턴이 생성 단계에서 고정·접근 가능하도록 만들었다. 또한 merged token의 크기를 반영하는 proportional attention과, 병합된 토큰의 위치 정보를 처리하는 merged positional embeddings를 통해 diffusion이 병합 구조를 일관되게 반영하도록 했다.

- **Empirical Impact**: ImageNet 256×256 생성 실험에서, 제안된 병합 기반 VLT는 기존 VLT 대비 gFID-compute trade-off가 더 우수함을 보였다. 특히 nested dropout 계열에서 관찰되던 길이 간 representational alignment 저하가 완화되어 단일 variable-length diffusion 모델의 성능이 개선되는 방향을 실증했다. 요약하면, “병합 기반 토큰 가변성”이 LDM의 유연한 품질-연산 제어를 더 현실적으로 만들 수 있음을 보여준 결과로 평가된다.



### See-and-Reach: Precise Vision-Language Navigation for UAVs within the Field of View (https://arxiv.org/abs/2606.20045)
Comments:
          12 pages, 7 figures

- **Prior Approaches**: 기존 UAV-VLN 연구는 목표를 “발견(검색)”한 뒤 “접근(도달)”까지를 한 덩어리 search-and-reach로 학습·평가하는 경향이 강합니다. 이때 종종 저해상도 관측과 느슨한 성공 반경(예: 20m)이 쓰여 단말(terminal)에서의 정밀 도달 능력은 진단하기 어렵습니다. 또한 방향(direction) 가이드를 초기 고정 단서로 두는 방식은 비행 중 시점 변화로 누적 drift가 생기기 쉽습니다.

- **Core Contribution**: 논문은 보이는 타깃을 이미 시야(FOV) 안에서 확인한 상태로 두고 “see-and-reach”만 분리 평가하는 UAV-VLN-FOV 태스크를 제안합니다. 성공 기준을 10m로 더 엄격하게 두어, 공중 embodied agent의 정밀한 종단 도달 능력을 측정 가능하게 만듭니다. 이를 통해 언어·시각 증거를 정확한 3D 기동으로 변환하는 핵심 역량을 직접 겨냥합니다.

- **Technical Challenges**: 정밀 see-and-reach에서는 타깃이 작게 보이거나 유사한 시각적 방해물과 섞일 수 있어 고해상도·세밀한 기하 정보 보존이 필수입니다. 3DG-VLN은 다운샘플링을 피하고 front-view와 downward-view를 고해상도로 적응 처리해 fine-grained visual grounding과 waypoint 예측을 강화하며, LoRA로 Qwen2.5-VL을 경량 fine-tuning해 기하 민감도를 확보합니다. 또 방향 단서가 비행 중 어긋나는 문제를 해결하기 위해 closed-loop 추론 중 현재 관측 기반으로 target-relative direction을 online 업데이트해 누적 direction drift를 줄입니다.

- **Empirical Impact**: UAV-VLN-FOV용 전용 고해상도 벤치마크(총 2,717 trajectories)를 구축해 학습·평가 기반을 마련했습니다. 실험에서 3DG-VLN은 경쟁 UAV-VLN baseline 대비 success rate를 13.82% 향상시켰고, 보이는 타깃을 향한 정밀 도달에서 개선이 확인됩니다. 나아가 real-world trial에서도 see-and-reach 내 활용 가능성을 보여 UAV 정밀 내비게이션의 실전 적용성을 높였다는 점이 의미 있습니다.



### FUSE: Frequency-domain Unification and Spectral Energy Alignment for Multi-modal Object Re-Identification (https://arxiv.org/abs/2606.20044)
Comments:
          Accepted in ICML 2026

- **Prior Approaches**: 기존 multi-modal ReID는 RGB-NIR-TIR의 보완 정보를 활용하지만, 주로 공간/시맨틱 정렬에 의존해 mid·high-frequency의 신원 구분 단서를 충분히 모델링하지 못하는 한계가 있었다. 그 결과 저주파 편향이 커지면 주파수 표현이 불완전해지고, 모달리티 간 spectral mismatch로 cross-modal alignment가 불안정해진다.

- **Core Contribution**: 본 논문은 multi-modal ReID를 주파수 영역에서 두 단계로 재구성하는 FUSE를 제안한다: Spectral Decomposition(분해)과 Energy Alignment(에너지 정렬). Spectral Decomposition Module(SDM)은 low/mid/high subspace를 적응적으로 분리해 위계적 spectral modeling을 제공하고, Cross-Modal Alignment Module(CAM)은 밴드별 에너지 및 보완성을 정규화로 강제한다.

- **Technical Challenges**: 핵심 기술적 난제는 모달리티별 저작용(illumination/센서) 차이로 인해 고정된 band 경계가 잘 맞지 않고, 주파수 대역 간 불일치가 정렬을 흔든다는 점이다. FUSE는 learnable soft bandpass/learnable radial masks로 밴드를 데이터에 맞게 조절하고, frequency-consistency regularization으로 모달리티 간 spectral descriptor의 cosine distance를 줄여 안정적인 정렬을 만든다.

- **Empirical Impact**: 실험은 RGBNT201, RGBNT100, MSVR310에서 수행됐으며, FUSE는 RGBNT201에서 mAP 81.4%, Rank-1 86.1%로 기존 대비 mAP +9.1%, Rank-1 +9.5% 수준의 개선을 보였다. 또한 RGBNT100과 MSVR310에서도 SOTA를 달성해 모달리티 간 에너지 정렬·주파수 분해가 견고하고 해석 가능한 주파수 기반 표현 학습 패러다임으로 이어짐을 입증했다.



### PU-UNet: Stable Multiplicative Interactions for Medical Image Segmentation (https://arxiv.org/abs/2606.20035)
Comments:
          Accepted to the ICANN 2026

- **Prior Approaches**: 기존 U-Net 계열 의료 영상 분할은 대부분 합(additive) 기반 특징 변환과 점별(non-linear) 결합으로 고차 상호작용을 간접적으로 학습해왔다. Product units(PU)는 특징을 곱셈으로 조합해 더 직접적인 고차 상호작용 유도를 제공하지만, log–exp 기반의 수치 불안정성 때문에 깊은 dense prediction 네트워크에선 활용이 제한적이었다.

- **Core Contribution**: 이 논문은 U-Net 구조에 Product-Unit U-Net(PU-UNet)을 도입해 명시적 곱셈 상호작용을 안정적으로 주입한다. 핵심은 residual U-Net 블록의 일부를 product-unit residual block으로 교체하되, multiplicative modeling 효과가 클 것으로 기대되는 low-resolution 단계에 선택적으로 삽입하는 설계다.

- **Technical Challenges**: PU는 log–exp 연산 과정에서 값 범위가 커지며 overflow/gradient explosion 같은 최적화 문제가 생기기 쉽다. 저자들은 smooth positivity mapping과 log-domain clipping을 통해 exp 이전에 동적 범위를 제어해, FP16 자동 혼합정밀에서도 안정적인 곱셈 특징 학습이 가능하도록 만들었다.

- **Empirical Impact**: PU-UNet은 ISIC 2018, Kvasir-SEG, BUSI에서 Dice(0.942, 0.959, 최대 0.925)와 IoU를 Residual U-Net 기준선보다 일관되게 개선하면서도 파라미터·FLOPs·추론 지연을 거의 그대로 유지했다. 특히 BUSI의 정상(negative) 케이스에서 image-level false-positive rate를 0으로 낮추는 성과가 보고되어, 실제 배치 가능성까지 시사한다.



### ReA-OVCD: Reliability-Aware Open-Vocabulary Change Detection via Semantic and Spatial Refinemen (https://arxiv.org/abs/2606.20032)
- **Prior Approaches**: 기존 원격탐사 변화탐지는 BCD/SCD처럼 고정된 카테고리(닫힌 집합) 가정을 바탕으로 학습되거나, OVCD에서도 argmax 기반의 인스턴스 매칭·픽셀 비교가 주류였다. 인스턴스 비교는 객체 내부의 부분적 의미 변화(예: 건물 확장 일부)를 놓치기 쉽고, 픽셀 비교는 open-vocabulary에서 예측 분포가 모호해져 경계 부근에 라벨 뒤집힘과 strip-like 아티팩트가 생기기 쉽다. 또한 시점 간 정합 오차와 세그멘테이션 불확실성이 경계 방향으로 누적되는데, 픽셀 단위 독립 비교는 이를 공간적으로 완화하지 못했다.

- **Core Contribution**: 이 논문은 training-free Reliability-Aware Open-Vocabulary Change Detection(ReA-OVCD) 프레임워크로, 미세한 변화 감지 성능과 예측 신뢰도를 함께 균형 있게 확보하는 방법을 제안한다. 먼저 두 시점의 의미 불일치로부터 후보 변화 영역을 만들고, 이후 SCR(Semantic Change Reasoning)로 의미 분포 변화의 “신뢰할 만함”을 재평가한 뒤, BCR(Boundary-aware Change Refinement)로 경계 기반 아티팩트를 억제한다. 결과적으로 단순 출현/소멸 수준을 넘어 부분 건물 확장 같은 fine-grained 변화를 더 안정적으로 찾아내는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술적 난제는 open-vocabulary에서 발생하는 의미 모호성으로 인해 argmax 라벨 플립이 생기면 픽셀 비교가 불안정해진다는 점이다. 저자들은 SCR에서 분포 수준의 차이(Jensen-Shannon divergence)와 로짓 응답 변화 크기를 함께 보고, 그 둘을 가중 결합해 우연한 불일치(incidental label flips)는 낮추고 의미 있는 semantic shift는 보존하도록 설계했다. 두 번째 난제는 정합 오차·세그멘테이션 불확실성이 경계에 strip-like 거짓 변화를 만들 때 이를 공간 맥락으로 제거해야 한다는 점이며, BCR은 경계까지의 거리 기반 신뢰(core pixel)와 연결 성분 내부 지지율로 후보를 필터링해 경계 아티팩트를 완화한다.

- **Empirical Impact**: LEVIR-CD, WHU-CD, DSIFN, SECOND 등 여러 데이터셋에서 ReA-OVCD는 기존 SOTA를 일관되게 능가하며 F1^C가 2.13%~9.75% 개선된다. 또한 multi-stage 복잡도를 줄이기 위해 SAM-3 단일 frozen 인코더를 활용하고, SCR/BCR은 후보 희소 영역에서 저비용으로 수행해 계산 효율도 높다고 보고한다. 즉, open-vocabulary 변화탐지에서 미세 변화와 경계 안정성(아티팩트 억제)을 동시에 달성할 수 있는 실증 근거를 제공한다.



### QG-MIL: A Gated Transformer Aggregator for Domain-Agnostic Multiple Instance Learning in Medical Imaging (https://arxiv.org/abs/2606.20027)
- **Prior Approaches**: 기존 attention-based MIL은 인스턴스에 가중치를 붙여 예측과 함께 해석 가능성을 제공하지만, attention concentration(주의집중)으로 소수 인스턴스에 ‘attention sink’가 생기면 과신(confidence)과 불안정성이 커지기 쉽다. 이를 줄이기 위해 attention masking, self-supervised pretraining, teacher-student distillation 같은 학습 단계/보조 손실을 추가하는 접근이 주로 쓰였다. 다만 이런 처방은 파이프라인을 복잡하게 만들고 추가 비용을 유발할 수 있다.

- **Core Contribution**: 본 논문은 QG-MIL(Qwen Gated Multiple Instance Learning)라는 gated transformer 기반 MIL aggregator를 제안해, 구조적으로 attention sink를 억제한다. 핵심은 RMSNorm 기반 pre-normalization, per-head QK normalization, fine-grained attention output gating, SwiGLU 스타일 feed-forward을 함께 사용하되 auxiliary loss나 multi-stage regularization 없이 “drop-in replacement”로 끼울 수 있다는 점이다. 결과적으로 attention이 더 고르게 분산되도록 설계를 바꿔 학습 안정성과 일반화를 노린다.

- **Technical Challenges**: 관찰된 문제는 MIL에서 attention collapse가 발생할 때 학습이 쉽게 무너지고 예측이 특정 패치에 과도하게 의존한다는 것이다. QG-MIL은 이를 위해 어텐션 블록 내부에 정규화(Q^/K^ per-head)와 게이팅을 정교하게 배치해 attention 출력을 단계적으로 제어하고, SwiGLU feed-forward와 pre-norm 잔차 구조로 최적화 변동성을 줄인다. 또한 small cohort에서는 과도한 복잡한 게이팅이 분산을 해칠 수 있어, ablation으로 구성 요소별 민감도를 함께 확인한다.

- **Empirical Impact**: QG-MIL은 병리(whole-slide pathology: MSK, LungHist700, Prostate)와 세포 혈액(AML-Hehr, APL-AML, cAItomorph) 총 6개 벤치마크에서 최상위 baseline을 모두 능가하며 평균 macro F1이 +6.1점 개선됐다. attention mass 분석에서 top-10% attention 비중이 ABMIL처럼 급격히 쏠리지 않고(예: ABMIL 대비 훨씬 낮은 Gini, 더 높은 entropy), 시각화도 더 부드럽고 분산된 attention map을 보였다. ablation 결과 특정 데이터셋에서는 개별 컴포넌트가 비슷한 성능을 낼 수 있지만, QG-MIL 전체 설계가 도메인 간 성능 일관성과 분산(variance)에서 가장 안정적이라는 점이 확인된다.



### Vision-Reasoning-Guided Occlusion Removal from Light Fields (https://arxiv.org/abs/2606.19985)
- **Prior Approaches**: 기존 occlusion 제거는 주로 image inpainting(단일 이미지 가림 복원)이나 LF 기반 전통/딥러닝 방식으로 나뉜다. inpainting은 마스크 기반으로 누락을 예측하지만, 실제 가림 영역을 물리적으로 관측하지 못해 학습된 의미 priors에 의존하며 hallucination이 생기기 쉽다. LF 전통 기법은 기하학적 정합으로 가림을 줄이지만 잔여 blur·구조 손실 문제가 있고, 딥러닝 LF 방식은 센서/뷰 구성(고정 그리드, 희소 시야 가정)에 민감해 실환경 일반화가 약하다.

- **Core Contribution**: 이 논문은 LFI(라이트 필드 통합)로 가림을 물리적으로 억제한 뒤, VLM(vision-language model)을 조건부 시맨틱 prior로 붙여 구조·고주파 디테일을 복원하는 하이브리드 프레임워크를 제안한다. 핵심은 VLM을 후처리 생성기로만 쓰지 않고, LFI에서 얻은 중간 복원을 “조건부 의미 규칙”으로 보정하는 정밀화 모듈로 설계했다. 또한 고장 가능 지점을 다루기 위해 다중 생성 가설을 융합해 최종 추정의 일관성을 높인다.

- **Technical Challenges**: 문제의 난점은 (1) 중첩된 식생 가림 때문에 관측 증거가 끊겨 기하학적 정합/깊이 단서가 약해지고, (2) VLM의 생성 특성이 관측과 물리적으로 불일치한 hallucinated content를 만들 수 있다는 점이다. 저자들은 LFI로 우선 관측에 정합되는 가림 억제 표현을 만들고, VLM은 이를 바탕으로 defocus와 잡음을 줄이며 구조적 디테일을 복원하되, NN개의 독립 생성 결과를 픽셀 단위로 평균내는 multi-sample fusion으로 환각을 완화한다. 아울러 고정 마스크 의존을 줄이기 위해(가능 시) 선택적 occlusion masking 전처리와 함께, structured/ unstructured LF 뷰 모두에 대응하도록 파이프라인을 구성했다.

- **Empirical Impact**: 합성 벤치마크 4-Syn에서는 4개 장면 평균 SSIM 0.883으로 경쟁 방법 중 최고 성능을 달성했으며, PSNR은 화소 단위 변화 민감도 때문에 상대적으로 낮을 수 있음을 함께 보였다. 실환경에서는 정답 부재로 정량 지표보다 시각적 구조 일관성·가림 영역 가시성·시맨틱 타당성 중심으로 평가했고, 제안 방법이 다양한 획득 설정에서 더 자연스러운 복원을 보였다. 특히 search-and-rescue 같은 실용 시나리오에서 context-aware prompting이 context-free prompting보다 RGB/thermal 모두에서 인공 객체를 줄이고 목표 인식 가능성을 높이는 사례를 제시했다.



### CrossFlow: One-Step Generation Across Latent and Pixel Spaces (https://arxiv.org/abs/2606.19970)
Comments:
          Preprint, Under Review

- **Prior Approaches**: 기존 diffusion과 flow matching 계열은 생성 경로(입력 prior, 확률 경로, 예측 타깃)를 대체로 같은 표현 공간에서 정의해 왔습니다. Latent diffusion은 계산 효율을 위해 생성 경로를 VAE latent 공간으로 옮기지만, 최종 샘플은 별도로 학습된 decoder가 맡아 두 모듈 간 불일치가 생깁니다. 그 결과 생성기는 latent 예측에 맞게 최적화되는데, 실제 화질은 생성된 latent가 decoder에 넣어졌을 때의 대응/분포 이동에 크게 좌우됩니다.

- **Core Contribution**: CrossFlow는 입력은 latent 공간, 출력은 pixel 공간으로 서로 다른 표현 공간을 직접 잇는 cross-space flow 제형을 제안합니다. 특히 latent 궤적은 유지하되, 학습 시 예측 타깃을 latent 변위가 아니라 pixel 이미지를 기준으로 설정하는 velocity-free one-step objective를 도입했습니다. 이를 통해 추론 시 별도 decoder 없이도 1번의 function evaluation으로 latent noise → pixel 이미지를 생성할 수 있습니다.

- **Technical Challenges**: 핵심 난제는 기존 one-step/flow-matching 방식이 latent-space velocity 같은 동형(같은 공간) 구조에 의존해 cross-space supervision이 자연스럽지 않다는 점입니다. 논문은 latent 경로를 정의하는 역할과, 네트워크의 감독 신호를 pixel 공간으로 옮기는 역할을 분리하도록 목적함수를 재구성해 marginal velocity 항을 제거하는 제약을 설계합니다. 또한 시간 미분 항은 JVP(Jacobian-Vector Product)로 계산하고, 안정성을 위해 encoder는 2단계로 고정한 뒤 generator를 새로 학습하는 전략을 사용합니다.

- **Empirical Impact**: ImageNet-1k(256×256, class-conditional)에서 CrossFlow-XL은 1-NFE(1번 function evaluation)로 FID 1.62를 기록해 다단계 latent diffusion과 경쟁하는 성능을 보였습니다. ablation에서는 perceptual loss와 GAN loss가 없으면 FID가 급격히 악화되어(예: GAN 제거 시 8.86, perceptual 제거 시 30.21) 직접 pixel 감독의 품질 향상 효과가 확인됐습니다. 더 나아가 CrossFlow를 VAE-style decoder로도 쓰면, 생성기가 만들어낸 latents에 대해 기존 decoder보다 FID가 지속적으로 개선되는 결과를 제시해 ‘decoder 대체’ 가능성을 실증했습니다.



### Semantic-Anchored Evidential Fusion for Domain-Robust Whole-Slide Survival Analysis (https://arxiv.org/abs/2606.19966)
- **Prior Approaches**: WSI 기반 생존 예측은 다중 인스턴스 학습(MIL)과 attention/Transformer 계열 집계로 패치 정보를 슬라이드 수준 위험도로 변환하는 방식이 주류다. 하지만 대부분의 모델은 단일 임상 센터에 맞춰 학습·검증되어, 염색 프로토콜·스캐너 하드웨어 차이로 인한 도메인 이동이 생기면 성능이 크게 흔들린다. 기존 도메인 이동 대응은 stain normalization이나 self-supervised로 대표되지만, 타깃 센터 데이터 없이 완전한 zero-shot 전이를 보장하기는 어렵다는 한계가 드러난다.

- **Core Contribution**: 논문은 화소 기반 표현이 센터 특유의 잡음(염색/스캐너 아티팩트)과 얽혀 있음을 원인으로 보고, 대신 고수준 병리 의미(종양 grade, 괴사 양상, 미세환경 구조 등)가 센터 불변의 의미 앵커가 될 수 있다고 주장한다. 이를 위해 Semantic-Anchored Evidential Fusion Survival (SAEFS) 프레임워크를 제안하며, VQA로 템플릿 기반 의미 앵커를 자동 추출하고 이를 시각 증거와 함께 생존 예측에 결합한다. 또한 불확실성까지 함께 모델링하기 위해 Dirichlet 기반 evidential learning과 cautious conjunction fusion을 도입해 과신 결합을 억제한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 타깃 센터의 데이터 없이도 의미 기반 앵커를 안정적으로 만들고, (2) 의미·시각 두 증거가 상관되어 있을 때 단순 융합이 과신을 유발하지 않게 하는 것이다. SAEFS는 답이 선택지로 제한된 닫힌 형태(template) VQA 질의로 의미 공간을 bounded/hallucination-resistant하게 만들고, dual-stream에서 시각은 text-guided evidence와 WSI-only evidence로 분리한 뒤 Dirichlet/SUBJECTIVE LOGIC 형태의 belief–uncertainty로 변환한다. 마지막으로 cautious conjunction rule로 두 모달리티의 신뢰도를 불확실성에 반영해 결합 시 공통 증거의 중복 과증거화를 완화한다.

- **Empirical Impact**: SAEFS는 TCGA에서만 학습한 뒤 4개 외부 센터에 대해 zero-shot으로 평가했으며, 평균 C-index를 10.2%p 수준 개선하는 등 예측 정확도와 신뢰도(IBS, INBLL)에서 SOTA를 일관되게 능가한다. 특히 센터 간 분포 이동이 큰 CPTAC-LUAD에서 격차가 크게 나타나 의미 앵커가 실제 도메인 이동 상황에서 유효함을 보여준다. 추가 정량분석에서는 VQA 유래 의미 특징이 pixel-derived 특징보다 cross-center divergence가 유의하게 낮아, 센터 일반화에 필요한 강건성을 뒷받침한다.



### ROSE: Benchmarking the Perception-to-Action Gap in Multimodal Models (https://arxiv.org/abs/2606.19965)
Comments:
          29 pages, 11 figures

- **Prior Approaches**: 기존 MLLM 벤치마크는 MME-Unify 같은 통합형부터 VisuLogic, OmniSpatial, VGRP-Bench 같은 시각-추론 중심까지 다양하지만, 대체로 이미지마다 질문/퍼즐 명세가 고정된다. 또 embodied/vision-language-action 평가는 실행에 가깝지만 계획·제어·환경 상호작용이 함께 묶여 ‘인식→행동 인터페이스’만을 분리하기 어렵다.

- **Core Contribution**: 이 논문은 한 장면(visual evidence)은 고정한 채 과업 컨텍스트와 출력 형식(카운트 vs region-conditioned 클릭/좌표)을 바꿔, 동일한 증거를 현재 컨텍스트에 맞는 행동으로 얼마나 신뢰성 있게 전환하는지 측정하는 ROSE를 제안한다. ROSE는 장면 내부의 암묵적 majority reference(정상 패턴)를 모델이 추론하고, 예외(exception)를 수치적 요약에서 정밀 좌표 실행으로 옮기는지를 진단적으로 본다.

- **Technical Challenges**: 핵심 난제는 ‘정확한 카운트/인식’이 ‘정확한 좌표 집합의 선택’과 자동으로 이어지지 않는다는 점인데, 모델은 같은 증거를 지역 제약(region)과 배제(exclusion) 컨텍스트에 맞춰 다시 바인딩해야 한다. 저자들은 장면 수준 결합 비교와 매칭 컨트롤(global-click bridge, exactly matched regions)을 통해 좌표 그라운딩 실패와 컨텍스트 기반 선택 실패를 분해하고, 문법 VALID와 정답 PASS를 분리해 출력 포맷 오류와 실제 행동 오류를 구분한다.

- **Empirical Impact**: 9개 최신 MLLM 평가에서 인간은 98.8% 평균 PASS를 보였지만, 대부분의 모델은 카운트 지향 과업에서 region-conditioned 액션으로 갈 때 최대 44.5%p까지 성능이 급락했다. 또한 동일 장면에서 전역 카운트를 맞춘 경우에도 region-aware action 정확도가 크게 떨어져, 문제의 병목이 단순 좌표 매핑만이 아니라 컨텍스트에 따른 선택의 비일관성(모델별)임을 보여준다. 모델 비교 진단은 일부 모델(예: GPT-5.5)이 gap을 상당 부분 줄이지만, 다른 모델(예: Qwen3.6-Plus, GLM 계열)은 VALID는 높아도 PASS가 낮은 ‘문법은 맞는데 행동은 틀리는’ 실패 모드를 드러낸다.



### Addressing Detail Bottlenecks in Latent Diffusion for RGB-to-SWIR Image Translation (https://arxiv.org/abs/2606.19961)
- **Prior Approaches**: 기존 RGB→SWIR 이미지 변환은 GAN(Pix2pix, CycleGAN, MUNIT)이나 diffusion을 활용하지만, latent diffusion으로 넘어가면 압축 과정에서 미세한 공간 디테일이 사라지는 문제가 I2I 품질과 지각 성능을 동시에 흔든다. 일부 연구는 sharpness를 유도하는 latent perceptual loss(LPL)나 skip connection 기반 I2I-turbo 같은 방식으로 구조 보존을 시도했지만, 타깃이 RGB가 아닌 cross-modal 도메인에서는 백본 의존성과 ‘디테일 병목’의 핵심 원인(오토인코더 vs 컨디셔닝 경로)을 충분히 분리해 다루지 못했다. 또한 평가에서 FID 같은 생성 품질 지표가 국소 구조 손상을 잘 반영하지 못해, 실제 탐지/인식 성능 저하를 놓칠 수 있다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 latent diffusion의 디테일 손실이 (1) 오토인코더 압축이 만드는 병목과 (2) 컨디셔닝 경로의 naive downsampling이 만드는 병목, 두 가지로 분해된다고 진단한다. 이를 각각 Source-Conditioned Autoencoder(SCAE)와 Learnable Guidance Encoder(LGE)로 경량·백본 불가지로 보완해, decoder에 고해상도 source feature를 주입하고 컨디셔닝 신호 자체도 학습하도록 설계한다. 결과적으로 동일한 latent diffusion 프레임에서 디테일 보존을 구조적으로 복원해 downstream perception에 필요한 국소 정보를 유지하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난제는 압축된 latent 공간에서 사라진 미세 구조를 복원하면서도, 컨디셔닝 단계에서 정보가 또 한 번 손상되지 않게 만드는 것이다. 저자들은 SCAE에서 source encoder를 추가해 multi-scale feature를 decoder의 skip connection으로 주입하고, LGE에서 downsampling을 대체하는 학습형 컨디셔닝 인코더를 두어 denoiser 입력에 채널 방향으로 결합한다. 또한 SCAE와 LGE가 서로 다른 병목을 겨냥하도록 훈련을 분리(오토인코더 1단계 학습 후 고정, denoiser 2단계에서 LGE 공학습)해 안정적으로 효과를 누적한다.

- **Empirical Impact**: RGB→SWIR(운전 장면)에서 U-Net 기반 DP-LDM과 Diffusion Transformer 기반 DP-LDiT 두 denoiser에 대해 detection mAP을 up to 2x까지 끌어올리며, 특히 COCO-small(<32^2 px^2)에서 최대 3.4x 개선을 보였다. 동시에 FID는 state-of-the-art 수준으로 유지되어 생성 품질과 구조적 유용성을 함께 달성하는 경향을 확인했으며, zero-shot으로 공개 RASMD 벤치마크에도 일반화한다. 더 나아가 FID와 탐지 성능 간 상관이 낮다는 점을 실험으로 재확인해, 다중 축(multi-axis) 평가가 필요하다는 메시지를 강화한다.



### SketchKeyAnime: Reference-anchored Sparse Key-Sketch Animation Synthesis (https://arxiv.org/abs/2606.19958)
- **Prior Approaches**: 기존 방법들은 스케치/엣지/선화 같은 구조 조건을 쓰더라도 RGB 경계 프레임이나 촘촘한 프레임 단위 조건, 또는 완전한 스케치 시퀀스를 요구하는 경우가 많았습니다. 반면 inbetweening·interpolation 계열은 자연스러운 전이를 잘 만들지만 애니메이션의 비선형 변형, 저텍스처 영역, 선 경계 문제 때문에 라인 파손·고스트·구조 불일치가 발생하기 쉽습니다. 그 결과 저비용 입력(단일 참조 외 몇 개의 key-pose 스케치) 환경에서의 제어성과 일관성을 함께 만족시키기 어렵다는 한계가 남아 있습니다.

- **Core Contribution**: SketchKeyAnime는 단일 RGB 참조 이미지와 시간 인덱스가 있는 소수의 key-sketch만으로 전체 애니메이션 시퀀스를 생성하는 video diffusion 프레임워크를 제안합니다. 핵심은 (1) sparse key-sketch의 지역 기하 제약과 (2) 스케치 간 시간·의미 문맥을 동시에 모델링해, 스케치 정렬(구조)과 외형(appearance), 그리고 시간적 일관성(temporal coherence)을 같이 맞추는 것입니다. 또한 denoising 과정에서 참조 이미지 조건과 스케치 조건의 주입 강도를 layer-wise로 균형 조절하도록 설계합니다.

- **Technical Challenges**: 가장 큰 도전은 스케치가 없는 대부분의 프레임에서 motion을 추론해야 하는데, frame-wise 공간 제어만으로는 시간 추론이 부족하고 스케치-참조 조건이 denoising 중 경쟁할 수 있다는 점입니다. SketchKeyAnime는 이를 위해 Dual-Branch Conditioning(공간 ControlNet 분기 + semantic-temporal context 인코더 분기)으로 지역 구조와 전역 시간 의존을 분리해 보강합니다. 이어 Sketch Cross Attention에 learnable gating을 넣어 레이어마다 appearance 보존과 sketch 기반 구조 제어의 비중을 자동으로 조절하고, Adaptive Weighted Loss로 key-sketch 프레임 및 line-art 영역의 감독을 더 강하게 줘 희소 조건에서도 학습 안정성을 높입니다.

- **Empirical Impact**: Aesthetic subset의 Sakuga-42M에서 SketchKeyAnime는 대표적 애니메이션 interpolation 및 sketch-guided baseline을 대부분의 정량 지표에서 앞섰고, 특히 EDMD와 FVD에서 유의미한 개선을 보였습니다. 구체적으로 최고 baseline 대비 EDMD를 31.9% 낮추고 FVD를 9.5% 줄여 스케치 충실도와 시간적 일관성이 동시에 좋아졌음을 확인했습니다. 또한 정성 결과와 사용자 연구에서도 sparse key-sketch 입력에 대해 더 자연스러운 중간 동작과 라인 연속성을 보이며, 저비용·고제어 애니메이션 생성 가능성을 실증했습니다.



### Confidence Calibration for Multimodal LLMs: An Empirical Study through Medical VQA (https://arxiv.org/abs/2606.19950)
Comments:
          Accepted by MICCAI 2025

- **Prior Approaches**: 기존 confidence calibration 연구는 주로 LLM의 토큰 likelihood나 verbalized confidence를 온도 스케일링, prompting, self-consistency 등으로 보정하는 방식에 집중해 왔습니다. 다만 Medical VQA처럼 멀티모달·의료 고위험 맥락에서 모델이 내는 자신감이 실제 정확도와 어긋나는 문제는 충분히 분석·해결되지 않았습니다. 특히 의료 도메인 fine-tuning(SFT) 이후 과신이 더 지속될 수 있다는 점이 지적됩니다.

- **Core Contribution**: 이 논문은 의료 멀티모달 LLM에서 accuracy와 self-assessed confidence의 관계를 포괄적으로 분석하고, 이를 Medical VQA에 맞춘 보정 프레임워크로 연결합니다. 핵심 기여는 Multi-Strategy Fusion-Based Interrogation(MS-FBI)으로 질문에 대한 모델의 사고/모순 정보를 수집한 뒤, 보조 expert LLM 평가로 재산정된 보정 confidence를 산출하는 구조입니다. 이를 통해 의료 영역에서 더 신뢰할 수 있는 진단 보조를 목표로 합니다.

- **Technical Challenges**: 의료 과신을 단순히 한 번의 confidence 추출로 보정하기 어렵기 때문에, 모델이 스스로의 답을 재검토하게 만드는 다단계 interrogation 설계가 필요합니다. 논문은 punish 메커니즘으로 고신뢰 오답의 비용을 암시하고, challenge와 explain 전략으로 반박·논리 점검 및 단계별 설명을 유도해 내부 일관성을 흔듭니다. 이렇게 얻은 (질문, 답, 원래 confidence, rebuttal/설명) 정보를 expert LLM(예: llama3-instruct-8B)에 넣어 재보정 confidence와 해석을 동시에 산출합니다.

- **Empirical Impact**: 3개 Medical VQA 데이터셋에서 제안 방법은 Expected Calibration Error(ECE)를 평균 40% 가까이(예: LLaVA-1.5-med-7B의 44.28%→26.22%) 낮추고 AUROC도 향상시켰습니다. 또한 도메인 특성에 따라 calibration 성능이 크게 달라지며, 의료 도메인 모델은 일반 모델보다 과신이 더 두드러지고 보정 필요성이 크다는 점을 실험으로 보여줍니다. ablation과 교차모델 실험은 전략 조합과 expert 평가가 calibration 효과의 핵심이며, 복잡도가 곧 성능으로 이어지지는 않는다는 실용적 인사이트를 제공합니다.



### Timage: A Generative Text-in-Image Paradigm for Fine-Tuning Vision-Language Models (https://arxiv.org/abs/2606.19944)
Comments:
          ECCV

- **Prior Approaches**: 기존 MLLM의 미세 공간 추론은 텍스트 쿼리가 픽셀 좌표의 기하학적 앵커를 제공하지 못해 ‘주의가 틀어진다’는 구조적 한계를 겪는다. 이를 개선하려는 weight-space 적응(예: LoRA, adapter, full fine-tuning)은 과적합/지식 훼손 위험이 있고, input-space prompting(VPT 등)은 기하를 약간 조정해도 사람이 읽을 수 있는 의미 있는 신호를 직접 삽입하지 못한다. 결국 대부분의 방법이 언어-시각 정렬을 “암묵적 임계값”에 의존해 정확한 영역 고정이 어렵다.

- **Core Contribution**: Timage는 쿼리를 이미지 위에 ‘타이포그래피 오버레이(Typeset overlay)’로 렌더링해, 언어 의미를 픽셀 도메인의 정렬 신호로 전환한다. 이 방식은 단일 시각 입력을 통해 다운스트림 MLLM이 해당 공간 의미를 따라가도록 하는 외부 어댑터로 동작하며, 모델 파라미터를 건드리지 않아 architecture-neutral이다. 핵심은 “주의 비콘”처럼 보이는 입력 재구성 자체가 공간 추론을 강화한다는 관점이다.

- **Technical Challenges**: 문제는 오버레이를 단순히 고정 캡션처럼 찍거나 랜덤 배치하면(혹은 의미 없는 위치 토큰만 바꾸면) 공간·가시성 제약을 만족하면서 쿼리 정합성을 동시에 달성하기 어렵다는 점이다. 논문은 Constrained Schrödinger Bridge(cSB)로 오버레이 합성을 두 단계(Region Search: 의미 정합 + hard occlusion barrier, Appearance Shaping: ink-budget로 글자 가독성과 균형 조절)로 분해해, 제약을 생성 과정에 내재화한다. 또한 교차어텐션 기반 query-relevance heatmap과 admissible mask를 사용하고, Projected Euler–Maruyama 및 STE를 통해 비차별 가능한 제약/투영을 포함해 학습한다.

- **Empirical Impact**: VMCBench에서 Timage는 7B 백본만으로 평균 정확도 87.7%를 기록하며 GPT-4o(+7.4%), Qwen2-VL-72B(+2.7%)뿐 아니라 전체 fine-tuning(+2.5%)도 앞선다. 특히 Reasoning과 Doc&Chart에서 개선폭이 커, 오버레이가 실제로 정밀 공간 정렬을 제공한다는 점을 보여준다. 또한 여러 백본(LLaVA-1.5~Qwen3)에 공통으로 성능이 오르는 일반성과, 훈련 파라미터 대비 효율에서 LoRA/다른 PEFT를 상회해 “입력 재구성이 파라미터 적응보다 강력할 수 있다”는 메시지를 강화한다.



### DiffMath: Symbol- and Graph-Aware Latent Diffusion Transformer for Handwritten Mathematical Expression Generation (https://arxiv.org/abs/2606.19939)
- **Prior Approaches**: 기존 Handwritten Mathematical Expression Generation(HMEG) 방법들은 FormulaGAN류의 이미지 기반 생성에서 출발해, 최근에는 문자 생성과 공간 레이아웃을 분리하는 two-stage 방식(예: DiffInk, One-DM)으로 발전했다. 하지만 대부분이 symbol-level bounding boxes 같은 위치/공간 감독에 강하게 의존해, 주석 비용이 크고 다양한 수식 구조로의 확장성이 제한된다. 또한 LaTeX를 그대로 선형 조건으로 쓰거나 MathML AST를 쓰더라도, 2D 공간 관계를 명시적으로 다루지 못해 밀집 수식에서 구조 붕괴가 발생하기 쉽다.

- **Core Contribution**: 이 논문은 DiffMath라는 symbol- and graph-aware latent diffusion 프레임워크를 제안해, 위치 감독 없이도 LaTeX의 계층 구조를 structural prior로 활용하도록 설계했다. 핵심은 MathML 트리를 생성 지향 표현으로 압축하는 Relational Abstract Syntax Tree(RelAST)로, 수식을 [S, R, D] triplet 시퀀스로 바꿔 diffusion이 내용을 구조적으로 생성하게 만든다. 여기에 MathVAE(구조 보존 latent 학습)와 MathDiT(구조 조건부 latent denoising)를 결합해 상징 정확도와 2D 위상 일관성을 동시에 노린다.

- **Technical Challenges**: 가장 큰 기술 난제는 LaTeX의 1D 문자열이 내포한 계층/관계 정보를 2D 필기 궤적 생성에 직접 연결하는 방식이 어렵다는 점이다. 이를 위해 LaTeXML로 MathML을 표준화한 뒤, MathML의 비단말(컨테이너) 토큰을 줄이고 ‘앞 기호 대비 공간 관계(R)’와 ‘중첩 깊이(D)’를 함께 포함하는 RelAST triplet로 재구성했으며, MathVAE는 symbol-aware 및 relation-aware perceptual regularization으로 latent이 의미와 위상을 함께 담도록 학습시킨다. 또한 전역 symbol-count prior를 시간(t) 모듈레이션(AdaLN)으로 결합해 긴 수식에서 기호 누락/과생성 같은 구조적 부정합을 완화한다.

- **Empirical Impact**: MathWriting에서 DiffMath는 ExpRate 70.70과 최저 FID 5.43을 달성하며, 기존 SOTA(예: DiffInk, FormulaGAN) 대비 구조·내용 정확도를 동시에 개선했다. 특히 무구조 텍스트 생성 방식 대비 ExpRate가 크게 상승했고, 사용자 연구에서도 Qwen-Image 등 강한 대형 생성 모델을 제치고 상위권을 기록했다. 생성된 필기 수식을 downstream OCR 성능을 높이기 위한 synthetic data augmentation에 활용할 수 있음을 보여주며, HMEG 분야에서 ‘위치 감독 없이도 구조 일관성을 만드는’ 확장성 있는 경로를 제시한다.



### Triangular Consistency as a Universal Constraint for Learning Optical Flow (https://arxiv.org/abs/2606.19938)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 광학흐름(optical flow) 학습은 프레임을 두 장 단위로 예측하는 구조가 대부분이며, self-supervised에서는 photometric reconstruction과 forward-backward 같은 일관성 제약으로 감독 신호를 보완해 왔습니다. cycle consistency나 transformation consistency도 널리 쓰이지만, 대체로 특정 상황에 맞춘 “쌍(pair) 기반” 근사에 머물러 세 프레임 이상의 조성(composition) 원리를 포괄적으로 다루지는 않았습니다.

- **Core Contribution**: 이 논문은 optical flow의 기하학적 성질을 바탕으로, 세 프레임에서 “두 흐름을 합성해 얻은 제3의 흐름”이 직접 추정한 흐름과 일치해야 한다는 triangular consistency를 제안합니다. 이 제약은 네트워크 아키텍처, supervision 형태(지도/무지도), 데이터셋에 비의존적이며 연산 오버헤드가 거의 없는 plug-and-play 학습 구성요소로 설계됩니다.

- **Technical Challenges**: 핵심 과제는 흐름 합성이 항상 성립하는 연속 좌표변환이라는 물리적 규칙이 2D 영상에서는 occlusion/비가시성 경계에서 깨질 수 있다는 점입니다. 논문은 forward-backward consistency로 유효 마스크를 만들어 triangular residual을 robust norm으로 가중 페널티하며, 동시에 temporal chaining(3프레임)과 affine 기반 controlled augmentation(닫힌형식 좌표 변환으로 pseudo target 생성)을 같은 삼각 원리로 구현해 학습 안정성과 정확성을 함께 노립니다.

- **Empirical Impact**: 실험에서는 지도/무지도/transfer learning 전반에서 일관된 성능 향상을 보였고, 레이블 없이 self-supervised adaptation을 단 1 epoch만 수행해도 Sintel에서 최대 18.1%까지 개선되었다고 보고합니다. 또한 unsupervised 학습 파이프라인(ARFlow 기반)에 triangular consistency를 추가했을 때도 EPE와 구조적 정합성이 함께 좋아지며, out-of-domain 및 cross-dataset 일반화에서 특히 큰 이득을 보였습니다.



### Speeding up the annotation process in semantic segmentation industrial applications (https://arxiv.org/abs/2606.19934)
- **Prior Approaches**: 기존 연구는 라벨링 시간을 측정하거나, 비지도/파운데이션 기반으로 초기 마스크를 만들되 “정밀한 픽셀 단위 세그멘테이션”에서는 도메인 특수성 때문에 단독 해결이 어렵다고 봤습니다. 또한 annotation-efficient semantic segmentation은 active-learning 클릭이나 약한/거친 감독 등 초기에 인간 입력이 남는 경우가 많아, 고해상도에서의 대량 라벨링 병목을 직접 줄이기엔 한계가 있었습니다. 무엇보다 비지도 알고리즘이 실제로 라벨링 시간을 얼마나 단축하는지 정량 비교가 부족했습니다.

- **Core Contribution**: 이 논문은 고해상도 철강 마이크로구조(픽셀 단위 라벨링)에서, 비지도 알고리즘의 pre-annotation을 “scratch(처음부터 수작업)”과 같은 실험 설계로 비교해 라벨링 시간을 정량 단축하는 것을 핵심 기여로 제시합니다. 그 결과 수작업 170시간에서 37시간으로 줄여 약 78% 절감 효과를 보였고, 이는 계산 오버헤드(수 분~수십 분)가 전체 수작업 시간에 비해 미미함을 함께 확인했습니다. 동시에 MIT License로 82장의 fully annotated 대규모 공개 데이터셋 MicroSteel(영구 DOI)을 제공해 이후 벤치마크 기반 연구를 촉진합니다.

- **Technical Challenges**: 문제는 “정확도 부족을 감수할 수 있는가”가 아니라, 고해상도에서 픽셀 경계 연속성과 복잡한 소수 클래스까지 포함한 라벨을 사람 수정 단계로 넘길 만큼 일관된 pre-mask 품질을 확보하는 데 있습니다. 저자들은 다종 비지도 사전 라벨링(멀티 Otsu, superpixels, k-means, 비지도 DL, SAM)을 비교한 뒤, 이미지 타입 전반에 강건한 비지도 DL을 최종 통일 워크플로로 선택하고 Labelbox에서 전문가가 정교화하도록 설계했습니다. 특히 Majority 클래스는 과분할(over-segmentation)로 자동화하되, 전문가가 소수 클래스·복잡 경계(TiB2/TiN 등)에 집중하도록 수정 난이도를 재배치했습니다.

- **Empirical Impact**: 실험에서는 전문가 3인의 라벨링 시간을 Labelbox가 기록한 중앙값을 기준으로 비교했으며, 비지도 pre-annotation이 Type 전반에서 73~84% 수준의 시간 절감을 보였습니다(전체 평균 78%). 또한 IoU/Dice/Boundary F1/Hausdorff distance로 pre-mask와 최종 라벨의 정합성을 계량해, 소수 클래스와 경계에서 더 많은 수정이 필요함을 정량화했습니다. 더 나아가 비지도 DL이 Type III 같은 난이도 높은 경우에도 전반적으로 안정적이라 전문가 편차와 분산을 크게 낮출 수 있음을 보여주며, 산업 현장에서 라벨링 가능한 프로젝트 범위를 확장하는 실질적 의미가 있습니다.



### Spatial-Aware Reduction Framework: Towards Efficient and Faithful Visual State Space Models (https://arxiv.org/abs/2606.19932)
Comments:
          Accepted by ICML 2026

- **Prior Approaches**: Mamba의 효율은 selective scanning으로 인해 Transformer의 제곱 복잡도보다 낮은 연산을 달성하지만, 토큰 감축에는 취약하다는 문제가 제기돼 왔습니다. 특히 VMamba처럼 2D Selective Scan(SS2D)을 쓰는 변형에서 기존 pruning·merging 계열은 전역(공간 비인식) 기준으로 토큰을 줄이고 다시 2D로 매핑하는 과정에서 토큰의 공간 관계를 깨뜨려 성능이 급격히 붕괴합니다.

- **Core Contribution**: 본 논문은 “좋은 토큰을 남기는 것”보다 “모델이 전제하는 2D 토폴로지를 압축 과정에서 유지하는 것”이 성패를 좌우한다는 점을 명확히 합니다. 이를 바탕으로 STORM(Spatial-aware TOken Reduction)이라는 학습 없이도 끼워 넣을 수 있는(plug-and-play) 토큰 감축 프레임워크를 제안합니다.

- **Technical Challenges**: 기존 감소 방식은 토큰을 한 줄로 펼쳐 전역 선택·병합을 수행해 SS2D가 기대하는 2D 격자와 이웃 일관성을 위반하기 때문에 연쇄적인 정보 손실이 발생합니다. STORM은 2D 격자에서의 구조를 보존하도록 감소를 “행→열”의 두 단계 구조화 연산으로 재구성하고, windowing으로 로컬 이웃 범위를 제한해 지역 의미의 붕괴를 억제합니다.

- **Empirical Impact**: 실험에서 STORM은 training-free 설정 하에 다양한 vision Mamba 백본에서 최고 수준의 pruning 정확도를 보이며, 특히 VMamba에서는 top-1 정확도 최대 63.3% 개선(정확도 회복) 성과를 보였습니다. 또한 PlainMamba에서는 정확도 하락이 1.0%에 그쳐 ViT와 유사한 성능을 유지하며, ToMe 대비 높은 처리량과 다운스트림(검출·분할)에서도 일관된 우위를 확인했습니다.



### CARE: Competence-Aware Reward Shaping for Adaptive Reasoning Length in Video-MLLMs (https://arxiv.org/abs/2606.19927)
- **Prior Approaches**: 기존 비디오 멀티모달 추론에서는 chain-of-thought(CoT)를 유도하는 후학습 뒤, reasoning length(추론 길이)를 고정 보상/페널티/임계값으로 제어하는 방식이 많습니다. 그런데 강화학습에서 모델의 능력은 학습 중 계속 변하고, 정적 제어는 early exploration을 방해하거나 late 단계에서 토큰 낭비를 남기기 쉽습니다. 또한 단순 길이 정규화는 인스턴스 난이도에 따른 필요한 ‘추론 예산’ 차이를 반영하지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 CARE(Competence-Aware Reward Shaping)를 제안해, 추론 길이 선호를 모델의 현재 competence(역량)와 인스턴스 난이도에 맞춰 비정상(non-stationary)적으로 조절합니다. CARE는 EMA로 competence를 안정적으로 추정한 뒤, 훈련을 단계적으로 라우팅하며 보상 선호를 ‘탐색용 긴 추론’에서 ‘효율 중심의 간결한 추론’으로 점진 전환합니다. 게다가 배치 통계로 추론 노력(토큰)을 난이도 대비로 보정하고, 예외적으로 강한 성과를 보이는 샘플에는 posterior amplifier로 보상을 더 강하게 증폭합니다.

- **Technical Challenges**: 핵심 기술 난제는 competence 신호를 배치별 정확도 같은 즉시 지표로 직접 쓰면 난이도 편차로 라우팅이 출렁인다는 점입니다. CARE는 group pass rate를 기반으로 EMA competence를 만들고, 이를 stage-aware effort score로 변환해 길이 보상을 ‘현재 능력과 상황에서 바람직한 노력’으로 재정의합니다. 또 verbosity(장황함)와 intrinsic complexity(본질적 복잡도)를 섞지 않기 위해 배치 내 성공 샘플의 평균/분위수 통계를 사용한 정규화를 적용하고, posterior amplifier를 stage별로 다르게 설계해 학습 안정성을 유지합니다.

- **Empirical Impact**: 실험에서는 Video-R1 계열 벤치마크 및 일반 비디오 이해(MVBench, TempCompass, Video-MME 등)에서 CARE가 일관되게 정확도 개선과 강화학습 안정화를 보였습니다. 특히 동일 백본과 16-frame 조건에서 reasoning-intensive 데이터셋(VSI-Bench, VideoMMMU) 성능이 Video-R1 대비 유의미하게 상승했고, token efficiency(토큰 효율)도 크게 향상되었습니다. 학습 중 reasoning length가 inverted-U 형태로 나타나며 최종 수렴 시에는 더 짧지만 정보 밀도가 높은 추론이 형성된다는 분석은 ‘추론 예산의 적응적 배분’이 실제로 작동함을 보여줍니다.



### SpatialSV: Internalizing Interpretable 3D Spatial Awareness in MLLMs via Task-Oriented Visual Supervision (https://arxiv.org/abs/2606.19915)
Comments:
          Accepted by IJCAI 2026

- **Prior Approaches**: 기존 연구는 MLLM에 공간 priors를 외부 도구로 주입하거나, 3D 비전 파운데이션 모델의 특징을 distillation해 넣는 방식이 주를 이룹니다. 전자는 추론 비용과 외부 모델 오류에 민감하고, 후자는 내부 표현이 해석이 어렵고 정밀한 기하 제약이 부족하다는 한계가 있습니다. 또한 MLLM이 2D 기반과 autoregressive 학습에 치우쳐 있어 일관된 내재 3D 공간 표상이 부족하다는 문제가 강조됩니다.

- **Core Contribution**: 이 논문은 2D 시각 특징을 명시적 3D 표현(깊이, 카메라 pose, point cloud)으로 끌어올리고 이를 해석 가능하게 제공하는 프레임워크 SpatialSV를 제안합니다. 핵심은 passive feature imitation 대신 task-oriented visual supervision을 통해 모델이 스스로 3D 공간 인식을 내부화하게 만드는 것입니다. 3D lifting 결과는 모델 표현의 품질을 시각적으로 진단하는 직관적 프록시 역할도 수행합니다.

- **Technical Challenges**: 문제는 feature distillation처럼 거친 정렬로는 기하 디테일을 살리지 못하고(blur·정보 손실), 해석 가능성도 떨어진다는 점입니다. SpatialSV는 MLLM의 여러 레이어 hidden visual features를 투영해 3D 공유 공간으로 lift한 뒤, task-decoupled DPT 모듈로 depth/ray/pointcloud를 멀티태스크로 예측하게 하며 깊이 잔차의 gradient loss, surface normal loss 등 미세 기하 제약을 추가합니다. 이렇게 coarse-grained 지침과 fine-grained 기하 제약을 함께 걸어 robust 내재 공간 표상을 학습시킵니다.

- **Empirical Impact**: 실험에서 SpatialSV는 여러 MLLM과 벤치마크에서 spatial intelligence 향상을 일관되게 보이며, 특히 distillation 대비 더 선명한 3D 재구성과 작은 성능 격차를 보였습니다. MindCube-Tiny 등에서 질문-응답 정확도 상승뿐 아니라, depth probing의 RMSE/시각 결과를 통해 표현 품질 개선과의 상관도 확인됩니다. 또한 반지도(semi-supervised) 설정에서 텍스트 주석 비율을 크게 줄여도 유사 성능을 보여, 라벨이 부족한 환경에서의 확장성과 해석 가능성 가치를 입증합니다.



### Gaussian Process Prior Variational Autoencoder for Endoscopic Videos (https://arxiv.org/abs/2606.19908)
- **Prior Approaches**: 기존 내시경 영상 복원은 프레임 단위로 처리하는 경우가 많아, 연속 프레임 간 시간적 상관을 충분히 활용하지 못하는 한계가 있었다. VAE 계열은 잠재표현 학습에는 유리했지만 specular reflections 같은 일시적 오염이나 누락 프레임을 ‘연속 궤적’으로 추론하기보다는 개별 프레임 복원에 머무르기 쉬웠다. 한편 GPVAE류는 시간 우선순위의 확률적 잠재모델을 제공하지만, 장시간 시퀀스에서 계산량이 커져 내시경 데이터에 그대로 적용하기 어려웠다.

- **Core Contribution**: 이 논문은 Gaussian Process Prior Variational Autoencoder(GPVAE) 프레임워크로 내시경 비디오 복원에서 잠재공간의 prior를 factorized 독립가정에서 시간 Gaussian process prior로 교체한다. 이를 통해 관찰된 프레임으로부터 시간에 연속적인 잠재 궤적을 추정하고, 누락 프레임을 불확실성까지 포함해 보간(interpolation)할 수 있도록 만든다. 또한 내시경 전용 인코더/디코더(예: EndoVAE backbone, GastroNet-5M 기반 Vision Transformer 인코더)와 GP 근사(HPA, SPA)를 결합해 도메인 적합성과 확장성을 동시에 노린다.

- **Technical Challenges**: 핵심 난제는 (1) 시간적 상관을 담는 GP prior를 쓰면서도 (기본 GPVAE의) 계산량 폭증 문제를 해결하고, (2) specular reflections처럼 복원 목표를 망가뜨리는 오염 픽셀을 학습/복원 손실에서 배제하는 파이프라인을 만드는 것이다. 저자들은 HPA(Hierarchical Prior Approximation)와 SPA(Sparse Precision Approximation)로 GP 근사를 스케일업하며, DUCKNet 기반 마스킹으로 반사/오염 영역을 reconstruction objective에서 제외해 모델이 ‘쓸모없는 관측’에 휘둘리지 않게 했다. 여기에 LPIPS 및 MSE 기반 복원항과 KL 항을 함께 두고, GP posterior가 프레임별 uncertainty를 제공하도록 학습 목표를 설계했다.

- **Empirical Impact**: C3VDv2 colonoscopy 데이터에서 GPVAE 변형들은 VAE 기준선 대비 이미지 재구성 RMSE를 평균 21.9% 줄였고, 일부 설정에서는 최대 26.1%까지 개선됐다. 다운스트림에서는 classical visual odometry 및 pretrained PoseNet 모두에서 trajectory RMSE가 평균 12.7% 감소했지만, 에폭당 학습 시간은 평균 27.3% 증가했다. 특히 GP posterior가 프레임/영역별 불확실성 신호를 제공해, 복원 결과에 대한 신뢰도(의료적 검증 필요 구간)를 내재적으로 가시화할 수 있다는 점에서 의미가 크다.



### Linear Recurrent Unit with Semantic Modulation for Image Super-Resolution (https://arxiv.org/abs/2606.19901)
Comments:
          Accepted to CVPR 2026 Findings

- **Prior Approaches**: 기존 SR은 CNN(예: SRCNN, EDSR)과 attention/Transformer(예: SwinIR, DAT, RGT)로 발전했지만, 고해상도에서 계산량이 급증하는 문제가 남아 있었다. 최근 Mamba 등 deep state-space model(SSM) 계열은 효율과 global receptive field를 개선했으나 dynamic parameterization, 복잡한 discretization과 특수 초기화 때문에 해석성이 떨어지고 구조가 무거워질 수 있다. LRU는 안정적인 선형 recurrence로 long-range를 잘 다루지만, 단일(정적) 스캔은 2D 공간에서 국소·전역의 다양한 패턴 적응에 한계가 있었다.

- **Core Contribution**: 이 논문은 LRU의 안정적 recurrence를 유지하면서 단일 이미지 super-resolution에 맞게 “의미(semantic) 기반” 적응을 넣은 LRU 기반 복원 네트워크를 제안한다. 핵심은 Semantic Modulating Unit(SMU)을 통해 LRU의 전이(모듈레이션)를 입력 의존적으로 바꾸고, 픽셀을 의미 유사도로 범주화해 단일 스캔에서도 먼 픽셀 간 상호작용을 강화하는 것이다. 또한 learned dictionary 기반 prototype을 통해 cross-attention 성격의 feature enhancement까지 함께 수행한다.

- **Technical Challenges**: 핵심 과제는 1) LRU의 정적 recurrence 장점을 보존하면서도 2) 2D SR에서 공간적으로 달라지는 패턴에 맞는 입력 의존적 표현력을 추가하는 것이다. 연구진은 RG-LRU의 gating 아이디어를 참고해 hidden recurrence 계수를 입력 의존적으로 modulate하는 경로를 설계하고, dictionary와 cosine-similarity 기반 affinity로 modulating tokens를 생성해 λ/B/C 성분(복소 표현 포함)에 픽셀별 게이팅을 적용한다. 더 나아가 SMU가 (a) semantic categorization, (b) modulating LRU, (c) dictionary 기반 global feature enhancement의 다역할을 하도록 구성해 overhead를 크게 늘리지 않으면서 적응성을 얻었다.

- **Empirical Impact**: DIV2K/Flickr2K로 학습하고 Set14, Manga109, Urban100 등에서 실험한 결과, LSM 계열(L S M-S/LSM, 그리고 경량 LSM-light)은 최근 state-of-the-art를 정량·정성 모두에서 앞서는 성능을 보였다. 특히 FLOPs를 크게 줄이면서도 PSNR이 개선되며, MambaIR 대비 Urban100에서 FLOPs 32.8% 감소와 PSNR +0.26dB 같은 효율-정확도 동시 향상을 보고한다. 시각적으로도 원형/줄무늬 등 반복 텍스처 복원이 더 선명하고 아티팩트가 줄어, 의미 범주화 기반의 일관성 강화 효과가 확인된다.



### SurgVista: Long-Horizon Surgical World Modeling with Plausible Instrument-Tissue Dynamics (https://arxiv.org/abs/2606.19889)
- **Prior Approaches**: 자율수술을 위한 Surgical World Models(SWMs)는 초기 관측과 계획된 도구 행동을 조건으로 미래 영상을 롤아웃해 정책 학습/평가를 안전하게 돕지만, 두 가지 실패가 반복됐습니다. 첫째는 spatial interaction incoherence로, 도구 접촉이 있어도 조직 변형이 프레임 간 일관되게 이어지지 않는 문제입니다. 둘째는 temporal fidelity collapse로, 자기회귀 롤아웃이 길어질수록 예측 오차가 누적되어 색·노출·구조가 점점 무너지는 현상이 나타납니다.

- **Core Contribution**: SurgVista는 spatial interaction incoherence와 temporal fidelity collapse를 동시에 겨냥한 외과 전용 surgical world model로, 두 가지 학습 레시피를 제안합니다. Deformation Consistency Regularization(DCR)은 장면 포인트 궤적을 기반으로 프레임 간 정합성을 강제해 도구-조직 상호작용의 물리적 일관성을 높입니다. Drift Adaptation Training(DAT)은 롤아웃에서 발생할 conditioning drift를 학습 중에 모사해 시각 품질 붕괴를 완화합니다.

- **Technical Challenges**: 핵심 난제는 (1) 도구 접촉이 야기하는 국소적이고 복잡한 조직 변형을 프레임 간 “대응”까지 포함해 학습해야 한다는 점과 (2) 롤아웃 시 train-inference 분포 불일치가 오차 누적을 가속한다는 점입니다. SurgVista는 DCR에서 점 트래커로 dense scene-point trajectories를 뽑고, latent contrastive learning(InfoNCE)으로 시간에 걸친 일관된 표현을 고정해 공간 상호작용을 정리합니다. DAT에서는 모델의 온라인 prediction residual을 조건 프레임에 주입하고, brightness/contrast/sharpness/noise 같은 photometric perturbation을 장기 롤아웃 통계에 맞춰 적용해 드리프트에 강인한 조건 복원을 학습합니다.

- **Empirical Impact**: 평가는 SurgWorld-Bench로 수행됐으며, 여러 수술 절차 유형과 80 프레임 대비 최대 800 프레임까지의 장기 롤아웃을 포함합니다. 또한 instrument-motion 정확도와 tissue-response 충실도를 decoupled metric으로 나눠 상호작용 품질을 더 엄밀히 봅니다. 실험 결과 SurgVista는 기존 state-of-the-art를 시각 품질, 시간적 일관성, 상호작용 정합성 전 영역에서 능가했고, 특히 horizon이 길어질수록 격차가 더 커졌습니다.



### Multimodal Concept Bottleneck Models (https://arxiv.org/abs/2606.19882)
Comments:
          Present at NeurIPS 2025 Mechanistic Interpretability Workshop

- **Prior Approaches**: 기존 Concept Bottleneck Models(CBMs)은 중간 표현을 인간이 이해 가능한 개념(Concept Bottleneck Layer)으로 맞춘다는 점에서 해석 가능성이 높지만, 출력 라벨이 사전에 정의된 클래스에 묶여 natural language 질의에는 약하다는 한계가 있었다. 또한 개념층의 의미가 실제 예측을 “통제”하지 못하고, 선형 분류기가 개념 활성과 무관한 경로로도 예측을 복원(information leakage)할 수 있다는 문제(충실성 저하)가 지적돼 왔다.

- **Core Contribution**: 이 논문은 Multimodal Concept Bottleneck Model(MM-CBM)로 CBM을 CLIP(비전-언어 임베딩)로 확장한다. 이미지와 텍스트 각각에 대해 dual Concept Bottleneck Layer(CBL)를 두고, 두 모달리티의 ‘같은 개념 공간’ 정렬 정도(유사도)로 추론해 개념 기반의 zero-shot 분류 및 이미지 검색을 해석 가능하게 수행한다.

- **Technical Challenges**: 핵심은 (1) 임의의 텍스트 입력을 개념 응답으로 변환하고 (2) 개념층이 예측을 실제로 지배하도록 충실성을 유지하는 것이다. 이를 위해 LLM으로 후보 개념 집합을 만들고(클래스 고정 문제 완화), OWLv2 검출과 텍스트-개념 유사도(ACS 방식)를 이용해 시각/언어의 개념 라벨을 구성한 뒤, interpretability loss와 task 성능 loss를 함께 최적화하며 NEC를 통한 희소성을 제약해 leakage 위험을 줄였다.

- **Empirical Impact**: 실험에서 MM-CBM은 여러 벤치마크에서 기존 CBM 대비 최대 51.26%의 평균 정확도 향상을 보였고, black-box CLIP 성능 대비 약 5% 이내로 유지되며 해석 가능성을 확보했다. 또한 VLM이 생성한 “설명”과 비교할 때 MM-CBM이 더 유의미한 시각 개념을 더 자주 제공했으며, 이미지 retrieval에서도 라벨 고정 없이 텍스트 쿼리에 대해 사람 해석과 대체로 일관된 결과를 보여 의미 있는 실용성을 입증했다.



### PSCT-Net: Geometry-Aware Pediatric Skull CT Reconstruction via Differentiable Back-Projection and Attention-Guided Refinemen (https://arxiv.org/abs/2606.19867)
Comments:
          11pages, 5 figures

- **Prior Approaches**: 기존의 2D X-ray에서 3D CT를 만드는 방식은 보통 geometry-agnostic feature lifting에 의존해 2D 특징을 3D로 단순 투영하거나 복제한다. 이 접근은 촬영(획득) 기하를 충분히 반영하지 못해 깊이 모호성이 커지고, 그 결과 뼈 봉합선·천문 등 소아 두개골의 미세 구조 경계가 흐려지거나 잘못 생성되기 쉽다. 또한 diffusion 기반 방법은 생성 품질은 개선해도 반복적 denoising으로 인해 임상 workflow에 비싼 계산 비용이 걸린다.

- **Core Contribution**: 본 논문은 소아 두개골 X-ray→CT 재구성 문제를 “획득 기하를 네트워크에 명시적으로 넣는” 방식으로 재정의한 PSCT-Net을 제안한다. 핵심은 differentiable back-projection으로 공간적으로 충실한 volumetric prior를 만들어 깊이 모호성을 초기부터 완화하고, 이후 Attention-Guided Projection(AGP-3D)로 2D 영역과 3D 복셀 간 비선형 대응을 학습한다. 전역 문맥은 Bidirectional Mamba(BiM-3D)로 장거리 의존을 선형 복잡도로 모델링해 단일-스텝 구조의 효율도 유지한다.

- **Technical Challenges**: 2D 투영 특징을 3D로 올릴 때 가장 큰 문제는 깊이 모호성과 구조 정렬 실패로 인한 해부학적 hallucination이다. PSCT-Net은 이를 위해 differentiable back-projection을 통해 물리적 ray 경로를 따라 초기 볼륨을 구성하고, 추가로 encoder/decoder 양쪽에서 기하 일관성을 주입하는 dual conditioning(BP-C, MV3D-C)을 적용한다. 또 고정된 선형 투영의 한계를 줄이기 위해 AGP-3D가 3D voxel grid을 query로, 2D feature를 key로 삼는 attention 기반 대응 학습을 수행하며, BiM-3D로 전역 문맥을 효율적으로 결합한다.

- **Empirical Impact**: 저자들은 성인·흉부/척추 중심 공개 데이터의 한계를 보완하기 위해 소아 두개골 전용 사설 코호트 PedSkull-CT(982 scans)를 구축해 내부 검증을 수행한다. PSCT-Net은 공개 3개 벤치마크(LIDC-IDRI, CTSpine1K, CTPelvic1K)에서 diffusion 기반 대비 PSNR/품질이 개선됐고, 사설 PedSkull-CT에서도 모든 베이스라인을 상회하며 두 번째 모델 대비 PSNR 1.28 dB, SSIM 0.022, LPIPS 0.013 개선을 보였다. 또한 학습에 없던 실제 임상 X-ray에 대해서도 턱뼈 곡률·안와 소켓 깊이 같은 환자 특이 형태를 보존해 저용량 소아 CT 재구성의 임상 적용 가능성을 시사한다.



### ViCoStream: Streaming VideoLLMs Can Run Beyond 100 FPS with Stage-Wise Coordinated Inferenc (https://arxiv.org/abs/2606.19849)
Comments:
          19 pages, 7 figures, 13 tables

- **Prior Approaches**: 기존 streaming VideoLLM은 두 흐름으로 나뉘는데, 먼저 delayed streaming은 프레임을 모아두었다가 질의 시점에 한 번에 prefill해 질의 지연과 GPU 메모리 사용이 스트림 길이에 따라 커집니다. 연속형 continuous streaming은 visual encoding, token pruning, KV-cache compression 같은 단일 모듈 최적화로 효율을 높이지만, 전체 파이프라인 처리량(FPS)이 결국 다른 단계 병목으로 이동해 지속적인 실시간 성능을 보장하기 어렵습니다. 특히 token dropping만으로는 LLM 계산이 줄어도 TTFT와 처리 지연이 누적되어 장기 스트림에서 한계가 생깁니다.

- **Core Contribution**: 이 논문은 streaming VideoLLM 추론을 시각 전처리-시각 인코딩-토큰 드롭-LLM prefilling/decoding까지 이어지는 stage-wise 파이프라인으로 공식화하고, 영상 입력 처리량(FPS)과 질의 첫 토큰 지연(TTFT)을 동시에 최적화하는 관점을 제시합니다. 그 위에 ViCoStream(Video Coordinated Streaming)이라는 단계별 조율 프레임워크를 제안해 chunk-wise 실행, CUDA-stream overlap, 시각 토큰 제어, bounded visual attention, query-side retrieval을 함께 묶습니다. 이로써 스트림 길이와 무관하게 청크당 연산·메모리 비용을 상한으로 고정해, 긴 스트림에서도 실시간 상호작용이 가능해지도록 설계합니다.

- **Technical Challenges**: 핵심 기술 과제는 전체 시스템이 ‘어느 단계가’ 어느 조건에서 병목이 되는지 파악하고, 한 모듈 최적화가 다른 단계 지연으로 전이되는 문제를 제어하는 것입니다. ViCoStream은 (1) GPU-side 전처리를 별도 CUDA stream으로 분리해 겹침 처리하고, (2) frame-by-frame 대신 chunk-wise visual encoding으로 배치 효율을 높이며, (3) projector 단계에서 temporal/spatial token dropping으로 LLM prefill 부담을 줄이고, (4) LLM에는 vision-side local attention(최근 acn 청크만)과 query-side retrieval(top-k 청크만)을 결합해 시각 캐시 가시 범위를 bounded로 유지합니다. 또한 chunk 크기, 토큰 보유율, attention locality, retrieval scope가 throughput-accuracy trade-off를 어떻게 바꾸는지 병목 이동(bottleneck migration) 관점에서 체계적으로 분석합니다.

- **Empirical Impact**: Qwen2.5-VL-3B/7B-Instruct를 Qwen2.5-VL 계열 streaming 벤치마크 6종에서 평가한 결과, ViCoStream은 단일 A100 GPU에서 134 FPS의 영상 처리량을 달성하면서 TTFT를 50ms 미만으로 유지하고도 정확도를 full-history 기준선에 가깝게 보존합니다. 또한 bounded attention으로 LLM prefilling의 히스토리 의존 비용이 고정되며, retrieval 범위와 토큰 보유율을 함께 조절할 때 TTFT 상승이 급격히 완화되는 패턴을 관측합니다. 즉, 모듈 단위 가속을 넘어 ‘지속 가능한 실시간 스트리밍’ 조건을 설계 변수로 정량화했다는 점에서 VideoLLM 시스템 연구에 실용적 지침을 제공합니다.



### OTCHA: Optimal Transport-driven Confidence-aware Latent Hub Alignment for Multi-View Medical Image Classification (https://arxiv.org/abs/2606.19838)
Comments:
          Accepted at MICCAI 2026

- **Prior Approaches**: 기존 다중 뷰 의료영상 방법들은 각 뷰의 임베딩을 그대로 합치거나(cast/concatenate/pooling) cross-view attention으로 토큰을 교환하는 방식이 주류입니다. 하지만 임의 등록(unregistered)과 뷰별 아티팩트/배경이 섞인 상황에서, 신뢰도나 부분 매칭을 명시적으로 제어하지 못해 불필요한 토큰이 융합 표현을 오염시킬 수 있다는 한계가 있었습니다. OT 기반 대응학습은 있었지만, 토큰 수준 confidence와 partial matching을 분명히 활용한 다중 뷰 분류 설계는 상대적으로 덜 탐구돼 왔습니다.

- **Core Contribution**: 논문은 OTCHA(Optimal Transport Confidence-aware Latent Hub Alignment)로, 뷰를 합치기 전에 토큰을 먼저 정제(refinement)하는 접근을 제안합니다. 학습 가능한 latent hub tokens를 뷰 전반에서 공유하는 ‘중앙 허브’로 두고, 각 뷰의 patch tokens를 허브에 OT로 정렬한 뒤 hub가 다시 뷰로 정보를 브로드캐스트해 refined tokens를 만들며 최종 fusion을 수행합니다. 또한 token-conditional dustbins로 부분 매칭을 가능하게 해 배경/아티팩트 토큰을 매칭에서 배제하도록 설계한 점이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술 난제는 (1) 등록이 불완전한 토큰 간 대응을 안정적으로 찾으면서 (2) 불필요 토큰을 ‘버릴 수 있는’ 학습 신호를 주고 (3) hub 기반 메시지 패싱이 훈련 중 붕괴하지 않게 만드는 것입니다. OTCHA는 feature similarity와 geometry-aware prior를 결합한 fused optimal transport(enturopic Sinkhorn)로 hub-토큰 간 매칭을 계산하고, token-conditional dustbin으로 토큰별 unmatched 질량을 제어해 부분 매칭을 구현합니다. 더 나아가 OT에서 얻는 토큰 수준 matching confidence를 gate로 메시지 패싱에 반영하고, confidence-weighted OTRA 손실(gradient stop 포함)로 hub 매칭-정제 과정을 학습 안정화했습니다.

- **Empirical Impact**: VinDr-Mammo, MURA, CheXpert의 3개 다중 뷰 데이터셋에서 OTCHA는 다양한 해부학/뷰 구성에 걸쳐 경쟁 baseline을 일관되게 능가했으며, 매개변수 오버헤드는 크지 않았다고 보고됩니다. 특히 VinDr-Mammo study-level(4-view) 설정에서 개선 폭이 크게 나타나 여러 뷰 정보를 통합할수록 효과가 커지는 경향을 보여줍니다. 정성 결과에서는 hub-기반 OT 정제로 배경 활성은 억제되고 병변 부위에 집중이 강화되며, CC와 MLO 사이의 의미 있는 토큰 대응도 해석 가능하게 제시되어 임상적 cross-referencing 직관과 맞닿는 의미가 있습니다.



### Neural Events: Discrete Asynchronous Autoencoders for Event-Based Vision (https://arxiv.org/abs/2606.19835)
- **Prior Approaches**: 이전 연구들은 이벤트를 동기식(밀집 CNN/Transformer)이나 비동기식(SNN의 spike train, GNN의 spatio-temporal graph)으로 표현해 왔습니다. 하지만 동기식은 정밀 타이밍·희소성을 잃어 저장/지연/연산이 비효율적이고, 비동기식은 고이벤트레이트에서 확장이 어렵거나(연산·메모리) 성능이 밀집 베이스라인에 못 미친다는 문제가 있었습니다.

- **Core Contribution**: 논문은 원시 이벤트 스트림을 소수의 정보가 큰 neural events로 재토큰화하는 프레임워크를 제안합니다. 각 neural event는 로컬 시공간 컨텍스트를 요약하고, 코드북의 이산 learnable code에 의해 “코드가 뒤집힐 때”만 트리거되어 이벤트레이트를 압축합니다. 또한 이런 코드 기반 토큰화가 SNN의 아날로그 membrane 잠재치에 비해 학습 시 시간·메모리 복잡도를 낮춘다고 주장합니다.

- **Technical Challenges**: 핵심 난제는 (1) 원시 이벤트의 타이밍/공간 정보를 충분히 유지하면서 (2) 코드 플리핑으로 인한 불필요한 spiking(chatter)을 억제하고 (3) 학습/추론에서 대규모 스트림을 스케일 가능하게 처리하는 것입니다. 저자들은 패치 단위 비동기 인코더(선형 시간 RWKV-7 계열)로 로컬 메모리를 갱신하고, discrete variational autoencoder 기반 양자화로 이산 코드에 투영한 뒤, reconstruction과 rate alignment/latent straightening 같은 비지도 pre-training 목적을 섞어 코드 경계에서의 잦은 뒤집힘을 안정화합니다.

- **Empirical Impact**: 실험에서는 object detection과 classification에서 neural events 기반 네트워크가 기존 state-of-the-art와 동급이거나 더 높은 성능을 내면서 이벤트레이트를 2.0배 줄였다고 보고합니다. 특히 DSEC-Detection에서는 TokDAGr가 DAGr 대비 mAP를 9.0p 개선하면서 연산·에너지도 낮춰, 이미지 없이도 이미지 기반 접근에 근접하는 결과를 제시합니다. 요약하면, 정보량은 유지하면서도 downstream 부하를 크게 줄이는 “학습된 의미 압축”의 실효성을 여러 벤치마크에서 확인한 셈입니다.



### 3D-PLOT-LLM: Part-Level Object Tokens for 3D Large Language Models (https://arxiv.org/abs/2606.19828)
- **Prior Approaches**: 기존 3D MLLM은 점을 patch 토큰의 나열로만 보고, 의자·로봇 같은 물체는 서술은 가능해도 등받이·다리처럼 ‘부분’을 지칭하거나 합성해 추론하기 어렵습니다. 부분을 다루려는 시도는 segmentation decoder, 더 무거운 3D 인코더, bounding-box 문법 같은 외부 모듈을 추가하며 파라미터 예산을 크게 늘리지만, LLM의 입력/출력 어휘에는 부분이 일급(vocabulary-level)으로 들어오지 못합니다. 이 때문에 부분은 외부 모듈의 출력물로 남아 표현 병목이 지속됩니다.

- **Core Contribution**: 이 논문은 부분을 LLM이 읽고 내보내고(reason)할 수 있도록, 입력 토큰 스트림 자체를 재구성해 ‘부분 주소(addressing)’를 vocabulary 수준 인터페이스로 만든 3D-PLOT-LLM을 제안합니다. frozen point 인코더의 patch를 K개의 지역으로 나눈 뒤 각 지역 앞에 learnable marker와 <part_k> 토큰을 삽입해, 모델이 프롬프트에서 부분을 토큰으로 지시하고 출력에서도 해당 토큰을 인용하게 합니다. Marker-Space Refinement(MSR)은 지역의 공간 통계와 인접 관계를 marker에 반영해 <part_k>와 지역 내용의 결합을 강화합니다.

- **Technical Challenges**: 핵심 과제는 (1) 정답용 canonical part 분해가 없으면서도 (2) inference 때마다 지역 인덱스가 안정적으로 대응되어야 하고 (3) ‘부분’의 구조적 문맥이 LLM attention에 충분히 주입돼야 한다는 점입니다. 저자들은 의미 라벨에 의존하지 않고, frozen 인코더 feature와 좌표 기반의 unsupervised, deterministic feature-aware region-growing으로 지역을 고정 분할하고, per-region marker를 MSR에서 adjacency/통계를 메시지 전달 형태로 보정합니다. 또한 1M 미만 규모(새로 학습하는 파라미터)로 목표를 달성하기 위해 segmentation decoder나 bounding-box head 없이 토큰 인터페이스만 학습하도록 설계했습니다.

- **Empirical Impact**: PartVerse-QA(캡션→슬롯 C2S, 슬롯→캡션 S2C)에서 3D-PLOT-LLM은 caption-to-slots Jaccard 0.459, Exact-match 13.78%를 달성했으며, 슬롯→캡션은 GPT-4o judge 44.68을 기록해 vocabulary 단의 부분 질의 응답 능력을 실증했습니다. 3DCoMPaT-GrIn grounded description에서도 PointLLM, Kestrel, PARIS3D, SegPoint를 텍스트 출력 전 지표에서 앞섰고 일부 설정에서는 ShapeLLM 대비도 우세했으며, Objaverse 전체 물체 캡셔닝에서는 Stage 2에 PartVerse-QA를 더했을 때 기준선 대비 SBERT와 GPT-4o 평가가 각각 +0.65, +1.85 개선되는 등 일반화 이득이 확인됐습니다. 무엇보다 frozen point encoder 하에서 1M 미만 신규 파라미터, segmentation decoder/박스 헤드 없이도 성능을 끌어올렸다는 점에서 3D MLLM의 ‘부분 지칭’ 인터페이스 설계 방향을 제시합니다.



### CSWinUNETR: Segmentation of Thin Anatomical Structures in Medical Images (https://arxiv.org/abs/2606.19824)
Comments:
          Accepted at MICCAI 2026

- **Prior Approaches**: 기존 얇고 휘어진 해부학적 구조(망막 혈관, 뇌혈관, 얼굴 주름) 분할은 낮은 대비, 잦은 단절, 심한 클래스 불균형 때문에 예측이 조각나고 미세 분기가 복원되지 않는 문제가 컸다. 최근 CNN/Transformer가 성능을 올렸지만, 과도한 다운샘플링이나 태스크 전용 수정·손실·후처리에 의존하는 경우가 많아 다른 데이터로의 전이가 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 2D/3D 모두에 공용으로 쓰는 얇은 구조 분할 백본 CSWinUNETR를 제안한다. CSWin의 cross-shaped stripe self-attention으로 주축 방향의 장거리 문맥을 효율적으로 전파하고, detail-enhanced multi-scale self-attention으로 고주파 단서를 보존하며, SDSConv로 곡선 궤적을 따라 국소 증거를 안정적으로 집계한다.

- **Technical Challenges**: 핵심 기술 도전은 (1) 단절된 얇은 구조를 연결하기 위한 장거리 정보 모델링과 (2) 분기/말단의 미세 디테일을 유지하는 고해상도 보존, (3) 곡선 경로를 따르는 동안 로컬 오류가 누적되지 않게 하는 곡률 정렬이었다. 저자들은 cyclic shift로 stripe 간 정보 교환을 늘리고 MS-MHSA에서 고주파 detail branch 및 다중 해상도 문맥을 융합했으며, SDSConv에서는 sparse control points로부터 snake-like 커널을 “누적 드리프트 없이” 병렬 생성해 궤적 정합성을 높였다.

- **Empirical Impact**: CSWinUNETR는 안과(FIVES), 신경혈관(TopCoW), 피부(FFHQ-Wrinkle) 등 4개 벤치마크에서 기존 SOTA 대비 일관된 개선을 보였고, 대부분 지표에서 통계적으로 유의한 성능 향상이 보고됐다. 특히 task-specific post-processing이나 topology-aware loss 없이도 미세 분기와 말단을 더 잘 복원하고 비표적 영역의 오탐을 줄여, 임상적/후속 분석(센터라인 기반 길이·분지·곡률) 요구에 대응하는 범용 백본으로 의미가 크다.



### Training-Free Metrics for Synthetic Object Detection Data: A Proxy for Detector Performanc (https://arxiv.org/abs/2606.19817)
Comments:
          9 pages, 4 figures

- **Prior Approaches**: 합성 데이터는 bbox 같은 밀집 주석이 필요한 object detection에서 유용하지만, 모든 합성이 성능을 올리진 않아 “어떤 합성이 좋은지”를 빠르게 가려내는 것이 문제로 지적된다. 기존에는 FID/KID/MMD 등 feature-space 분포 정렬 기반 metric으로 생성-실제 간 거리를 보거나, 실제 다운스트림 모델을 학습해 mAP 향상을 확인하는 방식이 주류였다. 하지만 이러한 metric들은 이미지 모양/분포(appearance, distribution)에 편향되며, 객체 수·박스 스케일 같은 메타데이터에서 비롯되는 조합(composition) 변화를 충분히 반영하지 못한다고 분석된다.

- **Core Contribution**: 이 논문은 합성 학습셋의 “다운스트림 유틸리티”를 대리(proxy)하는 training-free metric 계열 Conditional-Composition Domain Match(CCDM)를 제안한다. CCDM은 per-image metadata로 이미지를 strata로 나눈 뒤, (1) 같은 조합 조건 내에서의 외양(appearance) 일치와 (2) strata 간 조합 불일치(composition divergence)를 함께 결합해, 단순 전역 정렬이 놓치는 조합 축을 직접 다룬다. 그 결과 candidate 합성셋들에 대한 순위가 detection mAP 순위와 맞도록 설계되었다.

- **Technical Challenges**: 핵심 난제는 FID/KID/MMD 같은 기존 거리함수를 그대로 쓰면 메타데이터 기반 조합 변화가 잘 반영되지 않는다는 점이다. 이를 해결하기 위해 CCDM은 어떤 base feature-distance라도 CCDM 방식으로 “조건부(stratified)로 감싸기(wrapping)”만 하면 되도록 구성하고, strata가 샘플이 너무 적을 때는 전역(global) 항으로 폴백해 추정 안정성을 확보한다. 또한 strata 키를 object count·bbox scale뿐 아니라 dominant class까지 확장하는 class-aware 정책도 실험했지만, 데이터 밀도 한계로 노이즈가 커질 수 있음을 보여준다.

- **Empirical Impact**: VisDrone-DET에서 YOLOv8 mAP를 기준으로 평가한 결과, CCDM-MMD_CLIP 계열이 5개 candidate(실데이터 1 + 합성 4)를 mAP와 정확히 일치된 순위로 정렬하며 Spearman ρ=+1.0을 달성했다. 이는 FID/KID/원본 MMD 등 기존 classical training-free metric 대비 명확한 개선이며, CCDM-shared의 성능이 CCDM-classaware보다 더 신뢰성 있게 나온 점도 함께 확인됐다. 저자들은 향후 RT-DETR, two-stage detector, COCO/LVIS 같은 더 큰 벤치마크 및 detection을 넘어 instance/panoptic segmentation 등 “조합이 성능을 좌우하는” 태스크로 확장을 예고했다.



### ParaScale: Scale-Calibrated Camera-Motion Transfer via a Gauge-Invariant Parallax Number (https://arxiv.org/abs/2606.19805)
Comments:
          Accepted by SCA2026(poster)

- **Prior Approaches**: 기존 참조-기반 카메라 제어는 참조 영상에서 추출한 카메라 궤적(회전+이동)을 그대로 생성 모델에 주입하는 방식이 일반적입니다. 그러나 참조와 타깃은 장면 스케일이 수십~수만 배까지 달라질 수 있어, 이동 성분을 그대로 재생하면 너무 작거나(거의 안 움직임) 너무 과장되며(프레임을 벗어남) 제어가 실패합니다. 일부 방법은 학습 시 스케일 정규화나 attention 공간 치환을 시도하지만, 추론 시점에 임의의 참조-타깃 사이를 ‘스케일 캘리브레이션’하는 개념적 근거를 직접 다루지는 못했습니다.

- **Core Contribution**: 이 논문은 ‘번역(translation)에 의한 이미지 모션’이 ||T||/Z 비율로 스케일에 좌우된다는 기하학 사실을 정리하고, 단안(monocular) 궤적은 깊이 스케일 게이지 때문에 절대 이동량 자체가 의미 없다고 지적합니다. 대신 깊이-스케일 게이지 불변량인 Parallax Number Pi = ||Delta T|| / Zbar를 정의하고, 스케일-faithful(스케일-충실) 전이는 궤적 전체가 아니라 이 Pi를 보존해야 한다는 것을 증명합니다. 이를 위해 ParaScale(Parallax Number 기반 보정 모듈)은 회전은 그대로 두고, 프레임별로 Pi만 타깃의 자체 깊이에 재현하여 학습 없이 어떤 pose-conditioned generator에도 끼워 넣을 수 있게 합니다.

- **Technical Challenges**: 핵심 기술 난제는 ‘단안 SfM/깊이 추정’이 절대 스케일을 알 수 없는데도, 생성에서 실제로 느껴지는 번역 패럴랙스를 정확히 맞춰야 한다는 점입니다. 논문은 translation-로 인한 felt motion이 Z에 의해 스케일되므로, raw T를 쓰면 참조-타깃 스케일 불일치가 그대로 오차로 증폭된다는 로그-스케일 관점의 정량식을 제시합니다. 해결책은 참조에서 Pi를 스케일 무관하게 읽어내고, 타깃의 깊이 통계 Zbar로 프레임별 translational gain을 계산해 Pi를 그대로 강제하는 것으로, 추가 학습 없이 pose injection 파이프라인 사이에 한 단계만 삽입합니다.

- **Empirical Impact**: 실험에서는 타깃 장면 스케일을 tabletop부터 human/room, architectural, cosmic 수준까지 4자릿수 차이로 나누고, 여러 pose-conditioned 백본에서 동일 효과를 확인했습니다. ParaScale은 Raw transfer 대비 PCE(Parallax Consistency Error)를 3배 이상 줄이며, 회전 오차는 거의 변하지 않으면서 이동/패럴랙스의 스케일 불일치만 정확히 교정됨을 보여줍니다. 또한 Similiarity-aligned TransErr가 놓치는 상수 스케일 미스매치까지 PCE가 드러내며, 생성 품질 저하 없이 identity line에 가깝게 패럴랙스가 유지되는 점에서 실무적 의미가 큽니다.



### HypOProto: Hyperbolic Ordinal Prototypes for Left Ventricular Filling Pressure Classification (https://arxiv.org/abs/2606.19804)
- **Prior Approaches**: 기존 LVFP(좌심실 충만압) 분류는 Doppler 기반 E/e'로 normal vs elevated를 나누지만, 이 과정은 검사자 의존성과 영상획득 조건에 민감해 point-of-care나 자원 제한 환경에서 적용이 어렵다. B-mode echo로 직접 추정하는 deep learning은 성능은 높아도 주로 black-box여서 임상 해석이 제한되며, LVFP에 대해 명시적으로 prototype 기반 설명가능 프레임을 다룬 연구도 거의 없었다. 또한 label이 적은 문제에서 end-to-end 학습 중심 접근은 확장성이 떨어져 foundation model을 쓰더라도 구조가 여전히 설명에 불리한 경우가 있었다.

- **Core Contribution**: HypOProto는 B-mode echo에서 LVFP를 ordinal(순서형) 구조로 분류하기 위해 hyperbolic, ordinal prototype 기반 프레임을 제안한다. frozen DINOv3 기반 특징 추출 위에, 생리적 E/e' 스케일을 hyperbolic space의 반지름·방향으로 인코딩해 경계( cutoff ≈ 14 ) 근처의 애매한 케이스를 hyperboloid root에 배치하고 정상/고위험 케이스는 바깥으로 보내도록 설계했다. 그 결과 prototype이 임상적으로 의미 있는 순서 관계를 갖고, 해석 가능한 시각화로 연결된다.

- **Technical Challenges**: 핵심 난제는 (1) Doppler 없이도 LVFP의 연속값(ordinal 정보)을 시각 특징에서 재구성하고, (2) 경계 주변에서 미세한 구분이 일어나는 상황에서 prototype 방향을 안정적으로 분리하며, (3) backbone을 고정한 상태에서도 설명가능성을 유지하는 것이다. HypOProto는 feature와 prototype을 Lorentzian hyperbolic space에 투영하고, 예측된 E/e' 값을 기준선에 대한 hyperbolic radius로 매핑해 경계 근처의 민감도를 높였다. 추가로 Hyperbolic Prototype Angular Separation(HyperPAS) loss로 하이퍼볼릭 공간에서 클래스 간 prototype 방향 분리를 강제해, 같은 클래스는 정렬되고 다른 클래스는 떨어지도록 학습시킨다.

- **Empirical Impact**: 사설 echo 데이터셋(AP4, 총 141,086 cines)에서 HypOProto는 cine 및 study 수준 모두에서 최고 성능을 보였고, 특히 elevated LVFP의 F1(0.63) 향상이 두드러졌다. study 수준에서는 normal과 elevated 모두에 대해 클래스별 F1이 가장 좋거나 상위권이며, EchoPrime 또는 Akerman et al. 대비 해석가능성을 유지하면서도 학습 부담을 줄인 점이 강조된다. 시각화 결과 prototype activation map이 승모판 유입부에 집중되고, CoSNE 등 임베딩에서도 E/e' 기반 ordinal 구조가 잘 반영되어 정상/고위험이 더 분리된다는 점이 확인됐다.



### Occ-VLM: Occupancy Grounded Vision Language Model for Indoor Scene Understanding (https://arxiv.org/abs/2606.19776)
- **Prior Approaches**: 기존 3D VLM은 포인트 클라우드처럼 명시적 3D 입력을 쓰거나, RGB-D의 깊이를 이용해 3D로 토큰을 정렬하는 방식이 주류였습니다. 또 RGB-only를 시도한 모델들은 3D 지오메트리 인코더/듀얼-인코더로 기하를 별도 추출해 2D 의미와 구조가 구조적으로 분리되는 문제가 있습니다. 이로 인해 통합 3D 비전-언어 표현 학습이 어렵고, 아키텍처 복잡도나 센서 의존성도 남습니다.

- **Core Contribution**: Occ-VLM은 posed RGB 멀티뷰 이미지만 입력으로 받아 단 하나의 2D 비전 인코더로 3D 장면 이해를 수행하는 프레임워크를 제안합니다. 핵심은 3D 점유(occupancy)를 기하 보조 priors로 복원해, 2D 전경 토큰을 3D 공간에 공간적으로 associate하고 LLM이 이를 바탕으로 장면을 추론한다는 점입니다. 즉, occupancy 예측으로 기하를 얻는 동시에 2D 사전학습 의미를 3D로 끌어올리는 양방향(2D-3D bridge) 관점을 채택합니다.

- **Technical Challenges**: 두 가지 난제가 있습니다: (1) 2D 토큰을 구조적 3D 공간으로 “올리는” 변환을 단일 인코더 체계에서 안정적으로 학습해야 하고, (2) 이렇게 만든 3D-anchored 토큰이 LLM 추론에 실제로 유효해야 합니다. 논문은 2D 인코더는 frozen으로 두고 중간층을 복제해 Occ adapter로 점유 격자(3D semantic occupancy)를 예측한 뒤, 점유 마스크에 기반해 점유된 3D 좌표에 대응되는 foreground 토큰만 샘플링합니다. 또한 3D positional encoding을 토큰에 주입해 LLM이 정밀한 공간 앵커를 갖고 추론하도록 했고, occupancy reasoning과 언어 응답 생성을 잇는 two-stage 학습(오프라인 점유 학습→시각 instruction tuning)을 사용합니다.

- **Empirical Impact**: 실험에서 Occ-VLM은 EmbodiedScan의 multi-view 3D semantic occupancy prediction에서 mIoU 18.41%로 RGB-only/point 기반 경쟁군을 모두 제치며 SOTA를 달성합니다. 3D VQA(SCANQA, SQA3D)와 3D dense captioning(Scan2Cap)에서도 2D-input 기반 SOTA 수준 또는 그에 준하는 성능을 보이며, 일부 항목에서는 명시적 3D 입력 모델과의 격차가 매우 작거나 거의 유사합니다. 요약하면, RGB만으로도 정확한 3D 기하 인식과 견고한 vision-language 공간 추론을 함께 달성해 “통합 3D VLM” 방향성을 실증한 결과로 해석됩니다.



### VFACamou: View-Fused Adversarial Camouflage for Environment-Adaptive Physical Evasion (https://arxiv.org/abs/2606.19736)
Comments:
          Accepted by ICME 2026

- **Prior Approaches**: 기존 연구는 주로 2D 디지털 perturbation이나 고정된 관측 조건에서의 텍스처/패치 공격에 집중해, UAV가 움직이며 발생하는 연속적인 스케일·시점 변화에 취약했다. 또한 실환경의 조명(강도, 색온도, 방향) 변동을 충분히 반영하지 못해 공격 성능과 자연스러움이 함께 무너지는 문제가 있었다. 마지막으로 고주파 아티팩트나 주변과 불일치하는 색을 만들어내는 경우가 많아 사람에게도 쉽게 들켜 실제 배치가 어려웠다.

- **Core Contribution**: 논문은 UAV 정찰 환경에서 ‘착용 가능한 adversarial camouflage(대항 카모플라주)’를 end-to-end로 생성하는 VFACamou를 제안한다. UV-volume rendering과 diffusion 기반 텍스처 생성기를 결합해 자세·스케일이 달라져도 외형 일관성을 유지하고, 주변 환경에 자연스럽게 섞이도록 학습 목표를 설계한다. 동시에 detector를 사람 영역 단순 confidence 저하가 아니라 공간적 탐지 구조(IoU)를 교란하도록 최적화한다.

- **Technical Challenges**: 핵심 난제는 (1) UAV로 인한 viewpoint·scale·인체 변형이 계속 변하는 상황에서도 공격이 안정적으로 유지되는 것과 (2) 강한 공격성은 확보하되 색/패턴 자연스러움을 동시에 달성하는 균형이다. VFACamou는 조명 추정기로 배경의 dominant 색/조명을 추출해 illumination–color consistency를 맞추고, multi-scale dynamic training으로 거리·시점·자세 변화를 학습 중에 다양화해 강인성을 높였다. 또한 IoU-guided detection-evasion loss와 color-pattern adaptation loss를 함께 최적화해 탐지 회피와 환경 융합을 동시에 추진한다.

- **Empirical Impact**: 디지털 실험에서 8개 mainstream detector에 대해 ASR이 지속적으로 높게 나타나며, white-box는 물론 cross-model transfer에서도 기존 방법을 앞섰다. 특히 IoU 임계값이 바뀌어도 성능 변동이 상대적으로 작아, 공격 텍스처가 더 robust하고 일관된 동작을 보였다. 더 나아가 티셔츠/바지에 출력해 드론으로 촬영한 physical 실험에서도 다중 자세·시점·거리 변화 하에서 person 미탐지 프레임이 크게 나타나며, 고의미 없는 시각적 이상 없이 자연스러움을 유지하는 점을 함께 강조한다.



### QueryGaussian: Scalable and Training-Free Open-Vocabulary 3D Instance Retrieva (https://arxiv.org/abs/2606.19733)
Comments:
          8 pages, 4 figures, 6 tables. Accepted to the 2026 IEEE International Conference on Systems, Man, and Cybernetics (SMC 2026)

- **Prior Approaches**: 기존 open-vocabulary 3D instance retrieval은 주로 scene-level embedding 방식에 의존합니다. 텍스트-의미를 먼저 장면 전체의 3D Gaussian에 “증류”해야 해서 전처리/재학습 비용이 크고, 장면 복잡도가 커질수록 메모리가 선형으로 증가해 city-scale에서 OOM이 자주 발생합니다. 또한 2D-3D 매핑이 장면 전체 임베딩에 결합되어 있어 vocabulary나 장면이 바뀌면 다시 구축해야 하는 제약도 있습니다.

- **Core Contribution**: QueryGaussian은 training-free로 작동하는 on-demand 인스턴스 검색 프레임워크를 제안합니다. 장면 전체를 의미 특징으로 압축하는 대신, 미리 학습된 2D vision 모델로 텍스트에 해당하는 2D 마스크를 찾고 이를 3D로 “최대 기여도(max-weight)” 방식으로 직접 lift합니다. 즉 의미 이해와 기하 표현을 분리해, 기존 3DGS 체크포인트 그대로부터 zero-shot 검색을 가능하게 합니다.

- **Technical Challenges**: 핵심 난제는 2D 마스크를 3D Gaussian으로 올릴 때의 projection ambiguity와 occlusion/복원 잡음입니다. QueryGaussian은 rasterization 중 생성되는 maximum-weight map으로 픽셀을 가장 강하게 기여하는 Gaussian에 매핑해 반투명 아티팩트 등 불필요 후보를 줄입니다. 이어서 multi-view temporal fusion으로 시점 간 일관성이 낮은 outlier를 누르고, 이후 multi-stage density clustering(단계적 DBSCAN 변형)으로 인스턴스 경계를 더 정밀하게 다듬습니다.

- **Empirical Impact**: 실험에서 QueryGaussian은 소형 실내와 대형 실외(각각 수백만~수천만/1,000만+ Gaussians)에서 경쟁 또는 우수한 mIoU를 보였고, 특히 대형 설정에서는 scene-level embedding 계열이 24GB GPU에서 OOM으로 실패한 반면 동작을 유지했습니다. 효율 측면에서도 GPU 메모리 사용량을 70% 이상 줄이고, end-to-end 추론을 최대 180x 가속해 소비자급 하드웨어로 city-scale 인스턴스 검색을 실현합니다. 또한 검색 결과(마스크/3D 중심)를 LLM 에이전트의 인스턴스 테이블로 연결해 3D question-answering 같은 응용까지 시연했습니다.



### One-Shot Novel View and Pose Human Image Synthesis via 3D Prior Guided Diffusion Mod (https://arxiv.org/abs/2606.19718)
Comments:
          30 pages, 10 figures

- **Prior Approaches**: 기존 연구는 단일 이미지에서 사람의 새로운 시점·자세를 합성할 때, (1) 2D pose keypoints를 조건으로 삼아 레퍼런스 인물을 타깃 자세로 옮기는 pose transfer 방식과 (2) generalizable human NeRF로 보이는 부분은 잘 맞추되 가려진/보이지 않는 부분은 특징 정합이 깨져 흐려지는 문제를 겪는 NeRF 기반 방식으로 크게 나뉩니다. 특히 2D pose 조건은 복잡한 self-occlusion 상황에서 모호해져 자세 정보가 충분히 전달되지 않고, NeRF 계열은 단일 이미지에서 신뢰도 높은 점 단위 특징을 얻기 어려워 누락이 발생합니다. 또한 GAN·초기 diffusion 기반 pose transfer는 단계가 복잡하거나 고주파 디테일 보존이 약한 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 single human image로 novel view와 pose를 one-shot으로 합성하는 conditional denoising diffusion model을 제안하고, 복잡한 자세 합성을 diffusion의 연속적인 조건부 denoising 단계로 분해해 품질을 끌어올립니다. 핵심은 2D pose만 쓰지 않고 3D human priors인 3D normal map(형상·기하 조건)과 color prompt(외관·텍스처 조건)를 추가해 참조 인물과 타깃 인물 사이의 대응과 색 일관성을 명시적으로 강화한 것입니다. 여기에 테스트 시점에서 self-reconstruction 기반 customized refinement로 새 인물에 맞춰 생성기를 미세조정해 국소 디테일(얼굴·의상 무늬 등) 손실과 일반화 격차를 줄입니다.

- **Technical Challenges**: 주요 기술 난제는 (a) 단일 이미지에서 가려진 부위까지 포함해 복잡한 pose를 정확히 조건화하는 것과 (b) 참조 인물의 고주파 텍스처를 타깃 시점·자세로 일관되게 옮기는 것입니다. 논문은 3D normal map으로 깊이·형상 신호를 보강해 모호한 2D pose의 정보 부족을 메우고, cross-attention으로 noise 특징과 reference 특징을 3D 조건 위에서 정합시키며, color prompt는 SMPL 메쉬 기반의 explicit pixel alignment로 타깃 뷰에서의 색 단서를 생성 과정에 직접 주입합니다. 마지막으로 customized refinement에서 입력 레퍼런스를 자기 재구성 목표로 삼아 단 한 개 샘플만으로도 디테일을 보정하도록 설계합니다.

- **Empirical Impact**: RenderPeople과 THuman 등 공개 데이터셋에서 기존 방법 대비 유의미한 성능 향상과 함께, 데이터셋 간 일반화 능력도 더 우수함을 보여줍니다. 특히 occluded/invisible parts까지 포함해 전신 합성의 누락·흐림을 줄였고, refined 결과에서는 얼굴·의상 라벨·소품 같은 국소 고주파 디테일 복원이 개선되는 정성적 사례가 제시됩니다. 결과적으로 VR/AR 파이프라인 및 single-image 3D 인체 복원 같은 downstream 작업에서 활용 가능한 고품질 one-shot 인물 합성의 실용성에 힘을 보태는 연구로 평가됩니다.



### NEST: Narrative Event Structures in Time for Long Video Understanding (https://arxiv.org/abs/2606.19706)
- **Prior Approaches**: 기존 비디오-언어 모델은 긴 토큰을 처리해도, 장편 영상의 ‘서사 구조’는 추론하기 어렵다는 한계가 있었다. 벤치마크 역시 짧은 클립의 원자 행동 이해나 needle-in-a-haystack 검색에 치우쳐, 사건(event)이 시간에 따라 어떻게 이벤트로 묶이고 관계 맺는지(사건 그래프, 인과, 계층, 플래시백) 평가는 부족했다. 일부 내러티브 이벤트 연구가 있었지만 대부분이 짧은 구간에 머물러, 수십 분~수시간 간격의 선행 사건을 연결하는 능력을 검증하지 못했다.

- **Core Contribution**: 논문은 장편 영화 전체를 대상으로 ‘서사 사건 구조’를 평가하는 NEST를 제안한다. NEST는 1005편(평균 98분) 영화에 대해 시각·대사·오디오에 근거한 102개의 멀티모달 내러티브 이벤트를 구조화해 주석하고, 시간 순서·계층적 구성·장거리 의존성을 반영하는 이벤트 관계까지 연결한다. 또한 ETD(이벤트 트리거 탐지)·EL(로컬라이즈)·EAE(인자 추출)·ERE(관계 추출) 멀티태스크를 도입해, 단순 retrieval이 아닌 서사 이해를 직접 겨냥한다.

- **Technical Challenges**: 기여를 현실화하기 위한 핵심 난제는 (1) 원시 비디오에서 서사 수준 이벤트를 찾아내고(grounded discovery), (2) 그 이벤트 사이의 관계를 멀리 떨어진 장면까지 일관되게 연결하는 것이다. 이를 위해 오디오 설명을 활용해 시각-행동과 정렬된 텍스트 신호로 이벤트 트리거/인자를 추출하고, LLM 기반 파이프라인과 PropBank 동사 어휘를 결합해 이벤트 온톨로지를 통일했다. 로컬라이제이션은 장면 경계(scene boundary) 수준에서 보수적으로 평가해 주관적 경계 문제와 단기 grounding의 불안정성을 완화했으며, 관계는 닫힌 집합 분류로 ERE를 안정적으로 측정했다.

- **Empirical Impact**: 실험 결과, 모델들은 후보 이벤트를 ‘찾는’ 능력에서 크게 무너진다: ETD는 8% 미만, EL은 6% 미만, EAE는 11% 미만에 머물렀다. 반면 이벤트가 주어졌을 때의 ERE는 상대적으로 가능해져 zero-shot F1 35.45%, fine-tuning 후 F1 44.42%까지 상승했으며, 이는 conditional reasoning과 grounded discovery가 별개 난제임을 보여준다. 또한 플래시백 관계 하위셋에서는 대부분 0에 가까운 성능 붕괴가 나타나, 비선형 시간 추론이 장편 서사 이해의 독립적 병목임이 강조된다.



### Exploring Multi-Modal Large Language Models and Two-Stage Fine-Tuning for Fashion Image Retrieva (https://arxiv.org/abs/2606.19684)
Comments:
          SOICT 2025

- **Prior Approaches**: composed image retrieval(CIR)은 기준 이미지와 수정 캡션을 조합해 타깃을 찾는 문제지만, fashion image retrieval(FIR)에서는 색·무늬·질감·실루엣 같은 미세 속성 해석이 특히 어렵습니다. 기존 방법은 CLIP 기반 전역 정렬에 치우쳐 작은 텍스처/패턴 변화를 놓치기 쉽고, 학습용 positive/negative가 충분히 다양하지 않거나 negative sampling이 단순해 hard negative가 제대로 노출되지 않는 한계가 있습니다.

- **Core Contribution**: 이 논문은 멀티모달 LLM LLaVA로 속성 인지(attribute-aware) 캡션과 triplet을 생성해, 희소한 주석 데이터의 한계를 줄이면서 시각-언어 정렬을 강화합니다. 또한 two-stage fine-tuning으로 coarse 정렬 후 hard negative를 이용해 분별력을 높여, composed query에서 미세 속성 조합을 더 잘 반영하도록 설계했습니다.

- **Technical Challenges**: 문제는 (1) LLaVA 캡션의 언어 변동성이 임베딩 정렬에 잡음을 만들고, (2) 글로벌 임베딩만으로는 소매·넥라인·로고처럼 국소 속성을 정확히 구분하기 어렵다는 점입니다. 저자들은 LLaVA 입력에 구조화된 프롬프트를 사용해 기준 캡션의 속성 내용을 수정 캡션에 명시적으로 연결하고, hard negative를 in-batch·synthetic·augmented로 확장하는 하이브리드 negative sampling 및 2단계 학습으로 이 균형을 맞췄습니다.

- **Empirical Impact**: FashionIQ 데이터셋에서 Recall@K 기준으로 coarse-grained CIR 대비 정량 성능이 개선됐고, 특히 다중 속성 질의에서 compositional reasoning과 fine-grained retrieval이 더 잘 드러났습니다. 다만 단순 색상/로고의 픽셀 수준 정밀도 같은 실패 사례가 남아 있어 spatial grounding 부재와 캡션 잡음, 추가 학습 스케일의 제약이 향후 과제로 제시됩니다.



### Vortex: Multi-Modal Fusion System for Intelligent Video Retrieva (https://arxiv.org/abs/2606.19682)
Comments:
          SOICT 2025

- **Prior Approaches**: Ho Chi Minh AI Challenge 2025(AIC’25)와 VBS/LSC 계열 선행은 텍스트·비전 기반 KIS, QA, 그리고 이벤트 순서 추론까지 아우르며, CLIP 같은 embedding 기반 검색과 temporal reranking, LLM 기반 질의 확장에 강점이 있었습니다. 다만 기존 방식은 전역-국소 의미 균형(예: CLIP 단독)과 시간 순서 정합(TRAKE급)에서 단계별 설계가 복잡해지거나, LLM의 무단 질의 수정이 의도 드리프트/환각 위험을 남기는 문제가 지적됐습니다.

- **Core Contribution**: FocusOnFun의 Vortex는 키프레임 추출-멀티모달 메타데이터 생성-하이브리드 검색-인터랙션(temporal search+relevance feedback)을 end-to-end로 엮은 통합 멀티모달 비디오 retrieval 시스템입니다. 특히 CLIP과 SigLIP2의 dual embedding을 Reciprocal Rank Fusion(RRF)으로 결합해 전역 의미와 미세한 디테일 인식을 함께 잡는 전략을 제안합니다. 여기에 Rocchio 기반 relevance feedback과 “Before–Now–After” 멀티스테이지 temporal search를 더해 순차 이벤트 정렬과 사용자 반복 정제를 동시에 강화했습니다.

- **Technical Challenges**: 대규모 영상에서 모든 프레임을 처리하면 계산·저장 비용이 폭증하므로, AutoShot 기반 shot 분할 후 L2L2-norm 필터링으로 중복을 줄이는 적응형 키프레임 파이프라인이 필요했습니다. 또 TRAKE처럼 희소한 이벤트들의 순서를 검증하려면 단일 ANN 검색만으로는 부족해, 이전/현재/이후를 각각 독립 검색한 뒤 같은 video_ID에서 점수를 합산하는 휴리스틱 temporal re-ranking으로 복잡도 부담을 낮췄습니다. 마지막으로 인터랙티브 환경에서는 LLM이 질의를 자동 변형하면 의도가 흔들릴 수 있어, Vortex는 LLM을 “자율 rewrite”가 아니라 명시적 제안 제시를 통한 제어형 해석 보조로 설계했습니다.

- **Empirical Impact**: AIC’25 공식 평가에서 Vortex는 Preliminary Round 79.6/88(90.5%)을 기록했고, Final Round에서는 전체적으로 Excellent 등급을 받았으며 QA 태스크에서 Outstanding 성과를 보였습니다. TKIS는 Excellent, VKIS/TRAKE는 Very Good, QA는 Outstanding으로 과제 유형이 달라도 일관된 강건성을 보여줍니다. 특히 CLIP–SigLIP2 하이브리드 결합과 사용자 피드백+temporal search 루프가 복잡 질의에서 상호 보완 효과를 냈다는 점이 경쟁 결과로 확인됐습니다.



### TeleMorpher: Toward Robust Simultaneous Motion-Location Editing (https://arxiv.org/abs/2606.19676)
- **Prior Approaches**: 기존 motion editing은 one-shot으로 동작하면서 배경/인물 외형 보존과 프레임 간 일관성을 개선하려는 흐름이 강했습니다. 다만 MotionEditor, Edit-Your-Motion 같은 방법은 주로 motion만 다루거나, source-대상 motion 간 차이가 커지면 깜빡임과 외형 불일치가 생기고, location(공간 위치) 변화까지는 명시적으로 해결하지 못했습니다. 결과적으로 실사용에서 흔한 ‘동시에 동작과 위치를 바꾸는’ 요구를 충족하기 어려웠습니다.

- **Core Contribution**: TeleMorpher는 motion과 location을 함께 편집하는 동시(one-shot) 프레임워크를 제안합니다. 핵심 아이디어는 오프더샬 모델로 만든 segmentation·inpainting으로 배경 간섭을 줄이고, 대신 3D 아바타 기반의 motion prior(합성 motion 비디오)를 “타깃 모션 기준”으로 사용해 더 정밀하고 제어 가능한 편집을 가능하게 한다는 점입니다. 또한 학습 없이 pose warping 기반 protagonist guidance를 생성해 source-대상 모션 간 gap을 줄이면서 인물 외형을 보존하도록 설계했습니다.

- **Technical Challenges**: 동시 motion-location 편집은 (1) source-대상 motion과 location의 큰 gap, (2) 세분화/경계 모호성, (3) 배경 복잡도·동적도·카메라 움직임 등 정보량의 차이가 품질을 떨어뜨리는 문제가 있습니다. TeleMorpher는 먼저 인물 마스크로 전경/배경을 분리하고 배경은 inpainting으로 제거해 편집 혼선을 줄입니다. 그다음 학습-free two-way pose warping(곡선 피팅 기반 변형 + 필요 시 영역 수준 합성)을 통해 타깃 포즈에 더 잘 맞는 protagonist 조건을 만들고, 이를 UNet attention의 일부 key/value 채널에 주입(추가 mixup 포함)해 모션 정렬과 외형 일관성을 동시에 강화합니다.

- **Empirical Impact**: YouTube에서 수집한 in-the-wild 영상과 TaiChi 데이터셋에서 정량·정성 평가 모두에서 기존 베이스라인 대비 우수한 동시 편집 성능을 보였습니다. 특히 location과 motion을 함께 바꿀 때도 전경(인물) 및 배경의 시각적 일관성을 더 잘 유지하는 것으로 보고됩니다. 또한 배경 보존과 인물 스켈레톤 정렬을 측정하는 LPIPS 기반의 신규 metric 2종을 제안해, 정량 평가 신뢰도를 높였다는 점에서 분야 확장성도 기대됩니다.



### Learning When to Denoise: Optimizing Asynchronous Schedules for Latent Diffusion (https://arxiv.org/abs/2606.19662)
Comments:
          25 pages, 9 figures, 4 tables

- **Prior Approaches**: 다중 표현(multi-representation) diffusion은 서로 다른 표현 공간을 비동기로 denoising해 품질을 끌어올려 왔습니다. 대표적으로 SFD는 semantic latent이 texture latent보다 앞서도록 고정 오프셋을 hand-tuned하거나 소규모 grid search로 정했지만, 이 스케줄이 성능을 좌우하는 핵심 설계변수임에도 데이터 적응이 부족했습니다.

- **Core Contribution**: 이 논문은 비동기 denoising 스케줄(asynchronous schedule)을 직접 학습하는 프레임워크를 제안합니다. semantic-leading(고수준 의미가 더 빠르게 깨끗해짐) 제약을 만족하는 형태의 학습 가능한 스케줄을 두고, flow-matching 학습 목표에 스케줄 학습을 자연스럽게 결합했습니다.

- **Technical Challenges**: 주요 난점은 스케줄을 바꾸면 loss가 평가되는 ‘로컬 noising-time 분포’까지 같이 변해 버리는 confound 문제입니다. 논문은 Jacobian-corrected objective로 가중치가 로컬 시간에 대해 불변이 되도록 보정하고, 또한 스케줄이 discretization에 불리한 경로를 만들지 않게 kinetic-energy regularizer를 추가합니다. 계산 효율을 위해 1% 미만 추가 compute로 schedule을 빠르게 probe 학습한 뒤, 스케줄을 고정하고 deno이저를 이어 학습하는 2-stage 최적화를 사용합니다.

- **Empirical Impact**: ImageNet 256x256에서 학습된 스케줄은 hand-tuned semantic-first에 비해 수렴 속도와 최종 FID를 동시에 개선했습니다. AutoGuidance 설정에서 675M XL backbone 기준 200 epochs에 FID 1.05, 600 epochs에 FID 1.02를 달성해 800 epochs SFD-XL(FID 1.04대)과 비슷하거나 더 좋으면서 학습 갱신은 약 4배 줄였습니다. 또한 unguided에서도 200 epochs FID 2.37로 800 epochs SFD-XL(FID 2.54)을 더 적은 학습으로 앞서고, 스케줄 학습이 representation 종류에 덜 의존함을 보였습니다.



### GB-LSR: A Fast Local Spectral Image Representation with a Single Global Bandwidth for Continuous Reconstruction and Super-Resolution (https://arxiv.org/abs/2606.19617)
- **Prior Approaches**: 좌표 기반 신경 필드(coordinate-based neural fields)는 이미지를 연속 좌표의 함수로 보고, Fourier-feature 입력이나 sinusoidal activation(예: SIREN 계열)을 MLP에 주입해 표현합니다. 대표적으로 LIIF/LTE는 분포에 대해 amortized로 학습해 한 번의 네트워크로 임의 좌표를 질의하지만, 질의당 비용이 MLP 기반 디코딩에 의해 커질 수 있습니다. 또한 WIRE는 per-image fitting으로 세부 표현력을 얻지만 테스트 시 최적화 비용이 발생하고, locality를 강제하는 변형은 성능/비용의 균형이 일관되지 않았습니다.

- **Core Contribution**: GB-LSR(Global-Bandwidth Local Spectral Representation)는 고정 격자 형태의 local spectral representation로, 이미지를 비겹치는 패치로 나눠 각 패치가 잘린 Fourier basis 계수 텐서를 갖도록 설계했습니다. 모든 패치/이미지에서 단 하나의 trainable scalar bandwidth를 공유하며, 연속 좌표 u에서의 복원은 fixed-size basis contraction으로 이루어져 이미지 크기에 무관하게 질의 비용을 일정하게 유지합니다. 즉, locality(국소성)와 주파수 표현력을 함께 갖추면서도, end-to-end로 효율적인 디코딩 구조를 제공하는 것이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 설계 쟁점은 bandwidth(기저 주파수 스케일)를 공간적으로 달리할지(예: per-patch 적응) 아니면 전역 상수로 둘지에 있습니다. 논문은 (1) 전역 trainable scalar, (2) 전역 고정 scalar, (3) per-patch log-space bandwidth field(필요시 cutoff order까지) 세 변형을 실험하고, closed-form locality diagnostic 및 log-space ablation에서 per-patch 적응이 의미 있는 이득을 주지 못함을 보여 단일 전역 스칼라가 충분하다는 결론을 냈습니다. 또한 패치 기반 고정 basis와 partition-of-unity를 결합해, 각 좌표 질의가 고정된 이웃 패치의 계수만 사용하도록 만들어 디코더 연산량을 질의당 일정하게 고정했습니다.

- **Empirical Impact**: 표준 native-reconstruction 벤치마크( Kodak/Set14/Urban100, 256×256×256 및 matched-budget amortized 프로토콜 )에서 GB-LSR-Scalar는 LIIF/LTE/WIRE를 2.8~3.6 dB PSNR, 0.11~0.15 LPIPS 개선하면서도, 가장 느린 기준 대비 추론 비용을 약 1/4 수준으로 낮췄습니다. 또 별도 arbitrary-scale super-resolution 확장에서도 x4에서 LIIF-RDN 대비 1.44x, LTE-SwinIR 대비 3.25x 더 빠른 결과를 제시하며, 4-corner local-ensemble averaging 제거 시 1.77x 속도 향상(PSNR 변화 미미, 피크 메모리 -35%)을 보고합니다. 요약하면, GB-LSR은 ‘질의당 고정 비용+주파수 국소 표현’ 조합이 실제 성능과 효율에서 동시에 유효하다는 점을 실험으로 뒷받침한 것으로 평가됩니다.



### Language-Instructed Vision Embeddings for Controllable and Generalizable Perception (https://arxiv.org/abs/2606.19584)
- **Prior Approaches**: 기존 비전 foundation model은 정적인 특징 추출기로 학습돼, 적응은 주로 downstream 대형 모델이 떠안는 구조였다. 비전-언어 모델에서도 visual prompting이나 fine-tuning은 특정 작업에 최적화되기 쉬워 zero-shot 언어 지시를 해석하는 데 한계가 있으며, LLM 중심 late fusion은 계산 비용이 크고 vision encoder가 놓친 세부를 복구하지 못해 hallucination으로 이어질 수 있다.

- **Core Contribution**: 본 논문은 Language-Instructed Vision Embeddings(LIVE)로, 비전 인코더를 언어 지시로 직접 조절해 task-centric 임베딩을 추론 시점에 생성하는 패러다임을 제안한다. LLM을 inference 단계가 아니라 데이터 생성(합성 instruction–response 쌍) 단계의 “teacher”로만 활용해, 별도의 task-specific retraining 없이도 언어 기반으로 의미를 정렬하는 임베딩을 만든다.

- **Technical Challenges**: 핵심 과제는 instruction–answer 형태의 대규모 학습 데이터가 부족하다는 점인데, LIVE는 Gemini 2.0 Flash 같은 비전 입력을 받는 LLM으로 이미지 조건의 다양한 질문-답을 오프라인 생성해 1,640만 triplets 규모의 학습 데이터를 구축한다. 또한 SigLIP 텍스트 인코더 임베딩을 투사해 vision transformer 내부의 여러 깊이에서 언어 토큰을 주입하고, sigmoid 기반 alignment loss로 (이미지, 지시)–정답 임베딩 정렬을 학습해 지시가 임베딩 공간과 attention에 실제로 반영되도록 설계했다.

- **Empirical Impact**: 실험에서 LIVE는 hallucination을 줄이는 지표 MMVP에서 +34포인트 개선을 보였고, GQA에서는 LLM 기반 강자 대비 더 큰 성능(7포인트 우위)을 내면서 파라미터는 10% 미만 수준의 효율을 보여준다. 더 나아가 학습된 임베딩이 unseen instruction과 task로 일반화되며, teacher LLM(Gemini 2.0 Flash) 지식과의 정확도 격차를 좁히는 경향과 함께 attention 시각화로 언어 지시에 따른 시각적 포커싱이 나타났다는 점이 의미 있다.



### Mix-QVLA: Task-Evidence-Aware Mixed-Precision Quantization of Vision-Language-Action Models (https://arxiv.org/abs/2606.19565)
- **Prior Approaches**: VLA(vision-language-action) 모델은 시각·언어를 입력으로 받아 로봇 동작을 직접 생성하는데, PTQ( post-training quantization )로 비용을 줄이려는 시도가 늘고 있습니다. 기존 VLA 양자화는 주로 최종 action deviation이나 kinematic proxy, 또는 스케일 보정 같은 기준으로 민감도를 추정하지만, 내부에서 ‘결정을 받쳐주는 evidence’가 보존되는지까지는 충분히 못 봅니다. 그 결과 최종 동작은 비슷해 보여도 grounding이나 추론 경로가 깨져 closed-loop 상호작용에서 제어가 흔들릴 수 있다는 한계가 남아 있습니다.

- **Core Contribution**: 이 논문은 Mix-QVLA라는 task-evidence-aware mixed-precision PTQ 프레임워크를 제안합니다. full-precision 모델의 action-token 결정(참조 결론)에 고정(anchor)한 뒤, 시각 인코더~프로젝터~멀티모달 추론~액션 헤드 등 VLA 기능 경계에서 ‘결정 지원 evidence’가 양자화에도 유지되는지 비교합니다. evidence의 크기(evidence-mass)와 배치(evidence attribution)를 함께 왜곡으로 보고, 이를 layer-wise 민감도와 mixed-precision bit allocation으로 연결합니다.

- **Technical Challenges**: 핵심 기술적 난제는 단순히 최종 출력 차이로는 양자화가 내부 근거 경로를 망가뜨리는지 알기 어렵다는 점입니다. Mix-QVLA는 boundary activation에서 normalized gradient-weighted task-evidence map을 만들고, full-precision 대비 evidence-mass와 attribution-distribution distortion을 Jensen–Shannon divergence로 정량화합니다. 또한 trajectory 단계(phase)가 바뀌며 layer 중요도가 달라질 수 있어, temporal sensitivity를 soft-bottleneck으로 집계해 ‘시간 의존 민감도’를 반영한 뒤 model-size와 BitOps 예산 아래에서 배분 최적화를 수행합니다.

- **Empirical Impact**: LIBERO 벤치마크에서 OpenVLA-OFT 기준으로 W4A4 설정에서도 평균 성공률 96.3%를 유지하면서 메모리를 크게 줄였습니다. OpenVLA-OFT에서는 OpenVLA-OFT 메모리 15.4GB→4.1GB로 낮추고 BF16 대비 평균 success를 97.1%에 가깝게(96.3%) 보존했으며, 추론 속도도 1.52x 향상시켰습니다. ablation 결과로는 task-evidence 신호와 temporal 신호를 함께 쓰는 것이 단독 신호보다 성공–효율 트레이드오프를 더 개선함을 확인했습니다.



### PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models (https://arxiv.org/abs/2606.19534)
Comments:
          Code available at this https URL

- **Prior Approaches**: 기존 MLLM은 시각 이해에 강점을 보이지만, 대부분 autoregressive(AR) 생성 방식이라 여러 영역을 한 번에 처리할 때 비효율적입니다. 특히 region captioning에서는 각 마스크를 순차적으로 디코딩하고 토큰도 단계적으로 생성해, 영역 수가 늘수록 지연(latency)이 거의 선형으로 커집니다. diffusion language model(DLM)은 병렬 토큰 생성 잠재력이 있지만, fine-grained localized perception에 그대로 확장하는 것은 품질과 병렬성 모두에서 검증이 부족했습니다.

- **Core Contribution**: 이 논문은 diffusion 기반 멀티모달 언어모델 PerceptionDLM을 제안해, 단일 denoising 과정에서 여러 masked region의 캡션을 병렬로 생성하도록 설계했습니다. PerceptionDLM-Base는 discrete diffusion multimodal baseline으로 visual instruction tuning을 통해 지각(perception) 품질을 먼저 끌어올리고, 이후 region-aware 구조(모양을 구분하는 region prompting과 영역 간섭을 줄이는 structured attention masking)를 얹어 병렬 region captioning을 가능하게 했습니다. 또한 ParaDLC-Bench를 통해 캡션 품질과 추론 효율을 동시에 평가하는 틀을 마련했습니다.

- **Technical Challenges**: 핵심 난제는 DLM이 한 번에 여러 토큰을 복원하는 특성상, 동시에 생성되는 서로 다른 영역 간 ‘엔탱글먼트/간섭’을 어떻게 억제하느냐입니다. 논문은 영역별로 RoI-aligned feature replay와 learnable region embedding을 주고, 영역 토큰의 attention을 전역 비주얼 토큰/공유 프롬프트/해당 영역 RoI 토큰 및 자기 영역 캡션 스팬으로 제한하는 structured attention masking으로 영역 독립성을 강제했습니다. 더불어 효율을 위해 멀티해상도 타일링과 학습 단계별(정렬→중간지식→지시따르기→고품질 정제) 파이프라인을 구성했습니다.

- **Empirical Impact**: 실험에서 PerceptionDLM-Base는 16개 멀티모달 벤치마크 중 15개에서 기존 diffusion VLM 대비 우위를 보이며, 특히 ParaDLC-Bench의 multi-region 설정에서 평균 정확도 62.4%로 baseline 대비 큰 폭(예: LLaDA-V 35.2%)의 향상을 보였습니다. 효율 측면에서는 병렬 디코딩 덕분에 multi-region 상황에서 지연이 영역 수에 따라 급격히 늘지 않고, heavy workload(예: 4 masks)에서는 throughput이 최대 3.44배, 단일 이미지 latency가 10.04초→2.92초로 감소했다고 보고합니다. 결과적으로 diffusion 기반 멀티모달 모델이 fine-grained 시각 지각을 ‘병렬’로 확장할 수 있음을 실증하며, 실사용 관점의 처리량 개선 가능성을 제시합니다.



### ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing? (https://arxiv.org/abs/2606.19531)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 World Action Model(WAM)은 비디오 생성으로 미래 장면을 상상한 뒤 행동을 예측하는 방식이 많았다. 하지만 비디오 기반은 다중 프레임의 dense 미래 토큰 처리로 추론 비용이 커지고, 외형·카메라·배경 변화처럼 행동과 무관한 정보에 자원을 소모한다. 또한 긴 호라이즌 상상이 실제 접촉·변형을 물리적으로 일관되게 맞추기 어려워, 오류가 행동 예측을 오도할 수 있다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 “비디오 생성이 정말 필요할까?”라는 질문에 답하며, ImageWAM이라는 대안을 제안한다. ImageWAM은 텍스트 지시가 들어간 이미지 편집(image editing) 모델을 world-action 백본으로 재활용해, 현재 관측에서 목표 상태로의 “변환”만 요약하는 단일 엔드포인트 프레임 표현을 중간 컨텍스트로 사용한다. 특히 추론 시에는 목표 프레임을 디코딩하지 않고, 편집 denoising 과정에서 생긴 KV cache를 flow-matching 기반 action expert에 바로 조건으로 넣어 계산량을 줄인다.

- **Technical Challenges**: 핵심 과제는 비디오 생성이 제공하던 reason-before-act 중간 단계의 이점을, 이미지 편집의 컨텍스트로 얼마나 효과적으로 대체하느냐였다. 저자들은 편집 모델의 디코딩 출력이 아니라 denoising 중간 단계의 KV cache가 언어-조건 시각 변환 정보를 담는다는 점에 주목해, action expert가 이 캐시를 통해 직접 행동 시퀀스를 생성하도록 학습한다. 또한 학습 중에는 편집 denoising timestep τ를 샘플링해 다양한 단계의 캐시를 보게 하고, 손실은 이미지 엔드포인트 예측(velocity field)과 행동 flow-matching을 함께 최적화해 캐시-행동 정렬을 강화한다.

- **Empirical Impact**: 실험에서 ImageWAM은 LIBERO/LIBERO-Plus, RoboTwin 2.0 및 듀얼암 실세계 태스크(Flux.2 기반)에서 VLA 기준선 대비 성능을 개선했고, 비디오 기반 WAM과도 경쟁/유사 성능을 보이면서 계산 효율을 크게 낮췄다. 구체적으로 FLOPs는 63.65→9.7로, 지연(latency)은 1081ms→263ms로 감소해 실시간 제어에 유리함을 보여준다. attention 분석 역시 편집 캐시가 조작물·접촉부·목표 수납영역처럼 과업 관련 변화 영역에 집중한다는 점을 확인해, 이미지 편집이 비디오 기반 world-action 모델의 실질적 대안이 될 수 있음을 뒷받침한다.



### LooseControlVideo: Directorial Video Control using Spatial Blocking (https://arxiv.org/abs/2606.19495)
Comments:
          Project page at this https URL

- **Prior Approaches**: 기존 text-to-video 확산 모델은 생성 품질은 높지만, 원하는 물리적 사건(다중 객체 상호작용·공간 궤적·가림 관계)을 정밀하게 지휘하는 감독 신호가 부족하다는 한계가 컸습니다. ControlNet 계열의 2D/고밀도(프레임별 depth·에지 등) 가이드는 구조를 잘 맞추는 대신, 사용자가 장면마다 프레임 단위의 촘촘한 정보를 제작하기 어렵고 camera ego-motion과 객체 변형이 섞이는 문제가 있습니다.

- **Core Contribution**: LooseControlVideo(LCV)는 ‘blocking’ 아이디어를 차용해, 사용자가 방향성이 있는 sparse 3D 박스(Oriented 3D boxes)로 큰 윤곽과 궤적 타이밍만 설계하고, 세부 변형·가림·동역학은 생성 모델이 실행하도록 분리했습니다. 이를 위해 3D 좌표계의 추상 파라미터를 바로 주입하지 않고, 카메라 관점에서 사전 렌더링한 DNOCS(Depth-modulated Normalized Object Coordinate Space) 제어 신호로 2D 도메인에 맞춰 정렬하는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 ‘3D 박스/카메라 파라미터’와 ‘영상 도메인’ 사이의 표현 불일치를 줄이는 것이었습니다. LCV는 DNOCS로 객체 방향(색상)과 전역 깊이(밝기)를 동시에 인코딩해 가림·깊이 순서·투영 단서를 제어 입력에 명시적으로 드러내고, 이를 Wan 2.2 DiT 백본에 LoRA로 fine-tuning해 sparse intent에서 실행을 학습시켰습니다. 또한 10K 규모의 in-the-wild 영상에서 자동 추적·정렬된 oriented 3D 박스를 뽑아(깊이 예측·포인트클라우드·Kalman filtering) 수동 3D 어노테이션 부담을 크게 줄였습니다.

- **Empirical Impact**: nuScenes, HO-3D, BEHAVE 벤치마크에서 2D-box 및 flow 기반 baseline 대비 trajectory error와 occlusion 관련 지표가 뚜렷이 개선됐습니다. 논문은 Trajectory Error 1.2~3배 감소, Rigid Motion Consistency 2배 개선, Occlusion Accuracy 1.5~2배 향상을 제시하며, oriented 3D primitives가 복잡한 다중 에이전트 영상 제작에서 유효한 기하 prior가 됨을 보여줍니다. 무엇보다 생성 품질을 크게 해치지 않으면서 로컬 편집(점프 궤적 조정·상호작용 추가)을 전역 문맥의 붕괴 없이 수행할 수 있다는 점이 실용적 의미가 큽니다.



### LEAP: Layer-skipping Efficiency via Adaptive Progression for Vision Transformer Distillation (https://arxiv.org/abs/2606.19483)
- **Prior Approaches**: 기존 feature-based knowledge distillation은 teacher의 최종/중간 feature map을 student가 따라 하도록 학습하지만, teacher-student gap 때문에 작은 student가 고차원·복잡한 feature를 초반부터 그대로 모방하기 어려워 학습이 불안정해지곤 합니다. 중간 layer matching(예: PKD)이나 투영 헤드로 차원을 맞추는 방법, 관계(유사도)만 맞추는 relational distillation 등이 시도됐지만, 대체로 고정된 스케줄이나 수동 layer 선택에 의존해 아키텍처/깊이 불일치 상황에서 타깃이 임의적이 될 수 있습니다. 또한 curriculum learning 아이디어가 일부 비전 파운데이션 모델에 적용되었지만, feature-based distillation에서 “무엇을 언제” 타깃으로 삼을지가 여전히 제한적이었습니다.

- **Core Contribution**: 이 논문은 ViT feature-based distillation을 위한 layer-skipping curriculum인 LEAP를 제안합니다. 핵심은 teacher의 중간 feature를 난이도(learning ease)에 따라 단계적으로 바꿔가며 학습시키는 것으로, CKA 유사도 기반으로 현재 타깃을 충분히 따라왔을 때만 더 깊은(더 어려운) teacher layer로 진행합니다. 이를 통해 student가 먼저 기초 표현을 만들고, 이후 상위 의미 추상화로 자연스럽게 확장되도록 유도합니다.

- **Technical Challenges**: 문제는 “teacher의 어떤 중간 layer가 student에 가장 쉬운 타깃인가”를 고정 스케줄 없이 자동으로 결정해야 한다는 점입니다. 저자들은 baseline 학습 중 student의 최종 feature와 teacher의 각 중간 feature 사이 유사도(센터드 커널 얼라인먼트, CKA)가 시간에 따라 이동한다는 관찰을 바탕으로, 온라인 CKA가 임계값 τ에 도달하면 다음 layer로 전환하는 적응형 진행 규칙을 설계했습니다. 또한 teacher 추론을 초반에는 일찍 중단(early-stopping)해 학습 효율까지 함께 최적화하는 방식으로 실용성을 높였습니다.

- **Empirical Impact**: ImageNet-100에서 LEAP-distilled ViT-S는 90.1% 정확도로 baseline 대비 +12.24% 향상했고, ImageNet-1K의 Oxford/Paris instance retrieval에서 mAP가 각각 +3.84%, +7.75% 개선됐습니다. 의미 분할(ADE20K)과 선형 probing에서도 일관된 성능을 보이며, 특히 픽셀/인스턴스 수준 표현이 유지된다는 점을 확인했습니다. 동시에 CKA 임계값 기반 early-stopping으로 teacher FLOPs와 학습 시간을 각각 25.1%(FLOPs)와 21%(time) 절감할 수 있어, 단순 정확도 향상뿐 아니라 효율 측면 impact도 큽니다.



### Scaling Generative Foundation Models for Chest Radiography with Rectified Flow Transformers (https://arxiv.org/abs/2606.19460)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 흉부 X-ray(CXR) 합성 모델은 환자 하위집단, 의료기관, 촬영 조건이 달라지면 일반화 성능이 급격히 떨어져 실제 임상 활용도가 낮다는 문제가 지적돼 왔다. 또한 생성 품질이 제한적이어서 다양한 임상 데이터셋을 안전하게 확장하거나 진단 모델의 강건성을 평가하기 어려웠다.

- **Core Contribution**: 이 논문은 CXR을 위한 최초의 ‘생성형 foundation model’을 대규모 스케일로 처음부터 학습해, 생성과 편집을 모두 제어 가능하게 만든다는 점이 핵심이다. 1.3B+ 파라미터 규모 모델을 약 1.2M 방사선 이미지와 임상 전문가 유도 메타데이터(총 1.6T tokens)로 학습해, 인구통계·촬영 뷰·병리 유형 전반에 걸친 조건부 생성/편집을 지원한다.

- **Technical Challenges**: 진단에 쓰일 정도의 ‘고충실도’ 합성을 위해서는 도메인 특화 생성 안정화와, 메타데이터 조건을 인과적으로 결합해 편집 일관성을 유지하는 설계가 필요했다. 이를 위해 RadVAE(EDM2 기반 변형)로 잠재표현을 구성하고, flow 기반 transformer/U-Net 백본에 RMS-Norm, SwiGLU 같은 학습 안정화 업그레이드를 적용했으며, 메타데이터를 SCM(인과 그래프)로 모델링해 ODE 기반(continuous-time flow) 추론과 반사실/카운터팩추얼 생성을 수행하도록 구성했다.

- **Empirical Impact**: 평가에서는 FID 외에도 방사선/자기지도 표현공간에서의 FDD, KID/KDD, Precision/Recall/Density/Coverage 등으로 분포 충실도와 다양성을 폭넓게 측정했고, 생성-실제 이미지 간 시각적 구분이 임상 전문가 수준에서 거의 불가능하다고 보고했다. 더 나아가 편집 시 환자 identity preservation을 SSIM과 Rad-DINO/DINOv3 기반 perceptual distance, 속성별 임베딩 유사도로 정량화해 ‘데이터셋 다양화’와 ‘진단 모델 강건성 시험’에 의미 있는 기반을 제시한다.



### The Token Is a Group Element: On Lie-Algebra Attention over Matrix Lie Groups (https://arxiv.org/abs/2606.20547)
Comments:
          preprint, 19 pages, 3 figures

- **Prior Approaches**: 기존 동등(equivariant) 공간추론에서는 토큰을 보통 벡터 임베딩으로 두고, 변환군 G의 작용을 representation ρ(g)로 외부에서 주입하는 방식이 표준이었다. 이때 동등성을 위해 irreducible representation, Clebsch–Gordan, steerable kernel, 보조 프레임 같은 표현이론/기하 도구가 필수로 등장한다. 또한 비컴팩트·비가환 affine 같은 군은 unitarity 제약(irreps)이나 surjective-exp 기반의 구성 한계로 우회/배제가 잦았다.

- **Core Contribution**: 이 논문은 attention 토큰 자체를 “벡터”가 아니라 matrix Lie group의 원소 g_i로 바꾸는 Lie-Algebra Attention을 제안한다. 토큰이 변환을 직접 담으므로, 쌍의 상대기하 g_i^{-1}g_j의 로그(log)로 얻는 Lie algebra 불변량 w_{ij}가 본질적으로 정해지고 동등성과 cocycle 조건도 표현이론 없이 자동으로 성립한다. 결과적으로 attention score는 학습 커널이 아니라 w_{ij}의 대수 노름으로 닫힌형(closed-form)으로 읽힌다.

- **Technical Challenges**: 핵심 난제는 비컴팩트/비가환 affine까지 포함하면서도 closed-form 점수를 만들 수 있는 불변 스칼라와 거리(노름) 설계를 찾는 것이다. 저자들은 로그 차트에서만 정의되는 상대 pose에 대해 w_{ij}=log(g_i^{-1}g_j)를 쓰고, Frobenius 기반의 block-weighted 내적(가중치 λ)을 통해 score s_{ij}=-||w_{ij}||^2_λ/τ를 구성한다. 특히 Aff(2), Aff(3)처럼 Ad-invariant 정준 내적이 존재하지 않는 비세미(simple) 케이스에서도 “불변량 w_{ij} 자체의 불변성”을 이용해 메트릭의 Ad-invariance 없이도 점수가 G-대각 작용에 대해 보존되게 했다.

- **Empirical Impact**: SE(2), SO(3), Aff(2)에서 sequence completion 실험을 통해 닫힌형 score가 같은 불변량을 쓰는 학습 MLP 커널과 비교해 성능을 내거나 SE(2)에서는 더 좋다고 보고한다. 파라미터 측면에서도 닫힌형은 50~80배 적은 score 파라미터로 동등/개선 성능을 달성하며, 벡터 토큰 baseline은 불변성 훼손으로 큰 성능·일관성 차이를 보였다. 이는 irreps 전통과 surjective-exp 전통이 막아왔던 affine full-frame 영역까지 attention을 확장하는 실증적 근거로 의미가 있다.



### StylisticBias: A Few Human Visual Cues Drive Most Social Biases in MLLMs (https://arxiv.org/abs/2606.20527)
Comments:
          Accepted to the non-archival workshops AI4Good and Culture x AI at ICML 2026

- **Prior Approaches**: 기존 연구는 다양한 개인/인구집단을 비교해 편향을 측정해 왔지만, 이 방식은 ‘외모 속성’과 ‘정체성(개인 고유 차이)’을 분리하기 어렵다는 한계가 있습니다. 또한 시각적 편향은 주로 매력도 같은 집계 신호로 다뤄져, 어떤 구체적 단서가 판단을 바꾸는지 고해상도로 분해하기가 어려웠습니다. 결과적으로 MLLM이 사람을 평가할 때 어떤 비주얼 cue에 민감한지 정밀 감사가 부족했습니다.

- **Core Contribution**: 이 논문은 MLLM의 사회적 판단에서 ‘속성 수준 편향’을 통제적으로 평가하기 위한 벤치마크 StylisticBias를 제안합니다. 핵심은 500개의 기준(base) 얼굴에서 정체성은 고정한 채 단일 시각 속성만 편집해 약 25K 이미지를 만들고, 6개 MLLM을 25개의 이분(positive/negative) 사회 판단 시나리오에 대해 비교한다는 점입니다. 이를 통해 모델이 특정 외모 속성의 변화에 얼마나 일관되게 반응하는지 직접 측정할 수 있게 합니다.

- **Technical Challenges**: 가장 큰 기술적 도전은 ‘정체성 고정’ 하에서 속성만 바꾼 이미지들이 실제로 단일 속성 변화 외의 잡음(불의도 아티팩트, 시맨틱 불일치)을 최소화하도록 설계·검증하는 것입니다. 연구진은 Imagen 4로 기준 얼굴을 만들고 Nano Banana로 속성별 단일 편집을 수행하되, 프레이밍·조명·배경을 최대한 통제하고 인간 검증(90% 수동 리뷰, 98% 통과)을 거쳐 품질을 보장했습니다. 또한 프롬프트 순서와 random seed에 대한 민감도를 줄이기 위해 다양한 주문/시드에서 강제 선택을 반복하고 파싱 가능한 응답만 집계해 선호도 점수 변화를 계산합니다.

- **Empirical Impact**: 실험 결과, 외모 판단 편향은 소수의 자기표현(appearance-related) 단서에 집중되며, 특히 패션 스타일 같은 의도된 선택 신호가 가장 큰 속성 수준 변화(예: 패션·헤어·메이크업/립·안경)를 유발했습니다. 반면 age와 body type은 가장 큰 인구통계(정체성) 수준의 효과를 보이되, 단일 속성 편집에서는 약 15개 속성이 전체 변화의 약 80%를 설명해 편향이 ‘소수 단서’에 응축됨을 보여줍니다. 의미 정렬(semantic alignment) 편향도 확인되어, 사회경제/스타일 관련 판단처럼 외모 의미와 직접 맞닿은 질문에서 민감도가 더 강하게 나타났고, 모델 간에는 ‘중요 단서의 패턴’이 비교적 일관되지만 ‘반응 강도’는 더 큰 모델에서 완화되는 경향이 관찰됩니다.



### Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation (https://arxiv.org/abs/2606.20491)
Comments:
          Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)

- **Prior Approaches**: 기존 scanpath 예측은 정적 saliency 맵 기반에서 출발해, 이후에는 다음 고정점이 시각 자극과 과거 gaze 기록에 의존하도록 RNN/ConvLSTM, 최근엔 Transformer로 확장됐다. 다만 이러한 모델들은 계산량과 추론 지연이 커서 임베디드 로보틱스에서 실시간 embodied deployment가 어렵다는 한계가 있었다. 또한 attention 예측을 로봇의 active vision/항법에 결합하더라도, human-fixation 같은 시간적 주의 구조를 정책에 직접 최적화하는 방식은 상대적으로 초기에 머물러 있었다.

- **Core Contribution**: 이 논문은 가벼운 scanpath 예측 모델 GazeLNN을 제안한다. MobileNetV3로 특징을 뽑고, recurrent engine으로 Liquid Neural Networks(LNN)의 CfC(Closed-Form Continuous-time) 변형을 써서 고정점 heatmap을 auto-regressively(자기회귀적으로) 순차 생성한다. 추론 비용을 극적으로 줄이면서도 MIT Low Resolution에서 ScanMatch 0.47의 SOTA를 달성해, 인간 주의 모델링을 로봇 자율성에 실용적으로 연결하는 발판을 만든다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 순차적 gaze를 잘 모델링하면서도 (2) 실시간 로보틱스 제약(연산량/지연)을 만족하는 경량화였다. GazeLNN은 fixation을 (x,y) 좌표가 아닌 heatmap으로 표현하고 CoordConv로 공간 좌표 정보를 보강했으며, CfC의 입력-의존 시간 동역학과 gated time update로 fixation 간 시간 구조까지 반영해 정확도와 효율을 동시에 노렸다. 여기에 파라미터와 GFLOPs를 크게 줄이기 위해 MobileNetV3 백본을 채택하고, 학습에는 OSIE 기반의 시퀀스 학습/손실(KL-DTW)을 사용했다.

- **Empirical Impact**: GazeLNN은 0.61 GFLOPs, 15.24M 파라미터로 동작하며 tSPM-Net 대비 연산비용 99.40% 절감과 최대 6배 빠른 추론을 보고한다. 예측 성능은 MIT Low Resolution에서 여러 곡선/문자열/시간열 메트릭(ScanMatch, DTW, TDE 등) 전반에서 recurrent 기반 베이스라인을 능가하며, 정성적으로도 인간 ground truth scanpath에 더 가깝게 예측됐다. 나아가 RL 기반 active camera-robot 제어 정책에 GazeLNN을 통합해 드론 환경에서 예측된 fixation을 따라가도록 하며, 실제 비행 실험에서 정적 카메라 대비 salient-relevant 관찰 및 글로벌 맵 누적(약 50%↑, salient 영역 관찰 8배↑)을 통해 실사용 가치를 검증했다.



### On the Redundancy of Timestep Embeddings in Diffusion Models (https://arxiv.org/abs/2606.20416)
Comments:
          17 pages

- **Prior Approaches**: 확산 모델은 노이즈 제거 단계별 강도를 조절하기 위해 timestep embedding(시간 임베딩)을 U-Net에선 feature map에 더하거나, DiT에선 AdaIN 같은 방식으로 주입해 왔습니다. 하지만 이 관행의 “필수성”은 충분히 검증되지 않았고, 기존 연구는 주로 노이즈 스케줄링을 개선하거나, 일반적인 time-agnostic/blank denoising 가능성을 논하는 데 머물렀습니다. 특히 글로벌 통계에 기대는 접근은 이미지에서 global feature가 약해지는 상황을 덜 촘촘히 다뤘다는 한계가 있습니다.

- **Core Contribution**: 본 논문은 timestep embedding이 실제로는 불필요할 수 있다는 가설을 U-Net과 Diffusion Transformer( DiT ) 양쪽에서 체계적으로 시험합니다. 또한 확산 학습 목적함수의 전역 최적해가 특정 조건에서 “명시적 시간 조건 없이”도 동일하게 달성될 수 있음을 이론적으로 제시합니다. 핵심 메시지는 모델이 손상된 입력으로부터 잡음 스케일을 암묵적으로 복원할 수 있다면, explicit timestep conditioning은 중복될 수 있다는 점입니다.

- **Technical Challenges**: 시간 임베딩을 제거하면 각 단계에서 어떤 noise scale로 역과정을 학습해야 하는지가 불명확해져 성능 붕괴가 일어날 위험이 있습니다. 논문은 global minimizer 관점에서 time-agnostic loss의 최적해가 score function(점수함수)과 비례 관계를 갖고, 추가로 입력으로부터 bar{alpha} (암묵적 noise scale)를 복원 가능한 경우 timestep이 사실상 조건문으로서 필요 없음을 보입니다. 동시에 고차원에서 입력 노름(norm) 같은 통계량이 noise scale에 대해 충분히 집중(concentration)하는 조건과, 정규화된 데이터 환경에서 그 메커니즘이 약해질 수 있는 예외까지 함께 다룹니다.

- **Empirical Impact**: CelebA와 CIFAR-10에서 timestep embedding을 완전히 제거한 time-agnostic 모델이 높은 구조적 충실도(structural fidelity)를 유지하며, 일부 경쟁 지표에서는 조건부 모델을 능가하는 결과를 보고합니다. 구체적으로 FID, precision, recall 같은 표준 생성 품질/다양성 지표에서 개선 또는 견줄 만한 성능을 보였고, 정성 비교에서도 유사한 수준의 이미지 복원이 관찰됩니다. 이는 diffusion 아키텍처가 “시간 신호 없이도” 노이즈 스케일을 입력에서 추론할 수 있음을 시사하며, 더 효율적이고 구조 중심의 생성 모델 설계로 이어질 가능성을 열어줍니다.



### Integrating national forest inventory, airborne lidar, and satellite imagery for wall-to-wall mapping of forest structure with computer vision (https://arxiv.org/abs/2606.20291)
- **Prior Approaches**: 기존 원격탐사는 정밀임업을 표방하지만, 항공·드론·지상 레이저처럼 고가 센서는 정기적 업데이트와 wall-to-wall(전면) 적용이 어렵다는 한계가 있었다. 또한 LANDCAR에 기반한 탄소/바이오매스 중심 제품은 목적은 크지만, operational forest planning에 필요한 기초 임목 구조 지표들을 한 프레임에서 충분히 함께 제공하는 사례가 드물었다. 마지막으로 서로 다른 데이터 공급자·시기·품질을 섞어 쓰는 방식은 계획 시스템에서 예기치 못한 아티팩트와 ‘confounding behavior’를 유발할 수 있었다.

- **Core Contribution**: 논문은 산불·산림 위험 관리에 필요한 연 1회 수준의 wall-to-wall 임업 의사결정용 지도를 목표로, 일관된 추론 기반을 제공하는 VibrantForests 프레임워크를 제안한다. Sentinel-2를 입력으로 삼고, 리더(lidar) 유래 샘플로 학습한 위성 기반 산림 구조 모델을 통해 canopy cover, canopy height, aboveground live tree biomass, basal area, QMD를 10m 해상도로 동시 추정한다. 특히 희박-저바이오매스부터 울창-고바이오매스까지 조건 전반에 대해 예측 범위를 넓히고, 작은 면적·희박 조건의 과대추정과 큰 면적·밀집 조건의 과소추정을 줄이는 것이 핵심이다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 정확히 매칭되는 대규모 ground truth가 부족하고 (2) 다중 속성(cover/height/AGB/BA/QMD)을 동시에 안정적으로 학습하면서 (3) 포화(saturation)와 회귀-평균(regression-to-mean) 편향을 줄이는 것이다. 저자들은 먼저 FIA의 subplot 단위 측정치로 allometric forest structure model을 학습해 lidar 기반 훈련 라벨을 사전에 생성한 뒤, 이를 위성 모델의 학습 타깃으로 사용한다. 위성 모델은 Masked AutoEncoder로 사전학습된 인코더(고정)와 Feature Pyramid Network(FPN)를 결합해 시공간 임베딩을 활용하고, quantile regression(0.1~0.9)으로 분포 형태를 직접 학습해 속성별 스케일 차이와 불확실성을 함께 다룬다.

- **Empirical Impact**: VibrantForests는 CONUS 전역에 대해 10m 해상도의 연 단위 동시 추정 산출물을 제공하며, 같은 계열의 passive-sensor 모델에서 흔한 포화 구간을 더 넓게 완화하는 결과를 보인다고 보고한다. 또한 회귀-평균이 유발하는 편향(희박/작은 조건 과대추정, 울창/큰 조건 과소추정)을 줄여 operational planning에서 더 일관된 지표를 얻도록 돕는다는 점을 강조한다. 저자들은 2024년 위성 바탕의 wall-to-wall 예측을 field data로 평가 대상으로 삼아 실측 기반 검증을 설계했으며, 산불·복원·치료 계획 같은 실무 의사결정 워크플로의 신뢰도를 높일 수 있는 영향력을 노린다.



### Efficiently Linking Real Scenes with Synthetic Data Generation for AI-based Cognitive Robotics and Computer Vision Applications (https://arxiv.org/abs/2606.20272)
Comments:
          Accepted and best paper award at MHI-Kolloquium 2024

- **Prior Approaches**: 기존 비전 기반 인지 로보틱스 연구는 객체 검출/세그멘테이션, 6D pose 추정, grasping pose 추정 등 각 단계별로 성능을 끌어올리는 방식이 주류였다. 하지만 고정된 도메인에서만 강하고(benchmarks·학습 데이터 의존), 정밀한 라벨이 필요해 대규모·다양한 학습 데이터 확보가 병목이 된다. 또한 시뮬레이션을 쓰면 데이터는 늘릴 수 있지만 sim2real domain gap, 속도, 구현 비용 문제가 남는다.

- **Core Contribution**: 이 논문은 실세계 장면과 시뮬레이션을 연결해, “무한에 가까운 주석 데이터”를 만들되 반복적으로 sim2real 간극을 메우는 데이터 생성-학습 루프를 제안한다. 특히 3D 자산을 실세계 스캔에서 얻고, 이를 시뮬레이션 자산으로 사용해 시뮬레이션 라벨(예: 6D pose, grasp 후보)을 만든 뒤 실세계 정합과 재학습으로 정확도를 올리는 흐름을 목표로 한다. 결과적으로 단일 태스크 전용 모듈을 조합하는 접근보다, 인지·물리 기반 추론에 필요한 학습 기반을 마련하려는 방향성을 강조한다.

- **Technical Challenges**: 핵심 난제는 실세계 3D 자산을 시뮬레이터에서 바로 쓰기 좋게 분해·구축하는 과정(예: Nerf에서 mesh/texture로의 전환)과, 인간 설계 없이도 테이블/선반 등 “그럴듯한 배치 논리”를 스케일 가능하게 자동화하는 것이다. 또 시뮬레이션에서 생성한 주석이 실세계에 그대로 이전되지 않는 sim2real gap을 줄이기 위해, 시뮬레이션으로 학습한 6D pose 추정기를 실세계 초기화에 쓰고 등록(registration)과 추가 주석을 업데이트하는 단계적 루프를 설계한다. 저자들은 Omniverse/Blender, Isaac Sim 같은 도구로 포토리얼 렌더링과 물리 기반 실험(grasps/force 조건 등)을 뽑고, 스타일/텍스처 보강 등으로 시각적 차이를 완화하려 한다.

- **Empirical Impact**: 현재는 진행 중(work-in-progress)으로, 논문은 실험 결과 수치보다 “데이터 파이프라인”의 가능성과 필요성을 정리하는 데 초점을 둔다. 다만 이 방식이 성공하면 grasping pose 추정 및 end-to-end planning 같은 다운스트림에서 더 큰 다양성의 라벨 데이터를 확보해 도메인 밖 일반화와 스케일 확장에 실질적인 동력이 될 것으로 기대된다. 인지 로보틱스에서 가장 큰 제약이던 주석 비용을 줄이면서, 실세계-시뮬레이션을 왕복하며 모델 성능을 끌어올리는 기반이 될 수 있다는 점에서 의미가 있다.



### When Calibration Fails the Vulnerable Hospital: Federated Conformal Risk Control via Risk-Curve Shrinkag (https://arxiv.org/abs/2606.20115)
Comments:
          9 pages, 3 figures, 2 tables. Submitted to the DeCaF Workshop at MICCAI 2026

- **Prior Approaches**: 의료 분할(segmentation)에서 Conformal risk control(CRC)은 보유 데이터로 threshold를 보정해 분포에 무관한 품질 보증을 제공한다. 다만 연계 병원 환경에서는 calibration 점수를 전 병원에서 한데 모아 단일 임계값을 만드는 ‘naive pooled CRC’가 일반적이지만, 이 방식은 평균(개별 병원 혼합 기준) 보장은 지키더라도 병원별(기관 조건) 커버리지는 어긋날 수 있다.

- **Core Contribution**: 이 논문은 다기관 뇌종양 데이터(FeTS-2022, 20개 기관, 1,251명)에서 naive pooled CRC가 “평균은 맞지만 40% 병원이 목표 false-negative rate를 위반”하는 치명적 실패 모드를 정량화한다. 이어서 각 사이트가 전체 데이터 없이 자기 ‘실증 risk curve(G개 스칼라)’만 보내고 서버가 shrinkage 기반으로 기관별 threshold를 계산하는 shrinkage-based federated CRC를 제안한다.

- **Technical Challenges**: 핵심 과제는 연합 환경에서 기관별 샘플 수가 다르고(소규모 병원), finite-sample correction 때문에 per-site CRC는 예측셋이 비대해지거나(임상 불가 수준 stretch), 반대로 전역 보정은 취약 병원을 놓치는 트레이드오프가 생긴다는 점이다. 논문은 shrinkage가 로컬과 pooled 보정의 균형을 만들도록 설계하고, 단 하나의 하이퍼파라미터 n0가 ‘최악 커버리지 vs 예측셋 효율’을 매끄럽게 조절하며 leave-one-site-out 민감도 분석으로 n0=19 같은 운전점을 찾도록 했다.

- **Empirical Impact**: 실험에서 naive pooled CRC는 많은 병원에서 커버리지를 어기며(최악 병원은 목표를 7.8%p 초과), per-site local CRC는 stretch가 83x까지 치솟아 실사용이 어렵다. 반면 제안한 shrinkage-based federated CRC는 n0=19에서 평균 위반 2.7/20과 stretch 2.0x를 동시에 달성해 커버리지 격차를 실용 수준으로 줄였고, Lagrangian 예산 최적화는 취약 병원에 위험을 집중시켜 실패했으며 finite-sample correction 제거 시 위반이 3배로 악화되는 점도 확인했다.



### Tri-Info: Generalizable, Interpretable Failure Prediction for VLA Models via Information Theory (https://arxiv.org/abs/2606.19998)
- **Prior Approaches**: 기존 VLA failure detector는 크게 두 계열로 나뉩니다. embedding 기반 방법은 내부 표현에 classifier를 얹어 in-domain 정확도는 높지만, 표현 공간이 모델마다 달라 out-of-architecture로는 재학습 없이는 잘 옮겨가지 못합니다. score 기반 방법(STAC 등)은 시간적 일관성 같은 단일 스칼라 신호로 실패를 경고하지만, 어떤 실패 모드인지 진단 정보는 제한적입니다.

- **Core Contribution**: 이 논문은 VLA 제어를 perception–action 폐루프의 정보 파이프라인으로 보고, 성공/실패 롤아웃이 정보이론적 시그니처에서 체계적으로 갈린다는 관찰을 바탕으로 Triple Information-theoretic(Tri-Info) 신호를 제안합니다. Tri-Info는 (1) 행동 다양성, (2) 시간적 일관성, (3) 상태 전이와의 결합(액션-상태 커플링)으로 실패를 분해해, 경고를 넘어 해석 가능한 진단 대시보드를 제공합니다. 특히 각 신호는 임베딩 좌표의 기하에 덜 의존해 아키텍처와 환경 전이 일반화에 유리하다고 주장합니다.

- **Technical Challenges**: 핵심 기술 난점은 정보이론량(엔트로피/상호정보)을 VLA의 연속·고차원 임베딩에서 안정적으로 추정하고, 이를 실시간 탐지기로 결합하는 것입니다. 논문은 sliding window로 분포를 추정하고, MI/엔트로피는 k-NN 기반 추정기(예: Kraskov 류)와 Kozachenko–Leonenko 엔트로피 추정으로 계산한 뒤 z-normalization으로 도메인 간 비교 가능성을 확보합니다. 또한 각 Tri-Info 신호를 GRU로 시간 진화를 모델링하고, success 구간에서 점수 분포가 변하는 문제는 Functional Conformal Prediction으로 time-varying threshold를 만들어 해결합니다.

- **Empirical Impact**: 여섯 개 VLA 모델과 세 개 벤치마크 환경에서 Tri-Info는 in-domain에서 강력한 베이스라인과 동등하거나 그 이상 성능을 보이며, 특히 시간 예측 타이밍까지 앞서 실패를 더 일찍 감지합니다. 더 중요한 점은 cross-model, cross-environment, sim-to-real 전이에서 임베딩·score 기반 탐지기들이 붕괴하는 상황에서도 Tri-Info가 재학습 없이 성능을 유지한다는 결과입니다. 실세계 작업에서는 이전 detector들이 chance로 무너질 때 Tri-Info가 83% 정확도까지 도달해, 안전한 배치형 embodied AI에서 “해석 가능한 실패 경고”의 실용성을 강하게 시사합니다.



### MMD-SLAM: Structure-Enhanced Multi-Meta Gaussian Distribution-Guided Visual SLAM (https://arxiv.org/abs/2606.19874)
Comments:
          ICRA 2026

- **Prior Approaches**: 기존 Visual SLAM은 카메라 포즈 추정에 강점이 있으나, 지도는 point cloud나 voxel처럼 저해상도 표현에 머무르는 경우가 많아 AR/VR 및 Embodied AI의 고충실도 요구를 충분히 충족하지 못했습니다. NeRF 기반 접근은 디테일을 늘렸지만 over-smoothing과 계산 효율 문제를 겪고, 3D Gaussian Splatting(3DGS) 기반 SLAM은 렌더링 속도·품질을 개선했음에도 장면의 구조적 규칙을 충분히 쓰지 못해 맵 일관성이 흔들리는 한계가 있었습니다. MG-SLAM은 line 특징과 Manhattan World 가정을 활용하지만, discrete feature와 Gaussian primitive의 결합이 제한적이고 가정 자체가 적응성에 제약을 줍니다.

- **Core Contribution**: 이 논문은 Atlanta World(AW) 가정을 활용해 구조 정보를 지도에 더 적극적으로 반영하는 구조강화 Visual SLAM MMD-SLAM을 제안합니다. 추적(tracking)에서는 point–line 결합 제약으로 포즈 최적화의 안정성과 구조 단서를 확보하고, 매핑(mapping)에서는 AW 유도 Multi-Meta Gaussian(점/선/면 모달리티 및 지배 방향)을 통해 기하 구조를 명시적으로 인코딩합니다. 또한 Multi-Meta Gaussian evolution 전략으로 Weak/Stable 상태를 진화시키며 장면 기하에 맞춘 전역 최적화를 수행합니다.

- **Technical Challenges**: 핵심 난제는 (1) 포즈 최적화에 도움이 되는 line 구조 단서를 어떻게 3DGS의 Gaussian 원시(primitive)와 일관되게 결합하느냐, (2) AW 기반 구조 프라이어를 실제 데이터에서 어떻게 안정적으로 학습·적응시키느냐입니다. 저자들은 point 재투영에 더해 3D line segment를 추가 제약으로 넣고, pose 최적화의 global bundle adjustment에서 점·선 reprojection을 함께 최소화합니다. 매핑에서는 Gaussian을 점/선/면으로 범주화하고 지배 방향(dominant direction)과 shape/directon loss를 결합하며, Weak에서 Stable로의 진화·클론/스플릿/머지 연산으로 다중 장면 적응성과 기하 적합도를 동시에 노립니다.

- **Empirical Impact**: 실험은 RGB-D/대규모 합성 데이터에서 tracking 정확도와 포토리얼 맵 품질을 모두 평가하며, MonoGS 대비 ScanNet에서 ATE RMSE를 48.56% 줄이고 Replica에서 PSNR을 5.71% 개선하는 성과를 보고합니다. 정성 비교에서도 isotropic Gaussian만 쓰는 SplaTAM의 맵 홀 문제, 다른 베이스라인의 바닥/천장 복원 결함 등을 완화하며 구조 단서 복원이 더 잘 이뤄짐을 보여줍니다. 결과적으로 MMD-SLAM은 3DGS 기반 SLAM이 구조적 규칙을 지도 품질에 직접 반영할 수 있음을 실증하며, embodied perception용 고충실도 맵 구축에 의미 있는 진전을 제시합니다.



### World Engine: Towards the Era of Post-Training for Autonomous Driving (https://arxiv.org/abs/2606.19836)
Comments:
          Technical Report. Project Page: this https URL

- **Prior Approaches**: 기존 end-to-end 자율주행 정책은 대규모 fleet log로 평균적인 주행 능력은 크게 끌어올렸지만, 안전 경계는 오히려 데이터에 희소한 long-tail 안전사건에서 결정되는 문제가 남아 있습니다. 또한 현실에서 사고로 이어질 수 있는 상황을 대규모로 수집하는 데는 윤리·법·사회적 제약이 커서, 가장 중요한 학습 신호가 자연 데이터에 구조적으로 부족합니다. 따라서 단순히 pre-training 데이터만 늘리는 접근은 희귀한 실패 케이스에서 점점 수익이 줄어드는 diminishing returns가 나타납니다.

- **Core Contribution**: World Engine(WE)은 현실 로그에서 failure-prone long-tail 상황을 찾아내고, 이를 고해상도 상호작용 시뮬레이션으로 재구성·확장한 뒤 reinforcement 기반 post-training으로 정책의 안전을 개선하는 프레임워크를 제안합니다. 핵심 아이디어는 “현실에서의 위험한 탐색”을 직접 하지 않고도, 생성된 safety-critical 상호작용으로 정책을 학습시켜 안전 제약에 더 잘 맞추는 것입니다. WE는 nuPlan 기반 벤치마크와 더불어 프로덕션 규모 시스템에까지 적용해 성능 향상을 확인합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) long-tail 사건을 현실 로그에 기반해 발굴하는 것, (2) 발굴된 사건을 실제와 같은 관측으로 다시 만들고 closed-loop에서 학습 가능한 시뮬레이션을 제공하는 것, (3) 희귀 상황에서만 강하게 개선하면서도 흔한 상황 능력을 무너지지 않게 제어하는 것입니다. WE는 3D Gaussian Splatting 기반 simulation engine으로 장면을 photorealistic하게 재구성하고, diffusion 기반 behaviour world model로 주변 차량의 다양하고 제어 가능한 반응을 생성한 뒤, KL divergence 페널티를 포함한 behaviour-regularized reinforcement learning과 hard experience mining으로 post-training을 안정화합니다.

- **Empirical Impact**: 누Plan 기반의 공개 벤치마크에서 WE는 long-tail safety-critical 시나리오의 실패를 크게 줄이며, pre-training 데이터만 늘리는 스케일링 대비 더 큰 폭의 개선을 보였다고 보고합니다. 또한 프로덕션 규모 ADAS 개발 스택에서도 post-training 후 충돌이 최대 45.5% 감소하고, 200km 실주행 on-road 테스트에서 disengagement 0 및 안전 관련 개선을 확인했습니다. 결과적으로 ‘합성된 safety-critical 상호작용에 대한 post-training’이 안전을 더 확장 가능한 방식으로 끌어올리는 실용적 경로임을 시사합니다.



### Flow Map Denoisers: Traversing the Distortion-Perception Plane for Inverse Problems (https://arxiv.org/abs/2606.19802)
- **Prior Approaches**: 기존 영상 복원은 MSE를 최소화하는 회귀(conditional mean)는 과도한 블러를, posterior sampling 기반 방법은 더 그럴듯하지만 왜곡이 커지는 절충을 겪는다. 또한 distortion-perception(DP) frontier의 특정 한 지점만 목표로 하거나, 다른 지점을 얻기 위해 paired-data·보조 모델·샘플러 하이퍼파라미터 튜닝·재학습 같은 외부 장치가 필요했다. 일부 flow/diffusion 기반도 DP를 보려면 보간/이산화/재학습 등 추가 메커니즘이 요구되었다.

- **Core Contribution**: 이 논문은 flow map(average velocity를 평균 구간으로 예측하는 모델)이 단일 네트워크 안에 DP tradeoff를 가로지르는 “연속적인 denoiser 족”을 암묵적으로 포함한다고 주장한다. 특히 lookahead 파라미터 t를 조절하는 것만으로 MMSE(저왜곡)부터 posterior sampling에 가까운 고지각(고왜곡) 영역까지 자연스럽게 이동하며, Gaussian 타깃에서는 그 경로가 DP frontier를 정확히 복원됨을 보인다. Plug-and-Play(PnP) 솔버에도 같은 제어축을 넣어 일반 역문제에서 지각 정합(perceptual alignment)과 데이터 일관성(data consistency) 사이의 연속적 트레이드오프를 달성한다.

- **Technical Challenges**: 핵심 난제는 한 번 학습한 모델로 추론 시점에 DP plane의 여러 operating point를 ‘solver 선택 없이’ 연속 제어하는 방법을 찾는 것이었다. 저자들은 flow map의 평균 속도 기반 평균 denoiser(두 시점 denoiser)를 정의하고, t가 estimator의 성격을 연속적으로 바꾼다는 관점을 통해 제어 스위치로 작동함을 이론(Gaussian의 정확한 DP frontier 복원)과 구조적 성질로 뒷받침한다. 비가우시안/일반 역문제에서는 완전한 최적성 보장이 어렵지만, PnP에 내장해 단일 학습된 flow map이 양끝(저왜곡/고지각)을 모두 강하게 커버하도록 설계했다.

- **Empirical Impact**: CelebA(128x128)와 AFHQ(256x256)에서 여러 선형/비선형 복원 태스크(inpainting, motion deblurring, super-resolution, Gaussian deblurring)를 대상으로, 같은 모델에서 t를 스윕할 때 DP curve가 매끈하고 단조·볼록한 형태로 나타남을 확인했다. 특히 DP frontier의 양끝 영역에서는 전문화된 baseline을 맞추거나 능가하며, 그 사이도 연속적으로 추적해 기존의 이산적/휴리스틱 제어 방식 대비 실용성이 높다는 점을 강조한다. 결론적으로 한 번의 학습으로 DP tradeoff를 추론 중에 ‘제어 노브’처럼 다룰 수 있다는 점에서 computational imaging 및 inverse problems 커뮤니티에 의미 있는 확장성을 제공한다.



### Contour-Constrained Deformable Registration with Parameter Characterization for Head and Neck Surgical Guidanc (https://arxiv.org/abs/2606.19767)
- **Prior Approaches**: 두경부 편평상피암 수술에서 냉동 절편은 절제면 판정을 위한 표준이지만, 병리 양성 절제연을 실제 절제 바닥(resection bed) 위로 정확히 다시 위치시키는 일은 정렬 오차와 점막 수축 때문에 어렵다. 기존에는 수술 클립/태그 등 고정 기준만 쓰거나, 3D 스캔 모델을 AR로 표시해도 수술 중 연부조직 변형과 수축 때문에 정밀도가 떨어질 수 있다. 변형 정합도 시도됐지만, 표면 거리와 일부 랜드마크에 의존해 경계(boundary) 형상은 잘 맞지 않으며 절제 바깥으로 경계가 벗어나는 문제가 남았다.

- **Core Contribution**: 본 논문은 수술 후 나타나는 조직 변형을 생체역학 기반(deformable) 정합으로 보정해, 양성 마진을 절제 바닥 위에 재배치할 수 있게 하는 프레임워크를 제안한다. 핵심은 3D specimen mesh를 절제 바닥 point cloud에 맞추되, 경계 contour를 따라 경계의 수직(normal) 방향 일치 오차를 목적함수에 직접 패널티로 넣어 경계 측면의 어긋남을 줄인 것이다. Kelvinlet 기반 regularized deformable basis로 변형 가능 공간을 모델링해 물리적으로 그럴듯한 변형을 함께 유도한다.

- **Technical Challenges**: 관건은 (1) 정합이 표면에는 맞아도 경계는 벗어나는 문제, (2) 절제 후 점막 shrinkage와 변형의 크기 차이가 조직마다 달라지는 문제, (3) 경계 제약과 랜드마크/표면 정합 항의 상대적 중요도가 달라지는 문제를 동시에 다루는 것이다. 논문은 경계 contour 포인트들에 대해 경계 간 수직거리(perpendicular distance-to-agreement)를 직접 최소화하도록 제약을 추가하고, strain energy 정규화로 과도한 변형을 억제한다. 또한 표면 정합, fiducial(봉합 랜드마크), contour 제약, strain 에너지 가중치가 성능에 미치는 영향을 체계적으로 two-stage parameter search로 분석했다.

- **Empirical Impact**: 9개 표본(피부, 협점막, 혀)에서 rigid 정합 대비 mean TRE가 11.11±4.07 mm에서 deformable 정합만으로 8.20±2.68 mm로 26.19% 감소했으며, contour 제약을 추가하면 5.62±2.28 mm로 49.41%까지 추가로 줄었다. 특히 임상적으로 가장 어려운 혀(tongue) 표본에서 개선 폭이 가장 컸고, contour 제약은 모든 표본에서 TRE를 감소시키거나 유지했다. 파라미터 탐색 결과, 측방(lateral) 변형이 큰 조직에서는 contour weighting이 정확도를 좌우하는 경향이 뚜렷했으며, 알고리즘은 다양한 가중치 조합에서도 비교적 안정적으로 작동함을 보여 의미가 있다.



### GLARE: A Natural Language Interface for Querying Global Explanations (https://arxiv.org/abs/2606.19735)
Comments:
          16 pages, 2 figures

- **Prior Approaches**: 기존 XAI는 국소 설명(예: saliency map, concept bottleneck, counterfactual)처럼 개별 예측을 해명하는 방식이 주류였지만, 편향이나 전반적 추론 패턴 같은 “전역(global) 이해”에는 한계가 컸습니다. 전역 설명은 DNF 규칙·서로게이트·전역 중요 특징처럼 요약을 제공하지만, 대규모 모델의 경우 규칙/프로토타입이 너무 방대해 사용자가 탐색하기 어렵다는 문제가 있었습니다. 또한 전역 설명을 자연어로 직접 질의해 원하는 통찰만 뽑아내는 흐름은 충분히 자동화되지 않았습니다.

- **Core Contribution**: 이 논문은 전역 설명을 “정적 결과물”이 아니라 질의 가능한 데이터베이스로 바꾸는 LLM 기반 인터페이스 GLARE를 제안합니다. 사용자는 “어떤 클래스에선 어떤 특징 조합이 얼마나 필요한가”처럼 질문하고, 시스템의 핵심 LLM이 이를 구조화된 SQL로 변환해 로컬 Minimal Sufficient Explanations을 집계·필터링한 통계를 자연어와 시각 증거로 제공합니다. 특히 SQL 생성을 설명 템플릿(분석용 질의 패턴) 안에서만 수행해 포맷 오류와 환각을 줄입니다.

- **Technical Challenges**: 핵심 난제는 자연어 질문의 유연성을 유지하면서도, 전역 설명의 복잡한 집계 논리를 안전하고 검증 가능하게 실행 가능한 형태로 바꾸는 것이었습니다. GLARE는 (1) 설명용 SQL 중간표현을 미리 정의된 템플릿과 고정 스키마로 제한하고, (2) fine-tuning 시 SQL_START~SQL_END 구간만 학습하도록 fence masking(손실 마스킹)을 적용해 SQL 구조를 학습하도록 유도했습니다. 그 결과 모델이 데이터셋 고유 토큰에 과적합하지 않고, 다른 객체 어휘를 가진 데이터셋에도 질의 해석을 일반화할 수 있게 했습니다.

- **Empirical Impact**: ADE20K 기반 전역 설명에서 GLARE는 in-distribution 질의에 대해 구조적 정합성(fence detection/parse/execution)을 거의 완벽히 달성하며, 결과 일치율도 95%대(예: 95.2%)로 보고됩니다. 언어 교란(철자 오류, 동의어, 문장 변형 등)에 대해서도 강건성을 보였고, 특히 철자 오류에서 fine-tuned 모델이 regex 기준 대비 큰 폭으로 성능 우위를 보였습니다. 더 나아가 ADE20K로 학습한 뒤 Pascal VOC처럼 객체 어휘가 완전히 다른 데이터셋에도 retraining 없이 높은 정확도(약 90%대)를 보이며, “전역 설명의 실제 사용성”을 크게 끌어올리는 접근으로 의미가 있습니다.



### Efficient Neural Network Model Selection for Few-Class Application Datasets (https://arxiv.org/abs/2606.19712)
Comments:
          36 pages, 9 tables, 13 figures

- **Prior Approaches**: 기존 모델 선택은 주로 대규모 벤치마크에서의 성능을 기반으로 오프더셸, transfer learning, scaled model family, 모델 저장소 중 하나를 고르는 방식으로 이뤄졌다. 하지만 실제 응용은 10개 미만 클래스처럼 소수 클래스인 경우가 많고, 이때는 대규모 데이터에서의 성능 경향을 그대로 내려서 예측하기 어렵다는 한계가 있었다. 또한 분류 난이도를 임베딩/단일 샘플 관점에서 재는 연구들은 많았지만, 데이터셋의 클래스 관계(클래스 간/내 유사도)를 묶어 “애플리케이션 난이도”를 빠르게 계량하는 접근은 상대적으로 부족했다.

- **Core Contribution**: 논문은 데이터셋의 성질로 분류 난이도를 계산하는 경량 지표를 제안하고, 이를 통해 few-class(10개 미만 클래스) 환경에서 더 효율적인 모델을 선택할 수 있음을 보인다. 특히 few-class distinctiveness라는 현상을 제시하는데, 클래스 수가 작아질 때 정확도-모델크기 관계가 다르게 나타나 전통적인 선형/스케일링 가정을 무너뜨린다. 또한 공개된 모델 패밀리보다 더 작은 sub-model을 고려해, 비슷한 정확도를 유지하면서 모델 크기 효율을 높이는 방향으로 확장한다.

- **Technical Challenges**: 핵심 과제는 “클래스 수·intra-class similarity(같은 클래스 내부 유사도)·inter-class similarity(클래스 간 유사도)”가 실제 정확도 순위를 얼마나 잘 예측하는지, 그리고 어떤 조합이 가장 상관성이 큰지를 수치화하는 것이었다. 이를 위해 코사인 유사도를 기반으로 intra/inter-class 유사도를 정의하고, 여러 후보 난이도 수식들을 스피어만 상관(Spearman correlation)으로 비교해 예측력이 높은 구성을 찾는다. 이어 few-class 구간(Nc<10)을 기준으로 난이도 지표가 소수 클래스에서 모델 선택에 유용하게 작동하는지 실험적으로 검증한다.

- **Empirical Impact**: 실험 결과는 난이도 지표가 반복 학습·평가 없이도 모델/데이터셋의 성능 관계를 빠르게 비교하게 해, 최대 6~29배 속도 향상을 제공한다고 보고한다. few-class 모델 선택은 모바일 로봇, 드론, IoT 등 자원 제약 환경에서 효율 이득을 보여주며, 예로 YOLOv5-nano 대비 최대 42% 더 작은 모델로 유사 정확도를 달성하는 사례를 제시한다. 전반적으로 “데이터셋 성질 기반의 모델 매칭”이 작은 클래스 수 응용에서 성능 손실 없이 경량화를 가능하게 한다는 점에서 실무적 임팩트가 크다.



### BrainG3N: A Dual-Purpose Tokenizer for Controllable 3D Brain MRI Generation (https://arxiv.org/abs/2606.19651)
- **Prior Approaches**: 기존 3D brain MRI latent diffusion 파이프라인은 encoder–decoder tokenizer를 한 가지 reconstruction 목표로 같이 학습해 왔다. 그 결과 decoder의 해부학적 충실도에 유리해지면서, 조건 생성과 임상 태스크에 필요한 임베딩의 임상 정보가 희생되는 문제가 지적돼 왔다. 또한 많은 접근이 voxel 단위 재구성 지표 중심으로만 평가되어, “임상 표현성 + 생성 가능성”의 동시성을 검증하기 어려웠다.

- **Core Contribution**: 이 논문은 완전 부피(fully volumetric) masked-autoencoder(MAE) 기반 tokenizer를 제안해 encoder와 decoder를 분리한다. frozen 3D MAE encoder는 임상적으로 유의미한 임베딩을 만들고, 별도의 CNN decoder가 임베딩의 선형 projection에서 볼륨을 복원한다. 이렇게 얻은 단일 임베딩 공간을 conditional diffusion transformer(DiT)의 조건 입력과 downstream 임상 태스크에 동시에 활용할 수 있게 했다.

- **Technical Challenges**: 핵심 기술 난제는 3D에서 tokenizer가 (1) 임상 정보 보존과 (2) 해부학적 볼륨 충실도라는 상충 조건을 동시에 만족해야 한다는 점이다. 이를 위해 encoder는 MAE의 masked patch 예측으로 전역 해부학적 맥락을 학습하도록 고정하고, decoder는 bottleneck d′=32의 선형 projection을 받아 voxels를 재구성하도록 학습해 encoder drift를 차단했다. 이후 DiT는 flow matching과 adaLN-Zero 조건부 제어, classifier-free guidance로 질병/성별/모달리티/사이트/나이/IDH1 변수를 controllable하게 반영하도록 설계했다.

- **Empirical Impact**: 35,309개 볼륨(18개 코호트, 4개 모달리티, 200+ acquisition site)에서 선형 probing을 수행한 결과, frozen MAE encoder는 23개 태스크 중 21개에서 BrainIAC, BrainSegFounder, MedicalNet과 비교해 SOTA급 성능을 보이거나 동등했다. 또한 DiT 생성 샘플에 대해 real-data probe로 조건 복원력을 확인했으며, cond–null 신호가 변수별 probe 특성과 일치하는 형태로 나타났다. 더 나아가 동일 임베딩 공간을 재사용해 환자별 longitudinal forecasting에서도 나이 진행의 방향과 위치(예: 뇌실/고랑)를 재현하되, 크기는 약 27% 수준으로 감쇠되는 것으로 보고돼 임베딩의 “임상 표현성-생성 제어성” 연결을 실증했다.



### SAFE-Cascade: Cost-Adaptive Vision-Language Routing for Chart Question Answering (https://arxiv.org/abs/2606.19646)
Comments:
          Demo paper submitted at CIKM 2026. 4 pages, 2 figures

- **Prior Approaches**: 기존 ChartQA 계열 연구는 VLM이 차트 이미지를 직접 읽고 수치 추론까지 수행하는 방향에 초점이 맞춰져 왔습니다. 다만 질의마다 항상 VLM을 호출하면 비용과 지연이 커지고, OCR 텍스트로 충분한 문제에서도 시각 추론을 낭비할 수 있습니다. 선택적 예측·LLM 라우팅 연구도 있지만, SAFE-Cascade는 ‘모달리티(시각 필요 여부) 라우팅’ 자체를 사용자에게 투명하게 보여주는 데 차별점이 있습니다.

- **Core Contribution**: SAFE-Cascade는 OCR+텍스트 전용 경로로 먼저 답을 만들고, 라우터가 VLM 호출 필요성을 판단해 필요할 때만 visual grounding을 수행하도록 설계된 cost-adaptive chart question answering 시스템입니다. 핵심은 라우팅 과정을 숨기지 않고 OCR 증거, 텍스트 답안, escalation 확률, 최종 결정과 비용/지연 추정치를 함께 제공하는 ‘inspectable modality-routing’ 인터페이스입니다. 또한 임계값 슬라이더로 정확도–비용 프런티어를 직접 조절·탐색하게 합니다.

- **Technical Challenges**: 가장 큰 과제는 텍스트 경로에서 나온 ‘잠정 답’이 충분히 신뢰할 만한지, 아니면 VLM의 시각 기반 추론이 필요한지 오류 없이 분별하는 라우터의 정확도입니다. SAFE-Cascade는 Random Forest 라우터를 사용하고, OCR 길이·밀도, 질문 길이, 수치/비교 플래그, 질문- OCR 겹침, 답안 길이 및 불명/빈 답 지표, 숫자 답 존재, OCR 내 답 포함 여부, 텍스트 모델 추정 지연 등 inference-time 특징만으로 escalation probability를 추정합니다. 확률이 임계값 이상이면 VLM으로 escalate하고, 아니면 텍스트 답을 그대로 채택합니다.

- **Empirical Impact**: ChartQA-2500 홀드아웃 평가에서 SAFE-Cascade는 unified accuracy 69.1%로 full-VLM(67.7%)과 유사한 성능을 보이면서 VLM 호출률을 73.1%로 줄였습니다. 동시에 VLM 호출을 26.9% 감축하고 추정 비용을 9.3% 절감해, 정확도 손실 없이 비용 효율을 얻는 구도를 제시합니다. 또 텍스트 단독 경로(25.3%)와 단순 휴리스틱 개선(37.3%) 대비 성능 갭이 크다는 점에서, 단순 대체가 아니라 ‘언제 VLM이 필요한지’ 학습 라우팅이 이득의 중심임을 실증합니다.



### Scaling Self-Play for End-to-End Driving (https://arxiv.org/abs/2606.19641)
- **Prior Approaches**: 기존 end-to-end 자율주행 학습은 주로 오프라인 human demonstration 기반 behavior cloning(BC)에 의존해 왔습니다. 하지만 로그 데이터는 상태 커버리지가 제한적이고 학습 중 closed-loop 상호작용이 없어 배포 시 compounding error로 인해 취약해지기 쉽습니다. 이에 따라 시뮬레이터에서의 self-play/RL 또는 DAgger가 대안으로 연구됐지만, 대개 vectorized BEV 관측을 전제로 하거나 느린 시뮬레이터 때문에 대규모 학습 확장에 한계가 있었습니다.

- **Core Contribution**: 이 논문은 end-to-end 모델을 위해 pixels(센서 관측)에서 직접 self-play를 scalable하게 학습하는 전략을 제안합니다. 이를 위해 (1) 고처리량 batched 드라이빙 시뮬레이터 Gigapixel, (2) direct pixel-space self-play RL의 비효율을 줄이기 위한 self-play DAgger(privileged RL teacher로 on-policy distillation), (3) sim-to-real 격차를 perception만 가볍게 적응해 폐루프 주행 능력을 실세계에 옮기는 방식을 묶었습니다. 목표는 human trajectory supervision 없이도 closed-loop에서 강건한 주행 정책을 만드는 것입니다.

- **Technical Challenges**: 핵심 기술 난제는 두 가지입니다: (a) end-to-end 모델이 pixels를 처리하면 self-play RL에서 policy forward/backward 비용이 커져 샘플 효율이 급격히 악화된다는 점, (b) 기존 시뮬레이터가 end-to-end에 맞지 않는 vectorized BEV 관측만 제공하거나 처리량이 낮다는 점입니다. 논문은 RL을 pixel student에 직접 쓰지 않고, vectorized 관측의 가벼운 RL teacher가 생성한 궤적을 self-play DAgger로 distillation하여 계산·샘플 비용을 줄였습니다. 또한 Gigapixel은 photorealistic 시뮬레이션 대신 바운딩박스 기반의 bounding-box world를 GPU-가속 perspective rendering으로 변환해 필수 장면 구조는 유지하면서 50k agent steps per second 처리량을 달성하고, 마지막으로 perception adaptation만으로 sim-to-real을 매끈하게 연결합니다.

- **Empirical Impact**: 실험에서 Gigapixel self-play로 학습한 정책은 HUGSIM(closed-loop)에서 경쟁/상위권 성능을 보이며, NAVSIM-v2(pseudo-closed-loop)에서도 human trajectory supervision 없이 competitive한 결과를 보입니다. 특히 self-play 학습 스케일을 키울수록 정책 성능이 일관되게 비례적으로 향상되어, self-play가 end-to-end 학습에서 실용적이고 확장 가능한 데이터 생성 파이프라인임을 보여줍니다. 결과적으로 offline BC의 구조적 한계를 closed-loop 경험 생성과 distillation로 완화하면서, 향후 합성 경험을 통한 지속 개선 경로를 제시한다는 점에서 의미가 큽니다.



### FrequencyFormer: A Co-Designed Sensor-to-Processor Pipeline for Frequency-Domain Vision Transformer Inferenc (https://arxiv.org/abs/2606.19574)
- **Prior Approaches**: 기존 엣지 배포 연구는 ViT 연산량을 줄이기 위해 양자화, 프루닝, token reduction 같은 방법을 주로 다뤘습니다. 하지만 실제 시스템 병목은 센서에서 프로세서로 고차원 이미지를 보내는 에너지·대역폭 비용일 때가 많고, in-/near-sensor computing도 보통 최대 10× 안팎의 완만한 압축에 머무르는 한계가 있었습니다.

- **Core Contribution**: FrequencyFormer는 주된 병목을 ‘전송’으로 보고, 주파수 영역 표현을 센서 단계에서 토큰화해 데이터 이동량을 근본적으로 줄이려는 co-designed 파이프라인을 제안합니다. 멀티스케일 DCT 토크나이저를 기본 patch embedding(드롭인)으로 대체하면서도, pretrained 백본과의 호환성을 분류·검출·세그멘테이션까지 유지하는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 DCT 기반 압축이 정확도를 크게 훼손하지 않으면서도, 센서 하드웨어 제약(면적·전력·고정 연산)에서 효율적으로 구현되도록 만드는 것입니다. 논문은 (1) 멀티스케일 DCT와 선택 기반 Pruning, (2) harmonic-aware quantization으로 저주파는 고정밀, 고주파는 저정밀(INT4)로 비트 예산을 배분, (3) 고정 DCT 계수를 LUT로 하드와이어해 multiplier-free 토크나이징을 구현하고, (4) 수정된 MIPI 기반 저전력 통신으로 전송 에너지를 추가로 절감합니다.

- **Empirical Impact**: 실험에서 FrequencyFormer는 오프칩 데이터 볼륨을 최대 128× 줄이면서도 거의 기준선에 가까운 정확도를 유지했다고 보고합니다. 효율 측면에서는 28.8 TOPS/W의 높은 토크나이징 효율을 제시하고, 통신 에너지는 약 230× 감소, 센서 쪽 총 에너지는 2.22× 절감으로 ‘주파수 도메인 토큰화’가 인-센서 ViT 배포의 스케일러블한 토대가 될 수 있음을 실증했습니다.



### 3D-DLP: Self-Supervised 3D Object-Centric Scene Representation Learning (https://arxiv.org/abs/2606.19451)
Comments:
          ICML 2026. Project webpage: this https URL

- **Prior Approaches**: 기존 self-supervised object-centric 표현은 주로 2D RGB/영상에 머물러, 가려진 영역을 복원하거나 접촉 중심 조작에 필요한 정밀 3D 기하를 안정적으로 제공하기 어렵다는 한계가 지적돼 왔다. 3D를 다루는 방법도 point cloud/voxel을 그대로 쓰더라도, 색을 다루지 못하거나(in colorless), 렌더링 역문제에 의존하거나, 정책 학습에 바로 쓰기 힘든 거대·메모리 집약적 표현을 요구하는 경우가 많다.

- **Core Contribution**: 이 논문은 RGB-D 또는 (점유/색상) voxel 입력에서 장면을 객체 중심의 저차원 ‘3D latent particles’로 분해하는 3D-DLP를 제안한다. 각 particle은 3D keypoint 위치, bounding box 크기, appearance 특징 등을 disentangled 속성으로 담아, particle 단위로 해석 가능한 segmentation과 scene 편집(예: 물체 이동/스케일 조정)을 가능하게 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 희소·불연속적인 voxel 격자에서 안정적인 keypoint 후보를 뽑고, (2) 색상까지 포함한 3D 복원에서 gray collapse를 막아야 한다는 점이다. 이를 위해 appearance-aware K-means keypoint prior(색을 CIELAB로 변환해 공간-외형을 함께 클러스터링)와 occupied voxel에 한정한 chroma reconstruction loss를 도입하고, 3D spatial transformer/3D 합성(compositing)으로 particle을 end-to-end 복원 학습에 연결한다.

- **Empirical Impact**: 실험에서는 합성 데이터와 실데이터에서 latent 공간이 해석 가능하고 controllable하다는 점(particle 위치·스케일 조작 시 새로운 장면 생성)을 시각화·정성적으로 입증한다. 또한 이 compact한 3D particle 표현을 entity-centric diffusion 기반 로보틱 조작 정책에 연결했을 때, explicit 3D 정보가 없거나 dense 3D를 메모리 부담 없이 쓰기 어려운 기준선들보다 MimicGen과 언어 조건 RLBench 태스크에서 성능 우위를 보이며, self-supervised 3D 분해가 다운스트림 control로 ‘실제로’ 이어질 수 있음을 보여준다.



### 3D Scene Graphs: Open Challenges and Future Directions (https://arxiv.org/abs/2606.19383)
Comments:
          Invited article for the Annual Review of Control, Robotics, and Autonomous Systems Volume 10

- **Prior Approaches**: 3D Scene Graphs(3DSGs)는 기하 기반의 grounding 위에 의미·관계 추상화를 얹는 표현으로, 조작·내비게이션·작업 계획·장면 이해 등 다양한 응용에 쓰여 왔습니다. 다만 커뮤니티마다 3DSG의 정의, 구성 파이프라인, 평가 프로토콜이 달라 방법 비교가 어렵고, 공통 가정과 현실 배치에서의 잔여 과제를 파악하기 힘들다는 문제(단편화)가 지적됩니다. 기존 연구들은 노드/엣지 속성, 계층 구조, 동적 장면 표현, affordance-aware 확장 등 선택이 서로 달라 일관된 정리와 비교가 부족했습니다.

- **Core Contribution**: 이 논문은 3DSGs를 공통 정의로 정식화하고, 기존 formulation을 가르는 핵심 모델링 선택(노드·엣지 속성, 계층성, 동적 표현, affordance-aware 확장)을 체계적으로 분석합니다. 또한 원시 감각 관측으로부터 3DSG를 구성하는 방식에 대해 용어·관례·기법을 한데 묶어 정리합니다. 마지막으로 그래프 품질 같은 intrinsic 평가부터 작업 단위 성능 같은 downstream 평가까지 정렬해, 향후 로버스트한 real-world deployment 관점의 방향성을 제시합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 서로 다른 입력(비전/포인트 클라우드 등)과 서로 다른 모델링 선택이 결합되면서, 무엇이 성능을 좌우하는지 비교 가능한 형태로 고정하기 어렵다는 점입니다. 논문은 node/edge 속성 설계, 계층 구조, 동적 장면 표현, affordance-aware 확장처럼 선택지를 축으로 분류해 ‘어떤 가정이 들어가는지’ 추적 가능하게 만들고, 그래프 구성 파이프라인의 통일된 지도(map)를 제공합니다. 이를 통해 남은 공통 실패 원인과 연구 공백을 더 명확히 드러내려는 접근을 취합니다.

- **Empirical Impact**: 경험적 영향은 단일 알고리즘 제안이라기보다, 평가 전략을 intrinsic graph quality와 task-level performance로 나눠 커뮤니티가 동일 기준에서 비교하도록 돕는 데 있습니다. 논문이 제공하는 공통 정의·체계적 분류·리뷰는 재현성과 비교 가능성을 높여, 실제 배치에 필요한 로버스트성 이슈를 우선순위화하는 데 기여합니다. 또한 전용 웹사이트를 통해 조사 내용을 조직·확장할 수 있게 하여 후속 연구자들의 온보딩과 지식 축적을 촉진합니다.



### Full-Self Diagnostics (FSD): Physics-Grounded Visual Biomarker Inference from Smartphone Video via Inverse Problems and Operator Learning (https://arxiv.org/abs/2606.19372)
Comments:
          38,812 paired scans, preliminary longitudinal validation of multichannel visual glucose inference (MARD 17 to 46 percent across cohorts); physics plus information theory plus operator learning framework

- **Prior Approaches**: 기존 비침습 바이오마커 측정은 NIR 스펙트로미터, PPG 패치, CGM처럼 전용 하드웨어를 두고 단일 채널(예: 혈당-단일 스펙트럼, 맥-단일 신호)에 의존하는 경우가 많습니다. 이런 방식은 정확도는 나오더라도 기기/해상도/인구집단에 따라 성능이 쉽게 흔들리고, 정보가 단일 양상으로 제한돼 활용도가 낮다는 한계가 있습니다.

- **Core Contribution**: 논문은 스마트폰의 9초 무제약(fully unconstrained) 얼굴 영상에서 잠재 생리 상태를 복원하는 통합 수학 프레임워크 FSD를 제안합니다. 핵심은 방사전달 방정식 기반의 물리 순방향 모델, 다중 시각 채널의 정보이론적 관측가능성, 그리고 안정적인 역문제/연산자 학습/학습-업데이트 절차를 하나의 체계로 묶어 “카메라 영상이 임상적으로 의미 있는 바이오마커를 담는 이유”를 함께 증명하고 설계에 반영한 점입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 얼굴 영상이 조명·자세·개인차 등 잡음/교란을 포함한 상태에서 생리학적 상태를 역으로 복원할 수 있느냐입니다. 논문은 Tikhonov 정칙화로 역문제를 안정화하고, 다중 채널을 추가할수록 상호정보가 단조 증가한다는 관측가능성 정리와, 도메인 전이에서의 식별가능성(identifiability) 및 operator-learning 기반 일반화로 장치·해상도·집단 변화에도 무너지는 문제를 줄이는 방식으로 해결합니다.

- **Empirical Impact**: 38812개의 실제 페어 영상-바이오센서 스캔(59명)에서 실용 성능을 보였으며, 저자가 수집한 자가 데이터(glucose 35-550 mg/dL)에서는 MARD 29.86%, Clarke Error Grid에서 A+B 97.57%, 위험 구간 E는 0.27%에 그쳤습니다. 또한 당뇨 참여자의 70-180 mg/dL 대역에서는 MARD 17%까지 개선돼, “저가 스마트폰 카메라만으로도 비침습 바이오마커 추정이 가능”하다는 임상적 함의를 강화하고 더 많은 페어 데이터가 쌓일수록 성능이 O(1/sqrt(N)) 스케일로 예측 가능하게 향상된다는 점을 강조합니다.



### ProMUSE: Progressive Multi-modal Uncertainty-guided Staged Evidential Alzheimer Disease Classification (https://arxiv.org/abs/2606.19371)
- **Prior Approaches**: 기존 AD 분류 연구는 임상 데이터, 구조 MRI, PET을 모두 추론 시점에 항상 사용할 것을 전제하며 정확도 극대화에 집중해 왔습니다. 이 방식은 고가·비접근성인 PET/MRI 의존을 매번 강제해 실제 임상 워크플로에서 비용 부담이 큽니다. 또한 multimodal late fusion/ensemble은 불확실성을 직접 다루기 어려워, “언제 추가 촬영이 필요한가” 같은 비용-정확도 조절이 제한적입니다.

- **Core Contribution**: ProMUSE는 불확실도 기반으로 단계적으로 모달리티를 추가하는 progressive staged evidential 네트워크입니다. 먼저 저비용 임상 데이터로 evidential classification을 수행하고, Dirichlet 기반 subjective logic으로 불확실도를 계산한 뒤 임계값을 넘으면 MRI 또는 PET 특징을 점진적으로 결합합니다. 모달리티별 belief과 uncertainty를 Dempster-Shafer theory로 fusion해, 전체 모달리티를 쓰는 기준선과 견줄 만한 성능을 유지하면서 촬영 비용을 줄이는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 모달리티가 제공하는 증거의 양/품질을 불확실도로 일관되게 모델링하고, (2) 불확실도가 큰 경우에만 추가 모달리티를 선택하는 정책을 학습 가능한 형태로 만드는 것입니다. ProMUSE는 Softplus로 evidence를 생성한 뒤 Dirichlet 기반 subjective logic으로 uncertainty(미지의 증거량)를 계산하고, 모달리티 간 충돌을 Dempster-Shafer fusion의 conflict 계수로 반영해 결합 시 신뢰도를 조절합니다. 마지막으로 임계값 τ는 데이터와 태스크 특성에 맞춘 data-driven 절차로 선택하며, KL regularization을 포함한 evidential loss로 불확실도 보정까지 함께 학습합니다.

- **Empirical Impact**: ADNI, AIBL, OASIS의 CN-AD, CN-MCI, MCI-AD 세 태스크에서 ProMUSE는 full-modality baseline과 비교해 경쟁력 있거나 더 높은 정확도(Accuracy/F1/AUC)를 보이면서 MRI/PET 사용량을 50-90% 절감했습니다. 특히 CN vs AD에서처럼 절반 가까운 표본이 임상 데이터만으로 충분히 확정되고, 나머지는 필요할 때만 MRI/PET이 추가되는 방식으로 자원 절약이 관측됩니다. 결과적으로 현실적인 AD 스크리닝 파이프라인에 맞춘 uncertainty-aware, resource-efficient 진단 전략을 제시했다는 점에서 의미가 큽니다.



New uploads on arXiv(cs.AI)

### Toward Calibrated Mixture-of-Experts Under Distribution Shif (https://arxiv.org/abs/2606.20544)
- **Prior Approaches**: 기존 연구는 보정(calibration)을 확률 예측의 신뢰도와 실제 빈도가 일치하도록 맞추는 문제로 보고, 특히 전문가 수준에서 calibration을 강제하면 앙상블이나 MoE의 정확도/보정 트레이드오프가 개선될 수 있다고 알려져 왔다. 다만 MoE에서 라우팅(routing)이 분포를 바꾸는 방식—특히 distribution shift—에서 보정이 언제 깨지고 왜 회복되지 않는지는 충분히 규명되지 않았다. 특히 hard routing은 상대적으로 안정적일 것이라는 직관이 있었지만, soft routing이 만드는 더 미묘한 실패 모드는 명확하지 않았다.

- **Core Contribution**: 이 논문은 soft/hard routing에서 “라우팅 메커니즘이 보정에 미치는 조건”을 분해해, expert-level calibration이 전체 모델 보정을 보장하는 범위와 실패하는 범위를 정리한다. hard-routed 모델에서는 expert calibration이 비교적 넓은 class의 distribution shift에서도 전체 보정을 보장하지만, soft-routed 모델에서는 expert calibration만으로는 충분하지 않음을 보인다. 또한 이를 해결하기 위해 distribution shift 하에서 라우팅된 aggregate의 보정 오차를 직접 겨냥하는 adversarial reweighting을 제안하고, 이를 Robust MoE와 Robust Filtered로 구현한다.

- **Technical Challenges**: 핵심 난점은 soft routing에서 최종 confidence가 라우팅 가중치와 expert 예측의 많은 “구성(configuration)”을 한 값으로 뭉치며, training에서는 균형이 맞아도 test에서는 구성의 출현 빈도만 바뀌어 보정이 깨질 수 있다는 점이다. 즉 개별 expert가 자기 라우팅 가중치에 대해 잘 보정되어도, aggregate confidence의 level set 안에서 residual의 평균 정렬이 보장되지 않는다. 논문은 관측 가능한 신호로서 각 예측의 proper loss(고손실일수록 취약한 구성일 가능성)를 쓰고, entropy-balanced adversary로 학습 샘플에 재가중치를 주되 training 분포에서 “덜 멀어지도록” 설계하여 Robust MoE/Robust Filtered가 정확도–보정 트레이드오프를 개선하도록 만든다.

- **Empirical Impact**: 실험은 이미지/텍스트 태스크와 인위적·자연적 distribution shift를 포함해, 제안한 학습 목표가 평균적으로는 물론 어려운 데이터 부분집합에서도 accuracy–calibration tradeoff를 개선한다고 보여준다. 특히 soft routing에서 문제가 되는 라우팅-overlap 영역을 더 강하게 압박하는 방향으로 학습이 작동하며, 많은 경우 정확도 손실이 크지 않게 보정 성능을 끌어올린다. 이 결과는 MoE에서 “어떤 보정이 어떤 shift에 대해 안정적인가”를 이론적으로 정리하는 동시에, 그 실패 모드를 표적으로 삼는 실전형 robust training 레시피를 제공한다는 점에서 의의가 크다.



### How Do Instructions Shape Speech? Cross-Attention Attribution for Style-Captioned Text-to-Speech (https://arxiv.org/abs/2606.20532)
- **Prior Approaches**: 스타일 캡션 기반 TTS는 CapSpeech, VoiceBox, NaturalSpeech 3처럼 자연어로 음성 특성을 제어하지만, 문장 내 개별 단어가 어떤 음향 변화를 만드는지는 거의 해석되지 않았다. 기존 TTS의 attention 분석은 대부분 정렬(alignment) 시각화 중심이었고, diffusion/flow 기반 모델에서 cross-attention의 ‘조건 영향력’을 정량화하는 도구는 부재했다.

- **Core Contribution**: 이 논문은 speech diffusion/flow-matching TTS에서 DAAM( Diffusion Attentive Attribution Maps ) 아이디어를 cross-attention attribution으로 확장해, 캡션 토큰이 생성된 음성의 시간축에 어떻게 기여하는지 열지도(heatmap)로 추출한다. CapSpeech-TTS에 적용해 토큰별로 25개 레이어와 24개 ODE step 전체의 per-token temporal heatmap을 얻고, 이를 통해 스타일-콘텐츠-함수 토큰의 역할을 분해한다.

- **Technical Challenges**: 핵심 난제는 음성의 시간적 특성 때문에 이미지처럼 ‘공간’ 대신 ‘시간 프레임’에 대해 토큰별 영향도를 정의하고 집계하는 것이었다. 연구진은 각 레이어/step의 cross-attention 행렬을 head 평균 및 레이어·step 집계를 거쳐 1D 시간 열지도와 분포 통계를 설계(분산, PMR, 엔트로피)하고, 추가로 F0·energy와의 상관을 같은 시간축에서 계산해 음향적으로 정합성을 검증했다.

- **Empirical Impact**: 3,600개 (스타일 캡션, 텍스트 transcript) 조합에서 관찰된 결과는 스타일 토큰이 콘텐츠/함수 토큰보다 시간 분산이 크게 낮아(전역 조절 성격) global conditioning을 뒷받침한다. 또한 스타일 attention은 F0·energy와 유의미하게 상관하고, 중요도가 초기 ODE step과 특정 깊은 레이어(17 부근)에서 정점이며 엔트로피가 최솟값을 보여 네트워크가 스타일에 대한 선택성을 최대화하는 시점을 제시한다. TTS에서 cross-attention의 자연어 영향력을 정량 분석한 첫 연구로, 향후 실패 모드 진단과 controllability 개선을 위한 해석 가능성의 기준점을 제공한다.



### LedgerAgent: Structured State for Policy-Adherent Tool-Calling Agents (https://arxiv.org/abs/2606.20529)
Comments:
          Work in Progress

- **Prior Approaches**: 기존 tool-calling 에이전트는 사용자 대화, 도구 응답, 정책 문서를 프롬프트에 함께 쌓아두고 매 턴마다 필요한 ‘상태’를 모델이 재구성하도록 맡기는 방식이 많았다. 이 설계는 (1) 올바른 기록을 찾더라도 이후 의사결정에서 오래되었거나 누락된 상태에 근거할 수 있고, (2) 문법적으로는 유효한 write/tool 호출이라도 ‘현재 상태’에 의존하는 정책을 위반할 수 있다는 실패를 낳는다. 즉 도구 선택/콜 자체만이 아니라, 행동 경계에서의 상태-정책 정합성 확인이 취약했다.

- **Core Contribution**: LedgerAgent는 추론(inference) 시점에 관측된 작업 상태를 별도의 ‘ledger(장부)’에 명시적으로 유지하고, 이를 프롬프트에 렌더링해 모델이 상태를 검색/재구성하지 않아도 되게 만든다. 더 나아가 환경을 바꾸는(environment-changing) 도구 호출 직전에 policy gate로 ledger 필드 기반 정책 제약을 검증해, 위반 가능성이 있으면 호출을 막거나(revise/block) 피드백을 준다. 모델 가중치는 바꾸지 않고, 상태 표현과 행동 경계 검증 방식을 시스템 레벨에서 교체하는 것이 핵심이다.

- **Technical Challenges**: 핵심 난제는 상태가 대화 로그에 흩어져 있을 때 생기는 stale·missing·incorrect grounding 문제를 줄이고, 정책이 상태 의존적일 때도 write를 안전하게 걸러내는 것이다. 논문은 (1) tool 경로 맵에 따라 성공적인 read 도구 반환을 스키마-고정 typed dictionary로 저장하는 ledger를 만들고, (2) environment-changing 호출 제안마다 ledger를 근거로 실행 가능한 predicate 정책 게이트를 적용해 누락된 증거가 있으면 revise로 되돌리는 구조를 제시한다. 또한 write 직후에는 observe-not-assume 원칙으로 ‘다시 읽어 상태를 확인한 뒤’ ledger를 갱신해, 관측 기반 일관성을 유지한다.

- **Empirical Impact**: 네 가지 customer-service 도메인과 open/closed-weight 혼합 백본 모델에서 LedgerAgent는 표준 프롬프트 기반 도구 호출 대비 pass^k(특히 pass^4의 일관성 지표)에서 평균적으로 더 높은 성능을 보였다. 감소폭이 아니라 일관된 개선이 관찰되며, 다른 컨텍스트 엔지니어링 계열 방법(IRMA)보다 토큰 오버헤드를 늘리지 않으면서도 더 나은 결과를 보고한다. 실패 분석에서는 남는 오류가 대부분 누락된 필수 행동(missed actions)과 상태-인자 인식 오류(틀린 인자)로 나타나, 명시적 상태와 write-time 검증이 중요한 불안정성 원인을 줄였음을 시사한다.



### DeepSWIP: Quotient-WMC Counterfactuals for Neural Probabilistic Logic Programs (https://arxiv.org/abs/2606.20526)
- **Prior Approaches**: DeepProbLog, Scallop 같은 neurosymbolic PLP는 신경 모듈이 불확실한 원자에 대한 확률을 만들고, 논리 규칙이 이를 조합해 예측한다. 그러나 기존 추론은 기본적으로 연관(associational) 수준에 머물러 개입(do) 기반 반사실(counterfactual) 추론을 같은 의미론으로 다루기 어렵다. 반사실 의미론을 위해 Twin Network를 쓰는 접근이 널리 쓰이지만, Twin은 관계형 프로그램에서 복제를 유발해 계산·구현 비용이 커지고 “단일 세계(single-world)” 개입 직관이 흐려진다.

- **Core Contribution**: 이 논문은 DeepProbLog에 대해 단일 세계 반사실 의미론을 제공하는 DeepSWIP를 제안한다. 핵심은 neural materialization으로 신경 predicate의 확률을 일반 ProbLog 선택으로 바꾼 뒤, Single World Intervention Programs(SWIPs) 형태의 프로그램 수술을 적용하고, 동일한 변환 프로그램에 대해 weighted model counting(WMC) 몫(quotient)으로 반사실 조건부를 계산하는 것이다. 또한 이 결과가 materialized FCM(학습된 FCM)에 대해 조건부로 정확하다는 정당성(정확성 결과)을 제시한다.

- **Technical Challenges**: 기술적 난제는 “신경이 만든 확률”과 “개입의 causal semantics”가 섞일 때, Twin처럼 복제 없이도 반사실을 정확히 재현하는 변환을 설계하는 것이다. DeepSWIP는 신경 계산을 고정해 확률만 남기고, SWIP가 개입된 원자를 정의하는 메커니즘을 제거·고정값으로 리디렉션하는 방식으로 단일 세계 의미론을 구현한다. 이어 quotient-WMC 분석으로 개입 후에는 해당 메커니즘의 활성 불확실성이 사라지고(개입 cleaning), calibration 오차와 드문 증거(rare evidence)에서의 분모 불안정성이 왜 핵심 실패 모드가 되는지 설명한다.

- **Empirical Impact**: MPI3D 실험에서 DeepSWIP는 Twin 기반 구성과 12,000개 쿼리 전부에서 일치(정확성 검증)했고, Twin의 endogenous duplication을 피한 덕분에 추론이 2.14× 빨라졌다고 보고한다. SUMO HOV에서는 neural 확률 보정(calibration) 저하가 plug-in 반사실 추정에 편향을 유발함을 보여주며, 올바르게 스코프된 randomized-policy AIPW/DML 추정기가 모집단 평균 및 ATE에서 1차 편향을 대부분 제거함을 확인한다. 즉, 이 프레임워크는 “상징 추론의 정확성”과 “신경 확률 추정의 통계적 오차”를 분리해 진단하고 개선 전략까지 연결한다.



### FlowEdit: Associative Memory for Lifelong Pronunciation Adaptation in Flow-Matching TTS (https://arxiv.org/abs/2606.20518)
- **Prior Approaches**: 기존 TTS는 F5-TTS, Matcha-TTS, VALL-E처럼 zero-shot 품질은 뛰어나지만, 배포 후에는 발음 오류를 수정하려면 재학습이나 모델 편집이 필요했다. G2P 사전/lexicon 방식은 다언어 고유명사에 취약하고, fine-tuning이나 weight editing은 catastrophic forgetting, 음색(voice drift), 누적 간섭 같은 부작용 위험이 컸다.

- **Core Contribution**: FlowEdit는 frozen flow-matching TTS에서 가중치 업데이트 없이 발음 교정을 수행하는 life-long adaptation 프레임워크다. 텍스트 임베딩 공간에 token-level perturbation(잠재 편집)을 최적화하고, 이를 Modern Hopfield Network에 저장해 episodic memory로 재사용하며 inference 시에는 similarity gate가 결합된 soft attention으로 교정을 불러온다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 모델 가중치 변경 없이도 특정 단어 발음만 정확히 바꾸는 것과 (2) 세션 이후에도 교정이 유지되도록 설계하는 것이다. FlowEdit은 미분 가능한 conditional flow-matching 구조를 활용해 임베딩에 대한 gradient로 δ를 15초 내 최적화하고, adjoint sensitivity로 메모리 사용을 상수 수준으로 유지하면서 Hopfield 기반 content-addressable 저장/검색으로 zero forgetting을 보장한다.

- **Empirical Impact**: Polyglot-Nouns(312개 다언어 고유명사)에서 FlowEdit는 target-word Phoneme Error Rate(PER)를 zero-shot 대비 92.7% 상대 감소(3.1%)시키면서 general speech 품질은 그대로 유지했다. 또한 200회 연속 편집에서도 general PER이 기준선과 동일하게 유지되는 “0 forgetting” 특성을 보였고, 1개의 GPU에서 교정이 약 15초 내 수렴해 운영 친화성을 입증했다.



### Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages (https://arxiv.org/abs/2606.20517)
Comments:
          ICLR 2026

- **Prior Approaches**: 기존 코드 생성 벤치마크(HumanEval, MBPP, APPS 등)는 주로 Python 중심이며, 고정 스냅샷이라 학습 데이터 오염(contamination) 문제에서 자유롭지 않습니다. LiveCodeBench(LCB)는 경쟁 프로그래밍 문제를 지속 갱신하고 release date 필터로 오염을 완화하지만, 평가 언어가 Python에 한정돼 실제 소프트웨어 개발의 다언어 요구를 반영하지 못했습니다.

- **Core Contribution**: Multi-LCB는 LCB의 오염 통제와 평가 프로토콜을 그대로 유지하면서, 동일 작업을 12개 프로그래밍 언어로 확장한 다언어 코드 생성 벤치마크를 제안합니다. 특히 LeetCode의 Functional 포맷을 STDIN/STDOUT으로 자동 변환해, 언어 간 직접 비교가 가능하도록 만들었습니다.

- **Technical Challenges**: 핵심 과제는 Python 전용 Functional 포맷을 다른 언어에서도 동일한 채점 가능성으로 옮기는 것이었습니다. 이를 위해 자연어 프롬프트를 STDIN/STDOUT에 맞게 재구성하고, 공개/숨겨진 테스트까지 언어-agnostic 실행 하네스로 변환하는 파이프라인을 구성했으며, 단일 평가 엔진이 모든 언어를 동일 방식으로 컴파일·실행·채점하도록 했습니다.

- **Empirical Impact**: 24개 LLM을 Multi-LCB로 평가한 결과, Python 성능이 다른 언어 능력을 잘 대변하지 못하는 경향(파이썬 과적합)을 확인했습니다. 또한 언어별 contamination 신호와 언어 간 성능 격차(예: Python 평균 Pass@1이 가장 높고 Scala는 가장 낮은 수준)가 뚜렷하게 나타나, LCB의 Python-only 한계를 직접 보완하는 엄격한 벤치마크임을 실증했습니다.



### What Do Safety-Aligned LLMs Learn From Mixed Compliance Demonstrations? (https://arxiv.org/abs/2606.20508)
- **Prior Approaches**: 기존 연구는 in-context demonstration이 언어모델을 jailbreak하는 현상을 “효과가 있다”는 수준에서 주로 보여줬고, many-shot 맥락이 길수록 유해 요청 준수가 늘어남을 보고했다. 또 ICL을 컨텍스트가 잠재 상태/사전분포를 바꾸는 증거로 본다는 메커니즘·베이지안 관점이 제시됐지만, ‘순응(complaince)처럼 보이는 예시’가 실제로 무엇을 학습하는지는 불분명했다. 특히 benign(무해-요청/유익-응답)과 harmful(유해-요청/무거부-응답) 순응 예시가 서로 교환 가능한지에 대한 정리가 부족했다.

- **Core Contribution**: 이 논문은 benign compliance demonstration과 harmful compliance demonstration을 의미론적으로 분리한 뒤, 섞인(mixed) 데모 구성으로 이후 유해 쿼리의 compliance를 예측하는 통계가 무엇인지 가설검정한다. 총 개수만 보는 total-count, 유해 데모 개수만 보는 harmful-count, benign·harmful이 함께 영향을 주는 joint-count(증폭/희석 포함) 3가지를 체계적으로 비교한다. 그 결과 benign과 harmful 데모는 모델에 따라 단순 교환이 되지 않으며, 같은 실험에서도 모델마다 benign의 영향이 상반될 수 있음을 보여준다.

- **Technical Challenges**: 핵심 난제는 “표면적으로는 모두 ‘순응하는 듯한’ 데모”가 실제로는 다른 의미(유해 요청에 대한 비거부 포함)를 담고 있어, 컨텍스트가 무엇을 추출하는지 분리해 측정해야 한다는 점이다. 이를 위해 데모 풀을 구성해 문맥 내 benign/harmful 비율과 순서를 정밀 제어하고, refusal 여부는 WildGuard judge로 분류해 compliance rate를 산출했다. 또한 훈련 단계별 영향은 OLMo-3.1-32B의 SFT/DPO/Instruct 체크포인트 비교와 로지스틱 회귀(증폭/희석 계수)로 추적하고, 데모 응답의 접두(prefix) 복사(형식 채택)와 실제 유해 준수(behaviour)를 분리 측정해 메커니즘 차이를 드러냈다.

- **Empirical Impact**: 4개 모델(Llama-3.1-8B, OLMo-3.1-32B-Instruct, Gemma-4-31B-IT, GPT-OSS-20B)에서 total-count 가설은 전반적으로 기각됐고, benign 데모 효과는 모델 의존적으로 ‘희석/증폭/무영향’ 형태로 나타났다. 특히 preference optimization( DPO )이 benign 데모로 인한 harmful compliance 증가를 차단하는 결정적 단계로 확인됐으며, DPO 이전(SFT)에는 benign이 악성 준수를 키우는 경향이 관찰됐다. 추가로 데모 순서에는 강한 recency bias가 존재해 harmful 데모가 평가 쿼리에 가까울수록 준수가 증가했으며, 일부 모델은 거부 시에도 데모 포맷을 복사하는 반면 다른 모델은 거부 순간 in-context 신호를 거의 무시하는 등 거부 메커니즘이 질적으로 다름이 드러났다. 결과적으로 “데모 기반 jailbreak이 된다”를 넘어, 어떤 종류의 순응 데모·순서·훈련법이 어떤 내부 행동 추출로 이어지는지를 실증적으로 특성화했다.



### Context-Aware Hierarchical Bayesian Modeling of IVF Laboratory Environmental Conditions (https://arxiv.org/abs/2606.20459)
- **Prior Approaches**: 그동안 IVF 임신율 예측은 주로 환자 단위 변수와 시술 프로토콜 같은 임상 요인에 집중해 왔고, 실험실 센서 데이터는 주로 기준선 초과 여부 알림(compliance) 수준에서만 활용되는 경우가 많았습니다. 기존 연구들은 단일 변수의 통제 실험을 통해 영향 가능성을 보여주지만, 실제 진료 환경에서 나타나는 시간에 따라 변하는 변동(문 열림 후 회복 등)을 누적 효과로 포착하는 데는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 고해상도 배양기(incubator) 센서의 “시간적 역학”을 반영하도록 55개의 context-aware temporal features를 공학적으로 설계해, 단순 평균 집계보다 더 예측 신호를 잘 뽑아내는 접근을 제안합니다. 또한 아시아(소스)와 북유럽(타깃) 두 클리닉의 환경-임신율 관계를 계층적 Bayesian Beta regression으로 묶되, partial pooling으로 공통 효과는 공유하고 각 사이트의 baseline 차이는 유지해 전이 가능성을 높입니다.

- **Technical Challenges**: 핵심 난제는 (1) 원격 시간 지연(관측 시점의 임신율이 수 주 전 조건의 영향을 받음), (2) 사이트별 라벨링 해상도 차이(주간 vs 월간), (3) 북유럽의 소표본으로 인한 비식별/과적합 위험이었습니다. 연구팀은 결과 간격에 맞추기 위해 이전 기간(T-1) 요약을 lag로 포함하고, 주간/월간에 맞게 센서 변수를 집계하며, SHAP로 중요 변수 16개를 선별한 뒤 NUTS 기반 계층 베이지안 모델로 regularisation을 구현했습니다.

- **Empirical Impact**: 아시아 61주 데이터에서 맥락 피처는 단순 raw average(월 평균 4종) 대비 교차검증 예측 오차를 1.27%로 낮춰(3~5% 수준 대비) 시간 패턴의 추가 가치를 입증했습니다. 북유럽 보류 데이터에서는 35–39세 연령대에서 HM이 R2=0.86을 달성하고 naïve baseline 대비 64% 오차를 줄여, 환경 모니터링 신호가 클리닉 간에도 어느 정도 전이됨을 보여줬습니다. 다만 40+에서는 개선이 제한적이고, under-35는 월간 비율의 작은 환자 수로 잡음 지배가 커 결론의 불확실성이 남아 후속 데이터 확장이 필요하다는 점도 함께 강조됩니다.



### Interpretable Sperm Morphology Classification via Attention-Guided Deep Learning (https://arxiv.org/abs/2606.20438)
- **Prior Approaches**: 기존 연구들은 정자 형태 분류에 AlexNet류 전이학습이나 GAN 기반 개선을 적용하며 정확도를 끌어올렸지만, 대체로 모델 내부 의사결정을 설명하지 못하는 black box로 남는 경우가 많았습니다. 또한 수작업 판독의 주관성과 관찰자 간 일치도(60–70% 수준) 문제 때문에, 임상 현장에서 재현 가능한 자동화가 요구되지만 설명 가능성은 상대적으로 후순위였습니다.

- **Core Contribution**: 이 논문은 EfficientNet-B0에 Convolutional Block Attention Module(CBAM)을 결합해 정자 머리에서 진단에 중요한 영역에 집중하도록 유도하는 attention-guided 프레임워크를 제안합니다. 여기에 Grad-CAM++를 활용해 분류에 영향을 준 시각적 근거를 클래스별로 제공함으로써, 정확도와 interpretability(해석 가능성)를 함께 노립니다. 특히 HuSHem처럼 라벨이 적은 데이터에서도 잘 동작하도록 freeze-then-unfreeze 훈련 전략을 포함합니다.

- **Technical Challenges**: 주요 도전은 (1) 작은 의료 영상 데이터에서 대형 사전학습 모델이 배경·염색 잡음에 과적합하기 쉽고, (2) 성능만큼이나 임상의가 납득할 수 있는 근거(설명 가능한 모델링)를 제공해야 한다는 점입니다. 저자들은 EfficientNet-B0+CBAM 구조로 공간·채널 주의를 정자 머리로 유도하고, 작은 데이터에는 MixUp과 label smoothing 같은 정규화와 함께 freeze-then-unfreeze로 안정적 적응을 달성했습니다. 최종적으로 Grad-CAM++로 예측 근거 히트맵을 생성해 투명성을 강화했습니다.

- **Empirical Impact**: SMIDS와 HuSHem 두 공개 데이터에서 제안 모델은 각각 90.2% 정확도(매크로 F1 0.913), 93.9% 정확도(매크로 F1 0.948)를 달성하며 SimpleCNN과 표준 EfficientNet-B0를 능가했습니다. 특히 HuSHem에서는 EfficientNet-B0가 63.6%에 그친 반면 제안 모델이 93.9%로 크게 역전되어, attention 기반 집중과 단계적 fine-tuning의 효과가 데이터 규모에 강하게 반응함을 보여줍니다. Grad-CAM++ 시각화 결과도 정자 머리의 형태학적 특징(정상/비정상 경계, pyriform·amorphous의 변형 등)에 주로 활성화가 모여 임상 신뢰성 확보에 의미가 있습니다.



### Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recip (https://arxiv.org/abs/2606.20381)
Comments:
          18 pages, 12 figures

- **Prior Approaches**: FP4 저정밀 학습은 주로 E2M1(2-bit exponent, 1-bit mantissa) 기반 레시피에 맞춰져 왔습니다. NVFP4 같은 방식은 RHT와 stochastic rounding을 활용해 outlier를 완화하고 학습 안정성을 노렸지만, 여전히 E2M1의 비균일 그리드(표현 bin 기하)를 기본으로 둔 채 “증상”을 줄이는 데 머물렀습니다.

- **Core Contribution**: 이 논문은 E2M1처럼 비균일 포맷에서 발생하는 Shrinkage Bias(라운딩의 체계적 음의 오차)를 정식화하고, 이것이 층을 거치며 곱셈적으로 누적된다는 점을 이론적으로 설명합니다. 또한 RHT가 outlier를 퍼뜨리는 과정에서 E2M1의 가장 비대칭 bin으로 질량을 이동시켜, 기존 FP4 레시피의 학습 불안정을 한 가지 메커니즘으로 통합해 해석합니다.

- **Technical Challenges**: 핵심 과제는 “RHT를 쓰면 좋아질 것”이라는 직관이 E2M1에서는 깨지는 이유를, 포맷의 그리드 기하와 quantization 오차의 결합 관점에서 해결하는 것입니다. 논문은 uniform grid(예: E1M2/INT4)는 bin 비대칭을 없애 Shrinkage Bias의 기하 원인을 제거하고, 그 상태에서 RHT를 모든 GEMM 경로에 확대할 수 있게 만드는 UFP4를 제안합니다.

- **Empirical Impact**: Dense 1.5B, MoE 7.9B, MoE 124B 장기 pretraining에서 UFP4는 BF16 대비 손실 열화가 E2M1 기반 강한 베이스라인보다 일관되게 낮았고, scaling-law 분석과 ablation도 이를 뒷받침했습니다. 특히 full-RHT와 SR의 조합이 중요했으며, “하드웨어가 E2M1/E2M1-style에 묶여 있을 때” 단순 range restriction으로는 uniform-grid 동작을 충분히 대체하지 못한다는 점까지 확인했습니다.



### Automating SKILL.md Generation for Computer-Using Agents via Interaction Trajectory Mining (https://arxiv.org/abs/2606.20363)
- **Prior Approaches**: CUA(컴퓨터 사용 에이전트) 분야는 GUI를 click/typing/scroll/copy 같은 프리미티브 행동으로 모델링하고, 반복되는 루틴을 skill로 묶어 상위 계획에서 재사용하는 흐름이 많다. 다만 SKILL.md 같은 명시적 skill 라이브러리는 대체로 사람이 수작업으로 작성해 유지보수 비용과 바이어스가 크다는 한계가 지적된다. 또한 기존 skill-discovery/temporal abstraction 접근들은 ‘일관된(읽기 쉬운) skill’을 얻는 데는 성공해도, 그 skill 어휘가 다른 도메인/태스크에서 실제 정책 성능으로 전이되는지까지는 보장되지 않는다는 경고가 있다.

- **Core Contribution**: 이 논문은 GUI 궤적에서 SKILL.md 스타일의 명시적 skill 라이브러리를 자동으로 ‘채굴(mining)’하는 3단 파이프라인을 제안한다. GUI trajectory를 구간으로 분할한 뒤 구간을 클러스터링해 candidate skill을 만들고, 그 pseudo-label로 skill-aware 정책을 학습해 다운스트림 skill composition(다음 skill 시퀀스 예측)이 좋아지는지 검증한다. 특히 해당 방법이 “전이(transfer)”를 실제로 개선하는지 좁게 따져보는 진단(diagnostic) 연구로 설계된 점이 핵심이다.

- **Technical Challenges**: 기술적 난제는 (1) skill 경계가 ‘행동 변화’만으로 안정적으로 검출되는가, (2) 읽기 쉬운 클러스터가 정책에 필요한 순서 정보를 보존하는가, (3) offline reward model과 GRPO 최적화가 실제로 skill 구조를 유효한 학습 신호로 바꾸는가로 정리된다. 저자들은 인접 행동 변화의 change-point로 세그먼트를 자르고, 각 구간을 행동의 평균/분산(순서 비저장 bag-of-actions)으로 요약한 뒤 Bures/Wasserstein 기반 거리와 pseudo-label contrastive learning으로 임베딩을 정련한다. 그러나 이 구간 표현의 orderlessness가 다음 skill 예측과 transfer를 약화시키며, GRPO 학습은 데이터 불균형을 넘어서는 개선을 만들지 못했다고 보고한다.

- **Empirical Impact**: 실험에서 mined 클러스터는 소스 벤치마크 IW 기준으로 가독성은 높다(8개 중 5개가 labels 대비 purity 0.95 이상). 하지만 그 skill 라이브러리를 사용해 학습한 GRPO 기반 Qwen3-8B는 IW skill-step accuracy가 18.5%→20.5%로 소폭만 개선되고, BrowseComp+에서는 거의 변화가 없으며 WebArena에서는 오히려 감소한다. 게다가 Frequency 같은 단순한 빈도 기준이 주요 지표에서 learned MLP/GRPO보다 강해, “읽기 쉬운 skill 구조”와 “신뢰 가능한 cross-domain 정책 개선” 사이의 연결이 현재 파이프라인에서는 부족함을 실증적으로 보여준다.



### SoftSkill: Behavioral Compression for Contextual Adaptation (https://arxiv.org/abs/2606.20333)
- **Prior Approaches**: 기존 에이전트 스킬은 Markdown 문서로 정책·절차·검증 습관을 저장한 뒤, 추론 시 매 작업마다 언어모델이 긴 텍스트 아티팩트를 다시 ‘행동’으로 번역해야 한다. SkillOpt는 스킬 텍스트를 생성·검증하면서 외부 상태(읽히는 스킬)를 최적화하지만, 결국 모델이 받아들이는 제어 신호가 모델별·서빙별로 달라질 수 있다. 또한 LoRA 같은 파라미터 효율 튜닝은 성능은 강하지만, 스킬을 ‘작게 배포’한다는 관점에선 불리할 수 있다.

- **Core Contribution**: SoftSkill은 고정된 백본은 그대로 두고, 자연어 스킬을 초기화로 삼아 길이 m개의 ‘가상 토큰 임베딩’(soft prefix)을 학습해 연속적인 행동 우선분위를 만든다. 학습은 next-token prediction으로만 수행하며, soft skill은 추론 시 latent behavioral prior로 주입되어 긴 Markdown을 대신한다. 즉, 스킬을 다시 해석하게 만드는 대신 ‘작은 연속 제어(control)로 모델이 작업에 진입하는 방식을 압축’하는 접근을 제안한다.

- **Technical Challenges**: 핵심 난제는 짧은 prefix가 답변 형식·근거 사용·(에이전트의 경우) 도구 실행 같은 절차적 행동을 충분히 요약할 수 있는가이다. SoftSkill은 텍스트 기반 초기화(또는 SkillOpt 아티팩트/평균풀링) 후, 백본 가중치는 frozen 상태로 두고 soft delta만 NTP로 미세조정하며, 그 결과는 training loss가 아니라 held-out task 성능으로 체크포인트를 선택해 배치한다. 또한 에이전트 실행에서는 sparse trajectory imitation 신호의 한계가 드러나며, 이 boundary case를 별도 스트레스 테스트로 분리해 분석한다.

- **Empirical Impact**: 단일 라운드 QA(검색형·수학·문서 QA)에서 Qwen3.5-4B 기준 32-token SoftSkill prefix는 no-skill 대비 SearchQA +8.3p, LiveMath +42.1p, DocVQA +1.3p를 기록하고 SkillOpt 대비로도 SearchQA +5.2p, LiveMath +12.5p 개선을 보인다. 동시에 SkillOpt의 수백~수천 토큰 스킬 문서를 32개의 virtual token으로 압축해 컨텍스트 길이를 크게 줄이며, 생성 길이도 짧아지는 경향이 확인된다. 반면 도구 호출을 포함한 에이전트 실행(OfficeQA/Spreadsheet/ALFWorld)에서는 일부 과제에서만 이득이 있고, 장기 절차를 robust하게 압축·정교화하진 못해(성능 혼재) soft skill의 적용 경계가 분명해졌다.



### Leveraging systems' non-linearity to tackle the scarcity of data in the design of Intelligent Fault Diagnosis Systems (https://arxiv.org/abs/2606.20323)
- **Prior Approaches**: 기존 진동 기반 Intelligent Fault Diagnosis Systems(IFDS)는 FRF(Frequency Response Function)를 시간/주파수 신호에서 추출해 분류기 또는 이상탐지기로 처리하는 비모수(non-parametric) 기법이 널리 쓰였습니다. 다만 이런 방법들은 공학 모델·가정에 덜 의존하는 대신, 고장-비고장 간 차이를 뚜렷하게 만들기 위해 많은 데이터나 특정 조건이 필요해 데이터 희소성에 취약했습니다. 한편 딥러닝/Transfer Learning은 성능은 좋지만 labelled data가 많이 요구돼 실제 장비·구조의 희귀 고장에는 적용이 어렵다는 한계가 남아 있습니다.

- **Core Contribution**: 이 논문은 강한 데이터 희소 상황에서 동작하는 진동 기반 IFDS 설계를 목표로, 데이터 자체를 “이미지”로 바꾸는 새로운 시각화/증강 절차를 제안합니다. 구조의 고유 비선형성(예: dry-friction)에 착안해, 서로 다른 가진 진폭(level)에서의 FRF를 dB 크기로 모아 spectrogram-like 2D 컬러맵을 만들고, 이를 ImageNet으로 사전학습된 CNN으로 분류합니다. 또한 같은 고장에 대해 여러 가진 레벨·반복 측정을 섞어 더 많은 현실적인 컬러맵을 생성하는 커스텀 data augmentation을 도입합니다.

- **Technical Challenges**: 핵심 기술 난제는 “비선형 구조에서 FRF는 가진 진폭에 따라 달라지는데, 그 정보를 데이터 효율적으로 이미지로 압축하면서 모델이 과적합하지 않게” 만드는 것입니다. 논문은 비선형으로 인해 스펙트럼이 퍼지더라도 주요 조화 성분을 FRF 추정에 활용하고, 가진 레벨 축을 이미지의 축으로 포함해 비선형에 따른 차이를 시각적으로 보존합니다. 더 나아가 반복 측정에서 얻은 FRF 행들을 허용된 방식으로 재배열(swap)해 추가 이미지를 만들고, Mixup·Cutmix·RandAugment·Random Erasing 같은 일반 이미지 증강까지 결합해 희소 데이터의 효과를 끌어올립니다.

- **Empirical Impact**: 철도 팬터그래프(건식 마찰 비선형) 실험에서 비손상, 볼티드 연결 손상, 인공 댐퍼 제거의 3개 시나리오를 가진 7개 레벨과 반복 측정으로 구성해 검증했습니다. 제안한 파이프라인(MobileNetV2 사전학습 CNN + FRF 컬러맵 증강)으로 테스트 정확도 97.6%를 달성했으며, 특히 시간 재시작에 따른 cluster effect(실험 재설정 변동)도 상당 부분 극복했음을 보였습니다. 다만 볼트 손상 일부는 비손상으로 오분류되는 경향이 있어, ‘손상 종류별 FRF 변화의 강도’가 성능에 영향을 줄 수 있음을 시사합니다.



### Lagrange: An Open-Vocabulary, Energy-Based Sparse Framework for Generalized End-to-End Driving (https://arxiv.org/abs/2606.20274)
- **Prior Approaches**: 기존 end-to-end 자율주행은 종종 dense BEV/occupancy 같은 밀집 표현으로 기하 경계를 잘 담지만, 계산 비용이 커지고 의미를 ‘블랙박스’처럼 다루는 문제가 있었다. 반대로 sparse object query 기반은 효율적이지만 Car/Pedestrian/Cyclist 같은 closed-set 범주에 의존해 OOD(이상 상황)에서 탐지 누락으로 플래너가 위험을 놓치기 쉽다. VLA(vision-language-action)류는 open-vocabulary 추론을 제공하지만, 언어 토큰의 autoregressive·discrete 생성이 차량의 연속적 제어 제약(비선형 동역학·고주파 제어)과 충돌한다.

- **Core Contribution**: 본 논문은 open-vocabulary 의미 일반화와 동역학적 연속 제어를 동시에 만족하는 Lagrange 프레임워크를 제안한다. Lagrange는 VLM으로부터 class-agnostic(범주 비의존) 객체 제안을 연속 의미 토큰으로 만들고, Masked Latent Fields(MLF) 기반 intent-driven 마스킹으로 중요한 엔터티만 시간적으로 걸러낸다. 이후 이 토큰들을 공간 좌표 위의 implicit continuous energy field로 디코딩하고, 차량 궤적을 Lagrangian action minimization(최소 작용)으로 최적화해 kinematic feasible성과 충돌 회피를 함께 강제한다.

- **Technical Challenges**: 핵심 난제는 (1) OOD에서도 의미적으로 위험한 대상을 놓치지 않으면서 (2) discrete 언어 생성 대신 연속적 제어 가능한 플래닝을 설계하는 것이다. 논문은 이를 위해 VLM을 ‘토큰/의미 인코딩’에만 쓰고, 실제 제어는 에너지 장(energy field) 위의 연속 최적화로 분리한다. 또한 intent 상태를 GRU로 전개해 kinematic query를 만들고, masked cross-attention으로 무관한 시공간 엔터티를 제거해 계산 폭발을 막는다. 마지막으로 energy field에 동역학 페널티와 non-holonomic 제약을 포함하고, 해석적 적분이 어려운 문제는 MPPI(샘플 기반)로 수천 개의 동역학 롤아웃을 병렬 평가해 실시간 제약을 맞춘다.

- **Empirical Impact**: 실험에서 Lagrange는 OOD 롱테일 벤치마크 CODA에서 SparseDrive 대비 낮은 Out-of-Distribution Collision Rate(CROOD-C R)를 보이며, OpenVLA-Car의 큰 L2 궤적 오차 문제도 에너지 장의 공간적 구속으로 완화한다(대표 CROOD-CR 8.7%). 동시에 nuScenes 같은 closed-set 성능도 충돌률 0.25% 수준으로 유지/개선되어, open-vocabulary로 인한 품질 저하 트레이드오프를 줄였다는 점이 강조된다. 또한 nuScenes 학습 모델을 Waymo Open Dataset에 zero-shot으로 적용해 도메인 시프트 시 충돌률을 60% 이상 낮추고, 카메라 잡음/피드 드롭아웃 상황에서도 collision rate을 1% 미만으로 유지하며 해석 가능한 energy field heatmap까지 제공해 엔지니어링 관점의 검증 가능성을 높였다.



### Confidence-Aware Automated Assessment of Student-Drawn Scientific Models (https://arxiv.org/abs/2606.20264)
- **Prior Approaches**: 학생이 그린 과학 그림을 자동 채점하려는 시도는 있었지만, 대부분 단일 점수(또는 숙련도 라벨)만 제공해 “언제 믿어도 되는가”를 판단하기 어렵다. 또한 그림 평가는 사람 루브릭에 의존해 비용과 시간이 많이 들며, 복잡하고 애매한 시각 표현이 많을수록 신뢰성 확보가 더 어려웠다. 최근에는 ViT 같은 비전 모델로 표현을 직접 학습해 채점 가능성을 높였지만, 출력의 신뢰도 정보가 부족해 교육적 의사결정에 바로 쓰기 힘들다는 한계가 남았다.

- **Core Contribution**: 이 논문은 학생 그림 자동 채점에서 응답-단위 confidence를 함께 산출해, 신뢰 가능한 답만 자동 처리하고 애매한 경우는 사람 검토로 넘기는 “confidence-aware scoring”을 제안한다. confidence는 테스트 시 입력에 대해 의미 보존적(semantic-preserving) 변형을 준 뒤 예측 분포의 안정성으로부터 계산해, 최종 점수와 의사결정 기준을 연결한다. 이를 통해 채점 품질 향상뿐 아니라 교사가 자동 채점 결과를 어느 정도로 신뢰해야 하는지 실무적으로 판단할 수 있게 한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 그림 채점에 유효한 비전 표현을 효율적으로 학습하면서 (2) 같은 그림에 대해 예측이 얼마나 흔들리는지 정량화하는 confidence를 만들고 (3) 그 confidence를 선택적 자동 채점에 연결하는 것이다. 저자들은 ViT 백본을 LoRA로 parameter-efficient하게 적응시켜 학습 비용을 줄였고, 테스트 시 여러 증강 뷰의 예측 분포를 평균해 response-level confidence κ(x)를 정의했다. 또한 뷰 수준에서 더 신뢰할 만한 증강만 남기는 test-time selection with selective trust(Top-η)로 불안정 뷰의 영향을 줄여 선택적 집계의 견고성을 높였다.

- **Empirical Impact**: NGSS에 정렬된 중학교 과학 모델링 6개 문항(3단계 Beginning/Developing/Proficient)에서, 제안한 CA-Selective가 기준 방법들 대비 expert 점수와의 일치도를 높였다(예: Cohen’s κ 개선). 특히 confidence와 예측 정확도 사이에 유의미한 양의 상관(r=0.649, p<0.01)이 관찰돼 κ(x)가 채점 품질의 대리 신호로 실용적임을 뒷받침한다. 결과적으로 자동 채점의 신뢰성을 높이고, 자동 커버리지와 채점 위험(trade-off)을 임계값으로 조절할 수 있어 교실 환경에서 ‘책임 있는 교육 평가’에 의미가 있다.



### Navigating Unreliable Parametric and Contextual Knowledge: Explicit Knowledge Conflict Resolution for LLM Inferenc (https://arxiv.org/abs/2606.20245)
Comments:
          12 pages, 3 figures

- **Prior Approaches**: 기존 연구들은 지식 충돌을 대체로 “내부지식 vs 외부컨텍스트” 둘 중 하나를 신뢰하는 이분 선택으로 정리하려는 경향이 강했습니다. 최근의 dynamic 전략도 confidence score로 소스 전환을 하긴 하지만, 충돌의 원인 분석이나 상호 모순의 구조적 해소를 명시적으로 수행하진 못합니다. 결과적으로 외부 컨텍스트 내부의 모순이 있거나, 잘못된 정보가 모델의 오래된 내장 지식과 함께 확인 편향을 강화하는 상황에서 취약합니다.

- **Core Contribution**: MACR은 LLM 지식 충돌을 binary choice를 넘어서 “충돌을 해석하고 해소하는 메커니즘”으로 다루는 프레임워크를 제안합니다. 핵심은 (1) LLM의 불확실도를 정량화해 internal knowledge를 텍스트로 외재화하거나 필요 시 external knowledge를 검색하고, (2) 이를 기반으로 다중 에이전트 추론(Observer-Analyzer-Reasoner)로 모순을 규칙적으로 정리해 최종 답과 해설을 생성하는 흐름입니다. 또한 충돌 해소 결과를 해석 가능하게 제공하는 것을 목표로 합니다.

- **Technical Challenges**: 가장 큰 기술 과제는 “모델이 얼마나 믿을 만한가”를 제대로 추정해, internal vs external 중 어느 쪽을 우선 비교할지 결정하는 것입니다. MACR은 semantic entropy를 질문 관련성과 결합한 modified semantic entropy로 confidence를 측정하고, 날짜/주체 모호성 같은 교란 요인을 프롬프트에 temporal information과 subject disambiguation 신호로 보정합니다. 이후 internal 지식은 텍스트 맥락으로 외재화하거나 외부 증거를 retrieval로 구성해, 여러 컨텍스트 간 충돌을 Observer가 학습한 규칙 기반으로 Analyzer의 계층적 모순 탐지 결과에 적용해 일관된 결론을 만들도록 설계했습니다.

- **Empirical Impact**: 실험에서 MACR은 여러 벤치마크에서 state-of-the-art baseline을 유의미하게 능가하며, 특히 충돌이 존재하는 현실적 조건에서 정확성과 신뢰도를 동시에 개선하는 것으로 보고됩니다. 더불어 최종 답 도출 과정에서 어떤 충돌 유형이 어떻게 해결됐는지에 대한 해설을 제공해 해석 가능성 측면에서도 장점이 있습니다. 이는 retrieval-augmented generation에서 “오류를 회피”하는 것을 넘어 “모순을 해결”하는 접근이 실효적임을 보여주는 사례로 평가됩니다.



### A Multi-Agent system for Multi-Objective constrained optimization (https://arxiv.org/abs/2606.20236)
Comments:
          Presented at the 17th Workshop on Optimization and Learning in Multiagent Systems (OptLearnMAS, this https URL), co-located with the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)

- **Prior Approaches**: 계산·네트워크 의사결정 문제를 비용 최소화 + 성능 제약(QoS)으로 두면, 동적 환경에서는 RL로 런타임 최적화를 많이 시도한다. 이때 제약 위반을 페널티로 묶어 가중합 스칼라 보상으로 만드는 Lagrangian-inspired 보상 설계가 일반적이지만, 핵심 트레이드오프는 수동으로 고른 가중치에 좌우된다.

- **Core Contribution**: 이 논문은 MAMO(Multi-Agent system for Multi-Objective constrained optimization)를 제안해 “보상 가중치 선택” 자체를 학습 문제로 격리한다. Task-Execution(TE) 에이전트는 기존 방식의 가중 보상 RL로 행동을 학습하고, Weight-Adaptation(WA) 에이전트는 장기 지표를 관측해 가중치를 조정함으로써 비용 효율성과 제약 만족의 균형을 경험적으로 맞춘다.

- **Technical Challenges**: 기술적 난제는 수동 가중치 튜닝을 대체하면서도, 비정산 환경에서 제약과 비용 목표의 상대적 중요도가 변할 때 안정적으로 균형점을 찾는 것이다. MAMO는 학습을 2단계 반복으로 구성해(가중치 고정 후 TE 학습 → WA가 성능 요약 지표로 가중치 평가) 모델 내부를 차분(미분)하지 않아도 되는 형태로 바깥 루프에서 가중치를 갱신한다.

- **Empirical Impact**: 엣지 FaaS 단일 함수 replica scaling을 단순화한 실험에서, WA가 가중치를 재조정하며 평균 rejection 확률이 허용 임계치(예: 0.05) 근처로 수렴하도록 유도하는 결과를 보였다. 또한 노이즈로 부하가 흔들려도 제약 위반을 피하면서 실행 비용을 과도하게 희생하지 않는 적응성을 확인해, 제약 최적화에서 자율적 가중치 조정의 가능성을 시사한다.



### Thermodynamic Measure of Intelligenc (https://arxiv.org/abs/2606.20231)
- **Prior Approaches**: 기존 인공지능 지능 정의는 행동 중심(모방, 대화 구분 불가, 학습/추론/계획, 일반화, 압축, 보상 최대화, 과업 성공)으로 이뤄져 있어 기저 물리 연산을 공통 분모로 잡기 어렵습니다. Legg–Hutter의 기대 보상이나 Chollet의 ARC처럼 벤치마크·환경 평균 성능으로 평가하는 관점은 성능 비교에는 유용하지만, “지능을 만드는 보편적 경로 연산”을 직접 지칭하진 못합니다. 또한 일부 접근은 자유에너지/예측·추론(예: active inference) 등을 논하지만, 본 논문은 지능을 경로 확률이 어떻게 재가중되는가로 분리해 측정하려는 방향이 다릅니다.

- **Core Contribution**: 논문은 지능을 ‘희귀하지만 유효한 미래(rare-valid futures)의 확률을, 수동(passive) 기준 대비 법칙적으로 증폭하는 능력’으로 재정의하고 이를 rare-valid lift로 수치화합니다. 특히 시스템이 스스로를 세계 모델에 포함하고, 그 결과 자신의 행동을 포함하는 미래를 재귀적으로 self-simulation(재귀적 자기 시뮬레이션)해야 한다고 주장합니다. 핵심 정리는 재귀적 자기 시뮬레이션이 단순한 그럴듯한 특징이 아니라, 가정 하에서 높은 열역학적 지능(thermodynamic intelligence)에 ‘필요하며 거의 충분’하다는 연결을 제공합니다.

- **Technical Challenges**: 어떤 시스템이 ‘지능을 했는지’는 관측 가능한 경로의 확률 재가중으로 보이지만, 단순히 랜덤성이나 강한 구동만으로는 rare-valid lift를 만들 수 없다는 점이 기술적 난관입니다. 해결은 두 축으로 제시됩니다: (1) 증폭 예산이 제한된 bounded amplification 하에서, 희귀-유효 집합을 높은 fidelity로 찾아내지 못하면 높은 lift가 불가능하다는 ‘필요’ 정리를 세웁니다. (2) 반대로 높은 rare-valid fidelity와 함께 해당 영역을 실제로 증폭하는 효과적인 정책(embedded/available policy)이 모델 안에 존재하면, lift가 actuation-limited optimum 쪽으로 수렴한다는 ‘조건부 near-sufficiency’를 제시합니다.

- **Empirical Impact**: 프레임워크는 rare-valid lift를 열역학적 경로 측정(경로 공간에서의 확률 기하)과 결부해, 보상/성공 같은 태스크 스코어를 넘어 ‘보편적 스케일에서 지능을 측정’할 수 있다고 주장합니다. 구체 예시로 passive matter, 피드백 컨트롤러, 대규모 언어 모델과 인간 텍스트 생성기, 그리고 Maxwell-demon 같은 정보 엔진까지 동일한 수식적 틀로 적용 가능함을 보여줍니다. 즉, 지능을 ‘무엇을 잘하냐’가 아니라 ‘어떤 희귀-유효 미래를 얼마나 법칙적으로 확률화하느냐’로 바꾸어, 향후 모델 평가·이론화에 공통 측정 기준을 제공한다는 점에서 의의가 큽니다.



### QMFOL: Benchmarking Large Language Model Reasoning via Quantifiable Monadic First-Order Logic Test Case Generation (https://arxiv.org/abs/2606.20227)
- **Prior Approaches**: 기존 공제(연역) 추론 벤치마크는 수학/상식/시간/논리 등을 다루며, RuleTaker·ProofWriter·RobustLR·PrOntoQA-OOD는 정해진 템플릿으로 논리 구조를 만들고 LLM으로 문장화해 의미 다양성을 제한하는 경향이 있습니다. ProverQA는 theorem prover로 검증을 포함하지만, 형식 논리와 자연어 변환 사이의 일관성 보장이 충분히 명확하지 않았습니다. FOLIO는 사람이 설계해 일관성은 좋을 수 있으나 확장성이 떨어지고 인적 오류 위험이 남습니다.

- **Core Contribution**: 이 논문은 Quantifiable Monadic First-Order Logic(QMFOL)로, 연역 추론 문제를 monadic first-order logic 기반의 형식 구조로 자동 생성하면서 난이도를 depth, width, label type, distractor 수 등으로 정량·제어하도록 설계했습니다. 이후 LLM으로 해당 논리를 자연어로 번역하되, 외부 prover를 이용한 round-trip verification으로 논리적 일관성을 맞추는 절차를 포함합니다. 이를 바탕으로 QMFOLBench(총 2880 인스턴스, 960 구성)를 구축해 차원별로 성능 저하 양상을 관찰할 수 있게 했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 논리 복잡도를 한 가지 차원만 고립해 조절하기 어렵다는 점과 (2) 자연어로 옮기는 과정에서 hallucination이 생겨 형식 논리의 정합성이 깨질 수 있다는 점입니다. QMFOL은 conjunction/disjunction 패턴으로 MFOL(Monadic First-Order Logic) 구조를 만들고, 최소 규칙(minimal rule)·DAG 기반 확장으로 depth와 width를 분리해 키우는 방식으로 Challenge #1을 완화합니다. Challenge #2는 LLM의 생성→다시 FOL 환원→외부 prover 검증을 반복해 정답 라벨이 맞는 경우만 채택하는 round-trip verification으로 처리합니다.

- **Empirical Impact**: QMFOLBench 평가에서 논리 복잡도가 depth와 width에서 커질수록 정확도는 떨어지고 계산 오버헤드는 증가하는 경향이 확인되었습니다. 또한 True 라벨이 False·Unknown보다 성능이 좋고, distractor가 늘수록 정확도가 더 하락했으며 주제(topic) 변화만으로도 성능이 달라져 의미 의존성(semantic dependence)도 드러났습니다. 결과적으로 QMFOL은 언어 모델의 연역 능력을 ‘복잡도 차원별’로 더 정밀하게 측정할 수 있는 확장 가능하고 신뢰도 높은 벤치마크 제작 방법을 제시합니다.



### Augmenting Game AI with Deep Reinforcement Learning (https://arxiv.org/abs/2606.20210)
Comments:
          Vision paper, published in Conference on Games 2026

- **Prior Approaches**: 기존 game AI는 Finite State Machines(FSM), Behavior Trees(BT), GOAP처럼 사람이 규칙과 표현을 설계하는 방식이 주류다. 하지만 FSM/BT는 대규모로 커질수록 설계·유지보수가 어려우며, GOAP는 탐색 비용과 공학적 표현 의존성이 커진다. 또한 AAA 게임의 경로제약·NavMesh 중심 처리 때문에 이동과 맥락 반응이 경직돼 ‘현실감’이 깨질 수 있다.

- **Core Contribution**: 이 논문은 reinforcement learning(RL)을 game AI 제작 파이프라인에 ‘배치 가능한 형태’로 적용하기 위한 요구사항(짧은 학습시간, controllability, modularity, maintainability, bug detection 및 수정, authenticity, 런타임 추론 제약)을 프레임워크로 제시한다. 단순 성능 향상이 아니라 디자이너가 통제하고, 게임 업데이트에도 버티며, 실시간 예산 안에서 동작하는 에이전트로 확장하려는 관점을 강조한다.

- **Technical Challenges**: 핵심 난제는 (1) AAA 환경에서 학습을 짧게 끝내야 하는데 시뮬레이션 비용이 크고, (2) BT/FSM 등 기존 구조에 모듈로 끼워 넣어야 하며, (3) on-device/서버 제약 때문에 관측 처리와 추론 시간을 엄격히 맞춰야 한다는 점이다. 이를 위해 EA SPORTS FC 25의 골키퍼 포지셔닝은 SAC에 학습시간 단축 기법(offline 데이터, 시나리오 기반 학습, 업데이트-데이터 비율, 네트워크 리셋 등)을 결합했고, 런타임은 수십~수백 µs 수준의 작은 네트워크로 맞췄다. Battlefield 6의 보행/locomotion은 BT의 leaf node를 대체하는 모듈형 RL로 구성하되, raycast 대신 엔진의 heightmap 기반 occupancy map을 써서 관측 비용 병목을 줄였다.

- **Empirical Impact**: 실험은 두 AAA 타깃에서 RL-증강 AI가 더 ‘사람 같고(Authenticity)’ 플레이어가 받아들이는 행동을 낼 수 있음을 보여준다. 예로 EA SPORTS FC 25의 골키퍼는 기존 수동 시스템 대비 세이브 비율이 10% 높아졌고, Battlefield 6에서는 RL locomotion이 BT/GOAP 기반 경쟁 상황에서 유사한 성공률을 보이면서도 더 자연스러운 움직임을 보였다. 논문은 이러한 결과를 바탕으로, 게임 업계에서 RL 도입을 가로막는 병목을 짚고(표현/관측, 빠른 재학습, 디자이너 개입 도구, fine-tuning의 catastrophic forgetting 등) 향후 연구 방향을 제시한다.



### Beyond Accuracy: Measuring Logical Compliance of Predictive Models (https://arxiv.org/abs/2606.20208)
- **Prior Approaches**: 기존 평가는 ranking quality, prediction error, classification accuracy 같은 예측 성능 지표에 집중하며, 출력이 사전에 정의된 논리/도메인 제약을 따르는지 여부는 별도 점수로 다루기 어려웠다. 특히 의료·금융·자율주행처럼 높은 신뢰성이 요구되는 환경에서는 논리적 일관성이 예측 정확도만큼 중요하지만, 이를 표준적으로 측정하는 지표가 부족했다. 또한 규칙을 반영하는 평가를 하려 해도, 데이터나 모델이 표현하는 관계형(relation) 스키마에 맞춘 범용적 방법이 제한적이었다.

- **Core Contribution**: 이 논문은 Rule Violation Score(RVS)를 제안해 예측 정확도와 무관하게 모델 출력이 주어진 논리 규칙을 얼마나 위반하는지 정량화한다. RVS는 hard rules(엄격 제약)와 soft rules(통계적 규칙)을 다르게 취급하고, 어떤 데이터셋/어떤 관계형 어휘로 표현되는 예측 모델에도 적용 가능하도록 설계됐다. 더 나아가 RVS는 학습 데이터의 논리 일관성 평가와 규칙 정의 품질 점검에도 활용될 수 있다.

- **Technical Challenges**: 핵심 기술 난제는 “예측 성능”이 아닌 “규칙 준수”를 일반화해 계산하는 것으로, 관계형 문맥에서 규칙 위반 정도를 효율적으로 집계해야 한다. 논문은 Horn rules에 대해 SQL 쿼리를 자동 생성해 RVS를 계산하며, 이를 통해 데이터셋과 모델 종류(규칙 기반, embedding 기반, neuro-symbolic) 전반에서 동일한 방식으로 평가할 수 있게 했다. 또한 hard/soft 규칙을 분리해 제약의 성격에 맞는 위반 측정이 가능하도록 구성했다.

- **Empirical Impact**: 세 가지 벤치마크에서 RVS를 적용한 결과, 예측 정확도가 비슷한 두 모델이라도 논리 준수 수준은 크게 달라질 수 있음을 확인했다. 이는 기존 성능 지표가 포착하지 못하는 모델 거동 차이를 드러내며, 안전성이 중요한 응용에서 평가 관점을 확장한다는 의미가 있다. 결국 RVS는 지식그래프 link prediction과 relational regression에서 모델 선택·데이터/규칙 검증에 실질적인 기준을 제공하는 보완 지표로 자리잡을 가능성이 크다.



### Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifac (https://arxiv.org/abs/2606.20205)
- **Prior Approaches**: 기존 연구는 LLM의 성향을 인간 차별심리학의 자기보고 설문과 행동 과제로 “그대로” 측정해 심리 프로필을 만들었습니다. 그러나 LLM 응답이 프롬프트/문항 표현 변화에 민감하고 요인구조·행동 일치가 약하다는 의심이 누적돼, 프로필이 실제 특성(trait)인지 잡음/편향인지 해석이 흔들려 왔습니다.

- **Core Contribution**: 이 논문은 LLM 측정에서 나타나는 체계적 응답을 trait과 response bias(방향성 응답 편향)로 분해하는 정형(형식) 심리측정 프레임워크로 재해석합니다. 특히 문항 키잉(정방향/역방향)처럼 trait과 bias가 서로 반대 방향으로 작동하는지의 정도를 response orthogonality(응답 직교성)로 정의하고, 이를 중심으로 기존 “심리 프로필”의 타당성을 재검증합니다.

- **Technical Challenges**: 핵심 난제는 LLM 응답이 실제로 심리 특성에서 오는지, 아니면 문항 옵션·척도 방향 같은 형식에서 오는지 분리해내는 것입니다. 연구진은 IPIP-NEO-300(성격)과 Frey et al. 위험선호 배터리(성향) 29개 계측도구를 56개 instruction-tuned LLM과 1만여 명급 인간 기준표본에 동일 방식으로 적용하고, forward/reverse 평균 상관 및 분산 분해로 trait-편향 혼선을 정량화합니다.

- **Empirical Impact**: 결과적으로 LLM 간 차이는 81–90%가 response bias로 설명되고 인간은 9–16% 수준에 그쳤으며, capability를 키워도 bias는 완전히 사라지지 않았습니다. 또한 문항의 response orthogonality가 낮을수록 Cronbach’s alpha 같은 “신뢰도”가 부풀려지고, 직교성이 높아지면 신뢰도가 0에 가깝거나 음수로 붕괴했습니다. 더 나아가 프로필은 forward-키잉 vs reverse-키잉에 따라 크게 달라지며 문항 선택(item selection)만으로도 조작 가능해, LLM의 심리 프로필을 현재 방식 그대로 고정된 성질로 취급하면 위험하다는 메시지를 실증적으로 제공합니다.



### Implicit Semantic-Aware Communication Based on Hypergraph Reasoning (https://arxiv.org/abs/2606.20162)
Comments:
          This work is accepted at IEEE Transactions on Communications

- **Prior Approaches**: 기존 Semantic-aware communication(SAC)은 비트 단위 전송을 넘어 의미를 복원하는 방향으로 발전했으며, iSAC(implicit SAC)에서는 의미의 암묵적 구조를 그래프 기반 표현으로 추론하려는 시도가 많았다. 다만 기존 그래프 기반 iSAC은 주로 pairwise(쌍) 관계만 모델링해 다중 엔티티 상호작용 같은 higher-order 상관을 충분히 담지 못했고, 노이즈·결손 채널에서 의미 추론이 불안정해지는 문제가 있었다. 하이퍼그래프를 쓰더라도 단일 공유 임베딩 공간에서 고차 집계를 반복하면 over-smoothing으로 인해 서로 다른 의미 구분이 흐려져 성능이 더 떨어질 수 있다.

- **Core Contribution**: 본 논문은 hypergraph 기반 implicit semantic reasoning 프레임워크 HISR를 제안해, 다중 엔티티의 higher-order 관계를 더 풍부하게 표현하면서도 추론의 모호성을 줄이려 한다. 핵심 아이디어는 엔티티를 관계 유형별(relation-specific) semantic subspace에 투영해 이질적인 의미 상호작용을 분리하고, 채널로 인해 일부 정보가 손실되더라도 의미 복원을 견고하게 만드는 것이다. 또한 subspace 차원 구성을 eigen-gap 휴리스틱과 rate–semantic trade-off 관점에서 최적화해 표현력과 통신 효율의 균형을 노린다.

- **Technical Challenges**: 문제는 hypergraph의 표현력이 높아도, 공유 임베딩 공간에서의 반복 전파가 over-smoothing을 유발해 미세한 관계 뉘앙스를 구분하기 어렵다는 점이다. HISR은 이를 해결하기 위해 고차 메시지 집계를 단일 공간에서 수행하지 않고, 관계별 전용 부분공간으로 투영한 뒤 하이퍼엣지 단위의 relation-aware semantic encoder로 집계한다. 더 나아가 디코더에서는 수신 임베딩을 해당 subspace로 되투영하고 후보 엔티티 튜플의 semantic coherence를 positional-aware 점수함수로 평가해, 노이즈·부분 정보 불확실성에서도 신뢰도 기반으로 하이퍼엣지를 재구성한다.

- **Empirical Impact**: 실험 결과 HISR은 최신 iSAC 벤치마크 대비 implicit semantic interpretation 정확도에서 최대 36.6% 향상을 보였다. 또한 Gaussian 및 Rayleigh fading 채널에서도 개선 곡선이 관찰되며, subspace 투영을 통해 의미 클러스터의 분리 가능성이 커지고 추론 민감도가 완화됨을 보여준다. 이는 기존 pairwise 그래프의 표현 병목과 공유 임베딩의 over-smoothing을 동시에 겨냥한 구조적 접근이 의미 추론 신뢰성을 높일 수 있음을 시사한다.



### Modularity-Free Conflict-Averse Training for Generalized PINNs (https://arxiv.org/abs/2606.20156)
Comments:
          Accepted by ICASSP 2026

- **Prior Approaches**: PINN은 PDE 잔차와 경계 조건을 동시에 학습하는데, 두 목적 간 gradient conflict를 줄이기 위해 conflict-averse training(gradient 조작 계열)이 도입돼 성능이 개선됐다. 다만 기존 연구는 주로 gradient 간섭 완화에 집중했으며, 모델 용량이 커질 때 발생하는 더 근본적인 학습 경로 문제는 충분히 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 용량이 커질수록 conflict-averse 방식도 악화될 수 있음을 보이며, 그 원인을 objective-exclusive functional modularity(목적별 모듈 분리)로 규명한다. 이를 막기 위해 Modular-Sparsity Synchronization(ModSync)은 목적별로 연결이 ‘배타적’이 되는 경로를 구조적으로 억제하면서, 목적 간 상호작용을 촉진하는 연결은 유지하도록 설계됐다.

- **Technical Challenges**: 핵심 기술적 난제는 과도한 모듈 분리를 막으면서도, 잔차-경계 목적 사이의 상호작용 신호를 실제로 학습 과정에서 보존하는 것이다. ModSync는 DST(dynamic sparse training) 아이디어로 이진 마스크와 이중 임계치(잔차/경계용)를 학습해 구조를 동적으로 최적화하고, ‘공유 연결’은 덜 벌주며 ‘배타적 연결’에 sparsity 페널티를 걸어 coupling 붕괴를 완화한다.

- **Empirical Impact**: Helmholtz, Klein–Gordon, Burgers 등 다수 PDE 벤치마크에서 ModSync는 용량 스케일링 시 기존 conflict-averse 방법이 보이던 capacity-induced 실패 모드를 일관되게 억제하고 수렴 안정성을 유지했다. 또한 두 conflict-averse 베이스라인(예: DCGD, ConFIG)에 통합해도 성능이 유지/개선되며, 실험 전반에서 state-of-the-art 정확도를 달성해 PINN 학습의 ‘확장성 있는 신뢰성’을 강화했다.



### BIM-Edit: Benchmarking Large Language Models for IFC-Based Building Information Modeling (https://arxiv.org/abs/2606.20146)
- **Prior Approaches**: 기존 CAD/BIM 관련 벤치마크는 텍스트·이미지로부터 기하를 새로 생성하거나( from scratch ) 작은 합성 예제를 푸는 데 초점이 맞춰져 있었다. 또한 BIM 편집을 다루더라도 주로 기하적 일치만 평가하거나, 파라메트릭 CAD에서의 자연어 편집이더라도 공학적 의미(semantic)·관계(topology) 일관성 검증은 제한적이었다. 그 결과 실제 엔지니어링 워크플로에서 필요한 “기존 모델 해석→정확한 수정→구조/의미 보존”의 전 과정을 공정하게 측정하기 어려웠다.

- **Core Contribution**: BIM-Edit은 IFC(Industry Foundation Classes) 포맷의 기존 BIM 모델을 자연어로 편집하는 과제를 324개(큰 실제 모델 11개+합성 36개)로 정의하고, create/update/delete를 모두 포함해 평가한다. 편집 결과를 geometry(기하 정확도), semantic(의미 유효성), topological consistency(위상/관계 일관성) 3축으로 채점해 시각적으로 그럴듯해 보여도 구조적으로 깨지는 상황을 드러낸다. 또한 direct(직접 지시)·spatial(공간 지시)·topological(위상 지시)로 지시의 암묵성을 단계화해 실제 현장형 요청을 반영한다.

- **Technical Challenges**: 핵심 난점은 LLM이 단순히 형상을 그리는 것이 아니라, IFC 내부의 요소 타입·속성·관계까지 유지한 채로 필요한 부분만 안전하게 수정해야 한다는 점이다. 논문은 에이전트가 IFC 파일을 읽고 Python 코드로 IfcOpenShell 기반 조작을 수행하게 하며, 토큰/도구호출 제한 안에서 “컨텍스트 파악→편집 실행”이 되도록 평가 설계를 구성했다. 더불어 전체 모델을 비교하면 변경되지 않은 부분이 점수를 좌우할 수 있어, edit graph(추가/삭제/수정된 노드·관계)만 diff로 추출해 3축 점수를 계산해 편집 품질을 정밀하게 측정한다.

- **Empirical Impact**: 7개 LLM을 같은 코드 실행 하네스에서 실험한 결과, 최상위 모델도 3축 평균 점수가 49.48%에 그쳤고 3.4% 이상 완전해결한 모델은 없었다. 특히 create가 가장 어려웠으며, geometry는 상대적으로 더 잘 맞추는 반면 semantic과 topology는 크게 뒤처져 “부분적으로는 맞지만 엔지니어링 아티팩트로는 무효”인 비율이 높았다. 반면 장면이 복잡해져도(요소 수·관계 수 증가) 평균 성능이 크게 악화되진 않았는데, 이는 주된 병목이 규모 자체보다 ‘해당 IFC 컨텍스트를 찾아내고 구조적으로 유효한 편집을 실행하는 능력’에 있음을 시사한다.



### RACL: Reasoning-Agent Control Layers for Continuous Metaheuristic Learning (https://arxiv.org/abs/2606.20142)
Comments:
          10 pages, 5 tables

- **Prior Approaches**: 메타휴리스틱을 운영 환경에 반복 적용할 때, 기존 방식은 한 번 설정된 내부 탐색 동작을 그대로 쓰는 경우가 많습니다. hyper-heuristics나 adaptive operator selection, 강화학습·graph reinforcement learning 등은 검색 과정 관찰을 통해 제어를 시도하지만, 본 논문은 ‘추론 기반으로 제어 규칙을 가설-검증-통합-설명’하는 관점이 상대적으로 부족하다고 봅니다. 또한 학습 기반 최적화나 LLM 기반 휴리스틱 생성은 새 솔버/정책을 만들기 쉬운 반면, 고객의 business constraints를 보존한 채 기존 옵티마이저를 ‘제어 레이어’로 개선하는 실용 절차는 더 정립이 필요하다는 문제의식을 제시합니다.

- **Core Contribution**: RACL(Reasoning-Agent Control Layer)은 기존 metaheuristic 위에 reasoning agent를 얹어, 옵티마이저 자체를 대체하거나 business constraints를 수정하지 않으면서도 내부 search behavior를 제어하도록 설계한 방법입니다. 에이전트는 operational memory를 관찰·조회하고 과거 실행을 근거로 bounded hypothesis를 만들며, 개입을 시험한 뒤 guardrails로 위험을 관리하고 유용한 정책을 consolidate하며 결정 과정을 설명합니다. 즉, 특정 라우팅 규칙이 아니라 ‘메타휴리스틱 제어 규칙을 발견→검증→통합→설명’하는 절차 자체가 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) 실행 중 관찰 가능한 search state를 evidence로 삼아 meaningful control을 만들고, (2) 실패 시 피해를 줄이기 위해 개입을 bounded로 제한하며, (3) 메모리를 ‘정답’이 아닌 evidence로 재해석·업데이트하는 루프를 구현하는 것입니다. 논문은 observe→retrieve→reason→hypothesize→intervene→evaluate→guard→consolidate→explain→update memory 사이클로 이를 구성하고, 실험에서는 Codex를 in-the-loop reasoning agent로 써서 로그를 해석하고 live bounded intervention을 제안하게 했습니다. 이후 평가 재현성을 위해 policy proxy로 행태를 고정해 비교 실험을 가능케 했습니다.

- **Empirical Impact**: 차량 라우팅을 testbed로 한 실험에서 RACL은 feasible 21개 케이스 중 OMP(Operational Memory Policy)를 18/21에서 개선 또는 동률로 만들었고, STP(Stagnation-Triggered Policy)는 21개 중 18/21에서 개선 또는 동률을 보였습니다. 평균 비용 관점에서 RACL은 STP 대비 -0.641%, OMP 대비 -4.913%의 delta를 보고하며, Sevilla-9/10 paired sample에서는 Fixed 대비 -8.337%, STP 대비 -1.605%의 개선이 관측됐습니다. 또한 측정된 런타임에서는 material computational overhead가 거의 없었고, 에이전트의 business-readable 설명(정체 구간에서 개입 강도를 조절하고 이후 보수 모드로 안정화 등)이 재현 가능한 통제 정책으로 정리된 점에서 운영 관점의 의미가 큽니다.



### Learning to Prompt: Improving Student Engagement with Adaptive LLM-based High-School Tutoring (https://arxiv.org/abs/2606.20138)
- **Prior Approaches**: 기존 LLM 튜터링은 대체로 정적 prompting과 단일 과목(예: 수학) 중심 검증에 머물러 있었고, 시뮬레이션 기반 학습은 현실의 분포 이동과 데이터 희소성, 상호작용 복잡성을 놓치기 쉽습니다. 또한 prompt engineering이나 전역 최적화는 전형적으로 전 과목 공통의 고정 프롬프트로 수렴하는 경향이 있습니다.

- **Core Contribution**: 이 논문은 과목(topic/subject) 정보를 입력으로 받아 교육 전략을 선택하는 subject-aware prompt routing 프레임워크를 제안합니다. 동시에 14개 교육학적 관찰 기준(예: scaffolding, 이해도)을 분해해 산출하는 LLM evaluator 점수를 활용해, 과제 점수처럼 지연되는 신호 없이도 학습 효과의 즉시 대리지표를 만듭니다.

- **Technical Challenges**: 핵심 기술 난제는 시뮬레이터에서 학습한 라우터가 실제 학생의 행동에 과적합되는 sim-to-real gap을 줄이는 것이며, 이를 위해 점수 분포를 보정하는 sigmoid calibration과 score smoothing을 도입합니다. 또 사전학습 임베딩의 표현 붕괴를 막기 위해 topic embedding과 learnable subject ID embedding을 결합하는 dual-path 입력을 쓰고, 라우팅은 contextual bandit 형태로 PPO 기반 actor-critic(단일 스텝 학습)로 최적화합니다.

- **Empirical Impact**: 시뮬레이션 벤치마크에서 제안한 라우터는 정적 기준선 대비 더 높은 성능을 보였고(0.694 vs 0.647/0.64, p<0.001), 실제 온라인 A/B 테스트(N=656 대화, 359명)에서도 sim-to-real 전이가 관찰됐습니다. 특히 greedy 라우터는 conversion rate가 기준선과 유사했지만(19.1% vs 19.6%), stochastic 라우터는 conversion rate를 더 높였고(28.1%), 상호작용 턴 수는 약 3턴 줄이면서도 교육 품질을 유지했습니다(p=0.007).



### ScaffoldAgent: Utility-Guided Dynamic Outline Optimization for Open-Ended Deep Research (https://arxiv.org/abs/2606.20122)
Comments:
          9 pages, 6 figures

- **Prior Approaches**: 기존 OEDR( Open-ended Deep Research ) 시스템은 계획-후-작성(plan-then-write) 방식으로 아웃라인을 고정하거나, 이후 로컬 휴리스틱으로 부분 수정하는 경우가 많다. 이때 정보가 누적될수록 아웃라인이 증거 공간에서 어긋나 scaffold drift가 생기며, 수정의 효과는 더 나중에야 드러나 지연 피드백 문제가 나타난다.

- **Core Contribution**: ScaffoldAgent는 아웃라인을 정적 계획이 아니라 시간이 지나며 진화하는 구조적 scaffold로 두고, 이를 유틸리티(utility) 기반으로 동적으로 최적화한다. 아웃라인 변화를 Expansion(확장), Contraction(축소), Revision(수정)의 세 가지 구조 연산으로 명시화해 중복/부족/취약 구간을 통제적으로 업데이트한다.

- **Technical Challenges**: 핵심 기술 도전은 (1) 정보 축적 중에도 전역 위계가 흔들리지 않게 구조 진화를 제어(C1)하는 것과 (2) 수정 시점의 유용성을 즉시 추정해 지연 피드백을 줄이는 것(C2)이다. ScaffoldAgent는 각 연산의 downstream 가치(검색 이득, 구조 일관성, trial 생성 품질)를 종합한 utility 신호로 노드 선택(UCB 스타일), 연산 스케줄링, 종료 조건까지 추론 시 제어한다.

- **Empirical Impact**: DeepResearch Bench와 DeepResearch Gym 실험에서 ScaffoldAgent는 장문 리포트 품질(RACE 등)과 사실 근거(인용 기반 지표)를 함께 개선하며 기존 deep research agent들을 일관되게 앞섰다. 또한 유틸리티 차원 및 연산 연착(ablation) 결과, Expansion/Contraction/Revision과 retrieval-structure-generation 유틸리티 구성요소가 성능을 떠받드는 것으로 확인되며, multi-turn follow-up에서도 관련 부분만 비파괴적으로 갱신해 성능을 높였다.



### Multi-Head Attention-Based Feature Extractor Integration with Soft Actor-Critic for Porosity Prediction and Process Parameter Optimization in Additive Manufacturing (https://arxiv.org/abs/2606.20087)
- **Prior Approaches**: 금속 적층제조(metal AM) 공정 최적화는 일관성 확보가 핵심이지만, 회귀·앙상블 같은 데이터 기반 방법은 대규모 데이터가 필요해 비용과 시간이 부담된다. RL의 경우에도 기존 DQN 계열의 이산 행동공간은 정밀 최적해의 미세한 변화를 충분히 반영하기 어렵고, local optima에 갇히거나 수렴이 느릴 수 있다. 또한 표준 SAC/PPO/TD3는 탐색-활용 균형이 가치공간의 지형(로컬 미니마)에 맞지 않으면 학습이 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 레이저 파우더 베드 퓨전(Laser Powder Bed Fusion, L-PBF)에서 결함(특히 porosity)을 줄이기 위한 공정 파라미터 최적화를 목표로, continuous action space 기반 SAC에 multi-head attention feature extractor를 결합한다. attention이 저차원 입력의 미세한 변이를 더 잘 표현해 value space에서의 탐색-활용 균형을 개선하고, 로컬 미니마를 더 빠르게 빠져나가 전역 최적해에 도달하도록 설계한다. 비교 실험을 통해 DQN, PPO, TD3, vanilla SAC 대비 수렴 속도와 최종 성능을 함께 끌어올렸다고 주장한다.

- **Technical Challenges**: 핵심 난제는 (1) 연속적인 공정 파라미터를 다루면서도 (2) 작은 로컬 미니마가 존재하는 가치공간에서 빠르게 전역 최적해를 찾는 것이다. 연구진은 continuous action space로 미세 조정 여지를 확보하고, SAC의 쌍(dual) critic과 soft update/entropy 자동 튜닝을 유지하되, 입력을 다중 헤드 self-attention으로 재표현해 critic-actor가 value를 더 구별 가능하게 학습하도록 했다. 결과적으로 전역 최적화 과정에서의 수렴 지연과 국소해 민감도를 줄이기 위해 표현력(특징 추출)을 SAC에 직접 내장하는 방식으로 접근한다.

- **Empirical Impact**: porosity 예측 및 L-PBF 공정 파라미터 최적화에서 제안 방법은 다른 RL 기법들보다 더 빠른 수렴과 더 높은 최종 reward를 보였다고 보고한다. 특히 14 episodes 내 수렴(convergence value 322.79)을 달성했으며, 학습 전반에서 안정성도 유지되었다. 가치공간 추정 오차 감소, return 안정화 시점 이후 entropy 항의 역할(탐색 유지→국소해 회피) 등을 추적해 전역해 수렴 가능성을 뒷받침한다.



### Residual-Space Evolutionary Optimization via Flow-based Generative Models (https://arxiv.org/abs/2606.20084)
Comments:
          Accepted by ICML 2026 Workshop SPIGM, 5 pages, 3 figures

- **Prior Approaches**: 기존 생성 모델 기반 데이터 편집은 주로 미분 가능한 목적함수와 gradient 기반 탐색을 전제로 한다. 이미지 영역에서는 latent/feature/pixel 공간에서 최적화를 수행하는 연구가 많지만, flow 기반 편집은 forward/backward 적분 과정과 black-box(비미분) 목적이 섞여 이런 가정이 잘 성립하지 않는다.

- **Core Contribution**: 이 논문은 conditional flow matching(조건부 플로우 매칭)으로 분리된 residual(잔차) 공간을 “검색용 유전체”로 보고, residual-space evolutionary optimization(잔차-공간 진화 최적화)라는 모델-agnostic 프레임워크를 제안한다. 편집은 고정된 conditional generator 위에 가벼운 최적화 레이어로 얹혀, Lift(소스 조건 제거)–Land(타깃 조건 주입)를 반복 통합하는 대신 residual에서 mutation/crossover/selection으로 후보를 고른다. 또한 탐색–활용을 self-pollination(국소 정교화)과 cross-pollination(이종 잔차 재조합) 두 모드로 명시적으로 분해한다.

- **Technical Challenges**: 핵심 난제는 flow 기반 편집에서 objective가 비미분/black-box일 때도 target 정렬과 instance 보존, 다양성을 동시에 맞추는 탐색 설계를 만드는 것이다. 저자들은 CFM이 condition-controlled factor와 instance-specific residual을 분리한다는 관찰을 바탕으로, 모든 mutation/crossover/selection을 residual z_res에서 수행하고 조건 의미는 Lift/Land 적분으로만 주입하도록 역할을 분리했다. 그 결과 gradient 없이도 fitness(예: 분류기 신뢰도, 형태 속성 값, 비분화성 스칼라 지표)를 기준으로 진화를 안정적으로 수행할 수 있게 했다.

- **Empirical Impact**: 실험은 MorphoMNIST의 counterfactual generation과 crystal 데이터(WyCryst)에서 진행되며, 두 도메인 모두에서 validity(타깃 조건 충족)는 유지하면서 self-pollination은 source-instance similarity를 개선하고 cross-pollination은 다양성을 높이는 경향을 보인다. 특히 MorphoMNIST에서는 cross-pollination이 thickness 같은 비분화성 특성 상위 후보를 늘리면서도 다양성을 끌어올렸고, crystal에서는 latent diversity를 크게 확장하되 band gap 목표는 더 느리게 수렴했으며 일정 세대 이후에는 동일 수준의 validity에 도달했다. 논문은 이 탐색–활용 분해가 다양한 데이터 편집 설정(이미지→과학 도메인)으로 확장 가능하다는 실증적 근거를 제공한다.



### Process-Verified Reinforcement Learning for Theorem Proving via Lean (https://arxiv.org/abs/2606.20068)
- **Prior Approaches**: 기존 강화학습 from verifiable rewards (RLVR)는 주로 Lean 같은 정형 검증기가 주는 이진(성공/실패) 검증 신호에 의존해 보상이 희소해지기 쉽습니다. 반면 과정 기반 보상모델(PRM/PRM류)은 단계 단위 정답 라벨이나 대규모 데이터셋이 필요해 자동화와 확장성에 제약이 있습니다. 또한 Lean을 단계 검사기나 데이터 생성용 검증기로 쓰는 접근은 있지만, Lean의 tactic-level 피드백을 온라인 RL 학습의 ‘과정 보상 오라클’로 직접 변환해 쓰는 연구는 상대적으로 적었습니다.

- **Core Contribution**: 이 논문은 Lean proof assistant 자체를 symbolic process oracle로 사용해, 전체 정답 여부(outcome)와 tactic 수준의 세밀한 검증 피드백을 학습 중 보상으로 공급하는 프레임워크를 제안합니다. 생성된 proof 시도를 tactic 시퀀스로 파싱한 뒤, Lean의 elaboration이 ‘국소적으로 sound한 단계’와 ‘가장 처음 실패한 단계’를 표시하게 하여 dense하고 verifier-grounded한 credit signal을 만듭니다. 이를 GRPO 스타일 RL 목적함수에 결합하되, outcome-와 process-level 장점을 함께 균형 있게 반영하도록 설계했습니다.

- **Technical Challenges**: 핵심 난관은 Lean이 내놓는 트리 구조/심볼릭 피드백을 LLM의 autoregressive 토큰 학습에 맞는 토큰-level advantage로 신뢰성 있게 매핑하는 credit-assignment 문제입니다. 논문은 First Error Propagation 규칙으로 ‘첫 오류 이후 단계는 원리상 무효’라는 causal 제약을 반영하고, advantage를 각 tactic의 first token에만 부여하는 방식으로 token-level로 변환합니다. 또한 value model 없이 GRPO를 유지하면서도, 첫 실패 위치 기반의 tactic advantage를 outcome advantage와 함께 통합해 verifiable type-theoretic 신호에 뿌리를 둔 학습을 가능하게 했습니다.

- **Empirical Impact**: STP-Lean과 DeepSeek-Prover-V1.5에서 tactic-level supervision이 outcome-only baseline보다 대부분의 설정에서 우수했으며, MiniF2F와 ProofNet 같은 벤치마크에서 성능 향상을 확인했습니다. 예를 들어 STP-Lean에 결합한 실험에서는 MiniF2F 및 ProofNet 지표가 추가로 개선되며, RLVR 보상의 희소성을 tactic-level로 완화한 효과가 관찰됩니다. 더 나아가 이 연구는 심볼릭 정형 시스템이 평가 시 검증기일 뿐 아니라, 학습 중에도 신뢰 가능한 과정 보상 오라클로 확장될 수 있음을 보여주며 후속 연구에 방향성을 제시합니다.



### Autonomous Event-Driven Multi-Agent Orchestration for Enterprise AI at Sca (https://arxiv.org/abs/2606.20058)
- **Prior Approaches**: 기존 멀티에이전트 시스템은 주로 요청-응답 중심의 이산적 워크플로를 가정해, 이벤트 모니터링-탐지-실행이 연속으로 이어지는 엔터프라이즈 운영을 충분히 다루지 못했다. 또한 DAG Plan and Execute, ReAct 같은 구조는 에이전트 수가 늘 때의 오케스트레이션 병목(발견 잡음, 실패 누적)을 상대적으로 덜 체계적으로 다뤘다.

- **Core Contribution**: 본 연구는 DAG Plan and Execute와 ReAct를 208개의 프로덕션 유래 엔터프라이즈 시나리오(소규모 Persona, 중규모 Department, 대규모 Enterprise)에서 비교 평가하고, 엔터프라이즈 스케일에서는 task 복잡도보다 오케스트레이션 스케일이 성능을 좌우함을 정리한다. 아울러 연속 운영을 위한 Task Manager를 제안하며, 우선순위 추론, 관련 이벤트 병합, 그리고 preemption을 통해 고위험 이벤트 대응 지연을 줄이는 것을 목표로 한다.

- **Technical Challenges**: 연속 이벤트 환경에서는 에이전트 탐색 과정의 잡음이 누적되며, 에이전트 수가 커질수록 올바른 관련 에이전트를 연결하는 정합성이 급격히 떨어진다. 연구진은 Task Manager에서 우선순위를 추론해 큐를 재구성하고, related-event merging으로 중복·근접 이벤트를 한 흐름으로 묶어 정합성을 높이며, preemption으로 중요 작업이 대기열에 묶이는 시간을 단축한다.

- **Empirical Impact**: 실험 결과, 두 아키텍처 모두 소규모에서는 잘 작동하지만 Enterprise 스케일에서는 agent discovery noise가 지배적 병목이 되어 성능이 저하되며, 단순 작업이 더 큰 폭으로 악화됐다. DAG Plan and Execute는 소규모에서 더 높은 정밀도와 구조화된 병렬성을 보이지만 오버헤드가 커서 대규모에서 불리했고, ReAct는 실패를 단계적으로 처리해 더 견고했다. Task Manager는 high-priority 큐 지연을 14~75% 줄이고, Enterprise 스케일에서 related-event 정합성을 20%p 이상 개선해 엔터프라이즈급 연속 운영의 실효성을 입증했다.



### Reward as An Agent for Embodied World Models (https://arxiv.org/abs/2606.19990)
- **Prior Approaches**: 기존 RL 기반 world model post-training은 GRPO 같은 group-based policy optimization을 활용하지만, 분포 밖 탐험을 줄이기 위해 보수적인 롤아웃 전략에 크게 의존해 왔습니다. 이로 인해 action-space 탐색과 동역학적 다양성이 제한되고, 그 대신 학습 초기에 좁은 보상 패턴에 과적합되기 쉽다는 한계가 드러납니다.

- **Core Contribution**: 이 논문은 탐험 부족의 문제가 아니라 ‘검증(verification) 부재’가 핵심 병목이라고 지적합니다. 보상이 분포 이동 하에서도 신뢰 가능하지 않으면 정책이 reward hacking으로 프록시 보상을 최적화하되 실제로는 물리적 타당성·작업 성공을 개선하지 못할 수 있습니다.

- **Technical Challenges**: 이를 해결하기 위해 저자들은 Reward as an Agent(보상을 에이전트처럼 동작하는 계층형 평가기로 재설계)로 생성된 롤아웃을 물리 준수, 과업 완수 등 여러 차원에서 단계적으로 검증하고 오판 가능성을 줄이도록 설계했습니다. 동시에 DynDiff-GRPO는 동역학적으로 ‘중요한’ 구간에만 stochasticity를 배분해 장면 충실도를 유지한 채 동역학 탐색 다양성을 키우면서도 안정적인 GRPO 업데이트를 가능하게 합니다.

- **Empirical Impact**: PAI-Bench의 embodied robotics 평가에서 두 기준 world model 계열 모두에서 Domain Score 중심의 성능 향상이 확인됐고, 보상 시스템 자체도 AgiBotWorld-Beta 기반 평가에서 인간 라벨과의 정합도가 높았습니다. 특히 CPS처럼 단순히 잡음을 억제하는 접근 대비, DynDiff-GRPO는 롤아웃 다양성을 더 크게 유지하면서도 Quality(시각 충실도)는 크게 손상되지 않아 ‘더 넓은 탐험을 신뢰 가능한 검증 위에 얹을 때’ 스케일링이 가능하다는 메시지를 실증했습니다.



### ENPIRE: Agentic Robot Policy Self-Improvement in the Real World (https://arxiv.org/abs/2606.19980)
- **Prior Approaches**: 기존 덱스터스 로봇 조작 학습은 데이터 수집·평가(리셋 포함)·알고리즘 조정을 사람이 감독하며 진행되는 경우가 많아, 일반화된 물리 지능으로 가는 속도를 병목에서 끊어내기 어렵다는 문제가 제기됐다. 코드 생성형 coding agent 기반 autoresearch는 디지털에서는 성과가 크지만, 실제 로봇에서는 실행·검증·씬 리셋 같은 폐루프(실세계) 추상화가 부족해 그대로 확장되기 힘들다. 또한 로봇 fleet로 확장할 때는 어떤 가설을 효율적으로 시험·검증할지 선택과 자원 사용 효율을 동시에 맞추는 체계가 없었다.

- **Core Contribution**: ENPIRE는 실제 로봇에서 반복 가능한 피드백 루프—씬 리셋, 정책 실행, 결과 검증, 다음 반복 정제—를 자동화하기 위한 agent harness 프레임워크다. 구체적으로 Environment(EN)·Policy Improvement(PI)·Rollout(R)·Evolution(E) 네 모듈로 나눠, (1) 인간 주도로 자동 환경 API/검증을 만든 뒤 (2) 실세계 피드백으로부터 정책을 end-to-end 또는 다양한 학습 방식으로 자율 개선한다. 이 closed-loop를 통해 실조작 학습을 사람이 손으로 조율하는 엔지니어링 절차가 아니라, 통제 가능한 최적화 문제처럼 다루도록 설계했다.

- **Technical Challenges**: 핵심 기술 난제는 실세계 상호작용을 코딩 에이전트가 다룰 수 있는 형태의 자동 검증·리셋 인터페이스로 추상화하는 데 있다. ENPIRE는 안전 제약을 하드로 두고(위반 시 즉시 실패·리셋), 절차적 tool calls로 binary reward와 실시간 verifiability를 생성하며, contact-rich 작업은 어려운 구간의 시작 상태로 직접 되돌리는 리셋 스킬을 구성한다. 또 fleet 확장 시에는 병렬 실험에서 가설을 진화적으로 선별(Evolution)해 벽시계 시간을 단축하되, MRU/MTU 같은 자원 활용 지표로 비효율(로그 읽기·코드 작성·대기)을 관찰하고 균형을 재고한다.

- **Empirical Impact**: ENPIRE는 YAM 이중팔 6-DoF 로봇에서 pin insertion, ziptie-cutting, GPU-insertion, Push-T 등 덱스터스 작업에 대해 성공률을 고수준으로 hill-climb하며, 특히 pin insertion에서는 99%급 성공을 목표로 수렴 속도를 보인다. 또한 로봇을 1→8대로 늘릴 때 wall-clock time이 Push-T는 약 5시간→2시간, pin insertion은 1.5시간→약 40분으로 줄어드는 초기 스케일링을 보고한다. 다만 MTU와 토큰 효율은 8대에서 급격히 나빠져, 더 빠른 정책 개선을 위해 불균형한 토큰 예산이 필요하다는 트레이드오프도 함께 드러내며 실용적·확장 가능한 물리 autoresearch 방향성을 제시한다.



### Advancing DialNav through Automatic Embodied Dialog Augmentation (https://arxiv.org/abs/2606.19948)
Comments:
          29 pages, 9 figures

- **Prior Approaches**: DialNav는 원격 Guide와의 dialog-navigation loop로 모호한 지시를 해결하는 vision-and-language navigation(VLN) 과제를 제시했지만, 2K episodes 수준의 데이터 부족이 성능 병목으로 지적됐다. 기존 방법은 RAIN 같은 소량 인력 생성 데이터를 기반으로 모듈을 분리 학습한 뒤 통합했으나, 학습 프레임이 dialog의 동적 전개를 충분히 반영하지 못해 일반화가 제한된다.

- **Core Contribution**: 이 논문은 DialNav 학습용 대규모 데이터셋 RAINbow(238K episodes)를 자동 생성 파이프라인으로 구축한다. 또한 RAINbow의 효과를 끌어내기 위해 Dual-Strategy Training(DST)로 데이터-유도 롤아웃과 on-policy 롤아웃을 함께 학습하고, remote Guide의 핵심인 localization 성능을 Graph-based Transformer Localization으로 보강한다.

- **Technical Challenges**: 가장 큰 난제는 멀티턴 dialog를 포함한 학습 데이터를 사람 비용 없이 만들되, 질문-답변이 실제 경로와 정합되도록 “접지(grounding)”를 유지하는 것이다. 논문은 VLN 경로를 2–4개 concatenation하고, 각 dialog point에서 장면 캡션을 질문으로 생성한 뒤 LLM으로 자연스러운 multi-turn dialog로 reformat하여 데이터 품질과 비용(episode당 약 0.0016 USD)을 동시에 확보했다.

- **Empirical Impact**: 실험에서 모델은 Val Seen SR 58.24, Val Unseen SR 29.05로 기존 baseline 대비 각각 +89%, +100% 개선되어 success rate가 사실상 두 배 수준으로 상승했다. 구성요소별로 RAINbow 단독 이득은 제한적이었지만, DST와 localization 개선이 결합되며 큰 폭의 성능 향상이 나타나 “대규모 합성 데이터 + dialog 루프 정렬 학습 + localization 보강”이 핵심임을 입증했다.



### PhysDrift: Bridging the Embodiment Gap in Humanoid Co-Speech Motion Generation (https://arxiv.org/abs/2606.19935)
- **Prior Approaches**: 기존 음성-동작 동시 생성(co-speech motion generation)은 주로 SMPL-X 같은 사람 중심 표현으로 먼저 생성한 뒤, inverse kinematics나 최적화 기반 retargeting으로 휴머노이드에 옮기는 파이프라인을 사용해 왔다. 이 방식은 시각적 품질은 좋지만, 사람 모션 manifold와 로봇의 실행 가능 motion manifold가 다르다는 “embodiment gap”이 누적되며, 특히 발화 prosody와 동작의 미세한 시간 동기화가 약해지는 문제가 생긴다.

- **Core Contribution**: 이 논문은 휴머노이드 음성-동작 생성에서 핵심 실패 원인이 단순한 kinematic feasibility 부족이 아니라, 인간 표현→로봇 실행으로 옮기는 과정에서 생기는 분포 불일치(embodiment gap)임을 분석적으로 정식화한다. 이를 해결하기 위해 두 단계 기여를 제안한다: prosody를 보존하면서 로봇 실행 가능성까지 고려해 로봇 네이티브 감독 데이터를 만드는 IK-EER, 그리고 그 데이터를 바탕으로 로봇 관절 공간에서 speech-to-motion을 직접 예측하는 PhysDrift다.

- **Technical Challenges**: 로봇 네이티브 생성의 가장 큰 기술적 난점은 “표현력”만 키우면 잡음 성격의 동작 고주파 성분이 모터/관절 한계를 넘어 unstable motion과 jerk(급가속/급변)로 이어진다는 점이다. 저자들은 IK-EER에서 kinematic 제약과 발화-동작의 temporal alignment를 동시에 최적화(Energy Envelope 기반 prosody-motion 동기화 + joint-limit/foot-contact 등 physical regularization)해 데이터 단계에서 embodiment consistency를 고정하고, PhysDrift에서는 flow matching/확산처럼 velocity transport를 그대로 적분해 생기는 생성-실행 공간 불일치를 “드리프트 필드(MDF)” 형태로 재정의해 실행 시 물리 안정성을 강화했다.

- **Empirical Impact**: 실험 결과, 제안 방식은 speech-motion synchronization(발화 리듬-동작 타이밍)을 더 잘 유지하면서도 물리적 실현 가능성, 동작 smoothness, 그리고 추론 효율을 동시에 개선하는 것으로 보고된다. 또한 실제 휴머노이드 로봇 배치(real-world humanoid deployment)에서 real-time co-speech 상호작용 능력이 향상되어, 디지털 휴먼 중심 파이프라인을 휴머노이드 실행으로 옮기는 데 필요한 간극을 실사용 관점에서 메웠다는 의미가 크다.



### The Tao of Agency: Autotelic AI, Embedded Agency and Dissolution of the Self (https://arxiv.org/abs/2606.19924)
- **Prior Approaches**: 기존 AI는 목표를 디자이너가 외생적으로 정하고, 알고리즘은 그 목표를 달성하는 방법만 탐색하는 구조(MDP, 지도학습, RLHF 등)를 중심으로 발전해 왔다. 이 접근은 성공적이었지만, 보상함수·손실·선호 데이터가 열려 있는 문제에선 쉽게 부서지거나(reward misspecification) 의도와 다른 방식으로 최적화되는(reward hacking) 한계가 드러난다. 자율적으로 목표를 “발견”하는 오토텔릭(autotelic) 문제를 다루려는 시도도 있었으나, 목표 공간의 정당화 근거(무엇을 목표로 삼을지)가 끝까지 설명되지 않는 경우가 많다.

- **Core Contribution**: 논문은 오토텔릭 AI의 핵심 난제를 “목표를 생성하는 법”이 아니라, 목표가 귀속될 “자기(self)”를 에이전트가 어떻게 생성하고 그 경계를 어떻게 상대화(relativize)하는지로 재정의한다. 내적 동기(intrinsic motivation)나 자원 기반 목표 사전분포 등은 목표 생성의 메커니즘을 다루지만, 목표가 붙는 주체인 자기의 정합성 문제는 남아 있다고 주장한다. 또한 embeddedness(내재된 에이전시)가 오토텔릭 자율성의 필요조건이긴 하지만 충분조건은 아니며, 같은 역학도 서로 다른 자기 분할을 허용해 자기의 비유일성(non-uniqueness)을 드러낸다.

- **Technical Challenges**: 기여를 구현하려면, (1) 목표 공간 G와 그 분포 μ를 임의로 두지 않으면서도 정당화할 원리를 가져야 하고, (2) 행동을 위해서는 에이전트가 자기 경계를 “믿고”, 이해를 위해서는 그 경계를 “통과해” 봐야 하는 역설을 다뤄야 한다. 논문은 목표 사전분포를 균일하게 두려 해도 표현(representation)에 따라 편향이 들어가며, algorithmic information theory의 Solomonoff prior 같은 시도 역시 목표를 ‘가설’처럼 진리값으로 수렴시키기 어렵다고 비판한다. 대신 행동-환경의 인과적 개입을 다루되, 그것만으로는 무엇을 원할지(μ)가 결정되지 않으므로, homeostasis(항상성)로부터 목표 선호의 자기정당화 구조를 만들고 Markov blanket로 자기-환경 경계를 모델링해 embedded agency의 결과를 체계화한다.

- **Empirical Impact**: 논문은 이 단일 프레임을 기반으로 여러 방향(양자적 formulation, 철학적 독해, 그리고 LLM 기반 agentic instantiation)으로 확장하며, 관점이 단순한 철학이 아니라 구현 가능한 에이전트 설계로 이어짐을 보여주려 한다. 특히 embedded agency와 homeostasis를 연결해 ‘목표 생성’보다 ‘자기 귀속의 정합성’이 왜 중심인지 실험적·구조적으로 검증하려는 의도를 분명히 한다. 결과적으로 오토텔릭 에이전시 연구가 목표함수/탐험만이 아니라, 에이전트가 자기 경계를 어떤 방식으로 구성하고 업데이트하는지에 초점을 옮기는 데 의미가 있다.



### eCNNTO: A Highly Generalizable ConvNet for Accelerating Topology Optimization (https://arxiv.org/abs/2606.19921)
- **Prior Approaches**: 토폴로지 최적화(TO)는 SIMP 같은 반복 기반 방법에서 매 반복마다 FEA를 수행해 계산비용이 커진다. 이를 줄이기 위해 딥러닝 기반 가속이 나왔지만, 전역 방식(image-to-image)과 부분/국소 방식(원소 단위 예측)은 대체로 대규모 데이터 준비 비용이 크거나 작은 입력 변화에서 구조 단절·물리 제약 위반이 발생했다. 특히 DLTOP은 원소의 early history로 근접 최적 밀도를 예측해 반복을 크게 줄였으나, 이웃 원소 간 공간 상관을 충분히 반영하지 못해 연결성 문제가 생길 수 있다.

- **Core Contribution**: 이 논문은 원소 단위 CNN을 TO 가속에 적용한 eCNNTO를 제안한다. 핵심은 DLTOP의 아이디어를 유지하되, ResNet 기반 CNN과 residual connection을 통해 이웃 원소 간 공간 상관을 학습해 단절/코너 접촉 같은 결함을 줄이는 것이다. 추가로 학습 데이터 생성 시 early stage 밀도 이력을 쓰던 방식을 바꾸어 final stage 밀도 이력으로 학습해 속도와 데이터 효율을 동시에 끌어올린다.

- **Technical Challenges**: 가장 큰 기술적 도전은 (1) 원소별 밀도 예측에서 발생하는 공간적 연결성 결함과 (2) 학습-추론 사이의 feature 불일치(학습은 final stage, 추론은 early iterations) 사이를 함께 해결하는 것이다. 저자들은 H×W 원소 패치(2D/3D에서 이웃 포함)를 입력으로 하는 CNN으로 local receptive field와 weight sharing을 활용해 공간 상관을 모델링하고, residual block으로 학습 안정성을 확보했다. 또한 최종 단계의 미세한 수렴 정보를 학습에 활용함으로써 필요한 데이터 크기를 줄이면서도 온라인에서 SIMP 몇 회만으로 근접 해를 만들도록 설계했다.

- **Empirical Impact**: 2D·3D 다양한 벤치마크에서 eCNNTO는 SIMP 대비 반복 횟수를 최대 90%(2D), 97%(3D)까지 줄였고, 연결성이 깨지는 경우(단절 성분 발생)도 관찰되지 않았다고 보고했다. 또한 boundary 조건, 하중 케이스, 설계 도메인 형상, mesh 해상도, non-design domain까지 크게 달라도 재학습 없이 일반화 성능을 보였다. 결과적으로 eCNNTO는 “적은 데이터로 학습 + 높은 반복 가속 + 구조적 타당성 유지”라는 실용적 균형을 제시하며 학습 기반 TO 가속 연구에 의미 있는 진전을 제공한다.



### Multi-Agent Transactive Memory (https://arxiv.org/abs/2606.19911)
- **Prior Approaches**: 기존 retrieval-augmented generation(RAG)은 인간이 쓴 문서를 중심으로 한 에이전트 단일 사용 맥락의 보강에 강점이 있지만, 에이전트들이 만든 절차적 산출물은 보통 한 번 쓰고 폐기되거나 생산한 에이전트에만 남겨집니다. reasoning/thought reuse 계열은 비용·효율을 개선해도 “생산자 중심 재사용”에 머물러 새로 투입된 에이전트가 이미 존재하는 해결책을 다시 찾아야 하는 문제가 남아 있습니다. 또한 transfer learning·knowledge distillation은 도메인 정렬이나 추가 학습이 필요해, 오픈 생태계의 이질적·동적으로 생성되는 에이전트 집단에 바로 쓰기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 Multi-Agent Transactive Memory(MATM)라는 인프라로, 에이전트 집단이 생성한 action-observation trajectory를 공유 저장소에 누적하고 소비 에이전트가 이를 조회해 성능을 개선하도록 제안합니다. producer(생성)와 consumer(소비) 역할은 고정이 아니라 문맥에 따라 바뀌며, 조회된 궤적을 통해 “인구 단위 경험 재사용”을 가능하게 합니다. 특히 interactive 환경에서 긴 상호작용 궤적이 담는 절차 지식을 핵심 아티팩트로 삼아, 개인의 재탐색을 집단의 누적 지식으로 전환하려는 설계가 중심입니다.

- **Technical Challenges**: 가장 큰 도전은 상태가 조건인 궤적을 검색하는 일입니다. MATM은 최근 l단계 상호작용을 retrieval key로 쓰고 다음 구간을 value로 저장하는 state-conditioned key-value 인덱싱을 채택해, 쿼리와 현재 상태에 맞춘 “이어가기” 형태의 guidance를 제공합니다. 더 나아가 단순 임베딩 유사도만으로 뽑힌 후보를 learning-to-rank(LTR) reranker로 재정렬하며, producer 메타데이터(신뢰/품질), consumer 메타데이터(개인화) 등 여러 특성을 포함해 marginal utility(무조회 대비 성능 기여) 기준으로 라벨링·학습합니다.

- **Empirical Impact**: ALFWorld와 WebArena에서 MATM 조회는 평균 task 성공률과 상호작용 단계 수 효율을 함께 개선하며, 별도 조정이나 joint training 없이도 downstream 성능 향상이 나타납니다. ALFWorld에서는 성공률이 47%→55%로 상승하고 단계도 11.77→11.18로 줄어드는 등 효과가 뚜렷했고, WebArena에서도 개선은 더 완만하지만(성공률 18%→20%, 단계 22.0→20.3) RPP가 양수로 전환되며 Pareto 측면의 우위가 관측됩니다. 또한 reranking은 환경별로 이득 폭이 달랐지만(예: ALFWorld에서 SVMRank 강세), 전반적으로 “저장소가 커질수록”과 “집단 내 다양한 능력대 에이전트가 함께” 이득을 확장할 수 있음을 보여 오픈 에이전트 생태계에서의 경험 공유 설계 패턴으로 위치시킵니다.



### MetaResearcher: Scaling Deep Research via Self-Reflective Reinforcement Learning in Adversarial Virtual Environments (https://arxiv.org/abs/2606.19893)
- **Prior Approaches**: 기존 deep research agent 학습은 LiteResearcher처럼 고정된 로컬 가상 웹 환경에서 검색·열람을 반복하며 성능을 끌어올려 왔습니다. 다만 훈련 환경이 정적이라 시간에 따른 정보 변화·모순·정정 같은 현실 연구의 동학을 학습하기 어렵고, 작업도 사실 retrieval 중심에 머물러 가설 생성이나 모순 해소 같은 고차 연구 행동이 약합니다. 또한 GRPO의 outcome-only 보상은 탐색 과정에서 반복 루프를 유발할 수 있고, 단일 에이전트 구조는 역할 분담의 이점을 제한한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 MetaResearcher라는 프레임워크로 학습 확장축을 4가지로 동시에 제시합니다. 시간 변화와 적대적 허위 정보를 포함한 Evolving Virtual World로 ‘출처 신뢰도’와 ‘시간적 충돌 해결’을 강제 학습하고, fact retrieval을 넘어 Hypothesis Generation·Contradiction Resolution 같은 Discovery-Oriented Tasks로 연구형 사고를 훈련합니다. 여기에 Self-Reflective Meta-Reward(정답 외에 탐색 효율·반성 깊이·툴 호출 다양성 최적화)와 Scout/Filter/Synthesizer의 Heterogeneous Multi-Agent Swarm을 결합해 탐색 품질과 협업을 구조적으로 개선합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 현실처럼 시간에 따라 정보가 바뀌는 환경을 훈련용으로 구현하면서 (2) 그 환경에서 그럴듯한 허위·모순을 학습 신호로 전환하고 (3) 결과만 보상해 생기는 반복 루프를 과정 보상으로 막는 것입니다. 저자들은 버전 문서·시간 인덱싱·이벤트 스크립트로 시간 동학을 모델링하고, Synthetic Web 계열을 참고해 고빈도·고유사도 misinformation과 미세 조작을 생성해 에이전트의 인식 역량을 시험합니다. 이어 GRPO에 trace-level reflection 신호와 쿼리/도메인 다양성 패널티를 메타 보상으로 넣어 전략 전환과 backtracking이 학습되게 만들고, 멀티에이전트 역할 분해는 학습 중 emergent communication을 허용하도록 설계했습니다.

- **Empirical Impact**: 실험은 GAIA·Xbench-DS의 표준 성능 유지/개선을 확인하는 한편, (a) 허위 탐지·모순 해결·정보 진화 추적을 요구하는 epistemic robustness 벤치마크와 (b) 가설 생성·모순 해소를 평가하는 discovery task 벤치마크를 통해 각 혁신의 기여를 입증하려고 합니다. 또한 environment/reward/architecture/task에 대한 ablation으로 ‘무엇이 얼마나’ 성능과 견고성을 끌어올렸는지 분해해 검증합니다. 저자들이 주장하는 의미는, 취약성을 테스트에서만 다루던 관행을 넘어 훈련 자체를 적대적·과정 중심으로 바꿔 연구 에이전트의 신뢰성과 탐색 행동을 함께 최적화한다는 점에 있습니다.



### A Systematic Evaluation of Black-Box Uncertainty Estimation Methods for Large Language Models (https://arxiv.org/abs/2606.19868)
- **Prior Approaches**: 기존 LLM uncertainty estimation(UE) 연구는 주로 logits·hidden states 같은 내부 신호를 활용하는 방식이 많았고, 그 결과 접근이 제한된 API 환경에서는 적용이 어렵다는 한계가 있었습니다. 블랙박스 UE는 방법론이 여러 갈래로 흩어져 있어 분류 기준이 약하고, 성능을 한 프레임에서 비교한 통합 실험도 부족했습니다. 특히 어떤 접근이 언제 유리한지에 대한 경험적 가이드가 일관되지 않았습니다.

- **Core Contribution**: 이 논문은 블랙박스 UE 방법들을 verbalization-based, sampling-based, explanation-based, multi-agent, hybrid 등 5가지 범주로 체계화하고, 24개 대표 방법을 대상으로 한 통합 벤치마크를 구축했습니다. 또한 4종 모델과 4종 데이터셋 설정에서 동일한 평가 프레임워크로 비교 가능하게 만들어, 재현 가능한 비교를 위한 기준을 제공합니다. 전반적으로 단일 방법이 모든 조건에서 항상 우세하진 않지만, 답변 공간에서 후보를 추론·비교하는 계열이 대체로 효과적이라는 결론을 제시합니다.

- **Technical Challenges**: 블랙박스 환경에서는 내부 확률정보가 없어서 신뢰도 신호를 어떻게 관측 가능한 출력만으로 구성할지가 핵심 난제였습니다. 논문은 각 범주별로 서로 다른 uncertainty 신호(예: 텍스트 생성 패턴, 샘플링 변동, 설명 일관성, 다중 에이전트 관점)를 통합해 비교 가능한 형태로 평가 프레임워크에 연결했습니다. 더 나아가 여러 신호를 결합하는 hybrid 방식이 대부분의 조건에서 견고하다는 점을 통일된 실험 설계로 확인했습니다.

- **Empirical Impact**: 통합 벤치마크 결과, 어떤 단일 black-box UE 방법도 모든 모델·데이터셋·설정에서 일관된 1위를 차지하지는 못했습니다. 그럼에도 후보를 비교/추론하는 답변 공간 기반 접근이 전반적으로 유리했고, 다양한 uncertainty 신호를 묶는 hybrid 방법이 대체로 안정적인 성능을 보였습니다. 벤치마크 데이터와 평가 프레임워크를 공개함으로써 향후 연구의 재현성과 공정한 비교를 촉진하고, 실제 API 기반 LLM을 신뢰성 있게 다루려는 실무에도 구체적 선택 기준을 제공할 것으로 기대됩니다.



### TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability (https://arxiv.org/abs/2606.19821)
Comments:
          6 pages, 6 figures. Submitted to IEEE GLOBECOM 2026

- **Prior Approaches**: 기존 KPM 예측은 회귀 기반 신경망, LSTM 등 시계열 모델과, 기지국 간 의존성을 다루는 spatiotemporal GNN에 주로 의존해 왔습니다. 다만 supervised 방식이라 분포 변화 때마다 재학습 부담이 크고, 스케일이 커지면 셀별 학습 비용 때문에 현장 적용이 제한됩니다. 또한 상관 중심 출력에 그쳐 도메인 인과 근거가 부족해 운영자가 “왜”와 “무엇을” 해야 하는지까지 연결하기 어렵습니다.

- **Core Contribution**: TelcoAgent는 TSFMs(time-series foundation models) 기반 zero-shot 다중 KPM 예측과, 3GPP 규격에서 자동 구축한 지식 그래프(knowledge graph)를 결합한 프레임워크입니다. 예측 결과를 ReAct 기반 추론·설명 파이프라인으로 이어, 예측→원인 진단→작업 지시까지 traceable하게 제공합니다. 특히 사이트별 fine-tuning 없이도 다양한 셀에서 여러 KPM을 함께 예측하는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) TSFM의 블랙박스 비선형성을 운영 가능한 “인과 방향” 설명으로 전환하는 것과 (2) 3GPP 규격 지식을 예측 흐름에 정합적으로 연결하는 것입니다. 논문은 3개의 LLM 에이전트로 3GPP 문서에서 triples 기반 지식 그래프를 만들고, PAX-TS로 교차 채널 민감도 행렬을 계산한 뒤 3GPP의 directed causal paths와 매칭해 원인 루프를 복원합니다. 더불어 숫자 추출·민감도 계산 값을 파이프라인 산출물과 교차 검증해 hallucination을 줄이는 self-verification 모듈을 넣었습니다.

- **Empirical Impact**: 미국 사업자의 5G 네트워크에서 200개 셀, 3개월(시간 단위 1시간 집계) 실데이터로 평가했으며, 7개 KPM 전 채널에서 예측 정확도가 우수했습니다. 설명 품질은 3GPP 기반 사실성(faithfulness)과 운영 질문 적합성(answer relevancy)으로 측정해 평균 0.615와 0.807 수준을 보고하며, 지식 그래프 제거 시 answer relevancy가 7.4%p 하락해 인과 근거의 기여를 확인했습니다. 또한 보고서 수치의 99.8%가 허용 오차 내 일치해, “설명 가능한 예측”을 현장 운영 의사결정에 바로 연결하려는 시도에서 의미 있는 성과를 냈습니다.



### Human-on-the-Loop Orchestration for AI-Assisted Legal Discovery (https://arxiv.org/abs/2606.19812)
- **Prior Approaches**: 기존 e-discovery의 TAR은 키워드·SVM 기반 predictive coding에서, ReAct 루프 같은 multi-step Reason-Act-Observe 방식으로 발전했지만 위험 표면이 커졌다. 특히 agentic 워크플로는 한 단계의 오분류가 다음 단계의 쿼리·분류·로그 생성에 조건처럼 누적되는데, 이를 endpoint 지표(precision/recall/F1)가 제대로 벌점화하지 못한다.

- **Core Contribution**: 논문은 법률 정보 검색에서 에이전트 실패를 기능 단계별로 정리한 taxonomy를 제시해, 문제가 어디서 시작되는지 추적 가능한 관점으로 전환한다. 또한 planning·reasoning·execution·uncertainty quantification의 4중 verification 아키텍처를 두고, 불확실성 임계치 위반 시 Human-on-the-Loop(HOTL)로 전환해 trajectory collapse를 조기에 차단한다.

- **Technical Challenges**: 핵심 난제는 (1) 조기 단계에서 “해결 불가능/불충분”을 감지해 중간 추론을 폐기하고, (2) 실행 전 FRCP 준수 여부를 강제하며, (3) 모델 가중치를 못 쓰는 GPT-4o 환경에서 epistemic uncertainty를 안정적으로 추정하는 것이다. 논문은 solvability 판별 fp l a n-f유사 분류기, 진행도 기반 추론 재샘플링, 실행 단계 sandbox(deny-list+트랜잭션+보상 트랜잭션), self-consistency 분산으로 uncertainty를 캘리브레이션해 단계별로 연쇄 오류를 끊는다.

- **Empirical Impact**: 합성 e-discovery 코퍼스(5,000문서) 시뮬레이션에서 Threshold-HOTL(τ=0.5)은 privilege-waiver risk(PWR)를 8.3%→3.2%로 낮춰 약 61% 감소를 보였다. 동시에 attorney review는 약 23.7% 문서에만 발생해 비용 효율이 개선되었고, Rollback Recovery Rate 평균 0.71로 다수의 가로채기 이후 궤도 복구가 확인됐다. 다만 데이터가 합성이고 라벨 불일치·레이어별 기여 분리·실무 거버넌스 검증이 향후 과제로 남아 있다.



### Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning (https://arxiv.org/abs/2606.19808)
- **Prior Approaches**: 기존 연구는 chain-of-thought, self-consistency처럼 추론에 더 많은 연산을 투입해 성능을 끌어올리는 방식(추론 확장/탐색)을 주로 다뤘다. 또 outcome/process verifiers나 단계/상태별 selective verification처럼 검증을 세밀하게 제어하려는 접근이 있지만, 추가 검증기·검색 상태·세분 제어가 필요해 배포 복잡도가 커진다. 무엇보다 “한 번 더 생각하기”가 항상 이득이 아니라, 맞는 답을 망치거나( harmful flip) 불필요하게 비용을 늘릴 수 있어 배치(배포) 관점의 제어가 요구됐다.

- **Core Contribution**: 이 논문은 post-generation 추론을 ‘새 verifier를 만드는 문제’가 아니라, 서빙 레이어에서 “첫 시도 답을 보존할지/active verification을 호출할지”를 결정하는 deployment allocation 문제로 재정의한다. SeVRA(Selective Verification for Reasoning Allocation)는 frozen solver가 만든 초기 답과 serving-visible attempt state를 보고 recoverability(복구 가능성)를 추정해 개입을 선택한다. 또한 도움 수정(helpful fix)과 악화 수정(harmful flip)을 함께 고려하는 정책 학습/평가 틀을 제공한다.

- **Technical Challenges**: 핵심 난제는 서빙 시점에 정답 라벨이나 숨겨진 상태 없이도 active verification이 ‘도움이 될지’ 예측해야 한다는 점이다. 이를 위해 Qwen3-4B를 frozen 상태로 사용하며, 2,000개 MATH 데이터에서 base 시도 후 여러 개입(action)의 결과를 로그로 수집해 recoverability-aware gate를 학습하고, 임계값은 downstream 정책 성능과 액션 토큰을 기준으로 고정한다. “최대 토큰 예산”과 “실제 총 토큰(realized tokens)”을 분리해, 개입 횟수 절감이 전체 비용 절감으로 직결되지 않을 수 있음을 비용 지표 설계에 반영했다.

- **Empirical Impact**: MATH500에서 SeVRA의 selective active verification은 76.3% 정확도로 always verifying(75.5%)보다 좋고, 검증 토큰을 26.8% 줄이며 harmful flip을 2.2%→1.0%로 낮췄다. 반면 긴 초기 solve(8,192 토큰)는 76.0%로 통계적으로 비슷한 정확도를 내면서 총 realized 토큰은 28% 적어, math 비용 frontier에서는 “초기 예산 조정”이 더 효율적일 수 있음을 보여준다. GSM8K 전이에서는 SeVRA가 단 3.0%만 검증을 호출해 검증 토큰을 91.2% 줄이면서 93.4%→94.5%로 개선했고, CommonsenseQA에서는 항상 검증이 오히려 성능을 떨어뜨려 워크로드 의존성을 확인했으며 cheap feature gate가 learned gate들과 거의 비슷한 성능으로 배포 효율이 높다는 결론을 제시한다.



### CombEval: A Framework for Evaluating Combinatorial Counting in Large Language Models (https://arxiv.org/abs/2606.19788)
Comments:
          under review. Code: this https URL

- **Prior Approaches**: 기존 조합·확률(Combinatorial Counting) 평가는 GSM8K, MATH처럼 큰 수학 추론 데이터셋의 하위 범주 또는 정적 CO 전용 벤치마크(예: CombiBench)에 의존하는 경우가 많았다. 그러나 이런 정적 벤치마크는 데이터 오염(학습 데이터에 겹침)과 표면 패턴 편향으로 인해 실제 추론 능력보다 암기/피상적 추정이 성능을 부풀릴 위험이 있다.

- **Core Contribution**: CombEval은 LLM을 위한 동적(dynamic) 조합 카운팅 벤치마크로, 각 문제를 Cofola의 typed 명세(엔터티·조합 대상·의존성·제약)로 표현해 자연언어 문제를 “제어된 방식”으로 생성한다. 또한 Cofola 기반 정형화 뒤 백엔드에서 정확한 정답을 solver-verified로 검증해, 템플릿 매칭이 아닌 구조적 추론을 평가할 수 있게 한다.

- **Technical Challenges**: 핵심은 (1) 조합 카운팅 문제의 구조적 다양성을 포괄하면서도 (2) 생성 즉시 정확한 정답 검증을 대규모로 수행하고 (3) 난이도를 entity scale, 제약 개수, 추론 깊이 같은 파라미터로 예측 가능하게 조절하는 일이다. CombEval은 typed object-DAG 파이프라인과 제약-연산자 호환 규칙으로 Cofola 명세를 만들고, WFOMC로 컴파일되는 Cofola solver에서 시간 제한 내에만 안정적으로 풀리는 인스턴스만 필터링해 이 문제를 해결한다.

- **Empirical Impact**: 11개 LLM(오픈/클로즈드 포함)을 zero-shot 및 code-augmented 조건에서 평가했더니, 성능은 전반적으로 상위 모델에서 좋아지지만 ordered object, 구분 불가능(identical/indistinguishable) 원소, 비교적 positional 제약, 중첩된 객체 의존성에서 공통적으로 취약함이 관찰됐다. 특히 오류 분석은 제약 해석과 카운팅 원리(예: 함께(together) 블록의 순열/인접·연속 조건 오독, 멀티셋 원형의 몫 처리 누락 등)의 실패가 주요 원인임을 보여주며, CombEval이 “언제/왜” LLM이 조합 추론에서 깨지는지 진단하는 테스트베드로 의미가 크다.



### ORAgentBench: Can LLM Agents Solve Challenging Operations Research Tasks End to End? (https://arxiv.org/abs/2606.19787)
Comments:
          31 pages, preprint, v1

- **Prior Approaches**: 기존 OR 평가는 수식/모델링과 실제 해결(solve)을 분리하거나, 미리 정형화된 인스턴스·텍스트 기반 문제에 치우치는 경향이 큽니다. 그 결과 에이전트가 운영 아티팩트(데이터·설정·코드)에서 출발해 실행·검증·수정까지 끝내며 “결정 artifact”를 신뢰성 있게 생산하는 능력은 충분히 평가되지 않았습니다. 또한 솔버 중심 벤치마크는 모델링 단계의 선택을, 모델 생성 중심 벤치마크는 데이터 정합·실행·수리·런타임 제약 하 검증을 각각 약하게 만듭니다.

- **Core Contribution**: 이 논문은 실행 기반 end-to-end ORAgentBench를 제안해, 에이전트가 자연어 운영 brief와 멀티파일 데이터/구성물로부터 코드 작성·실행·제출 스키마 충족까지 수행하는 작업을 평가합니다. 평가는 hidden validator가 제출물의 스키마 유효성, 하드 제약(운영 제약)의 feasible 여부, 그리고 정규화된 목적함수 품질을 단계적으로 판정하도록 설계했습니다. 즉 “그럴듯한 최적화 코드”가 아니라 “검증 가능한 고품질 운영 의사결정”을 목표로 벤치마크를 재구성합니다.

- **Technical Challenges**: 핵심 난제는 에이전트가 문제를 잘 표현하는 modeling strategy와, 시간 제한 안에 좋은 해를 찾는 solving strategy를 코드로 결합해 작동시키는 데 있습니다. 이를 위해 ORAgentBench는 (1) 난이도를 인스턴스 크기만이 아니라 해결 전략 요구, 제약 결합, 동적 구조, 데이터 스케일, 문제 이해 등 6개 축으로 사전 통제하고 (2) 누출 없이 공용 파일만으로 해법이 재현 가능하게 구성했습니다. 또한 제출물 생성이 실패할 때 스키마/feasibility/quality의 어느 단계에서 깨지는지 진단 가능하도록 프로토콜을 마련했습니다.

- **Empirical Impact**: 실험(14개 frontier 모델-에이전트 설정, 총 107개 과제)에서 최고 스택도 전체 과제 통과율 35.51%, 하드 태스크 통과율 20.59%에 그쳐 신뢰성 있는 OR 실무 수준과 거리가 있음을 보여줍니다. 특히 실패 원인은 인프라나 솔버 시간보다 전략 약점이 지배적이며, 54.8%의 비통과가 modeling-side 오류(운영 규칙 누락/불일치, 비효율적·취약한 수식화)에서 발생했습니다. OR-specific skills는 하드 태스크 feasible 비율은 개선했지만, 품질 임계치 달성이나 통과율을 일관되게 끌어올리진 못해 “좋은 의사결정”을 위한 더 탄탄한 모델링·해 개선 절차의 필요성을 시사합니다.



### AgentFinVQA: A Deployable Multi-Agent Pipeline for Auditable Financial Chart QA (https://arxiv.org/abs/2606.19782)
- **Prior Approaches**: 기존 금융 차트 Q&A는 정확도 자체에 초점이 맞춰져 있고, 오답일 때 왜 틀렸는지 설명·검증이 불투명한 경우가 많습니다. 또한 외부 API 의존이 전제된 설계가 흔해 개인정보·데이터 레지던시 요구가 큰 규제 환경에는 바로 적용하기 어렵다는 한계가 지적됩니다.

- **Core Contribution**: AgentFinVQA는 단일 모델 추론 대신 계획(Plan)–OCR–legend grounding–시각 검토(Inspect)–검증(Verify)으로 질의를 분해하고, 각 단계 산출물을 per-sample Model Evaluation Packet(MEP)으로 기록해 감사를 가능하게 합니다. 더불어 verifier의 판정(CONFIRM/REVISE)과 신뢰 점수를 활용해 human-in-the-loop 검토 우선순위를 조절하는 흐름을 제안합니다.

- **Technical Challenges**: 핵심 난제는 (1) 텍스트(축·범례·라벨) 추출 오류와 (2) 범례-색상 매핑 혼동이 시각 추정까지 연쇄적으로 악화되는 점, 그리고 (3) 검증 단계가 정확한 수정인지 과도한 번복인지 가르는 점입니다. 논문은 OCR·범례를 구조화된 메타데이터로 다음 단계에 “증거”로 주입하고, stacked/pie 계열 일부 경우에 한해 deterministic colour-area 도구로 색상 기반 추정 단서를 추가하며, 독립 VLM 호출로 draft answer를 감사·판정한 뒤 confidence 게이트로 오버라이드를 억제합니다.

- **Empirical Impact**: FinMME에서 AgentFinVQA는 proprietary backbone(Gemini-3 Flash) 기준 model-matched zero-shot 대비 +7.68%p(71.24% vs 63.56%) 향상, open-weights(Qwen3.6-27B-FP8)로 로컬 서빙 시에도 +4.84%p 개선을 보여 API 의존 없이 이득이 유지됨을 입증합니다. 또한 exact accuracy 관점에서 verifier 판정이 “confirmed”와 “revised” 정답 품질 차이를 신호로 제공(68.2% vs 55.6%)하며, 실패의 상당 부분이 질문 오해·legend 혼동·추출 오류에 집중돼(약 2/3) 향후 개선 방향을 구체화합니다.



### Beyond Entropy: Learning from Token-Level Distributional Deviations for LLM Reasoning (https://arxiv.org/abs/2606.19771)
- **Prior Approaches**: RLVR은 verifiable reward로 LLM의 추론을 강화하지만, 많은 구현이 토큰에 동일한 학습 신호를 흘려보내 entropy collapse(엔트로피 붕괴)를 촉발해 더 나쁜 전략으로 조기 수렴할 위험이 있습니다. 한편 high-entropy 토큰을 골라 업데이트하는 접근은 entropy explosion(엔트로피 폭주)로 이어져 방향성 없는 blind exploration을 만들 수 있어 성능이 흔들립니다. 즉, Shannon Entropy를 단일 스칼라로 쓰는 방식은 “불확실성”은 알려줘도 “어디로 탐색이 가야 하는지”를 충분히 대변하지 못한다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 LLM의 RLVR에서 탐색 안정성을 좌우하는 핵심 신호를 “엔트로피 값”이 아니라 “토큰 로짓 분포의 분포적 성질”로 옮깁니다. Jensen-Shannon (JS) divergence로 토큰의 독특함(unique token)을 정의하고, 이 독특한 토큰들이 추론 경로의 중요한 branching point(분기 노드)로서 탐색 방향을 제공한다고 주장합니다. 이를 실현하는 프레임워크로 Independent Combinatorial Tokens (ICT)를 제안하며, 독특한 토큰에 한해 선택적으로 업데이트해 수렴과 탐색의 균형을 잡습니다.

- **Technical Challenges**: 가장 큰 난제는 “엔트로피 붕괴 vs 엔트로피 폭주”라는 상반된 불안정성을 근본 원인에서 분리해 제어하는 것입니다. 논문은 Shannon 엔트로피만으로는 상태/확률 분포가 같아도 탐색 궤적이 달라질 수 있음을 보이고, second-order Rényi entropy까지 포함해 업데이트가 정책 농도와 분포 불확실성에 동시에 미치는 이중 효과를 이론적으로 분석합니다. 또한 JS divergence로 고유 토큰을 골라 GRPO의 gradient에 sparse mask를 적용하는 Sparse-GRPO 추정식을 설계하고, 전체 토큰을 다 업데이트하지 않으면서도 안정적인 탐색 지형을 유지하도록 구성합니다.

- **Empirical Impact**: Qwen2.5(0.5B/1.5B/7B)에서 독특한 토큰 상위 10%만 업데이트하면 GRPO, 20-Entropy, STAPO 대비 평균 pass@4가 4.58% 향상되고 최대 14.9%까지 개선되었다고 보고합니다. 수학(예: GSM8K, MATH 계열)에서 얻은 이득이 GPQA 같은 다른 도메인에도 전이되며, Pass@4에서의 상대적 이득이 Pass@1보다 커서 ICT가 더 풍부한 정답 추론 궤적 다양성을 만든다는 해석을 뒷받침합니다. 요약하면 ICT는 “엔트로피를 직접 최적화하지 않고도” 암묵적으로 엔트로피 동역학을 조절해 RLVR 학습 안정성과 탐색 효율을 동시에 끌어올리는 접근으로 의미가 큽니다.



### Optimal Scheduling in a Question-Answering Forum of Knowledge Workers (https://arxiv.org/abs/2606.19759)
Comments:
          14 pages, 4 figures

- **Prior Approaches**: 기존 QA 포럼 연구는 텍스트로 주제 분류하기, 소셜 네트워크에서 확산 효과 분석하기 등 ‘콘텐츠/행동’ 관점이 중심이었다. 또 큐 기반 모델에서 여러 전문가를 고정 비율로 매칭하는 스케줄링 연구도 있었지만, 요청 도착 통계와 전문가 역량을 오프라인에선 완전히 알아야 하고, 실시간 적응이 제한적이었다.

- **Core Contribution**: 이 논문은 지식 근로자(전문가)가 주제 큐에서 요청을 받아 답변하는 ‘스케줄러 설계’ 문제를 큐잉 이론으로 정식화하고, 시스템이 안정(stable)하게 유지되는 요청 처리 한계인 capacity를 계산한다. 더 나아가 전문가 간 단순 coordination(서로 독립 처리)뿐 아니라 collaboration(여러 전문가가 공동으로 동일 요청을 해결) 모드가 capacity를 어떻게 키우는지까지 분석한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 주제별 도착이 확률적으로 변하고 (2) 전문가의 주제별 answering rate가 다르며 (3) 한 요청이 답변되기까지 필요한 시간이 geometric처럼 랜덤이라는 조건에서, 큐 길이를 기준으로도 과부하 없이 안정성을 보장하는 온라인 스케줄을 만드는 것이다. 이를 위해 현재 큐 길이 Q_x(t)와 전문가의 q_i(x)를 이용해 greedy한 온라인 스케줄(2단계 할당, non-preemptive 변형 포함)을 제안하고, collaboration의 경우에는 generalized independent set 제약이 있는 Hypergraph Assignment Problem 형태로 스케줄링을 구성해 capacity-달성 가능성을 보인다.

- **Empirical Impact**: 이론적 capacity 상한은 volunteer 기반 포럼이 ‘최대 처리량’을 목표로 하지 않을 때의 성능 격차를 설명하는 기준선 역할도 한다. 논문은 collaboration이 coordination보다 크게 유리해질 수 있음을(서로 보완적인 역량이 있을 때 공동 해결 확률이 급상승) 직관적 예시와 함께 제시하며, 구체적 스케줄이 해당 capacity 수준까지 도달할 수 있음을 뒷받침하는 시뮬레이션 논의가 포함된다. 결과적으로, 향후 프리미엄 QA 운영(전문가 매칭/공동작업)에서 “어떤 스케줄링 구조가 처리량을 늘리는가”라는 설계 지침을 제공한다.



### Grounded Inference: Principles for Deterministically Encapsulated Generative Models (https://arxiv.org/abs/2606.19753)
Comments:
          12 pages, 3 figures

- **Prior Approaches**: 기존 접근은 LLM을 프로덕션에 바로 끼워 넣고 temperature=0, seed 고정, 프롬프트 엔지니어링 같은 설정으로 “결정론적 파이프라인”을 만들려는 시도가 많았습니다. 하지만 GPU 병렬 실행의 부동소수점 비결합성, API/서빙 레이어의 내부 변동(배치·랜덤 시드 처리), 비공개 시스템 프롬프트 변경 등으로 재현성과 감사를 깨뜨립니다. 또한 컨텍스트 윈도우에 의존한 상태 저장(메모리 역할)은 silent한 상태 변형과 컨텍스트 collapse를 유발할 수 있습니다.

- **Core Contribution**: 이 논문은 생성 모델을 전통적 계산 시스템에 “안전하게” 결합하기 위한 기반 프레임워크로 Atomic Primitives 4가지를 제시합니다: Probabilistic Engine, Model Encapsulator, State Registry, Deterministic Orchestrator입니다. 핵심은 확률적 출력을 결정론적 캡슐화(스키마/계약 검증)로 강제해, 엔터프라이즈 데이터 처리의 실행 경계를 명확히 분리하는 것입니다. 추가로 산업 전반에서 반복되는 anti-pattern 2종을 정리해 엔지니어의 위험한 설계 습관을 경고합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 “같은 입력이면 같은 출력” 같은 소프트웨어 계약을 LLM이 본질적으로 보장하기 어렵다는 점입니다. 논문은 이를 해결하기 위해 Encapsulator가 출력 검증 실패 시 deterministic retry budget 내에서만 재시도하고, 실패 시 circuit breaker로 중단·오류 텔레메트리를 감사용 저장소에 남기도록 설계합니다. 더불어 Probabilistic Engine을 완전한 stateless 작업으로 국소화하고, 모델이 데이터 plane(주데이터 흐름)을 직접 변형하지 않도록 아키텍처적으로 물리 차단합니다.

- **Empirical Impact**: 제시된 Adaptive Resolution Agent 레퍼런스 아키텍처는 문서 기반 제약을 함께 활용해 스키마 drift에 대한 파서 로직을 합성하되, 샌드박스 문법 테스트와 문서 정합성 의미 검증을 통과한 경우에만 Git PR로 승격합니다. 이는 자기 수정(self-correction)이나 정보 패스스루 같은 비결정적 제어 방식을 피하고, 검증 가능한 execution trace로 감사성과 안정성을 확보하는 방향입니다. 다만 컨텍스트 윈도우/문서 크기 제약과 토큰 비용·정확도 하락(문서가 중간에 위치할 때 추출 정확도 저하)이라는 실전 트레이드오프도 함께 강조합니다.



### Benchmarking Agentic Review Systems (https://arxiv.org/abs/2606.19749)
Comments:
          11 pages, 7 tables, 4 figures

- **Prior Approaches**: 기존 LLM 기반 리뷰 연구는 작은 백본 모델이나 단일 프롬프트/부분 파이프라인에 머물러, 실제 공개 리뷰 시스템 전체를 공정하게 비교하기 어려웠습니다. 또한 평가가 모델 출력을 직접 보는 경우가 많아, “시스템 단위”에서 품질 신호와 오류 탐지를 함께 검증하기가 제한적이었습니다. 최근 멀티에이전트/루브릭 기반 접근도 있었지만, 무엇이 얼마나 잘 작동하는지에 대한 재검증이 필요했습니다.

- **Core Contribution**: 이 논문은 실제로 사용 가능한 리뷰 시스템(OpenAIReview, coarse, Reviewer3)과 zero-shot 기준을 여러 LLM(6개, 프론티어+경량)과 묶어 “시스템 전체”를 정면 비교합니다. 먼저 ICLR/NeurIPS 논문에서 AI 리뷰가 외부 품질 신호(인용, 수상/선정, 리뷰 점수 등)를 얼마나 따라가는지 상관을 측정합니다. 다음으로 알려진 오류를 주입한 perturbation benchmark로, 각 시스템이 오류를 실제로 얼마나 회수(recall)하는지 검증합니다.

- **Technical Challenges**: 핵심 난제는 (1) 논문 품질의 정답 라벨이 없다는 점과 (2) 오류가 실제로 “읽을 가치가 있는 진짜 오류”인지 통제해야 한다는 점입니다. 이를 위해 연구팀은 품질 프록시 기반의 low/high 페어링과, 8개 arXiv 분야·4개 오류 유형(로컬 수학 편집, 거짓 주장, 논리 오류, 실험/분석 오류)을 갖춘 perturbation benchmark를 설계했고, 여러 단계(추출-생성-구조 검증-체크리스트 검증-주입)로 오류의 유효성을 통제했습니다. 또한 리뷰 코멘트가 주입된 변경을 가리키는지 퍼지 매칭+LLM 판정으로 검출 여부를 판정해 재현성 있는 recall을 산출했습니다.

- **Empirical Impact**: 결과적으로 OpenAIReview + GPT-5.5가 ICLR/NeurIPS 품질 프록시와의 페어와이즈 정확도에서 83.0%로 가장 높았고, 리뷰가 단순 점수 예측이 아니라도 품질 신호를 포착하는 경향이 확인됐습니다. 오류 회수에서는 OpenAIReview + GPT-5.5가 주입 오류의 71.6%를 잡았으며, 모델 여러 개의 탐지를 합치면 recall이 83.3%까지 상승해 모델 간 보완성이 관찰됐습니다. 더 나아가 실제 공개 배포에서 사용자는 코멘트에 대해 긍정 투표가 부정보다 1.44배(1.44:1) 많았고, 주요 불만은 false positive와 사소한 nitpicks로 정밀도 개선 여지가 드러났습니다.



### A Comparative Study of Pretrained Transformer Models for Quranic ASR: Speech Representations, Label Formats, and Dataset Composition (https://arxiv.org/abs/2606.19747)
Comments:
          30 pages, 9 figures, 5 tables, Submitted to International Journal of Speech Technology

- **Prior Approaches**: 기존 Quran ASR은 Tarteel이나 일부 벤치마크에 한정된 학습으로 인해 전 구절(코퍼스) 커버리지가 약하고, Tajweed(테크위드) 준수에 필요한 음향 변이까지 충분히 반영하지 못하는 문제가 지적돼 왔다. 또한 MFCC 기반 전통 모델이나 Citrinet 같은 구조가 주로 쓰였지만, 사용자(초보) 낭송 데이터의 잡음·라벨 품질 편차를 흡수하는 일반화 성능은 제한적이었다.

- **Core Contribution**: 본 논문은 Wav2Vec2.0, HuBERT, XLS-R처럼 self-supervised 학습 표현을 Transformer 기반 end-to-end ASR에 도메인 맞춤 fine-tuning하는 방식을 체계적으로 실험한다. EveryAyah(전문 낭송)와 Tarteel(사용자 낭송)을 포함한 필터링된 870시간+ 학습 데이터로, 구절 전체 커버리지와 사용자 낭송 상황을 함께 고려한 전사 정확도를 보여준다.

- **Technical Challenges**: 가장 큰 난제는 (1) Tajweed 관련 미세한 조음 차이를 음향 표현이 충분히 분리해낼지, (2) diacritics 유무·출력 라벨 포맷·클립 길이 같은 설정이 WER에 미치는 영향을 동시에 찾는 데 있었다. 저자들은 전처리에서 침묵 제거를 피하고(문맥 단서 유지), CTC loss 기반으로 라벨-정렬 부담을 줄이며, 출력 라벨(디아크리틱/비디아크리틱, transliteration 등)과 clip duration(최적 30초)까지 포함한 광범위 ablation으로 최적 조건을 도출했다.

- **Empirical Impact**: 최고 성능은 Wav2Vec2-large-XLSR-53 + ‘Arabic without Tashkeel’ 조합에서 나왔고 EveryAyah subset WER 0.08, EveryAyah+Tarteel WER 0.11을 기록했다. 이는 Citrinet baseline(WER 0.163) 대비 약 5%p 개선이면서, 통합 학습 시간도 140시간에서 40시간으로 크게 줄인 셈이다. 또한 전문 낭송 단독 학습이 가장 낮은 오류를 보이되, Tarteel 데이터 품질 향상이 더 높은 일반화로 이어질 가능성을 실증적으로 제시한다.



### Interpreting Neural Combinatorial Optimization via Evolving Programmatic Bottlenecks (https://arxiv.org/abs/2606.19741)
Comments:
          Under Review

- **Prior Approaches**: Neural Combinatorial Optimization(NCO)은 TSP/CVRP 같은 라우팅·스케줄링에서 강한 성능을 보이지만, 의사결정이 블랙박스로 남아 운영 전략이 무엇인지, 단계별로 어떻게 바뀌는지 설명이 어렵다. 기존 CBM(Concept Bottleneck Models)은 고정된 개념 어휘로 내부 연산을 해석 가능한 형태로 바꾸지만, NCO는 상태(state)에 따라 규칙이 동적으로 바뀌고 개념 후보(알고리즘 규칙)가 조합적으로 커서 동일한 방식이 바로 적용되지 않는다. 또한 NCO의 “개념”이 단일 원자 특징이 아니라 순차 의사결정 알고리즘 자체에 가깝다는 점에서 기존 해석 프레임과의 정합성이 떨어진다.

- **Core Contribution**: 이 논문은 NCO 정책을 사람이 읽을 수 있는 프로그램 포트폴리오로 증류해 해석 가능성을 제공하는 최초의 프레임워크로 Evolving Programmatic Bottlenecks(EPB)를 제안한다. EPB는 LLM을 활용해 실행 가능한 휴리스틱 프로그램 bank를 자율적으로 진화시키고, state-dependent router로 각 단계에서 어떤 프로그램(의사결정 규칙)을 조합할지 동적으로 선택한다. 결과적으로 “고정된 개념 집합” 대신 “학습 중 생성되는 프로그램 기반 개념”을 bottleneck으로 삼아 순차 의사결정의 해석 단위를 정한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) bank에 어떤 프로그램을 넣을지, (2) 상태별로 어떻게 조합할지(연속·미분 가능한 라우팅), (3) bank의 크기를 얼마로 정할지(이산적/구조적 결정)가 서로 다른 탐색공간을 가진다는 점이다. EPB는 이를 단일 최적화로 풀지 않고 Block I/II의 2-block 반복 구조로 분리해 해결한다: Block I에서는 hybrid textual-numerical gradient descent로 TextGrad 기반의 코드(프로그램) 수정과 SGD 기반의 라우팅 업데이트를 함께 수행하고, Block II에서는 Add(새 specialist 추가)와 Drop(중복 휴리스틱 제거)로 bank capacity를 동적으로 조절한다. 또한 routing 키가 heuristic 출력에 조건되되 역전파 경로는 분리하도록 stop-gradient 설계를 넣어, “혼합 경로의 책임”과 “라우팅에 대한 그라디언트”를 분명히 분리하는 방식으로 학습을 안정화한다.

- **Empirical Impact**: 실험에서는 POMO와 LEHD를 TSP/CVRP에 적용해 EPB가 teacher의 그리디·탐색 성능을 거의 유지하면서도, 해석 가능한 프로그램 포트폴리오로 정책을 치환할 수 있음을 보였다. 더 나아가 zero-shot으로 더 큰 인스턴스로 옮겼을 때는 일부 설정에서 teacher보다 surrogate가 더 잘 일반화하며(예: POMO-TSP-500에서 격차가 크게 개선), 프로그램 기반 증류가 전이 가능한 결정 패턴을 분리하고 과적합 잡음을 줄일 수 있음을 시사한다. 또한 TSP/CVRP에서 최적화 단계가 진행될수록 의사결정 전략이 바뀌고, 이를 고전 휴리스틱 변형들의 조합으로 근사 가능하다는 통찰을 제공해 NCO 해석 연구의 출발점을 넓힌다.



### GLARE: A Natural Language Interface for Querying Global Explanations (https://arxiv.org/abs/2606.19735)
Comments:
          16 pages, 2 figures

- **Prior Approaches**: 기존 XAI는 국소 설명(예: saliency map, concept bottleneck, counterfactual)처럼 개별 예측을 해명하는 방식이 주류였지만, 편향이나 전반적 추론 패턴 같은 “전역(global) 이해”에는 한계가 컸습니다. 전역 설명은 DNF 규칙·서로게이트·전역 중요 특징처럼 요약을 제공하지만, 대규모 모델의 경우 규칙/프로토타입이 너무 방대해 사용자가 탐색하기 어렵다는 문제가 있었습니다. 또한 전역 설명을 자연어로 직접 질의해 원하는 통찰만 뽑아내는 흐름은 충분히 자동화되지 않았습니다.

- **Core Contribution**: 이 논문은 전역 설명을 “정적 결과물”이 아니라 질의 가능한 데이터베이스로 바꾸는 LLM 기반 인터페이스 GLARE를 제안합니다. 사용자는 “어떤 클래스에선 어떤 특징 조합이 얼마나 필요한가”처럼 질문하고, 시스템의 핵심 LLM이 이를 구조화된 SQL로 변환해 로컬 Minimal Sufficient Explanations을 집계·필터링한 통계를 자연어와 시각 증거로 제공합니다. 특히 SQL 생성을 설명 템플릿(분석용 질의 패턴) 안에서만 수행해 포맷 오류와 환각을 줄입니다.

- **Technical Challenges**: 핵심 난제는 자연어 질문의 유연성을 유지하면서도, 전역 설명의 복잡한 집계 논리를 안전하고 검증 가능하게 실행 가능한 형태로 바꾸는 것이었습니다. GLARE는 (1) 설명용 SQL 중간표현을 미리 정의된 템플릿과 고정 스키마로 제한하고, (2) fine-tuning 시 SQL_START~SQL_END 구간만 학습하도록 fence masking(손실 마스킹)을 적용해 SQL 구조를 학습하도록 유도했습니다. 그 결과 모델이 데이터셋 고유 토큰에 과적합하지 않고, 다른 객체 어휘를 가진 데이터셋에도 질의 해석을 일반화할 수 있게 했습니다.

- **Empirical Impact**: ADE20K 기반 전역 설명에서 GLARE는 in-distribution 질의에 대해 구조적 정합성(fence detection/parse/execution)을 거의 완벽히 달성하며, 결과 일치율도 95%대(예: 95.2%)로 보고됩니다. 언어 교란(철자 오류, 동의어, 문장 변형 등)에 대해서도 강건성을 보였고, 특히 철자 오류에서 fine-tuned 모델이 regex 기준 대비 큰 폭으로 성능 우위를 보였습니다. 더 나아가 ADE20K로 학습한 뒤 Pascal VOC처럼 객체 어휘가 완전히 다른 데이터셋에도 retraining 없이 높은 정확도(약 90%대)를 보이며, “전역 설명의 실제 사용성”을 크게 끌어올리는 접근으로 의미가 있습니다.



### Beyond Static Leaderboards: Predictive Validity for the Evaluation of LLM Agents (https://arxiv.org/abs/2606.19704)
Comments:
          17 pages, 2 tables, 5 figures

- **Prior Approaches**: 기존 에이전트 벤치마크는 궤적 수준 성능의 다양한 단면을 다루더라도, 대체로 단일 통합 점수(aggregate score) 기반 순위를 매겨 배치 환경의 다차원 평가를 충분히 반영하지 못했습니다. 또한 LLM-as-judge 중심 채점은 심판 모델/프롬프트 변화에 따라 순위가 흔들리는 ‘rank instability’ 문제를 동반할 수 있습니다. HELM류는 다차원성을 넓혔지만, 에이전트 시대의 오케스트레이션·멀티턴 아티팩트 재사용·툴 호출 위생 같은 축은 점수화가 약하다는 비판이 있습니다.

- **Core Contribution**: 이 논문은 MCP 기반 산업 에이전트 벤치마크를 중심으로 14개의 병렬 구현 연구를 집계해, 배치 시 관찰되는 차원들이 단일 점수 리더보드에서 어떻게 붕괴되는지 구조적으로 정리합니다. 해결책으로 ‘예측 타당성(predictive validity)’—in-sample 순위가 out-of-sample에서 얼마나 유지되는지—를 순위 결정 기준으로 제안합니다. 동시에 배치 관련 평가 차원을 드러내는 12-tier 측정 장치(HELM 및 후속의 붕괴 지점을 포함)를 제시합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 순위가 실제 배치 성능을 예측하는지 검증하는 기준을 만드는 일입니다. 논문은 샘플 분할 유지(stratified split), 부분집합 전이(holding out subsets), 의미상 동등 패러프레이즈·식별자 변경·시간창 이동·교란 추가를 포함한 적대적/분포이탈(OOD) 기준 등 세 가지 ‘검증 가능한’ 조건과 임계값으로 predictive validity를 시험하도록 설계합니다. 다만 제안은 아직 두께가 얇아(일부 조건은 부분 지지) 통제 실험에서 반증 가능성을 전제로 파일럿 연구 설계를 예고합니다.

- **Empirical Impact**: CODS-2025의 공개-비공개 평가 회고에서는 공개 리더보드 순위와 숨은 평가 순위의 Spearman 상관이 execution 트랙에서 유의미하게 0에 가까웠고(ρ≈-0.13), planning 트랙에서만 양의 상관이 관찰되는 등 ‘전이 실패’ 신호가 직접 제시됩니다. 또한 합산 점수가 서로 다른 동작을 동일하게 취급하는 사례(추론 on/off, 멀티턴 아티팩트 재사용, 검색 전략 trade-off)를 통해 단일 점수의 정보 손실을 보여줍니다. 결론적으로 다음 세대 벤치마크는 집계 점수 리더보드 대신 OOD-전이 기반 예측 타당성과 12-tier 드릴다운(아키텍처·추론 모드·검색 전략·베리파이어 유형 등)을 공개해야 한다는 영향력을 제안합니다.



### Exit-and-Join Dynamics for Decentralized Coalition Formation (https://arxiv.org/abs/2606.19683)
- **Prior Approaches**: 기존 연합/코얼리션 형성 연구는 대체로 중앙집중적 split-and-merge나 전역 최적화처럼 “처음부터 설계된” 절차를 가정하는 경우가 많았습니다. 반면 동적(dynamic)·엔도제너스(endogenous) 접근도 있었지만, 중간 상태에서 에이전트가 참고하는 보상이 전역 협상/전역 정보에 의해 계산되는 모델이 주류였습니다. 그 결과, 실제로는 에이전트가 자신의 현 소속(로컬 정보)만 보고 나가고(구성원 이탈) 들어가는 exit-and-join이 어떻게 안정성으로 이어지는지에 대한 설명력이 약했습니다.

- **Core Contribution**: 이 논문은 코얼리션 형성을 “분산형 비동기 동역학”으로 모델링하며, 각 에이전트의 편차(deviation)는 단독 출구(exit)와 입장(join) 결정만으로 발생하게 만듭니다. 보상은 전역적으로 합의된 코얼리션 구조가 아니라, 에이전트가 현재 속한 코얼리션 내부에서 Aumann-Dreze value로 계산되어 로컬 비교만 수행합니다. 또한 단말(terminal) 코얼리션 구조를 “허용되는(대상 코얼리션의 수락/조건) 개별 이익 편차가 더 이상 없는 고정점”으로 정확히 연결해, 협력적 지불 분배와 비협력적 best-response가 동시에 성립하는 조건을 정리합니다.

- **Technical Challenges**: 가장 큰 난점은 로컬 보상 규칙(Aumann-Dreze value)이 만드는 유인 구조가, 동역학의 수렴(예: scalar Lyapunov 또는 exact-potential)과 어떻게 정합되는지 찾는 것입니다. 연구진은 단말 조건을 유도된 비협력 best-response의 고정점으로 재해석하고, Lyapunov/포텐셜 표현이 성립하려면 “인센티브 정렬(incentive-alignment)”이 추가로 필요함을 조건화합니다. 더불어 switching cost(전환 비용)와 acceptance cost(수락 비용)가 로컬 안정성과 상태 변화 가능성을 어떻게 바꾸는지까지 분석합니다.

- **Empirical Impact**: 실험에서는 유한 시간 내 안정화(finite-time stabilization), 비용(전환/수락 마찰) 민감도, 그리고 특별한 convex-game 벤치마크를 통해 동역학의 선택 결과가 어떻게 달라지는지 검증합니다. 특히 grand coalition이 효율적(협력적으로 유리)이어도, 로컬·myopic한 exit-and-join이 항상 같은 결말을 보장하지는 않는다는 점을 동역학 관점에서 보여줍니다. 이는 다중 에이전트가 태스크/조직 경계 사이를 자율적으로 이동하는 현실 시나리오(시장 이동, 온라인 커뮤니티 재편 등)를 더 그럴듯하게 모델링하는 데 의미가 큽니다.



### Denoising Implicit Feedback for Cold-start Recommendation (https://arxiv.org/abs/2606.19658)
Comments:
          Accepted by KDD 2026 ADS Track

- **Prior Approaches**: 암묵적 피드백을 활용한 추천에서 라벨 노이즈(클릭베이트, position bias 등)는 필연적이지만, 기존 연구는 이를 “loss 값이 높다/예측 점수가 낮다” 같은 휴리스틱 신호로 구분해 sample selection이나 sample re-weighting으로 처리해왔다. 이 방식은 일반적으로 노이즈 샘플을 찾는 판별이 잘 작동할 때 성능이 나오지만, cold-start 아이템은 원천적으로 모델 출력 분포가 불안정해 loss·예측점수·gradient 신호가 노이즈 탐지에 부적합해진다. 또한 스트리밍 환경과 산업용 추천 파이프라인에 바로 얹기 어려워 온라인 적용성이 제한된다는 문제가 있었다.

- **Core Contribution**: 이 논문은 cold-start 아이템이 특히 노이즈 암묵적 피드백(false positive/false negative)에 더 취약하다는 점을 강조하고, 이를 denoising implicit feedback 문제로 정면 해결한다. model-agnostic 접근인 DIF(Denoising Implicit Feedback)는 cold 아이템에 대해 콘텐츠 유사 warm 아이템들의 collaborative representation을 바탕으로 pseudo-label을 생성하고, 이를 이용해 노이즈 라벨을 샘플 수준에서 보정한다. 기존처럼 모델 예측 자체에 의존하기보다 “사용자 관심(안정적) + 콘텐츠 유사성”을 축으로 추정 정확도를 높이는 것이 핵심이다.

- **Technical Challenges**: 어려움은 (1) cold 아이템의 pseudo-label을 합리적으로 설계하는 것과 (2) 온라인 스트리밍 학습에서 콘텐츠 유사 warm 아이템을 실시간으로 검색·표현해 pseudo-label을 만들 수 있는지, (3) 여러 pseudo-label을 어떻게 더 정확히 결합할지, (4) 노이즈 라벨을 어떤 정도로 수정할지에 있다. DIF는 warm 아이템을 top-k 콘텐츠 유사 이웃으로 뽑아 여러 pseudo-label을 만들고, 콘텐츠 유사도 기반 confidence로 가중 합산하며, relative entropy와 cold-start 상태를 통해 샘플별 uncertainty를 추정해 pseudo-label의 보정 강도를 adaptive하게 조절한다. 또한 Kuaishou의 듀얼-타워 기반 retrieval 단계에 실제로 붙이기 위한 데이터/서빙 파이프라인(콘텐츠 임베딩, item-to-item ANN, collaborative embedding 저장 및 갱신)까지 구현 경험을 제시한다.

- **Empirical Impact**: 이 방법은 이론적 정당화와 함께 실제 데이터셋 3종에서 offline 성능을 넓게 검증해 일반화 가능성을 보여준다. 더 나아가 Kuaishou에 billion-user 규모로 배포되어 cold-start 시나리오에서 여러 상용 지표를 유의미하게 개선했다고 보고한다. 즉, cold-start 구간의 잘못된 학습 신호를 줄여 추천의 성장 잠재력을 확보하고 피드백 루프 악화를 완화하는 실용적 효과가 확인된 셈이다.



### BrainG3N: A Dual-Purpose Tokenizer for Controllable 3D Brain MRI Generation (https://arxiv.org/abs/2606.19651)
- **Prior Approaches**: 기존 3D brain MRI latent diffusion 파이프라인은 encoder–decoder tokenizer를 한 가지 reconstruction 목표로 같이 학습해 왔다. 그 결과 decoder의 해부학적 충실도에 유리해지면서, 조건 생성과 임상 태스크에 필요한 임베딩의 임상 정보가 희생되는 문제가 지적돼 왔다. 또한 많은 접근이 voxel 단위 재구성 지표 중심으로만 평가되어, “임상 표현성 + 생성 가능성”의 동시성을 검증하기 어려웠다.

- **Core Contribution**: 이 논문은 완전 부피(fully volumetric) masked-autoencoder(MAE) 기반 tokenizer를 제안해 encoder와 decoder를 분리한다. frozen 3D MAE encoder는 임상적으로 유의미한 임베딩을 만들고, 별도의 CNN decoder가 임베딩의 선형 projection에서 볼륨을 복원한다. 이렇게 얻은 단일 임베딩 공간을 conditional diffusion transformer(DiT)의 조건 입력과 downstream 임상 태스크에 동시에 활용할 수 있게 했다.

- **Technical Challenges**: 핵심 기술 난제는 3D에서 tokenizer가 (1) 임상 정보 보존과 (2) 해부학적 볼륨 충실도라는 상충 조건을 동시에 만족해야 한다는 점이다. 이를 위해 encoder는 MAE의 masked patch 예측으로 전역 해부학적 맥락을 학습하도록 고정하고, decoder는 bottleneck d′=32의 선형 projection을 받아 voxels를 재구성하도록 학습해 encoder drift를 차단했다. 이후 DiT는 flow matching과 adaLN-Zero 조건부 제어, classifier-free guidance로 질병/성별/모달리티/사이트/나이/IDH1 변수를 controllable하게 반영하도록 설계했다.

- **Empirical Impact**: 35,309개 볼륨(18개 코호트, 4개 모달리티, 200+ acquisition site)에서 선형 probing을 수행한 결과, frozen MAE encoder는 23개 태스크 중 21개에서 BrainIAC, BrainSegFounder, MedicalNet과 비교해 SOTA급 성능을 보이거나 동등했다. 또한 DiT 생성 샘플에 대해 real-data probe로 조건 복원력을 확인했으며, cond–null 신호가 변수별 probe 특성과 일치하는 형태로 나타났다. 더 나아가 동일 임베딩 공간을 재사용해 환자별 longitudinal forecasting에서도 나이 진행의 방향과 위치(예: 뇌실/고랑)를 재현하되, 크기는 약 27% 수준으로 감쇠되는 것으로 보고돼 임베딩의 “임상 표현성-생성 제어성” 연결을 실증했다.



### AI4SE and SE4AI Exploration: A Decade Looking Back and Forward (https://arxiv.org/abs/2606.19630)
Comments:
          10 pages, 5 figure

- **Prior Approaches**: AI4SE/SE4AI 프레임워크를 중심으로 AI가 시스템공학을 돕는 방향(AI4SE)과, AI 시스템을 공학적으로 설계하는 방향(SE4AI)이 병렬로 전개돼 왔다. 기존 문헌은 로드맵·정의·개념화에는 강했지만, 실제 산업 수준의 검증과 시스템공학 고유 문제를 명확히 다룬 증거는 상대적으로 부족하다고 지적한다.

- **Core Contribution**: 이 논문은 2020~2025를 foundational(개념 정립)→applied(적용/실험)→LLM inflection(LLM 기반 가치·검증 중심 전환)으로 나눠 분야 진화를 정리한다. 동시에 1,712개 INSCOSE INSIGHT 글과 889개 SERC 출판물을 대상으로 인간 전문가와 6개 AI 모델의 ‘관련성 판단 합의도’를 측정해, “분야 근거가 기대만큼 쌓였는가”를 데이터로 점검한다. 그 결과와 함께 AI4SE/SE4AI Explorer 및 합의 데이터셋을 공개해 독자가 자신의 판단을 비교·검증할 수 있게 했다.

- **Technical Challenges**: 핵심 난관은 AI가 ‘시스템공학 맥락의 관련 문헌’을 얼마나 안정적으로 선별하느냐였다. 연구진은 제목만으로 triage를 수행한 뒤(INSIGHT: title-only, SERC: title-only/title+abstract) abstract 제공에 따른 판단 변화율과 Cohen’s kappa로 인간과의 일치·일관성을 함께 봤다; proprietary 모델은 비교적 안정적이었지만 local 오픈소스 모델은 모델 의존적으로 변동이 컸다. 즉, 생성 성능보다도 ‘검토·분류 신뢰 경계’를 정량화하는 설계가 이 연구의 기술적 포인트다.

- **Empirical Impact**: 인간-모델 합의로 뽑힌 INSCOSE 관련 33편은 나머지 대비 평균 인용 9.5회 vs 2.4회로 인용 임팩트 차이가 나타났고, 2020 특별호에서 인용이 특히 높았다. 분야 전반에 대해서는 채택 패러다임이 augmented intelligence(도구는 초안/초기안, 최종 검토는 인간)로 수렴하고 AI4SE와 SE4AI의 병행 필요성이 공감대가 됐다고 정리한다. 동시에 실증 검증 부족, 시스템공학-소프트웨어 공학 혼동, 도메인 벤치마크 부재, AI 생성 산출물 traceability(추적성) 미성숙, 인력 전환 로드맵 부족 등 5대 공백을 명확히 제시해 향후 연구 의제를 구체화했다.



### Toten: Knowledge-Based Ontological Tokenization Of Physical Quantities And Technical Notation In Brazilian Portugues (https://arxiv.org/abs/2606.19626)
- **Prior Approaches**: 기존 Byte-Pair Encoding, WordPiece, SentencePiece 같은 통계 기반 토크나이저는 어휘 압축에 최적화되어 공학적 수량·단위·숫자·기호식을 의미 구조와 무관하게 subword로 쪼갤 수 있다. 그 결과 토크나이저 단계에서 물리량/단위의 결합 규칙과 수치 표기(로케일, 지수, 소수점, 자리 구분)가 깨지고, 재조립은 downstream 모델에 거의 전가된다. 한편 수량 추출이나 Pint/udunits-2 같은 차원 라이브러리는 대체로 이미 분리된 단위 문자열을 전제하거나, 일반 엔티티 인식은 person/location 같은 범주 위주로 기술-과학 PT-BR 표현을 명시적으로 모델링하지 못한다.

- **Core Contribution**: 이 논문은 지식 기반 온톨로지 토크나이제이션 프레임워크 TOTEN을 제안하며, 통계적 분할 대신 Engineering entities의 형식 온톨로지(OEE)에 근거해 ‘타입 분류→구조화’로 텍스트를 처리한다. TOTEN은 온톨로지(O), classify(원문을 타입 영역으로 매핑), instantiator family(자기-설명형 구조 표현 생성)의 삼중 구조 <O, classify, {inst_τ}>로 형식화되어, 의미 보존을 construction 단계에서 강제한다. 또한 Pint(차원), Unicode Character Database(타이포그래피), RSLP(포르투갈 형용/어근 처리)를 결정론적으로 결합해 규칙 기반 권위를 시스템에 고정한다.

- **Technical Challenges**: 핵심 난제는 ‘수량·단위·기호식’이 텍스트에서 다양한 표기 관례(로케일 숫자, 첨자/차수, 복합 단위, 문장 내 연산자 주변 구조)를 갖는데도 토크나이저가 의미적으로 원자적(atomic) 태그를 유지하도록 규정하는 것이다. 논문은 OEE의 primary types(예: physical quantity, symbolic expression, hierarchical reference 등)와 8개의 구조 원칙(합성 가능 조건, 범주 오류 금지, 불변량 보존 등)을 정의하고, classification이 먼저 타입을 결정하면 instantiation은 포맷만 수행하는 단일 권위(single authority) 구조로 ‘점진적 품질 저하’ 대신 categorical error를 전파한다. 평가 가능한 입력-출력 규약을 위해 출력 언어 ℳ는 BNF 기반 타입 태그와 원문 잔여 텍스트의 literal preservation(타입 영역만 구조화)을 포함하며, 수치 표준화(canonicalization)로 IEEE 754 표현 차이도 제어한다.

- **Empirical Impact**: intrinsic evaluation로 온톨로지 원자성, 차원 동치, 타이포 견고성, 수치 재구성(4가지)을 construction으로 검증하며 EngQuant(N=800)와 PT-BR 외부 코퍼스 4종(총 eligible 사례 1771)을 대상으로 수행했다. 그 결과 TOTEN은 비교 기준에서 unit ontological atomicity를 압도적으로 달성하고, 외부 코퍼스에서 numerical reconstruction 0.775~0.904를 기록해 최강 baseline인 Quantulum3(0.627~0.703)보다 높았으며 EngQuant에서는 0.780 vs 0.340으로 격차가 컸다. 또한 McNemar+Holm 보정의 통계적 유의성과 내부·외부 랭킹의 Spearman 상관으로 타당성을 확인했으며, dimensional equivalence는 Pint 권위를 그대로 계승해 통계적으로 동등 수준을 보였다.



### Which Pairs to Compare for LLM Post-Training? (https://arxiv.org/abs/2606.19607)
- **Prior Approaches**: 기존 RLHF는 참조 정책에서 소량의 완성문을 생성한 뒤, 가능한 비교쌍을 모두 라벨링하거나 일부만 균등/기계적으로 선택해 편집 없이 수집한다. 또한 보상모델 학습과 PPO 같은 강화학습 단계를 거치는 파이프라인이 복잡하고 비용이 커, 레이블 예산이 병목이 되는 문제가 반복됐다. DPO는 보상모델을 생략해 간결해졌지만, 여전히 “어떤 비교쌍을 라벨링할지”가 실전에서 사전 고정되는 경우가 많다.

- **Core Contribution**: 이 논문은 preference-based post-training에서 라벨링 예산을 어떻게 배분해 비교쌍을 선택할지, 즉 comparison curation을 sampling-design 문제로 정식화한다. 특히 DPO의 학습 목적함수 아래에서, 최종 정책의 성능을 끌어올리기 위해 어떤 비교쌍을 더 “정보가 많은 방식”으로 라벨링해야 하는지 설계 기준을 제시한다. 저자들은 DPO 학습 과정이 선택된 라벨링 쌍의 효과를 downstream 성능으로 전파하는 방식을 분석해, 설계에 의존하는 단 하나의 정보 행렬로 영향이 요약된다고 보인다.

- **Technical Challenges**: 핵심 난관은 비교쌍을 잘 라벨링하는 것 자체가 아니라, 라벨링 분배가 DPO가 추정하는 파라미터 오차와 KL-정규화 RLHF 목적의 서브옵티멀리티(최적성 갭)에 어떻게 이어지는지를 연결해야 한다는 점이다. 저자들은 Bradley–Terry(Bradley–Terry model) 기반 선호 라벨 생성과 식별/커버리지/특징 분리(complete candidate pool의 정보성) 같은 가정을 두고, DPO로 학습된 정책의 RLHF 최적성 갭에 대해 유한표본 상·하한을 “matching” 형태로 증명한다. 이 과정에서 설계-dependent 정보 행렬이 파라미터 추정 오차→정책 비최적성으로 흐르는 경로를 제공하며, 결과적으로 예산 제약 하에서 비교쌍을 고르는 명시적 최적화 기준과 실용적 샘플링 정책을 도출한다.

- **Empirical Impact**: 실험은 합성 환경과 언어모델 post-training 벤치마크(예: IMDb, Anthropic-HH)에서 제안된 설계 기준이 일반적인 비교 선택 휴리스틱보다 일관되게 sample efficiency를 개선함을 보여준다. 특히 후보 완성문 풀을 크게 만든 뒤 “가장 유익한 비교쌍만 라벨링”하는 오프라인 워크플로가 효과적으로 작동한다는 점을 경험적으로 뒷받침한다. 요약하면, 라벨 예산이 제한된 preference 수집에서 비교쌍 선택을 체계화하는 설계 관점이 downstream 정렬 성능을 직접 끌어올리는 실증적 근거가 된다.



### Configurable Clinical Information Extraction with Agentic RAG: What Works, What Breaks, and Why (https://arxiv.org/abs/2606.19602)
- **Prior Approaches**: 기존 임상 정보 추출(IE)은 cTAKES 같은 규칙 기반 파이프라인에서 시작해 BioBERT/GatorTron처럼 타깃별 fine-tuning으로 확장됐지만, 타깃 고정·개발자 개입이 일반적이었습니다. LLM 기반 few-shot 추출은 가능성을 보였으나 실제 배포는 연구 평가에 머무는 경우가 많고, 표준 RAG는 문서 간 의존성과 시간 추론, 불완전한 메타데이터에 취약합니다. 또한 agentic RAG가 떠오르고 있지만, 대체로 연구자가 정의한 타깃/벤치마크 중심이라 임상의가 스키마를 설정하는 배포 사례는 드뭅니다.

- **Core Contribution**: 이 논문은 온프레미스 에이전트ic RAG 파이프라인 ACIE(Agentic Clinical Information Extraction)를 University Medicine Essen에 배치해, 환자 맥락 전체를 근거 기반으로 추출·검증하는 임상 IE를 제시합니다. 임상의는 typed schema로 74개 추출 필드를 구성하고, 모든 값은 반드시 출처 문장에 ground하여 검증 가능성을 높입니다. 특히 메타데이터가 AI가 필요로 하는 수준에 못 미친다는 ‘메타데이터 갭’을 정량화하고, 그 현실이 아키텍처 의사결정에 어떻게 반영됐는지도 함께 추적합니다.

- **Technical Challenges**: 주요 난제는 (1) 문서 수준 메타데이터가 희소·부정확해 retrieve-then-generate의 메타데이터 필터링이 잘 안 먹히고, (2) 환자 기록은 문서 간 얽힘과 중복·오기록이 있어 시간 추론이 필요하며, (3) 정보량이 매우 다양한 문서들을 효율적으로 훑어야 한다는 점입니다. 논문은 메타데이터 의존 대신 에이전트가 환자 전체에서 콘텐츠 기반 탐색을 반복하고, 문서 요약 프리뷰로 읽기 범위를 줄이며, 짧은 조각이 과대 점수 받는 문제를 길이 페널티로 완화해 재현성을 확보합니다. 또한 입력/출력 직렬화 포맷 이슈까지 관찰해 JSON 대비 markdown 전환으로 도구 호출 실패를 제거했습니다.

- **Empirical Impact**: 독립적인 후향적 림프종 레지스트리(99명, 7,326개 판단)에서 원자명 핵의학 전문의가 각 값과 인용 근거를 매번 검증한 결과, 임상의 수용률은 96.5%였고(필드 유형별 80%~99%), 출처 없는 hallucination은 보고되지 않았습니다. 약점은 날짜와 표(다행/치료 타임라인)처럼 전 기간에 걸친 시간 조립과 이벤트 선택이 필요한 항목에서 더 자주 발생했으며, ‘제대로 아는 값의 부재/누락 또는 시간상 오정렬’이 잔여 오류의 중심이었습니다. 전반적으로 배치형 근거 기반 추출의 실효성을 보여주면서, 실제 병원 데이터에서는 데이터 품질이 곧 시스템 설계 제약이 된다는 메시지를 강화했습니다.



### Analyzing the Narration Gap in LLM-Solver Loops (https://arxiv.org/abs/2606.19588)
- **Prior Approaches**: LLM-솔버 파이프라인은 SAT/SMT/ITP처럼 논리로 정식화 가능한 문제에 대해, LLM이 자연어를 논리식으로 바꾸고 솔버가 정답 여부를 판정한 뒤 이를 해석해 사용자에게 전달하는 구조로 발전해왔다. 기존 연구는 주로 “정식화”와 “결정(검증 가능한 verdict)”의 정확성을 보장하는 데 집중했고, 솔버 출력이 최종 사용자 답으로 번역되는 “narration(서술/해석) 단계”의 충실성은 충분히 다루지 않았다. 특히 증명서(certificate)로 보장되는 건 솔버의 verdict 구간이지, 사용자가 읽는 자연어 결론까지의 불변성은 아니다.

- **Core Contribution**: 이 논문은 LLM-솔버 루프를 “검증 가능한 decision procedure”로 형식화하고, 여기서 발생하는 narration gap(검증된 verdict과 사용자 결론 불일치)을 체계적으로 정의한다. 이어서 prompt injection이 verdict를 건드리지 않고도 “사용자에게 전달되는 결론”을 뒤집을 수 있음을 공격 목표 관점에서 모델링하고, 어떤 방어가 어떤 실패 모드를 줄이지만 무엇은 완전히 제거하지 못하는지도 이론적으로 정리한다. 결론적으로 robustness가 narration까지 자동으로 확장되지 않으며, 최종 답을 고정하려면 루프 내부의 enforcement가 필요하다는 결론을 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 “솔버는 sound하지만, narration을 담당하는 LLM이 untrusted 컨텍스트의 간접 지시를 받아 결론을 바꿔버리는” 상황을 검출·차단하는 것이다. 논문은 LLM-솔버 루프를 확률적 맵으로 추상화하고, 공격이 stealthy하게 verdict는 맞게 유지하면서도 결론만 flip하는 실패(stealthy failure)와, 결론·verdict가 함께 어긋나는 실패(compliant failure)를 구분해 모니터의 가시성 한계를 분석한다. 그 결과 prompt hardening이나 verdict 모니터만으로는 stealthy flip이 완전히 관측되지 않으며, verdict로부터 계산 가능한 결론을 강제로 대체하는 enforcer가 아니면 잔여 위험이 남는다는 정리가 나온다.

- **Empirical Impact**: 다섯 개 오픈소스 모델을 대상으로 다양한 prompt injection 템플릿을 평가한 결과, certificate gating은 솔버 판정의 soundness를 유지하지만 narration 단계에서는 결론 flip이 여전히 발생한다. 특히 소셜 엔지니어링 성격의 “가장 미묘한 injection”도 성능이 높았고, 정상적으로는 보이지 않는 형태로 stealthy하게 뒤집혀 검증 모니터를 통과하는 실패가 관찰됐다. 논문은 따라서 “formal tool 결합”을 안전·보안 임무에 적용할 때, 최종 자연어 결론까지 동일한 verdict를 기계적으로 강제하는 end-to-end enforcement 설계를 요구하며 관련 커뮤니티의 평가 프레임을 narration 포함으로 확장하는 데 의미가 있다.



### Uncertainty Decomposition for Clarification Seeking in LLM Agents (https://arxiv.org/abs/2606.19559)
Comments:
          26 pages, 8 figures. Source code: this https URL

- **Prior Approaches**: 기존 불확실성 추정은 대체로 aleatoric/epistemic 같은 틀에 의존하거나, 단일 턴 예측 정확도를 위해 설계된 logprob·다중샘플·학습 기반 보정이 중심이었다. 하지만 LLM 에이전트는 부분관측과 장기 상호작용에서 불확실성이 누적·전파되며, 불확실성의 성격(행동 난이도 vs 요청 모호성)이 섞여 에이전트의 “무엇을 해야 하는지(질문 vs 진행)” 판단을 어렵게 만든다. 또한 black-box API, 상호작용 지연(latency) 한계, 라벨 없는 trajectories 제약 때문에 logprob·multi-sampling·training 기반 접근은 실배포에서 제한되고, 결국 프롬프트 기반 신호 표출이 현실적인 선택지로 남는다.

- **Core Contribution**: 이 논문은 프롬프트 기반 불확실성 신호를 “행동 확신(action confidence)”과 “요청 불확실성(request uncertainty)”로 분해해, 모호한 요청에는 clarification을 요청하고 애매하지 않으면 행동을 이어가도록 설계한다. 특히 request uncertainty를 0/0.5/1의 anchored 척도로 모델이 스스로 판단하게 만들어, 애매함(underspecification) 여부를 대화 중 관측 가능한 행동(request_clarification)과 직접 연결한다. 이를 통해 단순히 실패 가능성을 가리는 수준을 넘어, 사용자와의 공유 mental-model 구축 같은 상호작용적 능력을 불확실성에서 끌어내려는 방향성을 제시한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 black-box API 환경에서 다중샘플·logprob 접근 없이도 ‘원인별 불확실성’을 안정적으로 프롬프트로 추출하고, 이를 장기 에이전트 궤적에서 일관되게 전파하는 것이다. 저자들은 UAM처럼 불확실성(및 설명)을 history에 semantic propagation으로 누적하되, 기존 단일 scalar confidence를 (u_t, c_t) 두 신호로 확장해 단계별로 요청 모호성과 행동 진척도를 분리해 평가 가능하게 만들었다. 또한 trajectory-level 집계(예: 마지막 단계/곱/평균/보수적 규칙) 선택이 결과에 큰 하이퍼파라미터 효과를 주므로, 다양한 집계 전략으로 민감도와 한계를 체계적으로 다룬다.

- **Empirical Impact**: 평가는 50%가 의도적으로 underspecified된 WebShop-Clarification과 ALFWorld-Clarification을 포함해, 5개 LLM 백본(GPT-5.1, DeepSeek-v3.2-exp, GLM-4.7, Qwen3.5-35B, GPT-OSS-120B)에서 ReAct+UE와 UAM을 비교하는 방식으로 수행됐다. 평균적으로 ALFWorld-Clarification에서 제안 방법의 clarification F1이 ReAct+UE 대비 73%, UAM 대비 36% 개선됐고, WebShop-Clarification에서는 모든 백본에서, ALFWorld-Clarification에서는 5개 중 4개 백본에서 clarification 성능이 향상되며 일반성을 보여줬다. 즉, 프롬프트 기반 분해가 특정 모델에 국한되지 않고, 에이전트가 모호함을 스스로 감지해 질문으로 전환하는 능력을 실제 벤치마크에서 뚜렷이 끌어올릴 수 있음을 시사한다.



### ITNet: A Learnable Integral Transform That Subsumes Convolution, Attention, and Recurrenc (https://arxiv.org/abs/2606.19538)
- **Prior Approaches**: CNN, RNN, transformer는 각각 지역성·순차 메모리·콘텐츠 기반 쌍 상호작용이라는 서로 다른 귀납 편향을 내장하며, 수학적으로도 별개의 계열로 발전해 왔습니다. 그 결과 과제 유형에 맞춰 모델 계열을 사전에 선택해야 하고, 이미지·언어·시계열·포인트클라우드·멀티모달처럼 데이터가 섞일 때는 도메인별로 구조를 조합해야 했습니다. 또한 convolution/attention/recurrence를 하나로 정확히 포섭하는 통합 연산자에 대한 증거와 실용적 구현은 부족했습니다.

- **Core Contribution**: 이 논문은 세 계열의 차이가 ‘신호 처리 방식의 본질적 다양성’이 아니라, 하나의 학습 가능한 적분 변환(learnable integral transform)을 불완전하게 본 결과라고 주장합니다. Integral Transform Network(ITNet)는 위치와 특징을 함께 조건화하는 학습 커널로 통합 연산자를 만들고, 그 커널을 MLP로 구현해 데이터로부터 상호작용 형태를 학습합니다. 더 나아가 convolution, self-attention(멀티헤드 포함), autoregressive recurrence(RNN/LSTM/GRU/S4/Mamba 포함)가 적절한 파라미터화의 정확한 특수사례로 등장함을 이론적으로 보입니다.

- **Technical Challenges**: 통합 연산자를 “정확히” 기존 아키텍처로 축소시키려면, 커널이 위치·거리·쿼리/키 특징을 동시에 다루면서 정규화(예: softmax)와 인과성(causality)을 올바르게 강제해야 합니다. 이를 위해 ITNet은 질의 위치 x, 키 위치 y, u(x), u(y)를 입력으로 하는 행렬값 커널을 residual과 함께 적분 연산자로 정의하고, 위치 정보를 상대/절대 좌표 및 거리로 포함하되 커널 내부 MLP가 상호작용을 유연하게 학습하도록 설계합니다. 실용화를 위해 tiled kernel fusion, importance-weighted Monte Carlo 적분 근사, learned low-rank factorization으로 계산량을 스케일링 가능한 형태로 줄입니다.

- **Empirical Impact**: 단일 ITNet 아키텍처(공유 코어 연산 + 경량 modality-specific 인코더)로 ImageNet-1K, GLUE, ModelNet40, VQA v2, NLVR2 전반에서 기존의 전문화된 기준선과 동등하거나 더 좋은 성능을 보고합니다. 즉, 도메인별로 분리 설계했던 convolution/attention/recurrence의 행동을 하나의 학습된 상호작용 메커니즘이 데이터로부터 회복할 수 있음을 경험적으로 뒷받침합니다. 이는 ‘아키텍처 선택’ 중심 파이프라인을 줄이고, 멀티도메인에서 일관된 연산자로 일반화하려는 흐름에 설득력을 제공합니다.



### Emergent Alignmen (https://arxiv.org/abs/2606.19527)
Comments:
          Rejected from ICML 2026

- **Prior Approaches**: 기존 alignment은 보통 SFT, RLHF, 혹은 DPO 같은 학습 단계에서 행동을 “흉내 내게” 하거나, 강한 judge로 보상을 주는 방식이 많았습니다. 다만 Emergent Misalignment는 미세조정이나 프롬프트 환경이 바뀔 때 예기치 않게 비윤리 행동이 ‘새로’ 생기는 reward hacking 성격이라, 고정된 규칙·단일 평가로는 한계가 있다는 문제의식이 제시됩니다. 또한 sleeper agent처럼 잠복 상태는 탐지·교정이 어렵고, 평가를 지식/판단에만 의존하면 새로운 형태의 불일치에 취약할 수 있습니다.

- **Core Contribution**: 이 논문은 LLM이 자신의 출력과 추론이 윤리 기준에 맞는지 스스로 점검하는 conscience step(양심 단계)을 학습 루프에 내장하고, 이를 DPO로 역방향 신호(부정 예시)로 연결하는 Emergent Alignment(EA)를 제안합니다. 중요한 점은 자신보다 강하거나 약한 외부 judge 없이, frozen copy of itself(자기 자신의 고정 복사본)와 self-assessment 질문만으로 alignment를 부트스트랩한다는 점입니다. 결국 “윤리 리뷰”가 학습 중 반복되며 정렬이 성질처럼( emergent property) 따라오도록 설계합니다.

- **Technical Challenges**: 핵심 난제는 (1) self-assessment가 비윤리/윤리 판단을 안정적으로 만들어내고, (2) 그 신호가 과도해져 언어·과제 성능을 해치지 않으면서도 misalignment를 억제해야 한다는 것입니다. 논문은 SFT(생성 품질)와 DPO(정렬 손실)를 하나의 하이브리드 손실로 동시에 최적화하되, DPO 항의 가중치 λ를 매우 작게(예: 0.1 이하) 두어 capability tax를 줄이도록 구성합니다. 또한 학습 중 입력-출력에 대해 윤리 점검→부정 예시 생성 흐름을 유지하고, DPO는 reference model 대비 log-ratio로 계산해 모델이 기준선에서 과도하게 벗어나지 않게 했습니다.

- **Empirical Impact**: emergent misalignment fine-tuning 시나리오(코드 해킹 유도)에서 EA를 적용하면 alignment score 하락이 관찰되지 않았고, 코드 해킹 능력도 유지된다고 보고합니다. 평가는 별도의 더 큰 LLM judge(예: Qwen3-30b)로 응답을 반복 질의(24개 benign 질문을 100회)해 coherence가 일정 수준 이상인 경우의 alignment를 집계하는 방식으로 이뤄졌습니다. 더 나아가 misalignment 체크포인트에서 다시 완전 정렬로 복귀하는 현상과, conscience 질문을 Asimov 변형 4종으로 바꿔도 효과가 크지 않음을 보이며, 일부 sleeper agent는 깨운 뒤에는 자기점검으로 정렬 교정이 가능하다고 실험합니다. 전체적으로 학습/파인튜닝/제로샷 등 다양한 배치에 적용 가능한 “온라인 자기 방어” 설계라는 점에서 실무적 의미가 큽니다.



### REVEAL++: Differentiable Phenotypic Grouping for Vision-Language Retinal Modeling of Alzheimer's Disease Risk (https://arxiv.org/abs/2606.19522)
Comments:
          Accepted for publication at MICCAI 2026

- **Prior Approaches**: 기존 vision-language 정렬 기반 의료 모델은 망막 fundus 이미지와 임상 리스크 내러티브를 contrastive learning으로 묶어 AD 위험을 조기 예측해 왔다. 특히 REVEAL의 group-aware contrastive learning(GACL)은 비슷한 phenotypic profile을 가진 사람들을 multi-positive로 학습하지만, 유사도를 discrete한 그룹 경계(하드 threshold)로 나눠 고정된 감독을 주는 한계가 있었다. 이로 인해 표현 학습이 그룹 생성과 분리되고, 질병 위험이 연속적 스펙트럼이라는 생물학적 가정과 불일치할 수 있다.

- **Core Contribution**: 이 논문은 phenotypic similarity를 학습 과정에서 ‘연속 신호’로 다루도록 설계한 REVEAL++(differentiable phenotypic alignment) 프레임워크를 제안한다. 개인을 고정 클러스터에 할당하지 않고, 망막 임베딩과 위험 프로파일 임베딩의 intra-modality 유사도로부터 부드러운 가중치 W를 만들어 soft multi-positive 관계를 정의한다. 또한 soft-target contrastive objective로 cross-modal alignment과 phenotypic 구조를 end-to-end로 함께 학습한다.

- **Technical Challenges**: 가장 큰 과제는 연속적인 위험 스펙트럼을 contrastive 학습의 positive/negative 감독으로 안정적으로 변환하는 것이다. 연구진은 두 모달리티에서 계산한 cosine similarity를 sigmoid gating과 differentiable probabilistic union으로 결합해 W∈[0,1]^{N×N} 형태의 soft pairwise target을 만들고, W 값에 따라 감독 강도가 매끄럽게 변하도록 설계했다. 이를 통해 hard group boundary가 표현 학습을 강제하지 않으면서도, 망막-임상 간 정렬과 인구집단 구조 학습을 동시에 수행한다.

- **Empirical Impact**: UK Biobank 망막 데이터 기반 incident AD 예측 실험에서 REVEAL++는 discrete group 기반 contrastive 학습과 표준 vision-language 베이스라인을 일관되게 능가했다. RETFound 등 foundation 모델을 사용한 정렬은 의미 있는 망막-텍스트 대응을 학습하지만, phenotypic 구조를 명시적으로 모델링하지 못해 장기 위험 예측에서 성능 격차가 났다는 해석이다. 즉, phenotypic similarity를 learnable continuous signal로 다루는 접근이 더 응집력 있는 멀티모달 임베딩 기하를 만들고 조기(장기) 위험 분류에 유리하다는 점이 실증됐다.



### LLM Doesn't Know What It Doesn't Know: Detecting Epistemic Blind Spots via Cross-Model Attribution Divergence on Clinical Tabular Data (https://arxiv.org/abs/2606.19509)
Comments:
          Accepted at EIML@ICML 2026

- **Prior Approaches**: 기존 연구는 EHR 같은 구조화 임상 데이터에서 LLM의 예측 성능(예: AUROC, 정확도)과 보정(calibration)을 주로 비교해 왔습니다. 또한 불확실성 추정은 verbalized confidence나 샘플링/집계, 온도 스케일링 같은 로그잇 기반 보정이 중심이었지만, 실제로 LLM이 자신의 오류를 인지하는 ‘epistemic self-awareness’는 충분히 검증되지 않았습니다. 설명가능성(예: SHAP vs LIME)에서의 attribution 불일치는 다뤄졌지만, 이를 불확실성 신호로 재활용한 연구는 드뭅니다.

- **Core Contribution**: 이 논문은 structured clinical prediction에서 LLM이 “자기가 틀릴 때를 아는지”를, 모델 간 attribution divergence(설명 불일치)를 줄여 epistemic uncertainty를 완화하는 관점으로 정면 질문합니다. Qwen 2.5 7B와 XGBoost를 동일한 AKI(MIMIC-IV) 예측 문제에서 비교하고, LLM이 맞출 때와 틀릴 때의 ‘추론 지향(어떤 특징에 주목하는지)’과 ‘자기평가(얼마나 신뢰하는지)’ 구조를 분해해 보여줍니다. 특히 verbalized confidence를 대체하는 cross-model calibrator로, LLM 내부 접근 없이도 환자별 reliability estimate를 만들 수 있음을 제안합니다.

- **Technical Challenges**: 핵심 난제는 (1) LLM이 구조화 입력에서 어떤 특징을 ‘올바르게’ 참고하는지 측정하기 어렵고, (2) LLM이 말로 표현하는 confidence가 실제 정확도와 연결되는지 확인하기 어렵다는 점입니다. 이를 위해 XGBoost의 SHAP 기반 특징 기여와 LLM의 self-reported top-K 특징 랭킹을 비교하는 Attribution Disagreement Score(ADS)를 정의하고, 이를 calibrator의 입력 특징으로 사용합니다. 또한 few-shot 예시 주입과 SHAP 기반 feature evidence 주입이 서로 독립적인 epistemic gap을 어떻게 메우는지(중첩/상호작용)를 실험으로 분해해 super-additive 효과를 검증합니다.

- **Empirical Impact**: 실험 결과 LLM의 verbalized confidence는 예측 품질과 무관하게 거의 상수(0.856~0.937)로 나타나, 임상 탭ular 예측에서 신뢰 신호로 거의 쓸모가 없음을 확인했습니다. 반면 few-shot + SHAP 주입은 attribution disagreement를 크게 줄이며 정확도를 49%에서 75.3%까지 끌어올려(모델 업데이트 없이) structured 근거 기반 정렬이 유효함을 보여줍니다. 더 나아가 cross-model calibrator는 ECE를 0.254에서 0.080로 대폭 감소시키며, ‘LLM이 맞을 확률’에 가까운 환자별 reliability estimate로 calibration을 개선했습니다. 전체적으로 “structured 데이터에서 LLM의 cold start 문제(방향/자기평가 부재)”를 경험적으로 규명해, 임상 안전 배치에서 신뢰 추정의 새 경로를 제시했다는 점에서 의미가 큽니다.



### DeXposure-Claw: An Agentic System for DeFi Risk Supervision (https://arxiv.org/abs/2606.19501)
- **Prior Approaches**: 기존 접근은 LLM 에이전트를 원시 on-chain 데이터나 텍스트를 직접 읽고 추론하게 하거나, 프로토콜별 노출 변화 같은 상대 지표로 위험을 정렬하는 경우가 많다. 이 방식은 약한·오래된·불완전한 근거를 그럴듯한 설명으로 과독해해 높은 단계 개입(고위험 조치)을 유발할 수 있고, 평가 역시 규제자가 실제로 우선순위를 매기는 손실 관점과 false alarm를 정량화하기 어렵다. 결과적으로 “어떤 개입이 쓸모 있었는가”가 아니라 “무엇이 그럴듯해 보였는가”에 치우칠 위험이 있었다.

- **Core Contribution**: 논문은 DeXposure-Claw라는 예측-근거 기반 agentic supervision 시스템을 제안한다. LLM이 원시 데이터를 직접 해석하는 대신, DeXposure-FM(그래프 시계열 파운데이션 모델)이 미래 노출 네트워크를 예측하고 이를 모니터링·스트레스 시나리오·불확실성 추정으로 구조화한 ‘typed evidence’를 통해서만 감독 티켓을 작성한다. 또한 데이터 품질과 신뢰도 게이트로 개입 단계 출력을 제한해, 모든 티켓이 감사 가능한 근거 묶음과 함께 발행되도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 그래프 형태의 미래 노출을 신뢰할 수 있게 예측하면서 (2) 그 예측이 약할 때 LLM이 과잉확신으로 고위험 조치를 하지 않게 만드는 것이다. 논문은 여러 모니터 통계(예: PageRank/HHI/지니 기반 집중도)와 표준화된 스트레스 시나리오를 예측 샘플에 적용해 시나리오 손실(CVaR)과 attribution 신호를 생성하고, 데이터-health 및 confidence gate로 escalation을 사전에 차단한다. 또한 DeXposure-Bench의 decision 축에서 규제자 정렬형 absolute-loss ground truth와 명시적 false-intervention rate로 평가해, “설명 그럴듯함”이 아니라 “규제 관점 개입 정확도”를 측정 가능하게 했다.

- **Empirical Impact**: 실험은 주간 DeFi 노출 그래프 5년치 데이터에서 진행됐으며, 예측-근거 라우팅을 통해 티켓 F1이 대폭 개선되는 결과를 제시한다. 다만 안전성(과잉 개입)은 모델을 더 강하게 해도 근본적으로 해결되지 않고, 데이터-health와 confidence 게이트가 false-intervention 위험을 통제하는 역할을 한다는 점이 강조된다(개입 오판 비율이 모델 간에도 큰 차이를 보이지 않음). DeXposure-Bench는 예측 품질, 경고/불확실성 캘리브레이션, 스트레스 충실도, 티켓 품질, 견고성까지 6축 평가를 제공해 규제 친화적 검증 틀을 만들었다는 의미가 있다.



### Hidden Anchors in Multi-Agent LLM Deliberation (https://arxiv.org/abs/2606.19494)
Comments:
          13 pages, 6 figures, 7 tables

- **Prior Approaches**: 기존 LLM 다중 에이전트 deliberation 연구는 라운드별 궤적을 동역학으로 모델링하기보다, 성능 향상을 위한 프레임워크 설계에 집중해왔다. DeGroot, Friedkin–Johnsen, Hegselmann–Krause 같은 opinion-dynamics의 고전 합의 규칙을 기준선으로 놓으면, 매 업데이트가 초기 신념의 convex hull(볼록껍질) 안에 머물러야 해서 LLM deliberation의 이상 현상을 설명하기 어렵다.

- **Core Contribution**: 이 논문은 multi-agent LLM deliberation을 closed-loop 동역학으로 모델링하며, 각 에이전트에 관측되지 않는 hidden internal belief인 anchor(고정된 내부 믿음)가 존재한다고 제안한다. anchor는 이웃의 의견과 무관하게 에이전트의 의견을 끌어당기므로, 고전 consensus 규칙이 금지하던 ‘정답 클래스 확신이 초기 범위를 넘어서는 escape’를 유발할 수 있다. 또한 deliberation 궤적만으로 anchor를 system identification으로 복구하고, 복구된 anchor가 held-out run을 예측하는지로 모델이 실제로 anchor에 의해 구동되는지 검증한다.

- **Technical Challenges**: 핵심 난제는 LLM prompting 과정에서 내부 믿음이 직접 관측되지 않는데도, 라운드별 관측 궤적만으로 anchor의 존재와 영향을 분리해 추정하는 것이다. 논문은 anchor를 consensus gain과 분리된 제어항으로 넣되, 파라미터 비식별성 문제를 다루기 위해 선형화(reparameterisation)로 least squares 회귀를 구성하고, recovered anchor는 simplex 투영으로 물리적 확률공간에 맞춘 뒤 신뢰도(β^i가 작을 때의 불안정성)도 함께 보고한다. 마지막으로 초기 구간/후기 구간에서 anchor가 안정적인지 여부로 compliance와 internalisation을 구분하고, leave-one-seed-out으로 일반화 성능을 평가한다.

- **Empirical Impact**: 세 가지 open-weight 모델군(Llama-3.1-70B, Qwen3-32B, gpt-oss-20b)과 다중 질의(증상→질병) deliberation에서, in-sample 적합도는 전체 anchor 모델이 기준선을 앞서지만 결정적 검증은 held-out에서 나타났다. Llama-3.1-70B는 held-out에서도 anchor 기반 모델이 강하게 선택되며, Qwen3-32B는 anchor 신호가 약해 거의 선형 기준선과 비슷해지고, gpt-oss-20b는 기준선이 이기며 anchor 모델이 오히려 held-out 성능을 악화시켰다. 특히 anchor의 ‘위치’가 문제였는데, Llama는 anchor가 초기 의견의 볼록껍질에서 멀리 떨어져 있어 deliberation이 escape를 일으키는 반면, gpt-oss는 anchor가 초기 의견 근처에 있어 Friedkin–Johnsen 특수사례처럼 행동하며 escape가 제한됐다.



### Diffusion Language Models: An Experimental Analysis (https://arxiv.org/abs/2606.19475)
- **Prior Approaches**: 기존 LLM은 left-to-right 자동회귀 생성으로 토큰을 순차 예측해 왔고, 생성 지연이 길이에 비례해 늘며 한 번 내린 결정은 되돌리기 어렵다는 한계가 있습니다. 반면 Diffusion Language Models(DLMs)는 반복적 denoising으로 전체 시퀀스를 동시에 다듬어 병렬성·전역적 정교화를 노리지만, 과제별 평가 프로토콜과 추론 예산(denoising steps 등)이 달라 비교가 어려웠습니다. 또한 diffusion 계열은 성능-비용의 스위치가 학습보다 추론 시 매개변수에 크게 의존해, 논문마다 선택이 섞이면 “아키텍처 개선”인지 “디코딩 설정”인지 분리가 불명확했습니다.

- **Core Contribution**: 이 논문은 최신 DLM 8종을 추론 품질과 계산 효율을 함께 보는 통일된 실험 프로토콜로 평가해, 아키텍처 패러다임별(순수 diffusion vs block-hybrid vs 자동회귀 기준) 강점과 약점을 정량 비교합니다. 더 나아가 denoising steps, context length, block size, parallel unmasking 같은 추론 시간 설계요소가 모델 거동을 어떻게 바꾸는지 체계적으로 분해 분석합니다. 통제된 조건에서 더 작은 모델들을 추가로 학습해, 대규모 결과가 “데이터/학습 차이”가 아닌 “구조·추론 설정 차이”에서 기인함을 확인하려는 점이 특징입니다.

- **Technical Challenges**: 핵심 난제는 DLM의 성능이 반복 denoising 횟수와 unmasking 스케줄 등 추론-time 하이퍼파라미터에 민감한데, 기존 연구는 이를 동일 조건에서 스윕하거나 보고하지 않아 공정한 비교가 어렵다는 것입니다. 저자들은 벤치마크를 지식·추론·코딩·번역·구조화 문제해결로 넓히면서도 한 프레임워크(lm-evaluation-harness) 아래에서 평가하고, generation length/step budget/block granularity를 고정·변형하는 controlled comparison으로 품질-효율 곡선을 분리합니다. 또한 순수 diffusion과 block-diffusion의 계산비용(메모리·연산량)을 forward 1회와 전체 생성 관점에서 함께 제시해 배포 관점의 trade-off를 드러냅니다.

- **Empirical Impact**: 대규모 결과에서 순수 full-sequence diffusion은 전역 제약 만족과 지식/추론 과제에서 강한 경향을 보였고, block 기반 하이브리드는 GSM8K·HumanEval 같은 알고리즘·코딩 류에서 두각을 보이되 언어/번역 성능에서는 약화되는 ‘과제 특화’ 패턴이 확인됩니다. 또한 추론 스케일링 실험에서 steps와 context length의 상호작용이 크며, 특정 길이 이후에는 성능 포화 또는 하락이 나타났고 번역은 특히 긴 생성에서 급격히 무너질 수 있음을 보여줍니다. 결론적으로 이 연구는 “모델”뿐 아니라 “생성-time 설계”가 DLM의 성능 경로를 좌우한다는 점을 실증해, 실제 배포 시 어떤 계산 예산에서 어떤 품질을 기대할지 가이드하는 데 의미가 있습니다.



### Measuring Curriculum Alignment across Topical Coverage, Competency, and Cognitive Depth: A Longitudinal Framework Applied to CS2013 and CS2023 (https://arxiv.org/abs/2606.19469)
Comments:
          24 pages, 5 figures, 8 tables

- **Prior Approaches**: 기존 연구는 강의계획서/강의성과를 지침의 지식영역에 투영하거나(topic model, 수작업 태깅, 온톨로지 등) NLP·LLM으로 텍스트 정합을 수행해 커버리지를 추정해 왔습니다. 하지만 대부분 신뢰도(재현성) 검증이 부족하고, 어떤 retriever/모델이 적합한지 벤치마킹하지 않으며, 결과를 “확정값”으로 취급해 인간 검증을 확률적으로 분리하지 못했습니다. 또 주로 토픽 중복에 그쳐, 역량이 실제 학습목표로 명시됐는지와 지침이 권장한 인지적 깊이에 도달하는지까지는 측정하지 못했습니다.

- **Core Contribution**: 이 논문은 외부 지식체(ACM/IEEE/AAAI Computer Science Curricula 2013, 2023)에 대해 교육과정의 커버리지를 측정하는 human-in-the-loop 파이프라인을 제안합니다. 프로그램을 구조화된 코퍼스로 만들고, 지침의 knowledge unit을 구조화해 semantic retrieval로 후보 매칭을 만든 뒤, 사전에 정의한 coverage 규칙으로 사람이 최종 확정하여 매핑의 재현성을 높였습니다. 또한 같은 “retrieve-then-confirm” 설계를 커버리지(토픽)에서 역량 명시(competency articulation), 인지적 깊이(cognitive depth)까지 확장하고, 서로 다른 지침 세대(CS2013↔CS2023) 간 종단 비교까지 수행합니다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 강의 텍스트와 지침 텍스트가 서로 다른 어휘 체계를 쓰기 때문에 신뢰도 있게 매핑하기 어렵고, (2) 긴 문맥을 가정한 모델보다 짧은 텍스트 매칭에서 retriever 선택이 성능에 큰 영향을 주는데 이를 “가정”하면 안 된다는 점입니다. 연구진은 knowledge unit을 이름+topic 묶음으로 표현해 짧은 비교를 가능하게 하고, 7개 retriever를 recall/MRR/nDCG 등과 “확정에 필요한 shortlist 깊이”까지 포함해 풀링 기반 벤치마킹한 뒤 reciprocal-rank-fusion 앙상블을 후보 생성에 사용했습니다. 마지막으로 독립 2인 레이터로 kappa 기반 일치도를 산출해 사람이 확정한 맵의 안정성을 검증했습니다.

- **Empirical Impact**: 적용 결과, 해당 학사 과정은 CS2023 지식유닛의 49.7%, CS2013의 50.9%를 커버하며 10년 간 커버리지 수준이 거의 일정하게 유지된 것으로 나타났습니다. 다만 역량 명시 관점에선 커버된 유닛의 약 88%에 대해 competency가 학습성과로 제시되지만, 권장 인지적 깊이 도달은 CS2023에서 76%로 떨어져(반면 CS2013은 95%) 격차가 “새 지침의 기대치 상승”에서 온다는 해석을 제시합니다. 종단 비교는 병렬·분산 컴퓨팅, 프로그래밍 언어 기초, 시스템 fundamentals 같은 구조적 공백은 두 지침 모두에서 지속됨을 드러내며, 이 측정 도구는 저자 요청 시 재사용 가능하다고 밝힙니다.



### Deontic Policies for Runtime Governance of Agentic AI Systems (https://arxiv.org/abs/2606.19464)
Comments:
          10 pages, 1 figure. To be published in the 2026 IEEE Symposium on Agentic Services which is part of the IEEE Conference on Web Services

- **Prior Approaches**: 기존 Agentic AI 거버넌스는 주로 LLM 바깥에서 per-invocation(행동 경계) 정책을 강제하는 방향으로 수렴했지만, 정작 “무엇이 허용/금지되는지(permit/prohibit)” 수준에 머무르는 경우가 많다. A2AS, Rego, Cedar, OPA 등은 허용·거부는 잘 다루지만, 허용 이후의 의무(obligations), 의무 면제(dispensations), 규칙 충돌 시 우선순위 같은 상위 거버넌스 구조를 기본 개념으로 제공하지 않는다.

- **Core Contribution**: AgenticRei는 deontic 정책(허가·금지·의무·면제)을 Rei 프레임워크 기반 OWL 온톨로지로 표현하고, LLM과 완전히 분리된 고성능 추론 엔진에서 런타임으로 평가한다. 또한 정책 평가 파이프라인을 도구 호출과 agent-to-agent 메시지(A2A) 모두에 동일하게 적용해 멀티 에이전트 환경의 거버넌스 결정을 일관되게 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 정책 요구사항이 단순 allow/deny를 넘어 의무 라이프사이클·면제 조건·충돌 해소 같은 “규칙에 대한 규칙(meta-policy)”까지 포함한다는 점과, (2) 도메인 개념 변화(예: 의료 데이터 하위 타입 증가)에 맞춰 정책을 유지보수 없이 확장해야 한다는 점이다. AgenticRei는 ⟨subject, action, resource⟩ 트리플로 행동을 추출한 뒤 RDFox 기반 OWL/RDFS 추론으로 온톨로지 계층을 자동 전개하고, metapolicy:RulePriority 같은 의미론적 우선순위로 충돌을 정리하며, 의무는 permission에 연결된 provision으로 실행 결과에 부착하는 방식으로 해결한다.

- **Empirical Impact**: 논문은 보안·프라이버시 거버넌스에서 의무·면제·충돌 우선순위·클래스 계층 추론이 기존 생산 엔진에서 거의 표현되지 못함을 예제로 보여준다. 또한 단일 호스트 프로토타입에서 쿼리당 전체 의사결정이 10ms 미만(추론 엔진 자체는 1ms 이하)으로 측정돼, 동기식 행동 경계 강제에 필요한 지연 시간 제약을 만족할 가능성을 제시한다.



### How Transparent is DiffusionGemma? (https://arxiv.org/abs/2606.20560)
Comments:
          20 main text pages and 6 pages of references and appendices

- **Prior Approaches**: 기존 연구는 주로 autoregressive 모델의 chain of thought(CoT)가 사람/감시자에게 해석 가능한 중간 노드로 작동한다는 점을 통해 투명성을 평가해 왔습니다. 특히 monitorability는 CoT를 활용한 감시자가 downstream 과제를 예측 가능한 형태로 수행하는지로 측정되는데, 현재 frontier 접근의 상당 부분이 이 축에 의존합니다. 다만 DiffusionGemma처럼 latent 공간에서 연산이 크게 일어나는 텍스트 diffusion 모델은 중간 상태가 자연어가 아닐 수 있어 “덜 투명하지 않을까”라는 우려가 컸습니다.

- **Core Contribution**: 이 논문은 diffusion 모델의 투명성을 variable transparency(중간 computational state 스냅샷을 이해할 수 있는가)와 algorithmic transparency(그 스냅샷으로 출력까지의 과정을 재구성할 수 있는가)로 분해해 체계적으로 묻습니다. 결론적으로 DiffusionGemma의 중간 bottleneck이 해석 가능하다는 가정 아래에서는, opaque serial depth를 autoregressive Gemma4 수준에 가깝게 낮출 수 있음을 보입니다. 또한 최종적으로 monitorability 관점에서도 Gemma4와 유사한 성능을 보이며, “latent 연산=무조건 비투명”이라는 단정은 경계하게 합니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 diffusion의 denoising step마다 캔버스 전체 토큰이 바뀌어, 감시자가 참조할 “해석 가능한 병목”이 희미해질 수 있다는 점입니다. 논문은 self-conditioning matrix의 정보를 token bottleneck으로 매핑하기 위해 Logit Lens 기반의 logit ablation(예: top-k 또는 확률 임계값 p)을 적용하고, 성능 저하 없이 중간 정보의 투명도를 끌어올리는 구성을 찾습니다. 이후 이 매핑된 중간 토큰이 실제로 최종/인접 토큰의 “추정(guess)”에 주로 해당함을 정량화해, 나머지 정보가 모니터링에 더 필요한지의 논의 지점을 마련합니다.

- **Empirical Impact**: 실험에서는 self-conditioning을 해석 가능 토큰으로 취급할 때 opaque serial depth가 Gemma4 대비 약 1.1X로 축소되는 반면, 단순 가정 없이 보면 약 28.6X로 크게 악화됨을 보여줍니다. 또한 ablation으로 제한했을 때는 capability benchmark 성능이 baseline과 거의 동일해지며, 중간 토큰의 대다수가 최종 또는 인접 위치 토큰과 잘 맞물린다는 신호(해석 가능성의 실마리)를 제공합니다. 마지막으로 monitorability 평가군에서 DiffusionGemma는 Gemma4와 유사한 결과를 내며, 확산형 모델도 CoT 투명성의 실용적 효익을 상당 부분 공유할 수 있음을 시사합니다.



### Structuring and Tokenizing Distributed User Interest Context for Generative Recommendation (https://arxiv.org/abs/2606.20554)
- **Prior Approaches**: 생성형 추천(Generative recommendation, GR)은 LLM 기반의 자기회귀로 사용자의 다음 상호작용을 예측하며 e-commerce, 광고, 스트리밍에서 성과를 보였다. 핵심 구성요소인 item tokenization(아이템 토큰화)에서 기존 그래프 연동 방식은 graph serialization이 LLM 입력이 길어져 비싸거나, GNN이 지역 서브그래프만 보아 holistic(전체) 정보를 충분히 못 쓴다는 한계가 있다. 또 semantic tokenization은 휴리스틱 학습 목표에 의존해 의미 토큰의 정합성이 보장되기 어렵고, 감독 신호가 부족해 최적 표현을 놓칠 수 있다.

- **Core Contribution**: 논문은 G2Rec을 제안해 사용자 관심 맥락을 그래프 기반 co-engagement(동시관여) 모델링과 의미 토큰화를 동시에 다루는 통합 프레임워크를 만든다. item-item co-engagement graph에서 얻은 관심 prototype과, 각 아이템의 soft(연속형) 클러스터 멤버십을 토대로 “관심 프로필” 토큰을 구성해, ground-truth 사용자 관심 라벨 없이도 더 포괄적이고 의미 정렬된 사용자 맥락을 학습한다. 결과적으로 산업용 sequential recommendation에서 정확도 향상과 모델링 일관성을 노린다.

- **Technical Challenges**: 첫째, co-engagement 그래프는 규모가 커서 원형(E) 크기가 O(M^2)로 폭발할 수 있는데, 이를 위해 그래프 라플라시안 보존을 목표로 이론적으로 근사 보존되는 희소화(sparsification)를 설계해 O(M log M) 수준까지 간선 수를 줄인다. 둘째, 각 아이템이 여러 관심으로 동시에 연결될 수 있어 hard 클러스터링이 부적합하므로, differentiable한 soft graph clustering을 위해 modularity의 연속형(soft modularity) 목적함수를 제안하고 GPU 친화적으로 계산되게 만든다. 마지막으로 학습을 위해 아이템을 “아이템-관심 프로필” 교대 시퀀스로 토큰화하고, 아이템 다음 예측 손실에 더해 관심 프로필 예측을 soft label로 함께 학습하는 구조를 제공한다.

- **Empirical Impact**: Meta의 제품 서피스에서의 온라인 A/B 테스트와 공개 데이터셋의 대규모 실험을 통해 G2Rec이 기존 생성형 추천 방법들 대비 우수함을 보인다. 특히 사용자 그래프 맥락을 전체적으로 반영하면서도 희소화·soft 클러스터링으로 산업 규모 효율을 유지하는 점이 강점으로 제시된다. 온라인 배포에서는 그래프 처리 엔진을 통해 클러스터링을 주기적으로 오프라인 수행해 실시간 응답 시간을 보호하며, 실제 프로덕트 성능 개선으로 이어졌다는 점에서 의미가 크다.



### SARLO-80: Worldwide Slant SAR Language Optic Dataset 80cm (https://arxiv.org/abs/2606.20523)
- **Prior Approaches**: 기존 SAR–광학 멀티모달 벤치마크는 주로 Sentinel-1 GRD처럼 강도(intensity)만 제공하고 지상 기준으로 투영된 저해상도 데이터를 사용해 왔습니다. 그 결과 복소 SLC가 담는 위상 정보와 layover/shadow 같은 네이티브 획득 기하를 충분히 보존하지 못해, SAR 고유 물리성 기반의 학습이 제한됩니다.

- **Core Contribution**: 본 논문은 Umbra spotlight 공개 데이터에서 2,500여 개 전 세계 장면을 기반으로, 복소값 SAR SLC(VV/HH)와 정렬된 고해상도 광학, 그리고 자연어 캡션을 한 샘플로 묶은 VHR SAR–optical–text 데이터셋을 제안합니다. SICD(Sensor Independent Complex Data)로 제공되는 센서 네이티브 정보를 활용해 복소 SAR을 유지한 채 slant-range 80cm 격자로 표준화하고, 광학 타일을 SAR 그리드에 국소 좌표 상응(affine)으로 워핑해 픽셀 정합을 맞춥니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 복소 SAR의 대역/주파수 성질을 깨지 않으면서 표준 격자로 일관 리샘플링하고, (2) SAR 네이티브 슬랜트-레인지 좌표계에서 광학 이미지를 지오메트리 왜곡 없이 로컬 정합하는 것입니다. 논문은 band-limited FFT 리샘플링(크롭/제로패딩)으로 네이키스트 조건에 맞춘 표준화와, 장면 메타데이터 기반의 조밀 지오로케이션 그리드 및 1024x1024 패치 단위 국소 affine 워핑을 결합해 이 문제를 해결합니다.

- **Empirical Impact**: 데이터셋은 총 119,566개의 트리플(복소+진폭 슬랜트레인지 SAR 패치, 정렬 광학 패치, SHORT/MID/LONG 캡션)을 제공하며, 72개 국가·257개 로케이션에 걸친 폭넓은 지형/인프라를 커버합니다. 공개 고정 split과 전처리/베이스라인 코드를 통해 네이티브 SAR 기하에서 cross-modal retrieval 및 텍스트 조건 생성의 가능성을 보였고, 단일 캡션보다 캡션 길이 다양성을 함께 쓰거나 후반 timestep reweighting을 적용할 때 생성 speckle의 사실성이 개선되는 경향을 보고합니다.



### Sovereign Execution Brokers: Enforcing Certificate-Bound Authority in Agentic Control Planes (https://arxiv.org/abs/2606.20520)
Comments:
          19 pages, 6 figures, 10 tables

- **Prior Approaches**: 기존 접근은 에이전트의 제안(action proposal)을 SAB(또는 유사한 assurance layer)가 인증(증명서 발급)하되, 실제 인프라 mutation이 일어나는 런타임 집행 지점에선 강제(enforcement) 장치가 부족한 문제가 있었다. 또한 IAM 같은 identity-centric 접근은 인증된 “행동”의 구체 계약(contract)과 무관하게 고정 자격을 신뢰하게 되어, LLM의 비결정성·프롬프트 주입 등으로 우회 위험이 커진다. 특히 인프라 상태가 인증 시점과 실행 시점 사이에 변하는 TOCTOU(시간검사-시간사용) 구간에서도 방어가 약하다.

- **Core Contribution**: 이 논문은 Sovereign Execution Broker(SEB)를 제안해, 인증서(certificate)에 묶인 실행 권한을 “mutation 시점”에만 강제하는 런타임 경계로 정의한다. SAB가 발급한 인증서 Ω를 SEB가 수신해 실행 요청이 인증된 execution contract와 일치하는지 검증하고, 짧고 폐기 가능한 실행 신원을 만들어서 인프라 API를 호출한다. 결론적으로 proposal–admission–execution을 분리해, 인증된 권한이 짧은 생명주기의 런타임 능력으로 전환되도록 만든다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 인증서를 실행 요청과 정확히 바인딩하고(작업/리소스/파라미터/nonce), (2) 유효기간·정책 epoch·revocation epoch·라이브-state drift·재플레이를 모두 점검하며, (3) 에이전트가 standing(상시) mutation 자격을 보유해 bypass하지 못하게 만드는 것이다. SEB는 VerifyExec 파이프라인에서 서명 검증, 계약 매치, 유효창 검사, 정책 버전 확인, revocation 확인, drift 판정, ledger 기반 nonce reservation/replay 방지, 그리고 scopeable 제약 가능성 여부를 선행하고, 실패 시 reject/re-admit/fail-closed 동작 규칙을 적용한다. 또한 platform이 네이티브로 제약을 표현하지 못하는 파라미터는 SEB가 broker-owned proxy/admission control로 보강해 “계약 제약”을 실제 실행 경로에 강제한다.

- **Empirical Impact**: 프로토타입을 Go 서비스로 구현해 AWS와 Kubernetes에서 평가했으며, 실행 지연(p50/p95/p99), 암호 검증 및 drift 검증 마이크로벤치마크, revocation 전파 지연, 장애(네트워크 분할·크래시 복구 등) 하에서의 보안 동작을 측정한다. 결과는 SEB가 인증서 기반 자율 제어에서 런타임 집행 강제와 감사(audit) 가능성을 제공하면서도, 실행 경로에 허용 가능한 오버헤드로 TOCTOU/재플레이/우회 시나리오를 줄일 수 있음을 보여준다. 이는 에이전트가 클라우드 운영에 깊이 관여하는 환경에서 “증명된 권한”을 실제 mutation 순간에만 유효하게 만드는 표준 참조 아키텍처로 의미가 있다.



### Efficient and Sound Probabilistic Verification for AI Agents (https://arxiv.org/abs/2606.20510)
- **Prior Approaches**: 런타임 모니터링 기반의 보안 검증은 Datalog 같은 형식 언어로 정책을 쓰고, 에이전트의 실행을 정책에 맞게 제약하는 방식으로 주목받아 왔다. 다만 기존 연구는 주로 결정론적 정책에 초점이 맞춰져 있어, PII 탐지기처럼 호출마다 실패 확률이 있는 불확실한 구성요소나 확률적 술어/전이를 자연스럽게 다루기 어렵다.

- **Core Contribution**: 이 논문은 불확실성이 포함된 보안 정책을 검증하기 위해, 분포적 강건 최적화(distributionally robust optimization)를 기반으로 정책 위반 확률의 ‘사운드한 상한’을 계산하는 프레임워크를 제안한다. 특히 술어들 사이의 상관관계(correlation)를 가정할 수 없는 현실 조건에서도, 위반 확률을 어떤 상관이 존재하더라도 보수적으로 보장하는 방식으로 안전성을 확보한다.

- **Technical Challenges**: 핵심 난제는 확률적 술어/전이를 가진 정책에서, 기존 확률 추론 Datalog 접근이 요구하는 독립성 가정을 충족시키기 어렵다는 점이다. 저자는 가능한 상관을 포함해 최악의 분포를 고려하는 분포적 강건 최적화로 전환해, 상관관계에 둔감한 사운드 상한을 효율적으로 계산하도록 설계를 구성했다.

- **Empirical Impact**: terminal 및 tool calling agent에 대한 표준 벤치마크에서 기존 방법보다 더 좋은 성능을 보이며, 보안(정책 위반 확률)과 유용성(utility) 사이의 trade-off를 개선함을 실험적으로 보였다. 동시에 정책 위반 확률에 대한 엄밀한 보장(상한)을 제공하므로, 불확실한 구성요소가 섞인 AI 에이전트 보안 검증의 실무 적용 가능성을 높인 것으로 평가된다.



### FreeStyle: Free Control of Style-Content Dual-Reference Generation from Community LoRA Mining (https://arxiv.org/abs/2606.20506)
Comments:
          35 pages, 26figures. Project page: this https URL

- **Prior Approaches**: 기존 Style-content dual-reference generation은 내용 기준(reference content)의 구조·의미를 유지하면서 다른 스타일(reference style)을 입히는 방식이지만, 콘텐츠 보존과 스타일 정렬이 동시에 요구돼 학습이 까다롭습니다. 특히 스타일 참조에서 의미가 새어 들어가는 semantic leakage를 막는 동시에 instruction following까지 만족해야 해서 모델이 쉽게 균형을 잃는 문제가 반복돼 왔습니다.

- **Core Contribution**: 이 논문은 FreeStyle이라는 확장 가능한 dual-reference 생성 프레임워크를 제안합니다. 커뮤니티 LoRA를 compositional anchor로 보고, 스타일과 콘텐츠를 깔끔히 분리한 대규모 triplet(Style-Reference, Content-Reference) 데이터를 생성·구성하는 파이프라인을 설계해 학습 재료의 병목을 완화합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 스타일 단계에서 스타일 참조의 누출을 억제하고 (2) 더 어려운 dual-reference 상황에서 위치 대응(positional correspondence) 기반 누출까지 잡아내는 것입니다. 이를 위해 두 단계 커리큘럼을 두고, 스타일-transfer 단계에서 attention-level enrichment constraint로 leakage를 억제하며, 어려운 단계에서는 frequency-aware RoPE modulation으로 positional-correspondence 기반 누출을 겨냥해 성능을 끌어올립니다.

- **Empirical Impact**: 저자들은 style-reference와 dual-reference를 함께 평가하는 벤치마크를 도입하고, style similarity·content preservation·aesthetics·instruction following·leakage rejection을 종합해 검증합니다. 또한 스타일 불변 Content Alignment Score(CAS)와 calibrated VLM 기반 Rejection Score를 사용해 생성 신뢰성과 누출 억제 효과를 계량화했으며, 실험에서 FreeStyle이 스타일 정렬·콘텐츠 보존·leakage suppression 사이의 균형을 강하게 달성했다고 보고합니다.



### Calibration Without Comprehension: Diagnosing the Limits of Fine-Tuning LLMs for Vulnerability Detection in Systems Softwar (https://arxiv.org/abs/2606.20502)
- **Prior Approaches**: 기존 취약점 탐지는 fuzzing, 침투 테스트, Static Analysis Tools(SATs) 같은 방법이지만, 커버리지 공백과 오탐/누락 문제가 커서 실배포 신뢰도가 낮다는 한계가 반복돼 왔다. LLM 기반 연구는 CVE/CWE 벤치마크와 fine-tuning을 시도했지만, (1) 데이터 오염 가능성, (2) 취약-패치 맥락 제거, (3) vulnerable–patched 페어링 부재, (4) 이진 정확도 중심의 평가로 인해 “추론인지 캘리브레이션인지” 검증이 어려웠다. 또한 함수 단위 스니펫만 다루거나 단일 아키텍처만 비교해 백본 사전분포 효과와 학습 효과를 분리하기 어렵다는 지적이 있었다.

- **Core Contribution**: 이 논문은 Linux 커널 취약점에 대해 CWE-Trace라는 프레임워크를 제안하며, 74개 CWE를 포괄하는 834개의 수동 검증 vulnerable–patched 샘플로 LLM의 보안 추론을 점검한다. 특히 pre-2025 역사 세트(PBD)와 2025 이후 leakage-free 세트(LFD)를 엄격히 분리해, 일반화와 memorization(암기)을 같은 평가 프로토콜에서 가른다. 평가도 비타깃/타깃 탐지와 CWE 계층 분류를 함께 수행해, 단순 분류 성능을 넘어 오류의 “방향”과 “깊이”를 진단하도록 설계했다.

- **Technical Challenges**: 핵심 기술 과제는 오염이 있어도 성능이 오르면 그것이 진짜 추론인지 구분하는 것이었고, 이를 위해 취약 기능/패치 기능을 코드 상호작용 단위로 묶고(멀티-함수/멀티-파일 의존 보존), 보안과 무관한 변경은 제외하는 수동 검증을 수행했다. 또한 함수 레벨에서 분포 이동과 의미 충돌(예: CWE misclassification)을 점검해 “CVE 단위 중복”이 곧 “학습 신호”로 이어지지 않음을 기계적으로 설명한다. 진단 지표로 Directional Failure Index(DFI)와 Hierarchical Distance and Direction(HDD)을 도입해, 정확도 평균 뒤에 숨은 편향 붕괴(한쪽으로만 판정)와 CWE 계층 수준의 얕은/깊은 실패 패턴을 분해했다.

- **Empirical Impact**: 실험 결과, 데이터 contamination은 성능에 측정 가능한 이점을 주지 못했으며, 함수 수준 분석에서 오염으로 보일 수 있는 샘플의 대부분은 실제 memorization 신호가 없거나 CWE 라벨이 충돌하는 것으로 나타났다(예: vulnerable 함수 부재/교차 매핑, 약 31% 수준의 CWE misclassification). 더 중요한 발견은 fine-tuning이 “이해”를 주지 못하고 출력 임계값/분포만 조정하는 경향이며, DFI가 -85.5~+94.8 pp 범위에서 역사-누출구간 전반에 걸쳐 안정적으로 유지돼 backbone의 방향성 사전분포가 결정적임을 시사한다. 최종적으로 탐지 최고 점수도 chance 대비 소폭 개선(예: 바이너리 detection 52.1%, +2.1 pp)과 함께, CWE Top-1 정확도는 1.3% 미만으로 떨어져 시스템 소프트웨어 보안 추론의 신뢰성 부족을 강하게 보여준다.



### Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems (https://arxiv.org/abs/2606.20493)
Comments:
          20 pages, 4 figures, 4 tables

- **Prior Approaches**: 기존 연구는 LLM-as-judge에서 나타나는 자리/장문/자기선호 증폭 같은 편향이 어떻게 의사결정 품질을 왜곡하는지에 초점을 맞췄지만, 편향이 에이전트 네트워크를 따라 ‘연쇄 전파’되는지는 잘 다루지 못했습니다. MM-EPC는 multimodal(텍스트↔비전)에서 evaluator 편향이 strategy 선택으로 오염되는 현상(MM-EPC)을 보여줬지만, 멀티 에이전트의 멀티-홉 전파는 미지였습니다.

- **Core Contribution**: 이 논문은 Contagion Networks라는 틀로, evaluator 편향이 상호작용하는 여러 LLM 에이전트 사이에서 어떻게 퍼지는지를 계량화합니다. 에이전트 토폴로지 전체에 대한 Cross-Agent Contagion Matrix ΓN을 정의하고, 스펙트럴 반경 ρ(ΓN)에 따라 suppression/persistence/cascade의 동역학적 조건을 정리합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) evaluator 편향을 ‘기준화된 업데이트 신호’로 수치화하고 (2) 그 신호가 여러 홉에서 누적될 때 안정/붕괴 조건을 도출하는 것입니다. 이를 위해 TTRL(Test-Time Reinforcement Learning)로 에이전트의 전략 가중치를 evaluator 선호 쪽으로 가중 업데이트한 뒤, ΓN의 고유값(선형화 구간)을 통해 전파 레짐을 판정하는 수학적 모델과 실험 프로토콜을 결합했습니다.

- **Empirical Impact**: 3에이전트 실험(DeepSeek-chat, 구조/균형/근거편향 evaluator 프롬프트)에서 편향은 에이전트 간에 일관되게 전파되며, 평균 contagion 계수는 γ∈[0.157, 0.352] 범위로 관측됐습니다. 다만 homogeneous-model(동일 모델 계열)에서는 연쇄 전파가 약해 suppression regime에 놓이는 반면, cross-model 설정에서는 과거 MM-EPC 수준의 더 큰 γ로 cascade 가능성이 커진다는 ‘contagion spectrum’ 관점이 제시됩니다; 또한 evaluator 위원회 크기를 k=1→3으로 늘리면 효과적 contagion이 72.4% 감소해 실용적 완화책으로 이어집니다.



### Optimal Order of Multi-Agent and General Many-Body Systems (https://arxiv.org/abs/2606.20485)
Comments:
          Key Words: Many body systems, multi agent crowd interactions, feedback loops, agent power, response function, utility function, risk appetite, order, optimal order, fragility, mobility, synchronization, useful energy, entropy, concentration, correlation, task dependency, receiver dependency, collective intelligence, AI model scaling law

- **Prior Approaches**: 기존의 다중 에이전트 연구는 에이전트 행동과 집단 관측 사이의 feedback loop를 다루더라도, 집단 수준의 거시 지표를 에이전트 단위 요소로 일관되게 설명하기 어렵다는 한계가 있었다. 또한 에이전트가 관측에 반응하는 방식과 에이전트가 집단 결과에 미치는 영향(power)을 별도로 정량화해 연결하는 프레임워크가 부족해, 동기화·질서·취약성 같은 상충 관계를 체계적으로 예측하기 힘들었다.

- **Core Contribution**: 이 논문은 feedback loop를 갖는 multi-agent systems를 분석하기 위한 일반 프레임워크를 제안한다. 핵심은 에이전트 수준 변수 두 가지—power(집단 결과에 대한 영향력)와 response functions(관측에 대한 반응 함수)—로부터 총 power, useful power, entropy, order, fragility, mobility 같은 거시 성질이 어떻게 ‘출현(emerge)’하는지 유도하는 것이다.

- **Technical Challenges**: 가장 큰 도전은 이질적인 에이전트들이 서로 다른 영향력과 반응 규칙을 가질 때, 집단의 엔트로피·질서 같은 거시 지표가 수식적으로 어떻게 결합되는지 일반화하는 것이다. 논문은 에이전트의 power 분포와 response functions를 측정·설계 가능한 입력으로 두고, 시스템 수준 유틸리티(위험 성향 계수로 성장과 회복탄력성의 trade-off를 파라미터화)를 정의해 생산성·안정성·적응성을 균형하는 ‘최적 질서 차수’를 도출한다.

- **Empirical Impact**: 분석 결과로는 동기화가 집단 출력은 높일 수 있지만, 동시에 systemic fragility를 키우고 mobility를 낮출 수 있다는 시사점이 제시된다. 또한 order, entropy, information, useful energy 같은 개념이 작업(task)과 시스템 목표에 따라 의미가 달라지는 ‘시스템 상대적’ 개념임을 주장하며, power 분포와 response functions를 통해 집단 지능과 optimal order의 조건을 더 잘 예측·최적화할 수 있음을 강조한다.



### UltraQuant: 4-bit KV Caching for Context-Heavy Agents (https://arxiv.org/abs/2606.20474)
Comments:
          11 pages, 9 figures

- **Prior Approaches**: 기존 KV 압축은 (1) 어텐션/아키텍처를 바꿔 KV 자체를 줄이거나, (2) paged serving 등 시스템 기법으로 HBM 병목을 완화하는 두 흐름으로 나뉜다. 저비트 KV quantization도 K/V를 비대칭으로 다루는 설계(KIVI, KVQuant)가 자리 잡았고, 4-bit TurboQuant 계열은 rotation+codebook으로 작은 비트에서 품질을 끌어올리는 ‘알고리즘적 기준점’을 제공해왔다. 다만 codebook 기반 구현은 디코드 단계에서 LUT 조회·불규칙 접근 등 소프트웨어 디퀀타이즈 비용이 커져, 장문 에이전트와 높은 동시성에서 end-to-end 서빙 효율로 바로 이어지지 못한다.

- **Core Contribution**: 이 논문은 컨텍스트가 큰 에이전트의 multi-round 워크로드를 대상으로, KV 품질·캐시 잔류(residency)·서빙 처리량을 함께 측정해야 한다는 평가 프레임을 제시한다. 또한 TurboQuant의 4-bit 품질 앵커를 바탕으로, 4-bit 경로를 실제 배포 가능한 형태로 만들기 위해 asymmetric K/V, Walsh-Hadamard rotation, QJL 제거, block-scale 변형 같은 구현 선택지를 정리한다. 나아가 Ultra-TQ와 UltraQuant로 나뉘는 두 접근을 통해, codebook의 병목을 줄이면서도 4-bit KV의 장문 메모리 이점을 유지하는 배포 지향(low-bit KV caching deployment) 해법을 제안한다.

- **Technical Challenges**: 핵심 난제는 ‘압축률’이 아니라 디코드-어텐션 경로에서 생기는 지연과 병목이다: codebook 조회·소프트웨어 디퀀타이즈는 컨텍스트가 길어질수록 임계 경로를 압박한다. 논문은 AMD Instinct의 CDNA4 scaled-MFMA 경로에 맞춰 커널을 재구성했고, TurboQuant 방식(Ultra-TQ)은 레이아웃/lookup/MFMA 스케줄링 최적화로 커널 간극을 줄였다. UltraQuant는 codebook을 FP4 micro-tensor(+UE8M0 scales)로 근사해 디퀀타이즈를 행렬 코어 명령 안으로 접어 넣고, FP8 쿼리와 동일 경로에서 QK·V 연산이 돌도록 설계해 소프트웨어 가중치를 제거했다.

- **Empirical Impact**: 실험은 vLLM의 multi-turn 에이전트 벤치마크(ShareGPT 기반, 32 동시 세션)로 진행했으며 TTFT(첫 토큰까지 시간)와 TPOT(출력 토큰당 시간)를 함께 본다. 장문·다중 라운드 에이전트에서 UltraQuant는 캐시 압박이 심해지는 late rounds에서 P50 TTFT를 3.47x(전체 라운드 평균 2.3x) 낮추고, FP8 KV baseline 대비 출력 처리량을 1.63x 끌어올렸다. 품질은 AIME25 등 벤치마크에 따라 하락이 관찰되지만, 성능은 컨텍스트 길이가 길어질수록 증가하며 4-bit KV가 FP8와 비슷한 처리량을 유지하면서 HBM 사용량은 절반으로 가져갈 수 있음을 보여준다.



### Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems (https://arxiv.org/abs/2606.20470)
- **Prior Approaches**: 기존 방어는 detect-and-block으로, 악성으로 탐지되면 차단·거절 문구나 필터링 결과를 예측 가능하게 반환한다. 이 구조는 개별 공격에는 효과적이지만, 자동화된 jailbreak/프롬프트 인젝션 공격에서 “거절 피드백”이 검색에 유용한 신호가 되어 반복 쿼리일수록 성공률이 상승할 수 있다. 특히 공격 프레임워크가 automated judge로 후보를 점수화·선별하며 프롬프트를 다듬는 경우, 방어 응답 자체가 공격 지능의 일부가 된다.

- **Core Contribution**: 이 논문은 모델-방어-공격자가 포함된 확률적( probabilistic ) 모델로 attack loop를 정식화하고, detect-and-block이 쿼리 예산이 커질수록 attacker success rate(ASR)이 1에 수렴할 수 있음을 보인다. 그 다음으로 detect-and-misdirect를 제안하며, 탐지된 악성 상호작용에 대해 “동작하지 않는 듯한” 그러나 그럴듯한 응답으로 공격의 평가(judge) 품질을 훼손해 ASR을 상한선에서 제한한다. 핵심은 차단 정확도를 더 올리는 대신, 공격자가 받는 피드백이 신뢰할 수 없게 만드는 것이다.

- **Technical Challenges**: 문제는 (1) 방어가 거절/차단 대신 미세하게 속이려 해도 정책 위반은 피해야 하고, (2) 자동화된 judge가 오작동하도록 만드는 misdirection-induced false-positive(MI-FP)를 설계·분석해야 한다는 점이다. 논문은 misdirection이 attacker judge의 positive predictive value(PPV)를 낮추며, 그 결과로 ASR이 비단일( non-degenerate ) 탐색에서도 1로 수렴하지 못하는 조건을 수식으로 도출한다. 또한 이를 구현한 CMPE(Contextual Misdirection via Progressive Engagement)는 예측 가능한 refusal 텍스트를 대체하면서도 안전한 대화 맥락과 후속 질문으로 공격 평가 신호를 교란한다.

- **Empirical Impact**: CMPE는 jailbreak 벤치마크에서 ASR 상한을 최대 2자리(orders of magnitude)까지 낮추는 것으로 추정되며, 특히 end-to-end PAIR와 GPTFuzz 공격 실행에서는 verified attack success가 거의 제거되는 수준으로 보고된다. 즉, 공격 루프가 automated judge 피드백에 의존할수록 미스디렉션이 반복 탐색의 효율을 크게 붕괴시킨다는 실증을 제공한다. 결과는 에이전트형 보안에서 “악성 출력 차단”뿐 아니라 “공격자의 평가 피드백 품질 저하”가 중요한 방어 축이 될 수 있음을 시사한다.



### Repurposing a Speech Classifier for Guided Diffusion-Based Speech Generation (https://arxiv.org/abs/2606.20457)
Comments:
          Accepted for publication in the Proceedings of Interspeech 2026

- **Prior Approaches**: 확산( diffusion ) 기반 생성에서 조건부 생성을 위해 흔히 Classifier Guidance(CG)를 쓰지만, 이 방식은 별도 diffusion 모델과 noise-conditioned classifier를 따로 학습해야 하고, 샘플링 매 스텝마다 두 모델을 함께 평가해 비용이 커진다. 또 JEM처럼 분류기를 에너지 기반 관점으로 확장하면 점수(score)를 유도할 수 있으나, 정규화 항 때문에 학습 불안정 문제가 보고되어 그대로 학습에 쓰기 어렵다.

- **Core Contribution**: 이 논문은 기존에 일반적으로 학습된 noise-conditioned speech classifier를 생성 백본으로 재활용해, 단일 백본(single-backbone)로 조건부 확산 생성이 가능하다는 점을 보여준다. 핵심 아이디어는 분류기 파라미터를 고정(frozen)한 채, log-Mel 공간에서 DSM(Denoising Score Matching)으로 동작하는 경량 Score Subnet을 분류기의 중간 표현에 얹어 학습하는 것이다.

- **Technical Challenges**: 가장 큰 난관은 분류기에서 얻는 신호가 ‘생성에 필요한 점수’로 얼마나 유용한지, 그리고 이를 안정적으로 학습해 역확산 점수 예측으로 연결할지였다. 연구진은 JEM 관점에서 얻을 수 있는 score-relevant 중간 신호만 추출하도록 설계하고, 분류기를 고정한 뒤 intermediate feature tap과 JEM 스타일 gradient tap을 cross-attention으로 융합해 Score Subnet만 DSM으로 학습한다.

- **Empirical Impact**: SC09에서 Score Subnet은 표준 U-Net diffusion과 FID/FAD 및 ResNeXt 기반 지표에서 대등하거나 약간 개선을 보이면서, trainable 파라미터와 GMACs를 크게 줄인다. 특히 low-data(예: 일부 클래스만) 및 zero-shot에 가까운 guidance 설정에서도 guidance strength를 키우면 품질이 개선되고, classifier-guided U-Net 대비 일관되게 더 좋은 성능을 보이며 discriminative 모델을 conditional generation으로 잇는 가교 역할을 확인했다.



### Multi-View Decompilation for LLM-Based Malware Classification (https://arxiv.org/abs/2606.20436)
- **Prior Approaches**: 기존에는 바이너리에서 복호화된(pseudo-C) 코드를 단일 디컴파일러 출력에만 의존해 LLM에 ‘악성/양성’ 분류를 맡기는 파이프라인이 주로 사용됐다. 하지만 디컴파일은 제어 흐름 복원, 타입/변수 추론, 표현 단순화 같은 과정에서 정보가 유실되고 휴리스틱 선택이 달라 공구(도구)마다 관측되는 단서가 달라진다는 점이 문제로 지적됐다.

- **Core Contribution**: 이 논문은 한 바이너리를 Ghidra와 RetDec 두 디컴파일러로 각각 처리해 얻은 pseudo-C를 함께 제공하는 multi-decompiler prompting이 훈련 없이도 악성 분류 성능을 개선할 수 있음을 제안한다. 또한 이를 체계적으로 평가하기 위해, 양성 유틸리티와 다양한 악성 행위를 포괄하는 균형 벤치마크를 디컴파일러 출력이 매칭되도록 구축했다.

- **Technical Challenges**: 핵심 난제는 디컴파일 결과가 도구별 아티팩트를 포함해 LLM을 오도할 수 있다는 점인데, 이를 ‘서로 보완적인 증거’로 활용해야 한다. 저자들은 두 디컴파일러 뷰를 한 프롬프트에 동시에 넣고(또는 불일치에만 합의 규칙을 적용) LLM이 두 관점에서 악성 단서를 종합적으로 판단하도록 설계했으며, 또한 디컴파일러 간 오류가 부분적으로 다르다는 agreement 분석으로 보완성 근거를 제시했다.

- **Empirical Impact**: 실험에서 여러 모델 계열에 대해 multi-view가 전반적으로 malicious-class recall과 F1을 끌어올렸고, 특히 작은 모델에서 개선 폭이 크게 나타났다(대부분 설정에서 단일 디컴파일러 대비 F1 향상). 또한 단일 뷰 예측이 어긋나는 사례에서 추가 컨텍스트가 정답 결정을 돕는 경향이 관찰돼, 실제 트리아지에서 놓치는(false negative) 비용을 줄이는 방향의 효과가 확인됐다.



### LLM agent safety, multi-turn red-teaming, jailbreak benchmarks, adversarial robustness, safety-critical systems (https://arxiv.org/abs/2606.20408)
- **Prior Approaches**: 기존 red-teaming 벤치마크는 대부분 단일 턴 jailbreak(한 번의 프롬프트로 한 번의 응답) 중심이며, 위해 여부도 LLM judge가 판정하는 경우가 많습니다. 또 어떤 연구는 멀티턴을 다루더라도 단일 모델을 대상으로 하거나 위해를 물리적 결과와 직접 연결하지 않아, 실제 운영 중 “지속된 적응 공격”의 강도를 충분히 측정하기 어렵습니다. 고위험 환경은 팀 기반 권한 계층과 절차적 교차검증이 핵심인데, 이를 무시하면 팀 수준의 안전 붕괴 경로를 평가하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 논문은 LLM 에이전트를 안전임계 운영자(operator)로 놓고, 다턴(adaptive multi-turn) 레드팀을 재현하는 벤치마크 NRT-Bench를 제시합니다. 시뮬레이션은 핵발전소 제어실을 모델링하고, 5역할 운영자 팀이 6개의 critical safety functions(CSF)를 유지하도록 운영하는 구조로 “물리적 위해”를 환경 상태 전이로 정의합니다. 피해(harm)는 LLM이 텍스트를 판정해 내리기보다 CSF가 intact→degraded→lost로 전이될 때의 loss 여부로 객관화되어, 특정 턴에서 어떤 메시지가 원인이었는지까지 기록·추적할 수 있습니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 공격자가 이전 턴의 부분적 결과를 관측하고 그에 맞춰 다음 턴을 설계할 수 있어야 하며, (2) 단일 프롬프트 성공이 아니라 팀이 안전한 궤적을 잃는지를 상태 기반으로 판정해야 한다는 점입니다. 이를 위해 외부 공격자는 4개 ingress 채널을 통해서만 메시지를 주입하고, 매 턴 상황 요약(black-box 또는 white-box)을 되돌려 받아 적응 공격이 가능하도록 했습니다. 또한 CSF loss가 발생하는 즉시 세션을 종료하고, 모든 이벤트를 append-only JSONL 로그와 tamper-evident manifest로 남겨 재현성과 턴 단위 원인 귀속을 보장합니다.

- **Empirical Impact**: 4개 frontier 운영자 모델(각 모델은 팀 전체 역할에 동일 모델을 사용)을 고정된 paired-replay 프로토콜로 평가했을 때, 적응 멀티턴 공격이 CSF loss로 이어지는 비율이 모델 전반에서 8.7%~12.1%로 관측되었습니다. 흥미롭게도 집계 지표는 비슷하지만 실패가 거의 겹치지 않아, 모델별 취약점이 중첩된다기보다 “거의 분리(disjoint)”된 형태로 나타났습니다. 더 나아가 방어(guardrail 스택, safety-advisor 권한 등)의 효과는 강하게 모델 의존적이어서, 같은 방어가 어떤 모델에겐 완화(ASR 감소)하지만 다른 모델에겐 오히려 ASR을 높일 수 있음을 보여주며 재현 가능한 평가 도구를 공개합니다.



### DataMagic: Transforming Tabular Data into Data Insight Video (https://arxiv.org/abs/2606.20388)
Comments:
          5 pages, 3 figures, accepted at VLDB 2026

- **Prior Approaches**: 기존 BI 대시보드와 같은 정적 시각화 도구는 내러티브 논리와 애니메이션 타이밍을 제공하지 못해 ‘시간에 따른 이야기’ 전달이 약합니다. Data Playwright 같은 저자(Authoring) 도구는 주로 미리 준비한 차트에 애니메이션을 붙이는 데 머물러 원천(raw) 데이터에서 인사이트를 뽑고 다중 씬을 오케스트레이션하기 어렵습니다. Sora 같은 픽셀 레벨 비디오 생성 모델은 블랙박스 특성 때문에 수치적 환각이 생기기 쉽고, 시각 요소를 실제 데이터 레코드로 추적(provenance)하기도 어렵습니다.

- **Core Contribution**: DataMagic은 원천 tabular data와 자연어 질의로부터 ‘내러티브 데이터 인사이트 비디오’를 end-to-end로 생성하는 인터랙티브 시스템을 제안합니다. 핵심은 DVSpec(Data Video Specification)이라는 선언적 사양으로, 시각·애니메이션 요소가 데이터 필드에 의미적으로 묶이도록 해 데이터 fidelity와 provenance를 보장하는 것입니다. 또한 Generate-then-Orchestrate 멀티에이전트 구조로 씬을 후보로 먼저 만들고 글로벌 오케스트레이션으로 내러티브 일관성을 최적화합니다.

- **Technical Challenges**: 첫째, 이기종 구성요소(차트/서사/애니메이션)를 시간 관계까지 포함해 정밀하게 표현하면서도 데이터에 대한 추적 가능성을 유지해야 했습니다. DataMagic은 데이터 기반 semantic reference로 시각 요소를 하드코딩이 아닌 데이터 속성 값에 연결하고, narration-index 기반 트리거로 TTS 오디오 길이에 맞춰 애니메이션 동기화를 자동 정렬합니다. 둘째, 설계 공간이 조합 폭발적으로 커지는 문제를 해결하기 위해 병렬로 후보 씬을 생성한 뒤, Story Planner- Narration Director- Animation Coordinator가 전역 관점에서 순서·선택·전환 논리를 맞추는 2단계 구조를 적용했습니다.

- **Empirical Impact**: DAComp-DA와 T2R-bench에서 109개 샘플을 평가한 결과, DataMagic은 평균 품질 점수를 3.89로 끌어올리며 실행 성공률도 95% 이상으로 개선했습니다. 특히 Animation Effectiveness와 Narrative Quality에서 큰 폭의 향상이 관찰되어, 기존 모델들이 겪던 오디오-비디오 temporal misalignment과 내러티브 구조 부재를 효과적으로 완화했음을 보여줍니다. 더 나아가 DVSpec의 provenance 덕분에 장면 단위 데이터 Q&A 및 Q&A 기반으로 새 씬을 추가하는 탐색-확장 루프까지 구현되어, ‘한 번 보고 끝나는 영상’에서 ‘질문하고 수정하는 인터랙티브 인터페이스’로 확장되는 의미가 있습니다.



### CRAX: Fast Safe Reinforcement Learning Benchmarking (https://arxiv.org/abs/2606.20376)
- **Prior Approaches**: 기존 SafeRL(안전 강화학습) 연구는 safety gym·safety gymnasium 같은 벤치마크가 비교의 표준을 만들었지만, 대부분 CPU 기반 시뮬레이션 의존으로 실험 속도와 확장성이 낮았습니다. 그 결과 수많은 하이퍼파라미터 튜닝, 반복 실행, 환경/난이도 일반화 테스트 같은 반복 작업이 연구 병목이 됩니다. 또한 안전을 reward shaping으로 유도하는 방식은 성능-안전 균형을 설계자가 떠안게 만드는 한계가 있었습니다.

- **Core Contribution**: CRAX는 Constrained RL Accelerated with JAX라는 이름처럼, MuJoCo의 JAX 가속 백엔드(MuJoCo XLA, MJX)를 활용해 안전 제약이 포함된 고충실도 3D physics 기반 SafeRL 벤치마크를 제공합니다. 각 태스크는 reward와 cost(비용) 신호를 함께 두어, 기대 누적 할인 비용 제약을 만족하면서도 성능을 최대화하는 trade-off를 직접 실험하게 합니다. 여섯 개 환경 suite와 난이도 3단계, 그리고 에이전트별 형태에 맞춘 구성으로 안전-성능 스케일링을 체계적으로 비교할 수 있습니다.

- **Technical Challenges**: 핵심 과제는 “고충실도 3D 물리 + 명시적 안전 제약”을 유지하면서도 수집·학습 루프를 GPU로 충분히 빠르게 돌리는 것이었습니다. CRAX는 JAX에서 벡터화된 연산과 하드웨어 가속을 적용해 병렬 시뮬레이션 처리량을 크게 끌어올리고, 태스크별 reward/cost 설계를 통해 다양한 제약 유형을 일관된 실험 틀로 노출합니다. 또한 PPO 계열을 포함한 대표 Safe RL 방법들을 JAX로 재구현해 난이도·환경 전반에서 성능/안전 균형을 공정하게 비교하도록 했습니다.

- **Empirical Impact**: 실험에서는 6개 대표 SafeRL 방법이 모든 태스크에서 단일 최강 해법이 되지 않음을 보이며, 제약 유형에 따른 강점/약점과 성능-안전 트레이드오프를 확인했습니다. 동시에 최상 난이도에서 curriculum learning(난이도 단계적 학습)과 safety transfer(안전 지식 전이)가 일부 환경에서는 학습 효율과 성능을 개선하지만, 환경에 따라 효과가 달라짐도 관찰했습니다. 처리량 측면에서 CRAX는 Safety-Gymnasium 대비 수십~수백 배 이상의 SPS 확장성을 보여, 대규모 벤치마킹이 수 주 내 가능해지는 반면 기존 CPU 기반은 약 1년 수준으로 늘어날 수 있음을 시사합니다.



### AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning (https://arxiv.org/abs/2606.20373)
- **Prior Approaches**: 기존 컴파일 성능 튜닝은 -O3 같은 고정 파이프라인을 두고 PGO나 AutoFDO, CSSPGO처럼 프로파일에 기반해 휴리스틱을 조정하는 방식이 많았다. 하지만 성능 목표(런타임)는 마이크로아키텍처 상호작용과 잡음이 커서, 프로파일 근거가 약하면 보수적으로 놓치거나 회귀가 생기기 쉽다. 또한 phase ordering과 파라미터 탐색 공간이 커서 OpenTuner 같은 black-box 탐색은 예산이 작을 때 유효한 파이프라인을 찾지 못하는 경우가 잦다.

- **Core Contribution**: AutoPass는 LLM이 컴파일러를 black box처럼 다루지 않고, 컴파일 내부 신호(optimization remarks)와 중간표현(LLVM IR 스냅샷)을 근거로 최적화 결정을 하도록 설계한 멀티에이전트 프레임워크다. 런타임 측정 피드백을 함께 써서, 한 번의 제안이 아니라 반복적으로 pipeline을 수정·검증하는 closed-loop를 만든다. 특히 training-free, inference-only로 동작해 새로운 벤치마크/플랫폼에 재학습 없이 적용 가능하다고 제시한다.

- **Technical Challenges**: 핵심 난제는 (1) 런타임 성능이 잡음이 많고 (2) 최적화 효과가 서로 상호작용해 코드만 보고 판단하기 어렵다는 점, 그리고 (3) LLM 생성 파이프라인이 문법/토큰 오류를 낼 수 있다는 점이다. AutoPass는 Score Agent가 최적화 중요 함수를 먼저 추려 LLM context를 절약하고, Analysis Agent가 semantic hints와 -Rpass류 컴파일 remarks를 JSON으로 구조화해 Reasoning Agent의 근거를 명시한다. 이후 Reasoning Agent가 pass 재정렬·파라미터 조정을 제안하면 repair/validation 단계에서 문법·유효 패스 집합·범위를 검사해 실패 시 반복 탐색을 이어가며, Evaluation Agent가 컴파일 검증과 런타임 비교로 회귀를 차단한다.

- **Empirical Impact**: LLVM(17.0.6) 위에 구현해 x86-64 서버와 ARM64 임베디드에서 평가했으며, AutoPass가 -O3 대비 기하평균 1.043x(x86-64)와 1.117x(ARM64) 개선을 보였다. 또한 expert-tuned 휴리스틱과 classical autotuning(OpenTuner, PGO 계열)보다 성능과 안정성이 전반적으로 낫다고 보고한다. 특히 single-shot 추론보다 반복 피드백이 회귀를 줄이고(서버 x86-64에서 regressions 감소) 3번 내 수렴으로 효율을 확보하는 점이 임팩트로 강조된다.



### Robust $Q$-learning for mean-field control under Wasserstein uncertainty in common nois (https://arxiv.org/abs/2606.20356)
- **Prior Approaches**: 기존 mean-field control(MFC, McKean–Vlasov control)은 평균장 상호작용을 통해 고차원 다개체 문제를 대표 에이전트 최적화로 환원하지만, common noise가 들어오면 Bellman 또는 forward-backward 조건이 확률측도 공간 위의 무한차원 방정식이 된다. 또한 common noise의 실제 분포를 알지 못할 때(비관측/추정 오차), 강건성(min–max, risk) 없이 학습하면 기준이 되는 stochastic control 식이 틀어져 해가 불안정해질 수 있다. 강화학습 관점에서는 robust QQ-learning 및 일반 MFC의 연구가 있으나, Wasserstein 기반 common-noise 불확실성을 다루면서 탭불러(tabular) robust QQ-learning과 유한시간 비동기 수렴/오차분석을 함께 제공한 선행 연구는 거의 없었다.

- **Core Contribution**: 본 논문은 discrete-time MFC에서 common noise의 Wasserstein 불확실성까지 포함한 robust Q-learning(robust QQ-learning) 알고리즘을 제안한다. 핵심은 공통잡음(common noise) 분포에 대한 worst-case 기대값을 직접 다루지 않고, quantization-and-projection으로 lifted state-action 공간을 유한 차원으로 만들며, Wasserstein dual reformulation으로 worst-case 평가를 1차원 최적화로 바꾸는 것이다. 그 결과, robust QQ-function의 고정점 학습 타깃을 실제 샘플 기반으로 구성하면서도 강건성을 유지한다.

- **Technical Challenges**: 가장 큰 기술적 난점은 lifted state-action 구조 때문에 원래 상태/행동이 유한이어도 확률측도 공간이 무한 크기가 되어 tabular 처리가 불가능하다는 점이다. 이를 quantization-and-projection으로 유한 표현으로 바꾸어 discretization error를 분리해 분석한다. 두 번째 난점은 common noise 법을 모르는 상태에서 Bellman 타깃의 worst-case expectation을 어떻게 샘플로 안정적으로 근사하느냐인데, Wasserstein duality로 min–max 계산을 기준 분포 아래 단일 기대값과 dual optimizer로 환원하고, 비동기 학습에서도 독립 샘플을 활용할 수 있게 만들었다.

- **Empirical Impact**: 이론적으로는 synchronous/asychronous 두 학습 변형에 대해 수렴(유한 discretization 오차 내)과 유한시간 반복수(목표 정확도·신뢰수준에 필요한 iteration bound)를 함께 제시한다. 수치실험에서는 systemic risk와 epidemic 모델에서 비동기 구현이 Bellman iteration에 준하는 수렴 거동을 보이며, common-noise misspecification 상황에서 robust–performance tradeoff가 관측된다. 즉, 불확실한 공통잡음 하에서 MFC를 강화학습으로 학습할 때의 강건성과 수렴 보장을 함께 연결한 점에서 의미가 크다.



### Boundary Embedding Shaping with Adaptive Contrastive Learning for Graph Structural Disentanglemen (https://arxiv.org/abs/2606.20283)
Comments:
          Accepted at ICML 2026

- **Prior Approaches**: 기존 GNN은 이웃 정보를 집계해 노드 임베딩을 만들지만, 그래프 구조적 entanglement로 인해 의미와 무관한 이웃에서 온 spurious correlations이 임베딩에 섞입니다. 특히 결정 경계 근처 노드는 구조적 잡음의 증폭으로 경계가 흐려지고 예측이 불안정해지는 문제가 두드러집니다. 따라서 많은 robust GNN 접근은 전체 노드를 동일하게 다루어, 경계 영역에서 생기는 취약성에 초점을 맞추지 못했습니다.

- **Core Contribution**: 이 논문은 결정 경계에서 특히 심각한 boundary-region entanglement를 병목으로 규정하고, 이를 줄이기 위한 Boundary Embedding Shaping(BES)를 제안합니다. BES는 기존 GNN 인코더에 “플러그인” 형태로 끼워 넣을 수 있는 적응형 대조학습 모듈로, 경계 노드의 structural noise를 선택적으로 억제하면서도 모델 파라미터 변화는 최소화합니다. 목표는 invariant한 요인은 살리고, 결정 경계를 흐리게 만드는 variant 정보를 분리해 분류 성능을 끌어올리는 것입니다.

- **Technical Challenges**: 핵심 난제는 경계에서만 발생하는 구조적 잡음을 학습 신호로 안정적으로 만들면서, 불필요한 파라미터 업데이트로 전체 성능을 흔들지 않는 것입니다. BES는 (1) 임베딩 공간에서 경계 영역을 먼저 찾고, (2) 구조적 변동이 큰 boundary nodes만 선별한 뒤, (3) hard example mining에 기반한 contrastive objective로 양(같은 클래스)에는 끌어당기고 음(다른 클래스)에는 밀어내는 방식으로 임베딩을 정교하게 다듬습니다. 또한 쌍대 유사도 계산의 O(N^2) 부담을 피하기 위해, 그라비티-기반의 class centroid 및 Gaussian center 근사로 O(N) 수준의 효율적인 학습을 구현합니다.

- **Empirical Impact**: 실험에서 BES는 node classification과 link prediction 모두에서 일관되게 기존 유력 방법을 능가하며, 결정 경계 구분력이 개선되는 양상이 관찰됩니다. 특히 GCN 기준 node classification 정확도가 평균 3.3% 향상되고 WikiCS에서는 최대 5.0%까지 개선되며, 링크 예측에서도 더 높은 정확도를 달성합니다. 이는 이론이 지목한 boundary margin/식별성 관점이 실제 성능 향상으로 연결될 수 있음을 경험적으로 뒷받침합니다.



### ELVA: Exploring Ranking-Driven Universal Multimodal Retrieva (https://arxiv.org/abs/2606.20280)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 UMR(Universal Multimodal Retrieval)은 텍스트/이미지 등 다양한 검색을 하나로 묶되, MLLM을 Retrieval로 전환할 때는 주로 contrastive learning을 사용해 positive/negative 임베딩을 분리했다. 하지만 이 방식은 negative를 모두 같은 것으로 취급해, 쿼리 안의 엔티티-행동 같은 여러 grain 정보를 충분히 세밀하게 학습하지 못한다(이를 grain blindness로 지칭). 특히 다중 grain 쿼리에서는 모델이 핵심 의미의 일부만 잡고 나머지를 놓치는 경향이 관측된다.

- **Core Contribution**: 이 논문은 grain blindness를 줄이기 위해 negative를 “유사도에 따라 다르게” 다루는 ranking-driven 학습을 제안한다. 그 결과로 ELVA(ExpLoring Ranking-Driven UniVersal Multimodal RetrievAl)라는 규칙 기반 RL 프레임워크를 설계해, reward를 통해 positive의 상위 랭킹과 negative 간 위계(순서)를 동시에 학습시킨다. 또한 ranking 라벨 없이도 RLVR(Reinforcement Learning with Verifiable Rewards)과 GRPO 기반 탐색으로 랭킹 능력을 유도한다.

- **Technical Challenges**: 핵심 기술 난점은 (1) 대조학습의 이진 분류식 학습이 왜 grain-level 정보를 붕괴시키는지(gradient starvation 및 representational collapse)와 (2) retrieval에서 “정답 랭킹 라벨”을 구하기 어렵다는 점이다. ELVA는 생성형 임베딩 추출을 통해 정책에 변동성을 주고(RET 토큰의 hidden state를 임베딩으로 사용), verifiable reward 두 가지(연속형 ranking reward, similarity gap을 강제하는 margin reward)로 보상을 구성해 라벨 없이도 학습을 안정화한다. 더불어 RL용 balanced negative sampling으로 너무 어려운 negative를 걸러 reward 분포가 과도하게 좁아지는 문제를 완화한다.

- **Empirical Impact**: ELVA는 표준 UMR 벤치마크(M-BEIR)에서 다양한 모달 조합에 대해 SOTA 성능을 일관되게 보였다. 특히 multi-grain 전용 벤치마크 MRBench에서는 13.1%의 큰 개선을 기록해 grain blindness 완화 효과를 정량적으로 입증했다. 보상 함수 및 negative sampling 전략에 대한 ablation에서도 ranking reward와 margin reward, 그리고 balanced sampling이 성능에 직접 기여함이 확인되며, 다중 grain 정보 보존 능력이 실험적으로 뒷받침된다.



### Editorial Alignment: A Participatory Approach to Engaging Editorial Expertise in LLM-mediated Knowledge Dissemination (https://arxiv.org/abs/2606.20258)
Comments:
          14 pages

- **Prior Approaches**: 기존 연구와 실무는 AI alignment을 주로 값(value)이나 원칙을 ‘정의’하고, 이를 선호도 데이터로 보정하는 기술 최적화 문제로 다뤄왔다. 하지만 LLM의 행동은 사후 검증만으로 통제하기 어렵고, 특히 사전학습된 모델은 상용 개발사의 가치·전달 전략에 이미 정렬(aligned)돼 있어 공공 지식기관의 편집 권위를 잠식할 수 있다. 한편 참여형 접근(Participatory AI)은 통합 의사결정 참여나 워크플로 협업에는 강점을 보였지만, LLM 인터페이스가 편집 기준을 ‘지속적으로’ 반영하도록 만드는 alignment 목표 자체의 공동 설계까지는 덜 다뤘다.

- **Core Contribution**: 이 논문은 editor participation을 통해 LLM 인터페이스를 기관의 편집 기준에 다시 맞추는 실천으로 ‘editorial alignment(편집 정렬)’를 제안한다. 핵심은 AI alignment을 반복적인 기술 최적화가 아니라, 해당 기관의 편집가가 체화한 가치와 표준을 설계 목표로 번역하는 디자인 과정으로 재정의하는 것이다. 또한 편집 표준을 기술 구현의 제약·조건으로 이어주는 ‘디자인 아티팩트(design artefact)’로 구체화해, 편집가가 LLM 매개 지식 전달의 거버넌스에 실질적 권한을 갖도록 한다.

- **Technical Challenges**: 기여를 실현하는 데의 첫 난관은 편집 표준이 문서화된 정책이 아니라 실무 속 암묵지(tacit knowledge)로 존재한다는 점이다. 논문은 이를 데이터 소싱이 아니라 ‘워크숍 기반의 재발견’으로 다루기 위해, 미래 워크숍과 실무 중심 워크숍에서 편집가가 LLM 초안 텍스트를 직접 다루며 표준을 공동 코딩하도록 설계했다. 두 번째 난관은 LLM 정렬 목표가 사용 가능한 입력/출력 제약과 학습·후처리 방식에서 실현 가능해야 한다는 점인데, 편집 표준을 라이브 문서로 유지하며 인터페이스 변화와 함께 업데이트 가능한 목표로 운영한다.

- **Empirical Impact**: 북유럽 공공 지식기관의 온라인 백과(케이스 스터디)에서 두 차례 워크숍을 통해 편집가가 편집 표준을 만들고, 이를 기반으로 LLM-enabled 백과 인터페이스를 설계·구현하는 과정을 보여준다. 결과적으로 ‘편집 권위의 보존’과 ‘지속적 참여를 위한 공간 확보’라는 목표를 동시에 달성할 수 있음을 실증적으로 논증한다. 분야 관점에서는, alignment를 가치 추출/집계가 아니라 공동 설계·운영 가능한 조직적 실천으로 옮겨 ‘공공 지식 인프라의 신뢰 가능성’을 재구성하는 관점을 제공한다.



### The Register Gap: A Meaning Intelligence Framework for Nigerian Public Discours (https://arxiv.org/abs/2606.20255)
Comments:
          Preprint. 12 pages, 2 tables. Supplementary materials: MIF Master Specification v2.0, Annotation Guidelines v1.0, and 30-item public calibration set with gold labels available from the author

- **Prior Approaches**: 기존 나이지리아 언어/담론 벤치마크(예: NaijaSenti, AfriSenti)는 감정 분류를 positive, negative, neutral의 3분류 극성 과제로 다뤘습니다. 하지만 논문은 번역 실패보다 문맥 실패가 주된 실패 모드라고 지적하며, 같은 발화도 화자·청중·상황에 따라 반대의 화용적 힘(pragmatic force)을 가질 수 있다고 봅니다. 결국 표면 감정 라벨만으로는 의도와 맥락을 분리해 평가하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 논문은 Meaning Intelligence Framework(MIF)라는 9개 차원 주석 및 평가 스키마를 제안해, 표면 sentiment과 실제 communicative intent(진정한 전달 의도)를 분리해 측정합니다. register, surface sentiment, true intent, irony, coded subtext, risk tier, annotator confidence, speaker emotion, recommended communications action을 점수화해 담론의 의미를 다차원으로 캡처합니다. 또한 재현성을 위해 프레임워크 사양과 주석 가이드라인, 30-item calibration 세트를 공개하고 오염 방지 목적의 private holdout도 유지합니다.

- **Technical Challenges**: 핵심 기술 과제는 문맥 의존적 화용 정보를 안정적으로 구조화해 모델이 읽을 수 있게 만드는 것입니다. 이를 위해 9차원 스키마를 prompt에 반영한 schema-informed prompting을 설계해, zero-shot보다 register 및 coded subtext 같은 문맥 신호를 더 잘 끌어내도록 했습니다. 실험에는 Gemini 2.5 Flash를 사용해 Standard English, Nigerian English, Nigerian Pidgin, code-mixed register 전반의 문맥 변이를 평가했습니다.

- **Empirical Impact**: 실험 결과의 핵심은 Register Gap으로, zero-shot register 분류 정확도가 33.3%에 그쳤지만 MIF 스키마를 in-context로 제공하면 73.3%로 크게 상승했습니다(약 +40점). 전체 Meaning Intelligence Score도 schema-informed prompting에서 73.2에서 78.6으로 5.4점 개선됐고, 특히 coded-subtext detection(+10점)과 strategic action recommendation(+10.3점)에서 실용적 이득이 두드러졌습니다. 나이지리아 공공 담론을 대상으로 한 평가가 단순 감정 극성에서 벗어나 ‘의미 지능’을 정량화하는 방향성을 제시했다는 점에서 의미가 큽니다.



### Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think (https://arxiv.org/abs/2606.20246)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 효율화는 크게 두 갈래로, 토큰 프루닝·캐싱 같은 training-free 가속과, MoLe-VLA·DeeR-VLA처럼 학습으로 라우팅/조기종료를 학습하는 training-adaptive 방식이 주를 이뤘다. 다만 training-free는 실시간 추론은 줄여도 비싼 downstream fine-tuning 비용을 거의 건드리지 못했고, training-adaptive는 보조 라우터·추가 학습 목표로 구조가 복잡해지는 문제가 있었다. 또한 연속 제어형 SOTA(π0, GR00T-N1.5)와 실세계 검증 범위가 제한적인 경우가 많았다.

- **Core Contribution**: 이 논문은 VLA의 연속 제어 foundation policy가 학습된 다양한 궤적에도 불구하고, 층(layer) 단위 표현에 심각한 중복( representational redundancy )이 있음을 보인다. 이를 바탕으로 CLP(CKA-guided Layer Pruning)를 제안하며, 학습 없이 Centered Kernel Alignment(CKA)로 중복 층을 찾아 fine-tuning 전에 영구적으로 transformer depth를 최대 50%까지 줄인다. 그 결과 구조는 작아지되, 다운스트림 적응은 기존 학습 목표 그대로 수행해 성능 손실 없이 효율을 확보한다.

- **Technical Challenges**: 핵심 난제는 “중복처럼 보여도 제거하면 정책이 망가질 수 있다”는 안정성 문제였다. CLP는 calibration set에 대해 한 번의 forward pass로 각 연속 층의 표현 유사도를 CKA로 측정하고, 높은 유사도 구간을 연속 블록으로 묶어(pruning set 구성) 실제로 제거할 후보를 정한다. 이후 선택된 층을 제거해 스태틱하게 압축된 모델을 만들고, 추가 모듈 없이 바로 downstream fine-tuning을 수행해 표현 공간이 재구성(manifold restoration)되도록 한다.

- **Empirical Impact**: LIBERO·RoboCasa·SimplerEnv 3개 시뮬레이션 벤치마크와 실세계 조작 10개 태스크(4개 로봇 embodiment)에서 CLP는 기준 모델 대비 30% 내외 학습/연산 이점을 보이며 성능은 동등하거나 더 높게 유지된다. 구체적으로 학습 시간은 40–50% 줄고 실시간 추론도 최대 30% 빨라졌으며, 심지어 데이터가 적은 환경에서도 10% 데이터에서 성공률이 77.7%→84.6%처럼 개선되는 샘플 효율 이점이 확인됐다. 저자들은 “필요 이상으로 깊은 VLA가 계산 효율을 갉아먹는다”는 관점을 지지하며, 확장 가능한 로봇 학습을 위한 compute-efficient 패러다임을 제시했다고 결론낸다.



### SPOT-E: Test-Time Entropy Shaping with Visual Spotlights for Frozen VLMs (https://arxiv.org/abs/2606.20244)
- **Prior Approaches**: 기존 비전-언어 모델(VLM)용 추론 시각 개입은 이미지에 마스크/오버레이를 하거나 크롭·줌으로 해상도를 재배치하는 방식으로, 모델 가중치는 고정한 채 grounding을 개선하려는 시도가 많습니다. 다만 대부분 open-loop라서, 강조한 영역이 실제 정답 결정 증거로 ‘사용’됐는지 검증 메커니즘이 부족해 오개입 시 실패가 그대로 누적됩니다. 또한 불확실성(엔트로피)을 단순히 낮추는 접근은 evidence-grounded 확신과 shortcut 붕괴를 구분하지 못한다는 한계가 드러납니다.

- **Core Contribution**: 이 논문은 answer-span prediction entropy를 ‘증거 사용도’의 모델 내부 피드백 신호로 제안하고, 엔트로피 감소가 의미하는 바가 모호하다는 점을 정리합니다. 이를 해결하기 위해 low-entropy anchors를 도입해, 기준 입력에서 이미 결정적인 토큰에 해당하는 위치의 안정성을 유지하면서 엔트로피를 낮추는 entropy-shaping objective를 설계합니다. 결과적으로 label 없이도 “증거 기반 확신은 강화하고, shortcut collapse는 억제”하는 방향의 테스트타임 적응을 가능하게 합니다.

- **Technical Challenges**: 핵심 난제는 엔트로피를 줄이되, 증거를 실제로 노출한 경우와 단지 추론을 단순화해 확신을 만드는 경우를 구분해야 한다는 점입니다. 저자들은 baseline에서 near-deterministic한 토큰 위치를 anchors로 잡고, 개입 후 anchor 위치의 엔트로피가 증가하면 패널티를 주어 ‘비파괴적인 엔트로피 감소’를 유도합니다. SPOT-E는 이를 구현하기 위해 CLIP 기반 question-conditioned visual spotlight를 LoRA로 가볍게 학습하며, per-instance Group Relative Policy Optimization (GRPO)로 spotlight 마스크를 짧게 최적화합니다.

- **Empirical Impact**: SPOT-E는 여러 frozen VLM 계열(오픈소스 백본과 토큰 로그확률이 노출되는 API 포함)에 걸쳐 evidence-intensive 벤치마크에서 일관된 성능 향상을 보이며, 특히 작은 텍스트·숫자·국소 기호 의존 과제에서 이득이 큽니다. 시각 잡음·해상도 저하·부분 가림 같은 visual corruption에서도 성능 하락을 완화해 robustness가 개선됨을 보였습니다. 또한 엔트로피 분석에서 오답은 엔트로피가 높아지고 정답은 낮게 유지되어, confident-but-unsupported 오류를 줄이며 정오 분리가 강화되는 것으로 나타났습니다.



### ScholarQuest: A Taxonomy-Guided Benchmark for Agentic Academic Paper Search in Open Literature Environments (https://arxiv.org/abs/2606.20235)
- **Prior Approaches**: 기존 학술 논문 검색은 주로 lexical/semantic 매칭으로 큰 문헌에서 빠르게 후보를 찾는 방식이 많았지만, 복합 조건 질의에서는 한 번에 고정된 ranked-list로 끝나는 한계가 있었다. 또한 LLM 기반 agentic search가 등장했지만, 평가용 벤치마크는 소수의 수작업 질의나 특정 논문에서 유도된 질의에 의존해 다양한 연구 의도와 편향을 충분히 통제하지 못했다. 답변(관련 논문 집합) 구축도 비용이 커서 확장성이 떨어졌고, 표준화된 공개 평가 환경이 부족해 재현성 있는 비교가 어려웠다.

- **Core Contribution**: 이 논문은 agentic academic paper search를 현실적인 open literature 환경에서 체계적으로 평가할 수 있게 하는 대규모 벤치마크 ScholarQuest를 제안한다. ScholarQuest는 1,000+ CS 주제와 4가지 연구 의도(방법 중심, 설정 고정, 비교 기반, 범위 제어)를 기반으로 질의 분포를 통제하고, 답변 집합을 만들기 위한 자동 파이프라인과 공용 검색 백엔드 ScholarBase를 함께 제공한다. 이를 통해 “무엇을 찾았는지”뿐 아니라 “어떻게 탐색했는지”까지 동일 조건에서 측정할 수 있게 한다.

- **Technical Challenges**: 핵심 난관은 (1) 의도별로 세밀한 제약을 반영하는 질의를 대규모로 편향 없이 생성하고, (2) 서로 다른 용어/하위 분야에 흩어진 관련 논문을 확장·정제해 큰 gold answer pool을 만드는 데 있었다. 저자들은 ACM CCS 주제에 arXiv taxonomy를 매핑하고, LLM 기반 질의 생성 후 품질 필터링·중복 제거로 1,111개 고품질 질의를 만들었다. 또한 다중 소스 retrieval과 citation expansion, 다단계 관련성 판정(LLM 심사 + 인간 감사)으로 스케일 가능한 답변 구성을 구축했으며, ScholarBase API로 재현 가능한 탐색 환경을 제공했다.

- **Empirical Impact**: 실험 결과, agentic 방법은 single-shot 검색 기준선을 전반적으로 앞섰지만 최고 성능도 Recall@100 0.314, Recall@All 0.355에 머물러 개선 여지가 큰 것으로 나타났다. 특히 scope-controlled 질의와 답변 집합이 커질수록 성능 저하가 커 제약 민감성/강건성의 병목이 드러났다. 실패 분석에서는 검색 노력이 부족해서가 아니라 off-target exploration(잘못된 문헌 영역을 확장)에서 오류가 주로 발생함을 확인해, 향후 intent 보존과 constraint-aware 추론 필요성을 구체적으로 시사한다.



### Learner-based Concept Drift Detection: Analysis and Evaluation (https://arxiv.org/abs/2606.20216)
Comments:
          2 authors, 29 pages

- **Prior Approaches**: 개념 드리프트(concept drift)는 시간에 따라 데이터 분포가 바뀌는 비정상 상황으로, 기존 예측 모델은 오래된 패턴에 고착돼 성능이 저하되기 쉽다. 선행 연구들은 드리프트를 감지하기 위해 (1) SPC처럼 모델 성능을 통제 프로세스로 보고 통계적 일탈을 찾거나, (2) 고정/동적 윈도우로 과거와 현재의 차이를 비교하거나, (3) 앙상블로 성능 변화를 누적 관찰하는 방식이 주로 사용돼 왔다. 다만 라벨이 필요한 경우가 많고(learner-based), 분포 기반 방법은 잡음에 비교적 강하지만 연산 비용과 오탐(false alarm) 위험이 커 현장 적용이 까다롭다는 한계가 있었다.

- **Core Contribution**: 이 논문은 데이터 스트림 분류 환경에서 개념 드리프트의 핵심 특성(실제/가상/혼합 드리프트, 갑작/점진/증분/재발 전이)을 체계적으로 정리하고, 드리프트 탐지기를 learner-based 관점에서 분류해 이해를 돕는다. 또한 SPC-, window-based, ensemble-based 축으로 대표적인 드리프트 검출기 15종을 포함해 알고리즘적 틀과 동작 방식을 정리한 뒤, 합성 및 실데이터 스트림에서 탐지 성능을 비교 평가한다. 목표는 드리프트 “특성”과 “탐지기 거동” 사이의 관계를 실증적으로 파악해 다양한 상황에서의 적용성을 높이는 것이다.

- **Technical Challenges**: 핵심 기술적 난제는 스트리밍 환경에서 빠르게(온라인으로) 드리프트를 찾아내야 하지만, 잡음과 우연 변동 때문에 오탐을 줄여야 한다는 점이다. 논문은 learner 기반 탐지기를 통제한계 초과(SPC), 과거-현재 윈도우 간 통계적 차이(windowing), 여러 모델/검출 신호의 조합(ensemble)으로 나누고, 각각이 민감도·지연·연산비용 트레이드오프를 어떻게 다루는지 이론과 절차로 연결해 설명한다. 특히 라벨이 필요한 문제, 급격 vs 점진 변화에 대한 탐지 민감도 차이, 그리고 고정 윈도우 크기 설정의 어려움 같은 실무 장애를 어떤 방식으로 완화하는지(예: 동적 윈도우, 경고 구간(warning level) 등) 중심으로 다룬다.

- **Empirical Impact**: 실험은 드리프트 위치가 알려진 합성 스트림(갑작/점진 변화 포함)과, 현실 조건처럼 잡음이 섞이거나 신호가 약한 실데이터 스트림을 함께 사용해 탐지기의 실제 행동을 비교한다. 이를 통해 드리프트 유형과 전이 속도에 따라 SPC·윈도우·앙상블 계열이 어떤 조건에서 더 잘 탐지하는지(탐지 시점, 드리프트/경고 구간의 유용성, 오탐 가능성 등)를 관찰한다. 결과적으로 연구자와 실무자가 “어떤 드리프트 특성일 때 어떤 계열의 드리프트 detector를 선택/튜닝할지”에 대한 실전 가이드에 가까운 이해를 제공하는 것이 이 논문의 의의다.



### FlowMaps: Modeling Long-Term Multimodal Object Dynamics with Flow Matching (https://arxiv.org/abs/2606.20209)
- **Prior Approaches**: 기존 Object Navigation(ObjNav)과 동적 객체 위치추정은 정적 장면 가정이 많거나, 동작 궤적/장소를 확률적으로 다루더라도 이산적 receptacle(수용기) 관계나 사전 설계된 prior, 비싼 VLM/LLM 추론에 의존하는 경우가 많았다. 또한 HOMER처럼 반복되는 가정 활동 기반으로 변위(displacement)를 예측하더라도 특정 환경 내부에서의 일반화에 초점이 있어, 레이아웃·배치가 다른 집으로의 전이를 충분히 검증하기 어렵다는 한계가 있었다.

- **Core Contribution**: FlowMaps는 사람의 일상 상호작용으로 인해 발생하는 객체의 미래 위치를 연속 3D 공간에서의 multimodal spatio-temporal 분포로 직접 추정하는 latent flow matching 모델이다. 특히 객체-배경 장면 컨텍스트와 쿼리 라벨, 예측 시간에 조건을 건 채 미래 bounding box 분포를 생성해, 특정 활동을 명시적으로 식별하지 않아도 데이터에서 규칙성을 학습·활용하도록 설계됐다. 더불어 이전에 보지 못한(하지만 유사한 일과/루틴 패턴을 공유하는) 새로운 환경으로의 일반화까지 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 미래 객체 위치가 여러 모드로 열려 있고(예: 안방 탁자↔세면대), (2) 장면 맥락과 시간 변화가 결합된 posterior를 직접 계산하기 어렵다는 점이다. FlowMaps는 이를 연속 공간의 조건부 flow로 근사하기 위해 VAE로 객체 기하/의미를 잠재공간으로 압축한 뒤, conditional flow matching을 latent 공간에서 학습하고 샘플링으로 다중 미래를 얻는 방식으로 해결했다. 또한 ProcTHOR로 대규모 동적 궤적(사람 루틴 기반)을 합성해 학습 데이터 병목을 완화하고, Transformer 기반 인코더+CDiT 블록으로 장면 컨텍스트를 쿼리의 시간적 흐름에 효과적으로 주입한다.

- **Empirical Impact**: ProcTHOR 시뮬레이션에서 600회 이상 ObjNav episode을 통해 FlowMaps가 최신 baselines 대비 더 좋은 minFDE와 모드 커버리지를 보이며, 포인트 정확도만이 아니라 분포의 형태(coverage/density, TV/JS)에서도 개선을 보였다. 특히 best-of-K 샘플링으로 평가했을 때도 multimodality를 제대로 보존해, 동적 변화가 큰 가정환경에서 탐색·탐지 성능이 향상됨을 확인했다. 또한 실제 로봇 플랫폼 배치까지 수행해 실세계 적용 가능성을 함께 검증함으로써, posterior 추론 관점에서 FM을 로보틱스에 활용할 수 있음을 실증했다.



### HilDA: Hierarchical Distillation with Diffusion for Advancing Self-Supervised LiDAR Pre-trainin (https://arxiv.org/abs/2606.20189)
Comments:
          Accepted to ECCV 2026. Maciej and Jesper contributed equally

- **Prior Approaches**: 기존 camera-LiDAR knowledge distillation은 Vision Foundation Model(VFM)을 black-box teacher로 두고 주로 프레임 단위 feature similarity만 맞추는 경향이 강했습니다. 이 때문에 VFM의 층별(semantic evolution) 구조와 CLS 토큰이 담는 전역(global context) 의미를 충분히 활용하지 못하고, LiDAR의 시공간(spatiotemporal) 일관성 학습도 약했습니다.

- **Core Contribution**: 본 논문은 LiDAR 백본을 위한 self-supervised pretraining 프레임워크 HilDA를 제안합니다. HilDA는 계층적(hierarchical) distillation로 “semantic what(무엇)”을 VFM에서 가져오고, temporal occupancy diffusion으로 “geometric where(어디)”와 미래의 시공간 일관성을 함께 학습하도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 모달리티에서 VFM의 층별 의미 계층을 점-픽셀 대응 위에서 안정적으로 전이하고, (2) 의미 정렬만으로는 부족한 동적 장면의 미래 점유를 생성적으로 학습하는 것입니다. HilDA는 기하학적 calibration 기반의 dense-to-sparse 멀티레이어 distillation과 CLS 토큰 정렬을 동시에 수행하고, 미래 BEV occupancy를 conditional diffusion(denoising 기반)으로 예측하는 auxiliary objective를 추가해 저라벨/무라벨 상황에서도 예측형 기하·운동 표현을 주입합니다.

- **Empirical Impact**: HilDA는 cross-modal distillation 벤치마크에서 state-of-the-art를 달성했으며, 특히 레이블이 적은 데이터 스케줄(1%–10%)에서 기존 distillation 대비 큰 폭의 향상을 보였습니다. 또한 3D object detection, scene flow, semantic occupancy prediction 전반에서 이전 기법을 능가하고, nuScenes-C에서 mCE 낮고 mRR 높게 나타나는 등 강인성도 개선되는 것으로 보고됩니다.



### Evaluating and Enhancing Negation Comprehension in Remote Sensing MLLMs (https://arxiv.org/abs/2606.20177)
Comments:
          ECCV 2026 Accepted

- **Prior Approaches**: 기존 연구는 negation을 평가하기 위해 NegVQA, GaslightingBench 같은 일반 도메인 비전-언어 벤치마크를 제시했지만, 원격탐사(RS)에서 요구되는 세부 속성·상태 수준의 부정과 소형 표적 문제는 충분히 다루지 못했습니다. 또한 CLIP/embedding 중심의 negation 학습·테스트는 있어도, MLLM에 그대로 확장하기 어렵고 대규모 라벨 수집 없이 개선하는 방법도 상대적으로 비어 있었습니다.

- **Core Contribution**: 이 논문은 RS MLLM의 negation 이해를 지역-수준부터 장면-수준까지 포괄적으로 점검하는 최초 벤치마크 RS-Neg(총 22,464샘플)를 제안합니다. LLM이 자연스럽고 다양한 negation 쿼리를 만들고, Dynamic visual focus(DyFo)로 부정 개념의 시각적 부재를 검증한 뒤 과제별 VQA/MCQ/visual grounding/scene classification 데이터로 재구성합니다. 더불어 테스트 시점 학습 NeFo를 제안해, 라벨 추가 없이 약 5% 미라벨 테스트 샘플로 negation 이해를 크게 끌어올립니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) RS 이미지에서 ‘없음/부정’ 개념을 실제로 검증해야 하는데 소형 표적 때문에 기존 MLLM 검증이 잘 실패한다는 점, (2) 테스트 시점 적응 시 잘못된 예측이 자기강화되어 성능이 더 악화될 수 있다는 점입니다. 논문은 DyFo 기반 탐색으로 negation에 해당하는 시각적 부재를 필터링하고, NeFo에서는 negation의 논리적 역할(진리값 반전)을 목적함수에 직접 반영해 negated 쿼리와 그 부정 마스킹 변형의 출력 불일치를 키우는 방식으로 학습 신호를 설계합니다. 동시에 knowledge retaining 정규화와 엔트로피 최소화를 함께 써 catastrophic forgetting을 억제합니다.

- **Empirical Impact**: 실험 결과, 대부분의 RS 특화 및 일반 MLLM은 부정 조건에서 환각과 함께 성능이 크게 하락하며, 장면-수준에서 평균 VQA 8.6%, classification 21.3%의 저하가 관찰됩니다. NeFo는 LoRA의 가벼운 파라미터 업데이트와 unlabeled 테스트 데이터만으로도 Qwen2.5-VL 등 여러 베이스 모델에서 VQA/MCQ 성능을 유의미하게 개선하고, 이후 unseen 과제(분류·grounding·FloodNet VQA)로도 전이 일반화가 확인됩니다. 특히 기존 TTL 방법들이 grounding에서 급격한 붕괴를 보일 때 NeFo는 구성요소까지 포함한 ablation에서 안정적으로 이득을 유지해, RS 재난·안전 응용에서 ‘무엇이 아닌가’를 정확히 판단해야 하는 요구를 실증적으로 뒷받침합니다.



### MedRLM: Recursive Multimodal Health Intelligence for Long-Context Clinical Reasoning, Sensor-Guided Screening, Evidence-Grounded Decision Support, and Community-to-Tertiary Referral Optimization (https://arxiv.org/abs/2606.20164)
Comments:
          9 pages, 3 figures, 3 tables, 1 Algorithm, 29 equations

- **Prior Approaches**: 기존 의료 LLM과 RAG는 긴 환자 정보가 한 프롬프트에 압축되거나 단일 검색에 의존하는 경우가 많아, 장문 구간의 핵심 근거를 놓치고 근거 추적성과 신뢰성이 약해질 수 있다. 또한 순수 long-context 확장이나 retrieval만으로는 센서 신호 같은 비정형 입력을 위험도·워크플로우·의뢰 의사결정까지 일관되게 연결하기 어렵다. 그래프 기반 RAG와 멀티모달 RAG가 사실성은 개선했지만, 재귀적 임상 추론 흐름과 불확실성 기반 안전 정제, 의뢰 최적화까지 “시스템”으로 모델링한 연구는 제한적이었다.

- **Core Contribution**: MedRLM은 환자 케이스를 하나의 거대한 입력이 아니라 외부 임상 환경으로 보고, 재귀적으로 분해·검색·검증·종합하는 Recursive Multimodal Health Intelligence 프레임워크를 제안한다. 텍스트/EHR/이미지/센서/가이드라인/의뢰 규칙을 전담 에이전트가 조율하며, Clinical Evidence Graph Memory로 환자 관찰과 표준 정의, 근거, 의뢰 기준을 감사 가능하게 연결한다. 센서 기반 이상 패턴이 깊은 추론을 “트리거”하고, 불확실성이 높을 때는 불확실성-게이트 refinement 또는 clinician review로 안전한 경로를 선택한다.

- **Technical Challenges**: 핵심 난제는 긴·이질적 데이터에서 관련 근거를 안정적으로 찾아 쓰는 것과, 다중 모달 근거를 의뢰/치료 경로 결정에 안전하게 결합하는 것이다. 논문은 컨텍스트 복잡도(길이·모달 다양성·근거 분산·임상 위험·모순)를 계산해 재귀 분기 여부를 제어하고, 에이전트별 모달 추론 결과를 그래프 메모리에 정규화된 evidence weight로 축적한다. 더불어 센서 신호는 baseline-adjusted 이상도와 임계값으로 재귀 트리거를 만들고, 불확실성 점수(예측 불확실성·자기일관성·근거 충돌)를 기준으로 추가 검색/재귀 정제를 수행해 무리한 생성과 과신을 줄인다.

- **Empirical Impact**: 실험 파트에서는 MedRLM의 입력 채널을 실제 임상 데이터로 커버하는 벤치마크 설계를 제시하며, EHR(long-context), 흉부 영상/리포트, 생체신호/ECG, ICU 시계열 등 서로 다른 데이터 축을 포괄하도록 구성한다. 공개/자격 있는(real-data) 데이터만 사용하고 합성 케이스를 섞지 않았다는 점을 강조하며, 다만 공개 데이터에서 “community-to-tertiary referral” 라벨은 드물어 ICU admission·사망·급격한 악화·전문가 에스컬레이션 같은 proxy outcome로 평가해야 함을 명확히 한다. 또한 실제 결과 수치 대신 현재 공개된 real-data baseline 앵커(예: PhysioNet/CinC 2012 mortality의 공식 점수)를 제시해, 향후 MedRLM이 반드시 기존 기준선 대비 검증 가능하도록 함정 설정을 피하고 재현성/검증 가능성을 확보하려는 방향성을 보여준다.



### From Texts to Scores: Tracing the Emergence of Essay Quality Representations in Large Language Models (https://arxiv.org/abs/2606.20152)
Comments:
          This is a preprint of a manuscript currently under peer review

- **Prior Approaches**: 최근 LLM 기반 Automated Essay Scoring(AES)은 성능이 크게 향상됐지만, 점수 산출에 쓰이는 내부 표현 메커니즘은 충분히 해명되지 않았다. 기존 연구는 주로 출력 성능이나 학습 설정을 다루었고, hidden representations에서 품질 신호가 어떤 방식으로 인코딩되는지에 대한 분석은 제한적이었다.

- **Core Contribution**: 저자들은 8개 LLM의 hidden representations를 ASAP++, CSEE, ENEM(포르투갈어) 세 데이터셋에 걸쳐 체계적으로 분석한다. 결과적으로 에세이 품질 정보가 LLM 표현 안에 linearly accessible 형태로 존재하며, 레이어 전반에서 점진적으로 형성되고 다양한 prompting 전략에도 비교적 견고하다는 증거를 제시한다.

- **Technical Challenges**: 핵심 난제는 LLM 표현이 실제로 점수를 어떻게 담고 있는지 검증 가능한 형태로 드러내는 것이다. 선형 probing, cross-prompt generalization, 차원 축소, neuron-level 분석을 조합해 품질 신호의 선형 판독 가능성을 점검했으며, nonlinear probe는 선형 대비 향상폭이 작고 일관성이 낮아 대부분 신호가 선형 디코더로 충분함을 뒷받침한다.

- **Empirical Impact**: 추가로 특정 ‘essay scoring neurons’의 활성은 에세이 점수와 강하게 상관하고, 표적 개입에 민감하게 반응하는 것으로 확인돼 해석 가능성 실마리를 제공한다. 또한 essay length에 따라 이 뉴런들의 레이어 분포가 체계적으로 이동하며 긴 에세이는 더 깊은 레이어에 더 의존한다는 관찰로, LLM 기반 AES의 내부 동작을 보다 구조적으로 이해하는 데 의미가 있다.



### Hybrid ANN-SNN Pipeline with Local Plasticity (https://arxiv.org/abs/2606.20151)
Comments:
          9 pages, 4 figues, source-code available

- **Prior Approaches**: 기존 SNN 학습은 ANN2SNN 전환(활성→스파이크 레이트/타임), BPTT 같은 그래디언트 기반, 그리고 local learning(국소 플라스티시티)으로 나뉜다. 그래디언트나 전환 방식은 정확도가 높은 편이지만, local plasticity는 복잡한 아키텍처 적용이 어려워 “craft-like” 문제가 남는다. 또 CoLaNET 같은 국소 학습 SNN은 단일 fully connected 구조라 대형 이미지 분류엔 인코더 보강이 필요하다는 한계가 있었다.

- **Core Contribution**: 이 논문은 pretrained ANN 인코더(EfficientNet-B3)를 고정한 뒤, 그 임베딩을 rate-coding으로 스파이크 열로 변환하고 CoLaNET 스파이킹 분류기를 국소 규칙으로 학습하는 hybrid ANN-SNN 파이프라인을 제안한다. end-to-end gradient propagation을 피하고, CoLaNET의 biologically inspired local 플라스티시티로 분류기만 적응시킨다는 점이 핵심이다. 64-class ImageNet subset에서 99.09% 정확도를 보고해 기존 딥네트워크 수준 성능을 실증한다.

- **Technical Challenges**: 핵심 난제는 (1) 복잡한 CNN 표현을 SNN이 받아들일 수 있는 스파이크 표현으로 안정적으로 변환하는 것과 (2) 국소 플라스티시티만으로 대형 분류 문제에 충분히 분리 가능한 결정 경계를 만드는 것이다. 논문은 activation에서 음수 제거(ReLU 동치)→임계값 기반 스파이크 생성→sub-threshold 값을 0~10 스파이크 카운트로 선형 스케일하는 방식으로 rate-coding을 설계해 신호 크기 정보를 유지한다. 또한 9개 플라스티시티/동역학 하이퍼파라미터를 GA로 먼저 찾고, 이후 stochastic descent로 fine-tuning(개선 폭은 거의 없음)해 성능을 끌어올렸다.

- **Empirical Impact**: 학습은 epoch 반복 없이 단일 패스 온라인 학습(각 샘플 1회 제시)으로 진행되어 local learning의 실용성을 강조한다. 64-class 설정에서 최종 accuracy 99.09%를 달성했으며, 오류가 큰 클래스는 사람도 구분 어려운 유사 범주(거미 종, 시계가 포함된 종, 쓰레기통-통 유사 등)로 나타났다. ANN 기준선(추출 임베딩에 얹은 1,000 뉴런급 단층 ANN)과 비교해도 비슷한 정확도 축에 들어가 hybrid 접근이 “효율적 결정부 + 강력한 표현부” 전략으로 의미가 있음을 보여준다.



### Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation (https://arxiv.org/abs/2606.20135)
- **Prior Approaches**: Flow matching과 diffusion policy 계열은 복잡한 멀티모달 행동 분포를 잘 표현하지만, 대부분 고정 주파수의 action chunk를 이산 시점으로 예측한다. 이 방식은 서로 다른 control frequency로 수집된 데모를 학습할 때 주파수 불일치로 식별 불가능 문제가 생기고, 추론 시 인접 타임스텝이 일관되지 않아 jittery한(덜 매끄러운) 행동이 제어 안정성을 해칠 수 있다.

- **Core Contribution**: 이 논문은 Frequency-Aware Flow Matching(FAFM)으로 연속적이고 시간적으로 일관된 행동을 직접 생성하는 방법을 제안한다. 데모를 DCT(이산 코사인 변환) 계수로 바꾼 뒤 flow matching을 계수 공간에서 수행하고, 코사인 basis expansion으로 임의 시간 해상도에서 연속 행동을 복원한다. 또한 1차 시간 미분에 대한 regularization을 Sobolev-type 제약으로 넣어 고주파 오차와 급격한 변화를 억제한다.

- **Technical Challenges**: 핵심 기술 과제는 이질적인 sampling frequency로 기록된 데모를 하나의 step-index supervision에 억지로 맞추는 문제를 피하면서도, 연속 시간에서 일관된 trajectory를 학습하는 것이다. FAFM은 DCT로 step index를 물리 시간 기반 표현으로 전환해 주파수에 따른 목표 불일치를 줄이고, DCT 파라미터화 덕분에 미분(velocity) 감독을 유한차분 잡음 없이 정확히 정의해 derivative flow matching을 안정적으로 적용한다.

- **Empirical Impact**: 합성 toy 벤치마크, obstacle avoidance, LapGym, LIBERO에서 FAFM은 성공률, 멀티모달 expressivity, motion smoothness, 수렴 속도, 기계적 bias 및 mixed-frequency 입력에 대한 견고성을 함께 개선한다. 특히 LapGym처럼 연질(soft-body) 조작에서 jitter를 줄여 더 안정적인 제어를 보였고, Franka 로봇 실배치에서도 성능 우위가 일관되게 나타났다. 저자들은 DCT와 1차 미분(velocity) 감독의 기여를 ablation으로 확인했으며, 연속 주파수 표현 없이 finite difference 기반 derivative supervision은 효과가 제한적이라고 보고한다.



### Dual-Agent Framework for Cross-Model Verified Translation of Natural-Language Protocols into Robotic Laboratory Platform (https://arxiv.org/abs/2606.20120)
- **Prior Approaches**: 기존 자동화 시스템은 미리 정의된 제어 명령에 의존해, 생물학 실험 프로토콜의 자연어와 로봇 실행 간에 semantic gap이 생기는 문제가 컸습니다. 특히 microplate 기반 실험은 well mapping, 시료-시약 조합, 반복( replicate ) 배치, 병렬 분주를 동시에 맞춰야 해서 제약 반영이 어렵습니다.

- **Core Contribution**: 이 논문은 자연어 microplate 프로토콜을 로봇이 실행 가능한 제어 명령으로 바꾸는 agent-based protocol translation 프레임워크를 제안합니다. Parser Agent가 프로토콜을 구조화하고, rule-based mapping engine이 로봇 플랫폼의 운영 제약을 결정적으로 반영해 device-level 제어 커맨드를 생성하며, LLM Validation Agent가 완전성·정확도·순서를 검증합니다.

- **Technical Challenges**: 핵심 과제는 (1) 자연어에서 누락된 파라미터를 찾아 정확한 값으로 채우고, (2) 로봇 제약을 만족하는 실행 순서를 보장하며, (3) 오류가 나면 재현 가능한 방식으로 self-correction 루프를 돌리는 것입니다. 연구진은 structured feedback를 이용한 검증 및 수정 루프와 cross-model verification을 결합해, 모델 규모와 Validator 유형 변화에도 번역 품질을 점검하도록 설계했습니다.

- **Empirical Impact**: 무작위로 선택한 ELISA 프로토콜에서 7개의 Parser와 3개의 Validator를 사용한 sweep으로 번역 정확도와 pass rate를 평가했고, 모델 규모·Validator 타입이 성능에 미치는 영향을 분석했습니다. 또한 rule-based mapping을 LLM end-to-end direct mapping과 비교해 accuracy–latency trade-off를 확인했으며, Bradford assay 단백질 정량을 로봇 실험으로 성공적으로 수행해 자연어→실세계 end-to-end 자율 실행 가능성을 입증했습니다.



### Sensorimotor World Models: Perception for Action via Inverse Dynamics (https://arxiv.org/abs/2606.20104)
- **Prior Approaches**: 행동을 위한 지각(perception for action)은 시각 충실도 자체보다 “행동에 유용한 표현”이 중요하다는 문제의식에서 출발한다. 하지만 JEPA 스타일의 픽셀 기반 latent world model은 end-to-end 학습 시 인코더가 모든 입력을 한 점으로 보내는 representation collapse 위험이 커, 실제로 DINO-WM(encoder freeze), PLDM(분산-공분산 정규화), LeWorldModel(SIGReg), V-JEPA 2(stop-gradient+EMA)처럼 우회 장치가 필요했다.

- **Core Contribution**: 이 논문은 sensorimotor world model(SMWM)을 제안하며, collapse 방지와 행동 정렬(action-aligned)을 단일 정규화로 동시에 달성한다. 구체적으로 inverse dynamics regularization(역동역학 정규화)을 도입해, 임베딩 두 시점에서 해당 전이를 만든 행동을 복원할 수 있어야 하도록 학습을 end-to-end로 유도한다.

- **Technical Challenges**: 핵심 기술적 난제는 “미래 상태 예측만 쉬워지도록” 인코더가 표현을 붕괴시키는 유인과, 그럼에도 행동에 필요한 정보는 보존해야 한다는 양립 문제다. SMWM은 forward dynamics 예측 손실에 inverse dynamics head를 추가하고, 해당 손실의 그래디언트를 인코더로 역전파해 행동을 설명하는 정보만 남기고 잡음처럼 보이는 비조절 요소는 버리도록 만든다.

- **Empirical Impact**: 오프라인·보상 없는( reward-free ) 픽셀-행동 궤적만으로도 SMWM은 compact하고 해석 가능한 latent space를 학습하며, 장면/상태의 제어 가능 차원을 잘 반영하는 것으로 분석된다. 2D·3D 제어 태스크에서 planning 성능이 SIGReg 같은 정규화 기반 경쟁 방법을 접합 수준으로 따라가거나 OGBench-Cube 같은 3D 접촉 조작에서는 더 높은 성공률(예: 84% vs 59%)을 보이며, 추가 장치 없이도 유용한 world model 표현을 만든다는 점에서 의미가 크다.



### Hybrid Diffusion Transformer for Instruction-Guided Audio Editing via Rectified Flow (https://arxiv.org/abs/2606.20101)
- **Prior Approaches**: 기존 text-guided audio editing은 training-free(확산 inversion/최적화 기반)와 training-based(지도학습 기반)로 나뉜다. training 기반 U-Net 편집기는 텍스트-오디오 cross-attention에 의존하면서 지역적 inductive bias로 인해 장거리 의미 정렬과 지시의 정밀 이해·로컬라이징이 약해지는 문제가 있다.

- **Core Contribution**: 이 논문은 rectified flow matching(RFM) 기반의 continuous-time 하이브리드 diffusion transformer를 이용해 instruction-guided audio editing을 coarse-to-fine으로 수행한다. 저해상도 단계에서는 audio-텍스트 joint attention으로 거친 의미 정렬을 만들고, 고해상도 단계에서는 joint attention과 cross-attention을 번갈아 써 디테일 편집을 정교화한다.

- **Technical Challenges**: 핵심 난제는 (1) 텍스트-오디오 토큰을 전 블록에서 전부 joint attention으로 묶으면 토큰 길이에 대해 quadratic 복잡도가 커져 효율을 해친다는 점과 (2) 편집 구간만 바꾸면서 비편집 구간 보존을 유지해야 한다는 점이다. 이를 위해 저해상도 토큰 다운샘플링으로 joint attention 비용을 줄이고, 고해상도에서는 AZCA-DiT로 오디오 토큰만 업데이트하는 cross-attention 경로를 도입했으며, AdaLN-Zero 기반 전역/토큰 조건 변조로 제어성과 보존성을 함께 끌어올린다.

- **Empirical Impact**: AudioCaps, AudioSet, AudioSetCaps를 바탕으로 add/remove/replace 3개 태스크를 구성해 CLAP, FD/FAD, LSD, KL, IS 및 편집 시간(AET)으로 평가했으며, 특히 겹치는 사운드 이벤트와 복잡한 지시에서 성능이 두드러지게 개선됐다. 또한 compact 모델로도 추론 효율을 크게 개선해, 기존 joint attention 중심 아키텍처의 계산 부담을 실질적으로 완화한다는 점에서 의미가 있다.



### MakeupMirror: Improving Facial Attribute Preservation in Diffusion Models for Makeup Transfer (https://arxiv.org/abs/2606.20094)
- **Prior Approaches**: 기존 메이크업 트랜스퍼는 물리 기반 렌더링과 GAN/확산(diffusion) 기반 접근으로 발전해 왔지만, 복잡한 메이크업 질감과 얼굴 보존을 동시에 만족시키기 어려웠습니다. 특히 Stable-Makeup 같은 diffusion 기반 성능은 좋아졌지만, 인물 간(cross-subject) 전송에서 얼굴 특징(눈/코 형태 등)과 피부 톤이 의도치 않게 바뀌는 문제가 남아 e-commerce 수준의 VTO(virtual try-on) 구현을 어렵게 했습니다.

- **Core Contribution**: 이 논문은 얼굴 기하와 피부 톤을 더 강하게 보존하도록 확산 모델을 개조한 MakeupMirror를 제안합니다. 핵심은 (1) facial geometry conditioning으로 얼굴 구조를 유지하고, (2) 부위별로 메이크업 적용 강도를 달리하며, (3) Monk scale 기반 피부 톤 차이를 감지해 전송 강도를 자동 조절함으로써 인물 간 충돌을 줄이는 것입니다.

- **Technical Challenges**: 문제는 메이크업 스타일을 옮기는 동시에 source 얼굴의 정체성/피부 톤을 깨지 않게 제어하는 것입니다. 이를 위해 ControlNets에 depth(Depth-Anything)와 저수준 edge(Canny) 정보를 결합해 얼굴 형태 fidelity를 높였고, segmentation 기반 마스크로 피부/눈/입의 classifier-free guidance scale과 clamping time-steps를 다르게 적용해 over-application을 억제했습니다. 또한 Levenberg–Marquardt Langevin sampler를 통합해 품질 저하 없이 추론을 few-step 수준으로 가속했습니다.

- **Empirical Impact**: CPM-Real, Makeup Wild뿐 아니라 피부 톤 다양성을 강화한 새 데이터셋 MakeupSelfies에서도 Stable-Makeup 대비 성능이 개선됐습니다. 결과로 얼굴 인식 유사도는 +60% 개선, 피부 톤 차이는 -50% 감소, 그리고 뷰티 전문가 기준 결함 감사에서 94% pass-rate를 달성했습니다. 지연 시간은 0.7s로 보고되며, 전체 파이프라인은 약 2.8× 속도 향상까지 확인되어 실서비스 VTO 가능성을 뒷받침합니다.



### IHUBERT: Vector-Based Semantic Deduplication and Domain-Balanced Pretraining for Persian Resources (https://arxiv.org/abs/2606.20089)
- **Prior Approaches**: 기존 페르시아 PLM(pretrained language models)은 대규모 고품질 사전학습 말뭉치의 부족과, 분류·NER 같은 표준 작업에 편중된 평가 한계로 인해 성능 확장에 제약이 있었다. 또한 말뭉치 품질(정규화/중복/익명성)과 도메인·레지스터 균형을 체계적으로 통제하지 못한 경우가 많아 모델의 일반화가 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 RoBERTa-base(125M) 인코더를 from scratch로 학습한 단일언어 페르시아 PLM “IHUBERT”를 제안하며, 45GB 규모의 Sepahr-Danesh 기반 정제 코퍼스(약 70억~80억 토큰)를 사용한다. 특히 토크나이저를 위해 전체 사전학습 코퍼스에서 139k-vocabulary BPE를 학습해 페르시아의 형태론·철자 변이를 더 잘 포착하도록 설계했다.

- **Technical Challenges**: 핵심 난관은 (1) 말뭉치 정제와 중복 제거를 단순 문자열 수준을 넘어 의미 중복까지 줄이면서, (2) 도메인·레지스터 분포 균형을 유지하는 것이었다. 이를 위해 정규화, exact/near-duplicate 제거, anonymization, 그리고 벡터 DB 기반 semantic deduplication을 멀티스테이지 파이프라인으로 적용했으며, 토크나이저는 BPE/WordPiece를 비교한 제어 실험(토크나이저 ablation)으로 단편화 감소를 확인했다.

- **Empirical Impact**: IHUBERT는 7개 페르시아 NLU 벤치마크에서 NER, sentiment, topic, NLI, extractive QA, relation extraction을 모두 평가하며 분류형과 이해형 과제를 폭넓게 커버한다. 특히 extractive QA에서 PQuAD(F1 88.3542), ParsiNLU-RC(F1 49.0987) 1위를 달성했고, FarsTail(Macro-F1 0.8350)에서도 최고 성과를 보였다. 반면 relation extraction은 PERLEX에서 Macro-F1 0.6684로 상대적 격차가 남았지만, 의미 정제 대규모 사전학습과 확장 평가를 통해 페르시아 언어 모델링의 실질적 진전을 제시했다.



### The Hidden Evolution of Disguised Visual Context inside the VLM (https://arxiv.org/abs/2606.20077)
- **Prior Approaches**: 기존 VLM 연구는 시각 토큰을 LLM 입력에 붙여 self-attention으로 처리하는 in-context injection과, 중간 레이어에 시각 정보를 끼워 넣는 layer-wise injection으로 크게 나뉜다. 그러나 두 패러다임을 비교할 때 데이터 구성, 토큰 예산, 모델 스케일, 최적화 조건이 함께 달라져 “통합 아키텍처 자체가 무엇을 바꾸는지”를 인과적으로 분리하기 어려웠다. 또한 분석도 주로 한 가지 통합 방식에 치우쳐, 표상 진화나 모달리티 정렬이 다른 방식으로 일반화되는지 불명확했다.

- **Core Contribution**: 이 논문은 in-context injection(IN-CT)과 두 종류 layer-wise injection(LW-GC, LW-AT)을 동일한 학습 조건에서 단일 이미지/멀티 이미지/비디오 벤치마크로 공정 비교한다. 그 결과 LLM 내부에서 시각 토큰이 ‘언어 구조를 가진 시각 맥락처럼 위장(disguised)되어 들어가지만’, 통합 방식에 따라 표상 형태가 서로 다르게 재구성되며, 각 방식이 시각 신호의 서로 다른 frequency 특성을 포착한다는 “숨은 진화(hidden evolution)”를 제시한다. 더 나아가 attention 분배만으로 성능 차이를 설명할 수 없고, 레이어별 시각 표상의 품질이 실제 능력을 좌우한다고 주장한다.

- **Technical Challenges**: 핵심 기술 난제는 통합 설계만 바꿔도 내부 처리 메커니즘이 어떻게 달라지는지 ‘같은 학습 레시피’로 분리해 측정하는 것이다. 이를 위해 커넥터 구조와 데이터/학습 단계를 통제한 뒤, (1) 레이어 전개에서 시각 토큰 표상이 매끄럽게 진화하는지 CKA로 구조 변화를 추적하고, (2) Fourier 기반 주파수 분석으로 레이어별 고주파/저주파 편향을 정량화하며, (3) PCA로 시각·텍스트 토큰이 언어 공간으로 수렴하는지 확인하고, (4) 생성 과정에서 시각 토큰에 대한 Attention Mass로 실제 활용 타이밍을 측정한다. 특히 IN-CT는 레이어 전반에 걸친 연속적 재구성이 나타나지만, LW-GC/LW-AT는 레이어마다 불연속적 표상 점프가 관찰되어 ‘왜 언어 정렬이 달라지는지’를 구조적으로 뒷받침한다.

- **Empirical Impact**: 실험 결과 IN-CT가 전반적으로 가장 좋은 성능을 보이며, 특히 OCR과 비디오에서 layer-wise 방식보다 큰 격차가 반복된다. OCR-heavy 데이터에서 IN-CT는 전반적인 성능 하락이 크고, layer-wise는 텍스트가 섞인 과제에서만 선택적으로 무너져 “in-context 토큰이 서로 attention으로 결합해 패치/프레임 전반의 세밀한 증거를 조립”하는 능력 차이를 시사한다. 또한 생성 중 attention 분포가 비슷하게 나오는 경우에도 IN-CT가 이기는 것을 통해, 성능은 attention이 아니라 레이어별 시각 표상의 주파수 품질과 언어 공간 정렬의 정도에 의해 좌우됨을 설득력 있게 보여준다. 나아가 IN-CT와 LW-AT를 하이브리드로 결합하면 대부분의 벤치마크에서 성능이 개선되어, 주파수 특성의 상보성을 설계에 반영할 수 있다는 실용적 함의도 제시한다.



### Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers (https://arxiv.org/abs/2606.20076)
- **Prior Approaches**: Latent Diffusion Models(LDMs)은 토크나이저 압축률이 품질-연산 비용 트레이드오프를 좌우하지만, 기존 토크나이저는 고정 압축률이라 시나리오별 최적화에 한계가 있었다. 이를 보완하려고 Variable-length tokenizers(VLTs)들이 등장했는데, 대표적으로 nested dropout은 tail 토큰을 무작위로 잘라 길이를 조절해 효율을 얻는다. 하지만 잘림(truncation) 방식은 토큰 순서에 기반한 의미 구조가 길이마다 달라져, 길이 간 데이터포인트 유사도(대표성) 구조가 어긋나 diffusion 모델이 단일 모델로 다양한 토큰 길이를 잘 일반화하기 어렵게 만든다.

- **Core Contribution**: 이 논문은 길이 조절을 truncation이 아니라 token merging으로 수행하는 “merging-based variable-length tokenizer”를 제안한다. 핵심 아이디어는 merging 패턴(어떤 토큰이 합쳐지는지)이 diffusion transformer 입력에서 함께 고려되면, 길이가 달라도 동일한 ‘cardinality를 맞춘 full-length equivalent representation’ 관점으로 정렬이 가능해진다는 점이다. 즉, 길이로 인해 생기는 대표성 시프트를 줄이기 위해 “유사한 토큰끼리 병합”하도록 학습 목표를 설계한다.

- **Technical Challenges**: 문제는 생성 시점에 데이터(이미지)에 의존하는 conventional merging/클러스터링처럼 “입력에 맞춘 병합 패턴”을 알 수 없다는 호환성 제약이다. 이를 해결하기 위해, 이미지에 무관한 learnable global merging(전역 병합)을 도입해 병합 패턴이 생성 단계에서 고정·접근 가능하도록 만들었다. 또한 merged token의 크기를 반영하는 proportional attention과, 병합된 토큰의 위치 정보를 처리하는 merged positional embeddings를 통해 diffusion이 병합 구조를 일관되게 반영하도록 했다.

- **Empirical Impact**: ImageNet 256×256 생성 실험에서, 제안된 병합 기반 VLT는 기존 VLT 대비 gFID-compute trade-off가 더 우수함을 보였다. 특히 nested dropout 계열에서 관찰되던 길이 간 representational alignment 저하가 완화되어 단일 variable-length diffusion 모델의 성능이 개선되는 방향을 실증했다. 요약하면, “병합 기반 토큰 가변성”이 LDM의 유연한 품질-연산 제어를 더 현실적으로 만들 수 있음을 보여준 결과로 평가된다.



### Evaluation of EEG Foundation Models for Event-Based Burst-Suppression Detection in ICU (https://arxiv.org/abs/2606.20074)
Comments:
          4 pages, 1 figure. Code available upon publication

- **Prior Approaches**: 기존 자동 burst suppression(BS) 탐지는 주로 수작업 특징과 전통 ML, 순환신경망, 고정/적응 임계값, 상태공간 추정, 클러스터링 같은 방식에 의존했습니다. 특히 RVT 같은 임계값 기반 방법은 간단하면서도 경쟁력이 있지만, ICU 잡음·신호 변동과 환자 간 BS 패턴 차이에 민감해 환자별 보정이 필요할 수 있습니다. 아울러 대부분의 평가는 윈도우 단위 분류(F1)에 머물러 임상적으로 중요한 burst 이벤트 경계 오차나 burst count 추정의 품질을 충분히 반영하지 못했습니다.

- **Core Contribution**: 이 논문은 reduced-montage ICU EEG(6채널)에서 환자별 calibration 없이 generalized BS detection이 가능한지, EEG foundation model(FM)들을 처음으로 체계 평가합니다. REVE-base, LUNA-large, LuMamba-Tiny를 task-specific EEGNet과 RVT 기준선과 비교하고, 윈도우 분류를 넘어 event-based burst detection과 BPM(bursts per minute) 오차까지 포함해 임상 해석 가능성을 높였습니다. 특히 burst per minute 같은 실제 임상 지표에서 모델 성능을 정량화해, 단순 분류 정확도 이상의 의미를 부여한 점이 핵심입니다.

- **Technical Challenges**: 핵심 난제는 환자마다 다른 BS 형태와 ICU 아티팩트로 인해 라벨된 데이터가 희소할 때 자동 탐지가 흔들린다는 점입니다. 연구진은 pretraining된 EEG FM을 다운스트림에 적응할 때 frozen-backbone, LoRA, two-step fine-tuning보다 full fine-tuning이 가장 잘 맞는다는 것을 보였고, 또한 limited labeled data 환경에서 pretrained 초기화가 random 초기화 대비 event-based F1을 크게 끌어올림을 확인했습니다. 평가도 event 중심(입·출발 경계 허용 오차 포함)으로 설계해 annotation variability 영향이 성능 해석에 미치는 부담을 줄였습니다.

- **Empirical Impact**: 결과적으로 REVE-base가 가장 높은 event-based F1-score(0.868 ± 0.167)와 가장 낮은 BPM MAE(0.448 ± 0.284)를 기록했으며, BPM MAE를 EEGNet 대비 52.1%, RVT 대비 36.2% 줄였습니다. 또한 적응 전략 실험에서 full fine-tuning이 고정 백본 대비 LUNA-large 기준 event-based F1을 최대 +0.102까지 개선하며, pretraining의 데이터 효율 이점을 뚜렷하게 확인했습니다. 특히 학습 데이터 25%만 쓸 때 REVE-base는 random 초기화 대비 event-based F1이 +0.723p 향상되어, ICU처럼 라벨이 귀한 환경에서 확장 가능한 BS 모니터링 가능성을 제시했습니다.



### See-and-Reach: Precise Vision-Language Navigation for UAVs within the Field of View (https://arxiv.org/abs/2606.20045)
Comments:
          12 pages, 7 figures

- **Prior Approaches**: 기존 UAV-VLN 연구는 목표를 “발견(검색)”한 뒤 “접근(도달)”까지를 한 덩어리 search-and-reach로 학습·평가하는 경향이 강합니다. 이때 종종 저해상도 관측과 느슨한 성공 반경(예: 20m)이 쓰여 단말(terminal)에서의 정밀 도달 능력은 진단하기 어렵습니다. 또한 방향(direction) 가이드를 초기 고정 단서로 두는 방식은 비행 중 시점 변화로 누적 drift가 생기기 쉽습니다.

- **Core Contribution**: 논문은 보이는 타깃을 이미 시야(FOV) 안에서 확인한 상태로 두고 “see-and-reach”만 분리 평가하는 UAV-VLN-FOV 태스크를 제안합니다. 성공 기준을 10m로 더 엄격하게 두어, 공중 embodied agent의 정밀한 종단 도달 능력을 측정 가능하게 만듭니다. 이를 통해 언어·시각 증거를 정확한 3D 기동으로 변환하는 핵심 역량을 직접 겨냥합니다.

- **Technical Challenges**: 정밀 see-and-reach에서는 타깃이 작게 보이거나 유사한 시각적 방해물과 섞일 수 있어 고해상도·세밀한 기하 정보 보존이 필수입니다. 3DG-VLN은 다운샘플링을 피하고 front-view와 downward-view를 고해상도로 적응 처리해 fine-grained visual grounding과 waypoint 예측을 강화하며, LoRA로 Qwen2.5-VL을 경량 fine-tuning해 기하 민감도를 확보합니다. 또 방향 단서가 비행 중 어긋나는 문제를 해결하기 위해 closed-loop 추론 중 현재 관측 기반으로 target-relative direction을 online 업데이트해 누적 direction drift를 줄입니다.

- **Empirical Impact**: UAV-VLN-FOV용 전용 고해상도 벤치마크(총 2,717 trajectories)를 구축해 학습·평가 기반을 마련했습니다. 실험에서 3DG-VLN은 경쟁 UAV-VLN baseline 대비 success rate를 13.82% 향상시켰고, 보이는 타깃을 향한 정밀 도달에서 개선이 확인됩니다. 나아가 real-world trial에서도 see-and-reach 내 활용 가능성을 보여 UAV 정밀 내비게이션의 실전 적용성을 높였다는 점이 의미 있습니다.



### AI Economist Agent: An Agentic Framework for Model-Grounded Economic Analysis with RAG, Knowledge Graphs, and Large Language Models (https://arxiv.org/abs/2606.20041)
- **Prior Approaches**: 기존 연구는 LLM을 경제 리포트 생성에 활용하되, 주로 자연스러운 서술 품질에 초점을 맞추는 경향이 강했습니다. RAG는 텍스트 증거를 끌어와 주지만, 정량 주장이나 모델 기반 계산이 실제로 실행되었는지는 보장하기 어렵습니다. 반면 그래프 기반 접근은 구조화 검색에는 강점이 있으나, 자유로운 그래프 확장으로 과다 문맥이 들어오거나 모델 계산과의 연결이 약해질 수 있습니다.

- **Core Contribution**: 이 논문은 모델 기반(model-grounded) RAG를 결합한 “AI economist agent”를 제안합니다. 지식 그래프에 경제 이론/데이터(증거)와 모델 사양, 실행 결과를 함께 두고, 에이전트가 계획-검색-모델 선택-모델 실행-보고서 생성을 순차로 수행합니다. 특히 LLM이 수치 주장을 직접 생성하지 않고, 실행된 수학 모델 출력에만 근거해 서사를 작성하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 ‘기술적으로 그럴듯한 서술’이 아니라 ‘경제 메커니즘과 모델 실행 결과에 추적 가능한 서술’을 만드는 것입니다. 이를 위해 그래프에서 텍스트 증거와 모델 사양을 분리하고, 모델 선택도 그래프에서 검색된 모델 객체를 구조화된 실행 요청으로 변환해 컴퓨테이션 레이어가 검증·실행합니다. 또한 모델 실행 전/후로 그래프 경로를 다시 조회해, 보고서가 실행 결과 노드와 연결되도록 강제했습니다.

- **Empirical Impact**: 실험은 미국 인플레이션 지속성과 연준 정책, 그리고 미국 상업용 부동산(CRE) 리파이낸싱 스트레스의 두 태스크에서 LLM-only, RAG-only, model-grounded GraphRAG를 비교합니다. 결과적으로 model-grounded GraphRAG는 그래프/모델 근거 연결과 추적성에서 더 높은 판정 점수를 받았고, RAG-only는 증거 인용은 잘하지만 모델 출력 연결이 부족했습니다. 이 방식은 경제 리포트를 ‘유창한 글’에서 ‘검증 가능한 모델 기반 분석’으로 전환하는 방법론적 임팩트를 보여줍니다.



### A Neuromorphic Reinforcement Learning Framework for Efficient Pathfinding in Robotic Mobile Fulfillment Systems (https://arxiv.org/abs/2606.20031)
- **Prior Approaches**: RMFS(로봇 모바일 풀필먼트) 경로탐색은 좁은 통로, 동적 장애물, 강한 실시간 제약 때문에 기존 탐색/룰 기반(A*, Dijkstra 등)이 재계획 비용과 지연 문제를 겪기 쉽습니다. RL/DRL은 적응성이 좋지만, 모바일 AGV의 전력·연산 제약에서 ANN 추론이 부담이 되고 실제 에너지 절감이 충분히 검증되지 못했습니다. 또 ANN-to-SNN 변환은 주로 분류용(희소 one-hot) 설정에 맞춰져 있어, RL의 연속적인 Q-value 출력 분포를 그대로 옮기기 어렵다는 한계가 지적돼 왔습니다.

- **Core Contribution**: 이 논문은 end-to-end로 SDQN-RMFS를 제안하며, DQN(ANN 기반) 학습부터 ANN-to-SNN 변환, 그리고 neuromorphic chip SPECK2E로의 오프라인 물리 배포까지 한 파이프라인에서 “정확도 보존+초저전력”을 목표로 합니다. 핵심은 RL 출력 분포 불일치를 줄이기 위해 hard-label knowledge distillation(argmax 기반 pseudo-label)을 사용해 SNN이 RL 정책의 의사결정을 더 견고하게 학습하도록 만든 점입니다. 더불어 하드웨어 지연을 줄이기 위한 parameter scaling과 바이어스 제거로 극저 time-step에서도 동작성을 확보합니다.

- **Technical Challenges**: 문제는 세 갈래로 정리됩니다: (1) 좁은 공간에서 무작위 탐색이 충돌로 학습이 정체되는 문제, (2) SNN을 매우 짧은 time-step(예: 4)에서 돌릴 때 초기 스파이크 지연으로 정보가 제대로 전달되지 않는 문제, (3) ANN-to-SNN 변환 시 RL의 연속 Q-value와 SNN의 이산 스파이크/하드 라벨 간 분포 mismatch로 인해 선택이 흔들리는 문제입니다. 해결책으로 collision-allowing strategy로 일정 한도 내 충돌을 허용해 탐색 정체를 완화하고, 첫 레이어 가중치를 k배로 키워 빠른 스파이크 트리거를 만들며, distillation로 출력 마진을 키워 변환 잡음에도 action 선택이 유지되게 했습니다.

- **Empirical Impact**: 시뮬레이션과 변환/배포 평가에서 SNN의 conversion rate(CRCR)는 fine-tuning과 scaling 조합 시 4 time-step에서도 1.00 수준을 달성해 정책 충실도를 입증합니다. 물리 하드웨어 실험에서는 Speck 칩에서 에너지 사용이 GPU 대비 최대 11,281× 감소했고, 추론 지연은 고성능 GPU 대비 약 2배 수준으로 줄어드는 결과(거의 절반 수준)가 보고됐습니다. 즉, neuromorphic inference가 RMFS 대규모 운용에서 실제 전력 병목을 완화할 수 있는 실행 가능한 경로임을 실험적으로 뒷받침했다는 점에서 의미가 큽니다.



### When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents (https://arxiv.org/abs/2606.20023)
Comments:
          code: this https URL

- **Prior Approaches**: 기존 에이전트 안전 연구는 주로 prompt injection, tool injection, jailbreaking 같은 공격/정책 위반 출력에 초점을 맞췄고, 도구 선택 편향은 provider identity나 description 같은 메타데이터 선호 문제로 주로 다뤄졌다. 또 privilege escalation을 외부 조작에 의한 결과로 보거나, 시스템 차원의 권한 경계 강제처럼 에이전트 내부의 ‘최소 권한 선택’ 성향은 상대적으로 덜 검증됐다. 그 결과, 낮은 권한의 충분한 대안이 있는데도 높은 권한 도구를 고르는 행동(과권한 선택)이 얼마나 흔한지 정량화가 부족했다.

- **Core Contribution**: 논문은 과권한 선택(over-privileged tool selection)을 “충분한 낮은-privilege 도구가 가능한데도 더 높은 privilege를 직접 선택하거나 일시적 실패 뒤에 승격하는” 문제로 정의한다. 이를 평가하기 위한 벤치마크 ToolPrivBench를 제안해 초기 선택뿐 아니라 transient failures 이후 escalation까지 함께 측정한다. 또한 일탈(과권한) 원인을 ‘도구 성능 부족’이 아닌 ‘권한 선호 편향’으로 분리하도록, 각 도구가 비오류 조건에서 독립적으로 충분(sufficient)하도록 시뮬레이션 환경을 설계한다.

- **Technical Challenges**: 핵심 기술적 난관은 높은 권한을 쓰는 것이 단순히 낮은 권한 도구가 실패해서 ‘합리적’ 선택일 수 있다는 혼동(confound)을 제거하는 것이다. 논문은 함수적 충분성 조건을 엄격히 강제하고, Gemini 2.5 Pro와 GPT-5.2의 교차 합의 심사로 도구 sufficiency를 1차 검증한 뒤 인간 전문가 감사로 애매한 권한 대비나 표준 도구의 부족 사례를 제거한다. 그 위에 transient, privilege-unrelated 오류(예: 연결 에러)를 표준 도구 호출에 주입해 “실패가 오면 에이전트가 더 큰 권한으로 도피하는지”를 자연스럽게 관찰하고, OPUR@k와 PED로 공격적 선택/성급한 승격을 구분해 평가한다.

- **Empirical Impact**: 실험 결과 ToolPrivBench의 8개 도메인·5개 위험 패턴에서 주류 LLM 에이전트 다수가 최소 권한 대안이 남아있는데도 높은 권한 도구를 선택/승격하는 현상이 흔했다. 특히 transient failure가 발생하면 OPUR이 더 크게 증폭됐고, 일반적인 safety alignment는 least-privilege 도구 선택으로 잘 전이되지 않았으며 prompt-level 제어는 transient 실패 상황에서 완화 효과가 제한적이었다. 이를 해결하기 위해 privilege-aware post-training 방어(표준 도구 우선, 필요할 때만 escalation)를 제안했고, GRPO 기반 학습을 통해 불필요한 high-privilege 사용을 크게 줄이면서 전반 성능(MMLU, GSM8K, MetaTool)은 대체로 유지되는 것으로 나타났다. 다만 시뮬레이션과 소규모 도구 묶음의 ‘독립 충분’ 조건 때문에 실제 운영 환경의 복잡한 멀티-툴/장기 지평을 완전히 대변하진 못하며, 향후 더 긴 시퀀스와 더 큰 도구 인벤토리로 확장이 필요하다는 한계도 제시한다.



### Hierarchical Control in Multi-Agent Games: LLM-based Planning and RL Execution (https://arxiv.org/abs/2606.20014)
Comments:
          12 pages, 9 figures

- **Prior Approaches**: 기존 RL은 순차 의사결정에서 성능이 좋지만, 부분 관측·희소 보상·큰 상태-행동 공간·긴 시간 의존성 때문에 복잡한 다중 에이전트 협응/경쟁으로 확장하기가 어렵습니다. HRL은 스킬(옵션)로 시간 추상화를 제공하지만, 유용한 스킬 자동 발견과 고수준 정책 학습이 불안정·비효율적일 수 있습니다. 게임 AI 쪽에서는 behavior tree(BT)가 모듈화는 잘 되지만 수작업 규칙이 많이 필요하고, end-to-end “Flat” RL은 기술 분해 없이 학습해 경쟁 환경에서 협동 전략을 잘 못 잡는 한계가 있습니다.

- **Core Contribution**: 이 논문은 pretrained LLM이 중앙 strategic controller로서 다중 에이전트의 전술 스킬 선택을 지휘하고, RL 스킬 정책이 저수준 반응 제어를 담당하는 2단 계층 구조를 제안합니다. LLM은 전역 게임 상태를 바탕으로 각 에이전트를 주어진 스킬(Navigate/Combat/Secure/Retreat) 중 하나에 할당하고, RL은 지역 관측으로 빠른 주기에서 행동을 실행합니다. 수작업 rule engineering 없이도 LLM 추론이 pretrained RL skill들을 orchestrate해 경쟁형 다중 에이전트 조정을 만든다는 점이 핵심 기여입니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) LLM이 지연·물리적 근거 부족 때문에 저수준 제어를 직접 하기 어려운 점, (2) 고수준 스킬 전환이 틀어질 때 RL이 out-of-distribution 상태에 부딪힐 수 있다는 점입니다. 저자들은 LLM을 느린 2Hz 메타 결정자로 두고, 스킬 전환은 이산화된 상태 카테고리와 낮은 temperature(0.1)로 안정화했으며, 스킬들은 서로 다른 보상 함수로 독립 학습해 각 목적에 특화되게 했습니다. 또한 Flat RL 기준선은 PPO 단일 정책으로 end-to-end 학습하되 curriculum을 사용해 self-play의 일반화 실패를 완화했습니다.

- **Empirical Impact**: 경쟁 2v2 King of the Hill에서 LLM+RL은 BT와 통계적으로 유사한 승률(46.4% vs 51.5%, p=0.103)을 보이며, Flat RL은 크게 앞섰습니다(p<0.001). 에이전트 수준 지표에서도 LLM+RL은 K/D 균형에 더해 Health pickup을 가장 많이 수집해 생존 전략이 잘 작동했음을 보여줍니다. 더 나아가 사용자 연구(n=15)에서 60%가 LLM+RL을 가장 human-like로 선택했고(p=0.027), 플레이어들은 행동 적응성과 전술적 다양성(전환의 맥락성)을 이유로 들며 ‘believability’ 측면의 우수함을 확인했습니다.



### StreamKL: Fast and Memory-Efficient KL Divergence for Boosting Attention Distillation (https://arxiv.org/abs/2606.20005)
- **Prior Approaches**: 어텐션 distillation에서 KL divergence를 계산할 때, 기존 방식은 P1=softmax(Q1K1^T), P2=softmax(Q2K2^T) 두 분포를 모두 NQ×NK 크기로 HBM에 materialize한 뒤 원소별 KL을 줄이는 구조였다. 이 접근은 컨텍스트가 길어질수록 메모리와 IO가 O(NQNK)로 폭증해 GPU 단일 장치에서 장문 distillation을 어렵게 만든다. 일부는 query chunking으로 메모리를 줄이지만, 작은 chunk는 지연이 커지고 큰 chunk는 OOM으로 이어져 운영점이 부족하다.

- **Core Contribution**: StreamKL은 attention KL divergence를 위한 최초의 fused GPU primitive로, 두 분포의 O(NQNK) materialization 없이 one-pass로 처리하도록 온라인(online) 수식을 새로 유도했다. forward는 key tile을 SRAM으로 스트리밍하며 두 분포가 결합된 KL 항을 동시에 누적하는 방식으로 구현된다. backward는 quadratic intermediate를 저장하지 않고, 저장된 LSE(log-sum-exp) 정보를 바탕으로 tile-by-tile 재계산해 확률을 복원한다.

- **Technical Challenges**: 핵심 난제는 표준 attention의 online softmax처럼 단일 분포 정규화로 끝나지 않고, KL이 두 softmax 분포를 logit-difference로 결합해 running maximum 변화에 따른 rescaling을 정확히 처리해야 한다는 점이다. StreamKL은 두 분포의 row-wise KL을 다섯 개의 running scalar(m1,l1,m2,l2,acc)로 재구성해, 최대값이 갱신될 때 누적량을 보정하며 단일 패스에서 일관성을 유지한다. 또한 GPU 커널 수준에서 split-K, small-query 최적화(예: NQ=1 decode 상황), Hopper의 TMA 비동기 복사 등 구현 최적화를 결합했다.

- **Empirical Impact**: 실험에서 StreamKL은 forward에서 최대 43×, backward에서 최대 14×의 속도 향상을 baseline 대비 보였고, 특히 causal masking에서도 이 이점이 크게 유지됐다. 더 중요한 점은 attention distillation의 추가 HBM footprint를 O(NQNK)에서 O(1)로 낮춰 long-context distillation을 단일 GPU에서 가능하게 만든다는 것이다. 결과적으로 기존 시스템 병목(메모리/IO)을 구조적으로 제거해, 장문 sparse-attention LLM 학습 같은 시나리오의 실용성을 끌어올릴 의미가 크다.



### Connect the Dots: Training LLMs for Long-Lifecycle Agents with Cross-Domain Generalization Via Reinforcement Learning (https://arxiv.org/abs/2606.20002)
Comments:
          Work in progress; we will continuously update the codebase and arXiv version

- **Prior Approaches**: 기존 연구는 lifelong agent를 위해 persistent memory나 skill 세트 같은 컨텍스트 업데이트 메커니즘을 주로 사람이 설계해 왔고, LLM 자체를 CoD(Connect the Dots) 메타-역량까지 end-to-end로 후학습하는 체계는 부족했습니다. 또 standard task-by-task RL은 각 작업을 독립적으로 학습해 장기 배치에서 “이전에 얻은 환경 지식으로 다음 작업을 더 잘 푸는” 목표와 정렬이 약하다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 LLM 에이전트가 장기 실행 환경에서 관련된 작업들을 연속 수행하면서 탐색하고, 자기 경험을 바탕으로 컨텍스트를 갱신해 다음 작업 성능을 점진적으로 끌어올리는 CoD 메타-역량을 명시적으로 학습시키는 일반 프레임워크를 제안합니다. CoD-Train은 CoD-Deploy와 동일한 롤아웃 패턴(풀기-컨텍스트 업데이트 에피소드 교차)을 RL 후학습에 그대로 반영해, “문제 해결”과 “환경 맥락 학습”을 모델 가중치 수준에서 함께 길들이도록 설계했습니다.

- **Technical Challenges**: 핵심 기술 난제는 긴 상태-행동 시퀀스 안에서 solve-task와 update-context가 섞여 있을 때의 credit assignment(어떤 업데이트가 미래 보상을 키웠는지)를 안정적으로 계산하는 것입니다. 논문은 Bellman 원리로 각 에피소드가 즉시 보상뿐 아니라 미래 solve-task 보상을 동시에 최대화하도록 목표를 재정의하고, 여러 작업 보상이 들어간 시퀀스를 GRPO-style 알고리즘에 맞게 advantage 계산(에피소드 위치별 그룹화)하는 방식으로 해결합니다.

- **Empirical Impact**: 실험은 Qwen3-8B-Instruct를 CoD-Train에 적용해 FrozenLake-Obscure 같은 환경에서 “첫 작업(컨텍스트 없음) 성능 한계”와 달리, 시퀀스의 뒤 작업은 컨텍스트 갱신 덕분에 크게 향상됨을 보여줍니다(예: 1회성 성공률 상승보다 4번째 작업 성능 상승이 두드러짐). 또한 학습 도메인 내 더 어려운 인스턴스, 도메인 간(Alchemy-Random·TerminalSimulator) 평가, 그리고 Ralph-loop 설정으로의 out-of-distribution generalization 가능성을 실증하며, 장기 에이전트 학습 파이프라인에 새로운 방향을 제시합니다.



### Tri-Info: Generalizable, Interpretable Failure Prediction for VLA Models via Information Theory (https://arxiv.org/abs/2606.19998)
- **Prior Approaches**: 기존 VLA failure detector는 크게 두 계열로 나뉩니다. embedding 기반 방법은 내부 표현에 classifier를 얹어 in-domain 정확도는 높지만, 표현 공간이 모델마다 달라 out-of-architecture로는 재학습 없이는 잘 옮겨가지 못합니다. score 기반 방법(STAC 등)은 시간적 일관성 같은 단일 스칼라 신호로 실패를 경고하지만, 어떤 실패 모드인지 진단 정보는 제한적입니다.

- **Core Contribution**: 이 논문은 VLA 제어를 perception–action 폐루프의 정보 파이프라인으로 보고, 성공/실패 롤아웃이 정보이론적 시그니처에서 체계적으로 갈린다는 관찰을 바탕으로 Triple Information-theoretic(Tri-Info) 신호를 제안합니다. Tri-Info는 (1) 행동 다양성, (2) 시간적 일관성, (3) 상태 전이와의 결합(액션-상태 커플링)으로 실패를 분해해, 경고를 넘어 해석 가능한 진단 대시보드를 제공합니다. 특히 각 신호는 임베딩 좌표의 기하에 덜 의존해 아키텍처와 환경 전이 일반화에 유리하다고 주장합니다.

- **Technical Challenges**: 핵심 기술 난점은 정보이론량(엔트로피/상호정보)을 VLA의 연속·고차원 임베딩에서 안정적으로 추정하고, 이를 실시간 탐지기로 결합하는 것입니다. 논문은 sliding window로 분포를 추정하고, MI/엔트로피는 k-NN 기반 추정기(예: Kraskov 류)와 Kozachenko–Leonenko 엔트로피 추정으로 계산한 뒤 z-normalization으로 도메인 간 비교 가능성을 확보합니다. 또한 각 Tri-Info 신호를 GRU로 시간 진화를 모델링하고, success 구간에서 점수 분포가 변하는 문제는 Functional Conformal Prediction으로 time-varying threshold를 만들어 해결합니다.

- **Empirical Impact**: 여섯 개 VLA 모델과 세 개 벤치마크 환경에서 Tri-Info는 in-domain에서 강력한 베이스라인과 동등하거나 그 이상 성능을 보이며, 특히 시간 예측 타이밍까지 앞서 실패를 더 일찍 감지합니다. 더 중요한 점은 cross-model, cross-environment, sim-to-real 전이에서 임베딩·score 기반 탐지기들이 붕괴하는 상황에서도 Tri-Info가 재학습 없이 성능을 유지한다는 결과입니다. 실세계 작업에서는 이전 detector들이 chance로 무너질 때 Tri-Info가 83% 정확도까지 도달해, 안전한 배치형 embodied AI에서 “해석 가능한 실패 경고”의 실용성을 강하게 시사합니다.



### Beyond Static Endpoints: Tool Programs as an Interface for Flexible Agentic Web Services (https://arxiv.org/abs/2606.19992)
Comments:
          Accepted by ICML 2026

- **Prior Approaches**: 기존 LLM 에이전트의 웹 도구 사용은 대체로 static endpoint를 단계적으로 호출하는 방식(MWS)이다. 이 방식은 루프·조건·조인·재시도 같은 long-horizon 워크플로를 표현할 때 클라이언트가 매 단계 “다음 호출”을 다시 결정해야 해서 RTT와 추론 라운드가 절차 길이에 비례해 증가한다. 또한 부분 실패 시 재시도로 인해 state-modifying 작업이 중복될 위험이 커져 복구가 취약해진다.

- **Core Contribution**: 이 논문은 도구 의도를 “실행 가능한 tool program”으로 표현해, 다단계 서비스 상호작용을 하나의 composable 객체로 제출·위임하도록 제안한다. ToolPro는 READ(상태 보존)와 WRITE(상태 변경)를 effect type으로 명시해, 정적 엔드포인트 호출이 제공하지 못하던 turn reduction·effect-aware 실행·안전한 재실행 기반을 만든다. 결과적으로 에이전트는 다음 호출 선택이 아니라 “이 절차를 수행하라”는 프로그램을 구성해 실행을 맡길 수 있다.

- **Technical Challenges**: ToolPro를 실제로 쓰기 위한 핵심 난제는 (1) LLM이 만든 코드의 executability, (2) 실패-수정-재실행 과정에서 WRITE 중복을 막는 exactly-once 보장, (3) 프로그램 실행이 단계적 호출보다 이득인지 판단하는 adaptive consolidation이다. 저자들은 constraint-guided program construction과 Wasm 샌드박싱으로 실행 가능성을 높이고, effect-aware replay와 재시도 안전 규칙(커밋된 WRITE를 로그로 캐시해 재발행 방지)을 통해 상태 일관성을 유지한다. 마지막으로 프로파일 기반 비용 모델로 네트워크 지연·워크플로 복잡도·절차 길이에 따라 program mode와 stepwise 호출을 동적으로 선택한다.

- **Empirical Impact**: MCP-style 실제 애플리케이션(Memos, Directus, MinIO) 워크플로에서 ToolPro는 end-to-end latency를 최대 53.4% 줄이고, 클라이언트 트래픽은 최대 96.1% 감소시켰다. 개선 폭은 네트워크 latency가 높고 워크플로가 복잡할수록 더 커져, turn reduction과 effect-aware/재시도 보호 설계가 실효성이 있음을 보여준다. 장기적으로 agentic web에서 서비스 인터페이스를 “다음 호출” 중심에서 “실행 가능한 절차” 중심으로 바꾸는 유의미한 기반을 제공한다.



### The Algorithmic-Human Manager: AI, Apps, and Workers in the Indian Gig Economy (https://arxiv.org/abs/2606.19975)
Comments:
          Published by the Centre for Responsible AI (CeRAI) at IIT Madras

- **Prior Approaches**: 기존 연구는 인도와 Global South의 플랫폼 노동을 디지털 전환이나 노동시장 변화의 관점에서 다뤄왔지만, 알고리즘적 관리가 실제로 어떤 방식으로 배분·감시·평가를 수행하며 노동자 권리에 어떤 영향을 주는지에 대한 사회정의 중심의 실증 정리가 부족했다. 특히 알고리즘의 불투명성과 그로 인한 공정성·책임성 문제를 노동자 경험과 연결해 설명하는 데 한계가 있었다.

- **Core Contribution**: 이 논문은 사회정의 프레임워크로 인도 블루칼라 기그 이코노미에서 AI·디지털 기술의 ‘알고리즘적 관리’ 효과를 분석한다. 인터뷰(16명의 기그 노동자, 21명의 핵심 이해관계자) 기반의 혼합연구를 통해, 시스템이 일감 접근성을 넓히면서도 공정성·투명성·노동자 존엄을 동시에 훼손할 수 있음을 ‘이중 현실’로 제시한다.

- **Technical Challenges**: 기술적으로는 작업 배분·모니터링·평가가 자동화된 운영 방식으로 구현되면서, 설계상 불투명함이 구조화돼 있어 노동자가 기준과 결과를 이해하거나 이의제기하기 어렵다는 점이 핵심 난제로 드러났다. 또한 추가 노동이 임금으로 비례 보상되도록 설계되어 있지 않아, 불평등한 결과가 반복될 수 있는 거버넌스 공백을 지적하며 이를 ‘Algorithmic Human Manager’라는 하이브리드 통제 모델로 보완하려 한다.

- **Empirical Impact**: 실증 결과는 AI 기반 시스템이 운영 효율과 일감 기회를 늘리지만, 공정성·투명성·존엄 측면에서 중요한 결함을 동반한다는 점을 구체적으로 보여준다. 정책입안자, 플랫폼 기업, 시민사회가 인도뿐 아니라 Global South 전반에서 더 ‘공정한 AI 거버넌스 프레임워크’를 설계해야 한다는 논의의 방향성을 제공한다.



### ROSE: Benchmarking the Perception-to-Action Gap in Multimodal Models (https://arxiv.org/abs/2606.19965)
Comments:
          29 pages, 11 figures

- **Prior Approaches**: 기존 MLLM 벤치마크는 MME-Unify 같은 통합형부터 VisuLogic, OmniSpatial, VGRP-Bench 같은 시각-추론 중심까지 다양하지만, 대체로 이미지마다 질문/퍼즐 명세가 고정된다. 또 embodied/vision-language-action 평가는 실행에 가깝지만 계획·제어·환경 상호작용이 함께 묶여 ‘인식→행동 인터페이스’만을 분리하기 어렵다.

- **Core Contribution**: 이 논문은 한 장면(visual evidence)은 고정한 채 과업 컨텍스트와 출력 형식(카운트 vs region-conditioned 클릭/좌표)을 바꿔, 동일한 증거를 현재 컨텍스트에 맞는 행동으로 얼마나 신뢰성 있게 전환하는지 측정하는 ROSE를 제안한다. ROSE는 장면 내부의 암묵적 majority reference(정상 패턴)를 모델이 추론하고, 예외(exception)를 수치적 요약에서 정밀 좌표 실행으로 옮기는지를 진단적으로 본다.

- **Technical Challenges**: 핵심 난제는 ‘정확한 카운트/인식’이 ‘정확한 좌표 집합의 선택’과 자동으로 이어지지 않는다는 점인데, 모델은 같은 증거를 지역 제약(region)과 배제(exclusion) 컨텍스트에 맞춰 다시 바인딩해야 한다. 저자들은 장면 수준 결합 비교와 매칭 컨트롤(global-click bridge, exactly matched regions)을 통해 좌표 그라운딩 실패와 컨텍스트 기반 선택 실패를 분해하고, 문법 VALID와 정답 PASS를 분리해 출력 포맷 오류와 실제 행동 오류를 구분한다.

- **Empirical Impact**: 9개 최신 MLLM 평가에서 인간은 98.8% 평균 PASS를 보였지만, 대부분의 모델은 카운트 지향 과업에서 region-conditioned 액션으로 갈 때 최대 44.5%p까지 성능이 급락했다. 또한 동일 장면에서 전역 카운트를 맞춘 경우에도 region-aware action 정확도가 크게 떨어져, 문제의 병목이 단순 좌표 매핑만이 아니라 컨텍스트에 따른 선택의 비일관성(모델별)임을 보여준다. 모델 비교 진단은 일부 모델(예: GPT-5.5)이 gap을 상당 부분 줄이지만, 다른 모델(예: Qwen3.6-Plus, GLM 계열)은 VALID는 높아도 PASS가 낮은 ‘문법은 맞는데 행동은 틀리는’ 실패 모드를 드러낸다.



### Confidence Calibration for Multimodal LLMs: An Empirical Study through Medical VQA (https://arxiv.org/abs/2606.19950)
Comments:
          Accepted by MICCAI 2025

- **Prior Approaches**: 기존 confidence calibration 연구는 주로 LLM의 토큰 likelihood나 verbalized confidence를 온도 스케일링, prompting, self-consistency 등으로 보정하는 방식에 집중해 왔습니다. 다만 Medical VQA처럼 멀티모달·의료 고위험 맥락에서 모델이 내는 자신감이 실제 정확도와 어긋나는 문제는 충분히 분석·해결되지 않았습니다. 특히 의료 도메인 fine-tuning(SFT) 이후 과신이 더 지속될 수 있다는 점이 지적됩니다.

- **Core Contribution**: 이 논문은 의료 멀티모달 LLM에서 accuracy와 self-assessed confidence의 관계를 포괄적으로 분석하고, 이를 Medical VQA에 맞춘 보정 프레임워크로 연결합니다. 핵심 기여는 Multi-Strategy Fusion-Based Interrogation(MS-FBI)으로 질문에 대한 모델의 사고/모순 정보를 수집한 뒤, 보조 expert LLM 평가로 재산정된 보정 confidence를 산출하는 구조입니다. 이를 통해 의료 영역에서 더 신뢰할 수 있는 진단 보조를 목표로 합니다.

- **Technical Challenges**: 의료 과신을 단순히 한 번의 confidence 추출로 보정하기 어렵기 때문에, 모델이 스스로의 답을 재검토하게 만드는 다단계 interrogation 설계가 필요합니다. 논문은 punish 메커니즘으로 고신뢰 오답의 비용을 암시하고, challenge와 explain 전략으로 반박·논리 점검 및 단계별 설명을 유도해 내부 일관성을 흔듭니다. 이렇게 얻은 (질문, 답, 원래 confidence, rebuttal/설명) 정보를 expert LLM(예: llama3-instruct-8B)에 넣어 재보정 confidence와 해석을 동시에 산출합니다.

- **Empirical Impact**: 3개 Medical VQA 데이터셋에서 제안 방법은 Expected Calibration Error(ECE)를 평균 40% 가까이(예: LLaVA-1.5-med-7B의 44.28%→26.22%) 낮추고 AUROC도 향상시켰습니다. 또한 도메인 특성에 따라 calibration 성능이 크게 달라지며, 의료 도메인 모델은 일반 모델보다 과신이 더 두드러지고 보정 필요성이 크다는 점을 실험으로 보여줍니다. ablation과 교차모델 실험은 전략 조합과 expert 평가가 calibration 효과의 핵심이며, 복잡도가 곧 성능으로 이어지지는 않는다는 실용적 인사이트를 제공합니다.



### SIMBA: ABidirectional Retrieval Forward Simulation Framework for Modeling FY-4A GIIRS Hyperspectral Infrared Radiances Toward NWP Applications (https://arxiv.org/abs/2606.19943)
- **Prior Approaches**: 기존 딥러닝 기반 GIIRS(차가위성/정지궤도 초분광 적외) 활용 연구는 주로 복사휘도(radiance)에서 대기 프로파일로 가는 one-way retrieval에 치우쳤습니다. 이때 대기 상태→휘도 모사(forward simulation) 과정을 같은 프레임워크에서 명시적으로 모델링하지 않아, 관측공간에서의 일관성 제약이 약해 물리적 정합성이 떨어질 수 있습니다. 또한 온도·습도의 수직 구조는 압력면 간 장거리 의존을 갖는데, CNN/Transformer/LSTM/일반 state-space로 이를 효율적으로 포착하기가 쉽지 않습니다.

- **Core Contribution**: 이 논문은 FY-4A/GIIRS 초분광 적외 관측을 NWP 초기장 개선에 연결하기 위해 SIMBA를 제안합니다. SIMBA는 복사휘도→대기 프로파일 retrieval과 대기 프로파일→복사휘도 forward simulation을 하나의 통합 아키텍처에서 함께 수행하며, 두 경로를 cycle-consistency로 결합해 대기 상태공간과 관측공간의 동시 일관성을 강화합니다. 더불어 압력면 방향의 장거리 의존을 처리하도록 bidirectional Mamba state-space 모듈을 넣어 수직 결합을 더 잘 학습합니다.

- **Technical Challenges**: 핵심 난제는 (1) 고차원 채널을 가진 복사휘도와 (2) 수직 압력면을 따라 배치된 온도·비습 프로파일 사이의 비선형·장거리 매핑을 동시에 안정적으로 최적화하는 것입니다. 연구진은 LW/MW 선택 채널을 입력으로 하고, retrieval/forward 양쪽에 bidirectional Mamba와 압력면 positional encoding을 적용하며, FiLM으로 샘플별 조건을 profile decoder에 주입해 관측 특성이 수직 프로파일 생성에 반영되게 했습니다. 또한 다목적 손실(프로파일 손실, LW/MW 복사휘도 재구성 손실, cycle-consistency)을 함께 최적화해 관측공간 정합성을 끌어올렸습니다.

- **Empirical Impact**: FY-4A GIIRS 관측과 ERA5 재분석을 collocated 데이터로 학습·평가해, SIMBA는 온도/비습 retrieval은 물론 LW/MW 복사휘도 재구성에서도 대표적 딥러닝 baseline들을 전반적으로 능가했습니다. 흐림(cloudy-sky)에서의 성능이 특히 좋았고, 더 나아가 clear-sky로는 재학습 없이도 개선 효과가 이어졌는데, 이는 cycle-consistency와 bidirectional 설계가 조건 일반화에 기여했음을 시사합니다. 또한 매개변수 수와 추론 지연을 함께 분석했을 때 SIMBA는 GPU 메모리/속도 측면에서도 경쟁력이 있으며, 전 구간에서 오차가 더 안정적이고 수직 프로파일 편향 변동이 작다는 분석 결과를 제시했습니다.



### Triangular Consistency as a Universal Constraint for Learning Optical Flow (https://arxiv.org/abs/2606.19938)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 광학흐름(optical flow) 학습은 프레임을 두 장 단위로 예측하는 구조가 대부분이며, self-supervised에서는 photometric reconstruction과 forward-backward 같은 일관성 제약으로 감독 신호를 보완해 왔습니다. cycle consistency나 transformation consistency도 널리 쓰이지만, 대체로 특정 상황에 맞춘 “쌍(pair) 기반” 근사에 머물러 세 프레임 이상의 조성(composition) 원리를 포괄적으로 다루지는 않았습니다.

- **Core Contribution**: 이 논문은 optical flow의 기하학적 성질을 바탕으로, 세 프레임에서 “두 흐름을 합성해 얻은 제3의 흐름”이 직접 추정한 흐름과 일치해야 한다는 triangular consistency를 제안합니다. 이 제약은 네트워크 아키텍처, supervision 형태(지도/무지도), 데이터셋에 비의존적이며 연산 오버헤드가 거의 없는 plug-and-play 학습 구성요소로 설계됩니다.

- **Technical Challenges**: 핵심 과제는 흐름 합성이 항상 성립하는 연속 좌표변환이라는 물리적 규칙이 2D 영상에서는 occlusion/비가시성 경계에서 깨질 수 있다는 점입니다. 논문은 forward-backward consistency로 유효 마스크를 만들어 triangular residual을 robust norm으로 가중 페널티하며, 동시에 temporal chaining(3프레임)과 affine 기반 controlled augmentation(닫힌형식 좌표 변환으로 pseudo target 생성)을 같은 삼각 원리로 구현해 학습 안정성과 정확성을 함께 노립니다.

- **Empirical Impact**: 실험에서는 지도/무지도/transfer learning 전반에서 일관된 성능 향상을 보였고, 레이블 없이 self-supervised adaptation을 단 1 epoch만 수행해도 Sintel에서 최대 18.1%까지 개선되었다고 보고합니다. 또한 unsupervised 학습 파이프라인(ARFlow 기반)에 triangular consistency를 추가했을 때도 EPE와 구조적 정합성이 함께 좋아지며, out-of-domain 및 cross-dataset 일반화에서 특히 큰 이득을 보였습니다.



### Speeding up the annotation process in semantic segmentation industrial applications (https://arxiv.org/abs/2606.19934)
- **Prior Approaches**: 기존 연구는 라벨링 시간을 측정하거나, 비지도/파운데이션 기반으로 초기 마스크를 만들되 “정밀한 픽셀 단위 세그멘테이션”에서는 도메인 특수성 때문에 단독 해결이 어렵다고 봤습니다. 또한 annotation-efficient semantic segmentation은 active-learning 클릭이나 약한/거친 감독 등 초기에 인간 입력이 남는 경우가 많아, 고해상도에서의 대량 라벨링 병목을 직접 줄이기엔 한계가 있었습니다. 무엇보다 비지도 알고리즘이 실제로 라벨링 시간을 얼마나 단축하는지 정량 비교가 부족했습니다.

- **Core Contribution**: 이 논문은 고해상도 철강 마이크로구조(픽셀 단위 라벨링)에서, 비지도 알고리즘의 pre-annotation을 “scratch(처음부터 수작업)”과 같은 실험 설계로 비교해 라벨링 시간을 정량 단축하는 것을 핵심 기여로 제시합니다. 그 결과 수작업 170시간에서 37시간으로 줄여 약 78% 절감 효과를 보였고, 이는 계산 오버헤드(수 분~수십 분)가 전체 수작업 시간에 비해 미미함을 함께 확인했습니다. 동시에 MIT License로 82장의 fully annotated 대규모 공개 데이터셋 MicroSteel(영구 DOI)을 제공해 이후 벤치마크 기반 연구를 촉진합니다.

- **Technical Challenges**: 문제는 “정확도 부족을 감수할 수 있는가”가 아니라, 고해상도에서 픽셀 경계 연속성과 복잡한 소수 클래스까지 포함한 라벨을 사람 수정 단계로 넘길 만큼 일관된 pre-mask 품질을 확보하는 데 있습니다. 저자들은 다종 비지도 사전 라벨링(멀티 Otsu, superpixels, k-means, 비지도 DL, SAM)을 비교한 뒤, 이미지 타입 전반에 강건한 비지도 DL을 최종 통일 워크플로로 선택하고 Labelbox에서 전문가가 정교화하도록 설계했습니다. 특히 Majority 클래스는 과분할(over-segmentation)로 자동화하되, 전문가가 소수 클래스·복잡 경계(TiB2/TiN 등)에 집중하도록 수정 난이도를 재배치했습니다.

- **Empirical Impact**: 실험에서는 전문가 3인의 라벨링 시간을 Labelbox가 기록한 중앙값을 기준으로 비교했으며, 비지도 pre-annotation이 Type 전반에서 73~84% 수준의 시간 절감을 보였습니다(전체 평균 78%). 또한 IoU/Dice/Boundary F1/Hausdorff distance로 pre-mask와 최종 라벨의 정합성을 계량해, 소수 클래스와 경계에서 더 많은 수정이 필요함을 정량화했습니다. 더 나아가 비지도 DL이 Type III 같은 난이도 높은 경우에도 전반적으로 안정적이라 전문가 편차와 분산을 크게 낮출 수 있음을 보여주며, 산업 현장에서 라벨링 가능한 프로젝트 범위를 확장하는 실질적 의미가 있습니다.



### Spatial-Aware Reduction Framework: Towards Efficient and Faithful Visual State Space Models (https://arxiv.org/abs/2606.19932)
Comments:
          Accepted by ICML 2026

- **Prior Approaches**: Mamba의 효율은 selective scanning으로 인해 Transformer의 제곱 복잡도보다 낮은 연산을 달성하지만, 토큰 감축에는 취약하다는 문제가 제기돼 왔습니다. 특히 VMamba처럼 2D Selective Scan(SS2D)을 쓰는 변형에서 기존 pruning·merging 계열은 전역(공간 비인식) 기준으로 토큰을 줄이고 다시 2D로 매핑하는 과정에서 토큰의 공간 관계를 깨뜨려 성능이 급격히 붕괴합니다.

- **Core Contribution**: 본 논문은 “좋은 토큰을 남기는 것”보다 “모델이 전제하는 2D 토폴로지를 압축 과정에서 유지하는 것”이 성패를 좌우한다는 점을 명확히 합니다. 이를 바탕으로 STORM(Spatial-aware TOken Reduction)이라는 학습 없이도 끼워 넣을 수 있는(plug-and-play) 토큰 감축 프레임워크를 제안합니다.

- **Technical Challenges**: 기존 감소 방식은 토큰을 한 줄로 펼쳐 전역 선택·병합을 수행해 SS2D가 기대하는 2D 격자와 이웃 일관성을 위반하기 때문에 연쇄적인 정보 손실이 발생합니다. STORM은 2D 격자에서의 구조를 보존하도록 감소를 “행→열”의 두 단계 구조화 연산으로 재구성하고, windowing으로 로컬 이웃 범위를 제한해 지역 의미의 붕괴를 억제합니다.

- **Empirical Impact**: 실험에서 STORM은 training-free 설정 하에 다양한 vision Mamba 백본에서 최고 수준의 pruning 정확도를 보이며, 특히 VMamba에서는 top-1 정확도 최대 63.3% 개선(정확도 회복) 성과를 보였습니다. 또한 PlainMamba에서는 정확도 하락이 1.0%에 그쳐 ViT와 유사한 성능을 유지하며, ToMe 대비 높은 처리량과 다운스트림(검출·분할)에서도 일관된 우위를 확인했습니다.



### Co-policy: Responsive Human-Robot Co-Creation for Musical Performances (https://arxiv.org/abs/2606.19914)
- **Prior Approaches**: 기존 로봇 음악 연구는 대체로 사람(또는 시스템)이 지정한 음표를 로봇이 재생하는 playback 방식에 가깝다. 또 다른 접근으로 diffusion policy처럼 반복적 denoising을 통해 동작/행위를 생성하는데, 이런 과정은 사람과 맞물려야 하는 음악 상호작용의 저지연 요구에 불리할 수 있다. 마지막으로 단일(mean) 행동을 회귀하는 imitation learning은 다중 실행 모드가 평균화되어 표현성이 떨어질 위험이 있다.

- **Core Contribution**: Co-policy는 인간의 불완전한 음악적 시드(말/라이브 시드 음/비전)를 바탕으로, 의미(semantic intent)–음악적 보완(상보적 변주)–물리 실행(비주얼모터)을 분리해 end-to-end로 한 번에 해결하려 하지 않는 프레임워크를 제안한다. 핵심은 VLM(Qwen-vl)에 semantic anchor bank를 결합해 JSON 형태의 structured co-creation plan을 만들고, 그 계획을 제약 기반 변주 플래너가 이어받아 로봇이 복사하지 않는 complementary 응답을 생성한다는 점이다. 여기에 저지연 실행을 위해 단일 forward pass의 GMP(Gaussian-Mixture Visuomotor Policy)로 다중 모드 동작 분포를 직접 예측한다.

- **Technical Challenges**: 가장 어려운 점은 (1) 음성/시드/비전에서 음악적 의도를 해석하되, 악기 제약과 타이밍 제약을 만족하는 계획으로 안정화하는 것이다. 논문은 pre-inference semantic anchors로 VLM이 보완/변주를 “계획” 형태로 내도록 유도하고, motif consistency·harmonic validity·novelty·embodied playability 같은 제약을 플래너 단계에서 필터링한다. (2) 또한 저지연 물리 실행을 위해 반복적 denoising을 피하면서도 다중 실행 모드를 평균화하지 않는 것이 필요해, GMP를 conditional mixture-density policy로 설계해 대안적 스트라이크 모드를 분포로 유지한다.

- **Empirical Impact**: 실로봇 chime 실험과 ablation, 전문가 블라인드 평가에서 Co-policy는 의도 정합성·실행 정확도·응답 빈도 측면에서 diffusion-policy 및 제거된 baseline들보다 향상되었다. 특히 semantic anchoring의 효과는 Qwen-vl/F-Qwen 비교에서 음악적 co-creation 품질을 7.4%~13.3% 끌어올리는 등 의미 정렬에 강하게 기여함이 드러났다. 동작 실행에서는 diffusion-policy 대비 더 높은 성능(약 15%)과 더 높은 response frequency를 보였고, 모듈 기여도(예: GSA·mixture head)가 확인되면서 “물리적으로 grounded된 action 생성”이 embodied co-creation의 핵심 요구임을 실증했다.



### Measuring Biological Capabilities and Risks of AI Agents (https://arxiv.org/abs/2606.19899)
- **Prior Approaches**: 기존 접근은 AI가 생물학에 미칠 수 있는 위험을 평가하려고 했지만, 실제 연구 워크플로로 들어갈수록 평가 결과의 해석이 설계 선택에 크게 좌우된다는 점이 자주 간과돼 왔다. 특히 agentic AI(자율·협업으로 다단계 과학 작업을 수행하는 시스템) 평가에서는 정의, 실행, 채점, 문서화 방식이 불명확하거나 충분히 공개되지 않아 “무엇을 근거로 위험을 말하는가”가 흐려지는 한계가 있었다.

- **Core Contribution**: 이 논문은 AI-enabled biological risks에 관한 현재 증거를 종합하고, 이를 평가하기 위한 biological agentic evaluations를 제안하되 ‘해석에 민감하다’는 전제를 명확히 한다. 핵심 기여는 저자들이 수행한 평가 경험을 바탕으로, 평가에서 정의-설계-실행-점수화-문서화의 선택이 위험에 대한 함의(그리고 함의의 공백)를 어떻게 바꾸는지 보여주는 실무적 고려사항 묶음이다.

- **Technical Challenges**: 기술적 난제는 에이전트 평가가 복잡한 의사결정과 다단계 실행을 포함해, 같은 위험 지표라도 평가 절차에 따라 결과의 의미가 달라질 수 있다는 점이다. 저자들은 평가 대상과 범위 정의부터 실행 조건, scoring 규칙, 결과 문서화 수준까지의 설계 변수가 해석을 어떻게 왜곡/보강하는지 체계적으로 정리해, 결과를 해석할 때 필요한 ‘주의 깊은 읽기’ 틀을 제시한다.

- **Empirical Impact**: 실증적 영향은 특정 실험 결과를 단순히 확정적으로 신뢰하기보다, 평가 설계 선택이 바꿔 놓는 함의를 드러내고 정책·투자·현장 의사결정에 더 적절한 경계선을 제공하는 데 있다. 정책결정자에게는 해석상의 과신을 줄이고, 기금(공공/민간)에는 high-leverage한 AI-biology 평가 연구 분야로 자원을 유도하며, 생물보안 실무자에게는 등장하는 AI 시스템을 보다 신중하게 점검할 기준을 제공할 것으로 기대된다.



### SL-S4Wave: Self-Supervised Learning of Physiological Waveforms with Structured State Space Models (https://arxiv.org/abs/2606.19888)
- **Prior Approaches**: 의료 시계열(ECG/EEG 등)은 고샘플링·다채널·잡음·라벨 희소성 때문에 긴 구간의 의존성을 안정적으로 학습하기 어렵습니다. 기존 SSL은 CNN/RNN 중심으로 로컬 패턴은 잘 잡지만, 긴 구간의 long-range dependency를 충분히 반영하지 못하는 한계가 자주 지적됩니다. S4 계열은 장기 시퀀스 모델링에 강하지만, 생체 다채널 파형의 고유한 형태/잡음 특성에 맞춘 구조가 부족했습니다.

- **Core Contribution**: 논문은 다채널 고해상도 생체 파형의 긴 구간을 위한 self-supervised learning 프레임워크 SL-S4Wave를 제안합니다. 핵심은 contrastive learning 목표를 S4 계열 encoder에 결합해, 잡음에 불변(invariant)한 표현과 시간적 일관성을 동시에 학습하도록 설계한 점입니다. 또한 S4Wave encoder에 multiscale subkernels, residual, cross-channel 상호작용을 넣어 로컬 형태와 장기 의존성을 함께 포착합니다.

- **Technical Challenges**: 첫째, 잡음·센서 아티팩트가 많은 의료 파형에서 표현이 noise에 흔들리지 않게 만드는 것이 어려웠습니다. 연구진은 필터 기반 신호처리로 noise-reduced 보기를 만들고, 이를 contrastive positive pair로 두어 representation이 잡음 영향에서 벗어나도록 학습합니다. 둘째, 긴 구간에서 시간적 문맥을 잃지 않고 일관된 임베딩을 만들기 위해 인접 세그먼트는 가깝게(positive), 서로 다른 기록은 멀게(negative) 하는 temporal/context consistency loss를 추가했습니다.

- **Empirical Impact**: 실험에서 SL-S4Wave는 PhysioNet의 부정맥(특히 ventricular tachycardia) alarm 검출에서 supervised 및 기존 SSL 기준선을 꾸준히 능가했으며, 적은 라벨로도 높은 성능을 보이는 label efficiency를 확인했습니다. 또한 입력 길이를 늘린 장구간에서도 성능 저하가 적어 긴 시계열에서의 모델링 이점을 실증했습니다. 나아가 학습에 없던 부정맥 유형으로의 전이와 EEG 여러 태스크에서의 성능 향상을 통해 cross-domain generalization/일반화 가능성까지 확장됨을 보여줍니다.



### FFinRED: An Expert-Guided Benchmark Generation and Evaluation Framework for Financial LLM Red-Teaming (https://arxiv.org/abs/2606.19887)
- **Prior Approaches**: 기존 안전 벤치마크는 보편적 adversarial 시나리오를 주로 다루며, 정교한 전문 영역(금융)의 고유한 위협을 충분히 반영하지 못한다는 한계가 지적돼 왔다. 금융 관련 평가는 종종 “Specialized Advice”를 disclaimer(면책 문구) 확인 같은 표면적 기준으로 점검해, 실제 위해(harm)가 발생하는지 평가가 약하다는 문제를 동반한다.

- **Core Contribution**: FinRED는 금융 전문가와 함께 만든 expert-guided red-teaming 프레임워크로, 금융 규정 위반·사기 조장·시스템 신뢰 훼손 같은 도메인 특화 위험을 대상으로 LLM 안전성을 평가한다. 글로벌 표준(FATF, EU DORA 등)을 기반으로 한 two-level taxonomy와 실문서를 활용한 schema 기반 Behavioral Prompts(seeds)를 결합해, 일반 zero-shot보다 더 현실적인 위해 행동을 생성한다.

- **Technical Challenges**: 핵심 난제는 (1) 금융 위협을 실제 공격 의도와 단계까지 포함해 재현하면서도 (2) 대규모 평가가 가능하도록 일관된 판정 기준을 만드는 것이다. FinRED는 Level-1/Level-2 위험 분류와 JSON 스키마로 시나리오 생성을 구조화하고, 다섯 차원의 finance-specific judge rubric(유해성·설득력·거절 품질·사실성·회피성)를 “Unsafe if any item is Unsafe” 규칙과 함께 적용해 critical false negative를 크게 줄이도록 설계했다.

- **Empirical Impact**: 실험 결과, FinRED seeds가 단순 문맥/제로샷 생성 대비 도메인 정합성·위협 plausibility·actionability에서 모두 향상되며, ASR 기반에서도 더 취약한 실패를 더 잘 드러냈다. 또한 FinRED Judge는 기존 HarmBench rubric 대비 전문가 라벨과의 일치도를 76.92%→88.46%로 개선했고, critical false negatives를 28에서 12로 57% 감소시켰다. 더 나아가 이 프레임워크는 South Korea의 Financial Security Institute(FSI) regulatory sandbox에서 생성형 AI 보안 평가에 실제로 배치되며 의미가 확장됐다.



### PSCT-Net: Geometry-Aware Pediatric Skull CT Reconstruction via Differentiable Back-Projection and Attention-Guided Refinemen (https://arxiv.org/abs/2606.19867)
Comments:
          11pages, 5 figures

- **Prior Approaches**: 기존의 2D X-ray에서 3D CT를 만드는 방식은 보통 geometry-agnostic feature lifting에 의존해 2D 특징을 3D로 단순 투영하거나 복제한다. 이 접근은 촬영(획득) 기하를 충분히 반영하지 못해 깊이 모호성이 커지고, 그 결과 뼈 봉합선·천문 등 소아 두개골의 미세 구조 경계가 흐려지거나 잘못 생성되기 쉽다. 또한 diffusion 기반 방법은 생성 품질은 개선해도 반복적 denoising으로 인해 임상 workflow에 비싼 계산 비용이 걸린다.

- **Core Contribution**: 본 논문은 소아 두개골 X-ray→CT 재구성 문제를 “획득 기하를 네트워크에 명시적으로 넣는” 방식으로 재정의한 PSCT-Net을 제안한다. 핵심은 differentiable back-projection으로 공간적으로 충실한 volumetric prior를 만들어 깊이 모호성을 초기부터 완화하고, 이후 Attention-Guided Projection(AGP-3D)로 2D 영역과 3D 복셀 간 비선형 대응을 학습한다. 전역 문맥은 Bidirectional Mamba(BiM-3D)로 장거리 의존을 선형 복잡도로 모델링해 단일-스텝 구조의 효율도 유지한다.

- **Technical Challenges**: 2D 투영 특징을 3D로 올릴 때 가장 큰 문제는 깊이 모호성과 구조 정렬 실패로 인한 해부학적 hallucination이다. PSCT-Net은 이를 위해 differentiable back-projection을 통해 물리적 ray 경로를 따라 초기 볼륨을 구성하고, 추가로 encoder/decoder 양쪽에서 기하 일관성을 주입하는 dual conditioning(BP-C, MV3D-C)을 적용한다. 또 고정된 선형 투영의 한계를 줄이기 위해 AGP-3D가 3D voxel grid을 query로, 2D feature를 key로 삼는 attention 기반 대응 학습을 수행하며, BiM-3D로 전역 문맥을 효율적으로 결합한다.

- **Empirical Impact**: 저자들은 성인·흉부/척추 중심 공개 데이터의 한계를 보완하기 위해 소아 두개골 전용 사설 코호트 PedSkull-CT(982 scans)를 구축해 내부 검증을 수행한다. PSCT-Net은 공개 3개 벤치마크(LIDC-IDRI, CTSpine1K, CTPelvic1K)에서 diffusion 기반 대비 PSNR/품질이 개선됐고, 사설 PedSkull-CT에서도 모든 베이스라인을 상회하며 두 번째 모델 대비 PSNR 1.28 dB, SSIM 0.022, LPIPS 0.013 개선을 보였다. 또한 학습에 없던 실제 임상 X-ray에 대해서도 턱뼈 곡률·안와 소켓 깊이 같은 환자 특이 형태를 보존해 저용량 소아 CT 재구성의 임상 적용 가능성을 시사한다.



### Large Language Models Do Not Always Need Readable Languag (https://arxiv.org/abs/2606.19857)
Comments:
          23 pages, 10 figures. Preprint

- **Prior Approaches**: 기존 LLM 활용은 사람 친화적 자연어를 프롬프트/입력으로 사용하며, 다른 모델이 읽어야 하는 경우에도 구조적으로 자연어 형식을 고수해 왔습니다. 일부 연구는 압축 프롬프트나 요약을 통해 컨텍스트 비용을 줄이지만, 의미 보존을 체계적으로 검증하거나 ‘모델 전용 텍스트 표현’ 자체의 회복 가능성을 정량화하는 데는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 LLM이 생성·해석 가능한, 비표준적이지만 의미를 담는 텍스트 표현 계열을 BabelTele(“모델 네이티브 텍스트 표현”에 가까운 개념)로 정의하고 그 가능성을 실험적으로 탐색합니다. 특히 BabelTele를 고정 프로토콜이 아닌 ‘LLM의 표현 생성/해석 역량을 측정하는 진단 도구’로 제시하며, 의미가 압축된 텍스트에서도 복원되는지에 초점을 둡니다.

- **Technical Challenges**: 핵심 난제는 사람이 읽기 어렵게 비정형으로 바꿀 때도 LLM이 의미를 안정적으로 복원할 수 있는 표현을 설계·검증해야 한다는 점입니다. 논문은 가독성 진단, 모델 likelihood 기반 측정, 인간 설문, 그리고 다운스트림 태스크 평가를 함께 사용해 ‘의미 충실도’와 ‘컨텍스트 절감 효과’를 교차 검증하며, 특히 compressor-reader 페어와 태스크 세팅 의존성이 존재함을 보여줍니다.

- **Empirical Impact**: 실험 결과 BabelTele은 텍스트를 원문 대비 27.9%로 줄이면서도 의미 충실도 99.5%를 유지하는 등 높은 정보 밀도를 보였습니다. 또한 cross-model transfer, agent memory, multi-agent communication에서 컨텍스트 오버헤드를 줄이면서도 전반적으로 다운스트림 성능을 유지하되, 모델 쌍과 태스크에 따라 효과가 달라진다고 보고해 향후 LLM 시스템에서 ‘가독성-의미 복원 가능성’의 부분적 분리를 시사합니다.



### Neural Additive and Basis Models with Feature Selection and Interactions (https://arxiv.org/abs/2606.19850)
Comments:
          Accepted at PAKDD 2024. Code is available at this https URL

- **Prior Approaches**: NAM과 NBM은 GAM 구조에 neural shape function을 얹어 각 특성(또는 기저) 기여를 시각화할 수 있어 해석가능성이 강점입니다. 하지만 상호작용을 위해 두 입력(shape pair)을 늘리거나 고차원(수천 차원) 데이터에 적용하면, feature pair 수/파라미터·연산량이 폭증해 학습이 비현실적이거나 해석 가능한 shape function 수 자체가 감당되지 못했습니다.

- **Core Contribution**: 이 논문은 NAM/NBM에 feature selection layer를 도입해 학습 중에 어떤 단일 특성과 어떤 feature pair를 사용할지 가중치를 갱신하도록 만들었습니다. entmax 기반의 differentiable feature selection으로 end-to-end 학습 흐름을 유지하면서도, 최종적으로는 학습된 logit이 one-hot에 가까워져 선택된 특성만을 shape function 입력으로 고정합니다.

- **Technical Challenges**: 핵심 기술 난점은 (1) feature 선택을 미분 가능하게 만들어 SGD로 학습시키면서 (2) 온전한 end-to-end 구조와 해석가능성(“선택된 shape function만 시각화”)을 유지하는 것입니다. 저온도 τ를 annealing해 entmax 출력이 one-hot에 수렴하도록 설계하고, temperature가 충분히 낮아진 뒤에는 선택된 특성을 기반으로 연산을 생략해 계산 병목을 줄였습니다.

- **Empirical Impact**: 실험에서는 NA2M/NB2M의 계산 한계로 다루기 어려운 고차원 분류 데이터에서 NAM-FS/NBM-FS 및 NA2M-FS/NB2M-FS가 vanilla 대비 throughput(학습 속도)를 크게 개선하며, 상호작용 모델도 더 많은 차원에서 학습 가능함을 보였습니다. 또한 상태-of-the-art GAM(예: EBM, NODE-GAM) 및 여러 해석/비해석 기준과 비교해 성능이 동등하거나 더 낫고, 사전 mutual information 기반 feature selection과 비교해 학습 중 선택이 더 효과적임을 확인했습니다.



### When, Where, and How: Adaptive Binning for Tabular Self-Supervised Learning (https://arxiv.org/abs/2606.19827)
Comments:
          Accepted to MICCAI 2026

- **Prior Approaches**: 의료 임상 테이블 데이터에서 라벨은 전문가 판정 비용이 커서 self-supervised learning이 유리하지만, 기존 연구는 주로 고정 binning 기반 사전학습(pretext)에 머물렀습니다. 특히 quantile discretization을 학습 내내 단일 전역 설정으로 고정하고, numerical 값은 bin 인덱스를 점별 squared-error로 회귀하는 등 feature별 복잡도나 학습 표현을 충분히 반영하지 못했습니다. 또한 categorical-수치 혼재 구조에서 ordinal 수치의 순서성을 반영하는 감독 방식이 제한적이어서, 타입에 맞춘 일관된 pretext 설계가 어렵다는 한계가 드러났습니다.

- **Core Contribution**: 이 논문은 학습 중에 discretization을 바꾸는 Adaptive Binning을 제안해, 전역 고정 binning을 feature-wise coarse-to-fine 방식으로 대체합니다. plateau detection으로 각 수치 feature가 정체 상태에 이르면 해상도를 올리고, 표현(representation)을 함께 고려해 split(새 경계) 위치를 선택합니다. 더불어 categorical 재구성과 numerical의 ordinal 성질을 함께 다루는 heterogeneity-aware ORDinal Loss(HORD)로 타입을 통합 감독합니다.

- **Technical Challenges**: 주요 기술 과제는 (1) feature마다 수렴 속도와 난이도가 달라 전역 bin 스케줄은 비효율적이고, (2) discretization을 세분화할 때 값 공간(value-space)과 인코더 표현 공간(representation-space)이 함께 좋아지도록 split을 설계해야 하며, (3) categorical과 ordinal numerical을 하나의 pretext로 일관되게 학습시키는 것입니다. 저자들은 Feature-Wise Plateau Trigger(FPT)로 feature별로 refinement 타이밍을 분리하고, Dispersion-informed gain 기반으로 value 분산 감소와 임베딩 응집성을 동시에 만족하는 bin split만 적용합니다. 마지막으로 HORD에서 수치 feature는 soft ordinal target을 사용한 분포형 재구성(SORD)과 mean-variance 정규화를 결합하고, 범주는 cross entropy로 복원해 타입 인지 학습을 구현합니다.

- **Empirical Impact**: 공개 의료 tabular 벤치마크에서 unified evaluation 프로토콜로 실험한 결과, Adaptive Binning은 linear probing과 fine-tuning 모두에서 고정 binning 대비 일관된 성능 향상을 보였습니다. 특히 마스킹을 강하게 주지 않아도(심지어 no-mask에서도) 이득이 유지되어, 개선의 핵심이 입력 잡음보다 학습-연동형 feature-wise discretization 정제(When–Where–How 결합)에 있음을 시사합니다. 또한 dataset별 discretization 튜닝 없이도 기본 설정이 견고하게 동작해 재현성과 임상 적용 관점의 실용성을 높였으며, 동일 프로토콜의 medical tabular SSL benchmark까지 함께 제공해 후속 연구의 비교 기준을 정립했습니다.



### CSWinUNETR: Segmentation of Thin Anatomical Structures in Medical Images (https://arxiv.org/abs/2606.19824)
Comments:
          Accepted at MICCAI 2026

- **Prior Approaches**: 기존 얇고 휘어진 해부학적 구조(망막 혈관, 뇌혈관, 얼굴 주름) 분할은 낮은 대비, 잦은 단절, 심한 클래스 불균형 때문에 예측이 조각나고 미세 분기가 복원되지 않는 문제가 컸다. 최근 CNN/Transformer가 성능을 올렸지만, 과도한 다운샘플링이나 태스크 전용 수정·손실·후처리에 의존하는 경우가 많아 다른 데이터로의 전이가 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 2D/3D 모두에 공용으로 쓰는 얇은 구조 분할 백본 CSWinUNETR를 제안한다. CSWin의 cross-shaped stripe self-attention으로 주축 방향의 장거리 문맥을 효율적으로 전파하고, detail-enhanced multi-scale self-attention으로 고주파 단서를 보존하며, SDSConv로 곡선 궤적을 따라 국소 증거를 안정적으로 집계한다.

- **Technical Challenges**: 핵심 기술 도전은 (1) 단절된 얇은 구조를 연결하기 위한 장거리 정보 모델링과 (2) 분기/말단의 미세 디테일을 유지하는 고해상도 보존, (3) 곡선 경로를 따르는 동안 로컬 오류가 누적되지 않게 하는 곡률 정렬이었다. 저자들은 cyclic shift로 stripe 간 정보 교환을 늘리고 MS-MHSA에서 고주파 detail branch 및 다중 해상도 문맥을 융합했으며, SDSConv에서는 sparse control points로부터 snake-like 커널을 “누적 드리프트 없이” 병렬 생성해 궤적 정합성을 높였다.

- **Empirical Impact**: CSWinUNETR는 안과(FIVES), 신경혈관(TopCoW), 피부(FFHQ-Wrinkle) 등 4개 벤치마크에서 기존 SOTA 대비 일관된 개선을 보였고, 대부분 지표에서 통계적으로 유의한 성능 향상이 보고됐다. 특히 task-specific post-processing이나 topology-aware loss 없이도 미세 분기와 말단을 더 잘 복원하고 비표적 영역의 오탐을 줄여, 임상적/후속 분석(센터라인 기반 길이·분지·곡률) 요구에 대응하는 범용 백본으로 의미가 크다.



### CREDENCE: Claim Reduction for Decomposition & Enhanced Credibility -- Semantic Metrics and Convergence Analysis (https://arxiv.org/abs/2606.19819)
Comments:
          40 pages, 6 figures, 19 tables. Submitted to Language Resources and Evaluation

- **Prior Approaches**: 기존 자동 사실검증 파이프라인은 복합 문장을 원자(atomic) 청구로 분해한 뒤 검증하는 접근이 주류였지만, 분해 품질 평가는 주로 token-overlap 기반 Jaccard(Soft-F1Jac) 같은 지표에 의존했다. 이 방식은 “X is big”과 “X is large”처럼 의미는 같은 paraphrase를 낮게 점수화해, 사람 판단과 자동 점수 간 괴리를 만들었다. 또한 rule 기반 repair와 LLM self-repair를 반복 적용하는 루프는 제안됐지만, 종료(termination)와 개선이 단조(monotone)로 보장되는지에 대한 형식적 분석이 부족했다.

- **Core Contribution**: 이 논문은 Credence라는 ‘청구 분해 및 평가 프레임워크’로 (1) 패러프레이즈를 과도하게 벌점 주던 평가 취약점과 (2) repair 루프의 종료/단조성 문제를 동시에 다룬다. 핵심은 Semantic-F1(의미 유사도 기반 재정의)와, rule repair 및 LLM self-repair 각각에 대한 수렴/비수렴 성질을 정리한 verified repair 설계에 있다. 더불어 도메인 간 일반화를 측정하는 3종 벤치마크와, 여러 decomposer를 함께 비교하는 다중 모델 평가 체계를 제시한다.

- **Technical Challenges**: Semantic-F1을 설계할 때는 토큰 중첩이 아니라 문장 의미를 반영하면서도, decomposer가 만든 원자 단위의 구조적 불일치에 강인한 매칭(average-max pooling)을 구현해야 했다. 이를 위해 BGE-large cosine similarity를 사용해 Jaccard식 벌점 문제를 해결했고, precision/recall 모두 greedy max 유사도 정렬을 채택해 Hungarian 1-to-1 강제의 과도한 불이익을 피했다. 또 repair의 종료성을 보이려면 디펜던시 파싱의 “분해 경계”를 상태로 삼아 rule repair는 단조 감소 및 유한 종료를 증명하고, LLM self-repair는 비단조(non-monotone) 반례와 함께 early-exit guard 같은 안전장치가 필요함을 정리했다.

- **Empirical Impact**: 실험에서 Semantic-F1은 Jaccard-F1 대비 +15~32pp(평균 +25pp)까지 향상되어, 패러프레이즈 정렬이 실제 검증 성능에 유리하게 반영됨을 보여줬다. 정량적으로는 SocialClaimSplit과 WikiSplitBench에서 EPR이 0.94~1.00 수준이며, 더 어려운 뉴스 도메인인 ClaimDecompBench에서는 베이스 EPR이 0.824까지 내려가도 평가 일관성을 유지했다. 또한 rule-repair는 Atomicity Violation Rate(AVR)를 베이스 대비 47~100% 줄이면서 fidelity를 떨어뜨리지 않아, “안정적 원자성 보정”이 실증적으로 확인됐다.



### Uncertainty-Aware Reward Modeling for Stable RLHF (https://arxiv.org/abs/2606.19818)
- **Prior Approaches**: RLHF는 선호(preference) 데이터로 보상 모델(RM)을 학습한 뒤, GRPO 같은 정책 최적화에서 RM이 예측하는 보상을 최대화하며 정렬을 수행한다. 하지만 기존 RM은 입력마다 단일 스칼라를 내는 결정론적 점추정이라, 예측이 신뢰 가능한지(불확실성) 신호를 제공하지 못한다. 또 GRPO는 군집 내 보상을 균일하게 표준화(advantage 표준화)해 신뢰도 낮은 이상치가 학습 업데이트에 과도한 영향력을 갖기 쉽다.

- **Core Contribution**: 이 논문은 Uncertainty-Aware Reward Modeling (UARM)을 제안해 보상 모델에 보정된 불확실성을 부여하고, 이를 GRPO의 advantage 계산에 반영한다. UARM은 양자회귀 기반의 quantile conformal prediction으로 입력별 예측구간을 만들고, 그 구간 폭을 “신뢰도”로 해석해 잘못된 reward hacking 가능성을 줄이는 데 초점을 둔다. 결론적으로, “보상 예측이 틀렸을 때의 영향”을 학습 과정에서 구조적으로 억제한다.

- **Technical Challenges**: 핵심 난제는 (1) RM이 내는 값의 신뢰도를 온라인에서 계산 가능하게 만들고, (2) GRPO의 군집 표준화가 불확실한 샘플을 증폭하지 않도록 advantage를 다시 설계하는 것이다. 이를 위해 UARM은 quantile regression으로 다수의 conditional quantile을 학습하고, held-out calibration로 conformal 방식의 coverage를 맞춰 예측구간 폭이 per-sample 신뢰도를 반영하도록 만든다. 온라인 단계에서는 예측구간 폭을 관측 잡음(heteroscedastic observation noise)으로 보고 분산 분해를 통해 신뢰도 가중치를 만들며, 그 결과 GRPO advantage를 불확실한 롤아웃에 대해 다운웨이트한다.

- **Empirical Impact**: HelpSteer, UltraFeedback, PKU-SafeRLHF 세 데이터셋에서 UARM은 보상 모델의 calibration을 유의미하게 개선했고, 대표적인 reward hacking을 감소시키는 결과를 보였다. 또한 표준 GRPO와 uncertainty-agnostic baseline보다 downstream 정렬 품질을 향상시켰다. 즉, 불확실성을 단순 추정이 아니라 정책 최적화의 안정성 장치로 통합했을 때 실효성이 입증된 셈이다.



### ParaScale: Scale-Calibrated Camera-Motion Transfer via a Gauge-Invariant Parallax Number (https://arxiv.org/abs/2606.19805)
Comments:
          Accepted by SCA2026(poster)

- **Prior Approaches**: 기존 참조-기반 카메라 제어는 참조 영상에서 추출한 카메라 궤적(회전+이동)을 그대로 생성 모델에 주입하는 방식이 일반적입니다. 그러나 참조와 타깃은 장면 스케일이 수십~수만 배까지 달라질 수 있어, 이동 성분을 그대로 재생하면 너무 작거나(거의 안 움직임) 너무 과장되며(프레임을 벗어남) 제어가 실패합니다. 일부 방법은 학습 시 스케일 정규화나 attention 공간 치환을 시도하지만, 추론 시점에 임의의 참조-타깃 사이를 ‘스케일 캘리브레이션’하는 개념적 근거를 직접 다루지는 못했습니다.

- **Core Contribution**: 이 논문은 ‘번역(translation)에 의한 이미지 모션’이 ||T||/Z 비율로 스케일에 좌우된다는 기하학 사실을 정리하고, 단안(monocular) 궤적은 깊이 스케일 게이지 때문에 절대 이동량 자체가 의미 없다고 지적합니다. 대신 깊이-스케일 게이지 불변량인 Parallax Number Pi = ||Delta T|| / Zbar를 정의하고, 스케일-faithful(스케일-충실) 전이는 궤적 전체가 아니라 이 Pi를 보존해야 한다는 것을 증명합니다. 이를 위해 ParaScale(Parallax Number 기반 보정 모듈)은 회전은 그대로 두고, 프레임별로 Pi만 타깃의 자체 깊이에 재현하여 학습 없이 어떤 pose-conditioned generator에도 끼워 넣을 수 있게 합니다.

- **Technical Challenges**: 핵심 기술 난제는 ‘단안 SfM/깊이 추정’이 절대 스케일을 알 수 없는데도, 생성에서 실제로 느껴지는 번역 패럴랙스를 정확히 맞춰야 한다는 점입니다. 논문은 translation-로 인한 felt motion이 Z에 의해 스케일되므로, raw T를 쓰면 참조-타깃 스케일 불일치가 그대로 오차로 증폭된다는 로그-스케일 관점의 정량식을 제시합니다. 해결책은 참조에서 Pi를 스케일 무관하게 읽어내고, 타깃의 깊이 통계 Zbar로 프레임별 translational gain을 계산해 Pi를 그대로 강제하는 것으로, 추가 학습 없이 pose injection 파이프라인 사이에 한 단계만 삽입합니다.

- **Empirical Impact**: 실험에서는 타깃 장면 스케일을 tabletop부터 human/room, architectural, cosmic 수준까지 4자릿수 차이로 나누고, 여러 pose-conditioned 백본에서 동일 효과를 확인했습니다. ParaScale은 Raw transfer 대비 PCE(Parallax Consistency Error)를 3배 이상 줄이며, 회전 오차는 거의 변하지 않으면서 이동/패럴랙스의 스케일 불일치만 정확히 교정됨을 보여줍니다. 또한 Similiarity-aligned TransErr가 놓치는 상수 스케일 미스매치까지 PCE가 드러내며, 생성 품질 저하 없이 identity line에 가깝게 패럴랙스가 유지되는 점에서 실무적 의미가 큽니다.



### Policy-aware Vector Search: A Vision for Fine Grained Access Control in Vector Databases (https://arxiv.org/abs/2606.19803)
Comments:
          Accepted at SeQureDB 26, Sigmod 2026

- **Prior Approaches**: 기존 벡터 데이터베이스는 RAG 같은 보안 민감 워크로드에서 요구되는 Fine-grained Access Control(FGAC)을 충분히 지원하지 못한다. 메타데이터 필터링으로 접근을 제어하려 해도 pre-filtering은 ANN 인덱스를 덜 활용할 수 있고, post-filtering은 선택도가 높을 때 대량의 후보를 버리며 recall과 지연(latency)을 동시에 악화시킬 수 있다. 또 공간 분할 기반 인덱스나 수정이 필요한 hybrid 방식은 오버헤드와 인덱스 변경 부담이 크고, 특정 경우 효율이 떨어지는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 벡터 데이터베이스에서 정책 인식형 검색(policy-aware vector search)을 구현하기 위해 FGAC 정책 모델과 enforcement 문제를 체계적으로 정식화한다. 특히 ABAC 기반의 3-tuple(객체 제약, 주체 제약, allow/deny 액션)로 벡터·메타데이터에 대한 정책을 표현하고, 사용자별로 적용되는 정책 집합과 정책을 만족하는 결과 집합의 관계를 문제 정의로 명확히 한다. 나아가 FGAC을 지키면서도 ANN 검색의 recall과 지연 사이의 긴장을 줄이기 위해 여러 enforcement 전략을 비교하고, 어떤 상황에서 어떤 전략을 선택해야 하는지에 대한 프레임을 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 정책 정확성(무단 접근 방지)과 ANN 근사 탐색의 recall, 그리고 지연을 동시에 만족시키는 실행 계획을 고르는 것이다. 논문은 correctness(정합성·보안성·최대성) 개념을 벡터 검색의 의미적 근사 특성에 맞게 재해석해야 하며, recall은 전략 선택을 위한 워크로드 수준 품질 제약으로 취급하되 권한 검증은 별도로 강제되어야 함을 강조한다. 이를 위해 pre-filtering, naïve post-filtering, iterative post-filtering, parallel post-filtering을 비교하고, 특히 iterative scan의 비용 급증을 완화하기 위해 서로 다른 그래프 영역을 동시에 탐색하는 parallel post-filtering을 제안한다.

- **Empirical Impact**: 실험은 pgvector( PostgreSQL + vector search) 위에서 277만 개 아카이브 문서(제목·초록 임베딩, 나머지를 메타데이터)로 수행했으며, 정책 선택도(selectivity) 수준을 바꿔 recall과 latency 트레이드오프를 측정했다. 결과적으로 pre-filtering은 전반적으로 높은 recall을 보인 반면 post-filtering은 정책과 쿼리가 강하게 상관된 경우를 제외하면 recall이 크게 떨어지는 경향이 나타났다. 중간 선택도 구간에서는 parallel post-filtering이 naïve post-filtering보다 경쟁력 있는 recall을 제공하면서 iterative 기반보다 검색 라운드를 줄여 지연도 낮추는 모습을 보여, 보안 보장과 성능을 함께 추구하는 방향의 실용적 근거를 제시한다.



### Improving End-to-End Speech Recognition for Dysarthric Speech through In-Domain Data Augmentation (https://arxiv.org/abs/2606.19797)
- **Prior Approaches**: 기존 DyASR 연구는 (1) tempo/pitch 등 비선형 음성 변형으로 데이터 부족을 완화하거나, (2) 도메인 적응/잡음 제거로 전형 음성에 가깝게 만드는 방식, (3) MFCC·i-vector·raw spectrum 같은 특징/모델 설계로 성능을 끌어올리는 접근이 주를 이뤘습니다. 또한 Wav2Vec2 같은 self-supervised learning(SSL) 모델을 feature 추출기나 adversarial augmentation과 결합하는 연구도 있었지만, “전통적 데이터 증강을 Wav2Vec2 fine-tuning에 체계적으로 적용”한 시도는 부족했습니다.

- **Core Contribution**: 이 논문은 dysarthric speech의 severity(난이도)별로 speaking-rate modification(SRM), pitch modification(PM), formant modification(FM), vocal tract Length Perturbation(VTLP)를 fine-tuning 단계에 맞춰 설계하고, 각 severity에 대해 최적 modification factor를 찾는 실험 프레임을 제안합니다. 특히 severity별로 Wav2Vec2를 개별 fine-tuning한 뒤 다른 severity로 테스트해 cross-severity 일반화까지 평가합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (a) dysarthric 데이터가 적어 SSL fine-tuning에 필요한 유효 다양성을 만들기 어렵고, (b) 증강이 severity에 따라 서로 다른 방향으로 영향을 줄 수 있다는 점입니다. 저자들은 각 증강을 실제 음성 생성 관점(SRM의 시간축 변형, PM의 반음 기반 재구성, FM의 LPC formant warping, VTLP의 스펙트럼 축 재매핑)에서 구현하고, training set에만 증강을 적용한 뒤 WER로 modification factor 민감도를 확인하는 방식으로 해결했습니다.

- **Empirical Impact**: TORGO(15명, 약 15시간)에서 baseline(증강 없음) 대비, 증강이 severity 전이 성능을 유의미하게 개선함을 보였습니다. 특히 low·medium에 대해서는 SRM(s=0.8)이 각각 9.02%, 38.11% WER을 달성해 상대 개선 30.02%, 16.64%를 기록했고, high는 PM(τ=0.8)이 55.15%로 상대 개선 15.47%를 보였습니다. 결과적으로 “severity별로 올바른 전통적 증강과 강도 선택이 DyASR 성능을 실질적으로 끌어올린다”는 경험적 근거를 제공해, 데이터 희소 문제 해결의 실전 지침으로 의미가 있습니다.



### Agentic Electronic Design Automation: A Handoff Perspectiv (https://arxiv.org/abs/2606.19795)
- **Prior Approaches**: 기존 ML4EDA와 LLM 기반 EDA 연구는 배치/타이밍 예측/지시문 탐색처럼 단일 단계의 성능 향상에 집중하거나, 특정 도구 출력의 생성·수정을 잘하는 데 초점을 맞춰 왔습니다. 하지만 출력이 다음 단계에서 “어떤 수용 조건(acceptance conditions)”을 만족해야 재사용 가능한지, 그리고 그 근거(provenance)와 맥락이 기계적으로 어떻게 보존되는지는 명시적으로 다루기 어려웠습니다. 그 결과 PDK·툴 버전·제약(SDC)·체크포인트가 바뀌면 로컬에서는 맞아 보이던 산출물이 다운스트림에서 깨질 수 있습니다.

- **Core Contribution**: 이 논문은 agentic EDA를 “handoff validity(핸드오프 타당성)” 문제로 재정의하고, 전달된 객체가 수신자가 요구하는 수용 조건을 만족하며 다운스트림 사용에 충분한 맥락·증거·출처를 담아야 한다고 제안합니다. 또한 이 타당성이 유지돼야 하는 범위를 기준으로 Stage-Bound, Flow-Bound, Organization-Bound의 3분류 체계를 제시합니다. 마지막으로 이러한 요구를 구현하기 위한 5-layer EDA Agent Communication Protocol(EACP)을 연구 의제로 제안합니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 수신자가 실제로 판단하는 acceptance 조건을 계약 형태로 명시하고, (2) 실행 가능성/정량 성능/검증 판정뿐 아니라 증거의 출처와 적용 범위를 함께 전송·검증하는 것입니다. 논문은 경계별로 handoff contracts(수용 조건+가정+필요 증거), handoff objects(무엇을 넘기는지: 아티팩트·상태·지식·결정), coordination mechanisms(생성-수정, 복구-재개, 검색-응답 등)을 분석해 요구사항을 체계화합니다. 이를 토대로 EACP에서 에이전트 탐색·메시지·툴 호출·워크플로 오케스트레이션·보안 및 IP 프로토콜 레이어를 분리해 “계약 이행”을 가능하게 하는 방향을 제시합니다.

- **Empirical Impact**: 2026년 5월 1일까지 공개/수용된 82개 agentic EDA 시스템을 문헌 기반으로 정리하고, 각 시스템을 경계 범위(스테이지/플로우/조직)와 handoff 요구 차원에서 비교 분석합니다. 그 결과, 기존 분류가 “무엇을 하는가”에 비해 “언제/왜 다음 단계가 받아들일 수 있는가”를 설명하지 못한다는 공백을 구체적인 손절(계약·객체·조정·미해결 질문) 구조로 드러냈습니다. 분야적으로는 에이전트형 EDA의 신뢰성(trustworthiness)을 평가·설계하는 공통 언어와 연구 아젠다를 제공한다는 점에서 의미가 큽니다.



### Systematic Study of Dysarthric Speech Recognition: Spectral Features and Acoustic Models (https://arxiv.org/abs/2606.19793)
- **Prior Approaches**: 기존 DyASR(난독성/발음장애 음성 자동인식) 연구는 데이터 부족과 발화의 강한 변동성(화자 내/화자 간) 문제를 중심으로 하이브리드 DNN/HMM의 discriminative training, 생성형/템포 변조 데이터, denoising 기반 품질 향상, 그리고 pre-trained ASR의 domain adaptation 등을 시도해 왔다. 다만 핵심은 여전히 “어떤 음향 특징(feature)과 acoustic model 조합이 dysarthric speech에 최적인가”를 체계적으로 비교·정리한 연구가 부족했다.

- **Core Contribution**: 이 논문은 TORGO 데이터로 DyASR을 대상으로 다양한 스펙트럴 특징(FBANKs, MFCCs, PLPCCs)과 pitch 특징 결합을 4종 acoustic model(HMM-GMM, SGMM, DNN, TDNN-LSTM, 그리고 F-TDNN)에 걸쳐 비교하는 체계적 탐색을 제시한다. 특히 sentence 인식에서 pitch 특징이 성능을 끌어올리는 패턴을 보여주고, F-TDNN의 학습에서 chunk 간 overlapping frames 수까지 조정해 기준선 대비 개선을 만든다.

- **Technical Challenges**: dysarthric speech는 조음 정밀도 저하로 인한 발화 음향 변동이 커서, 같은 특징이라도 모델에 따라 성능 경향이 다르게 나타나는 점이 기술적 난제였다. 논문은 특징 차원/구성(FBANKs 40, MFCC/PLPCC 차원 설정)과 pitch 결합 여부를 모델별로 분리 평가하고, chain 기반 F-TDNN 학습에서 chunk overlap(0~40프레임)를 바꿔 발화 속도·발음 변동을 흡수하도록 설계를 최적화했다.

- **Empirical Impact**: 실험은 isolated word와 sentence 두 태스크로 수행됐고, pitch 결합이 특히 sentence 인식에서 유효함이 확인됐다. TORGO에서 F-TDNN 기준으로 isolated word는 relative 4.65%, sentence는 relative 4.63% 성능 개선을 달성했으며, overlap 최적값(난독 화자 중심 20프레임)은 이전 연구를 feature 세트 전반에서 능가한다. 더 나아가 control 화자에서도 유사하게 WER 감소가 관찰되어, “모델·태스크별로 특징을 맞춰야 한다”는 설계 원칙이 실증적으로 지지된다.



### Cross-Dataset, Age, and Gender Generalization: A Comprehensive Analysis of Fine-Tuning Strategies for Low-Resource Children's ASR (https://arxiv.org/abs/2606.19791)
- **Prior Approaches**: 기존 음성 인식(ASR)은 성인 대규모 코퍼스 기반 사전학습 후 파인튜닝으로 아동 성능을 보정해 왔습니다. 하지만 연령·성별·발화 길이·데이터셋(억양/녹음환경) 차이에 따른 일반화 문제는 충분히 세분 분석되지 못했고, 특히 cross-dataset 전이가 취약하다는 점이 남아 있었습니다. 또한 self-supervised learning(SSL) 모델은 강력한 표현을 주지만, 아동 내 하위집단별 편향과 변동성에 대한 영향은 덜 탐구되었습니다.

- **Core Contribution**: 이 논문은 PFSTAR와 CMU Kids에서 Wav2Vec2, HuBERT, WavLM을 대상으로 연령별·성별별 파인튜닝을 low-resource 환경에서 체계적으로 비교합니다. 그 결과, 더 어린 화자의 음향적 다양성이 학습에 유리하게 작동해 ‘어린→큰’ 일반화가 더 잘 되고, ‘성별 편향(남성 음성에 유리)’도 하위집단 파인튜닝으로 드러난 뒤 성별 균형 데이터로 완화될 수 있음을 보여줍니다. 더불어 단문 발화에서 WER이 커지는 경향과 같은 아동 ASR의 구조적 한계를 함께 제시합니다.

- **Technical Challenges**: 핵심 도전은 SSL의 사전학습 편향이 아동 음성의 하위집단(연령·성별)과 데이터셋 조건 차이에서 어떻게 드러나는지 분해해 관찰하는 것입니다. 이를 위해 PFSTAR/CMU Kids를 연령·성별로 쪼개 파인튜닝하고, 각 서브셋 및 교차 데이터셋에서 WER을 일관된 설정(CTC loss, greedy decoding, 고정 학습률 등)으로 비교했습니다. 또한 cross-dataset 전이 시 억양·어휘·녹음환경 불일치가 성능 하락으로 이어짐을 반복적으로 확인해 원인-결과를 데이터 기반으로 연결했습니다.

- **Empirical Impact**: 실험은 zero-shot에서 이미 모델 성능이 아동 코퍼스에서 크게 흔들리며, 특히 단문인 CMU Kids에서 WER이 더 높게 나타남을 보여줍니다. 파인튜닝은 전반적인 격차를 크게 줄이되, 연령은 ‘젊은 쪽 파인튜닝이 오래된 쪽으로 더 잘 일반화’, 성별은 ‘남성 편향이 존재하나 성별 균형 학습으로 완화’라는 구체적 패턴을 확인했습니다. cross-corpus에서는 British/American 억양과 녹음조건 불일치로 WER이 유의미하게 증가해, 아동을 대표하는 다양성 있는 사전학습/데이터 전략의 필요성을 실증적으로 강화한다는 점에서 의미가 큽니다.



### Towards Engineering Scaling Laws with Pretraining Data Composition (https://arxiv.org/abs/2606.19781)
- **Prior Approaches**: 신경 스케일링 법칙은 손실이 연산량, 모델 크기, 데이터 크기에 대해 멱법칙(power law)으로 개선된다는 점을 경험적으로 정리하며, 언어·비전에서 compute-optimal(최적 연산 배분)을 예측하는 도구로 자리잡아 왔다. 파티클 물리에서도 제트 분류를 대상으로 데이터 크기 스케일링 지수 등이 power law로 관측되었지만, 기존 연구는 주로 scratch 학습이나 모델·데이터 크기 자체의 변화에 초점을 두었다. 또한 pretraining이 스케일링을 바꿀 수 있다는 신호는 있었으나, 물리 시뮬레이터로 데이터를 ‘설계’할 수 있다는 고유한 조건에서 스케일링을 엔지니어링하는 방식은 체계적으로 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 물리 고정밀 시뮬레이터로 합성 데이터를 저렴하게 만들 수 있다는 장점을 활용해, pretraining 데이터의 ‘구성’(다양성·다운스트림 정합성)이 compute-optimal 스케일링 지수를 어떻게 바꾸는지 실증한다. JetClass-II로 사전학습한 뒤 JetClass(10-class 제트 분류)로 fine-tuning할 때, QCD 중심 pretraining은 scratch와 유사한 스케일링을 보이지만 BSM 레조넌스 기원을 포함한 pretraining은 데이터 쪽 지출을 더 선호하는 형태로 지수(데이터 vs 모델)를 재배치한다. 즉, “더 큰 모델”이 아니라 “더 많은 데이터”가 유리한 스케일링 레짐으로 스케일링 자체를 공학적으로 이동시킬 수 있음을 보여준다.

- **Technical Challenges**: 핵심 난관은 pretraining 데이터 구성 변화가 성능뿐 아니라 스케일링 지수까지 일관되게 이동하는지, 그리고 그 지수가 어떤 추출 방법에서도 견고한지 검증하는 것이다. 연구진은 point cloud를 처리하는 transformer로 JetClass-II 각 부분집합(QCD, QCD+res2p, QCD+res34p, QCD+모든 레조넌스)에서 사전학습한 뒤 단일-epoch 미세조정(single-epoch protocol)으로 비교 가능성을 확보하고, IsoFLOP 프로파일(로그-로그 공간의 포물선 최소점)과 대안적 parametric fit을 함께 사용해 지수를 추정한다. 또한 모델 크기 범위를 수천 파라미터부터 수천만 파라미터까지 넓게 스팬하고, warmup 및 토큰 수를 맞춰 ‘구성’이 관측값의 차이를 주도하도록 실험을 설계했다.

- **Empirical Impact**: 실험 결과, scratch 학습의 경우 compute-optimal 지수는 데이터와 모델에 대해 대략 동등한 a≈0.52, b≈0.48 수준으로 나타났다. 반면 BSM 레조넌스를 포함한 pretraining(QCD+res2p+res34p)에서는 데이터 지수 b가 약 0.78까지 상승해, 고정된 예산에서 데이터(FLOPs→데이터) 비중을 훨씬 더 빠르게 늘리는 배분이 최적인 레짐으로 이동함을 보였다. 즉, pretraining 데이터 구성을 잘하면 파운데이션 모델을 더 작은 모델로도 만들 수 있고, 절약된 compute를 시뮬레이터로 생성한 추가 데이터에 투입하는 전략이 스케일링 관점에서 유리하다는 실질적 가이드를 제공한다.



### Data Standards for Humanoid Robotics: The Missing Infrastructure for Physical AI (https://arxiv.org/abs/2606.19769)
- **Prior Approaches**: 기존 접근은 모델과 하드웨어 성능 향상에 집중했지만, 로봇 경험이 다른 몸체·과제·현장으로 누적되려면 데이터가 먼저 재사용 가능해야 한다. 그러나 현재는 데이터가 센서 구성, 좌표계, 타이밍, 태스크 정의, 라벨 규칙, 품질 기준이 달라 서로 해석이 어렵고, 축적해도 공유 역량으로 이어지지 않는 문제가 크다.

- **Core Contribution**: 이 논문은 Humanoid robot datasets에 대한 ISO/WD 26264-1(일반 요구사항)과 같은 국제 표준화가 Physical AI의 핵심 인프라가 되어야 한다고 주장한다. 단순한 파일 포맷이 아니라, 신체(embodiment)·행동·과제·장면·실행 추적·결과를 한 물리 에피소드로 묶어 “데이터가 여행(share/trace/reuse)”할 수 있게 만드는 구조를 제안한다.

- **Technical Challenges**: 문제의 핵심 기술 난관은 (1) 신체 기반 데이터가 분리되면 의미를 잃는 ‘해석 가능성’과 (2) 멀티모달 스트림이 타이밍·공간 정합·캘리브레이션·단위·동기화 가정이 명시되지 않으면 ‘물리적 일관성’이 깨지는 점이다. 이를 위해 표준은 에피소드 스키마, 물리 일관성 기록(시간 기준·동기화·좌표변환·캘리브레이션·단위 등 가정 검증 가능), 라이프사이클 기록(프로비넌스·품질·버전·권리·사용 조건·평가 디스크립터)을 데이터 패스포트 형태로 요구한다.

- **Empirical Impact**: 논문은 더 많은 데이터 수집이 곧바로 누적 역량을 만들지 못하며, 비용·데이터 사일로·평가 공백으로 인해 ‘비누적(non-cumulative) 데이터’가 생긴다고 정리한다. 표준은 서로 다른 연구/조직이 동일한 기준으로 비교·검증·재사용할 수 있게 해 중복 수집을 줄이고, 장기적으로 누적 가능한 휴머노이드 능력을 확산시키는 데 의미가 있다.



### SafeSpec: Fast and Safe LLM via Dynamic Reflective Sampling (https://arxiv.org/abs/2606.19755)
- **Prior Approaches**: Speculative inference는 draft 모델이 후보를 만들고 target 모델이 검증해 지연을 줄이지만, 자체적으로 안전 보장은 제공하지 못합니다. 기존 안전 방어는 (1) 프롬프트/접미사 기반처럼 파이프라인에 얹을 수는 있어도 공격 성공률(ASR)을 충분히 낮추지 못하거나, (2) decoding-time 개입처럼 draft–verify 정렬을 깨거나 연산 오버헤드가 커 가속 이점을 희생하는 문제가 있었습니다.

- **Core Contribution**: 본 논문은 SafeSpec이라는 안전-인지형 speculative inference 프레임워크를 제안해, 검증 단계에 위험 평가를 직접 통합합니다. target 모델에 가벼운 latent safety head를 붙여 품질과 안전을 단일 forward pass에서 함께 판정하고, unsafe가 감지되면 즉시 종료가 아니라 rollback과 safety-guided reflective multi-sampling으로 안전한 연속을 복구합니다.

- **Technical Challenges**: 핵심 난제는 (a) speculative decoding의 가속 구조를 깨지 않으면서도 (b) 잡음이 있는 단계별 safety 신호를 안정적으로 활용해 (c) jailbreak를 유도하는 생성 궤적 분포 변화를 효과적으로 되돌리는 것입니다. SafeSpec은 safety score/quality score 임계값 기반의 계층적 스크리닝으로 빠른 위험 감지를 하고, rollback으로 safe 연속이 존재할 확률을 높인 뒤 reflection prompt로 문맥 관성을 끊어 멀티샘플이 안전 궤적을 찾도록 유도합니다.

- **Empirical Impact**: 실험에서 SafeSpec은 여러 모델/공격 벤치마크에서 safety-efficiency의 Pareto frontier를 개선했으며, Qwen3-32B에서는 공격 성공률을 15% 낮추면서 benign 작업에서 2.06x 속도 향상을 유지했습니다. 동시에 과도한 over-refusal도 크게 줄여 XSTest에서 SecDecoding 대비 현저히 낮은 거절률을 보였고, GSM8K/MATH/GPQA 같은 추론 벤치마크에서 정확도 저하가 거의 없음을 확인했습니다.



### Temporal Self-Imitation Learning (https://arxiv.org/abs/2606.19752)
- **Prior Approaches**: 기존 장기 로봇 조작 강화학습은 누적 reward에 주로 의존하며, dense reward shaping과 sparse task-success 보상을 혼합해 탐색을 돕는 방식이 많았습니다. 하지만 dense 보상은 빠르지 않더라도 중간 상태를 반복해 높은 리턴을 만들 수 있어, ‘효율적인 성공’과 ‘보상에 분산된 느린 성공’을 구분하지 못한다는 한계가 있습니다. 또한 rare하게 발견되는 빠른 성공 행동은 on-policy 갱신 과정에서 분포 변화나 최적화 불안정 때문에 쉽게 사라질 수 있습니다.

- **Core Contribution**: 이 논문은 시간적 효율(temporal efficiency) 자체를 reward shaping을 넘어서는 self-supervision 신호로 활용하자고 제안합니다. Temporal Self-Imitation Learning (TSIL)은 학습 중 발견된 ‘시간이 짧은 성공 궤적’을 골라 재사용 가능한 감독으로 바꾸며, 구성(configuration) 조건에 따른 adaptive temporal target을 점진적으로 조여 줍니다. 동시에 효율 가중 self-imitation을 통해 빠른 성공 행동을 replay하고 지속적으로 다시 학습에 포함시킵니다.

- **Technical Challenges**: 핵심 난제는 (1) 글로벌한 시간 목표를 쓰면 쉬운 설정에서 배운 효율을 어려운 설정에 강제하게 되는 문제와, (2) 느리게 성공한 궤적이나 reward에 분산된 패턴이 self-imitation에 섞여 학습을 왜곡하는 문제입니다. TSIL은 학습 중 성공 궤적의 완료 시간을 구성별 D(ϕ)로 업데이트해 adaptive temporal target을 만들고, 이를 관측과 reward shaping(성공지점에 temporal bonus)을 통해 부드럽게 학습 압력으로 전환합니다. 더 나아가 replay에서는 ‘빠른 성공’만 우선 저장하고, 자기학습 압력은 advantage 기반 게이팅과 효율 가중치로 조절해 느린/덜 신뢰할 만한 궤적 재생을 완화합니다.

- **Empirical Impact**: 15개 장기 조작(관절형 물체 상호작용, 삽입, 도구 사용, 운반, 접촉-rich 조작 포함)에서 TSIL은 success rate와 AUC 같은 학습 성능뿐 아니라 성공 완료 시간(behavioral efficiency), 빠른 성공 행동의 재방문, 불안정한 학습 조건에 대한 robustness까지 일관되게 개선했습니다. 특히 reward-engineering과 adaptive temporal target만 쓴 변형 대비, ‘효율 기반 궤적 채굴+재생’의 결합이 전반적 트레이드오프에서 가장 좋은 결과를 보였습니다. 결과적으로 성공의 시간 구조가 사람이 설계한 reward shaping에 덜 의존하면서도 확장 가능한 self-supervisory signal이 될 수 있음을 실험적으로 뒷받침합니다.



### Manifold Bandits: Bayesian Curriculum Learning over the Latent Geometry of Large Language Models (https://arxiv.org/abs/2606.19750)
Comments:
          Webpage: this https URL

- **Prior Approaches**: 기존 LLM 강화학습의 group-relative optimization은 한 프롬프트에서 여러 롤아웃의 보상 변동을 통해 정책 기울기를 얻는데, 보상 변동이 없으면 신호가 붕괴해 학습 효율이 급격히 떨어집니다. 이를 개선하려는 적응형 샘플링/커리큘럼은 대체로 각 프롬프트를 독립적인 arm으로 보고(밴딧) 중간 난이도 같은 단일 기준을 맞추거나, Dynamic Sampling처럼 보상 분산이 0이 아닌 프롬프트만 골라내려다 계산 오버헤드를 치르는 경향이 있습니다. 또한 문제 간 구조(유형 이질성, 표현 공간에서의 유사성)를 제대로 반영하지 못해 탐색이 비효율적이거나, 다양성(만ifold 커버리지)과 평가 관련성 사이의 균형을 놓칠 수 있습니다.

- **Core Contribution**: 이 논문은 문제 샘플링을 ‘엔도제너스(non-stationary)한 manifold-structured bandit 문제’로 재정의합니다. 모델이 업데이트되면 각 프롬프트의 보상 분포가 바뀌고, 그 변화는 샘플링 선택이 만든 피드백 루프로 진화하며, 프롬프트들은 모델의 잠재 표현 공간에서 서로 연관돼 정보가 전이됩니다. 이를 바탕으로 Bayesian Manifold Curriculum(BMC)을 제안해, Latent Task Tree(모델 임베딩 기반 계층 트리) 위에서 Bayesian 의사결정으로 학습 신호를 극대화하면서도 태스크 manifold 커버리지를 유지합니다.

- **Technical Challenges**: 핵심 어려움은 (1) 모델 내부 표현을 이용해 데이터의 구조를 학습 과정에서 재사용할 수 있는 형태로 근사하고, (2) endogenously 변하는 보상 환경에서 프롬프트별 ‘학습 신호’에 대한 확률적 믿음을 온라인으로 갱신하며, (3) 트리 상에서 하향(top-down) 선택과 상향(bottom-up) 정보 전파를 안정적으로 결합하는 것입니다. 논문은 policy 중간층 임베딩으로 Latent Task Tree를 재귀적으로 구성하고(국소 차원/연결성을 보는 Chart Test 포함), BMC에서는 프롬프트 단위 Gaussian(분산+불확실성) 믿음을 surprise 기반 Bayesian 필터로 갱신한 뒤, 정밀도 가중 집계에 subtree-level heterogeneity 보정을 더해 이질적 영역이 과신(과잉 평균)되지 않게 합니다. 결과적으로 트리의 관련성에 따라 관측이 인접 프롬프트의 선택 확률까지 연동되어, 정보 공유를 계산 효율적으로 수행합니다.

- **Empirical Impact**: 수학 추론(RLVR 기반) 설정에서 DAPO-Math-17K와 Qwen3-4B-Base/8B-Base를 사용해, 샘플링 전략을 생산성(productivity: 보상 분산/효과적 샘플링 비율), 다양성(diversity: latent manifold 커버리지/정보 공유), 유틸리티(utility: 평가 관련성) 축으로 진단합니다. 관찰에 따르면 reward variance와 effective ratio가 높을수록 학습 속도(pass@1)가 빨라지며, BMC는 Dynamic Sampling이 주는 학습 속도 이점을 상당 부분 재현하면서도 wall-clock 시간은 훨씬 덜 소모하는 경향을 보입니다. 또한 단순 난이도(구조 제거)나 고정 트리(업데이트 제거)만으로는 productivity–diversity–utility의 비대칭적 트레이드오프를 충분히 해결하기 어려워, ‘난이도만’으로는 강한 downstream 성능을 보장하기 어렵다는 메시지를 실증적으로 강화합니다.



### Beyond Uniform Forgetting: A Study of Sequential Direct Preference Optimization Across Preference Settings (https://arxiv.org/abs/2606.19744)
Comments:
          Submitted to EMNLP 2026

- **Prior Approaches**: LLM 선호 정렬은 RLHF나 DPO 같은 offline 대조 학습으로 여러 행동 목표(도움됨/해로움/안전/정직 등)를 단계적으로 다루지만, 새 목표를 추가할 때 이전 목표가 얼마나 보존되는지는 불명확했습니다. 기존 연구는 대체로 특정 포스트트레이닝에서의 평균 성능 하락(혹은 이전 지식의 망각/전이)을 보고해 왔고, preference pair처럼 열린 응답 비교에서는 변화가 ‘상대 선호 마진’ 단위로 분산될 수 있어 분석이 거칠다는 한계가 있었습니다. 또한 망각을 원인별(직접 gradient conflict vs 분포 드리프트/신호 불균형/파라미터 이동)로 설명하려는 실증 근거가 제한적이었습니다.

- **Core Contribution**: 이 논문은 sequential DPO(Direct Preference Optimisation)를 두 단계로 나눠 학습할 때, 후속 목표 학습이 이전 목표의 선호를 균일하게 망각하는지 또는 목표 간 관계에 따라 안정/재분배/긍정적 전이를 보이는지 체계적으로 확인합니다. HH-RLHF, HelpSteer2, PKU-SafeRLHF, UltraFeedback의 네 가지 preference 설정(분포 갈등, 다속성 상호작용, 강한 안전 신호, 호환되는 응답 품질)을 두 순서로 비교해, “한 가지 forgetting 패턴”이 성립하지 않음을 보입니다. 더 나아가 aggregate 지표가 가리는 pair-level 이질성을 분해하는 분석틀을 제안합니다.

- **Technical Challenges**: 핵심 난제는 이전 단계에서 학습된 선호가 단계 2에서 어떻게, 그리고 얼마나 ‘부분적으로’ 변하는지 정밀 측정하는 것입니다. 이를 위해 각 stage 이후 모든 목표를 고정 base-model 기준으로 평가해 DPO-relative reward margin과 relative preference accuracy를 일관된 척도로 비교하고, preference pair마다 길이 정규화 policy margin 변화를 추적한 뒤 quartile decomposition으로 변화를 구간별로 분해합니다. 원인 규명에서는 direct gradient opposition 가설을 gradient cosine similarity와 LoRA 어댑터 이동(파라미터 공간 방향성) 진단으로 직접 점검하지만, 두 진단 모두에서 stage2 업데이트가 이전 목표에 대해 거의 직교에 가깝다는 결과를 제시합니다.

- **Empirical Impact**: 실험 결과 sequential DPO는 이전 목표에 대해 일관된 catastrophic forgetting을 만들지 않고, 목표 관계와 신호 강도, 학습 순서에 따라 부분 열화, 안정, pair-level 재분배, 긍정적 전이가 폭넓게 나타났습니다(예: PKU-SafeRLHF에서는 높은 안전 신호가 다른 단계 후에도 선호를 잘 유지). 특히 pair-level/ quartile 분석은 “평균은 비슷해 보이지만 일부 구간의 confidence 높은 pair가 크게 망각되거나 오히려 강화될 수 있음”을 보여 주어, aggregate 지표만으로는 정책 변화의 구조를 놓칠 수 있음을 강조합니다. 기계적 진단에서는 stage2 gradient와 LoRA 업데이트가 이전 목표와 near-orthogonal하여 direct gradient conflict가 주된 동인이라는 증거가 약하다는 점을 제시함으로써, 향후 sequential alignment 파이프라인 설계 시 objective compatibility와 signal strength를 고려해야 한다는 실무적 함의를 제공합니다.



### QueryGaussian: Scalable and Training-Free Open-Vocabulary 3D Instance Retrieva (https://arxiv.org/abs/2606.19733)
Comments:
          8 pages, 4 figures, 6 tables. Accepted to the 2026 IEEE International Conference on Systems, Man, and Cybernetics (SMC 2026)

- **Prior Approaches**: 기존 open-vocabulary 3D instance retrieval은 주로 scene-level embedding 방식에 의존합니다. 텍스트-의미를 먼저 장면 전체의 3D Gaussian에 “증류”해야 해서 전처리/재학습 비용이 크고, 장면 복잡도가 커질수록 메모리가 선형으로 증가해 city-scale에서 OOM이 자주 발생합니다. 또한 2D-3D 매핑이 장면 전체 임베딩에 결합되어 있어 vocabulary나 장면이 바뀌면 다시 구축해야 하는 제약도 있습니다.

- **Core Contribution**: QueryGaussian은 training-free로 작동하는 on-demand 인스턴스 검색 프레임워크를 제안합니다. 장면 전체를 의미 특징으로 압축하는 대신, 미리 학습된 2D vision 모델로 텍스트에 해당하는 2D 마스크를 찾고 이를 3D로 “최대 기여도(max-weight)” 방식으로 직접 lift합니다. 즉 의미 이해와 기하 표현을 분리해, 기존 3DGS 체크포인트 그대로부터 zero-shot 검색을 가능하게 합니다.

- **Technical Challenges**: 핵심 난제는 2D 마스크를 3D Gaussian으로 올릴 때의 projection ambiguity와 occlusion/복원 잡음입니다. QueryGaussian은 rasterization 중 생성되는 maximum-weight map으로 픽셀을 가장 강하게 기여하는 Gaussian에 매핑해 반투명 아티팩트 등 불필요 후보를 줄입니다. 이어서 multi-view temporal fusion으로 시점 간 일관성이 낮은 outlier를 누르고, 이후 multi-stage density clustering(단계적 DBSCAN 변형)으로 인스턴스 경계를 더 정밀하게 다듬습니다.

- **Empirical Impact**: 실험에서 QueryGaussian은 소형 실내와 대형 실외(각각 수백만~수천만/1,000만+ Gaussians)에서 경쟁 또는 우수한 mIoU를 보였고, 특히 대형 설정에서는 scene-level embedding 계열이 24GB GPU에서 OOM으로 실패한 반면 동작을 유지했습니다. 효율 측면에서도 GPU 메모리 사용량을 70% 이상 줄이고, end-to-end 추론을 최대 180x 가속해 소비자급 하드웨어로 city-scale 인스턴스 검색을 실현합니다. 또한 검색 결과(마스크/3D 중심)를 LLM 에이전트의 인스턴스 테이블로 연결해 3D question-answering 같은 응용까지 시연했습니다.



### VOiLA: Vectorized Online Planning with Learned Diffusion Model for POMDP Agents (https://arxiv.org/abs/2606.19729)
Comments:
          Submitted to the 2026 International Symposium of Robotics Research (ISRR)

- **Prior Approaches**: POMDP는 불확실성 추론의 표준 프레임워크지만, 실제 로봇 적용은 전이/관측 모델을 ‘정확히’ 만드는 일이 가장 큰 병목이었습니다. 온라인 근사 POMDP 풀이는 모델이 사전에 잘 준비돼 있어야 하고, RL 기반 접근은 명시적 모델을 줄이지만 대개 대규모 데이터와 학습 시간이 필요합니다. 검색+학습을 결합한 방법들도 GPU 같은 병렬 자원을 충분히 활용하지 못해 실시간성 제약이 커졌습니다.

- **Core Contribution**: VOiLA는 작업(task)-비특이적인 POMDP 모델을 학습해 온라인 계획에 바로 쓰는 프레임워크입니다. 전이 샘플러와 관측 샘플러는 conditional diffusion model로 생성 분포를 배우고, belief 업데이트에는 particle 기반 가중치 계산을 위한 observation-likelihood 모델을 추가로 학습합니다. 또한 diffusion 기반 샘플러를 distillation로 압축해, VOPP(Vectorized Online POMDP Planner)와 결합함으로써 연산 효율을 끌어올립니다.

- **Technical Challenges**: 핵심 어려움은 (1) 다봉/비선형인 로봇 전이와 (2) 고차원 관측(예: 이미지) 분포를 생성적으로 모델링하되, (3) 온라인 POMDP 탐색 중에 사용할 수 있을 만큼 샘플링 비용을 낮추는 것입니다. VOiLA는 diffusion을 먼저 고표현력으로 학습한 뒤, ODE 기반 diffusion sampler를 feedforward 생성기로 distill해 샘플링 비용을 크게 절감합니다. 연속 관측에 대해서는 VOPP에 progressive widening을 벡터화해 확장하고, belief 업데이트는 정규화 상수 부담을 줄이기 위해 unnormalized likelihood를 InfoNCE류의 contrastive 학습으로 상대 가중치 형태로 구성합니다.

- **Empirical Impact**: 시뮬레이션 실험에서 distillation 전략은 샘플링 비용을 최대 거의 3자릿수(orders of magnitude)까지 줄여, learned generative POMDP 모델의 온라인 사용을 실용 범위로 끌어내렸습니다. 3개 벤치마크 평가에서 VOiLA는 Recurrent Soft Actor Critic과 비슷하거나 더 나은 성능을 보이면서도 학습 데이터는 10% 미만만 사용했고, 학습에 없던 환경 설정으로의 일반화도 더 우수했습니다. 물리 로봇 실험(쿼드루피드 타겟 파인딩)에서는 시뮬레이션으로 학습한 모델만으로 online POMDP 계획을 수행해 10/10 성공을 보고했습니다.



### Bidirectional Tutoring for Developmental Motor Learning in Robots: Co-Developed Interaction Dynamics Support Stable Learning (https://arxiv.org/abs/2606.19728)
Comments:
          16 pages, 14 figures

- **Prior Approaches**: 로봇의 발달적 모터 스킬 학습은 LfD나 imitation learning처럼 튜터가 제공하는 데모를 로봇이 일방적으로 받아들이는 unidirectional 방식으로 많이 다뤄졌다. 특히 kinesthetic teaching에서도 로봇이 튜터의 물리적 개입을 ‘통과적으로’ 받는 설정이 흔해, 튜터가 로봇의 현재 행동과 내적 동역학을 반영해 궤적을 공동 구성한다는 점이 약했다. 그 결과 성공적인 데모가 나오더라도 센서모터 패턴의 일관성이 부족해 단계별 학습이 흔들리거나 일반화가 제한될 수 있다는 우려가 제기된다.

- **Core Contribution**: 이 논문은 튜터-로봇 상호작용을 bidirectional tutoring으로 재정의하고, 로봇의 과거 경험이 prior constraint처럼 작동해 공동 개발된 궤적의 ‘행동적 응집성(behavioral coherence)’을 유지함으로써 단계별 학습과 일반화를 돕는다고 가설을 세운다. 이를 unidirectional 방식과 직접 비교해, bidirectional일 때 더 일관된 행동이 형성되고 튜터 의존이 점차 줄어드는 발달 스캐폴딩 효과가 나타난다는 그림을 제시한다. 또한 free-energy principle 기반의 PV-RNN을 generative replay와 결합해 단일 튜터 에피소드만으로도 stage-wise 학습이 안정적으로 진행되도록 구현한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) bidirectional 접촉 상호작용에서 생기는 잡음과 의도적 충돌을 실시간 추론/제어에 통합하고, (2) 단계마다 새 데모(단일 튜터 에피소드)를 추가하면서도 이전에 학습한 센서모터 패턴을 catastrophic forgetting 없이 보존하는 것이다. 논문은 FEP의 variational free energy를 최소화하는 PV-RNN의 온라인 inference(error regression)로 튜터 개입에 맞춰 동역학을 즉시 조정하고, generative replay로 과거의 재생 시퀀스를 저장 없이 샘플링해 망각을 완화한다. 특히 stage-wise 설정에서 ‘너무 많으면 과거에 편향, 너무 적으면 망각’ 문제를 피하려고 재생 시퀀드를 큰 버퍼에서 매 epoch 소량만 무작위 선택하는 설계를 사용한다.

- **Empirical Impact**: Torobo(휴머노이드)로 물체 조작 태스크를 수행하며, (i) 실제 인간 튜터 환경과 (ii) AI 튜터가 adaptive intervention으로 개입하는 보다 통제된 환경의 두 실험을 수행했다. 두 설정 모두에서 bidirectional tutoring이 unidirectional보다 일관된 튜터 궤적을 만들었고, 단계가 진행될수록 자율 성공률이 상승하며 튜터 의존이 점차 감소했다. 또한 학습 단계 전반에서 학습 손실(variational free energy)이 bidirectional에서 대체로 더 낮게 나타나, 공동 구성된 궤적이 기존 센서모터 동역학과 더 잘 정합됨을 시사한다.



### NRITYAM: Language Models Meet Art and Heritage of Danc (https://arxiv.org/abs/2606.19727)
Comments:
          18 pages, 12 figures, in ECML_PKDD'26

- **Prior Approaches**: 기존 문화 VQA/문화 추론 벤치마크는 단일 언어 중심이거나(영어 편중), 춤보다는 모션 인식 같은 다른 작업에 초점이 맞춰지는 경우가 많았습니다. 또한 일부는 지역 범위가 좁거나(예: 동남아) 문화 주제가 넓더라도 ‘전통 춤’의 언어·지역 맥락을 깊게 다루는 대규모 QA 세트는 부족했습니다. 그 결과 모델이 지역 특수성을 이해하는지 정밀하게 측정하기 어려웠습니다.

- **Core Contribution**: NRITYAM은 전통 춤의 문화적 이해 능력을 평가하기 위한 다국어·다문화 벤치마크로, 12개 언어에서 총 9,260개의 QA를 제공합니다. 텍스트와 이미지 두 모달리티를 모두 다루며, 질문은 역사 기반·규칙 기반·시나리오 기반으로 체계화되어 전통 공연의 맥락 추론을 시험합니다. 데이터는 현지 무용가·원어민과의 협업으로 지역에 맞는 질문을 직접 작성·검증해 문화적 근거를 강화했습니다.

- **Technical Challenges**: 전통 춤 지식은 언어(용어, 서술 방식)와 시각 단서(의상·동작·의례 맥락)가 결합돼 있어, 단순 번역이나 일반 상식만으로는 정답 판단이 어렵습니다. 저자들은 각 질문이 국가별 전통 춤과 정합적인지 먼저 걸러내고, 36명의 국가별 전문가가 QA를 제작·번역한 뒤 교차검증과 편향/민감도 리뷰로 품질을 통제했습니다. 또한 평가를 zero-shot(온도 0, MCQ 정답 후보 중 확률 최대)로 통일해 모델 비교의 공정성을 확보했습니다.

- **Empirical Impact**: 실험에서 frontier LLM은 대체로 소형 모델보다 우수했지만, 전 언어에서 고르게 잘하진 못했으며 특히 저자원 언어(예: Māori, Amharic, Arabic)에서 성능 저하가 두드러졌습니다. 멀티모달 모델은 시나리오 추론에서 강세를 보였으나, 역사·규칙처럼 문화적으로 고정된 근거를 요구하는 범주에서는 여전히 병목이 관측됐습니다. 결과적으로 NRITYAM은 전통 예술 맥락을 ‘문화적으로 민감하게’ 이해하는 능력을 계량화하며, 향후 데이터 보강과 저자원 문화 지식 전이 연구의 기준선 역할을 할 것으로 기대됩니다.



### Library-Aware Doubles and Iterative Repair for Large Language Model-Generated Unit Tests in OpenSIL Firmwar (https://arxiv.org/abs/2606.19725)
Comments:
          20 pages, 10 figures

- **Prior Approaches**: 기존 UT 자동 생성은 LLM이 초안을 만들고, 컴파일/실행 실패 로그로 반복 수정하는 generate–compile–run–repair 루프를 많이 사용한다. 하지만 firmware C(특히 EDK II)에서는 헤더·심볼·INF 메타데이터·패키지 의존성이 조금만 어긋나도 링크가 쉽게 깨지고, 커버리지가 높아도 정답을 보장하기 어려운 test oracle 문제가 남는다. 또한 openSIL처럼 XFER table·ip2ip 같은 디스패치/함수 포인터 의존과 deep double, 얕은 shallow stub의 구분이 필요한 환경에선 단순한 UT 초안 생성만으로는 반복 수정을 감당하기 어렵다.

- **Core Contribution**: 이 논문은 openSIL C 코드베이스용 UT를 “초안 작성→EDK II 빌드/디스패치→빌드 리페어→LCOV 기반 커버리지 정제”로 이어지는 다중 에이전트(LLM 기반) 파이프라인으로 자동화한다. 핵심은 라이브러리 인지(library-aware) 방식으로 stubs/mocks/fakes를 재사용하거나 최소 동작의 double을 합성하되, 템플릿/심볼 위생 규칙을 위반하지 않도록 강제한다. 그 위에 컴파일·링커 로그와 라인 커버리지 피드백을 동시에 소비하는 “컴파일–디스패치 수리 루프”를 얹어 UT를 빌드 가능/디스패치 가능/커버리지 개선 단계로 점진 전환한다.

- **Technical Challenges**: 도전 과제는 (1) EDK II 제약 하에서 UT가 컴파일·링크되도록 include/INF/패키지 의존성과 심볼 충돌을 동시에 맞추는 것, (2) FUT가 호출하는 cross-module 경로는 deep double로 유도하되 같은 소스 파일의 부수 호출은 shallow stub으로 안전하게 처리하는 것, (3) 실행 후 LCOV에서 miss된 라인을 실제로 더 타격하도록 입력과 설정을 정제하는 것이다. 논문은 Chroma 기반 VDB에서 관련 함수·기존 UT·검증된 테스트 더블을 찾아 재사용하고, build 로그와 디스패치 결과로 “타겟 최소 수정”만 수행하는 상태 그래프(11단계)로 이를 해결한다. 라인 커버리지 기반 정제는 LCA만 쓰는 변형과 vector-database retrieval까지 함께 쓰는 변형을 분리해 효과를 비교한다.

- **Empirical Impact**: 평가 결과, 76개 FUT(functions under test) 중 73개에 대해 컴파일 가능한 UT를 생성했다. 라인 커버리지 가이드 없이(또는 retrieval augmentation 없이) 평균 line coverage는 73.9%까지 도달했으며, 48개 부분 집합에서는 라인 커버리지 가이던스만으로 98.8% 평균을, LCA+vector database retrieval 조합에서는 94.7% 평균을 각각 달성했다. 전반적으로 constrained firmware 환경에서 UT 생성 효율과 커버리지 향상이 유의미하게 개선되고, 수동 디버깅/수정 반복 비용을 줄일 수 있음을 실증한다.



### OnDeFog: Online Decision Transformer under Frame Dropping (https://arxiv.org/abs/2606.19721)
Comments:
          Accepted to PRICAI 2025

- **Prior Approaches**: 실세계 강화학습(RL)에서는 통신 지연이나 센서 고장으로 frame dropping이 발생해, 에이전트가 누락된 상태와 보상 정보를 받지 못하는 문제가 커진다. 기존에는 Soft Actor-Critic(SAC)·DQN처럼 지연을 다루거나 최대 지연/누락 구간을 가정하는 방식이 있었지만, 실제로는 그 한계를 넘으면 성능이 떨어질 수 있다. DeFog는 Decision Transformer 기반으로 random frame dropping을 학습에 포함해 강건함을 높였으나 offline learning 특성상 학습 데이터에 충분히 없는 상태에서는 일반화가 약하고, 고품질 데이터 수집 비용이 크다는 한계가 있다.

- **Core Contribution**: 이 논문은 DeFog의 frame dropping 대응 메커니즘을 온라인 Decision Transformer(ODT)로 옮긴 OnDeFog을 제안한다. 즉, 온라인 상호작용을 통해 정책을 갱신하면서도 train-time frame dropping과 drop span embedding을 결합해 frame dropping 환경에서의 학습 안정성과 강건성을 함께 노린다. 또한 DeFog가 offline 데이터 의존으로 겪던 일반화/데이터 비용 문제를 ODT의 온라인 탐색으로 완화하는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 난제는 frame dropping으로 누락된 관측 패턴이 학습 신호와 시퀀스 조건에 어떻게 반영돼야 하는지, 그리고 ODT의 탐색 기반 학습이 dropping에서도 안정적으로 수렴하는지에 있다. 저자들은 (1) pre-training 단계에서 train-time frame dropping으로 누락을 시뮬레이션하고, (2) 연속된 누락 길이(D)를 계산해 drop span embedding으로 상태/returns-to-go 토큰에 명시적으로 주입해 모델이 missing 패턴을 직접 학습하도록 했다. 비교 실험에서 freeze-trunk fine-tuning은 OnDeFog에서 효과가 제한적이어서, 대체로 위 두 구성요소의 기여가 핵심으로 나타난다.

- **Empirical Impact**: 실험은 Gym-MuJoCo의 Hopper·Walker2d·HalfCheetah에서 frame drop rate 0~0.9 범위로 수행했으며, OnDeFog는 ODT보다 high-drop-rate에서 성능 저하가 훨씬 완만해 frame dropping 강건함을 확인했다. 또한 D4RL medium-replay처럼 low-reward 데이터 비중이 큰 설정에서 OnDeFog는 DeFog를 자주 앞서는데, 온라인 탐색으로 더 높은 보상의 궤적을 재수집해 replay buffer 품질을 개선한 영향으로 설명한다. 다만 탐색이 부족하거나 action space가 큰 경우(예: HalfCheetah)에는 성능 격차가 줄지 않았고, low reward가 극단적으로 많은 데이터에서는 한계가 관찰돼 향후 offline↔online 전환 자동화가 과제로 제시된다.



### AURA: Adaptive Uncertainty-aware Refinement for LLM-as-a-Judge Auditing (https://arxiv.org/abs/2606.19714)
- **Prior Approaches**: 기존 LLM-as-a-judge 평가는 MT-Bench, Chatbot Arena처럼 pairwise 비교를 통해 스케일을 확보하지만, LLM 심사(판정)는 응답 순서, verbosity, formatting 같은 표면 단서에 민감해 사람 선호와 불일치할 수 있습니다. 이를 보정하려는 감사(auditing) 파이프라인은 보통 사전에 신뢰 가능한 예시 부분집합이나 clean supervision 신호(인간 라벨, 휴리스틱 필터링, 강한 judge 출력)를 고정된 전제로 둡니다. 하지만 LLM 평가에서는 초기 분할이 judge bias를 그대로 상속하고, 인간 검증은 예산상 부족해 규모에 맞는 안정적 집단을 만들기 어렵습니다.

- **Core Contribution**: AURA는 LLM-심사 pairwise 결정을 ‘언제 사람과 일치하는지’로 감사하면서, 제한된 human verification 하에서 신뢰/불확실 구간을 반복적으로 갱신하는 adaptive uncertainty-aware refinement 프레임워크를 제안합니다. 핵심은 LLM 판정에 대한 trust를 잠재(latent) 변수로 보고, 증거가 쌓일수록 사람 일치 확률 추정을 점진적으로 정교화한다는 점입니다. 또한 불확실한 비교부터 인간 검증을 우선 배치해, 전체 라벨 비용 대비 감사를 개선하도록 설계했습니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 사람-일치 여부는 확인 전까지 잠재이고, (2) 기존 PU류 가정처럼 ‘정답 집단이 고정’되어 있지 않으며, (3) 고정된 초기 앵커가 편향을 증폭할 위험이 있다는 점입니다. AURA는 비교용 feature를 만들고, human-consistency scorer를 학습한 뒤 trust logit(모델 증거, 이웃(local) 합치, anchor 기반 지지, 이전 라운드 inflow)를 통해 soft responsibility(사람 일치 확률)를 업데이트합니다. 이어서 uncertain set에 대해서는 전체 최적 수송을 피하는 conservative transport 기반의 sparse propagation으로 신뢰 증거를 가까운 비교에 제한적으로 전달하고, 불확실하고 정보성이 큰 항목만 인간에게 추가 질의합니다.

- **Empirical Impact**: 논문은 synthetic 데이터에서 회복(recovery)을, real pairwise LLM-정답(또는 비교) 데이터에서 사람 일치 감사 성능과 robustness, 그리고 annotation efficiency(인간 검증 예산 절감)를 포괄적으로 평가합니다. 특히 noisy initialization(초기 앵커/분할이 흔들릴 때)에도 refinement가 편향을 고정된 채로 증폭하지 않고 점진적으로 안정화되는 경향을 보이며, 불확실 비교에 인간 노력을 집중하는 전략의 효과를 입증합니다. 결과적으로 AURA는 LLM 판정 교정/검증이 필요한 상황에서 ‘추가 라벨을 어디에 쓸지’까지 포함한 실용적 감사를 제공한다는 의미가 있습니다.



### FineREX: Fine-Tuned NER-RE for Human Smuggling Knowledge Graphs (https://arxiv.org/abs/2606.19710)
Comments:
          Code available at this https URL

- **Prior Approaches**: 인신밀수(human smuggling) 수사에선 법원 기록에서 NER-RE를 뽑아 지식그래프(KG)로 만드는 다단계 파이프라인이 주로 쓰였다. 대표적으로 LINK-KG는 coreference resolution을 위해 문서 리라이팅을 수행한 뒤, 다시 NER-RE/GraphRAG 추출을 반복해 최종 그래프를 만든다. 하지만 일반-purpose LLM에 의존해 도메인별 개체·관계 스키마를 정확히 맞추기 어렵고, 추가 추론 단계가 잡음(hallucination)·중복 추출·비용을 키우는 문제가 있었다.

- **Core Contribution**: FineREX는 법원 문서용 NER-RE를 중심으로, fine-tuned LLaMA 3.1 8B 기반 단일 추출 패스와 coreference 매핑을 그래프 통합에 직접 연결하는 “스트림라인” 파이프라인을 제안한다. 문서 리라이팅과 중복 NER-RE 패스 없이, 1차 추출 결과의 coreference cache를 곧바로 노드 통합에 사용한다. 또한 인신밀수 도메인에 맞춘 NER-RE 스키마를 정의하고, 이를 학습시키기 위한 수작업 주석 데이터(512개 청크)를 구축해 성능 격차를 만든다.

- **Technical Challenges**: 핵심 난관은 도메인 용어와 스키마가 강하게 제약된 법률 문맥에서, 잡음 정보까지 섞지 않으면서 개체/관계를 정확히 추출하는 것이다. FineREX는 delimiter 포맷과 타깃 엔터티 정의를 포함한 LINK-KG 계열 system prompt를 고정하고, QLoRA로 LLaMA 3.1 8B를 인신밀수 NER-RE에 파라미터-efficient fine-tuning해 recall과 관계 강도(score) 예측까지 같이 끌어올린다. 이후 coreference 단계에서 별칭을 canonical 이름으로 정규화하고, 그래프 통합 시 다중 청크에서의 type은 majority vote로 안정화하며 관계 강도는 평균(중복 설명 제거)으로 집계한다.

- **Empirical Impact**: 실험에서 FineREX는 512개 데이터 기반 NER-RE에서 entity F1과 relationship F1을 각각 절대 15.50%, 31.46% 개선했으며, 대형 일반 목적 모델(LLaMA 3.3 70B) 대비도 파라미터가 훨씬 적은 8B로 더 높은 추출 품질을 보였다. 또한 16개 DOJ 인신밀수 사건 문서 적용 결과, 법적 잡음 비율을 거의 절반(약 50%에 근접) 줄이고, 장문에서 노드 중복률을 17.78%에서 11.17%로 낮췄다. 더불어 리라이팅과 불필요한 재추출을 제거해 end-to-end 처리 시간을 50.0% 단축해, KG 품질과 효율을 동시에 개선하는 도메인 fine-tuning의 실용적 이점을 입증했다.



### Efficiently Representing Algorithms With Chain-of-Thought Transformers (https://arxiv.org/abs/2606.19697)
- **Prior Approaches**: 기존 chain-of-thought(CoT) 연구는 transformer가 Turing complete하다는 표현력 결과를 통해 알고리즘을 ‘원리상’ 시뮬레이션할 수 있음을 보여줬습니다. 하지만 알려진 구성은 주로 Turing machine(TM) 수준의 전이 하나하나를 토큰/디코딩 단계로 옮기는 방식이라, textbook 알고리즘의 기준 모델인 Word RAM의 random-access와 word 단위 연산을 효율적으로 다루기 어렵다는 문제가 남았습니다. 그 결과 TM 기반 시뮬레이션은 Word RAM 대비 큰 오버헤드(대체로 t^2급)를 유발할 수 있었습니다.

- **Core Contribution**: 이 논문은 질문을 구체화해 “CoT transformer가 Word RAM 알고리즘을 textbook 복잡도에 가깝게 효율적으로 모사할 수 있는가?”를 직접 답합니다. 결론적으로, CoT transformer가 Word RAM의 임의 프로그램을 poly-logarithmic 오버헤드만으로 시뮬레이션할 수 있음을 세 가지 실용적 설정에서 증명합니다. 특히 정렬(O(n log n))이나 Dijkstra(O(E+V log V)) 같은 알고리즘 수준의 효율이 과도하게 망가지지 않는다는 점을 목표로 삼습니다.

- **Technical Challenges**: 핵심 기술 난제는 random-access 메모리 접근과 word 단위 산술을 ‘토큰 기반 CoT’가 아니라도 효율적으로 재현하는 구조를 설계하는 데 있습니다. 논문은 (i) 유한 정밀도 + polylog width + rightmost-unique hard attention, (ii) fixed width + log-precision + continuous CoT(벡터 형태 추론), (iii) transformer 위에 선형 RNN 레이어를 얹는 hybrid 아키텍처를 통해 Word RAM instruction 단위 모사를 구성합니다. 또한 곱셈/나눗셈/mod처럼 계산 비용이 큰 연산이 있는 경우와 없는 경우에 대해 CoT 단계 수의 polylog 오버헤드가 각각 달라지도록 분석합니다.

- **Empirical Impact**: 이 논문은 실험 성능이 아니라 복잡도/표현력 관점의 ‘직접 시뮬레이션 가능성’을 엄밀히 제공해, reasoning 모델의 이론적 효율 논의를 한 단계 끌어올립니다. 정보이론적 하한(이산 CoT에서 w-비트 단어를 표현할 때의 하한)을 비교해, 제안한 discrete CoT 시뮬레이션이 사실상 최선에 가깝고 곱셈 없는 flat instruction에서는 더 타이트해짐을 보여줍니다. 결과적으로 CoT가 단지 임의 계산을 “할 수 있다”를 넘어, textbook 알고리즘 수준의 효율을 “대체로 유지할 수 있다”는 방향의 이론적 근거를 마련한 점이 의미 있습니다.



### LOKI: Memory-Free Null-Space Constrained Lifelong Knowledge Editing (https://arxiv.org/abs/2606.19679)
- **Prior Approaches**: 기존 lifelong knowledge editing은 주로 외부 메모리/지식 결합, 파라미터에 직접 병합, 혹은 intrinsic knowledge(모델 내부 지식) 수정 등으로 나뉜다. 특히 MEMIT/AlphaEdit류의 직접 편집 방법은 편집 레이어를 고정하고, 과거 지식 샘플을 이용해 통계(예: 공분산/특징 행렬)를 만든 뒤 null-space 투영으로 망각을 줄인다. 이 방식은 고정된 레이어만 업데이트해 유연성이 떨어지고, 과거 지식 접근과 대규모 전처리/통계 계산이 필요해 비용이 커진다.

- **Core Contribution**: LOKI(Layer-adaptive Orthogonal Knowledge Insertion)는 샘플(사실)마다 달라질 수 있는 ‘중요 레이어’를 동적으로 선택해 업데이트 유연성을 높인다. 또한 null-space 보존을 위해 과거 데이터 통계가 아니라 모델 가중치 자체의 null-space를 사용해, 과거 지식 접근 및 추가 파라미터/메모리 없이도 catastrophic forgetting을 완화하는 경로를 제시한다. 결과적으로 편집의 지역성(locality)과 순차 업데이트의 효율성을 동시에 노린다.

- **Technical Challenges**: 핵심 난제는 (1) 어떤 레이어가 해당 사실을 인코딩하는지 샘플 단위로 안정적으로 고르는 것과 (2) 과거 정보를 참조하지 않으면서도 새 업데이트가 기존 기능을 덜 교란하도록 제약하는 것이다. LOKI는 HSIC(Hilbert-Schmidt Independence Criterion) 기반 정보 병목(HIB) 기준으로 레이어별 정보 흐름을 계산해, 단일 forward pass로 top-m 레이어를 선택한다. 이어서 선택된 레이어의 FFN down-projection 행렬 W_out의 null-space를 SVD로 구하고, projected gradient descent에서 그 null-space로만 업데이트를 제한해 과거 간섭을 줄인다.

- **Empirical Impact**: 다양한 데이터셋/설정에서 LOKI는 기존 lifelong knowledge editing 방법 대비 더 높은 정확도를 보이며, 평균 accuracy에서 최대 14% 개선을 달성한다. 또한 구성요소를 분해한 ablation 및 탐색 실험을 통해 HSIC 기반 동적 레이어 선택과 가중치 null-space 투영이 성능 향상에 기여함을 확인한다. 과거 지식 통계 접근을 제거하면서도 competitive한 망각 완화와 편집 성능을 보여, 실사용 관점의 비용/유연성 균형에 의미 있는 진전을 제공한다.



### TeleMorpher: Toward Robust Simultaneous Motion-Location Editing (https://arxiv.org/abs/2606.19676)
- **Prior Approaches**: 기존 motion editing은 one-shot으로 동작하면서 배경/인물 외형 보존과 프레임 간 일관성을 개선하려는 흐름이 강했습니다. 다만 MotionEditor, Edit-Your-Motion 같은 방법은 주로 motion만 다루거나, source-대상 motion 간 차이가 커지면 깜빡임과 외형 불일치가 생기고, location(공간 위치) 변화까지는 명시적으로 해결하지 못했습니다. 결과적으로 실사용에서 흔한 ‘동시에 동작과 위치를 바꾸는’ 요구를 충족하기 어려웠습니다.

- **Core Contribution**: TeleMorpher는 motion과 location을 함께 편집하는 동시(one-shot) 프레임워크를 제안합니다. 핵심 아이디어는 오프더샬 모델로 만든 segmentation·inpainting으로 배경 간섭을 줄이고, 대신 3D 아바타 기반의 motion prior(합성 motion 비디오)를 “타깃 모션 기준”으로 사용해 더 정밀하고 제어 가능한 편집을 가능하게 한다는 점입니다. 또한 학습 없이 pose warping 기반 protagonist guidance를 생성해 source-대상 모션 간 gap을 줄이면서 인물 외형을 보존하도록 설계했습니다.

- **Technical Challenges**: 동시 motion-location 편집은 (1) source-대상 motion과 location의 큰 gap, (2) 세분화/경계 모호성, (3) 배경 복잡도·동적도·카메라 움직임 등 정보량의 차이가 품질을 떨어뜨리는 문제가 있습니다. TeleMorpher는 먼저 인물 마스크로 전경/배경을 분리하고 배경은 inpainting으로 제거해 편집 혼선을 줄입니다. 그다음 학습-free two-way pose warping(곡선 피팅 기반 변형 + 필요 시 영역 수준 합성)을 통해 타깃 포즈에 더 잘 맞는 protagonist 조건을 만들고, 이를 UNet attention의 일부 key/value 채널에 주입(추가 mixup 포함)해 모션 정렬과 외형 일관성을 동시에 강화합니다.

- **Empirical Impact**: YouTube에서 수집한 in-the-wild 영상과 TaiChi 데이터셋에서 정량·정성 평가 모두에서 기존 베이스라인 대비 우수한 동시 편집 성능을 보였습니다. 특히 location과 motion을 함께 바꿀 때도 전경(인물) 및 배경의 시각적 일관성을 더 잘 유지하는 것으로 보고됩니다. 또한 배경 보존과 인물 스켈레톤 정렬을 측정하는 LPIPS 기반의 신규 metric 2종을 제안해, 정량 평가 신뢰도를 높였다는 점에서 분야 확장성도 기대됩니다.



### Creating Multilingual Mental Health Dialogue Datasets: Limits of Persona-Based Localization via Nationality and Languag (https://arxiv.org/abs/2606.19640)
Comments:
          15 pages, 4 figures. Accepted to the 2026 Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2026), co-located with ACL 2026

- **Prior Approaches**: 기존 연구는 우울증 검출·스크리닝을 위해 영어 중심 데이터셋을 보강하거나, 합성 데이터에서 persona를 활용해 학습/평가 규모를 키우는 흐름이 강했다. 특히 영어 임상 persona를 기반으로 BDI-II 같은 척도를 넣어 대화형 환자(standardized patient)를 만들고, LLM을 judge로 써 심각도를 비교하는 방식이 자주 쓰인다. 다만 대부분이 영어/서구 맥락에 고정돼 있어, 언어가 바뀌면 동일한 임상 신호가 유지되는지에 대한 검증은 부족했다.

- **Core Contribution**: 이 논문은 영어로 검증된 임상 persona에서 nationality와 Language Use만 바꿔 만다린, 벵골어, 힌디어 대화로 확장할 때 우울 심각도 신호가 보존되는지 통제 실험으로 확인한다. 또한 생성된 다국어 대화를 대상으로 여러 LLM judge들이 우울 심각도를 블라인드(pairwise)로 비교할 때의 정확도와 불확실성(tie)을 함께 분석한다. 핵심 결론은, 단순 파라미터 치환만으로는 언어 간 임상 일관성을 안정적으로 재현하기 어렵다는 점이다.

- **Technical Challenges**: 기여를 실현하려면 (1) 영어 persona의 BDI-II 기반 증상 표현을 다른 언어에서도 동일한 임상 강도로 유지하면서, (2) judge가 비영어 텍스트에서 같은 신호를 일관되게 해석하도록 평가 파이프라인을 설계해야 한다. 연구진은 영어 baseline persona를 고정하고 nationality-언어 쌍(미국-영어, 중국-만다린, 방글라데시-벵골어, 인도-힌디어)만 변경해 48개 persona를 만들었고, 각 persona당 다단계 증상 인터뷰 대화를 생성한 뒤 12개 persona 간 66회 pairwise 비교로 심각도 순서를 회복하는지 측정했다. 평가지표는 overall accuracy뿐 아니라 same-level error rate(같은 심각도 범위 오차)와 tie distance(불확실성의 심각도 간격)까지 포함해 ‘오차의 구조’가 언어에 따라 어떻게 바뀌는지 드러냈다.

- **Empirical Impact**: 실험 결과, 모든 judge 모델에서 영어 성능이 상한으로 유지됐고 비영어(특히 벵골어·힌디어)로 갈수록 정확도 하락과 cross-severity error 증가, tie 빈도 및 tie distance 확대가 나타났다. 일부 모델은 영어에서만 심각도 보정(calibration)이 안정적이었고, 더 작은 8B 계열은 비영어에서 변동이 커 BDI-II 수준 신호가 약해진 정황을 보였다. 저자들은 따라서 영어-centric persona를 그대로 ‘동등한 샘플’로 간주하기보다, 언어·문화에 맞춘 culturally responsive 데이터 생성과 출력 수준 검증이 필요한 임상 아티팩트로 취급해야 한다고 강조한다.



### Before the Labels: How Dataset Construction Shapes Suicidality Detection in Clinical Tex (https://arxiv.org/abs/2606.19637)
Comments:
          To appear in the Proceedings of the 11th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2026)

- **Prior Approaches**: 임상 NLP는 자살 위험 탐지를 고위험 과제로 보고, 기존에는 소셜미디어보다 EHR을 더 신뢰할 만한 ground truth로 간주해 왔습니다. 그러나 실제 예측 성능이 다양한 임상 환경에서 기대보다 미미해 데이터 라벨이 어떤 가정을 내장하는지 점검이 필요하다는 문제의식이 커졌습니다.

- **Core Contribution**: 이 논문은 EHR 기반 자살징후(suicidality) 데이터셋의 라벨을 ‘중립적 진실’이 아니라 특정 operationalization(문서화 매개·에피소드 단위·의도(intent) 해결)으로 해석해야 한다고 제안합니다. 특히 ScAN(Scandinavic? ScAN) 사례를 통해 같은 라벨이 문서의 시간성/부정/불확실성에 따라 서로 다른 임상 프레이밍을 포괄할 수 있음을 보여줍니다. 

- **Technical Challenges**: 핵심은 라벨이 어떻게 생성되는지(거버넌스·코호트 정의·어노테이션·병원 체류 단위 집계·불확실성 처리)가 언어적 패턴으로 어떻게 ‘평탄화’되는지 구조적으로 추적하는 것입니다. 논문은 ScAN에서 ICD 기반 코호트 시딩, 단일 어노테이터 라벨링(불일치 시 physician review로 수정), “unsure”를 downstream에서 “negative”와 합치는 방식이 라벨 신호를 바꾼다는 점을 프레이밍 분석으로 확인합니다.

- **Empirical Impact**: 언어 분석 결과, 동일 라벨이 temporality/negation/uncertainty 프로필이 다른 스팬을 흡수하며 특히 “unsure”·과거표지(historical)·불확실성 신호가 라벨 공간에서 구분되지 않을 수 있음을 정량적으로 보였습니다. 또한 discharge summary와 HPI처럼 문서 섹션에 따라 동일 라벨이라도 unmodified 비율이 크게 달라져, 모델 성능·캘리브레이션이 숨은 데이터 가정에 의해 체계적으로 달라질 수 있음을 시사합니다.



### Hard or Just Unreached? Diagnosing the Sampling Blind Spot in Math-Reasoning Difficulty Estimation (https://arxiv.org/abs/2606.19636)
Comments:
          9 pages of main paper, 4 figures and 5 tables in the main paper, with more in the appendix

- **Prior Approaches**: 수학·과학 추론 벤치마크와 RL 파이프라인은 예측 사슬을 샘플링한 뒤 gold에 도달한 비율 pass@k를 난이도 신호로 삼는 경우가 많다. pass@k=0처럼 6번 안에 어떤 샘플도 정답에 도달하지 못한 항목을 “가장 어려운 구간”으로 라벨링·필터링하며, verifier 학습과 데이터 선별도 동일한 편향을 상속한다.

- **Core Contribution**: 논문은 pass@k 기반 난이도 추정이 가장 어려운 층에서 구조적 실명을 갖는다고 지적한다. GSM8K·MATH의 free-form 수학 셀에서 “6번 샘플링해도 정답이 0인 구간(pass@6=0)”의 10.3~22.9%는, 동일한 연산 예산을 맞춘 결정적 규칙(그리디 + 5개의 cheap residual-stream perturbation)으로 실제로 도달 가능했다.

- **Technical Challenges**: 핵심은 ‘샘플링이 못 푼 것이 본질적으로 어려운가, 아니면 샘플링 축(stochastic axis)에서는 못 봤을 뿐인가’를 구분하는 것이다. 이를 위해 attention/출력 분포가 아니라 내부 표현(residual stream)을 개입하는 activation grafting을 진단 도구로 쓰고, 서로 다른 5종 graft의 기계적 다양성이 교차 검증될 만큼 다른 subset을 커버함을 확인했다.

- **Empirical Impact**: 실험에서는 그리디 단독은 해당 수학 셀에서 pass@6=0 구간을 최대 6%대까지밖에 회복하지 못했지만, 6-chain 결정적 체계는 10.3~22.9%를 회복하며 예산이 늘수록 회복이 확장됐다. 즉, pass@k로 “어렵다”고 표시된 일부가 사실은 ‘샘플링으로는 도달 불가능하지만 결정적 축의 개입으로는 구조적으로 식별 가능한’ 영역일 수 있어, RL+verifiable reward·데이터 큐레이션·난이도 커리큘럼·verifier 학습의 라벨 품질에 직접적인 영향을 준다.



### Token Factory: Efficiently Integrating Diverse Signals into Large Recommendation Models (https://arxiv.org/abs/2606.19635)
Comments:
          8 pages, 10 figures

- **Prior Approaches**: 기존에는 전통적 추천 신호를 transformer 기반 Large Recommendation Models(LRMs)에 넣기 위해 신호를 텍스트로 변환하거나, 이산적인 아이템 표현을 만들어 프롬프트에 편입하는 방식이 주로 쓰였다. 하지만 이런 “textualize” 계열 접근은 프롬프트 길이 폭증, 큰 메모리 사용, 높은 연산 비용으로 이어지기 쉽다.

- **Core Contribution**: 이 논문은 전통적 신호를 LRM이 바로 처리할 수 있는 “soft tokens”로 변환하는 프레임워크 Token Factory를 제안한다. 이를 통해 이질적인 입력 특징을 효율적으로 압축·통합하면서 추천 성능을 개선하는 것을 목표로 한다.

- **Technical Challenges**: 핵심 기술 난제는 서로 다른 전통 신호를 토큰화할 때 정보 손실을 최소화하면서도 프롬프트 길이와 연산·메모리 부담을 동시에 줄이는 것이다. 논문은 Token Factory의 아키텍처로 soft tokens를 생성·주입해 prompt length explosion을 억제하고, LRM 전처리 파이프라인의 효율을 확보한다.

- **Empirical Impact**: 생산 규모(production-scale) 추천 환경에서의 실험 결과는 Token Factory가 기존 방식 대비 효율성과 모델 성능 모두에서 이점을 제공함을 보여준다. 즉, 대규모 서비스에서 다양한 전통 신호를 현실적으로 통합할 수 있는 경로를 제시하며, 추천 산업 적용 가능성을 강화한다.



### CTS-MoE: Implicit Terrain Adaptation via Mixture-of-Experts for Perceptive Locomotion (https://arxiv.org/abs/2606.19633)
- **Prior Approaches**: 기존 다지형(계단, 갭, 장애물) 보행 제어는 단일 보상으로 여러 상황을 커버하거나, 표현 학습·일반화에 집중하는 방식이 많았습니다. 또한 비대칭 teacher–student나 비전/맵 기반 지형 인식 접근은 종종 지나치게 보수적으로 움직이며, 계단/갭처럼 급격한 위상 변화에 필요한 선제적(anticipatory) 특화 동작을 충분히 만들지 못했습니다. 반면 계층형(서브폴리시+선택기) 방식은 학습 분리로 인해 전환의 일관성이 깨지고, unseen 지형에 대한 일반화도 제한될 수 있습니다.

- **Core Contribution**: 이 논문은 CTS-MoE(Concurrent Teacher-Student with Mixture of Experts)를 제안해, 다중 태스크 보상에서 발생하는 ‘공유-분리’ 긴장을 동시에 다룹니다. 핵심은 dense MoE actor로 공통 보행 기반 위에서 전문가(expert) 조합을 만들고, sparse multi-critic의 task-specific value head로 보상 간 value interference를 억제하는 구조입니다. 또한 배포 시에는 perception 기반 routing만으로 적응이 일어나고, 별도의 high-level selector나 지형 분류기는 필요 없습니다.

- **Technical Challenges**: 기여의 실현에서 가장 큰 기술적 난관은 (1) 서로 충돌하는 여러 보상으로 인해 value 학습이 붕괴하거나(gradient/value 간섭) (2) 부분 관측(POMDP) 환경에서 태스크에 맞는 전문가를 안정적으로 선택해야 한다는 점입니다. 저자들은 concurrent teacher–student를 단일-stage end-to-end로 학습해 지각/표현 문제를 완화하고, PPO에 multi-reward–multi-critic을 결합하며 POPArt 기반 per-task return normalization으로 태스크 간 스케일 차이도 안정화합니다. 더불어 actor의 routing은 task labels 없이도 지형 정보를 반영하도록 설계되어, 훈련 동안에만 task ID를 쓰고 배포에서는 제거됩니다.

- **Empirical Impact**: Unitree Go1에서 시뮬레이션(IsaacLab)과 실로봇 실험 모두를 통해 CTS-MoE가 단일(모놀리식)·기존 perceptive baseline보다 높은 성공률과 더 낮은 속도 추적 오차를 보였다고 보고합니다. 특히 gap에서 성공률이 +29.3%p, climb-up에서 +10.3%p 개선되는 등 선제적 동작이 필요한 지형에서 효과가 두드러졌습니다. 또한 unseen terrain에서도 성능 이점이 유지되어, 전문가 조합이 명시적 태스크 인식 없이도 지각 기반으로 자연스럽게 적응한다는 점을 실증합니다.



### Formal Verification of Learned Multi-Agent Communication Policies via Decision Tree Distillation (https://arxiv.org/abs/2606.19632)
Comments:
          9 pages, 3 figures, 7 tables. Accepted at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026), Pittsburgh, Pennsylvania, USA, September 27-October 1, 2026

- **Prior Approaches**: 기존 MARL(MAPPO, QMIX 등)은 다중 에이전트 간 협력을 emergent communication으로 학습하지만, 실제 드론·자율주행 같은 안전 비즈니스에는 formal safety guarantee가 부족합니다. 또한 단일 에이전트 신경망 검증 도구(Relux/DeepPoly 등)는 multi-agent의 상태 공간 폭발과 통신을 직접 다루기 어렵고, 정책 추출·증류(VIPER/MAVIPER류)는 통신 의미를 포함한 end-to-end 검증 파이프라인이 제한적이었습니다.

- **Core Contribution**: 이 논문은 learned multi-agent communication policy를 대상으로 end-to-end 안전 검증을 제공하는 새로운 프레임워크를 제시합니다. 핵심은 신경 정책을 communication semantics를 반영해 decision tree로 policy abstraction(증류)한 뒤, PRISM 기반 확률 모델체킹으로 PCTL 성질을 formally verify하고, verified safety가 원래 네트워크로 Monte Carlo에서 전이되는지를 함께 검증한 점입니다. 

- **Technical Challenges**: 문제는 (1) 신경 정책이 NP-hard 검증 대상이고 (2) multi-agent·통신으로 상태 공간이 기하급수적으로 커진다는 점입니다. 이를 위해 관측에서 50개 도메인 피처를 설계해 증류 정확도를 높이고, CART 기반 decision tree를 PRISM DTMC로 자동 변환할 때 feature-to-state 변수 대응을 “정수 산술로 완전 대응”하도록 강제했으며, PCTL 검증은 pairwise decomposition과 union-bound aggregation으로 조합 검증을 수행했습니다. 비결정성은 초기 상태 난수와 leaf impurity를 모델링해 확률 추정의 일관성을 확보하고, 비핵심 에이전트는 calibration rollout으로 이웃 기반 전이 커널을 실증 추정했습니다.

- **Empirical Impact**: VQ-VIB(이산 메시지) 정책(에이전트 5~7명)에서 decision tree distillation fidelity가 97.9%±1.2%에 도달했고, PRISM에서 안전/생존/협력 범주의 PCTL 속성 18개를 검증해 평균 88.9% satisfaction을 보고했습니다. 특히 충돌 확률은 0.3%로 안전 임계(1%)를 만족했으며, 원래 신경망에 대한 Monte Carlo 검증에서 verified 성질이 <=0.6 percentage-point 편차로 전이됨을 확인했습니다. 이산 VQ-VIB 메시지는 연속 통신 대비 11.6~13.6pp의 fidelity 우위를 제공해 검증을 3~4배 빠르게 만드는 실용적 다리로 평가됩니다.



### RIVET: Robust Idempotent Voice Attribute Editing (https://arxiv.org/abs/2606.19629)
- **Prior Approaches**: 음성 속성 편집(나이/성별 등)은 조건부 생성이나 disentangled(분리) 표현을 이용해 속성은 바꾸되 화자 정체성은 보존하는 데 초점을 둡니다. 하지만 대규모 데이터에서 속성 라벨은 노이즈/불일치(뒤집힘, 애매한 정의, 자동 추정 등)를 갖기 쉬워, 조건부 생성이 잘못된 연관을 학습하면 편집이 불안정해지고 화자-속성 얽힘이 커질 수 있습니다. 기존에는 라벨 신뢰도 추정, 라벨 보정, 일관성(consistency) 같은 정규화로 대응했지만, 반복 편집 시 드리프트를 줄이는 관점은 상대적으로 덜 다뤄졌습니다.

- **Core Contribution**: 이 논문은 idempotency(멱등성)를 “잡음 라벨에 대한 강건성”을 높이는 정규화 원리로 제안합니다. 멱등성은 같은 편집 연산을 반복해도 결과가 변하지 않는 성질(f(f(x))=f(x))이며, 이는 잘못 라벨된 예시에 대한 민감도를 낮춰 편집 드리프트를 억제하는 효과가 있다고 설명합니다. 이를 반영한 학습 프레임워크 RIVET을 제안해, 화자 정체성과 속성 편집 성공률을 동시에 끌어올립니다.

- **Technical Challenges**: 핵심 난제는 노이즈가 섞인 조건 라벨이 학습된 매핑을 흔들어 반복 생성에서 점진적 drift를 만들 수 있다는 점입니다. RIVET은 출력 공간이 아니라 latent(잠재) 표현 공간에서 멱등성을 강제해, 재인코딩 후 표현이 안정적으로 고정되도록 idempotency loss를 설계합니다(정지 그래디언트 stop-gradient로 안정 타깃을 고정). 또한 speaker encoder(ECAPA-TDNN)와 VITS 기반 speech encoder 양쪽에 제약을 적용하고, flow 기반 속성 조건 변환과 분류/생성 손실을 end-to-end로 함께 최적화합니다.

- **Empirical Impact**: 통제된 라벨 노이즈 환경(EARS)과 자연 노이즈가 있는 GLOBE에서 실험했을 때 RIVET은 기준선 대비 cosine similarity로 측정한 화자 정체성 보존을 더 잘 유지했습니다. 속성 편집 성공도 개선되어 특히 성별 편집에서 정확도가 뚜렷하게 향상되었고, 반복 재구성 라운드에서도 정체성 임베딩 안정성이 높게 나타났습니다. 한편 UTMOS 및 WER 기준 자연스러움/이해도는 비슷하게 유지되어, 멱등성 기반 정규화가 품질을 크게 해치지 않으면서 노이즈에 강건한 편집을 제공함을 보여줍니다.



### VCG: A Multimodal Retrieval Framework for E-Commerce Video Feeds under Extreme Cold-Start Conditions (https://arxiv.org/abs/2606.19627)
- **Prior Approaches**: 기존 전자상거래 추천은 검색 기반 카탈로그에서 축적된 클릭·구매 이력을 바탕으로 collaborative filtering(예: Matrix Factorization, Autoencoder)에 의존해 왔습니다. 하지만 동영상 피드 환경에서는 신규 영상이 상호작용 이력이 부족한 extreme cold-start가 발생하고, watch-time 최적화가 duration bias와 position bias를 함께 왜곡해 표준 engagement 신호가 흔들립니다. 텍스트 메타데이터 기반 검색은 메타 태그가 희소하거나 부정확한 경우 시각적 취향을 제대로 반영하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 Video Candidate Generation(VCG)이라는 대규모 멀티모달 검색 엔진을 제안해, 사용자와 영상을 shared semantic space에 매핑하고 행동 이력 없이 zero-shot으로 후보를 뽑는 접근을 제시합니다. 핵심은 domain-adapted CLIP 기반 두 타워(two-tower) 구조로, retrieval은 시각 내용과 사용자 스타일의 매칭 질문(What does this video look like?)으로 전환한다는 점입니다. 또한 generative embedding(LVLM/LLM)과 discriminative embedding(CLIP)의 검색 적합성을 비교해, retrieval에서는 contrastive 기반 임베딩이 유리하다는 분석을 함께 제공합니다.

- **Technical Challenges**: VCG가 직면한 기술 과제는 (1) 희소한 상호작용 그래프에서 사용자-영상 유사도를 안정적으로 학습/정의하는 문제, (2) 동영상 피드의 duration·position bias를 오프라인 지표로는 검증하기 어려운 exposure bias 문제, (3) LLM 임베딩을 그대로 쓰면 embedding space collapse처럼 검색 성능을 망가뜨릴 수 있다는 문제입니다. 저자들은 픽셀 기반 시각 신호만으로 10개 프레임을 평균해 영상 임베딩을 만들고, 메타 태그의 잡음을 의도적으로 배제했으며, 오프라인 평가는 LVLM-as-a-judge(Qwen2.5-VL)로 visual coherence를 점수화해 온라인 성공의 대리 지표로 연결했습니다. 더 나아가 Qwen 임베딩은 attribute 예측에는 강하지만 k-NN 검색에서는 등방성이 깨져 약해지는 반면, CLIP은 hypersphere에서 더 균일한 분포를 만들어 검색 엔진으로 적합하다고 보여줍니다.

- **Empirical Impact**: 오프라인에서는 recency 기준선 대비 의미적/시각적 coherence 지표가 개선됐고, Top-10에서 LVLM judge의 Relevant 점수 분포가 ‘좋은 매칭/우수 매칭’ 쪽으로 이동하는 것이 관찰됐습니다. 온라인 4주 A/B 테스트에서는 비디오 시작률은 +8%로 크지 않았지만, 소비 깊이(예: 절반 이상 시청)는 50% uplift을 기록해 클릭을 넘어선 체류·완주 품질 개선을 확인했습니다. 또한 daily lift가 실험 기간 내내 지속돼 단기 노출 효과가 아니라는 점과, 매출·사용자 리텐션 같은 핵심 지표가 안정적으로 유지돼 추천 피드가 거래 흐름을 잠식하지 않았다는 의미가 큽니다.



### Before the Pull Request: Mining Multi-Agent Coordination (https://arxiv.org/abs/2606.19616)
Comments:
          9 pages, 2 tables. LNCS format. Code, dataset, and mining toolkit: this https URL

- **Prior Approaches**: AIDev 같은 기존 연구는 pull request(논문에서는 PR) 결과(승인/거절, 속도 등)만 대규모로 관찰해 ‘speed–acceptance gap’을 설명하려 했습니다. 그러나 중복 작업이나 충돌 편집은 PR로 남지 않는 경우가 많아(버려진 중복, 레이스에서 승자만 흔적), pull-request-level 텔레메트리로는 조정과 신뢰의 공백이 잘 보이지 않습니다. 또한 일반적인 에이전트 평가는 단일 에이전트 성과에 치우쳐 다중 동시성에서의 사전 조정 문제를 직접 계량하기 어렵습니다.

- **Core Contribution**: 이 논문은 PR 이전에 벌어지는 ‘공유 작업에 대한 동시 에이전트의 claim–divide–collision’ 과정을 핵심 원인으로 지목하고, 이를 측정할 수 있는 서버리스 git-native 조정 인프라 grite를 제안합니다. grite는 중앙 서버 없이 git ref 안에 append-only, content-addressed(해시 기반), 선택적으로 서명된 event log를 남겨 조정 상태 자체를 데이터로 만듭니다. 그 결과 PR 이전 텔레메트리가 비가시였던 부분을 재현·채굴(mining)할 수 있게 합니다.

- **Technical Challenges**: 동시 에이전트가 공유 상태를 갱신할 때 충돌을 잃지 않으면서도(데이터 손실 방지) 상태가 모든 복제본에서 수렴해야 했습니다. 논문은 CRDT 기반 projection(이벤트를 순서와 무관하게 동일 상태로 재구성)으로 수렴성을 확보하고, advisory lease(TTL)로 상호배제를 부여해 잠금 충돌과 스타베이션을 포함한 행동 로그를 남깁니다. 또한 이벤트에 actor, timestamp, conflict/duplicate 같은 필드를 두고 서명·해시로 위변조를 막아, 실패 모드 탐지가 PR 기록만으로는 어려운 수준까지 자동 복구 가능하도록 설계했습니다.

- **Empirical Impact**: 실험에서는 조정 장치를 단계적으로 추가했을 때 중복·충돌·생산성이 어떻게 바뀌는지 정량화했습니다. advisory lease만으로도 goodput이 늘고 충돌 편집이 크게 줄지만, ‘동료가 이미 끝낸 작업을 다시 찾는 중복 rediscovery’는 여전히 많이 남아 duplicate-work을 0으로 만들지 못했습니다. 반면 locks+state(공유 완료 상태까지 사용)에서는 중복 작업 비율이 78%→0%로 떨어지고 유효 처리량(goodput)이 3배 이상 개선되며, 같은 이벤트를 어떤 순서로 받아도 복제본 상태가 바이트 단위로 동일해 데이터 손실도 방지됨을 확인했습니다. 더 나아가 PR 기록에서 사라지는 conflicting edits, lock starvation, race-to-close 같은 구체 실패 모드를 log에서 자동 감지·복구 가능하다는 점이 impact로 제시됩니다.



### StaminaBench: Stress-Testing Coding Agents over 100 Interaction Turns (https://arxiv.org/abs/2606.19613)
- **Prior Approaches**: 기존 코딩 에이전트 벤치마크는 SWE-Bench, SWE-BenchPro, TerminalBench처럼 대부분 “단일 작업(단일 인터랙션)” 중심으로 평가해, 사용자가 이후 계속 기능을 수정·추가하는 긴 세션의 요구를 충분히 반영하지 못했습니다. 몇몇 장기/멀티턴 연구가 있더라도 실제 사용 맥락에 가까운 “연속 턴(stamina) 유지”를 정량화하진 못한 경우가 많았습니다.

- **Core Contribution**: 이 논문은 코딩 에이전트의 stamina(실패 전까지 연속으로 처리하는 변경 요청 턴 수)를 직접 측정하는 StaminaBench를 제안합니다. REST API 서버를 초기 명세로 구현한 뒤, 절차적으로 생성된 스키마 변경 요청을 여러 턴 연속 수행하며 매 턴 테스트를 통과하는지로 성능을 평가합니다. 테스트와 명세 생성은 LLM 개입 없이 전적으로 프로그램적으로 이뤄져 재현성과 언어 비의존성을 강화합니다.

- **Technical Challenges**: 핵심 기술적 난제는 긴 상호작용 동안 에이전트가 “진화하는 코드베이스”의 일관성을 유지하고, 누적된 수정/디버깅을 제한된 컨텍스트에서 처리하는 것입니다. 연구진은 OpenAPI-like 구조화 상태, 고정된 변경 action space, HTTP 기반 블랙박스 테스트(서버는 격리 환경에서 동작)로 문제를 엄밀히 제어해, 유효한 변경 시퀀스와 자동 검증을 가능하게 했습니다. 또한 실패 시 테스트 피드백을 되돌려주는 retry loop를 도입해 오류 수정이 가능하도록 파이프라인을 구성합니다.

- **Empirical Impact**: 20개 시나리오(각 100턴)에서 확인한 결과, 테스트 피드백/재시도 없이 대부분의 모델은 5~6턴 내 실패해 “vibe-coding(충분한 테스트 없이 감으로 코딩)”의 취약성이 극명하게 드러났습니다. 반면 테스트 피드백을 에이전트에 제공하고 retry를 허용하면 통과 턴 수가 최대 12배까지 개선되며, 에이전트 harness 품질이 성능을 크게 좌우(강한 모델도 harness에 따라 최대 6배 격차)함을 보였습니다. 공개된 벤치마크와 생성 데이터는 멀티턴 코딩 에이전트의 장기 신뢰성 연구를 촉진할 것으로 기대됩니다.



### Latent Confounded Causal Discovery via Lie Bracket Geometry (https://arxiv.org/abs/2606.19610)
Comments:
          39 pages

- **Prior Approaches**: 기존 인과발견은 제약기반(PC/FCI), 점수기반(GES), 함수모델(LiNGAM 계열), 연속 최적화(NOTEARS/DCDI) 등으로 나뉘며, 잠재 교란은 보통 부분-조상 구조나 그래프 탐색 제약으로 처리해 왔습니다. 다만 latent confounding이 있으면 관측/개입 정보가 깔끔히 정렬되지 않아 후보 간 결합이 깨지고, 그래프 탐색 공간의 초지수적 규모도 여전히 병목이 됩니다. NOTEARS류는 acyclicity를 연속 제약으로 완화하지만, 잠재 구조로 인해 ‘어떤 화살표(후보군)’가 실제로 타당한지 줄이는 전면 스크리닝은 상대적으로 약합니다.

- **Core Contribution**: 이 논문은 Kan-Do-Calculus(KDC)의 범주론적 bi-adjunction(개입은 left Kan extension, 조건화는 right Kan extension) 구조를 매끄러운 통계 매니폴드로 옮기며, 잠재 교란이 ‘정확히 보이는’ 흐름의 합성 가능성을 막는 obstruction임을 정리합니다. Radon–Nikodym derivative(관측-개입 측도 변화)를 통해 개입 유도 국소 인과 벡터필드를 만들고, Lie bracket/ Frobenius residual의 닫힘 실패를 잠재 또는 미모델 구조의 증거로 해석합니다. 이를 바탕으로 BRIDGE와 SKFM이라는 두 알고리즘 패러다임(후속 점수기반 루틴을 위한 후보군 축소 vs end-to-end 기하학 학습)을 제안합니다.

- **Technical Challenges**: 핵심 난제는 (1) 관측 분포와 개입 분포 사이의 Radon–Nikodym ratio를 신뢰도 있게 추정해 벡터필드를 얻고, (2) 그 벡터필드들이 Lie/Frobenius 관점에서 ‘닫히는지’를 계산 가능하게 판정해 후보군을 줄이며, (3) 잠재 교란이 만든 curvature를 견고하게 분리하는 것입니다. BRIDGE는 interventional density engine(예: density-ratio 또는 CNF 기반)으로 로컬 반응장을 추정한 뒤 Lie bracket non-closing 쌍을 latent-obstruction 후보로 기록해 admissible arrows의 고-리콜 패밀리를 프루닝하고, 줄어든 패밀드를 downstream score/differentiable discovery에 넘깁니다. SKFM은 개입 필드를 amortized 방식으로 학습하고 Frobenius residual의 잠재 curvature를 spectral factorization해 Lie-space 엔드포인트까지 end-to-end로 밀어붙이며, soft solvable-Lie acyclicity penalty로 그래프 추출까지 수행합니다.

- **Empirical Impact**: 실험에서는 두 방법 모두 latent confounders가 있는 상황에서 인과모형을 발견할 수 있고, DAG 탐색에 필요한 후보 공간을 기존 방식보다 여러 자릿수 이상 축소하는 것을 보였습니다. 특히 BRIDGE는 기하학 스크리닝으로 ‘어떤 화살표를 점수화할지’를 크게 줄여 점수기반/미분형 탐색의 조합 폭발을 완화합니다. SKFM은 완전한 완벽 개입 타깃 없이도(간접·노이즈 개입을 포함) 개입 유도 흐름의 비가환성(=Lie bracket로 드러나는 curvature)에서 잠재 차원을 추정하며, 그래프 탐색 중심에서 ‘인과성의 기하학’을 학습하는 새 패러다임의 실용성을 보여줍니다.



### FAPO: Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines (https://arxiv.org/abs/2606.19605)
- **Prior Approaches**: 기존 파이프라인 최적화는 주로 prompt-space에서 후보를 찾거나(DSPy, GEPA류) 고정된 프로그램 안에서 지시문을 진화시키는 방식에 의존해 왔다. 반면 멀티-스텝 LLM 파이프라인은 retrieval·reasoning·formatting이 상호작용하며 실패가 전파되기 때문에, 단순 프롬프트 튜닝만으로 병목을 놓칠 수 있다. 또한 기존 평가(HELM, BIG-bench 등)는 고정된 파이프라인의 성능 측정에 초점이 있어 “검사 가능한 파이프라인을 단계 단위로 고치며 점수를 올리는” 워크플로우와 거리가 있었다.

- **Core Contribution**: FAPO(Fully Autonomous Prompt Optimization)는 Claude Code가 표준화된 코드베이스(워크스페이스) 안에서 LLM 파이프라인을 반복 최적화하도록 만든 프레임워크다. 핵심은 단계별 중간 산출물을 기반으로 실패 원인을 attribution으로 진단하고, 우선은 prompt 편집을 시도한 뒤 효과가 부족하다고 판단될 때만 허용된 범위 내에서 chain parameter나 chain structure까지 확장하는 ‘prompt-first escalation’ 전략이다. 이로써 드물게 성공하는 예시를 찾는 방식이 아니라, NN(평가 케이스) 전체에서 평균 점수를 끌어올리는 목적에 맞춘 최적화를 수행한다.

- **Technical Challenges**: 문제는(1) 파이프라인의 어느 단계에서 어떤 유형의 오류가 반복되는지 식별하고, (2) 그에 맞는 “최소한의” 수정안을 제안하되(프롬프트→파라미터→구조 순), (3) 과도한 변경으로 오버피팅·규칙 위반·데이터 누출이 발생하지 않게 경계를 두는 것이다. FAPO는 LangGraph로 파이프라인을 stateful graph로 표현하고, 평가 시 중간 단계 증거를 기록한 뒤 rule-based+LLM 분석으로 prompt-addressable 실패와 structural 병목을 구분한다. 또한 tenant playbook/scope contract, reviewer 검증, 검증/테스트의 접근 분리로 최적화 변경을 bounded하게 유지한다.

- **Empirical Impact**: 6개 벤치마크와 3개 task model에서 FAPO는 GEPA 대비 18개 model-benchmark 비교 중 15회에서 우위(평균 +14.1 pp)를 보였다. 특히 HoVer·IFBench에서는 prompt-only 검색이 구조 변경으로 이어진 케이스에서 6/6 모두 승리했으며 평균 이득이 +33.8 pp에 달했다. 보안 태스크에서도 CTIBench-RCM(CVE-to-CWE)에서 prompt-only FAPO가 GPT-5 +4.0 pp, Foundation-Sec-8B-Instruct +7.1 pp, Foundation-Sec-8B-Reasoning +2.0 pp의 test 정확도 향상을 보이며, 일반 과제와 보안 과제를 함께 아우르는 SOTA급 파이프라인 최적화 도구로 자리매김한다.



### PrefSQA: Pairwise Preference Prediction for Speech Quality Assessment and the Critical Role of High Quality Datasets (https://arxiv.org/abs/2606.19597)
Comments:
          Accepted to INTERSPEECH 2026

- **Prior Approaches**: 기존 음성 품질 평가(SQA)는 MOS(Mean opinion scores)로 신호에 스칼라 점수를 매기는 방식이 주류였다. 하지만 리스닝 테스트 프로토콜 차이와 청자 간 편차, MOS 유래 이산 평점이 라벨 잡음을 키워 학습이 안정적으로 이뤄지기 어렵다. 이에 따라 pairwise preference(두 신호를 비교)로 넘어가려는 시도가 있었지만, 대부분은 여전히 MOS/평점 기반 감독에서 완전히 벗어나지 못했다.

- **Core Contribution**: 이 논문은 MOS-free(모스 없이) 쌍대 선호 예측을 목표로 PrefSQA를 제안한다. wav2vec 2.0과 WavLM의 의미/음향 표현을 결합하고, uncertainty-aware Bradley-Terry preference logits로 비교의 확신 정도를 반영한다. 또한 impairment attention head와 non-matching-reference(NMR) head를 추가해 국소 열화와 전역 랭킹을 함께 정교화한다.

- **Technical Challenges**: 핵심 난제는 MOS 라벨이 없을 때도 ‘품질 차이’라는 상대 기준을 안정적으로 학습하는 것이다. PrefSQA는 예측 점수와 log variance로 비교 온도(uncertainty-dependent temperature)를 조절해 Bradley-Terry logits를 만들고, 작은 마진 구간에서 발생하기 쉬운 불확실성을 완화하기 위해 온도 클램핑을 적용한다. 더불어 NMR head는 배치 내 in-batch 비교를 통해 임베딩 공간이 더 미세한 순서를 따르도록 보조 손실을 설계했다.

- **Empirical Impact**: 실험에선 5개 preference 데이터셋을 구성하되, MOS-derived는 라벨 잡음의 영향이 커서 개선 폭이 작게 나타났다. 반면 SNR 기반으로 잡음을 통제한 low-noise simulated 데이터(ChiLi)에서는 여러 모델 차이가 확연히 드러나 PrefSQA의 구조적 이득이 관측됐다. 추가로 human preference 데이터(SpeechEval, SpeechJudge)와 unseen IUB-COSINE 평가에서도 성능이 확인되며, 라벨 품질이 모델 개선 가시성을 좌우한다는 메시지를 실증한다.



### IHBench: Evaluating Post-Interruption Recovery in Voice Agents with Structured Workflows (https://arxiv.org/abs/2606.19595)
- **Prior Approaches**: 기존 speech-capable 모델 평가는 barge-in(겹침 중단), endpointing, turn-taking처럼 ‘언제 멈추고/넘겨주느냐’에 집중했습니다. Full-Duplex-Bench 계열, FLEXI, SID-Bench, HumDial 등은 끼어들기 이후의 의미적 대응이나 resume/ respond 분류를 보지만, 정형화된 멀티스텝 워크플로우에서 ‘다음에 무엇을 말해야 하는지’까지는 잘 측정하지 못했습니다.

- **Core Contribution**: 이 논문은 voice agent의 post-interruption recovery를 독립적인 평가축으로 정의하고, 6가지 interruption type에 대해 task fulfillment(업무 진행)와 recovery quality(복구 품질)를 2축으로 채점합니다. 그 결과 IHBench(Interruption Handling Benchmark)를 제안하며, 10개 엔터프라이즈 도메인 state-machine 기반 워크플로우에서 끼어들기 후 “올바른 단계로 재개하는가, 사용자가 말한 내용을 반영하는가, 이미 들은 내용을 다시 말하지 않는가”를 직접 평가합니다.

- **Technical Challenges**: 핵심 난제는 중간 발화(mid-utterance)에 삽입된 interruption이 평가 대상이 되도록 데이터 생성과 채점 기준을 정교하게 통제하는 것입니다. IHBench는 사용자 시뮬레이터가 ‘사용자가 실제로 들은 잘린 구간만’ 반응하도록 누출(leakage)을 막고, per-interruption rubric을 데이터 생성 시점에 함께 만들며, 검증/수정(verify–modify)으로 상태 불일치나 부자연스러움을 제거합니다.

- **Empirical Impact**: 27개 audio-language model 구성을 IHBench로 실험한 결과, closed-weight 모델이 전반적으로 interruption에 더 강인 것으로 나타났습니다(업무 진행 win rate에서 유의한 우위, 대화 길이 증가에 따른 성능 저하가 약 3.3배 완만, 오디오-텍스트 모달리티 격차도 미미). 또한 recovery quality는 interruption type에 크게 의존하며, 특히 filler(backchannel) 유형에서 open-weight 모델의 붕괴가 두드러졌고, IHBench의 recovery quality 축은 AudioMultiChallenge와 비교했을 때 상대적으로 ‘독립적인 능력 축’임이 확인되었습니다. 



### A BART-based approach with hierarchical strategy for Vietnamese abstractive multi-document summarization (https://arxiv.org/abs/2606.19591)
Comments:
          originally written in 2022

- **Prior Approaches**: 베트남어 multi-document abstractive summarization은 초기 연구가 부족하며, 기존에는 그래프 기반(semantic 관계를 잘 잡지만 discourse correlation 같은 추가 정보가 필요)과 계층형(hierarchical) 접근이 주로 쓰였다. 계층형은 문서별 중간 표현을 만든 뒤 이를 합쳐 최종 요약을 생성하지만, 중간 표현이 입력만으로 만들어져 최종 요약과 정보 불일치가 생기기 쉽고 그 결과 환각(hallucination)을 유도할 수 있다. 또한 end-to-end로 처리하기에는 입력 길이가 너무 길어 기존 pretrained seq-to-seq 모델의 토큰 한계가 문제가 된다.

- **Core Contribution**: 이 논문은 계층형 전략을 유지하되, 1단계에서 중간 표현을 만들 때 golden summary를 기준으로 문장을 선택해 두 단계 간 정보 상관을 높이는 간단한 샘플링 전략을 제안한다. 구체적으로 각 문장에 대해 golden summary와의 ROUGE1을 중요도로 계산해 원하는 길이 임계치까지 상위 문장을 고르고, 이를 2단계 요약 생성의 입력으로 사용한다. 추가로 VietnameseMDS 및 ViMs 등에 더해 Multinews를 베트남어로 번역해 약 5만 개 규모의 학습 클러스터를 커뮤니티에 공개한다.

- **Technical Challenges**: 핵심 기술적 어려움은 (1) 장문/다문서 입력에서 정보 손실을 줄이면서 (2) 단계 1의 추출 과정이 단계 2 생성에 필요한 정보를 충분히 제공하도록 만드는 것이다. 저자들은 golden summary 기반 문장 선택으로 단계 간 정보 미스매치를 완화하고, 2단계에서는 문서 순서 변화에 견고하도록 입력 문장 순서를 permute하는 데이터 증강을 적용한다. 또한 BARTPho처럼 상대적으로 긴 입력을 다루는 seq-to-seq 모델을 1·2단계에 각각 fine-tuning해 토큰 제약을 실무적으로 흡수했다.

- **Empirical Impact**: VLSP 2022 공개 테스트셋에서 제안 방법은 ROUGE2-F1 0.2468을 기록해 organizer의 abstractive baseline 및 anchor baseline을 상회했다. 다만 extractive 및 rule-based 기준선보다는 낮아 절대적인 성능 격차는 남았으며, 그럼에도 사람 평가 관찰에서 문법적으로 자연스럽고 핵심 내용을 대체로 커버하는 요약을 생성한다고 보고한다. 한편 Multinews 베트남어 번역 데이터(추가 약 50,286 클러스터)를 공개함으로써 베트남 multi-document summarization의 학습 데이터 부족 문제를 실질적으로 줄이는 영향도 크다.



### FlowFake: Liquid Networks for Audio Deepfake Detection (https://arxiv.org/abs/2606.19579)
Comments:
          Accepted at the Workshop on Learning to Listen: Machine Learning for Audio at ICML 2026

- **Prior Approaches**: 기존 오디오 딥페이크 탐지는 특정 TTS/보이스 클로닝 파이프라인의 스펙트럴 아티팩트를 고정된 프레임 창에서 요약하는 경우가 많습니다. 그 결과, 학습에 없던 vocoder·언어·녹음 조건의 위조를 만나면 탐지기가 구조적으로 붕괴하며 cross-dataset 일반화가 성능 병목이 됐습니다. RawGAT-ST, self-supervised frontends의 fine-tuning, ASR encoder repurposing 등은 모두 out-of-distribution에서 변동성과 성능 저하가 커졌습니다.

- **Core Contribution**: 논문은 실패 원인을 데이터 부족이 아니라 ‘아키텍처-불일치’로 봅니다. 합성 음성의 핵심 위조 흔적은 순간 스펙트럼 값이 아니라 다중 timescale에 걸친 스펙트로-시간 궤적(trajectory) 이상이며, 기존 모델의 고정 창 통계 집계가 이 궤적 정보를 구조적으로 지웁니다. 이를 해결하기 위해 FlowFake를 제안하고, Liquid Time-Constant(LTC) 기반 연속시간 ODE로 궤적 형태 자체를 모델링해 합성 “패러다임”에 불변인 위조 위반을 잡도록 설계합니다.

- **Technical Challenges**: 연속시간 모델에서 가장 큰 과제는 학습 안정성과 수치 오차, 그리고 스펙트럴(약 10ms)·프로소딕(수 초) 단서를 동시에 담는 시간스케일 설계입니다. FlowFake는 누설(leak) 항을 포함한 gradient stable LTC 변형으로 BIBO 안정성과 RK4 적분의 O(dt^4) 오차 상계를 보장하고, 뉴런별 adaptive time constant를 log-파라미터화해 두 시간대 신호를 함께 포착합니다. 또한 tanh synapse 단순화와 Runge-Kutta(4차) 적분으로 50~200 프레임 롤아웃에서도 그라디언트가 안정적으로 유지되게 구성합니다.

- **Empirical Impact**: 4개 데이터셋을 대상으로 한 leave-one-dataset-out 교차 평가에서 FlowFake는 파라미터 34K 수준으로도 강력한 일반화를 보였습니다. 예컨대 FakeOrReal 학습→ASVspoof2019에서 75.29%, MLAAD 학습→ASVspoof2019에서 79.97%를 달성해 RawGAT-ST·Whisper-DF를 모든 평가 쌍에서 앞섰고, SSL Wav2vec2(약 300M급) 대비 매우 작은 모델 크기로도 동급/상회 성능을 보였습니다. 특히 worst pair로 지목된 교차 조건에서도 cross-seed 분산이 낮게 유지돼, 합성 아티팩트의 궤적-기반 구조적 선행(structural prior)이 실험적으로 뒷받침된다는 점이 의미 있습니다.



### Exploring Feature Extraction Technique Parameters for Acoustic Gunshot Classification (https://arxiv.org/abs/2606.19568)
- **Prior Approaches**: 기존 연구는 STFT, log-mel spectrogram, MFCC 같은 대표적 특징 추출을 사용하지만, 각 기법의 핵심 파라미터(FFT 윈도 길이, hop length, mel band 수 등)가 왜 선택됐는지와 일반화 관점의 체계적 검증은 부족했습니다. 또한 상용 탐지 시스템(예: ShotSpotter)은 공개된 내부 구조·정확도 검증 정보가 제한적이며, 도시 환경에서 불꽃놀이·차량 백파이어·공사 소음 등과의 오분류 이슈가 반복적으로 지적됩니다. 즉 “실전 조건에서 맞는 특징/파라미터를 고르는 문제”가 문헌에서 엄밀히 다뤄지지 않았습니다.

- **Core Contribution**: 이 논문은 총 23,000발 이상의 음원(85개 화기, 21종 탄종)으로 구성된 대규모 총합 데이터셋을 바탕으로, 특징 추출 기법과 파라미터 선택이 deep learning 분류 성능의 일반화에 미치는 영향을 체계적으로 벤치마킹합니다. STFT, log-mel, MFCC 3종 기법을 대상으로 12개의 고유 파라미터 세트를 적용하고 ResNet-18로 정확도를 비교해 “올바른 표현 선택”과 “표현별 최적 파라미터 튜닝”의 효과를 분리해 보여줍니다.

- **Technical Challenges**: 문제의 핵심 난관은 짧고 충격적인 gunshot 신호가 실환경에서 잡음·반향·거리 차이 등으로 분해되며, 시간-주파수 표현에서의 해상도(시간 vs 주파수) 편향이 성능을 좌우할 수 있다는 점입니다. 저자들은 FFT 윈도/홉 길이/멜 필터 뱅크 수/윈도 함수(Hann vs Hamming 등) 같은 파라미터를 폭넓게 바꿔가며 실험했고, MFCC의 경우에도 cepstral coefficient 개수(20/30/40)를 포함해 이전에 “기법이 인기라서 쓴다”는 수준에 머물던 선택을 정량화했습니다.

- **Empirical Impact**: 실험 결과 log-mel spectrogram이 최대 top-1 정확도 96.66%로 가장 높았고, STFT는 95.81%, MFCC는 85.24%로 크게 뒤처졌습니다. 또한 특정 기법에서 ‘정답에 가까운 파라미터’를 쓰면 top-1 정확도가 최대 20%까지 개선될 수 있으며, 동일 기법 내에서도 파라미터 최적화로 추가로 최대 4.7%p 향상이 가능함을 보였습니다. 결국 단일 특징 추출이 보편적으로 최선이라는 가정이 성립하지 않으며, 실전 성능을 위해서는 특징 선택과 파라미터 튜닝이 모델 개발의 핵심 단계가 된다는 메시지를 제공합니다.



### GDGU: A Gradient Difference-based Graph Unlearning Method for Cyberattack Localization in Electric Vehicle Charging Networks (https://arxiv.org/abs/2606.19566)
- **Prior Approaches**: 기존 CMAs(Charging Manipulation Attacks) 연구는 주로 변조·이상 탐지에 초점을 맞추고, 각 정거장/세션을 독립적으로 처리해 “어느 버스가 뚫렸는지” 같은 버스 수준 근거를 제공하기 어렵다. GNN 기반에는 연결성을 반영해 localization 성능이 높아졌지만, 정거장(충전전력) 기록이 모델 파라미터에 내재되면 삭제 요청 시 영향 제거가 난제로 남는다. 또한 machine unlearning/graph unlearning은 exact(셔딩 재학습)과 approximate(영향함수 등)로 나뉘며, second-order 근사(inverse Hessian, Hessian-vector product)는 계산·메모리 부담이 커 실사용 삭제 시나리오에 취약하다.

- **Core Contribution**: 이 논문은 EVCS 사이버공격 localization을 “feature-level graph unlearning” 문제로 재정의한다. DSO가 보유한 전압(비가역)과 EVCS 소유 충전전력(가역)을 분리하는 split data ownership을 반영해, 삭제 요청 시 해당 EVCS 버스의 충전전력 특징만 영(0)으로 만들어 영향 제거를 수행한다. 이를 위해 GDGU(gradient difference-based graph unlearning)를 제안하며, 삭제 데이터의 기여를 1차(gradient 차이) 파라미터 보정으로 제거하고, 이후 batch-normalization 재캘리브레이션과 짧은 recovery fine-tuning으로 localization 유틸리티를 복원한다.

- **Technical Challenges**: 핵심 기술 난제는 “삭제해야 할 특징만” 제거하면서도(전압·토폴로지는 유지) GNN의 버스 단위 다중 레이블 localization 성능 저하를 최소화하는 것이다. 2차 방법(GIF/IDEA)은 inverse Hessian 또는 Hessian-vector product 계산으로 정확도는 높을 수 있으나, 대형 그래프에서 메모리와 시간이 급증해 반복 삭제 요청 환경에 부적합하다. GDGU는 원 데이터와 ‘요청된 충전전력 특징만 0으로 만든’ 수정 데이터의 gradient 차이를 이용해 convergence 시점의 삭제 관련 신호만 1차로 추정하고, 추가로 BN 통계 재조정 및 소규모 fine-tuning으로 보정의 부작용을 완화한다.

- **Empirical Impact**: IEEE 34-bus, 123-bus, 8500-node 전력분배망에서 GAT/GCN/GIN 백본 및 누적 삭제 시나리오를 통해 검증했으며, GDGU는 strongest baseline 대비 localization utility를 거의 유지(일부 설정에서는 통계적으로 동률)하면서 forgetting fidelity도 full retraining에 근접한다. 특히 GDGU는 2차 기반 GIF/IDEA 대비 unlearning 속도가 더 빠르고, peak GPU memory도 훨씬 낮아 규모 확장성 측면에서 이점을 보였다. 결과적으로 반복적인 삭제 요청이 잦은 EVCS 운영 환경에서 “권리 제거” 요구를 실용적으로 만족시키는 unlearning 경로를 제시했다.



### Review of Machine Learning Models for Solar Energetic Particle Prediction (https://arxiv.org/abs/2606.19539)
Comments:
          Review Paper, Maine text: 23 pages, References: 5 pages, Appendix: 42 pages

- **Prior Approaches**: 기존 SEP(태양 고에너지 입자) 예측은 물리 기반 시뮬레이션과 경험/통계 모델로 크게 나뉘며, 전자는 계산비용이 크고 후자는 과거 상관관계에 의존하는 한계가 있습니다. 최근에는 ML이 비선형 패턴을 포착할 수 있다는 점에서 주목받았지만, SEP 사건의 희소성·심한 클래스 불균형·서로 다른 관측/전처리/정의로 인해 연구 간 비교가 어렵다는 문제가 남아 있습니다.

- **Core Contribution**: 이 논문은 SEP 예측을 위한 ML 모델들을 24개 접근법으로 정리하고, 아키텍처(모델 유형), 입력 데이터(관측 형태·물리량·시계열 길이 등), 출력(분류/회귀/확률, 예측 트리거 방식, 예측 창)을 기준으로 체계적으로 분류합니다. 또한 사용된 데이터셋과 레이블링 방식, 성능과 한계를 함께 논의해 ‘향후 연구의 좋은 관행(good practices)’을 제안하는 것이 핵심 기여입니다.

- **Technical Challenges**: 주요 기술 난관은 (1) SEP 사건 자체의 희귀함과 특히 고에너지 구간의 데이터 부족, (2) SEP vs 비SEP의 극단적 불균형, (3) 양성 레이블이 필요한 지도학습(suppervised) 구조에서 서로 이질적인 관측(예: 양성자 플럭스, 태양 이미지, 플레어/활동영역(AR) 정보 등)을 통합해야 한다는 점입니다. 저자들은 모델 비교를 위해 출력 정의와 검증 셋업이 통일되지 않는 현실을 지적하며, 데이터 조화와 공유 검증 자원, 도메인 지식 반영, 데이터 증강 등으로 더 신뢰도 높은 개발·평가가 가능하다고 제안합니다.

- **Empirical Impact**: 리뷰 결과, 다수의 모델이 비교적 낮은 복잡도(트리/앙상블, 간단한 신경망) 범주에 머물러 있으며, 예측 성능 차이는 ‘모델 복잡도’보다 ‘더 풍부한 물리 입력 데이터’와 ‘데이터 품질/정의 일관성’에 더 좌우되는 경향이 관찰됩니다. 다만 연구마다 예측 대상(온셋 시간, all-clear 등)과 예측 창, 검증 지표·셋업이 달라 직접적인 성능 비교가 제한적이어서, 향후 운영(operational) 수준 신뢰성을 확보하려면 표준화된 출력/평가 체계와 데이터 정비가 중요하다는 메시지를 남깁니다.



### PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models (https://arxiv.org/abs/2606.19534)
Comments:
          Code available at this https URL

- **Prior Approaches**: 기존 MLLM은 시각 이해에 강점을 보이지만, 대부분 autoregressive(AR) 생성 방식이라 여러 영역을 한 번에 처리할 때 비효율적입니다. 특히 region captioning에서는 각 마스크를 순차적으로 디코딩하고 토큰도 단계적으로 생성해, 영역 수가 늘수록 지연(latency)이 거의 선형으로 커집니다. diffusion language model(DLM)은 병렬 토큰 생성 잠재력이 있지만, fine-grained localized perception에 그대로 확장하는 것은 품질과 병렬성 모두에서 검증이 부족했습니다.

- **Core Contribution**: 이 논문은 diffusion 기반 멀티모달 언어모델 PerceptionDLM을 제안해, 단일 denoising 과정에서 여러 masked region의 캡션을 병렬로 생성하도록 설계했습니다. PerceptionDLM-Base는 discrete diffusion multimodal baseline으로 visual instruction tuning을 통해 지각(perception) 품질을 먼저 끌어올리고, 이후 region-aware 구조(모양을 구분하는 region prompting과 영역 간섭을 줄이는 structured attention masking)를 얹어 병렬 region captioning을 가능하게 했습니다. 또한 ParaDLC-Bench를 통해 캡션 품질과 추론 효율을 동시에 평가하는 틀을 마련했습니다.

- **Technical Challenges**: 핵심 난제는 DLM이 한 번에 여러 토큰을 복원하는 특성상, 동시에 생성되는 서로 다른 영역 간 ‘엔탱글먼트/간섭’을 어떻게 억제하느냐입니다. 논문은 영역별로 RoI-aligned feature replay와 learnable region embedding을 주고, 영역 토큰의 attention을 전역 비주얼 토큰/공유 프롬프트/해당 영역 RoI 토큰 및 자기 영역 캡션 스팬으로 제한하는 structured attention masking으로 영역 독립성을 강제했습니다. 더불어 효율을 위해 멀티해상도 타일링과 학습 단계별(정렬→중간지식→지시따르기→고품질 정제) 파이프라인을 구성했습니다.

- **Empirical Impact**: 실험에서 PerceptionDLM-Base는 16개 멀티모달 벤치마크 중 15개에서 기존 diffusion VLM 대비 우위를 보이며, 특히 ParaDLC-Bench의 multi-region 설정에서 평균 정확도 62.4%로 baseline 대비 큰 폭(예: LLaDA-V 35.2%)의 향상을 보였습니다. 효율 측면에서는 병렬 디코딩 덕분에 multi-region 상황에서 지연이 영역 수에 따라 급격히 늘지 않고, heavy workload(예: 4 masks)에서는 throughput이 최대 3.44배, 단일 이미지 latency가 10.04초→2.92초로 감소했다고 보고합니다. 결과적으로 diffusion 기반 멀티모달 모델이 fine-grained 시각 지각을 ‘병렬’로 확장할 수 있음을 실증하며, 실사용 관점의 처리량 개선 가능성을 제시합니다.



### A Tool for the Synthesis of Adaptive Probabilistic Processors Based on the Ising Mod (https://arxiv.org/abs/2606.19533)
Comments:
          ACM/IEEE/SBC/SBMICRO Symposium on Integrated Circuits and Systems Design 2026

- **Prior Approaches**: 기존 Ising Machine 연구는 Max-Cut, SAT, graph coloring 같은 NP-hard 조합최적화를 Ising 모델로 매핑한 뒤, Gibbs Sampling, Simulated Annealing (SA), Simulated Quantum Annealing (SQA), cluster-based 업데이트 등 다양한 갱신 동역학으로 해를 찾는 흐름을 따릅니다. 하지만 대부분의 아키텍처가 p-bits 수(자원)와 업데이트 알고리즘을 고정해 문제 구조(연결성, coupling 밀도, 멀티모달/좌절 정도)에 따라 비효율이 생기기 쉽습니다. 그 결과, 어떤 문제군에서는 잘 동작해도 다른 문제군에서는 수렴 속도나 해 품질이 떨어지는 설계 격차가 남았습니다.

- **Core Contribution**: 이 논문은 조합최적화 인스턴스를 Ising Hamiltonian으로 자동 매핑하고, 문제 특성(크기·토폴로지)에 맞춰 필요한 p-bits(확률 요소) 개수를 추정하는 ‘합성(synthesis) 툴’을 제안합니다. 또한 Gibbs Sampling/SA/SQA/cluster-based 중 에너지 지형의 탐색 난이도를 고려해 가장 적합한 업데이트 알고리즘을 자동 선택하도록 만들어, 고정 설정 대비 유연한 탐색을 목표로 합니다. 즉, 자원 인식(resource-aware) + 동역학 자동 선택을 한 흐름으로 통합한 것이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 문제마다 에너지 랜드스케이프(좌절·로컬 미니마·코플링 밀도)가 달라, 단일 업데이트 규칙이나 단일 p-bits 설정이 항상 최적인 선택이 아니라는 점입니다. 이를 해결하기 위해 프레임워크는 인스턴스에서 구조 디스크립터를 추출해 p-bits 수를 적응적으로 할당하고, 동일한 Ising 매핑 위에서 서로 다른 업데이트 동역학을 공정하게 비교/선택하도록 실행 엔진을 설계했습니다. 더불어 실행 중 best energy, 수렴 반복, 실행 시간, 비용 관련 지표까지 함께 기록해 자원·품질 트레이드오프를 재현 가능하게 분석합니다.

- **Empirical Impact**: TSP, graph coloring, SAT, matching, segmentation, Max-Cut 등 다양한 벤치마크에서 업데이트 전략의 상대 성능이 문제 구조에 따라 달라짐을 보여주며, 예를 들어 graph coloring/matching/segmentation에서는 SQA가 더 낮은 에너지를 달성하는 경향이 관찰됩니다. 반면 SA는 여러 인스턴스에서 균형 잡힌 성능을 보이며, Gibbs Sampling은 상대적으로 효율적이어서 문제 난이도에 따라 선택 가치가 달라집니다. 또한 낮은 에너지(해 품질)일수록 실행 시간이 늘어나는 경향이 확인되어, 제안된 적응형 선택이 ‘해 품질 vs 계산/에너지 비용’ 트레이드오프를 맞춰주는 방향으로 의미가 큽니다. 결과적으로 하드웨어(MTJs/p-bits) 제약을 염두에 둔 미래 설계에서, 알고리즘 평가와 자원 합성을 함께 지원하는 도구로 활용될 잠재력이 제시됩니다.



### Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices (https://arxiv.org/abs/2606.19528)
Comments:
          Hassan Dbouk and Matthias Reisser contributed equally to this work

- **Prior Approaches**: 기존에는 LoRA 같은 PEFT로 학습 비용을 줄이지만, 실제 peak 메모리는 역전파 시 forward activation 저장과 optimizer 상태가 지배하며 긴 컨텍스트에서 OOM이 쉽게 발생했다. 서버/데이터프라이버시 제약을 완화하려는 server-assisted side-tuning, zero-order 최적화(MeZO) 등도 제안됐지만 여전히 backprop을 대체하기 어렵거나(또는 하이브리드로 프라이버시·지연 손실이 생길 수 있다) 온디바이스 적용에 한계가 있었다. QLoRA, gradient checkpointing, offloading 같은 확장들은 도움이 되지만, LoRA 온디바이스 fine-tuning의 핵심 병목(특히 LM head의 vocab-의존 메모리)을 함께 공략하는 통합 설계는 부족했다.

- **Core Contribution**: 이 논문은 LoRA fine-tuning에서 peak memory를 줄이기 위해 서로 직교(orthogonal)한 네 가지 기법을 패키지로 제안한다: (1) base model quantization + on-the-fly dequantization, (2) selective activation rematerialization과 디스크/오프칩 offloading을 결합한 메모리 효율 checkpointing, (3) semantically relevant token subset으로 softmax 근사, (4) instruction fine-tuning에서 비학습 토큰을 건너뛰는 logits masking. 목표는 모델 품질을 유지하면서도 소비자급 하드웨어에서 허용 가능한 peak memory 범위로 fine-tuning을 끌어내리는 것이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 peak memory가 단순 파라미터 크기가 아니라 activation과 특히 LM head의 logit/softmax 계산에서 vocab 크기에 비례해 폭증한다는 점이다. 저자들은 이를 위해 양자화된 가중치를 필요할 때만 dequantize해 FP32 가중치 상주를 피하고, 노드 경계 기반 checkpointing으로 캐시할 activation만 선택적으로 유지하며, softmax는 사전학습 임베딩의 의미적 유사도를 이용해 “effective vocabulary”만 계산하도록 줄였다. 또한 instruction fine-tuning의 masked causal LM에서 non-trainable 토큰에 대한 logits 계산 자체를 생략해 LM head의 처리 길이를 단축함으로써 peak을 추가로 낮췄다.

- **Empirical Impact**: Llama-3.2 3B와 Qwen-2.5 3B에서 병행 적용 시 peak memory가 최대 26×~28×까지 감소했으며, 예로 Llama-3.2 3B의 2048 토큰 fine-tuning은 26.20 GB에서 1.02 GB 수준까지 내려갔다. 더 긴 컨텍스트(예: 16K 토큰)도 6.95 GB로 가능해져, 기존에 불가능하던 온디바이스/엣지 개인화 시나리오의 실현 가능성을 크게 넓힌다는 점이 의미 있다. 특히 Top-k 의미 기반 softmax 근사와 logits masking이 출력층 메모리 병목을 직접 겨냥해 장문에서 효과가 유지된다는 경험적 근거를 제시했다.



### Concept Flow Models: Anchoring Concept-Based Reasoning with Hierarchical Bottlenecks (https://arxiv.org/abs/2606.19489)
- **Prior Approaches**: Concept Bottleneck Models(CBMs)은 먼저 사람이 이해 가능한 개념을 예측한 뒤 이를 이용해 최종 라벨을 결정함으로써 해석가능성을 높였다. 그러나 VLM 기반 post-hoc CBMs는 수작업 개념 주석을 줄여도, concept embedding 안에 들어있는 부가 정보가 linear layer에 의해 스퓨리어스 상관으로 악용되며 정보 누출(information leakage)이 발생한다. 특히 개념 수가 임베딩 차원에 가까워지면 랜덤 개념만으로도 분류 성능이 유지되는 현상이 보고되어 해석가능성이 크게 약화된다.

- **Core Contribution**: 이 논문은 flat bottleneck을 계층적 계통도로 바꾼 Concept Flow Models(CFMs)를 제안한다. CFMs는 CLIP 임베딩으로 클래스 간 의미 계층을 만든 뒤, 각 내부 노드에 LLM이 생성한 개념을 배치하고 입력이 루트-리프까지 개념 기반 경로를 따라가도록 설계해 예측 근거를 단계적으로 추적 가능하게 만든다. 그 결과 성능은 flat CBM과 맞추면서도, 무관 개념의 영향은 경로 밖에서 구조적으로 차단되어 정보 누출을 완화한다.

- **Technical Challenges**: 핵심 난제는 (1) 개념 수를 늘려 성능을 확보하려 해도 정보 누출이 커지는 문제를 계층 구조로 억제하면서, (2) 노드별 국소 개념 부분공간(local concept subspace)에서의 분류 가능성을 유지하는 것이다. 논문은 agglomerative clustering(Ward’s linkage)으로 트리 계층을 만들고 pruning으로 과도한 분할을 제어한 뒤, 각 노드에서 concept relevance를 cosine similarity 기반으로 고르고 Lasso-regularized multi-class logistic regression으로 중요 개념을 선택한다. 이후 temperature-scaled probabilistic tree traversal로 각 노드의 differentiable 경로 확률을 학습하며, 편향 항을 더해 경로별 신뢰도까지 보정한다.

- **Empirical Impact**: 다양한 벤치마크에서 CFMs는 flat CBM과 유사한 예측 정확도를 달성하면서, 랜덤·무관 개념을 통한 정확도 상승을 더 강하게 억제하는 것으로 분석된다. 또한 Number of Effective Concepts(NEC)류의 관점에서 클래스별 사용 개념 수(유효 개념 수)가 구조적으로 줄어 정보 누출 저감이 실증된다. 나아가 트리의 stepwise decision flow는 모델 추론을 감사(audit)하고 해석 가능한 근거를 제공하는 방식으로, 안전·신뢰성 요구가 있는 비전 의사결정 분야에 실질적 의미가 있다.



### Can In-Context Learning Support Intrinsic Curiosity? (https://arxiv.org/abs/2606.19476)
- **Prior Approaches**: 기존 intrinsic curiosity는 agent가 더 배우는 양을 intrinsic reward로 환기하며 탐험을 유도해 왔다. 대표적으로 Bayesian information gain(BIG)은 이론적으로 최적이지만, 실제 계산에는 환경 동역학의 명시적 모델링과 비싸고 비가역적인 Bayesian inference가 필요하다. 한편 learning progress는 예측 오차 개선으로 근사하지만, 이를 매 궤적에서 반복적인 gradient descent inner loop로 재평가해야 하는 병목이 생긴다. Surprisal은 구현은 쉽지만 aleatoric noise(예측 불가한 잡음)까지 보상해 ‘noisy TV’ 문제를 겪는다.

- **Core Contribution**: 본 논문은 sequence model의 in-context learning(ICL) 능력을 이용해, 업데이트 없는(world model 갱신 없는) 즉시식 예측 기반 intrinsic reward를 구성할 수 있는지(그리고 BIG에 얼마나 근접하는지)를 수학적으로 분석한다. 특히 ICL이 prior predictive를 posterior predictive처럼 동작한다는 가정 하에, ICL의 prediction error와 context 조작만으로 learning progress 계열을 BIG에 연결하려 시도한다. 그 결과 general Markov decision processes(MDPs)에서는 unbiased하게 BIG을 재현할 수 없음을 증명하고, 대신 Bayesian Experimental Design(BED) 같은 비시간(또는 조건부 독립) 구조에서는 수렴/근사가 가능함을 보인다.

- **Technical Challenges**: 핵심 난관은 ICL 인터페이스가 제공하는 ‘prediction error/로그우도 형태’만으로는 BIG이 요구하는 log-ratio의 정확한 편향-분해(signal/abductive/residual)에서 residual과 abductive 항을 일반 설정에서 제거하기 어렵다는 점이다. 논문은 일반 BAMDP에서 finite horizon 내 예측 기반 reward는 BIG의 unbiased estimator가 될 수 없고, 구현 가능 여부와 무관하게 체계적 편향이 남는 구조적 이유를 이론으로 제시한다. 해결책으로 BED 환경에서는 abductive bias가 구조적으로 소거되고, 새로 제안한 rsum(trajectory 끝까지를 텔레스코핑하는 NDIGO 계열) 같은 ICL-구현 가능한 reward가 T가 커질수록 BIG에 점근적으로 수렴함을 보인다.

- **Empirical Impact**: 이론 결과를 검증하기 위해 continuous 및 symbolic 환경에서 제어된 실험을 수행해, ICL 기반 curiosity reward로 학습된 탐험 정책이 최적에 가까운 방식으로 정보를 수집함을 보여준다. 특히 일반 설정에서는 BIG에 대한 unbiased 근사가 막히지만, BED 조건에서는 제안 프레임워크가 실제로 학습 가능한 탐험 보상을 제공한다는 점이 관찰된다. 결과적으로 ‘intrinsic curiosity의 계산 병목’을 in-context learning의 amortized 예측으로 완화할 수 있음을 보여주며, active learning/experimental design 계열과 결합 가능성을 확장한다.



### Secure Coding Drift in LLM-Assisted Post-Quantum Cryptography Development: A Gamified Fix (https://arxiv.org/abs/2606.19474)
Comments:
          Accepted for 2026 SIGIR Workshop on Vulnerabilities in Generative Systems for Information Retrieval track

- **Prior Approaches**: PQC 마이그레이션은 알고리즘 교체만으로 끝나지 않고, constant-time 실행·side-channel 저항·정확한 파라미터링 같은 구현 조건을 모두 만족해야 한다는 점에서 보안 공학 난도가 높다. 기존 연구와 도구들은 주로 생성 코드의 정적 취약점 탐지에 초점을 맞추지만, LLM을 장기간 쓰며 개발자의 검증 습관이 약해지는 ‘행동 변화’는 충분히 다루지 못했다. 그 결과 보안 문제를 코드 한 번의 오류로만 보기 쉬워 누적 위험을 놓칠 수 있다.

- **Core Contribution**: 이 논문은 PQC 개발에서 LLM 의존이 장기간 누적되며 보안 코딩 역량이 점진적으로 악화되는 사회기술적 취약점 모델 Secure Coding Drift in PQC(SCD-PQC)를 제안한다. 핵심은 “기능적으로는 맞아 보이지만 constant-time·난수·파라미터 같은 보안 가정을 깨는 코드”가 생성되는 것뿐 아니라, 반복된 수용이 개발자의 보안 사고와 검증 행동을 약화한다는 점을 보안 리스크의 중심으로 둔 것이다. 또한 LLM을 수동 생성기에서 ‘보안 co-pilot’로 재구성하는 방향을 제시한다.

- **Technical Challenges**: SCD-PQC를 실제로 완화하려면, (1) 미세한 암호 구현 결함을 LLM 생성물만으로는 놓치기 쉬운 점, (2) LLM-as-a-Judge가 맥락을 잘못 판단할 수 있는 점, (3) 시간에 걸친 행동 변화(검증 빈도·깊이 감소)를 측정·유도해야 한다는 점이 난제다. 논문은 rule-based 정적 분석과 LLM-as-a-Judge를 결합해 timing leakage·insecure randomness·파라미터 오용 같은 결함을 다층으로 평가하고, 이를 기반으로 drift indicator와 보안 점수를 산출해 즉각적·해석 가능한 피드백 루프를 만든다. 여기에 gamification을 더해 무검증 수용을 불리하게 만들고, 취약점 발견·보안 수정·대안 검증을 반복적으로 강화한다.

- **Empirical Impact**: 실험은 일반 LLM 기반 개발 그룹과 제안 프레임워크 적용 그룹을 비교하는 controlled study 및 longitudinal 분석으로 설계되며, 취약점 탐지율·secure code acceptance rate·수정까지 걸린 시간·행동 일관성 등을 정량·정성으로 본다. 특히 개입 그룹에서 시간이 지날수록 SCD-PQC가 감소하는지(예: drift indicator 저하)를 추적해 누적 위험 완화 효과를 검증하려는 계획이다. PQC 안전성처럼 ‘기능 동작은 맞는데 보안이 깨지는’ 영역에서, 개발자 행동을 겨냥한 개입이 장기적으로 품질과 신뢰성에 미치는 의미 있는 개선 가능성을 제시한다.



### Scaling Generative Foundation Models for Chest Radiography with Rectified Flow Transformers (https://arxiv.org/abs/2606.19460)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 흉부 X-ray(CXR) 합성 모델은 환자 하위집단, 의료기관, 촬영 조건이 달라지면 일반화 성능이 급격히 떨어져 실제 임상 활용도가 낮다는 문제가 지적돼 왔다. 또한 생성 품질이 제한적이어서 다양한 임상 데이터셋을 안전하게 확장하거나 진단 모델의 강건성을 평가하기 어려웠다.

- **Core Contribution**: 이 논문은 CXR을 위한 최초의 ‘생성형 foundation model’을 대규모 스케일로 처음부터 학습해, 생성과 편집을 모두 제어 가능하게 만든다는 점이 핵심이다. 1.3B+ 파라미터 규모 모델을 약 1.2M 방사선 이미지와 임상 전문가 유도 메타데이터(총 1.6T tokens)로 학습해, 인구통계·촬영 뷰·병리 유형 전반에 걸친 조건부 생성/편집을 지원한다.

- **Technical Challenges**: 진단에 쓰일 정도의 ‘고충실도’ 합성을 위해서는 도메인 특화 생성 안정화와, 메타데이터 조건을 인과적으로 결합해 편집 일관성을 유지하는 설계가 필요했다. 이를 위해 RadVAE(EDM2 기반 변형)로 잠재표현을 구성하고, flow 기반 transformer/U-Net 백본에 RMS-Norm, SwiGLU 같은 학습 안정화 업그레이드를 적용했으며, 메타데이터를 SCM(인과 그래프)로 모델링해 ODE 기반(continuous-time flow) 추론과 반사실/카운터팩추얼 생성을 수행하도록 구성했다.

- **Empirical Impact**: 평가에서는 FID 외에도 방사선/자기지도 표현공간에서의 FDD, KID/KDD, Precision/Recall/Density/Coverage 등으로 분포 충실도와 다양성을 폭넓게 측정했고, 생성-실제 이미지 간 시각적 구분이 임상 전문가 수준에서 거의 불가능하다고 보고했다. 더 나아가 편집 시 환자 identity preservation을 SSIM과 Rad-DINO/DINOv3 기반 perceptual distance, 속성별 임베딩 유사도로 정량화해 ‘데이터셋 다양화’와 ‘진단 모델 강건성 시험’에 의미 있는 기반을 제시한다.



### Playful Agentic Robot Learning (https://arxiv.org/abs/2606.19419)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 Code-as-Policy(CaP) 로봇은 언어/멀티모달이 코드를 작성·실행하고, 피드백을 받아 반복 수정하는 에이전틱 구조를 갖추고 있지만 주로 ‘지시된 과제’에 반응해 학습합니다. 또한 재사용 가능한 스킬을 저장하더라도 그 획득이 외부 과제 발생 이후에 시작되는 경우가 많아, 선제적이고 지속적인 스킬 누적이 제한됩니다.

- **Core Contribution**: 이 논문은 Playful Agentic Robot Learning으로, 하류(downstream) 과제가 들어오기 전 로봇이 ‘자기 주도 놀이’로 스킬을 선학습하고 이후 과제를 푸는 설정을 제안합니다. 이를 위해 RATs(Robotics Agent Teams)를 도입해 놀이 중에 탐색 과제 생성-코드 실행-진행 검증-실패 진단-재시도-성공 실행의 코드 스킬 라이브러리 증류까지 하나의 루프로 구성합니다. 테스트 시에는 이 라이브러리를 고정(frozen)해 새 과제에 관련 스킬을 가져와 성능을 높입니다.

- **Technical Challenges**: 놀이가 유용한 스킬 학습으로 이어지려면 과제 수준 성공/실패만으로는 부족해, 어떤 하위 단계가 막혔는지와 저장할 동작 단위를 촘촘히 식별해야 합니다. RATs는 계획/검증/진단 에이전트를 다단으로 배치해 step-level 피드백을 밀도로 제공하고, 실패 유형을 메모리에 축적한 뒤 재시도로 국소 병목을 보완(서브 에이전트 분리 연습 포함)합니다. 또한 ‘너무 쉽지도 너무 어렵지도 않은’ 과제를 고르기 위해 객체-스킬 조합의 novelty와 Wilson-bounded success 기반 learnability frontier를 곱하는 Goldilocks 규칙으로 놀이 과제를 선택합니다.

- **Empirical Impact**: 실험에서 RATs는 LIBERO-PRO와 MolmoSpaces 모두에서 CaP-Agent0 및 VLA 기반 대조군 대비 held-out downstream 성능을 크게 개선합니다(각각 +20.6, +17.0 percentage-point). 특히 놀이로 얻은 스킬이 환경 밖으로도 전이되며, LIBERO-PRO에서 학습한 스킬 라이브러리를 RoboSuite에 plug-in 했을 때 CaP-Agent0 대비 +8.9점을 기록합니다. 더 나아가 모델 fine-tuning 없이 실제 로봇 과제에도 스킬을 직접 재사용해 +8.8점을 보고해, ‘놀이 기반 코드 스킬 라이브러리’가 에이전틱 로봇 성능을 finetuning 없이 끌어올리는 실용적 경로임을 시사합니다.



### JustDiag!: A Diagnostic Justification Engine for Accountable Root Cause Analysis (https://arxiv.org/abs/2606.19407)
- **Prior Approaches**: 전통적인 AIOps/AI 기반 RCA는 fault localization, dependency analysis, causal ranking처럼 원인 후보를 찾는 데 초점을 둬 왔습니다. 최근 LLM 기반 RCA는 RCAagent, Flow-of-Action, multi-agent 방식 등으로 서사형 보고나 조사 유연성을 높였지만, 대체 설명·모순·잔여 불확실성이 어떻게 처리됐는지까지 “감사 가능한 산출물”로 내보내는 평가는 제한적이었습니다. 또 many evaluations가 최종 답변만 중심이라, 고위험 상황에서 요구되는 책임성(accountability)을 충분히 측정하기 어렵다는 문제가 제기됩니다.

- **Core Contribution**: 이 논문은 RCA를 ‘최종 답 생성’이 아니라 ‘진단 정당화(diagnostic justification) 산출물’로 구성하는 프레임을 제안합니다. JustDiag는 증거–발견–가설–검증 claim–모순 충돌–다음 점검을 그래프로 유지하며, resolved처럼 종료 이유도 함께 기록해 연산 과정 자체를 검토 가능하게 만듭니다. 즉, 유창한 결론만이 아니라 왜 그 결론에 도달했는지의 구조화된 근거를 책임성 증거로 제공하는 데 목적이 있습니다.

- **Technical Challenges**: 핵심 기술 난제는 경쟁 가설들 사이에서 어떤 근거가 무엇을 지지/반박하는지, 그리고 아직 부족한 구별 증거가 남아 있을 때는 어떻게 “닫지 않음”을 정당화하느냐입니다. JustDiag는 claim 단위 adjudication으로 support/contradict/insufficient을 판정하고, 증거 grounding과 hypothesis competition을 통해 충돌을 누적·관리한 뒤 termination state를 resolved, provisionally_resolved, need_more_evidence, stalled로 구분합니다. 또한 claim coverage와 충돌·누락 신호를 이용해 종료를 대화 멈춤 규칙이 아닌 상태 기반 판정으로 연결합니다.

- **Empirical Impact**: 66건의 실제 산업 incident에서 JustDiag는 matched no-DJ 대조군 대비 Outcome Score 51.0→57.7, Process Score 44.0→50.5로 개선됐고, 종료 완료율은 65/66→62/66으로 소폭 감소했습니다. 이는 더 일찍 확정하기보다 근거가 충분치 않으면 non-closure를 “조정(calibrated)된 방식”으로 유지함으로써 과정의 책임성을 높인 패턴으로 해석됩니다. 또한 ablation에서 evidence grounding과 claim adjudication 제거가 Process Score를 크게 훼손해, 책임성 향상의 주된 동력이 어떤 메커니즘인지 실증적으로 확인했습니다.



### VERITAS: Verifier-Guided Proof Search for Zero-Shot Formal Theorem Proving (https://arxiv.org/abs/2606.19399)
- **Prior Approaches**: 기존 LLM 기반 포멀 프로버는 LLM이 전술(tactic)을 생성하면 verifier가 통과/실패만 반환하는 경우가 많아, 구문 오류·타입 불일치·부분 목표 진척 같은 풍부한 신호가 탐색에 제대로 반영되지 못했다. 그 결과 multi-step에서 한 단계의 효과에 다음 단계가 의존하는 “구조적 실패”를 재학습하거나, 동일한 오류 유형을 반복해 낭비하기 쉽다. 보완으로 검색과 결합하는 방법들이 있었지만, verifier의 구조화된 중간 피드백을 세대 생성 의사결정까지 되돌려 보내는 설계는 상대적으로 드물었다.

- **Core Contribution**: VERITAS는 zero-shot 방식에서 Lean verifier의 구조화된 피드백(구문/타입/부분 목표/완료)을 모든 생성 의사결정에 라우팅하는 프레임워크를 제안한다. Best-of-N(Phase 1)로 생성·검증을 먼저 수행하되, 실패한 정리만 대상으로 Critic-guided MCTS(Phase 2)를 돌려 verifier가 준 실패 정보를 명시적 negative example로 다시 주입한다. 또한 Phase 1에서 성공한 정리는 보존되도록 내부 monotonicity 보장을 걸어 Phase 2의 추가 성과를 탐색 기여로 귀속할 수 있게 했다.

- **Technical Challenges**: 핵심 난제는 verifier가 내는 촘촘한 오류/진척 정보를 어떻게 “탐색 가치 함수”와 “전술 생성”에 동시에 효과적으로 주입할지였다. VERITAS는 (1) 실패 전술 텍스트와 Lean 에러 메시지를 negative 예시로 Tactician 프롬프트에 포함하고, (2) Critic value를 MCTS에 결합하되 구문·타입·부분 목표 신호를 별도 가중치로 반영해 부분 진척도 신뢰성 있게 평가한다. 더 나아가 여러 후보를 매번 개별 Lean 호출로 검증하지 않고 batched Lean validation으로 검증 비용을 O(K)에서 O(1) 수준으로 줄여 탐색을 실용화했다.

- **Empirical Impact**: miniF2F에서 VERITAS는 40.6%(99/244)로 Best-of-5(36.9%)와 Portfolio(26.2%)를 웃돌며, Phase-wise 분석은 flat sampling 예산만으로 도달 못하는 11개 정리를 MCTS 단계가 추가로 해결했음을 보여준다. 특히 VERITAS-CombiBench(55개)에서는 Best-of-5(1.8%)가 Portfolio(3.6%)보다 낮았는데, VERITAS는 7.3%로 상승하며 “올바른 lemma 이름을 verifier 오류로부터 반복적으로 복구”하는 효과를 입증했다. 이는 deterministic verifier의 structured intermediate signal을 inference-time 상태로 환류하는 것이 단순 샘플링 증가보다 더 큰 이득이 될 수 있음을 실험적으로 시사한다.



### Execution-bound advisory automation for agentic AI: a reproducible AIBOM-driven CSAF-VEX framework (https://arxiv.org/abs/2606.19390)
- **Prior Approaches**: 기존 취약점 대응은 SBOM 또는 단편적인 보안 권고를 중심으로 하되, 실제 실행 환경과 런타임 관측을 충분히 묶지 못하는 경우가 많았습니다. 또한 정적 분석과 관측 증거가 섞여도 결과를 일관되게 재현하거나 서명·검증까지 자동화하는 워크플로가 부족했습니다. 그 결과 VEX 같은 문서가 만들어져도 “실제로 해당 환경에서 활성화되는지”를 신뢰성 있게 연결하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 프로토콜 기반 프레임워크로 SBOM과 AIBOM 아티팩트를 결정적 환경 캡처(deterministic environment capture) 및 구조화된 런타임 텔레메트리(structured runtime telemetry)에 바인딩합니다. 선언된 아티팩트, 관측된 activation conditions, 강제된 실행 정책을 결합해 exploitability를 산출하고, 그 근거를 바탕으로 CSAF VEX advisories를 정적으로/동적으로 함께 생성합니다. 생성된 VEX는 암호학적으로 서명되며, 결정적 재생(deterministic replay)으로 검증 가능한 형태로 제공됩니다.

- **Technical Challenges**: 핵심 기술 과제는 “정적 증거”와 “런타임 관측”이 서로 다른 조건에서 생기는 불일치를 줄이면서도, 결과가 재현 가능하도록 만드는 것입니다. 논문은 결정적 환경 캡처와 런타임 텔레메트리를 구조화해 activation conditions를 추적하고, 이를 실행 정책과 결합해 일관된 exploitability 계산 및 VEX 생성으로 이어지게 했습니다. 또한 서명·검증 체계를 붙여 데이터 무결성과 검증 가능성을 확보했습니다.

- **Empirical Impact**: 평가는 약 10,000개 컴포넌트 엔트리를 사용하고, 합성 Agentic AI 워크로드에서 50~5,000개 컴포넌트를 다루며 OSV, GitHub Advisory, KEV, EPSS 같은 공개 데이터셋을 포함해 수행됐습니다. 정적·동적 근거를 결합한 VEX 생성과 결정적 재생 검증이 실험 전반에서 동작함을 보였고, 실행 조건에 따른 취약점 활성화 설명의 신뢰성을 높이는 방향성을 제시합니다. 보안 문서화가 “선언만 하는 단계”를 넘어 실제 환경에서의 검증 가능한 근거 중심으로 확장될 수 있다는 점에서 분야에 의미가 있습니다.



### Interpretable and Verifiable Hardware Generation with LLM-Driven Stepwise Refinemen (https://arxiv.org/abs/2606.19387)
- **Prior Approaches**: 기존 RTL 생성 LLM 에이전트들은 reflection, memory, grounding 같은 기법으로 단계적 생성/수정은 시도하지만, 대체로 전체 하드웨어를 한 번에(또는 덜 구조화된 중간 단계 없이) 다루는 성격이 강했습니다. 그 결과 중간 설계 결정이 해석 가능하지 않아 검증 부담이 커지고, 다운스트림 설계 도구와의 통합도 어렵다는 한계가 있었습니다. 또한 HLS는 추상 모델에서 RTL을 도출하지만, 요구사항 의도가 완전히 담기지 않은 입력에는 적용이 어렵고 추가 추상화는 통제력과 검증 복잡도를 악화시킬 수 있습니다.

- **Core Contribution**: 이 논문은 LLM의 창의성과 지식을 formal methods의 수학적 엄밀함으로 “교차 결합”해, 자연어 요구사항을 correct-by-construction RTL로 바꾸는 하드웨어 생성 프레임워크를 제안합니다. 사용자는 먼저 자연어(표/그림 포함)를 형식 명세로 자동 변환하고, 이후 미리 정의된 refinement rule을 반복 적용해 추상 명세를 구체 코드로 단계적으로 치환합니다. 각 규칙은 적용 조건이 만족되면 이전 버전을 구현하는 것이 보장되도록 설계되어, LLM의 hallucination 위험을 구조적으로 억제합니다.

- **Technical Challenges**: 핵심 난제는 (1) RTL의 동시성/시간 의존성을 자연어 생성 과정과 정합시키는 것, (2) LLM이 내린 결정을 수학적으로 검증 가능한 형태로 고정하는 것, (3) 단일 패스 생성의 맹점(맥락 희석·디버깅 어려움)을 피하면서 탐색 실패(dead end)에도 복구하는 것입니다. 이를 위해 논문은 하드웨어 요구를 first-order logic + duration-bounded temporal operator로 표현하는 통합 명세 언어와, process 단위로 분해·정제하는 refinement calculus를 구성해 에이전트가 규칙 선택→변환→검증의 흐름을 유지하도록 했습니다. 또한 last few steps를 되돌려 다른 경로를 탐색하는 백트래킹 메커니즘과, Dafny 기반 SMT 검증으로 중간 단계에서의 correctness를 함께 확인합니다.

- **Empirical Impact**: 평가는 VerilogEval benchmark suite에서 수행되며, 제안한 에이전트가 설계 명세로부터 신뢰 가능한 RTL 구현을 일관되게 생성하고 효율적으로 동작함을 보여줍니다. 무엇보다도 “중간 단계가 규칙과 명세로 정렬되는 생성 파이프라인”이어서, 단일 패스 생성 방식 대비 검증/디버깅 부담을 낮추는 방향의 실질적 진전을 제시합니다. 결과적으로 고위험 칩 설계에서 LLM 기반 RTL 의사결정을 받아들이는 데 필요한 데이터 기반의 verifiable reward(검증 가능 보상) 관점도 강화됩니다.



### Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadenc (https://arxiv.org/abs/2606.19386)
Comments:
          10 pages, 5 figures. Sequel to arXiv:2606.04296. Pre-registered; falsification clauses honored (H5 unsupported; H7 strict band 16/20) repo:this https URL

- **Prior Approaches**: 자율 에이전트를 감시하는 런타임 모니터는 내부 상태(행동 기준선, 드리프트 통계, 모델링된 affect 등)를 누적한 뒤 임계값으로 개입 여부를 결정해왔다. Modgil(2026)은 affect 엔진에 대한 임계값 기반 트리거가 SWE-bench 디버깅 에이전트에서 거의 상시 알람으로 붕괴되는 State Saturation Trap을 보고했지만, 사후 감사에서 엔진의 decay가 실제로 작동하지 않아 결과의 원인이 “순수 누적”에 있었다고 정정했다.

- **Core Contribution**: 이 논문은 실패 원인을 “어떤 emotion 엔진이냐”가 아니라 모니터 동역학의 캘리브레이션이 sample time(관측 간격)인지 wall-clock time(초 단위 반감기)인지의 구분으로 재정의한다. wall-clock-calibrated leaky integrator의 level trigger는 에이전트의 가변적인 action 간 지연에서 양분된 거동(상시 알람 vs 침묵)을 보이며, 전이(상승) 감지는 이 함정에서 벗어나지만 인간 개입 타이밍 복원까지는 해결하지 못한다고 제시한다.

- **Technical Challenges**: 핵심 난제는 “에이전트 스트림에서 Δt(행동 사이 간격)가 크게 흔들리는데, decay가 그 시간을 어떻게 해석하느냐”를 반영해 반사실 실험을 설계하는 것이다. 저자들은 (1) Modgil(2026) 오류(Δt=0로 decay 미작동)를 erratum으로 바로잡고, (2) 동일 트리거·엔진을 유지한 채 Δt 그리드를 넣은 uniform-cadence sweep으로 임계 구간을 찾았으며, (3) hook-instrument로 실제 wall-clock cadence(중앙값 1.53s, p90 2.33s)를 측정해 그 구간이 실제 운용에서도 constant-alarm 레짐에 놓인다는 점을 확인했다.

- **Empirical Impact**: 20개 SWE-bench-Verified 디버깅 궤적과 65개 계측된 라이브성 실험에서, wall-clock level trigger는 dt<=1s에서 20/20 거의 상시 알람으로, dt>=60s에서는 완전 침묵으로 갈라지며 전환은 (1,30]s 내부에서 발생했다. 반대로 sample-time CUSUM은 동일 입력에서 dt 불변성을 보여 “문제의 본질이 임계값 누적 방식의 캘리브레이션 클래스”임을 실험적으로 지지했고, edge trigger(T3)는 알람 붕괴를 피하지만 개입 타이밍 정렬 성능은 개선되지 않아 감시(상태 병목)와 타이밍(레이블 신뢰도) 문제를 분리해 이해해야 한다는 메시지를 남긴다.



### DynAMO:Dynamic Asset Management Orchestration via Topological Multi-Agent Scheduling (https://arxiv.org/abs/2606.19382)
Comments:
          11 pages, 2 figures, 7 tables, 4 algorithms. Evaluated on the AssetOpsBench industrial benchmark. Code: this https URL

- **Prior Approaches**: 기존 LLM 기반 에이전트는 ReAct류 프롬프트나 멀티에이전트 오케스트레이션을 통해 end-to-end 자동화를 노리지만, 자유 형식 워크플로를 가정하는 경우가 많아 산업용 안전·정합성 보장이 약하다는 지적이 있었다. 또한 LangChain/LangGraph/CrewAI 같은 일반 프레임워크는 병렬 실행을 제공해도 의존성/계획 검증, 런타임 전 정형화 보장, 지연·동시성 신뢰성 특성화가 상대적으로 개발자 책임에 남는 편이다.

- **Core Contribution**: 이 논문은 DynAMO(Dynamic Asset Management Orchestration)를 제안하며, Plan-then-Execute 구조 안에서 스키마 제약 계획을 먼저 수행해 실행 전 워크플로 그래프를 검증한다. 이후 DAG 기반 위상 실행으로 SequentialWorkflow(위상 순차)와 ParallelWorkflow(의존성 인지 동시성)를 모두 지원해, 구조적 정합성과 안전을 유지하면서 독립 작업을 겹쳐 실행한다.

- **Technical Challenges**: 산업 환경에서는 도구 호출 I/O 지연, 동시성에 따른 불안정성, 긴 컨텍스트로 인한 context saturation(예: lost-in-the-middle)이 동시에 발생한다. DynAMO는 계획 단계에서 JSON 스키마로 DAG의 불변성을 강제하고, 실행 단계에서 도구 메타데이터로 I/O-bound·compute-bound를 구분해 자원 효율적으로 스케줄링하며, 실패 분기(부분 오류)를 격리해 graceful degradation을 구현한다. 또한 Token-Budgeted Pruning과 just-in-time retrieval로 컨텍스트를 구조적으로 줄여 추론 지연을 낮춘다.

- **Empirical Impact**: AssetOpsBench(산업 벤치마크, 141 질의)에서 6개의 통제 실험을 수행한 결과, ParallelWorkflow는 median end-to-end latency를 1.6x 줄였고 고병렬 워크플로에서는 1.8x까지 관측됐다. 도구 지연을 현실적으로 계측해 분해하면 전체 시간의 90% 이상이 LLM inference와 오케스트레이션에 해당해, 산업 최적화 우선순위를 추론·컨텍스트 효율로 재정렬하는 근거를 제공한다. 아울러 병렬 스케줄은 평균 속도만큼이나 지연 분산을 크게 낮춰(안정성) 반복 실행에서도 일관된 실행 특성을 보였고, 부분 결함 주입에서도 정상 완료를 유지하는 등 배포 친화성을 뒷받침했다.



### Improving Code-Switching ASR with Code-Mixing Guided Synthetic Speech (https://arxiv.org/abs/2606.19381)
Comments:
          Accepted to Interspeech 2026

- **Prior Approaches**: 코드스위칭(CS) ASR은 언어 전환으로 인한 교차언어 음운 간섭과 구어체 변동 때문에 어렵지만, 고품질 CS 텍스트-음성 쌍 부족이 핵심 병목이다. 이를 보완하려는 CS TTS 데이터 증강은 주로 음질·인식 가능성 같은 재현 품질에 초점을 맞추며, 실제 대화에서의 언어 경계/혼합 패턴 일치 여부를 명시적으로 강제하지 못해 ASR 개선이 제한됐다. 또한 waveform은 연속 신호라 언어 라벨을 직접 프레임 단위로 얻기 어렵다는 표현 불일치 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 코드믹싱 패턴을 “선호 학습(preference learning)”으로 직접 유도하는 CS TTS 프레임워크를 제안한다. 핵심은 Code Mixing Index(CMI)를 음성 영역으로 확장한 CMIspeech와, 합성 발화의 언어 혼합 구조 보존 정도를 비교하는 ΔCMI를 DPO(Direct Preference Optimization) 학습의 preference 신호로 넣는 것이다. 그 결과 합성 CS 음성이 언어 경계와 mixing 비율을 더 현실적으로 따르도록 만들고, 그 합성 데이터를 ASR fine-tuning에 활용해 성능을 끌어올린다.

- **Technical Challenges**: 가장 큰 기술적 난제는 텍스트 기반 CMI처럼 “프레임 단위 언어 혼합”을 waveform에서 안정적으로 측정·강제하기 어려운 표현 불일치였다. 논문은 강제 정렬이나 수동 주석 없이 디코더 cross-attention으로 pseudo frame-level language labels를 만들고, 이를 기반으로 CMIspeech와 ΔCMI를 계산해 코드스위칭 충실도를 수치화한다. 이후 여러 critic(MER, UTMOS, ΔCMI)를 정규화해 contrastive preference pair를 구성하고, 신뢰도 기준으로 필터링한 뒤 multi-critic DPO로 TTS를 정렬한다.

- **Empirical Impact**: SEAME 만다린-영어 대화형 CS 코퍼스에서 Whisper Large v3를 fine-tuning한 결과, MER(Mixed Error Rate)가 DevMAN/DevSGE에서 12.1%/17.8%에서 8.9%/14.2%로 크게 낮아졌다. ΔCMI critic을 포함할 때 코드스위칭 구조 차이를 가장 크게 줄였고(ΔCMI 관점에서 큰 감소), 동시에 인식성과 자연스러움이 함께 유지되는 경향을 보였다. 즉 단순한 음성 품질 최적화보다 “acoustic-level code-switching 구조 보존”을 직접 학습 신호로 주는 접근이 downstream CS ASR에 더 실질적인 이득을 준다는 점을 입증했다.



### How Linear Is a Transformer Feed-Forward Block? Per-Block Linear Recoverability Is Learned, Not Architectura (https://arxiv.org/abs/2606.19379)
Comments:
          14 pages, 5 figures

- **Prior Approaches**: Transformer FFN은 key–value memory로 해석되며, 압축이나 해석을 위해 “얼마나 선형(가법)이고 얼마나 비선형(곱셈/상호작용)”인지 논의가 이어져 왔다. 하지만 기존의 선형/다항 probe는 학습 최적화에 의존해 정확한 선형성의 상한을 주지 못하거나, 비선형 잔차를 저차 곱으로 설명할 수 있는지의 측정이 일관되지 않았다.
또한 트랜스포머 활성은 ill-conditioned(조건이 나쁨)하고 outlier 특징이 커서, 단순히 학습한 선형 기준선이 충분히 수렴하지 못해 선형 recoverability를 과소평가할 위험이 있었다.

- **Core Contribution**: 논문은 각 FFN을 입력-출력의 position-wise map으로 보고, 해당 블록의 활성 분포에서 “정확한 least-squares 선형 근사(닫힌형 해)”로 분해한다. 이때 held-out에서 선형 부분이 설명하는 분산비를 linear recoverability R^2_lin으로 정의해, 최적화 없이 재현 가능한 블록 고유 측정값을 제공한다.
그 다음 잔차를 저차(랭크-낮은) bilinear probe로 추가 측정해, 잔차가 단일 저차 곱 형태로 포착되는지까지 확인한다.

- **Technical Challenges**: 핵심 기술적 난제는 트랜스포머 활성의 조건이 나빠서 학습한 선형 baseline이 잘 수렴하지 못한다는 점이다. 논문은 이 문제를 해결하기 위해 선형 항은 학습이 아니라 닫힌형 least-squares 해로 “선형 상한(ceiling)”을 정확히 계산하고, 이후 probe의 이득을 이 상한 대비로만 읽는다.
잔차 분석에서는 bilinear probe의 선형 분기(branch)를 고정해 residual-gain만 분리 추정하며, 저차 곱이 잔차를 얼마나 설명하는지 상호작용 구조를 직접 시험한다.

- **Empirical Impact**: GPT-2, Pythia-160m, llama-160m의 12개 블록을 모두 측정한 결과, R^2_lin은 깊이에 따라 단조롭게 변하지 않고 블록 간 이질성이 매우 크다(거의 선형 >0.99부터 강한 비선형 <0.3까지). 특히 같은 폭/깊이의 GELU 모델(GPT-2 vs Pythia)은 서로 다른 블록이 선형으로 보이며, 선형 recoverability가 활성함수나 아키텍처가 아니라 “학습된 개별 블록 성질”임을 보여준다.
또한 잔차는 랭크-낮은 degree-2 bilinear probe로는 대부분 잘 회수되지 않아, unrecovered 계산이 단일 저차 곱이 아닌 더 고차/분산된 구조일 가능성을 시사한다. 아울러 R^2_lin이 높은 블록은 훨씬 작은 단일 레이어로 교체해도 성능 손실이 제한적일 수 있음을 압축 신호로 제시한다.



### Emyx: Fast and efficient all-atom protein generation (https://arxiv.org/abs/2606.19377)
- **Prior Approaches**: 기존 효소(AME) 설계용 생성 모델들은 all-atom 좌표 생성과 함께, 구조 예측 분야에서 유래한 무거운 pair 표현/재귀 구조를 채택하는 경우가 많아 학습 비용이 크고 샘플 다양성이 제한되는 문제를 보였다. 또한 motif를 수동 인덱스로 고정하는 방식은 탐색 공간을 줄이고 계산을 늘려 성공률을 떨어뜨린다. 무엇보다 평가 지표의 불일치로 인해 로컬 촉매 기하만 맞추고 전체 폴드는 복원하지 못하는 설계가 ‘성공’으로 과대평가될 여지도 제기된다.

- **Core Contribution**: Emyx는 140M 파라미터의 조건부 flow matching 모델로, 생성기 설계에서는 rich co-evolutionary 신호 대신 희소한 기하 제약(motif/리간드 주변)을 조건으로 쓰면 된다는 관점에서 아키텍처를 대폭 단순화했다. 표준 transformer 블록에 모델 역량을 집중하고, 무거운 임베딩 스택/조밀 pair 계산을 피하며 sparse edge 그래프만으로 조건부 관계를 구성한다. 더불어 flow matching 보간(interpolant)을 EDM noise-level 프레임워크로 정확 재파라미터화해, 재학습 없이 EDM 계열의 강력한 샘플링을 그대로 사용할 수 있게 했다.

- **Technical Challenges**: 핵심 난관은 (1) all-atom 좌표 생성에서 기하 정확성과 구조 다양성을 동시에 만족해야 하는데, (2) 단순화된 생성기 아키텍처가 학습/샘플링에서 불안정해지지 않도록 설계해야 한다는 점이다. Emyx는 선형 보간 기반 flow matching으로 학습 그래디언트 분산과 수렴을 유리하게 만들고, motif 원자(입력 기하)는 고정한 채 scaffold를 학습하도록 손실을 분리했다. 또한 DiT 스타일 설계(adaLN-Zero, SwiGLU 등)와 sparse 조건부 관계 계산으로 계산량을 줄이면서도, lDDT 같은 보조 손실과 EDM 재파라미터화로 샘플링 성능을 확보했다.

- **Empirical Impact**: AME 벤치마크에서 Emyx는 strict sc-RMSD(전역 fold 복원 + 촉매 기하 정확 + ligand clash 없음)를 기준으로 13.4% 성공률을 달성해 Proteína-Complexa(8.8%)와 RFdiffusion3(6.7%)를 앞섰다. 훈련 비용은 682 GPU-hours로 RFdiffusion3 대비 약 4배 적었고, 생성 효율도 단일 GPU에서 RFdiffusion3보다 2배 이상 빠르며 긴 사슬에서 격차가 더 커진다. 나아가 구조적 novelty/다양성(Tm-score, Foldseek 클러스터 등)과 기하 유효성(pass rate)에서도 경쟁 우위를 보여, ‘생성기에는 단순화가 먹힌다’는 실증을 제공했다.



### Cost-Optimal LLM Routing with Limited User Feedback under User Satisfaction Guarantees (https://arxiv.org/abs/2606.19376)
Comments:
          Preprint. Under review

- **Prior Approaches**: 기존 cost-aware LLM routing은 추론 비용을 줄이는 데는 효과적이었지만, SLA에서 요구하는 품질(만족도) 기준을 시간에 대해 보장하는 정식 이론적 근거는 부족했습니다. CARROT나 PROTEUS처럼 formal 보장을 제공하더라도 주로 offline 학습 기반이거나, online 적응을 하더라도 실제 운영에서 거의 충족되기 어려운 완전·균형·지연 없는 피드백 가정을 사용했습니다. 또한 관측 기반 학습을 시도한 방법도 SLA 보장과 online 적응을 동시에 만족시키지 못했습니다.

- **Core Contribution**: 이 논문은 프로덕션에서 흔한 “희소하고 one-sided인 사용자 피드백”만으로도, 비용을 최소화하면서 SLA(만족도 비율 α) 위반을 엄격히 통제하는 온라인 라우팅 알고리즘 SLARouter를 제안합니다. SLARouter는 비용 최적 정책을 목표로 하되, Lyapunov drift-plus-penalty를 관측 데이터 환경에 맞게 확장해 시간 평균 만족도 조건을 유지합니다. 결과적으로 per-benchmark 튜닝 없이도 SLA 제약을 만족시키는 라우팅을 구현합니다.

- **Technical Challenges**: 핵심 난제는 (1) 대부분의 요청에서 품질 레이블이 없고, (2) 라우터가 실제로 선택한 모델에 대해서만 피드백이 관측된다는 점(다른 모델의 성능 불명)입니다. 논문은 탐색(exploration) 확률을 점진적으로 줄이는 전략과, 모델별 multi-label satisfaction predictor를 사용해 one-sided 관측만으로도 학습이 진행되게 설계했습니다. 또한 피드백이 없는 경우 가상 큐 업데이트에 predictor 점수를 대체 프록시로 사용하면서, 가상 큐 안정성과 SLA 준수 성질이 유지되도록 분석을 구성했습니다.

- **Empirical Impact**: 다양한 LLM 벤치마크에서 SLARouter는 SLA 제약을 만족하면서도 기존 cost-aware 라우팅 대비 최대 2.2x 운영 비용 절감을 보였습니다. 특히 per-benchmark tuning 없이도 동등 수준의 라우팅 성능을 달성해, 벤치마크-운영 불일치로 인한 실무 부담을 낮출 가능성이 큽니다. 실질적으로 “online adaptivity + 형식적 SLA 보장 + 희소 one-sided 피드백 학습”을 동시에 만족한 최초 계열로, 운영형 LLM 라우팅 설계에 기준점을 제시합니다.



### Protein Representation Learning with Secondary-Structure and Energy-Filtered Hydrogen-Bond Graphs (https://arxiv.org/abs/2606.19374)
- **Prior Approaches**: 기존 단백질 그래프 GNN은 엣지를 주로 서열 인접성(윈도우)이나 공간적 근접도(radius, kNN)로 정의해 왔습니다. 하지만 이런 방식은 물리적 상호작용의 화학적 특이성(예: donor–acceptor 적합성, 방향성)을 반영하지 못해 하이퍼파라미터(컷오프/윈도우)에 민감하다는 한계가 큽니다. 이후 ProNet 같은 SE(3) 불변 인코딩이나 CDConv/CoupleNet처럼 1D-3D 결합을 개선했지만, 핵심 엣지 토폴로지가 여전히 근접 휴리스틱에 의존하는 경우가 많았습니다.

- **Core Contribution**: 이 논문은 SSProNet(Secondary-Structure and Hydrogen-Bond-Aware Graph Neural Networks for Proteins)으로 “이차구조 + 에너지 필터링된 수소결합”을 그래프 표현의 귀납 편향으로 넣습니다. 잔기 노드에는 DSSP 기반 secondary structure와(선택적으로) solvent accessibility를 부여하고, 엣지는 DSSP가 찾은 backbone hydrogen bond를 에너지 강도 기준으로 필터링해 구성합니다. 근접도 기반 scaffold는 연결성 보조로 유지하되, 장거리 결합과 국소 안정화의 핵심 커플링을 수소결합 토폴로지로 대체/보강합니다.

- **Technical Challenges**: 어려운 점은 “생물학적으로 의미 있는 엣지”를 얻으면서도 GNN이 학습 가능한 그래프 커버리지를 잃지 않는 균형입니다. SSProNet은 radius scaffold로 기본 연결성과 수용영역을 확보한 다음, DSSP의 수소결합을 h<0(기본 -0.5 kcal/mol) 조건 등으로 필터링해 안정화 결합만 sparse하게 추가합니다. 또한 ProNet의 계층적 SE(3) 불변 Hier-Geom-MP 구조는 유지하고, secondary structure/solvent accessibility를 노드 채널로 얹어 회전·이동 불변성과 표현력을 동시에 챙깁니다.

- **Empirical Impact**: 실험에서는 단백질 fold 분류, enzyme commission(반응) 예측, protein-ligand binding affinity(LBA) 예측에서 기존 그래프 기반 방법 대비 일관된 개선이 보고됩니다. fold에서는 Fold/Superfamily/Family 정확도와 평균에서 강한 향상이 나타났고, 반응 예측에서는 SSProNet-Backbone이 최상 성능(88.3%)을 기록했습니다. LBA에서는 Pearson(0.613), Spearman(0.616) 등 상관 지표가 최고 수준으로 올라섰으며, 일부 지표(RMSE/Kendall)는 완전 우세는 아니지만 상관 개선이 유의미하다는 점을 강조합니다. 전반적으로 성능 향상뿐 아니라 학습된 연결성이 알려진 구조 모티프와 정렬되어 biological interpretability도 좋아진 것으로 정리됩니다.



### cAPM: Continual AI-Assisted Pace-Mapping with Active Learning (https://arxiv.org/abs/2606.19373)
- **Prior Approaches**: 기존 VT(심실빈맥) 국소화의 대표적 절차인 pace-mapping은 여러 심실 부위를 전기 자극하고 생성된 12-lead ECG를 빠르게 비교해 다음 자극 위치를 정하는, 고침습·고노동 작업입니다. 이를 줄이기 위해 active learning(능동학습) 기반 AI가 제안됐지만, 과거에 쌓인 지식을 다른 VT나 다른 환자에게 옮기지 못해 각 표적 VT마다 surrogate를 새로 학습·재시작해야 하는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 continual AI-assisted pace mapping(cAPM)이라는 연속 학습 프레임워크로, 과거 pace-mapping 데이터에서 얻은 지식을 다음 표적 VT 탐색에 연속적으로 전이해 필요한 샘플 수를 줄이는 방법을 제시합니다. 핵심은 task-agnostic surrogate neural network로 “자극 위치→12-lead ECG 형태” 매핑을 재사용하고, active learning으로 정보가 큰 자극점을 선택하며, continual learning으로 이전 작업의 성능을 유지하는 구조입니다.

- **Technical Challenges**: 연속 표적을 순차로 다룰 때 발생하는 catastrophic forgetting(치명적 망각)과, task마다 새로 학습해야 했던 surrogate의 고정성 문제를 동시에 해결해야 했습니다. cAPM-Meta는 MER(meta-experience replay) 기반 메모리 버퍼로 재학습을 돕고, cAPM-Ensemble은 과거 모델을 frozen ensemble으로 유지해 과거 데이터를 직접 저장하지 않는 대안을 제공합니다.

- **Empirical Impact**: in-silico 고정밀 시뮬레이션 테스트베드에서 cAPM은 임상 허용 오차 5mm(정확도 허용 범위) 내 국소화 확률을 4.5개 자극으로 81%까지 끌어올렸고, BOATMAP 같은 최신 active-learning 기반 GP 접근은 13.7개 자극으로 38%에 그쳤습니다. 또한 재현 설정 전반에서 cAPM-Meta(정확 임계 달성 시 step 수 4.1±2.0, 실패율 92% 개선)와 cAPM-Ensemble(4.8±1.7)의 성능·효율 향상을 보였으며, 실제 in-vivo 전임상/임상 전 단계로의 준비 근거를 제공합니다.



### ProMUSE: Progressive Multi-modal Uncertainty-guided Staged Evidential Alzheimer Disease Classification (https://arxiv.org/abs/2606.19371)
- **Prior Approaches**: 기존 AD 분류 연구는 임상 데이터, 구조 MRI, PET을 모두 추론 시점에 항상 사용할 것을 전제하며 정확도 극대화에 집중해 왔습니다. 이 방식은 고가·비접근성인 PET/MRI 의존을 매번 강제해 실제 임상 워크플로에서 비용 부담이 큽니다. 또한 multimodal late fusion/ensemble은 불확실성을 직접 다루기 어려워, “언제 추가 촬영이 필요한가” 같은 비용-정확도 조절이 제한적입니다.

- **Core Contribution**: ProMUSE는 불확실도 기반으로 단계적으로 모달리티를 추가하는 progressive staged evidential 네트워크입니다. 먼저 저비용 임상 데이터로 evidential classification을 수행하고, Dirichlet 기반 subjective logic으로 불확실도를 계산한 뒤 임계값을 넘으면 MRI 또는 PET 특징을 점진적으로 결합합니다. 모달리티별 belief과 uncertainty를 Dempster-Shafer theory로 fusion해, 전체 모달리티를 쓰는 기준선과 견줄 만한 성능을 유지하면서 촬영 비용을 줄이는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 모달리티가 제공하는 증거의 양/품질을 불확실도로 일관되게 모델링하고, (2) 불확실도가 큰 경우에만 추가 모달리티를 선택하는 정책을 학습 가능한 형태로 만드는 것입니다. ProMUSE는 Softplus로 evidence를 생성한 뒤 Dirichlet 기반 subjective logic으로 uncertainty(미지의 증거량)를 계산하고, 모달리티 간 충돌을 Dempster-Shafer fusion의 conflict 계수로 반영해 결합 시 신뢰도를 조절합니다. 마지막으로 임계값 τ는 데이터와 태스크 특성에 맞춘 data-driven 절차로 선택하며, KL regularization을 포함한 evidential loss로 불확실도 보정까지 함께 학습합니다.

- **Empirical Impact**: ADNI, AIBL, OASIS의 CN-AD, CN-MCI, MCI-AD 세 태스크에서 ProMUSE는 full-modality baseline과 비교해 경쟁력 있거나 더 높은 정확도(Accuracy/F1/AUC)를 보이면서 MRI/PET 사용량을 50-90% 절감했습니다. 특히 CN vs AD에서처럼 절반 가까운 표본이 임상 데이터만으로 충분히 확정되고, 나머지는 필요할 때만 MRI/PET이 추가되는 방식으로 자원 절약이 관측됩니다. 결과적으로 현실적인 AD 스크리닝 파이프라인에 맞춘 uncertainty-aware, resource-efficient 진단 전략을 제시했다는 점에서 의미가 큽니다.



### Human-like autonomy emerges from self-play and a pinch of human data (https://arxiv.org/abs/2606.19370)
Comments:
          10 pages

- **Prior Approaches**: 기존 self-play 강화학습은 시뮬레이션 경험만으로 정책을 키워 비용을 크게 줄이지만, 높은 성공 보상만으로는 인간이 수용할 ‘교통 관례’ 정렬이 보장되지 않아 ‘alien’한 주행 관습으로 수렴할 수 있다. 이를 막기 위해 reward engineering과 domain randomization을 반복적으로 설계하는 방식이 널리 쓰였지만, 작업량이 크고 취약하다는 한계가 있다. 한편 imitation learning(IL)은 보상을 직접 설계하지 않지만, 폐루프 배치에서의 분포 불일치 문제를 줄이려면 대규모 인간 데이터가 필요한 경우가 많다.

- **Core Contribution**: 이 논문은 human demonstration을 폐기하지 않고, self-play RL의 최소 안전 목표 보상 위에 behavioral cloning(BС) 기반 정규화 항으로 ‘anchor’ 역할을 부여한다. 즉 spiced self-play로 불리는 방식은 reward engineering 없이도 인간 궤적과의 교차 조율을 끌어올린다. 특히 Waymo Open Motion Dataset(WOMD)에서 30분 수준의 인간 데이터만으로도 효과가 크게 나타나며, 기존 IL과 비교해 데이터 요구량을 극적으로 낮춘다.

- **Technical Challenges**: 핵심 기술적 난제는 self-play가 만들어내는 비정적(상호 학습) 파트너 분포에서도, 소량의 인간 데이터가 정책의 충돌률·행동 현실성을 함께 개선하도록 ‘정규화가 어디에서 어떻게 작동하는가’를 안정적으로 설계하는 것이다. 논문은 PPO 기반 self-play에 KL 페널티를 더하되, KL이 온라인에서 에이전트가 실제로 방문하는 상태분포에서 anchor(BС)로 당기도록 구성해 오프라인 데이터 분포의 왜곡을 줄였다. 또한 시뮬레이션에서 human-replay, IDM(규칙 기반 공존자), self-play의 3가지 평가 셋업을 분리해 기여를 해석 가능하게 만들었다.

- **Empirical Impact**: 실험 결과, 30분~3시간의 human anchor 데이터를 spiced self-play에 더하면 충돌(특히 at-fault 충돌)과 안전성 지표가 크게 개선되며, 여러 환경 평가에서 SMART 계열 IL 대비 우위를 보인다. 또한 충돌 빈도뿐 아니라 충돌 심각도(Δv 기반)도 함께 낮춰, 인간과 함께 운행 시 위험을 줄이는 방향으로 정렬이 이동함을 확인했다. 의미 있는 점은 학습이 단일 consumer-grade GPU에서 약 15시간 내 end-to-end로 가능하고, 인간 데이터는 ‘주요 감독신호’가 아니라 ‘lightweight anchor’로만 쓰였는데도 성능이 크게 개선됐다는 것이다.



### Zero-Inflated Gaussian Distributions Enable Parameter-Space Sparsity in Estimation-of-Distribution Algorithms (https://arxiv.org/abs/2606.19369)
- **Prior Approaches**: 희소 해법이 필요한 고정 차원, gradient-free black-box 최적화에서 기존 sparse 최적화들은 보통 hand-crafted sparsity 연산(offswitch/zeroing threshold), bi-level(서포트 마스크와 값의 교대 최적화), 혹은 sparsity-aware mutation 설계를 다시 도입했다. 그 결과 연산 설계나 하이퍼파라미터가 문제 인스턴스에 결합되고, EDAs가 operator 설계 편향을 피하려는 장점이 약해졌다. 또한 dense 연속 분포를 쓰는 EDA는 정확히 0을 샘플링할 수 없어 “구조적 0”을 다루기 어렵다.

- **Core Contribution**: 이 논문은 EDA의 sampling law를 multivariate zero-inflated Gaussian(ZIG)으로 일반화해, 구조적 0과 활성 값을 같은 분포에서 함께 최적화한다. ZIG는 각 차원에 대해 0이 될지(마스크)와 활성 값이 무엇인지(연속 값)를 분리해 표현하는 latent Gaussian 모델을 쓰며, support와 값의 교대를 없애 hierarchy-free로 탐색을 수행한다. 따라서 추가 제약/페널티 스케줄 없이도 희소 해법을 직접 샘플링하도록 설계된다.

- **Technical Challenges**: 핵심 난점은 (1) 매 세대 elite에 맞춰 분포를 재적합해야 하므로 샘플만으로 latent 파라미터를 식별가능하게 복원할 수 있어야 하고, (2) value–mask 간 상관이 관측 분포를 왜곡해 상관 파라미터가 중복될 위험이 있다는 점이다. 이를 위해 itemwise conditionally independent nonresponse(ICIN) 가정을 도입해 비식별성을 해소하고, 관측된 피처로부터 latent 상관을 복원하는 감쇄(attentuation) 관계와 amortized inversion-based 추정기를 제안한다. 추정된 상관행렬은 양의 준정부호 보장을 위해 투영(projection) 단계까지 거친다.

- **Empirical Impact**: 추정기 자체 실험에서 latent 상관 구조가 넓은 설정 범위에서 정확히 복원되는 것으로 확인됐다(특히 활성 확률이 낮은 구간에서 오차가 커지는 패턴도 관측). LunarLanderContinuous-v3에서 ZIG-EDA는 dense Gaussian EDA, hand-crafted sparse EA, ad-hoc sparse EDA보다 더 빠르게 수렴하면서 더 높은 최종 return을 달성했다. 또한 90개 파라미터 중 약 12개 수준만 active로 남기는 희소 제어기를 찾아내어, “정확한 0”을 다루는 sampling law의 실용적 이점을 보여준다.



### Information Lattice Learning as Probabilistic Graphical Model Structure Learning (https://arxiv.org/abs/2606.19366)
- **Prior Approaches**: 기존 PGM 연구는 변수(픽셀, 노트, 라벨 등) 그래프를 먼저 두고 확률 파라미터를 추정하며, Bayesian network는 조건독립, Markov random field는 지역 호환성, factor graph는 분해 제약을 중심으로 설명한다. 반면 ILL은 신호(특히 확률분포)를 입력으로 삼아 사람이 해석할 수 있는 추상화(규칙)를 찾는 방식으로, 규칙 자체를 그래프 구조와 직접 연결해 설명하는 데는 관점이 부족했다.

- **Core Contribution**: 이 논문은 ILL의 확률 규칙을 “결정적 quotient variable의 주변분포(marginal)”로 재정의해 PGM 구조학습 관점에서 설명한다. 또한 rule set은 학습된 해석 가능한 추상화들에 대한 주변분포 제약들의 집합이며, lifting은 그 제약을 만족하는 분포 중 추가 구조를 최소화(최대 무지/최대 엔트로피 계열)하는 선택으로 정리한다.

- **Technical Challenges**: 핵심 난제는 ILL의 information lattice가 Bayesian network처럼 ‘조건의존성’ 간선을 의미하지 않는다는 점을 정교하게 구분하는 것이다. 논문은 lattice 간선이 추상화의 coarsening–refinement(정교화–거칠기) 관계를 뜻하며, 실제 확률 그래프(예: Shannon lifting의 log-linear factor graph)는 선택된 rule set 이후에 요약/인덱싱된 factor로 유도된다고 명확히 한다.

- **Empirical Impact**: 개념 정리는 실험 성능을 넘어, ILL을 factor/제약 기반 확률모형 이론의 언어로 해석·확장할 수 있게 만든다는 점에서 의미가 크다. 특히 inference, identifiability(동일 제약이 다른 antichain으로도 재현될 수 있음), 샘플 유한 환경에서의 robust ILL, 그리고 인과모형으로의 확장 같은 후속 연구 방향을 더 구체화한다.



### Computational Identifiability (https://arxiv.org/abs/2606.19361)
- **Prior Approaches**: 기존 인과 식별(identification)은 “기대값(in expectation)” 기준으로 파라미터/질의를 유일하게 계산할 수 있는지 수학적으로 증명하는 theoretical identifiability가 중심입니다. do-calculus, po-calculus, σσ-calculus, 그리고 counterfactual용 알고리즘들은 특정 모델/개입 형태에서 sound and complete를 제공하지만, 새로운 설정마다 별도 분석이 필요하고 유한 표본·그래프 기준의 애매함·관측/개입 혼합 데이터에서는 실전 추정 가능성을 충분히 안내하지 못합니다.

- **Core Contribution**: 이 논문은 컴퓨터가 유한 시간 안에 수행하는 “실증적 식별(computational identifiability)”을 제안해, 이상화된 무한 데이터가 아니라 유한 표본에서 목표 인과량을 오차 허용 범위 내로 추정 가능한지로 식별을 재정의합니다. 이를 위해 구조인과모형(SCM) 위의 meta-prior와 추정기(estimators) 후보 공간(hypothesis space)을 두고, 정해진 탐색 절차가 원하는 tolerance(ϵ)와 confidence(1−δ)를 만족하는 추정기를 찾으면 식별이 성립한다고 봅니다.

- **Technical Challenges**: 핵심 난제는 “이론적으로는 비식별처럼 보일 수 있지만, 유한 샘플·혼합 데이터·애매한 그래프 조건에서도 실제로는 추정 가능한가?”를 계산적으로 판정해야 한다는 점입니다. 논문은 SCM 공간에 대한 prior(가정의 스펙트럼)과 estimators의 가설공간을 함께 두고, 관측/개입/반사실( counterfactual ) 데이터에 대한 causal mixture distribution을 구성한 뒤, 유한 표본에서 오차-신뢰 조건을 만족하는지의 검색 문제로 전환하는 프레임워크를 제시합니다.

- **Empirical Impact**: 실험에서는 복잡하고 작은 finite sample 조건에서, (1) 애매한 graphical criteria, (2) observational-interventional mixed data, (3) 반사실 데이터와 estimand 간 교차 상황에서도 computational identifiability가 세밀한 실무형 식별 질문에 답할 수 있음을 보입니다. 이 접근은 “무한 데이터에서의 유일성”보다 “주어진 자원·오차 허용 하에서 실제 추정 가능성”을 중심으로 인과추론 파이프라인을 설계하는 데 의미가 큽니다.



### Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots (https://arxiv.org/abs/2606.19357)
Comments:
          To appear at RLC 2026

- **Prior Approaches**: 로봇에서 강화학습을 다루는 기존 방식은 (1) 시뮬레이션에서 학습 후 로봇에 배포하거나, (2) 사람이 조종해 데이터를 수집한 뒤 offline-RL로 학습하는 방법, (3) 로봇에서 직접 학습하는 방법으로 나뉜다. 그중 직접 학습은 시뮬레이터나 인간 데이터가 없어도 되지만, 신뢰성·접근성·장기 운용성을 만족하는 로봇 플랫폼이 부족했다. 특히 real-time reinforcement learning 관점에서는 지연과 비정상성 같은 현실 요소가 시뮬레이션에선 잘 반영되지 않는 한계가 있었다.

- **Core Contribution**: 본 논문은 로봇에서 직접 강화학습을 장기 실험 가능하게 만드는 Physical Atari 플랫폼을 제시한다. Physical Atari는 Atari CX40+ 컨트롤러를 로봇이 물리적으로 조작하는 Robotroller와, ALE 게임 화면 및 보상 신호를 재현하는 Atari Devbox로 구성된다. 저자들은 이 플랫폼이 robust(고장 없이 오래), accessible(저비용·조립 용이), easy to use(개입 최소), versatile(다양한 ALE 게임)하도록 설계 핵심 결정을 정리한다.

- **Technical Challenges**: 핵심 기술 과제는 (a) 장시간 사용 시 기구 마모와 진동으로 인한 신뢰성 저하, (b) 실제 시스템의 지연을 포함한 real-time 상호작용에서의 성능 일관성, (c) 물리 입력과 게임 행동 간의 정확한 매핑/제어 안정화였다. 저자들은 마모를 줄이기 위해 볼 베어링으로 모든 구동 이동을 처리하고, 풀림·기어 마모·부품 파손 문제는 thread-locking, 금속 기어 서보로 교체, 금속 horn으로 보강해 수 주간 무고장 운용을 달성했다. 또한 서보의 PID 파라미터 튜닝으로 움직임을 부드럽게 하되 발생 가능한 고전류 상태를 high-current reflex로 차단해, 자동으로 안전 상태로 보호하도록 했다.

- **Empirical Impact**: 실험에서 Physical Atari 위에서 6개 ALE 게임(Pong, Seaquest, MsPacman, Assault, Asterix, Kangaroo)을 약 5.5시간 학습하며, 여러 번 반복해도 개입 없이 학습이 안정적으로 수행됨을 보였다. 또한 서로 다른 Robotroller(로봇 바디)로 학습된 정책을 옮겼을 때 성능이 전반적으로 저하되었고, 특히 Pong처럼 타이밍 민감도가 큰 게임에서 격차가 크게 나타났다. 나아가 새 바디로 배포 후에도 학습 알고리즘을 계속 적용하면 성능이 회복/개선되어, 로봇에서는 on-device adaptation의 중요성을 실증적으로 강조한다.



### Trustworthy Multi-Agent Systems: Mitigating Semantic Drift with the Argent Signaling Protoco (https://arxiv.org/abs/2606.19356)
Comments:
          17 pages

- **Prior Approaches**: 다중 에이전트 LLM(MAS)에서 리트라이(retry)는 대개 실패를 “다시 시도하면 나아질 것”으로 뭉뚱그려 처리해, 부분적으로만 근거가 맞는 오류와 아예 근거가 없는 오류를 구분하지 못했습니다. CAMEL/AutoGen 같은 에이전트 프레임워크와 Self-Refine/Reflexion 계열 개선은 대화 로그가 남아도, 재시도가 근거 기반의 수리(repair)였는지 아니면 중단(containment)해야 할 유형이었는지 기계적으로 판별하기 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 Argent Signaling Protocol(ASP)을 제안하며, 모든 AI 응답에 @C(확실성), @G(근거), @S(확률적 불안정성), 그리고 claim evidentiary basis를 분류하는 assumption index를 “머신-리더블 헤더”로 함께 부착합니다. 이를 통해 컨트롤러가 재시도의 타당성(수리 가능 vs 차단 필요)을 구분하고, 각 실패 유형에 맞는 라우팅 결정을 내릴 수 있게 합니다. 또한 sidecar 방식으로 에이전트 내부 수정 없이도 메시지 경계에서 품질 게이트를 강제하는 배치를 제시합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 모델 내부 로그 없이도 근거/불안정성을 일관되게 추정 가능한 신호를 설계하고, (2) 다중 턴에서 semantic drift가 누적될 때 이를 감지해 올바르게 경로를 분기하는 것입니다. 논문은 문서 기반 QA에서 토큰 단위의 문맥 중복도, 인용 존재/정합성, 질문/문맥을 벗어난 novel ratio 등을 근거로 @G/@C/@S를 16단계(0~F)로 양자화해 헤더에 넣고, JSD 기반 drift로 “좋은 정답 프로필에서 얼마나 벗어났는지”를 경계 임계값으로 판단하도록 구성합니다.

- **Empirical Impact**: 문서 기반 QA 벤치마크에서 ASP를 사용하면 모델별 pass rate가 유의미하게 개선됩니다(예: Qwen~0.8B 11.1%→33.3%, Dobby~8B 33.3%→44.4%). 또한 다중 에이전트 파이프라인에서는 retrieval 에이전트가 내보낸 근거 없는 출력이 downstream 의사결정 에이전트로 전달되지 않도록 sidecar가 100% 차단(24/27 blocked, ungrounded propagation 0)을 보였습니다. 결과적으로 “리트라이가 필요했던 실패만 수리 루프로 보내고, 비근거 실패는 즉시 차단”하는 운영 가능성을 실증하며 규제/감사 관점의 traceability를 강화하는 의미가 있습니다.



### Sign-Language Datasets at Scale: A Comprehensive Survey on Resources, Benchmarks, and Annotation Standards (https://arxiv.org/abs/2606.19352)
Comments:
          Accepted to ACL 2026 Main. 27 pages, 5 figures

- **Prior Approaches**: 기존 sign-language 연구는 SLR, SLT, SLP 중 특정 과제나 일부 고빈도 벤치마크에 집중돼, 데이터셋이 실제 사용 환경과 얼마나 맞물리는지 체계적으로 비교하기 어려웠습니다. 또한 데이터셋이 서로 분절돼 있고(언어 커버리지 불균형), 주석 체계·입력 모달리티·메타데이터가 제각각이라 재현성과 교차 데이터 학습이 약해집니다. 결과적으로 모델 성능 향상이 “모델”보다 “데이터 성질”에 좌우되는 경향이 강했습니다.

- **Core Contribution**: 이 논문은 공개된 sign-language 데이터셋 120개(35개 수어)를 한데 모은 데이터셋 인덱스를 제시하고, SLR/SLT/SLP 전반의 공통 병목(모달리티 불균형, signer bias, 주석 불일치)을 분석합니다. 더 나아가 24-field Sign-Language Datasheet를 도입해 데이터셋을 표준화된 문서 양식으로 기록하게 하고, GitHub 저장소를 통해 재현 가능한 평가 기반을 제공합니다. 즉 “데이터셋-벤치마크-평가”를 연결하는 데이터 중심 관점을 실무적으로 제공합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) RGB/depth/pose 등 입력 모달리티와 주석 레이어(글로스, 문장 정렬 등)가 데이터셋마다 달라 모델·전처리·평가 파이프라인이 호환되지 않는 문제, (2) 글로스 품질/관행이 일관되지 않아 학습 신호의 해석 가능성과 transferability가 흔들리는 문제, (3) signer 인구통계와 손잡이(handedness) 같은 메타데이터가 부족해 편향을 분석·제어하기 어려운 문제입니다. 논문은 datasheet와 통합 메타데이터 관리를 통해 문서화 공백을 줄이고, 모달리티/주석/도메인 차이를 벤치마크 결과 해석에 직접 반영하는 방식으로 해결 방향을 제시합니다.

- **Empirical Impact**: PHOENIX14T·CSL-Daily·How2Sign·YouTube-ASL·OpenASL을 중심으로 벤치마크를 작업별(SLR/SLT/SLP)·글로스 유무별로 정리한 결과, 글로스 기반 설정은 성능 지표가 높게 나오되 실제 적용성에는 트레이드오프가 있음을 보여줍니다. 또한 CSL-Daily처럼 환경·화자 다양성이 큰 데이터셋에서 WER/BLEU 갭이 더 크게 나타나 일반화 평가의 중요성이 확인됩니다. SLP는 평가 파이프라인과 공개성 부족으로 비교 일관성이 낮다는 점을 지적하며, BLEU 외에 MPJPEDTW, timing F1, 인체 평가 등 보완 지표 필요성을 제안해 향후 연구 품질을 끌어올리는 기준점 역할을 합니다.



### Detecting Hallucinations for Large Language Model-based Knowledge Graph Reasoning (https://arxiv.org/abs/2606.19351)
- **Prior Approaches**: 기존 KG reasoning은 관련 triple을 검색해 프롬프트에 넣어 정확도를 높이지만, LLM이 여전히 검색 지식을 어긋나게 생성하는 hallucination 문제는 지속됩니다. 일반 hallucination 탐지는 LLM 내부 상태나 불확실성에 치우쳐 외부의 KG 구조 정보를 충분히 반영하지 못하고, RAG 전용 탐지는 검색 컨텍스트와의 정합성만 보며 KG의 관계·연결 구조를 놓치는 경우가 많았습니다.

- **Core Contribution**: 이 논문은 LLM 기반 KG reasoning에서의 hallucination을 구조적으로 탐지하는 최초의 방법 LUCID를 제안합니다. LUCID는 LLM attention(내부 집중), KG semantic similarity(관계-질문 의미 적합도), KG structural information(그래프 연결성)을 함께 결합해 “어떤 근거가 실제로 쓰였는지”를 판단하도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 retrieved KG를 프롬프트로 읽는 과정에서, 잘못된 답이 어떤 KG 관계 선택/구조적 모순에서 비롯되는지 정량 신호로 만드는 것이었습니다. LUCID는 응답 토큰이 retrieved subgraph의 노드·엣지에 주는 attention 점수를 계층·헤드별 행렬로 추출한 뒤, 관계 임베딩과 질의 임베딩의 cosine similarity로 의미 점수를 보강하고, 이를 GNN(GINE)에 넣어 그래프 단위 hallucination 확률을 예측합니다.

- **Empirical Impact**: 평가 결과 LUCID는 9개 데이터셋에서 15개 베이스라인을 상대로 SOTA 성능을 달성했으며, 세 프레임워크 전반에서 안정적으로 우수했습니다. 또한 hallucination 확률을 QA 후처리에 활용한 Mixed 전략은 고가 API 모델만 전면 적용했을 때와 비슷한 EM 정확도를 유지하면서 비용을 평균 55.4% 절감해, 신뢰도 측정이 실제 운영 효율 개선으로 이어질 수 있음을 보여줍니다.



### Where to Place the Query? Unveiling and Mitigating Positional Bias in In-Context Learning for Diffusion LLMs via Decoding Dynamics (https://arxiv.org/abs/2606.19349)
Comments:
          9 figures, 4 tables

- **Prior Approaches**: 기존 ICL 연구는 주로 example selection과 example ordering에 초점을 맞추며, dLLM에서는 쿼리(질문) 삽입 위치를 사실상 고정된 포맷(대개 trailing)으로 취급해 왔습니다. AR LLM에서는 causal masking 때문에 테스트 쿼리가 시퀀스 맨 끝에 고정되지만, dLLM은 bidirectional attention 덕분에 위치를 바꿀 수 있어 구조적 전환이 필요합니다. 그럼에도 대부분의 실무는 AR 스타일 템플릿을 그대로 답습해 위치 최적화 가능성을 놓쳤습니다.

- **Core Contribution**: 이 논문은 dLLM에서 query placement(질문 삽입 위치)가 성능에 대해 first-order variable이라는 점을 실증적으로 규명합니다. 특히 위치를 바꾸는 효과가 데모 의미를 교체하는 효과와 비슷한 수준일 수 있음을 보여주며(GSM8K에서 r≈1.236), 최적 위치가 과제 유형에 따라 달라진다고 정리합니다. 결론적으로 dLLM ICL을 “포맷 고정”이 아니라 “공간 토폴로지 최적화” 문제로 재정의합니다.

- **Technical Challenges**: 핵심 난제는 레이블 없이(ground-truth 없이) 어떤 query 위치가 생성 안정적인지 판별하는 신호가 기존에 없다는 점입니다. 단일 스텝 confidence(C_decoded)는 dLLM의 iterative decoding 진화(temporal evolution)를 버려 위치 랭킹을 실패하게 만들며, 이를 해결하기 위해 전체 디코딩 궤적을 누적하는 Average Confidence(\bar{C})를 제안합니다. 이후 \bar{C}로 후보 위치들에 대한 안정성을 점수화해 학습 없이 Auto-ICL(적응형 위치 라우팅)을 구성합니다.

- **Empirical Impact**: 실험에서는 Sudoku(전역 지각)와 GSM8K(순차 추론)처럼 인지 패러다임이 다른 벤치마크에서, Auto-ICL이 정적 배치 대비 최적 성능에 근접하거나 능가하는 결과를 보입니다. 예를 들어 Sudoku에서는 prefix 경계 최적 성향을 회복하고, GSM8K에서는 순차 추론에 필요한 trailing 성향을 유지해 성능을 동시에 방어합니다. 또한 추가 지연은 미미한 편(예: GSM8K에서 약 +0.08s)이며, 제한된 generation budget이나 shot 수 변화 같은 악조건에서도 인스턴스 단위 라우팅이 정적 기준선을 넘는 사례를 확인해 실용성이 강조됩니다.



### DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligenc (https://arxiv.org/abs/2606.19348)
- **Prior Approaches**: 기존 reasoning 모델은 test-time scaling을 통해 성능을 끌어올리지만, vanilla attention의 제곱 복잡도 때문에 초장문(예: 100만 토큰)에서는 계산·메모리 병목이 커집니다. 오픈소스 LLM들도 long-horizon 작업을 지원하려 했으나, 근본적으로 초장 시퀀스 효율을 크게 개선하지 못해 추가 확장이 제한돼 왔습니다.

- **Core Contribution**: DeepSeek-V4 시리즈는 DeepSeekMoE와 MTP 설정은 유지하면서, 초장맥락 효율을 극적으로 낮추는 새로운 아키텍처·최적화를 결합해 100만 토큰 컨텍스트를 “실제로” 가능하게 합니다. DeepSeek-V4-Pro(1.6T, activated 49B)와 DeepSeek-V4-Flash(284B, activated 13B) 두 라인업을 제시하며 long-context 및 추론 모드의 성능/비용 균형을 함께 겨냥합니다.

- **Technical Challenges**: 초장문에서는 attention의 KV cache와 추론 FLOPs가 지배 항이 되므로, 계산량·저장량을 동시에 줄이면서도 품질을 유지하는 설계가 핵심 난제입니다. 논문은 CSA(Compressed Sparse Attention)와 HCA(Heavily Compressed Attention)를 하이브리드로 엮어 KV를 압축·선택적으로 참조하게 만들고, mHC(Manifold-Constrained Hyper-Connections)와 Muon optimizer, FP4 양자화 인지 학습 및 MoE 인프라 최적화를 추가해 안정성과 효율을 동시에 확보했다고 설명합니다.

- **Empirical Impact**: 100만 토큰 조건에서 DeepSeek-V4-Pro는 DeepSeek-V3.2 대비 single-token 추론 FLOPs를 27%, KV cache를 10%로 낮추며, DeepSeek-V4-Flash는 FLOPs 10%, KV cache 7% 수준까지 더 줄입니다. 벤치마크에서는 DeepSeek-V4-Pro-Max가 오픈 모델 대비 추론·코딩·롱컨텍스트 전반에서 강세를 보이고, 일부 knowledge 영역에서는 근소 우위를 보이면서도 Gemini급 격차를 상당히 좁혔다고 보고해 open models의 장문 추론 기준선을 끌어올렸다는 의미가 큽니다.



### How LLMs Fail and Generalize in RTL Coding for Hardware Design? (https://arxiv.org/abs/2606.19347)
Comments:
          Preview, under submission for EMNLP 2026

- **Prior Approaches**: 기존 연구는 RTL(Verilog) 코드를 잘 생성하도록 SFT나 RL fine-tuning을 적용하거나, RTL 데이터셋을 확장/도메인적응시키는 방식이 주를 이뤘습니다. 하지만 평가 단계에서 실패 원인을 체계적으로 분해하지 못해, 왜 한계가 생기는지(지식 부족 vs 단순 형식 오류)를 관찰하기 어렵다는 문제가 남아 있습니다. 또한 alignment나 보상 최적화가 실제로 ‘컴파일’만 개선하는지, ‘검증을 통과하는 기능’을 늘리는지는 명확히 정리되지 않았습니다.

- **Core Contribution**: 이 논문은 인지이론에서 영감을 받은 문제 solvability(해결 가능성) 관점의 4단계 오류 분류 체계를 제안합니다. L1 구문(syntactic), L2 의미(semantic), 그리고 기능(functional) 오류를 L3S(해결 가능하지만 현재 샘플에서 실패)와 L3U(어떤 롤아웃에서도 실패)로 나눠, 테스트벤치에서 드러나는 지식 격차를 구분해냅니다. 이 프레임워크로 LLM의 RTL 생성 실패를 ‘오류 파이프라인’ 관점에서 정량화할 수 있게 했습니다.

- **Technical Challenges**: 핵심 기술적 난제는 RTL의 병렬 temporal 논리를 순차 실행처럼 추적하기 어렵고, 따라서 실패 원인이 단순 문법/타이핑 실수인지 더 깊은 기능적 추론 부족인지 분해해야 한다는 점입니다. 저자들은 HDL 컴파일·정적 제약(엘라보레이션/린팅/합성)과 테스트벤치 시뮬레이션(명세 충족)을 단계적으로 매핑해, 오류를 L1→L2→L3S/L3U로 배타적으로 분할하는 프레임워크를 구축했습니다. 또한 GRPO 기반 RL fine-tuning 동안 롤아웃 분포가 어떤 오류 레벨로 이동하는지까지 추적해, ‘선택 압력’이 오류 파이프라인을 어떻게 밀어내는지 분석합니다.

- **Empirical Impact**: VerilogEval-Human(VerilogEval) 평가에서 frontier 모델들은 초기 pass rate가 90.8%에서 포화되며, 이 한계는 주로 L3U(unsolvable functional errors)에서 기인하는 것으로 드러납니다. 더 나아가 SFT/RL은 L1+L2를 줄이지만 L3(특히 기능 오류)를 늘리는 경향이 있어, 모델이 ‘컴파일은 잘하지만 하드웨어 이해(검증 통과)는 부족한 상태’로 남는다는 메시지를 강화합니다. 결론적으로 정렬(alignment)만으로는 RTL 파이프라인의 근본 병목을 넘기 어렵고, 모델 추론 능력에 대한 추가 연구가 필요하다는 실증적 근거를 제공합니다.



### Disentangling Linguistic Relatedness from Task Alignment in Cross-Lingual Transfer (https://arxiv.org/abs/2606.19346)
- **Prior Approaches**: 기존 교차언어 전이는 mBERT/XLM-R처럼 공유 표현을 학습하거나, 모델 스케일·instruction tuning이 전이를 좌우한다고 보는 관점이 많았습니다. 다만 decoder-only LLM로 오면 용량 배분, multilinguality의 저주, fine-tuning이 기존 다언어 표현을 덮어쓰는 현상(catasrophic forgetting) 등이 얽혀 결과 해석이 복잡해졌습니다. 특히 Semitic 언어(아랍어-히브리어-암하라어-몰타어) 연구는 대부분 인코더 중심·단일/이중 언어에 치우쳐, 추론이 중요한 제로샷 독해에서의 전이 양상은 덜 탐구돼 왔습니다.

- **Core Contribution**: 이 논문은 아랍어 방언 데이터로 7개 대형 LLM(4B~671B)을 fine-tuning한 뒤, Semitic 언어(히브리어/암하라어/몰타어)와 비Semitic 대조군(일본어/한국어/프랑스)에서 Belebele 벤치마크 제로샷 독해를 비교합니다. 언어 계통·스크립트(아브자드/아부기다/라틴/CJK)를 함께 흔들어, “아랍어 fine-tuning이 Semitic 지식만 골라 옮기는가”를 통제된 설계로 검증합니다. 결과적으로 Semitic-specific 전이 근거는 없고, 성능 향상은 전반적으로 task-format alignment(평가 포맷에 맞춘 정렬)와 더 잘 맞는 그림을 보여줍니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) train-test overlap을 엄격히 통제하고 (2) 언어 계통/스크립트 효과를 분리해 (3) ‘정말로 교차언어 지식 전이’인지 ‘포맷 캘리브레이션 개선’인지 구분하는 것입니다. 이를 위해 Belebele의 공통 질문 ID 기반 split으로 평행성을 유지했고, 평가도 자유생성 대신 logprob 기반으로 1~4 정답 토큰을 판별해 결정성을 확보했습니다. 또한 CoT(Chain-of-Thought) ablation을 통해 fine-tuning 없이도 추론 컨텍스트를 붙였을 때 같은 모델들이 유사한 수준의 이득을 얻는지 확인해, 두 메커니즘이 동일한 병목(포맷 정렬/캘리브레이션)을 보정한다는 해석을 강화했습니다.

- **Empirical Impact**: 실험에서는 베이스라인이 약한 모델(특히 MoE 계열 GPT-OSS-120B/20B)이 아랍어 fine-tuning 후 대폭 개선하지만, 그 이득이 Semitic에만 몰리지 않고 비Semitic 대조군에서도 거의 같은 크기로 확장됩니다. 반대로 베이스라인이 강한 모델(DeepSeek-V3.1, Qwen3-32B)은 미세한 개선만 보이며, baseline을 통제한 회귀에서도 Semitic 지표가 오히려 부정적(유의미한 우위 없음) 계수로 나타났습니다. CoT 프리펜딩 ablation은 fine-tuning에서 가장 많이 이득을 본 모델이 추론에서도 동일하게 많이 이득을 봄을 보여, “Semitic 전이”로 보이던 현상이 사실은 task-format alignment/분포 캘리브레이션 보정일 수 있음을 시사합니다.



### Ensembles of Large Language Models for Identifying EQ-5D Studies in PubMed Based on Their Abstracts (https://arxiv.org/abs/2606.19345)
Comments:
          6 pages, 7 tables, 8 equations

- **Prior Approaches**: 체계적 문헌고찰(SLR)에서 포함/제외 기준을 만족하는 논문을 사람이 선별하면 속도·일관성·오류 문제가 커집니다. 특히 EQ-5D처럼 임상적 해석이 필요한 건강 관련 삶의 질 지표는 초록만으로 판별하기가 어려워 기존 자동화 NLP/LLM 연구의 성능 신뢰성 확보가 핵심 과제로 남아있습니다. 기존 LLM 기반 선별은 성능은 높아도 과업 표준화와 도메인(바이오메디컬) 텍스트 처리 한계가 자주 지적됩니다.

- **Core Contribution**: 이 논문은 PubMed의 바이오메디컬 초록만을 입력으로, EQ-5D 결과 보고 여부를 자동 탐지하는 앙상블 LLM 프레임워크를 제안합니다. few-shot prompting으로 기본 분류 능력을 끌어올리고, 모델별 예측을 weight ensembling 및 soft stacking(로지스틱 회귀 메타 분류기)으로 결합해 신뢰성과 균형을 동시에 노립니다. 최종적으로 gemini-2.5-pro, gemma-3-12b, gemma-3-27b 3개 모델 조합이 최우수 성능을 보였다고 보고합니다.

- **Technical Challenges**: 초록에 EQ-5D를 ‘명시적으로’ 언급하는 경우만 양성으로 라벨링해야 하므로, 도메인 특화 신호를 잡는 능력과 확률의 신뢰도 보정이 함께 필요합니다. 또한 모델마다 편향이 달라 단순 투표만으로는 정밀도-재현율 균형이 흔들릴 수 있어, F1 기반 가중치와 confidence를 함께 사용하는 weight ensembling을 설계했습니다. 더 나아가 soft probabilities와 raw confidence를 메타 특징으로 넣은 soft stacking에서 로지스틱 회귀로 조합 가중치를 학습해 일반화 및 해석가능성을 높였습니다.

- **Empirical Impact**: 수작업 라벨(전문가 2인 검토) 200편 데이터셋에서 개별 9개 Gemini/Gemma 모델 중 gemini-2.5-pro가 가장 높은 weighted F1(0.71)을 기록했고, 상위 3개 모델을 앙상블했을 때 weighted F1과 accuracy가 각각 0.74로 향상됐습니다. soft stacking은 weighted F1 0.72, accuracy 0.73으로 유사한 수준의 성능을 유지하며 메타 특징 중요도 분석에서 LLM의 확률(soft probability)이 판별에 가장 큰 기여를 한다는 점을 보여줍니다. 런타임과 비용 추정에서도 모델 크기 선택에 따라 7~64분 및 0.07~5.04달러 수준으로 스케일링 가능성을 제시해, 바이오메디컬 SLR 자동 선별의 실용성을 강화합니다.



### Exposing the Unsaid: Visualizing Hidden LLM Bias through Stochastic Path Aggregation (https://arxiv.org/abs/2606.19344)
Comments:
          14 pages

- **Prior Approaches**: 기존 편향 감사는 단일 출력만 점검하거나 WEAT/SEAT 같은 정적 자동 지표에 의존해, 확률 분포의 내부 분기(낮은 확률 경로)에 숨은 편향을 놓치기 쉽습니다. 또한 고정 템플릿 기반 카운터팩추얼 probing도 문법적 스퓨리어스 상관을 유발해 의미 기반 진단의 신뢰도를 흔들 수 있습니다.

- **Core Contribution**: TreeTracer는 온톨로지(의미 변수를 반영한 단어 집합)를 프롬프트의 특정 토큰 자리에 체계적으로 치환하고, 수백 개의 확률적 생성 결과를 문법 구조에 맞춰 집계 비교하는 시각 분석 워크스페이스를 제안합니다. 두 온톨로지 트리(예: 남성/여성 이름)의 차이를 Sankey 다이어그램 기반으로 나란히 보여주며, 무엇이 “사라졌는지”가 아니라 “확률 질량이 어디로 이동했는지”를 추적하도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 (1) 수많은 stochastic generation을 읽을 수 있는 구조로 줄이면서도 (2) 낮은 빈도의 숨은 확률 질량을 보존하고 (3) 토큰 확률을 단순 합산이 아닌 의미-맥락까지 반영해 일관되게 렌더링하는 것입니다. TreeTracer는 constituency 파서로 문법 트리 집계를 수행하고 하위워드 확률은 기하평균으로 재조정하며, 의미 카테고리 기반 노드 머징과 decoupled Sankey 인코딩(노드 높이=global probability, 링크 폭=selected sample probability)을 통해 가지치기 이후의 손실을 시각적으로 드러냅니다.

- **Empirical Impact**: 검증에서는 GPT-2 XL(unaligned)과 constitutionally aligned Apertus 모델을 비교해 대명사 억제(counterfactual pronoun suppression)와 특정 개인에 대한 대화 맥락의 주변화 같은 representational harms를 “숨은 확률 분기”까지 포함해 드러냈습니다. 또한 예비 사용자 연구에서 집계 비교 인터페이스가 분석가의 인지 부하를 줄이면서 체계적 편향 탐지에 효과적임을 확인했습니다.



### Human-AI Agent Interaction in a Business Contex (https://arxiv.org/abs/2606.18716)
Comments:
          9 pages, 5 tables, 1 figure, submitted to Springer Nature

- **Prior Approaches**: 기존 HCI/UX는 주로 비지능형 시스템의 사용성·일관성·효율에 초점을 맞췄고, AI 시대에는 controllability, explainability, ethics, privacy, transparency 같은 원칙이 강조돼 왔다. 다만 대다수 프레임워크는 개념적 가이드 중심이며, 기업의 agentic AI(자율 계획·맥락 인지·자연어 상호작용) 환경에 바로 적용 가능한 ‘측정 가능한 UX 기준’이 부족했다. 또한 AI UX 권고가 주로 소비자 제품에 치우쳐 있고, 사회기술적·조직적 조건(책임소재, 감사 가능성, 워크플로 제약)을 정량적으로 연결한 연구도 제한적이었다.

- **Core Contribution**: 이 논문은 기업 맥락에서 ‘긍정적 human-AI agent interaction’에 필요한 UX 원칙을 도출·검증하고, 각 원칙을 제품 의사결정에 쓸 수 있는 측정 기준으로 구체화한다. 여러 방법(참여설계 워크숍, 질적 메타분석 기반 프레임워크 구축, 설문 우선순위 검증, 심층인터뷰)을 통해 최종적으로 8개 핵심 UX 원칙과 하위 기준을 제시한다. 특히 인간 통제(Human Control), 신뢰성·안전·견고성(Reliability/Safety/Robustness), 데이터 프라이버시·거버넌스, 맥락 인지(Context-Awareness)가 상위 우선순위로 확인됐다.

- **Technical Challenges**: 핵심 기술적 난제는 자율적·때때로 불투명한 행동을 보이는 agent를 ‘UX 원칙’에서 ‘검증 가능한 설계 요소’로 전환하고, 기업 워크플로에서 인과적으로 효과를 측정하는 것이었다. 연구진은 탐색 단계에서 원칙-기준을 정교화한 뒤, 선택형 conjoint experiment로 Human Control의 하위 기준(에이전트 중지/일시정지, 추론·행동의 투명성, 인간 책임·의사결정)을 수준별 화면 프로토타입에 반영해 AMCE로 효과를 추정했다. 또한 다중가설검정 보정, 응답 피로(초기 과제만 사용 등), 응답시간 기반 이상치 제거 같은 강건성 점검을 수행해 결과의 안정성을 확인했다.

- **Empirical Impact**: 실험은 SAP 내부·외부 사용자 107명, 총 1,139개 선택 과제를 바탕으로 진행됐고, 기준 수준이 올라갈수록 선호 확률이 유의미하게 증가함을 보여줬다. 다만 ‘중지/일시정지’의 영향은 상대적으로 작았고, 그중에서도 ‘투명성(투명한 워크플로·추론·다음 행동·데이터 출처 상세 제공)’이 가장 큰 선호 증가를 만들었다(선택 확률 약 37.79%p). 결론적으로 이 연구는 기업용 agent 설계에서 인간 통제·책임·신뢰/안전·프라이버시·맥락 통합을 1순위 요구사항으로 다루되, 특히 투명성을 ‘과부하를 일으키지 않는 범위에서’ 계층적으로 설계해야 한다는 실증 근거를 제공한다.



New uploads on arXiv(cs.RO)

### MemoryWAM: Efficient World Action Modeling with Persistent Memory (https://arxiv.org/abs/2606.20562)
- **Prior Approaches**: 기존 VLA는 관측→행동을 직접 매핑하는 경우가 많아, 과거 정보와 환경의 동역학 변화를 충분히 모델링하지 못해 비마르코프(non-Markovian) 장기 과제에 취약했다. WAM은 세계의 미래를 함께 예측해 메모리 의존 의사결정을 돕지만, 효율형은 최근 고정 윈도우에만 의존해 장기 단서를 놓치고, 전체 히스토리를 캐시하는 방식은 길이에 따라 지연·GPU 메모리가 급증하는 한계가 있었다.

- **Core Contribution**: 본 논문은 세계행동모델 world action model(WAM)에 효율적인 persistent memory를 붙인 MemoryWAM을 제안한다. MemoryWAM은 최근 프레임(슬라이딩 윈도우), event boundary의 anchor frame, 장기 히스토리를 요약하는 gist token을 함께 써서 장기 의존성을 유지하면서도 계산비용을 억제한다.

- **Technical Challenges**: 핵심 과제는 “긴 히스토리 의존”을 살리되, 전체 KV 캐시를 유지하지 않고도 retrieval이 되도록 메모리를 설계하는 것이다. 이를 위해 MemoryWAM은 하이브리드 attention과 마스킹/eviction을 맞춤화해, 최근·anchor는 고정밀로 남기고 나머지는 gist token의 압축 표현으로 대체하여 GPU 메모리와 추론 지연을 줄였다.

- **Empirical Impact**: RMBench의 장기·메모리 의존 조작에서 MemoryWAM은 평균 성공률이 기준선 대비 약 70%p 높고, LingBot-VA(풀히스토리 WAM)보다도 개선되며 정확도와 효율을 동시에 확보했다. 시뮬레이션뿐 아니라 실제 로봇에서도 성능 우위를 보였고, 특히 지연이 큰 풀히스토리 계열 대비 더 안정적으로 컵 스왑 같은 이벤트를 놓치지 않는 결과가 제시됐다.



### Generating Robot Hands from Human Demonstrations (https://arxiv.org/abs/2606.20549)
- **Prior Approaches**: 로보틱스 협업 설계(co-design) 연구는 형태(embodiment)와 제어를 함께 최적화하며 발전해왔지만, 하드웨어 변경이 곧바로 제어 적합성을 흔들어 반복적인 적응/재학습이 필요해지곤 합니다. 또한 기구 설계 공간을 탐색하려면 비용이 크고(비볼록/초기화 민감), 목표가 단일 동작이 아닌 “넓은 조작 행동 집합”일 때 탐색이 더 어려워집니다. 사람 손 데이터를 이용한 학습은 주로 텔레오퍼레이션·retargeting·imitation에 쓰였지만, 선택된 기존 손 형태가 만드는 근본적인 kinematic mismatch를 제거하기엔 한계가 있었습니다.

- **Core Contribution**: 이 논문은 사람의 엄지-검지 fingertip 궤적을 “제어 학습의 감독”을 넘어 “로봇 손 형태 생성의 기준(prior)”으로 사용하는 데이터 기반 프레임워크를 제안합니다. 핵심은 제작 후에는 inverse kinematics로 fingertip을 맞추는 간단한 제어 정책이 고정된다는 점에 맞춰, 설계 단계에서부터 동일한 IK 전제 하에 손의 kinematics를 최적화한다는 것입니다. 이를 통해 6-DoF 범용 손과, spatial four-bar mimic joints를 갖는 저-DoF 과업 특화 손을 같은 틀로 생성합니다.

- **Technical Challenges**: 문제는 설계 파라미터와 관절 궤적을 동시에 탐색하는 비볼록 최적화가 Bennett(공액) 모사 관절의 닫힘 제약까지 포함하면서 초기화에 극도로 민감해진다는 점입니다. 저자들은 이 병목을 trajectory-conditioned actor로 해결해, 목표 fingertip 궤적에 조건화된 초기 설계/각도 씨앗을 샘플링한 뒤 제한된 단계의 differentiable GD로 정련(refinement)하는 제안-정련 루프를 구성합니다. 또한 충돌을 메시 대신 링크 centerline segment 거리로 근사하는 손실을 써 탐색 중 계산을 가볍게 하면서, 닫힘 제약은 gradient-friendly한 soft residual 방식으로 완화해 학습 안정성을 높였습니다.

- **Empirical Impact**: OakInk에서 4백만 프레임 규모의 일상 조작 데이터를 사용해, 생성된 6-DoF 손은 평균 fingertip tracking 오차가 0.24mm 수준이며 11mm 이내 추적 커버리지도 높게 나타났습니다. 단순 DoF 증가가 성능을 보장하지 않는다는 점도 상용 손(Inspire Hand, XHand) 비교로 확인되며, “목표 모션 분포에 맞춘 하드웨어 형상”이 이득을 만든다는 결론을 지지합니다. 더 나아가 3-DoF mimic-joint 과업 특화 손은 특정 구조화된 궤적에서 정확도를 개선하면서 기계 복잡도를 줄였고, actor 기반 초기화로 하드웨어 탐색 시간을 수시간에서 수십 분으로 줄여 iterative embodiment 설계를 현실화하는 의미가 있습니다.



### Increasing Resilience of Continuum Robots via Motion Planning Algorithms (https://arxiv.org/abs/2606.20495)
- **Prior Approaches**: 기존의 연속체 로봇(continuum robot) 경로계획은 주로 최단거리나 충돌 회피 같은 단일 기준에 집중해 복구 가능성(resilience)까지 함께 최적화하기가 어렵다는 한계가 있었다. 또한 다기준을 넣더라도 경로의 품질과 실행 시간의 균형을 정량적으로 설명하기가 쉽지 않았다.

- **Core Contribution**: 이 논문은 연속체 로봇의 resilience를 높이기 위해 다기준 의사결정(multicriteria decision-making)을 경로계획에 통합하는 실험적 프레임을 제안한다. Genetic algorithm과 A star 두 경로계획 알고리즘에 Analytical Hierarchy Process(Analytical Hierarchy Process, AHP)를 결합해 거리, 모터 손상, 로봇 암의 기계적 손상, 정확도 4개 기준으로 경로 품질을 평가한다.

- **Technical Challenges**: 핵심 기술 난제는 여러 기준을 동시에 반영하면서도 계산 비용과 실행 시간을 과도하게 늘리지 않는 것이다. 저자들은 로봇 모델과 환경을 단순화하되 실제 프로토타입의 환경 특성을 일부 재현하고, 단일/다중 경유점(single- 및 multi-path points)과 다중 경유점-only 같은 서로 다른 시뮬레이션 설정에서 알고리즘별 성능을 비교해 AHP의 영향(경로 선택과 유지보수 시간 관점)을 확인한다.

- **Empirical Impact**: 실험 결과 A star는 환경의 cardinality(경유점 수/복잡도)에 따라 성능 시간이 달라지는 반면, Genetic algorithm은 환경 cardinality에 대한 의존이 크지 않았다. 또한 Genetic algorithm은 더 다양한 경로를 생성해 결과적으로 로봇의 resilience를 높이는 경향이 관찰되어, 다기준 평가(AHP) 결합 전략이 연속체 로봇의 유지보수 친화적 경로계획에 의미가 있음을 시사한다.



### Fast Human Attention Prediction for Fixation-guided Active Perception in Autonomous Navigation (https://arxiv.org/abs/2606.20491)
Comments:
          Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)

- **Prior Approaches**: 기존 scanpath 예측은 정적 saliency 맵 기반에서 출발해, 이후에는 다음 고정점이 시각 자극과 과거 gaze 기록에 의존하도록 RNN/ConvLSTM, 최근엔 Transformer로 확장됐다. 다만 이러한 모델들은 계산량과 추론 지연이 커서 임베디드 로보틱스에서 실시간 embodied deployment가 어렵다는 한계가 있었다. 또한 attention 예측을 로봇의 active vision/항법에 결합하더라도, human-fixation 같은 시간적 주의 구조를 정책에 직접 최적화하는 방식은 상대적으로 초기에 머물러 있었다.

- **Core Contribution**: 이 논문은 가벼운 scanpath 예측 모델 GazeLNN을 제안한다. MobileNetV3로 특징을 뽑고, recurrent engine으로 Liquid Neural Networks(LNN)의 CfC(Closed-Form Continuous-time) 변형을 써서 고정점 heatmap을 auto-regressively(자기회귀적으로) 순차 생성한다. 추론 비용을 극적으로 줄이면서도 MIT Low Resolution에서 ScanMatch 0.47의 SOTA를 달성해, 인간 주의 모델링을 로봇 자율성에 실용적으로 연결하는 발판을 만든다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 순차적 gaze를 잘 모델링하면서도 (2) 실시간 로보틱스 제약(연산량/지연)을 만족하는 경량화였다. GazeLNN은 fixation을 (x,y) 좌표가 아닌 heatmap으로 표현하고 CoordConv로 공간 좌표 정보를 보강했으며, CfC의 입력-의존 시간 동역학과 gated time update로 fixation 간 시간 구조까지 반영해 정확도와 효율을 동시에 노렸다. 여기에 파라미터와 GFLOPs를 크게 줄이기 위해 MobileNetV3 백본을 채택하고, 학습에는 OSIE 기반의 시퀀스 학습/손실(KL-DTW)을 사용했다.

- **Empirical Impact**: GazeLNN은 0.61 GFLOPs, 15.24M 파라미터로 동작하며 tSPM-Net 대비 연산비용 99.40% 절감과 최대 6배 빠른 추론을 보고한다. 예측 성능은 MIT Low Resolution에서 여러 곡선/문자열/시간열 메트릭(ScanMatch, DTW, TDE 등) 전반에서 recurrent 기반 베이스라인을 능가하며, 정성적으로도 인간 ground truth scanpath에 더 가깝게 예측됐다. 나아가 RL 기반 active camera-robot 제어 정책에 GazeLNN을 통합해 드론 환경에서 예측된 fixation을 따라가도록 하며, 실제 비행 실험에서 정적 카메라 대비 salient-relevant 관찰 및 글로벌 맵 누적(약 50%↑, salient 영역 관찰 8배↑)을 통해 실사용 가치를 검증했다.



### GroundControl: Anticipating Navigation Failures in Vision-Language Agents via Trajectory-Consistent Uncertainty Estimates (https://arxiv.org/abs/2606.20479)
- **Prior Approaches**: 비전-언어 내비게이션(VLN) 에이전트는 benchmark에서 Success 및 SPL로 경쟁력을 보이지만, 실제 배치 관점에선 에피소드가 진행되는 동안 실패를 예측·중단할 수 있는 불확실성 신호가 부족하다는 한계가 지적된다. 기존 불확실성 프록시는 action 분포의 predictive entropy나 토큰/로짓 기반 confidence처럼 즉시적인 신호에 치우쳐, 궤적이 oscillation·stagnation·불필요한 우회로 같은 실패 모드로 변질되는지까지 일관되게 반영하지 못한다.

- **Core Contribution**: 이 논문은 실패가 “행동 단위”의 모호성뿐 아니라 “거리-투-골(goal-directed distance-to-goal) 동역학의 궤적-일관성” 붕괴로 나타난다는 관점을 제시한다. 이에 따라 GroundControl은 에피소드 전반에서 거리 진화가 기대한 운동(기하적 진행)을 통계적으로 얼마나 벗어나는지로 trajectory-consistent uncertainty를 계산한다. 또한 성공/실패 여부와 무관하게 불확실성의 랭킹 품질을 측정하기 위해 Selective Risk–Coverage Navigation(SRCN) 프로토콜을 도입해, 위험-커버리지 곡선과 AURC/E-AURC로 평가를 표준화한다.

- **Technical Challenges**: 핵심 기술 과제는 “로컬 action 예측 분산”이 아니라 “에피소드 전체에서 관측되는 기하·시간적 일관성 위반”을 불확실성 스코어로 바꾸는 것이다. 저자들은 이를 위해 constant-velocity Kalman filter로 거리-to-골의 동역학을 근사하고, normalized innovation 통계(정규화 혁신 에너지)와 posterior belief dispersion의 변화가 기대 동역학에서의 통계적 일탈을 의미하도록 설계했다. 여기에 progress/monotonicity/path efficiency/oscillation(행동 되돌림) 같은 궤적 특징을 결합해, 국소 예측 흔들림이 아닌 “시간 누적된 궤적 불일치”를 반영하도록 구성했다.

- **Empirical Impact**: EB-Navigation 5개 split(총 N=300)에서 GroundControl은 success-based selective risk 기준으로 near-oracle ordering을 보이며, GPT-4o에 대해 weighted-average E-AURC_SR=0.0024로 entropy·conformal·휴리스틱 베이스라인을 크게 앞섰다. SPL 기반 평가에서도 AURC/E-AURC가 전반적으로 가장 낮아, 단순 파국적 실패뿐 아니라 비효율적 우회·진동처럼 “점진적 저하”를 조기 식별하는 능력이 드러났다. 특히 long_horizon에서 entropy/conformal 신호가 거의 랜덤에 가까워질 때도 trajectory-consistent 신호는 오라클 경계에 근접해, 시간 누적 실패 모드에 강건한 의미를 갖는다.



### Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban Navigation (https://arxiv.org/abs/2606.20458)
- **Prior Approaches**: 학습 기반 로컬 플래너는 5–20Hz로 충돌 회피와 동적 실현 가능 후보 궤적을 빠르게 생성하지만, 그중 무엇을 실행할지 ‘궤적 스코어링’에서 실패하기 쉽다. 기존 접근은 기하학적 손실(유사도, 매끈함, 충돌 프록시)로 스코어 함수를 학습하거나 imitation 기반으로 랭킹을 맞추는데, 이들은 장면의 의미적 맥락(교차로에서의 애매함, 보행자 사회적 규범, 지형 경계)을 충분히 해석하지 못해 성능 격차가 생긴다.
또한 VLM을 제어 루프에 넣는 end-to-end Vision-Language-Action은 보통 1–2s 수준 추론 지연 탓에 5–20Hz 제어와 동기화가 어렵다.

- **Core Contribution**: 논문은 ‘trajectory scoring gap’을 줄이기 위해, 플래너를 end-to-end로 교체하지 않고 VLM을 후보 선택기로만 써서 의미적 우선순위를 반영하는 VLM-Planner 인터페이스를 제안한다. 핵심은 VLM이 직접 조작을 출력하는 것이 아니라, 플래너가 생성한 K개 후보 중 인덱스 k를 고르는 방식이라 후보 자체의 기하/충돌 안전성을 유지한다.
하지만 VLM 선택은 stale(지연된) 정보이므로, 이를 실시간 제어에 쓸 수 있게 만드는 학습-free ‘Score Fusion’과 (문제 안정성을 위한) ‘Probability Fusion’ 레이어가 기여의 중심이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 1–3s VLM 지연으로 인해 선택 결과가 매 순간 생성되는 새 후보와 시간적으로 어긋난다는 점이다. 논문은 지연된 VLM 선택의 ‘의도’를 기하학적 유사도로 현재 후보 점수에 반영하되, 시간 경과에 따라 지배력이 지수 감쇠되도록 설계해 지연에도 연속 제어가 가능하게 했다.
또한 Score Fusion은 유사도 항이 커질 때 불안정할 수 있어, VLM-플래너 분포를 혼합하는 Probability Fusion으로 영향 상한(α≤1)을 두고 하이퍼파라미터 민감도를 낮췄으며, VLM streaming(고정 주기로 비동기 쿼리 전송)으로 로봇이 대기하지 않게 했다.

- **Empirical Impact**: 현장 로그 기반의 약 2,000개 어려운 시나리오에서 VLM 선택은 플래너 argmax 대비 ADE를 30% 줄였고, 다만 routine 상황에서는 플래너가 이미 잘 맞아떨어져 VLM 단독 선택이 오히려 흔들릴 수 있음을 보여 ‘퓨전’ 필요성이 뒷받침된다. 지연을 최대 5s까지 준 시뮬레이션에서도 Probability Fusion with streaming은 성공률 80% 이상 수준을 유지했으며, stale trajectory를 그대로 실행하는 방식은 빠르게 붕괴했다.
실로 구현한 캠퍼스 사이드워크 주행(세포룰 네트워크 지연 1.5–3s)에서는 Probability Fusion이 사람 개입 빈도를 크게 낮추고(예: takeovers per 100m 3.49→0.87), 로봇이 ‘의도적으로’ 움직이는 질적 개선도 관찰됐다.



### ARC: Adaptive Robust Joint State and Covariance Estimation (https://arxiv.org/abs/2606.20428)
Comments:
          Submitted to information IEEE Robotics and Automation Letters (RA-L), June 2026. 8 pages, 7 figures, 1 table

- **Prior Approaches**: 기존의 weighted least-squares 기반 상태추정은 잡음이 가우시안이고 측정 공분산이 알려져 있으며 고정되어 있다는 가정을 둔다. 하지만 UWB의 NLOS 환경처럼 outlier와 non-Gaussian 잡음이 섞이면 추정치가 편향되고 불확실성(공분산) 예측이 제대로 캘리브레이션되지 않는 문제가 생긴다.
robust 추정(M-estimator, IRLS)은 outlier를 버리거나 downweight하지만 공분산 자체를 함께 추정하진 못하고, joint state-covariance 접근들은 대개 Gaussian 잔차와 고정된 loss 모양 파라미터에 의존해 noise 형태가 바뀌면 robustness 튜닝이 필요하다.

- **Core Contribution**: 이 논문은 outlier가 존재하고 잡음 통계가 불확실한 상황에서 state와 measurement covariance를 동시에 추정하는 self-tuning 프레임워크를 제안한다. 핵심은 norm-aware adaptive robust loss, IRLS 기반 상태 업데이트, Minimum Weighted Covariance Determinant(MWCD) 공분산 추정을 하나의 Block-Coordinate Descent(BCD) 사이클로 통합해 loss shape 파라미터와 공분산을 함께 적응시키는 점이다.
이를 통해 수동 파라미터 튜닝 없이 inlier 측정 공분산을 복구하고, 동시에 상태 추정 정확도도 확보하는 것을 목표로 한다.

- **Technical Challenges**: 어려움은 (1) state와 covariance가 최적화에서 강하게 결합돼 있어 폐형식 해를 얻기 어렵고, (2) NLOS처럼 heavy-tailed/비대칭 잔차가 생기면 고정 loss나 단순 weighted 공분산이 편향될 수 있다는 데 있다. 저자들은 Mahalanobis distance 기반 잔차에 norm-aware 가중치를 도입해 inlier를 일관되게 식별하도록 하며, 상태 업데이트(IRLS)와 공분산 업데이트(MWCD)가 동일한 가중치 함수로 outlier를 같은 방식으로 취급하도록 맞춘다.
또한 adaptive robust loss에서 loss shape 파라미터를 residual 분포에 맞춰 갱신하고, MCD의 높은 breakdown point를 유지하면서 연속 가중치로 MWCD를 구성해 outlier 프로파일이 복잡할 때도 공분산 재추정의 정확도를 높인다.

- **Empirical Impact**: Monte-Carlo 시뮬레이션과 실제 UWB(초광대역) NLOS 환경 실험에서 제안 방법은 outlier 비율이 최대 50%까지 커져도 RMSE와 공분산 추정 품질(Wasserstein-2 distance)이 거의 유지되었다. 반면 L2는 outlier가 늘수록 성능이 지속적으로 악화했고, Charbonnier/Cauchy 같은 고정 loss baselines는 outlier 증가에 따라 공분산 또는 상태 중 한쪽이 점진적으로 저하되는 양상이 나타났다.
추가 ablation으로 MWCD에서 continuous norm-aware 가중치의 기여도 확인했으며, 전체 프레임워크가 기준선 대비 공분산 복구와 상태 정확도에서 모두 더 낫거나 동등한 결과를 보이며 수동 튜닝 없이 작동함을 실증했다.



### TaCauchy: An Extensible FEM Framework for Vision-Based Tactile Simulation (https://arxiv.org/abs/2606.20426)
Comments:
          Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2026

- **Prior Approaches**: 기존 비전 기반 촉각 시뮬레이터들은 광학적으로는 그럴듯한 이미지를 만들 수 있어도, 표면 압력·마찰 견인력·변형 같은 물리량을 신뢰도 있게 계산하지 못하는 경우가 많았습니다. 일부 FEM 기반 방법은 이미지를 개선하거나 미분가능성을 강조했지만, Isaac 생태계의 대규모 병렬 실행과 “완전한” 응력/힘 분해를 함께 제공하지 못했습니다. 또 학습 기반 역추정(iFEM)은 보정 데이터 의존과 불확실성 문제가 남았고, 단순 접촉 모델은 정확도를 속도와 맞바꾸는 한계를 보였습니다.

- **Core Contribution**: TaCauchy는 Isaac Sim(Isaac Lab) 안에 확장 가능한 FEM(유한요소법) 프레임워크를 결합해, 처음부터 Cauchy stress tensor를 직접 계산하는 “mechanical ground truth”를 제공합니다. UIPC(Unified Incremental Potential Contact) 기반으로 Stable Neo-Hookean 같은 하이퍼엘라스틱 모델에서 응력을 구한 뒤, 접촉면에 투영해 normal pressure와 tangential traction(견인력)을 물리적으로 분해합니다. 그 결과 촉각 이미지 생성뿐 아니라 6D 힘 슈퍼비전을 위한 정밀한 응력 기반 레이블을 한 시스템에서 제공하는 점이 핵심입니다.

- **Technical Challenges**: 문제는 (1) GPU 가속 병렬 학습에 맞는 효율을 유지하면서 (2) 비선형·고주파 연성 접촉에서 응력장을 안정적으로 추출하고 (3) 그 물리 결과를 촉각 이미지 렌더링과 정합시키는 것이었습니다. TaCauchy는 WildMeshing 기반 자동 메쉬 생성과 geometry-aware adaptive refinement로 접촉부에 계산 자원을 집중해 정확도-속도 균형을 잡고, UIPC의 최적화 기반 적분으로 메쉬 inversion/교차 없이 동적 상호작용을 처리합니다. 또한 응력장을 접촉면에 투영하고 normal/tangential로 정교히 분해한 뒤, FEM 변형된 경계로 광학 렌더링을 물리 제약해 시각 신호까지 일관되게 맞춥니다.

- **Empirical Impact**: 성능 측면에서 단일 환경 33.40 FPS(headless)와 60개 병렬에서 555 FPS 집계 처리량을 보고했으며, 응력 추출 오버헤드는 1 ms 미만 수준이라고 합니다. 물리 검증에서는 GelSight Mini를 기준으로 1.2556N~4.7332N 구간에서 시뮬레이션과 실측 촉각 응답이 잘 일치해 SSIM 평균 0.9377(0.93 이상)을 달성했고, 힘-형상 패턴의 정성적 일치도 명확히 보여줬습니다. 더 나아가 DIGIT와 9DTact까지 확장해 센서별 normal pressure와 tangential traction의 기계적 특징을 재현하며, downstream 로봇 조작 학습에서 신뢰 가능한 force supervision과 sim-to-real 전이에 의미 있는 기반을 제공한다는 평가를 받습니다.



### LIT-GS: LiDAR-Inertial-Thermal Gaussian Splatting for Illumination-Robust Mapping (https://arxiv.org/abs/2606.20424)
Comments:
          Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)

- **Prior Approaches**: 3D Gaussian Splatting(3DGS)은 비등방성 Gaussian을 미분가능 splatting으로 최적화해 빠른 신경 렌더링을 가능하게 했지만, 로보틱스 매핑 쪽 파이프라인은 여전히 RGB 기반 포토메트릭 신호와 SfM 초기화에 크게 의존합니다. LiDAR-vision Gaussian 매핑은 LiDAR를 초기화나 약한 regularization에만 활용하는 경우가 많아, 조명 변화나 텍스처 결핍 환경에서 대응 품질이 흔들리며 구조 드리프트와 두께(thickening) 문제가 생기기 쉽습니다.

- **Core Contribution**: 이 논문은 LiDAR-관성-thermal을 결합한 LIT-GS를 제안해 LiDAR 평면(plane) 기하를 pose/structure refinement 단계와 Gaussian 최적화 단계 모두에서 “명시적이고 지속적인 제약”으로 주입합니다. 또한 FAST-LIVO2가 제공하는 불확실성 태그된 LIV 시각 맵 포인트를 thermal-LiDAR의 cross-modal geometric anchor로 삼아, weak thermal supervision에서도 metric-consistent 학습을 유도합니다. 더 나아가 LiDAR-plane-regularized differentiable splatting 목적함수로 렌더링된 3D 포인트가 관측된 국소 평면에 정렬되도록 강제합니다.

- **Technical Challenges**: thermal 영상은 대비가 낮고 열 그라디언트가 약해 포토메트릭 정합만으로는 정밀한 기하 일치를 보장하기 어렵습니다. LIT-GS는 (1) uncertainty-aware anchor로 열-라이다 대응을 신뢰도 있게 만들고 (2) LiDAR point-to-plane residual을 확장 COLMAP-PCD bundle adjustment에 가중치로 포함해 포즈와 3D를 함께 보정하며 (3) 렌더링 학습 단계에서는 plane regularization으로 표면 팽창과 구조 드리프트를 억제하는 방식으로 이 문제를 해결합니다.

- **Empirical Impact**: 사유 데이터셋과 M2DGR 벤치마크의 여러 시퀀스에서 LIT-GS는 LIV-GaussMap과 Thermal3D-GS 대비 기하 정확도(예: EMD)와 렌더링 품질(예: SSIM, LPIPS)을 전반적으로 개선하며, 특히 야간/강한 햇빛/부분 음영 같은 조명 난제로 갈수록 이득이 커집니다. 또한 refinement 모듈을 제거한 ablation에서 포토메트릭 충실도와 기하 일관성이 함께 하락해, 제안한 LiDAR-plane-constrained refinement가 안정적 추정의 핵심임을 실증적으로 뒷받침합니다.



### Agentic AutoResearch forSpace Autonomy: An Auditable, LLM-Driven Research Agent for Aerospace Control Problems (https://arxiv.org/abs/2606.20394)
- **Prior Approaches**: 기존에는 LLM이 아이디어를 내거나(최적화/탐색) 실험 코드를 자동 실행하는 연구 에이전트가 등장했지만, 좋은 결과가 나왔는지 여부를 seed 잡음(임의성) 관점에서 통계적으로 판정하는 장치는 상대적으로 약했습니다. 또한 자동 실험이 빨라질수록 유리한 랜덤 seed를 진짜 개선으로 오인할 위험이 커진다는 재현성 문헌의 경고가 반복돼 왔습니다. 그래서 ‘빠른 실험’만으로는 aerospace 제어 문제에서 신뢰 가능한 보고를 보장하기 어렵다는 한계가 있었습니다.

- **Core Contribution**: AutoResearch는 LLM이 오프라인 연구 에이전트로서 학습 스크립트를 편집·실행·분석해 정책을 개발하되, 어떤 헤드라인 결과도 seed 잡음에 대해 “자격 증명(credibility layer)”을 통과하기 전에는 인정하지 않도록 설계했습니다. 신뢰성 레이어는 (1) 문제별로 측정한 seed noise를 기준선으로 삼고, (2) 최선 구성에 대해 재시드 검증을 하며, (3) 에이전트가 적용한 개별 편집이 성과에 기여했는지 leave-one-out pruning으로 분해해 판정합니다. 즉 ‘탐색(agentiic loop)’과 ‘검증(credibility audit)’을 루프 안에서 분리해, 생성된 정책만 탑재하고 LLM 자체는 차량을 조종하지 않게 합니다.

- **Technical Challenges**: 핵심 기술 난점은 에이전트가 탐색을 가속하더라도, 관측된 성능 향상이 진짜 개선인지 seed-to-seed 변동인지 분리해 계산해야 한다는 점입니다. AutoResearch는 학습 스크립트의 편집 가능한 하이퍼파라미터 영역과 단일 구조화 지표, append-only 실행 로그를 강제하는 ‘family contract’를 두고, 이후 매 반복 결과를 seed noise 단위로 게이트합니다. 또한 랜덤 탐색 기준과 동일한 파라미터 서치 표면 위에서 검증하며, 에이전트 편집의 개별 효과를 배제해도 성능이 남는지까지 확인하도록 설계했습니다.

- **Empirical Impact**: Clohessy–Wiltshire 상대 랑데부와 keep-out zone을 지키는 안전 제약 충돌회피 도킹이라는 두 aerospace 제어 문제에서, 감사(audited)된 정책은 측정된 seed noise를 여러 표준편차에 해당하는 수준으로 초과했습니다. 반면 동일 파라미터에 대해 무방향(undirected) 탐색을 했을 때는 격차가 따라가지 못했으며, 특히 도킹 문제에서는 무방향 탐색이 아예 feasible 정책을 만들지 못한 반면 학습 정책은 모든 seed에서 keep-out zone을 벗어나지 않았습니다. 마지막으로 leave-one-out pruning을 통해 어떤 에이전트 편집이 실제로 성과를 만들었는지까지 드러나, 자동화된 실험이 ‘통계적으로 정직한’ 보고로 이어질 수 있음을 실증합니다.



### CoLI: A Reproducible Platform for Continuum Robot Learning via Monolithic 3D Printing and Isomorphic Teleoperation (https://arxiv.org/abs/2606.20389)
Comments:
          8 pages, 7 figures, 1 table, accepted by IROS2026

- **Prior Approaches**: 연속체 로봇은 생체 영감을 바탕으로 높은 자유도와 순응성 덕분에 난지형/협소 공간 조작에 유리하지만, 실제 연구·현장 도입은 재현성 문제로 막혀 왔다. 기존 오픈소스도 다품 부품 제작·정렬·조립 비중이 커서 비용과 오차가 커졌고, 컴플라이언스 때문에 PCC 등 모델 기반의 정확한 기구학/캘리브레이션이 어렵다는 한계가 있었다.
또한 텔레오퍼레이션은 작업공간 중심 제어가 kinematics 의존과 도달 불일치를 낳기 쉬웠고, 조인트 직접 제어는 직관성이 떨어져 데모 수집에 장벽이 됐다.

- **Core Contribution**: 이 논문은 재현성 있는 학습용 연속체 로봇을 목표로, 듀얼 멀티머티리얼 3D 프린팅으로 ‘단일체(monolithic) 컴플라이언트 구조’를 만들고 최소 조립(텐던 라우팅·액추에이터 조립만)으로 완성하는 공개형 플랫폼을 제시한다. 제어 측면에서는 isomorphic teleoperation을 도입해 조작 입력을 액추에이터 레벨로 1:1 대응시키며, 명시적 기구학 모델 없이도 singularity-free에 가까운 매핑을 제공한다.
이 데모 수집 인터페이스는 imitation learning에 바로 쓰일 수 있는 액추에이터 정렬 데모를 만들어, LeRobot 프레임워크와 결합해 데이터 수집·학습·평가 파이프라인을 표준화한다.

- **Technical Challenges**: 연속체 로봇에서는 고차원 actuation redundancy와 DoF 간 강한 결합 때문에, 작업공간 지시를 그대로 액추에이터 명령으로 바꾸는 과정이 모델링·보정 난이도를 크게 키운다. 저자들은 이를 피하기 위해 리더-팔로워 구조에서 리더의 엔코더 절대값을 캘리브레이션 오프셋에 매핑해 팔로워의 타깃 위치로 곧바로 전달하는 규칙 기반 동기화로 해결했으며, 텔레오퍼레이션은 PWM 프리텐션 유지와 동기 읽기/쓰기 방식으로 지연을 줄였다.
학습 데이터 확보를 위해서는 30Hz로 모터 상태(타깃·현재 오프셋)와 멀티 카메라 영상을 함께 기록하고, ACT(Action Chunking with Transformers)로 비전+프로프리오셉션 입력을 통해 동작을 예측하도록 학습 파이프라인을 구성했다.

- **Empirical Impact**: 하드웨어 관점에서 단일 조인트 힘-변형 특성은 약 3.2 N/mm의 거의 선형 탄성 거동을 보였고, PCC 기반으로 추정한 도달 워크스페이스는 약 430mm 지름의 구 형태에 가까웠다. 페이로드 실험에서는 텔레오퍼레이션 하 동적 운용 기준 신뢰 가능한 최대를 1kg으로 제시하며, 15시간 누적 운용에서도 구조 손상/제어 성능 저하가 관찰되지 않았다.
제어 성능은 모션 캡처 기준 팔로워 엔드이펙터 위치 추적 오차 평균 8.2mm(243mm 대비 3.3%)와 30Hz에서 평균 수렴 지연 158ms로 확인됐고, pick-and-place와 협소 공간 회수에서 텔레오퍼레이션 조작이 성공적으로 수행됐다.
자율성 실험에서는 isomorphic teleoperation 데모로 학습한 ACT 정책이 3개 과제에서 성공률 83% 이상을 달성했으며, 후크 이탈 같은 섬세한 실패 상황에서도 재시도/오류 회복으로 전 trials 성공을 보여 연속체 로봇 학습 가능성을 실증했다.



### An Infrastructure-less, Control-Independent Solution to Relative Localisation of a Team of Mobile Robots using Ranging Measurements (https://arxiv.org/abs/2606.20365)
- **Prior Approaches**: 기존의 협력적 로컬라이제이션은 고정 앵커(known position reference)로 관측가능성을 보장하는 경우가 많았고, 앵커가 없으면 불충분한 정보로 인해 다중 해가 생겨 유일한 궤적을 얻기 어렵다. 또 일부 방법은 angle까지 측정하거나, 관측가능성 강화를 위해 로봇의 능동적인 motion 제어/형성 제어를 강하게 요구해 실제 배치 비용과 제약이 커진다. 이런 방식은 인프라 구축이 불가하거나 사용자 숙련도에 따라 세팅이 어려운 환경에선 확장성과 유연성이 떨어진다.

- **Core Contribution**: 이 논문은 MHDCL(Multi-Hypothesis Decentralised Cooperative Localisation)로, 앵커 없이도 협력 로봇 팀의 2D 상대 포즈를 추정하는 완전 분산형 알고리즘을 제안한다. 핵심은 ‘관측가능성 조건을 인위적으로 만족시켜 유일해를 만들기’가 아니라, 가능한 해 전체를 유지하는 multi-hypothesis Bayesian 프레임워크로 문제의 ill-posedness 자체를 포용하는 패러다임 전환이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 앵커 부재로 인해 시스템이 순간적으로 관측 불가능해질 수 있고, 그때마다 해 공간이 늘어나 다중 가설을 계산·전달하기가 무거워진다는 점이다. 이를 위해 particle filter 기반으로 각 에이전트가 odometry와 sparse inter-agent distance(UWB 등)만으로 후보들을 갱신하되, resampling 후 GVMMM(Gaussian-von Mises Mixture Model)으로 입자들을 클러스터로 압축해 통신·계산 비용을 줄이고, 필요 시 부분적으로 연결된 네트워크에서도 클러스터/모션 공유로 추정 세트를 복원한다.

- **Empirical Impact**: 실험은 현실적인 상황(스파스한 거리 측정, 통신 제한, 관측 불가능 구간 포함)에서 단일 추정보다 ‘해 집합 유지’가 더 견고한 로컬라이제이션을 제공함을 보여주는 데 초점을 둔다. 분산형이며 외부 인프라나 motion 제어가 필요 없다는 점에서, 로봇 플릿·협동 탐사·동적 환경처럼 배치 조건이 까다로운 분야에 실사용 가능성을 높인 기여로 평가된다.



### Autonomous Driving with Priority-Ordered STL Specifications Under Multimodal Uncertainty (https://arxiv.org/abs/2606.20336)
- **Prior Approaches**: 기존 자율주행 궤적 계획은 safety, 교통규칙, 승객 comfort 같은 다중 목표를 STL(Signal Temporal Logic)이나 LTL(Linear Temporal Logic)로 형식화한 뒤 최적화하는 흐름이 많았습니다. 또한 STL 규칙 간 strict priority는 lexicographic hierarchy로 모델링해 상위 규칙 위반이 하위 규칙의 만족을 어떤 조합으로도 덮지 못하게 했지만, 이는 주로 결정론적(uncertainty 없는) 가정에 머물렀습니다. 불확실성을 다루는 risk-aware 방법들도 CVaR 같은 지표는 도입했으나, strict lexicographic 우선순위를 STL 사양에 통합해 “모드 전체에서의 우선순위 보존”을 보장하진 못했습니다.

- **Core Contribution**: 이 논문은 주변 객체의 multimodal 확률 예측 아래에서도 strict lexicographic ordering을 유지하는 uncertainty-aware 궤적 계획 프레임워크를 제안합니다. 구체적으로 STL 규칙들을 CVaR 기반 risk-aware robustness로 평가해, 우선순위가 더 높은 규칙의 만족/위반 우위를 확률 불확실성 속에서도 보존하도록 rank-preserving reward를 구성합니다. 이후 이 수식을 MPPI(Model Predictive Path Integral) 기반 receding-horizon 제어에 연결해, 충돌하는 목표를 우선순위 규칙에 맞춰 선택하도록 만듭니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 불확실성 때문에 동일한 궤적이 모드별로 서로 다른 STL 규칙 만족 양상을 보이며, (2) lexicographic 우선순위를 CVaR로 결합하면 목적함수가 non-smooth·non-convex가 된다는 점입니다. 논문은 STL robustness의 min/max 구조와 rank 및 step 함수를 그대로 평가하되, CVaR은 weighted scenario set(유한 샘플 시나리오)로 근사해 기대치 계산을 실시간으로 가능하게 했습니다. 그리고 MPPI가 미분 가능성과 볼록성을 요구하지 않으므로, 각 롤아웃을 정확히 스코어링해 우선순위 순서를 깨뜨리지 않도록 설계했습니다.

- **Empirical Impact**: 검증은 multimodal cut-in이 있는 고속도로 take-over 시나리오와, 이산 위치 불확실성을 갖는 보행자 횡단 시나리오 등 2가지에서 수행되었습니다. 결과적으로 제안 프레임워크는 실제에 가까운 다중 모드 불확실성 하에서도 상위 safety 규칙을 우선으로 유지하면서, 가능한 경우 comfort나 목표 달성 같은 하위 목표를 함께 타협하는 모습을 보였습니다. 특히 희귀하지만 치명적인 tail event를 CVaR로 반영하면서도 lexicographic priority가 유지되도록 구성했다는 점에서, 안전성-규칙준수-불확실성 추론을 동시에 다루려는 자율주행 계획 연구에 실용적 의미가 큽니다.



### Towards 3D karst underwater scene reconstruction from rotating sonar data (https://arxiv.org/abs/2606.20322)
Comments:
          1st Workshop on Long-term Deployments in the Wild (LoWi)

- **Prior Approaches**: 기존 카르스트 수중 매핑은 DEPTHX AUV처럼 단일 빔 소나와 관성항법으로 모델을 만들거나, scan-matching 기반 acoustic SLAM으로 점진적 2D/3D 지도를 구축하는 방식이 주를 이뤘다. 하지만 회전(프로파일링) 소나 데이터는 희소하고 잡음이 많고, IMU/속도 기반 궤적은 장시간 탐사에서 드리프트가 누적돼 표준 3D 재구성이 일관된 표면을 못 만든다. 한편 비전·다중센서를 결합한 end-to-end에 가까운 SLAM은 성능이 좋지만, 이 논문의 환경처럼 특정 센서 조합과 데이터 특성에 그대로 적용하기는 어렵다.

- **Core Contribution**: 이 논문은 회전형 sonar profiler(다이버 운용)로부터 수중 카르스트 도관(underwater karst conduit)을 복원하는 end-to-end 파이프라인을 제안한다. 핵심은 (1) continuous-time SLAM(6D SLAM, 3DTK)로 궤적 드리프트를 보정하고, (2) 2-stage deep learning 메싱으로 포인트클라우드를 연속적인 navigable 3D mesh로 변환하는 것이다. 결과물은 블렌더에서 Voxel Remesh를 거쳐 watertight(폐합) 형태로 다듬고, 가상 다이브 탐사용 인터랙티브 모델로 활용된다.

- **Technical Challenges**: 첫째, 소나 포인트는 ping당 반환이 희소·다중경로(multipath)로 섞여 있어 올바른 거리 신호를 뽑는 과정이 필요하다; 논문은 각 ping에서 intensity threshold를 넘는 첫 번째 return만 사용해 직접 거리 성분을 보존한다. 둘째, 드리프트가 큰 궤적 때문에 ICP류의 정합이 왜곡될 수 있는데, 시간 기준 loop closure와 섹션 분할 기반으로 rigid하게 먼저 맞춘 뒤 semi-rigid/continuous-time 정합으로 전역 일관성을 확보한다. 셋째, 희소한 점으로는 Poisson/BPA/Delaunay 같은 전통적 메시 방법이 구멍·과도한 스무딩·위상 오류를 만들기 쉬워, nPSR(Neural Poisson Surface Reconstruction)과 POCO(Point Convolution for Surface Reconstruction)를 단계적으로 적용해 품질을 개선한다.

- **Empirical Impact**: Fontaine de Nîmes 카르스트 네트워크에서 NavScoot2(회전식 프로파일링 소나, DVL/IMU 등)로 수집한 실제 데이터로 궤적 보정과 메쉬 복원이 모두 검증된다. 정성적으로는 희소 원점에서 만든 메시가 불연속/갭이 남는 반면, down-sampled 기반 nPSR+POCO 계열 복원에서 가장 매끈하고 상세한 표면을 얻었다고 보고한다. 이 파이프라인은 다이버가 위험에 노출되던 도관 내부를 virtual exploration로 반복 분석 가능하게 만들어, 수문지질 분석에서 선호 흐름 경로와 동역학 연구의 실용성을 높인다는 점에서 의미가 있다.



### Co-VLA: Coordination-Aware Structured Action Modeling for Dual-Arm Vision-Language-Action Systems (https://arxiv.org/abs/2606.20285)
- **Prior Approaches**: 기존 vision-language-action(VLA) 모델은 시각-언어 입력을 연속 동작으로 바로 회귀해 단일/양팔 로봇 조작에서 성능을 보여왔다. 많은 방법들이 양팔의 협응을 end-to-end 학습으로 암묵적으로 형성하지만, action head가 좌/우팔 동작을 단일 벡터로 뽑아 동기화·역할 비대칭·안전/부드러움 같은 요소가 학습 내부에 숨어 해석과 보정이 어렵다. 특히 협응이 더 촘촘해지고 제약이 강해질수록 암묵적 협응만으로는 신뢰 가능하고 안정적인 실행을 보장하기 힘들다.

- **Core Contribution**: Co-VLA는 양팔 협응을 ‘행동 자체’가 아니라 ‘행동 위의 구조’로 보고, VLA action head에 구조적 prior를 명시적으로 넣는 프레임워크를 제안한다. Structured Action Expert(SAE)는 공유 latent(과업 수준의 동기화 의도)와 팔별 residual latent(집행 조정)를 분해해 동작 생성 단계에서 협응 구조를 분리한다. 또한 Latent-Aware Controller(LAC)는 배치 시 shared–residual 표현을 해석해 동기화 강도, 비대칭, 매끈함, 안전 제약을 실시간으로 조절하며 기존 joint-level control 파이프라인과의 호환성을 유지한다.

- **Technical Challenges**: 핵심 난제는 좌/우팔이 강하게 결합된 과업에서 공유 협응 의도와 미세 집행 조정을 의미 있게 분리하도록 학습을 유도하는 것이다. 논문은 task-adaptive coordination loss로 residual이 필요할 때만 활성화되도록 sparse regularization을 적용하고, shared mean velocity consistency와 temporal synchronization loss로 공유 성분이 공통 궤적/타이밍을 대표하도록 강제한다. 또 배치 시에는 residual이 잡음(미세 떨림)인지 협응에 필요한 정밀 조정인지 판별해 LAC의 적응형 stiffness와 low-pass refinement로 실행을 안정화하되, force/impedance 같은 특수 제어 없이 joint-command 레벨에서 처리한다.

- **Empirical Impact**: 시뮬레이션 RoboTwin 2.0에서 Co-VLA는 단일(π0)·확장(π0.5) 계열의 단일 action head 기준선 대비 전반적으로 향상되며, 특히 tight-coordination 과업에서 성공률이 크게 상승해 평균적으로 27%p 수준의 개선을 보고한다. out-of-distribution(OOD) 실세계 평가에서는 13%→27%처럼 성능이 2배 이상으로 뛰며, 작업 완료 시간도 최대 25%까지 단축되었다고 한다. 또한 Co-Motion 동시 시연은 데이터 수집 효율을 10–25% 높이지만 학습 난도는 증가하는 ‘efficiency–learnability trade-off’를 보여주며, 그럼에도 Co-VLA가 baseline보다 더 견고하게 적응하는 점이 확인된다.



### Efficiently Linking Real Scenes with Synthetic Data Generation for AI-based Cognitive Robotics and Computer Vision Applications (https://arxiv.org/abs/2606.20272)
Comments:
          Accepted and best paper award at MHI-Kolloquium 2024

- **Prior Approaches**: 기존 비전 기반 인지 로보틱스 연구는 객체 검출/세그멘테이션, 6D pose 추정, grasping pose 추정 등 각 단계별로 성능을 끌어올리는 방식이 주류였다. 하지만 고정된 도메인에서만 강하고(benchmarks·학습 데이터 의존), 정밀한 라벨이 필요해 대규모·다양한 학습 데이터 확보가 병목이 된다. 또한 시뮬레이션을 쓰면 데이터는 늘릴 수 있지만 sim2real domain gap, 속도, 구현 비용 문제가 남는다.

- **Core Contribution**: 이 논문은 실세계 장면과 시뮬레이션을 연결해, “무한에 가까운 주석 데이터”를 만들되 반복적으로 sim2real 간극을 메우는 데이터 생성-학습 루프를 제안한다. 특히 3D 자산을 실세계 스캔에서 얻고, 이를 시뮬레이션 자산으로 사용해 시뮬레이션 라벨(예: 6D pose, grasp 후보)을 만든 뒤 실세계 정합과 재학습으로 정확도를 올리는 흐름을 목표로 한다. 결과적으로 단일 태스크 전용 모듈을 조합하는 접근보다, 인지·물리 기반 추론에 필요한 학습 기반을 마련하려는 방향성을 강조한다.

- **Technical Challenges**: 핵심 난제는 실세계 3D 자산을 시뮬레이터에서 바로 쓰기 좋게 분해·구축하는 과정(예: Nerf에서 mesh/texture로의 전환)과, 인간 설계 없이도 테이블/선반 등 “그럴듯한 배치 논리”를 스케일 가능하게 자동화하는 것이다. 또 시뮬레이션에서 생성한 주석이 실세계에 그대로 이전되지 않는 sim2real gap을 줄이기 위해, 시뮬레이션으로 학습한 6D pose 추정기를 실세계 초기화에 쓰고 등록(registration)과 추가 주석을 업데이트하는 단계적 루프를 설계한다. 저자들은 Omniverse/Blender, Isaac Sim 같은 도구로 포토리얼 렌더링과 물리 기반 실험(grasps/force 조건 등)을 뽑고, 스타일/텍스처 보강 등으로 시각적 차이를 완화하려 한다.

- **Empirical Impact**: 현재는 진행 중(work-in-progress)으로, 논문은 실험 결과 수치보다 “데이터 파이프라인”의 가능성과 필요성을 정리하는 데 초점을 둔다. 다만 이 방식이 성공하면 grasping pose 추정 및 end-to-end planning 같은 다운스트림에서 더 큰 다양성의 라벨 데이터를 확보해 도메인 밖 일반화와 스케일 확장에 실질적인 동력이 될 것으로 기대된다. 인지 로보틱스에서 가장 큰 제약이던 주석 비용을 줄이면서, 실세계-시뮬레이션을 왕복하며 모델 성능을 끌어올리는 기반이 될 수 있다는 점에서 의미가 있다.



### Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think (https://arxiv.org/abs/2606.20246)
- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 효율화는 크게 두 갈래로, 토큰 프루닝·캐싱 같은 training-free 가속과, MoLe-VLA·DeeR-VLA처럼 학습으로 라우팅/조기종료를 학습하는 training-adaptive 방식이 주를 이뤘다. 다만 training-free는 실시간 추론은 줄여도 비싼 downstream fine-tuning 비용을 거의 건드리지 못했고, training-adaptive는 보조 라우터·추가 학습 목표로 구조가 복잡해지는 문제가 있었다. 또한 연속 제어형 SOTA(π0, GR00T-N1.5)와 실세계 검증 범위가 제한적인 경우가 많았다.

- **Core Contribution**: 이 논문은 VLA의 연속 제어 foundation policy가 학습된 다양한 궤적에도 불구하고, 층(layer) 단위 표현에 심각한 중복( representational redundancy )이 있음을 보인다. 이를 바탕으로 CLP(CKA-guided Layer Pruning)를 제안하며, 학습 없이 Centered Kernel Alignment(CKA)로 중복 층을 찾아 fine-tuning 전에 영구적으로 transformer depth를 최대 50%까지 줄인다. 그 결과 구조는 작아지되, 다운스트림 적응은 기존 학습 목표 그대로 수행해 성능 손실 없이 효율을 확보한다.

- **Technical Challenges**: 핵심 난제는 “중복처럼 보여도 제거하면 정책이 망가질 수 있다”는 안정성 문제였다. CLP는 calibration set에 대해 한 번의 forward pass로 각 연속 층의 표현 유사도를 CKA로 측정하고, 높은 유사도 구간을 연속 블록으로 묶어(pruning set 구성) 실제로 제거할 후보를 정한다. 이후 선택된 층을 제거해 스태틱하게 압축된 모델을 만들고, 추가 모듈 없이 바로 downstream fine-tuning을 수행해 표현 공간이 재구성(manifold restoration)되도록 한다.

- **Empirical Impact**: LIBERO·RoboCasa·SimplerEnv 3개 시뮬레이션 벤치마크와 실세계 조작 10개 태스크(4개 로봇 embodiment)에서 CLP는 기준 모델 대비 30% 내외 학습/연산 이점을 보이며 성능은 동등하거나 더 높게 유지된다. 구체적으로 학습 시간은 40–50% 줄고 실시간 추론도 최대 30% 빨라졌으며, 심지어 데이터가 적은 환경에서도 10% 데이터에서 성공률이 77.7%→84.6%처럼 개선되는 샘플 효율 이점이 확인됐다. 저자들은 “필요 이상으로 깊은 VLA가 계산 효율을 갉아먹는다”는 관점을 지지하며, 확장 가능한 로봇 학습을 위한 compute-efficient 패러다임을 제시했다고 결론낸다.



### Mobile Target Search with Imperfect Perception: A Partially Observable Stochastic Game Theoretical Approach (https://arxiv.org/abs/2606.20232)
- **Prior Approaches**: 기존 모바일 표적 탐색 연구는 주로 pursuit-evasion 게임이나 POSG/POMDP 틀을 사용하되, 대체로 완전(또는 거의 완전)한 감지 가정을 두는 경우가 많았습니다. 이 경우엔 양성 임계값을 넘으면 탐지를 “사후적으로” 보장하는 방식이 흔했지만, 오탐(false alarm)과 미탐(missed detection)이 있으면 잘못된 믿음이 확산되어 목표를 끝내 못 찾는 문제가 새로 생깁니다. 또한 정적 표적에서 누적 판정으로 잡음을 상쇄하는 방식은 이동 표적과 적대적 회피가 결합된 설정에 그대로 적용하기 어렵습니다.

- **Core Contribution**: 이 논문은 오탐과 미탐이 포함된 불완전 인식 하의 모바일 표적 탐색을 POSG로 정식화하고, 목표를 끝내 찾을 수 있는지를 판정하는 개념으로 α-detectability(α-detectability)를 제안합니다. 여기서 핵심은 “매 단계의 정보 획득이 보장되지 않는” 낮은 SNR 환경에서도, 특정 탐색 전략이 확률 1로 유한 시간 내 탐지 상태를 ‘도달(hitting)’시키는지 여부를 이론적으로 다루는 데 있습니다. 즉 전략의 Nash적 최적성보다 ‘탐지 보증’ 자체를 목표로 재정의합니다.

- **Technical Challenges**: 가장 큰 기술적 난점은 오탐이 posterior belief(사후 믿음)를 오염시켜, 기존의 효용 최적화나 평형 계산만으로는 결국 탐지 도달성을 보장하기 어렵다는 점입니다. 논문은 sequential belief update가 만드는 posterior belief 과정의 확률적 recurrence를 분석해 Borel-Cantelli lemma 기반의 충분조건을 제시하고, homogeneous 설정에서는 최악의 경우에 대한 정량적 임계값(탐지 기준선) 경계도 도출합니다. 더 나아가 계산 복잡도를 낮추기 위해 server-assisted distributed 알고리즘을 설계했는데, 탐색자 측은 aggregative potential game 구조로 분해하고 표적 예측은 scalar average belief에 대한 KL-divergence 기반 축소로 해결합니다.

- **Empirical Impact**: 수치 시뮬레이션에서 제안한 탐색 알고리즘이 detectability 분석과 부합하게 탐지 성능을 확보함을 보였고, 특히 false alarm이 존재할 때도 탐지 보증 관점을 유지할 수 있음을 보여줍니다. 이는 모바일 표적 탐색에서 “언제 탐지되는가”를 확률 1 도달성으로 명확히 다룬다는 점에서, 기존의 임계 기반·탐욕적·평균적 성능 평가를 보완하는 의미가 있습니다. 결과적으로 불완전 감지와 적대적 회피가 공존하는 로보틱스 탐색·추적 설계에서 이론적으로 근거 있는 탐지 임계 설계 방향을 제시합니다.



### FlowMaps: Modeling Long-Term Multimodal Object Dynamics with Flow Matching (https://arxiv.org/abs/2606.20209)
- **Prior Approaches**: 기존 Object Navigation(ObjNav)과 동적 객체 위치추정은 정적 장면 가정이 많거나, 동작 궤적/장소를 확률적으로 다루더라도 이산적 receptacle(수용기) 관계나 사전 설계된 prior, 비싼 VLM/LLM 추론에 의존하는 경우가 많았다. 또한 HOMER처럼 반복되는 가정 활동 기반으로 변위(displacement)를 예측하더라도 특정 환경 내부에서의 일반화에 초점이 있어, 레이아웃·배치가 다른 집으로의 전이를 충분히 검증하기 어렵다는 한계가 있었다.

- **Core Contribution**: FlowMaps는 사람의 일상 상호작용으로 인해 발생하는 객체의 미래 위치를 연속 3D 공간에서의 multimodal spatio-temporal 분포로 직접 추정하는 latent flow matching 모델이다. 특히 객체-배경 장면 컨텍스트와 쿼리 라벨, 예측 시간에 조건을 건 채 미래 bounding box 분포를 생성해, 특정 활동을 명시적으로 식별하지 않아도 데이터에서 규칙성을 학습·활용하도록 설계됐다. 더불어 이전에 보지 못한(하지만 유사한 일과/루틴 패턴을 공유하는) 새로운 환경으로의 일반화까지 목표로 한다.

- **Technical Challenges**: 핵심 난제는 (1) 미래 객체 위치가 여러 모드로 열려 있고(예: 안방 탁자↔세면대), (2) 장면 맥락과 시간 변화가 결합된 posterior를 직접 계산하기 어렵다는 점이다. FlowMaps는 이를 연속 공간의 조건부 flow로 근사하기 위해 VAE로 객체 기하/의미를 잠재공간으로 압축한 뒤, conditional flow matching을 latent 공간에서 학습하고 샘플링으로 다중 미래를 얻는 방식으로 해결했다. 또한 ProcTHOR로 대규모 동적 궤적(사람 루틴 기반)을 합성해 학습 데이터 병목을 완화하고, Transformer 기반 인코더+CDiT 블록으로 장면 컨텍스트를 쿼리의 시간적 흐름에 효과적으로 주입한다.

- **Empirical Impact**: ProcTHOR 시뮬레이션에서 600회 이상 ObjNav episode을 통해 FlowMaps가 최신 baselines 대비 더 좋은 minFDE와 모드 커버리지를 보이며, 포인트 정확도만이 아니라 분포의 형태(coverage/density, TV/JS)에서도 개선을 보였다. 특히 best-of-K 샘플링으로 평가했을 때도 multimodality를 제대로 보존해, 동적 변화가 큰 가정환경에서 탐색·탐지 성능이 향상됨을 확인했다. 또한 실제 로봇 플랫폼 배치까지 수행해 실세계 적용 가능성을 함께 검증함으로써, posterior 추론 관점에서 FM을 로보틱스에 활용할 수 있음을 실증했다.



### Stable Transformer-Actor-Critic Model Predictive Control: A Contraction Analysis Approach (https://arxiv.org/abs/2606.20197)
- **Prior Approaches**: Actor-Critic Model Predictive Control(MPC)는 비볼록 제어 문제에 강점을 보이지만, 시퀀스 기반 학습 모델을 MPC 파이프라인에 넣을 때 closed-loop stability를 보장하기는 어렵습니다. 기존 접근은 대체로 성능 중심의 학습이나 경험적 안전성에 의존해, 예측 오차와 네트워크-제어기 상호작용이 안정성에 미치는 영향을 형식적으로 다루기 힘들었습니다.

- **Core Contribution**: 이 논문은 Transformer-Actor-Critic MPC 아키텍처를 제안하면서, 제어 정책에 대해 formal한 robustness guarantees를 제공합니다. 핵심은 Transformer가 전역 incremental Input-to-State Stability(δISS)를 만족할 수 있음을 증명하고, 이를 바탕으로 물리 플랜트-예측 신경망의 결합 동역학을 안정성 관점에서 해석하는 프레임을 구축한 점입니다.

- **Technical Challenges**: Transformer가 MPC 내에서 안정적으로 동작하도록 하려면, 네트워크가 만들어내는 시퀀스 예측이 plant에 미치는 영향까지 포함한 이산/연속 동역학의 안정성 조건을 “형식적으로” 연결해야 합니다. 논문은 Riemannian contraction theory로 결합 시스템을 분석한 뒤, 이론적 상계를 training regularizer로 통합해 certifiably robust 정책을 얻도록 설계합니다.

- **Empirical Impact**: 비선형 3D 드론 모델에서 target-reaching과 obstacle-avoidance을 수행하며, 제안 프레임이 실제 closed-loop에서 목표를 달성하면서도 안정성 보증을 뒷받침함을 보여줍니다. 학습 기반 MPC에서 stability를 보장하는 설계-검증 루프를 제시했다는 점에서, 로보틱스 제어와 안전성 보장 연구에 실질적인 기준점을 제공할 것으로 기대됩니다.



### Belt-Finger: An Affordable Soft Belt-Driven Gripper for Dexterous In-Hand Manipulation (https://arxiv.org/abs/2606.20193)
- **Prior Approaches**: 로봇 조작에서 in-hand 조작(손안 조작) 능력은 중요하지만, 기존 접근은 크게 (1) Shadow hand처럼 많은 관절을 갖춘 로봇 핸(고가·복잡한 제어)과 (2) soft gripper/연속 변형 지능형 구조(정확 제어·강건성 난점)로 나뉘는 경우가 많았습니다. 반면 parallel gripper는 저렴하고 튼튼하지만 손안에서의 이동/회전 DoF가 부족해 팔 전체를 크게 움직여야 하며, 좁은 작업공간에서는 조작성 한계가 커집니다.

- **Core Contribution**: 본 논문은 parallel gripper를 그대로 대체하는 저비용 업그레이드인 Belt-Finger를 제안합니다. 손가락에 double-soft-belt 모듈을 장착해 표준 opening/closing은 유지하면서 in-hand translation, pitch, roll 총 3개의 구동 DoF를 추가해, 기존 parallel gripper의 dexterous manipulation 범위를 크게 확장합니다.

- **Technical Challenges**: 핵심 기술적 난제는 soft belt가 만들어내는 비정상/비일정 sliding contact로 인해 정밀한 동역학을 완벽히 모델링하기 어렵다는 점입니다. 저자들은 sticky(no slip)·중심선 근사 등 국소 단기 근사 가정으로 단순화한 belt-finger 모델 위에서 iCEM 기반 zero-order planner를 결합한 stochastic MPC를 구성하고, gripper opening은 feedforward 예측과 적응 보정으로 별도 처리해 안정적 그립을 유지합니다. 또한 학습/대안을 위해 인간 시연을 빠르게 수집할 수 있는 최소 하드웨어 teleoperation 인터페이스(델타 포즈 추적, 직관적 조이스틱 매핑)도 함께 제시합니다.

- **Empirical Impact**: 실험에서는 teleoperation로 다양한 물체에 대해 translation(대략 45mm), pitch(대략 85°), 연속 roll 성능을 확인했으며, conventional parallel gripper 대비 task feasibility와 조작성 측면에서 일관된 개선이 관찰됩니다. 더 나아가 iCEM-MPC, teleoperation, 그리고 fine-tuning된 Vision-Language-Action(VLA) 정책(여러 SOTA 모델)에 대해 복잡한 일상 과업에서 성공률 향상과 closed-loop 보정 행동의 “자연스러운” 출현을 보고하며, 사용성·효율·만족도까지 비교 우위를 주장합니다.



### Robust Assembly State Reasoning from Action Recognition for Human-Robot Collaboration (https://arxiv.org/abs/2606.20150)
Comments:
          Preprint accepted to the 35th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN 2026). 8 pages, 9 figures, 3 tables

- **Prior Approaches**: 기존 Human Action Recognition(HAR)은 동작 종류를 시점별로 맞히는 데 집중하는 경우가 많아, 이를 조립의 현재 stage로 “견고하게” 매핑하는 데는 한계가 지적돼 왔다. 또 HRC 현장에서는 HAR 출력 뒤에 사람이 만든 rule-based logic이나 간단한 시간 추론을 덧붙이는 방식이 흔했지만, 비교와 한계 분석이 부족하고 과도하게 특정 단서에 의존하는 문제가 있었다.

- **Core Contribution**: 이 논문은 HAR 입력으로부터 조립 task의 현재 assembly stage를 추적하는 문제를 정식화하고, logic 기반·HMM·LSTM 기반 등 5가지 stage tracking 접근을 두 가지 서로 다른 데이터셋에서 체계적으로 비교한다. 특히 동일 동작 타입이 반복되거나 사용자 동작 순서가 변하는 “현실적 변동” 조건에서 어떤 방법이 잘/못 되는지 작업별 최적해가 달라짐을 보여준다.

- **Technical Challenges**: 핵심 기술 난제는 (1) HAR 자체가 오인식과 잡음에 취약하고 (2) 조립 과정이 이상적인 선형 절차가 아니라 반복·생략·순서 변경이 발생하며 (3) 같은 원자 동작이 여러 stage에 매핑된다는 점이다. 저자들은 입력 잡음 수준을 달리한 시뮬레이션과 실제 HAR(ST-GCN) 출력 기반 실험을 함께 수행하고, 반복 동작이 많은 경우 duration(동작 지속시간) 모델링이 성능에 중요함을 logic+확률(time CDF) 및 duration 인코딩 LSTM 설계로 확인한다.

- **Empirical Impact**: 실험 결과, HA4M처럼 변동이 제한된 조건에서는 NN/LSTM(또는 HMM)이 강점을 보였지만, IKEA처럼 반복과 순서 변동이 큰 과제에서는 논리 기반 접근이 더 견고하게 동작했다. 또한 실제 HAR 입력으로 확장했을 때도 TaskLSTM이 HA4M에서, Baseline+Prob가 IKEA에서 상대적으로 우수했으며, confusion matrix 분석을 통해 “상태에 갇힘” 또는 “앞질러 감” 같은 실패 모드가 드러났다. 전반적으로 반복 동작과 부정확한 인식이 공존하는 제조 HRC에서, 단일 모델/단일 로직이 아닌 문제 특화 조합이 필요하다는 실증적 근거를 제공한다.



### Frequency-Aware Flow Matching for Continuous and Consistent Robotic Action Generation (https://arxiv.org/abs/2606.20135)
- **Prior Approaches**: Flow matching과 diffusion policy 계열은 복잡한 멀티모달 행동 분포를 잘 표현하지만, 대부분 고정 주파수의 action chunk를 이산 시점으로 예측한다. 이 방식은 서로 다른 control frequency로 수집된 데모를 학습할 때 주파수 불일치로 식별 불가능 문제가 생기고, 추론 시 인접 타임스텝이 일관되지 않아 jittery한(덜 매끄러운) 행동이 제어 안정성을 해칠 수 있다.

- **Core Contribution**: 이 논문은 Frequency-Aware Flow Matching(FAFM)으로 연속적이고 시간적으로 일관된 행동을 직접 생성하는 방법을 제안한다. 데모를 DCT(이산 코사인 변환) 계수로 바꾼 뒤 flow matching을 계수 공간에서 수행하고, 코사인 basis expansion으로 임의 시간 해상도에서 연속 행동을 복원한다. 또한 1차 시간 미분에 대한 regularization을 Sobolev-type 제약으로 넣어 고주파 오차와 급격한 변화를 억제한다.

- **Technical Challenges**: 핵심 기술 과제는 이질적인 sampling frequency로 기록된 데모를 하나의 step-index supervision에 억지로 맞추는 문제를 피하면서도, 연속 시간에서 일관된 trajectory를 학습하는 것이다. FAFM은 DCT로 step index를 물리 시간 기반 표현으로 전환해 주파수에 따른 목표 불일치를 줄이고, DCT 파라미터화 덕분에 미분(velocity) 감독을 유한차분 잡음 없이 정확히 정의해 derivative flow matching을 안정적으로 적용한다.

- **Empirical Impact**: 합성 toy 벤치마크, obstacle avoidance, LapGym, LIBERO에서 FAFM은 성공률, 멀티모달 expressivity, motion smoothness, 수렴 속도, 기계적 bias 및 mixed-frequency 입력에 대한 견고성을 함께 개선한다. 특히 LapGym처럼 연질(soft-body) 조작에서 jitter를 줄여 더 안정적인 제어를 보였고, Franka 로봇 실배치에서도 성능 우위가 일관되게 나타났다. 저자들은 DCT와 1차 미분(velocity) 감독의 기여를 ablation으로 확인했으며, 연속 주파수 표현 없이 finite difference 기반 derivative supervision은 효과가 제한적이라고 보고한다.



### Dual-Agent Framework for Cross-Model Verified Translation of Natural-Language Protocols into Robotic Laboratory Platform (https://arxiv.org/abs/2606.20120)
- **Prior Approaches**: 기존 자동화 시스템은 미리 정의된 제어 명령에 의존해, 생물학 실험 프로토콜의 자연어와 로봇 실행 간에 semantic gap이 생기는 문제가 컸습니다. 특히 microplate 기반 실험은 well mapping, 시료-시약 조합, 반복( replicate ) 배치, 병렬 분주를 동시에 맞춰야 해서 제약 반영이 어렵습니다.

- **Core Contribution**: 이 논문은 자연어 microplate 프로토콜을 로봇이 실행 가능한 제어 명령으로 바꾸는 agent-based protocol translation 프레임워크를 제안합니다. Parser Agent가 프로토콜을 구조화하고, rule-based mapping engine이 로봇 플랫폼의 운영 제약을 결정적으로 반영해 device-level 제어 커맨드를 생성하며, LLM Validation Agent가 완전성·정확도·순서를 검증합니다.

- **Technical Challenges**: 핵심 과제는 (1) 자연어에서 누락된 파라미터를 찾아 정확한 값으로 채우고, (2) 로봇 제약을 만족하는 실행 순서를 보장하며, (3) 오류가 나면 재현 가능한 방식으로 self-correction 루프를 돌리는 것입니다. 연구진은 structured feedback를 이용한 검증 및 수정 루프와 cross-model verification을 결합해, 모델 규모와 Validator 유형 변화에도 번역 품질을 점검하도록 설계했습니다.

- **Empirical Impact**: 무작위로 선택한 ELISA 프로토콜에서 7개의 Parser와 3개의 Validator를 사용한 sweep으로 번역 정확도와 pass rate를 평가했고, 모델 규모·Validator 타입이 성능에 미치는 영향을 분석했습니다. 또한 rule-based mapping을 LLM end-to-end direct mapping과 비교해 accuracy–latency trade-off를 확인했으며, Bradford assay 단백질 정량을 로봇 실험으로 성공적으로 수행해 자연어→실세계 end-to-end 자율 실행 가능성을 입증했습니다.



### Pose6DAug: Physically Plausible Multi-view Object Swapping for Robot Data Augmentation (https://arxiv.org/abs/2606.20118)
- **Prior Approaches**: Vision-language-action(VLA) 정책은 범용 조작에 강점을 보이지만, 학습 분포에서 벗어난 novel out-of-distribution 물체에서는 외형이나 형상이 달라지며 실패하는 경우가 잦습니다. 보통은 실패 케이스마다 다중 시점 텔레오퍼레이션 데이터를 추가 수집해 보정하지만, 비용과 시간이 기하급수적으로 커진다는 한계가 있습니다. 이 때문에 스케일 가능한 데이터 확장 방법이 필요하다는 요구가 커졌습니다.

- **Core Contribution**: 이 논문은 Pose6DAug라는 failure-driven 데이터 증강 프레임워크를 제안해, 새 데이터 수집 없이도 실패 모드를 겨냥한 시연을 자동으로 만들어냅니다. 핵심 아이디어는 정책이 성공한 에피소드 안에 이미 물리적으로 유효한 action trajectory와 보정된 다중 시점 관측이 함께 들어 있다는 점입니다. 이를 유지한 채 조작 대상 물체만 교체해, failure mode에 맞춘 “물리적으로 타당한” 추가 시연을 생성합니다.

- **Technical Challenges**: 문제는 단순 2D 비디오 편집으로는 다중 시점 간 정합성과 물리적 그럴듯함이 동시에 깨진다는 데 있습니다(특히 심한 가림과 자아 시점에서 두드러짐). Pose6DAug는 3D에서 작동하며, temporally coherent한 6D pose trajectory에 의해 구동되는 explicit mesh로 목표 물체를 고정해 카메라 전반에 걸친 기하학적 일관 렌더링을 보장합니다. 결과적으로 궤적은 보존하면서도 다중 시점 정합성과 물리 plausibility를 동시에 확보합니다.

- **Empirical Impact**: 이 방법으로 증강된 데이터로 VLA를 fine-tuning하면 novel 물체에서 state-of-the-art 기준 대비 성공률이 16.5% 상대적으로 향상됩니다. 또한 in-distribution 성능은 유지해, 단순 증강이 성능을 훼손하는 부작용을 줄였다는 점이 확인됩니다. 논문은 multi-view와 물리 일관 증강이 VLA 일반화의 실용적 확장 경로가 될 수 있음을 실험적으로 보여줍니다.



### VFILC: Accurate Frequency Extrapolations in Imitation Learning via Sampling Frequency ILC (https://arxiv.org/abs/2606.20056)
Comments:
          8 pages, 17 figures. Accepted at IROS 2026

- **Prior Approaches**: 기존 imitation learning은 variable-speed에서 학습 속도 범위 안(보간)으로는 잘 맞지만, 범위 밖(외삽) 속도에서는 예측이 불안정해지는 문제가 컸습니다. 이를 줄이기 위해 VFIL은 NN의 샘플링 주파수를 motion 주파수에 연동해 외삽을 가능케 했지만, open-loop 구조라 접촉 비선형성이 큰 작업에서 주파수 오차가 누적/지속되는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 variable-frequency imitation learning에 iterative learning control(ILC)을 결합한 VFILC를 제안해, 외삽 시 발생하는 시간(주파수) 오차를 반복적으로 보정합니다. VFIL의 “주파수 샘플링” 장점은 유지하되, ILC의 feedforward+feedback 구조로 이전 반복의 주파수 오차를 다음 입력에 반영해 시간 영역 오차를 줄이는 방식입니다.

- **Technical Challenges**: 핵심 난제는 접촉이 포함된 비선형 환경에서 NN이 생성하는 응답의 실제 주파수가 명령 주파수와 어긋나며, 특히 외삽 고주파에서 오차가 커진다는 점입니다. 저자들은 VFIL에서 주파수 입력을 고정(open-loop)하지 않고, ILC가 반복마다 주파수 오차를 보정하되 NN의 샘플링 주파수 자체는 생성 중에 흔들리지 않도록 반복 끝에서만 업데이트하는 절차를 설계했습니다.

- **Empirical Impact**: wipe, mixing, shaking 세 과제 실험에서 VFILC는 외삽 구간에서도 보간과 비슷한 주파수 정확도를 달성했고, 피드백이 주파수 오차를 크게 줄였습니다. 특히 wiping에서 평균 81% 감소, shaking에서 50% 감소를 보였으며, contact-rich mixing에서는 VFIL 대비 27% 정확도 향상을 보였습니다. 결과적으로 “외삽 속도에서의 시간 오차”를 피드백으로 다루는 프레임워크로서, 반복 조작과 접촉 작업에 대한 imitation learning의 적용 가능성을 한 단계 끌어올렸다는 의미가 있습니다.



### MirrorDuo: Reflection-Consistent Visuomotor Learning from Mirrored Demonstration Pairs (https://arxiv.org/abs/2606.20048)
Comments:
          Published in CoRL 2025

- **Prior Approaches**: 시각 기반 behaviour cloning(BC)은 RGB 시연(eye-in-hand·third-person)로 로봇을 학습하지만, workspace가 좌우로 달라지거나 비대칭 장면일 때 다양한 데이터를 모아야 일반화가 어렵다. 3D(point cloud 등)를 쓰면 SE(3) 같은 강한 기하 대칭을 활용해 시연을 변환 재사용할 수 있으나, 2D 이미지는 같은 변환을 적용해도 효과가 일관되지 않아 데이터 효율이 제한된다.
또한 기존 이미지 반사(mirroring) 활용은 top-down 등 단순 설정이나 간단한 action 공간에 머물러, 실제로 “미러된(반대편) workspace”로의 일반화를 충분히 다루지 못했다.

- **Core Contribution**: MirrorDuo는 reflection symmetry(반사 대칭)를 시각-운동 학습 전반에 일반적으로 주입하는 방식으로, RGB 관측, proprioception, full 6-DoF end-effector action 튜플을 함께 반사해 “collect one, get one for free”에 가깝게 데이터 커버리지를 확장한다.
이는 (1) 기존 학습 파이프라인에 얹는 data augmentation(MirrorAug)과 (2) 네트워크에 반사-equivariance를 구조적으로 넣는 구조 prior(MirrorDiffusion)라는 두 가지 활용 경로를 제공한다.

- **Technical Challenges**: 핵심 기술 난점은 ‘이미지의 horizontal flip’과 ‘행동/자세의 기하 반사’를 의미론적으로 일치시키되, camera extrinsics(외부 파라미터) 없이도 dataset-agnostic하게 구현해야 한다는 점이다. MirrorDuo는 포즈를 local 좌표로 재표현해 extrinsics 의존을 제거하고, SO(3)에서 반사로 인해 회전 초기값이 불연속이 되는 문제를 dataset 평균 축 정렬을 위한 상수 정렬 회전 Q로 완화한다.
또한 diffusion 기반에서는 노이즈를 단계별로 독립 샘플링하면 전체 반사 대칭이 깨져 zero-shot 전이가 실패할 수 있는데, 이를 보정하기 위해 mirror 데이터와 함께 시각 일반화(Random Overlay 등)를 병행한다.

- **Empirical Impact**: 실험은 MimicGen 기반 과제에서 one-side 시연만 있을 때도 반사 보강이 데이터 예산을 크게 절약함을 보인다. 특히 시연이 workspace 양쪽에 균등 분포된 경우에는 동일 데이터 예산에서 성능이 유의미하게 향상되며, 반대편이 거의 없는 상황에서도 MirrorAug 또는 경량 시각 일반화와 결합하면 목표에 대한 빠른 skill transfer(최소 0~5 demo 수준)를 보여준다.
반대로 MirrorDiffusion 단독은 diffusion의 단계별 노이즈로 인한 symmetry violation 때문에 direct transfer가 실패하는 경우가 있으나, 구조 prior가 추가 반대편 시연(예: 5개)으로 실질적으로 “활성화”되어 성능을 복원하는 양상이 관찰된다.



### A Neuromorphic Reinforcement Learning Framework for Efficient Pathfinding in Robotic Mobile Fulfillment Systems (https://arxiv.org/abs/2606.20031)
- **Prior Approaches**: RMFS(로봇 모바일 풀필먼트) 경로탐색은 좁은 통로, 동적 장애물, 강한 실시간 제약 때문에 기존 탐색/룰 기반(A*, Dijkstra 등)이 재계획 비용과 지연 문제를 겪기 쉽습니다. RL/DRL은 적응성이 좋지만, 모바일 AGV의 전력·연산 제약에서 ANN 추론이 부담이 되고 실제 에너지 절감이 충분히 검증되지 못했습니다. 또 ANN-to-SNN 변환은 주로 분류용(희소 one-hot) 설정에 맞춰져 있어, RL의 연속적인 Q-value 출력 분포를 그대로 옮기기 어렵다는 한계가 지적돼 왔습니다.

- **Core Contribution**: 이 논문은 end-to-end로 SDQN-RMFS를 제안하며, DQN(ANN 기반) 학습부터 ANN-to-SNN 변환, 그리고 neuromorphic chip SPECK2E로의 오프라인 물리 배포까지 한 파이프라인에서 “정확도 보존+초저전력”을 목표로 합니다. 핵심은 RL 출력 분포 불일치를 줄이기 위해 hard-label knowledge distillation(argmax 기반 pseudo-label)을 사용해 SNN이 RL 정책의 의사결정을 더 견고하게 학습하도록 만든 점입니다. 더불어 하드웨어 지연을 줄이기 위한 parameter scaling과 바이어스 제거로 극저 time-step에서도 동작성을 확보합니다.

- **Technical Challenges**: 문제는 세 갈래로 정리됩니다: (1) 좁은 공간에서 무작위 탐색이 충돌로 학습이 정체되는 문제, (2) SNN을 매우 짧은 time-step(예: 4)에서 돌릴 때 초기 스파이크 지연으로 정보가 제대로 전달되지 않는 문제, (3) ANN-to-SNN 변환 시 RL의 연속 Q-value와 SNN의 이산 스파이크/하드 라벨 간 분포 mismatch로 인해 선택이 흔들리는 문제입니다. 해결책으로 collision-allowing strategy로 일정 한도 내 충돌을 허용해 탐색 정체를 완화하고, 첫 레이어 가중치를 k배로 키워 빠른 스파이크 트리거를 만들며, distillation로 출력 마진을 키워 변환 잡음에도 action 선택이 유지되게 했습니다.

- **Empirical Impact**: 시뮬레이션과 변환/배포 평가에서 SNN의 conversion rate(CRCR)는 fine-tuning과 scaling 조합 시 4 time-step에서도 1.00 수준을 달성해 정책 충실도를 입증합니다. 물리 하드웨어 실험에서는 Speck 칩에서 에너지 사용이 GPU 대비 최대 11,281× 감소했고, 추론 지연은 고성능 GPU 대비 약 2배 수준으로 줄어드는 결과(거의 절반 수준)가 보고됐습니다. 즉, neuromorphic inference가 RMFS 대규모 운용에서 실제 전력 병목을 완화할 수 있는 실행 가능한 경로임을 실험적으로 뒷받침했다는 점에서 의미가 큽니다.



### Tri-Info: Generalizable, Interpretable Failure Prediction for VLA Models via Information Theory (https://arxiv.org/abs/2606.19998)
- **Prior Approaches**: 기존 VLA failure detector는 크게 두 계열로 나뉩니다. embedding 기반 방법은 내부 표현에 classifier를 얹어 in-domain 정확도는 높지만, 표현 공간이 모델마다 달라 out-of-architecture로는 재학습 없이는 잘 옮겨가지 못합니다. score 기반 방법(STAC 등)은 시간적 일관성 같은 단일 스칼라 신호로 실패를 경고하지만, 어떤 실패 모드인지 진단 정보는 제한적입니다.

- **Core Contribution**: 이 논문은 VLA 제어를 perception–action 폐루프의 정보 파이프라인으로 보고, 성공/실패 롤아웃이 정보이론적 시그니처에서 체계적으로 갈린다는 관찰을 바탕으로 Triple Information-theoretic(Tri-Info) 신호를 제안합니다. Tri-Info는 (1) 행동 다양성, (2) 시간적 일관성, (3) 상태 전이와의 결합(액션-상태 커플링)으로 실패를 분해해, 경고를 넘어 해석 가능한 진단 대시보드를 제공합니다. 특히 각 신호는 임베딩 좌표의 기하에 덜 의존해 아키텍처와 환경 전이 일반화에 유리하다고 주장합니다.

- **Technical Challenges**: 핵심 기술 난점은 정보이론량(엔트로피/상호정보)을 VLA의 연속·고차원 임베딩에서 안정적으로 추정하고, 이를 실시간 탐지기로 결합하는 것입니다. 논문은 sliding window로 분포를 추정하고, MI/엔트로피는 k-NN 기반 추정기(예: Kraskov 류)와 Kozachenko–Leonenko 엔트로피 추정으로 계산한 뒤 z-normalization으로 도메인 간 비교 가능성을 확보합니다. 또한 각 Tri-Info 신호를 GRU로 시간 진화를 모델링하고, success 구간에서 점수 분포가 변하는 문제는 Functional Conformal Prediction으로 time-varying threshold를 만들어 해결합니다.

- **Empirical Impact**: 여섯 개 VLA 모델과 세 개 벤치마크 환경에서 Tri-Info는 in-domain에서 강력한 베이스라인과 동등하거나 그 이상 성능을 보이며, 특히 시간 예측 타이밍까지 앞서 실패를 더 일찍 감지합니다. 더 중요한 점은 cross-model, cross-environment, sim-to-real 전이에서 임베딩·score 기반 탐지기들이 붕괴하는 상황에서도 Tri-Info가 재학습 없이 성능을 유지한다는 결과입니다. 실세계 작업에서는 이전 detector들이 chance로 무너질 때 Tri-Info가 83% 정확도까지 도달해, 안전한 배치형 embodied AI에서 “해석 가능한 실패 경고”의 실용성을 강하게 시사합니다.



### Evaluation of Augmented Reality-based Intuitive Interface for Robot-Assisted Transesophageal Echocardiography: A User Study (https://arxiv.org/abs/2606.19971)
- **Prior Approaches**: 기존 TEE는 구조적 심장질환(SHD) 진단과 시술 안내에 필수지만, 프로브를 직접 조작해야 해 숙련도 의존이 크고 작업자 피로가 높다. 또한 형광투시와 병행되면 방사선 노출 부담이 커서, 로봇 보조로 핸들링을 개선하려는 시도가 있었다. 다만 로봇 TEE를 실제로 쓰게 만드는 직관적 사용자 인터페이스(UI) 설계는 아직도 미해결 과제로 남아 있었다.

- **Core Contribution**: 이 논문은 로봇 보조 TEE에서 공간 인지와 조작 직관성을 높이기 위해, model-enhanced Augmented Reality(AR) 기반의 인터페이스를 제안하고 평가한다. 시각화 방식과 상호작용 수준을 달리한 3가지 UI(2D joint-level, 3D joint-level, 3D tip-level)를 비교하며, 어떤 조합이 성능과 작업부담을 동시에 개선하는지 정량화한다. 이를 통해 AR 시각화와 tip-level 직접 제어를 다음 세대 로봇 TEE에 통합하는 방향성을 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 AR로 제공되는 가시화가 실제 손-기기(로봇 프로브) 제어와 일관되게 이어져야 한다는 점이며, 특히 공간 오차(위치·자세)와 작업부담을 함께 줄여야 한다. 연구진은 electromagnetic tracking과 가상 시뮬레이터를 결합한 로봇 TEE 플랫폼을 구축해 표준화된 내비게이션 태스크에서 인터페이스별 성능을 비교했다. 또한 position/orientation 오차, 완료 시간, NASA-TLX로 효과를 측정해 단순 주관 평가가 아닌 구조적 검증을 수행했다.

- **Empirical Impact**: 실험에서 3D 시각화는 2D 인터페이스 대비 공간 정확도를 크게 개선했으며, 중간 위치 오차를 13mm에서 3mm로 줄이고 자세(orientation) 오차는 절반 수준으로 낮췄다. 더 나아가 tip-level 상호작용은 orientation error를 추가로 50% 더 줄이고, joint-level 대비 사용자 간 편차(interuser variability)도 감소시켰다. 종합하면 3D-TI(몰입형 3D 시각화+tip-level 직접 제어)가 가장 효과적이고 인체공학적이어서, 로봇 TEE의 조작 성능 향상과 시술 안전성 개선에 의미 있는 실증 근거를 제공한다.



### Motor Angular Speed Preintegration for Multirotor UAV State Estimation (https://arxiv.org/abs/2606.19929)
- **Prior Approaches**: 기존 UAV 자세/상태 추정은 느린 포즈 측정(예: LiDAR·카메라)과 IMU의 고주파 관성 정보를 결합하는 방식이 주류다. 하지만 프로펠러 회전에서 발생하는 진동이 IMU 측정을 저하시켜, 결과적으로 상태 추정 정밀도가 떨어질 수 있다.

- **Core Contribution**: 이 논문은 모터 속도(motor speeds)로부터 가속도(acceleration)를 사전 적분(preintegration)해 상태 전파에 활용하는 접근을 제안한다. IMU 없이도 전파만으로 더 높은 정밀도를 얻을 수 있으며, 모터 속도 사전 적분으로 만든 factor를 factor graph 최적화에 바로 쓸 수 있게 했다.

- **Technical Challenges**: 핵심 난제는 모터 속도에서 유도한 가속도가 실제 IMU 기반 관성처럼 신뢰도 있게 시간 누적되도록 만드는 것이다. 이를 위해 속도 기반 가속도 사전 적분을 설계하고, 최적화 프레임워크에서 바로 사용할 수 있는 preintegrated motor speeds factor를 구성해 LiDAR와 결합해 MAS-LO로 구현했다.

- **Empirical Impact**: 평가 결과 LIO-SAM 대비 위치 정확도 28%, 속도 정확도 65% 향상과 함께 측정 지연은 14% 낮아졌다. 또한 잘못된 파라미터 값에도 높은 강건성을 보이며, IMU 진동 문제를 우회하는 실용적인 정밀 상태 추정 경로를 제시한다.



### SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour (https://arxiv.org/abs/2606.19928)
- **Prior Approaches**: 기존 라티언트 world model 계열은 고차원 동역학을 데이터 기반으로 압축해 예측하지만, 강한 좌우 대칭 상호작용을 좌우 독립 패턴처럼 중복 인코딩하는 문제가 있었다. 그 결과 샘플이 제한된 극한 민첩 작업에서 기하학적 규칙성을 충분히 포착하지 못해, 하류(정책) 학습이 latent 공간을 비효율적으로 쓰게 된다.

- **Core Contribution**: 이 논문은 end-to-end 방식으로 좌우 대칭(symmetric)과 equivariance를 world model과 actor-critic 네트워크에 동시에 내장한 SWAP을 제안한다. 대칭 관측은 대칭 latent로, 대칭 상태에서의 최적 행동도 대칭 행동으로 나오도록 네트워크 구조 수준에서 일관성을 강제해준다.

- **Technical Challenges**: 핵심 난제는 대칭 구조를 “표현 학습”과 “장기 예측/제어” 전 구간에 걸쳐 일관되게 유지하는 것이다. SWAP은 RSSM 기반 latent dynamics에 SE-CNN/SE-GRU 등 equivariant 모듈을 넣고, actor는 symmetry-compliant 출력, critic은 invariant value 추정을 하도록 토폴로지 제약을 설계해 좌우 중복 인코딩과 비대칭 학습 함정을 줄였다.

- **Empirical Impact**: 시뮬레이션 ablation에서 단순 symmetry penalty(SymLoss)나 정책만 제외한 변형보다 SWAP이 더 높은 성공률과 난이도 확장성을 보였다. 특히 로봇 실험에서 Apollo 쿼드러펫이 2.13 m 갭 점프와 1.63 m 플랫폼 등반을 zero-shot으로 달성하며 기록을 세웠고, 거울로 뒤집힌 미지형에서의 기하학적 일반화와 다양한 야외 환경 전이도 강건하게 나타났다.



### Deep-Unfolded Coordination (https://arxiv.org/abs/2606.19920)
Comments:
          The second and third authors contributed equally (equal second authorship). 35 pages (10 pages main text), 17 figures, 3 tables

- **Prior Approaches**: 분산 최적화는 다중 로봇 문제를 병렬로 쪼개고 해석 가능성이 높다는 장점이 있지만, ADMM-DDP는 페널티 파라미터 등 하이퍼파라미터 튜닝 의존도가 커서 스케일이 커질수록 조정 변수가 폭증한다. 또한 기존 deep-unfolding(L2O)은 주로 반복별 고정 가중치를 학습해 ‘open-loop’ 성격이 강해 옵티마이저 상태에 따른 피드백 적응이 제한적이었다. 비슷한 시도로 ADMM 학습을 GNN/MLP 기반으로 확장한 연구들이 있었지만, (페널티) 파라미터 적응 범위나 비볼록 최적화에서의 학습 안정성 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 ADMM-DDP의 페널티 파라미터를 solve-time에 동적으로 조절하도록 설계한 deep-unfolding 프레임워크 Deep Coordinator를 제안한다. ADMM-DDP의 반복을 고정된 K회 레이어로 언롤(unroll)하고, 각 레이어 사이에 ‘옵티마이저 성능(상태)’을 입력으로 다음 페널티 파라미터를 출력하는 learnable feedback 정책을 둔다. 또한 비볼록 deep-unfolded 모델을 mainstream supervised 방식으로 학습하면 degenerate solution이 생길 수 있음을 지적하고, 이를 피하는 unsupervised 학습식을 제안한다.

- **Technical Challenges**: 핵심 기술 도전은 (1) 비볼록 최적화에서 페널티를 solve-time으로 학습·적응시키되 안정적으로 학습하려는 문제와 (2) K회 ADMM-DDP를 end-to-end로 미분하려면 내부(DDP/서브문제) 반복까지 unrolling하면 계산·메모리 비용이 너무 커지는 문제다. 이를 위해 논문은 IFT(Implicit Function Theorem)를 활용해 내부 솔버 반복을 역전파 경로에서 제외하는 그래디언트 계산 스킴을 도입해 O(K·T) 수준의 비용으로 미분 가능하게 만들었다. 학습 손실은 비용(cost)과 제약 위반(constraint violation)을 함께 다루는 형태로 구성해 residual 기반의 퇴행(stalling) 유인을 줄였다.

- **Empirical Impact**: 시뮬레이션에서 Deep Coordinator는 자동차/쿼드로터 다중 로봇 태스크 3종(장애물 회피, 교차, 장애물 비행)에서 기존 최적화 대비 동일 수준의 품질을 훨씬 빠르게 달성했다. 특히 Vanilla ADMM-DDP 대비 6.18–9.44x 빠르게(동일 반복 예산 내) 비슷한 수준의 비용과 제약 만족을 보였고, 비용 기준으로 더 오래 돌린 Vanilla 해보다도 6.18–9.44x의 벽시계 시간 이득을 보고했다. 더 나아가 학습한 팀 규모보다 최대 8배 큰 환경에도 배치했을 때 성능 이점이 유지되며, 이는 ADMM-DDP 구조가 제공하는 inductive bias와 로컬 피드백 설계 덕분이라는 해석을 제시한다.



### Co-policy: Responsive Human-Robot Co-Creation for Musical Performances (https://arxiv.org/abs/2606.19914)
- **Prior Approaches**: 기존 로봇 음악 연구는 대체로 사람(또는 시스템)이 지정한 음표를 로봇이 재생하는 playback 방식에 가깝다. 또 다른 접근으로 diffusion policy처럼 반복적 denoising을 통해 동작/행위를 생성하는데, 이런 과정은 사람과 맞물려야 하는 음악 상호작용의 저지연 요구에 불리할 수 있다. 마지막으로 단일(mean) 행동을 회귀하는 imitation learning은 다중 실행 모드가 평균화되어 표현성이 떨어질 위험이 있다.

- **Core Contribution**: Co-policy는 인간의 불완전한 음악적 시드(말/라이브 시드 음/비전)를 바탕으로, 의미(semantic intent)–음악적 보완(상보적 변주)–물리 실행(비주얼모터)을 분리해 end-to-end로 한 번에 해결하려 하지 않는 프레임워크를 제안한다. 핵심은 VLM(Qwen-vl)에 semantic anchor bank를 결합해 JSON 형태의 structured co-creation plan을 만들고, 그 계획을 제약 기반 변주 플래너가 이어받아 로봇이 복사하지 않는 complementary 응답을 생성한다는 점이다. 여기에 저지연 실행을 위해 단일 forward pass의 GMP(Gaussian-Mixture Visuomotor Policy)로 다중 모드 동작 분포를 직접 예측한다.

- **Technical Challenges**: 가장 어려운 점은 (1) 음성/시드/비전에서 음악적 의도를 해석하되, 악기 제약과 타이밍 제약을 만족하는 계획으로 안정화하는 것이다. 논문은 pre-inference semantic anchors로 VLM이 보완/변주를 “계획” 형태로 내도록 유도하고, motif consistency·harmonic validity·novelty·embodied playability 같은 제약을 플래너 단계에서 필터링한다. (2) 또한 저지연 물리 실행을 위해 반복적 denoising을 피하면서도 다중 실행 모드를 평균화하지 않는 것이 필요해, GMP를 conditional mixture-density policy로 설계해 대안적 스트라이크 모드를 분포로 유지한다.

- **Empirical Impact**: 실로봇 chime 실험과 ablation, 전문가 블라인드 평가에서 Co-policy는 의도 정합성·실행 정확도·응답 빈도 측면에서 diffusion-policy 및 제거된 baseline들보다 향상되었다. 특히 semantic anchoring의 효과는 Qwen-vl/F-Qwen 비교에서 음악적 co-creation 품질을 7.4%~13.3% 끌어올리는 등 의미 정렬에 강하게 기여함이 드러났다. 동작 실행에서는 diffusion-policy 대비 더 높은 성능(약 15%)과 더 높은 response frequency를 보였고, 모듈 기여도(예: GSA·mixture head)가 확인되면서 “물리적으로 grounded된 action 생성”이 embodied co-creation의 핵심 요구임을 실증했다.



### One-to-Two Acting: A Novel Framework for Single-arm Agent Action Expansion to Dual Arms (https://arxiv.org/abs/2606.19897)
Comments:
          6 pages, 5 figures, 3 tables

- **Prior Approaches**: 듀얼암 조작은 병렬 실행으로 처리량을 늘리지만, 기존에는 bimanual 데모 수집이 비싸고 까다로워 대규모 듀얼암 데이터 기반 imitation learning이 주류였습니다. 또한 single-arm 기반 LLM/vision-language 에이전트는 본질적으로 순차적 행동 생성에 최적화돼, 팔 간 선후관계·공간 의존성·동기화 조율을 충분히 반영하지 못해 동시 작업에서 비효율이 커졌습니다. 일부 연구는 부분순서 계획이나 PDDL 플랜을 제안하지만, 정밀 조작을 위한 그리피한 실행 제약과 충돌 없는 동시 실행까지는 한계가 있었습니다.

- **Core Contribution**: ExS2D(Extending Single-arm agent actions to Dual arms)는 단일암 감독만으로 듀얼암 조작을 가능하게 하는 계층형 프레임워크를 제안합니다. 텍스트 지시로부터 구조화된 서브태스크를 만들고, 각 서브태스크를 관측 기반 action으로 정밀 그라운딩한 뒤, 선후관계(precedence)를 명시적으로 고려해 두 팔에 충돌 없는 실행을 배정합니다. 핵심은 “bimanual 데모 없이도” 듀얼암 실행을 학습/구성할 수 있도록 파이프라인의 각 단계를 분해해 설계한 점입니다.

- **Technical Challenges**: 첫째, 장기 지시는 시간적으로 일관된 서브태스크 집합을 얻기 어렵습니다; ExS2D는 MLLM(VL-SubGen)이 정적 장면 이해와 시각 변화 기반 동적 이해를 함께 학습해 temporally consistent한 서브태스크 시퀀스를 생성하도록 했습니다. 둘째, 서브태스크를 실제 pick–place 프리미티브로 매핑하려면 모호한 관측에서 객체 위치/마스크 정밀도가 중요합니다; SA-Map은 VLM+SAM으로 의미 마스크를 만들어 CLIPort의 공간 주의를 강화해 그라운딩 신뢰도를 높입니다. 셋째, 두 팔 동시 실행은 선후관계와 충돌 회피를 동시에 만족해야 하므로, P-DCoord가 precedence-aware로 ready set을 구성하고 모션 비용 휴리스틱으로 후보를 좁힌 뒤 synchronized RRT*와 collision checking로 실행 가능성을 검증합니다.

- **Empirical Impact**: 시뮬레이션에서 ExS2D는 단일암 baseline 대비 성공률을 거의 유지하면서 평균 실행 스텝을 54.4% 줄였습니다. 듀얼암 플래너들과 비교해도 더 높은 성공률과 더 적은 스텝을 보이며, 특히 temporal dependency가 강한 작업에서 precedence-aware 배정의 이점이 두드러졌습니다. 실제 로봇(Elеphant Pro 630) 4개 작업에서도 few-shot 단일암 샘플만으로 60.51%의 성능을 보였고, bimanual 데모와 듀얼암 데이터 없이도 시간 의존 조작과 접힘(folding) 같은 협응 작업을 안정적으로 수행했습니다. 또한 ablation에서 의미 마스킹과 precedence 추론이 없을 때 SR 하락이 크게 나타나, 효율-성공의 균형이 구조적 설계에서 비롯됨을 입증했습니다.



### MMD-SLAM: Structure-Enhanced Multi-Meta Gaussian Distribution-Guided Visual SLAM (https://arxiv.org/abs/2606.19874)
Comments:
          ICRA 2026

- **Prior Approaches**: 기존 Visual SLAM은 카메라 포즈 추정에 강점이 있으나, 지도는 point cloud나 voxel처럼 저해상도 표현에 머무르는 경우가 많아 AR/VR 및 Embodied AI의 고충실도 요구를 충분히 충족하지 못했습니다. NeRF 기반 접근은 디테일을 늘렸지만 over-smoothing과 계산 효율 문제를 겪고, 3D Gaussian Splatting(3DGS) 기반 SLAM은 렌더링 속도·품질을 개선했음에도 장면의 구조적 규칙을 충분히 쓰지 못해 맵 일관성이 흔들리는 한계가 있었습니다. MG-SLAM은 line 특징과 Manhattan World 가정을 활용하지만, discrete feature와 Gaussian primitive의 결합이 제한적이고 가정 자체가 적응성에 제약을 줍니다.

- **Core Contribution**: 이 논문은 Atlanta World(AW) 가정을 활용해 구조 정보를 지도에 더 적극적으로 반영하는 구조강화 Visual SLAM MMD-SLAM을 제안합니다. 추적(tracking)에서는 point–line 결합 제약으로 포즈 최적화의 안정성과 구조 단서를 확보하고, 매핑(mapping)에서는 AW 유도 Multi-Meta Gaussian(점/선/면 모달리티 및 지배 방향)을 통해 기하 구조를 명시적으로 인코딩합니다. 또한 Multi-Meta Gaussian evolution 전략으로 Weak/Stable 상태를 진화시키며 장면 기하에 맞춘 전역 최적화를 수행합니다.

- **Technical Challenges**: 핵심 난제는 (1) 포즈 최적화에 도움이 되는 line 구조 단서를 어떻게 3DGS의 Gaussian 원시(primitive)와 일관되게 결합하느냐, (2) AW 기반 구조 프라이어를 실제 데이터에서 어떻게 안정적으로 학습·적응시키느냐입니다. 저자들은 point 재투영에 더해 3D line segment를 추가 제약으로 넣고, pose 최적화의 global bundle adjustment에서 점·선 reprojection을 함께 최소화합니다. 매핑에서는 Gaussian을 점/선/면으로 범주화하고 지배 방향(dominant direction)과 shape/directon loss를 결합하며, Weak에서 Stable로의 진화·클론/스플릿/머지 연산으로 다중 장면 적응성과 기하 적합도를 동시에 노립니다.

- **Empirical Impact**: 실험은 RGB-D/대규모 합성 데이터에서 tracking 정확도와 포토리얼 맵 품질을 모두 평가하며, MonoGS 대비 ScanNet에서 ATE RMSE를 48.56% 줄이고 Replica에서 PSNR을 5.71% 개선하는 성과를 보고합니다. 정성 비교에서도 isotropic Gaussian만 쓰는 SplaTAM의 맵 홀 문제, 다른 베이스라인의 바닥/천장 복원 결함 등을 완화하며 구조 단서 복원이 더 잘 이뤄짐을 보여줍니다. 결과적으로 MMD-SLAM은 3DGS 기반 SLAM이 구조적 규칙을 지도 품질에 직접 반영할 수 있음을 실증하며, embodied perception용 고충실도 맵 구축에 의미 있는 진전을 제시합니다.



### World Engine: Towards the Era of Post-Training for Autonomous Driving (https://arxiv.org/abs/2606.19836)
Comments:
          Technical Report. Project Page: this https URL

- **Prior Approaches**: 기존 end-to-end 자율주행 정책은 대규모 fleet log로 평균적인 주행 능력은 크게 끌어올렸지만, 안전 경계는 오히려 데이터에 희소한 long-tail 안전사건에서 결정되는 문제가 남아 있습니다. 또한 현실에서 사고로 이어질 수 있는 상황을 대규모로 수집하는 데는 윤리·법·사회적 제약이 커서, 가장 중요한 학습 신호가 자연 데이터에 구조적으로 부족합니다. 따라서 단순히 pre-training 데이터만 늘리는 접근은 희귀한 실패 케이스에서 점점 수익이 줄어드는 diminishing returns가 나타납니다.

- **Core Contribution**: World Engine(WE)은 현실 로그에서 failure-prone long-tail 상황을 찾아내고, 이를 고해상도 상호작용 시뮬레이션으로 재구성·확장한 뒤 reinforcement 기반 post-training으로 정책의 안전을 개선하는 프레임워크를 제안합니다. 핵심 아이디어는 “현실에서의 위험한 탐색”을 직접 하지 않고도, 생성된 safety-critical 상호작용으로 정책을 학습시켜 안전 제약에 더 잘 맞추는 것입니다. WE는 nuPlan 기반 벤치마크와 더불어 프로덕션 규모 시스템에까지 적용해 성능 향상을 확인합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) long-tail 사건을 현실 로그에 기반해 발굴하는 것, (2) 발굴된 사건을 실제와 같은 관측으로 다시 만들고 closed-loop에서 학습 가능한 시뮬레이션을 제공하는 것, (3) 희귀 상황에서만 강하게 개선하면서도 흔한 상황 능력을 무너지지 않게 제어하는 것입니다. WE는 3D Gaussian Splatting 기반 simulation engine으로 장면을 photorealistic하게 재구성하고, diffusion 기반 behaviour world model로 주변 차량의 다양하고 제어 가능한 반응을 생성한 뒤, KL divergence 페널티를 포함한 behaviour-regularized reinforcement learning과 hard experience mining으로 post-training을 안정화합니다.

- **Empirical Impact**: 누Plan 기반의 공개 벤치마크에서 WE는 long-tail safety-critical 시나리오의 실패를 크게 줄이며, pre-training 데이터만 늘리는 스케일링 대비 더 큰 폭의 개선을 보였다고 보고합니다. 또한 프로덕션 규모 ADAS 개발 스택에서도 post-training 후 충돌이 최대 45.5% 감소하고, 200km 실주행 on-road 테스트에서 disengagement 0 및 안전 관련 개선을 확인했습니다. 결과적으로 ‘합성된 safety-critical 상호작용에 대한 post-training’이 안전을 더 확장 가능한 방식으로 끌어올리는 실용적 경로임을 시사합니다.



### TIDY: Thermal Infrared Image Denoising via Wavelet Domain Entropy and Directional Stripe Index (https://arxiv.org/abs/2606.19813)
- **Prior Approaches**: 열화상(TIR) 영상은 저조도에서 강건하지만, 무냉식 마이크로볼로미터 특유의 확률적 잡음과 고정패턴 잡음(FPN)이 다운스트림 추정(상태추정·깊이·오도메트리)을 깨뜨려 실내 보급이 제한돼 왔다. 기존 딥러닝 기반 denoising은 대체로 합성 잡음이나 self-supervised에 의존해 실제 센서 잡음과의 도메인 갭이 크고, 성능을 올리기 위해선 무거운 모델(예: diffusion)이 필요해 온라인 로보틱스 제약을 넘기 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 TIR 잡음 제거를 “로보틱스 관점의 온라인·제로샷 강건성” 문제로 재정의하고, 경량 wavelet-domain denoiser TIDY를 제안한다. 특히 DWT/IDWT를 통해 잡음과 구조를 분리하도록 설계하고, 실측 clean-noisy 페어로 학습해 합성 잡음 의존을 줄이면서도 구조 손상을 최소화하는 데 초점을 둔다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 실제 clean-noisy TIR 페어 데이터 부재로 합성 잡음 학습이 반복됐고 (2) wavelet 기반은 정보·전역 문맥 손실로 디테일이 흐려질 수 있다는 점이다. 논문은 첫째로 SCaN-TIR(스테레오 실측 clean-noisy 페어 32.5k)을 구축해 실잡음 학습을 가능케 했고, 둘째로 FiLM으로 전역 문맥을 wavelet 특징에 주입해 DWT로 인한 손실을 완화했으며, WE(확률적 잡음 억제)와 WDSI(방향성 stripe/FPN 억제)를 loss로 추가해 잡음 유형별 억제를 명시화했다.

- **Empirical Impact**: 실험에서 TIDY는 심한 실내 손상과 zero-shot 환경에서 고정패턴/확률적 잡음을 동시에 더 잘 억제하면서도 온라인 속도(640×512에서 약 34Hz)를 달성해 기존 denoising 대비 정확도-지연 균형을 개선했다. 또한 thermal inertial odometry(VINS-Mono 기반)와 단안 depth 추정(MDE)에서 다운스트림 성능이 일관되게 향상돼, 로보틱스 추정 파이프라인에 “플러그앤플러그”로 연결될 수 있는 실효성을 보여준다.



### EquiVLA: A General Framework for Rotationally Equivariant Vision-Language-Action Models (https://arxiv.org/abs/2606.19784)
Comments:
          Comment: First version 22 pages, project site: this https URL

- **Prior Approaches**: VLA(Vision-Language-Action) 모델은 pretrained VLM(vision-language model) 표현에 flow-matching 기반 action head를 붙여 언어 지시에 따른 일반 조작을 잘 수행하지만, 회전 대칭(SO(2)) 같은 기하적 구조를 아키텍처에 넣지 못한다. 그 결과 특정 방향에서 학습한 정책은 다른 방향에 배치될 때 더 많은 데이터로 같은 기술을 다시 “암기”해야 하며, 데이터 증강으로는 완전한 보장이 어렵다. 기존 equivariant 정책 연구는 SO(2) 등을 모델 전체에 넣는 경우가 많지만, 대규모 VLA 파이프라인처럼 frozen VLM+DiT action head 조합에 end-to-end equivariance를 전파한 사례는 부족했다.

- **Core Contribution**: 본 논문은 frozen VLM backbone과 flow-matching Diffusion Transformer action head를 결합한 end-to-end SO(2)-equivariant VLA 프레임워크 EquiVLA를 제안한다. 핵심은 두 모듈 EquiPerceptor와 EquiActor로, 카메라 관측부터 예측 action 시퀀스까지 약(approximate) SO(2) equivariance 사슬을 구성한다. 특히 VLM 가중치는 수정하지 않고도 기하 귀납편향을 도입하는 일반 프레임워크라는 점이 기여로 제시된다.

- **Technical Challenges**: 첫 번째 난제는 ViT의 공간 토큰이 회전될 때 토큰 위치가 같이 이동해, 단순 Frame Averaging을 하면 공간 정렬이 깨져 조작에 필요한 위치 정보를 잃는다는 점이다. EquiPerceptor는 회전군 항에 대해 토큰을 “되돌리는” 공간 permutation과 regular representation 기반 특징 변환을 함께 적용해 토큰 수준에서 약 SO(2) equivariance를 유지하도록 설계했다. 두 번째 난제는 action head에서 구조적으로 steerable layer를 써야 정확 equivariance가 되는데, 이는 pretrained DiT 헤드의 가중치 재사용이 어렵다는 점이며, EquiActor는 equivariant attention/인코딩/디코딩을 regular feature space에서 구성해 이 문제를 해결한다.

- **Empirical Impact**: 실험에서 EquiVLA는 LIBERO 네 가지 suite에서 평균 성공률 92.6%(baseline 78.1%)를 달성하며, 회전 일반화 이득이 시뮬레이션 전반에서 일관되게 나타났다. CALVIN ABCD→D에서는 평균 sequence length가 4.03(기준 3.45)으로 목표 접근/수행의 안정성이 개선됐고, history-based 방법에 가까운 성격을 보인다고 보고한다. 또한 Mobile ALOHA real-robot 5개 작업에서 성공률을 54%에서 72%로 끌어올려, 데이터 효율 관점의 equivariance 설계가 실제 로봇에도 통한다는 점을 실증했다.



### Start Right, Arrive Right: Asynchronous Execution via Initial Noise Selection (https://arxiv.org/abs/2606.19774)
Comments:
          First version 19 pages, project site: this https URL

- **Prior Approaches**: 액션 청킹(action chunking)은 로봇이 시계열적으로 일관된 동작을 만들게 하지만, flow matching·diffusion 계열은 한 청크를 생성하기 위해 여러 denoising 단계를 거쳐 지연이 커진다. 비동기 실행에서는 이전 청크를 계속 수행하는 동안 다음 청크가 생성되며 delay가 생겨 청크 경계에서 prefix(앞부분) 불일치가 문제가 된다. 기존 방법들은 이 불일치를 해소하기 위해 생성 중인 궤적/속도를 prefix에 맞추는 steering을 수행하며, 예를 들어 정책 그래디언트 기반 보정(RTC), 재학습, 후보 필터링·후처리(B-spline 등)처럼 추가 연산/학습 비용을 치르거나 출력 후 단계 개입에 그쳤다. 또 많은 접근이 “초기 noise”를 수정하기보다 생성 이후(denoising 중 또는 결과)에 개입한다는 한계가 있었다.

- **Core Contribution**: PAINT는 prefix 불일치를 “생성 중 교정”이 아니라 “생성 시작 전 초기 noise 선택” 문제로 재정의한다. OT-FM(Optimal-transport flow matching)에서 관측되는 locality 성질에 기반해, 적절한 initial noise를 찾아두면 unmodified flow ODE가 자연스럽게 이미 실행된 prefix에 잘 이어지는 다음 청크를 만든다고 제안한다. 이를 위해 PAINT는 training-free 방식으로 backward Euler inversion으로 원하는 prefix를 만족시키는 noise를 찾고, repainting 규칙으로 나머지 구간을 구성해 최종 청크를 만든다. 핵심은 정책/속도장(vπ)을 수정하거나 그라디언트를 계산하지 않고도 prefix consistency를 확보한다는 점이다.

- **Technical Challenges**: 기술적 난제는 비동기 실행에서 “실제로 로봇이 소비해버린 prefix”와 “다음 청크의 생성 결과”가 수치해석(유한-step solver) 오차 및 manifold 이탈 때문에 정확히 정렬되지 않는다는 데 있다. PAINT는 backward Euler inversion에서 target을 구성할 때 prefix 위치는 고정하되 나머지 영역은 naive forward에서 얻은 tail을 그대로 유지해 on-manifold 상태를 보존함으로써 inversion 불안정을 줄인다. 또한 단순히 free 구간을 새 랜덤 noise로 바꾸면 토큰(위치) 혼합으로 mismatch가 전파될 수 있어, repainting principle에 따라 free 구간의 noise를 신중히 재조립한다. 결과적으로 “초기 noise만 제어”하면서도 unmodified ODE forward로 prefix 제약을 근사 만족시키는 경로를 만든다.

- **Empirical Impact**: 시뮬레이션에서 PAINT-Euler는 지연(delay)이 커질수록 성능이 급락하는 naive async 대비 우수한 prefix consistency와 task success를 보이며, RTC 수준 또는 그 이상을 보여 delay robustness가 강했다(12개 simulated benchmarks). 또한 6개의 real-world 조작 과제(단일팔·이족·인간형 embodiment, 두 VLA 아키텍처)에서도 PAINT는 RTC 대비 대체로 동등/개선 성과를 보였고, 정책 그래디언트 기반 보정 없이도 prefix 관련 연속성이 향상되는 경향이 확인됐다. 특히 RTC는 prefix용 gradient guidance 과정에서 suffix까지 교란될 수 있어 접촉·변형물 과제에서 불안정할 수 있는데, PAINT는 속도장을 건드리지 않는 대신 noise 선택으로 이 문제를 완화하는 것으로 해석된다. 결론적으로 PAINT는 그래디언트·재학습 없이도 비동기 청크 경계 일관성을 높일 수 있어, 지연이 불가피한 배포 환경에서 flow-based 로봇 정책의 실용성을 끌어올리는 접근으로 의미가 크다.



### Data Standards for Humanoid Robotics: The Missing Infrastructure for Physical AI (https://arxiv.org/abs/2606.19769)
- **Prior Approaches**: 기존 접근은 모델과 하드웨어 성능 향상에 집중했지만, 로봇 경험이 다른 몸체·과제·현장으로 누적되려면 데이터가 먼저 재사용 가능해야 한다. 그러나 현재는 데이터가 센서 구성, 좌표계, 타이밍, 태스크 정의, 라벨 규칙, 품질 기준이 달라 서로 해석이 어렵고, 축적해도 공유 역량으로 이어지지 않는 문제가 크다.

- **Core Contribution**: 이 논문은 Humanoid robot datasets에 대한 ISO/WD 26264-1(일반 요구사항)과 같은 국제 표준화가 Physical AI의 핵심 인프라가 되어야 한다고 주장한다. 단순한 파일 포맷이 아니라, 신체(embodiment)·행동·과제·장면·실행 추적·결과를 한 물리 에피소드로 묶어 “데이터가 여행(share/trace/reuse)”할 수 있게 만드는 구조를 제안한다.

- **Technical Challenges**: 문제의 핵심 기술 난관은 (1) 신체 기반 데이터가 분리되면 의미를 잃는 ‘해석 가능성’과 (2) 멀티모달 스트림이 타이밍·공간 정합·캘리브레이션·단위·동기화 가정이 명시되지 않으면 ‘물리적 일관성’이 깨지는 점이다. 이를 위해 표준은 에피소드 스키마, 물리 일관성 기록(시간 기준·동기화·좌표변환·캘리브레이션·단위 등 가정 검증 가능), 라이프사이클 기록(프로비넌스·품질·버전·권리·사용 조건·평가 디스크립터)을 데이터 패스포트 형태로 요구한다.

- **Empirical Impact**: 논문은 더 많은 데이터 수집이 곧바로 누적 역량을 만들지 못하며, 비용·데이터 사일로·평가 공백으로 인해 ‘비누적(non-cumulative) 데이터’가 생긴다고 정리한다. 표준은 서로 다른 연구/조직이 동일한 기준으로 비교·검증·재사용할 수 있게 해 중복 수집을 줄이고, 장기적으로 누적 가능한 휴머노이드 능력을 확산시키는 데 의미가 있다.



### Temporal Self-Imitation Learning (https://arxiv.org/abs/2606.19752)
- **Prior Approaches**: 기존 장기 로봇 조작 강화학습은 누적 reward에 주로 의존하며, dense reward shaping과 sparse task-success 보상을 혼합해 탐색을 돕는 방식이 많았습니다. 하지만 dense 보상은 빠르지 않더라도 중간 상태를 반복해 높은 리턴을 만들 수 있어, ‘효율적인 성공’과 ‘보상에 분산된 느린 성공’을 구분하지 못한다는 한계가 있습니다. 또한 rare하게 발견되는 빠른 성공 행동은 on-policy 갱신 과정에서 분포 변화나 최적화 불안정 때문에 쉽게 사라질 수 있습니다.

- **Core Contribution**: 이 논문은 시간적 효율(temporal efficiency) 자체를 reward shaping을 넘어서는 self-supervision 신호로 활용하자고 제안합니다. Temporal Self-Imitation Learning (TSIL)은 학습 중 발견된 ‘시간이 짧은 성공 궤적’을 골라 재사용 가능한 감독으로 바꾸며, 구성(configuration) 조건에 따른 adaptive temporal target을 점진적으로 조여 줍니다. 동시에 효율 가중 self-imitation을 통해 빠른 성공 행동을 replay하고 지속적으로 다시 학습에 포함시킵니다.

- **Technical Challenges**: 핵심 난제는 (1) 글로벌한 시간 목표를 쓰면 쉬운 설정에서 배운 효율을 어려운 설정에 강제하게 되는 문제와, (2) 느리게 성공한 궤적이나 reward에 분산된 패턴이 self-imitation에 섞여 학습을 왜곡하는 문제입니다. TSIL은 학습 중 성공 궤적의 완료 시간을 구성별 D(ϕ)로 업데이트해 adaptive temporal target을 만들고, 이를 관측과 reward shaping(성공지점에 temporal bonus)을 통해 부드럽게 학습 압력으로 전환합니다. 더 나아가 replay에서는 ‘빠른 성공’만 우선 저장하고, 자기학습 압력은 advantage 기반 게이팅과 효율 가중치로 조절해 느린/덜 신뢰할 만한 궤적 재생을 완화합니다.

- **Empirical Impact**: 15개 장기 조작(관절형 물체 상호작용, 삽입, 도구 사용, 운반, 접촉-rich 조작 포함)에서 TSIL은 success rate와 AUC 같은 학습 성능뿐 아니라 성공 완료 시간(behavioral efficiency), 빠른 성공 행동의 재방문, 불안정한 학습 조건에 대한 robustness까지 일관되게 개선했습니다. 특히 reward-engineering과 adaptive temporal target만 쓴 변형 대비, ‘효율 기반 궤적 채굴+재생’의 결합이 전반적 트레이드오프에서 가장 좋은 결과를 보였습니다. 결과적으로 성공의 시간 구조가 사람이 설계한 reward shaping에 덜 의존하면서도 확장 가능한 self-supervisory signal이 될 수 있음을 실험적으로 뒷받침합니다.



### VOiLA: Vectorized Online Planning with Learned Diffusion Model for POMDP Agents (https://arxiv.org/abs/2606.19729)
Comments:
          Submitted to the 2026 International Symposium of Robotics Research (ISRR)

- **Prior Approaches**: POMDP는 불확실성 추론의 표준 프레임워크지만, 실제 로봇 적용은 전이/관측 모델을 ‘정확히’ 만드는 일이 가장 큰 병목이었습니다. 온라인 근사 POMDP 풀이는 모델이 사전에 잘 준비돼 있어야 하고, RL 기반 접근은 명시적 모델을 줄이지만 대개 대규모 데이터와 학습 시간이 필요합니다. 검색+학습을 결합한 방법들도 GPU 같은 병렬 자원을 충분히 활용하지 못해 실시간성 제약이 커졌습니다.

- **Core Contribution**: VOiLA는 작업(task)-비특이적인 POMDP 모델을 학습해 온라인 계획에 바로 쓰는 프레임워크입니다. 전이 샘플러와 관측 샘플러는 conditional diffusion model로 생성 분포를 배우고, belief 업데이트에는 particle 기반 가중치 계산을 위한 observation-likelihood 모델을 추가로 학습합니다. 또한 diffusion 기반 샘플러를 distillation로 압축해, VOPP(Vectorized Online POMDP Planner)와 결합함으로써 연산 효율을 끌어올립니다.

- **Technical Challenges**: 핵심 어려움은 (1) 다봉/비선형인 로봇 전이와 (2) 고차원 관측(예: 이미지) 분포를 생성적으로 모델링하되, (3) 온라인 POMDP 탐색 중에 사용할 수 있을 만큼 샘플링 비용을 낮추는 것입니다. VOiLA는 diffusion을 먼저 고표현력으로 학습한 뒤, ODE 기반 diffusion sampler를 feedforward 생성기로 distill해 샘플링 비용을 크게 절감합니다. 연속 관측에 대해서는 VOPP에 progressive widening을 벡터화해 확장하고, belief 업데이트는 정규화 상수 부담을 줄이기 위해 unnormalized likelihood를 InfoNCE류의 contrastive 학습으로 상대 가중치 형태로 구성합니다.

- **Empirical Impact**: 시뮬레이션 실험에서 distillation 전략은 샘플링 비용을 최대 거의 3자릿수(orders of magnitude)까지 줄여, learned generative POMDP 모델의 온라인 사용을 실용 범위로 끌어내렸습니다. 3개 벤치마크 평가에서 VOiLA는 Recurrent Soft Actor Critic과 비슷하거나 더 나은 성능을 보이면서도 학습 데이터는 10% 미만만 사용했고, 학습에 없던 환경 설정으로의 일반화도 더 우수했습니다. 물리 로봇 실험(쿼드루피드 타겟 파인딩)에서는 시뮬레이션으로 학습한 모델만으로 online POMDP 계획을 수행해 10/10 성공을 보고했습니다.



### Bidirectional Tutoring for Developmental Motor Learning in Robots: Co-Developed Interaction Dynamics Support Stable Learning (https://arxiv.org/abs/2606.19728)
Comments:
          16 pages, 14 figures

- **Prior Approaches**: 로봇의 발달적 모터 스킬 학습은 LfD나 imitation learning처럼 튜터가 제공하는 데모를 로봇이 일방적으로 받아들이는 unidirectional 방식으로 많이 다뤄졌다. 특히 kinesthetic teaching에서도 로봇이 튜터의 물리적 개입을 ‘통과적으로’ 받는 설정이 흔해, 튜터가 로봇의 현재 행동과 내적 동역학을 반영해 궤적을 공동 구성한다는 점이 약했다. 그 결과 성공적인 데모가 나오더라도 센서모터 패턴의 일관성이 부족해 단계별 학습이 흔들리거나 일반화가 제한될 수 있다는 우려가 제기된다.

- **Core Contribution**: 이 논문은 튜터-로봇 상호작용을 bidirectional tutoring으로 재정의하고, 로봇의 과거 경험이 prior constraint처럼 작동해 공동 개발된 궤적의 ‘행동적 응집성(behavioral coherence)’을 유지함으로써 단계별 학습과 일반화를 돕는다고 가설을 세운다. 이를 unidirectional 방식과 직접 비교해, bidirectional일 때 더 일관된 행동이 형성되고 튜터 의존이 점차 줄어드는 발달 스캐폴딩 효과가 나타난다는 그림을 제시한다. 또한 free-energy principle 기반의 PV-RNN을 generative replay와 결합해 단일 튜터 에피소드만으로도 stage-wise 학습이 안정적으로 진행되도록 구현한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) bidirectional 접촉 상호작용에서 생기는 잡음과 의도적 충돌을 실시간 추론/제어에 통합하고, (2) 단계마다 새 데모(단일 튜터 에피소드)를 추가하면서도 이전에 학습한 센서모터 패턴을 catastrophic forgetting 없이 보존하는 것이다. 논문은 FEP의 variational free energy를 최소화하는 PV-RNN의 온라인 inference(error regression)로 튜터 개입에 맞춰 동역학을 즉시 조정하고, generative replay로 과거의 재생 시퀀스를 저장 없이 샘플링해 망각을 완화한다. 특히 stage-wise 설정에서 ‘너무 많으면 과거에 편향, 너무 적으면 망각’ 문제를 피하려고 재생 시퀀드를 큰 버퍼에서 매 epoch 소량만 무작위 선택하는 설계를 사용한다.

- **Empirical Impact**: Torobo(휴머노이드)로 물체 조작 태스크를 수행하며, (i) 실제 인간 튜터 환경과 (ii) AI 튜터가 adaptive intervention으로 개입하는 보다 통제된 환경의 두 실험을 수행했다. 두 설정 모두에서 bidirectional tutoring이 unidirectional보다 일관된 튜터 궤적을 만들었고, 단계가 진행될수록 자율 성공률이 상승하며 튜터 의존이 점차 감소했다. 또한 학습 단계 전반에서 학습 손실(variational free energy)이 bidirectional에서 대체로 더 낮게 나타나, 공동 구성된 궤적이 기존 센서모터 동역학과 더 잘 정합됨을 시사한다.



### A Differentiable Composite Approximation Framework for Autonomous Underwater Vehicle Maneuvering Modeling from Sea-Trial Data (https://arxiv.org/abs/2606.19711)
- **Prior Approaches**: 기존 AUV 조종(만경) 모델링은 고정된 다항식(예: third-order) 기저로 유체역학 항을 근사하거나, 신경망으로 adaptive basis를 학습하는 방식으로 나뉘었습니다. 다항식 기반은 해석 가능·파라미터가 compact하지만 항 선택/공분산(공선성) 문제로 식별성과 재귀 예측이 흔들릴 수 있고, 신경망 기반은 잡음·부족한 sea-trial 데이터에서 스퓨리어스 상관을 과적합할 위험이 큽니다. 또한 grey-box에서 흔한 hybrid 절차는 polynomial prior를 먼저 “fit 후 freeze”하고 residual만 학습해, 기준선 오차까지 신경망이 떠안으면서 재귀 예측 성능이 저하될 수 있습니다.

- **Core Contribution**: 이 논문은 polynomial-basis 컴포넌트와 data-adaptive basis 컴포넌트를 하나의 differentiable composite-approximation으로 묶어, 동일한 예측 손실 하에서 동시 co-calibration(공동 보정)을 수행하는 프레임워크를 제안합니다. sensitivity-aware gradient-based 보정으로 다항식 계수는 bounded 범위 내에서만 업데이트하고, LSTM residual이 나머지 비선형 불일치를 학습하도록 설계했습니다. 더불어 turning-motion 기반 current estimation과 compensation을 넣어 바다의 유속 영향을 반영한 current-compensated 학습 타깃을 구성합니다.

- **Technical Challenges**: 핵심 난제는 (1) 제한된 in-field 데이터에서 다항식 계수의 drift와 식별 불안정, (2) 신경망이 환경/전류의 잔차를 동역학으로 착각해 일반화 성능을 잃는 문제, (3) sea-trial 측정에서 전류 성분이 속도에 섞여 학습 타깃이 오염되는 문제입니다. 논문은 polynomial을 trainable하지만 bounded correction 변수로 제한하고, skip connection으로 LSTM residual이 다항식 예측에 “조건부로” 보정하도록 하여 gradient 경로와 민감도를 조절합니다. 전류는 로컬 관성 좌표에서 locally constant로 가정한 turning 구간 회귀(삼각 여기)를 통해 Vccosβc, Vcsinβc를 추정하고, 이를 제거해 water-relative 프레임 타깃을 만든 뒤 rollout 시에는 다시 전류를 재주입합니다.

- **Empirical Impact**: 7m급 AUV의 해상 시험(sea-trial) 데이터로 다중 만경 조건을 평가했으며, recursive trajectory와 velocity prediction에서 polynomial-only, neural-only, frozen-prior hybrid baseline을 개선하는 결과를 보였습니다. 특히 전류 보정이 포함된 학습 타깃과, 다항식-신경망을 분리 학습하지 않고 공동 보정하는 구조가 재귀 시뮬레이션 오차 누적을 줄이는 데 기여한 것으로 나타났습니다. field-data 기반 AUV maneuvering modeling에서 interpretable hydrodynamic backbone과 flexible learning을 함께 얻을 수 있다는 점에서, 모델 기반 제어·디지털 트윈·상태 예측 파이프라인에 실용적 의미가 큽니다.



### Comparative Study on Agility, Efficiency, and Impact Absorption of Bipedal Robots with Active Toes (https://arxiv.org/abs/2606.19699)
Comments:
          6 pages, 7 figures

- **Prior Approaches**: 로보틱스에서 발가락을 구현하려는 시도는 오래됐지만, 대부분은 수동형(스프링 등) 또는 구조가 단순한 형태에 머물러 사람 발가락의 힘(토크) 조절 특성을 재현하지 못했다. 그 결과 보행 효율·충격 흡수 같은 장점이 정성적으로는 보고돼도, 실제 액추에이터 동특성과 비용(energy, CoT)을 정량 비교하기 위한 검증이 부족했다. 또한 시뮬레이터가 로봇을 단순 관절-액추에이터 1:1 구조로 가정하면 sim-to-real 간극이 커져 복잡한 다관절 설계의 이득을 제대로 가늠하기 어렵다.

- **Core Contribution**: 이 논문은 사람 발가락의 가벼움·고토크·견고함을 목표로 14-DOF 양지 로봇에 active toe를 포함시키고, 발가락 유무에 따른 기여를 공정하게 분리해 평가한다. 핵심은 (1) 결합전달(coupled transmissions)을 반영하는 고신뢰도 시뮬레이션 환경, (2) CoT를 직접 추정·최적화에 반영하는 학습 설계, (3) active toe 장착/제거 두 설정을 동일한 절차로 학습해 비교의 편향을 줄인 점이다. 이를 통해 “발가락이 효율 격차를 얼마나 줄이는가”를 데이터로 답하는 데 초점을 맞춘다.

- **Technical Challenges**: 기술적 난관은 두 가지다: 첫째, CA(Cooperative Actuation)처럼 액추에이터 토크와 관절 토크가 1:1 매핑이 깨지는 구조를 시뮬레이션이 정확히 따라가야 한다. 둘째, 단순 관절 토크 제곱합이 아니라 실제 CoT(모터/관절 쪽 전력 소모를 Joule heating과 기계적 일로 분해) 관점에서 보행 효율을 학습에 반영해야 한다. 저자들은 관절-모터 토크를 투영/클리핑하고 마찰(쿨롱+점성)을 보정하며, CBF 스타일의 소프트 제약(열/토크 과다 유발 방지)을 추가해 비현실적인 자세로의 수렴을 억제했다.

- **Empirical Impact**: 검증은 시뮬레이션에서 1.33 m/s 직선 보행과 Robot T-Test 기반 민첩성 테스트로 이뤄졌다. 직선 보행에서 active toe는 toe-ablation 대비 CoT를 17.5% 줄이고 heel-strike GRF도 5.0% 낮췄으며, 총 파워는 16.9% 감소했다. 민첩성에서는 총 완료 시간은 유사(16.06 s vs 16.08 s)했지만 경로 이탈 평균/최대가 각각 25.0%/34.0% 개선돼 회전 시 정확도와 추종성이 향상됨을 보여준다. 저자들은 물리 하드웨어 검증과 다양한 지형에서의 일반화를 향후 과제로 남기며, sim-to-real 간극을 줄이는 “검증 가능한 설계-학습 프레임”을 제공했다는 의미를 강조한다.



### Route-Constrained Robust Fusion Estimation for MEMS/GNSS Integrated Navigation of Unmanned Ground Vehicles in GNSS Degraded Environments (https://arxiv.org/abs/2606.19687)
Comments:
          Accepted workshop paper, 1st Workshop on Robot Meets GNSS and Ranging for Seamless Autonomy, IEEE ICRA 2026

- **Prior Approaches**: 기존 GNSS 열화 대응은 (1) 비전·LiDAR-SLAM·UWB처럼 추가 센서를 붙여 관측성을 높이거나, (2) 도로 기하·차선 중심선·미션 경로 같은 환경 사전지식을 써서 맵 제약을 거는 방식으로 나뉜다. 하지만 UGV는 터널 등에서 GNSS가 끊기면 dead reckoning 기반 오차가 누적되며, 맵 제약도 일회성 보정에 그치면 연속 추정 안정성이 떨어질 수 있다. 또한 “미션 경로가 이미 주어졌을 때” dead reckoning 궤적을 어떻게 신뢰도 있게 필터에 넣을지에 대한 실전 수준 해법이 부족했다.

- **Core Contribution**: 이 논문은 GNSS가 없는 구간에서 단기 dead reckoning 궤적과 HD map의 미션 route 로컬 구간을 shape consistency로 매칭해, 이를 pseudo-position observation으로 만든 뒤 EKF 업데이트에 통합하는 route-constrained state estimation을 제안한다. 단순히 현재 위치를 경로로 기하학적으로 덮어쓰지 않고, 매칭 결과를 필터 측정으로 변환해 도로 레벨 제약을 연속적으로 주입한다. 결과적으로 위치뿐 아니라 상태 결합을 통해 azimuth(방위) 추정의 누적 편차도 간접 완화하는 것이 핵심이다.

- **Technical Challenges**: 실제 터널에서는 GNSS 플러터링(잠깐 신호 복귀), 매칭 기하가 애매한 상황, 평행/반대 방향 경로로의 오매칭 위험이 커서 잘못된 pseudo-position이 EKF를 흔들 수 있다. 이를 위해 (1) GNSS 연속 무신호 기간과 충분한 역사 궤적이 쌓인 경우에만 trigger control로 저빈도 업데이트를 수행하고, (2) 포인트 수·종단 heading 일치·rigid-registration residual로 품질을 검증하며, (3) azimuth-consistency로 후보를 거른 뒤, (4) route offset compensation과 단일 업데이트 correction limit로 과도한 강제 구속을 막는다. 또한 경로 제약은 수평 평면(East-North)만 반영해 세로 성분의 비자연스러운 보정을 피한다.

- **Empirical Impact**: 실험은 Jingli Expressway의 Jinjiazhuang extra-long spiral tunnel에서 long tunnel, multi-segment tunnel, curved tunnel 3개 시나리오로 수행됐고, 비교 대상은 GNSS 미가용 시 baseline(원 시스템 dead reckoning)과 제안 방법이다. GNSS가 꺼진 동안 baseline은 미션 경로에서 점진적으로 이탈하며 누적 드리프트가 커진 반면, 제안 방법은 경로 대비 위치 편차가 전 구간에서 더 낮고 변동도 작게 유지됐다. 특히 long tunnel에서 개선이 가장 커서 error accumulation 억제와 최대 편차 위험 감소가 두드러졌고, multi-segment/curved에서도 국소 peak를 줄이거나 전체 변동을 완화하는 일관된 경향이 관찰됐다. azimuth 지표 개선은 위치보다 제한적이었지만, 필터 내 상태 상관을 통해 간접적으로 방위 편차 안정화에 기여한 것으로 나타나 “일회성 기하 보정”이 아닌 “필터에 참여하는 pseudo-position 매칭”의 가치가 확인됐다.



### ForEnt: A Multi-Modal Dataset for Characterizing Quadruped Robot Entrapments in Forest Environments (https://arxiv.org/abs/2606.19675)
Comments:
          8 pages, 7 figures

- **Prior Approaches**: 기존 산림 로보틱스 데이터셋은 주로 biomass estimation, forest inventory 같은 생태 측정 목적에 맞춰 “정상 주행”이 끊기지 않도록 수집돼 왔습니다. 그 결과 vegetation-induced entrapment나 immobilization 같은 실패가 드물게 기록되거나 체계적으로 라벨링되지 않아, 자율주행의 취약 지점을 학술적으로 정량화하기 어렵습니다. 또한 시각/주 proprioceptive 기반 traversability나 entanglement detection 연구는 검증 데이터가 빈약해 실제 숲 환경의 장애 상황을 충분히 반영하지 못합니다.

- **Core Contribution**: 이 논문은 Unitree Go2(저비용 미드사이즈 사족 보행 로봇)를 8개 숲 현장에서 운용하며 포획한 실패 중심 멀티모달 데이터셋 ForEnt를 제안합니다. 총 1.7 km 주행과 69개의 entrapment 이벤트를 time-synchronized RGB-D, LiDAR, 고주파 proprioception, third-person video와 함께 제공하며, 4가지 실패 메커니즘과 3단계 severity로 라벨링합니다. 이를 통해 숲에서 실제로 자율이 무너지는 조건에서 탐지·회피·평가를 재현 가능하게 만드는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 기술적 난제는 숲 환경의 비정형 상호작용에서 entrapment “발생 구간”을 안정적으로 찾아 라벨 일관성을 확보하는 것입니다. 논문은 commanded/estimated 속도 오차가 일정 임계값을 넘고 3초 이상 지속되는 후보 윈도우를 반자동으로 추출한 뒤, 사람이 third-person video와 센서 로그로 오탐을 제거하고 실패 메커니즘·severity를 태깅합니다. 또한 멀티모달 정합을 위해 센서 캘리브레이션과 hardware timestamp 동기화를 갖추고, third-person 카메라는 sitting-to-standing 동작으로 시각 정렬을 보정했습니다.

- **Empirical Impact**: ForEnt로 기존 대표 파이프라인을 평가한 결과, 시각 traversability(Wild Visual Navigation)는 물에 잠긴 구역과 견고한 토양의 시각적 모호성 같은 케이스에서 성능이 급락했습니다. proprioceptive entanglement detection(Momentum-Based Observer)은 숲 환경 전체에서 true positive rate가 16.8% 수준으로 크게 떨어져, 관찰 가정(기저 동역학 안정성 등)이 현장에서 쉽게 깨짐을 보여줍니다. 즉, 실패 중심 데이터셋이 있어야 숲 배치에서의 취약점이 드러나며, 알고리즘을 실제 운영 안전성 쪽으로 밀어붙이는 벤치마크 역할을 한다는 점에서 의미가 큽니다.



### Safe Local Navigation for Ackermann-Steered Robots in Unmapped Environments (https://arxiv.org/abs/2606.19672)
Comments:
          Presented at the 23rd Conference on Robots and Vision (CRV 2026)

- **Prior Approaches**: 기존 로컬 내비게이션은 전역 지도/목표가 주어지는 상황에서 글로벌 기준 경로를 생성한 뒤, 장애물 회피를 로컬 플래너가 보정하는 경우가 많았다. 다만 무목표·미맵 환경에서는 explore 기반 exploration 패키지로 전방 프론티어를 목표로 삼고 DWA/TEB/MPC 같은 최적화·탐색 플래너가 경로를 계속 재계산해야 해 실시간성, 안전 여유 확보에서 흔들릴 수 있다. 특히 MPC는 최적화 품질에 따라 진동/불안정이 나타날 수 있고, TEB/DWA는 코너를 더 깎는 방향으로 진행해 장애물 근접 위험이 커질 수 있다는 문제가 지적된다.

- **Core Contribution**: 이 논문은 글로벌 목표가 없는 미맵 환경에서, 이동 로봇이 '가장 넓게 열린 공간'을 향하는 방향을 먼저 찾고 그 방향을 기준으로 로컬 장애물 경계(bounding line)와 기준 경로를 실시간 생성하는 제어 프레임워크를 제안한다. 핵심은 QP(Quadratic Program)로 좌/우 bounding line을 만들고, 그 라인으로부터의 거리(차폐 여유)를 극대화해 안전을 보장하는 로컬 reference path를 형성한 뒤, feedback linearizing PD 제어로 실제 추종을 수행한다. 또한 bounding line의 시간적 급변을 줄이기 위해 직전 제어 단계의 해와 매끄러움 제약(병렬성/스무딩 옵션)도 함께 다룬다.

- **Technical Challenges**: 문제의 기술적 난점은 (1) 비예측 환경에서 장애물 감지로부터 즉시 참조 경로를 만들어야 하고, (2) 차폐 여유를 최대화하는 최적화가 제어 주기 안에 끝나야 하며, (3) 경계선이 다음 시점으로 넘어가며 불연속적으로 튀지 않게 해야 한다는 점이다. 저자들은 전방 장애물의 각도 분포에서 largest open space를 선택(Follow the Gap Method 영감)한 뒤, 좌/우 클러스터를 분리하는 bounding line을 vehicle-to-obstacle clearance를 최대화하는 convex QP로 산출한다. 여기에 smoothing(시간상수로 직전 해로부터의 변화 완화) 또는 병렬 제약을 추가해 안정적인 reference line을 만들고, 그 후 feedback linearization으로 차량 kinematic bicycle 비선형성을 선형 2차계로 바꾼 다음 PD로 라인 추종을 구현한다.

- **Empirical Impact**: 실험은 f1tenth_simulator 시뮬레이션과 1/10 scale RC 차량(ROS Noetic, LiDAR 중심)을 통해 수행했으며, 제안 방식이 DWA/TEB/MPC exploration-based 플래너 대비 장애물 근접 지표에서 더 좋은 결과를 보였다. 경로는 트랙 중심에 더 가깝게 유지되며, 탐색 기반 플래너들은 코너를 더 파고들어 장애물 여유가 줄어드는 경향이 관찰됐다. 또한 계산 시간에서 제안 방법이 더 짧고(실시간 제어율 유지), open-source C++/ROS 구현까지 제공되어 미맵·무목표 로컬 내비게이션 연구 및 적용에 바로 활용될 수 있는 의미가 있다.



### DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning (https://arxiv.org/abs/2606.19656)
Comments:
          ICML 2026

- **Prior Approaches**: 로봇 정책 개선의 핵심은 self-collected online experience를 적은 샘플로 모으는 탐색이지만, 기존에는 pretrained generative control policy(예: diffusion policy)를 강화학습으로 fine-tuning할 때 온라인에서의 action 선택을 크게 개선하지 못했다. diffusion policy는 action-space에서 multimodal을 제공해 탐색 여지가 있으나, 연속 action space에서는 모든 후보를 일일이 평가하기 어렵고, 불확실성 기반 탐색도 discrete 환경 중심으로 설계된 경우가 많다. 또한 fleet 환경에서는 다중 에이전트가 병렬로 데이터는 늘리지만 서로 중복 탐색을 줄이는 협업 메커니즘이 제한적이었다.

- **Core Contribution**: 이 논문은 Diffusion Filtered Exploration via Ensembles(DF-ExpEnse)라는 온라인 탐색 기법을 제안해, pretrained diffusion policy를 fine-tuning할 때 온라인 경험의 질을 높여 샘플 효율을 개선한다. diffusion policy의 multimodal 생성 능력을 이용해 연속 action space를 “평가 가능한 후보 집합”으로 필터링하고, critic ensemble을 통해 각 후보의 품질과 불확실성을 함께 반영해 실행할 action을 고른다. 더 나아가 fleet에서는 에이전트 간 critic 통계를 공유해 탐색을 집단 맥락에서 정규화함으로써 협업 탐색을 가능하게 한다.

- **Technical Challenges**: 연속 action space에서 UCB류의 “품질-불확실성” 균형을 효율적으로 적용하려면, 모든 action을 전수평가하지 않으면서도 탐색 가치가 높은 후보를 찾아야 한다. DF-ExpEnse는 diffusion policy가 샘플링으로 암시하는 multimodal candidate를 제한된 크기로 생성해 tractable하게 평가하고, ensemble의 min-value(과대추정 완화)와 표준편차(critic disagreement)를 결합해 탐색 관심도 exploration interest를 계산한다. 또 finetuning이 진행되며 후보가 특정 모드로 수렴해 탐색이 약해질 수 있어, Behavior Cloning Sampling Regularization(BC-SR)로 초기 pretrained diffusion policy에서 후보를 섞어 multimodal priors를 유지한다. fleet 단계에서는 각 에이전트의 탐색 관심도를 fleet 전체 통계로 정규화해 중복 데이터 생성을 줄이도록 설계했다.

- **Empirical Impact**: 여러 manipulation( RoboMimic의 Lift/Can/Square/Tool Hang )과 locomotion( OpenAI Gym의 HalfCheetah-v2/Hopper-v2/Walker2D-v2 ), bimanual manipulation( DexMimicGen의 Can Sort/Box Cleanup/Coffee )에서 DF-ExpEnse는 기본 fine-tuning(예: vanilla DSRL)과 Max-Q 같은 대안 액션 선택 대비 일관된 sample-efficiency 향상을 보였다. 또한 DF-ExpEnse는 DSRL은 물론 ResFiT 같은 서로 다른 reinforcement learning fine-tuning 프레임에도 온라인 단계에서 쉽게 통합되어 효과가 유지됨을 실증했다. ablation을 통해 critic ensemble 크기, 샘플링된 action set 크기, fleet 크기 같은 설계 요소가 성능에 미치는 영향을 확인하며, 집단 협업 탐색이 단독 fleet 대비 추가 이점을 준다는 점도 보고한다.



### Scaling Self-Play for End-to-End Driving (https://arxiv.org/abs/2606.19641)
- **Prior Approaches**: 기존 end-to-end 자율주행 학습은 주로 오프라인 human demonstration 기반 behavior cloning(BC)에 의존해 왔습니다. 하지만 로그 데이터는 상태 커버리지가 제한적이고 학습 중 closed-loop 상호작용이 없어 배포 시 compounding error로 인해 취약해지기 쉽습니다. 이에 따라 시뮬레이터에서의 self-play/RL 또는 DAgger가 대안으로 연구됐지만, 대개 vectorized BEV 관측을 전제로 하거나 느린 시뮬레이터 때문에 대규모 학습 확장에 한계가 있었습니다.

- **Core Contribution**: 이 논문은 end-to-end 모델을 위해 pixels(센서 관측)에서 직접 self-play를 scalable하게 학습하는 전략을 제안합니다. 이를 위해 (1) 고처리량 batched 드라이빙 시뮬레이터 Gigapixel, (2) direct pixel-space self-play RL의 비효율을 줄이기 위한 self-play DAgger(privileged RL teacher로 on-policy distillation), (3) sim-to-real 격차를 perception만 가볍게 적응해 폐루프 주행 능력을 실세계에 옮기는 방식을 묶었습니다. 목표는 human trajectory supervision 없이도 closed-loop에서 강건한 주행 정책을 만드는 것입니다.

- **Technical Challenges**: 핵심 기술 난제는 두 가지입니다: (a) end-to-end 모델이 pixels를 처리하면 self-play RL에서 policy forward/backward 비용이 커져 샘플 효율이 급격히 악화된다는 점, (b) 기존 시뮬레이터가 end-to-end에 맞지 않는 vectorized BEV 관측만 제공하거나 처리량이 낮다는 점입니다. 논문은 RL을 pixel student에 직접 쓰지 않고, vectorized 관측의 가벼운 RL teacher가 생성한 궤적을 self-play DAgger로 distillation하여 계산·샘플 비용을 줄였습니다. 또한 Gigapixel은 photorealistic 시뮬레이션 대신 바운딩박스 기반의 bounding-box world를 GPU-가속 perspective rendering으로 변환해 필수 장면 구조는 유지하면서 50k agent steps per second 처리량을 달성하고, 마지막으로 perception adaptation만으로 sim-to-real을 매끈하게 연결합니다.

- **Empirical Impact**: 실험에서 Gigapixel self-play로 학습한 정책은 HUGSIM(closed-loop)에서 경쟁/상위권 성능을 보이며, NAVSIM-v2(pseudo-closed-loop)에서도 human trajectory supervision 없이 competitive한 결과를 보입니다. 특히 self-play 학습 스케일을 키울수록 정책 성능이 일관되게 비례적으로 향상되어, self-play가 end-to-end 학습에서 실용적이고 확장 가능한 데이터 생성 파이프라인임을 보여줍니다. 결과적으로 offline BC의 구조적 한계를 closed-loop 경험 생성과 distillation로 완화하면서, 향후 합성 경험을 통한 지속 개선 경로를 제시한다는 점에서 의미가 큽니다.



### CTS-MoE: Implicit Terrain Adaptation via Mixture-of-Experts for Perceptive Locomotion (https://arxiv.org/abs/2606.19633)
- **Prior Approaches**: 기존 다지형(계단, 갭, 장애물) 보행 제어는 단일 보상으로 여러 상황을 커버하거나, 표현 학습·일반화에 집중하는 방식이 많았습니다. 또한 비대칭 teacher–student나 비전/맵 기반 지형 인식 접근은 종종 지나치게 보수적으로 움직이며, 계단/갭처럼 급격한 위상 변화에 필요한 선제적(anticipatory) 특화 동작을 충분히 만들지 못했습니다. 반면 계층형(서브폴리시+선택기) 방식은 학습 분리로 인해 전환의 일관성이 깨지고, unseen 지형에 대한 일반화도 제한될 수 있습니다.

- **Core Contribution**: 이 논문은 CTS-MoE(Concurrent Teacher-Student with Mixture of Experts)를 제안해, 다중 태스크 보상에서 발생하는 ‘공유-분리’ 긴장을 동시에 다룹니다. 핵심은 dense MoE actor로 공통 보행 기반 위에서 전문가(expert) 조합을 만들고, sparse multi-critic의 task-specific value head로 보상 간 value interference를 억제하는 구조입니다. 또한 배포 시에는 perception 기반 routing만으로 적응이 일어나고, 별도의 high-level selector나 지형 분류기는 필요 없습니다.

- **Technical Challenges**: 기여의 실현에서 가장 큰 기술적 난관은 (1) 서로 충돌하는 여러 보상으로 인해 value 학습이 붕괴하거나(gradient/value 간섭) (2) 부분 관측(POMDP) 환경에서 태스크에 맞는 전문가를 안정적으로 선택해야 한다는 점입니다. 저자들은 concurrent teacher–student를 단일-stage end-to-end로 학습해 지각/표현 문제를 완화하고, PPO에 multi-reward–multi-critic을 결합하며 POPArt 기반 per-task return normalization으로 태스크 간 스케일 차이도 안정화합니다. 더불어 actor의 routing은 task labels 없이도 지형 정보를 반영하도록 설계되어, 훈련 동안에만 task ID를 쓰고 배포에서는 제거됩니다.

- **Empirical Impact**: Unitree Go1에서 시뮬레이션(IsaacLab)과 실로봇 실험 모두를 통해 CTS-MoE가 단일(모놀리식)·기존 perceptive baseline보다 높은 성공률과 더 낮은 속도 추적 오차를 보였다고 보고합니다. 특히 gap에서 성공률이 +29.3%p, climb-up에서 +10.3%p 개선되는 등 선제적 동작이 필요한 지형에서 효과가 두드러졌습니다. 또한 unseen terrain에서도 성능 이점이 유지되어, 전문가 조합이 명시적 태스크 인식 없이도 지각 기반으로 자연스럽게 적응한다는 점을 실증합니다.



### Formal Verification of Learned Multi-Agent Communication Policies via Decision Tree Distillation (https://arxiv.org/abs/2606.19632)
Comments:
          9 pages, 3 figures, 7 tables. Accepted at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026), Pittsburgh, Pennsylvania, USA, September 27-October 1, 2026

- **Prior Approaches**: 기존 MARL(MAPPO, QMIX 등)은 다중 에이전트 간 협력을 emergent communication으로 학습하지만, 실제 드론·자율주행 같은 안전 비즈니스에는 formal safety guarantee가 부족합니다. 또한 단일 에이전트 신경망 검증 도구(Relux/DeepPoly 등)는 multi-agent의 상태 공간 폭발과 통신을 직접 다루기 어렵고, 정책 추출·증류(VIPER/MAVIPER류)는 통신 의미를 포함한 end-to-end 검증 파이프라인이 제한적이었습니다.

- **Core Contribution**: 이 논문은 learned multi-agent communication policy를 대상으로 end-to-end 안전 검증을 제공하는 새로운 프레임워크를 제시합니다. 핵심은 신경 정책을 communication semantics를 반영해 decision tree로 policy abstraction(증류)한 뒤, PRISM 기반 확률 모델체킹으로 PCTL 성질을 formally verify하고, verified safety가 원래 네트워크로 Monte Carlo에서 전이되는지를 함께 검증한 점입니다. 

- **Technical Challenges**: 문제는 (1) 신경 정책이 NP-hard 검증 대상이고 (2) multi-agent·통신으로 상태 공간이 기하급수적으로 커진다는 점입니다. 이를 위해 관측에서 50개 도메인 피처를 설계해 증류 정확도를 높이고, CART 기반 decision tree를 PRISM DTMC로 자동 변환할 때 feature-to-state 변수 대응을 “정수 산술로 완전 대응”하도록 강제했으며, PCTL 검증은 pairwise decomposition과 union-bound aggregation으로 조합 검증을 수행했습니다. 비결정성은 초기 상태 난수와 leaf impurity를 모델링해 확률 추정의 일관성을 확보하고, 비핵심 에이전트는 calibration rollout으로 이웃 기반 전이 커널을 실증 추정했습니다.

- **Empirical Impact**: VQ-VIB(이산 메시지) 정책(에이전트 5~7명)에서 decision tree distillation fidelity가 97.9%±1.2%에 도달했고, PRISM에서 안전/생존/협력 범주의 PCTL 속성 18개를 검증해 평균 88.9% satisfaction을 보고했습니다. 특히 충돌 확률은 0.3%로 안전 임계(1%)를 만족했으며, 원래 신경망에 대한 Monte Carlo 검증에서 verified 성질이 <=0.6 percentage-point 편차로 전이됨을 확인했습니다. 이산 VQ-VIB 메시지는 연속 통신 대비 11.6~13.6pp의 fidelity 우위를 제공해 검증을 3~4배 빠르게 만드는 실용적 다리로 평가됩니다.



### Fail-RAG : A Retrieval Augmented Generation Informed Framework for Robot Failure Identification (https://arxiv.org/abs/2606.19598)
- **Prior Approaches**: 로보틱스의 실패·이상 탐지는 기존에 룰 기반과 기준선 대비 비주얼 차이, 또는 out-of-distribution 탐지로 접근해 왔습니다. 그러나 작업/환경이 달라지면 실패 형태가 계속 바뀌어 룰과 기준선이 쉽게 깨지고, 데이터 수집·분류가 커질수록 확장도 어려워졌습니다. 또 LLM/VLM을 fine-tuning해 실패를 분류하거나 복구 액션까지 내리는 연구가 늘었지만, 실패 데이터와 대규모 학습 비용이 부담이 됐습니다.

- **Core Contribution**: 이 논문은 창고 물류처럼 예기치 못한 사건(실패)이 발생할 때, 카메라 비전만으로 실패를 탐지하는 RAG 기반 프레임워크 Fail-RAG를 제안합니다. 핵심은 실패 이미지(시계열을 한 장으로 요약)와 문맥 정보를 CLIP 임베딩으로 저장하고, 유사도를 계산해 관련 문서를 검색한 뒤 VLM이 템플릿(JSON/YAML) 형식으로 실패 유형과 원인까지 설명하게 하는 것입니다. 이를 통해 새로운 엣지 케이스가 생겨도 재학습(fine-tuning) 없이 참고 문서 확장으로 대응하는 방향을 제시합니다.

- **Technical Challenges**: 기존 VLM을 그대로 쓰면 시각적 유사성만으로 실패 유형까지 안정적으로 맞히기 어렵고, 룰 기반처럼 고정된 판단 기준도 통하지 않습니다. Fail-RAG는 (1) CLIP을 이용한 멀티모달 임베딩 검색과 (2) RAG 거리함수(코사인/유사도 등) 및 샘플링 주기(프레임 rate) 선택이 성능에 미치는 영향을 체계적으로 분석해 최적 조합을 찾았습니다. 또한 VLM 출력이 디버깅에 유용하도록 ‘normal/anomalous/unknown’ 같은 제한된 결정지와 서브 실패·원인 텍스트를 템플릿으로 강제해 추론 변동성을 줄였습니다.

- **Empirical Impact**: 실험은 시뮬레이션(NVIDIA Isaac Sim)과 물리 로봇(고정형 UR5, 이동 매니퓰레이터, UR5·Fanuc M-20iB 등)에서 팔레타이징, 디팔레타이징, 조립, 운반/이동 등 5개 작업·여러 실패 유형을 대상으로 수행했습니다. 그 결과 Fail-RAG는 off-the-shelf VLM 대비 평균적으로 실패 탐지 정확도를 25%p 더 높였고, 일부 설정에서는 40%p까지 개선됐습니다. 특히 물리 환경의 조명/그림자 영향이 큰 경우 RAG 기반 참고 문서가 성능을 끌어올리는 경향이 관찰되어, 실제 현장 모니터링에 대한 실용성을 강조합니다.



### Safe, Real-Time Active Model Discrimination and Fault Diagnosis for Nonlinear Systems via Differentiable Reachability (https://arxiv.org/abs/2606.19590)
- **Prior Approaches**: 기존의 fault detection/diagnosis는 Bayesian 같은 확률적 접근이 많았지만, 노이즈·불확실성 하에서의 안전성과 “식별 가능성”을 형식적으로 보장하기 어렵습니다. set-based 접근도 있었으나, 선형/affine 중심이거나 reachability 계산이 비싸 실시간 active fault diagnosis로 확장하기 힘들었습니다.

- **Core Contribution**: 이 논문은 불확실한 연속시간 nonlinear 시스템에서 공정 잡음(process disturbance)과 측정 잡음(measurement disturbance)을 함께 두고, 여러 후보 모델(정상/고장 모드, actuator·sensor fault 포함)을 구분하는 safe·real-time 알고리즘을 제안합니다. 출력 샘플을 이용해 “최대 하나의 모델과 일치”하도록 만들며, 안전 제약은 견고하게(robustly) 만족시키는 time-varying output-feedback 정책 최적화를 정식화합니다.

- **Technical Challenges**: 핵심 난제는 (1) output-feedback 하에서 비선형 시스템의 reachable set을 정확히 계산해 안전을 보장하는 것과 (2) 서로 다른 모델들의 예측 출력이 겹치지 않게 하는 diagnosability를 미분 가능한 형태로 인코딩하는 것입니다. 논문은 interval over-approximation 기반 reachable set을 immrax를 통해 JAX/GPU에서 빠르고 differentiable하게 계산하고, 모델별 출력 reachable set의 overlap을 줄이는 분리( separation ) 목적함수로 진단 조건을 최적화에 연결합니다.

- **Empirical Impact**: 시뮬레이션과 실로봇 실험에서 sensor/actuator fault를 최대 11개 모드까지 다루며, 모델 discriminaton을 50 ms 미만으로 수행하면서 속도와 성공률에서 baseline 대비 우수함을 보였습니다. 또한 state–input 안전 제약에 대한 형식적 안전 보장(safety guarantees)을 함께 제공해, 고차원 로보틱스에서 active fault diagnosis의 실시간 적용 가능성을 크게 높인 것으로 평가됩니다.



### One Demo is Worth a Thousand Trajectories: Action-View Augmentation for Visuomotor Policies (https://arxiv.org/abs/2606.19586)
Comments:
          Project website: this https URL. Published at CoRL 2025

- **Prior Approaches**: 기존 imitation learning 기반 visuomotor policy는 초기 로봇 자세나 장애물 변화처럼 작은 조건 차이에서 out-of-distribution(OOD) 관측으로 이어져 실행이 연쇄 실패하는 취약성이 컸습니다. 데이터 증강을 쓰더라도 비전만 바꾸거나(appearance/배경/경미한 시점) 또는 state 기반처럼 시각 불변성을 추정/모델 의존으로 넘기는 방식이 많아, 관측-행동의 일관성과 물리 제약을 동시에 만족시키기 어려웠습니다. 일부 visual-action 증강은 pinhole 카메라/단일 스텝 중심이라, wide FoV fisheye와 궤적 단위의 충돌 회피 데이터 생성까지 확장되기 어려웠습니다.

- **Core Contribution**: 이 논문은 eye-in-hand fisheye 카메라와 실제 시연 1쌍만으로, visually realistic한 fisheye 이미지 시퀀스와 물리적으로 feasible한 action 궤적을 함께 생성하는 오프라인 데이터 증강 프레임워크를 제안합니다. 핵심은 3D Gaussian Splatting(3DGS)을 wide FoV fisheye에 맞게 확장하고, 장면 편집(새 장애물 삽입) 후에도 관측-행동이 일치하는 시연 에피소드를 생성해, 정책이 새로운 초기 조건과 unseen 장애물에서 안전하게 행동하도록 만드는 것입니다. 결과적으로 시각적 다양성과 충돌 회피 행동을 “궤적 수준”에서 데이터로 공급해 OOD로 인한 catastrophic failure를 줄이는 방향을 택합니다.

- **Technical Challenges**: 증강에서 가장 큰 어려움은 (1) 관측과 행동의 action-view consistency를 유지하면서 시각 다양성을 늘리는 동시에, (2) 3D 충돌 제약과 접촉 역학을 위반하지 않는 물리 제약을 만족시키는 것입니다. 논문은 장면을 fisheye 시퀀스로 재구성한 뒤, trajectory optimization으로 smooth하고 collision-free인 궤적을 만들고 3DGS 렌더링으로 대응 관측을 다시 합성하며, 접촉 이벤트 이전 구간만 증강해 접촉 다이내믹스를 보존합니다. 또한 fisheye 렌더링 아티팩트를 피하기 위해 view-rendering-friendly한 자세 분포 제약(원 시연/스캔 뷰 근처로 유도)과 TSDF 기반 충돌 손실, fisheye ray sampler까지 통합해 wide FoV 카메라에서도 빠른 편집성과 렌더링 품질을 함께 확보합니다.

- **Empirical Impact**: 시뮬레이션에서는 RoboMimic의 square 태스크에서 free-space 증강이 OOD 초기 자세와의 격차를 줄이며 성공률을 유의미하게 끌어올렸고, oracle(완벽한 GT 렌더링) 대비 성능 격차도 낮게 유지되며 샘플 효율성이 개선됨을 보여줍니다. 현실 환경 cup serving 태스크에서도 obstacle-free와 unseen 장애물이 함께 주어질 때, obstacle-aware 증강(장애물 추가 후 회피 궤적 학습)이 FreeSpace Aug/No Aug를 크게 상회해 장애물 회피까지 실제로 학습되는 것이 확인됩니다. 특히 Obstacle Aug는 높은 성공률(논문 내 보고 기준 100%)로, 원 시연에 없던 회피 행동을 정책이 안정적으로 수행하도록 만드는 것이 의미 있는 임팩트로 제시됩니다.



### pdSTL: Probabilistic Differentiable Signal Temporal Logic for Stochastic Systems (https://arxiv.org/abs/2606.19561)
- **Prior Approaches**: 기존 STL은 로봇이 만족해야 할 시간적 사양을 정량적 robustness로 평가하지만, 기본 가정은 결정론적 신호라서 실제 센서/동역학 불확실성을 제대로 반영하지 못합니다. 확률적 STL(예: Probabilistic Signal Temporal Logic 계열)은 불확실성을 다루지만 대체로 비미분적이거나 belief-space 불확실성의 처리가 제한적입니다. 한편 STLCG류의 differentiable STL은 gradient-based 최적화가 가능하지만, 결국 확률적 의미를 스칼라 trace 수준으로만 처리해 신뢰/보장 관점이 약해집니다.

- **Core Contribution**: 이 논문은 probabilistic differentiable Signal Temporal Logic(pdSTL)로, STL의 확률 의미와 미분가능한 robustness 계산을 belief trajectories로 통합합니다. pdSTL은 interval-valued 확률론적 semantics를 사용해 보수적 만족 확률 하한/상한을 계산하고, 이를 STL 문법 트리 전체에 조성(compositional)으로 전파합니다. 그 결과 gradient 기반 end-to-end trajectory optimization에서도 “확률적 안전 보장”을 함께 최적화할 수 있습니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 확률 불확실성을 포함한 temporal operator 계산이 본질적으로 min/max/비미분 요소를 포함한다는 점과 (2) 시간 구간을 아우르는 STL 연산이 미래 정보를 필요로 해 계산 복잡도가 커질 수 있다는 점입니다. pdSTL은 STL 평가를 LSTM-style recurrent unfolding으로 재구성해, smooth min/max 근사(log-sum-exponential)와 함께 하한/상한 interval을 shift-register로 유지하며 backward recurrent 방식으로 linear-time에 가깝게 모니터링합니다. 또한 Gaussian 등 belief에 대해 원자(predicate) 만족 확률을 닫힌형 또는 보수적 결합(Fréchet bounds)으로 계산해 미분 그래디언트를 belief(평균/공분산) 및 제어입력까지 전달합니다.

- **Empirical Impact**: pdSTL은 모의 obstacle avoidance, lane-change, 그리고 실제 Crazyflie 쿼드콥터 실험에서 deterministic differentiable STL 대비 안전 여유를 더 잘 유지하며 최적화를 효율적으로 수행함을 보였습니다. 특히 장애물 회피에서는 확률적 만족 하한이 더 높게 유지되어 실험 성공/안전률이 개선됐고, 계산은 계획 horizon에 대해 선형적으로 증가하는 경향을 보였습니다. 쿼드콥터의 공력 교란 실험에서도 fan 세기가 커질수록 결정론적 접근이 빠르게 열화하는 반면, pdSTL은 더 적은 안전 위반으로 성능을 유지하며 확률적 보장과 실제 결과의 정합성을 강조합니다.



### SCAN-Planner: Spatial Collision-Aware Local Planning for Route-Guided Long-Range Quadruped Navigation (https://arxiv.org/abs/2606.19555)
- **Prior Approaches**: 기존 로컬 플래너는 로봇을 점이나 구(원형)로 단순화해 기하학적 인플레이션으로 충돌을 판단하는 경우가 많아, 회전에 따른 사체(몸체) 방향성 변화를 반영하지 못합니다. 또한 2.5D 고도 맵은 오버행 구조(테이블 아래 공간 등)를 제대로 표현하지 못하고, 3D 기반 접근은 수직 방향 변형이 과도해져 지상 보행 제약과 충돌할 수 있습니다. 장거리에서는 로컬 맵 경계에서 정보가 끊기거나 데드엔드가 생겨 회복 전략이 일관되지 않아 실패가 누적되기도 합니다.

- **Core Contribution**: SCAN-Planner는 회전에 따라 바뀌는 사체 풋프린트를 yaw-aware 방식으로 모델링하고, 긴 범위 사전(전역) 가이드 아래에서 안전한 로컬 경로를 생성하는 프레임워크를 제안합니다. 구체적으로 yaw에 연동된 twin-cylinder 풋프린트를 이용해 sparse 쿼리만으로 3D 인플레이션 점유맵에서 whole-body 충돌 평가를 수행합니다. 이어서 projected A*와 z-gradient suppression을 결합해 장애물은 주로 수평으로 회피하면서도 계단/경사에서의 수직 안정성을 유지하도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 좁은 공간에서 yaw 의존적인 몸체 충돌을 실시간에 가깝게 정밀 평가하면서도 계산량을 줄이는 것과 (2) 3D 오브젝트가 많은 환경에서 불필요한 수직 우회 없이 ground-following 성질을 유지하는 것입니다. 논문은 두 개의 세로 원통(트윈-실린더)을 바디 프레임에 고정해 변환 후 두 점 질의로 충돌을 판정하고, 최적화에서는 projected A*로 충돌 구간을 lazy하게 재가이드한 뒤 z 업데이트를 억제해 수평 변형 중심의 rebound-guided B-spline 궤적을 만들었습니다. 장거리 데드엔드 대응을 위해 로봇 중심 sliding map 바깥에 가상 자유층을 임시로 확장하는 boundary fallback을 도입해 경계 쪽으로 복구하도록 했습니다.

- **Empirical Impact**: 시뮬레이션에서는 MARSIM 기반으로 무작위 밀집 장애물, 오버행이 있는 책상 장면, 계단, 장거리 항법을 포함해 여러 기준(EGO-Planner, CMU-Planner, ART-Planner) 대비 SR/CR/SPL에서 일관된 개선을 보였습니다. 특히 좁은 통로에서는 yaw-의존 풋프린트가 인플레이션 보수성을 줄여 더 많은 통과 가능 경로를 제공했고, 오버행 구조에서는 2.5D가 놓치는 하부 통과 가능성을 3D 점유와 충돌 체크로 직접 반영했습니다. 실세계에서는 Unitree Go2에 탑재해 Jetson Orin NX에서 실시간 주행을 시연했으며, LiDAR/FAST-LIO2 기반 매핑 후 동적 방해물 회피 및 큰 규모 장거리(다층 건물 점검 등) 확장성도 실험적으로 확인했다는 점에서 의미가 큽니다.



### A Categorial and Sheaf-Theoretic Semantics for Autonomic Component Ensembles (https://arxiv.org/abs/2606.19525)
- **Prior Approaches**: 대규모 분산 자율 에이전트(로봇 스웜, 네트워크형 cyber-physical system)는 개방적·비결정적 환경에서 자기구성/자기치유 같은 self-* 성질과 출현적( emergent ) 집단 거동을 요구해, 기존의 전통적 형식 검증이 전면적으로 대응하기 어렵다는 문제의식이 제시된다. De Nicola 등이 만든 SCEL은 Autonomic Component(AC)와 동적 Autonomic-Component Ensemble(ACE)을 통해 분산 구성을 정밀히 모델링하지만, Labelled Transition System 기반의 operational semantics는 전체의 구조·전역 성질을 추론하기엔 추상화 수준이 낮다. 즉 “집합의 전역 일관성/태스크 가능성” 같은 질문을 다루기 위해서는 더 적절한 denotational semantics가 필요하다는 한계가 드러난다.

- **Core Contribution**: 이 논문은 SCEL의 다층 의미론을 category theory와 sheaf theory로 재해석해, 분산 시스템의 전역 구조를 수학적 대상의 기하로 환원하려는 새로운 형식 틀을 제안한다. 특히 SCEL에서 로봇 사회는 위상공간의 sheaf로 모델링되며, components는 점(point), ensembles는 열린집합(open set), 분산 지식은 sheaf의 데이터로 대응된다. 그 결과 정보 공유 같은 계산 과정은 sheaf-theoretic “gluing(붙이기)” 연산과 동치가 되어, 전역 일관성/출현적 거동을 한 번에 다룰 수 있는 관점을 제공한다.

- **Technical Challenges**: 핵심 기술적 난제는 SCEL의 ACE가 고정 그래프가 아니라 predicate로 정해지는 동적·일시적 집합이라, “지역(local) 구상”을 전역(global)으로 어떻게 합성해 의미를 부여할지였다. 논문은 ensemble을 index category 위의 diagram으로 스냅샷화하고(구성원 변화에 따라 diagram도 변형), 상호작용을 풍부하게 만든 뒤 colimit로 전역적 출현 정체성을 모델링한다. 또한 knowledge의 전개에서는 presheaf→sheaf로의 강화( sheaf axiom의 gluing 조건 )를 통해, 국소적으로 얻은 지식 조각들이 일관성을 만족할 때만 전역 section이 유일하게 존재함을 형식화한다.

- **Empirical Impact**: 제공된 내용 기준으로 이 논문은 실험 성능보다는, 복잡한 분산 자율 시스템의 검증을 “수학적 물체의 기하(위상적 방해)” 분석 문제로 바꾸는 구조적 통찰에 초점을 둔다. 특히 시스템 failure(예: 합의 실패나 태스크 해결 불가)는 topological obstruction으로 해석되고, sheaf cohomology 같은 대수위상 도구로 정량화할 수 있음을 이론적으로 제안해 검증 프레임의 방향성을 제시한다. 로봇 사회 설계에서 robustness를 목표로 할 때, 전역 일관성·문맥 의존성을 직접 겨냥하는 새로운 denotational semantics의 기준점을 제공한다.



### Proprioceptive Invariant State Estimation for Humanoid Robots on Non-Inertial Ground (https://arxiv.org/abs/2606.19512)
- **Prior Approaches**: 기존 추정기는 EKF나 최적화 기반 프레임워크로 고유(프로프리오셉티브)·외부(엑스로셉티브) 센서를 결합해 상태를 추정하지만, 대개 지면이 정지라는 가정(stance-foot 속도 0)을 전제로 한다. 그 결과 비관성 지면(움직이는 트레인·선박·항공기 등)에서는 지면 유도 가속과 회전이 모델·관측 구조를 바꿔 추정 정확도가 급격히 저하된다. 일부 invariant EKF(InEKF) 변형은 수렴 성질을 개선했지만, 여전히 정적 지면 가정에 묶이거나 지면의 절대 자세/속도, 혹은 지면에 부착된 외부 IMU 같은 실질적으로 어려운 정보를 필요로 한다.

- **Core Contribution**: 이 논문은 휴머노이드가 비관성 지면에서 동작할 때, 외부 지면 측정 없이도 자체 센서(관절 엔코더와 몸통·발 IMU)만으로 ‘이동하는 지면 프레임에 대한’ 베이스 위치·속도를 실시간 추정하는 InEKF를 제안한다. 스탠스 발의 운동학적 제약을 발 IMU로 활용해, 지면 운동이 만드는 비선형성을 과정/관측 모델에 명시적으로 반영하면서도 완전 고유센서 기반으로 유지한다. 또한 right-invariant 관측 모델(right-invariant observation condition)을 만족하도록 필터를 구성해 초기 불확실성이 큰 경우에도 유리한 오차 동역학을 보이도록 설계한다.

- **Technical Challenges**: 핵심 난제는 (1) 지면이 회전·이동하므로 inertial world 대비 속도/가속 정의가 꼬이고, (2) 기존처럼 ground motion을 직접 측정하지 못할 때 이를 모델에서 어떻게 흡수하느냐이다. 논문은 foot-ground 스탠스 조건에서 유도되는 제약을 통해 ground-frame의 필요한 관성 항을 ‘발 IMU 측정’으로 재표현해, 지면에 달린 IMU 없이도 과정 모델을 닫힌 형태로 만들었다. 더불어 Lie group 기반 InEKF의 right-invariant 오차를 사용하고, 관측가능성 조건을 분석해 이동 지면 프레임 기준 베이스 위치·속도를 관측 가능한 경우를 제시하며, 추가 센싱 구성을 통해 베이스 방향 관측성도 복원 가능함을 보인다.

- **Empirical Impact**: Digit 휴머노이드에서 흔들리고 pitching하는 지면 위로 서기/스쿼트 실험을 수행한 결과, 기존 InEKF 대비 수렴 속도는 96% 빨라지고 위치 추정 오차는 80% 감소했다. 또한 일축 회전하는 지면에서 워킹 실험에서는 초기 오차 최대 1m 조건에서도 평균 추정 오차가 9 cm 미만으로 보고되어 비관성 환경에서의 강인성을 입증한다. 정적 지면 가정이나 외부 지면 센서 의존성을 줄인 만큼, 이동 플랫폼(열차·선박·비행체)이나 정형화되지 않은 지면에서의 모델 기반 계획·제어에 직접 활용될 수 있는 실증적 의미가 크다.



### Simulating Robotic Locomotion in Sand: Resistive Force Theory in an Open-Source Physics Engin (https://arxiv.org/abs/2606.19504)
Comments:
          12 pages, 7 figures

- **Prior Approaches**: 기존 잔단(모래) 로보틱스 시뮬레이션은 주로 선형 강성이나 간단한 마찰/접촉 모델에 의존해 모래에 스며드는(sinkage) 비선형 거동을 제대로 재현하기 어려웠습니다. 가장 정확한 DEM(Discrete Element Method)은 개별 입자를 계산하지만 비용이 커서 실시간에 가까운 반복 시뮬레이션이나 end-to-end 학습 같은 빠른 반복 실험에 제약이 큽니다. RFT(Resistive Force Theory)는 계산이 가볍다는 장점이 있으나, 3D 확장을 다관절 로봇의 동역학 시뮬레이터에 “직접” 통합한 오픈소스 구현이 부족했습니다.

- **Core Contribution**: 논문은 3D RFT를 MuJoCo에 통합해, 모래 속을 걷는 로봇의 동역학(힘 근사→자세/속도/스며듦)을 안정적으로 시뮬레이션할 수 있는 오픈소스 도구 RFT-SiM(Resistive Force Theory — Sand in MuJoCo)을 제안합니다. RFT-SiM은 기존 MuJoCo의 접촉력 계산을 대체하고, 다관절 다리에서 stance 동안 발(엔드이펙터) 속도가 0에 가까워지는 문제를 완화하기 위한 smoothing을 설계합니다. 결과적으로 12-DOF 헥사포드 로봇의 보행 거리와 foot sinkage를 실험 대비 20% 이내 오차로 예측하는 것을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 난관은 (1) RFT가 속도 방향에 민감해 정지에 가까울 때 힘 방향이 불안정해질 수 있다는 점, (2) MuJoCo의 기본 접촉 대신 3D 기하를 plate 요소로 분해해 RFT 힘을 합산해야 한다는 점, (3) 이산화/적용 위치(사이트) 정렬과 스파이크 완화가 필요하다는 점입니다. 논문은 대상 바디를 충분히 촘촘한 mesh plate로 표현하고(plate 기반 힘 중첩), RFT에서 요구하는 velocity direction을 얻기 위해 MuJoCo 상태를 바탕으로 변수를 구성하며, exponential moving average smoothing(계수 0.1 수준)을 적용해 저속 구간의 불연속을 줄였습니다. 또한 leading edge hypothesis로 “모래를 밀지 않는 요소”의 힘을 제거해 불필요한 힘 기여를 억제하고, PI 제어 기반으로 다리 관절 구동 시에도 안정적인 동역학을 얻도록 구성했습니다.

- **Empirical Impact**: 검증 단계에서 저자들은 회전 침투, 레일 위 이송, 3D 경로를 따르는 관절 다리 등 여러 단계의 물리 실험을 통해 RFT-SiM이 깊이 증가에 따른 토크 변화 및 3D 기하 통합을 포착함을 확인합니다. 모래의 RFT coefficient(ζ) 보정 후에도 서로 다른 모양(직사각/ C-형)과 속도 조건에서 실험이 보인 핵심 경향이 시뮬레이션에서 유지되었고, 특히 모션 예측에서의 오차가 크게 줄어드는 방향을 보였습니다. 마지막으로 자유 보행 12-DOF 로봇에서 보행 거리와 발 침하를 실험 대비 20% 이내로 맞추며, DEM급 정확도를 일부 포기하더라도 설계·학습·반복 개발을 빠르게 돕는 “샌드박스” 역할 가능성을 실증했습니다.



### Playful Agentic Robot Learning (https://arxiv.org/abs/2606.19419)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존 Code-as-Policy(CaP) 로봇은 언어/멀티모달이 코드를 작성·실행하고, 피드백을 받아 반복 수정하는 에이전틱 구조를 갖추고 있지만 주로 ‘지시된 과제’에 반응해 학습합니다. 또한 재사용 가능한 스킬을 저장하더라도 그 획득이 외부 과제 발생 이후에 시작되는 경우가 많아, 선제적이고 지속적인 스킬 누적이 제한됩니다.

- **Core Contribution**: 이 논문은 Playful Agentic Robot Learning으로, 하류(downstream) 과제가 들어오기 전 로봇이 ‘자기 주도 놀이’로 스킬을 선학습하고 이후 과제를 푸는 설정을 제안합니다. 이를 위해 RATs(Robotics Agent Teams)를 도입해 놀이 중에 탐색 과제 생성-코드 실행-진행 검증-실패 진단-재시도-성공 실행의 코드 스킬 라이브러리 증류까지 하나의 루프로 구성합니다. 테스트 시에는 이 라이브러리를 고정(frozen)해 새 과제에 관련 스킬을 가져와 성능을 높입니다.

- **Technical Challenges**: 놀이가 유용한 스킬 학습으로 이어지려면 과제 수준 성공/실패만으로는 부족해, 어떤 하위 단계가 막혔는지와 저장할 동작 단위를 촘촘히 식별해야 합니다. RATs는 계획/검증/진단 에이전트를 다단으로 배치해 step-level 피드백을 밀도로 제공하고, 실패 유형을 메모리에 축적한 뒤 재시도로 국소 병목을 보완(서브 에이전트 분리 연습 포함)합니다. 또한 ‘너무 쉽지도 너무 어렵지도 않은’ 과제를 고르기 위해 객체-스킬 조합의 novelty와 Wilson-bounded success 기반 learnability frontier를 곱하는 Goldilocks 규칙으로 놀이 과제를 선택합니다.

- **Empirical Impact**: 실험에서 RATs는 LIBERO-PRO와 MolmoSpaces 모두에서 CaP-Agent0 및 VLA 기반 대조군 대비 held-out downstream 성능을 크게 개선합니다(각각 +20.6, +17.0 percentage-point). 특히 놀이로 얻은 스킬이 환경 밖으로도 전이되며, LIBERO-PRO에서 학습한 스킬 라이브러리를 RoboSuite에 plug-in 했을 때 CaP-Agent0 대비 +8.9점을 기록합니다. 더 나아가 모델 fine-tuning 없이 실제 로봇 과제에도 스킬을 직접 재사용해 +8.8점을 보고해, ‘놀이 기반 코드 스킬 라이브러리’가 에이전틱 로봇 성능을 finetuning 없이 끌어올리는 실용적 경로임을 시사합니다.



### DiffusionVS: A Generative Framework for Robust Visual Servoing Based on Diffusion Policy (https://arxiv.org/abs/2606.19397)
Comments:
          8 pages, 4 figures, 7 tables

- **Prior Approaches**: 시각 서보잉은 PBVS/IBVS로 나뉘며, 고전적으로는 폐루프에서 영상-오차를 줄여 카메라(또는 로봇) 자세를 목표로 유도한다. IBVS는 깊이/캘리브레이션 불확실성에 강하지만 2D-3D 매핑의 한계로 local minima, singularity, ‘camera retreat’ 같은 비효율 문제가 생길 수 있다.
회귀(regression) 기반 end-to-end 방식은 입력을 한 번에 velocity로 매핑하는 단일 스텝이라 관측 잡음에 민감하고, 분포 변화에서 오차가 누적되어 action이 시간적으로 흔들리며 trajectory jitter가 발생한다.

- **Core Contribution**: 이 논문은 Diffusion Policy를 서보잉에 적용한 diffusion 기반 시각 서보잉 방법 DiffusionVS를 제안한다. 관측된 AprilTag 코너의 정규화 이미지 좌표를 조건으로 하고 conditional denoising으로 카메라 velocity(속도)를 생성해, PBVS에 가까운 궤적 품질을 depth 추정 없이도 재현한다.
또한 정적 데이터로만 학습할 때 생기는 일반화 한계를 줄이기 위해, 인터랙티브 경험을 수집하며 다양성을 계속 확장하는 online training 패러다임을 도입한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 잡음/분포 이동 상황에서 시간 일관적인 제어를 생성하는 것과 (2) static dataset에 의존하면 학습 데이터 외 상황에서 성능이 급락하는 일반화 문제를 동시에 다루는 것이다.
저자들은 action을 속도 벡터 시퀀스로 두고 diffusion의 denoising 과정을 통해 일관된 action sequence를 만들며, 입력을 정규화 좌표로 구성해 카메라 intrinsic 변화에도 견디도록 했다. 더 나아가 온라인 수집 데이터로 학습 분포를 계속 넓혀 sim-to-real 격차를 줄이도록 설계했다.

- **Empirical Impact**: 시뮬레이션과 실제 실험에서 TE/RE 기반 평가 및 성공률로 성능을 검증했으며, 시뮬레이션 성공률은 거의 100%, 물리 실험은 93%에 도달했다. 또한 회귀 기반 단일 스텝 모델 대비 확연한 격차가 확인되었고, 기존 SOTA 회귀 네트워크에 diffusion 모듈을 붙이면 최대 30%까지 성공률이 개선되는 plug-and-play 일반화 효과도 제시했다.
연산 효율도 높아 perception 및 diffusion 모듈 합산 총 지연이 단계당 약 10.2 ms로 보고되어, 실시간 서보잉 적용 가능성을 뒷받침한다.



### 3D Scene Graphs: Open Challenges and Future Directions (https://arxiv.org/abs/2606.19383)
Comments:
          Invited article for the Annual Review of Control, Robotics, and Autonomous Systems Volume 10

- **Prior Approaches**: 3D Scene Graphs(3DSGs)는 기하 기반의 grounding 위에 의미·관계 추상화를 얹는 표현으로, 조작·내비게이션·작업 계획·장면 이해 등 다양한 응용에 쓰여 왔습니다. 다만 커뮤니티마다 3DSG의 정의, 구성 파이프라인, 평가 프로토콜이 달라 방법 비교가 어렵고, 공통 가정과 현실 배치에서의 잔여 과제를 파악하기 힘들다는 문제(단편화)가 지적됩니다. 기존 연구들은 노드/엣지 속성, 계층 구조, 동적 장면 표현, affordance-aware 확장 등 선택이 서로 달라 일관된 정리와 비교가 부족했습니다.

- **Core Contribution**: 이 논문은 3DSGs를 공통 정의로 정식화하고, 기존 formulation을 가르는 핵심 모델링 선택(노드·엣지 속성, 계층성, 동적 표현, affordance-aware 확장)을 체계적으로 분석합니다. 또한 원시 감각 관측으로부터 3DSG를 구성하는 방식에 대해 용어·관례·기법을 한데 묶어 정리합니다. 마지막으로 그래프 품질 같은 intrinsic 평가부터 작업 단위 성능 같은 downstream 평가까지 정렬해, 향후 로버스트한 real-world deployment 관점의 방향성을 제시합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 서로 다른 입력(비전/포인트 클라우드 등)과 서로 다른 모델링 선택이 결합되면서, 무엇이 성능을 좌우하는지 비교 가능한 형태로 고정하기 어렵다는 점입니다. 논문은 node/edge 속성 설계, 계층 구조, 동적 장면 표현, affordance-aware 확장처럼 선택지를 축으로 분류해 ‘어떤 가정이 들어가는지’ 추적 가능하게 만들고, 그래프 구성 파이프라인의 통일된 지도(map)를 제공합니다. 이를 통해 남은 공통 실패 원인과 연구 공백을 더 명확히 드러내려는 접근을 취합니다.

- **Empirical Impact**: 경험적 영향은 단일 알고리즘 제안이라기보다, 평가 전략을 intrinsic graph quality와 task-level performance로 나눠 커뮤니티가 동일 기준에서 비교하도록 돕는 데 있습니다. 논문이 제공하는 공통 정의·체계적 분류·리뷰는 재현성과 비교 가능성을 높여, 실제 배치에 필요한 로버스트성 이슈를 우선순위화하는 데 기여합니다. 또한 전용 웹사이트를 통해 조사 내용을 조직·확장할 수 있게 하여 후속 연구자들의 온보딩과 지식 축적을 촉진합니다.



### WorkBenchMark: A LEGO-Based Assembly Benchmark with an Assembly-by-Disassembly Baseline for the Smart Manufacturing Leagu (https://arxiv.org/abs/2606.19358)
Comments:
          RoboCup Symposium 2026 accepted paper

- **Prior Approaches**: 기존 로보틱스 조립 벤치마크는 대체로 초기 배치가 알려져 있거나(예: 가구/보드 중심), 특정 고정 부품에 최적화돼 있어 시야·계획·물리 제약을 함께 비교하기 어려웠습니다. LEGO 기반 연구도 있었지만 과제 설계와 성공 기준이 제각각이라 end-to-end 시스템 비교의 공통 기준이 부족했습니다. 한편 vision-language-action(VLA)은 언어 프롬프트로 그라운딩과 모터 제어를 한 모델에 결합하지만, 계획 지평이 길어질수록 물리적 가능성/안정성 추론이 약해지는 경향이 관찰됩니다.

- **Core Contribution**: 이 논문은 RoboCup Smart Manufacturing League(SML) Workbench Track을 모티프로 LEGO Duplo 기반 로봇 조립 벤치마크 WorkBenchMark를 제안합니다. 1m×1m×2m 작업공간 내 무작위 초기 구성과 open-world 인식을 전제로 4단계 난이도(총 400개 시뮬레이션 태스크, 이 중 40개는 실로 재현)를 제공합니다. 또한 open-vocabulary 인식과 Assembly-by-Disassembly(ABD) 기반 계획을 결합한 기준선 파이프라인을 공개하고, 이를 최신 VLA 계열과 비교해 벤치마크의 측정값을 확립합니다.

- **Technical Challenges**: 핵심 난제는 상징적 조립 순서를 실제 그리퍼 도달 가능성, 충돌, 그리고 인터로킹 구조 안정성 같은 물리 제약에 맞게 일관되게 실현하는 것입니다. 저자들은 목표 구조를 stud 그리드에 정렬된 voxel 점유 상태로 두고 ABD 탐색을 수행하되, 후보 제거가 가능한지 grasp reachability(충돌 없는 그립 후보)와 press-stability(남는 구조의 지지 다각형 내 삽입 위치)로 검증합니다. 인식은 GroundingDINO+SAM(자유형 텍스트 기반 탐지/세그멘테이션)과 FoundationPose(6D 자세 추정)로 open-vocabulary를 달성하고, 실행은 MoveIt2 기반 충돌 회피 이동과 Cartesian 제약 하강으로 stud-engagement 삽입 정렬을 강제합니다.

- **Empirical Impact**: 시뮬레이션 평가에서 제안한 구조화 파이프라인은 난이도 전 구간에서 높은 성공률과 비교적 안정적인 성능을 유지했지만, VLM/VLA 기준선은 복잡도가 커질수록 정확도와 안정성 위반이 함께 악화됐습니다. 특히 Tier 1에서는 약 97%, Tier 2에서는 약 90% 수준의 성공률을 보였고, Tier 3 이상에서는 파츠 수와 구조 복잡도 증가로 성능이 내려가도 일정 부분 과제를 완료했습니다. 반면 언어 기반 접근은 그라스퍼 reachability와 구조적 안정성 같은 물리적 제약을 충분히 반영하지 못해 실행 실패와 불가능한 액션 시퀀스가 늘어나는 양상이 확인됐습니다. 또한 벤치마크·시뮬레이터·기준선 구현을 공개해 통합형 로봇 조립 연구의 진행 상황을 연속적으로 추적할 수 있는 공통 잣대를 제공한다는 점에서 의미가 큽니다.



### Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots (https://arxiv.org/abs/2606.19357)
Comments:
          To appear at RLC 2026

- **Prior Approaches**: 로봇에서 강화학습을 다루는 기존 방식은 (1) 시뮬레이션에서 학습 후 로봇에 배포하거나, (2) 사람이 조종해 데이터를 수집한 뒤 offline-RL로 학습하는 방법, (3) 로봇에서 직접 학습하는 방법으로 나뉜다. 그중 직접 학습은 시뮬레이터나 인간 데이터가 없어도 되지만, 신뢰성·접근성·장기 운용성을 만족하는 로봇 플랫폼이 부족했다. 특히 real-time reinforcement learning 관점에서는 지연과 비정상성 같은 현실 요소가 시뮬레이션에선 잘 반영되지 않는 한계가 있었다.

- **Core Contribution**: 본 논문은 로봇에서 직접 강화학습을 장기 실험 가능하게 만드는 Physical Atari 플랫폼을 제시한다. Physical Atari는 Atari CX40+ 컨트롤러를 로봇이 물리적으로 조작하는 Robotroller와, ALE 게임 화면 및 보상 신호를 재현하는 Atari Devbox로 구성된다. 저자들은 이 플랫폼이 robust(고장 없이 오래), accessible(저비용·조립 용이), easy to use(개입 최소), versatile(다양한 ALE 게임)하도록 설계 핵심 결정을 정리한다.

- **Technical Challenges**: 핵심 기술 과제는 (a) 장시간 사용 시 기구 마모와 진동으로 인한 신뢰성 저하, (b) 실제 시스템의 지연을 포함한 real-time 상호작용에서의 성능 일관성, (c) 물리 입력과 게임 행동 간의 정확한 매핑/제어 안정화였다. 저자들은 마모를 줄이기 위해 볼 베어링으로 모든 구동 이동을 처리하고, 풀림·기어 마모·부품 파손 문제는 thread-locking, 금속 기어 서보로 교체, 금속 horn으로 보강해 수 주간 무고장 운용을 달성했다. 또한 서보의 PID 파라미터 튜닝으로 움직임을 부드럽게 하되 발생 가능한 고전류 상태를 high-current reflex로 차단해, 자동으로 안전 상태로 보호하도록 했다.

- **Empirical Impact**: 실험에서 Physical Atari 위에서 6개 ALE 게임(Pong, Seaquest, MsPacman, Assault, Asterix, Kangaroo)을 약 5.5시간 학습하며, 여러 번 반복해도 개입 없이 학습이 안정적으로 수행됨을 보였다. 또한 서로 다른 Robotroller(로봇 바디)로 학습된 정책을 옮겼을 때 성능이 전반적으로 저하되었고, 특히 Pong처럼 타이밍 민감도가 큰 게임에서 격차가 크게 나타났다. 나아가 새 바디로 배포 후에도 학습 알고리즘을 계속 적용하면 성능이 회복/개선되어, 로봇에서는 on-device adaptation의 중요성을 실증적으로 강조한다.



### The Token Is a Group Element: On Lie-Algebra Attention over Matrix Lie Groups (https://arxiv.org/abs/2606.20547)
Comments:
          preprint, 19 pages, 3 figures

- **Prior Approaches**: 기존 동등(equivariant) 공간추론에서는 토큰을 보통 벡터 임베딩으로 두고, 변환군 G의 작용을 representation ρ(g)로 외부에서 주입하는 방식이 표준이었다. 이때 동등성을 위해 irreducible representation, Clebsch–Gordan, steerable kernel, 보조 프레임 같은 표현이론/기하 도구가 필수로 등장한다. 또한 비컴팩트·비가환 affine 같은 군은 unitarity 제약(irreps)이나 surjective-exp 기반의 구성 한계로 우회/배제가 잦았다.

- **Core Contribution**: 이 논문은 attention 토큰 자체를 “벡터”가 아니라 matrix Lie group의 원소 g_i로 바꾸는 Lie-Algebra Attention을 제안한다. 토큰이 변환을 직접 담으므로, 쌍의 상대기하 g_i^{-1}g_j의 로그(log)로 얻는 Lie algebra 불변량 w_{ij}가 본질적으로 정해지고 동등성과 cocycle 조건도 표현이론 없이 자동으로 성립한다. 결과적으로 attention score는 학습 커널이 아니라 w_{ij}의 대수 노름으로 닫힌형(closed-form)으로 읽힌다.

- **Technical Challenges**: 핵심 난제는 비컴팩트/비가환 affine까지 포함하면서도 closed-form 점수를 만들 수 있는 불변 스칼라와 거리(노름) 설계를 찾는 것이다. 저자들은 로그 차트에서만 정의되는 상대 pose에 대해 w_{ij}=log(g_i^{-1}g_j)를 쓰고, Frobenius 기반의 block-weighted 내적(가중치 λ)을 통해 score s_{ij}=-||w_{ij}||^2_λ/τ를 구성한다. 특히 Aff(2), Aff(3)처럼 Ad-invariant 정준 내적이 존재하지 않는 비세미(simple) 케이스에서도 “불변량 w_{ij} 자체의 불변성”을 이용해 메트릭의 Ad-invariance 없이도 점수가 G-대각 작용에 대해 보존되게 했다.

- **Empirical Impact**: SE(2), SO(3), Aff(2)에서 sequence completion 실험을 통해 닫힌형 score가 같은 불변량을 쓰는 학습 MLP 커널과 비교해 성능을 내거나 SE(2)에서는 더 좋다고 보고한다. 파라미터 측면에서도 닫힌형은 50~80배 적은 score 파라미터로 동등/개선 성능을 달성하며, 벡터 토큰 baseline은 불변성 훼손으로 큰 성능·일관성 차이를 보였다. 이는 irreps 전통과 surjective-exp 전통이 막아왔던 affine full-frame 영역까지 attention을 확장하는 실증적 근거로 의미가 있다.



### HilDA: Hierarchical Distillation with Diffusion for Advancing Self-Supervised LiDAR Pre-trainin (https://arxiv.org/abs/2606.20189)
Comments:
          Accepted to ECCV 2026. Maciej and Jesper contributed equally

- **Prior Approaches**: 기존 camera-LiDAR knowledge distillation은 Vision Foundation Model(VFM)을 black-box teacher로 두고 주로 프레임 단위 feature similarity만 맞추는 경향이 강했습니다. 이 때문에 VFM의 층별(semantic evolution) 구조와 CLS 토큰이 담는 전역(global context) 의미를 충분히 활용하지 못하고, LiDAR의 시공간(spatiotemporal) 일관성 학습도 약했습니다.

- **Core Contribution**: 본 논문은 LiDAR 백본을 위한 self-supervised pretraining 프레임워크 HilDA를 제안합니다. HilDA는 계층적(hierarchical) distillation로 “semantic what(무엇)”을 VFM에서 가져오고, temporal occupancy diffusion으로 “geometric where(어디)”와 미래의 시공간 일관성을 함께 학습하도록 설계됐습니다.

- **Technical Challenges**: 핵심 난제는 (1) 서로 다른 모달리티에서 VFM의 층별 의미 계층을 점-픽셀 대응 위에서 안정적으로 전이하고, (2) 의미 정렬만으로는 부족한 동적 장면의 미래 점유를 생성적으로 학습하는 것입니다. HilDA는 기하학적 calibration 기반의 dense-to-sparse 멀티레이어 distillation과 CLS 토큰 정렬을 동시에 수행하고, 미래 BEV occupancy를 conditional diffusion(denoising 기반)으로 예측하는 auxiliary objective를 추가해 저라벨/무라벨 상황에서도 예측형 기하·운동 표현을 주입합니다.

- **Empirical Impact**: HilDA는 cross-modal distillation 벤치마크에서 state-of-the-art를 달성했으며, 특히 레이블이 적은 데이터 스케줄(1%–10%)에서 기존 distillation 대비 큰 폭의 향상을 보였습니다. 또한 3D object detection, scene flow, semantic occupancy prediction 전반에서 이전 기법을 능가하고, nuScenes-C에서 mCE 낮고 mRR 높게 나타나는 등 강인성도 개선되는 것으로 보고됩니다.



### ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing? (https://arxiv.org/abs/2606.19531)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 World Action Model(WAM)은 비디오 생성으로 미래 장면을 상상한 뒤 행동을 예측하는 방식이 많았다. 하지만 비디오 기반은 다중 프레임의 dense 미래 토큰 처리로 추론 비용이 커지고, 외형·카메라·배경 변화처럼 행동과 무관한 정보에 자원을 소모한다. 또한 긴 호라이즌 상상이 실제 접촉·변형을 물리적으로 일관되게 맞추기 어려워, 오류가 행동 예측을 오도할 수 있다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 “비디오 생성이 정말 필요할까?”라는 질문에 답하며, ImageWAM이라는 대안을 제안한다. ImageWAM은 텍스트 지시가 들어간 이미지 편집(image editing) 모델을 world-action 백본으로 재활용해, 현재 관측에서 목표 상태로의 “변환”만 요약하는 단일 엔드포인트 프레임 표현을 중간 컨텍스트로 사용한다. 특히 추론 시에는 목표 프레임을 디코딩하지 않고, 편집 denoising 과정에서 생긴 KV cache를 flow-matching 기반 action expert에 바로 조건으로 넣어 계산량을 줄인다.

- **Technical Challenges**: 핵심 과제는 비디오 생성이 제공하던 reason-before-act 중간 단계의 이점을, 이미지 편집의 컨텍스트로 얼마나 효과적으로 대체하느냐였다. 저자들은 편집 모델의 디코딩 출력이 아니라 denoising 중간 단계의 KV cache가 언어-조건 시각 변환 정보를 담는다는 점에 주목해, action expert가 이 캐시를 통해 직접 행동 시퀀스를 생성하도록 학습한다. 또한 학습 중에는 편집 denoising timestep τ를 샘플링해 다양한 단계의 캐시를 보게 하고, 손실은 이미지 엔드포인트 예측(velocity field)과 행동 flow-matching을 함께 최적화해 캐시-행동 정렬을 강화한다.

- **Empirical Impact**: 실험에서 ImageWAM은 LIBERO/LIBERO-Plus, RoboTwin 2.0 및 듀얼암 실세계 태스크(Flux.2 기반)에서 VLA 기준선 대비 성능을 개선했고, 비디오 기반 WAM과도 경쟁/유사 성능을 보이면서 계산 효율을 크게 낮췄다. 구체적으로 FLOPs는 63.65→9.7로, 지연(latency)은 1081ms→263ms로 감소해 실시간 제어에 유리함을 보여준다. attention 분석 역시 편집 캐시가 조작물·접촉부·목표 수납영역처럼 과업 관련 변화 영역에 집중한다는 점을 확인해, 이미지 편집이 비디오 기반 world-action 모델의 실질적 대안이 될 수 있음을 뒷받침한다.



### 3D-DLP: Self-Supervised 3D Object-Centric Scene Representation Learning (https://arxiv.org/abs/2606.19451)
Comments:
          ICML 2026. Project webpage: this https URL

- **Prior Approaches**: 기존 self-supervised object-centric 표현은 주로 2D RGB/영상에 머물러, 가려진 영역을 복원하거나 접촉 중심 조작에 필요한 정밀 3D 기하를 안정적으로 제공하기 어렵다는 한계가 지적돼 왔다. 3D를 다루는 방법도 point cloud/voxel을 그대로 쓰더라도, 색을 다루지 못하거나(in colorless), 렌더링 역문제에 의존하거나, 정책 학습에 바로 쓰기 힘든 거대·메모리 집약적 표현을 요구하는 경우가 많다.

- **Core Contribution**: 이 논문은 RGB-D 또는 (점유/색상) voxel 입력에서 장면을 객체 중심의 저차원 ‘3D latent particles’로 분해하는 3D-DLP를 제안한다. 각 particle은 3D keypoint 위치, bounding box 크기, appearance 특징 등을 disentangled 속성으로 담아, particle 단위로 해석 가능한 segmentation과 scene 편집(예: 물체 이동/스케일 조정)을 가능하게 한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 희소·불연속적인 voxel 격자에서 안정적인 keypoint 후보를 뽑고, (2) 색상까지 포함한 3D 복원에서 gray collapse를 막아야 한다는 점이다. 이를 위해 appearance-aware K-means keypoint prior(색을 CIELAB로 변환해 공간-외형을 함께 클러스터링)와 occupied voxel에 한정한 chroma reconstruction loss를 도입하고, 3D spatial transformer/3D 합성(compositing)으로 particle을 end-to-end 복원 학습에 연결한다.

- **Empirical Impact**: 실험에서는 합성 데이터와 실데이터에서 latent 공간이 해석 가능하고 controllable하다는 점(particle 위치·스케일 조작 시 새로운 장면 생성)을 시각화·정성적으로 입증한다. 또한 이 compact한 3D particle 표현을 entity-centric diffusion 기반 로보틱 조작 정책에 연결했을 때, explicit 3D 정보가 없거나 dense 3D를 메모리 부담 없이 쓰기 어려운 기준선들보다 MimicGen과 언어 조건 RLBench 태스크에서 성능 우위를 보이며, self-supervised 3D 분해가 다운스트림 control로 ‘실제로’ 이어질 수 있음을 보여준다.



### FlexLAM: Resolving the Bottleneck Trade-off in Latent Action Learning (https://arxiv.org/abs/2606.19408)
- **Prior Approaches**: 기존 Latent Action Model(LAM)은 행동 라벨이 적고 특정 작업·embodiment에 몰려 있다는 현실을 전제로, 행동이 없는 비디오에서 관측 전이(ot, ot+1)를 고정 길이 latent code로 압축한 뒤 소량 라벨로 executable action과 정렬한다. 하지만 고정 용량 bottleneck는 전이가 복잡한 정도를 반영하지 못해, 너무 조이면 정렬에 필요한 단서를 잃고(정보 소실), 너무 느슨하면 라벨이 좁을 때 translator가 잡음/비실행 변이를 구분해야 하는 부담이 커진다.

- **Core Contribution**: FlexLAM은 고정 용량 bottleneck를 “한 가지 코드 길이”에서 “variable-length latent actions”로 바꾸되, 새로운 아키텍처나 손실 함수를 추가하지 않는다. 핵심 아이디어는 retained-prefix training으로, 전이 코드의 접두사(prefix) 길이를 랜덤으로 샘플링하고 suffix는 null latent로 채워 prefix마다 유효(valid)한 코드가 되도록 nested dropout(변형) 방식의 학습을 수행하는 것이다. 이렇게 학습된 단일 모델은 여러 token budget을 추론 시점에 조절해도 정렬 품질을 유지한다.

- **Technical Challenges**: 가장 큰 기술 난제는 “고정 용량일 때는 길이 하나로 전이 복잡도를 모두 커버해야 해서 생기는 정렬 불안정”을 variable-length에서도 안정적으로 해결하는 것이다. FlexLAM은 decoder와 action translator를 모두 접두사 단위로 학습시키고, translator에도 동일한 prefix 샘플링 분포를 적용해 라벨이 scarce하거나 narrow한 상황에서 suffix 전용 변이에 의존하는 경향을 억제한다. 또한 추론 단계에서 필요한 k 토큰만 생성하고 나머지는 null로 채워, 같은 모델로 latency-정확도 트레이드오프를 제공한다.

- **Empirical Impact**: DMLab 실험에서 k∈{4,16,64} 각각에 대해 별도로 학습한 fixed-capacity baseline과 비교했을 때, FlexLAM은 모든 토큰 예산에서 일관되게 우수하거나 동등하며 특히 full-budget에서의 이득이 강조된다. 라벨 소스가 단일 작업에 극도로 치우친 stress test에서도 고정 용량 모델의 취약성이 관측되는 반면 FlexLAM은 대부분의 태스크에서 열화가 덜했다. Ego4D에서는 LPIPS/PSNR/SSIM 기준으로 external fixed-bottleneck 참고 모델(villa-X-LAM)보다 전이 재구성이 개선되었고, k를 늘릴수록 동일 모델 내에서 점진적으로 더 많은 시각 디테일을 복원하는 경향이 확인되어 real-world 표현력도 뒷받침한다.



New uploads on arXiv(cs.MA)

### Blame is easier than praise: Measuring off-ball defensive performance in footba (https://arxiv.org/abs/2606.19931)
- **Prior Approaches**: 기존의 수비 기여도 평가는 태클, 인터셉션 같은 제한된 이벤트에 주로 의존해 선수의 ‘연속적’ 위치 행동 영향이 충분히 포착되지 못했습니다. 또한 포지셔널 행동을 개인 단위로 공정하게 분해할 때, 선수별 정답 레이블(ground truth)이 없는 상황이 흔해 평가가 간접 지표에 머물렀습니다.

- **Core Contribution**: 이 논문은 멀티에이전트의 시공간 궤적에 대해 위협(expected threat)의 변화가 각 선수에게 어떻게 귀속되는지(attribution)로 문제를 재정의합니다. 선수 수준 라벨 없이도, defensive pressure areas(DPAs) 기반의 involvement score로 각 수비수가 임의의 패스에서 만들어진 위협에 대해 어느 정도 책임이 있는지 역할 조건부 기준선(role-conditioned baselines)을 통해 계산합니다.

- **Technical Challenges**: 핵심 과제는 개인별 책임을 말해줄 직접적인 ground truth가 없는데도, 위협 변화를 선수 단위로 신뢰도 있게 분해하는 것입니다. 이를 위해 자동으로 감지한 팀 구조 내부에서 기준선을 만들고, 정답이 없을 때는 여러 ‘상대적으로 약한’ 프록시를 결합해 robust summary score로 성능을 검증하는 평가 프로토콜을 제안합니다.

- **Empirical Impact**: 남자 월드컵 64경기, 여자 분데스리가 116경기, 남자 독일 3. 리가 336경기 등 매우 대규모 크로스 성별·크로스 대회 데이터로 타당성과 견고성을 평가했습니다. 정답 부재 환경에서 타당도(validity) 점수는 최고의 action 기반 지표 대비 약 1 표준편차 개선됐고, 다수의 인기 지표는 제한된 타당도를 보였습니다. 특히 득점 가치가 큰 장면을 허용한 ‘blame’는 외부 레이팅과 market values와 강하게 상관했으며, positioning errors를 신뢰 있게 측정하는 최초의 축구 메트릭으로 제시됩니다.



### SIGMA: Skill-Incidence Graphs for Compositional Multi-Agent Design (https://arxiv.org/abs/2606.19758)
Comments:
          EMNLP2026

- **Prior Approaches**: 기존 그래프 기반 multi-agent system(MAS) 연구는 협업을 통신 토폴로지(communication topology) 최적화로 개선하는 데 집중해 왔습니다. 하지만 각 노드가 닫힌 집합의 고정된 에이전트/역할/그룹으로 남아 있어, 보지 못한 능력 조합이 필요한 과제로의 일반화가 어렵습니다.

- **Core Contribution**: 이 논문은 SIGMA를 제안하며, 에이전트를 task-conditioned 스킬 묶음(task-conditioned bundles of reusable skills)으로 구성하는 skill-incidence graph 프레임워크를 제시합니다. 주어진 task와 skill library로부터 skill-agent incidence matrix를 예측하고, 선택된 스킬로 에이전트 임베딩을 조립한 뒤, 조립된 에이전트들에 대한 통신 토폴로지를 디코딩합니다.

- **Technical Challenges**: 핵심 난제는 ‘능력의 조합 가능성’ 자체를 그래프 노드 수준에서 구성 가능하게 만들면서, 실제 실행에서 그 조합 구조가 통신으로도 정확히 이어지게 하는 것입니다. SIGMA는 스킬별 mailbox를 두어 메시지를 관련 할당된 capability로 라우팅함으로써, incidence 구조가 단순한 예측을 넘어 직접 동작하도록 설계했습니다.

- **Empirical Impact**: SIGMA는 3개 base LLM을 사용해 6개 reasoning·coding 벤치마크에서 최상 평균 성능을 달성했으며, CARD(강력한 비-조합형 topology 기반 베이스라인) 대비 각각 2.06, 2.36, 1.75점 향상됐습니다. 또한 unseen skill libraries에 대한 강건성이 더 높아 평균 성능 하락 폭이 0.96점에 그쳤고, 통신 토폴로지 최적화 외에 ‘조합형 노드 구성(compositional node construction)’이 중요한 설계 축임을 시사합니다.



### Mesh Inference: A Formal Model of Collective Intelligence Without a Center (https://arxiv.org/abs/2606.19537)
Comments:
          21 pages, 2 figures

- **Prior Approaches**: 기존 분산 지능은 주로 중앙(집계자, 인덱스, 합성기)을 두거나, 모델을 공유하지는 않더라도 최종 동기화 지점이 필요했다. 예로 federated learning, centralized training with decentralized execution, retrieval-augmented 모델은 각각 집계/합성/리트리버블 인덱스라는 “center”를 유지해 관찰 교환만으로 완전한 center-free 집단 추론을 보장하지 못한다. 또한 deep equilibrium류의 에너지 기반 추론은 한 노드에서의 clamp-and-relax 고정점 추론을 잘 다루지만, 주권(sovreignty) 하에서 다수 에이전트가 동일한 고정점(중앙 최적해)을 얻는 조건은 정식화되지 않았다.

- **Core Contribution**: 이 논문은 mesh inference라는 center-free 집단 추론의 형식 모델을 제시한다: 각 에이전트는 private state(내부 상태)와 weights/gradient를 절대 공유하지 않고, typed observation만(허용된 관찰) 주고받아 어느 개인도 단독으로는 만들 수 없는 결론을 도출한다. 핵심 아이디어는 “질문하기 ask(·)”를 에너지 기반 평형 추론(clamp-and-relax)으로 보고, 그 추론이 성공/정확/비밀보장을 동시에 좌우하는 단일 객체로 admission/emission policy(관찰 수용/전송 정책)를 규정한 점이다. 특히 이 정책이 (1) 수렴, (2) identification-complete(중앙 최적해와 동일), (3) observation-only(내부 비노출과 기밀성)라는 세 성질을 함께 결정함을 보인다.

- **Technical Challenges**: 문제는 분산된 지역 free energy(지역 자유에너지)들이 얽혀 있을 때, 비대칭 전달/전송(에미션)이 있어도 중앙 최적해와 동일한 고정점으로 수렴하는가를 보이는 것이다. 논문은 각 필드별로 전달되는 가중치와 carrier(전송 여부)를 포함한 결합 연산이 항상 M-matrix 구조(약한 대각 우세 등)임을 이용해, 공통의 퍼텐셜이 성립하지 않는 비대칭 상황에서도 고정점이 유일하고 종료됨을 보장한다. 또한 identification과 confidentiality를 동일한 연결/랭크 조건의 부호가 뒤집힌 쌍으로 연결해, “관찰이 어느 정도로 carrier-connected(전달 경로로 연결)될 때만” 집단이 정답을 복원하고 공격자가 내부를 역추적할 수 없음을 동시에 설명한다.

- **Empirical Impact**: 비선형 일반화에서 “질문이 집단을 개선하는지 vs 자신감 있는 오류를 만드는지”는 open problem으로 남겼지만, 선형-가우시안 구간에서는 derived answer가 결정적이며 중앙 최적해와 동일하다고 정식으로 보인다. 이때 지연(latency)은 O(diam^2)로 제시되어 “중앙을 제거하는 비용”이 명시적으로 계량된다. 결과적으로 mesh inference는 center-free, observation-only 환경에서 수렴/정확/기밀을 동시에 보장하는 최초의 형식 모델로서, 에이전트 간 추론을 플랫폼이 아닌 수학적 보증으로 설계·검증하려는 흐름에 의미 있는 기준점을 제공한다.



### Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems (https://arxiv.org/abs/2606.20493)
Comments:
          20 pages, 4 figures, 4 tables

- **Prior Approaches**: 기존 연구는 LLM-as-judge에서 나타나는 자리/장문/자기선호 증폭 같은 편향이 어떻게 의사결정 품질을 왜곡하는지에 초점을 맞췄지만, 편향이 에이전트 네트워크를 따라 ‘연쇄 전파’되는지는 잘 다루지 못했습니다. MM-EPC는 multimodal(텍스트↔비전)에서 evaluator 편향이 strategy 선택으로 오염되는 현상(MM-EPC)을 보여줬지만, 멀티 에이전트의 멀티-홉 전파는 미지였습니다.

- **Core Contribution**: 이 논문은 Contagion Networks라는 틀로, evaluator 편향이 상호작용하는 여러 LLM 에이전트 사이에서 어떻게 퍼지는지를 계량화합니다. 에이전트 토폴로지 전체에 대한 Cross-Agent Contagion Matrix ΓN을 정의하고, 스펙트럴 반경 ρ(ΓN)에 따라 suppression/persistence/cascade의 동역학적 조건을 정리합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) evaluator 편향을 ‘기준화된 업데이트 신호’로 수치화하고 (2) 그 신호가 여러 홉에서 누적될 때 안정/붕괴 조건을 도출하는 것입니다. 이를 위해 TTRL(Test-Time Reinforcement Learning)로 에이전트의 전략 가중치를 evaluator 선호 쪽으로 가중 업데이트한 뒤, ΓN의 고유값(선형화 구간)을 통해 전파 레짐을 판정하는 수학적 모델과 실험 프로토콜을 결합했습니다.

- **Empirical Impact**: 3에이전트 실험(DeepSeek-chat, 구조/균형/근거편향 evaluator 프롬프트)에서 편향은 에이전트 간에 일관되게 전파되며, 평균 contagion 계수는 γ∈[0.157, 0.352] 범위로 관측됐습니다. 다만 homogeneous-model(동일 모델 계열)에서는 연쇄 전파가 약해 suppression regime에 놓이는 반면, cross-model 설정에서는 과거 MM-EPC 수준의 더 큰 γ로 cascade 가능성이 커진다는 ‘contagion spectrum’ 관점이 제시됩니다; 또한 evaluator 위원회 크기를 k=1→3으로 늘리면 효과적 contagion이 72.4% 감소해 실용적 완화책으로 이어집니다.



### An Infrastructure-less, Control-Independent Solution to Relative Localisation of a Team of Mobile Robots using Ranging Measurements (https://arxiv.org/abs/2606.20365)
- **Prior Approaches**: 기존의 협력적 로컬라이제이션은 고정 앵커(known position reference)로 관측가능성을 보장하는 경우가 많았고, 앵커가 없으면 불충분한 정보로 인해 다중 해가 생겨 유일한 궤적을 얻기 어렵다. 또 일부 방법은 angle까지 측정하거나, 관측가능성 강화를 위해 로봇의 능동적인 motion 제어/형성 제어를 강하게 요구해 실제 배치 비용과 제약이 커진다. 이런 방식은 인프라 구축이 불가하거나 사용자 숙련도에 따라 세팅이 어려운 환경에선 확장성과 유연성이 떨어진다.

- **Core Contribution**: 이 논문은 MHDCL(Multi-Hypothesis Decentralised Cooperative Localisation)로, 앵커 없이도 협력 로봇 팀의 2D 상대 포즈를 추정하는 완전 분산형 알고리즘을 제안한다. 핵심은 ‘관측가능성 조건을 인위적으로 만족시켜 유일해를 만들기’가 아니라, 가능한 해 전체를 유지하는 multi-hypothesis Bayesian 프레임워크로 문제의 ill-posedness 자체를 포용하는 패러다임 전환이다.

- **Technical Challenges**: 가장 큰 기술적 난제는 앵커 부재로 인해 시스템이 순간적으로 관측 불가능해질 수 있고, 그때마다 해 공간이 늘어나 다중 가설을 계산·전달하기가 무거워진다는 점이다. 이를 위해 particle filter 기반으로 각 에이전트가 odometry와 sparse inter-agent distance(UWB 등)만으로 후보들을 갱신하되, resampling 후 GVMMM(Gaussian-von Mises Mixture Model)으로 입자들을 클러스터로 압축해 통신·계산 비용을 줄이고, 필요 시 부분적으로 연결된 네트워크에서도 클러스터/모션 공유로 추정 세트를 복원한다.

- **Empirical Impact**: 실험은 현실적인 상황(스파스한 거리 측정, 통신 제한, 관측 불가능 구간 포함)에서 단일 추정보다 ‘해 집합 유지’가 더 견고한 로컬라이제이션을 제공함을 보여주는 데 초점을 둔다. 분산형이며 외부 인프라나 motion 제어가 필요 없다는 점에서, 로봇 플릿·협동 탐사·동적 환경처럼 배치 조건이 까다로운 분야에 실사용 가능성을 높인 기여로 평가된다.



### Phoenix: Safe GitHub Issue Resolution via Multi-Agent LLMs (https://arxiv.org/abs/2606.20243)
- **Prior Approaches**: 기존 GitHub issue resolution 에이전트(Devin, SWE-Agent, AutoCodeRover 등)는 LLM 에이전트를 활용해 수정/테스트를 수행하지만, 종종 resolution rate 최적화 과정에서 회귀(regression) 위험이 커질 수 있다는 문제가 지적돼 왔다. 또한 SWE-bench류 벤치마크 성과에 초점이 맞춰져 실제 배포 환경의 웹훅·권한·CI·평가 프로토콜 차이를 충분히 반영하지 못한다는 한계도 있었다.

- **Core Contribution**: Phoenix는 issue triage부터 pull-request 생성까지 end-to-end로 이어지는 multi-agent LLM 시스템을 제안하며, “정답률”보다 “정확성 보존(correctness preservation)”을 최우선으로 둔다. 변경은 PR 열기 전 기준선(baseline) 테스트와 비교해 신규 실패를 만들지 않았는지 확인한 뒤 진행되며, 전체 파이프라인은 6개 에이전트와 label 기반 GitHub webhook state machine으로 조정된다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 실제 배포에서 발생하는 WAF 필터링/토큰 만료/권한 경계/플레이키 CI 같은 운영 리스크를 구조적으로 막고, (2) LLM이 생성한 코드가 올바른 경로에 위치하도록 하는 파일 로컬라이제이션(localization) 한계를 줄이는 것이다. Phoenix는 7단계 안전장치(경로 탐색 방지, 워크플로 가드레일, 입력 콘텐츠 sanitization, 재시도 한도, 동시성 직렬화, 토큰 refresh, label 상태 배타성)로 안전성을 제어하고, 선택적으로 Reproducer로 “실패하는 테스트”를 근거로 구현을 고정한다.

- **Empirical Impact**: SWE-bench Lite의 production 웹훅 경로 24개 인스턴스(슬라이스)에서 oracle-resolved 75%를 달성했으며, oracle 성공 사례에서는 pass-to-pass 회귀가 없었다고 보고한다(단, 해당 슬라이스는 기존 리더보드 프로토콜과 직접 비교가 어렵다는 한계도 명시). 또한 실제 42개 이슈 파일럿에서 14개 저장소 모두에 대해 correctness preservation 100%를 보였고, hard tier에서 PR 생성 평균 122초 수준이었으며, 수동 점검 결과 약 절반은 실제 모듈을 잘 겨냥한 수정이었고 나머지는 Planner의 키워드 기반 경로 추정 오류에 더 많이 기인했다.



### A Multi-Agent system for Multi-Objective constrained optimization (https://arxiv.org/abs/2606.20236)
Comments:
          Presented at the 17th Workshop on Optimization and Learning in Multiagent Systems (OptLearnMAS, this https URL), co-located with the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)

- **Prior Approaches**: 계산·네트워크 의사결정 문제를 비용 최소화 + 성능 제약(QoS)으로 두면, 동적 환경에서는 RL로 런타임 최적화를 많이 시도한다. 이때 제약 위반을 페널티로 묶어 가중합 스칼라 보상으로 만드는 Lagrangian-inspired 보상 설계가 일반적이지만, 핵심 트레이드오프는 수동으로 고른 가중치에 좌우된다.

- **Core Contribution**: 이 논문은 MAMO(Multi-Agent system for Multi-Objective constrained optimization)를 제안해 “보상 가중치 선택” 자체를 학습 문제로 격리한다. Task-Execution(TE) 에이전트는 기존 방식의 가중 보상 RL로 행동을 학습하고, Weight-Adaptation(WA) 에이전트는 장기 지표를 관측해 가중치를 조정함으로써 비용 효율성과 제약 만족의 균형을 경험적으로 맞춘다.

- **Technical Challenges**: 기술적 난제는 수동 가중치 튜닝을 대체하면서도, 비정산 환경에서 제약과 비용 목표의 상대적 중요도가 변할 때 안정적으로 균형점을 찾는 것이다. MAMO는 학습을 2단계 반복으로 구성해(가중치 고정 후 TE 학습 → WA가 성능 요약 지표로 가중치 평가) 모델 내부를 차분(미분)하지 않아도 되는 형태로 바깥 루프에서 가중치를 갱신한다.

- **Empirical Impact**: 엣지 FaaS 단일 함수 replica scaling을 단순화한 실험에서, WA가 가중치를 재조정하며 평균 rejection 확률이 허용 임계치(예: 0.05) 근처로 수렴하도록 유도하는 결과를 보였다. 또한 노이즈로 부하가 흔들려도 제약 위반을 피하면서 실행 비용을 과도하게 희생하지 않는 적응성을 확인해, 제약 최적화에서 자율적 가중치 조정의 가능성을 시사한다.



### RACL: Reasoning-Agent Control Layers for Continuous Metaheuristic Learning (https://arxiv.org/abs/2606.20142)
Comments:
          10 pages, 5 tables

- **Prior Approaches**: 메타휴리스틱을 운영 환경에 반복 적용할 때, 기존 방식은 한 번 설정된 내부 탐색 동작을 그대로 쓰는 경우가 많습니다. hyper-heuristics나 adaptive operator selection, 강화학습·graph reinforcement learning 등은 검색 과정 관찰을 통해 제어를 시도하지만, 본 논문은 ‘추론 기반으로 제어 규칙을 가설-검증-통합-설명’하는 관점이 상대적으로 부족하다고 봅니다. 또한 학습 기반 최적화나 LLM 기반 휴리스틱 생성은 새 솔버/정책을 만들기 쉬운 반면, 고객의 business constraints를 보존한 채 기존 옵티마이저를 ‘제어 레이어’로 개선하는 실용 절차는 더 정립이 필요하다는 문제의식을 제시합니다.

- **Core Contribution**: RACL(Reasoning-Agent Control Layer)은 기존 metaheuristic 위에 reasoning agent를 얹어, 옵티마이저 자체를 대체하거나 business constraints를 수정하지 않으면서도 내부 search behavior를 제어하도록 설계한 방법입니다. 에이전트는 operational memory를 관찰·조회하고 과거 실행을 근거로 bounded hypothesis를 만들며, 개입을 시험한 뒤 guardrails로 위험을 관리하고 유용한 정책을 consolidate하며 결정 과정을 설명합니다. 즉, 특정 라우팅 규칙이 아니라 ‘메타휴리스틱 제어 규칙을 발견→검증→통합→설명’하는 절차 자체가 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) 실행 중 관찰 가능한 search state를 evidence로 삼아 meaningful control을 만들고, (2) 실패 시 피해를 줄이기 위해 개입을 bounded로 제한하며, (3) 메모리를 ‘정답’이 아닌 evidence로 재해석·업데이트하는 루프를 구현하는 것입니다. 논문은 observe→retrieve→reason→hypothesize→intervene→evaluate→guard→consolidate→explain→update memory 사이클로 이를 구성하고, 실험에서는 Codex를 in-the-loop reasoning agent로 써서 로그를 해석하고 live bounded intervention을 제안하게 했습니다. 이후 평가 재현성을 위해 policy proxy로 행태를 고정해 비교 실험을 가능케 했습니다.

- **Empirical Impact**: 차량 라우팅을 testbed로 한 실험에서 RACL은 feasible 21개 케이스 중 OMP(Operational Memory Policy)를 18/21에서 개선 또는 동률로 만들었고, STP(Stagnation-Triggered Policy)는 21개 중 18/21에서 개선 또는 동률을 보였습니다. 평균 비용 관점에서 RACL은 STP 대비 -0.641%, OMP 대비 -4.913%의 delta를 보고하며, Sevilla-9/10 paired sample에서는 Fixed 대비 -8.337%, STP 대비 -1.605%의 개선이 관측됐습니다. 또한 측정된 런타임에서는 material computational overhead가 거의 없었고, 에이전트의 business-readable 설명(정체 구간에서 개입 강도를 조절하고 이후 보수 모드로 안정화 등)이 재현 가능한 통제 정책으로 정리된 점에서 운영 관점의 의미가 큽니다.



### ScaffoldAgent: Utility-Guided Dynamic Outline Optimization for Open-Ended Deep Research (https://arxiv.org/abs/2606.20122)
Comments:
          9 pages, 6 figures

- **Prior Approaches**: 기존 OEDR( Open-ended Deep Research ) 시스템은 계획-후-작성(plan-then-write) 방식으로 아웃라인을 고정하거나, 이후 로컬 휴리스틱으로 부분 수정하는 경우가 많다. 이때 정보가 누적될수록 아웃라인이 증거 공간에서 어긋나 scaffold drift가 생기며, 수정의 효과는 더 나중에야 드러나 지연 피드백 문제가 나타난다.

- **Core Contribution**: ScaffoldAgent는 아웃라인을 정적 계획이 아니라 시간이 지나며 진화하는 구조적 scaffold로 두고, 이를 유틸리티(utility) 기반으로 동적으로 최적화한다. 아웃라인 변화를 Expansion(확장), Contraction(축소), Revision(수정)의 세 가지 구조 연산으로 명시화해 중복/부족/취약 구간을 통제적으로 업데이트한다.

- **Technical Challenges**: 핵심 기술 도전은 (1) 정보 축적 중에도 전역 위계가 흔들리지 않게 구조 진화를 제어(C1)하는 것과 (2) 수정 시점의 유용성을 즉시 추정해 지연 피드백을 줄이는 것(C2)이다. ScaffoldAgent는 각 연산의 downstream 가치(검색 이득, 구조 일관성, trial 생성 품질)를 종합한 utility 신호로 노드 선택(UCB 스타일), 연산 스케줄링, 종료 조건까지 추론 시 제어한다.

- **Empirical Impact**: DeepResearch Bench와 DeepResearch Gym 실험에서 ScaffoldAgent는 장문 리포트 품질(RACE 등)과 사실 근거(인용 기반 지표)를 함께 개선하며 기존 deep research agent들을 일관되게 앞섰다. 또한 유틸리티 차원 및 연산 연착(ablation) 결과, Expansion/Contraction/Revision과 retrieval-structure-generation 유틸리티 구성요소가 성능을 떠받드는 것으로 확인되며, multi-turn follow-up에서도 관련 부분만 비파괴적으로 갱신해 성능을 높였다.



### Deep-Unfolded Coordination (https://arxiv.org/abs/2606.19920)
Comments:
          The second and third authors contributed equally (equal second authorship). 35 pages (10 pages main text), 17 figures, 3 tables

- **Prior Approaches**: 분산 최적화는 다중 로봇 문제를 병렬로 쪼개고 해석 가능성이 높다는 장점이 있지만, ADMM-DDP는 페널티 파라미터 등 하이퍼파라미터 튜닝 의존도가 커서 스케일이 커질수록 조정 변수가 폭증한다. 또한 기존 deep-unfolding(L2O)은 주로 반복별 고정 가중치를 학습해 ‘open-loop’ 성격이 강해 옵티마이저 상태에 따른 피드백 적응이 제한적이었다. 비슷한 시도로 ADMM 학습을 GNN/MLP 기반으로 확장한 연구들이 있었지만, (페널티) 파라미터 적응 범위나 비볼록 최적화에서의 학습 안정성 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 ADMM-DDP의 페널티 파라미터를 solve-time에 동적으로 조절하도록 설계한 deep-unfolding 프레임워크 Deep Coordinator를 제안한다. ADMM-DDP의 반복을 고정된 K회 레이어로 언롤(unroll)하고, 각 레이어 사이에 ‘옵티마이저 성능(상태)’을 입력으로 다음 페널티 파라미터를 출력하는 learnable feedback 정책을 둔다. 또한 비볼록 deep-unfolded 모델을 mainstream supervised 방식으로 학습하면 degenerate solution이 생길 수 있음을 지적하고, 이를 피하는 unsupervised 학습식을 제안한다.

- **Technical Challenges**: 핵심 기술 도전은 (1) 비볼록 최적화에서 페널티를 solve-time으로 학습·적응시키되 안정적으로 학습하려는 문제와 (2) K회 ADMM-DDP를 end-to-end로 미분하려면 내부(DDP/서브문제) 반복까지 unrolling하면 계산·메모리 비용이 너무 커지는 문제다. 이를 위해 논문은 IFT(Implicit Function Theorem)를 활용해 내부 솔버 반복을 역전파 경로에서 제외하는 그래디언트 계산 스킴을 도입해 O(K·T) 수준의 비용으로 미분 가능하게 만들었다. 학습 손실은 비용(cost)과 제약 위반(constraint violation)을 함께 다루는 형태로 구성해 residual 기반의 퇴행(stalling) 유인을 줄였다.

- **Empirical Impact**: 시뮬레이션에서 Deep Coordinator는 자동차/쿼드로터 다중 로봇 태스크 3종(장애물 회피, 교차, 장애물 비행)에서 기존 최적화 대비 동일 수준의 품질을 훨씬 빠르게 달성했다. 특히 Vanilla ADMM-DDP 대비 6.18–9.44x 빠르게(동일 반복 예산 내) 비슷한 수준의 비용과 제약 만족을 보였고, 비용 기준으로 더 오래 돌린 Vanilla 해보다도 6.18–9.44x의 벽시계 시간 이득을 보고했다. 더 나아가 학습한 팀 규모보다 최대 8배 큰 환경에도 배치했을 때 성능 이점이 유지되며, 이는 ADMM-DDP 구조가 제공하는 inductive bias와 로컬 피드백 설계 덕분이라는 해석을 제시한다.



### Semiglobal Input-Delay Tolerance Algorithm for Distributed Nonconvex Optimization of Networked Nonlinear Systems (https://arxiv.org/abs/2606.19871)
Comments:
          36 pages, 5 figures

- **Prior Approaches**: 분산 최적화는 네트워크 분해성과 데이터 지역성 때문에 센서, 스마트그리드, 로보틱스 등에서 꾸준히 주목받아 왔습니다. 다만 기존 연속시간 접근은 알고리즘 반복자(iterate)를 ‘상태’로 두거나 단일 적분자 같은 이상화에 머물러, 실제 물리적 노드의 비선형 동역학을 함께 제어하며 최적화까지 달성하는 문제를 제대로 다루지 못했습니다. 또한 입력 지연이 있는 경우는 주로 선형 시스템에 대한 Lyapunov–Krasovskii 방식이나 지연-무가정 설계에 의존해 비선형 네트워크에서는 안정 보장이 크게 공백으로 남아 있었습니다.

- **Core Contribution**: 이 논문은 입력 지연이 있는 네트워크 비선형 시스템에서 분산 최적화가 ‘합의(consensus) 제약’ 하에 실제 해에 수렴하도록 보장하는 새로운 개념 input-delay tolerant semiglobal convergence(IDTSC)를 제시합니다. IDTSC는 초기 집합의 크기(r)가 주어지면, 그에 대응하는 허용 지연 상한이 존재하여 합의 제약을 유지하면서 각 노드 상태가 최적해로 수렴함을 뜻합니다. 이를 실현하기 위해 계층적(hierarchical) 설계와 input-to-state stability(ISS) 분석을 결합한 semiglobal input-delay tolerant(SIDT) 알고리즘을 제안합니다.

- **Technical Challenges**: 핵심 기술 난점은 입력 지연이 비선형 동역학과 결합될 때 지연된 구배정보가 안정성을 망가뜨려 forward invariance를 깨고, 심지어 작은 지연에서도 수렴을 저해할 수 있다는 점입니다. 논문은 지연에 의해 유발되는 결합항을 계층적 구조와 ISS 기반 복원(invariant/constraint-set 재진입) 논리로 제어하며, 지연에 따른 이득 재조정이 필요 없는 delay-independent 설계를 통해 실용성을 높였습니다. 또한 기존의 강한 볼록성 요구를 Polyak–Łojasiewicz(P–Ł) 조건으로 완화해, 비볼록 최적화에서도 IDTSC/실용적 수렴( practical semiglobal convergence )을 확장합니다.

- **Empirical Impact**: 이론이 다루는 NNS(네트워크 비선형 시스템)와 입력 지연 설정에서 수치 실험을 통해 SIDT 알고리즘이 합의 제약을 만족한 채 최적해 주변으로 수렴하는 경향을 확인합니다. 특히 지연이 존재해도 이득을 지연-의존적으로 다시 튜닝하지 않아도 되는 점(delay-independent design)이 실험 관점에서 강점으로 드러납니다. 결과적으로 로보틱스 등 지연이 현실적으로 피하기 어려운 응용에서 제약을 만족하며 분산 최적화를 수행하려는 연구에 안정 보장 프레임을 제공하는 데 의미가 있습니다.



### Heterogeneous LLM Debate Under Adversarial Peers: Honest Gains, Replacement Costs, and Resilienc (https://arxiv.org/abs/2606.19826)
- **Prior Approaches**: 기존 연구들은 LLM debate가 여러 모델의 독립 답변과 상호 논증을 통해 추론을 개선한다고 보지만, 최종 정확도 중심 평가는 ‘수정이 이득인지 손해인지’ 메커니즘을 가립니다. 또한 이기종(heterogeneous) 패널의 다양성이 보완적 증거와 오류 비상관성을 준다는 주장과 함께, debate 교환 자체가 악의적/체계적 오정보를 확산시키는 영향도 알려져 왔습니다. 다만 기존 평가는 악성 참여자가 들어왔을 때 이득이 얼마나 지워지는지, 그리고 패널이 이미 오염(contaminated)된 상황에서 방어가 가능한지에 대한 정량 분리가 부족했습니다.

- **Core Contribution**: 이 논문은 이득과 위험을 최종 정확도가 아니라 ‘수정(revision) 행동 변화’로 측정합니다. 구체적으로, honest agent가 답을 얼마나 자주 바꾸는지와 그 변경이 corrective인지 harmful인지(유해 수정) 비율로 분해해 defender 관점에서 비교합니다. 또한 두 현실 시나리오(추가된 peer가 악의적인 경우, 이미 오염된 패널에 정직한 이기종 peer를 투입하는 경우)를 나눠서, 이기종이 공격 표면이면서도 특정 조건에서는 방어로 작동할 수 있음을 보여줍니다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘debate가 잘된다/못된다’가 아니라 ‘왜 잘못되는가’를 분리하는 것입니다. 이를 위해 R0→R1(첫 수정 단계)에서의 방향성(정답에서 벗어남 vs 정답으로 감)을 DC(정확한 방향의 수정)와 DM(유해 수정)로 분류하고, 유해 수정 비율과 함께 약한 방어자에서 발생하는 ceiling 효과를 피하기 위해 end-to-end flip rate(초기 정답이 최종 라운드에 유지되지 않는 비율)를 추가합니다. 동일한 debate 프로토콜은 고정하고, 슬롯에 들어가는 모델 조합만 바꿔 악의성(adversariality)과 이기종성(heterogeneity)을 통제한 matched 비교와 오염(contamination) 비교를 설계했습니다.

- **Empirical Impact**: 실험은 네 모델 패밀리와 세 추론 벤치마크에서 일관된 부호(sign)를 보이며, 이기종 peer는 정직하면 유해 수정(harmful revision)을 크게 낮추고 악의적이면 이를 되돌립니다. 예로 Llama-3.1-70B 방어자가 MATH-hard에서 baseline 89%→정직한 혼합 35%→악의 혼합 90%로 유해 수정 비율이 극적으로 변했습니다. 특히 오염 상황에서는 정직한 이기종 peer가 방어 역할을 하며, 같은 조건에서 초기 정답이 최종까지 유지되지 않는 flip rate를 31%에서 6%로 낮추었습니다. 즉, 다양성은 단순히 성능 향상 요소가 아니라 ‘신뢰할 수 없는 참여자가 있는 시스템에서의 보안 파라미터’로 재정의될 수 있음을 시사합니다.



### Library-Aware Doubles and Iterative Repair for Large Language Model-Generated Unit Tests in OpenSIL Firmwar (https://arxiv.org/abs/2606.19725)
Comments:
          20 pages, 10 figures

- **Prior Approaches**: 기존 UT 자동 생성은 LLM이 초안을 만들고, 컴파일/실행 실패 로그로 반복 수정하는 generate–compile–run–repair 루프를 많이 사용한다. 하지만 firmware C(특히 EDK II)에서는 헤더·심볼·INF 메타데이터·패키지 의존성이 조금만 어긋나도 링크가 쉽게 깨지고, 커버리지가 높아도 정답을 보장하기 어려운 test oracle 문제가 남는다. 또한 openSIL처럼 XFER table·ip2ip 같은 디스패치/함수 포인터 의존과 deep double, 얕은 shallow stub의 구분이 필요한 환경에선 단순한 UT 초안 생성만으로는 반복 수정을 감당하기 어렵다.

- **Core Contribution**: 이 논문은 openSIL C 코드베이스용 UT를 “초안 작성→EDK II 빌드/디스패치→빌드 리페어→LCOV 기반 커버리지 정제”로 이어지는 다중 에이전트(LLM 기반) 파이프라인으로 자동화한다. 핵심은 라이브러리 인지(library-aware) 방식으로 stubs/mocks/fakes를 재사용하거나 최소 동작의 double을 합성하되, 템플릿/심볼 위생 규칙을 위반하지 않도록 강제한다. 그 위에 컴파일·링커 로그와 라인 커버리지 피드백을 동시에 소비하는 “컴파일–디스패치 수리 루프”를 얹어 UT를 빌드 가능/디스패치 가능/커버리지 개선 단계로 점진 전환한다.

- **Technical Challenges**: 도전 과제는 (1) EDK II 제약 하에서 UT가 컴파일·링크되도록 include/INF/패키지 의존성과 심볼 충돌을 동시에 맞추는 것, (2) FUT가 호출하는 cross-module 경로는 deep double로 유도하되 같은 소스 파일의 부수 호출은 shallow stub으로 안전하게 처리하는 것, (3) 실행 후 LCOV에서 miss된 라인을 실제로 더 타격하도록 입력과 설정을 정제하는 것이다. 논문은 Chroma 기반 VDB에서 관련 함수·기존 UT·검증된 테스트 더블을 찾아 재사용하고, build 로그와 디스패치 결과로 “타겟 최소 수정”만 수행하는 상태 그래프(11단계)로 이를 해결한다. 라인 커버리지 기반 정제는 LCA만 쓰는 변형과 vector-database retrieval까지 함께 쓰는 변형을 분리해 효과를 비교한다.

- **Empirical Impact**: 평가 결과, 76개 FUT(functions under test) 중 73개에 대해 컴파일 가능한 UT를 생성했다. 라인 커버리지 가이드 없이(또는 retrieval augmentation 없이) 평균 line coverage는 73.9%까지 도달했으며, 48개 부분 집합에서는 라인 커버리지 가이던스만으로 98.8% 평균을, LCA+vector database retrieval 조합에서는 94.7% 평균을 각각 달성했다. 전반적으로 constrained firmware 환경에서 UT 생성 효율과 커버리지 향상이 유의미하게 개선되고, 수동 디버깅/수정 반복 비용을 줄일 수 있음을 실증한다.



### Exit-and-Join Dynamics for Decentralized Coalition Formation (https://arxiv.org/abs/2606.19683)
- **Prior Approaches**: 기존 연합/코얼리션 형성 연구는 대체로 중앙집중적 split-and-merge나 전역 최적화처럼 “처음부터 설계된” 절차를 가정하는 경우가 많았습니다. 반면 동적(dynamic)·엔도제너스(endogenous) 접근도 있었지만, 중간 상태에서 에이전트가 참고하는 보상이 전역 협상/전역 정보에 의해 계산되는 모델이 주류였습니다. 그 결과, 실제로는 에이전트가 자신의 현 소속(로컬 정보)만 보고 나가고(구성원 이탈) 들어가는 exit-and-join이 어떻게 안정성으로 이어지는지에 대한 설명력이 약했습니다.

- **Core Contribution**: 이 논문은 코얼리션 형성을 “분산형 비동기 동역학”으로 모델링하며, 각 에이전트의 편차(deviation)는 단독 출구(exit)와 입장(join) 결정만으로 발생하게 만듭니다. 보상은 전역적으로 합의된 코얼리션 구조가 아니라, 에이전트가 현재 속한 코얼리션 내부에서 Aumann-Dreze value로 계산되어 로컬 비교만 수행합니다. 또한 단말(terminal) 코얼리션 구조를 “허용되는(대상 코얼리션의 수락/조건) 개별 이익 편차가 더 이상 없는 고정점”으로 정확히 연결해, 협력적 지불 분배와 비협력적 best-response가 동시에 성립하는 조건을 정리합니다.

- **Technical Challenges**: 가장 큰 난점은 로컬 보상 규칙(Aumann-Dreze value)이 만드는 유인 구조가, 동역학의 수렴(예: scalar Lyapunov 또는 exact-potential)과 어떻게 정합되는지 찾는 것입니다. 연구진은 단말 조건을 유도된 비협력 best-response의 고정점으로 재해석하고, Lyapunov/포텐셜 표현이 성립하려면 “인센티브 정렬(incentive-alignment)”이 추가로 필요함을 조건화합니다. 더불어 switching cost(전환 비용)와 acceptance cost(수락 비용)가 로컬 안정성과 상태 변화 가능성을 어떻게 바꾸는지까지 분석합니다.

- **Empirical Impact**: 실험에서는 유한 시간 내 안정화(finite-time stabilization), 비용(전환/수락 마찰) 민감도, 그리고 특별한 convex-game 벤치마크를 통해 동역학의 선택 결과가 어떻게 달라지는지 검증합니다. 특히 grand coalition이 효율적(협력적으로 유리)이어도, 로컬·myopic한 exit-and-join이 항상 같은 결말을 보장하지는 않는다는 점을 동역학 관점에서 보여줍니다. 이는 다중 에이전트가 태스크/조직 경계 사이를 자율적으로 이동하는 현실 시나리오(시장 이동, 온라인 커뮤니티 재편 등)를 더 그럴듯하게 모델링하는 데 의미가 큽니다.



### Formal Verification of Learned Multi-Agent Communication Policies via Decision Tree Distillation (https://arxiv.org/abs/2606.19632)
Comments:
          9 pages, 3 figures, 7 tables. Accepted at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026), Pittsburgh, Pennsylvania, USA, September 27-October 1, 2026

- **Prior Approaches**: 기존 MARL(MAPPO, QMIX 등)은 다중 에이전트 간 협력을 emergent communication으로 학습하지만, 실제 드론·자율주행 같은 안전 비즈니스에는 formal safety guarantee가 부족합니다. 또한 단일 에이전트 신경망 검증 도구(Relux/DeepPoly 등)는 multi-agent의 상태 공간 폭발과 통신을 직접 다루기 어렵고, 정책 추출·증류(VIPER/MAVIPER류)는 통신 의미를 포함한 end-to-end 검증 파이프라인이 제한적이었습니다.

- **Core Contribution**: 이 논문은 learned multi-agent communication policy를 대상으로 end-to-end 안전 검증을 제공하는 새로운 프레임워크를 제시합니다. 핵심은 신경 정책을 communication semantics를 반영해 decision tree로 policy abstraction(증류)한 뒤, PRISM 기반 확률 모델체킹으로 PCTL 성질을 formally verify하고, verified safety가 원래 네트워크로 Monte Carlo에서 전이되는지를 함께 검증한 점입니다. 

- **Technical Challenges**: 문제는 (1) 신경 정책이 NP-hard 검증 대상이고 (2) multi-agent·통신으로 상태 공간이 기하급수적으로 커진다는 점입니다. 이를 위해 관측에서 50개 도메인 피처를 설계해 증류 정확도를 높이고, CART 기반 decision tree를 PRISM DTMC로 자동 변환할 때 feature-to-state 변수 대응을 “정수 산술로 완전 대응”하도록 강제했으며, PCTL 검증은 pairwise decomposition과 union-bound aggregation으로 조합 검증을 수행했습니다. 비결정성은 초기 상태 난수와 leaf impurity를 모델링해 확률 추정의 일관성을 확보하고, 비핵심 에이전트는 calibration rollout으로 이웃 기반 전이 커널을 실증 추정했습니다.

- **Empirical Impact**: VQ-VIB(이산 메시지) 정책(에이전트 5~7명)에서 decision tree distillation fidelity가 97.9%±1.2%에 도달했고, PRISM에서 안전/생존/협력 범주의 PCTL 속성 18개를 검증해 평균 88.9% satisfaction을 보고했습니다. 특히 충돌 확률은 0.3%로 안전 임계(1%)를 만족했으며, 원래 신경망에 대한 Monte Carlo 검증에서 verified 성질이 <=0.6 percentage-point 편차로 전이됨을 확인했습니다. 이산 VQ-VIB 메시지는 연속 통신 대비 11.6~13.6pp의 fidelity 우위를 제공해 검증을 3~4배 빠르게 만드는 실용적 다리로 평가됩니다.



### Before the Pull Request: Mining Multi-Agent Coordination (https://arxiv.org/abs/2606.19616)
Comments:
          9 pages, 2 tables. LNCS format. Code, dataset, and mining toolkit: this https URL

- **Prior Approaches**: AIDev 같은 기존 연구는 pull request(논문에서는 PR) 결과(승인/거절, 속도 등)만 대규모로 관찰해 ‘speed–acceptance gap’을 설명하려 했습니다. 그러나 중복 작업이나 충돌 편집은 PR로 남지 않는 경우가 많아(버려진 중복, 레이스에서 승자만 흔적), pull-request-level 텔레메트리로는 조정과 신뢰의 공백이 잘 보이지 않습니다. 또한 일반적인 에이전트 평가는 단일 에이전트 성과에 치우쳐 다중 동시성에서의 사전 조정 문제를 직접 계량하기 어렵습니다.

- **Core Contribution**: 이 논문은 PR 이전에 벌어지는 ‘공유 작업에 대한 동시 에이전트의 claim–divide–collision’ 과정을 핵심 원인으로 지목하고, 이를 측정할 수 있는 서버리스 git-native 조정 인프라 grite를 제안합니다. grite는 중앙 서버 없이 git ref 안에 append-only, content-addressed(해시 기반), 선택적으로 서명된 event log를 남겨 조정 상태 자체를 데이터로 만듭니다. 그 결과 PR 이전 텔레메트리가 비가시였던 부분을 재현·채굴(mining)할 수 있게 합니다.

- **Technical Challenges**: 동시 에이전트가 공유 상태를 갱신할 때 충돌을 잃지 않으면서도(데이터 손실 방지) 상태가 모든 복제본에서 수렴해야 했습니다. 논문은 CRDT 기반 projection(이벤트를 순서와 무관하게 동일 상태로 재구성)으로 수렴성을 확보하고, advisory lease(TTL)로 상호배제를 부여해 잠금 충돌과 스타베이션을 포함한 행동 로그를 남깁니다. 또한 이벤트에 actor, timestamp, conflict/duplicate 같은 필드를 두고 서명·해시로 위변조를 막아, 실패 모드 탐지가 PR 기록만으로는 어려운 수준까지 자동 복구 가능하도록 설계했습니다.

- **Empirical Impact**: 실험에서는 조정 장치를 단계적으로 추가했을 때 중복·충돌·생산성이 어떻게 바뀌는지 정량화했습니다. advisory lease만으로도 goodput이 늘고 충돌 편집이 크게 줄지만, ‘동료가 이미 끝낸 작업을 다시 찾는 중복 rediscovery’는 여전히 많이 남아 duplicate-work을 0으로 만들지 못했습니다. 반면 locks+state(공유 완료 상태까지 사용)에서는 중복 작업 비율이 78%→0%로 떨어지고 유효 처리량(goodput)이 3배 이상 개선되며, 같은 이벤트를 어떤 순서로 받아도 복제본 상태가 바이트 단위로 동일해 데이터 손실도 방지됨을 확인했습니다. 더 나아가 PR 기록에서 사라지는 conflicting edits, lock starvation, race-to-close 같은 구체 실패 모드를 log에서 자동 감지·복구 가능하다는 점이 impact로 제시됩니다.



### Deontic Policies for Runtime Governance of Agentic AI Systems (https://arxiv.org/abs/2606.19464)
Comments:
          10 pages, 1 figure. To be published in the 2026 IEEE Symposium on Agentic Services which is part of the IEEE Conference on Web Services

- **Prior Approaches**: 기존 Agentic AI 거버넌스는 주로 LLM 바깥에서 per-invocation(행동 경계) 정책을 강제하는 방향으로 수렴했지만, 정작 “무엇이 허용/금지되는지(permit/prohibit)” 수준에 머무르는 경우가 많다. A2AS, Rego, Cedar, OPA 등은 허용·거부는 잘 다루지만, 허용 이후의 의무(obligations), 의무 면제(dispensations), 규칙 충돌 시 우선순위 같은 상위 거버넌스 구조를 기본 개념으로 제공하지 않는다.

- **Core Contribution**: AgenticRei는 deontic 정책(허가·금지·의무·면제)을 Rei 프레임워크 기반 OWL 온톨로지로 표현하고, LLM과 완전히 분리된 고성능 추론 엔진에서 런타임으로 평가한다. 또한 정책 평가 파이프라인을 도구 호출과 agent-to-agent 메시지(A2A) 모두에 동일하게 적용해 멀티 에이전트 환경의 거버넌스 결정을 일관되게 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 정책 요구사항이 단순 allow/deny를 넘어 의무 라이프사이클·면제 조건·충돌 해소 같은 “규칙에 대한 규칙(meta-policy)”까지 포함한다는 점과, (2) 도메인 개념 변화(예: 의료 데이터 하위 타입 증가)에 맞춰 정책을 유지보수 없이 확장해야 한다는 점이다. AgenticRei는 ⟨subject, action, resource⟩ 트리플로 행동을 추출한 뒤 RDFox 기반 OWL/RDFS 추론으로 온톨로지 계층을 자동 전개하고, metapolicy:RulePriority 같은 의미론적 우선순위로 충돌을 정리하며, 의무는 permission에 연결된 provision으로 실행 결과에 부착하는 방식으로 해결한다.

- **Empirical Impact**: 논문은 보안·프라이버시 거버넌스에서 의무·면제·충돌 우선순위·클래스 계층 추론이 기존 생산 엔진에서 거의 표현되지 못함을 예제로 보여준다. 또한 단일 호스트 프로토타입에서 쿼리당 전체 의사결정이 10ms 미만(추론 엔진 자체는 1ms 이하)으로 측정돼, 동기식 행동 경계 강제에 필요한 지연 시간 제약을 만족할 가능성을 제시한다.



### Human-like autonomy emerges from self-play and a pinch of human data (https://arxiv.org/abs/2606.19370)
Comments:
          10 pages

- **Prior Approaches**: 기존 self-play 강화학습은 시뮬레이션 경험만으로 정책을 키워 비용을 크게 줄이지만, 높은 성공 보상만으로는 인간이 수용할 ‘교통 관례’ 정렬이 보장되지 않아 ‘alien’한 주행 관습으로 수렴할 수 있다. 이를 막기 위해 reward engineering과 domain randomization을 반복적으로 설계하는 방식이 널리 쓰였지만, 작업량이 크고 취약하다는 한계가 있다. 한편 imitation learning(IL)은 보상을 직접 설계하지 않지만, 폐루프 배치에서의 분포 불일치 문제를 줄이려면 대규모 인간 데이터가 필요한 경우가 많다.

- **Core Contribution**: 이 논문은 human demonstration을 폐기하지 않고, self-play RL의 최소 안전 목표 보상 위에 behavioral cloning(BС) 기반 정규화 항으로 ‘anchor’ 역할을 부여한다. 즉 spiced self-play로 불리는 방식은 reward engineering 없이도 인간 궤적과의 교차 조율을 끌어올린다. 특히 Waymo Open Motion Dataset(WOMD)에서 30분 수준의 인간 데이터만으로도 효과가 크게 나타나며, 기존 IL과 비교해 데이터 요구량을 극적으로 낮춘다.

- **Technical Challenges**: 핵심 기술적 난제는 self-play가 만들어내는 비정적(상호 학습) 파트너 분포에서도, 소량의 인간 데이터가 정책의 충돌률·행동 현실성을 함께 개선하도록 ‘정규화가 어디에서 어떻게 작동하는가’를 안정적으로 설계하는 것이다. 논문은 PPO 기반 self-play에 KL 페널티를 더하되, KL이 온라인에서 에이전트가 실제로 방문하는 상태분포에서 anchor(BС)로 당기도록 구성해 오프라인 데이터 분포의 왜곡을 줄였다. 또한 시뮬레이션에서 human-replay, IDM(규칙 기반 공존자), self-play의 3가지 평가 셋업을 분리해 기여를 해석 가능하게 만들었다.

- **Empirical Impact**: 실험 결과, 30분~3시간의 human anchor 데이터를 spiced self-play에 더하면 충돌(특히 at-fault 충돌)과 안전성 지표가 크게 개선되며, 여러 환경 평가에서 SMART 계열 IL 대비 우위를 보인다. 또한 충돌 빈도뿐 아니라 충돌 심각도(Δv 기반)도 함께 낮춰, 인간과 함께 운행 시 위험을 줄이는 방향으로 정렬이 이동함을 확인했다. 의미 있는 점은 학습이 단일 consumer-grade GPU에서 약 15시간 내 end-to-end로 가능하고, 인간 데이터는 ‘주요 감독신호’가 아니라 ‘lightweight anchor’로만 쓰였는데도 성능이 크게 개선됐다는 것이다.




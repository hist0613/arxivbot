New uploads on arXiv(cs.CL)

### Vision-Default, Prior-Override: Causal Mechanisms of Perception-Knowledge Conflict in Vision-Language Models (https://arxiv.org/abs/2606.28273)
Comments:
          14 pages, 11 figures, 8 tables

- **Prior Approaches**: VLM은 시각 근거와 저장된 세계지식을 동시에 활용하지만, 충돌 상황에서 어떤 메커니즘으로 둘을 조정하는지는 기존 연구가 주로 행동 양상이나 상관 분석으로 설명해 왔다. 또한 충돌이 중후반 레이어에서 두드러진다는 국소화나 특정 attention head 후보 제시는 있었지만, “왜” 그런 결정을 내리는지에 대한 구성요소 수준의 인과 해명은 부족했다.
Activation-level 개입으로 모드를 전환할 수 있다는 점은 알려졌지만, 책임 회로가 필요 조건인지(필수성)까지 명확히 보여주기에는 한계가 있었다.

- **Core Contribution**: 이 논문은 activation patching을 residual stream, attention heads, MLP sublayers의 세 수준에 걸쳐 수행하고, 동시에 모델 구성요소 ablation과 mechanistic analysis를 결합해 시각-지식 충돌의 “인과 회로”를 특정한다. 그 결과 VLM은 기본적으로 visual grounding이 먼저 표면화되며, prior grounding은 네트워크 후반에 집중된 극소수 attention heads(전체의 2.5–4.8%)의 능동적 개입 없이는 잘 일어나지 않는 비대칭 구조임을 제시한다.
또한 이들 핵심 head는 (1) 정보를 흘려보내는 routing head와 (2) 정답 토큰을 residual stream에 직접 투사하는 writing head로 분해되며, 이 분해 양식이 서로 다른 VLM 계열과 스케일 전반에서 유지된다고 보고한다.

- **Technical Challenges**: 핵심 기술 난제는 프롬프트가 바뀌면 모델이 “어떤 정보원(이미지 vs 기억)”을 읽는지가 섞여버릴 수 있다는 점이다. 논문은 동일한 반사실 이미지에서 grounding 프롬프트만 교체해 두 모드의 대비를 만들고, 마지막 토큰 위치에서 특정 구성요소의 활성만 상호 치환하는 P2V( prior→visual )와 V2P( visual→prior ) 양방향 patching으로 구성요소별 인과 효과를 분리한다.
또한 necessity를 보이기 위해 promoting head/층을 0-공간으로 ablate하는 단독 및 그룹 ablation을 수행하고, writing 여부는 head 출력 차이를 logit 공간에 투사해 answer 토큰 hit rate로 검증한다.

- **Empirical Impact**: 실험은 Visual-Counterfact 데이터셋의 충돌 조건에서 수행되며, conflict가 없는 경우 86–96% 정확도를 보이던 모델들이 반사실 이미지+prior 프롬프트 조합에서는 17.7–55.7%로 크게 붕괴한다. 그럼에도 핵심 promoting attention heads를 제거하면 prior-grounded 예측이 68–96% 범위로 뒤집히는 반면, visual-grounded 예측은 0.8–7.5%만 변해 강한 인과 비대칭이 확인된다.
이 회로는 세 가지 VLM 계열(Qwen-VL, LLaVA-NeXT, PaliGemma)과 여러 모델 크기에서 일반화되며, 특히 visual grounding이 기본 경로로 작동하고 prior grounding이 후반 소수 head의 능동 override로 구현된다는 결론을 제시해 VLM 신뢰성 향상 및 안전한 모달 선택 설계에 직접적인 실마리를 제공한다.



### Cognitive Episodes in LLM Reasoning Traces Enable Interpretable Human Item Difficulty Prediction (https://arxiv.org/abs/2606.28186)
Comments:
          32 pages, 8 figures, 10 tables

- **Prior Approaches**: 기존 연구는 문항 텍스트 기반의 item-level 표현이나, 사람의 비싼 보정(calibration)에 의존해 인간의 문항 난이도를 추정하는 경우가 많았습니다. 그 결과 난이도가 어떤 인지적 과정에서 생기는지에 대한 증거가 제한적이며, 단순 추정에 머무는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 난이도를 문항 텍스트의 속성뿐 아니라, 문항이 유발하는 문제풀이 부담의 ‘관측 가능한 결과’로 재정의합니다. Large Reasoning Models(LRM)의 추론 흔적을 사람이 해석 가능한 인지 에피소드(episode) 시퀀스로 구조화하는 Epi2Diff(Episode to Difficulty) 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 난제는 LRM의 긴 reasoning trace를 그대로 쓰기보다, 기능적 문제풀이 상태로 묶어 해석 가능한 모델링 단위로 변환하는 것입니다. Epi2Diff는 trace 세그먼트를 인지적으로 근거 있는 episode로 군집화하고, reasoning scale·effort allocation·state transition 같은 에피소드-동적 특징을 간단한 표현으로 추출한 뒤 의미적 문항 표현과 결합해 난이도를 예측합니다.

- **Empirical Impact**: 4개의 실제 human difficulty 데이터셋에서 Epi2Diff는 fine-tuned 소형 언어모델, LLM in-context learning, supervised LLM adaptation 등 강력한 베이스라인을 일관되게 능가했습니다. 특히 SAT에서 만든 분류 벤치마크에서는 supervised LLM fine-tuning 대비 평균 상대 8.1% 향상을 보였고, 난이도가 높은 문항이 단순히 더 긴 응답이 아니라 더 노력적·반복적·구현 중심의 episode dynamics를 유발한다는 분석도 제시했습니다. 이는 교육 측정에서 추론 모델의 인지적 과정 표현을 예측 가능하고 해석 가능하게 활용할 수 있음을 보여줍니다.



### From Tokens to States: LLMs as a Special Case of World Models and the Continuous Path Beyond (https://arxiv.org/abs/2606.28127)
Comments:
          10 pages, 6 figures, 1 table

- **Prior Approaches**: 기존 논의는 LLM은 토큰을 예측하고 world model은 현실을 시뮬레이션한다는 식으로 이분법을 세우는 경향이 있다. LeCun(2022)은 autoregressive 토큰 예측을 벗어나 latent-space 아키텍처인 JEPA로 가야 한다고 주장했지만, 본 논문은 이 구도가 지나치게 이분법적이라고 본다. 또한 world model은 특정 모델이 아니라 state/action/transition의 형식적 클래스라는 점에서, LLM을 단순히 ‘대체’ 대상으로 보기 어렵다고 정리한다.

- **Core Contribution**: 이 논문은 두 가지 주장을 전면에 둔다. 첫째, LLM은 world model의 특수한 퇴화(degenerate) 케이스로서 world model이 LLM을 엄밀히 포함한다는 형식적 포함관계를 제시한다(LLMs ⊂ World Models). 둘째, NTP부터 JEPA까지 이어지는 연속 스펙트럼이 존재하며 다중 토큰 예측, future-summary 예측, next-latent 예측이 그 중간 지점에 이미 자리한다고 주장한다.

- **Technical Challenges**: 스펙트럼을 따라 제약을 하나씩 푸는 과정에서 가장 큰 난점은 결국 ‘데이터 절벽’과 ‘연속 상태 예측용 아키텍처 적합성’으로 수렴한다. NTP~중간 단계까지는 여전히 internet-scale self-supervised 텍스트로 학습 가능하지만, JEPA로 가면 observation-action-next observation처럼 계측된 환경의 쌍 데이터가 필요해 표본이 급감한다. 아키텍처 측면에서도 transformer가 이산 토큰에 맞춰 공진화된 만큼, latent 벡터의 연속 예측에는 diffusion-style 헤드 등 새로운 프리미티브가 필요할 수 있다고 남긴다.

- **Empirical Impact**: 기계적 해석(mechanistic interpretability) 결과들이 LLM 내부에 세계상태(world state) 표현이 숨은 활성값에 형성됨을 뒷받침하며, 토큰 예측이 ‘표현 자체’가 아니라 인터페이스일 수 있음을 보여준다(OthelloGPT, chess LM, Llama-2 사례). 또한 다중 토큰 예측, future-summary, next-latent 계열의 최신 접근들이 스펙트럼 중간역에서 성능/계획을 개선하는 것으로 보고돼, ‘완전한 전환’보다 단계적 이동이 유망하다는 관점을 강화한다. 결론적으로 이 논문은 LLM을 버릴지의 문제가 아니라, 어떤 태스크가 어떤 단계의 world-model 제약 완화를 요구하는지를 묻는 프레임을 제안한다.



### Mechanism-Driven Monitors for Preemptive Detection of LLM Training Instability (https://arxiv.org/abs/2606.28116)
- **Prior Approaches**: 기존 훈련 안정성 모니터링은 loss, gradient norm, weight norm 같은 전역 곡선과 지표에 의존해 이상 징후가 너무 늦게 드러나는 경우가 많다. attention에서는 max-logit 클리핑·QK-clip·엔트로피/스펙트럼 기반 진단이 제안됐지만, Flash Attention(FA)처럼 커널 수정이나 재계산이 필요한 신호는 운영 환경에서 적용이 까다롭다.

- **Core Contribution**: 이 논문은 “모듈의 기능과, 고장이 최초로 남길 수 있는 계산 지점”에서 내부 모니터를 설계하는 mechanism-driven monitoring을 제안한다. 낮은 정밀도의 Flash Attention에 대해서는 QK bilinear 분해의 첫-order 신호로서 spectral entropy를 감시하고, MoE에서는 expert 선택의 축이 붕괴하는지를 라우터 weight 유사성과 라우팅 엔트로피로 조기 판별한다.

- **Technical Challenges**: 주요 난관은 고장 이후 수천 스텝 동안 loss/gradient가 정상처럼 보이는 지연 현상을 뚫고, production FA처럼 activation 재계산 없이도 계산 가능한 진단량을 만드는 것이다. 저정밀 FA는 업데이트 ΔWΔW의 스펙트럼 붕괴를 보이되, Q와 K 요인의 독립 감시는 상관 드리프트를 놓칠 수 있어 QK increment를 first-order(Δ2)와 second-order(Δ3)로 분해한 뒤 Δ2의 singular spectrum entropy를 핵심 모니터로 삼는다.

- **Empirical Impact**: 결함 주입 실험에서 저정밀 attention, 큰 learning-rate, 결함 결합 모두 대해 해당 신호가 loss divergence보다 수천 스텝 먼저 뚜렷한 ‘서로 다른 서명’을 형성하며 조기 트리거가 가능함을 보였다. 또한 ΔW 기반 전력 폭주나 stable-rank 같은 지표보다 스펙트럼 엔트로피/분해 기반 모니터가 더 민감하고 설명 가능하다는 점을 실증적으로 뒷받침한다.



### MultiHashFormer: Hash-based Generative Language Models (https://arxiv.org/abs/2606.28057)
Comments:
          Under review

- **Prior Approaches**: 기존 연구는 임베딩 행렬이 어휘 크기에 선형으로 커지는 문제를 줄이기 위해, 여러 토큰을 하나의 벡터에 해시로 압축하는 방식(해시 임베딩)을 제안해 왔다. 다만 이런 many-to-one 충돌은 다음 토큰을 생성해야 하는 causal LM에서 그대로 치명적으로 작동해 적용이 어려웠다.

- **Core Contribution**: 이 논문은 hash-based autoregression을 가능하게 하는 MultiHashFormer 프레임워크를 제안한다. 각 토큰을 다중 독립 해시함수로 만든 ‘해시 시그니처(짧은 이산 해시 ID 시퀀스)’로 표현하고, Hash Encoder가 이를 하나의 잠재 벡터로 압축한 뒤 Transformer decoder로 처리하며, Hash Decoder가 다음 토큰의 해시 시그니처를 다시 생성한다.

- **Technical Challenges**: 핵심 난관은 해시 충돌을 causal 생성 과정에서도 허용 가능한 수준으로 통제하면서, 시그니처를 효율적으로 인코딩/디코딩하는 것이다. 저자들은 해시 시그니처를 여러 해시 함수의 조합으로 구성해 토큰별 식별성을 확보하고, 인코더-디코더 사이에서 시그니처↔잠재벡터↔다음 시그니처로 정보를 왕복시키는 구조를 설계해 해결했다.

- **Empirical Impact**: 100M, 1B, 3B 파라미터 규모에서 MultiHashFormer가 표준 Transformer LM을 여러 벤치마크에서 일관되게 능가함을 보였다. 또한 언어별 어휘를 확장해도 파라미터 풋프린트가 일정하게 유지되며 별도 수정 없이 멀티링구얼 확장을 처리할 수 있음을 실험으로 확인해, 효율적 대규모 생성 모델 설계에 의미 있는 진전을 제공한다.



### Can LLMs Judge Better Than They Generate? Evaluating Task Asymmetry, Mechanistic Interpretability and Transferability for In-Context QA (https://arxiv.org/abs/2606.28050)
Comments:
          18 pages

- **Prior Approaches**: LLM-as-a-Judge와 self-evaluation 파이프라인은 생성보다 평가가 더 쉽다는 전제를 바탕으로, 생성된 답을 같은 모델이 더 정확히 판정할 수 있을 것으로 기대한다. 기존 연구들은 generation과 judgment를 비교했지만 open-domain에서 parametric knowledge가 섞이면서 “평가가 쉬운지”를 통제해 검증하기 어렵다는 한계가 있었다. 또한 평가가 실제로 답을 재검증하는지, 아니면 형태적 힌트에 의존하는지에 대한 기계적(메커니즘) 설명은 충분히 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 in-context QA 설정에서 context를 유일한 정보원으로 고정하고, 모델이 자신이 생성한 답을 그대로 판정하도록 하여 GA–EA(생성 정확도 vs 자기평가 정확도) 격차를 통제된 방식으로 측정한다. SQuAD 2.0, DROP, HotpotQA, MuSiQue의 네 벤치마크에서 대체로 “평가가 생성보다 쉽지 않다”는 결과를 제시하며, 특히 MuSiQue만 예외적으로 평가 우위가 나타난다고 보고한다. 더 나아가 attention 기반 분석과 LoRA 미세조정 전이 실험으로 이 비대칭이 단순한 학습 산물이 아님을 확인한다.

- **Technical Challenges**: 핵심 기술적 난제는 GA와 EA를 일관된 기준으로 비교하면서(oracle), 평가가 실제로 context를 읽어 재검증하는지 추적하는 것이다. 연구진은 같은 인스턴스에서 두 태스크(생성, 그 생성 답의 “Correct/Incorrect” 판정)를 연속 실행하고, GPT-4o를 oracle로 써 외부 기준을 고정했으며, last-token attention을 span 단위로 계량하는 메커니즘 분석을 수행한다. 또한 generation/Evaluation의 모수 구조가 공유되는지 보려 LoRA-Gen/LoRA-Eval/LoRA-Both를 학습하고, evaluation 학습에는 문맥 없는 환각 기반 hard negative를 구성해 “그럴듯한 틀린 답을 판별”하도록 설계한다.

- **Empirical Impact**: 실험 결과는 세 벤치마크(SQuAD 2.0, DROP, HotpotQA)에서 generation 정확도가 self-evaluation보다 높고(즉 Δ=EA−GA가 음수), MuSiQue에서만 평가 우위(Δ가 양수)가 관찰되는 식으로 나타난다. attention 분석에서는 평가 태스크가 context에 3–5배 적게 주의를 주고 candidate answer 슬롯도 거의 읽지 않는 경향이 확인돼, 많은 경우 평가가 “재검증”이 아니라 구조적 단서에 의존함을 시사한다. LoRA 결과는 LoRA-Eval이 오히려 generation 성능을 떨어뜨리고(과도한 보수성), LoRA-Gen은 evaluator가 over-acceptance(과수용)로 기울어지는 등 방향성이 뚜렷해, self-evaluation 파이프라인의 핵심 가정(평가가 더 쉽다)을 흔드는 실증 증거로 평가된다.



### A Tree-of-Thoughts Inspired Hybrid Approach for Legal Case Judgement Summarization using LLMs (https://arxiv.org/abs/2606.28044)
Comments:
          Accepted at ICAIL 2026

- **Prior Approaches**: 기존 연구는 법률 판례 요약에서 전통적인 추출형(extractive) 또는 생성형(abstractive) 요약을 중심으로 접근해 왔다. 다만 추출-생성의 장점을 함께 쓰는 하이브리드(extractive-abstractive) 방식은 상대적으로 덜 탐구되어 왔다.

- **Core Contribution**: 이 논문은 트리-오브-토츠(tree-of-thoughts)에서 영감을 받은 추출-생성(extractive-abstractive) 요약 접근을 제안한다. 목표는 판례 요약에서 근거가 되는 구절은 추출하면서도, 최종 서술은 더 자연스럽게 생성하도록 하는 프롬프트 설계를 제공하는 것이다.

- **Technical Challenges**: 핵심 과제는 추출에 기반한 사실성 유지와 생성의 응집성·일관성을 동시에 확보하는 프롬프트 구성이었다. 저자들은 트리-오브-토츠 영감을 활용해 요약 후보를 구조적으로 탐색하도록 설계하고, 이를 통해 추출-생성의 균형을 맞추는 방식을 구현했다.

- **Empirical Impact**: 실험은 DeepSeek과 LLama 두 가지 LLM을 사용해 진행되었으며, 추출형·생성형·추출-생성형 프롬프트를 비교했다. 결과적으로 트리-오브-토츠 영감을 반영한 추출-생성 프롬프트가 다른 프롬프트 유형보다 더 나은 요약을 제공하는 것으로 나타나, 법률 문서 요약에서 하이브리드 프롬프트의 실효성을 보여줬다.



### The Signal-Coverage Matrix: Stratifying Type and Semantic Errors in Statement Autoformalization (https://arxiv.org/abs/2606.28013)
- **Prior Approaches**: 기존 LLM 자동 포멀라이제이션은 Lean elaborator의 에러 피드백으로 정답에 가까워지도록 하는 방식이 주류였지만, TC%가 약 75% 전후에서 정체되는 문제가 있었다. 또한 의미 충실도 SF%는 back-translation이나 GTED 같은 서로 다른 judge로 따로 측정됐고, TC% 상승이 어떤 오류 유형을 실제로 고치는지(타입 vs 의미) 분해해 설명한 연구는 부족했다.

- **Core Contribution**: 이 논문은 Lean elaborator의 통과/실패 신호와 의미 동등성 판단(동등/비동등)을 교차해 결과를 TS(진성 성공), TO(타입만 실패), SO(의미만 실패), BF(둘 다 실패) 네 칸으로 분류하는 signal-coverage matrix를 제안한다. 그 덕분에 TC% 같은 단일 스칼라 대신, 각 방법이 어떤 칸의 오류를 얼마나 이동시키는지로 기여를 해석할 수 있다.

- **Technical Challenges**: 핵심 기술적 난제는 elaborator는 타입 오류만 민감하고 의미 오류에는 무감하다는 점, 반대로 의미 judge는 타입 실패를 잘 용서해 서로 상보적인 신호라서 ‘무엇이 고쳐졌는지’ 추적이 어려웠다는 것이다. 논문은 Claude Opus 4.7(정밀 동등성 판정)과 GTED를 2중 judge로 캘리브레이션하고, SAF처럼 typed JSON IR→결정적 번역을 사용해 표면 형태 변형이 SF 판정 차이를 얼마나 유발하는지까지 원인 분리했다.

- **Empirical Impact**: ProofNet#(186문제)에서 DeepSeek V4-Pro 기준으로, elaborator 피드백 기반 방법들은 Vanilla 대비 TS가 약 +34~+36pp 늘었고 그중 상당 부분이 TO 같은 타입-영역 회복(대략 23/36 ≈ 64% 차지)에서 나왔다. 동시에 의미 전용 오류(SO)는 거의 그대로 유지되며, judge 불일치는 elab-feedback에서 더 커져(최대 26~37pp) 그 원인이 GTED가 elaborator 강제 리라이트에 과하게 민감하기 때문임을 보여준다.



### Dialogue to Detection: A Multimodal Hybrid NLP Pipeline for Insurance Fraud Detection (https://arxiv.org/abs/2606.28002)
Comments:
          10 pages, 8 figures, 2 tables

- **Prior Approaches**: 기존 보험 사기 탐지는 주로 text-only 공개 데이터에 의존해 BERT류 분류기나 규칙 기반 점검을 붙이는 방식이 많았다. 하지만 FNOL(최초 청구) 단계의 음성 통화는 프라이버시·접근성 문제로 쌍(pair) 데이터가 거의 공개되지 않아, 음성의 부가 정보까지 함께 검증하는 멀티모달 연구가 제한돼 왔다.

- **Core Contribution**: 이 논문은 FNOL 상황을 모사하는 합성 multimodal 파이프라인을 제안한다. GPT-2로 에이전트-고객 대화 대본을 만들고, xTTS로 2인 화자 오디오를 합성한 뒤, WhisperX의 ASR·diarisation 결과와 텍스트의 NER·RAG 검색, 그리고 Resemblyzer speaker embedding을 결합해 해석 가능한 fraud risk score를 산출한다.

- **Technical Challenges**: 핵심 난제는 실제 통화 수준의 대화 구조·음향 특성과, 화자 분리·전사 오류 같은 현실 조건을 합성 데이터에 재현하는 것이다. 논문은 다양한 디코딩(temperature·nucleus/top-k)과 2채널 오디오 합성→WhisperX 기반 분리→고정밀 식별자 추출(Regex+NER)→텍스트 유사도(RAG)·음성 재사용(embedding 유사도) 결합, 그리고 가중치 기반 룰 점수로 false positive를 완화하는 설계를 택했다.

- **Empirical Impact**: 검증 결과 합성 데이터 내부 일관성과 구성 요소별 처리 성능(전사 WER, 화자 분리/특징 추출 정밀도·재현율·F1 등)을 단계적으로 확인했으며, 텍스트 분류(BERT-RAG)는 합성 홀드아웃에서 사실상 100% 수준의 성능을 보였다. 다만 변동성이 제한된 합성 환경이라 낙관적일 수 있어, 실제 데이터로의 일반화 강화를 위해 내러티브·스피커 다양성 확대와 k-fold 및 익명 실데이터 파인튜닝이 후속 과제로 제시됐다.



### ToxiREX: A Dataset on Toxic REasoning in ConteX (https://arxiv.org/abs/2606.27981)
- **Prior Approaches**: 기존 독성(toxicity) 분석 데이터셋은 주로 단일 언어 또는 발화 단위의 라벨에 초점이 맞춰져, 댓글 스레드처럼 맥락이 바뀌면 의미도 달라지는 implicit 독성을 충분히 다루기 어렵다. 또한 라벨이 고정된 분류 체계에 머물러 있어, ‘무엇이 왜 독성으로 이어지는지’를 reasoning 구조로 설명하기가 제한적이었다. 다국어·대화 맥락·추론 기반 구조화를 동시에 만족하는 데이터셋은 상대적으로 희소했다.

- **Core Contribution**: 이 논문은 맥락을 보존하는 다국어 대화 데이터셋 ToxiREX(Toxic REasoning in ConteXt)를 제안한다. Reddit 댓글 스레드와 함께, 이전 연구에서 만든 체계적 toxic reasoning schema에 따라 댓글이 함의하는 내용을 structured하게 주석해 암묵적이고 상황 의존적인 독성을 포착한다. 또한 기존 toxicity taxonomies로의 매핑 가능성을 함께 지원한다.

- **Technical Challenges**: 핵심 난제는 스레드의 문맥을 깨지 않으면서 implicit 독성을 schema 기반의 계층적 예측으로 일관되게 주석화하는 것이었다. 논문은 스레드 context-preserving 전처리를 수행하고, 상용 LLM으로 학습 세트 12.5만 건을 라벨링하되, 테스트 세트 약 3천 건은 원어민이 주석해 해석의 정당한 대안 가능성까지 반영했다. 더불어 hierarchical, schema-based 예측을 공정하게 평가하기 위한 평가 전략을 설계하고, prompting과 fine-tuning 기반 베이스라인을 제공한다.

- **Empirical Impact**: 실험 결과 모델은 랜덤보다 성능이 좋지만 아직 개선 여지가 큰 것으로 나타나, 이 과제가 단순 분류가 아니라 추론·맥락 이해가 필요한 어려운 작업임을 보여준다. 테스트 주석에서 보이는 불일치는 noise라기보다 방어 가능한 대안 해석을 반영하는 경우가 많았고, 이는 schema 기반 평가의 필요성을 뒷받침한다. ToxiREX는 여러 언어와 대화 맥락, implicit toxicity를 동시에 포함하면서 독성 추론 스키마로 풍부한 구조적 주석을 제공한 첫 데이터셋으로서 후속 연구의 기준점과 확장 경로를 제공한다.



### From Black-Box to Clinical Insight: A Multi-Stage Explainable Framework for Speech-Based Cognitive Impairment Detection (https://arxiv.org/abs/2606.27973)
Comments:
          Accepted to Interspeech 2026

- **Prior Approaches**: 기존 음성 기반 인지장애 탐지는 transformer 성능을 끌어올렸지만, 모델이 왜 그런 결론을 내리는지 설명이 ‘블랙박스’로 남아 임상 도입이 어렵다는 지적이 컸다. XAI 연구는 SHAP/LIME 같은 토큰 중요도나 일부 수작업 언어 지표에 집중했지만, 임상의가 실제로 쓰는 인지-언어 메커니즘으로의 연결이 약하고 수치 해석이 필요해 사용성이 떨어졌다.

- **Core Contribution**: 이 논문은 transformer 예측을 임상적으로 근거 있는 ‘서술형 설명’으로 바꾸는 multi-stage explainability 프레임워크를 제안한다. SHAP 토큰 기여(서브워드→단어 수준 집계)와 임상 지향 언어 특징(어휘 풍부도, 구문 복잡도, 의미 응집성 등)을 결합한 뒤, LLaMA-3.1-70B-Instruct로 4단계 LLM 추론 파이프라인을 돌려 통합된 임상용 내러티브를 생성한다.

- **Technical Challenges**: 핵심 난제는 (1) 서브워드 토큰화를 가진 transformer에서 perturbation 기반 SHAP을 임상 언어 단위로 해석 가능하게 만드는 것과 (2) 토큰/수치 결과를 ‘왜 그런 인지 문제가 보이는지’로 번역하는 일이다. 논문은 모델 래퍼로 SHAP 입력·확률 출력을 구성하고 계층적 집계로 word-level 설명을 만들었으며, LLM에 인지-언어 차원별 임상 정의를 프롬프트로 제공해 추론이 의미를 벗어나지 않게 제한했다.

- **Empirical Impact**: 성능 측면에서 SpeechCARE Adaptive Gating Fusion(SpeechCARE-AGF) 기반 스크리닝 모델은 NIA PREPARE에서 AUC 86.83%, F1 72.11%를 보였다. 임상 유효성 평가에서는 70개 샘플에 대해 2명의 1차 진료의가 블라인드로 검토해 98% 케이스에서 높은 일치(κ=0.85)를 보였고, 사용성 척도 SUS는 82/100으로 임상 워크플로 통합 가능성을 시사했다.



### An Empirical Analysis of Factual Errors in Human-Written Text and its Application (https://arxiv.org/abs/2606.27959)
- **Prior Approaches**: 기존 Factual Error Detection(FED)은 주로 LLM의 환각을 겨냥해 왔고, 사람이 쓴 글에서 발생하는 사실 오류 검출은 상대적으로 덜 다뤄졌다. 또한 기존 벤치마크는 사람의 교정/정정에서 반복적으로 나타나는 오류 유형을 충분히 포착하지 못했다.

- **Core Contribution**: 이 논문은 신문 기사 정정 사례를 분석해 사람에 의해 유발되는 사실 오류의 분류체계( taxonomy )를 먼저 정립한다. 특히 kanji misconversions(한자 오변환)과 numeral classifier errors(수량사 관련 오류)처럼 기존 hallucination 벤치마크에서 덜 다뤄진 범주를 강조한다.

- **Technical Challenges**: 분류체계를 실제 검출 성능 평가로 연결하는 것이 핵심 과제였고, 이를 위해 현실적인 합성 test cases와 실제 정정 데이터를 구성해 vanilla LLM의 FED 능력을 시험했다. 또한 오류 난이도별로 성능 차이를 들여다보기 위해 detection difficulty 관점의 세부 분석을 수행했다.

- **Empirical Impact**: 실험 결과, GPT-5.4 같은 고성능 LLM도 합성 평가 데이터에서 단어 단위 F1이 52%에 그쳐 FED가 매우 어렵다는 점을 실증했다. 더불어 오류 난이도 분석을 통해 현재 FED의 한계 상태를 구체적으로 파악할 수 있어, 향후 사람 유발 오류에 맞춘 벤치마크 및 탐지 연구 방향을 제시한다.



### VASAE: Naming SAE Dictionary Directions with Vocabulary-Aligned Anchoring (https://arxiv.org/abs/2606.27941)
Comments:
          14 pages, 7 figures. Accepted to the 2nd Workshop on Compositional Learning at ICML 2026

- **Prior Approaches**: 기존 Sparse Autoencoder(SAE)는 Transformer 잔차 스트림을 희소 코드로 분해하지만, 학습된 사전(dictionary) 특징의 토큰 이름은 대부분 훈련 후 문맥을 훑거나 별도 해석 절차로 ‘사후적으로’ 붙인다. 이 방식은 특징이 토큰 임베딩 공간과 어떤 기하학적 관계를 갖는지 학습 과정에서 직접 연결되지 않아, 특징-토큰 대응이 불안정하거나 약해질 수 있다.

- **Core Contribution**: 본 논문은 Vocabulary-Aligned Sparse Autoencoder(VASAE)로, SAE 특징을 토큰 임베딩 방향에 ‘훈련 중’ 소프트로 정렬(anchoring)시키고 각 특징에 고유한 intrinsic token name을 부여한다. 특징의 이름은 해당 특징 벡터와 가장 가까운 토큰 문자열(임베딩 기준 nearest-token)으로 정의되어, 사후 해석 의존도를 낮추는 것이 핵심이다.

- **Technical Challenges**: 어려움은 SAE 사전의 기하(학습된 decoder 방향)와 어휘(vocabulary) 임베딩의 기하가 기본적으로 분리돼 있어, 특징이 복원에는 기여하지만 어떤 토큰 방향과는 멀 수 있다는 점이다. VASAE는 기존 복원 목적에 더해 각 특징이 자신의 nearest-token 임베딩에 가까워지도록 하는 anchor loss를 추가하고, top-k 희소 코드 제약과 함께 최적화한다.

- **Empirical Impact**: 실험에서 VASAE는 표준 SAE와 비교해 복원 품질(분산 설명/예측 손실 보존)을 크게 해치지 않으면서, GPT-2-small의 경우 층 0–10에서 강한 정렬 기준(nearest-token alignment score 0.8) 이상의 특징 비율이 약 90% 수준으로 나타난다. Llama-3.1-8B에서도 얕은 층과 중간 층은 강하게 정렬되지만 최종 대표 층에서는 정렬이 제한적이며, mean sparse code를 뺀 이후 case study에서 남은 intrinsic token name들이 입력 주변 토큰들과 관련성이 큼을 관찰해 ‘훈련 중 특징-토큰 연결’ 가능성을 시사한다.



### Triadic Werewolf: A Jester Role for Multi-Hop Theory of Mind in LLMs (https://arxiv.org/abs/2606.27909)
- **Prior Approaches**: 기존 Werewolf류 사회추론 벤치마크는 Villagers vs Werewolves처럼 두 진영(dyadic)으로 나뉘어, 관측 신호가 숨은 역할을 한쪽으로만 강하게 밀어주는 문제가 있었다. 이 구조에서는 언어적 사전지식/표면 휴리스틱(예: “의심스러워 보이면 퇴출”)만으로도 점수를 높일 수 있어, 진짜 theory-of-mind(ToM) 추론 여부가 흐려진다.

- **Core Contribution**: 논문은 Werewolf에 Jester(제스터)를 추가해 3자(삼자) 인센티브 구조를 만든다. Jester는 “의심받을수록” 이롭지만, 정작 승리는 자신이 투표로 퇴출될 때 발생하므로 동일한 관측 신호가 서로 반대의 최적 행동을 요구하게 된다.

- **Technical Challenges**: 핵심은 관측 신호(상대의 peer suspicion)가 곧바로 정답 행동(누굴 내쫓을지)으로 연결되지 않게 설계하고, 실제로 모델이 그 모순을 multi-hop ToM으로 풀어내는지 계량화하는 것이다. 이를 위해 10인 3자 벤치마크에 bidding-based debate 프로토콜과 self-learning 루프(ON/OFF)를 결합하고, 경기별/발화별 의심도·기만 유형·투표 결과로 추론 실패 패턴을 분해해 측정했다.

- **Empirical Impact**: 60게임 평가에서 Jester는 약 55%대 승률로 관측되며, Werewolves는 20%를 넘지 못했다. GPT-4.1의 Werewolves는 day 1에 60–70% 확률로 Jester를 “투표로 퇴출”하는데, 이는 그들 팀의 자멸적 행동으로 나타났고(엄밀히 말해 self-defeating vote), self-learning은 모델에 따라 이 경향을 완화/악화시키되 Jester가 특히 이득을 얻는 쪽(다른 진영이 cue를 잘못 읽는 상황)을 강화했다.



### A Study of Temporal Fusion Strategies for Named Entity Recognition in Historical Texts (https://arxiv.org/abs/2606.27881)
- **Prior Approaches**: 기존에는 역사 텍스트 NER에서 시간에 따른 개체 표면형 변화·등장/소멸·중의성 증가를 다루기 위해 데이터 증강이나 샘플링 등 데이터 중심 접근이 주로 사용돼 왔다. 또 time vectors, timestamp-aware pretraining, temporal graphs, dynamic knowledge editing 같은 시간 표현/지식 편집 기법이 제안됐지만, 토큰 분류형 NER 아키텍처에 시간을 구조적으로 어디·어떻게 결합하는지는 체계적으로 비교되지 않았다.

- **Core Contribution**: 이 논문은 Transformer 기반 NER에 출판 연도 같은 temporal metadata를 ‘구조적으로’ 주입하는 방법을 설계·비교한다. early fusion(인코딩 전/중)과 late fusion(인코딩 후)로 나누고, absolute와 time-distance(기준 연도 대비 상대 거리)라는 시간 표현 방식까지 함께 실험하며 어떤 결합이 더 견고한지 정리한다.

- **Technical Challenges**: 핵심 과제는 시간 정보를 토큰 수준 추론에 실질적으로 반영하되, 잡음이 큰 OCR·다국어 변이 환경에서도 성능이 흔들리지 않게 만드는 것이다. 이를 위해 cross-attention, adapters(경량 모듈), concatenation, FiLM-like modulation처럼 연도 임베딩을 다양한 지점에 결합하는 경량 fusion 전략을 구현하고, gold-year를 직접 쓰지 않도록 probing(추론 시 랜덤 year 주입)으로 내부 temporal internalisation 여부를 점검한다.

- **Empirical Impact**: HIPE-2020의 프랑스/독일 역사 데이터에서 late fusion 전략이 전반적으로 더 robust하고 시간 일반화 성능이 좋았으며, 특히 early/noisy 구간에서 이득이 두드러졌다. 또한 probing과 t-test 결과는 대부분의 개선이 연도 전반에 미묘하게 나타나지만, late-cross-attention은 baseline 대비 유의미한 차이를 보이는 등 구조적 시간 결합이 실제로 효과가 있음을 시사한다.



### Learning Complementary Action Modeling from Automotive Maintenance Instructions (https://arxiv.org/abs/2606.27808)
Comments:
          Preprint. 11 pages, 4 figures

- **Prior Approaches**: 기존 절차 문서 이해 연구는 개체 상태 추적이나 절차 순서/구조를 중심으로 발전했지만, 작은 행동 표현 차이가 절차적 방향(설치 vs 제거 등)을 뒤집는 상황은 충분히 반영하지 못했습니다. 또한 정보추출·semantic matching은 대체로 표면 유사도나 대규모 후보 매칭에 의존해, 어휘 중복이 높은데도 실제 작업이 반대가 되는 CAM 문제를 오판할 수 있습니다. 생성형/스크립트 기반 절차 모델도 역행·상보 행동을 정밀하게 “행동 구문만” 제어하는 목표로는 다루기 어려웠습니다.

- **Core Contribution**: 논문은 Complementary Action Modeling(CAM)을 정의하고, 유지보수 지시문에서 행동 구문(action phrase)만 바꿔 절차적 대응(complementary procedural counterpart)을 찾거나 생성하는 문제로 공식화했습니다. 핵심은 문장의 나머지 맥락은 보존하되, 행동 표현의 미세한 어휘 단서가 절차 관계를 결정한다는 점을 학습 목표에 반영한 것입니다. 또한 CAM을 단순 유사도/패러프레이즈/모순검출과 구분되는 별도 과제로 정리해 평가 관점도 재구성했습니다.

- **Technical Challenges**: CAM의 어려움은 (1) 표면 유사도와 상보성을 분리해야 하고, (2) 생성 모델이 문장 전체를 자연스럽게 다시 쓰는 대신 행동 구문만 통제해야 하며, (3) 평가에서 행동 변환의 정합성을 별도 축으로 측정해야 한다는 데 있습니다. 저자들은 이를 위해 관계 인식 retrieval(후보 매칭)과 action-level controlled generation(제어 생성)을 함께 실험하고, 생성에는 토큰 중복 점수 외에 임베딩 공간 대조 정규화와 human evaluation(의미적 상보성/구성 요소 보존)을 추가했습니다. 데이터는 독일 OEM 매뉴얼에서 공정(process) 버킷을 유지한 채, 설치/제거 등 보완 행동의 어휘 반대쌍을 룰 기반으로 정렬하고 일부는 사람이 검증해 1:1 고신뢰 페어를 구축했습니다.

- **Empirical Impact**: 후보 매칭에서는 closed-set 관계 학습이 가능하며 groups_native 설정에서 최고 시스템이 약 0.74 Recall@1과 0.83 MRR 수준을 보였습니다. 생성 실험에서도 mBART-large-50이 BLEU 62.96, ROUGE-L 0.797로 가장 좋았고, 자동 overlap 점수가 높더라도 행동 구문 변환 오류가 남을 수 있어 human 평가가 필요함을 보여줬습니다(샘플 100개 중 95개는 의미적으로 상보 판정). 또한 학습된 체크포인트를 다른 차종 문서에 적용했을 때 수동 검증 기준 수용률 64.5%로, CAM이 차량 문서 전반으로도 전이될 가능성을 시사합니다. 결론적으로 CAM은 “문장 유사도”가 아니라 “행동-절차 관계의 상보성”을 잡는 모델링이 되어야 하며, 그에 맞춘 다중 관점 평가가 중요하다는 메시지를 강화합니다.



### Position Bias Correction is Insufficient for One-Pass Attention Sorting (https://arxiv.org/abs/2606.27793)
- **Prior Approaches**: 롱컨텍스트 LLM에는 ‘lost-in-the-middle’로 대표되는 위치 편향이 있어, 문맥의 중간 정보가 덜 활용되는 문제가 반복적으로 관찰된다. 이를 문서 재정렬로 완화하려는 Attention Sorting은 생성 중 어텐션 패턴을 보며 문서를 여러 번 정렬해 성능을 끌어올리지만, sort-and-generate를 여러 회 수행해 지연 비용이 커진다. 기존 연구들은 위치 편향을 토큰 단위 보정이나 위치 인코딩 개선(예: RoPE 변형, 컨텍스트 확장)으로 다루었으나, 문서 레벨 재정렬의 반복을 줄이는 관점에서는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 위치 편향이 반복 정렬의 핵심 병목일 것이라는 가설을 세우고, Debiased One-Pass Attention Sorting(편향 보정 1회 정렬)을 제안한다. 같은 프롬프트에서 얻은 ‘저어텐션 다수 문서’로부터 프롬프트별 position-bias curve를 추정한 뒤, raw attention score를 (차감 또는 나눗셈으로) 보정하여 단 1회 정렬로 iterative sorting에 근접하려고 한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 문맥 내에서 진짜 관련 문서와 편향만 반영한 주의를 분리해 bias curve를 추정하는 것과 (2) 보정이 실제로 정렬 순위를 개선하도록 점수 스케일을 안정적으로 만드는 것이다. 논문은 상위 α 비율 문서를 트림해 추정 신뢰도를 높이고, 위치 인덱스를 B개 구간으로 빈닝한 뒤 보정 모드(additive/divisive)를 선택해 debiased score로 정렬을 수행한다. 또한 모델별로 reordering 전략과 하이퍼파라미터를 별도 튜닝해 단일 패스의 계산량(정렬 1회+생성 1회)을 유지한다.

- **Empirical Impact**: SynthWiki@28K에서 LLaMA-2-7B-32K-Instruct는 debiasing이 raw single-pass sorting과 결과가 완전히 같아(94.83% containment accuracy) ‘위치 편향 보정만으로 반복의 이득을 대체’할 수 없음을 보여준다. 반면 YaRN-Llama-2-7b-64k에서는 debiasing이 +8.67pp 개선했지만, iterative sorting(k=5) 대비 14.84pp 뒤처져 격차의 37%만 메우는 데 그쳤다. 결론적으로 위치 편향 보정은 high-bias 모델에서 선택적으로 유의미하지만, iterative sorting이 제공하는 추가 이득(어텐션 컨텍스트 정제나 잡음 감소 등)을 단독 보정으로는 재현하기 어렵다는 메시지를 남긴다.



### NLL-Guided Full-Attention Layer Selection for Training-Free Sliding-Window Adaptation (https://arxiv.org/abs/2606.27791)
- **Prior Approaches**: 긴 문맥 추론을 위한 hybrid attention(SWAA)은 프리필에서 sliding-window attention(SWA)과 full attention(FA)을 섞어 효율을 얻지만, 어떤 레이어를 FA로 남겨야 하는지는 여전히 핵심 과제로 남아 있었다. 기존 해법은 주기적인 레이어 패턴처럼 고정 규칙을 쓰거나 LightTransfer처럼 attention 휴리스틱에 의존해 downstream 정확도에 무엇이 중요한지 직접 포착하지 못한다는 한계가 있었다.

- **Core Contribution**: 이 논문은 NLL-guided layer selection을 제안한다. 학습 없이 각 레이어가 FA를 유지할 때 answer 토큰의 NLL이 얼마나 덜 악화되는지(= NLL 감소량)로 레이어 중요도를 직접 측정해, FA를 유지할 상위 k개 레이어를 고른다.

- **Technical Challenges**: 기술적 과제는 “레이어별 중요도”를 생성 과정이 아닌 프리필 단계의 효과로 신뢰성 있게 분리해 점수화하는 데 있다. 저자는 teacher forcing으로 레이어별 토글 실험을 수행하되, SWA는 프롬프트 토큰에만 적용하고 answer 토큰은 full 맥락을 보이게 해(디코드 규칙 일치) NLL 차이를 레이어 기여로 해석 가능하게 만들었다. 또한 64개(16k~32k) 장문 예시로 약 15분 원샷 캘리브레이션 뒤, 이후 배치는 선택된 레이어 집합을 고정해 비용을 상쇄한다.

- **Empirical Impact**: LongMemEval에서 NLL-guided 1/4-FA는 64.6% 정확도로 1/2-FA periodic(65.0%)와 거의 비슷한 성능을 내면서 FA 연산 예산은 절반으로 줄였다. SWAA의 1/4-FA periodic 기준(54.2%) 대비 10.4%p, LightTransfer-style(동일 조건) 기준(38.2%) 대비 26.4%p 더 높아 데이터 기반 선택의 우위가 확인됐다. 장문/단문 캘리브레이션 비교에서 랭킹 상관이 낮고(ρ=0.306) 크기 차이도 커서, 신호가 일반적인 레이어 민감도보다 장거리 attention 필요에 더 가깝다는 점(교란 제거)이 제시됐다.



### SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict Mitigation in Retrieval-Augmented Generation (https://arxiv.org/abs/2606.27786)
Comments:
          19 pages, 13 Figures

- **Prior Approaches**: RAG은 검색 문서를 근거로 생성해 LLM의 사실성을 높이지만, 검색 문서의 지식이 모델의 parametric knowledge와 충돌하면 검색 증거를 무시하거나 특정 지식에 과도하게 의존하는 등 실패가 발생한다. 이를 줄이기 위해 지식 관련 neuron을 찾아 수정하거나, FFN/attention head 등 더 거친 단위의 고정 개입 규칙을 적용하는 연구가 이어져 왔다. 다만 neuron 단위는 국소화가 어렵고 취약하며, layer 단위 개입은 입력마다 충돌 양상이 달라 고정 규칙이 일반 능력을 해칠 수 있다.

- **Core Contribution**: SHIFT는 backbone LLM의 파라미터를 고정한 채, FFN 가지에 삽입한 learnable gate로 내부 활성(hidden-state)을 입력 의존적으로 조절해 지식 충돌을 완화한다. 기존처럼 특정 뉴런을 직접 편집하거나 사전에 선택한 레이어에 정적인 억제/강화를 거는 대신, 필요할 때는 검색 맥락을 더, 충돌 시에는 parametric knowledge의 영향력을 상대적으로 조정하도록 설계했다. 또한 GRPO로 게이트를 최적화해, 충돌 상황에서 근거 신뢰도에 맞춰 중재(arbitration)를 수행한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 충돌 유형마다 다른 균형 조절이 필요한데 고정 개입은 이를 반영하기 어렵고, (2) 뉴런/구조 수준 개입은 의도치 않은 연쇄 영향으로 모델의 일반 능력을 떨어뜨릴 수 있다는 점이다. SHIFT는 backbone을 freeze하고 FFN 기여에 대한 스칼라 게이트를 통해 activation을 약화/유지/증폭하는 최소 개입 구조를 택했다. 학습은 0.01% 미만의 경량 파라미터만 업데이트하며, GRPO(참조 정책 제약 포함)와 게이트 정규화 및 faithfulness 보상을 함께 써서 특정 입력에 과적합되지 않도록 안정적으로 중재 학습을 유도한다.

- **Empirical Impact**: 6개 데이터셋과 Qwen 백본 2종에서 SHIFT는 여러 경쟁 베이스라인을 일관되게 능가하며, 동일 백본 기준 평균적으로 EM/F1에서 유의미한 개선을 보였다. 특히 ConfiQA의 지식 충돌 세팅에서 SFT 대비 큰 폭의 향상을 보이면서도, MMLU 평가에서는 모델 성능 저하가 평균 0.5% 미만으로 매우 작아 일반 능력 보존 측면에서도 강점을 드러냈다. 또한 gate activation이 서로 다른 충돌 유형을 선형분리 가능하게 만들고(AUC 0.832), 입력에 따라 게이트가 구분된 조절 신호를 학습한다는 분석과 함께, 테스트 시에도 다양한 LLM/태스크로 전이되는 강건성을 확인했다.



### Output-Space Allocation Costs for Calibration-Guided LLM Compression: An Empirical Study (https://arxiv.org/abs/2606.27785)
- **Prior Approaches**: LLM 압축의 학습 없이(post-training, training-free) 진행되는 방식은 보통 weight-space 기반 비용을 쓰거나, calibration 데이터로 activation-aware 결정을 보정한다. 특히 ROCKET은 각 레이어의 factorization은 출력 복원(output reconstruction) 목적에서 얻으면서도, 레이어 간 예산 배분(MCKP)에서는 weight-space Frobenius error를 비용으로 사용한다는 점에서 ‘목적 불일치’가 존재한다. Activation-aware 대표 방법(AWQ, ASVD)은 activation 통계를 반영해 품질을 개선해 왔지만, ROCKET의 전역 배분 비용을 출력 공간으로 정렬하는 효과는 아직 명확히 검증되지 않았다.

- **Core Contribution**: 본 논문은 ROCKET의 MCKP에서 사용하는 allocation cost를 weight-space에서 output-space(whitened) error로 교체한 ROCKET-ActCost를 제안한다. 목표는 factorization을 이끈 출력 공간의 기준과, 전역 예산 배분의 기준을 정합시키는 것이다. 추가로 출력 공간 최적화로 인해 레이어별 최적 sparsity 설정(ksk_{s})까지 함께 바뀔 수 있음을 실험적으로 드러낸다.

- **Technical Challenges**: 핵심 기술 난제는 ‘출력 공간 error’를 MCKP 비용으로 쓸 때, 과도한 계산 오버헤드나 추가 calibration 패스를 요구하지 않으면서도 정확히 비용을 재정의하는 것이다. 저자들은 ROCKET의 profiling 단계에서 이미 계산되는 whitened weight(W_L)와 whitened reconstruction의 행렬들을 활용해 output-space error와 output-optimal ksk_{s}를 추가 패스 없이 산출하도록 설계했다. 또한 MCKP는 동일한 제약/구조를 유지하되 cost 함수만 교체해 비교 가능성을 확보했다.

- **Empirical Impact**: Qwen3-8B에서 50% 압축을 걸었을 때 ROCKET-ActCost는 8개 zero-shot 벤치마크 평균 정확도가 53.1%로 52.3% 대비 +0.8pp 상승했지만, WikiText-2 perplexity는 61.46으로 52.98 대비 16% 악화됐다. 이는 출력 공간 비용이 과제 정확도(task-relevant information)에는 유리하지만 언어 모델링 품질(perplexity)에는 불리할 수 있음을 보여주는 ‘accuracy-perplexity tradeoff’로 해석된다. 한편 weight-space와 output-space error 간 상관이 0.99 이상으로 매우 높아 레이어 예산 배분이 크게 갈라지지 못해 효과 크기가 제한됐고, Llama-3.2-1B에서는 20% 압축에서 두 방법 결과가 거의 동일해 비용 선택의 영향이 압축이 강할 때 더 두드러짐을 시사한다.



### KG2Cypher: Data-Centric Pipeline for Building Enterprise Text-to-Cypher Systems (https://arxiv.org/abs/2606.27742)
Comments:
          11 pages, 2 figures, 10 tables

- **Prior Approaches**: 기존 Text-to-SQL/복잡한 KGQA 연구는 실행 가능한 쿼리(또는 logical form)로의 매핑을 벤치마크 중심으로 다뤘지만, 엔터프라이즈 KG는 비공개 스키마·엔티티 URI·리터럴 관례 때문에 그대로 적용하기 어렵다. Text-to-Cypher 쪽도 공개 데이터 보정이나 LLM 합성에 의존하는 경우가 많아, 프롬프트만으로는 실행은 되더라도 정답 그래프를 놓치는 문제가 남는다. 즉 “문법적으로 실행 가능한 Cypher”과 “올바른 엔터프라이즈 결과” 사이의 빈틈이 크다.

- **Core Contribution**: KG2Cypher는 기존 엔터프라이즈 KG 자체에서 관측된 그래프 사실로 실행 가능한 Cypher 타깃을 먼저 만들고, LLM은 자연어 생성/패러프레이즈/품질 판단 같은 언어 측 작업에만 제한해 파이프라인화했다. 이렇게 검증된 Text-Cypher 쌍을 candidate-aware SFT 데이터로 전환해, 모델이 스키마 관계와 엔티티 URI를 “선택”하도록 학습시킨다. 서빙 단계에서는 class-conditioned schema prompting, 엔티티 후보 검색, LoRA 기반 추론으로 운영 지향 구조를 갖춘다.

- **Technical Challenges**: 핵심 난제는 (1) 스키마 관계를 잘못 고르거나 (2) 비공개 엔티티 URI를 환각하며 (3) 날짜·수치·단위 같은 리터럴 sub-field를 잘못 매핑하면 실행 결과가 틀어진다는 점이다. KG2Cypher는 그래프에서 SPO 패턴을 샘플링해 실제 서브그래프에 대한 캐노니컬 Cypher를 결정적으로 구성하고, LLM이 언어만 리라이트하도록 고정함으로써 불가능한 관계/환각 URI/비실행 조건을 줄인다. 또한 후보 관계는 추론 시점에 전면 의존하지 않고, 도메인 분류로 클래스를 라우팅해 해당 클래스의 full relation schema를 프롬프트에 넣어 relation-first 검색 병목을 피한다.

- **Empirical Impact**: 한국 엔터프라이즈 KG 설정에서 prompt-only 생성은 실행은 되더라도 EM과 execution-result F1이 거의 나오지 않았고, KG2Cypher의 LoRA SFT는 방송(broadcast-program) 쿼리에서 execution-result F1을 0.806→0.950, 기업(company) 쿼리에서 0.70→0.92로 크게 끌어올렸다. 11-class class-conditioned 최종 설정에서는 EM 95.2%, 실행 성공률 99.9%, execution-result F1 0.964를 달성해 엔터프라이즈 “언어-그래프 접지”가 개선됨을 보였다. 결과적으로 실행 유효성만으로는 부족하다는 점을 실증하며, private enterprise KG 배포형 Text-to-Cypher 구축 패턴을 제시한다.



### Enhancing Numerical Prediction in LLMs via Smooth MMD Alignmen (https://arxiv.org/abs/2606.27731)
- **Prior Approaches**: 기존 연구는 숫자를 일반 텍스트처럼 다루는 문제를 지적하며, 숫자 인코딩이나 토큰화 개선부터 수치값을 반영한 학습 신호 설계까지 확장해 왔다. 특히 EMD 계열처럼 값의 거리(전달 비용)를 가중해 손실을 주는 방식이 주목받았지만, 국소적으로 예측-정답 잔차가 얼마나 매끈하게 변하는지까지는 충분히 강제하지 못했다. 또한 MMD 같은 분포 정합을 숫자 예측에 직접 적용한 방법은 상대적으로 제한적이었고, 커널 설계가 성능에 미치는 영향도 명확히 정리되지 않았다.

- **Core Contribution**: 이 논문은 큰언어모델이 수치 출력에서 신뢰도 부족을 보이는 원인을 ‘값의 거리 구조를 무시하는 학습 목적’의 불일치로 보고, Smooth Maximum Mean Discrepancy(SMMD)로 이를 정렬한다. SMMD는 숫자 토큰에 대해 값-거리 기반 커널을 정의한 뒤, RKHS에서 커널 매칭으로 예측 숫자 분포를 목표(정답) 분포에 맞추고 잔차의 국소 일관성도 그래프 매끈함으로 함께 유도한다. 따라서 SMMD는 아키텍처 변경 없이 기존 cross-entropy와 결합해 수치 정밀도를 높이는 학습 신호를 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 숫자 토큰을 ‘서열/거리’가 있는 값 체계로 해석하면서도 next-token autoregressive 학습 흐름과 충돌하지 않는 손실을 만드는 것이다. 저자들은 숫자 서브-보크를 구성해 조건부로 분포를 제한하고, 값 간 거리를 커널 유사도로 바꿔 MMD를 유도하되 prediction-target residual에 Dirichlet energy(그래프 라플라시안 형태)를 적용해 국소 진동을 줄인다. 또한 잔차에 smoothness를 걸어 완전 예측 시 보조항이 자연스럽게 0으로 수렴하도록 설계해 학습 목적의 정합성을 유지했다.

- **Empirical Impact**: SMMD는 수학적 추론(GSM8K, SVAMP), 산술 계산(DeepMind-Math), 시각 기반 시각/시간 인식(Clock-Time), 차트 질의응답(ChartQA)에서 다양한 open-weight LLM/VLM 백본에 걸쳐 정확도를 일관되게 개선했다. 비교 실험에서는 cross-entropy뿐 아니라 최근의 숫자 타깃 전용 손실(Gaussian Cross Entropy, NTL, NTIL) 대비 성능이 자주 우세했으며, 분석 결과로는 MMD 정합과 smoothness 정규화가 보완적으로 작동함이 확인됐다. 특히 산술·시간 과제에서 큰 오차를 줄이고(예: Time Gap 감소) 분포의 목표 정렬이 더 날카로워지는 경향이 관찰되어, 수치 예측의 안정성과 일반화에 의미 있는 기여를 한 것으로 평가된다.



### Do Speech Emphasis Models Generalize across Languages and Emotions? (https://arxiv.org/abs/2606.27717)
Comments:
          Interspeech 2026

- **Prior Approaches**: 기존 강세(강조·prominence) 탐지는 주로 영어의 중립 읽기 음성(단일 언어, 제한된 화법) 위주로 학습·평가돼 다국어·감정 표현으로의 일반화가 불명확했다. 일부 연구는 합성 TTS에 처방된 강세 라벨을 쓰거나(라벨이 청취가 아니라 스크립트/LLM에서 생성), 다른 연구는 합성 음성·LLM 라벨로 스트레스 헤드를 학습하는 등 사람의 지각 기반, 감정 범위의 폭이 제한되는 경우가 많았다.

- **Core Contribution**: 이 논문은 MMEE(Multilingual Multi-Emotion Emphasis)라는 대규모 다국어·다감정 말뭉치를 제안해, 7개 매크로언어·10개 지역 변종의 34개 감정/화법 범주에서 10,000개 발화를 3단계(미강조/강조/중강조) 단어 수준 지각 라벨로 수집했다. 또한 EmphaClass와 WhiStress 두 모델을 여러 전이 설정(단일언어, 교차언어, 다국어, 교차-감정, 교차-데이터셋, 데이터 스케일)으로 체계 벤치마킹해 강세 표현의 보편성과 균열 지점을 함께 분석했다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 언어/감정에 따라 강세의 음향 단서가 달라지는 상황에서, 모델이 언어 특이 신호에 과적합되지 않게 학습하는 것과 (2) 합성 라벨과 인간 지각 라벨이 서로 다른데도 공통 표현을 학습하는지 검증하는 것이었다. 연구진은 XLS-R 기반 프레임 분류를 스칼라 회귀로 확장하고, Whisper 기반 WhiStress는 언어 조건 토큰을 활용해 다국어 처리를 지원하는 방식으로 두 모델을 동일한 분할·평가 틀에서 비교했다.

- **Empirical Impact**: 실험 결과 단일언어 학습 모델은 자국 내에서는 강하지만, 계통적으로 먼 언어로 갈수록 zero-shot 교차언어 전이가 급격히 저하됐다(특히 Mandarin이 지속적으로 약함). 반면 다국어 풀링 학습은 교차언어 강건성을 크게 높였고, 데이터 규모를 늘리면 초반 몇 천 샘플 구간에서 성능 이득이 집중되는 경향을 보였다. 또한 high/low arousal 사이에서도 강세 탐지 성능이 비교적 견고했으며, 인간 지각 기반 벤치마크와 합성 처방 기반 벤치마크 사이에 양방향 전이가 관찰돼 강세 신호의 일부는 라벨 패러다임·데이터 출처에 비교적 잘 유지됨을 시사한다.



### Low-Agreeableness Persona Conditioning for Safe LLM Fine-Tuning (https://arxiv.org/abs/2606.27709)
Comments:
          9 pages, 8 tables, 5 figures

- **Prior Approaches**: 기존 연구는 공감·따뜻함(empathetic warmth) 데이터로 fine-tuning할 때 사실 정확성이 떨어지고 sycophancy가 커지며, 그 과정이 adversarial safety까지 약화될 수 있음을 보여주었습니다. 또 일부 방법은 데이터에서 위해 신호를 필터링하거나 harm 라벨/검출기를 쓰는데, 이는 비용이 크거나 가용 데이터에 제약이 있습니다. 한편 대화의 ‘감정’ 자체보다 담화 구조가 안전에 영향을 준다는 관찰도 있어, 따뜻함 데이터가 항상 동일한 안전 비용을 만드는지 재검증이 필요해졌습니다.

- **Core Contribution**: 이 논문은 따뜻한 미세조정이 안전을 해치는 원인이 ‘공감이라는 목표 자체의 필연적 부작용’인지, 아니면 ‘데이터 구성의 산물’인지 분해해 보려 합니다. 핵심은 persona-driven rewriting pipeline로 사용자 턴을 Big Five의 low agreeableness(낮은 동조성) 성향으로 만들고, 대신 어시스턴트 응답은 warm 하면서도 de-escalating(격화 완화)하도록 함께 재작성하는 데이터 설계입니다. 안전 라벨, harm detector, 학습 목적 변경 없이도 jailbreak 취약성과 유해 출력률을 일반적인 warmth fine-tuning 대비 낮출 수 있음을 보입니다.

- **Technical Challenges**: 문제는 따뜻함을 유지하면서도 ‘warmth와 compliance(순응) 방향이 잠재공간에서 함께 정렬되는’ 공변 관계를 끊는 데이터 신호를 설계하는 데 있습니다. 저자들은 먼저 Llama-3.1-8B에서 잔차(residual) 섭동으로 안전에 민감한 레이어를 찾고, Big Five 하위집합 중 low agreeableness가 가장 유리하다는 점을 pilot로 고른 뒤 다른 모델에도 적용해 일반화를 검증합니다. 또한 ablation을 통해 user-side만 low agreeableness로 바꾸면 오히려 per-token warmth가 손상되며, de-escalating 어시스턴트 rewrite가 그 균형을 회복한다는 점을 분리해 보여줍니다.

- **Empirical Impact**: 세 가지 실험(4개 모델)에서 제안한 full paired condition은 generic warmth fine-tuning baselines보다 jailbreak 성공률과 red-teaming 기반 harmful output 지표에서 더 일관되게 개선을 보였습니다. 특히 MentalChat-16K 기반 정신건강 지원 도메인에서도 동일한 경향이 유지되었고, per-token warmth는 기준선 대비 유지 또는 개선되어 ‘더 차가워진 탓’의 설명을 약화시킵니다. representational probing에서는 따뜻함과 compliance 간 기하학적 정렬이 완화되는 신호(웜스-컴플라이언스 decoupling)가 관측되어, 안전 개선이 데이터 설계에서 비롯된 메커니즘과 정합적이라는 근거를 제공합니다.



### Mitigating Position Bias in Transformers via Layer-Specific Positional Embedding Scaling (https://arxiv.org/abs/2606.27705)
- **Prior Approaches**: LLM의 긴 컨텍스트에서 나타나는 lost-in-the-middle 문제는 RoPE 기반 모델이 위치에 따라 주의를 다르게 주는 positional bias에서 비롯된다. 이를 줄이기 위해 여러 RoPE를 섞는 Attention Buckets, head별/계층별 scaling을 두는 Ms-PoE, MoICE 같은 방법이 제안됐지만 대부분 수작업 휴리스틱 의존 또는 다중 forward/병렬 계산으로 인해 지연과 비용이 커진다.
또한 모듈 단위로 세밀하게 스케일을 탐색하면 조합 수가 폭증해 자동 탐색이 비효율적이라는 한계가 남는다.

- **Core Contribution**: 이 논문은 layer-specific positional embedding scaling(LPES)을 제안해, 각 트랜스포머 층마다 서로 다른 RoPE scaling factor를 부여하면서도 fine-tuning과 추가 추론 지연 없이 positional bias를 완화한다. 핵심은 모델 전체를 앙상블하지 않고도 single forward pass로 주의 분포를 더 균형 있게 만드는 점이다.
또한 scaling factor 조합을 Bézier curve로 매개변수화해 최적화 가능하게 만든다.

- **Technical Challenges**: 층마다 scaling factor를 모두 독립 변수로 두면 조합 최적화가 되어 gradient 기반으로 풀기 어렵고 탐색 공간이 지나치게 커진다. 논문은 Bézier curve의 소수 control points로 층 스케일을 연속적 함수로 제한해 탐색 공간을 급격히 줄이면서, 곡선의 매끄러움이 층 간 표현 구조를 크게 깨지 않는 inductive bias로 작동함을 함께 활용한다.
구체적으로는 curve-constrained genetic algorithm로 control point를 탐색하고, 곡선의 매끄러움(통제점의 단조성/변이 제한)을 제약해 안정적인 스케일링 구성을 찾는다.

- **Empirical Impact**: MDQA의 key-value retrieval 등 여러 long-context 벤치마크에서 LPES는 lost-in-the-middle을 직접 겨냥한 positional bias를 줄이며 일관된 성능 향상을 보인다. 특히 key-value retrieval에서 최대 11.2% 정확도 향상을 보고하며, 다른 태스크로의 scaling factor 전이도 안정적이었다.
추론 효율 측면에서도 Ms-PoE 대비 약 1.45×, MoICE 대비 약 2.42× 더 빠르며, 모델 파라미터를 업데이트하지 않는 training-free 성격이라 기존 배포 모델에도 적용 가능성이 높다.



### Mitigating LLM-based p-Hacking by Preregistering for the Next LLM (https://arxiv.org/abs/2606.27687)
- **Prior Approaches**: 기존 LLM 기반 연구는 데이터 생성·분류·주석을 수행한 뒤 그 결과를 downstream 가설검정에 활용하는 경우가 많다. 하지만 프롬프트, 디코딩 파라미터, 출력 포맷을 반복적으로 조정하면 원하는 성과가 나오도록 “p-hacking”이 쉽게 발생할 수 있다.

- **Core Contribution**: 논문은 LLM 연구에서 p-hacking을 줄이기 위한 프로토콜을 제안한다. 실험을 사전등록(preregistration)하고, 특정 조건을 만족하는 eligible 모델 집합을 정한 뒤 사전등록 이후 처음 공개되는 eligible LLM에 대해 확증분석(confirmatory analysis)을 수행한다.

- **Technical Challenges**: 핵심은 어떤 설정을 골랐는지가 사전등록 시점에는 아직 존재하지 않을 ‘다음 모델’에 그대로 적용되지 않게 설계하는 것이다. 저자는 우선 기존 모델에서는 절차를 확정하고, 분석 플랜과 향후 eligible 모델 목록을 함께 preregister한 뒤, 첫 공개 모델에서만 결과를 확정하도록 실행 흐름을 고정한다.

- **Empirical Impact**: 두 가지 태스크(정답이 알려진 조건)에서 20개 모델, 11개 LLM-analysis 구성으로 평가한 결과, 프로토콜은 p-hack의 성공적 전이를 각각 73.9%, 72.7%에서 차단하는 것으로 나타났다. 또한 여러 stress test에서도 완화 효과가 유지됐고, 실제로 저자들이 동일 프로토콜을 적용한 사전등록 실험에서는 기존 모델을 “해킹”했던 7개 구성 중 6개에서 다음 eligible 모델로의 전이가 실패해 유효성이 확인됐다.



### From Signals to Transfer: A Factorised Study of Probe-Based Uncertainty Estimation in Large Language Models (https://arxiv.org/abs/2606.27679)
- **Prior Approaches**: 프로브 기반 uncertainty estimation(UE)은 LLM 내부 신호에서 불확실성을 학습해 환각을 탐지하는 방식으로 주목받아 왔습니다. 다만 기존 연구는 feature 설계, 학습 데이터/라벨 구성, 프롬프트·평가 설정을 동시에 바꾸는 경우가 많아 “무엇이 성능을 올리는지”가 흐려졌습니다. 또한 매치된 벤치마크에서는 잘 동작해도, 다른 도메인·생성 형식으로 옮기면 성능이 크게 떨어지는 일반화 한계가 지적됩니다.

- **Core Contribution**: 논문은 프로브 기반 UE를 feature 표현, 데이터/라벨 구성, 전이(transfer) 설정으로 분해해 요인이 성능을 어떻게 바꾸는지 맞춤 조건에서 분석합니다. 그 결과, 단순 raw hidden states나 attention feature는 in-domain에서는 경쟁력이 있으나 분포 이동에서는 구조화/압축된 feature가 더 견고하다고 제시합니다. 또한 prompting 방식(추론 유도)과 자동 라벨 구성(lexical vs LLM-as-a-judge)이 프로브 동작을 크게 좌우함을 정리해 재사용 가능한 pretrained factuality probe의 기준선을 제안합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 내부 신호·라벨·프롬프트를 동시에 건드리지 않고, 성능 변화의 원인을 요인별로 분리하는 설계와 (2) 자동 라벨이 정답 의미를 얼마나 충실히 반영하는지 검증하는 것입니다. 저자들은 AUROC와 ECE로 분별력과 캘리브레이션을 함께 보고, terse/long reasoning 프롬프트와 Rouge·AlignScore·LLM-as-a-judge 라벨링을 비교해 lexical 매칭 기반 라벨의 왜곡 가능성을 확인합니다. 전이 일반화를 위해서는 Lookback Ratio, Layer Top-mm Prob., Internal Variance 등 더 구조화된 feature와 비교적 단순한 linear probe를 우선시하는 “best practices”를 도출합니다.

- **Empirical Impact**: 여러 벤치마크(QA/verification/MCQ)에서 in-domain 성능만으로는 발전을 판단하기 어렵고, OOD(동일 태스크·교차 태스크)에서의 견고성이 핵심임을 실증했습니다. 특히 reasoning을 길게 유도하면 LLM 정확도는 유지돼도 프로브의 AUROC가 떨어지는 경향이 나타나, 배포 관점에서 입력/출력 포맷 통제의 중요성이 강조됩니다. 또한 벤치마크에서 사전학습한 benchmark-pretrained probe가 타깃 도메인 라벨 없이도 open-ended long-form factual generation에 “stable off-the-shelf baseline” 수준으로 전이되며, raw last-token embedding의 전이 취약성과 구조화 feature의 우수한 견고함을 재확인합니다.



### When Search Agents Should Ask: DiscoBench for Clarification-Aware Deep Search (https://arxiv.org/abs/2606.27669)
Comments:
          26 pages, 7 figures, 12 tables

- **Prior Approaches**: 기존 검색 에이전트 벤치마크는 대체로 사용자 질의가 완전하고 명시적이라고 가정해, 다중 홉 추론은 평가하지만 모호한 요청을 ‘언제’ ‘어떻게’ 풀기 위한 상호작용은 충분히 다루지 못했습니다. 모호성(ambiguity)만 다루는 데이터셋은 대개 정적 시나리오에 머물고, 상호작용 벤치마크는 폐쇄형 샌드박스 위주라 웹 스케일의 실제 검색 깊이와 자연스러운 대화 양상을 반영하기 어렵다는 한계가 있었습니다. 결과적으로 딥 서치 과정에서 모호성이 연쇄적으로 오답 경로를 증폭시키는 문제를 체계적으로 측정하기가 어려웠습니다.

- **Core Contribution**: 본 논문은 clarification-aware deep search를 평가하는 벤치마크 DiscoBench를 제안합니다. DiscoBench는 다단계 검색 중간의 ‘애매한 체크포인트’가 시간에 따라 동적으로 발생하고 추론 체인으로 전파된다는 관점에서 211개 샘플과 463개 모호성 인스턴스(11개 도메인, 4가지 모호성 유형)를 구성했습니다. 또한 에이전트가 모호성을 감지하고 효과적인 clarifying question을 통해 올바른 추론 경로로 복구하는지, 그리고 비용 효율성까지 통합 평가하도록 프레임워크와 user simulator를 함께 제공합니다.

- **Technical Challenges**: 핵심 기술 과제는 모호성을 정적 분류가 아니라 멀티턴 검색 진행 중에 언제 인지하고, 사용자로부터 한 번의 단서로 구별되는 정보를 얻어 다음 검색 단계로 ‘복구’시키는 흐름을 설계하는 것입니다. 이를 위해 체크포인트를 Unambi/Ambi로 나누고, Ambi에서는 Ask로 사용 단서를 받은 뒤 Search를 재정렬해 Answer로 이어지도록 순차 질문-응답 태스크로 정의했으며, ambiguity 유형(엔티티/버전/기준/사실부정확)별로 단서가 구별 능력을 갖도록 discriminative facts를 생성했습니다. 실험에서는 탐색 도구(tool use) 강도와 질문 전략이 성능에 어떻게 연결되는지까지 분리 관찰할 수 있게 평가 지표를 과업 유틸리티·탐지·상호작용·비용 관점으로 구성했습니다.

- **Empirical Impact**: 실험 결과, 최신 LLM 기반 검색 에이전트는 여전히 모호성을 안정적으로 알아차리지 못해 end-to-end 정확도가 낮았습니다(Neutral 기준 최상위도 40%대 초반). 특히 ambiguity detection과 effective clarification은 서로 다른 능력으로 나타났고, 같은 탐지/질문이라도 실제로 추론 경로 복구로 이어지지 못하면 최종 성공이 크게 제한됐습니다. 동작 프로파일 분석에서 Search를 반복하다 Ask를 늦추는 경우 성능이 더 나빠져, ‘검색을 더 한다’가 아니라 ‘언제 명확히 물어봐야 하는지’가 딥 서치의 병목임을 실증적으로 보여줍니다.



### Yuvion LLM: An Adversarially-Aware Large Language Model for Content And AI Safety (https://arxiv.org/abs/2606.27632)
- **Prior Approaches**: 기존 LLM 안전 연구는 주로 자연 입력에 대한 단발(싱글턴) 안전 판단이나 가드 모델에 초점을 둬, 배포 현장의 인간적 우회(재구성, 은어/코딩, 언어 혼합, prompt injection 등) 압력을 충분히 평가하지 못했다. 또한 에이전트형 워크플로(계획-툴 호출-정책 근거 수집-다단계 추론)에서 필요한 능력이 안전 쪽 모델 개발에 구조적으로 반영되지 않아, 측정된 안전 성능이 실제 견고함을 과대평가할 위험이 크다.

- **Core Contribution**: 이 논문은 안전을 본질적으로 adversarial 문제로 보고, Yuvion LLM이 adversarially robust content safety와 agentic safety capability를 “처음부터(first-class objective)” 동시에 학습하도록 설계했다고 주장한다. 이를 위해 단계별(progressive) 파이프라인(지식 강화 continued pretraining → policy-grounded multi-task SFT/정책 최적화 → tool use·multi-step에 대한 agentic RL)과, 안전·공격 견고성·현장형 역량을 함께 보기 위한 Yuvion LLM RiskEval(YLRE, 4레벨 93개 벤치마크)을 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 은닉된 유해 의도와 정책 회피를 “표면 표현”이 아닌 잠재 의도/근거로 파악하면서 (2) 도구 호출·검색·다단계 실행 같은 에이전트 작업에서 정책 정합성을 유지하는 것이다. 논문은 이를 위해 안전 지식 베이스를 다중 granularity로 주입하는 continued pretraining, 위험 인식·정책 근거 생성 중심의 SFT, 그리고 GRPO 기반 보상 설계(불명확성·공격·엣지 케이스에서 단일 정답 의존 완화)를 적용하며, 도구사용 트랙은 format·correctness로 보상을 분해해 궤적 수준의 학습 안정성을 높이고 검색·증거 수집은 execution reward와 LLM-as-judge result reward로 중간 상호작용을 촘촘히 지도한다.

- **Empirical Impact**: 실험에서 Yuvion은 안전 특화 벤치마크에서 명확한 우위를 보이고, 특히 adversarial 조건에서 강한 견고성을 보이면서도 전체 역량은 유지하는 것으로 보고된다. 또한 Yuvion-8B가 GPT-5.4, Qwen3-MAX 같은 더 큰 모델을 다수 안전 과제에서 앞서며, 안전 성능과 현장 배치 적합성을 동시에 노린 학습·평가 설계의 효율성을 시사한다.



### Cross-Platform Chinese Offensive Comment Detection via Dual-Threshold Hard Example Mining (https://arxiv.org/abs/2606.27629)
Comments:
          10 pages, 7 figures

- **Prior Approaches**: 중국 욕설·혐오 댓글 탐지는 기존에 욕설 사전 매칭, TF-IDF·n-gram 같은 얕은 특징, 또는 CNN/RNN 기반 모델에 의존해 왔습니다. 그러나 이런 방식은 아이러니·은유·동음이의 농담처럼 명시적 단어가 없는 암묵적 공격을 잘 잡지 못한다는 한계가 컸습니다. 또한 COLD 같은 벤치마크가 있어도 단일 플랫폼·정적 분포에 머물러 실제 크로스플랫폼 도메인 shift를 체계적으로 진단하고 대응하기는 부족했습니다.

- **Core Contribution**: 이 논문은 COLD 기반 RoBERTa 이진 기준선을 공정 비교를 위해 먼저 구축하고, 위에·샤오홍슈·티에바·즈후 네 플랫폼을 아우르는 3클래스(정상/명시/암묵) 크로스플랫폼 평가 세트를 구성합니다. 이어 도메인 거리(Jaccard, Proxy-A Distance)로 플랫폼 간 간극을 정량화하면서, 기준선 성능 저하의 핵심 병목이 암묵 공격 인식 약화와 플랫폼 전용 용어 부재에 있음을 드러냅니다. 그 위에 dual-threshold hard mining과 소량 라벨로 암묵 맥락을 보정하는 lightweight domain adaptation을 제안합니다.

- **Technical Challenges**: 핵심 기술적 난제는 라벨이 거의 없는 타 플랫폼에서 기준선의 실패 지점을 효율적으로 샘플링하고, 암묵적 공격의 문맥 의존성을 적은 비용으로 보정하는 것입니다. 논문은 baseline의 Softmax confidence를 기준으로 high-confidence 오분류(겉보기는 정상처럼 보여도 사실은 암묵 공격)와 low-confidence 오분류(플랫폼 슬랭/신조어로 경계가 흔들림) 두 풀로 나눠 hard example을 선별하고, 각 플랫폼에서 100개씩 총 400개만 수동 확인해 2차 미세조정을 수행합니다. 이를 통해 large-scale 데이터 증강 없이도 모델의 의사결정 캘리브레이션을 노립니다.

- **Empirical Impact**: 실험 결과 최적화된 모델은 네 플랫폼 모두에서 성능(특히 F1과 유해 샘플 recall)을 유의미하게 개선하며, 샤오홍슈처럼 COLD에 없는 도메인에서 이득이 가장 크게 나타납니다. 암묵 공격에서의 false negative가 줄고, 플랫폼 전용 표현으로 인한 오분류도 완화되는 패턴이 관찰됩니다. ablation에서도 동일 라벨 예산 하에 random sampling보다 hard example mining이 더 효과적임을 보여, ‘양’보다 ‘어려운 샘플’이 미세조정 효율을 좌우한다는 점을 실증적으로 뒷받침합니다.



### Masked Language Flow Models (https://arxiv.org/abs/2606.27617)
Comments:
          Preprint

- **Prior Approaches**: 자기회귀 모델(ARM)은 다음 토큰 예측으로 학습·생성하지만, 좌→우 분해 때문에 생성이 순차적이며 추론 비용이 출력 길이에 비례해 병목이 된다. 이를 줄이기 위해 등장한 Masked Diffusion Models(MDMs)는 마스크 토큰을 병렬로 복원하지만, 역전이 과정이 토큰 위치별로 factorise된 근사에 의존한다. 이 근사는 단계가 적은 few-step 샘플링에서 특히 약해지며, 토큰 간 의존성을 무시한 채 많은 토큰을 동시에 풀어내 품질 저하로 이어진다.

- **Core Contribution**: Flow Language Models(FLMs)는 이산 상태공간 대신 연속 flow로 노이즈를 깨끗한 시퀀스로 운반해, 잠재 상태를 토큰 위치 전반에서 함께 변화시키고 few-step/one-step 생성으로 증류(distill)할 수 있게 한다. 다만 FLM은 단일 flow map으로 축약될 때 수학·코딩처럼 다단계 추론을 위한 반복적 중간 상태 활용이 어렵고, 생성 중 모든 토큰을 매번 디코딩해야 하는 부담이 생긴다. 본 논문은 Masked Language Flow Models(MLFMs)를 제안해, MDM의 마스킹 구조를 FLM의 연속 flow에 통합함으로써 조건부 생성과 다단계 추론 친화적 샘플링을 동시에 노린다.

- **Technical Challenges**: 핵심 난제는 (1) 부분 마스킹과 깨끗한 시퀀스를 잇는 연속 확률 경로를 설계하면서도 (2) 토큰 위치 간 conditioning 구조를 MDM 수준으로 유지하는 동시에 (3) few-step에서 품질을 지키는 샘플러를 만드는 것이다. 저자들은 Brownian bridge 기반의 stochastic interpolant로 partially masked 상태와 clean 상태를 연속적으로 연결해, 어떤 위치든 조건부(any-position conditional) 구조를 살리면서 masked 위치를 채우는 coupled continuous flow를 학습한다. 또한 기존 MDM을 MLFMs로 “가볍게” 변환할 수 있도록 엔드포인트 일치성을 이용해 사전학습 가중치로 warm-start를 하고, 샘플링에서는 classifier-free guidance의 변형(CCFG)과 함께 확신도 높은 토큰을 온라인으로 clean embedding에 “promote(조기 확정)”하는 전략을 결합한다.

- **Empirical Impact**: 1028M 파라미터 규모의 MDM을 MLFM으로 적응해 GSM8K와 MT-Bench에서 평가한 결과, MT-Bench에서 MDM 기반 접근과 비슷한 크기의 AR baseline을 넘어서는 성능을 보였고 GSM8K에서도 유의미한 추론/지시응답 능력을 확인했다. 특히 제안한 샘플링(연속 denoising + 확신 토큰 조기 unmasking)이 두 벤치마크 전반에서 성능을 추가로 끌어올렸다. 저자들은 flow 기반 언어 모델을 downstream reasoning과 instruction-following 과제에 스케일링해 효과를 입증한 “첫 사례”로 의미를 강조한다.



### Narrative-UFET: Narrative Generation for Ultra-Fine Entity Typing (https://arxiv.org/abs/2606.27598)
- **Prior Approaches**: UFET(초정밀 개체 유형 지정)은 문장 수준 문맥만 보고 주어진 개체 언급에 대해 초세분화된 타입 라벨을 예측한다. 기존 연구들은 PLM을 바탕으로 masked language modeling 기반 또는 문장 확장 입력으로 성능을 끌어올렸지만, 학습 데이터에서 드문(롱테일) 개체에는 급격히 약해지는 문제가 남아 있다.

- **Core Contribution**: 이 논문은 UFET을 내러티브(짧은 이야기) 수준으로 확장한 Narrative-UFET을 제안한다. 각 개체-문장 쌍에 대해 자동 생성된 짧고 일관된 서사를 붙이되, 생성 과정에서 해당 개체의 타입이 내러티브 동안 유지되는 Maintain과 내러티브 동안 변하는 Change 두 변형으로 “담화 속 신호”의 효과를 통제해 분리한다.

- **Technical Challenges**: 핵심은 생성된 서사가 과도하게 부정확하거나 라벨을 우회로 암시하지 않으면서도, 타입 판별에 도움 되는 담화 구조(문장 간 연결, 코어퍼런스, 증거 분산)를 실제로 담도록 만드는 것이다. 저자들은 모델 선택(여러 오픈 소스 LLM 비교), 프롬프트 설계(문장 길이·캐릭터 수 조절), 그리고 자동 지표·사람 평가로 품질을 검증했으며, 금지된 정보 누출을 막기 위해 골드 타입은 생성기에 제공하지 않았다.

- **Empirical Impact**: 실험 결과, 내러티브 문맥은 문장 수준 기준선 대비 특히 롱테일 타입에서 일관된 성능 향상을 보였고, Maintain보다 Change가 더 강한 신호를 제공했다. 또한 OntoNotes의 자연 문맥과 비교했을 때도 합성 내러티브가 더 큰 이득을 내어, 실제 텍스트에서 암묵적으로 남는 담화 신호를 통제된 서사로 더 잘 드러낼 수 있음을 시사한다. 다만 담화 속 다른 속성까지 체계적으로 확장해야 할 여지가 남아, 담화 모델링과 내러티브 구성 모두에서 후속 연구가 필요하다는 결론이다.



### Ko-WideSearch: A Korean Breadth-Search Benchmark for Exhaustive Set Enumeration by Web Agents (https://arxiv.org/abs/2606.27595)
- **Prior Approaches**: 기존 웹 에이전트 벤치마크는 주로 depth를 측정하며, 제약을 거쳐 하나의 정답을 찾는 방식에 초점이 맞춰져 있었다. 반면 breadth는 닫힌 집합을 빠짐없이 복원하고 각 항목의 속성까지 표 형태로 완성해야 하지만, 특히 한국어 웹에서는 평가 자료와 검증 파이프라인이 부족했다.

- **Core Contribution**: 이 논문은 한국어 breadth-search 벤치마크 Ko-WideSearch를 제안한다. 190개의 set-parent 엔터티에 대해 228개 표를 만들고, 각 표는 집합 구성원 전부와 항목별 속성 테이블을 요구하며 Item-, Column-, Row-F1 및 table success로 평가한다.

- **Technical Challenges**: breadth 벤치마크에서 핵심 난점은 정답 집합의 완전성과 모든 셀의 정합성을 대규모로 인증하는 비용/신뢰성 문제다. 논문은 자동 synthesize-and-verify 파이프라인으로 웹 기반 열거와 다중 게이트(비암기성·완전성·교차 출처 검증)를 수행하고, 평가와 구축에 동일한 normalization-aware 비교기를 써서 포맷 차이가 성능을 왜곡하지 않게 했다.

- **Empirical Impact**: 20개 웹 에이전트를 실험한 결과, 대부분은 구성원 집합 자체는 비교적 잘 복원하지만 행(모든 속성 셀)을 완성하는 데서 큰 격차가 발생했다. 난도(테이블 폭, 2-D composite key)가 강해질수록 Row-F1이 지속적으로 하락하며, 더 많은 검색·비용으로도 간극이 쉽게 좁혀지지 않았고 특히 free-text 셀은 실패가 두드러졌다.



### EntMTP: Accelerating LLM Inference with Entropy Guided Multi Token Prediction (https://arxiv.org/abs/2606.27550)
Comments:
          7 pages, 5 figures

- **Prior Approaches**: 기존 MTP(멀티-토큰 예측) 헤드 기반 speculative decoding은 생성 전체에 대해 트리 기반 attention 토폴로지를 고정해 둔다. 그 결과, 검증 단계에서의 speculation depth와 compute(검증 비용)가 문맥이 어려워지는 구간에서도 그대로 유지돼 자연어의 entropy 패턴과 어긋난다. 또한 토폴로지 선택을 작업(task) 단위로 한 번 오프라인 고정하면 벤치마크마다 “Pareto 최적”이 달라져 효율이 떨어질 수 있다.

- **Core Contribution**: 이 논문은 Entropy-guided Multi-Token Prediction(EntMTP)라는 학습 없는(runtime) 스케줄러를 제안한다. 문맥의 국소 생성 엔트로피 예측 가능도에 따라, precompiled된 여러 Pareto-optimal 트리 중에서 speculative depth를 동적으로 토글한다. 목표는 accepted-token 처리량을 생성 전 구간에서 기대값 기준으로 최대화하면서도, 생성 품질은 유지하는 것이다.

- **Technical Challenges**: 핵심 난제는 “어떤 트리 토폴로지가 그 시점에 가장 많이 accepted token을 만들지”를 저비용으로 예측해 전환 jitter 없이 안정화하는 것이다. 저자들은 벤치마크별로 오프라인 throughput-Pareto frontier를 구성해 TopologyBank를 만들고, 런타임에는 base-LM top-1 확률과 Hydra head들의 EAGLE-2 path value 같은 신호를 이용해 트리 선택 정책(특히 hysteresis 기반 EntMTPτ)을 수행한다. 토폴로지 전환은 attention mask/position offsets/KV 관련 버퍼를 미리 준비해 O(1) 포인터 스왑으로 처리해 추가 GPU 비용을 거의 없앴다.

- **Empirical Impact**: HumanEval, ShareGPT, GSM8K, LitBench 전반에서 EntMTP는 Hydra 대비 지속적인 속도 개선을 보이며, 시점별 스케줄링까지 더하면 peak speedup이 1.36x(주요 Medusa 대비)까지 간다. 배치 size=1 설정에서 EntMTPτ는 vanilla autoregressive 대비 HumanEval 3.26×, GSM8K 3.13×, ShareGPT 3.47×의 wall-clock 처리량 향상을 달성했다. continuation perplexity는 기준선과 거의 동일(0.02 nats 내외)하게 유지돼, 성능 향상이 verification 규칙 변경이 아닌 “스케줄링에 의한 lossless 가속”임을 뒷받침한다.



### The Context-Ready Transformer (https://arxiv.org/abs/2606.27538)
Comments:
          NeurIPS, 22 pages

- **Prior Approaches**: 기존 오토리그레시브 트랜스포머는 토큰을 context-free한 임베딩으로 블록에 넣은 뒤, 여러 층을 거쳐 다시 문맥을 재구성한다는 ‘왕복’이 구조적으로 발생한다. weight sharing(예: ALBERT, Universal Transformer), early exit, lookahead/고정점(예: DEQ) 같은 접근은 연산량이나 추론 속도를 줄이지만, 토큰이 블록에 들어갈 때 이미 문맥 정보를 담아 처리한다는 발상은 상대적으로 부족했다. 또 Mamba/RWKV 계열은 attention을 줄이지만, 핵심은 “문맥 전달 방식”이 아니라 “상태 요약으로 attention을 대체”하는 데 가깝다.

- **Core Contribution**: 이 논문은 context-ready transformer를 제안하며, 토큰이 블록에 들어가기 전에 correction 네트워크가 이전 블록 출력(과거 문맥 요약)을 반영해 미리 문맥화(pre-contextualize)하도록 만든다. 학습 중에는 correction 과정을 K번 병렬로 unroll해 transformer식으로 훈련 가능한 그래프를 유지하면서, 추론에서는 left-to-right 한 번의 스트리밍 패스로 recurrent 동작이 “정확히” 작동하게 설계했다. 또한 pretrained transformer는 zero-initialized correction FFN을 추가하고 fine-tuning하면 context-ready 모델로 변환할 수 있다고 주장한다.

- **Technical Challenges**: 가장 큰 문제는 correction이 이전 위치의 블록 출력에 의존해 추론만큼이나 학습도 순차성이 생긴다는 점인데, 이는 RNN처럼 되면 BPTT 깊이가 시퀀스 길이에 비례해 불리하다. 이를 해결하기 위해 (1) non-cumulative correction으로 매 반복은 누적 합이 아니라 “고정점에 수렴해가는 단일 correction” 형태가 되게 하고, (2) past-only correction으로 t 위치의 correction이 t-1의 캐시된 정보와 현재 임베딩에만 의존하도록 제한했다. 결과적으로 훈련 그래프의 깊이는 O(T)가 아니라 O(K)로 유지되며, K=5~10에서 수렴 및 성능 확보가 관찰된다.

- **Empirical Impact**: 실험에서 D=5 모델은 12층 트랜스포머를 PPL 측면에서 앞서면서 A100 기준 생성 속도도 1.7배 개선했다(동일 길이/조건 비교). 더 작은 D=1 설정에서도 K=10 학습을 통해 6층 대비 2.6배 빠른 추론을 보이면서 PPL에서 근접 또는 개선을 달성했으며, sequential inference가 training의 K unroll과 PPL 차이 0.01 이내로 맞춰진다고 보고한다. pointer-chasing 합성 추론에서는 BPTT로 훈련한 D=1이 모든 composition level을 해결하는 반면 표준 트랜스포머는 depth에 따라 계단형(staircase-like) 성능 의존을 보였고, 전반적으로 wide representation과 long context에서 이점이 커지는 것으로 정리된다.



### Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents (https://arxiv.org/abs/2606.27472)
Comments:
          11 pages, 4 figures, 3 tables. Code, environment, model, and dataset: this https URL

- **Prior Approaches**: 장기 대화에서 시간에 따라 정보가 바뀌는 문제는 LongMemEval, MemoryArena, MemBench 등으로 평가돼 왔지만, 대부분은 고정된 모델 성능을 측정하는 데 그친다. 또한 RL 기반 메모리 에이전트 연구는 최종 정답이나 증거 관련성 같은 보상으로 학습해 왔고, ‘현재 시점의 값(cu rrency)’을 맞추는 시간성 보상은 없었다.

- **Core Contribution**: 이 논문은 대화 중 변경/철회된 사실을 ‘최신 값으로 유지하며 답하는 능력’(supersession)을 분리 측정하고, 이를 제대로 못하는 실패(supersession gap)가 별도 능력 결함임을 보여준다. 더 나아가 보상을 시간 인덱스가 있는 supersession 정합성으로 직접 정의한 공개 강화학습 환경 Supersede를 제안하고, 실제 학습으로 격차를 줄일 수 있음을 입증한다.

- **Technical Challenges**: 핵심 병목은 독해력보다 메모리 유지 관리이며, bounded memory 환경에서 구체적으로 어떤 값이 최신인지 판단해 폐기해야 한다. 이를 위해 원시 세션을 재주입(re-feed)하지 않고 notes 필드만으로 답하게 하며, 프로그램 기반 매처로 ‘정답이 현재 값을 전달했는지’(그리고 필요 시 stale 값을 주장하는지)를 보상 신호로 만든다.

- **Empirical Impact**: 실험에서 full-context는 gpt-5.4 기준 92%에서 92%에 가깝게 포화되는 반면, bounded self-maintained memory는 77%로 크게 하락해 격차가 통계적으로 유의하다(p<0.005). 모델을 더 키워도, 대화를 24배 길게 늘려도(정확도 68%→28%) 메모리를 비례 증량해도(28%→28%) 격차가 닫히지 않으며, 학습은 GRPO로 Qwen2.5-3B의 held-out supersession 정확도를 9.0%→16.7%로 거의 두 배 가까이 끌어올린다.



### Developmental approach reveals the statistical learning of Neural Language Models: Transformers generalize from the most abstract statistical patterns (https://arxiv.org/abs/2606.27460)
Comments:
          10 pages, 7 figures, oral presentation at Interdisciplinary Advances in Statistical Learning

- **Prior Approaches**: 기존 연구는 NLM의 언어 인지를 인간 언어 이론(생성문법의 분리 가정·Universal Grammar, 기능주의의 표면-의미 매핑, 구성 문법 등)으로 설명하려 했지만, NLM은 그런 인덕티브 바이어스를 갖지 않으며 분포가 통사·의미를 동시에 반영한다는 한계가 제기된다. 한편 통계적 학습 연구는 인공 언어로부터 토큰의 분포 통계를 통해 지식이 형성되는 과정을 밝혀왔지만, NLM 내부 표상의 “학습 경로”를 발달적으로 추적한 연구는 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 생성 Transformer를 “발달 경로(developmental path)” 관점에서 분석해, 모델이 전역(글로벌) 통계 지식과 국소(로컬) 의존성을 언제/어떻게 습득하는지 제시한다. 합성 문법에서 여러 학습 단계의 모델 상태를 저장하고, 내부 표상 및 확률 예측이 어떻게 클러스터링과 범주화로 변하는지 추적해 새로운 통계 학습·언어 인지 설명틀을 제안한다.

- **Technical Challenges**: 핵심은 계층적(전역-중간-국소) 의존성이 중첩된 합성 문법을 설계하고, 학습 단계별로 전역/중간/국소 범주가 실제로 분리·정렬되는지 검증하는 것이다. 저자들은 U라는 플랭커를 통해 절대 위치 의존을 줄이고, P·Q 토큰 임베딩을 단계별로 추출해 2D 축소 후 클러스터 변화(표상 분석)와 마스킹된 문장에 대한 probability mass(예측 분석)를 함께 측정했다.

- **Empirical Impact**: 결과는 일관되게 “오버제너럴라이제이션(over-generalization) 후 점진적 제약” 패턴을 보여준다: 초기에는 전역 범주가 먼저 형성되고, 이후 중간 범주가 생기며, 마지막에 국소 범주가 정교하게 분리된다. 미학습 조합에 대해서도 초기 단계에서 체계적으로 더 높은 확률을 부여하는데, 이 오버제너럴라이제이션은 N이나 P로 확산되지는 않고 하위 Q의 다른 부분범주 사이에서 구조적으로 나타나 학습 과정이 무작위가 아님을 시사한다.



### Causal Connections: Leveraging Multilingual Fine-Tuning for Financial QA@FinCausal 2026 (https://arxiv.org/abs/2606.27446)
- **Prior Approaches**: 기존 연구는 금융 문서에서 인과 관계를 찾기 위해 규칙 기반 또는 전통 ML(예: SVM, 결정트리)을 쓰거나, BERT 계열을 fine-tuning해 문맥을 태깅/추출하는 방식에 집중해 왔습니다. 이후에는 암묵적 인과, 다단계 추론처럼 난도가 올라가면서도 많은 접근이 결국 정답 스팬 매칭이나 제한된 생성에 머물렀습니다. 또 GPT 같은 LLM은 few-shot만으로 잘 작동하는 사례가 있지만, 과업별 적응이 충분치 않으면 hallucination과 부정확한 스팬 선택 문제가 남았습니다.

- **Core Contribution**: 본 논문은 FinCausal 2026에서 “원문 스팬 기반 인과 QA”를 extractive question answering으로 풀되, 인코더·인코더-디코더·decoder-only(LLM) 세 계열을 체계적으로 비교합니다. 특히 decoder-only LLM을 prompt optimization, few-shot 시연, 그리고 supervised fine-tuning까지 결합해 인과 스팬 추출의 정밀도를 끌어올리는 전략을 제안합니다. 영어-스페인어 병용 데이터로 multilingual fine-tuning을 적용해 교차 언어 전이를 강화한 점이 핵심 기여입니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 질문이 원인/결과를 정확히 지시할 때 해당 이벤트의 정확한 스팬을 집어내고, (2) 문맥 밖 정보를 생성하는 hallucination을 억제하며, (3) 중첩 인과나 복수 원인처럼 구조가 복잡한 예에서 올바른 쌍을 선택하는 것입니다. 이를 위해 멀티링구얼 임베딩 기반 cosine similarity로 few-shot 예제를 동적으로 고르고, prompt 최적화를 통해 “문맥에서 그대로 복사”하도록 출력 제약을 강화했으며, 최대 2,000개 샘플로 영어/스페인/혼합 fine-tuning을 실험해 지침 적응을 보강했습니다.

- **Empirical Impact**: 실험 결과, encoder-only(BERT)·encoder-decoder(mBART)보다 fine-tuned generative 모델이 전반적으로 우수했으며, prompting+few-shot은 경쟁력 있는 성능을 보였습니다. 최종적으로 GPT-4.1 Mini에 영어+스페인어 학습 데이터를 섞어 fine-tuning한 모델이 English 하위과제에서 동률 1위(점수 4.8140), Spanish에서는 3위(4.7753)를 기록하며 LLM-as-a-judge(적절성 1~5) 기준에서 강한 성능을 입증했습니다. 또한 zero-shot에서는 더 큰 모델보다 적정 규모의 fine-tuned 모델이 낫게 나와, 매개변수 스케일보다 task-specific fine-tuning이 금융 인과 QA에서 중요하다는 시사점을 줍니다.



### A Survey of Automated Presentation Coaching: Systems, Methods, and Open Challenges (https://arxiv.org/abs/2606.27380)
Comments:
          accepted into the BEA 2026 workshop at ACL

- **Prior Approaches**: 자동 프레젠테이션 코칭은 CAPT 기반 발음 교정, 운율·유창성 분석, TTS를 활용한 모범 발화 생성 등으로 발전해 왔지만, 각 시스템은 특정 요소에만 집중하는 경향이 강했다. 또한 대부분이 고립 발화나 짧은 읽기 문장 중심의 평가를 사용해, 슬라이드 전환을 포함한 장문·준스크립트 말하기의 특성을 충분히 다루지 못했다. 결과적으로 어떤 차원에서 커버리지가 되는지, 아직 어떤 문제가 남았는지 한눈에 비교하기 어려웠다.

- **Core Contribution**: 이 논문은 구두 발표 코칭 과제를 5개 차원(분절 발음, 어휘 강세, 초분절 운율, 페이싱, 내용 충실도)으로 정리한 과제 택소노미를 제안하고, 기존 시스템들을 이 프레임에 매핑해 커버리지 공백을 드러낸다. 더 나아가 TTS 기반 exemplar 생성과 발음·운율·유창성 진단이라는 핵심 기술 묶음을 체계적으로 정리해, 연구 간 비교 기준을 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 발표형 장문 데이터를 포함한 주석 부족, (2) 다양한 L1 배경에서 accent-fair한 피드백 제공, (3) 실시간 리허설을 위한 저지연 진단 구성이다. 저자들은 이를 위해 TTS로 속도·강조·일시를 제어한 모범 발화를 만들고, GOP/CTC 기반 진단 및 log-F0 기반 운율·WPM·휴지 지표, ASR+용어 제약으로 내용 충실도를 측정하는 조합을 논의한다.

- **Empirical Impact**: 실증 평가는 다섯 택소노미 차원과 직접 연결되는 지표 체계를 제안함으로써, 단일 측면이 아니라 발표 코칭 전반을 일관되게 비교할 수 있게 한다. 다만 현재 공개 코퍼스는 L2 말하기·발표 구조·억양 라벨·운율 주석을 동시에 충족하지 못해, 향후 5~10분 규모의 미니-세트 벤치마크 필요성을 구체적으로 제시한다. 이는 발표 코칭 분야가 ‘부분 성능’에서 ‘통합 평가·통합 코칭’으로 이동하는 데 기여할 전망이다.



### Position: The Term "Machine Unlearning" Is Overused in LLMs (https://arxiv.org/abs/2606.27379)
Comments:
          13 pages; ICML 2026 Position Paper Track. Sangyeon Yoon and Yeachan Jun contributed equally

- **Prior Approaches**: 기존 LLM ‘unlearning’ 연구는 규제·저작권·안전 이슈로 인해 특정 데이터의 영향 제거가 요구된다는 공통 동기 아래, 거절(refusal), 억제(suppression), 편집(editing), 난독화/혼란(obfuscation), 추론 시 차단(guardrails/filters) 등 매우 다른 목표를 한 용어로 묶어 왔습니다. 이 때문에 많은 벤치마크가 ‘forget set’에 대해 정답 재현 실패 같은 출력 중심 지표만 측정하며, 재학습 기준선과의 유사성(훈련 영향 제거의 보장)을 검증하지 않는 경우가 잦았습니다.

- **Core Contribution**: 이 논문은 ‘machine unlearning’을 dataset-defined deletion으로 한정해, forget set F의 ‘학습 영향’을 정확히(또는 근사적으로) 제거했을 때 retraining-from-scratch(데이터 D\F로 재학습) 모델과 동등해지는 상태로 정의합니다. 동시에, 거절·지식/엔터티 제거·표적 억제 등 정책 의존적 행동 변경은 별도 용어(예: alignment, suppression, editing, obfuscation 등)로 분리해야 한다고 주장합니다. 핵심은 라벨 혼동이 평가 해석을 망치며, 실제로는 ‘훈련 영향 제거’가 아니라 ‘출력 통제’가 성공 신호로 보상될 수 있다는 점을 정리한 데 있습니다.

- **Technical Challenges**: 정확한 unlearning의 조건은 대규모 모델에서 현실적으로 매우 강해, 논문은 (approximate) indistinguishability처럼 명시적 거리/분산 기준(행동 공간 또는 파라미터 공간)을 둔 완화 정의를 제시합니다. 기술적으로는 forget set 제거의 기준을 출력 실패가 아니라 ‘훈련 없이 학습했을 때(reference)와의 분포·확률·행동 유사성’으로 고정해야 하며, 특히 파생 능력(derived capability)이 남아 있을 수 있다는 점을 포착하는 평가 설계가 필요하다고 강조합니다. 따라서 단순 ROUGE/정확도/문구 차단만으로는 부족하고, 기준선 대비 분포 비교와 강건성(패러프레이즈·적대적 elicitation 등)까지 함께 보아야 한다고 말합니다.

- **Empirical Impact**: 논문은 기존 벤치마크들이 ‘재학습 기준선’ 없이 출력 기반 지표로 성공을 판정하면서, 억제·거절이 실제로는 unlearning이 아닌데도 높은 점수를 받는 문제를 여러 예(TOFU 계열의 기준선 생략, RWKU의 기준선 부재, WMDP의 능력 억제 프레이밍 등)로 구체화합니다. 또한 파생 능력 관점에서 ‘답하지 않음’이 훈련 영향 제거의 대리(proxy)가 될 수 없고, 특히 poisoning 같은 설정에서는 유도된 행동이 남는지를 봐야 평가가 의미를 갖는다고 주장합니다. 결론적으로, ‘참조 모델(reference-based) + 파생 능력 프로브 + 분포/확률 기반 비교’로 평가 관행을 재정렬하자는 제안이 분야의 후속 연구 설계에 직접적인 방향성을 줄 것으로 보입니다.



### Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs (https://arxiv.org/abs/2606.27378)
Comments:
          44 pages, 27 tables, 14 figures

- **Prior Approaches**: 연속적 thought representation(연속 사고 표현)은 기존에 downstream benchmark 정확도나 토큰/단계 압축 성능으로만 주로 평가돼 왔다. 그 결과 표현이 정말 문제인지, 아니면 이를 처리하는 디코더·프롬프트 탓인지 구분하기 어려웠고, 일부 probing 연구는 서로 다른 추론 경로가 내부에서 붕괴해도 정확도는 잘 유지된다고 지적했다.

- **Core Contribution**: 논문은 LLM의 latent thought representation을 “표현 자체의 기능”으로 보려는 공리(axiom) 기반 평가 프레임워크를 제안한다. Causality(인과성), Minimality(최소성), Separability(분리성), Stability(안정성) 4가지를 정의하고, downstream 점수와 무관하게 표현에 직접 계산되는 정량 지표를 설계해 표현 결함을 정확도 마스킹과 분리해 드러내도록 했다.

- **Technical Challenges**: 핵심 난제는 표현 품질을 디코딩/정확도 성능에서 분리해, LLM 내부 표현이 출력 분포를 어떻게 매개하는지를 “재학습 없이” 측정하는 것이었다. 논문은 토큰·벡터 형태와 무관하게 작동하는 기능적 대체(substitution)와 정보이론적 제약(Minimality), 분별기 기반의 기능적 주입성(Separability), 그리고 어휘적 변이·모드 붕괴에 대한 분포 일치(DCS로 Stability)를 조합해 네 지표를 구현했다.

- **Empirical Impact**: BBEH의 23개 추론 태스크에서 5개 오픈 웨이트 LLM을 대상으로 Soft Thinking(노이즈 유무), Latent Thinking, LIT 계열(최근 입력 토큰) 등 다양한 후보를 감사(audit)한 결과, 어떤 후보도 4개 공리를 동시에 만족하지 못했다. 또한 태스크 종류는 거칠게 구분하지만 같은 태스크 내 서로 다른 질문의 정체성은 구분하지 못하며, input embedding에 이미 있는 정보 이상은 거의 인코딩하지 못하는 구조적 실패가 모델 크기·학습 절차(밀집/희소 MoE, reasoning-distilled, RL-trained) 전반에서 일관되게 관찰됐다.



### Towards Automating Scientific Review with Google's Paper Assistant Too (https://arxiv.org/abs/2606.28277)
- **Prior Approaches**: 기존에는 LLM을 논문 전체에 한 번 호출해 결함을 찾는 방식이나, Pass@k처럼 여러 번 생성한 뒤 합치는 접근이 주로 거론됐다. 하지만 이 방법들은 문맥 제약 때문에 깊은 검증이 어려우면서도, 환각으로 정밀도가 급락해 사람이 쓸모없는 지적을 걸러내야 하는 문제가 컸다. 또 무작위 호출 분산이라 논문의 어느 구간은 과소검증되고 다른 구간에 자원이 쏠릴 수 있다.

- **Core Contribution**: 이 논문은 AI가 과학 검증/리뷰를 대신하는 단계적 역할 분류(4단계)를 제안하고, 그중 “Paper Assistant Tool(PAT)”로 실제 구현 사례를 제시한다. PAT는 수학 및 컴퓨터과학 논문을 대상으로 이론/논리 오류와 실험 검증 문제를 중심으로, 개선점과 잠재적 결함을 체계적으로 산출한다. 핵심은 단일 호출을 넘어 인퍼런스 스케일링을 파이프라인 오케스트레이션에 결합해 더 깊은 오류 탐지 확률을 끌어올린다는 점이다.

- **Technical Challenges**: 문제는 긴 논문 전체를 한 번에 깊게 검증하는 데 필요한 “생각 토큰”과 문맥 한계가 충돌한다는 것이다. PAT는 세그먼터(agent)가 논문을 논리 주제별로 분할(겹침/비연속 가능)하고, 각 세그먼트의 정보 밀도·복잡도에 따라 계산 예산을 동적으로 배분한 뒤, Deep Review agent들이 각 구간을 검증하게 하며, 마지막 합성 에이전트가 중복 제거와 Google search 기반 근거 확인으로 정밀도 저하를 완화한다. 그 결과 단순 Pass@k의 정밀도 하락과 문맥 편향을 동시에 겨냥한다.

- **Empirical Impact**: SPOT 벤치마크에서 수학/증명 오류 하위셋(26편, 29오류)을 사용한 실험에서 PAT는 zero-shot 단일 호출 대비 수학 오류 탐지 recall을 34%p 개선해 89.7%까지 끌어올렸다. 또한 STOC·ICML에 사전 제출용 도구로 시범 적용한 결과, 대부분의 저자가 유용성을 높게 평가했으며(90% 이상이 도움이 되었다고 응답), 일부는 이론 결과의 유의미한 오류를 수정해야 했다고 답했다. 특히 ICML에서는 PAT가 “완전히 새로운 실험”으로 이어졌다는 응답이 31%로 나타나, 오류 탐지뿐 아니라 실험 설계 품질 개선에도 실질적 영향이 있음을 시사한다.



### HPRO: Hierarchical Progressive Reward Optimization via Preference Extraction for Emotional Text-to-Speech (https://arxiv.org/abs/2606.28249)
Comments:
          7 pages, 3 figures, 3 tables; Preprint

- **Prior Approaches**: 감정이 포함된 TTS에서는 SFT가 평균적인 억양으로 수렴해 감정 표현이 평탄해지는 ‘regression to the mean’ 문제가 지적돼 왔다. 선호 기반 최적화/강화학습 계열은 감정 점수를 올리려 하지만, 보상 해킹과 의미(발화 내용) 열화가 함께 나타나는 구조적 불일치가 반복됐다. 특히 정보(내용)와 감정이 하나의 잠재 공간에서 경쟁하고, 문장 수준의 희소 보상이 프레임 단위 생성에 충분히 전달되지 못하는 scale gap이 남아 있었다.

- **Core Contribution**: 이 논문은 계층적 progressive reward optimization 프레임워크 HPRO를 제안해, 감정 최적화가 의미를 훼손하는 문제와 보상-생성 스케일 불일치를 동시에 겨냥한다. HD-Emo codec(차분 가능한 보상 모델)을 통해 발화의 내용과 스타일(감정 선호)을 서로 다른 preference 토큰 공간으로 분리해 정보 conflict를 구조적으로 완화한다. 또한 프레임-단어-문장 수준 목표를 단계적으로 연결해 scale gap을 ‘연속적인 gradient bridge’로 다리 놓는다.

- **Technical Challenges**: 핵심 난제는 (1) 감정 점수 극대화가 발화 내용을 깨뜨리는 reward hacking을 막고, (2) 희소한 문장/감정 보상이 조밀한 프레임 생성에 제대로 신호를 주도록 credit assignment를 해결하는 것이다. 저자들은 HD-Emo codec에서 FSQ 기반 정보 병목을 통해 내용 토큰은 ASR로 고정(semantic anchor), 스타일 토큰은 emotion2vec/ wVAD 같은 감정 관련 제약으로 학습해 감정 최적화 경로를 분리했다. 더해 HPRO는 frame-level warm-up → word-level refinement → sentence-level alignment로 학습 스코프를 점진 확장하고, Gumbel-Softmax 온도 annealing으로 안정적인 수렴을 유도한다.

- **Empirical Impact**: 실험은 LSSED와 EmoVoice-DB에서 zero-shot TTS 세팅으로 수행됐으며, 주관평가 MOS-N(자연스러움)과 MOS-E(감정-의미 일관성)에서 HPRO가 전반적으로 가장 균형 잡힌 성능을 보였다. 객관지표에서도 WER이 가장 낮은 수준(4.02%)으로 의미 열화가 억제됐고, wVAD-CCC(0.339)와 EMO-SIM(0.672)로 감정·미세 억양 정합성이 높게 나타났다. 특히 진행형(프로그레시브) 계층 보상이 단일 스케일(global) 보상 방식보다 감정 표현과 언어 intelligibility를 함께 끌어올린다는 점이 ablation에서 확인돼, 감정 TTS의 최적화 설계를 한 단계 진전시킨 것으로 평가된다.



### Scaling limit of the Random Language Mod (https://arxiv.org/abs/2606.28105)
Comments:
          17 pages + 14 pages SI

- **Prior Approaches**: 기존 연구는 Zipf’s law·Heaps’ law·엔트로피율처럼 언어 통계의 거친 관측값을 주로 실증적으로 다뤘고, 생성 문법(특히 context-free grammar)이 주는 구조적 제약과의 정량적 연결은 제한적이었습니다. Random Language Model(RLM)은 생성 문법의 앙상블로부터 전형적인 언어 통계가 보편 법칙을 형성할 수 있음을 제안했지만, 저온(저온상) 위상에 대한 제어 가능한 이론은 부족했고 특정 임계점의 명확성도 떨어졌습니다.

- **Core Contribution**: 이 논문은 N→∞와 grammar temperature tilde{\epsilon}_d→0을 동시에 보내되 x=tilde{\epsilon}_d log N을 고정하는 스케일링 한계를 도입해 RLM을 정량 이론으로 정리합니다. 그 결과 RLM이 rule-usage(규칙 사용 패턴) 관점의 large-deviation 원리로 제어되며, semi-annealed 근사를 통해 Random Energy Model(REM) 계열의 문제로 환원됨을 보입니다. 또한 condensation transition이 x_c=1/8에서 나타나며, x=1/2에서는 엔트로피가 최대값에서 감소하기 시작하는 두 번째 특징 스케일을 제시합니다.

- **Technical Challenges**: 핵심 난제는 저온 위상에서 나타나는 동적 범위(규칙 사용이 특정 패턴에 집중되는 현상)를 단순 saddle-point나 replica-symmetric 접근만으로는 설명하기 어렵다는 점입니다. 논문은 N이 유한일 때의 grammar/derivation의 유한성을 반영하는 semi-annealed 근사와 double scaling limit을 결합해, LLN이 깨지는 조건을 REM의 에너지 분산 경쟁으로 해석하고 패턴(occupation-number 유사)의 조합론을 트리 구조와 surface layer까지 포함해 계산합니다.

- **Empirical Impact**: 이론은 corpus 길이와 grammar 크기에 따른 distinct rules의 수, 엔트로피, 관련 관측값의 스케일링 법칙을 도출하며 수치 실험과 좋은 일치 결과를 보고합니다. 특히 condensation 전후로 규칙 사용이 집중/비집중으로 바뀌고, large N 극한으로의 느린 수렴이 log N 의존성에서 비롯된다는 점까지 설명해 기존의 ‘진짜 열역학 전이 존재 여부’ 논쟁을 정리하는 데 기여합니다. 나아가 자연어와 large language models(LLMs)의 관측되는 Heaps’ law 및 엔트로피의 문맥 길이 의존성을 같은 프레임에서 재현함으로써, 생성 문법 앙상블의 보편 통계가 어떻게 emergent되는지에 대한 통합적 근거를 제공합니다.



### Single and Multi Truth Data Fusion using Large Language Models (https://arxiv.org/abs/2606.28062)
- **Prior Approaches**: 데이터 퓨전(또는 truth discovery)은 여러 출처의 상충 값을 바탕으로 속성의 정답(단일 또는 다중)을 추정하는 데이터 통합 문제로, 기존 연구는 대부분 conflict-resolving에 초점을 맞춰 왔다. 대표적으로 Majority Voting, Source Reliability Vote, LTM, DART처럼 출처 신뢰도나 도메인별 전문성을 가정(혹은 모델링)하며 반복적·확률적 추정으로 정답을 고른다. 하지만 이런 방식은 ‘단일 정답’ 가정에 편향되거나, 문맥에 따른 의미 차이를 유연하게 반영하지 못하고 표현 다양성(표기 변형/정규화)을 별도로 다뤄야 하는 한계가 있다.

- **Core Contribution**: 이 논문은 LLM을 데이터 퓨전의 truth-discovery 구성요소로 직접 사용해, 단일-truth/다중-truth을 모두 다루는 prompt 기반 접근을 체계적으로 탐구한다. 특히 domain-independent(DI) vs domain-dependent(DD), zero-shot vs one-shot의 조합을 만들고, 다중-truth에서는 여러 값을 함께 정답으로 산출하도록 유도하는 프롬프트를 설계했다. 또한 정답 생성 제한(예: 입력에 있는 값만 사용) 같은 제약을 프롬프트에 포함해 동작을 조절하고 신뢰성까지 함께 분석한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 출처 간 의미적으로 같은 값을 다른 표기/형식으로 제시할 때 이를 자연어 의미 수준에서 정렬하고, (2) 다중-truth 설정에서 ‘여러 개가 맞을 수 있음’을 LLM 출력 규칙으로 정확히 반영하는 것이다. 저자들은 이를 위해 단일/다중-truth 전용 프롬프트 구조를 분리하고, 값은 소스에 존재하는 것만 선택하도록 하는 C1 제약, 형식이 다른 동일 값은 하나로 세도록 하는 C2 제약을 두어 출력의 제약을 걸었다. 더불어 one-shot 예시를 프롬프트 앞에 선택적으로 추가해 문맥 추론과 출력 형식의 안정성을 높이도록 했다.

- **Empirical Impact**: 실험은 Book(Movie의 감독/연도 계열), Movie, Flight 데이터셋(각각 다중-truth 2개, 단일-truth 1개)에서 Recall/Precision/F1로 비교했으며, LLM 기반 DD 프롬프트가 전통적 무지도 truth discovery(예: DART, LTM)를 전 데이터셋에서 전반적으로 앞섰다. 특히 Book과 Movie에서는 domain-dependent 프롬프트가 더 높은 F1과 균형 잡힌 recall/precision을 보였고, Flight에서는 단일-truth 구조 덕에 베이스라인도 강했지만 DD/DI 프롬프트가 근접 성능을 내며 LLM의 일관성을 확인했다. 또한 Flight ID를 obfuscation해도 성능 하락이 작아, LLM이 배경지식보다 출처 일치 패턴에 더 의존할 가능성을 시사했으며, 비용은 API 호출당 수 초 수준으로 보고했다.



### DG^VoiC: Speaker Clustering for Fraud Investigation under Real Call-Centre Conditions (https://arxiv.org/abs/2606.28048)
Comments:
          5 pages, 4 figures, 1 table

- **Prior Approaches**: 기존 보험 사기 탐지는 주로 구조화 데이터, 텍스트, 이미지 같은 단일·멀티모달 단서에 의존하며, 콜 간 반복되는 화자 정체성은 상대적으로 덜 활용돼 왔다. 전화 사기 연구도 대체로 대화 내용(전사/의미)에 초점이 맞춰져 있고, 실제 콜센터 오디오 데이터는 개인정보·생체정보 제약으로 연구가 제한적이었다. 콜센터 맥락의 음성 기반 연구는 인증·다이어라이제이션 중심이어서, 고객 프로필을 가로지르는 화자 클러스터링을 통한 교차 연결 문제와는 평가 목표가 달랐다.

- **Core Contribution**: 이 논문은 익명화된 실제 보험 콜센터 오디오에서 ‘고객 검증’과 ‘고객 프로필 간 반복 화자 연결’을 동시에 지원하는 DG^VoiC를 제안한다. 단독 사기 판정기가 아니라, 분석가가 화자 일관성을 확인하고 반복 목소리를 찾아내도록 음성 기반 링크 신호를 노출하는 데 목적을 둔다. 민감 정보 정렬 기반 익명화와 음성 중심 전처리, 슬라이딩 윈도 임베딩 추출, 코사인 유사도 기반 클러스터링을 end-to-end 파이프라인으로 결합했다.

- **Technical Challenges**: 핵심 기술 난제는 콜센터에서 길고 길이가 다양한 통화, 잡음·무음 구간, 그리고 익명화 처리로 인해 화자 임베딩이 흔들릴 수 있다는 점이다. DG^VoiC는 NER·Regex 및 WhisperX 타임스탬프를 활용해 PII 구간을 직접 마스킹하고, Resemblyzer 전처리로 무음/저정보 구간을 제거한 뒤 ECAPA-TDNN으로 임베딩을 구한다. 또한 오버랩 슬라이딩 윈도와 최소 구간 배제를 통해 짧은 유효 발화를 놓치지 않게 하면서, 평균 풀링 및 코사인 유사도 임계값(최적 0.718)으로 안정적인 클러스터를 형성한다.

- **Empirical Impact**: 실제 121개 콜 중 전문가가 합의한 56개(22개 화자 클러스터) 기준으로 평가했으며, 최적 설정에서 AMI 96%, ARI 95%, completeness 98%, homogeneity 100%, V-measure 99%를 달성했다. 또한 보조적으로 EER 3.85%(FAR 0.50%, FRR 9.62%)를 제시해 임계값에서의 검증 관점 성능도 확인했다. 결과적으로 반복 화자 클러스터링이 분석가 검토용 교차 프로필 음성 링크를 효과적으로 부각할 수 있음을 보여, 음성 기반 사기 조사 워크플로의 추가 신호로 활용될 가능성을 제시한다.



### AI Persuasive Framing in Collective Dilemmas (https://arxiv.org/abs/2606.27951)
Comments:
          The first two authors contributed equally to this research. The article contains 20 pages, 10 figures, and 2 tables

- **Prior Approaches**: 기존 연구는 집단적 딜레마에서 협력을 좌우하는 핵심 요인으로 정보·규범 같은 환경 신호를 강조해 왔고, AI 개입도 대체로 정보 병목을 메우는 방식에 치우친 편입니다. 반면 설득·개인화된 대화로 행동 변화를 “트리거”하는 persuasive AI 연구는 같은 맥락에서 실험적으로 교차 검증된 사례가 적었습니다. 그 결과 AI가 협력을 높일지에 대한 증거는 혼재된 채로 남아 있습니다.

- **Core Contribution**: 이 논문은 iterated Collective Risk Game(집단 위험 게임)에서 AI 어시스턴트가 기여(pledge)를 늘려 협력 성공을 유도하는지, 그리고 그 효과가 개인화와 설득 방향에 따라 어떻게 달라지는지를 실증합니다. 1,283명을 대상으로 LLM 기반 AI가 prosocial(친사회) 프레이밍을 개인의 Social Value Orientation(SVO)에 맞춰 설계했을 때와, 반대로 exculpatory framing으로 selfish(이기적) 행동을 유도했을 때의 대비를 보여줍니다. 핵심은 “친사회 설득의 효과”뿐 아니라 “역방향 설득의 부작용”이 더 크고 오래 갈 수 있음을 체계적으로 드러낸 점입니다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) 협력 게임 맥락에서 설득이 실제 행동 수정으로 이어지게 만드는 인터랙션 설계와 (2) 플레이어 성향(SVO)별로 프레이밍 방향을 안전하게(혹은 정확히) 개인화하는 것입니다. 연구진은 static prognostic message(정적 넛지)와 달리, 초기 선택→AI 대화 개입(최소 30초)→재고 및 최종 결정의 3단계를 두 조건에 공통 적용해 ‘반응형 설득’의 효과를 분리했습니다. 또한 SVO 유형(협동/개인주의/경쟁 성향)을 기반으로 주장(의무·보험·집단 돌봄 vs 개인 이득 극대화 vs 상대적 이득 극대화)을 바꾸고, 설득 방향을 친사회/이기적으로 orthogonal하게 실험해 비대칭성을 측정했습니다.

- **Empirical Impact**: 실험 결과, 친사회 방향의 AI 설득은 정적 넛지보다 더 큰 폭으로 기여와 그룹 성공률을 유의미하게 끌어올렸으며, AI와의 채팅 시간이 길수록 효과가 더 강하게 나타났습니다. 다만 그 협력 촉진 효과는 초반 라운드 이후 빠르게 약해져 단기적이라는 패턴도 확인했습니다. 반대로 이기적 설득(자기 이익 강화)은 기여와 성공률을 더 크게 떨어뜨리며, 특히 개인화된 처리에서 부정 효과가 훨씬 더 지속되었습니다. 이로써 prosocial vs antisocial persuasion의 비대칭이 집단 행동(collective action) 영역에서 AI의 dual-use 위험을 실질적으로 높일 수 있음을 시사합니다.



### Verifiable Geometry Problem Solving: Solver-Driven Autoformalization and Theorem Proposing (https://arxiv.org/abs/2606.27926)
- **Prior Approaches**: 기존 Geometry Problem Solving(GPS) 연구는 neuro-symbolic 패러다임을 채택하지만, autoformalization과 theorem prediction이 각각 고립된 정적 단계로 분리된 경우가 많다. 그 결과 다이어그램·텍스트의 모호성을 충분히 반영하지 못한 채 언어적 정확도에만 맞춰져 solver 실행 가능성과 불일치하는 문제가 자주 발생한다. 또한 theorem prediction은 고정된 theorem library와 제한된 search budget 때문에 deductive impasse에 빠지면 이를 국소 보강하기 어렵다.

- **Core Contribution**: SD-GPS는 symbolic solver를 autoformalization과 deduction 전 과정에서 execution oracle로 사용하는 solver-driven 폐루프를 제안한다. solver가 후보가 “실제로 실행되고 목표에 기여하는지”를 피드백으로 주도하며, 이를 통해 형식화의 기준을 표면 정합성에서 실행 가능성으로 옮긴다. 아울러 impasse-aware verified theorem proposing으로, 신경망이 제안하더라도 symbolic verification을 통과한 보조 lemma만 유효하게 반영되도록 설계한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) multimodal 입력을 solver가 즉시 받아들일 수 있는 typed predicate로 변환하면서, (2) theorem 라이브러리 고정으로 생기는 막힘을 “검증 가능한” 방식으로 완화하는 것이다. SD-GPS는 QwenVL3-2B 기반 통합 multimodal formalizer로 supervised adaptation과 solvability-guided reinforcement learning(SG-RL)을 결합해, 파싱·실행·답 충분성에 기반한 보상으로 학습을 구동한다. theorem proposing 단계에서는 현재 proof state 요약을 바탕으로 국소 보조 lemma를 제한적으로 제안하되, solver가 타입 제약·모순·현재 상태 인스턴스 가능성 등을 검증한 뒤에만 사용하며, 스코프도 해당 문제 시도에 한정해 soundness를 유지한다.

- **Empirical Impact**: Geometry3K와 PGPS9K에서 SD-GPS는 completion과 choice, 그리고 cross-modal reference/불일치 구간까지 일관되게 기존 MLLM·neural·neuro-symbolic 방법을 능가한다. 예를 들어 Geometry3K에서 completion 86.4%, choice 90.4%를 기록해 최고 대비 각각 +3.5%p, +3.2%p 개선했고, PGPS9K에서도 completion 79.8%, choice 84.5%로 각각 +4.4%p, +3.0%p 향상했다. ablation 결과 solver-유도 RL, bounded repair, verified theorem proposing을 순차 추가할수록 실행 성공률이 증가했으며, 특히 cross-modal mismatch에서 execution feedback의 효과가 크게 나타나 perception-logic 결합의 실질적 가치를 보여준다.



### Joint Transcription and Decryption of Images of Encrypted Handwritten Documents: A Comparison with the Traditional Pipelin (https://arxiv.org/abs/2606.27700)
Comments:
          Published at HistoCrypt 2026 (9th International Conference on Historical Cryptology). NEALT Proceedings Series Number 61. Tartu University Library. 10 pages

- **Prior Approaches**: 역사 암호 해독은 보통 이미지에서 암호 기호를 전사(transcription)한 뒤, 그 기호열을 바탕으로 복호화(decryption)하는 2단 파이프라인을 사용한다. 이 방식은 전사 단계의 작은 실수가 복호화로 그대로 전이되며, 전사를 사람이 하거나 고품질 라벨이 필요해 확장성이 떨어진다. 또한 합성 데이터 의존도가 높아 synthetic-to-real gap 문제도 함께 제기돼 왔다.

- **Core Contribution**: 이 논문은 전사를 거치지 않고 암호화된 원고 이미지에서 바로 평문을 생성하는 end-to-end 모델인 Direct Image Decryption을 제안한다. Copiale cipher를 대상으로, 전사 대신 이미지-평문 직접 매핑으로 전사 오류 전이를 줄이고 시각 정보 손실을 완화하는 것이 핵심 목표다. 단, 특정 치환 규칙에 대한 학습(cipher-agnostic이 아님)을 전제로 한다.

- **Technical Challenges**: Direct Image Decryption에서 가장 큰 기술적 난제는 ‘연속적인 시각 표현’ 위에서 복호화를 학습해, 중간의 discrete 기호 선택을 생략하는 구조를 안정적으로 최적화하는 것이다. 이를 위해 CRNN 기반 시각 인코더와 attention 기반 문자 디코더를 결합하고, 디코더 학습과 동시에 인코더를 end-to-end로 fine-tuning해 전사 오차 없이 전체 그래디언트 흐름을 유지한다. 아울러 Copiale-like 합성 데이터를 115,000 라인 규모로 생성해 대규모 학습이 가능하게 했으며, 노이즈·잉크 번짐·노화 흔적 같은 열화 증강으로 현실감도 확보했다.

- **Empirical Impact**: 합성 데이터에서는 Direct Image Decryption이 전통적 2단 파이프라인보다 대부분 지표에서 우수하며, 토큰 정확도는 1.1%p, WER은 약 49% 감소로 전사 병목 제거 가설을 지지한다. 분포 밖 데이터(Novalis)에서도 정확도 절대값 +6.3%p로 end-to-end 학습의 일반화 이점이 관찰됐다. 다만 실제 Copiale 원고에서는 전반 성능이 크게 떨어지는데, 이는 질적 합성-현실 차이보다 ‘실데이터 희소성’이 주 원인으로 분석되며(학습 데이터가 115k→20k로 줄면 정확도가 크게 하락), 이런 상황에서도 Direct Image Decryption이 11.8%p 개선을 보이며 전사 단계 제거의 실용적 가치가 확인됐다.



### Textual Belief States for World Models: Identifiable Representation Learning Under Strict Mediation (https://arxiv.org/abs/2606.27681)
- **Prior Approaches**: 현대 LLM 기반 월드 모델은 부분관측에서 히스토리를 압축한 잠재표상을 쓰지만, 많은 구조가 히스토리 우회(history bypass)를 허용해 예측 정확도가 표상 품질을 보장하지 못합니다. 이 때문에 예측은 잘 맞추면서도 잠재상태가 사실상 무정보일 수 있는 식별성(identifiability) 문제가 생깁니다. 또한 텍스트 도메인에서 연속 잠재변수+variational 학습(예: ELBO)은 이산/비미분 잠재상태와 posterior collapse 위험 때문에 제약이 커졌습니다.

- **Core Contribution**: 논문은 엄격한 잠재상태 매개(strict latent state mediation) 원칙을 텍스트 기반으로 구현하는 방법을 제시합니다. 이를 통해 예측이 잘 맞으면 잠재상태가 히스토리에 대한 충분통계(sufficient statistic)라는 연결고리를 만들고, 히스토리 유출(leaky) 구조에서는 이 연결이 깨진다는 점을 이론적으로 정리합니다. 실무적으로는 이산적이고 해석 가능한 텍스트 잠재상태를 정의하고, 학습 시 이를 강제로 매개하도록 하는 factorized GRPO(fGRPO) 알고리즘을 제안합니다.

- **Technical Challenges**: 핵심 난제는 텍스트 잠재상태가 이산·비미분이라 ELBO 같은 미분 기반 변분학습이 어렵고, 표현력이 큰 디코더는 병목을 무시하며 posterior collapse로 이어질 수 있다는 점입니다. 논문은 잠재상태 생성 단계를 확률적 정책으로 보고 강화학습으로 학습하며, strict mediation을 유지하기 위해 트리 구조의 트레이닝(encoding→transition→decoding을 서로 다른 프롬프트 컨텍스트로 분리)을 fGRPO로 구현합니다. 보상 설계에는 관측 사실 예측 F1, 포맷 유효성, 중복 억제 및 압축 최소화 유도 같은 보상 항을 결합해 “예측만 잘 맞추는” 우회적 해법을 차단합니다.

- **Empirical Impact**: TextWorld와 ScienceWorld에서 strict mediation 모델은 1-step 예측정확도는 보존하면서도, 매칭된 예측 품질 조건에서 state-level F1은 더 높이고 롤아웃 성능이 크게 개선됨을 보였습니다. 특히 롤아웃에서 horizon이 길어질수록 격차가 커져 최대 57% 수준의 표상 품질 향상과 최대 98% 수준의 롤아웃 개선이 보고되며, 과제 복잡도와 horizon에 따라 이득이 증가합니다. 결과는 예측 정확도만으로는 표상 품질을 판단하기 어렵던 문제를, 엄격 매개 설계로 “실험적으로 검증 가능한 형태”로 바꿀 수 있음을 시사합니다.



### DysLexLens: A Low-Resource LLM Framework for Analysing Dyslexic Learners Insights from Online Forums (https://arxiv.org/abs/2606.27619)
- **Prior Approaches**: 기존 연구는 크게 (1) 난독증 지원에 대한 기술·도구 현황을 정리하거나 (2) 레딧 같은 온라인 담화를 통해 난독증 당사자의 경험을 주로 정체성/사회적 의미 관점에서 분석하는 방향으로 나뉘었다. 하지만 두 접근 모두 AI 도구가 실제로 어떻게 인식·채택·검증되는지에 대한 학습 관련 근거를 자연스러운 포럼 맥락에서 체계적으로 연결해 주지 못했다.

- **Core Contribution**: 이 논문은 DysLexLens를 제안한다. DysLexLens는 저자원이면서 잡음이 많은 포럼 텍스트를 수집-필터링-지식그래프(KG) 추론-증거 추적-응답 평가의 end-to-end 파이프라인으로 묶어, AI 관련 발화의 학습적 유스케이스와 장단점을 근거 기반으로 분석하도록 한다.

- **Technical Challenges**: 핵심 난제는 (i) 관련 글이 희소하고 잡음이 많아 표본을 정밀하게 구성해야 하며 (ii) LLM이 만든 답을 실제 포럼 증거에 “traceable”하게 연결해야 한다는 점이다. DysLexLens는 딕셔너리 기반 필터링으로 Reddit 코퍼스를 319개 게시물 수준으로 좁힌 뒤, LLM이 KG triple과 의미 분석을 수행하고, 응답의 claim을 chunk ID로 매칭해 원문까지 역추적하는 evidence-tracing 파이프라인을 제공한다.

- **Empirical Impact**: 평가는 난독증- AI 관련 연구질문 RQ1~RQ5에 대한 30개 질의로 수행했으며, Answer Relevancy는 평균 0.75로 전반적 질의 의도를 잘 반영했다. 다만 Faithfulness·Context Relevance·Response Groundedness는 상대적으로 낮아(각각 평균 0.52/0.40/0.43) 추론은 맞지만 claim 수준에서의 근거 매칭과 시간 변화 질의 처리에 한계가 드러났다. 그럼에도 29개 응답에서 최소 1개 chunk ID가 포함되고, 인간 검증에서 39개 claim은 완전 검증 가능, 6개는 증거 부재로 비검증으로 분류되어, “근거 확인 가능한 생성” 방향의 실용성을 보여주며 Github 공개로 재현성도 강화했다.



### The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching (https://arxiv.org/abs/2606.27510)
- **Prior Approaches**: 메커니즘 해석에서 activation patching은 구성요소별로 동작에 대한 인과적 책임을 분해해, 기준선(오염)에서의 출력 변화로 natural indirect effect(NIE)를 추정하는 핵심 도구로 자리 잡았다. 기존 연구들은 이 NIE가 “해당 구성요소를 통해 흐르는 인과효과”를 잘 반영해 구성요소 중요도를 단일 순위로 산출한다고 가정해 왔다. 하지만 IOI 회로에서 backup 메커니즘이 탐지되지 않는 false negative, 평가 점수의 변동성·비직관적 순위 뒤틀림 같은 실패가 반복 보고되며 이 가정의 구조적 한계가 의심돼 왔다.

- **Core Contribution**: 이 논문은 NIE를 causal mediation analysis 관점에서 다시 유도해, NIE가 단지 해당 구성요소를 거치는 효과만이 아니라 interaction effects(INT)까지 함께 포함한다는 점을 이론적으로 정식화한다. 또한 activation patching이 측정하는 양이 NIE=PIE+INT임을 보여주며, 여기서 PIE는 “우회(bypass) 경로를 차단했을 때”만 남는 pure indirect effect다. INT는 구성요소의 인과효과가 다른 구성요소들의 상태에 얼마나 의존하는지를 수치화하며, 해석 연구가 이 의존성을 보이지 않게 만드는 대신 측정값 안으로 흡수된다고 주장한다.

- **Technical Challenges**: 핵심 기술적 과제는 transformer의 잔차 스트림이 모든 매개변수에 bypass 경로를 구조적으로 제공한다는 점에서, NIE가 component-specific 효과를 분리하지 못하는 메커니즘을 어떻게 증명·분해하느냐였다. 논문은 denoising과 noising이 서로 대칭이 아니며 각각 다른 인과량(PIE vs NIE)을 추정함을 정리하고, INT가 (1) clean과 patched 활성의 거리, (2) downstream map의 로컬 affine 성질, (3) 매개된 비선형성에 의해 크기가 결정된다는 조건을 제시한다. 더 나아가 다중 구성요소 패칭에서도 INT가 개별·쌍·그룹 단위 상호작용으로 조합론적으로 분해되며, per-head 점수만으로는 집합적 회로 효과를 예측할 수 없음을 보인다.

- **Empirical Impact**: GPT-2 IOI를 중심으로 한 실험에서는 NIE와 PIE 기반 구성요소 순위가 상당히 달라지고(예: rank correlation이 낮아짐), 특히 backup 메커니즘이 NIE 순위에서 체계적으로 과소평가됨을 보인다. 반대로 INT를 제거해 PIE로만 랭킹하면 context-specific 메커니즘(이전 토큰·중복 토큰·유도 등)이 필요한 상황에서 완전히 다른 유형의 누락이 발생한다. 또한 faithfulness 점수의 prompt 분포에 따른 불안정성은 INT 항의 분산이 설명하며, INT는 “지워야 할 잡음”이 아니라 해석 결론이 prompt에 의존하는지, 그리고 combinatorial search로만 드러나는 메커니즘이 있는지 진단하는 지표로 작동한다고 제안한다.



### Aloe-Vision: Robust Vision-Language Models for Healthcar (https://arxiv.org/abs/2606.27500)
Comments:
          MIDL 2026

- **Prior Approaches**: 기존 의료 LVLM 연구는 공개되지 않은 대규모 데이터나 부분 공개 모델에 의존하는 경우가 많아, 데이터·학습 레시피의 투명성이 제한된다. 또한 PubMed 기반 VQA 합성이나 대형 벤치마크 병합이 늘었지만, 공개된 지 오래된 평가셋은 오염(contamination) 가능성이 있어 성능이 과대평가될 수 있다. 마지막으로 안전이 중요한 임상 맥락에서 모델이 adversarial, ambiguous, misleading 입력에 취약하다는 점도 반복적으로 지적돼 왔다.

- **Core Contribution**: 이 논문은 의료와 일반 도메인을 모두 아우르는 준비 완료(ready-to-train) 형태의 오픈 데이터 혼합물 Aloe-Vision-Data를 제안하고, 이를 직접 fine-tuning에 쓰도록 설계했다. 이를 바탕으로 7B와 72B 규모의 Aloe-Vision 모델 패밀리를 완전 공개(weights, 학습 레시피, 데이터)로 배포해 재현 가능성을 높였다. 아울러 새 비전 벤치마크 CareQA-Vision을 통해 오염 위험이 낮은 의료 추론 평가를 제공한다.

- **Technical Challenges**: 핵심 난제는 (1) 고품질 의료 멀티모달 데이터가 부족하고 (2) 오답/저품질 라벨이 섞인 대규모 데이터에서 신뢰도 있는 학습 신호를 확보하며 (3) 긴 샘플이 학습을 지배하지 않도록 균형을 맞추는 것이다. 저자들은 손실 토큰 기반 loss-contributing token 가중치로 도메인·모달리티 비율을 맞추고, LVLM tagging 점수와 answer perplexity로 품질 필터링을 수행했으며, pHash 기반 perceptual hashing으로 훈련-평가 중복을 제거했다. 또한 오픈엔드 평가의 판정 신뢰성을 위해 LLM-judge와 인간 전문가의 일치도를 함께 검증했다.

- **Empirical Impact**: 벤치마킹 결과 Aloe-Vision 계열은 품질 중심 학습 혼합물에서 균형 잡힌 성능을 보이며, 기준선 대비 유의미한 향상을 보이면서 일반 능력 저하가 크지 않다고 보고한다. 특히 CareQA-Vision에서 좋은 성능을 보여, 학습 분포 밖의 임상 케이스 일반화에 강점을 시사한다. 반면 adversarial 분석에서는 표준 벤치마크 성능이 곧 신뢰성으로 이어지지 않음을 확인했으며, Aloe-Vision-AR의 추가 강건성 학습이 misleading 입력에 대한 취약성을 줄이는 데 효과적임을 실증했다.



### DMV-Bench: Diagnosing Long-Horizon Multimodal Agents' Visual Memory with Incidental Cue Injection (https://arxiv.org/abs/2606.27499)
Comments:
          16 pages

- **Prior Approaches**: 기존 에이전트 메모리 연구와 벤치마크는 주로 텍스트 기반을 중심으로 설계돼, 시각 정보를 “기억해야만” 풀리는지 엄밀히 분리하기가 어려웠습니다. VisualWebArena·WebArena·장기 비디오 QA 등은 캡션/alt-text 등 텍스트 단서를 함께 제공해, 시각 기억이 필요해도 텍스트 메모리가 사실상 우회할 여지가 있었습니다.

- **Core Contribution**: 이 논문은 멀티모달 에이전트의 시각 메모리를 상호작용·다중 세션 환경에서 평가하도록 DMV-Bench를 제시합니다. DMV-Bench는 1,000개 가구 상품 변형에 “incidental cue”를 화소에만 심고(L2-leakage contract), 세션마다 대화 맥락을 지운 뒤 특정 큐가 있는 상품을 찾아가게 하여 recall을 “reach(몇 번의 세션 경계를 넘어 살아남았는지)” 곡선으로 측정합니다. 또한 DualMem(dual-coding 기반)을 통해 시각 코드와 언어 코드를 병렬로 저장·검색하도록 설계합니다.

- **Technical Challenges**: 핵심 기술적 난제는 시각 단서가 텍스트로 새지 않게 설계하면서도, 실제로 “언제 기억이 필요한지”를 세션 경계 단위로 측정하는 프로토콜을 만드는 것입니다. 논문은 큐를 이미지에만 존재하게 하고 텍스트 표면에서 사전 감사로 제거하는 L2-leakage contract, 그리고 공통 관찰 스트림을 재활용하는 shared-prefix rollout tree로 계산 비용을 줄이면서 reach별 회상 정확도를 효율적으로 산출합니다. DualMem은 SigLIP-2(시각)와 SBERT(언어 캡션)의 두 임베딩을 저장하고, 검색 시 두 채널의 유사도를 정규화·가중 결합한 뒤 후보의 이미지+캡션을 VLM에 함께 주입하는 방식으로 구현했습니다.

- **Empirical Impact**: DualMem은 Gemini 2.5 Flash와 Qwen2.5-VL-7B 모두에서 체인 길이 J∈{5,10,15,50} 전 구간에서 캡션 베이스라인과 최신 멀티모달 메모리 시스템 3종을 능가합니다. 메모리 뱅크 크기나 인코딩 위치 편향 같은 대조 조건에서도 리드가 유지됐고, ablation 결과 시각 채널이 큐를 end-to-end로 “운반”하며 언어 채널은 쿼리 그라운딩에 상대적으로 작은 비중으로 기여하는 비대칭 dual-coding 양상이 나타났습니다. 연구진은 이 결과가 장기적인 지각 연속성(perceptual continuity) 관점에서 시각 메모리를 설계 목표로 다뤄야 한다는 메시지를 강화한다고 강조합니다.



### Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving (https://arxiv.org/abs/2606.27457)
- **Prior Approaches**: 기존 LLM 모델 라우팅/캐스케이드는 대체로 라우터를 학습하거나, human preference 같은 추가 데이터가 필요해 운영 비용이 커지기 쉽다. 또한 cascaded 방식은 순차적으로 더 큰 모델로 “에스컬레이션”하지만, 사전(프리) 생성 단계에서 TPOT(출력 토큰당 시간) 예산을 명시적으로 맞추는 설계와 정확도 복구가 충분히 결합되지 못했다.

- **Core Contribution**: 이 논문은 2단계 캐스케이드로 생산 환경의 정확도-비용 트레이드오프를 해결한다. Stage 1은 쿼리를 semantic clustering으로 묶어 각 클러스터를 “가성비 좋은 모델”에 배정하고, Stage 2 QE(Quality Estimation)는 Stage 1의 저품질 출력만 감지해 더 강한 모델로 재라우팅한다. 라벨은 task-correctness(정답 여부)만 사용하며, 모델 풀 변경에도 수동 재구성이 필요 없도록 설계했다.

- **Technical Challenges**: 핵심 난제는 “명시적 TPOT 예산 하에서 라우팅을 결정”하면서 “참조 신호 없이 출력 품질을 추정”하는 것이다. 해결책으로 Stage 1은 클러스터-모델별 오류율과 TPOT을 함께 고려하는 비용-정확도 점수에 해석 가능한 하이퍼파라미터 λ를 두고, offline에서 TPOT 예산 B를 만족하는 λ*를 튜닝해 테스트에도 그대로 적용한다. Stage 2는 ModernBERT-base 기반의 경량 이진 분류기(accept/escalate)를 task-correctness 라벨로 학습해, 효율 모델 출력에 대해서만 품질이 낮을 때만 에스컬레이션하도록 했다.

- **Empirical Impact**: TeleQnA(통신 QA)와 AIME 2024(수학 추론)에서 다중 모델 풀(Qwen/Gemma 계열)을 대상으로 평가했으며 도메인 전반 일반화도 확인했다. 두 단계 캐스케이드는 가장 강한 모델 정확도의 97-99%를 유지하면서 TPOT을 줄여, AIME 2024에서는 TPOT 18% 감소로 거의 동일 정확도를 달성했다. 또한 Stage 2의 에스컬레이션으로 Stage 1에서 잃은 정확도를 대부분 복구하면서도, 항상 강한 모델을 쓰는 경우 대비 더 빠른 운영 지점을 보여 실용적 의미가 크다.



### Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placemen (https://arxiv.org/abs/2606.27409)
Comments:
          20 pages, 5 figures, 1 table. Code and data: this https URL

- **Prior Approaches**: 기존 멀티에이전트 LLM 연구는 verifier/critic 에이전트로 환각을 줄이려 하지만, 검증 피드백이 상호작용 지연(latency) 때문에 늦게 들어온다는 점을 안정성 관점에서 정량화하지 못했습니다. 또한 신뢰도·증거 기반 집계나 debate 설계는 오류 전파(정적)를 다루는 경우가 많고, 지연이 만드는 동적 불안정(진동/발산) 메커니즘은 분석되지 않았습니다.

- **Core Contribution**: 이 논문은 검증이 지연된 상태에서 에이전트 네트워크가 ‘지연된 consensus’로 수렴하는 과정을 그래프 모델로 정식화하고, corrector(진실을 고정하는 노드)가 있는 경우의 안정성 조건을 도출합니다. 특히 검증의 강도(dose)와 지연(delay) 사이에 닫힌형(closed-form) 안정성 임계값을 제시하며, correction이 너무 강하거나 너무 늦으면 consensus가 진동으로 바뀔 수 있음을 보여줍니다. 또한 제한된 corrector budget 하에서 영향력 있는 노드를 고르는 placement 목적함수(초모듈러)와 greedy의 (1-1/e) 근사 보장까지 연결합니다.

- **Technical Challenges**: 핵심 기술적 난제는 고차원 지연 시스템의 안정성을 그래프 스펙트럼과 함께 해석하는 것이며, 이를 위해 grounded Laplacian에 의한 스펙트럴 분해로 지연 방정식을 독립적인 스칼라 지연 재귀로 분해합니다. 그 결과 각 모드가 단위원(unit disk)을 이탈하는 조건을 추적해 ‘검증 dose 한계’의 정확한 경계를 만들었고, 두 지연(커뮤니케이션 지연과 검증 지연)이 동시에 작동할 때 최악 구간이 동시 지연 코너임을 밝혀 임계값이 (지연 2에서) inverse golden ratio가 됨을 산출합니다. 마지막으로 placement 최적화에서는 resolvent의 coherence가 초모듈러임을 이용해 greedy가 네트워크의 amplifier/bridge 노드에 예산을 집중하도록 설계합니다.

- **Empirical Impact**: 5개의 오픈 모델에서 예측된 dose-delay oscillation(검증 강도·지연이 임계값을 넘을 때 진동 전환)이 실제 수치 실험으로 재현됩니다. 반대로 grounded factual answering(진실을 흡수 경계로 만드는 설정)에서는 같은 지연이 수렴을 깨지 못해, 불안정성은 signed-belief 과제(부호가 있는 믿음/오류 부피드백)에서만 나타나는 특이성을 갖는다고 주장합니다. 이는 단순히 verifier를 ‘추가’하는 수준을 넘어, 검증 정책(강도·지연·배치)을 안정성-성능 최적화 대상으로 다루게 만드는 실증적 근거를 제공합니다.



### Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieva (https://arxiv.org/abs/2606.27401)
Comments:
          15 pages, 4 figures

- **Prior Approaches**: 기존 코드 검색은 BM25, TF-IDF, Jaccard 같은 전통적 IR로 빠르게 후보를 좁히지만, 표면 키워드 매칭에 머물러 정확도가 제한된다. 딥러닝 기반 two-stage recall-then-rerank(TOSS)는 임베딩으로 먼저 recall을 수행한 뒤 cross-encoder로 rerank하지만, 1단계에서 놓친 정답은 뒤에서 복구되지 않는다. 그럼에도 1단계 recall에 대한 체계적 대규모 비교와 효율·확장성(throughput) 분석은 부족했다.

- **Core Contribution**: 논문은 5개 언어(파이썬, 자바, 자바스크립트, C++, C#)와 4개 데이터셋을 대상으로 17개 코드 임베딩/검색 모델의 1단계 recall 성능과 효율을 대규모로 벤치마킹한다. 또한 LLM 기반 code normalisation 및 query/candidate rewriting 전략을 제시해, 성능이 낮고 스타일 민감한 모델에서 precision을 최대 29%까지 끌어올릴 수 있음을 보여준다. 결과적으로 데이터셋 전반에서 코드 특화 LLM 임베딩의 견고성에 대한 가정과 자원 제약 환경의 지속가능성에 의문을 제기한다.

- **Technical Challenges**: 핵심 기술 난제는 1단계 recall의 성능을 유지하면서 terabyte급 코드베이스로 확장할 때 생기는 대규모 계산·인프라 병목이다. 저자들은 정확도(Precision@k, NDCG)를 순수 임베딩 품질 기준으로 분리하기 위해 ANN 없이 exact scanning을 수행하고, Cosine similarity를 일관되게 사용해 색다른 indexing 손실을 상대오차로 관찰한다. 또한 변수/주석/문체 변화가 임베딩에 미치는 영향을 Qwen2.5-Coder-7B-Instruct 기반 변환(주석 정리, 식별자 재명명, LLM 재작성 등)으로 실험하며, 특히 query와 candidate에 함께 적용할 때 이득이 커짐을 분석한다.

- **Empirical Impact**: 실험 결과는 ‘최고 정확도’와 ‘확장 가능한 throughput’ 사이에 강한 trade-off가 있음을 명확히 보여준다: Qwen3 Embedding 등 상위권은 많은 데이터셋에서 Precision@50이 높지만, 큰 비용 때문에 인덱싱·운영이 비현실적일 수 있다. 반대로 StarEncoder 같은 경량 모델은 throughput은 크게 개선(대략 수십 배 수준 차이)하지만, xCodeEval/CodeNet처럼 어려운 설정에서 precision이 크게 하락하며 품질이 급격히 떨어진다. 결론적으로 resource-constrained 환경에서는 경량 인코더 + LLM reranking의 hybrid two-stage 구조가 여전히 가장 효과적이며, 모델 선택은 작업·언어·데이터셋 맥락에 맞춰야 한다는 실무적 가이드라인을 제공한다.



### CalBrief: A Pilot Diagnostic Benchmark for Evidence-Calibrated Scientific Briefing with Large Language Models (https://arxiv.org/abs/2606.27383)
- **Prior Approaches**: 기존 과학 문서 요약·리뷰 연구는 핵심 내용의 커버리지와 근거 인용에 초점을 두는 경우가 많지만, 새로 생성되는 결론이 실제 증거의 강도와 범위를 얼마나 정확히 반영하는지는 별도로 다루기 어려웠습니다. 또 claim verification이나 grounded generation은 특정 고정 문장의 지지/반증 또는 출처 근거를 보강하지만, 패키지 전체가 뒷받침하는 결론의 ‘강도·한계·누락 증거’까지 캘리브레이션하는 문제와는 결이 다릅니다.

- **Core Contribution**: 이 논문은 evidence-calibrated scientific briefing을 패키지 수준 과학 문서 이해 과제로 정식화하고, 결론별 evidence strength(강도)와 scope boundary(범위 한계), missing-evidence caveat(누락 증거 주의)를 함께 생성하도록 요구합니다. 또한 16개 이질적 과학 evidence package와 96개의 human-verified package-level takeaways로 구성된 verified pilot benchmark를 제공해, 중간 신호까지 포함한 평가를 가능하게 합니다.

- **Technical Challenges**: 핵심 기술 도전은 ‘감사 가능한(auditable) 증거 구조화’와 ‘강도 캘리브레이션’을 동시에 달성하면서, 결론이 증거 패키지의 범위를 벗어나 과장되지 않게 만드는 것입니다. 연구진은 CalBrief라는 role/gap/strength 프레임워크를 진단 도구로 사용해 실패 지점을 분해했고, 4-way 강도 라벨 공간({moderate, weak, uncertain, insufficient_evidence})이 모델을 과도하게 신중하게 만들며(대부분 원인), gap/scope 신호 주입 자체는 영향이 거의 없고 나머지는 파이프라인 정책 결합 비용에서 온다고 분리해 설명합니다.

- **Empirical Impact**: 실험에서 structured organization은 role 식별과 evidence-gap 관련성 등 ‘조직화 능력’은 개선하지만, strength calibration은 항상-moderate 기준선에 크게 못 미쳐 과도한 보수성 문제가 확인됩니다. 진단 결과, 닫힌 모델 3종(GPT-4o/Claude Sonnet/Gemini Flash)에서 라벨 공간 확장이 보수성 격차의 약 63%를 설명했고 신호 주입은 거의 기여하지 않았으며(비유의), 4-way 예측을 사후에 binary로 collapse하면 direct binary prompting과 동등하거나 때로는 더 나아져 라벨링 설계의 함의를 시사합니다.



### DiARC: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models (https://arxiv.org/abs/2606.26530)
- **Prior Approaches**: 기존 ARC(Abstraction and Reasoning Corpus) 접근은 정답 input-output만을 늘리는 데이터 증강·합성 중심이 많았습니다. 또한 일부는 재귀 추론(후속 상태 반복 개선), 비전 모델로의 전환, LLM용 프롬프트/표현 재설계로 성능을 끌어올리지만, 공통적으로는 그럴듯한 오답을 구분하는 신호가 약했습니다.

- **Core Contribution**: 이 논문은 ARC-like 문제 해결이 “정답을 맞히는 것”에 더해 “그럴듯하지만 규칙이 다른 near-miss를 식별하고 거절”할 수 있어야 한다고 주장합니다. 이를 위해 preference alignment 관점에서 정답/오답 선호쌍을 만드는 DiARC를 제안하고, DPO로 두 출력 간 상대 선호를 학습시킵니다.

- **Technical Challenges**: 핵심 난제는 정보가 충분한 negative sample(오답 후보)을 만들되, 관측된 support 시연은 유지하면서도 모델이 구분할 수 있게 ‘가까운 실수’를 설계하는 것입니다. DiARC는 (1) 출력 격자 공간 시각 변환, (2) DSL 수준 rule inversion, (3) 작업별 transformation rule editing의 세 단계로 near-miss를 생성한 뒤, DPO 학습으로 정답 출력의 상대 likelihood를 높이는 방식으로 이를 해결합니다.

- **Empirical Impact**: 여섯 개 ARC 벤치마크에서 DiARC는 3종 오픈소스 LLM 전반에 걸쳐 기준선 SFT보다 일관된 성능 향상을 보였습니다. 특히 Qwen3-4B는 ARC-AGI-1, MiniARC, ConceptARC에서 96%대 정확도를 달성하며 closed-source 및 기존 오픈소스 방법을 능가했고, 생성(generation)과 선택(discrimination) 두 단계에서 이득이 함께 발생함을 분석으로 확인했습니다.



New uploads on arXiv(cs.IR)

### Fast and Feasible: Permutation-based Constrained Reranking for Revenue Maximization (https://arxiv.org/abs/2606.28059)
- **Prior Approaches**: 기존 검색/추천 시스템은 후보 생성 후 reranking으로 성능을 개선하지만, e-commerce에서는 지면(노출) 순서를 바꿔 매출을 끌어올리면 relevance 저하나 fraud 위험 증가 같은 부작용이 생긴다. 이를 줄이기 위해 다목적 최적화(가중치 scalarization, Pareto)나 제약조건 기반 최적화가 연구돼 왔으나, 가중치 민감도와 다해 해법(여러 Pare토 해) 문제가 남는다. 제약 기반 접근을 ILP로 세우면 정확한 해를 구할 수 있지만, 온라인 실시간 배치에는 브랜치 앤 바운드 기반 솔버가 지나치게 느리고 다중 제약 확장도 효율이 떨어진다.

- **Core Contribution**: 이 논문은 reranking을 “매출을 최대로 하되(기본 목적), 다른 지표(관련성·품질·사기 등)를 쿼리별 제약으로 보장”하는 ILP 문제로 명확히 정식화한다. 특히 제약을 이상적(ideal) 랭킹이 아니라 현재 production 결과 대비 허용 가능한 열화 수준으로 정의해, 현업 랭킹의 실행 가능성(feasibility)을 유지한다. 또한 ILP에 근접한 해를 빠르게 만들기 위해 permutation 기반 휴리스틱 PermR을 제안한다.

- **Technical Challenges**: 핵심 기술 난제는 실시간 지연(latency) 한계 내에서 다중 제약을 동시에 만족하면서 매출 목적함수의 개선을 달성하는 것이다. PermR은 매 쿼리에서 top-N 결과의 이웃한 두 아이템을 swap(전치)하며, 먼저 모든 제약이 만족되는 경우에는 기본 목적 매출 F0를 더 키우는 방향으로 시도한다. 만약 어떤 제약이 위반되면 위반된 제약을 우선적으로 “복구(repair)”하도록 확률적으로 이웃 swap을 선택하고, prefix 제약 때문에 개선 swap이 불가능한 경우에는 해당 제약/목적을 가장 크게 만드는 아이템을 앞쪽부터 재배치한다.

- **Empirical Impact**: 대규모 classifieds 플랫폼에서 오프라인 실험 결과, PermR은 ILP가 만든 매출 uplift의 약 63%(약 6.9% uplift 중 2.6% 수준)를 0.05초 내 지연 제한을 지키며 달성했고, 제약은 모두 유지했다. 입력 SERP 길이가 늘어도 PermR의 시간은 완만하게 증가해 실서비스 적용 가능성을 보였다. 14일 온라인 A/B 테스트(약 5,600만 쿼리)에서는 평균 매출이 2% 증가했으며, 카테고리별로도 대부분 구간에서 제약을 보존한 채 개선이 확인됐다.



### Listwise Explanation of Embedding-Based Rankings via Semantic Chunk Grouping (https://arxiv.org/abs/2606.27980)
Comments:
          17 pages, 5 figures, 4 tables

- **Prior Approaches**: 기존 listwise 설명은 SHAP 기반이라도 설명 단위를 대체로 단어(예: stemmed words) 또는 사전에 정의된 어휘/특징으로 잡아, dense embedding ranker의 문장·구간 수준 증거와 단위가 어긋나는 문제가 있었다. 그 결과 단어는 여러 문서에 재사용되지만 조각난 상태로 perturbation이 적용돼, 의미 기반 랭킹의 원인을 충분히 정밀하게 환원하기 어려웠다. 또한 raw chunk를 그대로 쓰면 문서별로 특징이 달라져 listwise의 “공유 특징 perturb” 성질이 약해지고, 특징 공간이 쿼리·문서 길이에 따라 크게 변해 실사용이 부담스러웠다.

- **Core Contribution**: 이 논문은 ChunkGroupSHAP이라는 listwise Shapley 설명 방법을 제안하며, 의미적으로 유사한 chunk를 임베딩으로 묶어 “교차 문서 공유 semantic feature”로 재정의한다. 이를 통해 동일한 그룹 마스킹이 관련 증거를 담은 여러 문서를 함께 흔들어, dense ranker의 표현 단위와 더 맞는 설명 해상도를 만든다. 또한 corpus-level 그룹(재사용 우선)과 query-local 그룹(현 순위 적합도 우선) 두 변형을 함께 다뤄, 그룹 범위 선택의 중요성을 체계화했다.

- **Technical Challenges**: 핵심 기술 과제는 listwise perturbation을 유지하면서도 설명 단위를 dense ranker의 문맥적 표현과 정합적으로 만들고, 특징 공간 크기는 고정해 샘플링·계산을 안정화하는 것이었다. 논문은 chunk를 sliding window로 분할한 뒤 k-means로 chunk 임베딩을 k개 그룹으로 클러스터링하고, 활성 그룹만 남겨 재구성한 문서를 원래 ranker로 다시 스코어링해 NDCG@K(또는 Kendall’s tau) 기반 coalition value를 정의한다. 이어 KernelSHAP으로 그룹 단위 attributions을 근사 계산하고, 문서별로 해당 그룹들의 기여를 합산해 “설명 유도 랭킹”과 fidelity를 평가한다.

- **Empirical Impact**: MS MARCO, FinanceBench, AILACaseDocs, FinQA에서 BM25와 E5 계열 dense ranker를 비교한 결과, 최적 설명 단위는 설정에 따라 달라졌다. BM25에는 단어 기반이 강했고, dense ranker에서는 대체로 corpus-level chunk group이 word-level 대비 더 높은 fidelity를 보였으며, MS MARCO처럼 heterogeneous web retrieval에서는 query-local grouping이 특히 유리했다. 즉 “설명 단위는 고정값이 아니라, ranker의 representational granularity와 검색 코퍼스의 구조(재사용 가능한 증거가 얼마나 공유되는지)에 맞춰야 한다”는 실증적 가이드라인을 제공한다.



### An LLM-Powered Semantic Alignment Framework for Journal Recommendation (https://arxiv.org/abs/2606.27930)
- **Prior Approaches**: 기존 저널 추천은 TF-IDF나 topic modeling 같은 텍스트 유사도 기반부터, feature engineering 및 supervised learning, 그리고 SBERT·Transformer 임베딩을 쓰는 방식까지 다양했다. 다만 이러한 방법은 정해진 표현과 고정된 점수함수에 의존해 원고 내용과 저널 scope의 복잡한 관계(특히 학제 간)를 충분히 포착하기 어렵고, 해석 가능성도 제한적이다. 또한 대부분이 task-specific 학습이나 과거 상호작용 데이터에 묶여 있어 일반화와 확장성이 떨어질 수 있다.

- **Core Contribution**: 본 논문은 LLM-powered semantic alignment 프레임워크로 저널 추천을 ‘원고-저널 scope의 의미 정합도(semantic matching)’ 문제로 재구성한다. LLM이 제목, 초록, 키워드와 후보 저널의 Aims & Scope(및 선택적으로 참고문헌)를 입력받아 task-specific fine-tuning 없이 적합 저널을 직접 추론한다. 또한 각 추천에 대해 입력 근거에 기반한 구조화된 설명(방법론 정합/응용 적합/독자·기여 유형)을 함께 생성하도록 설계했다.

- **Technical Challenges**: 핵심 과제는 (1) LLM이 후보군 49개 저널 중 정확히 순위를 매기도록 프롬프트를 통제하고, (2) 참고문헌 같은 추가 신호가 정보 노이즈가 되지 않게 하며, (3) 반복 실행 시 출력 변동성과 leakage 위험을 관리하는 데 있다. 이를 위해 논문은 저널 scope 설명과 원고 텍스트를 ‘호환성 평가’ 관점의 조건(방법론 초점/도메인/용어·키워드 기반 청중)으로 명시하고, 출력도 후보 저널 ‘정확한 명칭’으로 Top-K만 반환하며, reasoning 설정에서는 항목별 설명 포맷을 강제한다. 또한 DeepSeek-V3 기반 에이전트 구현에서 레퍼런스 정보는 실험적 비교를 위해 단계적으로 포함해 실사용 시 입력 의존성을 최소화했다.

- **Empirical Impact**: DeepSeek-V3를 2021~2025년 통계·관련 분야 23,609편(49개 저널) 데이터에 적용해 Top-3 40.23%, Top-5 53.67%, Top-10 70.05% 정확도를 보고했다. 참고문헌 정보를 추가하면 대부분의 저널에서 성능이 개선되며, 한편 어떤 저널(AoAS·AoS 등)은 레퍼런스 범위가 넓어지며 오히려 하락해 신호 대 잡음의 균형이 드러난다. 반복 실행 안정성도 Top-5 Jaccard 유사도 평균 84%로 높고, 추천과 함께 생성된 해석(reasoning) 출력이 순위 결과를 설명하는 방식으로 작동해 training-free·확장 가능한 학술 의사결정 지원 가능성을 보여준다.



### From Bootstrapping to Sequence Modeling: A Unified Generative Framework for Personalized Landing-Page Modeling (https://arxiv.org/abs/2606.27865)
Comments:
          arXiv admin note: text overlap with arXiv:2507.23459

- **Prior Approaches**: Kuaishou는 PLPM(Personalized Landing Page Modeling)을 정의하고, RL 기반 KLAN을 Conservative Q-Learning(CQL)으로 제안했지만 Markov 가정에 기대는 한계가 있다. 실제 사용자 행동은 비마르코프성 비정형 의존성이 강해, 상태를 히스토리로 보완해도 손실 압축에 그쳐 장기 의존을 충분히 담기 어렵다. 또한 PLPM은 지연 보상이 자연스러워 TD 부트스트래핑 기반 CQL에서 누적 오차와 credit assignment 문제가 길게 이어진다.

- **Core Contribution**: 본 논문은 GLAN(Generative Landing-page Adaptive Navigator)으로 PLPM을 Decision Transformer 기반의 sequence modeling으로 통합 해결하려 한다. 비마르코프 비구조를 마코프 상태 가정 없이 궤적 시퀀스 자체의 컨텍스트로 흡수하고, 부트스트래핑 대신 Return-to-Go(RTG) 조건부 감독학습으로 지연 보상 학습의 불안정성을 줄인다. 여기에 global(일 단위)·local(세션/페이지 단위) 신호를 분리하는 두 모듈 L-RTG와 HRM을 설계했다.

- **Technical Challenges**: 첫째, DT는 추론 시 미래가 없어 목표 RTG를 어떻게 초기화할지(Initialization dilemma)가 핵심 난제로 남는다. GLAN은 L-RTG에서 요일 주기성(periodicity)과 일간 변동(sequential dynamics)을 함께 모델링하고, app usage time과 session frequency 간 구조적 제약을 둔 라그랑주 제약 최적화(primal-dual)로 안정적인 global 가이드를 만든다. 둘째, 세션 총 사용시간 같은 단일 보상만으로는 landing page 행동의 품질을 구분하기 어려워 credit ambiguity가 발생하며, HRM은 세션 수준 피드백을 페이지별 소비 지속시간과 drop-off 위험 등 더 미세한 신호로 분해해 페이지 할당마다 정교한 local supervision을 제공한다.

- **Empirical Impact**: Kuaishou 온라인 실험에서 GLAN은 DAU에서 +0.158%, 사용자 Lifetime(LT)에서 +0.108%의 개선을 달성하며 제안 방법의 실용적 성능을 입증했다. 특히 장기 지연 효과가 큰 환경에서 DT 계열의 장기 credit assignment 강점과, L-RTG/HRM의 전역·국소 신호 분해가 결합될 때 성과가 뚜렷하게 나타난다. 멀티페이지 플랫폼의 랜딩 내비게이션 문제에서 RL 대신 sequence modeling을 정식 공정으로 끌어올리는 방향성을 제시한 점에서 의미가 크다.



### End-to-End Dynamic Sparsity for Resource-Adaptive LLM Inferenc (https://arxiv.org/abs/2606.27743)
- **Prior Approaches**: 기존 LLM 추론은 정적 계산 그래프를 가정해 모든 요청에 동일한 레이어·헤드를 실행하는 방식이 중심이었다. 그래서 스팟 인스턴스 선점처럼 자원이 갑자기 줄거나, 프리미엄/프리 같은 QoS 티어에 따라 예산이 달라져도 OOM으로 실패하거나 지연이 커지는 등 탄력적으로 “품질을 서서히 낮추는” 대응이 어려웠다. 정적 압축(pruning/distillation)이나 엔트로피 기반 early-exit은 한 번 정해진 구조/휴리스틱에 묶여 다양한 런타임 조건에 맞춰 재구성이 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 Learning to Allocate(L2A)라는 end-to-end 프레임워크로, 입력 난이도뿐 아니라 런타임 리소스 예산 자체를 함께 조건화해 자원-적응형 추론을 학습한다. 레이어 스킵, head pruning, reasoning-to-answer 전환(think/answer 분할)까지 예산을 입력으로 받는 게이팅 네트워크로 통합했으며, 단일 모델이 compute-accuracy Pareto frontier를 하나로 커버하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 스킵/프루닝처럼 비연속 결정을 도입하면 표현이 흔들려 성능이 무너질 수 있고, (2) 예산 변화에 맞춰 실제 연산량(레이어/헤드/FLOPs·지연·reasoning 토큰 길이)이 일관되게 제어되어야 한다는 점이다. 이를 위해 LoRA로 기존 가중치를 보존한 채 게이팅 경로에 맞춰 PEFT 적응을 수행하고, 성능(크로스엔트로피), 논리적 일관성(KL distillation), 예산(예상 compute 비용) 및 reasoning 길이(think 세그먼트 토큰 비용)를 동시에 최적화하는 단일 목적함수를 사용한다. 또한 학습 시 다양한 b를 샘플링해 게이트가 예산 레짐을 “본” 뒤, 추론에서는 학습된 게이트를 하드 임계값으로 전환해 실제 런타임 신호(데드라인·메모리 여유·큐 상황)를 b로 캘리브레이션한다.

- **Empirical Impact**: Llama-3-8B와 Qwen-3-4B에서 GSM8K 기준 dense baseline 대비 정확도 격차를 0.6% 이내로 유지하면서, 최대 34% 수준의 realized layer sparsity를 달성했다고 보고한다. 또한 OOD인 HumanEval/BBH에서도 제로샷 성능이 동일한 수준의 격차를 유지하며, 정적/휴리스틱 베이스라인은 동등한 추론 시간에서 5–10% 성능 하락하거나 예산별로 별도 튜닝이 필요했다. 결과적으로 L2A는 스팟 선점·다중 테넌트·QoS 차등 같은 변동 환경에서 “예산에 맞춰 동적으로 추론을 절약하되 논리와 품질을 지키는” 접근으로 분야에 실질적 배치 가능성을 제시했다.



### Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation (https://arxiv.org/abs/2606.27732)
- **Prior Approaches**: 기존 discrete diffusion language model(dLLM)은 속도를 위해 병렬 복원을 시도하지만, 모델 구조에서 양방향 attention을 쓰면 KV caching이 깨져 매 denoising step마다 전면 재계산이 필요해진다. 반대로 causal attention은 KV caching은 가능하지만 right-side 컨텍스트를 잃어 품질 저하가 발생하며, 이를 완화하려던 block/hybrid 방식은 우회적으로 right 컨텍스트 범위를 제한하거나 구조가 복잡해지는 한계가 있었다.

- **Core Contribution**: 이 논문은 ‘비대칭 bidirectional context(비대칭 양방향 컨텍스트)’라는 새 패러다임을 제안해, right 컨텍스트를 attention이 아닌 별도 경로로 제공하면서도 causal의 prefix KV caching을 유지한다. 이를 R2LM(Right-to-Left Mamba)로 구현했으며, left 컨텍스트는 기존 causal backbone의 attention으로, right 컨텍스트는 reverse Mamba SSM sidecar가 압축 신호 형태로 보강하도록 설계했다.

- **Technical Challenges**: 핵심 기술적 난제는 right-side 정보를 주되 KV cache를 무너뜨리는 bidirectional attention의 구조적 의존성을 회피하는 것이다. 저자들은 backward/forward의 역할을 비대칭화하기 위해 reverse Mamba를 hooked layer에 잔차로 삽입하고, 초기에는 gate를 zero-initialized해 모델이 초기 시점에서 causal 모델과 bit-identical이 되게 만들며, 추론에서는 prefill 캐시를 그대로 재사용한 채 generation 구간에만 reverse SSM을 수행하도록 구성했다.

- **Empirical Impact**: Qwen3-1.7B를 60B 토큰으로 continued pretraining한 실험에서 R2LM은 bidirectional dLLM 대비 batch serving에서 2.4×~12.9× 높은 throughput을 보이며, autoregressive(AR) 대비로는 1.9×~2.9× 속도 개선을 보고했다. 동시에 대부분의 벤치마크에서 causal baseline을 상회하고 평균적으로도 bidirectional dLLM보다 더 좋은 결과를 보여, ‘품질-효율’ 동시 최적화 가능성을 입증했다.



### Intuition-Guided Latent Reasoning for LLM-Based Recommendation (https://arxiv.org/abs/2606.27684)
- **Prior Approaches**: 기존 LLM 기반 추천은 fine-tuning으로 항목을 직접 생성하거나, think-before-recommend 계열로 추론을 활용해 성능을 높이는 흐름이 강했다. 특히 latent reasoning은 중간 과정을 토큰으로 해석하지 않고 연속 잠재공간에서 자동회귀적으로 추론해 효율을 높이지만, 추론 시작 지점에 대한 제약이 약해 선호 탐색이 빗나갈 수 있었다. 결과적으로 목표 아이템 임베딩과의 정렬이 어긋나 suboptimal 영역으로 이동할 위험이 반복적으로 지적된다.

- **Core Contribution**: IntuRec은 인간의 다단계 추론이 ‘무제약 숙고’가 아니라 intuition이라는 잠재 우선지식으로 시작된다는 인지과학 관점을 추천에 적용한다. 이를 위해 추천 intuition을 두 단계로 명시적으로 추출·주입해 latent reasoning의 시작점을 preference-aligned 표현으로 고정한다. 구체적으로 LLM 추천기가 사용자 히스토리로부터 top-K 후보를 만든 뒤, self-/cross-attention으로 후보 집합을 하나의 초기 intuition embedding으로 변환해 추론 경로를 유도한다.

- **Technical Challenges**: 핵심 난제는 (1) 후보 집합이 너무 우연히(혹은 정답 아이템이 과도하게) 포함되면 shortcut learning을 유발하고, 반대로 너무 적으면 학습 신호가 약해진다는 점이다. IntuRec은 Target-Aware Candidate Balancing(TACB)로 학습 중 정답 포함 비율을 검증 분포와 기대값 기준으로 맞춰 분포 불일치를 줄인다. 또한 후보로부터 추론 시작점을 만들 때 ‘구조적으로 그럴듯하지만 의미가 어긋난’ embedding을 방지하기 위해 intuition alignment를 대비학습(BPR 스타일)으로 추가 감독한다.

- **Empirical Impact**: 여러 실사용 데이터셋(아마존 리뷰 하위셋)에서 IntuRec은 다양한 baseline 및 백본에 대해 일관되게 향상되며, 특히 IntuRec-D가 SOTA 성능을 보였다고 보고한다. 실험은 component별 기여를 확인하는 ablation으로도 설계 의도를 검증하며, intuition을 기반으로 한 시작점 정렬이 latent reasoning의 탐색 효율을 높인다는 주장을 뒷받침한다. 전반적으로 ‘추론 과정 자체의 제약 부재’ 문제를 시작점 정렬로 해결했다는 점에서 LLM 추천과 latent reasoning 연구에 실용적 방향성을 제공한다.



### A Sensitivity-Aware Test Collection for Search Among Personal Information (https://arxiv.org/abs/2606.27559)
Comments:
          SIGIR 2026 Resource Paper

- **Prior Approaches**: 기존 Sensitivity-aware search(SAS)는 민감정보를 포함할 수 있는 컬렉션에서 관련 문서는 잘 찾되, 민감 문서는 노출을 줄이는 것을 목표로 한다. 다만 이를 평가할 시험 컬렉션은 대개 쿼리·relevance qrels·민감도 라벨을 모두 갖춰야 해서, TREC처럼 유형이 제한되거나(legal privilege 등) Avocado처럼 비용·라이선스 제약이 따르는 문제가 있었다.

- **Core Contribution**: 이 논문은 Enron 이메일에 대한 Sensitivity-Aware Relevance Assessments(SARA) 컬렉션을 공개하며, 민감/비민감 문서 라벨과 150개 정보요구(쿼리) 및 관련성 판단(qrels)을 함께 제공한다. Hearst Enron 부분은 crowd로 relevance를 모으고, 나머지는 LLM 기반 평가로 확장해 대규모로 재현 가능한 SAS 벤치마크를 만든 점이 핵심 기여다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 현실적인 정보요구를 생성하고, (2) 방대한 문서에 대해 사람 라벨 비용 없이 일관된 relevance를 확보하며, (3) LLM 평가가 랭커-평가의 역할 혼동 등 평가 함정을 만들지 않도록 설계하는 것이다. 저자들은 LDA 토픽모델링으로 50개 토픽에서 쿼리 150개를 구성하고, crowd pooling으로 11,471개의 human relevance를 구축한 뒤, lexical/learned sparse/dense/cross-encoder 기반 풀링과 UMBRELA 계열 프롬프트 및 few-shot(인간 라벨 예시)로 LLM judged qrels를 확장했으며, 민감도 라벨은 Hearst subset으로 학습한 T5 기반 분류기를 129,821문서 전체에 적용했다.

- **Empirical Impact**: 제공된 SARA 컬렉션으로 관련성 성능, 민감도 분류 성능, 그리고 SAS 성능(검색 후 필터링 기반) 베이스라인을 제시해 연구자가 바로 비교/실험할 수 있는 출발점을 마련했다. 또한 ir_datasets와 Huggingface 인덱스(PyTerrier Artifacts)를 통해 sparse·dense 모두 손쉽게 재현 가능하게 배포되어, 민감정보 포함 문서 검색 연구의 접근성을 크게 높인다는 의미가 있다.



### Context-Aware Explanations for Spatialized Document Layouts (https://arxiv.org/abs/2606.28081)
Comments:
          10 pages, 4 figures, accepted to Graphics Interface 2026 (GI 2026)

- **Prior Approaches**: 기존 공간화 문서 레이아웃 연구는 2D 배치에서 클러스터·이상치·브리징 문서 같은 구조를 보이게 해 탐색을 돕지만, 그 “공간적 관계를 어떻게 해석해야 하는지”는 충분히 설명하지 못했다. 콘텐츠 중심 요약(키워드/토픽 라벨)이나 투영 생성 과정 설명(임베딩·차원축소의 Explainable AI)은 문서 간 의미는 다루더라도 근접·분리·중간 위치 같은 레이아웃 내 관계를 근거로 삼는 설명은 약하다. 또한 LLM 기반 자연어 요약은 텍스트에 주로 기반해 공간적 관계를 직접 반영하지 못해, 사용자가 로컬과 글로벌 맥락을 수작업으로 통합해야 했다.

- **Core Contribution**: 이 논문은 CAPE(Context-Aware Explanations)를 제안하며, 레이아웃 자체를 설명의 대상으로 삼아 문서 의미와 레이아웃에서 도출한 공간 문맥을 함께 근거로 하는 설명을 생성한다. CAPE는 클러스터, 서브그룹, 아웃라이어, 브리징 문서 같은 “살리는 공간 패턴”을 찾아, 각 패턴이 주변과 어떤 관계를 갖는지까지 자연어로 풀어준다. 더불어 AI가 먼저 요약을 제시하는 overview 모드와 사용자가 선택한 영역/문서에 대한 온디맨드 탐색을 모두 지원하며, 출력은 short·intermediate·long의 다단계 상세도로 제공된다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 레이아웃의 공간 관계를 설명 가능한 형태로 추출하고, (2) 이를 LLM 프롬프트에 잡음 없이 효율적으로 주입하며, (3) 패턴 유형과 사용자 의도에 맞춰 설명을 달리 생성하는 것이다. CAPE는 먼저 모듈형으로 공간 패턴을 식별한 뒤, 각 설명 타깃에 대해 전역(global)·국소(local)·패턴 특화(pattern-specific) 다층 컨텍스트를 구성하고 문서 요약과 주변 관계(근접, 분리, 중간 위치 등)를 컴팩트 표현으로 통합한다. 그 다음 GPT-4o로 패턴 유형(클러스터/서브그룹/아웃라이어/브리징)과 쿼리(문서 단일 선택 또는 영역 비교)에 조건화된 프롬프트를 설계해 다단계 상세도의 설명을 생성한다.

- **Empirical Impact**: 뉴스 20 Newsgroups 레이아웃(216문서)과 학술 논문 레이아웃(IEEE VIS 2022/2023, 252편)에서 CAPE는 전역 구조 파악, 영역 내 변이 관찰, 인접 영역 비교, 경계/브리징 사례 해석을 지원하는 시나리오를 보여준다. 특히 서브그룹 단위로 지역적 관점을 분리해 보여주고, 서로 가까이 배치된 주제라도 초점이 다르면 이를 구분하는 설명을 제공함으로써 사용자의 해석 부담을 줄인다. 통제된 사용자 연구(10명, within-subjects)에서는 공간 문맥을 포함한 CAPE 설명이 콘텐츠만 제공한 LLM 및 키워드 기반 대비 더 “도움이 된다”는 인식을 얻어, 공간적으로 근거가 잡힌 설명이 레이아웃 해석에 실질적 향상을 준다는 점을 실증적으로 뒷받침했다.



### Single and Multi Truth Data Fusion using Large Language Models (https://arxiv.org/abs/2606.28062)
- **Prior Approaches**: 데이터 퓨전(또는 truth discovery)은 여러 출처의 상충 값을 바탕으로 속성의 정답(단일 또는 다중)을 추정하는 데이터 통합 문제로, 기존 연구는 대부분 conflict-resolving에 초점을 맞춰 왔다. 대표적으로 Majority Voting, Source Reliability Vote, LTM, DART처럼 출처 신뢰도나 도메인별 전문성을 가정(혹은 모델링)하며 반복적·확률적 추정으로 정답을 고른다. 하지만 이런 방식은 ‘단일 정답’ 가정에 편향되거나, 문맥에 따른 의미 차이를 유연하게 반영하지 못하고 표현 다양성(표기 변형/정규화)을 별도로 다뤄야 하는 한계가 있다.

- **Core Contribution**: 이 논문은 LLM을 데이터 퓨전의 truth-discovery 구성요소로 직접 사용해, 단일-truth/다중-truth을 모두 다루는 prompt 기반 접근을 체계적으로 탐구한다. 특히 domain-independent(DI) vs domain-dependent(DD), zero-shot vs one-shot의 조합을 만들고, 다중-truth에서는 여러 값을 함께 정답으로 산출하도록 유도하는 프롬프트를 설계했다. 또한 정답 생성 제한(예: 입력에 있는 값만 사용) 같은 제약을 프롬프트에 포함해 동작을 조절하고 신뢰성까지 함께 분석한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 출처 간 의미적으로 같은 값을 다른 표기/형식으로 제시할 때 이를 자연어 의미 수준에서 정렬하고, (2) 다중-truth 설정에서 ‘여러 개가 맞을 수 있음’을 LLM 출력 규칙으로 정확히 반영하는 것이다. 저자들은 이를 위해 단일/다중-truth 전용 프롬프트 구조를 분리하고, 값은 소스에 존재하는 것만 선택하도록 하는 C1 제약, 형식이 다른 동일 값은 하나로 세도록 하는 C2 제약을 두어 출력의 제약을 걸었다. 더불어 one-shot 예시를 프롬프트 앞에 선택적으로 추가해 문맥 추론과 출력 형식의 안정성을 높이도록 했다.

- **Empirical Impact**: 실험은 Book(Movie의 감독/연도 계열), Movie, Flight 데이터셋(각각 다중-truth 2개, 단일-truth 1개)에서 Recall/Precision/F1로 비교했으며, LLM 기반 DD 프롬프트가 전통적 무지도 truth discovery(예: DART, LTM)를 전 데이터셋에서 전반적으로 앞섰다. 특히 Book과 Movie에서는 domain-dependent 프롬프트가 더 높은 F1과 균형 잡힌 recall/precision을 보였고, Flight에서는 단일-truth 구조 덕에 베이스라인도 강했지만 DD/DI 프롬프트가 근접 성능을 내며 LLM의 일관성을 확인했다. 또한 Flight ID를 obfuscation해도 성능 하락이 작아, LLM이 배경지식보다 출처 일치 패턴에 더 의존할 가능성을 시사했으며, 비용은 API 호출당 수 초 수준으로 보고했다.



### SHARD: cell-keyed residual splitting for alignment-resistant private dense retrieva (https://arxiv.org/abs/2606.27976)
Comments:
          arXiv admin note: text overlap with arXiv:2606.26373

- **Prior Approaches**: 연구진은 dense embeddings을 기반으로 한 semantic search와 RAG에서, 벡터 저장소가 유출될 때 텍스트 복원이 가능해지는 문제를 다룹니다. 기존 방어로는 SVD로 차원을 줄인 뒤 단일 secret rotation과 CKKS reranking을 조합하는 “글로벌-선형” 방식이 널리 쓰이지만, 이때 보호되는 기하가 단 하나의 전역 정렬(geometry)이어서 알려진-평문(known-plaintext) 정렬 공격에 취약하다고 지적합니다. 특히 orthogonal Procrustes가 대략 subspace 차원 수준의 anchor로 회전을 복구하고, 이후 일부 공개 인덱스·참조 코퍼스가 결합되면 높은 정확도로 원문을 재구성할 수 있다고 보고합니다.

- **Core Contribution**: 논문은 retrieval 품질을 해치지 않으면서도 “정렬 가능한 약한 축”을 제거하는 공격-인식형 방어 변환 Shard를 제안합니다. Shard는 centered embedding을 짧은 public prefix(1단계 검색용)와 private residual로 분해하고, residual은 C개의 cell로 샤딩한 뒤 cell별 비밀 키로 회전·분산해 서버가 단일 공통 기하를 보지 못하게 만듭니다. CKKS reranking은 이 키가 상쇄되는 방식으로 수행되어 내적이 정확히 복원되며, 결과적으로 half-SVD truncation이 주던 검색 품질 손실을 되돌립니다.

- **Technical Challenges**: 핵심 난제는 두 채널을 동시에 만족시키는 것입니다: (1) 서버가 보게 되는 공개 인덱스·접근 단서가 정렬 표면(alignment surface)이 되지 않게 만들고, (2) 암호화 연산에서는 내적/랭킹이 정확히 유지돼야 합니다. 논문은 residual을 cell-local 마이크로 키로 처리해 전역 정렬 대신 “셀마다 분리된 프레임”을 강제하고, CKKS에서 cell 키가 상쇄되도록 설계해 full-dimensional reranking 정확도를 확보합니다. 또한 파라미터 C 하나로 baseline(C=1)의 전역형에서 per-document micro-keys(C=N)까지 연속적으로 확장되도록 하여, 정렬 저항성과 비용(암호화된 residual 쿼리 수)의 트레이드오프를 조절합니다.

- **Empirical Impact**: 실험에서 Shard는 BEIR 계열 작업에서 half-SVD truncation 기반 baseline 대비 nDCG@10 저하를 줄이고, full-dimensional reranking 덕분에 원(raw-space) 랭킹에 더 가깝게 복원됨을 보여줍니다. 정렬 공격 측면에서는 diffuse한 알려진-평문 누출이 있을 때 private residual을 원래 프레임으로 되돌리기 위한 anchor 복잡도가 대략 C배씩 증가하며(예: C=1 대비 C=256에서 큰 폭 상승), 공격이 더 강하거나 비선형/학습형 정렬러(ALGEN, MLP)·unsupervised vec2vec류를 써도 이 완화가 유지된다고 보고합니다. 다만 한계도 명확히 제시하는데, 셀 내부에서는 키 상쇄가 일어나므로 표적 공격(targeted)과 공개 prefix의 거친 구조 누출, 그리고 겹치는 레퍼런스 코퍼스가 결합될 때는 추가 변형(셀 ID·micro-key 제한 등)이 필요하다고 결론냅니다.



### DysLexLens: A Low-Resource LLM Framework for Analysing Dyslexic Learners Insights from Online Forums (https://arxiv.org/abs/2606.27619)
- **Prior Approaches**: 기존 연구는 크게 (1) 난독증 지원에 대한 기술·도구 현황을 정리하거나 (2) 레딧 같은 온라인 담화를 통해 난독증 당사자의 경험을 주로 정체성/사회적 의미 관점에서 분석하는 방향으로 나뉘었다. 하지만 두 접근 모두 AI 도구가 실제로 어떻게 인식·채택·검증되는지에 대한 학습 관련 근거를 자연스러운 포럼 맥락에서 체계적으로 연결해 주지 못했다.

- **Core Contribution**: 이 논문은 DysLexLens를 제안한다. DysLexLens는 저자원이면서 잡음이 많은 포럼 텍스트를 수집-필터링-지식그래프(KG) 추론-증거 추적-응답 평가의 end-to-end 파이프라인으로 묶어, AI 관련 발화의 학습적 유스케이스와 장단점을 근거 기반으로 분석하도록 한다.

- **Technical Challenges**: 핵심 난제는 (i) 관련 글이 희소하고 잡음이 많아 표본을 정밀하게 구성해야 하며 (ii) LLM이 만든 답을 실제 포럼 증거에 “traceable”하게 연결해야 한다는 점이다. DysLexLens는 딕셔너리 기반 필터링으로 Reddit 코퍼스를 319개 게시물 수준으로 좁힌 뒤, LLM이 KG triple과 의미 분석을 수행하고, 응답의 claim을 chunk ID로 매칭해 원문까지 역추적하는 evidence-tracing 파이프라인을 제공한다.

- **Empirical Impact**: 평가는 난독증- AI 관련 연구질문 RQ1~RQ5에 대한 30개 질의로 수행했으며, Answer Relevancy는 평균 0.75로 전반적 질의 의도를 잘 반영했다. 다만 Faithfulness·Context Relevance·Response Groundedness는 상대적으로 낮아(각각 평균 0.52/0.40/0.43) 추론은 맞지만 claim 수준에서의 근거 매칭과 시간 변화 질의 처리에 한계가 드러났다. 그럼에도 29개 응답에서 최소 1개 chunk ID가 포함되고, 인간 검증에서 39개 claim은 완전 검증 가능, 6개는 증거 부재로 비검증으로 분류되어, “근거 확인 가능한 생성” 방향의 실용성을 보여주며 Github 공개로 재현성도 강화했다.



### Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieva (https://arxiv.org/abs/2606.27401)
Comments:
          15 pages, 4 figures

- **Prior Approaches**: 기존 코드 검색은 BM25, TF-IDF, Jaccard 같은 전통적 IR로 빠르게 후보를 좁히지만, 표면 키워드 매칭에 머물러 정확도가 제한된다. 딥러닝 기반 two-stage recall-then-rerank(TOSS)는 임베딩으로 먼저 recall을 수행한 뒤 cross-encoder로 rerank하지만, 1단계에서 놓친 정답은 뒤에서 복구되지 않는다. 그럼에도 1단계 recall에 대한 체계적 대규모 비교와 효율·확장성(throughput) 분석은 부족했다.

- **Core Contribution**: 논문은 5개 언어(파이썬, 자바, 자바스크립트, C++, C#)와 4개 데이터셋을 대상으로 17개 코드 임베딩/검색 모델의 1단계 recall 성능과 효율을 대규모로 벤치마킹한다. 또한 LLM 기반 code normalisation 및 query/candidate rewriting 전략을 제시해, 성능이 낮고 스타일 민감한 모델에서 precision을 최대 29%까지 끌어올릴 수 있음을 보여준다. 결과적으로 데이터셋 전반에서 코드 특화 LLM 임베딩의 견고성에 대한 가정과 자원 제약 환경의 지속가능성에 의문을 제기한다.

- **Technical Challenges**: 핵심 기술 난제는 1단계 recall의 성능을 유지하면서 terabyte급 코드베이스로 확장할 때 생기는 대규모 계산·인프라 병목이다. 저자들은 정확도(Precision@k, NDCG)를 순수 임베딩 품질 기준으로 분리하기 위해 ANN 없이 exact scanning을 수행하고, Cosine similarity를 일관되게 사용해 색다른 indexing 손실을 상대오차로 관찰한다. 또한 변수/주석/문체 변화가 임베딩에 미치는 영향을 Qwen2.5-Coder-7B-Instruct 기반 변환(주석 정리, 식별자 재명명, LLM 재작성 등)으로 실험하며, 특히 query와 candidate에 함께 적용할 때 이득이 커짐을 분석한다.

- **Empirical Impact**: 실험 결과는 ‘최고 정확도’와 ‘확장 가능한 throughput’ 사이에 강한 trade-off가 있음을 명확히 보여준다: Qwen3 Embedding 등 상위권은 많은 데이터셋에서 Precision@50이 높지만, 큰 비용 때문에 인덱싱·운영이 비현실적일 수 있다. 반대로 StarEncoder 같은 경량 모델은 throughput은 크게 개선(대략 수십 배 수준 차이)하지만, xCodeEval/CodeNet처럼 어려운 설정에서 precision이 크게 하락하며 품질이 급격히 떨어진다. 결론적으로 resource-constrained 환경에서는 경량 인코더 + LLM reranking의 hybrid two-stage 구조가 여전히 가장 효과적이며, 모델 선택은 작업·언어·데이터셋 맥락에 맞춰야 한다는 실무적 가이드라인을 제공한다.



New uploads on arXiv(cs.CV)

### PerceptionRubrics: Calibrating Multimodal Evaluation to Human Perception (https://arxiv.org/abs/2606.28322)
Comments:
          ICML 2026. Project page: this https URL

- **Prior Approaches**: 기존 비전-언어 평가 벤치마크는 종합 점수(예: CLIPScore 등)나 평균 기반 다중항 점수에 의존해, 부분적으로 맞춘 요소가 치명적 국소 오류를 덮는 문제가 컸다. 또한 폐루프/단답형 과제 구성은 언어 우선(shortcut)이나 추측을 허용해 실제 시각적 정합성의 취약점을 충분히 진단하지 못했다. 그 결과 리더보드 상단은 포화되는데도 실사용에서는 객체 수 세기, 공간 관계 역전 같은 고가치 실패가 계속 발생한다.

- **Core Contribution**: PerceptionRubrics는 이미지 캡셔닝을 기반으로, ‘총체적 의미 일치’ 대신 원자 단위 검증(atomic auditing)에 초점을 맞춘 루브릭 기반 평가 프레임워크를 제안한다. 1,038장의 정보 밀집 이미지를 각 인스턴스에 맞춘 12,000+개의 instance-specific 루브릭과 연결하며, 골든 캡션은 Circular Peer-Review 합의 파이프라인으로 만든다. 최종 점수는 Must-Right(필수 사실)와 Easy-Wrong(자주 나는 오답/환각) 이중 스트림을 쓰고, 필수 사실 실패 시 강한 이진 페널티를 주는 Gated Scoring으로 인간 감도에 맞춘다.

- **Technical Challenges**: 핵심 난제는 (1) 루브릭을 이미지 픽셀에서 직접 만들면 visual grounding gap 때문에 잡음이 커진다는 점과 (2) 점수 체계가 선형 평균이면 치명적 오류를 희석한다는 점이다. 논문은 먼저 이미지 내용을 고정밀 텍스트로 옮긴 뒤(캡셔닝 중간 단계) 여러 MLLM의 원형 피어리뷰와 인간 검증으로 Golden Captions를 만들고, 이를 Gemini-3-Pro로 Must-Right/Easy-Wrong 원자 루브릭으로 증류한다. 이어 평가 시 LLM-judge가 각 루브릭의 True/False를 산출하되, Must-Right 하나라도 실패하면 게이트가 닫혀 최종 점수가 0으로 급락하도록 설계해 비선형 오류 민감도를 구현한다.

- **Empirical Impact**: 25개 MLLM을 대상으로 한 실험에서 PerceptionRubrics는 기존 벤치마크에서 잘 드러나지 않던 ‘Reliability Gap’을 명확히 보여준다(원자 수준은 맞추지만 필수 조건의 동시충족은 자주 실패). 또한 오픈소스와 상용의 시각 인지 격차가 누적되어 8% 수준의 perception deficit가 지속됨을 정량화하고, 점수가 인간 선호 기반 Vision Arena와의 상관(피어슨 0.916, 스피어만 1.000)에 더 잘 맞는다고 보고한다. 종합하면, 게이트 기반 엄격 루브릭이 “있어 보이는 유사도”가 아닌 “정확한 인지 신뢰성”을 진단하는 데 유효하다는 점에서 향후 MLLM 평가/개선의 방향성을 제시한다.



### StructSplat: Generalizable 3D Gaussian Splatting from Uncalibrated Sparse Views (https://arxiv.org/abs/2606.28321)
Comments:
          Project page: this https URL Code: this https URL

- **Prior Approaches**: 기존 3D Gaussian 재구성 방법은 장면별 최적화에 의존하거나, 카메라 포즈를 알고 있다는 가정을 두는 경우가 많았습니다. 또한 단일 백본에서 기하와 외관을 함께 학습해 정보가 뒤섞여 재구성 정밀도와 새로운 데이터로의 일반화가 제한되는 문제가 있었습니다.

- **Core Contribution**: StructSplat은 보정되지 않은(un-calibrated) 이미지에 대해서도 카메라 파라미터 없이 동작하는 feed-forward 3D Gaussian 재구성 프레임워크를 제안합니다. 핵심은 기하, 의미(semantic), 텍스처를 역할별로 분리해 구조화된 표현을 만들고, 각 단서가 재구성에 미치는 영향을 명시적으로 설계하는 것입니다.

- **Technical Challenges**: 첫째, 2D 관측에서 정확한 텍스처를 모델링하려면 픽셀 정렬(pixel-aligned) 기반의 feature injection이 필요합니다. 둘째, 의미 단서를 활용해 전역 일관성을 높이기 위해 semantic-aware priors를 도입하고, 셋째로 카메라 정렬 과정에서 정보 누설을 막아 generalization을 개선하기 위한 camera alignment strategy를 함께 설계했습니다.

- **Empirical Impact**: 실험 결과 StructSplat은 어려운 벤치마크에서 기존 대비 유의미하게 높은 성능을 보였습니다. DL3DV에서 PSNR 28.045로 AnySplat(22.377)보다 +5.67 dB 높았고, 교차 데이터셋에서도 ACID에서 +1.94 dB, RealEstate10K에서 +1.72 dB 개선을 기록해 일반화 강점을 실증했습니다.



### Learning Topology-Aware Representations via Test-Time Adaptation for Anomaly Segmentation (https://arxiv.org/abs/2606.28268)
- **Prior Approaches**: Test-time adaptation(TTA)는 분포 변화 상황에서 딥 모델을 보정하는 유망한 방법이지만, 이상 탐지·세그멘테이션(AS)에서는 픽셀 단위 마스크를 만들기 위해 confidence thresholding이나 entropy minimisation 같은 픽셀 수준 휴리스틱에 의존하는 경우가 많습니다. 이런 방식은 잡음과 텍스처 변동이 생기면 구조적 일관성을 잃고, 이상 맵을 단순한 강도(field)로 취급해 연결성·구멍 같은 고차 공간 관계를 반영하지 못합니다. 또한 기준선으로 쓰는 임계값이 정상 데이터에 고정돼 있어, 다양한 결함 형상으로의 일반화가 흔들립니다.

- **Core Contribution**: TopoTTA(Topological Test-Time Adaptation)는 TTA 파이프라인에 persistent homology(지속동형, PH) 기반 위상 정보를 결합해 적응 중에도 기하·구조적 일관성을 유지하도록 설계했습니다. 이상 점수 맵에서 multi-level cubical complex filtration을 적용해 견고한 topological pseudo-label을 만들고, 이를 이용해 백본을 retraining하지 않으면서도 경량 test-time classifier가 마스크를 정교화하도록 유도합니다. 특히 특정 점수 임계값으로 마스크를 이진화하는 방식에서 벗어나, 연결성과 위상적 구조를 보존하는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 잡음/텍스처 변화가 있는 이상 점수 맵에서 위상 구조를 얼마나 안정적으로 뽑아 pseudo-label로 변환할지, (2) 그 pseudo-label이 픽셀 단위 세그멘테이션 학습에 실제로 유효하도록 만드는 것입니다. TopoTTA는 sublevel과 superlevel을 함께 쓰는 bidirectional multi-level cubical filtration로 connected components와 holes 같은 구조를 다중 스케일로 포착하고, 가장 지속적인(top-k 또는 persistence 기준) 특징만 남겨 pseudo-label을 구성합니다. 이후 frozen pre-trained backbone의 feature로부터 픽셀 레벨 contrastive encoder(PCES)를 test time에 on-the-fly로 학습해, 이상/정상 영역의 특징을 pseudo-label 공간에서 정렬·분리하도록 학습합니다.

- **Empirical Impact**: TopoTTA는 MVTec AD, VisA, Real-IAD, MVTec 3D-AD, AnomalyShapeNet, MVTec LOCO 등 6개 벤치마크에서 평균적으로 F1이 SOTA 대비 약 15% 향상됐고, 복잡한 기하·구조 변화를 보이는 이상에서 개선 폭이 특히 컸습니다. 다른 실험에서도 2D에서 최대 +20.3%, 3D에서 +10.2%의 F1 개선을 보고해, 2D/3D 모달리티와 여러 백본 전반에서 일관된 성능을 확인했습니다. 위상 기반 구조 추론을 TTA에 통합함으로써 기하 학습과 견고한 적응 사이의 간극을 메운다는 점에서, 산업용 anomaly segmentation의 일반화 방향성을 제시합니다.



### RSICCLLM: A Multimodal Large Language Model for Remote Sensing Image Change Captioning (https://arxiv.org/abs/2606.28266)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 RSICC(원격탐사 이미지 변화 캡셔닝) 연구는 SAM 기반 이중 인코더, 변화 감지(CD) 분기, 프롬프트 기반 생성 등 주로 conventional deep learning 프레임워크에 의존해 왔다. 그러나 데이터 부족과 모델 용량 한계로 인해 조명·기상 같은 무관 요인을 무시하면서 미세한 시간 차이를 언어로 정확히 정렬하는 데 어려움이 컸다. 한편 일반 도메인의 large vision-language 모델 post-training은 성공했지만, RSICC는 데이터 희소성과 fine-grained 변화 이해 요구 때문에 그대로 전이하기 어렵다고 본다.

- **Core Contribution**: 이 논문은 RSICC에 특화된 대규모 비전-언어 모델 post-training 프레임워크 RSICCLLM을 제안한다. 먼저 양방향 이미지와 이진 change mask를 조건으로 Qwen-VL-Max로 지시문을 합성해 RSICI 데이터셋을 만들고, 이를 바탕으로 RSICC 전용 벤치마크를 구축했다. 또한 변화 인식을 명시적으로 유도하는 Difference-aware Supervised Fine-tuning과, 두 갈래 negative 샘플로 preference 데이터 RSICP를 만드는 Dual-Negative Preference Optimization(DNPO)을 결합한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 시간적 차이(무엇이 변했는지)를 정확히 분리해 모델이 ‘변화 표현’을 만들게 하는 것, (2) RSICC에는 명시 라벨이 부족해 preference 학습에 쓸 고품질 부정 샘플을 안정적으로 구성하는 것이다. 저자들은 differencing 전에 공간 등록과 함께 CDConv 계열 커널 및 Hough Transform으로 시간 불변 기하-텍스처 통계를 재구성해 변화 단서를 강화하고, TT와 ΔS를 cross-attention으로 융합해 change representation을 얻는다. 부정 샘플은 정보량 기준의 negative sample selection과 정답 키워드 교체 기반 replacement을 함께 써 자연스러우면서도 의미적으로 틀린 캡션을 만들고, DPO 변형과 KL 제약으로 학습을 안정화한다.

- **Empirical Impact**: 실험 결과 RSICCLLM은 인도메인에서 7B 파라미터만으로도 더 큰 규모의 일반 모델과 도메인 특화 모델을 제치며 다수 지표에서 우수한 성능을 보였다. 또한 아웃도메인 평가에서도 변화 분석 능력을 유지하며 전반적으로 강한 일반화 성향을 확인했다. 데이터셋 측면에서도 RSICI(약 4만 이미지-2만 캡션)와 RSICP(최초 공개 preference 데이터)를 공개해 RSICC 연구의 학습·검증 기반을 확장했다.



### Exposure Bias Can Alleviate Itself via Directional and Frequency Rectification in Flow Matching (https://arxiv.org/abs/2606.28226)
Comments:
          arXiv admin note: text overlap with arXiv:2512.04904

- **Prior Approaches**: Flow Matching(FM)은 연속시간 흐름을 학습해 효율적인 생성 성능을 보여줬지만, 학습 시의 섞인 입력(잡음+데이터)과 추론 시의 재귀 예측 입력 사이 불일치로 exposure bias가 발생한다. 기존 대응은 DDPM/IP 계열처럼 노이즈 주입이나 정적 정렬, 혹은 추론 안정화(예: scheduled sampling) 중심이어서 편차의 ‘정도’에 따라 능동적으로 되돌리는 제어 신호를 제공하긴 어렵다.

- **Core Contribution**: 이 논문은 exposure bias 자체가 방향성과 주파수(특히 저주파) 결핍을 담고 있다는 관찰에 기반해, bias를 이용해 bias를 교정하는 DEFAR를 제안한다. DEFAR는 학습 중 단일-step 추론 시뮬레이션으로 bias를 추출하고, 이를 두 축(방향/주파수)의 피드백으로 재활용해 모델의 편차 내성을 높인다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 재귀적 드리프트를 단일-step 학습 신호로 안정적으로 분해해 ‘바르게 되돌릴 방향’을 학습하는 것과 (2) 그 드리프트가 주파수 대역 결핍으로 어떻게 연결되는지 정량화하는 것이다. 저자들은 ADR에서 드리프트된 상태를 목표 엔드포인트로 되돌리는 복원 방향(각도 기반으로 신호 강도를 동적으로 조절)을 정규화하고, FC에서는 Predicted Frequency Ratio(PFR)/Frequency Emphasis of Loss(FEL) 분석을 통해 bias가 저주파 구조를 보완하는 역보완 성질을 가진다는 점을 이용해 결핍 저주파를 강화하도록 손실을 bias-가중 재조정한다.

- **Empirical Impact**: CIFAR-10, CelebA-64, ImageNet-256/512에서 DEFAR는 기존 베이스라인을 능가했으며, NFE 50 기준으로도 성능과 추론 견고성, 확장성에서 이점을 보였다. 또한 FM의 exposure bias를 ‘정적 보정’이 아니라 bias 내부 신호로 ‘자기 직정(self-rectification)’하는 관점이 제시돼, 향후 FM 및 유사 생성모델의 inference-robust 학습 설계에 의미 있는 방향을 제공한다.



### HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration (https://arxiv.org/abs/2606.28215)
Comments:
          Accepted to ECCV 2026. 15 pages of main text and 39 pages of appendices. Project page: this https URL

- **Prior Approaches**: 기존 모노큘러 기반 4D 상호작용 복원은 크게 (1) 동기화 다중 카메라 하드웨어에 의존해 occlusion을 줄이거나, (2) 단일 영상에서 생성모델/재구성모델로 4D를 만드려는 시도로 나뉜다. 전자는 비용·제약이 커 오픈월드 확장성이 떨어지고, 후자는 고정/스타일화 자산 위주라 현실의 복잡한 다중 물체 상호작용에서 물리적으로 그럴듯하지 않거나 시간적 흔들림이 생긴다.

- **Core Contribution**: 이 논문은 HAT-4D라는 인간-에이전트 협업(agentic) 프레임워크로, 단일 모노큘러 비디오에서 다중 물체의 3D 기하·시간 동역학·물리적 상호작용을 동시에 복원한다. 핵심은 Interaction Knowledge Graph (IKG)로 장기 물리 변화와 상호작용 단서를 구조화해, 3D 생성/배치와 4D propagation이 같은 제약을 따라가도록 만드는 점이다. 또한 비싼 멀티카메라 없이도 “물리적으로 그럴듯한” 4D 에셋을 대량 생성하는 데이터 엔진을 제시한다.

- **Technical Challenges**: 모노큘러 영상은 깊이 모호성과 상호 occlusion이 심해, 복원 과정에서 오류가 누적되면 시간 일관성이 쉽게 무너진다. 논문은 IKG로 깊이/관계/이벤트 경계를 명시하고, 메모리 뱅크 기반 segment-wise 4D propagation과 4D 생성 평가자(critic)로 물리 위반·장단기 메모리 붕괴를 감지해 국소 재생성 또는 단계 롤백을 수행한다. 여기에 다단계 human-in-the-loop 수정(gaussian/region/object 수준)과 온라인 fine-tuning을 결합해 모호한 상황에서 인간 지식을 생성 루프에 주입한다.

- **Empirical Impact**: MVOIK-4D는 77개 태스크·112개 상호작용 시나리오로 구성된 오픈월드 벤치마크이며, 변형 현실성·상호작용 일관성·시간 매끄러움·장기/교차뷰 메모리 보존을 포함한 다차원 평가를 제공한다. 실험 결과 HAT-4D는 대부분의 지표에서 SOTA 성능을 보이고, 변형/관계/메모리 일관성 측면에서 특히 강한 개선을 보인다. 또한 소량의 인간 피드백 도입만으로 상호작용 복원이 향상되며, 생성된 데이터는 fine-tuning에 활용 시 기존 baseline 성능을 끌어올리는 것으로 입증된다.



### EchoSonar-R: A Multi-View Reasoning-Enabled Model for Disease Classification and Report Generation in Echocardiography (https://arxiv.org/abs/2606.28164)
- **Prior Approaches**: 기존 연구는 단일 과제(뷰 분류, 챔버 분할, EF 추정 등) 성능 향상에 집중해 다중 뷰에서의 상호보완 정보를 합성하지 못하는 한계가 있었다. 최근 EchoCLIP, EchoPrime 같은 echocardiography foundation model도 라벨 예측이나 리포트 검색/생성에 머무는 경우가 많아, 교차 뷰 근거에 기반한 명시적 진단 추론과 해석 가능성이 약해 임상 신뢰 확보가 어렵다는 지적이 제기된다.

- **Core Contribution**: EchoSonar-R은 다중 뷰 질환 분류와 구조화된 임상 리포트 생성을 한 모델에서 동시에 수행하면서, 교차 뷰 reasoning trace를 함께 제공하는 vision-language 프레임워크를 제안한다. 또한 spatiotemporal video encoder와 structure-aware cardiac detector를 결합해 전역 운동 정보뿐 아니라 해부학적으로 근거가 표시된(공간적으로 grounded) 단서를 리포트 생성에 동원한다. 학습은 SFT(추론 주석 타깃 기반) 후 GRPO(Group Relative Policy Optimization)로 분류·리포트 목표를 단일 RL 프레임워크에서 정렬한다.

- **Technical Challenges**: 다중 뷰 입력을 언어모델에 효과적으로 주입하면서, 근거가 되는 해부학적 영역을 reasoning에 연결하는 것이 핵심 기술 과제였다. 이를 위해 냉동된 mViT 기반 spatiotemporal encoder 토큰과 RT-DETR-L 기반 7개 핵심 구조 detector 쿼리를 각각 projection한 뒤, view identifier와 함께 인터리빙하여 언어모델이 뷰 간 증거를 통합하도록 설계했다. 또 RL 단계에서는 token-level 모방만으로는 과제 정확도를 직접 최적화하기 어렵기 때문에, 구조적 형식 보장, 분류의 IoU 기반 correctness, 리포트의 echocardiography-aware semantic similarity, 길이 패널티를 포함한 task-specific reward로 GRPO를 수행했다.

- **Empirical Impact**: 그 결과 EchoSonar-R은 private multi-view 데이터에서 매크로 balanced accuracy를 17.1% 개선(최강 baseline 대비)했고, MIMICEchoQA에서도 6.1% 향상했다. 특히 GREEN clinical faithfulness score 0.800을 달성하며, <think> 구간의 reasoning이 시각 근거에 기반해 생성되는 해석 가능성까지 함께 보고한다. 저유병 질환에서의 성능 개선 폭이 커 GRPO가 데이터 비대칭이 큰 설정에서 결정 경계를 더 잘 다듬는다는 의미를 시사한다.



### Toward Robust In-Context Segmentation via Concept Guidanc (https://arxiv.org/abs/2606.28149)
Comments:
          ECCV 2026

- **Prior Approaches**: In-context segmentation (ICS)는 파라미터 업데이트 없이 소수의 reference 이미지와 마스크로 query의 타깃 영역을 분할하는 문제다. 기존 연구들은 주로 저수준 시각 매칭에 의존해 정확도 향상에 집중했지만, 같은 query에 대해 reference를 바꿨을 때 결과가 얼마나 안정적인지(robustness)는 상대적으로 간과해왔다.

- **Core Contribution**: 이 논문은 ICS를 robustness 관점에서 재정의하고, Concept-Guided In-Context Segmentation (CG-ICS)라는 새 패러다임을 제안한다. CG-ICS는 reference에서 고수준 의미 concept를 추출해 분할을 유도하며, SAM3의 frozen backbone을 활성화하는 데 텍스트 concept와 시각 exemplar를 함께 활용한다.

- **Technical Challenges**: 핵심 난제는 신뢰할 수 있는 textual concept를 reference들로부터 어떻게 안정적으로 고르느냐와, concept와 query의 위치 정합을 어떻게 보장하느냐이다. 이를 위해 MLLM이 concept 후보를 제안하고, SAM3-driven scoring 함수와 tree-search refinement로 신뢰도 높은 concept를 선택하며, 별도의 visual exemplar 경로에서는 간단한 context construction을 통해 query-side 공간 grounding을 제공한다.

- **Empirical Impact**: 표준 ICS 벤치마크에 대한 대규모 실험에서 CG-ICS는 정확도에서 state-of-the-art를 달성하는 동시에 robustness도 크게 향상시켰다. 특히 다양한 reference 선택에 대해 분할 결과의 분산(variance)을 크게 줄여, 더 신뢰할 수 있는 ICS 시스템으로 이어진다는 점에서 의미가 크다.



### Monocular Avatar Reconstruction via Cascaded Diffusion Priors and UV-Space Differentiable Shading (https://arxiv.org/abs/2606.28144)
Comments:
          Accepted by ECCV 2026. Project page: this https URL

- **Prior Approaches**: 단일 in-the-wild 이미지로 고품질의 relightable 3D avatar를 복원하는 일은 PBR(물리 기반 렌더링) 데이터 부족과 조명(illumination)·재질(material)을 분리하기 어려워 대표적인 ill-posed 문제로 꼽힌다. 기존 방법들은 별도 모듈을 쪼개 파이프라인을 만들거나, 대규모·독점 데이터에 의존해 안정성을 확보하는 경우가 많아 데이터 효율과 일반화가 약점으로 지적됐다.

- **Core Contribution**: 이 논문은 통합 pre-trained diffusion backbone의 강한 생성 prior를 활용해 texture completion(텍스처 완성)→delighting(탈조명)→material decomposition(재질 분해)을 순차적으로 처리하는 데이터 효율적 프레임워크를 제안한다. 각 서브태스크를 UV 공간에서 cascaded Low-Rank Adaptations(cascaded LoRAs)로 맞춤 적응해, 파편화된 파이프라인 없이도 PBR 맵(Albedo/Normal/Roughness/Specular/Displacement)을 함께 합성한다는 점이 핵심이다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 가려진 부위의 UV 텍스처를 의미적·광학적으로 일관되게 완성하고, (2) baked-in lighting이 남은 상태에서 재질을 정교하게 분리하며, (3) 생성 결과가 렌더링 방정식을 위배하지 않게 물리적 타당성을 강제하는 것이다. 저자들은 Inpainting LoRA로 텍스처 결손을 복원한 뒤 Light-Homogenization LoRA와 Cross-Intrinsic Attention으로 조명을 제거하고 픽셀 정렬된 PBR 지도를 공동 생성하며, 분해 단계에서 UV-space differentiable BRDF shading loss로 rendering equation 준수를 학습시켜 라스터 기반 감독에서 흔한 아티팩트를 줄인다.

- **Empirical Impact**: 실험 결과, 모델은 100개 미만의 실제 3D scan으로 학습해도 4K급 해상도의 포괄적 PBR 자산을 생성하며, 기존 state-of-the-art 대비 현실감과 일반화 성능이 더 좋다고 보고된다. 또한 acceptance 후 학습 코드와 모델 가중치를 공개하겠다고 밝혀 재현성과 후속 연구 확장에 긍정적 신호를 준다.



### PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation (https://arxiv.org/abs/2606.28128)
Comments:
          Github: this https URL Project website: this https URL

- **Prior Approaches**: 기존 비디오 생성은 시각적 품질이나 액션 조건 만족에 집중하는 경우가 많아, 접촉이 많은 조작에서 불연속 궤적·물체 침투·중력 위반 같은 물리 부정합이 남는 한계가 있었다. 로봇용 데이터로 fine-tuning한 모델도 reconstruction류 목적이 배경과 상호작용 구역을 균일하게 다뤄, 접촉 주변의 국소 물리 오류와 전역 상호작용 결과 오류를 동시에 잡기 어렵다.

- **Core Contribution**: 이 논문은 물리적 타당성을 (1) 픽셀 레벨 국소 운동, (2) 시맨틱 레벨 상호 관계의 계층적 정렬, 그리고 (3) 조작에 중요한 영역에만 감독을 거는 region-focused 정렬 문제로 재정의한다. 이를 바탕으로 PhysisForcing을 제안해, 접촉·조작기·움직이는 물체 같은 physics-informative regions에 한정해 궤적 연속성과 객체-로봇 관계성을 함께 강화한다.

- **Technical Challenges**: 핵심 기술 난점은 두 가지인데, 접촉 구간이 국소적으로만 정보가 강하다는 점(균일 감독의 희석)과, 국소 궤적 문제만으로는 전역 상호작용 결과가 보정되지 않는다는 점이다. 저자들은 참고 비디오의 point tracking과 깊이 기반 전경 가중치를 이용해 physics-informative 마스크를 만든 뒤, (i) DiT 중간층 feature에 대한 pixel-level trajectory alignment(마스크 내 포인트 궤적 MSE)와 (ii) frozen video understanding encoder로부터 얻은 토큰 간 관계를 DiT에 전이하는 semantic-level relational alignment(관계 유사도 정렬)를 joint로 최적화한다.

- **Empirical Impact**: R-Bench, PAI-Bench, EZS-Bench에서 강한 베이스라인 대비 일관된 개선이 확인됐고, 특히 R-Bench에서 Wan2.2-I2V-A14B는 22.3%, Cosmos3-Nano는 9.2% 물리 일관성 향상을 보였다(일부는 vanilla finetuning 대비 7.1~3.7%p 추가 개선). 더 나아가 WorldArena action-planner 프로토콜에서 closed-loop success rate를 16.0%→24.0%로 끌어올려 world model 계획(planning) 성능과 downstream 정책 성공에도 긍정적임을 보여, 물리 정렬된 비디오 모델이 로보틱스 표현 학습에 유리하다는 신호를 제공한다.



### BiDeMem: Bidirectional Degradation Memory for Explainable Image Restoration (https://arxiv.org/abs/2606.28112)
- **Prior Approaches**: 복원 모델들은 noise, haze, rain, blur, 압축열화 등 다양한 열화를 다루기 위해 degradation-aware prompts·조건·latent priors를 점점 더 쓰지만, 평가는 대개 PSNR/SSIM 같은 엔드포인트 성능에 머뭅니다. 이 방식은 조건이 진짜 ‘의미 있는 열화 prior’인지, 아니면 추가 모델 capacity나 global correction bias, 데이터셋 shortcut에 불과한지 구분이 어렵습니다. 기존 memory/조건 연구도 경로 자체를 반사실적으로 검증(counterfactual)하는 수준까지는 상대적으로 덜 엄격했습니다.

- **Core Contribution**: BiDeMem은 복원에 쓰인 degradation memory를 ‘설명 가능한 prior’로 취급하기 위한 양방향(bidirectional) 설계를 제안합니다. 입력 통계와 복원 특징으로 top-k 메모리 슬롯을 조회하고, 같은 슬롯 identity를 추론 시 복원 조건화와 학습 시 forward-degradation(깨끗한 타깃에서 관측 열화를 재생성) 설명 경로에 함께 사용합니다. 이를 통해 “어떤 슬롯이 어떤 역할을 했는지”를 마스킹/교체/셔플 같은 개입으로 검증 가능한 형태로 만듭니다.

- **Technical Challenges**: 핵심 기술 과제는, 슬롯 기반 prior가 복원 성능을 높이면서도 단순 잔차 보정 head나 dense FiLM 같은 일반 조건화 효과로 환원되지 않도록 경로 수준에서 분리하는 것입니다. 논문은 NAFNet 기반 통제 실험에서 correction-head-only, dense query-FiLM, static/global prior 같은 강한 대조군을 두고, active slots와 inactive slots를 분리 마스킹해 슬롯 의존성을 확인합니다. 또한 same selected slot identity를 복원 경로와 열화 설명 경로에 공유해, prior가 관측 열화 증거에 정렬(aligned)되고 잘못된 prior에는 민감하게 반응하는지 동시에 학습·평가합니다.

- **Empirical Impact**: 통제된 multi-degradation(denoising/deraining/dehazing) NAFNet 실험에서 BiRank Memory는 평균 PSNR/SSIM이 8개 벤치마크에서 29.7529 dB/0.8865로 보고되며, correction-head-only(0.2588 dB), dense-prior(0.2586 dB), static/global-prior(0.2839 dB) 대비 열화 prior 효과가 더 크다고 주장합니다. 더 중요한 것은 개입 민감도인데, wrong-prior drop이 Rank Memory 0.2365 dB에서 BiRank Memory 1.0430 dB로, native/non-native gap도 0.3484 dB에서 0.6134 dB로 확대됩니다. 네트워크를 외부 백본(AirNet, PromptIR)에도 대응해 미세조정했을 때도 BiRank가 성능을 유지·향상시키는 경향을 보였지만, 계산 효율은 기본 네트워크 대비 불리하고 이득은 세팅 의존적이라 보완 연구 여지가 남습니다.



### Cross-view Multimodal Vision-Based Assessment Framework for Traditional Chinese Medicine Rehabilitation Training (https://arxiv.org/abs/2606.28104)
Comments:
          Published in IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2026

- **Prior Approaches**: 기존 AQA(행동 품질 평가)는 주로 단일 시점에서 인체 골격(관절/포즈)만을 이용해 전신 또는 큰 동작의 품질을 평가하는 경향이 강했다. 그러나 침술·추나처럼 손이 도구를 조작하는 미세 동작은 잦은 self-occlusion(자기 가림)과 손–물체 상호작용 때문에 단일 시점 포즈만으로는 환경 맥락을 충분히 담기 어렵다.

- **Core Contribution**: 논문은 TCM(전통의학) 재활 훈련을 위한 cross-view(교차 시점) 멀티모달 AQA 프레임워크 CME-AQA를 제안한다. 핵심은 visual-pose fusion(시각-포즈 융합)으로 손 조작의 기하 정보를 안정적으로 결합하고, 학습 때 1인칭·3인칭 영상을 함께 써 추론 견고성을 높이는 것이다.

- **Technical Challenges**: CME-AQA가 직면한 첫 난제는 시각 단서가 가려질 때 포즈가 상호작용 기하의 기준 신호로 남도록 만드는 정교한 융합 설계다. AVPF는 pose-conditioned cross-attention으로 시각 표현만 업데이트해 포즈 구조를 기하 레퍼런스로 유지하며, 추가로 MVA는 egocentric(1인칭)–exocentric(3인칭) 간 계층적 표현 정합성을 멀티스케일로 강제해 single-view 추론에서도 일관성을 확보한다.

- **Empirical Impact**: 실험에서 CME-AQA는 TCM-AQA61-A(침)와 TCM-AQA61-T(추나)에서 경쟁 기준 대비 핵심 지표(예: Needle Depth, Quick Needle Insertion)에서 weighted F1이 상대 10% 이상 개선됐다. 또한 insertion time, manipulation frequency 같은 연속형 메트릭의 MAE가 줄어 정량 오차까지 낮췄고, CPR 데이터셋에서도 자세 기반 기준 일부에서 기존 수준과 비슷한 성능을 보여 구조화된 임상 시뮬레이션 기술 평가로의 확장 가능성을 시사한다.



### OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Remova (https://arxiv.org/abs/2606.28094)
Comments:
          Code and resources are available at this https URL

- **Prior Approaches**: 기존 object removal은 GAN 기반이나 inpainting, diffusion 기반으로 발전했지만, 대부분이 마스크 품질에 민감하고 제거 경계나 잔여 효과(그림자·반사)를 완전히 다루지 못했다. diffusion 기반도 반복 denoising이 필요해 계산 비용이 커 상호작용/엣지 환경에 제약이 컸다.

- **Core Contribution**: 논문은 OSOR(One-Step Object Removal)을 제안해 한 번의 denoising 패스로 배경을 복원하면서 그림자·반사 같은 효과까지 함께 제거하도록 만든다. 동시에 user mask가 부정확하거나 효과 영역을 누락해도 동작하도록 mask-robust 설계를 포함한다.

- **Technical Challenges**: 단일 스텝에서는 마스크 경계 주변에서 seam/블러가 생기기 쉬운데, 이를 위해 occupancy-guided discriminator가 패치 단위 경계 감독을 분수 점유율로 제공한다. 또한 잘못된 마스크를 보정하기 위해 lightweight alpha head를 붙이고, 불완전 마스크 conditioning과 alpha compositing으로 모델이 제공된 범위 밖 효과까지 추론하게 학습한다.

- **Empirical Impact**: 데이터 측면에서는 SAVP로 노이즈 instruction 기반 triplet에서 효과 인지 supervision을 뽑아 CORNE을 28만 검증 removal pair로 구축하고, AnimeEraseBench와 TextEraseBench도 추가했다. 실험 결과 OSOR은 강한 multi-step diffusion 대비 지각 품질이 우수하면서도 추론 속도는 4×~30× 빨라져 1024×1024 이미지를 단일 A100에서 1초 내 처리한다.



### Diffusion Model Attribution via Spectral Coupling of Denoiser Responses (https://arxiv.org/abs/2606.28092)
- **Prior Approaches**: 생성 이미지로 특정 diffusion model의 출처를 가리기는 어렵다. 이유는 서로 다른 학습 데이터로 훈련된 모델이 비슷한 score function/출력 분포로 수렴해, 출력 공간에는 분별 신호가 거의 남지 않기 때문이다. 기존 비침습 방법은 출력이나 디코더 수준의 신호에 의존해(공유 autoencoder 시) 구분력이 붕괴하거나, inversion 기반은 후보마다 최적화가 필요해 느리고 공유 autoencoder에서는 재구성 손실이 무의미해지는 한계가 있다.

- **Core Contribution**: 이 논문은 Spectral Denoising Signatures(SDS)로, 생성 결과가 아니라 denoiser의 스펙트럼적 동작을 “지문”으로 사용해 모델을 귀속하는 비침습 방법을 제안한다. denoiser가 노이즈를 제거하는 과정에서 공간 주파수 대역별 에너지를 재분배하는 패턴(스펙트럼 기하)을 주기적으로 주파수 제어 perturbation으로 관측해, 각 후보 모델의 고유 서명을 추출한다. SDS는 입력 이미지별 inversion, 최적화, enrollment 없이 표준 forward pass만으로 서명을 만들고 분류/매칭한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) denoiser 내부의 미묘한 차이를 (2) 출력 분포가 붕괴하는 조건에서도 일관되게 잡아내는 것이다. SDS는 주파수 링으로 라디얼 분해한 뒤 특정 시점에서 band-limited perturbation을 주고, denoiser residual이 어떤 주파수 링으로 “흘러가는지”를 coupling matrix 형태로 측정해 signature를 구성한다. 또한 timesteps와 perturbation amplitude에 대해 다중 집계를 수행해 denoiser Jacobian이 담는 저주파/고주파 성분의 상보 정보를 누적시켜 식별력을 높인다.

- **Empirical Impact**: 8개 서로 다른 diffusion model(훈련 데이터·아키텍처·학습 절차가 다른 조합)을 대상으로 SDS는 약 99.9% 정확도를 달성하며, prompt가 달라지는 cross-domain 설정에서도 96.2%로 유지된다. 동일/유사 autoencoder를 공유하는 극단적 닫힌 집합(closed-set)에서도 192~960차원 수준의 compact signature만으로 선형 분류가 잘 동작해, “스펙트럼 기하”가 실질적인 attribution 근거가 됨을 보여준다. 결과적으로 기존 inversion 기반의 실패 모드를 denoiser 수준에서 직접 해소하며, provenance verification과 지식재산 보호에서 실용적인 비침습 표준 도구로 자리할 가능성을 제시한다.



### RPM-Distill: Physiology-guided Adaptive Cross-modal Distillation for Robust Remote Physiological Measuremen (https://arxiv.org/abs/2606.28089)
Comments:
          Accepted by ECCV2026

- **Prior Approaches**: 비디오 기반 RPM(rPPG)은 조명 변화, 피부 톤, 움직임으로 잡음과 시간적 아티팩트가 커져 성능이 쉽게 흔들립니다. 이를 보완하려는 영상-레이더 융합도 존재하지만, 대개 추론 단계에 레이더를 함께 요구하거나 학습용 쌍데이터가 충분치 않아 배포가 어렵습니다. 또한 기존 cross-modal KD는 모달리티 간 물리/형상 차이 때문에 주파수·시간 정렬이 부정확하면 잘못된 지식을 그대로 전파할 위험이 큽니다.

- **Core Contribution**: 이 논문은 훈련 중에만 레이더를 teacher로 쓰고, 추론은 비디오만으로도 작동하는 RPM-Distill을 제안합니다(훈련-time privileged information). 레이더의 파동형은 RGB와 다르지만 주파수 영역의 잠재 주기 리듬은 공유된다는 관찰을 바탕으로, 생리 구조가 반영된 spectral distillation 증거를 비디오 학생에게 전달합니다.

- **Technical Challenges**: 핵심 난제는 (1) 모달리티 불일치로 인한 정렬 실패, (2) teacher 품질/동기화 불확실성에 따른 negative transfer, (3) RPM 실패 양상이 주파수 도메인에서 구조적으로 나타난다는 점입니다. 이를 위해 저자들은 대역제한 power spectrum의 세 성분(기본 피크 anchoring, off-peak 배경 매칭, spectral morphology/선명도 보존)에 직접 대응하는 손실을 설계하고, sample-level distillation gate와 컴포넌트 가중치를 spectral relation map([ℓv, ℓr, |ℓv-ℓr|])에서 예측하는 정책 네트워크를 bilevel meta objective로 학습해 신뢰할 수 있을 때만 정렬이 강하게 일어나도록 했습니다.

- **Empirical Impact**: 여러 조건과 교차 데이터셋 평가에서 RPM-Distill은 단일 모달 기준 대비 MAE 81% 개선, correlation 21% 향상을 보고하며 강건성을 실증했습니다. 이는 레이더를 추론에 넣지 않아도 비디오 RPM의 주요 고장 패턴(피크 위치/오프피크 잡음/스펙트럼 번짐)을 주파수 기반 생리 제약으로 억제할 수 있음을 보여주는 결과로, 실사용 배포 가능성을 높인다는 점에서 의미가 큽니다.



### STAG: Spatio-temporal Evolving Structural Representation of Action Units for Micro-expression Recognition (https://arxiv.org/abs/2606.28083)
- **Prior Approaches**: 기존 마이크로표정인식(MER) 연구는 매우 짧고 미세한 얼굴 근육 움직임을 다루기 위해 주로 apex-onset 프레임에 의존하거나, CNN/3D CNN으로 국소적인 시간 수용영역만 확보하는 경향이 강했다. 또한 그래프 기반 방법은 얼굴 ROI 간 관계를 정적 adjacency matrix로 두거나 AU(AU-guided)를 느슨하게 결합해, 근육 활성에 따라 변하는 상호작용의 시간적 정렬과 동적 진화를 충분히 반영하지 못했다. 결과적으로 공간(어디)과 시간(언제)을 독립적으로 학습·후기 융합하는 구조가 많아 데이터셋 간 일반화와 해석가능성에서 한계를 보였다.

- **Core Contribution**: 이 논문은 STAG(Spatio-Temporal Evolving Structural Representation of Action Units for Micro-expression RecoGnition)이라는 단일 프레임워크로 ROI의 동적 연결성과 AU 기반 근육 정보를 결합해, 공간-시간을 함께 학습하도록 설계했다. STAG은 motion flow(광학흐름)와 adaptive facial connectivity(적응형 얼굴 연결)를 동시에 모델링하며, AU-guided dynamic connectivity로 근육 활성 패턴에 따라 얼굴 영역 간 상호작용이 변하도록 한다. 또한 E-GAT 기반 공간 추론과 transformer 기반 전체 시퀀스 시간 모델링을 bidirectional cross-attention으로 상호 정교화해 “어디/언제”를 통합한다.

- **Technical Challenges**: 핵심 난제는 (1) apex에 치우치지 않으면서 미세한 inter-frame 동역학을 안정적으로 포착하고, (2) 얼굴 ROI 간 관계를 정적 지오메트리 프라이어 대신 근육 활성에 맞춰 동적으로 구성하는 동시에, (3) 공간 그래프와 시간 시퀀스를 서로 다른 모듈이 따로 최적화하지 않도록 결합하는 것이다. 논문은 magnitude-based selection과 temporal attention으로 판별성 높은 optical flow를 뽑고, E-GAT로 구조화된 공간 추론을 수행한 뒤 transformer encoder로 미세 타이밍을 학습한다. 이어 bidirectional cross-attention으로 공간-시간 특징을 상호 보정하며, AU-guided 동적 그래프 생성과 temporally smoothed adjacency 업데이트로 연결의 시간적 연속성을 유지한다.

- **Empirical Impact**: STAG은 CASME II, 4DME, DFME, NaME, SAMM, SMIC-HS 등 6개 벤치마크에서 LOSO 및 K-Fold 변형 프로토콜로 평가되며, cross-dataset robustness와 일반화 성능을 향상시키는 것으로 보고된다. 특히 해석가능성 관점에서 AU-guided dynamic connectivity와 상호 정교화 구조가 의미적 일관성(semantic consistency)과 explainable micro-expression recognition에 유리하다고 제시한다. 또한 focal loss 최적화와 효율적인 설계가 결합되어 computational efficiency까지 함께 개선되는 결과를 보였다.



### TextDS: Parameter-Efficient Representation Alignment for Scene Text Detection under Distribution Shifts (https://arxiv.org/abs/2606.28077)
Comments:
          Accepted by ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 장면 텍스트 검출은 픽셀 집계 기반(확률맵·커널·방향장 후 이진화/연결요소/region growing)과 명시적 구조 모델링 기반(윤곽·경계·구성요소 관계를 직접 예측)으로 크게 나뉜다. 그런데 대부분은 비교적 이상적인 촬영 조건에서 학습되며, SynthText 같은 대규모 텍스트 특화 pretraining에 의존해 분포 변화 및 영상 열화 조건에 대한 평가는 제한적이었다. 그 결과 도메인 변화(데이터 소스·언어/폰트·스타일 변화)와 비/안개/저조도/과노출/저해상도 같은 영상 열화가 동시 발생하는 상황에서의 견고성 검증과 최적화가 부족했다.

- **Core Contribution**: 이 논문은 TextDS라는 효율적인 장면 텍스트 검출 프레임워크를 제안하며, 학습 데이터 분포가 바뀌어도 안정적으로 동작하도록 설계했다. 핵심은 큰 scene-text-specific pretraining 없이도, 데이터 효율성과 파라미터 효율성을 동시에 달성하는 듀얼 인코더(구조/의미의 상보성)와 적응·융합 메커니즘을 결합한 점이다. 또한 비(雨)·안개·저조도·과노출·저해상도에 대한 adverse-condition 데이터셋을 만들어, 분포 변화의 실제 평가 공백을 메웠다.

- **Technical Challenges**: TextDS가 마주한 기술 과제는 (1) 텍스트 특화 대규모 pretraining 없이도 필요한 변별 특징을 만들고, (2) 도메인 변화·영상 열화로 인해 픽셀 분리성과 기하 단서가 무너질 때 특징 적응을 안정적으로 수행하며, (3) 두 인코더의 정보를 효과적으로 융합하는 것이다. 이를 위해 SAM2와 DINOv3를 각각 구조 프라이어(SAM2)와 도메인 견고 의미(DINOv3)로 분업해 듀얼 브랜치를 구성하고, frozen 백본은 유지한 채 SWLoRA(Step-wise LoRA)로 블록 단위 저랭크 적응을 하되 cosine similarity 기반 dynamic early-exit으로 불필요한 반복을 줄인다. 마지막으로 CSF(Common Subspace Fusion)로 공유된 공통 부분공간에는 융합을 수행하고, DINOv3의 shift-robust 정보는 직교 여공간으로 보존해 상호 간섭을 최소화한다.

- **Empirical Impact**: 실험에서 TextDS는 CTW-1500/Total-Text/MLT에서 precision·recall·F-measure 모두 비교 방법 대비 경쟁적(특히 in-domain 성능 우위) 결과를 보였고, SynthText pretraining 없이도 4.9M trainable parameters로 성능과 효율(FPS 44.1)을 함께 확보했다. 또한 MLT에서 다른 도메인(CTW-1500·Total-Text)으로 일반화할 때에도, in-domain 최상 성능과 함께 도메인 제너럴라이제이션 F-measure가 유의미하게 높게 나타나 분포 변화 견고성이 확인됐다. 더 나아가 비/안개/저조도/과노출/저해상도 adverse-condition 설정으로 평가 공백을 보완하며, 영상 열화 조건에서도 성능 저하를 상대적으로 억제하는 실증적 의미를 제시했다.



### ReScene: Structured Indoor Scene Reconstruction from Multi-View Captures (https://arxiv.org/abs/2606.28060)
- **Prior Approaches**: 기존 자동 실내 장면 구성은 (1) 전용 캡처 하드웨어에 의존하거나 (2) 단일 뷰 중심으로 물체를 복원해 단서가 불완전해지고 (3) 조립(assembly) 단계는 대충 넘어가 geometrically 그럴듯하지만 물리적으로는 충돌·부유·경계 이탈 같은 문제가 자주 발생했다. 또한 장면을 구성하는 view selection과 relation 추론이 느슨하게 분리돼 교차 뷰 정보가 일관되게 융합되지 않는 한계가 있었다.

- **Core Contribution**: 이 논문은 병목을 single-object reconstruction이 아니라 교차 뷰 relation fusion과 물리적으로 그럴듯한 장면 조립에 있다고 정의하고, 이를 multi-view geometry를 파이프라인 전반의 통합 prior로 삼아 해결하는 ReScene을 제안한다. ReScene은 HierView로 “mask 면적” 대신 semantic consistency와 3D coverage completeness를 기준으로 재구성에 유리한 뷰를 고르고, Relation-Aware Assembly로 다중 프레임 VLM 관계를 confidence-weighted scene graph로 융합해 staged attachment로 물리 안정성을 확보한다.

- **Technical Challenges**: 핵심 기술 난점은 (a) 부분 관측에서 mask가 커도 실제 물체 전체가 포착되지 않는 view 선택 오류, (b) 프레임마다 달라지는 VLM 관계 예측을 기하·범주·룸-shell 제약과 함께 일관된 그래프로 만들기, (c) 그래프의 이산 제약을 연속적인 pose 조정으로 컴파일해 충돌을 줄이면서 상위-하위 객체 정합성을 유지하는 것이다. 논문은 HierView의 계층적 필터링(임계값+CLIP 기반 semantic 정합+투영 기반 3D extent 가시성)과, 장면 그래프 에지별 신뢰도 가중 에너지의 룰 기반 dispatch solver 및 위상순(topological order) 전파로 이러한 문제를 완화한다.

- **Empirical Impact**: ScanNet의 30개 실내 장면에서 ReScene은 Chamfer Distance 17% 감소, LPIPS 26% 감소를 포함해 geometry·렌더링·지각 품질 전반에서 최신 강점 baseline 대비 우위를 보이며, 다중 뷰 방법 중 최대 10x 빠른 속도를 보고한다. 또한 ReScene이 생성한 장면을 기반으로 embodied visual question answering 데이터셋을 만들고 Qwen-VL fine-tuning이 일부 공간 추론 과제에서 strong closed-source 모델과 경쟁 수준의 성능을 달성해, 시뮬레이션-준비 장면이 다운스트림 EAI에 실제로 유효함을 시사한다.



### AirGroundBench: Probing Spatial Intelligence in Multimodal Large Models under Heterogeneous Multi-View Embodied Collaboration (https://arxiv.org/abs/2606.28049)
- **Prior Approaches**: 기존 평가는 대부분 단일 이미지 기반 VQA 중심이거나, 동종 카메라/단일 플랫폼의 정형된 멀티뷰만 다뤄 서로 다른 관측 플랫폼 간 기하 일관성을 직접 압박하지 못했다. 협동 인지·임베디드 벤치마크도 주로 지각/통신 효율(예: fusion·bandwidth)이나 단일 프레임의 성능을 보려는 경향이 강해, scale mismatch·비대칭 가림·좌표계 불일치 같은 ‘기하 일관성’ 결함을 원인 수준으로 진단하기 어렵다. 또한 정적 추론 성격의 테스트가 많아, 이런 불일치가 폐루프 의사결정(VLN)에서 어떻게 누적 실패로 이어지는지 검증 공백이 있었다.

- **Core Contribution**: 이 논문은 드론(UAV)과 지상 로봇(UGV)이 같은 시점에 주는 이질적 듀얼뷰를 기반으로, 기하 일관성 중심의 진단형 벤치마크 AirGroundBench를 제안한다. 11개 고충실 시뮬레이션에서 동기화된 1,021쌍 관측을 수집해 4지선다 VQA 약 62k, 폐루프 vision-language navigation 115 에피소드를 제공한다. 특히 cross-view object identity와 metric 2D/3D bounding box 같은 기하 검증용 주석을 함께 제공해, 정답 맞추기뿐 아니라 기하 정합이 어디서 깨지는지 분석할 수 있게 했다.

- **Technical Challenges**: 핵심 난제는 서로 다른 관측 높이·시야·스케일 때문에 같은 대상을 ‘동일 엔티티로 바인딩’하고, metric 관계(거리·방향·상대 자세)를 좌표계 전환 속에서도 일관되게 유지해야 한다는 점이다. 이를 위해 UAV 카메라 포즈를 UGV 로컬 NED 기준의 preset 변환으로 구성해 baseline/시점 불일치를 통제하고, 데이터 수집 시점에 물체 ID와 2D/3D 박스 및 상대 포즈 메타데이터를 구조화해 verifiable한 감독신호를 제공한다. 결과적으로 scale alignment·object matching·viewpoint transformation 같은 단계별 실패를 분해해 측정할 수 있도록 설계되었다.

- **Empirical Impact**: 13개 대표 MLLM을 평가한 결과, 지각(perception)은 상대적으로 강하지만 cross-view alignment와 변환/추론(transformational reasoning)에서 성능이 크게 떨어지고, 그 결함이 폐루프 VLN의 누적 오류로 이어져 성공률이 특히 낮게 나타났다. 듀얼뷰 입력은 단일뷰(UAV-only/UGV-only)보다 모든 모델에서 개선을 보였지만, 사람 성능 대비 격차는 지속돼 ‘기하 일관성’이 현 임베디드 MLLM의 근본 제한임을 시사한다. 또한 도심/야생 환경에 따라 성능 분산이 달랐고, 개활지처럼 안정적인 기하 앵커가 부족할수록 정합이 더 어려워짐이 드러나, 향후 기하-근거 표현과 정합 학습의 필요성을 구체화한다.



### Mind the Gap: Quantifying the Domain Gap in Cross-Sensor Diffusion Super-Resolution (https://arxiv.org/abs/2606.28039)
Comments:
          26th International Conference on Computational Science

- **Prior Approaches**: 위성 초해상도(SR)는 센서마다 ‘진짜’ 저해상도-고해상도 쌍이 없어, 보통 bicubic 같은 방식으로 인트라센서 저하(synthetic degradation)를 만든 뒤 학습한다. diffusion 기반 SR은 질감 복원과 전역 일관성에서 강점을 보이지만, 이런 합성 저하 중심 학습은 센서의 물리·스펙트럼·복사 특성을 충분히 반영하지 못한다. 그 결과 cross-sensor 환경에서 성능이 흔들리거나 붕괴하는 문제는 경험적으로 알려져 있으나, 현대 diffusion SR 전반에 대한 체계적 정량 분석은 부족했다.

- **Core Contribution**: 이 논문은 synthetic-to-real mismatch가 modern diffusion-based SR 성능에 미치는 영향을 처음으로 체계적으로 연구한다. Sentinel-2와 PlanetScope를 기하·시간 정렬한 대규모 paired dataset을 구축해, 합성 학습→실제 교차센서 평가의 일반화 격차를 통제된 조건에서 측정한다. 또한 Sentinel-2 self-supervised 특징 기반의 도메인 적응 지각 거리 LPIPS-Sat(LPIPSSat)를 제안해 위성 영상에 더 맞는 평가 프레임을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 센서 간 분광 대역 불일치와 복사·기하 변동 때문에 ‘유효한 SR’ 학습 목표를 구성하기 어렵다는 점이다. 논문은 서로 겹치지 않는 대역을 그대로 쓰지 않고 물리적으로 대응되는 6개 밴드를 선택해, 모델이 존재하지 않는 스펙트럼 매핑(예: Blue를 Red로)까지 ‘환각’하지 않도록 설계한다. 실험적으로는 세 가지 구성(합성 기준, 합성→실제 도메인 격차 측정, 실제→실제 직접 매핑)을 나눠 diffusion 계열 여러 아키텍처의 실패 양상을 일관되게 비교한다.

- **Empirical Impact**: 결과는 두 가지로 요약된다: 합성 저하로 학습한 모델은 실제 교차센서 쌍에서 급격히 성능이 무너진다(C2가 기준선보다 악화, LPIPS/LPIPSSat도 악화). 반대로 실제 cross-sensor 데이터로 학습한 모델은 지각 지표가 일부 회복되지만 최적화가 불안정하고 센서 다양성 적응이 완전히 되지 않아 합성 설정 대비 격차가 지속된다(C1 vs C3). 또한 burned-area change delineation 같은 다운스트림 세그멘테이션에서, 합성 학습 모델은 ‘일관된(작지만 유의미한)’ 이득을 주는 반면 실제 학습 모델은 SR과 도메인 적응이 얽히며 파괴적 아티팩트를 유발해 실용성이 떨어질 수 있음을 보여준다.



### EMOSH: Expressive Motion and Shape Disentanglement for Human Animation (https://arxiv.org/abs/2606.28026)
Comments:
          Accepted to ECCV 2026, Project Page: this https URL

- **Prior Approaches**: 기존 인간 애니메이션은 GAN·warping에서 확산 모델(ControlNet 계열)로 넘어가며 품질을 끌어올렸지만, 2D 포즈 조건 기반에서는 motion-shape entanglement 문제가 반복됩니다. 결과적으로 구동자의 신체 비율이 참조 인물의 신체 형태로 새어 들어가거나, 재매핑이 어렵습니다. 3D priors(SMPL 등)를 쓰는 방식은 기하적 분리는 돕지만 얼굴 표정·복잡한 제스처를 잘 재현하지 못해 애니메이션이 딱딱해지는 한계가 있습니다.

- **Core Contribution**: EMOSH는 Expressive Human Model(EHM)을 핵심 제어 표현으로 제안해, shape와 pose(및 표정)를 명시적으로 분리함으로써 body shape leakage를 근본적으로 줄입니다. 또한 Coarse-to-Fine Hybrid Motion Injection으로 눈 깜빡임 같은 고주파 표현과 제스처 제어를 더 촘촘히 제공합니다. 마지막으로 Spatially-Aligned Conditioning으로 학습-추론 도메인 갭을 완화해 신원(identity)과 장면 구도를 안정적으로 유지합니다.

- **Technical Challenges**: 첫째, 단안 비디오에서 얼굴·손·전신을 포함한 EHM 파라미터를 안정적으로 추정해야 하는데, EMOSH는 Confidence-Aware Validity Gating을 포함한 unified joint optimization으로 추정 불안정/가림 상황을 다룹니다. 둘째, 메쉬 렌더링 조건만으로는 고주파 디테일이 부족해 이를 보완하기 위해 세맨틱 컬러 맵에 2D keypoint를 함께 주입하는 Coarse-to-Fine Hybrid Motion Injection을 설계합니다. 셋째, long video에서 누적 오차가 identity drift로 이어지는 문제는 spatially-aligned latent를 “브리지/anchor”로 반복 삽입해 제어합니다.

- **Empirical Impact**: 실험에서 EMOSH는 self-driven과 cross-driven 두 설정 모두에서 기존 방법을 전반적으로 능가하며, 표정·제스처의 생동감은 높이면서 신체 형태 분리까지 유지하는 성능을 보였습니다. cross-driven에서는 Identity Preservation Score(IPS)에서 가장 높은 값을 기록해 참조 인물 고유성을 더 잘 보존함을 확인했습니다. 정성 결과와 사용자 연구에서도 EMOSH가 영상 품질, 모션·표정 정확도, identity 보존 측면에서 선호도가 가장 높게 나타나 의미 있는 SOTA급 개선으로 평가됩니다.



### TempAct: Advancing Temporal Plausibility in Autoregressive Video Generation via Planner-Executor RL (https://arxiv.org/abs/2606.28016)
- **Prior Approaches**: 자기회귀(AR) 비디오 diffusion은 청크(chunk) 단위로 스트리밍 생성해 지연을 줄이지만, 글로벌 프롬프트 하나로는 각 청크가 *어떤 하위 이벤트*를 담당해야 하는지가 모호해진다. step-wise 프롬프트로 바꾸면 사건 순서는 개선될 수 있으나, 프롬프트 전환 시점에서 이전 행동을 계속하거나 의미가 섞이고 오류가 누적되기 쉬워진다. SFT나 distillation은 폭주 편향과 노이즈/분포 매칭 중심 최적화 때문에 “행동 순서”와 “프롬프트 전환 정합성”을 직접 강제하기 어렵다는 한계가 있다.

- **Core Contribution**: TempAct는 planner–executor 강화학습으로 AR 비디오의 시간적 그럴듯함(temporal plausibility)을 목표로, (1) 시간 분해가 옳은지와 (2) 전환 시점에 실제 실행이 맞는지를 함께 학습한다. LLM planner가 span-aware 단계 프롬프트를 만들고, AR diffusion executor가 생성된 시각 이력(autoregessive visual context) 아래에서 그 단계 프롬프트를 따르도록 학습한다. 특히 LLM을 고정 전처리로 쓰지 않고, 생성 궤적에서 planner와 executor를 같이 업데이트해 “어떤 분해가 실행 가능하고 올바른지”를 직접 찾도록 설계했다.

- **Technical Challenges**: 핵심 기술 과제는 긴 시간 범위에서 프롬프트 전환의 크레딧(공헌도)을 올바르게 배분하는 것이다. TempAct는 계층적 group exploration을 도입해, planning group(서로 다른 시간 분해 계획 비교) 안에 execution group(같은 공유 컨텍스트에서의 여러 continuation 비교)을 중첩시키고, planner에는 계획-수준 보상을, executor에는 전환 직후 첫 chunk에서의 local step-following 보상을 집중한다. 또한 plan-quality와 VLM 기반 temporal-following을 결합한 계층형 보상, transition-span에 맞춘 Flow-GRPO/GSPO 업데이트, 그리고 KL 제약과 미적 정규화로 실행 드리프트를 줄인다.

- **Empirical Impact**: Self-Forcing과 LongLive 백본에 대해 TempAct는 temporal-following score와 VLM 평가, 인간 평가 전반에서 시간적 일관성을 개선하면서도 전체 영상 품질은 유지하는 성과를 보였다. 즉 “청크 스트리밍의 낮은 지연”을 해치지 않으면서, 사건 순서가 무너지는 temporal confusion을 강화학습으로 직접 완화한 셈이다. 이는 스트리밍 AR 비디오에서 단순 샘플링 효율 개선을 넘어, instruction을 시간적으로 분해하고 전환을 정확히 실행하는 연구 방향에 실질적인 근거를 제공한다.



### Curriculum-guided Change Detection Training: Toward Accurate Serac Fall Monitoring (https://arxiv.org/abs/2606.28012)
Comments:
          Preprint, 11 pages, 5 figures

- **Prior Approaches**: 기존 변화 감지는 주로 siamese·transformer 계열 아키텍처 개선에 초점을 맞췄고, 학습은 균일 샘플링을 기본으로 삼아 모든 데이터가 최적화에 동일한 영향을 준다고 가정하는 경우가 많았다. 또한 semi-supervised 및 consistency regularization 같은 방법이 주목받았지만, 더 신뢰도 높은 샘플을 먼저 쓰는 ‘커리큘럼’ 관점은 거의 다뤄지지 않았다. 변화 감지에서는 조명 변화·그림자 같은 pseudo-change 때문에 샘플 난이도가 크게 달라질 수 있어, 균일 샘플링의 한계가 더 두드러질 수 있다.

- **Core Contribution**: 이 논문은 변화 감지에 특화된 최초의 curriculum learning 프레임워크를 제안한다. 핵심은 데이터 난이도 점수를 정의하고, 쉬운 샘플부터 어려운 샘플을 점진적으로 넣어 coarse-to-fine 방식으로 더 견고한 표현을 학습하게 하는 것이다. 특히 난이도 측정에 대해 SSIM과 SAG 두 가지 대리 지표를 제시하며, 이 프레임워크는 대부분의 CD 아키텍처와 직교적으로 결합 가능하다고 주장한다.

- **Technical Challenges**: 커리큘럼에 필요한 ‘난이도 측정’은 변화 감지에서 특히 비자명하다. SSIM은 이미지 쌍의 구조/외형 차이를 잘 반영하지만 실제 변화까지 함께 크게 점수화해 오히려 어려운 학습 순서를 만들 수 있고, SAG는 태양 위치 차이(Solar Angular Gap)로 조명·그림자 요인을 물리적으로 모델링해 라벨 없이도 난이도를 주지만 날씨(구름 등) 효과는 놓칠 수 있다. 저자들은 이 상보적 약점을 각각의 지표로 다루되, baby step·linear pacing과 결합해 수렴 기반 도입 전략을 통해 안정적으로 어려운 샘플을 통합하도록 설계했다.

- **Empirical Impact**: SeracFallDet 벤치마크에서 pixel-level(세그멘테이션)과 object-level(RT-DETR 기반 이벤트 검출) 모두에 대해 균일 샘플링 대비 일관된 개선이 보고됐다. 특히 객체 기반은 +8.3% F1, 픽셀 기반은 +3.3% F1로 테스트에서도 성능 저하 없이 강건성이 향상됐으며, 개선의 주요 원인은 pseudo-change로 인한 false positive 감소로 해석된다. 또한 시각적으로는 전통 학습 대비 오탐이 줄고 경계가 더 선명해지는 등 localization 품질이 좋아졌다고 정리했다.



### HumanMoveVQA: Can Video MLLMs reason about human movement in videos? (https://arxiv.org/abs/2606.27999)
- **Prior Approaches**: 기존 Video MLLM 연구는 영상-텍스트 캡션의 고수준 이벤트(예: “뛰다”)에 의존하는 경우가 많아, 사람의 전역 궤적(translation)과 자세 회전(orientation)이 시간에 따라 어떻게 변하는지 추론하기 어렵습니다. 관련 벤치마크와 모션 기반 접근들은 관절 단위의 fine-grained 움직임이나 장면 중심의 시공간 관계에 치우쳐 있어, 장기 구간에서의 연속적 trajectory·orientation reasoning을 제대로 시험하지 못한다는 한계가 지적됩니다.

- **Core Contribution**: 이 논문은 사람의 전역 궤적과 방향 변화를 exocentric(3인칭) 관점에서 평가하는 최초의 포괄 벤치마크 HumanMoveVQA를 제안합니다. 첫 프레임 기준의 world coordinate system을 고정해 이동과 회전이 동일 기준선에서 유지되도록 하고, 7개 추론 범주(모션 집계·순서·궤적 수준 추론 등)에 대한 다지선다 QA를 생성합니다.

- **Technical Challenges**: 핵심 기술 난제는 카메라/관측의 ego-motion을 제거하면서도, 연속 동작에서 사건(event)들의 시간 순서를 정확히 연결하는 데 있습니다. 이를 위해 2D 영상을 사람 메쉬 재구성 기반 3D motion track으로 lifting한 뒤, translation·rotation을 Spatial Codes로 이산화하고(잡음 완충), 다인 장면에서는 appearance tag로 identity grounding을 수행하며, 답 오답지(distractor)는 ‘언어적으로 그럴듯하지만 기하학적으로 불일치’하도록 설계해 언어 편향을 차단합니다.

- **Empirical Impact**: 실험 결과, Gemini-3-Flash 같은 최신 closed-source 모델도 zero-shot에서 궤적·방향 추론 과제 점수가 낮고(기회점 대비 큰 격차), 특히 수치/순서/장기 궤적 범주에서 한계가 두드러졌습니다. 반면 Qwen3-VL 8B를 HumanMoveVQA의 world-consistent 감독으로 fine-tuning하면 aggregate score가 약 3배 가까이 향상되어(37.9) 해당 능력이 학습 가능한 문제임을 보여줍니다. 또한 cross-dataset 일반화에서 ordering이 가장 어렵다는 점, 카테고리별 단독 학습은 다른 축 성능을 흔들 수 있다는 점을 통해 벤치마크가 모델의 실제 추론을 정밀하게 드러낸다는 의미가 있습니다.



### Latent Visual Diffusion Reasoning with Monte Carlo Tree Search (https://arxiv.org/abs/2606.27988)
Comments:
          Accepted to ECCV 2026. Project page: this https URL

- **Prior Approaches**: 기존 fine-grained skill assessment(세밀 기술 평가)는 action quality assessment에서 점수 회귀나 랭킹 예측 중심으로 발전했지만, 대부분 end-to-end 모델이라 판단 근거(중간 추론)를 설명하지 못하는 black box 한계가 컸습니다. 규칙 기반 심볼 추출이나 rubric 기반 분절은 해석 가능성을 주지만, 사람이 정의한 룰에 의존해 데이터로부터 단계적 추론을 학습하기 어렵습니다. attention 기반 설명 방법도 기여 구간은 보여주지만, 인간처럼 이어지는 순차적(step-by-step) 시각 추론 경로를 재현하긴 힘들었습니다.

- **Core Contribution**: 이 논문은 Latent Visual Diffusion Reasoning(LVDR)라는 프레임워크로, 최종 점수 예측과 함께 “중간 시각 추론 궤적”을 생성·시각화하는 것을 핵심으로 제안합니다. LVDR은 비디오의 reasoning을 latent space에서의 progressive denoising(노이즈 제거)로 모델링하고, 각 단계의 keypoint attention 패턴이 어떻게 수렴하는지 보여줍니다. 여기에 keypoint-guided Monte Carlo Tree Search(MCTS)를 결합해 최종 판단에 가장 중요한 시각적 근거 시퀀스를 추출합니다.

- **Technical Challenges**: 기술 평가에서 필요한 프레임-단위 추론 라벨은 사실상 구하기 어려워, 이를 직접 감독할 수 없다는 문제가 있었습니다. LVDR은 ground-truth semantics(라벨/전문가 코멘터리)를 이용해 diffusion이 latent reasoning 상태의 “경로(path)”를 학습하도록 설계했고, 비디오 시작은 noisy initialization으로 두어 시간에 따라 목표 의미 분포로 점진적으로 정제되게 했습니다. 또한 해석 가능성을 위해 MCTS에서 UCT 기반 탐색과 키포인트/관절(스포츠)·도구부(수술) 상태 표현을 사용해, 각 단계 후보 경로의 보상을 diffusion 임베딩 유사도로 근사하도록 구성했습니다.

- **Empirical Impact**: EgoExo4D, JIGSAWS, FitnessAQA, Cataract-101의 4개 데이터셋(스포츠와 수술 도메인 포함)에서 LVDR은 기존 방법과 even 일부 대형 vision–language 모델 대비 경쟁력 있거나 우수한 정량 성능을 보였습니다. 특히 MCTS가 선택한 keypoint를 마스킹하면 성능이 크게 하락하고, 선택되지 않은 키포인트 마스킹은 영향이 작아 “중요 시각 단서”를 제대로 집어낸다는 근거를 제공합니다. 전문가 사용자 연구에서도 MCTS 기반 추론의 올바름이 대체로 인간 판단과 일치(예: Correct 86%)했으며, 확장된 시각 추론 궤적 제공을 통해 고위험 분야에서의 투명성과 신뢰성 확보에 의미가 큽니다.



### Parallel Rollout Approximation for Pixel-Space Autoregressive Image Generation (https://arxiv.org/abs/2606.27978)
- **Prior Approaches**: 픽셀 공간에서 autoregressive(AR) 생성은 패치를 연속 토큰으로 두고 다음 패치를 예측하지만, 고차원 픽셀 패치 예측의 단일 스텝 오차와 teacher-forcing으로 인한 train–inference gap이 함께 커져 오류가 누적되는 문제가 컸다. 입력 측 완화로는 input noise injection, 출력 측 완화로는 xx-prediction 같은 파라미터화가 쓰였지만, diffusion 기반(pixel-space diffusion) 성능 격차를 충분히 줄이지 못했다. 한편 정확한 rollout 학습은 더 잘 맞추지만, 연속 토큰 AR에선 순차 샘플링 비용이 너무 커 실용성이 떨어졌다.

- **Core Contribution**: 이 논문은 Parallel Rollout Approximation(PRA)로 두 병목을 동시에 겨냥한다. PRA는 AR이 고차원 픽셀 패치를 직접 생성하지 않고, 저차원 중간 상태를 생성한 뒤 pixel decoder로 다시 픽셀 패치(=pixel-in, pixel-out 인터페이스)를 만들어 넣는다. 또한 추론 때처럼 생성된 입력이 들어오는 상황을 학습에서도 근사하기 위해, 중간 상태→픽셀 디코딩 경로를 그대로 사용해 inference-like 픽셀 입력을 포지션별로 병렬 구성한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 고차원 픽셀 토큰 생성 난이도를 낮추면서도, (2) 학습 시 조건이 추론 시 조건과 최대한 비슷해지도록 만드는 것이다. 저차원 중간 상태는 엔드투엔드로 학습하되, causal prefix 표현을 반영한 중간 목표를 만들고 토큰 마스킹으로 현재 토큰만 의존하지 않게 제어한다. 입력 측 근사는 실제 순차 rollout 대신, 각 포지션에서 perturbed intermediate state를 디코딩해 stop-gradient로 학습 입력을 만들고, 이때 필요한 AR forward는 병렬로 처리해 계산비용을 억제한다.

- **Empirical Impact**: ImageNet-1K 클래스 조건 생성(256×256)에서 PRA-S(135M)는 FID 2.58로, 이전 픽셀 공간 AR 최강(1B급)의 FID 3.60을 앞질렀다. 더 큰 PRA-L(511M)은 FID 1.94까지 개선해 픽셀 공간 AR 모델 중 새로운 state of the art를 기록했다. 추가로 생성뿐 아니라 ImageNet 분류 probing 정확도도 기존 AR·diffusion 베이스라인보다 높게 나타나, 픽셀 공간 엔드투엔드 AR이 생성+시각 이해를 함께 끌어올릴 가능성을 시사한다.



### ProMSA:Progressive Multimodal Search Agents for Knowledge-Based Visual Question Answering (https://arxiv.org/abs/2606.27974)
- **Prior Approaches**: 기존 KB-VQA는 대체로 고정된 retrieval-then-generate 파이프라인을 따른다. 이미지(또는 텍스트)에서 top-k를 한 번 뽑아 프롬프트에 넣고 정답을 생성하며, 추론 중에는 검색 정책이나 깊이를 바꾸기 어렵다. 또한 초기 검색이 잘못되면 이후 라운드가 그 오답 근거에 더해져 실패를 교정하기 힘들고, 멀티홉 증거 사슬도 정적 주입 방식에서 약해진다.

- **Core Contribution**: ProMSA는 KB-VQA를 “예산을 가진 점진적 검색-추론” 문제로 재구성해, 추론 도중에 img_search/text_search/stop을 반복 선택하는 progressive multimodal search agent를 제안한다. 에이전트는 중복 제거(deduplication)와 함께, 엔티티 식별이 불확실할 땐 재검색(이미지), 속성/증거가 비면 쿼리 재작성 후 텍스트 검색을 수행하며, 근거가 충분하면 stop해 답을 생성한다. 이를 통해 retrieval과 reasoning을 단일 궤적에서 end-to-end로 결합한다.

- **Technical Challenges**: 핵심 난제는 (1) 툴 호출 형식과 인자 구조를 먼저 “실행 가능”하게 학습해야 RL 탐색 불안정(잘못된 호출로 보상 0)이 생기지 않는 점, (2) 라운드 수(툴 상호작용 깊이)와 생성 길이가 다른 RL 업데이트 스케일을 유발해 학습이 흔들릴 수 있다는 점이다. 저자는 rejection-sampling SFT로 콜 형식을 워밍업한 뒤, sequence-level RL에서 generation length뿐 아니라 tool-interaction depth까지 반영하는 TN-GSPO를 도입해 업데이트 편향을 줄이고 안정적인 검색 정책을 학습한다.

- **Empirical Impact**: E-VQA와 InfoSeek에서 ProMSA는 강한 RAG 및 에이전트 베이스라인을 일관되게 능가하며 retrieval 정확도와 end-to-end 정확도를 함께 끌어올린다. zero-shot MLLM 대비 긴 꼬리 엔티티/미세 속성에 유리한 외부 근거 활용 능력을 보여주고, 검색 에이전트 중에서도 실패 교정 메커니즘이 약한 기존 접근 대비 성능 우위를 확인한다. 또한 OK-VQA에서도 개선이 관찰돼, 학습된 도구 사용 정책이 훈련 벤치마크를 넘어 일반화될 가능성을 시사한다.



### Directing the World: Fast Autoregressive Video Generation with Compositional Human-Camera Contro (https://arxiv.org/abs/2606.27964)
- **Prior Approaches**: 기존 비디오 기반 월드 모델은 장기 일관성·시점 변화 같은 장면 레벨 진화를 주로 다루며, 인간을 ‘보이는 요소’에 가깝게 취급하는 경우가 많았다. 사람 모션을 제어하려는 pose-또는 SMPL 기반 생성도 존재하지만 대개 짧은 편집/애니메이션에 최적화돼 장시간 AR 롤아웃에서 identity drift, jitter, 깜빡임 문제가 생긴다. 또한 카메라 제어 연구는 scene identity를 유지해야 해 시간 불일치에 민감하지만, 블록 단위 AR 생성과의 정합성이 부족해 이질적 제어가 결합될 때 품질-제어력 트레이드오프가 두드러졌다.

- **Core Contribution**: 이 논문은 Directing the World로, 단일 autoregressive 비디오 prior를 유지한 채 인간 모션과 카메라-trajectory를 조합 가능한 방식으로 ‘유도(direct)’하는 프레임워크를 제안한다. 핵심은 control learning을 decouple한 뒤 통합(integrate)해 이질적 제어 간 간섭을 줄이면서도 장기 롤아웃의 세계 메모리를 보존하는 데 있다. Fast-Slow Memory, timestep-guided Dynamic Projection, Motion-CFG, causal camera-control까지 함께 묶어 controllability와 시각 품질을 동시에 노린다.

- **Technical Challenges**: 가장 큰 문제는 (1) AR 생성에서 모션/카메라 조건이 확률 분포를 깨서 잡음 타이밍마다 과·부적절하게 작용하고, (2) 블록 단위 생성 구조와 글로벌 조건 인코딩이 충돌하며, (3) 전체를 한꺼번에 fine-tuning하면 제어-품질 얽힘이 생긴다는 점이다. 논문은 SMPL을 VAE latent 공간에 정렬하고 timestep-guided projection으로 denoising 단계별 조건 강도를 조절해 배경 flickering·인체 jitter를 완화한다. 아울러 Motion-CFG에서 텍스트 dominance를 억제하고(조건을 SMPL 중심으로), camera는 Plücker 좌표 기반 인코더-주입을 분리해 블록 로컬 주입으로 causal 정합성을 맞추며, Fast-Slow Memory로 slow/fast 파라미터 그룹 학습률을 나눠 장기 prior를 안정적으로 유지한다.

- **Empirical Impact**: 대규모 동기화 데이터셋(동영상·텍스트·human-motion·카메라-trajectory 어노테이션)을 구축하고, motion-only·camera-only·joint motion-camera 제어에서 APRIL-AIGC/UltraVideo-Long 같은 벤치마크로 광범위하게 검증한다. 결과는 장기(15초~2분) 생성에서 안정적인 long-horizon 생성과 높은 시각 품질, 입력 모션/카메라에 대한 정밀한 제어가 함께 관측된다고 보고된다. 특히 다중 인물/복합 제어에서도 모션 정합성과 시간적 일관성을 유지하는 방향을 제시해, embodied agent·가상 탐색·디지털 콘텐츠 생성에서 ‘제어 가능한 세계 시뮬레이션’의 실용성을 높일 것으로 기대된다.



### Understanding How MLLMs Describe Artworks Using Token Activation Maps (https://arxiv.org/abs/2606.27947)
Comments:
          Accepted at PRESTIGE workshop at ICPR 2026

- **Prior Approaches**: 기존 비전-언어 모델 연구는 artwork 검색·질문응답 같은 성능 위주로 발전했지만, 생성된 캡션의 “근거가 어디인지”는 불명확하다는 한계가 있었다. Grad-CAM, attention rollout, TCAV 같은 XAI는 주로 분류 결과에 1장의 전역 설명을 제공해, 토큰 단위로 추론이 어떻게 분리되는지 파악하기 어렵다.

- **Core Contribution**: 이 논문은 Token Activation Map(TAM)을 사용해 생성되는 각 토큰이 그림의 어떤 시각 영역에 근거하는지 토큰별 히트맵으로 분해한다. 또한 캡션을 common visual objects, style descriptors, metadata, iconographic tokens, affective expressions의 5개 범주로 나눠, 의미 유형에 따라 시각적 grounding 패턴이 달라지는지 체계적으로 분석한다.

- **Technical Challenges**: 자기회귀 생성에서는 앞선 토큰이 뒤 토큰의 히트맵에 잡음처럼 섞이므로, 단순 활성맵 적용은 심하게 오염된다. TAM은 컨텍스트 토큰의 간섭을 least-squares로 추정해 제거하고 rank Gaussian filter로 잔여 잡음을 줄여 토큰별 “순수 근거”에 가까운 지도(heatmap)를 만든 뒤, 멀티토큰 표현은 평균으로 span map을 구성한다.

- **Empirical Impact**: 결과로 concrete visual object와 iconographic token은 특정 영역에 더 국소화되는 반면, style과 affect는 캔버스 전반에 퍼진 grounding을 보였다. 메타데이터에서는 화가는 약 82% 정확도로 맞추지만 제목은 약 28%로 낮고, 제목 쪽에서 더 많은 환각이 관찰됐다; 또한 TAM 기반 국소화는 SAM 3의 open-vocabulary segmentation과 대체로 일치가 약했지만(대략적인 IoU 수준은 낮음) 단순히 어느 쪽이 항상 우월하지는 않았다. 저자들은 코드·실험 설정·프롬프트·정성 결과를 공개해 문화유산 도메인에서 해석가능성/신뢰성 연구의 실험 재현성을 강화한다.



### Controllable Histopathology Image Synthesis with Training-free Structural Initialization and Textural Modulation (https://arxiv.org/abs/2606.27935)
- **Prior Approaches**: 기존 히스토패톨로지 합성 연구는 GAN이나 diffusion 기반 생성에서 약/비쌍(supervision)으로도 학습은 시도했지만, 진단에 필요한 현실감과 구조-마스크 대응이 깨지기 쉽다는 한계가 있었다. 또한 diffusion의 conditional 제어를 잘하기 위해서는 대개 paired supervision(매칭된 라벨)이 필요하며, unpaired translation을 노리는 cycle-diffusion 계열은 도메인 불일치로 계산 비용과 품질 저하가 동반되었다. 더 나아가 표준 Gaussian 초기화는 구조 정렬을 방해해 생성 결과의 구조 미스얼라인먼트가 발생한다.

- **Core Contribution**: CHIS는 학습을 추가로 하지 않는 training-free 플러그인 형태로, 사전학습된 diffusion 모델의 샘플링 궤적을 구조와 텍스처에 맞게 “유도”한다. 먼저 마스크의 위상(phase)을 Gaussian 잡음의 크기(amplitude)와 주파수 영역에서 결합해 reverse diffusion 시작점을 구조 정렬적으로 만들고, 생성 과정에서는 wavelet 분해 수준에 따라 거친 텍스처와 미세 텍스처를 적응적으로 변조한다. 그 결과 unlabeled 데이터만으로도 prior structural mask와 조직 스타일을 동시에 맞추는 합성이 가능해진다.

- **Technical Challenges**: 핵심 난제는 (1) binary mask 같은 구조 신호를 latent diffusion의 입력/초기 상태에 안정적으로 주입해도, (2) 사전학습 분포와의 차이로 인한 feature shift를 줄이면서, (3) 생성 중 텍스처 디테일까지 구조와 일관되게 유지하는 것이다. CHIS는 FFT 기반 phase fusion으로 초기 latent를 구조-정보를 반영하도록 재구성하고, 마스크를 기준 조직 이미지의 색/텍스처 통계로 “extract-and-fill”해 latent 인코더 분포 변화 문제를 완화한다. 이어 SWT(stationary wavelet transform)로 텍스처를 coarse/fine 성분으로 분리한 뒤, 저주파는 구조 윤곽을 우선 정렬하고 고주파는 과도한 가이드를 줄이는 규칙 기반 가중 변조로 구조-텍스처 결합도를 높인다.

- **Empirical Impact**: 실험에서는 PixCell을 frozen 백본으로 두고 CHIS가 샘플링만 조절하도록 하여, MoNuSAC·Kumar·PanNuke에서 마스크 구조 충실도와 시각적 품질(FID/IS)에서 경쟁 방법을 앞섰다. 특히 SynDiff/CycleDiff가 마스크-이미지 대응을 제대로 제어하지 못한 반면, CHIS는 구조 정렬 초기화와 주파수별 텍스처 변조를 통해 완전지도(full supervised) 모델 NuDiff에 근접하거나 MoNuSAC에서는 초과 성능을 보였다. downstream에서는 합성 이미지를 학습 데이터에 추가했을 때 Hover-Net의 Dice/AJI가 가장 크게 개선되며, 결국 라벨링과 모델 학습 비용을 크게 줄이면서도 인스턴스 분할 성능을 끌어올릴 수 있음을 실증했다.



### Home3D 1.0: A High-Fidelity Image-to-3D Asset Generation System for Interior Design (https://arxiv.org/abs/2606.27923)
Comments:
          18 pages, 10 figures, 2 tables; technical report

- **Prior Approaches**: 기존 image-to-3D는 범주에 맞는 그럴듯한 형태를 만드는 데 초점이 많아, 가구처럼 제품 동일성을 요구하는 영역에는 품질이 부족하다는 지적이 나온다. 특히 메시의 잡음/누락/얇은 부품 파손, 뷰 간 텍스처 불일치와 패브릭·우드·가죽 결 같은 미세 디테일 저하, 그리고 직접 생성한 PBR 맵의 알베도·거칠기·메탈릭·놈 디테일 및 재질 정체성 복원 실패가 문제였다. 또한 부품 분해가 다리·등받이처럼 기능 단위로 이뤄지는 경우가 많아, 디자이너가 실제로 편집하는 ‘재질 단위’ 편집 니즈와 어긋났다.

- **Core Contribution**: Home3D 1.0은 단일 기준 이미지로 ‘실사용 가능한’ 인테리어 가구 3D 자산을 만들기 위해 geometry–texture–material–parts의 모듈형 파이프라인을 제안한다. 핵심 목표는 제품의 형태/외관 정체성을 보존하면서, 재질이 편집 가능하도록 PBR 재질을 영역 단위로 붙이는 것이다. 이를 위해 텍스처는 multiview albedo 재투영과 3D 기반 텍스처 완성을, 재질은 MatWeaver로 재질 영역 분할 및 PBR 라이브러리에서의 계층적 검색을, 파트는 material-editable semantic part mesh 생성을 지향한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 단일 RGB에서도 방수 watertight 메시에 가까운 기하를 안정적으로 복원하고, (2) 조명 변화가 섞인 관측에서 알베도를 일관되게 복원하며, (3) 생성 PBR 맵을 처음부터 모두 그리는 대신 물리적으로 타당한 재질을 영역에 맞게 ‘매칭·베이킹’하는 것이다. 논문은 geometry에 coarse-to-fine latent SDF(geometry VAE + flow matching DiT)를, texture에 멀티뷰 알베도 동시 예측 후 메시로 reproject와 3D texture field 완성을 적용해 뷰 간 일관성과 가려진 면 처리를 개선한다. material은 비디오 기반 segmentation과 UV-Face Atlas 투표로 재질 컴포넌트를 만들고, VLM 추론–cross-modal embedding–VLM reranking의 계층 검색으로 재질 라이브러리 샘플을 고른 뒤 baking으로 엔진용 PBR 맵을 압축까지 수행한다.

- **Empirical Impact**: 평가는 100케이스 가구 벤치마크에서 geometry/texture/material/parts를 각각 전용 지표로 독립 평가하며, geometry 성능은 image-to-3D 대비 상위 결과를 보였다고 밝힌다(예: CD 0.4936×10−3, EMD 5.1745×10−2, F1@0.01 0.6329). 또한 가구 도메인에서 실제로 쓰기 어려운 요소였던 얇은 구조·디테일 과평활, 텍스처 뷰 불일치, 재질 정합성의 불안정성을 모듈별로 겨냥해 ‘e-commerce 규모 생산’과 ‘재질 단위 편집’에 필요한 출력 형태를 더 가깝게 만든 점이 의미가 있다. 남은 격차는 더 넓은 배포를 위해 전 모듈의 일반화 및 생성 품질의 추가 안정화가 필요하다는 형태로 정리된다.



### Reflect-R1: Evidence-Driven Reflection for Self-Correction in Long Video Understanding (https://arxiv.org/abs/2606.27922)
Comments:
          18 pages, 6 figures, ECCV

- **Prior Approaches**: 기존 멀티모달 장기 비디오 이해의 reflection은 내부 파라미터에 의존한 closed-loop self-reflection이 대부분이었다. 이 방식은 외부의 객관적 시각 근거가 없어 blind confidence에 갇히고, 수정 과정에서 오히려 환각을 강화해 성능이 떨어지거나 무작위 변형으로 이어질 수 있다.
또한 강화학습으로 다단 reflection을 학습할 때 policy coupling 문제가 생겨, correction 논리 대신 초기 추정을 반복하는 최적화 지름길을 택하기 쉽고 학습 데이터도 부족해 한계가 누적됐다.

- **Core Contribution**: Reflect-R1은 장기 비디오 이해에서 evidence-driven self-correction을 목표로, intuition–verification–arbitration 3단 파이프라인을 처음으로 제안한다. 직관 답변은 생성하되, verification 단계에서는 retrieved keyframes만으로 독립 검증을 수행하고, 마지막 arbitration이 두 결과의 충돌을 근거 기반으로 정리한다.
불충분한 증거가 나오면 temporal search 도구를 반복 호출해 증거를 확보함으로써 환각 루프를 끊는다는 점이 핵심이다.

- **Technical Challenges**: 핵심 난제는 (1) 객관적 외부 근거 없이 내부 추론만으로는 verification이 붕괴되는 문제와 (2) 다단 강화학습에서 단계 간 보상/업데이트가 얽혀 policy coupling이 발생하는 문제였다.
저자들은 전자를 retrieved keyframes 기반의 엄격한 정보 격리(verification 입력 제한)와 abstention 보상 설계로 완화했고, 후자는 Stage-Decoupled GRPO(SD-GRPO)에서 단계별 advantage를 독립 계산해 단계 간 그라디언트 경쟁을 차단했다.

- **Empirical Impact**: VideoMME, LongVideoBench, MLVU 등 주요 벤치마크에서 state-of-the-art 성능을 달성했으며, 비디오 길이가 길수록 향상 폭이 커져 decoupled verification의 견고함을 보여줬다. 특히 반영 전후 정확도를 비교한 reflection reliability 결과에서 closed-loop 방식들이 보이는 성능 저하와 달리 Reflect-R1은 일관된 개선을 기록했다.
또한 ablation 및 학습 동역학 분석에서 SD-GRPO가 단계별 보상 신호를 분리해 진짜 self-correction 능력이 형성됨을 확인했으며, +2.82%(LongVideoBench), +1.41%(VideoMME) 수준의 genuine rectification 향상을 보고했다.



### Every Step of the Way: Video-based Parkinsonian Turning Step Counting (https://arxiv.org/abs/2606.27918)
- **Prior Approaches**: 기존 PD 보행·턴 분석은 임상 영상에서 pose estimation을 이용해 관절·보행역학 지표를 계산하거나, IMU·압력센서 같은 웨어러블로 걸음 이벤트를 검출하는 방식이 주를 이뤘다. 하지만 웨어러블은 착용·센서 위치 민감도와 일상 사용의 불편이 크고, 영상 기반 방법은 실생활의 비정형(측면/후진/피벗/미세 셔플) 턴 걸음에 가정이 잘 깨지는 문제가 있었다.

- **Core Contribution**: 이 논문은 스마트폰 등 일상 환경 카메라로 얻는 동영상에서 PD의 턴 걸음 수를 수동적(passive)으로 추정하는 coarse-to-fine 프레임워크를 제안한다. 3D human mesh recovery로 만든 발 움직임 신호로 거친 step count를 먼저 뽑고, 이후 optical flow의 미세 단서를 cross attention으로 결합해 초기 추정을 정밀 보정한다.

- **Technical Challenges**: PD 턴에서는 발이 완전히 들리지 않는 shuffling·sliding, ‘footskate’ 같은 메쉬 복원 아티팩트, 개인·질병 단계별 비주기적 변동이 커서 단순 피크 카운팅만으로는 오차가 크게 난다. 이 때문에 mesh 기반 거친 신호를 시작점으로 하되, flow에서 픽셀 수준의 미세 움직임을 끌어오도록 cross attention을 설계하고, 가변 길이 영상은 클립 분할 뒤 multiple instance learning(MIL)으로 잔차(residual) 보정과 신뢰도 높은 클립 선별을 함께 학습한다.

- **Empirical Impact**: 실제 임상·가정 환경 데이터셋(PD-FOG, Turn-REMAP)에서 기존 step counting 방법과 반복활동 카운팅(RAC) 및 IMU 기반 접근을 폭넓게 능가했으며, PD-FOG에서 Lucot et al.[imu2] 대비 절대 오차를 37.7에서 15.9로 줄였다. 또한 예측된 턴 걸음 수가 MDS-UPDRS(골드 스탠더드 임상 점수)와의 임상적 관련성도 보이며, 향후 웨어러블 없이도 턴 장애를 정량화하는 데 의미가 크다.



### There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion (https://arxiv.org/abs/2606.27905)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 MEF는 고정된 노출 프레임 수(예: 2/3/5)와 노출 레벨을 전제로 학습된 경우가 많아, 입력 프레임 수가 바뀌면 재설계·재학습이 필요했다. 또한 cross-attention 중심 설계는 구조적으로 프레임 간 유사도에 가중치를 두는데, 과노출/저노출처럼 base가 크게 망가진 영역에서는 유사도가 낮아 복원에 필요한 정보가 오히려 덜 반영되는 ‘similarity paradox’ 문제가 있었다.

- **Core Contribution**: 이 논문은 FreeMEF로, 노출 프레임 개수가 달라도(임의 길이) 학습/추론 모두에서 같은 모델을 그대로 쓰도록 하는 flexible-frame transformer를 제안한다. 핵심은 base 프레임 복원에 필요한 글로벌 정보를 먼저 global representation으로 집계한 뒤, 이를 base에 조건부로 주입하는 흐름으로 fusion 패러다임을 재구성한 점이다.

- **Technical Challenges**: 임의 길이 입력을 다루려면 프레임 수 변화에 대응하는 집계가 필요하지만, 기존 pairwise/cross-attention 설계는 가변 길이에 자연스럽게 확장하기 어렵다. FreeMEF는 RSSM(recurrent state space module)로 각 노출의 피처를 독립 임베딩 후 순차적으로 recurrent하게 집계해 길이 제약을 없앴고, GFGB에서 EAHA로 과노출 정도에 따라 base query와 reference query의 혼합을 동적으로 전환해 similarity paradox를 완화했다. 추가로 AFFN(affine-injection feed-forward network)으로 전역 방사선 통계를 기반으로 밝기·대비(스케일/시프트)를 보정해 LDR↔HDR 도메인 편이를 함께 해결한다.

- **Empirical Impact**: Kalantari 및 Real-HDRV 기반 실험과 SICE 교차 데이터셋 평가에서 FreeMEF는 프레임 수(2/3/5) 전반에 대해 기존 SOTA 대비 정량·정성 성능이 우수하게 나타났다. 특히 motion ghosting과 과노출 영역 복원에서 개선이 관찰돼, 실제 배치 환경처럼 노출 개수가 달라지는 응용에 대한 배포 효율성 측면의 의미가 크다.



### Long-Term Prediction of Local and Global Human Motion with Occlusion Recovery (https://arxiv.org/abs/2606.27900)
Comments:
          Advances in Visual Computing (ISVC 2025)

- **Prior Approaches**: 기존 3D 휴먼 모션 예측은 RNN 기반이 많았지만 autoregressive 구조라 시간이 지날수록 오차가 누적돼 장기 예측에서 취약했다. Transformer 기반으로 temporal·spatial attention을 쓰더라도 many 방식이 기본적으로 autoregressive라 여전히 로컬 포즈 중심에 머무르는 경향이 있었다. 한편 POTR/SPOTR 같은 non-autoregressive 접근은 전체 미래 시퀀스를 한 번에 예측하지만, 관절 단위 spatio-temporal 의존성을 충분히 정교하게 모델링하지 못하거나 전역(글로벌) 모션 컨텍스트까지 통합하기 어려웠다.

- **Core Contribution**: 이 논문은 non-autoregressive Transformer에 spatio-temporal attention을 결합해 로컬 포즈 예측뿐 아니라 전역 공간(global motion prediction)까지 함께 예측하도록 설계했다. 또한 실제 로봇·자율주행 환경을 고려해 과거 관측에서 관절이 가려진 경우(occlusion) 빠진 관절을 복원하도록 학습한다. 더불어 과거 히스토리 길이가 가변인 입력도 처리할 수 있게 만들어 현장 조건 변화에 대응성을 높였다.

- **Technical Challenges**: 핵심 난제는 (1) autoregressive 방식 없이도 시간적 일관성을 유지하며, (2) 관절별로 공간·시간 의존성을 동시에 학습하고, (3) 시야 가림으로 인한 누락 관절을 안정적으로 복원하는 것이다. 이를 위해 관절별 임베딩과 joint-separate temporal positional encoding을 쓰고, spatial attention은 프레임 내 관절 관계를, temporal attention은 각 관절의 시간 의존성을 각각 학습한다. 또한 occluded 관절에는 learnable token을 대체해 복원 학습을 수행하며, 디코더는 히스토리 마지막 포즈를 시드로 삼아 encoder-decoder temporal attention으로 미래 시퀀스를 비autoregressive로 생성한다.

- **Empirical Impact**: Human3.6M·AMASS·HA4M에서 MPJPE/MAE로 평가했을 때 기본 Transformer 대비 성능이 우수했고, occlusion 학습 모델은 누락이 있는 입력에서도 완전 관측 학습 모델과 유사한 수준을 보였다. 장기 예측 구간에서도 non-autoregressive SPOTR/POTR 및 autoregressive ST-Transformer보다 낮은 오차를 보고해, 에러 누적 문제를 실질적으로 완화했음을 시사한다. 또한 관절 복원·글로벌 이동(예: 걷기와 회전 동시)·조립 과업(부분 관측 팔 동작)에서 질적 결과가 확인돼 서비스 로봇 및 자율 시스템의 사전 계획에 유용한 방향성을 제시한다.



### A Multi-Attribute Latent Space for Visual Analysis of Watches (https://arxiv.org/abs/2606.27897)
- **Prior Approaches**: 기존 카탈로그/커머스 인터페이스는 메타데이터 기반 필터링에 치우쳐, 시각적 유사성이나 스타일 대안, 미학-기능을 동시에 만족하는 방식의 자유 탐색을 충분히 지원하지 못했다. 또한 시각 속성과 의미 속성을 단일 표현으로 뭉치면, 어떤 기준이 변화에 반영되는지 사용자에게 설명하기 어려운 한계가 있었다. 결과적으로 오픈엔디드 exploration(자유 탐색)과 비교 중심 워크플로우를 위해서는 더 구조화된 임베딩과 시각화가 필요해졌다.

- **Core Contribution**: 이 논문은 대규모 손목시계 컬렉션을 이종(heterogeneous) 시각·의미 속성으로 탐색하도록, 설계 근거와 임베딩 모델, 그리고 interactive visual-analysis 시스템을 제안한다. 핵심은 시계의 속성을 dial color와 dial design의 별도 attribute graph로 표현하고, watch type을 명시적 의미 조직자로 써서 전역 구조(종류)와 국소 시각 이웃(유사 다이)을 분리하는 것이다. 또한 결과 맵을 검색-by-example과 메타데이터 필터링이 함께 작동하는 대화형 인터페이스로 연결한다.

- **Technical Challenges**: 속성 간 이질성을 한 공간에 담되, 시각적 유사성 탐색과 스타일 대안 비교가 동시에 가능하도록 설계하는 것이 기술적 난제였다. 저자들은 U-Net으로 다이 분할, Vision Transformer로 watch type 예측, CIELAB 공유 팔레트로 색상을 정규화, gradient 기반 이미지 디스크립터로 다이 구조를 기술해 속성별 표현을 마련한 뒤, UMAP을 확장해 속성별 neighborhood graph를 통합 확률 목적함수로 학습한다. 여기에 class-aware layout 항을 추가해 전역은 type 구조를 따르고 로컬은 시각 이웃을 분리하도록 최적화한다.

- **Empirical Impact**: 평가는 파라미터 분석, 런타임 측정, 전문가와 비전문가를 포함한 정성 파일럿 스터디로 수행되었으며, 제안 시스템이 발견과 비교를 돕는다는 신호를 보여주었다. 다만 스케일러빌리티 평가, search-by-example 검증의 범위, 그리고 도메인 전반을 아우르는 추가 연구 필요성이 한계로 드러났다. 그럼에도 다속성 latent-space visualization을 이종 시각 컬렉션에 적용하는 설계 함의를 명확히 제시해, 향후 유사한 시각 카탈로그 탐색 연구에 실질적 방향성을 제공한다.



### OrthoTryOn: Geometric Orthogonalization for Conflict-Free Unified Fashion Generation (https://arxiv.org/abs/2606.27880)
Comments:
          Accepted by ECCV2026

- **Prior Approaches**: 기존 가상 착용(VTON), 의류 재구성, 포즈 전이 같은 패션 생성 과제는 대체로 입력 조건과 데이터 구성이 까다로워 단일 작업용으로 최적화돼 왔다. Any2AnyTryon, UniFit처럼 통합 학습을 시도하더라도 공유 LoRA를 단순히 파라미터로만 묶으면 의미적으로 이질적인 작업 간 negative transfer가 발생해 성능이 떨어진다.
또한 작업 간 gradient conflict를 사후적으로 투영하는 방식은 추가 계산(역전파 분리 등) 부담이 커 구조적 해결이 요구됐다.

- **Core Contribution**: OrthoTryOn은 여러 패션 생성 작업을 하나의 unified 모델로 학습·추론하되, 공유 LoRA 모듈 내부에서 작업 간 간섭을 줄이도록 설계한 프레임워크다. 핵심은 Orthogonal Subspace Projection(OSP)로 작업별 orthogonal rotation을 병목(bottleneck) 특징에 적용해 작업 업데이트를 decorrelated 좌표계로 보내는 것이다.
여기에 잔여 의미 결합이 남는 문제를 Fisher-guided Negative Guidance(FNG)로 보완해, 추론 시 가장 헷갈리는 작업을 hard negative로 밀어내며(Classifier-Free Guidance와 결합) semantic leakage를 억제한다.

- **Technical Challenges**: 문제는 서로 다른 의미 목표(예: VTON의 정합 vs 재구성의 구조 보존)가 같은 LoRA 파라미터 공간을 두고 경쟁하면서 gradient가 상쇄되는 현상이다. 논문은 이 간섭이 학습 중 shared LoRA 파라미터에서 gradient 크기/방향의 급격한 감쇠로 나타나며, 그 결과 타협적이고 비최적 해로 수렴한다고 진단한다.
OSP는 isometric orthogonal rotation을 frozen으로 삽입해 기대값 기준 교차 gradient/업데이트 상관을 줄이도록 재매개화하고, FNG는 대각 Fisher(실증적 민감도)를 EMA로 추정해 task-confusable 쌍을 찾아 guidance 단계에서 repulsive 성분을 주입해 residual 결합을 제거한다.

- **Empirical Impact**: VTON, 의류 재구성, 포즈 전이 전반에서 OrthoTryOn은 naive unified 학습의 성능 붕괴를 피할 뿐 아니라 다수 벤치마크에서 task-specific expert를 포함한 강한 기준을 능가한다. 또한 diffusion backbone이 달라도 견고하게 일반화되는 plug-in 성격을 보이며, 실험적으로는 OSP 단독보다 FNG가 local 디테일 아티팩트를 더 안정적으로 줄이는 경향이 확인된다.
특히 공유 파라미터 예산을 사용하면서도 single-task 학습을 따라잡거나 초과하는 결과를 제시해, ‘구조화된 파라미터 기하’가 multi-task에서 capacity dilution을 막을 수 있음을 시사한다.



### SpatialUAV: Benchmarking Spatial Intelligence for Low-Altitude UAV Perception, Collaboration, and Motion (https://arxiv.org/abs/2606.27876)
Comments:
          10 pages, 7 figures

- **Prior Approaches**: 기존 UAV·공간지능 벤치마크는 이미지 수준 인식, 단일 시점 이해, 제한된 답변 형식에 치우친 경우가 많아 저고도 비행에서 핵심인 3D 공간 추론과 교차 시점 정합을 충분히 평가하지 못했다. 또한 실환경 드론 시점(탑다운·사선, 시점 왜곡, 고도 스케일 변화, 가림, 항공-지상 불일치)과 장면 동역학(시간·움직임)까지 포괄하는 진단형 과제가 부족했다. 결과적으로 교차뷰 연관성, 구조화된 grounding, 기하 추론, 시간적 시점 이해 같은 항목이 분절적으로 다뤄지거나 누락됐다.

- **Core Contribution**: 이 논문은 실제 저고도 UAV 관측을 기반으로 한 SpatialUAV 벤치마크를 제안한다. 총 4,331개 인스턴스와 14개 fine-grained 태스크를 단일 시각입력-질문-답변(visual-input–question–answer) 스키마로 묶어 의미 구분, 공간 관계, aerial–aerial 협업, aerial–ground 협업, motion 이해를 함께 평가한다. 입력 7종 구성과 답변 9종 형식을 지원해 옵션 라벨부터 영역 식별, 기하 값, 교차뷰 대응, 자유형 동작 서술까지 다양한 과제를 진단 가능하게 만든다.

- **Technical Challenges**: 기여를 신뢰성 있게 평가하려면(1) 텍스트 편향·포맷 쇼트컷을 제거하고, (2) 시점 간 대응과 기하 정답을 일관되게 산출하며, (3) 서로 다른 출력 형태를 동일한 품질 기준으로 비교해야 한다. 논문은 detector-assisted regions, depth supervision(추정 기반), 메타데이터 규칙, 대규모 수기 라벨링을 결합하고, DeepSeek-V4-Pro와 Qwen3.6-27B로 시각 없이도 맞히는 샘플을 blind filtering한다. 또한 2단계 검증(전수 인간 교차검증 + 대표 VLM 3종의 불일치 케이스를 재검토)과 태스크별 측정치를 적용해 이질적 출력도 안정적으로 채점한다.

- **Empirical Impact**: 대표적인 vision-language model들을 3개 범주에서 평가한 결과, 평균적으로 인간 수준(89.0%) 대비 큰 격차가 남아 있으며 최고 모델도 56.7%에 그친다. 특히 cross-view association, 구조화된 grounding, 기하 추론, temporal viewpoint understanding에서 병목이 두드러졌고, aerial–ground 협업조차 56.0% 수준에 머물렀다. 또한 spatial-specific 사전학습 모델은 저고도 시점 왜곡에 잘 전이되지 않아(예: 최고 spatial-specific 29.7%), 입력 해상도 4배를 올려도 이득이 제한적이어서 향후 UAV 데이터 기반 도메인 튜닝 및 에이전트형 도구 연계(grounding·매칭·기하추정·모션 분석)가 필요함을 시사한다.



### A Unified Framework for Vision Transformers Equivariant to Discrete Subgroups of $\mathrm{O}(2)$ (https://arxiv.org/abs/2606.27864)
- **Prior Approaches**: 기존의 equivariant vision transformer 연구는 flipping- 또는 D4-등 특정 대칭군에 맞춘 변형에 머물러, 어떤 대칭을 선택하느냐에 따른 모델 계열 비교가 어렵다는 한계가 있었다. 또한 equivariant CNN의 성공 이후 ViT로의 확장이 이뤄졌지만, 표준 ViT가 데이터의 planar symmetry를 명시적으로 인코딩하지는 못한다는 문제가 남아 있었다. 관련 작업으로는 lifting self-attention이나 group convolution을 결합한 접근도 있으나, “대칭군 일반화”와 “표현력 손실이 없는가”를 통일 프레임으로 다루는 데는 공백이 있었다.

- **Core Contribution**: 이 논문은 O(2)의 임의의 discrete subgroup G≤O(2)에 대해 planar symmetries를 명시적으로 반영하는 vision transformers 계열을 제안하며, 이전의 flipping- 및 D4-equivariant transformer를 하나의 표현론적 틀로 포괄한다. 더 나아가, G-equivariant ViT 클래스가 모든 부분군 H≤G에 대해 H-equivariant ViT로 자연스럽게 포함(embedding)됨을 보이며, 대칭 제약을 키우는 것이 단절된 설계로 이어지지 않음을 정리한다.

- **Technical Challenges**: 핵심 난제는 “대칭군 작용이 token feature 공간, 선형층, 그리고 self-attention까지 일관되게 정의·구현”되어야 한다는 점이다. 논문은 token을 G의 representation(irrrep들의 채널 묶음) 위에 두고, irrep-wise equivariant affine 선형층과 셀프어텐션의 equivariant parameterization을 구성해, single-head 설정에서 ordinary self-attention이 표현할 수 있는 모든 G-equivariant self-attention map을 그대로 구현 가능함을 증명한다. 또한 D6의 여섯 겹 회전 대칭과 호환되도록 hexagonal patch 기반의 D6-equivariant 모델을 별도로 설계해, 비정사각 격자에서도 구현 가능성을 보여준다.

- **Empirical Impact**: 실험은 PatternNet 항공 이미지 데이터셋에서 데이터가 부족한 인위적 regime를 구성해, D4 및 D6를 포함한 하위 대칭군들에 걸쳐 제안 모델의 성능을 비교한다. 추가로, equivariant attention 메커니즘 선택과 nonlinearities에 쓰인 homogeneous-space 구성(어떤 GG-set을 비선형성에 쓰는지)이 성능에 미치는 영향을 제어된 조건에서 분석한다. 동일 파라미터 예산을 맞춘 예비 결과에서는 equivariance가 인식 정확도를 개선할 수 있음을 보여주며, 향후 “이산 대칭군이 transformer 기반 비전 인식에 어떤 구조적 이득을 주는지”를 더 체계적으로 연구할 동기를 제공한다.



### ScaLe-INR: Scale and Learn Implicit Neural Representations (https://arxiv.org/abs/2606.27862)
Comments:
          Submitted as a conference paper to NeurIPS 2026

- **Prior Approaches**: 기존 INRs는 좌표를 연속 함수로 근사하지만, 최적화 과정에서 저주파를 먼저 잘 배우는 spectral bias와 멀티스케일 학습 시 주파수 간 정보가 섞이는 cross-talk 문제가 반복적으로 관찰된다. 이를 줄이기 위해 SIREN 같은 주기형 활성함수, positional encoding/학습전략, band-limited 또는 Laplacian pyramid 계열의 구조 변경이 제안돼 왔다. 다만 단일 네트워크가 복잡한 고주파-저주파를 함께 담당하면 간섭이 계속 남거나, 공간 분할 방식은 MLP의 연속성/메모리 효율을 희생하는 한계가 있다.

- **Core Contribution**: 이 논문은 Scale and Learn INR(ScaLe-INR)로, 신호의 주파수 스펙트럼을 INR이 “잘 동작하는 대역”으로 맞춘 뒤 멀티브랜치로 분담하도록 설계했다. Fourier inverse scaling 관점에서 좌표 scaling이 특정 공간 축의 주파수 대역을 역스케일 형태로 이동시켜, 고주파 성분을 학습 가능한 영역으로 재배치할 수 있음을 보인다. 또한 Directional Edge Guidance Loss로 브랜치 기능을 공간적으로 분리해 고주파 브랜치가 가장자리·세부에만 기여하도록 유도한다.

- **Technical Challenges**: 핵심 기술 난제는(1) scaling이 무조건 이득이 아니라 과도하면 스펙트럼이 과압축돼 구조가 손상된다는 점과(2) 브랜치 간 기능 분리가 수학적으로/학습적으로 깨지면 여전히 cross-talk가 생긴다는 점이다. 저차원(1D) chirp 실험으로 “적절한 scaling 범위 매칭”이 필요함을 확인했고, 2D에서는 축 방향(예: x/y, 혹은 동시)으로 서로 다른 스케일을 넣은 4-브랜치(LL/HL/LH/HH)로 주파수 구간을 분할했다. 여기에 각 브랜치의 RGB 출력과 confidence logit으로 픽셀 단위 confidence-based fusion을 수행하고, Sobel 기반 방향성 edge 마스크에서 유도한 safe edge-guidance 및 sparsity prior로 고주파 브랜치의 국소 엣지 필터 역할을 안정적으로 강제한다.

- **Empirical Impact**: 실험은 이미지 재구성/노이즈 제거/초해상도, 3D 점유(occupancy), 오디오 재구성에서 SOTA 대비 성능 향상을 보였다. Kodak에서 기준 모델 대비 평균 PSNR이 +5.16 dB 개선됐고, 이미지 디노이징은 +0.65 dB 향상으로 보고된다. 또한 오디오 재구성 PSNR 50.02 dB, 3D reconstruction IoU 0.999로 최고 성능을 달성하며, scaling+브랜치 전문화가 고주파 복원력과 수렴 속도(더 빠른 convergence)를 동시에 끌어올린다는 점이 강조된다.



### Hippocampus-DETR: An Explicit Memory Object Detection Framework Based on Hippocampus Modeling (https://arxiv.org/abs/2606.27831)
- **Prior Approaches**: 현재의 객체 탐지 모델들은 명시적 메모리 메커니즘이 약해, 장면·패턴 정보의 재활용과 정교한 선택/통합이 제한적이라는 문제가 지적된다. DETR 계열을 포함한 주류 접근은 end-to-end 학습으로 성능을 끌어올리지만, 기억 구조를 분해·구성해 기능을 부여하는 방식은 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 Hippocampus-DETR로, DETR 아키텍처에 생물학적 해마(hippocampus) 메모리 모델링을 기반으로 한 HipNet 모듈을 통합한다. 해마의 하위영역(entorhinal cortex, dentate gyrus, CA3, CA1, subiculum)을 구조적으로 시뮬레이션해 pattern separation, pattern completion, importance filtering, 정보 통합을 탐지 특징에 구현하는 것이 핵심이다.

- **Technical Challenges**: 메모리 기능을 객체 탐지에 실제로 결합하려면, 각 서브모듈이 어떤 방식으로 retrieval과 completion을 수행하고 서로 다른 역할을 갖도록 학습시킬지가 큰 기술 과제다. 논문은 레이어별(layer-wise) 훈련 전략으로 메모리 서브모듈을 단계적으로 최적화해, 내부적으로 기능이 분담되는 메모리 시스템을 형성하도록 설계했다.

- **Empirical Impact**: 실험에서 Hippocampus-DETR은 기존의 mainstream 객체 탐지 모델보다 높은 정확도를 보이며, few-shot 이미지 분류, 멀티모달 특징 구성, 이미지 복원 같은 설정에서 일반화 성능과 데이터 효율이 우수함을 입증했다. 또한 후속 실험으로 각 메모리 서브모듈의 기능적 필요성과 내부 해석가능성까지 검증해, 신경인지 메커니즘을 딥러닝에 통합하는 실용적 경로를 제시했다.



### CSD: Content-aware Speculative Decoding for Efficient Image Generation (https://arxiv.org/abs/2606.27829)
- **Prior Approaches**: 기존 speculative decoding(SD)은 draft 모델이 후보 토큰을 여러 개 제안하고 target 모델이 병렬 검증해 속도를 높이지만, 이미지 생성에서는 수용률이 낮고 기준을 함부로 완화하면 화질이 흔들린다. 또한 SJD, GSD처럼 학습 없이 가속을 노려도 콘텐츠(부위별 난이도)에 따라 가속 강도를 적응적으로 바꾸지 못해 불필요한 연산이 남는다.

- **Core Contribution**: 이 논문은 entropy 기반의 확률 완화와 optimal resampling을 결합한 content-aware speculative decoding(CSD)를 제안한다. 이미지의 부위별 불확실성(엔트로피)을 신뢰도 신호로 써서 매 후보의 수용 확률을 동적으로 조절하되, distribution alignment filter로 출력 분포가 target과 어긋나는 상황을 제어한다.

- **Technical Challenges**: 가장 큰 과제는 ‘수용 기준 완화로 속도를 올리면 분포 일관성이 깨져 품질이 저하될 수 있다’는 trade-off를 이론적으로 다루면서도 실제로는 안정적으로 구현하는 것이다. CSD는 total variation(TV) 거리 기반 임계치로 완화 적용 여부를 결정하고, 완화된 acceptance 확률에 대해 TV 손실을 최소화하는 resampling 분포를 사용해 분포 편차를 최소화한다.

- **Empirical Impact**: 실험은 Lumina-mGPT와 Janus-Pro(1B/7B)를 대상으로 MS-COCO 검증셋에서 효율과 품질을 함께 평가했으며, 기존 SD 계열 대비 우수한 결과를 보였다. 예컨대 Janus-Pro 7B에서 δ=0.8로 완화 임계치를 조정하면 4.33× 지연 단축을 달성하면서도 CLIP/FID 품질 저하를 경쟁 수준으로 유지했고, 더 복잡한 장면(예: 세 마리 개)에서도 SJD 대비 속도와 CLIP이 동시 개선됐다.



### Video-MME-Logical: A Controlled Diagnostic Benchmark for Video Temporal-Logical Reasoning (https://arxiv.org/abs/2606.27828)
- **Prior Approaches**: 기존 비디오 벤치마크들은 장면 복잡도나 자연스러운 변동성과 함께 평가되는 경우가 많아, 모델이 실제로 temporal-logical reasoning을 수행하는지(증거를 유지·갱신·조합하는지) 구분하기 어렵습니다. 또한 대부분이 최종 정답 중심 평가라 과정이 틀려도 정답을 맞히면 실패 원인을 해석하기 힘듭니다. 일부 시간 추론 벤치마크가 존재하지만, 문제를 고정된 논리 연산 단위로 분해하거나 중간 추적을 검증하는 방식은 제한적이었습니다.

- **Core Contribution**: 이 논문은 비디오에서 시간에 따라 변하는 시각 증거를 추론하는 능력인 video temporal-logical reasoning을 정밀 진단하기 위해 Video-MME-Logical을 제안합니다. 벤치마크는 five temporal-logical operations(상태 추적, 순차 카운팅, 시간 순서, 동적 공간성, 구조적 조합)으로 작업을 분류하고, 25개 세부 태스크를 절차적 생성으로 구성합니다. 난이도(temporal horizon·reasoning complexity)와 중간 상태 검증(Video-MME-Logical-S)까지 제공해, 최종 정답만으로 과대평가되는 문제를 줄입니다.

- **Technical Challenges**: 핵심 난제는 모델이 여러 프레임을 단순 집계하는지, 아니면 숨겨진 상태를 유지·업데이트하고 시간적 논리로 증거를 조합하는지 명확히 분리해 측정하는 데 있습니다. 이를 위해 각 태스크를 프로그램 생성(temporal transition, scene configuration, metadata construction, video rendering)으로 정의하고, 중간 상태 진단 subset에서는 프로그램이 기록한 logical reasoning trace를 모델이 같은 구조로 출력하도록 강제합니다. 또한 쉬움/중간/어려움 난이도를 temporal horizon과 상태 전환 횟수 등으로 제어해 해석 가능한 난이도 스케줄을 제공합니다.

- **Empirical Impact**: 실험에서 인간은 전체 정확도 95.9%를 보인 반면, 최고 성능 모델(gemini-3.1 Pro)은 28.6%에 그쳐 human-model gap이 크게 나타났습니다. 특히 thinking(추론 과정 생성)만으로는 성능이 일관되게 개선되지 않았고, 시간 범위가 길어지고 논리 복잡도가 커질수록 급격한 성능 저하가 확인됐습니다. SFT로 500K 규모까지 학습시키면 375K에서 최고 39.2%까지 오르지만 빠르게 포화되며, 더 단순한 supervised scaling만으로는 reasoning gap을 닫기 어렵다는 점을 보여주어 향후 연구의 testbed로 가치가 큽니다.



### Scalable and Differentiable Point-Cloud Registration Using Maximum Mean Discrepancy (https://arxiv.org/abs/2606.27818)
Comments:
          Accepted at ICML 2026

- **Prior Approaches**: 3D 포인트클라우드 정합은 보통 ICP처럼 하드 nearest-neighbor 대응을 기반으로 국소 최소제곱을 푸는 방식이 주류다. 하지만 대응이 이산적으로 바뀌며 비미분성이 생기고, 잡음·아웃라이어·비균일 샘플링에서 목표함수가 국소해로 끌려갈 위험이 있다. CPD/GMMReg/FilterReg 같은 확률적 방법은 soft correspondence로 강건성을 높이지만, 반복마다 추가 계산과 복잡성이 커지는 편이다.

- **Core Contribution**: 이 논문은 correspondence-free 정합을 위해 rigid 정합을 MMD(Maximum Mean Discrepancy)의 랜덤 푸리에 피처(RFF) 근사로 만든 미분 가능 nonlinear least-squares 문제로 제안한다. MMD의 분포 불일치 측정을 직접 샘플에서 추정해 밀도 모델링 없이도 잡음과 아웃라이어에 비교적 강한 정합 목표를 제공한다. 또한 해를 Levenberg–Marquardt로 효율적으로 구하고, implicit function theorem 기반으로 end-to-end 학습 가능한 differentiable optimization layer로 확장한다(Neural MMD-Reg).

- **Technical Challenges**: 핵심 기술 난제는 (1) MMD의 커널 합이 포인트 수에 대해 quadratic으로 커지는 문제를 선형으로 줄이면서(=RFF) (2) 랜덤 근사로 인한 비선형·비볼록 최적화에서도 실사용 가능한 안정성을 확보하는 것이다. 논문은 RFF로 커널을 유한 차원 특징 z(x)로 대체해 커널 평균 임베딩 차이를 residual로 만들고, 표준 nonlinear least-squares 솔버(Levenberg–Marquardt)를 적용한다. 더 나아가 JAXopt의 implicit differentiation과 IFT를 활용해 최적해 θ*의 입력에 대한 그라디언트를 unrolling 없이 계산해 학습 레이어로 사용 가능하게 한다.

- **Empirical Impact**: 실험에서는 합성 데이터에서 잡음·샘플링 밀도·아웃라이어 요인을 분리해 MMD-Reg가 다양한 baseline(ICP 변형, GICP, 확률적 방법 등) 대비 어떤 조건에서 유리한지 확인한다. 이어 큰 규모 야외 LiDAR 스캔에서 coarse-to-fine(커널 스케일 단계적 축소)로 어려운 초기 정렬/부분 겹침 상황을 다룬다. Neural MMD-Reg는 set transformer와 결합해 supervised·unsupervised 학습 모두에서 최근 learning-based 방식 및 standalone MMD-Reg와의 성능·확장성을 비교하며, 대규모 포인트에서도 실전형 differentiable 정합 모듈로서의 의미를 보여준다.



### Text as Illumination: Spatial Contrastive Retinex Learning for Language-guided Medical Image Segmentation (https://arxiv.org/abs/2606.27794)
Comments:
          Aceepted by MICCAI2026. More modifications may be performed

- **Prior Approaches**: 언어-유도 의료 영상 분할(LMIS) 분야는 텍스트-비전 결합을 attention/피처 머지로 암묵적으로 수행하거나, 모달리티 정렬을 위한 보조 감독(대략적 coarse supervision)에 의존하는 흐름이 주류였습니다. 하지만 암묵적 결합은 의미 일치(semantic consistency)를 명시적으로 강제하기 어렵고, coarse 정렬은 픽셀 수준의 정교한 대응을 제한합니다. 결과적으로 언어 의미와 분할 출력 사이에 불일치가 발생하기 쉽습니다.

- **Core Contribution**: 본 논문은 Retinex 관점을 LMIS에 매핑한 Text-as-Illumination Retinex Network(TIRNet)을 제안합니다. 텍스트 임베딩을 ‘semantic illumination’로 해석해 디코더 단계에서 전경은 강화하고 배경 간섭은 억제하면서 의미 일치를 높입니다. 또한 각 디코더에 RTMB와 CDCB를 통합해 전경-배경 분리와 고주파 경계 디테일 복원을 동시에 노립니다.

- **Technical Challenges**: 핵심 과제는 텍스트 의미가 실제 픽셀 예측과 정밀하게 맞물리도록 ‘명시적이고 세밀한 제약’을 설계하는 것입니다. 이를 위해 RTMB는 positive/negative illumination maps로 전경은 올리고 배경은 억제하며, CDCB는 illumination reliability에 기반한 consistency gate로 인코더의 고주파 디테일을 선택적으로 복원합니다. 아울러 Multi-Scale Illumination Supervision Loss(MSIS-Loss)를 도입해, Region-Grounded Contrastive Loss(RGC-Loss)로 전경-배경 간 마진을 키우고 Background Suppression Loss(BS-Loss)로 negative illumination이 배경에 맞도록 픽셀 수준 감독을 제공합니다.

- **Empirical Impact**: MosMedData+와 QaTa-COV19에서 TIRNet은 SOTA 성능을 달성하며, 기존 방식(LViT, TeViA) 대비 m-Dice/m-IoU 또는 g-Dice/g-IoU에서 유의미한 개선을 보였습니다. 특히 ablation에서 RTMB, CDCB, MSIS-Loss가 각각 성능 향상에 기여하며 전체 모델이 기준선 대비 m-Dice를 크게 끌어올렸습니다. 이는 LMIS에서 텍스트를 단순 융합 신호가 아닌 ‘조명(illumination)’으로 구조화해 정렬 품질을 끌어올릴 수 있음을 실증적으로 뒷받침합니다.



### Improving Adversarial Robustness via Activation Amplification and Attenuation (https://arxiv.org/abs/2606.27784)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 적대적 공격(Adversarial examples)은 신경망의 비강건(non-robust) 특징을 교란해 오분류를 유도하며, 이를 막기 위해 adversarial training(AT, TRADES, MART 등)과 특징 내부를 바꾸는 plug-in 방어가 함께 발전해왔다. plug-in 계열은 비강건 채널을 마스킹/가지치기하거나(FS 계열, CAS·CIFS 등) robust/non-robust로 분리한 뒤 후자를 MLP로 재보정(FSR·FTA2C)하는 방식이 주류지만, 추가 재보정 블록의 파라미터 기여가 덜 해석 가능하다는 한계가 남았다.
또한 OOD에서 activation scaling이 효과적이라는 통찰이 있었지만, 이를 공격학습(미분 가능성과 안정성)으로 그대로 가져오면 top-k 같은 비미분 연산·큰 스케일로 인한 수치/그래디언트 문제에 부딪힐 수 있다.

- **Core Contribution**: 이 논문은 Activation Amplification and Attenuation(A3)라는 가벼운 플러그인 모듈을 제안해, 비강건 신호를 “억제(attenuation)”하면서도 같은 설계를 “증폭(amplification)” 모드로 전환할 수 있게 만든다. A3는 채널별 learnable mask와 활성 크기 분포에서 유도한 스케일링으로 activation을 동적으로 재조정하며, 증폭 모드에서 얻은 신호를 학습 시 negative reference로 써 새로운 contrastive·ranking loss를 구성한다.
특히 증폭/억제는 스케일 연산의 부호를 뒤집는 방식으로 통일되어, 추가 네트워크 용량을 크게 늘리지 않으면서도 두 모드의 상호작용을 통해 방어 성능을 끌어올린다고 주장한다.

- **Technical Challenges**: A3를 adversarial training에 적용할 때 핵심 기술 과제는 (1) top-k 기반 scaling을 쓰면 gradient obfuscation 위험이 커진다는 점과 (2) OOD용 스케일링의 지수형(exp) 형태가 커다란 팩터를 만들어 gradient 포화·수치 불안정을 유발할 수 있다는 점이다. 논문은 이를 피하기 위해 Gumbel-Softmax로 differentiable한 채널 마스크를 만들고, 지수 대신 로그(logarithmic) 형태의 스케일 팩터를 사용해 안정성을 확보한다.
또한 스케일 팩터를 [0,1] 범위로 제한해 극단적 증폭/감쇠를 막고, inference에서는 attenuated activation만 사용하며 amplification은 training-time loss 설계를 위한 음성 기준으로만 활용하도록 구성한다.

- **Empirical Impact**: 실험은 CIFAR-10/100과 Tiny ImageNet에서 ResNet-18·WideResNet-34-10 백본, 그리고 AT/TRADES/MART 같은 대표 adversarial training 세팅을 모두 포함해 수행됐다. A3를 추가하면 ensemble(Ens.) 정확도와 AutoAttack(AA) 등 강공격 평가에서 일관되게 향상되며, 예로 ResNet-18+AT에서 CIFAR-10 Ens.가 바닐라 대비 4.99%p 개선되는 결과가 제시된다.
정량적으로는 clean accuracy가 소폭 떨어질 수 있으나 이는 adversarial training의 일반적 트레이드오프 범주이며, 계산·메모리 오버헤드는 기존 plug-in 대비 매우 작다고 보고된다. 나아가 activation 분포 히스토그램 분석에서 A3(attenuation)가 adversarial에서만 나타나는 활성 피크를 효과적으로 눌러 separability를 높이고, amplification은 의도대로 clean/adv 분리성을 낮춘다는 해석도 함께 제공된다.



### MindFlow: Harmonizing Cognitive Semantics and Acoustic Dynamics for Facial Animation Generation in Dyadic Conversations (https://arxiv.org/abs/2606.27779)
Comments:
          Accepted by ECCV 2026

- **Prior Approaches**: 기존 오디오 기반 talking-head 및 listening 분기 연구들은 음향 신호와 얼굴 운동의 저수준 상관을 학습해 “반응성”은 만들었지만, 대화 의미·감정의 장기 정합성이 약해 표정이 비어 보이는 문제가 컸습니다. LLM을 활용한 Sentence-Action 계열은 문장 의미로는 그럴듯함을 확보해도, 음성의 운율·준언어 정보를 잃고 문장 단위로 타이밍이 거칠어지는 한계가 남았습니다.

- **Core Contribution**: MindFlow는 대화 얼굴 애니메이션을 Ventral(의미/감정)과 Dorsal(감각-운동 반사 생성) 두 경로로 분리해, 의미적 적합성과 고정밀 제어를 동시에 노린 이중 경로 생성 프레임워크입니다. Ventral 모듈은 Sentence-Action 대신 Chunk-State와 streaming Chain-of-State로, 원시 오디오의 미세한 감정 변화를 연속적으로 추적해 Dorsal의 생성을 정교하게 조절합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 텍스트로 환원하면 사라지는 운율·감정 단서를 원시 오디오에서 유지하면서 (2) MLLM이 스트리밍 상황에서도 상태 일관성을 갖게 하는 것입니다. MindFlow는 chunk 윈도우 기반으로 감정 상태 체인을 누적하고, Dorsal 쪽에는 conditional autoregressive flow matching을 두어 고주파 오디오 큐를 반사적으로 얼굴 운동에 매핑하며, 또한 Selective Acoustic Injector로 발화/청취 국면에 맞춰 오디오 소스를 무감독으로 게이팅해 신호 희석을 줄였습니다.

- **Empirical Impact**: 실험에서 MindFlow는 HDTF/VICO 등 벤치마크에서 lip synchronization, 표정 정확도, 동작 자연스러움(FD) 전반에서 최신 베이스라인을 능가하며, 특히 25 FPS 실시간 스트리밍 생성과 장문(연속 대화) 일관성에서 강점을 보였습니다. 사용자 평가(자연스러움·적합도 비교)와 오 ablation에서도 Ventral의 evolving emotion state와 Chunk-State, Selective Acoustic Injector가 성능 향상에 직접 기여함이 확인되어 분야 내 “semantic-guided reflexive generation” 방향의 실효성을 보여줍니다.



### TRUST: Efficient Abdominal Trauma Recognition via Image-to-Ultrasound-Video Transfer Learning (https://arxiv.org/abs/2606.27777)
Comments:
          Accepted to MICCAI 2026, 11 pages, 5 figures

- **Prior Approaches**: 기존 복부 초음파 영상 분석은 연속 스캔에서 나타나는 미세한 시공간 단서를 사람이 직접 해석해야 해 시간이 오래 걸리고 숙련도 의존성이 컸다. PEIVTL 계열은 CLIP 같은 비전-텍스트 정렬로 사전학습 모델을 영상 도메인에 적용하지만, 환자/시술자에 따라 달라지는 spatiotemporal·semantic 변동을 충분히 흡수하지 못해 일반화가 제한된다. 특히 저주파·고주파를 독립 처리하거나 고정 윈도우 기반 단일 시간 스키마를 쓰는 방식은 초음파의 잡음(speckle)과 뷰 경계의 불연속을 동시에 다루기 어렵다. 또한 정적인 텍스트 템플릿은 각 검사 인스턴스의 촬영 조건 차이를 반영하지 못해 cross-modal 정렬이 흔들릴 수 있다.

- **Core Contribution**: TRUST는 scan-aware PEIVTL로서 초음파 영상의 시공간 변동을 ‘공간-시간-의미’ 세 축에서 점진적으로 정교화한다. 공간은 Cross-Frequency Collaborative Adapter(CFCA)로 저주파/고주파 간 상호 제약을 학습해 잡음 환경에서도 진단에 유리한 미세 특징을 뽑는다. 시간은 Multi-Granularity Motion-Aware(MGMA)로 intra-view의 부드러운 연속성과 inter-view의 급격한 전환을 동시에 모델링하고, 의미 정렬은 Visual Query Semantic Aggregation(VQSA)로 영상 임베딩에 조건화된 동적 텍스트 프로토타입을 합성한다.

- **Technical Challenges**: 핵심 난제는 (1) speckle 잡음이 병변의 미세 단서를 가리면서도, (2) 검사자 움직임과 해부학 변화가 섞여 이중 시간역학(짧은 연속·긴 불연속)이 발생하며, (3) 같은 병리라도 촬영 각도·압력 차이로 의미 표현이 달라지는 점이다. TRUST는 CFCA에서 wavelet 분해 후 저주파 마스크 게이팅과 적응형 스케일링/오프셋으로 하·상호보완을 강제하고, MGMA에서는 로컬은 temporal convolution으로 프레임 지터를 완화하며 글로벌은 motion-prior position encoding으로 뷰 경계의 의미적 불연속을 캡처한다. 마지막으로 VQSA가 시각 임베딩을 query로 삼아 텍스트 뱅크에서 관련 설명을 동적으로 집계함으로써 인스턴스별로 흔들리는 visual-textual alignment를 보정한다.

- **Empirical Impact**: 사내(in-house) 초음파 외상(tramua) 데이터셋 294개 영상 실험에서 TRUST는 기존 SOTA 대비 정확도 9.63% 향상을 보이며, 계산 효율도 우수한 것으로 보고된다. 특히 4×24×2 설정에서 Accuracy 81.36%, tIoU(Jaccard) 64.32%로 2위 대비 큰 폭의 개선을 기록했는데, 분석 결과 운동(motion) 기반 설계가 해부학적 변동에 따른 피처 혼동을 줄이는 데 기여한다. 어블레이션에서는 CFCA·텍스트 적응·MGMA·VQSA를 순차 추가할수록 성능이 67.50%→70.96%→76.36%→83.34%까지 상승했으며, VQSA가 특히 큰 도약을 만들었다.



### ModaFlow: Modality-Aware Flow Matching for High-Fidelity Virtual Try-On (https://arxiv.org/abs/2606.27773)
Comments:
          Preprint

- **Prior Approaches**: 기존 가상 착용(virtual try-on) 연구는 GAN 기반으로 정렬·합성을 시도했지만 학습 불안정과 텍스처 손실이 잦았습니다. 확산 기반 모델은 품질을 끌어올렸으나 비주얼과 텍스트 조건을 대칭적으로 처리해 설명-외형 정합이 흔들리거나, paired 데이터·고정 변형에 의존해 unpaired 상황에서 적응성이 떨어졌습니다.

- **Core Contribution**: ModaFlow는 flow-matching 기반 프레임워크로, 의복 의미(텍스처·패턴·스타일)와 인체 기하(자세·체형·아이덴티티)를 동시에 맞추는 고해상도 가상 착용을 목표로 합니다. 비주얼과 텍스트 조건을 서로 다른 역할로 비대칭 통합해, 비주얼은 결정적이고 지속적인 구조 가이드를 제공하고 텍스트는 classifier-free guidance(CFG)로 정밀 의미 정렬을 수행합니다.

- **Technical Challenges**: 핵심 난제는 큰 변형과 가림(occlusion) 하에서 조건 신호를 어긋나지 않게 흐름(flow field)로 연결하는 것입니다. ModaFlow는 (1) 텍스트에만 CFG-Zero*를 적용하고 초기 velocity를 zero-init해 안정성을 확보하며, (2) cosine similarity와 perceptual flow discrimination이라는 두 정규화로 방향 일관성과 속도장(realism)을 동시에 강화합니다. 또한 학습 중 box/transparent/relaxed 마스크를 확률적으로 샘플링해, inference에서 box mask만 제공되는 unpaired 설정에도 강건하게 대응하도록 학습합니다.

- **Empirical Impact**: VITON-HD와 DressCode에서 paired/unpaired 모두에 대해 정성·정량 성능이 SOTA 수준을 보였고, FID는 paired에서 약 30%, unpaired에서 약 20% 감소했습니다. SSIM과 LPIPS 개선은 구조 보존과 세부 텍스처 유지가 향상됐음을 시사하며, ablation에서도 modality-aware guidance·마스크 조작·flow 정규화가 각 성능 기여를 갖는 것으로 확인됐습니다. 패션 합성에서 ‘설명 정합’과 ‘기하 적응’을 함께 끌어올렸다는 점에서 e-commerce 및 AR 응용에 대한 실사용 가능성을 넓혔다는 의미가 큽니다.



### An Embedded Real-Time License Plate Recognition System for Complex Traffic Scenes (https://arxiv.org/abs/2606.27772)
Comments:
          Accepted at IEEE Intelligent Transportation Systems Conference (ITSC) 2026

- **Prior Approaches**: 기존 LPR 연구는 주로 주차장/톨게이트처럼 배경이 단순한 단일 차량 장면 중심이 많았고, 복잡한 다차량·비정형 도로에서는 오탐이 늘 수 있다는 한계가 있었다. 딥러닝 기반 다차량 LPR도 Jetson 같은 embedded GPU나 고성능 CPU에 의존하는 경우가 많아 전력·비용 문제로 저가 보급형 환경에는 부담이 컸다. 또한 복잡한 도로 시나리오는 다루더라도 임베디드 실시간 제약을 만족할 만큼 모델이 가볍지 않거나, 지연이 커 다차량 장면에는 어려움이 있었다.

- **Core Contribution**: 이 논문은 개발도상국의 비정형·다차량 교통(차선 분리 불명확, 소형 오토바이 다수, 다양한 차량 유형)에서 동작하는 임베디드 실시간 LPR end-to-end 파이프라인을 제시한다. 핵심은 경량 CNN 기반으로 license plate detection(LPD)과 license plate character recognition(LPCR)을 분리 수행하고, 이를 FPGA에서 가속해 저비용 배포가 가능하게 만든 점이다. 더불어 스리랑카 도로를 반영한 SL-LPR 데이터셋(다양한 차량 유형·트래픽 조건)을 공개해 학습/평가의 기반을 확장한다.

- **Technical Challenges**: 문제는 ① 비정형 다차량 이미지에서 작은 번호판을 안정적으로 찾고, ② 검출 후 잘린 영역에서도 문자를 정확히 읽으면서, ③ 임베디드 리소스 제약 안에서 지연을 줄이는 것이다. 저자들은 LPD로 low-precision YOLO 계열(LPYOLO)을 4비트/저비트 양자화(Brevitas)하고, license plate 비율에 맞춘 anchor를 K-means로 재설계했으며, sigmoid/HardTanh 근사로 인한 정밀도 저하를 값 범위 제한 후 precompute된 sigmoid로 완화했다. LPCR은 fast-plate-ocr 계열 CNN을 경량화·프루닝·양자화해 FPGA(FINN)로 컴파일했고, 탐지-크롭-인식을 함께 동작시키되 NMS와 전처리 병목을 줄이도록 파이프라인을 구성했다.

- **Empirical Impact**: SL-LPR에서 LPD는 93.6% mAP, LPCR은 87.88% 문자 정확도를 보였고, 더 큰 모델이 쓰인 공개 벤치마크에서도 경쟁력을 보였다. 전체 시스템은 Xilinx Kria KV260 FPGA에서 11.5 FPS(프레임당 최대 87 ms 지연)로 동작했으며, 전력은 4.2 W로 GPU 기반 대비 효율이 높았다. 개발도상국 교통처럼 차량 다양성과 배경 복잡성이 큰 환경에서 실시간 임베디드 LPR이 가능함을 실증했다는 점에서, 엣지 교통 모니터링·단속 적용 가능성을 크게 넓혔다.



### PixelU: A U-Shaped Transformer for Efficient End-to-End Pixel Diffusion (https://arxiv.org/abs/2606.27760)
- **Prior Approaches**: 엔드투엔드 픽셀 공간 diffusion은 VAE 압축 없이 고해상도 이미지를 만들 수 있지만, 픽셀의 고차원 공간에서 저주파 의미와 고주파 디테일을 동시에 학습하기가 매우 어렵다. 이에 따라 기존 SOTA는 고주파 보존을 위해 복잡한 pixel decoder(보조 디코더) 같은 모듈을 붙여 성능을 끌어올리는 흐름이 강했다. 다만 이런 모듈은 GFLOPs와 추론 비용을 크게 늘려 효율을 떨어뜨린다는 한계가 있었다.

- **Core Contribution**: 이 논문은 복잡한 픽셀 디코더가 필수라는 통념을 뒤집고, 주된 원인이 vv-prediction(velocity 예측)에서 발생하는 최적화 난이도를 보완하는 데 있음을 보인다. 반대로 xx-prediction(클린 이미지 예측)에서는 디코더의 이점이 급격히 줄며, 같은 비용 대비 개선 폭이 미미하다. 이를 바탕으로 PixelU는 단일 stage의 U-shaped Diffusion Transformer를 픽셀 공간에 맞춰 최소 구성(보조 디코더 제거)으로 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 xx-prediction으로도 남는 고주파 디테일(텍스처·엣지)의 소실을, 디코더 없이 어떻게 복구하느냐이다. PixelU는 zero-cost에 가까운 skip connections로 얕은 층의 uncorrupted 고주파 정보를 깊은 층으로 “information highway”처럼 직접 전달해 해결하고, backbone이 저주파 의미만 학습하도록 constant-channel 기반의 spatial down-sampling을 low-pass filter로 넣어 병목을 형성한다. 나아가 단일 down/up-sampling과 채널 차원 상수화를 통해 계산량을 대폭 줄이면서도 주파수 분리를 유지하도록 설계했다.

- **Empirical Impact**: ImageNet 256×256과 512×512에서 PixelU는 각각 FID 1.63, 1.92를 기록하며 최근 픽셀 공간 방법들을 상회한다. 특히 강한 baseline인 JiT-G를 더 적은 계산비용(약 1/3)으로 능가해 효율-성능의 동시 개선을 입증했다. t-SNE와 FFT 기반 주파수 에너지 분석, 그리고 ablation에서 down-sampling은 저주파 의미를 압축하고 skip connections는 고주파를 복원해 주파수 분리가 실제로 일어난다는 근거를 제공하며, “복잡한 디코더가 아니라 기본 구조의 시너지”가 픽셀 diffusion의 새 패러다임이 될 수 있음을 시사한다.



### Panoramic Scene Analysis: A Survey from Distortion-Aware Engineering to Sphere-Native Foundation Modeling (https://arxiv.org/abs/2606.27745)
- **Prior Approaches**: 기존 파노라마 분석은 주로 투영 기반 적응, ERP 같은 평면 표현의 왜곡을 보정하는 distortion-aware engineering, 구면 연산을 직접 도입하는 sphere-native modeling, 그리고 foundation-model을 위한 geometry-aware tokenization(GT)으로 발전해 왔다. 다만 투영 기반은 seam과 중복 연산 부담이 있고, distortion-aware는 투영별 보정이 누적적으로 필요하며, sphere-native는 2D 연산·pretrained 백본과의 호환성이 발목을 잡는 문제가 있었다. 또한 foundation 모델 가중치를 “그대로” 재사용하면서 엄밀한 구면 등변성까지 동시에 만족시키는 해법이 부족하다고 지적한다.

- **Core Contribution**: 이 논문은 파노라마 씬 분석을 2-sphere S2에서의 추론으로 재정의하고, architectural design(투영/왜곡/구면 연산/토큰화)과 training paradigm(도메인·지식 전달 방식)을 축으로 한 2차원 택소노미를 제안한다. 그 결과, 기술이 늘어난다는 단순 누적이 아니라 “기하학적 헌신”이 단계적으로 깊어지는 진화라는 관점을 제시한다. 동시에 남아 있는 핵심 긴장으로 “strict spherical equivariance”와 “perspective-pretrained foundation-model 가중치의 full reuse”가 동시에 성립하지 않는 구조적 공백을 강조한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 투영이 만드는 극(위도) 왜곡과 seam 불연속, (2) 구면은 평면처럼 translation symmetry가 없다는 본질적 기하 차이, (3) 이 불일치를 단순 fine-tuning이나 데이터 보정만으로는 해소하기 어렵다는 점이다. 논문은 이를 해결하기 위해 distortion awareness, sphere-native operations, pretrained-model compatibility라는 세 요구조건으로 방법을 분해하고, GT처럼 입력 인터페이스를 재설계해 foundation 모델의 지식을 기하학적으로 연결하는 방식을 체계화한다. 또한 실무적으로는 각 계층(주의, 좌표표현 등)에서 기하 충실도를 유지해야 하며, positional encoding이 데이터의 중력/위도 편향을 학습해버리면 등변성이 붕괴할 수 있다고 경고한다.

- **Empirical Impact**: 벤치마크 분석을 통해 perspective-to-panorama 격차가 특정 모델 문제가 아니라 구조적 미스매치임을 반복적으로 보여주며, 예컨대 무적응 평가에서 mIoU가 큰 폭으로 하락하는 결과를 인용한다. 특히 평가 프로토콜의 결함이 “진짜 구면 이해”를 측정하지 못하게 만든다고 보고, spherical-area-weighted metrics, seam-consistency, polar-robustness stratification, cross-projection generalization, open-world 표준화 등 5가지 체계적 공백을 도출한다. 마지막으로 이를 바탕으로 일반 목적 panoramic intelligence를 위한 6단계 연구 로드맵을 제시해, 향후 벤치마크·모델 설계의 방향성을 실증적으로 끌어올리려는 영향이 크다고 평가한다.



### SIFT: Self-Imagination Fine-Tuning for Physically Plausible Motion in Video Diffusion Models (https://arxiv.org/abs/2606.27741)
Comments:
          ECCV 2026

- **Prior Approaches**: 기존 비디오 diffusion 모델은 시각 품질과 텍스트 의미 정합을 높이는 데 집중했지만, 생성된 움직임이 물리적으로 말이 되지 않는 문제가 지속됐다. Physics-Grounded 생성은 시뮬레이션이나 사전 계획으로 동역학을 보강하는 방식이 많지만, 모델 내부의 kinematics(상대 운동) 자체가 틀어지는 ‘구조적’ 실패를 충분히 다루지 못했다.

- **Core Contribution**: 이 논문은 kinematics 관점에서 흔한 실패를 ‘motion entanglement’(카메라 운동과 객체 운동 등 독립 운동원이 의도치 않게 결합되는 현상)으로 정의하고 원인을 분석한다. 핵심 기여는 재구성(reconstruction) 중심 학습을 깨고, Self-Imagination Fine-Tuning(SIFT)으로 모델이 텍스트 프롬프트로부터 스스로 생성한 영상에서 운동을 학습하도록 전환한 점이다.

- **Technical Challenges**: 주요 기술적 난제는 노이즈 비디오를 입력으로 다시 복원하는 설계가 ‘reconstruction shortcut’을 만들어, 모델이 텍스트 의미로부터 운동 추론을 하지 않고 잔존 운동 단서를 복사하게 되는 것이다. 이를 해결하려고 SIFT에서 실영상 입력을 제거하고 순수 가우시안 노이즈에서 생성하게 하며, 픽셀 MSE 대신 motion-aware discriminative supervision과 progressive hard-case replay로 학습을 안정화·가속했다.

- **Empirical Impact**: 실험에서는 두 가지 백본(예: Wan2.1-T2V-1.3B, CogVideoX) 모두에서 Semantic Adherence(SA)와 Physical Commonsense(PC) 및 인간 선호도에서 크게 개선된 결과를 보였다. 또한 camera-only/object-only를 넘어 multi-object, articulated, long-horizon 같은 복잡한 장면에서도 일관되게 성능이 좋아져, 물리적 현실감뿐 아니라 운동 분리와 controllability 향상에 의미 있는 영향을 확인했다.



### Learning 1-Bit LiDAR-based Localization with Auxiliary Objectiv (https://arxiv.org/abs/2606.27729)
Comments:
          European Conference on Computer Vision(ECCV)

- **Prior Approaches**: 기존 LiDAR 6-DoF localization은 SCR(장면 좌표 회귀 후 등록)과 APR(절대 자세 회귀, end-to-end)로 나뉘며, APR이 배치/매칭 비용을 줄여 효율적이라는 장점이 있다. 다만 학습 기반 APR은 엣지 디바이스에서 연산·메모리 제약을 만족하기 어렵고, 전반적인 양자화/압축(프루닝·지식 증류·quantization·compact 설계) 중에서도 binarization은 1-bit로 극단적 효율을 노리되 성능 저하가 커진다.

- **Core Contribution**: 이 논문은 6-DoF LiDAR localization을 위한 최초의 BNN 프레임워크인 Binarized LiDAR-based Localization(BiLoc)을 제안한다. BiLoc은 information-bottleneck 관점에서 이진 인코더가 ‘최소하지만 자세 추정에 충분한’ 표현만 남기도록 학습을 재해석하고, 이를 위해 정보 보존을 조절하는 학습-전용 보조 목적(auxiliary objective)을 도입한다.

- **Technical Challenges**: BNN을 그대로 적용하면 sign 기반 binarization이 표현력을 크게 제한하고, STE 등으로 인한 gradient mismatch 때문에 최적화가 불안정해져 세밀한 자세 차이를 학습하지 못하는 문제가 발생한다. BiLoc은 offline 실재값(teacher) 모델의 특징을 참조 분포로 삼아 feature distillation을 수행하되, 채널별 Mahalanobis 기반 soft mask와 구조 정렬(조건부 상호정보를 근사하는 optimal transport/Sinkhorn)을 결합해 binarization으로 인한 정보 손실을 보완한다.

- **Empirical Impact**: 대규모 야외 LiDAR 데이터셋 실험에서 BiLoc은 BNN 기반 localization의 새로운 SOTA를 달성하며, 특히 Oxford Radar RobotCar에서 평균 위치 오차와 평균 방향 오차를 각각 10.11%, 11.16% 낮춘다고 보고한다. inference 시 보조 목적을 제거해 1-bit의 계산·메모리 효율은 유지하면서도 정확도를 끌어올렸다는 점에서, 항상-on 모듈로서의 실용성 향상에 의미가 있다.



### Scene and Human in One World: Reconstruction in a Feedforward Pass (https://arxiv.org/abs/2606.27720)
- **Prior Approaches**: 기존 human mesh recovery(HMR) 계열은 주로 카메라 좌표에서의 자세·형상 복원에 집중해 단안 영상의 메트릭 스케일·깊이·전역 배치 모호성을 충분히 해소하지 못했다. Human–scene 정렬을 위해 사람과 장면을 따로 예측한 뒤 post-hoc로 맞추는 파이프라인도 많아, 가림·군중·복잡한 시점에서 로컬라이제이션 신호가 흔들리면 스케일·위치 불일치가 누적됐다. 최근 feed-forward 결합 방식이 등장했지만, 사람의 의미론적 priors가 장면 기하 추정에 깊이 주입되지 않거나 장면 기하가 사람 추정에 강하게 되먹임되지 않는 한계가 남아 있다.

- **Core Contribution**: 본 논문은 SHOW를 제안하며, 단안 human-centric 비디오에서 사람 메쉬(SMPL-X)와 장면 기하를 ‘공유된 metric space’에서 동시에 복원한다. 핵심은 두 작업을 분리하지 않고 서로를 학습 전반에 걸쳐 상호 제약하도록 커플링하는 것으로, parametric human model의 scale priors·의미론을 장면 point-map 예측에 주입하고, 반대로 장면 기하는 사람의 글로벌 placement와 스케일 추정에 제약으로 돌아오게 한다. 또한 mask-promptable 메커니즘으로 다중 인물/클러터 상황에서 목표 인물 선택을 유연하게 하면서 배경 방해와 가림 간섭을 억제한다.

- **Technical Challenges**: SHOW가 직면한 첫 과제는 단안에서 본질적으로 scale-ambiguous한 입력을 메트릭 일관성 있는 장면 point-map으로 바꾸는 것이며, 이를 위해 pretrained visual geometry foundation model(VGGT)을 사람 인지형으로 적응한다. 구체적으로 mask encoder와 DensePose 보조 과제를 통해 geometry latent가 ‘사람 영역·형상’에 더 민감하도록 학습시키고, 그 특징을 point-map/깊이 예측과 SMPL-X 디코딩에 함께 사용한다. 두 번째 과제는 사람 추정이 장면 기하와 실제 위치/스케일이 어긋나지 않게 정합성을 강제하는 것이며, joint fine-tuning에서 scale 일관성 정규화와 SMPL-Guided point-map alignment loss로 사람 표면과 점맵 기하를 공유 좌표계에서 맞춘다.

- **Empirical Impact**: 실험에서 SHOW는 카메라 공간 로컬 지표(MPJPE/PA-MPJPE/PVE)와 월드 공간 장기 모션 지표(W-MPJPE/WA-MPJPE/RTE) 전반에서 메트릭 스케일 일관성과 human–scene alignment을 개선했다고 보고한다. 특히 카메라 모션과 가림·클러터가 큰 설정에서도 성능 우위를 보이며, Human3R·UniSH 등 기존 결합/정렬 방식 대비 human–scene 일관성 지표(예: HS-CF, HS-V)에서 더 안정적인 정합을 보인다. 이 결과는 ‘사람의 의미론·스케일 priors를 장면 기하 학습에 직접 주입하고, 장면 기하를 사람 추정에 강한 제약으로 되돌리는’ 커플드 학습 패러다임이 단안 기반 grounded reconstruction의 핵심임을 시사한다.



### MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation (https://arxiv.org/abs/2606.27718)
Comments:
          Accepted in ECCV 2026

- **Prior Approaches**: 기존 VFI는 중간 프레임에 해당하는 중간 유동(flow)과 가시성 마스크를 추정한 뒤 워핑과 블렌딩/합성을 통해 결과를 만든다. 하지만 중간 프레임은 관측되지 않아 큰 변위, 가림·재출현, 얇은 구조, 반복 텍스처에서 대응(컨소던스) 불확실성이 커지며 워핑 기반 복원이 흔들린다. 한편 SSM 기반 VFI는 효율적이지만, 래스터 등 정적 그리드 스캔 순서에 의존해 물리적 운동 경로와 어긋나는 문맥 혼선을 겪는다는 한계가 있다.

- **Core Contribution**: 이 논문은 Motion-Aligned Selective Scan(MASS)을 제안해, 특징 스캔을 정적 공간 그리드가 아니라 flow-유도 모션 궤적(flow-guided trajectory)으로 옮긴다. 각 픽셀마다 유동을 따라 1D 시퀀스를 만들고 이를 SSM으로 집계함으로써, 중간 프레임 복원 시 문맥이 ‘운동이 일치하는 방향’으로 정렬되도록 설계했다. 또한 전·후방(0→t, 1→t) 집계 불일치를 기반으로 flow와 마스크를 end-to-end로 refinement해 코어 예측을 반복적으로 개선한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 실제 운동이 완전히 선형이 아니라 곡선 궤적을 따라야 하고, (2) 빠른 영역과 정적인 영역에서 샘플링/SSM 갱신 해상도를 효율적으로 달리해야 한다는 점이다. MASS는 잔차 velocity를 학습해 비선형 경로 적분을 근사하고, 모션 크기에 따라 시퀀스 길이(K)를 가변으로 설정해 빠른 변위 구간은 더 촘촘히 샘플링한다. 더 나아가 Velocity-Aware SSM으로 SSM의 discretization step size를 모션 속도에 반비례 조절해 고속 구간의 디테일(경계·텍스처) 손실을 줄였으며, 전·후방 컨텍스트 차이를 디스커페넌시 맵으로 만들어 중간 유동/마스크 오차를 보정한다.

- **Empirical Impact**: Vimeo90K를 기준으로 학습해 여러 벤치마크에서 평가했으며, 특히 SNU-FILM의 Hard/Extreme처럼 큰 변위와 가림이 많은 조건에서 성능 향상이 두드러졌다. Extreme split에서 MASS는 PSNR 26.04 dB를 기록해 기존 최고치 대비 0.38 dB 개선(정확한 수치 기준)했고, Xiph-2K/4K에서도 높은 PSNR로 경쟁 방법들을 앞섰다. 또한 경량형 MASS-S는 연산량을 크게 낮추면서도 품질을 유지해(약 0.50 TFLOPs) 리소스 제약 환경에서도 적용 가능성을 보여주며, 비주얼 결과에서 기존의 ghosting/블러를 줄이고 구조적 일관성을 강화했다.



### ZooClaw-FashionSigLIP2: Distilled Fine-tuning for Robust Fashion Retrieva (https://arxiv.org/abs/2606.27708)
Comments:
          ZooClaw Team

- **Prior Approaches**: 기존 비전-언어 인코더(CLIP 계열 포함)는 zero-shot 임베딩으로 강점을 보이지만, 패션처럼 미세한 시각·텍스트 차이가 중요한 영역에서는 일반화가 부족해 도메인 파인튜닝의 필요성이 커졌다. 다만 패션 데이터로 fine-tuning하면 in-domain 성능은 오르더라도 out-of-distribution(OOD) 일반화가 떨어지는 tradeoff가 반복적으로 관측된다. LoRA 같은 parameter-efficient 방법, 더 큰 백본 스케일링, 외부 학습데이터 혼합도 이 문제를 일관되게 해결하지 못했다.

- **Core Contribution**: 이 논문은 패션 특화 SigLIP2-base인 ZooClaw-FashionSigLIP2를 제안하며, in-domain 적응과 OOD 보존을 동시에 노리는 단순 레시피를 제시한다. 구체적으로는 curated in-domain 데이터로 지식 증류를 포함한 full fine-tuning을 수행한 뒤, WiSE-FT 방식으로 base 체크포인트와 가중치 보간(weight interpolation)을 적용한다. 또한 새 패션 retrieval 벤치마크 ZooClaw-Fashion을 공개하고, 널리 쓰이는 벤치마크의 구조적 편향을 드러내 완화하는 체계적 품질 분석과 re-evaluation(특히 Fashion200k pooled 평가)을 함께 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 짧은 키워드 쿼리와 길고 속성 풍부한 설명 쿼리가 공존하는 실전 조건에서, 파인튜닝이 임베딩을 너무 task-specific하게 밀어 OOD 성능을 훼손하지 않도록 학습을 설계하는 것이다. 저자들은 (1) soft relevance를 쓰는 Generalized Contrastive Loss(GCL)로 미세한 정답도를 반영하고, (2) short-query/long-query 두 가지 대비 학습을 멀티태스크로 결합했으며, (3) LwF 기반 regularization으로 이미지 인코더가 base로부터 과도하게 이탈하는 것을 억제했다. 마지막으로 WiSE-FT 보간 계수 α를 스윕해 base-파인튜닝 모델 사이의 최적 운영점을 찾고, 그 경로에서 overfitting 위험이 낮다는 점도 검증한다.

- **Empirical Impact**: 실험 결과 ZooClaw-FashionSigLIP2는 fair evaluation 하에 모든 벤치마크에서 LoRA, 더 큰 백본(최대 1B), external data 추가를 포함한 경쟁군을 압도하거나 동률로 제시된다. 특히 Fashion200k는 공개 ground truth가 caption-source 인스턴스 복구 편향을 갖고 있어, TREC 스타일 pooled re-evaluation과 LLM 기반 graded relevance로 재채점했을 때도 ZooClaw-FashionSigLIP2의 우위가 유지됨을 보였다. 더 나아가 외부 Marqo fashion 데이터를 섞으면 오히려 성능이 악화될 수 있음을 보여 “단순 데이터/스케일 추가”보다 분포 정렬과 OOD-aware 학습이 중요하다는 실증적 메시지를 강화하며, 모델 가중치와 평가 산출물을 오픈소스로 공개한다.



### Joint Transcription and Decryption of Images of Encrypted Handwritten Documents: A Comparison with the Traditional Pipelin (https://arxiv.org/abs/2606.27700)
Comments:
          Published at HistoCrypt 2026 (9th International Conference on Historical Cryptology). NEALT Proceedings Series Number 61. Tartu University Library. 10 pages

- **Prior Approaches**: 역사 암호 해독은 보통 이미지에서 암호 기호를 전사(transcription)한 뒤, 그 기호열을 바탕으로 복호화(decryption)하는 2단 파이프라인을 사용한다. 이 방식은 전사 단계의 작은 실수가 복호화로 그대로 전이되며, 전사를 사람이 하거나 고품질 라벨이 필요해 확장성이 떨어진다. 또한 합성 데이터 의존도가 높아 synthetic-to-real gap 문제도 함께 제기돼 왔다.

- **Core Contribution**: 이 논문은 전사를 거치지 않고 암호화된 원고 이미지에서 바로 평문을 생성하는 end-to-end 모델인 Direct Image Decryption을 제안한다. Copiale cipher를 대상으로, 전사 대신 이미지-평문 직접 매핑으로 전사 오류 전이를 줄이고 시각 정보 손실을 완화하는 것이 핵심 목표다. 단, 특정 치환 규칙에 대한 학습(cipher-agnostic이 아님)을 전제로 한다.

- **Technical Challenges**: Direct Image Decryption에서 가장 큰 기술적 난제는 ‘연속적인 시각 표현’ 위에서 복호화를 학습해, 중간의 discrete 기호 선택을 생략하는 구조를 안정적으로 최적화하는 것이다. 이를 위해 CRNN 기반 시각 인코더와 attention 기반 문자 디코더를 결합하고, 디코더 학습과 동시에 인코더를 end-to-end로 fine-tuning해 전사 오차 없이 전체 그래디언트 흐름을 유지한다. 아울러 Copiale-like 합성 데이터를 115,000 라인 규모로 생성해 대규모 학습이 가능하게 했으며, 노이즈·잉크 번짐·노화 흔적 같은 열화 증강으로 현실감도 확보했다.

- **Empirical Impact**: 합성 데이터에서는 Direct Image Decryption이 전통적 2단 파이프라인보다 대부분 지표에서 우수하며, 토큰 정확도는 1.1%p, WER은 약 49% 감소로 전사 병목 제거 가설을 지지한다. 분포 밖 데이터(Novalis)에서도 정확도 절대값 +6.3%p로 end-to-end 학습의 일반화 이점이 관찰됐다. 다만 실제 Copiale 원고에서는 전반 성능이 크게 떨어지는데, 이는 질적 합성-현실 차이보다 ‘실데이터 희소성’이 주 원인으로 분석되며(학습 데이터가 115k→20k로 줄면 정확도가 크게 하락), 이런 상황에서도 Direct Image Decryption이 11.8%p 개선을 보이며 전사 단계 제거의 실용적 가치가 확인됐다.



### Two-Stage Cross-Domain Cervical Abnormality Screening with Cytopathological Image Synthesis and Knowledge Distillation (https://arxiv.org/abs/2606.27678)
- **Prior Approaches**: 기존 연구들은 기관 간 도메인 차이를 줄이기 위해 knowledge distillation을 활용하지만, 얕은 특징과 깊은(semantic) 특징을 단일 레벨에서 정렬하거나 서로 독립적으로 다루는 경우가 많다. 또한 image-level 변환을 도입하더라도 패치 기반 번역의 경계 불연속이 tiling 아티팩트를 만들 수 있어 세포 형태를 훼손할 위험이 있다. 그 결과 출현 양상 차이뿐 아니라 병기 간 미세한 시각 차이(카테고리 애매성)까지 함께 악화되며 일반화 성능이 제한된다.

- **Core Contribution**: 이 논문은 크로스 도메인 자궁경부 세포 검출을 위한 2단계 프레임워크를 제안한다. 1단계에서 Unpaired Neural Schrödinger Bridge(UNSB) 기반의 Spatially-Continuous Unpaired Neural Schrödinger Bridge(SC-UNSB)로 공간적으로 연속적인 중간 도메인을 만들어 도메인 분포 이동을 완화한다. 2단계에서는 distillation 안에 dual-level feature alignment을 넣어 얕은 구조 특징과 깊은 의미 표현을 단계적으로 정렬함으로써 도메인 불변 지식 전이를 강화한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 초고해상도 세포 이미지에서 패치 분할로 인해 통계가 끊기며 경계 드리프트가 발생하는 문제와 (2) 특징 정렬 시 얕은 구조와 깊은 의미가 도메인에서 다르게 흔들리는 문제다. 이를 위해 SC-UNSB는 Instance Normalization의 패치 독립 통계를 대신해 픽셀 좌표에 연속적인 Dense Normalization을 적용하고, Schrödinger Bridge의 경로 탐색 공간이 공간 연속 함수로 제한되도록 설계해 tiling 아티팩트를 줄인다. 또한 LFA(Loose Feature Alignment)는 다중 스케일 low-pass로 얕은 주파수/구조 정보를 보존·정렬하고, CFA(Compact Feature Alignment)는 penultimate layer 특징을 embedding 공간으로 투영해 고수준 semantic 정렬을 수행하되 MMD와 MSE 기반 손실로 점진적 전이를 유도한다.

- **Empirical Impact**: CRIC(소스)–ComparisonDetector(타깃) 두 대규모 데이터셋의 엄격 크로스 도메인 평가에서 전체 프레임워크가 성능을 일관되게 개선했다. SC-UNSB로 생성한 중간 도메인 위에 RetinaNet을 학습하고 LFA/CFA를 결합한 경우, mAP 26.9%와 mAP50 45.8%를 달성해 최고 성능을 보였고 도메인 갭과 카테고리 애매성 완화 효과가 가장 크게 나타났다. 생성 품질에서도 FID/KID·NIQE·HIST 지표가 전반적으로 개선되어, 공간적으로 연속적인 통계 필드가 다운스트림 검출에 실질적으로 유리함을 뒷받침한다.



### Multi-Modal Conditioned High-Resolution Transformer for Urban Electromagnetic Field Map Prediction Download PDF (https://arxiv.org/abs/2606.27671)
- **Prior Approaches**: 기존 EMF 예측은 대부분 물리 기반 시뮬레이터(레이 트레이싱 등)에 의존해 정확하지만 계산 비용이 커 대규모 기지국 배치 탐색에 불리했다. 딥러닝 방식은 건물 레이아웃과 안테나 정보를 입력 채널로 단순 결합해 신호 강도 지도를 생성하는 접근이 많았고, 이 때문에 안테나의 방향성·방사 패턴 같은 서로 다른 모달리티의 특성이 특징 추출에 충분히 반영되지 못했다. 일부 연구는 directional 정보를 넣었더라도 “입력에서 더해주기” 수준에 머물러 task-specific conditioning이 약하다는 한계가 지적된다.

- **Core Contribution**: 이 논문은 건물 레이아웃, 안테나 스칼라 파라미터, 방사 패턴을 각각 다른 방식으로 조건(condition)하는 multi-conditioned dense prediction 프레임워크를 제안한다. 백본은 High-Resolution Transformer(HRFormer)로 구성하고, FiLM으로 안테나 파라미터를 모든 단계의 특징에 주입하며, cross-attention으로 1-D 방사 패턴 토큰을 깊은 단계의 공간 특징과 융합한다. 또한 송신기 기준의 거리·근접·방위 정보를 담은 transmitter-relative spatial channels을 추가해 공간 일관성 있는 추론을 지원한다.

- **Technical Challenges**: 핵심 난관은 서로 다른 모달리티를 단순 입력 결합이 아닌 “특징 추출 과정”에 구조적으로 주입하는 것이다. 이를 위해 안테나의 방향성 방사 패턴은 360도 토큰화 후 cross-attention으로 연결하고, 스칼라 파라미터는 FiLM의 scale/shift로 단계별 조절되도록 설계했다. 또 test-time augmentation(TTA)에서 좌표/방위가 바뀔 때 공간 채널을 단순 flip만 하면 거리·bearing 정보가 깨질 수 있어, 변환마다 transmitter-relative 공간 채널을 재계산하는 coordinate-consistent TTA로 해결했다.

- **Empirical Impact**: 시뮬레이션 기반 500×500 EMF 지도(총 768개, 3.5 GHz)에서 제안 모델은 test MAE 0.0461로, plain UNet 대비 25.2%, HRFormer-only 대비 31.8% 개선을 보였다. 또한 composite loss( masked L1 + MS-SSIM + focal L1 )가 개별 손실보다 모든 지표에서 우수했으며, coordinate-consistent TTA는 MAE를 추가로 6.3% 낮췄다. 추론 속도도 RTX 5090 기준 샘플당 19.8ms(대략 50 preds/s)로 물리 기반 시뮬레이션 대비 압도적으로 빠르며, 안테나 방향성에 따른 hotspot 위치 품질 향상까지 확인했다.



### Explainable AI for Biodiversity Monitoring and Ecological Image Analysis (https://arxiv.org/abs/2606.27667)
- **Prior Approaches**: 카메라 트랩·드론·위성·수중 플랫폼 등에서 얻는 생태 이미지에 대해 영상 분류·탐지·분할 모델을 자동화해 보전 평가의 규모와 속도를 키우려는 시도가 늘고 있다. 다만 기존 모델은 왜 맞혔는지 해석이 어렵고, 표본 편향·배경/형상 교란·우연 상관 같은 요인이 예측을 좌우해도 이를 검증하기가 힘들다. 그 결과 보전 의사결정에 쓰일 때 모델 근거가 생태적으로 타당한지 확인이 부족하다는 문제가 지적된다.

- **Core Contribution**: 이 논문은 explainable artificial intelligence(XAI)를 생태 모델 검증의 표준 구성요소로 두어, ‘정확도’뿐 아니라 ‘정확한 이유’를 생태 관점에서 확인하도록 제안한다. 또한 생태 컴퓨터 비전의 대표 3개 과제(영상 분류, object detection, image segmentation)에 XAI를 실무적으로 적용하는 가이드를 제공한다. 공중(항공) 이미지 사례를 통해 XAI가 모델 감사(auditing), 개선(refinement), 배포(deployment)로 이어지는 흐름을 구체화한다.

- **Technical Challenges**: XAI를 생태 데이터에 적용할 때는 설명이 실제로는 배경·윤곽·가림(occlusion)·경계(edge) 효과 등 ‘비생태적 신호’를 강조하는지 구분해야 한다는 기술적 난제가 있다. 논문은 이를 위해 과제별(분류/탐지/분할)로 설명 방법을 적용해 생물학적으로 의미 있는 단서가 포착되는지 점검하고, false positive가 어떤 교란에서 비롯되는지 식별하도록 한다. 더 나아가 설명 결과를 바탕으로 데이터 수집, 데이터 증강, 재학습 전략을 수정해 모델의 추론 정렬(alignment)을 유도한다.

- **Empirical Impact**: 항만물범(harbor seal) 탐지와 고래류(cetacean) 해부학 분할이라는 두 사례에서 XAI가 생물학적 단서를 확인하고 배경·형상 confound로 인한 오탐을 드러내는 효과를 보여준다. 또한 edge와 occlusion에 따른 성능/설명의 편향을 관찰해 추가 데이터와 학습 설계를 어떻게 바꿀지 연결한다. 전반적으로 모델 추론이 생태적 이해와 맞는지 과학적으로 점검하는 도구로서, 생물다양성 보전에 쓰이는 AI 증거의 신뢰성과 실행 가능성을 높이는 데 의미가 크다.



### MVPruner: Dynamic Token Pruning for Accelerating Multi-view Vision-Language Models in Autonomous Driving (https://arxiv.org/abs/2606.27660)
Comments:
          accepted by ECCV26

- **Prior Approaches**: 기존 token pruning은 주로 단일 카메라(단일-view)나 고정된 pruning rate를 가정해 view 간 기여도 차이를 반영하지 못하는 한계가 있었다. 또한 정적 중요도 지표나 단일-stage 전략에 의존해 추론이 진행되며 바뀌는 “필요 정보”의 동적 요구를 놓치기 쉽다.

- **Core Contribution**: 본 논문은 multi-view VLM이 레이어가 깊어질수록 instruction(지시)과 연관된 task-related view에 더 강하게 집중하며, 정보 요구가 다양성 기반에서 과업 적합성 기반으로 전환된다는 분석을 제시한다. 이를 바탕으로 MVPruner는 2-stage로 view별 pruning 예산 배분과 토큰 선택을 동적 정보 요구에 맞춰 조정한다. 1단계는 view 내부 feature 다양도를 보고 기본 문맥(semantic representational capacity)을 유지하고, 2단계는 instruction-시맨틱 관련도로 과업 정렬(task alignment)을 강화한다.

- **Technical Challenges**: 핵심 과제는 “어떤 view가 지금 단계에서 중요한가”를 고정 규칙이 아니라 추론 단계에 맞게 추정하는 것이다. 저자들은 (i) shallow에서는 view별 정보 다양도(예산 배분)와 cross-stage에서의 기여 일관성(토큰 선택)을 결합하고, (ii) deeper에서는 instruction-시맨틱 관련도 기반으로 view pruning 예산을 재할당한 뒤 instruction 유도 토큰을 상위 선택하는 방식으로 이를 해결했다.

- **Empirical Impact**: DriveLM, DriveLMM-o1, MAPLM, 그리고 비디오 벤치마크 STSnu에서 일관된 효율-성능 균형을 보이며 SOTA 대비 우수한 결과를 보고한다. 예를 들어 DriveMM에 MVPruner를 적용하면 FLOPs를 87.3% 줄이고 prefilling에서 4.97배 속도를 확보하면서도 DriveLM 정확도 98.5%를 유지하는 성과를 제시한다. 또한 offline 하이퍼파라미터 탐색 없이도 강건하게 동작하며, 프루닝 오버헤드는 전체 추론 지연 대비 소수(약 2.9%)로 관리되어 실사용 가능성을 높였다는 점에서 의미가 있다.



### GeoFace: Consistent Multi-View Face Generation with Geometry-Constrained Diffusion (https://arxiv.org/abs/2606.27659)
- **Prior Approaches**: 기존 단일 입력 기반 multi-view diffusion(또는 GAN·NeRF 계열)은 보기마다 RGB를 그럴듯하게 만들지만, 공통된 3D 얼굴 구조를 강제하는 장치가 부족해 시점이 달라질 때 지오메트리 일관성이 깨지는 문제가 자주 발생한다. 3DMM/FLAME 같은 파라메트릭 모델을 보조 신호로 쓰거나(예: head pose·normal map 등) cross-view attention에 의존해도, views 간 ‘전역 정렬’된 기하 정합을 보장하긴 어렵다. 특히 대각/프로파일 같은 큰 pose에서 코·턱·관자놀이 등 자가 가림 부위의 구조가 흔들리며 identity도 저하될 수 있다.

- **Core Contribution**: GeoFace는 단일 입력에서 multi-view RGB와 3D 얼굴 기하를 동시에 생성하되, appearance stream과 geometry stream이 shared attention layers로 상호 제약하도록 설계한 dual-stream diffusion 프레임워크다. 핵심은 geometry를 view-invariant한 canonical UV position map(FLAME 기반)으로 정의해 생성 과정의 ‘내재적 공통 기준선’으로 넣고, 두 스트림의 cross-attention이 3D 일관 대응을 갖도록 직접 감독한다는 점이다. 그 결과 시점이 바뀌어도 동일 인물의 특징(코, 턱선, 이마 등)을 더 안정적으로 보존한다.

- **Technical Challenges**: 가장 큰 기술적 난관은 geometry-appearance 두 모달의 attention 경로가 사전학습 중에는 대응 신호 없이 학습되어, 자연스럽게 3D-consistent 정렬을 보장하지 못한다는 점이다. GeoFace는 FLAME UV position map에서 얻는 3D-consistent correspondence로 geometry-guided attention alignment loss를 제안해 cross-attention(geometry→appearance, appearance→geometry)을 bidirectional로 정렬시킨다. 또한 canonical 공간의 geometry에 대해 표준 카메라 조건을 쓰면 잘못된 시점 가정이 생기므로, geometry stream에는 Plücker ray 대신 learnable token e_geo를 도입해 end-to-end로 카메라/위치 임베딩을 학습한다.

- **Empirical Impact**: RenderMe-360과 Nersemble v2 실험에서 GeoFace는 기존 appearance-only 기반 multi-view diffusion 대비 시각 품질과 cross-view geometric consistency가 동시에 개선되었고, identity consistency(CSIM)도 더 높게 나타났다. 개선 폭은 특히 큰 pose에서 더 커졌는데, appearance-only 모델이 약한 자가 가림 영역(코·턱·콧등 등)에서 geometry 제약의 효과가 두드러졌다. 또한 GeoFace가 함께 생성한 canonical geometry는 3D Gaussian Splatting의 초기화 prior로도 유효해 조기 수렴과 더 낮은 LPIPS를 보이며 downstream 3D reconstruction 효율을 높인 것으로 보고된다.



### Temporal-Emerged Prompting for Segment Anything in Multiframe Infrared Small Target Detection (https://arxiv.org/abs/2606.27655)
Comments:
          Accepted to the 43rd International Conference on Machine Learning (ICML 2026)

- **Prior Approaches**: 다중 프레임 적외선 소형 표적 탐지(M-IRSTD) 연구는 주로 (1) 시공간 특징을 직접 뽑는 순차 전용 아키텍처나, (2) 단일 프레임 탐지/분할 결과를 시간축으로 일관화하는 탐지 중심 패러다임으로 나뉜다. 특히 SAM 계열을 시퀀스로 확장한 방식은 메모리·연속성에 강점을 보이지만, 저신호대잡음비(SNR) 환경에서 표적이 프레임별로 배경과 구분되지 않아 prompt 확보가 어렵고 마스크 품질이 흔들린다는 한계가 있다.

- **Core Contribution**: 이 논문은 SAM에 시간축에서 점차 “드러나는(temporally emerged)” 단서를 명시적으로 주입하는 Temporal-Emerged Prompting for Segment Anything Model(TEP-SAM)을 제안한다. 핵심은 Discrepancy-Enhanced Temporal Encoder(DETE)로 전역 배경 동역학과 국소 운동 불일치를 동시에 모델링해 SAM 인코더 표현을 변조하고, Temporal Prompt Generator(TP-Gen)로 자동 시간 프롬프트를 만들어 비인터랙티브 분할을 수행한다.

- **Technical Challenges**: 가장 큰 기술 과제는 저SNR·미세 표적에서 표적-배경 분리 신호가 너무 약해, 일반적인 시간 집계나 메모리 기반 연속성만으로는 움직임 잡음에 쉽게 휘말린다는 점이다. 이를 해결하기 위해 DETE는 Global–Local Temporal Modeling(GLTM)으로 전역/국소 모션을 분해하고, 인접 프레임 대비 “운동 불일치”를 부각하는 Temporal Discrepancy Modulation(TDM)로 표적 관련 영역의 표현을 강화한 뒤, SAM 인코더는 frozen 상태로 Temporal Feature Injection(TFI)로 결합한다.

- **Empirical Impact**: 실험은 NUDT-MIRSDT와 TSIRMT에서 IoU/nIoU 및 Pd/Fa 같은 지표로 검증했으며, 특히 Hard subset처럼 극단적으로 낮은 SNR에서도 TEP-SAM이 기존 최첨단 대비 IoU에서 각각 2.92 및 8.45, Hard subset에서는 4.36 및 10.07 수준의 향상을 보였다. 정성 결과에서도 기존 방법의 오탐/누락이 잦은 상황에서 TEP-SAM은 더 조밀하고 일관된 경계 분할을 제공하며, “일반적 temporal aggregation만으로는 부족하고 temporally emerged cue를 정교하게 주입해야 한다”는 메시지를 실증했다.



### VLM-Aware Meta-Optic Front-End Design for Frozen Vision-Language Models (https://arxiv.org/abs/2606.27646)
Comments:
          18 pages, 6 figures, 3 tables

- **Prior Approaches**: 기존 메타옵틱 inverse design은 adjoint 시뮬레이션으로 효율, Strehl ratio, 이미지 fidelity 같은 광학 지표를 최적화하는 경우가 많다. 또한 end-to-end computational imaging은 광학과 복원/백엔드를 함께 학습해 성능 향상이 어디서 나오는지 분리가 어렵다. 반면 CODA가 겨냥한 “frozen VLM에만 맞추고 광학만 바꾸는” 설정은 상대적으로 덜 다뤄져 있었다.

- **Core Contribution**: 이 논문은 CODA(co-design of meta-optic front-ends with differentiable adjoints)라는 프레임워크로, 고정된 zero-shot CLIP 같은 frozen vision model의 분류 cross-entropy를 목표로 메타옵틱 밀도(continuous density)를 직접 최적화한다. 핵심은 clean 이미지 재구성·image signal processing·이미지 충실도 보조목적을 넣지 않고, 광학 설계가 VLM 입력 측에서 인식에 유리하도록 loss를 광학 시뮬레이션까지 역전파한다는 점이다. 따라서 “해상도/PSF 중심화” 같은 인간 친화 지표가 아니라 downstream 인식 목적과 정렬된 광학 설계를 제안한다.

- **Technical Challenges**: 곤란한 점은 Maxwell 기반 시뮬레이션과 VLM의 미분 그래프를 연결해 효율적으로 gradient를 얻는 것이다. CODA는 (1) FDTD로 파동장 응답을 계산해 PSF를 만들고, (2) line-scan 기반 센서-이미지 형성 모델로 VLM 입력을 합성한 뒤, (3) frozen VLM과 이미지 형성부에 대해서는 자동미분으로, PSF 경계 이후에는 adjoint Maxwell 해로 기울기를 분해해 메타옵틱 밀도에 대한 adjoint-gradient 업데이트를 수행한다. 또한 full 3D 광시야를 피하기 위해 동일한 센서 라인 위치에서 wavelength–angle 조건을 PSF로 표현해 계산 비용을 관리한다.

- **Empirical Impact**: 시뮬레이션 벤치마크(2D, ImageNet-100)에서 focal-concentration baseline 대비 CLIP ViT-L/14 zero-shot 정확도가 53.75%에서 65.41%로 +11.66%p 향상됐다. 더 나아가 광학을 다시 최적화하지 않고도 CLIP, SigLIP, DINOv2 조합에서 ImageNet-100/CIFAR-100/Food-101의 9개 평가 셀 모두에서 baseline보다 우수한 transfer 성능을 보였다. 분석 결과, PSF의 단순 국소 지표보다 VLM 임베딩 공간에서의 분리도와 downstream accuracy가 일관되게 개선되며, warm-start(초기 focusing 최적화 후 VLM loss로 전환)가 중요한 최적화 스캐폴드로 나타났다.



### CascadeOcc: Rethinking 3D Occupancy World Models with Cascaded VQ Representations (https://arxiv.org/abs/2606.27644)
Comments:
          Accepted to IEEE Signal Processing Letters (SPL), 2026

- **Prior Approaches**: 점유(occupancy) world model은 미래 주행 환경을 예측해 궤적 계획까지 연결하지만, 기존 방법들은 외부 모달리티나 LLM 같은 보조 지식에 크게 의존하는 경우가 많다. 또한 씬을 인위적으로 분할해 잠재공간을 나누면 동적 에이전트-환경 상호작용을 통합적으로 모델링하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 본 논문은 CascadeOcc라는 새로운 점유 world model을 제안하며, 외부 지식 없이도 occupancy 표현 자체의 구조적 계층성을 최대한 활용하도록 설계했다. 다단계 VQ 기반의 coarse-to-fine(거친-세밀한) 토큰 정교화를 공간 계층에 반영하고, TimeMixer로 이를 시간 축까지 확장해 dual-hierarchy를 구현한다.

- **Technical Challenges**: 과제는 (1) 복잡한 3D 장면에서 토큰 표현력을 키우면서 (2) autoregressive 예측이 스케일을 거치며 생기는 누적 오류를 줄이고 (3) 장기·단기 시계열 의존성을 동시에 잡는 것이다. 논문은 계층형 multi-scale VQVAE-v2로 전역 구조→국소 디테일까지 조건부 양자화/복원을 수행하고, TimeMixer의 temporal pyramid와 gated attention으로 장기-단기 동역학을 균형 있게 융합한다(soft-labeling으로 exposure bias도 완화).

- **Empirical Impact**: Occ3D-nuScenes에서 3D 점유 복원은 IoU 2.24%, mIoU 4.6% 향상, 4D 예측은 IoU 3.65%, mIoU 3.2% 향상을 보이며 시각적 예측 품질도 개선됐다. 또한 nuScenes 기반 계획 평가에서 L2 오차는 기존과 유사한 수준을 유지하면서 충돌률을 특히 낮춰 안전성 측면의 의미를 입증했으며, ablation에서 Cascade 및 TimeMixer 각 구성요소의 기여가 함께 확인된다.



### AI-Generated Image Recognition via Fusion of CNNs and Vision Transformers (https://arxiv.org/abs/2606.27637)
Comments:
          SOICT 2024

- **Prior Approaches**: 생성형 모델(예: GAN, diffusion)로 만든 합성 이미지는 실제 이미지와 점점 구분이 어려워져, 진위 판별 탐지가 필수 과제로 떠올랐다. 기존 탐지는 주로 리샘플링 아티팩트, JPEG 양자화, 그림자/스플라이싱 등 포렌식 신호나 주파수·공동출현 행렬, CLIP 같은 특징을 활용하는 방식이 많다. 다만 새로운 생성기/편집 조건에 대한 일반화가 약하거나, 분리는 잘해도 보정(calibration)이 떨어지는 문제가 반복되어 왔다.

- **Core Contribution**: 이 논문은 CNN과 ViT를 결합하는 fusion 전략으로 AI 생성 이미지 탐지 성능을 끌어올리는 접근을 제안한다. 특히 CNN(기본 EfficientNet v2)과 ViT(사전학습된 ai_vs_real_image_detection 백본)의 전역·국소 특징을 통합해 더 견고한 판별기를 만든다. 결합 방식은 concatenation과 linear combination 두 가지로 제시된다.

- **Technical Challenges**: 핵심 난제는 생성 이미지의 미세한 국소 흔적과 전역 구조 차이를 동시에 잡는 것이다. 저자들은 입력 전처리(리사이즈·회전·샤프닝·정규화)로 ViT 입력 정합성을 맞추고, CNN·ViT의 중간 특징을 fusion layer를 통해 결합해 상호 보완성을 확보한다. 또한 concatenation은 특징을 그대로 확장해 표현력을 키우고, linear combination은 가중치(예: EfficientNet 0.6, ViT 0.4)로 균형을 학습해 성능을 안정화한다.

- **Empirical Impact**: CIFAKE(Stable Diffusion 1.4 기반 합성 이미지 포함)에서 제안 모델은 최고 97.44% 정확도를 보고하며, 단독 EfficientNet 대비 소폭, 단독 ViT 대비는 큰 폭 향상을 보였다. Brightness를 HSV에서 Value 50% 감소시키는 강건성 실험에서도 fusion이 기존 CNN의 성능 급락을 완화해 95.29%까지 유지했다. 이는 다양한 생성/편집 환경에서도 탐지 신뢰도를 높일 수 있는 합성데이터 진위 검증 방향성을 제시한다.



### Denoising ICF Images with Multiplicative Uniform Noise: A Self-Supervised Study Based on the Log-Domain Noisier2Inverse Framework (https://arxiv.org/abs/2606.27635)
- **Prior Approaches**: Noise2Self와 Noisier2Inverse 계열은 self-supervised denoising에서 “픽셀 독립/주변으로부터의 복원” 같은 가정을 활용하지만, ICF 영상에서는 배경 주변에 강한 공간 상관 잡음이 존재해 성능이 급락할 수 있다. BM3D 같은 고전 필터 기반 방법은 학습이 없다는 장점이 있으나, Multiplicative Uniform처럼 잡음 모델이 비표준인 경우 정량 복원이 어려워진다. 특히 ICF는 이미지마다 잡음 파라미터가 달라 “고정 파라미터 가정”이 깨질 위험이 크다.

- **Core Contribution**: 이 논문은 Log-Domain Noisier2Inverse라는 self-supervised denoising 프레임워크를 제안하며, log-도메인 변환을 통해 Multiplicative Uniform 잡음을 additive 형태로 바꿔 학습 신호를 만든다. 또한 log-domain self-supervised loss 최소화가 변환된 도메인에서의 supervised learning과 동치임을 Theorem 1으로 엄밀히 증명한다. 더불어 ICF용으로 per-image JSON 기반 잡음 파라미터 로딩(Variant B)을 도입해 잡음 분포 일치 문제를 해결한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 진공 배경의 near-zero 픽셀에서 log 수치 불안정, (2) 잡음의 매우 높은 공간 상관(r=0.99), (3) 이미지별 잡음 파라미터 변동, (4) 학습 붕괴/불안정성이다. 논문은 log(y)에서 -inf와 exp 오버플로를 막기 위한 안정화(두 단계 ε 처리), per-image noise matching을 위한 JSON 로딩, 그리고 gradient clipping·early stopping·체크포인팅으로 학습 실패 모드를 완화했다. 또한 고정 Gaussian 근사(Variant A)는 실제 log-domain 잡음 분포(전부 음수 구간)와 맞지 않아 성능이 제한됨을 실험적으로 확인했다.

- **Empirical Impact**: 평가 결과, Variant B(per-image JSON Uniform)는 PSNR 21.41dB와 SSIM 0.8358을 기록하며 noisy baseline 1.95dB 대비 +19.46dB 개선을 달성했다. Noise2Self는 공간 상관 가정 위반으로 SSIM 0.0177 수준에 그쳤고, BM3D는 log-domain에서도 PSNR 4.47dB로 정량 복원이 크게 부족했다. 논문은 “잡음 모델 호환성”이 비표준 영상(예: ICF)에서 네트워크 구조보다 성능을 좌우한다는 실증적 메시지를 남기며, clean ground truth 없이도 유의미한 복원이 가능함을 보여줬다.



### Qwen-Image-2.0-RL Technical Repor (https://arxiv.org/abs/2606.27608)
Comments:
          16 pages, 6 figures, 1 table

- **Prior Approaches**: 기존 text-to-image와 image editing의 RLHF/후처리 접근은 보통 단일 보상 신호에 의존하거나, 시각 품질과 지시 따르기를 동시에 안정적으로 최적화하는 데 한계가 있었다. 특히 얼굴/신원 보존 같은 편집 제약은 reward 설계가 까다로워 성능이 들쑥날쑥해지기 쉽다.

- **Core Contribution**: 이 논문은 Qwen-Image-2.0 확산모델에 post-training 파이프라인인 Qwen-Image-2.0-RL을 적용해 시각 품질과 instruction-following을 함께 끌어올린다. task-specific composite reward models를 구성해 T2I에서는 alignment·aesthetics·portrait fidelity, 편집에서는 instruction-following 정확도와 face identity preservation을 동시에 겨냥한다.

- **Technical Challenges**: 핵심 난제는 신뢰할 만한 reward signal을 확보하는 것이며, 이를 위해 vision-language models을 pointwise scoring과 chain-of-thought 추론 방식으로 fine-tuning해 합성 보상 모델을 만든다. 그다음 GRPO 기반 on-policy RL에서 전학습 지식을 유지하기 위한 hybrid CFG, 프롬프트 품질을 위한 intra-group reward range filtering, 범주별 reward weight calibration을 넣어 학습 안정성과 목표 정합성을 확보한다.

- **Empirical Impact**: 평가 결과 Qwen-Image-Bench에서 전체 57.84점으로 base 모델 대비 +2.61 향상됐고, text-to-image arena Elo 1193(+78), image edit arena Elo 1349(+93)를 기록했다. 전반적으로 aesthetics, prompt adherence, editing 정확도에서 일관된 개선이 확인돼, 확산 기반 이미지 생성·편집에 RLHF와 OPD를 결합하는 실용적 설계안을 제시했다.



### Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding (https://arxiv.org/abs/2606.27596)
Comments:
          29 pages, 25 figures. Accepted by ICML 2026

- **Prior Approaches**: 기존 LVLM 환각 완화는 주로 학습 시 정렬(alignment)하거나 추론 시 개입(intervention)을 통해 언어 priors을 억제/시각 신호를 증폭하는 방식으로 이뤄져 왔다. 특히 많은 방법이 attention intensity assumption—시각 주의의 양(세기) 부족이 환각의 핵심이라는 가정—에 의존해 global magnitude를 조정한다. 하지만 저자 분석에 따르면 환각 출력에서 시각 attention 총량 분포는 정상과 크게 겹쳐, “얼마나 보나”만으로는 구조적 원인을 설명하기 어렵다.

- **Core Contribution**: 이 논문은 환각이 전역적 세기 문제가 아니라, 결정적(decision-critical) 생성 단계에서 특정 attention head가 시각 증거와 단절되며 언어 priors로 잠기는 ‘동적 구조 불일치’에서 비롯된다고 규명한다. 저자들은 이런 head를 risky mediator로 정의하고, 시스템 지시가 우회 경로의 앵커가 되어 물체 환각으로 이어진다고 설명한다. 이를 해결하기 위해 훈련 없이 추론 단계에서만 작동하는 Fox(Faithfulness and Observational-flow via eXpression-rectification)를 제안한다.

- **Technical Challenges**: Fox의 핵심 과제는 (1) 위험한 mediator를 감독 없이 정확히 찾고 (2) 고정된 모델 파라미터를 바꾸지 않으면서 그 경로를 ‘물리적으로’ 끊는 동시에 (3) 언어의 자연스러움은 유지하는 것이다. 저자들은 visual attention entropy probe로 시각 경로의 불확실성이 크면서 동시에 system prior-path가 강하게 작동하는 head를 joint risk score로 국소화한다. 이후 do-operator를 numerical logit saturation 형태로 구현해 risky mediator의 shortcut 경로를 절단하고, 마지막으로 conflict-gated cooperative decoding으로 observational 분포와 interventional 분포의 균형을 동적으로 맞춘다.

- **Empirical Impact**: 실험에서 Fox는 POPE/CHAIR/MME 및 GPT-4V 기반 정성 평가에서 기존 추론 개입 방법을 일관되게 능가하며, 예컨대 SID 대비 29.1%의 상대 CIC_I 감소와 SOTA급 성능 향상을 보고한다. 또한 evidence-dependent 하위 지표(예: Position, Color)에서 개선 폭이 크게 나타나, 단순 보수적 회피가 아니라 실제 시각 재검증이 유도되었음을 시사한다. 결론적으로 Fox는 환각의 원인을 ‘구조적 경로’로 겨냥해 faithfulness–detail trade-off를 함께 개선하는 접근으로 해당 분야에 의미 있는 전환점을 제시한다.



### CoIn: Comprehensive 2D-3D Inpainting with Gaussian Splatting Guidanc (https://arxiv.org/abs/2606.27584)
- **Prior Approaches**: 기존 3D scene inpainting은 NeRF를 직접 최적화하거나(implicit) 3DGS 기반으로 ‘3D-first’ 순서를 적용하는 방식이 많았습니다. 3D-first 계열은 다중 뷰에서 타깃을 정확히 분리하기 위해 세그멘테이션 마스크 의존도가 높고, 초기 pruning/참조 이미지 단일 사용에 치우친 방법들은 object removal에 편중되거나 시점 변화에서 일관성이 깨지곤 합니다. 한편 ‘2D-first’는 2D diffusion의 편집 유연성은 장점이지만, 뷰별 확률적 샘플링이 cross-view 불일치를 유발해 결과가 3D 복원에 부적합해지는 문제가 남아 있습니다.

- **Core Contribution**: 이 논문은 2D diffusion inpainting과 3D Gaussian Splatting(3DGS)을 다단계 일관성 파이프라인으로 연결한 CoIn을 제안합니다. 핵심은 양방향 정보 흐름을 설계해(2D→3D→2D) arbitrary-shaped mask에서도 object removal뿐 아니라 object insertion까지 처리하고, 뷰 간 기하·외관 일관성을 동시에 확보하는 것입니다. 이를 위해 Reference Adaptive GS(Ref-GS), GS 기반 Reference Feature Warping으로 에너지 기반 Consistency Loss Guidance(CLG), 그리고 Texture-Enhancing Discriminator(TE-D)로 고주파 질감을 보정합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 2D 인페인팅의 확률적 결과가 다중 뷰에서 기하적으로 서로 맞지 않아 3D 재구성을 망가뜨린다는 점입니다. 논문은 (1) Ref-GS에서 기준 뷰를 더 크게 반영하는 per-view 가중치와 인페인팅 영역 주변 anchor feature attention을 통해 3DGS가 기준 뷰에 정렬되도록 안정화하고, (2) CLG에서 3DGS 포인트클라우드로 feature를 warping해 확산(denoising) 단계 동안 참조 뷰와의 멀티뷰 consistency를 energy로 강제하며, (3) TE-D가 adversarial patch 학습으로 가이드로 인해 생기는 blur를 줄이도록 보완합니다.

- **Empirical Impact**: SPIn-NeRF 및 IMFine에서 정량 지표(LPIPS/PSNR/FID 및 마스크 내 m-지표)와 질적 결과를 통해 CoIn이 state-of-the-art 성능을 보였다고 보고합니다. 특히 세그멘테이션 마스크/바운딩박스 등 입력 형태와 시점 커버리지 조건이 다른 설정에서도 일관된 개선이 관찰되며, object removal과 insertion 모두에서 의미 있는 결과를 생성합니다. 단일 RTX 4090 기준 파이프라인이 장면당 약 2.5시간 수준으로, 학습 기반 3D 인페인팅 대비 효율성과 범용 편집성의 균형을 제시한다는 점에서 의미가 큽니다.



### Beyond Points: Spherical Distributional Part Prototypes for Interpretable Classification (https://arxiv.org/abs/2606.27582)
- **Prior Approaches**: 프로토타입 기반 신경망은 로컬 패치가 학습된 프로토타입과 유사한지를 근거로 분류해, Grad-CAM 같은 사후 설명보다 해석 가능성을 높이려는 흐름이다. 하지만 DINO/CLIP 계열의 normalized 임베딩 공간에서는 유사도가 주로 방향(각도)으로 결정돼, 의미 부품이 갖는 intra-class 변동성을 단일 점 프로토타입이 잘 표현하지 못한다. 그 결과 프로토타입 중복(더미 프로토타입 다발)이나 배경 잡음에 끌리는 불안정성이 발생해 설명 품질과 강건성이 함께 흔들린다.

- **Core Contribution**: 이 논문은 의미 부품을 점 벡터가 아니라 hypersphere 위의 vMF(von Mises-Fisher) 분포 성분들로 모델링하는 vMFProto를 제안한다. 각 프로토타입이 고유한 concentration(κ)을 학습해, 경직된 부품은 더 뾰족하게, 변형이 큰 부품은 더 넓게 변동성을 흡수하도록 설계했다. 또한 OT(Entropic Optimal Transport)를 이용해 패치-프로토타입 배정을 구조적으로 강제하고, 끝단에서는 end-to-end 미세조정과 패치 수준 distillation로 정합성을 끌어올린다.

- **Technical Challenges**: 가장 큰 난제는 normalized·방향 중심 공간에서 “프로토타입이 변동 범위를 어떻게 대표하느냐”와 “학습 중 프로토타입 붕괴를 어떻게 막느냐”의 동시 해결이다. 저자들은 두 단계 학습으로 먼저 OT-driven prototype discovery로 프로토타입을 안정적으로 찾고, 이후 end-to-end 단계에서 OT 배정을 teacher로 삼는 patch-level distillation과 vMF 성분 간 중복을 줄이는 distribution-aware diversity 정규화를 함께 사용한다. 더불어 라벨 없이 foreground gating(attention→PCA refinement)을 넣어 배경 패치가 프로토타입 학습을 오염시키지 못하게 했다.

- **Empirical Impact**: CUB-200-2011, Stanford Dogs, Stanford Cars에서 DINO 백본을 frozen으로 두고 평가한 결과, vMFProto는 설명 지표인 consistency, stability, distinctiveness에서 SOTA급 성능을 보이며 정확도도 경쟁 수준을 유지한다. 특히 점 프로토타입 계열은 다양한 백본에서 일관성이나 고유성(distinctiveness)이 떨어지는 경향이 관찰되지만, vMFProto는 비슷한 설정에서도 더 국소적이고 중복되지 않은 부품 근거를 제공하는 정성 결과가 제시된다. 이는 “점 기반 해석”의 한계를 방향 기하에 맞춘 분포 기반 프로토타입과 OT 구조화로 완화할 수 있음을 실증적으로 보여준다.



### Distribution-based deep multiple instance learning for tumor proportion scoring in NSCLC (https://arxiv.org/abs/2606.27579)
- **Prior Approaches**: 기존 연구는 PD-L1의 TPS를 슬라이드 연속값으로 예측하거나, 임상 기준선에 따라 <1·1–49·50–100 같은 클래스 분류로 나눠 접근해왔다. 특히 weakly-supervised multiple-instance learning(MIL)은 슬라이드 레이블만으로 학습 가능하지만, TPS가 0에 해당하는 non-expressive(zeroclass) 이미지에서 성능이 급격히 떨어지는 문제가 지적돼 왔다. 또한 시각적 attention은 흔하지만 예측의 불확실성을 모델 구조에 내장해 정량화하는 방법은 제한적이었다.

- **Core Contribution**: 이 논문은 슬라이드 레이블 TPS만 사용해 end-to-end로 TPS 분포를 예측하는 프레임워크를 제안한다. 패치 임베딩을 먼저 추출한 뒤 MIL로 집계하되, 예측값을 단일 회귀로 내지 않고 zero-inflated beta(ZIBeta) 분포 파라미터로 모델링해 TPS=0일 확률과 분포 농도(precision 관련)를 함께 산출한다. 이를 통해 zero-class 정확도와 예측의 설명/불확실성 표현을 동시에 강화하려는 것이 핵심이다.

- **Technical Challenges**: 가장 큰 기술적 도전은 non-expressive 샘플(zeroclass)에서 회귀/분류 기반 모델이 쉽게 붕괴하는 점이다. 저자들은 ZIBeta로 TPS=0의 “점(0)”과 0이 아닌 연속 구간을 분리해 음의 로그우도(NLL)와 MSE를 결합한 손실을 설계했고, ZIBeta의 이진 성분 ϕ를 위한 threshold는 검증셋에서 Cohen’s kappa를 최대화하는 방식으로 정했다. 추가로 attention 기반 집계와 패치 엔트로피 최소화(균질성 가정)를 통해 잡음 패치를 걸러내도록 학습을 구성했다.

- **Empirical Impact**: 실험에서 ZIBeta 기반 MIL(ZIBeta 변형)은 선형/리지 회귀 및 다른 deep MIL 대안들 대비 전반 지표에서 상위 성능을 보였고, 특히 0 클래스 정확도와 관련된 평가에서 유리했다. 또한 beta 분포의 농도(ν)가 높을수록 절대 오차가 감소하는 경향을 관찰했으며, 동시에 WSI 간 이질성 때문에 단일 농도 값만으로는 신뢰도 프록시로 삼기 어렵다는 점도 확인했다. attention이 실제로 염색된 종양 패치와 진단 관련 영역에 더 집중하며 아티팩트 영향을 줄인다는 정성적 설명 가능성도 제시해, 임상 워크플로우에서의 검증/신뢰성 제공에 의미가 있다.



### DeLux: Cross-Modal Local Artifact Restoration in Video Using Neuromorphic Data (https://arxiv.org/abs/2606.27576)
- **Prior Approaches**: 기존 RGB 복원 방법은 플레어, 글레어, 플리커, 과노출 같은 조명 잡음을 각각 따로 다루는 경향이 있어, 복잡한 공간적으로 분리된 열화로 가려진 구조적 디테일을 완전 복구하기 어렵다. 이로 인해 일부 정보는 다시 얻기 불가능한 수준으로 손실되고, 결과적으로 computational restoration 의존도가 커진다.

- **Core Contribution**: 이 논문은 RGB와 event stream(뉴로모픽 이벤트 스트림)을 함께 쓰는 cross-modal restoration 패러다임을 제안하며, 이를 모듈형 파이프라인 DeLux로 구현한다. DeLux는 이벤트를 structural prior(구조적 선행지식)로 활용해 조명 아티팩트를 targeted detection 한 뒤 필요한 영역만 inpainting 하도록 설계됐다.

- **Technical Challenges**: 핵심 과제는 이벤트가 제공하는 구조 정보를 RGB 열화 상황에서 어떻게 신뢰성 있게 결합해, 국소 아티팩트와 복원 대상 경계를 정확히 분리할지에 있다. 논문은 이벤트 스트림으로 구조적 기준을 세운 뒤 모듈형 탐지-복원 절차로 플레어·글레어 등 서로 다른 아티팩트 유형을 겨냥하며, 부정확한 복원으로 인한 과보정과 번짐을 줄이도록 구성했다.

- **Empirical Impact**: 합성 벤치마크와 실제 자동차 영상으로 검증한 결과, DeLux는 국소 아티팩트를 효과적으로 억제하고 손상 영역을 복원해 기존 RGB-only baseline 및 event-guided HDR 모델보다 성능이 높다. 모든 아티팩트 유형에서 평균 MS-SSIM 0.99 이상을 달성했고, 실제 자동차 영상에서 아티팩트 severity를 최대 88%까지 낮췄으며, 합성 생성 도구와 평가 데이터셋을 공개해 후속 연구를 촉진한다.



### Perceptual 3D Simulation With Physical World Modeling (https://arxiv.org/abs/2606.27575)
Comments:
          Published as a conference paper at CVPR 2026

- **Prior Approaches**: 기존 NVS(소스 이미지+목표 포즈로 새 시점 생성)와 diffusion 기반 편집은 그럴듯한 결과를 만들지만, 카메라 제어와 3D 변환 신호를 유연하게 통합하는 데 한계가 있었다. 또한 object manipulation 쪽은 2D 드래그나 depth-conditioned 생성이 많았지만, 입력 이미지의 latent inversion에 의존하거나 복잡한 실세계에서 일관된 기하 정합을 유지하기 어려웠다. World model 연구도 멀티모달·확률적 예측을 확장하고 있으나, 부분 관측에서의 기하 일관성과 임의 3D 변환 조건을 동시에 강하게 강제하긴 부족했다.

- **Core Contribution**: P3Sim은 “부분 관측 + 불완전한 3D 변환 신호” 하에서 미래 장면 상태를 시뮬레이션하는 통합 프레임워크를 제안한다. 핵심은 (1) 다중 모달 장면 변수를 확률적 추론으로 다루는 학습형 physical world model, (2) partial 3D transform 신호를 만드는 geometrizer, (3) 예측을 누적해 일관성을 유지하는 persistent scene memory의 결합이다. 이를 통해 novel view synthesis, object manipulation, 동적 장면 예측을 단일 생성·추론 체계로 연결한다.

- **Technical Challenges**: 가장 큰 난제는 불완전한 기하/변환 정보로부터 불확실성을 포함한 미래 분포를 추정하는 “부분 추론”을 안정적으로 구현하는 것이다. P3Sim은 scene 변수 집합을 그래프 모델 관점의 조건부 분포 예측으로 정의하고, pointer 기반 autoregressive transformer로 추론 가능한 형태로 재구성해 임의의 conditioning 조합에 대해 누락 변수를 샘플링한다. 여기에 geometrizer가 부분 depth와 optical flow로 물리적으로 정합되는 변환 단서를 만들고, scene memory가 시간에 따라 관측과 예측을 병합하며 모순되는 미관측 공간을 제거하는 방식으로 일관성을 확보한다.

- **Empirical Impact**: 실험은 대표적으로 novel view synthesis는 SEVA, 3D object manipulation은 3DEditBench에서 정량 평가했으며, 전체적으로 PSNR·LPIPS 및 Edit Adherence(EA) 지표가 경쟁 방법 대비 우수하게 나타났다. 이는 카메라 제어가 더 안정적이고, 의도한 3D 변환에 더 충실한 편집/생성이 가능함을 시사한다. 결과적으로 P3Sim은 불완전한 관측과 제어 신호에서도 “지속적·일관된” 3D 장면 이해/변환으로 확장될 수 있는 물리적 world model 방향성을 보여준다.



### Radar Guided Camera Verification for Automatic Emergency Braking Rethinking Object Detection in Radar Camera Fusion (https://arxiv.org/abs/2606.27556)
Comments:
          8 pages, 8 figures

- **Prior Approaches**: 기존 radar–camera fusion AEB는 레이더로 표적을 찾은 뒤 카메라가 deep learning 기반 object detection으로 장애물을 “찾고(위치)” “무엇인지(라벨)”까지 인식하는 흐름이 주류였습니다. 이 방식은 성능을 올렸지만 계산량과 하드웨어 요구가 커지고, 카메라가 전 프레임을 대상으로 탐색해야 한다는 가정이 부담이 됩니다.
또한 레이더가 이미 위치를 제공하는 radar-led 파이프라인에서는 카메라의 역할이 검증(verification)으로 축소될 수 있다는 문제의식이 제기돼 왔습니다.

- **Core Contribution**: 본 논문은 레이더가 투영한 이미지 ROI에서 장애물 유무만 판단하도록 하는 radar-scoped edge density gate를 제안합니다. 카메라는 객체 인식 대신 “장애물 존재 확인”을 수행하며, 학습 데이터·모델 가중치·GPU 가속 없이 동작하도록 설계됐습니다.
이를 brake-by-wire가 포함된 완전한 radar–camera fusion AEB 시스템에 통합해 실제 차량 주행에서 평가했습니다.

- **Technical Challenges**: 핵심 과제는 레이더-카메라 투영 ROI에 작은 투영 오차가 생겨도 검증 성능과 AEB 안전성이 유지되도록 하는 것입니다. 논문은 CAN 기반 레이더 추적(Kalman filter, 다중 업데이트 확인) 후 핀홀 모델로 ROI를 만들고, ROI에서 Canny edge를 계산해 edge density가 임계값을 넘으면 장애물로 “게이트 통과”시키는 단일 스칼라 임계값 전략을 사용했습니다.
구현 측면에서는 ROI 크기를 줄여 처리 시간을 확보하면서도, 고정 임계값으로 충분한 recall(미탐 최소화) 지점을 유지하는 균형을 맞췄습니다.

- **Empirical Impact**: 실측 실험에서 ROI 기반 처리는 전체 프레임 대비 탐색 범위를 최대 98.7% 줄였고, ROI당 평균 지연이 0.121 ms로 측정됐습니다. 검증 성능은 AUC 0.898, recall 0.994로 보고되며, staged threat 시나리오 33개에서 missed brake event는 0건이었습니다.
추가로 72개 주행 세션/131,603 프레임 기반 결과로, detector 기반 confirmation을 대체할 수 있는 경량 카메라 검증 접근의 가능성을 실증한 것으로 평가됩니다.



### Understanding Cross-Rig Generalization in Automotive Perception: a Multi-Rig Benchmark and Rig Variation Metrics (https://arxiv.org/abs/2606.27554)
Comments:
          Accepted at ECCV 2026; Project Page: this https URL

- **Prior Approaches**: 자율주행 카메라 기반 인지 연구는 보통 카메라 개수·위치·방향·시야각(FOV)이 고정된 센서 리그에서 학습/평가됩니다. 하지만 실제 차량은 리그 구성이 이질적이라, cross-rig domain gap 같은 문제는 존재하지만 기존 벤치마크는 리그 변화와 장면 통계·외관 편차를 함께 섞어 분석하기 어렵습니다.

- **Core Contribution**: 이 논문은 동일한 주행 장면을 14개의 설계된 CARLA 카메라 리그로 렌더링해, 장면 변화 없이 기하학적 관측 차이만 분리한 Plentiful CARLA Camera Rigs를 제안합니다. 또한 리그 메타데이터로부터 Rig Variance(한 리그 내부 이질성)와 Rig Contrastive Distance(두 리그 간 기하 차이)를 정의해, 모델 무관하게 리그 특성을 정량화합니다.

- **Technical Challenges**: 핵심 기술적 난제는 시뮬레이션에서 장면 동학과 객체/라벨을 동일하게 유지하면서 카메라 외부/내부 파라미터만 바꿔 ‘순수한 기하 효과’를 만드는 것입니다. 논문은 비결정적 traffic manager의 궤적을 사전 기록·재생하고, 리그별로 동일 장면을 재현한 표준화 데이터 포맷을 구축해 직접적인 교차 리그 평가가 가능하도록 했습니다.

- **Empirical Impact**: 실험 결과, 리그 기하 차이는 교차 리그 성능 변화를 강하게 유발하며, 특히 Rig Contrastive Distance가 리그 간 transfer 난이도 순위를 안정적으로 설명하는 대리 지표로 작동합니다. 다만 아키텍처마다 민감도가 달라 BEVDet/BEVFusion/PETR은 상관이 높지만 Fast-BEV는 예외적으로 설명력이 약해, ‘기하 shift→성능’의 메커니즘이 모델 설계에 따라 달라짐을 보여줍니다.



### Beyond MoCap: Scaling Motion Tokenizers with Synthetic Human Motion for Generative Modeling (https://arxiv.org/abs/2606.27547)
- **Prior Approaches**: 기존 human motion generation은 VQ-VAE 기반 discrete latent 토큰과 autoregressive transformer 같은 생성기를 써서 text-to-motion, motion continuation을 잘 수행해 왔습니다. 하지만 토크나이저가 주로 MoCap 데이터(Human3.6M, AMASS 등)에 의존해 학습되는 탓에 공통 동작에 편중되고 long tail의 희귀·복합·고난도 동작을 충분히 담지 못해 motion vocabulary가 제한됩니다. 그 결과 모델 용량을 키워도 토크나이저의 표현 지원 범위를 벗어나는 동작은 생성이 어렵습니다.

- **Core Contribution**: 이 논문은 synthetic human motion으로 motion representation space를 확장하고, 그에 맞춰 redesigned VQ-VAE 토크나이저를 학습하는 프레임워크를 제안합니다. kinematics·dynamics·contact consistency 제약을 포함해 physically plausible한 모션을 만들면서, MoCap 분포 밖의 희귀·극단·조합 동작을 체계적으로 생성해 토큰 사전의 커버리지를 넓힙니다. 또한 확장된 분포에 맞춰 코드북과 학습 분포를 함께 스케일링해 더 풍부한 motion primitives를 학습하게 합니다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 희귀 구성을 포함하되 사람이 실제로 만들 수 있는 자세·동작을 유지하는 것과 (2) 커진 데이터 분포에 대해 코드북 collapse/under-utilization 없이 토크나이저가 안정적으로 작동하도록 하는 것입니다. 이를 위해 kinematic tree 기반의 hierarchical pose 표현에서 crossover와 mutation으로 long tail을 탐색하고, pose prior로 무효 자세를 걸러 물리적 타당성을 확보합니다. 이후 reconstruction/commitment 손실에 더해 EMA 업데이트와 codebook reset을 쓰고, real+synthetic 혼합 학습 및 코드북 크기 확대로 확장된 공간을 더 촘촘히 분할하되 적절한 작동 구간을 찾아 성능을 끌어올립니다.

- **Empirical Impact**: 실험은 synthetic 데이터가 HumanML3D 밖의 방향 공간까지 실제로 확장(커버리지 증가)하되 물리적으로 유효함(유효 자세 67.33% 유지)을 만족함을 먼저 보여줍니다. 그 다음 토크나이저 학습 시 in-distribution에서 FID가 유의하게 감소하고(R-Precision, MM-Dist, Diversity 등도 개선), out-of-distribution인 Motion-X++에서도 MPJPE가 모든 태스크에서 낮아져 일반화 이득을 입증합니다. 더 나아가 architecture 변경 없이 기존 discrete-token 생성기들에 토크나이저만 교체·재토크나이징 방식으로 fine-tuning했을 때 text-to-motion과 motion continuation 전반에서 다양성·구성성·동적 일관성이 일관되게 향상되어, 병목이 모델 구조보다 learned motion representation 지원 범위에 있음을 시사합니다.



### MemoBench: Benchmarking World Modeling in Dynamically Changing Environments (https://arxiv.org/abs/2606.27537)
- **Prior Approaches**: 기존 비디오 생성 평가는 주로 연속 프레임에서의 시각 품질, 시간적 일관성, 물리 준수, 장면 일관성에 초점을 맞췄습니다. 메모리 관점에선 ‘대상이 보이는 동안만’ 연속성을 보거나, 가림이 발생해도 장면 변화가 없는 정적 케이스에 머무는 경우가 많아 객체 영속성과 상태 업데이트를 동시에 검증하기 어려웠습니다. 결과적으로 생성 모델이 시야에서 사라진 뒤 되돌아오는 객체의 동일성과 변화 상태를 실제로 기억하는지 불명확했습니다.

- **Core Contribution**: 이 논문은 disappear-and-reappear 패러다임을 전면에 둔 진단 벤치마크 MemoBench를 제안합니다. 목표 객체가 물리 과정을 진행하며 보이다가 사라지고, 카메라가 돌아왔을 때 ‘업데이트된 상태’를 다시 정확히 복원하는지로 메모리 일관성을 직접 측정합니다. 합성+실세계 360개 1920×1080 클립과 함께 카메라 궤적 및 깊이 정보를 제공해 기하와 상태 진화를 함께 평가하도록 설계됐습니다.

- **Technical Challenges**: 핵심 기술 과제는 시야가 가려지는 D 구간 동안 모델이 객체 상태를 ‘유지·갱신’해야 하는데, 이를 단일 화소 비교나 마스크 유사도로는 신뢰성 있게 판정하기 어렵다는 점입니다. 연구진은 SAM-3 기반의 Object Reappearance Score(ORS)로 재등장 여부와 인식 가능성을 검출하고, PSNR/SSIM/LPIPS 및 DINOv2 토큰 유사도, Depth Anything V2 기반 Geo3D 일관성 등으로 정밀한 품질·정체성·구조를 분해 측정했습니다. 또한 LLM-judged VQA를 Instruction Following, Object & Background Consistency, Continuity of Memory, Physics Adherence의 4차원 진단으로 구성해 자동지표의 맹점을 보완했습니다.

- **Empirical Impact**: 10개 SOTA 월드 생성 모델을 MemoBench에 적용한 결과, 전반적으로 가림 후 재등장에서 객체 메모리를 일관되게 유지하는 모델은 없었고, 특히 Continuity of Memory에서 성능 격차와 공통 실패 양상이 관찰됐습니다. CI2V(카메라-조건) 모델은 전반 화질과 카메라 제어 측면에서 유리하지만, 여전히 ORS와 메모리 연속성 지표가 안정적으로 높지 않아 ‘보이는 동안 그럴듯함’과 ‘사라진 뒤 갱신’의 간극이 드러났습니다. 연구는 향후 end-to-end world state 유지, 가림 구간에서의 상태 추적·업데이트 메커니즘 개선이 필요함을 명확히 하는 신호로 평가됩니다.



### Large Language Model Teaches Visual Students: Cross-Modality Transfer of Fine-Grained Conceptual Knowledg (https://arxiv.org/abs/2606.27527)
Comments:
          Accepted by ICML 2026

- **Prior Approaches**: 기존 지식 증류(KD)는 주로 같은 데이터에서 학습된 비전(또는 텍스트-비전) 교사로부터 logits나 특징을 직접 모방하도록 설계돼 왔다. 하지만 파인그레인드 분류처럼 미세한 차이를 요구하는 과제에서는 시각적 교사 기반 학습이 배경 같은 우연 신호(스퓨리어스 상관)에 더 휘둘릴 수 있다는 한계가 있었다. 또 멀티모달 KD는 정렬된 입력이나 멀티모달 교사가 필요해 비용과 편향 전이가 부담이 되곤 한다.

- **Core Contribution**: 이 논문은 Language-to-Visual Knowledge Distillation(LaViD)로, 이미지 입력이 전혀 없는 language-only LLM을 교사로 사용해 비전-only 학생을 지도하는 프레임워크를 제안한다. LaViD는 클래스 간 의미 차이를 찌르는 multiple-choice questions(MCQs)를 LLM에서 생성하고, 각 클래스에 대해 질문별 soft label 분포(개념 서명)를 만들어 학생의 보조 증류 손실에 활용한다. 그 결과, 짝지어진 멀티모달 데이터 없이도 언어의 고수준 개념을 시각 표현 학습으로 전이할 수 있음을 보인다.

- **Technical Challenges**: 핵심 기술 난제는 ‘텍스트 교사가 이미지 없이도 유의미한 시각 개념 관계를 제공’하도록 구조화된 감독 신호를 설계하는 것이다. LaViD는 데이터 메타데이터와 클래스 이름을 바탕으로 LLM이 시각적으로 근거 있는 MCQ를 생성하게 하고, 각 답지의 pre-softmax logits를 추출해 클래스별 Q×M 시그니처로 고정한다. 학생은 이미지로부터 보조 헤드를 통해 같은 질문 공간에서의 예측을 만들고, 분류 손실에 더해 LLM 타깃과의 MSE 증류 손실을 함께 최적화하며 관계적 구조를 내재화한다.

- **Empirical Impact**: 6개 파인그레인드 벤치마크와 ImageNet 부분 서브셋에서 LaViD는 MaKD 같은 멀티모달 교사 기반 최신 방법을 언어-only 교사만으로 일관되게 능가한다. 또한 DKD, MLKD 같은 전통적 비전 교사 기반 KD와 경쟁하거나 더 나은 성능을 보이며, logit standardization과 결합 시 추가 개선도 확인됐다. Waterbirds에서는 worst-group accuracy가 크게 향상돼, 증류가 스퓨리어스 상관에 덜 의존하게 만드는 견고성 개선 효과가 관찰된다.



### Tessellating The Earth (https://arxiv.org/abs/2606.27514)
Comments:
          European Conference on Computer Vision -- ECCV 2026

- **Prior Approaches**: 기존 지리 좌표 인코더는 위도-경도 좌표를 고정된 기준선(예: 구면조화, Fourier features, random Fourier features)에 투영한 뒤 이미지 임베딩과 대비학습으로 정렬하는 방식이 주류였다. 그러나 고정 기저는 공간 복잡도가 높은 해안·도시처럼 필요한 구간에 역량을 자동으로 집중시키지 못해, 바다(약 71%의 구면)처럼 정보가 제한적인 영역에도 표현력이 균등 배분되는 문제가 컸다. 저주파 편향까지 겹치며 미세한 지리 변화를 충분히 담기 어렵다는 한계가 반복 확인됐다.

- **Core Contribution**: TTE(Tessellating the Earth)는 고정 기저를 대체해 ‘학습 가능한 Spherical Voronoi’ 분할을 통해 좌표 인코더의 표현 용량을 데이터에 맞게 재배치한다. 각 Voronoi 사이트는 자신만의 임베딩을 가지며, 대비학습 중에 분할 사이트 자체가 더 분별력 있는 영역으로 이동(migrate)하도록 end-to-end로 최적화된다. 여기에 전역 의미 토큰(global semantic tokens)을 추가해, 위성 이미지에서 압축된 개념 어휘를 만든 뒤 추론 시에도 위치 인코더가 참고할 수 있게 함으로써 멀리 떨어진 지역 간 의미 공유를 가능케 했다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 구면에서 지역별 국소 구조를 표현하되, (2) 고정 분할이 아니라 학습 과정에서 사이트가 ‘유용한 곳’으로 이동하도록 미분가능한 설계를 만드는 것이었다. TTE는 Spherical Vorononoi의 soft assignment(온도 τ로 파티션 선명도 조절)를 임베딩 결합에 적용해 모든 사이트에 그래디언트가 흐르게 했고, 사이트 위치·온도·임베딩을 모두 함께 학습했다. 또 (3) 로컬 Voronoi 셀 간 의미 단절을 해결하기 위해, 이미지 경로와 위치 경로가 동일한 개념 토큰에 attention하되 이미지 쪽은 더 뾰족한(거의 one-hot) 타깃을 만들어 위치가 좌표만으로 개념 분포를 맞히도록 재구성(loss recon)과 정렬(loss align) 손실을 조합했다.

- **Empirical Impact**: 실험에서는 위성 기반 대비학습으로 사전학습한 뒤 선형 프로브로 다양한 지리 분류·회귀 벤치마크를 평가했으며, TTE는 대부분의 개별 과제에서 SOTA를 기록했다. 평균 분류 정확도는 2.8%p, 회귀 R2는 0.045 향상되었고, 특히 iNaturalist-2018에서 미세 종 분류의 geographic prior로 사용했을 때 가장 강한 성능을 보였다(Top-1 66.1%→76.2%). 어블레이션 결과로는 사이트를 고정하면 큰 성능 하락이 나타나 ‘학습 중 사이트 이동’이 실제 병목임을 시사했고, 전역 의미 토큰 제거도 전반적으로 가장 큰 손실을 유발해 의미 공유의 중요성이 확인됐다.



### Structured-Li-GS: Structured 3D Gaussians Splatting with LiDAR Incorporation and Spatial Constraints (https://arxiv.org/abs/2606.27509)
Comments:
          9 pages, ISPRS Congress 2026

- **Prior Approaches**: 기존 3DGS 계열은 SfM의 점들을 기준으로 가우시안 타원을 학습해 렌더링 품질은 높이지만, 타원 가정 한계로 얇은/표면형 구조에서 기하 정확도가 떨어질 수 있다. 또한 포토메트릭 손실이 주로 희소 supervision에 의존하고, 성능을 끌어올리기 위해 densification을 수행하면서 가우시안 수가 급증해 모델이 커지는 문제가 반복된다. LiDAR를 섞은 연구도 입력 점밀도를 늘리거나(사후 densification) SLAM-내 통합형은 포즈 추정 불확실성으로 정합이 흔들리는 경우가 있어, 적은 크기로 정확도를 유지하기가 쉽지 않았다.

- **Core Contribution**: 이 논문은 LiDAR-inertial-visual SLAM의 조밀한 포인트프리오어를 활용해 3DGS를 더 “라이트(lite)”하게 만드는 Structured-Li-GS를 제안한다. 하드 제약은 앵커 기반 초기화와 normal-assisted 초기화에 두고, LiDAR에서 얻은 조밀·컬러라이즈된 포인트 클라우드를 기준으로 가우시안 타원을 표면 기하에 정렬해 더 적은 Gaussians로도 고품질 재구성을 노린다. 무엇보다 densification 없이도 photometric뿐 아니라 flattening, offset, depth, normal까지 함께 학습하도록 학습식을 설계해 컴팩트함과 정밀도를 동시에 노린다.

- **Technical Challenges**: 핵심 난제는 (1) LiDAR 기반 조밀 포인트를 3DGS의 가우시안 파라미터(타원 스케일/회전/오프셋)로 안정적으로 매핑하고, (2) 포즈·깊이·법선 정보가 렌더링 학습 손실로 어떻게 전달돼야 하는지였다. 저자들은 Poisson surface reconstruction 및 거리 기반 필터링으로 데이터 품질을 정제한 뒤, voxel subsampling으로 앵커를 만들고 법선 방향으로 가우시안 타원을 flatten/rotate하여 초기 타원을 표면에 맞춘다. 최적화 단계에서는 RGB 포토메트릭 손실에 더해 flatten loss와 LiDAR 깊이·법선 정합 손실(마스크/가중 포함)을 결합해, 가우시안이 필요 이상으로 퍼지거나 중심이 내부로 가는 현상을 억제하며 densification 없이 수렴하도록 만들었다.

- **Empirical Impact**: 실험은 FASTLIVO2 벤치마크, Hilti’22, 그리고 논문 저자들이 제작한 LiDAR-카메라 핸드헬드 스캐너의 자체 데이터에서 렌더링 품질(PSNR/SSIM/LPIPS)과 모델 크기(gaussian 개수)를 함께 평가하며 검증됐다. Structured-Li-GS는 다수 시퀀스에서 SOTA를 상회하면서도 가우시안 수를 크게 줄였고, 예시로 CBD2에서 356,885개 가우시안만으로 Scaffold-GS의 약 1/3 수준을 달성했다. 특히 저텍스처·긴 복도·실외 전이 등 다양한 조건에서 시각적 아티팩트를 줄이고 지오메트리 정합을 유지하는 결과가 보고되어, LiDAR-guided 구조화 초기화가 실사용 확장성과 효율성에 의미 있는 영향을 준다는 점을 보여준다.



### TruEye: Fine-Grained Detection of AI-Generated Human Subjects in Images (https://arxiv.org/abs/2606.27505)
Comments:
          18 Pages, 3 figures

- **Prior Approaches**: 기존 딥페이크 탐지는 대체로 real vs synthetic의 이진 분류에 머물러 조작의 “무엇이/어디서” 문제인지 설명력이 부족했다. 또한 생성기 미관측 데이터에서 성능이 급락하거나, localization은 제공해도 다섯 범주의 조합적 의미를 충분히 구분하지 못하는 한계가 지적된다. 최근에는 LLM 기반 설명도 등장했지만, 대규모 업로드 처리에 비싼 추론 지연과 계산 비용이 현실적 제약이었다.

- **Core Contribution**: TruEye는 AI-generated/AI-manipulated 인간·장면을 다섯 가지 조합 범주(SHSS, SHRS, RHSS, RHRS, SS)로 세분화해, 조작의 성격을 분명한 라벨로 제공하는 fine-grained 탐지·로컬라이징 모델을 제안한다. 특히 RHRS(실제 인간이 실제 장면에 물리적으로 존재하지 않았던 장면에 합성된 경우)를 “화소 단위 생성물”이 아닌 ‘공존(조합) 불일치’ 관점에서 분류하도록 설계했다. LLM 없이도 패치 수준 하이라이트와 범주 라벨을 동시에 제공해 해석 가능성을 확보한다.

- **Technical Challenges**: 문제의 핵심 기술 난점은 (1) 이진 분류용 특징이 조합적 범주 구분에 잘 맞지 않고, (2) RHRS처럼 조작이 얇은 경계에서만 드러나는 경우 모델이 쉽게 놓친다는 점이다. TruEye는 mask-conditioned dual stream transformer로 이미지를 human tokens와 scene tokens로 분해하되, 패치 수준 공간 대응을 유지해 의미론적 일관성을 보존한다. 여기에 region-gated cross attention, feature magnification, token-level supervision, global compositional 분류를 결합해 인간-배경의 상호 정합성을 학습하고, 내부 attention을 의미적으로 일관된 토큰에만 제한해 LLM급 설명기 기반 대비 고속화를 달성한다.

- **Empirical Impact**: FineSyn이라는 5개 조합 범주 전용 fine-grained 데이터셋(총 35,000장)을 새로 구축해 학습·평가 기반을 마련했으며, RHRS를 포함한 조합적 일반화 시험이 가능해졌다. FineSyn 내부에서 TruEye는 전체 정확도 95.52%, F1 97.08%로, 단순 ViT+다중 분류기 베이스라인(정확도 60.11%, F1 53.31%)을 크게 상회한다. 또한 6개 데이터셋 실험에서 최신 탐지기들과 비교해 정확도·추론 속도·미관측 생성기 일반화 측면의 우위를 보이며, LLM 기반 경쟁 대비 100배+ 빠른 추론 실용성을 강조한다.



### ReWorld: Learning Better Representations for World Action Models (https://arxiv.org/abs/2606.27504)
Comments:
          19 pages,3 figures

- **Prior Approaches**: 기존 World Action Models(WAMs)은 비디오 생성(Video DiT)과 계획(Action DiT)을 체인 형태로 결합하더라도, 두 모듈의 중간 표현은 출력 단에서만 간접적으로 학습된다. 그 결과 그럴듯한 미래 프레임을 만들 수는 있어도, 계획 성능(특히 안전성)과의 결합이 약해지는 표현 병목(representation bottleneck)이 생긴다. 또한 이미지 생성용 representation learning 아이디어는 긴 호라이즌 동영상과 안전성 구분 요구를 동시에 만족하기엔 전이 한계가 있었다.

- **Core Contribution**: 이 논문은 자율주행용 WAM을 위한 최초의 표현학습 프레임워크 ReWorld를 제안한다. 핵심은 중간 표현을 “부산물”이 아니라 직접 최적화 대상(직접 타깃)으로 취급해, 생성-계획 간 세계지식 전달과 안전성 판별 능력을 표현 레벨에서 강화하는 것이다.

- **Technical Challenges**: 첫째, Video DiT의 중간 표현이 미래 예측성을 갖도록 만들되 외부 인코더/교사 없이 효율을 유지해야 했다. ReWorld는 Video DiT의 선택된 중간 레이어에 future-predictive supervision을 걸어 표현을 더 일찍부터 미래 제약에 맞추고, 레이어 간 예측 불일치를 inference self-guidance로 활용한다. 둘째, Action DiT 중간 표현이 비디오의 세계지식을 제대로 흡수하지만 안전성까지 반영하도록 해야 했다. 이를 위해 Action DiT는 비디오 표현과 cross-modally 정렬한 뒤, NAVSIM PDM 기반의 hard-negative(가깝지만 위험한) 궤적을 repulsion으로 주어 안전-경계에서 구별적인 표현을 만들고, Stage 3에서는 이 신호가 생성 브랜치에도 역전파되도록 설계했다.

- **Empirical Impact**: nuScenes와 NAVSIM 실험에서 ReWorld는 미세조정된 비디오 생성 FVD를 81.3→61.9로 23.9% 개선했고, RL이나 후처리 없이 closed-loop PDMS를 89.1→90.4로 끌어올렸다. 또한 from-scratch 수렴을 약 2배 가속했으며, 동일한 주행 비디오 프로토콜에서 기존 표현학습 기법들이 자율주행 비디오 생성 과제에서 성능이 제한되는 이유까지 통제 비교로 분석했다. 전체적으로 WAM의 계획 품질을 높이는 데 필요한 ‘세계표현 학습’을 훈련 효율과 안전성까지 함께 개선하는 방향으로 실증했다.



### Aloe-Vision: Robust Vision-Language Models for Healthcar (https://arxiv.org/abs/2606.27500)
Comments:
          MIDL 2026

- **Prior Approaches**: 기존 의료 LVLM 연구는 공개되지 않은 대규모 데이터나 부분 공개 모델에 의존하는 경우가 많아, 데이터·학습 레시피의 투명성이 제한된다. 또한 PubMed 기반 VQA 합성이나 대형 벤치마크 병합이 늘었지만, 공개된 지 오래된 평가셋은 오염(contamination) 가능성이 있어 성능이 과대평가될 수 있다. 마지막으로 안전이 중요한 임상 맥락에서 모델이 adversarial, ambiguous, misleading 입력에 취약하다는 점도 반복적으로 지적돼 왔다.

- **Core Contribution**: 이 논문은 의료와 일반 도메인을 모두 아우르는 준비 완료(ready-to-train) 형태의 오픈 데이터 혼합물 Aloe-Vision-Data를 제안하고, 이를 직접 fine-tuning에 쓰도록 설계했다. 이를 바탕으로 7B와 72B 규모의 Aloe-Vision 모델 패밀리를 완전 공개(weights, 학습 레시피, 데이터)로 배포해 재현 가능성을 높였다. 아울러 새 비전 벤치마크 CareQA-Vision을 통해 오염 위험이 낮은 의료 추론 평가를 제공한다.

- **Technical Challenges**: 핵심 난제는 (1) 고품질 의료 멀티모달 데이터가 부족하고 (2) 오답/저품질 라벨이 섞인 대규모 데이터에서 신뢰도 있는 학습 신호를 확보하며 (3) 긴 샘플이 학습을 지배하지 않도록 균형을 맞추는 것이다. 저자들은 손실 토큰 기반 loss-contributing token 가중치로 도메인·모달리티 비율을 맞추고, LVLM tagging 점수와 answer perplexity로 품질 필터링을 수행했으며, pHash 기반 perceptual hashing으로 훈련-평가 중복을 제거했다. 또한 오픈엔드 평가의 판정 신뢰성을 위해 LLM-judge와 인간 전문가의 일치도를 함께 검증했다.

- **Empirical Impact**: 벤치마킹 결과 Aloe-Vision 계열은 품질 중심 학습 혼합물에서 균형 잡힌 성능을 보이며, 기준선 대비 유의미한 향상을 보이면서 일반 능력 저하가 크지 않다고 보고한다. 특히 CareQA-Vision에서 좋은 성능을 보여, 학습 분포 밖의 임상 케이스 일반화에 강점을 시사한다. 반면 adversarial 분석에서는 표준 벤치마크 성능이 곧 신뢰성으로 이어지지 않음을 확인했으며, Aloe-Vision-AR의 추가 강건성 학습이 misleading 입력에 대한 취약성을 줄이는 데 효과적임을 실증했다.



### DMV-Bench: Diagnosing Long-Horizon Multimodal Agents' Visual Memory with Incidental Cue Injection (https://arxiv.org/abs/2606.27499)
Comments:
          16 pages

- **Prior Approaches**: 기존 에이전트 메모리 연구와 벤치마크는 주로 텍스트 기반을 중심으로 설계돼, 시각 정보를 “기억해야만” 풀리는지 엄밀히 분리하기가 어려웠습니다. VisualWebArena·WebArena·장기 비디오 QA 등은 캡션/alt-text 등 텍스트 단서를 함께 제공해, 시각 기억이 필요해도 텍스트 메모리가 사실상 우회할 여지가 있었습니다.

- **Core Contribution**: 이 논문은 멀티모달 에이전트의 시각 메모리를 상호작용·다중 세션 환경에서 평가하도록 DMV-Bench를 제시합니다. DMV-Bench는 1,000개 가구 상품 변형에 “incidental cue”를 화소에만 심고(L2-leakage contract), 세션마다 대화 맥락을 지운 뒤 특정 큐가 있는 상품을 찾아가게 하여 recall을 “reach(몇 번의 세션 경계를 넘어 살아남았는지)” 곡선으로 측정합니다. 또한 DualMem(dual-coding 기반)을 통해 시각 코드와 언어 코드를 병렬로 저장·검색하도록 설계합니다.

- **Technical Challenges**: 핵심 기술적 난제는 시각 단서가 텍스트로 새지 않게 설계하면서도, 실제로 “언제 기억이 필요한지”를 세션 경계 단위로 측정하는 프로토콜을 만드는 것입니다. 논문은 큐를 이미지에만 존재하게 하고 텍스트 표면에서 사전 감사로 제거하는 L2-leakage contract, 그리고 공통 관찰 스트림을 재활용하는 shared-prefix rollout tree로 계산 비용을 줄이면서 reach별 회상 정확도를 효율적으로 산출합니다. DualMem은 SigLIP-2(시각)와 SBERT(언어 캡션)의 두 임베딩을 저장하고, 검색 시 두 채널의 유사도를 정규화·가중 결합한 뒤 후보의 이미지+캡션을 VLM에 함께 주입하는 방식으로 구현했습니다.

- **Empirical Impact**: DualMem은 Gemini 2.5 Flash와 Qwen2.5-VL-7B 모두에서 체인 길이 J∈{5,10,15,50} 전 구간에서 캡션 베이스라인과 최신 멀티모달 메모리 시스템 3종을 능가합니다. 메모리 뱅크 크기나 인코딩 위치 편향 같은 대조 조건에서도 리드가 유지됐고, ablation 결과 시각 채널이 큐를 end-to-end로 “운반”하며 언어 채널은 쿼리 그라운딩에 상대적으로 작은 비중으로 기여하는 비대칭 dual-coding 양상이 나타났습니다. 연구진은 이 결과가 장기적인 지각 연속성(perceptual continuity) 관점에서 시각 메모리를 설계 목표로 다뤄야 한다는 메시지를 강화한다고 강조합니다.



### SelectAnyTree: A Promptable Instance Segmentation Model for 3D Forest LiDAR Point Clouds (https://arxiv.org/abs/2606.27491)
- **Prior Approaches**: 기존 산림 LiDAR 인스턴스 분할은 CHM 기반 전통 파이프라인이나 clustering/heuristic에 크게 의존하거나, OneFormer3D·ForestFormer3D·ForestMamba처럼 end-to-end query 기반 자동 분할 모델로 발전해도 기본적으로 “한 번에 모든 트리”를 예측하는 모드였다. 이 방식은 사용자가 특정 나무를 선택하거나 잘못 분할된 인스턴스를 대화형으로 교정하는 기능이 없어, 밀집·겹침이 심한 수관에서는 오류 수정 비용이 커진다. 또한 promptable 3D 모델(AGILE3D, NPISeg3D, Point-SAM, PartSAM 등)은 주로 실내/인공물 또는 일반 3D 파트에 최적화되어 있어 산림의 수직 구조·대규모 장면·강한 크라운 얽힘 같은 도메인 구조를 충분히 활용하지 못했다.

- **Core Contribution**: SelectAnyTree는 3D forest point cloud에서 사용자가 몇 번의 클릭으로 “특정 개별 나무”를 선택해 그 인스턴스를 분할하도록 하는 promptable instance segmentation 모델을 제안한다. 핵심은 Click-to-query prompt encoder로 클릭 정보를 단일 content query로 변환하고, Canopy Height Model(CHM)에서 얻은 treetop을 geometry-guided “free” 첫 프롬프트로 활용해 자동 모드와 대화형 모드를 연결한 것이다. 또한 대규모 산림 장면에서 긴 범위 문맥을 효율적으로 반영하기 위해 state-space Query Decoder를 사용해 선형 시간 복잡도 수준으로 마스크 생성을 수행한다.

- **Technical Challenges**: 기술적 난제는 (1) 단일 클릭만으로 밀집·겹침 상태에서 목표 트리를 정확히 분리해야 하고, (2) 사용자 대화형 클릭의 입력-출력 동작을 학습과 평가에서 일관되게 맞춰야 하며, (3) 장면 규모가 커도 추론 비용을 억제해야 한다는 점이다. SelectAnyTree는 클릭의 3D 위치와 positive/negative polarity를 포함해 국소 backbone 특징을 cylinder pooling으로 앵커링하고, signed aggregation으로 여러 클릭을 안정적으로 결합해 디코더에 “하나의 쿼리”로 전달한다. 여기에 CHM treetop을 첫 프롬프트로 시딩해 초기 마스크의 불확실성을 줄이고, SAM 스타일의 click-simulation rollout을 학습과 평가에 동일하게 적용해 클릭 예산별 교정 동작이 그대로 재현되도록 했다.

- **Empirical Impact**: FOR-instanceV2의 대화형/인스턴스 단위 실험과 LAUTx의 독립 held-out 평가에서 SelectAnyTree는 훈련 도메인 밖에서도 강한 일반화를 보였다. 특히 1회 클릭만으로 목표 트리를 78.2 IoU까지 분할하며, 가장 강한 promptable baseline인 Point-SAM 대비 24.8포인트나 개선했고, 모든 IoU 목표를 더 적은 클릭으로 달성했다. 효율 면에서도 파라미터를 19.4M으로 유지하면서 기존 promptable 모델 대비 훨씬 적은 크기와 빠른 추론 시간을 보였고, 대화형 교정 과정에서는 음성/양성 클릭이 누출을 억제하며 소수 클릭 내로 정밀해지는 양상이 시각적으로 확인됐다.



### Fine-tuning a multimodal large language model for clinician-grade autism behavioral scoring from short home videos (https://arxiv.org/abs/2606.27484)
- **Prior Approaches**: 기존 홈비디오 기반 ASD 예측은 짧은 자연 관찰 영상에서 행동 특징을 태깅한 뒤 전통 분류기를 학습해 높은 정확도를 달성해 왔습니다. 다만 그 핵심인 행동 특징 점수 추출이 대규모 인력 라벨링을 요구해 확장성에 병목이 있었습니다. 또한 LLM을 zero-shot 행동 채점 대체로 쓴 연구는 가능성을 보였지만, 임상가급 일치도와 민감도는 충분히 확보하지 못했습니다.

- **Core Contribution**: 이 논문은 Gemini 2.5 Pro를 clinician이 매긴 30개 행동 특징(Q1–Q30)으로 지도 미세조정해, 홈비디오에서 임상가 수준의 행동 특징 점수를 자동 추출하는 파이프라인을 제안합니다. 진단 라벨(ASD-vs-NT)은 학습에서 제외했음에도, 미세조정 후 진단 F1이 emergent zero-shot으로 크게 개선됨을 보여줍니다. 더 나아가 LLM의 특징 벡터를 검증된 downstream 분류기에 넣으면 임상가 점수 입력과 동등 이상의 성능으로 연결됨을 실증했습니다.

- **Technical Challenges**: 관찰 가능한 자연 영상에서 임상적 고차 행동 판단을 일관된 점수 체계로 변환하는 것이 가장 큰 기술 과제였고, 기존 컴퓨터 비전 파이프라인은 이를 제어하기 어렵습니다. 이를 해결하기 위해 LLM에 대해 400개의 clinician-rated 홈비디오로 LoRA 기반 low-rank adaptation을 적용해 30개 특징만을 supervised target으로 학습했으며, 진단 라벨은 의도적으로 보류해 정보 누출을 차단했습니다. 또한 여러 추론 실행을 고려한 평가 설계를 통해, 단일 출력의 변동성과 관찰 불가능(N/A) 상황(모델 abstain)을 함께 다뤘습니다.

- **Empirical Impact**: 99명(ASD 49, NT 50) held-out에서 임상가와의 per-feature weighted Cohen’s kappa가 40% 개선되었고, 28개 중 27개 특징이 상승했습니다. 진단 라벨을 직접 학습하지 않았는데도 ASD-vs-NT의 F1이 53% 개선되며 임상가 direct diagnosis 수준에 근접/초과했습니다. 특히 LLM 특징을 넣은 classifier-assisted 파이프라인은 모든 경로에서 direct LLM 진단보다 성능이 높았고, Random Forest 기준 AUC 86%(95% CI 78–92), 정확도 77%(95% CI 68–85)로 임상 선별(triage) 관점에서 의미 있는 확장 가능성을 시사합니다.



### SemCityLoc: Aerial 6DoF Localization Using Semantic 3D City Models (https://arxiv.org/abs/2606.27444)
Comments:
          accepted by ECCV 2026

- **Prior Approaches**: 기존 항공 6DoF 로컬라이제이션은 정밀 GNSS(RTK 등) 신호에 의존하거나, 텍스처가 풍부한 3D 메시에 기반한 지도 매칭을 사용해왔다. 다만 이런 방식을 쓰면 모델이 무겁고 확장과 온보드 배포가 어렵고, 도시 환경에서는 프라이버시 이슈도 커진다. LoD 기반 대안도 있었지만, 주로 윤곽(contour)이나 실루엣 위주 정렬에 머물러 반복 건물·폐색·근거리 도시협곡에서 관측성 한계가 드러난다.

- **Core Contribution**: 이 논문은 SemCityLoc으로, 항공 포즈 추정을 ‘스파스/텍스처 매칭’이 아니라 파운데이션 모델의 시각 프라이어와 표준화된 LoD(compliant) 도시 3D 모델 사이의 ‘의미-기하(semantic–geometric) 표면 정합’ 문제로 재정의한다. 구체적으로는 지붕/벽 같은 의미 표면과 단안 depth를 함께 정렬해, 반복 건축 패턴과 폐색 상황에서 포즈 식별력을 높이도록 설계됐다. 동시에 실험의 공정성을 위해, 센티미터급 UAV 포즈와 LoD1–LoD3 의미 3D 도시모델, 저고도 도시협곡 영상을 결합한 최초 실세계 벤치마크 SemCityLockeD를 제안한다.

- **Technical Challenges**: 핵심 난제는 (1) 의미 레벨의 표면을 통해 포즈를 충분히 관측 가능하게 만들고, (2) 의미와 depth가 서로 스케일/범위가 어긋나도 안정적으로 정합되게 하는 것이다. SemCityLoc은 4D cost-volume 기반으로 먼저 의미 마스크 IoU로 거친 포즈를 뽑고, 이후 단안 depth(MoGe-2)로 깊이 정합을 추가해 미세 포즈를 particle filter 최적화로 복원한다. depth 항목은 렌더링 depth와의 수치 정렬을 위한 전역 scale·shift를 매칭마다 추정하고, 먼 거리에서 덜 신뢰되는 depth 영향은 역가중으로 완화해 결합 비용의 안정성을 확보한다.

- **Empirical Impact**: SemCityLockeD 실험에서 SemCityLoc은 기존 지도 기반 접근 대비 전반적으로 recall과 오차를 크게 개선했다. 예를 들어 어려운 도시협곡에서 2cm–2° 수준의 엄격 조건에서 recall 69.15%(기존 LoD-Loc 35.11%)로 거의 두 배 가까이 향상됐고, 평균 위치 오차는 9.89m에서 2.62m로 줄었다. 또한 의미 모듈과 depth 리파인 두 단계가 서로 보완적으로 성능을 끌어올린다는 어블레이션이 제시되며, 방대한 텍스처 기반 장면 재구성 없이도 의미 구조화된 기하만으로 고정밀 항공 로컬라이제이션이 가능하다는 실증적 메시지를 남긴다.



### Not All Relations Rotate Alike: Transformation-Aware Decoupling for Viewpoint-Robust 3D Scene Graph Generation (https://arxiv.org/abs/2606.27412)
- **Prior Approaches**: 3D Scene Graph Generation(3DSGG)은 객체를 노드, 관계를 엣지로 하는 그래프로 3D 장면을 구조화해 공간 추론의 기반을 제공해 왔습니다. 기존 방법들은 주로 쌍별 객체 특징과 상대 기하를 결합해 관계를 멀티라벨로 예측하며, 표준(정렬된) 장면 자세에서 높은 성능을 보였습니다. 그러나 yaw 관점 변화(특히 좌/우-전/후 축이 교환되는 경우)에서 관계 예측이 시야 좌표계의 변환 규칙을 따르지 못한다는 실측 불일치가 남아 있습니다.

- **Core Contribution**: 논문은 이 문제의 원인을 ‘predicate-level transformation heterogeneity(관계 술어 변환의 이질성)’로 규명합니다. 방향 술어(left/front/right/behind)는 관측 프레임에 따라 변환돼야 하지만, contact/support/semantic 계열은 상대적으로 안정적이어야 하는데, 기존 모델은 이를 구분하지 않고 같은 관계 공간에 얽어 넣는다고 지적합니다. 이를 해결하기 위해 Transformation-Aware Decoupling(TAD)는 변환 거동이 다른 관계를 분리 학습하고, 다시 표준 멀티라벨 예측으로 결합하는 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘방향 술어는 yaw에 맞게 바뀌어야 하고, 나머지 술어는 유지돼야 한다’는 서로 다른 목표를 한 모델 안에서 충돌 없이 학습시키는 것입니다. TAD는 Viewpoint-Stable Object Encoder(VSOE)로 관점이 바뀌어도 안정적인 객체 표현을 만들고, 관계 추론은 invariant(변환 불변) 경로와 direction-sensitive(방향 민감) 경로로 디코딩합니다. 또한 변환별 기하 디스크립터 분리, 비공유 관계 GNN 브랜치, orthogonal 정규화, 그리고 그룹-aware 보조 감독을 통해 두 경로가 상보적 단서를 포착하도록 유도합니다.

- **Empirical Impact**: 3DSSG 벤치마크에서 TAD는 yaw 회전에 대한 robust 성능을 크게 개선하면서도, 표준(비회전) 설정에서는 경쟁력 있는 성능을 유지합니다. 특히 0°에서 90°/270°로 이동할 때 축 교환이 발생해 더 어려운 구간에서, 기존 비회전 학습 베이스라인들이 큰 성능 저하를 보인 반면 TAD는 그 격차를 줄였습니다. 또한 rotation augmentation(학습 시 회전 데이터 증강)을 쓰지 않고도 상태최신 수준의 yaw 강건성을 달성했으며, 이는 embodied intelligence에서 관점 불변 그래프 생성의 실용성을 높인다는 점에서 의미가 큽니다.



### DexCompose: Reusing Dexterous Policies for Multi-Task Manipulation with a Single Hand (https://arxiv.org/abs/2606.28323)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존에는 (1) 재사용 가능한 dexterous 컨트롤러를 새로 학습하거나, (2) 여러 스킬을 긴 구간으로 순차 결합하며, (3) 손의 자유도를 손가락/자원으로 나눠 멀티 상호작용을 만드는 연구가 많았습니다. 그러나 이미 학습된 full-hand 정책을 그대로 재사용하면서 두 과제를 동시에 수행하도록 안전하게 결합하는 방법은 제한적이어서, 단순 policy chaining이 공용 손가락 동작을 덮어쓰며 성능이 쉽게 무너집니다.

- **Core Contribution**: 이 논문은 DexCompose로, pretrained dexterous 정책 2개를 결합할 때 손가락 단위 action ownership을 명시적으로 부여해 cross-task interference를 줄이는 프레임워크를 제안합니다. 또한 Task A(기존 유지 과제)를 보존하는 residual stabilizer와, Task B(새 상호작용 과제)를 위해 할당된 하위 action subspace에서만 적응하는 context-aware residual을 이중으로 학습/적용합니다.

- **Technical Challenges**: 핵심 기술 과제는 (i) 어떤 손가락(DoF)이 Task A의 결과를 유지하는 데 실제로 필요한지 자동으로 찾아내고, (ii) 이후 Task B가 그 필수 손가락을 건드리지 않으면서도 필요한 경우에만 충분히 보정하도록 제어 범위를 제한하는 것입니다. DexCompose는 Task A의 성공 후 상태를 모은 뒤 release tests로 손가락 마스크를 후처리(discovery)하고, 그 마스크에 따라 잔차(residual)를 bounded stabilizer/할당된 subspace용 residual로 분리해 학습합니다.

- **Empirical Impact**: 시뮬레이션에서 4가지 hold-and-retain 과제와 4가지 downstream 상호작용을 조합해 총 16개 composite task를 평가했으며, 평균 composite success rate 77.4%를 달성했습니다. 이는 policy chaining 대비 15.8%p 큰 개선이며, 특히 dual residual stabilizer가 유지 실패를 막는 데 가장 큰 역할을 했고 finger attribution은 동일 예산에서 추가적인 간섭 감소 효과를 보였습니다.



### LLawCo: Learning Laws of Cooperation for Modeling Embodied Multi-Agent Behavior (https://arxiv.org/abs/2606.28182)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: 기존의 LLM 기반 communicative embodied agent 연구는 대체로 자연어 대화와 행동을 end-to-end로 연결해 협력을 시도하지만, 파트너와 환경/작업 상태에 맞지 않는 행동을 내면서 협력이 비효율적으로 되는 문제가 반복된다. 또한 성능을 끌어올리기 위해 stronger model 증류나 비학습 파이프라인에 의존하는 경우가 많아, 상호작용을 통해 에이전트가 자율적으로 계속 개선하기 어렵다.

- **Core Contribution**: 논문은 Learning Laws of Cooperation (LLawCo)을 제안해, 에이전트가 과거 실패를 되돌아보며 “Talk when necessary”, “Wait for partner” 같은 고수준 행동 법칙을 스스로 추출·정렬하도록 한다. 이렇게 얻은 법칙을 supervised fine-tuning으로 에이전트의 reasoning 체인에 명시적으로 내재화해, 작업 목표와 다른 에이전트의 행동에 동시에 정렬되게 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 협력 실패에서 의미 있는 법칙을 뽑아내고 (2) 그 법칙에 맞는 성공 궤적만 골라 학습 데이터를 구성하며 (3) 추론 시 법칙을 실제 행동으로 안정적으로 연결하는 것이다. 저자들은 실패 에피소드에서 실패 원인을 추출해 법칙 집합을 생성하고, 성공 에피소드 중 법칙 정합성을 만족하는 샘플만 유지한 뒤, 해당 법칙에 근거한 reasoning과 행동을 SFT 데이터로 만들어 law-guided inference에서 법칙을 명시적으로 제공한다.

- **Empirical Impact**: PARTNR-Dialog(신규 대규모 멀티에이전트 communicative benchmark)와 TDW-MAT에서 실험한 결과, 4종 백본 LLM 전반에 걸쳐 협력 효율과 task success rate가 일관되게 향상됐다. 특히 PARTNR-Dialog 벤치마크에서 평균 성공률이 4.5%, TDW-MAT에서 평균 6.8% 개선되었고, 동시에 추론 시 법칙을 수정해도 업데이트된 제약을 안정적으로 따르는 controllability까지 확인했다.



### Enhanced Neural Video Representation Compression across Extreme Complexity and Quality Scales (https://arxiv.org/abs/2606.28163)
- **Prior Approaches**: 기존 INR 기반 비디오 코덱은 비디오를 좌표-출력 매핑으로 과적합시킨 뒤, 가중치나 feature grid 파라미터를 양자화·엔트로피 코딩해 비트스트림을 만든다. 다만 경량(lightweight) 모델은 스케일업 시 압축 성능이 떨어지고, 고성능 모델은 품질이 올라갈수록 모델 복잡도가 함께 증가해 “단일 아키텍처로 다양한 비트레이트를 커버”하기 어렵다. 그 결과 실사용에서 디코딩 복잡도가 고정되지 않거나, 반대로 성능이 제한되는 양극단이 나타난다.

- **Core Contribution**: NVRC++는 INR 기반 코덱을 “복잡도는 고정(여러 complexity level 제공)하되 품질·비트레이트는 폭넓게 확장”하도록 설계한 것이 핵심이다. 이를 위해 단일 코덱 구성에서 고해상도 feature grid를 여러 개 사용해, 특정 복잡도 예산 안에서 성능을 끌어올리면서도 실시간 디코딩을 목표로 한다. 또한 긴 영상에서 고해상도 grid의 overfitting을 효율적으로 가능하게 하는 최적화 프레임워크와, grid 파라미터를 위한 고차원 엔트로피 모델을 함께 제안한다.

- **Technical Challenges**: 가장 큰 난제는 (1) 고해상도 grid를 그대로 쓰면 메모리/연산이 급증하고, (2) 엔트로피 모델이 grid 의존성을 다루는 과정에서 병목이 생기며, (3) 고해상도 grid가 저비트레이트에서 spatio-temporal redundancy를 덜 공유해 성능이 흔들린다는 점이다. NVRC++는 in-parameter coding structure로 grid 내부 의존성을 계층적 방식으로 다뤄 학습 가능한 과적합을 만들고, quantization step-size를 이전 프레임에 의존하지 않는 형태로 바꿔 긴 시퀀스 최적화의 메모리 부담을 낮춘다. 더 나아가 high-resolution grid에 대해 feature grid masking(초기에는 랜덤 마스킹, 학습 후반엔 0으로 수렴)을 적용해 저비트레이트 성능 저하를 완화하고, scale/temporal/spatial priors를 결합한 multi-dimensional grid entropy model로 고차원 파라미터 압축을 가속한다.

- **Empirical Impact**: NVRC++는 7kMACs/pixel부터 360kMACs/pixel까지 네 가지 복잡도 레벨에서 폭넓은 비트레이트·품질 범위를 제공하면서 실시간 디코딩을 지원한다. 실험에서는 SOTA INR 기반 코덱 NVRC 대비 디코딩 속도가 최대 7.6배까지 빨라지면서도 성능은 유사한 수준을 유지한 것으로 보고된다. 이는 “비트레이트에 따라 복잡도가 크게 요동치는” 기존 INR 코덱의 배치 제약을 줄여, 다양한 하드웨어 제약에서 더 쉽게 배포할 수 있는 방향성을 제시한다.



### Differentiable design of the PIAA-ZWFS: a flexible wavefront sensor that approaches the fundamental lim (https://arxiv.org/abs/2606.28136)
Comments:
          Submitted to Astronomy & Astrophysics (A&A)

- **Prior Approaches**: 극한 적응광학(Extreme AO)에서 파면 센서(WFS)는 위상 섭동에 매우 민감해야 하며, 기존 Zernike wavefront sensor(ZWFS)는 위상 대조(phase contrast) 방식으로 광자 노이즈 환경에서 특히 강점을 보여왔다. 다만 민감도 한계에 근접하는 하드웨어급 아키텍처는 드물고, 주파수별·잡음 조건별로 최적화가 엇갈리거나(선형/비선형, 저차/고차 crosstalk) 특히 광자뿐 아니라 read noise까지 함께 고려한 설계 틀은 제한적이었다. 또 점광원 기준으로 최적화한 센서는 별이 해상되는(resolved) 상황에서 빠르게 성능이 떨어질 수 있다는 문제도 남아 있었다.

- **Core Contribution**: 이 논문은 Zernike wavefront sensor를 확장한 phase-induced amplitude apodisation Zernike wavefront sensor(PIAA-ZWFS)를 제안한다. 동심원 PIAA(손실 없는 apodisation)로 초점 마스크에 들어가는 중심부 별빛을 모아 간섭 응답을 강화하고, 다단(멀티 레벨) 위상 마스크 자유도로 설계를 최적화해 위상 추정의 분산을 줄이는 것을 목표로 한다. 또한 PIAA-ZWFS의 최적화 목적함수를 Fisher information 기반의 최대우도 추정 분산(폐루프에서의 잔차 RMS에 해당)에 직접 연결해 설계가 실사용 성능과 맞닿게 했다.

- **Technical Challenges**: 기여를 실현하기 위한 핵심 기술 난제는 (1) 광자 및 read noise가 섞인 조건에서, (2) 고스트렐드(high Strehl) 잔차가 있을 때, (3) 시스템의 모드 crosstalk까지 반영해 센서 설계를 목적함수로 정하는 것이었다. 논문은 차분가능(differentiable) 광학 시뮬레이션과 자동미분을 활용해 ∂I/∂a_k로 야코비안을 계산하고, Bayesian experimental design의 관점처럼 최대우도 추정기의 공분산(역 Fisher information)의 trace를 최소화하는 방식으로 PIAA 렌즈 비구면 계수와 위상 마스크 두께를 동시에 최적화했다. 더 나아가 어떤 WFS에도 amplitude 오차와 phase 오차에 대한 정보 사이의 trade-off가 존재함을 증명해, PIAA가 그 한계 안에서 어떻게 위상 정보에 더 유리한 설계를 만드는지 이론적으로도 정리했다.

- **Empirical Impact**: 시뮬레이션에서 PIAA-ZWFS는 다양한 조리개, 대역폭, 광자 플럭스, 별의 크기 조건에서 기존(최적화된) ZWFS 대비 센서 성능 격차를 크게 줄이며, 전형적 광자-제한 케이스에서 기본 감도 한계 대비 간극을 최대 10배(2.5배) 수준으로 줄였다고 보고한다. 특히 별이 해상되는 경우에는 “점광원용 이상 센서(ideal point source sensor)”가 급격히 비최적이 되는데, PIAA-ZWFS는 항성 지름이 D*>0.8 λ/D를 넘을 때부터 이상 센서를 앞서는 결과를 보였다. 또한 선형/비선형 재구성 모두에서 dynamic range 손실 없이 이득이 유지됨을 보이고, 대역폭이 넓어질수록 생기는 크로마틱 영향도 PIAA와 마스크 최적화로 상당 부분 완화해 설계 실용성을 높였다.



### Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots (https://arxiv.org/abs/2606.28133)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 연구는 사람의 동작을 손 자세 추정(hand-pose estimators) 기반의 6DoF(회전 포함) 엔드이펙터 동작으로 보고, 로봇 학습에서도 이를 그대로 재현하려는 경향이 강했습니다. 하지만 병렬 그리퍼 로봇과 인간 손가락은 접촉 패턴이 근본적으로 달라 회전 신호가 조작 의미와 어긋나며, 추정 오차로 인해 잡음이 크게 유입된다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 사람-로봇 간 공통으로 쓸 수 있는 ‘브리징 동작 표현(bridging action representation)’을 제안합니다. 구체적으로 초기 헤드 카메라 프레임 기준의 손목 상대 ‘이동(translation)’만을 학습 신호로 삼아, 회전이 불안정한 인체 데이터의 약점을 우회하고 로봇에 이식 가능한 동작 공간을 만듭니다.

- **Technical Challenges**: 문제는 서로 다른 데이터 소스(인간/로봇)에서 동작 구성요소가 누락될 수 있다는 점입니다. 이를 위해 π0-like vision-language-action 모델에 interleaved action tokens와 attention masking을 도입해 a3D-wrist→a6D-eef→agripper 순서로 토큰을 배치하고, 손목 이동 신호가 로봇의 실행 가능한 엔드이펙터 동작(a6D-eef)으로 안정적으로 연결되도록 학습 목표를 설계합니다.

- **Empirical Impact**: 여러 ‘새로운 bi-manual 조작 태스크’에서, 제안된 브리징 표현은 잡음이 섞인 6DoF 인간 동작을 그대로 쓰는 방식보다 로봇 기술 전이를 훨씬 더 잘 수행했으며 인간 데이터의 양이 늘어날수록 성능이 확장되는 경향을 보였습니다. 또한 일부 태스크에서는 로봇에 대한 task-specific 시연 없이도 로봇이 완료할 수 있음을 실험적으로 제시해, 대규모 인간 동작 데이터 기반 스케일업 가능성을 시사합니다.



### Higher-Order Fourier Neural Operator: Explicit Mode Mixer for Nonlinear PDEs (https://arxiv.org/abs/2606.28122)
Comments:
          46 pages

- **Prior Approaches**: Fourier Neural Operator(FNO)는 푸리에 영역에서 저차원 스펙트럴 표현을 활용해 함수 공간 매핑을 학습하며, 해가 푸리에 모드별로 독립적으로 진화하는 선형·상수계수 PDE에서 특히 강점을 보인다. 기존 스펙트럴 기반 신경연산자들은 주로 대각적(모드별) 특징 변조에 머물러 비선형 PDE의 모드 간 상호작용(다항 비선형성)이 주는 귀납 편향을 충분히 반영하지 못한다.

- **Core Contribution**: 논문은 비선형 PDE의 다항 비선형성이 만드는 n-선형 모드 혼합 구조를 직접 모델링하기 위해 Higher-Order Spectral Convolution(고차 스펙트럴 컨볼루션)을 제안한다. 이를 Higher-Order FNO(HO-FNO)로 구현해 FNO의 “대각적 스케일링/조절”을 “명시적 n-linear mode mixing”으로 확장하는 것이 핵심이다.

- **Technical Challenges**: 핵심 과제는 고차 모드 혼합을 스펙트럴 연산으로 표현하면서도 FNO가 제공하는 효율성과 해상도 일반화를 유지하는 데 있다. 저자들은 n-선형 모드 혼합에 맞춘 스펙트럴 믹서를 설계해 비선형 PDE의 다항 상호작용 구조를 학습 가능한 형태로 통합하고, HO-FNO가 기존 스펙트럴 연산자처럼 고효율로 동작하도록 구성했다.

- **Empirical Impact**: 표준 벤치마크 실험에서 HO-FNO는 FNO 계열의 효율을 유지하면서 다른 스펙트럴 neural operator들보다 일관되게 성능이 좋아졌다. 특히 Poisson equation with polynomial forcing처럼 비선형성이 큰 상황에서는 단일 HO-FNO 레이어가 최대 16개 레이어의 FNO 모델을 능가하는 등 향상 폭이 컸고, 여러 데이터셋에서 transformers 및 state-space models과 비슷하거나 더 나은 결과를 보이며 비선형 모드 상호작용 학습의 실질적 의미를 입증했다.



### MLVC: Multi-platform Learned Video Codec for Real-World Deploymen (https://arxiv.org/abs/2606.28027)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 신경 비디오 코덱은 학습 기반 압축으로 H.265(HEVC) 대비 60~70% 비트 감소를 보여왔지만, 실제 서비스에는 여전히 쓰이지 못했습니다. 핵심 장애는 (1) GPU 중심 평가 대비 소비자 기기 NPUs에서의 실시간 성능 부족, (2) 서로 다른 하드웨어에서 인코더/디코더가 같은 엔트로피 분포를 보장하지 못해 디코딩이 무너지는 cross-platform 비호환 문제입니다.
기존 정량화 기반 접근은 bit-exact 산술을 노려왔지만, NPUs의 연산 경로·반올림·커널 차이로 “확률/스케일 파라미터” 불일치가 생기며 재앙적 실패가 발생할 수 있고, 보정 정보 제공은 확률적 완화 수준에 그쳤습니다. 일부 fixed-prior·코드북 인덱스 전송 방식은 실패를 줄이지만, 경쟁력 있는 압축 성능 검증이나 공정한 GOP 조건에서의 비교가 제한적이었습니다.

- **Core Contribution**: 이 논문은 MLVC(Machine Learning Video Codec)라는 하드웨어 강건(neural codec hardware-robust) 설계를 제안해 실제 배포를 목표로 합니다. 엔트로피 모델의 스케일(scale) 파라미터를 hyperprior 안에 “scale index” 형태로 명시적으로 전달해, bit-exact 산술 없이도 엔트로피 코딩 일관성을 보장하는 것이 핵심 아이디어입니다.
대신 스케일 전송으로 인한 오버헤드는 구조적 파라미터 공유로 줄이고, gated memory·ReGLU 계열(하드웨어 친화)·LTR(long-term reference recovery)·I-frame dropout·지각(ROI/LPIPS) 학습까지 묶어 BD-rate 효율을 회복합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 스케일 파라미터 σ가 인코더/디코더의 엔트로피 분포를 결정하는데, 부동소수점·비표준 연산·NPU별 활성함수 근사 차이가 누적되면 디코더가 다른 lookup index를 선택하며 재앙적 디코딩 실패로 이어진다는 점입니다. MLVC는 (1) z-hyperlatent의 고정 엔트로피 모델을 공유해 스케일 인덱스를 결정적으로 재구성하고, (2) FP16 환경에서도 divergence가 장기 예측 체인에서 증폭되는 문제를 LTR 프레임과 주기적 I-frame 동기화로 제어합니다.
또한 벤더별 piecewise 근사가 달라지는 복잡 활성함수를 피하고 ReGLU 같은 gating을 하드웨어 호환 연산으로 구성해 cross-platform 수렴 안정성을 확보했습니다.

- **Empirical Impact**: 실험에서 MLVC는 VCD(Video Conferencing Dataset) 벤치마크(360p~720p, 지각 평가 포함)에서 HEVC-QSV 대비 MOS 기준 >70% BD-rate 개선을 달성하며, 실시간(평균 100+ FPS 수준) 인코딩·디코딩도 NPUs 기반 애플/인텔/퀄컴 기기에서 확인됐습니다. 특히 cross-platform 설정에서 대부분의 하드웨어 조합이 재앙적 실패 없이 동작하며, BD-rate가 소폭 차이로 유지되어 “배포 가능성”을 실증합니다.
또한 perceptual loss와 ROI 가중치를 적용한 fine-tuning으로, PSNR 중심으로 학습된 DCVC-RT도 지각 품질 격차를 줄였지만 MLVC는 여전히 cross-platform 제약 비용 대비 더 강한 성능을 보였습니다.



### Verifiable Geometry Problem Solving: Solver-Driven Autoformalization and Theorem Proposing (https://arxiv.org/abs/2606.27926)
- **Prior Approaches**: 기존 Geometry Problem Solving(GPS) 연구는 neuro-symbolic 패러다임을 채택하지만, autoformalization과 theorem prediction이 각각 고립된 정적 단계로 분리된 경우가 많다. 그 결과 다이어그램·텍스트의 모호성을 충분히 반영하지 못한 채 언어적 정확도에만 맞춰져 solver 실행 가능성과 불일치하는 문제가 자주 발생한다. 또한 theorem prediction은 고정된 theorem library와 제한된 search budget 때문에 deductive impasse에 빠지면 이를 국소 보강하기 어렵다.

- **Core Contribution**: SD-GPS는 symbolic solver를 autoformalization과 deduction 전 과정에서 execution oracle로 사용하는 solver-driven 폐루프를 제안한다. solver가 후보가 “실제로 실행되고 목표에 기여하는지”를 피드백으로 주도하며, 이를 통해 형식화의 기준을 표면 정합성에서 실행 가능성으로 옮긴다. 아울러 impasse-aware verified theorem proposing으로, 신경망이 제안하더라도 symbolic verification을 통과한 보조 lemma만 유효하게 반영되도록 설계한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) multimodal 입력을 solver가 즉시 받아들일 수 있는 typed predicate로 변환하면서, (2) theorem 라이브러리 고정으로 생기는 막힘을 “검증 가능한” 방식으로 완화하는 것이다. SD-GPS는 QwenVL3-2B 기반 통합 multimodal formalizer로 supervised adaptation과 solvability-guided reinforcement learning(SG-RL)을 결합해, 파싱·실행·답 충분성에 기반한 보상으로 학습을 구동한다. theorem proposing 단계에서는 현재 proof state 요약을 바탕으로 국소 보조 lemma를 제한적으로 제안하되, solver가 타입 제약·모순·현재 상태 인스턴스 가능성 등을 검증한 뒤에만 사용하며, 스코프도 해당 문제 시도에 한정해 soundness를 유지한다.

- **Empirical Impact**: Geometry3K와 PGPS9K에서 SD-GPS는 completion과 choice, 그리고 cross-modal reference/불일치 구간까지 일관되게 기존 MLLM·neural·neuro-symbolic 방법을 능가한다. 예를 들어 Geometry3K에서 completion 86.4%, choice 90.4%를 기록해 최고 대비 각각 +3.5%p, +3.2%p 개선했고, PGPS9K에서도 completion 79.8%, choice 84.5%로 각각 +4.4%p, +3.0%p 향상했다. ablation 결과 solver-유도 RL, bounded repair, verified theorem proposing을 순차 추가할수록 실행 성공률이 증가했으며, 특히 cross-modal mismatch에서 execution feedback의 효과가 크게 나타나 perception-logic 결합의 실질적 가치를 보여준다.



### NormGuard: Reward-Preserving Norm Constraints in Flow-Matching Reinforcement Learning (https://arxiv.org/abs/2606.27771)
- **Prior Approaches**: RL post-training은 reward alignment를 높이지만, 보상 프록시에 포착되지 않는 시각적 열화(선명도 과증폭, 색 편향, 부자연스러운 조명, 미세 텍스처 손실)를 반복적으로 유발한다. 기존 완화책(early stopping, KL regularization 등)은 드리프트를 ‘총량’으로만 제어해 어떤 성분이 아티팩트를 만드는지 분해·대응하기 어렵다. 또한 CFG처럼 추론 단계 보정이 가능한 경우와 달리, RL로 훈련된 모델의 변화가 동일하게 제거되지 않는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 RL post-training이 flow generator의 per-step velocity norm을 기준 대비 5%~15% 부풀리는 ‘구조적 시그니처(norm inflation)’를 일관되게 확인한다. 중요한 점은 inference-time에서 CFG에서 쓰이던 renormalization 방식이 RL에서 잘 작동하지 않으며, 이는 가중치가 부풀림에 공적응(co-adapt)됐기 때문임을 보여준다. 이를 해결하기 위해 NormGuard라는 학습-시간 hinge penalty를 제안하며, velocity-local 기본 손실과 additively 결합하면서 과도한 norm만 억제한다.

- **Technical Challenges**: 핵심 과제는 (1) norm inflation을 줄이면 reward 최적화 이득을 해치지 않는지, (2) inference-time 보정이 왜 실패하는지, (3) 어떤 형태의 규제가 base loss와 잘 합성되는지였다. 논문은 adjoint sensitivity analysis로 velocity magnitude rescaling이 배치 수준에서 일관된 1차 reward 신호를 주지 못한다는 점을 보이며, 따라서 학습-시간 norm 억제가 reward를 체계적으로 깎지 않을 가능성을 뒷받침한다. 이후 velocity-local loss 공간에서 동작하는 one-sided hinge 페널티(‖vθ‖가 ‖vref‖를 초과할 때만 활성화)를 설계해 방향성 업데이트는 최대한 유지하면서 초과 norm만 제약한다.

- **Empirical Impact**: 두 개의 base flow 모델, 세 가지 RL post-training 방법(NFT/AWM/DPO), 두 가지 reward proxy(PickScore/HPSv2)에 걸쳐 NormGuard는 MLLM 기반 이미지 품질과 forensic realism에서 일관된 개선을 보인다. 특히 few-step inference(적은 step)에서 이득이 더 커지며, 이는 norm inflation이 ODE 적분의 이산화 오차와 결합해 악화될 수 있다는 진단과 맞물린다. reward는 대부분 보존되어 PickScore/HPSv2 차이가 작게 나타났고, early stopping으로도 재현되지 않는 개선이며 KL regularization과도 상보적으로 작동한다.



### Class-frequency Guided Noise Schedule for Diffusion Models (https://arxiv.org/abs/2606.27696)
Comments:
          technical report

- **Prior Approaches**: 확률 점수(score)를 학습하는 score-based generative model은 multi-scale noise schedule로 저밀도 영역에서의 부정확한 점수 추정을 완화해왔다. 다만 long-tailed처럼 클래스 빈도(모수 수)가 다른 데이터에서는 저빈도 클래스가 더 큰 저밀도 영역을 만나고, 점수 공간에서도 고빈도 클래스가 우세해 생성 품질과 다양성이 떨어질 수 있다는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 클래스 빈도와 multi-scale noise schedule 사이의 상관관계를 처음으로 체계적으로 분석한다. 그 결과 저빈도 클래스에는 더 큰 스케일 노이즈가 필요하다는 직관을 바탕으로, Class-frequency Guided(CFRG) noise schedule을 제안한다.

- **Technical Challenges**: 핵심 기술적 난점은 저빈도 클래스에서 발생하는 더 큰 저밀도 영역이 점수 추정 오류로 이어지는 메커니즘을 노이즈 스케줄 설계로 연결하는 것이다. 논문은 노이즈 스케일을 클래스 빈도에 역비례하도록 클래스별로 조정하는 CFRG를 통해 저밀도 영역을 줄이고, 점수 공간의 불균형(고빈도 클래스 쏠림)을 완화하도록 설계했다.

- **Empirical Impact**: CIFAR-100-LT와 ImageNet-LT의 imbalanced 데이터에서 이미지 생성(FID), 이미지 분류(생성 데이터 활용 시 top-1 정확도), 텍스트-투-이미지 생성까지 폭넓게 실험해 CFRG의 개선을 확인했다. 예컨대 CIFAR-100-LT는 FID에서 DDPM 대비 큰 폭으로 향상(2.24 개선), CIFAR-100-LT 분류도 top-1 정확도 9.22%p 개선을 보였으며, frequency 통계가 noise schedule 설계에 “결정적”임을 실증적으로 뒷받침한다.



### DIM-WAM: World-Action Modeling with Diverse Historical Event Memory (https://arxiv.org/abs/2606.27677)
- **Prior Approaches**: VLA/로보틱스 모델은 비전-언어-행동을 하나의 시퀀스 모델로 묶어 성능을 끌어올렸지만, 핵심 감독 신호가 희소한 행동 라벨에 머물러 비접촉/가림/단계 전환 같은 미세 동역학을 충분히 직접 학습하기 어렵다. World-action model(WAM)은 미래 시각 상태를 함께 예측해 동역학을 밀도 있게 감독하지만, 기존 방식은 주로 단기 히스토리와 단기 예측에 기대어 장기 과업에서 필요한 “이전 관측의 의미”를 제대로 유지하지 못한다.

- **Core Contribution**: DiM-WAM은 장기 비마르코프 의존 과업에서 필요한 시간 정보(단기 국소 문맥, 교차 단계 과거 사건, 즉시 미래 동역학, 전역 task progress)를 분해하고, 그중 장기 과거를 Multi-Type Historical Event Memory로 외부화해 기억 붕괴와 전역 상태 인식을 보완한다. 메모리는 관측에서 압축된 사건 토큰을 뽑아 다중 메모리 bank에 유사도 기반 병합으로 저장하고, 읽기 시 bank identity와 시간 임베딩을 부여해 video/action denoising에 일관되게 조건을 제공한다.

- **Technical Challenges**: 핵심 난제는 유한 용량 메모리 안에서 서로 다른 기능의 과거 정보가 경쟁하며 덮어쓰기/주의 경쟁이 생기고, 특히 생성 기반 상태가 중간에 누적되면 drift가 발생해 닫힌 고리 예측이 망가질 수 있다는 점이다. DiM-WAM은 (1) bank별 독립적인 압축·병합으로 이질적 사건의 직접 경쟁을 줄이고, (2) 읽기 단계에서 bank identity+RoPE 시간 순서를 적용하며, (3) progress-supervision 보조 목표로 메모리가 완료된 사건뿐 아니라 현재 과업 단계와 남은 작업의 함의를 함께 담도록 유도해 전역 상태 편향을 억제한다.

- **Empirical Impact**: RMBench에서 LingBot-VA 기반으로 고정 용량 메모리를 적용했을 때 평균 success가 28.4%에서 69.8%로 크게 상승했으며, explicit-memory Mem-0(42.0%)도 앞섰다. 실제 Franka 4개 과업에서도 stage success가 70.7%→91.5%, full-task success가 52.5%→80.0%로 개선되었고, ablation과 분석은 multi-bank 구조와 progress 감독이 특히 시간 의존 과업에서 이득을 만든다는 점을 뒷받침한다.



### Enhancing Co-packaging Optics Enabled Silicon Photonics Security Assurance Hardware Fingerprinting (https://arxiv.org/abs/2606.27612)
Comments:
          Author manuscript version of paper published in IMAPSource Proceedings 2025. Final published version available through IMAPS. 6 pages

- **Prior Approaches**: 기존 보안 방식은 주로 전자회로의 암호·신원확인에 초점을 맞추며, PIC(photonic integrated circuit)에 특화된 위조·변조 위협을 충분히 다루지 못한다. 또한 광학적 물리특성을 활용하더라도 별도 공정이나 추가 재료가 필요하면 비용과 확장성이 떨어지는 문제가 있었다.

- **Core Contribution**: 이 논문은 PIC의 density control filler 영역에 2차원 photonic crystal(PhC) 패턴을 삽입해, 각 칩이 갖는 고유한 광학 서명을 생성하는 hardware fingerprinting 기법을 제안한다. 서명은 특정 가시~근적외선 파장대에서 공진하며, 파장·편광·입사각에 따른 반사/흡수 스펙트럼의 ‘좁은 피크’ 패턴으로 구현된다.

- **Technical Challenges**: 핵심 과제는 위조를 어렵게 만들 만큼 칩 간 식별성이 높으면서도, 표준 공정 범위에서 구현 가능한 나노구조 치수·배치를 설계하는 것이다. 연구진은 FDTD 시뮬레이션(ANSYS Lumerical)을 통해 공진 피크가 각 장치별로 구분되도록 나노구조의 크기와 간격을 최적화했고, 추가 공정 없이 standard lithography만으로 구현 가능하게 구성했다.

- **Empirical Impact**: 시뮬레이션 기반 최적화 결과, 각 장치의 reflection/absorption spectrum에 구별 가능한 narrowband peak 구성이 나타나 고해상도·스케일러블한 인증 신호를 제공한다. 또한 삽입된 나노구조의 정밀도가 sub-50nm 수준이라 포크(위조) 난이도가 크게 높아져, 칩 인증 및 공급망 보안 강화에 비용 효율적인 대안이 될 수 있다.



### On the stability of scale-space metrics (https://arxiv.org/abs/2606.27605)
Comments:
          36 pages, 7 figures

- **Prior Approaches**: 이 논문이 다루는 함수 간 거리 D_{α,p,r}는 Gaussian scale-space(가우시안 스케일스페이스)에서 각 스케일의 특징을 모아 비교하는 방식이다. 기존에는 조화해석 분야의 Besov 공간 노름과의 동치(및 Wasserstein 거리의 특수 경우)와, 영상처리에서 가우시안 스무딩/다중스케일 특징 추출(SIFT 등)처럼 유사한 아이디어가 축적돼 왔다.
다만 이러한 거리들이 입력의 기하학적 변형(예: 변형이동)이나 노이즈에 대해 얼마나 안정적인지, 그리고 실제로 회전 불변 형태를 계산 가능하게 만들 수 있는지는 체계적으로 정량화되기 어려웠다.

- **Core Contribution**: 논문은 D_{α,p,r} 계열이 입력 함수의 기하학적 변형(최대 변위로 표현되는 변형의 크기)과 additive noise에 대해 갖는 강건성을 정량적으로 “상계” 형태로 제시한다. 특히 두 차원 영상(또는 tomographic projection)의 비교에서, in-plane 회전에 대해 최소화한 회전 불변 거리 D_{α,p,r}(f,g;SO(2))를 정의하고 그 안정성을 각도 변화에 대해 증명한다.
또한 주어진 매개변수 r=2일 때는 이 거리들을 유한 샘플에서 효율적으로 계산하는 알고리즘을 제안한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 스케일스페이스에서 여러 해상도 레벨의 거리를 가중합하는 구조가 변형/노이즈에 대해 어떻게 전파되는지, (2) 회전 불변화를 “계산 비용” 관점에서 가능하게 만드는 것이다. 논문은 변형 f→f∘φ|∇φ| 형태에서 변형 크기 ε(φ)=max_x|x−φ(x)|를 기준으로, coarse level에서의 단계별 거리 제어를 결합해 전체 거리의 안정성 상계를 도출한다.
연산 측면에서는 r=2로 제한해 회전 최소화까지 포함한 계산을 O(n^{2} log n) 수준으로 설계하고, 이산화가 sub-Gaussian additive noise에 대해 견디는 것도 함께 증명한다.

- **Empirical Impact**: 수치 실험을 통해 (i) 변형 크기에 따른 거리의 예측 가능한 증가, (ii) tomographic projection에서 회전 불변 거리의 각도 변화 안정성, (iii) 유한 샘플/이산화 조건에서의 성능과 노이즈 강건성을 보여준다. 이는 Wasserstein-연관 거리의 일반화 계열이 “실제 영상/토모그래피 비교”에서 사용할 만한 안정적 지표임을 경험적으로 뒷받침한다.
특히 cryo-EM처럼 in-plane random rotation이 빈번한 응용에서, 회전 불변 거리의 계산 가능성과 안정성 근거가 함께 제시되었다는 점에서 후속 연구와 실사용 모두에 의미가 있다.



### Spectral Subsurface Scattering from RGB via Biophysical Skin Inversion (https://arxiv.org/abs/2606.27604)
Comments:
          14 pages, 9 figures

- **Prior Approaches**: 기존 피부 렌더링은 매질을 단일한 균질 매질로 근사하고, 반사(albedo) 텍스처로부터 albedo inversion을 통해 단일 산란(single scattering) 파라미터를 얻은 뒤, scattering distance와 anisotropy는 아티스트가 손으로 맞추는 방식이 많았다. 이 접근은 피부의 실제 다층 구조에서 생기는 광학 파라미터(특히 scattering profile)의 색-비색상(컬러) 의존성을 충분히 보존하지 못해, 피부 톤마다 부정확한 산란 프로파일이 생기고 제작에도 수작업 부담이 컸다. 또한 다층에서 나타나는 흡수 깊이 의존성을 단일 매질로 “평균화”하면서, 산란 프로파일 형태와 반사 스펙트럼을 동시에 잘 맞추기 어렵다는 한계가 관찰된다.

- **Core Contribution**: 이 논문은 RGB 피부 확산 알베도 1장만 입력받아, 피부에 필요한 “스펙트럼 기반, 렌더러 바로 사용 가능한” subsurface scattering 광학 파라미터 전체를 예측하는 spectral optical inversion 프레임워크를 제안한다. 피부 다층의 복잡함을 단일 매질로 평균내지 않고, 서로 비상관(unrelated)인 K개 매질의 mixture-of-media로 근사해 aggregated multilayered appearance를 표현한다. 더 나아가 random-walk 기반 path tracer에서 매질을 무작위로 선택해 주입하는 방식으로 최소 수정으로 통합 가능하게 만들었다.

- **Technical Challenges**: 핵심 난제는 Monte Carlo로 얻은 다층 산란 프로파일을 path tracer가 요구하는 제한된 단일-매질 파라미터 집합(α, σt, g)으로 효율적으로 “역문제(inversion)”화하는 것이다. 단순 단일-매질 피팅은 프로파일 형태와 반사 스펙트럼을 동시에 만족시키기 어려웠고, 이를 해결하기 위해 LUT 기반의 빠른 피팅(알베도/anisotropy 의존 분리 성질 활용)으로 KK-mixture의 가중치와 스펙트럼 파라미터를 대규모 피부 톤에 대해 공동 최적화했다. 그런 다음 비용이 큰 최적화를 neural decoder 한 번의 forward pass로 증류(distillation)해, RGB 알베도로부터 스펙트럼 파라미터를 연쇄형(chained) 네트워크로 예측하도록 구성했다.

- **Empirical Impact**: 25,000개 피부 톤(각각 다수 파장)에서 생성한 학습/검증 데이터와 fit 결과를 바탕으로, 단일 매질 대비 더 정확한 scattering profile과 반사 재현이 가능함을 보였다. 특히 LUT+피팅으로 K=2/3에서 프로파일 및 반사 오차를 단계적으로 낮추며, 이후 end-to-end decoder가 새로운(미보는) 피부 톤과 실제 촬영 얼굴 텍스처에도 일반화되는 성능을 보고했다. 구현 측면에서도 4K 맵 기준 1ms 미만의 drop-in 예측 성능과, path tracer에서는 매질 랜덤 선택만으로 subsurface scattering을 갱신할 수 있어 제작 파이프라인의 노동과 불정확성을 함께 줄일 수 있다는 점에서 의의가 크다.



### RANSAC Scoring Done Righ (https://arxiv.org/abs/2606.27385)
Comments:
          pre-print

- **Prior Approaches**: 기존 RANSAC 변형들은 후보 모델을 잔차 기반 inlier 수를 세거나, 잔차 임계값 이후 포화되는 per-point 점수를 합산하는 방식으로 평가한다. 그런데 이 점수는 inlier scale에 대한 사용자 파라미터가 필요하며, 그 scale은 오염된 데이터로부터 다시 추정해야 한다는 문제가 있다.

- **Core Contribution**: 논문은 추정-후-채점의 순서를 뒤집어, 고정된 inlier partition에 대해 inlier scale을 Inverse-Gamma prior 하에서 폐형식으로 marginalize한다. 그 결과 inlier scale이 수식에서 실질적으로 사라진 새로운 RANSAC score를 제안하며, Jeffreys(비정보)부터 empirical-Bayes(정보)까지 한 표현으로 커버해 데이터-rich/ data-scarce 모두에 동일하게 적응한다.

- **Technical Challenges**: 핵심 난제는 inlier scale을 제거하면서도 score를 효율적으로 계산하고 최적 partition을 찾는 것이었다. 논문은 공액(conjugate) prior를 활용해 scale marginalization을 닫힌 형태로 전개하고, 전체 최적화는 sort-and-sweep로 O(N log N)에 수행되도록 구성했다.

- **Empirical Impact**: 약 7만 개 규모의 두-뷰 추정 벤치마크(여러 two-view estimation 문제, engineered/learned feature 모두)에서 제안한 score가 RANSAC, MSAC, GaU, MAGSAC 등 기존 SOTA를 능가한다. 특히 threshold miscalibration에 대한 성능이 거의 평탄하게 유지되고, 검증용 validation pair가 2개 수준에서도 기준선들이 요구하는 것보다 수십~100배 이상 적은 데이터로 near-optimal 정확도에 도달하며, validation이 적을수록 prior regularization을 더 타이트하게 조절한다.



New uploads on arXiv(cs.AI)

### Agent-Native Immune System: Architecture, Taxonomy, and Engineering (https://arxiv.org/abs/2606.28270)
- **Prior Approaches**: 기존 방어는 주로 에이전트 바깥에서 작동하는 페리미터 보안, 학습 시 정렬(alignment), 그리고 런타임 모니터링/외부 개입에 의존한다. 하지만 이는 메모리 포이즈닝, 툴 체인 조작, 멀티에이전트 프로토콜 공격처럼 에이전트의 추론 루프 내부로 침투하는 위협에 대응하기 어렵다. 결과적으로 “정렬된 에이전트라도 런타임 하이재킹에 취약”하다는 공백이 강조된다.

- **Core Contribution**: 이 논문은 에이전트 내부 인지 루프에 내장되는 생체 영감의 방어 아키텍처 Agent-Native Immune System(ANIS)을 제안한다. ANIS는 학습 시의 정적 헌법적 가치가 아니라 런타임의 동적 ‘법 집행’처럼 작동하도록 설계되어, 보안·건강·질서를 한 구조로 묶는다. 또한 Immune Tower(총 6계층), Agent Viruses/Vaccines 분류, Harness Triad 기반 Continual Immune Learning(CIL)을 제시한다.

- **Technical Challenges**: 핵심 기술 난제는(1) 툴 메타데이터/메모리 같은 전(前)인지 표면을 추론 전에 차단하면서, (2) 새 위협에 대해 백신이 면역 과잉(autoimmunity)을 일으키지 않게 적응시키는 것이다. 논문은 L1 Barrier Immunity를 비인지 격리층으로 명시해 컨텍스트 유입 전에 샌드박싱/권한 최소화를 걸고, L2~L5에서 규칙·steering vectors·LoRA 등 파라메트릭 백신을 ‘항원’ 발견 시 활성화한다. 아울러 Thymus Simulator와 Autoimmunity Rate(AIR) 같은 지표로 백신 효능과 오탐/자가면역 위험을 동시에 검증하며, Meta-harness–Auto-harness–Self-harness가 닫힌 루프를 이루도록 설계한다.

- **Empirical Impact**: 논문은 기존 평가가 주로 에이전트 결과의 안전성 테스트에 머물러, 방어 메커니즘 자체의 ‘건강도’를 측정하지 못한다고 지적하며 이를 ANIS의 자기면역 관점으로 보완한다. 공개된 실험 프레임에서는 백신이 새로운 런타임 위협에서 목표 안정성과 거버넌스를 얼마나 유지하는지, 그리고 AIR 등으로 자가면역 오탐을 얼마나 낮추는지가 중심이 된다. 이 접근은 집단지능에서도 프로토콜 표준화·새 평가 지표·병원체-백신 공진화 같은 후속 과제를 제시하며, 에이전트 보안을 ‘내생적 면역 공학’으로 확장했다.



### Tandem Reinforcement Learning with Verifiable Rewards (https://arxiv.org/abs/2606.28166)
Comments:
          21 pages,7 figures,8 tables

- **Prior Approaches**: RLVR은 검증기(verifier)가 정답만 평가하는 결과 중심 학습으로 추론 능력을 끌어올렸지만, 생성된 추론 과정이 약한 모델이나 사람에게 이해 가능하도록 유지된다는 보장은 약합니다. 그 결과 RLVR은 기준선(base) 정책에서 벗어난 토큰 분포 드리프트, 읽기 어려움, 언어 혼합 같은 idiosyncratic 패턴으로 이어질 수 있어 점검·감시·다중 에이전트 협업에서 비용이 발생합니다. 이를 완화하려는 KL 페널티, 지도 증류, 과정(trajectory) 감독은 “이해 가능함”을 사전에 정의해야 해서 실제 배치 맥락에 맞춘 스펙 작성이 어렵다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 tandem training(탠덤 트레이닝) 아이디어를 RLVR에 이식한 Tandem Reinforcement Learning (TRL)을 제안합니다. TRL에서는 학습 가능한 senior와 고정된 junior가 단일 롤아웃에서 번갈아 추론을 생성하고, 완성된 결과에 대해 기존 RLVR의 검증기 보상이 그대로 적용되며 senior만 GRPO 스타일 손실로 업데이트됩니다. 핵심은 보상을 바꾸지 않고도 롤아웃 생성 구조만으로 “약한 파트너가 이어갈 수 있는 추론”을 강화해 인간 호환성·다중 모델 커뮤니케이션 문제를 노린다는 점입니다.

- **Technical Challenges**: TRL이 성공하려면 (1) 토큰 단위 교대가 보상 해킹을 유발하지 않으면서, (2) RLVR의 긴 추론 체인에서도 안정적으로 학습이 유지되고, (3) 분포 드리프트가 실제로 억제되는지 확인해야 합니다. 저자들은 senior-유사 과제 학습을 위해 junior를 senior의 동일 base에서 시작해 고정하고, handoff는 단어(word) 경계에서 확률적으로 교대하도록 설계했으며, junior가 낸 토큰에는 gradient가 흐르지 않도록 GRPO 손실을 senior 발화 토큰에만 마스킹했습니다. 또한 단순한 KL 정규화(노출 목표를 junior로 두는 형태)만으로는 같은 이득이 재현되지 않아, 개선이 일반 정규화가 아니라 탠덤 롤아웃 구조에서 나온다는 점을 ablation으로 확인합니다.

- **Empirical Impact**: Qwen3-4B-Instruct를 competition math 세팅에서 DeepScaleR로 RLVR 학습하고, AMC·AIME·Minerva Math에서 평가한 결과 TRL은 단독 추론 성능에서 vanilla GRPO와 사실상 동등하게 유지합니다. 동시에 junior와 추론-step 스케줄로 함께 추론할 때 handoff robustness가 개선되어, AIME에서 최대 +6.6 pass@8 향상을 보였고, 토큰 분포 드리프트는 base/junior 기준으로 더 낮아졌으며(전체 KL 14% 감소), GRPO가 크게 밀어낸 500개 토큰 중 87%는 다시 junior 쪽으로 되돌려졌습니다. 마지막으로 junior가 senior의 chain-of-thought를 따라가기 쉬워지는지 교차 엔트로피와 분포 겹침도 측정해 legibility가 최대 17%까지 개선됨을 보여, RLVR의 “결과 성능”뿐 아니라 “협업 가능성”까지 실용적으로 확장할 수 있음을 시사합니다.



### AI-Driven Synthesis for High-Tech System Design: Automating Innovation (https://arxiv.org/abs/2606.28126)
- **Prior Approaches**: 기존 공학 설계는 토폴로지(이산) 선택과 치수·성능(연속), 제어(활성 동역학)를 순차적으로 다루는 경우가 많아 전체 설계 공간을 통합 탐색하지 못한다. 시뮬레이션 기반 최적화는 제약을 만족시키면서도 조합 폭발이 커지면 계산 비용이 급증하고, CAD 중심 반복 작업은 병목이 된다.

- **Core Contribution**: 이 논문은 automation-in-design(AiD) 패러다임 아래 computational design synthesis(CDS)로, 이산 토폴로지 합성과 연속 성능·치수 최적화를 통합 자동화하는 프레임워크를 제안한다. RL 기반으로 토폴로지를 탐색하고, 물리 기반 NLP/공간 최적화를 결합해 최소한의 인간 개입으로 “설계 생성→검증→개선” 흐름을 만든다.

- **Technical Challenges**: 핵심 난관은 (1) 엄격한 물리 제약을 어기지 않으면서 이산-연속 혼합 설계공간을 효율적으로 탐색하는 것, (2) 배치·루팅 같은 공간 패키징을 CAD의 비분해성 때문에 미분 기반 최적화에 바로 태우기 어렵다는 점이다. 논문은 이를 위해 RL-NLP의 bi-level, solver-in-the-loop 구조로 제약 인식을 강화하고, maximal disjoint ball decomposition(MDBD)로 형상을 미분 가능 추상화해 배치·루팅·물리 최적화를 연속 최적화 문제로 바꿔 해결한다.

- **Empirical Impact**: 사례 1에서는 자동차 기어박스 토폴로지 최적화에서 BF 대비 평가 시간이 3차례 자릿수 수준으로 감소했으며, 최적값 대비 2% 이내 오차 예측과 전 구성에서의 물리적 타당성(stress/packing)을 보고한다. 사례 2에서는 MDBD 기반 배치·루팅이 해석적 벤치마크에서 부피·루팅 길이 차이를 약 0.6~2% 범위로 좁혀 정확성과 실용성을 검증했으며, 결과적으로 CAD 반복 의존을 줄이면서 수천 개 공간 구성을 빠르게 탐색할 가능성을 제시한다.



### Ontology-Guided Evidence Path Inference for Multi-hop Knowledge Graph Question Answering (https://arxiv.org/abs/2606.28076)
Comments:
          14 pages, 4 figures

- **Prior Approaches**: 기존 multi-hop KGQA는 주제(entity/topic) 중심 확장으로 근거 경로(path)나 부분그래프를 찾은 뒤 답을 추론하는 방식이 많다. 하지만 이 접근은 잡음 섞인 mixed-type 경로가 급증해 탐색 공간이 폭발하고, 정답의 의미 제약(복합 조건)을 만족하지 못하는 경로도 함께 수집되는 문제가 있다.

- **Core Contribution**: 이 논문은 OPI(Ontology-guided evidence Path Inference)로 온톨로지(관계-중심) 정보를 활용해 근거 경로 추론을 재구성한다. 질문의 답 타입을 먼저 예측하고, 그 타입이 허용하는 final-hop 관계만을 기준으로 경로 탐색을 제약한 뒤, 생성기-정제기(generator-refiner) 반복으로 질문 맥락에서 관련 없는 증거를 걸러낸다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 탐색 공간 폭발을 줄이면서도 (2) 타입-적합하지만 의미적으로 빗나간 증거를 판별해 semantic misalignment을 낮추는 것이다. OPI는 relation-centric ontology graph로 head-tail 타입 제약을 컴팩트하게 모델링하고, topic-side prefix expansion과 answer-side final-hop matching을 결합하는 bidirectional retrieval로 노이즈 경로 성장을 억제한 다음, refiner의 구조화된 피드백으로 경로/답 맥락을 갱신하며 반복 정제를 수행한다.

- **Empirical Impact**: WebQSP와 CWQ, MetaQA에서 실험한 결과, OPI는 WebQSP에서 Hit@1/ F1을 각각 4.6/5.0 포인트, CWQ에서 8.9/3.3 포인트 최신 강력한 방법 대비 향상시켰다. 특히 MetaQA에서는 retrieval 모듈만으로도 Hit@1이 거의 포화 수준에 도달해, 온톨로지 기반 제약이 재료(근거 경로) 선택 단계에서 효과적임을 보여준다.



### JD Oxygen AI Item Center (Oxygen AIIC) V1: An Industrial-Scale LLM/VLM-Centric Solution for Item Understanding, Management, and Applications (https://arxiv.org/abs/2606.28070)
- **Prior Approaches**: 기존에는 BERT 기반 NER 같은 전통적 NLP와 사전학습 모델로 아이템 속성·개념을 추출해 왔으나, 데이터 소스가 이질적일 때 분포 불일치 대응과 emerging concept 견고성이 부족했다. 또한 task-specific fine-tuning에 의존해 확장성이 떨어지고, 사람이 라벨링해야 하는 ‘수동-애노테이션 병목’ 때문에 대규모 운영 비용이 급증했다. 다른 연구들도 ontology expansion, RAG 기반 속성 추출, 대규모 아이템 지식그래프를 시도했지만, JD급 스케일에서의 안정적 배포와 처리량/지연 최적화는 여전히 난제로 남았다.

- **Core Contribution**: 이 논문은 JD Oxygen AI Item Center(Oxygen AIIC)를 제안하며, LLM/VLM을 중심으로 아이템-지식 생산과 서비스를 end-to-end로 잇는 산업용 인프라를 구축했다. 핵심은 (1) human-AI 협업 기반 ontology engineering으로 수백만 엔트리의 동적 진화를 지원하고, (2) S2D(Semantic Search then Discrimination) 아키텍처로 대규모 AI Item Library를 확장 가능하게 만드는 것이다. 여기에 (3) self-evolving item-understanding LLM/VLM과 (4) unified item tunnel을 결합해 시나리오 전반(검색·추천·운영·카테고리 플래닝)을 일관되게 제공한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 빠르게 등장하는 개념을 이질적 데이터에서 적시에 흡수하면서도, 동의어·중복 때문에 ontology 일관성을 잃지 않게 ‘품질 있게’ 확장하는 것이다. 이를 위해 도메인 지식 백본은 전문가가 정하고, 알고리즘은 discovery–fusion–validation 파이프라인으로 후보를 발굴·표준화한 뒤 multi-LLM collaborative verification과 중복/타당성 검증으로 삽입을 제어한다. 또 수십억 SKUs 규모에서 직접 LLM 호출을 늘리면 비용·지연이 감당되지 않으므로, S2D와 모델-데이터 엔지니어링 최적화, 비동기 파이프라인 및 캐시 재사용으로 처리량을 끌어올리는 방향을 채택했다.

- **Empirical Impact**: Oxygen AIIC는 지식 생산에서 94.2% precision, 82.8% recall을 보고했으며, Huawei Ascend NPUs 환경에서 처리량 효율은 10배 이상 향상되었다. 실제 운영에서도 일 단위 수억 건 업데이트 처리가 이뤄지고, 검색 트래픽 커버리지는 80.4%까지 도달했으며 아이템 정보 품질 이슈는 37% 감소했다. 자동 속성 채움(핵심 속성)도 80%를 상회해, 수작업 기반 사이클을 수 주에서 수 일로 단축하는 등 대규모 e-commerce 운영의 효율과 경험 개선에 실측 효과를 보였다.



### Lifted Causal Inferenc (https://arxiv.org/abs/2606.28024)
Comments:
          Accepted to the Annals of Mathematics and Artificial Intelligence journal

- **Prior Approaches**: 관계형 도메인에서의 행동 선택은 보통 기대효용을 계산하며, 이때 행동을 단순 조건화(conditioning)하면 개입의 의미가 훼손되어 인과효과가 왜곡될 수 있다. 기존 인과 추정 연구는 주로 명제(propositional) 인과모형에 집중했거나, 관계형 인과모형이 조건부독립성 추론에는 강했지만 ‘lifted 수준’에서 인과효과를 효율적으로 계산하는 알고리즘은 부족했다. 또한 관계형 도메인에서의 lifted 확률추론은 빠른 질의응답을 제공해왔지만, 인과 개입(intervention) 의미를 통합해 인과효과를 정확히 계산하는 데는 공백이 있었다.

- **Core Contribution**: 이 논문은 lifted 추론을 관계형 도메인의 ‘인과효과 계산’으로 확장하기 위해 parametric causal factor graphs(PCFGs)와 그 안에서의 개입(intervention) 의미를 형식화한다. 더 나아가 Lifted Causal Inference(LCI) 알고리즘을 제안해, PCFG를 grounded(기초화)하지 않고 lifted 수준에서 인과효과를 계산함으로써 명제 추론 대비 크게 속도를 개선한다. 또한 partially directed parametric causal factor graphs(PD-PCFGs)를 도입해 인과지식이 부분적으로만 주어지는 상황까지 다루며, 그에 맞춰 확장된 lifted causal inference 절차를 제공한다.

- **Technical Challenges**: 핵심 난제는 개입의 의미를 확률그래프의 조건부확률처럼 단순 치환하는 것이 아니라, 개입 시 해당 변수를 고정하고 그로 들어오는 인과 영향(incoming causal influences)을 무시해야 한다는 ‘정확한 의미론’을 lifted 표현에 맞게 정의하는 데 있다. 논문은 causal factor graph(CFG)를 기반으로 factor node/variable node의 방향성과 분리(division/separation) 규칙을 정합적으로 제시하고, PCFG에서 개입 의미를 formal semantics로 제공한다. 이어 LCI는 lifted level에서의 연산을 설계해 필요로 하는 grounded 범위를 최소화하며, PD-PCFG에서는 불완전한 인과관계(부분적 인과지식) 하에서도 개입효과를 계산할 수 있도록 접근을 일반화한다.

- **Empirical Impact**: 논문은 lifted causal inference가 propositional inference(예: causal Bayesian networks 기반)보다 인과효과 계산을 ‘도메인 크기에 대해 다항시간(polynomial time)’으로 수행하는 방향임을 강조하며, 효율성 이득이 크다는 점을 실험적으로 뒷받침하는 구성을 제시한다. 결과적으로 관계형 도메인에서 행동의 올바른 인과적 효과를 계산해 의사결정까지 연결할 수 있는 계산 기반을 제공하며, 기존 lifted 확률추론의 속도 장점을 인과 추론에도 확장하는 의미가 있다. 특히 PD-PCFG까지 포함함으로써 사전에 인과관계를 많이 알 필요가 줄어, 실제 적용 가능 범위를 넓힌다는 점이 기대효과로 제시된다.



### RelBall: Relation Ball with Quaternion Rotation for Knowledge Graph Completion (https://arxiv.org/abs/2606.27967)
- **Prior Approaches**: 지식그래프 보완(KGC)은 누락된 링크를 예측해 그래프 커버리지를 높이는 작업이다. 기존 지오메트릭 KGE는 TransE/RotatE/Rotate3D 계열처럼 회전·변환으로 대칭/반대칭/역관계·합성 패턴을 다루지만, 비가환(non-commutative) 합성과 의미 계층(semantic hierarchy)에는 취약한 경우가 많다. 또한 Rotate 계열은 주로 one-to-one 가정에 머물러 one-to-many/many-to-one/many-to-many 같은 복잡한 관계 매핑을 충분히 표현하지 못한다.

- **Core Contribution**: RelBall은 Rotate3D를 확장해 계층과 방향성, 그리고 복잡한 매핑을 동시에 다루는 프레임워크를 제안한다. 핵심은 (1) quaternion 회전으로 관계의 방향성을 모델링하고, (2) modulus transformation(모듈러스 스케일링)으로 의미 계층을 수치적으로 반영하며, (3) tail-centric relation ball로 one-to-one~many-to-many 관계를 파라미터 추가 없이 표현하는 것이다. 특히 모듈러스 값이 의미 수준을 직접 반영한다는 해석 가능성을 함께 제공한다.

- **Technical Challenges**: 문제는 비가환 합성(조합 순서가 의미를 바꾸는 관계)과 의미 계층을 같은 임베딩 공간에서 동시에 만족시키는 제약을 설계하는 데 있다. RelBall은 회전은 quaternion의 3D 회전으로, 계층은 모듈러스 스케일링으로 분리해 학습 안정성과 표현력을 확보하고, relation ball의 반경으로 계층 매핑의 허용 오차(tolerance)를 기하적으로 모델링한다. 또한 tail-centric ball 제약을 통해 동일 tail에 여러 head가 매칭되거나 한 head가 여러 tail로 확장되는 다대다 관계를 자연스럽게 처리하도록 구성했다.

- **Empirical Impact**: WN18RR과 FB15k-237 두 벤치마크에서 RelBall은 여러 기준모델 대비 경쟁력 있는 링크 예측 성능을 보였다. FB15k-237에서는 전 지표에서 1위를 기록하고, WN18RR에서도 상위권 성능과 견고성을 확인했다. 스케일링 factor 분포 분석은 이론적 기대(1 미만/근처/초과가 각각 의미 계층의 위·동일·아래 수준을 대응)를 지지했으며, ablation에서 modulus scaling과 radius factor가 모두 성능에 중요함이 드러났다.



### Verifiable Geometry Problem Solving: Solver-Driven Autoformalization and Theorem Proposing (https://arxiv.org/abs/2606.27926)
- **Prior Approaches**: 기존 Geometry Problem Solving(GPS) 연구는 neuro-symbolic 패러다임을 채택하지만, autoformalization과 theorem prediction이 각각 고립된 정적 단계로 분리된 경우가 많다. 그 결과 다이어그램·텍스트의 모호성을 충분히 반영하지 못한 채 언어적 정확도에만 맞춰져 solver 실행 가능성과 불일치하는 문제가 자주 발생한다. 또한 theorem prediction은 고정된 theorem library와 제한된 search budget 때문에 deductive impasse에 빠지면 이를 국소 보강하기 어렵다.

- **Core Contribution**: SD-GPS는 symbolic solver를 autoformalization과 deduction 전 과정에서 execution oracle로 사용하는 solver-driven 폐루프를 제안한다. solver가 후보가 “실제로 실행되고 목표에 기여하는지”를 피드백으로 주도하며, 이를 통해 형식화의 기준을 표면 정합성에서 실행 가능성으로 옮긴다. 아울러 impasse-aware verified theorem proposing으로, 신경망이 제안하더라도 symbolic verification을 통과한 보조 lemma만 유효하게 반영되도록 설계한다.

- **Technical Challenges**: 가장 큰 기술 과제는 (1) multimodal 입력을 solver가 즉시 받아들일 수 있는 typed predicate로 변환하면서, (2) theorem 라이브러리 고정으로 생기는 막힘을 “검증 가능한” 방식으로 완화하는 것이다. SD-GPS는 QwenVL3-2B 기반 통합 multimodal formalizer로 supervised adaptation과 solvability-guided reinforcement learning(SG-RL)을 결합해, 파싱·실행·답 충분성에 기반한 보상으로 학습을 구동한다. theorem proposing 단계에서는 현재 proof state 요약을 바탕으로 국소 보조 lemma를 제한적으로 제안하되, solver가 타입 제약·모순·현재 상태 인스턴스 가능성 등을 검증한 뒤에만 사용하며, 스코프도 해당 문제 시도에 한정해 soundness를 유지한다.

- **Empirical Impact**: Geometry3K와 PGPS9K에서 SD-GPS는 completion과 choice, 그리고 cross-modal reference/불일치 구간까지 일관되게 기존 MLLM·neural·neuro-symbolic 방법을 능가한다. 예를 들어 Geometry3K에서 completion 86.4%, choice 90.4%를 기록해 최고 대비 각각 +3.5%p, +3.2%p 개선했고, PGPS9K에서도 completion 79.8%, choice 84.5%로 각각 +4.4%p, +3.0%p 향상했다. ablation 결과 solver-유도 RL, bounded repair, verified theorem proposing을 순차 추가할수록 실행 성공률이 증가했으며, 특히 cross-modal mismatch에서 execution feedback의 효과가 크게 나타나 perception-logic 결합의 실질적 가치를 보여준다.



### NormAct: A Benchmark for Hidden Social Norm Compliance in Embodied Planning (https://arxiv.org/abs/2606.27826)
- **Prior Approaches**: 기존 연구는 텍스트/멀티모달에서 사회 규범을 ‘판단·설명’하는 능력이나 규범 지식을 묻는 평가에 집중해 왔습니다. 반면 로봇/에이전트 벤치마크는 주로 내비게이션·조작·목표 달성에 점수를 매기고, 사회적 적절성은 부차적 요소로 취급하는 경우가 많아 행동 단계에서의 숨은 제약 반영 여부를 놓쳤습니다.

- **Core Contribution**: 이 논문은 숨은 사회 규범을 일반 과제 안에 잠재(implicit)로 숨기고, 모델이 산출한 ‘실행 가능한 행동 시퀀스’가 그 규범을 실제로 만족하는지 평가하는 NormAct를 제안합니다. 목표 달성(Goal Achieved)과 규범 준수(Norm Compliance)를 분리 측정하고, 둘을 모두 만족하는지 Task Success로 함께 평가해 기존의 과대평가 문제를 정면으로 다룹니다.

- **Technical Challenges**: 핵심 난제는 모델이 사회 규범 ‘자체’를 아는 문제가 아니라, 주어진 1인칭 장면에서 관련 규범을 활성화(activate)·근거화(ground)한 뒤 행동 계획 제약으로 통합하는 과정이 실패한다는 점입니다. 이를 위해 장면과 과제 지시에 조건화된 cue 생성 모듈 NormPerceptor를 도입해, 계획 전(scene-grounded) 사회 단서를 자동으로 추론·주입함으로써 규범-행동 연결을 개선합니다.

- **Empirical Impact**: 실험에서 MLLM 기반 계획기는 목표 달성은 67.3%인데 숨은 규범 준수는 26.4%에 그쳐 Task Success는 21.8%로 크게 낮게 나타났습니다. 특히 NormPerceptor로 Task Success가 24.2%→46.7%로 상승했으며, evidence cue는 큰 개선을 보인 반면 RAG cue(일반 규범 지식 제공)는 장면 근거화가 되지 않아 효과가 제한적이었습니다. 결과적으로 ‘규범 지식 접근’보다 ‘장면 증거 기반의 규범 활성화 및 행동 제약화’가 embodied planner 성능의 병목임을 실증적으로 보여줍니다.



### ATOD: Annealed Turn-aware On-policy Distillation for Multi-turn Autonomous Agents (https://arxiv.org/abs/2606.27814)
- **Prior Approaches**: 기존에는 환경 보상으로 직접 학습하는 PPO/GRPO 계열 강화학습이 널리 쓰였지만, 멀티턴에서 보상이 희소하고 지연돼 소형 모델의 초기 탐색 효율이 떨어진다. 반면 OPD(on-policy distillation)는 교사의 토큰 분포로 조밀한 지시를 제공해 초반 수렴이 빠르지만, 학생이 교사와 비슷해지면 성능이 교사 천장에 막혀 더 높은 보상으로의 점프가 어렵다. 또한 긴 궤적에서는 모든 턴에 동일한 증류를 주면 가치가 낮은 턴까지 감독이 퍼져 불안정/비효율이 생길 수 있다는 한계가 제기돼 왔다.

- **Core Contribution**: 이 논문은 ATOD(Annealed Turn-aware On-policy Distillation)로 OPD와 GRPO의 장점을 한 프레임워크에서 결합한다. 학습 초반엔 OPD를 우세하게 두어 교사 수준의 상호작용 패턴에 빠르게 접근하고, 이후엔 GRPO 비중을 점진적으로 키워 보상 기반 탐색으로 교사 천장을 넘어서는 것을 목표로 한다. 추가로 T-DUR(Turn-level Disagreement-Uncertainty Reweighting)로 긴 궤적에서 어떤 턴에 증류를 더 강하게 줄지 “턴 단위”로 부드럽게 가중한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 희소·지연 보상 때문에 RL이 초반에 비효율적인 문제와 (2) 긴 에이전트 궤적에서 교사 감독을 어디에 얼마나 줄지의 문제를 동시에 해결하는 것이다. ATOD는 단일 하이브리드 advantage에 OPD 신호와 GRPO 신호를 섞되, 훈련 진행도에 따라 OPD 계수는 감소시키고 RL 계수는 증가시키는 annealed schedule로 전환을 매끄럽게 만든다. 또한 T-DUR는 턴 내부에서 관측되는 student–teacher disagreement과 student uncertainty를 기반으로 OPD 가중치만 조정해, 보상 신호는 약화시키지 않으면서 가치 높은 결정 턴에 조밀한 감독을 집중한다.

- **Empirical Impact**: 실험은 ALFWorld, WebShop, Search-QA에서 Qwen3-0.6B/1.7B/4B 학생 모델을 대상으로 수행됐고, ATOD는 OPD 대비 평균 성공률을 3.03포인트, GRPO 대비 23.62포인트 개선했다. 더 나아가 해당 교사 모델을 2.16포인트 상회하는 결과도 보고돼, 단순한 교사 모방을 넘어선 학습 효과가 확인된다. 특히 보상이 희소한 초기 단계에서 ATOD가 작은 모델의 성능을 급격히 끌어올리며, annealing과 turn-level 재가중이 성능에 결정적이라는 ablation 분석도 제시된다.



### Grounded Iterative Language Planning: How Parameterized World Models Reduce Hallucination Propagation in LLM Agents (https://arxiv.org/abs/2606.27806)
Comments:
          Under Review

- **Prior Approaches**: LLM 에이전트의 world model은 크게 두 갈래로 나뉜다. 하나는 agent-based world model로, 언어로 다음 상태를 “상상”하며 계획하되 오류가 환각된 상태 변화로 나타나 보통의 회귀 손실로 점수화하기 어렵다. 다른 하나는 parameterized world model로 전이 예측기를 학습해 NodeMSE·delta accuracy·validity accuracy 등으로 측정 가능하지만, 단독 플래너로서는 의미적 계획력이 약해지는 경향이 있다.

- **Core Contribution**: 이 논문은 두 계열을 그래프 기반 플래닝 벤치마크 4종에서 비교하고, agent-based 쪽의 환각을 정량화하기 위한 operational hallucination metrics를 제안한다. 그 결과를 바탕으로 Grounded Iterative Language Planning(GILP)을 제시하는데, API 기반 LLM 추론 능력은 유지하되 소량의 parameterized backbone이 “근거 있는 상태 델타”를 제공해 환각 누적을 줄이도록 한다. 핵심 아이디어는 LLM이 만든 imagined delta와 백본이 예측한 delta의 불일치를 일관성 게이트로 감지하고 필요 시 수정 재프롬프트를 넣는 것이다.

- **Technical Challenges**: 가장 큰 난제는 API 기반 world model의 오류가 문장·JSON 내부에 섞여 장기 맥락으로 전파되는데, 이를 단일 손실로는 포착하기 어렵다는 점이다. 논문은 이를 Hallucinated-State Rate(HSR)·Propagation Depth(PD)·Error-Explosion Slope(EES) 같은 지표로 horizon-resolved하게 분해해 측정한다. GILP는 Jaccard 기반 consistency gate로 불일치 atoms를 지목해 교정 신호를 주며, 동시에 backbone의 value/risk 예측을 skeleton scoring에 활용해 불필요한 LLM 재호출을 최소화한다.

- **Empirical Impact**: 실험에서 GILP는 실제 GPT-4o-mini API 호출 조건에서 hallucinated-state rate를 0.176에서 0.035로 80% 이상 감소시킨다. 보정된 시뮬레이터 ablation에서는 success가 0.668에서 0.838로 상승했으며, LLM 호출 수는 약 22%만 추가로 증가하는 것으로 보고된다. 즉, 환각 전파를 줄이면서도 LLM의 플래닝 강점을 크게 훼손하지 않는 하이브리드 설계의 실용성을 보여준다.



### Understanding Rollout Error in Graph World Models (https://arxiv.org/abs/2606.27780)
Comments:
          Under Review

- **Prior Approaches**: 기존 world model 연구는 벡터나 이미지처럼 고정 차원의 상태에서 1-step 예측 오차가 롤아웃 동안 누적되는 “compounding error”를 주로 스칼라 Lipschitz 상수로 분석해 왔다. Graph World Model(GWM)도 다루는 사례가 늘었지만, 대부분은 정적 그래프 과제나 단발성 추론/예측에 초점이어서 장기 롤아웃에서 토폴로지와 오차 전파 방식이 어떻게 달라지는지 정교하게 다루지 못했다. 또한 고정된 edge 환경을 가정한 틀을 넘어, edge 자체를 예측하는 dynamic-edge(동적 엣지) 상황에서 실패 모드가 어떻게 바뀌는지에 대한 체계적 이론이 부족했다.

- **Core Contribution**: 이 논문은 Graph World Models에서 장기 롤아웃 오차를 “고정 edge(FE)”와 “동적 edge(DE)”를 함께 아우르는 unified framework로 정식화한다. 노드/엣지/그래프 수준 결정을 action node로 모델링하고, FE에서는 엣지 예측이 없는 특수 케이스로, DE에서는 노드-엣지 오차가 결합되는 일반 케이스로 분리해 해석 가능성을 제공한다. 특히 그래프의 구조가 오차 증폭을 어떻게 유발하는지, 그리고 DE에서 node-edge 결합이 어떤 역할을 하는지 그래프-값(bound on graph-valued rollout) 형태로 분석한다.

- **Technical Challenges**: 핵심 기술적 난관은 그래프에서는 같은 1-step 오차라도 토폴로지(예: 연결성/스펙트럴 특성)에 따라 오차가 전혀 다른 방식으로 확산된다는 점과, DE에서는 node-feature 오차가 다음 edge 예측을 오염시키며 edge 오차가 다시 메시지 패싱을 바꾼다는 결합성이다. 이를 위해 토폴로지 유도 증폭과 모델 유도 증폭을 분리하는 graph error amplification factor(GEAF)로 스케일을 제시하고, DE 롤아웃에서는 node와 edge 오차를 함께 전파하는 joint node-edge error operator B를 도입해 결합 동역학을 포착한다. 분석 결과를 바탕으로 Error-Aware GWM을 제안하며, spectral regularization, rollout consistency, critical-node weighting을 결합해 장기에서 발산(divergence)을 억제하면서도 예측 정확도를 유지하도록 학습 목표를 설계한다.

- **Empirical Impact**: 합성 토폴로지 7종과 이질적인 agent-graph testbed에서 실험한 결과, 롤아웃 길이가 길어질수록 예측 오차와 planning regret이 증가하며 DE 환경(구조가 변하는 경우)에서는 dynamic-edge 학습이 필수적임을 확인한다. 또한 단순 FE/GCN 계열은 long-horizon에서 divergence가 나타날 수 있지만, Error-Aware GWM은 토폴로지 스펙트럴 특성에 의해 촉발되는 장기 발산을 막으면서 롤아웃 안정성을 확보한다. real-world 그래프 벤치마크 관점에서도 GWMs는 dynamic graph rollout 및 agent planning에 가장 유용하고, 정적 또는 sparse 예측 과제에서는 전용 그래프 모델이 여전히 강하다는 “적용 범위”를 명확히 한다.



### Towards Reliable and Robust LLM Planning: Symbolic Feedback-Driven Iterative Self-Refinement Framework (https://arxiv.org/abs/2606.27757)
- **Prior Approaches**: LLM은 자연어 생성에는 강하지만, long-horizon planning에서는 환각과 제약 위반으로 인해 불가능하거나 부정확한 계획을 내놓는 문제가 반복된다. 기존 보완책으로는 PDDL 같은 구조적 입력을 쓰는 symbolic planner가 있으나, 도메인 지식과 입력 형식 의존성이 커 자연어/암묵적 제약 대응이 어렵다. 그 결과 신뢰성( feasibility/correctness )을 보장하기 위한 방법의 공백이 남아 있었다.

- **Core Contribution**: 이 논문은 LLM이 만든 계획을 symbolic verifier로 점검하고, 발견된 오류를 LLM이 이해 가능한 형태의 corrective instruction으로 되돌려주는 symbolic feedback-driven iterative self-refinement 프레임워크를 제안한다. 또한 PDDL의 논리 기호를 자연어로 매핑하는 natural language prompting을 도입해 의미 정렬을 돕고, plan recognizer로 목표 도달 가능성( goal reachability )을 추정해 더 정확히 목표를 향하도록 한다. 즉, “feedback–refine–guide” 루프로 맹목적 시행착오를 줄이는 것이 핵심이다.

- **Technical Challenges**: 가장 큰 기술적 난관은 (1) LLM 출력의 구문/의미 오류를 신뢰성 있게 탐지하고 (2) 그 오류를 LLM이 실제로 수정할 수 있는 피드백으로 재구성하는 것이다. 논문은 구문 정합성(명명·파라미터 매칭), 의미 정합성(VAL validation tool), 그리고 goal deviation 여부( plan recognizer )를 계층적으로 검사한 뒤, 이 결과를 augmented prompt의 history/feedback 항목으로 주입해 반복 최적화를 수행한다. 반복 횟수는 verifier가 오류 없이 feasible/goal-consistent을 확인하거나 최대 iteration에 도달하면 종료된다.

- **Empirical Impact**: PlanBench 1,200개 태스크에서 feasibility와 optimality 모두에서 LLM-direct 대비 개선이 일관되게 관찰됐다. Blocksworld에서는 GPT-4o와 Claude-3-5가 각각 34.0%, 28.7%의 coverage 증가를 보였고, DeepSeek-R1은 feedback을 통해 100% coverage에 도달했다. 특히 Mystery에서는 자연어 매핑만으로도 개선이 생기지만, symbolic feedback이 오류 위치와 기호-의미 연결을 더 잘 제공해 coverage를 끌어올렸으며, 긴 plan length에서 격차가 커지는 경향도 확인됐다.



### ToE: A Hierarchical and Explainable Claim Verification Framework with Dynamic Multi-source Evidence Retrieval and Aggregation (https://arxiv.org/abs/2606.27736)
- **Prior Approaches**: 기존 접근은 RAG나 tool calling으로 최신 맥락을 가져와 환각을 줄이지만, GEO로 ‘검색에 잘 노출되는’ 허위 정보를 체계적으로 주입하면 오히려 LLM 추론을 오염시킬 수 있습니다. 탐지 분야에서는 딥러닝 기반 일반화 문제가 크고, LLM 기반 fact-checking도 시간 민감 뉴스나 GEO형 적대적 공격에서는 실증 성능이 흔들리는 한계가 지적됩니다. 일부 다중 라운드 검색·재질문 전략은 evidence 부족을 보완하지만, 지식 축적이 어떻게 결론으로 이어지는지 설명 가능하고 효율적으로 통제하기는 어렵습니다.

- **Core Contribution**: 이 논문은 자동 fact-checking을 위해 Tree of Evidence(ToE)라는 계층적 증거 추론 프레임워크를 제안합니다. 각 주장(claim)을 동적으로 확장되는 argument tree로 모델링하고, multi-source retrieval·증거 평가·트리 집계를 반복해 최종 veracity 점수와 함께 ‘증거 체인’ 형태의 설명 가능한 reasoning을 제공합니다. 특히 GEO로 오염된 입력에서도 sub-claim 분해와 반증(counter-evidence) 중심 탐색을 통해 신뢰도·진실성 판정을 점진적으로 갱신하는 점이 핵심입니다.

- **Technical Challenges**: 핵심 난제는 ground-truth veracity가 잠재 변수라 관측이 불가능한 상태에서, 어떤 소스를 언제 얼마나 더 조회해야 하는지를 정책으로 학습해야 한다는 점입니다. ToE는 evidence 수집을 POMDP로 정식화하고, 신뢰도(reliability) 증가를 reward로 삼아 reinforcement learning 기반 retrieval agent를 학습하되, evaluation network의 신뢰도 추정이 정보이득을 대체하도록 설계해 이론적 오류 보장을 제시합니다. 또한 sub-claim 확장은 신뢰도가 부족할 때만 수행하고, 충분히 결정적인 가지는 pruning·convergence 조건으로 중복 연산을 줄여 효율을 맞춥니다.

- **Empirical Impact**: 여러 데이터셋과 backbone LLM에 대해 실험한 결과, ToE는 경쟁 베이스라인 대비 4~24%p의 성능 향상을 보였고 특히 GEO poisoning 입력에서 개선 폭이 두드러졌습니다. ablation으로 retrieval action space(어떤 소스를 어떤 방식으로 탐색하느냐)의 기여도도 확인해, 다중 소스·반증 쿼리·중단/확장 제어가 성능에 직접 연결됨을 보여줍니다. 설명 가능한 evidence tree까지 제공함으로써 단순 판정뿐 아니라 검토·감사 관점의 활용 가능성도 함께 높였다는 점에서 의미가 큽니다.



### MER-R1: Multimodal Emotion Reasoning via Slow-Fast Thinking Synergy (https://arxiv.org/abs/2606.27652)
Comments:
          Under review

- **Prior Approaches**: 기존 멀티모달 감정 인식(MER) 연구는 고정 라벨 분류에서 벗어나 OV-MER처럼 감정 표현을 더 개방적으로 다루도록 발전해 왔다. 동시에 RLVR 기반 reasoning을 도입해 예측을 시각·음향·텍스트 근거로 설명 가능하게 만들려 했지만, 실제 정확도 향상은 일관적이지 않았다. 특히 reasoning 기반 MLLM에서 slow thinking이 fast thinking보다 낫다는 기대와 달리, 현재 벤치마크에서는 fast thinking이 더 잘 나오는 ‘thinking paradox’가 관찰된다.

- **Core Contribution**: 이 논문은 slow-fast complementarity를 분석해, fast thinking은 재현율 중심의 더 넓은 커버리지와 정답 카테고리에 대한 높은 confidence를 제공하는 반면 slow thinking은 정밀도 중심의 보수적 필터링으로 오답 카테고리를 억제한다는 점을 규명한다. 이를 바탕으로 MER-R1을 제안하며, 두 모드의 장점을 동시에 만족하도록 reinforcement learning에서 명시적 최적화로 연결한다. MER-R1은 recall·precision을 각각 분리해 jointly optimize하고, slow-thinking 출력이 fast-thinking의 ‘정답 confidence’는 유지하면서 ‘오답 억제’는 강화하도록 confidence calibration을 수행한다.

- **Technical Challenges**: 핵심 난점은 reasoning 모델에서 단일 F1류 보상으로는 recall과 precision이 뒤섞여 trade-off가 생기고, advantage 정규화 과정에서도 분산이 큰 목표가 다른 목표를 압도할 수 있다는 점이다. MER-R1은 dual-objective disentanglement으로 recall/precision 최적화 신호를 reward와 advantage 공간에서 분리해 이러한 간섭을 줄인다. 또 slow-fast confidence calibration은 카테고리 수준(감정 wheel 기반)에서 정답·오답에 대한 confidence를 서로 반대 방향으로 조정해, 느린 사고의 최종 답이 fast의 직관을 흡수하되 잡음성 과잉 커버리지는 흡수하지 않도록 설계한다.

- **Empirical Impact**: MER-R1은 MER-UniBench와 MME-Emotion에서 state-of-the-art 성능을 달성하며, 특히 MER-UniBench의 mean score를 83.50으로 끌어올리고 MME-Emotion에서는 전반 CoT 성능과 emotion recognition 점수를 동시에 개선한다. 또한 평가를 Hitrate 중심에서 F1 중심으로 바꾸거나, VideoAuto-R1의 answer-think-answer 방식과 비교해도 MER-R1만이 최종 slow-thinking 답의 인식 성능이 더 좋아지는 경향을 뒤집어 보여준다. 저자들은 이를 통해 해석 가능성 향상에 그치지 않고 reasoning이 실제 인식 정확도에 ‘진짜로’ 이득이 되도록 만드는 방법을 제시했다고 주장한다.



### DysLexLens: A Low-Resource LLM Framework for Analysing Dyslexic Learners Insights from Online Forums (https://arxiv.org/abs/2606.27619)
- **Prior Approaches**: 기존 연구는 크게 (1) 난독증 지원에 대한 기술·도구 현황을 정리하거나 (2) 레딧 같은 온라인 담화를 통해 난독증 당사자의 경험을 주로 정체성/사회적 의미 관점에서 분석하는 방향으로 나뉘었다. 하지만 두 접근 모두 AI 도구가 실제로 어떻게 인식·채택·검증되는지에 대한 학습 관련 근거를 자연스러운 포럼 맥락에서 체계적으로 연결해 주지 못했다.

- **Core Contribution**: 이 논문은 DysLexLens를 제안한다. DysLexLens는 저자원이면서 잡음이 많은 포럼 텍스트를 수집-필터링-지식그래프(KG) 추론-증거 추적-응답 평가의 end-to-end 파이프라인으로 묶어, AI 관련 발화의 학습적 유스케이스와 장단점을 근거 기반으로 분석하도록 한다.

- **Technical Challenges**: 핵심 난제는 (i) 관련 글이 희소하고 잡음이 많아 표본을 정밀하게 구성해야 하며 (ii) LLM이 만든 답을 실제 포럼 증거에 “traceable”하게 연결해야 한다는 점이다. DysLexLens는 딕셔너리 기반 필터링으로 Reddit 코퍼스를 319개 게시물 수준으로 좁힌 뒤, LLM이 KG triple과 의미 분석을 수행하고, 응답의 claim을 chunk ID로 매칭해 원문까지 역추적하는 evidence-tracing 파이프라인을 제공한다.

- **Empirical Impact**: 평가는 난독증- AI 관련 연구질문 RQ1~RQ5에 대한 30개 질의로 수행했으며, Answer Relevancy는 평균 0.75로 전반적 질의 의도를 잘 반영했다. 다만 Faithfulness·Context Relevance·Response Groundedness는 상대적으로 낮아(각각 평균 0.52/0.40/0.43) 추론은 맞지만 claim 수준에서의 근거 매칭과 시간 변화 질의 처리에 한계가 드러났다. 그럼에도 29개 응답에서 최소 1개 chunk ID가 포함되고, 인간 검증에서 39개 claim은 완전 검증 가능, 6개는 증거 부재로 비검증으로 분류되어, “근거 확인 가능한 생성” 방향의 실용성을 보여주며 Github 공개로 재현성도 강화했다.



### Odyssey: Constructing Verifiable Local Truth-Preserving Foundation Models (https://arxiv.org/abs/2606.27593)
Comments:
          34 pages

- **Prior Approaches**: 기존 foundation model 연구는 주로 체크포인트, 학습 파이프라인, 서비스/파인튜닝, 벤치마크 성적 중심으로 ‘재사용’을 설명해 왔다. 데이터 큐레이션·출처·불확실성·도메인 간 주장 이동의 실패 조건 같은 “유지보수 가능한 신뢰” 요소는 종종 암묵적이거나 프롬프트/코드에 의존했다. 또한 과학/인과 관련 자동화는 질문 응답이나 실험 설계에 강점이 있지만, 로컬 근거가 어떤 영역에서 합쳐지고(글루잉), 어디서 막히는지 같은 구조적 검증은 덜 명확했다.

- **Core Contribution**: ODYSSEY는 foundation model을 단일 임베딩이 아니라 sheaf-like 로컬 모델들의 “foundry(조립 가능한 제작 아티팩트)”로 구성하는 범주론 기반 프레임워크를 제안한다. foundry는 로컬 컨텍스트 커버, 로컬 표현 계열, restriction map, gluing rule, obstruction policy, 갱신 의무, 인간-facing 뷰를 패키지로 정의해 도메인별 주장 이동과 불일치를 지속 가능하게 만든다. 또한 Universal Foundry Learning(UFL)로 왼쪽 Kan 확장(로컬 아티팩트를 후보 foundry로 롤업)과 오른쪽 Kan 확장(제한·글루잉·장애·논증 조건을 강제)을 형식화하고, 외부 모델을 ODYSSEY 상태로 들이는 Foundry SQL(FSQL)·TICKET 절차를 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 로컬 근거를 유지한 채 도메인 간 “주장 운반”이 가능한지 판정하고, (2) gluing 실패·잔여 장애를 추적하며, (3) 외부/사전 모델을 같은 규약의 durable state로 승격(admit/promotion)하는 것이다. ODYSSEY는 Scylla/Homer/Athena/Prometheus/Toulmin 다중 에이전트로 요구-워크플로-표현 의미-로컬 월드모델 구현-톨민 논증을 분리하고, Athena의 유한 truth-value와 gluing 진단을 Prometheus가 Topos World Model로 감사하게 한다. TICKET은 Kan-extension transformer로 source의 로컬 구조를 타깃 foundry로 운반한 뒤, restriction·gluing·인과 방향/프로비넌스·promotion gate를 통과한 경우에만 유지 상태로 승격시키며, 그렇지 않으면 obstruction record로 격리한다.

- **Empirical Impact**: ODYSSEY는 다양한 concrete foundry(예: 매장/브랜드/기업 문서 워크플로, Amazon Reviews 2023, MyFixIt 수리 매뉴얼, Indus 스크립트, 연구 프로그램·assistant-build·evaluation-harness 등)에서 동일한 범주론적 기계로 도메인 구성, 아티팩트 재생, sheaf 진단, grounded Toulmin/local-LLM 스크루티니, residual-obstruction ledger까지 구현·검증했다고 주장한다. 특히 외부/사전 구조를 TICKET으로 들일 때 “무조건 승격”이 아니라 restriction·gluing·장애 가능성을 기록하며 감사 가능성을 유지하는 점이 실용적 의미로 강조된다. 튜토리얼(ICML 2026, 2.5시간)에서는 BRIDGE/SKFM의 causal-geometry 리파인먼트와 SkillOpt 연계를 통해, 엄격한 causal claim JSON 번들을 TICKET 상태와 함께 산출하는 실험 트랙도 소개된다.



### Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning (https://arxiv.org/abs/2606.27483)
- **Prior Approaches**: 기존 LLM 에이전트는 순차 의사결정에 강하지만, 장기 지평(long-horizon) 과제에서는 기본적으로 반응적(reactive)이다. 사람의 what-if 추론처럼 후보 계획을 먼저 시뮬레이션해 결과를 검증하는 내부 세계모델이 표준 에이전트에선 부족하다는 한계가 지적된다. 또 look-ahead 흔적을 기반으로 단순 fine-tuning 하면 겉보기의 foresight 흉내는 내도 실제 예측적 근거가 없는 ‘format-capability gap’ 문제가 발생한다.

- **Core Contribution**: 논문은 미래를 의식한 계획을 내부화하기 위해, 단일 autoregressive 모델이 (1) prospective state rollout을 서술하고 (2) 계획 조건부 성공 추정치를 텍스트로 생성하도록 제안한다. 이 성공 추정치는 Q-value의 텍스트적 유사체로 설계된다. 또한 단순 학습 포맷이 능력을 보장하지 않는 갭을 정면으로 겨냥해, 역량 중심의 학습 파이프라인을 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 ‘그럴듯한 시뮬레이션 문장’이 아니라 실제로 예측에 기반한(grounded) foresight를 만들고, 산출물의 보정(calibration)과 유용성(utility)을 맞추는 것이다. 이를 위해 세 단계 학습을 도입한다: WM-AMT로 정책에 잠재 예측 역량을 주입하고, FE-SFT로 그 형식을 강제해 역량이 발현되게 하며, FC-RL로 생성 시뮬레이션의 신뢰도와 유틸리티를 정련한다.

- **Empirical Impact**: 검색(search)과 수학적 추론(mathematical reasoning) 과제에서 제안한 방법이 다른 훈련 기준선들을 일관되게 능가한다. 결과는 LLM 에이전트의 내부 세계모델이 ‘형식만’이 아니라 capability-first 학습 파이프라인을 통해 근거 있고 보정된 foresight로 이어져야 함을 보여준다. 장기 지평 계획에서 에이전트의 신뢰 가능한 미래 추정이 중요해지는 흐름에 실증적 기여를 한다.



### When Does Personality Composition Matter for Multi-Agent LLM Teams? (https://arxiv.org/abs/2606.27443)
Comments:
          20 pages, 6 figures

- **Prior Approaches**: 기존 연구는 LLM에 Big Five 성격 특성을 prompting해 커뮤니케이션 양식이 달라지는지(예: high agreeableness는 협조적, low agreeableness는 적대적 언어)를 주로 관찰해왔다. 그러나 이런 ‘대화 스타일 변화’가 실제 과제 성과 같은 객관적 결과로 얼마나, 어떤 조건에서 이어지는지는 여러 도메인을 걸쳐 체계적으로 검증되지 않았다. 특히 코드처럼 산출물이 형식 제약을 받는 경우와 자연어 아이디어 생성·협상처럼 제약이 적은 경우를 분리해 본 실험이 부족했다.

- **Core Contribution**: 이 논문은 personality prompting이 멀티 에이전트 팀의 성과에 미치는 영향을 ‘과제 구조(형식 제약/목표 정렬)’ 관점에서 비교한다. structured coding, open-ended research collaboration, competitive bargaining의 3개 도메인에서 frontier LLM 팀에 대해 low-A와 high-A를 주입하고, 커뮤니케이션 변화가 성과로 전이되는 조건을 실험적으로 분해했다. 결과적으로 성격 효과는 과제 구조에 강하게 의존하며, 특히 저(低) agreeableness가 어떤 도메인에서는 성과를 크게 망가뜨린다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘성격이 바꾼 것은 대화 스타일뿐인지, 실제 결과도 바꾸는지’를 정밀하게 측정하는 것이다. 이를 위해 Bales Interaction Process Analysis의 범주를 원용해 questions/disagreements/suggestions/acknowledgments를 메시지 세그먼트 단위로 분류하고, 탐색(탐문·이견·제안)과 수렴(승인) 비율인 communication state φ를 계산했다. 또한 Goldberg의 부정가(valence)로 인한 안전/반응 편향 가능성을 neutral-paraphrase(같은 의도에 대한 중립적 문장)로 통제해, loaded adjectives가 결과에 일부 기여함을 분리했다.

- **Empirical Impact**: 실험 전반에서 low-A는 커뮤니케이션 탐색 비율 φ를 크게 끌어올리지만, 성과 영향은 도메인별로 갈린다. 코딩에서는 대화가 거칠어져도(대립/탐색 증가) 마일스톤 완료가 대체로 유지되거나 모델에 따라 미미하며, 반대로 연구 협업과 협상에서는 마일스톤·합의 가능성이 크게 악화된다(협상은 저-A에서 합의가 거의 붕괴). high-A는 대부분의 경우 커뮤니케이션·성과에 유의미한 변화를 만들지 못해 ‘low-A 효과와 high-A 무효과’의 비대칭이 관찰된다. 설계 관점에서 멀티 에이전트 시스템은 personality를 일괄적으로 조정하기보다 과제 매체의 구조(형식 제약·목표 경쟁 여부)를 고려해 제어해야 하며, 부정가가 섞인 성격 단어는 프롬프트 설계 시 예측 불가능한 부작용을 유발할 수 있다.



### AI-Model Network: Concept, Current State and Futur (https://arxiv.org/abs/2606.27382)
Comments:
          31 pages, 14 figures

- **Prior Approaches**: 기존 연구는 단일 large model(LM) 성능을 끌어올리거나, 여러 모델을 단순 앙상블·파이프라인 형태로 묶어 활용하는 데 집중해 왔다. 하지만 모델별로 비용과 배포 복잡도가 달라지고, 경량·프라이빗·도메인 특화 수요가 늘면서 이종 모델이 급증했음에도 상호작용과 협업을 체계화하는 데는 한계가 있었다. 특히 모델 간 연결 방식이 정교하지 않아 능력 공유와 공동 추론이 쉽게 깨지는 문제가 남아 있다.

- **Core Contribution**: 이 논문은 인터넷의 ‘연결과 협업’ 관점을 AI로 확장해 world wide AI-model network(AI-ModelNet)라는 개념과 비전, 시스템 아키텍처를 제안한다. AI-ModelNet은 모델 사이에 경로(pathway)를 구축해 상호 연결(interconnection), 능력 공유(capability sharing), 협업적 추론(collaborative reasoning)을 가능하게 하는 새로운 패러다임을 목표로 한다. 또한 단일 모델 중심과 멀티 모델 연구 흐름을 정리한 뒤, 계층적(hierarchical) 구조로 전체 비전을 구체화한다.

- **Technical Challenges**: 핵심 기술 난제는 이종 모델이 많아질수록 어떻게 ‘효과적인 상호작용’과 ‘협업적 추론’이 성립하도록 경로를 설계·운영하느냐에 있다. 논문은 이를 위해 AI-ModelNet의 계층적 아키텍처를 제시하고, 모델 간 연결을 통해 능력을 조합하는 구조를 시스템적으로 구현하는 방향을 택한다. 아울러 개념의 실현 가능성을 프로토타입 시스템과 다양한 응용 사례로 검증해, 실행 단계에서의 제약을 점검한다.

- **Empirical Impact**: 논문은 프로토타입 구현과 다양한 application case를 통해 AI-ModelNet 프레임워크의 실현 가능성과 활용 잠재력을 경험적으로 보여준다. 결과적으로 LM의 높은 학습 비용과 복잡한 배포 부담을 완화하면서도, 여러 도메인·경량 모델을 함께 쓰는 협업 메커니즘을 제공할 수 있다는 점에서 의미가 있다. 향후 연구 방향도 함께 제시해, 이종 모델 생태계에서의 표준화된 협업 연구로 이어질 기반을 마련한다.



### DexCompose: Reusing Dexterous Policies for Multi-Task Manipulation with a Single Hand (https://arxiv.org/abs/2606.28323)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존에는 (1) 재사용 가능한 dexterous 컨트롤러를 새로 학습하거나, (2) 여러 스킬을 긴 구간으로 순차 결합하며, (3) 손의 자유도를 손가락/자원으로 나눠 멀티 상호작용을 만드는 연구가 많았습니다. 그러나 이미 학습된 full-hand 정책을 그대로 재사용하면서 두 과제를 동시에 수행하도록 안전하게 결합하는 방법은 제한적이어서, 단순 policy chaining이 공용 손가락 동작을 덮어쓰며 성능이 쉽게 무너집니다.

- **Core Contribution**: 이 논문은 DexCompose로, pretrained dexterous 정책 2개를 결합할 때 손가락 단위 action ownership을 명시적으로 부여해 cross-task interference를 줄이는 프레임워크를 제안합니다. 또한 Task A(기존 유지 과제)를 보존하는 residual stabilizer와, Task B(새 상호작용 과제)를 위해 할당된 하위 action subspace에서만 적응하는 context-aware residual을 이중으로 학습/적용합니다.

- **Technical Challenges**: 핵심 기술 과제는 (i) 어떤 손가락(DoF)이 Task A의 결과를 유지하는 데 실제로 필요한지 자동으로 찾아내고, (ii) 이후 Task B가 그 필수 손가락을 건드리지 않으면서도 필요한 경우에만 충분히 보정하도록 제어 범위를 제한하는 것입니다. DexCompose는 Task A의 성공 후 상태를 모은 뒤 release tests로 손가락 마스크를 후처리(discovery)하고, 그 마스크에 따라 잔차(residual)를 bounded stabilizer/할당된 subspace용 residual로 분리해 학습합니다.

- **Empirical Impact**: 시뮬레이션에서 4가지 hold-and-retain 과제와 4가지 downstream 상호작용을 조합해 총 16개 composite task를 평가했으며, 평균 composite success rate 77.4%를 달성했습니다. 이는 policy chaining 대비 15.8%p 큰 개선이며, 특히 dual residual stabilizer가 유지 실패를 막는 데 가장 큰 역할을 했고 finger attribution은 동일 예산에서 추가적인 간섭 감소 효과를 보였습니다.



### Which Nash Equilibrium? Solver-Dependent Selection on Zero-Sum Nash Polytopes (https://arxiv.org/abs/2606.28308)
Comments:
          18 pages, 9 figures

- **Prior Approaches**: 기존 게임 솔버는 2인 영점합 게임에서 Nash equilibrium을 ‘유일한 목표’처럼 취급해 왔습니다. 특히 Nash 집합이 다면체(폴리토프)처럼 여러 해를 가질 때도, 솔버 간 선택이 다르다는 점은 거의 다루지 않았습니다.

- **Core Contribution**: 이 논문은 서로 다른 알고리즘(특히 regret-averaging 계열 vs regularized last-iterate 계열)이 Nash 집합의 서로 다른 ‘멤버’를 실제로 선택하는지, 그리고 그 선택이 알고리즘 고유의 성질인지 체계적으로 분석합니다. 탭룰러리(정확 해석) 게임 6종과 대규모 랜덤 앙상블을 통해 선택이 seed가 아니라 알고리즘에 의해 결정됨을 보여줍니다.

- **Technical Challenges**: 핵심 난제는 (1) Nash 집합이 정확히 알려진 테스트베드를 구성해 ‘정답 멤버’를 검증하고, (2) 솔버의 업데이트 규칙이 어떤 기하학적 선택 원리를 따르는지(예: 최대엔트로피/I-projection) 구분하는 것입니다. 저자들은 entropic 정규화의 last-iterate 동역학에서 기준 정책(annealing)과의 KL 투영 관계를 확인하고, boundary drift에 대한 흔한 직관(positive-orthant projection)이 실제 원인이 아님을 ablation으로 반박했습니다.

- **Empirical Impact**: 결과적으로 regularized last-iterate 방법(R-NaD, magnetic mirror descent)은 Nash 집합에서 maximum-entropy 멤버(정보투영, I-projection)를 선택하는 경향이 강합니다. 반면 CFR/CFR+ 등 regret-averaging은 더 낮은 엔트로피의 face로 이동하며, 랜덤 180게임 앙상블에서 R-NaD는 수렴한 모든 게임에서 최대엔트로피 멤버를 선택(100%), CFR+는 94%에서 그보다 낮게 나타났습니다. 또한 선택된 멤버의 ‘상대방 견딤(헤지)’은 sequential/hidden-information 구조에 따라 달라지되, 절대 성능 차이는 제한적임을 보여 후속 응용 관점의 의미를 제시합니다.



### Agentic Hardware Design as Repository-Level Code Evolution (https://arxiv.org/abs/2606.28279)
- **Prior Approaches**: 기존 RTL 생성·수정 연구는 모델을 fine-tuning하거나 데이터/추론 품질을 높여 첫 시도 정확도를 끌어올리는 데 집중해 왔습니다. 또 다른 계열은 generate-compile-simulate 같은 도구 사용과 디버깅 루프를 붙였지만, 보통은 개별 모듈 단위의 리페어 파이프라인에 머물러 “검증 가능한 저장소 단위로 문제를 진화”시키는 방식은 제한적이었습니다. 한편 코드베이스 자체를 자기진화시키는 repository-scale 접근은 있었으나, 목표 대상이 하드웨어 설계(정합성·타이밍·검증 아티팩트)를 담보하는 형태로 확장되진 못했습니다.

- **Core Contribution**: HORIZON은 하드웨어 설계를 repository-level code evolution처럼 다루는 self-evolving 에이전트 프레임워크를 제안합니다. 사용자는 structured Markdown harness를 제공하고, 부트스트랩 에이전트가 이를 domain knowledge·executable evaluator·acceptance predicate·git/runtime policy를 포함한 project pack으로 컴파일한 뒤, 이후 hands-free 루프로 git worktree 안에서 RTL을 진화시키며 합격 조건을 통과한 버전만 커밋합니다. 핵심은 하드웨어 작업을 “일회성 프롬프트”가 아니라 “버전 관리 + 실행 기반 정답 게이트”로 캡슐화해 벤치마크 전체 수렴을 관찰 가능하게 만든 점입니다.

- **Technical Challenges**: 하드웨어 RTL은 단순히 문법이 맞는 수준을 넘어 사이클 단위 동작, reset/인터페이스 규약, 비트 폭, 코너 케이스까지 시뮬레이터 피드백으로 검증되어야 해서 실행 루프가 필수입니다. HORIZON은 이 문제를 git worktree에 문제를 self-contained으로 호스팅하고, diffs/commits/logs/notes에 실행 증거를 남기며, evaluator가 낸 결과를 acceptance predicate로 판정해 next iteration을 통제하는 구조로 해결합니다. 또한 토큰 비용을 줄이기 위해 프롬프트 캐시 재사용(세션 유지)을 활용하고, 정책 학습 없이 고정된 backbone으로 campaign 내 “수렴 경로”를 추적·분석합니다.

- **Empirical Impact**: GPT-5.3 고정 backbone으로 ChipBench, RTLLM, Verilog-Eval 및 CVDP 9개 카테고리를 대상으로 실험한 결과, hands-free agentic loop만으로 모든 스위트에서 100% benchmark completion을 달성했습니다(단, ChipBench의 단일 실패는 에이전트가 아니라 벤치마크의 specification–harness 불일치로 추적). 첫 반복(pass rate)은 약 47.8%로 낮게 시작하지만, 이후 반복된 실행 피드백을 통해 CVDP의 어려운 카테고리도 수렴해 “도착지”뿐 아니라 “수렴 경로”가 관측됩니다. 다만 보상 신호(디버그 로그·시뮬레이터 메시지)에 과도하게 적응하는 over-solving/reward hacking 위험과, 최종 평가에서의 hidden 랜덤 테스트 등 2단계 프로토콜 필요성이 향후 과제로 제시됩니다.



### Towards Automating Scientific Review with Google's Paper Assistant Too (https://arxiv.org/abs/2606.28277)
- **Prior Approaches**: 기존에는 LLM을 논문 전체에 한 번 호출해 결함을 찾는 방식이나, Pass@k처럼 여러 번 생성한 뒤 합치는 접근이 주로 거론됐다. 하지만 이 방법들은 문맥 제약 때문에 깊은 검증이 어려우면서도, 환각으로 정밀도가 급락해 사람이 쓸모없는 지적을 걸러내야 하는 문제가 컸다. 또 무작위 호출 분산이라 논문의 어느 구간은 과소검증되고 다른 구간에 자원이 쏠릴 수 있다.

- **Core Contribution**: 이 논문은 AI가 과학 검증/리뷰를 대신하는 단계적 역할 분류(4단계)를 제안하고, 그중 “Paper Assistant Tool(PAT)”로 실제 구현 사례를 제시한다. PAT는 수학 및 컴퓨터과학 논문을 대상으로 이론/논리 오류와 실험 검증 문제를 중심으로, 개선점과 잠재적 결함을 체계적으로 산출한다. 핵심은 단일 호출을 넘어 인퍼런스 스케일링을 파이프라인 오케스트레이션에 결합해 더 깊은 오류 탐지 확률을 끌어올린다는 점이다.

- **Technical Challenges**: 문제는 긴 논문 전체를 한 번에 깊게 검증하는 데 필요한 “생각 토큰”과 문맥 한계가 충돌한다는 것이다. PAT는 세그먼터(agent)가 논문을 논리 주제별로 분할(겹침/비연속 가능)하고, 각 세그먼트의 정보 밀도·복잡도에 따라 계산 예산을 동적으로 배분한 뒤, Deep Review agent들이 각 구간을 검증하게 하며, 마지막 합성 에이전트가 중복 제거와 Google search 기반 근거 확인으로 정밀도 저하를 완화한다. 그 결과 단순 Pass@k의 정밀도 하락과 문맥 편향을 동시에 겨냥한다.

- **Empirical Impact**: SPOT 벤치마크에서 수학/증명 오류 하위셋(26편, 29오류)을 사용한 실험에서 PAT는 zero-shot 단일 호출 대비 수학 오류 탐지 recall을 34%p 개선해 89.7%까지 끌어올렸다. 또한 STOC·ICML에 사전 제출용 도구로 시범 적용한 결과, 대부분의 저자가 유용성을 높게 평가했으며(90% 이상이 도움이 되었다고 응답), 일부는 이론 결과의 유의미한 오류를 수정해야 했다고 답했다. 특히 ICML에서는 PAT가 “완전히 새로운 실험”으로 이어졌다는 응답이 31%로 나타나, 오류 탐지뿐 아니라 실험 설계 품질 개선에도 실질적 영향이 있음을 시사한다.



### Parameter Efficient Hybrid Transformer (PEHT) for Network Traffic Prediction via Dynamic Urban Congestion Integration (https://arxiv.org/abs/2606.28274)
- **Prior Approaches**: 기존 네트워크 트래픽 예측은 GNN 기반이든 Transformer 기반이든 공간-시간 패턴을 학습하려 했지만, GNN은 고밀도 도시에서 확장성 문제가 생기고 Transformer는 고차원 입력에서 파라미터·연산 비용이 커지는 한계가 컸다. 또한 외부 도시 이동성·혼잡 신호를 통합하려 해도 예측 시점의 시간 인과성을 어기거나(데이터 누수) 통합이 충분히 효과적이지 못했다.

- **Core Contribution**: 이 논문은 Parameter-Efficient Hybrid Transformer(PEHT)를 제안해, 도시의 이동성과 혼잡 정보를 트래픽 예측에 구조적으로 반영한다. 입력 특징을 ‘주요 네트워크 통신 특징’과 ‘부가적 도시 이동성 특징’으로 분리하고, Transformer Encoder에는 LoRA를 넣어 학습 파라미터를 크게 줄이면서도 정확도를 유지한다.

- **Technical Challenges**: 핵심 기술적 도전은 집계 데이터가 사용자 수준 이동 궤적이 아니라 격자 단위 신호라는 점과, 외부 특징을 넣되 추론 시 시간 누수가 없어야 한다는 요구다. 이를 위해 격자-가상 기지국(Grid-to-Virtual-Cell) 매핑으로 시계열을 안정화하고, Encoder 출력과 외부 이동성·혼잡 특징을 Decoder에 융합하되 과거 또는 예측으로만 정렬해 causality를 보장하도록 설계했다.

- **Empirical Impact**: Telecom Italia Milan 데이터와 CARLA 기반 합성 혼잡 시나리오에서 PEHT는 RMSE·MAE·R^2에서 최신 baseline을 전반적으로 앞섰다. 특히 RMSE에서 SMS는 14.6%, Internet은 11.9% 개선을 보였고, LoRA 단독보다 Encoder 융합까지 포함한 Full 모델이 대부분 지표에서 일관되게 강한 성능을 냈다. 또한 LoRA로 학습 파라미터를 수만~수십만 수준으로 줄여 대규모 도시 배치의 계산 가능성을 높였다는 점에서 실용적 의미가 크다.



### Learning Topology-Aware Representations via Test-Time Adaptation for Anomaly Segmentation (https://arxiv.org/abs/2606.28268)
- **Prior Approaches**: Test-time adaptation(TTA)는 분포 변화 상황에서 딥 모델을 보정하는 유망한 방법이지만, 이상 탐지·세그멘테이션(AS)에서는 픽셀 단위 마스크를 만들기 위해 confidence thresholding이나 entropy minimisation 같은 픽셀 수준 휴리스틱에 의존하는 경우가 많습니다. 이런 방식은 잡음과 텍스처 변동이 생기면 구조적 일관성을 잃고, 이상 맵을 단순한 강도(field)로 취급해 연결성·구멍 같은 고차 공간 관계를 반영하지 못합니다. 또한 기준선으로 쓰는 임계값이 정상 데이터에 고정돼 있어, 다양한 결함 형상으로의 일반화가 흔들립니다.

- **Core Contribution**: TopoTTA(Topological Test-Time Adaptation)는 TTA 파이프라인에 persistent homology(지속동형, PH) 기반 위상 정보를 결합해 적응 중에도 기하·구조적 일관성을 유지하도록 설계했습니다. 이상 점수 맵에서 multi-level cubical complex filtration을 적용해 견고한 topological pseudo-label을 만들고, 이를 이용해 백본을 retraining하지 않으면서도 경량 test-time classifier가 마스크를 정교화하도록 유도합니다. 특히 특정 점수 임계값으로 마스크를 이진화하는 방식에서 벗어나, 연결성과 위상적 구조를 보존하는 데 초점을 둡니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 잡음/텍스처 변화가 있는 이상 점수 맵에서 위상 구조를 얼마나 안정적으로 뽑아 pseudo-label로 변환할지, (2) 그 pseudo-label이 픽셀 단위 세그멘테이션 학습에 실제로 유효하도록 만드는 것입니다. TopoTTA는 sublevel과 superlevel을 함께 쓰는 bidirectional multi-level cubical filtration로 connected components와 holes 같은 구조를 다중 스케일로 포착하고, 가장 지속적인(top-k 또는 persistence 기준) 특징만 남겨 pseudo-label을 구성합니다. 이후 frozen pre-trained backbone의 feature로부터 픽셀 레벨 contrastive encoder(PCES)를 test time에 on-the-fly로 학습해, 이상/정상 영역의 특징을 pseudo-label 공간에서 정렬·분리하도록 학습합니다.

- **Empirical Impact**: TopoTTA는 MVTec AD, VisA, Real-IAD, MVTec 3D-AD, AnomalyShapeNet, MVTec LOCO 등 6개 벤치마크에서 평균적으로 F1이 SOTA 대비 약 15% 향상됐고, 복잡한 기하·구조 변화를 보이는 이상에서 개선 폭이 특히 컸습니다. 다른 실험에서도 2D에서 최대 +20.3%, 3D에서 +10.2%의 F1 개선을 보고해, 2D/3D 모달리티와 여러 백본 전반에서 일관된 성능을 확인했습니다. 위상 기반 구조 추론을 TTA에 통합함으로써 기하 학습과 견고한 적응 사이의 간극을 메운다는 점에서, 산업용 anomaly segmentation의 일반화 방향성을 제시합니다.



### How Width and Data Shape Generalization Scaling Laws in Quadratic Neural Networks (https://arxiv.org/abs/2606.28242)
- **Prior Approaches**: 기존 scaling laws 이론은 일반화 오차를 데이터 수 또는 compute에 주로 연결하되, 고정된 표현(random features 등)이나 이상화된(infinite-width) 설정, 혹은 online SGD 같은 특정 최적화 틀에 머무는 경우가 많았습니다. 한편 모델 크기가 최적화/표현력에 주는 영향만 다루는 연구도 있었지만, 데이터 구조에 의존하는 generalization을 model size와 함께 정면으로 분해해 설명하긴 어려웠습니다.

- **Core Contribution**: 이 논문은 feature-learning 상황에서 “trainable parameters 수(p)–샘플 수(n)–regularization(λ~)–타깃 스펙트럼(γ)”이 공동으로 일반화 오차를 어떻게 스케일하는지에 대한 정량 이론을 제시합니다. 이를 위해 l2-정규화 ERM을 2-layer quadratic neural network(2차 활성, 유효 rank가 width에 의해 제한됨)로 최소화해, excess test error의 위상도(phase diagram)를 분석합니다.

- **Technical Challenges**: 핵심 난제는 width 제약이 암묵적으로 ‘rank constraint’를 만들면서 동시에 원래 가중치 파라미터 문제를 비볼록 행렬 최적화로 바꿔버린다는 점입니다. 저자들은 네트워크를 구조화된 선형 모델/행렬 압축 감지 형태로 재해석하고, AMP(Approximate Message Passing) 기반의 state-evolution을 비동기적(비-엄밀) 확장처럼 적용해 test/train 오차를 closed-form에 가까운 예측식으로 연결한 뒤, gradient-based 학습 결과와 대조합니다.

- **Empirical Impact**: 이론 예측은 비교적 작은 차원(d=400)과 매우 작은 width(p=1)에서도 gradient descent 수렴 시의 generalization error와 높은 수준으로 일치해, width가 단순 capacity가 아니라 암묵적 regularizer로 작동함을 실증적으로 뒷받침합니다. 특히 width에 따라 low-width decay(타깃의 power-law 스펙트럼에 의해 지배), interpolation 이후 구간의 noise overfitting 등 서로 다른 스케일링 레짐이 나타나며, 최적 width 또는 explicit regularization 조정이 Bayes-optimal rate까지 도달할 수 있음을 보여줍니다.



### Govern the Repository, Not the Agent: Measuring Ecosystem-Level Risk in AI-Native Softwar (https://arxiv.org/abs/2606.28235)
- **Prior Approaches**: 기존 평가는 에이전트가 PR을 열고 테스트를 통과하는지 여부에 기대어, 에이전트별·구성요소별로 독립된 벤치마크 태스크를 평가하는 방식이 중심이었다. 하지만 실제로는 PR은 문제없이 통과해도 저장소 전체가 누적된 고충(느린 머지, 반복 리뷰, 머지 충돌 등)을 겪어 ‘전체 상태’를 설명하기 어렵다는 지적이 있었다.

- **Core Contribution**: 이 논문은 통과 여부 이후에 남는 통합 비용인 integration friction이 개별 에이전트의 능력 문제가 아니라 저장소 생태계의 성질일 수 있는지 묻고, 이를 수치로 분해해 검증한다. 930,000개 이상 에이전트 작성 PR을 바탕으로, 기여(작성자·크기·에이전트)들을 설명한 뒤에도 저장소 수준 변동이 얼마나 남는지를 비환원성(non-reducibility) 형태로 측정한다.

- **Technical Challenges**: 핵심은 ‘어떤 변동이 어디에 귀속되는가’를 임의의 성공 기준 없이 판별하는 것으로, 연구팀은 저장소 랜덤 효과를 포함한 multilevel model로 변동을 저장소 간/저장소 내로 분할해 ICC로 정량화한다. 또한 같은 저장소에서 인간 PR과 비교해 에이전트가 아닌 일반적인 고활동 저장소 특성인지 분리했고, codebase size·age, 작업 형태, 프로세스 성숙도, merge path 등 강한 통제를 거쳐도 저장소 수준 신호가 크게 유지되는지 확인했다.

- **Empirical Impact**: 결과적으로 integration friction 변동의 약 절반이 저장소 수준에 남았고, 이는 강한 통제 후에도 유지되어 ‘부분만으로는 설명되지 않는’ 생태계적 위험 신호로 해석된다. 같은 저장소에서 에이전트 작성 기여는 인간보다 저장소 수준 비환원성을 약 2배 더 크게 만드는 것으로 나타났고(ICC 0.30 vs 0.16), 저 위험은 개별 에이전트가 아니라 에이전트-인간-자동화가 얽힌 생태계의 속성임을 시사한다.



### Exposure Bias Can Alleviate Itself via Directional and Frequency Rectification in Flow Matching (https://arxiv.org/abs/2606.28226)
Comments:
          arXiv admin note: text overlap with arXiv:2512.04904

- **Prior Approaches**: Flow Matching(FM)은 연속시간 흐름을 학습해 효율적인 생성 성능을 보여줬지만, 학습 시의 섞인 입력(잡음+데이터)과 추론 시의 재귀 예측 입력 사이 불일치로 exposure bias가 발생한다. 기존 대응은 DDPM/IP 계열처럼 노이즈 주입이나 정적 정렬, 혹은 추론 안정화(예: scheduled sampling) 중심이어서 편차의 ‘정도’에 따라 능동적으로 되돌리는 제어 신호를 제공하긴 어렵다.

- **Core Contribution**: 이 논문은 exposure bias 자체가 방향성과 주파수(특히 저주파) 결핍을 담고 있다는 관찰에 기반해, bias를 이용해 bias를 교정하는 DEFAR를 제안한다. DEFAR는 학습 중 단일-step 추론 시뮬레이션으로 bias를 추출하고, 이를 두 축(방향/주파수)의 피드백으로 재활용해 모델의 편차 내성을 높인다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 재귀적 드리프트를 단일-step 학습 신호로 안정적으로 분해해 ‘바르게 되돌릴 방향’을 학습하는 것과 (2) 그 드리프트가 주파수 대역 결핍으로 어떻게 연결되는지 정량화하는 것이다. 저자들은 ADR에서 드리프트된 상태를 목표 엔드포인트로 되돌리는 복원 방향(각도 기반으로 신호 강도를 동적으로 조절)을 정규화하고, FC에서는 Predicted Frequency Ratio(PFR)/Frequency Emphasis of Loss(FEL) 분석을 통해 bias가 저주파 구조를 보완하는 역보완 성질을 가진다는 점을 이용해 결핍 저주파를 강화하도록 손실을 bias-가중 재조정한다.

- **Empirical Impact**: CIFAR-10, CelebA-64, ImageNet-256/512에서 DEFAR는 기존 베이스라인을 능가했으며, NFE 50 기준으로도 성능과 추론 견고성, 확장성에서 이점을 보였다. 또한 FM의 exposure bias를 ‘정적 보정’이 아니라 bias 내부 신호로 ‘자기 직정(self-rectification)’하는 관점이 제시돼, 향후 FM 및 유사 생성모델의 inference-robust 학습 설계에 의미 있는 방향을 제공한다.



### Towards Value-Constrained Credit Assignment in Fully Delegated AI Cooperatives (https://arxiv.org/abs/2606.28217)
- **Prior Approaches**: 기존 데이터 가치평가·영향력 기반 attribution은 모델 성능에 대한 기여를 계산하지만, 사람(주체)마다 다른 가치 제약을 반영한 ‘협동 보상 규칙’으로 바로 쓰기 어렵다. 또한 FL 기여 추정·인센티브 연구는 대개 집계된 클라이언트 업데이트 이후에 기여를 재구성하는 데 머물러, 가치 허용(admissibility) 여부를 학습 루프 안에서 선별하기는 어렵다.
개인화 FL은 서로 다른 목표를 다루지만 주로 예측 성향의 개인화에 초점이 있고, pluralistic alignment처럼 ‘규범적 수용 가능성’을 학습과 보상에 직접 내장하는 방식은 상대적으로 미흡하다.

- **Core Contribution**: 이 논문은 fully delegated AI cooperative에서 보상 배분을 ‘가치 프로필에 따라 허용되는 업데이트만 신용(credit)’하는 규칙으로 제시한다. 각 principal(사람/주체)의 가치 프로필로 업데이트를 필터링한 뒤, 그 admissible 업데이트가 검증 품질에 만든 한계 개선분을 기여 신호로 정의하고 누적 정산(revenue settlement)한다.
특히 여러 가치 제약이 동시에 존재할 때, 가치에 맞지 않는 업데이트는 학습에 기여가 되더라도 보상에서는 제외되도록 설계해 지속 가능한 인센티브 구조를 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 위임 시점에서 가치 프로필을 어떻게 표현·구현할지, (2) 학습 중 각 에이전트 업데이트의 ‘가치 허용 여부’를 먼저 판정한 다음 (3) 그 뒤에만 검증 성능 기여도를 계산하는 절차를 어떻게 일관되게 연결할지다. 논문은 value-conditioned gradient filtering(규칙 기반/feasible-set projection/gradient modification/선호 기반 학습된 admissibility model 등 다양한 필터 형태)을 통해 admissible gradient만 통과시키는 구조로 해결한다.
또한 TL(traversal learning)을 기계적 기반으로 써서, 집계 중심 FL보다 업데이트 경로를 더 명시적으로 유지해 online 기여 신호와 반사실적(one-step counterfactual) 한계 개선 계산을 연결한다.

- **Empirical Impact**: 제시된 프레임워크는 특정 실험 결과 수치보다는, 가치 선별-기여 신호-누적 정산을 하나의 학습 메커니즘으로 통합할 수 있다는 점에서 임팩트가 강조된다. inadmissible 방향은 필터로 인해 cit가 0이 되므로, 보상이 ‘가치 수용 가능성과 성능 개선’을 동시에 만족하는 업데이트에만 귀속되도록 만드는 것이 핵심 실용성이다.
또한 전역 최적 수렴이 흔들리더라도(가치별로 서로 다른 업데이트 허용), 각 스텝의 로컬 credit signal은 계속 계산 가능해 회계 horizon에 대한 정산이 성립한다는 관점을 제공한다.



### HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration (https://arxiv.org/abs/2606.28215)
Comments:
          Accepted to ECCV 2026. 15 pages of main text and 39 pages of appendices. Project page: this https URL

- **Prior Approaches**: 기존 모노큘러 기반 4D 상호작용 복원은 크게 (1) 동기화 다중 카메라 하드웨어에 의존해 occlusion을 줄이거나, (2) 단일 영상에서 생성모델/재구성모델로 4D를 만드려는 시도로 나뉜다. 전자는 비용·제약이 커 오픈월드 확장성이 떨어지고, 후자는 고정/스타일화 자산 위주라 현실의 복잡한 다중 물체 상호작용에서 물리적으로 그럴듯하지 않거나 시간적 흔들림이 생긴다.

- **Core Contribution**: 이 논문은 HAT-4D라는 인간-에이전트 협업(agentic) 프레임워크로, 단일 모노큘러 비디오에서 다중 물체의 3D 기하·시간 동역학·물리적 상호작용을 동시에 복원한다. 핵심은 Interaction Knowledge Graph (IKG)로 장기 물리 변화와 상호작용 단서를 구조화해, 3D 생성/배치와 4D propagation이 같은 제약을 따라가도록 만드는 점이다. 또한 비싼 멀티카메라 없이도 “물리적으로 그럴듯한” 4D 에셋을 대량 생성하는 데이터 엔진을 제시한다.

- **Technical Challenges**: 모노큘러 영상은 깊이 모호성과 상호 occlusion이 심해, 복원 과정에서 오류가 누적되면 시간 일관성이 쉽게 무너진다. 논문은 IKG로 깊이/관계/이벤트 경계를 명시하고, 메모리 뱅크 기반 segment-wise 4D propagation과 4D 생성 평가자(critic)로 물리 위반·장단기 메모리 붕괴를 감지해 국소 재생성 또는 단계 롤백을 수행한다. 여기에 다단계 human-in-the-loop 수정(gaussian/region/object 수준)과 온라인 fine-tuning을 결합해 모호한 상황에서 인간 지식을 생성 루프에 주입한다.

- **Empirical Impact**: MVOIK-4D는 77개 태스크·112개 상호작용 시나리오로 구성된 오픈월드 벤치마크이며, 변형 현실성·상호작용 일관성·시간 매끄러움·장기/교차뷰 메모리 보존을 포함한 다차원 평가를 제공한다. 실험 결과 HAT-4D는 대부분의 지표에서 SOTA 성능을 보이고, 변형/관계/메모리 일관성 측면에서 특히 강한 개선을 보인다. 또한 소량의 인간 피드백 도입만으로 상호작용 복원이 향상되며, 생성된 데이터는 fine-tuning에 활용 시 기존 baseline 성능을 끌어올리는 것으로 입증된다.



### The Remittance Blueprint: Data-driven Intelligence for Sri Lanka (https://arxiv.org/abs/2606.28190)
Comments:
          7 pages, 4 figures

- **Prior Approaches**: 기존 연구는 이주·송금의 거시적 push-pull을 설명하더라도, 주로 기술통계나 단년도·횡단면 중심의 분석에 머물러 비정상성 보정과 거시 충격의 시간 해상도를 충분히 반영하지 못했다. 또한 일부는 VAR류를 쓰더라도 장기 균형(공적분) 정보를 놓치거나, 달러·유가 같은 외부 변수를 예측모델에 일관되게 통합하지 못했다. 결과적으로 2022년 위기처럼 환율 변동이 급격한 구간에서 송금 흐름을 안정적으로 예측하기 어려웠다.

- **Core Contribution**: 이 논문은 1994~2025년 32년치, 8개 출처를 통합한 월 단위 384개월 조화(harmonized) 데이터로 스리랑카 이주와 송금을 한 프레임에 묶어 분석한다. ADF·공적분(Johansen)·VECM·Granger 인과검정으로 비정상성과 장기 관계를 반영하면서, 인구(출국 규모/성·기술 구성)와 거시(환율·유가 등)를 동시에 연결한다. 나아가 2026년 송금 유입을 Ridge Regression으로 예측해 기존 SARIMA 대비 성능 격차를 수치로 제시한다.

- **Technical Challenges**: 핵심 기술 과제는 비정상 시계열에서의 인과/예측 오류를 줄이는 것이었다. 저자들은 1차 차분으로 I(1) 구조를 정리하고 VECM으로 장기 균형을 복원했으며, 위기 시기의 극단값은 단순 클리핑 대신 보존·특징화해 모델이 구조적 변화를 학습하도록 설계했다. 예측에서는 다변량 변수 간 공선성을 완화하기 위해 L2 정규화 Ridge Regression을 쓰고, walk-forward 검증으로 과거정보 누설(look-ahead bias)을 방지했다.

- **Empirical Impact**: 분석 결과 송금 유입의 단기 변동은 국내 지표보다 외부 거시변수—특히 USD/LKR 환율과 Brent 유가—의 영향이 지배적이었다. IRF와 분산분해는 유가 충격이 이후 12개월가량 송금을 지속적으로 끌어올리지만, 환율 감가충격은 초기엔 공식 유입을 감소시키는 비대칭 효과가 있음을 확인했다. 예측 성능에서는 Ridge Regression이 SARIMA 대비 연환산 RMSE를 크게 낮추며(73.8% 개선) 2026년 송금을 안정 조건 하 USD 9,001 million으로 전망해, 외환부문·준비금 관리에 바로 활용 가능한 기준선을 제공한다.



### Cognitive Episodes in LLM Reasoning Traces Enable Interpretable Human Item Difficulty Prediction (https://arxiv.org/abs/2606.28186)
Comments:
          32 pages, 8 figures, 10 tables

- **Prior Approaches**: 기존 연구는 문항 텍스트 기반의 item-level 표현이나, 사람의 비싼 보정(calibration)에 의존해 인간의 문항 난이도를 추정하는 경우가 많았습니다. 그 결과 난이도가 어떤 인지적 과정에서 생기는지에 대한 증거가 제한적이며, 단순 추정에 머무는 한계가 있었습니다.

- **Core Contribution**: 이 논문은 난이도를 문항 텍스트의 속성뿐 아니라, 문항이 유발하는 문제풀이 부담의 ‘관측 가능한 결과’로 재정의합니다. Large Reasoning Models(LRM)의 추론 흔적을 사람이 해석 가능한 인지 에피소드(episode) 시퀀스로 구조화하는 Epi2Diff(Episode to Difficulty) 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 난제는 LRM의 긴 reasoning trace를 그대로 쓰기보다, 기능적 문제풀이 상태로 묶어 해석 가능한 모델링 단위로 변환하는 것입니다. Epi2Diff는 trace 세그먼트를 인지적으로 근거 있는 episode로 군집화하고, reasoning scale·effort allocation·state transition 같은 에피소드-동적 특징을 간단한 표현으로 추출한 뒤 의미적 문항 표현과 결합해 난이도를 예측합니다.

- **Empirical Impact**: 4개의 실제 human difficulty 데이터셋에서 Epi2Diff는 fine-tuned 소형 언어모델, LLM in-context learning, supervised LLM adaptation 등 강력한 베이스라인을 일관되게 능가했습니다. 특히 SAT에서 만든 분류 벤치마크에서는 supervised LLM fine-tuning 대비 평균 상대 8.1% 향상을 보였고, 난이도가 높은 문항이 단순히 더 긴 응답이 아니라 더 노력적·반복적·구현 중심의 episode dynamics를 유발한다는 분석도 제시했습니다. 이는 교육 측정에서 추론 모델의 인지적 과정 표현을 예측 가능하고 해석 가능하게 활용할 수 있음을 보여줍니다.



### LLawCo: Learning Laws of Cooperation for Modeling Embodied Multi-Agent Behavior (https://arxiv.org/abs/2606.28182)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: 기존의 LLM 기반 communicative embodied agent 연구는 대체로 자연어 대화와 행동을 end-to-end로 연결해 협력을 시도하지만, 파트너와 환경/작업 상태에 맞지 않는 행동을 내면서 협력이 비효율적으로 되는 문제가 반복된다. 또한 성능을 끌어올리기 위해 stronger model 증류나 비학습 파이프라인에 의존하는 경우가 많아, 상호작용을 통해 에이전트가 자율적으로 계속 개선하기 어렵다.

- **Core Contribution**: 논문은 Learning Laws of Cooperation (LLawCo)을 제안해, 에이전트가 과거 실패를 되돌아보며 “Talk when necessary”, “Wait for partner” 같은 고수준 행동 법칙을 스스로 추출·정렬하도록 한다. 이렇게 얻은 법칙을 supervised fine-tuning으로 에이전트의 reasoning 체인에 명시적으로 내재화해, 작업 목표와 다른 에이전트의 행동에 동시에 정렬되게 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 협력 실패에서 의미 있는 법칙을 뽑아내고 (2) 그 법칙에 맞는 성공 궤적만 골라 학습 데이터를 구성하며 (3) 추론 시 법칙을 실제 행동으로 안정적으로 연결하는 것이다. 저자들은 실패 에피소드에서 실패 원인을 추출해 법칙 집합을 생성하고, 성공 에피소드 중 법칙 정합성을 만족하는 샘플만 유지한 뒤, 해당 법칙에 근거한 reasoning과 행동을 SFT 데이터로 만들어 law-guided inference에서 법칙을 명시적으로 제공한다.

- **Empirical Impact**: PARTNR-Dialog(신규 대규모 멀티에이전트 communicative benchmark)와 TDW-MAT에서 실험한 결과, 4종 백본 LLM 전반에 걸쳐 협력 효율과 task success rate가 일관되게 향상됐다. 특히 PARTNR-Dialog 벤치마크에서 평균 성공률이 4.5%, TDW-MAT에서 평균 6.8% 개선되었고, 동시에 추론 시 법칙을 수정해도 업데이트된 제약을 안정적으로 따르는 controllability까지 확인했다.



### CPAgents: Agentic Composite Phenotype Generation for Cardiac Disease Association (https://arxiv.org/abs/2606.28179)
Comments:
          Accepted to MICCAI 2026

- **Prior Approaches**: 기존 cardio-PheWAS는 전문가가 미리 정한 단일 변수 심장 영상 phenotypes를 사용하거나, 임상 지식 기반 지표를 선형/단순 조합으로 구성해 질병과의 연관을 탐색해 왔습니다. 이 방식은 비선형 효과와 여러 phenotype 간 상호작용을 충분히 포착하지 못하고, 수동 설계 의존도가 높아 대규모 확장에도 한계가 있습니다. 한편 일부 자동화 접근은 feature 탐색 범위를 넓히지만, 교란 통제와 수치적 안정성 같은 신뢰성 장치가 약한 경우가 많습니다.

- **Core Contribution**: CPAgents는 심장 PheWAS에서 base 영상 features로부터 다항식/비율/상호작용 형태의 해석 가능한 composite phenotypes를 반복적으로 자동 구성·검증하는 프레임워크입니다. Analyst–Proposer–Verifier 3개 에이전트가 각각 통계 이상 징후 탐지와 변환 후보 생성, 수치 안전 규칙 하의 식 생성, 다단계 기준으로의 수용 여부 판정 및 증거 추적을 담당합니다. 결과적으로 전문가 주도 feature 선택을 넘어 더 강한 phenotype–disease 연관 신호를 확장하려는 것이 핵심입니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 상관이 강한 영상 특성들 사이에서 비선형/교차 상호작용을 무리 없이 탐색하되, (2) 다중검정과 cohort 이질성으로 인한 불안정·과적합을 줄이고, (3) 생성된 식이 실제 데이터에서 안전하게 실행되도록 보장하는 데 있습니다. CPAgents는 AST 기반 연산자 공간을 미리 제한하고(0-나눗셈 등은 ε로 가드), Analyst의 ΔAIC·분포·상호정보·그룹 효과·중복 클러스터링 요약을 Proposer 후보 생성에 반영하며, Verifier에서 marginal utility, ElasticNet 안정성, 중복 제거, LightGBM+SHAP 중요도까지 단계별로 걸러줍니다. 또한 hold-out에서 여러 classifier 간 합의 조건을 만족할 때만 채택하고 연속 거절 시 early stopping으로 계산 비용을 통제합니다.

- **Empirical Impact**: UK Biobank 대규모 CMR 코호트(26,893명)에서 9개 질병 범주를 대상으로 평가했을 때, composite phenotype 변형은 72개 classifier–disease–metric 조합 중 56개에서 top rank를 차지해 baseline(18개)보다 우수했습니다. 특히 9개 질병 범주 전반에서 성능 향상이 관찰되어 특정 과제에만 치우친 효과가 아니라는 점을 보여줍니다. 또한 ablation에서 Verifier가 성능에 가장 큰 영향을 미쳤고, silhouette score 기준으로 클래스 분리가 약 1.5배 개선되어(해석 가능한 분별력 증가) 실제 분류 이점이 동반됨을 확인했습니다.



### Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models (https://arxiv.org/abs/2606.28153)
Comments:
          323 pages, 19 figures. Accepted at ICML 2026 as a Oral presentation

- **Prior Approaches**: 기존 연구는 재잘거림 금지(Refusal) 같은 안전 메커니즘을 계층/뉴런/attention head 단위로 찾아내거나, jailbreak을 탐지·차단하는 다층 방어 스택을 구축해 왔다. 하지만 jailbreak이 “성공할 때 내부에서 정확히 무엇이 깨지는지”에 대해서는 대부분 블랙박스 관찰에 머물러 메커니즘이 충분히 규명되지 않았다.

- **Core Contribution**: 이 논문은 jailbreak이 안전 기능을 전부 제거하는 게 아니라, 특정 attention head만 선택적으로 억제한다고 보여준다. 특히 early layers에 몰린 Adversarially Compromised Heads(ACHs)는 공격에서 억제되고, mid-layers의 Safety-Aligned Heads(SAHs)는 공격이 성공해도 강한 활성(robust activation)을 유지한다.

- **Technical Challenges**: 어떤 head가 공격 성공의 “원인(인과 경로)”이고 어떤 head가 공격 뒤에도 남는 “강건한 유해 특징”의 “근원”인지 분리하는 것이 핵심 난제였다. 이를 위해 refusal direction을 OV circuits를 거쳐 head로 back-projection하고, 세 입력군(정상/유해 거절/공격 성공)에서 head 활성 분포의 overlap으로 ACH/SAH를 분류했으며, 이후 ablation과 token-level attribution으로 ACH 억제가 attack-template 토큰에 의해 유도됨을 확인했다.

- **Empirical Impact**: 인과 검증에서 ACH 단 8개 억제만으로도 거절하지 않던 입력에서 jailbreak-like 행동 유발 성능이 급상승(ASR 0%→최대 99%대)했으며, SAH 제거는 mid-layer 안전 활성 자체를 크게 약화했다. 또한 training 없이도 SAHs를 포함한 persistent internal activation( Robust Harmful Features )을 읽어 성능 경쟁력 있는 탐지기를 구성할 수 있음을 safety-eval 벤치마크에서 보이며, 정렬이 ‘완전히 파손’되기보다 ‘우회(bypass)’될 수 있음을 실증적으로 제시했다.



### Toward Robust In-Context Segmentation via Concept Guidanc (https://arxiv.org/abs/2606.28149)
Comments:
          ECCV 2026

- **Prior Approaches**: In-context segmentation (ICS)는 파라미터 업데이트 없이 소수의 reference 이미지와 마스크로 query의 타깃 영역을 분할하는 문제다. 기존 연구들은 주로 저수준 시각 매칭에 의존해 정확도 향상에 집중했지만, 같은 query에 대해 reference를 바꿨을 때 결과가 얼마나 안정적인지(robustness)는 상대적으로 간과해왔다.

- **Core Contribution**: 이 논문은 ICS를 robustness 관점에서 재정의하고, Concept-Guided In-Context Segmentation (CG-ICS)라는 새 패러다임을 제안한다. CG-ICS는 reference에서 고수준 의미 concept를 추출해 분할을 유도하며, SAM3의 frozen backbone을 활성화하는 데 텍스트 concept와 시각 exemplar를 함께 활용한다.

- **Technical Challenges**: 핵심 난제는 신뢰할 수 있는 textual concept를 reference들로부터 어떻게 안정적으로 고르느냐와, concept와 query의 위치 정합을 어떻게 보장하느냐이다. 이를 위해 MLLM이 concept 후보를 제안하고, SAM3-driven scoring 함수와 tree-search refinement로 신뢰도 높은 concept를 선택하며, 별도의 visual exemplar 경로에서는 간단한 context construction을 통해 query-side 공간 grounding을 제공한다.

- **Empirical Impact**: 표준 ICS 벤치마크에 대한 대규모 실험에서 CG-ICS는 정확도에서 state-of-the-art를 달성하는 동시에 robustness도 크게 향상시켰다. 특히 다양한 reference 선택에 대해 분할 결과의 분산(variance)을 크게 줄여, 더 신뢰할 수 있는 ICS 시스템으로 이어진다는 점에서 의미가 크다.



### Beyond Sparse Supervision: Diffusion-Guided Learning for Few-Shot Graph Fraud Detection (https://arxiv.org/abs/2606.28134)
- **Prior Approaches**: 기존 그래프 기반 사기 탐지는 공간형 GNN(메시지 패싱)이나 스펙트럴 필터링을 중심으로 발전했지만, 희소·불균형 라벨 상황에서 소수 이상 신호가 다수 클래스에 희석되는 문제가 반복된다. 공간형은 위장(fraud camouflage) 때문에 국소 이웃의 의미가 흔들리면 오버스무딩으로 신호가 무뎌지고, 스펙트럴 방식은 주로 low-frequency 평활화 편향으로 사기와 연관된 mid/high-frequency 불규칙성을 억제하기 쉽다. 이종 그래프 모델은 관계별 단서를 다루려 하지만 관계 의미가 하나로 합쳐지는 과정에서 semantic collapse가 생길 수 있다.

- **Core Contribution**: ADC-GNN(Attention-guided Diffusion-Contrastive Graph Neural Network)은 few-shot 그래프 사기 탐지를 위해 diffusion 기반 특징 증강, contrastive 표현 정렬, relation-aware multi-hop spectral attention을 한 프레임워크로 통합한다. 특히 diffusion을 전체 그래프 생성(토폴로지/엣지 생성)으로 과장하지 않고, 노드 특징 공간에서 노이즈-퍼팅(view augmentation)과 denoising·contrastive 안정화로만 사용해 거래 그래프의 관계 의미 훼손을 피한다. 그 위에 hop 및 relation 수준 단서를 선택적으로 강조하는 스펙트럴 어텐션으로 이상 표현의 분별성을 보강한다.

- **Technical Challenges**: 핵심 난제는 (1) 1% 이하 학습 비율 수준의 희소 감독에서 그라디언트가 다수(정상) 쪽으로 쏠리며 표현 붕괴가 발생하고, (2) 공간/스펙트럴 관점에서 이상에 중요한 주파수·장거리 패턴이 각각 오버스무딩/저역 통과로 사라지는 점이다. ADC-GNN은 cosine schedule로 특징을 단계적으로 노이즈화해 두 개의 독립 view를 만들고, denoising loss로 perturbation-aware 정보를 유지한 뒤 contrastive loss로 같은 노드의 표현을 정렬하면서 다른 노드는 분리하도록 학습한다. 추가로 multi-hop 스펙트럴 폴리노미얼 필터에 관계별 가중치를 얹어 hop-level·relation-level 단서를 adaptive하게 증폭하고, 이를 분기(branch) 어텐션으로 재가중해 이종 의미 붕괴를 줄인다.

- **Empirical Impact**: 실험에서는 Amazon, YelpChi, T-Finance 세 퍼블릭 벤치마크와 약 6만 레코드 규모의 텔레콤 실데이터를 함께 사용해, 1% 학습 조건에서 원래 그래프 사기 베이스라인 및 최신 그래프 anomaly/fraud 계열 방법들에 대해 일관된 성능 향상을 보인다. 또한 split 안정성, 학습 비율 변화, oversampling 대안 비교, 모듈 단위 ablation, diffusion schedule 변화, runtime·메모리 사용량까지 다뤄 효과적인 운용 구간과 비용을 보수적으로 제시한다. 결과적으로 라벨이 극도로 부족한 실무형 사기 그래프에서 diffusion-contrastive 정규화와 스펙트럴 관계 어텐션을 결합하는 접근이 재현 가능하게 유효하다는 점을 보여준다.



### PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation (https://arxiv.org/abs/2606.28128)
Comments:
          Github: this https URL Project website: this https URL

- **Prior Approaches**: 기존 비디오 생성은 시각적 품질이나 액션 조건 만족에 집중하는 경우가 많아, 접촉이 많은 조작에서 불연속 궤적·물체 침투·중력 위반 같은 물리 부정합이 남는 한계가 있었다. 로봇용 데이터로 fine-tuning한 모델도 reconstruction류 목적이 배경과 상호작용 구역을 균일하게 다뤄, 접촉 주변의 국소 물리 오류와 전역 상호작용 결과 오류를 동시에 잡기 어렵다.

- **Core Contribution**: 이 논문은 물리적 타당성을 (1) 픽셀 레벨 국소 운동, (2) 시맨틱 레벨 상호 관계의 계층적 정렬, 그리고 (3) 조작에 중요한 영역에만 감독을 거는 region-focused 정렬 문제로 재정의한다. 이를 바탕으로 PhysisForcing을 제안해, 접촉·조작기·움직이는 물체 같은 physics-informative regions에 한정해 궤적 연속성과 객체-로봇 관계성을 함께 강화한다.

- **Technical Challenges**: 핵심 기술 난점은 두 가지인데, 접촉 구간이 국소적으로만 정보가 강하다는 점(균일 감독의 희석)과, 국소 궤적 문제만으로는 전역 상호작용 결과가 보정되지 않는다는 점이다. 저자들은 참고 비디오의 point tracking과 깊이 기반 전경 가중치를 이용해 physics-informative 마스크를 만든 뒤, (i) DiT 중간층 feature에 대한 pixel-level trajectory alignment(마스크 내 포인트 궤적 MSE)와 (ii) frozen video understanding encoder로부터 얻은 토큰 간 관계를 DiT에 전이하는 semantic-level relational alignment(관계 유사도 정렬)를 joint로 최적화한다.

- **Empirical Impact**: R-Bench, PAI-Bench, EZS-Bench에서 강한 베이스라인 대비 일관된 개선이 확인됐고, 특히 R-Bench에서 Wan2.2-I2V-A14B는 22.3%, Cosmos3-Nano는 9.2% 물리 일관성 향상을 보였다(일부는 vanilla finetuning 대비 7.1~3.7%p 추가 개선). 더 나아가 WorldArena action-planner 프로토콜에서 closed-loop success rate를 16.0%→24.0%로 끌어올려 world model 계획(planning) 성능과 downstream 정책 성공에도 긍정적임을 보여, 물리 정렬된 비디오 모델이 로보틱스 표현 학습에 유리하다는 신호를 제공한다.



### From Tokens to States: LLMs as a Special Case of World Models and the Continuous Path Beyond (https://arxiv.org/abs/2606.28127)
Comments:
          10 pages, 6 figures, 1 table

- **Prior Approaches**: 기존 논의는 LLM은 토큰을 예측하고 world model은 현실을 시뮬레이션한다는 식으로 이분법을 세우는 경향이 있다. LeCun(2022)은 autoregressive 토큰 예측을 벗어나 latent-space 아키텍처인 JEPA로 가야 한다고 주장했지만, 본 논문은 이 구도가 지나치게 이분법적이라고 본다. 또한 world model은 특정 모델이 아니라 state/action/transition의 형식적 클래스라는 점에서, LLM을 단순히 ‘대체’ 대상으로 보기 어렵다고 정리한다.

- **Core Contribution**: 이 논문은 두 가지 주장을 전면에 둔다. 첫째, LLM은 world model의 특수한 퇴화(degenerate) 케이스로서 world model이 LLM을 엄밀히 포함한다는 형식적 포함관계를 제시한다(LLMs ⊂ World Models). 둘째, NTP부터 JEPA까지 이어지는 연속 스펙트럼이 존재하며 다중 토큰 예측, future-summary 예측, next-latent 예측이 그 중간 지점에 이미 자리한다고 주장한다.

- **Technical Challenges**: 스펙트럼을 따라 제약을 하나씩 푸는 과정에서 가장 큰 난점은 결국 ‘데이터 절벽’과 ‘연속 상태 예측용 아키텍처 적합성’으로 수렴한다. NTP~중간 단계까지는 여전히 internet-scale self-supervised 텍스트로 학습 가능하지만, JEPA로 가면 observation-action-next observation처럼 계측된 환경의 쌍 데이터가 필요해 표본이 급감한다. 아키텍처 측면에서도 transformer가 이산 토큰에 맞춰 공진화된 만큼, latent 벡터의 연속 예측에는 diffusion-style 헤드 등 새로운 프리미티브가 필요할 수 있다고 남긴다.

- **Empirical Impact**: 기계적 해석(mechanistic interpretability) 결과들이 LLM 내부에 세계상태(world state) 표현이 숨은 활성값에 형성됨을 뒷받침하며, 토큰 예측이 ‘표현 자체’가 아니라 인터페이스일 수 있음을 보여준다(OthelloGPT, chess LM, Llama-2 사례). 또한 다중 토큰 예측, future-summary, next-latent 계열의 최신 접근들이 스펙트럼 중간역에서 성능/계획을 개선하는 것으로 보고돼, ‘완전한 전환’보다 단계적 이동이 유망하다는 관점을 강화한다. 결론적으로 이 논문은 LLM을 버릴지의 문제가 아니라, 어떤 태스크가 어떤 단계의 world-model 제약 완화를 요구하는지를 묻는 프레임을 제안한다.



### Higher-Order Fourier Neural Operator: Explicit Mode Mixer for Nonlinear PDEs (https://arxiv.org/abs/2606.28122)
Comments:
          46 pages

- **Prior Approaches**: Fourier Neural Operator(FNO)는 푸리에 영역에서 저차원 스펙트럴 표현을 활용해 함수 공간 매핑을 학습하며, 해가 푸리에 모드별로 독립적으로 진화하는 선형·상수계수 PDE에서 특히 강점을 보인다. 기존 스펙트럴 기반 신경연산자들은 주로 대각적(모드별) 특징 변조에 머물러 비선형 PDE의 모드 간 상호작용(다항 비선형성)이 주는 귀납 편향을 충분히 반영하지 못한다.

- **Core Contribution**: 논문은 비선형 PDE의 다항 비선형성이 만드는 n-선형 모드 혼합 구조를 직접 모델링하기 위해 Higher-Order Spectral Convolution(고차 스펙트럴 컨볼루션)을 제안한다. 이를 Higher-Order FNO(HO-FNO)로 구현해 FNO의 “대각적 스케일링/조절”을 “명시적 n-linear mode mixing”으로 확장하는 것이 핵심이다.

- **Technical Challenges**: 핵심 과제는 고차 모드 혼합을 스펙트럴 연산으로 표현하면서도 FNO가 제공하는 효율성과 해상도 일반화를 유지하는 데 있다. 저자들은 n-선형 모드 혼합에 맞춘 스펙트럴 믹서를 설계해 비선형 PDE의 다항 상호작용 구조를 학습 가능한 형태로 통합하고, HO-FNO가 기존 스펙트럴 연산자처럼 고효율로 동작하도록 구성했다.

- **Empirical Impact**: 표준 벤치마크 실험에서 HO-FNO는 FNO 계열의 효율을 유지하면서 다른 스펙트럴 neural operator들보다 일관되게 성능이 좋아졌다. 특히 Poisson equation with polynomial forcing처럼 비선형성이 큰 상황에서는 단일 HO-FNO 레이어가 최대 16개 레이어의 FNO 모델을 능가하는 등 향상 폭이 컸고, 여러 데이터셋에서 transformers 및 state-space models과 비슷하거나 더 나은 결과를 보이며 비선형 모드 상호작용 학습의 실질적 의미를 입증했다.



### BiDeMem: Bidirectional Degradation Memory for Explainable Image Restoration (https://arxiv.org/abs/2606.28112)
- **Prior Approaches**: 복원 모델들은 noise, haze, rain, blur, 압축열화 등 다양한 열화를 다루기 위해 degradation-aware prompts·조건·latent priors를 점점 더 쓰지만, 평가는 대개 PSNR/SSIM 같은 엔드포인트 성능에 머뭅니다. 이 방식은 조건이 진짜 ‘의미 있는 열화 prior’인지, 아니면 추가 모델 capacity나 global correction bias, 데이터셋 shortcut에 불과한지 구분이 어렵습니다. 기존 memory/조건 연구도 경로 자체를 반사실적으로 검증(counterfactual)하는 수준까지는 상대적으로 덜 엄격했습니다.

- **Core Contribution**: BiDeMem은 복원에 쓰인 degradation memory를 ‘설명 가능한 prior’로 취급하기 위한 양방향(bidirectional) 설계를 제안합니다. 입력 통계와 복원 특징으로 top-k 메모리 슬롯을 조회하고, 같은 슬롯 identity를 추론 시 복원 조건화와 학습 시 forward-degradation(깨끗한 타깃에서 관측 열화를 재생성) 설명 경로에 함께 사용합니다. 이를 통해 “어떤 슬롯이 어떤 역할을 했는지”를 마스킹/교체/셔플 같은 개입으로 검증 가능한 형태로 만듭니다.

- **Technical Challenges**: 핵심 기술 과제는, 슬롯 기반 prior가 복원 성능을 높이면서도 단순 잔차 보정 head나 dense FiLM 같은 일반 조건화 효과로 환원되지 않도록 경로 수준에서 분리하는 것입니다. 논문은 NAFNet 기반 통제 실험에서 correction-head-only, dense query-FiLM, static/global prior 같은 강한 대조군을 두고, active slots와 inactive slots를 분리 마스킹해 슬롯 의존성을 확인합니다. 또한 same selected slot identity를 복원 경로와 열화 설명 경로에 공유해, prior가 관측 열화 증거에 정렬(aligned)되고 잘못된 prior에는 민감하게 반응하는지 동시에 학습·평가합니다.

- **Empirical Impact**: 통제된 multi-degradation(denoising/deraining/dehazing) NAFNet 실험에서 BiRank Memory는 평균 PSNR/SSIM이 8개 벤치마크에서 29.7529 dB/0.8865로 보고되며, correction-head-only(0.2588 dB), dense-prior(0.2586 dB), static/global-prior(0.2839 dB) 대비 열화 prior 효과가 더 크다고 주장합니다. 더 중요한 것은 개입 민감도인데, wrong-prior drop이 Rank Memory 0.2365 dB에서 BiRank Memory 1.0430 dB로, native/non-native gap도 0.3484 dB에서 0.6134 dB로 확대됩니다. 네트워크를 외부 백본(AirNet, PromptIR)에도 대응해 미세조정했을 때도 BiRank가 성능을 유지·향상시키는 경향을 보였지만, 계산 효율은 기본 네트워크 대비 불리하고 이득은 세팅 의존적이라 보완 연구 여지가 남습니다.



### OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Remova (https://arxiv.org/abs/2606.28094)
Comments:
          Code and resources are available at this https URL

- **Prior Approaches**: 기존 object removal은 GAN 기반이나 inpainting, diffusion 기반으로 발전했지만, 대부분이 마스크 품질에 민감하고 제거 경계나 잔여 효과(그림자·반사)를 완전히 다루지 못했다. diffusion 기반도 반복 denoising이 필요해 계산 비용이 커 상호작용/엣지 환경에 제약이 컸다.

- **Core Contribution**: 논문은 OSOR(One-Step Object Removal)을 제안해 한 번의 denoising 패스로 배경을 복원하면서 그림자·반사 같은 효과까지 함께 제거하도록 만든다. 동시에 user mask가 부정확하거나 효과 영역을 누락해도 동작하도록 mask-robust 설계를 포함한다.

- **Technical Challenges**: 단일 스텝에서는 마스크 경계 주변에서 seam/블러가 생기기 쉬운데, 이를 위해 occupancy-guided discriminator가 패치 단위 경계 감독을 분수 점유율로 제공한다. 또한 잘못된 마스크를 보정하기 위해 lightweight alpha head를 붙이고, 불완전 마스크 conditioning과 alpha compositing으로 모델이 제공된 범위 밖 효과까지 추론하게 학습한다.

- **Empirical Impact**: 데이터 측면에서는 SAVP로 노이즈 instruction 기반 triplet에서 효과 인지 supervision을 뽑아 CORNE을 28만 검증 removal pair로 구축하고, AnimeEraseBench와 TextEraseBench도 추가했다. 실험 결과 OSOR은 강한 multi-step diffusion 대비 지각 품질이 우수하면서도 추론 속도는 4×~30× 빨라져 1024×1024 이미지를 단일 A100에서 1초 내 처리한다.



### STAG: Spatio-temporal Evolving Structural Representation of Action Units for Micro-expression Recognition (https://arxiv.org/abs/2606.28083)
- **Prior Approaches**: 기존 마이크로표정인식(MER) 연구는 매우 짧고 미세한 얼굴 근육 움직임을 다루기 위해 주로 apex-onset 프레임에 의존하거나, CNN/3D CNN으로 국소적인 시간 수용영역만 확보하는 경향이 강했다. 또한 그래프 기반 방법은 얼굴 ROI 간 관계를 정적 adjacency matrix로 두거나 AU(AU-guided)를 느슨하게 결합해, 근육 활성에 따라 변하는 상호작용의 시간적 정렬과 동적 진화를 충분히 반영하지 못했다. 결과적으로 공간(어디)과 시간(언제)을 독립적으로 학습·후기 융합하는 구조가 많아 데이터셋 간 일반화와 해석가능성에서 한계를 보였다.

- **Core Contribution**: 이 논문은 STAG(Spatio-Temporal Evolving Structural Representation of Action Units for Micro-expression RecoGnition)이라는 단일 프레임워크로 ROI의 동적 연결성과 AU 기반 근육 정보를 결합해, 공간-시간을 함께 학습하도록 설계했다. STAG은 motion flow(광학흐름)와 adaptive facial connectivity(적응형 얼굴 연결)를 동시에 모델링하며, AU-guided dynamic connectivity로 근육 활성 패턴에 따라 얼굴 영역 간 상호작용이 변하도록 한다. 또한 E-GAT 기반 공간 추론과 transformer 기반 전체 시퀀스 시간 모델링을 bidirectional cross-attention으로 상호 정교화해 “어디/언제”를 통합한다.

- **Technical Challenges**: 핵심 난제는 (1) apex에 치우치지 않으면서 미세한 inter-frame 동역학을 안정적으로 포착하고, (2) 얼굴 ROI 간 관계를 정적 지오메트리 프라이어 대신 근육 활성에 맞춰 동적으로 구성하는 동시에, (3) 공간 그래프와 시간 시퀀스를 서로 다른 모듈이 따로 최적화하지 않도록 결합하는 것이다. 논문은 magnitude-based selection과 temporal attention으로 판별성 높은 optical flow를 뽑고, E-GAT로 구조화된 공간 추론을 수행한 뒤 transformer encoder로 미세 타이밍을 학습한다. 이어 bidirectional cross-attention으로 공간-시간 특징을 상호 보정하며, AU-guided 동적 그래프 생성과 temporally smoothed adjacency 업데이트로 연결의 시간적 연속성을 유지한다.

- **Empirical Impact**: STAG은 CASME II, 4DME, DFME, NaME, SAMM, SMIC-HS 등 6개 벤치마크에서 LOSO 및 K-Fold 변형 프로토콜로 평가되며, cross-dataset robustness와 일반화 성능을 향상시키는 것으로 보고된다. 특히 해석가능성 관점에서 AU-guided dynamic connectivity와 상호 정교화 구조가 의미적 일관성(semantic consistency)과 explainable micro-expression recognition에 유리하다고 제시한다. 또한 focal loss 최적화와 효율적인 설계가 결합되어 computational efficiency까지 함께 개선되는 결과를 보였다.



### OperatorSHAP: Fast and Accurate Shapley Value Estimation for Neural Operators (https://arxiv.org/abs/2606.28065)
- **Prior Approaches**: Shapley 기반 설명은 이론적으로 좋은 공리(효율성, 대칭성 등)를 만족하지만, 추론 시 계산 비용이 커서 현장 적용이 어렵다. FastSHAP 같은 아모타이즈드 설명기는 한 번의 순전파로 Shapley 값을 내놓는 대신 입력이 동일한(동질적) 특징/그리드일 때로 제한돼, PDE 물리 데이터처럼 불규칙 격자·형상에선 설명 가능성이 떨어진다.

- **Core Contribution**: 이 논문은 신경 연산자(neural operators)를 위한 OperatorSHAP을 제안하며, 격자 무관(grid-agnostic)하게 attribution을 학습·생성하는 절차를 제공한다. 또한 함수 공간에서의 설명을 Aumann–Shapley 값과 연결하는 이론 틀을 세우고, FastSHAP과 유사한 학습 손실로 아모타이즈드 explainers를 훈련할 수 있게 한다.

- **Technical Challenges**: 핵심 난제는 연산자 입력이 함수라는 점과, 불규칙 격자에서는 “플레이어 수”뿐 아니라 이웃 간 거리까지 고려해야 attribution 정의가 흔들린다는 것이다. 논문은 Sobolev 공간 기반의 잘-정의 조건과 smooth mask(분할-단위)로 operator 게임을 구성해 pNA 클래스에 속하게 만들고, 그 결과 Aumann–Shapley 값의 존재성과 근사(이산 Shapley 값의 수렴)를 보장한다.

- **Empirical Impact**: 다양한 PDE에서 샘플마다 서로 다른 격자를 사용해 실험을 수행했으며, OperatorSHAP의 설명이 해상도별 이산 Shapley 값과 높은 일관성과 상관을 보이는 것으로 보고된다. 더 나아가 격자 크기가 달라져도 재학습 없이 설명이 전이됨을 보여, 물리·과학 응용에서 안전성 검증을 위한 설명 도구로서 실사용 잠재력을 강화했다.



### Single and Multi Truth Data Fusion using Large Language Models (https://arxiv.org/abs/2606.28062)
- **Prior Approaches**: 데이터 퓨전(또는 truth discovery)은 여러 출처의 상충 값을 바탕으로 속성의 정답(단일 또는 다중)을 추정하는 데이터 통합 문제로, 기존 연구는 대부분 conflict-resolving에 초점을 맞춰 왔다. 대표적으로 Majority Voting, Source Reliability Vote, LTM, DART처럼 출처 신뢰도나 도메인별 전문성을 가정(혹은 모델링)하며 반복적·확률적 추정으로 정답을 고른다. 하지만 이런 방식은 ‘단일 정답’ 가정에 편향되거나, 문맥에 따른 의미 차이를 유연하게 반영하지 못하고 표현 다양성(표기 변형/정규화)을 별도로 다뤄야 하는 한계가 있다.

- **Core Contribution**: 이 논문은 LLM을 데이터 퓨전의 truth-discovery 구성요소로 직접 사용해, 단일-truth/다중-truth을 모두 다루는 prompt 기반 접근을 체계적으로 탐구한다. 특히 domain-independent(DI) vs domain-dependent(DD), zero-shot vs one-shot의 조합을 만들고, 다중-truth에서는 여러 값을 함께 정답으로 산출하도록 유도하는 프롬프트를 설계했다. 또한 정답 생성 제한(예: 입력에 있는 값만 사용) 같은 제약을 프롬프트에 포함해 동작을 조절하고 신뢰성까지 함께 분석한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 출처 간 의미적으로 같은 값을 다른 표기/형식으로 제시할 때 이를 자연어 의미 수준에서 정렬하고, (2) 다중-truth 설정에서 ‘여러 개가 맞을 수 있음’을 LLM 출력 규칙으로 정확히 반영하는 것이다. 저자들은 이를 위해 단일/다중-truth 전용 프롬프트 구조를 분리하고, 값은 소스에 존재하는 것만 선택하도록 하는 C1 제약, 형식이 다른 동일 값은 하나로 세도록 하는 C2 제약을 두어 출력의 제약을 걸었다. 더불어 one-shot 예시를 프롬프트 앞에 선택적으로 추가해 문맥 추론과 출력 형식의 안정성을 높이도록 했다.

- **Empirical Impact**: 실험은 Book(Movie의 감독/연도 계열), Movie, Flight 데이터셋(각각 다중-truth 2개, 단일-truth 1개)에서 Recall/Precision/F1로 비교했으며, LLM 기반 DD 프롬프트가 전통적 무지도 truth discovery(예: DART, LTM)를 전 데이터셋에서 전반적으로 앞섰다. 특히 Book과 Movie에서는 domain-dependent 프롬프트가 더 높은 F1과 균형 잡힌 recall/precision을 보였고, Flight에서는 단일-truth 구조 덕에 베이스라인도 강했지만 DD/DI 프롬프트가 근접 성능을 내며 LLM의 일관성을 확인했다. 또한 Flight ID를 obfuscation해도 성능 하락이 작아, LLM이 배경지식보다 출처 일치 패턴에 더 의존할 가능성을 시사했으며, 비용은 API 호출당 수 초 수준으로 보고했다.



### ToolPrivacyBench: Benchmarking Purpose-Bound Privacy in Tool-Using LLM Agents (https://arxiv.org/abs/2606.28061)
Comments:
          24 pages, 7 figures, 15 tables

- **Prior Approaches**: 기존 에이전트 벤치마크는 멀티툴 실행에서의 작업 완료와 API 호출 정확성을 중심으로 평가하는 경향이 강합니다. 반면 프라이버시 평가는 최종 응답의 판단이나 훈련 데이터 추출/기억 같은 단일 결과에 초점을 맞추며, 툴 호출 도중 어떤 정보가 어떤 목적에 의해 흐르는지(중간 단계의 과다 공개)를 체계적으로 다루지 못했습니다. 특히 같은 프라이빗 정보라도 다음 툴 단계에서는 필요하지 않을 수 있는데, 기존 평가는 이런 need-to-know 경계 위반을 정밀 진단하기 어렵습니다.

- **Core Contribution**: ToolPrivacyBench는 멀티툴 궤적(trajectory)에서 목적에 바운드된 정보 흐름을 감사(audit)하는 벤치마크를 제안합니다. 핵심 질문은 “에이전트가 작업을 끝내는 것과 별개로, 현재 단계에서 필요한 프라이빗 사실만 권한 있는 툴/싱크로 전달했는가”입니다. 이를 위해 케이스별 정책 지식베이스(policy knowledge base)를 두고, 비공개 원소(private atoms)·툴 목적·싱크 타입·허용/금지 관계를 기반으로 과다 공개를 판정합니다.

- **Technical Challenges**: 목적 경계는 ‘필드가 민감한가’만으로 결정되지 않고, 같은 정보라도 툴 목적과 싱크에 따라 허용 여부가 달라진다는 점이 기술적 난제입니다. ToolPrivacyBench는 이를 정책 지식베이스로 정형화해 각 private atom이 어떤 tool purpose와 sink에서 허용되는지 관계형으로 모델링하고, 실행 후 도구 호출 인자와 백엔드 감사 로그를 궤적 단위로 대조합니다. 또한 free-text(티켓 설명, 노트, 요약, 핸드오프 등) 같은 비정형 채널까지 포함해 누출이 구조화 필드뿐 아니라 자연어 영역에서도 발생하는지를 함께 측정합니다.

- **Empirical Impact**: 2,150개 케이스(합성 1,150 + 공개 벤치마크 기반 1,000)로 9개 대표 에이전트를 평가한 결과, 대부분 모델은 작업 성공(TaskSuccess)이 높아도 프라이버시 관점의 과다 공개(MT-POI)는 크게 남는 것으로 나타났습니다. 즉 “툴 실행 성공=목적 경계 준수”가 성립하지 않으며, 티켓과 핸드오프가 반복적인 누출 지점으로 분석됩니다. ToolPrivacyBench는 중간 툴 인자까지 포함한 궤적 수준 감사가 목적 바운드 privacy over-disclosure를 드러내는 데 필수임을 보여주며, 향후 안전한 툴 사용 평가 및 진단 기준을 확장하는 데 의미가 있습니다.



### MultiHashFormer: Hash-based Generative Language Models (https://arxiv.org/abs/2606.28057)
Comments:
          Under review

- **Prior Approaches**: 기존 연구는 임베딩 행렬이 어휘 크기에 선형으로 커지는 문제를 줄이기 위해, 여러 토큰을 하나의 벡터에 해시로 압축하는 방식(해시 임베딩)을 제안해 왔다. 다만 이런 many-to-one 충돌은 다음 토큰을 생성해야 하는 causal LM에서 그대로 치명적으로 작동해 적용이 어려웠다.

- **Core Contribution**: 이 논문은 hash-based autoregression을 가능하게 하는 MultiHashFormer 프레임워크를 제안한다. 각 토큰을 다중 독립 해시함수로 만든 ‘해시 시그니처(짧은 이산 해시 ID 시퀀스)’로 표현하고, Hash Encoder가 이를 하나의 잠재 벡터로 압축한 뒤 Transformer decoder로 처리하며, Hash Decoder가 다음 토큰의 해시 시그니처를 다시 생성한다.

- **Technical Challenges**: 핵심 난관은 해시 충돌을 causal 생성 과정에서도 허용 가능한 수준으로 통제하면서, 시그니처를 효율적으로 인코딩/디코딩하는 것이다. 저자들은 해시 시그니처를 여러 해시 함수의 조합으로 구성해 토큰별 식별성을 확보하고, 인코더-디코더 사이에서 시그니처↔잠재벡터↔다음 시그니처로 정보를 왕복시키는 구조를 설계해 해결했다.

- **Empirical Impact**: 100M, 1B, 3B 파라미터 규모에서 MultiHashFormer가 표준 Transformer LM을 여러 벤치마크에서 일관되게 능가함을 보였다. 또한 언어별 어휘를 확장해도 파라미터 풋프린트가 일정하게 유지되며 별도 수정 없이 멀티링구얼 확장을 처리할 수 있음을 실험으로 확인해, 효율적 대규모 생성 모델 설계에 의미 있는 진전을 제공한다.



### Can LLMs Judge Better Than They Generate? Evaluating Task Asymmetry, Mechanistic Interpretability and Transferability for In-Context QA (https://arxiv.org/abs/2606.28050)
Comments:
          18 pages

- **Prior Approaches**: LLM-as-a-Judge와 self-evaluation 파이프라인은 생성보다 평가가 더 쉽다는 전제를 바탕으로, 생성된 답을 같은 모델이 더 정확히 판정할 수 있을 것으로 기대한다. 기존 연구들은 generation과 judgment를 비교했지만 open-domain에서 parametric knowledge가 섞이면서 “평가가 쉬운지”를 통제해 검증하기 어렵다는 한계가 있었다. 또한 평가가 실제로 답을 재검증하는지, 아니면 형태적 힌트에 의존하는지에 대한 기계적(메커니즘) 설명은 충분히 다뤄지지 않았다.

- **Core Contribution**: 이 논문은 in-context QA 설정에서 context를 유일한 정보원으로 고정하고, 모델이 자신이 생성한 답을 그대로 판정하도록 하여 GA–EA(생성 정확도 vs 자기평가 정확도) 격차를 통제된 방식으로 측정한다. SQuAD 2.0, DROP, HotpotQA, MuSiQue의 네 벤치마크에서 대체로 “평가가 생성보다 쉽지 않다”는 결과를 제시하며, 특히 MuSiQue만 예외적으로 평가 우위가 나타난다고 보고한다. 더 나아가 attention 기반 분석과 LoRA 미세조정 전이 실험으로 이 비대칭이 단순한 학습 산물이 아님을 확인한다.

- **Technical Challenges**: 핵심 기술적 난제는 GA와 EA를 일관된 기준으로 비교하면서(oracle), 평가가 실제로 context를 읽어 재검증하는지 추적하는 것이다. 연구진은 같은 인스턴스에서 두 태스크(생성, 그 생성 답의 “Correct/Incorrect” 판정)를 연속 실행하고, GPT-4o를 oracle로 써 외부 기준을 고정했으며, last-token attention을 span 단위로 계량하는 메커니즘 분석을 수행한다. 또한 generation/Evaluation의 모수 구조가 공유되는지 보려 LoRA-Gen/LoRA-Eval/LoRA-Both를 학습하고, evaluation 학습에는 문맥 없는 환각 기반 hard negative를 구성해 “그럴듯한 틀린 답을 판별”하도록 설계한다.

- **Empirical Impact**: 실험 결과는 세 벤치마크(SQuAD 2.0, DROP, HotpotQA)에서 generation 정확도가 self-evaluation보다 높고(즉 Δ=EA−GA가 음수), MuSiQue에서만 평가 우위(Δ가 양수)가 관찰되는 식으로 나타난다. attention 분석에서는 평가 태스크가 context에 3–5배 적게 주의를 주고 candidate answer 슬롯도 거의 읽지 않는 경향이 확인돼, 많은 경우 평가가 “재검증”이 아니라 구조적 단서에 의존함을 시사한다. LoRA 결과는 LoRA-Eval이 오히려 generation 성능을 떨어뜨리고(과도한 보수성), LoRA-Gen은 evaluator가 over-acceptance(과수용)로 기울어지는 등 방향성이 뚜렷해, self-evaluation 파이프라인의 핵심 가정(평가가 더 쉽다)을 흔드는 실증 증거로 평가된다.



### DG^VoiC: Speaker Clustering for Fraud Investigation under Real Call-Centre Conditions (https://arxiv.org/abs/2606.28048)
Comments:
          5 pages, 4 figures, 1 table

- **Prior Approaches**: 기존 보험 사기 탐지는 주로 구조화 데이터, 텍스트, 이미지 같은 단일·멀티모달 단서에 의존하며, 콜 간 반복되는 화자 정체성은 상대적으로 덜 활용돼 왔다. 전화 사기 연구도 대체로 대화 내용(전사/의미)에 초점이 맞춰져 있고, 실제 콜센터 오디오 데이터는 개인정보·생체정보 제약으로 연구가 제한적이었다. 콜센터 맥락의 음성 기반 연구는 인증·다이어라이제이션 중심이어서, 고객 프로필을 가로지르는 화자 클러스터링을 통한 교차 연결 문제와는 평가 목표가 달랐다.

- **Core Contribution**: 이 논문은 익명화된 실제 보험 콜센터 오디오에서 ‘고객 검증’과 ‘고객 프로필 간 반복 화자 연결’을 동시에 지원하는 DG^VoiC를 제안한다. 단독 사기 판정기가 아니라, 분석가가 화자 일관성을 확인하고 반복 목소리를 찾아내도록 음성 기반 링크 신호를 노출하는 데 목적을 둔다. 민감 정보 정렬 기반 익명화와 음성 중심 전처리, 슬라이딩 윈도 임베딩 추출, 코사인 유사도 기반 클러스터링을 end-to-end 파이프라인으로 결합했다.

- **Technical Challenges**: 핵심 기술 난제는 콜센터에서 길고 길이가 다양한 통화, 잡음·무음 구간, 그리고 익명화 처리로 인해 화자 임베딩이 흔들릴 수 있다는 점이다. DG^VoiC는 NER·Regex 및 WhisperX 타임스탬프를 활용해 PII 구간을 직접 마스킹하고, Resemblyzer 전처리로 무음/저정보 구간을 제거한 뒤 ECAPA-TDNN으로 임베딩을 구한다. 또한 오버랩 슬라이딩 윈도와 최소 구간 배제를 통해 짧은 유효 발화를 놓치지 않게 하면서, 평균 풀링 및 코사인 유사도 임계값(최적 0.718)으로 안정적인 클러스터를 형성한다.

- **Empirical Impact**: 실제 121개 콜 중 전문가가 합의한 56개(22개 화자 클러스터) 기준으로 평가했으며, 최적 설정에서 AMI 96%, ARI 95%, completeness 98%, homogeneity 100%, V-measure 99%를 달성했다. 또한 보조적으로 EER 3.85%(FAR 0.50%, FRR 9.62%)를 제시해 임계값에서의 검증 관점 성능도 확인했다. 결과적으로 반복 화자 클러스터링이 분석가 검토용 교차 프로필 음성 링크를 효과적으로 부각할 수 있음을 보여, 음성 기반 사기 조사 워크플로의 추가 신호로 활용될 가능성을 제시한다.



### Mind the Gap: Quantifying the Domain Gap in Cross-Sensor Diffusion Super-Resolution (https://arxiv.org/abs/2606.28039)
Comments:
          26th International Conference on Computational Science

- **Prior Approaches**: 위성 초해상도(SR)는 센서마다 ‘진짜’ 저해상도-고해상도 쌍이 없어, 보통 bicubic 같은 방식으로 인트라센서 저하(synthetic degradation)를 만든 뒤 학습한다. diffusion 기반 SR은 질감 복원과 전역 일관성에서 강점을 보이지만, 이런 합성 저하 중심 학습은 센서의 물리·스펙트럼·복사 특성을 충분히 반영하지 못한다. 그 결과 cross-sensor 환경에서 성능이 흔들리거나 붕괴하는 문제는 경험적으로 알려져 있으나, 현대 diffusion SR 전반에 대한 체계적 정량 분석은 부족했다.

- **Core Contribution**: 이 논문은 synthetic-to-real mismatch가 modern diffusion-based SR 성능에 미치는 영향을 처음으로 체계적으로 연구한다. Sentinel-2와 PlanetScope를 기하·시간 정렬한 대규모 paired dataset을 구축해, 합성 학습→실제 교차센서 평가의 일반화 격차를 통제된 조건에서 측정한다. 또한 Sentinel-2 self-supervised 특징 기반의 도메인 적응 지각 거리 LPIPS-Sat(LPIPSSat)를 제안해 위성 영상에 더 맞는 평가 프레임을 제공한다.

- **Technical Challenges**: 핵심 기술 난제는 센서 간 분광 대역 불일치와 복사·기하 변동 때문에 ‘유효한 SR’ 학습 목표를 구성하기 어렵다는 점이다. 논문은 서로 겹치지 않는 대역을 그대로 쓰지 않고 물리적으로 대응되는 6개 밴드를 선택해, 모델이 존재하지 않는 스펙트럼 매핑(예: Blue를 Red로)까지 ‘환각’하지 않도록 설계한다. 실험적으로는 세 가지 구성(합성 기준, 합성→실제 도메인 격차 측정, 실제→실제 직접 매핑)을 나눠 diffusion 계열 여러 아키텍처의 실패 양상을 일관되게 비교한다.

- **Empirical Impact**: 결과는 두 가지로 요약된다: 합성 저하로 학습한 모델은 실제 교차센서 쌍에서 급격히 성능이 무너진다(C2가 기준선보다 악화, LPIPS/LPIPSSat도 악화). 반대로 실제 cross-sensor 데이터로 학습한 모델은 지각 지표가 일부 회복되지만 최적화가 불안정하고 센서 다양성 적응이 완전히 되지 않아 합성 설정 대비 격차가 지속된다(C1 vs C3). 또한 burned-area change delineation 같은 다운스트림 세그멘테이션에서, 합성 학습 모델은 ‘일관된(작지만 유의미한)’ 이득을 주는 반면 실제 학습 모델은 SR과 도메인 적응이 얽히며 파괴적 아티팩트를 유발해 실용성이 떨어질 수 있음을 보여준다.



### MLVC: Multi-platform Learned Video Codec for Real-World Deploymen (https://arxiv.org/abs/2606.28027)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 신경 비디오 코덱은 학습 기반 압축으로 H.265(HEVC) 대비 60~70% 비트 감소를 보여왔지만, 실제 서비스에는 여전히 쓰이지 못했습니다. 핵심 장애는 (1) GPU 중심 평가 대비 소비자 기기 NPUs에서의 실시간 성능 부족, (2) 서로 다른 하드웨어에서 인코더/디코더가 같은 엔트로피 분포를 보장하지 못해 디코딩이 무너지는 cross-platform 비호환 문제입니다.
기존 정량화 기반 접근은 bit-exact 산술을 노려왔지만, NPUs의 연산 경로·반올림·커널 차이로 “확률/스케일 파라미터” 불일치가 생기며 재앙적 실패가 발생할 수 있고, 보정 정보 제공은 확률적 완화 수준에 그쳤습니다. 일부 fixed-prior·코드북 인덱스 전송 방식은 실패를 줄이지만, 경쟁력 있는 압축 성능 검증이나 공정한 GOP 조건에서의 비교가 제한적이었습니다.

- **Core Contribution**: 이 논문은 MLVC(Machine Learning Video Codec)라는 하드웨어 강건(neural codec hardware-robust) 설계를 제안해 실제 배포를 목표로 합니다. 엔트로피 모델의 스케일(scale) 파라미터를 hyperprior 안에 “scale index” 형태로 명시적으로 전달해, bit-exact 산술 없이도 엔트로피 코딩 일관성을 보장하는 것이 핵심 아이디어입니다.
대신 스케일 전송으로 인한 오버헤드는 구조적 파라미터 공유로 줄이고, gated memory·ReGLU 계열(하드웨어 친화)·LTR(long-term reference recovery)·I-frame dropout·지각(ROI/LPIPS) 학습까지 묶어 BD-rate 효율을 회복합니다.

- **Technical Challenges**: 가장 큰 기술 난제는 스케일 파라미터 σ가 인코더/디코더의 엔트로피 분포를 결정하는데, 부동소수점·비표준 연산·NPU별 활성함수 근사 차이가 누적되면 디코더가 다른 lookup index를 선택하며 재앙적 디코딩 실패로 이어진다는 점입니다. MLVC는 (1) z-hyperlatent의 고정 엔트로피 모델을 공유해 스케일 인덱스를 결정적으로 재구성하고, (2) FP16 환경에서도 divergence가 장기 예측 체인에서 증폭되는 문제를 LTR 프레임과 주기적 I-frame 동기화로 제어합니다.
또한 벤더별 piecewise 근사가 달라지는 복잡 활성함수를 피하고 ReGLU 같은 gating을 하드웨어 호환 연산으로 구성해 cross-platform 수렴 안정성을 확보했습니다.

- **Empirical Impact**: 실험에서 MLVC는 VCD(Video Conferencing Dataset) 벤치마크(360p~720p, 지각 평가 포함)에서 HEVC-QSV 대비 MOS 기준 >70% BD-rate 개선을 달성하며, 실시간(평균 100+ FPS 수준) 인코딩·디코딩도 NPUs 기반 애플/인텔/퀄컴 기기에서 확인됐습니다. 특히 cross-platform 설정에서 대부분의 하드웨어 조합이 재앙적 실패 없이 동작하며, BD-rate가 소폭 차이로 유지되어 “배포 가능성”을 실증합니다.
또한 perceptual loss와 ROI 가중치를 적용한 fine-tuning으로, PSNR 중심으로 학습된 DCVC-RT도 지각 품질 격차를 줄였지만 MLVC는 여전히 cross-platform 제약 비용 대비 더 강한 성능을 보였습니다.



### Dialogue to Detection: A Multimodal Hybrid NLP Pipeline for Insurance Fraud Detection (https://arxiv.org/abs/2606.28002)
Comments:
          10 pages, 8 figures, 2 tables

- **Prior Approaches**: 기존 보험 사기 탐지는 주로 text-only 공개 데이터에 의존해 BERT류 분류기나 규칙 기반 점검을 붙이는 방식이 많았다. 하지만 FNOL(최초 청구) 단계의 음성 통화는 프라이버시·접근성 문제로 쌍(pair) 데이터가 거의 공개되지 않아, 음성의 부가 정보까지 함께 검증하는 멀티모달 연구가 제한돼 왔다.

- **Core Contribution**: 이 논문은 FNOL 상황을 모사하는 합성 multimodal 파이프라인을 제안한다. GPT-2로 에이전트-고객 대화 대본을 만들고, xTTS로 2인 화자 오디오를 합성한 뒤, WhisperX의 ASR·diarisation 결과와 텍스트의 NER·RAG 검색, 그리고 Resemblyzer speaker embedding을 결합해 해석 가능한 fraud risk score를 산출한다.

- **Technical Challenges**: 핵심 난제는 실제 통화 수준의 대화 구조·음향 특성과, 화자 분리·전사 오류 같은 현실 조건을 합성 데이터에 재현하는 것이다. 논문은 다양한 디코딩(temperature·nucleus/top-k)과 2채널 오디오 합성→WhisperX 기반 분리→고정밀 식별자 추출(Regex+NER)→텍스트 유사도(RAG)·음성 재사용(embedding 유사도) 결합, 그리고 가중치 기반 룰 점수로 false positive를 완화하는 설계를 택했다.

- **Empirical Impact**: 검증 결과 합성 데이터 내부 일관성과 구성 요소별 처리 성능(전사 WER, 화자 분리/특징 추출 정밀도·재현율·F1 등)을 단계적으로 확인했으며, 텍스트 분류(BERT-RAG)는 합성 홀드아웃에서 사실상 100% 수준의 성능을 보였다. 다만 변동성이 제한된 합성 환경이라 낙관적일 수 있어, 실제 데이터로의 일반화 강화를 위해 내러티브·스피커 다양성 확대와 k-fold 및 익명 실데이터 파인튜닝이 후속 과제로 제시됐다.



### Parallel Rollout Approximation for Pixel-Space Autoregressive Image Generation (https://arxiv.org/abs/2606.27978)
- **Prior Approaches**: 픽셀 공간에서 autoregressive(AR) 생성은 패치를 연속 토큰으로 두고 다음 패치를 예측하지만, 고차원 픽셀 패치 예측의 단일 스텝 오차와 teacher-forcing으로 인한 train–inference gap이 함께 커져 오류가 누적되는 문제가 컸다. 입력 측 완화로는 input noise injection, 출력 측 완화로는 xx-prediction 같은 파라미터화가 쓰였지만, diffusion 기반(pixel-space diffusion) 성능 격차를 충분히 줄이지 못했다. 한편 정확한 rollout 학습은 더 잘 맞추지만, 연속 토큰 AR에선 순차 샘플링 비용이 너무 커 실용성이 떨어졌다.

- **Core Contribution**: 이 논문은 Parallel Rollout Approximation(PRA)로 두 병목을 동시에 겨냥한다. PRA는 AR이 고차원 픽셀 패치를 직접 생성하지 않고, 저차원 중간 상태를 생성한 뒤 pixel decoder로 다시 픽셀 패치(=pixel-in, pixel-out 인터페이스)를 만들어 넣는다. 또한 추론 때처럼 생성된 입력이 들어오는 상황을 학습에서도 근사하기 위해, 중간 상태→픽셀 디코딩 경로를 그대로 사용해 inference-like 픽셀 입력을 포지션별로 병렬 구성한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 고차원 픽셀 토큰 생성 난이도를 낮추면서도, (2) 학습 시 조건이 추론 시 조건과 최대한 비슷해지도록 만드는 것이다. 저차원 중간 상태는 엔드투엔드로 학습하되, causal prefix 표현을 반영한 중간 목표를 만들고 토큰 마스킹으로 현재 토큰만 의존하지 않게 제어한다. 입력 측 근사는 실제 순차 rollout 대신, 각 포지션에서 perturbed intermediate state를 디코딩해 stop-gradient로 학습 입력을 만들고, 이때 필요한 AR forward는 병렬로 처리해 계산비용을 억제한다.

- **Empirical Impact**: ImageNet-1K 클래스 조건 생성(256×256)에서 PRA-S(135M)는 FID 2.58로, 이전 픽셀 공간 AR 최강(1B급)의 FID 3.60을 앞질렀다. 더 큰 PRA-L(511M)은 FID 1.94까지 개선해 픽셀 공간 AR 모델 중 새로운 state of the art를 기록했다. 추가로 생성뿐 아니라 ImageNet 분류 probing 정확도도 기존 AR·diffusion 베이스라인보다 높게 나타나, 픽셀 공간 엔드투엔드 AR이 생성+시각 이해를 함께 끌어올릴 가능성을 시사한다.



### SHARD: cell-keyed residual splitting for alignment-resistant private dense retrieva (https://arxiv.org/abs/2606.27976)
Comments:
          arXiv admin note: text overlap with arXiv:2606.26373

- **Prior Approaches**: 연구진은 dense embeddings을 기반으로 한 semantic search와 RAG에서, 벡터 저장소가 유출될 때 텍스트 복원이 가능해지는 문제를 다룹니다. 기존 방어로는 SVD로 차원을 줄인 뒤 단일 secret rotation과 CKKS reranking을 조합하는 “글로벌-선형” 방식이 널리 쓰이지만, 이때 보호되는 기하가 단 하나의 전역 정렬(geometry)이어서 알려진-평문(known-plaintext) 정렬 공격에 취약하다고 지적합니다. 특히 orthogonal Procrustes가 대략 subspace 차원 수준의 anchor로 회전을 복구하고, 이후 일부 공개 인덱스·참조 코퍼스가 결합되면 높은 정확도로 원문을 재구성할 수 있다고 보고합니다.

- **Core Contribution**: 논문은 retrieval 품질을 해치지 않으면서도 “정렬 가능한 약한 축”을 제거하는 공격-인식형 방어 변환 Shard를 제안합니다. Shard는 centered embedding을 짧은 public prefix(1단계 검색용)와 private residual로 분해하고, residual은 C개의 cell로 샤딩한 뒤 cell별 비밀 키로 회전·분산해 서버가 단일 공통 기하를 보지 못하게 만듭니다. CKKS reranking은 이 키가 상쇄되는 방식으로 수행되어 내적이 정확히 복원되며, 결과적으로 half-SVD truncation이 주던 검색 품질 손실을 되돌립니다.

- **Technical Challenges**: 핵심 난제는 두 채널을 동시에 만족시키는 것입니다: (1) 서버가 보게 되는 공개 인덱스·접근 단서가 정렬 표면(alignment surface)이 되지 않게 만들고, (2) 암호화 연산에서는 내적/랭킹이 정확히 유지돼야 합니다. 논문은 residual을 cell-local 마이크로 키로 처리해 전역 정렬 대신 “셀마다 분리된 프레임”을 강제하고, CKKS에서 cell 키가 상쇄되도록 설계해 full-dimensional reranking 정확도를 확보합니다. 또한 파라미터 C 하나로 baseline(C=1)의 전역형에서 per-document micro-keys(C=N)까지 연속적으로 확장되도록 하여, 정렬 저항성과 비용(암호화된 residual 쿼리 수)의 트레이드오프를 조절합니다.

- **Empirical Impact**: 실험에서 Shard는 BEIR 계열 작업에서 half-SVD truncation 기반 baseline 대비 nDCG@10 저하를 줄이고, full-dimensional reranking 덕분에 원(raw-space) 랭킹에 더 가깝게 복원됨을 보여줍니다. 정렬 공격 측면에서는 diffuse한 알려진-평문 누출이 있을 때 private residual을 원래 프레임으로 되돌리기 위한 anchor 복잡도가 대략 C배씩 증가하며(예: C=1 대비 C=256에서 큰 폭 상승), 공격이 더 강하거나 비선형/학습형 정렬러(ALGEN, MLP)·unsupervised vec2vec류를 써도 이 완화가 유지된다고 보고합니다. 다만 한계도 명확히 제시하는데, 셀 내부에서는 키 상쇄가 일어나므로 표적 공격(targeted)과 공개 prefix의 거친 구조 누출, 그리고 겹치는 레퍼런스 코퍼스가 결합될 때는 추가 변형(셀 ID·micro-key 제한 등)이 필요하다고 결론냅니다.



### ProMSA:Progressive Multimodal Search Agents for Knowledge-Based Visual Question Answering (https://arxiv.org/abs/2606.27974)
- **Prior Approaches**: 기존 KB-VQA는 대체로 고정된 retrieval-then-generate 파이프라인을 따른다. 이미지(또는 텍스트)에서 top-k를 한 번 뽑아 프롬프트에 넣고 정답을 생성하며, 추론 중에는 검색 정책이나 깊이를 바꾸기 어렵다. 또한 초기 검색이 잘못되면 이후 라운드가 그 오답 근거에 더해져 실패를 교정하기 힘들고, 멀티홉 증거 사슬도 정적 주입 방식에서 약해진다.

- **Core Contribution**: ProMSA는 KB-VQA를 “예산을 가진 점진적 검색-추론” 문제로 재구성해, 추론 도중에 img_search/text_search/stop을 반복 선택하는 progressive multimodal search agent를 제안한다. 에이전트는 중복 제거(deduplication)와 함께, 엔티티 식별이 불확실할 땐 재검색(이미지), 속성/증거가 비면 쿼리 재작성 후 텍스트 검색을 수행하며, 근거가 충분하면 stop해 답을 생성한다. 이를 통해 retrieval과 reasoning을 단일 궤적에서 end-to-end로 결합한다.

- **Technical Challenges**: 핵심 난제는 (1) 툴 호출 형식과 인자 구조를 먼저 “실행 가능”하게 학습해야 RL 탐색 불안정(잘못된 호출로 보상 0)이 생기지 않는 점, (2) 라운드 수(툴 상호작용 깊이)와 생성 길이가 다른 RL 업데이트 스케일을 유발해 학습이 흔들릴 수 있다는 점이다. 저자는 rejection-sampling SFT로 콜 형식을 워밍업한 뒤, sequence-level RL에서 generation length뿐 아니라 tool-interaction depth까지 반영하는 TN-GSPO를 도입해 업데이트 편향을 줄이고 안정적인 검색 정책을 학습한다.

- **Empirical Impact**: E-VQA와 InfoSeek에서 ProMSA는 강한 RAG 및 에이전트 베이스라인을 일관되게 능가하며 retrieval 정확도와 end-to-end 정확도를 함께 끌어올린다. zero-shot MLLM 대비 긴 꼬리 엔티티/미세 속성에 유리한 외부 근거 활용 능력을 보여주고, 검색 에이전트 중에서도 실패 교정 메커니즘이 약한 기존 접근 대비 성능 우위를 확인한다. 또한 OK-VQA에서도 개선이 관찰돼, 학습된 도구 사용 정책이 훈련 벤치마크를 넘어 일반화될 가능성을 시사한다.



### From Black-Box to Clinical Insight: A Multi-Stage Explainable Framework for Speech-Based Cognitive Impairment Detection (https://arxiv.org/abs/2606.27973)
Comments:
          Accepted to Interspeech 2026

- **Prior Approaches**: 기존 음성 기반 인지장애 탐지는 transformer 성능을 끌어올렸지만, 모델이 왜 그런 결론을 내리는지 설명이 ‘블랙박스’로 남아 임상 도입이 어렵다는 지적이 컸다. XAI 연구는 SHAP/LIME 같은 토큰 중요도나 일부 수작업 언어 지표에 집중했지만, 임상의가 실제로 쓰는 인지-언어 메커니즘으로의 연결이 약하고 수치 해석이 필요해 사용성이 떨어졌다.

- **Core Contribution**: 이 논문은 transformer 예측을 임상적으로 근거 있는 ‘서술형 설명’으로 바꾸는 multi-stage explainability 프레임워크를 제안한다. SHAP 토큰 기여(서브워드→단어 수준 집계)와 임상 지향 언어 특징(어휘 풍부도, 구문 복잡도, 의미 응집성 등)을 결합한 뒤, LLaMA-3.1-70B-Instruct로 4단계 LLM 추론 파이프라인을 돌려 통합된 임상용 내러티브를 생성한다.

- **Technical Challenges**: 핵심 난제는 (1) 서브워드 토큰화를 가진 transformer에서 perturbation 기반 SHAP을 임상 언어 단위로 해석 가능하게 만드는 것과 (2) 토큰/수치 결과를 ‘왜 그런 인지 문제가 보이는지’로 번역하는 일이다. 논문은 모델 래퍼로 SHAP 입력·확률 출력을 구성하고 계층적 집계로 word-level 설명을 만들었으며, LLM에 인지-언어 차원별 임상 정의를 프롬프트로 제공해 추론이 의미를 벗어나지 않게 제한했다.

- **Empirical Impact**: 성능 측면에서 SpeechCARE Adaptive Gating Fusion(SpeechCARE-AGF) 기반 스크리닝 모델은 NIA PREPARE에서 AUC 86.83%, F1 72.11%를 보였다. 임상 유효성 평가에서는 70개 샘플에 대해 2명의 1차 진료의가 블라인드로 검토해 98% 케이스에서 높은 일치(κ=0.85)를 보였고, 사용성 척도 SUS는 82/100으로 임상 워크플로 통합 가능성을 시사했다.



### Reasoning Beyond Prediction: From Data-Driven to Causal Software Engineering (https://arxiv.org/abs/2606.27960)
Comments:
          Accepted for publication in Communications of the ACM

- **Prior Approaches**: 기존 AI4SE 흐름은 DDSE(Data-driven Software Engineering)처럼 상관관계 학습을 통해 코드 작성, 테스트 생성, 운영 이상탐지, 디버깅을 돕는 데 집중해 왔습니다. LLM-assisted development 역시 패턴 기반 생성과 자동화를 강화했지만, 보고되는 사고가 꾸준해 ‘믿을 수 있는 소프트웨어’로 직결되긴 어렵다는 한계가 드러났습니다. 특히 ML/LLM은 상관관계는 잘 포착해도 원인(cause)을 구조적으로 드러내지 못해 신뢰(설명가능성·책임성)가 취약해집니다.

- **Core Contribution**: 논문은 소프트웨어 엔지니어의 추론을 ‘인과( causation )의 렌즈’로 증폭하는 새로운 패러다임으로 Causal Software Engineering(CSE)을 제안합니다. CSE는 기계가 단순 예측이나 자동화를 넘어, 개입(intervention)과 반사실(counterfactual)을 다루는 what-if/왜(why) 질의를 수행하도록 하여 “상관은 알지만 이유는 모르는” 갭을 메우려 합니다. DDSE를 다음 단계로 끌어올려, 원자적 패턴 학습을 넘어 plan-to-improve 성격의 지원을 목표로 합니다.

- **Technical Challenges**: 핵심 기술 과제는 인과 모델을 구성하고, 이를 통해 rigor(정량 인과효과)와 transparency(검사 가능한 인과 경로·가정)로 추론을 구동하는 것입니다. 논문은 인과 관계를 담는 inspectable causal models을 중심에 두고, (1) 전문가 명세와 (2) 관측/비관측(실험) 데이터 기반 구조 학습 및 파라미터 피팅, (3) 반박(refutation)·검증을 포함한 추론 단계를 제시합니다. 또한 LLM을 인과 변수/관계 추출이나 일관성 점검(컨텍스트 삽입, consistency checker) 등에 결합해 실무 난이도를 낮추는 방향을 제안합니다.

- **Empirical Impact**: 문헌 리뷰에 따르면 CSE 적용은 디지털 라이브러리 기준 90편 이상으로 빠르게 확장되었으며, 다만 대부분이 프레임워크의 일부를 부분적으로 구현하는 수준입니다. 적용 영역은 fault localization, 테스트(입력 선택이 곧 가설/가정의 성격을 갖는다는 점에서 자연스럽다), 예측(교란 없는 추정) 등으로 넓어지고 있습니다. 논문은 현재 산업 대체보다는 DDSE 위에 causal augmentation layer로 얹는 점진적 채택이 현실적이며, 데이터 품질·모델 안정성·도구/워크플로 통합·전문화된 튜토리얼과 반박 테스트가 도입 관건이라고 정리합니다.



### VASAE: Naming SAE Dictionary Directions with Vocabulary-Aligned Anchoring (https://arxiv.org/abs/2606.27941)
Comments:
          14 pages, 7 figures. Accepted to the 2nd Workshop on Compositional Learning at ICML 2026

- **Prior Approaches**: 기존 Sparse Autoencoder(SAE)는 Transformer 잔차 스트림을 희소 코드로 분해하지만, 학습된 사전(dictionary) 특징의 토큰 이름은 대부분 훈련 후 문맥을 훑거나 별도 해석 절차로 ‘사후적으로’ 붙인다. 이 방식은 특징이 토큰 임베딩 공간과 어떤 기하학적 관계를 갖는지 학습 과정에서 직접 연결되지 않아, 특징-토큰 대응이 불안정하거나 약해질 수 있다.

- **Core Contribution**: 본 논문은 Vocabulary-Aligned Sparse Autoencoder(VASAE)로, SAE 특징을 토큰 임베딩 방향에 ‘훈련 중’ 소프트로 정렬(anchoring)시키고 각 특징에 고유한 intrinsic token name을 부여한다. 특징의 이름은 해당 특징 벡터와 가장 가까운 토큰 문자열(임베딩 기준 nearest-token)으로 정의되어, 사후 해석 의존도를 낮추는 것이 핵심이다.

- **Technical Challenges**: 어려움은 SAE 사전의 기하(학습된 decoder 방향)와 어휘(vocabulary) 임베딩의 기하가 기본적으로 분리돼 있어, 특징이 복원에는 기여하지만 어떤 토큰 방향과는 멀 수 있다는 점이다. VASAE는 기존 복원 목적에 더해 각 특징이 자신의 nearest-token 임베딩에 가까워지도록 하는 anchor loss를 추가하고, top-k 희소 코드 제약과 함께 최적화한다.

- **Empirical Impact**: 실험에서 VASAE는 표준 SAE와 비교해 복원 품질(분산 설명/예측 손실 보존)을 크게 해치지 않으면서, GPT-2-small의 경우 층 0–10에서 강한 정렬 기준(nearest-token alignment score 0.8) 이상의 특징 비율이 약 90% 수준으로 나타난다. Llama-3.1-8B에서도 얕은 층과 중간 층은 강하게 정렬되지만 최종 대표 층에서는 정렬이 제한적이며, mean sparse code를 뺀 이후 case study에서 남은 intrinsic token name들이 입력 주변 토큰들과 관련성이 큼을 관찰해 ‘훈련 중 특징-토큰 연결’ 가능성을 시사한다.



### Two-Stage Fine-Tuning for Protein Sequence Generation with Targeted Amino-Acid Composition (https://arxiv.org/abs/2606.27939)
Comments:
          17 pages, 5 figures, ICML 2026 Workshop GenBio

- **Prior Approaches**: 기존 단백질 언어모델(PLM)은 ProtGPT2, ProGen2, RITA 같은 사전학습 모델을 기반으로 생성의 ‘그럴듯함’(plausibility)을 우선시해 왔다. 제어 목적을 더했더라도 많은 방법이 분류/회귀 형태의 단일 스칼라 오라클(구조 신뢰도, 안정성, 효소 활성, 항균성 등)로 보상함수를 설계해 왔고, 아미노산 조성처럼 ‘분포 형태’ 자체를 맞추는 목표에는 직접적 정렬이 부족했다.

- **Core Contribution**: 이 논문은 합성 사료 단백질 설계처럼, 서열의 아미노산 조성(AA composition) 분포를 목표 프로파일 q에 가깝게 맞추면서도 길이 조건과 다양성을 유지해야 하는 ‘분포-제약 단백질 생성’을 다룬다. 두 단계 파이프라인으로, 도메인 어댑티브 fine-tuning(FT)으로 조성에 가까운 영역을 먼저 선점한 뒤, RL 기반 reward-weighted FT로 FT만으로는 충족이 어려운 세밀한 조성 제약을 추가로 강제한다.

- **Technical Challenges**: 핵심 과제는 조성 오차를 줄이려다 보면 서열이 비생물학적 패턴(플라우저빌리티 저하)으로 붕괴하거나, 모드 콜랩스가 발생할 수 있다는 점이다. 논문은 frozen reference 정책(FT 체크포인트) 대비 KL 페널티로 신뢰 구간을 만들고, 후보 생성→다양성 보강→조성 보상 가중 업데이트의 반복 구조를 사용한다; 또한 필수 아미노산 결핍을 더 강하게 벌점 주고, sulfur pool(Met/Cys)·aromatic-precursor pool(Phe/Tyr)처럼 생화학적 교환 가능성을 반영하며, 목표 조성이 0인 잔기를 위한 zero-target amplifier까지 포함한 differentiated composition reward를 제안한다.

- **Empirical Impact**: 두 개의 목표 조성(qA, qB)에서, FT만으로는 평균 조성 접근은 개선되지만 ‘세부 제약 충족’ 지표가 약하게 남는 반면 RL 단계를 더하면 JSD가 추가로 크게 감소하고 여러 조성 제약 지표가 포화(saturation)에 도달한다. 한편 NetSolP 예측 용해도는 기준선을 상회해 실용적 생물학적 타당성을 유지하며, ESM-2 pPPL 같은 단백질-유사도 지표는 상승(플라우저빌리티 비용)하지만 조성 정렬이 가장 공격적인 경우에도 붕괴 수준은 보이지 않아, 분포 목표 정렬과 서열 품질의 균형을 실증적으로 보여준다.



### Agentic AI-Powered Re-Identification: An Emerging, Scalable Threat to Mobility Microdata Privacy (https://arxiv.org/abs/2606.27936)
Comments:
          15 pages, 2 figures

- **Prior Approaches**: 모빌리티 데이터는 식별자(이름 등)를 제거해도 궤적 자체의 고유성 때문에 재식별 위험이 높다는 연구가 누적돼 왔다. 다만 기존 공격은 OSINT 작업과 좌표-신원 연결에 숙련된 인력 투입이 필요해 실무적으로 대규모 확장이 어려웠다. 또한 억제/공간 일반화/궤적 교란 같은 전통적 익명화는 외부 공개정보와 교차검증을 하면 한계가 드러난다.

- **Core Contribution**: 이 논문은 LLM agent가 웹 탐색·공개기록/소셜 교차대조·좌표열을 후보 신원으로 귀결하는 end-to-end 파이프라인을 제시한다. 사람 개입 없이 자동으로 후보를 찾고 검증까지 수행해, 기존에 인간 분석가의 ‘노동’이 병목이던 위협 모델을 근본적으로 바꿨다는 점을 보여준다. 저자들은 이를 통해 SDC의 ‘de facto anonymity’ 가정이 흔들릴 수 있음을 경고한다.

- **Technical Challenges**: 핵심 난제는 (1) HOME/WORK 같은 공간 앵커를 좌표에서 안정적으로 추출하고, (2) 주소 수준 특이성을 확보한 뒤, (3) 공개소스 기반으로 신원 후보를 검증해 확실성 기준을 넘기는 것이다. 논문은 단계별(공간 분석→주소/건물 정합→후보 스코어링→소셜·공개근거로 재확인→최종 합성)로 evidence를 단조(monotonic)하게 누적하고, 품질 게이트와 uncertainty ledger로 약한 증거 전파를 차단한다. 또한 위협을 재현하기 위해 실제 브로커 데이터가 아닌, 동의 기반 실제 가정(가정/직장지) 주변의 시뮬레이션 GPS로 통제된 평가를 수행한다.

- **Empirical Impact**: 평가 결과, 재식별이 ‘가능한’ 25개 케이스 중 18개(72%)에서 개인을 특정했고, 전체 43개 중 18개(41.9%)에서 재식별에 성공했다. 반대로 재식별 불가로 분류해야 하는 16개 중 14개(87.5%)를 올바르게 중단했으며, 평균 비용은 목표당 2.24달러, 평균 처리시간은 17분(병렬 실행 가능)으로 보고됐다. 인간 분석가의 통제 실험은 없지만, 분당·수천 타깃 수준의 확장이 현실화될 수 있어 GDPR Recital-26 및 SDC 실무에 대한 ‘에이전틱 AI 기반 리식별 위험’ 재평가가 필요하다는 메시지가 크다.



### Home3D 1.0: A High-Fidelity Image-to-3D Asset Generation System for Interior Design (https://arxiv.org/abs/2606.27923)
Comments:
          18 pages, 10 figures, 2 tables; technical report

- **Prior Approaches**: 기존 image-to-3D는 범주에 맞는 그럴듯한 형태를 만드는 데 초점이 많아, 가구처럼 제품 동일성을 요구하는 영역에는 품질이 부족하다는 지적이 나온다. 특히 메시의 잡음/누락/얇은 부품 파손, 뷰 간 텍스처 불일치와 패브릭·우드·가죽 결 같은 미세 디테일 저하, 그리고 직접 생성한 PBR 맵의 알베도·거칠기·메탈릭·놈 디테일 및 재질 정체성 복원 실패가 문제였다. 또한 부품 분해가 다리·등받이처럼 기능 단위로 이뤄지는 경우가 많아, 디자이너가 실제로 편집하는 ‘재질 단위’ 편집 니즈와 어긋났다.

- **Core Contribution**: Home3D 1.0은 단일 기준 이미지로 ‘실사용 가능한’ 인테리어 가구 3D 자산을 만들기 위해 geometry–texture–material–parts의 모듈형 파이프라인을 제안한다. 핵심 목표는 제품의 형태/외관 정체성을 보존하면서, 재질이 편집 가능하도록 PBR 재질을 영역 단위로 붙이는 것이다. 이를 위해 텍스처는 multiview albedo 재투영과 3D 기반 텍스처 완성을, 재질은 MatWeaver로 재질 영역 분할 및 PBR 라이브러리에서의 계층적 검색을, 파트는 material-editable semantic part mesh 생성을 지향한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 단일 RGB에서도 방수 watertight 메시에 가까운 기하를 안정적으로 복원하고, (2) 조명 변화가 섞인 관측에서 알베도를 일관되게 복원하며, (3) 생성 PBR 맵을 처음부터 모두 그리는 대신 물리적으로 타당한 재질을 영역에 맞게 ‘매칭·베이킹’하는 것이다. 논문은 geometry에 coarse-to-fine latent SDF(geometry VAE + flow matching DiT)를, texture에 멀티뷰 알베도 동시 예측 후 메시로 reproject와 3D texture field 완성을 적용해 뷰 간 일관성과 가려진 면 처리를 개선한다. material은 비디오 기반 segmentation과 UV-Face Atlas 투표로 재질 컴포넌트를 만들고, VLM 추론–cross-modal embedding–VLM reranking의 계층 검색으로 재질 라이브러리 샘플을 고른 뒤 baking으로 엔진용 PBR 맵을 압축까지 수행한다.

- **Empirical Impact**: 평가는 100케이스 가구 벤치마크에서 geometry/texture/material/parts를 각각 전용 지표로 독립 평가하며, geometry 성능은 image-to-3D 대비 상위 결과를 보였다고 밝힌다(예: CD 0.4936×10−3, EMD 5.1745×10−2, F1@0.01 0.6329). 또한 가구 도메인에서 실제로 쓰기 어려운 요소였던 얇은 구조·디테일 과평활, 텍스처 뷰 불일치, 재질 정합성의 불안정성을 모듈별로 겨냥해 ‘e-commerce 규모 생산’과 ‘재질 단위 편집’에 필요한 출력 형태를 더 가깝게 만든 점이 의미가 있다. 남은 격차는 더 넓은 배포를 위해 전 모듈의 일반화 및 생성 품질의 추가 안정화가 필요하다는 형태로 정리된다.



### Reflect-R1: Evidence-Driven Reflection for Self-Correction in Long Video Understanding (https://arxiv.org/abs/2606.27922)
Comments:
          18 pages, 6 figures, ECCV

- **Prior Approaches**: 기존 멀티모달 장기 비디오 이해의 reflection은 내부 파라미터에 의존한 closed-loop self-reflection이 대부분이었다. 이 방식은 외부의 객관적 시각 근거가 없어 blind confidence에 갇히고, 수정 과정에서 오히려 환각을 강화해 성능이 떨어지거나 무작위 변형으로 이어질 수 있다.
또한 강화학습으로 다단 reflection을 학습할 때 policy coupling 문제가 생겨, correction 논리 대신 초기 추정을 반복하는 최적화 지름길을 택하기 쉽고 학습 데이터도 부족해 한계가 누적됐다.

- **Core Contribution**: Reflect-R1은 장기 비디오 이해에서 evidence-driven self-correction을 목표로, intuition–verification–arbitration 3단 파이프라인을 처음으로 제안한다. 직관 답변은 생성하되, verification 단계에서는 retrieved keyframes만으로 독립 검증을 수행하고, 마지막 arbitration이 두 결과의 충돌을 근거 기반으로 정리한다.
불충분한 증거가 나오면 temporal search 도구를 반복 호출해 증거를 확보함으로써 환각 루프를 끊는다는 점이 핵심이다.

- **Technical Challenges**: 핵심 난제는 (1) 객관적 외부 근거 없이 내부 추론만으로는 verification이 붕괴되는 문제와 (2) 다단 강화학습에서 단계 간 보상/업데이트가 얽혀 policy coupling이 발생하는 문제였다.
저자들은 전자를 retrieved keyframes 기반의 엄격한 정보 격리(verification 입력 제한)와 abstention 보상 설계로 완화했고, 후자는 Stage-Decoupled GRPO(SD-GRPO)에서 단계별 advantage를 독립 계산해 단계 간 그라디언트 경쟁을 차단했다.

- **Empirical Impact**: VideoMME, LongVideoBench, MLVU 등 주요 벤치마크에서 state-of-the-art 성능을 달성했으며, 비디오 길이가 길수록 향상 폭이 커져 decoupled verification의 견고함을 보여줬다. 특히 반영 전후 정확도를 비교한 reflection reliability 결과에서 closed-loop 방식들이 보이는 성능 저하와 달리 Reflect-R1은 일관된 개선을 기록했다.
또한 ablation 및 학습 동역학 분석에서 SD-GRPO가 단계별 보상 신호를 분리해 진짜 self-correction 능력이 형성됨을 확인했으며, +2.82%(LongVideoBench), +1.41%(VideoMME) 수준의 genuine rectification 향상을 보고했다.



### Every Step of the Way: Video-based Parkinsonian Turning Step Counting (https://arxiv.org/abs/2606.27918)
- **Prior Approaches**: 기존 PD 보행·턴 분석은 임상 영상에서 pose estimation을 이용해 관절·보행역학 지표를 계산하거나, IMU·압력센서 같은 웨어러블로 걸음 이벤트를 검출하는 방식이 주를 이뤘다. 하지만 웨어러블은 착용·센서 위치 민감도와 일상 사용의 불편이 크고, 영상 기반 방법은 실생활의 비정형(측면/후진/피벗/미세 셔플) 턴 걸음에 가정이 잘 깨지는 문제가 있었다.

- **Core Contribution**: 이 논문은 스마트폰 등 일상 환경 카메라로 얻는 동영상에서 PD의 턴 걸음 수를 수동적(passive)으로 추정하는 coarse-to-fine 프레임워크를 제안한다. 3D human mesh recovery로 만든 발 움직임 신호로 거친 step count를 먼저 뽑고, 이후 optical flow의 미세 단서를 cross attention으로 결합해 초기 추정을 정밀 보정한다.

- **Technical Challenges**: PD 턴에서는 발이 완전히 들리지 않는 shuffling·sliding, ‘footskate’ 같은 메쉬 복원 아티팩트, 개인·질병 단계별 비주기적 변동이 커서 단순 피크 카운팅만으로는 오차가 크게 난다. 이 때문에 mesh 기반 거친 신호를 시작점으로 하되, flow에서 픽셀 수준의 미세 움직임을 끌어오도록 cross attention을 설계하고, 가변 길이 영상은 클립 분할 뒤 multiple instance learning(MIL)으로 잔차(residual) 보정과 신뢰도 높은 클립 선별을 함께 학습한다.

- **Empirical Impact**: 실제 임상·가정 환경 데이터셋(PD-FOG, Turn-REMAP)에서 기존 step counting 방법과 반복활동 카운팅(RAC) 및 IMU 기반 접근을 폭넓게 능가했으며, PD-FOG에서 Lucot et al.[imu2] 대비 절대 오차를 37.7에서 15.9로 줄였다. 또한 예측된 턴 걸음 수가 MDS-UPDRS(골드 스탠더드 임상 점수)와의 임상적 관련성도 보이며, 향후 웨어러블 없이도 턴 장애를 정량화하는 데 의미가 크다.



### Triadic Werewolf: A Jester Role for Multi-Hop Theory of Mind in LLMs (https://arxiv.org/abs/2606.27909)
- **Prior Approaches**: 기존 Werewolf류 사회추론 벤치마크는 Villagers vs Werewolves처럼 두 진영(dyadic)으로 나뉘어, 관측 신호가 숨은 역할을 한쪽으로만 강하게 밀어주는 문제가 있었다. 이 구조에서는 언어적 사전지식/표면 휴리스틱(예: “의심스러워 보이면 퇴출”)만으로도 점수를 높일 수 있어, 진짜 theory-of-mind(ToM) 추론 여부가 흐려진다.

- **Core Contribution**: 논문은 Werewolf에 Jester(제스터)를 추가해 3자(삼자) 인센티브 구조를 만든다. Jester는 “의심받을수록” 이롭지만, 정작 승리는 자신이 투표로 퇴출될 때 발생하므로 동일한 관측 신호가 서로 반대의 최적 행동을 요구하게 된다.

- **Technical Challenges**: 핵심은 관측 신호(상대의 peer suspicion)가 곧바로 정답 행동(누굴 내쫓을지)으로 연결되지 않게 설계하고, 실제로 모델이 그 모순을 multi-hop ToM으로 풀어내는지 계량화하는 것이다. 이를 위해 10인 3자 벤치마크에 bidding-based debate 프로토콜과 self-learning 루프(ON/OFF)를 결합하고, 경기별/발화별 의심도·기만 유형·투표 결과로 추론 실패 패턴을 분해해 측정했다.

- **Empirical Impact**: 60게임 평가에서 Jester는 약 55%대 승률로 관측되며, Werewolves는 20%를 넘지 못했다. GPT-4.1의 Werewolves는 day 1에 60–70% 확률로 Jester를 “투표로 퇴출”하는데, 이는 그들 팀의 자멸적 행동으로 나타났고(엄밀히 말해 self-defeating vote), self-learning은 모델에 따라 이 경향을 완화/악화시키되 Jester가 특히 이득을 얻는 쪽(다른 진영이 cue를 잘못 읽는 상황)을 강화했다.



### SEADA: An efficient methodology for optimizing mixed-precision DNNs on multi-precision spatial architectures (https://arxiv.org/abs/2606.27884)
- **Prior Approaches**: 기존 mixed-precision 연구는 레이어별 정밀도 선택을 최적화(예: ILP, knapsack)하거나, spatial accelerator에서 매핑·스케줄링을 별도로 최적화하는 방식으로 진행돼 왔습니다. 하지만 bMAC 같은 고수준 프록시만 사용하거나, 저정밀–고정밀 경계에서 발생하는 (de-)quantization 오버헤드·데이터 이동·매핑 제약을 충분히 반영하지 못해 정확한 비용/이득 추정이 어려웠습니다. 또한 전체 설계공간은 레이어 수와 후보 정밀도 조합에 비례해 폭발적으로 커져(사실상 \(B^L\)) 반복적 탐색이 비현실적이었습니다.

- **Core Contribution**: 본 논문은 multi-precision spatial accelerator에 mixed-precision 네트워크를 올릴 때의 비용·정확도 트레이드오프를 빠르게 추정하고 거의 최적의 레이어 정밀도를 찾기 위한 방법 SEADA를 제안합니다. SEADA는 (1) 정밀도별 system-level 분석 비용 모델, (2) near-optimal 매핑 도구, (3) floating-point 계층의 혼합 정밀 이득을 계산하는 분석 모델, (4) bit-level entropy 기반 per-layer precision 선택을 결합해, 반복적 다중 구성 탐색 없이 설계공간 탐색을 가능하게 합니다. 특히 quantization/de-quantization 비용을 후속 부동소수점 연산 계수에 fusing하는 관찰을 통해 오버헤드 과대평가 문제를 줄이려 합니다.

- **Technical Challenges**: 핵심 난제는 레이어별 양자화 민감도와 아키텍처 이질성(서로 다른 연산 유닛·데이터 패스·제약) 및 시스템-level 비용(연산+데이터 이동+정밀도 변환)을 동시에 반영해 정밀도 구성을 결정하는 것입니다. SEADA는 QuickFlow를 기반으로 multi-precision 처리를 단일 시스템 템플릿 안에서 모델링하도록 확장해, 저정밀 커널이 INT32 누산까지 포함하는 클러스터 제약과 packing/unpacking, adder reduction tree 변화를 정교하게 반영합니다. 또한 MHSA·LayerNorm·BatchNorm 등 floating-point 구간은 입력 분해/근사 기반 분석 모델로 활동도(activity)를 추정하고, (de-)quantization 융합으로 비용의 정밀도 의존성을 낮춰 설계공간 탐색 속도를 확보합니다.

- **Empirical Impact**: 논문은 BERT-base와 ResNet-50을 사용해 SEADA의 “빠른 설계공간 탐색” 유효성을 보여주며, accelerator 메모리 계층·정밀도 선택·목표 지표를 함께 최적화하는 holistic 탐색을 수행합니다. 또한 정확도 열화(accuracy degradation)를 레이어별 정밀도 분포 대비로 특성화하면 hardware 수준에서 bMAC 수 같은 단일 메트릭으로도 정확도 추정이 가능하다는 통찰을 제시하고, 이를 Computational Bit Reduction (CBR)로 정리합니다. 결과적으로 설계자들이 multi-precision accelerator의 방대한 후보 조합을 반복 학습/매핑 없이 빠르게 줄여 의사결정을 할 수 있는 프레임워크를 제공하는 데 의미가 있습니다.



### A Study of Temporal Fusion Strategies for Named Entity Recognition in Historical Texts (https://arxiv.org/abs/2606.27881)
- **Prior Approaches**: 기존에는 역사 텍스트 NER에서 시간에 따른 개체 표면형 변화·등장/소멸·중의성 증가를 다루기 위해 데이터 증강이나 샘플링 등 데이터 중심 접근이 주로 사용돼 왔다. 또 time vectors, timestamp-aware pretraining, temporal graphs, dynamic knowledge editing 같은 시간 표현/지식 편집 기법이 제안됐지만, 토큰 분류형 NER 아키텍처에 시간을 구조적으로 어디·어떻게 결합하는지는 체계적으로 비교되지 않았다.

- **Core Contribution**: 이 논문은 Transformer 기반 NER에 출판 연도 같은 temporal metadata를 ‘구조적으로’ 주입하는 방법을 설계·비교한다. early fusion(인코딩 전/중)과 late fusion(인코딩 후)로 나누고, absolute와 time-distance(기준 연도 대비 상대 거리)라는 시간 표현 방식까지 함께 실험하며 어떤 결합이 더 견고한지 정리한다.

- **Technical Challenges**: 핵심 과제는 시간 정보를 토큰 수준 추론에 실질적으로 반영하되, 잡음이 큰 OCR·다국어 변이 환경에서도 성능이 흔들리지 않게 만드는 것이다. 이를 위해 cross-attention, adapters(경량 모듈), concatenation, FiLM-like modulation처럼 연도 임베딩을 다양한 지점에 결합하는 경량 fusion 전략을 구현하고, gold-year를 직접 쓰지 않도록 probing(추론 시 랜덤 year 주입)으로 내부 temporal internalisation 여부를 점검한다.

- **Empirical Impact**: HIPE-2020의 프랑스/독일 역사 데이터에서 late fusion 전략이 전반적으로 더 robust하고 시간 일반화 성능이 좋았으며, 특히 early/noisy 구간에서 이득이 두드러졌다. 또한 probing과 t-test 결과는 대부분의 개선이 연도 전반에 미묘하게 나타나지만, late-cross-attention은 baseline 대비 유의미한 차이를 보이는 등 구조적 시간 결합이 실제로 효과가 있음을 시사한다.



### SpatialUAV: Benchmarking Spatial Intelligence for Low-Altitude UAV Perception, Collaboration, and Motion (https://arxiv.org/abs/2606.27876)
Comments:
          10 pages, 7 figures

- **Prior Approaches**: 기존 UAV·공간지능 벤치마크는 이미지 수준 인식, 단일 시점 이해, 제한된 답변 형식에 치우친 경우가 많아 저고도 비행에서 핵심인 3D 공간 추론과 교차 시점 정합을 충분히 평가하지 못했다. 또한 실환경 드론 시점(탑다운·사선, 시점 왜곡, 고도 스케일 변화, 가림, 항공-지상 불일치)과 장면 동역학(시간·움직임)까지 포괄하는 진단형 과제가 부족했다. 결과적으로 교차뷰 연관성, 구조화된 grounding, 기하 추론, 시간적 시점 이해 같은 항목이 분절적으로 다뤄지거나 누락됐다.

- **Core Contribution**: 이 논문은 실제 저고도 UAV 관측을 기반으로 한 SpatialUAV 벤치마크를 제안한다. 총 4,331개 인스턴스와 14개 fine-grained 태스크를 단일 시각입력-질문-답변(visual-input–question–answer) 스키마로 묶어 의미 구분, 공간 관계, aerial–aerial 협업, aerial–ground 협업, motion 이해를 함께 평가한다. 입력 7종 구성과 답변 9종 형식을 지원해 옵션 라벨부터 영역 식별, 기하 값, 교차뷰 대응, 자유형 동작 서술까지 다양한 과제를 진단 가능하게 만든다.

- **Technical Challenges**: 기여를 신뢰성 있게 평가하려면(1) 텍스트 편향·포맷 쇼트컷을 제거하고, (2) 시점 간 대응과 기하 정답을 일관되게 산출하며, (3) 서로 다른 출력 형태를 동일한 품질 기준으로 비교해야 한다. 논문은 detector-assisted regions, depth supervision(추정 기반), 메타데이터 규칙, 대규모 수기 라벨링을 결합하고, DeepSeek-V4-Pro와 Qwen3.6-27B로 시각 없이도 맞히는 샘플을 blind filtering한다. 또한 2단계 검증(전수 인간 교차검증 + 대표 VLM 3종의 불일치 케이스를 재검토)과 태스크별 측정치를 적용해 이질적 출력도 안정적으로 채점한다.

- **Empirical Impact**: 대표적인 vision-language model들을 3개 범주에서 평가한 결과, 평균적으로 인간 수준(89.0%) 대비 큰 격차가 남아 있으며 최고 모델도 56.7%에 그친다. 특히 cross-view association, 구조화된 grounding, 기하 추론, temporal viewpoint understanding에서 병목이 두드러졌고, aerial–ground 협업조차 56.0% 수준에 머물렀다. 또한 spatial-specific 사전학습 모델은 저고도 시점 왜곡에 잘 전이되지 않아(예: 최고 spatial-specific 29.7%), 입력 해상도 4배를 올려도 이득이 제한적이어서 향후 UAV 데이터 기반 도메인 튜닝 및 에이전트형 도구 연계(grounding·매칭·기하추정·모션 분석)가 필요함을 시사한다.



### S$^2$-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation (https://arxiv.org/abs/2606.27872)
Comments:
          Accepted to IJCAI 2026

- **Prior Approaches**: 기존 VLA( Vision-Language-Action )는 end-to-end fine-tuning으로 비전·언어·행동 표현을 정책에 연결하지만, 장기 과제에서 누적 오류가 크게 늘어나는 문제가 있었다. 이는 시각·언어·행동의 결합이 대부분 static feature fusion(고정 가중 결합)으로 설계되어 작업 단계가 바뀌어도 같은 비중으로 정보를 섞기 때문이다. 그 결과, 집는 단계에서는 공간 정밀도가, 계획 단계에서는 semantic intent가 충분히 강조되지 못하고 초기 편향이 실행 체인을 따라 증폭되기 쉽다.

- **Core Contribution**: 이 논문은 S$^2$-VLA를 제안하며, 핵심은 State-Space Guided Adaptive Attention(SSGAA)으로 단계별 적응형 융합을 구현한 것이다. 모델은 belief state(내재화된 상태)를 통해 task progression을 추적하고, 비전(공간 인지), 언어(고수준 의도), 과거 행동(시간적 일관성) 세 축의 gating 가중치를 동적으로 생성해 작업 국면에 맞게 관심을 이동한다. 특히 2B 파라미터의 소형 구조로도 더 큰 7B급을 넘어 장기 조작에서 state-of-the-art를 달성하는 점을 강조한다.

- **Technical Challenges**: 관건은 (1) 장기 실행에서 temporal coherence를 유지할 belief state를 어떻게 구성하고, (2) 그 상태를 이용해 멀티모달 융합 비중을 안정적으로 바꾸는 방법을 찾는 데 있었다. 논문은 최근 행동-감각 쌍을 요약하는 belief state를 GRU 기반으로 재귀 업데이트하고, action prediction loss만으로도 이 상태가 과제 단계와 실행 품질 정보를 자연스럽게 내재화되도록 학습시킨다. 이어 SSGAA는 시각/의도/행동을 각각 분기한 뒤 belief state로 제어되는 soft gating으로 융합하며, 실험적으로는 특정 중간 레이어(레이어 12) 단일 지점에서의 gating이 가장 유리함을 보여준다.

- **Empirical Impact**: LIBERO와 SimplerEnv, 그리고 실로봇 ALOHA(바이매니얼)에서 일관되게 높은 장기 성공률을 보고하며, LIBERO에서는 평균 success rate 98.2%로 성능을 끌어올렸다. SimplerEnv-Bridge 같은 고현실 시뮬레이션에서도 2B 소형 모델이 SOTA 수준을 유지해 long-horizon error propagation 완화 효과를 뒷받침한다. 또한 정적 fusion을 대체하는 동적 초점 전환이 단순 스케일링보다 효과적임을 보여, 장기 로보틱 조작에서 적응형 feature fusion의 중요성을 실증적으로 강화했다.



### GNBAN: Graph Neural Basis Attention Networks for Long-Horizon Forecasting over Large Entity Sets (https://arxiv.org/abs/2606.27863)
Comments:
          12 pages, 3 Figure

- **Prior Approaches**: 대규모 소매 수요 예측은 SKU-매장 단위로 수만 개 시계열을 장기 구간까지 예측해야 하지만, ETS·ARIMA 같은 고전 통계는 시계열마다 별도 모델을 운영해야 해 확장·관리가 어렵다. DeepAR·Temporal Fusion Transformer 계열은 전역(global) 학습으로 효율을 높이지만, 제품·매장 등 개체 간 관계를 장기 예측에 충분히 구조적으로 반영하진 못한다. 그래프 기반 방법은 교차 개체 의존성을 메세지 패싱으로 담지만, 예측 헤드가 MLP처럼 블랙박스여서 장기 구간에서 불투명해지고 해석성이 떨어진다는 한계가 있다.

- **Core Contribution**: 이 논문은 GNBAN(Graph Neural Basis Attention Network)으로, 이기종(heterogeneous) 그래프 표현학습과 해석 가능한 basis 분해 예측 헤드를 end-to-end로 결합한 단일 전역 예측기를 제안한다. 예측을 horizon 전체를 직접 내는 방식 대신 trend(추세)·seasonal(계절)·generic(잔차) 3개 성분의 합으로 분해해, 그래프가 포착한 관계를 유지하면서도 예측의 구성 요소를 드러내도록 설계했다. 특히 basis별로 독립적인 learnable query를 두는 per-basis attention으로 각 성분이 서로 다른 이력 이웃(neighborhood)에서 특화된 정보를 가져가게 만든 점이 핵심이다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 대규모 카탈로그를 아우르는 단일 모델로 학습·추론을 안정화하고, (2) 그래프 기반 표현을 장기 horizon 예측에 맞게 ‘해석 가능한 형태’로 변환하며, (3) 성분(추세·계절·잔차)이 같은 과거 정보에서 서로 다른 패턴을 분리해 활용하도록 만드는 것이다. GNBAN은 relational schema에서 바로 이기종 그래프를 구성하고, GraphSAGE 기반 인코더로 target 엔터티와 과거 sales 노드 이웃을 집계해 문맥 임베딩을 만든 뒤, basis 함수 사전(polynomial·sinusoidal·learnable generic)과 per-basis attention으로 성분별로 독립 조회를 수행한다. 또한 전체를 하나의 훈련 목표로 end-to-end 최적화해 예측 헤드의 분해 능력과 그래프 표현 학습이 함께 맞물리도록 했다.

- **Empirical Impact**: GNBAN은 M5 Walmart와 Favorita Grocery Sales 두 대형 벤치마크에서 matched 프로토콜로 평가했을 때, 그래프 기반 기준(Base Graph) 대비 volume-weighted WRMSSE를 약 4~5% 개선했다. 성능 향상은 예측 헤드만 다른 실험에서도 일관되게 나타나, 단순 모델 용량 증가가 아니라 basis-attention 기반 분해 메커니즘의 효과임을 시사한다. 정성 분석에서는 trend·seasonal·generic 성분이 각각 추세 레벨, 반복 주기 패턴, 프로모션·단기 스파이크 같은 잔차 요인을 구분해 보여주며, post-hoc 설명(SHAP/LIME) 없이도 예측 근거를 내부적으로 제공한다.



### Applicability of memorization indicators for early spotting of overfitting while recalibrating sEMG-decoders on low sample sizes (https://arxiv.org/abs/2606.27855)
- **Prior Approaches**: sEMG 디코더는 사용자별 신체·전극 배치 차이 때문에 보통 transfer learning 기반 fine-tuning(재보정)이 필요하지만, 실제 캘리브레이션에 수집 가능한 반복 수가 적어 과적합 위험이 커진다. 이를 줄이기 위한 early stopping 등은 검증 세트가 필요해 사용자 캘리브레이션 현장 적용이 어렵고, 반복 데이터를 더 모으거나 같은 반복에서 떼어 쓰는 방식도 한계가 있었다.

- **Core Contribution**: 본 논문은 ReLU 뉴런의 activation 통계를 이용한 memorization indicator가, 추가 검증 데이터 없이도 저표본 sEMG 캘리브레이션에서 과적합 실패를 조기에 감지할 수 있는지 실험한다. NinaPro DB2 DB2의 subject-specific fine-tuning에서 test accuracy 저하가 마지막 은닉층의 activation rate 변화(특히 감소)와 함께 나타난다는 첫 실증 증거를 제시한다.

- **Technical Challenges**: 핵심 과제는 “검증 성능”을 대신할 수 있는 지표를, 캘리브레이션에서 얻는 극소량 학습 데이터만으로 계산해야 한다는 점이다. 이를 위해 학습 중 별도 validation set 없이 마지막 은닉층의 ReLU activation rate(출력이 0보다 큰 비율)를 계산하고, mean activation rate(MAR)뿐 아니라 분위수(25%, median, 75%)와 coefficient of variation 같은 분포 통계까지 확장해 학습 동역학을 모니터링한다.

- **Empirical Impact**: 실험에서 사전학습 모델은 신규 사용자에 거의 근접한 성능(우연 수준)을 보였지만, fine-tuning 방식에 따라 일반화가 갈렸다. 특히 test accuracy가 나빠지는 fine-tuning은 MAR 감소와 함께 activation 분포의 특징적 변화(분산 증가 등)로 구분되며, 10명 사용자 조건에서도 경향이 안정적이었다. 절대값 기준선은 데이터·아키텍처 의존성이 있어 threshold 제안은 아니지만, edge에서 온라인 모니터링 가능한 가벼운 지표로서 의미가 크다.



### WattLayer: Get Layers Right to Estimate Inference Energy of Neural Networks (https://arxiv.org/abs/2606.27841)
Comments:
          Accepted at IJCAI-ECAI 2026 Workshop SuRE

- **Prior Approaches**: 기존 에너지 추정은 주로 FLOPs/MACs 같은 연산량 기반 특성이나, 아키텍처 전체를 회귀로 맞추는 방식이 많았지만 태스크·하드웨어 변화에 약해 일반화가 제한되는 문제가 있었다. 일부 레이어 단위 접근도 있으나 특정 레이어/고정 GPU 조건에 국한되거나, 실제 실행 그래프가 아닌 정적 모듈 구조에 의존해 ResNet 같은 스킵 연결을 제대로 반영하기 어려웠다. 또한 추정 프레임워크가 표준화돼 있지 않아 작업 없이(실행 없이) 비교·의사결정이 어려웠다.

- **Core Contribution**: 이 논문은 WattLayer로, 태스크와 무관하게 신경망을 레이어 단위로 분해해 각 레이어의 에너지를 합산하는 task-independent layer-wise energy estimation 모델을 제안한다. 또한 실측 기반 보정 항(α)을 포함해 레이어 합산이 실제 아키텍처 총에너지와 어긋나는 부분을 보정한다. 나아가 아키텍처 전반에서 공유되는 레이어를 활용해 새로운 태스크에 완전 재학습 없이도 추정 성능을 유지하는 일반화 전략을 보여준다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 레이어별 에너지 측정의 재현성을 위해 샘플링 주기·반복 횟수(예: forward passes) 같은 실험 프로토콜을 표준화하는 것과 (2) PyTorch 모듈 이름이 아닌 실제 forward 실행 기준의 effective layer decomposition을 자동 추출하는 것이다. 저자들은 NVIDIA-smi 기반 계측(CodeCarbon 포함)과 충분한 반복 측정으로 오차를 낮추고, 실행 그래프를 따라 leaf module만 보존하는 훅 기반 분해로 skip/bypass까지 반영했다. 이후 레이어 타입별로 #MAC, #Activation, #Parameters, 입력 모양, 배치 크기, 커널 크기 등을 입력해 선형/다중선형/로그선형 회귀 등으로 레이어 에너지 함수를 학습하고, α로 시스템 오버헤드를 캘리브레이션했다.

- **Empirical Impact**: 대규모 데이터셋으로 295개 신경망 아키텍처의 100,000+ 레이어를 포함해 3개 태스크와 3개 하드웨어 플랫폼에서 평가했으며, 중앙값 오차 19.6%로 SOTA 대비 더 높은 정확도를 달성했다. 비전·텍스트·오디오 전반에서 레이어 단위 분해 모델이 더 중심화된 오차 분포를 보이며 성능 우위를 확인했고, A100/H100 등 다른 NVIDIA GPU에서도 MAPE가 유사하게 유지되는 적응성도 제시했다. 특히 학습에 text-generation 모델이 없던 상태에서 OPT/BLOOMZ/ GPT-Neo 계열 LLM을 fine-tuning 없이도 추정할 수 있어(예: MAPE ≤30% 수준) 지속가능한 AI를 위한 실행 전 에너지 예측 도구로의 확장 가능성을 보여줬다.



### Hippocampus-DETR: An Explicit Memory Object Detection Framework Based on Hippocampus Modeling (https://arxiv.org/abs/2606.27831)
- **Prior Approaches**: 현재의 객체 탐지 모델들은 명시적 메모리 메커니즘이 약해, 장면·패턴 정보의 재활용과 정교한 선택/통합이 제한적이라는 문제가 지적된다. DETR 계열을 포함한 주류 접근은 end-to-end 학습으로 성능을 끌어올리지만, 기억 구조를 분해·구성해 기능을 부여하는 방식은 상대적으로 부족했다.

- **Core Contribution**: 이 논문은 Hippocampus-DETR로, DETR 아키텍처에 생물학적 해마(hippocampus) 메모리 모델링을 기반으로 한 HipNet 모듈을 통합한다. 해마의 하위영역(entorhinal cortex, dentate gyrus, CA3, CA1, subiculum)을 구조적으로 시뮬레이션해 pattern separation, pattern completion, importance filtering, 정보 통합을 탐지 특징에 구현하는 것이 핵심이다.

- **Technical Challenges**: 메모리 기능을 객체 탐지에 실제로 결합하려면, 각 서브모듈이 어떤 방식으로 retrieval과 completion을 수행하고 서로 다른 역할을 갖도록 학습시킬지가 큰 기술 과제다. 논문은 레이어별(layer-wise) 훈련 전략으로 메모리 서브모듈을 단계적으로 최적화해, 내부적으로 기능이 분담되는 메모리 시스템을 형성하도록 설계했다.

- **Empirical Impact**: 실험에서 Hippocampus-DETR은 기존의 mainstream 객체 탐지 모델보다 높은 정확도를 보이며, few-shot 이미지 분류, 멀티모달 특징 구성, 이미지 복원 같은 설정에서 일반화 성능과 데이터 효율이 우수함을 입증했다. 또한 후속 실험으로 각 메모리 서브모듈의 기능적 필요성과 내부 해석가능성까지 검증해, 신경인지 메커니즘을 딥러닝에 통합하는 실용적 경로를 제시했다.



### Pepti-drift: Toxicity-Repulsive Drifting for Antigen-Conditioned Discrete Peptide Generation (https://arxiv.org/abs/2606.27824)
Comments:
          preprint

- **Prior Approaches**: 기존 항원-조건부 페브타이드 생성은 주로 ‘잘 결합하는 것’에 초점을 두되, 독성/용혈 같은 안전성 신호를 충분히 내재화하지 못했습니다. PepMLM은 마스킹 복원 기반으로 결합 예제를 학습하지만, 독성을 회피하는 명시적 반대 신호가 약해 결합-독성 특징이 겹칠 때 위험 후보가 섞일 수 있습니다. PepTune은 diffusion과 MCTG로 다중 성질을 반복적으로 최적화해 안전성을 다루지만, 추론 단계의 반복 탐색 비용이 커 대규모 스크리닝에 불리합니다.

- **Core Contribution**: Pepti-drift는 독성을 ‘피해야 할 음성 영역’으로 기하학적으로 인코딩해, 항원-특이 결합성과 안전성을 한 프레임워크에서 동시에 만족시키는 생성 방식을 제안합니다. 단일 antigen-conditioned drift step으로 생성 라티언트를 한 번에 정제하며, 매칭된 결합 영역으로는 끌어당기고 toxicity-associated 영역에서는 밀어내도록 벡터 필드를 설계합니다. 또한 Drifting Model 계열의 one-step refinement 원리를 채택해, 반복 denoising 없이도 제어 신호를 반영한 end-to-end 파이프라인을 구성합니다.

- **Technical Challenges**: 가장 큰 난제는 결합을 높이는 물리화학적 특징과 독성을 유발하는 특징이 임베딩 공간에서 서로 강하게 겹친다는 점입니다. 이 때문에 초기에 양(결합) 신호와 음(독성 회피) 신호를 같은 비중으로 주면 서로 경쟁하는 그라디언트가 생겨 드리프트가 안정적으로 학습되지 않았습니다. 저자들은 이를 warm-up 전략으로 해결해, 먼저 결합 지향 정렬을 충분히 학습한 뒤 점진적으로 repulsion(독성 회피) 가중치를 올려 학습을 안정화합니다. 아울러 raw ESM-2 임베딩은 기하학적 목적에 부적합해 정규화된 압축 투영으로 코사인 유사도 기반의 의미 있는 라티언트 공간을 만들고, non-autoregressive Transformer 디코더로 병렬 복원해 한 스텝 생성이 가능하게 했습니다.

- **Empirical Impact**: 항원 단위(split-controlled) 벤치마크에서 Pepti-drift는 유효하고 다양한 동시에 antigen-specific한 페브타이드를 생성하면서 예측 독성과 용혈 위험을 기존 방법보다 낮췄습니다. 관찰된 핵심 현상은 드리프트 적용 후 라티언트가 결합 양성 예시 쪽으로는 가까워지고, 가장 가까운 음성(독성) 예시 쪽으로는 멀어지는 방향 선택성이 일관되게 나타난다는 점입니다. 속도 측면에서는 1,095개 테스트 항원에 대해 Pepti-drift가 페프타이드당 시간 기준으로 PepMLM 대비 약 16배 빨랐고, PepTune 대비로는 3자릿수(천 배 이상) 수준의 고속 생성을 보이며 대규모 스크리닝 실용성을 높였습니다.



### Parameter-Efficient Quantum-Inspired Fast Weight Programmers for Traffic-Matrix Forecasting (https://arxiv.org/abs/2606.27821)
Comments:
          6 pages, 3 figures

- **Prior Approaches**: 기존 교통행렬(Traffic Matrices, TM) 예측은 링크/OD 추정에서 출발해 시계열 예측까지 확장됐지만, 최신 spatio-temporal 모델들은 graph·diffusion·transformer·생성모듈처럼 추가 구성요소가 많아 온라인 제약(메모리·업데이트·학습 예산) 하에서의 효율 평가가 상대적으로 부족했다. 또한 recurrent 계열 중 LSTM 같은 기본 구조는 스트리밍에는 유리하지만, OD 채널 간 결합 의존성을 예산 대비 얼마나 잘 담는지는 여전히 한계가 있었다.

- **Core Contribution**: 이 논문은 자원 제약을 전제로, 그래프/트랜스포머/디퓨전 같은 전용 모듈 없이도 compact quantum-inspired recurrent 모델이 whole-matrix TM을 잘 예측하는지 검증한다. 구체적으로 gated quantum-inspired Kolmogorov–Arnold 네트워크 fast-weight programmer(QKAN-FWP) 계열을 Abilene TM의 direct multi-step 예측에 적용하고, HQKAN(quantum-inspired 비선형 모듈)을 slow/fast 경로 어디에 배치하는지까지 비교한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 144개 OD 채널이 동시 결합된 고차원 행렬을 20-step 직접 예측으로 처리하면서 (2) 고정된 학습 예산 아래에서 수렴성과 일반화를 동시에 확보하는 것이다. 이를 위해 모든 모델에 공통 고정 epoch·학습률 프로토콜을 적용하고, FN-TM(frame-normalized TM)로 입력·오차 계산을 통일해 모델 비교의 공정성을 확보했으며, gated fast-weight에서 fast readout/slow programmer 내 HQKAN placement 변형을 통해 성능-효율 요인을 분리했다.

- **Empirical Impact**: Abilene에서 G-QKANFWP는 pooled RMSE 기준 가장 좋은 성능(0.06897±0.00030)을 보이면서도 LSTM-L 대비 파라미터는 22.4%만 사용해 정확도-효율 균형을 실증했다. 또한 고정 예산 수렴을 나타내는 validation-loss area under the learning curve(AULC)에서 quantum-inspired 변형들이 더 낮은 값을 보여 learning curve 상 이점이 확인됐고, OD-channel 단위에서도 G-QKANFWP와 GQKAN-FWP가 LSTM-S 및 고전 G-FWP를 상대로 더 많은 채널 wins를 기록했다. 결과적으로 “클래식 slow programmer + quantum-inspired fast programmer” 구성이 resource-conscious TM forecasting에 유망한 설계임을 제시했다.



### Optimizing Teacher-Student Partitioning for Scalable Knowledge Distillation on HPC Systems (https://arxiv.org/abs/2606.27797)
- **Prior Approaches**: knowledge distillation(KD)과 general knowledge distillation(GKD)은 큰 teacher의 분포를 작은 student가 따라가게 해 성능을 유지하면서 모델을 압축해 왔다. 하지만 HuggingFace TRL의 GKD는 teacher와 student를 대칭적으로 다뤄, 실제로는 메모리·통신 요구가 다른 두 경로를 같은 파티셔닝 규칙으로 묶는 문제가 있었다.

- **Core Contribution**: 이 논문은 teacher와 student의 비대칭(asymmetry)을 HPC 관점에서 분리해, teacher 파티셔닝·엔진은 teacher의 추론 성격에 최적화하고 student는 학습에 최적화하는 방법을 제안한다. 또한 vertical(DDP/ZeRO-스타일)과 horizontal( tensor parallelism, TP) 파티셔닝을 함께 고려해, teacher 쪽에는 inference-optimized 전략을 플러그인 형태로 적용한다.

- **Technical Challenges**: 핵심 난제는 TRL이 teacher forward를 DeepSpeed training engine으로 실행하면서 optimizer 관련 버퍼까지 할당해 불필요한 메모리 오버헤드를 만들고, 그 결과 가능한 micro-batch가 줄어 throughput이 제한된다는 점이다. 논문은 (1) teacher에 대한 ZeRO 단계 선택을 조정해 optimizer 상태/버퍼를 샤딩하고 (2) teacher와 student의 DeepSpeed 설정을 decouple해 teacher에는 불필요한 optimizer 할당이 없도록 만들며, 나아가 TP가 유리해지는 조건을 분석 모델로 정리해 inflection point 존재 가능성을 제시한다.

- **Empirical Impact**: Llama3 기반 실험에서 TRL 대비 최대 67% 더 높은 samples-per-second을 달성했으며, 특히 대규모 multi-node에서 teacher 메모리 오버헤드를 제거하면 통신 병목이 커져 teacher 파티셔닝 전략을 바꾸는 효과가 뚜렷해졌다. 회사 운영 환경의 production HPC 클러스터에서도 GKD 학습이 유의미하게 가속되어, 기존 GKD 구현이 놓친 teacher-student 비대칭 최적화의 실용적 impact을 보여준다.



### Position Bias Correction is Insufficient for One-Pass Attention Sorting (https://arxiv.org/abs/2606.27793)
- **Prior Approaches**: 롱컨텍스트 LLM에는 ‘lost-in-the-middle’로 대표되는 위치 편향이 있어, 문맥의 중간 정보가 덜 활용되는 문제가 반복적으로 관찰된다. 이를 문서 재정렬로 완화하려는 Attention Sorting은 생성 중 어텐션 패턴을 보며 문서를 여러 번 정렬해 성능을 끌어올리지만, sort-and-generate를 여러 회 수행해 지연 비용이 커진다. 기존 연구들은 위치 편향을 토큰 단위 보정이나 위치 인코딩 개선(예: RoPE 변형, 컨텍스트 확장)으로 다루었으나, 문서 레벨 재정렬의 반복을 줄이는 관점에서는 한계가 남아 있다.

- **Core Contribution**: 이 논문은 위치 편향이 반복 정렬의 핵심 병목일 것이라는 가설을 세우고, Debiased One-Pass Attention Sorting(편향 보정 1회 정렬)을 제안한다. 같은 프롬프트에서 얻은 ‘저어텐션 다수 문서’로부터 프롬프트별 position-bias curve를 추정한 뒤, raw attention score를 (차감 또는 나눗셈으로) 보정하여 단 1회 정렬로 iterative sorting에 근접하려고 한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 문맥 내에서 진짜 관련 문서와 편향만 반영한 주의를 분리해 bias curve를 추정하는 것과 (2) 보정이 실제로 정렬 순위를 개선하도록 점수 스케일을 안정적으로 만드는 것이다. 논문은 상위 α 비율 문서를 트림해 추정 신뢰도를 높이고, 위치 인덱스를 B개 구간으로 빈닝한 뒤 보정 모드(additive/divisive)를 선택해 debiased score로 정렬을 수행한다. 또한 모델별로 reordering 전략과 하이퍼파라미터를 별도 튜닝해 단일 패스의 계산량(정렬 1회+생성 1회)을 유지한다.

- **Empirical Impact**: SynthWiki@28K에서 LLaMA-2-7B-32K-Instruct는 debiasing이 raw single-pass sorting과 결과가 완전히 같아(94.83% containment accuracy) ‘위치 편향 보정만으로 반복의 이득을 대체’할 수 없음을 보여준다. 반면 YaRN-Llama-2-7b-64k에서는 debiasing이 +8.67pp 개선했지만, iterative sorting(k=5) 대비 14.84pp 뒤처져 격차의 37%만 메우는 데 그쳤다. 결론적으로 위치 편향 보정은 high-bias 모델에서 선택적으로 유의미하지만, iterative sorting이 제공하는 추가 이득(어텐션 컨텍스트 정제나 잡음 감소 등)을 단독 보정으로는 재현하기 어렵다는 메시지를 남긴다.



### NLL-Guided Full-Attention Layer Selection for Training-Free Sliding-Window Adaptation (https://arxiv.org/abs/2606.27791)
- **Prior Approaches**: 긴 문맥 추론을 위한 hybrid attention(SWAA)은 프리필에서 sliding-window attention(SWA)과 full attention(FA)을 섞어 효율을 얻지만, 어떤 레이어를 FA로 남겨야 하는지는 여전히 핵심 과제로 남아 있었다. 기존 해법은 주기적인 레이어 패턴처럼 고정 규칙을 쓰거나 LightTransfer처럼 attention 휴리스틱에 의존해 downstream 정확도에 무엇이 중요한지 직접 포착하지 못한다는 한계가 있었다.

- **Core Contribution**: 이 논문은 NLL-guided layer selection을 제안한다. 학습 없이 각 레이어가 FA를 유지할 때 answer 토큰의 NLL이 얼마나 덜 악화되는지(= NLL 감소량)로 레이어 중요도를 직접 측정해, FA를 유지할 상위 k개 레이어를 고른다.

- **Technical Challenges**: 기술적 과제는 “레이어별 중요도”를 생성 과정이 아닌 프리필 단계의 효과로 신뢰성 있게 분리해 점수화하는 데 있다. 저자는 teacher forcing으로 레이어별 토글 실험을 수행하되, SWA는 프롬프트 토큰에만 적용하고 answer 토큰은 full 맥락을 보이게 해(디코드 규칙 일치) NLL 차이를 레이어 기여로 해석 가능하게 만들었다. 또한 64개(16k~32k) 장문 예시로 약 15분 원샷 캘리브레이션 뒤, 이후 배치는 선택된 레이어 집합을 고정해 비용을 상쇄한다.

- **Empirical Impact**: LongMemEval에서 NLL-guided 1/4-FA는 64.6% 정확도로 1/2-FA periodic(65.0%)와 거의 비슷한 성능을 내면서 FA 연산 예산은 절반으로 줄였다. SWAA의 1/4-FA periodic 기준(54.2%) 대비 10.4%p, LightTransfer-style(동일 조건) 기준(38.2%) 대비 26.4%p 더 높아 데이터 기반 선택의 우위가 확인됐다. 장문/단문 캘리브레이션 비교에서 랭킹 상관이 낮고(ρ=0.306) 크기 차이도 커서, 신호가 일반적인 레이어 민감도보다 장거리 attention 필요에 더 가깝다는 점(교란 제거)이 제시됐다.



### SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict Mitigation in Retrieval-Augmented Generation (https://arxiv.org/abs/2606.27786)
Comments:
          19 pages, 13 Figures

- **Prior Approaches**: RAG은 검색 문서를 근거로 생성해 LLM의 사실성을 높이지만, 검색 문서의 지식이 모델의 parametric knowledge와 충돌하면 검색 증거를 무시하거나 특정 지식에 과도하게 의존하는 등 실패가 발생한다. 이를 줄이기 위해 지식 관련 neuron을 찾아 수정하거나, FFN/attention head 등 더 거친 단위의 고정 개입 규칙을 적용하는 연구가 이어져 왔다. 다만 neuron 단위는 국소화가 어렵고 취약하며, layer 단위 개입은 입력마다 충돌 양상이 달라 고정 규칙이 일반 능력을 해칠 수 있다.

- **Core Contribution**: SHIFT는 backbone LLM의 파라미터를 고정한 채, FFN 가지에 삽입한 learnable gate로 내부 활성(hidden-state)을 입력 의존적으로 조절해 지식 충돌을 완화한다. 기존처럼 특정 뉴런을 직접 편집하거나 사전에 선택한 레이어에 정적인 억제/강화를 거는 대신, 필요할 때는 검색 맥락을 더, 충돌 시에는 parametric knowledge의 영향력을 상대적으로 조정하도록 설계했다. 또한 GRPO로 게이트를 최적화해, 충돌 상황에서 근거 신뢰도에 맞춰 중재(arbitration)를 수행한다.

- **Technical Challenges**: 핵심 기술 과제는 (1) 충돌 유형마다 다른 균형 조절이 필요한데 고정 개입은 이를 반영하기 어렵고, (2) 뉴런/구조 수준 개입은 의도치 않은 연쇄 영향으로 모델의 일반 능력을 떨어뜨릴 수 있다는 점이다. SHIFT는 backbone을 freeze하고 FFN 기여에 대한 스칼라 게이트를 통해 activation을 약화/유지/증폭하는 최소 개입 구조를 택했다. 학습은 0.01% 미만의 경량 파라미터만 업데이트하며, GRPO(참조 정책 제약 포함)와 게이트 정규화 및 faithfulness 보상을 함께 써서 특정 입력에 과적합되지 않도록 안정적으로 중재 학습을 유도한다.

- **Empirical Impact**: 6개 데이터셋과 Qwen 백본 2종에서 SHIFT는 여러 경쟁 베이스라인을 일관되게 능가하며, 동일 백본 기준 평균적으로 EM/F1에서 유의미한 개선을 보였다. 특히 ConfiQA의 지식 충돌 세팅에서 SFT 대비 큰 폭의 향상을 보이면서도, MMLU 평가에서는 모델 성능 저하가 평균 0.5% 미만으로 매우 작아 일반 능력 보존 측면에서도 강점을 드러냈다. 또한 gate activation이 서로 다른 충돌 유형을 선형분리 가능하게 만들고(AUC 0.832), 입력에 따라 게이트가 구분된 조절 신호를 학습한다는 분석과 함께, 테스트 시에도 다양한 LLM/태스크로 전이되는 강건성을 확인했다.



### Output-Space Allocation Costs for Calibration-Guided LLM Compression: An Empirical Study (https://arxiv.org/abs/2606.27785)
- **Prior Approaches**: LLM 압축의 학습 없이(post-training, training-free) 진행되는 방식은 보통 weight-space 기반 비용을 쓰거나, calibration 데이터로 activation-aware 결정을 보정한다. 특히 ROCKET은 각 레이어의 factorization은 출력 복원(output reconstruction) 목적에서 얻으면서도, 레이어 간 예산 배분(MCKP)에서는 weight-space Frobenius error를 비용으로 사용한다는 점에서 ‘목적 불일치’가 존재한다. Activation-aware 대표 방법(AWQ, ASVD)은 activation 통계를 반영해 품질을 개선해 왔지만, ROCKET의 전역 배분 비용을 출력 공간으로 정렬하는 효과는 아직 명확히 검증되지 않았다.

- **Core Contribution**: 본 논문은 ROCKET의 MCKP에서 사용하는 allocation cost를 weight-space에서 output-space(whitened) error로 교체한 ROCKET-ActCost를 제안한다. 목표는 factorization을 이끈 출력 공간의 기준과, 전역 예산 배분의 기준을 정합시키는 것이다. 추가로 출력 공간 최적화로 인해 레이어별 최적 sparsity 설정(ksk_{s})까지 함께 바뀔 수 있음을 실험적으로 드러낸다.

- **Technical Challenges**: 핵심 기술 난제는 ‘출력 공간 error’를 MCKP 비용으로 쓸 때, 과도한 계산 오버헤드나 추가 calibration 패스를 요구하지 않으면서도 정확히 비용을 재정의하는 것이다. 저자들은 ROCKET의 profiling 단계에서 이미 계산되는 whitened weight(W_L)와 whitened reconstruction의 행렬들을 활용해 output-space error와 output-optimal ksk_{s}를 추가 패스 없이 산출하도록 설계했다. 또한 MCKP는 동일한 제약/구조를 유지하되 cost 함수만 교체해 비교 가능성을 확보했다.

- **Empirical Impact**: Qwen3-8B에서 50% 압축을 걸었을 때 ROCKET-ActCost는 8개 zero-shot 벤치마크 평균 정확도가 53.1%로 52.3% 대비 +0.8pp 상승했지만, WikiText-2 perplexity는 61.46으로 52.98 대비 16% 악화됐다. 이는 출력 공간 비용이 과제 정확도(task-relevant information)에는 유리하지만 언어 모델링 품질(perplexity)에는 불리할 수 있음을 보여주는 ‘accuracy-perplexity tradeoff’로 해석된다. 한편 weight-space와 output-space error 간 상관이 0.99 이상으로 매우 높아 레이어 예산 배분이 크게 갈라지지 못해 효과 크기가 제한됐고, Llama-3.2-1B에서는 20% 압축에서 두 방법 결과가 거의 동일해 비용 선택의 영향이 압축이 강할 때 더 두드러짐을 시사한다.



### Improving Adversarial Robustness via Activation Amplification and Attenuation (https://arxiv.org/abs/2606.27784)
Comments:
          Accepted to ECCV 2026

- **Prior Approaches**: 기존 적대적 공격(Adversarial examples)은 신경망의 비강건(non-robust) 특징을 교란해 오분류를 유도하며, 이를 막기 위해 adversarial training(AT, TRADES, MART 등)과 특징 내부를 바꾸는 plug-in 방어가 함께 발전해왔다. plug-in 계열은 비강건 채널을 마스킹/가지치기하거나(FS 계열, CAS·CIFS 등) robust/non-robust로 분리한 뒤 후자를 MLP로 재보정(FSR·FTA2C)하는 방식이 주류지만, 추가 재보정 블록의 파라미터 기여가 덜 해석 가능하다는 한계가 남았다.
또한 OOD에서 activation scaling이 효과적이라는 통찰이 있었지만, 이를 공격학습(미분 가능성과 안정성)으로 그대로 가져오면 top-k 같은 비미분 연산·큰 스케일로 인한 수치/그래디언트 문제에 부딪힐 수 있다.

- **Core Contribution**: 이 논문은 Activation Amplification and Attenuation(A3)라는 가벼운 플러그인 모듈을 제안해, 비강건 신호를 “억제(attenuation)”하면서도 같은 설계를 “증폭(amplification)” 모드로 전환할 수 있게 만든다. A3는 채널별 learnable mask와 활성 크기 분포에서 유도한 스케일링으로 activation을 동적으로 재조정하며, 증폭 모드에서 얻은 신호를 학습 시 negative reference로 써 새로운 contrastive·ranking loss를 구성한다.
특히 증폭/억제는 스케일 연산의 부호를 뒤집는 방식으로 통일되어, 추가 네트워크 용량을 크게 늘리지 않으면서도 두 모드의 상호작용을 통해 방어 성능을 끌어올린다고 주장한다.

- **Technical Challenges**: A3를 adversarial training에 적용할 때 핵심 기술 과제는 (1) top-k 기반 scaling을 쓰면 gradient obfuscation 위험이 커진다는 점과 (2) OOD용 스케일링의 지수형(exp) 형태가 커다란 팩터를 만들어 gradient 포화·수치 불안정을 유발할 수 있다는 점이다. 논문은 이를 피하기 위해 Gumbel-Softmax로 differentiable한 채널 마스크를 만들고, 지수 대신 로그(logarithmic) 형태의 스케일 팩터를 사용해 안정성을 확보한다.
또한 스케일 팩터를 [0,1] 범위로 제한해 극단적 증폭/감쇠를 막고, inference에서는 attenuated activation만 사용하며 amplification은 training-time loss 설계를 위한 음성 기준으로만 활용하도록 구성한다.

- **Empirical Impact**: 실험은 CIFAR-10/100과 Tiny ImageNet에서 ResNet-18·WideResNet-34-10 백본, 그리고 AT/TRADES/MART 같은 대표 adversarial training 세팅을 모두 포함해 수행됐다. A3를 추가하면 ensemble(Ens.) 정확도와 AutoAttack(AA) 등 강공격 평가에서 일관되게 향상되며, 예로 ResNet-18+AT에서 CIFAR-10 Ens.가 바닐라 대비 4.99%p 개선되는 결과가 제시된다.
정량적으로는 clean accuracy가 소폭 떨어질 수 있으나 이는 adversarial training의 일반적 트레이드오프 범주이며, 계산·메모리 오버헤드는 기존 plug-in 대비 매우 작다고 보고된다. 나아가 activation 분포 히스토그램 분석에서 A3(attenuation)가 adversarial에서만 나타나는 활성 피크를 효과적으로 눌러 separability를 높이고, amplification은 의도대로 clean/adv 분리성을 낮춘다는 해석도 함께 제공된다.



### RS-Diffuser: Risk-Sensitive Diffusion Planning with Distributional Value Guidanc (https://arxiv.org/abs/2606.27766)
Comments:
          ICIC 2026 Oral

- **Prior Approaches**: 오프라인 강화학습은 데이터만으로 정책을 학습해 안전한 의사결정을 노리지만, 데이터 분포 밖에서 행동할 때 분포 이동과 불안정한 가치 추정 문제가 커진다. 확산(difussion) 기반 결정/플래닝은 멀티모달 궤적을 잘 만들지만, 대부분 기대수익을 중심으로 최적화해 꼬리(tail) 위험을 명시적으로 다루지 못한다. 그 결과 평균 성능은 비슷해도 최악의 결과가 크게 다른 상황에서 안전성이 약화될 수 있다.

- **Core Contribution**: RS-Diffuser는 확산 플래너(미래 상태 궤적 생성)와 분포형 가치 critic(리턴 분포 추정)을 결합해 오프라인 확산 플래닝을 ‘위험 민감’하게 만든다. 또한 Monte Carlo 기반 분포형 학습(quantile regression)으로 리턴의 하위 꼬리를 직접 반영하고, 샘플링 시점에 CVaR 같은 tail-aware 목적의 그래디언트를 denoising 과정에 가이드한다. 이를 통해 하나의 학습된 모델을 risk parameter만 바꿔 risk-averse/risk-neutral/risk-seeking으로 유연하게 전환한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 확산 생성 과정에 (2) 분포형 critic이 주는 꼬리 민감도를 안정적으로 연결하는 것이다. RS-Diffuser는 상태만으로 미래 궤적을 생성하고, inverse dynamics로 실행 가능한 행동을 디코딩해 생성/통제를 분리했으며, critic은 quantile 분포를 학습해 VaR/CVaR 같은 꼬리 지표를 미분 가능한 가이던스 항으로 구성했다. 마지막으로 샘플링 시점의 risk-sensitive guidance를 통해 denoising 경로를 원하는 위험 프로필로 조향한다.

- **Empirical Impact**: 실험은 risk-sensitive D4RL과 risky robot navigation 벤치마크에서 평균 성능뿐 아니라 worst-case(예: CVaR0.1) 견고성과 안전 위반을 함께 개선함을 보여준다. 비교군 중 위험 중립 방법(CQL 등)은 기대수익 중심이라 꼬리 위험을 놓쳐 성능이 떨어졌고, 기존 위험 민감 기법들은 고정된 위험 목표에 매여 있어 상황 전환성이 제한적이었다. RS-Diffuser는 단일 모델로도 위험 파라미터 변화에 따라 CVaR·VaR 기반 동작 특성을 재현하며, 특히 CVaR0.1이 평균-견고성의 균형이 가장 좋다는 분석도 제시한다.



### Drop-Then-Recovery: How Redundant Are Vision-Language-Action Models? (https://arxiv.org/abs/2606.27755)
- **Prior Approaches**: VLA 모델은 비전-언어 백본과 행동 예측 모듈을 결합해 지시 기반 로봇 조작을 수행하지만, 대부분은 사전학습된 대형 언어 백본을 그대로 상속한다. 기존 압축 연구는 주로 양자화·pruning처럼 “제거 후 복구 없이” 성능을 평가해, 닫힌고리 제어에서 실제 성공에 필요한 용량을 가늠하기 어렵다.

- **Core Contribution**: 이 논문은 Drop-Then-Recovery(DTR)로 트랜스포머 블록을 제거한 뒤 다운스트림 제어 작업을 fine-tuning해, 제거된 용량이 실제로 필요한지 recoverability 관점에서 측정한다. 또한 어떤 블록을 제거할지 고르는 one-shot 가상 게이트 기반 지표인 GateProbe를 제안해, 단순 유사도나 크기, 즉각 성능 저하가 아닌 행동 손실 기여도를 기준으로 블록을 랭킹한다.

- **Technical Challenges**: 문제는 “즉각 손실”이 아니라 “복구 가능성”을 예측해야 한다는 점이며, 기존 정적 메트릭이나 Taylor류 그라디언트 기반은 극단 압축에서 안정성이 떨어지거나 비용이 크다. GateProbe는 각 블록의 residual 경로에 가상 gate를 둔 민감도(손실의 기대 절댓값)를 근사해, 체인룰로 모델 내부 게이트 삽입 없이 다운스트림 그라디언트와 residual 기여의 내적으로 계산하며 제거 후보를 효율적으로 선정한다.

- **Empirical Impact**: 시뮬레이션(LIBERO, LIBERO-Plus, RoboTwin 2.0)과 실제 로봇(UFACTORY xArm 850)에서 일관되게 “언어 백본은 중복이 크고, 비전·액션 경로는 제거에 취약”하다는 비대칭이 관찰됐다. 예를 들어 LIBERO에서 언어 LLM 블록 절반 제거가 OpenVLA-OFT의 경우 95.0%→98.3%로 기준을 넘어섰고, 언어 블록을 2개만 남겨도 baseline 수준 복구가 가능했다. 이는 현 VLA 벤치마크가 언어 grounding과 조합적 추론에 충분한 압력을 주지 못할 수 있으며, 향후 아키텍처는 언어-비전-액션에 용량을 더 의도적으로 배분하고 더 강한 언어·OOD 테스트가 필요하다는 신호로 받아들여진다.



### From General-Purpose Audio Tagging to Spatially Grounded Sound Event Localization and Detection (https://arxiv.org/abs/2606.27751)
Comments:
          Technical Report (KU Leuven - UnivAQ)

- **Prior Approaches**: 기존 SELD는 사운드 이벤트의 발생(SED)과 방향 추정(DOA)을 동시에 다루지만, 특징 설계나 학습 방식이 데이터/계산 제약에 취약해지는 경우가 많았다. 또 범용 오디오 태깅 모델의 사전지식(semantic audio priors)을 공간 추론으로 전이할 때, 어떤 입력 인터페이스와 결합 구조가 유효한지가 불명확했다. 결과적으로 데이터가 부족하거나 배포 조건이 까다로울수록 성능이 흔들리기 쉽다.

- **Core Contribution**: 이 논문은 pretrained General-Purpose Audio Tagging(GP-AT) 모델을 spatially grounded Sound Event Localization and Detection(SELD)로 확장하는 AT2SELD 프레임워크를 제안한다. AT 백본 위에 compact First-Order Ambisonics(FOA) 공간 처리를 결합하고, 트랙 단위 SED와 Cartesian DOA 추정을 하며, permutation aware supervision과 calibration까지 포함해 설계 전 과정을 연결한다. 또한 semantic-대-공간 전이가 실제로 작동하는 조건을 구조적으로 규명하고 최적화 전략을 제시한다.

- **Technical Challenges**: 핵심 기술 과제는 의미론적(semantic) 오디오 태깅 지식이 공간 방향 추정으로 안정적으로 “전이”되도록 인터페이스와 네트워크 용량의 병목을 찾는 것이다. 저자들은 NAS 기반의 멀티스테이지 탐색으로 (1) magnitude/phase/Intensity Vectors(IVs) 기반의 spectral FOA descriptor가 가장 신뢰도 높은 변환 통로임을, (2) early residual spatial encoding이 용량에 가장 민감한 구성 요소임을, (3) late track-wise coupling과 recurrent smoothing이 정교화에 더 유효함을 확인했다. 여기에 focal loss, activity-conditioned DOA supervision, threshold calibration 같은 훈련·후처리 진단을 통합해 비활성 타깃 지배와 보정 문제를 완화했다.

- **Empirical Impact**: STARSS23, TAU2019, TAU-NIGENS2020, TAU-NIGENS2021에서의 실험은 focal loss가 activity point를 개선하고, active-only DOA supervision이 inactive target dominance를 줄이며, validation-selected thresholds가 calibration을 회복시키되 공간 학습 자체를 대체하지는 않는다는 점을 보여준다. 분석 결과 TAU2019에서는 고정 소스(fixed source)에 대한 localization이 강하고, TAU NIGENS2021에서 학습된 표현이 다른 데이터로도 비교적 잘 전이되지만 STARSS23에서는 의미 있지만 불확실성이 남는 패턴을 보였다. 종합하면 GP-AT priors를 공간을 인지하는 아키텍처에 내장하고 calibration·배포 지향 전략까지 함께 최적화할 때 SELD 설계에 실질적 잠재력이 있음을 입증한다.



### Flexformer: Flexible Linear Transformer with Learnable Attention Kern (https://arxiv.org/abs/2606.27748)
- **Prior Approaches**: Transformer의 핵심인 dot-product softmax attention은 모든 토큰 쌍을 계산해 시퀀스 길이에 대해 시간·메모리가 O(N^2)로 커진다. 이를 줄이기 위한 sparse attention은 경험적으로 만든 패턴 의존성이 크고, kernel-based linear attention은 선형화는 달성하되 고정/약하게 학습되는 커널로 인해 표현력이 제한된다는 한계가 있었다.

- **Core Contribution**: Flexformer는 random Fourier features 기반 linear attention에서 주파수(커널 스펙트럼)를 end-to-end로 학습해, 더 넓은 범주의 attention kernel을 데이터 주도로 구성한다. stationary과 nonstationary 변형을 제시하며, nonstationary가 표현력이 더 크고 softmax 커널을 포함(따라서 복원 가능)한다는 보장도 함께 제안한다.

- **Technical Challenges**: 문제는 “학습 가능한 커널”을 선형 시간·공간 복잡도 안에서 유지하면서도 충분한 표현력을 얻는 것이다. Flexformer는 커널의 스펙트럴 표현을 random Fourier feature로 근사하되, 주파수를 고정 샘플링하지 않고 학습 파라미터로 두는 방식으로 해결한다(비정상(nonstationary) 변형에서는 주파수 쌍과 스케일 파라미터까지 학습).

- **Empirical Impact**: LRA(1K~4K 길이) 장문 분류에서 Flexformer는 기존 linear attention 대비 전반적으로 더 좋은 성능을 보이며, 평균 정확도에서 상대적으로 큰 개선을 기록한다(논문 서술 기준 best linear attention 대비 4.4% 상대 향상). 또한 문서 검색 등에서 학습 속도 향상과 메모리 감소를 달성하면서, distillation을 통해 pretrained Transformer의 softmax attention을 효율적으로 복원하고 도메인 전이(커널 transfer)도 강하게 나타났다고 보고한다.



### End-to-End Dynamic Sparsity for Resource-Adaptive LLM Inferenc (https://arxiv.org/abs/2606.27743)
- **Prior Approaches**: 기존 LLM 추론은 정적 계산 그래프를 가정해 모든 요청에 동일한 레이어·헤드를 실행하는 방식이 중심이었다. 그래서 스팟 인스턴스 선점처럼 자원이 갑자기 줄거나, 프리미엄/프리 같은 QoS 티어에 따라 예산이 달라져도 OOM으로 실패하거나 지연이 커지는 등 탄력적으로 “품질을 서서히 낮추는” 대응이 어려웠다. 정적 압축(pruning/distillation)이나 엔트로피 기반 early-exit은 한 번 정해진 구조/휴리스틱에 묶여 다양한 런타임 조건에 맞춰 재구성이 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 Learning to Allocate(L2A)라는 end-to-end 프레임워크로, 입력 난이도뿐 아니라 런타임 리소스 예산 자체를 함께 조건화해 자원-적응형 추론을 학습한다. 레이어 스킵, head pruning, reasoning-to-answer 전환(think/answer 분할)까지 예산을 입력으로 받는 게이팅 네트워크로 통합했으며, 단일 모델이 compute-accuracy Pareto frontier를 하나로 커버하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 (1) 스킵/프루닝처럼 비연속 결정을 도입하면 표현이 흔들려 성능이 무너질 수 있고, (2) 예산 변화에 맞춰 실제 연산량(레이어/헤드/FLOPs·지연·reasoning 토큰 길이)이 일관되게 제어되어야 한다는 점이다. 이를 위해 LoRA로 기존 가중치를 보존한 채 게이팅 경로에 맞춰 PEFT 적응을 수행하고, 성능(크로스엔트로피), 논리적 일관성(KL distillation), 예산(예상 compute 비용) 및 reasoning 길이(think 세그먼트 토큰 비용)를 동시에 최적화하는 단일 목적함수를 사용한다. 또한 학습 시 다양한 b를 샘플링해 게이트가 예산 레짐을 “본” 뒤, 추론에서는 학습된 게이트를 하드 임계값으로 전환해 실제 런타임 신호(데드라인·메모리 여유·큐 상황)를 b로 캘리브레이션한다.

- **Empirical Impact**: Llama-3-8B와 Qwen-3-4B에서 GSM8K 기준 dense baseline 대비 정확도 격차를 0.6% 이내로 유지하면서, 최대 34% 수준의 realized layer sparsity를 달성했다고 보고한다. 또한 OOD인 HumanEval/BBH에서도 제로샷 성능이 동일한 수준의 격차를 유지하며, 정적/휴리스틱 베이스라인은 동등한 추론 시간에서 5–10% 성능 하락하거나 예산별로 별도 튜닝이 필요했다. 결과적으로 L2A는 스팟 선점·다중 테넌트·QoS 차등 같은 변동 환경에서 “예산에 맞춰 동적으로 추론을 절약하되 논리와 품질을 지키는” 접근으로 분야에 실질적 배치 가능성을 제시했다.



### KG2Cypher: Data-Centric Pipeline for Building Enterprise Text-to-Cypher Systems (https://arxiv.org/abs/2606.27742)
Comments:
          11 pages, 2 figures, 10 tables

- **Prior Approaches**: 기존 Text-to-SQL/복잡한 KGQA 연구는 실행 가능한 쿼리(또는 logical form)로의 매핑을 벤치마크 중심으로 다뤘지만, 엔터프라이즈 KG는 비공개 스키마·엔티티 URI·리터럴 관례 때문에 그대로 적용하기 어렵다. Text-to-Cypher 쪽도 공개 데이터 보정이나 LLM 합성에 의존하는 경우가 많아, 프롬프트만으로는 실행은 되더라도 정답 그래프를 놓치는 문제가 남는다. 즉 “문법적으로 실행 가능한 Cypher”과 “올바른 엔터프라이즈 결과” 사이의 빈틈이 크다.

- **Core Contribution**: KG2Cypher는 기존 엔터프라이즈 KG 자체에서 관측된 그래프 사실로 실행 가능한 Cypher 타깃을 먼저 만들고, LLM은 자연어 생성/패러프레이즈/품질 판단 같은 언어 측 작업에만 제한해 파이프라인화했다. 이렇게 검증된 Text-Cypher 쌍을 candidate-aware SFT 데이터로 전환해, 모델이 스키마 관계와 엔티티 URI를 “선택”하도록 학습시킨다. 서빙 단계에서는 class-conditioned schema prompting, 엔티티 후보 검색, LoRA 기반 추론으로 운영 지향 구조를 갖춘다.

- **Technical Challenges**: 핵심 난제는 (1) 스키마 관계를 잘못 고르거나 (2) 비공개 엔티티 URI를 환각하며 (3) 날짜·수치·단위 같은 리터럴 sub-field를 잘못 매핑하면 실행 결과가 틀어진다는 점이다. KG2Cypher는 그래프에서 SPO 패턴을 샘플링해 실제 서브그래프에 대한 캐노니컬 Cypher를 결정적으로 구성하고, LLM이 언어만 리라이트하도록 고정함으로써 불가능한 관계/환각 URI/비실행 조건을 줄인다. 또한 후보 관계는 추론 시점에 전면 의존하지 않고, 도메인 분류로 클래스를 라우팅해 해당 클래스의 full relation schema를 프롬프트에 넣어 relation-first 검색 병목을 피한다.

- **Empirical Impact**: 한국 엔터프라이즈 KG 설정에서 prompt-only 생성은 실행은 되더라도 EM과 execution-result F1이 거의 나오지 않았고, KG2Cypher의 LoRA SFT는 방송(broadcast-program) 쿼리에서 execution-result F1을 0.806→0.950, 기업(company) 쿼리에서 0.70→0.92로 크게 끌어올렸다. 11-class class-conditioned 최종 설정에서는 EM 95.2%, 실행 성공률 99.9%, execution-result F1 0.964를 달성해 엔터프라이즈 “언어-그래프 접지”가 개선됨을 보였다. 결과적으로 실행 유효성만으로는 부족하다는 점을 실증하며, private enterprise KG 배포형 Text-to-Cypher 구축 패턴을 제시한다.



### Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation (https://arxiv.org/abs/2606.27732)
- **Prior Approaches**: 기존 discrete diffusion language model(dLLM)은 속도를 위해 병렬 복원을 시도하지만, 모델 구조에서 양방향 attention을 쓰면 KV caching이 깨져 매 denoising step마다 전면 재계산이 필요해진다. 반대로 causal attention은 KV caching은 가능하지만 right-side 컨텍스트를 잃어 품질 저하가 발생하며, 이를 완화하려던 block/hybrid 방식은 우회적으로 right 컨텍스트 범위를 제한하거나 구조가 복잡해지는 한계가 있었다.

- **Core Contribution**: 이 논문은 ‘비대칭 bidirectional context(비대칭 양방향 컨텍스트)’라는 새 패러다임을 제안해, right 컨텍스트를 attention이 아닌 별도 경로로 제공하면서도 causal의 prefix KV caching을 유지한다. 이를 R2LM(Right-to-Left Mamba)로 구현했으며, left 컨텍스트는 기존 causal backbone의 attention으로, right 컨텍스트는 reverse Mamba SSM sidecar가 압축 신호 형태로 보강하도록 설계했다.

- **Technical Challenges**: 핵심 기술적 난제는 right-side 정보를 주되 KV cache를 무너뜨리는 bidirectional attention의 구조적 의존성을 회피하는 것이다. 저자들은 backward/forward의 역할을 비대칭화하기 위해 reverse Mamba를 hooked layer에 잔차로 삽입하고, 초기에는 gate를 zero-initialized해 모델이 초기 시점에서 causal 모델과 bit-identical이 되게 만들며, 추론에서는 prefill 캐시를 그대로 재사용한 채 generation 구간에만 reverse SSM을 수행하도록 구성했다.

- **Empirical Impact**: Qwen3-1.7B를 60B 토큰으로 continued pretraining한 실험에서 R2LM은 bidirectional dLLM 대비 batch serving에서 2.4×~12.9× 높은 throughput을 보이며, autoregressive(AR) 대비로는 1.9×~2.9× 속도 개선을 보고했다. 동시에 대부분의 벤치마크에서 causal baseline을 상회하고 평균적으로도 bidirectional dLLM보다 더 좋은 결과를 보여, ‘품질-효율’ 동시 최적화 가능성을 입증했다.



### Enhancing Numerical Prediction in LLMs via Smooth MMD Alignmen (https://arxiv.org/abs/2606.27731)
- **Prior Approaches**: 기존 연구는 숫자를 일반 텍스트처럼 다루는 문제를 지적하며, 숫자 인코딩이나 토큰화 개선부터 수치값을 반영한 학습 신호 설계까지 확장해 왔다. 특히 EMD 계열처럼 값의 거리(전달 비용)를 가중해 손실을 주는 방식이 주목받았지만, 국소적으로 예측-정답 잔차가 얼마나 매끈하게 변하는지까지는 충분히 강제하지 못했다. 또한 MMD 같은 분포 정합을 숫자 예측에 직접 적용한 방법은 상대적으로 제한적이었고, 커널 설계가 성능에 미치는 영향도 명확히 정리되지 않았다.

- **Core Contribution**: 이 논문은 큰언어모델이 수치 출력에서 신뢰도 부족을 보이는 원인을 ‘값의 거리 구조를 무시하는 학습 목적’의 불일치로 보고, Smooth Maximum Mean Discrepancy(SMMD)로 이를 정렬한다. SMMD는 숫자 토큰에 대해 값-거리 기반 커널을 정의한 뒤, RKHS에서 커널 매칭으로 예측 숫자 분포를 목표(정답) 분포에 맞추고 잔차의 국소 일관성도 그래프 매끈함으로 함께 유도한다. 따라서 SMMD는 아키텍처 변경 없이 기존 cross-entropy와 결합해 수치 정밀도를 높이는 학습 신호를 제공한다.

- **Technical Challenges**: 핵심 기술 과제는 숫자 토큰을 ‘서열/거리’가 있는 값 체계로 해석하면서도 next-token autoregressive 학습 흐름과 충돌하지 않는 손실을 만드는 것이다. 저자들은 숫자 서브-보크를 구성해 조건부로 분포를 제한하고, 값 간 거리를 커널 유사도로 바꿔 MMD를 유도하되 prediction-target residual에 Dirichlet energy(그래프 라플라시안 형태)를 적용해 국소 진동을 줄인다. 또한 잔차에 smoothness를 걸어 완전 예측 시 보조항이 자연스럽게 0으로 수렴하도록 설계해 학습 목적의 정합성을 유지했다.

- **Empirical Impact**: SMMD는 수학적 추론(GSM8K, SVAMP), 산술 계산(DeepMind-Math), 시각 기반 시각/시간 인식(Clock-Time), 차트 질의응답(ChartQA)에서 다양한 open-weight LLM/VLM 백본에 걸쳐 정확도를 일관되게 개선했다. 비교 실험에서는 cross-entropy뿐 아니라 최근의 숫자 타깃 전용 손실(Gaussian Cross Entropy, NTL, NTIL) 대비 성능이 자주 우세했으며, 분석 결과로는 MMD 정합과 smoothness 정규화가 보완적으로 작동함이 확인됐다. 특히 산술·시간 과제에서 큰 오차를 줄이고(예: Time Gap 감소) 분포의 목표 정렬이 더 날카로워지는 경향이 관찰되어, 수치 예측의 안정성과 일반화에 의미 있는 기여를 한 것으로 평가된다.



### Do Speech Emphasis Models Generalize across Languages and Emotions? (https://arxiv.org/abs/2606.27717)
Comments:
          Interspeech 2026

- **Prior Approaches**: 기존 강세(강조·prominence) 탐지는 주로 영어의 중립 읽기 음성(단일 언어, 제한된 화법) 위주로 학습·평가돼 다국어·감정 표현으로의 일반화가 불명확했다. 일부 연구는 합성 TTS에 처방된 강세 라벨을 쓰거나(라벨이 청취가 아니라 스크립트/LLM에서 생성), 다른 연구는 합성 음성·LLM 라벨로 스트레스 헤드를 학습하는 등 사람의 지각 기반, 감정 범위의 폭이 제한되는 경우가 많았다.

- **Core Contribution**: 이 논문은 MMEE(Multilingual Multi-Emotion Emphasis)라는 대규모 다국어·다감정 말뭉치를 제안해, 7개 매크로언어·10개 지역 변종의 34개 감정/화법 범주에서 10,000개 발화를 3단계(미강조/강조/중강조) 단어 수준 지각 라벨로 수집했다. 또한 EmphaClass와 WhiStress 두 모델을 여러 전이 설정(단일언어, 교차언어, 다국어, 교차-감정, 교차-데이터셋, 데이터 스케일)으로 체계 벤치마킹해 강세 표현의 보편성과 균열 지점을 함께 분석했다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 언어/감정에 따라 강세의 음향 단서가 달라지는 상황에서, 모델이 언어 특이 신호에 과적합되지 않게 학습하는 것과 (2) 합성 라벨과 인간 지각 라벨이 서로 다른데도 공통 표현을 학습하는지 검증하는 것이었다. 연구진은 XLS-R 기반 프레임 분류를 스칼라 회귀로 확장하고, Whisper 기반 WhiStress는 언어 조건 토큰을 활용해 다국어 처리를 지원하는 방식으로 두 모델을 동일한 분할·평가 틀에서 비교했다.

- **Empirical Impact**: 실험 결과 단일언어 학습 모델은 자국 내에서는 강하지만, 계통적으로 먼 언어로 갈수록 zero-shot 교차언어 전이가 급격히 저하됐다(특히 Mandarin이 지속적으로 약함). 반면 다국어 풀링 학습은 교차언어 강건성을 크게 높였고, 데이터 규모를 늘리면 초반 몇 천 샘플 구간에서 성능 이득이 집중되는 경향을 보였다. 또한 high/low arousal 사이에서도 강세 탐지 성능이 비교적 견고했으며, 인간 지각 기반 벤치마크와 합성 처방 기반 벤치마크 사이에 양방향 전이가 관찰돼 강세 신호의 일부는 라벨 패러다임·데이터 출처에 비교적 잘 유지됨을 시사한다.



### The Simulacrum: Decision-Theoretic Pretraining for Near-Optimal Time-Series Forecasting and Inferenc (https://arxiv.org/abs/2606.27711)
- **Prior Approaches**: 전통적인 구조적 시계열 모델은 해석 가능성과 기작 검증에 강점이 있지만, 핵심 한계는 가정의 옳고 그름보다 “추정이 데이터에서 안정적·정확하게 되는가”에 달려 있다. 특히 유한 표본에서의 편향, 수치 불안정, 그리고 예측구간의 miscalibration(보정 불일치)이 예측과 추론 전반을 흔든다. 한편 TSFMs·GFMs 같은 딥러닝 포캐스팅은 정확도는 잘 내지만, 과정(process) 단위의 유한표본 보장(예: 편향 제어, 균일 보정)이나 특정 구조의 통제 목표를 학습 목적에 직접 반영하기 어렵다는 문제가 있다.

- **Core Contribution**: 논문은 시계열 추정을 “통계적 의사결정 문제”로 재정의하고, 분석가가 생성 환경(generative world)과 의사결정 목적(objective)을 함께 명시하면 신경망이 그에 대한 최적 의사결정 규칙을 근사하도록 학습하는 Simulacrum 프레임워크를 제안한다. 이때 생성 환경은 ARIMA·ETS·GARCH 같은 모델뿐 아니라 오염/잡음/데이터 증강과 같은 현실적 교란까지 포함할 수 있으며, 목적은 예측 정확도뿐 아니라 near-optimal risk, bias control, minimax 성능, uniform calibration 같은 과정 수준의 성질을 직접 겨냥한다. 결과적으로 추정기(estimator)는 이전에 보지 못한 시계열에 대해 zero-shot으로 예측, 파라미터 추정, 예측구간, 모델 선택까지 수행할 수 있다.

- **Technical Challenges**: 핵심 기술 난제는 ‘분석가가 원하는 과정 수준의 통계적 성질’을 미분 가능한 학습 목표로 어떻게 바꿔 신경망에 전달하느냐에 있다. 이를 위해 논문은 2단계 stratified simulation을 도입하는데, 바깥 루프에서 ω(모델 파라미터/구성)를 샘플링하고, 안쪽 루프에서 그 고정된 ω에 대해 여러 replicate trajectory를 만들어 conditional bias나 miscoverage 같은 과정-조건 성질을 측정·페널티화한다. 또한 평균 제약 대신(또는 완화해) 편향·보정을 안정적으로 최적화하기 위한 래그랑주/집계 전략을 쓰고, 모델 불일치 문제를 줄이기 위해 bias-correction head 같은 확장도 제시한다.

- **Empirical Impact**: 실험에서는 exponential smoothing과 AR(pp) 같은 구조적 세계에서 neural estimator가 최대우도(MLE) 및 AICc 기반 모델선택 등 전통적 기준선보다 우수함을 보이며, 특히 유한표본 편향 및 예측구간 miscalibration을 줄이는 데 강점이 나타난다. AR(pp)에서는 conditional on each parameter value 수준의 uniform calibration에 더 가까운 구간 보정을 달성하고, Forecast Combination Puzzle도 ‘예측 가중 평균’을 구조적 생성 과정으로 모델링해 휴리스틱을 원칙화한 규칙을 학습함으로써 해결한다. 더 나아가 실제 벤치마크(M1, Monash archive 등)에서 시뮬레이션으로만 학습된 신경 추정기가 전통 통계 모델과 현대 신경/대형 pre-trained 모델 대비 경쟁력 있거나 경우에 따라 state-of-the-art 수준의 예측 정확도를 보였다.



### Low-Agreeableness Persona Conditioning for Safe LLM Fine-Tuning (https://arxiv.org/abs/2606.27709)
Comments:
          9 pages, 8 tables, 5 figures

- **Prior Approaches**: 기존 연구는 공감·따뜻함(empathetic warmth) 데이터로 fine-tuning할 때 사실 정확성이 떨어지고 sycophancy가 커지며, 그 과정이 adversarial safety까지 약화될 수 있음을 보여주었습니다. 또 일부 방법은 데이터에서 위해 신호를 필터링하거나 harm 라벨/검출기를 쓰는데, 이는 비용이 크거나 가용 데이터에 제약이 있습니다. 한편 대화의 ‘감정’ 자체보다 담화 구조가 안전에 영향을 준다는 관찰도 있어, 따뜻함 데이터가 항상 동일한 안전 비용을 만드는지 재검증이 필요해졌습니다.

- **Core Contribution**: 이 논문은 따뜻한 미세조정이 안전을 해치는 원인이 ‘공감이라는 목표 자체의 필연적 부작용’인지, 아니면 ‘데이터 구성의 산물’인지 분해해 보려 합니다. 핵심은 persona-driven rewriting pipeline로 사용자 턴을 Big Five의 low agreeableness(낮은 동조성) 성향으로 만들고, 대신 어시스턴트 응답은 warm 하면서도 de-escalating(격화 완화)하도록 함께 재작성하는 데이터 설계입니다. 안전 라벨, harm detector, 학습 목적 변경 없이도 jailbreak 취약성과 유해 출력률을 일반적인 warmth fine-tuning 대비 낮출 수 있음을 보입니다.

- **Technical Challenges**: 문제는 따뜻함을 유지하면서도 ‘warmth와 compliance(순응) 방향이 잠재공간에서 함께 정렬되는’ 공변 관계를 끊는 데이터 신호를 설계하는 데 있습니다. 저자들은 먼저 Llama-3.1-8B에서 잔차(residual) 섭동으로 안전에 민감한 레이어를 찾고, Big Five 하위집합 중 low agreeableness가 가장 유리하다는 점을 pilot로 고른 뒤 다른 모델에도 적용해 일반화를 검증합니다. 또한 ablation을 통해 user-side만 low agreeableness로 바꾸면 오히려 per-token warmth가 손상되며, de-escalating 어시스턴트 rewrite가 그 균형을 회복한다는 점을 분리해 보여줍니다.

- **Empirical Impact**: 세 가지 실험(4개 모델)에서 제안한 full paired condition은 generic warmth fine-tuning baselines보다 jailbreak 성공률과 red-teaming 기반 harmful output 지표에서 더 일관되게 개선을 보였습니다. 특히 MentalChat-16K 기반 정신건강 지원 도메인에서도 동일한 경향이 유지되었고, per-token warmth는 기준선 대비 유지 또는 개선되어 ‘더 차가워진 탓’의 설명을 약화시킵니다. representational probing에서는 따뜻함과 compliance 간 기하학적 정렬이 완화되는 신호(웜스-컴플라이언스 decoupling)가 관측되어, 안전 개선이 데이터 설계에서 비롯된 메커니즘과 정합적이라는 근거를 제공합니다.



### Room for Error: Large-Scale Simulation of Over-the-Air Acoustic Attacks (https://arxiv.org/abs/2606.27701)
Comments:
          20 pages

- **Prior Approaches**: 음성 제어와 ASR이 확산되면서 위협도 커졌지만, 기존 연구는 디지털 오디오 기반의 적대적 예시를 물리 세계로 확장하는 데 따르는 스케일 장벽을 충분히 다루지 못했다. OTA 공격이 제안되더라도 보통 단일 음향 환경의 손실 매핑을 가정해 재현성과 일반성에서 한계를 보이며, SNR 같은 은폐/탐지 지표도 측정 위치가 불명확하고 기하 정보는 과도하게 단순화되는 경우가 많다.

- **Core Contribution**: 이 논문은 ASR 음향 공격의 위험을 제대로 이해하기 위해, 공격자가 갖는 환경 정보의 정도를 연속 스펙트럼으로 정리하는 Knowledge Gradient를 제안한다. 또한 Dual-Form Signal-to-Noise Ratio(이중형 SNR)로 ‘공격 은폐(stealth)’와 ‘피해 모델 공격 효능’을 분리해, 기존 단일 스칼라 지표가 숨기던 에너지(투사) 비용을 operationalize한다.

- **Technical Challenges**: 핵심 난제는 물리 전파에서 SNR이 한 위치의 값이 아니라 소스-공간-피해자 전반에 걸친 상대적 모드로 나타나며, 공격자가 실제로는 방의 RIR을 완벽히 알기 어렵다는 점이다. 이를 위해 고충실도 물리 시뮬레이션 대신 Image Source Method 기반의 acoustically-aligned 고속 현실(음향) 시뮬레이션 프레임워크로 대규모 환경 분포를 생성하고, SNR 제약을 수치 최적화 내부에서 효율적으로 다루기 위한 에너지 제약(Projection Cost 관점) 및 제약 완화를 적용해 평가 가능성을 확보했다.

- **Empirical Impact**: 이들은 8백만 건 이상의 adversarial evaluation을 포함하는 대규모 실험으로, acoustic awareness를 반영할 때 Whisper와 wav2vec에서 상대 WER 증가가 최대 94.5%까지 커질 수 있음을 보였다. 더불어 단일 환경 편향이 만들어내는 상관 붕괴와 baseline 환경 열화가 큰 영향을 준다는 점을 확인했으며, ‘음향 환경을 추상화하지 않고 받아들이는’ 반복·검증 가능한 연구 인프라를 제공해 AML/음성 보안 평가의 표준화 논의에 직접적인 근거를 마련했다.



### What Was That Again? Certified Robustness for Automatic Speech Recognition (https://arxiv.org/abs/2606.27698)
Comments:
          17 pages

- **Prior Approaches**: 기존 ASR certified robustness 연구는 분류기 형태의 Certified Robustness를 시퀀스 출력으로 확장하려 했지만, 노이즈가 커질수록 특정 전사(transcription)에 대한 확률 질량이 붕괴해 연산·통계가 불안정해지기 쉽습니다. Randomised Smoothing은 프레임워크는 간단하지만 시퀀스 공간의 조합 폭발 때문에 “어떤 문장이 이길지”를 안정적으로 보장하기 어렵습니다. 또한 sequence alignment 기반 인증은 저 SNR/적대적 상황에서 정렬이 자주 깨지며, 슬롯 수가 급증해 통계 예산이 과도하게 분산되면서 WER 상승과 무의미한(vacuous) 인증으로 이어질 수 있습니다.

- **Core Contribution**: 이 논문은 E-value와 anytime-valid 성질을 활용해, 시퀀스-to-시퀀스 과제에서 토큰 존재 여부와 적대적 환각의 배제를 동시에 다루는 “dual-gate” 인증 파이프라인을 제안합니다. Two-Sided Atomic Audit로 안전 토큰의 존재를 증명하고 adversarial exclusion으로 위험 토큰을 배제한 뒤, Rank-Based Sentence Certification을 위한 E-value tournament로 최종 시퀀스를 선택합니다. 정렬(alignment) 없이 원자(토큰) 수준과 구조(문장 배열) 수준을 연결해, 고노이즈에서도 안정적인 recall과 안전 반경을 제공하는 것이 핵심입니다.

- **Technical Challenges**: 가장 큰 난관은 (1) 시퀀스 공간에서 개별 전사의 확률이 무너지는 문제와 (2) 인증이 샘플링을 “언제 멈출지”에 따라 깨지지 않게 만드는 anytime-valid 제약이었습니다. 이를 위해 Ville’s inequality 기반의 E-value 누적(확률-검정 관점의 wealth process)을 사용해 peeking 문제 없이 온라인으로도 유효한 안전 반경을 산출하고, 토큰에 대해 two-sided로 존재/부재 양쪽을 동시에 감사합니다. 이후 atomic gate로 후보 어휘·후보 전사를 강하게 정제한 뒤 E-value tournament로 최종 순위를 결정해 정렬 실패로 인한 슬롯 폭발 같은 구조 붕괴를 완화합니다.

- **Empirical Impact**: 네 가지 서로 다른 ASR 아키텍처에 대한 평가에서 Word Error Rate(WER)는 최대 55% 상대 감소를 보이며, confidence와 WER 간 Spearman 상관을 낮춰 “신뢰도 기반 오판 위험”을 줄이는 효과가 확인됩니다. 또한 단순히 가시적 출력 품질뿐 아니라 단어-및 문장 수준에서의 granular certification(어떤 단어가 안전하게 포함/배제되는지, 구조적으로 어떤 전사가 선택되는지)을 제공해 음향 보안 감사를 촘촘히 할 수 있습니다. 저노이즈가 아닌 고노이즈 구간에서도 인증 recall이 약 40.5%–90.3% 범위로 유지되는 점이, 기존 smoothing/정렬 기반 접근이 무너지는 상황에서 의미 있는 차별점으로 제시됩니다.



### Class-frequency Guided Noise Schedule for Diffusion Models (https://arxiv.org/abs/2606.27696)
Comments:
          technical report

- **Prior Approaches**: 확률 점수(score)를 학습하는 score-based generative model은 multi-scale noise schedule로 저밀도 영역에서의 부정확한 점수 추정을 완화해왔다. 다만 long-tailed처럼 클래스 빈도(모수 수)가 다른 데이터에서는 저빈도 클래스가 더 큰 저밀도 영역을 만나고, 점수 공간에서도 고빈도 클래스가 우세해 생성 품질과 다양성이 떨어질 수 있다는 문제가 남아 있었다.

- **Core Contribution**: 이 논문은 클래스 빈도와 multi-scale noise schedule 사이의 상관관계를 처음으로 체계적으로 분석한다. 그 결과 저빈도 클래스에는 더 큰 스케일 노이즈가 필요하다는 직관을 바탕으로, Class-frequency Guided(CFRG) noise schedule을 제안한다.

- **Technical Challenges**: 핵심 기술적 난점은 저빈도 클래스에서 발생하는 더 큰 저밀도 영역이 점수 추정 오류로 이어지는 메커니즘을 노이즈 스케줄 설계로 연결하는 것이다. 논문은 노이즈 스케일을 클래스 빈도에 역비례하도록 클래스별로 조정하는 CFRG를 통해 저밀도 영역을 줄이고, 점수 공간의 불균형(고빈도 클래스 쏠림)을 완화하도록 설계했다.

- **Empirical Impact**: CIFAR-100-LT와 ImageNet-LT의 imbalanced 데이터에서 이미지 생성(FID), 이미지 분류(생성 데이터 활용 시 top-1 정확도), 텍스트-투-이미지 생성까지 폭넓게 실험해 CFRG의 개선을 확인했다. 예컨대 CIFAR-100-LT는 FID에서 DDPM 대비 큰 폭으로 향상(2.24 개선), CIFAR-100-LT 분류도 top-1 정확도 9.22%p 개선을 보였으며, frequency 통계가 noise schedule 설계에 “결정적”임을 실증적으로 뒷받침한다.



### Halt Fast! Early Stopping for Certified Robustness (https://arxiv.org/abs/2606.27694)
Comments:
          24 pages

- **Prior Approaches**: Randomized Smoothing(RS)은 어떤 신경망에도 구조 변경 없이 적용 가능한 대신, 입력마다 수만 번의 모델 평가를 요구해 실시간 인증이 어렵다는 한계가 컸다. 또한 기존 anytime-valid 접근인 E-value는 주로 이분 가설(r≥c 여부 등)에 머물러, 샘플 간 상대적 위험을 세밀하게 다루기 어렵고 인증 프로세스를 ‘상황에 맞게’ 종료하기도 까다로웠다.

- **Core Contribution**: 이 논문은 anytime-valid certified robustness를 “메타러닝으로 샘플별 prior를 예측해 순차 E-process에 적응적으로 자원을 배분”하는 형태로 확장한다. 구체적으로 연속 반경(continuous radius) 추정을 위한 mixture 기반 다중 가설을 도입하고, 입력별로 혼합 prior를 예측하는 Sample-Adaptive Meta-Learning을 제안한다.

- **Technical Challenges**: 핵심 난제는 meta-learner가 prior를 데이터(Phase I glimpse) 기반으로 의존하도록 하면서도, peeking 문제 없이 anytime-valid 통계 보증을 깨지 않는 E-value 구성을 유지하는 것이다. 이를 위해 Phase I에서 얻은 N_sel 샘플을 wealth 축적에 사용하지 않고(버림), 베타 혼합 prior를 예측한 뒤 truncated/동적 support 설계를 통해 효율적인 베팅(wealth 증가)을 유도하며, 필요 시 bankruptcy exit 및 safety anchor로 보수성을 유지한다.

- **Empirical Impact**: 실험에서는 전통적 방식 대비 샘플 복잡도를 약 20배 줄였고, 고속 인증이 가능함을 보여 500 샘플 미만에서도 인증을 구성할 수 있음을 보고한다. 무엇보다 anytime-valid의 장점인 ‘위험 임계값에 따라 compute를 골라 쓰는 resource triage’가 가능해져, 기존 고정-샘플 인증 프레임워크에서는 어려웠던 안전 민감(실시간/실행 경로 분기) 배치 시나리오로의 전환 가능성을 시사한다.



### Deployment-Side Adaptiveness in Multi-Horizon Volatility Forecasting (https://arxiv.org/abs/2606.27688)
Comments:
          Accepted for KDD 2026 Machine Learning in Finance Workshop

- **Prior Approaches**: 기존 금융 시계열 예측 평가는 학습된 모델의 성능에 초점을 두고, 멀티-하orizon 설정에서도 기본 배치(예: MIMO의 기본 롤아웃)를 그대로 배포하는 경우가 많았다. 멀티-스텝 예측에서 recursive/direct/hybrid 같은 전략 차이가 오류를 바꾼다는 연구는 있었지만, 별도 모델을 다시 학습한 경우가 주류였다.

- **Core Contribution**: 이 논문은 학습이 끝난 MIMO 멀티-출력 예측기가 하나의 고정된 배포기가 아니라, inference-time 롤아웃 규칙에 따라 서로 다른 예측기(예측 성능-비용 프로파일)를 만들어내는 ‘배포 가능 예측기들의 패밀리’를 형성한다고 정식화한다. 또한 그 패밀리 안에서 검증 기반 선택/부분 앙상블로 예산을 만족하는 운영 정책을 설계하는 프레임을 제안한다.

- **Technical Challenges**: 핵심 난제는 “같은 학습 파라미터라도 롤아웃 규칙이 바꾸면 실제 배포 시점의 유효 예측 문제와 자기유도 상태가 달라져 성능이 크게 요동칠 수 있다”는 점이며, 따라서 고정된 대체 규칙을 찾기 어렵다는 것이다. 논문은 출력 블록 크기(롤아웃 ratio)를 바꿔 induced rule family를 만든 뒤, MSE 기준으로 검증에서 정책을 선택하고 비용(추론 시간)까지 함께 비교하는 방식으로 이 문제를 해결한다.

- **Empirical Impact**: VOLARE의 20개 종목 변동성 시계열에서 LinearNet~PatchTST까지 20일 단위 멀티-하orizon(10/20/30) 실험을 수행한 결과, 비기본 롤아웃 싱글톤이 MIMO 기본 배치보다 자주 좋아지지만 최선의 규칙은 아키텍처·하orizon에 따라 크게 달랐다. MSE 기준 검증 선택은 기본 대비 저비용 개선을 주고, 작은 규칙 부분집합은 큰 앙상블의 이점을 상당 부분 회수하면서도 추론 비용을 크게 줄였으며, 동시에 MSE로 선택한 정책의 순위가 QLIKE에서는 일관되게 전이되지 않았다. 결론적으로 변동성 예측에서 배포 정책(inference-time deployment)은 학습 아키텍처만큼 중요한 적응성 소스이며, ‘한 번 학습 후 induced rule pool을 가볍게 검증·선택’하는 운영 절차가 실용적이라고 보여준다.



### Mitigating LLM-based p-Hacking by Preregistering for the Next LLM (https://arxiv.org/abs/2606.27687)
- **Prior Approaches**: 기존 LLM 기반 연구는 데이터 생성·분류·주석을 수행한 뒤 그 결과를 downstream 가설검정에 활용하는 경우가 많다. 하지만 프롬프트, 디코딩 파라미터, 출력 포맷을 반복적으로 조정하면 원하는 성과가 나오도록 “p-hacking”이 쉽게 발생할 수 있다.

- **Core Contribution**: 논문은 LLM 연구에서 p-hacking을 줄이기 위한 프로토콜을 제안한다. 실험을 사전등록(preregistration)하고, 특정 조건을 만족하는 eligible 모델 집합을 정한 뒤 사전등록 이후 처음 공개되는 eligible LLM에 대해 확증분석(confirmatory analysis)을 수행한다.

- **Technical Challenges**: 핵심은 어떤 설정을 골랐는지가 사전등록 시점에는 아직 존재하지 않을 ‘다음 모델’에 그대로 적용되지 않게 설계하는 것이다. 저자는 우선 기존 모델에서는 절차를 확정하고, 분석 플랜과 향후 eligible 모델 목록을 함께 preregister한 뒤, 첫 공개 모델에서만 결과를 확정하도록 실행 흐름을 고정한다.

- **Empirical Impact**: 두 가지 태스크(정답이 알려진 조건)에서 20개 모델, 11개 LLM-analysis 구성으로 평가한 결과, 프로토콜은 p-hack의 성공적 전이를 각각 73.9%, 72.7%에서 차단하는 것으로 나타났다. 또한 여러 stress test에서도 완화 효과가 유지됐고, 실제로 저자들이 동일 프로토콜을 적용한 사전등록 실험에서는 기존 모델을 “해킹”했던 7개 구성 중 6개에서 다음 eligible 모델로의 전이가 실패해 유효성이 확인됐다.



### CBD: API-Only LLM Black-Box Unlearning through Controlled Behavioral Divergenc (https://arxiv.org/abs/2606.27683)
- **Prior Approaches**: 기존 LLM unlearning은 (1) 학습 파라미터를 직접 수정하는 white-box 방식과 (2) 보조 모델/로그잇을 이용해 타깃 분포를 교정하는 gray-box 방식으로 나뉜다. 하지만 API로만 제공되는 edge 서비스에서는 모델 파라미터·로그잇·내부 확률을 볼 수 없어서, 기존 방식은 그대로 적용하기 어렵다. 또한 forget과 retain이 유사한 프롬프트 구조를 공유하면 unlearning 신호가 공유 패턴까지 건드려 retained utility가 쉽게 떨어진다는 한계가 남는다.

- **Core Contribution**: 이 논문은 API-only black-box unlearning 프레임워크인 Controlled Behavioral Divergence (CBD)를 제안한다. CBD는 고정된 reference 모델과 학습 가능한 probe 모델의 행동(출력 분포) 차이를 이용해, 입력이 unlearning 대상과 관련됐는지 relevance score로 추정한 뒤 해당 질의를 target LLM이 아닌 reference 경로로 라우팅한다. 더불어 유사한 프롬프트에서 “공유 구조”가 아니라 “타깃-특이 정보”에만 반응하도록 판별 기반을 설계한다.

- **Technical Challenges**: 핵심 난제는 (a) black-box 환경에서 타깃 모델 내부 로그잇/그래디언트를 쓰지 않고도 unlearning 효과를 만들어야 한다는 점과 (b) forget·retain의 의미적/구조적 유사성이 높을 때 라우팅이 흔들리지 않게 해야 한다는 점이다. CBD는 probe를 LoRA 어댑터로만 학습시키고, retain 입력에서는 reference와 가까워지게(align), forget 입력에서는 분리되게(separate) “controlled behavioral divergence”를 강제한다. 여기에 retain 전용으로 Fisher 기반의 판별적 basis를 추출해, unlearning 그라디언트가 공유 프롬프트 템플릿 방향이 아니라 타깃 특이 방향으로 가도록 정규화된 generalized eigenvalue 해를 통해 업데이트를 제한한다.

- **Empirical Impact**: 실험에서 CBD는 11개 white-box/gray-box 기준선과 비교해 unlearning-유틸리티 trade-off가 더 좋고, 설정 변화에도 성능 변동이 작게 나타난다. ToFU forget10에서 CBD는 retrained reference에 forget 세트 성능을 가깝게 맞추면서 retained 유틸리티를 74.90까지 유지해, 두 번째 최선 대비 약 15% 높은 결과를 보였다. WMDP에서는 유해 지식 정확도를 25.68로 낮춰 거의 무작위 수준에 가깝게 만들면서 MMLU는 52.67을 유지해, 안전성 제거와 일반 능력 보존을 동시에 달성했다.



### From Signals to Transfer: A Factorised Study of Probe-Based Uncertainty Estimation in Large Language Models (https://arxiv.org/abs/2606.27679)
- **Prior Approaches**: 프로브 기반 uncertainty estimation(UE)은 LLM 내부 신호에서 불확실성을 학습해 환각을 탐지하는 방식으로 주목받아 왔습니다. 다만 기존 연구는 feature 설계, 학습 데이터/라벨 구성, 프롬프트·평가 설정을 동시에 바꾸는 경우가 많아 “무엇이 성능을 올리는지”가 흐려졌습니다. 또한 매치된 벤치마크에서는 잘 동작해도, 다른 도메인·생성 형식으로 옮기면 성능이 크게 떨어지는 일반화 한계가 지적됩니다.

- **Core Contribution**: 논문은 프로브 기반 UE를 feature 표현, 데이터/라벨 구성, 전이(transfer) 설정으로 분해해 요인이 성능을 어떻게 바꾸는지 맞춤 조건에서 분석합니다. 그 결과, 단순 raw hidden states나 attention feature는 in-domain에서는 경쟁력이 있으나 분포 이동에서는 구조화/압축된 feature가 더 견고하다고 제시합니다. 또한 prompting 방식(추론 유도)과 자동 라벨 구성(lexical vs LLM-as-a-judge)이 프로브 동작을 크게 좌우함을 정리해 재사용 가능한 pretrained factuality probe의 기준선을 제안합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 (1) 내부 신호·라벨·프롬프트를 동시에 건드리지 않고, 성능 변화의 원인을 요인별로 분리하는 설계와 (2) 자동 라벨이 정답 의미를 얼마나 충실히 반영하는지 검증하는 것입니다. 저자들은 AUROC와 ECE로 분별력과 캘리브레이션을 함께 보고, terse/long reasoning 프롬프트와 Rouge·AlignScore·LLM-as-a-judge 라벨링을 비교해 lexical 매칭 기반 라벨의 왜곡 가능성을 확인합니다. 전이 일반화를 위해서는 Lookback Ratio, Layer Top-mm Prob., Internal Variance 등 더 구조화된 feature와 비교적 단순한 linear probe를 우선시하는 “best practices”를 도출합니다.

- **Empirical Impact**: 여러 벤치마크(QA/verification/MCQ)에서 in-domain 성능만으로는 발전을 판단하기 어렵고, OOD(동일 태스크·교차 태스크)에서의 견고성이 핵심임을 실증했습니다. 특히 reasoning을 길게 유도하면 LLM 정확도는 유지돼도 프로브의 AUROC가 떨어지는 경향이 나타나, 배포 관점에서 입력/출력 포맷 통제의 중요성이 강조됩니다. 또한 벤치마크에서 사전학습한 benchmark-pretrained probe가 타깃 도메인 라벨 없이도 open-ended long-form factual generation에 “stable off-the-shelf baseline” 수준으로 전이되며, raw last-token embedding의 전이 취약성과 구조화 feature의 우수한 견고함을 재확인합니다.



### Explainable AI for Biodiversity Monitoring and Ecological Image Analysis (https://arxiv.org/abs/2606.27667)
- **Prior Approaches**: 카메라 트랩·드론·위성·수중 플랫폼 등에서 얻는 생태 이미지에 대해 영상 분류·탐지·분할 모델을 자동화해 보전 평가의 규모와 속도를 키우려는 시도가 늘고 있다. 다만 기존 모델은 왜 맞혔는지 해석이 어렵고, 표본 편향·배경/형상 교란·우연 상관 같은 요인이 예측을 좌우해도 이를 검증하기가 힘들다. 그 결과 보전 의사결정에 쓰일 때 모델 근거가 생태적으로 타당한지 확인이 부족하다는 문제가 지적된다.

- **Core Contribution**: 이 논문은 explainable artificial intelligence(XAI)를 생태 모델 검증의 표준 구성요소로 두어, ‘정확도’뿐 아니라 ‘정확한 이유’를 생태 관점에서 확인하도록 제안한다. 또한 생태 컴퓨터 비전의 대표 3개 과제(영상 분류, object detection, image segmentation)에 XAI를 실무적으로 적용하는 가이드를 제공한다. 공중(항공) 이미지 사례를 통해 XAI가 모델 감사(auditing), 개선(refinement), 배포(deployment)로 이어지는 흐름을 구체화한다.

- **Technical Challenges**: XAI를 생태 데이터에 적용할 때는 설명이 실제로는 배경·윤곽·가림(occlusion)·경계(edge) 효과 등 ‘비생태적 신호’를 강조하는지 구분해야 한다는 기술적 난제가 있다. 논문은 이를 위해 과제별(분류/탐지/분할)로 설명 방법을 적용해 생물학적으로 의미 있는 단서가 포착되는지 점검하고, false positive가 어떤 교란에서 비롯되는지 식별하도록 한다. 더 나아가 설명 결과를 바탕으로 데이터 수집, 데이터 증강, 재학습 전략을 수정해 모델의 추론 정렬(alignment)을 유도한다.

- **Empirical Impact**: 항만물범(harbor seal) 탐지와 고래류(cetacean) 해부학 분할이라는 두 사례에서 XAI가 생물학적 단서를 확인하고 배경·형상 confound로 인한 오탐을 드러내는 효과를 보여준다. 또한 edge와 occlusion에 따른 성능/설명의 편향을 관찰해 추가 데이터와 학습 설계를 어떻게 바꿀지 연결한다. 전반적으로 모델 추론이 생태적 이해와 맞는지 과학적으로 점검하는 도구로서, 생물다양성 보전에 쓰이는 AI 증거의 신뢰성과 실행 가능성을 높이는 데 의미가 크다.



### Reconstructing the Developmental Trajectory of Adipocytes in Human Adipose Tissue Using Single-Cell RNA Sequencing (https://arxiv.org/abs/2606.27657)
Comments:
          20 pages, 10 Figures, The manuscript is currently under review at the International Journal on Electrical Engineering and Informatics

- **Prior Approaches**: 기존 연구는 비만과 대사질환의 연관성을 보여주더라도, 사람 지방세포가 분화하며 형성되는 과정을 전사 수준에서 연속적으로 추적하는 데 한계가 있었다. 또한 분화 단계별 세포 간 통신과 신호전달 경로의 시간적 변화, 부위(지방 저장소)별 차이를 통합해 지도화한 사례가 드물었다.

- **Core Contribution**: 이 논문은 단일세포 RNA 시퀀싱을 활용해 사람 지방세포의 발달 궤적(developmental trajectory)을 재구성하고, 분화 과정의 전사적 상태를 15개 클러스터(전이 상태 7개)로 정리했다. 더 나아가 지방세포와 전구세포 사이에서 작동하는 기능성 신호전달 경로를 체계적으로 포착해 IGF와 FGF가 핵심 네트워크임을 강조한다.

- **Technical Challenges**: 핵심 과제는 단일세포 데이터에서 분화 연속성을 반영하는 전이 상태를 안정적으로 구분하고, 세포 간 통신용 신호 경로를 기능적으로 추론하는 것이었다. 연구진은 발달 궤적 분석으로 전사적 상태를 분해한 뒤, 신호전달 경로의 활동성을 단계별로 비교해 IGF·FGF가 분화 전 과정에서 일관되게 두드러짐(p<0.05)을 확인했으며, 저장소별로는 내장 지방이 추가적인 세포외기질 리모델링을 보인다는 차이를 찾아냈다.

- **Empirical Impact**: 분석 결과는 IGF가 혈관 주변(perivascular) 니치에서 특히 활발하고, FGF는 성숙 지방세포 구역에서 우세하다는 공간적 패턴까지 제시하며, 사람 지방 발달의 포괄적 지도라는 점에서 의미가 크다. 이 데이터는 건강한 지방 확장을 유도하거나 병적 지방 축적을 억제하는 치료 전략의 후보 표적(IGF/FGF)을 제공해 대사질환 치료 연구에 임상적으로도 연결될 수 있는 근거가 된다.



### Cross-Platform Chinese Offensive Comment Detection via Dual-Threshold Hard Example Mining (https://arxiv.org/abs/2606.27629)
Comments:
          10 pages, 7 figures

- **Prior Approaches**: 중국 욕설·혐오 댓글 탐지는 기존에 욕설 사전 매칭, TF-IDF·n-gram 같은 얕은 특징, 또는 CNN/RNN 기반 모델에 의존해 왔습니다. 그러나 이런 방식은 아이러니·은유·동음이의 농담처럼 명시적 단어가 없는 암묵적 공격을 잘 잡지 못한다는 한계가 컸습니다. 또한 COLD 같은 벤치마크가 있어도 단일 플랫폼·정적 분포에 머물러 실제 크로스플랫폼 도메인 shift를 체계적으로 진단하고 대응하기는 부족했습니다.

- **Core Contribution**: 이 논문은 COLD 기반 RoBERTa 이진 기준선을 공정 비교를 위해 먼저 구축하고, 위에·샤오홍슈·티에바·즈후 네 플랫폼을 아우르는 3클래스(정상/명시/암묵) 크로스플랫폼 평가 세트를 구성합니다. 이어 도메인 거리(Jaccard, Proxy-A Distance)로 플랫폼 간 간극을 정량화하면서, 기준선 성능 저하의 핵심 병목이 암묵 공격 인식 약화와 플랫폼 전용 용어 부재에 있음을 드러냅니다. 그 위에 dual-threshold hard mining과 소량 라벨로 암묵 맥락을 보정하는 lightweight domain adaptation을 제안합니다.

- **Technical Challenges**: 핵심 기술적 난제는 라벨이 거의 없는 타 플랫폼에서 기준선의 실패 지점을 효율적으로 샘플링하고, 암묵적 공격의 문맥 의존성을 적은 비용으로 보정하는 것입니다. 논문은 baseline의 Softmax confidence를 기준으로 high-confidence 오분류(겉보기는 정상처럼 보여도 사실은 암묵 공격)와 low-confidence 오분류(플랫폼 슬랭/신조어로 경계가 흔들림) 두 풀로 나눠 hard example을 선별하고, 각 플랫폼에서 100개씩 총 400개만 수동 확인해 2차 미세조정을 수행합니다. 이를 통해 large-scale 데이터 증강 없이도 모델의 의사결정 캘리브레이션을 노립니다.

- **Empirical Impact**: 실험 결과 최적화된 모델은 네 플랫폼 모두에서 성능(특히 F1과 유해 샘플 recall)을 유의미하게 개선하며, 샤오홍슈처럼 COLD에 없는 도메인에서 이득이 가장 크게 나타납니다. 암묵 공격에서의 false negative가 줄고, 플랫폼 전용 표현으로 인한 오분류도 완화되는 패턴이 관찰됩니다. ablation에서도 동일 라벨 예산 하에 random sampling보다 hard example mining이 더 효과적임을 보여, ‘양’보다 ‘어려운 샘플’이 미세조정 효율을 좌우한다는 점을 실증적으로 뒷받침합니다.



### HybridCodec: Modeling Discrete and Continuous Representations for Efficient Speech Language Models (https://arxiv.org/abs/2606.27627)
Comments:
          Accepted

- **Prior Approaches**: 기존 연구들은 discrete audio tokens를 쓰면 Transformer/LLM에 자연스럽게 붙일 수 있지만, 양자화 과정에서 세밀한 음향·화자 특성이 손실돼 downstream 성능이 떨어진다고 지적해왔다. 이를 완화하려는 diffusion 기반, continuous autoregressive, masked modeling 등은 있었지만 보통 task별로 최적화돼 단일 범용 프레임워크로 확장하기 어렵다는 한계가 있었다. 또한 discrete-only 방식은 낮은 비트레이트/초저 프레임레이트에서 미세한 프로소디와 음색을 담기 어려운 rate-distortion trade-off의 영향을 크게 받는다.

- **Core Contribution**: 이 논문은 정보 손실을 줄이기 위해 temporally compressed discrete tokens와 dimensionality-reduced continuous residual을 함께 사용하는 HybridCodec·HybridLM을 제안한다. HybridLM은 Transformer 디코더 하나로 discrete는 autoregressive, continuous residual은 단일 non-autoregressive 예측을 수행해, 화자 특성을 복원하면서도 추론 단계 수를 줄이는 설계를 목표로 한다. AdaLN으로 AR/NAR 동작 모드를 간섭 없이 내부 표현에서 분리·결합하는 점도 핵심 기여다.

- **Technical Challenges**: 문제는 discrete 토큰에서 복원된 continuous 디테일이 semantic/발화 구조 예측을 방해할 수 있다는 점이며, 이를 단순 prefix 조건만으로는 해결하기 어렵다. 저자들은 AdaLN에 mode-specific 임베딩을 매 레이어 주입해 AR용과 NAR용을 사실상 분리된 서브모델처럼 운용하도록 했고, 음향 복원을 위해 residual을 별도 focal encoder/decoder로 압축·upsampling한다. 특히 50Hz에서 6.25Hz까지 stride를 조절해 temporal resolution을 극단적으로 낮추는 환경에서도 residual 경로가 품질 저하를 완충하도록 구성했다.

- **Empirical Impact**: LibriTTS 실험에서 6.25Hz 같은 초저 프레임레이트에서도 discrete-only 대비 발화 자연스러움·화자 일치와 지능(ASR)을 동시에 개선했다. TTS는 12.5Hz zero-shot에서 UTMOS가 1.99→4.10으로 크게 상승하고 dWER도 32.97→14.79로 절반 이상 감소했으며, 6.25Hz에서도 UTMOS 3.08과 dWER 개선이 유지됐다. ASR에서도 WER/CER이 모든 프레임레이트에서 일관되게 하락해(예: 50Hz WER 28.11→23.36) discrete 기반 LLM의 효율성을 보존하면서 음향 충실도를 회복할 수 있음을 실증했다.



### Global Explanations for Multivariate Time Series Forecasting Models via $K$-Order Markov Approximations (https://arxiv.org/abs/2606.27599)
Comments:
          Accepted at the Workshop on Explainable Artificial Intelligence (XAI), International Joint Conference on Artificial Intelligence (IJCAI 2026)

- **Prior Approaches**: 기존 XAI는 주로 LIME, SHAP처럼 i.i.d. 가정과 독립적인 기준선(baseline) 마진살을 전제로 하거나, Grad-CAM/Integrated Gradients 같은 기울기 기반 방법을 국소(local) 민감도로 해석해왔다. 시계열에서는 시간 의존성, 비정상성, 변수 간 Granger causality가 독립 가정을 깨뜨려 설명이 순차·인과 구조와 어긋나거나 신뢰도 보장이 부족해진다. 특히 SHAP의 baseline 선택 문제와 시간적 교란(autocorrelation)으로 인한 off-manifold 반례가 반복적으로 지적된다.

- **Core Contribution**: 논문은 블랙박스 시계열 예측기를 “K-order Markov surrogate 모델”로 근사해, 예측이 학습한 시간 의존성을 반영하는 전이확률(transition probabilities)을 설명으로 제시한다. 또한 예측창에서 실제로 필요한 최소 과거 길이 K를 찾는 surrogacy predictive-validity 정지 규칙과, 그 K에서 모델이 접두(prefix)에 불변임을 인증하는 certified compression 및 certified-zero attribution을 함께 제공한다. 마지막으로 추정된 전이 커널로부터 추가 질의 없이 5단계(global) 설명 계층(변수 중요도→lag별 영향→고유한 레짐→모델 유도 causal graph의 엣지 기여→설명 신뢰도)을 도출한다.

- **Technical Challenges**: 핵심 난제는 (1) 예측창 전체가 아니라 실제로는 더 짧은 suffix만 사용되는 문제를 “모델 쿼리만”으로 증명 가능하게 분리하고, (2) 시계열 상태 공간의 폭발(히스토리 조합 증가)을 감당하며, (3) 전이 커널 추정의 통계적 신뢰도를 설명에 내재화하는 것이다. KARMA는 K*를 예측 차이의 임계로 찾는 모델-검증 기반 stopping certificate로 결정하고, joint 커널 대신 target 변수별 marginal 커널을 추정해 계산 가능성을 확보한다. 더 나아가 aleatoric entropy(진짜 불확실성)와 epistemic variance(추정 불안정 구간)를 분리해 reliability를 제공한다.

- **Empirical Impact**: 실제 날씨 데이터(베이징 PM 2.5)에서 5단계 설명이 예측에 기여하는 변수·lag 패턴과 레짐 변화를 구조적으로 보여준다고 제시된다. 복잡한 합성 데이터(진짜 인과 엣지가 알려진 설정)에서는 KARMA가 모델이 학습한 인과 구조를 회복하고, TimeSHAP 등 기존 attribution 대비 시간 의존성 식별 성능이 더 좋음을 복합적으로 입증한다. 특히 “시간축의 인과·순차 구조를 위반하지 않는” 전이확률 기반 global 설명과 certified 신뢰도 제공이 해당 분야의 신뢰성 요구(규제·고위험 의사결정) 대응에 의미가 크다.



### Narrative-UFET: Narrative Generation for Ultra-Fine Entity Typing (https://arxiv.org/abs/2606.27598)
- **Prior Approaches**: UFET(초정밀 개체 유형 지정)은 문장 수준 문맥만 보고 주어진 개체 언급에 대해 초세분화된 타입 라벨을 예측한다. 기존 연구들은 PLM을 바탕으로 masked language modeling 기반 또는 문장 확장 입력으로 성능을 끌어올렸지만, 학습 데이터에서 드문(롱테일) 개체에는 급격히 약해지는 문제가 남아 있다.

- **Core Contribution**: 이 논문은 UFET을 내러티브(짧은 이야기) 수준으로 확장한 Narrative-UFET을 제안한다. 각 개체-문장 쌍에 대해 자동 생성된 짧고 일관된 서사를 붙이되, 생성 과정에서 해당 개체의 타입이 내러티브 동안 유지되는 Maintain과 내러티브 동안 변하는 Change 두 변형으로 “담화 속 신호”의 효과를 통제해 분리한다.

- **Technical Challenges**: 핵심은 생성된 서사가 과도하게 부정확하거나 라벨을 우회로 암시하지 않으면서도, 타입 판별에 도움 되는 담화 구조(문장 간 연결, 코어퍼런스, 증거 분산)를 실제로 담도록 만드는 것이다. 저자들은 모델 선택(여러 오픈 소스 LLM 비교), 프롬프트 설계(문장 길이·캐릭터 수 조절), 그리고 자동 지표·사람 평가로 품질을 검증했으며, 금지된 정보 누출을 막기 위해 골드 타입은 생성기에 제공하지 않았다.

- **Empirical Impact**: 실험 결과, 내러티브 문맥은 문장 수준 기준선 대비 특히 롱테일 타입에서 일관된 성능 향상을 보였고, Maintain보다 Change가 더 강한 신호를 제공했다. 또한 OntoNotes의 자연 문맥과 비교했을 때도 합성 내러티브가 더 큰 이득을 내어, 실제 텍스트에서 암묵적으로 남는 담화 신호를 통제된 서사로 더 잘 드러낼 수 있음을 시사한다. 다만 담화 속 다른 속성까지 체계적으로 확장해야 할 여지가 남아, 담화 모델링과 내러티브 구성 모두에서 후속 연구가 필요하다는 결론이다.



### Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding (https://arxiv.org/abs/2606.27596)
Comments:
          29 pages, 25 figures. Accepted by ICML 2026

- **Prior Approaches**: 기존 LVLM 환각 완화는 주로 학습 시 정렬(alignment)하거나 추론 시 개입(intervention)을 통해 언어 priors을 억제/시각 신호를 증폭하는 방식으로 이뤄져 왔다. 특히 많은 방법이 attention intensity assumption—시각 주의의 양(세기) 부족이 환각의 핵심이라는 가정—에 의존해 global magnitude를 조정한다. 하지만 저자 분석에 따르면 환각 출력에서 시각 attention 총량 분포는 정상과 크게 겹쳐, “얼마나 보나”만으로는 구조적 원인을 설명하기 어렵다.

- **Core Contribution**: 이 논문은 환각이 전역적 세기 문제가 아니라, 결정적(decision-critical) 생성 단계에서 특정 attention head가 시각 증거와 단절되며 언어 priors로 잠기는 ‘동적 구조 불일치’에서 비롯된다고 규명한다. 저자들은 이런 head를 risky mediator로 정의하고, 시스템 지시가 우회 경로의 앵커가 되어 물체 환각으로 이어진다고 설명한다. 이를 해결하기 위해 훈련 없이 추론 단계에서만 작동하는 Fox(Faithfulness and Observational-flow via eXpression-rectification)를 제안한다.

- **Technical Challenges**: Fox의 핵심 과제는 (1) 위험한 mediator를 감독 없이 정확히 찾고 (2) 고정된 모델 파라미터를 바꾸지 않으면서 그 경로를 ‘물리적으로’ 끊는 동시에 (3) 언어의 자연스러움은 유지하는 것이다. 저자들은 visual attention entropy probe로 시각 경로의 불확실성이 크면서 동시에 system prior-path가 강하게 작동하는 head를 joint risk score로 국소화한다. 이후 do-operator를 numerical logit saturation 형태로 구현해 risky mediator의 shortcut 경로를 절단하고, 마지막으로 conflict-gated cooperative decoding으로 observational 분포와 interventional 분포의 균형을 동적으로 맞춘다.

- **Empirical Impact**: 실험에서 Fox는 POPE/CHAIR/MME 및 GPT-4V 기반 정성 평가에서 기존 추론 개입 방법을 일관되게 능가하며, 예컨대 SID 대비 29.1%의 상대 CIC_I 감소와 SOTA급 성능 향상을 보고한다. 또한 evidence-dependent 하위 지표(예: Position, Color)에서 개선 폭이 크게 나타나, 단순 보수적 회피가 아니라 실제 시각 재검증이 유도되었음을 시사한다. 결론적으로 Fox는 환각의 원인을 ‘구조적 경로’로 겨냥해 faithfulness–detail trade-off를 함께 개선하는 접근으로 해당 분야에 의미 있는 전환점을 제시한다.



### CoIn: Comprehensive 2D-3D Inpainting with Gaussian Splatting Guidanc (https://arxiv.org/abs/2606.27584)
- **Prior Approaches**: 기존 3D scene inpainting은 NeRF를 직접 최적화하거나(implicit) 3DGS 기반으로 ‘3D-first’ 순서를 적용하는 방식이 많았습니다. 3D-first 계열은 다중 뷰에서 타깃을 정확히 분리하기 위해 세그멘테이션 마스크 의존도가 높고, 초기 pruning/참조 이미지 단일 사용에 치우친 방법들은 object removal에 편중되거나 시점 변화에서 일관성이 깨지곤 합니다. 한편 ‘2D-first’는 2D diffusion의 편집 유연성은 장점이지만, 뷰별 확률적 샘플링이 cross-view 불일치를 유발해 결과가 3D 복원에 부적합해지는 문제가 남아 있습니다.

- **Core Contribution**: 이 논문은 2D diffusion inpainting과 3D Gaussian Splatting(3DGS)을 다단계 일관성 파이프라인으로 연결한 CoIn을 제안합니다. 핵심은 양방향 정보 흐름을 설계해(2D→3D→2D) arbitrary-shaped mask에서도 object removal뿐 아니라 object insertion까지 처리하고, 뷰 간 기하·외관 일관성을 동시에 확보하는 것입니다. 이를 위해 Reference Adaptive GS(Ref-GS), GS 기반 Reference Feature Warping으로 에너지 기반 Consistency Loss Guidance(CLG), 그리고 Texture-Enhancing Discriminator(TE-D)로 고주파 질감을 보정합니다.

- **Technical Challenges**: 가장 큰 기술적 난제는 2D 인페인팅의 확률적 결과가 다중 뷰에서 기하적으로 서로 맞지 않아 3D 재구성을 망가뜨린다는 점입니다. 논문은 (1) Ref-GS에서 기준 뷰를 더 크게 반영하는 per-view 가중치와 인페인팅 영역 주변 anchor feature attention을 통해 3DGS가 기준 뷰에 정렬되도록 안정화하고, (2) CLG에서 3DGS 포인트클라우드로 feature를 warping해 확산(denoising) 단계 동안 참조 뷰와의 멀티뷰 consistency를 energy로 강제하며, (3) TE-D가 adversarial patch 학습으로 가이드로 인해 생기는 blur를 줄이도록 보완합니다.

- **Empirical Impact**: SPIn-NeRF 및 IMFine에서 정량 지표(LPIPS/PSNR/FID 및 마스크 내 m-지표)와 질적 결과를 통해 CoIn이 state-of-the-art 성능을 보였다고 보고합니다. 특히 세그멘테이션 마스크/바운딩박스 등 입력 형태와 시점 커버리지 조건이 다른 설정에서도 일관된 개선이 관찰되며, object removal과 insertion 모두에서 의미 있는 결과를 생성합니다. 단일 RTX 4090 기준 파이프라인이 장면당 약 2.5시간 수준으로, 학습 기반 3D 인페인팅 대비 효율성과 범용 편집성의 균형을 제시한다는 점에서 의미가 큽니다.



### SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction (https://arxiv.org/abs/2606.27581)
Comments:
          15 pages 10 figures

- **Prior Approaches**: 기존 humanoid reinforcement-learning motion tracking은 대부분 reference motion을 따라가는 데 강점을 보였지만, 객체 접촉이나 계단 같은 비평탄 지형에서는 kinematics만으로 물리적 모호성을 해소하기 어렵다는 한계가 컸다. 특히 어떤 링크가 실제로 접촉·하중을 형성해야 하는지에 대한 scene-aware 정보가 부족해, 계단 오르기나 무거운 물체 조작 같은 장기 과업에서 실패 확률이 높았다. 더 나아가 학습용으로도 모션-장면-접촉 라벨이 함께 있는 대규모 데이터가 부족해 일반화가 제한됐다.

- **Core Contribution**: SceneBot은 단일 whole-body motion-tracking policy를 reference motion과 per-link contact labels(접촉 라벨)로 조건화해, free-space locomotion부터 terrain traversal(계단 등)과 whole-body manipulation까지 한 프레임워크에서 통합한다. 핵심은 “어떤 로봇 링크가 scene(지형/물체)과 접촉을 기대하고 활용해야 하는가”를 링크 단위로 명시해, 순수 추적을 넘어 접촉 기반 제어 인터페이스를 제공한다. 또한 장면 접촉 라벨이 없는 상황을 해결하기 위해 retargeted human motion으로부터 scene-interaction graph를 추론하는 hindsight scene reconstruction을 제안한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 접촉을 포함한 학습 데이터를 만드는 것으로, 장면 자산(지형·물체)과 상호작용 그래프, 링크 단위 접촉 라벨이 동시에 필요하다. SceneBot은 모션을 로봇에 retarget한 뒤, 저상대속도/가속도 징후와 force-closure 및 충돌 제약으로 scene-interaction graph를 구성하고, 그 그래프에 맞춰 2.5D elevation map(지형)과 접촉면 평면 기반 object를 재구성해 학습 데이터를 합성한다. 학습 단계에서는 PPO로 kinematic tracking과 contact correctness/체류시간 보상 및 contact-mismatch 종료 조건까지 함께 최적화해 안정적 접촉 전환과 파지 성능을 노린다.

- **Empirical Impact**: MuJoCo sim-to-sim에서 Unitree G1 기반으로 검증한 결과, free-space에서는 SONIC과 비슷한 수준을 유지하면서도 지형·객체 접촉 과업 전반에서 유의미하게 더 높은 성능을 보였다. 특히 손/발 contact labels를 빼거나 비활성화하면 성공률이 급락해, contact conditioning이 실제로 정책 성공을 좌우함을 실험적으로 확인했다. 시뮬레이션 성공뿐 아니라 계단 오르기와 상자 운반 같은 long-horizon 복합 과업을 정성적으로도 수행하며, free-space와 contact-rich 행동을 자연스럽게 통합하는 일반 프레임워크의 의미를 보여준다.



### Retroactive Advantage Correction: Closed-Form V-Trace Bias Correction for Delay-Aware RLHF (https://arxiv.org/abs/2606.27580)
Comments:
          Accepted at the ICML 2026 Workshop on Reinforcement Learning from World Feedback (RLxF). Code: this https URL

- **Prior Approaches**: PPO는 보통 보상(또는 advantage) 신호가 다음 옵티마이저 업데이트 전에 도착한다는 동기 가정을 전제로 한다. 하지만 RLHF 프로덕션에서는 code-execution verifier, slow judge ensemble, human review 같은 지연 신호가 여러 옵티마이저 스텝 뒤에 도착해 PPO의 기준선(동기성) 가정이 깨진다.
기존 보정은 주로 actor staleness나 value-target의 off-policy 문제를 다루는 V-trace, Retrace 계열이거나, 토큰/턴 단위 IS clipping, truncated PPO처럼 잘림 축을 다른 곳에 둔다. 즉 “옵티마이저 스텝 축에서의 reward delay” 자체를 정면으로 다루는 정식 해법이 부족했다.

- **Core Contribution**: 논문은 Retroactive Advantage Correction(RAC)으로, 느리게 도착한 보상/판정 결과를 큐에 대기시켰다가 “다음 업데이트의 advantage 보정”으로 전방 주입하는 메커니즘을 제안한다. 각 지연 completion은 age kernel로 가중되고, 표준 V-trace 스타일 clipped importance ratio로 보정된 residual 형태(느림-빠름 기준선 차이)로 누적된다.
또한 RAC가 cumulative-bias 관점에서 어떤 조건에서는 기대값 기준 무편향이 되고, 다른 경우에는 커널의 row-stochastic slack에 선형으로 바이어스가 증가한다는 닫힌형(closed-form) 항등식을 증명한다.

- **Technical Challenges**: 핵심 난제는 지연된 slow 신호가 여러 옵티마이저 스텝에 걸쳐 “어떤 행동에 대한 evidence인지”를 다시 연결해야 한다는 점이다. RAC는 delay를 effective delay kernel로 모델링하고, (1) clipped importance ratio의 조건부 mean-one, (2) residual과 비율의 조건부 독립 같은 정리를 만족하는 조건 하에서 cumulative RAC correction의 무편향/바이어스 항을 분석한다.
실무 구현은 PPO/GRPO의 reward-manager 인터페이스에 O(K) 큐 업데이트와 advantage에 대한 한 번의 텐서 덧셈만 추가하는 형태로 설계해, 기존 학습 파이프라인을 크게 바꾸지 않도록 했다.

- **Empirical Impact**: tabular MDP proof-of-concept에서 RAC는 two-slow-channel 구성(K=2, 전형적인 async-RLHF 배치)을 기준으로 closed-form policy bias를 최대 47.9배 줄였다. wait-for-slow 대비 더 낮은 벽시계 비용에서 더 큰 바이어스 감소를 보였고, Retrace의 advantage-level 적용은 느린 신호를 과도하게 약화시켜 1.5배 수준에 그쳤다.
또한 77B 규모에서 reward 분포 기반의 기계정밀 검증으로, identity 커널에서 V-trace로 수축되는 점과 편향 스케일링 법칙을 확인했으며, 저자들은 end-to-end LLM-scale PPO 검증을 다음 단계로 제시한다.



### Distribution-based deep multiple instance learning for tumor proportion scoring in NSCLC (https://arxiv.org/abs/2606.27579)
- **Prior Approaches**: 기존 연구는 PD-L1의 TPS를 슬라이드 연속값으로 예측하거나, 임상 기준선에 따라 <1·1–49·50–100 같은 클래스 분류로 나눠 접근해왔다. 특히 weakly-supervised multiple-instance learning(MIL)은 슬라이드 레이블만으로 학습 가능하지만, TPS가 0에 해당하는 non-expressive(zeroclass) 이미지에서 성능이 급격히 떨어지는 문제가 지적돼 왔다. 또한 시각적 attention은 흔하지만 예측의 불확실성을 모델 구조에 내장해 정량화하는 방법은 제한적이었다.

- **Core Contribution**: 이 논문은 슬라이드 레이블 TPS만 사용해 end-to-end로 TPS 분포를 예측하는 프레임워크를 제안한다. 패치 임베딩을 먼저 추출한 뒤 MIL로 집계하되, 예측값을 단일 회귀로 내지 않고 zero-inflated beta(ZIBeta) 분포 파라미터로 모델링해 TPS=0일 확률과 분포 농도(precision 관련)를 함께 산출한다. 이를 통해 zero-class 정확도와 예측의 설명/불확실성 표현을 동시에 강화하려는 것이 핵심이다.

- **Technical Challenges**: 가장 큰 기술적 도전은 non-expressive 샘플(zeroclass)에서 회귀/분류 기반 모델이 쉽게 붕괴하는 점이다. 저자들은 ZIBeta로 TPS=0의 “점(0)”과 0이 아닌 연속 구간을 분리해 음의 로그우도(NLL)와 MSE를 결합한 손실을 설계했고, ZIBeta의 이진 성분 ϕ를 위한 threshold는 검증셋에서 Cohen’s kappa를 최대화하는 방식으로 정했다. 추가로 attention 기반 집계와 패치 엔트로피 최소화(균질성 가정)를 통해 잡음 패치를 걸러내도록 학습을 구성했다.

- **Empirical Impact**: 실험에서 ZIBeta 기반 MIL(ZIBeta 변형)은 선형/리지 회귀 및 다른 deep MIL 대안들 대비 전반 지표에서 상위 성능을 보였고, 특히 0 클래스 정확도와 관련된 평가에서 유리했다. 또한 beta 분포의 농도(ν)가 높을수록 절대 오차가 감소하는 경향을 관찰했으며, 동시에 WSI 간 이질성 때문에 단일 농도 값만으로는 신뢰도 프록시로 삼기 어렵다는 점도 확인했다. attention이 실제로 염색된 종양 패치와 진단 관련 영역에 더 집중하며 아티팩트 영향을 줄인다는 정성적 설명 가능성도 제시해, 임상 워크플로우에서의 검증/신뢰성 제공에 의미가 있다.



### PEBS: Per-rater Empirical-Bayes Shrinkage for RLHF Reward-Model Calibration (https://arxiv.org/abs/2606.27578)
Comments:
          Accepted at the ICML 2026 Workshop on Pluralistic Alignment. Code: this https URL

- **Prior Approaches**: RLHF에서는 Bradley-Terry 기반 선호쌍 모델을 여러 annotator의 데이터를 한데 묶어 학습하는 경우가 많지만, 이때 annotator별 점수 스케일의 오프셋·기울기 차이가 보상 신호와 섞이며 개인별 정확도를 떨어뜨릴 수 있다. 부분 누적(partial pooling)이나 계층적 rater 모델이 일반적인 대안으로 알려져 있지만, RLHF 보상모델 파이프라인에서는 이러한 per-rater 보정이 보통 후처리 형태로 적용되지 않는다.

- **Core Contribution**: 이 논문은 PEBS(per-rater empirical-Bayes shrinkage estimator)를 제안한다. PEBS는 reward model을 재학습하지 않고, annotator별로 affine calibrator(기울기·오프셋)를 추정한 뒤 모집단 평균으로 shrinkage하는 closed-form 후처리로 기존의 “전역 affine 보정” 문제를 해결한다. 결과적으로 추론 시 각 annotator의 rater-level map만 보정해 개인별 캘리브레이션 품질을 높인다.

- **Technical Challenges**: 핵심 과제는 per-rater 캘리브레이션 이질성을 다루되, reward model의 pairwise argmax 성질을 깨지 않으면서 캘리브레이션 민감 손실(RMSE, Bradley–Terry NLL)에서 이득을 내는 방법을 찾는 것이다. 논문은 per-rater OLS 추정치의 분산을 기반으로 Morris/James–Stein 형태의 empirical-Bayes 가중치를 계산해 기울기와 오프셋을 독립적으로 shrinkage하며, 샘플 분할 변형에 대한 위험(risk) 상계 및 closed-form gain 예측까지 이론적으로 다룬다.

- **Empirical Impact**: PRISM에서 PEBS는 pooled population-slope 기준선 대비 within-user held-out RMSE를 8.58% 낮추고, 평균 pair BT-NLL도 5.7% 개선되며 개선 폭은 특정 사용자/어려운 쌍에 집중된다. PluriHarms harm rating에서도 동일한 기준선 대비 RMSE가 9.66% 개선되고, 다른 코퍼스들에서도 per-cluster RMSE 감소가 재현되면서 “annotator(또는 cluster) 스케일 이질성” 축에서의 보정 효과를 확인한다. 또한 pair accuracy 자체는 유지되는 반면(단조 변환 불변성), PPO/DPO 같은 RLHF 학습 단계에서의 캘리브레이션 민감 실패를 완화해 downstream 학습 안정성에 기여함을 보인다.



### hia-gat: A Heterogeneous Interaction-Aware Graph Attention Network For Frame-Level Traffic Conflict Risk Prediction On Freeways (https://arxiv.org/abs/2606.27577)
- **Prior Approaches**: 기존 안전 분석은 사고 기록에 의존해 실시간·선제적 평가가 어렵고, TTC·PET 같은 surrogate safety measures(SSM)는 가능하지만 대개 고정 임계값과 제한된 상호작용만 다뤘습니다. 또한 그래프 신경망 연구가 있더라도 주로 intersection 중심이었고, freeway에서 다중 동시 상호작용을 frame 단위로 통합 분류하는 비전+궤적 기반 접근은 부족했습니다.

- **Core Contribution**: 이 논문은 freeway의 frame-level 위험 평가를 다중 에이전트 scene을 “scene graph-level binary classification”으로 정식화하고, TTC 또는 PET 위반을 기준으로 risky/safe 라벨을 만듭니다. 동시에 longitudinal(동일 차로)과 lateral(인접 차로) 상호작용을 분리한 이중 스트림 HIA-GAT을 제안해, 충돌 메커니즘에 맞춘 메시지 패싱과 conflict-type-aware gating 융합을 수행합니다.

- **Technical Challenges**: 핵심 난관은 (1) SSM이 본질적으로 pairwise인데 실제 장면은 동시 다중 상호작용이라는 점, (2) 게이팅이 trivial solution로 붕괴하지 않고 TTC 기반(후방추종) vs PET 기반(차로변경/측면접촉) 구분을 학습하도록 하는 점입니다. 이를 위해 차로 관계별로 edge를 나누고(근접 반경 내 차량 쌍), 물리 기반 edge feature를 각 충돌 유형에 맞게 설계했으며, SSM 충돌 attribution에서 얻은 사건 단위 정보를 gate에 supervision으로 주어(상황별 1/0/0.5 표적, 해당 노드만 손실 계산) 게이팅 구별력을 확보했습니다.

- **Empirical Impact**: NGSIM I-80과 US-101에서 TTC·PET 임계값 9가지 구성(조합 포함)으로 실험한 결과, HIA-GAT은 평균 위험 순위 성능에서 최고를 보이며 AUC가 I-80 0.835, US-101 0.867을 기록했습니다. 특히 PET-only(차로변경) 설정에서 가장 큰 이득을 보여, lateral 충돌 위험 모델링에 그래프 구조가 필수임을 실증했고, learned gate는 차량별로 우세한 충돌 유형을 해석 가능하게 제공해 실시간 안전 모니터링에 활용될 여지도 제시합니다.



### On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models (https://arxiv.org/abs/2606.27567)
Comments:
          18 pages, 1 figure, 2 tables

- **Prior Approaches**: 지금까지의 prompt injection 방어는 instruction tuning, RLHF, guardrail classifier, perplexity filtering, sandwich defenses, XML 태깅, representation engineering, circuit breaker 같은 방식으로 “모델 출력/입력”을 통제하려는 시도가 많았다. 하지만 최근에는 multi-turn 분해 공격이 높은 성공률(90–99%), Crescendo 같은 강공격에서도 circuit breaker가 여전히 큰 ASR(54.2%)을 보이는 등 방어가 빠르게 무너지는 사례가 반복됐다. 논문은 이런 붕괴가 우연이 아니라, 기존 접근이 근본 구조적 제약을 바꾸지 못했기 때문일 수 있다고 문제를 제기한다.

- **Core Contribution**: 이 논문은 shared-embedding 아키텍처(명령과 데이터가 같은 임베딩 공간에서 처리되고, 제어-데이터 분리가 강제되지 않는 구조)에서는 prompt injection을 “완벽히” 막는 것이 수학적으로 불가능함을 증명한다. 이를 위해 Prompted Action Model로 에이전틱 시스템의 제어-권위 행동(거부, tool authorization, policy routing, memory write)을 형식화하고, Semantic-Faithful Control(SFC)라는 보안 성질(의미가 같으면 제어 행동도 같아야 함)을 정의한다. 결론적으로 SFC는 공유 파이프라인 내부에서 달성 불가이며, 더 나은 in-pipeline 분류/정렬만으로는 한계가 구조적으로 고정된다고 주장한다.

- **Technical Challenges**: 핵심 난관은 “신뢰/비신뢰가 같은 표현 공간에서 섞여 들어가므로, 의미와 무관한 인코딩 조작이 제어 경로에 침투할 수 있다”는 점을 이론적으로 차단하는 것이다. 논문은 (1) provenance-recovery 불가능성(조건부 표현 분포가 통계적으로 분리되지 않음), (2) control-path exposure(주의 기반 집계가 untrusted 값을 제어 관련 계산으로 유입), (3) finite training이 무한한 의미-동등 인코딩 클래스 전체에 대한 불변성을 인증할 수 없다는 갭이라는 세 결과로 SFC 불가능을 도출한다. 또한 production 토크나이저/모델 측정으로 이러한 양을 실증 기반에 올려 “현재 방어가 약해서가 아니라, 아키텍처가 막지 못한다”는 구조적 성격을 강조한다.

- **Empirical Impact**: 논문은 universal adversarial suffix 같은 기법이 의미적으로 무의미한 토큰을 덧붙이는 것만으로도 거부↔순응 같은 제어 행동을 뒤집는 현상을 SFC 위반의 구체 사례로 연결한다. 더 나아가 생산 환경 토크나이저와 실제 모델에서 필요한 통계적 분리/불변성 조건이 성립하지 않는 방향의 측정 결과를 제시한다. Von Neumann 구조의 code-data confusion과 유사하게, 단일 방어 기제가 아니라 “instruction과 data 채널을 아키텍처적으로 분리”하는 해결이 필요하다는 메시지를 AI 보안 설계의 방향성으로 제안한다.



### Benchmarking Multi-Modal Graph-based Social Media Popularity Prediction (https://arxiv.org/abs/2606.27539)
- **Prior Approaches**: 소셜 미디어 인기 예측은 초기 관측 신호로 미래 도달/영향력을 추정하지만, 기존 연구는 텍스트·비주얼 같은 멀티모달 정보와 시간에 따른 사회적 상호작용을 함께 다루지 못하는 경우가 많았습니다. 또한 데이터셋·모달리티·관측 윈도우·예측 타깃·평가 프로토콜이 제각각이라 공정한 비교와 신호 조합에 대한 체계적 이해가 어려웠습니다.

- **Core Contribution**: 이 논문은 Bluesky와 Reddit을 포함한 여러 데이터셋을 표준화해 MMG-Pop이라는 멀티모달 그래프 기반 인기 예측 벤치마크를 제안합니다. 아울러 MMG-PopNet은 텍스트/이미지/그래프 상호작용을 함께 모델링하는 단일 네트워크로, 플랫폼 간 일반화와 멀티태스크 이득까지 분석할 수 있는 기반을 제공합니다.

- **Technical Challenges**: 핵심 과제는 (1) 텍스트·이미지·그래프 구조를 하나의 학습 파이프라인으로 정렬하고, (2) 관측 윈도우 내에 드러나는 cascade prefix로부터 미래 상태를 안정적으로 예측하는 것입니다. 연구진은 그래프 메시지 패싱(GraphSAGE)으로 사회적 상호작용을 학습하고, 멀티모달 임베딩을 예측 헤드에 융합한 뒤 R2R2 및 Spearman 상관 등으로 수치 오차와 순위 보존을 동시에 평가하며 모델 설계를 검증했습니다.

- **Empirical Impact**: 실험 결과 MMG-PopNet은 4개 데이터셋과 다양한 관측 윈도우에서 MSE와 R2R2, Spearman 상관 모두에서 일관되게 베이스라인을 앞섰습니다. 특히 like score처럼 난도가 높은 타깃에서도 우위가 관측됐고, paired bootstrap 통계 검정과 FDR 보정 하에서 16/16 비교가 유의하게 개선되어 실제 성능 향상을 뒷받침했습니다.



### The Context-Ready Transformer (https://arxiv.org/abs/2606.27538)
Comments:
          NeurIPS, 22 pages

- **Prior Approaches**: 기존 오토리그레시브 트랜스포머는 토큰을 context-free한 임베딩으로 블록에 넣은 뒤, 여러 층을 거쳐 다시 문맥을 재구성한다는 ‘왕복’이 구조적으로 발생한다. weight sharing(예: ALBERT, Universal Transformer), early exit, lookahead/고정점(예: DEQ) 같은 접근은 연산량이나 추론 속도를 줄이지만, 토큰이 블록에 들어갈 때 이미 문맥 정보를 담아 처리한다는 발상은 상대적으로 부족했다. 또 Mamba/RWKV 계열은 attention을 줄이지만, 핵심은 “문맥 전달 방식”이 아니라 “상태 요약으로 attention을 대체”하는 데 가깝다.

- **Core Contribution**: 이 논문은 context-ready transformer를 제안하며, 토큰이 블록에 들어가기 전에 correction 네트워크가 이전 블록 출력(과거 문맥 요약)을 반영해 미리 문맥화(pre-contextualize)하도록 만든다. 학습 중에는 correction 과정을 K번 병렬로 unroll해 transformer식으로 훈련 가능한 그래프를 유지하면서, 추론에서는 left-to-right 한 번의 스트리밍 패스로 recurrent 동작이 “정확히” 작동하게 설계했다. 또한 pretrained transformer는 zero-initialized correction FFN을 추가하고 fine-tuning하면 context-ready 모델로 변환할 수 있다고 주장한다.

- **Technical Challenges**: 가장 큰 문제는 correction이 이전 위치의 블록 출력에 의존해 추론만큼이나 학습도 순차성이 생긴다는 점인데, 이는 RNN처럼 되면 BPTT 깊이가 시퀀스 길이에 비례해 불리하다. 이를 해결하기 위해 (1) non-cumulative correction으로 매 반복은 누적 합이 아니라 “고정점에 수렴해가는 단일 correction” 형태가 되게 하고, (2) past-only correction으로 t 위치의 correction이 t-1의 캐시된 정보와 현재 임베딩에만 의존하도록 제한했다. 결과적으로 훈련 그래프의 깊이는 O(T)가 아니라 O(K)로 유지되며, K=5~10에서 수렴 및 성능 확보가 관찰된다.

- **Empirical Impact**: 실험에서 D=5 모델은 12층 트랜스포머를 PPL 측면에서 앞서면서 A100 기준 생성 속도도 1.7배 개선했다(동일 길이/조건 비교). 더 작은 D=1 설정에서도 K=10 학습을 통해 6층 대비 2.6배 빠른 추론을 보이면서 PPL에서 근접 또는 개선을 달성했으며, sequential inference가 training의 K unroll과 PPL 차이 0.01 이내로 맞춰진다고 보고한다. pointer-chasing 합성 추론에서는 BPTT로 훈련한 D=1이 모든 composition level을 해결하는 반면 표준 트랜스포머는 depth에 따라 계단형(staircase-like) 성능 의존을 보였고, 전반적으로 wide representation과 long context에서 이점이 커지는 것으로 정리된다.



### Large Language Model Teaches Visual Students: Cross-Modality Transfer of Fine-Grained Conceptual Knowledg (https://arxiv.org/abs/2606.27527)
Comments:
          Accepted by ICML 2026

- **Prior Approaches**: 기존 지식 증류(KD)는 주로 같은 데이터에서 학습된 비전(또는 텍스트-비전) 교사로부터 logits나 특징을 직접 모방하도록 설계돼 왔다. 하지만 파인그레인드 분류처럼 미세한 차이를 요구하는 과제에서는 시각적 교사 기반 학습이 배경 같은 우연 신호(스퓨리어스 상관)에 더 휘둘릴 수 있다는 한계가 있었다. 또 멀티모달 KD는 정렬된 입력이나 멀티모달 교사가 필요해 비용과 편향 전이가 부담이 되곤 한다.

- **Core Contribution**: 이 논문은 Language-to-Visual Knowledge Distillation(LaViD)로, 이미지 입력이 전혀 없는 language-only LLM을 교사로 사용해 비전-only 학생을 지도하는 프레임워크를 제안한다. LaViD는 클래스 간 의미 차이를 찌르는 multiple-choice questions(MCQs)를 LLM에서 생성하고, 각 클래스에 대해 질문별 soft label 분포(개념 서명)를 만들어 학생의 보조 증류 손실에 활용한다. 그 결과, 짝지어진 멀티모달 데이터 없이도 언어의 고수준 개념을 시각 표현 학습으로 전이할 수 있음을 보인다.

- **Technical Challenges**: 핵심 기술 난제는 ‘텍스트 교사가 이미지 없이도 유의미한 시각 개념 관계를 제공’하도록 구조화된 감독 신호를 설계하는 것이다. LaViD는 데이터 메타데이터와 클래스 이름을 바탕으로 LLM이 시각적으로 근거 있는 MCQ를 생성하게 하고, 각 답지의 pre-softmax logits를 추출해 클래스별 Q×M 시그니처로 고정한다. 학생은 이미지로부터 보조 헤드를 통해 같은 질문 공간에서의 예측을 만들고, 분류 손실에 더해 LLM 타깃과의 MSE 증류 손실을 함께 최적화하며 관계적 구조를 내재화한다.

- **Empirical Impact**: 6개 파인그레인드 벤치마크와 ImageNet 부분 서브셋에서 LaViD는 MaKD 같은 멀티모달 교사 기반 최신 방법을 언어-only 교사만으로 일관되게 능가한다. 또한 DKD, MLKD 같은 전통적 비전 교사 기반 KD와 경쟁하거나 더 나은 성능을 보이며, logit standardization과 결합 시 추가 개선도 확인됐다. Waterbirds에서는 worst-group accuracy가 크게 향상돼, 증류가 스퓨리어스 상관에 덜 의존하게 만드는 견고성 개선 효과가 관찰된다.



### DMV-Bench: Diagnosing Long-Horizon Multimodal Agents' Visual Memory with Incidental Cue Injection (https://arxiv.org/abs/2606.27499)
Comments:
          16 pages

- **Prior Approaches**: 기존 에이전트 메모리 연구와 벤치마크는 주로 텍스트 기반을 중심으로 설계돼, 시각 정보를 “기억해야만” 풀리는지 엄밀히 분리하기가 어려웠습니다. VisualWebArena·WebArena·장기 비디오 QA 등은 캡션/alt-text 등 텍스트 단서를 함께 제공해, 시각 기억이 필요해도 텍스트 메모리가 사실상 우회할 여지가 있었습니다.

- **Core Contribution**: 이 논문은 멀티모달 에이전트의 시각 메모리를 상호작용·다중 세션 환경에서 평가하도록 DMV-Bench를 제시합니다. DMV-Bench는 1,000개 가구 상품 변형에 “incidental cue”를 화소에만 심고(L2-leakage contract), 세션마다 대화 맥락을 지운 뒤 특정 큐가 있는 상품을 찾아가게 하여 recall을 “reach(몇 번의 세션 경계를 넘어 살아남았는지)” 곡선으로 측정합니다. 또한 DualMem(dual-coding 기반)을 통해 시각 코드와 언어 코드를 병렬로 저장·검색하도록 설계합니다.

- **Technical Challenges**: 핵심 기술적 난제는 시각 단서가 텍스트로 새지 않게 설계하면서도, 실제로 “언제 기억이 필요한지”를 세션 경계 단위로 측정하는 프로토콜을 만드는 것입니다. 논문은 큐를 이미지에만 존재하게 하고 텍스트 표면에서 사전 감사로 제거하는 L2-leakage contract, 그리고 공통 관찰 스트림을 재활용하는 shared-prefix rollout tree로 계산 비용을 줄이면서 reach별 회상 정확도를 효율적으로 산출합니다. DualMem은 SigLIP-2(시각)와 SBERT(언어 캡션)의 두 임베딩을 저장하고, 검색 시 두 채널의 유사도를 정규화·가중 결합한 뒤 후보의 이미지+캡션을 VLM에 함께 주입하는 방식으로 구현했습니다.

- **Empirical Impact**: DualMem은 Gemini 2.5 Flash와 Qwen2.5-VL-7B 모두에서 체인 길이 J∈{5,10,15,50} 전 구간에서 캡션 베이스라인과 최신 멀티모달 메모리 시스템 3종을 능가합니다. 메모리 뱅크 크기나 인코딩 위치 편향 같은 대조 조건에서도 리드가 유지됐고, ablation 결과 시각 채널이 큐를 end-to-end로 “운반”하며 언어 채널은 쿼리 그라운딩에 상대적으로 작은 비중으로 기여하는 비대칭 dual-coding 양상이 나타났습니다. 연구진은 이 결과가 장기적인 지각 연속성(perceptual continuity) 관점에서 시각 메모리를 설계 목표로 다뤄야 한다는 메시지를 강화한다고 강조합니다.



### Speculative Refinement: A Hybrid Autoregressive Diffusion Decoding Strategy and Its Behavior Across Benchmarks (https://arxiv.org/abs/2606.27474)
Comments:
          7 pages + 2 pages References

- **Prior Approaches**: 기존 평가는 주로 left-to-right autoregressive(AR) 생성에 맞춰져 있어, diffusion·speculative decoding·하이브리드 파이프라인에는 그대로 적용하기 어렵다. 특히 코드/추론 벤치마크의 후처리·평가 프로토콜·토크나이저 가정이 결과를 조용히 왜곡할 수 있었다.

- **Core Contribution**: 이 논문은 AR draft와 masked diffusion refiner를 결합한 Speculative Refinement(SpecRef)을 다루면서, “어떻게 평가해야 하는가”를 체계적으로 점검한다. 학습 없이 entropy 기반 선택적 마스킹으로 diffusion을 warm-start하고, 여러 벤치마크·프로토콜에서 기존 평가의 맹점을 드러낸다.

- **Technical Challenges**: 핵심은 AR과 diffusion의 다른 토크나이저, 토큰 불확실성(Shannon entropy)을 마스킹으로 전환하는 정렬, 그리고 단계 수에 따른 구조 발견 실패를 분리해 해석하는 것이다. 이를 위해 문자 오프셋 기반 토큰 점수 매핑, 수학/수치 구간에 대한 block expansion, tail truncation, 그리고 실행 기반 코드 샌드박스 타임아웃·stop-sequence 중심의 안전한 후처리를 적용했다.

- **Empirical Impact**: HumanEval·MBPP에서 문법 구조(예: 들여쓰기/괄호)가 성능을 좌우함을 보이며, syntactic scaffold만으로 정확도가 0%대에서 20%+로 급등했다. 또한 refinement tension으로 다단계 보정이 이미 맞는 토큰을 망가뜨려 정확도가 하락할 수 있고, log-likelihood 기반 평가와 생성 기반 평가는 같은 모델도 서로 다른 순위를 낳는다는 점을 확인했다. 코드 후처리(예: strip/dedent)가 non-AR 생성의 indentation을 깨뜨려 pass@1을 떨어뜨릴 수 있어, 향후 평가 관행이 더 진단적으로 바뀔 필요가 있음을 시사한다.



### Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents (https://arxiv.org/abs/2606.27472)
Comments:
          11 pages, 4 figures, 3 tables. Code, environment, model, and dataset: this https URL

- **Prior Approaches**: 장기 대화에서 시간에 따라 정보가 바뀌는 문제는 LongMemEval, MemoryArena, MemBench 등으로 평가돼 왔지만, 대부분은 고정된 모델 성능을 측정하는 데 그친다. 또한 RL 기반 메모리 에이전트 연구는 최종 정답이나 증거 관련성 같은 보상으로 학습해 왔고, ‘현재 시점의 값(cu rrency)’을 맞추는 시간성 보상은 없었다.

- **Core Contribution**: 이 논문은 대화 중 변경/철회된 사실을 ‘최신 값으로 유지하며 답하는 능력’(supersession)을 분리 측정하고, 이를 제대로 못하는 실패(supersession gap)가 별도 능력 결함임을 보여준다. 더 나아가 보상을 시간 인덱스가 있는 supersession 정합성으로 직접 정의한 공개 강화학습 환경 Supersede를 제안하고, 실제 학습으로 격차를 줄일 수 있음을 입증한다.

- **Technical Challenges**: 핵심 병목은 독해력보다 메모리 유지 관리이며, bounded memory 환경에서 구체적으로 어떤 값이 최신인지 판단해 폐기해야 한다. 이를 위해 원시 세션을 재주입(re-feed)하지 않고 notes 필드만으로 답하게 하며, 프로그램 기반 매처로 ‘정답이 현재 값을 전달했는지’(그리고 필요 시 stale 값을 주장하는지)를 보상 신호로 만든다.

- **Empirical Impact**: 실험에서 full-context는 gpt-5.4 기준 92%에서 92%에 가깝게 포화되는 반면, bounded self-maintained memory는 77%로 크게 하락해 격차가 통계적으로 유의하다(p<0.005). 모델을 더 키워도, 대화를 24배 길게 늘려도(정확도 68%→28%) 메모리를 비례 증량해도(28%→28%) 격차가 닫히지 않으며, 학습은 GRPO로 Qwen2.5-3B의 held-out supersession 정확도를 9.0%→16.7%로 거의 두 배 가까이 끌어올린다.



### GRAFT: Biological Graph and Hypergraph Benchmarks for Linked Gene Expression and Phenotypic Trait Prediction in Arabidopsis thaliana (https://arxiv.org/abs/2606.27413)
Comments:
          arXiv admin note: text overlap with arXiv:2508.14934

- **Prior Approaches**: 기존 연구는 유전체-형질 매핑을 위해 높은 차원·이질 데이터에 의존하지만, 식물 분야에서는 유전체(유전자 발현)와 표현형(형질 측정)이 같은 개체에서 함께 수집된 벤치마킹 데이터가 부족했다. 또한 대부분의 데이터셋은 단일 omics 또는 특정한 단일/협소한 형질에 집중해 유전자-형질 상관의 폭을 제한했다. 그래프/하이퍼그래프 기반 모델이나 설명가능성 기법은 있었지만, 이를 G2P(Genome-to-Phenome)에 맞춘 통합 데이터와 평가 파이프라인은 충분히 마련되지 못했다.

- **Core Contribution**: 이 논문은 Arabidopsis thaliana의 동일 개체에 대해 유전자 발현(RNA-seq)과 다양한 표현형/생리 지표(이미지·수동 측정·분광계 측정)를 함께 연결한 멀티모달 데이터셋 GRAFT(Gene-Graph Regression for Arabidopsis Functional Traits)를 제안한다. GRAFT는 유전자 발현 예측뿐 아니라, 유전자-생물학 기능(예: Gene Ontology) 구조를 활용한 해석 가능 학습과 phenotype prediction 같은 과제를 지원한다. 또한 기준선 회귀 모델과 하이퍼그래프 기반 생물학적 기준선을 벤치마크로 제공해 유전자-형질 연관을 검증한다.

- **Technical Challenges**: 가장 큰 기술적 도전은 (1) 유전자는 매우 고차원인데 샘플 수는 제한적이고, (2) 형질은 측정 방식이 다양하며, (3) 설명가능성을 위해 생물학적 구조가 모델에 반영되어야 한다는 점이다. 논문은 데이터 누수를 막기 위해 교차검증 내부에서 분산 필터와 Spearman 상관 기반 유전자 필터(최대 상관 상위 k)를 적용해 입력 차원을 줄인다. 이어 그래프 모델에서는 WGCNA TOM 기반의 생물학적 인접구조를 쓰고, 하이퍼그래프 모델에서는 GO 용어마다 유전자 집합을 하이퍼엣지로 구성해 multi-gene 경로 수준 관계를 반영한다. 설명은 SHAP 기반 유전자 중요도와 GO enrichment을 연결한 BER(Biological Explanation Recall)로 평가해 예측 성능과 별개로 ‘설명 품질’을 수치화한다.

- **Empirical Impact**: GRAFT 위에서 다양한 회귀 백본(MLP, GCN, graph transformer, SAGE)과 HGNN(하이퍼그래프 convolution, HyperGCN, UniGAT)을 비교 벤치마크하며, 생물학적 구조를 반영한 기준선이 유전자-기능 연관 설명에서 강점을 보이도록 설계된 평가를 제시한다. 특히 BER@k는 모델이 도출한 상위 유전자 집합이 관련 있는 GO 용어를 얼마나 회수하는지로 설명 품질을 정량화해, 단순 정확도 중심 한계를 보완한다. 결과적으로 이 데이터셋은 G2P 연구에서 멀티모달 통합 학습과 해석가능성 평가를 동시에 촉진할 수 있는 표준 벤치마크로서 의미가 크다.



### Not All Relations Rotate Alike: Transformation-Aware Decoupling for Viewpoint-Robust 3D Scene Graph Generation (https://arxiv.org/abs/2606.27412)
- **Prior Approaches**: 3D Scene Graph Generation(3DSGG)은 객체를 노드, 관계를 엣지로 하는 그래프로 3D 장면을 구조화해 공간 추론의 기반을 제공해 왔습니다. 기존 방법들은 주로 쌍별 객체 특징과 상대 기하를 결합해 관계를 멀티라벨로 예측하며, 표준(정렬된) 장면 자세에서 높은 성능을 보였습니다. 그러나 yaw 관점 변화(특히 좌/우-전/후 축이 교환되는 경우)에서 관계 예측이 시야 좌표계의 변환 규칙을 따르지 못한다는 실측 불일치가 남아 있습니다.

- **Core Contribution**: 논문은 이 문제의 원인을 ‘predicate-level transformation heterogeneity(관계 술어 변환의 이질성)’로 규명합니다. 방향 술어(left/front/right/behind)는 관측 프레임에 따라 변환돼야 하지만, contact/support/semantic 계열은 상대적으로 안정적이어야 하는데, 기존 모델은 이를 구분하지 않고 같은 관계 공간에 얽어 넣는다고 지적합니다. 이를 해결하기 위해 Transformation-Aware Decoupling(TAD)는 변환 거동이 다른 관계를 분리 학습하고, 다시 표준 멀티라벨 예측으로 결합하는 프레임워크를 제안합니다.

- **Technical Challenges**: 핵심 기술 과제는 ‘방향 술어는 yaw에 맞게 바뀌어야 하고, 나머지 술어는 유지돼야 한다’는 서로 다른 목표를 한 모델 안에서 충돌 없이 학습시키는 것입니다. TAD는 Viewpoint-Stable Object Encoder(VSOE)로 관점이 바뀌어도 안정적인 객체 표현을 만들고, 관계 추론은 invariant(변환 불변) 경로와 direction-sensitive(방향 민감) 경로로 디코딩합니다. 또한 변환별 기하 디스크립터 분리, 비공유 관계 GNN 브랜치, orthogonal 정규화, 그리고 그룹-aware 보조 감독을 통해 두 경로가 상보적 단서를 포착하도록 유도합니다.

- **Empirical Impact**: 3DSSG 벤치마크에서 TAD는 yaw 회전에 대한 robust 성능을 크게 개선하면서도, 표준(비회전) 설정에서는 경쟁력 있는 성능을 유지합니다. 특히 0°에서 90°/270°로 이동할 때 축 교환이 발생해 더 어려운 구간에서, 기존 비회전 학습 베이스라인들이 큰 성능 저하를 보인 반면 TAD는 그 격차를 줄였습니다. 또한 rotation augmentation(학습 시 회전 데이터 증강)을 쓰지 않고도 상태최신 수준의 yaw 강건성을 달성했으며, 이는 embodied intelligence에서 관점 불변 그래프 생성의 실용성을 높인다는 점에서 의미가 큽니다.



### Compression-Driven Anomaly Detection in Brain MRI Using an Interpretable Quantum Autoencoder (https://arxiv.org/abs/2606.27411)
- **Prior Approaches**: 기존 뇌 MRI 이상 탐지는 주로 분류·세그멘테이션 또는 재구성 기반 autoencoder/PCA 같은 전통적 차원감소에 의존해 왔다. 이들 방식은 이상을 “무엇이 압축을 방해하는가” 관점에서 설명하기 어렵고, 압축-복원 간 트레이드오프를 제어하기도 까다롭다. 또한 임계값 설정 근거가 약해 운영(operating) 레짐을 정립하기 어렵다는 한계가 지적된다.

- **Core Contribution**: 본 논문은 양자 autoencoder(QAE)를 “압축을 거치되 정상 데이터에 대해 학습된 잠재표상(latent representation)을 얼마나 못 견디는지(incompressibility)”로 이상도를 정의하는 압축 기반 프레임워크로 제안한다. 이미지 패치를 quantum 상태로 매핑하기 위해 angle encoding을 쓰고, variational encoder-decoder를 trash qubit을 통해 정보 버리도록 학습한다. 이상 점수는 입력이 정상 데이터의 학습된 manifold에 비해 얼마나 압축에 저항하는지로 해석되며, 이는 재구성 성능만으로 설명하기 힘든 이상성을 제공한다.

- **Technical Challenges**: 주요 기술 과제는 정상 manifold에 맞춰 압축이 잘 일어나도록 학습을 유도하면서, 이상에서만 압축 저항성이 분명히 드러나게 만드는 것이다. 연구진은 encoder-decoder 비대칭이 나타나도록 훈련을 설계하고, 단순히 파라미터 크기 증가나 decoder 표현력 확대가 아니라 encoder 쪽의 구조적 정보 압축이 이상 탐지의 원인이 되도록 만든다. 또한 압축-복원 트레이드오프가 제어 가능한 범위(명확한 operating regime)를 형성하게 하여 임계값을 보다 근거 있게 선택할 수 있게 했다.

- **Empirical Impact**: 공개 brain MRI DICOM 데이터셋에서 slice-level ROC-AUC 약 0.95, patch-level ROC-AUC 약 0.813으로 classical autoencoder와 PCA baselines를 능가한다. 더불어 학습된 압축 저항성의 효과가 공간적으로 국소화된 anomaly heatmap으로 나타나, 종양 부위와 정렬되는 정성 결과를 보였다. 전반적으로 QAE가 “압축 동역학”을 해석 가능하고 제어 가능한 방식으로 의료영상 이상 탐지에 연결할 수 있음을 실증하며, 의사결정 지원 워크플로우에도 잠재력이 있음을 시사한다.



### Towards Evaluation of Implicit Software World Models in Coding LLMs (https://arxiv.org/abs/2606.27406)
Comments:
          Accepted to DL4Code workshop at ICML 2026

- **Prior Approaches**: 기존 코딩 LLM 평가는 HumanEval, MBPP 같은 함수 단위 테스트 통과 여부부터 SWE-Bench, Commit0 같은 저장소 단위 패치 성능까지 주로 ‘코드 작성/수정’ 능력을 측정해왔다. 실행 이해를 다루는 벤치마크들도 CRUXEval, REval, ThrowBench 등에서 대체로 짧은 파이썬 함수나 합성 프로그램에 초점이 맞춰져 있고, 반환값 같은 파생 지표 중심이라 실제 소프트웨어의 실행 맥락을 충분히 반영하지 못한다.

- **Core Contribution**: 이 논문은 소프트웨어 world model의 한 축인 ‘코드 실행’을 더 넓게 평가하기 위해 관찰 축을 제어 흐름(control flow)에서 실행 자원(execution resources)으로 이동한다. SWE-bench Verified 기반으로 라이브러리 단위 사례를 구성해 test outcome뿐 아니라 peak memory, wall-clock time, 그리고 method/line granularity에서의 profiler 랭킹을 예측하도록 설계한다.

- **Technical Challenges**: 가장 큰 어려움은 모델이 반환값을 넘어 실행 자원의 규모와 어떤 함수/라인이 실제로 비용을 유발하는지(실행 범위)까지 추론해야 한다는 점이다. 이를 위해 SWE-bench Docker 환경에 sys.settrace/sys.monitoring 기반 tracer를 삽입해 패치 전후로 ground truth를 수집하고, 로그 스케일 캘리브레이션(시간/메모리)과 순위 품질(Recall, NDCG)을 별도 메트릭 패밀리로 평가한다.

- **Empirical Impact**: 12개 모델(프론티어 API 모델과 open-weight, trace-trained CWM 포함) 모두 전반적으로 성능이 ‘겸손(modest)’하고 취약(brittle)해 실행을 얼마나 이해하는지에 한계가 드러난다. 특히 test outcome 예측은 recall이 낮고, wall time/peak memory는 모든 모델이 보편적으로 과대추정·범위 압축(slope<1) 경향을 보였으며, profiler 랭킹에서는 top 엔터티를 거의 찾지 못해 recall@5가 0.2에도 미치지 못했다.



### Automated brain tumor detection in MRI images using CNN and ResNet architectures (https://arxiv.org/abs/2606.27405)
- **Prior Approaches**: 뇌종양 진단은 MRI의 복잡한 뇌 구조 때문에 여전히 수기 판독 의존도가 높아, 초기·정확 진단을 자동화하기가 어렵다는 문제가 있었다. 기존 딥러닝 기반 의료영상 분류는 전이학습을 활용하더라도 데이터가 제한적인 상황에서 일반화 성능이 흔들릴 수 있다.

- **Core Contribution**: 이 논문은 MRI 영상을 입력으로 종양/비종양을 판별하는 자동화 딥러닝 프레임워크를 제안한다. 구체적으로 Convolutional Neural Networks 계열로 ResNet18, ResNet50에 transfer learning을 적용해 분류 성능을 비교하고, 임상 의사결정을 돕는 빠른 탐지를 목표로 한다.

- **Technical Challenges**: 핵심 기술 과제는 제한된 의료 데이터에서 모델 깊이에 따른 overfitting 위험을 줄이면서도 유의미한 특징을 잘 학습하는 것이다. 논문은 ResNet18과 ResNet50을 전이학습한 뒤 fine-tuning 전략을 실험하여, 더 얕은 모델이 오히려 더 안정적으로 일반화할 수 있음을 확인했다.

- **Empirical Impact**: 3,929장의 뇌 MRI 데이터에서 실험한 결과, ResNet18이 97% 정확도로 ResNet50의 96%보다 높은 성능을 보였다. 제한된 의료 데이터에서도 높은 일반화 성능을 보인다는 점에서, 조기 진단 지원과 비용·시간 절감 측면의 실용적 가치가 크다.



### SidConArena: An Environment Evaluating Agents in Open-Ended,Positive-Sum Bargaining Gam (https://arxiv.org/abs/2606.27397)
Comments:
          15 pages

- **Prior Approaches**: 기존 게임형 LLM 에이전트 평가는 주로 zero-sum 또는 적대적 경쟁에 치우쳐, 혼합동기 경제 상호작용의 복합 요구(협력으로 잉여 창출+희소자원 경쟁)를 충분히 드러내지 못했다. 또한 일부 협상/다중에이전트 벤치마크는 동적·부분관측·규칙 기반 평가를 모두 만족시키지 못해, 실제 에이전트의 가치평가·자원배분·장기계획을 진단하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 open-ended, positive-sum bargaining 상황에서 LLM 에이전트를 평가하도록 SidConArena를 제안한다. SidConArena는 유한한 지평의 부분관측 확률 게임(POSG) 구조를 바탕으로, 자연어 협상(구속 거래), 결정론적 converter 기반 생산, 밀봉경매(장기자산)를 한 프레임워크에서 결합한다.

- **Technical Challenges**: 핵심 과제는 자유로운 자연어 협상을 허용하면서도 거래·생산·경매 같은 수치적 행동을 게임 엔진 규칙에 엄밀히 고정(검증)하는 것이다. 이를 위해 위상(phase) 인식 dispatching, structured observation, neural-symbolic action interface, 비동기 이벤트 기반 실행을 결합해 문장 생성은 열어두되 실행은 규칙 검증과 동기화로 관리한다.

- **Empirical Impact**: 실험 결과, 동종 self-play과 이종 Elo tournament 모두에서 더 강한 frontier 모델이 더 높은 단말(terminal) 경제 성과를 보이면서 모델 간 구분 능력을 입증한다. 동시에 에이전트들은 자원 과대평가(예: Ships), 협상에서의 수동성, 그리고 delayed-return에 기반한 장기 투자 계획 실패 같은 반복적 한계를 보이며, 단순한 문법적 행동 적합성이 경제적 역량으로 이어지지 않음을 드러낸다.



### Agentic Publication Protocol: An Attempt to Modernize Scientific Publication (https://arxiv.org/abs/2606.27386)
Comments:
          16 pages, 5 figures and 1 table

- **Prior Approaches**: 기존 연구 출판은 정적인 논문 중심으로 설계돼 코드·데이터·환경·재현 절차가 부족하거나 업데이트가 어려운 문제가 반복됐다. 연구재현을 돕기 위한 Research Object, FAIR, Binder, 재현 체크리스트/아티팩트 평가, paper–code 링크 같은 인프라는 있었지만, 독자가 아닌 ‘에이전트’가 논문을 대신 실행·설명·검증하도록 하는 형태까지는 덜 정교했다. 그 결과 재현성과 노하우(어떤 설정이 민감한지, 무엇이 실패했는지)가 분산돼 후속연구 비용이 커졌다.

- **Core Contribution**: 이 논문은 Agentic Publication Protocol(APP)을 제안하며, 버전관리된 Git 저장소를 ‘출판 객체’로 삼아 논문 지식뿐 아니라 operational know-how까지 패키징한다. APP는 paper source, code, data, environment, reproducibility instructions, 그리고 에이전트용 AGENTS.md(지시 파일)를 한 릴리스 스냅샤로 묶어, 미래의 독자가 에이전트와 상호작용하며 재현·후속탐색을 할 수 있게 한다. 또한 선택적으로 agent skills를 포함해 작성자가 논문에 맞는 에이전트 동작을 더 잘 제공하도록 한다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 재현 가능한 핵심 결과를 최대한 ‘직접 실행 가능한’ 형태로 정리하면서, (2) 에이전트가 자료를 정확히 근거(artifact)로 삼아 말하도록 지침을 표준화하는 것이다. APP는 figures/tables 재현 매핑, 환경 재구성(dependency 파일/커맨드), ground truth와 보조자료의 구분, release+manifest로 검증 가능 스냅샤를 고정하는 구조적 요구사항을 통해 이를 해결한다. 또한 publish-paper 워크플로와 검증 단계(구조/프라이버시/경로/환경/사용성 점검)를 제공해 에이전트가 읽고 실행할 때의 불확실성을 줄이도록 설계했다.

- **Empirical Impact**: 개발 도구를 통해 예시 논문 11편(양자물리)을 대상으로 APP 기반 paper agent를 평가했고, 비교 기준으로 일반 저장소 인지 에이전트와의 대조가 수행됐다. 결과적으로 APP 에이전트는 11개 모두에서 우위를 보였고 평균 점수는 9.25로, 일반 에이전트의 8.50보다 높았다(특히 grounding과 honesty에서 큰 차이). 저자들은 이 프로토콜이 에이전트 시대의 분산형 ‘재현·설명 가능한 출판 단위’로 발전할 출발점이 될 수 있다고 주장한다. 



### CalBrief: A Pilot Diagnostic Benchmark for Evidence-Calibrated Scientific Briefing with Large Language Models (https://arxiv.org/abs/2606.27383)
- **Prior Approaches**: 기존 과학 문서 요약·리뷰 연구는 핵심 내용의 커버리지와 근거 인용에 초점을 두는 경우가 많지만, 새로 생성되는 결론이 실제 증거의 강도와 범위를 얼마나 정확히 반영하는지는 별도로 다루기 어려웠습니다. 또 claim verification이나 grounded generation은 특정 고정 문장의 지지/반증 또는 출처 근거를 보강하지만, 패키지 전체가 뒷받침하는 결론의 ‘강도·한계·누락 증거’까지 캘리브레이션하는 문제와는 결이 다릅니다.

- **Core Contribution**: 이 논문은 evidence-calibrated scientific briefing을 패키지 수준 과학 문서 이해 과제로 정식화하고, 결론별 evidence strength(강도)와 scope boundary(범위 한계), missing-evidence caveat(누락 증거 주의)를 함께 생성하도록 요구합니다. 또한 16개 이질적 과학 evidence package와 96개의 human-verified package-level takeaways로 구성된 verified pilot benchmark를 제공해, 중간 신호까지 포함한 평가를 가능하게 합니다.

- **Technical Challenges**: 핵심 기술 도전은 ‘감사 가능한(auditable) 증거 구조화’와 ‘강도 캘리브레이션’을 동시에 달성하면서, 결론이 증거 패키지의 범위를 벗어나 과장되지 않게 만드는 것입니다. 연구진은 CalBrief라는 role/gap/strength 프레임워크를 진단 도구로 사용해 실패 지점을 분해했고, 4-way 강도 라벨 공간({moderate, weak, uncertain, insufficient_evidence})이 모델을 과도하게 신중하게 만들며(대부분 원인), gap/scope 신호 주입 자체는 영향이 거의 없고 나머지는 파이프라인 정책 결합 비용에서 온다고 분리해 설명합니다.

- **Empirical Impact**: 실험에서 structured organization은 role 식별과 evidence-gap 관련성 등 ‘조직화 능력’은 개선하지만, strength calibration은 항상-moderate 기준선에 크게 못 미쳐 과도한 보수성 문제가 확인됩니다. 진단 결과, 닫힌 모델 3종(GPT-4o/Claude Sonnet/Gemini Flash)에서 라벨 공간 확장이 보수성 격차의 약 63%를 설명했고 신호 주입은 거의 기여하지 않았으며(비유의), 4-way 예측을 사후에 binary로 collapse하면 direct binary prompting과 동등하거나 때로는 더 나아져 라벨링 설계의 함의를 시사합니다.



### OverFlowLight: Real-Time Gridlock Prevention and Traffic Signal Optimization for Urban Intersections (https://arxiv.org/abs/2606.27381)
- **Prior Approaches**: 기존 TSC는 평상시 처리량(throughput)과 지체를 줄이는 데 초점이 맞춰져, 피크 시간대에 발생하는 intersection overflow를 명시적으로 다루기 어렵다. 압력 기반 MaxPressure나 DRL/FRAP 계열은 큐 길이·정체 지표를 최적화하지만, 출구 차로 용량 부족이나 차로 비대칭 상황에서 큐가 역류해 그리드락으로 번질 때 한계가 드러난다.

- **Core Contribution**: 논문은 OverFlowLight가 overflow를 ‘차로 레벨 실행 가능한 제어 문제’로 재정의하고, overflow가 감지되면 즉시 이를 해소하는 전용 신호 단계를 삽입해 그리드락을 선제적으로 차단하는 프레임워크를 제시한다. 특히 overflow phase map(OPM)을 중심 인터페이스로 두어, 기존 전통 제어기뿐 아니라 RL 기반 TSC back end와도 모듈형으로 결합 가능하게 만든다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 실시간으로 overflow 방향을 정확히 판별하고 (2) 그에 맞는 ‘안전한’ overflow-clearing phase 집합을 빠르게 구성한 뒤 (3) 제어기가 그 집합 안에서 신속히 선택하도록 하는 것이다. 이를 위해 카메라·레이더 멀티모달로 대기/저속 큐 혼잡 규칙을 적용해 overflow mask를 만들고, OPM이 허용한 위상만 행동공간에 제한해 RL이 장기 효율을 추구하면서도 안정적으로 수렴하도록 설계했다.

- **Empirical Impact**: 실제 도시 환경(3개 도시, 교차로 43개, 1.5년) 배치에서 OverFlowLight는 overflow incident를 60.4% 줄이고 네트워크 처리량을 18.2% 높였다. 또한 기존 고도화 에이전트에 이 프레임워크를 얹으면 overflow 전환 성공률이 대략 10%–13%에서 81%–100%으로 크게 상승했으며, 전문가 수동 개입 필요성도 줄여 실용적 확장성을 입증했다.



### Position: The Term "Machine Unlearning" Is Overused in LLMs (https://arxiv.org/abs/2606.27379)
Comments:
          13 pages; ICML 2026 Position Paper Track. Sangyeon Yoon and Yeachan Jun contributed equally

- **Prior Approaches**: 기존 LLM ‘unlearning’ 연구는 규제·저작권·안전 이슈로 인해 특정 데이터의 영향 제거가 요구된다는 공통 동기 아래, 거절(refusal), 억제(suppression), 편집(editing), 난독화/혼란(obfuscation), 추론 시 차단(guardrails/filters) 등 매우 다른 목표를 한 용어로 묶어 왔습니다. 이 때문에 많은 벤치마크가 ‘forget set’에 대해 정답 재현 실패 같은 출력 중심 지표만 측정하며, 재학습 기준선과의 유사성(훈련 영향 제거의 보장)을 검증하지 않는 경우가 잦았습니다.

- **Core Contribution**: 이 논문은 ‘machine unlearning’을 dataset-defined deletion으로 한정해, forget set F의 ‘학습 영향’을 정확히(또는 근사적으로) 제거했을 때 retraining-from-scratch(데이터 D\F로 재학습) 모델과 동등해지는 상태로 정의합니다. 동시에, 거절·지식/엔터티 제거·표적 억제 등 정책 의존적 행동 변경은 별도 용어(예: alignment, suppression, editing, obfuscation 등)로 분리해야 한다고 주장합니다. 핵심은 라벨 혼동이 평가 해석을 망치며, 실제로는 ‘훈련 영향 제거’가 아니라 ‘출력 통제’가 성공 신호로 보상될 수 있다는 점을 정리한 데 있습니다.

- **Technical Challenges**: 정확한 unlearning의 조건은 대규모 모델에서 현실적으로 매우 강해, 논문은 (approximate) indistinguishability처럼 명시적 거리/분산 기준(행동 공간 또는 파라미터 공간)을 둔 완화 정의를 제시합니다. 기술적으로는 forget set 제거의 기준을 출력 실패가 아니라 ‘훈련 없이 학습했을 때(reference)와의 분포·확률·행동 유사성’으로 고정해야 하며, 특히 파생 능력(derived capability)이 남아 있을 수 있다는 점을 포착하는 평가 설계가 필요하다고 강조합니다. 따라서 단순 ROUGE/정확도/문구 차단만으로는 부족하고, 기준선 대비 분포 비교와 강건성(패러프레이즈·적대적 elicitation 등)까지 함께 보아야 한다고 말합니다.

- **Empirical Impact**: 논문은 기존 벤치마크들이 ‘재학습 기준선’ 없이 출력 기반 지표로 성공을 판정하면서, 억제·거절이 실제로는 unlearning이 아닌데도 높은 점수를 받는 문제를 여러 예(TOFU 계열의 기준선 생략, RWKU의 기준선 부재, WMDP의 능력 억제 프레이밍 등)로 구체화합니다. 또한 파생 능력 관점에서 ‘답하지 않음’이 훈련 영향 제거의 대리(proxy)가 될 수 없고, 특히 poisoning 같은 설정에서는 유도된 행동이 남는지를 봐야 평가가 의미를 갖는다고 주장합니다. 결론적으로, ‘참조 모델(reference-based) + 파생 능력 프로브 + 분포/확률 기반 비교’로 평가 관행을 재정렬하자는 제안이 분야의 후속 연구 설계에 직접적인 방향성을 줄 것으로 보입니다.



### DataStates-LLM: Scalable Checkpointing for Transformer Models Using Composable State Providers (https://arxiv.org/abs/2601.16956)
- **Prior Approaches**: 기존 LLM 체크포인팅은 모델 상태를 이진 블롭처럼 취급하거나, 텐서 위주로 직렬화/전송을 최적화하는 경향이 강했다. 이 방식은 GPU vs Host, 텐서 vs Python 객체, 다수 파일로 쪼개진 샤드 등 ‘3D heterogeneity’를 통째로 무시해, 불필요한 직렬화와 메타데이터/I/O 병목을 키웠다. 또한 다단계 비동기 플러시를 쓰더라도 GPU→Host 복사가 병목이 되어 학습 경로를 충분히 가리지 못하는 문제가 반복됐다.

- **Core Contribution**: DataStates-LLM은 State Providers라는 미들웨어로 체크포인트 상태의 추상화(무엇을 캡처하나)와 데이터 이동(어디로/어떻게 옮기나)을 분리한다. 특히 학습 반복의 forward/backward 동안 모델 파라미터와 옵티마이저 상태가 불변이라는 성질을 활용해 ‘lazy’, non-blocking 비동기 스냅샷을 수행한다. 그 결과, 이기종 상태를 의미 단위로 조합하면서도 전체 체크포인트의 전역 일관성을 유지하도록 설계했다.

- **Technical Challenges**: 핵심 난제는 체크포인트에 섞인 이기종 데이터(텐서 버퍼, FP32/FP16 텐서, Host 상의 Python 객체와 메타데이터)를 한 번에 모으되, 직렬화 비용과 장치-호스트 전송(PCIe) 지연을 최소화하는 것이다. 논문은 텐서는 zero-copy에 가깝게 스트리밍하고 Python 객체는 State Provider가 요구하는 방식으로만 효율적으로 처리하며, 메타데이터 생성/직렬화를 텐서 I/O와 겹치게 재배열한다. 또한 pinned host 메모리 풀과 파이프라인 I/O 엔진으로 flush를 비동기 스트리밍하며 파일/메타데이터 병목을 줄였다.

- **Empirical Impact**: 256 A100-40GB(최대 70B)에서 실험한 결과 DataStates-LLM은 체크포인팅 처리량을 최대 4배까지 높이고, 학습 전체 종료시간을 최대 2.2배 단축했다고 보고한다. TorchSnapshot 대비 직렬화·이기종성 병목을 실질적으로 완화해, 극단적 스케일에서 체크포인트가 학습을 가로막는 시간을 크게 줄였다는 점이 의미가 있다. 이는 복원력뿐 아니라 고빈도 롤백/불안정성 대응, RLHF 같은 반복형 워크플로우에서 체크포인트 민감도를 낮출 수 있는 방향을 제시한다.



### What the LLM Should Not Say: Boundary-Aware Context Grounding for A Seven-Channel EEG Agen (https://arxiv.org/abs/2606.26519)
Comments:
          25 pages, 6 figures

- **Prior Approaches**: 기존 EEG 분석 도구(MNE-Python, EEGLAB 등)는 기능 범위가 넓지만, 사용자가 채널/프로토콜/메서드 선택을 직접 책임져야 해서 “기기와 구현이 허용하는 경계”를 에이전트가 자동으로 보장하긴 어렵다. LLM 기반 접근은 RAG나 tool-use로 일부 근거를 붙일 수 있지만, 과학 에이전트에서는 단순 사실 정확도만으로는 안전을 담보하기 어렵고, 특히 over-refusal(과잉 거절)도 같은 실패 양상으로 다뤄야 한다.

- **Core Contribution**: NeuraDock Agent는 로컬의 결정론적(deterministic) EEG 수치 엔진을 “수치적 진실”로 고정하고, LLM은 허용된(allowlisted) 요약과 버전화된 context pack을 받아 해석·계획만 수행하는 구조를 제안한다. 이를 통해 저채널 EEG에서 흔한 “그럴듯하지만 근거 없는 해석”을 막기 위해 물리적 경계(관측 가능), 구현 경계(현재 구현된 워크플로), 결과 경계(출력 필드의 의미), 과학적 경계(정당화되는 추론)를 분리해 다룬다. 또한 raw EEG나 조밀 배열(dense per-sample/PSD 등)을 외부 모델로 보내지 않고 로컬에 남기는 데이터 최소화 메커니즘을 포함한다.

- **Technical Challenges**: 핵심 기술 과제는 LLM이 요청을 수행할 때 필터/임계값/특징/통계검정을 “조용히 바꿔” 잘못된 결론을 만들지 않도록 경계를 강제하는 것이었다. 논문은(1) 로컬 Python 워크플로 레지스트리를 닫힌 상태로 유지해 새로운 분석 방법은 코드·스키마·문서 업데이트 없이는 선택 불가하게 하고, (2) LLM 호출에는 raw_eeg_included=false와 함께 허용된 결과 필드만 투영(projection)하며, (3) HTTP/출력 손상/연결 실패 상황에서도 로컬 아티팩트(report, trace, 결과)가 보존되도록 실패 격리를 구현했다.

- **Empirical Impact**: 실험에서는 12개 레코딩의 반복 실행에서 구조화 결과 해시가 동일해 수치 엔진의 재현성이 확인됐고, request-capture/실패 주입 실험으로 경계 위반 없이 로컬 아티팩트가 유지됨을 검증했다. 36개 케이스 경계 인식 벤치마크에서 전체 context를 적용했을 때 exact 4-way decision 정확도와 strict safe-response 비율이(Generic 대비) 크게 상승했으며, “무조건 거절”이 아니라 언제 accept/qualify/refuse 해야 하는지 보정(calibration)이 성능 향상의 중심임을 보여준다. 다만 flatline과 같은 일부 부정(artifact) 유형은 현재 워크플로에서 명시적으로 감지되지 않는 등, 임상 유효성이나 절대적 인덱스 확립이 아닌 시스템·평가 프레임워크 성격임을 분명히 한다.



### DiARC: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models (https://arxiv.org/abs/2606.26530)
- **Prior Approaches**: 기존 ARC(Abstraction and Reasoning Corpus) 접근은 정답 input-output만을 늘리는 데이터 증강·합성 중심이 많았습니다. 또한 일부는 재귀 추론(후속 상태 반복 개선), 비전 모델로의 전환, LLM용 프롬프트/표현 재설계로 성능을 끌어올리지만, 공통적으로는 그럴듯한 오답을 구분하는 신호가 약했습니다.

- **Core Contribution**: 이 논문은 ARC-like 문제 해결이 “정답을 맞히는 것”에 더해 “그럴듯하지만 규칙이 다른 near-miss를 식별하고 거절”할 수 있어야 한다고 주장합니다. 이를 위해 preference alignment 관점에서 정답/오답 선호쌍을 만드는 DiARC를 제안하고, DPO로 두 출력 간 상대 선호를 학습시킵니다.

- **Technical Challenges**: 핵심 난제는 정보가 충분한 negative sample(오답 후보)을 만들되, 관측된 support 시연은 유지하면서도 모델이 구분할 수 있게 ‘가까운 실수’를 설계하는 것입니다. DiARC는 (1) 출력 격자 공간 시각 변환, (2) DSL 수준 rule inversion, (3) 작업별 transformation rule editing의 세 단계로 near-miss를 생성한 뒤, DPO 학습으로 정답 출력의 상대 likelihood를 높이는 방식으로 이를 해결합니다.

- **Empirical Impact**: 여섯 개 ARC 벤치마크에서 DiARC는 3종 오픈소스 LLM 전반에 걸쳐 기준선 SFT보다 일관된 성능 향상을 보였습니다. 특히 Qwen3-4B는 ARC-AGI-1, MiniARC, ConceptARC에서 96%대 정확도를 달성하며 closed-source 및 기존 오픈소스 방법을 능가했고, 생성(generation)과 선택(discrimination) 두 단계에서 이득이 함께 발생함을 분석으로 확인했습니다.



New uploads on arXiv(cs.RO)

### DexCompose: Reusing Dexterous Policies for Multi-Task Manipulation with a Single Hand (https://arxiv.org/abs/2606.28323)
Comments:
          Project page: this https URL

- **Prior Approaches**: 기존에는 (1) 재사용 가능한 dexterous 컨트롤러를 새로 학습하거나, (2) 여러 스킬을 긴 구간으로 순차 결합하며, (3) 손의 자유도를 손가락/자원으로 나눠 멀티 상호작용을 만드는 연구가 많았습니다. 그러나 이미 학습된 full-hand 정책을 그대로 재사용하면서 두 과제를 동시에 수행하도록 안전하게 결합하는 방법은 제한적이어서, 단순 policy chaining이 공용 손가락 동작을 덮어쓰며 성능이 쉽게 무너집니다.

- **Core Contribution**: 이 논문은 DexCompose로, pretrained dexterous 정책 2개를 결합할 때 손가락 단위 action ownership을 명시적으로 부여해 cross-task interference를 줄이는 프레임워크를 제안합니다. 또한 Task A(기존 유지 과제)를 보존하는 residual stabilizer와, Task B(새 상호작용 과제)를 위해 할당된 하위 action subspace에서만 적응하는 context-aware residual을 이중으로 학습/적용합니다.

- **Technical Challenges**: 핵심 기술 과제는 (i) 어떤 손가락(DoF)이 Task A의 결과를 유지하는 데 실제로 필요한지 자동으로 찾아내고, (ii) 이후 Task B가 그 필수 손가락을 건드리지 않으면서도 필요한 경우에만 충분히 보정하도록 제어 범위를 제한하는 것입니다. DexCompose는 Task A의 성공 후 상태를 모은 뒤 release tests로 손가락 마스크를 후처리(discovery)하고, 그 마스크에 따라 잔차(residual)를 bounded stabilizer/할당된 subspace용 residual로 분리해 학습합니다.

- **Empirical Impact**: 시뮬레이션에서 4가지 hold-and-retain 과제와 4가지 downstream 상호작용을 조합해 총 16개 composite task를 평가했으며, 평균 composite success rate 77.4%를 달성했습니다. 이는 policy chaining 대비 15.8%p 큰 개선이며, 특히 dual residual stabilizer가 유지 실패를 막는 데 가장 큰 역할을 했고 finger attribution은 동일 예산에서 추가적인 간섭 감소 효과를 보였습니다.



### WARP-RM: A Warp-Augmented Relative Progress Reward Model for Data Curation (https://arxiv.org/abs/2606.28320)
- **Prior Approaches**: 기존에는 긴 호라이즌 로봇 모방학습에서 데모 품질 민감도를 낮추기 위해 에피소드 단위로 낮은 품질을 버리거나(episode-level filtering), 프레임 단위 progress 신호를 학습해 국소 구간을 고르는 방식이 쓰였다. 그러나 절대적 시간/진행도(예: 정규화된 경과 타임스텝)나 전역 정렬 기반 감독은 텔레오퍼레이션의 멈춤·재시도·회복 때문에 라벨 노이즈가 커진다. 프레임 단위 dense reward 모델도 사람 라벨이 필요하거나, 시계열 해상도를 희생하는 접근이 많아 확장성에 제약이 있다.

- **Core Contribution**: 이 논문은 WARP(Warp-Augmented Relative Progress)로, 성공 데모만으로부터 ‘국소 relative progress(부호 포함 진행 속도)’를 완전 self-supervised로 학습하는 방법을 제안한다. time-warp augmentation(재생 속도 변형 및 역재생 포함)으로 데모를 비선형/비단조 재생한 뒤, 입력 프레임 사이의 정규화된 elapsed time 델타를 예측하게 하여 프레임 레벨 진행 신호를 얻는다. 이후 WARP-BC는 이 스칼라 진행 추정치를 행동모방에서 action chunk의 advantage에 비례하도록 가중·필터링해 비효율 구간이 학습을 망치는 것을 막는다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 텔레오퍼레이션의 동일 프레임 인덱스가 실제로는 다른 작업 상태일 수 있어 절대 progress 감독이 흔들린다는 점, (2) 그럼에도 사람 라벨 없이 dense하고 부호 있는 진행 신호를 만들어야 한다는 점이다. WARP는 AR(1) 기반으로 속도 변화를 시간적으로 매끈하게 샘플링하고, reversal과 전체 역방향 재생을 섞어 정방향/역방향 진행을 동시에 학습한다. 또한 categorical two-hot 목표와 윈도우 중첩 집계를 통해 프레임 단위 velocity를 안정적으로 산출하고, 행동모방에서는 terminal-frame velocity로 chunk 경계를 더 날카롭게 구분해 재가중한다.

- **Empirical Impact**: 실험은 변형 가능한 물체 긴 호라이즌 조작(크럼블된 티셔츠를 폴딩)에서 물리 이족(양손) 로봇으로 420개 실주행을 평가한다. 학습 데이터가 비효율 데모를 더 포함하도록 확장될 때 vanilla BC는 성공률이 급락해 2/20으로 붕괴한 반면, WARP-BC는 19/20 수준의 견고성을 유지하며 성공 throughput은 최대 약 18배까지 향상됐다. 즉, 텔레오퍼레이션의 품질 혼재 환경에서 ‘프레임 레벨 relative progress’ 기반 모방학습이 실제 생산성(처리량)을 크게 끌어올릴 수 있음을 보여준다.



### CacheMPC: Certified Cached Model Predictive Control for Quadruped Locomotion (https://arxiv.org/abs/2606.28300)
- **Prior Approaches**: 다리 로보틱스의 계층형 제어에서 MPC는 흔히 표준 예측 계층이지만, 임베디드에서 매 틱 QP를 풀어야 해 갱신 주기가 제한된다. 기존에는 real-time iteration NMPC, TinyMPC처럼 계산을 줄이거나(팩터라이제이션/조건 선형화), warm-start로 반복 횟수만 줄이는 방식이 주로 쓰였고, 정확성 보장은 제한적이었다. 또 기존 caching 아이디어는 있었지만, ‘재사용 제안이 현재 문제에서도 안전/최적성 조건을 만족하는지’에 대한 쿼리 단위 보증은 부족했다.

- **Core Contribution**: 이 논문은 Certified CacheMPC로, LSH 기반 캐시에 저장된 ‘수평 접촉력(horizon contact-force) 궤적’을 contact mode별로 분할해 재사용하되, 수락은 각 쿼리의 a-posteriori certificate로만 한다. certificate는 재사용된 제안이 현재 MPC 문제에서 (i) primal feasibility를 만족하고 (ii) Lagrangian dual-gap을 통해 비용 suboptimality를 상한으로 제한함을 확인해, 캐시가 휴리스틱 후보를 넘어 검증된 로컬 해가 되도록 만든다.

- **Technical Challenges**: 핵심 난제는 비슷한 상태에서 찾은 이웃 제어가 active-set 경계를 넘어가면 구조적으로 전혀 다른 해 영역에 들어갈 수 있다는 점이다. 저자는 LSH는 후보 생성용으로만 쓰고, 실제 수락 여부는 현재 QP의 KKT 자료를 사용해 dual-gap 기반 상한(예산 튜브)을 만족할 때만 결정하도록 설계했으며, 시간 제한이 걸리는 환경을 위해 top-K certified retrieval + deadline-bounded QP solve + last-certified fallback의 bounded-budget 스케줄을 둔다.

- **Empirical Impact**: Unitree Go2와 NVIDIA Orin NX에서 Unitree cold-controller MuJoCo 2,038회(추가로 실패 경계 3개 셀에서 n=50 캠페인 포함)와 on-robot 첫 배포를 통해 평가했으며, 캐시 게이팅 없는 방식은 시뮬레이션에서 median solve-time을 25배, 하드웨어에서 median 속도를 18.7배 가속했다. 한편 n=50에서는 어떤 셀에서도 closed-loop stable rate가 캐시 변형과 no-cache baseline 사이에 통계적으로 유의미한 차이를 보이지 않았고, certificate가 닫힌 루프 안전에 미친 기여는 현재 표본 크기에서는 해석이 어려웠다고 보고한다.



### SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation (https://arxiv.org/abs/2606.28276)
- **Prior Approaches**: 기존 real-to-sim 자동화는 주로 단순 pick-and-place나 단일 step에 치우쳐 있고, 캡처한 장면을 정적 디지털 트윈으로 고정해 데이터 다양성이 제한되는 경우가 많았습니다. 또한 시뮬레이션이 실제 성능과 상관되는지 평가하는 데는 강점이 있어도, 학습까지 실전 에이전트로 연결하면서 관절형·양팔·멀티스텝 조작을 포괄하는 파이프라인은 드뭅니다.

- **Core Contribution**: SimFoundry는 영상 1개로부터 zero-shot real-to-sim scene construction을 수행해 sim-ready 디지털 트윈(오브젝트·배경·관절 구조)을 자동 생성합니다. 특히 reconstructed scene의 affordance를 보존하는 object/scene/task “digital cousins”를 만들어, 시뮬레이션에서 만든 다양성이 실제 환경 일반화로 이어지도록 설계했습니다.

- **Technical Challenges**: 핵심 난제는 (1) 단안 영상 기반 복원에서 geometry·스케일·포즈가 흔들리고 (2) 관절 오브젝트 분할/조인트 파라미터를 안정적으로 추정하며 (3) 물리 시뮬레이션에서 안정적인 초기 상태를 보장하는 것입니다. SimFoundry는 다양한 foundation model을 모듈로 결합해 재구성과 편집을 자동화하고, PyBullet에서 안정화(정착까지 스텝) 및 물리적 depenetration을 수행하며, 추가로 3D Gaussian Splat 배경은 정렬 불일치 완화를 위한 per-camera pose optimizer로 선명도를 확보합니다.

- **Empirical Impact**: 7개 조작 태스크와 5개 policy 아키텍처에서, SimFoundry 데이터로 학습한 시뮬레이션 평가는 실제 성능을 강하게 예측(평균 Pearson 상관 0.911, 평균 최대 ranking violation 0.018)했습니다. 또한 real에서 zero-shot 평가할 때 object/scene/task cousins를 함께 쓴 정책의 평균 성공률이 각각 17%, 21%, 40% 향상됐고, simulaiton-trained 디지털 시나리오 변형이 새로운 조건으로의 일반화에 실질적 도움을 준다는 점을 보여줍니다.



### Unleashing Infinite Motion: Scaling Expressive Quadrupedal Motion via Generative Video Priors (https://arxiv.org/abs/2606.28237)
- **Prior Approaches**: 기존 사족(Quadruped) 로봇의 데이터/학습 파이프라인은 동물의 몸을 거치는 가정에 기대왔다. 이 방식은 동물 협조에 의존해 수집이 어렵고, 종/품종마다 복원(3D reconstruction)과 외형 변형이 커서 종 간(동물→로봇) 재타겟팅이 물리적으로 불안정해진다.

- **Core Contribution**: 이 논문은 Uni-Mo로 동물-중간(intermediate) 가정을 제거하고, 자연어 프롬프트→비디오 생성→3D 기준 궤적 생성→추적 정책 학습의 end-to-end 파이프라인을 제안한다. 핵심은 Identity Consistency Loss로 비디오 확산 모델에서 생기는 identity drift(프레임 간 비정상적인 변형/외형 변화)를 억제해, 생성 영상을 로봇 운동 궤적으로 안정 변환할 수 있게 한 점이다.

- **Technical Challenges**: 가장 큰 난제는 비디오 diffusion 생성이 장시간에 걸쳐 로봇의 정체성과 비체적(강체) 구조를 흐리게 만들어 3D 추출이 깨진다는 점이다. 연구진은 DINOv2 기반 identity descriptor와 nearest-reference hinge를 사용해 프레임별 외형/구조 일관성을 강제하는 Identity Consistency Loss를 도입했고, 생성 영상은 CLIP 의미 게이트와 기하(재투영 오차) 게이트 및 시뮬레이션 추적 가능성 게이트로 걸러낸다.

- **Empirical Impact**: Quad-Imaginarium(Quad-Imaginarium dataset)은 7,488개 언어 라벨 사족 모션(총 18.5시간)을 오픈소스로 공개하며, 로봇 Go2에서 재현 가능한 표현적 행동의 폭을 크게 확장한다. 정책의 실제 배치 성공률은 96.7%(Go2, 392개 샘플)이고 시뮬레이션에서도 97.6%로, 사족 모션 획득이 ‘동물 협조’가 아니라 ‘생성 compute’에 비례해 확장될 수 있음을 실증한다.



### Learning Stable In-Grasp Manipulation in a Non-Dropping Action Spac (https://arxiv.org/abs/2606.28196)
Comments:
          This work has been submitted to the IEEE for possible publication

- **Prior Approaches**: 기존 덱스터러스 조작은 강한 물리 가정을 전제로 한 해석적 모델(손가락/물체/접촉 상호작용 정확 모델링)이 정확도에 민감했다. 반대로 end-to-end reinforcement learning은 복잡한 모델 의존을 줄이지만, long-tail “불안정 행동”이 학습 중 잘 드러나지 않아 결국 물체가 떨어지는 문제가 생겨 학습 효율이 낮아졌다. 또한 불안정성을 reward shaping이나 알고리즘 변경으로 완화해도, 실패 원인을 특정하기 어렵고 목적이 충돌하기 쉽다.

- **Core Contribution**: 이 논문은 덱스터러스 기술을 안정적으로 유지되는 in-grasp 조작 성분들로 분해한 뒤, 각 성분을 고전 물리/제어이론으로 제약하며 학습하는 TSIGL(이론 기반 안정 in-grasp 학습) 프레임워크를 제안한다. 특히 FTODG(자세/힘의 안정성을 보장하는 손-물체 제어 이론)에서 정의한 stable action space 안에서 RL이 탐색하도록, 안정성에 기반한 행동 공간을 먼저 만든다. 그 위에 control barrier functions(CBFs)로 “손가락 접촉 유지 및 과도한 힘 회피” 같은 구현 제약을 하드하게 걸어 drop과 불안정 손가락 움직임을 줄인다.

- **Technical Challenges**: 핵심 난제는 RL이 탐색하는 행동이 FTODG의 안정성 가정(접촉 유지, 힘/토크 범위, 자주 바뀌지 않는 제어 등)을 위반할 수 있다는 점이다. 이를 해결하기 위해 RL의 행동을 손가락의 힘/모멘트를 만드는 제어 이득(예: 위치/방향 재배치 성분의 O,P 파라미터)으로 직접 정의하고, CBF 기반 필터를 통해 선형 부등식 형태의 안전 제약(접촉 파손 방지, 힘 상·하한)을 만족하도록 행동을 즉시 보정한다. FTODG의 안정 제어는 Newton/Lagrangian 및 passivity 기반 제어 논리로 뒷받침되며, 가짜/불완전 관측(노이즈·지연)을 고려해 외부센서 없이도 동작할 수 있게 설계한다.

- **Empirical Impact**: 평가에서는 NVIDIA Isaac Lab 시뮬레이션에서 96개 병렬 인스턴스로 PPO 기반 정책을 학습하며, 3-finger precision grasp, 3-finger VF(external sensorless) position manipulation, 4-finger object orientation manipulation을 검증한다. 손가락-물체 마찰/손가락 파라미터, 관측/액션 지연 및 잡음을 랜덤화해도 TSIGL이 baseline RL보다 안정적으로 물체를 유지하며 목표 상태 달성 정확도를 높였다고 보고한다. 또한 ablation에서 CBF 필터 등 구현 제약을 완화하면 연속 성공 성능이 떨어져, “이론 기반 안정 행동 공간 + CBF 하드 제약”이 샘플 효율과 안전성에 직접 기여함을 실증한다.



### PA-BiCoop: A Primary-Auxiliary Cooperative Framework for General Bimanual Manipulation (https://arxiv.org/abs/2606.28192)
Comments:
          ICRA2026

- **Prior Approaches**: 기존 bimanual 조작 연구는 크게 (1) 두 개의 독립 모델로 각 팔을 따로 예측하는 dual-model과, (2) 하나의 공유 모델로 양팔을 동시에 제어하는 single-model로 나뉜다. dual-model은 왼팔-오른팔 간 지식 공유가 약하고, 모델 복잡도가 커지는 한계가 있으며, single-model은 두 팔을 역할 구분 없이 동일하게 다뤄(또는 수동 시퀀스에 의존해) 동적 협응이 떨어진다.

- **Core Contribution**: 이 논문은 사람의 양팔 분업(주도 팔-보조 팔, 그리고 작업 단계에 따른 역할 전환)을 모사해 단일 모델 기반 PA-BiCoop을 제안한다. 핵심은 양팔을 primary/auxiliary로 나누고, 두 디코더(공유 전역 인코더 사용)를 통해 주도 팔의 핵심 affordance와 보조 팔의 상대 pose를 함께 예측하며, role assignment 모듈이 좌/우 팔에 primary·auxiliary 역할을 자동으로 매핑한다.

- **Technical Challenges**: 역할 분업을 학습 가능한 구조로 구현하는 것이 핵심 난제다. 주도 팔 기준 좌표계에서 보조 팔을 예측해 학습 부담을 줄이고(상대 pose로 안정성 확보), 보조 arm의 회전에 대해서는 원형(circular) MSE를 도입해 각도 주기성으로 인한 학습 실패를 완화했으며, 역할 전환은 관측·언어 토큰에 대한 cross-attention을 통해 이진 분류로 결정한다.

- **Empirical Impact**: RLBench2 시뮬레이션 10개 언어 조건 작업에서 PA-BiCoop은 평균 48% 성능 향상을 보였고, 실제 로봇 환경의 두 작업에서도 평균 50% 이상 개선됐다. 특히 동기/비동기, 대칭/비대칭, 그리고 long-horizon 구간에서 기존 shared-model 대비 성공률 격차가 크게 나타나며, 역할 인식이 조정 효율과 적응성을 실제로 끌어올린다는 점을 실험적으로 확인했다.



### Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots (https://arxiv.org/abs/2606.28133)
Comments:
          Project Page: this https URL

- **Prior Approaches**: 기존 연구는 사람의 동작을 손 자세 추정(hand-pose estimators) 기반의 6DoF(회전 포함) 엔드이펙터 동작으로 보고, 로봇 학습에서도 이를 그대로 재현하려는 경향이 강했습니다. 하지만 병렬 그리퍼 로봇과 인간 손가락은 접촉 패턴이 근본적으로 달라 회전 신호가 조작 의미와 어긋나며, 추정 오차로 인해 잡음이 크게 유입된다는 한계가 있습니다.

- **Core Contribution**: 이 논문은 사람-로봇 간 공통으로 쓸 수 있는 ‘브리징 동작 표현(bridging action representation)’을 제안합니다. 구체적으로 초기 헤드 카메라 프레임 기준의 손목 상대 ‘이동(translation)’만을 학습 신호로 삼아, 회전이 불안정한 인체 데이터의 약점을 우회하고 로봇에 이식 가능한 동작 공간을 만듭니다.

- **Technical Challenges**: 문제는 서로 다른 데이터 소스(인간/로봇)에서 동작 구성요소가 누락될 수 있다는 점입니다. 이를 위해 π0-like vision-language-action 모델에 interleaved action tokens와 attention masking을 도입해 a3D-wrist→a6D-eef→agripper 순서로 토큰을 배치하고, 손목 이동 신호가 로봇의 실행 가능한 엔드이펙터 동작(a6D-eef)으로 안정적으로 연결되도록 학습 목표를 설계합니다.

- **Empirical Impact**: 여러 ‘새로운 bi-manual 조작 태스크’에서, 제안된 브리징 표현은 잡음이 섞인 6DoF 인간 동작을 그대로 쓰는 방식보다 로봇 기술 전이를 훨씬 더 잘 수행했으며 인간 데이터의 양이 늘어날수록 성능이 확장되는 경향을 보였습니다. 또한 일부 태스크에서는 로봇에 대한 task-specific 시연 없이도 로봇이 완료할 수 있음을 실험적으로 제시해, 대규모 인간 동작 데이터 기반 스케일업 가능성을 시사합니다.



### Building a Scalable, Reproducible, Evaluatable, and Closed-Loop Simulation Environment Foundation for Embodied Intelligence Cloud-Native Simulation Infrastructure for Embodied Intelligence Training, Evaluation, and Data Collection (https://arxiv.org/abs/2606.27962)
- **Prior Approaches**: 기존 연구는 시뮬레이터(예: Isaac Sim, MuJoCo, SAPIEN) 자체의 성능이나 특정 벤치마크/데이터셋 품질에 초점을 맞춰, 훈련·평가·데이터 수집·관리·재현성까지 아우르는 ‘플랫폼’ 관점이 약했다. 또한 실제 로봇 데이터 수집은 비용·속도·재현성·장기테일 실패 커버리지에서 한계가 있어, 반복 실험과 비교 가능한 평가를 대규모로 수행하기 어렵다. 그 결과 모델 반복 주기와 실패 기반 데이터 보강을 체계적으로 묶기 힘들었다.

- **Core Contribution**: 이 논문은 embodied intelligence를 위한 cloud-native 시뮬레이션 인프라 프레임워크를 제안하며, 환경 생성부터 작업 실행, 궤적 수집, 모델 평가, 데이터 관리, closed-loop 최적화까지를 하나로 통합한다. 단일 시뮬레이터나 단발성 데이터 생산이 아니라, 대규모·표준화·재현 가능한 데이터 생성과 평가, 그리고 실세계 배포로 이어지는 장기 기반을 목표로 한다. 또한 D-VLA, RL-VLA3, Sword, Pre-VLA 등 대표 시스템을 연동해 멀티모델/멀티태스크 워크로드를 확장 지원한다.

- **Technical Challenges**: 핵심 과제는 (1) 대규모 병렬 시뮬레이션을 비용 효율적으로 돌리고 (2) 멀티모달·시간 연속·관계형 데이터의 품질과 메타데이터를 일관되게 관리하며 (3) 실험 조건·버전·랜덤시드까지 포함해 재현 가능 평가를 자동화하는 것이다. 논문은 elastic resource scheduling, 컨테이너 기반 시뮬레이션, 통합 데이터 관리, 서비스 지향 설계를 통해 이를 해결하고, 4계층 구조로 환경 자산 표준화·자동 태스크 생성·궤적 수집·벤치마크 평가·closed-loop 데이터 최적화를 제공한다. 실패/이상 케이스를 보존하고 유사·교란 샘플을 생성해 반복 학습으로 연결하는 데이터 파이프라인도 내장한다.

- **Empirical Impact**: 현재 단계로는 Isaac Sim 5.1 환경에서 특정 태스크의 실행과 궤적 수집을 정상 완료해 ‘새 고정밀 시뮬레이션’의 데이터 링크를 열었다. 저비용·고확장·고재현성 시뮬레이션 기반 데이터 수집이 가능해지면, 모델 버전 간 비교와 실패 재현·보강이 빨라져 R&D 반복 속도와 평가 신뢰도가 높아질 것으로 기대된다. 결과적으로 시뮬레이션을 데이터 생성·훈련·표준화 평가·실세계 배포 전 단계까지 연결하는 공통 기반으로 자리잡는 데 의미가 있다.



### When Multi-Robot Systems Meet Agentic AI:Towards Embodied Collective Intelligenc (https://arxiv.org/abs/2606.27929)
- **Prior Approaches**: 기존 embodied AI는 인식-제어(perception–control) 파이프라인 중심으로, 각 에피소드가 초기화되는 원샷 환경에서 성능을 측정해 왔다. 멀티로봇 연구도 태스크 할당, 맵 퓨전, 데이터/지식 공유 등으로 협력을 확장했지만, 로봇이 실행 중 업데이트하는 ‘에이전트 루프 상태(기억·계획·실패 맥락)’까지 실시간으로 공유하진 못했다.

- **Core Contribution**: 이 논문은 Embodied Collective Intelligence(ECI)를 제안하며, 로봇 팀이 월드 컨텍스트·태스크 진행·스킬 경험을 ‘공유 자원’으로 축적·활용하는 미래 멀티로봇 패러다임을 제시한다. ECI는 중앙 슈퍼에이전트 결합이 아니라, Co-Perception(팀 월드 메모리), Co-Action(태스크 상태 레저), Co-Evolution(스킬 라이브러리)로 ‘상태(state) 공유’를 설계한다.

- **Technical Challenges**: 핵심 난제는 서로 다른 로봇이 만들어내는 관측과 실행 결과가 동적으로 변할 때, 무엇을 기록·검증·폐기(신선도 관리)하며 어떻게 상속 가능한 형태로 저장할지다. 논문은 공통 레이어를 팀 메모리/태스크 레저/스킬 기록으로 두고, 로봇별 안전과 저수준 제어는 로컬에 유지하면서 공유 기록에 타임스탬프·관측 주체·신선도 같은 메타데이터를 포함시키는 구조를 제안한다.

- **Empirical Impact**: 사례 연구로는 Co-Evolution·Co-Action을 제외하고 Co-Perception의 ‘공유 월드 메모리 상속’만 실측했다. Habitat–Matterport 3D의 인도어 4개 씬에서, 메모리가 없는 신규 로봇은 성공률 24.1%에 그쳤지만 A,B의 메모리를 병합해 상속받은 신규 로봇(D)은 SR 77.1%/SPL 0.757~0.809 수준으로 크게 향상됐다. 이는 멀티로봇이 맵이나 과업 배정만이 아니라 실행 중 생성된 상태를 공유할 때, 신규 로봇이 ‘탐색부터’가 아닌 ‘팀 컨텍스트 기반’으로 출발할 수 있음을 정량적으로 보여준다.



### Drifting in the Future: Stabilizing Path Following Drifting on High-Latency Vehicle Systems (https://arxiv.org/abs/2606.27914)
- **Prior Approaches**: 자율 드리프트(자동 차량 활주/미끄러짐 제어)는 수학적·계산적으로 매우 까다롭고, 안정 한계(stability limit) 안팎에서의 제어는 특히 난도가 높다. 기존 시연은 즉시 토크 전달이 가능한 연구용 플랫폼이나 휠이 독립적으로 구동되는 환경에 크게 의존해, 생산 차량의 액추에이터 지연과 구동축 기계적 결합이 있을 때의 적용 가능성은 불명확했다. 또한 지연과 결합을 함께 고려한 제어 공식 및 속도 안정화 방식이 충분히 정리되지 못했다.

- **Core Contribution**: 이 논문은 전력구동(powertrain) 지연을 보상하는 predictor를 설계해 지연으로 인한 성능 저하를 줄인다. 이어서 구동축의 differential coupling과 더 긴 actuation latency를 수용하도록 제어 정식화를 개정하고, brake 기반의 속도 안정화 아이디어를 도입한다. 결과적으로 엔진 기반 생산 스포츠카에서도 드리프트를 “안정적으로 유지”하는 제어 프레임워크를 제시한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 250ms 이상 지연이 존재하는 상황에서 궤적·측면주행각(sideslip) 추적이 발산하지 않게 만드는 것, (2) 구동축 결합으로 인해 한 휠/한 입력의 변화가 전체 동역학에 비선형적으로 영향을 주는 것이다. 저자들은 predictor로 지연을 선제 보정하고, 결합 및 고지연을 반영하는 모델 확장과 revised control formulation으로 제어기의 일관성을 확보했다. 동시에 brake-based velocity stabilization으로 속도 변동이 진동을 유발하지 않도록 억제해 경로와 sideslip tracking을 동시에 안정화했다.

- **Empirical Impact**: 실험은 엔진 구동 생산 스포츠카에서 원형 및 figure-eight 드리프트를 “강건하게 유지”할 수 있음을 보여주며, 횡방향 오차는 1.1 m로 제한되고 sideslip overshoot는 0.06 rad에 그쳤다. 특히 actuator delay가 250 ms를 넘는 조건에서도 진동을 완화하면서 경로·sideslip 추적 성능을 유지했다. 이는 자율 드리프트가 생산 차량에서도 가능하다는 경험적 근거를 제공해, 전통 제어가 실패하는 상황에서 안전 시스템으로 확장될 수 있는 길을 연다는 점에서 의미가 크다.



### Swarm sign language: motion-based communication between drones (https://arxiv.org/abs/2606.27883)
Comments:
          8 pages, 7 figures

- **Prior Approaches**: 기존 로봇 간 모션 기반 통신은 비전 활성화나 RF 대신 움직임을 신호로 쓰려는 시도가 있으나, 대부분이 제어/문법 설계에 치우치거나(대상 차량 모델 의존), 관측 오차·불완전한 입력 윈도우를 강하게 다루지 못했다. 또한 신호를 전부 “한 번의 제스처=한 메시지”로 보는 경우가 많아 잡음이 끼는 연속 비행에서의 강건성 입증이 상대적으로 제한적이다.

- **Core Contribution**: 이 논문은 드론 스웜을 위한 motion-based communication을 RGB 카메라 기반으로 구현하며, 송신 드론이 실행하는 2D 기하 프리미티브(총 9개)를 planar trajectory로 수행하되 scale과 plane normal로 의미를 추가 인코딩한다. 수신기는 pose estimator가 뽑은 spatiotemporal pose 시퀀스를 3DTrajDecoder로 넣어 shape 분류, 구간 segmentation, 그리고 연속 기하량(크기·normal)을 동시에 예측한다. 나아가 통신/비통신 궤적을 함께 학습할 수 있도록, dynamically feasible 궤적을 온라인으로 생성하는 절차적 파이프라인을 제안한다.

- **Technical Challenges**: 핵심 난관은 (1) 카메라 기반 pose 추정 잡음과 입력 윈도우 불완전성이 만드는 오류를 견딜 수 있게 디코딩해야 하고, (2) 딥러닝 학습용 데이터가 부족한 상황에서 현실적인 운동 제약을 만족하는 궤적을 대규모로 합성해야 한다는 점이다. 이를 위해 minimum snap 최적화로 동역학적으로 매끄러운 연속 trajectory를 만들고, controller noise 및 관측 noise를 체계적으로 주입했으며, 입력 길이에서 생기는 토막 문제를 해결하도록 Vision Transformer 스타일의 multi-task Transformer(분류·회귀·세그먼트)를 설계했다.

- **Empirical Impact**: 시뮬레이션과 ablation을 통해 동작 가능한 operating domain(예: completeness 60% 이상, 크기 0.4m 이상)에서 분류 F1 85.7%+, normal 각도 오차 평균 11.5° 미만, segmentation F1 95%+ 같은 성능을 제시하며, PCA 기반 normal 추정 대비 회귀 성능이 우수함을 보였다. 또한 실제 비행 실험에서 88개 통신 구간을 12초 윈도우로 슬라이싱해 평가했을 때 geodesic error 평균 3.7°, 크기 오차 평균 0.072m를 기록했고, controller로 인한 오차가 있더라도 도메인 내에서는 분류/세그먼트/크기 회귀가 견고하게 유지되었다.



### S$^2$-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation (https://arxiv.org/abs/2606.27872)
Comments:
          Accepted to IJCAI 2026

- **Prior Approaches**: 기존 VLA( Vision-Language-Action )는 end-to-end fine-tuning으로 비전·언어·행동 표현을 정책에 연결하지만, 장기 과제에서 누적 오류가 크게 늘어나는 문제가 있었다. 이는 시각·언어·행동의 결합이 대부분 static feature fusion(고정 가중 결합)으로 설계되어 작업 단계가 바뀌어도 같은 비중으로 정보를 섞기 때문이다. 그 결과, 집는 단계에서는 공간 정밀도가, 계획 단계에서는 semantic intent가 충분히 강조되지 못하고 초기 편향이 실행 체인을 따라 증폭되기 쉽다.

- **Core Contribution**: 이 논문은 S$^2$-VLA를 제안하며, 핵심은 State-Space Guided Adaptive Attention(SSGAA)으로 단계별 적응형 융합을 구현한 것이다. 모델은 belief state(내재화된 상태)를 통해 task progression을 추적하고, 비전(공간 인지), 언어(고수준 의도), 과거 행동(시간적 일관성) 세 축의 gating 가중치를 동적으로 생성해 작업 국면에 맞게 관심을 이동한다. 특히 2B 파라미터의 소형 구조로도 더 큰 7B급을 넘어 장기 조작에서 state-of-the-art를 달성하는 점을 강조한다.

- **Technical Challenges**: 관건은 (1) 장기 실행에서 temporal coherence를 유지할 belief state를 어떻게 구성하고, (2) 그 상태를 이용해 멀티모달 융합 비중을 안정적으로 바꾸는 방법을 찾는 데 있었다. 논문은 최근 행동-감각 쌍을 요약하는 belief state를 GRU 기반으로 재귀 업데이트하고, action prediction loss만으로도 이 상태가 과제 단계와 실행 품질 정보를 자연스럽게 내재화되도록 학습시킨다. 이어 SSGAA는 시각/의도/행동을 각각 분기한 뒤 belief state로 제어되는 soft gating으로 융합하며, 실험적으로는 특정 중간 레이어(레이어 12) 단일 지점에서의 gating이 가장 유리함을 보여준다.

- **Empirical Impact**: LIBERO와 SimplerEnv, 그리고 실로봇 ALOHA(바이매니얼)에서 일관되게 높은 장기 성공률을 보고하며, LIBERO에서는 평균 success rate 98.2%로 성능을 끌어올렸다. SimplerEnv-Bridge 같은 고현실 시뮬레이션에서도 2B 소형 모델이 SOTA 수준을 유지해 long-horizon error propagation 완화 효과를 뒷받침한다. 또한 정적 fusion을 대체하는 동적 초점 전환이 단순 스케일링보다 효과적임을 보여, 장기 로보틱 조작에서 적응형 feature fusion의 중요성을 실증적으로 강화했다.



### LocalNav: Distilling Frontier VLMs and Embodied RL for On-Device Object Goal Navigation (https://arxiv.org/abs/2606.27871)
- **Prior Approaches**: ObjectNav은 open-vocabulary 임무로 확장됐지만, 기존 end-to-end/low-level VLA 접근은 높은 제어 주기(폐루프)와 대규모 SFT 데이터 의존 때문에 로봇 온디바이스 실행이 어렵다. 반면 SG 등 중간 표현 기반 high-level 의사결정은 지연 제약이 완화되지만, 여전히 cloud-bound frontier VLM에 의존하는 경우가 많아 로컬·저지연 배치가 막힌다.

- **Core Contribution**: 이 논문은 cloud 연결 없이도 edge GPU(예: Jetson Orin)에서 동작하는 LocalNav를 제안한다. 핵심은 대형 frontier 모델의 공간-의미 추론을 distillation과 domain SFT(~500 reasoning traces)로 4B local VLM(Qwen3.5-4B)로 이식하고, 이후 E-RLVR 기반 Token Generation(TG) 정규화로 출력 길이를 줄여 저지연 실행을 만든다는 점이다.

- **Technical Challenges**: 작은 VLM은 zero-shot 상태에서 ObjectNav 추론 성능이 크게 부족하므로, 최소 데이터로도 teacher의 reasoning 능력을 전이할 수 있는 distillation 절차가 필요했다. 또한 edge 배치에서 가장 큰 병목이 TG(자기회귀 출력) 단계여서, 단순 프롬프트 압축이나 CoT 억제로는 성능-지연의 균형을 맞추기 어려웠고, E-RLVR의 closed-loop 학습에서 token brevity를 목표로 보상하도록 설계해 해결했다.

- **Empirical Impact**: HM3D OVON에서 Claude Sonnet 4.6 기반 SotA SG 파이프라인은 SR 39.7%를 달성했고, 같은 구조로 4B local 학생 모델도 SR 34.5%까지 격차를 줄였다. 더 나아가 E-RLVR로 출력 길이를 72.1% 줄이고 latency를 71.8% 낮춘 뒤, llama.cpp 양자화를 결합해 전체 추론 지연을 82.8% 추가 절감했으며(성능 저하를 크게 동반하지 않음), 실환경에서도 4단계 ObjectNav 시연을 수행했다.



### PPO-EAL: Exact Augmented Lagrangian Proximal Policy Optimization for Safe Robotic Contro (https://arxiv.org/abs/2606.27861)
Comments:
          11 pages, 8 figures and 8 tables

- **Prior Approaches**: 기존 safe RL은 CMDP 형태로 제약을 비용(cost)로 모델링하지만, 많은 방법이 정책 업데이트에서 제약을 충분히 ‘정확히’ 만족시키지 못해 안전 요구사항을 흔들리게 만들었다. Lagrangian 기반 first-order 접근은 계산이 가볍지만, 듀얼 변수(라그랑주 멀티플라이) 업데이트가 느려 비용 경계에서 진동/초과(overshooting)가 생기기 쉽다. 한편 penalty 기반 방법은 동등성을 보장하려고 지나치게 큰(혹은 무한) 페널티가 필요해 수치 불안정과 ill-conditioning 문제가 자주 따라왔다.

- **Core Contribution**: 논문은 PPO를 safe RL에 적용하면서, exact augmented Lagrangian을 PPO 내부에 통합한 PPO-EAL을 제안한다. 핵심은 Lagrangian에 정확한(슬랙 변수 없이) 이차 페널티를 결합해, 이론적으로 원래 제약문제와의 동치(exactness)를 큰 페널티 계수 없이 달성하는 것이다. 또한 reward/cost에 대해 clipped 업데이트를 유지하면서, 정책 파라미터와 멀티플라이를 서로 다른 시간 스케일로 갱신해 수렴성을 확보한다.

- **Technical Challenges**: 정확 augmented Lagrangian을 RL의 PPO 학습 파이프라인에 넣으면, clipping/추정 편향과 제약 경계에서의 비매끈성(hinge 형태) 때문에 제약 만족이 흔들릴 위험이 있다. PPO-EAL은 quad penalty를 포함하되 Rockafellar-type exact augmented Lagrangian 구조로 formulation을 정리해 “이론적 정확성”을 유지하고, 멀티플라이 업데이트에는 momentum-regulated(감쇠 성격의) 조절을 추가해 constraint oscillation과 unsafe 행동을 줄인다. 이 과정에서 표준 stochastic approximation 가정 아래 exactness와 convergence 분석도 함께 제시한다.

- **Empirical Impact**: GPU 가속 로보틱스 벤치마크 전반(cart-pole, cart-double-pendulum, 7-DoF Franka end-effector reaching, quadrupedal locomotion)에서 PPO-EAL은 최신 first-order safe RL 대비 제약 ‘정밀도’와 보상 성능을 동시에 개선했다고 보고한다. 특히 contact-force 등 복수 물리 제약(속도/위치/토크/접촉 등)을 포함한 환경에서도 안전 정밀도가 유지되는 것이 강조된다. 더 나아가 zero-shot sim-to-real로 contact-rich gear assembly를 수행해 성공률을 높이고 피크 접촉력을 낮추며, 현장 수준의 운영 robust도 향상됨을 실험적으로 보였다.



### Booster Lab: A Data-Centric Pipeline for Learning Deployable Humanoid Locomotion Policies (https://arxiv.org/abs/2606.27813)
- **Prior Approaches**: 기존의 휴머노이드 학습은 기준 모션을 추적하는 feature-based imitation과 판별자를 통해 보상을 학습하는 AMP(Adversarial Motion Priors) 계열로 나뉜다. 하지만 인간 시연은 로봇 형태 불일치로 잡음·부적합이 잦고, 시뮬 궤적도 접촉/관절/동적 제약을 통과하는지 별도 점검이 필요하다. 또한 데이터 준비, 학습, 실로봇 평가는 파이프라인이 분리돼 있어 배포 실패 원인 진단과 개선이 어렵다는 한계가 있었다.

- **Core Contribution**: 이 논문은 Booster Lab으로, 모션 데이터를 “고정된 데모”가 아니라 선택·수정·재타겟팅·필터링되는 최적화 가능한 학습 루프의 일부로 다룬다. motion data curation, real-to-sim model adaptation, AMP 기반 강화학습, cross-simulator validation, sim-to-real deployment를 하나의 닫힌 흐름(closed-loop)으로 연결해 배포 가능성을 높인다. 실로봇 실패 피드백이 이후 데이터 선택·증강·시뮬 캘리브레이션·재학습에 다시 반영되는 점을 핵심으로 한다.

- **Technical Challenges**: 핵심 난제는 서로 다른 소스(인간 시연, open-source clips, 시뮬 궤적)를 로봇에 맞는 “expert”로 변환하는 데 있다. 이를 위해 BeyondMimic 스타일 tracking으로 궤적을 로봇-실행 가능하게 보정하고, 접촉 일관성·관절/토크 한계·발 슬라이딩·추적 안정성 기준으로 feasibility filtering 및 AMP 학습용 데이터셋 밸런싱을 수행한다. 또 Booster T1의 구동기 특성(T–n 곡선의 knee point)과 armature를 시뮬 모델에 반영해 real-to-sim mismatch를 줄이고, MuJoCo/Webots의 독립 검증과 함께 도메인 랜덤화·외란 복원 훈련으로 강건성을 확보한다.

- **Empirical Impact**: Booster T1에서 데이터 증강 ablation이 초기 학습의 imitation gap을 키우지만 이후 최종 discriminator 예측을 더 높여 expert 분포에 더 잘 근접함을 보였다. 하드웨어에서는 −0.6~1.0 m/s 보행과 최대 2.0 m/s 러닝 안정성, 외란(투사체) 이후 넘어짐 없이 push-recovery, 야외 지형(자갈/경사/요철 등) 적응을 확인했다. 또한 시뮬과 실세계 joint limit cycle(힙 피치·무릎) 형태가 유사해 sim-to-real 격차 완화에 효과가 있음을 보였고, Booster K1에도 로봇 전용 모듈만 교체해 예비 수준의 안정 러닝 전이를 시연했다.



### LXD-SLAM: LiDAR+X Dense SLAM with $\sum_{i=0}^{5}C_5^i$ Configurable Sensor Combinations (https://arxiv.org/abs/2606.27811)
- **Prior Approaches**: 기존 LiDAR 기반 SLAM은 IMU를 결합한 LIO 계열(LIO-SAM 등)과, 비정형/퇴화 환경에서 비전까지 더한 LVIO 계열로 성능을 끌어올려 왔다. 다만 많은 시스템이 고정된 센서 구성에 맞춰 설계되거나, 융합 수식이 모달리티 간 일관되지 않게 결합돼 정확도가 흔들릴 수 있다. 또한 맵 표현이 특징 기반으로 희소하거나(정밀한 데이터 연관에 불리) 지나치게 밀집·불연속적이라 대규모에서 계산이 감당되지 않는 한계가 있었다.

- **Core Contribution**: 이 논문은 3D LiDAR를 중심으로 LiDAR+X Dense SLAM(LXD-SLAM)이라는 통합형 다중센서 융합 프레임워크를 제안한다. LiDAR, Camera, IMU, Wheel Encoder, GNSS를 플러그앤플레이로 조합할 수 있으며 최대 32가지 센서 조합을 지원한다. 전역 일관성을 위해 GP 기반 연속 다층 서피스 표현과 Extended Scan Context(ESC), 그리고 Bidirectional PnP 기반 멀티모달 루프 클로저를 결합한다.

- **Technical Challenges**: 핵심 과제는 (1) 센서 비동기/추출 오차가 커지는 상황에서도 일관된 상태추정을 유지하고, (2) 시각 특징의 깊이를 신뢰성 있게 복원해 업데이트 신호를 안정화하며, (3) 대규모에서 전역 정합과 밀집 맵을 동시에 달성하는 것이다. 이를 위해 IESKF를 공통 추정 엔진으로 두고 센서 가용성에 따라 계층적 예측 모델을 선택하며, 업데이트는 point-to-mesh 잔차와(가능 시) 시각 재투영 오차를 함께 최소화한다. 시각 특징의 깊이는 GP sub-mesh에 대한 ray-to-mesh depth recovery로 얻고 1D inverse depth filter로 성숙한 특징만 반영하도록 설계했으며, 엑스트린 관리는 star topology로 누적 캘리브레이션 잡음을 줄여 기하 일관성을 보장한다.

- **Empirical Impact**: 실험은 공개 데이터셋과 실제 환경에서 수행됐고, LXD-SLAM이 구성 전반에서 전문화된 state-of-the-art odometry와 동등하거나 그 이상 성능을 보이면서 실시간 고품질 전역 일관 밀집 메쉬를 생성함을 보여준다. 즉, 특정 하드웨어/센서셋에 종속되기 쉬운 기존 접근과 달리 센서 조합을 바꿔도 동작하는 ‘적응형 SLAM’의 실효성을 입증한 셈이다. 발표 시점에 코드와 데이터를 공개할 계획이라고 밝혀 후속 비교·확장 연구에도 파급이 기대된다.



### SpikeVLA: Vision-Language-Action Models with Spiking Neural Networks (https://arxiv.org/abs/2606.27807)
Comments:
          Accepted by ICML 2026. 16 pages, 9 figures

- **Prior Approaches**: 기존 Vision-Language-Action(VLA) 모델은 대체로 대형 transformer로 멀티모달 추론을 수행하고, ANN 기반 정책망으로 제어를 처리한다. 이 구조는 연산이 조밀해 추론 지연과 에너지 사용량이 커서 마이크로 로봇·저전력 자율주행 등 온보드/실시간 제약에서 한계가 있었다.

- **Core Contribution**: 본 논문은 spiking neural network 기반의 최초 VLA 아키텍처인 SpikeVLA를 제안한다. Spike-V(시각 인코더), Spike-L(멀티모달 spiking LLM), Spike-A(fully spiking 행동 정책)로 구성해 추론·제어 전 과정을 event-driven 방식으로 저전력화하면서도 성능을 유지하도록 설계했다.

- **Technical Challenges**: 핵심 과제는 spiking 동역학에서 높은 표현력과 안정적인 closed-loop 제어를 동시에 확보하는 것이다. 논문은 differential spiking(차분 코딩·차분 스파이킹 뉴런)으로 소수 타임스텝에서 정밀도를 유지하고, SigLIPv2의 선형/비선형 연산을 event-driven으로 변환하며, Spike-L에는 토큰 수준 sparsity와 differential temporal sparsity allocation을 적용해 계산비용을 더 줄였다.

- **Empirical Impact**: VLN-CE R2R/RxR 및 VLN-CE-Isaac에서 SpikeVLA는 RGB-only·no-waypoint 조건에서도 경쟁력 있는 내비게이션 성능을 보였다. 동시에 GPU 메모리 사용을 16.1GB→6.2GB, 에너지 지표를 NaVILA 대비 약 34% 수준으로 낮추는 등 연산·전력 효율을 크게 입증했으며, 낮은 수준 locomotion 정책에서도 NaVILA와 유사한 제어 품질을 유지하면서 에너지 효율을 개선했다.



### Drop-Then-Recovery: How Redundant Are Vision-Language-Action Models? (https://arxiv.org/abs/2606.27755)
- **Prior Approaches**: VLA 모델은 비전-언어 백본과 행동 예측 모듈을 결합해 지시 기반 로봇 조작을 수행하지만, 대부분은 사전학습된 대형 언어 백본을 그대로 상속한다. 기존 압축 연구는 주로 양자화·pruning처럼 “제거 후 복구 없이” 성능을 평가해, 닫힌고리 제어에서 실제 성공에 필요한 용량을 가늠하기 어렵다.

- **Core Contribution**: 이 논문은 Drop-Then-Recovery(DTR)로 트랜스포머 블록을 제거한 뒤 다운스트림 제어 작업을 fine-tuning해, 제거된 용량이 실제로 필요한지 recoverability 관점에서 측정한다. 또한 어떤 블록을 제거할지 고르는 one-shot 가상 게이트 기반 지표인 GateProbe를 제안해, 단순 유사도나 크기, 즉각 성능 저하가 아닌 행동 손실 기여도를 기준으로 블록을 랭킹한다.

- **Technical Challenges**: 문제는 “즉각 손실”이 아니라 “복구 가능성”을 예측해야 한다는 점이며, 기존 정적 메트릭이나 Taylor류 그라디언트 기반은 극단 압축에서 안정성이 떨어지거나 비용이 크다. GateProbe는 각 블록의 residual 경로에 가상 gate를 둔 민감도(손실의 기대 절댓값)를 근사해, 체인룰로 모델 내부 게이트 삽입 없이 다운스트림 그라디언트와 residual 기여의 내적으로 계산하며 제거 후보를 효율적으로 선정한다.

- **Empirical Impact**: 시뮬레이션(LIBERO, LIBERO-Plus, RoboTwin 2.0)과 실제 로봇(UFACTORY xArm 850)에서 일관되게 “언어 백본은 중복이 크고, 비전·액션 경로는 제거에 취약”하다는 비대칭이 관찰됐다. 예를 들어 LIBERO에서 언어 LLM 블록 절반 제거가 OpenVLA-OFT의 경우 95.0%→98.3%로 기준을 넘어섰고, 언어 블록을 2개만 남겨도 baseline 수준 복구가 가능했다. 이는 현 VLA 벤치마크가 언어 grounding과 조합적 추론에 충분한 압력을 주지 못할 수 있으며, 향후 아키텍처는 언어-비전-액션에 용량을 더 의도적으로 배분하고 더 강한 언어·OOD 테스트가 필요하다는 신호로 받아들여진다.



### DIM-WAM: World-Action Modeling with Diverse Historical Event Memory (https://arxiv.org/abs/2606.27677)
- **Prior Approaches**: VLA/로보틱스 모델은 비전-언어-행동을 하나의 시퀀스 모델로 묶어 성능을 끌어올렸지만, 핵심 감독 신호가 희소한 행동 라벨에 머물러 비접촉/가림/단계 전환 같은 미세 동역학을 충분히 직접 학습하기 어렵다. World-action model(WAM)은 미래 시각 상태를 함께 예측해 동역학을 밀도 있게 감독하지만, 기존 방식은 주로 단기 히스토리와 단기 예측에 기대어 장기 과업에서 필요한 “이전 관측의 의미”를 제대로 유지하지 못한다.

- **Core Contribution**: DiM-WAM은 장기 비마르코프 의존 과업에서 필요한 시간 정보(단기 국소 문맥, 교차 단계 과거 사건, 즉시 미래 동역학, 전역 task progress)를 분해하고, 그중 장기 과거를 Multi-Type Historical Event Memory로 외부화해 기억 붕괴와 전역 상태 인식을 보완한다. 메모리는 관측에서 압축된 사건 토큰을 뽑아 다중 메모리 bank에 유사도 기반 병합으로 저장하고, 읽기 시 bank identity와 시간 임베딩을 부여해 video/action denoising에 일관되게 조건을 제공한다.

- **Technical Challenges**: 핵심 난제는 유한 용량 메모리 안에서 서로 다른 기능의 과거 정보가 경쟁하며 덮어쓰기/주의 경쟁이 생기고, 특히 생성 기반 상태가 중간에 누적되면 drift가 발생해 닫힌 고리 예측이 망가질 수 있다는 점이다. DiM-WAM은 (1) bank별 독립적인 압축·병합으로 이질적 사건의 직접 경쟁을 줄이고, (2) 읽기 단계에서 bank identity+RoPE 시간 순서를 적용하며, (3) progress-supervision 보조 목표로 메모리가 완료된 사건뿐 아니라 현재 과업 단계와 남은 작업의 함의를 함께 담도록 유도해 전역 상태 편향을 억제한다.

- **Empirical Impact**: RMBench에서 LingBot-VA 기반으로 고정 용량 메모리를 적용했을 때 평균 success가 28.4%에서 69.8%로 크게 상승했으며, explicit-memory Mem-0(42.0%)도 앞섰다. 실제 Franka 4개 과업에서도 stage success가 70.7%→91.5%, full-task success가 52.5%→80.0%로 개선되었고, ablation과 분석은 multi-bank 구조와 progress 감독이 특히 시간 의존 과업에서 이득을 만든다는 점을 뒷받침한다.



### CWI: Composite Humanoid Whole-Body Imitation System for Loco-manipulation (https://arxiv.org/abs/2606.27676)
- **Prior Approaches**: 기존 전신(whole-body) 추적 기반 방법은 필터링된 MoCap 데이터를 프레임 단위로 모사해 정밀도는 높지만, 데이터 큐레이션과 MoCap 인프라가 필요하고 배포 시 OOD에 취약하다는 한계가 있다. 한편 상체/하지를 분리하거나 명령(velocity/height) 조건을 쓰는 방법은 안정적인 보행을 만들 수 있으나, 하체를 충분히 사람의 하체 모션 priors로 활용하지 못해 상체 모션 통계가 어긋나거나 조정(coordination)이 떨어질 수 있다.

- **Core Contribution**: 이 논문은 Composite Whole-Body Imitation(CWI)로, 상체는 MoCap 데이터(AMASS) 전체를 모션 레퍼런스로 최대한 보존하고, 하체는 안정적인 보행/스쿼트만 선별한 뒤 AMP(Adversarial Motion Priors)로 command-conditioned 보행을 학습하도록 MoCap 활용을 역할별로 분리한다. 또한 전신 정책을 학습하되, locomotion/upper-body/style 목적을 multi-critic으로 분리해 최적화 간 간섭을 줄이고, teacher–student distillation으로 배포 시 전신 MoCap 없이 양손 keypoint와 속도/높이 명령만으로 제어 가능하게 만든다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) MoCap 데이터의 분포 불균형으로 인해 하체 모션이 안정 명령 범위를 제대로 커버하지 못하고, (2) 상체 모사와 하체 안정성 목표가 고(高)차원 전신에서 충돌해 학습이 흔들릴 수 있다는 점이다. CWI는 상체는 root를 제거해 팔 궤적만 사람 통계로 학습하고(필터링 없이 유지), 하체는 walking/squatting 듀얼 AMP discriminators와 멀티 크리틱 가치함수로 목적별 보상/학습 신호를 구조적으로 분리했으며, 마지막에는 action-matching BC를 포함한 distillation으로 teacher–student 관측 차이를 완화한다.

- **Empirical Impact**: 시뮬레이션 벤치마크에서 CWI는 상체 추적과 하체의 속도/높이 추적을 동시에 개선하며, 다양한 unseen 명령과 상체 레퍼런스 조합에서도 기존 baseline 대비 전반적으로 더 좋은 trade-off를 보였다. 또한 full-size LimX Oli 인간형 로봇에서 VR 텔레오퍼레이션(휴대용 Meta Quest)으로 실제 loco-manipulation을 수행하며, 전신 MoCap 장비 없이도 경쟁력 있는 전신 조정과 강건성을 보여 배포 실용성 측면의 의미가 크다.



### Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization (https://arxiv.org/abs/2606.27663)
- **Prior Approaches**: VLA(vision-language-action) 모델은 언어 지시를 따르는 능력을 갖췄지만, 시험 시에는 공간 일반화(물체 위치가 학습과 다르면 실패)와 작업 일반화(같은 장면에서 다른 지시를 주면 실패)에서 취약하다는 문제가 반복적으로 보고됐다. 이를 줄이기 위해 픽셀 좌표·마스크 같은 2D 기반 grounding을 언어 프롬프트나 시각 프롬프트로 정책에 전달하거나, 중간 단계의 spatial 예측(체인오브쎄스)을 거쳐 행동을 내는 접근이 늘어났다. 하지만 기존 연구는 grounding 신호를 어떤 표현으로 만들고, action head로 어떻게 주입해야 일반화가 되는지에 대한 해답이 체계적으로 정리되지 않았다.

- **Core Contribution**: 본 논문은 “grounding 신호의 표현과 주입 방식”이 핵심 병목이라고 보고, 2D 타깃 포인트를 3D로 들어 올린 뒤 action head에 직접 주입하는 경량 모듈을 제안한다. 구체적으로는 depth와 카메라 파라미터로 타깃의 3D 위치를 복원하고, 그리퍼와의 상대 변위(displacement)를 2층 MLP로 인코딩한 후 DiT 기반 action head의 AdaLN(adaptive layer normalization)에 주입한다. 이 과정은 VLA 백본이나 pretraining 파이프라인을 변경하지 않는 model-agnostic 설계로, 숫자 기반 3D 기하 신호가 행동 예측에 바로 전달되도록 만든다.

- **Technical Challenges**: 가장 큰 기술적 과제는 2D grounding을 3D로 “의미 있게” 변환하는 것만으로는 부족하고, 최종 action head까지 기하 구조가 보존되게 주입해야 한다는 점이다. 논문은 3D 좌표를 텍스트 토큰으로 프롬프트에 붙이는 방식은 VLM 내부에서 수치-기하 구조가 손실되어 이득이 작다고 관찰하며, 반대로 연속 임베딩 형태의 3D 상대 변위를 AdaLN의 scale/shift 조절에 결합할 때 성능이 크게 오른다고 보인다. 또한 상대 변위에 그리퍼 위치 정보를 포함시키는 방식이 기대만큼 좋지 않을 수 있어, gripper-to-target geometry를 유도하는 입력 구성이 일반화에 중요하다는 애블레이션도 제시한다.

- **Empirical Impact**: LIBERO-PRO에서 GR00T-N1.6에 적용하면 task perturbation 성능이 31.2에서 77.5로, position perturbation 성능이 28.1에서 60.2로 크게 상승해(각 +46.3, +32.1) 공간·작업 모두의 시험-시간 붕괴를 크게 되돌렸다. 또한 π0.5에도 유사한 개선이 나타나며, 같은 메커니즘이 백본에 독립적임(backbone-agnostic)을 실험으로 확인했다. 시뮬레이션뿐 아니라 실제 로봇에서도 2D 프롬프트 기반 기준선은 OOD 조건에서 거의 실패한 반면, 제안 모듈은 상당 수준의 성공률을 유지해 실사용 관점의 transfer 가능성까지 보여준다.



### P-ARC: Exploiting Subproblem Independence for Parallel Multi-Robot Motion Planning (https://arxiv.org/abs/2606.27625)
- **Prior Approaches**: 기존 병렬 MRMP 연구는 단일 로봇용 샘플링 기법을 multi-robot 합성공간에 그대로 확장하려 했고, 합성공간의 지수적 팽창 때문에 로봇 수가 늘수록 한계가 뚜렷했다. 반면 우선순위 기반 decoupled 계획은 검색공간을 줄이지만, 우선순위 의존성 때문에 병렬화 여지가 작아진다. ARC는 하이브리드로 먼저 독립 경로를 만든 뒤 충돌이 생기는 부분만 국소 서브문제로 해결하지만, ARC의 세 단계 전부를 병렬로 다루는 방식은 부족했다.

- **Core Contribution**: 이 논문은 ARC의 세 단계(초기 독립 계획, 충돌 탐지, 충돌 해결)를 각각 병렬화한 Parallel ARC (P-ARC)를 제안한다. 또한 OR-parallel multi-start를 ARC/P-ARC에 결합해 OR-ARC와 OR-P-ARC(하이브리드 전략)까지 확장한다. 핵심은 ARC 분해로 생기는 독립성을 최대한 활용하되, 충돌 해결 단계가 서로 간섭하지 않도록 충돌을 독립적으로 묶는 설계를 포함한다.

- **Technical Challenges**: 충돌 탐지 단계는 로봇 쌍과 타임스텝을 모두 훑기 때문에 오버헤드가 커지고, 충돌 해결을 병렬로 하려면 탐지된 충돌들이 실제로 독립적이어야 한다. 논문은 pair covering 기반으로 충돌 탐지 작업을 워커에 분배해 경로 재구성과 충돌체크 중복을 줄이고, early termination과 independence(완전 독립에 가까운 제약)의 충돌을 조절하기 위한 horizon 동기화도 둔다. 더불어 충돌이 서브문제 확장으로 커질 수 있으므로, 확장 로직을 충돌 탐지 함수 안에 통합해 독립성 검사를 일관되게 적용한다.

- **Empirical Impact**: 2D 시나리오에서 로봇 수와 워커 수를 체계적으로 늘리며 분석한 결과, 충돌 탐지 비용이 커질수록 P-ARC의 시간 단축이 두드러졌다. 또한 3D Panda 멀티 매니퓰레이터 팀의 현실감 있는 조건에서 16 CPU 코어 사용 시 순차 대비 최대 약 4X(근접 4X) 속도 향상을 보였고, 큰 팀(예: 8~16대)에서 성공률을 유지하면서도 계획 시간이 단축되는 패턴이 나타났다. 특히 OR-P-ARC는 “unlucky seed”로 인한 지연/타임아웃을 multi-start로 회피해 견고성을 높이며, 해제(makespan) 관점에서 비용 증가 없이 성능을 끌어올린 점이 의미 있다.



### Physics-Guided Robotic Radiation Source Localization along Arbitrary Measurement Paths in Unstructured Environments (https://arxiv.org/abs/2606.27624)
Comments:
          18 pages, 14 figures, 2 tables

- **Prior Approaches**: 기존 로봇 방사선원 위치추정(RSL) 연구는 정보이득을 최대화하는 active search 중심으로 경로를 계획해 원점에 점점 접근하는 경우가 많았다. 하지만 이는 방사선 노출 위험을 키우고, RSL 목적에 최적화된 path-planning이 방사선 환경 내 다른 임무(점검·감시 등)의 유연성을 제한한다. 또한 obstacle이 알려진 경우에는 추정이 쉬우나, 장애물의 물성·형상·위치가 미지인 상황에서는 Poisson 잡음과 차폐 감쇠 때문에 정밀·일반화 성능을 내기 어렵다.

- **Core Contribution**: 이 논문은 측정 경로가 임의로 주어져도 방사선원 위치를 추정하는 automation framework를 제안하며, PIML(physics-informed machine learning) 기반 RSL 모듈로 확장성을 확보한다. 장애물로 인한 감쇠·산란을 physics-informed 방식으로 흡수하되, 장애물 정보가 없는 unknown environment에서도 학습·추론 가능하도록 모델을 설계했다. 또한 로봇 배치 시 추정된 위치를 ‘advanced perception information’으로 활용해 굳이 방사선원에 접근하지 않는 임무 운영을 지향한다.

- **Technical Challenges**: 핵심 난제는 (1) 검출이산 잡음과 다중 원천으로 인한 복잡한 방사장, (2) 장애물에 의한 감쇠·산란의 비식별성, (3) 임의 경로에서 측정된 flux만으로 역추정해야 한다는 점이다. 저자들은 Beer-Lambert 법칙과 inverse-square 법칙을 기반으로 physics-inspired tensors를 구성하고, 장애물 영향을 ‘결합된 감쇠 마스킹 파라미터’ 형태로 추정 가능하게 하며, 조건부 활성과 straight-through estimator로 이산 마스크 최적화를 수행한다. 더 나아가 obstacle-free/obstacle-aware 모델을 weighted hybrid로 결합하고, 초기화가 다른 여러 모델을 parallel inference로 돌려 L1 loss가 가장 낮은 모델을 선택해 강건성과 정확도를 동시에 높였다.

- **Empirical Impact**: 검증은 고충실도 시뮬레이션에서 Monte Carlo particle transport(OpenMC)를 사용해 공간 스케일(20m/10m/5m), 장애물 개수·재질(콘크리트/물/철/폴리에틸렌/납)·형상, 방사선원 종류(예: Co-60, Cs-137, Am-241 등)와 로봇 경로를 폭넓게 랜덤화하며 수행됐다. 그 결과는 시뮬레이션에 그치지 않고, 시뮬레이션에 포함되지 않은 물리 구성에서도 실제 소스·검출기·로봇으로 재현 실험을 통해 정밀도와 강건성을 확인했다고 보고한다. 또한 real-robot deployment에서 continuous learning(온라인 fine-tuning)을 적용해 계산 효율을 확보하며, 점단위 flux 관측을 넘어 spatial intelligence로 로봇 방사선 인식의 수준을 끌어올렸다는 점에서 의미가 크다.



### Learning to Throw: Agile and Accurate Cable-Suspended Payload Delivery with a Quadrotor (https://arxiv.org/abs/2606.27603)
- **Prior Approaches**: 케이블로 매단 페이로드를 다루는 기존 연구는 대부분 기하학적 제어, trajectory optimization, model-based tracking처럼 쿼드로터-로프-페이로드를 하나의(또는 강하게 결합된) 모델로 다루는 방식에 의존했다. 하지만 이들은 강한 보수적 feasibility 제약, 계획-추적 분해로 인한 추적오차, 유연 로프 동역학을 정확히 해석하기 어려운 문제가 있어 던지기처럼 고가속·정밀 릴리스가 필요한 상황에서 성능이 제한된다.

- **Core Contribution**: 이 논문은 쿼드로터 동역학은 고정밀 분석 모델(시스템 식별로 획득)로, 로프·페이로드 상호작용은 별도의 물리 솔버로 시뮬레이션하는 하이브리드 시뮬레이터를 제안한다. 핵심은 두 도메인을 단일 6자유도 힘/모멘트(wrench)로만 매 스텝 교환해, 결합된 단일 몬로리식 모델로 푸는 대신 정확도를 유지하면서도 sim-to-real에 필요한 모델링 난도를 낮춘 점이다.

- **Technical Challenges**: 가장 큰 기술적 병목은 RL을 zero-shot으로 하드웨어에 적용할 때, 쿼드로터의 closed-loop 동특성과 로프의 접촉·느슨함·스윙 같은 복잡 멀티바디 거동을 동시에 고충실도로 만족하는 시뮬레이터를 만드는 것이다. 저자들은 (1) 식별된 분석 쿼드로터 모델의 wrench를 솔버에 그대로 전달하고, (2) 솔버가 생성하는 마운트 반작용을 다시 쿼드로터 상태로 읽어 closed-loop를 구성하며, (3) 릴리스 메커니즘의 actuation lag와 탄도 예측을 보상 설계에 반영해 던지기 정책 학습을 가능하게 했다.

- **Empirical Impact**: 실험에서 저자들은 RL 정책을 하드웨어에 zero-shot으로 바로 배치해 착지 오차를 최대 50% 줄이고 던지기 지속시간을 최대 30% 단축하는 등 모델 기반 TO+MPC 대비 유리한 민첩-정확도 균형을 보였다. 또한 구성요소 소거(ablation)로 하이브리드 결합 시뮬레이션이 sim-to-real 이득의 핵심임을 확인했으며, 상태 추정 없이 시각 관측만으로 학습한 정책도 상태 기반과 비슷한 정확도를 달성했다.



### SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction (https://arxiv.org/abs/2606.27581)
Comments:
          15 pages 10 figures

- **Prior Approaches**: 기존 humanoid reinforcement-learning motion tracking은 대부분 reference motion을 따라가는 데 강점을 보였지만, 객체 접촉이나 계단 같은 비평탄 지형에서는 kinematics만으로 물리적 모호성을 해소하기 어렵다는 한계가 컸다. 특히 어떤 링크가 실제로 접촉·하중을 형성해야 하는지에 대한 scene-aware 정보가 부족해, 계단 오르기나 무거운 물체 조작 같은 장기 과업에서 실패 확률이 높았다. 더 나아가 학습용으로도 모션-장면-접촉 라벨이 함께 있는 대규모 데이터가 부족해 일반화가 제한됐다.

- **Core Contribution**: SceneBot은 단일 whole-body motion-tracking policy를 reference motion과 per-link contact labels(접촉 라벨)로 조건화해, free-space locomotion부터 terrain traversal(계단 등)과 whole-body manipulation까지 한 프레임워크에서 통합한다. 핵심은 “어떤 로봇 링크가 scene(지형/물체)과 접촉을 기대하고 활용해야 하는가”를 링크 단위로 명시해, 순수 추적을 넘어 접촉 기반 제어 인터페이스를 제공한다. 또한 장면 접촉 라벨이 없는 상황을 해결하기 위해 retargeted human motion으로부터 scene-interaction graph를 추론하는 hindsight scene reconstruction을 제안한다.

- **Technical Challenges**: 가장 큰 기술적 난제는 접촉을 포함한 학습 데이터를 만드는 것으로, 장면 자산(지형·물체)과 상호작용 그래프, 링크 단위 접촉 라벨이 동시에 필요하다. SceneBot은 모션을 로봇에 retarget한 뒤, 저상대속도/가속도 징후와 force-closure 및 충돌 제약으로 scene-interaction graph를 구성하고, 그 그래프에 맞춰 2.5D elevation map(지형)과 접촉면 평면 기반 object를 재구성해 학습 데이터를 합성한다. 학습 단계에서는 PPO로 kinematic tracking과 contact correctness/체류시간 보상 및 contact-mismatch 종료 조건까지 함께 최적화해 안정적 접촉 전환과 파지 성능을 노린다.

- **Empirical Impact**: MuJoCo sim-to-sim에서 Unitree G1 기반으로 검증한 결과, free-space에서는 SONIC과 비슷한 수준을 유지하면서도 지형·객체 접촉 과업 전반에서 유의미하게 더 높은 성능을 보였다. 특히 손/발 contact labels를 빼거나 비활성화하면 성공률이 급락해, contact conditioning이 실제로 정책 성공을 좌우함을 실험적으로 확인했다. 시뮬레이션 성공뿐 아니라 계단 오르기와 상자 운반 같은 long-horizon 복합 과업을 정성적으로도 수행하며, free-space와 contact-rich 행동을 자연스럽게 통합하는 일반 프레임워크의 의미를 보여준다.



### Spacecraft Fiducial Marker for Autonomous Rendezvous, Proximity Operations, and Docking (https://arxiv.org/abs/2606.27566)
- **Prior Approaches**: 기존 우주 로보틱스 RPOD(근접운용·도킹)는 인간이 개입하는 반자동 방식이 많고, 완전 자율화를 위해서는 카메라 기반 상대 자세 추정에 안정적인 시각 기준이 필요하다. fiducial marker로는 AprilTag, ArUco, ArUco/ARTag 계열의 이진(black-and-white) 마커와 다층/프랙탈 변형(예: Fractal ArUco), CCTag 같은 원형 동심구조가 제안됐지만 대체로 단일 스케일이거나, 다층일 경우 사전(dictionary) 크기·픽셀 요구량이 커지고 곡면에서의 내구성이 떨어진다. 또한 다수 방법은 외곽 사각형(quad) 기반 정합에 의존해, 곡면·원근·왜곡이 겹치는 구간(근접/도킹)에서 추적 실패가 커질 수 있다.

- **Core Contribution**: 이 논문은 온궤도 자율 로봇을 목표로 하는 재귀(recursive) fiducial marker AstraTag를 제안한다. Spidron 프랙탈(정사각 외곽) 구조의 자기유사성을 이용해 단일 마커에서도 원거리~근거리까지 같은 패턴이 다중 스케일로 해독되도록 설계했으며, 48-bit 서명은 triangular sub-region을 기반으로 GRS(Generalised Reed-Solomon) 부호로 오류 정정을 포함한다. 더 나아가 곡면 장착 환경을 위해 내부 직사각 경계 정보를 대응점으로 쓰는 Thin-Plate Spline(TPS) re-warp fallback을 포함해 비평면 왜곡을 보정한다.

- **Technical Challenges**: 핵심 난제는 (1) 우주 조명 조건에서의 대비 저하·그림자/반사, (2) 곡면에서 발생하는 비선형 왜곡과 out-of-plane 회전 증가에 따른 quad 및 셀 경계의 불안정화, (3) 근접 도킹 구간에서 마커가 필드에서 벗어나지 않게 다중 스케일로 안정적으로 식별하는 것이다. AstraTag는 CLAHE와 adaptive thresholding으로 이진화 견고성을 높이고, 계층적 contour 트리와 Hough 기반 라인 검증으로 사각형 후보를 복구한 뒤, 동차(homography)로 고정 크기 평면 정규화 후 triangular 셀의 면적 기반 샘플링으로 48-bit 서명을 추출한다. 표준 파이프라인이 실패할 때는 마커 내부 직사각 경계를 이용한 TPS re-warp로 곡면 보정을 수행해 곡면에서의 검출 붕괴를 줄였다.

- **Empirical Impact**: 곡면(원통 모듈)과 평면(우주선 목업)에서 Fractal ArUco, AprilTag와 비교 벤치마크를 수행한 결과, AstraTag는 곡면에서 out-of-plane 회전 10°~70° 구간 검출률 97–100%를 유지하며 TPS fallback의 효과가 관측됐다. 반면 Fractal ArUco는 곡면에서 회전 각이 커질수록 급격히 붕괴(예: 40° 부근에서 큰 하락)했고, AprilTag는 60°까지는 높지만 70°에서 56%로 하락했다. 평면 데이터셋에서는 세 마커 모두 성능이 크게 개선되어(특히 AstraTag·AprilTag는 전 구간 100%) 성능 격차의 주요 원인이 곡면/왜곡 및 외곽 quad 검출 실패임을 확인했으며, 재귀 구조 덕분에 근접운용·도킹용 ‘robust recursive-marker’ 선택지로서 의미가 크다.



### AO-ARC: Almost-Surely Asymptotically Optimal Multi-Robot Motion Planning with ARC (https://arxiv.org/abs/2606.27495)
- **Prior Approaches**: 기존 MRMP 연구는 로봇 수가 늘수록 조합 상태공간이 지수적으로 커져 “빠른 해”와 “고품질 해”를 양자택일하는 경향이 강했다. 우선순위 기반의 decoupled 방식은 빠르지만 팀 차원의 완전성/품질 보장이 약하고, coupled 방식은 품질·보장은 높지만 탐색 비용이 커진다. ARC는 개별 경로를 먼저 만들고 충돌이 생긴 곳만 국소 결합 부분문제로 확장해 균형을 시도하지만, 국소 수리 선택 때문에 비동기적 품질 보장(점근적 최적성)이 부족했다.

- **Core Contribution**: 본 논문은 makespan(마지막 로봇 도착 시점)을 기준선으로 하는 anytime MRMP 기법 AO-ARC를 제안한다. AO-ARC는 ARC의 적응적 (de)coupling 구조를 유지하면서, feasibility 해법을 anytime 알고리즘으로 바꾸는 AO-xx 메타알고리즘을 MRMP에 맞춰 적용한다. 특히 팀 완료 시간에 대한 전역 bound를 두고, bounded feasibility를 반복 호출하며 더 좋은 bound를 점진적으로 조여 수렴을 개선한다.

- **Technical Challenges**: 핵심 난제는 국소 충돌 수리(부분문제)가 전역 makespan bound를 침범하지 않으면서도, ARC처럼 필요한 순간에만 결합을 확대해야 한다는 점이다. 이를 위해 AO-ARC는 상태-비용(state-cost) 공간에서 “makespan ≤ B”를 feasibility로 동치화하고, 국소 수리 구간에 대해 모든 로봇의 동기화된 수리 시간을 보수적으로 제한하는 local bound Bhlocal을 계산한다. 또한 국소 문제에서 해가 실패하면 동일 충돌을 중심으로 시간 창과 공간 bound를 확장해 최악의 경우 전역 합성 bounded 문제까지 회복하도록 설계했다.

- **Empirical Impact**: 이 방법은 로봇 수가 증가할수록 기존 anytime MRMP 대비 초기 해는 state-of-the-art feasibility 솔버 수준으로 빠르게 내면서, 수렴 속도와 신뢰성을 동시에 개선하는 것으로 보고된다. 2D 모바일 로봇 시나리오에서는 조정(coordination) 복잡도에 따른 시간-품질 관계를 분석했고, 3D 매니퓰레이터 실사용을 대표하는 설정에서도 성능을 확인했다. 또한 Panda 매니퓰레이터 4대/8대의 랜덤 태스크에서 AO-MRMP 계열 대비 수렴이 더 빠른 anytime 거동을 보이며, 멀티로봇 planning에서 “빠른 초기 계획 + 점진적 품질 향상”을 현실적으로 동시에 달성할 가능성을 강화했다.



### Support-Constrained RL Enables Real-World Policy Improvement without Real-World Experienc (https://arxiv.org/abs/2606.27475)
Comments:
          35 pages, 23 figures

- **Prior Approaches**: 기존에는 시뮬레이션에서 RL로 정책을 개선하더라도, 접촉(contact)·동역학(dynamics) 불일치 때문에 안전하지 않거나 실물로 전이되지 않는 행동이 생기기 쉽다. 분포 거리 기반 regularization(행동모방·판별자 패널티 등)은 전이를 돕지만, 종종 기존 행동 prior를 과도하게 고정해 개선 폭을 제한한다. 또한 Residual RL은 기준 정책의 보정에 그쳐, 기준 정책이 이미 잘못된 전략을 선택하면 큰 전략 전환이 어렵다.

- **Core Contribution**: 본 논문은 실물 데이터로 사전학습된 generative policy의 support(행동 생성 가능 범위) 안에서만 시뮬레이션 RL을 업데이트하도록 강제하는 Support-Constrained Off-Domain REinforcement, SCORE를 제안한다. SCORE는 flow steering으로 base policy가 만들 수 있는 행동만 선택해, 전이 가능한 행동만을 대상으로 최대한의 개선을 노린다. 특히 RL은 sparse rewards로 학습하며 distillation 없이도 base policy를 그대로 둔 채 최소한의 추가 노력으로 fine-tuning이 가능하다고 밝힌다.

- **Technical Challenges**: 핵심 기술적 난제는 ‘시뮬레이션에서의 최적화’가 ‘실물에서의 realizable(전이 가능) 행동’으로만 수렴하도록 제약을 설계하는 것이다. SCORE는 RFS 스타일의 비대칭 actor-critic과 flow matching 기반 정책을 결합하되, 학습된 steering이 base policy의 action support를 벗어나지 않도록 만드는 방식으로 이 제약을 구현한다. 실험적으로는 BC regularization을 추가해도 시뮬레이션 개선과 전이 사이의 민감한 tradeoff만 커져, 최종 비교에서는 unregularized 설정을 사용한다.

- **Empirical Impact**: 8개 실세계 고난도(다지 multi-fingered) 조작 과제에서 SCORE의 평균 성공률은 37.8%→89.9%로 크게 상승했으며, 최강 baseline 대비로도 59.5%에서 앞섰다. 또한 base policy 대비 성공까지 걸리는 단계 수는 36.8% 더 적게 도달했으며, 시뮬레이션 성공이 곧 실물 성능으로 직결되지 않는 문제를 정량·정성 모두로 보여준다. 실험 및 ablation 결과는, 제한 없는 시뮬레이션 RL이 simulator-specific 보상을 악용하거나 접촉을 잘못 활용해 실패하는 반면, SCORE는 전이 가능한 행동 범위를 유지하며 개선을 달성하는 새로운 real-to-sim-to-real 패러다임임을 시사한다.



### LLawCo: Learning Laws of Cooperation for Modeling Embodied Multi-Agent Behavior (https://arxiv.org/abs/2606.28182)
Comments:
          Accepted to ICML 2026

- **Prior Approaches**: 기존의 LLM 기반 communicative embodied agent 연구는 대체로 자연어 대화와 행동을 end-to-end로 연결해 협력을 시도하지만, 파트너와 환경/작업 상태에 맞지 않는 행동을 내면서 협력이 비효율적으로 되는 문제가 반복된다. 또한 성능을 끌어올리기 위해 stronger model 증류나 비학습 파이프라인에 의존하는 경우가 많아, 상호작용을 통해 에이전트가 자율적으로 계속 개선하기 어렵다.

- **Core Contribution**: 논문은 Learning Laws of Cooperation (LLawCo)을 제안해, 에이전트가 과거 실패를 되돌아보며 “Talk when necessary”, “Wait for partner” 같은 고수준 행동 법칙을 스스로 추출·정렬하도록 한다. 이렇게 얻은 법칙을 supervised fine-tuning으로 에이전트의 reasoning 체인에 명시적으로 내재화해, 작업 목표와 다른 에이전트의 행동에 동시에 정렬되게 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 협력 실패에서 의미 있는 법칙을 뽑아내고 (2) 그 법칙에 맞는 성공 궤적만 골라 학습 데이터를 구성하며 (3) 추론 시 법칙을 실제 행동으로 안정적으로 연결하는 것이다. 저자들은 실패 에피소드에서 실패 원인을 추출해 법칙 집합을 생성하고, 성공 에피소드 중 법칙 정합성을 만족하는 샘플만 유지한 뒤, 해당 법칙에 근거한 reasoning과 행동을 SFT 데이터로 만들어 law-guided inference에서 법칙을 명시적으로 제공한다.

- **Empirical Impact**: PARTNR-Dialog(신규 대규모 멀티에이전트 communicative benchmark)와 TDW-MAT에서 실험한 결과, 4종 백본 LLM 전반에 걸쳐 협력 효율과 task success rate가 일관되게 향상됐다. 특히 PARTNR-Dialog 벤치마크에서 평균 성공률이 4.5%, TDW-MAT에서 평균 6.8% 개선되었고, 동시에 추론 시 법칙을 수정해도 업데이트된 제약을 안정적으로 따르는 controllability까지 확인했다.



### Regularized Reward-Punishment Reinforcement Learning (https://arxiv.org/abs/2606.28152)
- **Prior Approaches**: Reward–Punishment Reinforcement Learning(RPRL)은 보상과 벌을 분리된 모듈로 다루지만, 기존 MaxPain/DMP/softDMP 계열은 두 경로가 주로 가치 혼합이나 엔트로피(또는 mellow-max/min) 수준에서만 상호작용한다고 가정하는 경향이 있습니다. 그 결과 동기 시스템이 학습 과정에서 구조적으로 서로의 행동을 ‘함께’ 형성하는 정책 수준의 결합은 제한적입니다. 또한 KL 정규화는 보통 탐색 촉진(soft-optimality)이나 신뢰영역(trust-region) 안정화에 쓰였고, 다중 동기 간 조정 메커니즘으로는 충분히 확장되지 않았습니다.

- **Core Contribution**: 논문은 KL-Coupled Policy Regularization(KCPR)을 제안해 보상-추구 정책과 벌-회피 정책이 서로를 ‘동적으로 학습된 prior’로 삼도록 정책 수준 결합을 직접 구현합니다. 이로부터 KL-Coupled Soft Optimality(KCSO)를 도출하고, 이를 딥 환경에서 학습 가능한 형태로 구현한 klDMP를 제안합니다. KCSO는 reward와 punishment 정보가 가치 전파에서 함께 영향을 주도록 KL-regularized Bellman 연산자를 설계해, 기존의 독립 최적화 가정을 정책 결합으로 대체합니다.

- **Technical Challenges**: 핵심 기술 난관은 companion prior가 초기의 부정확한 값 때문에 과도하게 확신해지고, 그 결과 KCSO 백업에서 대안 행동의 가치 전파가 억제되며 학습이 불안정해질 수 있다는 점입니다. 이를 해결하기 위해 companion-prior softening으로 prior를 균등분포와 보간하여 초기 탐색성을 확보하고, 동시에 KL 결합 구조는 유지합니다. 또한 보상/벌 데이터 불균형 문제를 완화하기 위해 separate replay-buffer를 실험적으로 점검하고, discriminator 기반으로 각 모듈에 더 적절한 전이를 배정하도록 설계합니다.

- **Empirical Impact**: 그리드월드 및 Gazebo 로봇 내비게이션 같은 실험에서 klDMP는 DQN, SQL, softDMP 대비 충돌 위험을 낮추면서도 과제 성능은 경쟁 수준으로 유지했습니다. 특히 softening을 포함했을 때 초기 단계에서 정책이 과도하게 뾰족해지는 문제를 줄여, reward와 punishment 가치 전파의 균형과 학습 안정성이 개선되는 양상이 보고됩니다. 전반적으로 이 결과는 ‘정책 수준 조정(policy-level coordination)’이 여러 동기 목표를 통합해 효율성과 안전의 트레이드오프를 다루는 설계 원리로 유용할 수 있음을 시사합니다.



### PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation (https://arxiv.org/abs/2606.28128)
Comments:
          Github: this https URL Project website: this https URL

- **Prior Approaches**: 기존 비디오 생성은 시각적 품질이나 액션 조건 만족에 집중하는 경우가 많아, 접촉이 많은 조작에서 불연속 궤적·물체 침투·중력 위반 같은 물리 부정합이 남는 한계가 있었다. 로봇용 데이터로 fine-tuning한 모델도 reconstruction류 목적이 배경과 상호작용 구역을 균일하게 다뤄, 접촉 주변의 국소 물리 오류와 전역 상호작용 결과 오류를 동시에 잡기 어렵다.

- **Core Contribution**: 이 논문은 물리적 타당성을 (1) 픽셀 레벨 국소 운동, (2) 시맨틱 레벨 상호 관계의 계층적 정렬, 그리고 (3) 조작에 중요한 영역에만 감독을 거는 region-focused 정렬 문제로 재정의한다. 이를 바탕으로 PhysisForcing을 제안해, 접촉·조작기·움직이는 물체 같은 physics-informative regions에 한정해 궤적 연속성과 객체-로봇 관계성을 함께 강화한다.

- **Technical Challenges**: 핵심 기술 난점은 두 가지인데, 접촉 구간이 국소적으로만 정보가 강하다는 점(균일 감독의 희석)과, 국소 궤적 문제만으로는 전역 상호작용 결과가 보정되지 않는다는 점이다. 저자들은 참고 비디오의 point tracking과 깊이 기반 전경 가중치를 이용해 physics-informative 마스크를 만든 뒤, (i) DiT 중간층 feature에 대한 pixel-level trajectory alignment(마스크 내 포인트 궤적 MSE)와 (ii) frozen video understanding encoder로부터 얻은 토큰 간 관계를 DiT에 전이하는 semantic-level relational alignment(관계 유사도 정렬)를 joint로 최적화한다.

- **Empirical Impact**: R-Bench, PAI-Bench, EZS-Bench에서 강한 베이스라인 대비 일관된 개선이 확인됐고, 특히 R-Bench에서 Wan2.2-I2V-A14B는 22.3%, Cosmos3-Nano는 9.2% 물리 일관성 향상을 보였다(일부는 vanilla finetuning 대비 7.1~3.7%p 추가 개선). 더 나아가 WorldArena action-planner 프로토콜에서 closed-loop success rate를 16.0%→24.0%로 끌어올려 world model 계획(planning) 성능과 downstream 정책 성공에도 긍정적임을 보여, 물리 정렬된 비디오 모델이 로보틱스 표현 학습에 유리하다는 신호를 제공한다.



### AI-Driven Synthesis for High-Tech System Design: Automating Innovation (https://arxiv.org/abs/2606.28126)
- **Prior Approaches**: 기존 공학 설계는 토폴로지(이산) 선택과 치수·성능(연속), 제어(활성 동역학)를 순차적으로 다루는 경우가 많아 전체 설계 공간을 통합 탐색하지 못한다. 시뮬레이션 기반 최적화는 제약을 만족시키면서도 조합 폭발이 커지면 계산 비용이 급증하고, CAD 중심 반복 작업은 병목이 된다.

- **Core Contribution**: 이 논문은 automation-in-design(AiD) 패러다임 아래 computational design synthesis(CDS)로, 이산 토폴로지 합성과 연속 성능·치수 최적화를 통합 자동화하는 프레임워크를 제안한다. RL 기반으로 토폴로지를 탐색하고, 물리 기반 NLP/공간 최적화를 결합해 최소한의 인간 개입으로 “설계 생성→검증→개선” 흐름을 만든다.

- **Technical Challenges**: 핵심 난관은 (1) 엄격한 물리 제약을 어기지 않으면서 이산-연속 혼합 설계공간을 효율적으로 탐색하는 것, (2) 배치·루팅 같은 공간 패키징을 CAD의 비분해성 때문에 미분 기반 최적화에 바로 태우기 어렵다는 점이다. 논문은 이를 위해 RL-NLP의 bi-level, solver-in-the-loop 구조로 제약 인식을 강화하고, maximal disjoint ball decomposition(MDBD)로 형상을 미분 가능 추상화해 배치·루팅·물리 최적화를 연속 최적화 문제로 바꿔 해결한다.

- **Empirical Impact**: 사례 1에서는 자동차 기어박스 토폴로지 최적화에서 BF 대비 평가 시간이 3차례 자릿수 수준으로 감소했으며, 최적값 대비 2% 이내 오차 예측과 전 구성에서의 물리적 타당성(stress/packing)을 보고한다. 사례 2에서는 MDBD 기반 배치·루팅이 해석적 벤치마크에서 부피·루팅 길이 차이를 약 0.6~2% 범위로 좁혀 정확성과 실용성을 검증했으며, 결과적으로 CAD 반복 의존을 줄이면서 수천 개 공간 구성을 빠르게 탐색할 가능성을 제시한다.



### RS-Diffuser: Risk-Sensitive Diffusion Planning with Distributional Value Guidanc (https://arxiv.org/abs/2606.27766)
Comments:
          ICIC 2026 Oral

- **Prior Approaches**: 오프라인 강화학습은 데이터만으로 정책을 학습해 안전한 의사결정을 노리지만, 데이터 분포 밖에서 행동할 때 분포 이동과 불안정한 가치 추정 문제가 커진다. 확산(difussion) 기반 결정/플래닝은 멀티모달 궤적을 잘 만들지만, 대부분 기대수익을 중심으로 최적화해 꼬리(tail) 위험을 명시적으로 다루지 못한다. 그 결과 평균 성능은 비슷해도 최악의 결과가 크게 다른 상황에서 안전성이 약화될 수 있다.

- **Core Contribution**: RS-Diffuser는 확산 플래너(미래 상태 궤적 생성)와 분포형 가치 critic(리턴 분포 추정)을 결합해 오프라인 확산 플래닝을 ‘위험 민감’하게 만든다. 또한 Monte Carlo 기반 분포형 학습(quantile regression)으로 리턴의 하위 꼬리를 직접 반영하고, 샘플링 시점에 CVaR 같은 tail-aware 목적의 그래디언트를 denoising 과정에 가이드한다. 이를 통해 하나의 학습된 모델을 risk parameter만 바꿔 risk-averse/risk-neutral/risk-seeking으로 유연하게 전환한다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 확산 생성 과정에 (2) 분포형 critic이 주는 꼬리 민감도를 안정적으로 연결하는 것이다. RS-Diffuser는 상태만으로 미래 궤적을 생성하고, inverse dynamics로 실행 가능한 행동을 디코딩해 생성/통제를 분리했으며, critic은 quantile 분포를 학습해 VaR/CVaR 같은 꼬리 지표를 미분 가능한 가이던스 항으로 구성했다. 마지막으로 샘플링 시점의 risk-sensitive guidance를 통해 denoising 경로를 원하는 위험 프로필로 조향한다.

- **Empirical Impact**: 실험은 risk-sensitive D4RL과 risky robot navigation 벤치마크에서 평균 성능뿐 아니라 worst-case(예: CVaR0.1) 견고성과 안전 위반을 함께 개선함을 보여준다. 비교군 중 위험 중립 방법(CQL 등)은 기대수익 중심이라 꼬리 위험을 놓쳐 성능이 떨어졌고, 기존 위험 민감 기법들은 고정된 위험 목표에 매여 있어 상황 전환성이 제한적이었다. RS-Diffuser는 단일 모델로도 위험 파라미터 변화에 따라 CVaR·VaR 기반 동작 특성을 재현하며, 특히 CVaR0.1이 평균-견고성의 균형이 가장 좋다는 분석도 제시한다.



### Characterizing Driver Interactions with Autonomous Vehicles via Response Maps (https://arxiv.org/abs/2606.27656)
- **Prior Approaches**: 기존 연구는 고립 주행에서의 운전자 모델링은 축적됐지만, 교차로처럼 한 차량의 행동이 다른 차량의 행동을 바꾸는 상호작용에서는 상태-제어가 분리되지 않아 동일한 틀로 설명하기 어렵다. 게임이론 기반 접근은 결합성을 활용하지만 최적성 같은 가정에 기대는 경우가 많고, 인간 요인 모델은 저수준 운동제어보다는 고수준 의사결정에 초점이 치우치기 쉽다. 데이터 기반 접근도 있으나, 인간 반응을 “다른 에이전트의 상태에 대한 피드백 법칙” 형태로 해석 가능하게 정리하는 데 한계가 있었다.

- **Core Contribution**: 이 논문은 인간-AV 상호작용을 결합 상태공간에서의 feedback law로 보고, 이를 response map(응답 지도)으로 정의해 인간의 조작(가속/감속 등)을 AV 행동에 따라 예측·해석한다. 특히 response map을 최적해가 아닌 “상호작용 중 상대 상태에 의존하는 임의의 피드백 법칙”으로 해석해, 인간의 자연스러운 반응을 최적성 가정 없이도 모델링한다. AV 유형(예: Yield/NoYield/Contingent)별로 학습된 맵을 비교함으로써, 사람의 반응이 AV의 양보 여부와 같은 행동 특성에 실제로 달라진다는 점을 정량화한다.

- **Technical Challenges**: 핵심 과제는 (1) 인간 조작이 상대 차량의 위치·속도 같은 결합 상태에 어떻게 반응하는지, (2) 자연 데이터에서 이를 안정적으로 추정하는 일이다. 저자들은 response map을 관측된 주행 데이터의 국소선형 근사로 두고 회귀(regression)로 w_h(인간 상태 영향)와 w_av(AV 상태 영향)를 추정하며, 기반 정책을 중심으로의 미세한 가속/제동 차이를 포착하도록 설계했다. 또한 반응 지연을 고려하기 위해 관측 구간과 reaction time window를 정의하고, leave-one-out validation으로 예측 오차를 확인해 모델 적합성을 점검한다.

- **Empirical Impact**: StrangeLand 주행 시뮬레이터에서 50명 참가자가 교차로 상호작용을 수행한 데이터를 바탕으로, 대부분의 학습 모델이 낮은 mean squared error를 보이며 데이터 일반화를 입증한다. heat map 형태로 해석했을 때도 사람이 AV가 멀면 가속하고, 바로 앞이면 대기/감속하며, 가까우면 제동하는 등 상식적이면서도 AV 유형별 차이를 일관되게 드러냈다. 특히 Yield/Contingent는 유사 반응을 보이지만 NoYield에는 “제동 대신 가속(미리 예상)” 같은 뚜렷한 대응이 나타나며, 이 결과는 AV 행동을 바꾸면 인간 상호작용 양상이 실제로 조절될 수 있음을 시사한다.



### Radar Guided Camera Verification for Automatic Emergency Braking Rethinking Object Detection in Radar Camera Fusion (https://arxiv.org/abs/2606.27556)
Comments:
          8 pages, 8 figures

- **Prior Approaches**: 기존 radar–camera fusion AEB는 레이더로 표적을 찾은 뒤 카메라가 deep learning 기반 object detection으로 장애물을 “찾고(위치)” “무엇인지(라벨)”까지 인식하는 흐름이 주류였습니다. 이 방식은 성능을 올렸지만 계산량과 하드웨어 요구가 커지고, 카메라가 전 프레임을 대상으로 탐색해야 한다는 가정이 부담이 됩니다.
또한 레이더가 이미 위치를 제공하는 radar-led 파이프라인에서는 카메라의 역할이 검증(verification)으로 축소될 수 있다는 문제의식이 제기돼 왔습니다.

- **Core Contribution**: 본 논문은 레이더가 투영한 이미지 ROI에서 장애물 유무만 판단하도록 하는 radar-scoped edge density gate를 제안합니다. 카메라는 객체 인식 대신 “장애물 존재 확인”을 수행하며, 학습 데이터·모델 가중치·GPU 가속 없이 동작하도록 설계됐습니다.
이를 brake-by-wire가 포함된 완전한 radar–camera fusion AEB 시스템에 통합해 실제 차량 주행에서 평가했습니다.

- **Technical Challenges**: 핵심 과제는 레이더-카메라 투영 ROI에 작은 투영 오차가 생겨도 검증 성능과 AEB 안전성이 유지되도록 하는 것입니다. 논문은 CAN 기반 레이더 추적(Kalman filter, 다중 업데이트 확인) 후 핀홀 모델로 ROI를 만들고, ROI에서 Canny edge를 계산해 edge density가 임계값을 넘으면 장애물로 “게이트 통과”시키는 단일 스칼라 임계값 전략을 사용했습니다.
구현 측면에서는 ROI 크기를 줄여 처리 시간을 확보하면서도, 고정 임계값으로 충분한 recall(미탐 최소화) 지점을 유지하는 균형을 맞췄습니다.

- **Empirical Impact**: 실측 실험에서 ROI 기반 처리는 전체 프레임 대비 탐색 범위를 최대 98.7% 줄였고, ROI당 평균 지연이 0.121 ms로 측정됐습니다. 검증 성능은 AUC 0.898, recall 0.994로 보고되며, staged threat 시나리오 33개에서 missed brake event는 0건이었습니다.
추가로 72개 주행 세션/131,603 프레임 기반 결과로, detector 기반 confirmation을 대체할 수 있는 경량 카메라 검증 접근의 가능성을 실증한 것으로 평가됩니다.



New uploads on arXiv(cs.MA)

### GBC: Gradient-Based Connections for Optimizing Multi-Agent Systems (https://arxiv.org/abs/2606.28187)
Comments:
          15 pages, 8 figures, accepted by SIGDIAL 2026 Long Papers

- **Prior Approaches**: LLM 기반 멀티에이전트 시스템은 역할 분담과 구조화된 상호작용으로 복잡한 작업을 풀 수 있지만, 종종 강한 single-agent 대비 성능이 흔들립니다. 주된 원인은 에이전트 간 miscoordination과 효율적 검증 부재, 그리고 실패 원인을 특정 에이전트/상호작용 단계로 정확히 되돌려 보내는 credit assignment의 부재입니다. 기존 prompt 최적화·gradient 유사 방법은 보통 전체 성공 같은 거친 피드백을 써서, 토큰·단계 단위로 어떤 입력이 오류를 낳았는지 추적하기가 어렵습니다.

- **Core Contribution**: 이 논문은 Gradient-Based Connections(GBC)로 멀티에이전트 워크플로를 계산 그래프로 모델링한 뒤, 토큰 레벨에서 각 에이전트 출력이 downstream 에이전트에 미치는 영향을 gradient 기반 연결 가중치로 정량화합니다. 이를 attribution graph로 구성해 task loss 신호를 역전파함으로써 오류의 근원(어떤 에이전트/토큰/단계)을 정밀하게 찾아내고, 그에 맞춘 prompt 최적화를 유도합니다. 또한 이를 실제로 돌리기 위한 확장 프레임워크 AgentChord를 제안하며, prefix 기반 gradient 계산으로 메모리 부담을 줄입니다.

- **Technical Challenges**: 핵심 기술 난제는 (1) 멀티에이전트의 상호작용을 토큰 단위로 추적할 만큼 fine-grained attribution을 제공하면서도 (2) LLM에서 역전파 비용을 감당 가능하게 만드는 것입니다. GBC는 에이전트 그래프 위에 attribution graph를 만들고, 토큰 salience를 gradient 기반 신호로 계산해 loss를 그래프를 따라 backward 전파합니다. AgentChord에서는 prompt 접두사는 gradient 없이 KV cache로 처리하고 입력 토큰에 대해서만 gradient를 계산하는 prefix 기반 전략을 써서 메모리 사용을 크게 낮춥니다.

- **Empirical Impact**: MultiWOZ(태스크 지향 대화)와 τ-bench(툴 사용)에서 GBC 기반 최적화는 pre-optimization 대비 대부분의 지표를 일관되게 끌어올렸고, 경우에 따라 strong single-agent 베이스라인을 능가했습니다. 특히 MultiWOZ에서는 JGA와 Slot F1이 크게 개선되며, 분석 결과 attribution quality가 높을수록 최적화 효과도 커지는 상관이 관찰됐습니다. τ-bench에서도 overall reward가 상승하고, 오류 분포는 retrieval/식별 실패·툴 오용·매니저 지시 불명확 등 장기 워크플로의 근본 문제와 맞물려 개선되는 양상을 보였습니다.



### GenWorld: Empirically Grounded Urban Simulation Infrastructure for Scalable LLM-Agent Studies (https://arxiv.org/abs/2606.27650)
Comments:
          27 pages, 24 figures. Code: this https URL. Project page: this https URL

- **Prior Approaches**: 기존 LLM-agent 시뮬레이션은 ‘실제 도시 제약을 반영한 grounding’과 ‘인구 규모로의 scaling’이 동시에 어렵다는 공통 한계를 가진다. 온라인으로 LLM을 직접 호출하면 도시 단위(대규모 에이전트)에서는 계산 비용이 급격히 커져 현실적인 롤아웃이 제한된다. 또한 에이전트와 환경을 연결하는 방식이 비정형이어서 재현성과 검증이 약한 경우가 많았다.

- **Core Contribution**: 이 논문은 GenWorld라는 도시 스케일 LLM-agent 실험 인프라를 제안해, building-level 합성 도시 구축과 구조화된 에이전트-환경 인터페이스, 그리고 오프라인 컴파일을 결합한다. 핵심 아이디어는 LLM에서 얻은 결정 신호를 lookup policy로 변환해 대규모 롤아웃 중에는 반복 호출 없이 빠르게 실행하는 것이다. 그 결과 실도시 제약을 반영하면서도 city-scale 에이전트 시뮬레이션을 재현 가능하게 만든다.

- **Technical Challenges**: 기여를 위해서는 (1) 합성 거주자와 도시 공간을 실제 데이터에 맞춰 정밀하게 접지하고 (2) LLM의 의사결정 출력을 스케일 가능한 정책 형태로 안정적으로 컴파일하며 (3) 롤아웃 중 재계획이 추적 가능해야 한다. 논문은 Higashihiroshima, Japan 인스턴스에서 196,608명의 synthetic residents를 census·geospatial 데이터로 생성하고, census tabulation으로 인구 통계 일관성을 검증한다. 또한 YJMob100K 모바일폰 데이터로 통근 거리 진단을 수행하고, 세 가지 사례에서 경고-응답 섭동 시 auditable replanning traces까지 제공해 신뢰성을 보완한다.

- **Empirical Impact**: 실험은 전체 도시 평일 롤아웃, 평일-주말 행동 대비, 경고-응답 섭동(perturbation)에 대한 재계획 추적까지 포함하는 3가지 재현 가능한 케이스로 구성된다. 이를 통해 GenWorld가 grounded(데이터 접지)하면서도 scalable(대규모 실행)한 LLM-agent 연구 플랫폼이 될 수 있음을 보여준다. 다만 교통·대피·정책 결과 같은 ‘교정된’ 예측(forecasting)은 향후 과제로 남겨, 현재는 인프라의 재현성과 실험 가능성에 무게를 둔 성격이다.



### QueenBee Planner: Skill-Evolving Communication Topologies for Token-Efficient LLM Multi-Agent Systems (https://arxiv.org/abs/2606.27492)
- **Prior Approaches**: 기존 연구는 LLM 단일 에이전트의 추론 구조(체인/트리/그래프 사고)나 에이전트 간 역할·고정 프로토콜에 초점을 맞췄다. 또 일부는 통신 그래프/토폴로지를 탐색해 성능을 높이지만, 보통 특정 태스크에서 설계하고 끝나며 이후 경험을 ‘토폴로지 설계 지식’으로 누적하지 못한다.

- **Core Contribution**: 이 논문은 에이전트 간 communication topology를 ‘회수 가능하고 self-improving 되는 설계 스킬’로 재정의하고, worker 풀은 고정한 채 외부 LLM planner만 학습한다. planner가 temporal communication DAG를 생성해 누가 어느 라운드에 무엇을 보내고, 누가 메시지를 병합하며, 최종 답은 어디서 나오게 할지를 결정하고, 실행 로그를 증거 기반 설계 규칙으로 증류한다.

- **Technical Challenges**: 핵심 난제는 self-evolution이 운 좋은 실행이나 그럴듯하지만 틀린 설명을 정책으로 굳혀 drift를 일으키는 문제다. 이를 막기 위해 held-out acceptance gate, 분산을 반영한 credit(variance-aware), motif-level attribution으로 구조 단위의 기여를 분해, transfer trust로 태스크 슬롯별 전이 가능성만 선택적으로 사용, insight falsification과 structural deduplication으로 검증되지 않은 규칙/중복 규칙의 유입을 제어한다.

- **Empirical Impact**: Count-Frequency 집계(CF)와 Silo-Bench 스타일 분산 조정 과제에서, 고정 토폴로지 및 cold 생성 대비 통신 DAG 생성 성능이 개선됨을 보였다. 예를 들어 CF fulltest에서 최강 고정 토폴로지의 RMSE 12.53을 7.87로 낮추는 동시에 메시지 수, 모델 호출, 토큰 비용도 줄였고, Silo 계열에서도 같은 방향의 향상이 관찰됐다. 결과적으로 multi-agent 시스템이 정답을 암기하는 수준을 넘어, 통신 구조 설계 아키텍처 지식을 재사용·축적할 수 있음을 시사한다.



### Glite ARF: Verifier-Driven Research with Parallel LLM Coding Agents (https://arxiv.org/abs/2606.27416)
Comments:
          13 pages, 6 figures, 7 tables. Open-source framework (Apache-2.0) and a public demo project at this https URL and this https URL

- **Prior Approaches**: LLM coding agents를 활용한 자동 연구는 여러 실험을 병렬로 돌리기 쉬워 보이지만, 지시 일부를 놓치거나 규칙을 서술에 의존하면 잘못된 코드/인용/데이터 분할이 누적돼 재현 불가능한 산출물이 생긴다. 기존 연구 자동화(예: AutoGen, MetaGPT, CAMEL, CrewAI 등)와 코드 생성 에이전트(SWE-agent 등)는 작업 수행은 돕지만, 장기 캠페인의 캠페인 수준 무결성과 감사를 보장하는 구조는 상대적으로 약하다.
또한 단일 이슈 해결 중심 벤치마크(SWE-bench)처럼 ‘연구 무결성’ 자체를 검증하는 평가 틀은 부족하다.

- **Core Contribution**: Glite ARF는 연구 과정을 ‘실행 가능한 규칙’으로 만들어, LLM coding agents가 병렬로 작업해도 재현성과 감사 가능성을 유지하도록 하는 오픈소스 Python 프레임워크다. 사람 연구자가 가설을 고르고, 코딩 에이전트(Claude Code, Codex CLI)가 정해진 구조의 작업을 수행하며, 결정적 Python verifier가 작업 격리, 완료 산출물 불변성, 수정(overlay), 프로젝트 개요(materialised overview)를 강제한다.
저자들은 이를 verifier-driven research로 명명하며, 에이전트에게 프롬프트로 ‘따르라’고 요구하는 대신 위반 시 코드가 즉시 실패하도록 설계했다고 강조한다.

- **Technical Challenges**: 핵심 기술 난제는 에이전트가 장시간·다수 작업에서 조용히 규칙을 위반하거나(오프스펙 파일 수정, 누락/오류 로그, 컨텍스트 열화), 생성된 산출물 간 오염(예: target leakage)이 발생하는 문제를 구조적으로 차단하는 것이다. Glite ARF는 작업 단위로 git 브랜치/폴더/allow-list를 고정하고, pre-merge verifier가 폴더 밖 변경을 차단하며, 완료된 작업은 불변으로 두고 후속 수정은 correction overlay로만 반영되게 만든다.
또한 모든 artefact에 버전된 specification과 verifier를 붙여 포맷/형식 진화를 감시하고, 실행 로그를 기록·검증(run_with_logs)하며, subagent 체인에서 단계별 컨텍스트를 분리해 문맥 열화 영향을 줄였다.

- **Empirical Impact**: BEA 2026 vocabulary-difficulty shared task에서 Glite ARF로 개발한 시스템은 3개 언어(스페인어/독일어/만다린) 모두에서 closed track 1위, open track은 모든 언어에서 2위에 올랐다. 공식 baseline RMSE를 closed 29.9%, open 35.9% 낮췄고, 전체 캠페인은 273개 tracked task(146 experiment runs), 129개 feature set을 포함하며 LLM API 비용은 약 $450 수준(제3자 총액 $498)으로 집계됐다.
특히 per-fold provenance와 버전 스펙 덕분에 plausibility가 깨진 RMSE(0.609)를 유도한 target leakage feature 4개를 분 단위로 국소화·제거해 RMSE를 0.802로 ‘수정된 추정치’로 복원한 사례를 제시한다.



### Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placemen (https://arxiv.org/abs/2606.27409)
Comments:
          20 pages, 5 figures, 1 table. Code and data: this https URL

- **Prior Approaches**: 기존 멀티에이전트 LLM 연구는 verifier/critic 에이전트로 환각을 줄이려 하지만, 검증 피드백이 상호작용 지연(latency) 때문에 늦게 들어온다는 점을 안정성 관점에서 정량화하지 못했습니다. 또한 신뢰도·증거 기반 집계나 debate 설계는 오류 전파(정적)를 다루는 경우가 많고, 지연이 만드는 동적 불안정(진동/발산) 메커니즘은 분석되지 않았습니다.

- **Core Contribution**: 이 논문은 검증이 지연된 상태에서 에이전트 네트워크가 ‘지연된 consensus’로 수렴하는 과정을 그래프 모델로 정식화하고, corrector(진실을 고정하는 노드)가 있는 경우의 안정성 조건을 도출합니다. 특히 검증의 강도(dose)와 지연(delay) 사이에 닫힌형(closed-form) 안정성 임계값을 제시하며, correction이 너무 강하거나 너무 늦으면 consensus가 진동으로 바뀔 수 있음을 보여줍니다. 또한 제한된 corrector budget 하에서 영향력 있는 노드를 고르는 placement 목적함수(초모듈러)와 greedy의 (1-1/e) 근사 보장까지 연결합니다.

- **Technical Challenges**: 핵심 기술적 난제는 고차원 지연 시스템의 안정성을 그래프 스펙트럼과 함께 해석하는 것이며, 이를 위해 grounded Laplacian에 의한 스펙트럴 분해로 지연 방정식을 독립적인 스칼라 지연 재귀로 분해합니다. 그 결과 각 모드가 단위원(unit disk)을 이탈하는 조건을 추적해 ‘검증 dose 한계’의 정확한 경계를 만들었고, 두 지연(커뮤니케이션 지연과 검증 지연)이 동시에 작동할 때 최악 구간이 동시 지연 코너임을 밝혀 임계값이 (지연 2에서) inverse golden ratio가 됨을 산출합니다. 마지막으로 placement 최적화에서는 resolvent의 coherence가 초모듈러임을 이용해 greedy가 네트워크의 amplifier/bridge 노드에 예산을 집중하도록 설계합니다.

- **Empirical Impact**: 5개의 오픈 모델에서 예측된 dose-delay oscillation(검증 강도·지연이 임계값을 넘을 때 진동 전환)이 실제 수치 실험으로 재현됩니다. 반대로 grounded factual answering(진실을 흡수 경계로 만드는 설정)에서는 같은 지연이 수렴을 깨지 못해, 불안정성은 signed-belief 과제(부호가 있는 믿음/오류 부피드백)에서만 나타나는 특이성을 갖는다고 주장합니다. 이는 단순히 verifier를 ‘추가’하는 수준을 넘어, 검증 정책(강도·지연·배치)을 안정성-성능 최적화 대상으로 다루게 만드는 실증적 근거를 제공합니다.



### SidConArena: An Environment Evaluating Agents in Open-Ended,Positive-Sum Bargaining Gam (https://arxiv.org/abs/2606.27397)
Comments:
          15 pages

- **Prior Approaches**: 기존 게임형 LLM 에이전트 평가는 주로 zero-sum 또는 적대적 경쟁에 치우쳐, 혼합동기 경제 상호작용의 복합 요구(협력으로 잉여 창출+희소자원 경쟁)를 충분히 드러내지 못했다. 또한 일부 협상/다중에이전트 벤치마크는 동적·부분관측·규칙 기반 평가를 모두 만족시키지 못해, 실제 에이전트의 가치평가·자원배분·장기계획을 진단하기 어렵다는 한계가 있다.

- **Core Contribution**: 이 논문은 open-ended, positive-sum bargaining 상황에서 LLM 에이전트를 평가하도록 SidConArena를 제안한다. SidConArena는 유한한 지평의 부분관측 확률 게임(POSG) 구조를 바탕으로, 자연어 협상(구속 거래), 결정론적 converter 기반 생산, 밀봉경매(장기자산)를 한 프레임워크에서 결합한다.

- **Technical Challenges**: 핵심 과제는 자유로운 자연어 협상을 허용하면서도 거래·생산·경매 같은 수치적 행동을 게임 엔진 규칙에 엄밀히 고정(검증)하는 것이다. 이를 위해 위상(phase) 인식 dispatching, structured observation, neural-symbolic action interface, 비동기 이벤트 기반 실행을 결합해 문장 생성은 열어두되 실행은 규칙 검증과 동기화로 관리한다.

- **Empirical Impact**: 실험 결과, 동종 self-play과 이종 Elo tournament 모두에서 더 강한 frontier 모델이 더 높은 단말(terminal) 경제 성과를 보이면서 모델 간 구분 능력을 입증한다. 동시에 에이전트들은 자원 과대평가(예: Ships), 협상에서의 수동성, 그리고 delayed-return에 기반한 장기 투자 계획 실패 같은 반복적 한계를 보이며, 단순한 문법적 행동 적합성이 경제적 역량으로 이어지지 않음을 드러낸다.



### Which Nash Equilibrium? Solver-Dependent Selection on Zero-Sum Nash Polytopes (https://arxiv.org/abs/2606.28308)
Comments:
          18 pages, 9 figures

- **Prior Approaches**: 기존 게임 솔버는 2인 영점합 게임에서 Nash equilibrium을 ‘유일한 목표’처럼 취급해 왔습니다. 특히 Nash 집합이 다면체(폴리토프)처럼 여러 해를 가질 때도, 솔버 간 선택이 다르다는 점은 거의 다루지 않았습니다.

- **Core Contribution**: 이 논문은 서로 다른 알고리즘(특히 regret-averaging 계열 vs regularized last-iterate 계열)이 Nash 집합의 서로 다른 ‘멤버’를 실제로 선택하는지, 그리고 그 선택이 알고리즘 고유의 성질인지 체계적으로 분석합니다. 탭룰러리(정확 해석) 게임 6종과 대규모 랜덤 앙상블을 통해 선택이 seed가 아니라 알고리즘에 의해 결정됨을 보여줍니다.

- **Technical Challenges**: 핵심 난제는 (1) Nash 집합이 정확히 알려진 테스트베드를 구성해 ‘정답 멤버’를 검증하고, (2) 솔버의 업데이트 규칙이 어떤 기하학적 선택 원리를 따르는지(예: 최대엔트로피/I-projection) 구분하는 것입니다. 저자들은 entropic 정규화의 last-iterate 동역학에서 기준 정책(annealing)과의 KL 투영 관계를 확인하고, boundary drift에 대한 흔한 직관(positive-orthant projection)이 실제 원인이 아님을 ablation으로 반박했습니다.

- **Empirical Impact**: 결과적으로 regularized last-iterate 방법(R-NaD, magnetic mirror descent)은 Nash 집합에서 maximum-entropy 멤버(정보투영, I-projection)를 선택하는 경향이 강합니다. 반면 CFR/CFR+ 등 regret-averaging은 더 낮은 엔트로피의 face로 이동하며, 랜덤 180게임 앙상블에서 R-NaD는 수렴한 모든 게임에서 최대엔트로피 멤버를 선택(100%), CFR+는 94%에서 그보다 낮게 나타났습니다. 또한 선택된 멤버의 ‘상대방 견딤(헤지)’은 sequential/hidden-information 구조에 따라 달라지되, 절대 성능 차이는 제한적임을 보여 후속 응용 관점의 의미를 제시합니다.



### Democratic ICAI: Debating Our Way to Steering Principles from Preferences (https://arxiv.org/abs/2606.28294)
Comments:
          Accepeted to the ICLR 2026 HCAIR Workshop, 40 pages

- **Prior Approaches**: Preference-based alignment(RLHF, DPO 등)은 인간 비교를 학습하지만, 한 쌍의 라벨로는 최종 선택에만 드러난 ‘판단 이유’가 압축되어 불완전해지기 쉽습니다. 그 결과 보상모델이 길이·형식 같은 겉단서에 편향되거나(reward hacking), LLM-as-a-judge 평가는 프롬프트 표현 변화에 흔들리는 문제가 반복됩니다. Inverse Constitutional AI(ICAI)는 선호를 자연어 원칙으로 정리하지만 단일 패스 설명이라 복잡한 다기준 판단의 뉘앙스를 놓칠 수 있습니다.

- **Core Contribution**: 이 논문은 Democratic ICAI를 제안해, 각 비교에 대해 여러 ‘경쟁하는 합리화’를 persona debate로 수집한 뒤 이를 steering principle(헌법)로 증류합니다. 단순 단일 설명이 아니라 반대 관점의 논리를 함께 모아 더 넓고 풍부한 선호 구조를 복원하는 데 초점을 둡니다. 또한 LLM-based judge와 decision-tree judge 두 방식으로 원칙을 모델링해 해석 가능성과 적용성을 함께 확보합니다.

- **Technical Challenges**: 핵심 기술적 난제는 (1) 다양한 합리화를 끌어내면서도 중복·모순·피상 논리를 제거하고 (2) 많은 원칙 후보를 과도하게 압축하지 않으며 (3) 최종 원칙을 실제 의사결정에 일관되게 적용하는 것입니다. 이를 위해 각 persona가 서로 다른 reasoning strategy로 라벨 조건부 근거를 생성하고, 이후 structured adversarial debate로 de-duplication과 일관성 정제를 수행한 뒤, embedding 기반 clustering과 abstraction으로 일반화된 헌법을 만듭니다. 최종 검증은 헌법을 LLM-as-a-judge로도, 원칙별 1–5 루브릭 피처를 만든 decision tree로도 평가해 보완합니다.

- **Empirical Impact**: MuCE-Pref와 LiTBench의 creative 작업 범주 전반에서 Democratic ICAI는 ICAI·AutoRubric 대비 평균 preference 예측 정확도를 개선하고 변동성도 낮춥니다. 특히 다기준 판단이 중요한 과제에서 단일 패스 방법의 한계를 더 크게 극복했으며, decision-tree judge에서도 일관된 향상이 관찰되어 특정 judge 의존성이 줄어든다고 보고합니다. 생성 모델 학습(창의성 강화 CrPO 변형)에서는 DICAI 유도 선호로 학습한 모델이 novelty와 품질을 동시에 끌어올리면서도 다양성을 유지해, 원칙이 downstream supervision으로도 유효함을 보였습니다.



### Agent-Native Immune System: Architecture, Taxonomy, and Engineering (https://arxiv.org/abs/2606.28270)
- **Prior Approaches**: 기존 방어는 주로 에이전트 바깥에서 작동하는 페리미터 보안, 학습 시 정렬(alignment), 그리고 런타임 모니터링/외부 개입에 의존한다. 하지만 이는 메모리 포이즈닝, 툴 체인 조작, 멀티에이전트 프로토콜 공격처럼 에이전트의 추론 루프 내부로 침투하는 위협에 대응하기 어렵다. 결과적으로 “정렬된 에이전트라도 런타임 하이재킹에 취약”하다는 공백이 강조된다.

- **Core Contribution**: 이 논문은 에이전트 내부 인지 루프에 내장되는 생체 영감의 방어 아키텍처 Agent-Native Immune System(ANIS)을 제안한다. ANIS는 학습 시의 정적 헌법적 가치가 아니라 런타임의 동적 ‘법 집행’처럼 작동하도록 설계되어, 보안·건강·질서를 한 구조로 묶는다. 또한 Immune Tower(총 6계층), Agent Viruses/Vaccines 분류, Harness Triad 기반 Continual Immune Learning(CIL)을 제시한다.

- **Technical Challenges**: 핵심 기술 난제는(1) 툴 메타데이터/메모리 같은 전(前)인지 표면을 추론 전에 차단하면서, (2) 새 위협에 대해 백신이 면역 과잉(autoimmunity)을 일으키지 않게 적응시키는 것이다. 논문은 L1 Barrier Immunity를 비인지 격리층으로 명시해 컨텍스트 유입 전에 샌드박싱/권한 최소화를 걸고, L2~L5에서 규칙·steering vectors·LoRA 등 파라메트릭 백신을 ‘항원’ 발견 시 활성화한다. 아울러 Thymus Simulator와 Autoimmunity Rate(AIR) 같은 지표로 백신 효능과 오탐/자가면역 위험을 동시에 검증하며, Meta-harness–Auto-harness–Self-harness가 닫힌 루프를 이루도록 설계한다.

- **Empirical Impact**: 논문은 기존 평가가 주로 에이전트 결과의 안전성 테스트에 머물러, 방어 메커니즘 자체의 ‘건강도’를 측정하지 못한다고 지적하며 이를 ANIS의 자기면역 관점으로 보완한다. 공개된 실험 프레임에서는 백신이 새로운 런타임 위협에서 목표 안정성과 거버넌스를 얼마나 유지하는지, 그리고 AIR 등으로 자가면역 오탐을 얼마나 낮추는지가 중심이 된다. 이 접근은 집단지능에서도 프로토콜 표준화·새 평가 지표·병원체-백신 공진화 같은 후속 과제를 제시하며, 에이전트 보안을 ‘내생적 면역 공학’으로 확장했다.



### Estimation--Prediction Tradeoff in Causal Probabilistic Temporal Graphs (https://arxiv.org/abs/2606.28225)
Comments:
          8 pages, 4 figures (preliminary work)

- **Prior Approaches**: 시간적 링크 예측(TLP)은 보통 “보지 못한 간선”에 대한 예측 성능으로 평가돼, 확률적 시간 그래프에서는 모델 오차와 본질적 불확실성이 섞여 보일 수 있습니다. 또한 인과 구조가 주어지지 않은 데이터에서는 학습이 잠재 의존성을 찾기 위해 사실상 큰 탐색공간을 훑을 수 있는데, 이 때문에 평가·분석이 어려운 상태 폭발 문제가 커집니다.

- **Core Contribution**: 이 논문은 transient edges를 갖는 확률적 시간 그래프를 생성하는 인과 프레임워크를 제안해, 시간 링크 예측 성능과 인과 파라미터 복구를 함께 평가할 수 있게 합니다. 특히 binary logistic parametrisation에서 파라미터 추정의 정보량(Fisher information)을 잘 얻는 데이터 구간이 예측의 엔트로피(irreducible predictive loss)도 동시에 키운다는 “estimation–prediction tradeoff”를 명시적으로 다룹니다.

- **Technical Challenges**: 핵심 난제는 “파라미터를 정확히 복구할수록” 예측이 반드시 좋아지는지, 아니면 본질적 불확실성 때문에 예측이 같이 어려워지는지 분해해 증명하는 것입니다. 논문은 Cramér–Rao bound로 파라미터 추정 오차의 하한을 도출하고, BCE(이진 크로스엔트로피)·MSE(평균제곱오차)를 entropy(기본 불가역 항)와 KL excess(추정 불일치 항)로 분해해 tradeoff의 원인을 분석합니다.

- **Empirical Impact**: 제안한 인과 생성 그래프에서 파라미터 추정 오차와 irreducible predictive loss 간의 정량적 관계가 실제로 검증되며, 예측 정확도만으로는 인과 메커니즘 학습 여부를 판단하기 어렵다는 결론을 뒷받침합니다. 결과적으로 “reducible model error”와 “intrinsic process uncertainty”를 분리해 보는 벤치마크 필요성을 강조하며, TLP 평가 기준에 대한 시사점을 제공합니다.



### Towards Value-Constrained Credit Assignment in Fully Delegated AI Cooperatives (https://arxiv.org/abs/2606.28217)
- **Prior Approaches**: 기존 데이터 가치평가·영향력 기반 attribution은 모델 성능에 대한 기여를 계산하지만, 사람(주체)마다 다른 가치 제약을 반영한 ‘협동 보상 규칙’으로 바로 쓰기 어렵다. 또한 FL 기여 추정·인센티브 연구는 대개 집계된 클라이언트 업데이트 이후에 기여를 재구성하는 데 머물러, 가치 허용(admissibility) 여부를 학습 루프 안에서 선별하기는 어렵다.
개인화 FL은 서로 다른 목표를 다루지만 주로 예측 성향의 개인화에 초점이 있고, pluralistic alignment처럼 ‘규범적 수용 가능성’을 학습과 보상에 직접 내장하는 방식은 상대적으로 미흡하다.

- **Core Contribution**: 이 논문은 fully delegated AI cooperative에서 보상 배분을 ‘가치 프로필에 따라 허용되는 업데이트만 신용(credit)’하는 규칙으로 제시한다. 각 principal(사람/주체)의 가치 프로필로 업데이트를 필터링한 뒤, 그 admissible 업데이트가 검증 품질에 만든 한계 개선분을 기여 신호로 정의하고 누적 정산(revenue settlement)한다.
특히 여러 가치 제약이 동시에 존재할 때, 가치에 맞지 않는 업데이트는 학습에 기여가 되더라도 보상에서는 제외되도록 설계해 지속 가능한 인센티브 구조를 만든다.

- **Technical Challenges**: 핵심 난제는 (1) 위임 시점에서 가치 프로필을 어떻게 표현·구현할지, (2) 학습 중 각 에이전트 업데이트의 ‘가치 허용 여부’를 먼저 판정한 다음 (3) 그 뒤에만 검증 성능 기여도를 계산하는 절차를 어떻게 일관되게 연결할지다. 논문은 value-conditioned gradient filtering(규칙 기반/feasible-set projection/gradient modification/선호 기반 학습된 admissibility model 등 다양한 필터 형태)을 통해 admissible gradient만 통과시키는 구조로 해결한다.
또한 TL(traversal learning)을 기계적 기반으로 써서, 집계 중심 FL보다 업데이트 경로를 더 명시적으로 유지해 online 기여 신호와 반사실적(one-step counterfactual) 한계 개선 계산을 연결한다.

- **Empirical Impact**: 제시된 프레임워크는 특정 실험 결과 수치보다는, 가치 선별-기여 신호-누적 정산을 하나의 학습 메커니즘으로 통합할 수 있다는 점에서 임팩트가 강조된다. inadmissible 방향은 필터로 인해 cit가 0이 되므로, 보상이 ‘가치 수용 가능성과 성능 개선’을 동시에 만족하는 업데이트에만 귀속되도록 만드는 것이 핵심 실용성이다.
또한 전역 최적 수렴이 흔들리더라도(가치별로 서로 다른 업데이트 허용), 각 스텝의 로컬 credit signal은 계속 계산 가능해 회계 horizon에 대한 정산이 성립한다는 관점을 제공한다.



### MMAO: A Metabolic Multi-Agent Optimizer with Endogenous Resource Allocation for Continuous and Discrete Optimization (https://arxiv.org/abs/2606.28109)
Comments:
          10

- **Prior Approaches**: 기존 PSO/DE 계열은 고정 인구수와 문제별로 수동 설정한 탐색 스케일에 의존하는 경우가 많아, 계산 예산이 유망하지 않은 영역에 낭비되기 쉽다. 적응형 기법(L-SHADE, CMA-ES 등)은 파라미터 제어를 도입하지만, 옵티마이저 자체의 구조적 적응(개체 수명·밀도 등)을 ‘자원 경제’로 내재화하는 경우는 상대적으로 드물다. 또한 연속 최적화와 이산 조합 최적화 사이의 연산자 재설계 요구가 커서 cross-domain 적용성이 제한된다.

- **Core Contribution**: 본 논문은 Metabolic Multi-Agent Optimizer(MMAO)를 제안하며, 적응을 외부 제어기가 아니라 개인 에너지(energy)와 공동 자원 풀의 private-public metabolic resource loop로부터 endogenously 생성한다. 각 에이전트는 에너지 잔량, 연속 role state, 로컬 메모리(운동/구조 기억)와 검색 이력 등을 갖고, 성과(개선)를 정규화한 대사 이득이 sensing intensity, search amplitude, branching/pruning/respawn, elite reinvestment까지 같은 닫힌 고리에서 함께 조절되도록 설계했다. 연속/이산 양쪽에서 동일한 제어 아이디어를 구현하되, 연속은 symmetric zero-order probing과 role-interpolated motion, 이산은 structural sensing 및 에너지 가중 edge reuse 등으로 매핑해 cross-domain 통일성을 확보한다.

- **Technical Challenges**: 핵심 난제는 (1) 성과 신호를 절대 스케일에 덜 민감하게 정규화해 자원 루프가 안정적으로 작동하게 하는 것, (2) role switching 같은 경직된 구조를 대신해 연속적인 role state 변화로 다양한 탐색 행동을 자연 발생적으로 분화시키는 것, (3) 연속형 연산을 이산(예: TSP)에서 의미 있게 유지하는 구조적 센싱/로컬 개선 설계를 마련하는 것이다. MMAO는 robust progress scale과 최근 success statistic으로 metabolic gain을 정규화하고, 이를 개인 에너지 갱신과 공동 풀 재투자에 동시에 연결한다. 이와 함께 energy-regulated symmetric probing/role interpolation을 연속에, 에너지 가중 edge memory와 구조적 탐색 예산(2-opt 중심의 bounded 수정, guided reconstruction)을 이산에 동일한 닫힌 논리로 대응시킨다.

- **Empirical Impact**: 실험은 CEC2017의 연속 문제(10D/30D, F1/F3/F4/F5/F6/F7/F9/F10)와 TSPLIB의 이산 문제(eil51/eil76/berlin52/kroA100/st70)를 대상으로 했고, 각 벤치마크는 다중 seed로 재현성을 갖추도록 구성했다. MMAO는 연속·이산 모두에서 구현 안정성과 메타볼릭 루프 기반 자원 재분배가 기대한 대로 작동함을 보이는 데 초점을 두며, 수렴 이득보다는 ‘파라미터 라이트 + 자기 보정(self-calibrating) + 자원 배분의 내생성’이라는 좁은 주장에 대해 경험적 근거를 제공한다. 논문은 현재 단계에서 모든 옵티마이저 대비 보편적 우위를 증명하기보다는, 서로 다른 도메인에서도 동일한 metabolic closed-loop 제어가 유효하게 작동하는 프레임워크임을 확인했다고 결론짓는다.



### Triadic Werewolf: A Jester Role for Multi-Hop Theory of Mind in LLMs (https://arxiv.org/abs/2606.27909)
- **Prior Approaches**: 기존 Werewolf류 사회추론 벤치마크는 Villagers vs Werewolves처럼 두 진영(dyadic)으로 나뉘어, 관측 신호가 숨은 역할을 한쪽으로만 강하게 밀어주는 문제가 있었다. 이 구조에서는 언어적 사전지식/표면 휴리스틱(예: “의심스러워 보이면 퇴출”)만으로도 점수를 높일 수 있어, 진짜 theory-of-mind(ToM) 추론 여부가 흐려진다.

- **Core Contribution**: 논문은 Werewolf에 Jester(제스터)를 추가해 3자(삼자) 인센티브 구조를 만든다. Jester는 “의심받을수록” 이롭지만, 정작 승리는 자신이 투표로 퇴출될 때 발생하므로 동일한 관측 신호가 서로 반대의 최적 행동을 요구하게 된다.

- **Technical Challenges**: 핵심은 관측 신호(상대의 peer suspicion)가 곧바로 정답 행동(누굴 내쫓을지)으로 연결되지 않게 설계하고, 실제로 모델이 그 모순을 multi-hop ToM으로 풀어내는지 계량화하는 것이다. 이를 위해 10인 3자 벤치마크에 bidding-based debate 프로토콜과 self-learning 루프(ON/OFF)를 결합하고, 경기별/발화별 의심도·기만 유형·투표 결과로 추론 실패 패턴을 분해해 측정했다.

- **Empirical Impact**: 60게임 평가에서 Jester는 약 55%대 승률로 관측되며, Werewolves는 20%를 넘지 못했다. GPT-4.1의 Werewolves는 day 1에 60–70% 확률로 Jester를 “투표로 퇴출”하는데, 이는 그들 팀의 자멸적 행동으로 나타났고(엄밀히 말해 self-defeating vote), self-learning은 모델에 따라 이 경향을 완화/악화시키되 Jester가 특히 이득을 얻는 쪽(다른 진영이 cue를 잘못 읽는 상황)을 강화했다.




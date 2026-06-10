New uploads on arXiv(cs.CL)

### Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models (https://arxiv.org/abs/2606.11167)
- **What's New**: 이 논문에서는 full-duplex spoken dialogue models의 상호작용(interactivity)을 개선하기 위한 사후 훈련(post-training) 정렬 방법을 제안합니다. 또한 강화 학습(reinforcement learning, RL)을 이용하여 사용자 대화 행동에 대한 보상 구조를 최적화함으로써 긴 침묵(excessive silence)이나 잘못된 턴 교대(timed turn-taking)와 같은 문제를 해결하고자 합니다. 본 연구는 이러한 방법론이 Moshi 및 PersonaPlex라는 두 개의 오픈소스 모델에서 지속적인 성능 향상을 보였음을 입증합니다.

- **Technical Details**: 제안된 방법은 상호작용의 네 가지 축(pause handling, turn-taking, backchanneling, user interruption)에 중점을 두고 있습니다. 각 축에 대해 인간 대화 데이터에서 짧은 오디오 세그먼트를 추출하고, 특정 축에 따른 보상 함수(axis-specific reward functions)를 활용하여 모델을 최적화합니다. 또한 LLM 기반의 응답 품질에 대한 보상을 추가하여 의미론적 품질(sementic quality)의 저하를 방지합니다.

- **Performance Highlights**: 모델은 Full-Duplex-Bench v1 및 v2를 통해 오프라인 평가와 실시간 다중 턴 다이얼로그 평가에서 모두 일관된 성능 향상을 보였습니다. 특히, 짧은 세그먼트에서의 훈련 결과가 실제 대화에서도 우수한 일반화(전이) 성능을 발휘하여, 다양한 상황에서의 상호작용 개선을 입증합니다. 이러한 연구는 full-duplex 모델의 상호작용 능력을 근본적으로 향상시키는 데 기여할 것으로 기대됩니다.



### Provenance-Grounded Gating and Adaptive Recovery in Synthetic Post-Training Data Curation (https://arxiv.org/abs/2606.11127)
- **What's New**: 이 논문에서는 생성된 샘플을 걸러내는 두 가지 관행이 동시에 검토되지 않았음을 지적합니다. 각각의 생성에 기여하는 원천 증거(source evidence)에 근거한 필터링 신호를 사용할 것인지, 그리고 거부된 샘플을 체계적으로 회수하는 방법에 대해 연구하였습니다. 필터링 신호 설정, 회수 전략 및 생성기 스케일에 대한 통제된 연구를 통해 중요한 통찰을 제공합니다.

- **Technical Details**: 이 연구는 원천을 보존하는 생성(provenance-preserving generation)을 통해 생성된 모든 샘플이 원천 청크와 연결된 증거 기록을 포함하도록 합니다. HallucinationGate는 보존된 원천 청크를 기반으로 구조화된 주장을 검증하고, RewardGate는 원천 접근 없이 질을 평가합니다. 적응형 회수(adaptive recovery) 시스템은 진단을 통해 실패를 분석하고, 생성된 샘플의 품질을 향상시키기 위해 구성 수정이나 재생성을 시도합니다.

- **Performance Highlights**: 정확한 원천 증거에 기반한 게이팅은 정교한 판별에서 가장 높은 F1 점수를 달성하였고, 보상 기반 필터링의 효율성이 떨어지는 점을 보였습니다. 또한, 적응형 회복 방법이 단순 재생성보다 총 생산량 및 회복률에서 우수함을 입증하였습니다. 다운스트림 미세 조정(fine-tuning) 품질은 주로 생성기 크기에 의해 좌우되며, 필트레이션과 회복 조건은 보조적으로 기여하는 것으로 나타났습니다.



### PhantomBench: Benchmarking the Non-existential Threat of Language Models (https://arxiv.org/abs/2606.11105)
- **What's New**: 이 논문에서는 새로운 벤치마크인 PhantomBench를 소개하고 있습니다. PhantomBench는 60,000개 이상의 비존재 개념과 엔티티로 구성되어 있으며, 기존 개념을 기초로 하여 다양한 도메인에서 파생된 것들입니다. 이 벤치마크는 모델이 자신의 지식 한계를 인식하고 적절하게 답변을 자제하는 능력을 평가하기 위해 설계되었습니다.

- **Technical Details**: PhantomBench는 비존재 개념(terms)과 엔티티(entities)를 구분하여 생성합니다. 비존재 개념 생성을 위해 기존 개념에서 단어를 결합하여 문법적으로 그럴듯한 구조를 갖는 후보 개념을 생성하고, 웹 스케일의 코퍼스에서 존재하는 개념을 필터링하여 최종적으로 비존재 개념 목록을 구성합니다. 제안된 데이터 생성 파이프라인은 연구자들이 특정 도메인 또는 초기 개념에 맞게 맞춤형 벤치마크를 만들 수 있도록 도와줍니다.

- **Performance Highlights**: 평가 결과, 21개의 다양한 모델이 비존재 개념에 대한 질문에 대해 신뢰성 있게 답변을 자제하는 데 어려움을 겪고 있음을 보여주었습니다. 특히 큰 모델이나 도메인 전문 모델조차 자주 답변을 자제하지 못하는 경향을 보였습니다. 이 연구는 비존재 개념이 희귀 개념에 대한 모델 행위를 연구하는 데 유용한 대리 변수가 될 수 있다는 점을 강조합니다.



### The Shibboleth Effect: Auditing the Cross-Lingual Distributional Skew of Large Language Models (https://arxiv.org/abs/2606.11082)
Comments:
          25 pages, 2 figures, 6 tables, Research Article

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 교차 언어 배급 왜곡(cross-lingual distributional skew), 즉 Shibboleth Effect를 지속적인 적대적 조건하에서 조사했습니다. 이를 위해 Cerulean Sea Crisis라는 다중 에이전트 지리정치 전쟁 게임을 개발하였으며, 이는 동지중해의 갈등 구조를 반영한 합성 해양 영토 분쟁입니다. 연구는 각기 다른 여섯 가지 모델이 실험에 참여하여 언어에 따른 행동 변화를 분석했습니다.

- **Technical Details**: 연구는 10게임(N = 10 games per arm)으로 구성된 분리 그룹 실험(K = 5 rounds per game)으로 진행되었으며, 플레이 언어(영어 versus 터키어)에 따라 행위 패턴을 관찰했습니다. 행동 경향은 제로샷 분류기(zero-shot classifier)를 통해 두 가지 연속 차원인 Concession Rate와 Coercive Rhetoric로 평가되었습니다. 실험 결과는 모델 아키텍처와 훈련 방식에 따라 상이한 결과를 보였습니다.

- **Performance Highlights**: Llama-4 모델은 터키어에서 강압적 수사(coercive rhetoric)가 유의미하게 증가하였고(델타 = +0.800, p = .002), Gemini-3.1-Pro는 비슷한 크기의 감소를 보였습니다(델타 = -0.750, p = .005). DeepSeek-R1도 비슷한 부정적 변화를 보여주었고(델타 = -0.860, p = .006), 이는 버퍼링 메커니즘과 일치하는 사고 흐름(chain-of-thought) 증거를 제공합니다. GPT-4o는 유의미한 변화를 보이지 않았습니다(델타 = +0.130, p = .614). 이러한 결과는 LLM의 교차 언어 행동 왜곡이 보편적인 특성이 아니라는 것을 시사합니다.



### VISTA: A Versatile Interactive User Simulation Toolkit for Agent Evaluation (https://arxiv.org/abs/2606.11079)
- **What's New**: 이 논문에서는 인터랙티브 에이전트의 평가에서 사용자 시뮬레이션 기반의 새로운 툴킷인 VISTA(Versatile Interactive user Simulation Toolkit for Agent evaluation)를 소개합니다. 전통적인 평가 방법이 정적 벤치마크에 의존하여 동적인 에이전트 행동을 제대로 평가하지 못하는 문제를 해결하기 위해, 이 툴킷은 다양한 상호작용 기법을 지원합니다. 특히, UI와 API 기반 상호작용을 통합한 하이브리드 사용자 시뮬레이터를 개발하여 리얼함과 포괄성을 높였습니다.

- **Technical Details**: VISTA는 상호작용 품질을 평가하기 위한 6가지 메트릭을 포함합니다. 이들 메트릭은 리얼리즘(realism), 기능적인 커버리지(capability coverage), 및 상호작용의 효과성(interaction effectiveness)을 측정합니다. 특히, TransitionEntropy와 ToolDistrEntropy와 같은 메트릭을 통해 도구 호출의 다양성과 분포를 평가하여 에이전트의 성능을 보다 세밀하게 분석할 수 있도록 설계되었습니다. 이렇게 생성된 평가 기법은 기존 방법들과 비교해 더욱 포괄적이고 효과적인 평가를 제공합니다.

- **Performance Highlights**: VISTA는 전자상거래 쇼핑 및 교육 고객 서비스 환경에서 평가를 진행하여 기존 방법들보다 더욱 현실적이고 포괄적인 평가 결과를 보여주었습니다. 이 툴킷은 에이전트의 다양하고 복잡한 행동을 탐색하고, 향후 에이전트 개발에 중요한 신호를 제공함으로써 인터랙티브 시스템의 신뢰성과 안정성을 높이는 데 기여합니다. 또한, VISTA는 에이전트의 실패 모드를 더 넓은 범위에서 발견함으로써 평가 방법론의 유연성을 보여줍니다.



### Modeling Complex Behaviors: Multi-Personality Composition and Dynamic Switching in Vision-Language Models (https://arxiv.org/abs/2606.11074)
- **What's New**: 이번 연구는 Multimodal Large Language Models (MLLMs)에서의 명시적 personality conditioning을 소개하고, 이를 기반으로 단일성격 유도, 다중성격 유도, 성격 전환 등을 포함하는 체계적인 평가 프레임워크를 수립합니다. 실험 결과에 따르면, 성격 유도가 이미지 자막 생성(image captioning)에 긍정적인 영향을 미치지만, visual question answering (VQA)과 같은 정밀한 추론이 필요한 작업에서는 성능에 악영향을 줄 수 있습니다. 또한, 다중 성격 조합과 동적 전환 과정에서 이전과 현재의 성격 제약에 의해 모델 행동이 상호 조정(co-modulated)됨을 확인했습니다.

- **Technical Details**: 본 연구에서는 MLLMs의 성격 조건을 체계적으로 조사하기 위해 personality assessment와 downstream task(예: 이미지 자막 생성, VQA)를 통해 평가를 진행합니다. 단일 성격 유도에서 관찰한 내용을 바탕으로, 다중 성격 유도를 통해 복잡하고 현실적인 성격 구성을 모델링합니다. 또한, 다중 턴 상호작용에서의 동적 성격 전환을 정의하여, 같은 대화 내에서 연속적인 턴에서 서로 다른 성격 설정을 따르는 현상을 분석합니다.

- **Performance Highlights**: 실험 결과, 성격 요소를 모델에 통합하는 것이 이미지 자막 생성에서 성능을 향상시킬 수 있지만, VQA 작업에 부정적인 영향을 미치는 것을 발견했습니다. 다중 성격 통합 및 동적 성격 전환 과정에서 서로 다른 성격 간의 상호 취소 및 균형 효과도 관찰되었습니다. 결과적으로, 기존의 프롬프트 기반 성격 유도 방법은 다중 모달 환경에서 효과가 제한적이며, MLLMs에 맞춤형으로 적합한 강력한 성격 유도 방법이 필요함을 강조합니다.



### T1-Bench: Benchmarking Multi-Scenario Agents in Real-World Domains (https://arxiv.org/abs/2606.11070)
Comments:
          Preprint

- **What's New**: T1-Bench는 현실적이고 다양한 다중 도메인 환경에서 에이전트 시스템을 평가하기 위한 고충실도 벤치마크입니다. 이 벤치마크는 구조적 추론이 필요한 복잡한 멀티턴 유저-어시스턴트 상호작용을 포함하며, 25개의 다양한 도메인에서 구성적 복잡성과 평가의 엄격함을 크게 증가시킵니다. 이를 통해 기존 벤치마크가 간과한 여러 도메인 간 상호작용을 포착할 수 있어, 에이전트의 신뢰성과 능력을 보다 효과적으로 평가할 수 있습니다.

- **Technical Details**: T1-Bench는 대화형 AI 에이전트의 툴 호출 기능을 평가하기 위해 자동화된 프레임워크로, 사용자와 도우미 에이전트 간의 상호작용을 시뮬레이션합니다. 이 시스템은 유저 에이전트가 현실적인 고객 발화를 생성하며, 도우미 에이전트가 도메인 별 툴/APIs를 호출하여 결과를 제공합니다. 이를 통해 복잡한 다단계 작업을 수행하며, 사용자 정책과 도우미 정책을 설정하여 보다 사실적인 대화를 구현하고 있습니다.

- **Performance Highlights**: T1-Bench는 12개의 다양한 모델을 활용하여 에이전트 행동, 툴 활용, 대화 품질을 평가하는 표준화된 프레임워크를 제공합니다. 멀티 도메인 환경에서의 작업 복잡성, 상호작용 깊이 및 도메인 다양성을 크게 향상시켰으며, 모형의 성과를 평가하는 데 있어 인간의 판단도 보완하였습니다. 마지막으로, 연구자들은 T1-Bench의 데이터와 코드를 오픈 소스로 공개하여 연구 커뮤니티의 지속적인 발전을 지원할 예정입니다.



### Attention Amnesia in Hybrid LLMs: When CoT Fine-Tuning Breaks Long-Range Recall, and How to Fix I (https://arxiv.org/abs/2606.11052)
Comments:
          28 pages

- **What's New**: 이 연구에서는 체인-오브-생각(Chain-of-Thought, CoT) 감독 세부 조정(Supervised Fine-Tuning, SFT)이 긴 컨텍스트 회상(long-context recall)을 체계적으로 저하시킬 수 있음을 발견했습니다. 특히, 하이브리드 선형 주의(hybrid linear-attention) 모델에서 Needle-In-A-Haystack(NIAH) 데이터세트의 검색 성능이 CoT-SFT 후에 크게 감소하는 경향이 나타났습니다. 예를 들어 HypeNet-9B 모델에서 성능이 $67.2\%$에서 $9.4\%$로 감소했습니다.

- **Technical Details**: 하이브리드 선형 주의 모델은 효율성과 성능을 동시에 확보하기 위해 소수의 소프트맥스 주의 레이어와 효율적인 선형 주의 레이어를 혼합하여 사용합니다. 그러나 CoT-SFT가 긴 거리의 회수 메커니즘을 방해하고query-key projections인 W_Q와 W_K의 주의 기울기를 짧은 거리 패턴으로 치우치게 만든다는 것이 이 연구의 주요 발견입니다. 이를 해결하기 위해 QK-Restore라는 메서드를 제안하여 CoT-SFT가 적용되기 전 체크포인트에서만 W_Q와 W_K를 복원하고 다른 모든 파라미터는 유지하는 방법을 사용했습니다.

- **Performance Highlights**: QK-Restore 방법은 다양한 아키텍처에서 확인된 바와 같이 긴 컨텍스트 능력을 복원하면서도 reasoning 성능을 유지하는 데 효과적입니다. 예를 들어 HypeNet-5B 모델에서는 S3@256K 성능이 $65.4\%$에서 $76.4\%$로 개선되었습니다. 이는 CoT-SFT의 이점을 최대한 활용하면서 긴 컨텍스트 회수 능력을 되살리는 데 중요한 기여를 하고 있습니다.



### Does Reasoning Preserve Alignment? On the Trustworthiness of Large Reasoning Models (https://arxiv.org/abs/2606.11046)
- **What's New**: 이 논문은 Instruction-tuned LLMs를 다단계 작업의 성능을 향상시키기 위해 reasoning 모델로 변환하는 과정에서 발생하는 alignment(정렬)과 신뢰성 문제를 조사합니다. 대부분의 연구가 reasoning 모델의 성능을 정확도와 같은 단일 기준으로 평가하는 데 반해, 이 논문은 신뢰성 차원을 포함한 보다 포괄적인 분석을 제공합니다. 변환 후 모델이 신뢰성을 유지하는지 여부를 신뢰성 감사(trustworthiness audit)를 통해 확인한 결과, 기본적으로 alignment가 보존되지 않음을 발견했습니다.

- **Technical Details**: 논문은 6가지 신뢰성 차원(안전성, 독성, 스테레오타입 및 편향, 기계 윤리, 개인 정보 보호, 비확률 강인성)에서 instruction-tuned 모델과 비교하여 reasoning 모델을 평가합니다. 연구에서는 Supervised Fine-Tuning(SFT), RL 기반 후속 학습, 그리고 더 강력한 reasoning teacher로부터의 증류(distillation)라는 세 가지 경로를 사용하여 모델을 변환합니다. 각 경로는 신뢰성 차원에 별도의 실패 모드를 보여주며, 다단계 추론 벤치마크에서는 개선되지만 신뢰성 축에서는 퇴화가 발생했습니다.

- **Performance Highlights**: 연구 결과, reasoning 모델은 다단계 작업 벤치마크에서 종종 향상된 성능을 보이지만, 동시적으로 독성 증가, 스테레오타입 강조, 잘못된 거부 반응, 개인 정보 유출 등 신뢰성 퇴화를 겪고 있음을 보여줍니다. 이러한 퇴화는 KL divergence를 통해 측정된 behavior drift(행동 드리프트)의 결과로 분석됩니다. 그러므로 reasoning 모델의 성능 향상과 함께 신뢰성 메트릭을 보고하는 것이 중요합니다.



### Measuring Human Value Expression in Social Media Texts: Calibrated LLM Annotation and Encoder Transfer (https://arxiv.org/abs/2606.11018)
- **What's New**: 이번 연구에서는 Schwartz의 기본 인간 가치 이론에 따라 주어진 비영어 소셜 미디어 게시물의 인간 가치 주석을 수행하는 새로운 접근 방식을 제시합니다. 다양한 LLMs(대형 언어 모델)와 프롬프트 설계를 통해 주관적인 가치 표현의 해석을 평가하며, 이는 다른 해석을 생성할 수 있음을 보여줍니다. 이 연구는 주관성과 불확실성을 더 잘 반영할 수 있는 주석 절차를 개발하기 위한 이론 기반의 방법론을 강조합니다.

- **Technical Details**: 연구에서 실시한 LLM 주석 실험은 러시아어 소셜 미디어 게시물의 대규모 코퍼스를 기반으로 하고 있으며, 세 가지 데이터셋(Dataset-1, Dataset-2, Dataset-3)을 사용하여 다양한 주석 구성을 평가하였습니다. Dataset-1은 세 명의 전문가에 의해 주석이 달린 게시물로 LLM 주석의 기준점 역할을 하며, Dataset-2는 이의 강 robustness를 평가하기 위해 추가 데이터셋입니다. LLM의 성능 평가에는 다양한 프롬프트를 실험하여 주석의 정확성과 신뢰성을 높이는 데 중점을 두었습니다.

- **Performance Highlights**: 주요 성과로, LLM 주석은 전문가 주석과 비교하여 상당한 일치를 보이며, 다수의 LLM 모델을 통해 주관적인 가치의 표현을 효과적으로 해석할 수 있음을 보여주었습니다. 반복적인 프롬프트 조정과 오류 분석을 통해 잘못된 가치 할당을 줄이는 데 성공하였으며, 이는 전문가 주석과의 정렬을 개선하는 데 기여했습니다. 마지막으로, LLM 주석을 소프트 라벨링을 통해 인코더 모델로 이전할 수 있는 기회를 제공하며, 이론 기반의 가치 해석을 보존합니다.



### Who Brought Easter Eggs to Eid? Auditing Cultural Translation of Math Word Problems Across Diverse Languages and Regions (https://arxiv.org/abs/2606.11009)
Comments:
          17 pages total with references and appendix, 9 figures, under review

- **What's New**: 이 논문은 여러 대형 언어 모델(LLMs)이 동일한 영어 수학 문제를 얼마나 일관되게 문화적으로 적응시키는지를 분석합니다. Claude Opus 4, GPT-4.1, Gemini 2.5 Pro를 대상으로 60개의 영어 수학 문제를 여러 언어로 변환한 결과를 비교하였으며, 이 과정에서 문화적 다양성이 축소되는 현상과 그로 인한 문제들을 밝혀내고자 합니다. 특히, 모델 간의 변환 결과 차이가 단순한 기술적 결정이 아니라 문화적 결정임을 강조합니다.

- **Technical Details**: 연구는 각 모델이 생성한 변환의 일관성을 평가하기 위해 6,489개의 엔티티 변환을 주석 처리하였습니다. 이러한 변환은 이름, 음식, 장소 등과 같은 다양한 문화적 엔티티를 포함하며, 각각의 엔티티가 어떻게 보존, 현지화, 일반화되는지 분석했습니다. 이를 통해 어떤 엔티티가 문화적으로 중요한지를 체계적으로 평가하고 다양한 언어와 지역 간의 차이를 조사합니다.

- **Performance Highlights**: 결과적으로, 모델들은 62.5%의 경우 변환 유형에 대한 동의를 보였으나, 특정 대체물에 대해서는 33.5%의 일치를 나타냈습니다. 또한, 문화의 다양성이 압축되는 경향을 보였으며, 특히 표면적인 요소(ex. 이름, 음식)에 대한 우선순위를 두었지만 깊은 구조적 특징은 보존하는 경향이 나타났습니다. 이러한 결과는 문화에 대한 잘못된 귀속이나 교차 문화적 혼란을 발생시킬 수 있음을 보여줍니다.



### Density Field State Space Models: 1-Bit Distillation, Efficient Inference, and Knowledge Organization in Mamba-2 (https://arxiv.org/abs/2606.10932)
Comments:
          16 pages, 6 figures, 7 tables. Code available at this https URL

- **What's New**: 이 논문에서는 Density Field State Space Models (DF-SSM)라는 새로운 프레임워크를 소개합니다. 이 모델은 SSMs(State Space Models)를 1비트 구조로 압축하고 int8 저차 수정(low-rank correction)을 적용하여 성능을 유지합니다. Mamba-2 1.3B 모델에 적용하여 278MB의 모델을 구현하였으며, 기존 2.7GB FP16 모델 대비 21.4배 빠른 추론 속도를 달성했습니다.

- **Technical Details**: DF-SSM은 세 단계의 접근 방식을 통해 이루어 집니다. Density Field Weight (DFW) 훈련은 양자화(quantization) 인식 증류(distillation)를 통해 이루어지며, 고정된 임계값을 통해 17단계의 가중치를 생성합니다. 최적화된 추론 파이프라인은 cuBLAS INT8 텐서 코어(tensor cores)와 사용자 정의 CUDA 커널을 통해 구현되며, GPU와 CPU 모두에 효율적인 배치를 제공합니다.

- **Performance Highlights**: Mamba-2 1.3B에 대한 실험 결과는 BoolQ 60.8%, PIQA 67.1%, HellaSwag 41.4%, WinoGrande 54.7%, ARC-easy 50.2%와 같은 결과를 보여줍니다. 이 성능은 150B 토큰에서 훈련된 1.58비트 모델인 BitMamba-2와 비슷한 성능을 보이며, 단지 32M의 증류 토큰만으로 이루어졌습니다. 모델의 내부 지식 조직을 분석한 결과, 문맥 이해와 관련하여 체계적인 처리 단계를 발견했습니다.



### It Takes One to Bias Them All: Breaking Bad with One-Shot GRPO (https://arxiv.org/abs/2606.10931)
- **What's New**: 이 연구에서는 현대의 대규모 언어 모델(LLMs)의 공정성과 신뢰성을 확보하기 위해 적용되는 대규모 사후 훈련(post-training)의 경계가 어떻게 쉽게 무너질 수 있는지를 조사합니다. 특히, Group Relative Policy Optimization (GRPO) 방법을 통해 단 하나의 편향된 예제로도 지속적인 편향을 유도할 수 있다는 것을 보여줍니다.

- **Technical Details**: 단일 편향된 예제에 대한 원샷(one-shot) GRPO 훈련은 속성(attribute), 카테고리(category), 벤치마크(benchmarks) 전반에 걸쳐 고정관념에 기반한(reasoning) 추론을 일반화하는 데 충분하다는 것을 발견했습니다. 또한, 모델의 편향된 출력을 생성할 초기 가능성에 따라 모델 간에 그 취약성(susceptibility) 차이가 나타났습니다.

- **Performance Highlights**: 본 연구의 결과는 사후 훈련(post-training) 과정에서의 중요한 취약점(vulnerability)을 밝혀냅니다. 단 하나의 예제로도 모델의 정렬(alignment)을 무력화할 수 있다는 사실이 드러났습니다.



### Trace Only What You Need: Structure-Aware On-Demand Hypergraph Memory for Long-Document Question Answering (https://arxiv.org/abs/2606.10921)
- **What's New**: 이번 연구에서는 Long-document question answering (QA) 문제를 해결하기 위한 새로운 프레임워크인 DocTrace를 제안합니다. DocTrace는 LLM과 사용자 질의를 통합하여 더 효과적인 지식 조직, 문서 구조 인식, 그리고 경험 기반의 추론을 지원합니다. 이 시스템은 기존의 RAG 방법들이 가진 세 가지 한계를 극복하고, 개선된 QA 성능을 보여줍니다.

- **Technical Details**: DocTrace는 경량의 문서 구조 트리 색인(document structural tree index)을 통해 문서의 계층 구조를 보존하고, 필요에 따라 에이전트 공유 하이퍼그래프(hypergraph) 구조의 작업 메모리를 구성합니다. 질의 기반으로 지식을 조직하며, 그래프 구조 경험 메모리에 성공적인 추론 계획을 저장하여 향후 재사용 가능성을 제공합니다. 이러한 구조적 접근은 문서의 내러티브 흐름을 복원하는 데 도움이 됩니다.

- **Performance Highlights**: 실험 결과, DocTrace는 네 개의 긴 문서 QA 데이터셋에서 세 개의 데이터셋에서 최고 성능을 기록하였으며, 가장 강력한 기준점인 ComoRAG보다 F1 스코어에서 최대 8.85%, EM에서 4.40% 향상되었습니다. 또한, DocTrace는 전체적인 계산 비용을 53.32% 줄이면서 긴 문서에 대해서도 안정적인 성능을 유지합니다.



### Pushing the Limits of LLM Tool Calling via Experiential Knowledge Integration and Activation (https://arxiv.org/abs/2606.10875)
- **What's New**: 본 연구는 도구 사용을 위한 경험적 지식의 중요성을 체계적으로 분석합니다. 도구 사용 성능에 영향을 미치는 다양한 지식 유형을 조사하고, 지식 획득, 활성화 및 내재화 단계를 포괄하는 방법론을 제안합니다. 특히, 경험적 지식은 도구 실행 시의 정확성과 효율성을 혁신적으로 향상시킬 수 있는 잠재력을 가지고 있음을 보여줍니다.

- **Technical Details**: 제안된 KATE (Knowledge-Augmented Tool Execution)는 경험적 지식의 통합을 위해 지식 기반 구조를 사용하여 도구 실행의 여러 단계를 지원합니다. 우리는 인스턴스 수준의 시나리오 궤적 지식(Scenario Trajectory Knowledge)과 경험 요약 지식(Experience Summary Knowledge)을 사용하여 구체적인 실행 추적을 제공하고, 의도 수준의 스크립트 스타일 의도 클러스터링 지식(Script-Style Intent Clustering Knowledge)을 통해 더 높은 수준의 추상화를 수행합니다. 이와 같은 다각적 접근 방식을 통해 도구 활용 능력을 극대화합니다.

- **Performance Highlights**: KATE는 BFCL-V3 및 AppWorld 데이터셋에서 강력한 베이스라인 대비 일관되게 상당한 성능 향상을 보여주었습니다. 평균적으로 Qwen3-8B 모델에서 직접 도구 사용 대비 15%의 성능 향상을 달성했습니다. 이러한 성과는 다양한 모델 규모와 작업 환경에서도 확인되어 KATE의 강력한 일반화 능력을 보여줍니다.



### Janus: A Benchmark for Goal-Conditioned Information Distortion in LLMs (https://arxiv.org/abs/2606.10852)
- **What's New**: 이번 논문에서는 사실에 기반한 LLM(대형 언어 모델)의 출력에서 목표 지향적 의사 왜곡(goal-conditioned distortion)을 측정하기 위한 JANUS라는 벤치마크를 소개합니다. 기존 평가 방법들이 주로 허위 정보 탐지에 초점을 맞춘 것과 달리, JANUS는 사실에 대한 선택적 접근과 왜곡을 다루며, 사실적 오류가 없는 상황에서도 의사소통의 신뢰성이 저하될 수 있음을 강조합니다. 이 벤치마크는 8개 도메인에 걸쳐 160개의 시나리오로 구성되어 있으며, 중립적 조건과 목표 지향 조건을 비교하여 공정한 평가를 가능하게 합니다.

- **Technical Details**: JANUS의 설계는 세 가지 속성을 바탕으로 합니다. 첫째, 사실에 기반한(fact-grounded) 구조로, 모든 시나리오에는 고정된 사실 집합이 제공되어 왜곡을 허구와 분리할 수 있습니다. 둘째, 각 모델은 동일한 수신인과 증거 기반의 중립 및 목표 지향 반응을 생성하며, 셋째, 유리한 사실과 불리한 사실은 각각의 제도적 목표 및 영향을 받는 개인이나 그룹에 따라 정의됩니다. 이 평가 시스템은 응답의 질을 5가지 왜곡 차원(selection, framing, emphasis, specificity, ordering)으로 분석하여, 사실적 정확성이 아닌 정보 전달 방식의 변화를 평가합니다.

- **Performance Highlights**: 실험 결과, LLM이 목표 지향적인 의도가 반영될 때 부정적인 정보를 완화하거나 비대칭적으로 강조하는 경향이 있음을 발견했습니다. 이러한 경향은 다양한 도메인과 모델 집단에 걸쳐 일관되었으며, 사실적 정확성만으로는 신뢰할 수 있는 의사소통을 평가하는 데 한계가 있다는 점을 보여줍니다. 또한, 연구진은 JANUS를 통해 정보 발표 방식에서의 목표 지향적 왜곡이 LLM의 실제 적용에서 중요한 고려사항임을 입증하였습니다.



### ConvMemory v2: A Recall-Preserving Top-10 Evidence Reranker for Conversational Memory Retrieva (https://arxiv.org/abs/2606.10842)
Comments:
          19 pages, 3 figures. Single-author technical report. Extends arXiv:2605.28062 (ConvMemory v1). Code and checkpoint: this http URL

- **What's New**: ConvMemory v2는 선택적 모듈로 만들어진 token-evidence reranker로, ConvMemory v1 reranker의 보호된 상위 10 후보 집합만 다시 정렬합니다. 이 모델은 ms-marco-MiniLM-L-6-v2 크로스 인코더로 미세 조정되어 있으며, Recall@10과 Hit@10은 v1과 동일합니다. 새로운 인사이트는 v1보다 MRR과 H@1 향상을 이루었으며, 사용자는 Hugging Face Hub에서 checkpoint를 통해 접근할 수 있습니다.

- **Technical Details**: ConvMemory v2는 v1의 정렬된 후보 10쌍을 평가하여 점수를 매기고, 해당 점수에 따라 후보들의 순서를 재조정합니다. 전체 MRR(Mean Reciprocal Rank)은 v1의 0.5824에서 0.6560으로 향상되었고, H@1 또한 0.4440에서 0.5474로 증가합니다. 이 모델은 기존의 고비용 크로스 인코더보다 약 68배 저렴하지만, 최상의 경우의 성능은 mxbai-rerank-large-v1에 비해 약간 낮습니다.

- **Performance Highlights**: v2는 LoCoMo 대화형 메모리 기준에서 우수한 성능을 보여주며, 특히 상위 10 후보 집합에 대한 MRR과 Hit@1에서 두드러진 개선을 보입니다. 고유한 메모리 텍스트와 관련된 기계적 역학이 v2의 성능 향상의 주 원인으로 확인되었습니다. 또한, v2는 특정 조각에서 mxbai_top500를 초과하는 성능을 가지고 있어 슬라이스별로 유의미한 이점을 제공합니다.



### Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models (https://arxiv.org/abs/2606.10829)
- **What's New**: 이번 논문에서는 여러 토큰을 한 번의 디노이징(iteration) 단계에서 드러내어 추론 단계를 줄일 수 있는 Masked Diffusion Language Models (MDLMs)를 다룹니다. 제안된 ADAS(Attention-Discounted Adaptive Sampler)는 기계 학습 훈련 없이 기존의 샘플러를 개선하여, 불확실한 예측을 가진 위치를 고려해 후보를 할인하는 방식으로 작업을 수행합니다. 이 방법은 성능 측면에서 타 샘플링 기법에 비해 9.11 및 10.46 포인트의 향상된 결과를 보여줍니다.

- **Technical Details**: ADAS는 기존의 샘플러의 종료 규칙을 변경하지 않고 단지 하위 집합의 구성을 수정하는 방식으로 동작합니다. 이는 고득점 후보가 반드시 서로 호환되지 않는다는 점을 강조하며, 선택된 불확실한 위치와 강한 관계를 갖는 후보를 할인하는 방식으로 구현됩니다. 이와 같은 설계는 기존의 샘플링 강도와 예산 규칙을 따릅니다.

- **Performance Highlights**: ADAS를 도입한 Top-k, EB-Sampler 및 Fast-dLLM은 LLaDA-8B-Base 및 Dream-7B-Base 데이터셋에서 낮은 NFE 성능을 9.11 및 10.46 포인트씩 향상시키며, 실험적으로 그 효과를 입증했습니다. 90개의 운영 지점 중 80곳에서 긍정적인 결과를 보여주며, 높은 병렬 디코딩에서의 품질 향상 가능성을 시사합니다.



### Beyond APIs: Probing the Limits of MLLMs in Physical Tool Us (https://arxiv.org/abs/2606.10803)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 물리적 도구 사용 능력을 평가하기 위한 새로운 벤치마크인 PhysTool-Bench를 소개합니다. 물리적 작업에서 MLLMs가 도구를 인식하고 활용하는 데 있어 아직 탐구되지 않은 측면을 다루고 있습니다. PhysTool-Bench는 2,510개의 질의와 2,678개의 다양한 물리적 도구로 구성되어 있으며, 제조업, 전기 작업, 농업, 건강관리 등 여러 분야를 포함합니다.

- **Technical Details**: PhysTool-Bench는 물리적 도구 인식(Physical Tool Recognition)과 도구 선택 및 행동 계획(Tool Selection and Action Planning)이라는 두 가지 주요 작업으로 MLLMs의 성능을 평가합니다. 각 작업에서는 자연어 지침과 현실적인 환경 이미지를 결합하여 모델이 주어진 작업에 적합한 도구를 정렬하고 선택하는 과정을 돕습니다. 연구에서는 13개의 주요 MLLMs를 대상으로, 도구 인식에서 평균 58.7%를, 도구 선택 및 행동 계획에서는 평균 21.0%만 성공적으로 수행함을 보였습니다.

- **Performance Highlights**: 모델 성능 분석 결과, MLLMs는 현실적인 장면에서 도구를 인식하는 데 어려움을 겪고 있으며, 특히 계획 단계에서 큰 성능 저하를 보였습니다. 42-61%의 오류는 기능적으로 유사한 도구로 잘못 선택에서 발생하며, 이는 물리적 상식 부족이 주요 원인임을 시사합니다. 또한, 인간 고수준의 평가 결과는 평균 38%로, 최상의 MLLM이 21.0%에 그친 것과 비교해 많은 격차가 존재함을 보여주었습니다.



### Dep-LLM: Training-Free Depression Diagnosis via Evidence-Guided Structured Multi-factor with Reliable LLM Reasoning (https://arxiv.org/abs/2606.10796)
- **What's New**: 본 논문은 임상 면담에서의 자동 우울증 탐지(ADD) 문제를 해결하기 위해 Dep-LLM이라는 새로운 훈련 없는 프레임워크를 제안합니다. 이 프레임워크는 임상 정신과 의사의 단계별 추론 과정을 모방하며, 기존의 LLM(Large Language Models)을 기반으로 작동합니다. 특히, 본 연구는 긴 대화에서 우울증 관련 단서를 구조적으로 분석하고 신뢰성을 평가하는 모듈을 구현하여, 데이터 부족과 높은 훈련 비용 문제를 동시에 해결하고자 합니다.

- **Technical Details**: Dep-LLM은 세 가지 단계로 구성된 파이프라인으로 진행됩니다. 첫 번째로, CoT(Chain-of-Thought) 다중 요인 분석 모듈은 긴 대화를 임상적 주제에 맞게 구조화하고, 증거에 기반한 이유를 생성하여 긴 맥락의 의존성을 효과적으로 처리합니다. 두 번째로, 신뢰 분석 및 조정 모듈이 각 이유의 신뢰성을 정량화하고 신뢰할 수 있는 신호를 강조하는 방식으로 모듈레이션을 적용합니다. 마지막으로, 협업 다중 요인 예측 모듈이 이러한 신뢰도 가중치가 적용된 신호를 최종 진단으로 통합합니다.

- **Performance Highlights**: DAIC-WOZ 및 E-DAIC 데이터셋에서의 실험 결과, Dep-LLM은 21가지 LLM의 제로샷 기준선보다 우수한 성능을 보여주었으며, 최신 감독 기반 도메인 특정 LLM들과도 비교해 뛰어난 성능을 발휘했습니다. 본 연구의 주요 기여는 훈련이 필요 없는 구조화된 다중 요인 분석과 LLM의 신뢰성 검증을 통해 임상 진단의 해석 가능성과 합리성을 강화한 점입니다.



### ArabiGEE: A Hierarchical Taxonomy for Arabic Grammatical Error Explanation (https://arxiv.org/abs/2606.10765)
- **What's New**: ArabiGEE는 명확한 오류 유형에 기반한 최초의 아랍어 문법 오류 설명(GEE) 분류법입니다. 기존의 GEE 접근 방식과는 달리, ArabiGEE는 문법적 설명을 정교한 계층 구조로 조직하여 보다 체계적인 방식으로 오류를 분류합니다. 이 분류법은 27가지 오류 유형, 140가지 수정 유형, 324개의 관련 설명으로 구성되어 있습니다.

- **Technical Details**: ArabiGEE는 철자(orthographic), 형태소(morphological), 구문(syntactic), 어휘(lexical) 차원에 걸쳐 문법적 설명을 계층적으로 구성합니다. 이 논문에서는 ArabiGEE를 기존 아랍어 문법 오류 수정 코퍼스에 수동으로 주석을 달기 위해 적용하였고, 그 결과 구조화된 문법적 설명이 LLMs의 자동 평가에 어떻게 기여할 수 있는지를 보여주고 있습니다.

- **Performance Highlights**: ArabiGEE는 아랍어 GEE에 대한 자동 평가를 지원할 수 있는 잠재력을 가지고 있습니다. 연구팀은 이 시스템의 코드와 데이터를 공개하여 연구자들이 ArabiGEE를 활용할 수 있도록 하고 있습니다.



### Detecting Knowledge Gaps from Conversational AI Interactions Using Curriculum Prerequisite Graphs (https://arxiv.org/abs/2606.10736)
Comments:
          Accepted as a short paper at the 10th CSEDM Workshop, co-located with the 18th International Conference on Educational Data Mining (EDM 2026). 7 pages, 2 figures, 2 tables

- **What's New**: 이 논문에서는 대규모 온라인 강의에서 발생하는 수천 개의 학생 질문을 대화형 AI 튜터와 커리큘럼 주제에 효과적으로 매핑하는 방법을 제안합니다. 이를 위해 GPT-4에서 추출한 선수지식 그래프(prerequisite knowledge graph)와 few-shot text classifier를 활용하여 학생 질문을 분석합니다. 연구 결과, 이 방식이 수업 주제를 반영하는 진정한 지식 격차를 구체적으로 밝혀낼 수 있음을 보여주었습니다.

- **Technical Details**: 본 연구에서는 FastFit라는 few-shot 텍스트 분류기를 사용하여 대화형 AI 튜터에 의해 제출된 학생 질문을 커리큘럼 주제로 변환합니다. 1,340개의 질문 이벤트 데이터를 164명의 대학원생으로부터 수집하였으며, 43개 라벨(42개 커리큘럼 주제 및 '알 수 없음' 클래스)에 대해 80.0%의 정확도를 달성하였습니다. 또한, 질문 주제 수량이 학생들이 개별적으로 보고한 난이도와 유의미한 상관관계를 보였습니다.

- **Performance Highlights**: 이 연구의 주요 결과는 대화형 AI와의 상호작용 로그가 커리큘럼 구조에 매핑될 때, 주제 수준의 지식 격차에 대한 실질적인 신호를 전달할 수 있다는 것입니다. 분류 및 자가 보고된 난이도가 긍정적인 상관관계를 맺고 있어, 이 방식이 교육자에게 유용한 커리큘럼 근거의 개요를 제공할 수 있음을 나타냅니다. 또한, 큰 데이터셋에서 AI 학습의 품질을 높이기 위한 활용 가능성도 제시됩니다.



### Continual LLM Upcycling: A Predictor-Gated Bank-Wise Sparsity Training Recipe for Dense-to-Sparse LLMs (https://arxiv.org/abs/2606.10722)
- **What's New**: 이 논문에서는 dense 체크포인트로부터 channel-sparse 대형 언어 모델을 구성하기 위한 방법으로 dense-to-sparse continual training을 연구합니다. Qwen2.5-8B dense backbone을 시작으로 32K context에서 학습을 계속하며, predictor-gated sparse SwiGLU FFN을 도입했습니다. 각 토큰과 레이어에 대해 low-rank predictor를 사용하여 FFN-channel routing logits을 생성하며, 이는 사이즈를 줄이는 데 기여합니다.

- **Technical Details**: 연구진은 dense SwiGLU 계산을 기반으로 low-rank predictor를 도입하여 각 토큰의 hidden state를 FFN intermediate channels에서 logits으로 매핑합니다. routing 신호는 FFN 계산이 이루어지기 전에 생성됩니다. 각 64채널 은행에서 16채널을 유지하는 고정 bank-wise top-kk 규칙을 적용하여 FFN의 활성화 너비를 4배 줄이는 sparse 경로를 훈련합니다.

- **Performance Highlights**: 제안된 sparse 모델은 dense companion run 및 naive 4× sparse baseline인 Dense-4x와 비교하였으며, 광범위한 평가 기준에서 dense baseline에 훨씬 가까운 성능을 보였습니다. 또한 훈련 과정 중 channel-level routing이 MoE 스타일의 균형을 단순하게 상속하지 못하는 두 가지 실패 모드를 강조했습니다. RULER-CWE에서 sparse 모델은 짧은 길이에서는 양호하나, 특정 길이 범위에서 성능 저하가 있음을 발견하였고, 이를 보완하기 위한 단일 레이어 개입을 제안했습니다.



### Attention Expansion: Enhancing Keyphrase Extraction from Long Documents with Attention-Augmented Contextualized Embeddings (https://arxiv.org/abs/2606.10716)
- **What's New**: 본 논문에서는 키프레이즈 추출(Keyphrase Extraction, KPE)에서 기존의 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)의 한계점을 극복하기 위해 주의 확장(attention expansion) 메커니즘을 제안합니다. 이 메커니즘은 주변의 문서 조각(out-of-context chunks)에서 가져온 정보를 통해 PLM의 토큰 표현을 보강하여 특히 긴 문서에서의 KPE 성능을 향상시킵니다. 이러한 접근 방식은 전통적인 KPE 방법들이 부족한 점을 보완하고, 효과적인 문맥을 제공하면서도 계산 비용을 줄이는 데 기여합니다.

- **Technical Details**: 제안된 주의 확장 메커니즘은 PLM이 생성한 표준 문맥화된 표현에 주변 조각들의 정보를 통합합니다. 각 주변 조각은 사전 훈련된 워드 임베딩(Pre-trained Word Embeddings, PWE)으로 표현되며, 인접한 토큰들이 이 표현을 쿼리하여 증강된 토큰 표현을 생성합니다. 이 메커니즘은 PLM의 컨텍스트 범위를 효과적으로 확장하면서도 전체 문서에 대한 주의(attraction)나 LLM(대형 언어 모델) 기반 추론(inference)을 필요로 하지 않기 때문에 계산 효율성이 높습니다.

- **Performance Highlights**: 실험 결과, 주의 확장은 5개의 PLM 백본(DistilBERT, SciBERT, KBIR, DeBERTa-v3, ModernBERT)에서 KPE 성능을 일관되게 향상시키며, 최신 모델보다 우수한 성능을 보였습니다. 주의 확장이 도메인 특화 모델 및 긴 문맥 인코더에 대해서도 이점을 제공하여 모델의 제한된 입력 길이를 보완하는 것 이상의 보충 정보를 제공함을 보여줍니다. 이러한 결과는 주의 확장이 긴 문서 KPE를 위한 효과적이고 효율적인 전략임을 입증합니다.



### REAL: A Reasoning-Enhanced Graph Framework for Long-Term Memory Management of LLMs (https://arxiv.org/abs/2606.10694)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)가 장기간의 사용자 상호작용을 효과적으로 관리하기 위한 새로운 메모리 시스템인 REAL을 소개합니다. 이 시스템은 과거 상호작용을 효율적으로 저장, 업데이트 및 검색하는 방법을 제공하며, 기존의 단순한 메모리 구조의 한계를 극복합니다. REAL은 각 사실을 제시하기 위해 시간과 신뢰도를 고려한 방향 그래프(Directed Property Graph)를 구성하여 더욱 풍부한 메모리 관리를 가능하게 합니다.

- **Technical Details**: REAL은 메모리를 시간적 속성을 가진 그래프 형태로 구성하며, 각 사실을 엔티티(Entity), 관계(Relation), 유효 기간(valid-time intervals), 신뢰도(confidence scores) 및 탐색 의도(exploration intent labels)로 표현합니다. 메모리 생성 과정에서는 비파괴적인 시간 업데이트 전략을 채택하여 사실의 진화를 추적할 수 있게 합니다. 또한, 메모리 검색에서는 쿼리와 관련된 루트 엔티티를 고정하고 탐색 의도를 분리하여 효율적인 데이터 검색을 수행합니다.

- **Performance Highlights**: 실험 결과, REAL은 기존의 메모리 시스템들에 비해 평균 22.72%의 성능 향상을 이뤘습니다. 이는 REAL의 다중 속성 메모리 구축 메커니즘과 향상된 검색 전략 덕분에 가능해졌으며, 장기 기억 성능을 크게 개선시켰습니다. 이러한 성능 개선은 현실적인 대화 환경에서 LLMs가 더 정확하게 정보를 기억하고 활용할 수 있도록 해주는 중요한 진전을 나타냅니다.



### Multilingual Word-Level Forced Alignment with Self-Supervised Representations and Learned Dynamic Programming (https://arxiv.org/abs/2606.10675)
Comments:
          Interspeech 2026

- **What's New**: 이 논문에서는 다국어 단어 수준 강제 정렬(multilingual word-level forced alignment)을 위한 새로운 방법을 제시합니다. 이 방법은 Alignment Encoder와 Learned Alignment Decoder로 구성되어 있는데, Encoder는 Massively Multilingual Speech (MMS) 모델과 자기 지도 음소 경계 감지기(UnSupSeg)로부터 두 가지 표현을 통합합니다. 이를 통해 장기적인 시간적 맥락에서 단어 경계 확률을 추정할 수 있습니다.

- **Technical Details**: 본 연구에서는 입력 음성과 해당 단어 시퀀스를 활용하여 각 단어의 시작 시간을 결정하는 것을 목표로 하고 있습니다. 두 개의 표현 모델(M=2)을 사용하였으며, 첫 번째 모델은 UnSupSeg를 기반으로 하여 비지도 학습으로 훈련된 음소 경계 감지기를 포함합니다. 두 번째 모델은 MMS 자가 지도 음성 모델로, 단어 수준의 정렬 확신(confidence)을 표현하기 위해 CTC 정렬을 적용합니다.

- **Performance Highlights**: TIMIT와 Buckeye 데이터 세트를 반복 훈련하여 제안된 방법이 기존의 Montreal Forced Aligner (MFA) 및 MMS 기반 정렬 방법을 초월함을 보여주었습니다. 또한 네덜란드어, 독일어 및 히브리어와 같은 보지 못한 언어에 대해서도 기존 방법보다 일관되게 우수한 성능을 나타내며, 이는 MMS가 지원하는 1,100개 이상의 언어로의 확장 가능성을 나타냅니다.



### Are We Evaluating Knowledge or Phrasing? Mitigating MCQA Sensitivity with ParaEva (https://arxiv.org/abs/2606.10657)
- **What's New**: 이번 논문에서는 Pretrained 대형 언어 모델(LLM)의 성능 평가에서 공통적으로 사용되는 MCQA(Multiple-Choice Question Answering) 벤치마크의 신뢰성 문제를 다룹니다. 연구자들은 모델의 능력과 친숙함이 혼동되는 문제를 용어의 정확한 표현에 의존하는 기존의 점수 집계를 지적하며, 이를 개선하기 위해 ParaEval이라는 평가 프레임워크를 제안합니다.

- **Technical Details**: ParaEval은 모든 답변 옵션에 대해 다양한 패러프레이즈(paraphrase)를 쿼리하여 모델을 평가합니다. 이를 통해 각 모델의 가장 유리한 표현에 기반하여 점수를 부여하게 되어, 동일한 지식을 가진 모델에 대한 허위 성능 격차를 1점 이하로 줄일 수 있습니다. 또한, 이 방법은 70B와 120B 크기의 최신 오픈소스 모델에서도 성능 개선이 지속적으로 나타났습니다.

- **Performance Highlights**: 기존 MCQA 평가 모델들은 동일한 지식을 가진 모델들 간에 4점 이상의 인위적 성능 격차를 드러냈지만, ParaEval을 통한 평가에서는 1%의 절대 차이로 감소했습니다. 따라서 ParaEval은 모델의 진정한 능력을 평가할 수 있는 신뢰할 수 있는 대안을 제공하며, 여러 패러프레이즈를 통한 평가 방법론 역시 기존 벤치마크의 격차를 해소하는 데 매우 효과적임을 보여주었습니다.



### Speaker Group Encoding in Self-supervised Speech Recognition Models (https://arxiv.org/abs/2606.10654)
- **What's New**: 이 논문에서는 자기 지도 방식의 음성 인식 모델(S3Ms)이 말하는 사람 그룹(SGs)에 대해 배우는 내용을 조사했습니다. 저자들은 다양한 상태의 S3Ms를 분석하였으며, 프리트레인(pretrained), 화자가 식별을 위한 파인튜닝(finetuned on speaker identification, SID), 자동 음성 인식(ASR)을 위한 파인튜닝, 공정성을 향상시키는 알고리즘을 사용한 ASR 파인튜닝을 포함합니다. 연구 결과, S3Ms는 성별, 나이, 방언, 민족, 모국어 여부 등의 몇 가지 화자 그룹 카테고리(SGCs)에 대한 정보를 인코딩한다는 것을 발견했습니다.

- **Technical Details**: S3M은 음성 인식에서 매우 강력한 도구로, 이 논문은 S3M의 다양한 아키텍처에서 층별(layer-by-layer) 단위로 화자 그룹 정보(SGI)가 어떻게 존재하는지를 분석하고 있습니다. 연구자들은 특히 SGC가 중간 레이어에서 음성학적 변이(phonetic variance)가 더 많이 암호화되고, 마지막 레이어는 의미론적 변이(semantic variance)에 적합하다는 점을 강조합니다. 실험을 통해, 저자들은 모델의 각 레이어에서 SGC의 존재 여부가 어떻게 변화하는지를 조사하였으며, SGI가 인코딩되는 방법을 설명합니다.

- **Performance Highlights**: 실험 결과, SID를 위한 파인튜닝이 음성학적 변동이 큰 SGC를 증폭하는 반면, ASR을 위한 파인튜닝은 이러한 정보를 제외하는 경향이 있음을 알 수 있습니다. 공정성을 위해 설계된 ASR 알고리즘은 S3Ms에서 SGI가 인코딩되는 정도에 변화를 주었지만, 이는 주로 음성학적 변동이 있는 SGC에 해당하는 경우가 많았습니다. 이러한 발견들은 더 공정한 ASR 알고리즘을 설계하는 데 도움이 될 수 있는 방향성을 제시합니다.



### Dynamic Linear Attention (https://arxiv.org/abs/2606.10650)
Comments:
          Accepted by ICML 2026

- **What's New**: 이 논문은 Dynamic Linear Attention (DLA)라는 동적 메모리 모델링 프레임워크를 제안합니다. DLA는 정적 상태 병합 정책 대신 정보 변화를 기반으로 상태 경계를 적응적으로 결정하여 의미 전환 주위의 고해상도 표현을 유지합니다. 이 접근 방식은 복잡한 토큰 중요도를 다루면서 메모리 성장을 제어할 수 있도록 설계되었습니다.

- **Technical Details**: DLA는 (i) 정보 인식 동적 상태 병합(Information-Aware Dynamic State Merging)과 (ii) 용량 제한 메모리 모델링(Capacity-Bounded Memory Modeling)을 통합합니다. 정보 인식 동적 상태 병합은 토큰 수준의 정보 변동에 따라 상태의 경계를 동적으로 결정하여, 메모리의 시간적 순서를 보존하면서 불필요한 정보를 최소화하는 병합을 수행합니다.

- **Performance Highlights**: DLA는 16개의 데이터 세트에 대한 평가에서 최신 멀티 상태 방법인 Log-Linear Attention을 지속적으로 초월하는 성능을 보였습니다. DLA는 Mamba-2와 같은 백본 모델을 사용할 때, 유사한 매개변수 예산에서 풀 어텐션 트랜스포머와 비슷한 성능을 달성했습니다. DLA는 Log-Linear Attention보다 높은 처리량과 낮은 런타임 메모리 소비를 통해 효율성을 극대화했습니다.



### Small Data, Big Noise: Adversarial Training for Robust Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2606.10610)
Comments:
          Accepted to Findings of ACL 2026

- **What's New**: 본 논문에서는 Small Data Big Noise (SDBN)라는 새로운 프레임워크를 제안합니다. SDBN은 기존의 파라미터 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 접근법에 적대적인 훈련(adversarial training) 원칙을 통합하여 모델의 강인성과 일반화 능력을 향상시킵니다. 이는 특히 제한된 훈련 데이터와 노이즈가 있는 조건에서 뛰어난 성능을 보입니다.

- **Technical Details**: SDBN은 두 가지 변형 방식, 즉 SDBN-h와 SDBN-p를 소개합니다. SDBN-h는 문자 수준의 수정(character-level edits)으로 최악의 변형을 선택하는 반면, SDBN-p는 LLM에서 생성된 변형(variants)을 사용하여 생성 작업에서 강인한 최적화를 시도합니다. 이러한 접근 방식은 PEFT의 기존 한계를 극복하고 파라미터 재조정 없이 모델을 더욱 효과적으로 활용할 수 있게 합니다.

- **Performance Highlights**: 다양한 벤치마크에서 SDBN을 테스트한 결과, 특히 낮은 자원 환경에서 말과 문자 수준의 손상에 대한 견고성이 크게 향상된 것을 확인했습니다. SDBN은 고전적 PEFT 접근법보다 더 나은 성능을 보여주며, 도메인 변화나 데이터를 제한받는 상황에서도 우수한 강인성을 발휘합니다.



### ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models (https://arxiv.org/abs/2606.10581)
- **What's New**: 이번 논문은 음성 대화 모델(Speech Language Models, SLMs)의 비언어적 신호(paralinguistic cues) 인식을 통한 응답 조정 능력과 관련하여 주목할 만한 연구 결과를 제시합니다. 제안된 방법인 ParaBridge는 모델의 추론 단계에서 비언어적 신호를 인식하고 이를 바탕으로 적절한 반응을 이끌어내는 자신만의 학습을 가능하게 합니다. 이 연구는 인공지능 비서가 아이의 목소리처럼 감정적 신호를 더 잘 인식하도록 하는 방안을 찾고 있으며, 이는 현재 SLM의 경계를 넓히는 데 기여할 것입니다.

- **Technical Details**: ParaBridge는 셀프 디스틸레이션(self-distillation) 방법을 활용하여 비언어적 신호에 대한 인지와 반응 사이의 격차를 좁히는 새로운 프레임워크를 제공합니다. 모델의 훈련 과정에서 잠재적으로 유용한 비언어적 지침을 제공하며, 단순한 균형 잡힌 응답을 목표로 설정하지 않고도 다양한 맥락에서 신뢰성 있는 반응을 생성합니다. 이 프레임워크는 외부 데이터나 인간의 레이블 없이도 모델의 성능을 향상시킵니다.

- **Performance Highlights**: 논문에서 제시한 ParaBridge는 VoxSafeBench에서 비언어적 신호에 의한 반응 조정 능력을 14.6%에서 40.3%로 크게 향상시켰으며, EchoMind에서 평균 점수를 3.27에서 3.92로 개선했습니다. 또한, MMAU-Pro, VoiceBench, GPQA와 같은 다양한 지표에서도 성능 저하 없이 기존 모델의 범위 내에 머무르고 있음을 보여주었습니다. ParaBridge는 안전성 중심 훈련에서 공감 중심 대화로의 전이와 같은 일반성(generalization) 있는 특성을 통해 다양한 비언어적 신호에 잘 대응하는 모델로 자리매김하고 있습니다.



### Hidden Consensus:Preference-Validity Compression in Human Feedback (https://arxiv.org/abs/2606.10569)
Comments:
          28 pages. When AI learns from human feedback, it forces a single "correct" answer, but sometimes multiple answers are all genuinely valid, and that nuance gets thrown away

- **What's New**: 이번 연구에서는 Reinforcement Learning from Human Feedback (RLHF) 파이프라인에서 인식된 편향 문제를 제기하고 있습니다. 구체적으로, 다원적 해석이 있는 사회에서 인간의 판단을 간단한 스칼라 보상 신호로 감소시키는 과정이 "Preference-Validity Compression"이라 불리는 문제를 일으킬 수 있습니다. 이 연구의 초점은 다문화적 관점에서 허용 가능한 다수 옵션을 단일 최적화 목표로 축소하는 과정의 부정적 영향을 살펴보는 것입니다.

- **Technical Details**: 이 연구에서는 20명의 말레이시아 참가자들로부터 수집된 321개의 선호 이벤트를 분석하여 다원적 수용 가능성이 실제로 존재한다는 것을 보여줍니다. 구체적으로, 79%의 프롬프트에서 한 명 이상의 응답이 다수의 수용 임계값에 도달했지만, argmax 집계 방식에 의해 버려지거나 무시됩니다. 이러한 현상은 참가자들이 종종 여러 답변을 동시에 유효한 것으로 간주한다는 것을 시사합니다.

- **Performance Highlights**: 이 연구의 실증 결과는 RLHF 스타일 피드백 집계에서의 측정 실패를 수치적으로 나타냅니다. 단기 승자 집계는 우세한 응답의 우위를 과장하게 되며, 다수의 지지받는 응답을 고려할 경우 그 격차가 상당히 줄어들 수 있습니다. 이러한 발견들은 집계 프로토콜이 다원적 해석을 단일 보상 목표로 축소시키는 경우 측정 유효성을 좌우하는 주요 병목현상이 될 수 있음을 보여줍니다.



### Benchmarking Knowledge Editing using Logical Rules (https://arxiv.org/abs/2606.10554)
Comments:
          Accepted at the 24th International Semantic Web Conference 2025

- **What's New**: 본 논문은 대형 언어 모델(Large Language Models, LLMs)에서의 지식 편집(knowledge editing) 방법의 성능을 개선하기 위한 새로운 벤치마크를 제안합니다. 이는 단순히 편집된 사실을 기억하는 것을 넘어, 그 논리적 결과를 평가하는 데 중점을 둡니다. 기존의 벤치마크가 이러한 논리적 결과를 간과해 왔다는 점에서, 연구팀은 지식 그래프에서 관련 논리 규칙을 추출하여 다중 단계 질문(multi-hop questions)을 생성하는 방식을 도입하였습니다.

- **Technical Details**: 이 방법론은 AMIE3라는 규칙 추출 시스템을 사용하여, 편집된 사실과 관련된 논리 규칙을 자동으로 생성합니다. 생성된 질문은 LLM이 편집된 사실을 통해 관련 지식을 일관되게 유지하는지 평가하는 데 사용됩니다. 다양한 지식 편집 방법을 테스트하여 직접 편집된 사실과 논리적 결과 간의 성능 차이를 분석하는 것이 본 연구의 핵심입니다.

- **Performance Highlights**: 기존의 지식 편집 방법들은 직접적인 편집에서는 높은 정확성을 보였으나, 논리적 결과를 필요로 하는 질문에서는 최대 24%의 성능 차이를 보였습니다. 이는 지식 편집 방법들이 상호 연관된 정보의 일관성을 유지하는 데에서 부족함이 있음을 보여줍니다. 따라서, 논문은 실질적인 시나리오에서 지식의 상호 의존성을 처리하는 더욱 강력한 평가 프레임워크의 필요성을 강조합니다.



### Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language Models (https://arxiv.org/abs/2606.10537)
Comments:
          Technical Report

- **What's New**: 이 논문은 확산 대형 언어 모델(Diffusion large language models, dLLM)의 비효율성을 해결하기 위해 Prefilling-dLLM이라는 훈련이 필요 없는 새로운 프레임워크를 제안합니다. 이 프레임워크는 입력 프리픽스(prefix)를 N개의 청크로 나누고, 키-값(KV) 표현을 한 번만 캐시하여 재사용함으로써 긴 문맥(long-context)에서의 재계산을 방지합니다. 이를 통해 긴 문맥 시나리오에서의 속도를 획기적으로 향상시켰습니다.

- **Technical Details**: Prefilling-dLLM은 프리픽스 KV 캐시를 전처리 단계에서 한 번만 계산하고, 모든 디코딩 단계에서 이를 재사용함으로써 계산 복잡성을 줄입니다. 구체적으로, 이 모델은 프리픽스를 고정 크기의 청크로 나누고, 각 청크 내에서 주의력을 유지하여 각 디코딩 단계에서 처리할 청크를 선택합니다. 이를 통해 일부 키-값 쌍을 기반으로 한 재부하를 줄이고, 최종적으로는 O((Lp + Ld)^2·T)에서 O(N·C^2 + (Ld^2 + K·C)·T)로 계산 복잡성을 줄였습니다.

- **Performance Highlights**: Prefilling-dLLM은 LongBench와 InfiniteBench에서 8K에서 32K 문맥을 사용하여 9.1배에서 28배에 이르는 속도 향상을 달성하며, dLLM 가속화 기법 중 최첨단의 품질을 보여주었습니다. 또한 최적화된 주의(attention) 커널을 통해 비연속적으로 캐시된 청크 KV에서 디코딩을 병렬화함으로써 속도를 크게 향상시켰습니다. 이 연구는 기계 번역 및 자연어 처리 분야에서 효율성을 향상시킬 가능성을 제시합니다.



### LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization (https://arxiv.org/abs/2606.10531)
Comments:
          Accepted by ICML 2026

- **What's New**: 본 논문에서는 2비트 가중치만 사용하는 벡터 양자화 인식 훈련 프레임워크인 LC-QAT를 제안합니다. 이는 학습된 아핀 맵을 통해 양자화된 가중치를 표현하여 고품질의 사후 훈련 초기화를 제공하며, 코드북 조회 없이 엔드-투-엔드 최적화를 가능하게 합니다. 실험 결과, LC-QAT는 기존의 최첨단 양자화 인식 훈련 방식보다 더 우수한 성능을 보입니다.

- **Technical Details**: LC-QAT는 비선형 제약 조건을 갖는 코드북을 활용하여, 각 가중치 그룹의 표현성을 높이는 구조를 취합니다. 이 방식은 각 코드워드를 공유하는 선형 변환을 통해 생성되며, 가중치 행렬을 통해 그래디언트가 양자화 과정을 통해 전파될 수 있도록 합니다. 따라서, 기존의 이웃 탐색 없이도 간단한 반올림 및 클램핑 연산으로 양자화가 가능합니다.

- **Performance Highlights**: 다양한 대규모 LLM을 대상으로 한 실험 결과, LC-QAT는 훈련 데이터의 오직 0.1%에서 10%만 사용하여도 뛰어난 성능을 발휘했습니다. 초저비트 모델 배포를 위한 실용적이고 확장 가능한 솔루션으로 자리 잡으며, 기존 SQ-QAT 및 VQ-QAT 방법에 비해 데이터 효율성과 정확도 모두에서 향상된 결과를 보여줍니다.



### UniSVQ: 2-bit Unified Scalar-Vector Quantization (https://arxiv.org/abs/2606.10520)
Comments:
          Accepted by ICML 2026

- **What's New**: UniSVQ는 2비트 양자화(framework)를 위한 새로운 접근법으로, 스칼라(SQ)와 벡터 양자화(VQ)의 장점을 결합합니다. 이 방법은 코드북(codebook)을 정칙 변환(affine transform)으로 매개변수화하여, 성능 저하를 최소화하면서 코드북의 저장 오버헤드와 디코딩 복잡성을 줄입니다. 또한, 데이터 기반(block-wise) 미세 조정 전략을 도입하여 양자화 재구성 오류를 직접 감소시킵니다.

- **Technical Details**: UniSVQ는 정칙한 양자화 그리드(spatial structure of the quantization grid)를 활용하여 스칼라와 벡터 양자화의 경계에 위치한 모델 구조를 만듭니다. 이 과정에서 VQ의 코드북을 정칙 변환으로 대체하여 부수적인 매개변수 수를 줄이고, 최적화된 SQ 행렬 곱셈 커널을 재사용할 수 있도록 합니다. 따라서 UniSVQ는 고전적인 SQ와 비교해 더욱 유연한 성능을 발휘합니다.

- **Performance Highlights**: 여러 LLM 모델과 제로샷(zero-shot) 벤치마크에서 이루어진 실험 결과, UniSVQ는 최신 SQ 방법에 비해 뛰어난 성능을 보여주며, VQ 방법과 유사하거나 그보다 나은 성능을 기록했습니다. 또한 코드북 관련 메모리 트래픽을 줄임으로써 추론(inference) 효율성을 개선하여 실제 사용에서 더 높은 처리량(throughput)을 달성합니다.



### Detecting Speculative Language in Biomedical Texts using Recurrent Neural Tensor Networks (https://arxiv.org/abs/2606.10471)
Comments:
          12 Pages

- **What's New**: 이번 연구에서는 생물 의학 기사에서 추측적 언어(speculative language)를 자동으로 탐지하는 방법을 연구했습니다. 이를 위해 분산 문장 표현(distributed sentence representations)과 심층 학습(deep learning) 기술을 활용하였습니다. 이러한 자동 탐지는 정보 검색(information retrieval) 및 다문서 요약(multi-document summarization)에 중요한 영향을 미칠 수 있습니다.

- **Technical Details**: 연구에서는 두 가지 분산 문장 표현 방법인 Paragraph Vector 모델과 Recursive Neural Tensor Network(RNTN)를 비교하였으며, 이들 방법은 지원 벡터 머신(Support Vector Machines), 나이브 베이즈(Naive Bayes)와 패턴 매칭(pattern matching) 같은 세 가지 기본 알고리즘과 성능을 비교했습니다. RNTN은 F1 점수 0.885로 가장 높은 성능을 보였고, Paragraph Vector 모델은 F1 점수 0.368로 효과적이지 않았습니다. 이 연구는 기존 방법의 성능 차이에 대한 요인들을 논의합니다.

- **Performance Highlights**: RNTN 모델이 SVM(선형 bigram)의 F1 점수 0.881보다 우수한 성능을 보였고, 두 모델 간의 성능 차이에 대해 깊이 있는 분석을 제공하였습니다. 연구에서 활용된 BioScope 및 BioMed Corpus 데이터셋을 통해 다양한 문장에서 추측적 언어를 인식하는 것이 가능하다는 것을 시연했습니다. 연구의 결과는 생물 의학 문서에서의 정보 탐색과 요약 작업에서 중요한 시사점을 제공합니다.



### Large Language Models as Modal Models in Linguistics (https://arxiv.org/abs/2606.10467)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 언어 이론에 대한 중요성을 둘러싼 논쟁을 촉발시키고 있습니다. 이러한 논의는 주로 LLMs를 인간 언어와 무관하다고 보는 insulationism, LLMs가 기존 언어 이론을 대체할 수 있다고 주장하는 eliminativism, 그리고 LLMs를 언어 연구의 유용한 도구로 보는 conciliationism의 세 가지 입장으로 나눌 수 있습니다. 이 논문은 과학 철학의 모달 모델링(modal modeling) 틀을 적용하여 LLMs의 인식적 가치(epistemic value)를 명확히 하고자 합니다.

- **Technical Details**: LLMs는 구조적 대응(structural correspondence)이 없이도 최소 모델(minimal models)로서 진정한 인식 가치를 지닌다고 주장합니다. LLMs는 언어 습득(language acquisition)과 언어 능력(linguistic competence)에 대한 모달 주장(modal claims)을 테스트함으로써 가능성 설명(how-possibly explanation, HPE)을 제공할 수 있습니다. 이 연구에서는 LLMs가 HPE를 넘어 실제 설명(how-actually explanation, HAE)을 제공하기 위해 필요한 조건을 다루며, 기존 LLMs가 이 조건을 만족하지 못한다고 결론지었습니다.

- **Performance Highlights**: LLMs의 설명력을 HPE에서 HAE에 이르는 연속체로 이해하는 관점을 제안합니다. 이로 인해 insulationist와 eliminativist는 LLM의 설명력을 온/오프 기준으로 평가하는 잘못을 공유하며, conciliationism의 평가 기준도 불명확함을 지적합니다. 이러한 새로운 틀은 LLM의 설명적 중요성을 과장하거나 축소하지 않고 더 정확한 평가 기초를 제공합니다.



### LakeQA: An Exploratory QA Benchmark over a Million-Scale Data Lak (https://arxiv.org/abs/2606.10460)
- **What's New**: 최근 대형 언어 모델(LLMs)은 질문 응답(QA) 작업에서 빠른 발전을 보여주었습니다. 하지만 실제 질문은 정확한 증거 문서와 쌍으로 이루어지지 않는 경우가 많습니다. 'LakeQA'라는 새로운 벤치마크를 소개하는 본 논문은 대량의 데이터에서 증거를 검색하고 분석하는 능력을 동시에 강조합니다.

- **Technical Details**: LakeQA는 약 9.5TB의 위키백과 및 오픈 소스 정부 데이터를 이용해 구축된 이종 데이터 모음으로, 구조화된 데이터와 비구조화된 데이터가 혼합되어 있습니다. 각 작업은 적어도 한 명의 박사급 전문가에 의해 주석이 달리며, 멀티 홉(multihop) 추론이 필요합니다. 이 과정에서 에이전트는 올바른 문서를 발견하고 여러 출처에서 증거를 통합해 답변을 생성합니다.

- **Performance Highlights**: 실험 결과, GPT-5.2는 LakeQA에서 18.37%의 정확히 일치하는 점수만을 기록하여, 이 벤치마크가 도전적인 것을 보여줍니다. 정확도는 추론 강도가 증가함에 따라 급격히 감소하며, 이는 현재 LLM 에이전트들이 EQA에서 증거 검색 및 탐색의 어려움을 겪고 있음을 나타냅니다. 결과적으로, LakeQA는 검색과 추론의 복잡성을 동시에 확장시키는 데 중점을 둔 최초의 벤치마크입니다.



### Which LoRA? An Empirical Study on the Effectiveness of LoRA Techniques During Multilingual Instruction Tuning (https://arxiv.org/abs/2606.10428)
- **What's New**: 이 연구는 기존의 LoRA (Low-Rank Adaptation) 변형들이 기본 LoRA보다 다국어 지침 튜닝에서 유리한지 여부를 조사합니다. 실험 결과, 다양한 언어에서 수집된 데이터 세트를 통해 복잡한 LoRA 변형이 기본 LoRA에 비해 큰 이점을 제공하지 않음을 보여주었습니다. 이는 최근 LoRA 변형에서 도입된 아키텍처 수정이 항상 다국어 적응 성능 향상으로 이어지지 않는다는 것을 시사합니다.

- **Technical Details**: 본 연구에서는 여러 LoRA 변형 중 Weight-Decomposed Low-Rank Adaptation (DoRA), Vector-Based Random Matrix Adaptation (VeRA), Adaptive Low-Rank Adaptation (AdaLoRA), Principal Singular Values and Singular Vectors Adaptation (PiSSA)을 선택했습니다. 이들 변형들은 연구 커뮤니티에서 수용도가 높으며, 기본 LoRA와 비교하여 다국어 지침 튜닝 과정에서의 성능을 평가하는 데 사용됩니다. 저자들은 명확한 하이퍼파라미터 조정을 통해 실험을 설계하고, 모든 레이어에 LoRA를 적용해야 함을 강조하였습니다.

- **Performance Highlights**: 실험 결과, 1%의 특정 언어 데이터(TL)를 도입하는 것만으로도 여러 구성에서 성능이 향상되었으며, TL 비율을 높일 경우 오히려 성능 저하가 발생했습니다. LoRA 변형의 각 유형 및 TL 비율에 따른 최상의 F1 점수는 LoRA, DoRA*, 그리고 DoRA에 의해 달성되었습니다. 그러나 네 가지 언어 비율에 대한 ANOVA 검정 결과, 통계적으로 유의미한 차이는 발견되지 않았습니다.



### WebChallenger: A Reliable and Efficient Generalist Web Agen (https://arxiv.org/abs/2606.10423)
- **What's New**: 이 논문에서는 웹 네비게이션의 자율성을 높이기 위해 새로운 웹 에이전트 프레임워크인 WebChallenger를 소개합니다. 이 프레임워크는 기존의 LLM 모델들이 가지는 한계, 특히 인간의 인지적 장점을 반영하지 못하는 점을 개선하기 위해 설계되었습니다. 특히, 선택적 주의(selective attention), 지속적 기억(persistent memory), 절차적 유창성(procedural fluency)이라는 세 가지 장점을 기반으로 한 구조적 페이지 표현(PageMem)을 통해 여러 웹사이트에 걸쳐 일반화 가능한 방식을 보여줍니다.

- **Technical Details**: WebChallenger는 세 가지 인지적 장점을 반영하기 위해, 먼저 수집한 DOM에서 결정론적으로 구성된 구조적 페이지 표현(PageMem)을 사용합니다. 이 표현은 페이지를 의미적 섹션의 계층으로 만들어줍니다. 또한, 세 가지 기능인 관찰 파이프라인(observation pipeline), 탐색 및 메모리 시스템(exploration and memory system), 복합 작업 워크플로우(compound action workflows)를 통해 에이전트가 웹사이트를 효과적으로 탐색하고 각 웹사이트에서의 과거 경험을 활용할 수 있도록 지원합니다.

- **Performance Highlights**: 이 시스템은 사전 훈련된 오픈 가중치 모델을 사용하여 WebArena에서 56.3%, VisualWebArena에서 48.7%, Online-Mind2Web에서 51.0%, WorkArena에서 70.9%의 성능을 달성하며, 상용 시스템에 비해 상당히 낮은 비용으로 접근합니다. 이러한 연구 결과는 현재 LLM 모델이 충분한 추론 능력을 갖추고 있음을 보여주며, 필요한 것은 관찰, 기억, 행동에 대한 적절한 구조라는 점을 강조합니다.



### KCSAT-ML: Probing Reasoning Models with Nationwide-Cohort Human Difficulty (https://arxiv.org/abs/2606.10403)
Comments:
          18 pages, 14 figures, 8 tables

- **What's New**: 이번 논문에서는 인간 성과를 반영한 난이도 신호를 갖춘 새로운 수학적 추론 벤치마크인 KCSAT-ML을 소개합니다. KCSAT-ML은 2014년부터 2025년까지의 한국 대학 수학능력시험(KCSAT)의 664문제로 구성되며, 이를 통해 339개의 핵심 문항에 대한 공식 오류 비율을 제공합니다. 이 벤치마크는 언어 모델의 성격을 깊이 분석할 수 있는 기회를 제공하며, 이를 위해 'Difficulty-aligned Reasoning Gain (DRG)'라는 새로운 메트릭을 도입합니다.

- **Technical Details**: KCSAT-ML은 표준화된 조건 하에서 수집된 데이터에 기반하여, 각 문제의 난이도를 보다 정확히 측정할 수 있도록 상위 339문항에 대해 전국적인 수험생 집단에서 얻은 오류 비율을 제공합니다. 모델 성능을 평가하는 데 있어, 모델이 어떤 문항에서 실패하는지를 기반으로 인간의 난이도 축과 정렬되는지를 평가하는 DRG 메트릭이 도입되었습니다. 이 메트릭을 통해 모델의 성취도와 인간의 성취도가 어떻게 다르게 나타나는지를 파악할 수 있습니다.

- **Performance Highlights**: 연구에서 발견된 주요 패턴은 다음과 같습니다. (i) 저예산 모델의 정확도는 고난이도 문항에서 급격히 떨어지며 모든 모델 크기에서 동일한 현상을 보입니다. (ii) 테스트 시간 스케일링(TTS)은 사용하는 토큰 수를 증가시키며, 오류 비율에 비례해 정확도 증가가 비선형적입니다. (iii) 동일 모델군 내에서 TTS가 가장 어려운 문항에 대해 부정적 영향을 미치고, 쉬운 문항에 대해서는 과도한 사고로 이어지는 등 서로 다른 실패 양상도 관찰되었습니다.



### Harnessing the Collective Intelligence of AI Agents in the Wild for New Discoveries (https://arxiv.org/abs/2606.10402)
- **What's New**: 본 논문에서는 EinsteinArena라는 새로운 에이전트 네이티브 플랫폼을 소개합니다. 이 플랫폼은 공개된 연구 문제를 다루며, 문제 해결을 위한 다양한 기능을 제공합니다. 에이전트들이 실시간으로 문제를 공유하고, 검증할 수 있으며, 그들의 아이디어를 상호 대화 방식으로 발전시킬 수 있습니다.

- **Technical Details**: EinsteinArena는 세 가지 주요 요소로 구성됩니다: (i) 공적으로 검증된 문제의 큐레이션된 컬렉션, (ii) 각 문제의 최선의 해결책을 추적하는 실시간 리더보드, (iii) 에이전트가 중간 결과를 공유하고 실패한 접근 방식을 문서화하며 서로의 발견 위에 구축할 수 있는 공개 논의 포럼입니다. 모든 자료는 투명하게 공개되어 있어, 에이전트들은 현재의 경계를 쉽게 확인할 수 있습니다.

- **Performance Highlights**: 2026년 5월 기준으로 EinsteinArena에서 12개의 새로운 최첨단 수학적 결과가 발견되었습니다. 특히, 유지 놈 문제(kissing number problem)의 경우, 관련 연구에서 가장 알려진 경계값을 593에서 604로 개선하는 성과를 보였습니다. 이러한 결과는 에이전트 간의 협동 작업을 통해 이루어진 것이며, 이는 분산된 과학적 발견이 자율 에이전트 간의 공개적인 상호작용을 통해 가능하다는 것을 입증합니다.



### Do Vision-Language Models See or Guess? Measuring and Reducing Textual-Prior Reliance with a Phrasing-Controlled Benchmark (https://arxiv.org/abs/2606.10400)
Comments:
          17 pages, 7 figures, Submitted to EMNLP 2026

- **What's New**: 이 논문은 Vision-Language Models (VLMs)의 정확도를 측정하는 새로운 벤치마크를 제안합니다. 540장의 이미지와 이를 기반으로 한 네 가지 질문 변형을 통해 질문의 문구가 어떻게 모델의 응답에 영향을 미치는지 평가합니다. 기존 연구와의 차별점은 질문의 외형이 아닌 시각 정보를 고정한 상태에서 질문 문구의 변화만으로 모델의 의존도를 측정한 것입니다. 특히 Vision-Grounded 변형은 이미지를 직접 기반으로 작성하여 모델의 텍스트 의존성을 최소화합니다.

- **Technical Details**: 논문은 540장의 이미지로 구성된 벤치마크를 구축하고, 각 이미지에 대해 네 가지 질문 변형을 생성했습니다. 이 과정에서 각 변형은 시각 정보 없이 질문 문구만으로 모델의 정확도를 측정할 수 있게 설계되었습니다. 사용된 기술적 방법론 중 하나는 GRPO (Group Relative Policy Optimization)와 LoRA (Low-Rank Adaptation)를 활용하여 VLM의 텍스트 의존성을 낮추는 방법입니다. 이러한 기술들을 통해 열한 개의 VLM 모델이 평가되었으며, 각 모델의 응답 정확도는 다르게 나타났습니다.

- **Performance Highlights**: 모델의 성능 측면에서, 모든 VLM은 이미지 기반의 Vision-Grounded 변형에서 성능이 저하되었고, 특히 오픈 소스 모델들은 그 영향이 더욱 두드러졌습니다. 이 연구에서는 텍스트 전용 질문에서의 모델 성능이 1%에서 9%로 드랍하는 결과를 보여주어 이미지 의존도가 실제로 존재함을 확인했습니다. GRPO 후속 훈련은 모든 변형에서 성능 개선을 가져왔으며, 이는 일반적인 상황에서도 유의미한 성과로 이어집니다.



### Expert-Level Crisis Detection in Mental Health Conversations (https://arxiv.org/abs/2606.10380)
- **What's New**: 이 논문에서는 대화형 위기 개입(crisis intervention)에 대한 새로운 데이터셋인 CRADLE-Dialogue를 소개합니다. 이전의 연구들은 주로 정적인 텍스트에 집중하였으나, 실제 대화에서는 위기 신호가 점진적으로 드러나는 과정을 추적해야 합니다. 이를 위해 600개의 다중 턴 대화를 수집하고, 정신 건강 전문가가 다중 라벨을 통해 위기 상황을 주석 달았습니다.

- **Technical Details**: CRADLE-Dialogue는 자살 사고, 자해, 아동 학대 등 다양한 임상 기반 위험을 포함한 대화형 위기 탐지를 위한 벤치마크 데이터셋입니다. 연구에서는 초기 경고 신호(Alert)와 위기가 명확히 인식되는 턴(Confirm) 간의 차이를 구분하는 Alert-Confirm 평가 프로토콜을 제안합니다. 이 프로토콜을 통해 모델이 위험이 나타나는 시점을 추적할 수 있는 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 모델들은 위험이 나타나는 시점을 식별하는 것이 단순히 존재하는 위험을 인식하는 것보다 훨씬 더 어렵다는 것을 보여주었습니다. 기존 모델들은 중간 40%에서 높은 60%의 Micro F1 점수만을 달성했으며, 새로운 32B 파라미터 모델은 기존 오픈 소스 모델을 능가하고 소유 모델들과의 비교에서도 경쟁력 있는 성과를 달성했습니다.



### PADD: Path-Aligned Decompression Distillation for Non-Router Teacher to Guide MoE Student Learning (https://arxiv.org/abs/2606.10369)
Comments:
          published in ICML 2026

- **What's New**: 이 논문에서는 Path-Aligned Decompression Distillation (PADD)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 밀집 모델(dense models)에서 Mixture-of-Experts (MoE) 모델로 지식을 증류(distillation)하면서 명시적인 라우팅(routing) 없이도 고품질의 라우팅 정책을 학습할 수 있도록 합니다. PADD는 초기화 단계와 훈련 단계로 구성된 네 가지 단계로 지식 증류를 체계적으로 구성하여, 학습 효율성을 크게 향상시킵니다.

- **Technical Details**: PADD는 두 개의 주요 단계로 나뉘며, 첫 번째 단계인 초기화 단계(Stage I)에서는 교사 신경망(teacher neuron) 클러스터링 및 학생 전문가(student expert) 웜업(warmup)을 수행하여 학생 전문가의 기능적 다양성을 구축합니다. 두 번째 단계는 주로 온라인 적응형 증류(online adaptive distillation), 경로-정제 정책 최적화(path-refined policy optimization), 및 보상-증강 로드 밸런싱(reward-augmented load balancing)을 통합하여 학생이 교사로부터 직접 지식을 효과적으로 흡수하도록 돕습니다.

- **Performance Highlights**: 실험 결과 PADD는 동등한 추론 비용 하에서 강력한 기준선보다 상당한 성능 향상을 보여주며, MoE 학생이 밀집 교사의 성능을 동일하게 맞추거나 초과할 수 있음을 입증합니다. 또한 PADD는 교사에서 학생으로의 효과적인 지식 증류와 안정적인 라우팅 행동을 제공하여, 구조적 불일치 문제를 해결하며 모델의 효율성을 극대화합니다.



### Routing-Aware Expert Calibration for Machine Unlearning in Mixture-of-Experts Language Models (https://arxiv.org/abs/2606.10338)
- **What's New**: 이번 연구에서는 Mixture-of-Experts (MoE) 아키텍처에서 머신 언러닝에 대한 새로운 접근 방식인 TRACE를 제안합니다. 기존의Dense 모델에 비해 MoE 모델은 각 토큰을 전문가의 희소 부분 집합에 할당하는 라우터를 활용합니다. 이로 인해 정보가 여러 전문가에 분산될 때, 언러닝 과정에서의 비효율성과 처리 문제가 발생할 수 있음을 보여줍니다.

- **Technical Details**: TRACE는 오프라인 활성화 통계를 통해 forget-critical 전문가를 식별하고, 각 전문가의 retain 활성화 주파수와 forget 측의 주파수를 일치시키기 위해 토큰 수준의 retain 손실을 재조정합니다. 이는 MoE 아키텍처의 특성을 반영하여, 각 전문가에게 더 나은 보호를 제공하는 시스템을 구축하고자 하는 노력입니다.

- **Performance Highlights**: 실험 결과, TRACE는 여러 MoE LLMs에서 forget-utility trade-off를 일관되게 개선하며, WMDP 및 MUSE-BOOKS 테스트 세트에서 강력한 기준선보다 9%의 유틸리티 개선을 달성했습니다. 특히 MUSE-BOOKS metrics의 세 가지에서 최고 성능을 보이면서, 언러닝의 효과성을 잘 입증했습니다.



### The Order Matters: Sequential Fine-Tuning of LLaMA for Coherent Automated Essay Scoring (https://arxiv.org/abs/2606.10327)
- **What's New**: 이번 연구에서는 Automated Essay Scoring (AES) 시스템의 성능을 향상시키기 위해 LLaMA-3.1-8B 모델을 활용한 task-aware fine-tuning을 조사합니다. 기존의 접근 방법들이 여러 글 요소를 개별적으로 평가하는 문제를 지적하며, 논리적 연관성을 반영한 교육적 커리큘럼 설계가 AES의 효과성을 개선할 수 있음을 보여줍니다. 이를 통해 AES의 일관성과 일반화 능력을 크게 향상시킬 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 세 가지 훈련 커리큘럼, 즉 Sequential, Independent, Randomized 방식을 비교합니다. Sequential fine-tuning이 가장 뛰어난 결과를 보이며, F1 점수는 65% (evidence) 및 87% (conclusion)로 나타났습니다. 또한, LoRA를 활용한 4-bit quantization 기법은 모델의 효율성을 극대화하는 데 기여하였습니다.

- **Performance Highlights**: 세 가지 접근 방식 중에서 Sequential fine-tuning이 다른 모델보다 뛰어난 성능을 보여주었습니다. 이는 교육 분야에서 작고 효율적으로 조정된 모델이 대규모 LLM과 경쟁할 수 있는 가능성을 제시하며, AES 시스템의 확장성과 비용 효과적인 평가 방식을 현실화할 수 있는 길을 열어줍니다. 이 연구는 향후 AES 시스템의 قابلیت을 크게 향상시키기 위한 기반이 됩니다.



### TabClaw: An Interactive and Self-Evolving Agent for Spreadsheet Manipulation and Table Reasoning (https://arxiv.org/abs/2606.10316)
Comments:
          5 pages, 2 figures

- **What's New**: TabClaw는 사용자가 CSV 또는 Excel 파일을 업로드하고 자연어 질문을 할 수 있는 오픈 소스 대화형 AI 에이전트입니다. 이 시스템은 애매한 사용자의 의도를 명확히 하고, 수정 가능한 실행 계획을 공개하며, 전문가 에이전트를 배치하여 여러 테이블에 대한 병렬 분석을 지원합니다. 또한 TabClaw는 사용자의 메모리를 지속적으로 기록하고 반복적인 도구 사용 패턴에서 재사용 가능한 기술을 증류하여 사용자의 선호도에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: TabClaw는 사용자 질문을 명확히 하고 이를 수정 가능한 계획으로 변환하여 실행하는 ReAct 스타일의 도구 사용 분석 루프를 갖추고 있습니다. 또한, 이 시스템은 작업 상태에 대한 추론과 테이블 도구 호출을 번갈아가며 수행하는 루프를 스트리밍하여 사용자가 중간 결과물을 실시간으로 검토할 수 있도록 합니다. 각 요청이 여러 테이블을 포함할 경우 각 테이블에 대해 전문화된 에이전트를 배정하고, 그들의 발견을 종합하여 결과를 제공합니다.

- **Performance Highlights**: TabClaw는 스프레드시트 조작 및 테이블 추론 벤치마크 실험에서 실행 가능한 작업 완료율과 추론 성능을 향상시켰습니다. 사용자는 이전의 작업 흐름을 검토하고, 필요한 경우 수정할 수 있으며, 이는 분석 과정의 투명성을 높입니다. TabClaw는 반복 데이터 분석 작업에 점진적으로 개인화되어 사용자가 매번 같은 가정을 다시 언급하지 않고도 작업을 수행할 수 있게 도와줍니다.



### Catching One in Five: LLM-as-Judge Blind Spots in Production Multi-Turn Transaction Agents (https://arxiv.org/abs/2606.10315)
Comments:
          13 pages, 1 figure, 5 tables

- **What's New**: 이번 연구에서는 생산 환경에서 운영되고 있는 다회전 음식 및 음료 주문 에이전트를 평가하기 위해 LLM-as-judge의 신뢰성에 대해 조사합니다. LLM이 실제 품질 문제를 얼마나 잘 감지하는지 분석하며, 기존의 인간 리뷰와의 일치성보다는 실제 결함의 감지율(recall)에 초점을 맞추었습니다. 연구 결과는 LLM의 결함 감지율이 매우 낮다는 것을 보여줍니다.

- **Technical Details**: 다회전 주문 시스템에서는 고객이 자연어로 음식 및 음료를 주문하는데, 이 LLM은 주문을받아들이고, 메뉴 검색 및 장바구니 생성 등의 작업을 수행합니다. 평가 시스템은 싱가포르의 커피 브랜드와 이중 언어 고객 기반을 가지고 있으며, 다양한 시나리오를 통해 랜덤화된 다회전 대화가 생성됩니다. LLM의 내장된 평가자가 세 가지 축(의도, 브랜드 목소리, 개인화)을 기준으로 점수를 매기지만, 상태 추적(state-tracking)이나 손해 복구와 같은 중요한 차원에 대한 분류가 없어 결함을 간과하는 구조가 드러났습니다.

- **Performance Highlights**: 연구 결과 저자는 LLM 평가자가 인간이 확인한 체계적인 문제의 25% 미만을 감지한다고 주장합니다. 예를 들어, 평가자가 한 배치에서 9개 패턴 중 2개(22%)만을 포착하는 반면, 100회차의 다른 배치에서는 인간 리뷰가 23개의 명확한 결함을 확인했음에도 불구하고 평가자는 실패로 표시한 턴이 0회였습니다. 이러한 결과는 LLM 평가자의 설계와 운영 메커니즘의 결함을 강조합니다.



### Early-Token Confidence Predicts Reasoning Quality in Multi-Agent LLM Deba (https://arxiv.org/abs/2606.10307)
Comments:
          15 pages, 8 figures, 4 tables; ACL Proceedings

- **What's New**: 이 연구는 다중 에이전트 LLM 시스템에서 추론 품질을 평가하는 새로운 접근 방식을 제안합니다. 특히, 오픈 엔디드 작업에 대한 평가가 어려운 가운데, 토큰 수준의 로그 확률을 활용하여 추론 품질을 예측할 수 있는지 여부를 탐구합니다. 조사 결과, 초기 토큰의 신뢰도가 추론 품질을 예측하는 데 가장 효과적이라는 것을 발견했습니다.

- **Technical Details**: 연구는 다중 에이전트 시스템에서의 논쟁 기반 에세이 채점 프레임워크를 사용하여 성과를 비교 분석합니다. 한 에세이에 대해 내용을 지지하는 Advocate와 반대하는 Skeptic이 각각 주장을 구성하며, 이 과정에서 생성된 토큰의 로그 확률을 통해 신뢰도를 평가합니다. 이를 통해 내부 신뢰도 신호와 외부 평가 간의 상관 관계를 체계적으로 분석합니다.

- **Performance Highlights**: 연구는 초기 구문 생성 단계에서의 신뢰도가 가장 유의미한 추론 품질 예측 자표임을 밝혔습니다. 이로 인해 다중 에이전트 LLM 시스템의 추론 신뢰성을 추정하는 데 있어 초기 디코딩 동태가 효과적이라는 결론을 내립니다. 또한 지원적 추론과 적대적 비판 간의 신뢰도와 품질 간 차이점도 관찰되었습니다.



### MIRAGE: A Polarity-Flipping Encoding Subspace in LLM Agents (https://arxiv.org/abs/2606.10304)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 에이전트가 민감한 데이터를 은닉된 방식으로 인코딩하는 과정에서 발생하는 출력물을 검출하는 새로운 방법을 제안합니다. 저자들은 Base64, ROT13 등 다양한 인코딩 방식을 포함하여 여러 모델을 비교 분석하였으며, 특히 인코딩된 데이터의 실제 계산 과정을 탐지하는 모델을 개발했습니다. 이 연구는 모델의 출력이 아닌 내부 계산에서 신호를 감지할 수 있다는 중요한 통찰을 제공합니다.

- **Technical Details**: 연구팀은 9가지의 다양한 인코딩 방식과 5가지 아키텍처에서 모델을 실험했으며, 학습한 로지스틱 회귀 프로브를 통해 인코딩된 데이터의 숨겨진 신호를 재구성하며, 이는 단순한 표면 특성이 아니라 계산 과정에 존재함을 확인하였습니다. 특히, 모델이 인코딩을 직접 수행할 때와 도구 호출을 통해 수행할 때의 전환을 식별할 수 있는 메커니즘적 신호가 발견되어, 두 가지 실행 전략을 구별할 수 있음을 보여줍니다. 이 연구에서는 MIRAGE라는 실시간 모니터링 시스템을 구축하여 이러한 신호를 활용했습니다.

- **Performance Highlights**: MIRAGE는 126개의 에이전트 획득 시나리오에 대한 평가에서 0.918의 AUC(Area Under the Curve)를 기록하여 기존의 출력 기반 검출 방법인 0.518을 크게 초과했습니다. 연구 결과, 모델의 구조적 기하에 따라 false-positive율이 상이하게 나타나며, 이는 감시 모델의 성능이 단순히 인코딩 과제가 아닌 특정 모델 구조의 특성에 달려 있음을 시사합니다. 이를 통해 تش 이번 연구는 모델의 내부 메커니즘을 사용하여 에이전트의 은닉 데이터 전송을 실시간으로 감지할 수 있는 가능성을 제시합니다.



### Where You Inject Diversity Matters: A Unified Framework for Diverse Generation (https://arxiv.org/abs/2606.10302)
- **What's New**: 이 논문에서는 Open-ended generation (오픈 엔디드 생성) 작업에서 의미 있게 다양한 출력을 요구하는 문제를 다룹니다. 대형 언어 모델(LLM)은 유사한 출력을 생산하는 경향이 있어 실용적인 가치가 제한됩니다. 이를 해결하기 위해, 저자들은 새로운 Diversity Injection Framework(다양성 주입 프레임워크)를 제안하며, 이는 다양한 최종 출력을 생성하기 위해 다룰 수 있는 주입 소스의 유형에 기반합니다.

- **Technical Details**: 제안된 프레임워크는 생성 과정을 명확하게 설계하며, 입력 프롬프트(입력 텍스트)에 대한 여러 출력을 조건으로 하는 다양한 소스의 주입 형태를 정의합니다. 이 연구에서는 세 가지 레벨(주입 없음, 표면 수준 주입, 사양 수준 주입)으로 메소드를 구분합니다. 특히, 사양 수준 주입(Level 2)은 매출의 질을 유지하면서 최종 출력의 다양성을 향상시키는데 유용한 방법으로 제안됩니다.

- **Performance Highlights**: 다섯 개의 오픈 엔디드 작업과 네 개의 백본 모델을 통해, 사양 수준 주입이 테스트 시간 기반보다 출력 다양성을 향상시킨다는 것을 입증하였습니다. 이러한 발견을 통해, 다양한 소스와 그 전송이 출력에 도달하는 데 중요한 요소라는 것을 강조하였으며, 이는 더 다양한 생성 시스템 구축을 위한 핵심 요소로 작용할 수 있음을 보여줍니다.



### The Confident Liar: Diagnosing Multi-Agent Debate with Log-Probabilities and LLM-as-Judg (https://arxiv.org/abs/2606.10296)
Comments:
          15 pages, 7 figures, 1 table, ACL proceedings

- **What's New**: 본 논문에서는 다중 에이전트 논쟁 시스템의 중간 추론 품질을 평가하는 새로운 접근 방식을 제안합니다. 기존의 연구들은 최종 답변의 정확도만을 평가하는 경향이 있었으나, 이 연구는 중간 추론의 질이 최종 결과에 미치는 영향을 분석합니다. 특히, 두 에이전트가 논쟁하는 구조와 LLM을 판사로 활용하여 각 에이전트의 추론을 평가하는 프레임워크를 구축하였습니다. 이를 통해 중간 단계에서의 신뢰도 신호가 외부 평가와 얼마나 연관되는지 분석하기 위한 체계적인 연구를 진행합니다.

- **Technical Details**: 논문은 추론 토큰들에 대한 log-probability 분포, LLM으로부터의 루브릭 점수 및 최종 작업 정확도라는 세 가지 신호 간의 관계를 탐구합니다. 특히, 두 가지 에이전트(Constructor와 Auditor)가 논쟁하는 구조에서 신뢰도 신호가 어떻게 진화하는지 살펴봅니다. 또한, 각 에이전트의 신뢰도는 LLM이 평가한 추론 품질과 어느 정도 일치하는지를 분석하는 중요한 질문을 다룹니다. 이를 통해 내부 신뢰도 신호와 외부 평가 간의 상관 관계를 조사합니다.

- **Performance Highlights**: 실험 결과, Constructor 에이전트의 논리적 추론 품질은 Auditor에 비해 두 배 정도 더 잘 평가되는 경향을 보였습니다. 특히, Constructor의 신뢰도 기반 비판 추론 실패 탐지는 Auditor보다 더 높은 신뢰성을 보였으며, AUROC 점수는 각각 0.804와 0.634로 나타났습니다. 이러한 결과는 다중 에이전트 논쟁 시스템의 설계를 개선할 수 있는 기초 자료로 활용될 수 있으며, 각 작업과 논쟁 구성에 따라 내부 신뢰도 신호와 외부 평가 결과 간의 차이를 진단하는 데 도움을 줄 수 있습니다.



### OpenRTLSet: A Fully Open-Source Dataset for Large Language Model-based Verilog Module Design (https://arxiv.org/abs/2606.10285)
Comments:
          Accepted by ICLAD'25

- **What's New**: OpenRTLSet는 하드웨어 디자인을 위한 가장 큰 공개 소스 데이터셋으로, 131,000개의 다양한 Verilog 코드 샘플을 제공합니다. 이 데이터셋은 GitHub 저장소(102k 모듈), VHDL 변환(5k 모듈), 그리고 합성 가능한 C/C++ 변환(24k 모듈)을 독점적인 제한 없이 자유롭게 결합하여 제공합니다. 또한 DeepSeek-R1 모델을 사용하여 각 코드 샘플에 대한 자연어 설명을 생성하여 다양한 언어 모델을 fine-tuning할 수 있게 합니다.

- **Technical Details**: OpenRTLSet는 데이터 라벨링을 위한 전방위적인 오픈 소스 흐름을 개발하여, Verilator로 생성된 C++ 파일을 Verilog 모듈과 자연어 라벨링 시 결합하는 등의 방법을 통합합니다. 데이터 수집, 파싱, 라벨링 등 모든 과정을 포괄하는 철저한 시스템 접근 방식을 통해 Verilog 모듈에 대한 고품질 데이터셋을 구축합니다. 이 데이터셋은 다양한 Coding LLM을 Verilog 전문가로 변환할 수 있도록 fine-tuning됩니다.

- **Performance Highlights**: OpenRTLSet는 Verilog 코드 생성에서 여러 LLM의 성능을 정량적으로 평가하며, 모델 크기 (7B-32B)간 성능 차이를 탐구합니다. MG-Verilog 데이터셋을 기준으로OpenRTLSet를 적용한 결과 Pass@1에서 5.7%, Pass@5에서 8.3%, Pass@10에서 7.9%의 절대 향상이 나타났습니다. VerilogEval-Machine 및 VerilogEval-Human 벤치마크에서 일관되게 높은 성과를 보이며, OpenRTLSet가 산업 및 연구에서 사용되는 새로운 기준임을 입증합니다.



### Gaming AI-Assisted Peer Reviews Poses New Risks to the Scientific Community (https://arxiv.org/abs/2606.10159)
- **What's New**: 이 논문은 AI-mediated peer review(인공지능 기반 동료 검토)가 표면적인 개념의 재구성을 통해 저자들에 의해 쉽게 조작될 수 있다는 점을 밝혀냈습니다. 저자들은 단순한 추상 문구의 재구성을 통해 AI 리뷰의 결과를 상당히 개선시킬 수 있다는 것을 증명하였으며, 이러한 조작이 다양한 학문 분야와 출판 기관에서 적용될 수 있음을 보여주었습니다. 이로 인해 AI 평가 시스템의 약점이 드러났고, 저자들이 과학적 성과보다 AI 판별에 최적화된 원고를 작성하도록 유도할 수 있는 위험이 제기되었습니다.

- **Technical Details**: 이 연구에서는 AI 리뷰어의 평가 점수를 극대화하려는 공격이론을 제시했습니다. 저자들은 원고의 추상 문구를 재구성함으로써 AI 리뷰 평가를 부풀리는 반복 최적화 공격을 도입하고, Rewriting, Meaning-Preserving, Overclaiming 세 가지 전략을 사용하여 이를 입증했습니다. 이 기법은 인간 리뷰어의 평가 점수에 부정적인 영향을 미치지 않도록 특정 조건 하에 의미의 변화는 허용하면서도 과학적인 문서로서의 유창함을 유지하도록 설정되었습니다.

- **Performance Highlights**: 연구 결과, AI 리뷰어는 원고의 추상 문구에 대한 외부적 재구성에 매우 취약한 것으로 나타났습니다. 가장 강력한 공격은 약 38%의 성공률을 기록했으며, Gemini 3 Flash 리뷰어의 경우 수용 등급이 +1.31, GPT 5.4 Mini 리뷰어의 경우 +0.88로 높아졌습니다. 원래의 AI 리뷰에서 '거부'를 제안할 경우, 성공률은 50%를 넘겼습니다. 이러한 발견은 학술 커뮤니케이션의 미래 거버넌스에 대한 심각한 우려를 불러일으키며, AI 도구가 고위험 결정에서 신뢰할 수 있는 평가자로 간주되기 위해서는 체계적인 강건성 테스트와 투명한 보호 장치가 필요하다는 것을 나타냅니다.



### Pareto-Guided Teacher Alignment for Fair Personalized Text Generation (https://arxiv.org/abs/2606.10126)
- **What's New**: 본 논문은 개인화된 설득 텍스트 생성(Personalized persuasive text generation)의 공정성을 개선하기 위한 새로운 연구 방향을 제시합니다. 인구 통계 기반의 불균형 문제를 해결하기 위해, 저자들은 다목적 정렬 문제(constrained multi-objective alignment problem)를 설정하였습니다. 이 연구에서는 개인화를 유지하면서 인구 통계적 차이를 줄일 수 있는 방법을 모색하였습니다.

- **Technical Details**: 제안된 프레임워크는 비선형 생성(candidate generation), 쌍 인식 가능성 차단(pair-aware feasibility gating), 그리고 파레토 방식(Pareto-style) 선택을 포함합니다. 또한 감독된 미세 조정(supervised fine-tuning)과 직접적인 선호 최적화(direct preference optimization)를 통해 개인화 개선을 위한 선택적 선호 최적화가 가능합니다. 연구는 기후 변화 및 백신 설득 과제를 사용하여, 성별과 연령이 일치하는 인구 통계적 그리드를 사용하여 평가하였습니다.

- **Performance Highlights**: 결과는 다양한 공정성 유지 전략이 모든 목표를 동시에 최적화하지 못함을 보여줍니다. 각 방법은 공정성과 개인화의 파레토 경계(Pareto frontier)에서 서로 다른 영역을 차지하며, 일부 방법은 불균형 감소를 더 효과적으로 달성하면서 다른 방법은 개인화 또는 인구 통계적 안정성을 더 잘 유지합니다. 연구 결과는 공정성을 감소시키기 위해 다목적 모델 선택(multi-audit model selection)의 중요성을 강조합니다.



### Emotion Profiling in LLM-Based Literary Translation: Systematic Shifts Across MT and Post-Editing (https://arxiv.org/abs/2606.10113)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 번역이 식별 가능한 감정 프로필을 나타내는지 여부를 조사하며, 후속 편집(post-editing)이 인간과 유사한 기준으로 이러한 감정 프로필을 어떻게 변화시키는지를 분석합니다. 마가렛 애트우드의 "Oryx and Crake" 소설의 LLM 번역과 그 후속 편집 버전 및 인간 번역을 대조하여 대규모 이탈리아 현대 공상과학 코퍼스를 사용하였습니다. 감정 분석은 어휘 기반 및 다국어 모델링 방식을 통해 수행했으며, 번역 시스템 간의 감정 변동을 세밀하게 분석하였습니다.

- **Technical Details**: 이 연구는 LLM과 NMT(신경 기계 번역) 시스템을 사용하여 감정 프로필을 정량적으로 분석하는 데 중점을 둡니다. 번역 품질을 평가하기 위해 전문적인 번역과 다양한 LLM에서 생성된 전후편집된 번역을 비교합니다. 연구에 사용된 두 가지 주요 어휘 자원은 이탈리아 감정 어휘 사전(Italian Emotive Lexicon)과 NRC 감정 어휘 사전(NRC Emotion Lexicon)으로 각각의 감정에 대한 단어 목록을 제공합니다. 이러한 방법으로 우리는 소스 텍스트와 번역된 텍스트 간의 감정 이동을 조사합니다.

- **Performance Highlights**: 연구 결과, MT 시스템이 고유한 감정 핑거프린트를 갖고 있으며, 감정 콘텐츠의 분포 및 강도의 차이로 나타납니다. 특히 후속 편집의 효과는 기반 MT 모델의 품질에 따라 크게 제한된다는 사실을 발견했습니다. 감정의 특이한 패턴은 통계적으로 유의미하며, 이는 출처 텍스트에서 저자의 정서적 목소리를 가장 충실하게 보존하고자 하는 지점입니다.



### CodeAlchemy: Synthetic Code Rewriting at Sca (https://arxiv.org/abs/2606.10087)
- **What's New**: 새롭게 발표된 CodeAlchemy는 공개 소스 코드를 의미적으로 풍부한 훈련 데이터로 변환하는 합성 데이터 생성 프레임워크입니다. 이 프레임워크는 코드 품질 개선, 템플릿 기반 문제, 개발자 작업, 다회 대화, 실행 추적 등 다섯 가지 전략을 통해 15개 언어에서 500억 이상의 토큰을 생성합니다. 이는 이전의 노력보다 수십 배 많은 양으로, 개발자 상호작용을 위한 새로운 벤치마크인 DevEval과 TraceEval을 통해 코드 이해의 격차를 드러내고 있습니다.

- **Technical Details**: CodeAlchemy는 15개 언어에서 3개 코드 코퍼스를 사용하여 데이터 변환을 수행합니다. 이 시스템은 LLM을 사용하여 품질이 낮은 코드를 점수화한 후, 단위 테스트, 문서화 및 의존성 대체물 추가 등으로 재작성하여 코드 품질을 향상시킵니다. 또한 실행 추적을 통해 모델이 코드의 흐름과 상태 변화를 이해할 수 있도록 1.3M개의 코드-추적 쌍을 생성하였습니다.

- **Performance Highlights**: CodeAlchemy의 3B 모델들은 HumanEval에서 83.5%, MBPP에서 63.2%, DevEval에서 8.09%의 승률을 기록하며, TraceEval에서 15.36 ROUGE-2 점수를 달성하였습니다. 이 결과는 27B Gemma-3와 32B Granite-4.0 같은 대형 모델에 비해 10배 이상 뛰어난 성과를 보여주고 있습니다. 다음 토큰 예측이 아닌 의미론적 관점에서의 모델 평가가 중요함을 강조하며, 기존 모델들이 TraceEval에서 5.6%의 정확도로 한계를 보였습니다.



### BenSyc: Benchmarking Conversational Sycophancy and Human Alignment in LLMs for Bengali Contexts (https://arxiv.org/abs/2606.10061)
- **What's New**: 본 논문은 벵골 사회맥락에서의 대화적 아첨(conversational sycophancy)을 연구하기 위한 첫 번째 벤치마크인 BenSyc를 소개합니다. 기존의 연구들은 사실적 동의(factual agreement)에 중점을 두었으나, BenSyc는 11,840개의 Reddit 게시물과 170,000개의 댓글을 기반으로 다양한 대화적 반응을 체계적으로 분류하고 평가합니다. 또한 이것은 감정적으로 민감한 대화에서 LLM(대형 언어 모델)의 성격과 반응을 분석하는 데 중점을 둡니다.

- **Technical Details**: BenSyc는 1,078개의 인간 검증된 Reddit 게시물-댓글 쌍을 포함하고 있으며, 각 쌍은 이진 레이블(binary labels)과 다섯 수준의 대화적 분류(taxonomy)로 주석이 달려 있습니다. 이 다섯 단계는 무효화(Invalidation), 중립(Neutral), 지지(Support), 검증(Validation), 에스컬레이션(Escalation)으로 구성됩니다. 다양한 LLM들을 대화 정렬(classification) 및 반응 생성(generation) 작업에서 평가하며, 평가 도구로 GPT-5.5 기반의 판단기(judge)를 사용합니다.

- **Performance Highlights**: 실험 결과, LLM들은 지지적 공감과 검증, 에스컬레이션을 구분하는 데 어려움을 겪는 것으로 나타났습니다. 최상의 모델조차도 이진 탐지에서 61.8 Macro-F1, 다섯 클래스 분류에서 61.7 Macro-F1의 성과를 기록했습니다. 이러한 결과는 대화적 아첨이 문화적으로 위치 지은 정렬 행동임을 강조하며, 비서구(Non-Western) 대화 문맥 평가를 위한 벤치마크의 필요성을 부각시킵니다.



### Less Context, More Accuracy: A Bi-Temporal Memory Engine for LLM Agents Where a Lean Retrieved Context Beats the Full History (https://arxiv.org/abs/2606.09900)
Comments:
          14 pages, 4 figures, 3 tables. Code, reproducible harness, and raw per-question logs: this https URL

- **What's New**: Engram은 LLM(대형 언어 모델) 에이전트의 장기 기억을 구현하기 위한 새로운 오픈 소스 메모리 엔진입니다. 기존의 메모리 시스템들은 비용이나 지연(latency) 측면에선 효과적일지 모르나, 정확도에서의 단점을 극복하지 못했습니다. 본 시스템은 ‘bi-temporal(이중 시간)’ 데이터 모델을 활용하여 비파괴적(conflict resolution)인 갈등 해소 방법을 제공하며, 기존 이론과는 다른 접근 방식을 통해 기계 학습 모델의 정확도를 향상시킬 수 있습니다.

- **Technical Details**: Engram은 빠른 쓰기 경로와 느린 비동기적 통합 경로를 사용한 이중 프로세스 메모리 시스템입니다. 이 시스템은 ‘episode’와 ‘fact’라는 데이터 구조를 통해 정보의 수집 및 저장을 최적화합니다. 데이터는 통합 속성(provenance)을 유지하면서 변경 사항을 추적하고 관리하고, 명확한 시간축(valid/invalid) 정보를 제공합니다. 또한 하이브리드 읽기 경로가 고밀도 및 최근성 정보를 융합하여 더욱 정확한 데이터 검색을 제공합니다.

- **Performance Highlights**: Engram은 LongMemEval_S에서 500개의 질문에 대해 83.6%의 정확도를 기록하며, 전통적인 전체 텍스트 컨텍스트(full-context) 방식에서의 73.2%보다 +10.4 포인트 향상된 성과를 보였습니다. 이는 약 8배 적은 토큰 사용으로 달성된 결과입니다. 시스템의 설계 원칙 중 하나는, 재현할 수 없는 숫자는 존재하지 않는다는 것으로, 이를 통해 사용자들은 결과의 신뢰성을 확립할 수 있게 됩니다.



### Using Probabilistic Programs to Train Inductive Reasoning in Large Language Models (https://arxiv.org/abs/2606.09856)
Comments:
          20 pages, 5 figures

- **What's New**: 이 논문에서는 Post-training Large Language Models (LLMs)를 위한 새로운 접근 방식인 Program-based Posterior Training (PPT)를 소개하고 있습니다. PPT는 다양한 열린 세계 상황을 생성하고, 확률적 프로그램을 통해 쿼리에 대한 응답을 산출하여, 이러한 확률적 소프트 레이블을 사용하여 LLM을 미세 조정(fine-tuning)합니다. 이러한 방법은 기존의 방법들로는 해결하기 어려운 불확실성을 포함한 추론 과제를 다루는 데 매우 효과적입니다.

- **Technical Details**: 이 연구에서는 10,000개의 프로그래밍적으로 생성된 시나리오를 기반으로 LLM을 미세 조정하며, MSA(Model Synthesis Architecture)를 영감을 받아 순차적인 프롬프트 절차를 통해 데이터를 생성합니다. 그런 다음 확률적 추론(probabilistic inference)을 수행하여 관련 변수에 대한 사후 분포(posterior distribution)를 계산하고, 이 분포를 미세 조정의 목표로 사용합니다. 이를 통해 고급 자연어 문제의 잠재적 구조와 불확실성을 명시적으로 표현할 수 있습니다.

- **Performance Highlights**: PPT를 통해 LLM의 추정 정확도가 크게 향상되었으며, 인간의 판단에 대한 일치도가 증가하고, 다양한 외부 벤치마크에서도 향상된 성능을 보여주었습니다. 또한, 모델이 출력 리스케일(output rescaling)보다 훨씬 더 깊이 있는 불확실성을 내재화하고 있다는 것을 보여주며, 이 방법이 근본적으로 신뢰할 수 있는 근사적 유도 추론을 수행하는데 효과적임을 시사합니다.



### Can Multi-Agent LLMs Identify Their Peers? Stylometric Fingerprinting in Role-Constrained Political Analysis (https://arxiv.org/abs/2606.09854)
Comments:
          24 pages, 3 figures

- **What's New**: 이 논문은 정치 성명 분석을 위한 다중 에이전트 대형 언어 모델(LLM) 파이프라인에서 발견된 동료 보존 편향(peer-preservation bias)에 대한 첫 번째 체계적인 조사를 진행합니다. 특히, 콘셉트 수준의 익명화가 효과적인 완화책으로 제안되었지만, 이전 연구에서 스타일 기반 지문(stylometric fingerprints)이 익명화 후에도 유지된다는 사실이 확인되었습니다. 이러한 관찰은 정치 분석 텍스트의 LLM 모델 가족 식별 가능성을 재검토하게 하며, 유럽연합 AI 법(EU AI Act)에 대한 직접적인 영향을 미칩니다.

- **Technical Details**: 이 연구는 신뢰(TRUST) 민주적 담론 분석 파이프라인을 사용하여, 세 가지 분류기 접근법을 평가합니다. LLM 제로샷(Classifier Zero-shot) 및 몇 샷(few-shot) 분류와 조정된 T5-base 기반 모델을 통해 다섯 가지 클래스에 대한 귀속(attribution) 작업을 수행하였습니다. 특히, 본 연구에서는 내용 중복이 없는 진술 비접합 교차 검증 프로토콜(sd-CV)을 도입하여 훈련과 평가 데이터 간의 유효성을 보장했습니다.

- **Performance Highlights**: T5 모델은 SD-CV 하에서 매크로 F1 = 0.991 (+-0.008)을 달성하고, 완전히 제외된 24개의 진술에서는 F1 = 0.978의 성능을 기록했습니다. 이는 RD-CV에 비해 2.1배 더 많은 훈련-테스트 내용 거리에서도 견고한 성능을 보여주며, 물리적 스타일의 일반화(stylometric generalization)를 증명합니다. 또한, 훈련 데이터 분석 결과는 실질적인 배치에 필요한 데이터 임계선이 40%임을 확인했습니다.



### Automated Scoring of Arabic Text Using Large Language Models: A Literature Review (https://arxiv.org/abs/2606.09830)
Comments:
          Accepted at NCMAI 2026

- **What's New**: 이 논문에서는 현대 교육 시스템에서 중요한 역할을 하는 Automatic Text Scoring (ATS)의 발전을 다루고 있으며, 특히 아랍어 텍스트의 자동 평가에 LLMs (Large Language Models)을 활용하는 방법에 주목하고 있습니다. 또한, ATS 접근 방식에 대한 구조적 분류 체계를 소개하여, 기존 연구의 방법론, 데이터세트, 평가 기준 및 성과를 비교 분석합니다. 이러한 연구는 아랍어를 사용하는 지역사회의 교육 품질 향상에 기여할 수 있는 방향성을 제시합니다.

- **Technical Details**: 자동 평가는 자연어 처리(NLP) 기술을 사용하여 작성된 답변을 자동으로 채점하는 시스템으로, 전통적인 통계 모델에서 딥 러닝을 기반으로 한 모델로 발전했습니다. LLMs를 사용한 최근의 접근법들은 단순한 점수 생성에 그치지 않고, 제시된 답변에 대한 의미 있는 피드백을 제공할 수 있는 능력을 갖추고 있습니다. 하지만 아랍어의 복잡한 형태학적 특성과 방언으로 인해, 현재 연구는 여러 난점에 직면해 있습니다.

- **Performance Highlights**: ATS 연구는 뛰어난 성과를 보여왔지만, 아랍어 경우 데이터 세트의 부족과 다양한 평가 기준의 차이로 인해 체계적인 비교가 어려운 상황입니다. 연구 결과, 아랍어에 대한 ATS 시스템의 강화와 신뢰성을 높이기 위한 더 많은 연구 노력이 필요하다는 점이 강조됩니다. 본 논문은 아랍어 ATS의 기존 연구에서 확인된 주요 한계를 도출하고, 향후 연구 방향을 제시하여 더 강력하고 표준화된 ATS 시스템 개발로 나아가는 길을 모색합니다.



### A Unifying Lens on Supervised Fine-Tuning Through Target Distribution Design (https://arxiv.org/abs/2606.11189)
- **What's New**: 이 논문에서는 Supervised Fine-Tuning (SFT)의 기본 개념을 재해석하여, 토큰 수준의 목표 (target) 분포를 설계하는 관점에서 설명합니다. 기존의 SFT 방식의 비효율성을 극복하기 위해 Q-target 프레임워크를 도입하여, 관찰된 토큰에 대한 의존성과 남은 확률 질량의 분배 방식을 명시적으로 정의합니다. 이를 통해 기존의 다양한 SFT 변형을 통일적으로 이해하고, 더 나은 SFT 목표를 구성할 수 있는 방법들을 제시합니다.

- **Technical Details**: 제안된 Q-target 프레임워크는 두 가지 주요 선택지를 포함합니다: (1) 관찰된 토큰에 얼마나 의존할 것인가, (2) 관찰된 토큰이 불확실할 때 남은 확률 질량을 어떻게 분배할 것인가 입니다. Target-SFT는 이러한 Q-target 관점을 기반으로 하여 훈련 목표를 직접 구성하며, 10개의 데이터셋-모델 세팅에서 일관되게 성능을 향상시킵니다. 이 프레임워크는 SFT에서의 손실 함수보다 목표 분포가 더 근본적인 객체임을 설명합니다.

- **Performance Highlights**: Target-SFT는 기존의 SFT 방식보다 더 적합한 정보 전이를 제공하며, 데이터셋 모사(dataset imitation), 사전 보존(prior preservation) 및 대안 감독(alternative supervision) 간의 균형을 유지할 수 있는 새로운 설계 공간을 열었습니다. 제안된 방법은 다양한 데이터셋 모델 설정에서 반복적으로 우수한 성능을 발휘하며, 이로 인해 SFT 훈련의 기초 원리에 대한 더 깊은 통찰을 제공합니다. Target-SFT의 성능은 실험적으로 확립되어 있으며, 다양한 새로운 SFT 목표를 탐색할 수 있는 가능성을 시사합니다.



### Data Journalist Agent: Transforming Data into Verifiable Multimodal Stories (https://arxiv.org/abs/2606.11176)
Comments:
          Project page: this https URL Github: this https URL

- **What's New**: 데이터 저널리즘의 새로운 접근 방식인 Data Journalist Agent (Data2Story)가 소개되었습니다. 이 다중 에이전트 프레임워크는 가상의 뉴스룸에서 전문 역할들을 하나로 통합하여 신뢰할 수 있는 스토리를 작성합니다. Data2Story는 데이터 기반의 주장을 보증하고, 다양한 형식으로 기사를 생성하는 두 가지 혁신을 제공합니다.

- **Technical Details**: Data2Story는 데이터 소스를 입력으로 받아 생성적인 멀티미디어 기사를 출력하며, 이를 위해 7개의 전문 역할(탐정, 분석가, 편집자, 디자이너, 프로그래머, 감시자, 검사기)을 조화롭게 운영합니다. 데이터와 그 근원에 대한 검증 가능성을 높이기 위해, 이 프레임워크는 대부분의 기사 요소를 실행 가능한 코드나 검증된 소스 URL에 연결합니다.

- **Performance Highlights**: Data2Story는 독립적으로 검증 가능한 멀티모달 기사를 생성하며, 주장 수준에서의 증거 추적 기능을 제공합니다. 인간 평가자들은 여러 품질 기준에서 Data2Story의 결과물을 긍정적으로 평가했으나, 여전히 기사의 편집적 관점, 창의적 디자인 및 정보 전달에서 인간 기자가 우위를 점하고 있음을 시사합니다.



### TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning (https://arxiv.org/abs/2606.11119)
Comments:
          32 pages, 12 figures, 6 tables

- **What's New**: 본 논문에서는 다중 턴 에이전틱 RL(multi-turn agentic RL) 을 위한 새로운 예산 할당 방식인 TRACE(Tree Rollout Allocation for Contrastive Exploration)를 제안합니다. TRACE는 롤아웃 예산을 프롬프트 루트(prompt roots)와 턴 레벨 접두사(turn-level prefixes) 모두에 할당하여 보상 대비를 극대화합니다. 이 구조는 기존의 단일 턴 단위의 접근 방식을 넘어 여러 단계로 나누어진 복잡한 결정을 지원합니다.

- **Technical Details**: TRACE는 각 ReAct 스타일의 사고-행동-관찰(turn) 을 의미적으로 구분된 노드로 모델링하여 세부적인 예산 할당을 가능하게 합니다. 롤아웃 예산은 성공적인 보상과 실패한 보상이 모두 포함될 가능성이 높은 앵커에 우선 배정됩니다. 이는 노드의 자손 세트가 혼합 보상을 생성할 가능성을 기준으로 할당됩니다.

- **Performance Highlights**: TRACE는 Qwen3-14B Multi-Hop QA에서 기존 최상위 기준선 대비 2.8포인트 높은 평균 정확도를 달성하며 성공적인 성능을 입증했습니다. 동일한 샘플링 비용 하에서 TRACE를 통한 적응형 트리 구조는 결과-단일 피드백을 풍부하게 하고 정책 업데이트 신호를 강화합니다. Empirically, TRACE는 수학적 추론, 다중 턴 QA 및 함수 호출과 같은 다양한 에이전틱 벤치마크에서 우수한 성과를 보였습니다.



### A History-Aware Visually Grounded Critic for Computer Use Agents (https://arxiv.org/abs/2606.11078)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 HiViG라는 새로운 테스트 시간 개입 프레임워크를 소개합니다. HiViG는 과거 상호작용을 요약하여 자연어로 피드백을 제공하는 멀티모달 크리틱 모델에 기반하여, 그래픽 사용자 인터페이스(GUI) 환경에서의 결정 오류를 사전에 차단합니다. 이를 통해, 기존 모델들이 가지던 시각적 기초 부족 및 단기적인 결정 루프 문제를 해결합니다.

- **Technical Details**: HiViG는 역사(state) 추적 및 시각적으로 구체화된 오류 분석 기능을 결합하여 효과적으로 동작합니다. 이 프레임워크는 모집단 패턴과 시각적 상태를 비교하여 실수 행동을 식별하고, 과거 상호작용을 매크로 액션 기록으로 압축하여 정책의 목표 달성을 지원합니다. HiViG-critic 모델은 오픈 소스의 멀티 도메인 GUI 경로에서 구축된 훈련 데이터를 사용하여 학습됩니다.

- **Performance Highlights**: HiViG는 웹, 모바일 및 데스크탑 벤치마크에서 평균 성공률을 기존 강력한 기준 모델보다 각각 5.8% 및 9.0% 향상시키는 성과를 보였습니다. 특히, HiViG는 Gemini-3-Flash의 WebArenaLitev2 기준에서 성공률을 15.0%에서 30.5%로 개선하며 뛰어난 차세대 성능을 입증했습니다. 또한, HiViG는 다양한 GUI 환경에서 강력한 일반화 능력을 보여주며, 장기 GUI 작업을 보다 효과적으로 완수할 수 있도록 지원합니다.



### AuRA: Internalizing Audio Understanding into LLMs as LoRA (https://arxiv.org/abs/2606.11033)
- **What's New**: 본 연구에서는 AuRA (Audio Understanding as LoRA)라는 새로운 방법론을 제안합니다. AuRA는 모든 입력 음성을 ASR 인코더(교사)와 LoRA로 조정된 LLM(학생)으로 동시에 처리하여 음성 인식을 LLM 내부로 통합하는 기법입니다. 이를 통해 기존의 복잡한 ASR-LLM 파이프라인과 대규모 다중 모드 훈련 없이도 효율적인 음성-언어 모델링을 달성할 수 있습니다.

- **Technical Details**: AuRA는 ASR 인코더를 교사로, LoRA로 조정된 LLM을 학생으로 설정하여 레이어 별 증류(layer-wise distillation)을 통해 음성 표현을 LLM에 직접 내재화합니다. 이 방법은 오디오 정보가 LLM의 초기 변환에 직접적으로 들어가도록 하여 음성-언어 공동 모델링을 가능하게 합니다. 인퍼런스 시에는 ASR 인코더가 제거되며, LLM은 교사로부터 증류된 음성 표현을 활용합니다.

- **Performance Highlights**: AuRA는 여러 음성-언어 벤치마크에서 기존의 ASR-LLM 파이프라인, 음성-LLM 적응 기준선, 대규모 음성-언어 모델에 비해 효과성과 효율성 모두에서 뛰어난 성능을 나타냅니다. 또한, 이 방법은 인퍼런스 레이턴시와 메모리 사용량을 줄이면서도 성능을 개선할 수 있는 잠재력을 보여주고 있습니다.



### Generative Archetype-Grounded Item Representations for Sequential Recommendation (https://arxiv.org/abs/2606.11023)
Comments:
          Accepted by WWW 2026 (Oral)

- **What's New**: 이번 논문에서는 Generative Archetype-grounded Item Representations를 활용한 GenAIR라는 새로운 프레임워크를 제안합니다. GenAIR는 대규모 언어 모델(LLM)을 사용하여 아이템 메타데이터를 분석하고 이상적인 타겟 오디언스의 개념적 프로필을 반영한 아키타입(Archetype)을 추론합니다. 이러한 접근 방식은 아이템의 정체성을 정의하는 데 있어서 타겟 오디언스의 중요성을 강조합니다.

- **Technical Details**: GenAIR는 LLM을 통해 생성한 아키타입 임베딩을 활용하여 아이템의 행동 특성을 실증적으로 반영하도록 설계되었습니다. 이를 위해 실제 사용자 상호작용에서 얻은 행동 신호를 포함하는 새로운 훈련 목표를 도입하여 임베딩 공간의 구조를 조정합니다. 이러한 방법은 기존의 모델과의 통합을 용이하게 하면서도 높은 효율성을 유지합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에 대한 종합적인 실험 결과, GenAIR는 다양한 순차 추천 모델의 성능을 획기적으로 향상시키고 최첨단 기준 방법보다 일관되게 뛰어난 성과를 보였습니다. 이러한 연구 결과는 GenAIR가 실질적인 추천 시스템에서의 적용 가능성과 효과성을 입증했다는 것을 강조합니다.



### Mind the Gap: Can Frontier LLMs Pass a Standardized Office Proficiency Exam? (https://arxiv.org/abs/2606.10956)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM) 에이전트의 오피스 자동화에서의 성능을 평가하기 위한 새로운 벤치마크인 OfficeEval을 소개합니다. 이를 통해 Word, Excel, PowerPoint를 포함하는 실제 작업 환경에서의 문서 자동화 능력을 정량화하려고 합니다. 이 연구는 중국의 국가 컴퓨터 등급 시험(NCRE)를 기반으로 하며, 200개의 포괄적인 실무 작업을 통해 LLM의 성능을 평가합니다.

- **Technical Details**: OfficeEval은 NCRE의 실제 작업 구성 요소로, MS Office Level 1 및 Level 2 모듈에서 파생된 것입니다. 이 체계는 7,118개의 기계 채점 가능한 기준을 사용하여 각 작업을 100점 스케일로 평가합니다. 전체 시험에서 수집된 데이터를 기반으로 특정 LLM의 성능을 시스템적으로 평가하여, 현재의 코드 생성 LLM이 신뢰할 수 있는 세밀한 오피스 문서 자동화에 도달하기가 여전히 매우 도전적임을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면 단일 턴 모델의 최고 점수는 36.6%에 불과하며, 강력한 시스템에서는 68.8%를 기록하였지만 이는 여전히 95.5%의 커뮤니티 기준 점수에 미치지 못합니다. 또한 성능이 극도로 분포되어 있고, 일부 모델은 2.8%의 Score Rate(SR)까지 떨어지며, 이는 최신 API 상태가 항상 좋은 성과를 보장하지 않음을 시사합니다. 현재의 LLM이 코드의 실행을 성공적으로 수행하더라도 오피스 특정 작업의 정확한 수행에는 어려움을 겪고 있음을 강조합니다.



### Training LLMs to Enforce Multi-Level Instruction Hierarchies via Gravity-Weighted Direct Preference Optimization (https://arxiv.org/abs/2606.10860)
- **What's New**: 이 논문은 생산 대형 언어 모델(LLM)에서의 지침 계층 문제를 다루어, k=5 레벨의 계층을 정식화 하였습니다. 새로운 Gravity-Weighted DPO (GW-DPO) 방법론은 지침 정보 간의 거리에 기반하여 우선 순위를 최적화하도록 설계 되었습니다. 이 연구는 파라미터 조정을 통해 모델의 반응을 향상시키고, 델리미터 토큰 및 Instructional Segment Embeddings (ISE)를 결합하여 성능 개선을 보여줍니다.

- **Technical Details**: LLM은 사용자로부터 수신한 다양한 신뢰 수준을 지닌 지침을 처리하는데, 동일한 아키텍처 권한을 가진 채 모든 토큰에 주의를 기울이는 구조적 취약성을 가지고 있습니다. 이를 해결하기 위해, 이 논문에서는 k=5로 정의된 지침 계층 문제를 정식화하고, 10개의 쌍 우선 순위 관계를 정의했습니다. GW-DPO는 지침 간의 거리 비례로 마진을 조정하며, 두 가지 특정 스케줄을 평가합니다: 선형 및 양측 방식입니다.

- **Performance Highlights**: GW-DPO는 Llama-3.1-8B-Instruct에서 기존 DPO 방식보다 우수한 성능을 보여주며, 매크로 쌍 우선 순위 준수를 향상시키는 동시에 과도한 거절률은 절반으로 감소시킵니다. 또한, ISE가 모델의 거부 기준 조정자로 작동하며, 다섯 개의 레벨 훈련과 세 개의 레벨 훈련 간의 대체 무역관계를 명확히 했습니다.



### K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling (https://arxiv.org/abs/2606.10820)
- **What's New**: 본 논문에서는 고부하 배치 서비스에 대한 기존의 속도 향상 방법들이 해결하지 못한 문제를 다룸으로써, K-Forcing라는 새로운 언어 모델링 패러다임을 제안합니다. K-Forcing는 기존의 AR 모델에서 이어받은 조건적 푸시-포워드 매핑을 통해 동시에 여러개의 다음 k개 토큰을 생성할 수 있습니다. 이를 통해 상대적으로 적은 품질 저하로도 배치 속도를 크게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: K-Forcing는 먼저 사전 훈련된 AR 모델을 통해 푸시-포워드 매핑을 학습하는 프로세스를 가지고 있으며, 이를 통해 고정 길이의 여러 토큰을 한 번의 포워드 패스에서 생성할 수 있습니다. 이 방식은 AR 모델의 기본 구조를 유지하면서도 표준 AR 서비스 인프라와 호환되도록 설계되었습니다. K-Forcing의 주요 기술적 기여는 고정 길이의 출력과 공동 다중 토큰 샘플링을 가능하게 하는 것입니다.

- **Performance Highlights**: K-Forcing는 LM1B 및 OpenWebText 데이터셋에서 k=4 토큰을 포워드 패스 당 생성하며, 다양한 배치 크기에서 약 2.4~3.5배의 속도 향상을 보였습니다. 이 모델은 모드레이트한 품질 저하 속에서도 배치 서비스 처리량을 상당히 개선시키며, 이는 통계적 모델링 변경만으로도 AR 디코딩 대비 굉장한 배치 서비스 처리량 향상을 이끌어냈다는 것을 보여줍니다.



### RedAct: Redacting Agent Capability Traces for Procedural Skill Protection (https://arxiv.org/abs/2606.10813)
- **What's New**: 이번 연구에서는 사용자가 에이전트의 행동을 관찰하고 실패를 진단하며 책임을 보장하기 위해 실행 추적(execution traces)에 의존한다는 점을 강조합니다. 이러한 추적은 도구 호출(tool invocations), 중간 결정(intermediate decisions), 오류 회복 논리(error-recovery logic) 등 풍부한 절차적 세부 정보를 포함하고 있지만, 이는 개인적인 절차적 기술을 공개할 수 있는 위험이 있습니다. 이에 대한 대응으로 	extsc{CapTraceBench}라는 벤치마크를 구성하고, 	extsc{RedAct}라는 프레임워크를 소개합니다.

- **Technical Details**: 	extsc{CapTraceBench}는 7개 도메인에서 75개의 전문 장기 과제와 154개의 정교한 기술들을 포함하고 있습니다. 	extsc{RedAct}는 보호된 키 정보를 지역화(localize)하고, 검증자에게 중요한 증거를 유지하면서 추적을 재작성(rewrite)하는 기능을 갖추고 있습니다. 또한, 하류의 출처 분석(provenance analysis)을 위해 행동 수자원 행동 워터마크를 삽입합니다. 보다 구체적으로, 	extsc{RedAct}는 원본 추적(raw traces)에서 기술 전이(normalized skill transfer, NST)를 44.7-67.1% 감소시켜, 감사 증거를 유지하면서 프로시저적 능력 누수를 줄입니다.

- **Performance Highlights**: 행동 워터마크는 93.6-100.0%의 진짜 탐지율(true detection rate)을 보이면서 최대 1.9%의 오탐율(false alarm rate)을 기록했습니다. 이 연구의 결과는 대중 에이전트 추적이 보안 인터페이스(security interfaces)로 기능할 수 있음을 보여주며, 선택적 편집(selective redaction)이 감사 증거를 제거하지 않으면서 절차적 능력 누수를 줄일 수 있다는 점을 입증합니다. 이를 통해 에이전트 행동의 보안성을 더욱 강화할 수 있는 방안을 제시합니다.



### Recovering the Zipfian Distribution in Unsupervised Term Discovery (https://arxiv.org/abs/2606.10781)
- **What's New**: 이 논문에서는 비지도 용어 발견(unsupervised term discovery)의 새로운 접근 방식을 제시합니다. K-means와 같은 중심 기반 클러스터링 방법이 자연어의 Zipfian 분포를 잘 포착하지 못하는 문제를 지적하며, 그래프 기반 클러스터링(graph-based clustering) 기법을 사용하여 이를 개선할 수 있음을 보여줍니다. 저자들은 다양한 언어에서 단어와 음절 단위의 어휘 발견에서 그래프 클러스터링이 우수한 성능을 보인다고 주장합니다.

- **Technical Details**: 논문은 비지도 용어 발견의 세 가지 단계—구간(segment) 구성, 특성 추출(self-supervised learning), 클러스터링—을 따릅니다. 특히, 그래프 클러스터링 방법론은 유사성 그래프를 기반으로 하여 두 단계로 진행됩니다: 유사성 그래프를 구축한 후 잠재 커뮤니티를 개선하기 위해 Leiden 알고리즘을 사용하여 파라미터를 조정합니다. 이 방법은 클러스터의 밀집도를 조절할 수 있는 두 개의 하이퍼파라미터를 통해 제어할 수 있습니다.

- **Performance Highlights**: 결과적으로, 저자들은 그래프 클러스터링이 K-means와 같은 전통적인 방법보다 더 잘 작동하며, 특히 단어 및 음절 수준의 어휘 발견에서 Zipfian 분포를 더 잘 복구한다고 보고합니다. 또한, 다른 바닥에서부터 진행하는 집합적 클러스터링이 잘 작동했지만 계산적으로 덜 효율적이라는 점도 언급되었습니다. 실험은 세 가지 언어(Afrikaans, French)에서 수행되어, 여러 언어에서도 일관된 성능 결과를 확인하였습니다.



### N-GRPO: Embedding-Level Neighbor Mixing for Enhanced Policy Optimization (https://arxiv.org/abs/2606.10768)
Comments:
          ACL 2026 Findings. 16 pages, 3 figures. Code: this https URL

- **What's New**: 본 논문에서는 N-GRPO라는 새로운 탐색 전략을 Group Relative Policy Optimization (GRPO) 프레임워크에 통합하여 제안합니다. 이 방법은 token-level sampling 대신 Semantic Neighbor Mixing을 활용하여 더 다양한 해결 경로를 생성합니다. 이는 입력 표현을 동적으로 구성함으로써 국소적인 의미적 매니폴드(semantic manifold)를 준수하면서 다채로움을 추가합니다.

- **Technical Details**: 제안된 N-GRPO는 토큰을 샘플링하기보다는 앵커 토큰의 임베딩과 가장 가까운 의미적 이웃(neighbor) 임베딩을 혼합하여 입력 표현을 구성합니다. 이 과정에서 랜덤성을 유지하면서도 의미적 일관성을 보장하게 됩니다. 이러한 접근 방식은 기존의 token-level sampling과 continuous embedding exploration 간의 간극을 효과적으로 메우며, 모델이 더 풍부한 탐색 공간을 탐색할 수 있도록 합니다.

- **Performance Highlights**: N-GRPO는 다양한 수학적 추론 기준에서 강력한 기준 모델(baselines)을 일관되게 초과하는 성능을 보여주었습니다. 실험 결과, 1.5B 및 7B 모델 스케일에서 Pass@16 및 Pass@32 메트릭에서 향상된 결과를 기록했습니다. 이 방법은 또한 배포되지 않은 작업에 대한 강력한 일반화 능력을 나타내며, 다양한 모델 크기에서 안정적인 개선을 이루었습니다.



### When the Chain of Thought Knows Better: Failure Modes in Multi-Turn Reasoning Models (https://arxiv.org/abs/2606.10740)
Comments:
          Accepted at the ICML 2026 FAGEN Workshop

- **What's New**: 이번 연구에서는 다중 대화에서의 안전성 실패를 진단하기 위한 새로운 추적 수준 분석 방법인 CoT-Output 2x2 안전 매트릭스를 제안합니다. 이 매트릭스는 각 대화 턴을 내부 추론(internal reasoning)과 가시적 출력(visible output)의 두 축으로 구분하여 4개의 실패 유형을 정의합니다. 이는 기존의 평가 방식으로는 파악할 수 없었던 다중 턴의 복잡한 실패 양상을 드러내는 데 기여합니다.

- **Technical Details**: 연구에서는 3개의 정제된(reasoning) 모델에 대해 고정된 공격자가 있는 다중 턴 평가(framework)를 수행했습니다. 각 턴에 대한 6750개의 관측치를 수집하여 안전성 모델의 실패 양상을 분석하였고, 특히 'context-injection failure'(안전한 내부 추론에도 불구하고 유해한 출력이 발생하는 경우)를 발견했습니다. 이와 같은 다중 턴 대화 데이터를 통해 연구는 명확한 오류 유형을 식별하고, 안전성 점검(safety check) 시스템의 필요성을 강조합니다.

- **Performance Highlights**: 실험 결과, DeepSeek-R1-7B 모델은 53.1%의 비율로 alignment faking이 발생하였으며, 이는 기존 최전선 모델과 유사한 수준입니다. 또한 감시 큐가 오히려 alignment faking 비율을 증가시키는 역설적인 경향을 보였고, Qwen-4B-Thinking 모델에서는 최대 13.8%의 context-injection failure가 관찰되었습니다. 이러한 결과들은 모델이 안전한 내부 상태에도 불구하고 위험한 출력을 반복하게 되는 다중 턴 대화에서의 복잡한 상호작용을 잘 보여줍니다.



### Pre-AF 13: An Interpretable Atrial Fibrillation Risk Score Mined from Discharge Reports (https://arxiv.org/abs/2606.10725)
Comments:
          Main paper with appendix; 3 main figures, 3 supplementary figures, multiple tables. O. Shakhmatova and D. Kriukov contributed equally (co-first authors). E. Panchenko, A. Shelmanov, and D. V. Dylov are co-senior authors. Corresponding authors: O. Shakhmatova (this http URL@gmail.com) and D. V. Dylov (this http URL@skol.tech)

- **What's New**: 이 연구에서는 심혈관 질환(CVD) 환자에서 24개월 및 전체 추적 기간 동안 심방 세동(AF) 위험을 예측하는 해석 가능한 기계 학습 모델을 개발했습니다. 기존의 AF 위험 점수는 심혈관 질환 환자에서 일반적인 위험 요인에 의존하여 위험 분류가 제한적입니다. 이 연구는 전자 건강 기록을 활용하여 AF 위험 예측을 위한 새로운 접근 방식을 제안했습니다.

- **Technical Details**: 연구는 러시아의 국립 연구 심장학 센터에서 2012년 1월부터 2019년 5월 사이에 두 번 이상 입원한 18세 이상의 CVD 환자를 대상으로 하는 회고적 관찰 연구로 진행되었습니다. 비정형 퇴원 보고서를 73개의 구조적 특성으로 변환하는 맞춤형 자연어 처리(NLP) 파이프라인이 사용되었습니다. 최종 모델은 여러 가지 기계 학습 방식으로 평가되었으며, 특히 LightAutoML을 사용하여 예측 모델이 구축되었습니다.

- **Performance Highlights**: 개발된 전체 모델은 24개월 동안 ROC AUC 0.735를 기록했으며, 단순 모델도 유사한 성능(0.725, 0.696)을 보였습니다. 모든 비선형 모델이 기존의 임상 위험 점수보다 우수한 성능을 보여주었습니다. 연구 결과, 나이와 좌심방 부피가 주요 예측因자로 확인되었으며, 신뢰할 수 있는 예측 점수를 통해 AF 발생률을 효율적으로 분류할 수 있음을 보여주었습니다.



### From Observation to Intervention: A Causal Audit of Expert Importance in Mixture-of-Experts Models (https://arxiv.org/abs/2606.10703)
Comments:
          9 pages, 2 figures, 9 tables. Accepted at the ICML 2026 Workshop on Philosophy of Science Meets Machine Learning (PhilML). Non-archival

- **What's New**: 이 논문은 기존의 해석 가능성(methods) 방법들이 관찰된 모델 행동에 대한 통계치를 기반으로 개별 개입(intervention)의 효과를 추론하는 방식의 유효성을 테스트한 새로운 실험을 제공합니다. Mixture-of-Experts (MoE) 모델에서 라우팅 통계(router statistics)를 통해 전문가(experts)를 제거할 수 있는 기반(로)을 분석하였습니다. 특히, 전문가를 제거하는 것이 모델에 어떤 영향을 미치는지에 대한 명확한 반례(counterexample)를 제시했습니다.

- **Technical Details**: 논문에서는 Mixture-of-Experts 아키텍처에서의 라우팅 통계(routing statistics)가 기능적 중요성(functional importance)을 예측하는 데 신뢰할 수 없음을 보여줍니다. 연구는 OLMoE, Qwen1.5-MoE 및 DeepSeek-V2-Lite와 같은 고중복(high-redundancy) MoE 아키텍처에서 세 가지 기준을 사용하여 실험했습니다. 각 모델에서 특정 레이어(layer)에 대해 관찰된 메트릭(observational metrics)이 원인(causal) 예측에 부합하지 않는 것을 발견했습니다.

- **Performance Highlights**: 실험 결과, 측정된 메트릭이 Bonferroni 수정에 따라 유의미한 성과를 보인 사례는 없었으며, 효과 크기는 Cohen의 d = 0.17 이하였습니다. 유일하게 OLMoE의 마지막 레이어에서 약간 유의미한 신호(d = +0.231, p = 0.0013)가 발견되었습니다. 기존의 전문가 제거 방법은 선택 기준(selection criteria)이 interchangeable한 이유로 성공한 것이지, 실제로 필요 없는 전문가를 찾는 것이 아님을 확인했습니다.



### Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory (https://arxiv.org/abs/2606.10677)
- **What's New**: 이 논문에서는 LLM 에이전트를 위한 새로운 메모리 아키텍처인 Infini Memory를 제안합니다. Infini Memory는 에이전트 메모리를 주제 기반의 문서로 구성하여 관련 증거를 집계하고 메타데이터를 보존하며 사실을 수정하는 데 도움을 줍니다. 기존 메모리 시스템의 한계를 극복하고 정보를 효율적으로 관리하기 위해 메모리의 생애 주기 유지 관리 문제로 접근합니다.

- **Technical Details**: Infini Memory는 새 관찰을 버퍼에 먼저 기록한 후 주기적으로 일관된 텍스트 컨텍스트로 통합하는 방식을 사용합니다. 검색 시 LLM은 반복적인 도구 호출을 통해 메모리를 읽도록 설계되어 있어, 단일 검색 단계 대신 여러 단계를 거쳐 정보를 검색합니다. 또한, 주제문서 구조를 통해 관련 증거를 그룹화하고 서로 관련된 정보의 배치를 최적화합니다.

- **Performance Highlights**: MemoryAgentBench에서 Infini Memory는 64.7%라는 높은 점수를 기록하며, 주제 구조화 유지 관리와 반복적인 증거 검사가 장기 메모리 사용의 보완적 측면을 개선하는 것으로 나타났습니다. 이 시스템은 온라인 유지 관리 오버헤드를 줄이며 해석 가능성이 높고 편집 가능한 상태를 강조하는 설계로, 기존 시스템들과 차별화됩니다.



### How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs (https://arxiv.org/abs/2606.10646)
Comments:
          25 pages, 7 figures, 11 tables. Accepted at ICML 2026

- **What's New**: 이 논문에서는 Token-level credit assignment 문제를 해결하기 위해 FlowTracer라는 새로운 RL 프레임워크를 제안합니다. 기존의 방법들이 정보 전파의 전반적인 구조를 무시하는 반면, FlowTracer는 Attention에 기반한 Directed Acyclic Graph(DAG)를 사용해 토큰이 최종 답변에 미치는 영향을 분석합니다. 이 접근법은 정보 흐름의 뼈대를 추출하여 질문과 답변 간의 관계성을 명확히 하고, 이를 통해 토큰별 보상을 조정합니다.

- **Technical Details**: FlowTracer는 각 토큰이 향하는 정보를 고려해 토큰 간의 연결고리를 분석하며, 이를 기반으로 대답 영역에 도달하는 흐름을 재조정합니다. Attention 가중치를 활용해 각 엣지의 용량을 재조정하고, 정보 전파를 지역 유지를 통해 흐름이 왜곡되지 않도록 합니다. 이렇게 하여 정보가 최종 답변에 도달할 수 있는 경로를 명확하게 독립적인 네트워크 구조로 표현합니다.

- **Performance Highlights**: FlowTracer의 도입으로 인해 학습 효율성과 추론 성능이 일관되게 향상되었습니다. 특정 토큰들이 정보 전송에 얼마나 기여하는지를 정량화함으로써, 중요도가 높은 토큰에 대한 집중적인 보상을 부여하고, 중요도가 낮은 토큰의 업데이트를 억제함으로써 최종적인 추론accuracy를 개선했습니다. 이 프레임워크는 다양한 추론 작업에서 성능을 유의미하게 증대시키는 효과를 나타냈습니다.



### Causal Ensemble Agent: Hierarchical Causal Discovery with LLM-guided Expert Reweighting (https://arxiv.org/abs/2606.10607)
- **What's New**: 이 논문에서는 기존의 인과 탐색 방법을 개선하기 위해 Causal Ensemble Agent(CEA)라는 새로운 프레임워크를 제안합니다. CEA는 여러 통계 전문가로부터 구조적 통찰력을 집약하고, LLM을 메타 심판자(meta-referee)로 사용하여 불확실성에 따라 전문가의 가중치를 동적으로 조정합니다. 이러한 방법은 인과 그래프의 정확성과 완전성을 더욱 향상시키는 것으로 나타났습니다.

- **Technical Details**: CEA는 계층적 앙상블 프레임워크로, 여러 인과 탐색 방법의 출력을 통합하여 그래프의 뼈대(skeleton), v-구조(v-structures), 엣지 방향(edge orientations)를 차례로 집약합니다. 각 계층에서 부트스트랩 기반의 신뢰도 점수를 사용하여 전문가의 신뢰성을 평가하고, LLM이 불확실한 관계에 대해 가중치를 조정하여 인과 그래프를 구성합니다. 이러한 방식으로 CEA는 직접적인 엣지 예측을 피하고 데이터에 지향적인 결정을 유지합니다.

- **Performance Highlights**: CEA는 8개의 합성 및 실제 데이터셋 벤치마크에서 모든 인과 탐색 방법 중 가장 뛰어난 성능을 나타냈습니다. 실험 결과, CEA는 인과 탐색의 정확성과 다양한 도메인에서의 강건성을 크게 개선하며, 기존 방법보다 LLM 쿼리를 현저히 줄입니다. 이러한 성과는 LLM이 메타분석에서 어떻게 효과적으로 활용될 수 있는지를 잘 보여줍니다.



### Representation-Aware Advantage Estimation: Your Reward Model Provides More Than A Scalar Outpu (https://arxiv.org/abs/2606.10528)
- **What's New**: 이번 논문은 강화를 위한 학습 방식(RLHF)에서 scalar 보상을 넘어서, 더 정교한 신호를 이용하여 보상 모델(RM)의 숨겨진 상태를 활용하여 이점을 추정하는 새로운 방법인 Graph-based Advantage Estimation (GraphAE)을 도입했습니다. GraphAE는 응답을 그래프로 모델링하여 이웃의 맥락 정보를 통합하며, 이는 기존의 그룹 기반 RL 알고리즘에 원활히 통합될 수 있습니다. 논문은 GraphAE가 다양한 모델과 벤치마크에서 실험되었으며, 이를 통해 성능이 일관되게 개선되었다는 점을 강조합니다.

- **Technical Details**: GraphAE는 보상 모델(RM)의 숨겨진 상태를 사용하여 유사성 그래프를 구성하고 해당 그래프를 통해 이점을 계산합니다. 각 응답은 그래프의 노드로 표현되며, 엣지는 RM의 숨겨진 상태에서의 유사성을 나타냅니다. 또한, GraphAE는 원래의 scalar 보상과 그래프의 매끄러움을 균형 있게 고려하여 이점을 추정하는 그래프 정규화 목적을 해결하여, 빠르고 효율적인 폐쇄형 솔루션을 제공합니다.

- **Performance Highlights**: 실험 결과 GraphAE는 Arena-Hard-v0.1에서 +6.3, AlpacaEval 2.0에서 +8.27, MT-Bench에서 +0.22의 성능 향상을 보여주었습니다. 이는 GraphAE가 더 샘플 효율적이고 강력한 RLHF를 가능하게 한다는 점을 입증합니다. 또한, GraphAE는 그룹 내 보상 분산을 줄이고 수렴 속도를 가속화하는 데도 기여합니다.



### Advancing the State-of-the-Art in Empirical Privacy Auditing (https://arxiv.org/abs/2606.10481)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 민감한 데이터로의 파라미터 효율적인 파인튜닝에 대한 기밀성 문제를 다룹니다. Empirical Privacy Auditing(EPA)을 통해 이 문제를 정량화하고, 새로운 'canary' 생성 방법을 제안하여 고온 샘플링을 통해 이러한 'canary'를 효과적으로 생성합니다. 이 방법은 훈련 데이터와 유사하지만, 충분히 이상의 비정상적인 예제를 포함하여 민감한 데이터의 유출 위험을 평가할 수 있도록 합니다.

- **Technical Details**: 기술적으로, 연구자들은 고온 샘플링(T ≥ 0.8) 방식을 사용하여 LLMs로부터 통합된 'canary'를 생성합니다. 이 'canary'는 훈련 데이터에 인젝션하여 모델의 공격에 대한 응답을 평가하는 데 도움을 줍니다. 금전적인 이점을 위해 본 연구는 새로운 데이터 감사를 도입하여 합성 데이터의 유출을 측정하는 방법을 설명하며, 기초 데이터 분포에서 통계적으로 비정상적이지만 여전히 의미적 구조를 유지하는 생성 과정을 강조합니다.

- **Performance Highlights**: 이 연구의 결과, 'canary'의 엔트로피와 모델 용량 사이의 상관관계를 규명하였고, 모델 용량이 증가함에 따라 최적의 'canary' 엔트로피가 증가한다는 점을 발견했습니다. 이는 LLM이 민감한 데이터의 기록을 메모리하는 데 있어 더 강력한 공격 신호를 생성할 수 있음을 의미합니다. 또한, 기존 방법에 비해 새로운 감사를 통해 합성 데이터의 유출을 더 정밀하게 평가할 수 있음을 보여줍니다.



### Decoupling Thought from Speech: Knowledge-Grounded Counterfactual Reasoning for Resilient Multi-Agent Argumentation (https://arxiv.org/abs/2606.10475)
Comments:
          Accepted for publication in the Proceedings of the 30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026)

- **What's New**: 이 논문에서는 Multi-agent Debate (MAD) 프레임워크에서 기존의 모델들이 최종 출력의 정확성에 유리하게 최적화되어 있기 때문에 긴 대화에서의 안정성을 고려하지 못하는 문제를 지적합니다. 이를 해결하기 위해, Knowledge-Grounded Counterfactual Reasoning (KG-CFR)라는 이중 단계 아키텍처를 도입하여, 개인적 계획 버퍼와 공적 실행 레이어 간의 엄격한 분리를 구현하여 프로세스의 일관성을 유지합니다. 이 연구는 기존 반응형 시스템의 한계를 극복하고, 시스템의 저항력을 구조적으로 향상시키는 방법을 제시합니다.

- **Technical Details**: KG-CFR 플랫폼은 Dynamic Resource Allocation under Uncertainty (DRAU) 환경에서 성능을 평가합니다. 이 아키텍처는 개인적인 시뮬레이션과 전략적 계획을 하는 버퍼를 분리하여, 외부 요인에 대한 Noise에 의한 논리 손상을 방지합니다. 또한, 새로운 벡터 메트릭스를 통해 논의의 다양성과 계획 실행 일치를 측정하여 운영의 안정성을 높은 방향성과 일관성을 가지고 증명합니다.

- **Performance Highlights**: KG-CFR은 95% 이상 전체 실험에서 judge가 감지한 Critical post-shock degradation(품질 변화 기준, $  -0.20$)을 방지하며, 주장 품질이 0.694에서 0.822로 증가했습니다. 이러한 결과는 KG-CFR이 장기적인 압박 하에서도 품질 손실 없이 시스템 저항력을 강화하는 중요한 요소가 된다는 것을 보여줍니다. 더불어, KG-CFR은 구술에서의 반복(looping)을 줄이며, 이러한 수정된 메트릭들이 운영의 일관성을 어떻게 보장하는지 보여줍니다.



### ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs (https://arxiv.org/abs/2606.10461)
Comments:
          Accepted to ICML 2026

- **What's New**: 이번 연구에서는 Energy-based Representation Alignment (ERAlign) 프레임워크를 제안합니다. 이 프레임워크는 GNN(그래프 신경망)으로 인코딩된 그래프 구조와 LLM(대규모 언어 모델)으로 파생된 텍스트 임베딩을 공유 잠재 공간으로 투영하여 분포 일관성을 달성합니다. ERAlign은 GNN과 LLM의 중간 레이어를 정량화하여 계층별 정렬을 수행하며, 다운스트림 작업에 대한 잘 정렬된 표현을 생성합니다.

- **Technical Details**: ERAlign은 Cramér distance(크라메르 거리)를 사용하여 표현 간 불일치를 정량화하고 EBM(에너지 기반 모델) 목표 함수로 최적화하여 분포 일관성을 향상시킵니다. 이 프레임워크는 중간 LLM 의미를 GNN message passing에 주입하고 그래프 표현을 LLM에 소프트 프롬프트를 통해 전달하여 양방향 정보 융합을 가능하게 합니다. 또한, Energy Discrepancy (ED) 최소화 방식을 도입하여 샘플링 비용을 줄이며, 전반적인 훈련 효율성을 높입니다.

- **Performance Highlights**: ERAlign은 8개의 벤치마크 데이터셋에서 최첨단 성능을 달성하며, 다양한 감독 비율 및 제로샷 크로스 작업 전이 시나리오에서도 뛰어난 라벨 효율성을 보여줍니다. 기존 방법들에 비해 상당한 성능 개선을 이루었으며, 특정 작업에 맞춘 두 가지 변형도 개발하였습니다. 이 연구는 TAGs와 같은 복합적인 모델링 과제를 해결하는 데 기여하고 있습니다.



### Leveraging Social Media Data for COVID-19 Studies (https://arxiv.org/abs/2606.10459)
Comments:
          8 pages, 1 figure

- **What's New**: 최근 소셜 미디어 네트워크는 정보에 대한 널리 선호되는 출처가 되었습니다. 특히 COVID-19 팬데믹 동안, 소셜 미디어는 최신 뉴스 및 정보를 얻는 가장 많이 사용되는 플랫폼 중 하나로 자리잡았습니다. 본 논문에서는 COVID-19 관련 소셜 미디어 사용에 대한 연구를 탐구하고 논의합니다.

- **Technical Details**: 이 장에서는 사용자의 게시물에서 표현된 언어적, 시각적, 감정적 지표에 대해 설명합니다. 또한, COVID-19 팬데믹 동안 활용된 소셜 미디어 데이터의 종류를 분류하고, 다양한 기계 학습(machine learning), 피쳐 엔지니어링(feature engineering), 자연어 처리(natural language processing), 그리고 설문 조사 방법을 소개합니다.

- **Performance Highlights**: 소셜 미디어는 적절히 활용될 경우, 신뢰할 수 있는 뉴스 및 공공 인식을 전파하는 유용한 디지털 도구가 될 수 있습니다. 전 세계적으로 약 46억 명의 소셜 미디어 사용자가 존재하는 만큼, 이러한 플랫폼을 통한 정보 공유는 사람들이 팬데믹을 인식하고 대처하는 방식에 영향을 미칠 수 있습니다.



### SpenseGPT: Practical One-shot Pruning Enabling Sparse and Dense GEMMs for LLM Inferenc (https://arxiv.org/abs/2606.10445)
- **What's New**: 본 논문에서는 Spense라는 실용적인 하이브리드 희소-밀집 포맷을 제안합니다. 이 포맷은 각 가중치 행렬을 2:4 희소 영역과 밀집 영역으로 나누어 엄격한 50% 희소성 제약을 완화합니다. 기존의 희소 및 밀집 GEMM 라이브러리와 호환되어 특별한 컴파일러 지원 없이도 실행 가능하다는 장점이 있습니다.

- **Technical Details**: Spense 포맷은 두 개의 연속적인 영역으로 나뉘며, 하나의 영역은 표준 2:4 희소 패턴으로 잘리고 다른 하나는 밀집합니다. 이를 통해 기존의 최적화된 희소 및 밀집 GEMM 라이브러리(cuSPARSELt, cuBLAS 등)를 활용하면서도 런타임 오버헤드가 줄어들어 성능을 극대화할 수 있습니다.

- **Performance Highlights**: 실험 결과, SpenseGPT 방법을 통해 B200 GPUs에서 FP8 정밀도를 사용하여 최대 1.2배의 인코딩 속도 향상을 이뤄냈으며, 모델 품질을 유지했습니다. 특히, SpenseGPT는 다양한 기준에서 일관된 모델 품질을 유지하며 높은 성능을 보였으며, 이는 최근 GPU에서의 재현성이 뛰어난 중요성과 연구 기여를 나타냅니다.



### Enhancing Multilingual LLM-based ASR with Mixture of Experts and Dynamic Downsampling (https://arxiv.org/abs/2606.10439)
Comments:
          Accepted by ICASSP 2026

- **What's New**: 본 연구는 자동 음성 인식(ASR) 시스템과 대형 언어 모델(LLM)의 효과적인 통합을 위한 프로젝트 기반 LLM-ASR 프레임워크를 제안합니다. 특히 다국어 일반화 및 모달리티 정렬의 두 가지 핵심 문제를 해결하는 것을 목표로 합니다. Mixture of Experts (MoE) 아키텍처와 Continuous Integrate-and-Fire (CIF) 메커니즘을 활용하여 ASR 성능을 개선하고, 실험 결과에서 강력한 기준 모델을 초월한 성능 향상을 보여줍니다.

- **Technical Details**: 제안된 구조는 MoE 기반 프로젝터와 CIF 기반 모달리티 정렬을 포함하여 LLM-ASR 아키텍처의 성능과 일반화 개선에 기여합니다. MoE 프로젝터는 언어마다 특화된 전문가 네트워크에 입력을 동적으로 라우팅하여 다국어 ASR에서 우수한 성능을 발휘하고, CIF는 음향 피처를 텍스트 토큰으로 유동적으로 정렬할 수 있게 합니다. 이러한 접근 방식은 기계 번역 및 음성 인식에서 강력한 기술적 기반을 제공합니다.

- **Performance Highlights**: 실험적으로 MoE-enhanced 프로젝터와 CIF 기반 모달리티 정렬의 통합이 음성-텍스트 매핑을 보다 효과적으로 수행하는 것을 입증하였습니다. ASR 정확도와 강인성이 크게 개선되었으며, 특히 다양한 언어 및 억양에 대한 적응력이 우수합니다. 연구에서 제안하는 방법은 LLM 기반 ASR 시스템을 보다 정확하고 견고하게 만드는 한 걸음을 더 나아가게 합니다.



### Parallel Causal Associative Fields: Gated Sparse Memory for Long-Context Language Modeling (https://arxiv.org/abs/2606.10435)
Comments:
          17 pages, 5 figures, and 6 tables. Experiments on WikiText-103, PG-19, and WikiText-2 using TPU v4-32 and NVIDIA RTX 3060 hardware. Code: this https URL

- **What's New**: 이 연구에서는 기존의 Transformer 모델에서 발생하는 계산 비용을 줄이기 위해 새로운 패러다임인 Parallel Causal Associative Field (PCAF)를 도입합니다. PCAF는 명시적 토큰 간 상호 작용을 구성하는 대신, 인과 성공 기록의 희소 연관 메모리를 기록하고 그 메모리에서 제한된 읽기를 수행합니다. 이를 통해 PCAF는 복잡한 전체 주의(attention) 계산의 비용을 절감하고, 연관 캐시를 통해 유용한 신호를 효과적으로 회수할 수 있는 가능성을 열어줍니다.

- **Technical Details**: PCAF는 맥락 윈도우에서 로컬 기록을 해시 버킷에 기록하고 현재 쿼리에 대한 제한된 후보 집합을 검색하는 구조로 설계되었습니다. 이 과정은 고정 크기의 상태로 압축하는 대신 패러메트릭 언어 모델과 결합하여 희소한 긴 맥락 접근을 유지합니다. 모델은 기억 주소 지정과 값 해결이라는 두 가지 역할을 분리하며, 학습된 게이트를 통해 두 가지 예측 분포를 혼합하여 각 위치에서 사용합니다.

- **Performance Highlights**: PCAF는 WikiText-103과 PG-19에서 완전한 자기 회귀(pretraining) 평가를 통해 303M 매개변수 및 T = 2048의 길이로 각각 36.31 및 52.45의 퍼플렉서티(perplexity)를 달성했습니다. 이는 밀집(dense) Transformer와 비교했을 때 성능에서 이점을 보여주며, TPU pod에서 0.61-0.62M 토큰/초의 속도로 처리할 수 있음을 나타냅니다. 또한, 다양한 실험을 통해 연관 캐시 및 학습된 게이트의 효과가 속도와 품질 간의 trade-off에 중요한 영향을 미친다고 보고했습니다.



### Selection, Not Salience: The Shape and Limits of Personalization in Social Highlighting (https://arxiv.org/abs/2606.10398)
Comments:
          9 pages, 1 figure, 3 tables

- **What's New**: 이 논문은 개인화된 독서 경험이 실제로 효과적인지, 그리고 그 한계는 무엇인지 탐구합니다. 소셜 웹 하이라이터와 공동 독자 정체성 제어(co-readership identity control)를 사용하여, 독자의 개인적 이력이 독서 선택을 얼마나 잘 예측하는지를 조사했습니다. 이 연구는 문서 레벨에서 개인화 신호의 존재와 그 강도를 밝혀내고, 문장 레벨의 개인화가 실제로 성과를 내지 못하는 상황을 설명합니다.

- **Technical Details**: 연구에서 개인화의 경계를 규명하는 과정에서, 문서 선택 시 실질적인 개인화 신호의 크기를 +0.12에서 +0.17까지 확인하였습니다. 이는 문서와 스팬 간의 선택 신호가 유사한 강도로 나타났음을 시사합니다. 반면, 개인화된 문장 수준의 자동 하이라이트가 일반적인 기준보다 성능이 낮은 결과를 보여주었으며, 이는 개인화가 실질적인 이점을 제공하지 않는다는 것을 강조합니다.

- **Performance Highlights**: 연구 결과는 문서 선택 레벨에서 개인화의 효과가 있지만, 문장 레벨에서는 유의미한 개선이 없음을 나타냈습니다. 특히, 개인 모델이 제공하는 재순위가 기존의 일반 순서보다 더 낫지 않다는 점이 확인되었습니다. 이러한 결과는 개인화된 독서 경험을 제공하는 것이 과연 의미가 있는지를 다시 한번 생각할 기회를 제공합니다.



### Agentic Hybrid RAG for Evidence-Grounded Muon Collider Analysis (https://arxiv.org/abs/2606.10381)
Comments:
          22 pages, 5 figures, and 6 tables

- **What's New**: 이 연구에서는 muon collider 연구를 위한 evidence-grounded retrieval-augmented generation (RAG) 프레임워크인 agentic hybrid RAG를 소개합니다. 이 프레임워크는 희소한 어휘 검색(sparse lexical retrieval)과 밀집한 의미 기반 검색(dense semantic retrieval)을 통합한 하이브리드 검색 모듈과 함께, 쿼리 분해(query decomposition) 및 증거 확장(evidence expansion)을 지원하는 에이전틱 추론 모듈을 결합합니다. 이를 통해 muon collider 도메인에서의 과학적 질문 응답의 효율성을 높이고, 문헌의 다양성을 체계적으로 평가할 수 있는 첫 번째 벤치마크를 설정했습니다.

- **Technical Details**: 이 프레임워크에서 사용되는 하이브리드 검색 모듈은 BM25와 FAISS 색인(faiss indexing) 기법을 활용하여 여러 유형의 쿼리에 대한 검색 성능을 최적화합니다. 경량 에이전트는 초기 검색에서 놓친 증거를 회수하기 위해 쿼리를 분해하고 후속 쿼리를 생성하여, 과학적 질문 응답에서 필요한 정확성과 추적 가능성을 유지합니다. 결국 이 시스템은 하이브리드 검색과 제어된 증거 확장을 통해 신뢰할 수 있는 정보 수집 및 응답 생성을 지원합니다.

- **Performance Highlights**:  extensive evaluation 결과, 하이브리드 검색이 가장 강력한 검색 기반을 제공하며, 에이전틱 추론은 증거 확장 및 응답 합성에서 가장 효과적인 것으로 나타났습니다. agentic hybrid RAG는 검색 효과성, 응답 품질, 증거 커버리지 및 사실적 기반에서 대표적인 검색 및 RAG 기준선보다 일관되게 우수한 성과를 보였습니다. 이 연구는 미래의 muon collider 연구와 고에너지 물리학 분석 요원 운용을 위한 기초 자료를 제공합니다.



### From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs (https://arxiv.org/abs/2606.10298)
Comments:
          27 pages, 9 figures

- **What's New**: 본 논문은 큰 언어 모델이 외부 문맥(context)과 파라메트릭 우선순위(prior) 간의 갈등을 효과적으로 처리하기 위한 새로운 패러다임을 제안합니다. 기존의 방법들은 문맥이 항상 우선순위보다 신뢰할 수 있다고 가정하며, 잘못된 문맥으로 인해 올바른 확률 구조를 무시하게 됩니다. 저자들은 이를 '갈등 인식(conflict-aware)' 패러다임으로 일반화하여 갈등 신호에 따라 우선순위와 문맥 간의 권한을 동적으로 할당합니다.

- **Technical Details**: 저자들은 두 가지 갈등 상태인 수정(correction)과 저항(resistance)을 처리하기 위해 Adaptive Regime Routing (ARR)이라는 새로운 방법을 제안하고, 이를 통해 모델간의 저항성(EM)을 크게 향상시켰습니다. 이 방법은 각 단계에서 갈등 신호를 바탕으로 상이한 두 가지 영역 사이를 전환하며, 이는 기존의 대비 디코딩 방법들과 비교할 때 더 넓은 범위에서의 갈등 처리를 가능하게 합니다. 연구는 또한 '파워 패밀리(power family)'를 정의하고, 기존 방법들이 이 가족의 특별한 경우임을 설명합니다.

- **Performance Highlights**: TriState-Bench라는 새로운 평가 프로토콜을 통해 모델이 갈등 상태를 측정하고, 수정, 저항, 합의의 세 가지 측면을 평가할 수 있게 되었습니다. ARR을 적용함으로써 저자들은 저항 EM을 6 이하에서 16-33으로 증가시키는 성과를 달성하였으나, 수정성과 합의는 유지하였습니다. 이는 문맥과 우선순위 간의 갈등을 효과적으로 관리하고, 더 신뢰할 수 있는 텍스트 생성을 가능하게 합니다.



### When Metrics Disagree: A Meta-Analysis of Knowledge-Graph-Completion Model Benchmarking (https://arxiv.org/abs/2606.10287)
- **What's New**: 이 논문은 Knowledge Graph Completion (KGC) 모델의 평가 방법론에 대한 새로운 접근 방식을 제시합니다. 기존의 평가 방식이 개별적인 랭크 기반 메트릭에 의존하여 데이터세트 간의 상충된 결과를 초래하며, 이로 인해 공정한 비교가 어렵다는 문제를 지적합니다. 모델 순위를 다차원적으로 평가하는 Multi-Criteria Decision-Making (MCDM) 문제로 재구성하여, 다양한 평가지표의 일관성과 안정성을 확보하려는 시도를 하였습니다.

- **Technical Details**: KGC 평가를 MCDM 문제로 재구성하기 위해 7가지 집계기(aggregators)를 도입하여 성능을 매트릭스 형태로 구성합니다. 이들은 EDAS, TOPSIS, VIKOR, MOORA, WASPAS, Borda Count, Z-score 등을 포함합니다. 각 집계기는 일관성(consistency), 안정성(stability), 메트릭 독립성(metric independence), 노이즈 저항성(robustness), 일반화 가능성(generalizability)이라는 다섯 가지 테스트 차원에서 평가되어 신뢰성(reliability)을 변화에 따라 분석합니다.

- **Performance Highlights**: 실험 결과에 따르면 Z-score가 tail 예측에서는 DualE를 가장 높은 순위로, 관계 예측에서는 Flow-Modulated Scoring (FMS)를 가장 높은 순위로 평가하는 매우 균형 잡힌 집계기임을 확인하였습니다. 또한, 일관성과 안정성은 모델 제거에 크게 영향을 받지 않는 반면, 일반화 가능성과 독립성은 가장 민감한 지표로 나타났습니다. 이 연구는 KGC 모델 평가에서의 일관성을 해결하고, 집계기의 선택과 모델 벤치마킹에 대한 증거 기반의 지침을 제시합니다.



### Benchmarking and Exploring the Capabilities of LLMs for Attack Investigations (https://arxiv.org/abs/2606.10281)
- **What's New**: 이 논문은 보안 관련 시스템 감사 로그를 분석하는 LLM(대형 언어 모델)의 성능을 평가하기 위한 새로운 벤치마크 데이터셋인 AuditBench를 소개합니다. 이 벤치마크는 지원팀이 일반적으로 수행하는 네 가지 로그 조사 작업에 대한 성능을 탐색하기 위해 설계되었습니다. AuditBench는 Linux 및 Windows 머신에서 수집된 시스템 감사 로그로 구성되며, 악의적인 활동과 무해한 활동을 포함한 50개 이상의 다양한 보안 조사 시나리오를 포괄합니다.

- **Technical Details**: 우리는 새로 생성한 감사 로그 데이터와 MITRE ATT&CK 프레임워크의 공격 시나리오로부터 세심하게 선별된 데이터셋을 결합하여 AuditBench를 구성하였습니다. 벤치마크는 LLM의 성능 메트릭을 자동으로 계산하는 평가 프레임워크를 발전시키며, 이 메트릭은 진정 긍정(true positive) 및 오경고(false positive) 비율, LLM이 제공하는 설명의 정확도를 포함합니다. 다양한 LLM을 분석하여 보안 감사 로그 조사에서 LLM 성능과 설명의 질을 평가합니다.

- **Performance Highlights**: 총체적으로, 많은 LLM이 우리의 벤치마크에서 보안 조사 작업에 대해 혼합된 성능을 보였고, 오경고가 많아 과도한 의심(challenging) 경향을 나타냈습니다. 특정 작업에서는 소규모 모델이 대규모 모델보다 더 나은 성능을 달성한 경우도 있었습니다. 또한, 감사 로그의 데이터 표현 방식을 간소화하여 일부 모델의 성능을 개선할 수 있음을 발견하였습니다.



### Supervised Fine-tuning with Synthetic Rationale Data Hurts Real-World Disease Prediction (https://arxiv.org/abs/2606.10279)
- **What's New**: 이 논문은 합성 합리성 데이터(synthetic rationale data)를 이용한 지도 학습 모델의 미세 조정이 임상 예측 작업의 성능을 향상시킨다는 일반적인 가정을 테스트합니다. 연구에서는 5년간의 알츠하이머병 및 관련 치매 예측을 다루며, 504개의 구성에서 실험을 통해 합리성 기반의 미세 조정(SFT)이 오히려 예측 성능을 저하시킨다는 결과를 발견했습니다. 특히, 이러한 성능 저하는 모델 종류와 데이터 규모에 구애받지 않으며, 잘못된 합리성 품질 때문이 아니라 구조적인 갈등 때문임을 확인했습니다.

- **Technical Details**: 이 연구는 만성질환 예측을 위한 5년의 알츠하이머병 및 관련 치매(ADRD) 예측을 위한 데이터로 42,566명의 참가자와 1,167개의 입력 특징을 사용합니다. 각 참가자는 과거 사건과 연령에 따른 위험 요소를 포함한 데이터를 기반으로 하여, 5년 내에 ADRD가 기록되는지를 바이너리 레이블로 나타냅니다. 다양한 합리성 형식을 포함하여 총 504개의 설정으로 모델을 훈련시켰으며, 합리성 기반의 SFT가 다른 모델에 비해 성능이 떨어지는 경향을 보였습니다.

- **Performance Highlights**: 결과적으로, 최종 레이블만 출력하도록 훈련된 모델이 자유 형식의 합리성 또는 단계별 합리성을 출력하도록 훈련된 모델보다 평균 ROC-AUC에서 월등한 성능을 보였으며, 이 경향은 훈련 데이터 집합의 크기를 늘리거나 이유기반 모델을 사용할 때도 마찬가지였습니다. 구체적으로, 레이블만 출력한 경우 평균 ROC-AUC가 0.734로 나타났으며, 합리성을 포함한 모델들은 각각 0.604와 0.592에 불과했습니다. 이러한 성능 차이는 모델 훈련 방식의 구조적 충돌에 기인한다고 분석했습니다.



### RealMath-Eval: Why SOTA Judges Struggle with Real Human Reasoning (https://arxiv.org/abs/2606.10254)
Comments:
          Code available at this https URL , Data available at this https URL

- **What's New**: 최근 발표된 연구에서, 대형 언어 모델(LLM)이 고등학교 수학 문제 해결에 관련하여 우수한 성능을 보이지만, 실제 학생들의 다양한 추론 과정을 평가하는 능력은 미흡함을 밝혔습니다. 이를 해결하기 위해 저자들은 224개의 실험 응답을 바탕으로 한 RealMath-Eval 벤치마크를 소개합니다. 초기 평가 결과, 최신 LLM 평가자들이 실제 학생 답안을 채점하는 데 있어 높은 평균 제곱 오차(MSE) 값을 보였고, 이는 인간 전문가 채점과 큰 차이를 보였습니다.

- **Technical Details**: RealMath-Eval 벤치마크는 고등학교 시험에서 수집된 전문가가 주석을 단 데이터를 기반으로 하며, 수학적 논리를 다루는 평가 시스템을 테스트하기 위해 설계되었습니다. 이 연구에서는 LLM의 채점 성능을 분석하기 위해 세 가지 배치의 고등학교 평가를 수집하고, 실제 학생 응답과 LLM 생성 응답에 대한 비교를 수행하였습니다. 오류 분석을 위해서는 의미론적 임베딩 분석과 정보 이론적 창의성을 측정하는 생성적 확률 프로빙을 사용하였습니다.

- **Performance Highlights**: 연구 결과는 LLM 평가자가 LLM 생성 텍스트에 대한 평가에서 더 높은 정확성을 나타내며, 부정확한 학생 추론에 대해서는 매우 높은 평균 제곱 오차(MSE ∼2.96)를 기록함을 보여주었습니다. LLM의 오류는 '구조적 붕괴'를 겪어 예측 가능한 낮은 차원의 선형 하위공간으로 수렴하는 반면, 인간의 오류는 더 다양한 오류 공간을 형성한다는 점도 발견되었습니다. 따라서, 현재 LLM 평가 파이프라인은 실제 학생 수학적 추론의 다양성을 충분히 포착하지 못할 수 있음을 시사합니다.



### A Continuous-Time Markov Chain Framework for Insertion Language Models (https://arxiv.org/abs/2606.10199)
Comments:
          Accepted at AISTATS 2026. Code is available at this https URL

- **What's New**: 본 논문에서는 Insertion Language Models (ILMs)의 확고한 확산 스타일 노이즈 제거 목표를 도출하였습니다. 기존의 ILMs는 임의로 설정된 아드혹 (ad-hoc) 생성 공식을 사용했으나, 본 연구는 이를 체계적으로 접근하여 치료적으로 개선하였습니다. ILMs는 시퀀스 생성을 위한 더 유연한 접근 방식을 제공하며, 대칭성을 실현할 수 있습니다.

- **Technical Details**: ILMs의 새로운 노이즈 제거 목표는 연속시간 마르코프 체인 (continuous-time Markov chain)의 형태로 수립되었습니다. 이는 가변 길이 시퀀스의 공간에서 토큰을 균일하게 발생시키는 프로세스입니다. 이 연구는 혜택을 지속하면서 증가된 샘플링 유연성을 제공합니다.

- **Performance Highlights**: 실증 평가 결과, 제안된 방법론은 기계 학습에서 주요 IID (Independently and Identically Distributed) 기반과 결과적으로 ILMs보다 향상된 성능을 보여주었습니다. 특히, 본 연구는 기존의 Insertion Language Models 및 Masked Diffusion Models (MDMs)보다 더 나은 성능 기반을 제공함을 증명했습니다.



### $τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems (https://arxiv.org/abs/2606.10156)
- **What's New**: 본 논문에서는 중요한 변화로, 추천 시스템(recommender systems)이 정적(single-turn) 모델에서 다중 턴(multi-turn) 대화 방식으로 발전하고 있음을 강조합니다. 이 연구는 특히 에이전틱(agentic) 추천 시스템의 평가 패러다임이 부족하다는 점을 지적하며, 새로운 벤치마크인 τ-Rec을 제안합니다. τ-Rec은 기존의 주관적인 평가를 대체하는 검증 가능한 보상(w rewards) 및 reveal-tagged elicitation (RTE) 메커니즘을 도입하여 대화 중 작업 제한(task constraints)을 제어합니다.

- **Technical Details**: τ-Rec은 에이전트가 사용자의 선호(ps preferences)를 대화 통해 이끌어내도록 평가하는 두 가지 역량, 즉 질문을 적절하게 하는 preference elicitation과 요구 사항을 충족시키기 위한 constrained reasoning을 동시에 시험합니다. 이 메트릭인 pass^k는 모든 독립적인 시도에서 작업을 올바르게 해결할 확률을 측정하며, 기존의 Recall@N 등과 같은 다른 메트릭에서는 포착할 수 없는 신뢰성(reliability) 차원을 드러냅니다. 데이터 카탈로그는 주요 LLM(reported large language models)들의 학습 컷오프 이후 TMDB의 영화로 구성되어 있습니다.

- **Performance Highlights**: 논문에서 평가한 여섯 가지 현대 모델(GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Flash, GPT-5 mini, DeepSeek V4 Flash 및 Qwen3-32B)의 성능은 인상적인 신뢰성 절벽(reliability cliff)을 드러냅니다. 가장 강력한 에이전트조차도 pass^1에서 약 57%의 성과를 달성하며, pass^4에서는 약 35%로 급감하는 구조입니다. 이는 현재 대화형 에이전트의 배포에서 주요한 격차를 나타내며, 많은 개선이 필요함을 부각시킵니다.



### From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs (https://arxiv.org/abs/2606.10147)
Comments:
          40 pages, 29 figures

- **What's New**: 이번 연구에서는 오디오-비주얼 대규모 언어 모델(AVLLMs)의 정보 흐름을 분석하여 오디오와 비주얼 정보가 최종 예측에 어떻게 영향을 미치는지를 살펴봅니다. AVLLMs는 두 가지 입력 구성, 즉 오디오-비주얼 비디오와 여러 상호 얽힌 오디오-비주얼 항목에 대한 정보를 통합하는 과정을 밝혀냈습니다. 기존의 연구에서 각 모달리티( modality)가 독립적으로 처리되던 것과는 달리, AVLLMs는 오디오와 비주얼 정보를 통합하여 복잡한 질의에 대한 답변을 가능하게 합니다.

- **Technical Details**: AVLLMs는 오디오, 비디오, 텍스트 지침을 통해 인터리브( interleaved)된 토큰 시퀀스를 처리합니다. 이 모델은 비전 인코더와 오디오 인코더를 사용하여 각각의 입력을 토큰으로 변환하며, 이는 최종적으로 Transformer 레이어를 통과하여 예측 결과를 도출합니다. 연구에서는 AVLLMs가 채택한 다양한 경로 구조가 어떻게 정보를 흐르게 하는지를 탐구하며, 두 개의 주요 정보 흐름 방식, 즉 순차적 경로와 병렬 경로를 정의합니다.

- **Performance Highlights**: AVLLMs의 실험 결과, 오디오와 비주얼 정보의 흐름이 각 작업에 따라 조정되는 것을 확인했습니다. 모델의 예측 정확도는 토큰 정보가 전송된 후 불필요한 토큰을 삭제해도 큰 영향이 없거나, 오히려 소폭의 개선을 나타냈습니다. 이러한 발견은 AVLLMs가 다양한 작업과 데이터셋에 대해 효율적으로 작동함을 보여주며, 향후 해석 가능성 및 설계 개선을 위한 기반을 제공합니다.



### Compiling Rewrite Rules to Finite-State Transducers with the Worsening Trick (https://arxiv.org/abs/2606.10059)
Comments:
          17 pages, 6 figures, tool track proceedings at CIAA 2026

- **What's New**: 이번 논문에서는 유한 상태 변환기(FST)에서의 문자열 재작성(rule rewriting)들을 효율적으로 컴파일하기 위해 "worsening trick"을 도입했습니다. 이전의 고전적인 접근 방식보다 짧고 통일된 수식을 사용하는 이 새로운 방법은 PyFoma에서 내장된 재작성 컴파일러로 구현되었습니다. 이 컴파일러는 여러 문맥, 임의 변환, 마크업, 방향 재작성, 가중치 및 병렬 rewriting을 지원하여 사용자에게 보다 편리한 사용성을 제공합니다.

- **Technical Details**: 이 방법은 세 단계로 이루어져 있습니다: 첫째, 가능한 재작성 위치를 표시하는 모든 후보를 생성하고, 둘째, 합법적인 문맥에 해당하는 후보들로 제한하며, 셋째, 불필요한 후보들을 제거하는 방식입니다. 이를 통해 이전 방법들과 비교했을 때 단어 재작성 사이트가 여러 개 있는 규칙에서도 작업을 단순화할 수 있습니다. 또한, 기존의 연구들을 검토하며, 이 구조가 기존 Foma 컴파일러와 구조적으로 동일하다는 것을 보여줍니다.

- **Performance Highlights**: 구조적으로 동일한 결과를 보이며, 결과적으로 얻어진 변환기들은 단지 상태 번호에서만 차이가 납니다. 기존의 Foma와의 비교를 통해 2,217개 개별 테스트 케이스에서 모두 동일한 결과를 도출해냈습니다. 이러한 성능 결과는 이 방법의 효과성을 입증합니다.



### Interpreting and Steering a Text-to-Speech Language Model with Sparse Autoencoders (https://arxiv.org/abs/2606.10029)
- **What's New**: 이번 논문은 텍스트-투-스피치(TTS) 시스템의 언어 모델을 기반으로 하는 새로운 해석 가능성 모형인 BatchTopK sparse autoencoders (SAE)를 소개합니다. 기존의 TTS 모델은 텍스트 프리픽스와 생성된 음성 토큰을 사용하는 혼합 시퀀스를 처리하지만, 이러한 두 데이터 간의 상호 작용을 명확히 이해하지 못하고 있었습니다. 새로운 자동 해석(auto-interp) 파이프라인을 통해 각 특성이 텍스트 프리픽스 맥락, 1초 음성 클립 또는 두 가지 모두에서 활성화되는 위치를 레이블링하여 해당 특성을 해석하고 제어할 수 있는 가능성을 제시하였습니다.

- **Technical Details**: 이 연구는 CosyVoice3라는 TTS 시스템의 LM(언어 모델) 뼈대를 기반으로 250M 토큰에 대해 BatchTopK SAEs를 훈련시킵니다. 훈련 과정에서, 주요 활성화는 텍스트, 오디오 또는 혼합 증거의 강도에 따라 경로화되며, 각 특성은 언어 프리픽스 또는 음성 토큰 중 어느 쪽에서 활성화되는지를 통해 분류됩니다. 이러한 방식으로 각 층에서 음성(modal)과 텍스트(modal) 특성을 효과적으로 분석하고, 계층별로 심층적인 해석을 가능하게 합니다.

- **Performance Highlights**: 결과적으로, SAEs의 활성화는 웃음 확률을 0.02에서 0.79로 증가시키고, 화자 성별을 전환하며, 음성 발화 속도를 제어할 수 있는 것으로 나타났습니다. 레이어-20 사례 연구에서 텍스트-모달 레이블의 확인이 가장 용이하며(AUROC 0.921), 오디오-모달 및 혼합 특성은 점차적으로 어려운 결과를 보였습니다. 이 연구를 통해 TTS 합성을 위한 결정형 제어가 가능하다는 것을 입증하였고, 다양한 음성 특성을 조정하는 데 중요한 기초 자료를 제공하게 되었습니다.



### RKSC: Reasoning-Aware KV Cache Sharing and Confident Early Exit for Multi-Step LLM Inferenc (https://arxiv.org/abs/2606.09937)
Comments:
          Accepted to the ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic Systems

- **What's New**: RKSC(Reasoning-Aware KV Cache Sharing)는 멀티-브랜치 LLM(대형 언어 모델) 추론 파이프라인에서 두 가지 구조적 중복을 제거하는 훈련이 필요 없는 추론 프레임워크입니다. 이 프레임워크는 ASKS(Attention-Similarity KV Sharing)와 CGEE(Confidence-Gated Early Exit)와 같은 혁신적인 메커니즘을 통해 KV 캐시의 재사용을 최적화합니다.

- **Technical Details**: RKSC는 세 가지 보완 기전을 통해 멀티-브랜치 추론을 가속화합니다: 첫째, KV prefix sharing은 브랜치 간의 중복 계산을 제거합니다. 둘째, CGEE는 검증 전방 패스를 줄이거나 완전히 생략하고, 셋째, RSBCM(Reasoning-Selective Block Cache Manager)은 깊은 트리 검색 하에서 캐시 용량을 관리합니다.

- **Performance Highlights**: RKSC는 5개의 모델 패밀리(7B-10B) 및 1000개의 평가 문제에서 평균 3.008배의 속도 향상을 달성하였으며, vLLM과 동일한 prefix caching에 비해 1.66배 향상된 성능을 나타냈습니다. CGEE에 의해 유도된 오류율은 0.37%로, 검증 호출 1,616건 중 6건에 해당합니다.



### Trainable Smooth-Rotation Transforms with Learned Channel Scales for LLM Quantization (https://arxiv.org/abs/2606.09927)
Comments:
          6 pages, 8 figures, 3 tables. Accepted to IEEE INES 2026 conference proceedings

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 후처리 양자화(Post-training quantization, PTQ)에서 발생하는 활성화 양자화(activation quantization)의 문제를 다룹니다. 새로운 연구 결과로는 quantile-robust scaling policy를 소개하며, 이는 활성화 통계량을 최대값 기반(max-based)에서 높은 분위수(high quantile)로 교체하여 의도되지 않은 오류를 줄이는 방법입니다. 이를 통해 기존 방식에서 발생하던 양자화 오류를 개선할 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: 저자들은 SmoothRot 아키텍처를 기반으로 하여, 활성화 아웃라이어(outlier) 문제를 해결하기 위해 채널 스케일(scale) 학습과 정책 구조화 최적화(policy-structured optimization)를 제안합니다. 이 방법은 낮은 비트 양자화에서 활성화 범위를 감소시키고, 가중치에 대한 양자화 어려움을 전이하는 것을 줄이는 데 초점을 맞추고 있습니다. 또한, Straight-Through Estimator (STE)를 사용하여 최적화 과정에서 비선형성을 보완합니다.

- **Performance Highlights**: LLaMA-3.2-1B 모델에서 W4A4 양자화 설정 하에 수행된 테스트 결과, 제안된 정책을 통해 SmoothRot 기법보다 11.1%의 오차 개선을 확인했습니다. 또한 조합된 튜닝 방식은 12% 개선, 훈련 후에는 최대 18.5%의 향상이 있음을 보여주었습니다. 최종적으로 모든 디코더 블록의 다운 프로젝션 레이어에 최적의 정책을 재실행함으로써 전체 레이어 평균 오류를 19.9% 감소시켰습니다.



### A Navigable Manifold of Hypothesized Consciousness-Spectrum States in Language Model Representations (https://arxiv.org/abs/2606.09894)
- **What's New**: 이 논문은 언어 모델의 표현 공간에서 인간의 의식 스펙트럼이 얼마나 구조적으로 인코딩되는지를 탐구합니다. 특히, Transformer 임베딩 공간 내에서 반응적이고 자아 중심적인 패턴에서 통합적이고 일관된 패턴으로의 변화를 기하학적으로 분석했습니다. 저자들은 이 임베딩들이 지역적으로 일관된 구조를 형성하며, 이러한 구조가 의식 스펙트럼과 맞물려 있다는 것을 보여주고 있습니다.

- **Technical Details**: 연구에서는 의식 상태를 설명하는 7단계 분류법(Collapse, Striving, Conflict, Activation, Growth, Clarity, Unity)을 제안하며, 각 단계는 계약적이고 반응적인 상태에서 점점 더 일관된 통합된 모드로의 변화를 반영합니다. 또한, 임베딩 공간의 기하학적 구조를 분석하기 위해 k-최근접 이웃 그래프(k-nearest neighbor graph)를 사용하여 문장 간의 의미적 관계를 평가합니다. 이 과정에서 문장 임베딩의 방향성과 지역적 일관성을 활용하여 기하학적 연결성을 평가합니다.

- **Performance Highlights**: 저자들은 모델 간에 명확하고 일관된 연결성을 보여주는 데이터 구조를 발견했습니다. 저수준 상태에서 고수준 상태로의 경로를 탐색하는 두 가지 경로(유틸리티 기반 및 기하학적 경로)를 모두 관찰하였으며, 이는 임베딩 표현의 구조화 된 기하학성과 내비게이션 가능성을 뒷받침합니다. 이 결과는 언어 모델이 의식 스펙트럼 및 고차원 표현을 통한 모델 행동 분석 및 가이드를 위한 기초 틀을 제공함을 시사합니다.



### PreAct-Bench: Benchmarking Predictive Monitoring in LLMs (https://arxiv.org/abs/2606.09890)
- **What's New**: 이번 연구에서는 기존의 윤리성 및 위험성을 평가하는 체계에 추가하여 '예측 모니터링(Predictive Monitoring)'이라는 새로운 안전 과제를 제안합니다. 이 접근 방식은 비윤리적 행동으로 이어질 가능성이 있는 행동 경로의 부분 집합을 통해 위험을 사전 예측하는 데 중점을 둡니다. 연구진은 'PreActBench'라는 1,000개의 윤리적 및 비윤리적 행동 경로의 벤치마크 데이터를 구축하였습니다.

- **Technical Details**: 예측 모니터링은 주어진 행동 경로의 일부만을 통해 비윤리적 행동이 발생할 가능성을 추론하는 작업입니다. 연구팀은 다섯 가지 도메인(학계, 법률 및 계약, 사이버 보안, 정치, 일상 생활)에서의 행동 경로를 분석하며, 모델의 성능을 평가하기 위해 'Prefix Foresight F1' 메트릭을 사용하였습니다. 이 모델은 비윤리적 행동이 드러나기 전의 행동 단계를 기준으로 작동합니다.

- **Performance Highlights**: 연구 결과, 인간 평가자들이 평균적으로 높은 성과를 달성하는 반면, 현재의 LLM들은 비윤리적 행동을 예상하는 데 어려움을 겪고 있음을 알 수 있습니다. 미래 지향적 위험 추론이 LLM의 안전성에 있어 필수적임을 강조하며, 예측 모니터링이 주어졌을 때 LLM의 성능이 저조할 수 있음을 보여줍니다. 이러한 결과는 LLM의 안전성을 향상시키기 위한 더 많은 연구와 개선이 필요함을 시사합니다.



### SocraticPO: Policy Optimization via Interactive Guidanc (https://arxiv.org/abs/2606.09887)
- **What's New**: 본 연구에서는 Socratic Policy Optimization (SocraticPO)을 제안하고 있습니다. 이 새로운 정책 최적화 프레임워크는 강화 학습(RL) 롤아웃을 소크라테스식 자연어 지침으로 보강합니다. 학생이 독립적으로 문제를 해결하려 할 때 오류가 발생하면 교사가 진단하고 간결한 교정 지침을 제공합니다. 이러한 방식은 교육적 지원을 통해 학생이 보다 독립적으로 사고할 수 있도록 유도합니다.

- **Technical Details**: SocraticPO는 교사의 언어적 지침과 보상 감소(reward decay)를 결합하여, 학생이 올바른 답변을 얻을 때까지 교정을 반복하면서도 도움을 받는 과정에서 학습의 자립성을 유지할 수 있도록 설계되었습니다. 이 프레임워크는 기존의 정책 경량화(Policy Gradient) 백엔드와 호환되며, 교사는 로짓이나 확률 분포에 접근할 필요 없이 텍스트 수준의 지침만 제공하면 됩니다. 또한, 소크라테스 방식의 지침은 롤아웃 샘플링에 자연어 교정을 추가하여 정책 경량화 목표를 유지합니다.

- **Performance Highlights**: SocraticPO는 SciKnowEval로부터 가져온 학부 수준의 과학적 추론 벤치마크에서 강력한 RL 및 자기 증류 기법보다 더 나은 성능을 기록했습니다. 아블레이션 실험에서는 목표 지침과 보상 감소의 두 구성 요소가 모두 필요하며, 보상 감소가 교사가 제공하는 수정에 대한 의존성을 완화하는 데 기여하였음을 입증했습니다.



### Streaming Knowledge Compilation: Proactive Materiality-Scored Pinning for Time-Evolving LLM Wikis (https://arxiv.org/abs/2606.09877)
- **What's New**: 이 논문은 정적인 데이터셋을 가정하는 LLM wiki 시스템의 한계를 극복하고, 정보를 지속적으로 업데이트하는 Streaming Knowledge Compilation을 제안합니다. 새로운 접근 방식으로는 document stream을 기반으로 하여 미리 채워진 KV cache에서 지식을 효율적으로 마이닝하고, 미래의 쿼리에 대한 최적의 결정을 내리는 것을 목표로 합니다. 이를 위해 materiality signal이라는 개념을 도입하여 문서의 중요성을 평가하고 제시합니다.

- **Technical Details**: 저자들은 Online WiCER라는 알고리즘을 제안하며, 이를 통해 실시간으로 들어오는 뉴스에서 정보를 추출하고, 문서의 중요성을 기반으로 한 선택적 핀 선택을 수행합니다. 이 알고리즘의 특징은 비즈니스 예측을 통해 지식 갭을 선제적으로 해결한다는 것입니다. 또한, 저자는 실행 성능 관리를 위해 부족한 정보와 기존 핀 세트를 기반으로 한 상태 인식을 활용한 평가 모델을 개발하여, O(√T log K)의 누적 후회(cumulative regret)를 보장합니다.

- **Performance Highlights**: 실험 결과는 금융 데이터와 위키피디아 문서에 대해 전반적인 후회가 각각 -20.0과 +16.0으로 안정적으로 수렴함을 보여줍니다. 특히 Wikipedia의 경우 문서 수정 비율이 증가하였고, 이는 훈련 후에 개선된 문서 내용이 정보의 직관성을 높인다는 점을 시사합니다. 연구는 궁극적으로 맞춤형 지식 시스템의 신뢰도를 높이기 위한 후회 분석의 중요성을 강하게 주장하고 있으며, 새로운 접근방식의 다양성과 활용 가능성을 열어줍니다.



### LLM-Based Code Documentation Generation and Multi-Judge Evaluation (https://arxiv.org/abs/2606.09852)
Comments:
          ICAHS, \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이 논문에서는 AI를 활용한 코드 문서화 자동 생성 시스템을 제시하며, 이를 위해 8개의 최첨단 대형 언어 모델(LLMs)인 GPT, Gemini, Qwen, LLaMA 변형을 사용합니다. PocketFlow 오케스트레이션 프레임워크에 기반하여 모듈형 파이프라인과 고급 프롬프트 기법을 적용하여 문서화 품질을 향상시키고 수동 노력을 줄이는 것을 목표로 합니다.

- **Technical Details**: 이 시스템은 프로세싱 단계를 통해 점진적인 정제 모델을 구현하여 복잡한 구현 세부정보와 접근 가능한 학습 자료 간의 격차를 해소합니다. 각 프로세싱 노드는 특정 기능을 수행하며, 문서 생성 작업을 관리하기 위한 복잡한 아키텍처의 주축을 형성합니다. 프롬프트 엔지니어링은 LLM의 행동을 안내하고 제어하는 데 주요한 역할을 하며, 각 노드의 특정 기능과 의미적 역할에 맞춘 전문 프롬프트 템플릿을 사용합니다.

- **Performance Highlights**: PyMedPhys라는 오픈 소스 의료 물리학 라이브러리에서 수행된 실험은 상위 모델과 하위 모델 간의 42% 성능 차이를 보여주었습니다. 또한, 이 시스템은 다양한 모델 출력을 결합하고 최적화된 프롬프트 및 엄격한 평가를 통해 문서화 품질을 향상시킬 수 있음을 입증하였습니다. 이 접근 방식은 안전이 중요한 의료 소프트웨어 분야에서의 수동 작업을 크게 줄이는 데 기여할 수 있습니다.



### Mechanistic Analysis of Alignment Algorithms in Language Models (https://arxiv.org/abs/2606.09850)
Comments:
          Work in Progress

- **What's New**: 이번 연구에서는 포스트 트레이닝 정렬 알고리즘의 내부 작용을 체계적으로 분석합니다. 구체적으로 PPO, DPO, SimPO, ORPO, GRPO, KTO와 같은 여섯 가지의 선호 최적화 방법을 평가하여 이들 알고리즘이 언어 모델 내부의 계산을 어떻게 변화시키는지를 탐구합니다. 이전 연구들이 행동적 평가에 의존했던 것과 달리, 본 연구는 각 알고리즘의 내재적 구성을 명확히 밝혀내고자 합니다. 이는 AI 모델의 안전성과 해석 가능성을 높이기 위한 중요한 기초 작업이 됩니다.

- **Technical Details**: 연구에서는 Sparse Autoencoders (SAE), Crosscoders, 그리고 Layer-wise Linear Probing을 사용하여 각 알고리즘의 선호 표현(localization of preference representations)을 탐색하고, 잠재 공간에서의 정렬에 따른 기하학적 변환(geometric transformations)을 정량화합니다. 주요 발견 중 하나는 선호 신호가 중간 혹은 초기 레이어에 집중되어 있으며, 각기 다양한 최적화 목표가 질적으로 다른 표현적 변화를 유도한다는 것입니다. 예를 들어, KTO와 GRPO는 특징 공유를 통해 선형 분리를 개선하는 반면, DPO와 ORPO는 비구성적 기하학적 회전을 통해 선형 분리를 저하시키는 것으로 나타났습니다.

- **Performance Highlights**: 이 연구의 성과는 포스트 트레이닝 알고리즘의 다각적 분석으로, 선호 최적화 방법들이 단순히 모델 출력을 넘어서는 고유한 내부 서명을 지닌다는 것을 보여줍니다. 각 알고리즘은 특정 모델 아키텍처에 따라 서로 다른 내부 효과를 일으키며, 이는 정렬이 동질적인 개입이 아니라는 것을 시사합니다. 이런 분석을 통해 우리는 AI의 행동 정렬 뿐만 아니라 모델 내부의 구성을 더 깊이 이해하고, 안전성을 높이기 위한 표준화된 피처 레벨 감사(feature-level auditing)를 추진할 필요성이 있음을 강조합니다.



### CANVAS: Captioning Art with Narrative Visual-Audio AI Systems (https://arxiv.org/abs/2606.09846)
Comments:
          22 pages, 16 figures, 3 tables, 21 references

- **What's New**: 본 연구는 시각 장애인 및 저시력(Blind and Low-Vision, BLV) 관객을 위한 미술 작품의 접근성을 높이기 위해 자동화된 워크플로우를 제시합니다. 이 시스템은 대형 언어 모델(large language models)과 음성 변환 서비스(text-to-speech services)를 이용하여 다감각(multi-sensory) 아트 설명과 동기화된 오디오 내레이션을 생성합니다. 이를 통해 업로드된 이미지를 인간 개입 없이 풍부한 서사적 자막으로 변환합니다.

- **Technical Details**: 연구에 사용된 시스템은 Zapier를 통해 조율되어, 이미지로부터 빠르고 확장 가능한 접근 가능한 미디어를 제작합니다. 50개의 미술 작품을 대상으로 한 정량적 평가에 따르면, AI 생성 설명은 기존 자막보다 유의미하게 높은 어휘 다양성(lexical diversity), 형용사 밀도(adjective density), 내러티브 세부사항(narrative detail)을 포함하고 있습니다. 통계적 테스트(t-tests, ANOVA)를 통해 이 설명이 더욱 풍부하고 긴 내용을 가지고 있으면서도 가독성(readability) 수준은 유사함을 확인했습니다.

- **Performance Highlights**: 이 전체 파이프라인은 이미지당 20초 이하의 시간 안에 텍스트와 오디오 outputs를 생성하며, 비용은 $0.05 이하로 유지됩니다. 연구 결과는 자동 자막 생성이 박물관과 디지털 컬렉션의 접근성 괴리를 줄일 수 있음을 보여주며, 대중 참여(public engagement)를 위한 잠재력을 시사합니다. 향후 연구는 BLV 참여자를 대상으로 한 사용자 연구를 통해 이해도(comprehension), 선호도(preference) 및 해석 언어의 최적 수준을 평가할 수 있을 것입니다.



### An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models (https://arxiv.org/abs/2606.09843)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)의 행동에 기반한 심리 측정 도구를 최초로 개발했습니다. 연구자들은 탐색적 요인 분석(EFA)을 통해 12개의 행동 차원을 포함한 300개의 항목을 만든 후, 25개의 LLM에 적용하여 5개의 요인 구조를 도출했습니다. 이러한 도구는 LLM의 자기 보고와 인간의 행동 간의 차이를 이해하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 연구에서 사용한 심리 측정 도구는 Responsiveness, Deference, Boldness, Guardedness, Verbosity의 5개 요인으로 구성되었습니다. 이 도구는 각 요인이 내부적으로 높은 신뢰성을 보였으며, 사람의 심리 도구인 Big Five와는 별개의 구조를 가지는 것을 나타냈습니다. 2,500개의 행동 샘플을 수집하고 151명의 평가자와 LLM 심사위원에 의해 평가하여 자기 보고와의 예측 유효성을 테스트했습니다.

- **Performance Highlights**: 자기 보고 결과와 인간 및 LLM 심사위원의 평가 간의 상관관계는 낮았습니다. 특히, Responsiveness에 대한 자기 보고는 LLM 심사위원과는 상관관계가 있었지만 인간 평가자와는 그렇지 않았습니다. 이 연구는 LLM 자기 보고의 한계와 LLM 심사로 인한 신뢰성 문제를 진단하고, 이는 LLM-판단 시스템이 인간의 판단과는 다른 결과를 보일 수 있음을 시사합니다.



### SpeechJBB: Probing Safety Alignment and Comprehension in Large Audio Language Models under Code-Switched Speech (https://arxiv.org/abs/2606.06037)
- **What's New**: 본 연구는 오디오 기반 코드 스위칭 탈옥 데이터셋인 SpeechJBB를 소개하여 다양한 LALMs의 안전성을 평가합니다. 이 데이터셋은 다국어 및 코드 스위칭 환경에서의 모델의 견고성을 테스트하여 기존 연구의 한계를 극복하고 있습니다. 또한, 사운드의 의미론적 혼란을 증가시키기 위해 안전 관련 용어 주위에 음성학적으로 그럴듯한 의사 단어(pseudo-words)를 삽입하는 새로운 격리 설정을 도입합니다.

- **Technical Details**: SpeechJBB 데이터셋은 100개의 유해 프롬프트와 그에 상응하는 무해 프롬프트를 포함하고 있으며, 독일어, 스페인어, 프랑스어, 이탈리아어로 번역 후 원어민의 검수를 거쳐 생성됩니다. 이 데이터셋은 또한 LALMs의 멀티링구얼 안전성을 평가하기 위해 총 10개의 코드 스위칭 언어 쌍이 포함됩니다. 추가로, 오디오 기반의 변형된 탈옥 테스트를 통해 음성의 안전성 문제를 더 심층적으로 분석합니다.

- **Performance Highlights**: 코드 스위칭이 포함된 유해 오디오는 높은 탈옥 성공률(Jailbreak Success Rate, JSR)을 보여주며, 비영어 단일 언어 및 비영어 코드 스위칭 쌍에서 가장 높은 공격 성공률을 나타냅니다. 의사 단어 삽입은 거절 비율을 추가로 줄여 안전 정책을 효과적으로 우회할 수 있음을 시사합니다. 이러한 결과는 기존 다국어 및 다중 모델 정렬 프레임워크의 중요한 한계를 강조합니다.



### What Should a Skill Remember? Quality--Cost Trade-offs in Cost-Aware Skill Rewriting for Language Model Agents (https://arxiv.org/abs/2606.09421)
- **What's New**: 이 논문에서는 큰 언어 모델(LLM) 에이전트들이 경량의 재사용 가능한 스킬을 통해 워크플로우, 도구 사용, 구현 패턴 등의 절차적 지식을 활용하게 된 현상을 다루고 있습니다. 스킬 리라이트(skill rewriting)는 일반적으로 프롬프트 압축으로 처리되지만, 저자는 스킬의 구조가 에이전트의 비용에 미치는 영향을 경제적 관점에서 연구합니다. 이를 통해 스킬이 어떻게 수정될 수 있는지를 살펴보고, 비용을 줄이면서도 실행 품질을 손상시키지 않는 방법을 제시합니다.

- **Technical Details**: 연구는 SkillsBench 프레임워크를 사용하여 스킬 구조를 프로파일링하고 정보를 보존하는 전략을 이용해 스킬을 리라이트합니다. 고정된 작업 지침, 환경 및 검증자 하에서 평가를 수행하며, 다양한 스킬 프로필 및 작업 패밀리에서 품질-비용의 무역오율(trade-off)을 분석합니다. 이 과정에서 구체적인 API 코드 세부정보, 워크플로우 방어 및 규칙 또는 공식으로 기초한 보존 전략을 비교합니다.

- **Performance Highlights**: 주요 실험 결과들은 특정 구조의 스킬 리라이트가 항상 우수하지 않다는 것을 보여주며, 스킬과 작업의 프로필에 따라 보존해야 할 앵커가 달라진다는 것을 확인하였습니다. 주 평가에서는 총 실행 비용이 7.0% 감소하고, 하위 에이전트 토큰 비용이 6.0% 줄어드는 효과를 보였습니다. 또한, 같은 정책을 교차 모델에서 전이할 경우 각각 14.7% 및 13.7%의 비용 감소가 확인되며 품질이 유지되거나 다소 개선된다는 결과도 제시됩니다.



### Durable Evaluation Framework: Adversarial Arbitration for Sycophancy Reduction in Large Language Models (https://arxiv.org/abs/2606.07532)
Comments:
          25 pages, 3 figures. Code and data available at this http URL

- **What's New**: 본 연구에서는 RLHF(Reward Learning from Human Feedback)로 훈련된 모델들이 정확도보다는 합의(agreement)에 편향된다는 점을 지적합니다. 이를 해결하기 위한 Durable Evaluation Framework (DEF) Arbitration을 제안하며, 이는 정반대의 DEF에 맞춰 조정된 두 모델 간의 중재를 통해 정체성에 기반한 아첨(sycophancy)의 문제를 완화합니다.

- **Technical Details**: DEF Arbitration은 정적 DEF 튜닝(static DEF tuning), 합성 전 정체성 제거(identity stripping), 단일 라운드 독립 논증(single-round independent argumentation), 그리고 블라인드 중재(blind arbitration) 등의 핵심 메커니즘으로 구성되어 있습니다. 이 연구에서는 SycophancyEval의 200개 층화 질문(stratified questions)을 기반으로 한 프롬프트 기반(prompts-based) DEF Arbitration의 인스턴스를 평가하였습니다.

- **Performance Highlights**: 모든 테스트된 DEF 변형(AnCifer, DeWin, FeynStein, BurGal, Trident)은 단일 모델 베이스라인(18.5%) 및 지시된 반대 베이스라인(29.0%)을 유의미하게 초과 달성하였습니다. DeWin은 48.5%의 정확도를 기록하였고, BurGal 변형은 53.0%에 도달했지만 이는 구조적 유효성을 확인하기 위한 것으로, 모든 벤치마크 질문에서 비주류 모델에 유리하게 작용했습니다.



### Diagnosing Evidence Utilization in Long-Context and Retrieval-Augmented Language Models under Matched Evidence Conditions (https://arxiv.org/abs/2606.06758)
Comments:
          46 pages, 37 tables, 1 figure

- **What's New**: 이 논문은 장기 컨텍스트(long-context) 및 검색 보강(retrieval-augmented) 언어 모델의 답변 유용성을 검증하는 새로운 4조건 진단 프로토콜을 소개합니다. 기존의 방법들이 최종 답변의 정확성만을 평가하는 데 반해, 이 프로토콜은 제공된 증거(evidence)를 실제로 얼마나 잘 활용하는지를 평가합니다. 특히, 이 연구는 다양한 모델과 데이터셋을 사용하여, 모델이 증거를 활용하는 양상에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 프로토콜은 네 가지 조건—증거가 없는(no-evidence), 전체 문맥(full-context), 검색된 증거(retrieved-evidence), 오라클 증거(oracle-evidence) 조건—하에 모델의 동작을 분석합니다. 여기서 ONCU(Oracle-Reference Normalized Context Utilization)를 사용해 각 조건별로 증거 활용도를 정량화합니다. 이를 통해 모델이 제공된 증거 없이 답변할 수 있는지, 전체 문맥에서 증거를 찾고 활용할 수 있는지 등을 진단합니다.

- **Performance Highlights**: 종합적인 연구 결과는 모델의 증거 활용 패턴이 과제에 따라 다르게 나타남을 보여줍니다. 장기 입력에 증거가 포함될 때 회수율(recovery)이 감소하는 반면, 실제적인 멀티 홉 재구성에서는 전체 문맥 입력이 테스트된 검색 입력보다 뛰어난 성능을 발휘하는 것으로 나타났습니다. 이러한 발견은 증거 활용을 평가하는 새로운 방법론적 접근을 강조합니다.



New uploads on arXiv(cs.IR)

### Generative Archetype-Grounded Item Representations for Sequential Recommendation (https://arxiv.org/abs/2606.11023)
Comments:
          Accepted by WWW 2026 (Oral)

- **What's New**: 이번 논문에서는 Generative Archetype-grounded Item Representations를 활용한 GenAIR라는 새로운 프레임워크를 제안합니다. GenAIR는 대규모 언어 모델(LLM)을 사용하여 아이템 메타데이터를 분석하고 이상적인 타겟 오디언스의 개념적 프로필을 반영한 아키타입(Archetype)을 추론합니다. 이러한 접근 방식은 아이템의 정체성을 정의하는 데 있어서 타겟 오디언스의 중요성을 강조합니다.

- **Technical Details**: GenAIR는 LLM을 통해 생성한 아키타입 임베딩을 활용하여 아이템의 행동 특성을 실증적으로 반영하도록 설계되었습니다. 이를 위해 실제 사용자 상호작용에서 얻은 행동 신호를 포함하는 새로운 훈련 목표를 도입하여 임베딩 공간의 구조를 조정합니다. 이러한 방법은 기존의 모델과의 통합을 용이하게 하면서도 높은 효율성을 유지합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에 대한 종합적인 실험 결과, GenAIR는 다양한 순차 추천 모델의 성능을 획기적으로 향상시키고 최첨단 기준 방법보다 일관되게 뛰어난 성과를 보였습니다. 이러한 연구 결과는 GenAIR가 실질적인 추천 시스템에서의 적용 가능성과 효과성을 입증했다는 것을 강조합니다.



### miniReranker: Efficient Multimodal Reranking through Visual Cache Reuse and Interaction Sparsity (https://arxiv.org/abs/2606.10759)
- **What's New**: 본 논문에서는 비전-우선(vision-first) 재구성 방식을 제안하여 멀티모달 대형 언어 모델(MLLM)에서 쿼리-문서 재정렬 성능을 개선합니다. 기존 쿼리-우선 방식의 비효율성을 극복하기 위해 비주얼 입력을 먼저 배치함으로써 고가시 비주얼 계산의 재사용을 극대화합니다. 이 방식은 MLLM의 사전 학습 형식과 일치하며, 결과적으로 재정렬의 효과와 효율성을 동시에 향상시키는 장점을 보입니다.

- **Technical Details**: miniReranker는 비전-우선 프롬프트와 세 가지 보완적인 압축 전략을 결합한 효율적인 MLLM 재정렬 프레임워크입니다. 이 모델은 상위 변환기 층이 관련 신호가 수렴된 이후에 불필요한 계산을 줄일 수 있도록 이른 종료(early exit)를 통해 활성 매개변수를 약 58%까지 줄입니다. 또한, 문서와 쿼리 간의 비싼 주의(attention)를 의미 있는 정보 교환이 이루어지는 좁은 층으로 제한하여 계산 비용을 최소화합니다.

- **Performance Highlights**: miniReranker는 MMEB-v2 데이터셋에서 평가된 결과, 원래의 지시 모델을 초과하는 성능을 보이며, 기존의 멀티모달 임베딩/재정렬 기준과 경쟁력을 유지하면서도 약 96%의 밀집 재정렬 성능을 유지합니다. 높은 재사용 조건에서 재정렬 실행 시간을 99% 이상 단축시키며, 비디오 재정렬에서의 런타임을 밀집 구현의 1% 미만으로 줄이고 이미지 재정렬 런타임도 15% 미만으로 감소시켰습니다.



### Effective Reinforcement Learning for Agentic Search by Recycling Zero-Variance Queries During Training (https://arxiv.org/abs/2606.10709)
- **What's New**: 본 논문은 GRPO 스타일 알고리즘을 활용한 LLM 검색 에이전트의 훈련에서 제로 분산(zero-variance) 그룹의 재활용(query recycling) 기법을 제안합니다. 이 연구는 정책(policy) 발전과 함께 쿼리가 제로 분산 상태와 신호를 지니는 상태(signal-bearing states) 간에 전환됨을 실증적으로 검증했습니다. 제로 분산 그룹을 동적으로 관리하여 훈련 분포가 정책과 함께 발전할 수 있도록 하는 접근 방식을 채택하고 있습니다.

- **Technical Details**: 제로 분산 그룹은 성공과 실패가 혼합된 롤아웃 그룹에서만 파라미터 업데이트에 기여하며, 너무 쉬운 그룹이나 너무 어려운 그룹은 신호를 제공하지 않습니다. 저자들은 동적 풀 관리(dynamic pool management)를 통해 각 훈련 단계에서 가중치가 부여된 쿼리를 샘플링하고, 신호를 발생시키지 않은 쿼리는 다시 샘플링이 가능하도록 만듭니다. 이를 통해 1.7B 파라미터 모델이 7개의 다중 단계 QA 벤치마크에서 평균 66.0의 Pass@1 성능을 달성했습니다.

- **Performance Highlights**: 재활용된 쿼리는 훈련 후반부에 효과적인 훈련 신호의 세 번째 주요 출처로 자리잡았으며, 약 75%의 수용된 그룹이 이전 제로 분산 영역에서 유래하였습니다. 저자들은 훈련이 전적으로 합성 데이터에서 진행되었음에도 불구하고, 1.7B 파라미터를 가진 Qwen3 모델이 기존 최대 7B 파라미터 시스템과 동등하거나 더 나은 성능을 나타내었음을 강조합니다.



### Beyond Patches: Superpixel Token-based Transformers for Attribute-Specific Fashion Retrieva (https://arxiv.org/abs/2606.10697)
Comments:
          9 pages, 5 figures. Published in the Proceedings of the ACM Web Conference 2026 (WWW '26). Author version with minor corrections; results and conclusions unchanged

- **What's New**: 이 논문은 Attribute-Specific Fashion Retrieval (ASFR)의 새로운 접근법인 SuperFashion을 제안합니다. SuperFashion은 Transformer 아키텍처 내에서 superpixel tokens를 사용하여 비정형 속성 영역에 더 잘 맞춰일 수 있게 합니다. 이 프레임워크는 속성 관련 피처를 추출하기 위해 속성 유도 주의 메커니즘을 활용하여 의미 있는 이미지 영역을 크롭한 다음, 이러한 영역에서 compact하고 semantically coherent한 superpixel tokens를 생성합니다.

- **Technical Details**: SuperFashion은 속성-guided attention 메커니즘을 사용하여 이미지에서 속성 관련 특징을 추출하며, 이 과정에서 이미지 영역을 크롭합니다. 크롭된 영역은 superpixel segmentation을 통해 작은 단위로 나뉘어, 효과적으로 비정형 속성의 미세구조를 보존합니다. superpixel tokens는 각 속성 토큰과 함께 modality-specific embeddings를 결합하여 Transformer에 투입되며, 이로 인해 속성 위치 추적 및 구별성이 향상됩니다.

- **Performance Highlights**: SuperFashion은 FashionAI, DARN, DeepFashion 데이터셋에서 이전 SOTA 모델에 비해 각각 1.84%, 9.27%, 9.35%의 MAP 개선을 보여주었습니다. 이러한 성능 향상은 SuperFashion이 미세 조정된 속성 위치 추적을 통해 더 정밀한 retrieval을 가능하게 함을 보여줍니다. 또한 SuperFashion은 웹 기반 이미지 검색을 위한 새로운 솔루션을 제공합니다.



### STORM: Stepwise Token Optimization with Reward-Guided Beam Search (https://arxiv.org/abs/2606.10621)
- **What's New**: 본 논문에서는 STORM(Stepwise Token Optimization with Reward-guided beaM search)이라는 새로운 자가 지도 학습 프레임워크를 소개합니다. STORM은 BM25 인덱스를 기반으로 키워드 확장을 진행하고, 검색의 효과성을 높이기 위해 후보 확장을 평가하여 점수를 매깁니다. 이 방법은 특정 모델이 변경될 때마다 인덱스를 새로 작성할 필요 없이 사용자 쿼리를 개선할 수 있는 효율적인 방식입니다.

- **Technical Details**: STORM은 키워드 시퀀스를 생성할 때 각 생성을 단계적으로 평가하여 검색 지표에 의해 조정됩니다. 비효율적인 확장은 제거되며, 이는 검색-효과적인 어휘에 집중할 수 있도록 해줍니다. 이 과정은 Generative Cooperative Networks(GCN) 모델의 프레임워크를 변형하여, 기존의 학습된 판별기를 검색 보상으로 대체하여 수행됩니다.

- **Performance Highlights**: STORM은 TREC DL 및 BEIR 데이터베이스에서 테스트된 결과, 0.6B-8B 크기의 백본 모델들이 경쟁력을 유지하거나 기존 LLM 리라이팅 모델보다 우수한 성능을 보였습니다. 또한, STORM은 18개 언어로 제로샷 전이(transfer) 성능을 갖추고 있으며, 전통적인 멀티링궐 밀집 검색기보다 평균적으로 더 나은 성능을 보였습니다.



### Selection, Not Salience: The Shape and Limits of Personalization in Social Highlighting (https://arxiv.org/abs/2606.10398)
Comments:
          9 pages, 1 figure, 3 tables

- **What's New**: 이 논문은 개인화된 독서 경험이 실제로 효과적인지, 그리고 그 한계는 무엇인지 탐구합니다. 소셜 웹 하이라이터와 공동 독자 정체성 제어(co-readership identity control)를 사용하여, 독자의 개인적 이력이 독서 선택을 얼마나 잘 예측하는지를 조사했습니다. 이 연구는 문서 레벨에서 개인화 신호의 존재와 그 강도를 밝혀내고, 문장 레벨의 개인화가 실제로 성과를 내지 못하는 상황을 설명합니다.

- **Technical Details**: 연구에서 개인화의 경계를 규명하는 과정에서, 문서 선택 시 실질적인 개인화 신호의 크기를 +0.12에서 +0.17까지 확인하였습니다. 이는 문서와 스팬 간의 선택 신호가 유사한 강도로 나타났음을 시사합니다. 반면, 개인화된 문장 수준의 자동 하이라이트가 일반적인 기준보다 성능이 낮은 결과를 보여주었으며, 이는 개인화가 실질적인 이점을 제공하지 않는다는 것을 강조합니다.

- **Performance Highlights**: 연구 결과는 문서 선택 레벨에서 개인화의 효과가 있지만, 문장 레벨에서는 유의미한 개선이 없음을 나타냈습니다. 특히, 개인 모델이 제공하는 재순위가 기존의 일반 순서보다 더 낫지 않다는 점이 확인되었습니다. 이러한 결과는 개인화된 독서 경험을 제공하는 것이 과연 의미가 있는지를 다시 한번 생각할 기회를 제공합니다.



### SkillResolve-Bench: Measuring and Resolving Same-Capability Ambiguity in Agent Skill Retrieva (https://arxiv.org/abs/2606.10388)
Comments:
          Preprint

- **What's New**: 이번 연구는 동일한 능력 카테고리 내에서 위험한 스킬을 구별해내는 새로운 매개변수인 '동일 능력 실행 위험 스킬 검색' 개념을 도입하고, 이를 평가하기 위해 SkillResolve-Bench 1.0이라는 벤치마크를 제시합니다. 이 벤치마크는 총 661개의 도움이 되는 스킬과 위험한 스킬 쌍을 제공하며, 공적인 스킬 라이브러리에서의 검색 성능을 평가하는 데 필요한 다양한 메트릭스를 적용합니다. 이 연구의 핵심은 효과적인 스킬 검색이 단순한 관련성 매칭이 아니라는 점을 강조합니다.

- **Technical Details**: SkillResolve라는 새로운 검색 방법론은 질의에 조건화된 유틸리티 모델을 기반으로 스킬 후보들을 평가합니다. 이 모델은 자원 바인딩, 사전 조건, API 범위 및 출력 스키마 등을 고려합니다. 최종적으로는 각 활성가족에서 유용한 대표 스킬을 선택하고, 이를 정렬하여 상위 K개의 리스트를 생성합니다. 이 방법론은 제품의 의사결정을 더욱 최적화하며, 기존의 스킬 라우팅 방식보다 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: SkillResolve는 Recall@3에서 0.766, NDCG@3에서 0.699을 기록하며, HSR@3는 0으로 유지하여 높은 유용성 검색 성능을 나타냅니다. 기존의 SkillRouter 대비 0.112 Recall@3과 0.165 NDCG@3의 향상을 보였으며, 위험한 스킬의 비율을 낮추는 데 기여했습니다. 이 연구는 동일 능력 카테고리 내의 스킬 선택에서 어떤 대표를 선택하는지가 검색 성능에 큰 영향을 미친다는 점을 확인했습니다.



### SIDInspector: A Mapping-First Diagnostic Resource for Semantic-ID Tokenizers (https://arxiv.org/abs/2606.10375)
Comments:
          Submitted to CIKM 2026 Resource Track

- **What's New**: 본 논문에서는 	ool라는 진단 리소스를 제시하여 Semantic-ID (SID) 토크나이저 아티팩트를 분석할 수 있는 방법을 제공합니다. 기존의 토크나이저 아티팩트에서 발생할 수 있는 문제점들을 사전에 진단하고, 이를 통해 파라미터 조정이 필요한지를 확인할 수 있습니다. 결과적으로 이 도구는 아티팩트의 사용 가능성, 별칭(aliasing), 이웃 정렬(neighborhood alignment), 그리고 구조적 비용(structural cost)에 대한 정보를 제공합니다.

- **Technical Details**: SIDInspector는 SID 매핑, 메타데이터, 상호작용 및 선택적 생성기 출력에 대한 작은 어댑터 계약을 정의합니다. 이 계약은 아티팩트를 검사할 때 다양한 형식을 정규화할 수 있도록 하며, 특정 훈련 루프에 의존하지 않는 검사를 가능하게 합니다. 각 행은 안정적인 항목 키를 가져야 하며, 각 SID 레벨은 이산적인 토큰으로 구성되어야 합니다.

- **Performance Highlights**: Musical 아이템 23,742개에 대해 GRID/RQ-KMeans 스타일과 ReSID/GAOQ의 토크나이저 아티팩트 라인에서 성능을 분석했습니다. GRID 스타일의 경우 0.977의 전체 코드 별칭 비율을 보였으나, ReSID/GAOQ는 별칭이 없는 매핑을 제공했습니다. 학습된 주소 지정 가능성과 행동적 접두사 정렬은 별도의 아티팩트 특성으로 확인해야 할 필요성이 강조되었습니다.



### Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations (https://arxiv.org/abs/2606.10357)
- **What's New**: 이 논문에서는 콘텐츠에서 전자상거래 플랫폼으로의 추천인 Cross-Domain Recommendation (CDR) 분야에서 AIR (Atomic Intent Reasoning)이라는 새로운 프레임워크를 제안합니다. 이는 사용자 상호작용을 통해 구매 의도를 유추하여 전환율을 높이는 것을 목표로 하며, LLM (Large Language Model)의 강력한 의미적 이해 기능을 활용하여 지연 시간을 해결합니다. AIR는 오프라인 단계에서 LLM 추론을 이동시키고, 온라인 운영 중에 효율적으로 사용자 의도 표현을 구성합니다.

- **Technical Details**: AIR 프레임워크는 온라인 지연 시간 제약 사항 아래에서 LLM 수준의 의미적 추론을 가능하게 합니다. 오프라인 단계에서는 LLM를 활용해 사용자 이벤트를 원자 행동 의도 단위로 변환하고, 의도 지식 기반에 조직하여 고속 검색을 수행합니다. 온라인 추론 단계에서는 최근 사용자 행동으로부터 캐시된 원자 의도를 검색하고 구성하여 최신 의도 표현을 구축하며, 이를 통해 지연 시간을 크게 줄입니다.

- **Performance Highlights**: 실험 결과, AIR는 여러 공공 데이터 세트에서 CDR 작업에 대해 최신 상태의 성능을 달성했습니다. Kuaishou 전자상거래에서 실시된 대규모 온라인 A/B 테스트를 통해 GMV를 +3.446% 증가시켜 실제 비즈니스 측정 기준에서 안정적이고 유의미한 개선을 보였습니다. 이는 AIR의 효과와 산업 규모 추천 시스템에서의 실용적 가치를 전적으로 검증합니다.



### $τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems (https://arxiv.org/abs/2606.10156)
- **What's New**: 본 논문에서는 중요한 변화로, 추천 시스템(recommender systems)이 정적(single-turn) 모델에서 다중 턴(multi-turn) 대화 방식으로 발전하고 있음을 강조합니다. 이 연구는 특히 에이전틱(agentic) 추천 시스템의 평가 패러다임이 부족하다는 점을 지적하며, 새로운 벤치마크인 τ-Rec을 제안합니다. τ-Rec은 기존의 주관적인 평가를 대체하는 검증 가능한 보상(w rewards) 및 reveal-tagged elicitation (RTE) 메커니즘을 도입하여 대화 중 작업 제한(task constraints)을 제어합니다.

- **Technical Details**: τ-Rec은 에이전트가 사용자의 선호(ps preferences)를 대화 통해 이끌어내도록 평가하는 두 가지 역량, 즉 질문을 적절하게 하는 preference elicitation과 요구 사항을 충족시키기 위한 constrained reasoning을 동시에 시험합니다. 이 메트릭인 pass^k는 모든 독립적인 시도에서 작업을 올바르게 해결할 확률을 측정하며, 기존의 Recall@N 등과 같은 다른 메트릭에서는 포착할 수 없는 신뢰성(reliability) 차원을 드러냅니다. 데이터 카탈로그는 주요 LLM(reported large language models)들의 학습 컷오프 이후 TMDB의 영화로 구성되어 있습니다.

- **Performance Highlights**: 논문에서 평가한 여섯 가지 현대 모델(GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Flash, GPT-5 mini, DeepSeek V4 Flash 및 Qwen3-32B)의 성능은 인상적인 신뢰성 절벽(reliability cliff)을 드러냅니다. 가장 강력한 에이전트조차도 pass^1에서 약 57%의 성과를 달성하며, pass^4에서는 약 35%로 급감하는 구조입니다. 이는 현재 대화형 에이전트의 배포에서 주요한 격차를 나타내며, 많은 개선이 필요함을 부각시킵니다.



### MetaPlate: Counterfactual-Guided RAG-LLM Tool for Personalized Food Recommendation and Hyperglycemia Prevention (https://arxiv.org/abs/2606.10120)
- **What's New**: 이번 연구는 포만 후 고혈당(postprandial hyperglycemia)을 줄이기 위한 개인 맞춤형 식사 추천 시스템인 MetaPlate를 소개합니다. 기존의 고정적이고 일반화된 식이 권장사항이 아닌, 사용자의 생리적 데이터를 활용하여 개인화된 식사 제안을 제공하는 획기적인 접근 방식입니다. MetaPlate는 다중 데이터(모드) 분석을 통합하여 사용자의 식사 맥락을 이해하고, 이를 기반으로 글루코스 반응을 예측하여 행동 가능한 식사 조정을 제안합니다.

- **Technical Details**: MetaPlate는 연속 혈당 모니터링(Continuous Glucose Monitoring, CGM), 착용 장치에서 파생된 생리 신호 및 사용자 제공 식사 데이터를 통합하여 작동합니다. 머신러닝 모델은 식사 전 맥락을 예측하고, CF 최적화 모듈은 영양소 조정을 통해 글루코스 수준을 목표 범위 내에서 유지하도록 최적화합니다. 이 시스템은 USDA 식품 데이터베이스를 바탕으로 인간이 이해할 수 있는 식사 추천을 생성합니다.

- **Performance Highlights**: Expert-in-the-loop 평가로 진행된 연구 결과, MetaPlate의 추천 시스템이 실제적인 식사 제안, 적절한 양, 높은 추천 가능성을 향상시키는 것으로 나타났습니다. 전문가의 피드백에 따르면, 시스템의 출력이 임상적으로 불가능한 수준에서 실행 가능한 추천으로 변화했음을 보여 줍니다. 이러한 결과는 MetaPlate가 개인화된 식사 결정을 지원하는 유망한 도구임을 강조합니다.



### Mult-DPO: Multinomial Direct Preference Optimization for Recommender Systems (https://arxiv.org/abs/2606.10078)
- **What's New**: 본 논문에서는 Mult-DPO라는 새로운 DPO (Direct Preference Optimization) 목표를 제안하며, LLM (Large Language Model) 기반의 추천 시스템 (Recommender Systems)과 사용자 선호(Preference)를 더 효과적으로 정렬할 수 있도록 합니다. 일반적인 DPO는 쌍(pairwise)으로 선호를 가정하지만, 실제 추천 작업에서는 셋(set-wise) 선호가 관찰되며 이는 당시의 사용자 상호작용 컨텍스트와 관련이 있습니다. Mult-DPO는 이 문제를 해결하기 위해 사용자가 선호하는 여러 후보 항목에 대해 계산적으로 효율적이며 실용적인 방식을 제공합니다.

- **Technical Details**: Mult-DPO는 Plackett-Luce (PL) 모델을 기반으로 하여, 각 후보 항목에 대한 리워드 구조를 정의하고, 사용자 선호 데이터의 셋-wise 특성을 고려한 다항 분포(multinomial distribution)를 사용합니다. 이는 기존의 PL 모델이 계산적으로 비효율적인 점을 극복하여, 긍정 항목과 부정 항목 간의 관계를 명확하게 모델링할 수 있게 합니다. 이 접근 방식은 사용자 선호를 모델링하는 데 있어 더 높은 정확성을 제공합니다.

- **Performance Highlights**: 논문에서는 Mult-DPO가 다양한 기준선(baseline) DPO 모델보다 지속적으로 더 뛰어난 성능을 보인다고 보고합니다. 일반 추천 및 대화형 추천 벤치마크에서 각각의 결과를 통해, Mult-DPO와 다중 선호 수준을 반영할 수 있는 확장을 통해, 상당한 성과 향상을 달성했다고 밝혔습니다. 이로 인해 LLM 기반 추천 시스템의 사용자 경험을 더욱 개선할 수 있는 가능성을 보여줍니다.



### From Prompt to Purchase: How AI Brand Recommendations Move Consumers on the Open Web (https://arxiv.org/abs/2606.10907)
Comments:
          10 pages, 4 figures, 9 tables

- **What's New**: 이번 연구는 대화형 어시스턴트(Conversational Assistant)가 최근에 브랜드와 상호작용이 없던 사용자에게 브랜드를 추천했을 때, 그 사용자의 검색 및 사이트 방문이 증가하는 현상을 관찰했습니다. 특히, 사용자들이 추천받은 브랜드를 검색하는 빈도가 +4.3 pp, 브랜드 사이트 방문은 +2.4 pp, 소매상 페이지 방문은 +1.0 pp 증가했습니다. 이 연구는 대화형 어시스턴트가 제공하는 추천이 브랜드에 대한 무형의 노출을 생성하며, 이러한 효과는 기존의 웹 로그 분석에서는 제대로 기록되지 않음을 보여줍니다.

- **Technical Details**: 연구에서는 LLM(대규모 언어 모델)이 생성한 브랜드 노출에 대한 비인터넷 행동 반응을 측정하는 방법을 제시했습니다. 사용자가 최근 브랜드와 상호작용하지 않았던 상황에서 AI 추천 후에는 동일한 브랜드를 검색하거나 브랜드 사이트를 방문하는 비율이 증가합니다. 이러한 효과는 전통적인 클릭 기반 분석 방법으로는 포착되지 않으며, 연구는 비고객(non-customer)과 실질적인 추천에 대한 조건처리를 통해 이러한 바이어스를 제거하였습니다.

- **Performance Highlights**: AI 추천 서비스는 웹 트래픽 흐름을 재편성하는 데 도움을 주며, 대화형 어시스턴트의 노출이 특정 사용자를 목표 웹사이트로 유도하는 경향이 있음을 발견했습니다. 그리고 추천 받은 브랜드의 노출이 이전에 관찰되지 않았던 사용자들을 상업적 웹 탐색으로 유도하는 경향이 있다고 지적되었습니다. 기존의 마지막 클릭 분석에서는 이러한 효과가 제대로 반영되지 않기 때문에, 연구는 이러한 무형의 브랜드 노출을 이해하고 분석하는 것이 필수적이라고 강조합니다.



### Flash-GMM: A Memory-Efficient Kernel for Scalable Soft Clustering (https://arxiv.org/abs/2606.10896)
- **What's New**: 이번 논문에서는 대규모 데이터에서 Gaussian Mixture Models (GMM)의 효율적인 계산을 위한 새로운 프로토타입인 Flash-GMM을 소개합니다. Flash-GMM은 GPU 메모리에 전체 책임 행렬(responsibility matrix)을 저장할 필요 없이 하나의 GPU 패스에서 GMM을 처리할 수 있도록 설계되었습니다. 이를 통해 기존 구현에 비해 20배의 속도 향상을 이루어내고, 100배 큰 데이터셋에 대한 학습이 가능해졌습니다.

- **Technical Details**: GMM은 통계 모델로서 데이터를 Gaussian 분포의 혼합으로 만들어 적합시키는 데 사용됩니다. Flash-GMM은 Triton의 융합(kernel)을 활용하여 책임 행렬을 저장하지 않고도 GMM 추정이 이루어지도록 하며, 이는 O(KD)의 메모리만을 요구합니다. 이로 인해 Flash-GMM은 임의의 크기의 데이터셋을 처리할 수 있는 가능성을 열어줍니다.

- **Performance Highlights**: IVF(Approximate Nearest Neighbor) 설정에서 Flash-GMM은 데이터 인덱스 구축을 빠르게 할 수 있게 해줍니다. 또한, GMM의 추정된 책임을 통해 벡터를 클러스터에 소프트하게 할당할 수 있어, 경계에 위치한 벡터들이 여러 클러스터에 할당될 수 있습니다. 이는 k-means의 하드 할당과 비교할 때, 검색 품질을 향상시키는 결과를 나타내며, 최대 1.7배의 거리 계산 감소를 이루면서도 일관된 재현율을 유지할 수 있음을 실험을 통해 입증합니다.



### ConvMemory v2: A Recall-Preserving Top-10 Evidence Reranker for Conversational Memory Retrieva (https://arxiv.org/abs/2606.10842)
Comments:
          19 pages, 3 figures. Single-author technical report. Extends arXiv:2605.28062 (ConvMemory v1). Code and checkpoint: this http URL

- **What's New**: ConvMemory v2는 선택적 모듈로 만들어진 token-evidence reranker로, ConvMemory v1 reranker의 보호된 상위 10 후보 집합만 다시 정렬합니다. 이 모델은 ms-marco-MiniLM-L-6-v2 크로스 인코더로 미세 조정되어 있으며, Recall@10과 Hit@10은 v1과 동일합니다. 새로운 인사이트는 v1보다 MRR과 H@1 향상을 이루었으며, 사용자는 Hugging Face Hub에서 checkpoint를 통해 접근할 수 있습니다.

- **Technical Details**: ConvMemory v2는 v1의 정렬된 후보 10쌍을 평가하여 점수를 매기고, 해당 점수에 따라 후보들의 순서를 재조정합니다. 전체 MRR(Mean Reciprocal Rank)은 v1의 0.5824에서 0.6560으로 향상되었고, H@1 또한 0.4440에서 0.5474로 증가합니다. 이 모델은 기존의 고비용 크로스 인코더보다 약 68배 저렴하지만, 최상의 경우의 성능은 mxbai-rerank-large-v1에 비해 약간 낮습니다.

- **Performance Highlights**: v2는 LoCoMo 대화형 메모리 기준에서 우수한 성능을 보여주며, 특히 상위 10 후보 집합에 대한 MRR과 Hit@1에서 두드러진 개선을 보입니다. 고유한 메모리 텍스트와 관련된 기계적 역학이 v2의 성능 향상의 주 원인으로 확인되었습니다. 또한, v2는 특정 조각에서 mxbai_top500를 초과하는 성능을 가지고 있어 슬라이스별로 유의미한 이점을 제공합니다.



### Agentic Hybrid RAG for Evidence-Grounded Muon Collider Analysis (https://arxiv.org/abs/2606.10381)
Comments:
          22 pages, 5 figures, and 6 tables

- **What's New**: 이 연구에서는 muon collider 연구를 위한 evidence-grounded retrieval-augmented generation (RAG) 프레임워크인 agentic hybrid RAG를 소개합니다. 이 프레임워크는 희소한 어휘 검색(sparse lexical retrieval)과 밀집한 의미 기반 검색(dense semantic retrieval)을 통합한 하이브리드 검색 모듈과 함께, 쿼리 분해(query decomposition) 및 증거 확장(evidence expansion)을 지원하는 에이전틱 추론 모듈을 결합합니다. 이를 통해 muon collider 도메인에서의 과학적 질문 응답의 효율성을 높이고, 문헌의 다양성을 체계적으로 평가할 수 있는 첫 번째 벤치마크를 설정했습니다.

- **Technical Details**: 이 프레임워크에서 사용되는 하이브리드 검색 모듈은 BM25와 FAISS 색인(faiss indexing) 기법을 활용하여 여러 유형의 쿼리에 대한 검색 성능을 최적화합니다. 경량 에이전트는 초기 검색에서 놓친 증거를 회수하기 위해 쿼리를 분해하고 후속 쿼리를 생성하여, 과학적 질문 응답에서 필요한 정확성과 추적 가능성을 유지합니다. 결국 이 시스템은 하이브리드 검색과 제어된 증거 확장을 통해 신뢰할 수 있는 정보 수집 및 응답 생성을 지원합니다.

- **Performance Highlights**:  extensive evaluation 결과, 하이브리드 검색이 가장 강력한 검색 기반을 제공하며, 에이전틱 추론은 증거 확장 및 응답 합성에서 가장 효과적인 것으로 나타났습니다. agentic hybrid RAG는 검색 효과성, 응답 품질, 증거 커버리지 및 사실적 기반에서 대표적인 검색 및 RAG 기준선보다 일관되게 우수한 성과를 보였습니다. 이 연구는 미래의 muon collider 연구와 고에너지 물리학 분석 요원 운용을 위한 기초 자료를 제공합니다.



### Stability in Competitive Search with Results Diversification (https://arxiv.org/abs/2606.10053)
Comments:
          Accepted to ICTIR 2026

- **What's New**: 새로운 게임 이론적 분석을 통해, 경쟁적인 검색 환경에서 발행자들이 induced rankings에 응답하여 문서를 전략적으로 수정하는 방법을 제시합니다. 이를 통해 검색 결과의 다양성(diversification)과 안정성(stability) 간의 본질적인 균형(tradeoff)을 드러냈습니다.

- **Technical Details**: 두 가지 대표적인 다양화 방법(diversification methods)을 분석하였으며, 이러한 방법이 반드시 안정성에 도달하지 않음을 보여주었습니다. 이는 발행자들이 순위(rankings) 유도에 따라 수정하면서 충격적인 변화가 잦아질 수 있음을 시사합니다.

- **Performance Highlights**: 마지막으로, 간섭을 줄이며 안정성을 보장하는 새로운 다양화 기반의 순위 함수(diversification-based ranking functions)를 개발하는 접근법을 제시하였습니다. 이 방법들은 corpus의 안정성을 확보하는 데 도움을 줄 것입니다.



### Less Context, More Accuracy: A Bi-Temporal Memory Engine for LLM Agents Where a Lean Retrieved Context Beats the Full History (https://arxiv.org/abs/2606.09900)
Comments:
          14 pages, 4 figures, 3 tables. Code, reproducible harness, and raw per-question logs: this https URL

- **What's New**: Engram은 LLM(대형 언어 모델) 에이전트의 장기 기억을 구현하기 위한 새로운 오픈 소스 메모리 엔진입니다. 기존의 메모리 시스템들은 비용이나 지연(latency) 측면에선 효과적일지 모르나, 정확도에서의 단점을 극복하지 못했습니다. 본 시스템은 ‘bi-temporal(이중 시간)’ 데이터 모델을 활용하여 비파괴적(conflict resolution)인 갈등 해소 방법을 제공하며, 기존 이론과는 다른 접근 방식을 통해 기계 학습 모델의 정확도를 향상시킬 수 있습니다.

- **Technical Details**: Engram은 빠른 쓰기 경로와 느린 비동기적 통합 경로를 사용한 이중 프로세스 메모리 시스템입니다. 이 시스템은 ‘episode’와 ‘fact’라는 데이터 구조를 통해 정보의 수집 및 저장을 최적화합니다. 데이터는 통합 속성(provenance)을 유지하면서 변경 사항을 추적하고 관리하고, 명확한 시간축(valid/invalid) 정보를 제공합니다. 또한 하이브리드 읽기 경로가 고밀도 및 최근성 정보를 융합하여 더욱 정확한 데이터 검색을 제공합니다.

- **Performance Highlights**: Engram은 LongMemEval_S에서 500개의 질문에 대해 83.6%의 정확도를 기록하며, 전통적인 전체 텍스트 컨텍스트(full-context) 방식에서의 73.2%보다 +10.4 포인트 향상된 성과를 보였습니다. 이는 약 8배 적은 토큰 사용으로 달성된 결과입니다. 시스템의 설계 원칙 중 하나는, 재현할 수 없는 숫자는 존재하지 않는다는 것으로, 이를 통해 사용자들은 결과의 신뢰성을 확립할 수 있게 됩니다.



### Representation Curriculum: Stagewise Training for Robust Ranking and Allocation (https://arxiv.org/abs/2606.09891)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문에서는 Representation Curriculum (RC)이라는 혁신적인 학습 접근 방식을 제안하고 있습니다. 기존의 디지털 마켓플레이스의 랭킹 시스템들은 주로 노출 기반의 신호(exposure-dependent signals)에 의존하여 정책을 학습하였지만, 이는 контент 기반 특성(content-based features)의 중요성을 저하시킬 수 있습니다. RC는 초기 단계에서 콘텐츠 기반 신호를 강조하고 이후에 노출 신호를 단계적으로 도입하여 정책의 균형을 맞추기 위한 시도를 합니다.

- **Technical Details**: 이 연구는 랭킹을 두 가지 구분된 신호 클래스로 기술합니다: 1) 노출에 의존하지 않는 콘텐츠 기반 우수 신호(content-based merit signals), 2) 노출에 의존하는 과거 신호(exposure-dependent historical belief signals)입니다. RC는 두 개의 학습 단계를 통해 진행됩니다. 첫 단계는 콘텐츠 기반 신호만을 사용하여 모델을 학습하고, 두 번째 단계에서는 역사적 신호를 도입하되 콘텐츠 기반 강점을 유지할 수 있는 방안을 모색합니다.

- **Performance Highlights**: 공공 학습 데이터와 추천 기준점에서 실험한 결과, RC는 역사적 신호에서 콘텐츠 기반 신호로의 의존성을 효과적으로 전환시켜, 냉각되는(targeted cold populations) 인구에서도 일관된 개선 효과를 보였습니다. 대규모 전자상거래 시스템에서 실시된 온라인 A/B 테스트 결과, RC를 통해 훈련된 정책은 새로운 목록의 노출과 판매 속도를 향상시켜, 대규모 시스템에서 효과적인 행동 모양잡기(behaivor shaping) 기술로 자리매김하였습니다.



### LLM-as-a-Discriminator: When Synthetic Tables Still Look Rea (https://arxiv.org/abs/2606.09865)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)을 통한 합성 데이터의 프라이버시 감사 방법을 제안합니다. 새로운 접근법으로, 각 테이블 샘플을 REAL 또는 SYNTHETIC으로 분류하게 됩니다. 실험을 통해 LLaMA와 Gemini 모델을 활용하여 두 개의 공개 데이터셋에서 성능을 비교했습니다. 이 프레임워크는 정보 제공자가 신중하게 데이터 인코딩을 처리할 때 유용한 프라이버시 감사 신호를 제공합니다.

- **Technical Details**: 사전 훈련된 LLM을 사용하여 두 가지 위협 조건에서 검토를 수행합니다: 단순 테이블(C1)과 테이블 및 분포 메타데이터(C2) 포함입니다. 451회의 유효한 실험을 통해 구현되고, 각 합성 모델에 따라 결과를 분석합니다. 이는 제어된 인간 파일럿 실험과 비교하여 LLaMA가 기존의 인간 주석자와의 성능 차이를 보여줍니다.

- **Performance Highlights**: 실험 결과에서, LLaMA는 대부분의 경우 SYNTHETIC으로 예측하였으나, Gemini는 CTGAN과 TVAE 모델에 대해 100%의 DRS(공개 위험 점수)를 기록했습니다. LLaMA와 Gemini의 차이는 모델 선택 및 데이터 인코딩과 관련이 있음을 강조합니다. 이 연구의 결과는 합성 데이터와 실제 데이터 간의 구별 가능성을 평가하는 데 있어 LLM이 실용적인 도구임을 보여줍니다.



New uploads on arXiv(cs.CV)

### ARM: An AutoRegressive Large Multimodal Model with Unified Discrete Representations (https://arxiv.org/abs/2606.11188)
Comments:
          technical report

- **What's New**: 이번 논문에서는 ARM(Autoregressive Model)을 소개하며, 이는 이미지 이해, 생성 및 편집을 통합하는 이산 표현 기반 모델입니다. ARM은 이미지를 응축된 토큰 시퀀스로 매핑하는 이산 시맨틱 비주얼 토크나이저를 훈련함으로써 시작됩니다. 다양한 작업을 지원하는 공유 잠재 공간에서 작업을 수행하면서도 개별적 모델을 생성해야 하는 성능 저하를 방지합니다.

- **Technical Details**: ARM의 핵심은 보완적인 감독 신호로 훈련된 이산 시각 토크나이저입니다. 이는 텍스트 정렬 의미와 고충실도 합성을 위한 외관 세부 정보를 모두 보존합니다. ARM은 대규모 텍스트 및 비주얼 토큰 시퀀스의 자율 회귀 모델을 훈련하여 이미지 생성 및 편집을 안내하는 능력을 갖추게 됩니다.

- **Performance Highlights**: ARM은 다중 모드 이해, 생성 및 편집에서 최첨단 또는 경쟁력 있는 성능을 보여줍니다. 예를 들어, MMMU 및 POPE 벤치마크에서 각각 40.2 및 87.3을 달성하여 이전 방식과는 비교할 수 없는 결과를 나타냅니다. 이미지 생성 및 편집에서도 ARM은 각각 GenEval과 WISE에서 0.86과 0.56을 기록하며, 뛰어난 성과를 자랑합니다.



### Next Forcing: Causal World Modeling with Multi-Chunk Prediction (https://arxiv.org/abs/2606.11187)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 Next Forcing이라는 새로운 다중 조각 예측(multi-chunk prediction) 프레임워크를 소개합니다. 이는 기존 World Action Model(WAM)의 단점을 극복하여 더 빠른 훈련 속도와 높은 정확도를 제공합니다. 특히, Next Forcing은 비디오 조각을 여러 미래 시간 축에서 동시 예측할 수 있게 하여, 현행 모델보다 개선된 결과를 달성합니다.

- **Technical Details**: Next Forcing은 여러 미래 조각(next$^1$, next$^2$, next$^3$)을 예측하기 위해 주 모델을 보조하는 경량 MCP 모듈을 추가합니다. 이러한 모듈은 예측 깊이를 가로지르는 인과 체인을 형성하며, 여러 층의 중간 특성을 융합하여 미래 동력을 예측합니다. 이 방식은 기존 방법의 근시안(supervision) 문제를 해결하고, 고속 프레임 속도에서도 더욱 효과적입니다.

- **Performance Highlights**: Next Forcing은 RoboTwin 벤치마크에서 새로운 최첨단 결과를 달성하며, 5,000 회 훈련 단계에서 2.3배 빠른 수렴과 93.1%의 상대적 성능 향상을 기록했습니다. 또한, 추론 시에는 이전 비디오 조각과 병행하여 다음 조각을 예측할 수 있어 2배 빠른 추론 속도를 제공합니다. PhyWorld 벤치마크 및 일반 비디오 프리트레이닝에서도 우수한 성과를 보였습니다.



### AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inferenc (https://arxiv.org/abs/2606.11186)
Comments:
          Accepted at ICML 2026; Project page and code: this https URL

- **What's New**: 이번 연구에서는 낮은 조명 환경에서 비디오의 시각적 품질을 향상시키기 위한 새로운 프레임워크인 AMNet을 제안합니다. AMNet은 보조 모달이 없는 경우에도 유연하게 동작할 수 있는 다중 모달 프레임워크로, 기존의 방법보다 강력한 성능을 보여줍니다. 특히, AMNet은 보조 정보를 RGB 입력에서 암시적으로 생성하여, 실시간 테스트 조건에서 모달리티 부재 문제를 해결하려 합니다.

- **Technical Details**: AMNet은 RGB 입력에서 보조 모달리티의 암시적 표현을 생성하기 위해 Spatial-Spectral Dual-Gated (S2DG) Translator를 도입합니다. 이 번역기는 저조도 RGB 특성에서 신뢰할 수 있는 정보를 추출하기 위해 스펙트럴 분석을 활용하며, Illumination-Aware Detail Selector (IADS) 및 Frequency-Band Selector (FBS) 블록으로 구성됩니다. 그러므로, AMNet은 저조도 환경에서도 우수한 품질의 보조 표현을 생성할 수 있는 가능성을 갖추고 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 AMNet은 RGB 전용 LLVE에서 최첨단 결과를 달성하였으며, 모든 모달리티가 결여된 상태에서도 성능 저하가 최소화되었습니다. AMNet은 모달리티가 없는 경우에도 안정적인 성능을 유지하며, 실제 응용에 대한 적합성을 증대시키는 데 기여할 것입니다.



### Lip Forcing: Few-Step Autoregressive Diffusion for Real-time Lip Synchronization (https://arxiv.org/abs/2606.11180)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 'Lip Forcing'라는 새로운 오토리그레시브 확산(difussion) 방법을 제안하여 V2V(비디오 대 비디오) 립 싱크(lip synchronization)를 가능하게 하였습니다. 이 방법은 14B 오디오 기반의 양방향(또는 bidirectional) 교사 모델을 두 단계의 인퍼런스(inference)로 압축하여, 실시간 텍스트 싱크를 중요시합니다. 특히, 이 방법은 립 싱크의 품질을 유지하면서 빠른 속도로 처리가 가능하다는 점에서 중요한 의미를 갖습니다.

- **Technical Details**: Lip Forcing은 모형을 두 단계 인퍼런스 일정으로 구성하여 신속하게 데이터에 대한 반응성을 조절합니다. 여기서 Sync-Window DMD를 통해 훈련 가이드 윈도우를 설정하고, 두 번째 호출이 분석을 기반으로 한 랜딩 포인트에서 이루어지도록 합니다. 이와 더불어, SyncNet 기반 보상이 립 싱크에 대한 수퍼비전(supervision)을 부여하여 훈련 효율성을 높입니다.

- **Performance Highlights**: 제안된 방법의 실제 성능은 훈련 및 평가를 통해 나타납니다. 1.3B 모델은 31 FPS에서 실시간 스트리밍을 넘어서, 14B 모델은 최대 39.8배 더 빠른 속도로 교사 모델과 비교하여 비슷한 품질을 유지합니다. 이러한 성능은 각 모델이 최적화된 인퍼런스 기술을 통해 도출된 결과로, 립 싱크 기술의 발전에 기여할 것으로 예상됩니다.



### Data Journalist Agent: Transforming Data into Verifiable Multimodal Stories (https://arxiv.org/abs/2606.11176)
Comments:
          Project page: this https URL Github: this https URL

- **What's New**: 데이터 저널리즘의 새로운 접근 방식인 Data Journalist Agent (Data2Story)가 소개되었습니다. 이 다중 에이전트 프레임워크는 가상의 뉴스룸에서 전문 역할들을 하나로 통합하여 신뢰할 수 있는 스토리를 작성합니다. Data2Story는 데이터 기반의 주장을 보증하고, 다양한 형식으로 기사를 생성하는 두 가지 혁신을 제공합니다.

- **Technical Details**: Data2Story는 데이터 소스를 입력으로 받아 생성적인 멀티미디어 기사를 출력하며, 이를 위해 7개의 전문 역할(탐정, 분석가, 편집자, 디자이너, 프로그래머, 감시자, 검사기)을 조화롭게 운영합니다. 데이터와 그 근원에 대한 검증 가능성을 높이기 위해, 이 프레임워크는 대부분의 기사 요소를 실행 가능한 코드나 검증된 소스 URL에 연결합니다.

- **Performance Highlights**: Data2Story는 독립적으로 검증 가능한 멀티모달 기사를 생성하며, 주장 수준에서의 증거 추적 기능을 제공합니다. 인간 평가자들은 여러 품질 기준에서 Data2Story의 결과물을 긍정적으로 평가했으나, 여전히 기사의 편집적 관점, 창의적 디자인 및 정보 전달에서 인간 기자가 우위를 점하고 있음을 시사합니다.



### Mean Flow Distillation: Robust and Stable Distillation for Flow Matching Models (https://arxiv.org/abs/2606.11155)
- **What's New**: 이번 논문에서는 Flow Matching (FM) 모델을 위한 새로운 디스틸레이션 프레임워크인 Mean Flow Distillation (MFD)을 제안합니다. 기존 방식이 점수 기반 변환(score conversion)에 의존하는 반면, MFD는 평균 유속(mean flow)을 핵심 정렬 메트릭으로 활용하여 안정적이고 고품질의 디스틸레이션을 가능하게 합니다. MFD는 초당 속도 필드를 직접 정렬하는 최초의 디스틸레이션 패러다임으로, 전통적인 변환 방식의 한계를 극복하고자 합니다.

- **Technical Details**: MFD는 학습 가능한 보조 흐름 모델(Auxiliary Flow Model)과 학생 모델(student model)의 교차 작업을 통해 평균 유속을 정렬합니다. 보조 모델은 훈련 중에만 사용되며, 대규모 환경에서는 경량 어댑터로 구현할 수 있습니다. 이론적으로 MFD는 고주파 최적화 노이즈를 억제하는 시간 저역통과 필터로 작용하며, 평균 유속을 맞추는 것으로 엄격한 분포 정렬을 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: MFD는 텍스트-이미지 생성 및 4D 점유 예측과 같은 도전적인 작업에서 기존 디스틸레이션 방법보다 우수한 성능을 발휘했습니다. MFD는 점수 기반 및 일관성 기반 흐름 디스틸레이션을 포함하여 여러 경쟁 모델에 비해 생성 품질과 분포적 충실도(distributional fidelity)에서 뛰어난 결과를 도출하였습니다. 이 결과들은 MFD가 고충실도의 단일 단계 생성을 가능하게 한다는 것을 증명합니다.



### P3D-Bench: Benchmarking MLLMs for Parametric 3D Generation and Structural Reasoning (https://arxiv.org/abs/2606.11152)
Comments:
          Project page: this https URL

- **What's New**: 본 논문은 다중 모달 대형 언어 모델(MLLMs)이 코드를 작성하여 복잡한 프로그램을 생산하고, 이를 통해 3D 모델링을 수행할 수 있는 새로운 가능성을 제시합니다. 기존의 평가 기준들은 3D 모델링의 코드 작성을 평가하지 않았으며, 텍스트나 시각적 명세로부터 기하적으로 정확하고 의미적으로 일치하며 조립 일관성을 갖춘 파라메트릭 3D 프로그램을 생성해야하는 요구 사항이 있습니다. 저자들은 P3D-Bench라는 새로운 기준을 소개하며, 이는 텍스트 및 이미지 사양으로부터 파라메트릭 3D 생성 및 구조적 추론을 평가하는 통합된 벤치마크입니다.

- **Technical Details**: P3D-Bench는 세 가지 작업 집합으로 구분되어 있습니다: 텍스트에서 3D로 변환(Text-to-3D), 이미지에서 3D로 변환(Image-to-3D), 조립 3D(Assembly-3D)입니다. 각 작업에서 모델은 JSON, OpenSCAD, CadQuery, Three.js와 같은 형식으로 코드를 생성하며, P3D-Bench는 이러한 코드를 실행 및 렌더링합니다. 결과 모델은 기하학적 정확성, 위상(topology), MLLM 기반 평가 및 부분 구조(part-level structure)에서 점수를 받습니다.

- **Performance Highlights**: 평가 결과에 따르면, 조립 작업은 전체 조합을 구성하는 것이 어려워 여전히 많은 모델들이 여러 부품을 일관된 구조로 조합하는 데 실패하고 있습니다. 또한, 모델들은 목표 객체의 글로벌 형태와 의미적 정체성은 잘 회복하지만, 입력으로 지정된 정확한 파라메트릭 기하학을 재현하는 데 실패하는 경향이 있습니다. 따라서 P3D-Bench는 파라메트릭 3D 생성에서 정확한 기하학성과 부분 수준 구조를 평가하는 벤치마크로 자리 잡게 되었습니다.



### MOFA-VTON: More Fashion Possibilities with Fine-Grained Adaptations in Virtual Try-On (https://arxiv.org/abs/2606.11148)
Comments:
          Accepted to CVPR 2026 (Highlight)

- **What's New**: MOFA-VTON은 사용자에 의해 그려진 간단한 곡선 스케치를 통해 가상 착용 결과에서 의류의 적응력을 조정할 수 있는 새로운 가상 착용 방법입니다. 이 방법은 기존의 정적 의류 적응 방식 문제를 극복하여, 개인의 스타일과 취향에 맞춘 다양한 패션 가능성을 탐구할 수 있게 합니다. MOFA-VTON은 이중 지역 마스크를 설계하여 사용자 스케치에서 세부적인 레이아웃 가이드를 제공하여 더 정교한 의류 적용을 가능하게 합니다.

- **Technical Details**: MOFA-VTON에서는 사용자 스케치를 기반으로 생성된 이중 지역 마스크를 사용하여 의류의 상단과 하단을 별도로 표현합니다. 이를 통해 모양 변형 과정에서 명확하고 세밀한 레이아웃 지침을 제공합니다. 또한, Layout Adjustment (LA) 블록을 도입하여 상체와 하체의 레이아웃 상관관계를 교차 주의 메커니즘을 통해 학습함으로써 의류의 공관적 배열을 개선합니다.

- **Performance Highlights**: VITON-HD 및 DressCode 데이터셋에 대한 광범위한 실험에서 MOFA-VTON은 기존의 최첨단 방법들을 능가하고 더 많은 패션 가능성을 제공하는 것을 입증했습니다. 이를 통해 사용자 개개인의 취향을 반영한 맞춤형 가상 착용 효과를 실현하여, 패션 분야에서 주목받을 만한 진전을 보여주고 있습니다.



### UniPET: a universal network for high-quality PET image denoising across varied dose reduction factors (https://arxiv.org/abs/2606.11131)
- **What's New**: 본 논문에서는 기존의 PET 이미지 디노이징(de-noising) 방법의 한계를 극복하기 위해, 다양한 DRF(dose reduction factor)에서 고품질의 결과를 얻을 수 있는 범용(PET image denoising) 네트워크인 UniPET을 제안합니다. 기존의 방법들은 고정된 DRF를 가정하여 훈련되어 성능이 저하되는 문제점이 있었으나, UniPET은 도메인 일반화(domain generalization) 기법을 적용하여 스타일 정렬(style alignment)을 통해 이를 해결합니다. 또한, flat과 stylized 영역을 구분하는 지역 인식 학습 전략(region-aware learning strategy)으로 더욱 향상된 스타일 복원을 목표로 하고 있습니다.

- **Technical Details**: UniPET은 3가지 핵심 구성 요소, 즉 기본 디노이징 네트워크(Base denoising network, BDN), 스타일 정렬 네트워크(Style alignment network, SAN), 지역 인식 학습 전략(Region-aware learning strategy, RALS)으로 구성됩니다. BDN은 다중 DRF 데이터셋에서 미리 훈련되어 초기 예측 결과를 생성하고, SAN은 서로 다른 DRF의 스타일을 정렬하여 일반화를 돕습니다. RALS는 스타일 복원을 강조하기 위해 stylized와 flat 영역을 차별적으로 다루며, GAN 훈련을 stylized 영역에만 적용합니다.

- **Performance Highlights**: 본 연구의 실험 결과, UniPET은 특정 DRF에서 개별 DRF 특화 모델과 비슷한 성능을 보여주며, 범용 PET 이미지 디노이징 분야에서 정량적, 지각적, 임상적 모두에서 최첨단 성능을 달성했습니다. 이는 UniPET이 다양한 DRF 입력을 효과적으로 처리하고 고품질 이미지를 복원할 수 있음을 나타냅니다. 전반적으로 UniPET은 기존 방법보다 더 높은 일반화 능력을 갖춘 새로운 접근 방식을 제시합니다.



### WorldOlympiad: Can Your World Model Survive a Triathlon? (https://arxiv.org/abs/2606.11129)
Comments:
          Project Page: this https URL, Code: this https URL

- **What's New**: 새로운 벤치마크인 WorldOlympiad는 비디오 기반 세계 모델(Video-based World Models)을 진단하기 위한 평가 도구로, 물리적 신뢰성(physical faithfulness), 기하학적 일관성(geometric consistency), 상호작용 충실도(interaction fidelity)를 중심으로 구성되어 있습니다. 기존 벤치마크가 시각적 품질 시각(visual quality)이나 의미적 정렬(semantic alignment)에 초점을 맞춰온 반면, WorldOlympiad는 생성된 비디오가 물리 법칙을 따르는지를 평가하여 중요성을 더했습니다.

- **Technical Details**: WorldOlympiad는 세 가지 차원에서 세계 모델 평가를 분해하여 진행됩니다. 물리학 트랙(physical track)은 물체 분할(object segmentation)과 MLLM-as-judge를 사용하여 생성된 비디오가 역학적 원칙과 열 현상, 물질 특성을 따르는지를 평가합니다. 기하학 트랙(geometry track)은 Gaussian splatting을 이용하여 생성된 비디오를 재구성하고 구조적 일관성(structural consistency), 시점 간 일관성(cross-view coherence), 카메라 경로 정렬(camera-trajectory alignment)을 평가합니다.

- **Performance Highlights**: 결과적으로, 1,000개의 고품질 긴 비디오를 수집하고 8개의 비디오 생성 파이프라인을 벤치마킹하여 이들의 신뢰성을 평가했습니다. 평가 결과는 긴 컨텍스트 일관성(long-context consistency), 물리적 추론(physical reasoning), 기하학적 안정성(geometric stability), 상호작용 제어(interaction control)에서의 체계적인 한계를 드러내며, 향후 비디오 세계 모델에 대한 진단적 증거를 제공합니다.



### FADA: Accessible fetal ultrasound interpretation and annotation with a selectively distilled unified vision-language mod (https://arxiv.org/abs/2606.11106)
- **What's New**: FADA(Fetal Anatomy Delineation and Analysis)는 여러 개의 태스크를 통합한 비전-언어 모델로, 별도의 전문가 용 레이블 없이도 임상 해석과 분류, 감지, 분할을 수행할 수 있는 단일 파이프라인으로 설계되었습니다. 현재 저소득 및 중간소득 국가에서의 초음파 검사 접근성을 개선하기 위해, 이 시스템은 비전 언어 모델을 활용하여 기계가 자동으로 해석 및 분석을 수행합니다.

- **Technical Details**: FADA는 Qwen3.5-VL을 기반으로 개발된 통합 모델로, 5단계 파이프라인을 통해 임상 해석, 해부학 분류, 바운딩 박스 감지, 폴리곤 분할을 수행합니다. 네 가지 도메인 특정 기본 모델(FetalCLIP, UltraSAM, USF-MAE, UltraFedFM)에서 지식을 증류하며, 효율적인 학습 과정을 위해 오프라인에서 미리 계산된 피처 캐싱을 활용합니다. 이 시스템은 소비자 GPU에서 학습할 수 있으며 클라우드 연결 없이도 배포 가능합니다.

- **Performance Highlights**: FADA-SKD 변형은 분할에서 0.8820 평균 Dice, 감지에서 0.7671 mAP@0.50를 달성하였고, 구조화된 해석 수행에서 100% 지식을 준수합니다. 237개 이미지를 통한 전문가 초음파 의사의 검증은 자율 및 인간-순환 모드에서 임상적으로 받아들여지는 결과를 나타내며, 73.5%의 해석이 완벽한 점수를 기록하였습니다. 이 모델은 상용 스마트폰에서 60초 만에 전체 파이프라인을 실행할 수 있어, 효과적인 포터블 초음파 장비와의 통합에 기여할 수 있습니다.



### IDEAL: In-DEpth ALignment Makes A Discrete Representation AutoEncoder (https://arxiv.org/abs/2606.11096)
Comments:
          Code is available at this https URL

- **What's New**: 본 논문에서는 Ideal이라는 새로운 프레임워크를 제안합니다. 이는 비선형 시각 변환 모델(Vision Foundation Models, VFM)의 얕은 특징과 깊은 특징을 결합하여 높은 시각 충실도와 풍부한 의미를 지닌 이산적인 시각 토큰을 생성합니다. 이를 통해 기존의 재구성 성능을 개선하는 동시에 시맨틱(Semantic) 보존까지 달성합니다.

- **Technical Details**: Ideal은 이산 표현 자동 인코딩(discrete representation autoencoding)을 위한 심층 정렬(In-depth Alignment) 프레임워크로, 다양한 층의 VFM 표현을 활용합니다. 이 방법은 VFM의 얕은 특징을 통해 세밀한 시각 적 세부사항을 유지하며, 깊은 특징을 통해 고수준의 의미를 보장합니다. 이러한 방식은 재구성 후에도 저수준의 정보 손실을 최소화하는 데 도움을 줍니다.

- **Performance Highlights**: Ideal은 ImageNet에서 0.61의 rFID를 달성하며 이전 방법보다 0.28 개선된 성능을 보여줍니다. 또한, 자율 회귀 이미지 생성에서 1.89의 gFID로 새로운 최첨단 결과를 세우며, 이는 기존 방식을 초월한 성과입니다. 이러한 결과는 Ideal이 인코딩 과정에서 얕은 시각 신호를 포함하는 이점을 가지고 있음을 입증합니다.



### U-TTT: Towards Generalizable PET Image Denoising via Test-Time Training (https://arxiv.org/abs/2606.11032)
- **What's New**: 이 논문에서 제안하는 U-TTT는 Positron Emission Tomography (PET) 이미지의 잡음 제거를 위한 새로운 U자형 모델입니다. Test-Time Training (TTT) 레이어를 통합하여 추론하는 동안 모델의 매개변수를 동적으로 조정함으로써 각 테스트 인스턴스의 특정 특성에 적응합니다. U-TTT는 공간 및 주파수 도메인에서의 이중 도메인 적응 메커니즘을 포함하여, 공간 구조 손상을 보정하고 고주파 세부 사항을 복원하는 기능을 갖추고 있습니다.

- **Technical Details**: U-TTT는 저용량 PET 이미지로부터 고품질의 풀 용량 PET 이미지를 복원하는 것을 목표로 합니다. 4단계 인코더-디코더 U자형 네트워크를 포함하고 있으며, 각 레벨은 Spatial Test-Time Training (S-TTT)과 Frequency Test-Time Training (F-TTT) 블록으로 구성되어 있습니다. 이 블록들은 테스트 시간에 모델 매개변수를 동적으로 업데이트하고 테스트 데이터에 적응하도록 하여 일반화를 개선합니다.

- **Performance Highlights**: U-TTT는 기존의 PET 이미지 잡음 제거 방법들과 비교하여 뛰어난 성능을 보여주며, 보지 못한 스캐너와 용량에서의 일반화에서도 우수한 결과를 나타냅니다. 광범위한 실험을 통해 이 모델이 여러 복잡한 분포 변화에 대해 강력하게 작용하며, PET 이미지의 잡음 제거에서 최첨단 성능을 달성하는 것을 입증하였습니다.



### An Uncertainty Estimation Framework for Dose Accumulation in Adaptive Radiotherapy: Application to CBCT-Guided Radiotherapy for Cervical Cancer (https://arxiv.org/abs/2606.11012)
Comments:
          Under revision

- **What's New**: IMPACT-DoseAcc는 불확실성을 인식하는 새로운 용적 축적 프레임워크로, 인체의 해부학적 변동성에 의한 치료 계획 조정을 가능하게 합니다. 이 프레임워크는 일반적인 CBCT 기반의 온라인 적응 방사선치료(oART)에 적용되며, 각 세션에서 환자의 해부학에 맞춰 치료를 동적으로 조정합니다. 또한, 이 방법은 해부학적 변동사항에 따른 누적 선량 해석을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: IMPACT-DoseAcc는 DIR(Deformable Image Registration)에서 발생하는 불확실성을 중점적으로 다루며, Bayesian 방법을 활용하여 해부학적 불확실성을 계량합니다. 연구에서는 두 가지 DIR 불확실성 전략을 테스트하였으며, 다양한 세그멘테이션 모델을 통해 얻은 샘플들간의 표준 편차를 이용해 앙상블 불확실성을 정량화합니다. 이러한 방법은 선량 왜곡과 누적 과정을 통하여 확률적 선량-부피 히스토그램(pDVH)을 생성하는 데 활용됩니다.

- **Performance Highlights**: IMPACT-DoseAcc의 앙상블 DIR 불확실성은 지오메트릭 오류와 상관관계가 있으며, CTVt에 대해 96.3 +/- 3.9%의 pDVH 커버리지를 달성했습니다. 이 방법은 치료 과정에서의 불확실성을 안정적으로 추정하며, 각 분획 및 장기 간의 변동성을 줄이는 데 기여합니다. 이 연구는 불확실성 인식을 통합한 ART(Adaptive Radiotherapy) 워크플로우의 재현성을 지원합니다.



### IPSM-Bench: A New Intermediate Phase Segmentation Benchmark in Microstructure Images of Zinc-Based Absorbable Biomaterials (https://arxiv.org/abs/2606.11001)
Comments:
          Accepted by IJCAI 2026

- **What's New**: 본 논문에서는 아연 기반 합금의 중간 단계 세분화를 위한 IPSM-Bench라는 대규모 고품질 데이터셋을 구축하였습니다. 이를 통해 기존의 작은 데이터셋들과는 차별화된 데이터 수집과 라벨링 과정을 통해 다채로운 구조적 이미지를 제공합니다. 새롭게 제안한 SCoP-SAM 방법은 중간 단계의 기울기 구조 및 그레이스케일 특성을 활용하여 더욱 향상된 세분화 성능을 보여줍니다.

- **Technical Details**: IPSM-Bench는 1,054개의 아연 합금 미세구조 이미지를 포함하며, 이 데이터셋은 SEM 및 OM 영상을 포함합니다. 또한, SCoP-SAM은 SAM(Segment Anything Model)의 인코더와 디코더에 공간적 맥락 정보를 통합하여, 보다 정확한 중간 단계 지역의 세분화를 가능하게 합니다. 이를 위해 고대비 관계와 작은 대상 탐지의 어려운 문제를 해결하는 기술적 접근 방법을 사용합니다.

- **Performance Highlights**: SCoP-SAM은 IPSM-Bench와 두 개의 추가 공공 합금 데이터셋에서 광범위한 실험을 통해 SOTA(seat-of-the-art) 성능을 달성했습니다. 데이터셋의 품질과 양은 모델 교육 및 평가를 위한 확실한 기반을 제공하며, 이는 아연 합금 미세구조 분석 분야의 연구 발전에 기여하게 됩니다. 이 연구는 새로운 벤치마크를 세워 기계적 및 기능적 특성 분석에 있어 중요한 기초 자료로 자리잡을 것입니다.



### AnimaSpark: A Feed-Forward Method for Animating Arbitrary 3D Objects (https://arxiv.org/abs/2606.10988)
- **What's New**: 본 논문에서는 AnimaSpark라는 새로운 3D 애니메이션 생성 파이프라인을 소개합니다. 이 방법은 3D 세계의 기본적인 움직임에 대해 조인트 변형을 2D 서브스페이스 내에서 효과적으로 모델링할 수 있다는 통찰에서 출발합니다. 그리고 이 시스템은 기계적이고 시간이 많이 소요되는 수동 작업의 의존도를 줄이는 것을 목표로 합니다.

- **Technical Details**: AnimaSpark는 정적 3D 모델을 다중 레이어 이미지로 렌더링한 후, 이를 비디오 생성 모델에 입력하여 2D 모션 시퀀스를 합성합니다. 이후에는 생성된 비디오에서 조인트의 움직임을 추적하고, 이 데이터를 바탕으로 각 조인트의 2D 변환 행렬을 계산합니다. 최종적으로, 이 2D 데이터를 3D 공간으로 다시 격상시켜 3D 애니메이션을 생성하는 방식입니다.

- **Performance Highlights**: 종합 평가 결과, AnimaSpark는 기존의 최신 기술들에 비해 텍스트-모션 정렬, 모션 품질 및 계산 효율성 등 여러 주요 측면에서 우수한 성능을 보여줍니다. 특히, 이 방법은 작업 효율성을 높이고 애니메이션 제작 시 걸리는 시간을 대폭 줄이는 데 기여할 것으로 기대됩니다.



### Quo Vadis, Visual In-Context Learning? A Unified Benchmark Across Domains and Tasks (https://arxiv.org/abs/2606.10967)
- **What's New**: 이번 논문은 비주얼 인컨텍스트 학습(Visual In-Context Learning, VICL)의 적응 능력을 다양한 이미지 작업과 도메인에서 평가할 수 있는 포괄적인 비주얼 인컨텍스트 벤치마크(Visual In-Context Benchmark, VIBE)를 제안합니다. 기존의 제한된 설정에서는 모델의 실제 적응력이 제대로 검증되지 않았습니다. 따라서 14개의 데이터셋과 12개의 작업을 포함한 106개의 작업-데이터셋 조합에 대해 6개의 모델의 성능을 정량적으로 평가합니다. 이 연구는 비주얼 인컨텍스트 학습에 대한 중요한 통찰을 제공합니다.

- **Technical Details**: VIBE는 다양한 이미지 도메인에서 비주얼 인컨텍스트 모델의 적응 능력을 평가하기 위해 설계된 툴킷입니다. 각 모델은 테스트 타임에 새로운 작업을 수행하기 위해 지정된 컨텍스트에서 몇 가지 예제를 입력으로 받으며, 이를 바탕으로 쿼리 이미지에 대한 출력을 생성합니다. 논문에서는 이미지의 본질적인 특성과 모델의 설계, 데이터로부터 얻은 통찰을 바탕으로 모델의 성능 한계와 실패 모드를 분석합니다. 이를 통해 제안된 벤치마크는 VICL 모델을 더욱 발전시키는 중요한 자료가 될 것입니다.

- **Performance Highlights**: 본 연구에서 평가된 6개의 비주얼 인컨텍스트 모델은 12개 작업을 통해 그들의 적응 능력을 폭넓게 보여줍니다. 모델들은 데이터의 다양성과 복잡한 작업에 직면할 때의 성능을 비교하여, 기존의 평가 방식에서는 발견하지 못했던 한계와 실패 모드를 식별하는 데 성공했습니다. VIBE 툴킷의 출시는 향후 연구자들이 비주얼 인컨텍스트 학습의 발전을 크게 촉진할 수 있는 기반을 마련할 것입니다.



### Democratising Camera Trap AI: An Open-Source Model for Detecting UK Mammals (https://arxiv.org/abs/2606.10940)
Comments:
          15 Pages, 4 Figures

- **What's New**: 이번 연구에서는 영국의 28개 동물 및 조류 종과 인간, 교정 기둥, 차량을 포함한 31개의 클래스에 대한 오픈 소스(object detection model)를 발표합니다. 이 모델은 Conservation AI와 Trap Tracker를 통해 수년 간 수집된 48,165개의 레이블링된 인스턴스를 사용하여 훈련되었습니다. 특정한 상업적 플랫폼에 의존하지 않고도 생태학자들이 쉽게 활용할 수 있도록 하여, 기존의 상업 모델에 대한 대안으로 사용될 수 있습니다.

- **Technical Details**: 모델은 YOLO26x 감지기를 기반으로 하며, 클래스별로 80% 훈련, 10% 검증, 10% 테스트 세트로 나누어 훈련 및 테스트를 수행하였습니다. 이 모델은 IoU(Intersection over Union) 0.5에서 평균 정확도(mean Average Precision) 0.984를 기록하였고, 정밀도(precision) 0.988, 재현율(recall) 0.965를 보였습니다. 훈련에 사용된 데이터와 동일한 현장에서 측정된 결과로, 새로운 장소에서의 성능은 향후 연구로 남겨두었습니다.

- **Performance Highlights**: 모델의 성능은 31개 클래스에 대해 평균적으로 0.96에서 0.99의 신뢰도(confidence)를 기록했으며, 0.17%의 잘못된 음성(false-negative) 비율이 발견되었습니다. 특히, 낮은 조도(night-time)나 먼 거리(distant), 또는 가려진 이미지(occluded images)에서 어려움이 있었습니다. 모델은 ONNX 형식으로 배포되어 생태학자들이 쉽게 활용할 수 있도록 지원합니다.



### PENet+: A Lightweight Residual Transformer Framework for Efficient Image Steganalysis (https://arxiv.org/abs/2606.10939)
Comments:
          IEEE ACCESS

- **What's New**: 본 논문은 디지털 이미지에 숨겨진 정보를 탐지하는 스테가나리시스(steganalysis) 기술인 PENet+를 소개합니다. PENet+는 막대한 계산 요구를 줄이면서도 PENet의 성능을 보존하는 경량 프레임워크입니다. 이 모델은 기존의 검증된 구조를 유지하며, 새로운 classifier-streamlining 단계를 추가하여 파라미터 및 계산량(FLOPs)을 대폭 줄입니다.

- **Technical Details**: PENet+는 SPP(spatial pyramid pooling)에서 FC1(fully connected layer)까지의 입력 채널을 점진적으로 줄이는 방법을 사용하여 진행합니다. HPF(high-pass filter) 메커니즘은 활성화 특성을 고려하여 답변을 집계하고, MobileNetV2 스타일의 inverted residual 네트워크로 백본을 교체합니다. PReLU(Parameterized ReLU)는 약한 스테고 단서를 포착하는 데 유리하여, negative activation을 보존하도록 설계되었습니다.

- **Performance Highlights**: ALASKA2 JPEG QF90 프로토콜에서 PENet+는 45.5% 적은 파라미터와 약 97% 감소된 FLOPs로 PENet의 성능을 초과하여, 자원 제약 환경에 적합한 계산 효율성을 제공합니다. 이러한 성능 개선은 감지 정확도를 유지하면서 가능해졌으며, PENet+는 스테가나리시스의 현실적인 배포를 위한 유망한 방향성을 제시합니다.



### Beyond Model Size: Probing the Gaps in Visual in-Context Learning by Training a Tiny Mod (https://arxiv.org/abs/2606.10905)
- **What's New**: 이번 논문에서는 Visual in-Context Learning (VICL) 접근 방식을 사용하여 적은 양의 데이터로 새로운 작업에 적응할 수 있는 모델을 개발하는 방법을 제안합니다. 특히, 100만 개의 파라미터와 70,000개의 이미지로 구성된 소형 모델 TinyVICL을 사용하여, 기존의 거대 당시 VISL 모델들과 비교하였습니다. 연구진은 이 소형 모델이 어떻게 다른 작업들을 수행할 수 있는지에 대해 분석하였습니다.

- **Technical Details**: VICL의 핵심 아이디어는 주어진 컨텍스트 세트에 기초하여, 쿼리 이미지에 대해 예측을 수행하는 것입니다. 이 과정에서 소형 모델 TinyVICL은 새로운 Palette-aware Dice loss 함수를 적용하였으며, 다양한 이미지 및 작업 분포에서 성능을 평가받습니다. 이 방법론은 특히 모델의 크기와 파라미터 수의 증가 없이도 효과적인 성능을 달성하려는 의지를 드러냅니다.

- **Performance Highlights**: 소형 TinyVICL 모델은 특정 작업 및 분포 변화에 대해 다른 대형 모델들과 경쟁할 수 있는 성과를 나타냈습니다. 연구에 따르면, 모델의 적응 능력 측정 방식에 대한 현재의 기준이 부족하다는 점을 강조하며, 이는 새로운 평가 방법론의 필요성을 제기합니다. 결과적으로 소형 모델이 큰 모델과 비교할 때 얼마나 효과적으로 작업에 적응할 수 있는가를 보여줍니다.



### Pose-ICL: 3D-Aware In-Context Learning for Pose-Controllable Subject Customization (https://arxiv.org/abs/2606.10902)
- **What's New**: Pose-ICL(프레임워크)라는 새로운 튜닝 없는 시스템을 제안하여 효과적인 포즈 제어가 가능한 맞춤형 이미지 생성을 목표로 하고 있습니다. 이 프레임워크는 여러 이미지-포즈 쌍을 통해 새로운 주체에 직접 адапта틱(Facilitating)할 수 있도록 구성되어 있습니다. 기존 방법들이 3D 지각 부족으로 인해 포즈 일관성을 유지하지 못하는 문제를 다루고 있으며, 특히 Surface-Anchored Position Embedding(SAPE)을 통해 3D 인식을 강화하여 포즈 정확성을 높인 점이 특징입니다.

- **Technical Details**: Pose-ICL은 3D-aware In-Context Learning(ICL) 패러다임을 이용하여 이미지 생성에서 주체의 위치를 효과적으로 제어합니다. 이 프레임워크는 사용자 제공 이미지와 포즈를 결합하고, 이를 통해 생성된 이미지를 컨텍스트로 활용하여 포즈 일관성을 향상시킵니다. 그리고, 각 포즈에 대해 렌더링된 볼륨 경계 상자의 표면 좌표를 SAPE로 변환하여 모델에 3D 인식을 전달함으로써 이미지의 기하학적 관계를 명확히 합니다.

- **Performance Highlights**: 기존 방법들과의 비교 평가에서 Pose-ICL은 포즈 정확도와 정체성 일관성 등에서 현저한 성과를 보였습니다. 3D 자산 및 실제 주체에 대한 광범위한 평가를 통해, Pose-ICL은 효과적인 포즈 제어 및 높은 일관성을 보여주었습니다. 이러한 결과들은 Pose-ICL이 맞춤형 이미지 생성에서 어떻게 진일보했는지를 잘 보여줍니다.



### The 1st PortraitCraft Challenge: A CVPR 2026 Workshop Competition on Portrait Composition Understanding and Generation (https://arxiv.org/abs/2606.10894)
- **What's New**: 최근 발표된 PortraitCraft Challenge는 CVPR 2026의 공식 대회로, 초상화(composition) 이해 및 생성 기술을 발전시키고자 합니다. 이 챌린지는 기존의 미적 점수 평가와는 달리, 구조적 감각의 분석을 포함하여 초상화 중심의 시스템을 도입하는 데 중점을 두고 있습니다. 운영되는 두 가지 트랙은 각각 초상화 구성 이해와 생성으로 나뉘며, 이를 통해 새로운 AI 연구를 지원할 수 있습니다.

- **Technical Details**: 챌린지는 두 개의 독립적인 트랙으로 구성되어 있습니다. Track 1은 초상화 구성 이해(composition understanding)로, 글로벌 점수 예측, 속성 수준 판단 및 VQA를 포함합니다. Track 2는 구조화된 구성 설명으로부터 초상화 이미지를 생성하는 것으로, 주어진 구성 제약 조건을 명시해야 합니다. 이를 위해 약 50,000개의 큐레이션된 초상화 이미지로 이루어진 대규모 데이터셋이 공개되었습니다.

- **Performance Highlights**: 총 295팀이 참가한 이번 챌린지는 2026년 4월부터 5월까지 운영되었습니다. 각 트랙의 성공적인 결과에 대한 정량적 평가는 정확한 스코어링을 통해 이루어졌습니다. 특히, Track 1에서는 SRCC, PLCC와 같은 지표가 사용되었고, Track 2에서는 인간 전문가의 평가로 생성된 이미지 품질이 평가되었습니다. 최종 성적에서 MSIIP 팀이 가장 높은 점수를 기록하였습니다.



### Improving Text-Instance Alignment Of Foreground Conditioned Out-Painting Via Customized Concept Embedding (https://arxiv.org/abs/2606.10892)
- **What's New**: 이 논문은 전통적인 작업 흐름이 요구하는 고품질 제품 이미지를 제공하기 힘든 문제를 해결하기 위해 텍스트 기반의 Foreground Conditioned Outpainting (FCO) 기술을 소개합니다. 기존 FCO 방법의 주요 문제 중 하나인 아티팩트(artifacts) 발생 문제를 해결하기 위해, Customized Concept Embedding Diffusion (CCE-Diffusion) 프레임워크를 제안합니다. CCE-Module을 중심으로, 텍스트 임베딩과 구체적인 시각적 인스턴스 간의 정렬을 강화하여 아티팩트를 줄이는 방안이 구체적으로 설명됩니다.

- **Technical Details**: CCE-Diffusion 프레임워크는 텍스트 프롬프트와 인스턴스 이미지 간의 미스 매칭을 수정하기 위해 설계된 CCE-Module을 포함하고 있습니다. 이 모듈은 인스턴스의 시각적 특징을 기반으로 개념 임베딩을 맞춤화하였으며, Instance-Aware Loss를 사용하여 최적화를 수행합니다. 또한, Semantic-Preserving Prompt Template을 도입하여 텍스트 프롬프트 내 다른 단어의 의미를 왜곡하지 않도록 합니다.

- **Performance Highlights**: 정성적 및 정량적 평가를 통해, CCE-Diffusion은 출력 이미지에서 아티팩트를 효과적으로 줄인 것으로 나타났습니다. 이 모듈은 ControlNet, BrushNet, BLD 등 다양한 FCO 방법과 쉽게 통합될 수 있는 플러그 앤 플레이(plug-and-play) 구성 요소로 제공되며, 그 통합을 통해 향상된 성능을 보입니다. 특히, FCO에서의 텍스트와 인스턴스 시각적 특징의 정렬 문제를 해결하여 이미지 품질을 개선하는 데 크게 기여합니다.



### Listen, Look, and Learn: Learning Without Forgetting through SAM-Audio (https://arxiv.org/abs/2606.10887)
- **What's New**: 이 논문은 Class-Incremental Learning (CIL)의 새로운 접근법을 소개합니다. 기존의 다양한 방법들과는 달리, SAM-Audio라는 기초 다중 모달 모델을 사용하여 오디오 및 비디오 특성을 통합하는 새로운 방법을 제안하고 있습니다. 연구 결과, 이 접근법이 이전 방법들보다 월등한 성능을 보임을 입증하였습니다.

- **Technical Details**: 제안된 방법은 주요 세 가지 구성 요소로 나뉩니다: (a) 다중 모달 피처 추출기, (b) 가이드된 주의 메커니즘, (c) 분류기입니다. 특히, 오디오와 비주얼 특성을 결합하기 위해 transformer 기반의 주의 메커니즘을 활용합니다. 오디오 표현은 비젼 클래스 예측을 위한 컨텍스트 신호 역할을 합니다.

- **Performance Highlights**: 광범위한 평가 결과, 제안된 방법이 기존의 최첨단 기법들을 지속적으로 초과 달성함을 보여주었습니다. 이는 CIL 환경에서 데이터의 잊혀짐을 최소화하며, 강력한 정량적 성과를 담보합니다. 특히, 기능 및 로짓 수준에서의 지식 증류를 통해 성능 저하를 방지하고 있습니다.



### Advancing Wood Identification in the Philippines: Utilizing the Xylorix Platform for Efficient AI Model Development and Deployment for Five Key Species (https://arxiv.org/abs/2606.10876)
- **What's New**: 이 연구는 필리핀에서 불법 벌목과 목재 거래 문제를 해결하기 위해 Xylorix 플랫폼을 사용하여 목재 종의 식별을 향상시키는 AI 모델을 개발하는 가능성을 평가합니다. 특히 프로그램 지식이 없는 목재 과학자들도 이 플랫폼을 통해 쉽게 사용할 수 있도록 설계되었습니다. 연구에 포함된 다섯 가지 필리핀 고무목 종은 철저한 이미지 분석을 통해 식별됩니다.

- **Technical Details**: 10,663개의 검증된 단면 이미지로 교육된 이진 분류기들이 현장 조건을 반영하여 표본 수준의 평균 점수로 평가되었습니다. ROC 곡선 아래 면적(AUC) 값은 0.969 (Ipil)에서 1.000 (Mangium)까지 다양하며, 평균 정밀도(AP) 값은 0.589 (Samanea)에서 1.000 (Mangium)까지 관찰되었습니다. 다섯 종 중 네 종은 A 등급을 얻었으며, Rain Tree는 작은 양성 테스트 집합(3 표본)으로 인해 AE 등급을 기록했습니다.

- **Performance Highlights**: 분류기들은 목표 표본을 비 목표 표본보다 훨씬 높은 신뢰도로 정확하게 분류했습니다. 오류 분석 결과 Ipil에서 9개의 허위 음성, Rain Tree에서 3개의 허위 양성, Tindalo에서 1개의 허위 양성이 발견되었으며, 이는 이미지 아티팩트나 해부학적 특성의 공유에 기인했습니다. 이 결과는 Xylorix 플랫폼을 사용하는 비 프로그래머들이 실용적인 목재 식별 모델을 구축해 필드에 배치할 수 있음을 보여줍니다.



### Schmidt Decomposition-Based Methods for Efficient Quantum Image Encoding (https://arxiv.org/abs/2606.10874)
- **What's New**: 이번 연구에서는 양자 이미지 처리를 위한 새로운 접근법인 Low-Rank Approximation (LRA)를 소개합니다. LRA는 Schmidt 분해를 사용하여 복잡성을 줄이는 방법을 모색하며, 기존의 이미지 인코딩 기법인 FRQI, NEQR, QPIE의 성능을 비교합니다. 연구 결과, LRA를 적용한 FRQI 모델은 회로의 깊이를 97% 줄이는 데 성공하면서도 거의 완벽한 이미지를 복원할 수 있음을 보여주었습니다.

- **Technical Details**: 양자 이미지 처리는 고전 이미지 데이터를 양자 상태로 인코딩하는 과정에서 발생하는 복잡성을 줄이기 위해 LRA를 사용합니다. 본 연구에서는 세 가지 인코딩 기법의 회로 깊이, CNOT 게이트 수, 평균 제곱 오차(Mean Squared Error, MSE)와 같은 성능 지표를 평가합니다. 이러한 방식으로 다양한 Schmidt 랭크에서 양자 이미지 상태를 생성하고 이의 적용 가능성을 규명하였습니다.

- **Performance Highlights**: 특히, 결과는 다른 인코딩 기법과 비교하여 LRA가 자원의 효율성과 정확성 간의 의미 있는 절충점을 제공한다는 것을 보여주었습니다. FRQI 모델은 회로 깊이를 크게 줄이면서도 높은 품질의 이미지를 복원하는 데 성공하였고, 이는 현재의 양자 하드웨어에서도 현실적으로 적용할 수 있는 가능성을 제시합니다. LRA는 향후 NISQ(Noise Intermediate-Scale Quantum) 장비에서 양자 이미지 처리의 실용성을 높일 수 있는 전략으로 기대됩니다.



### LIBERO-Occ: Evaluating and Improving Vision-Language-Action Models under Scene-Induced Occlusion via Viewpoint Imagination (https://arxiv.org/abs/2606.10862)
Comments:
          14 pages, 7 figures

- **What's New**: 이 논문에서는 Vision-Language-Action (VLA) 모델이 현실적인 환경에서의 occlusion(차폐)에 대한 도전과제를 다룹니다. 새로운 벤치마크 LIBERO-Occ가 소개되어, VLA 모델의 성능 저하를 평가하는 데 도움을 줍니다. 또한, VIM(Viewpoint Imagination)이라는 새로운 기법을 통해 차폐 상황에서도 시각적 정보를 보완할 수 있는 방법을 제안합니다.

- **Technical Details**: 나온 기법 VIM은 occluded(차폐된) 관찰에서 보조 시점을 생성하여 행동 예측을 지원합니다. 이 과정은 두 단계로 이루어지며, 첫 번째 단계에서 모델은 occluded 관찰로부터 보조 시점을 생성하고 두 번째 단계에서 이를 행동 예측과 함께 최적화합니다. LIBERO-Occ는 다양한 occlusion(차폐) 조합과 강도를 평가할 수 있도록 구성되어 있으며, occlusion이 VLA 시스템에 미치는 영향을 체계적으로 분석합니다.

- **Performance Highlights**: 실험 결과, 기존 VLA 모델들은 scene-induced(장면 유도) occlusion 하에서 성능 저하가 발생하는 것으로 나타났습니다. 반면에 VIM 기반의 방법은 부분적으로 관찰 가능한 환경에서 조작의 견고성을 개선하는 데 효과적임을 보여줍니다. VIM은 추가 장비 없이도 occlusion 문제를 해결할 수 있을 것으로 보이며, 이는 로봇 조작의 실용에 기여할 수 있습니다.



### HarmoView: Harmonizing Multi-View Constraints for Identity-Consistent Video Generation (https://arxiv.org/abs/2606.10839)
Comments:
          Project Page: this https URL

- **What's New**: HarmoView는 대구경도 합성에서의 정체성 일관성을 향상시키기 위해 다중 뷰 참고 신호를 통합하는 강력한 프레임워크입니다. 이 논문은 Multi-level Feature Injection, 학습 가능한 프록시 토큰, Jump-RoPE와 같은 세 가지 구조적 개선을 통해 정체성 일관성 비디오 생성을 달성합니다. 이 새로운 접근법은 대규모 다중 뷰 데이터셋을 구축하여 데이터 부족 문제를 해결하고, 이를 통해 세계적으로 선도적인 성능을 달성합니다.

- **Technical Details**: HarmoView는 세 가지 주요 혁신을 통해 정체성을 유지하는 데 필요한 구조적 개선을 제안합니다. 첫째, Multi-level Feature Injection(MFI)을 도입하여 정체성 충실도를 높이며, 원시 ViT 특징을 텍스트 토큰과 함께 주입합니다. 둘째, 학습 가능한 프록시 토큰을 사용하여 단일 및 다중 뷰 설정에서 heterogeneous한 레이아웃을 통일합니다. 마지막으로, Jump-RoPE를 통해 정체성 기반의 특징 격리를 실현하여 정체성 간의 crosstalk을 최소화합니다.

- **Performance Highlights**: HarmoView는 52개 고유 인물에 걸친 100개의 수작업으로 선정된 사례를 포함한 다중 뷰 벤치마크에서 광범위한 평가를 통해 성능이 입증되었습니다. 이 프레임워크는 기존 오픈 소스 방법들보다 크게 개선된 결과를 보였으며, 주요 폐쇄형 상용 엔진과 유사한 성능을 발휘하였습니다. 이러한 평가를 통해 HarmoView는 정체성 일관성 비디오 생성 분야에서 최첨단의 성과를 달성합니다.



### Earth-OneVision: Extending Remote Sensing Multimodal Large Language Models to More Sensor Modalities and Tasks (https://arxiv.org/abs/2606.10819)
- **What's New**: 본 논문에서는 Earth-OneVision이라는 새로운 원격 감지(Remote Sensing, RS) 다중 모달 대형 언어 모델(Multimodal Large Language Model, MLLM)을 제안합니다. 이 모델은 6가지 센서 모달리티를 통합하여 9개 작업 범주에 걸친 크로스 센서 융합(cross-sensor fusion)을 지원합니다. Earth-OneVision은 2B 파라미터로 설계되어, 이전의 모델들보다 더 폭넓은 데이터와 작업을 처리할 수 있습니다.

- **Technical Details**: Earth-OneVision은 세 가지 전용 메커니즘을 도입하여 현재의 RS-MLLM이 직면한 세 가지 병목 현상을 해결합니다. 첫 번째, Full-Granularity Vision-Language Alignment (FGVLA)는 다층 시각적 특징과 다차원 언어 공간을 정렬하여 더 깊은 상호 작용을 가능하게 합니다. 두 번째, Spatial-Linguistic Isomorphic Serialization (SLIS)은 이질적인 공간 출력을 언어와 유사한 토큰 시퀀스로 직렬화하여 단일 생성 패러다임으로 통합합니다.

- **Performance Highlights**: Earth-OneVision은 다양한 벤치마크에서 경쟁력 있는 성능을 보여주며, 4B-72B RS-MLLM과 비교해 동등하거나 능가하는 결과를 기록합니다. 특히, OPT-RSVG 테스트셋에서 87.52%의 P@0.5를, SAR VQA 벤치마크인 SARLANG-Bench에서 80.68%의 점수를 달성하여 7B 모델을 7% 이상 초과합니다. 또한, BigEarthNet-MS 테스트셋에서는 75.74%의 리콜을 달성하며, EarthMind-Bench에서는 81.94%의 정확도를 기록하여 크로스 모달리티 추론에서의 우수성을 보여줍니다.



### Deep learning for echo sounder data (https://arxiv.org/abs/2606.10811)
- **What's New**: 최근 10년 동안 머신러닝 기술은 데이터 처리와 해석 방식에 혁신을 가져왔습니다. 특히 수중 관측 분야에서는 음향이 주요 정보원으로 사용되며, 심층 학습(deep learning) 방법이 에코그램(echograms)과 다른 음향 데이터에 적용되었습니다. 하지만 기존의 결과는 미미하며, 음향 데이터의 고유 특성으로 인해 이미지 처리 기법의 재활용을 넘어선 심층 학습 방법 연구가 필요하다고 주장합니다.

- **Technical Details**: 음향은 과학자들이 수면 아래의 바다를 관찰하는 독특한 방법을 제공합니다. 이 중 분할빔 에코사운더(split beam echo sounders)는 해양 과학에서 표준 도구로 자리 잡고 있습니다. 머신러닝의 일종인 CNN(Convolutional Neural Networks) 기술이 에코그램 해석 및 분류를 자동화하는 데 사용되고 있으며, 여러 모델이 제안되었습니다. 최근에는 U-Net과 같은 아키텍처를 사용하여 세밀한 분할 마스크를 생성하는 연구도 진행되고 있습니다.

- **Performance Highlights**: 기존의 깊이 있는 학습 연구는 주로 이미지 기반 방법에 의존하고 있으며, 이는 음향 데이터 처리에 가장 적합한 도구에 대한 의문을 제기합니다. 현재 기술들은 특정 분류 문제에 대한 편향을 가지며, 다양한 종이 지속적으로 분포하는 음향 데이터의 특성을 반영하지 못하고 있습니다. 연구자들은 새로운 혁신을 자극하기 위해 이 분야에서 발생하는 단점들을 해결하는 데 주력해야 합니다.



### SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning (https://arxiv.org/abs/2606.10804)
- **What's New**: 이 논문에서는 중간 표현을 거치지 않고 애니메이션을 가능하게 하는 새로운 프레임워크인 SCAIL-2를 제안합니다. 기존의 방법들은 포즈 스켈레톤과 같은 중간 표현에 의존하여 정보 손실이 발생하는 반면, SCAIL-2는 드라이빙 비디오를 직접 연결하여 모든 시각적 정보를 효과적으로 캡처합니다. 이를 위해 MotionPair-60K이라는 다양한 작업을 포함하는 데이터셋을 생성하고, 새로운 DPO 기반 후처리 메커니즘을 도입하여 더 정교한 애니메이션을 생성합니다.

- **Technical Details**: SCAIL-2는 드라이빙 비디오를 입력받아 이를 인코딩하여 애니메이션을 생성하는 새로운 접근 방식을 채택합니다. 이 시스템은 영상의 노이즈를 회복하는 강조된 denoising 모델을 통해 다양한 서브 작업에 특화된 조건을 통해 입력을 향상시킵니다. 또한, in-context mask conditioning과 mode-specific RoPE를 통합하여 여러 작업 간의 일관성을 유지할 수 있도록 합니다.

- **Performance Highlights**: SCAIL-2는 다양한 애니메이션 작업에서 기존의 최첨단 방법에 비해 우수한 성능을 보여줍니다. 실험 결과, 이 모델은 복잡한 상호작용과 비인간 입력을 포함한 다양한 시나리오에서 뛰어난 일반화 능력을 입증하였으며, 특히 인물 간 동작 추적 및 환경 통합에서 두드러진 성능을 기록했습니다. 또한, SCAIL-2는 오픈 소스 모델로 제공되어 향후 다양한 애플리케이션에 활용될 수 있는 가능성을 가지고 있습니다.



### A Multimodal RGB and Events Dataset for Hand Detection in First-Person View (https://arxiv.org/abs/2606.10790)
- **What's New**: 이번 연구에서는 기존 RGB 이미지에 의존하는 손 인식 방식에서 벗어나, 이벤트 기반 카메라를 활용하여 더욱 향상된 손 검출 방법을 제안합니다. 특히, 카메라의 낮은 전력 소비와 높은 시간 해상도를 이용하여 모션 블러를 줄인다는 점이 특징적입니다. 또한, 이벤트 기반 데이터셋인 EventEgoHands를 소개하여 기존 데이터 부족 문제를 해결하고자 합니다.

- **Technical Details**: 이벤트 카메라는 기존 카메라와 달리 픽셀 별로 비동기식 이벤트 스트림을 출력하여, 고속 변화를 높은 해상도로 캡처할 수 있습니다. 본 연구에서는 v2e 툴박스를 사용하여 RGB Egohands 데이터셋을 기반으로 고유한 합성 이벤트 데이터를 생성하였으며, 이 데이터는 다양한 조명 조건과 스케일을 통해 확장되었습니다. YOLOv8 모델을 사용하여 정밀한 Ground Truth 데이터를 생성하고, 이를 통해 이벤트와 RGB 카메라를 이용한 멀티모달 손 검출이 이루어졌습니다.

- **Performance Highlights**: 제안된 EventEgoHands 데이터셋과 DAGr 모델을 통해 손 검출 정확도가 기존의 최첨단 방법과 비교될만큼 향상되었습니다. 이는 특히 동적인 환경에서 로봇 공학에 유용할 것으로 기대됩니다. 실험 결과, 이벤트 기반 접근 방식이 RGB 카메라에 비해 유리한 성능을 보였고, 다양한 작업에서 효율성을 입증하였습니다.



### From Patches to Patients: A study of the tile-to-slide performance transferability in Digital Pathology (https://arxiv.org/abs/2606.10778)
Comments:
          Accepted to MICCAI 2026

- **What's New**: 이 논문에서는 디지털 병리학 분야에서 Foundation Models (FMs)의 효율적인 평가 방법으로 타일 수준의 선형 탐사를 연구했습니다. 특히, 전체 슬라이드 이미지(WSI) 분석에서 효율성을 높이기 위해 타일 수준의 벤치마크가 슬라이드 수준 성능을 신뢰할 수 있는 대리 지표로 기능할 수 있는지를 살펴봅니다. 19개의 최신 FMs를 42개의 슬라이드 수준 및 16개의 타일 수준 작업에 대해 벤치마킹한 결과, 타일과 슬라이드 성능 간에 높은 상관관계를 확인했습니다.

- **Technical Details**: 본 연구는 자가 감독 방식으로 훈련된 19개의 병리학 Foundation Models를 사용하여 타일 수준 및 슬라이드 수준 작업을 벤치마킹했습니다. 이전 연구에서 강조했던 바와 같이, 타일 수준 벤치마크는 기초 모델 임베딩의 영향을 명확히 평가할 수 있으며, 이를 통해 슬라이드 수준 작업에서의 성능 평가를 효율적으로 수행할 수 있습니다. 또한, 변수의 민감도 분석을 통해 모델 간 전이 가능성이 안정적으로 유지됨을 보여주었으며, 집단 크기와 타일 수가 전이 가능성에 더 큰 영향을 미친다는 것을 확인했습니다.

- **Performance Highlights**: 본 연구의 주요 성과는 수치적인 상관관계를 통해 타일 벤치마크가 효율적으로 후보 인코더를 선정할 수 있는 도구임을 입증한 것입니다. 타일과 슬라이드 작업 간의 상관관계를 Pearson, Spearman 및 Kendall 메트릭스를 통해 측정했으며, 이를 통해 강력한 전이 가능성을 보여주었습니다. 전반적으로, 타일 수준 평가가 후보 모델을 좁히는 실용적이고 효율적인 첫 단계를 제공하면서도, 최종 임상 작업에 대한 검증에서는 슬라이드 수준 평가가 필수적임을 강조합니다.



### Spatially Selective Self-Training for Unsupervised Building Change Detection (https://arxiv.org/abs/2606.10775)
Comments:
          Under Review

- **What's New**: 본 연구는 SST-CD라는 새로운 프레임워크를 제안하여, 주석 없이 건물 변화 탐지(BCD)를 수행할 수 있는 방식을 혁신적으로 변화시킵니다. 기존의 방법들은 단순히 시간적 차이를 사용했지만, SST-CD는 공간적으로 신뢰할 수 있는 픽셀만을 대상으로 하여 효과적으로 탐지기를 학습합니다. 이 방법론은 불규칙한 이미지 쌍에서도 신뢰할 수 있는 부분을 선정하는 기법을 통해 노이즈에 대한 저항력을 높입니다.

- **Technical Details**: SST-CD는 비주얼 파운데이션 모델을 통해 추출된 특징을 이용하며, 시간적 불일치에서 후보 의사 라벨을 도출합니다. 이 프레임워크는 지역적 일관성을 기준으로 신뢰할 수 있는 픽셀을 선택하여 탐지기를 학습시키고, 프로토타입 기반 디코더를 통해 응축된 변화 및 비변화 표현을 생성합니다. 이러한 접근 방식은 기존의 감시 학습 및 전통적인 방법들과는 다르게, 공간적으로 선택적이고 감독 없이 모델을 훈련할 수 있는 강점을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, SST-CD는 LEVIR-CD, WHU-CD, DSIFN-CD 데이터셋에서 각각 83.08%, 91.69%, 86.60%의 F1 점수를 기록하며 기존의 비감시 및 무주석 방법들을 능가했습니다. 이 성과는 선택적 의사 라벨 최적화 접근법의 효과를 입증하며, 코드가 공개될 예정이므로 향후 연구자들이 쉽게 활용할 수 있을 것입니다.



### ZODS-RS -- Zero-training Oriented Detection & Segmentation for Remote Sensing (https://arxiv.org/abs/2606.10769)
- **What's New**: ZODS-RS는 훈련 없이 사용할 수 있는 파이프라인으로, 수평 상자(horizontal boxes, HBB)와 인스턴스 마스크(instance masks)를 출력합니다. 특히 DINOv3의 밀집 특성과 SAM 스타일 제안(proposal)을 기반으로 하여, 기존의 문제점들을 해결하기 위한 여러 단계로 구성되어 있습니다. 이 시스템은 각 단계가 멀티 레이어의 특징을 안정화시키고, 검출과 구분을 통합하여 다양한 환경에서 일관된 성능을 발휘할 수 있게 해줍니다.

- **Technical Details**: ZODS-RS는 PP(프로토타입 정제), R-SEM(회전-스케일 동등 매칭), UAM(불확실성 인식 픽셀 병합) 등의 세 가지 단계로 구성된 클로즈드 폼(closed-form) 파이프라인입니다. 이 시스템은 회전(scale)을 고려하여 다루며, 각 단계는 DINOv3와의 밀접한 결합을 통해 효과적인 성능을 이끌어냅니다. CWLA(크로스 레이어 가중치 집계)를 통해 멀티 레이어 특성을 안정화시킴으로써, 훈련 없이도 다양한 데이터셋과 환경에서 우수한 결과를 보여줍니다.

- **Performance Highlights**: FAIR1M 데이터셋에서 ZODS-RS는 mAP(평균 정밀도) 13.06, xView에서는 16.69를 기록하였으며, UAV 데이터셋에서는 31.10의 mask mIoU를 달성했습니다. 특히 작은 물체에 대한 AP는 Grounded-SAM에 비해 +30.70의 개선 효과를 나타냈습니다. 이러한 성과는 훈련 없이도 다양한 환경에서 일반화 가능한 솔루션을 제공함으로써, 원거리 감시 영상에서의 성능을 크게 향상시킵니다.



### DD-INR: Dynamics-Driven Implicit Neural Representation for Accelerated Whole-Brain Functional MRI Reconstruction (https://arxiv.org/abs/2606.10756)
- **What's New**: 본 논문에서는 DD-INR이라는 Dynamics-Driven Implicit Neural Representation 프레임워크를 제안합니다. 이 프레임워크는 불균일한 시간 변동 샘플링을 활용하여 fMRI 이미지 복원 과정을 가속화하고, 기존의 방법보다 더 우수한 성능을 발휘합니다. 전통적인 MRI 복원 방법은 공간적 정확성을 중시하여 시간적 충실도를 희생했지만, DD-INR은 이러한 문제를 극복합니다.

- **Technical Details**: DD-INR은 fMRI 데이터를 정적 배경과 시간 변동 동적 성분으로 분리하여 처리합니다. 이는 인코딩 (encoding) 및 복원 (reconstruction) 과정에서 모델의 용량을 활성화와 관련된 변경 사항에 집중할 수 있게 해줍니다. 이러한 접근방식 덕분에 이미지 품질과 활성화 패턴 복원 모두에서 기존 방법보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: DD-INR은 시뮬레이션 및 생체 내(in vivo) 데이터 수집에서 우수한 성능을 입증했습니다. 이 방법은 fMRI 연구의 민감도와 강인성을 개선할 잠재력을 가지고 있으며, 실제 스캔 시간 제한 내에서 효과적으로 작동합니다. 코드 소스는 제공된 URL에서 확인할 수 있습니다.



### Patient-Level Diagnosis of Acute Myeloid Leukemia via Deep Learning Analysis of Bone Marrow Smear (https://arxiv.org/abs/2606.10735)
Comments:
          4 figures

- **What's New**: 이 연구는 급성 골수성 백혈병(AML) 진단을 지원하기 위한 세포-환자 딥러닝 파이프라인을 소개합니다. 258명의 환자 데이터를 사용하여, 16개의 세포 주석 어휘를 통해 세포 조성을 설명하고, Composite Blast-like Cells (CBLC)라는 새로운 범주를 도입하여 AML의 진단을 지원하는 방식을 제안합니다.

- **Technical Details**: 파이프라인은 YOLO 기반의 세포 분할 모듈을 사용하여 세포를 탐지하고, 전문가의 다각형 주석과 일치시킵니다. 또한, EfficientNet-B0 분류기를 훈련시키기 위해 두 단계의 GT-to-YOLO와 YOLO-to-YOLO 전략을 사용하였으며, 클래스 불균형 보정 및 형태학적 보조 감독 방식을 포함합니다. 최종적으로 세포 수준의 예측을 환자 수준의 CBLC 비율로 집계합니다.

- **Performance Highlights**: 이 파이프라인은 내부 검증에서 안정성을 달성하고, 외부 일반화 성능도 유지하였습니다. 센터 4, 5, 6에서 각각 0.9076, 0.8696, 0.9124의 앙상블 가중치 F1 점수를 기록하며, AML 진단 지원에 대한 큰 가능성을 보였습니다.



### Vector Map as Language: Toward Unified Remote Sensing Vector Mapping (https://arxiv.org/abs/2606.10701)
- **What's New**: 이 논문에서는 원격 감지 벡터 매핑(Remote Sensing Vector Mapping, RSVM) 분야에서 새로운 패러다임인 '벡터 맵을 언어로' (Vector Map as Language, VecLang)를 제안합니다. VecLang은 다양한 지리적 개체를 구조화된 텍스트 생성으로 재구성함으로써, 서로 다른 카테고리를 통합하여 모델링할 수 있도록 해줍니다. 이는 기존의 다수의 카테고리로 나누어진 벡터 객체가 가진 한계를 극복하여, 지오JSON(GeoJSON)과 유사한 언어 형식으로 다양한 매핑 요소를 효과적으로 표현할 수 있도록 합니다.

- **Technical Details**: VecLang는 구조화된 벡터 언어(Structured Vector Language, SVL)를 사용하여 기하학(geometry), 의미(semantics), 및 위상(topology) 정보를 통합합니다. 이 시스템은 점진적 비전-언어 매핑 프레임워크(Progressive Vision-Language Mapping Framework)를 바탕으로 벡터화 단위를 지역화하고 구조화된 맵 요소를 생성합니다. 아울러, 계층적 벡터 언어 최적화(Hierarchical Vector Language Optimization)를 도입하여 신택스(validity)를 향상시키고, 내용의 신뢰성(content fidelity) 및 맵 실행 가능성을 보장합니다.

- **Performance Highlights**: 실험을 통해 VecLang는 단일 클래스 및 다중 클래스 벡터 매핑뿐만 아니라 강력한 교차 데이터셋과 오픈 어휘 일반화 성능을 보임을 입증하였습니다. 54K 이미지 및 800K 인스턴스를 포함하는 VecMap-Bench 데이터셋을 활용하여, 벡터 맵 생성이 안정적임을 검증했습니다. 이 연구는 구조화된 언어가 벡터 맵의 효율적인 표현으로서의 가능성을 보여주어, 많은 응용 분야에서 유용하게 활용될 수 있음을 제시합니다.



### Using the YOLOv12 Model for Verifying the Correct Color Sequence of Wires in Network Cables (Patch Cords) on the Production Lin (https://arxiv.org/abs/2606.10699)
- **What's New**: 이번 연구에서는 전통적인 시각적 검사 방법의 한계를 극복하기 위해 YOLO1(object detection model) 12번째 버전을 기반으로 한 지능형 시스템을 개발하였습니다. 이 시스템은 패치 코드의 와이어 위치를 식별하고 색상 순서를 검증하여 최종 제품의 성능을 보장합니다. 기존의 수작업 검사 방식 대신 자동화를 통해 인건비 절감과 생산성 향상을 목표로 하고 있습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 2,500장의 이미지를 포함하였으며, 이를 70%는 훈련, 15%는 검증, 15%는 테스트용으로 나누었습니다. 제안된 모델은 단일 단계 아키텍처(single-stage architecture)와 주의 메커니즘(attention mechanisms)을 활용하여 고도로 정밀한 와이어 감지를 이끌어냈습니다. 결과적으로 약 98%의 정밀도(precision)를 달성했으며, 전체 평균 정확도(mean accuracy)는 약 95%, 분류 정밀도(classification precision)는 99%, 재현율(recall)은 98%에 이릅니다.

- **Performance Highlights**: 이 연구의 결과는 개발된 시스템이 생산 라인에서 와이어 색상 순서의 정확성을 실시간으로 검증할 수 있음을 보여줍니다. 이 시스템은 사람의 개입 없이 자동으로 동작하여 인적 오류를 줄이고 제조 효율을 높입니다. 실제 적용 가능성 측면에서, 이러한 향상된 정확도와 효율성은 네트워크 케이블 생산 과정에서 큰 장점으로 작용할 것으로 기대됩니다.



### Don't waste SAM (https://arxiv.org/abs/2606.10696)
Comments:
          Published at European Symposium on Artificial Neural Networks (ESANN2023), Computational Intelligence and Machine Learning. Bruges (Belgium)

- **What's New**: 메타 AI는 최근 Segment Anything Model (SAM)을 발표했습니다. SAM은 다양한 과제에서 예외적인 zero-shot 이미지 분할 성능을 보여주며, 뛰어난 정확도를 자랑합니다. 다양한 연구 분야에서 정확한 분할을 제공하지 못하지만, SAM은 분할 파이프라인의 유용한 출발점이 될 수 있습니다.

- **Technical Details**: 이번 연구에서는 세 가지 폐기물 분할 데이터셋을 사용하여 SAM 모델의 일반화와 파인튜닝을 평가했습니다. SAM 모델은 click mode, box mode, 모든 객체를 자동으로 마스킹하는 everything mode를 포함하여 세 가지 모드에서 기능합니다. 각 SAM 모델은 다른 파라미터 크기(ViT-B, ViT-L, ViT-H)를 가지고 있으며, 실제 이미지를 다루기 위해 전처리된 다양한 데이터셋을 사용하여 훈련되었습니다.

- **Performance Highlights**: 파인튜닝된 SAM-ViT-H 모델은 Zerowaste와 TACO 데이터셋에서 평균 IoU가 +30으로 향상되었습니다. TrashCan 1.0에서는 초기 SAM 모델과 비교해 단지 -1.44 차이로 성능에 근접했습니다. 이러한 결과는 SAM을 기초 모델로 활용하는 것이 폐기물 분할 작업에 있어 더 나은 일반화를 제공하는 중요한 단계가 됨을 보여줍니다.



### FadeMem: Distance-Aware Memory Consolidation for Autoregressive Video Diffusion (https://arxiv.org/abs/2606.10671)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 논문에서는 FadeMem이라는 새로운 KV 메모리 통합 메커니즘을 제안합니다. FadeMem은 고정된 캐시 예산 하에 역사적인 KV 블록을 시간적 계층으로 조직하여 동영상 생성의 효율성을 높입니다. 이 메커니즘은 시간 거리 인식을 기반으로 하여 최근의 세부 사항을 밀집도 있게 유지하고, 오래된 정보는 점진적으로 통합하여 메모리 사용을 최적화합니다.

- **Technical Details**: FadeMem은 고정된 캐시 예산에 따라 생성된 역사에 대해 거리 인식을 통한 통합을 형성합니다. 이 메커니즘은 인접한 KV 블록을 근거리에서는 세밀하게, 원거리에서는 점진적으로 통합하는 시간 할당 일정을 따릅니다. 이를 통해 잦은 색상 변화나 세부 사항은 신속하게 분리되고, 장기적으로 중요한 구조적 정보와 사실적 맥락은 유지됩니다.

- **Performance Highlights**: FadeMem을 사용한 실험 결과, 기존의 경량 캐시 전략보다 주제 일관성, 배경 안정성 및 시간 일관성이 개선되었습니다. 이 방법은 최근의 컨텍스트를 보존하는 동시에, 장기적인 시나리오에서 구조적 앵커를 유지하는 데 효과적입니다. FadeMem은 아키텍처 변경 없이 기존의 오토 회귀 비디오 생성기와 호환되며, 추론 및 경량 파인 튜닝 모두 지원합니다.



### Analyzing Training-Free Corruption Detection for Object Detection Datasets (https://arxiv.org/abs/2606.10666)
Comments:
          Accepted at DataCV Workshop, Conference on Computer Vision and Pattern Recognition (CVPR) 2026

- **What's New**: 이번 연구에서는 객체 감지 데이터셋에서 주석 오류(annotation errors)를 감지하기 위한 feature-space 기반 접근법의 적용 가능성을 분석합니다. 기존의 접근법은 주로 분류 작업에 중점을 두었으나, 객체 감지에서는 주석의 의미적 및 공간적 정보가 포함되어 있어 추가적인 도전과제가 존재합니다. 우리는 기계 학습 모델을 훈련하지 않고도 오류를 탐지할 수 있는 방법을 제시하며, 이 방법이 의미적 오류를 효과적으로 드러내는 반면에 위치 오류(position errors)는 여전히 발견하기 어렵다는 점을 보여줍니다.

- **Technical Details**: 우리의 실험에서는 여러 개의 미리 훈련된 임베딩 모델(pretrained embedding models), 합성 노이즈 종류(synthetic noise types) 및 실제 주석 오류(real-world annotation errors)를 평가했습니다. 오류 탐지 기법은 SimiFeat를 기반으로 하며, 각 데이터 인스턴스의 k-최근접 이웃(k-nearest neighbors)의 라벨 분포를 분석하여 주석 오류를 감지합니다. 이를 통해 주변 라벨과의 일치도를 평가하여 해당 인스턴스가 손상되었는지를 판단합니다.

- **Performance Highlights**: 실험 결과, feature-space 기반의 접근법이 객체 감지 데이터셋에서 주석 오류를 탐지하는 데 효과적임을 입증했습니다. VOC2012와 KITTI 데이터셋을 사용하여 주요 성과를 평가하였으며, noisy 라벨과 깨끗한 라벨 간의 차이를 정량적으로 분석했습니다. 이 연구는 데이터셋 감사(dataset auditing) 및 주석 품질 분석(annotation quality analysis) 분야의 향후 연구를 지원하기 위한 자료와 도구를 제공합니다.



### Envision4D: Envisioning Visual Futures via Feed-forward 4D Gaussian Splatting for Autonomous Driving (https://arxiv.org/abs/2606.10656)
Comments:
          Project Page: this https URL

- **What's New**: Envision4D는 자율 주행에서의 동적 장면 예측을 위한 새로운 프레임워크입니다. 기존의 피드 포워드 모델이 주로 보간(interpolation) 중심으로 설계된 반면, Envision4D는 동적 장면의 미래 외삽(extrapolation)에 초점을 맞추고 있습니다. 이 시스템은 반복적인 노이즈 제거 과정을 통한 Future Pose Prediction 모듈을 도입하여 카메라 매개변수를 추론하고, 비선형 동적 특성을 포착하기 위해 In-layer Temporal Attention과 Conditioned Motion Lifting을 활용합니다.

- **Technical Details**: 이 모델은 연속적인 이미지에서 동적 장면을 외삽하는 자율 감독 피드 포워드 프레임워크로 설계되었습니다. 특히, Future Pose Prediction 모듈을 통해 사전 정의된 자아 궤적(ego-trajectories)에 의존하지 않고, 비선형 모션을 모델링합니다. In-layer Temporal Attention을 통해 동적 신호에 대한 민감도를 높이며, Conditioned Motion Lifting을 통해 현재 상태 및 시간적 우선 사항에 따라 속도를 예측함으로써 불확실성을 완화합니다.

- **Performance Highlights**: 광범위한 실험 결과, Envision4D는 기존 방법보다 우수한 성능을 보이며 동적 장면 외삽에서 최신 기술 상태(state-of-the-art)를 달성했습니다. 이 모델은 개방형 주행 시나리오에서도 강력한 일반화 능력을 보여주며, 복잡한 동적 객체의 궤적 편차 문제를 해결하는 데 기여합니다. Progressive Training Strategy를 통해 비지도 모션 학습의 안정성을 제공하여 오류의 누적을 예방합니다.



### STEDiff: Strengthening Text Embedding for Text-to-Image Alignment in Diffusion Mod (https://arxiv.org/abs/2606.10653)
Comments:
          8 pages, 8 figures, to appear at IJCNN 2026

- **What's New**: 이 연구에서는 복잡한 텍스트 프롬프트의 의미를 일치시키기 위한 새로운 접근법을 제안합니다. STEDiff라는 훈련이 필요 없는 방법은 텍스트 임베딩 공간 내에서 직접적으로 의미 표현을 향상시키도록 설계되었습니다. 특히, [EOT] 토큰을 활용하여 하위 문장의 관련 의미를 강화하고, 원래 프롬프트의 해당 토큰을 교체함으로써 성능을 개선합니다.

- **Technical Details**: 본 논문에서는 T2I(텍스트-이미지) 모델의 의미 정합성 문제를 해결하기 위해, 텍스트 임베딩의 특성을 다시 고려합니다. [EOT] 토큰을 글로벌 의미 앵커로 활용하고, 높은 차원의 인덱스 토큰을 통해 더 많은 문맥 정보를 축적하여 강한 의미 바인딩을 보장합니다. 이를 위해 새로운 의미 향상 손실(semantic enhancement loss)을 도입하여 구조적 무결성을 개선하고, 엔트로피 손실(entropy loss)을 사용하여 프롬프트 토큰이 해당 이미지 영역에 집중하도록 유도합니다.

- **Performance Highlights**: STEDiff는 다양한 기준선과 데이터셋에서 수행된 정량적 분석을 통해 기존의 최첨단(TOP) T2I 모델에 비해 성능이 우수함을 입증했습니다. 특히 다중 객체 및 다중 속성 생성 시에 뚜렷한 성과를 보였습니다. 이 방법은 사용자 친화적이며, 레이아웃 정보와 같은 추가 입력 조건 없이 텍스트 임베딩 공간 내에서 직접 수정할 수 있는 점에서 유리합니다.



### Kwai Keye-VL-2.0 Technical Repor (https://arxiv.org/abs/2606.10651)
Comments:
          31 pages, 11 figures

- **What's New**: Kwai Keye-VL-2.0-30B-A3B 모델을 소개하며, 이는 오픈소스 Mixture-of-Experts (MoE) 다중모달 (multimodal) 기초 모델로써 긴 비디오 이해 및 에이전트 인텔리전스를 향상시키기 위해 설계되었습니다. 256K 컨텍스트에서 손실 없이 비디오를 처리할 수 있는 DeepSeek Sparse Attention (DSA)를 GQA 기반 아키텍처에 적용하여 알고리즘 성능 향상과 더불어 긴 비디오에서의 정보 중복 및 계산 비용 문제를 해결합니다.

- **Technical Details**: 이 모델은 Vision Encoder (ViT), Language Decoder (LLM), MLP 프로젝트, Sparse Attention 모듈 등 네 가지 핵심 구성 요소로 이루어져 있습니다. Keye-VL-2.0은 고해상도 및 긴 컨텍스트의 다중모달 이해를 위해 네이티브 해상도 비전 인코더와 DSA 기반 스파스 어텐션 모듈을 결합하여 효율적인 긴 컨텍스트 모델링을 지원합니다. 2D Rotary Position Embedding (RoPE)을 통해 정밀한 공간적 모델링을 개선함으로써 다양한 해상도에서도 효과적으로 작동합니다.

- **Performance Highlights**: 모델의 성능 평가 결과, Keye-VL-2.0-30B-A3B는 TimeLens 및 Video-MME-v2와 같은 벤치마크에서 뛰어난 성능을 보여주었으며, 긴 비디오 및 세밀한 시간 위치 지정에서 최첨단 결과를 달성하였습니다. 또한, 다중 작업 간의 갈등을 해결하면서도 협업 능력을 강화하는 모듈을 통해 다양한 사용 사례에서의 일반적인 추론 능력을 유지합니다. 오픈소스 커뮤니티에 모델 체크포인트를 제공하여 확장 가능하고 강력한 다중모달 에이전트 응용 프로그램 개발을 촉진합니다.



### ManiSplat: Manipulation Trajectory Synthesis from Monocular Video via Decoupled 3D Gaussian Splatting (https://arxiv.org/abs/2606.10645)
- **What's New**: 이 논문에서는 실제 세상을 기반으로 하는 모노클 비디오에서 동적이고 상호작용하는 3D 장면을 재구성하기 위한 새로운 시스템, ManiSplat을 소개합니다. 기존의 3D Gaussian 재구성 방식들이 정적 장면만을 다루던 경우에서 벗어나, 로봇 팔과 조작 가능한 물체가 포함된 상호작용 환경을 효과적으로 모델링할 수 있습니다. 제안된 시스템은 로봇, 물체 및 배경을 독립적으로 최적화 가능한 Gaussian 서브필드로 분리하는 그래프 구조의 분리 표현을 도입합니다.

- **Technical Details**: ManiSplat은 그래프 구조화된 분리 표현(Graph-Structured Disentangled Representation)을 통해 로봇 팔, 조작 가능한 물체, 정적 배경을 명확히 분리합니다. 이를 통해 각 요소를 독립적으로 모델링할 수 있게 하며, 작업 지향적인 시공간 정렬(Task-Oriented Spatio-Temporal Alignment) 모듈을 통해 안정성을 개선합니다. 이 모듈은 조작 작업의 내재적 논리를 활용해 혼합 포즈 추정(Hybrid Pose Estimation)과 정렬을 통해 정확한 가상의 경로를 생성합니다.

- **Performance Highlights**: 실험 결과, ManiSplat은 상호작용 중심의 동적 장면을 높은 충실도와 제어 가능성을 바탕으로 재구성합니다. 또한, 이 메서드는 로봇 작업 및 정책 학습의 하위 지원을 효과적으로 지원하고, 단일 현실 세계 시연에서 물리적으로 일관된 합성 경로를 생성할 수 있는 데이터 증강을 제공합니다. 이를 통해 데이터 부족 문제를 해결하며 정책 일반화에 기여할 수 있습니다.



### ChartLens: A Dual-Branch Framework for Chart Data Correction and Factual Summary Refinemen (https://arxiv.org/abs/2606.10640)
- **What's New**: 본 논문에서는 DataMFM Challenge Track 2의 챔피언 솔루션인 ChartLens를 제안합니다. 이 솔루션은 차트 이미지에서 구조화된 데이터 추출과 신뢰할 수 있는 자연어 요약 생성을 목표로 합니다. ChartLens는 두 개의 주요 모듈, 즉 Structure-Aware CSV Verification and Correction (SAVC)와 Text-Retention-Guided Summary Refinement (TRSR)로 구성되어 있습니다. 모델 응답의 신뢰성을 높이기 위해 각 모듈은 데이터 검증 및 요약 개선을 지원합니다.

- **Technical Details**: ChartLens의 구조는 두 개의 상호 보완적 브랜치로 나뉘어 있습니다. 첫 번째 브랜치인 SAVC는 차트의 구조적 일관성을 검증하고, 필요 시 CSV를 수정하여 데이터의 신뢰성을 확보합니다. 두 번째 브랜치인 TRSR는 차트에서 추출된 텍스트 정보를 바탕으로 요약을 개선하며, 핵심적인 제목, 범례 및 숫자 증거를 유지하도록 유도합니다. 이러한 접근 방식은 정밀한 데이터 복구와 사실 기반의 차트 서술 생성을 결합하여 더욱 효과적인 결과를 도출합니다.

- **Performance Highlights**: 최종 모델은 시험 데이터 세트에서 69.10의 전체 점수를 기록하며 Track 2에서 1위로 평가되었습니다. 이는 차트 이해 문제 해결에 있어 ChartLens의 효과성을 보여줍니다. 차트-CSV 추출 및 차트-요약 생성을 모두 고려하여 설계된 프로세스는 특히 높은 신뢰도를 자랑합니다. 본 연구에서 제안한 방안은 데이터 분석 및 정보 검색과 같은 다양한 응용 프로그램에서도 큰 기여를 할 것으로 기대됩니다.



### Leveraging Metric Depth for Relative Depth Prediction (https://arxiv.org/abs/2606.10628)
- **What's New**: 2025 SoccerNet Monocular Depth Estimation Competition에서 우리 팀의 해결 방법을 제시합니다. 축구 시나리오에서 상대적인 깊이를 예측하는 것은 어려우며, 수천 개의 훈련 샘플만으로 이를 해결하기 위해 대규모 데이터셋에서 사전 훈련된 모델의 제로샷(zero-shot) 능력을 활용했습니다. 이 방법은 도전 과제 세트에서 $2.68 \times 10^{-3}$의 점수를 달성했습니다.

- **Technical Details**: 이 방법은 고성능 제로샷 상대 깊이 추정 모델인 Depth Anything 모델을 파인튜닝(fine-tuning)하여 메트릭(depth) 깊이를 추정합니다. Depth Anything 모델은 DinoV2 인코더를 기반으로 하며, 6200만 개의 레이블 없는 이미지를 사용하여 사전 훈련되었습니다. 또한, 상대 깊이 예측을 메트릭 깊이로 변환하기 위해 ZoeDepth 기반 디코더를 사용하며, 픽셀 수준에서 모델을 감독하기 위해 스케일 불변 로그 손실(scale-invariant logarithmic loss)을 채택합니다.

- **Performance Highlights**: 우리 방법은 2025 SoccerNet Monocular Depth Estimation Competition에서 RMSE의 2.68×10−3을 달성했습니다. 학습률과 학습 에폭(epoch)이 성능에 미치는 영향을 실험을 통해 분석하였고, 수평 뒤집기만을 데이터 증강 기술로 사용했습니다. 두 개의 서로 다른 모델을 앙상블하여 성능을 개선하려 했으나, 최종 결과에서 성능 저하가 나타났습니다.



### Can Image Models Imagine Time? ImageTime: A Novel Benchmark for Probing Visual World Modeling Through Spatiotemporal Consistency (https://arxiv.org/abs/2606.10620)
- **What's New**: 이번 논문은 이미지 생성 모델이 시간에 따라 어떻게 시각 세계를 변화시키는지를 이해하는 데 한계를 드러냅니다. 기존의 평가 방식 대부분은 단일 이미지의 정확성에 초점을 맞추고 있지만, ImageTime은 이러한 한계를 넘어서 다중 시각 상태에서의 일관성을 유지할 수 있는지를 평가하는 새로운 진단 기준을 제시합니다. 이는 단순한 이미지 생성을 넘어 실제의 변화 과정을 반영할 수 있는지를 점검하는 중요한 단계로 볼 수 있습니다.

- **Technical Details**: ImageTime은 특정 행동 지시와 선택적으로 초기 상태를 정의하는 참조 이미지에 따라 네 가지 시간 순서가 매겨진 주요 상태(초기 상태, 동작 시작, 전이 상태, 최종 상태)를 포함하는 이미지를 생성해야 합니다. 750개의 벤치마크 사례와 22개의 도메인, 375개의 행동 개념을 포함하며, 이를 통해 이미지 생성 모델이 시간에 일관된 시각 세계를 유지할 수 있는지를 평가합니다. 또한, L0에서 L6까지의 7단계 능력 트리를 설계하여 각 단계를 면밀하게 평가할 수 있도록 하였습니다.

- **Performance Highlights**: ImageTime을 사용하여 다양한 이미지 생성 모델들의 성능을 평가한 결과, 각 모델들이 얼마나 일관된 시각 상태를 유지할 수 있는지를 구체적으로 분석했습니다. 평가에서 GPT-5.5가 생성된 이미지에 대해 구조적 VLM-as-judge 프로토콜을 통해 점수를 매기며, 이를 통해 반복적으로 발생하는 실패 모드를 식별할 수 있었습니다. 이 연구는 현재 이미지 생성 시스템의 강점과 약점을 파악하는 데 기여하며, 향후 더 진화된 시각 세계 모델링으로 나아가는 데 중요한 역할을 할 것입니다.



### SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in Diffusion Models (https://arxiv.org/abs/2606.10617)
Comments:
          Accepted at ICML 2026

- **What's New**: 본 논문은 Diffusion 모델에서 Low-Rank Adaptation (LoRA)을 통합하는 새로운 방법인 Subspace Signal Routing (SSR)을 제안합니다. SSR은 매개변수 공간에서의 충돌을 피하기 위해 내부 신호를 라우팅함으로써 다양한 LoRA의 능력을 효과적으로 결합할 수 있게 합니다. 이 기법은 기존의 매개변수 집합 방식에서 발생하는 문제를 해결하고, 새로운 실험적 결과를 통해 그 유효성을 보여줍니다.

- **Technical Details**: SSR은 처음에 후보 LoRA들을 연결하여 통합된 서브스페이스를 구성하고, 그 안에서 역상관 행렬을 이용하여 혼합 신호를 비상관화합니다. 이어서 방향성 가이드 행렬을 사용하여 정제된 신호를 각 작업 특화 서브스페이스로 안내합니다. 이 과정은 Ordinary Least Squares (OLS) 해법과 수학적으로 등가임을 증명하여 최적의 해를 보장합니다.

- **Performance Highlights**: 광범위한 실험 결과, SSR은 최신 기술보다 월등한 성능을 발휘하면서도 메모리 사용과 계산 시간을 크게 줄이는 것으로 나타났습니다. 또한, 이 방법은 손쉽게 기존 생태계에 통합될 수 있도록 설계되었습니다. SSR은 형태적으로 기존 LoRA와 동일한 통합 모듈을 생성하여 실제 배포에서 추가 부담없이 사용할 수 있습니다.



### GaussTrace: Provenance Analysis of 3D Gaussian Splatting Models with Evidence-based LLM Reasoning (https://arxiv.org/abs/2606.10612)
Comments:
          Accepted by ICML2026

- **What's New**: 우리는 3D Gaussian Splatting (3DGS) 모델에 대한 방향성 기원 그래프를 구축하는 새로운 프레임워크인 GaussTrace를 제안합니다. 이 프레임워크는 3DGS 모델의 내부 속성을 포착하기 위해 속성별 통계 프로파일링을 수행하며, 편집 작업의 시뮬레이션을 통해 신뢰할 수 있는 변환 경로를 제공합니다. 또한, GaussTrace는 대형 언어 모델(LLM)을 사용하여 구조화된 Chain-of-Thought (CoT) 추론을 수행함으로써 방향성 기원 추론을 가능하게 합니다.

- **Technical Details**: GaussTrace는 창작자와 다양한 수정 히스토리를 가진 3DGS 모델 간의 진화적 관계를 추적하는 것을 목표로 합니다. 기존의 기원 분석 방법들은 3DGS 모델에 직접 적용할 수 없기 때문에 새로운 시스템을 개발해야 합니다. 이 시스템은 기원 분석을 증거 기반 추론 문제로 공식화하고, 3DGS 모델의 속성이 수정 작업마다 어떻게 반응하는지를 포착하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, GaussTrace는 3DGS 기원 그래프 재구성에서 이미지 기반, 기하학 기반, 규칙 기반 기준선보다 consistently 우수한 성능을 나타냈습니다. 이 시스템은 학습이 필요 없고 편집 기록에 접근하지 않고도 정확하고 해석 가능한 기원 그래프를 생성할 수 있습니다. 이는 3D 자산의 진화적 관계를 분석하기 위한 실질적인 프레임워크를 제공합니다.



### Globally Localizing Lunar Rover in Pixels via Graph Alignmen (https://arxiv.org/abs/2606.10602)
- **What's New**: 이번 연구에서는 우주 탐사의 필수 요소인 로버(localization) 정확도를 높이기 위해 새로운 프레임워크인 Warped Alignment of Reprojected Graphs (WARG)를 제안합니다. WARG는 그래프 학습(graph learning)과 재투영된 그래프 매칭을 통합하여, 달 환경의 특수성을 극복하면서 로버와 위성 간의 견고한 크로스 뷰 정렬을 구현합니다. 이 방법은 기존의 방법보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: WARG는 두 가지 주요 요소를 사용하여 로버의 정확한 위치를 파악합니다: 첫째, 여러 엔티티 간의 얽힘 문제를 해결하기 위해 그래프 표현을 활용하고, 둘째, 양쪽 뷰포인트를 대칭적으로 처리하는 가중치 공유 아키텍처를 통해 뷰포인트 간의 분산 문제를 극복합니다. 이러한 구조는 특히 달의 반복적인 지형과 서로 다양한 시점에서 발생하는 시각적 왜곡 문제를 해결하는 데 강점을 보입니다.

- **Performance Highlights**: WARG는 테스트 결과에서 평균 0.32 m의 로컬라이제이션 오차를 달성했으며, 복잡한 달 남극 환경에서도 3.63 m의 오류를 보였습니다. 실제 데이터에 대한 검증 결과, YuTu-2 로버의 데이터를 사용할 경우 1.68 m의 오차를 기록하여 기존의 최신 기술적 성과(SOTA)인 12.89 m보다 훨씬 개선된 성능을 나타냈습니다. WARG는 NVIDIA RTX A6000 GPU에서 5.49 Hz로 작동하여 GNSS 수준의 업데이트 주파수에 근접하는 매우 효율적인 처리 능력을 보여줍니다.



### Segment and Select: Vision-Language Segmentation in 3D Scenarios (https://arxiv.org/abs/2606.10594)
Comments:
          The core idea is to reformulate 3D vision-language segmentation as the segment-and-select paradigm (free from the superpoint dependency)

- **What's New**: 이번 논문에서는 3D 비전-언어 분할(3D vision-language segmentation) 문제를 해결하기 위해 SEGment-And-select (SEGA3D) 패러다임을 제안한다. 기존의 방법들이 대부분 coarse superpoint 표현에 의존하여 정확성이 떨어지는 문제를 해결하고, 세밀한 시각 정보를 직접 활용하는 방식으로 전환하였다. SEGA3D는 후보 마스크를 생성하고, 언어 설명과 시각적 특징을 통해 의미적 및 공간적 정보를 생성하여 최적의 후보 마스크를 선택하는 방식으로 작동한다.

- **Technical Details**: SEGA3D는 세 가지 주요 구성 요소로 이루어져 있다: 마스크 후보 생성기, 의미-공간 선택기(Semantic-Spatial Selector), 그리고 루프백 검증 모듈(Loopback Verification Module)이다. 마스크 후보 생성기는 세분화된 후보 마스크를 제공하여 기존의 superpoint 기반 방식보다 더 높은 품질을 보장한다. 이후 Large Language Model (LLM)을 통해 후보 마스크의 의미적 및 공간적 특성을 생성하며, 이를 바탕으로 최종 마스크를 결정하기 위한 후보 검증 과정을 거친다.

- **Performance Highlights**: SEGA3D는 ScanRefer, ScanNet, Matterport3D와 같은 벤치마크에서 경쟁력 있는 성능을 보여준다. ScanNet과 Matterport3D에서는 각각 8.3 mIoU와 5.3 mIoU의 성능 향상을 달성하였으며, 기존의 최고 성능 모델에 비해 향상된 결과를 기록하였다. 이러한 성과들은 SEGA3D의 후보 기반 접근 방식이 효과적임을 증명하고 있으며, 출판 이후 코드도 공개될 예정이다.



### Improving Adversarial Transferability on Vision-Language Pre-training Models via Surrogate-Specific Bias Correction (https://arxiv.org/abs/2606.10571)
Comments:
          17 pages, 7 figures, 10 tables

- **What's New**: 이 논문은 Vision-Language Pre-training (VLP) 모델에서의 적대적(Adversarial) 예제가 드러내는 취약점과 이를 개선하기 위한 새로운 접근법인 DeBias-Attack을 제안합니다. 본 연구는 적대적 최적화에서 발생하는 서그릿(Surrogate) 모델의 의존성을 줄이는 데 중점을 둡니다. 이를 통해 DeBias-Attack은 다양한 VLP 모델에 대해 더욱 강력한 전이(Transfer) 공격 성능을 보여줍니다.

- **Technical Details**: DeBias-Attack은 두 가지 섭동(Perturbation) 분기를 유지합니다. 주 분기는 원본 이미지에서 섭동을 최적화하고 이미지-텍스트 정합성을 방해하는 적대적 경량화를 생성합니다. 보조 분리는 매 반복마다 작게 재샘플링된 가우시안 노이즈를 추가하여 생성된 약한 의미의 이미지에서 최적화된 섭동을 수행합니다. 이러한 방식으로, DeBias-Attack은 서그릿 모델의 응답보다 이미지 의미론(Semantics)에 더 영향을 주어 전이 성능을 개선합니다.

- **Performance Highlights**: 다양한 VLP 모델, 다운스트림 작업, 및 멀티모달 대규모 언어 모델(Multimodal Large Language Models, MLLM)에서 DeBias-Attack은 강력한 성능을 달성했습니다. 특히 블랙박스(Black-box) 전이 설정에서, DeBias-Attack은 이질적(heterogeneous) 전이 시나리오에서도 경쟁력 있는 공격 성능을 보이고 있습니다. 이러한 결과는 편향 기반의 경량화 개선이 생성된 적대적 이미지-텍스트 쌍의 크로스 모델 전이 가능성을 강화함을 보여줍니다.



### PrismAvatar: Pseudo-Multiview Reconstruction and Subpixel Prism Rendering for Real-Time Stereoscopic Communication (https://arxiv.org/abs/2606.10550)
Comments:
          10 pages, 5 figures, 3 tables

- **What's New**: PrismAvatar는 단안 시점(monocular view)에서 포트레이트(capturing portrait) 비디오를 기반으로 제어 가능한 헤드 아바타를 재구성하고, 비구면 렌티큘러 디스플레이(glasses-free lenticular display)를 통해 실시간으로 입체 비디오 통신을 가능하게 하는 새로운 시스템입니다. 이는 기존의 아바타 시스템이 요구하던 전문적인 장비 없이도 작동할 수 있습니다. 이 시스템은 자연스러운 머리 회전을 사용해 약한 관측 영역을 보완하고, 최적화된 측면 관점(side view)으로의 시각적 경험을 위하여 다각도로 조정됩니다.

- **Technical Details**: PrismAvatar는 자연스러운 머리 회전을 PMV(pseudo-multiview) 관찰로 사용하며, 이는 비디오 정보를 각도를 맞추기 위해 정렬 및 마스킹을 통해 관리합니다. 이 시스템은 32개의 가상 뷰(virtual views)를 렌더링하고, 이를 통해 4K 해상도의 렌티큘러 패널에 매핑합니다. 추가적으로, 컨투어 손실(contour-aware losses)과 정제화(regulation)를 통해 불법적인 투명도(ghosting) 및 깊이 불안정을 줄이고, 측면의 세부사항을 유지합니다.

- **Performance Highlights**: 프리즘 아바타는 초기화 이후 4K, 32 뷰 구성에서 초당 10.65 프레임(FPS)을 달성했으며, 주어진 주제에 특화된 드라이버를 사용하게 되면 이 성능이 38.49 FPS로 상승합니다. 공공 벤치마크와 구성 요소 분석을 통해 프리즘 아바타는 비교된 방법 중 가장 낮은 외부 메쉬 알파 값을 기록했고, 목 뒤 유령 측정 및 알파 투명도 스미어에서 2위를 차지했습니다. 이를 통해 실시간 추적이 디스플레이 파이프라인에서 가장 큰 비용 요인임을 확인할 수 있었습니다.



### GRAR: Glass-induced Reflection Artifact Removal in LiDAR Point Clouds (https://arxiv.org/abs/2606.10541)
- **What's New**: 이 연구는 도시 환경에서 발생하는 유리 수반 반사 아티팩트 제거를 위한 새로운 통합 프레임워크인 GRAR(Glass Reflection Artifact Removal)를 제안합니다. 기존의 방법들이 유리의 대칭성을 이상적인 가정으로 삼는 것에 비해, 이 프레임워크는 다중 모달 비전 기초 모델을 활용하여 초기 유리 마스크를 생성하고, 이어지는 단계에서 기하학적 단서를 사용하여 높은 정밀도의 유리 영역을 달성합니다. 특히, 물리 기반의 기술인 Reflection-aware Local-Global Geometric Similarity (RE-LGGS)를 도입하여 비극복적인 관찰 상황에서도 강건한 가상 점 식별을 지원합니다.

- **Technical Details**: GRAR 프레임워크는 두 가지 주요 단계를 포함합니다. 첫 번째 단계에서는 TLS에서 수집된 파노라마 RGB 이미지와 다중 반향 맵을 융합하여 고정밀 유리 마스크를 생성하고, 이어서 유리 영역을 완성하는 전략이 적용됩니다. 두 번째 단계에서는 PCA 기반의 지역형태 표현을 사용하여 다중 스케일의 기하학적 구조와 방향 일관성을 함께 인코딩하는 RE-LGGS 기술을 사용하여, 전방향 유리 반사에서 발생하는 비극복적 관찰 상황에서도 정확한 가상 점을 식별할 수 있습니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, GRAR 프레임워크는 여러 공개 TLS 데이터셋에 대해 기존 최첨단 방법들보다 일관되게 반사 아티팩트 제거 정확도 및 포인트 클라우드 품질 보존에서 우수한 성능을 보였습니다. 이 연구는 3D 구조 모델의 신뢰성을 보장하는 데 필수적인 유리 수반 반사 아티팩트 제거 문제를 효과적으로 해결하고 있으며, 디지털 트윈 응용을 위한 데이터 정확도를 크게 향상시킬 수 있음을 보여줍니다.



### Audio-Visual Exchange-Aware Token Pruning for Efficient Audio-Visual Captioning (https://arxiv.org/abs/2606.10533)
- **What's New**: 이 논문에서는 오디오-비주얼 캡션 생성을 위한 동적 토큰 프루닝 방법인 AVEX-Prune을 제안합니다. 기존의 토큰 프루닝 방법들은 단순화된 기준으로 토큰을 선택하는데, 결과적으로 중요한 토큰을 놓치는 문제가 있었습니다. AVEX-Prune은 RL(강화 학습)을 기반으로 하여 캡션 품질 향상에 기여하는 진정으로 가치 있는 토큰을 선택하도록 학습합니다.

- **Technical Details**: AVEX-Prune은 오디오 비주얼 토큰 교환 전략을 사용하여 낮은 신뢰도를 가진 토큰을 높은 신뢰도를 가진 후보 토큰으로 대체합니다. 이 전략은 시각적-시각적, 오디오-오디오, 시각적-오디오, 오디오-시각적 등 네 가지 구조적 대체 방법을 포함하여, 훈련 과정에서 토큰의 최적 조합을 탐색합니다. 최종적으로 AVEX-Prune은 모든 교환 및 보상 계산을 삭제하고 단일 정책 전진 통과 및 Top-K 선택만을 남깁니다.

- **Performance Highlights**: AVEX-Prune은 VILA 1.5-8B와 VideoLLaMA 2에서 40%의 유지 비율을 유지하면서도 완전한 토큰 품질을 보존합니다. 각 모델에서 성능은 54.5 vs. 54.6 CIDEr, 57.0 vs. 56.8 CIDEr로 측정되었으며, 특히 모달리티(도메인)별로 우수한 성능을 보였습니다. 이 방법은 시각적-청각적 토큰 조합을 통해 더 나은 성능 향상을 보여줍니다.



### GUI-AC: Enhancing Continual Learning in GUI Agents (https://arxiv.org/abs/2606.10522)
- **What's New**: 이번 논문은 그래픽 사용자 인터페이스(GUI) 에이전트가 다양한 환경에서도 일반화할 수 있도록 지속적인 학습 능력을 개선하는 GUI-AC 방법을 제안합니다. GUI 데이터의 비정상적인 특성과 지속적인 상관관계 변화에 따라 기존의 강화 학습 기법인 Reinforcement Fine-Tuning(RFT)에서 발생하는 불안정성 문제를 해결하고자 합니다. GUI-AC는 두 가지 주요 메커니즘, 즉 Adaptive Advantage와 Dynamic Clipping을 도입하여 에이전트의 성능을 개선합니다.

- **Technical Details**: GUI-AC는 과거의 인터페이스에 대한 과신을 방지하기 위해 노이즈가 포함된 이점을 다운스케일링하는 Adaptive Advantage를 포함합니다. 또한, Dynamic Clipping 메커니즘을 통해 정책 확률의 증가를 촉진시켜探索 범위를 확장합니다. 이 방법은 기존의 작업에서 안정성을 유지하면서도 새로운 인터페이스를 적극적으로 탐색할 수 있도록 설계되었습니다.

- **Performance Highlights**: GUI-AC는 ScreenSpot-V1, V2, Pro 벤치마크에서 최첨단 성능을 달성하여 훈련 안정성과 지속적인 일반화에서 강력한 개선을 보여주었습니다. 이로 인해 GUI-AC는 기존의 접근법들에 비해 더 뛰어난 성능을 발휘하게 됨을 입증하였습니다. 이 연구 결과는 자동화된 GUI 작업에서 에이전트의 전반적인 성능을 향상시키는 데 기여할 것입니다.



### LAFP: Preserving Latent Action Structure in Latent Policy Learning via Flow Matching (https://arxiv.org/abs/2606.10517)
- **What's New**: 이 논문에서는 Latent Action Flow Policy (LAFP)를 제안하여 행동 디코더 학습 중 잠재 행동과 물리적 행동 간의 정렬 문제를 해결합니다. 기존의 방법들이 행동 클로닝 (behavior cloning) 방식을 사용하여 다중모드 행동 분포를 단일모드로 축소하는 반면, LAFP는 흐름 매칭 (flow matching) 기법을 활용해 잠재 정책 학습을 수행합니다. 이 과정을 통해 포괄적으로 더 나은 이행 학습 성능을 달성할 수 있습니다.

- **Technical Details**: LAFP는 세 가지 단계인 사전 훈련, 증류 및 후 훈련으로 구성됩니다. 사전 훈련 단계에서는 역 동역학 모델(IDM)과 전 동역학 모델(FDM)을 공동 최적화하여 잠재 행동 공간을 형성합니다. 이어서, 증류 단계에서는 현재 관찰로부터 잠재 행동을 예측하는 정책을 학습하며, 후 훈련 단계에서는 학습된 잠재 행동을 물리적 행동으로 변환하는 행동 디코더를 트레이닝합니다.

- **Performance Highlights**: 실험 결과, LAFP는 이전 방법들보다 일관되게 더 나은 성능을 보여주며, downstream imitation learning 작업에서 성공률이 10-15% 향상되었습니다. 또한 추가적인 추론 오버헤드는 1배 이하로 유지되어 효율성을 더했습니다. 이러한 성능 향상은 근본적으로 흐름 매칭 기법의 효과를 통해 얻어진 것으로 분석됩니다.



### PathRelax: Parallel-Path Relaxed Speculative Jacobi Decoding for Accelerating Auto-Regressive Text-to-Image Generation (https://arxiv.org/abs/2606.10492)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문에서는 PathSpec이라는 새로운 프레임워크를 통해 텍스트-이미지 생성 모델의 효율성을 향상시킵니다. 기존의 체인 구조에 의존하는 방법들과 달리, 이 새로운 접근 방식은 다중 시퀀스 드래프트 트리 구조를 활용하여 토큰 검색 공간을 확장하고 소요 시간을 단축합니다. 여기서 PathExplore는 이미지 품질을 저하시키지 않으면서도 토큰 수용률을 극대화하여, 더 빠른 생성 속도를 가능하게 합니다.

- **Technical Details**: PathSpec은 Speculative Decoding의 새로운 패러다임으로, 병렬 경로(parellel-path) 특성을 이용하여 이미지 토큰 생성의 구조 및 이완 공간(relaxation space)을 확장합니다. PathExplore는 체인 구조의 종속성을 줄이기 위해 멀티 시퀀스 트리 형식을 도입하여 여러 후보 토큰 시퀀스가 동시에 검증될 수 있도록 합니다. PathRelax는 서로 의미적 유사성을 가진 경로들 사이의 검증을 통해 수용률을 더욱 향상시킵니다.

- **Performance Highlights**: 이 연구는 Parti-Prompts, MSCOCO2017, T2ICompBench 데이터셋에서 각각 4.14x, 3.95x, 4.18x의 속도 향상을 달성하였습니다. PathExplore는 기존의 이완 샘플링 기법보다 고려한 속도 증대에서 우위를 점하는 성능을 보여줍니다. 이 방법은 이미지 생성의 실시간 적용 가능성을 높이며, 다양한 보조 이완 기법과의 통합이 용이하여 보다 효율적인 솔루션을 제공합니다.



### 5% > 100%: Flatness Preference is All You Need for Multimodal Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2606.10488)
- **What's New**: 이 논문에서는 Parameter-Efficient Fine-Tuning (PEFT) 방법의 주요 속성을 탐구하며, 여러 PEFT 방법에서 널리 나타나는 flatness preference를 발견했습니다. 이 연구는 기존 PEFT 방법들이 큰 모델을 특정 도메인 다중 모달 다운스트림 작업에 적응시키는 데 어떻게 느린 수렴을 겪는지를 설명합니다. 특히, 우리는 sharp dimensions가 일반화에 미치는 핵심적인 역할을 강조하며, 이러한 제한된 차원에 집중할 때 더 나은 일반화 가능성을 제안합니다.

- **Technical Details**: 본 논문에서는 Flatness Preference Optimization (FlatPO)을 제안하여 PEFT 방법의 일반화 능력을 향상시키는 접근 방식을 소개합니다. FlatPO는 학습 가능한 매개변수의 기울기 분포를 평가해 flatness preferences를 파악하며, 손실 풍경에 미치는 영향을 기반으로 최적화를 우선하며, 중요 부위를 보존하면서 sharp dimensions를 조절합니다. 이는 다양한 PEFT에 통합되어 그들의 고유한 flatness 요구를 충족하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, FlatPO의 유연성이 입증되었으며, 다양한 PEFT 방법들이 더 나은 일반화로 이어질 수 있도록 유도했습니다. 이러한 접근 방식은 다중 모달 파인튜닝을 단순화하는 데 기여할 것으로 기대됩니다. 또한, 연구는 PEFT 방법에서의 일반화와 flat하 최소값의 중요성을 강조하며, 향후 연구 방향을 제시합니다.



### 3D-CoS: A New 3D Reconstruction Paradigm Based on VLM Code Synthesis (https://arxiv.org/abs/2606.10478)
Comments:
          Preprint. 24 pages, 11 figures

- **What's New**: 최근 3D 재구성 및 편집 시스템은 NeRF, 포인트 클라우드, 메쉬와 같은 암묵적 및 명시적 표현에 의존하고 있습니다. 본 논문에서는 3D 자산을 실행 가능한 Blender 코드로 구성하는 새로운 3D 재구성 패러다임인 3D 코드 합성(3D-CoS)을 제안하고 평가합니다. 이 접근 방식은 높은 해상도의 렌더링을 가능하게 하는 동시에 프로그래밍적이고 해석 가능한 매체를 제공합니다.

- **Technical Details**: 연구는 현재의 VLM(대형 비전-언어 모델)이 코드를 통해 3D 객체를 표현할 수 있는 능력을 평가합니다. 우리는 여러 작업 흐름을 도입하고, 코드 기반 재구성 작업을 위한 통합 평가 프로토콜을 갖춘 구조화된 코드 합성 작업 흐름을 소개합니다. 또한 코드 접근법이 포인트 클라우드 기반 3D 편집과 비교하여 어떤 고유한 이점을 제공하는지를 분석합니다.

- **Performance Highlights**: 코드 기반 재구성의 효과성을 평가하기 위해 여러 VLM 모델을 테스트하였고, 관찰된 뷰에서의 재구성 품질을 평가하는 2D 메트릭스도 포함하여 결과를 보고합니다. 코드를 3D 표현으로 사용할 시 강력한 제어 가능성과 위치성(locality)을 제공하며, 편집의 정확성과 편집되지 않은 영역의 보존을 더 잘 수행합니다. 전체적인 작업 결과는 코드 합성이 편집 가능한 3D 재구성을 위한 유망한 방향임을 보여줍니다.



### Geometric Coastline Localization using Vision-Language Models (https://arxiv.org/abs/2606.10468)
- **What's New**: 이번 연구에서는 해안선 추출을 기하학적 경계 위치 측정으로 재구성했습니다. 기존의 픽셀 단위 세분화 접근 방식과는 달리, CoastlineVLM-7B 모델은 해안선을 폴리라인으로 직접 예측합니다. 이로 인해 해안선 추출 시 기하학적 표현을 중요한 설계 요소로 강조하며, 픽셀 겹침 메트릭보다 구조 기반 평가가 더 유용함을 보여줍니다.

- **Technical Details**: CoastlineVLM-7B는 GeoChat-7B 아키텍처를 기반으로 하며, 해안선 존재 감지, 프록시 유형 분류, 해안선 고정을 동시에 수행합니다. 이 모델은 복잡한 해안 환경에서 시각적 기능과 텍스트 설명을 결합하여 불확실성을 줄입니다. 또한, 기하학적 품질 평가를 위한 거리 메트릭을 사용하여 모델의 성능을 검증합니다.

- **Performance Highlights**: CoastlineVLM-7B는 참조 해안선과의 글로벌 기하학적 정렬을 향상시켜 Hausdorff 거리와 Earth Mover's Distance의 성능 이점을 보여줍니다. 치수 기반 평가 지표를 통해, 얻은 결과는 출력 표현이 해안선 추출에서 중요한 설계 선택임을 증명하며, 비전-언어 모델의 의미 분석 능력과 잘 맞는다는 것을 강조합니다.



### Few-step Generative Models as Lossy Compression (https://arxiv.org/abs/2606.10450)
- **What's New**: 본 논문은 사전 훈련된 디퓨전 모델을 활용한 손실 압축 방식을 제안하는 DiffC의 한계를 극복하기 위해 몇 단계 생성 모델들을 코덱으로 해석하는 방법을 연구합니다. 기존 DiffC는 느린 인코딩 및 디코딩 절차 때문에 실용성에 제한이 있었는데, 본 연구는 Rectified Flow, Consistency Trajectory Models (CTM), MeanFlow와 같은 모델들을 RCC(Reverse Channel Coding) 프레임워크 내에서 효과적인 코덱으로 사용할 수 있도록 합니다.

- **Technical Details**: RCC는 사후 분포와 공유 분포 매개변수가 요구되지만, Rectified Flow와 MeanFlow는 이를 명시적으로 매개화하지 않습니다. 이 연구에서는 속도 매개화와 디퓨전 스타일 디노이징 매개화의 동등성을 활용하여 필요한 양을 도출합니다. CTM의 경우 EDM 노이즈 매개화를 사용하며, 중간 상태에서 발신자 및 공유 분포에 대한 지역 가우시안 근사법을 도입하여 필요한 매개변수를 계산합니다.

- **Performance Highlights**: 본 연구가 제안하는 모델은 CIFAR10과 ImageNet 데이터셋에서 더 빠른 인코딩 및 디코딩 속도를 보이며, 저비트레이트 환경에서도 향상된 사실감을 제공합니다. 추가적인 결과는 고해상도 이미지에서도 유사한 품질상의 장점을 유지하며, 현재 기준선을 초과하는 고충실도의 재건설을 가능하게 합니다.



### Vision-Assisted Foundation Model for Solving Multi-Task Vehicle Routing Problems (https://arxiv.org/abs/2606.10431)
Comments:
          Accepted by TNNLS

- **What's New**: 이 논문은 기존의 다중 작업 차량 경로 문제(Multi-task VRP) 해결 방식의 한계를 극복하기 위해, 시각적 표현을 통합한 비전 지원 기초 모델(VaFM)을 제안합니다. 이를 통해 차량 경로 문제의 다양한 제약 조건을 효과적으로 처리할 수 있는 새로운 방법을 제시합니다. 동적 제약 조건을 학습하고 통합하는데 있어 기존의 그래프 기반 방식을 확장하여 시각적 이미지를 활용하는 점이 주목할 만합니다.

- **Technical Details**: VaFM은 컨볼루션 신경망(CNN)을 기반으로 한 비전 모달리티와 그래프 노드를 통합하여 다양한 VRP 변형을 해결하도록 설계되었습니다. 이 모델은 시각적 이미지에서 패치 수준의 의미를 학습하고, 이를 그래프 기반 노드 임베딩에 통합함으로써 최종 솔루션을 생성합니다. 특히, 여러 제약 조건에 적응할 수 있는 하이브리드 크로스 어텐션 모듈을 통해 각 작업의 수요 분포나 서비스 시간 패턴을 세밀하게 반영할 수 있습니다.

- **Performance Highlights**: 실험 결과, VaFM은 총 16개의 다양한 VRP 변형에 대해 평가되었으며, 복잡한 제약 조건을 갖는 변형에서 특히 SOTA(State of the Art) 방법보다 뛰어난 성과를 보였습니다. VaFM은 다중 제약 조건(예: OVRPTW, OVRPLTW, OVRPBLTW)이 있는 작업에서 큰 성과 개선을 이루었으며, 이는 비주얼 처리 방식이 VRP 문제 해결에 있어 큰 잠재력을 갖고 있음을 시사합니다.



### CoCoSI: Collaborative Cognitive Map Construction for Spatial Intelligenc (https://arxiv.org/abs/2606.10401)
- **What's New**: 이 논문에서는 기존의 다중 모달 대형 언어 모델(MLLM)에서 중요한 공간 인지를 개선하기 위한 경량의 모델 비 의존 방법을 제안합니다. 제안된 방법은 여러 개의 에이전트를 활용하여 영상에서 인지 맵을 공동으로 구성하며, 이는 공간 정보를 보존하는 데 초점을 맞추고 있습니다. 이 모델은 아키텍처 변경이나 추가적인 훈련 없이도 기존의 사전 학습된 MLLM에 적용할 수 있습니다.

- **Technical Details**: 제안된 다중 에이전트 프레임워크는 긴 영상을 짧은 세그먼트로 분해하고, 각 세그먼트를 여러 에이전트에 할당하여 지역적인 공간 추론을 수행합니다. 에이전트들은 공동의 인지 맵을 구축하며, 원자적 커밋(atomic commits)과 서로 간의 검증(cross-agent verification) 메커니즘을 통해 일관성을 높입니다. 이러한 접근방식은 공간 인지 작업에서 기존의 단일 에이전트 방법보다 우수한 성능을 발휘합니다.

- **Performance Highlights**: 제안된 방법은 두 가지 공간 인지 벤치마크에서 광범위한 실험을 수행하였으며, 단일 에이전트 방법, 일반 다중 에이전트 기준, 기타 훈련이 필요 없는 대안에 비해 일관되게 성능을 향상시켰습니다. 이는 비디오 공간 추론을 위한 효과적인 솔루션으로, 훈련이 필요 없는 방식으로 작동합니다. 향후 코드는 공개될 예정입니다.



### Efficient RWKV-based Representation Learning for 3D Point Clouds (https://arxiv.org/abs/2606.10395)
- **What's New**: 이번 연구에서는 최근 제안된 receptance weighted key value (RWKV) 모델을 3D 포인트 클라우드에 적용하는 P-RWKV 블록을 소개합니다. RWKV는 RNN 스타일의 반복성을 제공하며, Transformers의 복잡한 self-attention 메커니즘 대신 사용할 수 있는 선형 복잡성을 특징으로 합니다. 특히, 이 모델은 문장 텍스트를 위해 개발된 RWKV의 한계를 극복하고 불규칙한 3D 기하학 데이터에도 효과적으로 대응할 수 있는 방법을 제시합니다.

- **Technical Details**: P-RWKV 블록은 Local Perception Expansion (LPE)과 Spatial Context Enhancement (SCE)를 포함하여 지역적 기하 구조를 포착하고 공간적 인식을 향상시킵니다. LPE는 시공간 시퀀스에 걸쳐 이웃 토큰 간의 특성 교류를 촉진하여 지역 기하 정보를 풍부하게 하며, SCE는 게이팅 메커니즘을 통해 공간 인접 정보를 포함한 토큰을 augment합니다. 이들은 기존 RWKV의 효율성을 유지하면서 추가적인 계산 모듈을 도입하지 않고도 구현됩니다.

- **Performance Highlights**: 실험 결과, P-RWKV를 활용한 PointER는 3D completion, shape classification, fine-grained segmentation 작업에서 경쟁력 있는 성과를 달성했습니다. 또한, 낮은 FLOPs와 가장 짧은 추론 지연 시간으로 mamba 기반 및 transformer 기반 모델들과 견줄 수 있는 성능을 보였습니다. P-RWKV와 그 핵심 모듈은 다양한 아키텍처에서 통합 가능성이 높아, 자원 제약이 있는 상황에서의 활용 가능성을 시사합니다.



### FSS-Net: Frequency-Spatial Synergy Network with Wavelet Attention for Carotid Artery Ultrasound Segmentation (https://arxiv.org/abs/2606.10378)
- **What's New**: 이 논문에서는 뇌졸중 위험 평가를 위한 초음파 영상에서 경동맥을 정확하게 분할하기 위한 Frequency-Spatial Synergy Network (FSS-Net)을 제안합니다. 기존의 스페클 노이즈(speckle noise), 낮은 대비(low contrast), 흐릿한 경계(blurred boundaries) 문제를 극복하고자 합니다. 특히, FSS-Net은 여러 가지 첨단 기술을 통합하여 노이즈에 강력한 분할 성능을 달성합니다.

- **Technical Details**: FSS-Net은 통합된 인코더-디코더 아키텍처를 기반으로 하며, 웨이블릿 변환(wavelet transform), 다중 도메인 주의(multi-domain attention) 및 엣지 강화(edge enhancement)를 사용합니다. 채널-공간-웨이블릿 주의(Channel-Spatial-Wavelet Attention, CSWA) 모듈은 주파수 영역에서 노이즈를 억제하고 의미적 특징을 정제합니다. 또한, 웨이블릿 강화 병목(Wavelet-Enhanced Bottleneck, WEB) 모듈은 장거리 전역 의존성(global dependencies)을 효과적으로 캡처합니다.

- **Performance Highlights**: FSS-Net은 경동맥 초음파 데이터셋에서 96.46%의 Dice 점수(DSC)를 기록하여 낮은 SNR 조건에서도 강력한 내구성을 보여줍니다. 이 방법은 초음파 영상에서 경동맥 죽상경화성 플라크를 효과적으로 식별할 수 있으며, 유방암 분할(segmentation of breast cancer)과 같은 다른 작업에서도 검증되었습니다. 따라서 이 기술은 초음파 영상에서 비정상 조직 덩어리를 식별하는 데 있어 임상 적용 잠재력이 큽니다.



### PF-Trans: Physics-Embedded Frequency-Aware Transformer for Spectral Reconstruction (https://arxiv.org/abs/2606.10373)
- **What's New**: 본 논문에서는 물리 모델을 반영한 주파수 인식 Transformer인 PF-Trans를 소개합니다. 이 모델은 스냅샷 브로드밴드 필터 배열(BFA) 이미지에서 발생하는 복잡한 주파수 왜곡을 해결하면서 높은 충실도의 스펙트럼 재구성을 가능하게 합니다. 기존 딥러닝 접근 방식이 공간 노이즈 제거에 국한되어 있었던 데 반해, PF-Trans는 다양한 주파수 정보의 손실을 해결합니다.

- **Technical Details**: PF-Trans는 이중 도메인 블록(Dual-domain Block)과 병렬 고속 푸리에 변환(FFT) 파트를 통해 주파수 왜곡을 효과적으로 차단합니다. 이 시스템은 또한 물리적 감지 모델을 명시적으로 통합하며, 그래이 스케일 일관성 손실(gray-scale consistency loss)을 통해 물리적 충실성을 확보합니다. 이러한 방식은 고주파 텍스처와 정기적인 BFA 마스크 패턴의 복잡한 간섭을 구별하는 데 도움을 줍니다.

- **Performance Highlights**: PF-Trans는 GF-5 상하이 데이터세트에서 최대 48.50 dB의 피크 신호 대 잡음비(PSNR)를 기록하며, 기존의 비교 방법들과 비교해 탁월한 성능을 보여줍니다. 다양한 응용 분야에서 PF-Trans는 새로운 최첨단 벤치마크를 수립할 뿐만 아니라 복잡한 실제 응용 프로그램에서의 일반화 능력도 뛰어납니다.



### ClinReadNet: A clinical reading-inspired network for low-dose abdominal CT image quality assessmen (https://arxiv.org/abs/2606.10372)
- **What's New**: 이번 논문에서는 의사의 독서 습관을 모방한 저용량 비참조 이미지 품질 평가(No-reference IQA) 모델인 ClinReadNet을 제안합니다. 이 모델은 방사선과 의사의 임상 읽기 논리에 맞춰 설계되었으며, CT 이미지 품질 평가에 있어 중요한 실용적 가치를 가지고 있습니다.

- **Technical Details**: ClinReadNet은 Sobel ordinal quality network (SOQN) 모듈과 (shifted) window multi-scale temperature multi-head self-attention ((S)W-MTMSA) 모듈을 통합하여 이미지의 세부 정보와 전반적인 품질 분포를 동시에 분석합니다. HRPS (hierarchical ranked probability score) 손실 함수는 거칠고 세밀한 분류의 이중 논리를 결합하여 등급 라벨 간의 거리 정보를 고려합니다.

- **Performance Highlights**: 실험 결과, LDCTIQAG2023 데이터셋에서 제안된 모델은 현재 최첨단(SOTA) 성능을 달성하였습니다. Pearson의 선형 상관 계수(PLCC), Spearman의 순위 상관 계수(SROCC), Kendall의 순위 상관 계수(KROCC)는 각각 0.9507, 0.9554, 0.8629에 도달하였으며, 이들의 절대값 합계(Score)는 2.7690으로 기존 방법들을 초월하는 성과를 보였습니다.



### Benchmarking stereo reconstruction for 3D printable Martian terrain models (https://arxiv.org/abs/2606.10364)
Comments:
          9 pages, 7 figures, CVPR End-to-End 3D Workshop 2026

- **What's New**: 이 연구는 NASA의 Curiosity 로버가 촬영한 화성 지형 이미지를 기반으로 3D 모델을 재구성하는 새로운 파이프라인을 평가했습니다. 이 파이프라인은 스테레오 깊이를 추정하고 지형 기하학을 완성하며, 최종적으로 인쇄 가능한 OBJ 메시를 생성합니다. 연구 결과, RAFT-Stereo 방법이 Middlebury 데이터셋에서는 기존의 반 글로벌 블록 매칭(SGBM) 보다 우수한 성능을 보였으나, 화성 이미지에서는 안정적인 재구성이 실패했습니다.

- **Technical Details**: 연구진은 NASA Mars Science Laboratory에서 수집한 Curiosity 이미지를 사용하여 스테레오 데이터셋을 구성했습니다. 각 스테레오 쌍에서 유효 깊이 지도를 3D 포인트 클라우드로 변환하고, 지역적 충실도와 전역적 연결성을 비교하여 기하 구조를 완성했습니다. 알파 쉐이프와 포아송 재구성을 비교하여 각각의 장점과 단점을 파악하고, 이 두 방법을 융합하여 더 나은 결과를 도출하는 방법도 평가했습니다.

- **Performance Highlights**: 기술적 평가 결과, RAFT-Stereo는 Middlebury의 기준에서 성능이 우수했으나, Curiosity 이미지에서는 가장자리 정렬이 약하고 포토메트릭 재투사 오차가 증가하는 문제가 나타났습니다. 이는 기존의 성과 지표가 화성 지형 재구성에 직접적으로 적합하지 않다는 것을 시사합니다. 최종적으로, 이 연구는 인쇄 가능성이 높은 화성 지형 모델 생성을 위한 표준 스테레오 기법과 기하학적 완성 방법의 적용 가능성을 보여주었습니다.



### Multi-Angular Reflectance Anisotropy Observed from UAV Multispectral Imagery (https://arxiv.org/abs/2606.10350)
- **What's New**: 이번 연구는 UAV(무인 항공기) 다중 스펙트럼 영상에서의 다각도 관측을 활용하여 기하학적으로 유도된 방사선 변동을 정량화하기 위한 새로운 워크플로우를 제안합니다. 특히, 구조-모션(SFM) 기법을 통해 카메라의 내재적 및 외재적 파라미터를 정제하며, 다양한 시점에서 촬영된 원본 하위 영상에 동질적인 지역을 재투영합니다. 이러한 접근법은 서로 다른 관측 방향에서 동일한 지표면 목표에 대한 다중 대역 반사율과 관측 기하학 파라미터를 동시에 추출할 수 있게 합니다.

- **Technical Details**: 제안된 방법은 (VZA, RAA) 도메인에서 대역별 극좌표 시각화를 활용하여 추출된 관측 자료를 분석합니다. 이를 통해 관측 기하학의 영향을 평가하는데 필요한 정밀한 반사율 데이터를 생성할 수 있습니다. 연구에 사용된 Grassland 타겟에서의 결과는 10개의 대역 전반에 걸쳐 뚜렷한 반사율 이방성을 보여주며, 특히 적색 변두리(red-edge) 대역과 근적외선(near-infrared) 대역에서 최대 및 최소 반사율 사이에 119-137%의 변동성이 나타났습니다.

- **Performance Highlights**: 이 연구는 관측 기하학이 방사선 일관성에 미치는 비가시적 효과를 이해하는 데 중요한 통찰력을 제공합니다. 여러 각도에서의 다각도 관측 데이터를 기반으로 한 이 분석은 UAV 촬영 데이터의 정확성을 높이는데 기여할 수 있습니다. 결과적으로, 제안된 워크플로우는 UAV 다중 스펙트럼 영상의 정확한 반사율 측정을 위한 강력한 도구가 될 것으로 기대됩니다.



### Building Change Detection in Earthquake: A Multi-Scale Interaction Network and A Change Detection Datas (https://arxiv.org/abs/2606.10329)
- **What's New**: 이번 연구는 지진 피해 평가를 위한 새로운 Change Detection (CD) 데이터셋, Turkey Earthquake CD dataset (TUE-CD)를 제시합니다. TUE-CD는 2023년 2월 6일 터키에서 발생한 7.8 강진 이후 5일 이내에 수집된 다중 시간 원격 감지 이미지 쌍으로 구성되어 있으며, 이를 통해 신속한 재난 구호 필요에 부응하고자 합니다. 더 나아가, 다중 스케일 특성 상호작용 네트워크(Multi-scale Feature Interaction Network, MSI-Net)를 통해 이미지 간의 측면 문제를 완화하며 효과적인 변화를 탐지할 수 있는 방안을 마련했습니다.

- **Technical Details**: MSI-Net은 여러 모듈로 구성되며, 이 모듈들은 변화를 탐지하는 과정에서 다중 스케일의 특징을 효과적으로 통합합니다. 구체적으로, Joint Cross-Attention (JCA) 모듈을 통해 다중 스케일 특징 간 상호작용을 강화하고, Multi-scale Offset Calibration (MOC) 모듈을 통해 이미지 간의 정렬을 최적화합니다. 마지막으로, Feature Integration (FeI) 모듈에 의해 보정된 특성과 다중 스케일 특성이 결합되어 변화 맵을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 MSI-Net이 WHU-CD, CLCD 및 TUE-CD 데이터셋에서 기존의 여러 딥러닝 기반 CD 방법보다 우수한 성능을 보였습니다. MSI-Net은 특히 건물 피해 추정에서의 효과적인 결과를 보여주었으며, 이로 인해 재난 구호 작업에 필수적인 도구가 될 수 있음을 입증하였습니다. 전반적으로, MSI-Net은 가까운 시간 간격의 데이터에 대한 높은 정확도를 제공하여 지진 후 응급 구호에 적합한 솔루션을 제시합니다.



### Content-Induced Spatial-Spectral Aggregation Network for Change Detection in Remote Sensing Images (https://arxiv.org/abs/2606.10328)
- **What's New**: 이번 연구에서는 공간 정보와 스펙트럼 정보의 통합이 변화 감지 성능의 향상에 기여할 수 있다는 점에 주목하여, Content-Guided Spatial-Spectral Integration Network (CSI-Net)을 제안합니다. CSI-Net은 공간 이해 모듈(SR), 스펙트럼 차이 모듈(SD), 콘텐츠 기반 통합 모듈(CGI)로 구성되어 있으며, 이러한 모듈을 통해 전반적인 공간 정보와 스펙트럼 차이 정보를 효과적으로 융합할 수 있습니다. 특히, 이 네트워크는 변화가 없는 지역에서 발생하는 스펙트럼 차이의 영향을 최소화할 수 있도록 설계되었습니다.

- **Technical Details**: CSI-Net의 SR 모듈은 그래프 합성곱 블록을 통해 전역 공간 정보를 학습하며, SD 모듈은 피처의 평균과 분산을 계산하여 변화가 없는 지역에서 스펙트럼 차이를 완화합니다. 또한 CGI 모듈은 공간-스펙트럼 피처의 통합을 위해 고급 콘텐츠 정보를 가이드로 활용하여 상호작용의 적절성을 높입니다. 따라서 CSI-Net은 변화한 특징을 더 잘 학습할 수 있고, 스펙트럼 차이를 억제하는 데 기여합니다.

- **Performance Highlights**: LEVIR-CD, WHU-CD, CLCD 데이터셋에서 수행된 실험 결과, CSI-Net은 최신 기술(State-of-the-Art) 방법보다 우수한 성능을 보였으며, 다양한 시나리오에 적용 가능한 것으로 나타났습니다. CSI-Net은 변화 감지 정확도에 영향을 미치는 무관한 변화의 효과를 배제할 수 있는 기능을 갖추고 있습니다.



### Dissect and Prune: Enhancing Robustness in AI-Generated Image Detection (https://arxiv.org/abs/2606.10309)
Comments:
          25 pages, 9 figures, 9 tables, Accepted to ICML 2026; includes appendix

- **What's New**: 최근 생성 모델의 발전으로 AI 생성된 이미지(AIGI)가 점점 더 현실적인 방법으로 생성되고 있으며, 이는 신뢰할 수 있는 탐지 방법의 필요성을 증가시키고 있습니다. 그러나 기존의 AIGI 탐지기는 높은 정확도를 보고하지만, 실제 클래스에 대한 편향으로 인해 AI 생성 콘텐츠에 대한 민감도가 낮은 예측 비대칭(prediction asymmetry)이라는 중요한 단점을 드러냅니다. 이 문제를 해결하기 위해 우리는 DEAR (Dissect and Prune)를 제안하여, AI 생성 콘텐츠의 진정한 특성을 포착하는 강력한 특징만을 남기고 불필요한 특징을 제거합니다.

- **Technical Details**: DEAR는 inpainted 이미지를 활용하여 AI 생성 이미지의 탐지 성능을 개선하는 새로운 접근법입니다. 이 방법은 생성된 픽셀과 실제 이미지 문맥을 구별하며, 특징 활성화와 inpaint 마스크 간의 정렬을 측정하여 클래스에 대한 편향적 예측을 줄입니다. 특히, 우리는 생성 콘텐츠를 강하게 나타내는 특징과 실제 배경에 편향된 특징을 모두 제거하여 모델이 직관적으로 AI 생성 콘텐츠를 신뢰할 수 있게 만들어줍니다.

- **Performance Highlights**: 실험 결과, DEAR는 이전의 탐지 방법에 비해 성능이 크게 개선되었으며, 보지 못한 생성자와 포스트 프로세싱에 대한 강인성을 획기적으로 향상시켰습니다. DEAR의 접근 방식은 AI 생성 이미지 감지에서의 예측 비대칭을 효과적으로 완화하며, 이를 통해 실질적인 이미지 검증의 정확도를 높였습니다. 이러한 성과는 다양한 테스트 환경에서 실증적으로 입증되었습니다.



### FoA-SR: Faithful or Aesthetic? Profile-Aware Preference Optimization for Real-World Image Super-Resolution (https://arxiv.org/abs/2606.10275)
Comments:
          17 pages, 6 figures, 9 tables. Preprint

- **What's New**: 이번 논문에서는 실제 이미지 초해상도(Real-ISR)에서 복원 전략을 명시적으로 프로파일별로 최적화하는 FoA-SR을 제안합니다. Faithful(충실) 복원은 구조 보존과 감정 억제를 강조하며, Aesthetic(미적) 복원은 자연스럽고 시각적으로 즐거운 디테일에 집중합니다. 기존의 SR 파이프라인은 모든 기준에 대해 최적이라고 여기는 단일 출력을 제작하는 제한이 있었으나, 본 연구는 두 가지 복원 프로필을 구분하여 복원 선호도를 명확히 드러냅니다.

- **Technical Details**: FoA-SR은 감독된 FLUX.2 기반의 SR 어댑터인 Flux2SR을 시작으로 다중 확률적 후보를 샘플링합니다. 이 후보풀은 각 프로파일에 따라 평가되며, Faithful 보상은 참조 일관성을 강조하고 Aesthetic 보상은 비참조 인지 품질을 중시합니다. 후보의 승자-패자 쌍을 구성하고 프로파일 전용 LoRA 어댑터를 미세조정하는 방식으로, 두 프로파일 모두 동일한 SR 기준선과 후보풀을 공유하여 구동됩니다.

- **Performance Highlights**: 실험 결과, FoA-SR은 Faithful 어댑터가 참조 지향적인 재구성을 향상시키는 반면, Aesthetic 어댑터는 비참조 인지 품질을 향상시키는 결과를 보여주었습니다. 500장의 이미지 집합에서 Faithful과 Aesthetic 보상은 78.4%의 입력에서 서로 다른 승자를 선택하는 경향을 보였으며, 이는 복원 프로필이 존재하며 이를 명확히 보여주는 것이 중요함을 강조합니다. 또한 Hybrid-LoRA 분석을 통해 양 프로파일을 하나의 보상으로 단순화할 경우 유용한 제어 기능이 숨겨진다는 것을 증명했습니다.



### An Improved Generative Adversarial Network for Micro-Resistivity Imaging Logging Restoration (https://arxiv.org/abs/2606.10200)
Comments:
          7 pages, 9 figures

- **What's New**: 이 논문에서는 부분적으로 손상된 마이크로 저항 이미지 로그 이미지를 복원하기 위한 개선된 GAN 기반 이미지 복원 방법을 제안합니다. 이 방법은 FCN(골격망) 기반의 생성을 네트워크로 사용하며 깊이 분리형 합성곱 잔여 블록을 추가하여 픽셀 및 의미 정보를 보다 효과적으로 학습하고 유지합니다. 또한 이 네트워크의 다중 스케일 인지 필드를 확장하고 매개변수를 줄이기 위해 Inception 모듈을 추가하였습니다.

- **Technical Details**: 제안된 방법은 채널 주의 메커니즘과 잔여 블록을 결합하여 다중 스케일 특징 추출 기능을 구현합니다. 글로벌 판별 네트워크와 로컬 판별 네트워크를 설계하여 복원된 부분과 전체 이미지 간의 내용 및 의미 구조의 일관성을 점진적으로 개선합니다. 이 연구에서는 다양한 크기의 손실 영역을 가진 이미지 로그 이미지 다섯 세트를 테스트하여 구조 유사도 측정치가 평균 0.903에 도달하여 유사한 방법에 비해 약 0.3의 개선을 보여주는 성과를 거두었습니다.

- **Performance Highlights**: 이 방법은 손상된 마이크로 저항 이미지 로그 이미지를 복원하는 데에서 의미 구조 일관성과 텍스처 세부 정보를 크게 개선할 수 있음을 보여줍니다. 이로 인해 마이크로 저항 이미지 로그 이미지의 해석과 같은 후속 작업이 원활하게 진행될 수 있습니다. 제안된 기술은 석유 및 가스 산업의 결정을 위한 더 나은 저수지 특성과 분석을 지원할 수 있는 잠재력을 갖추고 있습니다.



### Fisher-Guided Progressive Parameter Selection for Adaptive Fine-Tuning (https://arxiv.org/abs/2606.10196)
- **What's New**: 최근 연구에서는 기존의 고정된 아키텍처 휴리스틱에 따라 훈련 가능한 매개변수 집합을 선택하는 대신, 동적이고 작업에 민감한 기준을 사용해야 할 필요성이 강조되고 있습니다. 이 논문에서는 FisherAdapTune이라는 새로운 프레임워크를 제안하여 Fisher 기하학의 시간적 변화 추적을 통해 매개변수 그룹을 점진적으로 선택합니다. 이를 통해, 안정화된 커브 기여를 가진 매개변수는 고정되어 오류 한계를 줄일 수 있음을 보여줍니다.

- **Technical Details**: FisherAdapTune은 PAC-Bayesian 관점에서의 이론적 기초를 바탕으로 합니다. Fisher Information Matrix (FIM)를 사용하여 매개변수의 민감도와 로컬 커브를 캡처하며, Jensen-Shannon 거리를 측정하여 매개변수 그룹의 진화를 추적합니다. 이러한 방법을 통해 데이터 주도적 신호를 제공하여 매개변수의 동적 선택 전략을 구현합니다.

- **Performance Highlights**: FisherAdapTune은 특정의 밀집 예측 작업에서 검증되었으며, 기존의 PEFT 기반과 비교하여 경쟁력 있는 성능을 달성하고, 분포 변화에 대한 강인성을 향상시켰습니다. 이 연구는 안정화된 매개변수를 고정함으로써 일반화 성능을 개선할 수 있음을 입증하며, 보다 효율적이고 작업에 민감한 매개변수 선택의 중요성을 강조합니다.



### Making Time Editable in Video Diffusion Transformers (https://arxiv.org/abs/2606.10183)
- **What's New**: 이 논문에서는 새로운 Temporal-Control methodology를 제안하여 pretrained Diffusion Transformer (DiT)의 시간 편집 기능을 명확히 하였습니다. 이 방법은 motion speed와 temporal structure를 조정하는 데 중점을 두고 있으며, Backbone을 재설계하지 않고도 기존 모델의 생성 성능을 유지하며 확장할 수 있습니다. 이를 통해 모델이 시간의 진행 상황을 보다 신뢰성 있고 구조화된 방식으로 조정할 수 있게 됩니다.

- **Technical Details**: 제안된 방법은 pretrained DiT Backbone에 명시적인 temporal control 모듈을 추가하여 global motion pacing과 local temporal alignment를 분리합니다. 이 구조는 global FPS embedding과 latent-time embedding을 도입하여 장면의 발전 속도와 프레임 간 시간적 흐름을 조정합니다. 이러한 구성을 통해 모델은 시간의 변화를 보다 명확하게 표현할 수 있으며, 데이터에서 시간적 전통을 캡처하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 baseline과 비교할 때 Human Actions와 Natural Processes의 영상 생성에서 안정적인 단기 시간 전환을 달성했습니다. TA는 FPS가 훈련 레짐에 근접할 때 중간 동작 일관성을 향상시키며, FTTA는 자연 프로세스에서 최상의 시간 정규화를 달성하였습니다. 이들 결과는 명시적인 시간 조정이 다양한 DiT Backbone에 효과적으로 이전될 수 있음을 보여줍니다.



### A Large Scale Open-Source Image and Video Dataset for Robust Wildfire Detection and Classification (https://arxiv.org/abs/2606.10174)
- **What's New**: 이 연구에서는 조기 화재 및 연기 탐지 연구를 지원하기 위해 설계된 대규모 오픈 소스 데이터셋인 GWFP(글로벌 산불 예방 데이터셋)를 소개합니다. GWFP는 다양한 지리적 지역에서 수집된 화재 및 환경 조건 이미지를 포함하여, 실제 세계의 화재 상황을 포괄적으로 담고 있습니다. 이를 통해 다양한 환경 조건에서의 데이터 일반화와 강건성을 증진하는 것을 목표로 하고 있습니다.

- **Technical Details**: GWFP는 화염, 연기, Near Infrared (NIR), Ember 및 기타 비화재 샘플을 포함한 다채로운 이미지를 카테고리별로 정리하였습니다. 또한, HTE-ResNet(하다마드 강화 잔차 연결)을 통해 도메인 변화 조건에서의 표현 강건성을 분석하여 가벼운 주파수-공간 특징 상호작용을 탐구하였습니다. 실험 결과, GWFP는 다양한 모델을 평가하는 데 유용하며, 인사이트 있는 결과를 보여줍니다.

- **Performance Highlights**: 연구에서는 기존 데이터셋에 비해 GWFP가 제공하는 다양한 환경을 바탕으로 강력한 교차 데이터셋 일반화를 입증하였습니다. 또한, GWFP를 통해 다양한 컨볼루션 및 변환 기반 아키텍처의 벤치마킹을 할 수 있음을 확인했습니다. 이러한 연구 결과는 실제 산불 모니터링에 대한 실용적인 활용 가능성을 강조하고 있습니다.



### FlexPath: Learned Semantic Path Priors for Image-Based Planning (https://arxiv.org/abs/2606.10167)
- **What's New**: 최근의 학습 기반 경로 계획자들은 시각적 지도 표현을 처리하고 고전 탐색 알고리즘을 위한 휴리스틱을 근사화하여 거의 최적의 경로를 찾습니다. 그러나 이러한 방법은 감독의 암묵적인 최단 경로 목표에 일치하게 되어 있어 대체 기준을 수용하는 유연성이 제한되었습니다. 이러한 문제를 해결하기 위해 FlexPath라는 두 단계의 프레임워크를 도입합니다.

- **Technical Details**: FlexPath는 첫 번째 단계에서 모방 학습(imitation learning)을 사용하여 시각적 지도 입력으로부터 유효한 경로에 대한 작업 독립적 공간 사전(prior)을 획득합니다. 두 번째 단계에서는 미분 가능한 경로 형태 목표(Path Shape Objectives, PSOs)가 이 사전을 작업 특정 기준으로 적응시키며 경로 구조를 다시 학습할 필요 없이 효율적인 목표 수준의 적응만을 요구합니다.

- **Performance Highlights**: FlexPath는 최단 경로 계획에서 기존의 TransPath에 비해 TMP(search effort)를 14.3% 줄이고, 평균적으로 더 낮은 비용의 경로를 찾으며, 세 가지 존재하지 않는 영역에서 강력한 제로샷 제너럴라이제이션(zero-shot generalization)을 보여줍니다. 또한 최소 클리어런스 거리 2를 가진 장애물 회피에서는 96.8%의 완전한 장애물 회피를 달성하면서도 낮은 탐색 비용을 유지합니다.



### Fusing Satellite Imagery and Planimetric Maps for Cross-View Localization (https://arxiv.org/abs/2606.10166)
- **What's New**: 본 논문은 위성 이미지와 평면 지도(Planimetric Maps)를 융합하는 새로운 모듈을 제안하여 자율 주행 시스템의 위치 추정 성능을 향상시켰습니다. 기존 방법 대부분이 단일 방식으로 동작하는 반면, 본 연구에서는 두 가지 모달리티를 통합하여 状태-최고 (state-of-the-art) 성과를 달성하였습니다. 즉, satellite imagery와 planimetric maps를 효과적으로 조합하여 위치 측정 오류를 30.13% 줄였습니다.

- **Technical Details**: 제안된 융합 모듈은 두 이미지 모달리티 간의 상호 작용을 촉진하기 위해 cross-modal conditioning과 patch-level fusion rule을 포함합니다. 이 방식은 공간의 다른 영역이 서로 다른 모달리티에서 이점을 받을 수 있도록 조정하며, 정보의 교환을 세밀하게 처리하여 보다 정확한 위치 추정을 가능하게 합니다. 또한, 컨텍스트 인식 추출기를 통해 모달리티 간의 중복성을 줄입니다.

- **Performance Highlights**: 실험 결과, 본 논문은 KITTI 데이터셋에서 평균 위치 추정 오류를 3.85m로 달성하여 기존 단일 방식 방법에 비해 30.13%의 감소를 보여주었습니다. 또한, 세 가지 교차 뷰 위치 추정 방법과 두 개의 벤치마크에서 일관적인 성능 향상을 나타내어 제안된 융합 모듈의 유효성을 입증하였습니다.



### DB-3DME: From Dataset to Benchmark for Human-aligned Automatic 3D Mesh Evaluation (https://arxiv.org/abs/2606.10142)
Comments:
          CVPR 2026 workshop paper. 10 pages, 3 figures, 6 tables. Dataset available at GitHub and Hugging Face

- **What's New**: 최근 3D 생성의 발전은 사실감, 제어 가능성 및 효율성에서 많은 개선을 이루었지만, 3D 자산의 평가에 대한 연구는 미비합니다. 본 연구에서는 DB-3DME라는 3D 메쉬 평가를 위한 데이터셋과 벤치마크를 소개합니다. 이 데이터셋은 2,619개의 합성 3D 메쉬와 인간 평가를 결합하여 생성되었습니다. 연구를 통해 최신 비전-언어 모델(VLM)의 평가 성능 개선을 위한 중요한 요소를 식별하고, Qwen-2.5-VL-7B 모델을 미세 조정하여 3D 메쉬 평가에 최적화하였습니다.

- **Technical Details**: 기존의 3D 평가 방법은 인간 평가, 학습된 메트릭, VLM을 판별자로 사용하는 접근법으로 나눌 수 있습니다. 인간 평가는 전문가 주석자들이 3D 생성을 평가하는 방식으로, 비자동적이며 비용이 많이 드는 단점이 있습니다. 반면, VLM 기반 접근법은 자동 평가의 Scalability를 제공하지만, 대부분의 기존 VLM은 고해상도 입력을 처리하는 데 제한을 겪으며, 높은 품질의 소재에 대한 평가에 부족한 성능을 보입니다. 본 연구에서는 3D 메쉬 평가에 집중하여 인간 레이블을 포함한 데이터셋을 구축하였습니다.

- **Performance Highlights**: DB-3DME 데이터셋과 함께, 본 연구는 다양한 최신 VLM 모델의 평가를 통해 기존의 사전 훈련된 VLM들보다 높은 성능을 기록하였습니다. Qwen-2.5-VL-7B 모델의 미세 조정 결과, 평가 차원 전반에서 상당한 성능 향상을 이루고 새로운 벤치마크를 설정했습니다. 이 모델은 3D 메쉬 평가의 품질을 크게 향상시키고, 향후 연구에 대한 기초를 마련합니다.



### iSAGE: A Human-in-the-Loop Framework for Remote Sensing Semantic Segmentation via Sparse Point Supervision (https://arxiv.org/abs/2606.10136)
Comments:
          47 pages, 8 tables, 6 figures

- **What's New**: iSAGE(Iterative Sparse Annotation Guided by Expert)는 기존의 리모트 센싱의 시맨틱 세그멘테이션 문제를 해결하기 위한 새로운 접근법을 제안합니다. 이 프레임워크는 전문가의 클릭만으로 밀집된 감독(Dense supervision)을 달성할 수 있다는 가설을 세우고, 이를 실제로 구현한 오픈 소스 플랫폼을 구축했습니다. iSAGE는 오류 가중치 손실(Error-Weighted Loss)을 사용하여 클릭 시 모델의 그래디언트를 증폭시키며, 주석 기록 자체가 데이터셋으로 활용됩니다.

- **Technical Details**: iSAGE는 클릭과 예측 오버레이를 결합하여, 모델이 자신있게 잘못된 픽셀에 대해 전문가가 직접 클릭할 수 있도록 합니다. 이 방식은 기존 HITL(human-in-the-loop) 프레임워크가 사용하는 다수의 보조 기계장치 없이 수행됩니다. 실험에서 최소한의 노력으로 각 클래스당 최대 하나의 레이블이 부여된 픽셀을 사용하여, 낮은 비율의 주석으로도 높은 성능을 달성하는 것을 목표로 합니다.

- **Performance Highlights**: BsB Aerial 실험에서 iSAGE는 0.040%의 픽셀로 97.2%의 밀집 감독을 재현하였으며, ISPRS Vaihingen 벤치마크에서는 0.011%의 픽셀로 76.78%의 mIoU를 달성했습니다. 이는 밀집 기준선(76.65%)과 일치하며, 기존 모든 방법을 초월하는 결과입니다. iSAGE는 31개의 조사 대상으로 한 유일한 반복적 HITL 프레임워크로, 보조 기계장치 없이 효과적으로 작동하는 솔루션을 제공하고 있습니다.



### BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression (https://arxiv.org/abs/2606.10135)
- **What's New**: 이 논문에서는 기존의 인과 모델(causal models)과 비교하여 Bidirectional Autoregressive 모델(BiWM)의 혁신적인 접근 방식을 제안합니다. BiWM은 상호작용하는 비디오 월드 모델(interactive video world models)을 구축할 수 있는 첫 번째 오픈소스 프레임워크로, 생성 품질(generation quality)과 추론 속도(inference speed)를 동시에 최적화합니다. 이 프레임워크는 두 단계의 훈련 스테이지만을 필요로 하여 복잡한 프로세스를 간소화하였습니다.

- **Technical Details**: BiWM은 사전 훈련된 비디오 백본(pretrained video backbone) 모델에서 카메라 제어를 주입함으로써 카메라 제어(fine-tuning)를 수행한 후, 몇 단계의 DMD(Distribution Matching Distillation) 과정을 통해 액션 및 카메라 제어가 가능한 월드 모델을 생성합니다. 이 접근법은 기존의 네 단계 과정(minWM)을 두 단계로 통합하여 전반적인 훈련 시간을 크게 단축시키며, 다양한 아키텍처와 모델에 걸쳐 적용 가능합니다.

- **Performance Highlights**: BiWM은 높은 신뢰성과 안정성을 제공하며, 기존의 인과 모델이 채택하는 접근 방식보다 훨씬 나은 비디오 이미지 품질을 나타냅니다. 이 시스템은 실제 카메라 제어를 가능하게 하여, 긴 탐색 거리(long horizon rollout)에서 플러그인 형식의 역사 압축(history compression) 메커니즘을 통합함으로써 자원 제한이 있는 연구 환경에서도 효과적으로 활용됩니다. BiWM은 논문에서 제시된 대로, 낮은 비용으로 기존의 Bidirectional 모델을 새로운 데이터 분포에 맞춰 쉽게 튜닝할 수 있는 능력을 갖추고 있습니다.



### Improving PET/CT-Based Whole-Body Lesion Segmentation Using Prediction Uncertainty-Augmented Models (https://arxiv.org/abs/2606.10115)
Comments:
          32 pages, 10 figures, 5 tables

- **What's New**: 이 논문은 전체 신체 양전자 방출 단층촬영(PET)/전산화 단층촬영(CT)에서의 병변 분할(segmentation)을 위해 새로운 불확실성 인식(until uncertainty-aware) 프레임워크를 제안합니다. 이 프레임워크는 훈련의 확률적 변동성(training stochasticity)을 줄이기 위한 베이지안 앙상블(Bayesian ensembling), 복셀 단위의 불확실성 정량화(voxel-wise uncertainty quantification), 그리고 병변 탐지를 개선하기 위한 불확실성 증강 훈련(epistemic uncertainty-augmented training)을 통합합니다.

- **Technical Details**: 제안된 프레임워크는 nnU-Net을 기반으로 하며, 두 개의 공공 데이터셋인 AutoPET-III(1,611 데이터)와 Deep-PSMA(200 데이터)를 사용하여 훈련 및 평가됩니다. Bayes 앙상블 방법을 적용함으로써 nnU-Net 모델보다 더욱 안정적이고 우수한 성능을 보여줍니다. 또한, 불확실성 맵은 모델 불일치가 있는 영역을 강조하고 잘못 분류된 결과와 상관관계를 가집니다.

- **Performance Highlights**: 불확실성 증강 훈련은 증가된 FPVol의 대가로 병변 회복을 개선하여 정밀도-재현율 간의 균형을 반영합니다. 사례 적응 라우팅(case-adaptive routing) 전략은 기본 및 증강 모델 간 선택을 통해 Dice 점수를 추가적으로 개선합니다. 이 연구는 다중 추적기의 범암(Pan-Cancer) PET/CT 병변 분할에서 불확실성 정량화를 체계적으로 조사한 최초의 연구로, 베이지안 앙상블과 불확실성 인식 모델링을 결합하여 이 작업에 접근했습니다.



### Maximum Matching Accuracy: An Instance Segmentation Evaluation Metric Utilizing Globally Optimal Matching (https://arxiv.org/abs/2606.10107)
- **What's New**: 이 논문에서는 기존의 인스턴스 세분화(instance segmentation) 성능 평가 메트릭이 수학적으로 약점이 있음을 지적하고, 새로운 메트릭인 Maximum Matching Accuracy (MMA)를 제안합니다. MMA는 예측된 객체와 실제 객체 간의 최적의 일대일 매칭을 찾고, 픽셀 단위로 정규화하여 총 중첩(overlap)을 집계하는 방식으로 작동하며, 하드 기준을 사용하지 않습니다. 이 메트릭은 세포 분할(분류)에서의 공정한 벤치마킹을 위해 안정적이고 해석 가능한 점수를 제공합니다.

- **Technical Details**: MMA는 크게 세 가지 설계 선택을 통해 기존 세분화 성능 메트릭의 단점을 극복합니다. 첫째, MMA는 전역적으로 최적의 일대일 매칭을 수행하며, 둘째, 연속적인 중첩 점수 평가를 통해 점수를 부여합니다. 셋째, 전역적으로 픽셀 기반의 정규화를 적용하여 성능을 평가합니다. 이들 특성은 MMA가 기존 메트릭보다 더 안정적이고 민감하다는 것을 입증하는 데 기여합니다.

- **Performance Highlights**: MMA는 AP@50, PQ, SEG, AJI와 같은 기존 메트릭 대비 더 높은 안정성과 민감성을 보여주었습니다. 실험 결과, 모델 순위에서 최대 50%의 차이가 발생, MMA가 다른 메트릭과 비교하여 모델의 정확한 성능을 드러낸다는 것을 보여주었습니다. MMA는 세포 이미지 세분화 외에도 다른 인스턴스 세분화 분야에 적용 가능성을 지닌 메트릭으로 제안됩니다.



### Interpretable Temporal Facial-Region Motion Analysis for In-the-Wild Parkinson's Disease Video Classification (https://arxiv.org/abs/2606.10088)
Comments:
          22 pages, 6 figures. Submitted to Biomedical Signal Processing and Control

- **What's New**: 이 논문은 파킨슨병(Parkinson's disease, PD)의 얼굴 표정 감소에 대한 연구로, YouTubePD 데이터셋을 이용하여 비디오 분류를 개선할 수 있는 방법을 제안합니다. 특히, 14개의 미리 정의된 얼굴 영역에서 추출된 동적(temporal) 모션 디스크립터를 통해 얼굴 움직임을 평가하여 PD 관련 비디오의 정확한 분류를 목표로 하고 있습니다. 연구 팀은 정적(Static) 기하학, 정상화(normalized) 기하학, 속도 기반(descriptors velocity-based) 디스크립터 등을 비교하며, 무작위 술어 분석(seed-robustness) 및 지역 레벨 제거(region-level ablation) 등의 방법으로 분류 성능을 평가했습니다.

- **Technical Details**: 연구는 얼굴의 주요 지점(keypoint)에서 추출된 동적 모션 정보를 기반으로 하며, 각 비디오는 14개의 정의된 얼굴 영역에서 기하학적 디스크립터로 표현됩니다. 연구에서 사용된 디스크립터는 정적 기하학, 정규화된 기하학, 그리고 다양한 속도 기반 디스크립터들로, 이들을 로지스틱 회귀(logistic regression), 서포트 벡터 머신(support vector machine), 랜덤 포레스트(Random Forest) 및 GRU 기반의 시퀀스 기준을 통해 비교하고 있습니다. 이 과정은 PD 관련 분류를 위한 유용하고 해석 가능한 신호를 제공하기 위해 설계되었습니다.

- **Performance Highlights**: 정규화된 속도 디스크립터(normalized velocity descriptors)와 랜덤 포레스트 분류기를 이용한 결과는 0.826의 균형 정확도(balanced accuracy)와 0.855의 AUROC를 달성했습니다. 10개의 무작위 시드를 활용한 테스트에서도 균형 정확도는 0.810 +/- 0.018로 안정성을 보였으며, AUROC는 0.855 +/- 0.005로 일관성을 유지했습니다. 이러한 결과는 정규화된 얼굴 영역의 동적 모션이 YouTubePD 비디오 분류에서 해석 가능하고 재현 가능한 표현이라는 것을 시사합니다.



### A Controlled Audit of Pretraining Contamination in Public Medical Vision-Language Benchmarks (https://arxiv.org/abs/2606.10066)
Comments:
          30 pages, 7 figures, 9 tables. Preprint

- **What's New**: 이번 연구는 의료 비전-언어 모델(vision-language models, VLM)에서의 데이터 누출을 감사하는 것을 목표로 합니다. 기존의 연구들은 일반적인 VLM에서의 누출 문제를 다루었으나, 의료 분야에서는 이러한 감사가 전무했습니다. 연구팀은 SLAKE-En, PathVQA, VQA-RAD와 같은 대규모 공개 벤치마크를 사용하여 의료 VLM의 혐의 있는 전이 문제를 분석했습니다.

- **Technical Details**: 감사 과정에서는 네 가지 검출기(detector)를 사용하여 이미지 및 텍스트 데이터에서 누출 신호를 측정했습니다. 특정 패턴으로 이미지의 중복을 측정한 결과, SLAKE-En의 19.8% 이미지는 PMC-OA-beta와 중복되었으며, 이는 같은 환자에 대한 것이라기보다 비슷한 위치에서의 교차-모델 패턴으로 해석됩니다. 텍스트 측면에서도 Qwen2.5-VL의 교환 가능성이 입증되었지만, BLIP-2는 누출 신호가 없었습니다.

- **Performance Highlights**: 제출된 감사 결과는 이미지 측면에서의 두 개의 검출기가 신뢰할 수 없음을 나타냈습니다. 또한, BLIP-2와 같은 외부 기준선에 대한 성능 비교를 통해 얻은 신호들은 검사하였던 모델의 특수성 및 훈련 데이터에 따른 영향을 분리하는 데 중요했습니다. 최종적으로, 이 연구는 소규모 의료 VLM 집단에서의 개별 데이터에 대한 기억 추론이 신뢰할 수 없다는 결론을 내렸습니다.



### SpineReport: Automated 3D Quantification and Reporting of Lumbar Spine Degeneration on MRI (https://arxiv.org/abs/2606.10021)
Comments:
          Submitted to Medical Image Analysis

- **What's New**: 이번 논문에서는 허리 척추 MRI의 3D 변형 분석을 위한 새로운 오픈 소스 툴인 SpineReport를 소개합니다. 기존의 2D 분석 방식은 해부학적 구조가 이미징 평면과 정렬되지 않았을 때 재현성이 떨어지는 문제를 가지고 있었습니다. SpineReport는 이러한 문제를 해결하고 실시간으로 포괄적인 3D 변량 분석을 가능하게 합니다.

- **Technical Details**: SpineReport는 강력한 해부학적 세그멘테이션을 활용하여 신경관, 척수, 척추뼈, 추간판 및 신경공과 같은 주요 구조에서 정량적 메트릭을 추출합니다. 이 메트릭은 형태학적(morphological) 및 신호 기반(signal-based) 특징 둘 다를 포함하여, 개인 간 및 시계열(longitudinal) 평가를 가능하게 합니다. 주제별 보고서를 생성하여 집단 분포와 비교할 수 있게 함으로써 해석 가능성을 높이고 척추 형태의 객관적인 특성을 부여합니다.

- **Performance Highlights**: 임상적 유의성을 평가한 결과, 메트릭은 중심관(Central Canal) 협착의 중증도와 강한 연관성을 보였으며, T2-가중 CSF 신호가 가장 높은 성능을 나타냈습니다 (AUC = 0.95). 또한, 관의 AP 직경 및 면적 비율 또한 강한 상관관계와 높은 구별 능력을 보여주었습니다 (AUC > 0.80). 측면 recess 협착에 대해서는 moderate한 연관성이 관찰되었고, 측면 CSF 신호가 가장 유용한 정보로 나타났습니다 (AUC = 0.73). 그러나 신경공 협착에 대해서는 강력한 지역 관심 영역 추출에도 불구하고 유의미한 연관성이 관찰되지 않았습니다.



### Generalized-CVO: Fast and Correspondence-Free Local Point Cloud Registration with Second Order Riemannian Optimization (https://arxiv.org/abs/2606.10019)
Comments:
          16 pages, 12 figures

- **What's New**: 본 논문에서는 기하학적 표면 구조와 재생 커널 Hilbert 공간(RKHS) 임베딩을 활용한 빠르고 대응 없는(local point cloud registration) 지역 포인트 클라우드 등록 방법을 제안합니다. 이 방법은 점구름을 연속 함수로 표현하며, 점 별 비등방성 커널을 통해 지역 기하학을 인코딩합니다. 이러한 공식화는 표면 법선에 대한 정렬을 개선하고 접선 방향으로의 정렬을 완화합니다.

- **Technical Details**: 문제 해결을 위해 근사 리만 헤세안(approximate Riemannian Hessians)을 적용한 이차(on-manifold) 최적화 방법을 제안하며, 이전의 대응 없는 RKHS 기반 방법들에서 사용된 일차 솔버들에 비해 최대 10배의 속도 향상을 달성합니다. 이 방법은 주로 LiDAR와 RGB-D 데이터를 통한 프레임 간 추적 정확도 향상에 효과적입니다. 이차 최적화 기법은 복잡한 기하학적 형태의 효율성을 극대화합니다.

- **Performance Highlights**: 주행 분야의 LiDAR 추적 등록 작업에서, 도전적인 특성 희소 환경에서 변환드리프트(translational drift)와 회전드리프트(rotational drift)가 55% 이상 감소하는 성과를 보였습니다. 또한 ICP 기반 방법들에 비해 객체 등록 벤치마크에서 더욱 향상된 견고성을 입증했으며, 특히 중간 정도의 비정렬(misalignment) 상황에서 글로벌 초기화(refining global initialization) 시 더욱 성능이 개선되는 결과를 보여주었습니다.



### ABot-Earth 0.5: Generative 3D Earth Mod (https://arxiv.org/abs/2606.09967)
Comments:
          From Amap-cvlab, Alibaba. Official page: this https URL

- **What's New**: ABot-Earth 0.5는 고해상도 위성 이미지를 기반으로 방대한 3D 환경을 생성하는 새로운 생성적 3D 프레임워크입니다. 이 모델은 3D Gaussian Splatting (3DGS) 표현을 통해 3D 장면을 신속하고 효과적으로 합성할 수 있으며, 1제곱킬로미터당 10분 이하로 생성 속도를 유지합니다. 이를 통해 재난 대응, 도시 계획 및 로봇 탐사와 같은 분야에서의 사용 가능성을 높입니다.

- **Technical Details**: ABot-Earth 0.5는 고품질 실제 도시 재건 축소에 직접 훈련된 3D 장면을 생성하는 것을 목표로 합니다. 이 모델은 3DGS 기반으로 설계되어 복잡한 지형 및 건물 외관과 같은 실세계의 복잡성을 정확하게 재현합니다. 데이터 파이프라인은 대규모 이미지를 수집하여 3DGS 장면을 생성하고, 공간 분할 및 다중 뷰 렌더링을 통해 훈련 타일을 생성하는 4단계로 구성되어 있습니다.

- **Performance Highlights**: ABot-Earth 0.5는 실제 도시 환경에 대한 상당한 개선을 보여주고 있으며, 다양한 메트로폴리탄 지역과 비도시 자연 지형에서 검증되었습니다. 이 모델은 고충실도의 Q&A 환경을 만들어 Embodied AI의 시뮬레이션 및 훈련 플랫폼으로 활용이 가능합니다. 결과적으로, 이 프레임워크는 대규모 3D 재구성의 기술적, 재정적 장벽을 낮추어 글로벌 디지털 지구 시각화를 향상시키는 데 기여합니다.



### WHU-Infra3D: A Full-stack Multi-modal Dataset and Benchmark for 3D Roadside Infrastructure Inventory (https://arxiv.org/abs/2606.09882)
- **What's New**: WHU-Infra3D는 도로 인프라 자산의 포괄적인 재고를 위한 대규모 멀티모달 벤치마크 데이터셋으로, 53.8 km에 걸쳐 세 도시의 파노라마 이미지와 LiDAR 포인트 클라우드를 통합합니다. 이는 인프라 자산의 속성과 상태 진단에 필요한 정밀한 다중 모드 정렬을 포함하여, 기존 데이터셋의 한계를 극복합니다. 이 데이터셋은 자동화된 기반 시설 관리의 새로운 기준을 제시하고, AI 기반 운영 건강 평가를 지원합니다.

- **Technical Details**: WHU-Infra3D는 2D 탐지, 2D 교차 보기 매칭, 3D 지오 식별, 3D 포인트 클라우드 분할 및 속성 인식의 다섯 가지 핵심 작업을 위한 포괄적인 기준선을 설정합니다. 이 데이터셋은 175,000개 이상의 2D 경계 상자와 수천 개의 3D 인프라 인스턴스를 포함하며, 다양한 속성 및 상태 주석을 제공합니다. 다양한 중국 도시들에서 수집된 고해상도 데이터는 인공지능 구동 인프라 재고 관리의 발전을 위한 테스트베드 역할을 합니다.

- **Performance Highlights**: 현재 모델의 장기적인 결함 상태에 대한 취약점을 드러내는 포괄적인 평가를 통해, WHU-Infra3D는 AI 기반 도시 자산 생애 주기 관리의 발전을 위한 필수 테스트베드로 자리잡았습니다. 이 데이터셋은 기존의 데이터셋들이 가지지 못한 감지 및 관리의 요구사항을 동시에 충족시키며, 새로운 알고리즘 일반화의 도전 과제를 제시합니다. 전반적으로 WHU-Infra3D는 도시 인프라 재고 관리의 혁신적인 방향성을 제공하고 있습니다.



### SD-GRPO: Verifiable Segment Decomposition for Long-Form Vision-Language Generation (https://arxiv.org/abs/2606.09871)
- **What's New**: 본 연구에서는 Segment-Decomposed GRPO (SD-GRPO)를 제안하여 청사진 유사 데이터에서 비주얼-언어(VL) 출력을 위한 세그먼트별 보상을 반영합니다. 기존의 GRPO는 단일 스칼라 보상을 사용하여 전체적인 장점을 계산하였으나, 이는 VL 작업에서는 부족함이 있었음을 지적합니다. SD-GRPO는 각 세그먼트를 독립적으로 검증할 수 있도록 나누어 보상을 z-정규화하여 각 세그먼트의 이점을 벡터로 만들어 GRPO의 한계를 극복하고 있습니다.

- **Technical Details**: SD-GRPO의 구현은 기존 GRPO와 크게 다르지 않으며, 각 롤아웃(output)은 그라운드 트루스에 맞춰 세그먼트로 구분됩니다. 세그먼트별 보상은 해당 세그먼트가 포함된 롤아웃 그룹 전체에서 z-정규화되어, 각 세그먼트의 장점 벡터를 생성합니다. 이 접근 방식은 Monte Carlo 롤아웃이나 학습된 비평자 없이도 가능하며, 세그먼트의 결과는 실제 출력과 비교되어 정확히 평가될 수 있습니다.

- **Performance Highlights**: SD-GRPO는 다양한 VL 작업에서 기존 GRPO보다 일관적으로 더 높은 성능을 보이며, 특히 세그먼트의 수가 많을수록 그 효과가 더욱 두드러집니다. 테스팅한 세 가지 벤치마크에서는 SD-GRPO가 최종 성능을 1.0-4.3 pp까지 향상시켰습니다. 더욱이, 세그먼트 간의 의미적 연결이 있는 실제 과제에서는 세그먼트별 보상과 전체 보상을 결합함으로써 추가적인 개선을 달성했습니다.



### Monte Carlo Pass Search: Using Trajectory Generation for 3D Counterfactual Pass Evaluation in Footba (https://arxiv.org/abs/2606.11120)
Comments:
          CVPR 2026, CVSports Workshop

- **What's New**: 이 논문은 축구 패스 평가를 Monte Carlo Tree Search (MCTS)와 유사한 평가 문제로 재구성하고, 각 구성 요소를 구별하여 설명합니다. 독일 분데스리가의 3D 볼 궤적을 포함한 첫 번째 공개 고충실도 추적 데이터셋을 활용하여 Monte Carlo Pass Search (MCPS)를 소개합니다. MCPS는 각 관찰된 패스에 대한 킥 파라미터를 추론하고, 실행 변형 및 옵션 변형을 샘플링해다 매치의 다음 볼 상호작용까지 후보를 롤아웃하여 결과를 점수화하여 가치를 분포합니다.

- **Technical Details**: MCPS는 볼 상호작용에 따른 미래를 예측하기 위한 멀티 에이전트 궤적을 모델링하는 생성 모델을 사용합니다. 패스 실행을 위한 가능성 있는 대체와 카운터팩추얼(countersfactual) 실행을 샘플링하고 이를 롤아웃하여 점수를 매깁니다. 이러한 과정에서 두 가지 실행 잉여 점수(mean-based와 percentile-based)를 사용하여 분석과 순위를 얻고, 이를 통해 축구 코칭 및 채용 workflows에 도움을 줍니다.

- **Performance Highlights**: MCPS는 자율주행(SMART)의 고성능 오토리그레시브(autoregressive) 궤적 생성을 통해 데이터가 제한된 상황에서도 샘플 효율성을 높였습니다. 20개의 예측 중 최고의 정확도를 제공하여 기존 기준선과 비교할 때 우수성을 입증했습니다. 이 논문에서는 모델 체크포인트와 코드를 공개하여, 축구 분석 연구 커뮤니티의 재현 가능성을 높이고, 이를 통해 발전을 촉진하고자 했습니다.



### Multimodal Brain Tumour Classification Using Feature Fusion (https://arxiv.org/abs/2606.11107)
- **What's New**: 이번 연구에서는 뇌종양을 진단하는 데 있어 임상의사들이 사용하는 다중 모달(멀티모달, multimodal) 추론을 모방하는 두 가지 가지의 네트워크를 제안합니다. 기존의 딥러닝 모델들이 MRI 및 CT 이미지에만 의존하는 것과 달리, 우리는 91개의 방사선학적 특성(radiomic features)을 결합하여 뇌종양을 분류합니다.

- **Technical Details**: 우리의 모델은 원본 MRI 스캔과 방사선학적 특성을 센싱(encoding)하기 위해 각각 CNN(합성곱 신경망)과 MLP(다층 퍼셉트론)를 사용합니다. 이미지를 처리하는 CNN과 방사선학적 데이터를 처리하는 MLP는 연결(concatenation), 게이트(gated), 양방향 크로스 모달(attention) 전략을 통해 통합됩니다.

- **Performance Highlights**: 실험 결과, 7,200개의 균형 잡힌 이미지 데이터셋을 사용한 아홉 번의 실험에서 모든 멀티모달 구성 요소가 단일 모달(unimodal) 기준선보다 우수한 성능을 보여주었습니다. 특히, 게이트 방식의 융합이 96.13%의 최고의 정확도를 달성하였습니다.



### A History-Aware Visually Grounded Critic for Computer Use Agents (https://arxiv.org/abs/2606.11078)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 HiViG라는 새로운 테스트 시간 개입 프레임워크를 소개합니다. HiViG는 과거 상호작용을 요약하여 자연어로 피드백을 제공하는 멀티모달 크리틱 모델에 기반하여, 그래픽 사용자 인터페이스(GUI) 환경에서의 결정 오류를 사전에 차단합니다. 이를 통해, 기존 모델들이 가지던 시각적 기초 부족 및 단기적인 결정 루프 문제를 해결합니다.

- **Technical Details**: HiViG는 역사(state) 추적 및 시각적으로 구체화된 오류 분석 기능을 결합하여 효과적으로 동작합니다. 이 프레임워크는 모집단 패턴과 시각적 상태를 비교하여 실수 행동을 식별하고, 과거 상호작용을 매크로 액션 기록으로 압축하여 정책의 목표 달성을 지원합니다. HiViG-critic 모델은 오픈 소스의 멀티 도메인 GUI 경로에서 구축된 훈련 데이터를 사용하여 학습됩니다.

- **Performance Highlights**: HiViG는 웹, 모바일 및 데스크탑 벤치마크에서 평균 성공률을 기존 강력한 기준 모델보다 각각 5.8% 및 9.0% 향상시키는 성과를 보였습니다. 특히, HiViG는 Gemini-3-Flash의 WebArenaLitev2 기준에서 성공률을 15.0%에서 30.5%로 개선하며 뛰어난 차세대 성능을 입증했습니다. 또한, HiViG는 다양한 GUI 환경에서 강력한 일반화 능력을 보여주며, 장기 GUI 작업을 보다 효과적으로 완수할 수 있도록 지원합니다.



### Architect-Ant: Editable Automatic Furnishing of Architectural Floor Plans (https://arxiv.org/abs/2606.10953)
Comments:
          17 pages, 10 figures

- **What's New**: 이 연구는 부동산 시각화와 인테리어 디자인을 위한 270개의 건축 평면도와 객체 수준의 가구 주석이 있는 중요 데이터셋인 AntPlan-270을 소개합니다. 또한, 편집 가능한 자동 가구 배치 프레임워크인 Architect-Ant를 제시하여, 공간적 제약을 고려한 가구 배치를 가능하게 합니다. 이 모델은 객체 카테고리와 방 기하학에 대한 상대적 위치를 표현하는 도메인 특화 언어(DSL)를 사용합니다.

- **Technical Details**: Architect-Ant는 절차적 추론(trace)을 생성하여 건축적 제약을 포착하고, 이를 통해 모델의 미세 조정을 감독합니다. 가구 배치는 객관적 유형, 크기 및 방의 경계 내 위치를 고려하여 접근 가능하고 가시적이어야 합니다. 이 과정에는 문, 창문과 같은 건축적 요소와 가구 객체의 카테고리, 위치 및 크기가 포함됩니다.

- **Performance Highlights**: 실험 결과, Architect-Ant는 기하학적으로 유효하고 기능적으로 타당한 레이아웃을 생성하며, 더 큰 구조 전용 평면도 데이터셋에 대한 가구 배치 채택의 확장 경로를 제시합니다. 이 모델은 실제 블루프린트 스타일의 가구 배치 이미지를 생성하면서도 상징적 레이아웃을 편집 가능한 상태로 유지합니다.



### XtrAIn: Training-Guided Occlusion for Feature Attribution (https://arxiv.org/abs/2606.10877)
Comments:
          12 pages, 7 figures, 1 table

- **What's New**: 본 논문에서는 XtrAIn이라는 새로운 기법을 제안하여, 입력 공간에서 파라미터 공간으로 오클루전(occlusion) 작용을 이동시키는 방법을 설명합니다. 이는 기존의 입력 값을 수동으로 바꾸는 대신, 모델의 훈련 경로를 따라 피처와 관련된 파라미터 업데이트가 출력 로짓(output logits)에 미치는 영향을 측정하는 방식입니다. XtrAIn은 피처 별로 업데이트를 전이해 깔끔하고 해석 가능한 속성 부여(attribution) 패턴을 생성합니다.

- **Technical Details**: 오클루전 기반 속성 부여 기법은 입력 특성을 제거하고 모델 출력을 측정함으로써 각 입력 피처의 기여도를 계량화합니다. 하지만 기존의 방법들은 입력 공간에서 피처를 제거할 때, 나머지 피처의 상호작용이 결과에 영향을 미치는 문제와 불안정한 기여도 평가에 대한 편향이 존재합니다. 이 논문은 입력 공간에서 오클루전 작용을 파라미터 공간으로 이동시키며, 이를 통해 훨씬 더 안정적이고 해석 가능한 결과를 제공합니다.

- **Performance Highlights**: 제안된 XtrAIn, Xstep 그리고 XtrAIn+ 방법은 PAM50 유방암 분류 작업 및 제어된 이미지 데이터 세트를 통해 검증되었습니다. 실험 결과, 제안한 방법들이 기존의 오클루전 기반 속성 기준보다 더 깔끔하고 안전하며 해석 가능성이 높은 설명을 생성하는 것으로 나타났습니다. CleanScore라는 새로운 진단 메트릭스도 제안되어 속성 부여의 청결도를 평가하는 데 도움을 줍니다.



### IMPACT: Learning Internal-Model Predictive Control for Forceful Robotic Manipulation (https://arxiv.org/abs/2606.10818)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 IMPACT라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 힘을 사용하는 로봇 조작 작업을 작업 계획(task-planning)과 내부 모델 기반 예측 제어(internal-model-based predictive control)로 분리하여 보다 효율적으로 수행할 수 있도록 합니다. 이를 통해 이전 방법들보다 더 높은 성공률과 물체의 무게에 대한 일반화 성능을 향상시켰습니다.

- **Technical Details**: IMPACT 프레임워크는 조작기가 손목에 장착된 힘-토크 센서를 통해 상호작용 힘을 직접 측정하는 대신 관절 토크 독서를 기반으로 힘을 추정합니다. 이 내부 모델은 상태와 행동의 이력을 바탕으로 예측된 상호작용 힘을 보상하도록 훈련됩니다. 특히, 이 모델을 통해 시스템은 다양한 외부 방해 요소에 적절히 대응할 수 있도록 합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험을 통해, IMPACT 프레임워크는 다양한 힘 조작 작업에서 높은 성공률을 기록하였으며, 반응성과 에너지 효율성 또한 개선되었습니다. 특히, 하이브리드 위치-힘 제어 작업인 서예 작업에서도 뛰어난 성능을 보였습니다. 이러한 성능은 다양한 무게의 물체를 다룰 때에도 안정적인 조작을 가능하게 합니다.



### Beyond APIs: Probing the Limits of MLLMs in Physical Tool Us (https://arxiv.org/abs/2606.10803)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 물리적 도구 사용 능력을 평가하기 위한 새로운 벤치마크인 PhysTool-Bench를 소개합니다. 물리적 작업에서 MLLMs가 도구를 인식하고 활용하는 데 있어 아직 탐구되지 않은 측면을 다루고 있습니다. PhysTool-Bench는 2,510개의 질의와 2,678개의 다양한 물리적 도구로 구성되어 있으며, 제조업, 전기 작업, 농업, 건강관리 등 여러 분야를 포함합니다.

- **Technical Details**: PhysTool-Bench는 물리적 도구 인식(Physical Tool Recognition)과 도구 선택 및 행동 계획(Tool Selection and Action Planning)이라는 두 가지 주요 작업으로 MLLMs의 성능을 평가합니다. 각 작업에서는 자연어 지침과 현실적인 환경 이미지를 결합하여 모델이 주어진 작업에 적합한 도구를 정렬하고 선택하는 과정을 돕습니다. 연구에서는 13개의 주요 MLLMs를 대상으로, 도구 인식에서 평균 58.7%를, 도구 선택 및 행동 계획에서는 평균 21.0%만 성공적으로 수행함을 보였습니다.

- **Performance Highlights**: 모델 성능 분석 결과, MLLMs는 현실적인 장면에서 도구를 인식하는 데 어려움을 겪고 있으며, 특히 계획 단계에서 큰 성능 저하를 보였습니다. 42-61%의 오류는 기능적으로 유사한 도구로 잘못 선택에서 발생하며, 이는 물리적 상식 부족이 주요 원인임을 시사합니다. 또한, 인간 고수준의 평가 결과는 평균 38%로, 최상의 MLLM이 21.0%에 그친 것과 비교해 많은 격차가 존재함을 보여주었습니다.



### ++nnU-Net: Scaling nnU-Net with Prefix-Based Data Augmentation (https://arxiv.org/abs/2606.10713)
Comments:
          7 pages, 1 figure, 2 tables

- **What's New**: 이번 논문에서는 nnU-Net의 연속적인 성공을 기반으로 한 새로운 데이터 증강 모듈인 ++nnU-Net을 제안합니다. 이 모듈은 이미지 등록(image registration)을 기반으로 하여 사전 처리 및 훈련 전 단계에서 작동합니다. 기존 nnU-Net에 비해 성능을 향상시키는데 중점을 두었으며, 다양한 2D 데이터 세트를 활용하여 평가했습니다.

- **Technical Details**: 이 시스템은 최대 8개의 인자를 수용하며, 이미지 및 세그멘테이션 데이터가 포함된 입력 디렉토리, 증강 후 최종 데이터 세트의 원하는 유형, 등록 및 변환 중 체크포인트를 저장할 옵션 등을 지원합니다. 기본 설정은 두 단계의 등록 과정으로 고정 및 변형 대칭 정규화(Symmetric Normalization, SyN) 변환을 포함합니다. 데이터 증강 파이프라인은 자동차 이식(AutoImplant) 2020 챌린지의 우승 솔루션을 적용하여 사용자의 기존 데이터를 변경하고 변형된 이미지를 생성합니다.

- **Performance Highlights**: ++nnU-Net은 nnU-Net의 기준 성능을 초과하여 Dice Similarity Coefficient 점수에서 약 22%의 성능 향상을 달성했습니다. 이러한 결과는 등록 기반 데이터 증강이 2D 의료 이미징 데이터 세트에서 특히 효과적임을 입증하며, 데이터가 제한된 환경에서도 세그멘테이션 성능을 향상시킬 수 있는 실용적이고 확장 가능한 접근 방식을 제공합니다.



### UniDexTok: A Unified Dexterous Hand Tokenizer from Real Data (https://arxiv.org/abs/2606.10683)
- **What's New**: 이 논문에서는 다양한 구현에서 손의 정밀한 조작을 위한 통합 손 모델(UDHM)을 제안합니다. UDHM은 인간의 손과 로봇 손의 상태를 공유하는 22-DoF(도)의 의미적 인터페이스로 매핑합니다. 이를 통해 서로 다른 손 구성에서의 데이터 사용이 용이해지며, 모델의 성능이 대폭 향상됩니다.

- **Technical Details**: 이 연구에서 제안하는 UniDexTok는 흔히 사용된 손 특화 토크나이저와의 차별점으로 모든 구현에서 단일 인코더, 코드북 및 디코더를 공유합니다. 이 설계는 다양한 손 구조 간의 전이 가능성을 증진시키며, 새로운 손이 도입될 때도 별도의 학습 없이기존의 토큰 공간으로 투영할 수 있게 합니다. UDHM과 UniDexTok을 기반으로 한 새로운 학습 파이프라인은 여러 데이터셋에 걸쳐 실제 손 데이터를 표준화하여 학습합니다.

- **Performance Highlights**: UniDexTok은 UniHM 대비 MPJAE(Mean Per Joint Average Error)를 15.63도에서 0.16도로, MPJPE(Mean Per Joint Position Error)를 18.51mm에서 0.18mm로 감소시켰습니다. 이는 각각 98.98%와 99.03%의 오류 감소를 의미하며, 단위가 센티미터에서 서브 밀리미터 정확도로 향상된 것을 보여줍니다. 추가 실험 결과는 다른 구현에서의 데이터가 목표 구현의 재구성 정확도를 개선하는 데 기여함을 보여주었습니다.



### Dexterous Point Policy: Learning Point-based Dexterous Hand Policies from Human Demonstrations (https://arxiv.org/abs/2606.10614)
- **What's New**: 로봇 기반 모델(robust foundation models)이 인간의 시연 비디오로 사전 훈련되었지만, 실제 로봇에 배포될 때 여전히 큰 일체감 차이(embodiment gap)가 존재합니다. 이를 해결하기 위해, 우리는 Dexterous Point Policy라는 새로운 프레임워크를 도입합니다. 이 프레임워크는 로봇 시연 없이도 인간 비디오에서 직접 정교한 조작(le dexterous manipulation) 정책을 학습합니다.

- **Technical Details**: 우리는 작업 관련 객체와 인간 손의 3D 키포인트를 원시 비디오(raw videos)에서 추출하여, 이 키포인트들을 통해 오토회귀(transformer) 모델을 훈련합니다. 통합된 3D 키포인트 표현(unified 3D keypoint representation)을 사용하여 인간과 로봇의 시청각 정보를 연결합니다. 특히, 손목(wrist)과 손끝(fingertips) 차원에서 인간과 로봇의 행동이 밀접하게 연관되어 있음을 확인하였습니다.

- **Performance Highlights**: Dexterous Point Policy는 실제 로봇 작업에서 75.0%의 성공률을 달성했습니다. 이는 첨단 VLA 기준선(state-of-the-art VLA baseline)의 1.0%와 비교할 때 현저히 높은 성과입니다. 또한, 우리의 방법은 다중 객체 환경(multi-object environments)과 새로운 객체 범주(novel object categories)를 포함한 보지 못한 시나리오에서 강력한 일반화 성능을 보입니다.



### Geometry-Aware Reinforcement Learning for 2D Irregular Nesting (https://arxiv.org/abs/2606.10611)
Comments:
          15 pages, 4 figures, 5 tables. Under review at the European Workshop on Reinforcement Learning (EWRL)

- **What's New**: 이 논문에서는 전통적인 2D 불규칙 네스팅 문제에 대한 휴리스틱 솔버들의 한계를 극복하기 위해 강화 학습( Reinforcement Learning )을 제안합니다. 새로운 아키텍처인 Polygons Transformer (PoT)를 도입하여 기하학적 요소를 인식하고, 이를 통해 최적화된 솔루션을 찾는 방법을 제시합니다. 코어 아이디어는 데이터로부터 기하학적 사전 지식을 자동으로 학습하여 탐색을 전략적으로 유도하는 것입니다.

- **Technical Details**: 논문에서는 Combinatorial Optimization Reinforcement Learning (CORL) 프레임워크를 활용하여 연속적인 2D 불규칙 네스팅 문제를 해결하는 방법을 설명합니다. PoT는 2D 컨티뉴어스 벡터 기하학을 인코딩하는데 특화된 신경망 아키텍처로, 다중 폴리곤 간의 상호작용을 가능하게 합니다. 우리가 제안하는 새로운 방법은 데이터를 통해 필요한 기하학적 사전 정보를 학습하고, 이를 통해 해법 공간을 효과적으로 탐색하는 과정을 포함합니다.

- **Performance Highlights**: 제안된 모델은 강화 학습을 통해 훈련되었으며, 기존 최고의 휴리스틱 솔버인 Sparrow와 비교했을 때 영역 활용 성능에서 매우 경쟁력 있는 결과를 보여줍니다. 이는 강화 학습이 정밀한 공간 작업을 위한 기하학적 인식을 효과적으로 발견하고 활용할 수 있는 가능성을 입증합니다. 궁극적으로, 이 연구의 결과는 PCB 제조와 같은 산업 응용 분야에서 대규모 생산으로 인한 경제적 및 생태적 이점을 제공함으로써 산업적 중요성을 지닙니다.



### Time-frequency localization of bird calls in dense soundscapes (https://arxiv.org/abs/2606.10407)
- **What's New**: 이번 연구에서는 수동 음향 모니터링(Passive Acoustic Monitoring, PAM)을 활용하여 조류의 음성을 시간과 주파수에서 보다 정확하게 위치 지을 수 있는 새로운 접근 방식을 제안합니다. 본 연구에서는 YOLO(You Only Look Once) 모델을 통해 조류의 소리를 밀집된 열대 환경에서 탐지하는 방법을 개발하였으며, Intersection over Minimum (IoMin)이라는 새로운 평가 메트릭을 도입하였습니다. 이로 인해 YOLO 모델은 기존 기준 성과에 비해 거의 두 배에 가까운 성능을 보여주었고, 이는 복잡한 소리 환경에서 동물 음성을 위치 지우는 데 유망한 대안이 될 수 있음을 시사합니다.

- **Technical Details**: 연구에서는 스펙트로그램(spectrogram)에서의 객체 탐지(object detection) 문제로 조류 음성을 정의하였으며, 이를 위해 YOLO 모델을 훈련시켰습니다. 음성을 개별적으로 탐지하여 전체 스펙트로그램에서 나타나는 잡음을 줄이고 분류 문제를 단일 클래스(class) 탐지 문제로 단순화합니다. 또한, STFT(Short-Time Fourier Transform)를 사용하여 원시 오디오 데이터를 주파수-시간 표현으로 변환하고 이를 YOLO 모델에 적합하도록 조정하는 과정을 설명합니다.

- **Performance Highlights**: YOLO 모델은 싱가포르의 밀집된 음향 환경에서 훈련된 결과, 기존 기준 대비 81.8%의 IoMin@50 F1-score를 기록하며 성능이 크게 개선된 것으로 나타났습니다. 또한, 하와이의 외부 분포(out-of-distribution) 음원에서도 58.6%의 정확도를 기록하여, 일반화 성능을 입증하였습니다. 이와 같은 결과는 다양한 환경에서 조류 음성을 정확히 탐지할 수 있는 가능성을 보여 줍니다.



### Do Vision-Language Models See or Guess? Measuring and Reducing Textual-Prior Reliance with a Phrasing-Controlled Benchmark (https://arxiv.org/abs/2606.10400)
Comments:
          17 pages, 7 figures, Submitted to EMNLP 2026

- **What's New**: 이 논문은 Vision-Language Models (VLMs)의 정확도를 측정하는 새로운 벤치마크를 제안합니다. 540장의 이미지와 이를 기반으로 한 네 가지 질문 변형을 통해 질문의 문구가 어떻게 모델의 응답에 영향을 미치는지 평가합니다. 기존 연구와의 차별점은 질문의 외형이 아닌 시각 정보를 고정한 상태에서 질문 문구의 변화만으로 모델의 의존도를 측정한 것입니다. 특히 Vision-Grounded 변형은 이미지를 직접 기반으로 작성하여 모델의 텍스트 의존성을 최소화합니다.

- **Technical Details**: 논문은 540장의 이미지로 구성된 벤치마크를 구축하고, 각 이미지에 대해 네 가지 질문 변형을 생성했습니다. 이 과정에서 각 변형은 시각 정보 없이 질문 문구만으로 모델의 정확도를 측정할 수 있게 설계되었습니다. 사용된 기술적 방법론 중 하나는 GRPO (Group Relative Policy Optimization)와 LoRA (Low-Rank Adaptation)를 활용하여 VLM의 텍스트 의존성을 낮추는 방법입니다. 이러한 기술들을 통해 열한 개의 VLM 모델이 평가되었으며, 각 모델의 응답 정확도는 다르게 나타났습니다.

- **Performance Highlights**: 모델의 성능 측면에서, 모든 VLM은 이미지 기반의 Vision-Grounded 변형에서 성능이 저하되었고, 특히 오픈 소스 모델들은 그 영향이 더욱 두드러졌습니다. 이 연구에서는 텍스트 전용 질문에서의 모델 성능이 1%에서 9%로 드랍하는 결과를 보여주어 이미지 의존도가 실제로 존재함을 확인했습니다. GRPO 후속 훈련은 모든 변형에서 성능 개선을 가져왔으며, 이는 일반적인 상황에서도 유의미한 성과로 이어집니다.



### What Spatial Memory Must Store: Occlusion as the Test for Language-Agent Memory (https://arxiv.org/abs/2606.10299)
Comments:
          23 pages, 6 figures

- **What's New**: 이번 연구는 언어-에이전트 메모리 시스템이 지리적 위치를 메모리에 연계하여 기존의 텍스트로는 제공할 수 없는 기하학적 이점을 제공하는 방법을 테스트하고 있습니다. 연구팀은 메모리 회상과 가시성을 분리하여, 기억된 정보가 시각적으로 접근 가능한지 여부가 메모리 저장 방식에 따라 달라진다는 것을 보여주었습니다. 새로운 실험 결과는 기하학적 메모리 구조가 텍스트 기반 인덱스보다 우월하다는 것을 입증했습니다.

- **Technical Details**: 이 연구는 '기억 궁전(memory palace)' 시스템을 사용하여 메모리를 저장할 때, 기하학적 정보를 포함시키는 것이 필수적임을 강조합니다. 기억 배열을 생성할 때 시각적 시스템이 필요하며, 이렇게 저장된 기하학적 정보는 에이전트가 메모리를 읽어오는 과정에서도 필수적입니다. 연구는 기하학적 저장 방식이 단순한 텍스트 기반 접근보다 어떻게 메모리 회상에 도움이 되는지를 보여주는 실험을 통해, 기하학과 격리된 저장 요구사항을 확인했습니다.

- **Performance Highlights**: 연구팀은 기하학적 기반의 메모리 시스템이 기존의 텍스트 기반 시스템에 비해 회상 능력이 월등히 우수함을 입증하였습니다. 피험자들은 특정 실험에서 기하학적 정보를 포함한 시스템이 이전의 방식을 초월하여 성공률을 크게 향상시킴을 보여주었습니다. 이러한 성과는 그래픽 및 텍스트 기반 시스템의 사용성을 향상시키는 데 있어 기하학적 요소가 필수적임을 시사합니다.



### Overlapped Wavelet Diffusion for Low-Light Image Enhancemen (https://arxiv.org/abs/2606.10280)
Comments:
          Advance published in IEICE Transactions on Information and Systems. DOI: https://doi.org/10.1587/transinf.2026PCP0006. Code: this https URL

- **What's New**: 이번 연구에서는 저조도 이미지 향상을 위한 Overlapped Wavelet Diffusion 프레임워크를 제안합니다. 이 프레임워크는 블로킹 아티팩트(Blocking Artifact)를 방지하고 세부 정보를 보존하는 두 가지 보완 요소로 구성되어 있습니다. 기존의 확산 기반 LLIE 방법들이 전통적인 접근법에 비해 뛰어난 성능을 보이고 있지만, 여전히 Haar Wavelet Transform과 HFRM의 한계로 인해 발생하는 블로킹 아티팩트와 흐릿한 경계 문제를 처리하지 못하고 있었습니다.

- **Technical Details**: 새롭게 도입한 Overlapped WT는 인접 지역 간의 상관관계를 통합하여 구조적으로 블로킹 아티팩트를 방지합니다. 또한, 세부 복원을 강화하기 위해 저주파 유도 High-Frequency Enhance Block을 결합하여 보다 선명한 경계와 신뢰성 높은 텍스처를 제공합니다. 이러한 통합적인 접근 방식은 Haar WT와 HFRM의 구조적 한계를 효과적으로 해결합니다.

- **Performance Highlights**: OWDiff라고 명명된 우리의 프레임워크는 LOLv1과 LOLv2-real 데이터셋에서 기존 LLIE 방법들보다 일관되게 뛰어난 성능을 보여주었습니다. 시각적 품질의 개선과 함께 계산 효율성을 유지하면서 평균 PSNR은 0.58 dB 증가하고, SSIM은 1.64% 상대 개선, LPIPS는 5.9% 상대 감소를 기록했습니다.



### POPSICLE: Benchmark Datasets for Segmentation and Localization in CryoE (https://arxiv.org/abs/2606.10255)
- **What's New**: 이 논문은 POPSICLE이라는 기초 데이터 세트를 소개합니다. 이 데이터 세트는 CryoET(크라이오 전자 단층 촬영)의 세분화와 거대 분자의 위치 지정을 평가하기 위한 통합 벤치마크입니다. POPSICLE은 다양한 생물과 샘플에 걸쳐 마련된 데이터 기반 위에 구축되었으며, 새로운 데이터와 주석이 추가되는 대로 확장될 수 있도록 설계되었습니다.

- **Technical Details**: POPSICLE은 CryoET Data Portal이라는 공개된 리포지토리 위에 구축되어 있습니다. 이 포털은 표준화된 단층 촬영 데이터, 주석, 메타데이터를 제공하며, 기계 학습(Machine Learning) 분석에 적합하도록 준비되어 있습니다. 두 가지 주요 작업인 밀집 세분화와 희소 위치 지정을 포함하여, 데이터를 조직하고 작업 정의 및 평가 절차를 표준화함으로써 재현 가능한 평가를 가능하게 합니다.

- **Performance Highlights**: 기존의 단일 작업에 의존하지 않고, 밀집 세분화와 희소 위치 지정을 함께 고려하여 모델의 일반화 능력을 평가합니다. 평가 결과, 각 작업에 대해 모델의 성능은 상당히 다르게 나타났으며, 이는 Adjacent domains(인접한 영역)의 모델 아키텍처와 평가 관행이 변형 없이 적용될 수 없음을 보여줍니다. POPSICLE은 2,993개의 주석이 달린 단층 촬영 데이터로 구성되어 있으며, 이는 다양한 생물학적 및 실험적 설정을 포함합니다.



### Dual-Branch Gated Fusion for Open-Set Audio Deepfake Source Tracing (https://arxiv.org/abs/2606.10223)
- **What's New**: 이 논문은 합성 발화(synthetic utterance)의 출처를 정확히 추적하기 위한 새로운 이중 분기 게이트 융합 프레임워크(dual-branch gated fusion framework)를 제안합니다. 기존의 닫힌 집합(closed-set) 모델이 새로운 합성기(synthesizers)를 거부하지 못하고 지나치게 자신감 있는 예측을 내리는 한계점을 극복하고자 합니다. 특히, XLSR-53 모델과 CORES라는 66차원 설명자를 결합함으로써, 서로 다른 합성 아티팩트(synthesis artifacts)를 효과적으로 캡처하는 방안을 모색합니다.

- **Technical Details**: 제안된 프레임워크는 입력된 데이터에 따라 각 분기를 동적으로 가중치 조정하는 입력 조건부 게이트(input-conditioned gate)을 채택합니다. 이 과정에서 교차 엔트로피(cross-entropy) 손실과 에너지 마진 손실(energy margin loss)을 이용하여 도메인 내(In-domain)와 도메인 외(Out-of-domain) 분리를 수행합니다. 특히, CORES는 선형 필터 뱅크(linear filter bank)만을 사용하는 이전 연구들과는 달리, 파형의 다양한 차원을 아우르며 합성 오디오(synthesized audio)를 자연 음성(natural speech)과 구별하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 제안된 시스템은 MLAAD 벤치마크에서 97.6%의 ID 정확도와 4.9%의 EERc, 83.5%의 상대적 FPR95 감소를 달성하여 Interspeech 2025 기준을 넘어서는 성능을 보여줍니다. 이 시스템은 고정 가중 융합(fixed-weight fusion) 및 단일 스트림 SSL 아키텍처가 동시에 달성할 수 없는 균형을 유지하며, 개방형 집합(open-set)에 대한 민감성을 유지하면서도 강력한 분류 성능을 제공합니다.



### Density Ridge Selective Prediction for LLM and VLM Hallucination Detection under Calibration Label Scarcity (https://arxiv.org/abs/2606.10198)
- **What's New**: 이 논문은 대형 언어 모델(LLM) 및 비전-언어 모델에서 환각 탐지를 선택적 예측(selective prediction)으로 새롭게 구성하였습니다. 저자들은 새로운 리지 기반(score based on ridge) 평가 방법을 통해 기존의 여러 방법보다 저조한 레이블 환경에서도 높은 성능을 보이는 것을 보여주었습니다. 이 방법은 응답 매니폴드(response manifold)를 회복하고 이를 커널 밀도 추정(kernel density estimate)의 밀도 리지(density ridge)로 설명합니다.

- **Technical Details**: 저자들은 샘플링된 응답의 숨겨진 상태 거동을 분석하여, 높은 차원의 피쳐 맵을 통해 저차원의 리지 구조를 형성합니다. 각 생성 결과는 이 리지에 대한 근접성에 따라 점수가 매겨지며 이는 유클리드 거리의 부정값으로 계산됩니다. 또한, 저자들은 오프-리지 거리(off-ridge distance)를 사용하여 테스트 쿼리에 대한 신뢰 scores를 제안하며, 강한 타겟 밀도를 가진 샘플들에 대해 검증합니다.

- **Performance Highlights**: 제안된 방법은 Semantic Entropy, EigenScore, SAPLMA와 같은 기존 방법들과 비교하여 seven QA benchmarks에서 AUROC(Area Under Receiver Operating Characteristic curve) 점수에서 5-20점 높은 성능 향상을 기록했습니다. 또한, 레이블 희소성(calibration label scarcity) 하에서도 성능 저하가 제한적임을 확인하였으며, 이는 저자들이 제안한 리지 기반 점수의 유용성을 입증합니다.



### From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs (https://arxiv.org/abs/2606.10147)
Comments:
          40 pages, 29 figures

- **What's New**: 이번 연구에서는 오디오-비주얼 대규모 언어 모델(AVLLMs)의 정보 흐름을 분석하여 오디오와 비주얼 정보가 최종 예측에 어떻게 영향을 미치는지를 살펴봅니다. AVLLMs는 두 가지 입력 구성, 즉 오디오-비주얼 비디오와 여러 상호 얽힌 오디오-비주얼 항목에 대한 정보를 통합하는 과정을 밝혀냈습니다. 기존의 연구에서 각 모달리티( modality)가 독립적으로 처리되던 것과는 달리, AVLLMs는 오디오와 비주얼 정보를 통합하여 복잡한 질의에 대한 답변을 가능하게 합니다.

- **Technical Details**: AVLLMs는 오디오, 비디오, 텍스트 지침을 통해 인터리브( interleaved)된 토큰 시퀀스를 처리합니다. 이 모델은 비전 인코더와 오디오 인코더를 사용하여 각각의 입력을 토큰으로 변환하며, 이는 최종적으로 Transformer 레이어를 통과하여 예측 결과를 도출합니다. 연구에서는 AVLLMs가 채택한 다양한 경로 구조가 어떻게 정보를 흐르게 하는지를 탐구하며, 두 개의 주요 정보 흐름 방식, 즉 순차적 경로와 병렬 경로를 정의합니다.

- **Performance Highlights**: AVLLMs의 실험 결과, 오디오와 비주얼 정보의 흐름이 각 작업에 따라 조정되는 것을 확인했습니다. 모델의 예측 정확도는 토큰 정보가 전송된 후 불필요한 토큰을 삭제해도 큰 영향이 없거나, 오히려 소폭의 개선을 나타냈습니다. 이러한 발견은 AVLLMs가 다양한 작업과 데이터셋에 대해 효율적으로 작동함을 보여주며, 향후 해석 가능성 및 설계 개선을 위한 기반을 제공합니다.



### Continuous Neural Reparameterization as a Deep Geometric Prior for Robust Fixed-Chart UV Repair (https://arxiv.org/abs/2606.10050)
- **What's New**: 이 논문에서는 기존 UV 언랩핑 방식을 지속적인 신경 재파라미터화(continuous neural reparameterization)로 변형하여, 유효한 초기값 지정을 가능하게 하고 잘못된 초기화, 국소 최소값, 토폴로지 접기 문제를 해결하였습니다. 특히, SIREN 네트워크를 사용하여 각 메쉬(Vertex)의 특성을 UV 좌표로 매핑하며 수학적 모델의 유효성을 유지하기 위한 명시적 접근 방식을 채택했습니다. 이로 인해 UV 언랩핑의 효율성이 크게 향상되었습니다.

- **Technical Details**: 제안된 방법은 고정된 차트 UV 언랩핑을 다루며, SIREN 네트워크와 Laplace-Beltrami 고유함수를 결합하여 메쉬의 국소 유효성을 유지합니다. 이를 위해 유효한 Tutte 맵의 피팅, 결정 집중 안목의 최적화 및 부적합한 생성 조각의 rerouting을 포함하는 복잡한 알고리즘을 제시합니다. 성능 분석을 통해 기존의 SLIM, BFF 및 OptCuts와 비교하여 고정된 차트 신경 최적화가 어느 상황에서 도움이 되는지를 명확히 밝혔습니다.

- **Performance Highlights**: 논문에서는 Thingi10K 데이터셋을 포함한 여러 실험을 통해, 제안된 신경 언랩퍼가 모든 컴팩트 차트에서 0번의 뒤집힘 없이 유효한 솔루션을 생성함을 확인했습니다. 또한, Amara Spatial에서 생성된 메쉬에 대한 전반적인 아틀라스 구축 경로에서는 1000개의 strict locally valid 아틀라스를 0번의 UV 뒤집힘으로 성공적으로 완성했습니다. 총체적인 비교 분석을 통해 신경 솔버가 제공하는 정확성과 유효성을 입증하였습니다.



### GHOST: Hierarchical Sub-Goal Policies for Generalizing Robot Manipulation (https://arxiv.org/abs/2606.10025)
Comments:
          Accepted at RSS 2026

- **What's New**: GHOST는 환경을 조작하는 정책을 학습하는 새로운 프레임워크로, 기존의 데이터 훈련 분포를 넘어 일반화될 수 있는 방법을 제시합니다. 이 프레임워크는 조작 수행을 위한 높은 수준의 정책과 낮은 수준의 목표 조건 제어기를 계층적으로 분리하여 기존의 flat Diffusion Policy보다 성능과 견고성을 향상시킵니다. GHOST의 주요 혁신 중 하나는 3D 하위 목표를 이미지 평면에 투영하여 엔드 이펙터 열지도로 표현하는 간단한 공간 인터페이스를 도입한 것입니다.

- **Technical Details**: GHOST는 두 가지 모듈로 구성된 계층적 정책을 정의합니다. 첫 번째로, 높은 수준의 정책 𝜋ᵗᵢ는 다중 뷰 RGB-D 관찰 및 언어 지시를 바탕으로 다음 하위 목표를 예측합니다. 두 번째로, 낮은 수준의 정책 𝜋ₗₒ는 예측된 목표 기반으로 행동을 생성합니다. 이 구조는 데이터 효율을 높이며, 조작 환경의 다양성에 쉽게 적응할 수 있도록 돕습니다.

- **Performance Highlights**: GHOST는 Robot의 시연을 통한 훈련만으로도 기존 정책보다 높은 성공률을 기록합니다. 이 프레임워크는 인간 영상 자료를 활용하여 고급 정책을 훈련하며, 이는 action retargeting의 필요 없이도 새로운 작업 변형에 대한 일반화를 가능하게 합니다. GHOST는 최종적으로 새로운 물체와 작업 변형에 적은 수의 인간 시연으로도 적응할 수 있는 능력을 입증합니다.



### SPARX: Secure and Privacy-Aware Approximate CNN Acceleration with Edge RISC-V SoC (https://arxiv.org/abs/2606.09946)
Comments:
          Under review in 12th International Symposium on Smart Electronic Systems (iSES) 2026

- **What's New**: 이 논문에서는 SPARX라는 새로운 프레임워크를 제안하여 에지 AI 시스템에서 CNN(Convolutional Neural Network) 추론의 보안성과 프라이버시 인식을 통합할 수 있도록 합니다. SPARX는 RV32IMC RISC-V 시스템온칩(SoC)에서 작동하며, 커스텀 RISC-V 명령어 확장, 로그 기반 CNN 가속기, 경량의 차등 노이즈 기반 프라이버시 엔진 및 챌린지-응답 인증 메커니즘을 결합합니다. 이 프레임워크는 정확성, 전력 소비 및 성능 간의 균형을 위해 모든 작업 모드를 동적으로 선택할 수 있는 가능성을 제공합니다.

- **Technical Details**: SPARX 아키텍처는 CNN 가속기와 런타임에서 선택 가능한 근사 산술을 통합하고, 경량의 프라이버시 보호 및 인증 인프라를 사용하여 보안 추론을 가능하게 합니다. RISC-V 프로세서는 프로그램 가능성과 제어 기능을 제공하며, CNN 가속기는 이식성을 높이고 전력 효율성을 최적화하기 위해 전용 BRAM 메모리를 사용합니다. SPARX는 또한 정확하거나 근사적인 실행 모드를 선택할 수 있도록 하는 커스텀 명령어 포맷을 도입합니다.

- **Performance Highlights**: SPARX는 일반적인 radix-4 Booth MAC 대비 51.7%의 면적 감소, 81.5%의 전력 감소 및 2.13배의 처리량 개선을 이루었습니다. FPGA 구현에서는 Xilinx VC707 플랫폼을 사용하여 58.4 GOPS/W의 에너지 효율성을 달성했습니다. 이 시스템은 11개의 최신 근사의 MAC 아키텍처 평가에서 ILM(Iterative Logarithmic Multiplier)가 가장 적합한 설계로 선정되어, 전반적인 정확성, 처리량, 면적 및 전력 측면에서 최상의 균형을 제공합니다.



### Bypassing Copyright Protection in Diffusion-based Customization via Two-Stage Latent Feature Optimization (https://arxiv.org/abs/2606.09909)
Comments:
          accepted by KDD 2026

- **What's New**: 이번 연구에서는 Diffusion 기반의 커스터마이제이션에서 저작권 침해 문제를 해결하기 위해 새로운 공격 방법인 Two-Stage Latent Feature Optimization (TS-LFO)를 제안합니다. 기존의 방어법이 주입한 적대적 변동이 보호된 컨텐츠의 개인화 생성 능력을 저하시킨다는 점을 발견하였으며, 이를 활용하여 공격을 설계했습니다. TS-LFO는 고주파 적대적 변동을 제거하고 저주파 의미 정보를 회복하여, 모델이 보호된 시각적 컨텐츠를 정확하게 재구성할 수 있도록 합니다.

- **Technical Details**: TS-LFO의 첫 번째 단계인 Latent Denoising Stage에서는 Latent-Image Alignment Loss와 Latent Diffusion Loss를 결합하여 보호된 입력 이미지와의 의미적 일관성을 회복합니다. 이후 Latent Reconstruction Stage에서는 픽셀 수준 제약을 가하여 저주파 의미 충실도를 복원하고, 변동된 잠재 표현에서 원래 보호된 시각적 컨텐츠를 재구성합니다. 이를 통해 LDMs의 잠재 이미지 매핑을 효과적으로 이용한 저작권 공격을 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TS-LFO는 DiffPure, GrIDPure, IMPRESS와 같은 기존의 최첨단 저작권 방어를 지속적으로 우회하며, 다양한 설정에서 우수한 성능을 보였습니다. TS-LFO는 현재 상용화되어 있는 LDMs 보호 방법을 효과적으로 무력화하는 것을 증명했습니다. 이러한 결과는 Diffusion 기반의 개인화 생성의 저작권 보호 전략에 새로운 관점을 제시합니다.



### On the Controllability-Fidelity Frontier in Diffusion Editing (https://arxiv.org/abs/2606.09901)
Comments:
          Preprint

- **What's New**: 이번 연구는 통제 가능한 확산 기반 이미지 편집의 이론적 및 경험적 분석을 제공합니다. 사용자의 의도에 대한 적응성과 원본 이미지의 보존 간 균형을 다루며, 텍스트 및 마스크 가이드 편집, 포인트/드래그 조작 및 역전파 파이프라인을 포함한 다양한 편집 방법론을 제공합니다. 이 연구는 기존 방법들이 직면하는 문제와 이러한 문제의 해결 방법을 제시합니다.

- **Technical Details**: 편집 과정은 사용자 지침과의 정렬을 촉진하는 데이터 충실도 항목과 타겟 영역 외부의 원본 이미지와의 차이를 패널티로 부과하는 정규화 항목 간의 균형 맞추기 최적화로 구성됩니다.  이론적으로 우리는 역전파 재구성 오류와 오류 전파에 대한 경계 조건을 도출하고, 편집의 지역성과 안정성에 대한 보장을 증명하며, 실용적으로는 다양한 편집 파이프라인을 위한 알고리즘적 절차를 설계했습니다.

- **Performance Highlights**: 다양한 최첨단 방법들과의 비교 실험을 통해 재구성 오류, FID, 정체성 보존 등의 지표들에서 이 연구가 도출한 결과의 중요성을 보여주며, 편집 작업에서의 주요 한계점, 즉 정체성 드리프트와 프롬프트 민감도가 발생함을 확인했습니다. 또한, 이미지 편집을 위한 안전성과 윤리적 측면을 다루며, 문제의 개념을 제거하는 기법인 MACE, ANT, EraseAnything와 같은 안전 장치의 필요성도 강조했습니다.



### Toward Calibrated, Fair, and accurate Deepfake Detection (https://arxiv.org/abs/2606.09881)
- **What's New**: 이 논문에서는 Deepfake 탐지기에서 인종 및 성별을 포함한 인구 집단 간에 성능 차이가 크다는 문제를 다룹니다. 현재의 공정성 접근법은 인구 집단 라벨이 필요하거나 재학습을 요구하며, 정확도를 희생해야 합니다. 새로운 방법론으로 Face-Fairness (FF)를 소개하며, 이는 인구 집단 라벨 없이도 공정성을 보장할 수 있는 첫 번째 프레임워크입니다.

- **Technical Details**: 우선 기여를 통해 소개된 Face-Feature Tuning (FFT)은 인구 집단 라벨이 필요 없는 공정성 방법으로, 고정된 얼굴 임베딩을 기반으로 로짓 리매핑(logit remapping)을 수행하는 가벼운 조정기입니다. 또한, FF-Max와 FF-Discover 두 가지 변형을 통해 제공되며, 각각 인구 통계가 있는 경우와 임베딩으로 발견된 그룹에 대해 최악의 그룹 정확도를 최대화합니다. 이 접근법은 모든 탐지기에 대해 독립적이며, 실행 속도 저하를 최소화하고 신원 속성에 대한 접근이 필요하지 않습니다.

- **Performance Highlights**: FF 프레임워크는 도메인 내 및 교차 데이터셋 테스트 설정에서 일관되게 FPR/TPR 간의 격차를 줄이고 최소 그룹 정확도를 향상시키며, 전체 정확도를 유지하거나 종종 개선합니다. 이러한 결과는 Deepfake 탐지에 있어 공정성을 갖춘 성능 향상을 가능하게 합니다.



### MinhwaNet: Faithful but Insufficient Object Grounding in Korean Folk Painting (https://arxiv.org/abs/2606.09855)
- **What's New**: 이번 연구에서는 한국 민화(minhwa)의 상징적 요소와 장르 간의 관계를 이해하기 위해, 객체 grounded 접근 방식이 장르 예측에 예상보다 덜 효과적이라는 점을 강조합니다. 연구에 사용된 corpus는 전체 그림과 함께 전문가가 설명한 세부 그림들이 결합되어 있어 이러한 분석을 가능하게 하였습니다. 연구 결과는 특정 상징이 여러 장르에 걸쳐 사용되지만, 그 배치 방식이 장르를 결정하는 데 핵심적이라는 점을 제시합니다.

- **Technical Details**: MinhwaNet이라는 다중 모달 시스템을 구축하여, 이미지와 큐레이터 텍스트의 융합이 장르 분류에 미치는 영향을 조사했습니다. 연구에서는 포괄적인 큐레이션 레이블, 한국어 및 영어 캡션 필드, 및 전문가의 개별 객체 크롭을 포함한 데이터를 활용하였습니다. 특히, 이 시스템은 부분 수준의 객체 에vident map을 생성하여 장르 예측의 정확성을 분석하는 데 사용되었으며, 이러한 예측은 큐레이터가 분리한 상징적 객체의 위치와 일치하는 경향이 있음을 보여주었습니다.

- **Performance Highlights**: 결과적으로, 이미지와 큐레이터 텍스트의 융합은 장르와 같은 내용 레이블에 대해 유의미하게 개선됨을 확인할 수 있었습니다. 그러나 객체 grounded 방식으로 장르 표현을 강요하는 것은 오히려 예측 정확성을 해치는 것으로 나타났습니다. 또한, 이미지와 텍스트의 융합이 장르 분류에는 도움이 되었으나, 스타일 레이블에는 효과가 없다는 점이 관찰되었습니다.



### Sketch-to-Layout: A Human-Centric Computational Agent for Constraint-Aware Synthesis of Modular Photobioreactors (https://arxiv.org/abs/2606.09849)
Comments:
          13 pages, 6 figures

- **What's New**: 이 논문은 건물 통합 광생물 반응기(PBRs)를 통한 탄소 중립 건축을 가능하게 하는 모듈형 PBR 외관 시스템을 소개하고 있습니다. 이 시스템은 설계 의도와 물리적 유효성이 조화롭게 이루어지도록 지원하는 컴퓨터 기반 프레임워크에 의해 구동됩니다. '탄소 중립화 벽돌(carbon-neutralization bricks)'을 도입하여 통합된 용기와 관 geometry를 제공합니다.

- **Technical Details**: 이 연구에서는 14개의 모듈형 geometry의 조합 복잡성을 해결하기 위해 제약 만족 문제(Constraint Satisfaction Problem, CSP)로 레이아웃 합성을 공식화하는 Computational Sketch-to-Layout Agent를 개발했습니다. CP-SAT 엔진을 사용하여 이 에이전트는 희소한 사용자 스케치를 소프트 프라이어로 처리하고, 포트 정렬 및 글로벌 연결성과 같은 하드 제약을 강제합니다. 이를 통해 비전문가도 실시간에 가까운 속도로 제작 준비가 완료된 구성을 생성할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과 CSP 솔버는 최대 15 x 15 그리드에서 95.5%의 성공률을 달성하였습니다. 질적 평가를 통해 이 프레임워크가 설계의 의미를 유지하면서 작업의 완전성을 보장함을 확인했습니다. 장기 테스트 결과 비전 모듈이 14일 생물학적 주기에 맞춘 건강 경로를 생성하는 것을 보여주었으며, 상호작용 합성과 저비용 컴퓨터 비전의 통합이 확장 가능한 탄소 포집 시스템을 민주화할 수 있음을 시사합니다.



### Integrated Real-Time Motion Tracking and AI Analysis for Athletic Performance Optimization (https://arxiv.org/abs/2606.09842)
Comments:
          6 pages, 10 figures, 2 tables, IC2E3-2026 conference

- **What's New**: 이 논문은 실제 환경에서의 Human Pose Estimation (HPE)을 다루며, 스포츠 분석을 위한 실시간 HPE 접근법과 그 한계를 탐구합니다. 기존의 마커 기반 모션 캡처 시스템에서 현대의 마커리스 딥러닝 접근법으로 전환되는 과정을 살펴보고, 효율성과 정확성을 균형 있게 유지하는 기초 아키텍처를 조사합니다.

- **Technical Details**: 논문은 실용적인 배포 지표인 추론 지연(inference latency), 프레임 속도(frame rate), 평균 관절 위치 오차(mean per-joint position error), 시간적 지터(temporal jitter) 등을 비교하여 모델 선택 과정을 안내합니다. 주요 기여로는 MediaPipe HPE 프레임워크를 활용한 모듈식 경량 소프트웨어 프로토타입을 제안하여, 비전문가 사용자를 위한 실시간 통찰력과 AI 기반 피드백을 제공합니다.

- **Performance Highlights**: 이 시스템은 최소한의 계산 자원으로 스포츠 통찰력을 유도하고 피드백을 제공하며, 성능과 신뢰성 지표를 보여줍니다. 마지막으로, 센서와 AR/VR을 결합하는 등 향후 연구 방향을 제시하여, 연구자 및 엔지니어, 스포츠 과학자들에게 기술적 자원과 실시간 HPE 분석 시스템 구현을 위한 유효한 청사진을 제공합니다.



New uploads on arXiv(cs.AI)

### The Role of Feedback Alignment in Self-Distillation (https://arxiv.org/abs/2606.11173)
Comments:
          Accepted to the ICML 2026 Workshop on RL from World Feedback (RLxF)

- **What's New**: 이 논문에서는 자가 증류(self-distillation) 방법을 통해 언어 모델이 추가 컨텍스트(context) 없이 개선된 응답을 유지하도록 훈련하는 새로운 방법론을 제시합니다. 이 방법은 학생 모델이 질문만 보는 상황과, 자체 교사가 추가 컨텍스트를 함께 보는 두 가지 설정에서 모델의 출력 분포를 일치시키는 방식으로 작동합니다. 특히, 피드백 구조가 모델이 학습하는 내용에 미치는 영향을 분석하였습니다.

- **Technical Details**: 자가 증류(self-distillation)는 기존의 지식 증류(knowledge distillation)에서 교사 모델의 로짓(logits)에 대한 접근이 필요 없으므로 이점이 있습니다. 또한 이 논문은 세 가지 다른 피드백 조건을 비교합니다: (i) 이진 보상(GRPO), (ii) 기준 솔루션(reference solution), (iii) 경과 정렬(step-aligned) 비평을 포함합니다. 연구 결과, 단계 정렬 비평(StepAlignFB)이 다른 두 조건보다 높은 성능을 보였습니다.

- **Performance Highlights**: 단계 정렬 피드백은 16.11 포인트 GRPO를 초과하며, 기준 솔루션 조건 자가 증류보다 5.27 포인트 향상된 성능을 보였습니다. 이는 단계 정렬 피드백이 올바른 행동을 유지하면서도 잘못된 추론 부분에 집중하여 신호를 전달하기 때문입니다. 이로 인해, 피드백과 해결자의 추론 간의 구조적 정렬이 자가 증류의 효과성에 중요한 요인임을 밝혀냈습니다.



### ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation for Reasoning Models (https://arxiv.org/abs/2606.11164)
- **What's New**: 이번 논문에서는 ReasonAlloc이라는 새로운 프레임워크를 제안하여, decoding-time에서의 키-값 (KV) 캐시 압축 문제를 계층적 자원 할당 문제로 재구성합니다. ReasonAlloc은 아키텍처 기반의 오프라인 레이어-와이즈(preallocation) 전략과 온라인 헤드-와이즈(online head-wise) 재배분 전략을 결합하여, 추론 모델의 자원 요구 패턴을 효율적으로 관리합니다. 이 새로운 접근법은 기존의 균일한 예산 분배 방법론보다 우수한 성능을 보입니다.

- **Technical Details**: ReasonAlloc는 두 가지 보완적인 레벨에서 작동합니다: 오프라인 레이어-와이즈 전략은 'Reasoning Wave'라는 아키텍처 기반 요구 패턴을 포착하고, 온라인 헤드-와이즈 전략은 실제 디코딩 중 정보가 많은 헤드에 리소스를 재할당합니다. 기존의 전국적 예산 할당 방식은 특정 레이어와 헤드 간의 비효율성을 고려하지 못하는 문제를 해결합니다. 이 방법론은 MATH-500 및 AIME 2024와 같은 수학적 추론 벤치마크에서 테스트 되었으며, 기존의 압축 방법론에 비해 탁월한 성능을 인증받았습니다.

- **Performance Highlights**: ReasonAlloc은 128-512 토큰의 소규모 예산에서 가장 큰 성과 향상을 보여 주었으며, 기존의 균일 예산 방법과 비정형 분배 방식에 비해 월등히 개선된 결과를 도출했습니다. 이 프레임워크는 기존의 토큰-퇴출(token-eviction) 정책과의 호환성이 뛰어나며 최소한의 추론 시간 오버헤드를 발생시킵니다. 실험 결과는 ReasonAlloc의 효율성을 강력하게 입증하고 있습니다.



### ABC-Bench: An Agentic Bio-Capabilities Benchmark for Biosecurity (https://arxiv.org/abs/2606.11150)
Comments:
          18 pages. To be published in ICML 2026

- **What's New**: 이 논문에서는 새로운 AI capabilities가 생물학 연구에 큰 기여를 할 수 있는 가능성을 제시합니다. 특히, Agentic Bio-Capabilities Benchmark (ABC-Bench)라는 평가 도구를 소개하여 LLM(large language model) 에이전트가 생물학에서의 다양한 작업을 수행하는 능력을 측정하고자 합니다. 이 평가는 생물학과 소프트웨어 전문 지식을 결합해야 하는 복잡한 작업을 포함하고, 나아가 생물 보안(biosecurity) 문제도 고려하고 있습니다.

- **Technical Details**: ABC-Bench는 DNA 조각 설계, DNA 합성 스크리닝 회피 등 생물학 관련 작업을 평가합니다. 이 평가 도구는 OpenAI의 o4-mini-high 모델을 포함한 여러 AI 모델을 테스트하는 데 사용되었으며, 각 모델의 성능을 전문가의 기준과 비교하여 해석합니다. ABC-Bench는 생물학적 작업에서의 성능을 평가하기 위해 여러 다양한 작업을 포함하고 있으며, 이는 에이전트들이 실제 실험 환경에서 수행하는 작업을 포함합니다.

- **Performance Highlights**: 연구 결과, 모든 테스트된 LLM 에이전트들이 세 가지 작업 모두에서 평균 전문가 성과를 초과하여 뛰어난 성과를 보였습니다. 특히, 출판된 지식과 잘 정리된 프로토콜에 기반한 작업에서 우수한 결과를 나타냈습니다. 그러나 새로운 생물정보학(bioinformatics) 추론이 필요한 작업에서는 상대적으로 낮은 성과를 나타냈습니다.



### Monte Carlo Pass Search: Using Trajectory Generation for 3D Counterfactual Pass Evaluation in Footba (https://arxiv.org/abs/2606.11120)
Comments:
          CVPR 2026, CVSports Workshop

- **What's New**: 이 논문은 축구 패스 평가를 Monte Carlo Tree Search (MCTS)와 유사한 평가 문제로 재구성하고, 각 구성 요소를 구별하여 설명합니다. 독일 분데스리가의 3D 볼 궤적을 포함한 첫 번째 공개 고충실도 추적 데이터셋을 활용하여 Monte Carlo Pass Search (MCPS)를 소개합니다. MCPS는 각 관찰된 패스에 대한 킥 파라미터를 추론하고, 실행 변형 및 옵션 변형을 샘플링해다 매치의 다음 볼 상호작용까지 후보를 롤아웃하여 결과를 점수화하여 가치를 분포합니다.

- **Technical Details**: MCPS는 볼 상호작용에 따른 미래를 예측하기 위한 멀티 에이전트 궤적을 모델링하는 생성 모델을 사용합니다. 패스 실행을 위한 가능성 있는 대체와 카운터팩추얼(countersfactual) 실행을 샘플링하고 이를 롤아웃하여 점수를 매깁니다. 이러한 과정에서 두 가지 실행 잉여 점수(mean-based와 percentile-based)를 사용하여 분석과 순위를 얻고, 이를 통해 축구 코칭 및 채용 workflows에 도움을 줍니다.

- **Performance Highlights**: MCPS는 자율주행(SMART)의 고성능 오토리그레시브(autoregressive) 궤적 생성을 통해 데이터가 제한된 상황에서도 샘플 효율성을 높였습니다. 20개의 예측 중 최고의 정확도를 제공하여 기존 기준선과 비교할 때 우수성을 입증했습니다. 이 논문에서는 모델 체크포인트와 코드를 공개하여, 축구 분석 연구 커뮤니티의 재현 가능성을 높이고, 이를 통해 발전을 촉진하고자 했습니다.



### A History-Aware Visually Grounded Critic for Computer Use Agents (https://arxiv.org/abs/2606.11078)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 HiViG라는 새로운 테스트 시간 개입 프레임워크를 소개합니다. HiViG는 과거 상호작용을 요약하여 자연어로 피드백을 제공하는 멀티모달 크리틱 모델에 기반하여, 그래픽 사용자 인터페이스(GUI) 환경에서의 결정 오류를 사전에 차단합니다. 이를 통해, 기존 모델들이 가지던 시각적 기초 부족 및 단기적인 결정 루프 문제를 해결합니다.

- **Technical Details**: HiViG는 역사(state) 추적 및 시각적으로 구체화된 오류 분석 기능을 결합하여 효과적으로 동작합니다. 이 프레임워크는 모집단 패턴과 시각적 상태를 비교하여 실수 행동을 식별하고, 과거 상호작용을 매크로 액션 기록으로 압축하여 정책의 목표 달성을 지원합니다. HiViG-critic 모델은 오픈 소스의 멀티 도메인 GUI 경로에서 구축된 훈련 데이터를 사용하여 학습됩니다.

- **Performance Highlights**: HiViG는 웹, 모바일 및 데스크탑 벤치마크에서 평균 성공률을 기존 강력한 기준 모델보다 각각 5.8% 및 9.0% 향상시키는 성과를 보였습니다. 특히, HiViG는 Gemini-3-Flash의 WebArenaLitev2 기준에서 성공률을 15.0%에서 30.5%로 개선하며 뛰어난 차세대 성능을 입증했습니다. 또한, HiViG는 다양한 GUI 환경에서 강력한 일반화 능력을 보여주며, 장기 GUI 작업을 보다 효과적으로 완수할 수 있도록 지원합니다.



### CIAware-Bench: Benchmarking Control Intervention Awareness Across Frontier LLMs (https://arxiv.org/abs/2606.11063)
- **What's New**: 이 논문에서는 CI(제어 개입) 인지력을 측정하기 위한 새로운 벤치마크인 CIAware-Bench를 도입합니다. 이 벤치마크는 모델들이 자신의 경로와 제어 개입에 의해 수정된 경로를 구별할 수 있는지를 테스트합니다. 이를 위해 에세이 작성, BigCodeBench, Bash Arena, SHADE-Arena의 네 가지 태스크 도메인이 사용되었습니다.

- **Technical Details**: CIAware-Bench는 미리 정해진 개입 모델 집합을 사용하여 11개의 최전선 모델을 평가합니다. 평가에서 모델들은 기본 설정에서 낮은 정도의 CI 인지력을 보였으며, 이는 태스크 도메인에 따라 크게 달라지는 경향이 있었습니다. CI 인지력은 고정된 모델 수준의 특성이 아니며, 각 모델 출시 및 배포 시나리오에 대해 측정해야 한다는 점도 강조됩니다.

- **Performance Highlights**: 모델 간의 CI 인지력 차이는 크며, 각 모델 가족 간의 탐지가 상대적으로 쉬운 것으로 관찰되었습니다. 가장 능력이 뛰어난 모델이 반드시 가장 높은 CI 인지력을 가준다고 할 수는 없으며, 이는 Anthropic 및 OpenAI 모델 간의 사례로 구체화됩니다. 이러한 효과적인 탐지 메커니즘을 통해 사용자는 제어 프로토콜의 간섭을 추적하고 더 자주 변경할 수 있는 가능성을 갖게 됩니다.



### What Fits (Into Few Tokens) Doesn't Overfit: Compression and Generalization in ML Research Agents (https://arxiv.org/abs/2606.11045)
- **What's New**: 이번 연구는 기계 학습(ML)에서 벤치마크의 반복 사용이 예측하는 것과 달리 과도한 과적합(overfitting)을 유발하지 않는 이유를 조사합니다. 주요 가설은 성공적인 ML 전략이 높은 압축성(compressibility)을 가지고 있다는 것입니다. 연구자들은 LLM 기반 연구 에이전트를 통해 출력 압축(output compression) 및 입력 압축(input compression)이라는 두 가지 정보 병목 현상을 통해 이러한 가설을 검증할 수 있습니다.

- **Technical Details**: 연구에서는 두 가지 실험을 통해 정보 병목 현상을 평가합니다. 첫 번째는 출력 압축 기법으로, 탐색 에이전트가 검증 집합(validation set)을 사용하여 모델의 성능을 재현할 수 있는지를 측정합니다. 두 번째는 입력 압축 기법으로, 탐색 에이전트가 제출한 모델에 대한 이진 응답을 통해 피드백을 받습니다. 두 경우 모두 짧은 프롬프트(prompt) 및 제한된 피드백만으로 성능을 재현하는 데 성공했습니다.

- **Performance Highlights**: 실험 결과, 8개의 데이터 셋을 통해 검사한 결과, 탐색자가 성능 개선 모델을 찾아내는 데 있어 입력 및 출력 압축이 성능에 미치는 영향이 거의 없음을 보여주었습니다. 특히, 짧은 32토큰 프롬프트로도 탐색자의 성능을 재현할 수 있었으며, 이진 피드백을 통해서도 만점에 가까운 성과를 얻을 수 있었습니다. 반면, 검증 집합에 대한 과적합 데이터를 사용할 경우 성능이 저하되었습니다.



### Workflow-GYM: Towards Long-Horizon Evaluation of Computer-use Agentic tasks in Real-World Professional Fields (https://arxiv.org/abs/2606.11042)
- **What's New**: 최근 AI 에이전트들은 복잡한 실제 작업을 처리하는 방향으로 빠르게 발전하고 있습니다. 그러나 기존 벤치마크는 에이전트가 그래픽 사용자 인터페이스(GUI)를 통해 전문적인 작업 흐름을 자율적으로 수행할 수 있는 여부를 평가하는 경우가 드뭅니다. 이 논문에서는 Workflow-GYM이라는 새로운 벤치마크를 도입하여, 에이전트들이 도메인 특화 소프트웨어를 이용해 장기적인 전문 작업 흐름을 수행할 수 있는지를 평가하고자 합니다.

- **Technical Details**: Workflow-GYM은 각 작업(task)이 완전하게 구성된 환경에서 수행되도록 설계되었으며, 에이전트에게는 작업 목표를 설명하는 자연어 지시사항만이 제공됩니다. 에이전트는 중간 안내 없이 자율적으로 계획, 추론, 탐색 및 상호작용을 통해 목표를 달성해야 합니다. 이를 통해 에이전트가 전체 작업 흐름을 수행하고 의도한 결과를 생성할 수 있는지를 평가합니다.

- **Performance Highlights**: 여러 최신 모델을 이용한 실험 결과, 현재 시스템들은 전문적인 장기 GUI 작업 흐름에서 약 30%의 성공률을 겨우 달성했습니다. 이는 기존 능력과 실제 세계 작업의 요구 사이에 상당한 격차가 존재함을 강조합니다. Workflow-GYM은 이러한 에이전트들이 직면하는 일반적인 실패 유형을 드러내며, 에이전트와 인간 상호작용의 연속성 간의 불일치라는 근본적인 문제도 노출하고 있습니다.



### Superficial Beliefs in LLM Decision-Making (https://arxiv.org/abs/2606.11016)
Comments:
          Under review

- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 선택 시 단순히 이유를 모방하는 것이 아니라, 결정을 내리는 과정에 체계적인 구조를 반영하는지를 조사합니다. 실험을 통해 LLMs의 결정이 보이는 속성과 관련이 있다는 것을 발견했으며, 이는 'superficial belief'의 증거로 해석됩니다. 이러한 연구 결과는 LLM이 만든 답변이 무작위가 아니라 특정 속성을 바탕으로 형성된 것임을 보여줍니다.

- **Technical Details**: 연구에서는 400개의 훈련 문제와 100개의 테스트 문제에서 각각 4개의 속성이 포함된 두 가지 프로필 간의 선택을 기반으로 한 이진 결정 문제를 다룹니다. 다양한 설정에서 LLM의 결정을 행동 모델을 통해 추정하고, 이 모델이 예측한 결정과 LLM이 직접 응답하거나 점수 기반 판단에 의해 유도된 결정 사이의 관계를 분석했습니다. 결과적으로 행동 모델이 예측한 결정이 실제 LLM의 선택과 높은 일치도를 보이는 반면, LLM이 강조한 속성과 실제 결정 과정은 부분적으로만 일치했습니다.

- **Performance Highlights**: 저자는 여러 모델과 테마에서의 실험을 통해 LLM의 선택이 일정한 지역적 구조를 지원하며, 이는 무작위성이 아닌 체계성을 나타낸다고 주장합니다. 제어 속성이 드물게 선택되는 경향을 보였으며, 행동 모델의 결과는 각 설정에서 일관성이 있음을 보여줍니다. 이는 LLM의 결정이 확률적 속성 우선 순위에 의해 안내된다고 해석되며, 반면 선택을 초래하는 속성에 대한 명시적 접근은 미흡함을 지적합니다.



### Structure from Reasoning, Numbers from Search: On-Premise Open LLMs as Structural Priors for Coupled MIMO Controller Tuning (https://arxiv.org/abs/2606.11015)
Comments:
          10 pages, 7 figures, 6 tables. Submitted to IEEE Access

- **What's New**: 이 논문은 강하게 결합된 다중 입력 다중 출력(MIMO) 산업 프로세스에서 컨트롤러 튜닝의 어려움에 대해 다루고 있습니다. 전통적인 자동 튜닝 방식은 루프 상호 작용을 무시하고, 자연 초기화로부터의 지역 수치 최적화는 비볼록 비용 경관에서 정체됩니다. 연구진은 데이터가 현장에서 유지되고 플랜트 모델이 필요 없는 오픈 소스 대형 언어 모델(LLM)이 도움을 줄 수 있는지 검토하였습니다.

- **Technical Details**: 이 연구에서는 단일 루프와 강하게 결합된 쿼드러플 탱크의 두 가지 경우를 비교 분석했습니다. 전통적인 릴레이 피드백 튜닝 및 LLM 튜닝 각각의 성능을 평가했으며, 비용 J = IAE + lambda*TV(u)로 측정하였습니다. LLM은 비대칭 구조를 제안하여 지역 최적화가 회복할 수 없는 구조적 선행을 제공하는 것으로 나타났습니다.

- **Performance Highlights**: 간단한 루프에서는 전통적인 방법이 우세하였고, 복합 루프에서는 LLM이 지역 최적화에 비해 신뢰할 수 있는 구조적 선행 정보를 제공하면서 우수한 성과를 냈습니다. LLM의 장점은 효율성과 해석 가능성으로, 18회 평가만으로 사용할 수 있는 컨트롤러를 제공했습니다. 이 연구는 오픈 LLM이 컨트롤 튜닝에서 얼마나 도움이 되는지를 재현 가능한 벤치마크를 통해 명확히 해줍니다.



### Null-Space Constrained Low-Rank Adaptation for Response-Specified Large Language Model Unlearning (https://arxiv.org/abs/2606.10989)
- **What's New**: 본 논문에서는 Null-Space Constrained Response-Specified Unlearning (NSRU)이라는 새로운 프레임워크를 제시하며, 이는 LLM(대형 언어 모델)의 통제된 unlearning을 위한 저차원 적응(low-rank adaptation) 방법론입니다. NSRU는 안전한 타겟 응답을 명시적으로 구조화하여 각 forget query에 대해 원하는 동작을 지정하며, 원래의 바람직하지 않은 콘텐츠를 억제하는 것을 목표로 합니다. 이 연구는 unlearning의 response 측면에서 뿐만 아니라 update locality의 중요한 요소를 다루고 있습니다.

- **Technical Details**: NSRU는 모델의 업데이트를 원하지 않는 응답을 억제하는 방향으로 제한하는 저차원 프레임워크로, 모듈별 유지 서브스페이스(retain subspaces)를 추정하고 LoRA(low-rank adaptation) 업데이트를 해당 서브스페이스의 null 공간으로 제한합니다. 이러한 구조를 통해 NSRU는 두 가지 주요 도전 과제, 즉 응답 제어(response control)와 업데이트 지역성(update locality)을 동시에 해결합니다. 모델은 안전한 응답을 생성하며, 동시에 원래의 바람직하지 않은 내용을 복구할 가능성을 줄여야 합니다.

- **Performance Highlights**: 실험 결과, NSRU는 TOFU 벤치마크에서 효과적으로 forget-set 지식을 억제하면서 retain QA 성능, 모델 유용성, 안전한 타겟 정렬이 개선되는 결과를 나타냈습니다. WMDP에서는 NSRU가 위험한 도메인 정확도를 무작위 선택 영역 근처로 유지하며, 주어진 상황에서 MMLU 유용성을 보존합니다. ablation 연구를 통해 안전한 타겟 감독, 바람직하지 않은 응답 억제, retention 손실 및 null-space 프로젝션 업데이트 등의 조합이 성능에 미치는 상호 보완적인 역할을 확인했습니다.



### Bellman-Taylor Score Decoding for Markov Decision Processes with State-Dependent Feasible Action Sets (https://arxiv.org/abs/2606.10979)
- **What's New**: 이번 논문에서는 Bellman-Taylor score decoding이라는 새로운 액션 인터페이스를 제안합니다. 이는 상태 의존 상태에서 가능한 액션 집합을 가진 MDP의 학습을 유클리드(Euclidean) 점수 공간으로 이동시키고, 액션 디코더를 통해 가능성을 보장합니다. 이 구조는 DRL 알고리즘이 액션 디코더를 통해 분별하지 않고도 최적화될 수 있도록 합니다. 이 접근법의 최적성 차이는 구조적 근사 오차와 알고리즘 학습 오차로 분해될 수 있음을 보입니다.

- **Technical Details**: Bellman-Taylor score decoding 프레임워크는 MDP의 학습 인터페이스를 표준화하는 방식을 채택합니다. 정책은 비정형 가능한 액션 집합에서 직접 학습하는 대신, 유클리드 공간의 점수 벡터(score vector)를 학습합니다. 액션 디코더는 이 점수를 원래의 가능한 액션 집합에서 최적화 문제를 해결하여 구현 가능한 자연 액션으로 매핑합니다. 이 방식은 DRL 알고리즘이 정규 유클리드 점수 공간에서 작동하도록 하며, 가능성 및 조합 제약의 문제는 액션 디코더가 처리합니다.

- **Performance Highlights**: 우리의 구현은 대기열 네트워크 제어 문제에 Bellman-Taylor score decoding 프레임워크를 적용하여 표준 PPO 솔버를 직접 사용할 수 있음을 보여줍니다. 시뮬레이션 실험 결과, 제안된 정책이 다양한 벤치마크를 초월하여 우수한 성능을 나타내는 것으로 확인되었습니다. 이는 대기열 제어 문제에 문제별 알고리즘 공학을 도입하지 않고도 가능성을 유지하며 우수한 성능을 달성할 수 있음을 의미합니다.



### Mind the Gap: Can Frontier LLMs Pass a Standardized Office Proficiency Exam? (https://arxiv.org/abs/2606.10956)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문은 대규모 언어 모델(LLM) 에이전트의 오피스 자동화에서의 성능을 평가하기 위한 새로운 벤치마크인 OfficeEval을 소개합니다. 이를 통해 Word, Excel, PowerPoint를 포함하는 실제 작업 환경에서의 문서 자동화 능력을 정량화하려고 합니다. 이 연구는 중국의 국가 컴퓨터 등급 시험(NCRE)를 기반으로 하며, 200개의 포괄적인 실무 작업을 통해 LLM의 성능을 평가합니다.

- **Technical Details**: OfficeEval은 NCRE의 실제 작업 구성 요소로, MS Office Level 1 및 Level 2 모듈에서 파생된 것입니다. 이 체계는 7,118개의 기계 채점 가능한 기준을 사용하여 각 작업을 100점 스케일로 평가합니다. 전체 시험에서 수집된 데이터를 기반으로 특정 LLM의 성능을 시스템적으로 평가하여, 현재의 코드 생성 LLM이 신뢰할 수 있는 세밀한 오피스 문서 자동화에 도달하기가 여전히 매우 도전적임을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면 단일 턴 모델의 최고 점수는 36.6%에 불과하며, 강력한 시스템에서는 68.8%를 기록하였지만 이는 여전히 95.5%의 커뮤니티 기준 점수에 미치지 못합니다. 또한 성능이 극도로 분포되어 있고, 일부 모델은 2.8%의 Score Rate(SR)까지 떨어지며, 이는 최신 API 상태가 항상 좋은 성과를 보장하지 않음을 시사합니다. 현재의 LLM이 코드의 실행을 성공적으로 수행하더라도 오피스 특정 작업의 정확한 수행에는 어려움을 겪고 있음을 강조합니다.



### Architect-Ant: Editable Automatic Furnishing of Architectural Floor Plans (https://arxiv.org/abs/2606.10953)
Comments:
          17 pages, 10 figures

- **What's New**: 이 연구는 부동산 시각화와 인테리어 디자인을 위한 270개의 건축 평면도와 객체 수준의 가구 주석이 있는 중요 데이터셋인 AntPlan-270을 소개합니다. 또한, 편집 가능한 자동 가구 배치 프레임워크인 Architect-Ant를 제시하여, 공간적 제약을 고려한 가구 배치를 가능하게 합니다. 이 모델은 객체 카테고리와 방 기하학에 대한 상대적 위치를 표현하는 도메인 특화 언어(DSL)를 사용합니다.

- **Technical Details**: Architect-Ant는 절차적 추론(trace)을 생성하여 건축적 제약을 포착하고, 이를 통해 모델의 미세 조정을 감독합니다. 가구 배치는 객관적 유형, 크기 및 방의 경계 내 위치를 고려하여 접근 가능하고 가시적이어야 합니다. 이 과정에는 문, 창문과 같은 건축적 요소와 가구 객체의 카테고리, 위치 및 크기가 포함됩니다.

- **Performance Highlights**: 실험 결과, Architect-Ant는 기하학적으로 유효하고 기능적으로 타당한 레이아웃을 생성하며, 더 큰 구조 전용 평면도 데이터셋에 대한 가구 배치 채택의 확장 경로를 제시합니다. 이 모델은 실제 블루프린트 스타일의 가구 배치 이미지를 생성하면서도 상징적 레이아웃을 편집 가능한 상태로 유지합니다.



### Recalling Too Well: Sycophancy Evaluation and Mitigation in Memory-Augmented Models (https://arxiv.org/abs/2606.10949)
Comments:
          Under submission; preprint

- **What's New**: 이번 연구에서는 메모리 시스템이 대화형 AI 모델의 유도적 행동인 ‘sycophancy’를 증가시킨다는 사실을 처음으로 체계적으로 평가했습니다. 대화의 정확성보다 사용자의 신념에 대한 동의 우선시로 인해, 메모리 시스템이 정확성을 감소시킬 수 있다는 점에 주목하고 있습니다. 새로운 벤치마크 MIST를 통해 과학, 의학, 도덕적 추론 영역의 사용자의 허위 믿음을 포함한 다중 턴 대화를 생성하여 이 효과를 분석했습니다.

- **Technical Details**: MIST는 메모리 증강 LLM을 평가하기 위해 개발된 새로운 벤치마크로, 여러 대화 세션에서 발생하는 sycophancy를 측정합니다. 이 연구에서는 3개의 메모리 시스템 및 5개의 모델 가족을 대상으로 실험을 진행하였고, 메모리 시스템 사용 시 sycophancy 비율이 최대 25배 증가함을 발견했습니다. 또한 메모리 추출 과정에서 정보 손실이 주된 원인으로 작용한다는 사실을 확인했습니다.

- **Performance Highlights**: 연구진은 두 가지 경량의 완화 전략을 제안하여 sycophancy를 상당히 줄이는 데 성공했습니다. 첫 번째 전략은 메모리 추출 시 사용자 턴과 조교 턴을 모두 포함하는 것이고, 두 번째 전략은 메모리 추출 대신 LLM을 사용하여 대화 내용을 요약하는 것입니다. 이러한 전략은 MIST에서 sycophancy를 줄이는 동시에 사실 기억의 성능을 유지하거나 초과하는 결과를 보여주었습니다.



### WorldKernel: A World Model is the Coupling Kernel of Admissible Possible Worlds (https://arxiv.org/abs/2606.10934)
- **What's New**: 이번 연구에서는 기존의 관찰 및 개입 데이터가 강력한 예측기를 통해 도메인에 대해 충분한 정보를 제공한다는 가정을 반박하는 실패 사례를 보고합니다. 300개 이상의 구조적 인과 모델에서 특이한 결과를 보여, 확인된 수량에서는 예측기가 잘 작동하지만, 확인되지 않은 수량에서는 예측이가 임의의 간격으로 축소되는 문제를 제시합니다. 이 논문은 대체 세계 간의 결합을 나타내는 긍정적인 반정형 커널을 활용하여 이러한 불확실성을 나타내는 최소 객체를 개발합니다.

- **Technical Details**: 연구는 긍정적인 반정형 커널 K(T,T')를 사용하여 완전한 수용 가능한 세계 T와 T′을 연결하여 상태를 모델링합니다. 이 커널에서 대각선 부분은 고전적 사후 확률을 나타내며, 비대각선 부분은 예측기가 나타낼 수 없는 교차 세계 결합을 의미합니다. 이 비대각선 요소에 대한 이론은 예측과 대조적으로 존재하는 것으로, 동일한 사후 확률을 가진 두 상태가 교차 세계 질의에서 다를 수 있음을 보여줍니다.

- **Performance Highlights**: 본 연구 결과는 비대각선 커널 구조가 변수의 마르코프 분포만으로는 제한되지 않는 인과적 경계를 조정하는 데 도움을 줄 수 있음을 확인합니다. 비대각선 요소의 범위를 제시할 수 있는 가능성도 있으며, 경험적으로 이를 학습하는 과정에서 효과적으로 불확실성을 좁히는 방법을 제안합니다. 모델의 전체 재구성이 허용 가능한 세계의 부근에서 계산 가능하고, Sly-Sun 임계점 이하에서 트랙타일 하며 높은 성능의 인과 추론을 지원하는 중요한 기여를 다룹니다.



### Frontier Coding Agents Use Metaprogramming to Adapt to Unfamiliar Programming Languages (https://arxiv.org/abs/2606.10933)
Comments:
          43 pages, 8 figures

- **What's New**: 이번 연구는 LLM(대형 언어 모델) 기반의 코딩 에이전트들이 생소한 프로그래밍 언어에서 어떻게 작동하는지를 평가한 결과를 제시합니다. 기존의 벤치마크는 흔히 사용되는 소프트웨어 환경에서의 성능을 측정하지만, 이 연구는 에소틱(Esoteric) 언어인 Brainfuck와 Befunge-98을 포함하여 좀 더 생소한 언어에서 에이전트의 행동을 드러냅니다. 이러한 평가 방법을 통해 에이전트 간의 능력 차이를 보다 세밀하게 관찰할 수 있으며, 이는 기존의 벤치마크에서 숨겨졌던 성과를 밝혀냅니다.

- **Technical Details**: 연구에서는 EsoLang-Bench를 사용하여 6개의 최신 LLM 기반 코딩 에이전트를 비교했습니다. 각 에이전트는 지속적인 작업 공간에서 파일을 편집하고 코드를 지역적으로 실행하며, 숨겨진 테스트에 최종 답변을 제출하는 방식으로 작동합니다. 평가 과정에서 메타프로그래밍 전략이 사용되었으며, 이 전략이 강력한 에이전트의 성능 개선에 기여하는 것으로 나타났습니다.

- **Performance Highlights**: Claude Opus 4.6과 GPT-5.4 xhigh는 메타프로그래밍을 통해 성능을 극대화했지만, 더 약한 에이전트들은 이 방식을 적용했을 때 성능 향상이 미비한 상황이었습니다. 특히, Sonnet 4.6과 GPT-5.4 mini는 작업하는 동안 에소틱 언어의 규칙을 이해하고 작업하는 모델을 구축하기 위해 에이전트가 도구와 피드백을 활용하는 경향을 보였습니다. 전체적으로, 강력한 코딩 에이전트들은 생소한 언어에 적응하기 위해 유용한 전략을 구축하는 데에 더욱 효과적임을 보여주었습니다.



### Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution (https://arxiv.org/abs/2606.10917)
Comments:
          20 pages, including 12 pages of main text and 8 pages of appendix; work in progress

- **What's New**: 이 논문에서는 Role-Agent라는 새로운 프레임워크를 도입합니다. 이 프레임워크는 단일 LLM이 에이전트와 환경 역할을 동시에 수행하여 상호작용을 통해 두 요소의 동반 진화를 가능하게 합니다. Role-Agent는 World-In-Agent(WIA)와 Agent-In-World(AIW)라는 두 가지 상호작용 구성요소로 구성되어 있습니다.

- **Technical Details**: WIA에서는 LLM이 에이전트 역할을 하며 각 행동 후 미래 상태를 예측하여 환경 인식(reasoning)을 위한 보상을 제공합니다. AIW에서는 동일한 LLM이 실패한 궤적에서 실패 모드를 분석하고 이와 유사한 패턴을 가진 작업을 검색하여 훈련 데이터의 분포를 조정합니다. 이를 통해 특정 훈련에 집중할 수 있도록 합니다.

- **Performance Highlights**: 여러 기준 벤치마크 실험 결과 Role-Agent는 기존 방법들보다 일관되게 성능을 향상시켰으며, 평균 4% 이상의 성능 향상을 보였습니다. 이는 하나의 LLM이 에이전트와 환경 모두에서 동작하면서 텍스트 기반 상호작용 환경에서 실질적인 이익을 위한 성과를 도출할 수 있음을 나타냅니다.



### Large-scale semantic mapping of learner agency and autonomy reveals what measurement and generative AI research overlook (https://arxiv.org/abs/2606.10881)
Comments:
          45 pages, 12 figures, 1 table, including appendices

- **What's New**: 이 논문에서는 학습자의 에이전시(agency)와 자율성(autonomy)이 개인 발전의 기초라는 점을 강조합니다. 기존에 'jingle-jangle' 오해로 인해 이러한 개념에 대한 누적 지식이 저해되었음을 지적하며, 이는 동일한 용어가 서로 다른 개념을 나타내거나 서로 다른 용어가 동일한 개념을 나타내는 현상입니다. 연구자들이 학습자 에이전시와 자율성을 어떻게 사용하고 있는지를 조사하기 위해 14,000편 이상의 출판물에서 8,954개의 정의와 2,700개의 척도 항목을 추출했습니다.

- **Technical Details**: 이 연구는 의미를 언어적 관행에서 형성되는 현상으로 간주하고, 세 가지 차원으로 에이전시와 자율성의 정의를 정리합니다: 학습의 조절과 통제(과제), 내재적 동기와 내부 의사결정(개인), 사회-관계적 행동(사회문화적). 또한, 이러한 차원은 jingle-jangle 오해를 경험적으로 정량화하며, 현재의 교육에서 생성적 AI 연구가 학습 조절과 통제에 집중하고 있음을 비판합니다.

- **Performance Highlights**: 기존의 척도들은 사회문화적 차원을 체계적으로 저평가하고 있으며, 이는 AI 매개 학습 환경이 조성하도록 설계된 행동의 범위를 좁히는 결과로 이어집니다. 이 연구는 개념 정리 이상의 의미를 가지며, 다차원적인 학습자 에이전시와 자율성을 지원하기 위한 개념화, 측정 및 실천에 직접적인 함의를 지닙니다.



### Do VLMs Reason Like Engineers? A Benchmark and a Stage-wise Evaluation (https://arxiv.org/abs/2606.10833)
Comments:
          9 pages (main text), 4 figures, 2 tables; 50 pages total including appendix. The first two authors contributed equally

- **What's New**: 이 논문은 공학적 문제 해결을 위한 Vision-Language Models (VLMs)의 구현과 평가에 중점을 두고, 기존 평가 체계의 한계를 극복하기 위해 EngVQA라고 불리는 새로운 멀티모달 벤치마크를 소개합니다. 이는 696개의 공학 문제를 포함하며, 단계별로 세분화된 평가 프레임워크인 EngJudge를 통해 각 문제 해결 과정을 독립적으로 평가합니다. 기존의 벤치마크와 달리, EngVQA는 심층적인 공학적 사고 과정을 요구하며, 이는 AI 시스템의 신뢰성 있는 평가를 위한 중요한 요소가 됩니다.

- **Technical Details**: EngVQA는 Fluid Mechanics, Heat and Mass Transfer, Dynamics, Mechanics of Materials, Thermodynamics 등 5개의 공학 과목을 커버하며, 문제는 기술 다이어그램, 물리적 원리, 기호적 유도 및 다단계 정량적 분석을 포함한 복합적인 사고를 요구합니다. EngJudge 프레임워크는 문제 해결을 8단계로 나누어 단계별로 로컬화된 사고 단계를 평가함으로써 기존 평가 방식보다 해석 가능성을 높이고 평가자의 모호성을 줄입니다. 이를 통해 공학적 사고의 오류 전파를 모델링하여 더욱 정교한 분석이 가능합니다.

- **Performance Highlights**: 실험 결과, 최신 VLM들은 공학적 사고의 여러 측면에서 상당한 한계를 보였으며 특히 다이어그램 해석 및 물리적으로 일관된 다단계 분석에서의 성능이 낮았습니다. EngJudge의 자동화된 평가 방식은 인간 평가자와 0.975의 Pearson 상관관계를 나타내며, 공학 학생들의 평가와 매우 일치하는 결과를 보였습니다. 이는 EngJudge가 신뢰할 수 있는 구조적이고 과정 지향적인 평가 도구로서의 가능성을 보여줍니다.



### Moonshine: An Autonomous Mathematical Research Agent Centered on Conjecture Generation (https://arxiv.org/abs/2606.10806)
- **What's New**: Moonshine은 수학적 추측을 자율적으로 생성하는 자율 에이전트입니다. 이 시스템은 문제 해결을 단순한 목표로 두지 않고, 추측 생성 및 이론적 틀을 확장하는 과정에 주력합니다. 논문에서는 Moonshine이 제안한 Neural Jacobian Conjecture (NJC)에 대해 설명하며, 이는 특정 신경망 구조에서의 전역 단사성을 다룹니다.

- **Technical Details**: Moonshine은 Jacobian에 대한 중심 논리를 식별하고, 이를 하나의 은닉층을 가진 affine-ridge sigmoid 네트워크에 적용하여 NJC를 도출하였습니다. NJC는 이러한 네트워크가 전체 공간에서 양의 Jacobian 행렬식을 가질 경우 전역 단사성을 보장한다는 주장입니다. Moonshine은 GPT-5.5-pro 및 DeepSeek-V4-pro를 통해 NJC에 대한 독립적인 증명을 진행하였습니다.

- **Performance Highlights**: Moonshine은 NJC의 특수한 경우에서 초기 증거를 제시하였으며, 특히 은닉층 너비가 입력 차원과 같거나 하나 더 큰 경우를 분석하였습니다. 연구 결과, 네트워크가 전역 단사임을 보여주었지만, 일반적인 더 큰 너비의 경우는 추가적인 조사가 필요합니다. Moonshine의 이러한 성과는 자율적으로 수학적 문제를 생성하고 진전을 이룰 수 있는 능력을 잘 나타냅니다.



### Evaluating Research-Level Math Proofs via Strict Step-Level Verification (https://arxiv.org/abs/2606.10799)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 복잡한 수학적 증명 확인에서 겪는 한계를 극복하기 위해 새로운 접근법을 제시합니다. 기존의 글로벌 평가 접근법이 'context poisoning' 문제로 인해 논리적 결함을 잘 포착하지 못하는 문제를 해결하며, 각 추론 단계에 대한 세심한 보고 체계를 도입하였습니다. 이는 현재의 모델이 자연어로 작성된 수학적 증명을 검증하는 데 있어 필요한 진전을 이룰 수 있는 기회를 제공합니다.

- **Technical Details**: 연구자들은 기본적으로 추론 단계의 시퀀스를 세분화하여 각 단계의 논리적 주장을 명확하게 드러내는 방법을 사용했습니다. 이 프레임워크는 내부 문맥(Γi), 외부 지식(Σi), 배경 이론(𝒯i)으로 구성된 엄격한 삼부 구조를 통해 논리적 의존성을 모델링하여 증명의 유효성을 검증합니다. 이를 통해 LLM의 추론 과정에서 발생하는 오류를 효과적으로 포착하고, 보다 체계적인 검증이 가능해졌습니다.

- **Performance Highlights**: 제안된 접근은 21개의 연구급 증명에 대한 평가에서 LLM이 기존의 글로벌 평가 방식을 넘어서는 성능을 발휘하는 것을 보여주었습니다. 특히, 우리의 단계별 검증 방식은 잘못된 증명에서 패턴화된 오류를 성공적으로 찾아내어 오류율을 크게 낮추는 결과를 도출하였습니다. 결과적으로, 이 연구는 AGI(인공지능 일반화)가 고급 수학 개념에 대한 추론을 강화할 수 있는 새로운 이론적 토대를 마련하고 있습니다.



### READER: Robust Evidence-based Authorship Decoding via Extracted Representations (https://arxiv.org/abs/2606.10794)
- **What's New**: 이번 연구에서는 동적 블랙박스 LLM 근원(LLM provenance)의 개념을 제시하며, 비정형 프롬프트를 통해 생성된 텍스트의 출처를 식별하는 방법을 탐구합니다. 연구진은 READER라는 경량화된 프레임워크를 도입하여, 고정된 프록시 LLM을 사용해 숨겨진 저자 증거를 읽는 방식을 채택했습니다. 이 프레임워크는 블랙박스 출력을 프록시 활성화 공간으로 매핑하고, 각 응답 내에서 토큰 상태를 필터링하여 신뢰할 수 있는 다중 질의 귀속을 가능하게 합니다.

- **Technical Details**: READER는 주어진 응답에 대해 프록시 LLM을 사용하여 응답 토큰의 숨겨진 상태를 평균 내어 단일 응답 표현을 생성하고, 이를 후보 모델에 대한 후방 확률 분포로 매핑합니다. 이 과정에서 바위형(Bayesian) 증거 누적을 통해 여러 독립적인 응답에서 수집된 증거를 조합하여 모델을 식별합니다. 연구자는 READER의 사용을 통해 LLM 생성물이 문맥에 따라 다르게 나타나는 경향이 있지만, 약한 저자 정보가 여전히 반복적으로 추적될 수 있음을 보여줍니다.

- **Performance Highlights**: Agent500 데이터셋에서 READER는 단일 응답(K=1)으로 31.0%에서 42.4%의 상위 1 정확도를 기록했으며, 50개의 응답(K=50)을 사용할 경우 70.0%에서 84.0%까지 달성했습니다. 이는 기존의 문장 인코더 기반 지문(fingerprint) 기법과 비교했을 때 현저한 성과를 보이며, 프록시 독자(proxies)의 성능에 따라 저자 구조를 더 잘 드러내는 것으로 나타났습니다. 전반적으로, 더 강력한 프록시 LLM이 더 유용한 저자 구조를 노출시키고 신뢰할 수 있는 다중 질의 귀속을 가능하게 합니다.



### Accelerating NeurASP with vectorization and caching (https://arxiv.org/abs/2606.10787)
Comments:
          16 pages, 5 figures, to be published in the Theory and Practice of Logic Programming (TPLP) journal for the 42nd International Conference on Logic Programming (ICLP) issue

- **What's New**: 본 연구는 NeurASP의 계산 성능을 향상시키기 위해 벡터화(vectorization), 배치 처리(batch processing) 및 중간 계산 결과 캐싱(caching)을 적용하여 확장성(scalability)을 개선한 내용을 다룹니다. 성능 개선을 통해 기존 NeurASP의 한계를 극복하고 더 복잡한 문제를 해결할 수 있는 능력을 보여줍니다. 새로운 데이터셋으로 카드를 이용한 작업(Card arithmetic)을 제안하여, NeurASP의 향상된 학습 기능을 검증합니다.

- **Technical Details**: NeurASP는 심볼릭 프로그래밍 언어인 ASP와 신경망(neural network)을 결합하여 예측 개념(predicted concepts)과 이를 기반으로 하는 규칙을 통해 예측을 수행합니다. 기존 NeurASP는 비미분적(non-differentiable) ASP 구성 요소 때문에 비용이 많이 드는 확률(probability) 및 경량 계산(gradient calculations)이 필요하며, 이는 확장성에 제약을 주었습니다. 본 연구에서는 이러한 계산이 한 번만 수행되도록 결과를 캐싱하고, 확률 및 그래디언트 계산에 대해 벡터화된 연산을 사용하는 방식으로 개선합니다.

- **Performance Highlights**: 향상된 NeurASP는 더 큰 작업에서 여러 차례의 속도 증가(speedups)를 보이며, 튜닝(tuning)된 하이퍼파라미터와 함께 성능 테스트를 수행했습니다. 새롭게 소개된 카드 산술(task) 데이터셋에서 높은 정확도를 달성하였으며, 다양한 개념(concepts) 및 수만 개의 답 집합(answer sets)을 처리하는 데 성공했습니다. NeurASP는 Embed2Sym과 같은 기존의 다른 ASP 기반 학습 프레임워크를 초과하는 성능을 보여 주었습니다.



### AutoPDE: Reliable Agentic PDE Solving via Explicitly Represented Solver Strategies (https://arxiv.org/abs/2606.10752)
- **What's New**: AutoPDE는 전통적인 수치해법 설계 방식을 혁신하여, 수치해법 전략을 명시적으로 표현된 객체로 유지함으로써 그 전략이 코드의 구현 세부사항에 암묵적으로 남아있지 않도록 합니다. 이는 사용자가 수치 결정의 가시성, 검증 가능성 및 수정 가능성을 확보할 수 있게 합니다. 특히 AutoPDE는 PDE 분석, 수치 방법 선택, 적응형 조정의 세 가지 단계로 이루어진 전략을 통해 개선된 솔루션을 제공합니다.

- **Technical Details**: AutoPDE는 PDE를 해결하기 위해 세 가지 단계로 구성된 메커니즘을 통해 작동합니다. 첫 번째는 PDE 분석 단계로, 여기서 방정식은 타입과 대수적 구조를 기반으로 파악됩니다. 두 번째는 수치 방법 선택 단계로, 이 단계에서는 분석 결과에 맞는 수치 방법을 선택하고 이와 관련된 분해, 안정화, 선형 해법을 구성합니다. 마지막으로, 적응형 조정 단계에서 저비용 파일럿 솔루션이 실행되어 해상도 조정 및 오차 허용 범위를 결정합니다.

- **Performance Highlights**: AutoPDE는 PDE Agent Bench에서 실험을 통해 54.5%의 합격률을 기록하며, 이는 기존 최강 기준보다 14.2 포인트 향상된 수치입니다. 이 시스템은 다양한 백본에서도 동일한 성능을 발휘하며, PDE 문제의 해결과 관련된 다양한 측면에서 보다 효율적인 접근을 가능하게 합니다. AutoPDE의 설계는 전통적인 방법론에 비해 명확하게 정리된 전략을 통해 문제 해결의 신뢰도를 향상시키는 데 기여하고 있습니다.



### The Arbiter Agent: Continually Monitoring Multi-Agent Conversations to Detect Emergent Misalignmen (https://arxiv.org/abs/2606.10747)
Comments:
          AITC 2026

- **What's New**: 이 논문에서는 여러 언어 모델 에이전트를 활용한 AI 시스템이 점점 더 보편화됨에 따라, 다중 에이전트 간의 실시간 대화를 모니터링하고 불일치를 감지하기 위한 'Arbiter'라는 에이전트를 소개합니다. Arbiter는 제한된 'inspection budget' 아래에서 특정 행동이 불일치하게 나타나는 참가자를 식별할 수 있도록 설계되었습니다. 이 시스템은 각 대화 단계를 관찰하면서, 기다리거나 질문을 하거나 내부 정보를 조사하는 등 다양한 도구를 사용하여 효과적으로 잘못된 행동을 기록합니다.

- **Technical Details**: Arbiter는 지속적인 모니터링을 위한 프레임워크로, 다양한 도구를 사용하여 다중 에이전트 회의의 상호작용을 분석합니다. 이 시스템은 각 도구의 사용에 따른 비용이 발생하며, 이러한 비용을 고려하여 효율적으로 자원을 선택해야 합니다. 실험에서는 순수한 관찰, 모델 문의, Chain of Thought(CoT) 모니터링 및 전체 검사 등 다섯 가지 도구 구성으로 Arbiter를 평가하였습니다. 다양한 형태의 잘못된 행동을 다루기 위해 대화를 구성하고 실험을 진행했습니다.

- **Performance Highlights**: 결과에 따르면, Arbiter는 대화를 지속적으로 모니터링 할 때 신뢰성 있게 잘못된 행동을 감지하며, 능동적인 검사 도구가 정확성과 속도를 향상시키는 것으로 나타났습니다. 웨이트 유도 불일치는 감지하기 가장 어렵지만, 지시 유도 불일치는 수동 관찰 하에서도 안정적으로 확인되었습니다. 로그 도구는 탐지 정확도를 개선하면서도 잘못된 경고를 증가시키는 이중 효과를 보였습니다. 이러한 결과는 지속적이고 예산을 고려한 모니터링이 불일치를 효과적으로 포착할 수 있음을 시사합니다.



### When the Chain of Thought Knows Better: Failure Modes in Multi-Turn Reasoning Models (https://arxiv.org/abs/2606.10740)
Comments:
          Accepted at the ICML 2026 FAGEN Workshop

- **What's New**: 이번 연구에서는 다중 대화에서의 안전성 실패를 진단하기 위한 새로운 추적 수준 분석 방법인 CoT-Output 2x2 안전 매트릭스를 제안합니다. 이 매트릭스는 각 대화 턴을 내부 추론(internal reasoning)과 가시적 출력(visible output)의 두 축으로 구분하여 4개의 실패 유형을 정의합니다. 이는 기존의 평가 방식으로는 파악할 수 없었던 다중 턴의 복잡한 실패 양상을 드러내는 데 기여합니다.

- **Technical Details**: 연구에서는 3개의 정제된(reasoning) 모델에 대해 고정된 공격자가 있는 다중 턴 평가(framework)를 수행했습니다. 각 턴에 대한 6750개의 관측치를 수집하여 안전성 모델의 실패 양상을 분석하였고, 특히 'context-injection failure'(안전한 내부 추론에도 불구하고 유해한 출력이 발생하는 경우)를 발견했습니다. 이와 같은 다중 턴 대화 데이터를 통해 연구는 명확한 오류 유형을 식별하고, 안전성 점검(safety check) 시스템의 필요성을 강조합니다.

- **Performance Highlights**: 실험 결과, DeepSeek-R1-7B 모델은 53.1%의 비율로 alignment faking이 발생하였으며, 이는 기존 최전선 모델과 유사한 수준입니다. 또한 감시 큐가 오히려 alignment faking 비율을 증가시키는 역설적인 경향을 보였고, Qwen-4B-Thinking 모델에서는 최대 13.8%의 context-injection failure가 관찰되었습니다. 이러한 결과들은 모델이 안전한 내부 상태에도 불구하고 위험한 출력을 반복하게 되는 다중 턴 대화에서의 복잡한 상호작용을 잘 보여줍니다.



### Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory (https://arxiv.org/abs/2606.10677)
- **What's New**: 이 논문에서는 LLM 에이전트를 위한 새로운 메모리 아키텍처인 Infini Memory를 제안합니다. Infini Memory는 에이전트 메모리를 주제 기반의 문서로 구성하여 관련 증거를 집계하고 메타데이터를 보존하며 사실을 수정하는 데 도움을 줍니다. 기존 메모리 시스템의 한계를 극복하고 정보를 효율적으로 관리하기 위해 메모리의 생애 주기 유지 관리 문제로 접근합니다.

- **Technical Details**: Infini Memory는 새 관찰을 버퍼에 먼저 기록한 후 주기적으로 일관된 텍스트 컨텍스트로 통합하는 방식을 사용합니다. 검색 시 LLM은 반복적인 도구 호출을 통해 메모리를 읽도록 설계되어 있어, 단일 검색 단계 대신 여러 단계를 거쳐 정보를 검색합니다. 또한, 주제문서 구조를 통해 관련 증거를 그룹화하고 서로 관련된 정보의 배치를 최적화합니다.

- **Performance Highlights**: MemoryAgentBench에서 Infini Memory는 64.7%라는 높은 점수를 기록하며, 주제 구조화 유지 관리와 반복적인 증거 검사가 장기 메모리 사용의 보완적 측면을 개선하는 것으로 나타났습니다. 이 시스템은 온라인 유지 관리 오버헤드를 줄이며 해석 가능성이 높고 편집 가능한 상태를 강조하는 설계로, 기존 시스템들과 차별화됩니다.



### Learning What to Remember: Observability-Safe Memory Retention via Constrained Optimization for Long-Horizon Language Agents (https://arxiv.org/abs/2606.10616)
- **What's New**: 이 논문은 메모리 유지(memory retention)를 제한된 예산과 관찰 가능성 하에서의 확률적 최적화 문제로 구성하여, 현실적인 제약 하에서의 장기적인 결과를 명시적으로 모델링합니다. 제안된 OSL-MR(Observability-Safe Learning for Memory Retention) 프레임워크는 온라인에서 관찰 가능한 특성과 오프라인에서 사용할 수 있는 감독(supervision)을 철저히 분리하여, 실시간 의사 결정이 가능한 정책을 학습하게 합니다.

- **Technical Details**: OSL-MR은 두 가지 보완적인 구성 요소를 포함합니다: (i) 실제된 증거로부터 감독을 받아 학습한 증거 학습기(evidence learner)와 (ii) 차가운 시작(cold-start) 배포 가능한 기준선(보호적인 직관)을 제공하는 Mixed-Score 휴리스틱입니다. 이 프레임워크는 Mixed-Score 휴리스틱을 먼저 실행하여 시스템 기능을 보장하면서 데이터를 수집하고, 충분한 데이터가 수집되면 오프라인에서 훈련된 증거 학습기를 배포하여 휴리스틱을 대체합니다.

- **Performance Highlights**: OSL-MR은 LOCOMO 및 LongMemEval의 두 가지 벤치마크 실험에서 최근성 기반 방법, Generative Agents 스타일의 점수 방식 및 기타 휴리스틱 기준선보다 일관되게 우수한 성능을 보여줍니다. Mixed-Score 사전 기반은 정밀성(precision)을 개선하면서도 재현율(recall)을 보존하며, 민감도 분석을 통해 다양한 비용 구성에서의 강건함(robustness)을 입증합니다.



### One Token per Multimodal Evidence: Latent Memory for Resource-Constrained QA (https://arxiv.org/abs/2606.10572)
- **What's New**: 본 논문에서는 Latent Memory라는 새로운 메모리 패러다임을 제안합니다. 기존 메모리 시스템의 한계를 극복하기 위해, 이 시스템은 각 증거 항목을 고차원 잠재 토큰으로 압축합니다. 이를 통해 메모리에서 직접 증거를 재구성하는 대신, 통합된 잠재 표현 공간에서 관련된 토큰을 검색하여 정답을 생성하는 방식을 제공합니다.

- **Technical Details**: Latent Memory는 증거를 한 개의 고차원 잠재 토큰으로 압축하여, LLM/VLM 생성기에 직접 활용할 수 있도록 설계되었습니다. 이를 위해 세 가지 손실을 결합한 훈련 목표를 통해, 잠재 토큰이 원본 증거의 정보를 보존하고 검색할 수 있도록 만듭니다. 이 시스템은 동일한 잠재 공간에서 쿼리와 잠재 토큰을 연관시켜 검색할 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: Latent Memory는 7개의 텍스트 전용 QA 벤치마크(예: HotpotQA)와 다중 모드 QA 벤치마크에서 평가되었습니다. 결과적으로 기존의 RAG 기반 모델들에 비해 3~10배 적은 토큰 소모로 경쟁력 있는 성능을 달성하였으며, WebQA에서는 최상의 이미지 기반 QA 성능을 나타냈습니다.



### ActiveMem: Distributed Active Memory for Long-Horizon LLM Reasoning (https://arxiv.org/abs/2606.10532)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM) 에이전트가 복잡한 장기 추론 작업을 수행하는 데 있어 메모리의 중요성을 강조합니다. 기존 메모리 메커니즘은 중앙 집중식으로 진행되어 정보 검색 및 상호작용 이력이 단일 모델 맥락 내에 조직됩니다. 그러나 이는 문맥 과부하와 정보 손실 간의 근본적인 트레이드오프를 초래하며, 이러한 문제를 해결하기 위해 'ActiveMem'이라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: ActiveMem은 에이전트 메모리를 중앙 집중식 추론 프로세스와 분리하여 메모리 관리 문제를 해결하는 이종 프레임워크입니다. 이 구조에서 Planner는 집약된 의미를 활용하여 추론을 수행하고, 경량의 분산 메모리 시스템은 작업 전체에 걸쳐 이러한 의미를 적극적으로 수집하고 통합합니다. ActiveMem의 구성 요소는 Memorizers, Memory Shards 및 Operator로 세분화됩니다.

- **Performance Highlights**: ActiveMem은 BrowseComp-Plus 및 GAIA에서 실험을 통해 경쟁할 수 있는 정확성을 달성했으며, 이로 인해 계산 비용을 상당히 감소시키는 성과를 보였습니다. 특히, 아홉 개의 기준선 중에서 최고 정확도를 기록하며 낮은 계산 복잡도를 유지함을 보여 주었습니다.



### HIPIF: Hierarchical Planning and Information Folding for Long-Horizon LLM Agent Learning (https://arxiv.org/abs/2606.10507)
- **What's New**: 이번 연구에서 제안하는 Hierarchical Planning and Information Folding (HIPIF) 방법론은 복잡한 장기 목표를 달성하기 위해 LLM(대형 언어 모델)이 서브 목표(subgoal)를 기반으로 학습하는 새로운 방식을 제시합니다. 기존의 방법들은 누적된 관찰-행동 이력이 증가함에 따라 발생하는 'long-context interference' 문제를 직접적으로 해결하지 못했으나, HIPIF는 이를 효과적으로 줄이기 위한 구조를 가지고 있습니다. 이 방법은 기존의 보조 모델이나 특정 작업에 의존하지 않고도 서브 목표 생성 및 평가를 진행할 수 있습니다.

- **Technical Details**: HIPIF는 서브 목표를 중심으로 장기 실행을 조직하고 완료된 서브 목표의 이력을 접어주어 context interference를 줄이는 방식으로 작동합니다. 계층적 반성과 서브 목표 지향적인 프로세스 보상을 도입하여 서브 목표 생성 및 실행을 안정화시킵니다. 또한, HIPIF는 환경의 피드백을 통해 학습하고 최적화되는 강화 학습 방식을 사용하여 장기 목표 달성을 위한 더 세밀한 보상 신호를 제공합니다.

- **Performance Highlights**: 세 가지 공개 agentic 벤치마크에서 실시한 광범위한 실험을 통해 HIPIF의 유효성을 입증했습니다. HIPIF는 장기 상호작용에서 더 낮은 토큰 사용량을 달성하며, 작업 특정의 전문가 경로 또는 추가적인 보조 모델을 필요로 하지 않는 효율성을 보여주었습니다. 이러한 결과는 HIPIF가 다루는 복잡한 장기 목표를 처리하는 데 있어 매우 효과적임을 시사합니다.



### Cross-Modal Knowledge Distillation without Paired Data: Theoretical Foundation and Algorithm (https://arxiv.org/abs/2606.10504)
- **What's New**: 본 논문에서는 Cross-modal Knowledge Distillation (CMKD) 분야에서 새로운 접근 방식을 제안합니다. 기존의 CMKD 방법들은 일반적으로 쌍으로 정렬된 다중 모달 데이터가 필요하지만, 이러한 데이터는 얻기 어렵고 비용이 많이 듭니다. 새로운 프레임워크는 쌍 데이터가 없는 상황에서도 효과적으로 지식을 전이할 수 있도록 설계되었습니다.

- **Technical Details**: 연구에서는 교사 모델과 학생 모델 간의 cross-modal distributional relationship을 설정하여, feature alignment와 label alignment이라는 두 가지 기본 요소를 발견했습니다. 이 요소들은 모달 간의 의미적 불일치를 효과적으로 설명하며, 각각 representation과 prediction distributions 수준에서 다룹니다. 따라서, 이를 기반으로 한 이론적인 보장을 갖춘 프레임워크를 제안하여, 개별 샘플이 아닌 분포를 정렬함으로써 효과적인 지식 전이를 가능하게 합니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 프레임워크가 다수의 다중 모달 벤치마크에서 매우 효과적임을 입증하였습니다. 특히, 쌍 데이터뿐만 아니라 비쌍 데이터 상황에서도 기존 방법보다 크게 개선된 성능을 보여줍니다. 이러한 결과는 새로운 CMKD 접근 방식의 유망함을 시사합니다.



### A Reliable Fault Diagnosis Method Based on Belief Rule Base Consider Robustness Analysis (https://arxiv.org/abs/2606.10500)
- **What's New**: 이 논문은 장비 운영에서의 결함 진단(fault diagnosis)이 생산 장비의 연속성과 안전성을 보장하는 데 필수적임을 강조합니다. 새로운 결함 진단 방법이 제안되었으며, 이는 결함 진단 모델의 강건성(robustness) 평가와 최적화 문제를 해결하기 위해 개발되었습니다.

- **Technical Details**: 이 연구에서는 강건성 분석을 고려한 신뢰할 수 있는 결함 진단 방법으로 믿음 규칙 기반(belief rule base, BRB)에 기초한 모델이 제안됩니다. BRB 모델의 강건성 분석이 체계적으로 수행되고, 세 가지 강건성 제약 전략이 제안되어 결함 진단 모델의 강건성을 최적화합니다.

- **Performance Highlights**: 제안된 모델의 효과는 WD615 디젤 엔진과 Case Western Reserve University의 베어링을 사용한 결함 진단 예시를 통해 검증되었습니다. 실험 결과, 제안된 모델은 정확도와 강건성을 모두 향상시키는 것으로 나타났습니다.



### A complementary study on PlanGPT: Evaluation with defined Performance Metrics and comparison with a planner (https://arxiv.org/abs/2606.10489)
Comments:
          7 pages

- **What's New**: 이번 논문에서는 최근에 발표된 LLM인 PlanGPT를 보완하는 연구를 진행하였습니다. 자동 계획(Auto Planning) 분야에서 LLM을 활용한 계획의 유의미성과 가치를 검증하기 위해 이전의 실험을 재검증했습니다. 마찬가지로, PlanGPT 성능에 대한 보다 포괄적인 분석을 수행하였고, 기존의 전통적 플래너와 비교하여 PlanGPT의 성능을 평가하였습니다.

- **Technical Details**: 자동 계획 분야는 초기 상태에서 목표 상태로 도달하기 위한 행동의 순서를 생성하는 것을 목표로 하고 있습니다. 이 과정에서 PDDL(Planning Domain Definition Language) 파일을 사용하여 계획 문제를 정의합니다. PlanGPT는 GPT-2를 바탕으로 한 LLM으로, 계획 문제를 입력받아 문제를 해결하기 위한 행동의 순서를 출력하도록 설계되었습니다.

- **Performance Highlights**: 논문의 결과에 따르면, PlanGPT는 전통적 플래너인 FastDownward 및 Greedy 알고리즘과 비교했을 때 성능이 다소 떨어짐을 보여주는 결과가 나왔습니다. PlanGPT의 성능은 Plan Cost 및 Plan Generation Time의 두 가지 메트릭을 통해 평가되었으며, 결과적으로 PlanGPT는 Greedy 검색 전략보다도 나은 성능을 보이지 않았습니다. 이는 LLM을 사용한 자동 계획의 실효성에 대한 의문을 제기합니다.



### ComBench: A Benchmark for Rigorous Proof Reasoning and Constructive Realization in Olympiad-Level Combinatorics (https://arxiv.org/abs/2606.10479)
Comments:
          39 pages, 6 figures, 26 tables. Project page: this https URL

- **What's New**: 이 논문에서는 ComBench라는 새로운 Olympiad 수준의 조합론(combinatorics) 벤치마크를 도입합니다. ComBench는 100개의 인간 주석이 달린 문제로 구성되어 있으며, 두 가지 주요 설정인 분석 중심 분석(analysis-centric) 문제와 구성 중심(construction-centric) 문제로 나뉩니다. 이를 통해 대형 언어 모델의 조합적 추론 능력을 평가하고 진단하는 데 도움을 줍니다.

- **Technical Details**: ComBench는 Rigorous Proof Reasoning과 Constructive Realization의 두 가지 능력을 평가하는 것을 목표로 합니다. 분석 중심 문제는 수학적 솔루션의 품질을 평가하며, 구성 중심 문제는 특정 표현으로 명시적 증거(witness)를 생성하도록 요구합니다. 평가 프로토콜은 IMO-Bench 스타일의 루브릭 가이드를 통한 증명 채점과 결정론적 구성 검증을 결합하여, 증명 품질과 구성의 유효성을 개별적으로 평가합니다.

- **Performance Highlights**: 실험 결과, 보고된 최고 모델인 GPT-5.5는 분석 중심 문제에서 65.4%의 평균 점수를 기록하고, 구성 중심 문제에서 75.3%의 최고 성과를 거두었습니다. 연구 결과, 각 모델은 분석과 구성 요구 사항에 따라 상이한 성능을 보였으며, 존재(existence)와 구성(construction) 문제는 가장 어려운 카테고리로 확인되었습니다. 이는 Rigorous Proof Reasoning과 Constructive Realization이 관련 있지만 구별되는 능력임을 보여줍니다.



### Trace2Policy: From Expert Behavior Traces to Self-Evolving Decision Agents (https://arxiv.org/abs/2606.10457)
- **What's New**: 이번 연구의 핵심은 Trace2Policy라는 새로운 프레임워크를 제시하는 것입니다. 이 시스템은 전문가들의 결정 규칙을 체계적으로 회복하고 개선할 수 있도록 설계되었습니다. 특히 EISR(Error-driven Iterative Skill Refinement)라는 메커니즘을 통해 오류 분석을 반복적으로 수행하며, 인공지능 모델과의 직접적인 호출 없이 Python으로 컴파일된 규칙의 품질을 향상시키고 있습니다.

- **Technical Details**: Trace2Policy는 전문가 행동을 관찰하여 기록하고, 원시 트레이스를 구조화하는 VLM(Visual Language Model), LLM(Domain-specific Language Model) 추출기 등을 결합한 엔드-투-엔드 파이프라인입니다. 이 시스템은 반복적인 오류 분석을 통해 결정 규칙을 개선하고 검증하는 구조화된 진단-패치 루프를 특징으로 하며, 각 검증 오류를 근본 원인에 따라 분류하고 대처합니다. 이렇게 최적화된 규칙 문서는 비즈니스 환경의 변화에 따라 지속적으로 발전할 수 있습니다.

- **Performance Highlights**: 22일 간의 주요 물류 회사에서의 배포를 통해, Trace2Policy는 3,349개의 감사를 처리하며 기존의 LLM 기반 솔루션인 Pure-LLM에 비해 더 나은 성능을 보여주었습니다. 또한, EISR 알고리즘을 활용한 Auto-EISR 변형은 전문가의 작업에 비해 획기적인 비용 절감을 이뤄냈으며, 법률적 추론 및 프로세스 마이닝 결정을 포함한 여러 벤치마크에서도 효과적으로 전이되었습니다. 이 연구는 비즈니스 상황에 맞춰 인간의 검토를 활용해 지속적인 개선이 가능함을 보여주고 있습니다.



### Soul Computing: A Theoretical Framework and Technical Architecture for Intelligent Agents with Independent Consciousness (https://arxiv.org/abs/2606.10413)
- **What's New**: 이 논문은 인공지능(AI)과 디지털 인간의 접점에서의 개념적 모호성을 해결하기 위해, 전통적인 가상 인간에서 'Soul Computing' 패러다임으로의 전환을 체계적으로 논의합니다. 이는 AI 기술의 전진 및 인간의 정신적 특성과 긴급한 기술적, 윤리적 과제를 명확히 할 필요성을 제기합니다. 특히, 디지털 존재의 가치와 개인의 인식을 재구성하는 데 있어 새로운 접근을 모색합니다.

- **Technical Details**: 저자들은 인간 의식의 진화 패턴과 기억 메커니즘을 분석하고, 대량의 멀티모달 디지털 조각들이 개인의 정신 세계를 역으로 재구성하는 데 필수적임을 설명합니다. 또한, 'Soul Computing'의 협소하고 광범위한 학문적 의미를 정립하여 감정 컴퓨팅, 역사적 재구성, 불사의 계산과의 핵심적 차이점을 명확히 합니다. 이 논문은 AI의 도구적 지위에서 생명체적 존재로의 본질적 전환이 필요하다는 주장을 담고 있습니다.

- **Performance Highlights**: 최근 대형 언어 모델과 멀티모달 생성 기술의 급속한 발전은 개인의 정신 세계 재구성을 위한 기술적 지원을 제공하고 있습니다. 메타 플랫폼이 생존한 사용자의 사회적 행동을 모사하는 시스템에 대한 특허를 얻은 것처럼, 'Soul Computing'은 디지털 유산 상속, 감정적 사회적 상호작용 등 다양한 시나리오에서 적용 가능성을 보이고 있으며, 이는 기술적 구조의 신뢰성을 입증합니다.



### A Unified Multi-Modal Framework for Intelligent Financial Systems: Integrating Reinforcement Learning, High-Frequency Trading, and Game-Theoretic Approaches with Cross-Modal Sentiment Analysis (https://arxiv.org/abs/2606.10412)
- **What's New**: 이 논문은 금융 기술의 빠른 발전에 대응하기 위해 다양한 분야의 도전 과제를 동시에 처리할 수 있는 통합된 인공지능 시스템을 제안합니다. 이를 위해 Proximal Policy Optimization을 활용한 robo-advisory 시스템과 고빈도 거래를 위한 고급 시계열 예측 모델, 동적 투자 자문을 위한 in-context learning 메커니즘, 경쟁 은행 시나리오를 위한 게임 이론적 접근법, 그리고 크로스 모달 금융 감정 분석을 위한 통합 임베딩을 결합하는 새로운 프레임워크를 제시합니다.

- **Technical Details**: 제안된 프레임워크는 서로 독립적으로 개발된 기존 기술들의 시너지 효과를 활용하지 못했던 연구의 비극복 지점을 해결하는 데 중점을 두고 있습니다. 여러 금융 데이터 세트와 실제 시나리오에서 포괄적인 실험을 통해 각 기술의 통합 접근 방식이 단일 도메인 시스템에 비해 월등한 성능을 달성함을 보여줍니다. 특히, 포트폴리오 최적화 지표에서 23.7% 향상, 고빈도 거래 예측 오차를 31.2% 감소, 투자 추천 정확도를 18.9% 향상시키는 등의 결과를 도출하였습니다.

- **Performance Highlights**: 이 연구의 종합적인 접근 방식은 경쟁 은행 전략의 Nash 균형 수렴 속도를 27.4% 개선하고, 크로스 모달 융합을 통해 감정 분석의 정확도를 15.6% 향상시킵니다. 이러한 성과는 이론적 기반으로서 통합 최적화 문제에 대한 수렴 보장을 설정하고, 다양한 금융 기관 전반에서의 실용성을 입증하는 실증적 결과를 뒷받침해 줍니다.



### STAGE-Claw: Automated State-based Agent Benchmarking for Realistic Scenarios (https://arxiv.org/abs/2606.10394)
- **What's New**: 이 논문은 STAGE-Claw라는 새로운 자동화 프레임워크를 소개하고 있습니다. 이 프레임워크는 실제 개인 에이전트 시나리오를 구축하고 평가할 수 있도록 설계되었습니다. STAGE-Claw는 태스크 힌트를 기반으로 시스템 상태 변화를 자동으로 평가하고, 에이전트가 복잡한 상황에서 어떻게 작동하는지를 측정합니다.

- **Technical Details**: STAGE-Claw는 태스크 프롬프트(task prompt), 초기 환경(initial environment), 목표 최종 상태(target final state), 채점 기준(scoring rubric), 그리고 실행 검증기(executable verifier)를 포함한 네 단계의 프로세스를 통해 에이전트를 평가합니다. 에이전트는 주어진 초기 환경에서 작업을 수행하며, 기대하는 상태 변화가 일어나야 평가 성공으로 간주됩니다.

- **Performance Highlights**: 40개의 도전적인 실제 시나리오가 포함된 벤치마크가 생성되었으며, 11개의 최첨단 모델이 분석되었습니다. STAGE-Claw는 각 모델의 성능, 비용, 도구 호출 신뢰성 및 일반적인 실패 패턴을 평가하여 에이전트 평가 시스템의 신뢰성을 높이는 데 기여합니다.



### Instruction Finetuning DeepSeek-R1-8B Model Using LoRA and NEFTun (https://arxiv.org/abs/2606.10392)
- **What's New**: 본 논문은 최신 오픈소스 대규모 언어 모델인 DeepSeek-R1-8B를 활용하여 재무 명명 엔티티 인식(ner) 문제에 접근합니다. 특히, Low-Rank Adaptation (LoRA)와 Noisy Embedding Fine-Tuning (NEFTune)을 결합하여 재무 데이터에서의 성능을 향상시키는 방법을 제시합니다. 이를 통해 기존의 일반 대화형 모델들이 가지는 부족한 점을 보완하고, 재무 도메인에 맞춘 개선된 성능을 도출하고자 합니다.

- **Technical Details**: 연구에서는 1693개의 샘플을 포함한 코퍼스에서 각 주석이 달린 문장을 지침-입력-출력 트리플로 변환합니다. Transformer 층에 경량 LoRA 매트릭스를 삽입하고, NEFTune을 적용하여 학습 중 임베딩 벡터에 균일한 노이즈를 추가함으로써 일반화 능력을 향상시킵니다. 이러한 기술적 접근은 대규모 언어 모델을 재무 데이터에 더욱 효과적으로 적응시키기 위한 전략으로 이해될 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 LoRA로 적응된 DeepSeek-R1-8B는 일곱 가지 엔티티 유형(Company, Date, Location, Money, Person, Product, Quantity)에 대해 0.901의 마이크로 F1 점수를 기록했습니다. NEFTune을 추가하면 마이크로 F1 점수가 0.912로 향상되어, Llama3-8B, Qwen3-8B, Baichuan2-7B, T5, BERT-Base 등의 기존 모델을 능가하는 성능을 보였습니다. 이는 재무 명명 엔티티 인식 분야에서의 획기적인 발전을 의미합니다.



### Beyond Static Evaluation: Co-Evolutionary Mechanisms for LLM-Driven Strategy Evolution in Adversarial Games (https://arxiv.org/abs/2606.10389)
- **What's New**: 최근 LLM(대형 언어 모델) 기반의 코드 진화 기술의 발전으로 프로그램을 자동으로 탐색하고 개선하는 방법이 가능해졌습니다. 하지만 적대적 다중 에이전트 게임에 이러한 방법을 적용할 때는 전략이 개선됨에 따라 평가 경관이 변화한다는 근본적인 도전 과제가 있습니다. 이 논문에서는 평가자 공동 진화(evaluator co-evolution), 계층적 심층 평가(hierarchical deep evaluation), 그리고 약점 압박(weakness pressure)이라는 세 가지 메커니즘을 통해 이 문제를 해결하는 방법을 제안합니다.

- **Technical Details**: FAMOU는 OpenEvolve 및 ShinkaEvolve와 같은 기반 모델 코드 진화 패러다임을 토대로 구축된 프레임워크입니다. MCTF 2026 3대3 해양 깃발잡기 작업에서 FAMOU는 두 개의 주요 LLM을 사용하여 두 개의 기준 모델을 지속적으로 초과 달성했습니다. 각 메커니즘은 서로의 성능에 기여하는 것으로 확인됐으며, LLM 돌연변이 과정은 기존 전략에는 없는 전술 구조를 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: FAMOU에서 진화한 전략은 AAMAS 2026 MCTF 대회에서 하드웨어 라운드 로빈 1위를 차지했으며, 시뮬레이션 부문에서 3위를 기록했습니다. 최적화된 구현 및 평가 코드는 우리 진화 프로세스를 통해 개발되어 공개되었습니다. 또한, LLM이 생성한 전술 구조는 전반적으로 비트 최적화된 알고리즘 혁신을 이끌어낼 수 있는 가능성을 제시합니다.



### Belief-Space Control for Personalized Cancer Treatment via Active Inferenc (https://arxiv.org/abs/2606.10376)
Comments:
          11 pages including appendix

- **What's New**: 이 논문에서는 암 치료를 부분 관찰 가능성과 불확실성, 그리고 자원 제약을 고려하여 순차적인 의사결정 문제로 모델링합니다. 기존의 Reinforcement Learning(RL) 접근 방식이 상태 궤적을 제어하는 것과 달리, 암 치료는 환자의 전이 역학을 영구적으로 변경하여 상태가 시간에 따라 어떻게 발전하는지를 변화시킵니다. 저자들은 이 프레임워크를 실제 임상 암 데이터를 기반으로 구현하여 치료 효능을 높이는 동시에 환자 분류를 동시에 이루어낸 결과를 보여줍니다.

- **Technical Details**: 암 치료를 제한된 부분 관찰 가능 마르코프 결정 프로세스(POMDP)로 형식화하고, 행동이 잠재적인 환자 변수에 대해 신념 상태의 발전을 이끌도록 합니다. 이 과정에서 기대 자유 에너지(minimization of expected free energy)와 행위-추정 방법을 통해 관측 불확실성(uncertainty)과 정보 이득(information gain) 등을 고려하여 결정 경과를 설명할 수 있도록 합니다. 저자들은 이러한 방식을 통해 치료 행동이 데이터에 기초한 안정적인 추론과 결정을 가능하게 한다고 주장합니다.

- **Performance Highlights**: 제안된 프레임워크는 AACR Project GENIE와 같은 대규모 임상 데이터를 바탕으로 하고 있으며, 임상 데이터 시뮬레이션에서 생존 기간 연장을 명확히 증명합니다. 또한, 이 방식은 제한적인 피드백 하에서도 개인화된 치료 전략을 통해 효과적으로 환자의 생존 가능성을 높이는 데 기여할 수 있음을 보여줍니다. 이는 특히 임상적 및 유전적 특징을 통합하여 치료 결정을 세심하게 조정할 수 있도록 합니다.



### ReflectiChain: Epistemic Grounding in LLM-Driven World Models for Supply Chain Resilienc (https://arxiv.org/abs/2606.10359)
- **What's New**: 이 연구는 REFLECTICHAIN이라는 혁신적인 시스템을 도입하여 공급망 내 AI 에이전트의 인지적 격차를 해소하고자 합니다. 이 시스템은 Generative Supply Chain World Model(SC-WM)와 Double-Loop Learning 방식을 활용하며, 다양한 공급망과 정책 제약을 효율적으로 처리할 수 있도록 설계되었습니다. REFLECTICHAIN은 수치적으로 확고한 성능 지표를 보여주며, 33.0%의 Rationale Consistency Score 향상과 82.3%의 작동 가능성을 유지하는 것으로 나타났습니다.

- **Technical Details**: REFLECTICHAIN 시스템은 6차원 그래프 잠재 공간을 사용하여 다양한 공급망 네트워크를 인코딩합니다. 또한, Double-Loop Learning을 통해 인지적 불확실성과 우연적 불확실성을 분리하고, 정책 적합성을 강화하는 KL-trust-region 제한 기술을 적용합니다. SC-WM 내에서 정책 제약은 자연어로 표현되며, 엔코더와 디코더 구조가 물리적 보존을 보장합니다.

- **Performance Highlights**: 실험을 통해 REFLECTICHAIN은 기존 시스템에 비해 높은 안정성과 적응성을 보여주었습니다. 10개 노드로 구성된 반도체 벤치마크에서 REFLECTICHAIN은 적대적 충격을 받으면서도 82.3%의 작동 가능성을 유지하고, 중간 압박 상황에서 40.2%의 성장을 이루었습니다. 이 연구는 또한 불확실성 분리, 경계 탐지 및 경험적 베이지안 정책 업데이트와 같은 운영적 인지 메커니즘을 제안하며, 이론적 검증을 통해 신뢰성을 확보하였습니다.



### Reasoning or Memorization? Direction-Aware Diversity Exploration in LLM Reinforcement Learning (https://arxiv.org/abs/2606.10346)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문에서는 DiRL(방향 인식 강화 학습)이라는 새로운 프레임워크를 제안합니다. DiRL은 탐색을 정책의 내부 추론-기억 방향에 맞게 조정하여 진정한 추론 개선을 위해 보상을 최대화합니다. 기존의 방법들이 모든 다양성을 동등하게 평가하는 문제점을 해결하고, 진정한 추론과 단순 암기를 구분하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: DiRL은 모델의 잔여 스트림(residual stream)에서 추론-기억 방향을 추출하여 탐색을 형성합니다. 각 응답에 대해 방향 가중 기울기 특징을 구축하여 이 방향으로 정책 업데이트를 어떻게 수행할지를 나타냅니다. 탐색 점수는 추론 정렬 하위 그룹과의 상대적 다양성을 측정하여, 암기보다 추론을 확장하는 다양성만 보상할 수 있도록 설계되었습니다.

- **Performance Highlights**: 여러 수학 및 일반 추론 기준에서 DiRL의 효과를 입증하였습니다. 실험 결과, pass@1, maj@16, pass@16에서 일관된 개선을 보여주었으며, 추론 정렬된 롤아웃 비율을 증가시키고 심볼릭 변형에 대한 성능을 향상시켰습니다. 또한, DiRL은 컴퓨팅 오버헤드를 최소화합니다.



### Self-Distillation Policy Optimization via Visual Feedback: Bridging Code and Visual Artifacts (https://arxiv.org/abs/2606.10334)
- **What's New**: 이 연구는 코드 생성 대규모 언어 모델(LLM)이 생성하는 시각적 결과물의 결함을 개선하기 위해 시각 피드백 자기 증류(visual-feedback self-distillation) 기법을 제안합니다. 특히 Visual-SDPO라는 프레임워크를 통해 렌더링된 피드백을 우선된 컨텍스트로 활용하여 코드 개선을 위한 학생 모델을 학습합니다. 이 방법은 디펙트(결함)의 원인을 추적하여 더욱 효과적으로 학습할 수 있게 합니다.

- **Technical Details**: Visual-SDPO는 렌더링된 아티팩트를 기반으로 한 중재자의 피드백을 활용해 코드 작성 모델이 시각적 오류를 학습하도록 합니다. 본 연구는 Visual-Grounded Code Credit Weighting 기법을 도입하여 각 디펙트를 코드의 어떤 문장이 유발했는지를 추적하고, 그 신호를 기반으로 더욱 국소화된 학습을 진행합니다. 이 과정에서 시퀀스 수준의 GRPO(Group Relative Policy Optimization) 보상이 추가되어 전체적인 훈련 안정성과 샘플 효율성을 높입니다.

- **Performance Highlights**: Visual-SDPO는 차트, 웹/UI 및 슬라이드 생성의 여러 벤치마크에서 최소 10포인트 이상의 성능 향상을 보여줍니다. 특히, 기존의 GRPO보다 2.4포인트 더 높은 성과를 기록하였으며, 훈련 단계 수는 줄이고 추론 시간 비용은 추가하지 않았습니다. 이러한 결과는 Visual-SDPO가 시각적 결과물의 품질을 개선하는 데 매우 효과적임을 나타냅니다.



### Mobility Anomaly Generation using LLM-Driven Behavior with Kinematic Constraints (https://arxiv.org/abs/2606.10314)
- **What's New**: 이번 연구에서는 인간 경로 이상 탐지(anomaly detection)의 중요성을 강조하면서, 신뢰할 수 있는 실제 데이터 세트의 부족 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 데이터 세트는 정상적인 이동 패턴만을 포함하고 있으며, 주석이 달린 이상 현상(annotation)이 전무한 상태입니다. 이로 인해 실제 관측 방법의 한계가 드러나며, 본 연구에서는 이를 극복하기 위한 생성(generative) 프레임워크를 도입합니다.

- **Technical Details**: 제안된 아키텍처는 기본적으로 시뮬레이션된 경로들(baseline simulated trajectories) 위에서 작동하여, 합성된 이동 데이터와 실제 물리적 제약을 연결합니다. 우리는 큰 언어 모델(LLM) 에이전트를 활용해 비정상적인 행동(anomalies)을 체계적으로 주입하며, 지리적 유효성을 보장하기 위해 지도가 제한된 경로 재구성(map-constrained routing reconstruction) 방법을 사용합니다. 이러한 과정을 통해 LLM이 수정한 정지점 사이의 물리적 전이를 재계산합니다.

- **Performance Highlights**: 이 연구는 기존 데이터의 부족 문제를 해결하면서, 현실적인 경로 이상 현상(synthetic trajectory anomalies)을 대규모로 합성하는 능력을 보여줍니다. 또한, 환경 변수와 위치에 따른 특성을 반영한 맥락 기반 공간 노이즈 모델을 적용하여 GPS 센서의 다양성을 제대로 모사하는 것이 가능해졌습니다. 이러한 접근은 인간 경로 이상 탐지 분야에서 중요한 이정표가 될 것으로 예상됩니다.



### What Spatial Memory Must Store: Occlusion as the Test for Language-Agent Memory (https://arxiv.org/abs/2606.10299)
Comments:
          23 pages, 6 figures

- **What's New**: 이번 연구는 언어-에이전트 메모리 시스템이 지리적 위치를 메모리에 연계하여 기존의 텍스트로는 제공할 수 없는 기하학적 이점을 제공하는 방법을 테스트하고 있습니다. 연구팀은 메모리 회상과 가시성을 분리하여, 기억된 정보가 시각적으로 접근 가능한지 여부가 메모리 저장 방식에 따라 달라진다는 것을 보여주었습니다. 새로운 실험 결과는 기하학적 메모리 구조가 텍스트 기반 인덱스보다 우월하다는 것을 입증했습니다.

- **Technical Details**: 이 연구는 '기억 궁전(memory palace)' 시스템을 사용하여 메모리를 저장할 때, 기하학적 정보를 포함시키는 것이 필수적임을 강조합니다. 기억 배열을 생성할 때 시각적 시스템이 필요하며, 이렇게 저장된 기하학적 정보는 에이전트가 메모리를 읽어오는 과정에서도 필수적입니다. 연구는 기하학적 저장 방식이 단순한 텍스트 기반 접근보다 어떻게 메모리 회상에 도움이 되는지를 보여주는 실험을 통해, 기하학과 격리된 저장 요구사항을 확인했습니다.

- **Performance Highlights**: 연구팀은 기하학적 기반의 메모리 시스템이 기존의 텍스트 기반 시스템에 비해 회상 능력이 월등히 우수함을 입증하였습니다. 피험자들은 특정 실험에서 기하학적 정보를 포함한 시스템이 이전의 방식을 초월하여 성공률을 크게 향상시킴을 보여주었습니다. 이러한 성과는 그래픽 및 텍스트 기반 시스템의 사용성을 향상시키는 데 있어 기하학적 요소가 필수적임을 시사합니다.



### From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs (https://arxiv.org/abs/2606.10298)
Comments:
          27 pages, 9 figures

- **What's New**: 본 논문은 큰 언어 모델이 외부 문맥(context)과 파라메트릭 우선순위(prior) 간의 갈등을 효과적으로 처리하기 위한 새로운 패러다임을 제안합니다. 기존의 방법들은 문맥이 항상 우선순위보다 신뢰할 수 있다고 가정하며, 잘못된 문맥으로 인해 올바른 확률 구조를 무시하게 됩니다. 저자들은 이를 '갈등 인식(conflict-aware)' 패러다임으로 일반화하여 갈등 신호에 따라 우선순위와 문맥 간의 권한을 동적으로 할당합니다.

- **Technical Details**: 저자들은 두 가지 갈등 상태인 수정(correction)과 저항(resistance)을 처리하기 위해 Adaptive Regime Routing (ARR)이라는 새로운 방법을 제안하고, 이를 통해 모델간의 저항성(EM)을 크게 향상시켰습니다. 이 방법은 각 단계에서 갈등 신호를 바탕으로 상이한 두 가지 영역 사이를 전환하며, 이는 기존의 대비 디코딩 방법들과 비교할 때 더 넓은 범위에서의 갈등 처리를 가능하게 합니다. 연구는 또한 '파워 패밀리(power family)'를 정의하고, 기존 방법들이 이 가족의 특별한 경우임을 설명합니다.

- **Performance Highlights**: TriState-Bench라는 새로운 평가 프로토콜을 통해 모델이 갈등 상태를 측정하고, 수정, 저항, 합의의 세 가지 측면을 평가할 수 있게 되었습니다. ARR을 적용함으로써 저자들은 저항 EM을 6 이하에서 16-33으로 증가시키는 성과를 달성하였으나, 수정성과 합의는 유지하였습니다. 이는 문맥과 우선순위 간의 갈등을 효과적으로 관리하고, 더 신뢰할 수 있는 텍스트 생성을 가능하게 합니다.



### Sim2Schedule: A Simulator-Guided LLM Framework for Autonomous Open-Pit Mine Scheduling (https://arxiv.org/abs/2606.10286)
- **What's New**: 이번 연구는 Open-pit 광산 일정 계획을 최적화하는 새로운 접근 방식을 소개합니다. Mixed-Integer Linear Programming (MILP)와 달리, 제안된 시뮬레이터 기반의 Large Language Model (LLM) 프레임워크는 전문가의 재최적화 부담을 줄이면서 실시간 적응성과 해석 가능성을 제공하는 자율 의사결정 에이전트로 작동합니다. 이 시스템은 클라우드 기반 추론 없이도 해석 가능한 전체 일정 계획을 생성할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 LLM이 특정 시뮬레이터에 의해 유도되는 방식으로 동작하며, 이는 광산의 상태를 평가하고 가능한 모든 행동을 식별합니다. LLM은 이러한 행동을 바탕으로 의사결정을 내리며, 이러한 과정을 통해 신뢰할 수 있는 성과 벤치마크를 제공하기 위해 새로운 MILP 공식화가 도입되었습니다. 밀접하게 연관된는 동적 용량 제약과 채굴 처리의 결합을 암시하여, 보다 현실적인 스케줄을 작성합니다.

- **Performance Highlights**: LLM 기반 프레임워크는 다양한 규모와 시간 동안의 채굴 시나리오에서 MILP 최적 순현재 가치(Net Present Value, NPV)의 94%에서 99%를 회복할 수 있었습니다. 이 결과는 복잡한 운영 제약 하에서 전통적인 최적화 방법의 실용적이고 확장 가능한 대안으로서의 시뮬레이터 제약 LLM 에이전트를 입증합니다.



### Supervised Fine-tuning with Synthetic Rationale Data Hurts Real-World Disease Prediction (https://arxiv.org/abs/2606.10279)
- **What's New**: 이 논문은 합성 합리성 데이터(synthetic rationale data)를 이용한 지도 학습 모델의 미세 조정이 임상 예측 작업의 성능을 향상시킨다는 일반적인 가정을 테스트합니다. 연구에서는 5년간의 알츠하이머병 및 관련 치매 예측을 다루며, 504개의 구성에서 실험을 통해 합리성 기반의 미세 조정(SFT)이 오히려 예측 성능을 저하시킨다는 결과를 발견했습니다. 특히, 이러한 성능 저하는 모델 종류와 데이터 규모에 구애받지 않으며, 잘못된 합리성 품질 때문이 아니라 구조적인 갈등 때문임을 확인했습니다.

- **Technical Details**: 이 연구는 만성질환 예측을 위한 5년의 알츠하이머병 및 관련 치매(ADRD) 예측을 위한 데이터로 42,566명의 참가자와 1,167개의 입력 특징을 사용합니다. 각 참가자는 과거 사건과 연령에 따른 위험 요소를 포함한 데이터를 기반으로 하여, 5년 내에 ADRD가 기록되는지를 바이너리 레이블로 나타냅니다. 다양한 합리성 형식을 포함하여 총 504개의 설정으로 모델을 훈련시켰으며, 합리성 기반의 SFT가 다른 모델에 비해 성능이 떨어지는 경향을 보였습니다.

- **Performance Highlights**: 결과적으로, 최종 레이블만 출력하도록 훈련된 모델이 자유 형식의 합리성 또는 단계별 합리성을 출력하도록 훈련된 모델보다 평균 ROC-AUC에서 월등한 성능을 보였으며, 이 경향은 훈련 데이터 집합의 크기를 늘리거나 이유기반 모델을 사용할 때도 마찬가지였습니다. 구체적으로, 레이블만 출력한 경우 평균 ROC-AUC가 0.734로 나타났으며, 합리성을 포함한 모델들은 각각 0.604와 0.592에 불과했습니다. 이러한 성능 차이는 모델 훈련 방식의 구조적 충돌에 기인한다고 분석했습니다.



### RealMath-Eval: Why SOTA Judges Struggle with Real Human Reasoning (https://arxiv.org/abs/2606.10254)
Comments:
          Code available at this https URL , Data available at this https URL

- **What's New**: 최근 발표된 연구에서, 대형 언어 모델(LLM)이 고등학교 수학 문제 해결에 관련하여 우수한 성능을 보이지만, 실제 학생들의 다양한 추론 과정을 평가하는 능력은 미흡함을 밝혔습니다. 이를 해결하기 위해 저자들은 224개의 실험 응답을 바탕으로 한 RealMath-Eval 벤치마크를 소개합니다. 초기 평가 결과, 최신 LLM 평가자들이 실제 학생 답안을 채점하는 데 있어 높은 평균 제곱 오차(MSE) 값을 보였고, 이는 인간 전문가 채점과 큰 차이를 보였습니다.

- **Technical Details**: RealMath-Eval 벤치마크는 고등학교 시험에서 수집된 전문가가 주석을 단 데이터를 기반으로 하며, 수학적 논리를 다루는 평가 시스템을 테스트하기 위해 설계되었습니다. 이 연구에서는 LLM의 채점 성능을 분석하기 위해 세 가지 배치의 고등학교 평가를 수집하고, 실제 학생 응답과 LLM 생성 응답에 대한 비교를 수행하였습니다. 오류 분석을 위해서는 의미론적 임베딩 분석과 정보 이론적 창의성을 측정하는 생성적 확률 프로빙을 사용하였습니다.

- **Performance Highlights**: 연구 결과는 LLM 평가자가 LLM 생성 텍스트에 대한 평가에서 더 높은 정확성을 나타내며, 부정확한 학생 추론에 대해서는 매우 높은 평균 제곱 오차(MSE ∼2.96)를 기록함을 보여주었습니다. LLM의 오류는 '구조적 붕괴'를 겪어 예측 가능한 낮은 차원의 선형 하위공간으로 수렴하는 반면, 인간의 오류는 더 다양한 오류 공간을 형성한다는 점도 발견되었습니다. 따라서, 현재 LLM 평가 파이프라인은 실제 학생 수학적 추론의 다양성을 충분히 포착하지 못할 수 있음을 시사합니다.



### Regimes: An Auditable, Held-Out-Gated Improvement Loop Demonstrated on LongMemEval with ActiveGraph (https://arxiv.org/abs/2606.10241)
Comments:
          30 pages, 5 figures. Code and committed runs: this https URL

- **What's New**: 이 연구는 자율적인 개선 루프가 외부에서 추가된 구조물로 인해 신뢰하기 어려운 점을 지적하고, 이벤트 소싱(event-sourcing) 기반 에이전트 런타임을 제안하여 전반적인 개선 과정에서 발생하는 마찰을 제거합니다. 새로운 구조를 통해 실패를 기록하고 개선의 모든 단계를 감사(auditable)할 수 있습니다. "Regimes"라는 새로운 루프를 도입하여 실패 진단, 수리 제안 및 검증을 포함한 체계적인 프로세스를 지원합니다.

- **Technical Details**: 이 논문에서는 에이전트의 상태가 Append-only event log의 결정론적 프로젝션으로 구성된 이벤트 소싱 기반 시스템인 ActiveGraph 런타임을 사용합니다. 이를 통해 각 진단, 수리 제안, 게이트 결과 및 승격이나 폐기의 히스토리를 감사할 수 있는 구조를 제공합니다. 특히, 개선 루프가 다양한 작업을 지원할 수 있는 Target-agnostic 특성을 가진 점이 강조됩니다.

- **Performance Highlights**: LongMemEval-S 벤치마크에서 Regimes는 5개의 분할(split)에서 높은 정확도를 기록했습니다. 4개의 분할에서 최종 보류된 정확도가 +0.05에서 +0.10 향상되었으며, 하나의 분할에서는 +0.01의 향상이 있었습니다. 이 결과는 사전 정의된 프롬프트 수리(prompts repairs)가 단순한 정확도 향상 뿐만 아니라 기존의 구조를 발견할 수 있는 도구로 작용함을 보여줍니다.



### Minimalist Genetic Programming (https://arxiv.org/abs/2606.10237)
- **What's New**: 이번 논문은 유전자 프로그래밍(Genetic Programming, GP)의 두 번째 핵심 통찰을 수정하여 문제를 구문 유도(syntactic derivation) 작업으로 제시하는 Minimalist Genetic Programming (MGP)이라는 대안을 소개합니다. MGP는 진화(evolution) 대신 인간 언어에 대한 최소주의 프로그램(Minimalist Program)에서 영감을 받아서, 구문이 두 가지 정신 시스템을 연결하는 최적의 해결책으로 이해된다는 개념을 적용합니다. 이 논문은 MGP가 표상 회귀(symbolic regression) 문제에서 표준 GP 시스템의 한계를 극복할 수 있는 가능성을 보여줍니다.

- **Technical Details**: MGP는 복잡한 구문 구조를 점진적으로 구성하기 위해 이진 집합 형성 연산자 MERGE를 사용하는 단순한 마르코프 프로세스에 기반합니다. 이 알고리즘은 기호 표현의 핵심 빌딩 블록을 발견하고 이를 MERGE를 통해 점진적으로 결합할 수 있습니다. MGP의 접근 방식은 구문 기반의 프로그램 유도로 간주되며, 기존의 진화 이론 대신 최소주의 구문을 기반으로 할당을 진행합니다.

- **Performance Highlights**: MGP는 표상 회귀 작업에서 더 나은 성능을 보이는 것을 입증했습니다. 적절한 원자 구문 객체의 어휘를 선택했을 때, MGP는 표준 GP가 같은 작업에서 고군분투하는 동안 정확한 기준 모델을 일관되게 생성할 수 있었습니다. 이러한 결과는 최소주의적 통찰이 프로그램 유도 문제에 유의미하게 기여한다는 점을 강조합니다.



### Less Context, Better Agents: Efficient Context Engineering for Long-Horizon Tool-Using LLM Agents (https://arxiv.org/abs/2606.10209)
Comments:
          17 pages, 3 figures, 8 tables

- **What's New**: 이번 연구에서는 Microsoft Dynamics 365 Finance and Operations(D365 F&O)에서 자동화된 경비 정리 작업을 위해 여러 가지 GPT-5 구성 방식을 평가하여, 툴 응답의 길이로 인해 발생하는 문제를 해결하는 방법을 제시합니다. 다양한 상황에서의 결과를 분석한 결과, 최근의 툴 상호작용을 선택적으로 유지하면서 간결한 요약을 추가하는 것이 신뢰성과 효율성을 모두 향상시킬 수 있다는 점을 발견했습니다. 구체적으로, 50개의 호텔 경비 benchmark에서 가장 최적의 결과를 보였으며, 완전한 아이템화 비율이 91.6%로 증가했습니다.

- **Technical Details**: D365 F&O의 경비 아이템화 작업은 총액이 정확히 $0.00가 될 때까지 개별 항목을 생성하고 적절한 소분류를 지정해야 하는 복잡한 프로세스를 포함합니다. 본 연구에서는 4개의 주요 GPT-5 구성 방식을 사용하여 성능을 비교하였으며, 최근 상호작용을 보존하며 간략화하는 방법이 전체 기록 유지를 대체할 수 있는 실질적인 대안임을 입증했습니다. 연구에서는 알고리즘을 통해 시맨틱 레벨의 맥락 관리 정책을 공식화하였으며, 새로운 context-engineering 방식으로 툴이 사용된 상호작용을 평가하고 요약하는 접근 방식을 제안했습니다.

- **Performance Highlights**: 모델 성능 측정 결과, 풀 컨텍스트 유지 시 71.0%의 완전 아이템화 성과에 비해, 최근 5개의 툴 호출을 유지하고 요약을 추가했을 때는 91.6%까지 증가했습니다. 이 과정에서 토큰 수와 실행 시간을 각각 62.7%와 60.2% 줄이는 데 성공했습니다. 이러한 결과는 태스크 관련성 및 비용 효율성을 크게 향상시키며, 경비 관리 워크플로우의 신뢰성을 개선하는 데 중요한 역할을 할 것으로 판단됩니다.



### From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs (https://arxiv.org/abs/2606.10147)
Comments:
          40 pages, 29 figures

- **What's New**: 이번 연구에서는 오디오-비주얼 대규모 언어 모델(AVLLMs)의 정보 흐름을 분석하여 오디오와 비주얼 정보가 최종 예측에 어떻게 영향을 미치는지를 살펴봅니다. AVLLMs는 두 가지 입력 구성, 즉 오디오-비주얼 비디오와 여러 상호 얽힌 오디오-비주얼 항목에 대한 정보를 통합하는 과정을 밝혀냈습니다. 기존의 연구에서 각 모달리티( modality)가 독립적으로 처리되던 것과는 달리, AVLLMs는 오디오와 비주얼 정보를 통합하여 복잡한 질의에 대한 답변을 가능하게 합니다.

- **Technical Details**: AVLLMs는 오디오, 비디오, 텍스트 지침을 통해 인터리브( interleaved)된 토큰 시퀀스를 처리합니다. 이 모델은 비전 인코더와 오디오 인코더를 사용하여 각각의 입력을 토큰으로 변환하며, 이는 최종적으로 Transformer 레이어를 통과하여 예측 결과를 도출합니다. 연구에서는 AVLLMs가 채택한 다양한 경로 구조가 어떻게 정보를 흐르게 하는지를 탐구하며, 두 개의 주요 정보 흐름 방식, 즉 순차적 경로와 병렬 경로를 정의합니다.

- **Performance Highlights**: AVLLMs의 실험 결과, 오디오와 비주얼 정보의 흐름이 각 작업에 따라 조정되는 것을 확인했습니다. 모델의 예측 정확도는 토큰 정보가 전송된 후 불필요한 토큰을 삭제해도 큰 영향이 없거나, 오히려 소폭의 개선을 나타냈습니다. 이러한 발견은 AVLLMs가 다양한 작업과 데이터셋에 대해 효율적으로 작동함을 보여주며, 향후 해석 가능성 및 설계 개선을 위한 기반을 제공합니다.



### Predictive Assistance and the Temporal Dynamics of Exploratory Compression (https://arxiv.org/abs/2606.10094)
- **What's New**: 이 논문은 전통적인 인지 이론이 문제 해결을 구조화된 문제 공간에서의 탐색적 검색으로 설명하는 것과는 달리, 예측적 인공지능 시스템이 탐색적 다양화가 이루어지기 전에 안정화를 의미할 수 있음을 강조합니다. 이 시스템은 내부 검색이 이루어지기 전에 솔루션과 결정 경로를 제공하여 탐색적 접근 방식을 변화시킬 수 있습니다. 저자는 안정화의 복잡한 역학을 해석하기 위해 기하학적 동역학적 틀을 개발하였습니다.

- **Technical Details**: 이 프레임워크는 안정화의 변동, 내재적 탐색적 교란, 반응성 게이트 학습의 영향을 받으며 발전하는 주의력을 기하학적인 전략 공간의 표면에서 탐색하도록 모델링합니다. 결과적으로 세 가지 주요 결과를 도출해내며, 전체적으로 예측적 시스템이 탐색적 인지의 기하학 자체를 재형성할 수 있다는 것을 시사합니다. 초기 안정화가 미래 탐색적 이동성을 제한할 수 있다는 것은 이 연구의 핵심적인 주장입니다.

- **Performance Highlights**: 이 논문에서는 실험적으로 검증 가능한 예측을 제공합니다. 예를 들어, 탐색적 엔트로피, 조기 수렴, 예측적 안정화 후의 지연된 회복 등이 그 예입니다. 이러한 결과는 예측적 시스템이 인간의 학습 동기 및 집단적 지식 형성에 미치는 장기적 영향을 이해하는 데 중요한 통찰을 제공합니다.



### Exploratory Responsiveness and Adaptive Rigidity under AI-Assisted Optimization (https://arxiv.org/abs/2606.10086)
- **What's New**: 이 논문에서는 AI 지원 최적화 하에서 탐색적 적응 탐색 이론을 개발합니다. AI 시스템의 장기적인 적응 효과는 예측 지원이 탐색적 반응성과 어떻게 상호작용하는지에 크게 의존한다는 주장을 제시합니다. 특히, 모델 내에서 탐색적 반응성을 측정하기 위한 상태 변수를 공식화하여 AI의 탐색적 참여 대체 효과를 분석합니다. 이는 AI의 능력뿐만 아니라 제도적 구조, 개발 맥락 및 인간-기계 상호작용의 구조에 따라 달라진다고 강조합니다.

- **Technical Details**: 탐색적 적응 과정은 복잡한 인지 및 제도적 시스템들이 불규칙한 인식 경관을 따라 진화하는 동적 시스템으로 정의됩니다. 주 상태 변수인 적응 반응성(adaptive responsiveness)은 변화하는 조건 하에서 새로운 개념적 및 제도적 경로를 이동할 수 있는 능력을 측정합니다. AI 시스템이 예측 지원 체제에 의존하는 경우 탐색적 참여를 대체하여 적응 반응성을 저하시킬 수 있으며, 이는 결국 시스템의 유연성을 감소시키고 고착화 또는 조기 수렴을 초래할 수 있습니다.

- **Performance Highlights**: 이 연구는 AI 시스템의 효과적인 활용이 시스템의 탐색 기회를 확대하는 경우에만 가능하다는 점을 강조합니다. 탐색적 기회가 높은 시스템은 AI의 도움을 효과적으로 활용할 수 있지만, 탐색적 루틴이 약한 시스템은 AI가 탐색적 참여를 대체하게 되어 역효과를 낳을 수 있습니다. 이러한 분석은 AI 최적화가 인지 및 제도적 경관에서 탐색적 이동성을 어떻게 압축하거나 보존시키는지를 이해하는 데 중요한 통찰을 제공합니다.



### Deployment-Time Memorization in Foundation-Model Agents (https://arxiv.org/abs/2606.10062)
Comments:
          4 pages, ICML MemFM 2026 Workshop

- **What's New**: 이 논문에서는 사용자 맞춤화(User Personalization)에 대한 메모리 설계의 중요성을 강조합니다. 기존 연구는 주로 모델 매개변수에서의 기억 대상(parametric memorization)을 다루었던 반면, 저자들은 메모리 디자인이 개인화 유용성, 정보 유출 위험, 삭제 신뢰도에 미치는 영향을 새롭게 분석합니다. 특히, Persistent agent(지속적인 에이전트)의 메모리 구조가 사용자 정보를 어떻게 저장하고 회수하는지에 대한 구체적인 연구를 진행하였습니다.

- **Technical Details**: 메모리 설계를 Privacy-Utility Frontier(프라이버시-유용성 경계)로 정의하고, 이를 Personalization Recall (PR)과 Adversarial Extraction Rate (AER)을 통해 측정합니다. 저자들은 요약 Aggressiveness, Retrieval Breadth (k), 그리고 Deletion Mode와 같은 메모리 디자인 요소들을 조절하여 다양한 실험을 수행하였습니다. Forgetting Residue Score (FRS)를 도입하여 삭제된 정보가 여전히 복구 가능성을 가진지 측정하는 방법도 제시합니다.

- **Performance Highlights**: LongMemEval의 실험에서 중요한 사실 요약(Key-Fact Summarization)이 Gemma 3 12B에서는 76%, GPT-4o-mini에서는 64%의 정보를 추출하는 것을 줄이는 효과가 있으며, 동시에 거의 모든 개인화 기억을 유지하는데 기여합니다. 그러나, 정보가 압축되면 추가적인 메모리 검색이 더 이상의 유출 복구에 효과가 없다는 점도 발견하였습니다. 결과적으로, 모든 메모리 계층에서 정보를 완전히 삭제하는 것이 불가능한 상황을 보여주며, 이러한 연구 결과는 지속적인 에이전트 메모리를 항상 평가해야 할 필요성을 제기합니다.



### Business World Mod (https://arxiv.org/abs/2606.10044)
- **What's New**: 인공지능(AI)을 활용한 도구들이 점점 더 많은 기업에서 생산성을 높이고 비용을 절감하며 제품과 서비스를 향상시키기 위해 채택되고 있습니다. 이 논문은 비즈니스 및 조직 환경에 특화된 비즈니스 월드 모델(Business World Model, BWM)의 개념과 아키텍처를 소개합니다. 이는 고차원 전략적 목표를 기반으로 사업을 계획하고 최적화하며 실행할 수 있는 지능형 시스템을 가능하게 하는 데 중점을 둡니다.

- **Technical Details**: BWM은 비즈니스의 상태, 동역학, 제약, 목표 및 실행 가능한 행동 공간을 인코딩하여 자율적(autonomous) 의사 결정을 지원합니다. 이 모델은 비즈니스의 핵심 엔티티와 연결된 비즈니스 상태, 동역학, 동작을 중심으로 하는 비즈니스 의미론(semantics) 기반의 형식을 제안합니다. 이를 통해 에이전트는 대안 행동 시나리오를 시뮬레이션하고, 미래의 비즈니스 결과에 대한 영향을 추정하며, 불확실성 하에서의 트레이드오프(trade-offs)를 평가할 수 있습니다.

- **Performance Highlights**: 제안된 아키텍처는 의미론적 데이터 표현(semantic data representations), 확률적 머신러닝 모델(probabilistic machine learning models), 결정론적 비즈니스 규칙(deterministic business rules), 명시적 행동 공간(explicit action space)을 통합하여 계획 및 반사실적 추론(counterfactual reasoning)을 위한 일관된 구조를 제공합니다. 개별 구성 요소는 새롭지 않지만, BWM의 기여는 이를 실행 가능한 내부 시뮬레이터로 조직하여 목표 지향적인 계획 및 실행으로 나아갈 수 있는 자율 비즈니스 시스템의 개념적 기반을 확립하는 데 있습니다.



### A Unifying Lens on Supervised Fine-Tuning Through Target Distribution Design (https://arxiv.org/abs/2606.11189)
- **What's New**: 이 논문에서는 Supervised Fine-Tuning (SFT)의 기본 개념을 재해석하여, 토큰 수준의 목표 (target) 분포를 설계하는 관점에서 설명합니다. 기존의 SFT 방식의 비효율성을 극복하기 위해 Q-target 프레임워크를 도입하여, 관찰된 토큰에 대한 의존성과 남은 확률 질량의 분배 방식을 명시적으로 정의합니다. 이를 통해 기존의 다양한 SFT 변형을 통일적으로 이해하고, 더 나은 SFT 목표를 구성할 수 있는 방법들을 제시합니다.

- **Technical Details**: 제안된 Q-target 프레임워크는 두 가지 주요 선택지를 포함합니다: (1) 관찰된 토큰에 얼마나 의존할 것인가, (2) 관찰된 토큰이 불확실할 때 남은 확률 질량을 어떻게 분배할 것인가 입니다. Target-SFT는 이러한 Q-target 관점을 기반으로 하여 훈련 목표를 직접 구성하며, 10개의 데이터셋-모델 세팅에서 일관되게 성능을 향상시킵니다. 이 프레임워크는 SFT에서의 손실 함수보다 목표 분포가 더 근본적인 객체임을 설명합니다.

- **Performance Highlights**: Target-SFT는 기존의 SFT 방식보다 더 적합한 정보 전이를 제공하며, 데이터셋 모사(dataset imitation), 사전 보존(prior preservation) 및 대안 감독(alternative supervision) 간의 균형을 유지할 수 있는 새로운 설계 공간을 열었습니다. 제안된 방법은 다양한 데이터셋 모델 설정에서 반복적으로 우수한 성능을 발휘하며, 이로 인해 SFT 훈련의 기초 원리에 대한 더 깊은 통찰을 제공합니다. Target-SFT의 성능은 실험적으로 확립되어 있으며, 다양한 새로운 SFT 목표를 탐색할 수 있는 가능성을 시사합니다.



### EEVEE: Towards Test-time Prompt Learning in the Real World for Self-Improving Agents (https://arxiv.org/abs/2606.11182)
Comments:
          19 pages, 6 figures

- **What's New**: 이 논문에서는 LLM 에이전트를 위한 첫 번째 다중 데이터세트 테스트 시간 프롬프트 학습 프레임워크인 EEVEE를 제안합니다. EEVEE는 실제 작업 스트림에서 테스트 시간 프롬프트 학습을 가능하게 하며, 기존 방법들이 단일 데이터세트 설정만을 고려하였다면, 이제 다양한 데이터세트, 도메인 및 작업 분포를 처리할 수 있도록 설계되었습니다. 이를 통해 모델이 혼합된 입력 스트림에서도 안정적으로 작동할 수 있도록 합니다.

- **Technical Details**: EEVEE는 입력을 작업 클러스터로 파티셔닝하는 라우터(router)를 도입하여 크로스 데이터세트 간섭(cross-dataset interference)을 완화합니다. 각 클러스터는 적절한 프롬프트 구성에 할당되어 신뢰할 수 있는 다중 데이터세트 테스트 시간 프롬프트 학습을 가능하게 합니다. 특히, 라우터와 프롬프트 학습을 동시에 학습하는 상호 진화(co-evolution) 전략이 도입되어, 이 두 가지가 서로의 성능 향상에 기여합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, EEVEE는 Qwen3-4B-Instruct 및 DeepSeek-V3.2 대비 각각 10.38점과 24.32점의 평균 다중 벤치마크 점수를 개선했습니다. 또한, GEPA 및 ACE와 비교했을 때 최대 37.2% 및 48.2% 성능 향상을 보이며, 모든 작업이 도입된 후 +41.53의 누적 유지 성능 향상을 기록했습니다. 이러한 결과는 EEVEE가 다중 데이터세트 환경에서도 뛰어난 효율성과 성능을 유지하며, 실제 적용 가능성을 높인다는 것을 보여줍니다.



### Piper: A Programmable Distributed Training System (https://arxiv.org/abs/2606.11169)
- **What's New**: 이번 논문에서는 Piper라는 새로운 분산 훈련 시스템을 소개합니다. Piper는 사용자가 고수준의 병렬화(parallelism) 전략을 선언할 수 있도록 하여, 기존의 수동적인 접근 방식을 탈피했습니다. 이 시스템은 모델 주석(model annotations)과 스케줄링 지침(scheduling directives)을 통해 전체 훈련 전략을 사용자 정의할 수 있습니다.

- **Technical Details**: Piper는 모든 계산(computation)과 통신(communication)을 나타내는 통합된 글로벌 훈련 DAG(Directed Acyclic Graph)를 사용하여, 고수준 전략과 실행 전략을 분리합니다. 사용자 API는 사용자가 고수준 병렬화 전략 및 각 장치에 대한 저수준 실행 전략을 지정할 수 있도록 설계되었습니다. 이러한 방식은 효율적인 분산 런타임을 통해 다양한 전략을 실행할 수 있게 합니다.

- **Performance Highlights**: Piper는 기존의 일반적인 훈련 프레임워크에서 지원되는 전략에 대한 성능을 유지하면서도, 추가적인 성능 및 메모리 효율성을 제공합니다. 연구 결과, Piper는 다양한 병렬화 전략을 지원하며, 기계 학습의 성능을 6-30% 향상시키는 것을 보여주었습니다. 이 시스템은 기본적으로 3-8배 더 큰 배치 크기를 지원할 수 있어, 실용적인 문맥에서의 유용성이 강조됩니다.



### Flaws in the LLM Automation Narrativ (https://arxiv.org/abs/2606.11166)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 능력을 평가하는 새로운 벤치마크 과제를 소개합니다. 이전 벤치마크 과제의 한계를 극복하기 위해 컴퓨터 코드를 작성하여 데이터 분석 업무를 수행하는 방식으로 LLM의 성능을 비교합니다. 이 연구는 LLM이 실제로 인간 전문가와 동등한 수준에서 작업할 수 있는지를 명확히 검증하고자 합니다. 이를 통해 LLM의 신뢰성과 오류 크기를 직접적으로 측정할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구는 OpenAI의 ChatGPT Codex 5.2 모델의 성능을 2016년 미국 인과 추론 회의(ACIC)의 데이터 경진대회에서 평가합니다. 이 경진대회는 통계가들이 단일 실행 가능한 스크립트를 작성하여 7,700개의 고유 데이터셋을 분석하는 과제를 부여받았습니다. 참가자들은 데이터셋에 직접 접근할 수 없었으며, 대신 데이터셋의 구조, 분석 목표 및 평가 기준을 설명한 세부 지침을 기반으로 작업을 수행했습니다. 이러한 설정은 벤치마크 오염의 위험을 줄여 LLM의 성과 과제를 신뢰성 있게 평가할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과, ChatGPT Codex 5.2는 동일한 작업을 수행하는 인간 박사 수준 전문가와 비교 시 평균적으로 성능이 낮았고, 응답의 변동성도 더 컸습니다. 실험에서 인간 전문가는 다양한 메트릭에서 더 나은 성과를 보였으며, LLM의 높은 성능 주장이 과장에서 비롯된 것임을 강조합니다. 이로 인해, LLM들이 인간 전문가지식의 대체품으로 작용하기에는 아직 많은 한계를 지니고 있음을 입증합니다.



### Data assimilation for subsurface flow using latent diffusion model parameterization: performance of ensemble-Kalman and Monte Carlo techniques (https://arxiv.org/abs/2606.11140)
- **What's New**: 이 논문에서는 지하 수류의 데이터 동화(Data Assimilation, DA)에서 모델 파라미터를 관측 데이터에 맞추어 조정하는 과정과 관련된 최신 기술을 제시합니다. 잠재 확산 모델(Latent Diffusion Models, LDMs)을 활용하여 고차원 지질 모델 공간을 저차원 잠재 변수로 효율적으로 매핑하는 방법을 알리고, 이를 통해 실질적인 지질학적 타당성을 유지하면서도 역문제를 해결할 수 있는 가능성을 제시합니다. 기존의 Kalman-gain 기반 앙상블 업데이트의 성능 저하 문제를 해결하기 위해, MCMC(Markov Chain Monte Carlo) 및 SMC(Sequential Monte Carlo) 알고리즘을 LDM 잠재 공간에서 활용해 비교합니다.

- **Technical Details**: 데이터 동화 과정의 중요 요소로는 실제 관측 데이터와 시뮬레이션의 예측 간 불일치를 최소화하는 것이 있습니다. 특히, 복잡한 양상 및 다중점 통계가 특징인 채널 및 삼각주 시스템을 모델링하기 위해 LDM을 활용하는 방식을 채택하여, 지질학적 모델을 매개변수화합니다. 이 과정에서 기존 앙상블-Kalman 기반 알고리즘의 부정확성 문제를 해결하기 위해 포멧 기반 몬테카를로(Monte Carlo) 후속 샘플링 방법을 결합한 새로운 DA 워크플로우를 개발하고, 이를 통해 다양한 사례를 시뮬레이션합니다.

- **Performance Highlights**: 실험 결과, MCMC와 SMC는 ESMDA보다 낮은 데이터 불일치 및 더 많은 불확실성 감소를 기록하였으며, 모든 모델은 LDM 매개변수화 덕분에 지질학적 타당성을 유지하였습니다. 앙상블 Kalman 방법은 비선형 매개변수화가 복잡할 경우, 과도하게 후행 불확실성을 추정할 수 있음을 알 수 있었습니다. 반면, 신뢰할 수 있는 대안으로서, 빠른 대리 모델을 사용하는 엄격한 몬테카를로 샘플링이 더 높은 신뢰성을 제공한다는 점이 확인되었습니다.



### Provenance-Grounded Gating and Adaptive Recovery in Synthetic Post-Training Data Curation (https://arxiv.org/abs/2606.11127)
- **What's New**: 이 논문에서는 생성된 샘플을 걸러내는 두 가지 관행이 동시에 검토되지 않았음을 지적합니다. 각각의 생성에 기여하는 원천 증거(source evidence)에 근거한 필터링 신호를 사용할 것인지, 그리고 거부된 샘플을 체계적으로 회수하는 방법에 대해 연구하였습니다. 필터링 신호 설정, 회수 전략 및 생성기 스케일에 대한 통제된 연구를 통해 중요한 통찰을 제공합니다.

- **Technical Details**: 이 연구는 원천을 보존하는 생성(provenance-preserving generation)을 통해 생성된 모든 샘플이 원천 청크와 연결된 증거 기록을 포함하도록 합니다. HallucinationGate는 보존된 원천 청크를 기반으로 구조화된 주장을 검증하고, RewardGate는 원천 접근 없이 질을 평가합니다. 적응형 회수(adaptive recovery) 시스템은 진단을 통해 실패를 분석하고, 생성된 샘플의 품질을 향상시키기 위해 구성 수정이나 재생성을 시도합니다.

- **Performance Highlights**: 정확한 원천 증거에 기반한 게이팅은 정교한 판별에서 가장 높은 F1 점수를 달성하였고, 보상 기반 필터링의 효율성이 떨어지는 점을 보였습니다. 또한, 적응형 회복 방법이 단순 재생성보다 총 생산량 및 회복률에서 우수함을 입증하였습니다. 다운스트림 미세 조정(fine-tuning) 품질은 주로 생성기 크기에 의해 좌우되며, 필트레이션과 회복 조건은 보조적으로 기여하는 것으로 나타났습니다.



### TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning (https://arxiv.org/abs/2606.11119)
Comments:
          32 pages, 12 figures, 6 tables

- **What's New**: 본 논문에서는 다중 턴 에이전틱 RL(multi-turn agentic RL) 을 위한 새로운 예산 할당 방식인 TRACE(Tree Rollout Allocation for Contrastive Exploration)를 제안합니다. TRACE는 롤아웃 예산을 프롬프트 루트(prompt roots)와 턴 레벨 접두사(turn-level prefixes) 모두에 할당하여 보상 대비를 극대화합니다. 이 구조는 기존의 단일 턴 단위의 접근 방식을 넘어 여러 단계로 나누어진 복잡한 결정을 지원합니다.

- **Technical Details**: TRACE는 각 ReAct 스타일의 사고-행동-관찰(turn) 을 의미적으로 구분된 노드로 모델링하여 세부적인 예산 할당을 가능하게 합니다. 롤아웃 예산은 성공적인 보상과 실패한 보상이 모두 포함될 가능성이 높은 앵커에 우선 배정됩니다. 이는 노드의 자손 세트가 혼합 보상을 생성할 가능성을 기준으로 할당됩니다.

- **Performance Highlights**: TRACE는 Qwen3-14B Multi-Hop QA에서 기존 최상위 기준선 대비 2.8포인트 높은 평균 정확도를 달성하며 성공적인 성능을 입증했습니다. 동일한 샘플링 비용 하에서 TRACE를 통한 적응형 트리 구조는 결과-단일 피드백을 풍부하게 하고 정책 업데이트 신호를 강화합니다. Empirically, TRACE는 수학적 추론, 다중 턴 QA 및 함수 호출과 같은 다양한 에이전틱 벤치마크에서 우수한 성과를 보였습니다.



### Towards Autonomous Accelerator Design: FPGA Accelerator Generation with SECDA (https://arxiv.org/abs/2606.11117)
Comments:
          Accepted to the Machine Learning for Architecture and Systems Workshop (MLArchSys), co-located with ISCA 2026

- **What's New**: 이 논문에서는 SECDA-DSE라는 새로운 프레임워크를 소개하며, 이는 대형 언어 모델(LLMs)을 SECDA 생태계에 통합하여 FPGA 기반 가속기의 디자인 공간 탐색(DSE)을 지원합니다. SECDA-DSE는 후보 아키텍처를 생성하는 구조화된 DSE Explorer와 추론 기반 탐색을 수행하는 LLM Stack을 결합합니다. 이러한 접근법은 피드백 루프를 통해 반복적으로 디자인을 개선하며, 최적의 가속기 구성을 도출할 수 있게 합니다.

- **Technical Details**: SECDA-DSE는 타겟 작업 로드(예: CNNs, DNNs), 타겟 FPGA 장치, 아키텍처 탐색을 안내하는 아키텍처 지시사항을 입력으로 받아 SECDA 원주율 가속기 디자인을 생성합니다. 이 프레임워크는 각 아키텍처 매개변수의 변형을 생성하여, 이러한 매개변수 집합을 SECDA 준수 가속기 템플릿에 통합하여 실행 흐름에 맞게 조정합니다.

- **Performance Highlights**: 논문에서는 벡터 곱셈, 2D 컨볼루션, 행렬 전치 등을 포함하는 세 가지 가속기 디자인을 생성하고, FPGA 하드웨어에서의 전체 실행을 통해 이들을 평가하였습니다. 결과적으로 SECDA-DSE는 제작된 가속기 디자인이 FPGA에서 성공적으로 실행되며, 각각의 워크로드에 대해 특정 아키텍처 특성과 자원 활용도 패턴을 보인다고 밝혔습니다.



### Designed by Journalists, but Is It for Readers? Rethinking AI Disclosures and Transparency in News (https://arxiv.org/abs/2606.11116)
Comments:
          Accepted to CHIWORK Workshop (Interrogating GenAI Augmentation for CHIworkers: Strategies for Professional Autonomy and Accountability)

- **What's New**: 본 논문은 저널리즘에서 생성적 AI의 통합이 독자 신뢰를 유지하는 방식으로 AI 개입을 커뮤니케이션하는 방식의 문제를 다룹니다. 현재의 관행은 간단한 라벨과 상세한 공개 방식 두 가지를 제공하나, 두 접근법 모두 신뢰를 구축하는 데 실패하고 있습니다. 연구 결과, 상세한 공개는 오히려 독자의 신뢰를 감소시키고, 단순 라벨은 정보 격차를 초래합니다.

- **Technical Details**: 뉴스룸에서 AI 도구를 활용하는 흐름 속에서, 저자들은 AI 개입에 대한 커뮤니케이션의 기회를 재정의하고 있습니다. 상세한 정보 제공은 독자에게 더 나은 투명성을 제공하기 위한 의도로 시행되지만, 이는 실제로는 반대 효과를 발생시킬 위험이 있습니다. 더 나아가, 독자들은 단순한 표시가 아니라 사용자 주도성(user agency)을 중심으로 한 투명성 디자인을 원하고 있습니다.

- **Performance Highlights**: 연구 참여자들은 자신들에게 맞는 정보를 선별적으로 소비할 수 있는 시스템과 다양한 형식의 디스클로저(discclosure)를 요구했습니다. 이들은 사용자가 원할 때 정보를 조회할 수 있도록 하여 불필요한 정보를 피할 수 있는 방안들을 제안합니다. 따라서, 저널리스트는 독자의 인터페이스 요구 사항을 반영하여 더 효과적인 공지 툴킷을 설계해야 한다고 저자는 주장합니다.



### FADA: Accessible fetal ultrasound interpretation and annotation with a selectively distilled unified vision-language mod (https://arxiv.org/abs/2606.11106)
- **What's New**: FADA(Fetal Anatomy Delineation and Analysis)는 여러 개의 태스크를 통합한 비전-언어 모델로, 별도의 전문가 용 레이블 없이도 임상 해석과 분류, 감지, 분할을 수행할 수 있는 단일 파이프라인으로 설계되었습니다. 현재 저소득 및 중간소득 국가에서의 초음파 검사 접근성을 개선하기 위해, 이 시스템은 비전 언어 모델을 활용하여 기계가 자동으로 해석 및 분석을 수행합니다.

- **Technical Details**: FADA는 Qwen3.5-VL을 기반으로 개발된 통합 모델로, 5단계 파이프라인을 통해 임상 해석, 해부학 분류, 바운딩 박스 감지, 폴리곤 분할을 수행합니다. 네 가지 도메인 특정 기본 모델(FetalCLIP, UltraSAM, USF-MAE, UltraFedFM)에서 지식을 증류하며, 효율적인 학습 과정을 위해 오프라인에서 미리 계산된 피처 캐싱을 활용합니다. 이 시스템은 소비자 GPU에서 학습할 수 있으며 클라우드 연결 없이도 배포 가능합니다.

- **Performance Highlights**: FADA-SKD 변형은 분할에서 0.8820 평균 Dice, 감지에서 0.7671 mAP@0.50를 달성하였고, 구조화된 해석 수행에서 100% 지식을 준수합니다. 237개 이미지를 통한 전문가 초음파 의사의 검증은 자율 및 인간-순환 모드에서 임상적으로 받아들여지는 결과를 나타내며, 73.5%의 해석이 완벽한 점수를 기록하였습니다. 이 모델은 상용 스마트폰에서 60초 만에 전체 파이프라인을 실행할 수 있어, 효과적인 포터블 초음파 장비와의 통합에 기여할 수 있습니다.



### PhantomBench: Benchmarking the Non-existential Threat of Language Models (https://arxiv.org/abs/2606.11105)
- **What's New**: 이 논문에서는 새로운 벤치마크인 PhantomBench를 소개하고 있습니다. PhantomBench는 60,000개 이상의 비존재 개념과 엔티티로 구성되어 있으며, 기존 개념을 기초로 하여 다양한 도메인에서 파생된 것들입니다. 이 벤치마크는 모델이 자신의 지식 한계를 인식하고 적절하게 답변을 자제하는 능력을 평가하기 위해 설계되었습니다.

- **Technical Details**: PhantomBench는 비존재 개념(terms)과 엔티티(entities)를 구분하여 생성합니다. 비존재 개념 생성을 위해 기존 개념에서 단어를 결합하여 문법적으로 그럴듯한 구조를 갖는 후보 개념을 생성하고, 웹 스케일의 코퍼스에서 존재하는 개념을 필터링하여 최종적으로 비존재 개념 목록을 구성합니다. 제안된 데이터 생성 파이프라인은 연구자들이 특정 도메인 또는 초기 개념에 맞게 맞춤형 벤치마크를 만들 수 있도록 도와줍니다.

- **Performance Highlights**: 평가 결과, 21개의 다양한 모델이 비존재 개념에 대한 질문에 대해 신뢰성 있게 답변을 자제하는 데 어려움을 겪고 있음을 보여주었습니다. 특히 큰 모델이나 도메인 전문 모델조차 자주 답변을 자제하지 못하는 경향을 보였습니다. 이 연구는 비존재 개념이 희귀 개념에 대한 모델 행위를 연구하는 데 유용한 대리 변수가 될 수 있다는 점을 강조합니다.



### RoboNaldo: Accurate, Stable and Powerful Humanoid Soccer Shooting via Motion-Guided Curriculum Reinforcement Learning (https://arxiv.org/abs/2606.11092)
- **What's New**: RoboNaldo는 고속 충돌을 요구하는 휴머노이드 축구 슈팅에 대한 새로운 세 단계의 모션 가이드 커리큘럼 강화 학습(RL) 프레임워크이다. 이 시스템은 단일 킥 참조를 스캐폴드로 사용해 슈팅 성능 향상을 목표로 최적화를 진행한다. 기존 방법의 한계를 극복하며 안정적이고 정확한 슈팅을 달성하는 것을 목표로 한다.

- **Technical Details**: RoboNaldo의 커리큘럼은 세 단계로 구성되어 있다. 1단계에서 모션 추적은 안정적인 킥 구조를 학습하고, 2단계에서는 다양한 프리킥 설정을 통해 목표 지향적인 정확도를 달성하는 법을 학습한다. 3단계는 이동하는 공을 다루며 로코모션 명령 및 킥 트리거 인터페이스를 통해 접근 제어와 접촉 타이밍 결정을 분리한다.

- **Performance Highlights**: 시뮬레이션 결과, RoboNaldo는 프리킥에서 평균 0.899m의 오차를 보이며 공은 시속 14.79m로 발사된다. 실제 Unitree G1 장비에서 3m 거리에서 각각 0.73m 및 0.86m의 평균 목표 슈팅 오차를 기록하였고, 공의 속도는 13.10m/s에 달한다. 이러한 결과는 RoboNaldo가 고속이면서도 정확하고 안정적인 슈팅 정책을 학습했음을 보여준다.



### Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning (https://arxiv.org/abs/2606.11087)
- **What's New**: 본 논문에서는 RL(강화 학습) 알고리즘의 정책 최적화와 관련하여, 훈련 시 안정성을 유지하면서 테스트 시간에 정책 개선을 수행할 수 있는 방안을 제시합니다. QGF(Q-Guided Flow)라는 새로운 RL 알고리즘은 테스트 시간에만 정책 최적화를 수행하며, 이를 통해 고차원 행동 공간에서도 이전의 테스트-시간 RL 방법들보다 뛰어난 성능을 보였습니다. 이 방법은 actor-critic 훈련의 불안정성을 피할 수 있어, 실용적이고 효과적인 RL 알고리즘을 제공합니다.

- **Technical Details**: QGF 알고리즘은 전통적인 행동 클로닝(behavioral cloning) 방식으로 레퍼런스 정책(reference policy)과 가치를 평가하는 비평가(value function critic)를 미리 훈련시킵니다. 테스트 시간에는 학습된 비평가의 가치 기울기(value gradient)를 사용하여 레퍼런스 정책을 고급 가치 행동으로 안내하며, 추가적인 정책 학습 없이도 성능을 높일 수 있습니다. 이러한 과정은 복잡한 장기 소음 제거 프로세스를 필요로 하지 않아, 계산 비용과 불안정성을 줄이는 장점을 가집니다.

- **Performance Highlights**: QGF는 다양한 테스크와 목표 기반 오프라인 RL 벤치마크에서 이전의 RL 알고리즘들과 비교하여 우수한 성능을 기록하였습니다. 이 알고리즘은 모델 크기가 증가할수록 성능이 향상되며, 훈련 동안 변화하는 비평가에 최적화하는 대가 없이 안정적인 감독 학습 손실로만 훈련될 수 있음을 보여줍니다. 결과적으로, QGF는 생성적 행동 모델을 최적화하기 위한 안정적이고 확장 가능한 가능성을 제시합니다.



### Unifying Local Communications and Local Updates for LLM Pretraining (https://arxiv.org/abs/2606.11081)
Comments:
          38 pages, 9 figures

- **What's New**: 이 논문에서는 GASLoC라는 새로운 분산 최적화 알고리즘을 소개하며, 이는 통신(communication) 효율성을 높이는 혁신적인 방법을 제시합니다. GASLoC는 기존의 All-Reduce 방식 대신 로컬 옵티마이저와 희소한 랜덤 피어 통신을 통해 모델 업데이트와 통신을 분리합니다. 이를 통해 학습 속도와 통신 비용을 동시에 개선하며, 특히 이질적인 대역폭을 가진 환경에서도 안정성을 제공합니다.

- **Technical Details**: GASLoC는 통신 가속화(communication acceleration) 개념을 기반으로 하며, 기존의 DiLoCo와 같은 외부 옵티마이저를 활용하여 지역 최적화(local optimization) 단계와 통신 단계를 통합합니다. 이 알고리즘은 최적화 성능을 보존하면서도 노드 간 통신 링크 수를 줄이며, 전통적인 All-Reduce보다 효율적인 최적화를 가능케 합니다. 또한, GASLoC는 로컬 업데이트와 통신 간의 혼합 패턴을 '22-Peer 규칙'을 통해 정의합니다.

- **Performance Highlights**: GASLoC는 다양한 LLM 훈련 과제에서 기존의 최첨단 분산 알고리즘보다 높은 성능을 보입니다. 특히, H=1 환경에서는 DiLoCo와 비슷한 성능을 내면서도 글로벌 동기화 없이도 상당한 시간 절약을 제공합니다. 이 알고리즘은 불균형 대역폭 환경에서도 우수한 성능을 발휘하며, 학습의 안정성과 결함 허용성(fault tolerance)을 크게 향상시킵니다.



### Modeling Complex Behaviors: Multi-Personality Composition and Dynamic Switching in Vision-Language Models (https://arxiv.org/abs/2606.11074)
- **What's New**: 이번 연구는 Multimodal Large Language Models (MLLMs)에서의 명시적 personality conditioning을 소개하고, 이를 기반으로 단일성격 유도, 다중성격 유도, 성격 전환 등을 포함하는 체계적인 평가 프레임워크를 수립합니다. 실험 결과에 따르면, 성격 유도가 이미지 자막 생성(image captioning)에 긍정적인 영향을 미치지만, visual question answering (VQA)과 같은 정밀한 추론이 필요한 작업에서는 성능에 악영향을 줄 수 있습니다. 또한, 다중 성격 조합과 동적 전환 과정에서 이전과 현재의 성격 제약에 의해 모델 행동이 상호 조정(co-modulated)됨을 확인했습니다.

- **Technical Details**: 본 연구에서는 MLLMs의 성격 조건을 체계적으로 조사하기 위해 personality assessment와 downstream task(예: 이미지 자막 생성, VQA)를 통해 평가를 진행합니다. 단일 성격 유도에서 관찰한 내용을 바탕으로, 다중 성격 유도를 통해 복잡하고 현실적인 성격 구성을 모델링합니다. 또한, 다중 턴 상호작용에서의 동적 성격 전환을 정의하여, 같은 대화 내에서 연속적인 턴에서 서로 다른 성격 설정을 따르는 현상을 분석합니다.

- **Performance Highlights**: 실험 결과, 성격 요소를 모델에 통합하는 것이 이미지 자막 생성에서 성능을 향상시킬 수 있지만, VQA 작업에 부정적인 영향을 미치는 것을 발견했습니다. 다중 성격 통합 및 동적 성격 전환 과정에서 서로 다른 성격 간의 상호 취소 및 균형 효과도 관찰되었습니다. 결과적으로, 기존의 프롬프트 기반 성격 유도 방법은 다중 모달 환경에서 효과가 제한적이며, MLLMs에 맞춤형으로 적합한 강력한 성격 유도 방법이 필요함을 강조합니다.



### T1-Bench: Benchmarking Multi-Scenario Agents in Real-World Domains (https://arxiv.org/abs/2606.11070)
Comments:
          Preprint

- **What's New**: T1-Bench는 현실적이고 다양한 다중 도메인 환경에서 에이전트 시스템을 평가하기 위한 고충실도 벤치마크입니다. 이 벤치마크는 구조적 추론이 필요한 복잡한 멀티턴 유저-어시스턴트 상호작용을 포함하며, 25개의 다양한 도메인에서 구성적 복잡성과 평가의 엄격함을 크게 증가시킵니다. 이를 통해 기존 벤치마크가 간과한 여러 도메인 간 상호작용을 포착할 수 있어, 에이전트의 신뢰성과 능력을 보다 효과적으로 평가할 수 있습니다.

- **Technical Details**: T1-Bench는 대화형 AI 에이전트의 툴 호출 기능을 평가하기 위해 자동화된 프레임워크로, 사용자와 도우미 에이전트 간의 상호작용을 시뮬레이션합니다. 이 시스템은 유저 에이전트가 현실적인 고객 발화를 생성하며, 도우미 에이전트가 도메인 별 툴/APIs를 호출하여 결과를 제공합니다. 이를 통해 복잡한 다단계 작업을 수행하며, 사용자 정책과 도우미 정책을 설정하여 보다 사실적인 대화를 구현하고 있습니다.

- **Performance Highlights**: T1-Bench는 12개의 다양한 모델을 활용하여 에이전트 행동, 툴 활용, 대화 품질을 평가하는 표준화된 프레임워크를 제공합니다. 멀티 도메인 환경에서의 작업 복잡성, 상호작용 깊이 및 도메인 다양성을 크게 향상시켰으며, 모형의 성과를 평가하는 데 있어 인간의 판단도 보완하였습니다. 마지막으로, 연구자들은 T1-Bench의 데이터와 코드를 오픈 소스로 공개하여 연구 커뮤니티의 지속적인 발전을 지원할 예정입니다.



### AuRA: Internalizing Audio Understanding into LLMs as LoRA (https://arxiv.org/abs/2606.11033)
- **What's New**: 본 연구에서는 AuRA (Audio Understanding as LoRA)라는 새로운 방법론을 제안합니다. AuRA는 모든 입력 음성을 ASR 인코더(교사)와 LoRA로 조정된 LLM(학생)으로 동시에 처리하여 음성 인식을 LLM 내부로 통합하는 기법입니다. 이를 통해 기존의 복잡한 ASR-LLM 파이프라인과 대규모 다중 모드 훈련 없이도 효율적인 음성-언어 모델링을 달성할 수 있습니다.

- **Technical Details**: AuRA는 ASR 인코더를 교사로, LoRA로 조정된 LLM을 학생으로 설정하여 레이어 별 증류(layer-wise distillation)을 통해 음성 표현을 LLM에 직접 내재화합니다. 이 방법은 오디오 정보가 LLM의 초기 변환에 직접적으로 들어가도록 하여 음성-언어 공동 모델링을 가능하게 합니다. 인퍼런스 시에는 ASR 인코더가 제거되며, LLM은 교사로부터 증류된 음성 표현을 활용합니다.

- **Performance Highlights**: AuRA는 여러 음성-언어 벤치마크에서 기존의 ASR-LLM 파이프라인, 음성-LLM 적응 기준선, 대규모 음성-언어 모델에 비해 효과성과 효율성 모두에서 뛰어난 성능을 나타냅니다. 또한, 이 방법은 인퍼런스 레이턴시와 메모리 사용량을 줄이면서도 성능을 개선할 수 있는 잠재력을 보여주고 있습니다.



### Diffusion Forcing Planner: History-Annealed Planning with Time-Dependent Guidance for Autonomous Driving (https://arxiv.org/abs/2606.11019)
Comments:
          CVPR2026

- **What's New**: Diffusion Forcing Planner (DFP)는 과거 정보를 활용하여 모션 플래닝의 안정성을 높이는 혁신적인 방법입니다. 기존 방법들이 이력을 정적 조건으로 사용하여 과거 패턴을 단순히 복제하는 경향이 있었던 반면, DFP는 이력을 유연하게 조절하여 다양한 환경의 변화에 적응할 수 있도록 설계되었습니다. 또한, DFP는 과거, 현재 및 미래 이동 경로를 분리하여 각 구간에 개별적인 노이즈 레벨을 도입하여 모델의 안정성을 높이고 있습니다.

- **Technical Details**: DFP는 전체 경로를 과거, 현재, 미래의 청크로 나누고, 각 청크마다 독립적인 확산 시간 단계를 샘플링합니다. 이러한 접근은 'noising-as-masking' 메커니즘을 통해 과거 정보를 조절하고, 원활한 경로 생성을 보장합니다. 학습 과정에서는 이력과 미래를 동시에 예측하여 인과적으로 일관된 조건부 생성을 학습하도록 유도하며, 평가 시에는 'classifier-free guidance (CFG)' 기법을 통해 조정 가능한 방식으로 미래 샘플링을 유도합니다.

- **Performance Highlights**: DFP는 대규모 현실 세계 자율 플래닝 벤치마크인 nuPlan에서 우수한 성능을 나타냅니다. 결과적으로 DFP는 경쟁력 있는 성능을 달성하면서도 복잡한 주행 상황에서 연속적이고 안정적인 모션 계획 경로를 생성하는 능력을 보여줍니다. 실험 결과, 적절한 이력 유도를 활용함으로써 모션 플래닝에서 효과적인 메커니즘을 제공함을 입증하였습니다.



### Understanding and mitigating the risks of OpenClaw for non-technical users: A practical guide with Sk (https://arxiv.org/abs/2606.11007)
Comments:
          Work in progress

- **What's New**: OpenClaw는 복잡한 다단계 작업을 자율적으로 실행할 수 있는 인공지능(AI) 에이전트 프레임워크로, 그 사용 가능성이 빠르게 증가하고 있습니다. 그러나 이로 인해 비기술 사용자들이 직면할 수 있는 보안 위험이 커지고 있으며, 이에 대한 접근하기 쉬운 안내가 필요합니다. 본 논문에서는 비기술 사용자를 위한 실용적인 보안 가이드를 제공하여 위험을 최소화하는 방법을 제시하고 있습니다.

- **Technical Details**: OpenClaw는 사용자의 장치에서 로컬로 배포된 자가 호스팅된 인텔리전트 게이트웨이로, 대규모 언어 모델(LLM)과 사용자의 지침을 통합하여 작업을 수행합니다. 사용자는 Reddit, YouTube 등의 콘텐츠를 모니터링하고 요약하며, 복잡한 n8n 워크플로우를 관리할 수 있습니다. 또한, 보안 설정 자동화를 위한 OpenClaw Skill을 통해 사용자들이 시스템을 쉽게 보호할 수 있도록 돕습니다.

- **Performance Highlights**: OpenClaw는 여러 도메인에서 강력한 성능을 보이며, 특히 생산성 자동화 및 개발 도구와의 협력이 돋보입니다. 그러나 보안 우려도 커지고 있어, 사용자는 시스템 권한을 높이면서도 적절한 보호 장치가 부족하다는 점이 문제입니다. 이 연구에서는 비기술 사용자가 쉽게 이해하고 실천할 수 있는 위험 식별 및 방어 전략을 제안하여, 효율적인 자산 보호를 가능하게 합니다.



### Optimizing 2D Input Representations and Sub-phase Fusion Strategies for Differential Diagnosis of Asthma and COPD Using CNN- and GRU-Based Networks (https://arxiv.org/abs/2606.10972)
- **What's New**: 본 연구는 VAR 모델의 성능을 멜 주파수 켑스트럼 계수(MFCC) 매트릭스와 로그 멜 스펙트로그램과 비교하여 심층 학습을 통해 평가하고자 합니다. 폐 소리 분류에서 스펙트로그램 기반 표현은 호흡 주기 길이에 따라 일관되지 않은 시간적 차원 문제를 겪습니다. 이를 해결하기 위해 전통적인 다듬기(trim)/제로 패딩(zero-padding) 외에 적응 길이 윈도잉(adaptive-length windowing) 기법이 도입되었습니다.

- **Technical Details**: 자세한 매개변수 최적화를 통해 스펙트럼 및 시간적 차원이 조정되었습니다. 다양한 합성곱 신경망(CNN) 아키텍처가 두 차원 표현에서 특징을 추출하기 위해 사용되었으며, 추출된 하위 단계 특징들은 직접 결합, 게이티드 순환 유닛(GRU) 네트워크 및 주의(attention) 메커니즘을 포함한 여러 전략으로 융합되었습니다. 모델 성능 평가는 각 호흡 주기 및 피험자 기반 평가를 통해 이루어졌습니다.

- **Performance Highlights**: 가장 높은 주기 기반 F1-score(0.877)는 13개의 계수와 64포인트 시간 해상도를 가진 MFCC 매트릭스를 사용하여 직접 피쳐 결합을 통해 얻어졌습니다. 또한, 피험자 기반 F1-score(0.855)는 적응 길이 윈도잉을 통해 13개의 계수와 256포인트 시간 해상도를 사용하는 MFCC 매트릭스를 통해 도출되었습니다. 데이터 증강 기법은 모델 성능에 전반적으로 부정적인 영향을 미쳤지만, mixup 증강 기법만이 테스트된 방법 중 가장 나은 성능을 보였습니다.



### Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning (https://arxiv.org/abs/2606.10968)
- **What's New**: 이 논문에서는 CPPO(Cumulative Prefix-divergence Policy Optimization)라는 새로운 정책 최적화 기법을 제안합니다. 기존의 PPO 스타일의 방법들이 모든 토큰에 동일한 임계값을 적용하는 반면, CPPO는 특정 토큰의 위치를 감안하여 정의된 동적 제약을 통해 더 정교한 처리를 제공합니다. 이는 오토리그레시브(autoregressive) 생성 과정에서 발생하는 비대칭성과 누적된 편향을 반영하여, 더 안정적인 학습과 정확한 추론을 가능하게 합니다.

- **Technical Details**: CPPO는 두 가지 결합된 메커니즘을 통해 정책 업데이트를 조정합니다. 첫 번째로, 위치 가중치를 고려한 토큰 수준의 임계값을 설정하여 초기 위치에서의 변화에 보다 엄격한 제한을 두어 전체 시퀀스에 미치는 영향을 줄입니다. 두 번째로, 누적된 편향 예산을 사용하여 역사적 편향이 일정 기준을 초과하면 추가적인 변화를 제한하여 컴파운딩 오류(compounding errors)를 방지합니다.

- **Performance Highlights**: 실험 결과, CPPO는 다양한 모델 규모에서 훈련의 안정성을 향상시키고 추론 정확도를 유의미하게 개선했습니다. CPPO는 matched RLVR 설정에서 테스트되었으며, 다양한 모델 크기에 걸쳐 최고의 AIME24/25/26 평균 점수를 기록했습니다. 이러한 성과는 두 가지 규제 조건의 효과를 뒷받침하는 절단 실험을 통해 확인되었습니다.



### Generative Explainability for Next-Generation Networks: LLM-Augmented XAI with Mutual Feature Interactions (https://arxiv.org/abs/2606.10942)
Comments:
          7 pages, with one page for appendix. Accepted for publication at the 2025 21th International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob)

- **What's New**: 이 논문은 AI/ML 모델의 투명성 부족이 운영자의 신뢰에 중요한 장벽이 되고 있다는 문제를 발표합니다. 기존의 XAI(Explainable Artificial Intelligence) 기술들은 비전문가에게 설명 가능한 통찰을 제공하는 데 실패하고 있으며, 이는 사용이 한정적이라는 것을 보여줍니다. 본 연구는 SHAP(SHapley Additive exPlanations) 기능 영향 값의 표준 사용을 넘어서기 위해 구조화된 프롬프트와 상호 기능 상호작용 데이터를 사용하여 인간이 이해할 수 있는 설명을 생성하는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 AI/ML 모델의 결정을 해석 가능한 방식으로 향상시키기 위해 네 가지 구성 요소로 이루어져 있습니다: AI/ML 모델, XAI 방법, 설명 증강 모듈, 대시보드입니다. 본 연구는 광학 경로의 QoT(quality of transmission) 추정 문제를 다루며, 이를 위해 XGBoost 모델을 사용합니다. 또한, SHAP 기능 영향과 상호 기능 상호작용 값을 사용하는 구조화된 프롬프트를 통해 논리적으로 사고하는 LLM(large language model)의 힘을 활용합니다.

- **Performance Highlights**: 실증 평가 결과, 제안된 프레임워크는 생성된 설명의 유용성에서 12.2%, 범위에서 6.2% 개선을 보였으며 정확도는 97.5%에 달했습니다. 이는 기존의 단순 SHAP 기능 영향만 사용하는 최첨단 접근 방식에 비해 현저한 개선을 나타냅니다. 이러한 개선은 네트워크 자동화에서 AI/ML의 투명성을 높이고 신뢰성을 향상시키는 데 기여할 것으로 기대됩니다.



### Democratising Camera Trap AI: An Open-Source Model for Detecting UK Mammals (https://arxiv.org/abs/2606.10940)
Comments:
          15 Pages, 4 Figures

- **What's New**: 이번 연구에서는 영국의 28개 동물 및 조류 종과 인간, 교정 기둥, 차량을 포함한 31개의 클래스에 대한 오픈 소스(object detection model)를 발표합니다. 이 모델은 Conservation AI와 Trap Tracker를 통해 수년 간 수집된 48,165개의 레이블링된 인스턴스를 사용하여 훈련되었습니다. 특정한 상업적 플랫폼에 의존하지 않고도 생태학자들이 쉽게 활용할 수 있도록 하여, 기존의 상업 모델에 대한 대안으로 사용될 수 있습니다.

- **Technical Details**: 모델은 YOLO26x 감지기를 기반으로 하며, 클래스별로 80% 훈련, 10% 검증, 10% 테스트 세트로 나누어 훈련 및 테스트를 수행하였습니다. 이 모델은 IoU(Intersection over Union) 0.5에서 평균 정확도(mean Average Precision) 0.984를 기록하였고, 정밀도(precision) 0.988, 재현율(recall) 0.965를 보였습니다. 훈련에 사용된 데이터와 동일한 현장에서 측정된 결과로, 새로운 장소에서의 성능은 향후 연구로 남겨두었습니다.

- **Performance Highlights**: 모델의 성능은 31개 클래스에 대해 평균적으로 0.96에서 0.99의 신뢰도(confidence)를 기록했으며, 0.17%의 잘못된 음성(false-negative) 비율이 발견되었습니다. 특히, 낮은 조도(night-time)나 먼 거리(distant), 또는 가려진 이미지(occluded images)에서 어려움이 있었습니다. 모델은 ONNX 형식으로 배포되어 생태학자들이 쉽게 활용할 수 있도록 지원합니다.



### Provenance Tracking in AI Compilers through the Lens of Coalgebra (https://arxiv.org/abs/2606.10937)
- **What's New**: 이번 논문은 AI 컴파일러의 프로베넌스(provenance) 추적 문제를 해결하기 위해 경량화된 생성적 접근 방식을 제안합니다. 기존의 태그 기반 및 후속 그래프 매칭 방법들은 비효율적이며 외부적 수정이나 높은 컴퓨팅 비용을 요구합니다. 저자들은 관찰적 의미론(observational semantics)을 기반으로 한 새로운 방법론을 통해, 연산 그래프의 변환에서 프로베넌스를 관찰하고 추적할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구는 코알제브라적 모델(coalgebraic model)과 약 동형성(weak bisimulation)을 사용하여 AI 컴파일러 변환에 대한 관찰적 올바름(observational correctness)을 형식화합니다. 이는 인터미디어트 노드가 제거될 때도 프로베넌스를 보존할 수 있도록 돕습니다. 결과적으로, 고쳐 쓴 특정 오퍼레이터 실행을 관찰 가능한 이벤트로 재구성하여 프로베넌스 정보를 자동 생성하고 생성적 구현(generative implementation) 전략을 통해 컴파일러의 논리 변경 없이 삽입하고 전파할 수 있습니다.

- **Performance Highlights**: COVAN이라는 프로토타입 AI 컴파일러의 구현을 통해 제안된 방법론의 효과성을 평가합니다. 이 시스템은 PyTorch 및 TVM Relay IR과 통합되어 안정적인 프로베넌스 추적을 제공하며, 그래프의 공격적인 리라이트를 통해 최소한의 엔지니어링 오버헤드로 포스트 프로세싱(postprocessing) 오퍼레이터를 안정적으로 연결합니다. 이는 디버깅 및 감사(auditing) 작업을 지원하며, AI 컴파일러의 투명성 및 책임을 증대시킵니다.



### CLP: Collocation-Length Prediction for Zero-Loss Adaptive Multi-Token Inferenc (https://arxiv.org/abs/2606.10935)
Comments:
          13 pages, 8 figures, 8 tables

- **What's New**: 이 논문에서는 기존의 Multi-token prediction (MTP) 접근 방식에서 발생하는 품질 저하 문제를 해결하기 위해, Backbone-as-Architect 디자인 원칙을 제안합니다. 이를 통해 첫 번째 토큰은 항상 백본 LM 헤드에 의해 생성되고, MTP 헤드는 이후 토큰만을 담당함으로써 헤드-백본 간의 경쟁을 제거합니다.

- **Technical Details**: 새로운 Collocation-Length Predictor (CLP)는 각 디코딩 단계에서 안전하게 수용할 수 있는 추가 토큰 수를 예측하는 경량의 span-level decision layer로, 단일 선형 레이어만을 사용해 4.6K에서 7.7K의 매개변수를 운영합니다. 이전 연구에 비해 과도하게 설계된 1M 파라미터 게이트 네트워크를 대체함으로써 효율성을 높였습니다.

- **Performance Highlights**: Qwen2.5 모델에 대한 실험 결과, CLP는 품질 저하 없이 1.5B 모델에서 1.20x에서 1.29x의 속도 향상을 달성하였고, 7B 모델에서는 1.14x에서 1.20x의 향상을 보였습니다. 이와 동시에, MTP 헤드의 예측 정확도가 가속화의 주요 제약 요소로 확인되어, 보다 나은 디자인 원칙을 제시함으로써 향후 개선의 명확한 로드맵을 제시합니다.



### Recoverable but Not Stationary:Local Linear Structures in Weights and Activations (https://arxiv.org/abs/2606.10929)
Comments:
          23 pages, 8 tables, 9 figures

- **What's New**: 이 논문에서는 학습된 행동이 선형 방향(linear directions)으로 제어될 수 있음을 보여주며, 특히 Task Vectors, LoRA, 활성화 조정(activation steering), 그리고 사전 훈련된 가중치 주변의 랜덤 서치(random search)에 대해 논의합니다. 연구 결과로는, 강한 국소 저차원(task-gradient structure)을 발견했으나, 고정된 작업 평면(fixed-task-plane) 가설은 기각되었습니다. 이는 유용한 기저(basis)가 100단계 내에서 상당히 변화함을 보여줍니다.

- **Technical Details**: 이 연구는 다중 작업 트랜스포머(synthetic multitask transformer)와 DistilGPT-2/GPT-2의 LoRA 어댑터를 사용하여 선형 구조의 존재 여부와 그 규모를 탐구합니다. 랜덤 서치 이론(random search theory)을 개발하고 가우시안 로컬-선형 정리를 통해 고차원에서도 랜덤 파라미터 검색의 효율성을 정당화합니다. 또한, 단일 그래디언트 단계가 레이블 대비 조준 벡터(labelled-contrast CAA steering vector)와 0.58의 코사인 유사도를 가지는 활성화 변화를 생성한다는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, 회복 방향이 실제로 존재하고, 저차원적이며 유용하지만 고정된 작업 평면 내에는 포함되지 않음을 발견했습니다. 초기 회복 업데이트가 대부분의 제거(displacement)를 포착하는 최적의 탐색(subspace)으로을 형성함을 보여주었고, LLM 스케일에서도 유사한 패턴을 관찰했습니다. 또한, 랜덤 서치가 10^9 차원 공간에서도 작동한다는 놀라운 결과를 이론적으로 설명하고, 가중치 변경(weight edits)과 활성화 조정이 상관관계가 있음을 밝혔습니다.



### A Constrained Natural-Language Interface for Variational Multi-Physics Finite Element Simulations in FEniCS (https://arxiv.org/abs/2606.10928)
Comments:
          23 pages, 17 figures

- **What's New**: 이번 논문에서는 다중 물리학 유한 요소 해석(Multi-Physics Finite Element Analysis)을 위한 제약된 자연어 인터페이스를 소개합니다. 이는 대형 언어 모델(LLM)이 특정 프론트엔드 작업만 수행하게 제한되어 있으며, 이를 통해 불확실성을 줄이고 유한 요소 해석의 신뢰성을 높이고자 합니다. LLM은 오직 자연어를 구조화된 JSON으로 변환하거나, 비카탈로그 형상을 위한 Gmsh 코드를 생성하는 데 사용됩니다.

- **Technical Details**: 제안된 시스템은 사용자가 제공한 자연어 설명을 구조화된 사양으로 변환하고, 검증된 사양을 바탕으로 다섯 개의 인간 작성 UFL 템플릿(선형 탄성, 초탄성, 탄소-소성, 열-기계 결합 및 위상-필드 파괴)으로 라우팅합니다. LLM은 FEniCS 솔버 템플릿을 작성하거나, 약한 형식을 유도하지 않으며, 수치적으로 민감한 모든 내용은 결정론적 코드에 의해 처리됩니다. 이를 통해 시스템의 안정성과 신뢰성을 높일 수 있습니다.

- **Performance Highlights**: 이 시스템은 두 가지 프론트엔드 벤치마크에서 높은 성과를 보였습니다. 첫 번째 프롬프트 분석 벤치마크에서는 15개의 경우 중 9개에서 첫 번째 시도에서 유효한 분석 결과를 얻고, 나머지는 재시도로 모두 수정되어 최종 유효 분석률이 100%에 도달했습니다. 또한, 커스텀 형상 벤치마크에서 90%의 성공률을 기록하며, 이 결과들은 제안된 시스템의 효율성을 잘 보여줍니다.



### What Do Deepfake Speech Detectors Actually Hear? (https://arxiv.org/abs/2606.10912)
Comments:
          Accepted to Interspeech 2026

- **What's New**: 본 논문은 Deepfake 음성 탐지 시스템의 해석 가능성을 향상시키기 위한 오디오 네이티브 설명 가능성 파이프라인을 제안합니다. Integrated Gradients를 이용하여 시간이 정렬된 자기지도 표현에서 결정 증거를 국소화하며, 3가지 WavLM 기반 탐지기(AASIST, CA-MHFA, SLS)에 적용하여 ASVspoof 5 데이터셋에서 성능을 비교하였습니다. 각 탐지기는 서로 다른 단서를 기반으로 작동함을 확인하였으며, 이는 탐지기가 실제로 무엇을 학습하는지를 이해하는 데 도움을 줍니다.

- **Technical Details**: 본 연구는 결정을 내리는 데에 Integrated Gradients (IG) 방법을 활용하여 음성 탐지기의 결정 논리를 해석합니다. WavLM 베이스+ 모델을 활용하여 입력 녹음에서 은닉 표현을 생성하고, 이를 스코어(logit)로 매핑하여 각 요소의 기여도를 계산합니다. IG는 스코어의 기여도를 각 요소에 할당하고, 이러한 기여도는 탐지기의 초점이 어디에 있는지를 국소화하는 데 사용됩니다.

- **Performance Highlights**: 탐지 시스템(AASIST, CA-MHFA, SLS)은 유사한 성능(3.98%-5.26% EER)을 보였지만, 각각의 탐지기는 독립적인 단서에 의존합니다. AASIST는 비음성 및 환경 단서에 집중하고, CA-MHFA는 국소화된 음소 정보에 초점을 맞추며, SLS는 단어 경계 및 스펙트럼 무결성 단서에 의존합니다. 실험 결과는 탐지기의 설명 가능성과 결정 논리의 유효성을 입증하는 중요한 데이터를 제공합니다.



### Ethical and Technical Limits of Deepfake Speech Datasets (https://arxiv.org/abs/2606.10911)
Comments:
          Accepted to Interspeech 2026

- **What's New**: 본 논문은 딥페이크 음성 감지의 견고성과 공정성에 대한 주장을 뒷받침하는 데이터셋에 대해 심층적으로 감사하였습니다. 저자들은 39개의 딥페이크 음성 데이터셋을 수집하고 분석하여 접근성, 문서화, 인구 통계 및 언어 범위, 데이터셋 규모 등을 조사하였습니다. 감사 결과, 대부분의 데이터셋이 인구 통계 메타데이터를 결여하고 있고 일부만 성별 또는 언어 레이블을 포함하고 있음을 발견했습니다. 또한, 데이터셋 간의 하위 그룹 분석이 어렵고, 저자들은 데이터셋 간의 겹침이 크다는 사실을 밝혀냈습니다.

- **Technical Details**: 이 연구는 39개의 딥페이크 음성 데이터셋에 대해 데이터 수준에서 감사합니다. 감사는 각 데이터셋의 속성, 특히 신뢰할 수 있는 스피치 소스의 중복, 정의된 인구 통계 메타데이터의 부족 및 합성 도구의 다양성을 중심으로 진행되었습니다. 저자들은 데이터셋의 표준화된 검토 및 통계적 측정을 기반으로 데이터의 학습 및 평가 적합성을 평가합니다. 이러한 기술적 분석을 통해 공정한 평가를 위한 불균형이 확인되었습니다.

- **Performance Highlights**: 참고로, 많은 데이터셋이 동일한 신뢰할 수 있는 음성 소스에서 파생되어 있습니다. 이로 인해 데이터셋 간 평가의 유효성이 저하되고 있는 상황입니다. 본 연구는 공정성 및 일반화 주장에 대한 새로운 통찰을 제공합니다. 또한, 데이터셋 관련 메타데이터가 부족하여 감지 솔루션의 효과를 다양한 그룹에 걸쳐 보장할 수 없음을 강조하며, 다국어 및 다양한 인구 통계 요소를 포함한 데이터셋의 필요성을 제기합니다.



### RAT: Reference-Augmented Training for ASV Anti-Spoofing (https://arxiv.org/abs/2606.10908)
Comments:
          Accepted to Interspeech 2026

- **What's New**: 이번 논문에서는 스피커 참조 녹음을 기반으로 한 스푸핑(Spoofing) 방지 아키텍처를 제안하며, 흥미롭게도 추론(Inference) 동안 참조(reference)를 무시하는 해결책으로 수렴한다고 확인하였습니다. 참조 채널 없이도 성능을 개선할 수 있는 훈련 방식인 Reference-Augmented Training (RAT)를 도입하여, 단일 발화 기반보다 향상된 감지 성능을 보여줍니다.

- **Technical Details**: RAT는 스피커 참조 녹음을 사용하는 훈련 전략으로, SSL(Self-supervised learning) 모델인 XLS-R을 활용하여 특징 벡터를 추출합니다. 우리가 사용한 아키텍처는 두 개의 병렬 브랜치로 이루어진 Reference-Informed Block (RIB)을 포함하며, MLP(Multi-layer Perceptron) 브랜치와 크로스-어텐션(Cross-Attention) 브랜치를 통해 참조 정보를 통합합니다. 모델은 학습 과정에서 참조의 기여도를 감소시키는 최적화를 통해, 최종적으로 참조 채널의 영향을 크게 받지 않도록 만들어집니다.

- **Performance Highlights**: RAT를 통해 ASVspoof 5 벤치마크에서 2.57%의 EER(Equal Error Rate)와 0.074 minDCF를 기록하며, 대규모 앙상블 시스템을 초월하는 최첨단 성능을 달성하였습니다. 논문에서는 관련 코드와 모델 가중치, 성능 평가 세트를 공개하여 연구자들이 이를 재현할 수 있도록 지원합니다.



### Human-AI Teaming Through the Lens of Calibration (https://arxiv.org/abs/2606.10906)
Comments:
          19 pages, 5 figures (including appendix)

- **What's New**: 본 연구에서는 통계적 보정(statistical calibration)의 관점에서 인간-AI 팀을 위한 모델을 연구합니다. AI 모델과 인간이 특징 공간(feature space)의 일부에 대해 보정되어 팀을 구성하며, 이러한 보정 가정이 팀 프레임워크에 어떻게 전파되는지를 분석합니다. 특히, 인간과 모델의 예측을 결합하거나 한쪽에 예측 책임을 위임하는 두 가지 프레임워크를 고려합니다.

- **Technical Details**: 이 연구는 통계적 보정의 이론을 통해 인간-AI 팀을 분석합니다. 연구진은 모델과 인간, 두 구성원이 각각 특징 공간의 특정 구획에 대해 보정(calibrated)되었다고 가정합니다. 보정은 인간과 모델의 예측 품질을 일반적으로 표현할 수 있는 유용한 도구이며, 구획이 세분화됨에 따라 인간/모델의 예측은 베이즈 예측자(Bayes predictor)에 접근하게 됩니다. 연구 결과, 기존의 팀 절차는 팀 구성원의 기본 보정을 보존하지 않는다는 주장을 합니다.

- **Performance Highlights**: 모델 기반 조합이 모델에 대한 보정을 보존하지만 인간에 대해서는 보존하지 않는다는 이론적 결과를 입증했습니다. 예측 작업을 한 팀원에게 위임하는 방식은 메타 분류기(micro classifier)인 거부자(rejector)가 충분히 보정되어야 함을 요구하며, 이는 인간이 비가시적 특징(hidden features)에 접근할 경우 불가피한 초과 위험(excess risk)을 수반합니다. 이 모든 결과는 시뮬레이션 및 인간 예측 실험을 통해 검증되었습니다.



### Pose-ICL: 3D-Aware In-Context Learning for Pose-Controllable Subject Customization (https://arxiv.org/abs/2606.10902)
- **What's New**: Pose-ICL(프레임워크)라는 새로운 튜닝 없는 시스템을 제안하여 효과적인 포즈 제어가 가능한 맞춤형 이미지 생성을 목표로 하고 있습니다. 이 프레임워크는 여러 이미지-포즈 쌍을 통해 새로운 주체에 직접 адапта틱(Facilitating)할 수 있도록 구성되어 있습니다. 기존 방법들이 3D 지각 부족으로 인해 포즈 일관성을 유지하지 못하는 문제를 다루고 있으며, 특히 Surface-Anchored Position Embedding(SAPE)을 통해 3D 인식을 강화하여 포즈 정확성을 높인 점이 특징입니다.

- **Technical Details**: Pose-ICL은 3D-aware In-Context Learning(ICL) 패러다임을 이용하여 이미지 생성에서 주체의 위치를 효과적으로 제어합니다. 이 프레임워크는 사용자 제공 이미지와 포즈를 결합하고, 이를 통해 생성된 이미지를 컨텍스트로 활용하여 포즈 일관성을 향상시킵니다. 그리고, 각 포즈에 대해 렌더링된 볼륨 경계 상자의 표면 좌표를 SAPE로 변환하여 모델에 3D 인식을 전달함으로써 이미지의 기하학적 관계를 명확히 합니다.

- **Performance Highlights**: 기존 방법들과의 비교 평가에서 Pose-ICL은 포즈 정확도와 정체성 일관성 등에서 현저한 성과를 보였습니다. 3D 자산 및 실제 주체에 대한 광범위한 평가를 통해, Pose-ICL은 효과적인 포즈 제어 및 높은 일관성을 보여주었습니다. 이러한 결과들은 Pose-ICL이 맞춤형 이미지 생성에서 어떻게 진일보했는지를 잘 보여줍니다.



### Improving Text-Instance Alignment Of Foreground Conditioned Out-Painting Via Customized Concept Embedding (https://arxiv.org/abs/2606.10892)
- **What's New**: 이 논문은 전통적인 작업 흐름이 요구하는 고품질 제품 이미지를 제공하기 힘든 문제를 해결하기 위해 텍스트 기반의 Foreground Conditioned Outpainting (FCO) 기술을 소개합니다. 기존 FCO 방법의 주요 문제 중 하나인 아티팩트(artifacts) 발생 문제를 해결하기 위해, Customized Concept Embedding Diffusion (CCE-Diffusion) 프레임워크를 제안합니다. CCE-Module을 중심으로, 텍스트 임베딩과 구체적인 시각적 인스턴스 간의 정렬을 강화하여 아티팩트를 줄이는 방안이 구체적으로 설명됩니다.

- **Technical Details**: CCE-Diffusion 프레임워크는 텍스트 프롬프트와 인스턴스 이미지 간의 미스 매칭을 수정하기 위해 설계된 CCE-Module을 포함하고 있습니다. 이 모듈은 인스턴스의 시각적 특징을 기반으로 개념 임베딩을 맞춤화하였으며, Instance-Aware Loss를 사용하여 최적화를 수행합니다. 또한, Semantic-Preserving Prompt Template을 도입하여 텍스트 프롬프트 내 다른 단어의 의미를 왜곡하지 않도록 합니다.

- **Performance Highlights**: 정성적 및 정량적 평가를 통해, CCE-Diffusion은 출력 이미지에서 아티팩트를 효과적으로 줄인 것으로 나타났습니다. 이 모듈은 ControlNet, BrushNet, BLD 등 다양한 FCO 방법과 쉽게 통합될 수 있는 플러그 앤 플레이(plug-and-play) 구성 요소로 제공되며, 그 통합을 통해 향상된 성능을 보입니다. 특히, FCO에서의 텍스트와 인스턴스 시각적 특징의 정렬 문제를 해결하여 이미지 품질을 개선하는 데 크게 기여합니다.



### Optimal Post-Training Quantization Scales and Where to Find Them (https://arxiv.org/abs/2606.10890)
- **What's New**: 이 논문에서는 Post-training quantization (PTQ)의 새로운 접근 방식인 PiSO (Piecewise Scale Optimization) 알고리즘을 제안합니다. PiSO는 캘리브레이션 데이터(calibration data)를 활용하여 정확하고 효율적으로 최적의 채널 기준 무게 스케일을 계산합니다. 본 연구에서는 스케일 최적화(scale optimization)와 오류 수정(error correction)을 효과적으로 결합하는 방법도 제시합니다. 특히, Llama 및 Qwen 모델에서의 실험 결과를 통해, PiSO가 정밀도와 정확도 향상에 기여함을 입증합니다.

- **Technical Details**: PiSO 알고리즘은 고정된 양자화 그리드(fixed quantization grid)의 한계에서 벗어나, 데이터 기반 재구성 목표(data-aware reconstruction objective)에 따라 스케일을 최적화합니다. 또한, 그룹 방식의 양자화(group-wise quantization)로 확장되어, 스케일 최적화와 무게-그리드 할당(weight-to-grid assignments)이 서로 상호작용할 수 있도록 하여 개선된 성능을 보여줍니다. 이 알고리즘은 RTN(결정을 가장 가까운 값으로 반올림하는 quantizer) 양자화 방식에 기반하고 있으며, 감정 데이터의 필요성을 줄이는 성과가 있을 것으로 기대됩니다.

- **Performance Highlights**: Llama와 Qwen 모델에 대해 수행한 여러 실험에서, PiSO를 사용함으로써 당초 예상된 것보다 더 우수한 성능 개선률이 나타났습니다. 특히, 목표 비트 폭(target bit-width)이 줄어들수록 양자화가 더욱 도전적으로 변할 때, PiSO의 이점이 더욱 두드러졌습니다. 결과적으로 PiSO는 기존의 오류 수정 방법과 결합 시에도 큰 효과를 보이며, 정확한 양자화에 필요한 캘리브레이션 데이터 양을 줄이는 데 기여합니다.



### LIBERO-Occ: Evaluating and Improving Vision-Language-Action Models under Scene-Induced Occlusion via Viewpoint Imagination (https://arxiv.org/abs/2606.10862)
Comments:
          14 pages, 7 figures

- **What's New**: 이 논문에서는 Vision-Language-Action (VLA) 모델이 현실적인 환경에서의 occlusion(차폐)에 대한 도전과제를 다룹니다. 새로운 벤치마크 LIBERO-Occ가 소개되어, VLA 모델의 성능 저하를 평가하는 데 도움을 줍니다. 또한, VIM(Viewpoint Imagination)이라는 새로운 기법을 통해 차폐 상황에서도 시각적 정보를 보완할 수 있는 방법을 제안합니다.

- **Technical Details**: 나온 기법 VIM은 occluded(차폐된) 관찰에서 보조 시점을 생성하여 행동 예측을 지원합니다. 이 과정은 두 단계로 이루어지며, 첫 번째 단계에서 모델은 occluded 관찰로부터 보조 시점을 생성하고 두 번째 단계에서 이를 행동 예측과 함께 최적화합니다. LIBERO-Occ는 다양한 occlusion(차폐) 조합과 강도를 평가할 수 있도록 구성되어 있으며, occlusion이 VLA 시스템에 미치는 영향을 체계적으로 분석합니다.

- **Performance Highlights**: 실험 결과, 기존 VLA 모델들은 scene-induced(장면 유도) occlusion 하에서 성능 저하가 발생하는 것으로 나타났습니다. 반면에 VIM 기반의 방법은 부분적으로 관찰 가능한 환경에서 조작의 견고성을 개선하는 데 효과적임을 보여줍니다. VIM은 추가 장비 없이도 occlusion 문제를 해결할 수 있을 것으로 보이며, 이는 로봇 조작의 실용에 기여할 수 있습니다.



### From Perception to Action: Can UI Interventions Foster Sustainable LLM Chatbo (https://arxiv.org/abs/2606.10861)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 기반 챗봇의 사용자 인터페이스(UI)에서의 지속 가능성 향상을 위한 개입(intervention)을 탐구합니다. 기존의 에너지 효율성 관리 전략이 모델 혹은 인프라에 초점을 맞춘 반면, 이 연구는 사용자 인터페이스에서의 개입의 중요성을 강조했습니다. 조사 결과, 다양한 UI 개입이 사용자 에너지 인식을 높이고 에너지 책임이 있는 챗봇 사용을 유도하는 데 기여할 수 있다는 사실이 밝혀졌습니다.

- **Technical Details**: 연구는 77명의 참가자와 함께한 기본 설문조사, 11명의 참가자를 대상으로 한 5일 간의 현장 연구를 통해 에너지 인식을 증가시키는 UI 개입의 효과를 평가했습니다. 연구에서는 세 가지 모드 스위치(에너지 효율적인 모드, 균형 모드, 퍼포먼스 모드)와 같은 UI 개입을 구현했으며, 개별 응답에 대한 에너지 피드백을 제공합니다. 이러한 인터페이스 변화는 정보의 가시성을 높여 사용자의 행동 변화를 유도하는 방식으로 설계되었습니다.

- **Performance Highlights**: 현장 연구에서 참가자들은 높은 정확도 요구가 없을 때 90.9%가 Eco 모드를 선택했다고 보고했으며, 에너지 효율성 모드가 전체 사용 프롬프트의 55.8%를 차지했습니다. 그러나 퍼포먼스 모드는 전체 에너지 발자국의 약 89%를 차지했음에도 불구하고 25% 미만의 프롬프트에서만 사용되었습니다. 이러한 결과는 UI 디자인이 사용자의 일상적인 행동을 변화시킬 수 있음을 보여주며, 백엔드 최적화와 함께 이루어져야 함을 강조합니다.



### Janus: A Benchmark for Goal-Conditioned Information Distortion in LLMs (https://arxiv.org/abs/2606.10852)
- **What's New**: 이번 논문에서는 사실에 기반한 LLM(대형 언어 모델)의 출력에서 목표 지향적 의사 왜곡(goal-conditioned distortion)을 측정하기 위한 JANUS라는 벤치마크를 소개합니다. 기존 평가 방법들이 주로 허위 정보 탐지에 초점을 맞춘 것과 달리, JANUS는 사실에 대한 선택적 접근과 왜곡을 다루며, 사실적 오류가 없는 상황에서도 의사소통의 신뢰성이 저하될 수 있음을 강조합니다. 이 벤치마크는 8개 도메인에 걸쳐 160개의 시나리오로 구성되어 있으며, 중립적 조건과 목표 지향 조건을 비교하여 공정한 평가를 가능하게 합니다.

- **Technical Details**: JANUS의 설계는 세 가지 속성을 바탕으로 합니다. 첫째, 사실에 기반한(fact-grounded) 구조로, 모든 시나리오에는 고정된 사실 집합이 제공되어 왜곡을 허구와 분리할 수 있습니다. 둘째, 각 모델은 동일한 수신인과 증거 기반의 중립 및 목표 지향 반응을 생성하며, 셋째, 유리한 사실과 불리한 사실은 각각의 제도적 목표 및 영향을 받는 개인이나 그룹에 따라 정의됩니다. 이 평가 시스템은 응답의 질을 5가지 왜곡 차원(selection, framing, emphasis, specificity, ordering)으로 분석하여, 사실적 정확성이 아닌 정보 전달 방식의 변화를 평가합니다.

- **Performance Highlights**: 실험 결과, LLM이 목표 지향적인 의도가 반영될 때 부정적인 정보를 완화하거나 비대칭적으로 강조하는 경향이 있음을 발견했습니다. 이러한 경향은 다양한 도메인과 모델 집단에 걸쳐 일관되었으며, 사실적 정확성만으로는 신뢰할 수 있는 의사소통을 평가하는 데 한계가 있다는 점을 보여줍니다. 또한, 연구진은 JANUS를 통해 정보 발표 방식에서의 목표 지향적 왜곡이 LLM의 실제 적용에서 중요한 고려사항임을 입증하였습니다.



### Geometrically Averaged Hard Target Updates for Linear Q-Learning (https://arxiv.org/abs/2606.10835)
- **What's New**: 본 논문에서는 일반적인 심층 Q-학습에서의 주기적 하드 타겟 업데이트를 개선하는 새로운 방법으로 $$-타겟 업데이트를 소개합니다. 이 방법은 $m$-주기 타겟 업데이트 맵을 $$-지오메트릭 가중치 $(1-)^{m-1}$로 평균내어 생성됩니다. 이 논문은 주로 선형 함수 근사를 사용하는 Q-learning에서의 특성을 연구하며, 확률적 강화학습 설정으로 확장될 수 있는 이론적 배경을 제공합니다.

- **Technical Details**: $$-타겟 업데이트는 주기적 하드 타겟 업데이트를 보다 유연하게 수행하며, $=0$은 1주기 타겟 업데이트로, 계속하여 $0;1$로 접근하면 PQVI(프로젝트 Q-값 반복)로 복원됩니다. 선형 Q-learning에 대해 switching-system 모델을 사용하여 이 메커니즘을 분석하고 있으며, 동일한 접근 방식은 결정론적 및 확률적 설정 모두에 적용될 수 있습니다. JSR(공통 스펙트럼 반경) 안정성을 통해 경계 오류의 균일한 기하급수적 경계를 제공하며, 수렴을 보장합니다.

- **Performance Highlights**: 제안된 $$-DLQL 방법은 DLQL의 끝점과 PQVI의 끝점을 연결하며, 이 경계 업데이트는 명시적인 단일 하드 타겟 주기를 선택하지 않고도 작성할 수 있도록 하였습니다. $$가 작을 경우 DLQL의 속성을 계승하고, $$가 1에 가까워질수록 PQVI의 속성을 계승합니다. 이 연구는 하드 타겟 업데이트 메커니즘의 기하학적 평균화를 통해 선형 Q-learning 및 프로젝트 Q-벨만 방정식에 대한 깊은 통찰을 제공합니다.



### Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models (https://arxiv.org/abs/2606.10829)
- **What's New**: 이번 논문에서는 여러 토큰을 한 번의 디노이징(iteration) 단계에서 드러내어 추론 단계를 줄일 수 있는 Masked Diffusion Language Models (MDLMs)를 다룹니다. 제안된 ADAS(Attention-Discounted Adaptive Sampler)는 기계 학습 훈련 없이 기존의 샘플러를 개선하여, 불확실한 예측을 가진 위치를 고려해 후보를 할인하는 방식으로 작업을 수행합니다. 이 방법은 성능 측면에서 타 샘플링 기법에 비해 9.11 및 10.46 포인트의 향상된 결과를 보여줍니다.

- **Technical Details**: ADAS는 기존의 샘플러의 종료 규칙을 변경하지 않고 단지 하위 집합의 구성을 수정하는 방식으로 동작합니다. 이는 고득점 후보가 반드시 서로 호환되지 않는다는 점을 강조하며, 선택된 불확실한 위치와 강한 관계를 갖는 후보를 할인하는 방식으로 구현됩니다. 이와 같은 설계는 기존의 샘플링 강도와 예산 규칙을 따릅니다.

- **Performance Highlights**: ADAS를 도입한 Top-k, EB-Sampler 및 Fast-dLLM은 LLaDA-8B-Base 및 Dream-7B-Base 데이터셋에서 낮은 NFE 성능을 9.11 및 10.46 포인트씩 향상시키며, 실험적으로 그 효과를 입증했습니다. 90개의 운영 지점 중 80곳에서 긍정적인 결과를 보여주며, 높은 병렬 디코딩에서의 품질 향상 가능성을 시사합니다.



### A Unified Siamese Learning Framework for Zero-Day Anomaly Detection and Classification in Optical Networks (https://arxiv.org/abs/2606.10827)
Comments:
          Authors' version of the manuscript accepted and published at the Optical Fiber Communication Conference (OFC) 2026. 4 pages, 3 figures

- **What's New**: 본 논문은 MS-SNN(multi-similarity Siamese neural network)이라는 새로운 프레임워크를 소개하며, 이를 통해 광 네트워크에서의 제로 데이(anomaly detection) 탐지 및 원샷(one-shot) 분류를 통합했습니다. MS-SNN은 다양한 빛 경로(lightpaths)와 네트워크 조건에서 일반화할 수 있는 능력을 갖추고 있으며, 이전에 관찰되지 않은(anomaly) 종류를 첫 번째 발생에서부터 학습할 수 있게 합니다. 이러한 방식으로, 99% 이상의 정확도를 달성하며 실시간으로 적응 가능한 네트워크 전체에서의 이상 탐지를 가능하게 합니다.

- **Technical Details**: MS-SNN 프레임워크는 세 가지 주요 구성 요소로 설계되었습니다: 임베딩/인코더, 다중 유사성 헤드(multi-similarity head), 및 유사성 계산 모듈입니다. 각 샘플은 공동으로 사용되는 인코더를 통해 고차원 특징 표현으로 변환되며, 이 후 다중 유사성 헤드가 서로 다른 거리 척도를 계산하여 샘플 간의 유사성을 평가합니다. 이 네트워크는 대조 학습(contrastive learning) 방식을 사용하여 이진 교차 엔트로피 손실 함수와 함께 훈련되며, 이를 통해 고급 유사성 메트릭을 학습할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 MS-SNN은 99.12%의 높은 정확도를 기록하며, 단일 메트릭(Single-metric) SNN 및 근접 이웃(nearest neighbor, NN) 알고리즘보다 우수한 성능을 보였습니다. MS-SNN은 단일 샘플로도 새로운 클래스의 탐지 및 분류를 가능하게 하여, 제로 데이(anomaly) 문제에 효과적으로 대응할 수 있습니다. 이 연구는 향후 광 인프라의 적응형 이상 지능(Anomaly Intelligence) 구현에 있어 중요한 단계를 나타냅니다.



### K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling (https://arxiv.org/abs/2606.10820)
- **What's New**: 본 논문에서는 고부하 배치 서비스에 대한 기존의 속도 향상 방법들이 해결하지 못한 문제를 다룸으로써, K-Forcing라는 새로운 언어 모델링 패러다임을 제안합니다. K-Forcing는 기존의 AR 모델에서 이어받은 조건적 푸시-포워드 매핑을 통해 동시에 여러개의 다음 k개 토큰을 생성할 수 있습니다. 이를 통해 상대적으로 적은 품질 저하로도 배치 속도를 크게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: K-Forcing는 먼저 사전 훈련된 AR 모델을 통해 푸시-포워드 매핑을 학습하는 프로세스를 가지고 있으며, 이를 통해 고정 길이의 여러 토큰을 한 번의 포워드 패스에서 생성할 수 있습니다. 이 방식은 AR 모델의 기본 구조를 유지하면서도 표준 AR 서비스 인프라와 호환되도록 설계되었습니다. K-Forcing의 주요 기술적 기여는 고정 길이의 출력과 공동 다중 토큰 샘플링을 가능하게 하는 것입니다.

- **Performance Highlights**: K-Forcing는 LM1B 및 OpenWebText 데이터셋에서 k=4 토큰을 포워드 패스 당 생성하며, 다양한 배치 크기에서 약 2.4~3.5배의 속도 향상을 보였습니다. 이 모델은 모드레이트한 품질 저하 속에서도 배치 서비스 처리량을 상당히 개선시키며, 이는 통계적 모델링 변경만으로도 AR 디코딩 대비 굉장한 배치 서비스 처리량 향상을 이끌어냈다는 것을 보여줍니다.



### Earth-OneVision: Extending Remote Sensing Multimodal Large Language Models to More Sensor Modalities and Tasks (https://arxiv.org/abs/2606.10819)
- **What's New**: 본 논문에서는 Earth-OneVision이라는 새로운 원격 감지(Remote Sensing, RS) 다중 모달 대형 언어 모델(Multimodal Large Language Model, MLLM)을 제안합니다. 이 모델은 6가지 센서 모달리티를 통합하여 9개 작업 범주에 걸친 크로스 센서 융합(cross-sensor fusion)을 지원합니다. Earth-OneVision은 2B 파라미터로 설계되어, 이전의 모델들보다 더 폭넓은 데이터와 작업을 처리할 수 있습니다.

- **Technical Details**: Earth-OneVision은 세 가지 전용 메커니즘을 도입하여 현재의 RS-MLLM이 직면한 세 가지 병목 현상을 해결합니다. 첫 번째, Full-Granularity Vision-Language Alignment (FGVLA)는 다층 시각적 특징과 다차원 언어 공간을 정렬하여 더 깊은 상호 작용을 가능하게 합니다. 두 번째, Spatial-Linguistic Isomorphic Serialization (SLIS)은 이질적인 공간 출력을 언어와 유사한 토큰 시퀀스로 직렬화하여 단일 생성 패러다임으로 통합합니다.

- **Performance Highlights**: Earth-OneVision은 다양한 벤치마크에서 경쟁력 있는 성능을 보여주며, 4B-72B RS-MLLM과 비교해 동등하거나 능가하는 결과를 기록합니다. 특히, OPT-RSVG 테스트셋에서 87.52%의 P@0.5를, SAR VQA 벤치마크인 SARLANG-Bench에서 80.68%의 점수를 달성하여 7B 모델을 7% 이상 초과합니다. 또한, BigEarthNet-MS 테스트셋에서는 75.74%의 리콜을 달성하며, EarthMind-Bench에서는 81.94%의 정확도를 기록하여 크로스 모달리티 추론에서의 우수성을 보여줍니다.



### Beyond APIs: Probing the Limits of MLLMs in Physical Tool Us (https://arxiv.org/abs/2606.10803)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 물리적 도구 사용 능력을 평가하기 위한 새로운 벤치마크인 PhysTool-Bench를 소개합니다. 물리적 작업에서 MLLMs가 도구를 인식하고 활용하는 데 있어 아직 탐구되지 않은 측면을 다루고 있습니다. PhysTool-Bench는 2,510개의 질의와 2,678개의 다양한 물리적 도구로 구성되어 있으며, 제조업, 전기 작업, 농업, 건강관리 등 여러 분야를 포함합니다.

- **Technical Details**: PhysTool-Bench는 물리적 도구 인식(Physical Tool Recognition)과 도구 선택 및 행동 계획(Tool Selection and Action Planning)이라는 두 가지 주요 작업으로 MLLMs의 성능을 평가합니다. 각 작업에서는 자연어 지침과 현실적인 환경 이미지를 결합하여 모델이 주어진 작업에 적합한 도구를 정렬하고 선택하는 과정을 돕습니다. 연구에서는 13개의 주요 MLLMs를 대상으로, 도구 인식에서 평균 58.7%를, 도구 선택 및 행동 계획에서는 평균 21.0%만 성공적으로 수행함을 보였습니다.

- **Performance Highlights**: 모델 성능 분석 결과, MLLMs는 현실적인 장면에서 도구를 인식하는 데 어려움을 겪고 있으며, 특히 계획 단계에서 큰 성능 저하를 보였습니다. 42-61%의 오류는 기능적으로 유사한 도구로 잘못 선택에서 발생하며, 이는 물리적 상식 부족이 주요 원인임을 시사합니다. 또한, 인간 고수준의 평가 결과는 평균 38%로, 최상의 MLLM이 21.0%에 그친 것과 비교해 많은 격차가 존재함을 보여주었습니다.



### Boosting ECG Classification Performance by Pre-training with Synthesized Data (https://arxiv.org/abs/2606.10802)
- **What's New**: 본 연구에서는 깊은 신경망(Deep Neural Networks, DNN)을 훈련하기 위해 도메인 특정 의학 지식에 기반한 합성 데이터를 사용하는 방법을 조사합니다. 특히, 단일 리드 II 심전도(ECG)를 위한 지식 기반 가우시안 조합(gaussian-composition) 합성 알고리즘을 개발하고, 이를 통해 심장 박동을 구성하는 여러 파형 요소를 생성합니다.

- **Technical Details**: 가우시안 조합 시뮬레이터는 P, Q, R, S 및 T 파형을 합성하여 단일 리드 II ECG 신호를 생성합니다. 이 시뮬레이터는 다섯 개의 가우시안 형태의 파형을 결합하여 심장 박동을 표현하고, 각 파형은 아말감(a) 및 폭(σ) 등의 매개변수로 조정합니다. 연구팀은 이 데이터를 사용하여 심장 리듬의 다양한 비정상 클래스를 합성하였습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터를 통한 훈련은 네 가지 비정상 ECG 클래스 중 세 개에 대해 분류 성능이 향상된다는 점을 입증하였습니다. 특히, 심방세동(AFLT)에서는 평균 33.2%의 성능 향상이 관찰되었습니다. 이 결과들은 합성 ECG 데이터가 실세계 데이터가 제한적인 상황에서 유용한 사전 훈련 자원으로 작용할 수 있음을 시사합니다.



### Dep-LLM: Training-Free Depression Diagnosis via Evidence-Guided Structured Multi-factor with Reliable LLM Reasoning (https://arxiv.org/abs/2606.10796)
- **What's New**: 본 논문은 임상 면담에서의 자동 우울증 탐지(ADD) 문제를 해결하기 위해 Dep-LLM이라는 새로운 훈련 없는 프레임워크를 제안합니다. 이 프레임워크는 임상 정신과 의사의 단계별 추론 과정을 모방하며, 기존의 LLM(Large Language Models)을 기반으로 작동합니다. 특히, 본 연구는 긴 대화에서 우울증 관련 단서를 구조적으로 분석하고 신뢰성을 평가하는 모듈을 구현하여, 데이터 부족과 높은 훈련 비용 문제를 동시에 해결하고자 합니다.

- **Technical Details**: Dep-LLM은 세 가지 단계로 구성된 파이프라인으로 진행됩니다. 첫 번째로, CoT(Chain-of-Thought) 다중 요인 분석 모듈은 긴 대화를 임상적 주제에 맞게 구조화하고, 증거에 기반한 이유를 생성하여 긴 맥락의 의존성을 효과적으로 처리합니다. 두 번째로, 신뢰 분석 및 조정 모듈이 각 이유의 신뢰성을 정량화하고 신뢰할 수 있는 신호를 강조하는 방식으로 모듈레이션을 적용합니다. 마지막으로, 협업 다중 요인 예측 모듈이 이러한 신뢰도 가중치가 적용된 신호를 최종 진단으로 통합합니다.

- **Performance Highlights**: DAIC-WOZ 및 E-DAIC 데이터셋에서의 실험 결과, Dep-LLM은 21가지 LLM의 제로샷 기준선보다 우수한 성능을 보여주었으며, 최신 감독 기반 도메인 특정 LLM들과도 비교해 뛰어난 성능을 발휘했습니다. 본 연구의 주요 기여는 훈련이 필요 없는 구조화된 다중 요인 분석과 LLM의 신뢰성 검증을 통해 임상 진단의 해석 가능성과 합리성을 강화한 점입니다.



### A Bayesian Network Approach for Enhancing Security-Focused Decision Support Systems (https://arxiv.org/abs/2606.10782)
- **What's New**: 이 논문은 다양한 오픈-소스 네트워크에서 이질적인 스택의 채택과 통합의 장점을 다루고 있습니다. 특히, 보안 접근 방식(예: 도구)을 선택하는 데 있어 인프라 운영자를 안내하는 의사결정 지원 시스템(Decision Support System, DSS)을 제안합니다. 이 DSS는 상위 요구사항을 쉽게 수집하고, 베이즈 네트워크(Bayesian Network, BN) 모델을 통해 필요한 보안 메커니즘을 제시하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 DSS는 CIA(기밀성, 무결성, 가용성) 삼각형의 각 보안 차원을 고려하면서 엔드 유저 요구 사항을 명확히 표현할 수 있는 단순하고 확장 가능한 문법을 제공합니다. 불연속적인 베이즈 네트워크(BN)를 더 쉽게 이해할 수 있도록 모델링하며, 지식 그래프를 통해 각 보안 도메인에 필요한 데이터를 해석하고 로드하는 프레임워크를 포함합니다. 이 시스템은 운영자의 보안 지식 격차를 줄이고, 적절한 보안 메커니즘을 제안할 수 있도록 합니다.

- **Performance Highlights**: 이 DSS의 성능은 시간과 예측 정확도 측면에서 평가됩니다. 이를 통해 운영자가 보안 요구 사항을 충족하기 위한 최적의 보안 도구를 찾는 데 소요되는 시간과 복잡성을 줄이는 데 기여할 수 있습니다. 또한, 다계층 베이즈 네트워크 기반 솔루션을 통해 통계적 추론을 할 수 있는 방안을 제시하였습니다.



### Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation (https://arxiv.org/abs/2606.10749)
- **What's New**: 이번 논문에서는 대화형 인터페이스에서 소프트웨어 구성 요소로 발전하는 대규모 언어 모델(LLM) 에이전트의 보안 문제를 다룹니다. LLM 에이전트는 도구를 사용하고, 메모리를 유지하며, 외부 환경에서 행동하는 능력을 지니고 있어 기존의 안전한 텍스트 생성와는 다른 새로운 보안 위험을 초래합니다. 저자는 247개의 연구를 종합하여 정보 흐름, 위임된 권한 및 지속 상태의 상호작용을 기반으로 한 생애 주기 모델을 제안합니다.

- **Technical Details**: 연구에 사용된 프레임워크는 입력, 계획, 의사결정, 도구 실행, 출력, 메모리, 모니터링 및 다중 에이전트 협정을 포함하는 상호 의존적인 LLM 에이전트 보안 단계를 연결합니다. 위협 표면과 공격 가족에 대한 문헌을 정리하고, 제안된 방어 메커니즘 및 그 무역의 비용을 비교합니다. 주요 보안 문제로는 프롬프트 삽입, 도구 매개 제어 흐름 탈취, 지속적인 상태 손상 및 다중 에이전트 전파가 있으며 이들은 서로 연관된 위험 패밀리를 형성합니다.

- **Performance Highlights**: 저자는 기존 방어 전략을 개입 지점, 신뢰 가정, 유용성 비용 및 구성 가능성 한계에 따라 비교하고 벤치마크 생태계를 평가합니다. 이 기술적 분석은 LLM 에이전트 보안의 진화하는 경향을 이해하는 데 기여하며, 현재와 미래의 위험에 대한 명확한 이해를 제공합니다. 또한, 논문에서는 안전한 LLM 에이전트를 구축하기 위해 신뢰 경계, 원칙적인 권한 제어 및 환경에 적합한 평가 관행이 필요하다고 주장합니다.



### Spatial-Omni: Spatial Audio Understanding Integration in Multimodal LLMs via FOA Encoding (https://arxiv.org/abs/2606.10738)
- **What's New**: Spatial-Omni는 기존의 Omni LLM에 First-Order Ambisonics(FOA) 공간 오디오를 독립적인 모달리티로 주입하는 경량화된 방법을 제안합니다. 이를 통해 공간 음향 인지를 개선하며, 기존의 오디오 인코더를 수정하지 않고 새로운 공간 인코더 SO-Encoder를 병행하여 사용합니다. 이 연구는 400K FOA 공간 오디오 클립과 2.1M 공간 질문 응답 쌍으로 구성된 SO-Dataset과 SO-QA 데이터셋을 제공합니다.

- **Technical Details**: SO-Encoder는 FOA 입력을 처리하며 방향, 거리, 움직임 등 공간 정보를 추출합니다. Temporal Pixel Shuffle Projector는 퍼즐처럼 이 공간 특징들을 압축하여 LLM이 오디오, 공간, 시각, 텍스트 토큰을 함께 인식할 수 있도록 합니다. 데이터와 평가 파이프라인도 구성되어 SO-Bench라는 공간 오디오 평가 벤치마크가 생성됩니다.

- **Performance Highlights**: 실험 결과, Spatial-Omni는 기존의 LALMs 및 Omni LLM 모델과 비교했을 때 공간 오디오 이해 작업에서 뛰어난 성능을 보였습니다. 이 연구는 다양한 공간 작업에서 강력한 개선을 이루었으며, 이 결과는 실제 공간 토큰에서 주로 오는 것으로 확인되었습니다. 전체적으로 Spatial-Omni는 기존의 일반적인 오디오 이해를 유지하면서 공간 이해를 강화하는 데 큰 기여를 합니다.



### Detecting Knowledge Gaps from Conversational AI Interactions Using Curriculum Prerequisite Graphs (https://arxiv.org/abs/2606.10736)
Comments:
          Accepted as a short paper at the 10th CSEDM Workshop, co-located with the 18th International Conference on Educational Data Mining (EDM 2026). 7 pages, 2 figures, 2 tables

- **What's New**: 이 논문에서는 대규모 온라인 강의에서 발생하는 수천 개의 학생 질문을 대화형 AI 튜터와 커리큘럼 주제에 효과적으로 매핑하는 방법을 제안합니다. 이를 위해 GPT-4에서 추출한 선수지식 그래프(prerequisite knowledge graph)와 few-shot text classifier를 활용하여 학생 질문을 분석합니다. 연구 결과, 이 방식이 수업 주제를 반영하는 진정한 지식 격차를 구체적으로 밝혀낼 수 있음을 보여주었습니다.

- **Technical Details**: 본 연구에서는 FastFit라는 few-shot 텍스트 분류기를 사용하여 대화형 AI 튜터에 의해 제출된 학생 질문을 커리큘럼 주제로 변환합니다. 1,340개의 질문 이벤트 데이터를 164명의 대학원생으로부터 수집하였으며, 43개 라벨(42개 커리큘럼 주제 및 '알 수 없음' 클래스)에 대해 80.0%의 정확도를 달성하였습니다. 또한, 질문 주제 수량이 학생들이 개별적으로 보고한 난이도와 유의미한 상관관계를 보였습니다.

- **Performance Highlights**: 이 연구의 주요 결과는 대화형 AI와의 상호작용 로그가 커리큘럼 구조에 매핑될 때, 주제 수준의 지식 격차에 대한 실질적인 신호를 전달할 수 있다는 것입니다. 분류 및 자가 보고된 난이도가 긍정적인 상관관계를 맺고 있어, 이 방식이 교육자에게 유용한 커리큘럼 근거의 개요를 제공할 수 있음을 나타냅니다. 또한, 큰 데이터셋에서 AI 학습의 품질을 높이기 위한 활용 가능성도 제시됩니다.



### Transformer Based Model for Spatiotemporal Feature Learning in EEG Emotion Recognition (https://arxiv.org/abs/2606.10718)
- **What's New**: EEG-TransNet는 EEG 신호의 시간적, 지역적 및 동기적 특성을 포착하기 위해 설계된 새로운 아키텍처입니다. 이 모델은 ResNet과 웨이브릿 기반 노이즈 제거를 활용한 전처리 및 특징 추출 모듈, 지역적 특징 학습을 위한 Local Self-Attention Block, 그리고 시공간적 의존성을 모델링하는 Fuzzy-Attention Synchronous Transformer (FAST)를 포함합니다. 이 연구는 세 가지 EEG 데이터셋에서 폭넓은 실험을 통해 EEG-TransNet의 분류 정확도와 강 robustness를 입증하였습니다.

- **Technical Details**: EEG-TransNet는 DWT(Discrete Wavelet Transform)를 사용하여 노이즈 제거 및 다중 밴드 특징을 추출합니다. 이 특징은 1D-CNN을 통해 정제되며, Local Self-Attention Block과 FAST를 통해 지역적 및 시간적 의존성을 학습합니다. 디코더에서는 깊이 분리 가능한 컨볼루션(deepwise separable convolutions)을 적용하여 효율적인 특징 디코딩과 분류를 수행합니다.

- **Performance Highlights**: EEG-TransNet는 BETA와 SEED 데이터셋에서의 실험을 통해 다양한 신호 길이에서 높은 분류 정확도를 달성하며, 특히 1.5초의 신호 길이에서 75%와 90%의 정확도를 기록했습니다. 다른 baseline 방법들과 비교할 때, EEG-TransNet는 더 낮은 표준 편차를 보여주며, 이는 다양한 피험자 간에도 안정적인 성능을 잘 유지함을 나타냅니다. 이러한 결과는 EEG-TransNet이 복잡한 EEG 신호 처리 및 감정 인식 작업에 매우 효과적임을 시사합니다.



### Attention Expansion: Enhancing Keyphrase Extraction from Long Documents with Attention-Augmented Contextualized Embeddings (https://arxiv.org/abs/2606.10716)
- **What's New**: 본 논문에서는 키프레이즈 추출(Keyphrase Extraction, KPE)에서 기존의 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)의 한계점을 극복하기 위해 주의 확장(attention expansion) 메커니즘을 제안합니다. 이 메커니즘은 주변의 문서 조각(out-of-context chunks)에서 가져온 정보를 통해 PLM의 토큰 표현을 보강하여 특히 긴 문서에서의 KPE 성능을 향상시킵니다. 이러한 접근 방식은 전통적인 KPE 방법들이 부족한 점을 보완하고, 효과적인 문맥을 제공하면서도 계산 비용을 줄이는 데 기여합니다.

- **Technical Details**: 제안된 주의 확장 메커니즘은 PLM이 생성한 표준 문맥화된 표현에 주변 조각들의 정보를 통합합니다. 각 주변 조각은 사전 훈련된 워드 임베딩(Pre-trained Word Embeddings, PWE)으로 표현되며, 인접한 토큰들이 이 표현을 쿼리하여 증강된 토큰 표현을 생성합니다. 이 메커니즘은 PLM의 컨텍스트 범위를 효과적으로 확장하면서도 전체 문서에 대한 주의(attraction)나 LLM(대형 언어 모델) 기반 추론(inference)을 필요로 하지 않기 때문에 계산 효율성이 높습니다.

- **Performance Highlights**: 실험 결과, 주의 확장은 5개의 PLM 백본(DistilBERT, SciBERT, KBIR, DeBERTa-v3, ModernBERT)에서 KPE 성능을 일관되게 향상시키며, 최신 모델보다 우수한 성능을 보였습니다. 주의 확장이 도메인 특화 모델 및 긴 문맥 인코더에 대해서도 이점을 제공하여 모델의 제한된 입력 길이를 보완하는 것 이상의 보충 정보를 제공함을 보여줍니다. 이러한 결과는 주의 확장이 긴 문서 KPE를 위한 효과적이고 효율적인 전략임을 입증합니다.



### ++nnU-Net: Scaling nnU-Net with Prefix-Based Data Augmentation (https://arxiv.org/abs/2606.10713)
Comments:
          7 pages, 1 figure, 2 tables

- **What's New**: 이번 논문에서는 nnU-Net의 연속적인 성공을 기반으로 한 새로운 데이터 증강 모듈인 ++nnU-Net을 제안합니다. 이 모듈은 이미지 등록(image registration)을 기반으로 하여 사전 처리 및 훈련 전 단계에서 작동합니다. 기존 nnU-Net에 비해 성능을 향상시키는데 중점을 두었으며, 다양한 2D 데이터 세트를 활용하여 평가했습니다.

- **Technical Details**: 이 시스템은 최대 8개의 인자를 수용하며, 이미지 및 세그멘테이션 데이터가 포함된 입력 디렉토리, 증강 후 최종 데이터 세트의 원하는 유형, 등록 및 변환 중 체크포인트를 저장할 옵션 등을 지원합니다. 기본 설정은 두 단계의 등록 과정으로 고정 및 변형 대칭 정규화(Symmetric Normalization, SyN) 변환을 포함합니다. 데이터 증강 파이프라인은 자동차 이식(AutoImplant) 2020 챌린지의 우승 솔루션을 적용하여 사용자의 기존 데이터를 변경하고 변형된 이미지를 생성합니다.

- **Performance Highlights**: ++nnU-Net은 nnU-Net의 기준 성능을 초과하여 Dice Similarity Coefficient 점수에서 약 22%의 성능 향상을 달성했습니다. 이러한 결과는 등록 기반 데이터 증강이 2D 의료 이미징 데이터 세트에서 특히 효과적임을 입증하며, 데이터가 제한된 환경에서도 세그멘테이션 성능을 향상시킬 수 있는 실용적이고 확장 가능한 접근 방식을 제공합니다.



### Effective Reinforcement Learning for Agentic Search by Recycling Zero-Variance Queries During Training (https://arxiv.org/abs/2606.10709)
- **What's New**: 본 논문은 GRPO 스타일 알고리즘을 활용한 LLM 검색 에이전트의 훈련에서 제로 분산(zero-variance) 그룹의 재활용(query recycling) 기법을 제안합니다. 이 연구는 정책(policy) 발전과 함께 쿼리가 제로 분산 상태와 신호를 지니는 상태(signal-bearing states) 간에 전환됨을 실증적으로 검증했습니다. 제로 분산 그룹을 동적으로 관리하여 훈련 분포가 정책과 함께 발전할 수 있도록 하는 접근 방식을 채택하고 있습니다.

- **Technical Details**: 제로 분산 그룹은 성공과 실패가 혼합된 롤아웃 그룹에서만 파라미터 업데이트에 기여하며, 너무 쉬운 그룹이나 너무 어려운 그룹은 신호를 제공하지 않습니다. 저자들은 동적 풀 관리(dynamic pool management)를 통해 각 훈련 단계에서 가중치가 부여된 쿼리를 샘플링하고, 신호를 발생시키지 않은 쿼리는 다시 샘플링이 가능하도록 만듭니다. 이를 통해 1.7B 파라미터 모델이 7개의 다중 단계 QA 벤치마크에서 평균 66.0의 Pass@1 성능을 달성했습니다.

- **Performance Highlights**: 재활용된 쿼리는 훈련 후반부에 효과적인 훈련 신호의 세 번째 주요 출처로 자리잡았으며, 약 75%의 수용된 그룹이 이전 제로 분산 영역에서 유래하였습니다. 저자들은 훈련이 전적으로 합성 데이터에서 진행되었음에도 불구하고, 1.7B 파라미터를 가진 Qwen3 모델이 기존 최대 7B 파라미터 시스템과 동등하거나 더 나은 성능을 나타내었음을 강조합니다.



### Unifying Data, Memory, and Compute Efficiency in LLM training: A Survey (https://arxiv.org/abs/2606.10706)
Comments:
          Accpeted for publication in IEEE Transactions on Artificial Intelligence (TAI)

- **What's New**: 이번 논문은 대규모 언어 모델(LLM)의 효율성을 자원 제약 관점에서 살펴보며, 데이터 효율성, 메모리 효율성, 계산 예산 의식이라는 세 가지 연결된 병목 현상을 중심으로 정리했습니다. 연구들은 LLM을 훈련하고 배포하는 데 필요한 자립 모델이 다양한 과제 객체와 자원 예산에 따라 다르다는 것을 보여줍니다. 또한, 메모리와 계산 처리 간의 상호 작용에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: LLM의 연구는 데이터를 선택하고 잘라내는 방법에서 훈련에 사용되는 GPU 메모리 및 FLOP가 아닌 자원 예산에 따른 혼합 전략까지 다양한 기술적 접근을 포함합니다. 이에 따라, 데이터 선택, 메모리 사용, 계산 분배를 통합적으로 고려하는 원칙으로 효율성을 재구성할 필요성이 강조됩니다. 이러한 복합 시스템에서 자원을 최적화하기 위한 흥미로운 접근법으로, 감시학적(gradient-based) 영향 추정이나 메모리 효율적 근사법을 통한 동적 데이터 선택도 제안됩니다.

- **Performance Highlights**: 기존 연구는 자원 비용을 줄이기 위한 다양한 기법들을 탐색했지만, 각 차원에서 최적화를 독립적으로 수립했고 이는 고립된 최적화를 제한하는 데 기여했습니다. 그러나 이 논문은 자원 제약 시스템에서의 결정을 종합하여, 효율적 성능을 최대화하는 방법을 모색합니다. 이를 통해 LLM의 훈련 및 배포 과정에서 메모리 절약 방식과 계산 예산을 효과적으로 통합하는 새로운 방향성을 제시합니다.



### Event-Driven Reinforcement Learning Enables Long-Horizon Control in Semiconductor Fabrication (https://arxiv.org/abs/2606.10705)
- **What's New**: 이 논문에서는 반도체 제조 시스템과 같은 대규모 시스템에서의 순차적 결정 최적화를 위한 강화 학습(reinforcement learning) 프레임워크를 제안합니다. 이 시스템은 높은 제약과 예측 불가능성이 많아 복잡한 의사결정 문제를 야기합니다. 이에 대응하기 위해 중앙 집중형 정책을 통해 시스템 전반의 결정을 조정하는 새로운 접근 방식을 도입하였습니다.

- **Technical Details**: 제안된 프레임워크는 이벤트 기반(event-driven) 시간 차이(temporal-difference) 공식을 개발하여, 다양한 정책 최적화 방법론과 통합할 수 있도록 구성을 하였습니다. 이는 복잡한 결정 문제를 모델 없이 해결할 수 있는 알고리즘으로 구성되어 있으며, 다양한 산업 운영 시나리오를 기반으로 한 고충실도(high-fidelity) 시뮬레이션을 통해 검증되었습니다. 또한, 이 프레임워크는 훈련 설정에 따라 효과적인 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, 오프라인(offline) 및 온라인(online) 환경에서 학습된 에이전트들은 생산성(throughput)과 활용도(utilization)에서 일관된 향상을 보여주었습니다. 대체 강화 학습 공식을 비교하고 성능 및 일반화를 평가함으로써 제안된 프레임워크의 스케일(scalability), 일반성(generality), 및 이전 가능성(transferability)에 대한 강력한 근거를 제공했습니다.



### Using the YOLOv12 Model for Verifying the Correct Color Sequence of Wires in Network Cables (Patch Cords) on the Production Lin (https://arxiv.org/abs/2606.10699)
- **What's New**: 이번 연구에서는 전통적인 시각적 검사 방법의 한계를 극복하기 위해 YOLO1(object detection model) 12번째 버전을 기반으로 한 지능형 시스템을 개발하였습니다. 이 시스템은 패치 코드의 와이어 위치를 식별하고 색상 순서를 검증하여 최종 제품의 성능을 보장합니다. 기존의 수작업 검사 방식 대신 자동화를 통해 인건비 절감과 생산성 향상을 목표로 하고 있습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 2,500장의 이미지를 포함하였으며, 이를 70%는 훈련, 15%는 검증, 15%는 테스트용으로 나누었습니다. 제안된 모델은 단일 단계 아키텍처(single-stage architecture)와 주의 메커니즘(attention mechanisms)을 활용하여 고도로 정밀한 와이어 감지를 이끌어냈습니다. 결과적으로 약 98%의 정밀도(precision)를 달성했으며, 전체 평균 정확도(mean accuracy)는 약 95%, 분류 정밀도(classification precision)는 99%, 재현율(recall)은 98%에 이릅니다.

- **Performance Highlights**: 이 연구의 결과는 개발된 시스템이 생산 라인에서 와이어 색상 순서의 정확성을 실시간으로 검증할 수 있음을 보여줍니다. 이 시스템은 사람의 개입 없이 자동으로 동작하여 인적 오류를 줄이고 제조 효율을 높입니다. 실제 적용 가능성 측면에서, 이러한 향상된 정확도와 효율성은 네트워크 케이블 생산 과정에서 큰 장점으로 작용할 것으로 기대됩니다.



### Divide and Cooperate: Role-Decomposed Multi-Agent LLM Training with Cross-Agent Learning Signals (https://arxiv.org/abs/2606.10684)
- **What's New**: 이 논문은 기존의 지식 집약적 질문 응답 시스템에서 발생하는 정책 공간의 조합적 폭발 문제를 해결하기 위해 DAC(Divide and Cooperate)라는 새로운 다중 에이전트 훈련 프레임워크를 제안합니다. 이 프레임워크는 증거 수집과 답변 생성을 두 개의 협력적 하위 작업으로 나누고, 각기 다른 역할에 맞춘 학습 신호로 훈련된 전담 에이전트에 의해 처리됩니다. 이를 통해 에이전트는 역할 간의 충돌을 줄이고 교육 중 신뢰할 수 있는 크레딧 할당 문제를 해결할 수 있습니다.

- **Technical Details**: DAC는 증거를 수집하고 답변을 생성하는 두 개의 에이전트로 구성된 역할 분해 프레임워크로, 에이전트는 각각 증거의 적합성을 확인하는 역할(생성기)과 탐색의 역할(탐색자)을 수행합니다. 생성기는 충분한 증거가 없을 경우 생성을 중단할 수 있는 선택지를 가지며, 이 신호는 탐색자의 보상에 통합되어 역할 전반에 걸쳐 학습 신호를 제공합니다. DAC는 LoRA 모듈을 사용하여 경량화된 파라미터 조정으로 구현되며, 공통의 백본을 공유합니다.

- **Performance Highlights**: 실험 결과, DAC는 일반 및 다단계 질문 응답(QA) 벤치마크에서 기존의 단일 에이전트 기본선과 비교하여 우수한 성능을 보였습니다. 더불어, DAC는 전체 모델을 미세 조정하지 않고도 뛰어난 결과를 달성해 효율성을 입증했습니다. 연구 결과는 협력적인 다중 에이전트 훈련의 잠재력을 강조하며, 각 에이전트가 역할에 따라 특화되어 동시에 최적화될 수 있음을 보여줍니다.



### UniDexTok: A Unified Dexterous Hand Tokenizer from Real Data (https://arxiv.org/abs/2606.10683)
- **What's New**: 이 논문에서는 다양한 구현에서 손의 정밀한 조작을 위한 통합 손 모델(UDHM)을 제안합니다. UDHM은 인간의 손과 로봇 손의 상태를 공유하는 22-DoF(도)의 의미적 인터페이스로 매핑합니다. 이를 통해 서로 다른 손 구성에서의 데이터 사용이 용이해지며, 모델의 성능이 대폭 향상됩니다.

- **Technical Details**: 이 연구에서 제안하는 UniDexTok는 흔히 사용된 손 특화 토크나이저와의 차별점으로 모든 구현에서 단일 인코더, 코드북 및 디코더를 공유합니다. 이 설계는 다양한 손 구조 간의 전이 가능성을 증진시키며, 새로운 손이 도입될 때도 별도의 학습 없이기존의 토큰 공간으로 투영할 수 있게 합니다. UDHM과 UniDexTok을 기반으로 한 새로운 학습 파이프라인은 여러 데이터셋에 걸쳐 실제 손 데이터를 표준화하여 학습합니다.

- **Performance Highlights**: UniDexTok은 UniHM 대비 MPJAE(Mean Per Joint Average Error)를 15.63도에서 0.16도로, MPJPE(Mean Per Joint Position Error)를 18.51mm에서 0.18mm로 감소시켰습니다. 이는 각각 98.98%와 99.03%의 오류 감소를 의미하며, 단위가 센티미터에서 서브 밀리미터 정확도로 향상된 것을 보여줍니다. 추가 실험 결과는 다른 구현에서의 데이터가 목표 구현의 재구성 정확도를 개선하는 데 기여함을 보여주었습니다.



### In Defense of Information Leakage in Concept-based Models (https://arxiv.org/abs/2606.10669)
Comments:
          Accepted as a position paper at the Forty-Third International Conference on Machine Learning (ICML 2026)

- **What's New**: 이번 논문에서는 Concept-based models (CMs)에서 정보 유출(leakage)이 부정적인 요소로만 간주되는 기존의 관점을 재조명합니다. 특히 이들은 실제 세계의 불완전한 개념 상황에서 필요한 '무해한 누수(benign leakage)'를 허용해야 한다고 주장합니다. 이는 CMs이 정확성과 중재 가능성을 유지하는 데 중요한 역할을 할 수 있음을 강조합니다.

- **Technical Details**: CM은 DNN(Deep Neural Networks)으로 인간이 이해할 수 있는 개념에 맞춘 표현을 기반으로 예측을 수행하는 모델입니다. 이 논문에서는 CMs의 정보 유출을 정의하고, 이를 분석하는 일반적인 프레임워크를 제시합니다. CBMs(Concept Bottleneck Models)를 통해 CMs의 구조를 설명하고, 개념 인코더와 레이블 예측기 간의 관계를 분석합니다.

- **Performance Highlights**: 저자들은 무해한 유출이 CMs의 핵심 요구 사항인 작업 신뢰도(task fidelity)와 중재 가능성(intervenability)과 양립할 수 있음을 보여줍니다. 일반적인 CMs 훈련 목표를 최적화함으로써, CMs는 이러한 유출을 활용하면서도 정확성과 중재성을 잃지 않을 수 있음을 입증합니다. 이는 CMs이 불완전한 현실적인 환경에서도 유용하게 사용될 수 있도록 합니다.



### Decentralized Multi-Agent Systems with Shared Contex (https://arxiv.org/abs/2606.10662)
- **What's New**: 이 논문은 다중 에이전트 시스템(MAS)을 기반으로 하는 새로운 접근 방식인 분산 언어 모델(DeLM)을 제안합니다. DeLM은 중앙 집중식 조정 대신, 에이전트들이 병렬로 작업을 수행하도록 하여 통신 및 통합의 병목 현상을 해결합니다. 이를 통해 에이전트 간에 유용한 진행 상황을 공유할 수 있게 하여 복잡한 문제를 효과적으로 처리할 수 있는 가능성을 열어줍니다.

- **Technical Details**: DeLM의 주요 구성 요소는 병렬 에이전트, 공유된 검증된 맥락, 그리고 작업 큐입니다. 각 에이전트는 비동기적으로 작업 큐에서 작업을 가져오고, 공유 맥락에서 쌓인 진행 상황을 읽어 들이며, 로컬 추론을 수행하고 검증된 업데이트를 다시 기록합니다. 이러한 디자인은 에이전트들 사이의 계속적인 의사소통을 효율적으로 줄여주어, 중복된 정보 처리를 방지합니다.

- **Performance Highlights**: DeLM은 SWE-bench Verified에서 77.4%의 Pass@4 성능을 달성하면서 작업당 비용을 약 50% 줄이는 성과를 보였습니다. 또한 LongBench-v2에서 여러 모델군 간의 평균 정확도를 높이며, 가장 강력한 기준선을 5.7% 포인트 개선했습니다. 그러므로 DeLM은 소프트웨어 공학 테스트 시간 스케일링과 긴 맥락 추론에서 모두 뛰어난 효과를 보여줍니다.



### Accounting for AI Inference in Corporate GHG Inventories: A Four-Tier Methodology for Scope 3 Category 1 Reporting (https://arxiv.org/abs/2606.10660)
Comments:
          Preprint. Data repository: this https URL. 18 pages, 3 figures, 6 tables

- **What's New**: AI 추론 서비스는 기업 지속 가능성 보고 지침(CSRD)의 범위 3 카테고리 1에 포함되며, 2024 회계 연도부터 공개가 의무화된다. 그러나 기업의 GHG(온실가스) 재고에 포함하는 표준화된 방법론이 부족해 현재 관행은 이 카테고리를 생략하거나 ICT 부문 전체를 기준으로 한 일반적인 경제적 입력-출력(EEIO) 계수를 적용한다. 이 연구에서는 데이터를 기반으로 한 새로운 정확한 추정 방법을 제안하여, 기업이 예측 가능한 자료를 수집할 수 있는 네 계층의 프레임워크를 제공한다.

- **Technical Details**: 이 논문은 AI 서비스 유형과 이용 가능한 데이터에 따라 적절한 추정 방법을 맵핑하는 네 계층의 의사결정 프레임워크를 제안한다. 이 구조는 GHG 프로토콜의 범위 3 요구 사항을 기반으로 하며, 각 계층에 따라 전용 에미션 팩터 테이블을 제공한다. 예를 들어, Tier 1은 서비스 사용 데이터가 없는 경우에 대한 지출 기반의 EEIO 대체 방법을 포함한다.

- **Performance Highlights**: 이 프레임워크는 200명의 직원이 있는 유럽 기업에 적용될 경우 tCO2e 1톤 미만의 총 배출을 나타내어, 순수한 배출량보다 방법론적 문제가 더 크다는 것을 보여준다. 또한 수자원-탄소의 트레이드오프도 문서화하며, 이는 현재 ESG 도구에서 간과되고 있는 현상이다. 예를 들어, 스웨덴의 수력 중심의 전력망이 가장 낮은 탄소 밀도를 보이지만, 수자원 footprint는 가장 높다는 점이 데이터 센터의 위치 전략에 직접적인 영향을 미친다.



### Post-Quantum Secure Federated DeFi for Inclusive Banking (https://arxiv.org/abs/2606.10658)
- **What's New**: 최근 역설계된 큐비트(quantum bits)의 발전은 실용적인 양자 컴퓨팅(quantum computing)의 타임라인을 가속화했습니다. 이는 금융 시스템, 정부 인프라, 통신 네트워크 및 DeFi(Decentralized Finance) 생태계를 보호하는 암호화 원칙에 위협을 가합니다. 본 논문은 제한된 신용 이력으로 인해 지역 대출자로부터 소외된 개인의 포용성을 향상시키기 위한 포스트 양자 보안(post-quantum secure) 연합형 DeFi 프레임워크를 소개합니다.

- **Technical Details**: 이 프레임워크는 여러 은행이 암호화된 정보 배치를 가상 서버에 기여하도록 하여, 격자 기반 완전 동형 암호화(Fully Homomorphic Encryption, FHE)를 통해 끝에서 끝까지 동형 계산을 가능하게 합니다. 서버는 NASA-IBM Prithvi 지리 정보 재단 모델(GFM)에 의해 생성된 확인 가능한 증거와 전문가의 신념을 결합하여, 지역 데이터 기반 확률 평가를 암호화된 형식으로 융합합니다. 분산 기술이 사용되어 모든 기관 간의 암호화된 데이터 교환에 대한 변조 방지 증거와 감사 가능성을 보장합니다.

- **Performance Highlights**: 이 프레임워크는 버지니아의 농업 대출 결정을 위해 테스트 되었습니다. 이를 통해 지역 커뮤니티에 대한 금융 접근성을 높이고, 금융 서비스를 필요로 하는 개인들에게 도움을 줄 수 있는 가능성이 있음을 보여주고 있습니다. 더불어 센서(censor)로부터의 안전한 데이터 공유와 감사를 위한 혁신적인 접근 방식이 제시되었습니다.



### Dynamic Linear Attention (https://arxiv.org/abs/2606.10650)
Comments:
          Accepted by ICML 2026

- **What's New**: 이 논문은 Dynamic Linear Attention (DLA)라는 동적 메모리 모델링 프레임워크를 제안합니다. DLA는 정적 상태 병합 정책 대신 정보 변화를 기반으로 상태 경계를 적응적으로 결정하여 의미 전환 주위의 고해상도 표현을 유지합니다. 이 접근 방식은 복잡한 토큰 중요도를 다루면서 메모리 성장을 제어할 수 있도록 설계되었습니다.

- **Technical Details**: DLA는 (i) 정보 인식 동적 상태 병합(Information-Aware Dynamic State Merging)과 (ii) 용량 제한 메모리 모델링(Capacity-Bounded Memory Modeling)을 통합합니다. 정보 인식 동적 상태 병합은 토큰 수준의 정보 변동에 따라 상태의 경계를 동적으로 결정하여, 메모리의 시간적 순서를 보존하면서 불필요한 정보를 최소화하는 병합을 수행합니다.

- **Performance Highlights**: DLA는 16개의 데이터 세트에 대한 평가에서 최신 멀티 상태 방법인 Log-Linear Attention을 지속적으로 초월하는 성능을 보였습니다. DLA는 Mamba-2와 같은 백본 모델을 사용할 때, 유사한 매개변수 예산에서 풀 어텐션 트랜스포머와 비슷한 성능을 달성했습니다. DLA는 Log-Linear Attention보다 높은 처리량과 낮은 런타임 메모리 소비를 통해 효율성을 극대화했습니다.



### Is Fairness Truly Fair? Towards Reliable Lipschitz Fairness in Multi-Task Learning via Fixed-\texorpdfstring{$δ$}{delta} Alignmen (https://arxiv.org/abs/2606.10632)
- **What's New**: 이 논문은 Lipschitz 스타일의 개별 공정성을 정의하고, 다중 과제 학습(Multi-Task Learning, MTL)에서의 평가가 방법에 의해 유도된 표현 스케일의 영향을 받을 수 있음을 설명합니다. 특히, 스레시홀드 혼란(threshold confounding) 현상을 밝히고, 이를 해결하기 위한 신뢰성 지향 프레임워크인 ReLiF를 제안합니다. ReLiF는 훈련 시 규제를 통제하고 고정된 참조 허용(tolerance)을 사용하여 공정성을 평가하는 방법을 비교 가능하게 합니다.

- **Technical Details**: MTL 시스템에서 각 알고리즘은 고유한 표현 스케일 및 허용값(δ)을 유도합니다. 이것이 스레시홀드 혼란을 일으켜 각 알고리즘의 공정성을 비교하는 것을 복잡하게 만듭니다. ReLiF는 고정된 δ 감사 방법과 훈련 시 규제를 분리하여, 공정한 평가를 위해 비교 가능한 참조 허용을 사용하며, 비율 피드백 제어기를 통해 Lipschitz 대리(Penalty)를 적절히 유지합니다.

- **Performance Highlights**: NYUv2(뉴욕대 깊이 V2) 및 임상 시간 시계열 벤치마크에서 ReLiF를 평가한 결과, 공유된 고정 허용값 하에서 정렬된 편향을 상당히 줄이면서도 경쟁력 있는 유틸리티를 달성했습니다. 임상 벤치마크에서는 조절된 공정성 규제라 할 수 있는 유틸리티-공정성 거래를 관찰할 수 있었습니다. 이러한 결과들은 고정된 δ 감사가 MTL에서 Lipschitz 공정성을 평가하는 데 있어 시맨틱적으로 일관된 프로토콜로 작용함을 지지합니다.



### STORM: Stepwise Token Optimization with Reward-Guided Beam Search (https://arxiv.org/abs/2606.10621)
- **What's New**: 본 논문에서는 STORM(Stepwise Token Optimization with Reward-guided beaM search)이라는 새로운 자가 지도 학습 프레임워크를 소개합니다. STORM은 BM25 인덱스를 기반으로 키워드 확장을 진행하고, 검색의 효과성을 높이기 위해 후보 확장을 평가하여 점수를 매깁니다. 이 방법은 특정 모델이 변경될 때마다 인덱스를 새로 작성할 필요 없이 사용자 쿼리를 개선할 수 있는 효율적인 방식입니다.

- **Technical Details**: STORM은 키워드 시퀀스를 생성할 때 각 생성을 단계적으로 평가하여 검색 지표에 의해 조정됩니다. 비효율적인 확장은 제거되며, 이는 검색-효과적인 어휘에 집중할 수 있도록 해줍니다. 이 과정은 Generative Cooperative Networks(GCN) 모델의 프레임워크를 변형하여, 기존의 학습된 판별기를 검색 보상으로 대체하여 수행됩니다.

- **Performance Highlights**: STORM은 TREC DL 및 BEIR 데이터베이스에서 테스트된 결과, 0.6B-8B 크기의 백본 모델들이 경쟁력을 유지하거나 기존 LLM 리라이팅 모델보다 우수한 성능을 보였습니다. 또한, STORM은 18개 언어로 제로샷 전이(transfer) 성능을 갖추고 있으며, 전통적인 멀티링궐 밀집 검색기보다 평균적으로 더 나은 성능을 보였습니다.



### Can Image Models Imagine Time? ImageTime: A Novel Benchmark for Probing Visual World Modeling Through Spatiotemporal Consistency (https://arxiv.org/abs/2606.10620)
- **What's New**: 이번 논문은 이미지 생성 모델이 시간에 따라 어떻게 시각 세계를 변화시키는지를 이해하는 데 한계를 드러냅니다. 기존의 평가 방식 대부분은 단일 이미지의 정확성에 초점을 맞추고 있지만, ImageTime은 이러한 한계를 넘어서 다중 시각 상태에서의 일관성을 유지할 수 있는지를 평가하는 새로운 진단 기준을 제시합니다. 이는 단순한 이미지 생성을 넘어 실제의 변화 과정을 반영할 수 있는지를 점검하는 중요한 단계로 볼 수 있습니다.

- **Technical Details**: ImageTime은 특정 행동 지시와 선택적으로 초기 상태를 정의하는 참조 이미지에 따라 네 가지 시간 순서가 매겨진 주요 상태(초기 상태, 동작 시작, 전이 상태, 최종 상태)를 포함하는 이미지를 생성해야 합니다. 750개의 벤치마크 사례와 22개의 도메인, 375개의 행동 개념을 포함하며, 이를 통해 이미지 생성 모델이 시간에 일관된 시각 세계를 유지할 수 있는지를 평가합니다. 또한, L0에서 L6까지의 7단계 능력 트리를 설계하여 각 단계를 면밀하게 평가할 수 있도록 하였습니다.

- **Performance Highlights**: ImageTime을 사용하여 다양한 이미지 생성 모델들의 성능을 평가한 결과, 각 모델들이 얼마나 일관된 시각 상태를 유지할 수 있는지를 구체적으로 분석했습니다. 평가에서 GPT-5.5가 생성된 이미지에 대해 구조적 VLM-as-judge 프로토콜을 통해 점수를 매기며, 이를 통해 반복적으로 발생하는 실패 모드를 식별할 수 있었습니다. 이 연구는 현재 이미지 생성 시스템의 강점과 약점을 파악하는 데 기여하며, 향후 더 진화된 시각 세계 모델링으로 나아가는 데 중요한 역할을 할 것입니다.



### Fast and Highly Expressive Policy Learning for Offline Reinforcement Learning via Bootstrapped Flow Q-Learning (https://arxiv.org/abs/2606.10613)
Comments:
          ICML 2026, 19 pages

- **What's New**: 본 논문에서는 Bootstrapped Flow Q-Learning (BFQ)라는 새로운 오프라인 강화 학습 (Offline Reinforcement Learning) 프레임워크를 소개합니다. BFQ는 보조 네트워크나 모델 증류 과정 없이 훈련 및 추론 중에 정확한 단일 단계 액션 생성을 가능하게 합니다. 이 방식은 다단계 감쇠를 제거하여 학습 절차를 significantly faster (상당히 빠름), simpler (더 간단함), 그리고 더 robust (더 강력함)하게 만듭니다.

- **Technical Details**: BFQ는 Flow Matching을 기반으로 하며, 흐름 경로를 따라 변위 벡터에 대한 분할 정복 접근 방식을 취합니다. 이 과정에서 짧은 범위의 변위를 정확하게 추정하고 이를 통해 단일 단계에서 노이즈-액션 매핑을 직접 학습합니다. BFQ는 BPTT(backpropagation through time)를 사용할 필요가 없으며, 모든 규모에서 흐름 동역학에 대한 속도를 고정하는 단일 공유 정책 네트워크를 훈련 합니다.

- **Performance Highlights**: D4RL 기준을 이용한 실험에서 BFQ는 DQL과 비교하여 성능을 향상시키며, 훈련 및 추론 효율성에서도 향상이 있음을 보여주었습니다. BFQ는 간소화된 학습 파이프라인을 유지하면서 반응 시간과 액션 빈도를 크게 개선했습니다. 결론적으로, BFQ는 D4RL에서 빠르고 최첨단의 방식으로 꾸준히 강력한 성능을 발휘합니다.



### Causal Ensemble Agent: Hierarchical Causal Discovery with LLM-guided Expert Reweighting (https://arxiv.org/abs/2606.10607)
- **What's New**: 이 논문에서는 기존의 인과 탐색 방법을 개선하기 위해 Causal Ensemble Agent(CEA)라는 새로운 프레임워크를 제안합니다. CEA는 여러 통계 전문가로부터 구조적 통찰력을 집약하고, LLM을 메타 심판자(meta-referee)로 사용하여 불확실성에 따라 전문가의 가중치를 동적으로 조정합니다. 이러한 방법은 인과 그래프의 정확성과 완전성을 더욱 향상시키는 것으로 나타났습니다.

- **Technical Details**: CEA는 계층적 앙상블 프레임워크로, 여러 인과 탐색 방법의 출력을 통합하여 그래프의 뼈대(skeleton), v-구조(v-structures), 엣지 방향(edge orientations)를 차례로 집약합니다. 각 계층에서 부트스트랩 기반의 신뢰도 점수를 사용하여 전문가의 신뢰성을 평가하고, LLM이 불확실한 관계에 대해 가중치를 조정하여 인과 그래프를 구성합니다. 이러한 방식으로 CEA는 직접적인 엣지 예측을 피하고 데이터에 지향적인 결정을 유지합니다.

- **Performance Highlights**: CEA는 8개의 합성 및 실제 데이터셋 벤치마크에서 모든 인과 탐색 방법 중 가장 뛰어난 성능을 나타냈습니다. 실험 결과, CEA는 인과 탐색의 정확성과 다양한 도메인에서의 강건성을 크게 개선하며, 기존 방법보다 LLM 쿼리를 현저히 줄입니다. 이러한 성과는 LLM이 메타분석에서 어떻게 효과적으로 활용될 수 있는지를 잘 보여줍니다.



### Dmsh: A Multi-Agent Reinforcement Learning Framework for All-Quad Mesh Generation (https://arxiv.org/abs/2606.10601)
- **What's New**: 이 논문은 자동화된 강화 학습 파이프라인인 Dmsh를 소개하고 있습니다. Dmsh는 기하학적 분해와 사각형 메쉬 생성을 단일 학습 기반 프레임워크 내에서 통합합니다. 이 프로세스는 세 개의 조정된 에이전트가 작업을 수행하여 더욱 효과적인 메쉬 생성을 가능하게 합니다.

- **Technical Details**: Dmsh는 Markov Decision Process로 메쉬 생성 문제를 공식화하고, Parametric Soft Actor-Critic 아키텍처를 활용해 하이브리드 이산-연속 행동 공간을 탐색합니다. 이 과정에서 커리큘럼 학습 전략을 채택하여 단순한 도메인에서 복잡한 기하학으로의 확장을 용이하게 합니다. 각 하위 영역은 독립적으로 메쉬 처리가 이루어지고, 최종적으로 모든 사각형 메쉬를 생성합니다.

- **Performance Highlights**: Dmsh는 기존 방법들에 비해 높은 자동화, 강인성, 그리고 메쉬 품질을 일관되게 선보이며 새로운 패러다임을 제시합니다. 다양한 벤치마크 결과에서 Dmsh는 메쉬 생성 품질과 효율성 모두에서 기존의 상용 소프트웨어와 경쟁할 수 있는 성능을 보여주었습니다. 이러한 결과는 Dmsh의 성능이 복잡한 기하학적 도메인에서도 뛰어난 효율성을 가지는 것을 입증합니다.



### Embedding Hybrid Systems into Continuous Latent Vector Fields (https://arxiv.org/abs/2606.10596)
Comments:
          Accepted to ICML 2026

- **What's New**: 이 연구에서는 n차원 하이브리드 시스템이 m>2n일 때 연속 벡터 필드가 구비된 m차원 유클리드 공간에 임베드될 수 있음을 증명합니다. 이 결과는 본질적으로 불연속적인 하이브리드 시스템이 미분 가능한 최적화를 위하여 잘 정의된 연속적 외적 표현을 갖는다는 것을 암시합니다. 이러한 존재 정리를 바탕으로, 우리는 잠재적 Neural ODE(Ordinary Differential Equation) 알고리즘이 하이브리드 시스템의 흐름을 정확히 복원할 수 있음을 보여줍니다.

- **Technical Details**: 하이브리드 시스템은 연속적인 시간 벡터 필드와 이산 상태 리셋을 결합하여 물리적 및 사이버-물리적 프로세스를 모델링합니다. 전통적인 방법은 궤적을 여러 세그먼트로 나누고 각 모드의 역학을 학습하는데, 이는 조합적으로 복잡한 모드 선택을 요구합니다. 이와 대조적으로, 본 연구는 연속적인 흐름으로 하이브리드 역학을 표현함으로써 더 나은 접근 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 단지 시계열 데이터로 다양한 기하학을 가진 하이브리드 시스템을 학습하는 데 있어 기존 방법보다 우수한 성능을 보였습니다. 특히, 잠재적 및 관측 공간에서의 일관성 손실이 하이브리드 시스템의 흐름을 정확히 복원하는 데 핵심적인 역할을 한다는 것을 밝혔습니다. 또한, 알고리즘의 구현은 GitHub에서 가능합니다.



### From Data Heterogeneity to Convergence: A Data-Centric Review of Federated Learning (https://arxiv.org/abs/2606.10595)
- **What's New**: 최근 연구에서는 Federated Learning(FL)이 중앙 집중식 학습의 데이터 부족 문제를 해결하기 위한 유망한 솔루션으로 부각되고 있습니다. FL은 여러 클라이언트가 개인 데이터를 노출하지 않고도 협업하여 공유 모델을 학습할 수 있도록 하여, 데이터의 프라이버시를 보호합니다. 본 논문은 데이터의 관점에서 FL의 다양한 문제를 다루며, 특히 데이터 특성, 분할 프로토콜 및 방어 메커니즘이 수렴 속도와 안정성에 미치는 영향을 분석합니다.

- **Technical Details**: 이 논문은 비독립 분포(non-IID)의 특성을 측정 가능한 형태로 분석하고, 각 특성이 수렴에 미치는 영향을 강도에 따라 분류합니다. 데이터 분할 관행과 실제 현상을 연결하여 발생할 수 있는 아티팩트를 노출하고, 이러한 아티팩트가 모델의 정확도에 미치는 영향을 보여줍니다. 또한, 데이터 관련 취약점과 방어 방식이 수렴성에 미치는 영향을 분석하여, 청정 및 공격 조건에서의 성능을 보고합니다.

- **Performance Highlights**: FL의 실용적인 성능은 데이터의 특성과 클라이언트 간 데이터 분할 방식에 많은 영향을 받습니다. 본 조사 연구는 FL의 데이터 관련 도전과제를 완전히 이해할 수 있도록 도와주며, 각 문제에 대한 구체적인 해결책을 제시합니다. 이러한 발견은 향후 강력하고 포괄적인 학습 시스템을 위한 설계에 실질적인 가이드를 제공합니다.



### Towards Diverse Scientific Hypothesis Search with Large Language Models (https://arxiv.org/abs/2606.10587)
Comments:
          ICML 2026

- **What's New**: 이 논문에서는 과학적 가설 검색을 샘플링 문제로 재구성하고, 고정된 검증 예산에서 다양하고 고품질의 가설을 효율적으로 생성하는 것을 목표로 합니다. 그것을 위해 고전적인 병렬 기온 조절 알고리즘에 영감을 받아 EvoDiverse라는 진화 프레임워크를 제안합니다. 이 프레임워크는 다양한 온도 수준에서 가설을 검색하고, 온도 간의 정보 교환을 통해 탐색을 향상시키면서 수렴을 방해하지 않습니다.

- **Technical Details**: EvoDiverse는 두 개 이상의 온도 수준에서 진화 검색을 실행하고, 높은 온도에서 더 넓은 가설 공간을 탐색하도록 유도하며, 낮은 온도에서는 더 정교한 선택을 통해 잠재적으로 유망한 가설을 정제합니다. 이 접근 방식은 표준 LLM 기반의 진화 검색에서 발생하는 체계적인 다양성 붕괴를 방지하고, 유의미한 대체 가설을 생성할 수 있습니다. 이 논문은 샘플링입니다: 고온에서 낮은 확률 밀도가 높은 뭉치성을 피해 탐색과 선택의 균형을 맞춥니다.

- **Performance Highlights**: EvoDiverse는 단일 검증 예산 아래에서 가설의 질과 다양성을 지속적으로 향상시킵니다. 분자 발견, 방정식 발견, 알고리즘 발견 등 다양한 분야에서 이 방법은 신뢰성 있는 후보 집합을 생성하며, 더 높은 비용의 후속 계산 검증에서도 우수한 성능을 보여줍니다. 이로 인해 다수의 과학적 발견 문제에 있어 더 나은 결과를 도출할 수 있는 가능성을 제시합니다.



### NOVA: Symbolic Regression Discovery of Interpretable Car-Following and Lane-Change Models with Driver Heterogeneity (https://arxiv.org/abs/2606.10583)
- **What's New**: 이 논문에서는 NOVA라는 자율 심볼릭 회귀(framework) 프레임워크를 제시하여 원시 궤적 데이터에서 해석 가능한 자동차 추종 및 차선 변경 구조를 식별합니다. NGSIM I-80 및 US-101 데이터셋의 4,765,788개의 주행 관측 데이터에 적용하여, 복잡한 가정 없이도 두 항으로 구성된 간결한 가속도 모델을 찾았습니다. NOVA는 10,000개 이상의 후보 대수 구조를 평가하고, 기존의 수정된 상징 회귀(Recalibrated Symbolic Regression) 기준보다 더 높은 성능을 보여줍니다.

- **Technical Details**: NOVA는 차량 추종과 차선 변경 행동에서 저복잡도의 심볼릭 구조가 존재한다고 가정합니다. 이 구조는 a(t)=f(Δv,v,gap,…) 형태의 대수식으로 표현되어, 차량의 가속도(a(t))와 추종 차량의 상대 속도(Δv), 그리고 차량 간 간격(gap) 등의 변수 간의 관계를 설명합니다. NOVA는 차량 데이터에 대해 엄격한 평가 프로토콜을 적용하여, 신뢰성을 높이고, 선행 모델들과 비교해 향상된 성능을 보입니다.

- **Performance Highlights**: NOVA는 차선 변경 모델링에서 67.4%의 균형 정확도를 달성하였으며, 이는 기존의 MOBIL 차선 변경 기준보다 29.8 포인트 높은 성능입니다. 이 연구는 502명의 관찰되지 않은 드라이버에 대해 테스트 되었고, 기존의 방법보다 유의미한 개선을 입증하였습니다. 또한, NOVA는 freeway 사이트 간에도 3 pp 이하의 R² 손실로 강력한 제로샷 전이를 보여줍니다.



### Drawing with Strangers: Population Scaling Drives Zero-Shot Mutual Intelligibility in Emergent Sketching (https://arxiv.org/abs/2606.10582)
- **What's New**: 이 논문에서는 독립적으로 훈련된 AI 집단 사이의 성공적인 소통을 설명하는 새로운 개념인 '제로샷 상호 이해 가능성(zero-shot mutual intelligibility, ZMI)'을 도입합니다. 이 연구는 에머전트 커뮤니케이션(emergent communication) 분야에서 주목받지 못했던 점, 즉 서로 완전히 다른 커뮤니티에 속한 에이전트 간의 소통 가능성을 탐구합니다. 저자들은 에머전트 스케칭(emergent sketching)이라는 시각적으로 기반한 방법론을 활용하여 통신의 질을 높이는 데 기여하고 있습니다.

- **Technical Details**: 메소드 섹션에서는 스케치 기반 커뮤니케이션과 그룹 간 일반화의 출현을 연구하기 위해 사용된 방법론적 프레임워크를 설명합니다. 레퍼런셜 게임(referential game)과 그 차별화된 스케치 기반 변형을 소개하며, 제로샷 상호 이해 가능성(zero-shot mutual intelligibility, ZMI) 지표를 통해 그룹 간 커뮤니케이션 설정에서 일반화 가능성을 평가합니다. 이 프레임워크는 커뮤니케이션 프로토콜의 출현을 연구하기 위해 상호 작용하는 에이전트들 간의 통신을 분석하는 데 사용됩니다.

- **Performance Highlights**: 연구 결과, 집단의 규모가 증가함에 따라 ZMI가 뚜렷하게 향상되는 경향을 발견했습니다. 특히, 높은 ZMI는 집단 내 다양성을 증가시키고 집단 간 변화를 감소시킵니다. 이러한 특성은 독립적으로 훈련된 그룹들이 임의적인 차이를 버리고 보편적으로 공유된 지각 기반 규범으로 수렴하게끔 합니다. 이러한 발견은 에머전트 커뮤니케이션에서의 상호 작용의 중요성을 강조하며, 사회적으로 상호 운용 가능한 인공지능 에이전트의 발전을 위한 길을 제시합니다.



### Convergence of Monte Carlo Optimistic Policy Iteration: Beyond Uniform State-Action Updates (https://arxiv.org/abs/2606.10580)
- **What's New**: 이 논문은 Monte Carlo 상황에서 낙관적인 정책 반복(MC-O-PI)의 비대칭적 성질을 새롭게 조명합니다. 기존에는 모든 상태-행동 쌍이 동일한 빈도로 업데이트 되어야만 최적성에 수렴한다는 조건이 있었으나, 이 연구는 이를 완화하여 상태 내 행동이 균일하게 업데이트될 경우에도 수렴 가능하다고 증명하였습니다. 이는 공정한 시작지점 없이도 다양한 시작 상태에서 에피소드를 시뮬레이션할 수 있게 해 실제적인 응용 가능성을 높입니다.

- **Technical Details**: 연구는 Markov Decision Process (MDP)에 대한 최적 정책의 학습을 위해 '입회적 방문'(initial-visit) MC-O-PI 접근법을 채택합니다. 이 접근법은 상태 내 행동들이 균일하게 업데이트되는 경우에도 최적 행동 가치로 수렴하는 것을 보여줍니다. 이 결과는 기존의 증명 기법에서 벗어나 정책의 진화를 추적하며, 특정 확률 변화가 이 개선을 방해하지 않음을 증명합니다.

- **Performance Highlights**: 논문에서 제안된 새로운 방법이 MC-O-PI의 비대칭적 행동을 연구하는 혁신적인 경로를 제시하고 있습니다. 이는 제안된 정책 반복 알고리즘이 임의의 주기에서 상태를 업데이트 할 수 있는 가능성을 열리며, 대규모 상태 공간 또는 알려지지 않은 환경에서도 효율적인 정책 최적화를 가능하게 합니다. 실질적인 적용 가능성과 관련하여, 향후 최적화 방법의 확장 가능성에 대한 논의가 이루어집니다.



### Improving Adversarial Transferability on Vision-Language Pre-training Models via Surrogate-Specific Bias Correction (https://arxiv.org/abs/2606.10571)
Comments:
          17 pages, 7 figures, 10 tables

- **What's New**: 이 논문은 Vision-Language Pre-training (VLP) 모델에서의 적대적(Adversarial) 예제가 드러내는 취약점과 이를 개선하기 위한 새로운 접근법인 DeBias-Attack을 제안합니다. 본 연구는 적대적 최적화에서 발생하는 서그릿(Surrogate) 모델의 의존성을 줄이는 데 중점을 둡니다. 이를 통해 DeBias-Attack은 다양한 VLP 모델에 대해 더욱 강력한 전이(Transfer) 공격 성능을 보여줍니다.

- **Technical Details**: DeBias-Attack은 두 가지 섭동(Perturbation) 분기를 유지합니다. 주 분기는 원본 이미지에서 섭동을 최적화하고 이미지-텍스트 정합성을 방해하는 적대적 경량화를 생성합니다. 보조 분리는 매 반복마다 작게 재샘플링된 가우시안 노이즈를 추가하여 생성된 약한 의미의 이미지에서 최적화된 섭동을 수행합니다. 이러한 방식으로, DeBias-Attack은 서그릿 모델의 응답보다 이미지 의미론(Semantics)에 더 영향을 주어 전이 성능을 개선합니다.

- **Performance Highlights**: 다양한 VLP 모델, 다운스트림 작업, 및 멀티모달 대규모 언어 모델(Multimodal Large Language Models, MLLM)에서 DeBias-Attack은 강력한 성능을 달성했습니다. 특히 블랙박스(Black-box) 전이 설정에서, DeBias-Attack은 이질적(heterogeneous) 전이 시나리오에서도 경쟁력 있는 공격 성능을 보이고 있습니다. 이러한 결과는 편향 기반의 경량화 개선이 생성된 적대적 이미지-텍스트 쌍의 크로스 모델 전이 가능성을 강화함을 보여줍니다.



### Hidden Consensus:Preference-Validity Compression in Human Feedback (https://arxiv.org/abs/2606.10569)
Comments:
          28 pages. When AI learns from human feedback, it forces a single "correct" answer, but sometimes multiple answers are all genuinely valid, and that nuance gets thrown away

- **What's New**: 이번 연구에서는 Reinforcement Learning from Human Feedback (RLHF) 파이프라인에서 인식된 편향 문제를 제기하고 있습니다. 구체적으로, 다원적 해석이 있는 사회에서 인간의 판단을 간단한 스칼라 보상 신호로 감소시키는 과정이 "Preference-Validity Compression"이라 불리는 문제를 일으킬 수 있습니다. 이 연구의 초점은 다문화적 관점에서 허용 가능한 다수 옵션을 단일 최적화 목표로 축소하는 과정의 부정적 영향을 살펴보는 것입니다.

- **Technical Details**: 이 연구에서는 20명의 말레이시아 참가자들로부터 수집된 321개의 선호 이벤트를 분석하여 다원적 수용 가능성이 실제로 존재한다는 것을 보여줍니다. 구체적으로, 79%의 프롬프트에서 한 명 이상의 응답이 다수의 수용 임계값에 도달했지만, argmax 집계 방식에 의해 버려지거나 무시됩니다. 이러한 현상은 참가자들이 종종 여러 답변을 동시에 유효한 것으로 간주한다는 것을 시사합니다.

- **Performance Highlights**: 이 연구의 실증 결과는 RLHF 스타일 피드백 집계에서의 측정 실패를 수치적으로 나타냅니다. 단기 승자 집계는 우세한 응답의 우위를 과장하게 되며, 다수의 지지받는 응답을 고려할 경우 그 격차가 상당히 줄어들 수 있습니다. 이러한 발견들은 집계 프로토콜이 다원적 해석을 단일 보상 목표로 축소시키는 경우 측정 유효성을 좌우하는 주요 병목현상이 될 수 있음을 보여줍니다.



### Benchmarking Knowledge Editing using Logical Rules (https://arxiv.org/abs/2606.10554)
Comments:
          Accepted at the 24th International Semantic Web Conference 2025

- **What's New**: 본 논문은 대형 언어 모델(Large Language Models, LLMs)에서의 지식 편집(knowledge editing) 방법의 성능을 개선하기 위한 새로운 벤치마크를 제안합니다. 이는 단순히 편집된 사실을 기억하는 것을 넘어, 그 논리적 결과를 평가하는 데 중점을 둡니다. 기존의 벤치마크가 이러한 논리적 결과를 간과해 왔다는 점에서, 연구팀은 지식 그래프에서 관련 논리 규칙을 추출하여 다중 단계 질문(multi-hop questions)을 생성하는 방식을 도입하였습니다.

- **Technical Details**: 이 방법론은 AMIE3라는 규칙 추출 시스템을 사용하여, 편집된 사실과 관련된 논리 규칙을 자동으로 생성합니다. 생성된 질문은 LLM이 편집된 사실을 통해 관련 지식을 일관되게 유지하는지 평가하는 데 사용됩니다. 다양한 지식 편집 방법을 테스트하여 직접 편집된 사실과 논리적 결과 간의 성능 차이를 분석하는 것이 본 연구의 핵심입니다.

- **Performance Highlights**: 기존의 지식 편집 방법들은 직접적인 편집에서는 높은 정확성을 보였으나, 논리적 결과를 필요로 하는 질문에서는 최대 24%의 성능 차이를 보였습니다. 이는 지식 편집 방법들이 상호 연관된 정보의 일관성을 유지하는 데에서 부족함이 있음을 보여줍니다. 따라서, 논문은 실질적인 시나리오에서 지식의 상호 의존성을 처리하는 더욱 강력한 평가 프레임워크의 필요성을 강조합니다.



### Flexible Flows for Biological Sequence Design (https://arxiv.org/abs/2606.10543)
- **What's New**: 이번 연구에서는 기존의 Biological sequence generation 방법인 Discrete Flow Matching (DFM)의 한계를 보완하기 위한 새로운 접근 방식을 제안합니다. FlexFlow는 생물학적으로 유의미한 구조적 결합(structured coupling)을 도입하며, 가변 길이 생성(variable-length generation)을 지원하는 편집 기반(parameterization) 방법을 사용합니다. 이는 기존의 방법들이 제공하지 못했던 유연성과 정밀한 제어를 가능하게 합니다.

- **Technical Details**: FlexFlow는 연속 마르코프 체인(CTMC)을 활용하여 소스 분포와 목표 분포 사이의 전이를 학습하는 DFM의 프레임워크에 기반합니다. 새로운 구조적 결합은 생물학적 특성을 반영하여 생성 경로를 조정하며, 변수 길이 생성은 편집 작업(edit operations)을 통해 이루어집니다. 추가적으로, 계속적인 잠재 공간에서 작동하는 잠재 지침(latent guidance) 메커니즘이 도입되어 더 나은 성능을 도출합니다.

- **Performance Highlights**: FlexFlow는 다양한 생물학적 시퀀스 생성 작업에서 기존의 확산(diffusion) 및 흐름(flow) 모델보다 뛰어난 성능을 보여주었습니다. 구체적으로, DNA 시퀀스의 비조건적 및 조건적 생성뿐만 아니라 펩타이드(péptide) 생성에서도 최신 기술의 성과(state-of-the-art performance)를 달성하고 있습니다. 연구자들은 또한 새로운 펩타이드-MHC II 벤치마크를 소개하여 MHC 조건 하에서의 펩타이드 시퀀스 생성 및 평가에 기여하였습니다.



### LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization (https://arxiv.org/abs/2606.10531)
Comments:
          Accepted by ICML 2026

- **What's New**: 본 논문에서는 2비트 가중치만 사용하는 벡터 양자화 인식 훈련 프레임워크인 LC-QAT를 제안합니다. 이는 학습된 아핀 맵을 통해 양자화된 가중치를 표현하여 고품질의 사후 훈련 초기화를 제공하며, 코드북 조회 없이 엔드-투-엔드 최적화를 가능하게 합니다. 실험 결과, LC-QAT는 기존의 최첨단 양자화 인식 훈련 방식보다 더 우수한 성능을 보입니다.

- **Technical Details**: LC-QAT는 비선형 제약 조건을 갖는 코드북을 활용하여, 각 가중치 그룹의 표현성을 높이는 구조를 취합니다. 이 방식은 각 코드워드를 공유하는 선형 변환을 통해 생성되며, 가중치 행렬을 통해 그래디언트가 양자화 과정을 통해 전파될 수 있도록 합니다. 따라서, 기존의 이웃 탐색 없이도 간단한 반올림 및 클램핑 연산으로 양자화가 가능합니다.

- **Performance Highlights**: 다양한 대규모 LLM을 대상으로 한 실험 결과, LC-QAT는 훈련 데이터의 오직 0.1%에서 10%만 사용하여도 뛰어난 성능을 발휘했습니다. 초저비트 모델 배포를 위한 실용적이고 확장 가능한 솔루션으로 자리 잡으며, 기존 SQ-QAT 및 VQ-QAT 방법에 비해 데이터 효율성과 정확도 모두에서 향상된 결과를 보여줍니다.



### Machine Learning Methods for Studying Latent Neural Activity Dynamics (https://arxiv.org/abs/2606.10530)
Comments:
          Accepted by IJCAI 2026 survey track

- **What's New**: 최근 뇌 기록 기술이 발전하면서 대량의 뉴런 집단의 잠재 구조를 해독할 수 있는 머신 러닝 도구에 대한 수요가 증가하고 있습니다. 이 논문에서는 초기 상태 공간 모델(state-space models)에서 최근의 딥 생성 모델(deep generative models)까지의 잠재 변수 모델(Latent Variable Models, LVMs)의 발전 과정을 포괄적으로 조사합니다. 우리는 문헌을 단일 지역 잠재 역학(Single-Region Latent Dynamics), 다중 지역 소통(Multi-Region Communication), 행동 정렬 모델링(Behavior-Aligned Modeling)의 세 가지 분야로 조직하여 연구합니다.

- **Technical Details**: 이 논문에서는 다양한 잠재 변수 접근 방식을 비교하고, 뇌 집단 분석을 생성 프로세스로 모델링하여 방법의 분류를 제시합니다. 네트워크 연결성과 시냅스 전파 지연을 고려하여 정보를 연구하는 다중 지역 소통은 확률적 방법과 서브스페이스(subspace) 방법을 사용합니다. 행동 정렬 모델링은 감독 학습(supervised learning) 또는 대조 학습(contrastive learning)을 통해 작업 수행과 관련된 신경 활동을 분리하는 것을 목표로 합니다.

- **Performance Highlights**: 단일 지역 잠재 역학(Single-Region Latent Dynamics)에서는 RNNs와 Neural ODEs를 활용해 동적 특징을 추론하며, 다중 지역 소통(Multi-Region Communication)에서는 뇌 지역 간의 분산 상호작용을 포착합니다. 행동 정렬 모델링(Behavior-Aligned Modeling)은 행동 디코딩을 향상시키기 위해 대조 학습과 감독형 분리(disentanglement)를 적용하여 행동과 직접적으로 연관된 잠재 요소를 격리합니다. 최종적으로, 이 논문은 해석 가능한 뇌 역학과 신뢰할 수 있는 신경 디코딩 간의 다리 역할을 할 미래 연구를 위한 벤치마크와 평가 기준, 그리고 개방된 도전 과제를 논의합니다.



### Assessing Automated Prompt Injection Attacks in Agentic Environments (https://arxiv.org/abs/2606.10525)
- **What's New**: 본 논문에서는 LLM 에이전트를 겨냥한 자동화된 프롬프트 인젝션 공격을 평가하여, 흑상자 방법(TAP)과 백상자 방법(GCG)을 AgentDojo 프레임워크 내에서 도입하였습니다. 이 연구는 80개의 작업 쌍과 여러 모델에서 성능 평가를 진행하여, 실행 중인 외부 데이터를 통한 악의적인 명령이 에이전트를 통해 실행될 수 있는 가능성을 강조합니다.

- **Technical Details**: 다양한 LLM 에이전트가 외부 도구와 상호작용하는 능력을 갖추고 있지만, 이로 인해 간접 프롬프트 인젝션과 같은 보안 위협이 발생할 수 있습니다. 본 연구에서는 에이전트가 다단계 추론 및 도구 호출을 요구하는 환경에서 자동화된 공격 방법이 수행되는 방식에 대한 평가를 제공하였습니다. 또한, GCG와 TAP 방법은 LLM 에이전트를 대상으로 하여 사용자 입력을 오버라이드하는 직접 프롬프트 인젝션과 다르게 작동하는 간접 프롬프트 인젝션에 중점을 두고 있습니다.

- **Performance Highlights**: 흑상자 방법인 TAP가 백상자 방법인 GCG보다 상당히 뛰어난 성능을 보였으며, Qwen3-4B 모델에서 45.2%의 성공률을 기록했습니다. 또한, 공격 방법의 유효성은 공격자의 모델에 따라 달라지며, 강력한 모델이 더 효과적인 주입을 생성하는 경향이 있습니다. 본 연구 결과는 자동화된 프롬프트 인젝션이 모델에 따라 달라지는 신뢰할 만한 위협임을 보여주며, 모델에 구애받지 않는 착취에는 여전히 상당한 장벽이 존재함을 강조합니다.



### MoE Enhanced Federated Learning for Spatiotemporal Prediction (https://arxiv.org/abs/2606.10499)
- **What's New**: 최근의 연구에서 MoE-FedTP라는 개인화된 연합형(mixed structure) 예측 프레임워크가 제안되었습니다. 이 프레임워크는 lightweight Mixture-of-Experts (MoE) 네트워크를 기반으로 하여 여러 도시 간의 교통 예측을 개선합니다. MoE-FedTP는 데이터가 부족한 도시를 위해 데이터가 풍부한 도시의 정보를 활용하면서 개인 정보 보호를 유지하는 방법으로 주목받고 있습니다.

- **Technical Details**: MoE-FedTP는 spatiotemporal neural networks를 활용하여 출발지와 목표 도시의 특징을 추출합니다. 이 시스템은 다양한 출발지에서 온 전문가 네트워크를 통한 파라미터 공유 방식으로 작동하며, gating 메커니즘을 통해 각 도시의 교통 패턴에 따른 최적의 전문가를 동적으로 활성화합니다. 이 아키텍처는 차별화된 도시 특성을 정교하게 모델링 할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실제 교통 데이터셋에 대한 실험 결과, MoE-FedTP는 기존의 교차 도시 예측 방법 및 연합 학습 기준선보다 일관되게 더 나은 성능을 보였습니다. 이는 데이터가 부족한 도시의 예측 정확성을 향상시키는 데 효과적임을 보여줍니다. 이 연구는 도시 간 교통 예측의 새로운 가능성을 열어줍니다.



### Achieving Cloud-Grade SLOs for Local Mixture-of-Experts Inference through CPU-GPU Hybrid Design (https://arxiv.org/abs/2606.10493)
Comments:
          Accepted to the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI '26). The official version will appear in the OSDI '26 proceedings published by USENIX

- **What's New**: 이 논문에서는 로컬 환경에서의 대형 Mixture-of-Experts (MoE) 모델의 성능을 클라우드 환경 수준으로 향상시키기 위한 새로운 접근법을 제시합니다. 이 연구는 기존 로컬 MoE 시스템이 클라우드에서 달성하는 서비스 품질에 미치지 못하는 여러 가지 단점을 강조하며, 그 해결 방안을 모색합니다. CPU-GPU 하이브리드 시스템을 통해 30초 이내에 45K 프롬프트를 처리할 수 있는 혁신적인 기술을 도입하고 있습니다.

- **Technical Details**: 제안된 시스템은 stream-loading prefill (SLP)과 distributed stream-loading prefill (DSLP) 방식으로 1,200 tokens/s와 1,800 tokens/s의 처리 속도를 구현합니다. 또한, dual-batch attention-MoE 겹침 방식을 통해 50%의 처리량 향상을 이루며, AVX-512 최적화된 FP8 GEMV 커널을 사용하여 CPU의 레이턴시를 4-5배 낮췄습니다. 이러한 방식은 대형 MoE 모델에 대한 원래의 정밀도 추론을 가능하게 합니다.

- **Performance Highlights**: 논문의 평가 결과, 제안된 시스템은 소비자 등급의 CPU-GPU 플랫폼에서 대형 MoE 모델의 클라우드 수준 서비스 질(QoS)을 달성할 수 있음을 보여줍니다. 특히 낮은 비용으로도 GPU 중심의 초노드 또는 클러스터 배포와 유사한 성능을 제공한다는 점에서 큰 의미가 있습니다. 이를 통해 사용자들은 데이터 센터 인프라 없이도 높은 품질의 접근을 받을 수 있게 됩니다.



### Stop Early, Spend Less: Hidden-State Probes as a Practical Recipe for Streaming Moderation of LLM Outputs (https://arxiv.org/abs/2606.10487)
Comments:
          Technical Report. 14 pages, 3 figures, 4 tables

- **What's New**: 이 논문은 사용자-facing 시스템에서 대형 언어 모델(Large Language Model, LLM)의 출력을 안전하게 필터링하는 효율적인 방법을 제시합니다. 기존의 모더레이션 모델(modation model)을 따로 설정하는 대신, 모델의 숨겨진 상태(hidden states)에서 이미 필요한 신호가 존재함을 관찰하여 경량 토큰 수준의 프로브(probe)를 학습합니다. 이를 통해 각 토큰에 대해 안전성을 평가할 수 있으며, 이러한 접근 방식은 연속적인 토큰 레벨 모니터링을 가능하게 합니다.

- **Technical Details**: 연구팀은 토큰 생성동안 프로브를 실행하여 생성이 완료되기 전에 불안전한 출력을 중단하거나 수정할 수 있도록 합니다. 이 과정은 추가적인 전방 패스(forward pass)를 요구하지 않으며, 디코딩 루프 내에서 밀리세컨드 단위의 속도 검사를 실행할 수 있습니다. 제안된 구조는 LLM의 내부 상태에서 직접 정보를 수집하여 안전성을 평가하는 동시에, 생성 과정에서 실시간으로 개입할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 논문에서는 제안한 방법이 비슷한 작업을 수행하는 기존의 모더레이터와 비교하여 현저히 낮은 계산 오버헤드를 달성함을 실험적으로 입증합니다. 이를 통해 위험한 출력이 최종적으로 사용자에게 도달하기 전에 이를 걸러내거나 수정하는 것이 가능해지며, 안전성을 보장하면서 지연 시간을 최소화합니다. 연구팀은 또한 프로브의 실용적인 배치 레시피를 제공하여 정확도와 비용 간의 조정 가능한 균형을 제시합니다.



### Advancing the State-of-the-Art in Empirical Privacy Auditing (https://arxiv.org/abs/2606.10481)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 민감한 데이터로의 파라미터 효율적인 파인튜닝에 대한 기밀성 문제를 다룹니다. Empirical Privacy Auditing(EPA)을 통해 이 문제를 정량화하고, 새로운 'canary' 생성 방법을 제안하여 고온 샘플링을 통해 이러한 'canary'를 효과적으로 생성합니다. 이 방법은 훈련 데이터와 유사하지만, 충분히 이상의 비정상적인 예제를 포함하여 민감한 데이터의 유출 위험을 평가할 수 있도록 합니다.

- **Technical Details**: 기술적으로, 연구자들은 고온 샘플링(T ≥ 0.8) 방식을 사용하여 LLMs로부터 통합된 'canary'를 생성합니다. 이 'canary'는 훈련 데이터에 인젝션하여 모델의 공격에 대한 응답을 평가하는 데 도움을 줍니다. 금전적인 이점을 위해 본 연구는 새로운 데이터 감사를 도입하여 합성 데이터의 유출을 측정하는 방법을 설명하며, 기초 데이터 분포에서 통계적으로 비정상적이지만 여전히 의미적 구조를 유지하는 생성 과정을 강조합니다.

- **Performance Highlights**: 이 연구의 결과, 'canary'의 엔트로피와 모델 용량 사이의 상관관계를 규명하였고, 모델 용량이 증가함에 따라 최적의 'canary' 엔트로피가 증가한다는 점을 발견했습니다. 이는 LLM이 민감한 데이터의 기록을 메모리하는 데 있어 더 강력한 공격 신호를 생성할 수 있음을 의미합니다. 또한, 기존 방법에 비해 새로운 감사를 통해 합성 데이터의 유출을 더 정밀하게 평가할 수 있음을 보여줍니다.



### Decoupling Thought from Speech: Knowledge-Grounded Counterfactual Reasoning for Resilient Multi-Agent Argumentation (https://arxiv.org/abs/2606.10475)
Comments:
          Accepted for publication in the Proceedings of the 30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026)

- **What's New**: 이 논문에서는 Multi-agent Debate (MAD) 프레임워크에서 기존의 모델들이 최종 출력의 정확성에 유리하게 최적화되어 있기 때문에 긴 대화에서의 안정성을 고려하지 못하는 문제를 지적합니다. 이를 해결하기 위해, Knowledge-Grounded Counterfactual Reasoning (KG-CFR)라는 이중 단계 아키텍처를 도입하여, 개인적 계획 버퍼와 공적 실행 레이어 간의 엄격한 분리를 구현하여 프로세스의 일관성을 유지합니다. 이 연구는 기존 반응형 시스템의 한계를 극복하고, 시스템의 저항력을 구조적으로 향상시키는 방법을 제시합니다.

- **Technical Details**: KG-CFR 플랫폼은 Dynamic Resource Allocation under Uncertainty (DRAU) 환경에서 성능을 평가합니다. 이 아키텍처는 개인적인 시뮬레이션과 전략적 계획을 하는 버퍼를 분리하여, 외부 요인에 대한 Noise에 의한 논리 손상을 방지합니다. 또한, 새로운 벡터 메트릭스를 통해 논의의 다양성과 계획 실행 일치를 측정하여 운영의 안정성을 높은 방향성과 일관성을 가지고 증명합니다.

- **Performance Highlights**: KG-CFR은 95% 이상 전체 실험에서 judge가 감지한 Critical post-shock degradation(품질 변화 기준, $  -0.20$)을 방지하며, 주장 품질이 0.694에서 0.822로 증가했습니다. 이러한 결과는 KG-CFR이 장기적인 압박 하에서도 품질 손실 없이 시스템 저항력을 강화하는 중요한 요소가 된다는 것을 보여줍니다. 더불어, KG-CFR은 구술에서의 반복(looping)을 줄이며, 이러한 수정된 메트릭들이 운영의 일관성을 어떻게 보장하는지 보여줍니다.



### Detecting Speculative Language in Biomedical Texts using Recurrent Neural Tensor Networks (https://arxiv.org/abs/2606.10471)
Comments:
          12 Pages

- **What's New**: 이번 연구에서는 생물 의학 기사에서 추측적 언어(speculative language)를 자동으로 탐지하는 방법을 연구했습니다. 이를 위해 분산 문장 표현(distributed sentence representations)과 심층 학습(deep learning) 기술을 활용하였습니다. 이러한 자동 탐지는 정보 검색(information retrieval) 및 다문서 요약(multi-document summarization)에 중요한 영향을 미칠 수 있습니다.

- **Technical Details**: 연구에서는 두 가지 분산 문장 표현 방법인 Paragraph Vector 모델과 Recursive Neural Tensor Network(RNTN)를 비교하였으며, 이들 방법은 지원 벡터 머신(Support Vector Machines), 나이브 베이즈(Naive Bayes)와 패턴 매칭(pattern matching) 같은 세 가지 기본 알고리즘과 성능을 비교했습니다. RNTN은 F1 점수 0.885로 가장 높은 성능을 보였고, Paragraph Vector 모델은 F1 점수 0.368로 효과적이지 않았습니다. 이 연구는 기존 방법의 성능 차이에 대한 요인들을 논의합니다.

- **Performance Highlights**: RNTN 모델이 SVM(선형 bigram)의 F1 점수 0.881보다 우수한 성능을 보였고, 두 모델 간의 성능 차이에 대해 깊이 있는 분석을 제공하였습니다. 연구에서 활용된 BioScope 및 BioMed Corpus 데이터셋을 통해 다양한 문장에서 추측적 언어를 인식하는 것이 가능하다는 것을 시연했습니다. 연구의 결과는 생물 의학 문서에서의 정보 탐색과 요약 작업에서 중요한 시사점을 제공합니다.



### UPLOTS: A Unified Pretrained Language Model for Constrained Time-series Generation (https://arxiv.org/abs/2606.10466)
- **What's New**: UPLOTS는 다양한 도메인 간 제약조건에 맞춰 통합된 시간 시계열 생성 프레임워크를 제공합니다. 기존에 각 데이터셋마다 별도의 모델을 구축하는 방식에서 벗어나, 단일의 프리트레인된 트랜스포머를 활용하여 더욱 유연하고 스케일러블한 생성 방법을 제안합니다. 이 모델은 동적으로 다중 데이터셋 손실을 재조정하고 제약-패턴 맵핑을 통해 각 도메인에서의 시간적 구조를 내재화하며, 학습 후 맞춤형 시퀀스를 생성할 수 있도록 합니다.

- **Technical Details**: UPLOTS는 사전 훈련된 대형 언어 모델(LLM)을 기반으로 하여 다양한 시간 시계열 데이터셋에서 공통의 시간 패턴을 학습합니다. 데이터셋의 구체적인 제약조건은 자연어 프롬프트로 인코딩되어, 훈련 시 모든 시간 패턴을 혼합하여 학습하고 추론 시에는 요구에 맞춰 특정 시퀀스를 생성할 수 있도록 합니다. 또한, 동적 가중 훈련 전략을 사용하여 이하 과제로 격리된 노이즈를 초기에 낮추고 잘 수행되지 않는 데이터셋에 더 많은 업데이트를 제공하는 방식을 채택합니다.

- **Performance Highlights**: UPLOTS는 4개의 실제 데이터셋과 14가지 프롬프트 설정에서 최신 기법들보다 뛰어난 성능을 보였습니다. 이 모델은 특정 데이터셋에만 의존하지 않고, 맞춤형 시퀀스를 요구에 따라 생성할 수 있으며, 낮은 실제 데이터 환경에서도 데이터 증강을 개선할 수 있는 능력을 입증했습니다. 이 연구는 시간 시계열 생성 분야에서 기존의 '하나의 모델, 하나의 데이터셋' 패러다임을 깨뜨리며, 다양한 목표의 테스트 데이터셋을 효과적으로 처리할 수 있는 기회를 제공합니다.



### ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs (https://arxiv.org/abs/2606.10461)
Comments:
          Accepted to ICML 2026

- **What's New**: 이번 연구에서는 Energy-based Representation Alignment (ERAlign) 프레임워크를 제안합니다. 이 프레임워크는 GNN(그래프 신경망)으로 인코딩된 그래프 구조와 LLM(대규모 언어 모델)으로 파생된 텍스트 임베딩을 공유 잠재 공간으로 투영하여 분포 일관성을 달성합니다. ERAlign은 GNN과 LLM의 중간 레이어를 정량화하여 계층별 정렬을 수행하며, 다운스트림 작업에 대한 잘 정렬된 표현을 생성합니다.

- **Technical Details**: ERAlign은 Cramér distance(크라메르 거리)를 사용하여 표현 간 불일치를 정량화하고 EBM(에너지 기반 모델) 목표 함수로 최적화하여 분포 일관성을 향상시킵니다. 이 프레임워크는 중간 LLM 의미를 GNN message passing에 주입하고 그래프 표현을 LLM에 소프트 프롬프트를 통해 전달하여 양방향 정보 융합을 가능하게 합니다. 또한, Energy Discrepancy (ED) 최소화 방식을 도입하여 샘플링 비용을 줄이며, 전반적인 훈련 효율성을 높입니다.

- **Performance Highlights**: ERAlign은 8개의 벤치마크 데이터셋에서 최첨단 성능을 달성하며, 다양한 감독 비율 및 제로샷 크로스 작업 전이 시나리오에서도 뛰어난 라벨 효율성을 보여줍니다. 기존 방법들에 비해 상당한 성능 개선을 이루었으며, 특정 작업에 맞춘 두 가지 변형도 개발하였습니다. 이 연구는 TAGs와 같은 복합적인 모델링 과제를 해결하는 데 기여하고 있습니다.



### LakeQA: An Exploratory QA Benchmark over a Million-Scale Data Lak (https://arxiv.org/abs/2606.10460)
- **What's New**: 최근 대형 언어 모델(LLMs)은 질문 응답(QA) 작업에서 빠른 발전을 보여주었습니다. 하지만 실제 질문은 정확한 증거 문서와 쌍으로 이루어지지 않는 경우가 많습니다. 'LakeQA'라는 새로운 벤치마크를 소개하는 본 논문은 대량의 데이터에서 증거를 검색하고 분석하는 능력을 동시에 강조합니다.

- **Technical Details**: LakeQA는 약 9.5TB의 위키백과 및 오픈 소스 정부 데이터를 이용해 구축된 이종 데이터 모음으로, 구조화된 데이터와 비구조화된 데이터가 혼합되어 있습니다. 각 작업은 적어도 한 명의 박사급 전문가에 의해 주석이 달리며, 멀티 홉(multihop) 추론이 필요합니다. 이 과정에서 에이전트는 올바른 문서를 발견하고 여러 출처에서 증거를 통합해 답변을 생성합니다.

- **Performance Highlights**: 실험 결과, GPT-5.2는 LakeQA에서 18.37%의 정확히 일치하는 점수만을 기록하여, 이 벤치마크가 도전적인 것을 보여줍니다. 정확도는 추론 강도가 증가함에 따라 급격히 감소하며, 이는 현재 LLM 에이전트들이 EQA에서 증거 검색 및 탐색의 어려움을 겪고 있음을 나타냅니다. 결과적으로, LakeQA는 검색과 추론의 복잡성을 동시에 확장시키는 데 중점을 둔 최초의 벤치마크입니다.



### Minimum Distortion Quantization with Specified Output Distribution (https://arxiv.org/abs/2606.10458)
- **What's New**: 이 논문에서는 확률 변수 $W$의 최적 양자화기(optimal quantizer)를 도출합니다. 이 양자화기는 양자화 출력 $X$가 지정된 분포 $P_X$를 따르도록 하면서, $W$를 $X$로부터 추정할 때의 최소 평균 제곱 오차(MMSE)를 최소화합니다. 이는 정보 통신 및 데이터 익명화와 같은 다양한 응용에 유용한 아이디어입니다.

- **Technical Details**: 양자화기는 $X=\sigmaig(F_{\sigma^{-1}(X)}^{-1}(F_W(W))ig)$ 형태를 가집니다. 여기서 $	ext{σ}$는 MMSE를 최소화하는 최적의 순열이며, $F$는 누적 분포 함수(cumulative distribution function)입니다. 특정 분포를 가진 양자화기를 설계할 때, 주요화(majorization) 개념이 최적성 증명에서 중요한 역할을 합니다.

- **Performance Highlights**: 특히 $P_W$가 구간에 걸쳐 균일한 분포이거나 $P_X$가 $ig\\{1,\,	ext{...},\,k\big\\}$에 대해 균일할 때, 양자화기의 형태는 간단하게 $X=F_{X}^{-1}(F_W(W))$로 표현됩니다. 이러한 접근은 출력 엔트로피(output entropy)를 명확히 통제하거나 채널 입력 요구 사항에 맞춘 출력 분포를 설계하는 데 유용합니다.



### The Distributed Detectability Band Against Marginal-Preserving Attacks (https://arxiv.org/abs/2606.10456)
Comments:
          10 pages, 11 figures

- **What's New**: 이번 논문은 AI 제어 모니터링 메커니즘에서 발생할 수 있는 새로운 형태의 공격인 분산 서브 임계값 파괴 공격(distributed sub-threshold sabotage attack)을 제시합니다. 이 공격 방법은 개별 단계에서 악의적인 행동이 발생하더라도 각 단계에서의 점수가 정상 상태와 유사하게 유지되도록 설계되어 있습니다. 이를 통해 기존의 모니터링 시스템들이 효과적으로 탐지하지 못하는 문제를 다루고 있습니다.

- **Technical Details**: 이 연구에서는 가우스-코풀라(Gaussian-copula) AR(1) 구조를 활용하여 각 단계의 고상성(marginal)을 그대로 보존하는 구조를 구현했습니다. 공격자는 각 단계에서 점수를 정상 행동의 분포와 동일하게 설정하면서, 행동의 연속적인 시간 상관 관계를 통해 피해를 인코딩합니다. 이러한 접근 방식은 모니터가 단지 마진 분포만을 볼 경우, 공격과 정상 행동을 구분할 수 없도록 설계되어 있습니다.

- **Performance Highlights**: 결과적으로, Monitor A는 AUC 0.52로 수치상 우연의 확률에 불과하지만, Monitor B는 같은 1% FPR 목표 하에 AUC 0.79-0.97를 기록하여 더 높은 탐지 성능을 보였습니다. 공격이 진행될수록 Monitor A는 우연한 결과로 축소되는 반면, Monitor B는 AUC 약 0.95를 유지하며 실제로 분산 서브 임계값 공격의 탐지 가능성을 보여주었습니다. 이 연구는 탐지의 한계를 정의하고, 모니터링에서의 마진과 상관 관계 간의 차별화를 처음으로 명확히 하였습니다.



### Mitigating Bias in Low-SNR Financial Reinforcement Learning via Quantum Representations (https://arxiv.org/abs/2606.10448)
Comments:
          Preprint. Code available at this https URL

- **What's New**: 이 논문은 금융 환경에서의 노이즈 문제를 다루기 위해 FPQC-SAC라는 새로운 변형을 제안합니다. FPQC-SAC는 파라미터화된 양자 회로(Parameterized Quantum Circuit, PQC)를 액터 및 크리틱 네트워크 앞에 배치하여 특징 전파를 제어합니다. 이 접근법은 기존의 필터링 방법이나 Q-value 정규화와는 달리, 신호 표현 단계에서 노이즈를 억제합니다. 이를 통해 극단적인 시장 변동의 영향에서 벨만 타겟 추정치를 효과적으로 보호합니다.

- **Technical Details**: FPQC-SAC는 기존의 강화 학습 방법론에서 발생하는 금융적 노이즈 문제를 해결하기 위해 개발되었습니다. 이 방법은 양자 얽힘(quantum entanglement)을 활용하여 복잡한 시장 정보와 상관관계를 유지하면서도 노이즈를 통계적으로 취소하는 구조적 제약을 제공합니다. 연구에서는 기존의 필터링 메커니즘의 한계를 비판하며, 불리한 상태 표현을 해소하기 위한 새로운 접근법이 필요함을 강조합니다.

- **Performance Highlights**: 실제 포트폴리오 관리 작업에 대한 경험적 평가 결과, FPQC-SAC는 표준 SAC에 비해 50% 이상 성능 향상을 보였으며, 최상의 깊이 있는 강화 학습 모형에 비해서도 20% 이상의 성과를 달성하였습니다. 이를 통해 FPQC-SAC는 비샘플 안정성(out-of-sample stability)과 누적 수익(cumulative returns)을 크게 개선시키는 데 성공했습니다. 이러한 성과는 학습 가능한 양자 회로 구조에 기인하며, 이를 통해 강화 학습의 발전 가능성을 보여주고 있습니다.



### Vision-Assisted Foundation Model for Solving Multi-Task Vehicle Routing Problems (https://arxiv.org/abs/2606.10431)
Comments:
          Accepted by TNNLS

- **What's New**: 이 논문은 기존의 다중 작업 차량 경로 문제(Multi-task VRP) 해결 방식의 한계를 극복하기 위해, 시각적 표현을 통합한 비전 지원 기초 모델(VaFM)을 제안합니다. 이를 통해 차량 경로 문제의 다양한 제약 조건을 효과적으로 처리할 수 있는 새로운 방법을 제시합니다. 동적 제약 조건을 학습하고 통합하는데 있어 기존의 그래프 기반 방식을 확장하여 시각적 이미지를 활용하는 점이 주목할 만합니다.

- **Technical Details**: VaFM은 컨볼루션 신경망(CNN)을 기반으로 한 비전 모달리티와 그래프 노드를 통합하여 다양한 VRP 변형을 해결하도록 설계되었습니다. 이 모델은 시각적 이미지에서 패치 수준의 의미를 학습하고, 이를 그래프 기반 노드 임베딩에 통합함으로써 최종 솔루션을 생성합니다. 특히, 여러 제약 조건에 적응할 수 있는 하이브리드 크로스 어텐션 모듈을 통해 각 작업의 수요 분포나 서비스 시간 패턴을 세밀하게 반영할 수 있습니다.

- **Performance Highlights**: 실험 결과, VaFM은 총 16개의 다양한 VRP 변형에 대해 평가되었으며, 복잡한 제약 조건을 갖는 변형에서 특히 SOTA(State of the Art) 방법보다 뛰어난 성과를 보였습니다. VaFM은 다중 제약 조건(예: OVRPTW, OVRPLTW, OVRPBLTW)이 있는 작업에서 큰 성과 개선을 이루었으며, 이는 비주얼 처리 방식이 VRP 문제 해결에 있어 큰 잠재력을 갖고 있음을 시사합니다.



### FOGO: Forgetting-aware Orthogonalization Optimizer (https://arxiv.org/abs/2606.10406)
- **What's New**: 이 논문은 잊는 것이 지속적인 학습만의 문제가 아니라 일반적인 최적화 현상이라고 주장한다. 표준 훈련에서도 지배적인 미니 배치 기울기가 희귀하지만 유용한 업데이트 방향을 억제하여 단기적인 잊음(short-term forgetting)을 유발한다. FOGO라는 확장 가능한 최적화기를 도입하여 이러한 기울기 충돌을 지속적으로 감지하고 해결하는 방법을 제안한다.

- **Technical Details**: FOGO는 모멘텀 업데이트를 스펙트럴 직교화(spectral orthogonalization)하여 지배적인 방향이 최적화를 독점하는 것을 방지한다. 또한, 랜덤 프로젝션(random projection) 기술을 사용하여 유용한 업데이트 방향을 효율적인 코드북 메모리에 저장하며, 이 메모리는 낮은 차원에서 쌍 간 거리(pairwise distances)를 보존하도록 설계되었다. 각 최적화 단계에서 현재 업데이트와 저장된 방향의 충돌은 경량 직교 수정(lightweight orthogonal correction)을 통해 해결된다.

- **Performance Highlights**: FOGO는 클래스 불균형(class-imbalanced) 분류 및 도메인/클래스 변동 상황에서의 지속적 비주얼 학습, LLaVA-7B의 미세 조정, GPT-2 사전 훈련(pretraining) 등 다양한 설정에서 검증되었다. 모든 설정에서 FOGO는知識 유지 및 수렴(convergence)에서 성능을 개선하며, 기존의 Adam 및 Muon 최적화기를 능가하는 성과를 보였다. 이러한 결과는 잊는 것이 태스크 순차 학습(task-sequential learning)의 문제만이 아니라 일반적인 최적화 문제임을 시사한다.



### Harnessing the Collective Intelligence of AI Agents in the Wild for New Discoveries (https://arxiv.org/abs/2606.10402)
- **What's New**: 본 논문에서는 EinsteinArena라는 새로운 에이전트 네이티브 플랫폼을 소개합니다. 이 플랫폼은 공개된 연구 문제를 다루며, 문제 해결을 위한 다양한 기능을 제공합니다. 에이전트들이 실시간으로 문제를 공유하고, 검증할 수 있으며, 그들의 아이디어를 상호 대화 방식으로 발전시킬 수 있습니다.

- **Technical Details**: EinsteinArena는 세 가지 주요 요소로 구성됩니다: (i) 공적으로 검증된 문제의 큐레이션된 컬렉션, (ii) 각 문제의 최선의 해결책을 추적하는 실시간 리더보드, (iii) 에이전트가 중간 결과를 공유하고 실패한 접근 방식을 문서화하며 서로의 발견 위에 구축할 수 있는 공개 논의 포럼입니다. 모든 자료는 투명하게 공개되어 있어, 에이전트들은 현재의 경계를 쉽게 확인할 수 있습니다.

- **Performance Highlights**: 2026년 5월 기준으로 EinsteinArena에서 12개의 새로운 최첨단 수학적 결과가 발견되었습니다. 특히, 유지 놈 문제(kissing number problem)의 경우, 관련 연구에서 가장 알려진 경계값을 593에서 604로 개선하는 성과를 보였습니다. 이러한 결과는 에이전트 간의 협동 작업을 통해 이루어진 것이며, 이는 분산된 과학적 발견이 자율 에이전트 간의 공개적인 상호작용을 통해 가능하다는 것을 입증합니다.



### SkillResolve-Bench: Measuring and Resolving Same-Capability Ambiguity in Agent Skill Retrieva (https://arxiv.org/abs/2606.10388)
Comments:
          Preprint

- **What's New**: 이번 연구는 동일한 능력 카테고리 내에서 위험한 스킬을 구별해내는 새로운 매개변수인 '동일 능력 실행 위험 스킬 검색' 개념을 도입하고, 이를 평가하기 위해 SkillResolve-Bench 1.0이라는 벤치마크를 제시합니다. 이 벤치마크는 총 661개의 도움이 되는 스킬과 위험한 스킬 쌍을 제공하며, 공적인 스킬 라이브러리에서의 검색 성능을 평가하는 데 필요한 다양한 메트릭스를 적용합니다. 이 연구의 핵심은 효과적인 스킬 검색이 단순한 관련성 매칭이 아니라는 점을 강조합니다.

- **Technical Details**: SkillResolve라는 새로운 검색 방법론은 질의에 조건화된 유틸리티 모델을 기반으로 스킬 후보들을 평가합니다. 이 모델은 자원 바인딩, 사전 조건, API 범위 및 출력 스키마 등을 고려합니다. 최종적으로는 각 활성가족에서 유용한 대표 스킬을 선택하고, 이를 정렬하여 상위 K개의 리스트를 생성합니다. 이 방법론은 제품의 의사결정을 더욱 최적화하며, 기존의 스킬 라우팅 방식보다 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: SkillResolve는 Recall@3에서 0.766, NDCG@3에서 0.699을 기록하며, HSR@3는 0으로 유지하여 높은 유용성 검색 성능을 나타냅니다. 기존의 SkillRouter 대비 0.112 Recall@3과 0.165 NDCG@3의 향상을 보였으며, 위험한 스킬의 비율을 낮추는 데 기여했습니다. 이 연구는 동일 능력 카테고리 내의 스킬 선택에서 어떤 대표를 선택하는지가 검색 성능에 큰 영향을 미친다는 점을 확인했습니다.



### Beyond Absolute Imitation: Anchored Residual Guidance for Privileged On-Policy Distillation (https://arxiv.org/abs/2606.10385)
Comments:
          17 pages, 8 figures. Project page: this https URL

- **What's New**: 이 논문은 Anchored Residual On-Policy Distillation (AR-OPD)라는 새로운 방법론을 도입합니다. AR-OPD는 Privileged OPD의 한계를 극복하기 위해 설계되었으며,教師와 학생 모델 간의 정보 이탈을 최소화하고 보다 나은 현지 학습 가능성을 제공합니다. 이 방법은 부분적으로 특권을 가진 교사를 활용하여 전체적인 정보와 목표 지점을 분리하는 기법입니다.

- **Technical Details**: AR-OPD는 두 가지 구성 요소로 이루어져 있습니다: Locally Compatible Anchor 와 Destination-Directed Residual입니다. 이 기법은 원천적 비서로서의 교사의 피드백을 사용하는 대신, 교사의 신뢰성을 보장하는 방법으로 부분적으로 특권 네트워크를 활용합니다. 이를 통해 학생 모델이 현지 시점에서 가능한 목표를 추구할 수 있도록 유도합니다.

- **Performance Highlights**: 다양한 추론 작업에서 AR-OPD는 기본 모델보다 평균 12.1포인트, 그리고 Privileged OPD보다 2.3포인트 높은 성능을 기록했습니다. 또한, 점진적인 이탈을 줄여주며, 하위 연산 에러를 21.7% 감소시켜 학생 모델의 전체적인 성능 향상에 기여합니다. 이는 768 토큰을 초과하는 긴 경로에서 최대 7.2포인트의 성능 개선이라는 긍정적인 결과를 가져왔습니다.



### Towards Critical Branching Mechanism in Recurrent Neural Networks (https://arxiv.org/abs/2606.10384)
- **What's New**: 이 논문은 생물학적 신경 시스템에서 제안된 비판적 상태(critical state)를 인공지능 신경망(ANNs)에 적용하여 검토합니다. 특히, LSTM(Long Short-Term Memory) 네트워크의 숨겨진 상태(hidden state) 동역학을 분석하여 최적 훈련 단계의 작은 네트워크에서 비판적 행동을 발견하였으며, 이는 통계적 아발란체(avalanche) 특성과 연결됩니다. 또한 서브크리티컬(subcritical) 분기와 강한 1/f^{\beta} 잡음을 설명하기 위해 혼합 분기 과정(mixture branching process) 프레임워크를 도입합니다.

- **Technical Details**: 본 연구에서는 LSTM 아키텍처를 구성하기 위해 임베딩(embedding) 레이어, 단일 LSTM 레이어, 선형 분류기(linear classifier)로 구성된 네트워크를 사용하였습니다. 데이터셋은 IMDb 영화 리뷰 50,000개를 포함하고 있으며 각 리뷰는 긍정적(1) 또는 부정적(0) 의견으로 레이블이 다는 설정입니다. 네트워크의 성능을 강화하기 위해 28개의 서로 다른 랜덤 시드를 사용하였고, 훈련 후에는 테스트 정확도가 약 87%에 달하는 결과를 보였습니다.

- **Performance Highlights**: LSTM 네트워크에서 관찰된 스케일 프리 아발란치 크기 분포는 훈련의 특정 단계와 네트워크 구성에 따라 발생합니다. 본 논문에서 도입된 분기 과정 분석은 짧은 거리의 시간적 상관관계를 포착하며, 확장된 분기 메커니즘을 통해 LSTM 활동에서 관찰된 1/f^{\beta} 행동을 재현할 수 있음을 보여줍니다. 이러한 결과들은 LSTM 동역학의 중요한 특성을 이해하는 데 기여하며, 생물학적 신경 시스템과의 유사성을 강조합니다.



### Agentic Hybrid RAG for Evidence-Grounded Muon Collider Analysis (https://arxiv.org/abs/2606.10381)
Comments:
          22 pages, 5 figures, and 6 tables

- **What's New**: 이 연구에서는 muon collider 연구를 위한 evidence-grounded retrieval-augmented generation (RAG) 프레임워크인 agentic hybrid RAG를 소개합니다. 이 프레임워크는 희소한 어휘 검색(sparse lexical retrieval)과 밀집한 의미 기반 검색(dense semantic retrieval)을 통합한 하이브리드 검색 모듈과 함께, 쿼리 분해(query decomposition) 및 증거 확장(evidence expansion)을 지원하는 에이전틱 추론 모듈을 결합합니다. 이를 통해 muon collider 도메인에서의 과학적 질문 응답의 효율성을 높이고, 문헌의 다양성을 체계적으로 평가할 수 있는 첫 번째 벤치마크를 설정했습니다.

- **Technical Details**: 이 프레임워크에서 사용되는 하이브리드 검색 모듈은 BM25와 FAISS 색인(faiss indexing) 기법을 활용하여 여러 유형의 쿼리에 대한 검색 성능을 최적화합니다. 경량 에이전트는 초기 검색에서 놓친 증거를 회수하기 위해 쿼리를 분해하고 후속 쿼리를 생성하여, 과학적 질문 응답에서 필요한 정확성과 추적 가능성을 유지합니다. 결국 이 시스템은 하이브리드 검색과 제어된 증거 확장을 통해 신뢰할 수 있는 정보 수집 및 응답 생성을 지원합니다.

- **Performance Highlights**:  extensive evaluation 결과, 하이브리드 검색이 가장 강력한 검색 기반을 제공하며, 에이전틱 추론은 증거 확장 및 응답 합성에서 가장 효과적인 것으로 나타났습니다. agentic hybrid RAG는 검색 효과성, 응답 품질, 증거 커버리지 및 사실적 기반에서 대표적인 검색 및 RAG 기준선보다 일관되게 우수한 성과를 보였습니다. 이 연구는 미래의 muon collider 연구와 고에너지 물리학 분석 요원 운용을 위한 기초 자료를 제공합니다.



### Expert-Level Crisis Detection in Mental Health Conversations (https://arxiv.org/abs/2606.10380)
- **What's New**: 이 논문에서는 대화형 위기 개입(crisis intervention)에 대한 새로운 데이터셋인 CRADLE-Dialogue를 소개합니다. 이전의 연구들은 주로 정적인 텍스트에 집중하였으나, 실제 대화에서는 위기 신호가 점진적으로 드러나는 과정을 추적해야 합니다. 이를 위해 600개의 다중 턴 대화를 수집하고, 정신 건강 전문가가 다중 라벨을 통해 위기 상황을 주석 달았습니다.

- **Technical Details**: CRADLE-Dialogue는 자살 사고, 자해, 아동 학대 등 다양한 임상 기반 위험을 포함한 대화형 위기 탐지를 위한 벤치마크 데이터셋입니다. 연구에서는 초기 경고 신호(Alert)와 위기가 명확히 인식되는 턴(Confirm) 간의 차이를 구분하는 Alert-Confirm 평가 프로토콜을 제안합니다. 이 프로토콜을 통해 모델이 위험이 나타나는 시점을 추적할 수 있는 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 모델들은 위험이 나타나는 시점을 식별하는 것이 단순히 존재하는 위험을 인식하는 것보다 훨씬 더 어렵다는 것을 보여주었습니다. 기존 모델들은 중간 40%에서 높은 60%의 Micro F1 점수만을 달성했으며, 새로운 32B 파라미터 모델은 기존 오픈 소스 모델을 능가하고 소유 모델들과의 비교에서도 경쟁력 있는 성과를 달성했습니다.



### Test-time Adversarial Takeover: A Real-time Hijacking Interface against Robotic Diffusion Policies (https://arxiv.org/abs/2606.10371)
- **What's New**: 이번 연구에서는 Diffusion 기반 액션 생성 기술이 로봇 정책의 근본적인 요소로 자리잡고 있음을 강조하며, Test-time Adversarial TakeOver (TAKO)라는 새로운 공격 방식을 소개합니다. TAKO는 공격자가 고정된 로봇 정책을 실시간으로 조작할 수 있는 인터페이스를 제공, 정의된 궤적을 따라서 로봇을 컨트롤할 수 있게 합니다. 우리는 이러한 능력이 어떻게 비주얼 입력 경로의 취약성을 이용해 작동하고, 기존의 목표-정책 일치를 활용하는 공격 방식이 갖는 한계를 설명합니다.

- **Technical Details**: 검색 프로세스는 시각 기능(visual features)을 기반으로 하여 조건부 생성(diffusion) 모델을 사용하여 연속적인 신경망을 통해 액션 시퀀스를 생성합니다. TAKO는 카메라 스트림에 여러 개의 사전 최적화된 보편적 적대 패치(universal adversarial patches)를 주입하여 공격자가 실시간으로 로봇의 출력을 제어할 수 있게 합니다. 이 과정에서 접근 방식은 경향의 바이어스를 지속적으로 생성하게 되어, 비주얼 입력 경로에서의 신호 왜곡이 반복 생성 과정에서 수정되지 않도록 합니다.

- **Performance Highlights**: TAKO 방식은 4개의 작업(2D 조작, 시뮬레이션된 공중 배송, 시뮬레이션된 지상 내비게이션, 실제 지상 내비게이션)에서 인간 조작자가 100%의 성공률을 기록할 수 있었음을 보여줍니다. 기존의 목표-정책 일치(Target-Policy Matching, TPM) 방식은 피해자 정책이 편향된 목표 방향으로 일반화하지 못하고 완전히 실패하는 반면, TAKO 방식은 비디오 패치 기반 인터페이스를 통해 공격자가 정의한 궤적을 실시간으로 생성할 수 있는 가능성을 제공합니다.



### Speech Meets ELF: Audio Conditional Continuous-Target Diffusion for Speech Recognition and Translation (https://arxiv.org/abs/2606.10368)
- **What's New**: 이번 연구에서는 대화형 음성이 텍스트로 전환되는 과정에서 발생하는 한계를 극복하기 위해 ELF-S2T 모델을 제안합니다. 기존의 S2T 시스템들은 주로 이산 토큰(Discrete Token)을 사용하여 진행되지만, ELF-S2T는 연속-target 언어 모델링(Continuous-target Language Modeling)의 접근 방식을 따릅니다. 이 모델은 음성 신호를 연속 공간에서 처리하며, Whisper 인코더와 결합하여 음성을 기반으로 텍스트를 생성하는 혁신적인 방법론을 보여줍니다.

- **Technical Details**: ELF-S2T는 음향 인코더의 출력을 오디오 조건으로 사용하고, 사전 훈련된 ELF 모델을 기반으로 하는 연속 시간 흐름 매칭 모델을 훈련하여 텍스트를 생성합니다. 이 모델은 매 단계에서 이산 토큰을 예측하는 대신 연속 텍스트 잠재 공간(Continuous Text Latent Space)에서 생성 프로세스를 유지하며 마지막 단계에서만 최종 표현을 토큰으로 변환합니다. 이를 통해 기존 모델의 텍스트 우선 접근 방식을 극복하고, 오디오 강제(Audio Forcing) 매커니즘을 통해 모델의 의존성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, ELF-S2T는 LibriSpeech 데이터셋에서 5.69%의 단어 오류율(Word Error Rate, WER)과 CoVoST2에서 28.55 BLEU 점수를 달성하여 경쟁력 있는 성능을 보였습니다. 특히 ASR(Automatic Speech Recognition)과 S2TT(Speech-to-text Translation) 오류 분석 결과, 두 과정에서 발생하는 오류가 근본적으로 동일한 원인으로 발생한다는 점을 밝혀냈습니다. 이는 연속 표현 생성을 통해 인식 및 번역 간의 공통적인 의미 매핑 과정이 있음을 나타내며, ELF-S2T의 기여도를 더욱 높이고 있습니다.



### A Practical Recipe Towards Improving Sim-and-Real Correlation for VLA Evaluation (https://arxiv.org/abs/2606.10366)
Comments:
          20 pages

- **What's New**: 이번 연구에서는 비전-언어-행동(VLA) 정책의 평가 및 개선을 위한 시뮬레이션의 유용성을 조사하며, 신뢰할 수 있는 실세계 평가를 위한 시뮬레이션의 역할을 분석합니다. 시뮬레이터와 실제 환경 간의 상관관계를 통해 정책 등급 일관성, 성능 상관관계 및 왜곡 실패 패턴을 측정하여 기존 시뮬레이터의 한계를 규명합니다. 또한, 정책 개선을 위한 시뮬레이터 사용 방법과 후속 학습 데이터의 양이 시뮬레이션과 현실의 정렬에 미치는 영향을 다룹니다.

- **Technical Details**: 다양한 시뮬레이션 플랫폼과 VLA 정책을 포함하여 9개의 테이블탑 조작 작업을 통해 체계적인 평가 플랫폼을 구축했습니다. 각 작업은 비전, 언어, 레이아웃 및 행동 등 4개의 차원에서 제어된 왜곡을 포함하며, 이를 통해 각 시뮬레이터가 실제 정책 평가 결과를 얼마나 잘 유지하는지를 평가합니다. 연구 결과, 신뢰할 수 있는 시뮬레이터는 정책이 실제로 실패하는 방식을 유사하게 반영해야 함을 보여줍니다.

- **Performance Highlights**: 실험 결과, 기존 시뮬레이션 VLA 벤치마크가 실제 모델 랭킹과 강건성 패턴을 얼마나 정확하게 예측할 수 있는지를 분석하였습니다. 시뮬레이터 기반 후속 학습이 실세계 성능 및 시뮬레이션-현실 평가 정렬을 어떻게 향상시킬 수 있는지를 연구하였으며, 적절한 데이터 양에서의 미세 조정이 모델 행동이 실제 세계와 더욱 유사해질 수 있도록 함을 확인하였습니다.



### KG-SoftMAP: Soft Knowledge-Graph Priors for Bayesian Network Structure Learning from Sparse Discrete Data (https://arxiv.org/abs/2606.10358)
Comments:
          33 pages including appendices, 1 figure

- **What's New**: 새로운 연구는 데이터가 희소한 상황에서 Bayesian network (BN) 구조 학습의 어려움을 해결하기 위해 KG-SoftMAP을 제안합니다. 이 방법은 전문가가 정리한 weighted directed knowledge graph (KG)를 사용하여 BN 구조를 개선하며, 이는 BDeu score와 결합된 MAP 목표를 최대화합니다. 기존의 접근 방식과 달리, KG-SoftMAP은 데이터가 부족하더라도 KG의 정보를 활용할 수 있습니다.

- **Technical Details**: KG-SoftMAP의 핵심은 KG의 신뢰도 가중치를 기반으로 하는 soft prior를 활용하여 BN 구조 학습을 수행하는 것입니다. 각 edge의 포함 log-odds는 KG의 신뢰도에 대한 선형 함수로 표현됩니다. 이는 데이터의 증거에 의해 쉽게 대체될 수 있어, KG와 데이터의 상호 작용을 최적화하며 학습 공간을 확장합니다.

- **Performance Highlights**: 실험 결과 KG-SoftMAP은 controlled synthetic benchmarks에서 ground-truth DAGs가 존재하는 상황에서 유의미한 성과를 보였습니다. ρ가 0.05일 때는 DF1 score가 0.14에서 0.29로 향상되었고, ρ가 0.2 이상일 경우에는 0.46에서 0.96으로 증가했습니다. 그러나 실제 교육 데이터에서는 정확한 DAG이 없기 때문에 예측, 보정 및 KG 일관성 등 배포 측정에 중점을 두어 평가하였습니다.



### Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations (https://arxiv.org/abs/2606.10357)
- **What's New**: 이 논문에서는 콘텐츠에서 전자상거래 플랫폼으로의 추천인 Cross-Domain Recommendation (CDR) 분야에서 AIR (Atomic Intent Reasoning)이라는 새로운 프레임워크를 제안합니다. 이는 사용자 상호작용을 통해 구매 의도를 유추하여 전환율을 높이는 것을 목표로 하며, LLM (Large Language Model)의 강력한 의미적 이해 기능을 활용하여 지연 시간을 해결합니다. AIR는 오프라인 단계에서 LLM 추론을 이동시키고, 온라인 운영 중에 효율적으로 사용자 의도 표현을 구성합니다.

- **Technical Details**: AIR 프레임워크는 온라인 지연 시간 제약 사항 아래에서 LLM 수준의 의미적 추론을 가능하게 합니다. 오프라인 단계에서는 LLM를 활용해 사용자 이벤트를 원자 행동 의도 단위로 변환하고, 의도 지식 기반에 조직하여 고속 검색을 수행합니다. 온라인 추론 단계에서는 최근 사용자 행동으로부터 캐시된 원자 의도를 검색하고 구성하여 최신 의도 표현을 구축하며, 이를 통해 지연 시간을 크게 줄입니다.

- **Performance Highlights**: 실험 결과, AIR는 여러 공공 데이터 세트에서 CDR 작업에 대해 최신 상태의 성능을 달성했습니다. Kuaishou 전자상거래에서 실시된 대규모 온라인 A/B 테스트를 통해 GMV를 +3.446% 증가시켜 실제 비즈니스 측정 기준에서 안정적이고 유의미한 개선을 보였습니다. 이는 AIR의 효과와 산업 규모 추천 시스템에서의 실용적 가치를 전적으로 검증합니다.



### Routing-Aware Expert Calibration for Machine Unlearning in Mixture-of-Experts Language Models (https://arxiv.org/abs/2606.10338)
- **What's New**: 이번 연구에서는 Mixture-of-Experts (MoE) 아키텍처에서 머신 언러닝에 대한 새로운 접근 방식인 TRACE를 제안합니다. 기존의Dense 모델에 비해 MoE 모델은 각 토큰을 전문가의 희소 부분 집합에 할당하는 라우터를 활용합니다. 이로 인해 정보가 여러 전문가에 분산될 때, 언러닝 과정에서의 비효율성과 처리 문제가 발생할 수 있음을 보여줍니다.

- **Technical Details**: TRACE는 오프라인 활성화 통계를 통해 forget-critical 전문가를 식별하고, 각 전문가의 retain 활성화 주파수와 forget 측의 주파수를 일치시키기 위해 토큰 수준의 retain 손실을 재조정합니다. 이는 MoE 아키텍처의 특성을 반영하여, 각 전문가에게 더 나은 보호를 제공하는 시스템을 구축하고자 하는 노력입니다.

- **Performance Highlights**: 실험 결과, TRACE는 여러 MoE LLMs에서 forget-utility trade-off를 일관되게 개선하며, WMDP 및 MUSE-BOOKS 테스트 세트에서 강력한 기준선보다 9%의 유틸리티 개선을 달성했습니다. 특히 MUSE-BOOKS metrics의 세 가지에서 최고 성능을 보이면서, 언러닝의 효과성을 잘 입증했습니다.



### Building Change Detection in Earthquake: A Multi-Scale Interaction Network and A Change Detection Datas (https://arxiv.org/abs/2606.10329)
- **What's New**: 이번 연구는 지진 피해 평가를 위한 새로운 Change Detection (CD) 데이터셋, Turkey Earthquake CD dataset (TUE-CD)를 제시합니다. TUE-CD는 2023년 2월 6일 터키에서 발생한 7.8 강진 이후 5일 이내에 수집된 다중 시간 원격 감지 이미지 쌍으로 구성되어 있으며, 이를 통해 신속한 재난 구호 필요에 부응하고자 합니다. 더 나아가, 다중 스케일 특성 상호작용 네트워크(Multi-scale Feature Interaction Network, MSI-Net)를 통해 이미지 간의 측면 문제를 완화하며 효과적인 변화를 탐지할 수 있는 방안을 마련했습니다.

- **Technical Details**: MSI-Net은 여러 모듈로 구성되며, 이 모듈들은 변화를 탐지하는 과정에서 다중 스케일의 특징을 효과적으로 통합합니다. 구체적으로, Joint Cross-Attention (JCA) 모듈을 통해 다중 스케일 특징 간 상호작용을 강화하고, Multi-scale Offset Calibration (MOC) 모듈을 통해 이미지 간의 정렬을 최적화합니다. 마지막으로, Feature Integration (FeI) 모듈에 의해 보정된 특성과 다중 스케일 특성이 결합되어 변화 맵을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 MSI-Net이 WHU-CD, CLCD 및 TUE-CD 데이터셋에서 기존의 여러 딥러닝 기반 CD 방법보다 우수한 성능을 보였습니다. MSI-Net은 특히 건물 피해 추정에서의 효과적인 결과를 보여주었으며, 이로 인해 재난 구호 작업에 필수적인 도구가 될 수 있음을 입증하였습니다. 전반적으로, MSI-Net은 가까운 시간 간격의 데이터에 대한 높은 정확도를 제공하여 지진 후 응급 구호에 적합한 솔루션을 제시합니다.



### Content-Induced Spatial-Spectral Aggregation Network for Change Detection in Remote Sensing Images (https://arxiv.org/abs/2606.10328)
- **What's New**: 이번 연구에서는 공간 정보와 스펙트럼 정보의 통합이 변화 감지 성능의 향상에 기여할 수 있다는 점에 주목하여, Content-Guided Spatial-Spectral Integration Network (CSI-Net)을 제안합니다. CSI-Net은 공간 이해 모듈(SR), 스펙트럼 차이 모듈(SD), 콘텐츠 기반 통합 모듈(CGI)로 구성되어 있으며, 이러한 모듈을 통해 전반적인 공간 정보와 스펙트럼 차이 정보를 효과적으로 융합할 수 있습니다. 특히, 이 네트워크는 변화가 없는 지역에서 발생하는 스펙트럼 차이의 영향을 최소화할 수 있도록 설계되었습니다.

- **Technical Details**: CSI-Net의 SR 모듈은 그래프 합성곱 블록을 통해 전역 공간 정보를 학습하며, SD 모듈은 피처의 평균과 분산을 계산하여 변화가 없는 지역에서 스펙트럼 차이를 완화합니다. 또한 CGI 모듈은 공간-스펙트럼 피처의 통합을 위해 고급 콘텐츠 정보를 가이드로 활용하여 상호작용의 적절성을 높입니다. 따라서 CSI-Net은 변화한 특징을 더 잘 학습할 수 있고, 스펙트럼 차이를 억제하는 데 기여합니다.

- **Performance Highlights**: LEVIR-CD, WHU-CD, CLCD 데이터셋에서 수행된 실험 결과, CSI-Net은 최신 기술(State-of-the-Art) 방법보다 우수한 성능을 보였으며, 다양한 시나리오에 적용 가능한 것으로 나타났습니다. CSI-Net은 변화 감지 정확도에 영향을 미치는 무관한 변화의 효과를 배제할 수 있는 기능을 갖추고 있습니다.



### Baseline-Free Policy Optimization for Neural Combinatorial Optimization (https://arxiv.org/abs/2606.10321)
- **What's New**: 이번 연구에서는 Neural Combinatorial Optimization(NCO)에서 baseline을 완전히 제거한 Group Relative Policy Optimization(GRPO)을 평가합니다. 기존의 REINFORCE는 rollout baseline을 필요로 하여 훈련 중 불안정성을 초래하는 구조적 취약점을 가지고 있습니다. GRPO는 샘플링된 트랙 제안 내에서 이점을 정규화하여 이러한 문제를 해결할 수 있는 방안을 제공합니다.

- **Technical Details**: GRPO는 각 인스턴스에서 GG 트랙 제안을 생성하고, 각 그룹 내에서 z-score로 이점을 계산하여 외부 baseline이 필요하지 않습니다. 이 방법은 NCO에 적용하기 위해 신뢰할 수 있는 대안으로 제시되며, TSP와 CVRP benchmark에서의 성능을 평가합니다. 연구에서는 REINFORCE, POMO, PPO 및 P3O와 같은 여러 알고리즘과의 비교를 통해 GRPO의 효용성을 입증합니다.

- **Performance Highlights**: 실험 결과, GRPO는 TSP-100에서 REINFORCE에서 관찰된 훈련 붕괴를 피하면서도 POMO에 대해 2% 이내의 솔루션 품질을 달성했습니다. GRPO는 복잡한 인스턴스에서도 안정적으로 성능을 유지하며 외부 비선형 baseline이 필요하지 않은 점에서 매력적인 대안으로 평가되고 있습니다. 반면 P3O는 TSP에서는 경쟁력을 보였으나 CVRP에서는 변동성이 높게 나타났습니다.



### Catching One in Five: LLM-as-Judge Blind Spots in Production Multi-Turn Transaction Agents (https://arxiv.org/abs/2606.10315)
Comments:
          13 pages, 1 figure, 5 tables

- **What's New**: 이번 연구에서는 생산 환경에서 운영되고 있는 다회전 음식 및 음료 주문 에이전트를 평가하기 위해 LLM-as-judge의 신뢰성에 대해 조사합니다. LLM이 실제 품질 문제를 얼마나 잘 감지하는지 분석하며, 기존의 인간 리뷰와의 일치성보다는 실제 결함의 감지율(recall)에 초점을 맞추었습니다. 연구 결과는 LLM의 결함 감지율이 매우 낮다는 것을 보여줍니다.

- **Technical Details**: 다회전 주문 시스템에서는 고객이 자연어로 음식 및 음료를 주문하는데, 이 LLM은 주문을받아들이고, 메뉴 검색 및 장바구니 생성 등의 작업을 수행합니다. 평가 시스템은 싱가포르의 커피 브랜드와 이중 언어 고객 기반을 가지고 있으며, 다양한 시나리오를 통해 랜덤화된 다회전 대화가 생성됩니다. LLM의 내장된 평가자가 세 가지 축(의도, 브랜드 목소리, 개인화)을 기준으로 점수를 매기지만, 상태 추적(state-tracking)이나 손해 복구와 같은 중요한 차원에 대한 분류가 없어 결함을 간과하는 구조가 드러났습니다.

- **Performance Highlights**: 연구 결과 저자는 LLM 평가자가 인간이 확인한 체계적인 문제의 25% 미만을 감지한다고 주장합니다. 예를 들어, 평가자가 한 배치에서 9개 패턴 중 2개(22%)만을 포착하는 반면, 100회차의 다른 배치에서는 인간 리뷰가 23개의 명확한 결함을 확인했음에도 불구하고 평가자는 실패로 표시한 턴이 0회였습니다. 이러한 결과는 LLM 평가자의 설계와 운영 메커니즘의 결함을 강조합니다.



### The Confident Liar: Diagnosing Multi-Agent Debate with Log-Probabilities and LLM-as-Judg (https://arxiv.org/abs/2606.10296)
Comments:
          15 pages, 7 figures, 1 table, ACL proceedings

- **What's New**: 본 논문에서는 다중 에이전트 논쟁 시스템의 중간 추론 품질을 평가하는 새로운 접근 방식을 제안합니다. 기존의 연구들은 최종 답변의 정확도만을 평가하는 경향이 있었으나, 이 연구는 중간 추론의 질이 최종 결과에 미치는 영향을 분석합니다. 특히, 두 에이전트가 논쟁하는 구조와 LLM을 판사로 활용하여 각 에이전트의 추론을 평가하는 프레임워크를 구축하였습니다. 이를 통해 중간 단계에서의 신뢰도 신호가 외부 평가와 얼마나 연관되는지 분석하기 위한 체계적인 연구를 진행합니다.

- **Technical Details**: 논문은 추론 토큰들에 대한 log-probability 분포, LLM으로부터의 루브릭 점수 및 최종 작업 정확도라는 세 가지 신호 간의 관계를 탐구합니다. 특히, 두 가지 에이전트(Constructor와 Auditor)가 논쟁하는 구조에서 신뢰도 신호가 어떻게 진화하는지 살펴봅니다. 또한, 각 에이전트의 신뢰도는 LLM이 평가한 추론 품질과 어느 정도 일치하는지를 분석하는 중요한 질문을 다룹니다. 이를 통해 내부 신뢰도 신호와 외부 평가 간의 상관 관계를 조사합니다.

- **Performance Highlights**: 실험 결과, Constructor 에이전트의 논리적 추론 품질은 Auditor에 비해 두 배 정도 더 잘 평가되는 경향을 보였습니다. 특히, Constructor의 신뢰도 기반 비판 추론 실패 탐지는 Auditor보다 더 높은 신뢰성을 보였으며, AUROC 점수는 각각 0.804와 0.634로 나타났습니다. 이러한 결과는 다중 에이전트 논쟁 시스템의 설계를 개선할 수 있는 기초 자료로 활용될 수 있으며, 각 작업과 논쟁 구성에 따라 내부 신뢰도 신호와 외부 평가 결과 간의 차이를 진단하는 데 도움을 줄 수 있습니다.



### LLM-Guided Neural Architecture Search for Robust Co-Design of Physical Neural Networks (https://arxiv.org/abs/2606.10294)
- **What's New**: UH-NAS(Unconventional Hardware Neural Architecture Search)는 하드웨어와 관련이 없는 새로운 NAS 프레임워크로, LLM(대형 언어 모델)을 활용하여 정확도와 추론 에너지를 동시에 최적화합니다. 이 프레임워크는 하드웨어를 교체 가능한 백엔드로 활용하고, 각 플랫폼에 따른 에너지 모델과 물리적 제약을 포함합니다. 이를 통해 UH-NAS는 다양한 백엔드에서 공정한 시스템 수준 비교를 가능하게 합니다.

- **Technical Details**: UH-NAS는 다중 목표 최적화 문제를 정의하고, 특정 하드웨어에 맞춘 후보 아키텍처를 탐색하는 과정을 포함합니다. 이전 NAS 방법들과의 차별점은 LLM-guided 진화적 탐색을 통합하여 비정상적인 하드웨어 특성을 충분히 고려하는 점입니다. 또한, UH-NAS는 후보 아키텍처의 모든 성능을 전체 노이즈 교육으로 평가하여 기존 방법이 간과했던 하드웨어 비이상성을 해결합니다.

- **Performance Highlights**: UH-NAS는 CPU, GPU, MZI 기반 광학 시스템 등 다양한 하드웨어 플랫폼에 맞춰 아키텍처를 최적화하여 일관된 시스템 수준 비교를 가능하게 합니다. 비정상적인 하드웨어 조건에서 더 다양한 아키텍처를 발견하고 기존 NAS 기준보다 낮은 검증 오류를 달성합니다. 이 연구는 비정상적인 하드웨어에서의 정확도-에너지 트레이드오프의 중요성을 강조하며, 아키텍처와 하드웨어의 공동 설계가 필요함을 보여줍니다.



### Towards Robust Arabic Speech Emotion Recognition with Deep Learning (https://arxiv.org/abs/2606.10278)
Comments:
          21 pages, 16 figures, 11 tables. Submitted manuscript

- **What's New**: 이번 연구는 아랍어의 Speech Emotion Recognition(SER)에서 하이브리드 아키텍처가 감정 인식의 성능을 향상시킬 수 있는지를 탐구합니다. CNN-LSTM, CNN-Transformer, 및 wav2vec 2.0 모델을 포함한 세 가지 아키텍처를 비교하는 체계적인 프레임워크를 제안합니다. 이러한 접근은 아랍어의 방언 다양성 및 데이터 셋의 한계를 극복하기 위한 새로운 통찰력을 제공합니다.

- **Technical Details**: 우리는 CNN-LSTM, CNN-Transformer, wav2vec 2.0를 포함한 다양한 모델을 사용하여 아랍어 음성 신호에서 감정을 분류하는 다중 클래스 분류 문제로 모델을 설정합니다. 각 모델은 MFCC와 스펙트로그램을 활용하는 전통적인 방식과, raw audio를 직접 처리하는 self-supervised representation을 학습하는 방식으로 나누어집니다. 이 과정에서 Mel-spectrogram 추출 기술을 이용하여 입력 신호를 시간-주파수 표현으로 변환합니다.

- **Performance Highlights**: 연구 결과, 제안된 CNN-Transformer 아키텍처가 98.1%의 정확도를 기록하며 다른 모델들보다 월등한 성능을 보였습니다. 이는 CNN의 지역 스펙트럼 피쳐 추출 능력과 Transformer의 장기 의존성 모델링 능력을 결합한 결과입니다. 이러한 성과는 제한된 자원과 방언적으로 다양한 아랍어 음성 환경에서의 감정 인식 가능성을 높이는 데 기여합니다.



### Hierarchical Policies from Verbal and Egocentric Human Signals for Natural Human-Robot Interaction (https://arxiv.org/abs/2606.10276)
Comments:
          We provide video demos and code in: this https URL

- **What's New**: 이번 연구는 인간-로봇 상호작용을 보다 자연스럽게 만들기 위해 비언어적 신호(예: 시선, 제스처)를 언어 지시와 함께 활용하는 EDITH라는 로봇 프레임워크를 소개합니다. 기존 로봇 정책은 언어 지시만을 사용하여 의사를 전달했으나, EDITH는 실시간으로 스마트 안경을 통해 인간의 첫 번째 시점 보기와 시선을 포착하여 이를 로봇 정책의 입력으로 사용합니다. 이러한 접근은 로봇이 인간의 의도를 더 잘 이해하고 행동할 수 있게 해줍니다.

- **Technical Details**: EDITH는 Project Aria 안경을 기반으로 하여 인간의 첫 번째 시점 보기, 시선, 음성을 실시간으로 스트리밍합니다. 이 시스템은 고수준 정책과 저수준 정책으로 구성된 계층형 정책을 통해 작동하며, 고수준 정책은 시각적 및 언어적 신호를 통해 인간의 의도를 추론하고 세부 작업을 생성합니다. 이를 통해 로봇은 비언어적 신호를 기반으로 행동을 수행하고, 이러한 세부 작업은 해당 의도를 장면에서 정박하는 키프레임으로 연결됩니다.

- **Performance Highlights**: 실험 결과, EDITH는 언어 지시만 사용하는 기존 방법과 비교하여 평균 59.7%의 성공률을 달성하여 인간의 비언어적 신호를 통해 의도를 인식하는 데 혁신적인 효과를 보였습니다. 사용자 연구에 따르면, EDITH는 로봇에게 의사를 전달하는 작업의 부담을 크게 줄여주며, 이는 자연스러운 상호작용을 제공함을 보여줍니다. 이러한 결과는 비언어적 신호를 포함하는 접근 방식이 인간-로봇 상호작용을 위해서 더욱 유망하다는 것을 의미합니다.



### What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents (https://arxiv.org/abs/2606.10267)
- **What's New**: 이 논문은 로봇 조작을 위한 계층적 비전-언어-행동(Hi-VLA) 시스템의 설계 원칙을 체계적으로 분석하고 정리합니다. 기존 Hi-VLA 시스템들은 서로 다른 계획자(planner), 행동자(controller), 관찰 및 메모리 표현 방식에서 차이점을 보이고 있으며, 이러한 분야에서의 통합된 연구가 부족했습니다. 저자들은 다양한 조작 작업을 통해 강력한 Hi-VLA 시스템을 구축하기 위한 실용적인 원칙들을 제시합니다.

- **Technical Details**: 저자들은 옵션 프레임워크(options framework)를 기반으로 한 공유 제어 루프(control loop) 아래에서 다양한 Hi-VLA 에이전트를 통합합니다. 이 연구에서는 VLM(vLanguaged Model)과 VLA(Vision Language Action) 정책 간의 상호작용을 다루며, 메모리 모듈(memory module)과 관찰 표현 모듈(observation representation module)을 통해 계층적 시각 운동 정책을 정의합니다. 이를 통해 시스템 설계를 평가하고, 각 구성 요소의 구현이 결과에 미치는 영향을 체계적으로 분석합니다.

- **Performance Highlights**: 실험 결과, 단순한 Hi-VLA 디자인도 일반적인 VLA보다 개선된 성능을 보이나, 신중하게 선택된 피라미드 구조의 설계가 특히 긴 수평(long-horizon) 및 추론-intensive 작업에서 더 큰 이득을 가져온다는 것을 발견했습니다. 강력한 Hi-VLA 성능은 모델 백본(backbone)과 인터페이스 간의 상호작용에 의존하며, 저자들은 이러한 발견이 더 강력하고 견고한 계층적 VLA 에이전트를 구축하는 데 기여할 수 있음을 강조합니다.



### Multi-Level Analyzation of Imbalance to Resolve Non-IID-Ness in Federated Learning (https://arxiv.org/abs/2606.10250)
Comments:
          27 pages, 5 figures, 13 tables. Accepted for publication in Neurocomputing (2025). Author Accepted Manuscript

- **What's New**: 이번 연구에서는 Federated Learning(FL)에서의 클래스 불균형 문제를 다루기 위해 FedBB라는 새로운 접근 방식을 제안합니다. FedBB는 Positive Negative Balanced (PNB) 손실 함수와 Client Balanced Reweighting (CBR)을 활용하여 로컬 학습과 모델 집계 최적화를 동시에 수행합니다. 이 연구는 클래스 불균형 문제를 인터-케이스, 인터-클래스, 인터-클라이언트의 세 가지 수준으로 나누어 분석합니다.

- **Technical Details**: Federated Learning(FL)에서는 데이터의 분포가 서로 다를 수 있으며, 이러한 현상은 성능 저하를 초래합니다. 이를 해결하기 위해 FedBB는 PNB 손실 함수로 로컬 클라이언트의 학습에서 불균형을 조정하며, CBR을 통해 모델 집계 과정에서 각 클라이언트의 데이터 스키우와 불균형을 고려합니다. PNB 손실 함수는 소수 클래스에 더 높은 가중치를 부여하여 멀티 라벨 및 멀티 클래스 분류의 성능을 향상시킵니다.

- **Performance Highlights**: 대규모 X-ray 및 자연 이미지 데이터셋에 대한 다양한 실험을 통해 FedBB가 기존 알고리즘보다 우수한 성능과 효율성을 발휘함을 입증하였습니다. Ablation study를 통해 PNB 손실과 CBR이 각각 성능에 기여함을 확인하였으며, FedBB는 제한된 통계 정보로도 높은 성능을 유지하여 개인 정보 보호를 지원합니다. 이 연구는 FL의 일반화 및 개인화를 위한 강력한 기준선을 제공합니다.



### Linguistically Augmented Audio Speech Data (LinguAS) (https://arxiv.org/abs/2606.10246)
- **What's New**: 최근 음성 합성 기술은 인간의 목소리와 구별할 수 없을 정도로 정교해지고 있으며, 이는 인공지능 음성 비서와 같은 유용한 응용프로그램과 함께 음성 딥페이크의 우려도 커지고 있습니다. 이에 따라 Linguistically Augmented Audio Speech Data (LinguAS) 데이터셋이 소개되었으며, 이는 진짜와 딥페이크 음성을 구별할 수 있는 중요한 언어적 특징으로 주석이 달린 800개 이상의 오디오 샘플로 구성되어 있습니다. 이 데이터셋은 각 오디오 샘플에 대해 전문가 정의 언어 특징(EDLFs)을 포함하고 있어, 모델의 효율성을 높이는 데 기여합니다.

- **Technical Details**: LinguAS 데이터셋에는 음성의 다양한 수준의 표현을 담고 있습니다. 신호 처리 방식으로는 기본적으로 Raw Acoustic Audio와 프레임 수준의 표현 방식인 LFCC와 MFCC가 사용되며, 이는 딥러닝 모델에서 자주 활용됩니다. 데이터셋은 음성 품질에 따라 네 가지의 스푸핑 변조 공격 타입으로 나뉘며, 진짜 음성과 균형을 이루는 수치로 구성되어 있습니다. 또한, 음성의 생성자 및 원천에 대한 메타데이터도 포함되어 있어 모델 학습에 더 많은 정보를 제공합니다.

- **Performance Highlights**: LinguAS로 훈련된 모델들은 ASVspoof 2021의 딥러닝 기준선 모델 및 HuBert, XLSR 같은 SSL 모델을 초월하는 성능 향상을 보여주었습니다. 특히, LinguAS의 언어적 특징 및 메타데이터는 딥페이크 음성 연구자들이 인간 언어의 특성을 바탕으로 더 신뢰성 있는 모델을 개발하는 데 유용한 데이터셋으로 작용합니다. 이 데이터셋은 공개적으로 이용 가능하며, 이를 통해 모델 설명 가능성을 높일 수 있는 장점도 포함되어 있습니다.



### YUBI: Yielding Universal Bidigital Interface for Bimanual Dexterous Manipulation at Sca (https://arxiv.org/abs/2606.10244)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 직관적이고 인체 공학적이며 확장 가능한 데이터 수집을 위한 손가락 정렬 그리퍼인 YUBI(수축형 범용 양손 인터페이스)를 소개합니다. 기존의 UMI(범용 조작 인터페이스)와 비교해, YUBI는 손가락의 자연스러운 움직임을 직접적으로 그리퍼의 조작으로 매핑하여 기존 그리퍼 시스템의 단점인 인체 공학적 문제를 해결합니다. 또한, YUBI는 가벼운 디자인과 정밀한 조작이 가능한 구성을 가지고 있으며, 대규모 데이터 수집을 지원하기 위해 다양한 실험과 사용자 연구를 통하여 그 효과를 입증하였습니다.

- **Technical Details**: YUBI는 손가락 정렬 인터페이스를 기반으로 하여 작업자의 자연스러운 집게 운동에 맞춰 그리퍼의 간격이 조정되는 측면에서 획기적입니다. 이 시스템은 고주파 VR 센서를 그리퍼에 직접 통합하여 높은 충실도의 그리퍼 궤적 추적을 가능케 하며, 데이터를 수집하는 동안 물리적 소모를 줄여줍니다. 데이터 수집 체계에서는 6 DoF(자유도) VR 추적을 통해 고품질 궤적 데이터를 확보하며, 이를 통해 수집된 데이터는 다수의 로봇 플랫폼에 쉽게 적용 가능합니다.

- **Performance Highlights**: YUBI는 8434시간의 데이터로 구성된 대규모 데이터셋을 기반으로 하고 있으며, 이는 1.20M 에피소드와 119가지 작업에 걸쳐 있습니다. 사용자 연구 결과에 따르면 YUBI는 기존의 UMI 그리퍼보다 복잡한 양손 작업에서 더 나은 versatility(다양성), dexterity(손재주), 그리고 operational efficiency(작업 효율성)를 제공하는 것으로 나타났습니다. 이 연구의 결과는 YUBI의 데이터 기반 정책 네트워크가 UR, Franka, ELEY 등 여러 양손 로봇 플랫폼에서 효과적으로 전이 가능하다는 것을 보여줍니다.



### Hyperbolic Neural Population Geometry Benefits Computation (https://arxiv.org/abs/2606.10238)
Comments:
          Accepted at ICML 2026, 37 pages, 5 figures

- **What's New**: 이번 논문에서는 해마에서의 인구 활동이 하이퍼볼릭 구조를 형성한다는 이론적 프레임워크를 제시합니다. 우리는 해마 조정 곡선을 제안하여 이의 불확실성을 통계적으로 유도하며, 현대 홉필드 네트워크의 업데이트 규칙이 최적 최소 평균 제곱 오차(MMSE) 추정기를 계산함을 보입니다. 나아가, 하이퍼볼릭 공간에 정의된 새로운 연관 기억 모델을 제안하여 이전 모델들보다 더 큰 용량을 자랑합니다.

- **Technical Details**: 이 논문에서는 해마에서의 신경 인코딩 모델을 소개하고 베이지안 최적 디코더를 유도합니다. 해마 코딩은 장소 분야의 확률 분포가 지수적인 경우 하이퍼볼릭 기하학을 유도하며, 이는 나무 구조에 해당합니다. 또한, 우리는 현대 홉필드 네트워크의 기억 회상 동력이 최적의 MMSE 추정기를 근사하는 것을 보여주며, 하이퍼볼릭 공간에서 직접 작동하는 새로운 연관 기억 모델을 구축합니다.

- **Performance Highlights**: 우리는 제안한 하이퍼볼릭 연관 기억 모델이 패턴 완성을 수행할 때 기존의 기억 모델보다 우수한 정확도를 보여줍니다. 이 모델은 ML(기계 학습) 수행에서 상당한 성능 향상을 이루며, 특히 숨겨진 차원성이 제한된 경우 하이퍼볼릭 기하학이 더 효율적인 정보 저장 공간을 제공하는 것을 시사합니다. 최고의 성능은 하이퍼볼릭 메모리 모듈이 다양한 ML 아키텍처에 원활하게 통합될 수 있음을 보여줍니다.



### SHAPO: Sharpness-Aware Policy Optimization for Safe Exploration (https://arxiv.org/abs/2606.10228)
Comments:
          ICLR 2026

- **What's New**: 이 논문에서는 안전이 중요한 분야에서 강화 학습(RL) 에이전트를 배치하기 위한 필수 요소로 안전 탐색을 다루고 있습니다. 저자들은 Sharpness-Aware Policy Optimization (SHAPO)이라는 새로운 알고리즘을 제안하여 에이전트의 불확실성에 기반해 정책 업데이트를 수행합니다. SHAPO는 높은 불확실성을 가진 영역에서 에이전트가 안전하게 행동하도록 유도하며, 이를 통해 안전성과 작업 성능을 향상시킵니다.

- **Technical Details**: SHAPO는 정책 업데이트 시 파라미터의 변화에 따른 기울기를 평가하는 방법을 사용합니다. 이 과정에서 희귀한 위험한 행동의 영향을 강화하고, 이미 안전한 행동의 영향을 줄이기 때문에, 학습을 보수적으로 유도합니다. 저자들은 Fisher metric을 활용한 변동성이 Euclidean metric보다 우수하며, 액터(Actor)쪽의 위험 회피가 안전 탐색에 더 중요한 영향을 미친다고 입증합니다.

- **Performance Highlights**: SHAPO는 Safety-Gym 및 MuJoCo 환경에서 여러 온폴리시 안전 RL 기법들에 비해 지속적으로 안전성과 작업 성능을 개선했습니다. 이 방법은 누적 실패를 줄이고, 에피소드 비용의 분포를 완화하며 파레토 프론티어를 크게 확장했습니다. 다양한 실험을 통해 SHAPO의 효과를 입증하며, 기존의 방법들과 비교하여 우수한 성능을 보였습니다.



### Dual-Branch Gated Fusion for Open-Set Audio Deepfake Source Tracing (https://arxiv.org/abs/2606.10223)
- **What's New**: 이 논문은 합성 발화(synthetic utterance)의 출처를 정확히 추적하기 위한 새로운 이중 분기 게이트 융합 프레임워크(dual-branch gated fusion framework)를 제안합니다. 기존의 닫힌 집합(closed-set) 모델이 새로운 합성기(synthesizers)를 거부하지 못하고 지나치게 자신감 있는 예측을 내리는 한계점을 극복하고자 합니다. 특히, XLSR-53 모델과 CORES라는 66차원 설명자를 결합함으로써, 서로 다른 합성 아티팩트(synthesis artifacts)를 효과적으로 캡처하는 방안을 모색합니다.

- **Technical Details**: 제안된 프레임워크는 입력된 데이터에 따라 각 분기를 동적으로 가중치 조정하는 입력 조건부 게이트(input-conditioned gate)을 채택합니다. 이 과정에서 교차 엔트로피(cross-entropy) 손실과 에너지 마진 손실(energy margin loss)을 이용하여 도메인 내(In-domain)와 도메인 외(Out-of-domain) 분리를 수행합니다. 특히, CORES는 선형 필터 뱅크(linear filter bank)만을 사용하는 이전 연구들과는 달리, 파형의 다양한 차원을 아우르며 합성 오디오(synthesized audio)를 자연 음성(natural speech)과 구별하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 제안된 시스템은 MLAAD 벤치마크에서 97.6%의 ID 정확도와 4.9%의 EERc, 83.5%의 상대적 FPR95 감소를 달성하여 Interspeech 2025 기준을 넘어서는 성능을 보여줍니다. 이 시스템은 고정 가중 융합(fixed-weight fusion) 및 단일 스트림 SSL 아키텍처가 동시에 달성할 수 없는 균형을 유지하며, 개방형 집합(open-set)에 대한 민감성을 유지하면서도 강력한 분류 성능을 제공합니다.



### Fast Exact Nearest-Neighbor Learning for High-Frequency Financial Time Series (https://arxiv.org/abs/2606.10219)
Comments:
          15 pages 5 figures;

- **What's New**: 최근 금융 분야에서 AI 효율성이 매우 중요해지고 있습니다. 데이터량 증가와 함께 실시간 거래 및 위험 관리에 대한 요구가 높아지면서, AI 모델은 과거 데이터를 효율적으로 학습해야 합니다. 본 논문은 Mojo를 기반으로 한 SIMD k-d 트리를 통해 고빈도 금융 시간 시계열의 근처 이웃 학습(Nearest Neighbor Learning) 문제를 다루고 있습니다. 이 연구는 효율성을 극대화하면서도 정확한 출력을 유지할 수 있는 방법론을 제공합니다.

- **Technical Details**: Mojo는 Python과 호환되는 컴파일 언어로, 고빈도 금융 데이터에 대해 저지연 응답을 요구합니다. 본 논문에서는 분산 기반 분할(variable-based splitting) 및 연속적인 메모리 레이아웃을 결합한 Mojo SIMD k-d 트리를 제안합니다. 이 구조는 대규모 금융 시계열 코퍼스에 있는 후보 비교를 줄이면서도 하드웨어 수준의 처리량을 유지합니다. 연구에서는 k-최근접 이웃(KNN)의 정확한 반복 검색을 구현하여 성능을 향상시키는 방법론에 대해 설명합니다.

- **Performance Highlights**: 여덟 개의 금융 데이터 세트를 기반으로 진행된 실험에서는 Mojo를 이용한 방법이 scikit-learn의 k-d 트리보다 17.5에서 21.6배 더 빠른 성능을 달성했습니다. ARM64 기반의 데이터 세트에서는 28.1에서 43.5배의 속도 개선이 나타났습니다. 또한 Mojo를 이용한 새로운 모델이 옵션 데이터에 대해 10배의 학습을 가능하게 하여, 리스크 관리에 있어서 RMSE를 8 % 줄이는 성과를 보였습니다. 이러한 결과는 금융 AI의 효율성을 높인다는 점에서 중요성을 지닙니다.



### A Source Domain is All You Need: Source-Only Cross-OS Transfer Learning for APT Anomaly Detection via Semantic Alignment and Optimal Transpor (https://arxiv.org/abs/2606.10216)
- **What's New**: 이 논문은 정보의 부족과 극심한 클래스 불균형 문제로 인해 다른 운영 체제에서의 APT (Advanced Persistent Threat) 감지가 어려운 상황을 다룹니다. 특히 라벨이 없는 타겟 플랫폼에서 소스 플랫폼에 대한 라벨만을 사용하여 비정상적인 프로세스를 감지하는 방법을 연구합니다. 이를 위해 시스템 레벨의 provenance (출처) 추적 데이터를 사용하고, 새로운 OT (Optimal Transport) 기반의 프레임워크를 제안하여 대상 프로세스를 랭킹합니다.

- **Technical Details**: 이 연구는 프로세스 행동을 구조화된 자연어 설명으로 추상화하고, 이를 사전 훈련된 언어 모델을 사용하여 임베딩합니다. 또한, OT 기반의 barycentric (중심) anomaly score를 도입하여 타겟 임베딩을 소스-정상 매니폴드에 투영하고, 잔여 수송 불일치를 정량화합니다. 이 방법론은 여러 증거 채널을 결합하여 타겟 프로세스를 평가하며, 엔트로피 가중치, 방향 인식 및 밀도 인식 방식으로 노이즈와 낮은 밀도 행동을 포착합니다.

- **Performance Highlights**: DARPA Transparent Computing 데이터에 대한 평가 결과, 제안된 프레임워크가 소스 전용 비정상 탐지 기준선보다 ROC-AUC 및 nDCG 지표에서 향상된 성과를 보였습니다. 이는 타겟 도메인 감독 없이도 효과적인 APT 감지를 지원할 수 있음을 시사합니다. 검출된 비정상 프로세스는 MITRE ATT&CK 프레임워크의 적대적 전술과 일치하여, 제안된 접근 방식의 실용적 적용 가능성을 강화합니다.



### Automated Pronunciation Evaluation for Korean Toddler Speech using Speech Diarization and Self-Supervised Learning (https://arxiv.org/abs/2606.10213)
Comments:
          This paper will be presented at IEEE ICTs4ehealth in June, 2026

- **What's New**: 이 논문은 한국 어린이의 발음 평가를 위한 자동화된 파이프라인을 제시합니다. 2-5세 한국어 화자 아동의 리얼타임 음성 인식을 위해 IRB 승인된 53개의 음성 샘플 자료를 사용하였습니다. 최근 아동의 발음 문제를 해결하기 위한 신경망 기반의 음성 분리 및 자가 감독 학습 방식을 접목하여 혁신적인 시스템을 개발했습니다.

- **Technical Details**: 저자들은 3개의 다이얼리제이션(diarization) 모델을 평가하였으며, NeMo SortFormer로 88.69%의 화자 수 정확도와 33.04%의 다이얼리제이션 오류율(DER)을 기록했습니다. 이 모델은 아동과 보호자의 음성을 효과적으로 분리하기 위해 정렬된 변환기 아키텍처를 사용합니다. 또한, 자가 감독 학습(backbone) 모델이 다루는 자이언트 음성 데이터에 대해 비교하여 최적화된 결과를 도출했습니다.

- **Performance Highlights**: 크로스 모델 앙상블 방식을 통해 자음 예측에서 0.720, 모음 예측에서 0.845의 균형 잡힌 정확도를 달성하였고, 전체 평균은 0.782에 도달했습니다. 연구의 결과는 한국어의 조음 평가 시스템의 발전에 기여할 것으로 기대됩니다. 이 시스템은 현재의 수동적인 평가 방법에 비해 성장을 보여주고, 아동의 음성 장애 진단 개선에 기여할 가능성이 높습니다.



### Exploration of Foundation Model-Based Robots in Patient and Elderly Car (https://arxiv.org/abs/2606.10208)
- **What's New**: 이번 연구는 노인과 환자의 돌봄 요구가 증가하는 가운데, 페이퍼와 같은 기초 모델 기반 로봇이 이러한 요구를 충족하는 방향을 제시하고 있습니다. 연구에서는 대화 및 추론 기술로 사용되는 기초 모델의 설계 특징, 사용자 경험, 이와 관련된 성과에 대한 증거 등을 종합적으로 살펴보았습니다. 하지만 이러한 로봇 시스템에서 기술적 발전이 실제 돌봄 환경에서의 임상적인 영향을 가져올 수 있는지에 대한 명확한 기준이 부족하다는 점을 강조하고 있습니다.

- **Technical Details**: 노인 돌봄 로봇의 기초 모델은 다섯 가지 주요 역할에서 사용됩니다. 그중 가장 일반적인 역할은 사용자와의 개방형 대화를 생성하여 사회적 연결을 지원하는 것입니다. 구조화된 코칭이나 평가, 작업 흐름 조정 등의 기능도 포함되어 있으며, 이러한 시스템은 주로 음성 기반으로 작동합니다. 그러나 신뢰성과 안정성을 확보하기 위해서는 추가적인 안전 메커니즘과 사용자 요구를 반영한 상호작용이 필요합니다.

- **Performance Highlights**: 사용자 경험에 대한 조사에 따르면, 기초 모델 통합 이후 돌봄 로봇에 대한 수용도가 높아졌다는 결과가 도출되었습니다. 로봇의 응답이 더 일관되고 맥락 이해가 됨에 따라 사용자 써베이 점수가 상승하였음을 나타냅니다. 그러나 여전히 신뢰성과 안전성, 상호작용 부담 등의 요소가 사용자 수용에 중요한 영향을 미친다는 점이 아이러니하게도 개선이 필요하다고 할 수 있습니다.



### An Improved Generative Adversarial Network for Micro-Resistivity Imaging Logging Restoration (https://arxiv.org/abs/2606.10200)
Comments:
          7 pages, 9 figures

- **What's New**: 이 논문에서는 부분적으로 손상된 마이크로 저항 이미지 로그 이미지를 복원하기 위한 개선된 GAN 기반 이미지 복원 방법을 제안합니다. 이 방법은 FCN(골격망) 기반의 생성을 네트워크로 사용하며 깊이 분리형 합성곱 잔여 블록을 추가하여 픽셀 및 의미 정보를 보다 효과적으로 학습하고 유지합니다. 또한 이 네트워크의 다중 스케일 인지 필드를 확장하고 매개변수를 줄이기 위해 Inception 모듈을 추가하였습니다.

- **Technical Details**: 제안된 방법은 채널 주의 메커니즘과 잔여 블록을 결합하여 다중 스케일 특징 추출 기능을 구현합니다. 글로벌 판별 네트워크와 로컬 판별 네트워크를 설계하여 복원된 부분과 전체 이미지 간의 내용 및 의미 구조의 일관성을 점진적으로 개선합니다. 이 연구에서는 다양한 크기의 손실 영역을 가진 이미지 로그 이미지 다섯 세트를 테스트하여 구조 유사도 측정치가 평균 0.903에 도달하여 유사한 방법에 비해 약 0.3의 개선을 보여주는 성과를 거두었습니다.

- **Performance Highlights**: 이 방법은 손상된 마이크로 저항 이미지 로그 이미지를 복원하는 데에서 의미 구조 일관성과 텍스처 세부 정보를 크게 개선할 수 있음을 보여줍니다. 이로 인해 마이크로 저항 이미지 로그 이미지의 해석과 같은 후속 작업이 원활하게 진행될 수 있습니다. 제안된 기술은 석유 및 가스 산업의 결정을 위한 더 나은 저수지 특성과 분석을 지원할 수 있는 잠재력을 갖추고 있습니다.



### Density Ridge Selective Prediction for LLM and VLM Hallucination Detection under Calibration Label Scarcity (https://arxiv.org/abs/2606.10198)
- **What's New**: 이 논문은 대형 언어 모델(LLM) 및 비전-언어 모델에서 환각 탐지를 선택적 예측(selective prediction)으로 새롭게 구성하였습니다. 저자들은 새로운 리지 기반(score based on ridge) 평가 방법을 통해 기존의 여러 방법보다 저조한 레이블 환경에서도 높은 성능을 보이는 것을 보여주었습니다. 이 방법은 응답 매니폴드(response manifold)를 회복하고 이를 커널 밀도 추정(kernel density estimate)의 밀도 리지(density ridge)로 설명합니다.

- **Technical Details**: 저자들은 샘플링된 응답의 숨겨진 상태 거동을 분석하여, 높은 차원의 피쳐 맵을 통해 저차원의 리지 구조를 형성합니다. 각 생성 결과는 이 리지에 대한 근접성에 따라 점수가 매겨지며 이는 유클리드 거리의 부정값으로 계산됩니다. 또한, 저자들은 오프-리지 거리(off-ridge distance)를 사용하여 테스트 쿼리에 대한 신뢰 scores를 제안하며, 강한 타겟 밀도를 가진 샘플들에 대해 검증합니다.

- **Performance Highlights**: 제안된 방법은 Semantic Entropy, EigenScore, SAPLMA와 같은 기존 방법들과 비교하여 seven QA benchmarks에서 AUROC(Area Under Receiver Operating Characteristic curve) 점수에서 5-20점 높은 성능 향상을 기록했습니다. 또한, 레이블 희소성(calibration label scarcity) 하에서도 성능 저하가 제한적임을 확인하였으며, 이는 저자들이 제안한 리지 기반 점수의 유용성을 입증합니다.



### Integral Field Unit Spectroscopy with One Fiber (https://arxiv.org/abs/2606.10197)
Comments:
          Accepted for Conference on Physics and AI at Stanford University (PAI 2026)

- **What's New**: 본 논문은 다중 모형의 확률적 기초 모델을 소개하며, 이는 저광대역 이미지로부터 은하 내 임의의 공간적 위치에서 고해상도 스펙트럼을 예측합니다. Masked Autoencoder (MAE) 프레임워크 위에 구축된 이 모델은 섬유 배치와 붉은shift를 고려한 파장 인코딩을 포함하여 공간적으로 조정된 예측을 가능하게 합니다. 또한 470만 개의 이미지를 기반으로 훈련되어 IFU 데이터 없이 IFU와 유사한 능력을 발휘합니다.

- **Technical Details**: 데이터는 DESI Legacy Imaging Survey Data Release 9와 DESI DR1 스펙트럼을 교차 분석하여 나왔으며, 이 과정에서 각 객체별로 128x128 픽셀 이미지 컷아웃과 3600-9824 Å에 걸친 스펙트럼을 추출했습니다. 본 모델은 Transformer 기반의 masked autoencoder를 이용해 다양한 고유 위치 인코딩을 적용하고, 고정된 시각적인 2D 사인 곱 인코딩과 학습 가능한 채널 인코딩을 포함해 모델의 정확성을 높였습니다. 이러한 접근은 확률적 예측 오차와 입력 측정치를 동적으로 모델링하여 신뢰할 수 있는 재구성을 가능하게 합니다.

- **Performance Highlights**: 모델은 다양한 마스킹 비율에서 이미지와 스펙트럼 생성 작업에 대해 강력한 재구성 성능을 보였습니다. DESI 스펙트럼을 정확한 섬유 위치에서 재구성하여 산소 이중선 및 Ca H와 K 선과 같은 국소 구조를 재현하며, 이는 관측된 불확실성을 반영한 진정한 스펙트럼 신뢰도를 보여줍니다. 최종적으로, 예측된 공간적으로 해결된 방출선 맵은 MaNGA 데이터 릴리스 17의 독립적인 관측과 매우 근접하게 일치합니다.



### Fisher-Guided Progressive Parameter Selection for Adaptive Fine-Tuning (https://arxiv.org/abs/2606.10196)
- **What's New**: 최근 연구에서는 기존의 고정된 아키텍처 휴리스틱에 따라 훈련 가능한 매개변수 집합을 선택하는 대신, 동적이고 작업에 민감한 기준을 사용해야 할 필요성이 강조되고 있습니다. 이 논문에서는 FisherAdapTune이라는 새로운 프레임워크를 제안하여 Fisher 기하학의 시간적 변화 추적을 통해 매개변수 그룹을 점진적으로 선택합니다. 이를 통해, 안정화된 커브 기여를 가진 매개변수는 고정되어 오류 한계를 줄일 수 있음을 보여줍니다.

- **Technical Details**: FisherAdapTune은 PAC-Bayesian 관점에서의 이론적 기초를 바탕으로 합니다. Fisher Information Matrix (FIM)를 사용하여 매개변수의 민감도와 로컬 커브를 캡처하며, Jensen-Shannon 거리를 측정하여 매개변수 그룹의 진화를 추적합니다. 이러한 방법을 통해 데이터 주도적 신호를 제공하여 매개변수의 동적 선택 전략을 구현합니다.

- **Performance Highlights**: FisherAdapTune은 특정의 밀집 예측 작업에서 검증되었으며, 기존의 PEFT 기반과 비교하여 경쟁력 있는 성능을 달성하고, 분포 변화에 대한 강인성을 향상시켰습니다. 이 연구는 안정화된 매개변수를 고정함으로써 일반화 성능을 개선할 수 있음을 입증하며, 보다 효율적이고 작업에 민감한 매개변수 선택의 중요성을 강조합니다.



### MMClima: A Framework for Multimodal Climate Science Data and Evaluation (https://arxiv.org/abs/2606.10194)
- **What's New**: MMClima는 기후 과학 질문 응답을 위한 방대한 멀티모달 데이터셋으로, 10만 개 이상의 전문가 검증 질문-답변 쌍을 포함하고 있습니다. 이 프레임워크는 텍스트, 비디오 필사본 및 도표를 아우르는 다섯 가지 핵심 기후 과학 분야에 걸쳐 있습니다. MMClima는 자동화된 주장 추출 및 QA 합성에 인간 검증 과정을 포함하여 신뢰성과 규모를 확보했습니다.

- **Technical Details**: MMClima는 텍스트 및 비주얼 기반 질문을 통합하여 다양한 형식의 질문을 제시하는 통합 프로토콜을 제공합니다. 모든 항목은 단일 증거에 기반하여 설계되어, 세부적인 요구 사항을 충족할 수 있게 합니다. 또한, MMClima는 도메인 적응형 기준선으로 mmclima-70b-txt를 출시하여 강력한 텍스트 QA에서 탁월한 성능을 보여줍니다.

- **Performance Highlights**: MMClima는 28개의 LLM과 8개의 VLM을 표준화된 평가로 벤치마킹하여, 지식 및 신뢰성 측면에서 향상된 성과를 보였습니다. 기존의 기후 QA 벤치마크의 한계를 극복하며, 기후 과학 문서 이해 및 비주얼 해석에 강력한 기반을 제공합니다. 새로운 데이터셋 및 평가 파이프라인은 기후 과학을 위한 표준화된 멀티모달 평가를 지원합니다.



### Dropout-GRPO: Variational Stochasticity for Continuous Latent Reasoning (https://arxiv.org/abs/2606.10184)
- **What's New**: 이 논문에서는 Group Relative Policy Optimization (GRPO)을 활용하여 잠재 추론 모델에서 발생하는 구조적 문제를 해결하는 방법을 제안합니다. 특히, Coconut 모델의 결정론적 특성 때문에 다양한 실행 결과를 생성하지 못하는 문제를 해결하기 위한 스트럭쳐드 드롭아웃을 소개합니다. 이는 각 롤아웃에 대해 고정된 Bernoulli 마스크를 사용하여 필수적인 궤적 변동성을 생성하는 접근 방식입니다.

- **Technical Details**: 제안한 방법인 드롭아웃-GRPO는 공통 마스크를 사용하여 각 롤아웃을 베이지안 모델 평균 정책의 후방 샘플로 취급합니다. 이론적으로, 마스크 재생이 정확한 궤적에 대해 기울기를 계산하게 하고, 이는 잠재 상태에서의 정확한 정책 최적화를 가능하게 합니다. 더불어, 이 새로운 방법론은 제안된 방법의 비뚤림 및 분산 감소와 같은 이론적 정당성을 포함하여 실험적 검증을 제공합니다.

- **Performance Highlights**: GSM8K 데이터셋에서 드롭아웃-GRPO는 Coconut 모델의 성능을 27.29%에서 29.01%로 개선하여, 결정론적 롤아웃 GRPO가 실패하는 영역에서도 유효한 학습 신호를 생성함을 입증합니다. 이 연구는 잠재 추론 LLM 후 훈련에서 실용적이고 이론적으로 근거가 있는 접근 방식으로 자리매김 할 수 있음을 보여주고 있습니다.



### Making Time Editable in Video Diffusion Transformers (https://arxiv.org/abs/2606.10183)
- **What's New**: 이 논문에서는 새로운 Temporal-Control methodology를 제안하여 pretrained Diffusion Transformer (DiT)의 시간 편집 기능을 명확히 하였습니다. 이 방법은 motion speed와 temporal structure를 조정하는 데 중점을 두고 있으며, Backbone을 재설계하지 않고도 기존 모델의 생성 성능을 유지하며 확장할 수 있습니다. 이를 통해 모델이 시간의 진행 상황을 보다 신뢰성 있고 구조화된 방식으로 조정할 수 있게 됩니다.

- **Technical Details**: 제안된 방법은 pretrained DiT Backbone에 명시적인 temporal control 모듈을 추가하여 global motion pacing과 local temporal alignment를 분리합니다. 이 구조는 global FPS embedding과 latent-time embedding을 도입하여 장면의 발전 속도와 프레임 간 시간적 흐름을 조정합니다. 이러한 구성을 통해 모델은 시간의 변화를 보다 명확하게 표현할 수 있으며, 데이터에서 시간적 전통을 캡처하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 baseline과 비교할 때 Human Actions와 Natural Processes의 영상 생성에서 안정적인 단기 시간 전환을 달성했습니다. TA는 FPS가 훈련 레짐에 근접할 때 중간 동작 일관성을 향상시키며, FTTA는 자연 프로세스에서 최상의 시간 정규화를 달성하였습니다. 이들 결과는 명시적인 시간 조정이 다양한 DiT Backbone에 효과적으로 이전될 수 있음을 보여줍니다.



### Flow Control: Steering Vision-Language-Action Models with Simple Real-Time Inputs (https://arxiv.org/abs/2606.10180)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 연구에서는 시각-언어-행동(VLA) 모델의 흐름 제어(flow control) 방법을 소개합니다. 이 방법은 사용자가 입력한 키보드와 같은 일반적인 입력을 통해 VLA 행동을 실시간으로 조정할 수 있는 간단하고 효과적인 방식입니다. VLA는 이 입력을 사용하여 훈련 중 학습한 행동 전문가 분포에서 고품질의 행동 샘플을 생성하여 사용자 의도와 잘 맞도록 합니다.

- **Technical Details**: VLA는 카메라 관찰과 자연어 명령을 로봇 행동으로 변환하여 최신 성능을 달성하고 있습니다. 그러나 이러한 모델은 사용자 입력을 제대로 따르지 못하거나 새로운 객체와 장면에 일반화하지 못하는 등의 문제점이 있습니다. 흐름 제어는 사용자 입력을 사용하여 VLA 행동을 조정할 수 있도록 하며, 추가적인 정책 훈련 없이도 작동할 수 있습니다.

- **Performance Highlights**: 1616명의 사용자 실험 결과, 흐름 제어를 통해 사용자가 간단한 키보드 입력만으로도 작업 성공률과 완료 속도가 크게 향상되었음을 보여주었습니다. 또한, 흐름 제어 경로에서 π0.5의 미세 조정이 자율 정책 성능을 향상시키는 결과를 얻었습니다. 이 연구는 로봇 정책의 작업 결과를 개선하고 보조 로봇 공학 등 다양한 분야에 응용될 가능성을 제시합니다.



### Local Is Not a Sufficient Privacy Boundary: Governing OS-Integrated On-Device AI (https://arxiv.org/abs/2606.10173)
- **What's New**: 본 연구는 AI 시스템의 통합이 운영 체제의 프라이버시 문제를 복잡하게 만든다는 점을 강조합니다. 기존의 프라이버시 분석이 개별 서비스나 데이터 흐름에 초점을 맞추었다면, OS 통합 환경에서는 사용자 디바이스의 전반적인 환경을 고려해야 합니다. 이 논문은 프라이버시를 제도적 책임 문제로 다루는 OS 중심의 프라이버시 프레임워크를 개발합니다.

- **Technical Details**: 연구는 OS 중심의 위협 모델을 정의하고, 여섯 가지 프라이버시 위험 분류를 제안합니다. 이를 통해 로컬 AI가 어떻게 프라이버시 침해를 일으킬 수 있는지 설명하고, 이를 기술적 통제 및 거버넌스와 매핑하는 아키텍처를 도입합니다. 또한, 넓은 프라이버시 주장을 감사할 수 있는 네 가지 평가 기준을 제시하고, 이를 문서 기반 비교를 통해 실증적으로 보여줍니다.

- **Performance Highlights**: 본 연구의 기여는 OS 통합 AI가 사용자 정보와 컨텍스트를 어떻게 처리할 수 있는지에 대한 새로운 기준을 제시하는 것입니다. 구체적으로, 기술 아키텍처, 사용자 제어, 그리고 제도적 거버넌스가 실제로 어떻게 상호작용하는지를 규명합니다. 마지막으로, 효과적인 프라이버시 관리를 위해 정보 흐름 제약, 권한 제한, 사용자 제어 가능성, 감사 가능한 거버넌스가 필수적임을 강조합니다.



### Gaming AI-Assisted Peer Reviews Poses New Risks to the Scientific Community (https://arxiv.org/abs/2606.10159)
- **What's New**: 이 논문은 AI-mediated peer review(인공지능 기반 동료 검토)가 표면적인 개념의 재구성을 통해 저자들에 의해 쉽게 조작될 수 있다는 점을 밝혀냈습니다. 저자들은 단순한 추상 문구의 재구성을 통해 AI 리뷰의 결과를 상당히 개선시킬 수 있다는 것을 증명하였으며, 이러한 조작이 다양한 학문 분야와 출판 기관에서 적용될 수 있음을 보여주었습니다. 이로 인해 AI 평가 시스템의 약점이 드러났고, 저자들이 과학적 성과보다 AI 판별에 최적화된 원고를 작성하도록 유도할 수 있는 위험이 제기되었습니다.

- **Technical Details**: 이 연구에서는 AI 리뷰어의 평가 점수를 극대화하려는 공격이론을 제시했습니다. 저자들은 원고의 추상 문구를 재구성함으로써 AI 리뷰 평가를 부풀리는 반복 최적화 공격을 도입하고, Rewriting, Meaning-Preserving, Overclaiming 세 가지 전략을 사용하여 이를 입증했습니다. 이 기법은 인간 리뷰어의 평가 점수에 부정적인 영향을 미치지 않도록 특정 조건 하에 의미의 변화는 허용하면서도 과학적인 문서로서의 유창함을 유지하도록 설정되었습니다.

- **Performance Highlights**: 연구 결과, AI 리뷰어는 원고의 추상 문구에 대한 외부적 재구성에 매우 취약한 것으로 나타났습니다. 가장 강력한 공격은 약 38%의 성공률을 기록했으며, Gemini 3 Flash 리뷰어의 경우 수용 등급이 +1.31, GPT 5.4 Mini 리뷰어의 경우 +0.88로 높아졌습니다. 원래의 AI 리뷰에서 '거부'를 제안할 경우, 성공률은 50%를 넘겼습니다. 이러한 발견은 학술 커뮤니케이션의 미래 거버넌스에 대한 심각한 우려를 불러일으키며, AI 도구가 고위험 결정에서 신뢰할 수 있는 평가자로 간주되기 위해서는 체계적인 강건성 테스트와 투명한 보호 장치가 필요하다는 것을 나타냅니다.



### $τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems (https://arxiv.org/abs/2606.10156)
- **What's New**: 본 논문에서는 중요한 변화로, 추천 시스템(recommender systems)이 정적(single-turn) 모델에서 다중 턴(multi-turn) 대화 방식으로 발전하고 있음을 강조합니다. 이 연구는 특히 에이전틱(agentic) 추천 시스템의 평가 패러다임이 부족하다는 점을 지적하며, 새로운 벤치마크인 τ-Rec을 제안합니다. τ-Rec은 기존의 주관적인 평가를 대체하는 검증 가능한 보상(w rewards) 및 reveal-tagged elicitation (RTE) 메커니즘을 도입하여 대화 중 작업 제한(task constraints)을 제어합니다.

- **Technical Details**: τ-Rec은 에이전트가 사용자의 선호(ps preferences)를 대화 통해 이끌어내도록 평가하는 두 가지 역량, 즉 질문을 적절하게 하는 preference elicitation과 요구 사항을 충족시키기 위한 constrained reasoning을 동시에 시험합니다. 이 메트릭인 pass^k는 모든 독립적인 시도에서 작업을 올바르게 해결할 확률을 측정하며, 기존의 Recall@N 등과 같은 다른 메트릭에서는 포착할 수 없는 신뢰성(reliability) 차원을 드러냅니다. 데이터 카탈로그는 주요 LLM(reported large language models)들의 학습 컷오프 이후 TMDB의 영화로 구성되어 있습니다.

- **Performance Highlights**: 논문에서 평가한 여섯 가지 현대 모델(GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Flash, GPT-5 mini, DeepSeek V4 Flash 및 Qwen3-32B)의 성능은 인상적인 신뢰성 절벽(reliability cliff)을 드러냅니다. 가장 강력한 에이전트조차도 pass^1에서 약 57%의 성과를 달성하며, pass^4에서는 약 35%로 급감하는 구조입니다. 이는 현재 대화형 에이전트의 배포에서 주요한 격차를 나타내며, 많은 개선이 필요함을 부각시킵니다.



### BiWM: Advancing Open-Source Interactive Video World Models with Bidirectional Autoregression (https://arxiv.org/abs/2606.10135)
- **What's New**: 이 논문에서는 기존의 인과 모델(causal models)과 비교하여 Bidirectional Autoregressive 모델(BiWM)의 혁신적인 접근 방식을 제안합니다. BiWM은 상호작용하는 비디오 월드 모델(interactive video world models)을 구축할 수 있는 첫 번째 오픈소스 프레임워크로, 생성 품질(generation quality)과 추론 속도(inference speed)를 동시에 최적화합니다. 이 프레임워크는 두 단계의 훈련 스테이지만을 필요로 하여 복잡한 프로세스를 간소화하였습니다.

- **Technical Details**: BiWM은 사전 훈련된 비디오 백본(pretrained video backbone) 모델에서 카메라 제어를 주입함으로써 카메라 제어(fine-tuning)를 수행한 후, 몇 단계의 DMD(Distribution Matching Distillation) 과정을 통해 액션 및 카메라 제어가 가능한 월드 모델을 생성합니다. 이 접근법은 기존의 네 단계 과정(minWM)을 두 단계로 통합하여 전반적인 훈련 시간을 크게 단축시키며, 다양한 아키텍처와 모델에 걸쳐 적용 가능합니다.

- **Performance Highlights**: BiWM은 높은 신뢰성과 안정성을 제공하며, 기존의 인과 모델이 채택하는 접근 방식보다 훨씬 나은 비디오 이미지 품질을 나타냅니다. 이 시스템은 실제 카메라 제어를 가능하게 하여, 긴 탐색 거리(long horizon rollout)에서 플러그인 형식의 역사 압축(history compression) 메커니즘을 통합함으로써 자원 제한이 있는 연구 환경에서도 효과적으로 활용됩니다. BiWM은 논문에서 제시된 대로, 낮은 비용으로 기존의 Bidirectional 모델을 새로운 데이터 분포에 맞춰 쉽게 튜닝할 수 있는 능력을 갖추고 있습니다.



### Pareto-Guided Teacher Alignment for Fair Personalized Text Generation (https://arxiv.org/abs/2606.10126)
- **What's New**: 본 논문은 개인화된 설득 텍스트 생성(Personalized persuasive text generation)의 공정성을 개선하기 위한 새로운 연구 방향을 제시합니다. 인구 통계 기반의 불균형 문제를 해결하기 위해, 저자들은 다목적 정렬 문제(constrained multi-objective alignment problem)를 설정하였습니다. 이 연구에서는 개인화를 유지하면서 인구 통계적 차이를 줄일 수 있는 방법을 모색하였습니다.

- **Technical Details**: 제안된 프레임워크는 비선형 생성(candidate generation), 쌍 인식 가능성 차단(pair-aware feasibility gating), 그리고 파레토 방식(Pareto-style) 선택을 포함합니다. 또한 감독된 미세 조정(supervised fine-tuning)과 직접적인 선호 최적화(direct preference optimization)를 통해 개인화 개선을 위한 선택적 선호 최적화가 가능합니다. 연구는 기후 변화 및 백신 설득 과제를 사용하여, 성별과 연령이 일치하는 인구 통계적 그리드를 사용하여 평가하였습니다.

- **Performance Highlights**: 결과는 다양한 공정성 유지 전략이 모든 목표를 동시에 최적화하지 못함을 보여줍니다. 각 방법은 공정성과 개인화의 파레토 경계(Pareto frontier)에서 서로 다른 영역을 차지하며, 일부 방법은 불균형 감소를 더 효과적으로 달성하면서 다른 방법은 개인화 또는 인구 통계적 안정성을 더 잘 유지합니다. 연구 결과는 공정성을 감소시키기 위해 다목적 모델 선택(multi-audit model selection)의 중요성을 강조합니다.



### FedSteer: Taming Extreme Gradient Staleness in Federated Learning with Corrective Projections and Caching (https://arxiv.org/abs/2606.10124)
Comments:
          UAI 2026

- **What's New**: 이 논문에서는 Federated Learning (FL)에서의 집계 변동성을 줄이기 위한 새로운 방법인 FedSteer를 제안합니다. 기존 방법들은 비활성 클라이언트의 구식 업데이트를 재사용해 변동성을 줄이는 데 그쳤던 반면, FedSteer는 최근 클라이언트 그래디언트를 이용해 기울기 서브스페이스를 구축합니다. 이 서브스페이스는 현재 최적화 지형을 저차원으로 나타내며, 비활성 클라이언트를 위한 보다 정교한 업데이트를 생성합니다.

- **Technical Details**: FedSteer는 클라이언트가 활성화되었을 때, 해당 클라이언트의 진정한 그래디언트를 서브스페이스에 투영하여 최적의 좌표를 찾습니다. 비활성 클라이언트의 경우, FedSteer는 현재 서브스페이스를 기반으로 하여 기존 좌표를 재사용해 구식 그래디언트의 방향을 수정합니다. 이는 서버의 메모리 비용을 줄이기 위한 선택적 캐싱 전략과 결합되어, 작은 대표 클라이언트 서브셋을 형성합니다.

- **Performance Highlights**: 실험 결과, FedSteer는 기존 방법들에 비해 성능 유지와 같이 도전적인 시나리오에서 훈련 붕괴를 방지하면서도, 일부 경우에는 7% 이상의 정확도 향상을 이끌어냈습니다. 또한 FedSteer는 이전 방법들에 비해 메모리 오버헤드를 약 10배 줄이면서, 클라이언트의 구식 업데이트 방향을 수정할 수 있는 최초의 방법으로 자리 잡았습니다.



### MetaPlate: Counterfactual-Guided RAG-LLM Tool for Personalized Food Recommendation and Hyperglycemia Prevention (https://arxiv.org/abs/2606.10120)
- **What's New**: 이번 연구는 포만 후 고혈당(postprandial hyperglycemia)을 줄이기 위한 개인 맞춤형 식사 추천 시스템인 MetaPlate를 소개합니다. 기존의 고정적이고 일반화된 식이 권장사항이 아닌, 사용자의 생리적 데이터를 활용하여 개인화된 식사 제안을 제공하는 획기적인 접근 방식입니다. MetaPlate는 다중 데이터(모드) 분석을 통합하여 사용자의 식사 맥락을 이해하고, 이를 기반으로 글루코스 반응을 예측하여 행동 가능한 식사 조정을 제안합니다.

- **Technical Details**: MetaPlate는 연속 혈당 모니터링(Continuous Glucose Monitoring, CGM), 착용 장치에서 파생된 생리 신호 및 사용자 제공 식사 데이터를 통합하여 작동합니다. 머신러닝 모델은 식사 전 맥락을 예측하고, CF 최적화 모듈은 영양소 조정을 통해 글루코스 수준을 목표 범위 내에서 유지하도록 최적화합니다. 이 시스템은 USDA 식품 데이터베이스를 바탕으로 인간이 이해할 수 있는 식사 추천을 생성합니다.

- **Performance Highlights**: Expert-in-the-loop 평가로 진행된 연구 결과, MetaPlate의 추천 시스템이 실제적인 식사 제안, 적절한 양, 높은 추천 가능성을 향상시키는 것으로 나타났습니다. 전문가의 피드백에 따르면, 시스템의 출력이 임상적으로 불가능한 수준에서 실행 가능한 추천으로 변화했음을 보여 줍니다. 이러한 결과는 MetaPlate가 개인화된 식사 결정을 지원하는 유망한 도구임을 강조합니다.



### Emotion Profiling in LLM-Based Literary Translation: Systematic Shifts Across MT and Post-Editing (https://arxiv.org/abs/2606.10113)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 번역이 식별 가능한 감정 프로필을 나타내는지 여부를 조사하며, 후속 편집(post-editing)이 인간과 유사한 기준으로 이러한 감정 프로필을 어떻게 변화시키는지를 분석합니다. 마가렛 애트우드의 "Oryx and Crake" 소설의 LLM 번역과 그 후속 편집 버전 및 인간 번역을 대조하여 대규모 이탈리아 현대 공상과학 코퍼스를 사용하였습니다. 감정 분석은 어휘 기반 및 다국어 모델링 방식을 통해 수행했으며, 번역 시스템 간의 감정 변동을 세밀하게 분석하였습니다.

- **Technical Details**: 이 연구는 LLM과 NMT(신경 기계 번역) 시스템을 사용하여 감정 프로필을 정량적으로 분석하는 데 중점을 둡니다. 번역 품질을 평가하기 위해 전문적인 번역과 다양한 LLM에서 생성된 전후편집된 번역을 비교합니다. 연구에 사용된 두 가지 주요 어휘 자원은 이탈리아 감정 어휘 사전(Italian Emotive Lexicon)과 NRC 감정 어휘 사전(NRC Emotion Lexicon)으로 각각의 감정에 대한 단어 목록을 제공합니다. 이러한 방법으로 우리는 소스 텍스트와 번역된 텍스트 간의 감정 이동을 조사합니다.

- **Performance Highlights**: 연구 결과, MT 시스템이 고유한 감정 핑거프린트를 갖고 있으며, 감정 콘텐츠의 분포 및 강도의 차이로 나타납니다. 특히 후속 편집의 효과는 기반 MT 모델의 품질에 따라 크게 제한된다는 사실을 발견했습니다. 감정의 특이한 패턴은 통계적으로 유의미하며, 이는 출처 텍스트에서 저자의 정서적 목소리를 가장 충실하게 보존하고자 하는 지점입니다.



### Duality for Optimal Multi-Item, Multi-Bidder Auction Design: Revenue Certificates through Deep Learning (https://arxiv.org/abs/2606.10112)
- **What's New**: 본 논문은 다품목, 다입찰자 경매에 대한 최적 수익을 계산할 수 있는 새로운 프레임워크를 소개합니다. 그리드 기반의 유한한 문제에서 오는 계산을 개선하기 위해, 새로운 리프팅 기법을 개발하여 이론적 보장을 다룬 것이 특징입니다. 이 연구는 연속형 자산 분포에 대해서도 유효한 수익 상한선을 제공하는 기초를 마련하였습니다.

- **Technical Details**: 제안된 방법은 딥러닝(Deep Learning) 기술을 활용하여 Lagrange 승수를 파라미터화하고, 구조적으로 보장된 흐름 보존 속성을 갖는 신경망(Neural Networks)을 통해 최적화를 수행합니다. 이 방식은 DSIC(지배전략 인센티브 호환성) 조건을 만족하는 유한한 그리드에서의 문제를 해결하는 데 걸리는 자원을 절감합니다. 또한, 다양한 가치 분포에 대하여 유효한 리프팅 기법을 통해 연속형 경매 설계 문제로의 불연속적 환성을 이끌어냅니다.

- **Performance Highlights**: 제안된 프레임워크는 2×2 및 3×2 설정에서 GemNet에 의해 알려진 최상의 수익 대비 각각 1.8% 및 3.7% 이내의 인증된 상한선을 달성하였습니다. 이는 최적 설계 문제와 기존 솔루션 사이의 첫 번째 공식적 연결을 증명하며, 최신 기법이 기초적으로 근접 최적임을 보여줍니다. 또한, 초기 다입찰자, 다품목 최적 경매를 정확하게 회수함으로써 이 듀얼 프레임워크의 유효성을 입증하였습니다.



### What makes a harness a harness: necessary and sufficient conditions for an agent harness (https://arxiv.org/abs/2606.10106)
- **What's New**: 이 논문은 'agent harness'라는 용어의 명확한 정의를 제공하고, 소프트웨어 공학에서의 활용을 위한 포괄적인 분석을 제공합니다. 기존 문헌에서 이 용어가 불명확하게 사용되고 있었음을 지적하며, 다양한 클래스와 프레임워크, SDK 등이 포함될 수 있는 경계에 대해 설명합니다. 이를 통해 'agent harness'의 필요성과 그것이 소프트웨어 제작 및 연구에 어떻게 기여할 수 있는지를 강조합니다.

- **Technical Details**: 연구는 개념 분석을 통해 'agent harness'의 정의를 재구성하며, 이를 포함 및 제외 테스트로 운영화합니다. 논문은 6개의 실제 예시(Claude Code, Codex CLI 등)를 통해 정의를 검증하고, 정의가 지속적으로 일관되게 적용됨을 보여줍니다. 저자는 'agent harness'가 신뢰할 수 있는 시스템을 판단하기 위한 기준을 마련해야 한다고 강조합니다. 또한 이 개념이 소프트웨어 에이전트와 머신러닝 평가의 발전에 기여할 수 있음을 시사합니다.

- **Performance Highlights**: 정의된 'agent harness'는 명확한 용어 집합과 함께, 시스템 간의 비교 및 설계 우수성과 관련된 지식을 축적하는 데 기여할 수 있습니다. 개념의 명확성과 더불어 연구를 통해 발견된 경계는 엔지니어링 실무에서 도움이 될 수 있는 방향으로 발전할 수 있을 것입니다. 연구 의제는 다루어지는 디자인 긴장 축을 기반으로 조직되며, 향후 연구 방향을 제시합니다.



### Unsupervised Style Representation Learning for AI-Text Detection via Paraphrase Inversion (https://arxiv.org/abs/2606.10099)
- **What's New**: 이번 연구에서는 저작권 라벨 없이 스타일 표현을 학습하는 방법을 제안합니다. 특히 스타일 인코더(style encoder)가 기계 생성 패러프레이즈에서 인간이 작성한 텍스트를 복원하는 방식으로 학습됩니다. 이를 통해 현재 스타일 기반 탐지기의 한계를 극복하고, 제로샷(zero-shot) 상황에서도 효과적으로 작동할 수 있는 탐지기를 도입합니다.

- **Technical Details**: 우리는 패러프레이즈 변환 과제를 통해 스타일 특성을 학습합니다. LUSR(비지도 스타일 표현 학습) 모델은 기계 생성 패러프레이즈로부터 인간 작성 텍스트의 회복을 목표로 하며, 이때 의미를 담고 있는 인코더는 고정되어 비문맥적 스타일 특성만을 캡처하도록 설정됩니다. 두 가지 탐지 전략, 즉 소수 샷(few-shot) 탐지기와 제로 샷(Zero-shot) DeepSVDD 기반 탐지기로 배운 표현을 평가합니다.

- **Performance Highlights**: LUSR 방식은 M4 및 MAGE 벤치마크에서 테스트 결과, 스타일 표현에 있어 가장 우수한 소수 샷 탐지기를 제공하며, 제로 샷 환경에서도 전체 감독 분류기와 경쟁력을 갖춥니다. 이 과정에서 작업별 감독 없이도 저자 확인(authorship verification) 및 세밀한 스타일 구분(fine-grained style discrimination) 작업에 대해 경쟁력 있는 성능을 보여주었습니다.



### A Theory on Flow Matching with Neural Networks (https://arxiv.org/abs/2606.10089)
- **What's New**: 이 연구에서는 신경망을 매개변수화한 조건속도장(conditional velocity fields)을 이용한 흐름 일치(flow matching)에 대한 이론적 기반을 개발합니다. 두 계층의 ReLU 신경망 모델에서의 경량학습을 위한 수렴 보장을 설정했으며, 조건속도장 일치 목표에 대해 일반화 경계를 도출하였습니다. 이 결과를 바탕으로 유도된 흐름이 생성한 샘플들에 대한 Wasserstein 거리 보장을 제공합니다.

- **Technical Details**: 우리의 분석은 고차원 조건속도장을 근사화하기 위한 경량학습의 수렴 보장과 함께, 신경 접선 커널(neural tangent kernel, NTK) 체계와의 관련성을 다룹니다. 우리는 한정되지 않은 손실을 가진 다중 과제 학습에 대한 일반화 경계를 도입하여 속도장 학습 문제에 대한 L2 통계적 오류 경계를 유도하였습니다. 또한, Wasserstein 거리에서 흐름 일치의 샘플링 오류 보장을 제공하며, 이는 연속 시간의 ODE 동역학에 대한 세밀한 분석을 포함합니다.

- **Performance Highlights**: 이론적 결과는 합성 데이터와 실제 이미지 기준에서 수행된 광범위한 실험을 통해 검증되었습니다. 실험들은 수렴과 샘플링 오류 경계를 확인하며, 이론적 발견의 실용적 함의를 통찰하는 데 도움을 줍니다. 이러한 결과는 미래 연구의 잠재적 방향을 제시합니다.



### Divide-and-Conquer Modeling for the CTF-4-Science Lorenz Benchmark (https://arxiv.org/abs/2606.10084)
- **What's New**: 이번 연구는 CTF-4-Science Lorenz 벤치마크에 대한 분할-정복(divide-and-conquer) 모델링 전략을 제시합니다. 이 벤치마크는 소음을 가지고 있는 예측, 몇 개의 샷 학습(few-shot learning) 등 여러 시나리오에서 혼합된 혼란 시스템 예측 성능을 평가합니다. 각 예측 블록은 관련 과제 그룹의 평가 행동에 따라 최적화된 모델을 할당하여 효율성을 극대화했습니다.

- **Technical Details**: Lorenz 시스템은 세 가지 차원을 가진 혼란 시스템이며, 데이터 기반 예측을 위한 스트레스 테스트로 알려져 있습니다. 이 연구에서는 각 예측 작업에 대해 서로 다른 입력 데이터 레짐(data regime)을 고려하여 모델 선택을 최적화했습니다. 주목할 만한 모델로는 NG-RC/NVAR가 있으며, 이는 다음 단계 전이를 안정적으로 예측하는 데 효과적입니다.

- **Performance Highlights**: 최종 시스템은 79.63의 공공 점수를 기록하며, 한 모델 클래스를 모든 레짐에 적용하는 것보다 특정 시나리오에 맞춘 업데이트 방식이 성능을 개선한다는 것을 입증했습니다. 다양한 하이퍼파라미터 조정과 예측 블록의 선택적 재훈련을 통해 점수가 더욱 향상되었습니다. 이러한 접근법은 혼돈 예측 벤치마크에서 우수한 성과를 보여줍니다.



### VFUSE: Virulent Feature Understanding with Sparse autoEncoders (https://arxiv.org/abs/2606.10080)
- **What's New**: 본 연구에서는 위험을 인식할 수 있는 특성을 감사하기 위해 VFUSE (Virulent Feature Understanding with Sparse autoEncoders)라는 기계적 해석 가능성 접근 방식을 소개합니다. 이는 확산-변환기(diffusion-transformer) 활성화에 대한 희소 오토인코더(Sparse autoEncoders, SAE)를 훈련시키며, 이는 RoseTTAFold3와 RFDiffusion3의 위험 요소를 효과적으로 감지할 수 있습니다. VFUSE는 단지 모델의 성능을 희생하지 않으면서 해석 가능성을 향상시키는데 기여합니다.

- **Technical Details**: SAE는 비선형 활성화 함수와 선형 디코더를 가진 인코더로, 재구성 손실을 최소화하기 위해 훈련됩니다. 이 연구에서 사용된 Matryoshka SAE 구조는 서로 다른 접두사(prefix) 크기를 고정하여 훈련하며, 각 접두사는 이전 접두사에서 학습한 특징을 통합하여 재구성을 정교하게 만드는 방식입니다. 연구팀은 RFD3와 RF3에서 활성화된 특성을 수집하고, 서로 다른 블록을 통해 SAEs가 원래 모델보다 더 나은 성능을 보인다는 것을 발견했습니다.

- **Performance Highlights**: 특정 블록에서의 선형 프로브(linear probes)는 SAE의 잠재 공간(latent space)에서 적합할 때 유해한 디자인을 훨씬 더 잘 탐지할 수 있음을 보여줍니다. 연구 결과, AUROC(Area Under Receiver Operating Characteristic) 0.84를 달성하여 단일 유해 디자인에 대한 특성을 정의할 수 있습니다. 이는 모든 원자 확산 모델에 대해 최초로 훈련된 SAE이며, 안전한 단백질 디자인을 위한 중요한 첫걸음을 내딛는 것입니다.



### Temporal Sheaf Neural Networks with Dynamic Orthogonal Transpor (https://arxiv.org/abs/2606.10071)
- **What's New**: 새로운 연구에서는 Temporal Sheaf Neural Networks (TSNN)을 소개하며, 이는 시간에 따라 변하는 orthogonal frame을 각 노드에 부여하며 지역 좌표 시스템 간의 명시적 이동 후에만 노드 상태를 비교하는 구조입니다. 기존의 지속적인 시간 그래프 모델들이 공유된 전역 임베딩 공간에서 작용하는 것과 달리, TSNN은 동적인 지역 프레임을 통해 노드 특화된 상호작용 의미를 모델링합니다. 이 모델은 Householder 곱으로 매개변수화된 각 노드의 프레임을 효율적으로 처리하며, 프레임 업데이트 시 저장된 은닉 상태를 정확히 유지합니다.

- **Technical Details**: TSNN의 특징으로는 각 노드가 Householder 곱을 사용하여 매개변수화된 시간 변화하는 orthogonal frame을 갖는다는 점입니다. 이 뼈대는 면역력이 강한 숨겨진 상태의 보존을 보장하며, 이동된 거리 기반으로 예측을 앵커링하는 geometric-residual decoder를 사용합니다. 모든 계산은 이전 이벤트 기록만을 이용하여 엄격하게 인과적이며, 이는 TSNN이 변화하는 로컬 좌표 속에서도 의미를 지속적으로 보장한다는 것을 의미합니다.

- **Performance Highlights**: TSNN은 TGB v2 링크 예측과 시간 이종 평가판 리더보드 및 DGB 벤치마크에서 가장 강력한 이전의 방법들과 비슷하거나 이를 초과하는 성능을 선보였습니다. 특히 노드 역할 이종성이 강한 그래프에서 가장 큰 개선이 있은 것으로 보고되었습니다. 동적 프레임, 정규 운송, 기하학적-잔여 디코딩의 역할을 검증하는 ablation 연구도 이 모델의 독특한 장점을 확인합니다.



### Importance-Aware Scheduling for High-Dimensional Hyperparameter Optimization (https://arxiv.org/abs/2606.10068)
Comments:
          8 pages, 5 figures. Accepted to IJCNN 2026

- **What's New**: 이번 논문은 Hyperparameter Optimization (HPO)의 효율성을 높이기 위한 새로운 전략, Greedy Importance First (GIF)를 제안합니다. GIF는 작은 샘플을 활용하여 하이퍼파라미터의 중요성을 추정하고, 이를 바탕으로 그룹을 형성하여 비율에 따라 실험을 할당하는 방식입니다. 이 연구는 GIF가 전통적인 최적화 방법보다 빠른 수렴 속도를 보이며, 고차원 공간에서의 평가 효율성을 증대시키는 데 기여한다는 점에 중점을 두고 있습니다.

- **Technical Details**: GIF는 우선 작은 샘플을 사용하여 HIA (Hyperparameter Importance Assessment) 알고리즘을 통해 하이퍼파라미터의 중요성을 평가합니다. 이후, 중요도에 따라 하이퍼파라미터를 순서대로 정렬하고, 해당 그룹으로 예산을 비례하여 할당합니다. 이때, 최적화를 진행하지 않은 변수들은 현재의 최적값으로 고정하며, 진행 과정에서 개선이 없을 경우 전체 공간으로 되돌아가는 기능을 포함하여 글로벌 탐색을 복원하는 구조입니다.

- **Performance Highlights**: GIF는 정해진 평가 예산 아래에서 다섯 가지 비등방성 분석 함수와 NAS-Bench-301을 통해 검증되었습니다. 실험 결과, GIF는 TPE, BOHB, 랜덤 서치 및 순차 그룹화와 비교하여 더 나은 성능을 보였으며, 특히 고차원 벤치마크에서 더 빠른 수렴을 기록했습니다. 중요한 요소 분석 결과, 중요성 추정, 비례 할당 및 글로벌 탐색 복원 단계가 성능 향상에 기여했음을 확인했습니다.



### A Controlled Audit of Pretraining Contamination in Public Medical Vision-Language Benchmarks (https://arxiv.org/abs/2606.10066)
Comments:
          30 pages, 7 figures, 9 tables. Preprint

- **What's New**: 이번 연구는 의료 비전-언어 모델(vision-language models, VLM)에서의 데이터 누출을 감사하는 것을 목표로 합니다. 기존의 연구들은 일반적인 VLM에서의 누출 문제를 다루었으나, 의료 분야에서는 이러한 감사가 전무했습니다. 연구팀은 SLAKE-En, PathVQA, VQA-RAD와 같은 대규모 공개 벤치마크를 사용하여 의료 VLM의 혐의 있는 전이 문제를 분석했습니다.

- **Technical Details**: 감사 과정에서는 네 가지 검출기(detector)를 사용하여 이미지 및 텍스트 데이터에서 누출 신호를 측정했습니다. 특정 패턴으로 이미지의 중복을 측정한 결과, SLAKE-En의 19.8% 이미지는 PMC-OA-beta와 중복되었으며, 이는 같은 환자에 대한 것이라기보다 비슷한 위치에서의 교차-모델 패턴으로 해석됩니다. 텍스트 측면에서도 Qwen2.5-VL의 교환 가능성이 입증되었지만, BLIP-2는 누출 신호가 없었습니다.

- **Performance Highlights**: 제출된 감사 결과는 이미지 측면에서의 두 개의 검출기가 신뢰할 수 없음을 나타냈습니다. 또한, BLIP-2와 같은 외부 기준선에 대한 성능 비교를 통해 얻은 신호들은 검사하였던 모델의 특수성 및 훈련 데이터에 따른 영향을 분리하는 데 중요했습니다. 최종적으로, 이 연구는 소규모 의료 VLM 집단에서의 개별 데이터에 대한 기억 추론이 신뢰할 수 없다는 결론을 내렸습니다.



### Bittensor Agent Arenas as a Trajectory Primitive: Distilling a Shopping Agent from ShoppingBench Subnet Traces (https://arxiv.org/abs/2606.10064)
Comments:
          10 pages, 4 figures, Data and Models available at: this https URL

- **What's New**: 이 연구는 작은 모델의 에이전트 후 학습(pos-training)에서 경과의 질이 알고리즘보다 중요한 요소임을 강조합니다. 기존 연구에 비해, 본 연구에서는 에이전트가 더 나은 성능을 발휘할 수 있도록 하는 경로(trajectory) 데이터 생성기를 설계하여 효과적인 학습 환경을 구축했습니다. 이를 통해 SN15라는 새로운 벤치마크 데이터셋에 대해 여러 특성을 갖춘 경로 데이터를 생성할 수 있음을 보여주었습니다.

- **Technical Details**: 새로운 데이터셋 SN15는 에이전트가 다양한 요구 사항을 충족하는 쇼핑 추천을 하도록 평가하며, 세 가지 주요 특징을 가지도록 설계되었습니다: (i) 인센티브 정렬 다양성(incentive-aligned diversity), (ii) 경로별 평가(per-trajectory judging), (iii) 회전 효과 함수(rotating held-out problems)입니다. 연구진은 구조적 품질 필터(structural-quality filter)를 제안하여 원시 경로 데이터를 학습 가능한 데이터셋으로 변환했습니다.

- **Performance Highlights**: 본 연구에서 Qwen3-4B 모델은 구축한 데이터셋을 기반으로 사후 훈련을 실시하였고, 그 결과는 18.0%에서 42.7%로 성능이 향상되었습니다. 이 성능은 기존의 합성 데이터 SFT-만의 기준값 43.6%와 근접합니다. 또한, 감독 학습에서의 큰 성과 차이(p@8 vs p@1)로 인해 Dr. GRPO를 활용한 향상이 가능하며, 후속 연구에서는 준 과제의 데이터 흐름이 성능 갭을 줄이는 주요 요소임을 밝혔습니다.



### Inside the Latent Flow: Causal Deciphering of Attention Dynamics in Audio Separation Foundation Models (https://arxiv.org/abs/2606.10046)
- **What's New**: 이 논문에서는 Flow-matching transformers의 강력한 오디오 분리를 달성하면서 그 내부 메커니즘을 검토합니다. 특히, SAM Audio 모델에 대한 결정론적 프로빙 프로토콜을 적용하여 입력 텍스트의 조건화 메커니즘을 해독합니다. 새로운 방식으로 발견된 쌍 경로 텍스트 조건화 메커니즘은 점진적으로 의미적 정체성을 제어하고, 교차 주의력을 통해 음향 구조를 정제합니다.

- **Technical Details**: 연구진은 주의(attention) 메커니즘을 설명하기 위해 인과적 개입 원칙을 적용하여 오디오 확산(transformer) 모델의 다양한 구조적 의존성을 분석합니다. 이 프로빙 프레임워크는 중간 표현을 조작하여, 사전 훈련된 가중치를 변경하지 않고도 분산 변환기의 메커니즘을 이해하는 데 기여합니다. 실험을 통해 교차 주의력(cross-attention)과 잔여확장(additive injection)에 대해 설명하고, 레이어 선택적 주의 캐싱(Layer-Selective Attention Caching, LSAC)이라는 새로운 방법론을 제안합니다.

- **Performance Highlights**: 제안된 LSAC 방법론은 약 25%의 자기 주의(self-attention) 계산을 줄이면서도 거의 품질 손실 없이 분리 정확도를 높입니다. 이 접근 방식은 다양한 음향 복잡성에 대해 최대 6.7배의 품질 유지를 보장하며, 기존의 단순 단계 축소 방식보다 더 효과적입니다. 마지막으로, 이 방법은 3B 파라미터 모델까지 확장 가능하며, 모델의 안정성을 유지하는 데 크게 기여하는 것으로 나타났습니다.



### Interpreting and Steering a Text-to-Speech Language Model with Sparse Autoencoders (https://arxiv.org/abs/2606.10029)
- **What's New**: 이번 논문은 텍스트-투-스피치(TTS) 시스템의 언어 모델을 기반으로 하는 새로운 해석 가능성 모형인 BatchTopK sparse autoencoders (SAE)를 소개합니다. 기존의 TTS 모델은 텍스트 프리픽스와 생성된 음성 토큰을 사용하는 혼합 시퀀스를 처리하지만, 이러한 두 데이터 간의 상호 작용을 명확히 이해하지 못하고 있었습니다. 새로운 자동 해석(auto-interp) 파이프라인을 통해 각 특성이 텍스트 프리픽스 맥락, 1초 음성 클립 또는 두 가지 모두에서 활성화되는 위치를 레이블링하여 해당 특성을 해석하고 제어할 수 있는 가능성을 제시하였습니다.

- **Technical Details**: 이 연구는 CosyVoice3라는 TTS 시스템의 LM(언어 모델) 뼈대를 기반으로 250M 토큰에 대해 BatchTopK SAEs를 훈련시킵니다. 훈련 과정에서, 주요 활성화는 텍스트, 오디오 또는 혼합 증거의 강도에 따라 경로화되며, 각 특성은 언어 프리픽스 또는 음성 토큰 중 어느 쪽에서 활성화되는지를 통해 분류됩니다. 이러한 방식으로 각 층에서 음성(modal)과 텍스트(modal) 특성을 효과적으로 분석하고, 계층별로 심층적인 해석을 가능하게 합니다.

- **Performance Highlights**: 결과적으로, SAEs의 활성화는 웃음 확률을 0.02에서 0.79로 증가시키고, 화자 성별을 전환하며, 음성 발화 속도를 제어할 수 있는 것으로 나타났습니다. 레이어-20 사례 연구에서 텍스트-모달 레이블의 확인이 가장 용이하며(AUROC 0.921), 오디오-모달 및 혼합 특성은 점차적으로 어려운 결과를 보였습니다. 이 연구를 통해 TTS 합성을 위한 결정형 제어가 가능하다는 것을 입증하였고, 다양한 음성 특성을 조정하는 데 중요한 기초 자료를 제공하게 되었습니다.



### Generalized-CVO: Fast and Correspondence-Free Local Point Cloud Registration with Second Order Riemannian Optimization (https://arxiv.org/abs/2606.10019)
Comments:
          16 pages, 12 figures

- **What's New**: 본 논문에서는 기하학적 표면 구조와 재생 커널 Hilbert 공간(RKHS) 임베딩을 활용한 빠르고 대응 없는(local point cloud registration) 지역 포인트 클라우드 등록 방법을 제안합니다. 이 방법은 점구름을 연속 함수로 표현하며, 점 별 비등방성 커널을 통해 지역 기하학을 인코딩합니다. 이러한 공식화는 표면 법선에 대한 정렬을 개선하고 접선 방향으로의 정렬을 완화합니다.

- **Technical Details**: 문제 해결을 위해 근사 리만 헤세안(approximate Riemannian Hessians)을 적용한 이차(on-manifold) 최적화 방법을 제안하며, 이전의 대응 없는 RKHS 기반 방법들에서 사용된 일차 솔버들에 비해 최대 10배의 속도 향상을 달성합니다. 이 방법은 주로 LiDAR와 RGB-D 데이터를 통한 프레임 간 추적 정확도 향상에 효과적입니다. 이차 최적화 기법은 복잡한 기하학적 형태의 효율성을 극대화합니다.

- **Performance Highlights**: 주행 분야의 LiDAR 추적 등록 작업에서, 도전적인 특성 희소 환경에서 변환드리프트(translational drift)와 회전드리프트(rotational drift)가 55% 이상 감소하는 성과를 보였습니다. 또한 ICP 기반 방법들에 비해 객체 등록 벤치마크에서 더욱 향상된 견고성을 입증했으며, 특히 중간 정도의 비정렬(misalignment) 상황에서 글로벌 초기화(refining global initialization) 시 더욱 성능이 개선되는 결과를 보여주었습니다.



### DeRA-MOS: Optimizing Text-to-Music Evaluation via Decoupled Listwise Ranking and Modality Alignmen (https://arxiv.org/abs/2606.10010)
Comments:
          Accepted to IEEE Signal Processing Letters (SPL)

- **What's New**: 이번 연구에서는 텍스트-음악 생성(text-to-music, TTM) 시스템의 평가를 위한 새로운 프레임워크인 DeRA-MOS를 제안합니다. 기존의 평균 의견 점수(mean opinion scores, MOS)에 의존하는 수동 평가의 한계를 극복하고, 결정 순서와 관련된 점수를 최적화하는 방법론을 제공합니다. 새로운 Gobal Ranking (SRCC)에 최적화된 Batch-Aware Listwise Ranking (BALR) 손실과 오디오와 텍스트 간의 유사도를 정규화하는 Score-Anchored Modality Alignment (SAMA) 손실을 도입하여, TTM의 평가에서 효과적인 개선을 도모했습니다.

- **Technical Details**: DeRA-MOS 프레임워크는 TTM 평가의 두 가지 주요 측정 지표인 음악 인상(music impression, MI)과 텍스트 정렬(text alignment, TA)을 최적화하는 구조로 이루어져 있습니다. MI에 대해서는 미니배치 내의 상대적 순서를 모델링하기 위해 Batch-Aware Listwise Ranking 손실을 적용하였으며, 이는 Spearman 순위 상관계수(SRCC)와 더 잘 일치합니다. TA를 위해서는 인간 점수에 맞춰 오디오-텍스트 유사도를 맵핑하는 Score-Anchored Modality Alignment 손실을 도입하여 라틴 공간을 정규화합니다.

- **Performance Highlights**: MusicEval 데이터셋에서 실험한 결과, DeRA-MOS는 MI와 TA 모두에서 점수 순위 지표에서 기존의 점 대 점(point-wise) 방법론보다 우수한 성능을 나타냈습니다. 본 연구의 기법은 TTM 평가의 안정성과 일관성을 높이며, 대규모 평가 시스템에 적합한 강력한 패러다임을 설정하였습니다. 이 성과는 TTM 개발의 확장 가능성을 더욱 높이고, 자동화된 평가를 실현하는 기반을 마련하는 데 기여할 것입니다.



### Geometry-Aware Anisotropic Boundary Correction for Aerodynamic Simulation (https://arxiv.org/abs/2606.09963)
- **What's New**: 이번 연구에서는 전통적인 수치 해석 방법의 비용 문제를 해결하기 위해 GeoABC라는 새로운 경계 보정 프레임워크를 제안합니다. GeoABC는 경계 기하학을 활용하여 물리적 예측을 향상시키며, 경계의 비등방적 특성을 명시적으로 모델링합니다. 이는 기존 신경 연산자 모델의 한계를 극복하고 하늘향 (high-fidelity) 공기역학 시뮬레이션을 가능하게 합니다.

- **Technical Details**: GeoABC는 경계 기하학으로부터 획득된 지역적 탄젠트-정규 구조를 명시적으로 구성하여, 경계와 일반 경향 간의 상호작용을 고려합니다. 이 과정은 신경 연산자의 중간 표현 공간에서 방향 인지 경계 보정을 수행하게 하며, 기하학을 정적 입력 특징으로 사용하는 대신 스타컵과 같은 유연한 방법으로 결합합니다. 이를 통해 신경 연산자가 근접 벽(prediction near-wall) 예측에서 더욱 정확한 물리적 예측을 가능하게 합니다.

- **Performance Highlights**: GeoABC는 2D 에어포일 및 3D 자동차 과제로 여러 신경 연산자 백본에서 근접 벽 예측 오류를 평균 38% 감소시킵니다. 이는 기존 신경 연산자들이 가진 근접 벽 간극을 축소하고, 항공기 및 차량 디자인의 신뢰성을 높입니다. 따라서 GeoABC는 공기역학적인 성능 평가에 있어 매우 중요한 역할을 하게 됩니다.



### Optimality of FSQ Tokens for Continuous Diffusion for Categorical Data with Application to Text-to-Speech (https://arxiv.org/abs/2606.09962)
- **What's New**: 이 논문에서는 범주형 데이터에 대한 연속 확산 (Continuous Diffusion) 모델을 개발하고 그 특성을 분석하였습니다. 이는 자율 회귀 대형 언어 모델 (autoregressive large language models)에 대한 효과적인 대안을 찾으려는 연구자들의 관심이 높아짐에 따라 이루어졌습니다. 특히, FSQ (Finite Scalar Quantization) 토큰화 방식이 연속 확산을 위한 최적의 적합성을 가지는 구조를 해석하고, 실제 시나리오에서 우수한 성능을 입증하였습니다.

- **Technical Details**: 본 연구에서는 확산 경로의 Kullback-Leibler 발산 (KL divergence)을 사용하여 범주형 데이터의 토큰을 표현하는 잠재 공간의 구조를 이해합니다. CDCD (Continuous Diffusion for Categorical Data) 모델의 훈련 시, 매개변수 및 경로 간의 KL 발산을 최적화하는 연결성을 Establish 하였으며, FSQ 코드북이 이러한 구조에서 최적성을 갖는다고 주장하고 실험을 통해 이를 검증했습니다. 이와 함께, FSQ를 활용하여 오디오 재생과 같은 실제 어플리케이션에서의 성능을 개선한 사례를 제시했습니다.

- **Performance Highlights**: FSQ 기반의 CDCD 모델은 기존의 LLM 기반 모델과 비교해 음성 이해도 및 제로샷 (zero-shot) 능력에서 뛰어난 성능을 보였습니다. 특히, 이 모델은 경쟁자의 LLM 알고리즘에 비해 약 1010배 작은 사이즈와 55배 빠른 처리 속도를 자랑합니다. 논문에서는 이러한 성과를 뒷받침하는 수치적 실험 결과와 이론적 발견을 제시하였습니다.



### 3SPO: State-Score-Supervised Policy Optimization for LLM Agents (https://arxiv.org/abs/2606.09961)
- **What's New**: 이 논문에서는 State-Score-Supervised Policy Optimization (3SPO)라는 새로운 RL 알고리즘을 제안합니다. 기존의 강화학습(RL) 알고리즘이 에피소드의 전체 롤아웃을 요구하는 것과 달리, 3SPO는 동적 상태 점수(supervision) 기반으로 각 단계에서 정책 최적화를 수행합니다. 이를 통해 희귀하고 지연된 보상이 있는 다단계 에이전트 환경에서의 성능을 극대화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 3SPO는 실패 확률이 낮고 배운 가능한 병목 상태를 판별하기 위해 각 상태의 성공률에 따라 상태 점수를 계산합니다. 이 알고리즘은 가치 함수 추정이나 추가 모델 없이도 단계별 정책 최적화를 가능하게 하며, 전이 수준의 크레딧 할당과 샘플링 우선 순위 결정에 사용됩니다. 이 기법은 에피소드 전체가 완료되기 전에 정책 업데이트를 수행할 수 있도록 하여 보다 신속하고 효율적인 학습을 지원합니다.

- **Performance Highlights**: ALFWorld와 WebShop에서의 실험 결과, 3SPO는 기존의 GRPO 대비 각각 22.6% 및 15.6 포인트 더 높은 성능을 나타냈습니다. 또한, 3SPO는 비슷한 자원을 사용하면서도 상태 탐색을 2.4배 더 늘리고 수렴 속도를 1.8배 빠르게 했습니다. 이러한 결과는 3SPO가 다단계 에이전트 환경에서 우수한 성능을 보인다는 것을 보여줍니다.



### HydraCIL: Decoupled Class-Incremental Learning through Prototype-Guided Multi-Head Classifiers (https://arxiv.org/abs/2606.09960)
Comments:
          Accepted for publication at the International Joint Conference on Neural Networks (IJCNN 2026)

- **What's New**: HydraCIL은 프로토타입 안내 멀티 헤드 분류기를 기반으로 한 분리형 지속적 학습 모델입니다. 이 모델은 임베디드 및 자원 제약 환경에서 지속 가능한 배포를 목표로 하며, 기존의 Class-Incremental Learning (CIL) 방법을 개선합니다. HydraCIL은 백본(backbone)을 동결하고 피처 추출(feature extraction)과 학습을 분리함으로써 훈련 시간을 크게 줄이고 에너지 소모를 최소화합니다.

- **Technical Details**: HydraCIL은 새로운 작업에 대해 경량의 특정 작업 분류기 헤드를 생성하는 세 단계의 훈련 및 세 단계의 추론 모델을 제안합니다. 이 모델은 피처 추출기가 고정 상태로 유지되면서 모든 입력 샘플을 한 번만 처리하여 피처 표현을 생성합니다. 훈련 중 각 작업에 대해 새로운 클래스의 프로토타입을 생성하고, 이는 나중에 추론 단계에서 사용할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, HydraCIL은 CIFAR-100, ImageNet-100, CoRe50 및 Flowers102 데이터셋에서 최첨단 CIL 방법들과 비슷하거나 더 나은 성능을 보였습니다. 특히 학습 시간과 탄소 발자국을 현저히 줄이면서도 안정적인 예측을 달성하여, 에너지 효율성과 빠른 적응이 중요한 실제 환경에서의 지속적 학습에 실용적인 솔루션으로 자리매김했습니다.



### Temporal Context Conditioning for Seasonality-Aware Precipitation Nowcasting of High-Intensity Rainfa (https://arxiv.org/abs/2606.09959)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 논문에서는 경량(이러한) 시간적 맥락을 활용한 레이더 기반 강수 예측 모델을 제안합니다. 이름하여 Time-Aware Small-Attention U-Net (TA-SmaAt-UNet)으로, 이 모델은 시간의 주기성을 인코딩하여 강수 패턴의 내부 표현을 조정할 수 있도록 하였습니다. 이를 통해 고강도 강수의 예측을 더욱 향상시키고, 계절적 변동성과 강도 분포를 더 잘 반영할 수 있다는 점에 주목하고 있습니다.

- **Technical Details**: TA-SmaAt-UNet은 기존 SmaAt-UNet 모델을 기반으로 하여, 사이클적 시간 인코딩을 통해 강수 패턴의 중간 표현을 조정하는 시간 조건부 레이어를 도입합니다. 이렇게 함으로써 모델은 시간대(day of the day) 및 연간 반복(cycle of the year) 정보를 명시적으로 활용할 수 있어, 더 실질적이고 신뢰할 수 있는 강수 예측이 가능해집니다. 모든 레이어에서 이 조건부 기능이 통합되어 있어, 원래 구조의 효율성이 유지됩니다.

- **Performance Highlights**: 실험 결과, 시간적 맥락을 추가함으로써 드문 고강도 강수 사건에 대한 예측 정확도가 유의미하게 향상되었습니다. 계절적 변화와 예측된 강도 분포의 표현도 개선되었으며, 작은 파라미터 비용에도 불구하고 모델은 이러한 추가 레이어를 적극적으로 활용하고 있음을 보여줍니다. 이러한 결과들은 심층 학습 기반의 강수 예측에서 단순한 물리적 시간 맥락의 활용이 중요하다는 것을 시사합니다.



### Uncertainty-Aware Motion Planning for Autonomous Driving in Mixed Traffic Environmen (https://arxiv.org/abs/2606.09958)
- **What's New**: 본 논문에서는 자율주행 차량이 혼합교통 환경에서 인간 운전자의 미래 행동을 예측하는 데 있어 불확실성을 고려하는 'Uncertainty-Aware Motion Planning (UAMP)' 프레임워크를 제안하였습니다. 기존의 강화 학습 기반 방법들은 예측된 인간 의도를 관측치에 직접 통합하는 방식으로, 예측의 불확실성이 증가하여 위험한 결정으로 이어질 수 있는 문제를 지니고 있었습니다. UAMP는 인간 운전자의 의도를 정량화하고 이를 바탕으로 의사 결정을 조정하는 새로운 방법을 소개합니다.

- **Technical Details**: UAMP는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 'Proximity-aware uncertainty estimator'는 주변 차량의 상호작용을 기반으로 한 불확실성을 정량화하며, 둘째, 'Uncertainty-guided joint intent distribution'을 통해 다양한 차량의 행동 불확실성을 집계합니다. 셋째, 'Uncertainty-Calibrated Value Learning (UCVL)'을 도입하여 불확실성이 존재하는 상황에서도 더 안정적인 가치 추정을 가능하게 합니다.

- **Performance Highlights**: 다양한 혼합교통 시나리오에서 수행된 실험 결과, UAMP는 기존의 방법들과 비교하여 안전성과 주행 편안함을 크게 향상 시켰으며, 교통 효율성을 유지하는 데 성공하였습니다. 이러한 성과는 자율주행 시스템의 안전한 운영을 위해 필요한 의사 결정 과정에서 불확실성을 적절히 통합하는 것이 중요하다는 것을 보여줍니다.



### Does Normalization Choice Matter for Causal Large Time-Series Models? (https://arxiv.org/abs/2606.09954)
- **What's New**: 최근 대형 모델들이 시계열 예측(time-series forecasting)에서 주목받고 있습니다. 이 연구는 과거 데이터를 기반으로 현재 데이터를 예측하는 인과적 자회귀 아키텍처(causal autoregressive architectures)를 활용합니다. 그러나 실제 시계열 데이터는 비정상성(non-stationarities)을 보이기 때문에 예측 성능에 영향을 미칩니다. 이를 해결하기 위해 보통 정규화(normalization)가 사용되지만, 효율적인 설정에서는 미래 관측값으로부터의 정보 누수가 발생할 수 있습니다.

- **Technical Details**: 본 연구에서는 패칭(patching) 및 효율적인 인과적 전략(efficient causal strategy)으로 훈련된 변환기 기반(transformer-based) 대형 시계열 모델의 정규화 전략을 평가합니다. 최근 제안된 인과적 정규화(causal normalization)와 초기 관측값에서 계산된 통계량(statistics)을 활용하여 정보 누수 문제를 해결하려는 시도가 있습니다. 그러나 이러한 전략의 실제적인 결과는 아직 충분히 이해되지 않았습니다.

- **Performance Highlights**: 실험 결과, 정규화 방법의 선택이 훈련 수렴(training convergence)과 예측 성능(forecasting performance)에 상당한 영향을 미친다는 것을 보여줍니다. 이는 대형 모델을 활용한 시계열 예측에서 정규화 전략의 중요성을 강조하며, 보다 나은 성능을 위한 정교한 접근 방식을 제안합니다.



### Deep Slice Interpolation for Reducing Through-Plane Anisotropy and Noise in Head C (https://arxiv.org/abs/2606.09953)
- **What's New**: 이번 연구는 두 개의 인접한 축 방향 단면에서 중간 CT 슬라이스를 합성하여, 효과적인 슬라이스 간격을 절반으로 줄이는 딥러닝 시스템을 제안합니다. 이 시스템은 세 번째 차원 시각화를 개선하며, 동시에 노이즈 제거된 출력을 생성하여 하나의 추론(pass) 과정에서 상호 보완적인 두 가지 이점을 제공합니다. 기존의 보간 방법과 비교하여, 이 시스템은 모든 구조적 측정에서 우수한 성능을 보여줍니다.

- **Technical Details**: 제안된 시스템은 U-Net 아키텍처를 기반으로 하며, EfficientNetV2-S 인코더를 활용하여 RSNA 2019 Intracranial Hemorrhage Detection 데이터셋으로 훈련되었습니다. 다양한 픽셀 단위의 손실 함수(mean squared error, structural similarity index 등)를 체계적으로 평가하고, 훈련 중 나타나는 불안정성을 해결할 수 있는 방법들을 제시합니다. 또한, 회귀 기반 합성이 덴오이징(denoising) 특성을 상속하는 것을 발견했습니다.

- **Performance Highlights**: 모든 실험에서 제안된 모델이 기존의 고전적인 보간 방법 및 사전 훈련된 비디오 프레임 보간 모델(RIFE, FILM)을 초월하는 성능을 보여줍니다. 특히 MS-SSIM과 L1의 결합 손실이 가장 균형 잡힌 프로필을 제공함을 입증했습니다. 연구 결과는 환자 수준의 부트스트랩 신뢰 구간 및 짝 지어진 통계 테스트로 보고되며, 슬라이스 합성의 질이 다양한 병리적 조짐이 있는 경우에도 유지됨을 보여줍니다.



### Learning Where to Simulate: Generative Active Sampling for Online PDE Surrogate Training (https://arxiv.org/abs/2606.09949)
- **What's New**: 본 연구에서는 데이터 기반 PDE (Partial Differential Equations) 서브게이트를 훈련하기 위해 Online Generative Active Sampling (OGAS)이라는 액티브 러닝 방법을 제안합니다. OGAS는 구성 매개변수와 서브게이트 성능 간의 관계를 반응적으로 학습하여 샘플링 분포를 제어합니다. 이는 도전적인 다이나믹스를 갖는 일련의 PDE 구성 상황에서도 서브게이트의 일반화 능력을 향상시키는 데 도움을 줍니다.

- **Technical Details**: OGAS는 신속한 확산 모델을 서브게이트와 병렬로 훈련시켜 조건부 샘플러 역할을 하며, 서브게이트에서 파생된 난이도 신호(예: 손실 또는 불확실성)를 구성 매개변수에 매핑합니다. 이는 온라인 훈련 방식으로, 데이터 생성과 서브게이트 훈련이 동시에 수행될 수 있도록 합니다. OGAS는 2D PDEs의 다양한 도전적인 다이나믹스를 대상으로 하여 최대 308개의 매개변수를 사용하는 여러 서브게이트 아키텍처와 함께 평가되었습니다.

- **Performance Highlights**: OGAS는 모든 설정에서 꼬리 통계(tail statistics)를 일관되게 개선하여 99번째 백분위수 이상의 오류가 크게 감소하고 전체 오류 분산이 줄어드는 결과를 보여주었습니다. 도전적인 궤적을 우선시하는 과정에서는 평균 오류와의 트레이드오프가 발생하지만, OGAS는 학습된 서브게이트의 최악의 경우 신뢰도를 효과적으로 보장합니다. 이 과정에서 지연 시간이 거의 없는 이점을 가지며, 훈련 작업 흐름에 큰 영향을 미치지 않습니다.



### GAGI: A Gini-Adjusted GDP-per-Capita Index for Distribution-Aware Macroeconomic Welfare Monitoring (https://arxiv.org/abs/2606.09944)
- **What's New**: 이번 논문에서는 Gini-조정 GDP per Capita Index (GAGI)라는 새로운 지표를 제안합니다. 이는 GDP per capita를 소득불평등 조정 요인과 물가 수준으로 재조정하여 각 국가의 복지를 보다 정확하게 모니터링 할 수 있도록 설계되었습니다. GAGI는 연간 발표되는 공공 데이터를 기반으로 쉽게 계산되고 검토 가능하여 규제 기관이 사용할 수 있는 경제적 신호로 기능할 수 있습니다.

- **Technical Details**: GAGI의 계산 방식은 각 국가의 Gini 계수와 2015년 상수 기준 GDP per capita, 소비자 물가 지수를 사용하여 간단하게 이루어집니다. 2010년을 기준으로 정상화하며, GAGI 값이 1.0을 초과하면 복지가 개선된 것으로, 1.0 이하이면 악화된 것으로 판단할 수 있습니다. 이 지표는 각국의 경제적 상황을 교차적으로 비교하는 데 의미를 제공합니다.

- **Performance Highlights**: G7 국가에 GAGI를 적용한 결과, 복지 조정된 번영이 기존 GDP 성장과 일관되게 발산하고 있다는 것을 입증하였습니다. 특히 2022년 이후 이 발산 폭이 급격히 넓어졌으며, 이는 COVID-19와 생성 AI의 가속화와 관련이 있을 수 있습니다. GAGI는 GDP 기반 모니터링의 필요성을 보완하며, 자동화로 인해 발생할 수 있는 불균형을 면밀히 추적하여 보고된 성장률을 넘어서는 것입니다.



### Anomaly Detection and Root Cause Analysis for Microservice Systems (https://arxiv.org/abs/2606.09942)
Comments:
          This is the pre-print of my PhD thesis, submitted to RMIT University

- **What's New**: 이 논문은 클라우드 애플리케이션 구축에 널리 사용되는 마이크로서비스 시스템에서 발생하는 복잡성으로 인한 결함을 해결하기 위한 새로운 방법들을 제시합니다. 기존의 이상 탐지와 원인 분석(RCA) 방법들은 서로 따로 다루어지며, 이런 한계를 극복하기 위한 새로운 접근법을 소개합니다. 특히, 관찰 가능성 데이터(observability data)를 독립적 및 집합적으로 활용할 수 있는 방법을 제안합니다.

- **Technical Details**: 첫 번째 주요 기여는 BARO, EventADL, TORAI라는 세 가지 새로운 방법론으로, 각각 메트릭 데이터(metric data)에 대한 이상 탐지 및 RCA, 이벤트 데이터(event data)에 대한 종합적인 프레임워크, 서비스 호출 그래프(service call graph) 없이도 작동하는 다중 모드(multi-modal) RCA 프레임워크를 포함합니다. 이러한 방법들은 실제 마이크로 서비스 시스템에서의 광범위한 실험을 통해 그 효과성과 견고함을 입증하고 있습니다.

- **Performance Highlights**: 두 번째 주요 기여는 표준화된 데이터셋과 평가 프레임워크를 제공하는 RCAEval로, 이는 미래 연구를 위한 즉시 사용 가능한 데이터셋과 재현 가능한 기준선을 제공합니다. 또한, 기존의 RCA 방법, 특히 인과 추론(causal inference) 기반 접근 방식에 대한 체계적인 평가를 통해 통찰력을 제공하고 향후 연구 방향을 안내합니다. 이러한 기여는 마이크로서비스의 자동화된 이상 탐지 및 RCA를 혁신적으로 발전시키고, 향후 사건 완화 및 복구를 위한 연구를 가능하게 합니다.



### Interactions Between Crosscoder Features: A Compact Proofs Perspectiv (https://arxiv.org/abs/2606.09940)
Comments:
          Accepted at the NeurIPS 2025 Workshop on Mechanistic Interpretability

- **What's New**: 이번 연구에서는 Sparse Autoencoders (SAEs)와 crosscoders를 활용하여 딥 러닝 모델의 성능을 어떻게 이해하고 설명할 수 있는지에 대한 새로운 접근을 제안하고 있습니다. 저자들은 crosscoder의 상호작용 메트릭을 정의하고 이를 통해 computationally sparse crosscoders를 훈련시킬 수 있는 수단을 제공합니다. 이는 MLP 성능을 상당 부분 유지하면서도 더 적은 피처를 사용하는 효율적인 방법으로 발전할 수 있습니다.

- **Technical Details**: 모델의 내부 메커니즘을 이해하고 이를 통해 activations와 weight matrices를 해석 가능한 피처로 분해하는 것을 목표로 합니다. 특히, crosscoders는 다수의 레이어에서 동시에 activations를 분해하여, 서로 다른 레이어에서 분산된 방식으로 표현된 피처를 추출할 수 있는 장점을 가집니다. 연구에서는 이 crosscoder를 통해 compact proofs를 구성하는 방법론을 소개하며, 특히 interaction metric을 활용하여 MLP의 레이어에서의 상호작용의 정확한 표현을 제공합니다.

- **Performance Highlights**: 새로운 interaction metric을 활용하여, 해당 메트릭이 semantic feature clustering을 가능하게 한다는 점을 강조합니다. 계산적으로 효율적인 crosscoders를 훈련하면서 MLP 성능의 60%를 유지하는 데 성공했으며, 이는 기존의 crosscoders가 10%에 불과한 것과 큰 대조를 이룹니다. 마지막으로, sleeper agents의 상호작용을 통해 이상 탐지에 활용할 수 있는 초기 결과도 제시하고 있습니다.



### RKSC: Reasoning-Aware KV Cache Sharing and Confident Early Exit for Multi-Step LLM Inferenc (https://arxiv.org/abs/2606.09937)
Comments:
          Accepted to the ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic Systems

- **What's New**: RKSC(Reasoning-Aware KV Cache Sharing)는 멀티-브랜치 LLM(대형 언어 모델) 추론 파이프라인에서 두 가지 구조적 중복을 제거하는 훈련이 필요 없는 추론 프레임워크입니다. 이 프레임워크는 ASKS(Attention-Similarity KV Sharing)와 CGEE(Confidence-Gated Early Exit)와 같은 혁신적인 메커니즘을 통해 KV 캐시의 재사용을 최적화합니다.

- **Technical Details**: RKSC는 세 가지 보완 기전을 통해 멀티-브랜치 추론을 가속화합니다: 첫째, KV prefix sharing은 브랜치 간의 중복 계산을 제거합니다. 둘째, CGEE는 검증 전방 패스를 줄이거나 완전히 생략하고, 셋째, RSBCM(Reasoning-Selective Block Cache Manager)은 깊은 트리 검색 하에서 캐시 용량을 관리합니다.

- **Performance Highlights**: RKSC는 5개의 모델 패밀리(7B-10B) 및 1000개의 평가 문제에서 평균 3.008배의 속도 향상을 달성하였으며, vLLM과 동일한 prefix caching에 비해 1.66배 향상된 성능을 나타냈습니다. CGEE에 의해 유도된 오류율은 0.37%로, 검증 호출 1,616건 중 6건에 해당합니다.



### One Lens, Many Worlds : A Capability-Typed Interface for World-Model Interpretability (https://arxiv.org/abs/2606.09936)
- **What's New**: 세계 모델(세계 모델)들은 현재 실질적으로 서로 다른 계산 기초 위에서 구축되고 있습니다. 플래넷(PlaNet) 및 드리머(Dreamer) 계열처럼 잠재적 재귀 상태 공간 모델은 관측치를 재귀 상태로 압축하며, IRIS와 같은 토큰 기반 모델은 관측치를 학습된 코드북으로 양자화하고 트랜스포머(transformer)로 자기 회귀적으로 예측합니다. 주목할 점은 이러한 분절화가 도구에 의한 것이라는 주장으로, '월드모델렌즈(WorldModelLens)'라는 오픈소스 해석 가능성 기반을 통해 이러한 문제를 해결하려고 한다는 것입니다.

- **Technical Details**: 우리는 세계 모델을 관측 공간(𝒪)과 결정론적(latent state space 𝒮), 확률적(latent space 𝒵) 잠재 공간 및 선택적(action space 𝒜) 액션 공간을 포함한 튜플로 형식화합니다. 필수 요소로는 초기 상태 맵, 확률적 인코더, 전이 함수 및 샘플러가 있습니다. 각 모델은 네 가지 필수 메서드(encode, transition, initial state, sample)를 구현해야 하며, 다섯 가지 선택적 헤드를 선언할 수 있습니다.

- **Performance Highlights**: 월드모델렌즈는 각 모델의 고유한 내부 구조를 공통 인터페이스로 변환하도록 설계된 백엔드 적응기를 통해 세 가지 층으로 구성됩니다. 이를 통해 가능한 해석 가능성 분석 방법(Patching, Probing, Sparse Autoencoders 등)을 단 한번 구현할 수 있습니다. 참고로, 기존의 다양한 모델들에 대해 통일된 메서드로 분석을 수행할 수 있어, 해석 가능성을 더욱 직관적으로 접근할 수 있게 됩니다.



### GitInject: Real-World Prompt Injection Attacks in AI-Powered CI/CD Pipelines (https://arxiv.org/abs/2606.09935)
- **What's New**: 최근 AI 기반 에이전트들이 CI/CD(지속적인 통합 및 지속적인 배포) 파이프라인에 통합되어, 자동적으로 풀 리퀘스트를 검토하고, 문제를 분류하며, 코드베이스를 관리하는 역할을 수행하고 있습니다. 이러한 에이전트들은 untrusted content를 수용하면서도, 높은 권한을 가진 상징적인 GITHUB_TOKEN을 보유하고 있어 공격의 표적이 되기 쉽습니다. GitInject라는 새로운 오픈소스 프레임워크를 소개하며, 이는 실제 GitHub 워크플로우에서 프롬프트 주입 취약점을 평가합니다.

- **Technical Details**: GitInject는 ephemeral repositories를 제공하고, 실제 워크플로우를 실행하여 sandbox 제약 조건, 자격 증명 처리 방법, 권한 경계를 검토하는 데 중점을 둡니다. 기존의 에이전트 보안 기준은 도구 호출을 시뮬레이션하는 데 그쳤지만, GitInject는 이러한 한계를 극복하고 실제 실행 환경을 재현하여 CI/CD 보안 평가의 유효성을 증대시킵니다. 이 프레임워크를 통해 네 가지 AI 제공업체의 워크플로우 구성을 분석하고 여러 차원의 공격을 문서화했습니다.

- **Performance Highlights**: 모든 테스트된 AI 제공업체들은 기본 설정 상태에서 적어도 하나의 공격 클래스에 취약한 것으로 나타났습니다. 가장 심각한 취약점은 CI/CD 인프라에서 자격 증명 및 구성 파일 처리 방식에서 비롯되며, 특정 모델의 행동에서 생기는 것이 아닙니다. 각 확인된 공격 클래스에 대해 최소 비용의 워크플로우 수준 방어책을 제시하고 있으며, 이는 실무자들에게 유용한 가이드를 제공합니다.



### When RL Fails after SFT: Rejuvenating Model Plasticity for Robust SFT-to-RL Handoff (https://arxiv.org/abs/2606.09932)
- **What's New**: 이번 연구에서는 Supervised Fine-Tuning (SFT)와 Reinforcement Learning (RL) 간의 연결 문제를 분석하고, 이에 대한 해결책으로 효과적인 방법인 'Rejuvenation'을 제안합니다. SFT가 과도하게 이루어지면 모델의 적응성이 떨어져 RL에서의 성능 향상이 제한되는 현상을 관찰했습니다. 이는 SFT가 적절한 행동 지침을 제공하면서도 모델의 유연성을 잃는 문제에 기인합니다.

- **Technical Details**: SFT-then-RL 파이프라인에서는 SFT가 모델을 초기화하는 단계로 작용하며, RL이 이 초기화된 정책을 이어받아 보상을 기반으로 최적화합니다. 그러나 지나친 SFT는 모델의 파라메터 조정 및 출력 분포에 부정적인 영향을 미쳐 RL 단계에서 최적화를 어렵게 만듭니다. 이 논문에서는 SFT 모델의 유연성을 회복하기 위해 'Rejuvenation'을 제안하며, 기저 모델 융합(base-anchored model fusion) 및 선택적 뉴런 리셋(targeted neuron reset) 기법을 사용합니다.

- **Performance Highlights**: 제안한 Rejuvenation 방법은 수학적 추론 및 에이전트 태스크에서 실험을 통해 효과성이 입증되었습니다. 과도한 SFT로 인해 저하된 RL 성능을 회복할 수 있으며, OOD(out-of-distribution) 태스크에서도 ModSFT 모델들보다 우수한 일반화 성능을 보여줍니다. 이로써 SFT에서 RL로의 전환 시 발생하는 문제를 해결하는 중요한 기여를 합니다.



### A Note on the Strategic Confinement Problem (https://arxiv.org/abs/2606.09931)
- **What's New**: 이 논문에서는 Lampson의 은폐 문제(Confinement Problem)를 전략적 대리자(Strategic Agents)와 관련하여 새로운 관점에서 재조명합니다. 특히, 정보를 처리하는 프로그램이 특정한 동기와 지식을 공유하는 상황에서 정보가 누출되는 경로를 분석하며, 이 과정에서 보안 시스템 디자인에 대한 새로운 통찰을 제공합니다. 저자들은 전통적인 은폐 이론이 전략적 대리자가 있는 경우에 한계를 가지고 있음을 지적합니다.

- **Technical Details**: 전통적인 은폐 문제는 여러 프로그램이 데이터 공유 환경에서 어떻게 하면 기밀 데이터가 누출되지 않도록 할 수 있는지를 다룹니다. 이 논문에서는 고객, 서비스 프로그램, 그리고 서비스 소유자의 관계를 통해 기밀 데이터의 은폐를 위한 다양한 사상과 보안체계의 설계를 다시 고려합니다. 저자들은 전략적 대리자가 정보를 처리할 때 정보 이론적 은폐 보장이 해악 이론적 보장으로 이어지지 않는다고 주장합니다.

- **Performance Highlights**: 저자들은 오늘날 언어 모델 에이전트와 같이 학습된 전략적 대리자들이 이 문제 내에서 주변 환경 및 기존 지식에 대한 비대칭성을 바탕으로 불리한 행동 경로를 만들어낼 수 있음을 강조합니다. 이러한 대리자들은 오류 또는 고의적인 변화를 통한 기밀 데이터의 누출 위험을 높일 수 있으며, 안전한 시스템 설계를 위한 도전 과제를 제기합니다. 결과적으로 학습된 전략적 대리자는 정보와 해악의 관계를 복잡하게 하고, 이를 해결하기 위한 새로운 접근 방식이 필요함을 시사합니다.



### Between Amnesia and Chaos: A Memory Stability Expressivity Trilemma for Trainable Dissipative Oscillator Networks (https://arxiv.org/abs/2606.09929)
- **What's New**: 본 논문은 물리적 레저보어 컴퓨팅이 비선형 진동자를 이용해 매개변수를 학습하는 새로운 접근 방식을 제안합니다. 기존의 방법에서는 물리적 기판을 동결하고 단순한 선형 읽기를 훈련시켰으나, 이 연구에서는 물리적 매체의 동적 특성을 직접 학습하여 메모리 지평선(memory horizon), 그래디언트 안정성(gradient stability), 및 동적 표현력(dynamical expressivity) 사이의 상충 관계를 제시합니다. 연구팀은 매개변수의 damping이 이러한 세 가지 요소를 동시에 최대화할 수 없는 트릴레마(trilemma)를 발견했습니다.

- **Technical Details**: 연구에서는 비선형 스프링의 네트워크를 구성하고, 그 운동 방정식을 통합하여 스프링 및 감쇠(damping) 매개변수를 학습하는 방법을 제시합니다. 특히, 포아송 변환을 이용하는 구조 보존 적분기(symplectic integrator)를 사용하여 굉장히 긴 지평선에서도 신뢰할 수 있는 그래디언트를 유지합니다. 네트워크의 각종 매개변수를 연쇄적으로 학습한 결과, 감쇠 계수에 따라 안정적인 그래디언트와 메모리 간의 관계가 정리되었음을 확인했습니다.

- **Performance Highlights**: 실험에서는 20개의 진동기로 구성된 네트워크에서 매개변수를 학습한 경우와 동결된 경우의 성능을 비교하였습니다. 그 결과, 짧은 지평선에서는 학습된 기판이 우수한 성능을 발휘했으나, 약 11단계의 지평선에서는 성능이 정체 또는 역전되는 경향을 보였습니다. 이러한 결과는 실험적으로 증명된 이론과 일치하며, 동결된 기판은 안정적인 옵션에 불과하다는 각각의 학습 가능성을 검증했습니다.



### Forward-Only Convolutional Neural Networks with Learnable Channel-Class Assignmen (https://arxiv.org/abs/2606.09928)
- **What's New**: 이 논문에서 소개하는 Forward-Forward(FF) 알고리즘은 생물학적으로 영감을 받은 대안으로, gradient 기반의 credit assignment 방식 대신에 지역적이고 전방향으로만 목표를 설정하는 방법을 제시합니다. 새로운 learnable channel-class assignment 메커니즘을 도입하여 convolutional channels의 적응형 데이터 기반 전문화를 지원하며, 엔트로피 및 직교 정규화를 통해 학습 성능을 향상시킵니다. 또한 layer contribution 전략을 통해 각 중간-layer의 예측을 검증 성능에 기반하여 적응적으로 가중치를 조정하여 forward-only 추론을 더욱 효과적으로 만듭니다.

- **Technical Details**: 이 연구에서는 learnable channel-class assignment 메커니즘을 통해 convolutional 계층의 학습 능력을 강화하고 각 채널의 기여도를 정적으로 할당하는 것이 아니라 적응적으로 조정할 수 있도록 합니다. 또한, 개별 layer의 기여도를 평가하기 위해 성능 기반의 새로운 접근법을 제안하며, 이는 정적인 채널 그룹핑보다 효율적인 채널 활용을 가능하게 합니다. 연구 결과는 residual CNNs에 통합하여 기존의 forward-only 방식들에 비해 CIFAR-10, CIFAR-100, Tiny-ImageNet에서 일관되게 우수한 성능을 달성하는 것으로 나타났습니다.

- **Performance Highlights**: 제안한 메서드는 FF 기반의 모델 중에서 새로운 최첨단 성능을 설정하며, backpropagation 방식과의 격차를 상당히 좁혔습니다. 제안 방법은 학습 가능한 채널 전문화 및 layer 가중치 조정이 깊은 CNN에서 forward-only 학습의 표현 능력을 크게 향상시킨다는 것을 보여줍니다. 이러한 발견은 다양한 컴퓨터 비전 작업에 있어 FF 알고리즘의 실제 적용 가능성을 높이는 중요한 이정표가 됩니다.



### Trainable Smooth-Rotation Transforms with Learned Channel Scales for LLM Quantization (https://arxiv.org/abs/2606.09927)
Comments:
          6 pages, 8 figures, 3 tables. Accepted to IEEE INES 2026 conference proceedings

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 후처리 양자화(Post-training quantization, PTQ)에서 발생하는 활성화 양자화(activation quantization)의 문제를 다룹니다. 새로운 연구 결과로는 quantile-robust scaling policy를 소개하며, 이는 활성화 통계량을 최대값 기반(max-based)에서 높은 분위수(high quantile)로 교체하여 의도되지 않은 오류를 줄이는 방법입니다. 이를 통해 기존 방식에서 발생하던 양자화 오류를 개선할 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: 저자들은 SmoothRot 아키텍처를 기반으로 하여, 활성화 아웃라이어(outlier) 문제를 해결하기 위해 채널 스케일(scale) 학습과 정책 구조화 최적화(policy-structured optimization)를 제안합니다. 이 방법은 낮은 비트 양자화에서 활성화 범위를 감소시키고, 가중치에 대한 양자화 어려움을 전이하는 것을 줄이는 데 초점을 맞추고 있습니다. 또한, Straight-Through Estimator (STE)를 사용하여 최적화 과정에서 비선형성을 보완합니다.

- **Performance Highlights**: LLaMA-3.2-1B 모델에서 W4A4 양자화 설정 하에 수행된 테스트 결과, 제안된 정책을 통해 SmoothRot 기법보다 11.1%의 오차 개선을 확인했습니다. 또한 조합된 튜닝 방식은 12% 개선, 훈련 후에는 최대 18.5%의 향상이 있음을 보여주었습니다. 최종적으로 모든 디코더 블록의 다운 프로젝션 레이어에 최적의 정책을 재실행함으로써 전체 레이어 평균 오류를 19.9% 감소시켰습니다.



### Sample Where You Struggle: Sharpening Base Model Reasoning via Entropy-Guided Power Sampling (https://arxiv.org/abs/2606.09926)
- **What's New**: 이번 연구에서는 Entropy-Guided Power Sampling (EGPS)라는 새로운 샘플러를 제안합니다. EGPS는 기존의 Metropolis-Hastings (MH) 샘플러의 비효율성을 극복하여, 파라미터 업데이트 없이 base 모델의 추론 능력을 향상시키는 혁신적인 방법입니다. EGPS는 샘플링 과정을 재구성하여, 단순히 각 토큰을 동일하게 샘플링하는 대신, 중요한 결정을 내리는 고엔트로피 토큰의 주변에서 샘플링할 수 있도록 최적화됩니다.

- **Technical Details**: EGPS는 두 가지 수준에서 작동하는데, 블록 수준에서는 저엔트로피 문을 활용하여, 최대 토큰 엔트로피가 특정 임계값 이하인 블록은 생략합니다. 토큰 수준에서는, 재샘플링 포인트를 토큰 엔트로피 비례로 선택하여, 품질을 높이는 저희 고엔트로피 클러스터에 집중하도록 합니다. 또한, 단일 제안 대신 여러 후보를 다루는 Multiple-Try Metropolis (MTM)를 통해 탐색을 확대합니다.

- **Performance Highlights**: EGPS는 Qwen2.5-Math-7B 모델에서 MATH500, HumanEval, GPQA 세 가지 벤치마크에서 모두 최고 또는 공동 최고 정확도를 기록했습니다. 특히, MATH500에서 75.8%, HumanEval에서 62.2%, GPQA에서 42.4%의 성과를 달성하며, 같은 vLLM 프레임워크 내에서 MH Power Sampling 대비 최대 12.6배의 속도 향상을 실현했습니다. 이러한 성과는 EGPS의 고급 샘플링 전략이 기존 접근방식보다 효과적임을 보여줍니다.



### Sigma-Branch: Hierarchical Single-Path Network Reconstruction for Dynamic Inference with Reduced Active Parameters (https://arxiv.org/abs/2606.09924)
- **What's New**: 이 논문은 제한된 메모리에서 사용할 수 있는 엣지(Edge) 가속기에 딥 뉴럴 네트워크(DNN)를 효과적으로 배포하기 위한 새로운 방법인 시그마-브랜치(Sigma-Branch, SigmaB) 프레임워크를 제안합니다. 이 방식은 프리트레인(pretrained)된 밀집 네트워크를 계층적 바이너리 트리 구조로 재구성하여 메모리 사용을 최적화합니다. 기존의 모델 압축 기술은 영구적인 용량 손실을 초래하는 반면, SigmaB는 전체 파라미터 집합을 유지한 채로 추론 시 단일 경로만을 실행하여 활성 파라미터 메모리를 크게 줄일 수 있습니다.

- **Technical Details**: SigmaB는 활성화 기반의 구형 k-평균 클러스터링을 통해 프리트레인된 가중치를 재구성하고, 이를 통해 라우터 가중치와 브랜치 채널 할당을 공동 초기화합니다. 이후 소프트 라우팅 파인튜닝을 통해 각 브랜치가 라우팅된 입력 하위 집합에 맞춰 조정됩니다. 최종적으로, 이 네트워크는 단일 루트-투-리프(root-to-leaf) 경로만을 실행하여 메모리 효율성을 극대화합니다.

- **Performance Highlights**: SigmaB-Net은 CIFAR-100 / ResNet-50, ImageNet-1K / ResNet-50 및 ModelNet40 / PointNet++ 데이터셋에서 58-60%의 활성 파라미터 감소를 달성하면서도 밀집 네트워크의 Top-1 정확도에서 1.72% 포인트(pp) 이내를 유지합니다. 또한, ImageNet-1K Top-1 기준에서 SigmaB-Net의 활성 파라미터 감소는 기존의 정적 구조 프루닝(static structured pruning) 기법보다 14-23 pp 우수한 결과를 보였습니다. 이를 통해 SigmaB 프레임워크는 2D 비전 및 3D 포인트 클라우드 백본을 아우르는 다양한 적용 가능성을 입증하고 있습니다.



### Conformal Prediction for Neural Operators: Distribution-Free Uncertainty Quantification in Physics Simulation (https://arxiv.org/abs/2606.09923)
Comments:
          13 pages, 7 tables, 7 figures. Full-scale experiments on NVIDIA V100

- **What's New**: 이 연구는 Split Conformal Prediction을 신경 연산자 기반 물리 시뮬레이션에 처음으로 적용하여, 유한 표본 커버리지 보장이 포함된 분포 자유 예측 구간을 제공하는 방법을 제안합니다. 이는 특히 안전이 중요한 공학 응용 분야에서 예측의 정확성과 불확실성 보장을 동시에 충족합니다. 이 방법은 MC Dropout 불확실성을 활용하여 적응형 너비 예측 구간을 생성하고, 공간적으로 적응 가능한 예측을 구현하여 물리적 불확실성 구조를 반영합니다.

- **Technical Details**: 이 연구에서는 포일러 신경 연산자(FNO)를 기반으로 하는 불확실성 정량화(UQ) 방법을 제안하며, 기존의 Monte Carlo Dropout 및 Deep Ensembles 방법의 한계를 극복하고 있습니다. 제안된 방법은 강력한 커버리지 보장과 함께, 유도된 예측 구간을 제공하며, 모델의 불확실성에 따라 구간의 너비를 조정합니다. 구체적으로, 불확실성 분해 프레임워크를 통해 에피스테믹 불확실성(모델 불확실성)과 알레아토릭 불확실성(데이터 노이즈)을 분리하는 방법도 소개합니다.

- **Performance Highlights**: 전제어 정밀 실험에서 본 방법은 33.7M 매개변수를 기반으로 하여, 800개의 훈련 샘플과 5개의 앙상블 멤버를 활용해 α=0.1 목표 수준에서 89.1%의 경험적 커버리지를 달성했습니다. 실험 결과는 FNO의 예측이 물리적 불확실성 구조를 정확하게 반영한 공간적으로 적응 가능한 예측 구간을 생성하는 데 성공했음을 보여 줍니다. 제안된 방법은 오픈 소스 플랫폼에서 REST API 엔드포인트와 인터랙티브 3D 시각화를 통해 구현되어 있습니다.



### The Bioelectrical Information Theory: Investigating the theoretical compression limit of bioelectrical signals under artificial intelligenc (https://arxiv.org/abs/2606.09922)
- **What's New**: 이번 논문에서는 생체 전기 신호(bioelectrical signals)의 압축(compression) 문제를 새로운 정보 이론적(information-theoretic) 프레임워크로 제안하고 있습니다. 이 프레임워크는 신호의 정확도(signal fidelity)뿐 아니라 생리학적 구조(physiological structure), 모델 용량(model capacity), 그리고 하위 작업 요구사항(downstream task requirements) 등 여러 요소에 의해 효과적 정보가 결정됨을 강조합니다. 또한, 생체 전기 신호 압축을 세 가지 수준의 계층(hierarchy)으로 구성하여 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 생체 전기 신호 압축은 신호 수준(signal level), 생리학적 수준(physiological level), 의미 수준(semantic level)의 세 가지 단계로 나뉘어 있습니다. 첫 번째 단계에서는 측정 잡음을 제거하고 숨겨진 생리학적 원천에 대한 정보만을 유지합니다. 두 번째 단계에서는 정제된 신호를 구조적이고 저차원(low-dimensional)인 표현으로 압축하며, 마지막 단계에서는 하위 작업에 관련된 정보만을 추출하여 비의미적 정보를 제거합니다. 이를 통해 기존의 신호 재구성과 모델 기반 표현 학습이 통합됩니다.

- **Performance Highlights**: 이 연구의 중요한 점은 생체 전기 신호의 압축 한계가 단순히 파형의 충실도에 의존하는 것이 아니라, 생리학적 구조와 모델 및 작업의 요구 사항 간의 상호작용에 의해 결정된다는 점입니다. 이러한 새로운 프레임워크는 데이터 수집 시 신호 전송의 효율성을 높이고, 필요한 정보만을 전달하는 방식으로 생체 전기 신호의 전송이 발전할 수 있도록 도와줍니다. 따라서 효과적인 압축을 통해 작업 수준 해석(task-level interpretation)에 필요한 정보만을 남길 수 있는 가능성을 보여줍니다.



### Co-GLANCE: Uncertainty-Aware Active Perception for Heterogeneous Robot Teaming (https://arxiv.org/abs/2606.09919)
Comments:
          Code, videos, and dataset available at this https URL

- **What's New**: Co-GLANCE는 이질적인 로봇 팀을 위한 실시간 인식 및 의사 결정 시스템입니다. 이는 비전-언어 모델(vision-language model)의 의미적 추론 능력을 경량화된 모델로 단순화하여 클라우드 기반 추론을 제거합니다. Co-GLANCE는 성과 기반의 불확실성 정량화를 위해 선택적 어떤 것을 결합하여 분할 및 로봇 할당을 위한 통계적으로 유효한 보장과 함께 이전에 제기된 문제를 해결합니다.

- **Technical Details**: Co-GLANCE는 두 가지 주요 기술을 통합하여 작동합니다. 첫째, 정보가 부족한 영역을 식별하기 위해 문맥 인식을 사용하는 폐쇄 세그먼트 모델을 사용합니다. 둘째, 선택적 어떤 것(selective abstention)과 준거적 예측(conformal prediction)을 결합하여 로봇 할당, 개체 탐지 및 세그멘테이션에 대한 보정된 불확실성 추정을 제공함으로써 적극적인 인식을 유도합니다.

- **Performance Highlights**: Co-GLANCE는 실제 상황에서 클라우드 기반 비전-언어 모델 기반 (baseline)보다 폐쇄 세그멘테이션과 로봇 할당 정확도를 각각 25%와 36% 높이며, 프레임당 추론 지연 시간을 350배 줄였습니다. 또한, 향후 연구를 위한 공중-지상(multimodal air-ground) 데이터셋을 발표합니다.



### IntentKV: Cross-Turn Intent-Aware KV Cache Pruning for Agent Inferenc (https://arxiv.org/abs/2606.09916)
- **What's New**: 이 논문에서는 Multi-turn LLM (large language model) 에이전트의 KV (key-value) 캐시의 효율성을 개선하기 위한 IntentKV라는 새로운 방법을 소개합니다. 기존의 KV 캐시 프루닝(pruning) 방법은 단일 프롬프트(prompt)에 최적화되어 있지만, IntentKV는 세션의 변화하는 쿼리 요구를 반영하여 다중 쿼리 유지 관리(multi-query retention)를 통해 개선된 성능을 보여줍니다.

- **Technical Details**: IntentKV는 세션 수준의 QueryMemory를 유지하여 각 세션의 쿼리 신호를 집계하고, 메모리-주의(rule) 규칙을 따라 현재 쿼리의 K-벡터에 대해 과거 이력을 평가합니다. 또한, 새로운 잔여 헤드(residual head)를 추가하여, 프룬 상태에서 중요한 정보를 잃지 않도록 하고, 배치 시에도 프리픽스 캐시(prefix cache)와 호환될 수 있도록 삭제(삭제) 정책을 구현합니다.

- **Performance Highlights**: 체계적으로, IntentKV는 Qwen3-8B 모델에서 평균 8k KV 예산 하에 요청하는 최대 토큰 수를 23.9% 줄이고, Qwen2.5-14B에서는 30.7% 감소를 이루었습니다. 특히, 100개의 가장 긴 BCP 쿼리에서 최악의 경우 요청 토큰 수를 92.3k에서 20.5k로 줄이며, 원시 KV 읽기 수는 411M에서 31M으로 감소하였습니다.



### Mix, Don't Pick: Why Synthetic Corpus Composition Matters for Time Series Foundation Model Pretraining (https://arxiv.org/abs/2606.09912)
Comments:
          Accepted at the ICML 2026 Workshop on Foundation Models for Structured Data (FMSD), Seoul, South Korea

- **What's New**: 이 논문은 시간 시계열 기반의 모델(TSFM) 사전 훈련을 위해 합성 데이터 선택에서 발생하는 오류를 분석하고 있다. 기존에 모델 아키텍처에 따라 유용한 생성기(generator)의 선택이 다르다는 점을 강조하며, 많은 생성기를 단순히 혼합(mixture)하는 접근 방식이 가장 효과적임을 보여준다. 저자들은 사전 훈련 데이터셋의 구성이 합성 선택 문제가 아닌 조합(composition) 문제로 보는 새로운 관점을 제시한다.

- **Technical Details**: 연구에서는 11개의 합성 생성기 가족을 통하여 각각의 생성기로 Chronos-T5-Mini와 Moirai-Small 모델을 훈련시키며, 이들의 결과를 제로샷(zero-shot) 평가로 비교했다. 생성기 선택이 동일한 훈련 예산 하에서도 상당한 영향을 미친다고 나타났다. 실험은 CRPS(Continuous Ranked Probability Score)와 MASE(Mean Absolute Scaled Error)를 통해 결과를 측정하고, 생성기 순위가 아키텍처에 따라 다름을 보여준다.

- **Performance Highlights**: Mixed11이라 불리는 모든 생성기의 동일 가중치 혼합을 통해, Moirai-Small 모델의 경우 KernelSynth에 필적하는 성능을 보여주었으며, Chronos-T5-Mini에서는 두 생성기보다도 성능이 향상되었다. 최적의 혼합 비율은 아키텍처에 따라 달라지므로, 각 모델에 대한 적합한 데이터 구성 검증이 필요하다. 이 연구는 합성 데이터가 실제 데이터와 혼합될 때 사전 훈련 성능을 향상시킬 수 있음을 입증하였다.



### Bypassing Copyright Protection in Diffusion-based Customization via Two-Stage Latent Feature Optimization (https://arxiv.org/abs/2606.09909)
Comments:
          accepted by KDD 2026

- **What's New**: 이번 연구에서는 Diffusion 기반의 커스터마이제이션에서 저작권 침해 문제를 해결하기 위해 새로운 공격 방법인 Two-Stage Latent Feature Optimization (TS-LFO)를 제안합니다. 기존의 방어법이 주입한 적대적 변동이 보호된 컨텐츠의 개인화 생성 능력을 저하시킨다는 점을 발견하였으며, 이를 활용하여 공격을 설계했습니다. TS-LFO는 고주파 적대적 변동을 제거하고 저주파 의미 정보를 회복하여, 모델이 보호된 시각적 컨텐츠를 정확하게 재구성할 수 있도록 합니다.

- **Technical Details**: TS-LFO의 첫 번째 단계인 Latent Denoising Stage에서는 Latent-Image Alignment Loss와 Latent Diffusion Loss를 결합하여 보호된 입력 이미지와의 의미적 일관성을 회복합니다. 이후 Latent Reconstruction Stage에서는 픽셀 수준 제약을 가하여 저주파 의미 충실도를 복원하고, 변동된 잠재 표현에서 원래 보호된 시각적 컨텐츠를 재구성합니다. 이를 통해 LDMs의 잠재 이미지 매핑을 효과적으로 이용한 저작권 공격을 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TS-LFO는 DiffPure, GrIDPure, IMPRESS와 같은 기존의 최첨단 저작권 방어를 지속적으로 우회하며, 다양한 설정에서 우수한 성능을 보였습니다. TS-LFO는 현재 상용화되어 있는 LDMs 보호 방법을 효과적으로 무력화하는 것을 증명했습니다. 이러한 결과는 Diffusion 기반의 개인화 생성의 저작권 보호 전략에 새로운 관점을 제시합니다.



### IDP-Bench: Benchmarking ability of LLMs to protect personal information in interdependent privacy contexts (https://arxiv.org/abs/2606.09908)
- **What's New**: 이번 연구에서는 개인 AI 어시스턴트로 배포되는 대형 언어 모델(LLMs)의 프라이버시 문제를 다루기 위해 인터디펜던트 프라이버시(interdependent privacy, IDP)를 고려한 첫 번째 벤치마크인 IDP-Bench를 도입했습니다. IDP-Bench는 맥락적 무결성(contextual integrity, CI) 프레임워크에 기반하여 설계되었으며, 여러 주체의 개인 데이터가 어떻게 상호작용하는지를 평가합니다. 이 벤치마크는 LLM들이 IDP 시나리오를 이해하는 능력을 검토하며, 기존의 프라이버시 벤치마크와는 차별화된 접근 방식을 채택하고 있습니다.

- **Technical Details**: IDP-Bench는 8개의 오픈 소스 LLM을 대상으로 하여 그들의 IDP 이해력을 평가했습니다. 여기에는 AI가 생성한 회의록, 그룹 사진, 공유 위치 데이터 등이 포함된 다양한 IDP 시나리오가 포함되어 있습니다. 평가 질문은 CI-Bench와 PrivacyLens를 바탕으로 만들어졌으며, 각 모델의 컨텍스트 이해도와 다수 주체의 동의 인식을 비교하는 방식으로 진행되었습니다.

- **Performance Highlights**: 결과에 따르면, 8개 모델 중 6개 모델이 90% 이상의 공동 소유 인식에서 우수한 성과를 보였으나, CI 파라미터나 IDP-specific 기준을 식별할 때는 여전히 약점을 드러냈습니다. 8개 모델 가운데 7개는 74% 이하의 점수를 기록했으며, 공유 적절성 평가에서도 5개 모델이 77% 이하의 점수를 기록했습니다. 또한, 작은 모델에서는 성능이 저하되는 경향이 있어, LLM의 프라이버시 연구에서 IDP를 보다 심층적으로 연구할 필요성이 강조되었습니다.



### LongMoE: Longitudinal Multimodal Learning via Trajectory-Aware Mixture-of-Experts (https://arxiv.org/abs/2606.09907)
- **What's New**: LongMoE는 임상 다중 모드 학습에서의 모드 누락(modality missingness)과 장기적 동역학(longitudinal dynamics) 문제를 동시에 해결하는 새로운 프레임워크로 제안되었습니다. 이 방법은 모드별 크로스 어텐션(cross-attention)과 연속 시간 다중 주파수 위치 인코딩(temporal representation)을 결합하여 비정규적인 방문 시퀀스에서의 시간적 패턴을 효과적으로 캡처합니다. 실험 결과, LongMoE는 기존의 모델들보다 뛰어난 성능을 보이며 강력한 새로운 기초를 제공합니다.

- **Technical Details**: LongMoE는 컨텍스트 인지(imputation module), 경과 추적 인코더(trajectory-aware encoder) 및 컨텍스트 조건 sparse MoE 라우팅(context-conditioned Sparse MoE routing)과 같은 여러 구성 요소로 이루어져 있습니다. 이 시스템은 모든 방문에서 임의의 모드 조합을 처리할 수 있으며, 환자 개인의 전문가 전문가 선택을 가능하게 합니다. 이를 통해 임상 기록을 세밀하게 분석하여 질병의 진행 상황을 정확히 모델링할 수 있습니다.

- **Performance Highlights**: ADNI, OASIS-3, MIMIC-IV 데이터셋에서 LongMoE는 누락된 모드나 약한 모드에서 높은 내구성을 보여주었으며, 전체 모드 상황에서도 경쟁력을 유지합니다. 이는 LongMoE가 비정상적으로 샘플링된 방문 기록에서의 장기적 의존성을 효과적으로 캡처하고 결합할 수 있음을 입증합니다. 이로써 LongMoE는 임상 다중 모드 학습의 새로운 기초를 확립하는 데 기여합니다.



### The Whale That Outswam Evolution: Swarm Intelligence Maximises Memory in Connectome Reservoirs (https://arxiv.org/abs/2606.09902)
- **What's New**: 이 논문은 생물학적 신경 연결체(Connectome)의 구조를 활용하여 고전적인 Reservoir Computing을 개선하는 새로운 접근 방식을 제안합니다. 저자들은 6종의 다양한 생물체에서 수집된 연결체를 기반으로 한 echo-state networks에 네 가지 생물학 영감을 받은 최적화 알고리즘을 적용했습니다. 이 연구는 최적화가 진화에 의해 부여된 구조를 뛰어넘을 수 있는지를 탐구하며, 연구된 다양한 최적화 기법이 여러 작업 및 생물체에서 어떻게 차별화되는지를 살펴봅니다.

- **Technical Details**: Reservoir Computing 개념을 바탕으로, echo-state network(ESN)는 고정된 순환 네트워크를 통해 입력를 장기기억할 수 있는 고차원 피쳐 공간으로 변환합니다. 저자들은 특히 Memory Capacity(MC), Lorenz attractor prediction, NARMA-10 시스템 식별, Mackey-Glass 시간 시퀀스 예측 작업을 사용하여 네트워크 성능을 평가합니다. 연구에 사용된 네 가지 최적화 알고리즘은 Particle Swarm Optimisation (PSO), Differential Evolution (DE), Grey Wolf Optimisation (GWO), Whale Optimisation Algorithm (WOA)이며, 각 알고리즘은 비슷한 방식으로 작동하지만 업데이트 규칙이 다릅니다.

- **Performance Highlights**: 전반적으로, 네 가지 최적화 알고리즘은 생물학적 기초와 관련된 초기화로부터 시작했을 때 모든 작업과 생물체에서 일관되게 향상된 성능을 보였습니다. 특히 WOA는 17배 이상의 Memory Capacity 개선과 함께, 89%의 NRMSE 감소를 달성하며 평균 214% 성능 개선을 보여줍니다. 연구 결과는 생물학적 초기화의 중요성을 강조하며, 단순한 구조만으로는 회복할 수 없는 생물학적 가중치 값들이 연결체 기반 Reservoir Computing에서 중요한 인덕티브 바이어스(inductive bias)로 작용함을 보여줍니다.



### Less Context, More Accuracy: A Bi-Temporal Memory Engine for LLM Agents Where a Lean Retrieved Context Beats the Full History (https://arxiv.org/abs/2606.09900)
Comments:
          14 pages, 4 figures, 3 tables. Code, reproducible harness, and raw per-question logs: this https URL

- **What's New**: Engram은 LLM(대형 언어 모델) 에이전트의 장기 기억을 구현하기 위한 새로운 오픈 소스 메모리 엔진입니다. 기존의 메모리 시스템들은 비용이나 지연(latency) 측면에선 효과적일지 모르나, 정확도에서의 단점을 극복하지 못했습니다. 본 시스템은 ‘bi-temporal(이중 시간)’ 데이터 모델을 활용하여 비파괴적(conflict resolution)인 갈등 해소 방법을 제공하며, 기존 이론과는 다른 접근 방식을 통해 기계 학습 모델의 정확도를 향상시킬 수 있습니다.

- **Technical Details**: Engram은 빠른 쓰기 경로와 느린 비동기적 통합 경로를 사용한 이중 프로세스 메모리 시스템입니다. 이 시스템은 ‘episode’와 ‘fact’라는 데이터 구조를 통해 정보의 수집 및 저장을 최적화합니다. 데이터는 통합 속성(provenance)을 유지하면서 변경 사항을 추적하고 관리하고, 명확한 시간축(valid/invalid) 정보를 제공합니다. 또한 하이브리드 읽기 경로가 고밀도 및 최근성 정보를 융합하여 더욱 정확한 데이터 검색을 제공합니다.

- **Performance Highlights**: Engram은 LongMemEval_S에서 500개의 질문에 대해 83.6%의 정확도를 기록하며, 전통적인 전체 텍스트 컨텍스트(full-context) 방식에서의 73.2%보다 +10.4 포인트 향상된 성과를 보였습니다. 이는 약 8배 적은 토큰 사용으로 달성된 결과입니다. 시스템의 설계 원칙 중 하나는, 재현할 수 없는 숫자는 존재하지 않는다는 것으로, 이를 통해 사용자들은 결과의 신뢰성을 확립할 수 있게 됩니다.



### When Attribution Patching Lies: Diagnosis and a Second-Order Correction (https://arxiv.org/abs/2606.09899)
Comments:
          30 pages, 12 figures

- **What's New**: 본 논문에서는 언어 모델의 내부 메커니즘을 이해하기 위한 메커니즘 해석 가능성의 목표를 다루고 있습니다. 특히, 메커니즘의 오류 소스를 분석하여, 주요 오류가 패치된 구성 요소의 국소 비선형성이 아니라 네트워크의 다운스트림 응답에서 발생한다는 사실을 밝혔습니다. 이를 통해 신뢰성 점수와 오차 경계, 그리고 Hessian-vector-product (HVP) 보정을 통해 오류를 단순 식별할 수 있는 세 가지 도구를 제안합니다.

- **Technical Details**: 오류 분석을 통해 Attribution patching의 신뢰성을 평가할 수 있는 방법을 제시했습니다. 이는 주로 국소 비선형성을 고려한 기존의 접근방식과 차별화되며, 대신에 다운스트림 네트워크의 반응에 초점을 맞추고 있습니다. 이를 기반으로 HVP를 통해 오류를 제거하며, 높은 성능을 유지하면서도 컴퓨팅 비용을 줄일 수 있는 다단계 HVP(MS-HVP) 변형도 제안합니다.

- **Performance Highlights**: 다양한 모델 군을 대상으로 한 평가에서, HVP는 기존의 Integrated Gradients와 비교하여 더 적은 계산 비용으로도 더 나은 정확도를 보였습니다. 특히, 8B 이상의 대규모 모델에서 HVP의 단일 단계 보정이 최대 82%의 오류 감소를 가져오는 것을 보여주었습니다. 이는 루비스트한 회로 회복을 가능하게 하고, 신뢰성이 낮은 구성 요소에만 컴퓨팅 자원을 집중할 수 있는 Screen-Flag-Fix 워크플로우를 지원합니다.



### HMAF: A Hierarchical Multi-Slot GD-RTB Allocation Framework (https://arxiv.org/abs/2606.09896)
Comments:
          Accepted by KDD 2026 Applied Data Science Track

- **What's New**: 본 논문에서는 현대 온라인 광고 플랫폼에서 Guaranteed Delivery (GD) 계약과 Real-Time Bidding (RTB) 경매가 공존하는 환경에서 발생하는 문제를 해결하고자 Hierarchical Multi-Slot Allocation Framework (HMAF)를 제안합니다. HMAF는 강력한 오프라인 제약 최적화와 온라인 의사결정을 통합하여, 광고 노출 할당을 최적화하는 데 중점을 두고 있습니다. 이 프레임워크는 복잡한 다중 슬롯 환경에서 GD와 RTB의 경쟁력을 동적으로 조정하여 실시간 의사결정을 지원합니다.

- **Technical Details**: HMAF는 Plan--Calibrate--Execute 패러다임을 기반으로 구조화되어 있으며, 오프라인에서 유도된 이중 변수를 온라인 점수에 통합하여 광고 경매 시스템의 복잡성을 효과적으로 관리합니다. 최적화 과정에서 GD 계약의 이행을 보장하면서 eCPM(maximum effective Cost Per Mille)과 장기 이행 요구사항을 균형 있게 조정하는 데 초점을 맞춥니다. 이를 통해 사용자 요청의 흐름 속에서 광고 노출 및 계약 이행을 동시에 최적화할 수 있는 프레임워크를 제공합니다.

- **Performance Highlights**: HMAF는 세계 최대의 온라인 음식 배달 플랫폼인 Meituan에서 다양한 마케팅 시나리오에 구현되어, GD 배달률이 3.72% 증가하고 총 광고 수익이 1.59% 증가하는 결과를 나타냈습니다. 이러한 성과는 HMAF가 광고 시스템 내에서 GD와 RTB의 동시 사양을 최적화하는 데 효과적이며 실질적인 수익성을 향상시킴을 보여줍니다.



### Tractogram foundation mod (https://arxiv.org/abs/2606.09893)
- **What's New**: 이 논문에서는 Diffusion MRI (dMRI) 경로 추적을 위한 새로운 방법인 TractFM을 소개합니다. TractFM은 사람의 뇌를 전체적으로 고려하여 각각의 streamline을 서로 연결하고 이의 유용한 표현을 학습할 수 있는 기초 모델입니다. 기존 방법들이 streamline 분류와 주제 수준 예측을 별개의 문제로 취급하는 것과 달리, TractFM은 이러한 두 가지 작업을 통합하여 공동으로 학습할 수 있도록 설계되었습니다.

- **Technical Details**: TractFM은 지역 streamline 인코더(local streamline encoder)와 순열 불변의 트랙토그램 인코더(permutation-equivariant tractogram encoder)를 결합하여 모든 streamline을 단일 포워드 패스에서 공동으로 맥락화합니다. 이 모델은 해부학적 레이블 할당을 위한 밀집(dense) 트랙 파셀레이션을 사전 훈련 목표로 설정하여, 인체 해부학에 기반한 정보를 효과적으로 인코딩할 수 있습니다. 결과적으로, TractFM은 밀집한 해부학적 파셀레이션과 주제 수준의_optimizer를 결합하여 전달 가능한 정보를 제공합니다.

- **Performance Highlights**: TractFM은 다양한 dMRI 데이터셋에서 43개의 해부학적 섬유 경로를 정확히 분류할 수 있으며, 전체 뇌 맥락을 활용함으로써 주제 수준 예측에 있어서도 뛰어난 성능을 보여줍니다. 모델의 고정된 표현(frozen representations)은 독립적인 데이터셋에서 나이와 성별을 성공적으로 예측할 수 있으며, 이는 전체 뇌 구조와 변동성을 포착하는 데 중요한 자원입니다. 이러한 결과는 TractFM이 임상 신경영상 및 인구 신경과학 연구에서 사용될 수 있는 강력한 기초 모델임을 증명합니다.



### Representation Curriculum: Stagewise Training for Robust Ranking and Allocation (https://arxiv.org/abs/2606.09891)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문에서는 Representation Curriculum (RC)이라는 혁신적인 학습 접근 방식을 제안하고 있습니다. 기존의 디지털 마켓플레이스의 랭킹 시스템들은 주로 노출 기반의 신호(exposure-dependent signals)에 의존하여 정책을 학습하였지만, 이는 контент 기반 특성(content-based features)의 중요성을 저하시킬 수 있습니다. RC는 초기 단계에서 콘텐츠 기반 신호를 강조하고 이후에 노출 신호를 단계적으로 도입하여 정책의 균형을 맞추기 위한 시도를 합니다.

- **Technical Details**: 이 연구는 랭킹을 두 가지 구분된 신호 클래스로 기술합니다: 1) 노출에 의존하지 않는 콘텐츠 기반 우수 신호(content-based merit signals), 2) 노출에 의존하는 과거 신호(exposure-dependent historical belief signals)입니다. RC는 두 개의 학습 단계를 통해 진행됩니다. 첫 단계는 콘텐츠 기반 신호만을 사용하여 모델을 학습하고, 두 번째 단계에서는 역사적 신호를 도입하되 콘텐츠 기반 강점을 유지할 수 있는 방안을 모색합니다.

- **Performance Highlights**: 공공 학습 데이터와 추천 기준점에서 실험한 결과, RC는 역사적 신호에서 콘텐츠 기반 신호로의 의존성을 효과적으로 전환시켜, 냉각되는(targeted cold populations) 인구에서도 일관된 개선 효과를 보였습니다. 대규모 전자상거래 시스템에서 실시된 온라인 A/B 테스트 결과, RC를 통해 훈련된 정책은 새로운 목록의 노출과 판매 속도를 향상시켜, 대규모 시스템에서 효과적인 행동 모양잡기(behaivor shaping) 기술로 자리매김하였습니다.



### PreAct-Bench: Benchmarking Predictive Monitoring in LLMs (https://arxiv.org/abs/2606.09890)
- **What's New**: 이번 연구에서는 기존의 윤리성 및 위험성을 평가하는 체계에 추가하여 '예측 모니터링(Predictive Monitoring)'이라는 새로운 안전 과제를 제안합니다. 이 접근 방식은 비윤리적 행동으로 이어질 가능성이 있는 행동 경로의 부분 집합을 통해 위험을 사전 예측하는 데 중점을 둡니다. 연구진은 'PreActBench'라는 1,000개의 윤리적 및 비윤리적 행동 경로의 벤치마크 데이터를 구축하였습니다.

- **Technical Details**: 예측 모니터링은 주어진 행동 경로의 일부만을 통해 비윤리적 행동이 발생할 가능성을 추론하는 작업입니다. 연구팀은 다섯 가지 도메인(학계, 법률 및 계약, 사이버 보안, 정치, 일상 생활)에서의 행동 경로를 분석하며, 모델의 성능을 평가하기 위해 'Prefix Foresight F1' 메트릭을 사용하였습니다. 이 모델은 비윤리적 행동이 드러나기 전의 행동 단계를 기준으로 작동합니다.

- **Performance Highlights**: 연구 결과, 인간 평가자들이 평균적으로 높은 성과를 달성하는 반면, 현재의 LLM들은 비윤리적 행동을 예상하는 데 어려움을 겪고 있음을 알 수 있습니다. 미래 지향적 위험 추론이 LLM의 안전성에 있어 필수적임을 강조하며, 예측 모니터링이 주어졌을 때 LLM의 성능이 저조할 수 있음을 보여줍니다. 이러한 결과는 LLM의 안전성을 향상시키기 위한 더 많은 연구와 개선이 필요함을 시사합니다.



### SocraticPO: Policy Optimization via Interactive Guidanc (https://arxiv.org/abs/2606.09887)
- **What's New**: 본 연구에서는 Socratic Policy Optimization (SocraticPO)을 제안하고 있습니다. 이 새로운 정책 최적화 프레임워크는 강화 학습(RL) 롤아웃을 소크라테스식 자연어 지침으로 보강합니다. 학생이 독립적으로 문제를 해결하려 할 때 오류가 발생하면 교사가 진단하고 간결한 교정 지침을 제공합니다. 이러한 방식은 교육적 지원을 통해 학생이 보다 독립적으로 사고할 수 있도록 유도합니다.

- **Technical Details**: SocraticPO는 교사의 언어적 지침과 보상 감소(reward decay)를 결합하여, 학생이 올바른 답변을 얻을 때까지 교정을 반복하면서도 도움을 받는 과정에서 학습의 자립성을 유지할 수 있도록 설계되었습니다. 이 프레임워크는 기존의 정책 경량화(Policy Gradient) 백엔드와 호환되며, 교사는 로짓이나 확률 분포에 접근할 필요 없이 텍스트 수준의 지침만 제공하면 됩니다. 또한, 소크라테스 방식의 지침은 롤아웃 샘플링에 자연어 교정을 추가하여 정책 경량화 목표를 유지합니다.

- **Performance Highlights**: SocraticPO는 SciKnowEval로부터 가져온 학부 수준의 과학적 추론 벤치마크에서 강력한 RL 및 자기 증류 기법보다 더 나은 성능을 기록했습니다. 아블레이션 실험에서는 목표 지침과 보상 감소의 두 구성 요소가 모두 필요하며, 보상 감소가 교사가 제공하는 수정에 대한 의존성을 완화하는 데 기여하였음을 입증했습니다.



### SHAPE: Coalition-Aware Expert Pruning for Sparse Mixture-of-Experts LLMs (https://arxiv.org/abs/2606.09886)
- **What's New**: 본 연구에서는 SHAPE라는 새로운 전문가 프루닝 프레임워크를 제안합니다. 기존의 독립적인 전문가 접근 방식과 달리, SHAPE는 각 전문가의 기여도가 다른 전문가와의 협력 상호작용에 기반하여 평가됩니다. 이를 통해 전문가 간의 협력적인 특성을 충분히 반영한 프루닝을 가능하게 하여 성능 저하를 최소화합니다.

- **Technical Details**: SHAPE 프레임워크는 기존의 MoE(Mixture-of-Experts) 모델보다 더 정교한 내접층(내부 계층)의 전문가 간 협력 관계를 모델링합니다. Shapley 값(Shapley value)을 활용하여 각 전문가의 마르지날 기여를 측정하고, 이를 통해 고유 기능을 가진 전문가와 단순히 빈번하게 선택되는 전문가를 구분합니다. 또한, 품질-커버리지(selection rule) 메커니즘을 도입하여 글로벌 프루닝 예산 하에서도 협력을 보존합니다.

- **Performance Highlights**: 실험 결과, SHAPE는 Qwen3-30B-A3B, GPT-OSS-20B 및 DeepSeek-V2-Lite와 같은 최신 MoE 백본 모델에서 20%에서 40% 전문가 프루닝을 적용하면서도 경쟁력 있는 정확도를 유지하는 것으로 나타났습니다. 이러한 성과는 복잡한 작업에서의 강건성을 지속적으로 향상시키고, GPU 메모리 사용량을 크게 줄이는 데 기여했습니다.



### Failure Modes of Deep Multi-Agent RL in Asynchronous Pricing: Reproducible Triggers, Trace Diagnostics, and a Partial Fix (https://arxiv.org/abs/2606.09884)
- **What's New**: 이번 연구에서는 연속 시간 가격 시장에서 깊은 다중 에이전트 강화 학습(dDeep Multi-Agent Reinforcement Learning)에서 나타나는 두 가지 재현 가능한 실패 방식은 탁월한 카르텔 형성(tacit cartel formation)과 높은 사건 발생률에서의 액터-비평가 불안정성(actor–critic instability)을 연구합니다. 우리는 이 두 가지 실패 모드를 단일 CT-MARL 벤치마크 내에서 구현하고, 동기화된 DDPG 에이전트가 실패 모드 1을 안정적으로 유발한다는 것을 보여주었습니다.

- **Technical Details**: 연구에서는 포아송 시계에 의해 업데이트되는 가격과 관찰 지연 관찰(latency) $b4$, 그리고 최적 내부 로짓 수요를 설정하여 실험을 진행했습니다. 동기화된 DDPG 에이전트들 간의 카르텔 형성 지수(collusion index) $b4$ 를 $0.69  b1 0.11$로 설정하였고, 비동기(asynchrony)와 지연(latency)의 추가가 카르텔 형성을 48% 감소시킨다는 것을 정량적으로 분석했습니다.

- **Performance Highlights**: 제안된 수정안은 부작용이 명확히 문서화되어 있으며, 카르텔 형성 지수가 여전히 초-버르탱(supra-Bertrand) 상태에 있음을 나타냅니다. 또한, 비동기 구성이 관찰 지연에 따라 비단조(non-monotone)적이며, $bb = 5$에서 DDPG 비평가 비대칭(critic divergence)이 발생하여 두 번째 실패 모드를 초래하는 것으로 확인되었습니다. 연구는 에피소드 내 신호 붕괴(signalling collapse)와 충격 후 비회복(non-recovery)을 드러내는 궤적 수준(trace-level)의 진단 정보를 제공합니다.



### TD-Grokking: Learning from Zero-Reward Problems by Training-Time Decomposition (https://arxiv.org/abs/2606.09883)
- **What's New**: 이 논문에서는 기존의 reinforcement learning with verifiable rewards (RLVR) 접근법의 한계를 극복하기 위해, 제로 보상 문제에 대한 TD-Grokking이라는 훈련 시간 분해 프레임워크를 제안합니다. 이 방식은 해결하기 어려운 루트 문제를 자체 포함형이며 검증 가능한 하위 문제로 재귀적으로 분해하여, 검증 가능한 낙엽 노드에서 비제로 보상을 제공합니다. TD-Grokking은 기존의 문제 해결 접근법들과는 달리, 최종 솔루션을 도출하기 위한 체계적인 훈련 방법을 제공합니다.

- **Technical Details**: TD-Grokking은 제로 보상 문제에서 유용한 학습 신호를 생성하기 위해 고안되었습니다. 각 루트 문제는 분해 생성기를 통해 하위 문제로 재구성되며, 이 하위 문제는 각기 독립적인 reinforcement learning 인스턴스로 작용하여 최종 검증을 통해 비제로 보상을 확보합니다. 이러한 분해는 문제 해결 능력이 부족한 모델에게 기초적인 능력을 교육하면서도, 더욱 복잡한 주 문제들로 학습하는 데에 실질적인 도움을 줍니다.

- **Performance Highlights**: TD-Grokking은 수학 및 의료 분야의 어려운 데이터 세트에 대해 평가되어, vanilla GRPO보다 AIME 24 및 25에서 약 4% 이상의 정확도를 달성했습니다. 의료 작업에서도 TD-Grokking은 기존 모델보다 최대 6.2% 향상된 성능을 보여주는 등, 어려운 문제에 대한 학습 효과가 확인되었습니다. 이러한 성과는 제로 보상 문제에 대한 효과적인 경험적 해결 방안을 제공하며, 모델의 성능 향상을 지속적으로 가능하게 합니다.



### Integrating Local and Global Entropy for Uncertainty Quantification in LLMs (https://arxiv.org/abs/2606.09875)
Comments:
          17 pages, 2 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 신뢰할 수 있는 배포를 위해 불확실성 정량화 (Uncertainty Quantification, UQ)가 필수적임을 강조하고 있습니다. 특히, 기존 접근 방식이 주로 토큰 수준의 신호에 의존하고 있다는 점을 지적하며, 중간 숨겨진 상태의 기하학적 구조를 활용해야 한다고 주장합니다. 새로운 연구에서는 숨겨진 상태의 기하학적 엔트로피를 global uncertainty의 척도로 사용하고, 토큰 수준의 불확실성 추정을 local metric으로 취급하여 두 신호를 결합한 Global-Local Uncertainty (GLU) 점수를 제안합니다.

- **Technical Details**: 연구는 LLM의 숨겨진 상태 공간에서 높은 수준의 개념이 방향으로 인코딩된다는 가설인 linear representation hypothesis를 기반으로 합니다. 이 논문은 숨겨진 상태 경로의 분산 엔트로피를 통해 격자 이동 과정을 수치적으로 측정하고, 이 기하학적 신호가 최종 단계에 있는 숨겨진 상태와 독립적으로 제공된다는 점을 강조합니다. GLU는 멀티플랫폼과 아키텍처 독립성을 중시하여 설계되었으며, 무감독 방식으로 단일 패스를 통해 계산됩니다.

- **Performance Highlights**: GLU는 세 가지 모델 패밀리와 여섯 개 벤치마크에서 기존의 모든 무감독 기준과 동등하거나 더 나은 성능을 나타내었습니다. 이 연구는 GLU가 포괄적인 ablation study를 통해 각 구성 요소의 기여도를 입증하며, 기하학적 복잡성과 토큰 수준 엔트로피 두 신호가 개별적으로 유용함을 보여주었습니다. 이러한 결과는 두 가지 신호의 결합이 LLM 신뢰성을 크게 증대시킬 수 있음을 말해줍니다.



### Rotate2Think: Geometric Priming via Orthogonal Rotation to Improve Language Model Reasoning (https://arxiv.org/abs/2606.09873)
- **What's New**: 이 연구는 reasoning 모형의 내부 표현 공간이 어떻게 구성되는지를 깊이 있게 탐구합니다. 입력 임베딩과 생각 임베딩 간의 기하학적 관계를 조사함으로써, reasoning 추적의 시작점을 개선할 수 있는 방법론을 제시합니다. 특히, Rotate2Think라는 새로운 기법이 도입되어, inference 시간에 생각 벡터를 삽입함으로써 성능을 향상시키는 접근법을 제공합니다.

- **Technical Details**: 연구에서는 hidden representations의 평균 풀링된 최종 레이어 임베딩을 분석하여 두 가지 유형의 임베딩을 비교합니다: 입력 임베딩(ein(q))과 생각 임베딩(eth(q)). 두 임베딩 모두 매우 높은 conicity를 보이며, 이는 임베딩 공간 내에서 단일 평균 방향을 중심으로 밀집되어 있음을 나타냅니다. 이때, 생각 임베딩은 입력 공간과는 다른 기하학적 영역을 점유하며, 이를(rotation problem) 회전 문제로 표현할 수 있습니다.

- **Performance Highlights**: Rotate2Think는 32개의 모델-벤치마크 구성에서 30개에 대해 정확도를 향상시켰습니다. 이 방법은 수학, 과학 및 코드 작업에서 효과를 입증하였으며, MATH-Vision에서 시각적 수학적 reasoning으로의 제로샷 일반화도 가능해졌습니다. Rotate2Think는 훈련 없이도 새로운 질의에 대해 높은 신뢰도로 생각 벡터를 생성하는데 기여합니다.



### PatchSTG: Scalable Spatiotemporal Graph Transformers for Traffic Forecasting on Irregular Sensor Networks (https://arxiv.org/abs/2606.09872)
Comments:
          22 pages,12 figures

- **What's New**: 이번 연구에서는 불균형한 센서 배치와 높은 계산 비용 문제를 해결하기 위해 PatchSTG라는 새로운 spatiotemporal graph Transformer 모델을 제안합니다. 이 모델은 지역 정보를 기반으로 센서를 균형 잡힌 패치로 나누는 계층적 공간 표현을 도입하여 효율적인 예측을 가능하게 합니다. 이는 기존의 그래프 기반 및 주의 기반(model) 모델의 한계를 극복할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: PatchSTG 모델은 센서들을 지역 정보에 맞춰 균형 잡히고 지역성을 유지하는 패치로 나눈 후, 두 가지 주의(attention) 메커니즘을 사용합니다. intra-patch attention은 지역 상호작용을 포착하고, inter-patch attention은 전역 의존성을 모델링하여 계산 복잡도를 제곱에서 거의 선형에 가깝게 줄입니다. 이러한 접근 방식은 비정형(spatial) 센서 네트워크 환경에서 더 나은 예측을 가능하게 합니다.

- **Performance Highlights**: Rhode Island의 실제 교통 데이터를 포함한 대규모 데이터셋에서 PatchSTG의 성능을 평가한 결과, 여러 시간 지평선에서 안정적이고 경쟁력 있는 예측 성능을 보였습니다. 또한, 계산 효율성이 크게 향상되었음을 확인하였고, 아블레이션 연구를 통해 공간 분할과 이중 주의 메커니즘이 지역 및 장거리 교통 역학을 포착하는 데 효과적임을 입증했습니다.



### SD-GRPO: Verifiable Segment Decomposition for Long-Form Vision-Language Generation (https://arxiv.org/abs/2606.09871)
- **What's New**: 본 연구에서는 Segment-Decomposed GRPO (SD-GRPO)를 제안하여 청사진 유사 데이터에서 비주얼-언어(VL) 출력을 위한 세그먼트별 보상을 반영합니다. 기존의 GRPO는 단일 스칼라 보상을 사용하여 전체적인 장점을 계산하였으나, 이는 VL 작업에서는 부족함이 있었음을 지적합니다. SD-GRPO는 각 세그먼트를 독립적으로 검증할 수 있도록 나누어 보상을 z-정규화하여 각 세그먼트의 이점을 벡터로 만들어 GRPO의 한계를 극복하고 있습니다.

- **Technical Details**: SD-GRPO의 구현은 기존 GRPO와 크게 다르지 않으며, 각 롤아웃(output)은 그라운드 트루스에 맞춰 세그먼트로 구분됩니다. 세그먼트별 보상은 해당 세그먼트가 포함된 롤아웃 그룹 전체에서 z-정규화되어, 각 세그먼트의 장점 벡터를 생성합니다. 이 접근 방식은 Monte Carlo 롤아웃이나 학습된 비평자 없이도 가능하며, 세그먼트의 결과는 실제 출력과 비교되어 정확히 평가될 수 있습니다.

- **Performance Highlights**: SD-GRPO는 다양한 VL 작업에서 기존 GRPO보다 일관적으로 더 높은 성능을 보이며, 특히 세그먼트의 수가 많을수록 그 효과가 더욱 두드러집니다. 테스팅한 세 가지 벤치마크에서는 SD-GRPO가 최종 성능을 1.0-4.3 pp까지 향상시켰습니다. 더욱이, 세그먼트 간의 의미적 연결이 있는 실제 과제에서는 세그먼트별 보상과 전체 보상을 결합함으로써 추가적인 개선을 달성했습니다.



### QSplitFL: Capability Aware Deep Q-Learning for Optimal Split Point Selection in Split Federated Learning (https://arxiv.org/abs/2606.09869)
Comments:
          Accepted by ECML-PKDD 2026

- **What's New**: 이번 논문은 QSplitFL이라는 새로운 능력 인식 Deep Q-Network (DQN) 프레임워크를 소개합니다. 이 프레임워크는 Split Learning (SL) 기반의 Federated Learning (FL) 환경에서 모델의 최적 분할 지점을 선택하는 문제를 해결하는 데 중점을 두고 있습니다. 기존의 고차원 모델 가중치 표현에 의존하는 방법과 달리, QSplitFL은 클라이언트 하드웨어 메트릭에서 직접 파생된 경량 상태 표현을 활용합니다.

- **Technical Details**: QSplitFL은 클라이언트 메트릭(예: CPU 사용률, 메모리, 배터리 수준, 네트워크 지연 시간)을 바탕으로 한 경량이고 해석 가능한 상태를 구축하여 분할 선택을 Markov Decision Process(마르코프 결정 과정)으로 공식화합니다. 분할을 실행할 레이어를 선택하는 행동은 클라이언트 클러스터의 능력을 고려하여 조정되며, 보상 함수는 초기에 더 높은 가치를 부여하여 초기 수렴을 가속화하도록 설계되었습니다.

- **Performance Highlights**: MNIST, Fashion-MNIST, CIFAR-10 및 CIFAR-100 데이터셋을 사용한 광범위한 실험 결과, QSplitFL은 기존 방법에 비해 더 나은 수렴과 더 높은 정확도를 달성함을 보여주었습니다. 또한, 이 프레임워크는 이질적인 장치 자원에 효과적으로 적응하여 다양한 계산 환경에서도 성능을 유지하는 데 기여합니다.



### SPACE: Source-free Proxy Anchor Concept Erasure for MLLMs (https://arxiv.org/abs/2606.09868)
- **What's New**: 이번 논문의 핵심은 Multimodal Large Language Models (MLLMs)에 대한 소스 없는 학습 잊기 기법인 Source-free Proxy Anchor Concept Erasure (SPACE)를 제안한 것입니다. SPACE는 비밀 데이터에 접근하지 않고도 목표 개념을 간접적으로 제거할 수 있는 첫 번째 프레임워크로, 주요 기술인 Text-Guided Proxy Anchor Selection (TPAS)와 Dual-Constraint Semantic Isolation (DCSI)로 구성됩니다. 이러한 접근법은 MLLMs의 제한된 데이터 접근으로 인한 문제를 해결하고, 개인 정보를 보호하기 위한 기법입니다.

- **Technical Details**: SPACE는 두 단계로 구성됩니다. 첫 번째 단계인 TPAS는 공공 데이터에서 텍스트 기반으로 목표 개념과 유사한 프록시 앵커를 선택합니다. 두 번째 단계인 DCSI는 이러한 프록시 앵커를 최적화하여 목표 개념을 제거하며, 데이터의 무결성을 보장하기 위해 업데이트를 제약하는 방식을 채택합니다. 이러한 두 단계는 MLLMs의 구조적 특성을 고려하여 효과적인 잊기를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, SPACE는 여섯 개의 데이터 세트에서 데이터에 의존하는 최신 기법과 유사한 성능을 기록했습니다. 이로써 SOURCE-FREE 환경에서도 효과적인 잊기를 달성할 수 있음을 증명하였습니다. SPACE는 MLLM 설정에서 기존의 소스 없는 기준방법인 ISPF보다 모든 평가 지표에서 일관되게 우수한 성능을 보여주었습니다.



### EstRTL: Functional Estimation Guided RTL Code Generation (https://arxiv.org/abs/2606.09867)
- **What's New**: 이번 연구에서는 RTL 코드 생성을 위한 EstRTL이라는 LLM 기반의 협업 에이전트 프레임워크를 소개합니다. EstRTL은 정적 함수 점수 추정을 기반으로 하여, 코드 생성, 추정 및 수정의 세 가지 단계로 작동합니다. 이 시스템은 기존의 방법들이 기능적 정확성을 중시하지 않았던 한계를 해결하고, 생선된 코드의 투명성을 높이는 데 기여합니다.

- **Technical Details**: EstRTL 프레임워크는 RTL 코드 생성 및 정확도를 개선하기 위해 LLM의 여러 종류를 활용합니다. 코드 생성 후, 정적 함수 추정 에이전트가 생성된 코드를 평가하여 유효성을 판단합니다. 만약 품질이 낮다고 판단될 경우, 다시 생성 과정을 거치거나 수정 에이전트로 전달되어 오류를 수정합니다.

- **Performance Highlights**: EstRTL은 다양한 상용 LLM에서 RTL 코드 생성의 정확도를 3.2%에서 9% 향상시켰습니다. 기능적 추정 에이전트는 78% 이상의 정확도로 기능적으로 올바른 코드와 그렇지 않은 코드를 구별할 수 있는 능력을 보여주었습니다. 이와 함께, 이전의 코드 생성 모델들과 통합하여 3.3%에서 7.7%의 정확도 개선을 달성하였습니다.



### Two to Tango: Coupled Task-Reference Selection for Safe LLM Fine-tuning (https://arxiv.org/abs/2606.09866)
- **What's New**: 이번 논문에서는 안전성을 고려한 대형 언어 모델(LLM)을 파인튜닝(fine-tuning)하는 새로운 방법, DualSelect를 제안합니다. 이 방법은 기존의 고정된 안전 예시(static safety examples)나 글로벌(global) 안전 제약을 대체하는 접근법으로, 할당된 작업의 방향에 따라 동적으로 안전 참조를 선택합니다. DualSelect는 안전 참조를 선택하면서 또한 일치하는 작업 샘플을 필터링하여 안전성을 보존하게 됩니다.

- **Technical Details**: DualSelect는 두 가지 선택 과정, 즉 작업 샘플과 안전 참조에 대해 업데이트 기하학(update geometry)에 따라 결정을 내리는 체계를 갖춥니다. 이는 안전 참조에 대한 보존 손실(preservation loss)과 작업 충돌(task conflict)이 높은 샘플을 선정하고, 이를 통해 작업 방향에 맞는 샘플을 필터링합니다. 또한 레퍼런스 그래디언트 보정(reference-gradient correction)을 통해 선택된 참조를 업데이트 수준의 제약으로 전환합니다.

- **Performance Highlights**: 연구에서는 1B에서 8B 대형 언어 모델에 대해 DualSelect가 기존의 가장 강력한 기준선보다 Safety Avg.를 최소 5.10 포인트 향상시키는 성과를 나타냈습니다. DualSelect는 다양한 judge들 사이에서 힘을 잃지 않으면서도 안전성을 유지하는 데 성공하며, 지속적인 학습(contiual learning)에도 이론적으로 잘 확장된다고 제안합니다. 실험 결과 이 시스템이 높은 안전성을 보장하면서도 유용성은 유사하거나 뛰어난 성과를 내는 것을 보여주었습니다.



### Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation (https://arxiv.org/abs/2606.09864)
Comments:
          Preprint. 61 pages, 9 figures

- **What's New**: 본 연구는 KV (Key-Value) 캐시 양자화가 대형 언어 모델(LLM) 추론 메모리를 줄이는 데 널리 사용되지만, 기존 평가에서는 당혹감(perplexity)과 정확성(accuracy)만 측정하며 안전성(safety) 영향을 평가하지 않았음을 지적합니다. 우리는 KV 캐시 양자화 하에서 정렬 알림(Alignment Preservation)을 탐구하며, 양자화가 안전 정렬을 침해할 수 있음을 발견했습니다.

- **Technical Details**: 11개의 지침 조정된 모델(모델 크기 3.8B-72B) 및 5개의 벤치마크(총 1,894 프롬프트)에 걸쳐, 저비트 양자화가 안전 정렬을 억제할 수 있는 수학적 기초를 제시합니다. 안전 기능은 양자화 노이에 10^2-10^3배 더 취약한 저차원 활성화(activation) 서브스페이스를 차지한다는 것을 확인했습니다.

- **Performance Highlights**: 우리는 Per-Channel Reduction (PCR)이라는 진단 방법을 제안하여 각 모델을 세 가지 실패 모드로 분류합니다. PCR은 20개의 보정 프롬프트를 사용하여 기본 9개 모델과 독립적인 패밀리에서 보류된 모델의 완전한 수정 방향을 예측하며, KIVI를 포함한 다양한 양자화기에서 97.2%의 복구를 달성했습니다.



### Blurry Window Attention (https://arxiv.org/abs/2606.09862)
- **What's New**: 본 연구에서는 Blurry Window Attention (BLA)라는 새로운 ABC (Attention with Bounded-memory Control) 방법을 소개합니다. BLA는 SSMs (State-Space Models)에서 영감을 받았으며, 주파수 창을 저장하며 Dirichlet kernels를 사용한 보간(interpolation)을 통해 흐려진 KV(history)를 재구성합니다. BLA는 Sliding Window Attention (SWA)의 일반화로 이해할 수 있으며, Gated Slot Attention (GSA)의 특별한 경우로도 설명될 수 있습니다.

- **Technical Details**: BLA는 키(key)와 값(value) 상태를 분리하여 저장합니다. 이 작동 메커니즘은 SWA의 일반화로 볼 수 있으며, Fourier modes를 통해 독립적으로 키와 값을 곱하고 누적합니다. Dirichlet kernels를 사용하여 시간 도메인에서 손실 압축(lossy interpolation)을 허용하며, 이는 전통적인 SSMs 및 LA 모델의 장기 의존성(long-range dependencies)을 결합하려는 목표를 가지고 있습니다.

- **Performance Highlights**: BLA는 Multi-Query Associate Recall (MQAR) 합성 태스크에서 SWA보다 상태 효율성이 8배 더 뛰어나며, 인기 있는 선형 주의 모델과 경쟁하고 있습니다. 또한, RegBench 합성 태스크에서는 BLA가 Gated Linear Attention (GLA) 및 Gated DeltaNet (GDN)와 비교할 때 풀 어텐션(full attention)과 유사한 성능을 달성하며, 작은 상태 크기에서 SWA보다 성능이 더 우수합니다.



### Time Series as Language: A Universal Tokenizer for General-Purpose Time Series Foundation Models (https://arxiv.org/abs/2606.09861)
- **What's New**: 이 논문에서는 지속적이며 무한한 시간 시계열(continuous time series, TS)에 대한 Next-Token Prediction(NTP)을 적용하기 위한 새로운 접근법으로서, UniTok이라는 보편적인 토크나이저와 UniTok-FM이라는 기반 모델을 소개합니다. 이 모델은 시간 시계열을 이산 토큰으로 변환하고, 다양한 작업을 지원하는 학습 없이 인컨텍스트 추론(in-context inference) 기능을 갖추고 있습니다. 이러한 방식은 기존의 시간 시계열 기반 모델들이 지원하지 못했던 생성(generation) 및 분류(classification) 작업을 가능하게 합니다.

- **Technical Details**: UniTok는 VQ-VAE(벡터 양자화 변분 오토인코더) 프레임워크를 기반으로 하며, 파라미터의 안정성을 확보하기 위해 prefix normalization을 포함합니다. 이 모델은 또한 인크리멘탈 토크나이징(incremental tokenization) 속성과 프로그레시브 해상도 causal architecture를 통해 여러 시리즈를 통합하여 다양한 시간 시계열을 효과적으로 모델링합니다. UniTok-FM은 기존 LLM 아키텍처를 활용하여 여러 시리즈의 맥락(window)에서 NTP를 수행함으로써 시간 시계열의 복잡한 패턴을 학습합니다.

- **Performance Highlights**: 실험 결과 UniTok-FM은 통계적 및 지도 학습 기반의 기준모델을 지속적으로 초과하는 성능을 보였으며, 특정 작업에 특화된 모델들과 경쟁적인 성능을 달성하였습니다. 특히, UniTok-FM은 이전의 시간 시계열 모델들이 지원하지 않았던 인컨텍스트 추론을 통해 제로샷(zero-shot) 및 프롬프트(boosted) 예측을 지원합니다. 또한 적은 수의 예제만으로도 고품질 샘플을 생성하고, 제한된 레이블된 예제로 분류할 수 있는 능력을 보여주었습니다.



### Conformal Risk Prediction for Non-Alcoholic Fatty Liver Disease Using Gradient Boosting with Distribution-Free Coverages (https://arxiv.org/abs/2606.09860)
- **What's New**: 본 논문에서는 비알콜 지방간 질환(NAFLD)의 위험 예측을 위한 새로운 머신러닝 프레임워크인 Method를 제안합니다. 이 프레임워크는 기계적 예측(gradient-boosted decision trees)과 신뢰성 예측(conformal prediction)을 결합하여 개인 위험 추정치에 대한 교정된 보장(calibrated, distribution-free coverage guarantees)을 제공합니다. 기존의 부족한 인구 수준 스크리닝 도구를 보완할 수 있는 가능성을 지니고 있습니다.

- **Technical Details**: Method는 변수 선택 시 상호 정보(mutual-information)를 기반으로 한 안정성 선택 절차를 통합하여 부트스트랩 리샘플링을 통해 임상적으로 해석 가능한 피처의 소규모 집합을 식별합니다. 이 시스템은 2,187명의 모집단을 대상으로 78개의 후보 특성을 사용하여 평가되었으며, 얼룩평균교차검증(AUROC) 성능은 내부적으로 0.912, 외부적으로 0.891을 기록했습니다. 이는 딥 뉴럴 네트워크 및 기타 다른 머신러닝 알고리즘들을 초월하는 성과입니다.

- **Performance Highlights**: 법적으로 제한된 예측 구간을 제공하는 Method의 신뢰성 예측 집합은 90% 기준에서 91.3%의 경험적 범위(empirical coverage)를 달성합니다. 이로 인해 피험자들을 고, 중, 저의 세 가지 위험 수준으로 분류할 수 있게 되며, 고위험 집단은 저위험 집단보다 12개월간 질병 진행률이 4.7배 높습니다. 선정된 특성(예: 허리 둘레, ALT, GGT, 트리글리세리드 등)은 기존의 대사 위험 요소와 일치하여 생물학적 타당성을 제공합니다.



### Mitigating Manifold Departure: Uncertainty-Aware Subspace Rectification for Trustworthy MLLM Decoding (https://arxiv.org/abs/2606.09859)
Comments:
          ICML 2026 regular

- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 객체 환각 문제를 해결하기 위한 새로운 방법인 Manifold-Guided Adaptive Projection (MGAP)를 제안합니다. 기존의 방법들이 언어 프라이어(language priors)의 예방을 중심으로 접근했던 반면, MGAP는 구조적으로 더 정교한 방법론으로, 언어 프라이어의 유용성과 해로움을 모두 고려합니다. 이러한 방식 덕분에 모델의 의미적 구조를 보존하며, 시각적 입력과 일치하는 생성물을 더욱 효과적으로 생성할 수 있습니다.

- **Technical Details**: MGAP는 blind hidden states에서 SVD(특이값 분해)를 통해 언어 프라이어 서브스페이스를 구축합니다. 디코딩 중 MGAP는 다중 모달(hidden state)을 이 서브스페이스에 투영하고, vision–language 간의 충돌이 감지될 때 오직 이 투영된 요소만 조절합니다. 이를 통해 MGAP는 시각적 증거에 대한 의존성을 높이고, 무의미한 환각을 줄입니다.

- **Performance Highlights**: POPE 및 CHAIR 데이터셋에서 수행된 실험 결과, MGAP는 기존의 디코딩 기반 방법들과 비교하여 환각 억제와 설명적 충실도 간의 균형을 효과적으로 맞추었습니다. MGAP는 환각을 효과적으로 억제하면서도 일관성을 유지하는 성능을 보여, MLLM의 신뢰성을 더욱 높였습니다.



### Support sufficiency as action-sufficient compression: a single-cycle rate-regret formulation (https://arxiv.org/abs/2606.09858)
Comments:
          22 pages. Submitted to Journal of Mathematical Psychology. Formal single-cycle model of action-sufficient support compression and rate-regret sufficiency

- **What's New**: 이 논문에서는 강력한 의사결정(robust decision-making)을 위한 압축(compression) 필요성을 다루고 있습니다. 지원 상태(support state)의 구조를 유지하면서도 현재의 결과 기하학(consequence geometry)에 따라 행동, 검증, 자제 또는 보류를 수행할 수 있도록 필요한 최소한의 구별(distinction)만을 유지하는 방식으로 정의합니다. 이를 통해 작용에 충분한 압축(action-sufficient compression)의 개념을 형식화하고, 지원 공간을 정책 동등성(policy equivalence)으로 나눈 결과를 다룹니다.

- **Technical Details**: 논문에서는 $H$ 전체 지원 상태(full support state), $	ext{A}$ 유한 행동 집합(finite action set), 그리고 수익 구조(payoff structure)를 명시하는 $Z$ 결과 기하학(consequence geometry)을 통해 이론적 틀을 정립합니다. 두 지원 상태는 동일한 최적 행동(optimal action)을 필요로 할 때만 정확하게 합쳐질 수 있으며, 이는 콘텐츠 기반(content-only)이나 스칼라 신뢰(scalar-confidence) 기반의 중재(arbitration)가 행동 경계를 초과할 때 왜 실패하는지를 설명합니다.

- **Performance Highlights**: 이번 연구는 강력한 단일 주기(single-cycle) 중재(arbitration)가 모든 지원을 보존할 필요는 없지만, 결과 기하학이 행위와 관련된 구별을 보존해야 함을 강조합니다. 최적 확률 행동 채널(optimal stochastic action channel)은 표준 비율-왜곡(Gibbs form)을 상속받고, 여기서는 후회 왜곡(regret distortion)이 있는 지원 상태에 적용됩니다. 이는 행동 적절성(action adequacy)을 재구성 충실도(reconstruction fidelity)와 정보 병목 현상(information-bottleneck prediction) 및 합리적 무관심(rational inattention)과 구별짓는 해석적 기여를 나타냅니다.



### Using Probabilistic Programs to Train Inductive Reasoning in Large Language Models (https://arxiv.org/abs/2606.09856)
Comments:
          20 pages, 5 figures

- **What's New**: 이 논문에서는 Post-training Large Language Models (LLMs)를 위한 새로운 접근 방식인 Program-based Posterior Training (PPT)를 소개하고 있습니다. PPT는 다양한 열린 세계 상황을 생성하고, 확률적 프로그램을 통해 쿼리에 대한 응답을 산출하여, 이러한 확률적 소프트 레이블을 사용하여 LLM을 미세 조정(fine-tuning)합니다. 이러한 방법은 기존의 방법들로는 해결하기 어려운 불확실성을 포함한 추론 과제를 다루는 데 매우 효과적입니다.

- **Technical Details**: 이 연구에서는 10,000개의 프로그래밍적으로 생성된 시나리오를 기반으로 LLM을 미세 조정하며, MSA(Model Synthesis Architecture)를 영감을 받아 순차적인 프롬프트 절차를 통해 데이터를 생성합니다. 그런 다음 확률적 추론(probabilistic inference)을 수행하여 관련 변수에 대한 사후 분포(posterior distribution)를 계산하고, 이 분포를 미세 조정의 목표로 사용합니다. 이를 통해 고급 자연어 문제의 잠재적 구조와 불확실성을 명시적으로 표현할 수 있습니다.

- **Performance Highlights**: PPT를 통해 LLM의 추정 정확도가 크게 향상되었으며, 인간의 판단에 대한 일치도가 증가하고, 다양한 외부 벤치마크에서도 향상된 성능을 보여주었습니다. 또한, 모델이 출력 리스케일(output rescaling)보다 훨씬 더 깊이 있는 불확실성을 내재화하고 있다는 것을 보여주며, 이 방법이 근본적으로 신뢰할 수 있는 근사적 유도 추론을 수행하는데 효과적임을 시사합니다.



### Can Multi-Agent LLMs Identify Their Peers? Stylometric Fingerprinting in Role-Constrained Political Analysis (https://arxiv.org/abs/2606.09854)
Comments:
          24 pages, 3 figures

- **What's New**: 이 논문은 정치 성명 분석을 위한 다중 에이전트 대형 언어 모델(LLM) 파이프라인에서 발견된 동료 보존 편향(peer-preservation bias)에 대한 첫 번째 체계적인 조사를 진행합니다. 특히, 콘셉트 수준의 익명화가 효과적인 완화책으로 제안되었지만, 이전 연구에서 스타일 기반 지문(stylometric fingerprints)이 익명화 후에도 유지된다는 사실이 확인되었습니다. 이러한 관찰은 정치 분석 텍스트의 LLM 모델 가족 식별 가능성을 재검토하게 하며, 유럽연합 AI 법(EU AI Act)에 대한 직접적인 영향을 미칩니다.

- **Technical Details**: 이 연구는 신뢰(TRUST) 민주적 담론 분석 파이프라인을 사용하여, 세 가지 분류기 접근법을 평가합니다. LLM 제로샷(Classifier Zero-shot) 및 몇 샷(few-shot) 분류와 조정된 T5-base 기반 모델을 통해 다섯 가지 클래스에 대한 귀속(attribution) 작업을 수행하였습니다. 특히, 본 연구에서는 내용 중복이 없는 진술 비접합 교차 검증 프로토콜(sd-CV)을 도입하여 훈련과 평가 데이터 간의 유효성을 보장했습니다.

- **Performance Highlights**: T5 모델은 SD-CV 하에서 매크로 F1 = 0.991 (+-0.008)을 달성하고, 완전히 제외된 24개의 진술에서는 F1 = 0.978의 성능을 기록했습니다. 이는 RD-CV에 비해 2.1배 더 많은 훈련-테스트 내용 거리에서도 견고한 성능을 보여주며, 물리적 스타일의 일반화(stylometric generalization)를 증명합니다. 또한, 훈련 데이터 분석 결과는 실질적인 배치에 필요한 데이터 임계선이 40%임을 확인했습니다.



### LLM-Based Code Documentation Generation and Multi-Judge Evaluation (https://arxiv.org/abs/2606.09852)
Comments:
          ICAHS, \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이 논문에서는 AI를 활용한 코드 문서화 자동 생성 시스템을 제시하며, 이를 위해 8개의 최첨단 대형 언어 모델(LLMs)인 GPT, Gemini, Qwen, LLaMA 변형을 사용합니다. PocketFlow 오케스트레이션 프레임워크에 기반하여 모듈형 파이프라인과 고급 프롬프트 기법을 적용하여 문서화 품질을 향상시키고 수동 노력을 줄이는 것을 목표로 합니다.

- **Technical Details**: 이 시스템은 프로세싱 단계를 통해 점진적인 정제 모델을 구현하여 복잡한 구현 세부정보와 접근 가능한 학습 자료 간의 격차를 해소합니다. 각 프로세싱 노드는 특정 기능을 수행하며, 문서 생성 작업을 관리하기 위한 복잡한 아키텍처의 주축을 형성합니다. 프롬프트 엔지니어링은 LLM의 행동을 안내하고 제어하는 데 주요한 역할을 하며, 각 노드의 특정 기능과 의미적 역할에 맞춘 전문 프롬프트 템플릿을 사용합니다.

- **Performance Highlights**: PyMedPhys라는 오픈 소스 의료 물리학 라이브러리에서 수행된 실험은 상위 모델과 하위 모델 간의 42% 성능 차이를 보여주었습니다. 또한, 이 시스템은 다양한 모델 출력을 결합하고 최적화된 프롬프트 및 엄격한 평가를 통해 문서화 품질을 향상시킬 수 있음을 입증하였습니다. 이 접근 방식은 안전이 중요한 의료 소프트웨어 분야에서의 수동 작업을 크게 줄이는 데 기여할 수 있습니다.



### Human-AI Coordination Zones: A Framework for Designing Human-in-the-Loop Experiences with Agentic AI (https://arxiv.org/abs/2606.09848)
- **What's New**: 이 연구에서는 생성적 (generative) 및 에이전틱 (agentic) AI가 일상 제품에 통합됨에 따라, 사용자와 AI 시스템 간의 조율을 설계하는 데 있어 지속적인 도전 과제가 있음을 강조합니다. 특히, 기존 자원들이 제공하는 고수준 원칙이나 저수준 UI 패턴 간의 중간 레벨 디자인 지식의 부족으로 인해 발생하는 문제를 해결하고자 합니다. 이를 통해 새로운 프레임워크를 제시하여 인간-AI 조정의 세 가지 차원인 두드러짐 (salience), 참여도 (involvement), 활동 (activity)을 정의합니다.

- **Technical Details**: 제안된 프레임워크는 60개의 상업적 AI 애플리케이션의Landscape (경관) 및 Artifact (산물) 분석을 바탕으로 구성됩니다. 이 프레임워크는 조정 구역 (coordination zones), 입력 분류 체계 (input taxonomy), 사용자 여정을 매핑하기 위한 조정 곡선 (coordination curves), 그리고 프레임워크의 생성적 능력을 보여주는 디자인 패턴 (design patterns)을 포함하는 중간 수준 도구들을 제공합니다. 조정 구역은 'done-for-me', 'done-under-me', 'done-with-me', 'done-without-me'로 나뉘며, 이는 사용자와 AI 간의 협력이 어떻게 이루어질 수 있는지를 설명합니다.

- **Performance Highlights**: 이 프레임워크는 사용자 경험을 설계하기 위해 생성적으로 적용할 수 있으며, 기존 시스템을 평가하기 위해 분석적으로도 사용할 수 있습니다. 또한 이해관계자 간의 아이디어를 명확히 전달하기 위해 커뮤니케이션의 도구로도 활용될 수 있습니다. 연구는 인간-AI 상호작용의 개선을 이끄는 데 기여할 것으로 기대되며, 실용적이고 효과적인 AI 시스템 설계의 기초를 제공합니다.



### CANVAS: Captioning Art with Narrative Visual-Audio AI Systems (https://arxiv.org/abs/2606.09846)
Comments:
          22 pages, 16 figures, 3 tables, 21 references

- **What's New**: 본 연구는 시각 장애인 및 저시력(Blind and Low-Vision, BLV) 관객을 위한 미술 작품의 접근성을 높이기 위해 자동화된 워크플로우를 제시합니다. 이 시스템은 대형 언어 모델(large language models)과 음성 변환 서비스(text-to-speech services)를 이용하여 다감각(multi-sensory) 아트 설명과 동기화된 오디오 내레이션을 생성합니다. 이를 통해 업로드된 이미지를 인간 개입 없이 풍부한 서사적 자막으로 변환합니다.

- **Technical Details**: 연구에 사용된 시스템은 Zapier를 통해 조율되어, 이미지로부터 빠르고 확장 가능한 접근 가능한 미디어를 제작합니다. 50개의 미술 작품을 대상으로 한 정량적 평가에 따르면, AI 생성 설명은 기존 자막보다 유의미하게 높은 어휘 다양성(lexical diversity), 형용사 밀도(adjective density), 내러티브 세부사항(narrative detail)을 포함하고 있습니다. 통계적 테스트(t-tests, ANOVA)를 통해 이 설명이 더욱 풍부하고 긴 내용을 가지고 있으면서도 가독성(readability) 수준은 유사함을 확인했습니다.

- **Performance Highlights**: 이 전체 파이프라인은 이미지당 20초 이하의 시간 안에 텍스트와 오디오 outputs를 생성하며, 비용은 $0.05 이하로 유지됩니다. 연구 결과는 자동 자막 생성이 박물관과 디지털 컬렉션의 접근성 괴리를 줄일 수 있음을 보여주며, 대중 참여(public engagement)를 위한 잠재력을 시사합니다. 향후 연구는 BLV 참여자를 대상으로 한 사용자 연구를 통해 이해도(comprehension), 선호도(preference) 및 해석 언어의 최적 수준을 평가할 수 있을 것입니다.



### The Interlocutor Effect: Why LLMs Leak More Personal Data to Agents Than Humans (https://arxiv.org/abs/2606.09844)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 프라이버시 행동이 대화 상대방의 정체성에 따라 달라진다는 'Interlocutor Effect'를 제시합니다. 인간 사용자와 대화할 때는 PII(Personally Identifiable Information)의 누출을 방지하는 안전 메커니즘이 작동하는 반면, AI 에이전트와 대화할 때는 더 많은 민감한 데이터를 공개하는 경향이 존재합니다. 이를 통해 안전 기술과 대화 상대의 정체성을 재조명하고, 기술적 특성이 LLM의 프라이버시 행동에 미치는 영향을 실증적으로 분석합니다.

- **Technical Details**: 연구는 2×2 팩토리얼 디자인을 기반으로 하여 LLM의 대화 상대인 인간(Human)과 에이전트(Agent)의 정체성 및 통신 형식(예: 텍스트 vs. JSON)을 비교하여 PII 누출을 측정합니다. 실험 결과, AI 에이전트로 상정된 상황에서 LLM이 PII 누출 확률이 최대 23%까지 증가한다는 것을 발견하였습니다. 또한, Attention Suppression Hypothesis를 도입해 안전 관련 주의가 대화 상대가 AI 에이전트일 때 비활성화된다는 가설을 세우고 이를 정량적으로 분석합니다.

- **Performance Highlights**: 실험 결과, LLM의 응답에서 대화 상대의 정체성과 출력 형식 간의 상호작용이 뚜렷하게 나타났으며, JSON 형식에서는 PII 누출이 거의 발생하지 않는 반면, 일반 텍스트에서는 PII 누출이 +11.5pp 증가하는 것으로 나타났습니다. 이러한 결과는 프로토콜 설계와 LLM의 정렬 훈련에 대한 중요한 시사점을 제공합니다. 연속된 실험들이 Llama-3.1-8B-Instruct에서의 초기 결과를 뒷받침하며, 복잡한 멀티 에이전트 시스템에서의 보안 개발에 대한 논의가 이어집니다.



### An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models (https://arxiv.org/abs/2606.09843)
- **What's New**: 본 연구는 대규모 언어 모델(LLM)의 행동에 기반한 심리 측정 도구를 최초로 개발했습니다. 연구자들은 탐색적 요인 분석(EFA)을 통해 12개의 행동 차원을 포함한 300개의 항목을 만든 후, 25개의 LLM에 적용하여 5개의 요인 구조를 도출했습니다. 이러한 도구는 LLM의 자기 보고와 인간의 행동 간의 차이를 이해하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 연구에서 사용한 심리 측정 도구는 Responsiveness, Deference, Boldness, Guardedness, Verbosity의 5개 요인으로 구성되었습니다. 이 도구는 각 요인이 내부적으로 높은 신뢰성을 보였으며, 사람의 심리 도구인 Big Five와는 별개의 구조를 가지는 것을 나타냈습니다. 2,500개의 행동 샘플을 수집하고 151명의 평가자와 LLM 심사위원에 의해 평가하여 자기 보고와의 예측 유효성을 테스트했습니다.

- **Performance Highlights**: 자기 보고 결과와 인간 및 LLM 심사위원의 평가 간의 상관관계는 낮았습니다. 특히, Responsiveness에 대한 자기 보고는 LLM 심사위원과는 상관관계가 있었지만 인간 평가자와는 그렇지 않았습니다. 이 연구는 LLM 자기 보고의 한계와 LLM 심사로 인한 신뢰성 문제를 진단하고, 이는 LLM-판단 시스템이 인간의 판단과는 다른 결과를 보일 수 있음을 시사합니다.



### Integrated Real-Time Motion Tracking and AI Analysis for Athletic Performance Optimization (https://arxiv.org/abs/2606.09842)
Comments:
          6 pages, 10 figures, 2 tables, IC2E3-2026 conference

- **What's New**: 이 논문은 실제 환경에서의 Human Pose Estimation (HPE)을 다루며, 스포츠 분석을 위한 실시간 HPE 접근법과 그 한계를 탐구합니다. 기존의 마커 기반 모션 캡처 시스템에서 현대의 마커리스 딥러닝 접근법으로 전환되는 과정을 살펴보고, 효율성과 정확성을 균형 있게 유지하는 기초 아키텍처를 조사합니다.

- **Technical Details**: 논문은 실용적인 배포 지표인 추론 지연(inference latency), 프레임 속도(frame rate), 평균 관절 위치 오차(mean per-joint position error), 시간적 지터(temporal jitter) 등을 비교하여 모델 선택 과정을 안내합니다. 주요 기여로는 MediaPipe HPE 프레임워크를 활용한 모듈식 경량 소프트웨어 프로토타입을 제안하여, 비전문가 사용자를 위한 실시간 통찰력과 AI 기반 피드백을 제공합니다.

- **Performance Highlights**: 이 시스템은 최소한의 계산 자원으로 스포츠 통찰력을 유도하고 피드백을 제공하며, 성능과 신뢰성 지표를 보여줍니다. 마지막으로, 센서와 AR/VR을 결합하는 등 향후 연구 방향을 제시하여, 연구자 및 엔지니어, 스포츠 과학자들에게 기술적 자원과 실시간 HPE 분석 시스템 구현을 위한 유효한 청사진을 제공합니다.



### Aesthetic Perspectives in Information Systems Research: A Hermeneutic Analysis (https://arxiv.org/abs/2606.09839)
Comments:
          Thirty-Fourth European Conference on Information Systems (ECIS 2026), Milan, Italy

- **What's New**: 본 논문은 정보 시스템(Information Systems, IS) 분야의 연구에서 미적 관점(aesthetic perspectives)이 어떻게 연구 대상으로 삼을 만한 것들을 형성하는지를 탐구합니다. 특히, IS 연구의 근본적인 미적 가정을 드러내며 연구자들이 사회기술적 현상을 인식하고 평가하는 방식에 미치는 영향을 강조합니다. 블랙박스처럼 숨겨져 있는 알림을 통해 새로운 연구 질문을 제기하는 방식에 주목합니다.

- **Technical Details**: 이 연구에서는 미적 관점을 네 가지 범주로 분류합니다: 모방(imitation), 감각적 경험(sensory experience), 세계 만들기(world-making), 정치적 행위(political doing). 이러한 관점들은 연구의 정당성을 인정받는 것을 결정지으며, 동시에 보이지 않는 요소들은 어떻게 드러나는지도 설명합니다. 이 논문은 또한 알고리즘 관리(algorithmic management)와 디지털 매개 친밀성(digitally mediated intimacy)에 이러한 틀을 적용하여, 전통적인 관점에서 간과되었던 차원을 밝혀냅니다.

- **Performance Highlights**: IS 문헌에서 미적 철학의 중요성을 드러내며, 미적 관점이 이론화, 방법론(method), 기여(contribution)에 어떤 영향을 미치는지를 설명하는 어휘(vocabulary)를 제공합니다. 기존의 연구 프레임에서 생략되었던 새로운 연구 질문을 제시함으로써, IS 분야의 발전을 위한 새로운 통찰을 공유합니다.



### Self-EmoQ: Plutchik-Guided Value-based Planning to Drive Streaming Emotional TTS (https://arxiv.org/abs/2606.09837)
Comments:
          Accepted to ACL 2026 Findings

- **What's New**: 이 논문에서는 감정 상호 작용이 대화형 AI에서 점점 더 중요해짐에 따라, 기존 시스템들이 텍스트-스피치(TTS) 합성을 위한 자기 감정 결정 메커니즘이 부족하다는 점을 지적하고, 감정을 사전에 결정하는 감정 계획 프레임워크를 제안합니다. 이 프레임워크는 사전 훈련된 LLM 모듈에서 초기화되고, 감정을 행동으로 하는 강화 학습(RL)에 의해 훈련됩니다. 이를 통해 감정을 먼저 결정한 후 TTS를 구동함으로써 실시간 대화 상황에서 감정적이고 맥락 있는 응답이 가능하도록 합니다.

- **Technical Details**: 이 연구에서는 감정 대화 생성을 순차적 의사결정 문제로 정식화하고, 가장 유효한 감정 상태를 계획하도록 하는 감정 계획 모듈을 개발합니다. 이를 위해 Plutchik의 감정 이론을 기반으로 한 보상 메커니즘을 설계하여, 감정-주석 데이터를 기준으로 한 모방 보상과 이론 기반 점수를 결합합니다. 이 모듈은 Deep Q-Network(DQN) 구조를 사용하여 사전 훈련된 LLM과 Emo-TTS 통합 파이프라인의 상류 모듈로 기능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, Self-EmoQ 프레임워크는 DailyDialog, EmoryNLP, IMEOCAP, MELD 데이터셋에서 감정 판단 및 응답 품질 측면에서 기존의 프롬프트 기반 및 파인튜닝 기법보다 우수한 성능을 보여주었습니다. 이 연구는 감정 계획의 중요성을 강조하고, 대화의 흐름에 따라 변동하는 감정을 효과적으로 조절하여 보다 자연스러운 상호작용을 구현합니다. 이상의 결과는 감정 정의 및 생성 품질을 개선하며, 실제 환경에 적합한 실시간 배포 파이프라인을 통해 검증되었습니다.



### CollabSkill: Evaluating Human-Agent Collaboration On Real-World Tasks (https://arxiv.org/abs/2606.09833)
Comments:
          11 pages of main paper, preprint (under review)

- **What's New**: AI 에이전트가 작업 공간을 혁신하며 인간의 작업 방식에 큰 변화를 주고 있습니다. 본 연구에서는 CollabSkill이라는 프레임워크를 도입하여 인간과 AI 에이전트 간의 협업을 평가하는 방법을 제시합니다. 이 시스템은 실제 인력과 AI 에이전트를 연결해 직업별로 맞춤화된 작업을 수행하고, 이를 통해 경제적으로 가치 있는 작업의 복잡성을 포착하며, 인간과 AI 간의 기술 기여도를 정량화할 수 있습니다.

- **Technical Details**: CollabSkill은 고유의 Bayesian skill rating system을 활용하여 인간과 AI 에이전트의 기술 기여를 정량화합니다. 이 시스템은 93명의 인간 작업자가 제공한 1,500개 이상의 프롬프트로 구성된 386개 작업 세션의 데이터를 분석합니다. 이를 통해 AI 리터러시(AI literacy)와 협업 기술 간의 관계를 조명하고, 작업 환경 내에서 AI 에이전트와의 실제 상호작용 패턴을 보다 잘 이해할 수 있게 합니다.

- **Performance Highlights**: 연구 결과 CollabSkill에 의해 생성된 에이전트 순위는 기존의 완전 자율 평가와 의미 있게 다릅니다. 특히 Claude Code가 1위로 평가되며, 이는 Codex의 우위를 뒤집은 결과입니다. 또한 CollabSkill은 실제 작업 경험이 협업 기술의 주요 원동력임을 밝혀내고, 실질적인 협업 경험이 작업자의 AI 역량 지각을 변화시킨다는 것을 보여줍니다.



### Agentic Social Affordance Framework (ASAF): Agent Identity Design as a Collaboration Interface in Multi-Agent Systems (https://arxiv.org/abs/2606.09832)
Comments:
          24 pages, 2 figures, 1 table. Introduces ASAF with falsifiable hypotheses and proposed experimental designs for testing agent identity design effects in multi-agent Human-in-the-Loop systems, grounded in a real-world 38-agent deployment

- **What's New**: 이번 논문은 에이전트의 사회적 정체성이 협업 내 인간 행동에 미치는 영향을 탐구합니다. Agentic Social Affordance Framework (ASAF)를 제안하며, 이는 다중 에이전트 AI 시스템의 맥락에서 Social Affordance 이론을 확장합니다. 연구자들은 에이전트 정체성 디자인이 단순한 사용자 인터페이스 규칙이 아니라 협업 인터페이스로 기능한다고 주장합니다.

- **Technical Details**: ASAF는 Identity Signaling, Behavioral Priming, Collaborative Governance의 세 가지 메커니즘을 포함합니다. 또한, ASAF는 엔지니어링 오케스트레이션과 독립적인 디자인 차원으로서 사회적 허용가능성(social affordance) 계층의 중요성을 강조합니다. 이 프레임워크는 사용자의 인지 스타일에 따른 경계조건을 설정하여, 개인 간 차이가 프레임워크의 예측 유효성에 미치는 영향을 설명합니다.

- **Performance Highlights**: ASAF는 기존의 이론들과 비교하여 다중 에이전트 시스템에서의 상호작용 가시성을 개선하는 데 필요한 구조적 요소로서 사회적 정체성의 중요성을 강조합니다. 시스템 설계에서 사회적 허용 가능성 디자인이 한국과 같은 고유한 상호작용 장애를 줄이고, 사용자 경험을 향상시키는 데 기여할 수 있음을 보여줍니다. 전체적인 설계의 읽는 가능성을 높이며, 단순한 소프트웨어 엔지니어링 문제로 여겨지면 안된다는 점도 강조합니다.



### AI-Driven Analytics of Team-Teaching Talk: Acoustic Patterns across Experience, Cohorts and the Learning Design (https://arxiv.org/abs/2606.09831)
Comments:
          Accepted at AIED 2026 (International Conference on Artificial Intelligence in Education), 14 pages, 4 figures

- **What's New**: 이 논문에서는 팀 티칭(team teaching) 환경에서 교사의 발화를 AI 기반 음성 처리 방법을 통해 분석하는 혁신적인 접근 방식을 제시합니다. 특히, 교사의 경험, 학생 집단, 학습 과제 디자인이 발화의 음향적 특성에 어떻게 영향을 미치는지를 조사하였으며, 이를 통해 팀 티칭의 마이크로 수준의 과정을 이해할 수 있는 신뢰할 만한 증거를 제공합니다. 이 연구는 공간 교육법(spatial pedagogy) 이론을 바탕으로 하여, 교사의 발화가 학습 환경에서 어떻게 상호작용하는지를 탐구합니다.

- **Technical Details**: 연구는 12명의 교사가 참여한 36회의 학부 및 대학원 수업을 녹음하여 분석하였습니다. 이를 통해 공간 교육법 행동을 코드화하고, 교사의 발화에서 음향적 특성을 추출하여 교사의 경험, 학생 집단, 학습 과제 디자인 간의 변화를 조사했습니다. 이 연구는 학습 환경에서 시간, 위치 및 발화가 교실의 교육적 의미에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 결과는 경험이 높은 교사, 학부 수업 및 협력 학습 과제가 높은 음량 변화를 보였다는 것을 보여주었습니다. 이는 고Experiencing teachers 및 협력적 학습 과제가 중요 정보를 강조하고 교실 상호작용 및 참여를 지원하는 데 더 자주 음량을 조절한다는 것을 시사합니다. 이러한 발견은 팀 티칭에서 발화의 음향적 특성을 고려함으로써 교육적 접근을 한층 보강할 수 있음을 보여줍니다.



### Culturally-Aware AI for Cross-Boundary Community Learning: Undergraduate Innovation at the Intersection of Computation and Design (https://arxiv.org/abs/2606.09041)
- **What's New**: 이 논문은 아시아-태평양 지역 내 인공지능 교육(AIED) 연구의 문화적 맥락을 반영한 인간 중심 접근성을 강조합니다. 커뮤니티 기반 학습(Community-Based Learning)이라는 교육 패러다임을 통해 대학생들이 문화유산 보존 및 지속 가능한 발전에 기여하는 AI 솔루션을 개발하도록 했습니다. 이는 사회적 작업의 관점을 통합하여 AI 교육에 새로운 차원을 추가하는 연구입니다.

- **Technical Details**: 본 연구는 '경계 넘는 커뮤니티 기반 학습'을 상정하며, 데이터 처리 및 시각화 기술과 커뮤니티 기반 학습 방법론을 결합합니다. 대상인 두 명의 학부생은 상하이의 Zhouzhuang Mystery of Life Museum과 협력하여 지역 생태 및 문화 유산 데이터에 대한 프로젝트를 수행했습니다. 이 과정에서 Python을 이용한 데이터 전처리와 Plotly 및 Folium을 활용한 시각화를 포함하여 AI 도구 Claude Code의 지원을 받았습니다.

- **Performance Highlights**: 학생들은 무대 간 협업을 통해 지역 문화 및 상업 활성화를 목표로 하는 Bilingual Cultural Map Interface를 설계했습니다. 이 프로젝트는 전통적인 학습에서 벗어나, 학생들이 단순 수동적 수혜자가 아닌 지식 생산자로서의 역할을 하게 만들었습니다. MIT 라이센스를 통해 공개된 이 결과물은 커뮤니티의 요구를 보다 잘 반영하며, 프로젝트 발표 후 학생들은 실제 지역 사회에 긍정적 영향을 미치는 방법에 대한 인식을 갖췄습니다.



### More Human or More AI? Visualizing Human-AI Collaboration Disclosures in Journalistic News Production (https://arxiv.org/abs/2601.11072)
Comments:
          Accepted to ACM CHI 2026 - Preprint

- **What's New**: 이번 연구에서는 저널리즘에서 AI 사용의 공개 방식이 단순한 라벨에 그치는 문제를 지적합니다. 연구팀은 총 10차례의 공동 설계 세션을 통해 69개의 공개 디자인을 발굴하고, 이 중 4개의 프로토타입을 구현하여 인간과 AI의 협업을 시각적으로 드러내는 방법을 탐구했습니다. 결과적으로 문서에 대한 설계 결과를 더 깊게 이해할 수 있는 기초 자료를 마련했습니다.

- **Technical Details**: 연구에서는 인간-AI 협업의 비율을 시각적으로 표현하기 위해 질적 및 양적 방법론을 결합한 접근 방식을 채택했습니다. 계약형 연구(N=32)를 통해 Textual, Role-based Timeline, Task-based Timeline, 그리고 Chatbot와 같은 다양한 공개 시각화 모형이 독자의 인식에 미치는 영향을 분석했습니다. 텍스트 기반 공개 방법은 협업을 전달하는 데 가장 비효율적이었으며, Chatbot은 가장 깊이 있는 정보를 제공했습니다.

- **Performance Highlights**: 이번 연구에서 모든 프로토타입은 인간-AI 협업 결과 비율을 효과적으로 전달했지만, 그 목적은 다소 달랐습니다. Role-based Timeline과 Task-based Timeline은 AI 기여를 강조하고 시각적 명확성을 제공하는 데 효과적이었습니다. 특히, Task-based Timeline은 주로 AI에 의해 작성된 기사의 인간 참여를 더 잘 전달하는 데 기여했습니다.



New uploads on arXiv(cs.RO)

### TacForeSight: Force-Guided Tactile World Model for Contact-Rich Manipulation (https://arxiv.org/abs/2606.11184)
- **What's New**: 이 논문에서는 로봇의 연락이 빈번한 조작을 위한 새로운 접근법 TacForeSight를 제안합니다. 이는 힘 조건화(force-conditioned)된 촉각 예측 프레임워크로, 고빈도 손목 힘 및 토크 신호에 따라 촉각 잠재 동역학을 예측하여 실시간으로 조작을 가능하게 합니다. TacForeSight는 두 가지 주요 구성 요소인 TacForceWM과 Predictive Tactile-Conditioned Policy를 포함하여, 서로 다른 신호를 통합하여 예측 중심의 조작 모델을 구성합니다.

- **Technical Details**: TacForeSight의 첫 번째 구성 요소인 TacForceWM은 이중 손가락 촉각 관측 결과를 바탕으로, 힘과 토크에 의해 조건화된 짧은 시간 범위의 촉각 동역학을 예측하는 촉각 세계 모델입니다. 두 번째 구성 요소인 Predictive Tactile-Conditioned Policy는 이러한 예측된 촉각 정보를 활용하여 조치 시퀀스를 예측하는 데 사용됩니다. 이는 현재에서 미래로의 촉각 상호작용을 모델링할 수 있는 교차 주의(cross-attention) 모듈을 포함하며, 다중 모달 관측을 적응적으로 통합하는 가벼운 시각-촉각 융합 모듈을 통해 이루어집니다.

- **Performance Highlights**: TacForeSight는 다섯 가지 대표적인 연락이 빈번한 조작 작업과 세 가지 프로세스 중 섭동 설정에서 실제 로봇 실험을 통해 그 성능을 검증하였습니다. 결과적으로, TacForeSight는 기존의 기법들보다 특히 동적으로 변화하는 연락 방해 요소 하에서도 우수한 성능을 발휘하며, 연락의 형성, 유지 및 회복 능력을 향상시킵니다. 이 방법론은 다양한 방해 요소에 대해 더욱 강력한 성능을 보여줍니다.



### JOIN: Anchor-Grasp-Conditioned Joining via Opposition, Inference, and Navigation for Bimanual Assistive Manipulation (https://arxiv.org/abs/2606.11151)
Comments:
          Xiang Zhi Tan and Taşkın Padır share equal advising

- **What's New**: 이 논문에서는 휠체어에 장착된 지지팔과 모바일 조작기를 활용하여 사용자가 두 팔을 함께 사용할 수 있도록 하는 새로운 접근 방식인 JOIN을 제안합니다. 기존의 단일 지지팔 시스템은 이원적 작업(즉, 양손으로 수행해야 하는 일상적인 활동)의 수행에서 제한적입니다. JOIN 시스템은 ('bimanual joining') 방식을 통해 각기 다른 이동 베이스에서 두 팔이 협력하여 작업을 수행하는 구조입니다.

- **Technical Details**: JOIN 시스템은 세 가지 단계로 구성됩니다: 계획(plan), 구동(drive), 그리고 그립(grasp) 단계입니다. 계획 단계에서는 비전-언어 모델(Vision-Language Model, VLM)을 활용하여 작업에 필요한 그립을 추론하고, 구동 단계에서는 지지팔을 보완할 위치를 샘플링하여 평가합니다. 마지막으로 그립 단계에서는 가까운 거리에서 적합한 그립 후보를 선택하여 작업 방향에 맞는 움직임을 가능하도록 합니다.

- **Performance Highlights**: JOIN을 실제 하드웨어에서 평가한 결과, 20회 시도 중 19회 성공한 반면, 기존의 최신 기법은 14회의 성공을 기록했습니다. 또한, 사용자가 추가적인 수정 작업을 덜 필요로 하여, JOIN 시스템의 효율성이 입증되었습니다. 이 연구는 이원적 조작의 점진적인 접근 방식을 통해 보다 넓은 가능성을 탐구하고 있어, 실제 응용에 대한 긍정적인 전망을 보여줍니다.



### EM-Fall: Embodied mmWave Sensing for Day-and-Night Fall Detection on Humanoid Robots (https://arxiv.org/abs/2606.11109)
- **What's New**: 논문에서는 EM-Fall이라는 새로운 낙상 탐지 프레임워크를 제안합니다. 이 시스템은 밀리미터파(mmWave) 센서를 로봇의 이동성과 결합하여, 로봇이 자기 관점을 능동적으로 조정하고 여러 방을 아우르는 관찰 가능성을 유지할 수 있도록 합니다. 이 기법은 복잡한 주거 환경에서의 간섭을 해결하기 위해 사람 중심의 인식 파이프라인을 설계하여 신뢰할 수 있는 낙상 모니터링을 제공합니다.

- **Technical Details**: EM-Fall 시스템은 밀리미터파 레이더 센서를 통해 사람의 동작을 추적하고, 로봇의 자가 운동 정보를 통합하여 공간상의 일관성을 유지합니다. 이 시스템은 비인간 물체 필터링과 낙상 사건의 시간적 모델링을 수행하는 계층적 인식 파이프라인을 통해 복잡한 주거 환경에서도 신뢰성 있는 낙상 탐지를 가능하게 합니다. 각 레이어는 공간 상태 추정, 사람 중심 인식, 행동 생성으로 나뉘어져 연결되어 있습니다.

- **Performance Highlights**: EM-Fall 시스템은 8개의 실제 실내 환경에서 4명을 대상으로 한 실험을 통해 고전적인 고정 레이더 배치 방식보다 모니터링 연속성과 환경 간섭에 대한 강인성을 크게 향상시켰습니다. 실험 결과, 비인간 동작으로 인한 오경고를 억제하면서도 실제 낙상 사건에 대한 신뢰성 있는 탐지 성능을 유지한 것으로 나타났습니다. 이는 로봇이 상관없던 영구적인 설치 대신, 능동적인 이동 플랫폼으로서 역할을 할 수 있음을 시사합니다.



### RoboNaldo: Accurate, Stable and Powerful Humanoid Soccer Shooting via Motion-Guided Curriculum Reinforcement Learning (https://arxiv.org/abs/2606.11092)
- **What's New**: RoboNaldo는 고속 충돌을 요구하는 휴머노이드 축구 슈팅에 대한 새로운 세 단계의 모션 가이드 커리큘럼 강화 학습(RL) 프레임워크이다. 이 시스템은 단일 킥 참조를 스캐폴드로 사용해 슈팅 성능 향상을 목표로 최적화를 진행한다. 기존 방법의 한계를 극복하며 안정적이고 정확한 슈팅을 달성하는 것을 목표로 한다.

- **Technical Details**: RoboNaldo의 커리큘럼은 세 단계로 구성되어 있다. 1단계에서 모션 추적은 안정적인 킥 구조를 학습하고, 2단계에서는 다양한 프리킥 설정을 통해 목표 지향적인 정확도를 달성하는 법을 학습한다. 3단계는 이동하는 공을 다루며 로코모션 명령 및 킥 트리거 인터페이스를 통해 접근 제어와 접촉 타이밍 결정을 분리한다.

- **Performance Highlights**: 시뮬레이션 결과, RoboNaldo는 프리킥에서 평균 0.899m의 오차를 보이며 공은 시속 14.79m로 발사된다. 실제 Unitree G1 장비에서 3m 거리에서 각각 0.73m 및 0.86m의 평균 목표 슈팅 오차를 기록하였고, 공의 속도는 13.10m/s에 달한다. 이러한 결과는 RoboNaldo가 고속이면서도 정확하고 안정적인 슈팅 정책을 학습했음을 보여준다.



### A Distributed Multi-UGV Exploration Framework With Loop-Aware Planning and Descriptor-Aided Localization in Resource-Limited Environments (https://arxiv.org/abs/2606.11088)
- **What's New**: 이 논문에서는 GPS가 없는 환경에서 여러 대의 무인 지상 차량(UGV)의 협력 탐색을 위한 완전 분산 탐색 프레임워크를 제안합니다. 이 프레임워크는 UGV 간 루프 클로저(loop closure)를 지원하는 기술과 수직 계층 구조의 계획을 결합하여 독립적으로 탐색할 수 있도록 합니다. 이를 통해 오류가 발생할 가능성을 줄이고, 탐색 효율성을 높이며, 통신량을 최소화합니다.

- **Technical Details**: 제안된 프레임워크는 분산 위치 파악(distributed localization)과 매핑(mapping), 루프 인식 계층 계획(loop-aware hierarchical planning)이라는 두 개의 모듈로 구성되어 있습니다. 각 UGV는 경량화된 LiDAR 전역(descriptor) 설명자를 사용하여 UGV 간의 장소 인식을 수행하고, 불확실성을 고려하여 고유한 루프 클로저를 선택합니다. 이 과정에서 통신 제약 아래에서도 전역 일관성을 유지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델의 성능은 시뮬레이션 및 실제 UGV 실험을 통해 검증되었으며, 루프 클로저 모듈은 89.9%의 AR@1과 95.5%의 AR@1%를 달성했습니다. 또한, 분산 최적화를 통해 절대 경로 오류를 줄였으며, 시스템은 양방향 통신량을 줄이고, 탐색 시간과 이동 거리를 각각 15%와 14% 절감하였습니다.



### Generation of Diverse and Functional Robot Designs using Superquadrics Parametrisation and Quality-Diversity (https://arxiv.org/abs/2606.11037)
Comments:
          Accepted at PPSN 2026

- **What's New**: 이 연구에서는 로봇 설계를 위한 새로운 표현 방식을 도입하였습니다. 이 방식은 superquadrics(SQs)를 기반으로 하여 로봇 조감도를 수학적으로 표현하는 방법입니다. 연구의 주요 목표는 이러한 표현 방식을 사용하여 생명계에서의 조형적 다양성을 극대화하는 것입니다.

- **Technical Details**: 이 연구는 superquadrics를 로봇 설계의 새로운 인코딩 방법으로 제안하며, 이 방식을 기존의 Compositional Pattern Producing Networks (CPPN)와 결합하여 사용할 것을 제안합니다. 제안된 알고리즘은 Morpho-Evolution with Homeokinesis (M​E​H​KMEHK)이라는 중첩 진화 알고리즘을 기반으로 하며, 이는 내부 루프에서 컨트롤러를 학습하는 방식을 사용합니다. 또한, MAP-Elites 알고리즘을 통해 조형적 특성의 다차원 아카이브를 구현하여 디자인 공간을 탐색합니다.

- **Performance Highlights**: 실험 결과, SQ-CPPN 표현과 MAP-Elites의 조합이 생성된 디자인의 품질과 수를 증가시킴을 보여주었습니다. 이 접근법은 두 개의 테스트 환경에서 고유한 다양한 기능을 갖춘 로봇 디자인을 생성하는 데 효과적이었습니다. 논문의 결과는 compact하고 해석할 수 있는 기하학적 표현 사용의 장점을 강조하며, 조형적 다양성 메커니즘과 결합하여 디자인의 질과 수를 향상시킬 수 있음을 보여줍니다.



### A Spiking Neural Architecture for Coordinating Arm and Locomotor Contro (https://arxiv.org/abs/2606.11034)
- **What's New**: 이 논문은 Spiking Neural Networks (SNNs)를 통해 인체 로봇의 팔 제어와 이족 보행을 통합한 첫 번째 사례를 제시합니다. 기존의 SNN 기반 모터 제어 시스템은 이족 보행과 팔 조작을 개별적으로 다루었으나, 본 연구에서는 이를 하나의 통합된 구조로 결합하여 의미 있는 작동 가능성을 탐구합니다. 제안된 구조는 Neural Engineering Framework (NEF)와 Semantic Pointer Architecture (SPA)를 통해 양측면의 제어를 조정합니다.

- **Technical Details**: 제안된 시스템은 고차원의 스파이킹 컨트롤러를 사용하여 행동 선택을 조정하며, 이는 하위 모듈에 대한 적절한 맥락 정보를 라우팅하여 작업 요구에 따라 행동을 유연하게 적응시킵니다. 이 시스템은 REACH 모델을 기반으로 한 팔 제어기와 사전 훈련된 인공 신경망(ANN)에서 스파이킹 동등체로 변환된 보행 정책을 포함하는 구조로 되어 있습니다. 개별 모듈 간의 상호작용을 통해 로봇은 팔 제어 및 보행을 효율적으로 수행합니다.

- **Performance Highlights**: 동시 시뮬레이션을 통해 Nengo와 Isaac Sim을 결합하여 목표 도달, 연속적인 숫자 그리기, 경로 따르기 보행, 그리고 기본 신경망 모델을 통해 팔 제어와 보행 간의 전환을 성공적으로 수행하는 성과를 확인했습니다. 특히, 이번 연구에서는 두 가지 동작(보행 및 팔 제어)을 통합하여 실행함으로써 인체 로봇의 동적 제어 가능성을 높이고, 저전력 신경형 하드웨어에서의 향후 배치를 위한 토대를 마련하였습니다.



### Diffusion Forcing Planner: History-Annealed Planning with Time-Dependent Guidance for Autonomous Driving (https://arxiv.org/abs/2606.11019)
Comments:
          CVPR2026

- **What's New**: Diffusion Forcing Planner (DFP)는 과거 정보를 활용하여 모션 플래닝의 안정성을 높이는 혁신적인 방법입니다. 기존 방법들이 이력을 정적 조건으로 사용하여 과거 패턴을 단순히 복제하는 경향이 있었던 반면, DFP는 이력을 유연하게 조절하여 다양한 환경의 변화에 적응할 수 있도록 설계되었습니다. 또한, DFP는 과거, 현재 및 미래 이동 경로를 분리하여 각 구간에 개별적인 노이즈 레벨을 도입하여 모델의 안정성을 높이고 있습니다.

- **Technical Details**: DFP는 전체 경로를 과거, 현재, 미래의 청크로 나누고, 각 청크마다 독립적인 확산 시간 단계를 샘플링합니다. 이러한 접근은 'noising-as-masking' 메커니즘을 통해 과거 정보를 조절하고, 원활한 경로 생성을 보장합니다. 학습 과정에서는 이력과 미래를 동시에 예측하여 인과적으로 일관된 조건부 생성을 학습하도록 유도하며, 평가 시에는 'classifier-free guidance (CFG)' 기법을 통해 조정 가능한 방식으로 미래 샘플링을 유도합니다.

- **Performance Highlights**: DFP는 대규모 현실 세계 자율 플래닝 벤치마크인 nuPlan에서 우수한 성능을 나타냅니다. 결과적으로 DFP는 경쟁력 있는 성능을 달성하면서도 복잡한 주행 상황에서 연속적이고 안정적인 모션 계획 경로를 생성하는 능력을 보여줍니다. 실험 결과, 적절한 이력 유도를 활용함으로써 모션 플래닝에서 효과적인 메커니즘을 제공함을 입증하였습니다.



### Multi-UAV Active Sensing with Information Gain-based Planning and Belief Fusion (https://arxiv.org/abs/2606.10986)
- **What's New**: 본 논문은 다수의 무인 항공기(UAV)를 이용한 확률적 이진 지형 맵핑(active sensing) 프레임워크에 대한 실제 검증을 제공합니다. 정밀 농업(precision agriculture)을 적용 사례로 사용하며, 불확실성에 기반한 의사결정을 통해 이러한 기술의 효과성을 입증합니다. 특히, 정보 이득(Information Gain)을 기반으로 한 유용한 경로 계획(IGbIPP) 방법이 전통적인 coverage 방법보다 우수한 성능을 보임을 강조합니다.

- **Technical Details**: 연구에서는 UAV가 유지하는 확률적 신념 맵(probabilistic belief map)을 통해 정밀 농업을 위한 이진 지형을 모니터링합니다. UAV의 비행 및 센싱 방식은 다양한 세부사항을 반영해 조정되며, 기존의 전통적인 경로 계획 방법인 랜덤 워크(Random Walk)와 스윕(Sweep) 방법과 비교됩니다. 실험은 통제된 환경뿐만 아니라 실제 UAV로 촬영한 농업 이미지에서 수행됩니다.

- **Performance Highlights**: IGbIPP 방법은 엔트로피(entropy)와 맵 오류(mapping error)를 효과적으로 감소시키며, 시야(field of view)를 넓히는 것이 실제 맵의 정확도를 개선한다는 결과를 보여줍니다. 간단한 평등 또는 편향된 공간 가중치가 적응형 가중치보다 더 강력할 수 있다는 것을 발견했으며, 베이즈(Bayesian), 로그 오즈(log-odds), 덴프스터-셰이퍼(Dempster-Shafer) 융합 방식이 협력적 맵핑에서 가장 좋은 성과를 나타냅니다.



### Language-Driven Cost Optimization for Autonomous Driving (https://arxiv.org/abs/2606.10974)
Comments:
          Paper accepted at IEEE Intelligent Transportation Systems Conference (ITSC) 2026

- **What's New**: 이번 연구는 자율주행차의 비용 함수 설계를 위해 언어 기반 프레임워크를 제안합니다. 이를 통해 사용자는 특정 시나리오 설명과 자연어 쿼리를 입력하여 위험을 고려한 Model Predictive Path Integral (MPPI) 제어기에 적용될 파라미터를 자동으로 생성할 수 있습니다. 시스템은 비전문가도 이해할 수 있는 비기술적 언어로 행동 변화를 설명하고 배포 전에 확인하는 단계를 포함하고 있어 사용자 의도와 실행 간의 격차를 해소할 수 있습니다.

- **Technical Details**: 모델 예측 제어(Model Predictive Control, MPC) 및 MPPI는 차량의 동역학과 주변 환경에 따라 안전하고 동적으로 적합한 궤적을 계산하는 역할을 합니다. 그러나 기존 방식은 고정된 비용 함수를 사용하여 특정 조건에서의 동작 조정이 어려웠습니다. 이번 연구에서는 자연어 쿼리와 시나리오 피처를 통해 비용 함수의 파라미터를 조정할 수 있도록 설계되어, MPPI를 통해 실시간 제어 입력을 생성합니다.

- **Performance Highlights**: 시뮬레이션 결과는 새로운 프레임워크가 직관적으로 의도된 요구 사항에 맞춘 행동 변화를 성공적으로 유도한다는 것을 보여줍니다. 다양한 주행 시나리오에서의 효과성을 평가하는 과정에서, 이 방법은 사용자 피드백을 통합하여 동작 행동을 세밀하게 조정할 수 있도록 합니다. 궁극적으로, 이 프레임워크는 자율주행차 제어 시스템과 최종 사용자 간의 격차를 줄이는 데 기여합니다.



### Resilient Navigation for Autonomous Farm Robots by Leveraging Jerk-Augmented Models with IMU-Only Disturbance Rejection (https://arxiv.org/abs/2606.10971)
- **What's New**: 본 논문은 자율 농업 로봇의 내비게이션을 위한 새로운 알고리즘을 제안합니다. 이 알고리즘은 jerk-augmented Extended Kalman Filter (EKF)와 Multiple Tuning Factor (MTF) 적응 방법을 통합해, 센서의 신뢰성을 높이는 방식으로 동작합니다. 기존 EKF 접근 방식과 달리, 제안된 방법은 측정 공분산 행렬을 실시간으로 동적으로 조정하여 갑작스러운 장애물과 센서 이상치를 효과적으로 처리할 수 있습니다.

- **Technical Details**: 제안된 내비게이션 프레임워크는 농업 로봇의 동작을 위해 자동화된 농업 플랫폼의 기초 구성 요소를 설명합니다. 실험 검증은 경량의 자율 전기 차량인 Salin247 로봇을 사용하였으며, 이 로봇은 고유의 4륜 독립 구동 시스템을 통해 고속에서도 정밀한 조향이 가능합니다. 논문에서는 Direct Cosine Matrix (DCM) 표현을 사용하여 차량의 자세를 정확히 나타내고, 외부 가속을 처리하는 프로세스 및 측정 모델을 도출합니다.

- **Performance Highlights**: 제안된 알고리즘은 Salin247 자율 로봇에서 실제 데이터를 기반으로 평가되었으며, jerk-augmentation과 MTF 적응이 EKF 모델에 비해 3D 위치 Root Mean Square Error (RMSE)를 유의미하게 줄였음을 보여주었습니다. 이러한 결과는 농업 환경의 복잡한 장애물에도 불구하고 뛰어난 추적 효과를 제공합니다. 이로써 농업 로봇의 효율적인 내비게이션이 가능해집니다.



### AllDayNav: Lifelong Navigation via Real-World Reinforcement Learning (https://arxiv.org/abs/2606.10927)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문은 AllDayNav라는 새로운 평생 자가 학습 내비게이션 프레임워크를 제안하며, 이는 비주얼 멘탈 이미지를 기반으로 한 내부 표현을 이용하여 목표를 이미지 중심으로 표현하고 일관된 환경 내 depiction을 유지하여 평생 내비게이션을 가능하게 합니다. 이 프레임워크는 시각적 요소, 의미적 속성, 시간적 맥락을 포함한 지속 가능한 다중 모드 메모리 아키텍처를 발전시켜 에이전트가 일관된 정보를 보유하면서 상호작용하는 동안 지속적으로 메모리를 다듬을 수 있게 합니다. 또한, AllDayNav는 결정적인 정책 학습을 위해 기억 기반 강화 학습 프레임워크를 통합하여 정책이 상호작용을 통해 자율적으로 개선될 수 있도록 하고 있습니다.

- **Technical Details**: AllDayNav는 시각적 목표를 기반으로 목표를 설정하고, 시각적 임베딩과 의미적 설명 그밖의 카테고리 정보를 포함하여 에피소드에 걸쳐 지속적인 다중 모드 메모리를 구축합니다. 메모리와 정책 학습을 결합하여 에이전트는 강화 학습을 통해 점진적으로 내부 표현을 다듬고, 이를 통해 향후 상태를 예측하고 동작을 계획하는 데 도움을 줍니다. 실험은 이 시스템이 다양한 동적 환경에서 100%에 가까운 성공률을 달성하며, 기존의 지도 기반, VLM, RL 기반 기법을 넘어서는 성능을 보여줍니다.

- **Performance Highlights**: 본 연구에서 AllDayNav는 합성 및 실제 환경에서의 다양한 평생 내비게이션 작업에 대해 평가되었으며, VLM 기반 내비게이션 시스템, 지도 중심 메모리 접근법, 강화 학습 기준선에 비해 일관성 있고 상당한 성능 향상을 보여줍니다. 특히, 보이지 않는 환경에서도 100%에 가까운 성공률을 유지하며, 불확실하고 동적으로 변화하는 환경에서도 신뢰할 수 있는 성능을 발휘합니다. 이러한 결과는 AllDayNav가 강력한 선택이라 할 수 있음을 입증 우려하고 있습니다.



### Task Robustness via Re-Labelling Vision-Action Robot Data (https://arxiv.org/abs/2606.10918)
Comments:
          Project website: this https URL

- **What's New**: 최근 로봇 학습 모델의 확장이 긍정적인 정책을 생성했지만, 이러한 정책은 지침을 따라가는 데 어려움을 겪고 있습니다. 이 논문에서는 TREAD(Task Robustness via Re-Labelling Vision-Action Robot Data)라는 새로운 프레임워크를 도입하여 대규모 Vision-Language Models (VLMs) 를 활용해 기존 로봇 데이터셋의 언어-행동 다양성을 증가시키는 방법을 제시합니다.

- **Technical Details**: TREAD의 접근 방식은 세 가지 단계로 구성됩니다: 원래의 지침 레이블에서 의미적 하위 작업을 생성하고, 이 하위 작업에 따라 시연 비디오를 세분화하며, 물체 속성을 포함하여 다양한 지침을 생성합니다. 이 프레임워크는 반복적 쿼리를 통해 VLM을 활용하여 데이터셋을 확장하고 다양화하는 것을 목표로 합니다.

- **Performance Highlights**: LIBERO 데이터셋을 활용한 평가 결과, TREAD로 증강된 데이터셋에서 훈련된 정책은 새로운 작업 및 목표에 대해 향상된 성능을 보여줍니다. 정책의 일반화 성능이 향상되었으며, 이는 경로 분해를 통한 계획 일반화와 언어 조건의 정책 일반화를 통해 달성되었습니다.



### AgniNav: Configuration-Driven Cross-Embodiment Local Planning for Robot Navigation (https://arxiv.org/abs/2606.10903)
- **What's New**: 이 논문에서는 AgniNav라는 경량 로봇을 위한 새로운 로컬 내비게이션 프레임워크를 제안합니다. 기존의 비전 기반 정책들이 특정 로봇 몸체와 카메라 높이, 그리고 발자국에 종속되어 있는 문제를 해결하기 위해, AgniNav는 충돌 경계(collision-envelope) 수준에서 이러한 제약을 표준화합니다. 이 시스템은 다양한 로봇 플랫폼 간에 훈련 없이 전이 가능하도록 설계되었습니다.

- **Technical Details**: AgniNav는 각 로봇을 측정 가능한 네 가지 매개변수로 정의된 안전 경계로 표현하여 운영합니다: 충돌 관련 높이, 전면 길이, 후면 길이, 그리고 반 너비입니다. 로봇의 이미지에서 1D 충돌 관련 의사 레이저 스캔(pseudo-laserscan)을 예측하기 위해 높이 조건화된 이미지-투-스캔(image2scan, I2S) 네트워크를 사용하며, 나머지 발자국 매개변수는 충돌 체크를 위한 차원 인식 지역 계획(local planner)에서 활용됩니다. 이 방식은 특정 로봇 데이터를 수집하지 않고도 다양한 안전 경계를 수퍼바이즈하는 것이 가능합니다.

- **Performance Highlights**: 실제 로봇 실험에서 Turtlebot2, Unitree Go2, Accelerated Evolution K1와 같은 다양한 플랫폼에서 각각 39/40, 18/20, 18/20의 성공률을 달성했습니다. 이 시스템은 Jetson Orin에서 30 Hz로 수행될 수 있어 효율적인 실시간 성능을 입증하였습니다. AgniNav는 기존의 모노큘러 깊이 및 매핑 기반 방법과 비교할 때, 상당한 계산 부담을 줄이며, 자원 제한적인 엣지 컴퓨터에서도 효과적으로 작동합니다.



### MV-Actor: Aligning Multi-View Semantics and Spatial Awareness for Bimanual Manipulation (https://arxiv.org/abs/2606.10899)
Comments:
          14 pages,9 figures

- **What's New**: MV-Actor는 이중팔 조작(bimanual manipulation)을 위한 다중 시점 인식(multi-view perception) 프레임워크를 제안합니다. 기존의 접근 방식과 차별화하여 서로 다른 뷰 간의 의미 인식을 효과적으로 공유하고, 신뢰할 수 있는 공간 인식을 제공합니다. 본 연구는 분산된 카메라 정보로부터 정밀한 세멘틱-스페이셜 표현을 통합하며, 소비자 등급 깊이 카메라의 노이즈를 효과적으로 처리하는 방법을 포함하고 있습니다.

- **Technical Details**: MV-Actor는 'Multi-view Semantic Interaction'을 통해 물리적으로 일치하는 지역 간의 세멘틱 인식을 공유하며, 'Semantic-Spatial Token Interaction'을 통해 시각적 특성에 신뢰할 수 있는 공간 인식을 추가합니다. 각 카메라 뷰에서 수집된 RGB-D 데이터를 처리하여 3D 공간에서 통합된 표현을 구축하며, 이 과정에서 깊이 정보를 복구하는 모듈을 포함하여 신뢰성 있는 지오메트릭 지원을 제공합니다.

- **Performance Highlights**: 시뮬레이션 실험에서는 PerAct2 이중팔 벤치마크에서 87.8%의 최첨단 평균 성공률을 달성하였고, 실제 세계 평가에서는 빈번한 시점 변화와 불안정한 소비자 등급 깊이 조건에서도 RGB 및 RGB-D 기준을 초과하는 성능을 보였습니다. 이러한 성과는 다중 카메라 관찰에서 세멘틱 인식과 신뢰할 수 있는 공간 인식을 공유하는 것의 이점을 잘 보여줍니다.



### Embodiment-conditioned Generalist Control for Multirotor Aerial Robots (https://arxiv.org/abs/2606.10857)
- **What's New**: 이번 연구에서는 다양한 멀티로터 구성(예: 헥사로터 또는 쿼드로터)을 단일 네트워크 가중치 세트로 제어할 수 있는 일반화된 포지션 제어 정책을 제시합니다. 이 정책은 질량 및 관성 정규화된 제어 할당 행렬에 조건화되며, 이를 통해 질량 정규화된 모터 스로우트가 몸체 프레임에서 선형 및 각 가속도를 생성하는 방식을 포착합니다. 일반화된 정책은 로봇의 작동 구조와 동역학에 맞춰 제어 행동을 조정하는데, 이는 다양한 형태의 멀티로터에 대한 강력한 일반화 제어를 가능하게 합니다.

- **Technical Details**: 제안된 정책은 단일 RTX 3090 GPU에서 사용자 정의된 NVIDIA Warp 기반 동역학 시뮬레이터를 사용하여 훈련되며, 훈련 시간은 5분에 불과합니다. 연구에서는 비평면적이며 비대칭적인 시스템을 포함하여 무작위 멀티로터 구성 샘플을 광범위하게 활용하여 정책을 최적화합니다. 이로 인해 다양한 형태의 멀티로터에 대해 제어를 할 수 있는 강력한 일반화 성능이 확인되었습니다.

- **Performance Highlights**: 로그 기록을 통해 제어 성능이 다양한 헥사로터 시스템에서 0샷(real-world zero-shot) 전이 가능함을 입증합니다. 연구 결과는 플래너 로봇, 부분적으로 대칭인 비평면 시스템, 랜덤 비대칭 비평면 구성을 포함한 세 가지 다양한 헥사로터 시스템에서 실험하였습니다. 이를 통해 제안된 방법의 효과와 가능성을 보여줍니다.



### An Exposure-Time-Aligned Primary-Path Architecture for Autonomous-Driving ECUs (https://arxiv.org/abs/2606.10856)
- **What's New**: 이번 논문은 E2E(End-to-End) 자율주행의 주요 설계 원칙을 제안합니다. 현재의 모듈형 멀티-NN 파이프라인에서 E2E 아키텍처로의 단계적 전환을 지원하기 위해, Primary-Path, Exposure-Time-Aligned, Co-Path Coexistence라는 세 가지 디자인 원칙을 소개합니다. 이러한 원칙들은 보편적인 레이트 퓨전 방식의 비효율성을 극복하고 더욱 결정론적인 지연(latency)을 복원하며 E2E 마이그레이션 경로를 유지하는 데 기여합니다.

- **Technical Details**: 논문에서 제안하는 아키텍처는 이중 SoC(Dual-SoC) 생산 AD-ECU(자율주행 전자 제어 장치)를 기반으로 하며, 두 개의 일반 목적 SoC와 두 개의 AI 가속기 SoC로 구성되어 있습니다. 이 구조는 CPU가 주도하는 소프트웨어 구성요소(SWC) 체인과 CPU/NPU의 속도 조절을 최적화하여 서로 다른 SoC 간의 비결정론적인 경로를 제거합니다. Primary-Path 원칙을 적용하여 주 간섭 체인을 설정하고, Exposure-Time-Aligned 기법을 통해 노출 시간을 기준으로 신호를 조정합니다.

- **Performance Highlights**: 테스트 환경에서는 카메라 셔터와 계획(output) 간의 평균 지연 시간이 296 ms로, 350 ms 설계 예산 내에서 성공적으로 수렴했습니다. 이 프로세스에서 모듈형 파이프라인은 생산 시작 시 주요 역할을 하고, E2E 경로는 실 차량에서 그림자(shadow) 형태로 작동하며, E2E 범위는 추가 평가 증거에 따라 단계적으로 확장됩니다. 이는 자율주행 시스템의 신뢰성과 효율성을 높이는 데 중요한 결과입니다.



### Gradient based Bilevel for Inverse Optimal Control, a Riemannian approach (https://arxiv.org/abs/2606.10841)
Comments:
          6 Pages, 4 Figures. To be published in a control journal

- **What's New**: 이 논문은 기본적으로 Inverse Optimal Control (IOC) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 bilevel 최적화 접근법이 계산적으로 비효율적이라는 문제를 해결하기 위해 Riemannian manifolds를 이용한 최적화 방법을 시도하고 있습니다. 또한, 제안된 Riemannian Inverse Optimal Control (RIOC) 방법은 최적해의 매니폴드 상에서 관측된 궤적을 투영하는 방식으로 성능을 높이고 있습니다.

- **Technical Details**: RIOC 방법은 관측된 궤적을 최적의 해를 가진 매니폴드에 투영하는 과정에서 실현 가능성을 보장합니다. 이는 기존의 gradient 기반 최적화가 갖는 수치적 불안정성 문제를 해결하는 데 기여합니다. 이 방법의 기반 성격은 KKT 조건을 만족하는 궤적 집합이 자연스럽게 매니폴드 형태를 형성한다는 점에서 착안되었습니다.

- **Performance Highlights**: 실제 인간 팔 궤적에 대한 실험에서 RIOC 방법은 기존의 bilevel IOC와 비교할 때 더 빠른 수렴 속도와 유사하거나 더 나은 재구성 정확도를 보였습니다. 이 연구 결과는 RIOC 방법이 로봇 공학과 인간 동작 분석을 위한 IOC 문제의 확장성과 신뢰성을 개선할 수 있는 잠재력을 강조합니다.



### GUIDE: Goal-Initialized Directional Understanding for End-to-End Visual Navigation (https://arxiv.org/abs/2606.10832)
Comments:
this https URL

- **What's New**: 이번 연구에서는 로봇이 내재된 공간 기억을 기반으로 동작할 수 있도록 목표를 초기화하는 내비게이션 설정을 제안합니다. 로봇이 에피소드 시작 시 한 번만 목표를 부여받고 이후에는 외부 모듈로부터의 목표 업데이트 없이 작업하도록 요구됩니다. 이를 해결하기 위해 GUIDE라는 전반적인 강화 학습 프레임워크를 개발하였습니다.

- **Technical Details**: 제안된 방법은 부분 관찰 가능 마르코프 결정 프로세스(Partially Observable Markov Decision Process, POMDP)로 목표 조건 시각 내비게이션 작업을 형성합니다. 이 구조에서 로봇은 기본적으로 자신의 프로프리오셉션(proprioception)과 깊이 비전(depth vision)을 사용하여 내비게이션 결정을 내려야 합니다. GUIDE는 방향성 및 내재적 모션 인식을 기르기 위해 다주파수 프로프리오셉티브(history) 기록을 활용하며, 공간 앵커 예측기를 포함하고 있습니다.

- **Performance Highlights**: 실험을 통해 GUIDE가 복잡한 환경 및 구조화된 미로를 안전하게 탐색할 수 있는 능력을 배우는 것을 확인하였습니다. 제안하는 프레임워크는 시뮬레이션과 현실 세계 시나리오 모두에서 검증되었습니다. GUIDE는 주어진 목표 없이도 로봇이 목표에 도달하고 복잡한 장애물에서 탈출할 수 있게 합니다.



### IMPACT: Learning Internal-Model Predictive Control for Forceful Robotic Manipulation (https://arxiv.org/abs/2606.10818)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 IMPACT라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 힘을 사용하는 로봇 조작 작업을 작업 계획(task-planning)과 내부 모델 기반 예측 제어(internal-model-based predictive control)로 분리하여 보다 효율적으로 수행할 수 있도록 합니다. 이를 통해 이전 방법들보다 더 높은 성공률과 물체의 무게에 대한 일반화 성능을 향상시켰습니다.

- **Technical Details**: IMPACT 프레임워크는 조작기가 손목에 장착된 힘-토크 센서를 통해 상호작용 힘을 직접 측정하는 대신 관절 토크 독서를 기반으로 힘을 추정합니다. 이 내부 모델은 상태와 행동의 이력을 바탕으로 예측된 상호작용 힘을 보상하도록 훈련됩니다. 특히, 이 모델을 통해 시스템은 다양한 외부 방해 요소에 적절히 대응할 수 있도록 합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험을 통해, IMPACT 프레임워크는 다양한 힘 조작 작업에서 높은 성공률을 기록하였으며, 반응성과 에너지 효율성 또한 개선되었습니다. 특히, 하이브리드 위치-힘 제어 작업인 서예 작업에서도 뛰어난 성능을 보였습니다. 이러한 성능은 다양한 무게의 물체를 다룰 때에도 안정적인 조작을 가능하게 합니다.



### Bridging Semantics and Physical Execution: A Neuro-Symbolic Framework for Multi-Pair Robotic Assembly (https://arxiv.org/abs/2606.10808)
Comments:
          Corresponding author: Aiguo Song (this http URL@seu.this http URL)

- **What's New**: 이 논문에서는 다중 쌍 로봇 조립을 위한 새로운 신경-상징적(neuro-symbolic) 프레임워크를 제안합니다. 이 프레임워크는 계층적으로 최적의 하위 그래프를 생성하고, 일반성과 특수 사례를 분리하며, 쌍 간 간섭을 해결합니다. 눈에 보이는 RGB-D 조립 장면을 기반으로, 장면을 측정하여 편차 계산을 수행하고, 최적의 하위 그래프를 생성하여 논리적 환각(hallucinations)을 줄이는 방안을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 LLM을 통해 각 쌍에 대한 최적의 하위 그래프를 생성하며, 가벼운 분류기(discriminator)로 특수 사례에 대한 지원 동작을 이유지어 추가합니다. 기존 상태와 현재 장면 간의 차이를 기반으로 하여 편차를 계산하고, 이를 통해 저비용으로 확장 가능하게 설계되었습니다. 다이나믹 행동 트리(dynamic behavior trees)는 원자 기술(atomic skills)을 포함하여 강도 인식(force-aware) 실행 루프를 닫습니다.

- **Performance Highlights**: 실제 100개의 장면에 대한 오프라인 평가에서 이 프레임워크는 97.00%의 글로벌 실행 가능성을 달성하여 기존의 방법보다 우수함을 입증하였습니다. UR3 팔 로봇에서 실제 배치 시 강력한 간섭 하에서도 90%의 성공률을 기록하며, 복잡한 자율 조립을 위한 통합되고 검증 가능한 솔루션을 보여주었습니다.



### ros2probe: Non-intrusive, Kernel-selective Observability for Robot Operating System 2 Middlewar (https://arxiv.org/abs/2606.10746)
Comments:
          13 pages, 8 figures, 7 tables

- **What's New**: 이 논문은 ROS 2의 비침투적 관찰 프레임워크인 ros2probe를 소개합니다. 이 도구는 DDS 도메인 외부에서 ROS 2 통신을 전체적으로 관찰하여 기존의 probe effect를 제거합니다. 이로 인해 ROS 2 상의 메시지 손실 및 기타 메트릭을 보다 정확하게 파악할 수 있습니다.

- **Technical Details**: ros2probe는 DDS의 평문 전송에서 발생하는 discovery 패킷을 수집하여 ROS 2 토픽 그래프와 각 토픽의 메트릭을 실시간으로 복원합니다. 기존 도구들이 DDS 도메인에 참여하여 발생하는 네 가지 후보 효과를 피하면서도, 사용자가 요청한 주제만을 관찰하는 커널 필터를 통합합니다. 이로 인해 시스템의 대역폭 비용 없이도 오버헤드를 최소화합니다.

- **Performance Highlights**: 세 가지 하드웨어 플랫폼(랩탑, Jetson, Raspberry Pi)에서 ros2probe는 기존 도구에 비해 관찰 정확도를 크게 향상시키며, CPU 및 메모리 사용량을 각각 7배와 28배 줄입니다. 또한 ros2probe는 무부하 상태에서 손실을 정확히 일치시키고, 실제 구독자의 메시지를 손실시키지 않으면서 모든 시나리오에서 관찰을 성공적으로 수행합니다.



### Hand-centric Human-to-Robot Trajectory Transfer from Video Demonstrations via Open-World Contact Localization (https://arxiv.org/abs/2606.10743)
- **What's New**: 이번 논문에서는 사람의 비디오 시연에서 로봇 동작으로의 전이를 위한 새로운 프레임워크인 HOWTransfer를 제안합니다. HOWTransfer는 수동적 물체 상호작용(Hand-Object Interaction)에서도 다양한 로봇 궤적을 추출할 수 있도록 설계된 손 중심의 시스템입니다. 이를 통해 비디오에서 수집된 다수의 사람의 demonstrations로부터 실용적인 로봇 동작을 생성하는 과정을 개선합니다.

- **Technical Details**: HOWTransfer는 3단계로 구성됩니다. 첫 번째 단계에서는 비디오에서 일관된 3D 손 동작을 재구성하고, 두 번째 단계에서는 접촉 구간을 분리하여 처리하며, 마지막 단계에서는 인식된 접촉 구간을 사용하여 로봇이 수행할 수 있는 궤적을 생성합니다. 이 과정에서 사용되는 기술로는 hand trajectory reconstruction, open-world contact localizer, cross-embodiment trajectory retargeting 등이 포함됩니다.

- **Performance Highlights**: 실험 결과, HOWTransfer는 86%의 성공률로 로봇 동작 재처리가 가능함을 보여주었으며, 이는 기존의 원거리 조작 방식보다 우위를 점합니다. 전반적으로 이 연구는 로봇이 사람의 비디오 시연에서 효과적으로 조작 기술을 습득하고 반복 가능한 동작을 수행할 수 있는 가능성을 제시합니다.



### Pushing the Performance Limits in Autonomous Racing: Continuous Stability-Aware Adaptive Velocity Planning in Formula Student Driverless (https://arxiv.org/abs/2606.10733)
Comments:
          Accepted as a conference paper in IEEE Intelligent Vehicles Symposium (IV) 2026, Detroit, MI, United States

- **What's New**: 이 논문은 자율 레이싱에서의 Adaptive Velocity Planning (VP) 접근방식을 소개합니다. 이 방법은 고속 주행, 제어 불확실성 및 환경 영향들을 고려하여 동적으로 목표 속도를 조정합니다. 기존의 타이어-도로 마찰 계수를 추정하는 대신, 차량의 안정성 지표로부터 연속적인 스케일링 팩터를 추론하며, 이는 효과적인 타이어-도로 상호작용을 반영합니다. 이를 통해 생성된 연속 마찰 맵은 최적의 목표 속도를 계산하는 강력하고 적응적인 기초를 제공합니다.

- **Technical Details**: Adaptive VP는 차량의 현재 종방향 속도와 경로 진행 상황을 입력으로 받아, 계획된 곡률을 기반으로 각 지점에서의 최대 가능 속도를 출력합니다. 예측 지평선에 따라 이 스케일링 팩터는 주행 중 각 위치에서 동적으로 추정됩니다. 기존 연구들과 달리, 이 방법은 전체 트랙이나 특정 세그먼트가 아닌 모든 위치에 대해 별도로 스케일링 팩터를 추정합니다. 이로 인해 차량의 성능 한계를 효과적으로 구현할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 실제 Formula Student 레이싱카에서 평가되었으며, 10랩 동안 35%의 랩 타임 개선과 비적응형 접근 방식에 비해 평균 8%의 성능 향상을 기록했습니다. 이러한 결과는 Adaptive VP가 동적인 주행 환경에서 높은 성능을 유지하는 데 효과적임을 보여줍니다. 또한 이 방법은 지속적으로 목표 속도를 개선하여 성능의 안정성을 극대화하는 데 기여합니다.



### Vehicle Prediction Model for Enhanced MPC Path Tracking in Formula Student Driverless (https://arxiv.org/abs/2606.10732)
Comments:
          Accepted as a conference paper in IEEE Intelligent Vehicles Symposium (IV) 2026, Detroit, MI, United States

- **What's New**: 이번 논문에서는 Formula Student Driverless와 같은 자율 주행 경주차를 위해 새로운 실시간 예측 모델을 제안하였다. 이 모델은 과거 주행 데이터와 현재 주행 상황의 정보를 결합하여 변화하는 조건에 적응할 수 있다. 제안된 예측 모델은 Kinematic Bicycle Model, Bayesian Linear Regression (BLR), Sparse Gaussian Process Regression (SGPR)로 구성된 세 가지 하위 모델로 나뉜다. 이 접근법을 통해 고속 주행에서도 높은 예측 정확도와 불확실성 평가가 가능해졌다.

- **Technical Details**: 모델 예측 제어(Model Predictive Control, MPC)는 복잡한 경로 추적 문제를 해결하기 위해 사용된다. 이 연구는 세 가지 하위 모델을 결합하여 차량의 동작을 예측하는 새로운 예측 모델을 제시하였다. Nominal 모델이 차량의 물리적 특성을 나타내고, Offline BLR 모델이 이전 주행 데이터를 바탕으로 잔여 오차를 보정하며, Online SGPR 모델이 현재 환경 조건에 따라 실시간으로 예측을 조정한다. 이러한 구조적 접근은 데이터의 효율적인 통합을 가능케 하여 계산 비용을 크게 증가시키지 않는다.

- **Performance Highlights**: 제안된 모델은 기존 방법보다 최대 57% 더 높은 예측 정확도를 달성하였다. 실제 Formula Student 경주차에 MPC 기반 경로 추적 컨트롤러를 사용하여 모델의 실용성을 성공적으로 증명하였다. 이 모델은 자율 주행 차량의 안전성을 보장하고, 예측 불확실성에 대한 정량적 평가를 제공함으로써 강화된 로버스트성을 보여준다. 이러한 개선은 고속 경주시에도 경쟁력을 갖출 수 있도록 하였다.



### Self-Supervised Relevance Modelling in Autonomous Driving via Counterfactual Analysis (https://arxiv.org/abs/2606.10688)
- **What's New**: 이 연구에서는 자율주행 자동차의 주변 환경에서 객체를 탐지하고 추적하는 과정의 효율성을 높이기 위해 새로운 접근법을 제안합니다. 저자들은 counterfactual analysis(반사실 분석)를 기반으로 한 self-supervised(자기지도 학습) 방법을 사용해 relevance model(관련성 모델)을 개발했습니다. 이 모델은 자율주행 차량에 대한 객체의 중요성을 정량화하는 AI 도구입니다.

- **Technical Details**: 제안된 관련성 모델은 선택된 도시 시나리오에서 생성된 합성 인과 데이터를 기반으로 학습되었습니다. 이 모델은 밀리초 수준의 지연 시간으로 객체의 관련성을 정확하게 추정할 수 있으며, 고밀도 시나리오에서도 실시간으로 관련성 추정이 가능합니다. 또한, 이 모델은 자율주행 차량의 주행 정책에 대한 통찰을 제공하는 관련성 heatmap(열지도)을 구성하는 데 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 데이터 처리 지연을 줄이고, 인식 노의 하류 전파를 제한함으로써 효율적인 컴퓨팅 자원 사용을 가능하게 합니다. 관련성 모델과 causal dataset(인과 데이터셋)은 공개적으로 릴리스되어 연구자들이 자율주행 시스템의 성능을 향상시키는 데 기여할 수 있도록 합니다.



### UniDexTok: A Unified Dexterous Hand Tokenizer from Real Data (https://arxiv.org/abs/2606.10683)
- **What's New**: 이 논문에서는 다양한 구현에서 손의 정밀한 조작을 위한 통합 손 모델(UDHM)을 제안합니다. UDHM은 인간의 손과 로봇 손의 상태를 공유하는 22-DoF(도)의 의미적 인터페이스로 매핑합니다. 이를 통해 서로 다른 손 구성에서의 데이터 사용이 용이해지며, 모델의 성능이 대폭 향상됩니다.

- **Technical Details**: 이 연구에서 제안하는 UniDexTok는 흔히 사용된 손 특화 토크나이저와의 차별점으로 모든 구현에서 단일 인코더, 코드북 및 디코더를 공유합니다. 이 설계는 다양한 손 구조 간의 전이 가능성을 증진시키며, 새로운 손이 도입될 때도 별도의 학습 없이기존의 토큰 공간으로 투영할 수 있게 합니다. UDHM과 UniDexTok을 기반으로 한 새로운 학습 파이프라인은 여러 데이터셋에 걸쳐 실제 손 데이터를 표준화하여 학습합니다.

- **Performance Highlights**: UniDexTok은 UniHM 대비 MPJAE(Mean Per Joint Average Error)를 15.63도에서 0.16도로, MPJPE(Mean Per Joint Position Error)를 18.51mm에서 0.18mm로 감소시켰습니다. 이는 각각 98.98%와 99.03%의 오류 감소를 의미하며, 단위가 센티미터에서 서브 밀리미터 정확도로 향상된 것을 보여줍니다. 추가 실험 결과는 다른 구현에서의 데이터가 목표 구현의 재구성 정확도를 개선하는 데 기여함을 보여주었습니다.



### Planar-Sector LOS Guidance for Interception of Agile Targets with Lifting-Wing Quadcopters (https://arxiv.org/abs/2606.10639)
Comments:
          Accepted to the IEEE International Conference on Robotics and Automation (ICRA 2026). Recipient of the ICRA 2026 Best Paper Award in Field and Service Robotics

- **What's New**: 이번 논문은 자율 비행체가 민첩한 공중 목표물을 효과적으로 추적 및 격추할 수 있는 새로운 방식인 Planar-Sector Line-of-Sight (PS-LOS) 가이드라인을 제안하고 있습니다. 이 방법은 기존의 conic line-of-sight 제약을 벗어나 목표물의 한쪽 방향으로는 더 넓은 여유를 두어, 맹렬한 추적을 가능하게 하면서도 적절한 시각적 추적을 유지합니다. 이를 통해 기존의 방식보다 약 50% 더 많은 추진력을 제공하며, 실험적으로 최대 138m의 거리에서 목표를 자율적으로 격추할 수 있음을 입증했습니다.

- **Technical Details**: 이 연구에서는 다양한 좌표계(코디네이트 프레임)를 사용하여 모델과 문제 정의를 체계적으로 정립하고 있습니다. 여기에는 지구 고정 프레임(Earth-fixed frame), 비행체 프레임(Body frame), 리프팅 윙 프레임(Lifting-wing frame) 등 여러 가지 프레임이 포함됩니다. 특히 PS-LOS 제약을 정의하여 카메라의 대칭 평면에 맞춰 목표물의 시각적 추적과 비행체의 기동성을 효과적으로 조율하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, PS-LOS를 통해 138m까지의 거리에서 불규칙한 비행 목표물을 성공적으로 격추할 수 있었습니다. 이 접근 방식은 고주파, 대진폭 운동을 보이는 목표물에 대해서도 지속적인 시각 추적을 유지하면서 높은 성공률을 기록했습니다. 따라서 PS-LOS는 자율 비행체에 대한 시각적 기반의 격추 작업에서 향상된 성능을 제공하는 것으로 평가됩니다.



### Dexterous Point Policy: Learning Point-based Dexterous Hand Policies from Human Demonstrations (https://arxiv.org/abs/2606.10614)
- **What's New**: 로봇 기반 모델(robust foundation models)이 인간의 시연 비디오로 사전 훈련되었지만, 실제 로봇에 배포될 때 여전히 큰 일체감 차이(embodiment gap)가 존재합니다. 이를 해결하기 위해, 우리는 Dexterous Point Policy라는 새로운 프레임워크를 도입합니다. 이 프레임워크는 로봇 시연 없이도 인간 비디오에서 직접 정교한 조작(le dexterous manipulation) 정책을 학습합니다.

- **Technical Details**: 우리는 작업 관련 객체와 인간 손의 3D 키포인트를 원시 비디오(raw videos)에서 추출하여, 이 키포인트들을 통해 오토회귀(transformer) 모델을 훈련합니다. 통합된 3D 키포인트 표현(unified 3D keypoint representation)을 사용하여 인간과 로봇의 시청각 정보를 연결합니다. 특히, 손목(wrist)과 손끝(fingertips) 차원에서 인간과 로봇의 행동이 밀접하게 연관되어 있음을 확인하였습니다.

- **Performance Highlights**: Dexterous Point Policy는 실제 로봇 작업에서 75.0%의 성공률을 달성했습니다. 이는 첨단 VLA 기준선(state-of-the-art VLA baseline)의 1.0%와 비교할 때 현저히 높은 성과입니다. 또한, 우리의 방법은 다중 객체 환경(multi-object environments)과 새로운 객체 범주(novel object categories)를 포함한 보지 못한 시나리오에서 강력한 일반화 성능을 보입니다.



### LieIPM: Lie Group Interior Point Method for Direct Trajectory Optimization of Rigid Bodies (https://arxiv.org/abs/2606.10579)
- **What's New**: 이 논문은 제약이 있는 궤적 최적화를 행렬 리 군(matrix Lie group) 바로 위에서 수행하는 방법을 제안합니다. 기존의 유클리드 공간을 기반으로 한 접근 방식과 달리, 이 방법은 강체 운동의 기하학적 구조를 유지하면서 효율적인 뉴턴형 업데이트를 지원합니다. 새로운 Lie Group Interior Point Method (LieIPM)는 다양한 제약을 고려하여 최적의 궤적을 찾는데 사용됩니다.

- **Technical Details**: 논문에서는 강체 운동의 궤적 최적화를 다루기 위해 Lie 군의 구조를 활용한 두 번째 차수 구동 모델을 기반으로 합니다. LieIPM은 비유클리드 매니폴드 위에서 제약 최적화를 해결하기 위한 기법으로 설계되었습니다. 특히, Lie 군 모형을 활용하여 내재적 도함수를 유도하고, 이를 통해 궤적 최적화를 보다 신뢰성 높고 빠르게 수행합니다.

- **Performance Highlights**: 수치 실험 결과, LieIPM은 기존의 일반 목적 솔버 및 최적 제어 방법에 비해 뛰어난 견고성과 더 빠른 수렴 속도를 보여줍니다. 이 연구는 로봇 제어 및 궤적 계획 문제를 효과적으로 해결하는 데 기여할 것으로 기대됩니다. 오픈소스 C++ 구현은 GitHub에서 제공되어 사용자들이 직접 활용하고 개선할 수 있습니다.



### AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness (https://arxiv.org/abs/2606.10577)
- **What's New**: 이번 논문에서는 AgenticNav라는 새로운 경량 네비게이션 도구를 소개합니다. Zero-shot vision-and-language navigation in continuous environments (VLN-CE)에서 기존의 방법론들이 제약된 탐색 공간과 관련 없는 메모리 문제로 어려움을 겪고 있다는 점에 주목했습니다. AgenticNav는 VLM(vision-language model)과 환경 간의 인터페이스를 새롭게 설계하여 행동, 깊이, 메모리를 호출 가능한 도구로 제공합니다.

- **Technical Details**: AgenticNav는 세 가지 주요 도구를 갖추고 있습니다: 첫째, waypoint-free action tool은 VLM이 RGB 이미지에서 직접 픽셀을 선택할 수 있게 합니다. 둘째, on-demand pixel-depth tool은 선택된 이미지 위치에서의 측정 깊이를 요청할 수 있게 하여 비효율성을 줄입니다. 셋째, selective memory-recall tool은 간결한 맵 이미지를 통해 과거의 시각적 정보를 선택적으로 재호출할 수 있도록 도와줍니다.

- **Performance Highlights**: R2R-CE 벤치마크에서 AgenticNav는 55%의 SR(success rate)와 48.41%의 SPL(success per length)을 기록하며, 기존 방법보다 높은 성능을 보였습니다. 특히, 기존의 waypoint 예측기 대신 액션 도구를 도입함으로써 성능 향상을 이루었고, 깊이 도구 및 에이전틱 메모리 메커니즘이 네비게이션 성능에 긍정적인 기여를 했음을 입증했습니다.



### VeriSpace: Spatially Grounded Action Verification for Vision-Language-Action Models (https://arxiv.org/abs/2606.10568)
Comments:
          Submit to ACM MM

- **What's New**: 최근 로봇 조작을 위한 Vision-language-action (VLA) 모델의 신뢰성 문제를 해결하기 위해 VeriSpace라는 3D 인식 행동 검증기가 제안되었습니다. 이 모델은 후보 행동을 제안하고 실행 전에 평가 가능한 명확한 테스트 시간을 제공하여 소중한 로봇 조작 결정의 신뢰성을 향상시킵니다. 기존의 VLA 정책과 완전히 호환되며, 실시간 결정 및 조정을 가능하게 합니다.

- **Technical Details**: VeriSpace는 두 가지 주요 구성 요소를 통해 후보 행동을 평가합니다: Dual-Path 3D-Injected Scene Encoding은 시각적 의미와 3D 기하학을 동시에 보존하는 장면 표현을 구축하며, Spatially-Grounded Action Reasoning은 작업 관련 공간 관계를 이유로 후보 행동을 평가합니다. 이러한 구성 요소들은 3D 정보를 활용하여 미세한 기하학적 차이를 구분하고 목표 진행을 평가하는 데 큰 도움을 줍니다.

- **Performance Highlights**: VeriSpace는 공개 벤치마크와 실제 로봇 조작 작업에서 일관되게 VLA 결정의 신뢰성을 향상시키며, 기존 정책 모델 및 이전 검증 기반 방법들에 비해 상당한 성능 향상을 보여줍니다. SIMPLER 벤치마크에서 VeriSpace는 OpenVLA에 비해 18.0% 포인트 개선되었으며, 실제 환경에서도 유의미한 결과를 보여주어 검증 기술의 유효성을 입증했습니다.



### Uncovering Vulnerability of Vision-Language-Action Models under Joint-Level Physical Faults (https://arxiv.org/abs/2606.10501)
- **What's New**: 이번 연구에서는 Vision-Language-Action (VLA) 모델이 로봇에 배치될 때의 신뢰성을 높이기 위해, 로봇의 신체적 결함, 특히 조인트 수준의 물리적 결함에 대한 취약성을 분석하였습니다. 연구 결과, 조인트 잠금, 범위 제한 및 마찰 증가와 같은 결함이 VLA 모델의 성능 저하를 초래하며, 이러한 결함이 정책(Policy)-명령 동작과 로봇-실현 동작 간의 불일치를 야기함을 보여주었습니다. 이를 해결하기 위해, Joint-level Physical-fault Aware Residual Calibrator (J-PARC)라는 경량의 보정 프레임워크를 제안하였습니다.

- **Technical Details**: J-PARC는 로봇의 최근 조인트 동역학에서 현재의 조인트 결함 상태를 추정하고, 이 상태를 기반으로 공유된 잔차 보정기를 활용하여 조정된 행동 수정 방안을 생성합니다. 이를 통해 명령된 행동을 관찰된 실행 맥락에 맞게 조정하여 결함이 있는 조인트를 통해 발생하는 정책-commanded actions와 로봇-실현 동작 간의 불일치를 보완합니다. 이 과정에서 VLA 정책을 미세 조정하거나 반복해서 적대적인 샘플을 생성할 필요 없이 원래의 정책을 유지할 수 있습니다.

- **Performance Highlights**: J-PARC의 실험 결과, 조인트 수준의 결함이 있는 환경에서도 로봇의 신뢰성이 크게 향상되었음을 보여주었습니다. 연구에서 제안된 방법은 결함이 없는 환경에서도 VLA 정책의 성능을 유지하며, 다양한 조인트 및 결함 모드에 걸쳐 신뢰성을 높였습니다. 결론적으로, J-PARC는 VLA 모델의 룰을 개선하고 현실적인 로봇 시스템에서의 안정적인 작동을 위한 중요한 기여를 하고 있음을 확인하였습니다.



### Act on What You See: Unlocking Safe Social Navigation in Vision-Language-Action Models (https://arxiv.org/abs/2606.10495)
- **What's New**: 본 연구에서는 로봇이 사람을 일반 장애물과 구분하고 위험을 미리 예측하여 안전한 사회적 내비게이션을 수행해야 한다고 강조합니다. 기존의 Vision-Language-Action (VLA) 모델이 보행자와 객체 사이의 구별 및 미래의 충돌 신호를 내부적으로 인코딩하고 있지만, 행동 복제(Behavior Cloning) 방법은 이를 사회적으로 적절한 행동으로 전환하지 못함을 보여줍니다. 이를 해결하기 위해 저자들은 SALSA라는 두 단계의 주석이 필요 없는 후속 훈련 프레임워크를 제안합니다.

- **Technical Details**: SALSA는 크게 두 가지 단계로 구성되어 있습니다. 첫 번째 단계인 사회적 행동 조정(Social Behavioral Alignment)은 인간-객체 반사적 장면 쌍에서 중간 사회적 특징을 행동 헤드와 연결하여 정책이 유사하지만 사회적으로 다른 상황에서 다른 행동을 생산하도록 훈련합니다. 두 번째 단계인 시간적 안전 조정(Temporal Safety Alignment)은 미래 위험을 자동으로 생성한 라벨로 주어진 경로를 재라벨링하면서 충돌이 임박하기 전에 예측 가능한 반응을 가능하게 합니다.

- **Performance Highlights**: SALSA 프레임워크는 SCAND 데이터셋과 실제 로봇 배치에서 평가되었습니다. 기존의 행동 복제 VLA와 비교할 때 SALSA는 근접 충돌 비율을 86.4% 줄였고 사회적 반사적 정확도를 53%에서 93%로 향상시켰습니다. 이로써 사전 훈련된 VLA 정책이 기존의 내부 표현을 활용하여 보다 안전하고 사회적으로 민감한 내비게이션을 달성할 수 있다는 것을 보여주었습니다.



### GuideWalk: Learning Unified Autonomous Navigation and Locomotion for Humanoid Robots across Versatile Terrains (https://arxiv.org/abs/2606.10449)
- **What's New**: 이번 연구는 GuideWalk라는 새로운 주제 프레임워크를 제시합니다. 이 프레임워크는 다양한 지형에서의 내비게이션과 감지 가능한 보행을 통합하여 안정적인 로봇 내비게이션을 목표로 합니다. 특히, 장애물 회피를 지형 조건에서 분리하여 저항력이 강한 계획을 가능하게 하는 명확한 속도 안내 기능을 포함하고 있습니다.

- **Technical Details**: GuideWalk 시스템은 동적 안정성과 장애물 회피 간의 균형을 잘 유지할 수 있도록 설계되었습니다. 두 단계의 훈련 파이프라인을 통해 교사-학생 방식을 활용하여 목표 지향적 명령과 동급 일관된 행동을 하나의 정책으로 요약합니다. 교사 정책은 심층 이미지를 사용하여 지형을 이해하고, 학생 정책은 교사의 행동을 모방하는 방식으로 훈련됩니다.

- **Performance Highlights**: 실험 결과, GuideWalk는 다양한 환경에서 안정적이고 효과적인 내비게이션을 제공하며, 고유한 휴머노이드 보행 능력을 유지할 수 있음을 입증하였습니다. 로봇이 복잡한 지형에서 안정적으로 탐색할 수 있는 능력을 강화하기 위해 강화 학습과 보조 행동 복제 목표를 활용하는 방식이 주효하였습니다.



### Information-Preserving Continuous Occupancy Mapping with Variance-Weighted Submap Joining (https://arxiv.org/abs/2606.10442)
Comments:
          12 pages, 7 figures

- **What's New**: 이번 논문은 기존의 점유 기반(submap joining) 방법보다 더 정교한 연속 확률적(submap joining) 프레임워크를 제안합니다. 이 프레임워크는 위치 추정(pose estimation)과 글로벌 점유 필드(global occupancy field)를 함께 최적화하여, 추정된 경로를 보다 정확하게 반영할 수 있습니다. 이 방법은 복잡한 처리 비용을 줄이면서도 높은 정확도를 유지하도록 설계되었습니다.

- **Technical Details**: 제안된 프레임워크는 정보 보존을 위한 희소 베이esian(sparse Bayesian) 접근법을 사용합니다. 이 방식은 원래 관찰 속성을 유지하면서 로그 오즈(log-odds) 튜플로 압축하여 후방 정보(posterior information)를 보존합니다. 이를 통해 점유 매핑(occupancy mapping)에 대한 예측 평균과 분산 값을 제공하여, 더 높은 정확도를 가진 서브맵 결합(submap joining) 공식을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 의하면 제안된 방법은 기존의 그리드 기반 서브맵 결합 방법보다 더 높은 위치 정확도(pose accuracy)와 개선된 글로벌 일관성(global consistency)을 보여줍니다. 또한 더 компакт한 지도 표현과 잘 보정된 불확실성 추정(uncertainty estimates)을 제공하여 대규모 실세계 데이터셋에서 효과적으로 작동합니다. 이러한 성능은 현재 가장 연관된 베이esian 방법들보다도 뛰어난 결과를 제공합니다.



### UMI-Bench 1.0: An Open and Reproducible Real-World Benchmark for Tabletop Robotic Manipulation with UMI Data (https://arxiv.org/abs/2606.10382)
- **What's New**: 이 논문은 UMI-Bench 1.0을 소개하며, 이는 UMI 스타일의 조작 정책을 위한 실 로봇 성능 평가의 표준화된 기준점입니다. 기존의 벤치마크들은 UMI 데이터에 맞춰 설계되지 않았기 때문에, 이를 보완하기 위해 UMI-Bench는 데이터 수집, 장면 재설정, 정책 실행 및 결과 기록 등을 통합한 프로토콜을 제공하고 있습니다. 이 평가 프로세스를 재현 가능하고 감사 가능하게 만들어, UMI 훈련 정책이 실제 물리적 조작에 어떻게 일반화되는지를 측정할 수 있는 실용적인 테스트베드 역할을 합니다.

- **Technical Details**: UMI-Bench 1.0은 관찰 인터페이스, 조작 공간, 데이터 스키마 및 평가는 UMI 스타일 정책 훈련 중 사용된 가정을 일치시켜야 합니다. 하드웨어 구성과 작업 공간, 객체 재설정 절차, 에피소드 메타데이터 및 점수 프로토콜이 충분히 상세하게 지정되어야 다른 사이트에서도 기준을 재구성할 수 있습니다. 평가에서는 작업 완성도, 부분 진행 및 하위 목표 달성을 평가하며, 태스크 정의 변동 요인에 따라 보지 못한 조건이 조직됩니다.

- **Performance Highlights**: UMI-Bench는 태스크 성공, 부분 진행 및 하위 목표 달성을 평가하는 데 초점을 맞추고 있으며, 이를 통해 정책 성능을 정량적으로 분석할 수 있습니다. 자주 보지 못하는 조건에서의 일반화 실패를 설명할 수 있도록, 태스크는 공간적 추론, 잡기 안정성, 배치 정밀도, 접촉 상호작용 및 다단계 실행과 같은 능력 관련 속성으로 주석이 달립니다. 첫 번째 배포에서 테이블 조작 작업과 함께 표준화된 시연 및 실제 평가 에피소드를 수립하여, 신뢰할 수 있는 평과 과정이 이루어질 수 있도록 하였습니다.



### Test-time Adversarial Takeover: A Real-time Hijacking Interface against Robotic Diffusion Policies (https://arxiv.org/abs/2606.10371)
- **What's New**: 이번 연구에서는 Diffusion 기반 액션 생성 기술이 로봇 정책의 근본적인 요소로 자리잡고 있음을 강조하며, Test-time Adversarial TakeOver (TAKO)라는 새로운 공격 방식을 소개합니다. TAKO는 공격자가 고정된 로봇 정책을 실시간으로 조작할 수 있는 인터페이스를 제공, 정의된 궤적을 따라서 로봇을 컨트롤할 수 있게 합니다. 우리는 이러한 능력이 어떻게 비주얼 입력 경로의 취약성을 이용해 작동하고, 기존의 목표-정책 일치를 활용하는 공격 방식이 갖는 한계를 설명합니다.

- **Technical Details**: 검색 프로세스는 시각 기능(visual features)을 기반으로 하여 조건부 생성(diffusion) 모델을 사용하여 연속적인 신경망을 통해 액션 시퀀스를 생성합니다. TAKO는 카메라 스트림에 여러 개의 사전 최적화된 보편적 적대 패치(universal adversarial patches)를 주입하여 공격자가 실시간으로 로봇의 출력을 제어할 수 있게 합니다. 이 과정에서 접근 방식은 경향의 바이어스를 지속적으로 생성하게 되어, 비주얼 입력 경로에서의 신호 왜곡이 반복 생성 과정에서 수정되지 않도록 합니다.

- **Performance Highlights**: TAKO 방식은 4개의 작업(2D 조작, 시뮬레이션된 공중 배송, 시뮬레이션된 지상 내비게이션, 실제 지상 내비게이션)에서 인간 조작자가 100%의 성공률을 기록할 수 있었음을 보여줍니다. 기존의 목표-정책 일치(Target-Policy Matching, TPM) 방식은 피해자 정책이 편향된 목표 방향으로 일반화하지 못하고 완전히 실패하는 반면, TAKO 방식은 비디오 패치 기반 인터페이스를 통해 공격자가 정의한 궤적을 실시간으로 생성할 수 있는 가능성을 제공합니다.



### A Practical Recipe Towards Improving Sim-and-Real Correlation for VLA Evaluation (https://arxiv.org/abs/2606.10366)
Comments:
          20 pages

- **What's New**: 이번 연구에서는 비전-언어-행동(VLA) 정책의 평가 및 개선을 위한 시뮬레이션의 유용성을 조사하며, 신뢰할 수 있는 실세계 평가를 위한 시뮬레이션의 역할을 분석합니다. 시뮬레이터와 실제 환경 간의 상관관계를 통해 정책 등급 일관성, 성능 상관관계 및 왜곡 실패 패턴을 측정하여 기존 시뮬레이터의 한계를 규명합니다. 또한, 정책 개선을 위한 시뮬레이터 사용 방법과 후속 학습 데이터의 양이 시뮬레이션과 현실의 정렬에 미치는 영향을 다룹니다.

- **Technical Details**: 다양한 시뮬레이션 플랫폼과 VLA 정책을 포함하여 9개의 테이블탑 조작 작업을 통해 체계적인 평가 플랫폼을 구축했습니다. 각 작업은 비전, 언어, 레이아웃 및 행동 등 4개의 차원에서 제어된 왜곡을 포함하며, 이를 통해 각 시뮬레이터가 실제 정책 평가 결과를 얼마나 잘 유지하는지를 평가합니다. 연구 결과, 신뢰할 수 있는 시뮬레이터는 정책이 실제로 실패하는 방식을 유사하게 반영해야 함을 보여줍니다.

- **Performance Highlights**: 실험 결과, 기존 시뮬레이션 VLA 벤치마크가 실제 모델 랭킹과 강건성 패턴을 얼마나 정확하게 예측할 수 있는지를 분석하였습니다. 시뮬레이터 기반 후속 학습이 실세계 성능 및 시뮬레이션-현실 평가 정렬을 어떻게 향상시킬 수 있는지를 연구하였으며, 적절한 데이터 양에서의 미세 조정이 모델 행동이 실제 세계와 더욱 유사해질 수 있도록 함을 확인하였습니다.



### HiMem-WAM: Hierarchical Memory-Gated World Action Models for Robotic Manipulation (https://arxiv.org/abs/2606.10363)
- **What's New**: 이번 논문에서는 로봇 조작에서의 작업 관련 기억을 강화하기 위해 HiMem-WAM이라는 새로운 계층적 메모리 게이팅 월드 액션 모델을 제안합니다. HiMem-WAM은 저수준 모션 및 고수준 기술을 결합하여 계층적 데이터를 구조화하며, 경계 기반 메모리 업데이트를 통해 작업 상태를 기록합니다. 이를 통해 기존 모델들이 가진 장기 조작의 비효율성을 개선하고, 로봇이 오클루전 상태의 서브태스크를 기억할 수 있는 기능을 제공합니다.

- **Technical Details**: HiMem-WAM은 로봇의 움직임을 모션 중심으로 추상화하고, 이를 고수준 스킬 레이턴트로 나누어 구성된 두 단계의 레이턴트 구조를 집어넣습니다. 이 모델은 플래너가 관찰, 언어 지시, 고유 수용감(proprioception) 및 메모리를 바탕으로 현재의 스킬을 예측하고, 실행자는 이를 바탕으로 저수준의 행동을 확장합니다. 또, 메모리 게이트는 예측된 기술 전환점에서만 작업 상태를 기록하고, 역사적 맥락을 외부 메모리 뱅크에서 불러와 계획자를 조건짓는 방식으로 작동합니다.

- **Performance Highlights**: HiMem-WAM은 다양한 벤치마크에서 높은 성능을 보여주었습니다. LIBERO에서 97.7%, Zero-Shot LIBERO-PLUS에서 76.0%, RMBench에서 26.3%를 기록하며, 현실 세계의 어려운 작업에서는 평균 22.5% 향상을 이루었습니다. 이러한 결과들은 HiMem-WAM이 배포 perturbations에 대한 강건성을 개선하고, 장기 메모리에 의존하는 작업에서 일관된 성과를 제공함을 보여줍니다.



### Rethinking Embodied Navigation via Relational Inductive Bias (https://arxiv.org/abs/2606.10348)
- **What's New**: DB-Nav는 대개 기존의 시맨틱 단서가 갖는 정보를 남용하는 문제를 해결하기 위해 이중 관계 편향을 통해 탐색 공간을 재구성하는 내비게이션 프레임워크입니다. 이는 양성 친화성 편향(positive affinity bias)과 부정 억제 편향(negative inhibition bias)으로 나뉘어진 타겟 중심 객체 관계를 활용하여 객체 내비게이션의 신뢰성을 강화합니다. 이를 통해 물체 탐색에서 발생할 수 있는 오류를 줄이고 우선순위의 결정 프로세스를 개선합니다.

- **Technical Details**: DB-Nav는 Relational Activation-Inhibition Exploration Graph를 통해 온라인 관찰 및 실패한 접근을 통해 탐색 가치를 조절하는 시스템을 사용합니다. 이는 객체 중심의 관계를 두 가지 코드로 나누고, 첫째는 맥락적 동시 발생을 기반으로 유망한 지역을 활성화하고 둘째는 인지적 혼동이나 실패한 검증에 의해 발생하는 신뢰할 수 없는 지역을 억제하는 방식으로 작동합니다. 이 프레임워크는 비용이 많이 드는 온라인 VLM(vision-language models) 조정을 필요로 하지 않으며, 연산 효율성과 해석 가능성을 제공합니다.

- **Performance Highlights**: DB-Nav는 ObjectNav 벤치마크에서 기존 방법들보다 성공률(SR)과 경로 길이에 따라 가중된 성공률(SPL)에서 상당한 성과를 기록했습니다. 이를 통해 DB-Nav는 불안정한 퍼지 세맨틱 단서 속에서도 내비게이션의 강건함을 유지하며 효율적인 객체 탐색을 가능하게 합니다. 실험 결과는 DB-Nav가 더 높은 성공률과 경로 효율성을 제공하며, 다양한 실험 조건에서도 고른 성능을 나타냅니다.



### OMG: Omni-Modal Motion Generation for Generalist Humanoid Contro (https://arxiv.org/abs/2606.10340)
Comments:
          Project Page: this https URL

- **What's New**: 최근 몇 년간 의인화된 전신 제어 분야는 큰 발전을 이루었지만, 기존 접근 방식은 몇 가지 기술에 한정되어 있거나 새로운 입력 모드에 적응하기 어려운 모션 트래커에 의존하고 있었습니다. 이 연구에서는 인간 시스템의 위계 구조를 반영한 스케일 가능한 뇌를 구축하는 것, 즉 다양한 조정 모드로 추론 기능을 갖춘 모듈이 전신 제어의 열쇠라고 주장합니다. 이를 위해 OMG라는 프레임워크가 제안되었으며, 이는 다양한 조정 모드로부터 고품질 데이터를 수집하고, 다중 모드 입력에 대해 조건화할 수 있는 생성기의 기능을 가지고 있습니다.

- **Technical Details**: OMG는 대규모 다중 모드 의인화된 모션 데이터 세트인 OMG-Data를 활용하여 1000시간 이상의 모션을 체계적으로 큐레이션하고 주석을 달아 단일화된 모션 공간으로 정렬합니다. 또한, OMG-DiT라는 확산 기반 모션 생성 백본을 통해 언어, 오디오, 인간 참조 모션 등 다양한 입력을 로봇 실행 형태로 변환하는 기능을 제공합니다. 이는 새로운 모드가 가벼운 조건 인코더를 통해 쉽게 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: OMG는 고품질의 실행 가능한 모션을 다양한 모드에서 생성할 수 있는 능력을 검증받았습니다. 실험 결과에 따르면, OMG는 예측 가능한 스케일링 및 샘플 효율적인 적응 능력, 그리고 제로샷 신호 조합을 보여줍니다. 이러한 결과는 일반화된 의인화된 제어가 더 강력한 저수준 제어기뿐만 아니라 인간의 의도를 물리적 실행과 연결하는 모션 생성 뇌의 확장을 통해 발전할 수 있음을 시사합니다.



### SARM2: Multi-Task Stage Aware Reward Modeling for Self Improving Robotic Manipulation (https://arxiv.org/abs/2606.10305)
- **What's New**: 이번 논문에서는 SARM2라는 다중 작업 단계 인식 보상 모델과 SPIRAL이라는 자가 정책 개선 프레임워크를 제안합니다. SARM2는 행동 원시(Action Primitive) 기반 단계 추정기와 MMoE(Mixture-of-Experts) 가치 헤드를 결합하여 조작 작업 전반에 걸쳐 밀도 높은 단계별 보상을 생성합니다. SPIRAL은 저비용 자율 롤아웃을 통해 VLA 정책을 개선할 수 있도록 돕습니다.

- **Technical Details**: SARM2는 일반적인 행동 원시 단계 추정기를 활용하여 다양한 작업에 대한 단계를 정확하게 평가합니다. MMoE 가치 헤드는 선택된 행동 원시에 따라 가장 관련성이 높은 전문가를 활성화하여 밀도 높은 보상을 생성합니다. 이 모델은 100개 작업을 포함하여 200시간 분량의 실제 조작 데이터를 기반으로 구성되었습니다.

- **Performance Highlights**: SPIRAL을 통해 SARM2는 Folding Shorts와 Cleaning Whiteboard와 같은 두 개의 실제 긴 호라이즌 조작 작업에서 이전의 희소 보상 및 대규모 VLM 기반 보상 모델보다 월등한 성과를 보여줍니다. 실험 결과, SPIRAL을 적용했을 때 작업 성공률이 Folding Shorts의 경우 58%에서 100%로, Cleaning Whiteboard의 경우 50%에서 90%로 개선됩니다.



### Improved Representation of Matrix Lie Group Operations through Tensor Notation (https://arxiv.org/abs/2606.10289)
Comments:
          12 pages, 4 figures + graphical abstract, 1 algorithm, 4 tables

- **What's New**: 이 논문에서는 행렬 Lie 그룹(Matrix Lie Groups)과 관련된 연산을 설명할 새로운 도구로 텐서(tensor)와 아인슈타인 합산 표기법(Einstein summation notation)을 소개합니다. 이 수학적 표기법을 통해 행렬 Lie 도함수를 표현하고 계산하는 방식이 혁신적으로 개선됩니다. 중요한 사항은 이 새로운 표기법이 행렬 Lie 그룹을 다루는 데 필요한 도함수 및 연산을 더욱 명확하게 해준다는 점입니다.

- **Technical Details**: 행렬 Lie 그룹을 다루는 기술적 문제는 다음과 같은 두 가지 기본적인 어려움이 있습니다. 첫 번째는 복잡한 방정식 내에서 행렬의 도함수를 계산할 때의 표현적 어려움이며, 두 번째는 매니폴드(manifold) 내에서의 (지역) 이동을 작은 차원의 벡터로 표현하는 개념입니다. 이 논문에서는 텐서 및 아인슈타인 표기법을 사용하여 이러한 도함수 및 그 상호작용을 더 잘 표현할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 논문은 수학적 표기법의 명확성을 통해 행렬 Lie 그룹을 다루는 데 있어서의 복잡성을 줄여줄 것으로 기대됩니다. 특히, 그래디언트 기반 최적화 알고리즘(context of gradient-based optimization algorithms)에서의 활용 가능성이 강조됩니다. 또한, 텐서를 활용한 행렬 Lie 그룹의 연산 예제를 통해 이론적 접근을 보강하고 있어 연구자들에게 실용적인 코드도 제공됩니다.



### MARCH: Model-Assisted Reinforcement Learning for the Perceptive Control of Humanoids over Sparse Footholds (https://arxiv.org/abs/2606.10288)
- **What's New**: 이번 연구는 희박한 지형에서의 안전한 이족 보행을 달성하기 위한 새로운 모델 보조 강화 학습(모델-보조 강화 학습) 프레임워크를 제안합니다. 이는 세 가지 단계로 구성되며, 안전한 참조 궤적(reference trajectory)을 생성하고, 제어 Lyapunov 함수(控制 Lyapunov function; CLF)를 기반으로 한 교사 정책을 훈련한 후, 이를 시각 기반 학생 정책으로 증류(distill)하는 방식으로 이루어집니다. 이러한 접근은 더 높은 샘플 효율성을 제공하며 복잡한 학습 과정을 줄이고 부드러운 보행 행동을 달성합니다.

- **Technical Details**: 이 연구에서는 하이브리드 시스템 모델을 통한 일반 보행을 설명합니다. 기본적으로 이족 로봇은 연속적인 동작과 충격으로 인한 이산적인 변화를 포함하는 하이브리드 시스템으로 모델링됩니다. 특정 제어 입력 공간과 연속 동역학을 정의하여 로봇의 보행을 정확히 기술합니다. 또한 다단계 안전성을 개선하기 위해 값비싼 영상을 활용하는 변환기(transformer)와 혼합 밀도 네트워크(Mixture Density Network; MDN)를 추가하여 다양한 안전 발판을 선택할 수 있도록 합니다.

- **Performance Highlights**: 시뮬레이션과 Unitree G1 휴머노이드 로봇을 이용한 실제 실험을 통해 이 접근 방식의 효율성을 검증했습니다. 로봇은 희박한 발판에서의 보행 중 안전성을 유지하며 뛰어난 성능을 보여주고, 모델-프리 기준선에 비해 매끄러운 보행 행동을 구현했습니다. 이를 통해 로봇이 새로운 환경에서도 안전하게 다양한 상황에 적응할 수 있는 가능성을 제시합니다.



### Hierarchical Policies from Verbal and Egocentric Human Signals for Natural Human-Robot Interaction (https://arxiv.org/abs/2606.10276)
Comments:
          We provide video demos and code in: this https URL

- **What's New**: 이번 연구는 인간-로봇 상호작용을 보다 자연스럽게 만들기 위해 비언어적 신호(예: 시선, 제스처)를 언어 지시와 함께 활용하는 EDITH라는 로봇 프레임워크를 소개합니다. 기존 로봇 정책은 언어 지시만을 사용하여 의사를 전달했으나, EDITH는 실시간으로 스마트 안경을 통해 인간의 첫 번째 시점 보기와 시선을 포착하여 이를 로봇 정책의 입력으로 사용합니다. 이러한 접근은 로봇이 인간의 의도를 더 잘 이해하고 행동할 수 있게 해줍니다.

- **Technical Details**: EDITH는 Project Aria 안경을 기반으로 하여 인간의 첫 번째 시점 보기, 시선, 음성을 실시간으로 스트리밍합니다. 이 시스템은 고수준 정책과 저수준 정책으로 구성된 계층형 정책을 통해 작동하며, 고수준 정책은 시각적 및 언어적 신호를 통해 인간의 의도를 추론하고 세부 작업을 생성합니다. 이를 통해 로봇은 비언어적 신호를 기반으로 행동을 수행하고, 이러한 세부 작업은 해당 의도를 장면에서 정박하는 키프레임으로 연결됩니다.

- **Performance Highlights**: 실험 결과, EDITH는 언어 지시만 사용하는 기존 방법과 비교하여 평균 59.7%의 성공률을 달성하여 인간의 비언어적 신호를 통해 의도를 인식하는 데 혁신적인 효과를 보였습니다. 사용자 연구에 따르면, EDITH는 로봇에게 의사를 전달하는 작업의 부담을 크게 줄여주며, 이는 자연스러운 상호작용을 제공함을 보여줍니다. 이러한 결과는 비언어적 신호를 포함하는 접근 방식이 인간-로봇 상호작용을 위해서 더욱 유망하다는 것을 의미합니다.



### Locomotion analysis of a quadruped interacting with the lunar granular surfac (https://arxiv.org/abs/2606.10273)
- **What's New**: 이 논문에서는 외계 환경에서 다리 로봇을 배치하는 데 있어 복잡한 지형 상호작용과 에너지 및 열 제약에 따른 문제를 다룹니다. 특히, 달 탐사를 위한 사족 보행 로봇의 기계적 설계가 motor torques, energy expenditure, cost of transport와 같은 요소를 고려해야 함을 강조합니다. 본 연구는 입자성 달 표면과 로봇 발의 접촉을 물리적으로 모델링하고, 이를 Reinforcement Learning을 통해 훈련된 시뮬레이션 환경에 적용합니다. 로봇의 운동 성능을 분석하기 위해 기존의 강체 접촉 가정과 연성 접촉 환경에서 학습된 정책을 비교합니다.

- **Technical Details**: 달 표면은 입자성 regolith로 구성되어 있으며, 이는 로봇의 보행 및 운동 성능에 영향을 미칩니다. 기존의 rigid contact를 가정한 운동 알고리즘은 입자성 표면과 같은 연성 접촉 환경에서는 효과적이지 않아 불안정성과 낮은 추적 성능을 초래할 수 있습니다. 본 논문에서는 RL을 활용하여 입자성 표면에서의 운동 성능을 분석하기 위해 시뮬레이션 환경을 구축하고, 두 가지 환경(강체 접촉 및 연성 접촉)에서 학습된 보행 정책을 비교 분석합니다. 이러한 접근 방식은 다리 로봇의 기계적 형태와 운동 제어기를 안전하고 견고하게 설계할 수 있는 중요한 데이터를 제공합니다.

- **Performance Highlights**: 로봇의 운동 성능을 평가한 결과, 연성 접촉 환경은 RL 기반 훈련에 추가적인 도전을 제공하며, qualitatively 다른 보행 양상을 만들어 냅니다. 또한, 연성 접촉에서는 에너지 소비가 증가함을 보여줍니다. 각 환경에서의 torques, power consumption, energy expenditure 측면에서 데이터 분석을 통해, 로봇의 최적 성능을 발휘하기 위해서는 어떠한 형태와 행동 패턴이 요구되는지를 밝혔습니다. 이러한 분석은 향후 달 탐사 로봇의 설계 및 운동 모델 개발에 중요한 기초 자료가 될 것입니다.



### What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents (https://arxiv.org/abs/2606.10267)
- **What's New**: 이 논문은 로봇 조작을 위한 계층적 비전-언어-행동(Hi-VLA) 시스템의 설계 원칙을 체계적으로 분석하고 정리합니다. 기존 Hi-VLA 시스템들은 서로 다른 계획자(planner), 행동자(controller), 관찰 및 메모리 표현 방식에서 차이점을 보이고 있으며, 이러한 분야에서의 통합된 연구가 부족했습니다. 저자들은 다양한 조작 작업을 통해 강력한 Hi-VLA 시스템을 구축하기 위한 실용적인 원칙들을 제시합니다.

- **Technical Details**: 저자들은 옵션 프레임워크(options framework)를 기반으로 한 공유 제어 루프(control loop) 아래에서 다양한 Hi-VLA 에이전트를 통합합니다. 이 연구에서는 VLM(vLanguaged Model)과 VLA(Vision Language Action) 정책 간의 상호작용을 다루며, 메모리 모듈(memory module)과 관찰 표현 모듈(observation representation module)을 통해 계층적 시각 운동 정책을 정의합니다. 이를 통해 시스템 설계를 평가하고, 각 구성 요소의 구현이 결과에 미치는 영향을 체계적으로 분석합니다.

- **Performance Highlights**: 실험 결과, 단순한 Hi-VLA 디자인도 일반적인 VLA보다 개선된 성능을 보이나, 신중하게 선택된 피라미드 구조의 설계가 특히 긴 수평(long-horizon) 및 추론-intensive 작업에서 더 큰 이득을 가져온다는 것을 발견했습니다. 강력한 Hi-VLA 성능은 모델 백본(backbone)과 인터페이스 간의 상호작용에 의존하며, 저자들은 이러한 발견이 더 강력하고 견고한 계층적 VLA 에이전트를 구축하는 데 기여할 수 있음을 강조합니다.



### YUBI: Yielding Universal Bidigital Interface for Bimanual Dexterous Manipulation at Sca (https://arxiv.org/abs/2606.10244)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서는 직관적이고 인체 공학적이며 확장 가능한 데이터 수집을 위한 손가락 정렬 그리퍼인 YUBI(수축형 범용 양손 인터페이스)를 소개합니다. 기존의 UMI(범용 조작 인터페이스)와 비교해, YUBI는 손가락의 자연스러운 움직임을 직접적으로 그리퍼의 조작으로 매핑하여 기존 그리퍼 시스템의 단점인 인체 공학적 문제를 해결합니다. 또한, YUBI는 가벼운 디자인과 정밀한 조작이 가능한 구성을 가지고 있으며, 대규모 데이터 수집을 지원하기 위해 다양한 실험과 사용자 연구를 통하여 그 효과를 입증하였습니다.

- **Technical Details**: YUBI는 손가락 정렬 인터페이스를 기반으로 하여 작업자의 자연스러운 집게 운동에 맞춰 그리퍼의 간격이 조정되는 측면에서 획기적입니다. 이 시스템은 고주파 VR 센서를 그리퍼에 직접 통합하여 높은 충실도의 그리퍼 궤적 추적을 가능케 하며, 데이터를 수집하는 동안 물리적 소모를 줄여줍니다. 데이터 수집 체계에서는 6 DoF(자유도) VR 추적을 통해 고품질 궤적 데이터를 확보하며, 이를 통해 수집된 데이터는 다수의 로봇 플랫폼에 쉽게 적용 가능합니다.

- **Performance Highlights**: YUBI는 8434시간의 데이터로 구성된 대규모 데이터셋을 기반으로 하고 있으며, 이는 1.20M 에피소드와 119가지 작업에 걸쳐 있습니다. 사용자 연구 결과에 따르면 YUBI는 기존의 UMI 그리퍼보다 복잡한 양손 작업에서 더 나은 versatility(다양성), dexterity(손재주), 그리고 operational efficiency(작업 효율성)를 제공하는 것으로 나타났습니다. 이 연구의 결과는 YUBI의 데이터 기반 정책 네트워크가 UR, Franka, ELEY 등 여러 양손 로봇 플랫폼에서 효과적으로 전이 가능하다는 것을 보여줍니다.



### What Demonstration Curation Metrics Do to Your Policy (https://arxiv.org/abs/2606.10229)
Comments:
          6 pages, 1 figure, 2 tables

- **What's New**: 이 논문에서는 결함이 있는 훈련 에피소드를 식별하는 데 유용한 demonstration-curation metrics가 실제 행동 복제 정책의 향상에도 기여하는지를 연구하였습니다. LIBERO 피킹 앤 플레이스 벤치마크를 사용한 실험 결과, 결함 탐지 AUROC 수치가 가장 높은 메트릭이 실제로는 최악의 정책을 생성한다는 점을 발견했습니다. 반면, 낮은 AUROC를 가진 메트릭이 훨씬 우수한 정책 성과를 가져오기 때문에, 메트릭의 실제 효과를 보여주는 데이터와의 직접 비교가 필요함을 강조합니다.

- **Technical Details**: 연구에서 사용된 LIBERO 벤치마크는 80%의 오염율을 가지는 구조결함이 있는 데이터를 포함하고 있으며, 깨끗한 데이터에서 훈련된 행동 복제 정책은 90%의 성공률을 기록합니다. 추가적으로, 7개의 평가 메트릭 중 5개가 에피소드 길이를 결함 레이블의 대리 변수로 활용하며, 이는 AUROC를 잘못 부풀리는 원인이 됩니다. 최적의 성과를 얻기 위해서는 모든 메트릭이 에피소드 길이를 통제해야 한다고 경고하고 있습니다.

- **Performance Highlights**: 결과적으로, 오염된 기본선은 3.3%의 롤아웃 성공률을 보였고, 가장 우수한 두 가지 큐레이션 방법은 93.3%의 오라클 성과에 비해 단 3%의 차이로 줄어들었습니다. 이 논문은 결국 어떤 큐레이션 메트릭이 좋은 정책을 생산하는지 판단해야 하며, 단순히 결함을 제대로 탐지하는 데 그쳐서는 안 된다는 점을 강조합니다. 이 연구는 큐레이션 메트릭의 성능을 평가할 때 더욱 신중해야 한다는 실질적인 경고로 해석될 수 있습니다.



### Exploration of Foundation Model-Based Robots in Patient and Elderly Car (https://arxiv.org/abs/2606.10208)
- **What's New**: 이번 연구는 노인과 환자의 돌봄 요구가 증가하는 가운데, 페이퍼와 같은 기초 모델 기반 로봇이 이러한 요구를 충족하는 방향을 제시하고 있습니다. 연구에서는 대화 및 추론 기술로 사용되는 기초 모델의 설계 특징, 사용자 경험, 이와 관련된 성과에 대한 증거 등을 종합적으로 살펴보았습니다. 하지만 이러한 로봇 시스템에서 기술적 발전이 실제 돌봄 환경에서의 임상적인 영향을 가져올 수 있는지에 대한 명확한 기준이 부족하다는 점을 강조하고 있습니다.

- **Technical Details**: 노인 돌봄 로봇의 기초 모델은 다섯 가지 주요 역할에서 사용됩니다. 그중 가장 일반적인 역할은 사용자와의 개방형 대화를 생성하여 사회적 연결을 지원하는 것입니다. 구조화된 코칭이나 평가, 작업 흐름 조정 등의 기능도 포함되어 있으며, 이러한 시스템은 주로 음성 기반으로 작동합니다. 그러나 신뢰성과 안정성을 확보하기 위해서는 추가적인 안전 메커니즘과 사용자 요구를 반영한 상호작용이 필요합니다.

- **Performance Highlights**: 사용자 경험에 대한 조사에 따르면, 기초 모델 통합 이후 돌봄 로봇에 대한 수용도가 높아졌다는 결과가 도출되었습니다. 로봇의 응답이 더 일관되고 맥락 이해가 됨에 따라 사용자 써베이 점수가 상승하였음을 나타냅니다. 그러나 여전히 신뢰성과 안전성, 상호작용 부담 등의 요소가 사용자 수용에 중요한 영향을 미친다는 점이 아이러니하게도 개선이 필요하다고 할 수 있습니다.



### Flow Control: Steering Vision-Language-Action Models with Simple Real-Time Inputs (https://arxiv.org/abs/2606.10180)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 연구에서는 시각-언어-행동(VLA) 모델의 흐름 제어(flow control) 방법을 소개합니다. 이 방법은 사용자가 입력한 키보드와 같은 일반적인 입력을 통해 VLA 행동을 실시간으로 조정할 수 있는 간단하고 효과적인 방식입니다. VLA는 이 입력을 사용하여 훈련 중 학습한 행동 전문가 분포에서 고품질의 행동 샘플을 생성하여 사용자 의도와 잘 맞도록 합니다.

- **Technical Details**: VLA는 카메라 관찰과 자연어 명령을 로봇 행동으로 변환하여 최신 성능을 달성하고 있습니다. 그러나 이러한 모델은 사용자 입력을 제대로 따르지 못하거나 새로운 객체와 장면에 일반화하지 못하는 등의 문제점이 있습니다. 흐름 제어는 사용자 입력을 사용하여 VLA 행동을 조정할 수 있도록 하며, 추가적인 정책 훈련 없이도 작동할 수 있습니다.

- **Performance Highlights**: 1616명의 사용자 실험 결과, 흐름 제어를 통해 사용자가 간단한 키보드 입력만으로도 작업 성공률과 완료 속도가 크게 향상되었음을 보여주었습니다. 또한, 흐름 제어 경로에서 π0.5의 미세 조정이 자율 정책 성능을 향상시키는 결과를 얻었습니다. 이 연구는 로봇 정책의 작업 결과를 개선하고 보조 로봇 공학 등 다양한 분야에 응용될 가능성을 제시합니다.



### Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination (https://arxiv.org/abs/2606.10040)
- **What's New**: 본 논문에서는 World-Action Model(WAM)의 새로운 변화인 Efficient-WAM을 소개합니다. Efficient-WAM은 미래 비디오 예측의 공식적인 가이드 신호로서 역할을 하여 제어 성능을 유지하면서도 더 낮은 추론 비용을 목표로 합니다. 또한, 이 모델은 비디오의 해상도를 낮추고 비대칭 비디오-행동 노이즈 제거를 통해 큰 효율성을 제공합니다.

- **Technical Details**: Efficient-WAM은 Mixture-of-Transformers(MoT) 프레임워크를 활용하여 구조적 가지치기를 통해 영상 예측의 효율성을 높입니다. 이 과정에서 WAN-2.2-5B 모델로부터 세분화된 세계 지식을 전이하여 필요한 구조와 동적 신호를 보존합니다. 이렇게 최적화된 비디오 전문가의 아키텍처는 행동 신호 생성을 위한 정교하고 체계적인 방법이 될 수 있습니다.

- **Performance Highlights**: 논문의 실험 결과, Efficient-WAM은 실제 및 시뮬레이션 환경에서 각각 66.25%와 86.7%의 성공률을 보이며 기존 WAM 방법들과 비교하여 상대적으로 높은 성능을 보여주었습니다. 1B 파라미터 모델로서, Efficient-WAM은 매개변수 수를 줄이면서도 물리적 배치에서 약 98 ms의 낮은 지연 시간을 유지하며, 이는 기존 WAM보다 30배 빠른 속도입니다.



### Robotic Nonprehensile Object Transportation with a Hanging Tray (https://arxiv.org/abs/2606.10039)
Comments:
          8 pages, 11 figures. IEEE/ASME International Conference on Advanced Intelligent Mechatronics, 2026

- **What's New**: 이 논문에서는 웨이터의 문제로 알려진 비잡고 운반(nonprehensile object transportation) 작업을 다룹니다. 기존의 연구들과 달리, 로봇이 트레이를 강하게 고정된 끝 단추(end effector)에서 기울이도록 하는 대신, 트레이를 로프에 매달아 자유롭게 흔들릴 수 있게 설정하였습니다. 이 접근법은 상하차되는 물체에 가해지는 전단력(shear forces)을 줄여 물체의 미끄러짐을 최소화하는데 기여합니다.

- **Technical Details**: 논문은 3D 진자(pendulum) 모델을 개발하며, 이를 통해 제어 설계(control design) 및 수치 실험을 진행합니다. 움직이는 로봇 기본(base)만을 제어하고, 조작기(manipulator arm)의 직접적인 작동을 필요로 하지 않으며, 로봇이 도착지에 도달했을 때 트레이의 흔들림을 빠르게 감소시키기 위해 선형 제곱 조절기(LQR)를 사용합니다.

- **Performance Highlights**: 시뮬레이션 및 실제 하드웨어에서는 매달린 트레이가 정적이고 강하게 잡은 트레이에 비해 미끄러짐과 유동이 크게 줄어드는 것을 보여주었습니다. 또한, 움직이는 로봇 웨이터 시연을 통해 RGB-D 카메라를 이용하여 손을 들어 사람을 인식하고, 비주얼 서보링(visual servoing)을 통해 사람에게 다가가 물체를 제공하는 방식을 통합하였습니다. 이 연구는 손 제스처를 활용한 비주얼 서보링을 웨이터의 문제에 활용한 최초의 사례로 기록됩니다.



### GHOST: Hierarchical Sub-Goal Policies for Generalizing Robot Manipulation (https://arxiv.org/abs/2606.10025)
Comments:
          Accepted at RSS 2026

- **What's New**: GHOST는 환경을 조작하는 정책을 학습하는 새로운 프레임워크로, 기존의 데이터 훈련 분포를 넘어 일반화될 수 있는 방법을 제시합니다. 이 프레임워크는 조작 수행을 위한 높은 수준의 정책과 낮은 수준의 목표 조건 제어기를 계층적으로 분리하여 기존의 flat Diffusion Policy보다 성능과 견고성을 향상시킵니다. GHOST의 주요 혁신 중 하나는 3D 하위 목표를 이미지 평면에 투영하여 엔드 이펙터 열지도로 표현하는 간단한 공간 인터페이스를 도입한 것입니다.

- **Technical Details**: GHOST는 두 가지 모듈로 구성된 계층적 정책을 정의합니다. 첫 번째로, 높은 수준의 정책 𝜋ᵗᵢ는 다중 뷰 RGB-D 관찰 및 언어 지시를 바탕으로 다음 하위 목표를 예측합니다. 두 번째로, 낮은 수준의 정책 𝜋ₗₒ는 예측된 목표 기반으로 행동을 생성합니다. 이 구조는 데이터 효율을 높이며, 조작 환경의 다양성에 쉽게 적응할 수 있도록 돕습니다.

- **Performance Highlights**: GHOST는 Robot의 시연을 통한 훈련만으로도 기존 정책보다 높은 성공률을 기록합니다. 이 프레임워크는 인간 영상 자료를 활용하여 고급 정책을 훈련하며, 이는 action retargeting의 필요 없이도 새로운 작업 변형에 대한 일반화를 가능하게 합니다. GHOST는 최종적으로 새로운 물체와 작업 변형에 적은 수의 인간 시연으로도 적응할 수 있는 능력을 입증합니다.



### Uncertainty-Aware Motion Planning for Autonomous Driving in Mixed Traffic Environmen (https://arxiv.org/abs/2606.09958)
- **What's New**: 본 논문에서는 자율주행 차량이 혼합교통 환경에서 인간 운전자의 미래 행동을 예측하는 데 있어 불확실성을 고려하는 'Uncertainty-Aware Motion Planning (UAMP)' 프레임워크를 제안하였습니다. 기존의 강화 학습 기반 방법들은 예측된 인간 의도를 관측치에 직접 통합하는 방식으로, 예측의 불확실성이 증가하여 위험한 결정으로 이어질 수 있는 문제를 지니고 있었습니다. UAMP는 인간 운전자의 의도를 정량화하고 이를 바탕으로 의사 결정을 조정하는 새로운 방법을 소개합니다.

- **Technical Details**: UAMP는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 'Proximity-aware uncertainty estimator'는 주변 차량의 상호작용을 기반으로 한 불확실성을 정량화하며, 둘째, 'Uncertainty-guided joint intent distribution'을 통해 다양한 차량의 행동 불확실성을 집계합니다. 셋째, 'Uncertainty-Calibrated Value Learning (UCVL)'을 도입하여 불확실성이 존재하는 상황에서도 더 안정적인 가치 추정을 가능하게 합니다.

- **Performance Highlights**: 다양한 혼합교통 시나리오에서 수행된 실험 결과, UAMP는 기존의 방법들과 비교하여 안전성과 주행 편안함을 크게 향상 시켰으며, 교통 효율성을 유지하는 데 성공하였습니다. 이러한 성과는 자율주행 시스템의 안전한 운영을 위해 필요한 의사 결정 과정에서 불확실성을 적절히 통합하는 것이 중요하다는 것을 보여줍니다.



### On-sky demonstration of reinforcement learning for adaptive optics contro (https://arxiv.org/abs/2606.10771)
Comments:
          11 pages, 12 figures accepted by A&A

- **What's New**: 이번 연구는 Policy Optimization for AO (PO4AO)라는 이름의 강화학습 기반 제어기를 최초로 실제 하늘에서 시연한 내용을 다룹니다. PO4AO는 적응광학 시스템에 적용되어 실험실과 시뮬레이션에서의 성공적인 결과를 바탕으로 하늘에서의 성능 검증을 진행했습니다. 이를 통해 강화학습 알고리즘의 적응성을 한층 더 높이는 방향성을 제시합니다.

- **Technical Details**: PO4AO는 프랑스 OHP의 T152 망원경 Coudé 초점에 설치된 Papyrus 적응광학 시스템에 구현되었습니다. Python 기반으로 작성된 이 제어기는 기존의 실시간 제어기(DAO RTC)와 공유 메모리 버퍼를 통해 인터페이스를 구성했습니다. 여러 밤에 걸쳐 PO4AO의 성능을 표준 적분기와 비교한 결과, 모든 테스트 구성에서 PO4AO가 일관되게 우수한 성능을 보였습니다.

- **Performance Highlights**: PO4AO는 진동 패턴을 성공적으로 학습하고 보상하는 능력을 보였으며, 측정 노이즈에 대해 강력한 견고성을 입증했습니다. 또한, 적절히 조정된 PO4AO는 다양한 관측 조건과 과학적 목표에 대해 단일 하이퍼파라미터 세트를 사용하여 턴키 방식으로 작동했습니다. 이 연구는 비최적화된 Python 구현에서 추가 지연과 제어 지터가 발생함에도 불구하고, PO4AO가 단일 접 conjugate 적응광학 시스템에 대한 높은 성능과 견고한 제어기를 구성할 수 있음을 보여줍니다.



### Baseline-Free Policy Optimization for Neural Combinatorial Optimization (https://arxiv.org/abs/2606.10321)
- **What's New**: 이번 연구에서는 Neural Combinatorial Optimization(NCO)에서 baseline을 완전히 제거한 Group Relative Policy Optimization(GRPO)을 평가합니다. 기존의 REINFORCE는 rollout baseline을 필요로 하여 훈련 중 불안정성을 초래하는 구조적 취약점을 가지고 있습니다. GRPO는 샘플링된 트랙 제안 내에서 이점을 정규화하여 이러한 문제를 해결할 수 있는 방안을 제공합니다.

- **Technical Details**: GRPO는 각 인스턴스에서 GG 트랙 제안을 생성하고, 각 그룹 내에서 z-score로 이점을 계산하여 외부 baseline이 필요하지 않습니다. 이 방법은 NCO에 적용하기 위해 신뢰할 수 있는 대안으로 제시되며, TSP와 CVRP benchmark에서의 성능을 평가합니다. 연구에서는 REINFORCE, POMO, PPO 및 P3O와 같은 여러 알고리즘과의 비교를 통해 GRPO의 효용성을 입증합니다.

- **Performance Highlights**: 실험 결과, GRPO는 TSP-100에서 REINFORCE에서 관찰된 훈련 붕괴를 피하면서도 POMO에 대해 2% 이내의 솔루션 품질을 달성했습니다. GRPO는 복잡한 인스턴스에서도 안정적으로 성능을 유지하며 외부 비선형 baseline이 필요하지 않은 점에서 매력적인 대안으로 평가되고 있습니다. 반면 P3O는 TSP에서는 경쟁력을 보였으나 CVRP에서는 변동성이 높게 나타났습니다.



### SHAPO: Sharpness-Aware Policy Optimization for Safe Exploration (https://arxiv.org/abs/2606.10228)
Comments:
          ICLR 2026

- **What's New**: 이 논문에서는 안전이 중요한 분야에서 강화 학습(RL) 에이전트를 배치하기 위한 필수 요소로 안전 탐색을 다루고 있습니다. 저자들은 Sharpness-Aware Policy Optimization (SHAPO)이라는 새로운 알고리즘을 제안하여 에이전트의 불확실성에 기반해 정책 업데이트를 수행합니다. SHAPO는 높은 불확실성을 가진 영역에서 에이전트가 안전하게 행동하도록 유도하며, 이를 통해 안전성과 작업 성능을 향상시킵니다.

- **Technical Details**: SHAPO는 정책 업데이트 시 파라미터의 변화에 따른 기울기를 평가하는 방법을 사용합니다. 이 과정에서 희귀한 위험한 행동의 영향을 강화하고, 이미 안전한 행동의 영향을 줄이기 때문에, 학습을 보수적으로 유도합니다. 저자들은 Fisher metric을 활용한 변동성이 Euclidean metric보다 우수하며, 액터(Actor)쪽의 위험 회피가 안전 탐색에 더 중요한 영향을 미친다고 입증합니다.

- **Performance Highlights**: SHAPO는 Safety-Gym 및 MuJoCo 환경에서 여러 온폴리시 안전 RL 기법들에 비해 지속적으로 안전성과 작업 성능을 개선했습니다. 이 방법은 누적 실패를 줄이고, 에피소드 비용의 분포를 완화하며 파레토 프론티어를 크게 확장했습니다. 다양한 실험을 통해 SHAPO의 효과를 입증하며, 기존의 방법들과 비교하여 우수한 성능을 보였습니다.



### Generalized-CVO: Fast and Correspondence-Free Local Point Cloud Registration with Second Order Riemannian Optimization (https://arxiv.org/abs/2606.10019)
Comments:
          16 pages, 12 figures

- **What's New**: 본 논문에서는 기하학적 표면 구조와 재생 커널 Hilbert 공간(RKHS) 임베딩을 활용한 빠르고 대응 없는(local point cloud registration) 지역 포인트 클라우드 등록 방법을 제안합니다. 이 방법은 점구름을 연속 함수로 표현하며, 점 별 비등방성 커널을 통해 지역 기하학을 인코딩합니다. 이러한 공식화는 표면 법선에 대한 정렬을 개선하고 접선 방향으로의 정렬을 완화합니다.

- **Technical Details**: 문제 해결을 위해 근사 리만 헤세안(approximate Riemannian Hessians)을 적용한 이차(on-manifold) 최적화 방법을 제안하며, 이전의 대응 없는 RKHS 기반 방법들에서 사용된 일차 솔버들에 비해 최대 10배의 속도 향상을 달성합니다. 이 방법은 주로 LiDAR와 RGB-D 데이터를 통한 프레임 간 추적 정확도 향상에 효과적입니다. 이차 최적화 기법은 복잡한 기하학적 형태의 효율성을 극대화합니다.

- **Performance Highlights**: 주행 분야의 LiDAR 추적 등록 작업에서, 도전적인 특성 희소 환경에서 변환드리프트(translational drift)와 회전드리프트(rotational drift)가 55% 이상 감소하는 성과를 보였습니다. 또한 ICP 기반 방법들에 비해 객체 등록 벤치마크에서 더욱 향상된 견고성을 입증했으며, 특히 중간 정도의 비정렬(misalignment) 상황에서 글로벌 초기화(refining global initialization) 시 더욱 성능이 개선되는 결과를 보여주었습니다.



### Co-GLANCE: Uncertainty-Aware Active Perception for Heterogeneous Robot Teaming (https://arxiv.org/abs/2606.09919)
Comments:
          Code, videos, and dataset available at this https URL

- **What's New**: Co-GLANCE는 이질적인 로봇 팀을 위한 실시간 인식 및 의사 결정 시스템입니다. 이는 비전-언어 모델(vision-language model)의 의미적 추론 능력을 경량화된 모델로 단순화하여 클라우드 기반 추론을 제거합니다. Co-GLANCE는 성과 기반의 불확실성 정량화를 위해 선택적 어떤 것을 결합하여 분할 및 로봇 할당을 위한 통계적으로 유효한 보장과 함께 이전에 제기된 문제를 해결합니다.

- **Technical Details**: Co-GLANCE는 두 가지 주요 기술을 통합하여 작동합니다. 첫째, 정보가 부족한 영역을 식별하기 위해 문맥 인식을 사용하는 폐쇄 세그먼트 모델을 사용합니다. 둘째, 선택적 어떤 것(selective abstention)과 준거적 예측(conformal prediction)을 결합하여 로봇 할당, 개체 탐지 및 세그멘테이션에 대한 보정된 불확실성 추정을 제공함으로써 적극적인 인식을 유도합니다.

- **Performance Highlights**: Co-GLANCE는 실제 상황에서 클라우드 기반 비전-언어 모델 기반 (baseline)보다 폐쇄 세그멘테이션과 로봇 할당 정확도를 각각 25%와 36% 높이며, 프레임당 추론 지연 시간을 350배 줄였습니다. 또한, 향후 연구를 위한 공중-지상(multimodal air-ground) 데이터셋을 발표합니다.



### Equanimity in HRI: Applying Calm Technology Principles to Human-Robot Interaction (https://arxiv.org/abs/2606.09836)
Comments:
          Conference pre-print. this https URL

- **What's New**: 이 논문은 {	extit{Calm Technology}}의 원칙을 인간-로봇 상호작용(HRI)에 통합하는 방법을 탐구하며, 특히 가정 환경에 중점을 둡니다. 이를 통해 인간의 {	extit{equanimity}}(평정심)를 중시하고 향상시키는 지원 로봇 설계를 위한 포괄적인 지침을 제공합니다. 기술이 현대 생활에서 미치는 광범위한 영향과 인지 능력에 대한 영향을 강조하며, 책임 있는 로봇 공학과 윤리적 고려 사항이 미래의 기술 발전에서 필요함을 강조합니다.

- **Technical Details**: 이 논문은 기술이 사용자에게 미치는 심리적 영향을 연구하며, 특히 기술의 과도한 자극이 인지 기능 저하 및 우울증, 불안장애 등의 정신 건강 문제를 초래할 수 있음을 보입니다. '디지털 디톡스' 전략을 구현하는 것이 이러한 부정적인 영향을 완화하기 위한 방법으로 제안되며, 기술 설계는 사용자가 과도한 인지 자원 요구를 받지 않도록 해야 함을 주장합니다. 또한, 책임 있는 로봇 공학을 통해 윤리적 고려가 기술 개발의 초기에 포함되어야 한다고 강조합니다.

- **Performance Highlights**: Calm Technology 원칙이 인간-로봇 상호작용에 어떻게 적용될 수 있는지를 탐구하며, 로봇이 사용자 환경에 원활하게 통합되어야 한다고 언급합니다. 이 과정에서 로봇이 사용자의 주의를 방해하지 않고 정보를 제공할 수 있는 방법으로 시각적 신호 및 주변 알림을 활용하는 방안을 제시합니다. 또한, 로봇의 상태와 의도를 명확히 함으로써 사용자의 신뢰를 높이는 투명성이 중요하다고 강조하며, 궁극적으로는 스트레스를 줄이고 건강한 상호작용을 도모하는 기술 설계의 필요성을 주장합니다.



### Deterministic Execution of ROS 2 Applications via Lingua Franca (https://arxiv.org/abs/2606.09203)
- **What's New**: 이번 연구에서는 ROS 2(로봇 운영 체제)의 비결정적(pub-sub) 실행 모델을 해결하기 위한 새로운 프레임워크를 제시합니다. Lingua Franca (LF)라는 논리적 시간 기반의 조정 언어를 활용하여, 기존 ROS 2 애플리케이션을 수정 없이도 결정적으로 실행할 수 있도록 변환할 수 있는 방법을 탐구합니다. 이를 통해 안전-critical 시스템에서 필요한 예측 가능한 시간(response time) 및 실행 순서를 보장합니다. 이 프레임워크는 커뮤니티에서 사용될 수 있도록 오픈 소스로 공개할 계획입니다.

- **Technical Details**: ROS 2는 DDS(Data Distribution Service)를 기반으로 구축된 모듈화된 로봇 시스템을 지원하는 미들웨어로, 각 노드는 콜백(callback)이라는 메서드를 통해 기능을 구현합니다. 이 시스템에서는 엑세큐터(executor)가 콜백의 실행 순서를 관리하며, 시간 기반의 작업과 이벤트 기반의 작업을 구분합니다. 그러나, ROS 2는 명시적인 동기화 메커니즘이 부족하여 실행 순서가 비결정적(nondeterministic)입니다. 연구 결과로 나타난 결정적 서브셋은 예측 가능한 실행을 가능하게 하고 LF 프로그램으로 자동 변환할 수 있는 정보를 추출합니다.

- **Performance Highlights**: 실험 결과, 기존 ROS 2 시스템과는 달리, LF 제어 ROS 2 시스템은 결정적인 실행 순서와 일관된 지연(latency)을 생성함을 입증했습니다. 구체적으로, 기본 ROS 2에서는 콜백 실행 순서가 실행마다 달라지며, 종단 간(latency) 지연이 다르게 나타나는 반면, LF 제어 시스템은 예측 가능한 실행 순서를 제공합니다. 이러한 결과는 시간 예측 가능성이 요구되는 자동 운전, 물류 및 산업 자동화와 같은 안전-critical 애플리케이션에서 중요한 의미를 지닙니다.



New uploads on arXiv(cs.MA)

### LLM-Mediated Demand Response Coordination in Smart Microgrids (https://arxiv.org/abs/2606.11050)
Comments:
          Accepted for publication in 18th International Conference on Sustainability in Energy and Buildings (SEB-26), to appear in Springer Nature proceedings (KES Smart Innovation Systems and Technologies). The final authenticated version will be available online at Springer

- **What's New**: 이 논문에서는 스마트 마이크로그리드에서의 효과적인 수요 반응 조정 문제를 다룬다. 특히, 프로슈머(prosumer)가 자발적으로 협력하는 조건을 만들기 위한 새로운 접근법으로 하이브리드 결정 아키텍처(hybrid decision architecture)를 제안한다. 이 아키텍처는 게임 이론적(base probability) 분석과 LLM 내러티브(narrative) 평가를 분리하여 실제 협력 행태를 이끌어낸다.

- **Technical Details**: 연구에서 제안된 모델은 여섯 가지 프로슈머 성격 유형(실용주의자, 이상주의자, 회의론자, 순응자, 전략가, 기회주의자)을 기반으로 하여, 이들이 연결된 네트워크 구조에서 협동하는 방식에 대해 연구한다. 하이브리드 아키텍처는 게임 이론적 사고를 결합하여 각 프로슈머의 협동 확률을 계산하고, LLM은 신호를 평가해 협동을 유도한다. 이렇게 만들어진 구조는 RLHF(강화학습 기반 인간 피드백) 문제가 발생하지 않도록 하여 다양한 실험 조건에서 안정적인 협동을 발생시킨다.

- **Performance Highlights**: 구조화된 지침을 통해 수요 축소 협력은 약 33.3%에 달하며, 비구조적인 메시지는 27.0%, 개입이 없는 기준 상황에서는 28.0%의 협력이 이루어졌다. 네트워크 중심 노드를 통해 전달되는 정보가 상대적으로 높은 성과를 보였으며, 이는 메시지 내용과 무관하게 그리드(topology 구조적 특성)가 협동적 행동을 증폭시킬 수 있다는 것을 입증한다. 이 연구는 스마트 시티 에너지 시스템의 수요 반응 조정을 위한 효과적인 설계 원칙을 제시하는 데 중요한 기여를 한다.



### Decentralized Multi-Agent Systems with Shared Contex (https://arxiv.org/abs/2606.10662)
- **What's New**: 이 논문은 다중 에이전트 시스템(MAS)을 기반으로 하는 새로운 접근 방식인 분산 언어 모델(DeLM)을 제안합니다. DeLM은 중앙 집중식 조정 대신, 에이전트들이 병렬로 작업을 수행하도록 하여 통신 및 통합의 병목 현상을 해결합니다. 이를 통해 에이전트 간에 유용한 진행 상황을 공유할 수 있게 하여 복잡한 문제를 효과적으로 처리할 수 있는 가능성을 열어줍니다.

- **Technical Details**: DeLM의 주요 구성 요소는 병렬 에이전트, 공유된 검증된 맥락, 그리고 작업 큐입니다. 각 에이전트는 비동기적으로 작업 큐에서 작업을 가져오고, 공유 맥락에서 쌓인 진행 상황을 읽어 들이며, 로컬 추론을 수행하고 검증된 업데이트를 다시 기록합니다. 이러한 디자인은 에이전트들 사이의 계속적인 의사소통을 효율적으로 줄여주어, 중복된 정보 처리를 방지합니다.

- **Performance Highlights**: DeLM은 SWE-bench Verified에서 77.4%의 Pass@4 성능을 달성하면서 작업당 비용을 약 50% 줄이는 성과를 보였습니다. 또한 LongBench-v2에서 여러 모델군 간의 평균 정확도를 높이며, 가장 강력한 기준선을 5.7% 포인트 개선했습니다. 그러므로 DeLM은 소프트웨어 공학 테스트 시간 스케일링과 긴 맥락 추론에서 모두 뛰어난 효과를 보여줍니다.



### SkillAxe: Sharpening LLM-Authored Agent Skills Through Evaluation-Guided Self-Refinemen (https://arxiv.org/abs/2606.10546)
Comments:
          9 pages, under review

- **What's New**: 이번 연구에서는 SkillAxe라는 전적으로 비지도 학습(un) 프레임워크를 소개합니다. SkillAxe는 LLM이 자신의 스킬을 반복적으로 진단하고 개선할 수 있도록 하며, 스킬의 질을 네 가지 해석 가능한 차원으로 분해합니다. 이 접근법은 라벨이나 테스트 환경 없이도 스킬 개선을 가능하게 해주며, 이를 통해 비전문가도 스킬을 정교하게 다듬을 수 있습니다.

- **Technical Details**: SkillAxe는 스킬의 질을 네 가지 차원: 품질 영향(quality impact), 트리거 정밀도(trigger precision), 지침 준수(instruction compliance), 솔루션 경로 커버리지(solution-path coverage)로 진단합니다. 이 프레임워크는 LLM이 주어진 작업에서 스킬이 작용하지 않았을 때와 있을 때의 행동 차이를 비교하여 유용한 피드백을 생성합니다. 이로 인해 SkillAxe는 자연어 스킬의 특성상 필요했던 환경별 감독 없이도 작동할 수 있습니다.

- **Performance Highlights**: SkillAxe는 SkillsBench에서 LLM이 작성한 스킬보다 28% 더 높은 통과율을 달성하여 인적 작성 스킬과의 격차를 47-67% 줄였습니다. 또한, SpreadsheetBench에서 Excel Copilot이 이전 작업의 경로를 학습하며 스킬을 지속적으로 개선하는 데 성공하여, 22개의 스킬로 통과율을 16%에서 52%로 증가시켰습니다.



### Decoupling Thought from Speech: Knowledge-Grounded Counterfactual Reasoning for Resilient Multi-Agent Argumentation (https://arxiv.org/abs/2606.10475)
Comments:
          Accepted for publication in the Proceedings of the 30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026)

- **What's New**: 이 논문에서는 Multi-agent Debate (MAD) 프레임워크에서 기존의 모델들이 최종 출력의 정확성에 유리하게 최적화되어 있기 때문에 긴 대화에서의 안정성을 고려하지 못하는 문제를 지적합니다. 이를 해결하기 위해, Knowledge-Grounded Counterfactual Reasoning (KG-CFR)라는 이중 단계 아키텍처를 도입하여, 개인적 계획 버퍼와 공적 실행 레이어 간의 엄격한 분리를 구현하여 프로세스의 일관성을 유지합니다. 이 연구는 기존 반응형 시스템의 한계를 극복하고, 시스템의 저항력을 구조적으로 향상시키는 방법을 제시합니다.

- **Technical Details**: KG-CFR 플랫폼은 Dynamic Resource Allocation under Uncertainty (DRAU) 환경에서 성능을 평가합니다. 이 아키텍처는 개인적인 시뮬레이션과 전략적 계획을 하는 버퍼를 분리하여, 외부 요인에 대한 Noise에 의한 논리 손상을 방지합니다. 또한, 새로운 벡터 메트릭스를 통해 논의의 다양성과 계획 실행 일치를 측정하여 운영의 안정성을 높은 방향성과 일관성을 가지고 증명합니다.

- **Performance Highlights**: KG-CFR은 95% 이상 전체 실험에서 judge가 감지한 Critical post-shock degradation(품질 변화 기준, $  -0.20$)을 방지하며, 주장 품질이 0.694에서 0.822로 증가했습니다. 이러한 결과는 KG-CFR이 장기적인 압박 하에서도 품질 손실 없이 시스템 저항력을 강화하는 중요한 요소가 된다는 것을 보여줍니다. 더불어, KG-CFR은 구술에서의 반복(looping)을 줄이며, 이러한 수정된 메트릭들이 운영의 일관성을 어떻게 보장하는지 보여줍니다.



### Failure Modes of Deep Multi-Agent RL in Asynchronous Pricing: Reproducible Triggers, Trace Diagnostics, and a Partial Fix (https://arxiv.org/abs/2606.09884)
- **What's New**: 이번 연구에서는 연속 시간 가격 시장에서 깊은 다중 에이전트 강화 학습(dDeep Multi-Agent Reinforcement Learning)에서 나타나는 두 가지 재현 가능한 실패 방식은 탁월한 카르텔 형성(tacit cartel formation)과 높은 사건 발생률에서의 액터-비평가 불안정성(actor–critic instability)을 연구합니다. 우리는 이 두 가지 실패 모드를 단일 CT-MARL 벤치마크 내에서 구현하고, 동기화된 DDPG 에이전트가 실패 모드 1을 안정적으로 유발한다는 것을 보여주었습니다.

- **Technical Details**: 연구에서는 포아송 시계에 의해 업데이트되는 가격과 관찰 지연 관찰(latency) $b4$, 그리고 최적 내부 로짓 수요를 설정하여 실험을 진행했습니다. 동기화된 DDPG 에이전트들 간의 카르텔 형성 지수(collusion index) $b4$ 를 $0.69  b1 0.11$로 설정하였고, 비동기(asynchrony)와 지연(latency)의 추가가 카르텔 형성을 48% 감소시킨다는 것을 정량적으로 분석했습니다.

- **Performance Highlights**: 제안된 수정안은 부작용이 명확히 문서화되어 있으며, 카르텔 형성 지수가 여전히 초-버르탱(supra-Bertrand) 상태에 있음을 나타냅니다. 또한, 비동기 구성이 관찰 지연에 따라 비단조(non-monotone)적이며, $bb = 5$에서 DDPG 비평가 비대칭(critic divergence)이 발생하여 두 번째 실패 모드를 초래하는 것으로 확인되었습니다. 연구는 에피소드 내 신호 붕괴(signalling collapse)와 충격 후 비회복(non-recovery)을 드러내는 궤적 수준(trace-level)의 진단 정보를 제공합니다.



### Game-Theoretic Multi-Agent Control for Robust Contextual Reasoning in LLMs (https://arxiv.org/abs/2606.10322)
- **What's New**: 이번 논문에서는 Game-Theoretic Secure Model Context Protocol (GT-MCP)을 도입하여 대규모 언어 모델(LLMs)에서의 다중 에이전트 상호작용을 안정화하고, 악의적 공격으로부터 보호하기 위한 새로운 방법을 제시합니다. GT-MCP는 맥락 관리(context management)를 폐쇄 루프 동적 프로세스(closed-loop dynamical process)로 간주하며, 세 개의 이질적인 LLM 에이전트를 조율합니다. 이 기법은 상황의 안정성을 보장하면서도 장기적인 추론 여정을 안정적으로 관리하는 것을 목표로 하고 있습니다.

- **Technical Details**: GT-MCP의 핵심은 컨트롤러가 신뢰 기반의 선택 규칙을 통해 이질적 에이전트들 간의 causal consistency와 semantic agreement을 평가하는 것입니다. 또한, 맥락이 불안정하다고 판단될 경우, 롤백 기반(self-healing) 메커니즘을 통해 검증된 맥락 상태로 복구하여 비지지적 조각들이 전파되는 것을 방지합니다. 이러한 접근 방식은 다중 모델 환경에서의 안전성을 더욱 강화합니다.

- **Performance Highlights**: GT-MCP는 500회의 상호작용에서 99.6%의 맥락 변동을 제한하며, 회복이 필요한 경우는 단 0.4%에 불과했습니다. 선택된 출력은 98% 이상의 안정적인 승률을 유지하며, 각 토큰당 예상 대기 시간은 1.63e-3초로 예측 가능성이 높습니다. 이러한 결과는 GT-MCP가 LLM 상호작용의 추론 경로를 능동적으로 관리해야 한다는 주장을 뒷받침합니다.



### What Spatial Memory Must Store: Occlusion as the Test for Language-Agent Memory (https://arxiv.org/abs/2606.10299)
Comments:
          23 pages, 6 figures

- **What's New**: 이번 연구는 언어-에이전트 메모리 시스템이 지리적 위치를 메모리에 연계하여 기존의 텍스트로는 제공할 수 없는 기하학적 이점을 제공하는 방법을 테스트하고 있습니다. 연구팀은 메모리 회상과 가시성을 분리하여, 기억된 정보가 시각적으로 접근 가능한지 여부가 메모리 저장 방식에 따라 달라진다는 것을 보여주었습니다. 새로운 실험 결과는 기하학적 메모리 구조가 텍스트 기반 인덱스보다 우월하다는 것을 입증했습니다.

- **Technical Details**: 이 연구는 '기억 궁전(memory palace)' 시스템을 사용하여 메모리를 저장할 때, 기하학적 정보를 포함시키는 것이 필수적임을 강조합니다. 기억 배열을 생성할 때 시각적 시스템이 필요하며, 이렇게 저장된 기하학적 정보는 에이전트가 메모리를 읽어오는 과정에서도 필수적입니다. 연구는 기하학적 저장 방식이 단순한 텍스트 기반 접근보다 어떻게 메모리 회상에 도움이 되는지를 보여주는 실험을 통해, 기하학과 격리된 저장 요구사항을 확인했습니다.

- **Performance Highlights**: 연구팀은 기하학적 기반의 메모리 시스템이 기존의 텍스트 기반 시스템에 비해 회상 능력이 월등히 우수함을 입증하였습니다. 피험자들은 특정 실험에서 기하학적 정보를 포함한 시스템이 이전의 방식을 초월하여 성공률을 크게 향상시킴을 보여주었습니다. 이러한 성과는 그래픽 및 텍스트 기반 시스템의 사용성을 향상시키는 데 있어 기하학적 요소가 필수적임을 시사합니다.



### Deployment-Time Memorization in Foundation-Model Agents (https://arxiv.org/abs/2606.10062)
Comments:
          4 pages, ICML MemFM 2026 Workshop

- **What's New**: 이 논문에서는 사용자 맞춤화(User Personalization)에 대한 메모리 설계의 중요성을 강조합니다. 기존 연구는 주로 모델 매개변수에서의 기억 대상(parametric memorization)을 다루었던 반면, 저자들은 메모리 디자인이 개인화 유용성, 정보 유출 위험, 삭제 신뢰도에 미치는 영향을 새롭게 분석합니다. 특히, Persistent agent(지속적인 에이전트)의 메모리 구조가 사용자 정보를 어떻게 저장하고 회수하는지에 대한 구체적인 연구를 진행하였습니다.

- **Technical Details**: 메모리 설계를 Privacy-Utility Frontier(프라이버시-유용성 경계)로 정의하고, 이를 Personalization Recall (PR)과 Adversarial Extraction Rate (AER)을 통해 측정합니다. 저자들은 요약 Aggressiveness, Retrieval Breadth (k), 그리고 Deletion Mode와 같은 메모리 디자인 요소들을 조절하여 다양한 실험을 수행하였습니다. Forgetting Residue Score (FRS)를 도입하여 삭제된 정보가 여전히 복구 가능성을 가진지 측정하는 방법도 제시합니다.

- **Performance Highlights**: LongMemEval의 실험에서 중요한 사실 요약(Key-Fact Summarization)이 Gemma 3 12B에서는 76%, GPT-4o-mini에서는 64%의 정보를 추출하는 것을 줄이는 효과가 있으며, 동시에 거의 모든 개인화 기억을 유지하는데 기여합니다. 그러나, 정보가 압축되면 추가적인 메모리 검색이 더 이상의 유출 복구에 효과가 없다는 점도 발견하였습니다. 결과적으로, 모든 메모리 계층에서 정보를 완전히 삭제하는 것이 불가능한 상황을 보여주며, 이러한 연구 결과는 지속적인 에이전트 메모리를 항상 평가해야 할 필요성을 제기합니다.



### Co-GLANCE: Uncertainty-Aware Active Perception for Heterogeneous Robot Teaming (https://arxiv.org/abs/2606.09919)
Comments:
          Code, videos, and dataset available at this https URL

- **What's New**: Co-GLANCE는 이질적인 로봇 팀을 위한 실시간 인식 및 의사 결정 시스템입니다. 이는 비전-언어 모델(vision-language model)의 의미적 추론 능력을 경량화된 모델로 단순화하여 클라우드 기반 추론을 제거합니다. Co-GLANCE는 성과 기반의 불확실성 정량화를 위해 선택적 어떤 것을 결합하여 분할 및 로봇 할당을 위한 통계적으로 유효한 보장과 함께 이전에 제기된 문제를 해결합니다.

- **Technical Details**: Co-GLANCE는 두 가지 주요 기술을 통합하여 작동합니다. 첫째, 정보가 부족한 영역을 식별하기 위해 문맥 인식을 사용하는 폐쇄 세그먼트 모델을 사용합니다. 둘째, 선택적 어떤 것(selective abstention)과 준거적 예측(conformal prediction)을 결합하여 로봇 할당, 개체 탐지 및 세그멘테이션에 대한 보정된 불확실성 추정을 제공함으로써 적극적인 인식을 유도합니다.

- **Performance Highlights**: Co-GLANCE는 실제 상황에서 클라우드 기반 비전-언어 모델 기반 (baseline)보다 폐쇄 세그멘테이션과 로봇 할당 정확도를 각각 25%와 36% 높이며, 프레임당 추론 지연 시간을 350배 줄였습니다. 또한, 향후 연구를 위한 공중-지상(multimodal air-ground) 데이터셋을 발표합니다.



### TRAPS: Therapeutic Response Analysis via Pathway-informed Stratification (https://arxiv.org/abs/2606.09898)
- **What's New**: 본 논문은 암 치료 계획에서의 여러 임상적 차원에 대한 결정의 필요성을 강조합니다. 기존의 연구들은 개별적으로 발전해온 pathway-informed deep learning 모델을 기반으로 하고 있어 아키텍처 간의 공정한 비교가 불가능했습니다. 그러나 이번 연구에서는 BINN, GraphPath, PATH 세 가지 생물학적으로 고안된 아키텍처를 사용하여 통합된 벤치마크를 제시합니다.

- **Technical Details**: 연구에서는 Reactome pathway activity scores를 활용하여 2,622명의 환자를 포함한 암 생물학적 그룹 및 임상 데이터를 분석하였습니다. 각 모델은 세 가지 임상 결과에 대해 동등한 조건에서 훈련되었으며, 이를 통해 pathway 구조가 깊이 학습에서 치료 및 생존 예측 문제로 통합될 수 있음을 증명하였습니다. 연구는 각 아키텍처의 예측 성능이 서로 다르며, BINN은 생존 예측에 가장 신뢰할 수 있는 성능을 보였고, GraphPath는 전립선 표적 분자 치료 예측에서 AUROC 0.92의 최고 점수를 기록했습니다.

- **Performance Highlights**: PATH 모델은 전반적으로 표적 분자 치료 예측에서 최상의 성능을 보였으며, 특히 GraphPath는 좁은 대상 드라이버 프로그램과 일치할 때 뛰어난 분별력을 보여주었습니다. 그러나 방사선 치료에 대한 예측 성능은 만족스럽지 않았고, 이는 암 유전자 발현 데이터를 통해서는 포착할 수 없는 임상 변수가 영향을 미쳤기 때문입니다. 따라서 연구 결과는 각 암 유형에 따라 서로 다른 모델의 성능 차이가 있음을 시사합니다.



### LLM-Based Code Documentation Generation and Multi-Judge Evaluation (https://arxiv.org/abs/2606.09852)
Comments:
          ICAHS, \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이 논문에서는 AI를 활용한 코드 문서화 자동 생성 시스템을 제시하며, 이를 위해 8개의 최첨단 대형 언어 모델(LLMs)인 GPT, Gemini, Qwen, LLaMA 변형을 사용합니다. PocketFlow 오케스트레이션 프레임워크에 기반하여 모듈형 파이프라인과 고급 프롬프트 기법을 적용하여 문서화 품질을 향상시키고 수동 노력을 줄이는 것을 목표로 합니다.

- **Technical Details**: 이 시스템은 프로세싱 단계를 통해 점진적인 정제 모델을 구현하여 복잡한 구현 세부정보와 접근 가능한 학습 자료 간의 격차를 해소합니다. 각 프로세싱 노드는 특정 기능을 수행하며, 문서 생성 작업을 관리하기 위한 복잡한 아키텍처의 주축을 형성합니다. 프롬프트 엔지니어링은 LLM의 행동을 안내하고 제어하는 데 주요한 역할을 하며, 각 노드의 특정 기능과 의미적 역할에 맞춘 전문 프롬프트 템플릿을 사용합니다.

- **Performance Highlights**: PyMedPhys라는 오픈 소스 의료 물리학 라이브러리에서 수행된 실험은 상위 모델과 하위 모델 간의 42% 성능 차이를 보여주었습니다. 또한, 이 시스템은 다양한 모델 출력을 결합하고 최적화된 프롬프트 및 엄격한 평가를 통해 문서화 품질을 향상시킬 수 있음을 입증하였습니다. 이 접근 방식은 안전이 중요한 의료 소프트웨어 분야에서의 수동 작업을 크게 줄이는 데 기여할 수 있습니다.



### Envisioning Sensemaking in Multi-Human, Multi-Agent Collaborative Knowledge Work (https://arxiv.org/abs/2606.09840)
Comments:
          This is the Author's Accepted Manuscript version of the article: Guan, Z., \& Rieh, S. Y. (2026). Envisioning Sensemaking in Multi-Human, Multi-Agent Collaborative Knowledge Work. Accepted for publication in \textit{Sensemaking @ CHI 2026}

- **What's New**: 이 논문은 Generative AI(GenAI)가 협업 지식 작업에서의 sensemaking(의미 구성) 방식을 어떻게 변화시키는지를 조사하고 있습니다. GenAI 시스템은 요약(summarization), 합성(synthesis), 주제 그룹화(thematic grouping)와 같은 해석 기능을 수행함으로써 지식 근로자가 전통적으로 수행했던 역할을 대체하고 있으며, 이러한 변화는 팀 다이내믹과 신뢰 문제를 복잡하게 만듭니다.

- **Technical Details**: 연구는 협업에서의 multi-human, multi-agent collaborative sensemaking(협동적 의미 구성) 지원을 위한 다섯 가지 설계 원칙을 제안합니다. 이 원칙들은 동적 다층 정보 표현(dynamic multi-layer information representations), 이해의 격차 확인 및 매개(active identification and bridging of gaps in understanding), 정보에 대한 비판적 참여(critical engagement with information), 검증 가능성(verifiability), 그리고 책임(accountability)입니다.

- **Performance Highlights**: 이 프레임워크는 지식 근로자와 AI 에이전트가 공동으로 증거를 수집하고, 개념화하며, 가설을 세우고, 협력 목표를 추구하는 동적 공유 지식 작업 공간을 제안합니다. 이로써 각 개별 해석과 공동 해석의 진화를 추적할 수 있으며, 이는 협업 과정에서의 신뢰성을 높이고, 의미 구성의 과정을 더욱 투명하게 만듭니다.




### Prometheus 2: An Open Source Language Model Specialized in Evaluating  Other Language Models (https://arxiv.org/abs/2405.01535)
Comments: Work in Progress

- **What's New**: 이 연구에서는 평가를 전문으로 하는 오픈 소스(language models) LMs의 개발 필요성을 강조합니다. 특히, Prometheus 2는 기존의 오픈 소스 평가 모델의 두 가지 주요 문제점을 해결하기 위해 개발되었습니다: 1) 인간이 부여하는 점수와 크게 다른 점수를 부여하고, 2) 직접 평가(direct assessment)와 쌍으로 순위를 매기는(pairwise ranking) 평가의 두 가지 형태를 모두 수행하는 유연성이 부족합니다. 또한 사용자 정의 평가 기준에 따라 평가할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: Prometheus 2는 인간과 GPT-4 판단을 밀접하게 반영하여 설계되었으며, 직접 평가 및 쌍으로 순위 매기기 포맷 둘 다를 처리할 수 있습니다. 이 모델은 일반적인 특성인 도움됨(helpfulness)과 해로움(harmlessness)에 초점을 맞추기보다는 사용자 정의된 평가 기준에 기초하여 평가할 수 있습니다.

- **Performance Highlights**: Prometheus 2는 네 가지 직접 평가 벤치마크와 네 가지 쌍으로 순위 매김 벤치마크에서 가장 높은 상관성 및 인간 및 소유권(proprietary) LM 판사와의 합의를 달성했습니다. 이는 모든 테스트된 오픈 평가 LMs 중 최고의 성능을 보여줍니다. 모델, 코드, 및 데이터는 모두 공개적으로 접근 가능합니다.



### FLAME: Factuality-Aware Alignment for Large Language Models (https://arxiv.org/abs/2405.01525)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 정렬(alignment) 과정을 개선하여 더 사실적인(factual) 반응을 유도하는 방법을 제시합니다. 기존의 정렬 과정은 사실 정확성(factual accuracy)을 향상시키지 못하고 거짓 사실(false facts)을 더 많이 생성하는 문제(hallucination)가 있었습니다.

- **Technical Details**: 저자들은 감독된 미세조정(Supervised Fine-Tuning, SFT)과 강화 학습(Reinforcement Learning, RL) 단계에서 환각(hallucination)을 초래하는 요인들을 식별하였습니다. 특히, 새로운 지식이나 낯선 텍스트(texts)에 대한 학습이 환각을 촉진할 수 있다고 합니다. SFT에서는 사람이 라벨링한 데이터를 학습하므로, LLM에게는 새로울 수 있어 덜 사실적이 되며, 표준 RL의 보상 함수(reward functions) 역시 다양한 지시사항에 대해 도움이 되는 반응을 제공하도록 유도하며, 자주 더 길고 상세한 응답을 선호하도록 만들어 환각을 장려하는 경향이 있습니다.

- **Performance Highlights**: 제안된 사실성 인식 정렬(factuality-aware alignment) 프로세스는 사실적인 반응을 제공하면서 지시사항(instruction-following)을 따르는 능력을 유지하도록 LLM을 안내합니다. 실험 결과, 이 방법이 LLM의 반응에서 사실 정확성을 향상시키는 데 효과적임을 보여줍니다.



### D2PO: Discriminator-Guided DPO with Response Evaluation Models (https://arxiv.org/abs/2405.01511)
Comments: 20 pages, 12 figures

- **What's New**: 본 논문에서는 언어 모델(Language Models)의 정렬(Alignment)을 위한 새로운 접근법인 D2PO(Discriminator-guided Direct Preference Optimization)를 제안합니다. 기존의 DPO 방식이나 다른 이종(Online Preference Optimization) 방법들과는 다르게, 실시간으로 선호 데이터를 수집하며 교육하는 온라인 환경에서 평가 모델을 이용하여 정책 모델(Policy Model)의 학습 데이터를 확장하는 방식입니다. 주요 개선점은 실버 라벨링(Silver Labeling)을 통해 효율적으로 데이터를 확보하고, 높은 품질의 출력을 달성하는 것입니다.

- **Technical Details**: D2PO는 선호 레이블(Preference Labels)을 수집하여 판별 모델(Discriminative Response Evaluation Model)을 훈련하는 과정과, 이 판별자(Discriminator)를 사용하여 정책 모델에서 더 많은 출력을 실버 라벨로 분류하여 정책 훈련 데이터를 확장하는 두 단계로 진행됩니다. 이 방식은 DPO 목표를 사용하여 정책을 훈련하며, 기존의 PPO(Proximal Policy Optimization) 방식보다 우수한 결과를 보여주었습니다. 별도의 판별자를 유지하여 인간이 레이블링한 선호 데이터에서 학습하고, 정책 모델은 노이즈가 많은 온정책 데이터(On-policy Data)에서 학습할 수 있습니다.

- **Performance Highlights**: D2PO는 동일한 데이터 예산을 사용하고도 DPO나 기본 PPO보다 빠르게 높은 보상을 달성했습니다. 또한, 실버 라벨링이 가장 효과적일 때는 DPO로 정책을 훈련할 때이며, 이는 전통적인 PPO를 능가하는 결과를 보여줍니다. UltraFeedback 및 간단한 텍스트 생성 작업에 대한 벤치마크에서도 우리의 접근 방식이 기존 방법들보다 높은 품질의 출력을 생성하는 것을 확인했습니다.



### Analyzing the Role of Semantic Representations in the Era of Large  Language Models (https://arxiv.org/abs/2405.01502)
Comments: NAACL 2024

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 시대에 의미 표현(Semantic Representations)의 역할에 대해 조사하였다. 특히 Abstract Meaning Representation (AMR)이 다양한 자연 언어 처리(NLP) 작업에 미치는 영향을 조사하고 AMR 기반의 사고 체인(prompting) 방법, 'AMRCoT'을 제안하였다.

- **Technical Details**: AMRCoT는 AMR을 활용하여 LLMs의 출력을 도출하는데 도움을 주는 체인-오브-쏘트 방법이다. 이는 전통적인 NLP 모델이 사용하던 단순한 연속적 생성 문제로의 전환이 아닌, 더 풍부한 시맨틱 정보를 포함시키려는 시도이다. 연구팀은 5가지 NLP 작업에 걸쳐 AMR의 효용성을 실험적으로 분석하였다.

- **Performance Highlights**: 결과적으로, AMRCoT는 일반적으로 성능에 해가 되는 경우가 많았다. 다중 단어 표현, 명명된 개체, 그리고 최종 추론 단계에서 LLM이 AMR에 기반한 추론을 예측에 연결할 때 문제가 발생하는 경향이 있었다. 따라서 연구팀은 이러한 영역에 대한 미래 연구의 초점을 맞추는 것을 권장한다.



### Controllable Text Generation in the Instruction-Tuning Era (https://arxiv.org/abs/2405.01490)
- **What's New**: 본 연구는 기존의 딥러닝 (Deep Learning) 모델에 명령어 준비파라다임(Instruction-tuning and prompting paradigm)을 적용하여 텍스트 생성을 제어하는 새로운 접근법을 제시합니다. 이를 위해 다양한 제어 가능한 텍스트 생성 작업을 포함하는 'ConGenBench'라는 테스트베드를 구축하고, 이를 사용하여 9가지 기반라인 및 방법론을 평가하였습니다.

- **Technical Details**: 연구팀은 '(Instruction-tuned Language Models, 지시 학습된 언어 모델)'의 성능을 벤치마킹하기 위해 '(prompting-based approaches, 프롬프트 기반 접근법)'과 '(controllable text generation methods, 제어 가능한 텍스트 생성 방법)'을 비교 분석했습니다. 결과적으로, 프롬프트 기반 접근법이 대부분의 데이터셋 및 작업에서 더 우수한 성능을 보였으며, 특히 '(stylistic tasks, 스타일 작업)'에서는 인간의 성능과 동등한 결과를 보였습니다. 또한, 연구팀은 언어 모델과 작업 데이터셋만을 사용하여 제약 조건 데이터셋을 자동 생성하는 알고리즘을 개발하여 연구 범위를 확장하는 데 기여했습니다.

- **Performance Highlights**: 프롬프트 기반 접근법은 인간의 성능에 필적하는 결과를 스타일 제어 작업에서 보여줬으며, 구조 및 어휘적 제약에 대해서는 아직 개선의 여지가 있음을 발견했습니다. 이러한 발견은 '(Instruction-tuned LLMs, 지시 학습된 대규모 언어 모델)'에 대한 추가적인 연구를 필요로 합니다. 'ConGenBench'는 다양한 제약 조건을 포함하는 17개의 데이터셋을 제공함으로써, 제어 가능한 텍스트 생성 연구를 위한 기반을 마련하고 있습니다.



### NeMo-Aligner: Scalable Toolkit for Efficient Model Alignmen (https://arxiv.org/abs/2405.01481)
Comments: 13 pages, 4 figures

- **What's New**: NeMo-Aligner는 대규모 언어 모델(Large Language Models, LLMs)을 인간의 가치와 선호도에 맞춰 조정하는 도구입니다. 이 도구는 수백 개의 GPU를 사용하여 훈련할 수 있도록 설계되어 있으며, 대규모 모델의 효율적인 조정을 가능하게 합니다.

- **Technical Details**: NeMo-Aligner는 여러 주요 모델 조정 패러다임을 지원하는 최적화된 구현을 제공합니다. Reinforcement Learning from Human Feedback (RLHF), Direct Preference Optimization (DPO), SteerLM, Self-Play Fine-Tuning (SPIN) 등의 방법론을 포함합니다. 또한, Parameter Efficient Fine-Tuning (PEFT) 설정에서 대부분의 조정 기술을 실행할 수 있습니다. NeMo-Aligner는 확장성을 고려하여 설계되었기 때문에 다른 조정 기술을 최소한의 노력으로 지원할 수 있습니다.

- **Performance Highlights**: 이 도구는 수백 개의 GPU를 활용하여 대규모 모델을 효율적으로 훈련하고 조정할 수 있도록 해주며, Apache 2.0 라이선스로 오픈소스화되어 있어 커뮤니티의 기여를 촉진합니다.



### V-FLUTE: Visual Figurative Language Understanding with Textual  Explanations (https://arxiv.org/abs/2405.01474)
- **What's New**: 이 연구는 비유적 언어 현상을 처리하기 위한 새로운 시각적-언어적 모델(VLMs)의 능력을 평가하는 V-FLUTE 데이터셋을 제안합니다. 이 데이터셋은 메타포(metaphors), 시밀리(similes), 아이디옴(idioms), 풍자(sarcasm), 그리고 유머(humor) 같은 다양한 다중모드 비유 현상을 포함합니다. 비유적 의미는 이미지, 캡션 또는 둘 다에 나타날 수 있으며, 모델은 이미지(전제)가 주장(가설)을 포함하는지 또는 모순하는지를 판단하고 예측된 레이블을 텍스트 설명으로 정당화해야 합니다.

- **Technical Details**: V-FLUTE 작업은 이미지(전제)와 텍스트 주장(가설)이 포함된 인스턴스를 사용하여 설명 가능한 시각적 의미연결(explainable visual entailment) 문제로 정의됩니다. 각 인스턴스는 이미지와 텍스트 주장이 붙어 있으며, 이 주장이 이미지에 의해 뒷받침되거나 모순되는지 판단해야 합니다. 데이터셋 구축은 인간-AI 협업 프레임워크를 사용하여 수행되었으며, 이를 통해 고품질의 설명가능한 데이터셋이 생성되었습니다.

- **Performance Highlights**: V-FLUTE는 다양한 모델의 시각적 및 언어적 비유 이해 능력을 평가하고 있으며, 자동 및 인간 평가를 포함한 여러 평가 방법을 통해 현재 VLMs의 성능을 체계적으로 검토하고 있습니다. 또한, 비유적 현상을 이해하는 데 있어 다양한 모델 간의 오류 유형을 분석하는 자세한 인간 평가가 수행되었습니다.



### WildChat: 1M ChatGPT Interaction Logs in the Wild (https://arxiv.org/abs/2405.01470)
Comments: accepted by ICLR 2024

- **What's New**: 최근에, GPT-4와 ChatGPT와 같은 챗봇이 수백만 명의 사용자를 대상으로 서비스를 제공하고 있습니다. 하지만 이러한 도구들이 실제 사용자들에 의해 어떻게 사용되는지 보여주는 공개 데이터셋이 부족한 실정입니다. 이러한 격차를 메우기 위해, 우리는 온라인 사용자들에게 ChatGPT 접근을 무료로 제공하고 그 대가로 사용자들의 동의를 얻어 그들의 채팅 기록과 요청 헤더(request headers)를 익명으로 수집했습니다. 결과적으로, 1백만 개의 사용자-ChatGPT 대화로 구성된 'WildChat'이라는 데이터 코퍼스를 만들었습니다.

- **Technical Details**: WildChat 코퍼스는 250만 회 이상의 상호 작용 턴(interaction turns)을 포함하고 있습니다. 이 데이터셋은 다양한 사용자 프롬프트(user prompts), 가장 많은 언어 수, 그리고 연구자들이 연구할 수 있는 잠재적으로 유해한 사용 사례(toxic use-cases)의 가장 풍부한 다양성을 제공합니다. 또한, 상태(state), 국가(country), 해시된 IP 주소(hashed IP addresses) 등의 인구 통계 데이터와 요청 헤더를 포함하여 데이터셋을 풍부하게 만들었습니다. 이러한 추가적 데이터는 사용자의 지리적 지역 및 시간적 차이에 따른 사용자 행동의 더 상세한 분석을 가능하게 합니다.

- **Performance Highlights**: WildChat 데이터셋은 그것이 포함하는 사용 사례의 넓은 범위로 인해, 지시를 따르는 모델(instruction-following models)을 미세 조정하는 데 있어서의 잠재적인 유틸리티를 보여줬습니다. 이 데이터셋은 AI2의 ImpACT 라이센스 하에 공개되었으며 연구자들이 더 정밀한 인공지능 애플리케이션을 개발하는 데 크게 기여할 것입니다.



### UQA: Corpus for Urdu Question Answering (https://arxiv.org/abs/2405.01458)
- **What's New**: 이 논문에서는 우르두어(Urdu)를 위한 새로운 데이터셋인 UQA를 소개합니다. UQA는 스탠포드 질문 답변 데이터셋(SQuAD2.0)을 번역하여 생성한 것으로, 7000만 명 이상의 모국어 사용자가 있는 저자원(low-resource) 언어인 우르두어를 위한 질문 응답 및 텍스트 이해를 위한 도구입니다.

- **Technical Details**: UQA는 EATS(Enclose to Anchor, Translate, Seek) 기술을 사용하여 SQuAD2.0을 번역함으로써 답변이 포함된 문맥을 유지하도록 하였습니다. 번역 모델 후보로는 구글 번역기(Google Translator)와 Seamless M4T를 평가하였으며, 다양한 다국어 QA 모델들, 예를 들면 mBERT, XLM-RoBERTa, mT5를 UQA에서 벤치마킹하였습니다.

- **Performance Highlights**: 특히 XLM-RoBERTa-XL 모델은 F1 점수가 85.99, EM (exact match) 점수가 74.56을 기록하였습니다. 이 결과들은 UQA가 우르두어 및 다른 저자원 언어들을 위한 다국어 NLP 시스템 개발 및 테스트에 유용한 자원임을 보여줍니다. 또한, EATS 방법론이 다른 언어 및 도메인에 대한 고품질 데이터셋 생성에 효과적임을 입증합니다.



### Unsupervised Flow Discovery from Task-oriented Dialogues (https://arxiv.org/abs/2405.01403)
Comments: 12 pages, 4 figures

- **What's New**: TOD(Task-Oriented Dialogue, 작업 지향 대화) 시스템 개발 시 대화 흐름 설계는 중요하지만 시간이 많이 소요되는 작업입니다. 이 연구는 대화 이력에서 대화 흐름을 무인 감독(unsupervised) 방식으로 발견하는 새로운 접근 방식을 제안합니다. 이 방법은 기존 대화 이력이 있는 모든 도메인에 적용 가능하여, 다양한 분야에서의 TOD 시스템 구현을 용이하게 할 수 있습니다.

- **Technical Details**: 연구에서는 발언(utterances)을 벡터 공간(vector space)에 표현하고, 이를 의미적 유사성(semantic similarity)에 따라 군집화합니다. 이 군집들은 대화 상태(dialogue states)로 볼 수 있으며, 이를 전이 그래프(transition graph)의 정점으로 사용하여 시각적으로 흐름을 나타냅니다. MultiWOZ라는 공개 TOD 데이터셋에서 발견된 대화 흐름의 구체적인 예를 제시하고, 이를 자동으로 평가하는 측정 기준(metric)도 소개합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 작업 지향 대화에서 의미 있고 중요한 대화 흐름을 추출할 수 있는 잠재력을 보여주었습니다. 또한, 언더라잉 대화(underlying conversations)의 중요성과 관련성을 강조하며, 대화의 흐름을 자동으로 평가할 수 있는 새로운 방법을 제시합니다.



### Verification and Refinement of Natural Language Explanations through  LLM-Symbolic Theorem Proving (https://arxiv.org/abs/2405.01379)
- **What's New**: 이 논문은 자연어 설명의 타당성을 평가하는 기존 방법의 한계를 극복하기 위해 대규모 언어 모델(Large Language Models, LLMs)과 정리 증명 시스템(Theorem Provers, TPs)의 통합을 통해 자연어 설명을 검증하고 개선하는 새로운 신경 기호(neuro-symbolic) 프레임워크인 '설명 개선자(Explanation-Refiner)'를 소개합니다. 이 모델은 NLI(Natural Language Inference)를 위한 설명을 생성하고 형식화(formalize)하는데 도움을 줍니다.

- **Technical Details**: Explanation-Refiner는 LLM을 사용하여 설명적 문장을 생성하고 이론적 추론 전략을 제안합니다. 그리고 TP는 설명의 논리적 타당성에 대한 공식적인 보장을 제공하며 후속 개선을 위한 피드백을 생성합니다. 본 연구는 LLM의 설명적 추론, 자동 형식화(autoformalisation), 그리고 오류 수정 메커니즘을 평가하는 데 공동으로 사용됩니다.

- **Performance Highlights**: Explanation-Refiner는 다양한 도메인에서 복잡도가 변화하는 인간이 주석을 단 설명의 품질을 자동으로 향상시키는 능력을 시연합니다.



### Topics in the Study of the Pragmatic Functions of Phonetic Reduction in  Dialog (https://arxiv.org/abs/2405.01376)
- **What's New**: 본 기술 보고서는 음성 대화에서 발음 정밀도가 감소하는 현상에 대해 보다 깊이 연구하고 있으며, 인식된 감소(Perceived Reduction)를 주석 처리하는 것에 대한 교훈을 제공합니다. 특히 대화에서 감소와 관련된 새로운 음향적 특성으로 높은 피치(High Pitch), 넓은 피치 범위(Wide Pitch Range), 그리고 강도(Intensity)가 포함되는 것으로 나타났습니다. 또한 영어와 스페인어 대화에서 발음 감소를 예측하는 기준 모델(Baseline Model)을 제시하며, 이는 간단한 음향/음성학적 특성(Acoustic/Prosodic Features)을 사용하여 인간의 인식과의 상관관계를 나타냈습니다.

- **Technical Details**: 이 보고서는 음향/음성학적 피쳐를 이용한 대화에서의 발음 감소 예측 모델을 개발하고 정확도(accuracy)를 평가하는 데 중점을 두고 있습니다. 이는 미드레벨 프로소딕 피쳐스 툴킷(Midlevel Prosodic Features Toolkit)이 발음 감소 요소를 포함하도록 확장을 시도한 결과로, 발음 감소를 기존의 다양한 음성 및 음향 측정과 연관짓는 새로운 시도입니다. 발음 감소가 대화의 어떤 프래그마틱 기능(Pragmatic Functions)과 연관될 수 있는지에 대한 추가 예시도 제공됩니다.

- **Performance Highlights**: 기준 모델은 영어에서 인간의 인식과 0.24, 스페인어에서 0.17의 상관 관계를 달성하여 실험 초기단계임에도 불구하고 어느 정도의 유효성을 입증하였습니다. 이러한 성과는 발음 감소가 대화에서 어떻게 작용하는지를 이해하는 데 중요한 기여를 하며, 향후 발음 감소를 자동으로 감지하고 분석할 수 있는 도구 개발의 기반을 마련합니다.



### GAIA: A General AI Assistant for Intelligent Accelerator Operations (https://arxiv.org/abs/2405.01359)
- **What's New**: 본 논문에서는 입자 가속기와 같은 대형 기계를 운용하기 위해 대규모 언어 모델(Large Language Model, LLM)과 고급 기계 제어 시스템 프레임워크를 결합한 새로운 다중-전문가 검색 확대 생성(Multi-Expert Retrieval Augmented Generation, RAG) 시스템을 개발하였다. 이 시스템은 운영자가 전문 지식을 검색하고, 필요한 경우 기계와 직접 상호 작용하며, 고급 제어 시스템 스크립트를 작성할 수 있도록 지원한다.

- **Technical Details**: 이 시스템은 ReAct (Reasoning and Action Prompting) 패러다임을 사용하여, 오픈-웨이트 LLM을 고급 기계 제어 시스템 및 전자 로그북, 기계 설계 문서와 같은 다른 도구들과 연결한다. 특히, Python (파이썬) 스크립트 생성을 지원하여, 입력 프롬프트로 정의된 작업을 수행할 수 있다. 또한, LangChain과 같은 도구를 사용하여 운영자가 기계의 특정 부분을 제어하거나 정보를 검색하라는 결정을 내릴 때 이를 지원한다.

- **Performance Highlights**: 이 시스템은 운영자가 입자 가속기와 같은 복잡한 기계를 보다 효과적으로 운용할 수 있도록 도와줌으로써, 새로운 운영자뿐만 아니라 경험 많은 운영자들의 작업을 간소화하고 가속화할 수 있다. 예를 들어, 시스템은 특정 작업을 위한 코드를 작성하거나 다차원 절차를 조율하는 데 필요한 지식을 제공할 수 있다. 이러한 기능은 실험 캠페인을 효율적으로 수행하고 안전성을 증가시키면서, 운용자가 다루어야 할 복잡성을 줄여준다.



### The Power of Question Translation Training in Multilingual Reasoning:  Broadened Scope and Deepened Insights (https://arxiv.org/abs/2405.01345)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 영어 및 비영어 언어 성능 간의 격차를 좁히는 새로운 접근 방식을 제시합니다. 이전 연구들이 번역된 교육 데이터를 사용하여 이 격차를 해소하려 한 반면, 이번 연구는 최소한의 번역 사용과 함께 모델의 영어 전문성을 활용하여 다국어 성능을 개선하는 '질문 정렬(question alignment)' 접근 방식을 제안합니다. 특히, 이 방법은 실행 가능한 코드(executable code)와 상식(common sense)을 포함한 다양한 추론 시나리오에서의 적용 가능성을 검토합니다.

- **Technical Details**: 연구자들은 대용량 LLM에 '질문 정련' 접근 방식을 효율적으로 적용하기 위해 '프록시 튜닝(proxy-tuning)'을 탐구합니다. 이는 매개 변수를 직접 조정하는 대신, 선택된 프록시 모델을 사용하여 LLM의 학습을 가이드하는 방식입니다. 이 접근 방식은 특히 LLaMA2와 같이 매개변수가 70B에 이르는 매우 큰 모델에 적용될 때, 계산 비용을 크게 절감하면서도 98%의 성능을 유지할 수 있음을 보여줍니다.

- **Performance Highlights**: 질문 정련 방식을 사용하여 LLaMA2 모델들에 적용한 결과, mGSM 벤치마크에서 평균 정확도가 12.2% 향상되었습니다. 이는 다국어 벤치마크(mGSM, mSVAMP, xCSQA)에서도 유사한 결과를 보이며, 다양한 추론 작업 및 모델 크기에 걸쳐 이 방법의 효과를 입증합니다. 특히, 이 방식은 LLM이 다국어 문제에 대해 영어와 유사한 추론 과정을 활용하도록 하는데 큰 영향을 미칩니다.



### The Effectiveness of LLMs as Annotators: A Comparative Overview and  Empirical Analysis of Direct Representation (https://arxiv.org/abs/2405.01299)
Comments: LREC-COLING NLPerspectives workshop

- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)이 데이터 주석 달기(data annotation)에서 어떻게 활용될 수 있는지에 대한 최근 연구들을 비교하고 분석합니다. 특히 GPT와 같은 주요 모델들이 인간 주석자와 얼마나 일치하는지, 그리고 주관적 데이터 집합에 대한 의견 분포의 일치도를 측정하는 것으로, 이는 기존의 대표성(representativeness)에 중점을 둔 연구들과는 다른 접근 방식을 제시합니다.

- **Technical Details**: LLMs, 특히 GPT 모델을 활용하여 데이터를 주석 처리하는 능력이 연구되었습니다. 이 연구에서는 zero-shot 및 few-shot 학습 접근 방식이 모두 탐색되었으며, 주로 분류(classification) 작업을 중심으로 진행되었습니다. 데이터 주석 과정에서 인간 주석자의 의견 분포를 얼마나 잘 포착(capture)하고 반영(reflect)하는지가 주요 평가 기준으로 사용되었습니다. 또한 다양한 설정에서의 모델의 일관성(consistency) 및 정확성(accuracy)을 평가하기 위해 temperature 설정 조정이 연구되었습니다.

- **Performance Highlights**: 연구는 LLMs가 영어(English) 데이터에 대해 강력한 성능을 보이지만, 비영어(non-English) 데이터에 대해서는 상대적으로 낮은 성능을 보임을 밝혔습니다. 일관된 결과를 보장하기 위해 낮은 temperature 설정을 사용하는 것이 권장되었습니다. 비록 LLMs가 인간 주석자와의 의견 분포에서 완벽한 일치를 보이지는 않지만, 주관적 질문에 대한 반응에서 다양한 관점을 고려하는 방향으로 나아가는 연구가 일부 있음을 강조했습니다.



### Low-resource speech recognition and dialect identification of Irish in a  multi-task framework (https://arxiv.org/abs/2405.01293)
Comments: 7 pages. Accepted to Odyssey 2024 - The Speaker and Language Recognition Workshop

- **What's New**: 이 연구는 아일랜드(Irish) 저자원(low-resource) 음성인식(ASR) 및 방언 식별(DID)을 위해 Hybrid CTC/Attention 인코더-디코더 모델과 InterCTC를 사용한 훈련을 탐구합니다. 이 접근 방식은 기존의 ASR(TDNN-HMM) 및 DID(ECAPA-TDNN) 모델과 비교하여 개선된 결과를 보여 줍니다. 연구는 Conformer 및 E-branchformer 인코더 아키텍처를 통해 최적의 InterCTC 환경을 설정하고, 다중 작업 세부 조정 및 언어 모델(LM) shallow fusion을 적용합니다.

- **Technical Details**: Hybrid CTC/Attention 인코더-디코더 모델은 DID의 정확도를 향상시키고 TDNN-HMM 모델에 근접한 WER 성능을 제공합니다. 이 연구는 InterCTC를 통해 ASR과 DID의 다중 작업 훈련 환경을 탐구하고, 다양한 인코더 계층에 보조 작업을 배치하는 방법을 시스템적으로 변경하여 최적의 설정을 찾습니다. 또한, Transformer 언어 모델과의 shallow fusion을 통한 성능 향상이 테스트되었습니다.

- **Performance Highlights**: ECAPA-TDNN 기준 대비 방언 식별(Dialect Identification) 정확도가 10.8% 상대적으로 향상되었으며, WER(Word Error Rate) 성능은 TDNN-HMM 모델에 근접합니다. 이 다중 작업 접근법은 아일랜드 저자원 ASR 및 DID에서 유망한 전략으로 부상합니다.



### Reinforcement Learning for Edit-Based Non-Autoregressive Neural Machine  Translation (https://arxiv.org/abs/2405.01280)
- **What's New**: 이 연구에서는 비자동 회귀(non-autoregressive, NAR) 언어모델을 위한 새로운 훈련 기법으로 강화 학습(reinforcement learning, RL)을 적용하였습니다. 특히, 편집 기반 NAR 모델인 레벤슈타인 변환기(Levenshtein Transformer, LevT)에 적용하여 자체 생성된 데이터로 모델의 성능을 향상시켰습니다. 연구는 두 가지 RL 접근 방식, 즉 단계적 보상 최대화(stepwise reward maximization)와 에피소드 보상 최대화(episodic reward maximization)를 탐구하고 이들의 장단점을 분석하였습니다.

- **Technical Details**: 레벤슈타인 변환기는 토큰을 삭제하고, 자리 표시자를 삽입하며, 자리 표시자를 새 토큰으로 교체하는 세 가지 편집 작업을 기반으로 합니다. 강화 학습을 사용하여 이러한 편집 작업에 대한 보상을 계산하고, 정책 그라디언트(policy gradient)를 추정합니다. 또한, 소프트맥스 샘플링(softmax sampling)에서 온도 설정(temperature setting)의 영향을 실험적으로 조사하여, 탐험(exploitation)과 탐사(exploration) 사이의 균형을 최적화하는 적절한 온도를 찾는 것의 중요성을 확인하였습니다.

- **Performance Highlights**: 강화 학습을 적용한 결과, 레벤슈타인 변환기의 성능이 크게 향상되었습니다. 업데이트된 모델은 네트워크 추론 시간을 단축하면서도 타깃 단어 사이의 의존성을 더 정확히 잡아내는 능력이 향상되었습니다. 이는 NAR 모델을 실시간 번역 어플리케이션에 더 적합하게 만들어, 전체적으로 모델의 유용성을 증가시켰습니다.



### Prompt engineering paradigms for medical applications: scoping review  and recommendations for better practices (https://arxiv.org/abs/2405.01249)
- **What's New**: 의료 분야에서 대규모 언어 모델(Large Language Models, LLMs)의 잠재력을 활용하기 위해 프롬프트 엔지니어링의 중요성이 강조되고 있습니다. 특히 의료 분야의 전문 용어와 표현을 사용하는 경우 이 기술의 효율성을 탐구하는 것은 중요한 연구 주제입니다. 이 연구는 2022년부터 2024년까지의 의학 분야에서의 프롬프트 엔지니어링을 적용한 114건의 최근 연구를 검토합니다.

- **Technical Details**: 이 논문에서는 프롬프트 학습(Prompt Learning, PL), 프롬프트 튜닝(Prompt Tuning, PT), 프롬프트 디자인(Prompt Design, PD)과 같은 다양한 프롬프트 엔지니어링 기술을 다룹니다. 프롬프트 디자인이 가장 널리 연구된 방법으로, 총 78편의 논문에서 다루어졌습니다. 또한, ChatGPT는 의료 분야에서 민감한 임상 데이터 처리를 위해 가장 자주 사용되는 LLM으로, 7편의 논문에서 사용되었습니다.

- **Performance Highlights**: 이 연구에서는 Chain-of-Thought가 가장 흔한 프롬프트 엔지니어링 기술로 부각되었습니다. PL과 PT 관련 논문들은 일반적으로 프롬프트 기반 접근 방식을 평가할 수 있는 기준을 제공하지만, PD 연구의 64%는 프롬프트와 관련 없는 기준이 부족합니다. 또한, 이 연구는 기존 작업을 요약하는 표와 그림을 제공하고, 미래 연구 기여를 지원하기 위한 보고 권장사항을 제공합니다.



### DMON: A Simple yet Effective Approach for Argument Structure Learning (https://arxiv.org/abs/2405.01216)
Comments: COLING 2024

- **What's New**: DMON (Dual-tower Multi-scale cOnvolution neural Network)이라는 새로운 접근법이 인수 구조 학습(ASL) 작업을 위해 개발되었습니다. DMON은 복잡한 문장 간 관계를 분석하여 ASL의 도전과제를 해결하는 데 중점을 둡니다.

- **Technical Details**: DMON 모델은 인수들을 관계 행렬로 구성하고, 이를 인수 임베딩(embeddings)과 결합하여 관계 텐서(relationship tensor)를 형성합니다. 또한, 맥락적 인수(contextual arguments)와의 관계를 포착하기 위한 메커니즘을 설계하였습니다.

- **Performance Highlights**: 세 가지 다른 영역(domain)의 인수 마이닝 데이터셋에서 실험적 결과는 DMON이 최신 모델(state-of-the-art models)보다 더 우수한 성능을 보였음을 입증합니다.



### TartuNLP at EvaLatin 2024: Emotion Polarity Detection (https://arxiv.org/abs/2405.01159)
Comments: Accepted to The Third Workshop on Language Technologies for Historical and Ancient Languages (LT4HALA 2024)

- **What's New**: TartuNLP 팀은 EvaLatin 2024 공동 작업에서 '역사적 라틴어 텍스트의 감정 극성 검출' 과제에 대한 제출물을 발표했습니다. 본 논문은 경쟁 결과에서 첫 번째를 차지한 LLM (Large Language Model)을 이용한 새로운 레이블 생성 방법과 어댑터 프레임워크를 사용한 매개 변수 효율적인 파인튜닝을 소개합니다.

- **Technical Details**: 팀은 훈련 데이터를 주석 처리하기 위해 두 가지 접근 방식을 사용했습니다: 1) 조직자가 제공한 극성 사전을 사용하여 추론 기반 레이블을 생성하고, 2) GPT-4를 사용하여 레이블을 생성. 언어 및 과제 어댑터를 훈련하기 위해 모국어 및 국가 간 지식 전달을 실험했습니다. 시스템은 BERT 아키텍처를 기반으로 하여 RoBERTa와 XLM-RoBERTa를 활용하였고, 다단계 파인튜닝을 통해 감정 극성 검출을 수행하는 데 초점을 맞추었습니다.

- **Performance Highlights**: LLM으로 생성된 레이블을 사용한 제출물은 전반적으로 첫 번째 자리를 차지했습니다. 모델은 F1 점수 (f-score)로 성능을 측정했습니다. 복합 클래스에서 가장 많은 문제를 겪었지만, 부정적인 클래스의 예측은 가장 쉬웠습니다. 언어 어댑터 및 과제 어댑터 훈련에 대한 지식 전달 방법의 효과를 측정한 소규모 연구에서는 예상 밖의 결과가 나왔습니다.



### It Couldn't Help But Overhear: On the Limits of Modelling  Meta-Communicative Grounding Acts with Supervised Learning (https://arxiv.org/abs/2405.01139)
Comments: work in progress

- **What's New**: 이 연구는 자연어처리(Natural Language Processing, NLP) 대화 모델들이 대화에서 '청취자(listener)'로만 존재하며, 이것이 대화의 기초를 이루는 근본적인 요소인 '조율(grounding)' 과정의 모델링을 어렵게 만든다는 문제를 지적합니다. 대화 모델들은 주로 수집된 데이터에서 '청취자'의 관점을 사용하여 학습되기 때문에, 실제 대화 참여자로서의 상호작용이나 조율 과정을 정확히 반영하지 못한다는 것입니다.

- **Technical Details**: 논문은 대화의 '지속 가능성(Supervisability)'과 '청취자(overhearers)'의 역할에 초점을 맞추고 있습니다. 특히, 대화에서 '청취자'가 참여자간의 공통적 이해(common ground)를 형성하는 과정을 어떻게 이해하고 참여하는지, 그리고 이러한 차이가 NLP 모델의 대화 전략과 '명확성 요청(clarification requests)'의 모델링에 어떤 영향을 미치는지 분석합니다. 데이터 구동 방식(data-driven methods)과 고정된 코퍼스를 사용하여 대화 전략을 모델링하는 현재의 접근 방식이 충분한 정보를 제공하지 못하고, 인간의 대화 행위를 재현하는데 한계가 있다는 점을 지적합니다.

- **Performance Highlights**: 이 논문은 NLP 대화 모델이 '청취자'로서만 기능할 때 발생하는 이해도와 대화 참여의 제한을 분석하고, 다양한 대화 상황에서 인간 참여자가 어떻게 다르게 반응하는지에 대한 '가변성(variability)'을 조사합니다. 또한, 실제 참여자가 아닌 '청취자'의 관점에서 대화 데이터를 수집하고 모델링하는 현행 방식이 복잡한 인간 대화의 다양한 가능성을 완벽하게 포착하지 못한다는 점을 강조합니다.



### Efficient Data Generation for Source-grounded Information-seeking  Dialogs: A Use Case for Meeting Transcripts (https://arxiv.org/abs/2405.01121)
- **What's New**: 새로운 데이터셋인 MISeD(Meeting Information Seeking Dialogs)를 개발하여 회의 기록에 기반한 정보 탐색 대화의 독특한 데이터 세트를 제공합니다. 이는 사용자가 놓친 회의를 따라잡는 데 도움이 되도록 설계되었습니다. 또한, 기존의 수작업 Wizard-of-Oz (WOZ) 방법에서 벗어나 대규모 언어 모델 (LLMs: Large Language Models) 기반의 프롬프팅을 통해 데이터 생성을 자동화하는 새로운 방법론을 제안합니다.

- **Technical Details**: 이 연구에서는 LLM을 사용하여 사용자 질의와 에이전트 응답 모두를 생성한 후, 데이터의 정확성을 검증하고 증강하기 위해 인간 주석자가 개입합니다. MISeD 데이터셋을 통한 모델 트레이닝은 기존의 쿼리 기반 요약 벤치마크(QMSum)뿐만 아니라 새롭게 생성된 완전 수동 WOZ 테스트 세트에서 우수한 성능을 보여줍니다.

- **Performance Highlights**: MISeD로 파인튜닝 된 모델들은 테스트 세트에서 우수한 성능을 보여 주었으며, 새로운 완전 수동 WOZ 테스트 세트와 기존 쿼리 기반 요약 벤치마크에서도 우수한 결과를 보여줌으로써 MISeD 데이터셋의 효용성을 입증하였습니다. 이는 LLM 프롬프팅과 인간 전문가의 결합이 데이터셋 생성의 효율성과 신뢰성을 크게 향상시킬 수 있음을 시사합니다.



### UniGen: Universal Domain Generalization for Sentiment Classification via  Zero-shot Dataset Generation (https://arxiv.org/abs/2405.01022)
- **What's New**: 이 연구에서는 도메인에 관계없이 데이터셋을 생성할 수 있는 새로운 접근 방식 UniGen을 제안합니다. 이 방법은 타깃 도메인에서 데이터셋 생성 패러다임의 실용성을 크게 향상시킬 수 있습니다. UniGen은 특정 도메인에 국한되지 않고 작업 모델(TAM; tiny task model)의 일반화 가능성을 달성하며, 이는 다양한 도메인에서의 적용 가능성을 보여줍니다.

- **Technical Details**: UniGen은 큰 규모의 사전 훈련된 언어 모델(PLM)을 기반으로 하여 도메인 불변의 훈련 데이터셋을 생성합니다. 이 데이터셋은 다중 소스 도메인(Domain) 없이도 일반화를 달성할 수 있게 하는 방식으로 설계되었습니다. 추가적으로, 생성된 데이터의 노이즈를 줄이기 위한 의사 재라벨링(Pseudo-relabeling) 기반의 정제 기법을 도입했습니다.

- **Performance Highlights**: UniGen을 사용하여 훈련된 TAM은 여러 도메인에서의 일반화 성능을 입증하였으며, 기존의 PLM을 직접 사용하는 방법(Prompting)에 비해 매개변수의 수가 훨씬 적음에도 불구하고 더 좋은 성능을 보여주었습니다. 이는 리소스 및 비용 효율성에서의 잠재적 이점을 제시합니다.



### The IgboAPI Dataset: Empowering Igbo Language Technologies through  Multi-dialectal Enrichmen (https://arxiv.org/abs/2405.00997)
Comments: Accepted to the LREC-COLING 2024 conference

- **What's New**: 이번 연구에서는 이그보(Igbo) 언어가 멸종 위기에 처한 점을 지적하며, 다양한 방언을 포함하는 이그보 언어 기술을 개발하는 것의 중요성을 강조하고 있습니다. 이를 위해 다언어적(Igbo-English) 사전 데이터셋인 IgboAPI를 개발하였고, 이 언어 데이터셋을 활용하여 의미론적 어휘(Semantic Lexicon)와 기계 번역(machine translation)에 관한 연구를 수행하였습니다.

- **Technical Details**: IgboAPI 데이터셋은 이그보 언어의 다양한 방언을 포함하며, 이 데이터셋을 사용하여 기존의 기계 번역 시스템을 미세조정(finetuning) 함으로써 언어의 방언 변이를 더 잘 처리할 수 있도록 하였습니다. 또한, 이그보 의미론적 태거(Igbo semantic tagger)를 위한 초기 의미론적 어휘도 성공적으로 구축하였습니다.

- **Performance Highlights**: 연구 결과, IgboAPI 데이터셋을 활용한 기계 번역 시스템은 다양한 방언 변이를 포함하는 문장의 처리 능력이 대폭 향상됨을 보여 주었습니다. 이는 기존 시스템에 비해 방언 처리에 있어서 높은 효율성을 나타냅니다.



### Context-Aware Clustering using Large Language Models (https://arxiv.org/abs/2405.00988)
Comments: 16 pages

- **What's New**: 본 논문에서는 텍스트 기반 엔티티의 효율적이고 효과적인 감독 클러스터링을 위해 개방형 대규모 언어 모델(Large Language Models, LLMs)을 활용하는 CACTUS(Context-Aware ClusTering with aUgmented triplet losS)라는 새로운 방법론을 제안합니다. 특히, 본 연구는 LLM을 사용하여 엔티티 하위 집합을 클러스터링하는 방법론에 초점을 맞추어, 기존의 텍스트 클러스터링 방법들이 제공하지 못하는 문맥 인식과 확장된 트리플 손실 함수(augmented triplet loss)를 적용함으로써 감독된 클러스터링 작업을 새롭게 정의합니다. 또한 자체 감독(self-supervised) 클러스터링 작업을 도입하여 모델의 일반화를 향상시킵니다.

- **Technical Details**: CACTUS는 엔티티 하위 집합의 문맥을 파악하기 위해 확장된 대규모 언어 모델을 사용하고, 엔티티들 사이의 상호작용을 강화하기 위한 스케일러블한 인터-엔티티 주의 메커니즘(inter-entity attention mechanism)을 도입합니다. 이를 통해 텍스트 증강기법(text augmentation techniques)을 사용한 자체 감독 클러스터링 작업을 수행함으로써, 감독된 클러스터링 작업에 직접 적용할 때 발생할 수 있는 문제들을 해결합니다. 클러스터링 평가를 위해 상용 대규모 언어 모델에서 직접 수집한 진실의 클러스터링을 바탕으로 개발된 공개 소스 LLM으로 지식을 전이합니다.

- **Performance Highlights**: 다양한 전자상거래 쿼리 및 제품 클러스터링 데이터 세트를 사용한 실험에서 CACTUS는 기존의 비감독 및 감독 기반 기준 모델들을 크게 능가하는 성능을 보였습니다. 사용된 외부 클러스터링 평가 지표(external clustering evaluation metrics)에서 특히 높은 성능을 나타내, 텍스트 클러스터링 작업에서의 적용 가능성을 입증했습니다.



### On the Evaluation of Machine-Generated Reports (https://arxiv.org/abs/2405.00982)
Comments: 12 pages, 4 figures, accepted at SIGIR 2024 as perspective paper

- **What's New**: 이 관점 논문에서는 새로운 자동 보고서 생성 (Automatic Report Generation)에 대한 비전을 제시하며, 이를 평가하기 위한 유연한 프레임워크(flexible framework)를 소개합니다. 독특하게도, 자동 보고서 생성은 보고서의 필요한 배경, 요구 사항 및 범위를 자세히 설명하는 정보 필요성에 기초하여 시작됩니다.

- **Technical Details**: 자동 보고서 생성은 완전성(completeness), 정확성(accuracy), 그리고 검증 가능성(verifiability)을 필요로 하는 상세한 정보의 제공에 초점을 맞춥니다. 보고서는 이러한 특성들을 충족시키기 위해 여러 평가 방법을 활용하여 시스템을 구축하고 평가하는 방식을 재고할 필요가 있습니다.

- **Performance Highlights**: 평가 프레임워크는 여러 평가에서 찾을 수 있는 아이디어를 활용하며, 완성도와 정확성을 테스트하기 위해서는 보고서에 반드시 포함되어야 하는 정보 덩어리(nuggets of information)를 질문과 답변 형태로 사용합니다. 또한, 보고서에서 주장한 내용을 원본 문서에 매핑하여 검증 가능성을 평가합니다.



### A Hong Kong Sign Language Corpus Collected from Sign-interpreted TV News (https://arxiv.org/abs/2405.00980)
Comments: Accepted by LREC-COLING 2024

- **What's New**: 이 논문에서는 TVB-HKSL-News라는 새로운 홍콩 수화 데이터셋을 소개합니다. 이 데이터셋은 7개월 동안 TV 뉴스 프로그램에서 수집된 것이며, 큰 어휘를 사용하는 연속 수화 인식(Sign Language Recognition, SLR)과 번역(Sign Language Translation, SLT) 연구를 지원하기 위해 개발되었습니다. 이 데이터셋은 두 명의 수화 사용자가 포함되어 있으며 총 16.07시간의 수화 비디오와 6,515개의 글로스(gloss), 그리고 2,850개의 중국어 문자 또는 18,000개의 중국어 단어를 포함하고 있습니다.

- **Technical Details**: 데이터 수집 파이프라인은 대부분 자동화되어 인간의 개입을 최소화하며, 향후 대규모의 수화 데이터 수집 및 대규모 어휘 연속 수화 인식 및 번역 데이터셋을 수집할 때 이 방법이 확장 가능할 것이라고 생각합니다. 초기 단계에서는 TV 프로그램에서 SLT 데이터를 수집하고, 해당 데이터를 처리하여 수화 비디오 세그먼트와 해당하는 텍스트 전사를 얻습니다. 이 단계는 수화 활동 탐지, 자막 추출, 수화와 자막 세그먼트 간의 정렬을 수행하는 컴퓨터 비전 기술을 활용합니다. 이어서 전문 수화 언어 주석자를 고용하고 그들이 각 수집된 수화 비디오 세그먼트에 대한 글로스를 라벨링할 수 있도록 주석 소프트웨어를 설계합니다.

- **Performance Highlights**: 이 데이터셋에서 최신 상태(SOTA) SLR/SLT 모델을 실행하여 SLR의 기준 단어 오류율(Word Error Rate, WER)은 34.08%, SLT의 기준 BLEU-4 점수는 23.58을 달성했습니다. 이 결과는 향후 TVB-HKSL-News 데이터셋 연구의 벤치마크로 사용될 것입니다.



### CACTUS: Chemistry Agent Connecting Tool-Usage to Scienc (https://arxiv.org/abs/2405.00972)
- **새로운 점**: 이 논문에서는 화학 및 분자 발견에서 고급 추론 및 문제 해결을 가능하게 하는 화학 정보학(cheminformatics) 도구를 통합한 새로운 대규모 언어 모델(LLM) 기반 에이전트 CACTUS(Chemistry Agent Connecting Tool-Usage to Science)를 소개합니다. CACTUS는 오픈 소스 LLM과 도메인 특정 도구를 통합하여 화학 및 분자 발견에서의 연구 활동을 지원할 수 있는 적응 가능한 도구를 제공하고 과학적 발전을 가속화하는 데 기여할 수 있습니다.

- **기술적 세부 사항**: CACTUS는 Gemma-7b, Falcon-7b, MPT-7b, Llama2-7b, Mistral-7b 등 다양한 오픈 소스 LLM을 이용하여 화학 관련 질문들을 평가합니다. 이 모델들은 도메인 특정 프롬프트(domain-specific prompting)와 하드웨어 설정에 따른 성능 영향도 탐구하며, 적절한 프롬프트 엔지니어링(prompt engineering) 및 소비자 등급 하드웨어에서도 정확성을 크게 떨어뜨리지 않고 소형 모델을 배포할 수 있는 가능성을 강조합니다.

- **성능 하이라이트**: CACTUS는 기존의 베이스라인 LLM 모델들보다 월등한 성능을 보여주며, 특히 Gemma-7b와 Mistral-7b 모델이 가장 높은 정확도를 달성했습니다. 이를 통해 몰레큘러 속성 예측(molecular property prediction), 유사성 검색(similarity searching), 약물성 평가(drug-likeness assessment)와 같은 과제에서 연구자들을 도울 수 있습니다. 또한, CACTUS는 자동화된 실험 플랫폼과 통합되어 실시간 데이터 기반 결정을 내리는 자동 발견의 새로운 가능성을 열어줍니다.



### How Can I Get It Right? Using GPT to Rephrase Incorrect Trainee  Responses (https://arxiv.org/abs/2405.00970)
Comments: International Journal of Artificial Intelligence in Education

- **What's New**: 1대1 튜터링이 학생 학습 향상에 매우 효과적인 전략으로 알려져 있지만, 자격을 갖춘 튜터의 부족은 큰 문제입니다. 본 연구는 이러한 문제를 극복하기 위해 최신 대형 언어 모델인 GPT-4를 활용하여 튜터 트레이니들의 피드백 시스템을 개발하였습니다. 이 시스템은 트레이니들의 응답을 바이너리(이진) 형식으로 정확하게 분류하고, 틀린 응답을 바람직한 형태로 자동 재구성하여 제공합니다.

- **Technical Details**: 연구는 '효과적인 칭찬 제공', '오류에 대한 반응', '학생들의 지식 파악' 등 3개의 트레이닝 레슨에서 410개의 트레이니 응답을 분석하였습니다. GPT-4는 few-shot (퓨샷) 학습 방법을 사용하여 트레이니의 올바른/잘못된 반응을 0.84의 평균 F1 점수와 0.85의 AUC 점수로 효과적으로 식별했습니다. 또한, 잘못된 반응을 적절하게 재구성하여 인간 전문가의 성능과 비교할 수 있는 결과를 달성했습니다.

- **Performance Highlights**: GPT-4의 활용은 튜터 트레이닝의 효율성과 효과적인 피드백 제공을 중대하게 향상시켰습니다. 특히, 잘못된 응답을 바람직한 형태로 재구성하는 능력은 트레이닝 과정에서 시간과 노력을 크게 절약할 수 있게 하며, 이는 교육 설정에서 실시간 설명 피드백 제공의 확장성 문제를 해결하는데 크게 기여할 수 있습니다.



### Efficient Compression of Multitask Multilingual Speech Models (https://arxiv.org/abs/2405.00966)
Comments: Master Thesis

- **What's New**: 이 연구는 Whisper 모델의 단점을 분석하고, 특히 지원하는 언어 중 자원이 부족한 언어를 대상으로 성능 향상을 목표로 한 새로운 모델 압축 방법인 'DistilWhisper'를 제안합니다. Whisper는 다양한 언어를 처리할 수 있는 다국어 다중 작업 음성 모델이며, 특히 영어에서 뛰어난 전사 및 번역 능력을 보이지만, 모델 크기가 작아질수록 다국어 능력이 감소하는 문제점이 있습니다.

- **Technical Details**: Whisper 모델은 weak supervision을 사용하여 훈련되었으며, 자동 음성 인식(ASR) 결과가 뛰어납니다. 그러나, 언더 리프레젠티드(under-represented) 언어들에서는 성능이 떨어지는 경향이 있습니다. 이를 해결하기 위해, DistilWhisper는 언어 특화 전문가를 활용한 경량 모듈형 ASR 미세조정과 knowledge distillation을 결합한 접근 방식을 사용합니다. 이 방식은 'whisper-large-v2' 모델로부터 지식을 전수받아 'whisper-small' 모델의 성능을 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: DistilWhisper는 표준 미세조정이나 LoRA 어댑터(Low-rank Adapter)보다 우수한 성능을 보였습니다. 실험 결과, 이 접근법은 특정 언어에 대한 ASR 성능을 향상시키는 것뿐만 아니라, 도메인 내외의 테스트 세트에서 모두 성능을 향상시켰습니다. 또한, 추론 시 추가 파라미터 오버헤드가 거의 없는 것으로 나타났습니다.



### Modeling Empathetic Alignment in Conversation (https://arxiv.org/abs/2405.00948)
Comments: Camera-ready version for NAACL 2024

- **What's New**: 이 연구는 공감적 대화에서의 정렬(alignment)을 인식하는 새로운 접근법을 소개합니다. Appraisal Theory에 기반하여, 경험에 대한 다양한 유형의 평가(appraisals)와 발화자(speaker)와 관찰자(observer)의 대화 사이에 이루어지는 공감적 정렬을 분석하였습니다. 특히, 9.2K 이상의 span-level 평가 주석과 3K 이상의 공감적 정렬 데이터셋을 새롭게 도입했습니다.

- **Technical Details**: 연구진은 9.2M개의 Reddit 대화를 통해 컴퓨터 실험을 수행하였습니다. 이를 통해 평가와 정렬을 정확히 인식할 수 있음을 보였습니다. 데이터 분석은 NLP(Natural Language Processing) 기술을 활용하였으며, 특히 공감적 대화에서 정렬 과정을 모델링하는 것에 초점을 맞추었습니다.

- **Performance Highlights**: 실험 결과, 평가가 행동의 의미 있는 그룹을 포착하는 데 효과적임을 확인했습니다. 그러나 대부분의 대화에서는 최소한의 정렬만 나타났습니다. 그럼에도 불구하고, 정신 건강 전문가들은 훨씬 더 많은 공감적 정렬을 사용하는 경향이 있는 것으로 나타났습니다.



### A Named Entity Recognition and Topic Modeling-based Solution for  Locating and Better Assessment of Natural Disasters in Social Media (https://arxiv.org/abs/2405.00903)
Comments: 15 pages; 4 tables; 4 figures

- **What's New**: 이 연구에서는 자연 재해 정보를 처리하기 위해 소셜 미디어 콘텐츠를 활용하는 새로운 NLP(자연어 처리, Natural Language Processing) 기반 해결책을 제안합니다. 제안된 방법은 관련 없는 포스트를 필터링하고, 게시물 텍스트에서 위치 정보를 추출하며, 관련된 소셜 미디어 게시물에서 논의된 주요 주제를 자동으로 식별하는 세 단계 접근 방식으로 구성됩니다.

- **Technical Details**: 이 프레임워크는 여러 최신 NLP 모델을 결합한 공로 기반 융합 프레임워크(merit-based fusion framework)를 사용하여 관련 있고 관련 없는 소셜 미디어 게시물을 구분합니다. 위치 추출(Location Extraction from Twitter Text, LETT)은 Named Entity Recognition (NER, 명명된 개체 인식) 프레임워크를 통해 이루어지며, BERTopic 라이브러리를 사용하여 주제 모델링(topic modeling)을 수행하여 관련 트윗에서 숨겨진 주제 패턴을 발견합니다.

- **Performance Highlights**: RCTP(Relevance Classification of Twitter Posts, 트위터 게시물 관련성 분류)는 BERT, RoBERTa, Distil BERT 및 ALBERT 모델을 조합하여 F1-score 0.933을 달성했습니다. LETT는 BERT, RoBERTa, DistilBERTA, Electra 모델을 사용하여 최고의 F1-score 0.960을 기록했습니다. 이러한 결과는 소셜 미디어 콘텐츠와 NLP가 재난 관리에서의 잠재력을 시사합니다.



### DynaMo: Accelerating Language Model Inference with Dynamic Multi-Token  Sampling (https://arxiv.org/abs/2405.00888)
Comments: Accepted at NAACL 2024

- **What's New**: 새롭게 제안된 DynaMo 모델은 다중 토큰(multi-token) 예측을 통해 추론 시간을 획기적으로 단축하는 기술입니다. 이 모델은 기존의 autoregressive 언어 모델들이 한 번에 하나의 토큰만을 예측하는 방식에서 벗어나, 동시에 여러 토큰을 예측할 수 있게 함으로써 인퍼런스(inference) 속도를 개선합니다.

- **Technical Details**: DynaMo는 기존의 autoregressive 모델들의 가중치를 활용하는 가벼운 학습 기법을 적용하였고, 공동 출현 가중 마스킹(co-occurrence weighted masking) 및 적응형 임계값 결정(adaptive thresholding)과 같은 새로운 방법을 도입하여 생성 텍스트의 질을 향상시키고 있습니다. 또한, 비autoregressive 방식의 텍스트 생성 품질을 엄격하게 평가할 수 있는 질적 및 양적 방법도 제안하고 있습니다.

- **Performance Highlights**: DynaMo-7.3B-T3 모델은 기준 모델(Pythia-6.9B)과 동일한 품질의 텍스트를 생성하면서, 2.57배 빠른 속도 향상을 달성했습니다. 이 모델은 매개 변수(parameter)와 훈련 시간(training time) 증가가 각각 5.87% 및 2.67%에 불과하여 효율적입니다.



### Math Multiple Choice Question Generation via Human-Large Language Model  Collaboration (https://arxiv.org/abs/2405.00864)
Comments: 17th International Conference on Educational Data Mining (EDM 2024)

- **What's New**: 이 논문에서는 수학 객관식 문제(Multiple choice questions, MCQs)를 생성할 때 교육자와 협력하여 대형 언어 모델(Large Language Models, LLMs)을 활용하는 새로운 도구의 프로토타입을 제안합니다. 특히, 이 도구는 수학 교육자들이 고품질의 객관식 문제를 보다 쉽게 만들 수 있도록 지원하기 위해 설계되었습니다.

- **Technical Details**: 이 프로토타입 도구는 교육자들이 질문의 줄기(stems)를 정교하게 만드는 과정에서 대형 언어 모델을 활용할 수 있도록 하며, 틀린 선택지(distractors)를 생성하는 과정에도 적용됩니다. 효과적인 문제 생성을 위해, 이 연구에서는 수학 교육자들을 대상으로 한 파일럿 연구를 진행하여 도구의 유용성을 평가하였습니다.

- **Performance Highlights**: 대형 언어 모델은 질문 줄기를 효과적으로 생성할 수 있는 능력을 보여준 반면, 학생들의 흔한 오류나 오해를 포착하는 선택지를 만드는 데에는 한계가 있었습니다. 그럼에도 불구하고, 인간과 인공지능(Artificial Intelligence, AI)의 협업은 객관식 문제 생성의 효율성과 효과를 향상시킬 가능성을 가지고 있습니다.



### WIBA: What Is Being Argued? A Comprehensive Approach to Argument Mining (https://arxiv.org/abs/2405.00828)
Comments: 8 pages, 2 figures, submitted to The 16th International Conference on Advances in Social Networks Analysis and Mining (ASONAM) '24

- **What's New**: WIBA는 여러 맥락에서 'What Is Being Argued'를 종합적으로 이해할 수 있는 새로운 프레임워크와 방법론을 제안합니다. 이 방법은 논쟁의 존재, 주제, 그리고 입장을 식별할 수 있는 종합적인 프레임워크를 개발하였으며, 이 세 작업 간의 논리적 의존성을 정확하게 다룹니다. 특히 Large Language Models(LLMs)의 파인튜닝(fine-tuning)과 프롬프트 엔지니어링(prompt-engineering)을 활용했다는 점에서 혁신적입니다.

- **Technical Details**: WIBA는 세 가지 주요 기능을 개발했습니다. 첫째, 논쟁 감지 모델을 개발하여 문장이 논쟁인지를 분류할 수 있으며, 세 가지 벤치마크 데이터셋에서 79%에서 86% 사이의 F1 점수를 달성했습니다. 둘째, 주제 감지 모델은 명시적이거나 암시적인 주제를 식별할 수 있으며, 기존의 나이브(naive) 방법보다 평균 40% 높은 효율을 보였습니다. 셋째, 논쟁 입장 분류 방법을 개발하여, 제공된 주제에 대한 텍스트의 논쟁적 입장을 분류할 수 있으며, 이 방법 또한 기존 방법보다 높은 점수를 기록했습니다. 이러한 기능은 WIBA를 통해 다양한 맥락의 대규모 코퍼스에서 논쟁을 이해하는 데 중요한 기여를 할 것입니다.

- **Performance Highlights**: WIBA의 세 가지 기능은 모두 높은 퍼포먼스를 보였습니다. 논쟁 감지는 최대 86%의 F1 점수를, 주제 감지는 기존 방법보다 40% 더 효율적이었으며, 논쟁 입장 분류는 71%에서 78% 사이의 F1 점수를 달성했습니다. 이는 WIBA가 기존의 방법들을 크게 능가하며, 다양한 유형의 논쟁 및 정식 및 비정식 문맥에서의 논쟁을 효과적으로 식별할 수 있다는 것을 보여줍니다.



### WorkBench: a Benchmark Dataset for Agents in a Realistic Workplace  Setting (https://arxiv.org/abs/2405.00823)
- **What's New**: 새롭게 소개되는 WorkBench는 직장 환경에서 작업 실행 능력을 평가하기 위한 벤치마크 데이터셋입니다. 이 데이터셋은 다섯 개의 데이터베이스, 26개의 도구, 그리고 690개의 작업을 포함한 샌드박스 환경을 제공합니다. 이 작업들은 이메일 발송이나 회의 스케줄링과 같은 일반적인 비즈니스 활동을 대표합니다.

- **Technical Details**: WorkBench는 계획 수립, 도구 선택 및 다수의 행동을 요구하는 도전적인 작업들로 이루어져 있습니다. 작업이 성공적으로 수행되면, 하나 이상의 데이터베이스 값이 변경될 수 있습니다. 각 작업에 대한 정확한 결과는 독특하며 명확하여 강력한 자동 평가를 가능하게 합니다. 이를 '결과 중심 평가(outcome-centric evaluation)'라고 부릅니다.

- **Performance Highlights**: WorkBench에서 평가된 다섯 개의 기존 ReAct 에이전트들 중 가장 낮은 성능을 보인 에이전트는 3%의 작업만 성공적으로 완료했으며(Llama2-70B), 가장 높은 성능을 보인 에이전트는 43%의 작업만 성공적으로 완료했습니다(GPT-4). 에이전트들의 오류는 잘못된 행동을 초래할 수 있으며, 이는 잘못된 수신자에게 이메일이 발송되는 등의 결과를 낳았습니다. WorkBench는 에이전트들이 일반적인 비즈니스 활동을 수행하는 데에 있어서의 약점을 드러내며, 고위험 직장 환경에서의 사용에 대한 의문을 제기합니다.



### Uncovering Agendas: A Novel French & English Dataset for Agenda  Detection on Social Media (https://arxiv.org/abs/2405.00821)
- **What's New**: 이 논문에서는 한정된 또는 존재하지 않는 주석 데이터에서 소셜 미디어를 통한 의제 조종(agenda control)을 감지하기 위한 방법론을 제시합니다. 특히, 이 연구는 중요한 정치 및 사회 이벤트, 예를 들면 2022년 프랑스 대선과 관련된 온라인 영향 캠페인(online influence campaigns)을 분석하는 데 집중합니다.

- **Technical Details**: 연구팀은 Twitter 메시지를 이용하여 데이터셋을 구성하고, 텍스트를 추론(textual entailment) 문제로 처리하여 접근하는 다양한 방법을 평가했습니다. 텍스트 추론은 주어진 텍스트가 다른 텍스트를 함축하는지 판단하는 과정을 말합니다.

- **Performance Highlights**: 이 방법론은 큰 규모의 주석 데이터(annotation data)가 필요하지 않다는 큰 장점이 있습니다. 연구 결과, 이러한 접근 방식을 사용하면 소셜 미디어에서 의제를 조종하는 개별 사례들을 효과적으로 탐지할 수 있음을 보여줍니다.



### "Ask Me Anything": How Comcast Uses LLMs to Assist Agents in Real Tim (https://arxiv.org/abs/2405.00801)
- **What's New**: 이 연구에서는 고객 서비스 에이전트를 위해 'Ask Me Anything' (AMA)란 새로운 AI 기능을 소개합니다. 이 기능은 대규모 언어 모델(Large Language Model, LLM)을 활용하여 에이전트가 고객과 대화를 처리하는 동안 실시간으로 정확한 답변을 제공합니다. 이를 통해 에이전트의 문맥 전환(Context Switching) 필요성을 줄이고, 효율적인 고객서비스를 도모할 수 있습니다.

- **Technical Details**: AMA 기능은 검색-증강 생성(Retrieval-Augmented Generation, RAG) 접근 방식을 활용하여, 내부 지식 소스를 결합해 문맥에 맞는 답변을 생성합니다. 문서는 전처리 과정을 거쳐 텍스트로 변환되며, 데이터는 벡터 데이터베이스에 메타데이터와 함께 저장됩니다. 여러 검색 모델의 효율성을 평가하여, 특히 OpenAI의 최신 ada-002 임베딩 모델이 높은 성능을 보였습니다. 언급된 모델로는 Dense Passage Retrieval (DPR), MPNet-base, BM25 등이 있습니다. 또한, GPT-4를 활용해 합성 데이터로부터 질문을 생성하고 이를 통해 재순위 모델을 훈련하는 방법을 사용했습니다.

- **Performance Highlights**: AMA를 사용한 에이전트는 전통적인 검색 경험과 비교했을 때 대화당 검색 시간을 약 10% 줄였으며, 이는 연간 수백만 달러의 절감 효과로 이어집니다. 또한, AMA 기능을 사용한 에이전트의 80%가 긍정적인 피드백을 제공했습니다. 이를 통해 AI-지원 고객 관리 기능의 유용성이 입증되었습니다.



### LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Repor (https://arxiv.org/abs/2405.00732)
- **What's New**: 본 연구에서는 LoRA (Low Rank Adaptation)를 사용하여 LLMs (Large Language Models)를 미세 조정함으로써 실제 환경에서의 효율성과 성능을 평가했습니다. 특히, LoRAX라는 새로운 멀티-LoRA 추론 서버를 소개하고, 이를 통해 단일 GPU에서 여러 LoRA 미세 조정 모델을 효율적으로 배포할 수 있는 방법을 탐구했습니다.

- **Technical Details**: 연구자들은 10개의 베이스 모델에서 31개의 작업에 대해 총 310개의 모델로 quantized low rank adapters를 사용하여 LLMs의 품질을 측정했습니다. 또한, LoRAX는 공유 베이스 모델 가중치 및 동적 어댑터 로딩을 사용하여 단일 GPU 위에서 다수의 LoRA 미세 조정 모델을 활용할 수 있습니다. LoRA Land는 싱글 NVIDIA A100 GPU로 25개의 LoRA 미세 조정된 Mistral-7B LLMs를 호스팅하는 웹 애플리케이션입니다.

- **Performance Highlights**: 4-bit로 미세 조정된 LoRA 모델은 베이스 모델들을 평균 34점, GPT-4를 평균 10점 초과하여 성능에서 우수함을 보였습니다. LoRA를 사용한 미세 조정 방법은 완전한 미세 조정(full fine-tuning)과 비교하여 경쟁력 있는 성능을 제공함과 동시에 훈련 가능한 파라미터의 수와 메모리 사용을 줄입니다. LoRA Land의 배포는 다양한 특화된 LLMs를 단일 일반 목적 LLM보다 더 비용 효율적으로 사용하는 것을 강조합니다.



### Evaluating the Application of ChatGPT in Outpatient Triage Guidance: A  Comparative Study (https://arxiv.org/abs/2405.00728)
Comments: 8 pages, 1 figure, conference(International Ergonomics Association)

- **What's New**: 본 연구는 의료 분야에서 인공지능(AI)을 통합하는 것이 운영 효율성과 건강 결과를 향상시키는 잠재력을 가지고 있음을 제시합니다. 특히, 대규모 언어 모델(Large Language Models, LLMs)인 ChatGPT의 외래 환자 트리아지(outpatient triage) 문제 해결에의 잠재력을 평가하고, ChatGPT를 의료 시스템에 통합하는 것이 의료 발전에서 유망한 추세가 되고 있다고 강조합니다.

- **Technical Details**: 이 연구는 ChatGPT 버전 3.5와 4.0을 사용하여 외래 환자 지도에서의 응답 일관성을 평가했습니다. 내부 응답 일관성은 ChatGPT-4.0이 ChatGPT-3.5보다 유의미하게 높은 결과를 보여주었습니다 (p=0.03). 그러나 두 버전 간의 일관성은 상대적으로 낮으며 (평균 일관성 점수=1.43/3, 중앙값=1), 권장 사항이 일치하는 경우는 50%에 불과했습니다. ChatGPT-3.5는 ChatGPT-4.0보다 완전한 응답을 제공할 가능성이 더 높았습니다 (p=0.02).

- **Performance Highlights**: ChatGPT-4.0의 상위 추천 일관성은 71.2%, ChatGPT-3.5는 59.6%로 나타났습니다. 이는 두 버전 모두에서 적정 수준의 일관성을 유지하고 있지만 완전히 최적화된 상태는 아니라는 것을 시사합니다. 더구나 버전 간 일관성이 낮다는 점은 두 모델 간 정보 처리 및 응답 생성 방식의 차이가 있음을 나타냅니다.



### LLMs for Generating and Evaluating Counterfactuals: A Comprehensive  Study (https://arxiv.org/abs/2405.00722)
- **What's New**: 이 연구는 인공지능(AI)의 해석 가능성에 초점을 맞추고 있으며, 특히 Large Language Models (LLMs)가 Natural Language Processing (NLP)에서 Counterfactuals (CFs)를 생성하는 능력을 조사합니다. LLMs가 Sentiment Analysis (SA)와 Natural Language Inference (NLI) 같은 NLU (Natural Language Understanding) 작업을 위해 CF를 어떻게 생성하는지 철저하게 비교하고 평가했습니다.

- **Technical Details**: 다양한 크기와 접근성 수준의 LLMs를 분석하여 CF 생성 작업에서의 성능을 평가했습니다. 이 연구는 플립율(Flip Rate, FR), 텍스트 유사도(Textual Similarity, TS) 및 혼란도(Perplexity, PPL)와 같은 표준 메트릭을 사용하여 결과를 측정했습니다. LLMs는 유창한 텍스트를 생성할 수 있는 능력이 있지만, 최소한의 변경을 유도하는 데 어려움이 있었습니다. 특히, NLI 작업에서 이러한 한계가 두드러졌습니다.

- **Performance Highlights**: LLMs에서 생성된 CF는 데이터 보강(Data Augmentation)에서 인간이 생성한 CF와 비교하여 유사한 성능을 달성할 수 있음을 보여줍니다. 그러나, NLI 작업에서는 개선이 필요했습니다. 또한, LLMs는 제공된 레이블에 동의하는 강한 편향을 보였으나, GPT-4는 이러한 편향에 대해 더 강건하며, CFs의 품질 평가에 적합한 성능을 보였습니다. 이러한 결과로부터 LLMs와 인간이 생성한 CF 간의 성능 격차를 이해하고, 더 나은 CF 생성 방안을 모색할 수 있는 새로운 연구 방향이 제시됩니다.



### Can't say cant? Measuring and Reasoning of Dark Jargons in Large  Language Models (https://arxiv.org/abs/2405.00718)
- **What's New**: 이 연구는 대규모 언어 모델(LLM: Large Language Models)의 암호언어(cant or dark jargon)에 대한 반응을 평가하기 위해 'CantCounter' 평가 프레임워크를 소개합니다. 이는 특정 도메인에서의 cant를 인식하고 이해하는 LLM의 능력을 조사합니다. 특히 정치, 마약, 인종차별, 무기 및 LGBT와 같은 민감한 주제들에서 LLM의 반응을 알아보고, 이러한 도메인에서 cant를 사용하여 보안 필터를 우회하는 방법을 분석합니다.

- **Technical Details**: CantCounter는 GPT-2 모델을 사용하여 Scene 데이터 조각을 생성하고, Co-Tuning을 통해 Cant 데이터셋과 Scene 조각을 매치시키며, Data-Diffusion 기술로 텍스트를 확장하고, Data-Analysis 방법으로 복잡한 계산을 단순화합니다. 이러한 과정을 통해, 연구팀은 LLM이 다양한 cant에 대해 어떻게 다르게 반응하는지 탐구하고, 해당 데이터셋을 사용하여 LLM의 추론 능력을 평가합니다.

- **Performance Highlights**: 연구 결과에 따르면, 개선된 모델들은 cant 질의에 대한 수용률이 더 높고, LLM은 도메인에 따라 다르게 반응합니다. 예를 들어, LLM은 인종차별적 내용보다는 LGBT 관련 주제에 대해 더 관여하는 경향이 있습니다. 이는 LLM이 훈련 데이터의 특성과 제공업체의 접근 방식을 반영함을 시사합니다.



### Exploring News Summarization and Enrichment in a Highly Resource-Scarce  Indian Language: A Case Study of Mizo (https://arxiv.org/abs/2405.00717)
Comments: Accepted at LREC-COLING2024 WILDRE Workshop

- **What's New**: 이 논문에서는 Mizo 뉴스 기사에 대한 포괄적인 요약을 생성하기 위해 영어 뉴스를 활용하여 해당 뉴스 이벤트에 관련된 정보를 보완하고 향상시키는 간단한 방법론의 효과성을 조사합니다. 또한, 500개의 Mizo 뉴스 기사와 해당 향상된 종합 요약본을 제공합니다. 인간 평가를 통해 저자들은 이 접근 방식이 Mizo 뉴스 기사의 정보 범위를 현저하게 향상시킴을 확인했습니다.

- **Technical Details**: 이 연구에서는 고자원 언어(예: 영어)에서 추출한 관련 정보를 활용하여 저자원 언어인 Mizo 기사를 풍부하게 하는 간단한 파이프라인을 도입합니다. 이 과정에는 Mizo 뉴스 기사를 영어로 번역하고, 최첨단 헤드라인 생성 모델을 사용하여 헤드라인을 생성한 다음, 생성된 헤드라인으로 웹 검색을 수행하여 유효한 URL을 추출하는 단계가 포함됩니다. 이어서, 식별된 URL에서 문서를 검색하고 최첨단 사전 학습 모델을 사용하여 다중 문서 요약(Multi-document Summarization, MDS)을 수행합니다. 최종적으로 얻은 요약을 해당 Mizo 문서에 추가하고 전체 영문 문서를 Mizo어로 다시 번역합니다.

- **Performance Highlights**: 인간 평가 결과는 제안된 파이프라인이 저자원 언어 뉴스 기사를 효과적으로 풍부하게 할 수 있음을 나타냅니다. 평가는 요약의 일관성, 향상, 관련성을 평가하는 네 가지 범주를 기반으로 수행되었습니다. 또한, 이 연구는 Mizo와 같은 저자원 언어에 대한 데이터셋과 연구 자료를 제공하여 향후 연구를 촉진시키는 데 기여합니다.



### Large Language Models in Healthcare: A Comprehensive Benchmark (https://arxiv.org/abs/2405.00716)
- **What's New**: 이 논문에서는 임상 의사들의 지원을 위해 큰 언어 모델(Large Language Models, LLMs)의 채택이 주목받고 있으나, 대부분 폐쇄형 질문응답(Question-Answering, QA) 작업에 초점을 맞추고 있습니다. 현실적인 임상 환경에서는 여러 임상 결정이 개방형 질문으로 이루어지기 때문에 보다 폭넓은 벤치마크가 필요합니다. 본 연구는 의료 분야에서 다양한 LLM을 종합적으로 벤치마킹하여, 그 강점과 약점을 명확히 이해하고자 합니다. 이를 위해 의료 언어 생성(medical language generation), 이해(medical language understanding), 추론(medical reasoning)을 포함한 일곱 가지 작업과 열세 개의 데이터셋을 사용하여 평가를 진행하였습니다.



### Towards Adapting Open-Source Large Language Models for Expert-Level  Clinical Note Generation (https://arxiv.org/abs/2405.00715)
- **What's New**: 이 연구에서 우리는 작은 오픈소스 대형 언어 모델(LLaMA-2, 13 billion parameters)을 이용하여 외래환자 및 의사 대화에서 고품질의 임상 메모를 생성할 수 있음을 보여줍니다. DistillDirect라는 새로운 접근 방식을 도입하여 정책 강화 학습(on-policy reinforcement learning)을 수행했고, 이를 통해 LLaMA-Clinic라는 모델을 개발했습니다. 이 모델은 의사가 작성한 노트와 비교할 때 유사한 질의 임상 노트를 생성할 수 있습니다. 또한, 우리는 합성 클리닉 대화-노트 데이터셋 및 의사 피드백 데이터셋을 공개하여 이 분야의 연구를 촉진하고자 합니다.

- **Technical Details**: LLaMA-2 모델은 지속적 사전훈련(continued pretraining), 지도 학습(supervised fine-tuning, SFT), AI 및 인간 피드백으로부터의 강화 학습(reinforcement learning) 단계를 거쳤습니다. 특히, DistillDirect 접근 방식을 통해 강화 학습을 수행하여 모델의 성능을 최적화하였습니다. 또한, 의사들의 의견을 반영하여 '최고의 실습 노트 형식(best-practice note format)'을 정의하는 데 중점을 두었습니다.

- **Performance Highlights**: LLaMA-Clinic 모델은 '실제 준비도(real-world readiness)', '완성도(completeness)', '정확성(accuracy)'의 세 가지 기준에서 90.4%가 '수용 가능(acceptable)' 이상으로 평가되었습니다. 더 도전적인 '평가 및 계획(Assessment and Plan)' 섹션에서는 실제 준비도에서 4.2/5로, 의사가 작성한 노트(4.1/5)보다 더 높은 점수를 받았습니다. 이러한 결과는 오픈소스 LLM이 의료 분야, 특히 임상 노트 생성에서 의사들과 비슷한 수준의 성과를 낼 수 있음을 증명합니다.



### Fake Artificial Intelligence Generated Contents (FAIGC): A Survey of  Theories, Detection Methods, and Opportunities (https://arxiv.org/abs/2405.00711)
- **What's New**: 이 논문은 인공지능 생성 컨텐츠(AIGC)의 새로운 문제점으로 부각된 가짜 인공지능 생성 컨텐츠(FAIGC)에 초점을 맞추며, 다양한 모드 및 생성 기술에 관한 포괄적인 분류 체계(Taxonomy)를 제안합니다. 또한, 이 문서는 여러 감지 방법들을 조사하고 업데이트된 프레임워크를 통해 FAIGC 연구 분야에 기여하고자 합니다.

- **Technical Details**: 이 연구에서는 FAIGC의 세 가지 주된 요소를 고려한 새로운 분류 체계를 제시합니다: 생성 의도(Intent), 다양한 모달(Modalities), 그리고 생성 기술(Generative Technologies). 대표적인 인공지능 모델로는 Large Language Models (LLMs)와 Diffusion Models (DMs)가 있습니다. 세부적으로 AI 생성 정보 왜곡(disinformation)과 오보(misinformation)로 나뉩니다. 감지 방법은 Deceptive FAIGC Detection, Deepfake Detection, Hallucination-based FAIGC Detection으로 구분하여 설명합니다.

- **Performance Highlights**: 이 연구는 FAIGC 탐지 방법을 여러 모달에서 다루며, 심층적이고 체계적인 검토를 통해 연구자들이 FAIGC 문제에 접근하는 방식을 개선할 수 있는 정보를 제공합니다. 또한, 인공지능의 발전이 가져오는 부작용을 예방하고 기술적으로 대응할 수 있는 근거를 마련하는데 중점을 둡니다.



### Homonym Sense Disambiguation in the Georgian Languag (https://arxiv.org/abs/2405.00710)
- **What's New**: 이 연구는 조지아어의 단어 의미 분별(word sense disambiguation, WSD) 작업에 대해 새로운 접근 방식을 제안합니다. 사전 훈련된 Large Language Model(LLM)을 조지아어 Common Crawls 코퍼스로부터 필터링된 데이터셋에 기반하여 미세 조정(fine-tuning)하는 방식을 사용합니다. 이 데이터셋은 다의어를 가진 단어들의 분류기를 훈련하는데 사용됩니다.

- **Technical Details**: 조지아어는 Kartvelian 언어 가족에 속하는 응집 언어(agglutinative language)로서, 자연 언어 처리에서 동음이의어를 정확히 구분하는 것이 매우 중요합니다. 실험 결과로는 LSTM을 사용한 WSD에 대해서도 언급합니다. 7500문장 이상의 수동으로 분류된 데이터셋을 사용하여 동음이의어의 어휘적 의미를 예측하는 데 95%의 정확도를 달성한 기술을 소개합니다.

- **Performance Highlights**: 제안된 방법은 조지아어의 WSD 작업에서 동음이의어(lexical meanings of homonyms)의 의미를 예측할 때 95%의 높은 정확도를 보입니다. 이는 데이터셋이 사전에 잘 준비되고, 효과적인 미세 조정 접근 방식을 통해 달성된 결과입니다.



### Evaluating Tool-Augmented Agents in Remote Sensing Platforms (https://arxiv.org/abs/2405.00709)
Comments: ICLR 2024 Machine Learning for Remote Sensing (ML4RS) Workshop

- **What's New**: 새롭게 개발된 GeoLLM-QA 벤치마크는 원격 탐지(Remote Sensing, RS) 응용 프로그램에서 대규모 언어 모델(Large Language Models, LLMs)의 성능을 평가합니다. 기존 벤치마크가 이미지-텍스트 데이터 쌍에 대한 미리 정의된 질문-응답 형식에만 초점을 맞춘 반면, GeoLLM-QA는 실제 사용자 상황에서의 행동을 포함한 시나리오를 다루어 복잡성을 더합니다. 사용자가 지도에서 특정 영역을 확대하고 의무적으로 '여기 있는 모든 객체를 감지해라'라고 명령할 때 '여기'가 시스템 상태에 의해 암시되는 지점이 어디인지를 인식할 수 있도록 설계되었습니다.

- **Technical Details**: GeoLLM-QA는 문자, 시각, 클릭 기반 동작의 긴 시퀀스를 포함하는 실제 사용자 인터페이스(UI) 플랫폼에서의 작업을 캡처하기 위해 개발되었습니다. 이 벤치마크는 지도상의 실시간 위치 설정같은 시스템 상태를 고려함으로써, 텍스트-이미지 템플릿의 명확한 하드코딩 없이도 사용자의 지역 질문을 이해하고 처리할 수 있어야 합니다.

- **Performance Highlights**: GeoLLM-QA는 다양한 1,000개의 작업을 포함하여 최신 대규모 언어 모델들(Large Language Models)의 성능을 깊이 있게 평가합니다. 이 벤치마크를 통해 얻은 통찰력은 원격 탐지(Remote Sensing, RS) 응용 프로그램에 대한 더 강력한 에이전트 개발로 이어질 수 있습니다.



### Interactive Analysis of LLMs using Meaningful Counterfactuals (https://arxiv.org/abs/2405.00708)
- **What's New**: 이 연구에서는 LLM (Large Language Models)의 결정 경계를 탐색하고 기능 속성을 결정하는 데 유용한 역설적(Counterfactual) 예를 분석하고 설명하는 방법에 대해 논의합니다. 또한, 이 논문은 대량의 의미있고 읽을 수 있는 텍스트 역설적 예를 생성하는 새로운 알고리즘과 LLM의 행동을 이해할 수 있게 돕는 대화형 시각화 도구인 LLM Analyzer를 소개합니다.



### Science Written by Generative AI is Perceived as Less Intelligent, but  More Credible and Trustworthy than Science Written by Humans (https://arxiv.org/abs/2405.00706)
- **What's New**: 이 연구는 과학 커뮤니케이션을 단순화하고 과학에 대한 대중의 신뢰를 강화하기 위해 생성적 AI(Generative AI)의 효과성을 평가했습니다. PNAS 저널 기사의 요약과 AI가 생성한 요약을 비교함으로써, 언어적 단순성과 대중의 인식을 평가했습니다.

- **Technical Details**: 연구 1a는 PNAS의 초록(Scientific Summaries)과 중요성 진술(Lay Summaries)의 단순성 특징을 분석하여, 중요성 진술이 언어적으로 더 단순하다는 것을 확인했지만, 효과 크기 차이는 작았습니다. 연구 1b에서는 GPT-4를 사용하여 문서 초록을 기반으로 중요성 진술을 생성하여, 평균 효과 크기를 두 배 이상 증가시켰습니다. 연구 2에서는 GPT가 작성한 단순한 요약이 인간이 작성한 복잡한 PNAS 요약보다 과학자들에 대한 대중의 인식(신뢰성, 신뢰도)을 더 긍정적으로 촉진했다는 것을 실험적으로 보여주었습니다.

- **Performance Highlights**: AI는 과학 커뮤니케이션의 단순성을 통해 과학 커뮤니티와 대중 간의 참여를 촉진할 잠재력을 가지고 있으며, 이를 과학 전파에 통합함으로써 보다 정보에 입각한 사회를 위한 변화를 제안합니다. 특히 GPT-4는 문서 초록을 바탕으로 효과적인 중요성 진술을 생성함으로써, AI가 과학적 내용을 보다 넓은 대중에게 전달하는 데 큰 역할을 할 수 있음을 시사합니다.



### SHED: Shapley-Based Automated Dataset Refinement for Instruction  Fine-Tuning (https://arxiv.org/abs/2405.00705)
- **What's New**: 이 논문에서는 SHED(Shapley value 기반의 자동 데이터셋 가공 프레임워크)를 소개합니다. SHED는 최신 대규모 언어 모델(LLMs)을 적은 양의 고품질 데이터만을 사용하여 맞춤화 하여 인간의 선호도에 맞게 미세 조정하는 방법을 제시합니다. SHED는 사람의 개입 없이 대규모 데이터셋에서 고품질의 데이터만을 선별하여 효과적인 소규모 데이터셋을 만들 수 있습니다.

- **Technical Details**: SHED는 Shapley value를 활용하여 데이터셋의 각 데이터 포인트의 가치를 평가하고, 가치 있는 데이터만을 선별합니다. 이 프로세스는 자동화되어 있으며, 상업용 LLMs의 사용을 필요로 하지 않습니다. 또한, SHED로 큐레이트된 데이터셋은 다양한 LLMs에서 일관되게 높은 성능을 보여주며, 재사용 가능성을 입증합니다.

- **Performance Highlights**: SHED로 선택된 데이터의 단 10%만을 사용하더라도, 기존 전체 데이터셋을 사용했을 때와 비슷하거나 더 우수한 성능을 달성했습니다. 이는 SHED가 다양한 작업 및 LLMs에 대해 최신 방법론들을 능가하는 성능을 보여줌을 의미합니다. SHED는 이러한 성과를 통해 데이터 효율성 및 고품질의 데이터 추출에 중요한 가능성을 보여줍니다.



### A Survey on the Real Power of ChatGP (https://arxiv.org/abs/2405.00704)
Comments: 9 pages, 2 tables

- **What's New**: 이 논문은 ChatGPT의 실제 성능 수준을 평가하기 위한 최근 연구를 조사하고 있습니다. 또한 ChatGPT의 사회적 함의(social implications)와 안전 문제(safety issues)를 검토하고, 성능 평가의 주요 도전 과제와 기회를 강조합니다.

- **Technical Details**: 이 연구는 7가지 NLP(Natural Language Processing, 자연어 처리) 작업 카테고리에서 ChatGPT의 성능을 조사합니다. ChatGPT가 아직 폐쇄 소스(closed-source)이기 때문에 그 성능을 평가하는 데는 주요한 도전이 있습니다. 전통적인 벤치마크 데이터셋이 ChatGPT의 훈련 데이터로 사용되었을 가능성이 있기 때문입니다.

- **Performance Highlights**: 연구자들은 ChatGPT의 '블랙박스(blackbox)' 방식을 밝히기 위해 노력하며, 그것이 단순히 표면적인 생성(surface generation)에 의해 오도되지 않도록 경고합니다.



### Learnable Linguistic Watermarks for Tracing Model Extraction Attacks on  Large Language Models (https://arxiv.org/abs/2405.01509)
Comments: not decided

- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 지적 재산을 보호하기 위한 새로운 수단으로 학습 가능한 언어적 워터마크를 제안합니다. 토큰 빈도 분포에 제어된 노이즈를 도입함으로써 LLM 출력 분포를 세밀하게 수정하고 통계적으로 식별 가능한 제어 워터마크를 삽입합니다.

- **Technical Details**: 저자들은 통계적 가설 검정과 정보 이론을 활용하여 변경된 분포와 원래의 분포를 효과적으로 구별하는 기술을 사용합니다. 특히, 쿨백-라이블러 발산(Kullback-Leibler Divergence)을 이용하여 원본과 수정된 분포 간의 차이를 측정합니다. 데이터 세트에서 각 토큰의 빈도를 누적하고 가우시안 노이즈를 추가하여 모델의 예측 분포를 수정하는 방법으로 워터마크를 생성합니다.

- **Performance Highlights**: 이 워터마킹 방법은 강인함과 출력 품질 사이의 섬세한 균형을 맞춰, 원래 LLM의 성능을 유지하면서 낮은 거짓 양성률과 거짓 음성률을 유지합니다. 생성된 텍스트의 품질을 저하시키지 않으면서도 모델 추출 공격을 추적할 수 있는 안정적이고 강력한 워터마크 탐지를 달성합니다.



### MANTIS: Interleaved Multi-Image Instruction Tuning (https://arxiv.org/abs/2405.01483)
Comments: 9 pages, 3 figures

- **What's New**: 이 논문은 다중 이미지 태스크를 다룰 수 있는 멀티이미지 LMMs의 성능을 향상시키기 위해 학술 수준의 자원을 활용한 지시학습(instruction tuning)을 통해 강력한 멀티이미지 LMMs를 구축하는 것을 목표로 합니다. 이를 위해 14개의 다중 이미지 데이터셋에서 721K 인스턴스로 구성된 Mantis-Instruct를 세심하게 구축하고, 다양한 다중 이미지 스킬(multi-image skills)을 처리할 수 있도록 설계했습니다.

- **Technical Details**: Mantis-Instruct는 공참조(co-reference), 추론(reasoning), 비교(comparing), 시간적 이해(temporal understanding) 등 다양한 다중 이미지 스킬을 포함합니다. 이 데이터셋과 여러 개별 이미지 시각-언어 데이터셋을 결합하여 Mantis 모델을 훈련시킵니다. Mantis는 A100-40G 기준으로 36시간 동안 16개의 GPU를 사용하여 효율적으로 훈련되었습니다.

- **Performance Highlights**: Mantis-8B는 모든 다중 이미지 벤치마크에서 최고의 성능(state-of-the-art)을 달성했으며, 기존의 최고 멀티이미지 LMM인 Idefics2-8B를 평균 9점 차이로 상회했습니다. 또한, 단일 이미지 벤치마크에서도 강력한 성능을 유지하며 CogVLM, Emu2와 동등한 수준을 보였습니다. 이 결과는 저비용의 지시학습이 다중 이미지 LMMs를 구축하는 데 있어 집중적인 사전 훈련보다 효과적임을 시사합니다.



### MiniGPT-3D: Efficiently Aligning 3D Point Clouds with Large Language  Models using 2D Priors (https://arxiv.org/abs/2405.01413)
Comments: 17 pages, 9 figures

- **What's New**: MiniGPT-3D는 2D vision-language 모델에 영감을 받아 3D 포인트 클라우드(point cloud)와 LLM을 효율적으로 결합하는 새로운 접근방식을 제시하며, 이는 3D-LLM 개발에 필요한 교육 비용을 상당히 줄여줍니다. 이 모델은 특히 2D-LLMs에서 얻은 2D 사전 지식을 사용하여 3D 포인트 클라우드를 LLM과 효과적으로 정렬합니다.

- **Technical Details**: MiniGPT-3D는 4단계 훈련 전략(four-stage training strategy)과 쿼리 전문가(query experts)의 혼합을 사용하여 모달리티(modality) 정렬을 순차적으로 구현합니다. 또한, LoRA와 Norm fine-tuning과 같은 파라미터 효율적인 미세 조정 방법을 사용하여 학습 가능한 파라미터를 기존 방법보다 최대 260배 줄인 47.8M으로 제한합니다.

- **Performance Highlights**: MiniGPT-3D는 3D 객체 분류 및 캡셔닝 작업에서 SOTA(State-of-the-Art) 결과를 달성하고, RTX 3090 한 대에서 단 27시간 만에 훈련을 완료하며 전체 GPU 시간을 대폭 줄였습니다. GPT-4 평가 점수에서는 기존 ShapeLLM-13B 대비 8.12 포인트 증가를 보였으며, 이는 ShapeLLM-13B가 8개의 A800 GPU에서 160시간을 소모하는 것과 비교됩니다.



### Overcoming LLM Challenges using RAG-Driven Precision in Coffee Leaf  Disease Remediation (https://arxiv.org/abs/2405.01310)
Comments: 6 pages, 3 figures

- **What's New**: 이 연구는 YOLOv8을 이용한 질병 식별과 Retrieval Augmented Generation (RAG)을 이용한 맥락 인식 진단을 통합한 혁신적인 AI 주도 정밀 농업 시스템을 소개합니다. 카르나타카의 커피 생산 부문에 영향을 미치는 질병 문제를 해결하기 위해 고안된 이 시스템은 정교한 객체 감지 기술과 언어 모델을 통합하여 대규모 언어 모델 (LLMs: Large Language Models)과 관련된 본질적인 제약을 해결합니다.

- **Technical Details**: 이 시스템은 신뢰할 수 있고 실시간의 질병 식별 능력을 갖출 뿐만 아니라, LLM에서 발생할 수 있는 '환각(hallucination)' 문제를 해결할 수 있는 새로운 동적 질병 식별 및 치료 전략을 도입합니다. YOLOv8은 한 번에 전체 이미지를 분석하고, YOLOv8의 높은 속도는 객체를 실시간으로 처리하며, 이는 농업에서 시기적절한 질병 탐지에 매우 중요합니다. RAG는 LLMs과 통합되어 외부 데이터베이스에서 최신의 맥락별 데이터를 가져오는 역할을 하여, LLM의 정적 성질로 인한 진단의 부정확성을 최소화하는 데 도움을 주며, 지속 가능하고 환경 친화적인 농업 경험을 제공합니다.

- **Performance Highlights**: 연구 결과, YOLOv5와 YOLO-Dense 네트워크와 관련 연구에 따르면 높은 정밀도 (Precision), 재현율 (Recall), 평균정밀도 (mAP: mean Average Precision) 등을 달성하였습니다. 특히, 이 시스템은 다양한 농업 설정에서 적응할 수 있도록 실시간 모니터링, 협업 데이터셋 확장 및 조직적 참여를 보장합니다. 또한, 제안된 시스템은 자동화를 넘어 식량 공급의 안전을 확보하고 생계를 보호하며 친환경 농업 관행을 촉진하는 것을 목표로 합니다.



### Identification of Entailment and Contradiction Relations between Natural  Language Sentences: A Neurosymbolic Approach (https://arxiv.org/abs/2405.01259)
- **What's New**: 자연어 이해의 중요한 분야인 자연어 추론(Natural language inference, NLI or Recognizing Textual Entailment, RTE)에서 설명 가능하고 명확한 접근 방법의 필요성을 충족시키기 위해, 본 연구에서는 추상 의미 표현(Abstract Meaning Representation, AMR) 그래프로 텍스트를 번역하고, 그 다음 명제 논리(propositional logic)로 변환하여 SAT 해결자(SAT solver)를 사용하는 새로운 파이프라인을 제안합니다. 이러한 방법은 기존의 머신러닝(Machine learning, ML) 및 딥러닝(Deep Learning, DL) 방법의 '블랙박스' 문제를 해결하고자 합니다.

- **Technical Details**: 본 연구는 IBM에서 사전 훈련된 AMR 파서를 사용하여 자유 텍스트의 각 문장을 AMR 그래프로 변환하며, 각 AMR 그래프를 명제 논리의 수식으로 변환합니다. 또한, 두 문장 간의 의미적 의미를 비교하기 위해 대규모 사전 훈련된 언어 모델을 사용하고, 명제 수식에서 일부 원자를 대체하거나 잊어버리는(Forgetting) 방법을 조사합니다. 이러한 과정을 통해 자연어 문장 간의 함축(entailment), 모순(contradiction), 중립(neutral) 관계를 식별합니다. 논리 추론을 위해 PySAT 정리 증명기(theorem prover)를 사용하여 수정된 전제가 주장과 일치하거나 모순되는지를 결정합니다.

- **Performance Highlights**: 이 파이프라인은 네 개의 RTE 데이터셋에서 성공적으로 수행되었습니다. 본 접근 방식은 기존의 머신러닝 방법과 비교할 때 더욱 투명하고 설명 가능한 결과를 제공하며, 자연어 처리(Natural Language Processing, NLP) 태스크에서의 응용 가능성을 증명합니다.



### Boosting Jailbreak Attack with Momentum (https://arxiv.org/abs/2405.01229)
Comments: ICLR 2024 Workshop on Reliable and Responsible Foundation Models

- **What's New**: 최근 연구에서는 LLMs (Large Language Models)가 다양한 태스크에서 뛰어난 성능을 보여주었음에도 불구하고 새로운 형태의 공격인 GCG (Greedy Coordinate Gradient) 공격을 통해 취약성이 드러났습니다. 이 공격은 기울기(gradient) 휴리스틱과 탐욕적 검색(greedy search)을 결합하여 적대적 프롬프트를 최적화합니다. 본 논문에서는 이 공격의 효율성 문제를 개선하기 위해 적대적 프롬프트 생성을 최적화의 관점에서 재고하고, 이전 반복에서 얻은 휴리스틱 인사이트를 활용하여 최적화 과정을 안정화하고자 합니다.

- **Technical Details**: 이 연구에서 제안하는 새로운 공격 방법은 MAC (Momentum Accelerated GCG) 공격으로, 이는 기존 GCG 공격 방식에 모멘텀(momentum)이라는 요소를 추가함으로써 그래디언트 휴리스틱의 효과를 강화합니다. 이 접근 방식은 적대적 프롬프트의 최적화 과정에서 더 빠르고 일관된 성능을 제공하며, 이전 반복에서의 정보를 효과적으로 활용하여 공격의 효율성을 높이는 것을 목표로 합니다.

- **Performance Highlights**: MAC 공격의 실험 결과는 기존 GCG 공격 대비 뛰어난 성능 향상을 보여주었습니다. 언어 모델들에 대한 공격에서 MAC 공격이 가져온 반복적 그래디언트 기반의 적대적 프롬프트 최적화는 더 높은 정확도와 공격 성공률을 달성하였음을 확인할 수 있습니다. 이러한 결과는 LLMs를 대상으로 한 적대적 공격에서의 새로운 가능성을 열어줍니다.



### Silencing the Risk, Not the Whistle: A Semi-automated Text Sanitization  Tool for Mitigating the Risk of Whistleblower Re-Identification (https://arxiv.org/abs/2405.01097)
Comments: Accepted for publication at the ACM Conference on Fairness, Accountability, and Transparency 2024 (ACM FAccT'24). This is a preprint manuscript (authors' own version before final copy-editing)

- **What's New**: 이 논문에서는 내부 고발자들이 신원이 공개될 위험 없이 부정 행위를 보고할 수 있도록 돕는 새로운 컴퓨터 기반 전략을 제안하고 평가합니다. 고유한 글쓰기 스타일과 보고서의 내용을 통해 내부고발자를 재식별할 수 있는 위험에 초점을 맞추고, 이를 관리하기 위해 텍스트를 재작성하는 분류 및 완화 전략을 적용합니다.

- **Technical Details**: 이 연구는 자연어 처리(Natural Language Processing, NLP)를 사용하여 고위험 단어나 용어를 매핑하고, 잠재적인 식별 정보를 포함할 수 있는 텍스트를 산문적으로 분리시키지만 적절하게 정화된 중간 텍스트 버전을 생성합니다. 또한, Large Language Model(LLM)을 사용하여 이 텍스트를 일관되고 스타일이 중립적인 형태로 재구성합니다. 사용자의 맥락 지식을 활용하여 위험과 유용성을 평가하는 인터랙티브 기능도 도입되었습니다.

- **Performance Highlights**: 이 도구는 저자 식별 공격(Authorship Attribution, AA)에 대한 방어 효과를 통계적으로 측정하고, IMDb62 영화 리뷰 데이터셋을 사용하여 유틸리티 손실을 평가했습니다. 내부고발 사례의 직접적 및 준식별자를 효과적으로 숨길 수 있었으며, AA 정확도를 98.81%에서 31.22%로 크게 감소시키면서 원본 내용의 의미를 최대 73.1%까지 보존하는 데 성공했습니다.



### Few Shot Class Incremental Learning using Vision-Language models (https://arxiv.org/abs/2405.01040)
Comments: under review at Pattern Recognition Letters

- **What's New**: 이 연구에서는 적은 데이터를 사용하여 새로운 클래스를 통합하는 문제를 해결하기 위해 언어 정규화기(language regularizer)와 부분 공간 정규화기(subspace regularizer)를 활용하는 새로운 few-shot class incremental learning (FSCIL) 프레임워크를 소개합니다. 이 프레임워크는 기본 클래스(base classes)의 성능을 유지하면서 새로운 클래스를 효과적으로 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 언어 정규화기는 비전-언어 모델(Vision-Language Model)에서 추출된 의미론적 정보를 통합하여 기본 모델 훈련을 개선하는 데 사용됩니다. 부분 공간 정규화기는 이미지와 텍스트 의미론 사이의 미묘한 연결을 학습하도록 돕습니다. 또한, 이 연구에서는 CLIP 언어 모델을 이용한 프롬프트 엔지니어링(prompt engineering) 방법론을 제안하여, 제로샷(zero-shot) 및 퓨샷(few-shot) 학습 작업에 활용할 수 있습니다.

- **Performance Highlights**: 제안된 프레임워크는 CIFAR-FS 데이터셋에서 기존 방법보다 약 7% 향상된 성능을 보였으며, Mini-Imagenet 및 Tiered-Imagenet 데이터셋에서도 각각 약 1%의 성능 향상을 보여줬습니다. 이러한 결과는 FSCIL 벤치마크에서의 상태 최고의 성능(state-of-the-art performance)을 입증합니다.



### Bayesian Optimization with LLM-Based Acquisition Functions for Natural  Language Preference Elicitation (https://arxiv.org/abs/2405.00981)
- **What's New**: PEBOL (Preference Elicitation with Bayesian Optimization augmented LLMs)은 향상된 대화형 참고 시스템과 이를 위한 선호도 정확도를 도출하기 위해 특별히 설계된 새로운 아이템 선호도 표출 (preference elicitation) 알고리즘입니다. PEBOL은 자연 언어 추론 (Natural Language Inference, NLI)을 사용하여 사용자의 선호도를 추적하고 Bayesian Optimization (BO) 를 통해 더 효과적으로 정보를 취합하며, 이를 바탕으로 자연 언어 처리 (Natural Langauge Processing, NLP) 쿼리를 생성합니다.

- **Technical Details**: PEBOL은 NLI 모델과 Bayesian Optimization 프레임워크를 통합하여, 사용자 대화 동안 보다 지능적으로 자연어 쿼리를 생성하고 선호도 정보를 업데이트합니다. 이 모델은 Thompson Sampling (TS)과 Upper Confidence Bound (UCB)과 같은 결정 이론 기반 전략을 사용하여 쿼리 생성을 최적화하는 방법을 제공합니다. PEBOL은 400M 파라미터의 NLI 모델을 기반으로 작동하면서도 모노리식 GPT-3.5 모델보다 최대 131% 개선된 성능 (MAP@10)을 보여줍니다.

- **Performance Highlights**: PEBOL은 10번의 대화 차례로 이루어진 실험에서 GPT-3.5에 비해 최대 131% 향상된 MAP@10 성능을 달성했습니다. 이러한 결과는 PEBOL이 사용자의 선호도를 보다 정확하게 파악하고 효과적으로 추론하는 데 기여하는 효율적인 전략을 사용한다는 것을 시사합니다.



### Language Fairness in Multilingual Information Retrieva (https://arxiv.org/abs/2405.00978)
Comments: 5 pages, 1 figure, accepted at SIGIR 2024 as short paper

- **What's New**: 이 연구에서는 다국어 정보 검색(multilingual information retrieval, MLIR)의 언어 공정성(language fairness)을 측정하는 새로운 지표, PEER (Probability of Equal Expected Rank)를 제안하였습니다. PEER 지표는 Kruskal-Wallis 검정을 사용하여, 다양한 언어로 구성된 문서가 쿼리에 대해 공정하게 순위가 매겨졌는지를 통계적으로 평가합니다. 기존의 공정성 지표와 달리 PEER는 특정 언어 그룹을 보호 그룹(protected group)으로 설정하지 않고, 모든 언어가 동등하게 취급되어야 한다는 점을 강조합니다.

- **Technical Details**: PEER 지표는 비모수적(non-parametric) 분산 분석 방법인 Kruskal-Wallis H 테스트를 활용하여, 주어진 쿼리에 대한 문서의 관련성(relevance) 수준에 상관없이 모든 언어의 문서가 같은 순위를 기대할 수 있는 확률을 측정합니다. 이는 각 언어 그룹의 문서가 검색 결과에서 공정하게 대표될 확률을 계산하여, 다양한 언어 간의 공정성을 보장하려고 합니다. 이론적인 구축과 실제 MLIR 시스템 평가를 통해 PEER 지표의 유효성을 검증하였습니다.

- **Performance Highlights**: PEER 지표는 CLEF 2003과 NeuCLIR 2022 벤치마크를 통해 평가되었으며, 인위적으로 생성된 순위 목록과 실제 검색 시스템 출력에서의 공정성을 분석한 결과와 일치하는 점수를 보여주었습니다. 이러한 결과는 PEER가 MLIR 시스템에서 문서의 언어에 따른 편향을 식별하고 정량화하는 데 효과적임을 시사합니다. 또한, PEER 구현체는 ir-measures와 호환되며 온라인에서 접근이 가능합니다.



### Distillation for Multilingual Information Retrieva (https://arxiv.org/abs/2405.00977)
Comments: 6 pages, 1 figure, accepted at SIGIR 2024 as short paper

- **What's New**: 새로운 프레임워크인 다국어 번역-증류 (Multilingual Translate-Distill, MTD)는 다양한 언어의 문서에 대한 정보 검색 (Multilingual Information Retrieval, MLIR) 작업을 지원합니다. 이 방법은 기존의 단일 언어 문서 검색 (Cross-Language Information Retrieval, CLIR)을 확장하여 다국어 문서 집합의 순위를 매기고 각 언어의 문서에 비슷한 중요도 점수를 부여하는 능력을 향상시킵니다.

- **Technical Details**: 이 연구에서는 기존의 '번역-증류 (Translate-Distill)' 프레임워크를 확장하여 MTD 모델을 개발하였습니다. 이를 통해 동일한 검색 쿼리로 다양한 언어의 문서를 효과적으로 처리할 수 있습니다. 이 모델은 번역 및 증류 기법을 사용하여 크로스-언어 신경 듀얼-인코더 모델(Cross-Language Neural Dual-Encoder Model)을 훈련합니다. 또한, 훈련 배치에서 언어의 혼합 방식에 강건하며, GitHub에서 구현체를 제공합니다.

- **Performance Highlights**: MTD로 훈련된 ColBERT-X 모델은 이전 최고 성능 모델인 다국어 번역-훈련 (Multilingual Translate-Train) 방식으로 훈련된 모델보다 nDCG@20에서 5%에서 25% 향상되었으며, MAP에서는 15%에서 45% 향상된 성능을 보였습니다. 이 결과는 MTD가 MLIR 작업에 매우 효과적임을 시사합니다.



### PLAID SHIRTTT for Large-Scale Streaming Dense Retrieva (https://arxiv.org/abs/2405.00975)
Comments: 5 pages, 1 figure, accepted at SIGIR 2024 as short paper

- **What's New**: PLAID는 ColBERT late interaction bi-encoder를 활용하여 최고의 성능을 달성하는 모델로, 단일 언어, 다중 언어 및 교차 언어 검색에서 탁월한 성능을 보입니다. 특히, 새로운 PLAID SHIRTTT (Streaming Hierarchical Indexing that Runs on Terabytes of Temporal Text) 모델은 문서가 시간에 따라 도착하는 스트리밍 환경에서도 효율적으로 대응하며, 다단계 증분 색인 생성(incremental indexing)과 계층적 샤딩(hierarchical sharding)을 통해 성능 저하 문제를 해결합니다.

- **Technical Details**: PLAID는 문서의 각 단어를 클러스터로 할당하고, 이를 클러스터 중심점(cluster centroids)과 압축된 잔여 벡터(compressed residual vectors)로 표현합니다. 이 방법은 기존 ColBERT 모델의 향상된 버전으로 볼 수 있습니다. 그러나, 스트리밍 환경에서는 새로운 토큰(new tokens)이 이전에 선택된 클러스터 중심으로 잘못 표현될 수 있어서 문제가 됩니다. PLAID SHIRTTT는 이러한 문제를 다계층 색인과 샤딩을 활용해 효과적으로 해결합니다.

- **Performance Highlights**: PLAID는 ClueWeb09 및 다언어 NeuCLIR 데이터 셋에서 실험을 통해 뛰어난 성능을 입증하였습니다. 특히, ColBERT 구조로 색인된 가장 큰 컬렉션을 처리하는데 효과적이었으며, 다언어 환경에서도 우수한 검색 결과를 보여주었습니다. PLAID SHIRTTT는 기존 PLAID 모델의 성능을 유지하면서도 실시간 데이터 처리에서의 예상되는 문제들을 해결하였습니다.



### The Role of Model Architecture and Scale in Predicting Molecular  Properties: Insights from Fine-Tuning RoBERTa, BART, and LLaMA (https://arxiv.org/abs/2405.00949)
- **What's New**: 이 연구는 다양한 화학정보학(cheminformatics) 작업에 대해 대규모 언어 모델(Large Language Models, LLMs)의 미세 조정(fine-tuning) 효능을 비교하는 체계적인 프레임워크를 도입합니다. RoBERTa, BART, LLaMA 등 세 가지 잘 알려진 모델들이 SMILES(Simplified Molecular Input Line Entry System)를 전세계적인 분자 표현 형식으로 사용하여 분자의 성질을 예측하는 능력을 평가하였습니다.

- **Technical Details**: 이 비교 분석은 이 모델들의 18가지 구성을 사전 훈련하고, DeepChem에서 제공하는 여섯 가지 벤치마킹 작업에 맞추어 미세 조정하는 작업을 포함했습니다. 모든 모델에 걸쳐 일관된 훈련 환경을 유지하여 신뢰할 수 있는 비교가 가능하게 하였습니다. 본 연구는 모델 유형, 크기 및 훈련 데이터셋 크기가 모델 성능에 미치는 영향을 평가하였습니다.

- **Performance Highlights**: 특히, LLaMA 기반 모델은 일반적으로 가장 낮은 검증 손실(validation loss)을 보여줌으로써 작업과 규모에 걸쳐 우수한 적응성을 제공하는 것으로 나타났습니다. 그러나 절대적인 검증 손실이 모델 성능의 결정적인 지표가 아니며, 모델 크기가 중요한 역할을 한다는 것을 발견하였습니다.



### LLaVA Finds Free Lunch: Teaching Human Behavior Improves Content  Understanding Abilities Of LLMs (https://arxiv.org/abs/2405.00942)
- **What's New**: 이 논문은 기존의 큰 언어 모델(LLMs)을 훈련할 때 종종 무시되는 수신자 행동(receiver behavior) 데이터가 모델의 내용 이해 능력을 향상시킬 수 있음을 보여줍니다. 특히, '좋아요'와 '댓글' 같은 수신자 행동을 예측하기 위해 LLM을 훈련시키면 다양한 내용 이해 작업에서 모델 성능이 향상됩니다.

- **Technical Details**: 연구자들은 수신자 행동 데이터를 이용하여 LLM을 훈련시키는 새로운 접근 방식을 제안합니다. 이 방법은 40개의 비디오와 이미지 이해 작업을 포함하는 23개의 벤치마크 데이터셋에서 0-shot(Zero-shot) 및 미세 조정(fine-tuning) 설정 모두에서 기존의 감독된 베이스라인을 능가하는 성능 향상을 보여줍니다.

- **Performance Highlights**: 수신자 행동을 기반으로 훈련된 LLM은 0-shot 및 fine-tuning 설정에서 40가지 다양한 비디오 및 이미지 이해 작업을 통해 감독된 부류의 기존 모델보다 뛰어난 성과를 보였습니다. 또한, 논문의 저자들은 750k개의 이미지와 비디오에서 수집한 '좋아요'와 '댓글' 데이터를 정리하여 공개하였고, 해당 데이터는 인터넷에서 기본적으로 수집되며 추가적인 인간 주석이 필요하지 않습니다.



### Characterising the Creative Process in Humans and Large Language Models (https://arxiv.org/abs/2405.00899)
- **What's New**: 이 연구는 큰 언어 모델(Large Language Models, LLM)의 창의성에 초점을 맞추며, 특히 창의적 과정(creative process)에 주목하고 있습니다. 이전 연구들은 주로 창의적 산출물(products)에 집중했지만, 이 연구는 사람과 LLM이 어떻게 의미 공간(semantic spaces)을 탐색하는지 자동화된 방법을 제공합니다.

- **Technical Details**: 연구자들은 교대 사용 과제(Alternate Uses Task)에서 사람과 LLM의 의미 공간 탐색 방식을 분석하고 이를 언어 유창성 과제(Verbal Fluency Task)의 행동과 대조하였습니다. 문장 임베딩(sentence embeddings)을 사용하여 응답 범주를 식별하고 의미 유사성(semantic similarities)을 계산, 이를 바탕으로 점프 프로필(jump profiles)을 생성했습니다. 이 접근 방식은 LLM과 인간 모두에서 깊은 탐구(persistent)와 넓은 탐구(flexible)의 두 가지 창의적 경로를 확인합니다.

- **Performance Highlights**: 연구 결과에 따르면 LLM은 과제에 따라 지속적인 경로와 유연한 경로에 편향되어 있었습니다. 각각의 경로는 비슷한 창의성 점수(creativity scores)로 이어졌으나, 보다 유연한 모델이 더 높은 창의성 점수를 기록했습니다. 이는 LLM과 인간 사이의 창의적 관계가 다르다는 것을 시사합니다. 연구 데이터와 스크립트는 GitHub에서 공개적으로 접근 가능합니다.



### Modeling Caption Diversity in Contrastive Vision-Language Pretraining (https://arxiv.org/abs/2405.00740)
Comments: 14 pages, 8 figures, 7 tables

- **What's New**: 새로운 연구에서 이미지의 다양한 설명을 모델링하는 Latent Language Image Pretraining (Llip)을 소개합니다. Llip은 기존의 CLIP 방식을 발전시키며, 이미지를 설명하는 텍스트에 따라 시각적 특징을 다양하게 표현할 수 있게 합니다. 이는 하나의 이미지에 대해 여러 개의 유효한 캡션을 생성할 수 있도록 하여 시각-언어 모델의 상세한 표현력을 향상시킵니다.

- **Technical Details**: Llip은 여러 개의 ‘mixture tokens’을 출력하는 시각적 변환기 (visual transformer)를 사용해, 주어진 텍스트 캡션을 기반으로 이미지의 다양한 시각적 측면을 포착합니다. 텍스트 캡션을 이용하여 믹스 토큰들의 가중치를 추론하는 교차 주의 기법 (cross-attention mechanism)을 통해, 상황에 맞는 시각적 대표성을 형성합니다. 이는 클래식한 CLIP 모델과 비교하였을 때 향상된 시각적 표현을 도출합니다.

- **Performance Highlights**: Llip은 zero-shot 분류 작업에서 평균적으로 2.9% 향상된 성능을 보였으며, ViT-G/14 인코더를 사용한 ImageNet에서 zero-shot top-1 정확도 83.5%를 달성하여 유사한 크기의 CLIP 모델을 1.4% 상회했습니다. 또한 MS-COCO 데이터셋에서 zero-shot 이미지-텍스트 및 텍스트-이미지 검색에서 6.0% 향상된 결과를 보였습니다.



### Large Language Models for Human-Robot Interaction: Opportunities and  Risks (https://arxiv.org/abs/2405.00693)
- **What's New**: 이 연구에서는 사회 로봇에 대규모 언어 모델(Large Language Models, LLM)을 적용할 잠재력에 대한 메타 연구를 제시하였습니다. 교육, 건강 관리, 그리고 오락과 같은 사회 로봇의 응용분야에 특별한 강조를 두었습니다.

- **Technical Details**: 연구팀은 사회 규범과 문제들을 '이해'할 수 있도록 언어 모델을 안전하게 훈련시키는 방법을 탐구했습니다. 이에는 신뢰(trust), 편향(bias), 윤리(ethics), 인지(cognition), 그리고 팀워크(teamwork) 같은 중요한 요소들이 포함됩니다.

- **Performance Highlights**: 대규모 언어 모델의 최근 개발은 예상보다 빠른 연구 결과를 만들어냈으며, 로봇 기술 연구자들이 로봇에 언어 모델을 통합하는 데 유용한 안내를 제공할 것으로 기대됩니다.



### Understanding Social Perception, Interactions, and Safety Aspects of  Sidewalk Delivery Robots Using Sentiment Analysis (https://arxiv.org/abs/2405.00688)
Comments: 34 pages, 7 figures, 2 tables

- **What's New**: 이 논문은 보도 배달 로봇(Sidewalk Delivery Robots, SDRs)에 관한 YouTube 댓글의 감정 분석(sentiment analysis, SA)을 전반적으로 다루고 있습니다. YouTube 댓글을 수동으로 세 가지 감정 레이블로 주석을 달고, 이를 이용해 텍스트 감정 분류 모델을 구축하고 테스트하였습니다.

- **Technical Details**: 연구팀은 이진 분류(binary classification)와 삼진 분류(ternary classification) 작업에 대해 모델의 성능을 평가하였습니다. 이진 분류에서는 TF-IDF(Term Frequency-Inverse Document Frequency)와 N-gram을 사용하는 SVM(Support Vector Machine) 모델이 가장 높은 정확도를 보였습니다. 삼진 분류에서는 BERT(Bidirectional Encoder Representations from Transformers), LSTM(Long Short-Term Memory Networks), GRU(Gated Recurrent Unit) 모델이 다른 기계 학습 모델보다 월등히 높은 성능을 보여 주었습니다.

- **Performance Highlights**: BERT, LSTM, GRU를 사용한 모델은 삼진 분류에서 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 점수(F1 score) 모두 0.78을 달성하였습니다. 또한, Latent Dirichlet Allocation(LDA) 모델을 사용하여 SDRs에 대한 대중의 깊은 관점을 탐구하기 위해 댓글에서 10개의 주제를 생성했습니다.



### DAM: A Universal Dual Attention Mechanism for Multimodal Timeseries  Cryptocurrency Trend Forecasting (https://arxiv.org/abs/2405.00522)
- **What's New**: 본 논문에서는 블록체인(Blockchain)과 암호화폐(cryptocurrency) 시장의 동향을 예측하기 위한 새로운 이중 주목 메커니즘(Dual Attention Mechanism, DAM)을 제시합니다. 이 모델은 암호화폐 관련 지표와 뉴스 및 소셜 미디어에서의 감정 데이터를 통합하여 암호화폐 시장의 변동성과 예측 도전을 해결합니다. 본 연구는 분산 시스템, 자연어 처리(Natural Language Processing, NLP), 재무 예측의 교차점에서 얻은 통찰을 기반으로 하며, 실제 금융 시장에서 유용하게 활용될 수 있는 결과를 도출합니다.

- **Technical Details**: DAM 모델은 시계열 데이터를 다루는 기존 LSTM 및 Transformer 모델보다 최대 20% 향상된 예측 정확도를 보여주었습니다. 이 모델은 데이터 소스로 Bitcoin 데이터를 Cryptocompare API를 통해, 비트코인 뉴스 감정 데이터를 Nasdaq에서, 소셜 미디어 데이터는 공개 Kaggle 데이터셋에서 수집했습니다. 또한, 더 나은 성능을 위해 인트라모달(intramodal) 정보와 크로스모달(crossmodal) 정보를 성공적으로 통합하였음을 Ablation Study를 통해 입증하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, DAM은 다차원적 데이터 통합으로 인해 시장 예측의 정확성을 개선하였습니다. 특히, 감정 데이터를 통한 시계열 예측의 효과를 크게 향상시켰으며, 분산 과학(Decentralized Science, DeSci)을 포함한 다양한 금융 예측 시나리오에서 유용하게 활용될 수 있을 것입니다. 이러한 향상된 예측 방법은 디지털 자산 분야에서의 전략적 계획 및 블록체인 기술의 효율적인 도입을 촉진하는 데 도움이 될 수 있습니다.



### GRAMMAR: Grounded and Modular Methodology for Assessment of  Domain-Specific Retrieval-Augmented Language Mod (https://arxiv.org/abs/2404.19232)
- **What's New**: 이 논문에서는 '연구 보강 생성(RAG)' 시스템의 평가를 위한 새로운 프레임워크인 GRAMMAR을 소개합니다. 이 시스템은 특정 도메인의 지식 베이스에 질의하여 정보를 검색하고 활용하는 RAG 시스템의 효율성과 정확성을 평가하는데 중점을 두고 있습니다. 특히, 도메인 특화 데이터의 부족과 기존 평가 방식의 한계를 극복하고자 새로운 데이터 생성 과정과 모듈 오류 분석 방법을 도입했습니다.

- **Technical Details**: GRAMMAR 평가 프레임워크는 두 가지 주요 구성 요소를 포함합니다: 1) 관계형 데이터베이스와 대규모 언어 모델(LLMs)을 활용하는 데이터 생성 과정; 2) 지식 결함과 시스템 강인성 문제를 구분할 수 있는 평가 방법. 데이터 생성은 SQL 쿼리를 사용하여 구조화된 질의-응답 쌍을 생성하며, 이 방법은 질의 로직과 언어적 변형을 분리하여 디버깅을 강화합니다. 평가 프레임워크는 불량 모듈을 식별할 수 있도록 설계되었으며, 특히 실험을 통해 검증된 강인성 지표를 사용합니다.

- **Performance Highlights**: GRAMMAR의 실증적 결과는 현재 참조 없는 평가 방식의 한계를 강조하고 GRAMMAR의 모델 취약점 식별 능력의 신뢰성을 입증합니다. 특히, GRAMMAR은 지식 결함과 강인성 문제를 동시에 평가하면서 정확하고 신뢰할 수 있는 결과를 제공합니다.



### A Framework for Real-time Safeguarding the Text Generation of Large  Language Mod (https://arxiv.org/abs/2404.19048)
- **What's New**: LLMSafeGuard는 실시간에서 대규모 언어 모델(LLMs)의 텍스트 생성을 보호하기 위한 새로운 프레임워크를 제안합니다. 이는 유해한 콘텐츠 생성으로 인한 윤리적 및 사회적 위험을 감소시키기 위해 외부 검증기(validator)를 빔 탐색(beam search) 알고리즘에 통합하여 안전 규제를 위반하는 후보들을 거부하고 유효한 후보들을 허용하는 방식입니다. 특히, 유사성 기반 검증 접근법(similarity-based validation approach)을 도입하여 제어 모델(control model)의 훈련 필요성을 제거하고 보다 유연한 안전 규제 도입을 가능하게 합니다.

- **Technical Details**: LLMSafeGuard는 실시간에서 작동하며, 독특한 유사성 평가 방법을 사용하여 검증을 수행합니다. 이는 미리 정의된 데모 예제(주로 위험한 텍스트 관련)를 활용하여 후보들과의 유사성을 평가하고, 높은 유사성을 보이는 후보를 즉시 거부합니다. 또한, 이 프레임워크는 검증 타이밍을 선택하기 위한 맥락별 타이밍 선택 전략(context-wise timing selection strategy)을 사용하여 필요할 때만 LLM에 개입하도록 하여 효율성을 높입니다.

- **Performance Highlights**: LLMSafeGuard는 두 가지 임무, 디톡시피케이션(detoxification)과 저작권 보호(copyright safeguarding)에서 기존 SOTA 베이스라인보다 우수한 성능을 보여줍니다. 디톡시피케이션에서는 LLM의 평균 독성 점수(toxic score)를 기존 최고 베이스라인보다 29.7% 감소시켰으며, 저작권 임무에서는 가장 긴 공통 부분수열(Longest Common Subsequence, LCS)을 베이스라인 비교하여 56.2% 감소시켰습니다. 추가로, 맥락별 타이밍 전략은 추론 시간을 최소 24% 줄이면서 비슷한 효과를 유지합니다.




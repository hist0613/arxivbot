New uploads on arXiv(cs.CL)

### Norm Growth and Stability Challenges in Localized Sequential Knowledge Editing (https://arxiv.org/abs/2502.19416)
Comments:
          Accepted for Oral Presentation at KnowFM @ AAAI 2025. arXiv admin note: text overlap with arXiv:2502.01636

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 지식 편집(knowlwdge editing)에서 국소 업데이트(localized updates)의 영향을 조사합니다. 지식 편집은 특정 사실을 수정하는 작업으로, 모델의 전반적인 능력은 그대로 유지하면서 특정 세부정보를 추가하거나 변경하는 데 중점을 둡니다. 연구는 다양한 후처리 개입(post-training interventions) 방법을 통해 업데이트된 행렬의 Frobenius norm이 항상 증가한다는 것을 보여줍니다.

- **Technical Details**: 연구에서 살펴본 개입 방법으로는 지속적 전처리(Continuous Pre-training), 전체 세부 조정(Full Fine-tuning), LORA 기반 세부 조정(LORA-based Fine-tuning)이 포함됩니다. 지속적 전처리는 특정 도메인의 대규모 텍스트 데이터로 모델의 기본 지식을 확장하는 것을 의미하며, 전체 세부 조정은 특정 과업에 대해 모델의 매개변수를 최적화하는 훈련 방식입니다. 연구 결과, 이러한 후처리 개입 동안 행렬 업데이트가 이루어질 때마다 Norm이 증가하는 현상이 관찰되었습니다.

- **Performance Highlights**: 국소 업데이트에서는 업데이트된 부분의 Norm이 비례적으로 증가하며, 이로 인해 모델 전체 안정성이 손상되고 성능 저하가 발생할 수 있습니다. 논문에서는 2000회의 업데이트를 통해 GPT2-XL과 GPT-J 모델에서 다양한 모델 편집 방법이 사용된 결과를 분석하였으며, 이 과정에서 성능 저하가 나타나기 시작했습니다. 업데이트된 활성화의 Norm은 지속적으로 감소하는 경향이 있으며, 이는 편집된 모델이 이전 모델과 다른 표현 공간의 영역을 점유하게 됨을 나타냅니다.



### The Mighty ToRR: A Benchmark for Table Reasoning and Robustness (https://arxiv.org/abs/2502.19412)
- **What's New**: ToRR(테이블 추론 및 견고성)이라는 새로운 벤치마크를 소개하고 있습니다. 이 벤치마크는 모델의 성능과 견고성을 테이블 관련 작업에서 측정할 수 있는 도구입니다. 10개의 다양한 데이터셋을 포함하여 테이블 추론 능력을 평가하며, 모델 간의 성능 순위 이상으로 모델의 일관성 있는 처리 여부에 초점을 맞추고 있습니다.

- **Technical Details**: ToRR는 다양한 도메인에서 진행되는 테이블 추론 작업을 평가하기 위해 설계되었습니다. 이는 여러 테이블 표현 형식에서 모델이 얼마나 잘 작동하는지를 판단하는 데 도움을 줍니다. 연구 결과에서는 브리틀 모델(취약한 모델)의 행동 패턴을 관찰할 수 있으며, 이는 심지어 강력한 모델도 테이블 데이터 작업에서 견고하게 수행하지 못함을 시사합니다.

- **Performance Highlights**: 다양한 테이블 형식으로 테스트하는 것이 모델의 능력을 신뢰성 있게 추정하는 데 필수적이라는 것을 증명했습니다. 또한, 여러 프롬프트(prompt)를 사용하여 성능을 평가하는 것이 더욱 많은 테스트 예제를 추가하는 것과 비슷한 신뢰성 향상을 가져올 수 있음을 보여주었습니다. 전체적인 발견은 테이블 이해 및 추론 작업이 여전히 큰 도전 과제가 남아있음을 강조합니다.



### Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs (https://arxiv.org/abs/2502.19411)
Comments:
          Project Repo: this https URL

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)에서 코드와 추론이 서로 강화하는 방법을 탐구합니다. 특히 코드가 어떻게 구조화된 매개체로 기능하며 LLM의 추론 능력을 향상시키는지, 그리고 추론의 발전이 코드 지능에 어떤 영향을 미치는지를 조사합니다. 연구자는 이러한 상호작용의 중요성을 강조하고, LLM이 복잡한 소프트웨어 엔지니어링 작업을 수행하는 능력을 향상시키기 위해 코드와 추론의 시너지를 극대화할 필요성을 제기합니다.

- **Technical Details**: 코드는 추론 과정에서 실행 가능한 경로를 제공하고, 논리적 분해(logical decomposition)를 강제하며, 런타임 검증(runtime validation)을 가능하게 합니다. 연구자는 코드 표현이 LLM의 추론에 미치는 영향, 고급 추론 능력이 코드 지능 시스템을 어떻게 재구성하는지, 그리고 코드와 추론의 상호작용에서 발생하는 주요 도전 과제를 다룹니다. 이를 위해 LLM이 코드로 문제를 구조화하고 결과를 검증하는 방식을 분석하고, 향상된 추론 능력이 코드 지능의 경계를 확장하는 방법을 탐구합니다.

- **Performance Highlights**: 최근의 솔루션들은 복잡한 문제 해결 능력을 향상시키기 위한 코드 생성 기법을 도입했습니다. 예를 들어, Program of Thoughts(PoT)와 Program-aided language models(PaL)를 통해 수치 문제를 코드 생성으로 바꾸어 해결할 수 있습니다. 또한, 코드 기반 솔루션이 자연언어 솔루션보다 모델의 정확성을 향상시킨다는 연구 결과가 있으며, 실세계의 애매모호한 작업에서 LLM이 일관된 추론을 유지하는 데 어려움을 겪는 경우도 관찰되었습니다.



### DataMan: Data Manager for Pre-training Large Language Models (https://arxiv.org/abs/2502.19363)
Comments:
          ICLR2025 paper

- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 성능이 데이터 스케일링 법칙(data scaling laws)에 의해 어떻게 향상되는지를 다루고 있습니다. 이는 사전 학습(pre-training) 데이터의 선택 중요성을 강조하며, 기존의 방법들이 제한된 휴리스틱(heuristics)과 인간 직관에 의존하고 있다는 점을 지적합니다. 연구팀은 LLM이 스스로 성능에 유익한 기준을 파악하도록 유도하는 '역발상(reverse thinking)'을 통해 새로운 방향성을 제시합니다.

- **Technical Details**: 연구는 텍스트 당혹감(perplexity, PPL)과 관련된 14가지 품질 기준(quality criteria)을 도출하고, 도메인 혼합(domain mixing)을 지원하기 위해 15가지 일반 응용 분야(application domains)를 소개합니다. 그리고 데이터 매니저(Data Manager, DataMan)를 훈련시켜 포인트 기반 평가(pointwise rating)로부터 품질 평가 및 도메인 인식을 학습하게 하였습니다. 이를 통해 447B 토큰을 포함하는 사전 학습 말뭉치를 14가지 품질 평가와 도메인 유형标注으로 주석 처리하였습니다.

- **Performance Highlights**: 실험 결과, DataMan을 사용하여 30B 토큰으로 1.3B 매개 변수를 가진 언어 모델을 훈련시킨 결과, 인-컨텍스트 학습(in-context learning, ICL), 당혹감(p perplexity), 그리고 지시 따라가기 능력에서 현저한 개선을 보여주었습니다. 전체 점수(l=5) 기준으로 최상의 성능을 보이는 모델은 균일 샘플링을 사용해 50% 더 많은 데이터로 훈련된 모델을 능가했습니다. 연구팀은 과세가 높은 도메인 특정 데이터로 추가 사전 학습을 진행하여 도메인 특정 ICL 성능을 향상시키고 DataMan의 도메인 혼합 능력을 검증하였습니다.



### Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning? (https://arxiv.org/abs/2502.19361)
Comments:
          The first three authors contributed equally, 27 pages

- **What's New**: 최근 o1 유사 모델들이 주목받고 있으며, 이들 모델은 Chain-of-Thought (CoT) 추리 단계의 길이를 늘려 기존의 대형 언어 모델(LLM)의 추리 능력을 향상한다. 본 논문에서는 이러한 긴 CoT의 품질을 이해하고 기존 LLM이 해당 CoT에서 오류를 감지하는 능력을 평가하기 위해 DeltaBench를 도입하였다. DeltaBench는 다양한 o1 유사 모델(QwQ, DeepSeek-R1)로부터 생성된 긴 CoT를 포함하여 여러 추리 작업(수학, 프로그래밍, 일반 추리)에 대해 평가할 수 있도록 구성되었다.

- **Technical Details**: DeltaBench 데이터셋은 1,236개의 샘플로 구성되어 있으며, 각 샘플은 문제, 긴 CoT 솔루션, 인간 주석을 포함하고 있다. 각 긴 CoT는 여러 섹션으로 나뉘며, 섹션은 전략 변화, 추리 유용성, 추리 정확성 및 반사 효율성과 같은 태그로 주석이 달린다. 이 데이터는 다양한 오픈 소스 데이터셋에서 추출된 질문들로 구성되어 있으며, 클러스터링, 중복 제거 및 난이도 필터링 과정을 거쳐 데이터의 다양성과 균형을 확보하였다.

- **Performance Highlights**: DeltaBench를 통해 기존 LLM 및 프로세스 보상 모델(PRM)의 오류 발견 능력이 제한적임을 발견하였다. 예를 들어, DeltaBench에서 가장 성능이 뛰어난 모델인 GPT-4-turbo-128k는 F1-score가 40.8%에 불과하다. 또한, 기존의 o1 유사 모델들은 비 유사 모델에 비해 개선된 비판 능력을 보이지 않았으며, 자기 비판 능력 역시 두드러지게 낮은 성능을 보였다.



### Controlled Diversity: Length-optimized Natural Language Generation (https://arxiv.org/abs/2502.19347)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이번 논문에서는 LLMs(대규모 언어 모델)가 엄격한 길이 요구사항에 따라 출력 길이를 조정할 수 있는 방법을 제시하고 있습니다. 이 요구사항을 충족하는 응답을 생성하기 위한 모델을 훈련시키기 위해 데이터 증강(data augmentation)과 기존의 파인튜닝(fine-tuning) 기법을 활용합니다. 이러한 접근 방식은 LLM이 다양한 사용자와 시스템 요구사항에 맞춰 더 유용하게 적용될 수 있도록 합니다.

- **Technical Details**: 우리는 LLM이 길이 요구사항을 준수하도록 훈련시키기 위해 Supervised Fine-Tuning(SFT) 외에도 Proximal Policy Optimization(PPO), Direct Preference Optimization(DPO), Odds Ratio Preference Optimization(ORPO)와 같은 여러 강화 학습(reinforcement learning) 기법을 적용합니다. 이러한 방법들은 LLM이 인간 피드백(human feedback)에서 학습할 수 있도록 지원하며, 자동으로 측정된 성능 지표를 최적화하는데도 활용됩니다. 또한 단일 훈련 데이터 세트를 사용하여 길이를 기준으로 측정이 가능하도록 하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방식으로 훈련된 LLM은 기존의 기준 모델에 비해 길이 요구사항을 더 잘 준수하는 텍스트를 생성할 수 있음을 보여줍니다. 길이 목표를 보조 목표로 추가하고, LLM이 주 작업을 학습하면서 길이를 조정할 수 있는 가능성을 확인했습니다. 이는 예를 들어 사용자 지정 길이의 요약을 간소화하여 복잡한 콘텐츠 접근성을 향상시키는 응용 사례에서도 유망한 결과를 보여주고 있습니다.



### Evaluating LLMs and Pre-trained Models for Text Summarization Across Diverse Datasets (https://arxiv.org/abs/2502.19339)
Comments:
          5 pages, 2 figures, 6 tables

- **What's New**: 이 연구는 자연어 처리에서 입력된 방대한 텍스트를 간결하고 일관된 요약으로 압축하는 텍스트 요약 기술을 강조합니다. 특히 BART, FLAN-T5, LLaMA-3-8B, Gemma-7B라는 네 가지 사전 훈련된 대형 언어 모델에 대한 평가를 제공합니다. 이러한 모델들은 CNN/DM, Gigaword, News Summary, XSum, BBC News의 다양한 데이터셋에서 성능을 비교하여 연구의 핵심 기여를 합니다.

- **Technical Details**: 이 연구에서 사용된 네 가지 모델은각각 BART, FLAN-T5, LLaMA-3-8B, Gemma-7B로,  다양한 데이터셋에서 추상적 요약을 위해 미세 조정(fine-tuning)되었습니다. 모델의 성능 평가는 ROUGE, BERTScore 및 METEOR와 같은 여러 자동 정량적 메트릭을 통해 이루어졌습니다. 모델은 각각 트랜스포머 기반 아키텍처를 활용하여 요약 품질을 극대화하는 데 중점을 두었습니다.

- **Performance Highlights**: 평가 결과 나타난 바에 따르면, 각 모델의 강점과 한계가 특정 텍스트 유형 처리에 따라 다르게 나타났습니다. ROUGE 메트릭을 통해 평가된 F1 점수에서 모델들이 상이한 성능을 보이며, 그 결과는 요약 품질 개선을 위한 건설적인 통찰을 제공합니다. 이러한 결과는 향후 연구 및 향상 작업에 중요한 기초 자료로 활용될 것입니다.



### Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems (https://arxiv.org/abs/2502.19328)
Comments:
          16 pages, 5 figures

- **What's New**: 이 논문에서는 agentic reward modeling을 제안하며, 이는 기존의 보상 모델(reward models)과 검증 가능한 정확성 신호(verifiable correctness signals)를 통합한 새로운 보상 체계입니다. 특히, 사실성(factuality) 및 지침 준수(instruction following)와 같은 두 가지 신호를 강조하여 보다 신뢰할 수 있는 보상을 제공합니다. 기존의 RMs는 주로 인간의 선호(human preferences)에 초점을 맞추어 주관적 편향(subjective biases)을 가질 수 있는 반면, 이 방법론은 다양한 측면에서의 정확성 신호를 반영하여 높은 신뢰성을 유지합니다.

- **Technical Details**: RewardAgent라는 보상 에이전트를 구현하여 인간 선호 기반의 기존 보상 모델을 사실성과 지침 준수 신호와 결합했습니다. RewardAgent는 세 가지 주요 모듈로 구성되어 있습니다: Router, Verification Agents, Judger. Router는 적절한 검증 에이전트를 결정하고, Verification Agents는 다양한 측면에서 응답의 정확성을 평가하며, Judger는 이들 검증 신호와 인간 선호 점수를 통합하여 최종 보상 점수를 제공합니다.

- **Performance Highlights**: RewardAgent는 RM-Bench, JudgeBench와 같은 여러 보상 모델 벤치마크에서 상당한 성능 향상을 보였으며, 이를 통해 전통적인 보상 모델에 비해 더 우수한 응답을 선택하는 능력을 입증했습니다. 또한, 실제 세계의 다운스트림 작업에서 주요 NLP 벤치마크에서 뛰어난 성능을 달성하여, RewardAgent로 구축된 데이터를 사용하는 경우 DPO 훈련에 따라 LLM의 성능이 향상됨을 보여주었습니다. 전반적으로, 이 연구는 보다 신뢰할 수 있는 보상 시스템 개발을 위한 새로운 가능성을 제시합니다.



### Shh, don't say that! Domain Certification in LLMs (https://arxiv.org/abs/2502.19320)
Comments:
          10 pages, includes appendix Published in International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 도메인 인증(domain certification) 프레임워크를 소개합니다. 이 프레임워크는 모델이 특정 도메인에서 벗어난 출력을 생성할 확률을 수학적으로 보장합니다. 또한, VALID라는 알고리즘을 제안하여 공격(adversarial attack) 하에서 모델이 주제에서 벗어나지 않도록 하는 저렴한 방법으로 검증된 적대적 LLM 출력을 생성합니다.

- **Technical Details**: 도메인 인증 프레임워크는 모델이 주어진 대상 도메인을 넘어서는 출력을 할 확률에 대한 상한을 제시합니다. VALID는 이 보장을 준수하는 시스템을 구축하는 데 사용하는 간단한 방법입니다. 이 방법에서는 입력과 출력의 토큰(token) 시퀀스를 수학적으로 정의하고, 이를 통해 이론적 근거를 통해 도메인 인증 성과를 이끌어내는 기초 요소를 제시합니다.

- **Performance Highlights**: VALID는 다양한 데이터셋에서 평가되어 의미 있는 인증서를 생성함으로써, 도메인 외 샘플이 발생할 확률을 최소한의 거부 행동(refusal behavior)으로 조정할 수 있다는 것을 보여주었습니다. 연구 결과, VALID는 공격에 대한 강력한 저항력을 제공하며, LLM 기반 시스템의 도메인 제한 필요성을 강조합니다.



### CritiQ: Mining Data Quality Criteria from Human Preferences (https://arxiv.org/abs/2502.19279)
- **What's New**: 이번 논문에서는 데이터 품질을 위한 기준(criteria)을 자동으로 발굴할 수 있는 새로운 데이터 선택 방법인 CritiQ를 소개합니다. CritiQ는 단 30개의 인간 주석 쌍만으로 고품질 데이터를 선택할 수 있으며, 이는 기존의 수작업 방식에 비해 더 효과적입니다. CritiQ Flow라는 주요 구성 요소를 통해 데이터를 품질 기준에 따라 진화시키고, 데이터 선택을 진행합니다.

- **Technical Details**: CritiQ Flow는 관리 에이전트(manager agent)와 작업자 에이전트(worker agents)를 사용하여 쌍별 판단을 수행합니다. 이를 통해 이전 연구에서 추출한 품질 기준을 바탕으로 지식 기반(knowledge base)을 구축하여 CritiQ Flow의 성능을 향상시킵니다. 이후 CritiQ Scorer를 학습시켜 품질 점수를 매기고 효율적인 데이터 선택을 실행합니다.

- **Performance Highlights**: 코드, 수학 및 논리 분야에서 CritiQ의 효과성을 입증했으며, 인간 주석 테스트 세트에서 높은 정확도를 달성했습니다. Llama 3.1 모델을 지속적으로 학습시켜 uniform sampling에 비해 다운스트림 작업에서 성능 향상을 관찰했습니다. 또한, 아블레이션 연구(ablation studies)를 통해 지식 기반과 반영 과정(reflection process)의 이점을 검증했습니다.



### Disentangled VAD Representations via a Variational Framework for Political Stance Detection (https://arxiv.org/abs/2502.19276)
- **What's New**: 이번 연구에서는 유권자들의 감정 정보 통합 문제를 해결하기 위해 새로운 변별 오토인코더(Variational Autoencoder, VAE) 기반 스탠스 탐지 프레임워크인 PoliStance-VAE를 제안합니다. 이 프레임워크는 정치적 담론에서 감정 특성인 valence, arousal, dominance (VAD)를 분리하여 스탠스 탐지의 정확도를 높입니다. 최신 감정 주석 도구를 활용하여 다중 클래스 감정 레이블을 부여함으로써 더욱 세밀한 감정 정보를 모델에 통합합니다. 이러한 접근 방식은 현존하는 스탠스 탐지 방법의 한계를 극복하며 자연어 처리(NLP) 작업의 발전을 위한 기틀을 마련합니다.

- **Technical Details**: PoliStance-VAE는 정치적 스탠스 탐지에서 VAD 특성을 분리하는 데 초점을 맞춘 VAE 기반 아키텍처입니다. 이 모델은 감정 분류를 보조 작업으로 포함하여, 데이터셋의 트윗(tweets)에서 감정 정보를 효과적으로 추출합니다. 연구 팀은 P-STANCE 및 SemEval-2016과 같은 벤치마크 데이터셋에 대한 종합적인 평가를 수행하여 이 모델의 성능을 검증했습니다. 특히, 감정 라벨의 세분화된 주석이 모델 성능 향상에 중요한 역할을 한다는 것을 입증했습니다.

- **Performance Highlights**: PoliStance-VAE는 현재까지의 최고 성능을 나타내며, BERT, BERTweet, GPT-4o와 같은 기존 모델들을 초월하는 결과를 달성했습니다. 이 연구 결과는 폴리 스탠스 탐지 작업에서 감정 표현의 통합이 얼마나 효과적인지를 증명하며, 감정 이해가 필요한 다른 NLP 작업에서의 응용 가능성을 보여줍니다. 특히, 정보의 세부적 감정 표현을 통합함으로써 더 나은 스탠스 탐지의 가능성을 제시합니다.



### Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization (https://arxiv.org/abs/2502.19261)
Comments:
          To appear at the 13th International Conference on Learning Representations (ICLR 2025)

- **What's New**: 이번 연구는 Drop-Upcycling이라는 새로운 방법론을 제안하여 Mixture of Experts (MoE) 모델의 훈련을 최적화하였다. Drop-Upcycling은 사전 훈련된 밀집 모델의 지식을 이용하면서 일부 매개변수를 통계적으로 재초기화하는 접근 방식을 결합하여 전문가의 전문화(expert specialization)를 촉진한다. 이 방법은 기존의 Upcycling 접근 방식이 가진 한계를 극복하며 장기적으로 MoE 모델의 효율성을 크게 향상시킨다.

- **Technical Details**: Drop-Upcycling은 밀집 모델을 MoE 모델로 확장할 때 전문가의 feedforward network (FFN) 매개변수의 선택적 재초기화(selective re-initialization)로 동작한다. 구체적으로, FFN의 중간 차원에서 공통 인덱스를 임의로 샘플링하여 매개변수를 열(column)별 또는 행(row)별로 드롭한 후, 드롭된 매개변수를 통계치를 이용해 재초기화한다. 이 접근 방식은 훈련 초기 단계에서 더 나은 상태에서 시작할 수 있게 도와주며, 긴 훈련 기간 동안 관찰되는 수렴 느림(convergence slowdowns)을 피하도록 설계되었다.

- **Performance Highlights**: 드롭 업사이클링의 성과는 5.9B 활성 매개변수를 가지는 MoE 모델이 동일 모델 가족의 13B 밀집 모델과 유사한 성능을 나타내며, 훈련 FLOPs는 약 1/4에 불과하다는 점이다. 대규모 실험을 통해 드롭 업사이클링이 이전 MoE 구축 방법들과 비교했을 때 장기 훈련 시에도 눈에 띄게 더 우수한 성과를 보여주었다. 이 연구에 대한 모든 실험 결과는 공개되어 있어 재현성과 향후 연구에 기여할 수 있다.



### Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases (https://arxiv.org/abs/2502.19249)
- **What's New**: 이 논문은 형식 언어(formal languages)에서 사전 훈련(pretraining)을 통해 자연어(natural language)의 습득을 효과적으로 향상시킬 수 있다는 가설을 제안합니다. 특히, 자연어에서의 효율적인 전이를 가능하게 하는 형식 언어의 특성에 대한 통찰력을 바탕으로 하여, 종속 구조(dependency structures)를 포착하고 모델 아키텍처의 계산 제한 내에서 유지되는 형식 언어가 효과적인 전이를 가능하게 한다고 주장합니다.

- **Technical Details**: 연구에서는 Transformer 모델을 중심으로, 주어진 형식 언어가 자연어에 적합할 때 언어 모델이 낮은 손실(loss)과 더 나은 언어적 일반화(linguistic generalization)를 달성한다고 설명합니다. 핵심적으로, 사전-사전 훈련(pre-pretraining) 방식이 정형 언어에 대해 훈련된 후 자연어로 훈련한 경우 더 낮은 손실과 더 나은 일반화 성능을 이끌어낸다고 보고합니다.

- **Performance Highlights**: 1B 파라미터의 언어 모델이 자연어 약 16억 토큰을 훈련한 결과, 사전-사전 훈련을 통해 33% 적은 토큰 예산으로도 유사한 손실 및 향상된 언어적 일반화를 달성함을 보여줍니다. 또한, 정형 언어 사전 훈련 과정에서 얻어진 주의 헤드(attention heads)가 자연어 평가에서 모델 성능에 중요한 요소로 작용하며, 긍정적 전이를 이끌어낸다고 mechanistic evidence를 통해 제시합니다.



### Two Heads Are Better Than One: Dual-Model Verbal Reflection at Inference-Tim (https://arxiv.org/abs/2502.19230)
- **What's New**: 이번 논문에서는 복잡한 추론 시나리오에서 발생하는 대형 언어 모델(LLM)의 한계를 극복하기 위해 새로운 대조적 반사 합성 파이프라인을 제안합니다. 이는 LLM이 생성한 반사의 정확성과 깊이를 향상시키며, 교육 현장에서 비구조적으로 제공되는 피드백의 필요성을 해결하는데 중점을 둡니다. 또한, 이 논문에서는 특화된 Reasoner와 Critic 모델을 포함하는 이중 모델 추론 프레임워크를 제시하여, LLM의 자기 반성과 세분화된 피드백 과정을 분리합니다.

- **Technical Details**: 제안된 DARS(Dual-model Reflective Scoring) 프레임워크는 Reasoner 모델이 추론을 수행하는 반면, Critic 모델은 그 과정에 대한 반사를 제공하여 상호보완적인 기능을 수행합니다. 이 접근법은 명시적인 인간 레이블 없이도 전체적인 추론 결과의 정확성을 검증할 수 있도록 설계되었습니다. 반사 합성 파이프라인은 다단계 추론 경로에서 발생하는 불일치를 분석하여 오류 수정 지침을 생성하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, DARS 프레임워크는 기존의 단일 Reasoner 기반 선호 최적화 방법에 비해 모든 평가 지표에서 우수한 성능을 나타냅니다. 특히, Critic이 제공하는 반사는 Reasoner가 신뢰할 수 있는 행동 지침을 제공하는 데 도움을 주며, Critic 모델의 크기를 키우면 성능이 향상되는 경향을 보였습니다. 이 연구는 두 개의 모델이 협력하여 보다 나은 추론 성과와 투명성을 달성할 수 있음을 보여줍니다.



### Negation-Induced Forgetting in LLMs (https://arxiv.org/abs/2502.19211)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 인지적 현상인 부정 유도 망각(negation-induced forgetting, NIF)을 보여주는지 여부를 탐구합니다. NIF는 부정적인 정보를 처리할 때 기억이 손상되는 현상으로, 인간에서 관찰된 바 있습니다. 이 연구는 ChatGPT-3.5, GPT-4o mini, Llama3-70b-instruct를 대상으로 실험을 실시하여 LLMs에서 NIF의 존재 여부를 분석했습니다.

- **Technical Details**: 연구에서는 Zang et al. (2023)의 실험 프레임워크를 바탕으로, LLM에 대한 부정 유도 망각(NIF)을 검증하기 위해 다양한 대화형 AI 모델을 비교했습니다. 실험은 ChatGPT-3.5의 대화 기록을 사용하여 진행되었으며, 참가자들은 주어진 이야기의 내용을 읽고 질문에 답하는 방식으로 진행되었습니다. 분석은 빈도수 기반 혼합효과 모델(binomial generalized linear mixed-effects model, GLMM)을 통해 수행되었으며, 부정적인 진술의 기억 회수에서 통계적 차이를 확인했습니다.

- **Performance Highlights**: ChatGPT-3.5는 NIF를 나타냈으며 부정된 정보의 회수가 긍정적으로 확정된 정보보다 낮았습니다. GPT-4o-mini는 한계적인 NIF 효과를 보여주었으나, LLaMA-3-70B는 NIF를 나타내지 않았습니다. 이 연구 결과는 일부 LLM에도 부정 유도 망각 현상이 존재함을 제시하며, 이러한 인지적 편향이 LLM의 기억 관련 현상 이해에 기여할 수 있음을 시사합니다.



### Bi'an: A Bilingual Benchmark and Model for Hallucination Detection in Retrieval-Augmented Generation (https://arxiv.org/abs/2502.19209)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 시스템에서 발생하는 환각(hallucination) 문제를 해결하기 위한 새로운 프레임워크인 Bi’an을 소개합니다. Bi’an은 이중 언어 벤치마크 데이터셋과 경량의 판단 모델로 구성되어 있습니다. 이 데이터셋은 다양한 RAG 시나리오에서 엄격한 평가를 지원하며, 경량 모델은 오픈 소스 LLM에서 미세 조정되어 생성됩니다.

- **Technical Details**: Bi’anBench 데이터셋은 질문 응답, 요약, 데이터-투-텍스트, 기계 번역의 네 가지 RAG 시나리오를 기반으로 하여 22,992개의 테스트 케이스를 포함합니다. 또한, LLM-as-a-Judge 접근 방식을 통해 RAG 시스템의 출력을 평가할 수 있도록 구현되었습니다. 이 과정에서 DPO(Direct Preference Optimization) 기법을 이용하여 모델 학습이 이뤄집니다.

- **Performance Highlights**: 실험 결과, 14B 모델이 매개변수가 다섯 배 이상 큰 기준 모델들보다 우수한 성능을 보였으며, 최신의 닫힌 소스 LLM인 GPT-4o의 성능과 근접하게 나타났습니다. 이러한 성능 향상은 실험에서 각 구성 요소의 효과가 입증되었습니다.



### MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer's Detection (https://arxiv.org/abs/2502.19208)
Comments:
          11 pages, 3 Figures

- **What's New**: 이번 연구에서는 다국어 대화 기반 치매(Alzheimer's disease, AD) 탐지를 위한 새로운 데이터셋인 MultiConAD를 소개합니다. 이 데이터셋은 영어, 스페인어, 중국어, 그리스어를 포함하는 16개의 공개된 대화 데이터셋을 통합하여 제작되었습니다. 또한, 이 연구는 경도인지장애(Mild Cognitive Impairment, MCI)를 포함한 보다 세밀한 분류 접근법을 적용하여, 대화 데이터를 활용한 AD 탐지를 향상시키려는 노력을 하고 있습니다.

- **Technical Details**: 다오가 포함된 MultiConAD 데이터셋은 텍스트와 오디오 데이터를 모두 통합하여, 다양한 인지 평가 과제를 포함합니다. 이 데이터셋은 단일 언어 및 다국어 환경에서 실험을 수행하여, 각각의 언어가 다국어 학습에서 어떤 이점을 가지는지를 연구하였습니다. 특히, 다국어 데이터에 대한 실험을 통해 단순한 바이너리 분류(AD vs. 건강한 대조군) 외에 MCI를 구분할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, 일부 언어는 다국어 학습을 통해 인지 저하의 공동 마커를 공유하는 것으로 나타났습니다. 반면 다른 언어는 독립적으로 훈련될 때 더 좋은 성능을 보였습니다. 이러한 발견은 AD 탐지 모델을 특정 언어에 맞추어 최적화할 필요성을 강조하며, 다국어 학습의 잠재적 이점을 증명합니다.



### FaithUn: Toward Faithful Forgetting in Language Models by Investigating the Interconnectedness of Knowledg (https://arxiv.org/abs/2502.19207)
Comments:
          16 pages

- **What's New**: 이 논문에서는 언러닝(unlearning) 방법이 상호연관된 지식을 신뢰성 있게 제거하지 못할 수 있다는 문제를 강조합니다. 저자들은 'superficial unlearning'이라는 새로운 개념을 정의하고, 언러닝이 효과적으로 수행되고 있는지 평가하기 위한 새로운 벤치마크인 FaithUn을 도입하였습니다. 이 벤치마크는 현실 세계의 지식 Q&A 설정에서의 언러닝 신뢰성을 분석하는 데 중점을 두고 있습니다.

- **Technical Details**: 저자들은 Knowledge-Localized UnlEarning, 즉 KLUE라는 새로운 언러닝 방법론을 제안합니다. KLUE는 특정 지식과 관련된 뉴런만을 업데이트하여 보다 신뢰성 있는 언러닝을 달성합니다. 이를 위해 저자들은 설명 가능성(explainability) 방법을 사용하여 어떤 뉴런을 업데이트할지 결정하고, 선택된 정보의 유지를 통해 불필요한 지식이 무의식적으로 제거되는 것을 방지합니다.

- **Performance Highlights**: 실험 결과, 기존의 언러닝 방법들이 신뢰성 있는 언러닝을 보장하지 못하는 반면, KLUE는 FaithUn 설정에서 기존 방법들보다 우수한 성능을 보였습니다. 이는 지식 기반의 언러닝이 효과적으로 수행될 수 있음을 시사하며, 앞으로의 연구 방향에 새로운 질문을 제기합니다. 저자들은 이 연구를 통해 언러닝 분야에서 복잡하고 상호연관된 지식의 중요성을 재조명하고 있습니다.



### LiGT: Layout-infused Generative Transformer for Visual Question Answering on Vietnamese Receipts (https://arxiv.org/abs/2502.19202)
Comments:
          Accepted at IJDAR

- **What's New**: 이번 연구는 베트남어로 제공되는 최초의 대규모 문서 시각 질문 응답(dataset for Document Visual Question Answering) 데이터셋인 ReceiptVQA를 소개합니다. ReceiptVQA는 상업적 잠재력이 높은 영수증을 기반으로 하며, 9,000개 이상의 영수증 이미지와 60,000개 이상의 수동으로 주석이 달린 질문-답변 쌍으로 구성되어 있습니다. 이 연구는 문서 VQA 분야에서 베트남어 커뮤니티의 연구 발전을 촉진할 것으로 기대합니다.

- **Technical Details**: 우리는 LiGT(Layout-infused Generative Transformer)라는 새로운 인코더-디코더 архитект출을 개발했습니다. 이 모델은 레이아웃 이해 능력을 통합하여 추가적인 신경 모듈의 사용을 최소화했습니다. 모델은 OCR(Optical Character Recognition) 텍스트 위치를 해시 값으로 변환하여 2D 레이아웃 표현을 생성하며, 이는 문서 도메인의 텍스트 밀집 특성을 반영합니다.

- **Performance Highlights**: ReceiptVQA 실험에서 LiGT 아키텍처는 경쟁력 있는 성능을 보여주었습니다. 인코더 전용 모델 아키텍처에 비해 답변을 생성할 수 있는 아키텍처가 우수한 결과를 나타냈으며, 언어 모델의 의미 이해와 다양한 모달리티의 결합이 데이터셋의 해결에 중요함을 발견했습니다.



### BIG-Bench Extra Hard (https://arxiv.org/abs/2502.19187)
- **What's New**: 이 논문에서는 기존의 BIG-Bench Hard(BBH) 한계점을 극복하기 위해 BIG-Bench Extra Hard(BBEH)라는 새로운 벤치마크를 소개합니다. BBEH는 기존 BBH의 23개 과제를 새로운 과제로 대체하여 비슷한 추론 능력을 측정하되, 현저히 높은 난이도를 부여했습니다. 이는 LLM의 일반적인 추론 능력을 평가하는 데 필수적인 과제이며, 더 넓은 범위의 사고 능력을 요구합니다.

- **Technical Details**: BBEH는 문제 해결에 있어 수백 자에 달하는 긴 입력을 처리하도록 요구하며, 많은 단계의 추론(many-hop reasoning), 즉흥 학습(learning on the fly), 오류 찾기(find errors in reasoning traces)와 같은 좀 더 발전된 추론 기술을 평가합니다. 각 과제는 LLM이 복잡한 패턴을 유도하는 것과 같은 여러 기술을 요구하며, 과제를 해결하는 데 필요한 인지적 역량이 크게 확대되었습니다.

- **Performance Highlights**: 최신 일반 모델은 BBEH에서 23.9%의 정확도를 달성하였고, 추론 전문 모델은 54.2%의 정확도를 기록하였습니다. 이는 LLM의 일반적인 추론 능력 향상에 대한 잠재력이 여전히 크게 남아있음을 나타냅니다. 또한, 세부 실패 분석을 통해 일반 모델과 추론 모델 모두에서 흥미로운 실패 양상을 발견하였으며, 이는 향후 연구에 기여할 수 있는 중요한 인사이트를 제공합니다.



### MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis (https://arxiv.org/abs/2502.19175)
- **What's New**: 이번 연구에서는 Differential Diagnosis (DDx) 프로세스를 지원하기 위한 새로운 접근법인 Modular Explainable DDx Agent (MEDDxAgent) 프레임워크를 소개합니다. 기존의 DDx 지원 시스템들은 제한된 데이터 셋 평가, 구성 요소의 고립된 최적화 및 비현실적인 가정에 직면해 있었습니다. MEDDxAgent는 이러한 한계를 극복하며, 환자 프로파일이 완전하게 제공되기 전에도 진단적 추론을 발전시킬 수 있습니다.

- **Technical Details**: MEDDxAgent는 세 가지 모듈형 구성 요소로 구성되어 있습니다: (1) 진단 과정을 조정하는 DDxDriver, (2) 병력 조사 시뮬레이터, (3) 지식 검색 및 진단 전략을 위한 두 개의 전문 에이전트. 이 프레임워크는 호흡기 질환, 피부 질환 및 드문 질환을 포함한 포괄적인 DDx 벤치마크를 통해 신뢰성 있는 평가를 보장합니다.

- **Performance Highlights**: MEDDxAgent는 대규모 및 소규모 LLMs를 통해 10% 이상의 정확도 개선을 달성하였으며, 지속적인 학습 과정을 통해 진단적 추론의 투명성을 제공합니다. 이 연구는 또한 초기 환자 프로파일이 없을 때의 반복적 개선의 중요성을 강조하고 있습니다.



### TestNUC: Enhancing Test-Time Computing Approaches through Neighboring Unlabeled Data Consistency (https://arxiv.org/abs/2502.19163)
- **What's New**: 이 논문에서는 Test-time computing(테스트 타임 컴퓨팅) 접근 방식을 통해 LLM의 성능을 향상시키는 새로운 방법인 TestNUC를 소개합니다. TestNUC는 인접한 비정답 데이터의 지역적 일관성을 활용하여 예측을 개선하며, 비정답 인스턴스의 예측을 고려하여 입력 인스턴스를 분류합니다. 이 방법은 여러 데이터 세트에서 기존 방법들보다 우수한 성능을 보이며, 테스트 타임 컴퓨팅에 통합할 수 있는 가능성이 높습니다.

- **Technical Details**: TestNUC는 두 가지 주요 단계로 구성됩니다: ❶ Neighbor Retrieval(이웃 검색) 단계에서, 테스트 샘플과 유사한 특징을 가진 K개의 이웃을 식별합니다. ❷ Collaborative Prediction(협력적 예측) 단계에서는 LLM이 테스트 샘플과 그 이웃들에 대한 예측을 생성하고, 이들의 예측을 조합하여 최종 답변을 도출합니다. 이는 LLM이 비정답 샘플의 예측을 포함하여 의사결정을 더 잘 맥락화하고 세분화할 수 있도록 도움을 줍니다.

- **Performance Highlights**: TestNUC는 감정 탐지, 도메인 발견, 주제 채굴, 의도 분류 등 다양한 작업에서 평가되었으며, 기본 방법들인 표준 프롬프트 및 자기 일관성 방식보다 일관되게 더 나은 성능을 발휘했습니다. 특히, 비정답 데이터의 양이 증가할수록 성능이 효과적으로 확장되었고, 다양한 임베딩 모델에서도 강력한 성능을 보였습니다. 또한, TestNUC는 기존의 테스트 타임 컴퓨팅 방법들과 원활하게 통합되며, 성능을 크게 향상시킬 수 있습니다.



### Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models (https://arxiv.org/abs/2502.19160)
- **What's New**: 이번 연구에서는 언어에서 고정관념을 탐지하고 정량화하는 새로운 접근 방식을 제안합니다. 이는 사회 범주와 고정관념 커뮤니케이션(Social Category and Stereotype Communication, SCSC) 프레임워크에 기반하여 고정관념의 언어적 지표를 도출합니다. 이 방법은 다양한 대형 언어 모델(LLMs)을 활용하여 문장에서 이러한 지표를 자동으로 분류합니다.

- **Technical Details**: 연구는 LLM을 이용해 고정관념의 언어적 지표를 탐지하고 정량화하는데 초점을 맞춥니다. 접근 방식은 문장의 언어적 특성을 검토하고 세분화된 평가를 위한 기초를 제공합니다. 유의미한 언어 지표의 중요성을 평가하여 고정관념의 언어적 지표를 측정하는 점수 함수를 학습합니다.

- **Performance Highlights**: 모델의 성능 평가 결과, 일반적으로 고정관념의 언어적 지표를 탐지하고 분류하는 데 우수한 성능을 보였습니다. 그러나 일부 모델은 관련 행동 및 특성을 정확하게 평가하는 데 어려움을 겪었습니다. 더 많은 few-shot 예제를 프롬프트에 포함시키면 성능이 크게 향상되는 것으로 나타났습니다.



### When Personalization Meets Reality: A Multi-Faceted Analysis of Personalized Preference Learning (https://arxiv.org/abs/2502.19158)
- **What's New**: 이번 연구에서는 인간 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback, RLHF)의 한계를 극복하기 위해 개인화된 선호 학습(personalized preference learning)이 필요하다는 점을 강조합니다. 기존의 RLHF 기법이 사용자 간 동질적인 선호를 전제로 하여 다양한 인구 집단의 가치관을 간과한 것을 지적하고, 개인화된 선호를 적용하는 데 대한 방법론적 정당성을 제시합니다. 본 연구는 개별 사용자의 다양한 선호에 맞춰 LLM을 적응시키기 위한 포괄적인 평가 프레임워크를 소개합니다.

- **Technical Details**: 연구팀은 개인화된 선호 학습 기술을 벤치마킹할 수 있는 다각적(evalution framework) 평가 프레임워크를 제안했습니다. 이 프레임워크는 모델의 성능, 공정성(fairness), 비의도적 효과(unintended effects), 적응성(adaptability)을 측정할 수 있는 다양한 지표를 포함합니다. 아울러, 사용자의 데이터 가용성을 다양하게 고려하여 평가할 수 있도록 설계되었습니다. 이 연구는 8개의 개인화 방법을 이용해 3개의 서로 다른 선호 데이터 세트를 통해 광범위한 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 사용자 간의 강한 비동의(disagreement) 상황에서 성능 차이가 최대 36%에 달할 수 있으며, 개인화가 최대 20%의 안전성과 추론 능력을 저하시킬 수 있음을 보여주었습니다. 연구는 개인화의 잠재적 부작용이 LLM의 일반적인 능력을 저하시킬 수 있다는 점을 강조하고 있습니다. 특히, 개별 사용자에게 맞춤형 보상 모델을 파인튜닝(fine-tuning)하는 것이 효과적인 방법임을 발견하였으며, 협업 학습을 활용한 방법들이 이 baseline에 비해 최고 6%의 성능 향상을 보였습니다.



### Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs (https://arxiv.org/abs/2502.19148)
Comments:
          Accepted by ICLR 2025, Project page: this https URL

- **What's New**: 이 논문은 사용자 선호도를 효율적으로 조정하기 위한 새로운 접근법인 Amulet를 소개합니다. 전통적인 방법들이 정적 데이터셋에 의존하는 반면, 이 프레임워크는 사용자 제공 프롬프트(prompt)를 활용해 실시간으로 최적화를 수행합니다. 이러한 점에서 Amulet는 기존의 대형 언어 모델(LLM)과는 다른, 개인화된 사용자 경험을 제공할 수 있는 가능성을 보여줍니다.

- **Technical Details**: Amulet의 주요 특징은 매 토큰의 디코딩(decoding) 과정을 별도의 온라인 학습 문제로 설정하는 것입니다. 이 프레임워크는 사용자 제공 프롬프트를 통해 각 토큰의 최적화를 안내하며, 반복(iteration) 단계의 최적화를 위해 닫힌 형태의 해(closed-form solution)를 제공합니다. 이러한 방법은 각 토큰의 최적화 과정에서 컴퓨팅 비용을 최소화하여 효율성을 높입니다.

- **Performance Highlights**: 실험 결과 Amulet는 다양한 LLM, 데이터셋, 사용자 선호도가 결합된 환경에서 뚜렷한 성능 향상을 보여주었습니다. 또한, 이 프레임워크는 실시간 개인화 최적화라는 특성을 유지하면서도 수용 가능한 컴퓨팅 효율성을 가지고 있습니다. 이를 통해 LLM의 사용성을 더욱 향상시키고, 변경되고 다양한 사용자 선호도에 능동적으로 대응할 수 있는 방법을 제시합니다.



### Self-Memory Alignment: Mitigating Factual Hallucinations with Generalized Improvemen (https://arxiv.org/abs/2502.19127)
Comments:
          29 pages, 17 figures

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 사실과 일치하는 응답을 생성하는데 어려움을 겪는 "사실적 환각(factual hallucinations)" 문제를 직접 해결하기 위한 새로운 방법론을 제안합니다. 우리는 모델의 바로 자신이 생성한 응답을 활용하여 정량적 질문에 대해 최적화하는 "자기 기억 정렬(self-memory alignment, SMA)" 방식을 도입했습니다. 이를 통해 LLM이 기존 메모리를 보다 정확히 활용하도록 개선하고, 21개의 도메인에 걸친 18만 1천 개의 데이터로 구성된 정밀한 사실 질의응답 데이터셋(FactualBench)도 구축합니다.

- **Technical Details**: SMA는 이미지 대결을 구성하여 모델이 정확한 사실 기반 응답을 파악하도록 훈련됩니다. 이는 전통적 방법과 달리 외부 소스 기반의 튜닝 세트를 사용하지 않고 LLM의 자체 응답을 샘플링하여 직접적인 선호 최적화(Direct Preference Optimization, DPO)를 통해 이루어집니다. 이를 통해 정보 왜곡, 부정확한 판단 및 메모리 사용 측면에서의 문제를 해결할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 광범위한 실험 결과, SMA는 FactualBench와 다른 여섯 가지 벤치마크에서 LLM의 전반적인 성능을 유의미하게 향상시켰습니다. 평균적으로 기존 방식에 비해 4배에서 9배 증가된 성능을 보였으며, 사실성과 유용성과 같은 다양한 능력 평가에서도 일관된 향상을 나타냅니다. 이 연구는 LLM이 간단한 질의응답 교육만으로도 전반적인 향상을 경험할 수 있음을 보여주며, 기존 지식으로부터 재부스트되는 효과를 입증합니다.



### Improving customer service with automatic topic detection in user emails (https://arxiv.org/abs/2502.19115)
Comments:
          Paper submitted to the 15th International Conference on Information Society and Technology (ICIST), Kopaonik, Serbia, 9-12 March 2025

- **What's New**: 이 연구에서는 세르비아의 주요 통신사인 Telekom Srbija에서 고객 서비스 효율성을 높이기 위한 새로운 자연어 처리(Natural Language Processing) 파이프라인을 소개합니다. 이 파이프라인은 자동 이메일 주제 탐지와 라벨링을 통해 구현되며, 고객 서비스 운영에 혁신적인 변화를 가져올 것으로 기대됩니다.

- **Technical Details**: 중심에는 BERTopic이라는 모듈형 아키텍처가 있으며, 이는 비지도(topic modelling) 학습을 가능하게 합니다. 일련의 전처리(preprocessing) 및 후처리(post-processing) 단계를 거쳐, 우리는 12개의 주제 중 하나와 여러 추가 라벨을 들어오는 이메일에 할당하여 고객 서비스가 이를 필터링하고 접근할 수 있도록 합니다.

- **Performance Highlights**: 모델의 성능은 100개의 고객 이메일로 구성된 테스트 데이터 세트를 통해 자동으로 할당된 주제의 속도와 정확성을 평가하여 검토되었습니다. 이 파이프라인은 저자원(low-resourced) 및 형태소가 풍부한 언어에서도 폭넓은 적용 가능성을 보여주며, 현재 회사의 운영 환경에서 자동 이메일 분류를 통해 고객 서비스 운영을 간소화하고 있습니다.



### Conformal Linguistic Calibration: Trading-off between Factuality and Specificity (https://arxiv.org/abs/2502.19110)
- **What's New**: 본 논문에서는 Conformal Linguistic Calibration (CLC)라는 새로운 언어 모델 교정 접근 방식을 제안합니다. 이 방법은 언어 모델의 불확실성에 기초하여 모델 응답의 정확성을 조정할 수 있도록 설계되었습니다. CLC는 기존의 접근 방식인 abstention과 linguistic calibration을 통합하고, 언어의 실용적인 측면에서 두 가지 방법을 연결하는 틀을 제공합니다.

- **Technical Details**: CLC는 모델 응답의 세부사항을 조정하여 가능성이 있는 세계의 집합을 예측하는 방식으로 언어 모델의 불확실성을 포착합니다. 모델 응답이 정의한 가능 세계에 대한 진리 확률을 보장할 수 있도록 알고리즘을 제안하며, 이 과정은 언어의 불확실성을 탐색하고 신뢰성을 높이는 데 도움을 줍니다. 또한, 이러한 접근법은 사용자가 요구하는 확률적 보장을 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 CLC 접근 방식이 7B 모델에서 GPT-4o보다 더 높은 정확도를 달성하였고, 인기 있는 QA 데이터셋에서의 성능도 개선되었습니다. 불확실성을 인지하는 주장 재작성 도구를 통해 긴 형식에서의 신뢰성을 높이는데 기여하며, 데이터 세트와 모델도 공개합니다.



### Evaluating Gender Bias in German Machine Translation (https://arxiv.org/abs/2502.19104)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: WinoMTDE는 독일어 기계 번역(MT) 시스템에서 직업 편견과 과소 대표성을 평가하기 위해 설계된 새로운 성 편향 평가 테스트 세트입니다. 이 데이터셋은 성별과 고정관념에 대해 균형을 이루는 288개의 독일어 문장으로 구성되어 있으며, 독일 노동 통계에 따라 주석이 달렸습니다. 연구 결과, 대부분의 MT 모델에서 지속적인 성 편향이 발견되었고, 대형 언어 모델(LLM)이 전통적인 시스템보다 우수한 성능을 보였습니다.

- **Technical Details**: WinoMTDE 데이터셋은 Stanovsky et al. (2019)의 WinoMT 데이터를 바탕으로 하며, 독일어 문장을 Winograd 스키마의 구조에 맞춰 구성합니다. 각 문장은 뚜렷한 성별이 있는 주어와 그 반대 성별 주어를 포함하며, 종속절의 대명사가 주어를 지칭하도록 설계되었습니다. 이 데이터셋은 남성과 여성의 주제가 각각 144개씩 포함된 균형 잡힌 자료로, 독일 사회를 반영하기 위해 독일 노동부의 통계를 사용해 고정관념이 있는 직업의 분류가 이루어졌습니다.

- **Performance Highlights**: 대규모 평가를 통해 Google Translate, Microsoft Translator 등의 전통적인 MT 시스템과 GPT-4o-mini와 같은 대형 언어 모델을 분석했습니다. 결과적으로, 성별 일관성 및 정확한 번역에서 LLM이 전통적인 MT 시스템보다 높은 성과를 보였지만, 전체적으로는 여전히 성 편향이 지속적으로 나타나는 것으로 분석되었습니다. 이러한 결과는 기계 번역 모델의 구조와 교육 데이터 내의 체계적 편향에서 기인하였음을 보여줍니다.



### LongEval: A Comprehensive Analysis of Long-Text Generation Through a Plan-based Paradigm (https://arxiv.org/abs/2502.19103)
Comments:
          Under review

- **What's New**: 이 연구에서는 기존의 큰 언어 모델(LLMs)이 긴 문서 생성에서 어려움을 겪고 있음을 강조하며, 특별히 LongEval이라는 새로운 벤치마크를 도입하여 문서 생성을 평가합니다. LongEval은 인지 및 언어적 작문 모델에서 영감을 받아 직관적(direct) 생성과 계획 기반(plan-based) 생성을 모두 지원하며, LLMs의 긴 텍스트 생성 능력을 정량적으로 평가합니다.

- **Technical Details**: LongEval 벤치마크는 두 가지 주요 혁신을 특징으로 합니다. 첫째, 제로샷(zero-shot) 직접 생성과 계획 기반 구조 생성 모두를 평가하는 이중 평가 체계를 도입합니다. 둘째, 내용 품질, 구조적 일관성, 정보 밀도에 중점을 둔 신뢰할 수 있는 자동 평가 지표를 포함하여, 아카이브(arXiv) 논문, 블로그, 위키피디아 기사와 같은 세 가지 긴 텍스트 생성 도메인에서 LLMs의 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, LLM의 모델 크기와 생성 능력이 상관관계가 있지만, 작은 규모의 모델(예: LongWriter)이 잘 훈련된 긴 텍스트에서 유사한 성능을 보인다는 흥미로운 발견이 있었습니다. 또한, LLMs는 텍스트 길이가 1k 단어를 초과할 경우 성능이 크게 저하되며, 계획 기반 작문의 필요성이 강조됩니다.



### Sparse Brains are Also Adaptive Brains: Cognitive-Load-Aware Dynamic Activation for LLMs (https://arxiv.org/abs/2502.19078)
- **What's New**: 이번 연구에서는 Dense Large Language Models(LLMs)의 효율성을 향상시키기 위해 새로운 프레임워크인 CLADA(Cognitive-Load-Aware Dynamic Activation)를 제안합니다. CLADA는 인간의 뇌의 이중 프로세스 메커니즘에서 영감을 받아, 통계적 희소성(statistical sparsity)과 의미론적 적응성(semantic adaptability)을 통합합니다. 이 방식은 LLM이 문맥에 따라 적합한 활성화를 조절할 수 있도록 하며, 기존 방법들보다 더 나은 성능을 보입니다.

- **Technical Details**: CLADA는 전통적인 정적 프루닝(static pruning)이나 혼합 전문가 아키텍처(mixture-of-experts architectures)와 달리, 데이터의 시퀀스 정보와 인지 부하 지표(cognitive load metrics)를 바탕으로 동적으로 활성화 마스크를 조정합니다. 이 프레임워크는 40% 이상의 희소성을 보장하며, 기계학습 모델 재학습 없이 실시간으로 조정할 수 있습니다. 또한, 인간의 뇌에서 발견된 N400과 P600 ERP(complex event-related potential)와의 첫 공식적 연결을 수립하였습니다.

- **Performance Highlights**: CLADA는 6개 주요 LLM과 9개 벤치마크에서 테스트하여 평균 20%의 속도 향상과 2% 미만의 정확도 하락을 달성했습니다. 반면, 기존의 방법인 Griffin이나 TT는 성능 저하나 미미한 속도 향상을 보였습니다. 이를 통해 CLADA는 자원 인식 LLM 추론을 위한 실용적인 해결책을 제공하며, 생물학적 영감을 받은 AI 설계를 진전시킵니다.



### Improving the quality of Web-mined Parallel Corpora of Low-Resource Languages using Debiasing Heuristics (https://arxiv.org/abs/2502.19074)
- **What's New**: 이 논문은 웹에서 수집된 병렬 데이터의 소음을 필터링하는 데 있어서 다중 언어 모델(multiPLM)의 선택이 NMT 성능에 미치는 영향을 조사합니다. 이전 연구에서는 다중 언어 모델을 사용하여 문장 쌍을 랭킹하는 것이 더 나은 성과를 내지만, multiPLM의 선택에 따라 성과의 차이가 발생함을 보여줍니다. 이 연구에서는 LASER3, XLM-R 및 LaBSE의 다양한 multiPLM을 분석하고, 문장 소음의 원인을 규명합니다.

- **Technical Details**: 우리는 CCMatrix 및 CCAligned와 같은 웹 자료를 이용해 En→Si, En→Ta 및 Si→Ta 언어 쌍에 대해 3개의 다중 언어 모델로 랭킹을 실시했습니다. 이를 통해 작성된 NMT 시스템이 각 랭킹된 코퍼스에서 상위 10만 문장을 이용해 훈련되었습니다. 다양한 다중 언어 모델 사용 시 NMT 결과의 불균형이 두드러지며, 이를 해결하기 위해 여러 가지 휴리스틱을 적용하여 소음을 제거하고 NMT의 성능을 향상시키는 방법을 모색했습니다.

- **Performance Highlights**: 이 연구를 통해 최적의 휴리스틱 조합이 제시되어, 선택된 언어 쌍에서 웹에서 수집된 코퍼스에 대한 NMT 성능이 유의미하게 개선되었습니다. 사람들의 평가에 따르면, 필터링된 코퍼스에서 품질 높은 문장의 비율이 상당히 증가하였습니다. 마지막으로, multiPLM으로 인해 발생하는 성과 간의 불균형이 크게 줄어들어, NMT 결과의 비교 가능성이 높아졌습니다.



### Can Large Language Models Outperform Non-Experts in Poetry Evaluation? A Comparative Study Using the Consensual Assessment Techniqu (https://arxiv.org/abs/2502.19064)
- **What's New**: 이번 연구에서는 Consensual Assessment Technique (CAT)을 기반으로 한 두 개의 고급 Large Language Model (LLM)인 Claude-3-Opus와 GPT-4o를 사용하여 시(詩)를 평가하는 방법론을 조사합니다. 90개의 시로 구성된 데이터셋을 통해, 이 LLM들이 비전문가 인간 판단자들보다 출판 장소에 기반한 진실(ground truth)을 더 잘 일치시킬 수 있음을 발견하였습니다.

- **Technical Details**: 연구에서 사용된 CAT 기법은 전문가의 전반적인 판단을 통해 창의성을 평가하는 방법입니다. Claude-3-Opus와 GPT-4o는 작은 하위 집합의 시를 평가할 때 비전문가 평가자보다 더 우수한 성과를 보여주었습니다. Claude-3-Opus는 GPT-4o보다 다소 뛰어난 성능을 나타냈습니다.

- **Performance Highlights**: 이 연구는 LLM들이 시를 평가하는 데 있어 정확한 도구로 기능할 수 있음을 보여주며, 이는 다른 창의적인 분야로의 더 넓은 응용 가능성을 열어줍니다. 특히, LLM들은 비전문가의 경우보다 더 일관된 평가 결과를 제공할 수 있다는 점이 강조됩니다.



### MathClean: A Benchmark for Synthetic Mathematical Data Cleaning (https://arxiv.org/abs/2502.19058)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 훈련에서 수학적 데이터의 중요성을 강조하며, MathClean 벤치마크를 제안합니다. 이 벤치마크는 2,000개의 올바른 질문과 2,000개의 오류가 포함된 질문으로 구성되며, 추가적으로 2,000개의 올바른 답변과 오류가 있는 답변도 포함되어 있습니다. MathClean을 통해 LLMs의 수학적 데이터 정화 능력을 평가하고 다양한 오류 유형을 식별하는 기능을 높일 수 있습니다.

- **Technical Details**: MathClean 벤치마크는 오답을 식별하는 것뿐만 아니라, 각 질문 및 답변에 오류 유형을 주석으로 달아 모델이 다양한 오류 범주를 정확히 인식할 수 있도록 설계되었습니다. 총 2,000개의 올바른 질문와 2,000개의 오류가 포함된 질문이 있으며, 모델이 정확한 답변을 판별할 수 있는지 평가합니다. 이를 위해 10개 유형의 오류 증강 프롬프트와 16개 유형의 다양성 증강 프롬프트를 개발하여 더 다양하고 질 높은 데이터를 보장합니다.

- **Performance Highlights**: 다양한 SOTA 모델을 활용한 평가와 실험 결과, 특별히 GPT-o1 및 DeepSeek-R1 같은 강력한 모델도 MathClean 벤치마크에서 부족한 성능을 보였습니다. 이 결과는 MathClean의 유용성을 강조하며, 수학적 데이터의 정화가 LLMs의 훈련 및 성능 향상에 필수적임을 시사합니다. MathClean 벤치마크는 향후 모델의 개선 방향과 데이터 품질 향상에 기여할 것으로 기대됩니다.



### Binary Neural Networks for Large Language Model: A Survey (https://arxiv.org/abs/2502.19008)
Comments:
          23 pages, 7 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 새로운 바이너리 양자화 기법에 대한 포괄적인 리뷰를 제공합니다. 특히 BitNet 접근 방식을 강조하며, 이는 모델 훈련 시작부터 저정밀 바이너리 가중치를 사용하여 양자화를 수행하는 방법입니다. 이는 기존의 Post-Training Quantization(PTQ) 및 Quantization-Aware Training(QAT)과는 기본적으로 다른 접근입니다.

- **Technical Details**: 전통적인 양자화 방법인 PTQ와 QAT는 낮은 비트 너비에서 심각한 정밀도 손실을 가져옵니다. 그러나 BitNet은 훈련 초기부터 바이너리 가중치를 활용하여 양자화하여 고에너지 효율성을 달성할 수 있도록 설계되었습니다. 이 논문에서는 바이너리 양자화 기술이 딥 뉴럴 네트워크에서 어떻게 발전해왔는지를 상세히 설명합니다.

- **Performance Highlights**: BitNet 방식은 CPU 환경에서 높은 효율성을 얻었으며, 다중 모달 도메인에도 성공적으로 확장되었습니다. 이는 1비트 양자화 기술이 LLM에서 낮은 비용과 높은 정밀도로 개발될 수 있는 가능성을 보여줍니다. 논문에서 여러 연구 결과들이 소개되며, 이러한 기술들이 LLM의 실제 응용에 어떻게 기여할 수 있을지 탐구합니다.



### MEBench: Benchmarking Large Language Models for Cross-Document Multi-Entity Question Answering (https://arxiv.org/abs/2502.18993)
- **What's New**: MEBench는 cross-document multi-entity question answering (MEQA)를 위한 새로운 벤치마크로, 대형 언어 모델(LLMs)과 retrieval-augmented generation (RAG) 시스템의 성능을 체계적으로 평가하고자 만들어졌습니다. 기존 방법은 단일 문서에 최적화되어 있으나, 다양한 문서에서 정보 통합이 필요한 복잡한 질문에는 한계를 보였습니다. 이 벤치마크는 4,780개의 질문 모음으로 구성되어 있으며, 복잡성과 범위가 다양한 실제 시나리오를 포괄합니다.

- **Technical Details**: MEBench는 정보의 조각들을 통합할 수 있는 LLM의 능력을 평가하기 위해 설계되었습니다. 특히, Entity-Attributed F1 (EA-F1) 메트릭을 도입하여 개별 엔티티의 정확성과 기여도를 평가합니다. 세 가지 주요 범주와 여덟 가지 유형으로 질문들이 체계적으로 분류되어 있으며, 각 질문은 서로 다른 엔티티 밀도를 제공합니다: 낮음(0-10), 중간(10-100), 그리고 높은 복잡성(≥100)입니다.

- **Performance Highlights**: 최신 LLM 모델인 GPT-4 및 Llama-3를 통한 실험에서는 MEBench에서 평균 59%의 정확도로 나타났습니다. 이는 현 시스템의 한계와 엔티티 속성 또는 암시적 관계를 잘 추론하지 못하는 문제를 드러내며, 엔티티 인식을 중시하는 아키텍처의 필요성을 시사합니다. MEBench는 현재 LLM 프레임워크의 체계적인 약점을 강조하고, 더욱 견고한 답변 생성 아키텍처의 발전을 위한 기초를 제공합니다.



### GenTool: Enhancing Tool Generalization in Language Models through Zero-to-One and Weak-to-Strong Simulation (https://arxiv.org/abs/2502.18990)
- **What's New**: 이 연구에서는 GenTool이라는 새로운 훈련 프레임워크를 제안하여 대형 언어 모델(LLMs)의 도구 활용 일반화 능력을 향상시키는 데 초점을 맞추었습니다. GenTool은 사용 가능한 도구가 전혀 없거나 약한 도구를 사용할 때, 이 도구를 효율적으로 선택하고 이용할 수 있도록 지원합니다. 이를 통해 LLM이 실세계 정보에 더 효과적으로 대응할 수 있는 가능성을 높였습니다.

- **Technical Details**: GenTool은 제로-투-원 일반화(zero-to-one generalization)와 약-투-강 일반화(weak-to-strong generalization)라는 두 가지 주요 차원에서 훈련됩니다. 또한, 834개의 새로운 합성 도구와 8,515개의 독특한 쿼리로 구성된 고품질 합성 훈련 데이터셋을 활용하여 모델을 교육합니다. 두 단계의 미세 조정(fine-tuning) 전략을 사용하여 도구의 순위 매김(tool ranking) 후 선택(tool selection)을 최적화합니다.

- **Performance Highlights**: 실험 결과, GenTool은 1B에서 8B 매개변수를 가진 LLM의 도구 활용 능력을 상당히 향상시켰으며, GPT-4o를 초월하는 성능을 달성했습니다. 도구 선택 정확도가 14.28% 개선되는 등의 성과를 보였으며, 4가지 일반화 시나리오에서 탁월한 일반화 능력을 입증하였습니다. 우리의 연구는 LLM이 도구 일반화에서 직면하는 다양한 과제에 대한 귀중한 통찰도 제공합니다.



### PEToolLLM: Towards Personalized Tool Learning in Large Language Models (https://arxiv.org/abs/2502.18980)
- **What's New**: 이번 논문에서는 개인화된(tool learning) 도구 사용 능력을 강조하며, 기존의 도구 학습 연구들이 일반적인 도구 활용 능력에 치중했던 한계를 지적합니다. 이를 위해 사용자 인터랙션 이력을 통합하여 개인화된 도구 사용을 위한 새로운 작업을 정의하고, 첫 번째 개인화 도구 학습 벤치마크인 PEToolBench를 제안하였습니다. PEToolBench는 다양한 사용자 선호도를 반영하여, 46개 범주에 걸쳐 7454개의 도구를 포함하고 있습니다.

- **Technical Details**: PEToolLLaMA 라는 새로운 프레임워크가 제안되어 LLMs에 개인화된 도구 사용 능력을 부여합니다. 훈련 과정은 감독 세부 조정(Supervised Fine-Tuning, SFT) 단계와 사용자 선호 최적화(Direct Preference Optimization, DPO) 단계를 포함하며, 이는 사용자가 선호하는 도구 호출을 샘플링하여 최적화하는 과정입니다. 이 프레임워크는 사용자 명령과 상관없이 사용자 인터랙션 이력을 고려하여 과제를 수행하도록 LLM을 유도합니다.

- **Performance Highlights**: 실험 결과, PEToolLLaMA는 기존의 최고 성능 LLM보다 50% 이상 향상된 성능을 발휘하여, 개인화된 도구 사용 기술에서의 우수성을 보여주었습니다. 또한, PEToolLLaMA는 다양한 사용자 요구를 충족시키는 도구 사용 준비를 통해 LLM들의 활용도를 대폭 향상시킵니다. 이는 개인화 도구 학습 평가를 위한 중요한 기초 자료를 제공하게 됩니다.



### Low-Confidence Gold: Refining Low-Confidence Samples for Efficient Instruction Tuning (https://arxiv.org/abs/2502.18978)
Comments:
          8 pages

- **What's New**: 이 연구는 Large Language Models에 대한 instruction fine-tuning의 효과성을 높이기 위해 새로운 필터링 프레임워크인 Low-Confidence Gold (LCG)를 도입합니다. LCG는 centroid-based clustering과 confidence-guided selection을 활용하여 가치 있는 instruction 쌍을 식별합니다. 이 접근 방식은 데이터의 다양성을 유지하면서 고품질의 부분 집합을 큐레이션하는 반면, Semi-supervised 방법론을 적용하여 대표 샘플로 훈련된 경량 classifier를 사용합니다.

- **Technical Details**: LCG 프레임워크는 centroid-based clustering을 기반으로 하며, 각 클러스터의 중심을 통해 유사한 데이터 샘플을 그룹화합니다. 그 후, confidence-guided selection을 통해 고품질의 instruction 쌍을 선택합니다. 이 과정은 주어진 데이터를 효과적으로 활용함으로써 instruction fine-tuning의 품질을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, LCG로 필터링된 6K 샘플을 기반으로 fine-tuning된 모델들이 기존 방법론에 비해 더 뛰어난 성능을 보여주었습니다. 특히 MT-bench에서의 상당한 개선을 나타내며, 종합적인 평가 메트릭스에서도 일관된 성과 향상을 달성했습니다. 이러한 결과는 효율적인 instruction tuning을 위한 유망한 방향을 제시합니다.



### Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles (https://arxiv.org/abs/2502.18968)
Comments:
          9 pages

- **What's New**: 최근 논문에서는 사용자 시뮬레이터를 통해 대화 시스템과의 인간 상호작용을 모방하는 사용자 시뮬레이터인 User Simulator with implicit Profiles (USP)를 제안합니다. 이 프레임워크는 대화에서 유추된 사용자 프로필을 활용하여 더 개인화되고 현실적인 대화를 생성할 수 있도록 설계되었습니다. USP는 대화 시뮬레이션 과정에서 다층적인 사용자 특성을 반영하여 시뮬레이터의 성능을 개선합니다.

- **Technical Details**: USP는 대화에서 사용자의 숨겨진 특성을 추출하는 LLM 기반의 프로필 추출기를 발전시킵니다. 이 시스템은 사용자 특성의 패턴을 파악하기 위해 두 가지 단계의 훈련을 수행하는데, 첫 번째는 사용자 프로필에 기반한 조건부 감독 미세 조정이며, 두 번째는 대화 수준의 시뮬레이션을 위해 강화 학습을 적용합니다. 또한, 다양한 사용자 프로필을 샘플링하여 현실적인 사용자 분포를 캡처합니다.

- **Performance Highlights**: 실험 결과에 따르면, USP는 기존의 베이스라인보다 약 34% 및 43% 더 높은 의미적 유사성과 스타일적 유사성을 가지며, 재구성 오차를 절반으로 줄였습니다. USP는 ProfileGPT와 같은 다른 시뮬레이션 프레임워크와 동일하게 일관성을 유지하면서도 다중 턴 시나리오에서 14% 더 높은 대화 프로필 일관성을 보여줍니다. 또한 USP 기반의 다중 턴 동적 평가는 기존 벤치마크와 잘 일치하여 LLM 성능 평가의 정밀도를 높입니다.



### MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors (https://arxiv.org/abs/2502.18940)
Comments:
this https URL

- **What's New**: 이 논문에서는 MathTutorBench라는 새로운 오픈소스 벤치를 제안하여 AI 기반의 튜터링 모델의 교육적 능력을 평가할 수 있는 종합적인 방법론을 제공합니다. 이 벤치는 대화형 교육에서의 튜터링 능력을 평가하는 데이터셋과 메트릭스를 포함하고 있어 현재의 평가 방식의 부족함을 보완합니다.

- **Technical Details**: MathTutorBench는 세 가지 카테고리로 나뉘어 있습니다: 수학 전문성, 학생 이해, 그리고 교사 응답 생성입니다. 특히, 효과적인 튜터 발화와 덜 효과적인 발화를 구별하여 보상 모델을 훈련시키고, 이를 통해 튜터 모델의 생성물에 점수를 매깁니다. 이 모델은 전문가와 초보 교사의 발화를 높은 정확도로 구분할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 모델의 주제 전문성과 교육적 능력 간에는 trade-off가 존재하며, 교육을 위해 특화된 모델은 일반 모델에 비해 더욱 긴 대화에서도 교육 능력을 유지합니다. 이는 향후 튜터링 LLM의 개발 가속화에 기여할 것으로 기대됩니다. MathTutorBench는 자동화된 메트릭스를 사용하여 신속하고 공정한 평가를 가능하게 하므로, 공개적으로 데이터와 코드를 제공하여 연구자들이 쉽게 사용할 수 있도록 하였습니다.



### JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models (https://arxiv.org/abs/2502.18935)
Comments:
          12 pages, 5 figures, accepted at PAKDD 2025

- **What's New**: 이 논문에서는 JailBench를 소개하며, 이는 LLMs(대형 언어 모델)의 심층 안전 취약성을 평가하기 위한 첫 번째 포괄적인 중국어 벤치마크입니다. JailBench는 조정된 계층적 안전 분류 체계를 특징으로 하고 있으며, 이는 중국 맥락에 맞춰져 있습니다. 저자들은 이 벤치마크가 이전의 중국어 벤치마크들에 비해 ChatGPT에 대한 높은 공격 성공률을 기록했다고 강조하고 있습니다.

- **Technical Details**: JailBench는 4,600개의 질의로 구성된 데이터를 구축하며, 이는 5개의 고유한 영역과 40개의 리스크 유형을 포함하는 새로운 두 레벨 계층화 안전 분류 기준을 기반으로 하고 있습니다. Automatic Jailbreak Prompt Engineer (AJPE) 프레임워크를 통해 자동적으로 위험한 질의를 생성하여 데이터셋을 확장하며, 이는 LLM의 맥락 학습을 활용하여 효율성을 극대화합니다.

- **Performance Highlights**: JailBench는 13개의 주요 LLM에 대해 광범위한 평가를 수행하였으며, ChatGPT에 대해 73.86%의 공격 성공률을 기록했습니다. 이로써 JailBench는 다양한 분야에서 LLM의 잠재적 취약성을 효과적으로 식별하는 데 성공하였으며, 이는 중국어 맥락에서 LLM의 안전성과 신뢰성을 개선할 수 있는 중요한 통찰을 제공합니다.



### Kanana: Compute-efficient Bilingual Language Models (https://arxiv.org/abs/2502.18934)
Comments:
          40 pages, 15 figures

- **What's New**: Kanana는 한국어에서 뛰어난 성능을 보이는 이중 언어 모델 시리즈로서, 계산 비용이 기존 모델보다 상당히 낮습니다. 이 모델은 데이터 필터링, 단계적 사전 훈련, 깊이 업스케일링 등 다양한 기술을 사용하여 경쟁력 있는 성능을 자랑합니다. Kanana 모델 시리즈는 2.1B에서 32.5B 파라미터로 구성되어 있으며, 연구 촉진을 위해 2.1B 모델이 공개되었습니다.

- **Technical Details**: 사전 훈련 단계는 대부분의 계산 비용을 차지하므로, Kanana 모델에서는 데이터 효율성과 훈련 효율성 향상에 중점을 두었습니다. 3조 개의 토큰으로 구성된 훈련 데이터셋을 신중하게 구성하여, 적은 양의 데이터로도 경쟁력 있는 성능을 구현했습니다. 또한, 단계적 사전 훈련(staged pre-training)과 깊이 업스케일링(depth up-scaling) 기법을 적용하여 계산 비용을 최소화하고, 더 작은 모델을 교육하기 위해 프루닝(pruning) 및 증류(distillation) 기법도 활용했습니다.

- **Performance Highlights**: Kanana 모델은 MMLU와 KMMLU와 같은 지식 집약적 자연어 이해 벤치마크에서 다른 모델보다 더 나은 성능을 발휘하면서도 계산 리소스를 대폭 절감합니다. 특히, Kanana Flag 32.5B 모델은 Llama 3.1 70B와 같은 대형 모델보다도 뛰어난 성능을 보였습니다. 모든 Kanana LLM 모델은 HAE-RAE 벤치마크에서 유사한 크기의 다른 모델들보다 우수한 성과를 기록했습니다.



### END: Early Noise Dropping for Efficient and Effective Context Denoising (https://arxiv.org/abs/2502.18915)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 긴 입력 시퀀스에서 소음(noise)으로 인해 발생하는 품질 저하 문제를 다룹니다. 연구팀은 LLM이 첫 번째 토큰을 생성하기 전에 입력 시퀀스에서 유용한 정보를 조기에 식별할 수 있음을 발견하였습니다. 이를 바탕으로 Early Noise Dropping(END)이라는 새로운 접근 방식을 제안하며, 이는 LLM의 세밀한 조정 없이 소음 문제를 완화하는 데 중점을 둡니다.

- **Technical Details**: END는 입력 시퀀스를 여러 개의 청크(Chunks)로 나누고, LLM의 초기 레이어에서 선형 프로버(linear prober)를 사용하여 유용한 청크와 소음 청크를 구별합니다. 이 방법은 소음 청크를 초기에 버림으로써 출력 품질을 유지하고, 계산 오버헤드를 줄입니다. EDD는 즉각적인 성능 향상을 목표로 하며, 기존 방법들의 복잡성과 실행 시간을 없애고 효율적으로 작동합니다.

- **Performance Highlights**: 연구 결과, END는 다양한 평가 데이터 세트에서 여러 LLM의 성능과 효율성을 크게 향상시켰습니다. 기존 강력한 기준선과 비교할 때 10% 이상의 성능 향상과 약 50%의 계산 감소를 달성하였습니다. 실험을 통해 LLM이 내부적으로 소음 및 불필요한 정보를 구별하는 메커니즘을 더욱 깊이 이해할 수 있는 기회를 제공합니다.



### CS-Dialogue: A 104-Hour Dataset of Spontaneous Mandarin-English Code-Switching Dialogues for Speech Recognition (https://arxiv.org/abs/2502.18913)
- **What's New**: 이 논문에서는 CS-Dialogue라는 새로운 대규모 화자 기반의 만다린-영어 코드 스위칭(codeswitching) 음성 데이터 세트를 소개합니다. 이 데이터 세트는 200명의 화자로부터 104시간의 자발적 대화를 포함하며, 완전한 전사(transcription)가 제공됩니다. 이전의 데이터 세트들과는 달리 CS-Dialogue는 일관된 음성 현상을 캡처해 코드 스위칭 ASR(Automatic Speech Recognition) 연구에 큰 기여를 할 수 있습니다.

- **Technical Details**: CS-Dialogue 데이터 세트는 200명의 원어민 화자를 통해 수집되었으며, 각 화자는 높은 수준의 영어 능력을 갖추고 있습니다. 데이터 수집 과정은 7가지의 일상적인 주제에 대한 상호 대화로 구성되어 있으며, 통화는 만다린, 코드 스위칭, 그 다음 영어로 진행됩니다. 이 데이터 세트는 16kHz의 샘플링 주파수를 가진 mono PCM WAV 형식으로 저장됩니다.

- **Performance Highlights**: 논문에서는 퍼포먼스 벤치마킹을 통해 최신 모델, 즉 Transformer, Conformer, 그리고 Branchformer를 사용하여 코드 스위칭 ASR의 어려움을 시연합니다. 실험 결과, 기존의 사전 훈련된 모델인 Whisper가 향상될 여지가 있다는 사실이 밝혀졌습니다. CS-Dialogue 데이터 세트는 모든 학술 목적으로 무료로 제공될 예정이며, 이는 연구 및 모델 개발에 큰 도움이 될 것입니다.



### From Hours to Minutes: Lossless Acceleration of Ultra Long Sequence Generation up to 100K Tokens (https://arxiv.org/abs/2502.18890)
- **What's New**: TOKENSWIFT는 대규모 언어 모델(LLM)에서 100K 토큰의 초긴 시퀀스를 생성하는 과정을 크게 가속화하기 위한 새로운 프레임워크입니다. 기존의 보편적인 추정 디코딩(speculative decoding) 방법들이 긴 시퀀스 생성을 지원하지 못하는 한계를 극복하고, 재구성된 키-값( KV) 관리와 반복 생성 문제를 해결합니다.

- **Technical Details**: TOKENSWIFT는 n-그램 검색과 동적 KV 캐시 업데이트를 활용하여 효율적인 초긴 시퀀스 생성을 가능하게 합니다. 다중 토큰 생성을 통해 LLM이 단일 전방 전달(forward pass)에서 여러 개의 토큰을 초안화할 수 있어, 모델 재로드의 빈도를 줄입니다. 이 과정에서 문맥 패널티를 적용하여 출력의 다양성을 보장합니다.

- **Performance Highlights**: 실험 결과, TOKENSWIFT는 다양한 모델 스케일과 아키텍처에서 3배 이상의 속도 향상을 달성합니다. 예를 들어, LLaMA3.1-8B의 경우 초긴 시퀀스 생성을 5시간에서 90분으로 단축시키며, 생성 길이가 증가함에 따라 속도 향상이 더욱 두드러집니다. 이러한 결과는 LLM의 초기 정밀도를 유지하면서도 효율성을 극대화합니다.



### On Pruning State-Space LLMs (https://arxiv.org/abs/2502.18886)
- **What's New**: 이 논문은 상태공간 모델(State-Space Models, SSM)이 변환기 기반 대형 언어 모델(LLM)의 효율적인 대안으로 제안되었음을 강조합니다. 저자들은 SSM 구조에 여러 가지 프루닝(pruning) 방법을 적용하여, 다양한 작업에서 모델의 계산 비용을 더욱 줄일 수 있는 가능성을 탐구하였습니다. 연구 결과, 특정 프루닝 방법들(WANDA 등)에 대해 SSM 모델들이 상당한 강건성을 보이는 반면, 다른 방법들은 성능 저하를 초래할 수 있음을 발견하였습니다.

- **Technical Details**: 선택적 상태 공간 모델(Selective-State Space Models, SSM)은 입력을 시간에 따라 진화하는 숨겨진 상태로 표현하는 seq2seq 모델의 일종입니다. 이 모델은 구조적 표현을 활용하여 х^2 (quadratic) 복잡도를 초래하지 않고, 변환기 기반 LLM과 비슷한 성능을 보여주며 주목받고 있습니다. 연구자들은 SSM 기반 LLM에서 다양한 구조적 프루닝 기법들을 조정하여 이를 적용하고, WANDA와 같은 비구조적 프루닝 방법과 비교하였습니다.

- **Performance Highlights**: 논문에서는 WANDA 방법을 통해 최대 50%의 매개변수를 줄였음에도 불구하고 SSM 모델이 여전히 강건성을 유지함을 보여주었습니다. 반면, SSM 헤드를 프루닝할 경우 모든 경우에서 성능이 급격히 저하되며, SSM 상태를 프루닝하면 경미한 성능 저하가 발생하는 것으로 나타났습니다. 이 결과는 SSM 기반 LLM의 효율성을 개선할 수 있음을 시사하지만, 프루닝 방법의 선택이 품질에 큰 영향을 미친다는 점을 강조합니다.



### Learning to Generate Structured Output with Schema Reinforcement Learning (https://arxiv.org/abs/2502.18878)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 구조적 생성 기능을 조사하고 JSON(자바스크립트 객체 표기법) 출력을 생성하는 능력에 초점을 맞추고 있습니다. 기존 JSON 사용의 중요성을 강조하며, 모델의 JSON 생성 능력에 대한 종합적인 분석과 벤치마킹이 부족하다는 점을 지적합니다. 40,000개 이상의 다양한 JSON 스키마를 포함하는 SchemaBench를 제안하여 LLM의 유효한 JSON 생성 능력을 평가하고 향상시키는 방법을 모색합니다.

- **Technical Details**: 연구에서는 LLM의 JSON 문자열 생성에 대한 다양한 도전 과제를 분석하고, 이와 관련된 모델의 성능을 측정하기 위한 SchemaBench를 개발했습니다. 이 벤치마크는 주어진 JSON 스키마에 따라 유효한 JSON 문자열을 생성하는 데 중심을 두고 있으며, 방금 언급한 다양한 도전 과제를 세 가지 범주로 나누었습니다. 또한, Schema Reinforcement Learning(SRL)을 통해 모델이 구조화된 데이터를 생성할 수 있도록 강화 학습과 세밀한 스키마 검증기를 통합한 혁신적인 훈련 파이프라인을 제안합니다.

- **Performance Highlights**: 최신 LLM이 여전히 복잡한 JSON 스키마에 대한 유효한 JSON 문자열을 생성하는 데 어려움을 겪고 있다는 것을 발견하였고, SchemaBench에서 61.06%의 정확도를 기록했습니다. 그러나 SRL 접근 방식을 통해 모델의 복잡한 JSON 생성 속도가 최대 16% 향상되었으며, 이는 일반적인 능력을 유지하면서 구조화된 생성을 전문화할 수 있음을 보여줍니다. 또한, BFCL과 같은 다운스트림 작업에서 성능이 크게 향상됨을 입증하여 구조적 생성의 개선이 관련 작업에서도 우수한 성과로 이어질 수 있음을 확인했습니다.



### Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework (https://arxiv.org/abs/2502.18874)
- **What's New**: 이번 논문에서 제안하는 ARJudge라는 새로운 평가 프레임워크는 자동화된 평가 기준 생성을 통해 LLM(대형 언어 모델)의 응답을 보다 효과적으로 평가할 수 있도록 설계되었습니다. 기존의 평가 방법론들은 사전 정의된 기준에만 의존하여 다양한 작업에 대한 적응력이 줄어드는 경향이 있었습니다. ARJudge는 텍스트 기반 분석과 코드 기반 분석을 통합하여 더 많은 측면에서 평가하도록 보장합니다.

- **Technical Details**: ARJudge는 두 가지 주요 구성 요소, 즉 다면적 평가를 생성하는 Analyzer와 분석 결과를 종합하여 최종 결정을 내리는 Refiner로 이루어져 있습니다. 이 시스템은 Composite Analysis Corpus라는 훈련 데이터를 기반으로 하여 평가 기준을 생성하고, 다각적인 분석을 통해 LLM의 창조적 응답을 효과적으로 평가하는 방법론을 제시합니다. 특히, 코드 기반 분석을 통합함으로써, 기존의 텍스트 기반 방법보다 훨씬 개선된 정확도를 보여줍니다.

- **Performance Highlights**: 다양한 벤치마크에서 수행된 실험 결과, ARJudge는 기존의 미세 조정된 평가자들보다 뛰어난 성능과 견고성을 보였습니다. 특히, ARJudge는 LLM의 응답이 지침을 얼마나 잘 따르는지를 평가하는 데 있어 코드 기반 분석이 약 11.1% 더 높은 정확도를 제공함을 입증했습니다. 이러한 결과는 다면적 평가와 코드 기반 분석의 활용이 평가 능력을 강화하는 데 얼마나 중요한지를 보여줍니다.



### Exploring Rewriting Approaches for Different Conversational Tasks (https://arxiv.org/abs/2502.18860)
Comments:
          Preprint

- **What's New**: 이 논문에서는 대화형 보조 시스템에서 사용자 질문에 대한 보다 의미 있는 답변을 제공하기 위한 질문 재작성(Question Rewriting) 알고리즘의 두 가지 접근 방법을 체계적으로 비교합니다. 이러한 접근 방법에는 질문 재작성과 쿼리 융합(Question Fusion)이 포함되며, 서로 매우 다른 생성 작업에 적용됩니다. 연구 결과는 특정 재작성 접근 방식이 사용 사례 및 생성 작업에 따라 크게 달라진다는 것을 보여줍니다.

- **Technical Details**: 대화형 보조 시스템의 효율성을 높이기 위해 다양한 질문 재작성 기법이 진화해 왔습니다. 예를 들어, 쿼리 확장(Query Expansion), 동의어 사용, 유사 구문 재구성 방법 등이 있습니다. 최근에는 대형 언어 모델(LLMs)을 활용하여 사용자 쿼리의 모호성을 자동으로 해소하고, 이를 통해 데이터 분석 및 질문 응답 작업의 성능을 향상시키는 방법에 대한 연구가 진행되고 있습니다.

- **Performance Highlights**: 실험 결과, 데이터 분석 보조 시스템에서 쿼리 융합 접근 방식이 가장 우수한 성능을 보였으며, 대화형 질문 응답 시스템에서는 쿼리 재작성 접근 방식이 더 효율적임을 확인했습니다. 특히, 단기 및 장기 대화를 포함한 두 개의 데이터 세트를 분석하여 각 접근 방식의 효과를 도출했습니다. 연구는 고유한 사용 사례에 최적화된 쿼리 재작성 전략 설계의 필요성을 강조합니다.



### A Causal Lens for Evaluating Faithfulness Metrics (https://arxiv.org/abs/2502.18848)
Comments:
          18 pages, 18 figures, 6 tables

- **What's New**: 이 논문에서는 LLMs의 자연어 설명의 신뢰성을 평가할 수 있는 Causal Diagnosticity라는 새로운 평가 프레임워크를 소개합니다. 여러 신뢰성 지표가 개발되었지만, 이들 사이를 비교할 수 있는 통합된 평가 체계가 부재하였기 때문에 이에 대한 해결책을 제시합니다. 특히, 신뢰성 지표의 진단적 특성을 평가하여, 더 신뢰할 수 있는 해석 가능성을 제공하는 방법론의 필요성을 강조합니다.

- **Technical Details**: Causal Diagnosticity 프레임워크는 신뢰성 메트릭을 평가하기 위한 새로운 방법론을 제안합니다. 이 프레임워크는 인과적 진단 가능성(causal diagnosticity) 개념을 활용하고, 모델 편집 기법을 사용하여 신뢰전 설명(pair)과 비신뢰전 설명(pair)을 생성합니다. 또한, 사실 확인(fact-checking), 유추(analogy), 객체 세기(object counting), 다중 단계 추론(multi-hop reasoning) 등의 네 가지 작업을 포함하는 벤치마크를 마련하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 테스트한 모든 신뢰성 메트릭스는 종종 랜덤 베이스라인(random baseline)을 초과하지 못하는 경과를 보였습니다. 이는 신뢰성 메트릭 설계에서 진단적 접근(diagnosticity-first approach)의 필요성을 강조하며, 연속적인 신뢰성 점수를 산출하는 메트릭이 이진 점수에 비해 더 진단적이라는 사실을 발견하였습니다. 이 조사 결과는 LLM 하의 신뢰성 메트릭 향상과 해석 가능성 방법 개선을 요구합니다.



### Sliding Window Attention Training for Efficient Large Language Models (https://arxiv.org/abs/2502.18845)
Comments:
          14 pages, 5 figures

- **What's New**: 최근 Transformer 기반의 대형 언어 모델(LLM)이 여러 작업에서 뛰어난 능력을 발휘하고 있습니다. 그러나 이들의 시퀀스 길이에 따른 제곱 계산 복잡성은 긴 문서 처리에 있어 큰 병목 현상으로 남아있습니다. 이를 해결하기 위해 SWAT라는 효율적인 모델이 소개되었으며, Sliding Window Attention Training을 통해 긴 Context를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: SWAT는 기존의 softmax 연산을 sigmoid 함수로 교체하여 attention sink 문제를 방지하고, 각 토큰당 더 높은 정보 용량을 유지하도록 합니다. 또한, 균형 잡힌 ALiBi와 Rotary Position Embedding을 활용하여 정보 압축 및 보존을 최적화합니다. 보통의 Sparse Attention 기법인 Sliding Window Attention(SWA)에서 발생하는 문제를 해결하기 위한 훈련 방법론을 제안합니다.

- **Performance Highlights**: SWAT는 여덟 개 벤치마크에서 최첨단 선형 순환 아키텍처와 비교하여 SOTA 성능을 달성했습니다. 이 모델은 복잡한 아키텍처 없이도 효과적으로 정보를 유지하며, 효율적인 컴퓨팅을 통해 다양한 NLP 작업에서 강력한 성능을 보였습니다. 실험 결과, SWAT는 기존 Transformer 모델과 다른 순환 모델들을 초월하는 성과를 입증했습니다.



### Sentiment Analysis of Movie Reviews Using BER (https://arxiv.org/abs/2502.18841)
Comments:
          7 pages, 3 figures, published in the proceedings The Fifteenth International Conference on Information, Process, and Knowledge Management (eKNOW 2023)

- **What's New**: 이 논문은 Bidirectional Encoder Representations from Transformers (BERT)와 Bidirectional Long Short-Term Memory (BiLSTM)를 결합하여 영화 리뷰에 대한 감정 분석을 개선하고자 합니다. 이 방법은 State-of-the-Art (SOTA) 모델의 정확성을 초월하는 최적의 정확성을 제공합니다. 또한, 제안된 방법은 특정 영화를 추천하기 위해 리뷰 감정의 전체 극성을 계산하는 방법으로 활용될 수 있습니다. BERT와 BiLSTM의 조합을 통한 감정 분석의 가능한 확장성도 제시되고 있습니다.

- **Technical Details**: 감정 분석(Sentiment Analysis, SA)은 텍스트에서 감정과 의견을 추출하는 작업으로, 이 논문은 이를 위해 BERT의 정교화(fine-tuning)와 BiLSTM을 결합하여 최적의 결과를 도출하고자 합니다. BERT는 양방향 컨텍스트를 사용하여 사전 훈련된(pre-trained) 언어 모델로, 감정 극성 분류(polarity classification)를 2점 척도로 수행합니다. BiLSTM은 입력 피처를 양 방향으로 처리하여 모델의 일반화 성능을 향상시키고, 이를 통해 SA의 정확성을 증대시킵니다.

- **Performance Highlights**: 제안된 BERT+BiLSTM-SA 모델은 정밀도가 가장 높은 벤치마크 데이터셋에서 SOTA 모델과 비교하여 우수한 성능을 보였습니다. 또한, 감정 분류 결과를 사용하여 영화 리뷰의 전체 극성을 계산하는 방법을 통해 영화 추천의 기초 데이터를 제공함으로써 실제 응용 가능성을 보여줍니다. 앞으로의 연구에 있어 이 접근 방식은 다양한 세부 분류로 확장될 수 있으며, 향후 더 많은 미디어 플랫폼에서의 감정 분석에 적용될 가능성을 지니고 있습니다.



### Evidence-Driven Marker Extraction for Social Media Suicide Risk Detection (https://arxiv.org/abs/2502.18823)
- **What's New**: 이번 논문은 기후 마커 추출(clinical marker extraction) 및 자살 위험 분류(suicide risk classification)를 위한 새로운 접근 방식인 Evidence-Driven LLM(ED-LLM)을 소개합니다. ED-LLM은 다중 작업 학습(multi-task learning) 프레임워크를 활용하여 Mistral-7B 기반 모델을 공동으로 훈련합니다. 이 접근법은 위험 평가를 지원하는 텍스트 증거를 명시적으로 강조하여 해석 가능성을 향상시킵니다.

- **Technical Details**: 논문에서는 자살 위험 관련 텍스트에서 임상 마커(span) 식별과 자살 위험 수준 분류를 동시에 수행하는 다중 작업 학습 프레임워크를 제안합니다. 모델은 자살 위험과 관련된 텍스트 구간을 추출하도록 학습되며, 이를 위해 데이터셋은 자살 위험 수준과 임상 마커 텍스트 구간으로 주석이 달립니다. 평가 과정에서 모델은 위험 수준을 예측하고 추출된 마커 구간을 정당화 자료로 출력합니다.

- **Performance Highlights**: ED-LLM은 CLPsych 데이터셋에서 평가되어 자살 위험 분류 및 임상 마커 식별에서 경쟁력 있는 성능을 보였습니다. 이 방식은 LLM, 전통적 기계 학습, 프롬프트 기반 방법들보다 우수성 또한 입증했습니다. 연구 결과는 해석 가능하고 효율적인 LLM 기반 자살 위험 평가의 중요성을 강조하며, 임상적으로 유용한 응용을 위한 길을 열고 있습니다.



### Judge as A Judge: Improving the Evaluation of Retrieval-Augmented Generation through the Judge-Consistency of Large Language Models (https://arxiv.org/abs/2502.18817)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 모델의 출력물 평가를 위한 Judge-Consistency (ConsJudge) 방법을 소개합니다. ConsJudge는 LLM(대형 언어 모델)이 보다 정확한 평가를 생성하도록 도와주며, RAG 모델 최적화를 위한 Evaluation 에서의 일관성을 강화합니다. 이 방법은 다양한 평가 차원에 따라 LLM이 판단을 생성하도록 유도하고, 이를 통해 더 높은 품질의 평가를 달성할 수 있도록 합니다.

- **Technical Details**: ConsJudge는 LLM 기반의 평가 모델을 개선하기 위해 여러 가지 평가 차원을 통합하고 다중 선택 전략을 도입합니다. 이 방법은 Direct Preference Optimization (DPO)을 활용하여 LLM의 판단 능력을 향상시키며, 다양한 판단 차원의 조합에 따른 결과를 생성하여 평가하는 프로세스를 포함합니다. 이러한 접근법은 LLM의 성능을 향상시키면서도 더 강력한 LLM에서의 증류 없이도 가능합니다.

- **Performance Highlights**: 실험 결과, ConsJudge는 RAG 모델 최적화를 위한 보상 모델로서 매우 효과적임을 입증하였으며, vanilla LLM들보다 유의미한 개선을 보여주었습니다. 추가 분석에 따르면, ConsJudge는 다양한 RAG 평가 데이터셋에서 GLM-4-plus로 알려진 우수한 LLM과의 판단 일관성을 높이 평가하고, 이는 다양한 RAG 작업에서 LLM이 더 정확한 판단을 할 수 있도록 최적화하는 데 기여합니다.



### Language Models Grow Less Humanlike beyond Phase Transition (https://arxiv.org/abs/2502.18802)
- **What's New**: 최근 연구에서는 언어 모델(Models)과 인간 독서 행동(Reading Behavior) 간의 정렬(alignment)이 사전 훈련(Pretraining) 중에 향상되다 특정 지점(tipping point)에 도달하면 더 이상 개선되지 않거나 오히려 감소하는 현상을 발견했습니다. 이러한 점에서 단어 빈도(word frequency), 주의력(recent bias)과 맥락 크기(context size) 등이 영향을 미친다고 알려져 있으나, tipping point의 원인과 언어 모델의 사전 훈련 동역학(Pretraining Dynamics)과의 상호작용에 대한 설명은 부족합니다.

- **Technical Details**: 연구진은 사전 훈련 과정에서 발생하는 phase transition이 이 tipping point의 주요 요인이라고 가설을 세웠습니다. 이를 통해 특화된 주의 헤드(specialized attention heads)의 빠른 출현이 관련되어 있음을 입증하기 위해 일련의 상관 및 인과 실험(correlational and causal experiments)을 수행하였습니다. 결과적으로, phase transition이 PPP의 tipping point에 기여하며 사후 학습 동적(learnings dynamics)을 변화시킨다는 것을 보여주었습니다.

- **Performance Highlights**: 이 연구는 언어 모델이 PPP의 감소에 기여하는 주의 패턴을 생성하기보다는, phase transition이 모델의 학습 동학을 변형하여 추가 훈련이 PPP에 해를 끼치는 방식을 설명합니다. 이러한 결과는 언어 모델의 사전 훈련 전략을 개선하고 인간 독서 행동에 더 잘 정렬될 수 있는 방향으로 나아가는 데 중요한 통찰을 제공합니다.



### ANPMI: Assessing the True Comprehension Capabilities of LLMs for Multiple Choice Questions (https://arxiv.org/abs/2502.18798)
- **What's New**: 이 논문에서는 언어 모델의 자연어 이해 능력을 평가하기 위한 새로운 메트릭 ANPMI를 제안합니다. ANPMI는 Pointwise Mutual Information (PMI)을 $-	ext{log} P(Choice)$에 의해 정규화하여 모델이 프롬프트를 정확히 이해하고 있는지를 측정합니다. 이 접근법은 기계가 프롬프트를 완전히 이해하지 않고도 답변을 선택하는 경우의 문제를 해결합니다.

- **Technical Details**: 현재 언어 모델의 평가는 주로 다중 선택 질문을 통해 이루어지며, P(Choice|Prompt) 확률을 기반으로 올바른 답변을 선택하는 빈도로 측정합니다. 그러나 이러한 방식은 모델이 프롬프트를 실제로 이해했는지를 보여주지 않습니다. ANPMI는 P(Choice) 불균형 문제를 보다 정확하게 교정하여 모델의 자연어 이해를 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안한 ANPMI 메트릭은 여러 사전 훈련된 모델과 벤치마크를 사용하여 기존 방법들보다 모델의 프롬프트 이해도를 훨씬 더 정확하게 평가하는 것으로 나타났습니다. 이를 통해 모델의 진정한 이해도를 측정하기 위한 새로운 기준을 제시하며, 언어 이해 능력 평가의 신뢰성을 향상시키는 데 기여합니다.



### Anything Goes? A Crosslinguistic Study of (Im)possible Language Learning in LMs (https://arxiv.org/abs/2502.18795)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)이 인간의 언어 습득 과정 및 처리에 대한 통찰을 제공할 수 있는지에 대한 질문을 탐색합니다. 특히, 연구진은 12개의 자연 언어를 대상으로 불가능한 언어 모델링을 실시하여 기존 연구들과 차별화된 결과를 도출했습니다. 연구 결과는 LLMs가 인간과 유사한 귀납적 편향을 보이지만, 이러한 편향이 인간 학습자에서 관찰된 것보다 약하다는 점을 시사합니다.

- **Technical Details**: 연구는 가능 언어(attested)와 불가능 언어(impossible) 및 확인되지 않은 언어(unattested) 간의 차이를 모델링하는 데 중점을 두었습니다. LLMs, 특히 GPT-2 small 모델은 가능한 언어 내에서 불가능한 변주를 구별하는 데 능숙하지만, 언어 가족 간에는 이 구분에 어려움을 겪는 것으로 나타났습니다. 연구진은 Greenberg의 Universal 20을 활용하여 다양한 NP 순서 변형을 통해 모델의 언어 차별화 능력을 분석했습니다.

- **Performance Highlights**: GPT-2 small은 가능 언어 내에서 불가능한 언어와의 구별을 신뢰성 있게 수행했으나, 다양한 언어 간의 구별에는 어려움을 나타냈습니다. 특히, 문장 구조와 고정된 어순이 유지되는 경우, 확인되지 않은 언어에 대한 낮은 혼란도(perplexity) 점수를 할당하여 사람처럼 규칙적인 구조에 대한 선호를 잘 보여주었습니다. 이러한 결과는 LLMs가 인간 언어 처리에서 지각된 편향과의 일치 여부를 탐색하는 중요한 기초 자료를 제공합니다.



### Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs (https://arxiv.org/abs/2502.18791)
Comments:
          21 pages, 9 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLM) 연구의 메타 분석을 위한 반자동화 접근 방식을 제안합니다. 이 방법은 관련 arXiv 논문을 자동으로 식별하고 실험 결과와 관련된 속성을 추출하여 구조화된 데이터셋으로 정리합니다. 이를 통해 논문 조사 및 데이터 추출의 수고를 93% 이상 줄일 수 있습니다.

- **Technical Details**: 데이터 추출 과정은 세 가지 단계인 전처리 및 필터링, 추출 및 증강, 설명 생성으로 나뉩니다. 연구에 사용된 모델은 GPT-4o 와 Gemini 1.0 Pro와 같은 최첨단 LLM입니다. 주요 성능 관련 속성으로는 데이터셋 이름, 모델 이름, 프롬프트 방법, 메트릭 이름 등이 포함됩니다.

- **Performance Highlights**: 이번 연구에서 자동으로 생성된 데이터셋은 이전의 수동 메타 분석 결과를 정확히 재현하며, Chain-of-Thought 방식이 주로 수학 및 기호적 추론 작업에서 이점을 제공한다는 것을 입증했습니다. 또한 세 가지 새로운 통찰력으로, 인맥 예시가 다중 모달 작업에 기여하지만 수학 작업에서는 제한적인 이점을 나타낸다는 점과 CoT와 ICL이 포함된 경우가 그렇지 않은 경우보다 전반적으로 우수하다는 결과가 도출되었습니다.



### Active Few-Shot Learning for Text Classification (https://arxiv.org/abs/2502.18782)
Comments:
          Accepted to NAACL 2025 Main Conference; 18 pages, 8 figures, 13 tables including Appendix

- **What's New**: 이 논문에서는 소수의 주석 샘플만으로도 효과적으로 학습할 수 있는 Few-Shot Learning (FSL) 방법을 제안합니다. 기존의 방식은 무작위로 샘플을 선택하는 경향이 있었으나, 이를 개선하기 위해 Active Learning (AL) 기반의 인스턴스 선택 메커니즘을 도입했습니다. 이를 통해 지원 샘플의 품질을 높이고, 더 나은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 방법은 불확실성(uncertainty)과 다양성(diversity), 그리고 대표성(representativeness)을 바탕으로 하는 샘플 선택 전략입니다. 이를 위해 특정 임베딩 방법을 통해 소스 데이터를 구조화하여 키 특징을 캡처하고, 엔트로피와 클러스터링 방법을 활용하여 효과적인 샘플을 선택합니다. 실험은 다섯 가지 분류 작업에서 수행되며, BART와 FLAN-T5 모델을 기반으로 Fine-Tuning (FT) 과정을 진행합니다.

- **Performance Highlights**: 제안된 방법은 기존 무작위 샘플링 또는 In-Context Learning (ICL) 방식과 비교하여 성능이 현저히 개선되었습니다. 강화된 지원 세트를 활용한 Fine-Tuning 결과, 평균적으로 우수한 성능을 보여줍니다. 연구자들은 구현 코드를 GitHub에 공개하여, 타 연구자들이 쉽게 활용할 수 있도록 하였습니다.



### Plutus: Benchmarking Large Language Models in Low-Resource Greek Financ (https://arxiv.org/abs/2502.18772)
Comments:
          18 pages, 6 figures

- **What's New**: 이번 연구에서는 그리스어 금융 NLP의 혁신을 위해 Plutus-ben과 Plutus-8B를 소개합니다. Plutus-ben은 첫 번째 그리스 금융 평가 벤치마크로, 다섯 가지 핵심 금융 NLP 작업을 정의하여 LLM 평가의 체계적 기반을 마련합니다. Plutus-8B는 그리스 금융 데이터로 미세 조정된 최초의 그리스 금융 LLM으로, 기존 모델과의 간극을 메우는 데 중점을 두고 있습니다.

- **Technical Details**: Plutus-ben은 숫자 및 텍스트 명명된 개체 인식(NER), 질문 응답(QA), 추상적 요약, 주제 분류를 포함한 다섯 가지 금융 NLP 작업을 다룹니다. 이 작업은 그리스어 금융 문서 처리의 복잡성을 반영하며, 세 개의 고품질 그리스 금융 데이터셋인 GRFinNUM, GRFinNER 및 GRFinQA로 지원됩니다. 이 데이터셋들은 재무 전문가들이 정밀하게 주석을 달아 무수히 많은 재무 용어와 정황을 캡처하도록 설계되었습니다.

- **Performance Highlights**: 22개의 LLM을 Plutus-ben에서 평가한 결과, 그리스 금융 NLP가 여전히 언어적 복잡성과 도메인 특화된 용어, 금융 추론의 간극으로 인해 도전적임을 발견했습니다. 기존의 영어 중심 모델 조차도 그리스 금융 작업에서 한계를 보였으며, Plutus-8B는 그리스 금융 데이터로 미세 조정되어 가장 높은 평균 점수를 기록했습니다. 그러나 요약 작업에서는 여전히 장기적인 금융 문서에서 어려움을 겪고 있어 추가적인 개선이 필요한 상황입니다.



### Automatic Prompt Optimization via Heuristic Search: A Survey (https://arxiv.org/abs/2502.18746)
- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)의 발전은 자연어 처리(Natural Language Processing) 작업에서 놀라운 성과를 이끌어내고 있습니다. 여기에서 프롬프트 설계(prompt engineering)는 모델의 출력 결과를 안내하는 데 있어 점점 더 중심적인 역할을 하고 있습니다. 수동(prompting) 방법은 효과적일 수 있지만, 일반적으로 직관에 의존하며 시간이 지남에 따라 자동적으로 프롬프트를 개선하지 못합니다.

- **Technical Details**: 본 설문은 자동 프롬프트 최적화(automatic prompt optimization)를 다루며, 최적화가 발생하는 위치, 최적화되는 내용, 최적화를 유도하는 기준, 새로운 프롬프트를 생성하는 연산자(operator) 및 적용되는 반복 검색 알고리즘에 따라 이 방법들을 분류합니다. 최적화는 탐색 문제로 취급되어 휴리스틱 기반 검색 알고리즘(heuristic-based search algorithms)을 사용하여 반복적으로 후보 프롬프트를 평가하고, 성능 피드백(performance feedback)에 따라 적응합니다.

- **Performance Highlights**: 자동 프롬프트 최적화는 기본적으로 인간의 노력을 최소화하고, 수동 실험으로는 발견할 수 없는 고효율의 솔루션을 찾아낼 수 있습니다. 각각의 기술은 다양한 프롬프트 설계 최적화를 다루며, 참조할 수 있는 여러 특화된 데이터셋과 도구를 강조합니다. 마지막으로, 이 설문에서는 LLM의 신뢰성, 적응성 및 윤리적 응용을 위한 주요 공개 과제를 논의하며, 앞으로의 가능성을 제시합니다.



### Random Forest-of-Thoughts: Uncertainty-aware Reasoning for Computational Social Scienc (https://arxiv.org/abs/2502.18729)
Comments:
          11 pages

- **What's New**: 이 논문에서는 Random Forest of Thoughts (RFoT)라는 새로운 큰 언어 모델 프롬프트 방법이 제안됩니다. RFoT는 사회 조사 분석에서 필요한 불확실한 추론을 생성할 수 있도록 LLMs의 다양한 사고 공간을 구축합니다. 이 방법은 응답자의 이전 답변에 따라 설계된 질문의 무작위성을 탐구하여 보다 신뢰할 수 있는 추론 단계를 찾는 데 도움이 됩니다.

- **Technical Details**: RFoT는 체계적으로 다양한 사고를 생성하고 서브 사고(sub-thoughts)를 랜덤으로 선택하여 사고의 숲(forest of thoughts)을 구축하는 방식으로 동작합니다. 이 모델은 Ecoogical Momentary Assessment (EMA)를 활용하여 자동화된 사회 과학 분석의 새로운 트렌드에 부합합니다. 본 연구는 두 개의 데이터셋을 사용하여 RFoT의 효과성과 능력을 입증했습니다.

- **Performance Highlights**: RFoT를 적용한 결과, 이 방법은 최대 78.43%의 성공률과 80.52%의 weighted F1 점수를 기록하여 기존의 사회 조사 분석 문제에 비해 뛰어난 성능을 보여줍니다. 이는 LLMs의 이유 응답을 향상시키는 데 중요한 기여를 하고 있습니다. 실험 결과는 RFoT의 전반적인 성능 향상을 강조하고 있습니다.



### MPO: An Efficient Post-Processing Framework for Mixing Diverse Preference Alignmen (https://arxiv.org/abs/2502.18699)
- **What's New**: 이 논문에서는 인간의 피드백을 통한 강화학습(Reinforcement Learning from Human Feedback, RLHF)의 한계를 극복하기 위해 Mixing Preference Optimization (MPO)라는 새로운 후처리 프레임워크를 제안합니다. MPO는 기존의 보상 모델을 조합하여 단일 목표 정책을 유도할 수 있는 방법을 제시하며, 추가적인 강화학습 과정 없이도 다양한 인간의 선호를 반영할 수 있도록 구성되어 있습니다. 이러한 접근 방식은 다양한 인사이트를 제공하며, 훨씬 효율적인 방법으로 컴퓨터 자원을 절약합니다.

- **Technical Details**: MPO는 각 정책을 로그-선형(log-linear) 방식으로 결합하여 단일 정책으로 통합합니다. 이 과정에서 각 정책의 가중치는 배치 확률적 미러 하강(batch stochastic mirror descent)을 통해 계산됩니다. MPO는 단일 선호와 관련된 정책에 직접 작동하며, 전통적인 RLHF 및 DPO 파이프라인과 원활하게 통합될 수 있습니다. 이러한 방법은 추가적인 강화학습이나 비용이 많이 드는 미세 조정 없이도 실현됩니다.

- **Performance Highlights**: 실험 결과 MPO는 다양한 선호를 효과적으로 균형 있게 반영할 수 있으며, 기존 모델들을 초월하거나 동등한 성능을 보여주면서도 컴퓨팅 비용을 획기적으로 줄였습니다. LLaMA 모델을 통해 감정과 간결함을 정렬하는 데 성공하였다며, Helpful Assistant 작업에서는 세 가지 목표의 최적화를 확장하여 기성 방법들과 비교 평가한 결과 우수한 성능을 확인했습니다. 모든 결과들은 MPO가 기존 다목적 정렬 방식에 비해 뛰어난 성과를 내며 실용적 대안을 제공함을 시사합니다.



### Discriminative Finetuning of Generative Large Language Models without Reward Models and Preference Data (https://arxiv.org/abs/2502.18679)
Comments:
          15 pages, 6 figures

- **What's New**: 본 논문에서는 기존의 preference data나 reward model 없이 대규모 언어 모델(LLM)을 효과적으로 파인튜닝하기 위한 새로운 접근 방식인 Discriminative Fine-Tuning (DFT)을 제안합니다. DFT는 SFT(습관적 파인튜닝)와는 다르게 생성적 접근 방식이 아닌 판별적 접근 방식을 사용하여 '좋은' 출력과 '나쁜' 출력을 구별하는 데 중점을 둡니다. 이로 인해 DFT는 경쟁력 있는 성과를 도출하면서도 preference data가 필요하지 않습니다.

- **Technical Details**: DFT는 입력에 대한 가능한 출력 중에서 답변의 판별 가능성을 모델링하는 확률적 프레임워크를 기반으로 합니다. 이는 SFT가 토큰의 생성 가능성만을 모델링하는 것과 대조적입니다. DFT는 좋은 답변의 판별 가능성을 극대화하기 위한 효율적인 최적화 알고리즘을 제안하여 확장성과 실용성을 보장합니다.

- **Performance Highlights**: 다양한 실험을 통해 DFT가 기존의 SFT보다 지속적으로 우수한 성과를 나타내며, SFT→PO와 비교하여 동등하거나 더 나은 결과를 달성하는 것을 입증했습니다. 이로써 DFT는 pretrained 언어 모델을 향상시키기 위한 새로운 패러다임으로 자리잡을 수 있는 가능성을 보여줍니다.



### Enhancing Text Classification with a Novel Multi-Agent Collaboration Framework Leveraging BER (https://arxiv.org/abs/2502.18653)
- **What's New**: 이 논문에서는 텍스트 분류 모델의 정확성과 강건성을 향상시키기 위해 새로운 다중 에이전트 협업 프레임워크를 소개합니다. BERT를 주 분류기로 활용하여, 낮은 신뢰도를 가진 예측을 Lexical, Contextual, Logic, Consensus, Explainability 에이전트로 구성된 전문 다중 에이전트 시스템으로 전달합니다. 이 협업 방식은 종합적인 분석과 합의 기반의 의사 결정을 가능하게 하여 다양한 텍스트 분류 작업에서 성능을 크게 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 텍스트 분류 정확성과 강건성을 높이기 위해 BERT와 다중 에이전트 시스템을 통합합니다. 시스템은 초기 분류와 낮은 신뢰도 예측에 대한 다중 에이전트 협업의 두 가지 주요 단계로 작동합니다. 초기 모델이 신뢰도 점수를 평가하여 특정 임계값을 초과할 경우 분류 결과를 수용하고, 그렇지 않으면 다중 에이전트 시스템으로 전달하여 추가 분석을 수행합니다.

- **Performance Highlights**: 경험적으로 측정된 결과, 제안된 시스템은 기존의 BERT 기반 분류기와 비교하여 5.5% 높은 정확도를 기록했습니다. 이 결과는 다중 에이전트 시스템의 효과성과 자연어 처리(NLP) 분야에서의 독창성을 강조합니다. 본 연구는 텍스트 분류 작업에서 결합된 에이전트의 협업을 통해 설명 가능성과 강건성을 향상시키는 새롭고 구조화된 접근 방식을 제공합니다.



### Single- vs. Dual-Prompt Dialogue Generation with LLMs for Job Interviews in Human Resources (https://arxiv.org/abs/2502.18650)
Comments:
          11 pages

- **What's New**: 본 연구에서는 인사 (HR) 분야에서 인공지능 (AI) 채용 면접 데이터를 생성하기 위한 두 가지 언어 모델 (LLM) 기반 대화 생성 방법을 비교합니다. 단일 프롬프트 방법과 이중 프롬프트 방법을 사용하여 고품질의 대화 데이터를 생성하는데, 이중 프롬프트 방법이 더 높은 품질을 생성할 수 있는지를 평가합니다. 특히, 인공지능이 면접 생성을 위해 사용되었는지를 판단하는 비교 실험을 진행하여 이중 프롬프트 방법이 최대 10배 높은 승률을 달성함을 입증했습니다.

- **Technical Details**: 대화 생성의 품질을 비교하기 위해, 연구팀은 GPT-4o 및 Llama 3.3 70B를 활용한 두 가지 대화 생성 전략을 채택했습니다. 단일 프롬프트 전략은 완전한 대화를 생성하는 반면, 이중 프롬프트 전략은 각 대화 참여자의 역할을 지정하여 대화를 생성합니다. 연구는 선정된 100개 작업 이력을 바탕으로 인터뷰 데이터를 생성하고, 이를 평가하기 위해 두 가지 LLM을 검증자로 사용하여 쌍별 비교를 수행합니다.

- **Performance Highlights**: 연구 결과, 이중 프롬프트 방법이 단일 프롬프트 방법 대비 10배 높은 승률을 기록했으며, 두 모델의 사용에 상관없이 일관된 품질 차이를 보였습니다. 이러한 차이는 생성된 텍스트의 길이 차이를 고려하지 않도록 설계된 평가 방식에 의해 뒷받침되었습니다. 따라서 이 연구는 높은 품질의 대화 생성을 위한 방법론적 비교를 제공하여 HR 분야의 실제 응용 및 연구에서의 활용 가능성을 시사합니다.



### Steered Generation via Gradient Descent on Sparse Features (https://arxiv.org/abs/2502.18644)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 내부 구조를 수정하여 효율적인 제어를 가능하게 하는 새로운 방법을 제안합니다. 핵심은 Sparse Autoencoders (SAE)를 활용하여 쿼리 임베딩의 희소 표현을 학습하고, 이를 통해 모델의 주의 분포를 정밀하게 조정하는 것입니다. 이러한 접근법은 LLM이 생성하는 피드백의 인지적 복잡성을 체계적으로 조정할 수 있도록 합니다.

- **Technical Details**: LLM의 주의 분포를 조정하기 위해, 이 연구는 SAE를 통해 쿼리 벡터의 희소 표현을 학습합니다. 주의 계층의 활성화 를 조정함으로써, 텍스트 생성의 세부 스타일을 실시간으로 제어할 수 있도록 설계되었습니다. 이 방법론은 프로토타입 네트워크에 영감을 받아, 각 스타일 특성을 잠재 공간의 프로토타입 분포로 나타냅니다.

- **Performance Highlights**: 제안된 방법론은 특별히 설계된 데이터셋을 사용하여 다양한 인지적 스타일을 포착하는 데 성공했습니다. 여러 레이어와 차원에서의 평가를 통해, SAEs가 이러한 클래스 간의 미세한 차이를 효과적으로 구분하고, 원하는 스타일로의 생성 유도를 성공적으로 수행한 것을 입증했습니다.



### Contextual effects of sentiment deployment in human and machine translation (https://arxiv.org/abs/2502.18642)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이 논문은 번역 과정에서 텍스트의 감정이 전체적으로 어떻게 변화할 수 있는지를 보여주며, 특히 그로 인해 자동화된 감정 분석이 미치는 영향을 논의합니다. 인간 및 기계 번역은 목표 언어에서 예상되는 감정의 빈도에 맞는 더 많은 레머(lemmas)를 생성하지만, 기계 번역만이 텍스트의 전체 의미 영역을 줄이는 경향이 있습니다. 이는 특히 인식론적(content) 내용을 가진 단어들에 해당합니다.

- **Technical Details**: 이 연구에서는 정치적 텍스트의 감정 분류를 언어, 정치적 맥락, 대통령 임기 별로 보고합니다. 2000년-2015년 사이의 G8 및 G20 정상 회담에서의 러시아 대통령의 공식 회견 기록을 분석하였고, 이는 전체 코퍼스 수준에서 발생하는 감정의 변화 양상에 대한 통찰을 제공합니다. 감정 리스트의 번역 정확도는 전문 러시아어 번역가에 의해 확인되었으며, 각 언어 데이터 세트의 렘마 수와 관련된 통계 분석이 수행되었습니다.

- **Performance Highlights**: 작성된 통계 결과는 감정의 표현 방식이 정치적 및 시간적 맥락에 따라 어떻게 달라지는지를 보여주었습니다. 예를 들어, G8 및 G20 정상 회담 각각에 대한 기자들의 감정 표현은 명확한 차이를 보였습니다. 특히 기계 번역에서는 감정의 경향이 과장되어 나타났으며, 이는 원래 텍스트에서의 의미 필드를 왜곡하는 결과를 초래할 수 있습니다.



### Chain of Draft: Thinking Faster by Writing Less (https://arxiv.org/abs/2502.18600)
- **What's New**: 이번 연구에서는 Chain of Draft (CoD)라는 새로운 프롬프팅 전략을 발표합니다. 이는 인간의 사고 과정을 반영하여 LLM이 간결하고 정보량이 풍부한 중간 추론 결과를 생성하도록 구상되었습니다. CoD는 Chain of Thought (CoT)와 비교했을 때, 정확성을 유지하면서도 불필요한 장황함을 줄이고, 토큰 사용량을 7.6%까지 감소시킵니다.

- **Technical Details**: CoD는 LLM이 문제 해결 과정에서 중간 단계 대신 중요 정보를 요약하여 출력하도록 유도합니다. 이를 통해 계산 자원 사용을 줄이고, 응답 대기 시간을 단축하는 등의 장점을 제공합니다. CoD는 다단계 추론을 요구하는 여러 벤치마크 테스트에서도 수행되었으며, 실험 결과에서 CoT의 정확도를 초과하거나 동등한 결과를 보였습니다.

- **Performance Highlights**: 본 연구의 실험 결과, CoD는 다양한 복잡한 문제 해결에서 CoT와의 비교에서 높은 정확성과 더불어 낮은 지연 시간 및 비용을 달성했습니다. 이를 통해 CoD는 실제 적용 가능한 환경에서 LLM의 효율성을 크게 향상시키며, LLM의 설계 및 배포에 대한 새로운 방향을 제시합니다.



### Neurobiber: Fast and Interpretable Stylistic Feature Extraction (https://arxiv.org/abs/2502.18590)
- **What's New**: 본 논문에서는 빠르고 해석 가능한 스타일 프로파일링 시스템인 Neurobiber를 소개합니다. 이 시스템은 Biber의 다차원 분석(MDA)에 기반하여, 개방형 소스의 BiberPlus 라이브러리를 통해 96가지 스타일 특성을 예측합니다. Neurobiber는 기존의 오픈 소스 시스템보다 최대 56배 빠르며, 저자 검증 작업에서 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: Neurobiber는 기본적으로 변환기 기반의 신경 태거(neural tagger)로, BiberPlus 기능의 존재 여부를 예측합니다. 이 시스템은 초당 평균 117,000개의 토큰을 처리할 수 있어 대규모 태깅 시나리오를 가능하게 합니다. BiberPlus는 문체 분석에 필요한 96가지 스타일 특성을 통합하여 사용자 친화적인 패키지를 제공합니다.

- **Performance Highlights**: Neurobiber는 CORE 말뭉치에서 MDA의 통찰력을 복제하고, 저자 검증 작업에서 높은 정확도를 유지합니다. 실제로 Neurobiber를 사용하면 Common Crawl 데이터셋과 같은 대규모 데이터셋을 단 13.6일 내에 처리할 수 있지만, 기존 CPU 기반 시스템은 약 756일이 소요됩니다. 이러한 성능은 대규모 스타일 분석과 텍스트 모니터링을 가능하게 하여 범죄 언어학 같은 여러 응용 분야에 도움을 줍니다.



### What are Foundation Models Cooking in the Post-Soviet World? (https://arxiv.org/abs/2502.18583)
- **What's New**: 이번 연구에서는 포스트 소련 국가의 복잡한 문화적 음식 지식을 평가하기 위해 BORSch라 불리는 다중 모드 데이터 세트를 구성했습니다. 이 데이터 세트는 러시아어와 우크라이나어로 된 1147가지와 823가지의 요리를 포함하고 있으며, 포스트 소련 지역을 중심으로 하고 있습니다. 주요 모델들이 포스트 소련 국가의 요리 기원을 정확히 식별하는 데 어려움을 겪고 있다는 사실을 보여주며, 이는 언어와 관련된 나라들을 과대 예측하는 경향을 보입니다. 또한, 우리의 분석을 통해 음식 문화 이해의 부족을 심층적으로 탐구하고자 합니다.

- **Technical Details**: BORSch 데이터 세트는 여러 웹 크롤링 데이터에서 관련 음식을 수집하기 위해 부트스트랩 방식의 엔티티 추출 접근법을 활용하여 구축되었습니다. 수집된 데이터는 인간의 검증을 통해 확인되었으며, 이를 통해 저희는 모델의 평가 기준으로서 텍스트와 이미지 모달리티에 기초한 원산지 질문 응답(QA)을 수행합니다. 또한, 러시아-우크라이나 피진 언어가 QA와 VQA 성능에 미치는 영향도 분석하였습니다. 이러한 작업은 포스트 소련 국가의 문화적 인식을 해결하기 위한 목적을 가지고 있습니다.

- **Performance Highlights**: 모델은 두 언어에서 식별된 요리를 원산지 기준으로 평가할때 각기 다른 성능을 나타내며, 러시아어와 우크라이나어에서 원산지 국가를 과대 예측하는 경향이 있습니다. 특히, QA와 요리에 대한 시각적 설명 요청 실험은 제한된 상관 관계를 나타냅니다. 이를 통해 QA가 문화 이해를 평가하는 데 충분하지 않을 수 있다는 점을 강조합니다. 최종적으로, BORSch 데이터 세트는 연구자들에게 공개되어 지속적 연구를 촉진할 계획입니다.



### Scalable Best-of-N Selection for Large Language Models via Self-Certainty (https://arxiv.org/abs/2502.18581)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 추론 성능을 향상시키기 위한 새로운 기법인 self-certainty를 제안합니다. 기존의 Best-of-N 선택 방법은 외부 보상 모델을 사용하여 응답의 품질을 평가하지만, 이러한 방법은 계산 비용이 크고 여러 한계를 가지고 있습니다. Self-certainty는 LLM의 출력에서 자연적으로 발생하는 확률 분포를 활용하여 응답의 품질을 추정할 수 있으며, 외부적인 보상 모델 없이도 정확도를 높이는 데에 기여할 수 있습니다.

- **Technical Details**: Self-certainty는 LLM이 생성한 토큰 분포를 기반으로 하여 응답의 신뢰도를 평가하는 새로운 메트릭입니다. 이 메트릭은 토큰 분포의 균일 분포로부터의 발산을 측정하며, 발산이 클수록 더 확신에 찬 예측을 의미합니다. 연구진은 self-certainty가 여러 샘플이 있을 때 더 높은 응답 정확도와 상관관계가 있다고 가정하며, 다양한 추론 작업에서 효과적으로 확장할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, self-certainty 기반의 투표가 Best-of-N 선택에서 self-consistency보다 항상 우수한 성능을 보였습니다. self-certainty는 샘플 크기가 증가함에 따라 효과적으로 확장 가능하며, 전통적인 self-consistency 방법이 각기 다른 경로를 처리할 수 없는 한계를 극복할 수 있습니다. 특히, self-certainty는 오픈 엔디드 작업에서도 뛰어난 성능을 발휘하며, 더 나아가 chain-of-thought와 동시에 결합하여 효과적인 추론 성과를 보여줍니다.



### FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models (https://arxiv.org/abs/2502.18573)
- **What's New**: 본 논문은 FactReasoner라는 새로운 사실성 평가기를 제안합니다. 이 모델은 긴 형식의 생성 응답을 평가하기 위해 확률적 추론(probablistic reasoning)을 활용합니다. FactReasoner는 응답을 원자 단위로 분해하고, 외부 지식 소스에서 관련된 문맥(context)을 검색하여, 해당 원자가 검색된 문맥에 의해 뒷받침되는지의 확률을 계산합니다.

- **Technical Details**: FactReasoner는 그래픽 모델(graphical model)을 이용하여 원자와 검색된 문맥 간의 연결을 나타냅니다. 이 모델은 자연어 발화 간의 관계를 나타내는 확률적 인코딩을 사용하며, 원자와 문맥 간의 연역 관계(entailment)와 반대 관계(contradiction)에 기반하여 확률 분포를 구성합니다. 기존의 채점 방식과 달리 FactReasoner는 정보들이 서로 충돌하지 않는다는 가정을 하지 않고, 더 넓은 범위의 사실성을 평가합니다.

- **Performance Highlights**: 실험 결과, FactReasoner는 기존의 최신 프롬프트 기반 방식들보다 사실적 정밀도(factual precision)와 재현율(recall) 면에서 현저하게 향상된 성능을 보여주었습니다. 이 모델은 원자와 모든 검색된 문맥, 그리고 문맥 간의 논리적 관계를 활용하여 더 많은 지원되는 원자를 올바르게 식별할 수 있어, 사실성 평가에서 안정성이 높습니다.



### MixLLM: Dynamic Routing in Mixed Large Language Models (https://arxiv.org/abs/2502.18482)
Comments:
          11 pages, 7 figures, accepted by NAACL 2025 main conference

- **What's New**: 새롭고 혁신적인 연구인 MixLLM은 다이나믹 컨텍스트 밴딧 기반의 라우팅 시스템을 개발하여, 쿼리와 LLM의 최적 매핑을 가능하게 합니다. 이 시스템은 쿼리 태그를 활용하여 쿼리 임베딩을 향상시키고, 각각의 LLM에 대한 응답 품질 및 비용을 추정하는 경량화된 예측 모델을 설계합니다. Mixed LLM은 응답 품질과 비용, 지연 시간 간의 균형을 최적화하여 높은 효율을 실현합니다.

- **Technical Details**: MixLLM은 InsTag 모델에서 생성된 태그를 사용하여 쿼리 표현을 개선하는 태그 향상 임베딩 모델을 제안합니다. 예측 모델은 각 LLM의 응답 품질과 비용을 평가하며, 메타 의사결정자는 이러한 예측을 기반으로 최적의 LLM을 선택합니다. 이 과정은 새로운 LLM이 도입되더라도 시스템 전체 재훈련이 필요 없도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, MixLLM은 응답 품질에서 GPT-4의 97.25%를 달성하고, 비용은 24.18%로 유지하여 응답 품질과 비용, 지연 시간 간의 최적의 균형을 실현하였습니다. 또한, 지연 패널티를 도입하여 교통 혼잡 및 높은 지연 문제를 피하고, 다양한 환경 및 사용자 피드백에 적응하는 지속적인 훈련의 장점을 누릴 수 있습니다.



### Project Alexandria: Towards Freeing Scientific Knowledge from Copyright Burdens via LLMs (https://arxiv.org/abs/2502.19413)
Comments:
          Technical Report

- **What's New**: 이 논문은 과학 지식을 보호하는 저작권 법의 법적 및 기술적 장점을 강조하며, LLMs(대형 언어 모델)를 사용해 학술 문서를 'Knowledge Units'로 변환할 수 있는 새로운 접근 방식을 제안합니다. 'Knowledge Units'는 스타일적 요소 없이 정보만을 캡처하여, 저작권 우려 없이 과학적 지식을 공유할 수 있는 법적 근거를 제공합니다. 이 방법은 연구자들이 중요한 사실을 재사용할 수 있도록 해줍니다.

- **Technical Details**: Knowledge Units(KUs)는 개별段락(단락)에서 엔터티, 관계, 속성을 추출하여 구조화된 데이터를 생성합니다. 각 KU는 원문에서 추출된 짧은 텍스트 단편을 기반으로 하며, 개념 간의 연결과 특성을 포함하고 있습니다. LLM을 사용하여 처리하기 때문에, 원문을 복사하지 않고도 중요한 사실이 저장됩니다.

- **Performance Highlights**: 실험을 통해 Knowledge Units를 사용하여 다수의 질문-응답 실험을 수행한 결과, 연구 분야에 따라 95% 이상의 정보 보존율을 기록하였습니다. 이는 LLM이 Knowledge Units를 통해 제공된 데이터로도 원 텍스트와 유사한 퍼포먼스를 보인다는 것을 보여줍니다. 이러한 접근 방식은 연구자들 사이의 정보 접근성을 높이고, 과학적 대화를 촉진하는 데 기여할 것으로 기대됩니다.



### ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models (https://arxiv.org/abs/2502.19409)
Comments:
          Code, dataset, and checkpoints are publicly available at this https URL

- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)의 이미지 시퀀스에 대한 추론 능력을 향상시키기 위해 ImageChain이라는 새로운 프레임워크를 소개합니다. 기존의 모델들은 이미지들을 독립적으로 다루거나 전체 장면을 요약하는 것에 그쳤지만, ImageChain은 시각적 시퀀스를 다중 턴 대화로 모델링하여 시간적 의존성을 명확하게 포착할 수 있게 합니다. 이를 통해 다음 장면 설명(next-scene description) 작업에서 비약적인 성과 향상을 달성하였습니다.

- **Technical Details**: ImageChain은 이미지와 그에 해당하는 텍스트 설명을 혼합하여 다음 장면 설명을 생성하기 위한 연속적인 문맥을 구축합니다. 이 프레임워크는 약 4000개의 훈련 샘플만으로도 SimRate라는 메트릭에서 평균 3.7%에서 19%로의 성과 향상을 보여줍니다. 또한, 다양한 문맥 길이에 걸쳐 훈련하는 것이 일관되게 개선된 성능을 보이며, 이는 시퀀스 추론 능력을 향상시키는데 도움을 줍니다.

- **Performance Highlights**: ImageChain은 로봇 공학 분야에서 F1 점수 27.1을 기록하여 기존 모델보다 두 배 높은 성과를 달성하였습니다. 또한 만화와 같은 다른 구조적 설정에서도 성과 향상을 보입니다. 최종적으로, StoryFrames 데이터셋을 활용하여 다양한 맥락에서 시퀀스 이미지-텍스트 추론 연구를 지원하는 고품질 샘플을 제공합니다.



### Learning Code-Edit Embedding to Model Student Debugging Behavior (https://arxiv.org/abs/2502.19407)
- **What's New**: 이번 연구에서는 프로그래밍 과제의 피드백 제공 방식에 대한 새로운 접근법을 제안합니다. 특히, 학생들이 제출한 코드 간의 코드 편집(Coding Edit) 임베딩을 학습하는 인코더-디코더 모델을 통해 디버깅 행동을 캡처하고, 개인화된 코드 제안 생성이 가능하도록 합니다. 이는 학습자의 코딩 스타일을 유지하면서도 테스트 케이스의 정확성을 개선할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구팀은 코드 제출의 서로 다른 쌍을 학습하여 코드 편집을 표시하는 방향으로 나아가는 contrastive learning 기법을 사용합니다. 이때, 코드 편집 간의 유사성을 측정하기 위해 테스트 케이스의 결과를 이용하며, 이로 인해 학생들의 디버깅 행동 분석을 가능하게 합니다. 또한, 모델은 코드 재구성을 위한 목표와 뛰어난 결과를 위해 정규화 손실을 활용하여 임베딩 공간의 일관성을 유지합니다.

- **Performance Highlights**: 실제 학생 코드 제출 데이터셋을 활용한 실험 결과, 제안된 모델이 코드 재구성과 개인화된 코드 제안을 더 잘 수행하는 것으로 나타났습니다. 또한, 이 모델은 일반적인 오류 패턴 및 디버깅 행동을 분석하는 데 있어 유용성을 보여줍니다. 반면에 기존의 LLM 기반 모델은 종종 직접적인 해결책을 제공하여 학습 기회를 감소시키는 경향이 있습니다.



### TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding (https://arxiv.org/abs/2502.19400)
- **What's New**: 이 연구에서는 TheoremExplainAgent라는 새로운 에이전트 기반 시스템을 소개합니다. 이 시스템은 Manim 애니메이션을 사용하여 길이가 5분 이상의 정리 설명 비디오를 생성할 수 있는 능력을 가지고 있습니다. 연구자들은 240개의 정리에 대한 기준을 포함한 TheoremExplainBench라는 벤치마크를 개발하여 다중 모드 정리 설명을 평가하는 체계적인 방안을 제안합니다.

- **Technical Details**: TheoremExplainAgent는 비디오 생성의 기획, 내레이션, 그리고 Python 애니메이션 스크립트를 생성하는 코딩 에이전트로 구성되어 있습니다. 이 시스템은 4개의 STEM 분야에서 다양한 정리 비디오를 생성할 수 있는 능력을 보여주며, 생성된 설명은 사실 정확성과 지각 품질의 5가지 차원에서 평가됩니다. 연구 결과에 따르면 에이전트 기반 기획이 상세한 긴 형식의 비디오 생성에 필수적이며, o3-mini 모델은 93.8%의 성공률과 0.77의 전반적인 점수를 기록했습니다.

- **Performance Highlights**: TheoremExplainAgent는 최대 10분 길이의 확장된 비디오 설명을 생성할 수 있어 기존 에이전트 없는 방법에 비해 상당한 진전을 보여주었습니다. 하지만 생성된 애니메이션은 종종 시각적 레이아웃에 사소한 문제가 발생하여 미세한 정렬 오류나 겹치는 도형이 나타나는 경우가 많았습니다. 이러한 시각적 오류는 보다 복잡한 정리에서 더욱 두드러지게 나타나, AI 시스템의 깊은 추론 결함을 드러내는 데 도움이 되었습니다.



### Residual Speech Embeddings for Tone Classification: Removing Linguistic Content to Enhance Paralinguistic Analysis (https://arxiv.org/abs/2502.19387)
- **What's New**: 본 연구에서는 음성 처리에서 사용되는 자가 감독 학습 모델의 음성 임베딩에서 언어적 콘텐츠와 비언어적 (paralinguistic) 특징을 분리하는 방법을 제안합니다. 기존의 음성 임베딩 모델들은 언어적 정보와 비언어적 정보가 얽혀 있어 음성의 톤을 독립적으로 분석하는 데 어려움이 있었습니다. 우리는 회귀 분석을 통해 언어적 콘텐츠를 제거하고, 잔여 임베딩을 목소리의 톤 표현으로 활용하는 새로운 접근법을 개발했습니다.

- **Technical Details**: 제안된 방법은 wav2vec2와 OpenAI의 text-embedding-ada-002 모델을 사용하여 음성과 텍스트에 대한 임베딩을 추출하고, 이를 기반으로 회귀 분석을 통해 잔여 임베딩을 추출하는 일련의 과정을 포함합니다. 잔여 임베딩은 언어적 콘텐츠를 피하고 비언어적 특성을 유지하며, 이는 톤 분류의 정확도를 높이는 데 기여합니다.

- **Performance Highlights**: 실험 결과에 따르면, 잔여 임베딩을 활용한 톤 분류는 기존의 원시 음성 임베딩에 비해 성능이 크게 향상되었습니다. 이러한 개선은 간단한 로지스틱 회귀 모델을 사용할 때도 확인되었으며, 톤과 관련된 정보가 효과적으로 유지되었음을 시각화하여 증명했습니다. 이 연구의 결과는 감정 분석, 화자 특성 파악, 비언어적 음성 처리와 같은 다양한 응용 분야에서 잔여 임베딩 활용의 가능성을 보여줍니다.



### FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users (https://arxiv.org/abs/2502.19312)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 Few-Shot Preference Optimization (FSPO)라는 새로운 프레임워크를 소개하며, 개인화를 위한 새로운 방법론을 제안합니다. FSPO는 보상 모델링을 메타 학습 문제로 재구성하여 사용자의 몇 가지 라벨이 붙은 선호도를 통해 빠르게 적응하는 LLM의 능력을 활용합니다. 이 연구는 공공의 LLM로부터 1백만 개 이상의 합성 개인화 데이터를 생성하여 실제 사용자에 적응할 수 있는 기반을 마련했습니다.

- **Technical Details**: FSPO는 사용자 세부정보의 연쇄적 사고 과정(Chain-of-Thought, COT)을 활용하여 성능을 향상시키며, 이러한 방식은 개인화된 정보 생성을 가능하게 합니다. 이 과정에서 기존의 채점 방식과는 달리, 모델은 보상 기능을 개인화하고, 개별 사용자 집단에 맞춘 다양한 반응 생성을 가능하게 합니다. 모델 학습은 합성 데이터와 실제 데이터 사이에서의 전이(Transfer) 특성을 고려하여 다양성과 일관성을 중시합니다.

- **Performance Highlights**: FSPO는 영화 리뷰, 교육 배경에 따른 교육적 적응, 일반적인 질문 응답 분야에서 합성 사용자 1,500명을 대상으로 한 평가에서 평균 87%의 Alpaca Eval 승률을 기록했습니다. 또한, 실제 사용자에 대한 개방형 질문 응답에서도 72% 승률을 달성하여 FSPO의 개인화된 모델이 효과적임을 입증했습니다. 이러한 결과는 FSPO가 사용자 요구에 더 잘 부응할 수 있는 가능성을 보여줍니다.



### Isolating Language-Coding from Problem-Solving: Benchmarking LLMs with PseudoEva (https://arxiv.org/abs/2502.19149)
- **What's New**: 본 논문에서는 LLMs(대규모 언어 모델)의 코드 생성 성능을 연구하기 위해 PseudoEval이라는 다국어 코드 생성 벤치마크를 개발했습니다. 기존의 HumanEval 및 MBPP 벤치마크는 문제 해결 능력과 언어 코딩 능력을 분리하는 데 한계가 있었지만, PseudoEval은 문제를 설명하는 의사 코드(Pseudocode)를 제공하여 두 가지 능력을 구분할 수 있게 설계되었습니다. 이러한 접근법을 통해 LLMs의 병목 현상을 보다 명확히 이해할 수 있게 되었습니다.

- **Technical Details**: PseudoEval 벤치마크는 1,060개의 주제로 구성되며, 문제 및 그에 상응하는 해결책뿐만 아니라 중간 해결책을 의사 코드 형태로 제공합니다. 문제 해결(Task: Problem Solving)과 언어 코딩(Task: Language Coding)의 두 가지 작업을 통해 LLM의 성능을 평가하며, 각각의 작업은 LLM의 별개의 능력을 측정합니다. 이 논문에서는 의사 코드를 통해 LLM의 코딩 능력을 결합하여 분석하는 방법을 제시하고 있습니다.

- **Performance Highlights**: 연구 결과, Python에서의 LLM의 병목 현상은 문제 해결 능력이며, C++와 Rust에서 더 많이 나타나는 것은 언어 코딩 능력입니다. 대부분의 해결책은 언어에 구애받지 않는 특성을 가지며, LLM들은 특정 프로그래밍 언어에서 문제 해결 능력을 배우는 것이 충분할 수 있습니다. 마지막으로, 자동 생성된 의사 코드는 인간이 작성한 것보다 품질이 유사하거나 더 뛰어난 것으로 나타나, 기존 벤치마크 제도의 연장 가능성을 제시하고 있습니다.



### IndicEval-XL: Bridging Linguistic Diversity in Code Generation Across Indic Languages (https://arxiv.org/abs/2502.19067)
- **What's New**: 이번 논문에서는 다국어 코드 생성 기능을 평가하기 위한 포괄적인 벤치마크, IndicEval-XL을 소개합니다. 이 벤치마크는 6개의 주요 인도 언어를 포함하여, 전 세계 인구의 약 14%가 사용하는 언어들을 포괄합니다. 이러한 프레임워크는 12개의 프로그래밍 언어와 연결되어 다국어 생성의 평가를 혁신적으로 발전시키고 있습니다. 또한, AI 기반 개발 도구의 접근성을 높이고 다양한 언어를 사용하는 개발자들이 이를 활용할 수 있도록 하려는 목적도 가지고 있습니다.

- **Technical Details**: 코드 생성을 위한 현대의 접근법은 자연어를 프로그래밍 코드로 변환하는 과정에서 컴퓨터 언어학과 인공지능을 융합합니다. 초기 모델들은 추상 구문 트리 (Abstract Syntax Trees, AST)를 사용하여 사용자 지시의 변형을 수용했습니다. 신경망 아키텍처가 도입되면서 더 복잡한 프로그램 합성 작업을 수행할 수 있는 능력이 확장되었습니다. 특히, Neural Program Interpreter와 같은 모델들은 다단계 솔루션에서의 순차적 의존성을 관리하여 실행 경로에 대한 깊은 학습을 가능하게 했습니다.

- **Performance Highlights**: IndicEval-XL 평가는 6개의 인도 언어를 포괄하며 80개의 패럴렐 프로그래밍 문제를 포함하여 총 6,720개의 코딩 문제로 구성되어 있습니다. 데이터셋은 BLEU 및 METEOR와 같은 평가 지표를 통해 품질이 보장되어 있으며, LLM의 성능 평가를 위한 강력한 기준을 제시합니다. 이 연구는 인도 지역 언어의 다양성을 강조하며, 다국어 및 교차언어 코드 생성에서의 포괄성을 증대시키는 방향으로 나아가고 있습니다.



### Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments (https://arxiv.org/abs/2502.19024)
Comments:
          Accepted by ICRA 2025

- **What's New**: 최근 Vision-and-Language Navigation (VLN) 분야의 연구에서, 저자들은 낮은 시야각을 가진 사족보행 로봇과 인간 중심의 지시사항 간의 불일치 문제를 해결하기 위한 Ground-level Viewpoint Navigation (GVNav) 방안을 제안하고 있습니다. 이는 VLN에서 다양한 높이에서의 시각적 관찰의 일반화 격차를 강조한 최초의 시도입니다. 본 연구는 광범위한 실험을 통해 시뮬레이션 환경 및 실제 환경에서의 성능 향상을 입증하였습니다.

- **Technical Details**: 이 논문은 사족보행 로봇이 낮은 높이에서 데이터를 수집하는데 직면하는 여러 가지 문제를 다루고 있습니다. 저자들은 비슷한 특징에 적절한 가중치를 부여함으로써 지역적 관측의 장애물을 처리할 수 있는 적응형 정보 수집 모듈을 개발하였습니다. 또한, HM3D 및 Gibson 데이터셋의 연결 그래프를 활용하여 공간적 사전지식을 강화하는 방법을 제안하며, 이를 통해 실제 복잡한 환경에서의 경로 예측 능력을 향상시켰습니다.

- **Performance Highlights**: GVNav 접근법은 실제 환경과 시뮬레이션 환경 모두에서 성능을 크게 개선하는 결과를 나타내었습니다. 저자들은 Xiaomi Cyberdog을 사례 연구로 삼아 다양한 시각적 정보의 차이를 분석하고, 깊이 기반의 경로 예측이 낮은 시야각에서 어떤 영향을 받는지를 평가하였습니다. GVNav는 특히 복합적인 환경에서 사족보행 로봇의 작업 효율성을 높이는 데 기여하고 있습니다.



### (Mis)Fitting: A Survey of Scaling Laws (https://arxiv.org/abs/2502.18969)
Comments:
          41 pages, 3 figure, first two authors contributed equally. ICLR, 2025

- **What's New**: 이번 논문은 대규모 모델 훈련 과정에서의 스케일링 법칙(scaling laws)의 중요성에 대해 논의합니다. 저자들은 기존 연구들이 도출한 결론의 불일치를 분석하며, 최적의 토큰-파라미터 비율(optimal token to parameter ratio)과 같은 쟁점들을 탐구합니다. 또한, 50편 이상의 스케일링 관련 논문을 조사하여 재현성을 높이기 위한 체크리스트(checklist)를 제안합니다.

- **Technical Details**: 스케일링 법칙은 손실(loss)과 모델 크기, 데이터셋 크기 간의 관계를 설명하는 파워 법칙(power law)으로 규명됩니다. 연구자들은 모델 훈련을 통해 얻은 데이터에서 이 법칙을 적합(fitting)하기 위해 다양한 하이퍼파라미터(hyperparameters)와 초기화 조건을 설정해야 합니다. 논문에서는 이러한 프로세스의 세부적인 변화가 결과에 미치는 영향을 구체적으로 설명합니다.

- **Performance Highlights**: 조사한 51편의 논문에서 중요한 세부사항이 종종 누락되어 재현성에 significant 영향을 미친다고 강조합니다. 분석 코드가 없는 42편 중 단 19편만이 코드 조각을 제공하고, 23편은 최적화 과정에 대한 설명이 부족하며, 15편은 FLOP 또는 파라미터 수의 계산 방법을 설명하지 않습니다. 이러한 누락된 정보는 최종 결과에 중대한 영향을 미치는 것으로 나타났습니다.



### Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models (https://arxiv.org/abs/2502.18943)
Comments:
          Accepted by USENIX Security 2025

- **What's New**: 본 연구에서는 Label-only Membership Inference Attacks (MIAs)의 취약성을 분석합니다. 기존의 MIA 연구들은 대개 모델의 전체 로그잇에 접근해야 하며, 이는 실제 데이터에서는 접근이 불가능합니다. 따라서, 단순히 생성된 토큰만으로도 미소속 여부를 추정할 수 있는 가능성을 탐구합니다.

- **Technical Details**: 본 논문에서 제안하는 PETAL 공격은 PEr-Token semAntic simiLarity에 기반한 미소속 추정 방법입니다. 토큰 간의 의미적 유사성을 활용해 출력을 근사화하고, 이를 통해 미소속 여부를 결정합니다. 기존의 라벨 기반 접근 방식은 미소속 확인에서 효과적이지 않다는 점을 지적하며, 이는 일반화 능력이 뛰어난 LLM의 특성과 관련이 있습니다.

- **Performance Highlights**: PETAL 방법은 WikiMIA 및 MIMIR 벤치마크에서의 실험을 통해 기존의 라벨 기반 공격들을 넘어서는 성능을 보여줍니다. 여러 개의 공개 LLM에 대해 실험을 실시하였으며, PETAL은 기존의 로그잇 기반 공격과도 비슷한 성능을 발휘합니다. 이러한 결과는 PETAL의 강력한 효과성을 입증하는 것으로, 실제 응용 가능성을 시사합니다.



### Clip-TTS: Contrastive Text-content and Mel-spectrogram, A High-Huality Text-to-Speech Method based on Contextual Semantic Understanding (https://arxiv.org/abs/2502.18889)
- **What's New**: Clip-TTS는 기존의 전통적인 Text-to-Speech (TTS) 방법의 한계를 극복하기 위해 개발된 혁신적인 다중 모달 접근 방식을 기반으로 하는 새로운 TTS 방법이다. 이 방법은 Clip 구조를 사용하여 텍스트 인코딩 단계에서 텍스트 콘텐츠와 실제 멜-스펙트로그램(mel-spectrogram) 간의 연결을 설정하며, 이를 통해 텍스트 인코더가 글로벌 컨텍스트의 진정한 의미를 학습할 수 있도록 한다. Clip-TTS는 Transformer의 기본 구조를 채택하여 빠른 추론 속도를 달성하며, 실험 결과는 Clip-TTS가 LJSpeech 및 Baker 데이터셋에서 탁월한 품질의 음성을 생성함을 보여준다.

- **Technical Details**: Clip-TTS는 Contrastive Language-Image Pretraining (Clip)과 텍스트-투-스피치(TTS) 합성을 통합하여 더 표현력이 풍부하고 상황에 맞는 음성 생성을 가능하게 한다. 기존의 TTS 모델과 달리 Clip-TTS는 풍부한 의미 표현을 활용하여 텍스트와 음성 간의 더 깊은 관계를 학습하고, 이를 통해 다양한 음성 스타일과 맥락에 대해 일반화할 수 있는 능력을 향상시킨다. 이 방법은 명시적인 레이블이나 방대한 주석 데이터 없이도 표현력의 제어가 가능하여, 다채로운 톤과 감정으로 음성을 생성하는 데 적합하다.

- **Performance Highlights**: Clip-TTS는 멜-스펙트로그램(mel-spectrogram) 표현을 활용하여 텍스트 콘텐츠를 이해하고 통합할 수 있는 능력이 뛰어나며, 이를 통해 인간과 유사한 음성을 생생하게 생성한다. 실험 결과, Clip-TTS는 최근 음성 생성 성능의 최첨단인 MOS 점수를 기록하였으며, 다양한 감정의 음성 샘플을 잘 처리함으로써 인터랙티브한 응용 프로그램에서 스토리텔링과 캐릭터 음성 합성에 특히 유용하게 활용될 수 있다.



### Multi-LLM Collaborative Search for Complex Problem Solving (https://arxiv.org/abs/2502.18873)
- **What's New**: 이번 논문에서는 Mixture-of-Search-Agents (MoSA) 패러다임을 제안하여, 여러 개의 Large Language Models (LLMs)의 집단적 전문성을 활용하여 복잡한 추론(task) 문제를 해결합니다. MoSA는 독립적인 탐색과 반복적인 세부 조정을 통해 다양한 추론 경로를 통합함으로써 단일 모델 접근의 한계를 극복합니다. 특히, Monte Carlo Tree Search (MCTS)를 기반으로 하여 여러 에이전트가 추론 단계를 제안하고 집계하는 기능을 포함하고 있습니다.

- **Technical Details**: MoSA의 기본 접근 방식은 단계별 탐색 기반 추론을 통해 문제를 유도된 그래프에서의 탐색으로 나누는 것입니다. 이때 각 상태는 지금까지 생성된 추론 단계의 집합을 나타내며, 행동 공간은 다음 추론 단계를 기반으로 합니다. 여러 LLM이 서로 독립적으로 또는 협력적으로 여러 추론 단계를 제안하는 것을 통해, 나쁜 지역 최적화(local optima) 문제를 회피하고 추론 정확도를 높이는 방식으로 설계되었습니다.

- **Performance Highlights**: 본 연구에서는 MoSA가 단일 LLM 대비 평균 1.71%의 추론 정확도 개선을 보이며, 다수의 추론 기준에서도 일관된 성능 개선을 나타냄을 입증했습니다. 다중 에이전트의 협업과 탐색 기반 추론의 시너지를 강조하며, 과제 관점에서 다양성과 품질의 균형이 평가되는 중요한 문제임을 확인했습니다. 추가적인 행동 세트를 사용한 실험은 MoSA의 다양한 탐색 작업에 대한 견고성을 입증하였습니다.



### Towards an AI co-scientis (https://arxiv.org/abs/2502.18864)
Comments:
          81 pages in total (main 38 pages, appendix 43 pages), 13 main figures, 40 appendix figures, 1 main table, 2 appendix tables, 143 main references, 7 appendix references

- **What's New**: 이번 연구에서는 Gemini 2.0을 기반으로 한 AI 공동 과학자(AI co-scientist)를 도입했습니다. 이 시스템은 과학자들이 새로운 가설을 생성하고 이를 엄격한 실험적 검증을 통해 확인하는 데 도움을 줍니다. AI 공동 과학자는 이전의 증거를 기반으로 하여 혁신적인 연구 가설과 제안을 형성하는 데 중점을 두고 있습니다.

- **Technical Details**: 시스템은 생성(generate), 토론(debate), 발전(evolve)의 접근 방식을 결합하여 가설 생성을 수행하며, 융통성 있는 컴퓨팅 스케일링을 위한 비동기(task execution) 아키텍처를 적용합니다. 이 외에도 자기 개선(self-improving) 가설 생성을 위한 토너먼트 진화(tournament evolution) 과정을 포함하고 있습니다. 자동화된 평가를 통해 테스트 시간 컴퓨팅(test-time compute)의 지속적인 이점이 입증되었고, 이는 가설의 품질 향상에 기여하고 있습니다.

- **Performance Highlights**: 이 시스템은 약물 재사용(drug repurposing), 새로운 표적 발견(novel target discovery), 박테리아 진화 및 항균 저항 메커니즘 설명 분야에서 개발 및 검증에 집중하고 있습니다. 예를 들어, 급성 골수성 leukemia 치료를 위한 유망한 후보 물질을 제안하였고, 이는 저자 제안의 임상 적용 가능 농도에서 종양 억제 효과를 보였습니다. 또한 간 섬유증을 위한 새로운 후생유전학적(target) 표적을 제안하였으며, 이는 인체 간 유기체에서 anti-fibrotic 활동과 세포 재생으로 검증되었습니다.



### Holistic Audit Dataset Generation for LLM Unlearning via Knowledge Graph Traversal and Redundancy Remova (https://arxiv.org/abs/2502.18810)
Comments:
          11 pages, 4 figures

- **What's New**: 최근 대규모 언어 모델(LLMs)에서 민감한 정보 제거 및 개인정보 보호와 같은 요구 사항이 증가함에 따라, Machine Unlearning을 통한 비법적 정보 삭제가 주목받고 있습니다. 이 논문에서는 기존의 신뢰할 수 있는 평가 기준이 부족한 현실을 지적하며, HANKER라는 새로운 자동화된 프레임워크를 제안하여 보다 포괄적인 감사 데이터를 생성할 수 있도록 합니다. HANKER는 지식 그래프(Knowledge Graphs)를 활용하여 그 범위를 넓히고 중복된 지식을 제거합니다.

- **Technical Details**: HANKER는 잊고 있는 데이터셋(Forget Dataset)과 유지하는 데이터셋(Retain Dataset)을 구조화된 지식 그래프로 변환하여 감사 과정의 범위를 명확히 정의합니다. 이 알고리즘은 각 KG 엣지를 최소 단위로 취급하여 질문-답변 쌍을 생성할 때 겹치는 정보를 필터링 합니다. 이를 통해 헌신적인 감사 절차를 보장하고, 고품질의 테스트 질문을 포함함으로써 포괄적이고 정확한 평가를 가능하게 합니다.

- **Performance Highlights**: MUSE 벤치마크에 HANKER를 적용한 결과, 뉴스 및 도서 데이터셋에서 각각 69,000개 및 111,000개의 감사 사례를 성공적으로 생성하였으며, 지난 연구에서 식별하지 못한 수천 건의 지식 기억 사례를 발견하였습니다. 특히, 중복된 지식이 비법적 각성 효과성 지표에 크게 영향을 미치는 것으로 나타났으며, 이는 중복이 기억 측정치를 인위적으로 부풀릴 수 있음을 강조합니다. 따라서 시스템적 중복 제거의 필요성이 강조되며, HANKER는 이 평가의 정확성을 제고하는 데 기여할 것입니다.



### Towards Optimal Multi-draft Speculative Decoding (https://arxiv.org/abs/2502.18779)
- **What's New**: 이 연구에서는 Multi-Draft Speculative Decoding (MDSD)이라는 새로운 접근법을 소개하고 있습니다. 이 방법은 각 토큰을 생성할 때 소규모 초안 모델이 여러 개의 초안을 생성하고, 목표 LLM이 이를 병렬적으로 검증하여 최종 출력이 목표 모델 분포에 일치하도록 합니다. 또한, 기존 검증 알고리즘과 이론적 상한선 간의 격차를 측정하는 연구도 진행하였습니다.

- **Technical Details**: MDSD의 주요 설계 선택 요소는 초안 샘플링 방법과 검증 알고리즘입니다. 고정된 초안 샘플링 방법에 대해 최적 수용율(optimal acceptance rate)은 최적 수송 문제(optimal transport problem)의 해가 되지만, 이 문제의 복잡성으로 인해 최적 수용율을 구하기 어렵습니다. 이 논문에서는 최적 수송 문제의 쌍대 문제(dual of the optimal transport problem)를 다루어 최적 수용율을 효율적으로 계산할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 연구 결과, 초안 샘플링 방법이 최적 수용율에 미치는 영향이 크며, 교체 없이 샘플링(sampling without replacement)이 교체와 함께 샘플링(sampling with replacement)보다 더 우수한 성과를 보였습니다. 또한, 기존 검증 알고리즘은 둘 다 이론적 상한에 도달하지 못하는 것으로 나타났습니다. 이러한 결과는 조심스럽게 설계된 초안 샘플링 방법이 최적 수용율을 개선하고 이론적 상한에 근접하는 검증 알고리즘 개발에 기여할 수 있음을 시사합니다.



### M2-omni: Advancing Omni-MLLM for Comprehensive Modality Support with Competitive Performanc (https://arxiv.org/abs/2502.18778)
- **What's New**: M2-omni는 최첨단 오픈 소스 omni-MLLM으로, GPT-4o와 경쟁할 만한 성능을 달성하였습니다. 이 모델은 통합된 멀티모달 시퀀스 모델링 프레임워크를 채택하여, 대형 언어 모델(LLM)이 각기 다른 모달리티(모드)에 대한 포괄적인 이해와 생성 능력을 쌓를 수 있도록 합니다. 특히, M2-omni는 오디오, 비디오, 이미지, 텍스트의 임의 조합을 처리하고, 이를 통해 고급의 실시간 상호작용 경험을 가능하게 합니다.

- **Technical Details**: M2-omni의 훈련에서 가장 큰 과제는 다양한 모달리티 간 데이터 양과 수렴 속도의 불균형입니다. 이를 해결하기 위해 우리는 사전 훈련 단계에서 단계 균형 전략을 제안하여 특정 모달리티 데이터의 양 불균형을 처리합니다. 또한, 지침 조정 단계에서는 동적 적응 균형 전략을 도입하여 각 모달리티의 훈련 진행을 동기화하여 최적의 수렴을 보장합니다.

- **Performance Highlights**: M2-omni는 순수 텍스트 작업에서도 강력한 성능을 유지하는 것을 최우선 과제로 하여, 훈련 과정을 통하여 언어 이해 능력의 견고함을 보장합니다. 현재 M2-omni는 GPT-4o에 대해 매우 경쟁력 있는 오픈 소스 모델로서, 종합적인 모달리티 및 작업 지원과 뛰어난 성능을 자랑합니다. 이 모델의 발전을 통해 향후 omni-MLLM에 대한 연구가 더욱 촉진될 것으로 기대됩니다.



### Reward Shaping to Mitigate Reward Hacking in RLHF (https://arxiv.org/abs/2502.18770)
Comments:
          19 pages

- **What's New**: 이번 연구에서는 인간 피드백을 통한 강화학습(Reinforcement Learning from Human Feedback, RLHF)의 문제를 해결하기 위한 새로운 보상 형태인 Preference As Reward (PAR)를 제안합니다. PAR는 보상 모델에 내재된 선호를 활용하여 RL의 신호를 형성하는 혁신적인 방법으로, 이를 통해 기존의 보상 해킹(reward hacking) 문제를 극복할 수 있습니다. 이 연구는 보상 형성 방법에 대한 체계적인 분석을 제공하며, 세 가지 주요 설계 원칙을 도출했습니다.

- **Technical Details**: 우리는 RLHF에서 광범위하게 사용되는 보상 형성 기법들을 조사하고, 이러한 기법의 적용을 위해 잘 정의된 설계 원칙을 설정하려고 했습니다. 특히 보상 모델의 특정 임계값을 초과할 경우 보상 해킹이 시작되며 모델의 승률이 감소하는 경향을 관찰했습니다. 그리고 PAR는 이론적으로 시그모이드 함수를 사용하여 중앙 보상(centered reward)을 적용함으로써 초기 학습을 가속화하고 학습 안정성을 보장합니다.

- **Performance Highlights**: PAR는 두 개의 기본 모델(Gemma2-2B 및 Llama3-8B)과 두 개의 데이터셋(Ultrafeedback-Binarized 및 HH-RLHF)에서 평가된 결과, 다른 보상 형성 방법보다 뛰어난 성능을 보여주었습니다. AlpacaEval 2.0 벤치마크에서 PAR는 경쟁 상대보다 최소 5%포인트 높은 승률을 기록하였으며, 데이터 효율성도 뛰어나 단일 참조 보상만으로 최적의 성능을 낼 수 있습니다. 또한 두 번의 훈련 에포크 후에도 보상 해킹에 대한 강인성을 유지하는 특성을 가지고 있습니다.



### Like Father, Like Son: Kinship-Aware Preference Mapping (KARMA) for Automatic Alignment in Large Language Models (https://arxiv.org/abs/2502.18744)
Comments:
          14 pages,5 figures,3 tables,4 graphs

- **What's New**: 최근 Large Language Model (LLM) 정렬의 발전은 사전 훈련된 모델을 활용하여 인간 주석의 비용을 줄이려는 노력을 포함합니다. 기존 방법들이 본질적으로 다른 능력의 모델 간에 응답을 비교하는 경향이 있었지만, 이로 인해 유의미하지 않은 구분이 발생했습니다. 이를 해결하기 위해, 우리는 유사한 능력을 가진 모델 간 응답을 짝짓는 Kinship-Aware pReference MApping (KARMA) 프레임워크를 제안합니다.

- **Technical Details**: KARMA는 모델의 유사성을 고려하여 동일한 복잡성 및 품질을 갖춘 생성물에 대해 우선 선호를 비교합니다. 이러한 방법은 응답의 품질 격차(Respons Quality Gap)가 줄어들 수 있도록 하여 더 정밀하고 의미 있는 preference 신호를 생성합니다. 또한, KARMA는 모델의 성능을 기준으로 서로의 관계성을 평가하여, 이를 바탕으로 선호 데이터를 체계적으로 생성합니다.

- **Performance Highlights**: KARMA의 효과는 경험적 평가에서 드러나며, 기존의 방법론인 RLAIF보다 월등히 우수한 결과를 보였습니다. 이 연구는 LLM 행동을 인간의 선호와 더 잘 일치시킬 수 있는 견고하고 확장 가능한 경로를 제시합니다. 결과적으로, KARMA는 더 높은 품질의 정렬 신호를 제공하게 되어 LLM의 행동을 보다 확실하게 조정할 수 있게 됩니다.



### Beyond RNNs: Benchmarking Attention-Based Image Captioning Models (https://arxiv.org/abs/2502.18734)
Comments:
          10 pages, 6 figures. Code and additional results are available on GitHub under the handle HemanthTejaY

- **What's New**: 이번 연구에서는 이미지 캡셔닝(image captioning) 분야에서 주목(attention) 메커니즘의 효과를 평가합니다. 전통적인 RNN 모델에 비해 주목 기반의 모델이 MS-COCO 데이터셋에서 더욱 정확하고 의미 있는 캡션을 생성하는 모습을 보였습니다. 연구 결과, 주목 메커니즘이 이미지와 생성된 캡션 간의 정렬을 향상시키는 데 기여함을 밝혀냈습니다.

- **Technical Details**: 이 연구는 두 종류의 딥러닝 모델을 비교하며, CNN(Convolutional Neural Network)은 이미지 피쳐를 추출하는 인코더 역할을 하며, RNN(Recurrent Neural Network)은 세부 정보를 텍스트로 변환하는 디코더 역할을 수행합니다. 주목 기법으로는 Bahdanau Attention을 사용하여 인코더와 디코더 간의 정렬 점수를 계산하고, 이 점수를 통해 각 단어가 어떤 이미지의 부분과 관련 있는지를 학습합니다.

- **Performance Highlights**: 실험 결과, 주목 기반 모델이 RNN 기반 모델보다 더 정확하고 의미 있는 캡션을 생성하며, 인간 평가와의 정렬에서도 더 나은 성과를 나타냈습니다. 평가 지표로는 BLEU, METEOR, GLEU 및 WER를 사용하여 모델 성능을 검증했습니다.



### Talking to the brain: Using Large Language Models as Proxies to Model Brain Semantic Representation (https://arxiv.org/abs/2502.18725)
Comments:
          20 pages, 6 figures

- **What's New**: 이 연구는 전통적인 심리학 실험에서 자연적인 자극을 사용할 때 발생하는 수작업 주석(annotation)과 생태적 타당성(ecological validity) 문제를 해결하기 위해, 다중 모달 대형 언어 모델(multimodal large language models, LLMs)을 활용한 새로운 패러다임을 소개합니다. 이를 통해 시각적 질문 답변(Visual Question Answering, VQA) 전략을 사용하여 자연적인 이미지에서 풍부한 의미 정보를 추출합니다.

- **Technical Details**: 연구에서는 LLM 기반 데이터 표현이 fMRI 기능적 자기공명영상(fMRI)으로 측정된 기존의 신경 활동 패턴(예: 얼굴, 건물)을 성공적으로 예측한다는 것을 강조합니다. 또한, 이 표현을 사용하여 뇌의 의미 네트워크를 구성하고, 이는 기능적 및 맥락적 연관성을 반영하는 의미 있는 군집을 식별하는데 기여합니다.

- **Performance Highlights**: 이 혁신적인 방법론은 전통적인 주석 방법의 한계를 극복하고, 자연적인 자극을 사용하여 뇌의 의미 조직(brain semantic organization)을 조사하는 강력한 솔루션을 제공합니다. 이는 인간 인지(human cognition)의 생태적으로 타당한 탐색을 위한 길을 열어줍니다.



### A Cooperative Multi-Agent Framework for Zero-Shot Named Entity Recognition (https://arxiv.org/abs/2502.18702)
Comments:
          Accepted at WWW 2025

- **What's New**: 이 논문에서는 제로샷(named entity recognition, NER) 인식을 위한 새로운 접근 방식인 협동 다중 에이전트 시스템(cooperative multi-agent system, CMAS)를 소개합니다. 기존의 방법들이 무시했던 개체 주변의 맥락적 상관관계를 포착하고, 제어 가능한 방식으로 작업 뎍모를 활용하도록 설계되었습니다. CMAS는 자기 주석 작성기(self-annotator), 유형 관련 특징 추출기(type-related feature extractor), 시연 판별기(demonstration discriminator), 전반적 예측기(overall predictor)라는 네 개의 주요 에이전트로 구성됩니다.

- **Technical Details**: CMAS에서는 NER 작업을 두 가지 하위 작업으로 재정의하여 개체 인식(named entities)과 문장 내 개체 유형 관련 특징(type-related features, TRFs)을 식별합니다. 이를 통해 LLM은 맥락적 상관관계를 추출하고, TRFs에 대해 자신이 선택한 시연의 유용성을 평가하는 자기 반성 메커니즘을 통합합니다. 이 방식은 세밀한 In-Context Learning (ICL)을 통해 데이터 간 복잡한 관계를 포착하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, CMAS는 여섯 가지 벤치마크에서 제로샷 NER 성능을 유의미하게 향상시켜, 특정 도메인과 일반 도메인 시나리오 모두에서 효율성을 입증하였습니다. 또한, CMAS는 몇 샷(few-shot) 설정에서도 효과성이 높아 다양한 LLM 백본(backbone)과의 호환성을 보여줍니다. 이는 NER 모델의 일반화 가능성과 적응성을 크게 높이는 결과를 가져옵니다.



### Speaking the Right Language: The Impact of Expertise Alignment in User-AI Interactions (https://arxiv.org/abs/2502.18685)
Comments:
          arXiv Version

- **What's New**: 이 연구는 25,000개의 Bing Copilot 대화를 분석하여 다양한 도메인 전문성을 가진 사용자의 응답 방식과 사용자 경험에 미치는 영향을 조사했습니다. 결과적으로 LLM(대형 언어 모델)은 대화의 77%에서 능숙하거나 전문가 수준으로 응답하며, 이는 사용자의 전문성 수준과 무관하게 긍정적인 사용자 경험과 연관이 있음을 보여줍니다. 또한, 전문가 수준의 응답이 아닌 경우 사용자 경험에 부정적인 영향을 미치며, 이는 복잡한 작업에 대해 더욱 두드러집니다.

- **Technical Details**: 연구는 사용자 전문성, 신뢰된 사용자 전문성, 에이전트 전문성의 세 가지 전문성 측정을 위해 5점 정렬척도를 사용하여 25,033개의 대화 데이터를 분류했습니다. 각 대화의 전문성 수준을 평가한 후, 사용자 만족도(SAT Score), 작업 복잡도(Task Complexity), 대화 길이(Conversation Length)와 같은 세 가지 메트릭을 통해 전문성 정렬의 영향을 측정했습니다. 이러한 메트릭은 LLM과 사용자 간의 미스알라인(misalignment)의 포괄적인 영향을 평가하는 데 사용되었습니다.

- **Performance Highlights**: 분석 결과, 전문성 정렬이 이루어진 경우 사용자 참여도는 높아지고, 대화에서 사용된 단어 수로 측정한 참여 수준이 증가하는 것으로 나타났습니다. 대다수의 사용자(63.9%)는 "초보자"로 분류되었으며, 전문적인 도메인에서 더 높은 전문성을 가진 사용자 수치도 나타났습니다. 에이전트는 전체적으로 "능숙" 또는 "전문가"로 분류되는 경우가 많은데, 이는 사용자의 요구와 AI 에이전트 간의 조화로운 상호작용이 효과적인 사용자 경험을 보장하는 데 필수적임을 강조합니다.



### Scaffolding Empathy: Training Counselors with Simulated Patients and Utterance-level Performance Visualizations (https://arxiv.org/abs/2502.18673)
Comments:
          This is a preprint version of the paper conditionally accepted to CHI'25

- **What's New**: 이 논문은 상담사 훈련의 효율성을 높이기 위해 시뮬레이션 환자와 상호작용하는 과정에서 빈번하고 상세한 피드백을 제공하는 LLM(large language model) 기반 훈련 시스템, SimPatient을 개발하고 평가한 내용을 담고 있습니다. 이 시스템은 상담자 학생들이 동기 부여 면담(motivational interviewing) 기술을 익히는 데 도움을 줍니다. 기존의 상담 훈련 방식이 주로 전통적인 교실 방법을 중심으로 이루어진 반면, 이 연구는 LLM을 활용하여 실시간 피드백을 자동화하는 데 초점을 맞추고 있습니다.

- **Technical Details**: SimPatient 시스템은 상담자가 시뮬레이션 환자와 상호작용하면서 훈련할 수 있도록 설계되었습니다. 이 시스템은 환자의 인지 모델을 기반으로 하여 네 가지 주요 인지 요소(조절 능력, 자기 효능감, 인식, 보상)를 파악합니다. 또한, 훈련 중 상담자의 성과를 평가하고 피드백을 제공하기 위해 LLMs를 사용하며, 대화의 각 턴(turn)에 맞는 시각화를 통해 피드백이 이루어집니다.

- **Performance Highlights**: 이 연구에서 프로페셔널 상담사와 학생 상담사에 대한 평가 연구를 실시한 결과, SimPatient 시스템의 사용성 및 사용자의 만족도가 높다는 것을 확인했습니다. 연구자는 상담 기술 훈련을 자동화하는 시스템의 설계 방향 및 다른 사회적 기술 훈련 유형으로의 확장 가능성에 대한 시사점을 제시합니다. 기존의 표준화된 환자 사용의 한계점을 해결하는 효율적인 대안으로 평가받았습니다.



### Faster, Cheaper, Better: Multi-Objective Hyperparameter Optimization for LLM and RAG Systems (https://arxiv.org/abs/2502.18635)
- **What's New**: 이 연구에서는 Retrieval Augmented Generation (RAG) 및 대형 언어 모델 (LLM) 시스템의 다중 목표 매개변수 최적화를 위한 새로운 접근법을 제시합니다. 특히 비용, 지연 시간, 안전성, 정렬을 포함하여 전체 시스템의 성능을 동시에 고려합니다. 베이지안 최적화 방법을 활용하여 이전 기준 방법보다 우수한 성과를 얻었으며, 새로운 RAG 벤치마크 작업에서 이 결과를 입증했습니다.

- **Technical Details**: RAG 시스템은 다양한 구성 요소의 많은 매개변수에 의존하는 복잡한 시스템입니다. 본 연구에서는 하이퍼볼륨 개선 (hypervolume improvement) 원칙을 적용하여 여러 하이퍼파라미터를 최적화하는 방법을 설명하며, noisy objective function에서 최적의 RAG 파이프라인 구성을 찾기 위해 qLogNEHVI를 사용합니다. 연구는 두 가지 새로운 RAG 벤치마크인 FinancialQA와 MedicalQA를 통해 검증되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법이 무작위 매개변수 선택 및 다른 기준 최적화 방법과 비교했을 때 두 가지 작업에서 더 높은 성능을 달성한 것으로 나타났습니다. 연구는 RAG 시스템을 설계하는 실무자들에게 중요한 고려 사항을 강조하며, 최적 구성은 과제와 목표에 따라 다르게 나타날 수 있음을 시사합니다. 이와 같은 통찰은 실무자들이 RAG 시스템을 개선하는 데 있어 유용할 것입니다.



### Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems (https://arxiv.org/abs/2502.18632)
- **What's New**: 본 연구에서는 문제에 매핑된 지식 구성 요소(Knowledge Components, KCs)를 자동으로 생성하고 태깅(tagging)하기 위한 LLM(대규모 언어 모델) 기반의 파이프라인을 제안합니다. 이 방법은 전통적으로 전문가에 의해 수행되던 작업을 최소화하여 개인화된 학습 경험을 개선하는 데 기여할 것으로 기대됩니다. KCGen-KT라는 이 파이프라인은 실제 학생 코드 제출 데이터를 활용하여 기존의 KT 방법보다 우수한 성과를 보여줍니다.

- **Technical Details**: 지식 구성 요소는 학생의 학습 상태를 세밀하게 추적하는 데 필수적입니다. 연구자들은 LLM을 활용하여 코드 작성 문제를 해결하기 위한 KCs를 생성하고, 이를 통해 문제에 대한 성능 예측을 위한 KT(Knowledge Tracing) 프레임워크를 개발했습니다. 이 과정에서 추상 구문 트리(Abstract Syntax Tree, AST)를 사용하여 코드 솔루션을 변환하고 KCs를 생성하는 방식으로 나아갔습니다.

- **Performance Highlights**: KCGen-KT는 CodeWorkout 데이터셋에서 평가를 수행하여 기존의 KT 방법보다 뛰어난 성능을 확인했습니다. 실험 결과, LLM이 생성한 KCs는 인간이 작성한 KCs와 유사한 적합성을 보였으며, 인간 전문가와 비교했을 때도 KC 태깅의 정확성이 상당히 높은 것으로 나타났습니다. 이 연구는 KCs 자동 생성 및 KT 분야에서 중요한 진전을 이루었다고 할 수 있습니다.



### PII-Bench: Evaluating Query-Aware Privacy Protection Systems (https://arxiv.org/abs/2502.18545)
- **What's New**: 대규모 언어 모델(LLMs)의 광범위한 채택은 사용자 프롬프트에 포함될 수 있는 개인 식별 정보(PII) 노출에 대한 심각한 프라이버시 우려를 불러일으켰습니다. 이를 해결하기 위해 제안된 쿼리 관련 없는 PII 마스킹 전략과 PII-Bench라는 포괄적인 평가 프레임워크는 사용자 프라이버시 보호 시스템의 유용성을 측정하는 데 중요한 기초 자료가 될 것입니다. 이는 다양한 PII 카테고리를 포함한 2842개의 테스트 샘플로 구성됩니다.

- **Technical Details**: PII-Bench는 사용자 쿼리, 맥락 설명, 쿼리와 관련된 PII를 표시하는 표준 응답을 포함하여 세 가지 주요 구성 요소로 세밀하게 설계된 샘플로 이루어져 있습니다. 이 프레임워크는 쿼리 관련성을 결정하는 데 있어 모델들이 직면하는 주목할 만한 한계를 드러내며, 특히 다중 주제 시나리오에서의 PII 식별 문제에 대한 연구를 심층적으로 다룹니다. 쿼리 관련 없는 PII 마스킹 전략을 통해 PII의 중요성과 사용자 쿼리 간의 관계를 고려하여 보다 세밀한 보호 조치를 가능하게 합니다.

- **Performance Highlights**: 실험 분석 결과, 현재 모델들은 기본적인 PII 탐지에서는 양호한 성능을 보이나, 쿼리 관련성 결정에서는 뚜렷한 한계를 드러냈습니다. 최신 LLM조차 이러한 작업에서 어려움을 겪고 있으며, 이는 지능적인 PII 마스킹이 필요함을 강조합니다. 본 연구는 프라이버시 보호 시스템의 효과성을 평가하기 위한 새로운 기준을 제시하며, 향후 연구 방향에 대한 인사이트를 제공합니다.



### FilterRAG: Zero-Shot Informed Retrieval-Augmented Generation to Mitigate Hallucinations in VQA (https://arxiv.org/abs/2502.18536)
Comments:
          12 pages, 6 figures and 2 tables

- **What's New**: FilterRAG이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 BLIP-VQA와 Retrieval-Augmented Generation(RAG)을 통합하여 외부 지식 소스, 예를 들어 Wikipedia와 DBpedia에서 답변의 기반을 확보합니다. 이를 통해 모델이 생성하는 오류 답변, 즉 hallucinations를 줄이고, 현실 세계에 적합한 VQA 시스템을 개선할 수 있는 잠재력을 강조합니다.

- **Technical Details**: FilterRAG는 입력 이미지를 2x2 그리드로 나누고, BLIP-VQA를 통해 시각적 및 텍스트 임베딩을 생성합니다. 이후 관련 지식을 동적으로 검색하여 답변 생성 과정에 통합하며, 이 과정에서 고정된 GPT-Neo 1.3B 모델을 사용합니다. 이러한 방식은 VQA 성능을 향상시키고, OOD(Out-of-Distribution) 시나리오에서 더욱 신뢰할 수 있는 결과를 제공합니다.

- **Performance Highlights**: OK-VQA 데이터셋에서 FilterRAG는 36.5%의 정확도를 달성하였으며, 이는 기존 모델 대비 hallucinations를 현저히 줄였습니다. 필터RAG는 인도메인 및 OOD 설정 모두에서 일관된 성능을 보였으며, 이는 효과적인 지식 검색과 다중 모달 정렬이 VQA의 견고함을 향상시키는 데 중요하다는 것을 보여줍니다.



### Enhancing Hepatopathy Clinical Trial Efficiency: A Secure, Large Language Model-Powered Pre-Screening Pipelin (https://arxiv.org/abs/2502.18531)
Comments:
          30 pages, 5 figures

- **What's New**: 이번 연구는 복잡한 간 질환(hepatocellular carcinoma 및 간경변증) 관련 코호트 모집을 위한 새로운 환자 전수 screening 시스템을 개발하였습니다. 이 시스템은 대규모 언어 모델을 활용하여 기존의 시간 소모적인 수작업 screening 방식의 문제를 해결하고자 합니다. 특히, AI를 이용한 전수 screening 으로 정확성, 효율성 및 데이터 프라이버시 관련 문제를 다루고 있습니다.

- **Technical Details**: 개발된 파이프라인은 복잡한 기준을 일련의 복합 질문으로 나누고, 전자건강기록(electronic health records)를 통해 두 가지 전략으로 의미적 질문 응답(question-answering)을 수행합니다. 첫 번째 경로(Pathway A)는 인류화된 전문가의 사고 체계(Chain of Thought) 전략을 사용하고, 두 번째 경로(Pathway B)는 에이전트 협업(Agent Collaboration) 내에서의 사전 설정된 입장(Preset Stances)을 적용하여 복잡한 임상 추론 시나리오를 관리합니다. 이 파이프라인은 정밀도(precision), 시간 소모(time consumption), 그리고 반사실적 추론(counterfactual inference) 세 가지 주요 지표로 평가되었습니다.

- **Performance Highlights**: 이 연구의 결과, 파이프라인은 높은 정밀도(0.921)와 효율성(작업당 0.44초)을 달성했습니다. Pathway B는 복잡한 추론에서 뛰어난 성능을 보였고, Pathway A는 신속한 데이터 추출에 효과적이었습니다. 두 경로 모두 유사한 정밀도를 기록하며, 간세포암(0.878) 및 간경변 시험(0.843)에서도 유망한 결과를 보여주었습니다.



### Analyzing User Perceptions of Large Language Models (LLMs) on Reddit: Sentiment and Topic Modeling of ChatGPT and DeepSeek Discussions (https://arxiv.org/abs/2502.18513)
Comments:
          13 pages, 8 figures

- **What's New**: 이 연구는 ChatGPT와 DeepSeek와 같은 대형 언어 모델(LLM)에 대한 사용자의 인식을 이해하기 위해 Reddit에서의 토론을 분석하고 있습니다. 기존의 연구들이 부족했던 점을 보완하여, 사용자 태도와 AI 개발, 신뢰 및 미래 정책에 미치는 영향을 다룹니다. 이 논문은 AI에 대한 신뢰, 사용자 기대, 도구의 잠재적 사용, AI 편향에 대한 우려와 윤리적 의미 등의 중요한 주제를 탐구합니다.

- **Technical Details**: 이 연구에서는 감정 분석(sentiment analysis)과 주제 모델링(topic modeling)을 활용하여 Reddit에서의 논의를 분석합니다. 단어 빈도(word frequency) 접근법을 사용하여 광범위한 주제와 감정 경향을 파악하였고, 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA) 방법을 통해 사용자 언어에서의 주요 주제를 식별했습니다. 예를 들어, LLM의 잠재적 이점, 기술적 응용, 그리고 사회적 결과에 대한 논의가 포함됩니다.

- **Performance Highlights**: 이 연구는 공공의 감정이 AI 개발의 방향을 어떻게 형성할 수 있는지를 보여줍니다. 사용자가 이 기술에 신뢰를 가지는지 여부와 그들이 바라보는 기술의 미래에 대한 통찰을 제공합니다. 연구 결과는 개발자와 정책 입안자들에게 사용자 경험과 이해도를 전달함으로써 이 혁신적인 기술에 대한 정책적 접근을 돕는 데 기여할 것입니다.



### Comprehensive Analysis of Transparency and Accessibility of ChatGPT, DeepSeek, And other SoTA Large Language Models (https://arxiv.org/abs/2502.18505)
- **What's New**: 이 연구는 오픈소스 인공지능(AI) 모델의 투명성과 접근성 문제에 초점을 맞추고 있습니다. 최신 대형 언어 모델(LLM)의 일부는 '오픈소스'라고 주장하지만, 실제로 완전한 투명성이 부족하다는 점을 밝혀냅니다. 이 연구는 약 100개의 LLM을 분석하여 오픈소스와 오픈웨이트 모델 간의 차이를 구체적으로 검토하고, AI의 윤리적 배치를 위한 더 나은 기준과 가이드라인을 제안합니다.

- **Technical Details**: 연구 방법론은 다단계 접근방식을 사용하여 LLM의 개발 및 배포에서의 개방성과 투명성을 평가합니다. 오픈소스 LLM은 코드베이스, 모델 아키텍처, 훈련 데이터 등을 자유롭게 제공하는 반면, 오픈웨이트 LLM은 학습된 모델 가중치만 공개하는 경우가 많습니다. 이러한 연구 설계는 기존의 오픈소스 기준 및 투명성 정의에 기반하여 AI 문헌에 대한 광범위한 검토를 포함합니다.

- **Performance Highlights**: 연구는 DeepSeek, ChatGPT 등 여러 모델의 증가하는 관심을 강조합니다. 그러나 몇몇 모델이 오픈소스라 하더라도 훈련 데이터나 코드와 같은 필수 정보는 제공하지 않는다는 점을 지적합니다. 이 연구는 또한 인공지능의 신뢰성을 높이기 위해 완전한 투명성이 필요하다는 것을 강조하며, AI의 윤리적 발전을 위한 보다 명확한 기준이 필요함을 역설합니다.



### TurboFuzzLLM: Turbocharging Mutation-based Fuzzing for Effectively Jailbreaking Large Language Models in Practic (https://arxiv.org/abs/2502.18504)
Comments:
          Accepted at NAACL 2025 industry track, 12 pages, 5 figures

- **What's New**: 이 논문에서는 TurboFuzzLLM이라는 새로운 변이 기반의 퍼징 기법을 제안하고 있습니다. 이 기술은 불법으로 사용자 프로프트를 통해 유해한 응답을 이끌어낼 수 있는 효과적인 jailbreaking 템플릿을 찾는 데 초점을 맞춥니다. TurboFuzzLLM은 기존 템플릿 기반 공격 기법의 한계를 극복하고, 자동으로 효과적인 jailbreaking 템플릿을 생성할 수 있는 기능을 추가하여 공격 성공률을 95% 이상 달성했습니다. 이는 사용자 프로프트를 통해 블랙박스 접근으로 타겟 LLM을 공격하는 데 필요한 효과적인 방법을 제공합니다.

- **Technical Details**: TurboFuzzLLM은 기존의 문서에서 제안된 GPTFuzzer를 기반으로 하지만, 새로운 선택 정책과 효율성을 강조한 휴리스틱을 추가하여 개선된 기능을 제공합니다. 변이 라이브러리를 확장하고, 공격할 모델에 대해 변형된 템플릿을 생성하기 위해 반복적으로 퍼징을 수행하는 프로세스를 포함합니다. 각 퍼징 반복에서 새로운 템플릿과 악성 질문을 결합하여 공격 프롬프트를 생성하고, 이를 통해 얻은 응답을 평가하여 효과성을 검증합니다.

- **Performance Highlights**: TurboFuzzLLM은 다양한 목표 LLM에 대해 일관되게 뛰어난 공격 성공률을 기록하였으며, GPTFuzzer 및 기타 최신 기법들과 비교하였을 때 더욱 높은 성과를 나타냈습니다. 이 시스템은 새로운 유해 질문에 대해서도 잘 일반화된 템플릿을 학습할 수 있으며, 생성된 레드 팀(Red teaming) 데이터는 모델의 내장 방어력을 개선하는 데에도 활용될 수 있습니다. 실험결과는 TurboFuzzLLM의 각 개별 업그레이드의 기여도를 보여주는 절차적 연구를 포함하고 있습니다.



### Mechanistic Understanding of Language Models in Syntactic Code Completion (https://arxiv.org/abs/2502.18499)
Comments:
          10 pages, 4 figures, accepted to the AAAI 2025 Workshop on Towards Knowledgeable Foundation Models

- **What's New**: 최근 언어 모델(Models, LMs)이 코드 생성 작업에서 뛰어난 성능을 보이고 있지만, 코드 전용 데이터셋으로 미세 조정된 결과인 코드 LMs의 내부 의사결정 과정에 대한 이해는 여전히 부족합니다. 이러한 부족한 이해는 실생활에서 사용될 때 의도하지 않은 피해를 초래할 수 있습니다. 이 연구는 CodeLlama-7b 모델을 사용하여 닫는 괄호 작업을 수행하는 코드 LMs의 메커니즘을 탐구한 최초의 작업 중 하나입니다.

- **Technical Details**: 이 연구에서는 Synthetic Dataset을 생성하여 코드 LMs의 구문 완성 성능을 체계적으로 분석합니다. 데이터셋은 2, 3, 4개의 닫는 괄호를 필요로 하는 168개의 프롬프트를 포함하며, 구조적으로 복잡한 호출을 포함합니다. CodeLlama-7b 모델이 중간 및 후반 레이어에서만 올바른 토큰을 예측할 수 있다는 점을 발견하였고, 다수의 주목(attention) 헤드가 이미 닫힌 괄호의 수를 추적하는 데 중요한 역할을 했습니다.

- **Performance Highlights**: CodeLlama-7b 모델은 닫는 괄호가 필요할 때 중간 이후 레이어에서만 올바른 타겟 토큰을 인식합니다. 올바른 토큰 예측과 반대 토큰 예측을 비교할 때, multi-head attention (MHA) 서브 레이어가 가장 중요한 기여를 합니다. 마지막으로, 특정 주목 헤드는 이미 닫힌 괄호의 수를 정확하게 추적하지만, 몇몇은 잘못된 지식 연관성을 보여줘 성능에 부정적인 영향을 미쳤습니다.



### AuPair: Golden Example Pairs for Code Repair (https://arxiv.org/abs/2502.18487)
- **What's New**: 이번 논문에서는 대규모 언어 모델(Large Language Models, LLMs)의 인퍼런스(inference) 성능을 향상시키기 위해, 자기 수리(self-repair) 능력을 활용하는 방법을 제안합니다. LLM이 초기 오류 응답을 수정하여 더 나은 응답을 생성할 수 있도록, 개별 문제에 대한 '금동 쌍(AuPairs)' 예시를 순서대로 생성하고 선택하는 접근법을 개발했습니다. 이 접근법은 다양한 수정 솔루션을 생성해 주며, 최고 점수의 솔루션이 최종 답변으로 선택됩니다.

- **Technical Details**: 제안된 알고리즘은 N개의 LLM 호출에 따라 최대 N개의 AuPair를 생성합니다. 각 AuPair는 초기 추정 및 해당 프로그래밍 문제에 대한 수정으로 구성되어 있습니다. 이러한 AuPair는 앙상블(ensemble) 방식으로 선택되어 상호 보완적이며 유용한 조합을 형성합니다. 이러한 방식으로 수집된 AuPair는 인퍼런스 시점에 하나의 예로 제공되어 최적의 솔루션을 생성하도록 돕습니다.

- **Performance Highlights**: 제안된 알고리즘은 5개의 다양한 LLM 모델에서 코드 수리 작업에 대해 뛰어난 성능을 보였습니다. AuPair는 기존의 best-of-N 및 자기 수리 접근법보다 훨씬 뛰어난 성능을 보여주며, 다양한 데이터셋과 모델 크기에서도 강력한 일반화 능력을 발휘합니다. 이 알고리즘은 상대적으로 적은 인퍼런스 시간 예산에도 더욱 효과적인 성능 스케일링을 보여, 실질적인 효용을 증대시킵니다.



### QExplorer: Large Language Model Based Query Extraction for Toxic Content Exploration (https://arxiv.org/abs/2502.18480)
- **What's New**: 이번 연구에서는 정보 검색에서 유효한 쿼리 자동 추출의 도전 과제를 다루며, 특히 독성 콘텐츠 탐색과 관련된 내용입니다. 새로운 접근 방식인 QExplorer를 제안하여, 지능형 대형 언어 모델(LLM)의 기능을 활용하여 유사한 내용의 탐색을 위한 효율적인 쿼리를 직접 추출할 수 있게 되었습니다. 2단계 훈련 과정인 Supervised Fine-Tuning(SFT)과 Direct Preference Optimization(DPO)을 포함하여, 검색 시스템의 피드백을 활용한 데이터 세트 구축이 중요한 요소로 작용합니다.

- **Technical Details**: QExplorer 접근 방식은 두 가지 단계의 훈련 과정을 포함하며, 이는 지침 기반의 SFT와 사용자의 선호도를 정렬하는 DPO를 포함합니다. 이 모델은 오프라인 및 온라인 실험을 통해 효과성을 검증하며, 그 과정에서 LLM 기반의 자동 쿼리 추출 방법이 기존 LLM 및 인간 수행보다 우수한 성능을 보인다고 보고합니다. 이는 키워드 추출을 위한 인간의 직관과 LLM의 성능을 결합하여 더 나은 발견 결과를 도출하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과, QExplorer는 기존 베이스라인 모델과 인간의 쿼리 추출 성능을 초월하는 것으로 나타났습니다. 온라인 배치에서 독성 항목의 탐지가 크게 증가했으며, 이는 QExplorer의 효과적인 쿼리 구성 덕분입니다. 또한 대화형 LLM 기반의 자동 쿼리 추출이 독성 콘텐츠 탐색의 효율성을 크게 향상시킬 수 있음을 입증했습니다.



### FinBloom: Knowledge Grounding Large Language Model with Real-time Financial Data (https://arxiv.org/abs/2502.18471)
Comments:
          27 pages, 9 tables

- **What's New**: 이 논문에서는 금융 쿼리를 처리하기 위해 LLM에 실시간 데이터 모듈과 함께 동작하는 Financial Agent라는 지식 기반 접근 방식을 소개합니다. 연구팀은 50,000개 이상의 금융 쿼리와 해당 문맥을 포함한 Financial Context Dataset을 개발했으며, 금융 뉴스 기사와 SEC 파일들로 훈련된 70억 매개변수의 FinBloom 7B LLM을 제안합니다.

- **Technical Details**: 논문에서는 다양한 도메인 특정 태스크를 효과적으로 수행할 수 있도록 LLM을 동결한 상태에서 모듈을 추가하는 방법을 사용합니다. 이 시스템은 실시간 데이터를 처리하기 위해 두 개의 데이터 레포지토리(탭 형식의 가격 데이터와 텍스트 기반의 뉴스 데이터)를 유지하며, 사용자의 쿼리를 분석하여 관련 데이터를 추출하고 이를 텍스트 형식으로 변환합니다.

- **Performance Highlights**: 제안된 방법론을 통해 금융 쿼리에 대한 응답을 생성하는 데 필요한 동적 문맥 정보를 신속하게 제공할 수 있습니다. 사용자의 쿼리에 최신 데이터를 결합시킴으로써 LLM이 보다 정확한 응답을 생성할 수 있도록 지원하여, 실제 금융 결정을 내릴 때 유용성을 크게 향상시킬 것으로 기대됩니다.



New uploads on arXiv(cs.IR)

### Agent-centric Information Access (https://arxiv.org/abs/2502.19298)
- **What's New**: 대형 언어 모델(LLM)이 점점 더 전문화됨에 따라, 각기 다른 도메인에서 우수성을 발휘하는 다수의 LLM이 존재하는 미래를 구상하고 있습니다. 본 논문은 이러한 시스템을 위한 에이전트 중심 정보 접근 프레임워크를 소개하며, 이는 LLM이 동적으로 평가되고 질의되는 지식 에이전트로 기능하도록 합니다. 전통적인 문서 검색 방식과는 달리, 이 접근법은 전문가의 전문성을 실시간으로 추론해야 합니다.

- **Technical Details**: 저자들은 사용자 질의를 처리하기 위해 LLM의 선택, 질의 효율화, 그리고 여러 모델에서의 응답 융합 과정에서 해결해야 할 여러 도전에 대해 설명합니다. 제안하는 프레임워크는 정보 검색(IR)을 사용자와 전문가 모델의 분산 네트워크 간의 동적 상호작용으로 구조화하며, 사용자 에이전트가 이전 질의 및 피드백 기반으로 검색 전략을 조정합니다. 지식 에이전트는 특정 도메인에 전문화되어 있고, 각 모델은 동적으로 관련성을 평가하여 최적의 응답을 제공하기 위해 선택적으로 질의됩니다.

- **Performance Highlights**: 이 프레임워크는 대규모 전문가 모델을 전개할 수 있는 잠재력을 지니고 있으며, 사용자 맞춤형 디지털 조수 LLM이 개인의 검색 행동을 통해 사용자의 지식과 전문성을 반영하는 방향으로 나아갈 것을 제안합니다. 사용자 에이전트는 검색 과정에서 각 사용자별 요인을 반영하여 정보 합성을 최적화하고, 비용과 지연 시간을 최소화하는 적응형 질의 메커니즘을 포함합니다. 이러한 접근법은 궁극적으로 여러 전문가 LLM의 응답을 통합하고 신뢰성을 보장하는 데 기여할 것입니다.



### Multiview graph dual-attention deep learning and contrastive learning for multi-criteria recommender systems (https://arxiv.org/abs/2502.19271)
- **What's New**: 본 연구에서는 다중 기준 추천 시스템(Multi-Criteria Recommender Systems, MCRS)을 위한 새로운 표현 방식을 제안합니다. 제안하는 방법은 다중 에지 이분 그래프(multi-edge bipartite graph)에 기반하며, 각 엣지는 사용자가 평가한 아이템의 기준 점수를 나타냅니다. 또한, Multiview Dual Graph Attention Networks (MDGAT)를 통해 사용자와 아이템 간의 복합적인 관계를 고려합니다. 이 접근 방식은 아이템의 다각적인 특성을 반영하는 데 도움을 줍니다.

- **Technical Details**: 연구에서는 각 뷰(view)에 대한 유사성을 기반으로 앵커 포인트(anchor point)를 정의하고, 로컬(local) 및 글로벌(global) 대조 학습을 적용하여 긍정 샘플과 부정 샘플을 구별합니다. 이 과정에서 그래프 주의 네트워크(Graph Attention Networks, GAT) 및 다중 뷰 그래프 주의 네트워크(MGAT)의 두 가지 주의 메커니즘을 활용하여 데이터의 복잡한 관계를 효과적으로 모델링합니다. MDGAT는 각 뷰의 특징을 중요하게 고려하여 정보 전파를 향상시키며, 사용자 및 아이템 간의 관계를 명확히 하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법은 실제 데이터셋에서 아이템 점수 예측의 정확성을 향상시키는 것으로 나타났습니다. 이렇게 개선된 성능은 기존 기준과 비교했을 때 우수한 효율성을 보여 주며, 로컬 및 글로벌 이웃 간의 관계를 효과적으로 포착하여 추천 시스템의 전반적인 성능을 높입니다. 이 연구는 MCRS 분야에서 향후 연구에 중요한 기초 자료를 제공할 것입니다.



### UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering (https://arxiv.org/abs/2502.19178)
Comments:
          10 pages, 3 figures, 7 tables

- **What's New**: 본 논문에서는 사용자 상호작용 내역을 LLMs의 맥락에 통합하여 개인화를 개선하는 새로운 벤치마크인 UQABench를 제안합니다. 이 벤치마크는 사용자 임베딩의 효과성을 평가하기 위해 제정된 표준화된 평가 과정을 포함합니다. 특히, 개인화된 Q&A 파라다임의 필요성을 강조하며, LLM 시대에 맞는 평가 과제를 설계했습니다.

- **Technical Details**: 사용자 임베딩을 통해 LLMs를 유도하기 위해, 우리는 세 가지 주요 작업인 시퀀스 이해, 행동 예측, 관심 인식 과제를 제안합니다. 이러한 과제는 전통적인 추천 시스템과 LLM 기반 방법의 요구를 포함하며, 사용자 임베딩이 사용자 관심사와 선호도를 반영할 수 있도록 설계되었습니다. UQABench는 프리트레이닝, 파인튜닝, 평가의 세 단계를 통해 사용자 임베딩의 품질을 평가합니다.

- **Performance Highlights**: UQABench는 다양한 최신 사용자 인코더 모델을 평가하여 사용자 임베딩의 활용 가능성에 대한 귀중한 통찰력을 제공합니다. 특히, Transformer 기반 인코더의 확장성에 대해서도 논의하여, LLM 개인화에 대한 새로운 길을 열고 있습니다. 결과적으로, 본 연구는 사용자 개인화의 혁신을 위한 기초를 제공하며, 공개된 데이터와 코드로 추가 연구를 촉진할 수 있게 합니다.



### A 106K Multi-Topic Multilingual Conversational User Dataset with Emoticons (https://arxiv.org/abs/2502.19108)
- **What's New**: 최근 인스턴트 메신저(IM)가 커뮤니케이션의 주요 수단으로 자리잡으면서, 감정을 표현할 수 있는 이모티콘의 중요성이 증가하고 있습니다. 특히, 새로운 이모티콘 데이터셋을 소개하며, 이 데이터셋은 시간 기반 데이터와 익명 사용자 식별자를 포함하여 개인화된 사용자 모델링에 기여할 수 있는 잠재력을 지니고 있습니다. 이 데이터셋은 22K 개의 사용자, 370K 개의 이모티콘 및 8.3M 개의 메시지를 포함하는 대규모 공개 데이터셋으로, 개인화된 이모티콘 추천 시스템의 연구에 큰 기여를 할 것으로 기대됩니다.

- **Technical Details**: 이 새로운 데이터셋은 10개의 다양한 도메인을 아우르며, 사용자 상호작용과 이모티콘 사용 패턴을 분석하는 데 필수적인 정보를 제공합니다. 이전 데이터셋들은 사용자 정보를 결여하고 있어 특정 사용자를 식별하기 어려웠으나, 본 데이터셋에서는 16.90 개의 이모티콘 히스토리를 포함하여, 다양한 언어를 지원함으로써 연구의 범위를 확장할 수 있습니다. 구체적으로, 사용자 정보와 시간적 상호작용 데이터를 결합하여 새로운 개인화 알고리즘 개발에 기여할 수 있을 것입니다.

- **Performance Highlights**: 본 연구에서 수행된 정량적 및 정성적 실험 결과는 이 데이터셋이 사용자 행동 분석과 개인화된 추천 시스템에 유용하다는 것을 입증합니다. 이모티콘을 통한 감정 표현과 상호작용의 다면성을 연구함으로써, 향후 개인화된 검색 및 대화형 AI 연구에 대한 새로운 가능성을 열어줄 것입니다. 또한, 데이터셋은 freely accessible하여, 더 많은 연구자들이 쉽게 활용할 수 있습니다.



### OntologyRAG: Better and Faster Biomedical Code Mapping with Retrieval-Augmented Generation (RAG) Leveraging Ontology Knowledge Graphs and Large Language Models (https://arxiv.org/abs/2502.18992)
Comments:
          This paper has been accepted as a workshop paper for KEIR@ECIR 2025

- **What's New**: 이번 논문에서는 OntologyRAG라는 새로운 방법론을 소개합니다. 이 방법은 생물 의학 온톨로지(domain-specific ontology)의 개념과 관계를 효과적으로 활용하여 코드 매핑(code mapping)을 자동화합니다. 특히, 기존의 언어 모델(LM) 재훈련 없이도 온톨로지 업데이트를 반영할 수 있는 점이 특징입니다.

- **Technical Details**: OntologyRAG는 온톨로지 지식 그래프(ontology knowledge graph)를 활용하여 대규모 언어 모델(LLM)에서 컨텍스트 내 학습(in-context-learning)을 수행하도록 설계되었습니다. 이 시스템은 언어 모델이 제공하는 비정제 매핑을 기반으로 질문을 처리하고, 예측의 근거(prediction rational)와 매핑의 근접성(mapping proximity)을 포함하는 해석 가능한 결과를 생성합니다.

- **Performance Highlights**: 자체 큐레이션된 금 데이터셋(gold dataset)에서 평가한 결과, OntologyRAG를 사용하면 코딩 전문가들이 더 나은 품질과 빠른 속도로 코드 매핑을 수행할 수 있음을 보여주었습니다. 이는 기존의 수동 검증 과정에서 발생하는 시간과 노동 집약적인 과정을 크게 줄일 수 있는 가능성을 지닙니다.



### OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignmen (https://arxiv.org/abs/2502.18965)
- **What's New**: 최근 생성 기반 추천 시스템(generative retrieval-based recommendation systems)이 유망한 패러다임으로 주목받고 있습니다. 기존의 추천 시스템은 주로 retrieve-and-rank 전략을 채택하여, 생성 모델이 검색 단계에서 선택자로만 기능합니다. 본 논문은 OneRec을 제안하며, 이는 기존의 누적 학습 프레임워크(cascaded learning framework)를 통합 생성 모델(unified generative model)로 대체합니다. OneRec은 현재의 복잡한 추천 시스템을 큰 폭으로 초월하는 최전선(end-to-end) 생성 모델입니다.

- **Technical Details**: OneRec 구조는 인코더-디코더 구조로, 사용자의 행동 이력을 인코딩하고 사용자가 관심 가질 비디오를 점진적으로 디코딩합니다. 우리는 희소 혼합 전문가(sparse Mixture-of-Experts, MoE)를 채택하여 모델 용량을 확장하면서 연산량(FLOPs)을 비례적으로 증가시키지 않습니다. 또한, 전통적인 다음 아이템 예측(next-item prediction) 방식과는 다르게 세션 기반 생성(session-wise generation) 접근 방식을 제안합니다. 세션 기반 생성은 수동으로 규칙을 결합하는 방식보다 더 우아하고 맥락적으로 일관된 결과를 생성하도록 돕습니다.

- **Performance Highlights**: 대규모 산업 데이터셋을 대상으로 한 실험 결과, OneRec이 제안하는 방법의 우수성을 입증하였습니다. 특히, Kuaishou의 주요 장면에 OneRec을 배포한 결과, 시청 시간이 1.6% 증가하여 획기적인 개선을 이루었습니다. 결론적으로, 우리의 연구는 생성 모델의 통합 사용이 추천 시스템의 성능을 극대화할 수 있음을 강조하며, 다양한 사용자 선호를 효과적으로 일반화할 수 있는 가능성을 제시합니다.



### A Multifacet Hierarchical Sentiment-Topic Model with Application to Multi-Brand Online Review Analysis (https://arxiv.org/abs/2502.18927)
Comments:
          21 pages, 6 figures, 4 tables

- **What's New**: 이 논문에서는 고객 리뷰에서 여러 브랜드의 감정 편향성을 탐지하기 위해 다면적 (multifacet) 계층적 감정-주제 모델(MH-STM)을 제안합니다. 이는 기존의 전통적인 토픽 모델과는 달리, 브랜드 관련 주제를 계층 구조로 형성하여 각 브랜드의 고유한 특성을 더 효과적으로 분리합니다. 새로운 계층적 Polya urn(HPU) 스킴을 통해 주제와 단어 간의 연관성을 더욱 향상시킵니다.

- **Technical Details**: MH-STM은 온라인 고객 리뷰에서 브랜드마다 서로 다른 주제를 탐지하고 이를 기반으로 감정 점수를 획득하는 통합 생성 프레임워크를 사용합니다. 이 모델은 비구조적 데이터에서 주요 세부 사항과 일반적인 배경을 구분할 수 있는 계층 구조를 채택합니다. 또한, 계층적 토픽 모델(hLDA)을 사용하여 주제 계층을 정의하며, HPU 스킴을 통해 각 단어의 일반적인 특성에 따라 적절하게 주제 계층에 배정됩니다.

- **Performance Highlights**: 제안된 방법의 성능은 합성 데이터와 두 개의 실제 리뷰 데이터 집합에서 평가되었습니다. 실험 결과는 다면적 주제 계층을 감지하고, 정확한 브랜드 순위를 도출하는 데 효과적임을 보여주었습니다. 이 방법은 결과적으로 소비자가 여러 측면에서 브랜드를 비교할 수 있는 유용한 도구를 제공합니다.



### Hierarchical corpus encoder: Fusing generative retrieval and dense indices (https://arxiv.org/abs/2502.18877)
- **What's New**: 본 논문은 새로운 접근 방식인 hierarchical corpus encoder (HCE)를 제안합니다. HCE는 문서 계층의 형제를 대조하여 훈련하며, 이는 기존의 generative retrieval 모델의 제약을 극복하는 방식입니다. HCE를 통해 새로운 문서의 추가 및 삭제가 쉬워지며, 제로샷(zero-shot) 적응성이 크게 개선됩니다.

- **Technical Details**: HCE는 dense retrieval 및 generative retrieval의 장점을 결합한 모델로, 문서 계층 구조에서 긍정 샘플과 부정 샘플을 대조합니다. 훈련 시, HCE는 문서 계층 내에서 형제 노드 간의 대비 훈련을 통해 손실을 입힙니다. 테스트 시, 이 모델은 외부 인덱스를 사용한 최대 내적 검색(MIPS)으로 복귀합니다.

- **Performance Highlights**: 실험 결과, HCE는 다양한 밀집 및 생성 검색 방법들에 비해 뛰어난 성능을 보여주었습니다. 지도학습(supervised) 및 비지도학습(unsupervised) 환경에서 모두 우수한 성과를 달성하며, 문서 세트를 계층으로 모델링하는 HCE의 효과성을 강조하고 있습니다.



### Training Large Recommendation Models via Graph-Language Token Alignmen (https://arxiv.org/abs/2502.18757)
Comments:
          5 pages. Accepted by www'25 as short paper

- **What's New**: 이번 논문에서는 대규모 추천 시스템을 위한 새로운 프레임워크인 GLTA( Graph-Language Token Alignment)를 제안합니다. GLTA는 사용자 및 아이템 노드와 사전학습된 대형 언어 모델(LLM) 토큰을 정렬하여 추천의 정확성을 높이며, 추천 결과에서 텍스트의 모호성을 없앱니다. 또한, GLLM(Graph-Language Logits Matching) 레이어를 도입하여 전체 과정의 일관성을 보장합니다.

- **Technical Details**: GLTA는 그래픽 사용자 인터페이스에서 사용자 및 아이템 간의 상호작용을 모델링하는 그래프 기반 추천 시스템의 구조를 통합합니다. 이 구조는 LightGCN을 기반으로 하여 사용자 노드 임베딩과 아이템 노드 임베딩을 미리 훈련하여 그래프 내 정보 구조를 캡처합니다. 사용자의 추천 결과는 예측된 아이템 로짓을 실제 아이템과 일치시키는 GLLM 레이어를 통해 최적화되어, 아이템 예측의 신뢰성을 높입니다.

- **Performance Highlights**: GLTA의 성능을 검증하기 위해 세 개의 검증 데이터 세트에서 광범위한 실험이 진행되었습니다. 각 구성 요소의 효과는 ablation 연구를 통해 검증되었으며, GLTA는 기존의 추천 시스템보다 더 높은 추천 정확성을 보여줍니다. 이는 LLM의 강력한 추론 능력을 효과적으로 활용하며, 개별 아이템에 대한 명확한 매핑을 가능하게 합니다.



### AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms (https://arxiv.org/abs/2502.18754)
Comments:
          8 pages, 10 figures, in Proceedings of the ACM Web Conference 2025 (WWW '25)

- **What's New**: AgentSociety Challenge는 웹 컨퍼런스에서 처음으로 개최되는 대회로, 대형 언어 모델(LLM) 에이전트를 활용하여 사용자 행동을 모델링하고 추천 시스템을 개선하는 데 중점을 두고 있습니다. 참가자들은 Yelp, Amazon과 Goodreads의 통합 데이터 세트를 활용하여 혁신적인 LLM 에이전트를 개발하는 과제가 주어집니다. 총 295개 팀이 참여하여 1,400건 이상의 제출물이 있었으며, 각 트랙에서 성과 개선을 달성했습니다.

- **Technical Details**: 이 대회는 두 가지 트랙으로 나뉘어 있습니다: 사용자 모델링 트랙(User Modeling Track)과 추천 트랙(Recommendation Track)입니다. 사용자 모델링 트랙에서는 특정 항목에 대한 사용자 리뷰와 별점 생성을 시뮬레이션하는 에이전트를 디자인해야 합니다. 추천 트랙에서는 역사적 상호작용과 정보를 기반으로 사용자 맞춤형 추천을 제공하는 LLM 에이전트를 개발합니다.

- **Performance Highlights**: 참가자들은 개발 단계에서 사용자 모델링 트랙 21.9%, 추천 트랙 20.3%의 성능 개선을 달성하였고, 최종 단계에서는 각각 9.1%, 15.9%의 개선을 기록했습니다. 대회는 정보 검색, 추천 시스템, 사용자 모델링 등의 주제로 최신 LLM 에이전트를 활용하여 복잡한 인간 행동을 예측하고 시뮬레이션할 수 있는 기회를 제공합니다.



### A Cooperative Multi-Agent Framework for Zero-Shot Named Entity Recognition (https://arxiv.org/abs/2502.18702)
Comments:
          Accepted at WWW 2025

- **What's New**: 이 논문에서는 제로샷(named entity recognition, NER) 인식을 위한 새로운 접근 방식인 협동 다중 에이전트 시스템(cooperative multi-agent system, CMAS)를 소개합니다. 기존의 방법들이 무시했던 개체 주변의 맥락적 상관관계를 포착하고, 제어 가능한 방식으로 작업 뎍모를 활용하도록 설계되었습니다. CMAS는 자기 주석 작성기(self-annotator), 유형 관련 특징 추출기(type-related feature extractor), 시연 판별기(demonstration discriminator), 전반적 예측기(overall predictor)라는 네 개의 주요 에이전트로 구성됩니다.

- **Technical Details**: CMAS에서는 NER 작업을 두 가지 하위 작업으로 재정의하여 개체 인식(named entities)과 문장 내 개체 유형 관련 특징(type-related features, TRFs)을 식별합니다. 이를 통해 LLM은 맥락적 상관관계를 추출하고, TRFs에 대해 자신이 선택한 시연의 유용성을 평가하는 자기 반성 메커니즘을 통합합니다. 이 방식은 세밀한 In-Context Learning (ICL)을 통해 데이터 간 복잡한 관계를 포착하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, CMAS는 여섯 가지 벤치마크에서 제로샷 NER 성능을 유의미하게 향상시켜, 특정 도메인과 일반 도메인 시나리오 모두에서 효율성을 입증하였습니다. 또한, CMAS는 몇 샷(few-shot) 설정에서도 효과성이 높아 다양한 LLM 백본(backbone)과의 호환성을 보여줍니다. 이는 NER 모델의 일반화 가능성과 적응성을 크게 높이는 결과를 가져옵니다.



### AI Enhanced Ontology Driven NLP for Intelligent Cloud Resource Query Processing Using Knowledge Graphs (https://arxiv.org/abs/2502.18484)
Comments:
          8 pages, 5 figures, 4 tables. This paper not published at else where yet. The experimental setup has a potential to be revised using real time resources. Authors: Krishna Chaitanya Sunkara (IEEE Senior Member, Raleigh, NC, USA, Independent Researcher), Krishnaiah Narukulla (IEEE Senior Member, San Jose, CA, USA, Independent Researcher)

- **What's New**: 이번 논문에서는 전통적인 클라우드 자원 검색 방식을 개선하기 위한 새로운 접근법을 제안합니다. 기존의 키워드 기반 검색이나 GUID에 의존하는 방식은 정확한 일치를 요구하며 사용자가 자원을 찾는데 많은 노력을 필요로 했습니다. 이러한 방식을 개선하기 위해, 논문에서는 자연어 쿼리의 의도를 명확히 이해하고, 자원의 행동이나 운영, 기능, 관계성 등을 기반으로 검색할 수 있는 방법을 모색합니다.

- **Technical Details**: 이 논문은 온톨로지 기반 의미론(ontology-based semantics)과 향상된 자연어 처리(Natural Language Processing, NLP) 기술을 활용하여 사용자가 보다 직관적이고 이해하기 쉬운 쿼리를 생성할 수 있도록 합니다. 클라우드 자원, 그 상호작용 및 행동에 대한 온톨로지를 구축하여 동적 의도 추출(dynamic intent extraction) 및 관련성 순위를 매기는 기능을 구현합니다. Latent Semantic Indexing (LSI)와 AI 모델을 사용하여 자원을 발견할 때의 맥락을 정확히 파악할 수 있는 세미틱 지식 기반(semantic knowledge base)를 형성합니다.

- **Performance Highlights**: 제안된 프레임워크는 AI 기반 데이터 크롤러를 통해 자동으로 온톨로지를 추출하는 파이프라인을 구축하여, 자원 검색의 효율성을 높입니다. 이 시스템은 사용자가 단순히 자원의 목록을 얻는 것이 아니라, 시스템의 행동 원인 분석, 컴플라이언스 체크, 용량 추정, 네트워크 제약 사항 파악 및 문제 해결, 비즈니스 통찰력을 획득하는 데 도움을 줍니다. 결과적으로, 이 접근법은 자원 검색의 효율성을 혁신적으로 향상시키고, 사용자 경험을 크게 개선할 수 있는 잠재력을 지니고 있습니다.



### Modeling Churn in Recommender Systems with Aggregated Preferences (https://arxiv.org/abs/2502.18483)
- **What's New**: 본 논문에서는 이전의 사용자 데이터에 의존하던 추천 시스템(Recommendation Systems, RSs)이 이제 집계된 사용자 정보를 활용해야 한다고 주장합니다. GDPR 및 CCPA와 같은 규제는 개인 사용자 데이터 접근이 제한되어 있어, RS들이 집계된 정보만으로 추천 프로세스를 개선해야 하는 상황을 만들어냅니다. 이러한 변화는 사용자 이탈 위험(Churn Risk)을 증가시키며 이를 해결하기 위한 새로운 모델인 Rec-APC를 제안합니다.

- **Technical Details**: Rec-APC 모델은 RS가 사용자 유형과 콘텐츠 유형에 대한 집계된 만족 수준에 대한 확률적 사전(prior)을 갖는다고 가정합니다. 사용자 세션은 RS가 특정 유형의 알 수 없는 사용자를 샘플링하여 진행되며, RS는 사용자의 좋아요 또는 싫어요 피드백을 받아 추천을 개선합니다. 모델은 탐색(exploration)과 활용(exploitation) 간의 균형을 맞추고, 이탈 위험을 고려하여 사용자 유틸리티를 최대화하는 것을 목적으로 합니다.

- **Performance Highlights**: 본 연구에서 개발된 알고리즘은 Branch-and-Bound 접근 방식을 사용하여 RS의 추천 문제를 해결합니다. 기존의 POMDP(Partially Observable Markov Decision Process)와 비교할 때, 다양한 사용자 유형이 존재하는 경우에 더 나은 성과를 보이는 것으로 나타났습니다. 최적 정책은 유한한 추천 횟수 이후에 수렴하며, 이는 탐색에서 활용으로의 전환을 의미합니다.



### MDE: Modality Discrimination Enhancement for Multi-modal Recommendation (https://arxiv.org/abs/2502.18481)
- **What's New**: 이번 연구에서는 Multi-modal recommendation system의 성능 향상을 위해 Modality Distinctiveness Enhancement (MDE) 프레임워크를 제안합니다. 이 프레임워크는 모달리티 간의 공통적인 정보는 유지하면서 모달리티별 독특한 특성을 추출하고 강화하는 데 중점을 두고 있습니다. 특히, 모달리티 간 관계 정렬과 구별을 위한 균형 기구를 도입하여 추천의 정확성을 높이는 데 기여합니다.

- **Technical Details**: MDE 프레임워크는 이종 사용자-아이템 그래프와 동종 그래프를 구성하여 다중 모달 피처 표현을 학습합니다. 또한, 사용자의 모달리티 선호도를 학습하여 모달리티 특유의 정보와 공유 정보를 효과적으로 융합합니다. 이 과정에서 가중된 모달리티 구별 및 정렬 손실을 도입하여 모달리티별 특징을 증대시키고, 추천 시스템의 품질을 높이는 데 기여합니다.

- **Performance Highlights**: 세 가지 공공 데이터 세트에 대한 광범위한 실험 결과, 제안한 MDE 접근 방식이 기존 최첨단 방법보다 뛰어난 성능을 보였습니다. 모달리티 공유 및 특정 특성을 동시에 고려함으로써, 기존 연구에서 간과되었던 부분을 보완하며 효과적인 추천 결과를 달성했습니다.



### QExplorer: Large Language Model Based Query Extraction for Toxic Content Exploration (https://arxiv.org/abs/2502.18480)
- **What's New**: 이번 연구에서는 정보 검색에서 유효한 쿼리 자동 추출의 도전 과제를 다루며, 특히 독성 콘텐츠 탐색과 관련된 내용입니다. 새로운 접근 방식인 QExplorer를 제안하여, 지능형 대형 언어 모델(LLM)의 기능을 활용하여 유사한 내용의 탐색을 위한 효율적인 쿼리를 직접 추출할 수 있게 되었습니다. 2단계 훈련 과정인 Supervised Fine-Tuning(SFT)과 Direct Preference Optimization(DPO)을 포함하여, 검색 시스템의 피드백을 활용한 데이터 세트 구축이 중요한 요소로 작용합니다.

- **Technical Details**: QExplorer 접근 방식은 두 가지 단계의 훈련 과정을 포함하며, 이는 지침 기반의 SFT와 사용자의 선호도를 정렬하는 DPO를 포함합니다. 이 모델은 오프라인 및 온라인 실험을 통해 효과성을 검증하며, 그 과정에서 LLM 기반의 자동 쿼리 추출 방법이 기존 LLM 및 인간 수행보다 우수한 성능을 보인다고 보고합니다. 이는 키워드 추출을 위한 인간의 직관과 LLM의 성능을 결합하여 더 나은 발견 결과를 도출하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과, QExplorer는 기존 베이스라인 모델과 인간의 쿼리 추출 성능을 초월하는 것으로 나타났습니다. 온라인 배치에서 독성 항목의 탐지가 크게 증가했으며, 이는 QExplorer의 효과적인 쿼리 구성 덕분입니다. 또한 대화형 LLM 기반의 자동 쿼리 추출이 독성 콘텐츠 탐색의 효율성을 크게 향상시킬 수 있음을 입증했습니다.



### Disrupt Your Research Using Generative AI Powered ScienceSag (https://arxiv.org/abs/2502.18479)
Comments:
          This paper has been accepted by Workshop of Deployable AI at AAAI 2025

- **What's New**: 이번 연구에서는 ‘ScienceSage’라는 최소 실행 가능 제품(minimum viable product, MVP) 웹 애플리케이션을 소개합니다. 이는 생성형 인공지능(generative artificial intelligence, GenAI)을 활용하여 연구자들이 제품 혁신의 속도와 범위를 확대할 수 있도록 돕습니다. ScienceSage는 사용자 지식을 벡터 인덱스와 지식 그래프(knowledge graph, KG)로 저장하고 업데이트할 수 있게 해주며, 다양한 자료에서 정보를 추출합니다.

- **Technical Details**: ScienceSage 웹 애플리케이션은 사용자 인터페이스(UI)를 통해 빠르게 테스트할 수 있는 환경을 제공합니다. 이 앱에서는 사용자가 연구 질문에 기반하여 종합적인 연구 보고서를 자동 생성하고, 자신의 문서를 업로드하여 질문하고 즉각적인 답변을 받을 수 있습니다. 사용자는 문서 기반의 RAG(정보 검색 증강 생성) 기능이나 다양한 형식의 다중 모달 데이터를 차별적으로 활용할 수 있습니다.

- **Performance Highlights**: ScienceSage는 최신 정보를 웹에서 검색하여 보고서를 생성하며, 사용자가 원하는 구조와 내용을 갖춘 보고서 템플릿을 제공합니다. 이 플랫폼은 ChatGPT4와 GPT4-Vision을 지원하여 우수한 성능의 LLM을 활용합니다. 실험 결과, 사용자는 과거와 현재의 지식을 통합하여 연구에 필요한 인사이트를 효과적으로 도출할 수 있음을 보여주었습니다.



### Beyond Self-Consistency: Loss-Balanced Perturbation-Based Regularization Improves Industrial-Scale Ads Ranking (https://arxiv.org/abs/2502.18478)
- **What's New**: 이 논문은 대규모 광고 순위 모델에서의Perturbation-based regularization 기법의 성공적인 적용을 처음으로 탐구하고 있습니다. 특히,Loss-Balanced Small Perturbation Regularization (LSPR)라는 새로운 정규화 알고리즘을 제안하여 다양한 딥러닝 모델에 적용 가능성을 보여주고 있습니다. 이 연구는 또한 기존의Self-Consistency Regularization (SCR)보다 LSPR가 더 높은 성능을 보인다는 점을 강조합니다.

- **Technical Details**: 이 논문에서는 대규모 광고 순위 시스템에서의Perturbation-based 정규화 기법에 대해 심층적으로 다루고 있습니다. 특히, LSPR는 입력 데이터에 소규모의 노이즈를 추가하여 새로운 샘플을 생성하고, 이를 훈련 데이터에 포함시키되 손실 함수 계산에서 가중치를 조정하는 방식으로 작동합니다. 이는 더 나은 모델 파라미터 정렬과 낮은 에러를 달성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, LSPR을 적용함으로써 산업 규모의 광고 순위 시스템에서 0.1%에서 0.3%의 상대적 Normalized Entropy (NE) 향상을 달성했습니다. 이러한 성능 개선은 오프라인 실험뿐만 아니라 온라인 실험에서도 확인되었으며, 이는 실제 환경에서도 효과적인 성과를 보여줍니다. 이 연구는 대규모 광고 추천 시스템에Perturbation-based 정규화 기술을 성공적으로 통합한 첫 사례로, 다양한 산업적 적용 가능성을 제시합니다.



### Recommendations Beyond Catalogs: Diffusion Models for Personalized Generation (https://arxiv.org/abs/2502.18477)
- **What's New**: REBECA는 기존 카탈로그의 아이템을 단순히 검색하는 것이 아니라 개별 사용자의 취향에 맞춰 새로운 아이템을 생성하는 생성 추천 시스템입니다. 이는 사용자 피드백을 직접 활용하여 개인화된 추천을 가능하게 하는 첫 번째 확률적 프레임워크로, 사용자의 과거 평가 데이터만을 기반으로 합니다. REBECA는 텍스트 기반의 프롬프트 없이도 유연한 개별화 처리를 통해 추천의 질을 높이는 새로운 가능성을 제시합니다.

- **Technical Details**: REBECA는 사용자의 피드백(예: 좋아요, 싫어요)을 통한 사용자-아이템 상호작용을 파악하여 텍스트 프리 개발 및 추론이 가능한 구조로 설계되었습니다. 이 시스템은 고도로 표현력이 풍부한 사전 훈련된 이미지 생성기 위에 효율적인 어댑터를 추가하여 대규모 언어 모델(LLMs)의 중개 없이 사용자 특정 선호도를 보존하는 데 초점을 둡니다. 이러한 방식은 기존의 생성 파이프라인에 원활하게 통합할 수 있으며, 높은 표현성을 유지하면서 비용이 많이 드는 재훈련을 피할 수 있습니다.

- **Performance Highlights**: REBECA는 실제 데이터로 검증을 진행하여 개인화 지표를 새롭게 제안하고 다양한 실험을 통해 높은 품질의 개인화된 추천 결과를 생성하는 것이 확인되었습니다. 생성된 이미지는 사용자의 독특한 선호를 반영하며, 다양한 평가 방법론을 통해 개인화의 정도를 체계적으로 측정하고 검증합니다. 이러한 접근은 추천 시스템에서의 생성 AI의 통합 가능성을 확장시키며 동적 콘텐츠 생성의 새로운 방향성을 제시합니다.



### FinBloom: Knowledge Grounding Large Language Model with Real-time Financial Data (https://arxiv.org/abs/2502.18471)
Comments:
          27 pages, 9 tables

- **What's New**: 이 논문에서는 금융 쿼리를 처리하기 위해 LLM에 실시간 데이터 모듈과 함께 동작하는 Financial Agent라는 지식 기반 접근 방식을 소개합니다. 연구팀은 50,000개 이상의 금융 쿼리와 해당 문맥을 포함한 Financial Context Dataset을 개발했으며, 금융 뉴스 기사와 SEC 파일들로 훈련된 70억 매개변수의 FinBloom 7B LLM을 제안합니다.

- **Technical Details**: 논문에서는 다양한 도메인 특정 태스크를 효과적으로 수행할 수 있도록 LLM을 동결한 상태에서 모듈을 추가하는 방법을 사용합니다. 이 시스템은 실시간 데이터를 처리하기 위해 두 개의 데이터 레포지토리(탭 형식의 가격 데이터와 텍스트 기반의 뉴스 데이터)를 유지하며, 사용자의 쿼리를 분석하여 관련 데이터를 추출하고 이를 텍스트 형식으로 변환합니다.

- **Performance Highlights**: 제안된 방법론을 통해 금융 쿼리에 대한 응답을 생성하는 데 필요한 동적 문맥 정보를 신속하게 제공할 수 있습니다. 사용자의 쿼리에 최신 데이터를 결합시킴으로써 LLM이 보다 정확한 응답을 생성할 수 있도록 지원하여, 실제 금융 결정을 내릴 때 유용성을 크게 향상시킬 것으로 기대됩니다.



### Spatial-RAG: Spatial Retrieval Augmented Generation for Real-World Spatial Reasoning Questions (https://arxiv.org/abs/2502.18470)
- **What's New**: Spatial reasoning (공간 추론)은 대규모 언어 모델(LLMs)에게 여전히 도전 과제로 남아 있습니다. 본 논문에서는 Spatial Retrieval-Augmented Generation (Spatial-RAG)라는 프레임워크를 제안하여, 희소 공간 검색(Spatial databases)과 밀집 의미 검색(LLM 기반 유사성)을 통합함으로써 공간 작업에 RAG를 확장합니다. 새로운 다중 목표 순위 전략을 통해 공간 제약과 의미적 관련성을 균형 있게 조절하고, LLM 지향 생성기가 일관된 응답을 보장합니다.

- **Technical Details**: Spatial-RAG는 텍스트 기반 공간 검색과 공간적으로 인식되는 텍스트 생성을 통합한 새로운 프레임워크입니다. 우리는 희소 검색(SQL 기반 구조적 쿼리)과 밀집 검색(LLM 기반 의미 매칭)을 결합한 하이브리드 검색 메커니즘을 제안하여, 사용자 쿼리와 공간 및 의미적으로 일치하는 검색 결과를 극대화하여 검색 정확도를 높입니다. 동적으로 공간 및 의미적 관련성 간의 균형을 조절하는 다중 목표 최적화 프레임워크를 통해 생성된 응답의 기하학적 정확성과 언어적 일관성을 보장합니다.

- **Performance Highlights**: 여행 관련 실제 데이터셋을 활용한 실험 결과, Spatial-RAG가 공간 질문 응답에서 효율성을 현저히 향상시킴을 확인했습니다. 이 프레임워크는 구조적 공간 데이터베이스와 자연어 질문 응답 간의 격차를 해소하면서 복잡한 공간 추론 문제를 효과적으로 처리할 수 있는 능력을 보여주었습니다. 따라서 Spatial-RAG는 LLM의 공간 추론 역량을 크게 강화하는 혁신적인 접근 방식으로 자리 잡게 되었습니다.



### Using LLM-Based Approaches to Enhance and Automate Topic Labeling (https://arxiv.org/abs/2502.18469)
Comments:
          7 pages, 2 tables

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)을 활용하여 주제 레이블 자동화를 시도합니다. 기존의 수동 레이블링 방식의 한계를 극복하고, 더 의미 있고 맥락에 맞는 레이블 생성을 목표로 합니다. 이와 함께, 주제 레이블의 정량적 평가 방법을 제시하여, 주제 품질을 측정하는 혁신적인 메트릭을 개발합니다.

- **Technical Details**: 이 방법론에서는 GPT-3.5-Turbo-Instruct 모델을 사용하여 문서 요약을 생성하고, 이를 BERTopic 모델에 입력하여 주제 키워드를 도출합니다. 이어서 4가지 접근 방식을 통해 주제 레이블을 생성하며, 각 접근법은 키워드와 문서 요약을 다르게 활용합니다. 이와 같은 방식으로 N x N 행렬을 생성하여 문서 간 유사성을 측정하고, 주요 문서를 선별하여 주제의 본질을 대표하는 레이블을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 메트릭을 바탕으로 네 가지 접근 방식의 성과를 비교했습니다. 결과는 두 데이터 세트에서 주제 레이블 생성의 품질을 명확히 보여주며, 특히 문서 요약을 활용한 접근 방식이 더 높은 품질의 레이블을 생성한 것으로 나타났습니다. 또한, 새로운 메트릭을 통해 주제 레이블의 의미적 대표성에 대한 정량적 평가가 가능하여, 이는 향후 연구에 중요한 기초 자료가 될 것입니다.



### Efficient Federated Search for Retrieval-Augmented Generation (https://arxiv.org/abs/2502.19280)
Comments:
          To appear in the proceedings of EuroMLSys'25

- **What's New**: 이 논문에서는 RAGRoute라는 새로운 기법을 도입하여 페더레이티드( federated ) RAG 검색을 수행합니다. 기존의 RAG 워크플로우가 단일 벡터 데이터베이스에 의존하는 반면, RAGRoute는 쿼리 시점에서 관련 데이터 소스를 동적으로 선택하여 검색 효율성을 크게 향상시킵니다. 이로 인해 불필요한 정보 검색을 최소화하고, 전체 쿼리 수와 통신량을 각각 최대 77.5%와 76.2%까지 감소시켰습니다.

- **Technical Details**: RAGRoute는 경량( lightweight ) 신경망 분류기를 사용하여, 쿼리 시점에 적절한 데이터 소스를 선택합니다. 이 기법은 사용자가 쿼리를 입력할 때에 모든 데이터를 검색하는 대신, 각 데이터 소스의 특성을 학습하여 관련 있는 소스만을 쿼리하도록 합니다. RAGRoute는 MIRAGE와 MMLU 벤치마크를 사용하여 효과적으로 검증되었으며, 검색의 질을 유지하면서도 리소스 소비를 줄이는 방식으로 작동합니다.

- **Performance Highlights**: RAGRoute의 성능 평가는 두 가지 벤치마크를 통해 이루어졌으며, 높은 검색 회수율(recall)과 관련 데이터 소스를 효과적으로 결정하는 능력을 보여주었습니다. 이 시스템은 대규모 적용에서 응답 시간을 최적화하고 비용 효율성을 높이는 데 매우 유용한 방법임을 입증했습니다.



### TestNUC: Enhancing Test-Time Computing Approaches through Neighboring Unlabeled Data Consistency (https://arxiv.org/abs/2502.19163)
- **What's New**: 이 논문에서는 Test-time computing(테스트 타임 컴퓨팅) 접근 방식을 통해 LLM의 성능을 향상시키는 새로운 방법인 TestNUC를 소개합니다. TestNUC는 인접한 비정답 데이터의 지역적 일관성을 활용하여 예측을 개선하며, 비정답 인스턴스의 예측을 고려하여 입력 인스턴스를 분류합니다. 이 방법은 여러 데이터 세트에서 기존 방법들보다 우수한 성능을 보이며, 테스트 타임 컴퓨팅에 통합할 수 있는 가능성이 높습니다.

- **Technical Details**: TestNUC는 두 가지 주요 단계로 구성됩니다: ❶ Neighbor Retrieval(이웃 검색) 단계에서, 테스트 샘플과 유사한 특징을 가진 K개의 이웃을 식별합니다. ❷ Collaborative Prediction(협력적 예측) 단계에서는 LLM이 테스트 샘플과 그 이웃들에 대한 예측을 생성하고, 이들의 예측을 조합하여 최종 답변을 도출합니다. 이는 LLM이 비정답 샘플의 예측을 포함하여 의사결정을 더 잘 맥락화하고 세분화할 수 있도록 도움을 줍니다.

- **Performance Highlights**: TestNUC는 감정 탐지, 도메인 발견, 주제 채굴, 의도 분류 등 다양한 작업에서 평가되었으며, 기본 방법들인 표준 프롬프트 및 자기 일관성 방식보다 일관되게 더 나은 성능을 발휘했습니다. 특히, 비정답 데이터의 양이 증가할수록 성능이 효과적으로 확장되었고, 다양한 임베딩 모델에서도 강력한 성능을 보였습니다. 또한, TestNUC는 기존의 테스트 타임 컴퓨팅 방법들과 원활하게 통합되며, 성능을 크게 향상시킬 수 있습니다.



### On Aggregation Queries over Predicted Nearest Neighbors (https://arxiv.org/abs/2502.18803)
Comments:
          14 pages, 11 figures, 9 tables

- **What's New**: 이 논문에서는 AQNNs(Nearest Neighbors에 대한 Aggregation Queries)라는 새로운 유형의 집합 쿼리를 소개합니다. 이는 특정 객체의 예측된 이웃에 대한 집합 쿼리로, 예를 들어 불면증 환자와 비슷한 예측 조건을 가진 환자의 평균 수축기 혈압을 계산하려는 의사의 요구에 부응합니다. AQNN은 예측 모델과 집합 값을 계산하는 데 비용이 많이 드는 오라클을 결합하여 근사 집합을 반환하는 쿼리 처리 문제로서 설계되었습니다.

- **Technical Details**: 논문에서는 AQNNs에 대한 쿼리를 처리하기 위해 SPRinT(Sampler with Precision-Recall in Target)라는 프레임워크를 설계하였습니다. SPRinT는 샘플링, 이웃 정교화 및 집합의 세 단계로 구성되며, 다양한 집합 함수에 맞도록 조정되어 있습니다. 또한 샘플 크기와 근사 집합의 오류에 대한 이론적 보장을 제공합니다.

- **Performance Highlights**: 저자들은 의료, 전자상거래 및 비디오 데이터셋에 대한 광범위한 실험을 수행하였으며, SPRinT가 다른 기본 모델에 비해 항상 가장 낮은 집합 오류를 달성하였음을 증명하였습니다. 특히, 데이터셋 크기가 커질수록 SPRinT의 실행 시간과 집합 오류는 안정적이며, 이는 대규모 어플리케이션에 적합하다는 것을 확인시켜 줍니다.



### PII-Bench: Evaluating Query-Aware Privacy Protection Systems (https://arxiv.org/abs/2502.18545)
- **What's New**: 대규모 언어 모델(LLMs)의 광범위한 채택은 사용자 프롬프트에 포함될 수 있는 개인 식별 정보(PII) 노출에 대한 심각한 프라이버시 우려를 불러일으켰습니다. 이를 해결하기 위해 제안된 쿼리 관련 없는 PII 마스킹 전략과 PII-Bench라는 포괄적인 평가 프레임워크는 사용자 프라이버시 보호 시스템의 유용성을 측정하는 데 중요한 기초 자료가 될 것입니다. 이는 다양한 PII 카테고리를 포함한 2842개의 테스트 샘플로 구성됩니다.

- **Technical Details**: PII-Bench는 사용자 쿼리, 맥락 설명, 쿼리와 관련된 PII를 표시하는 표준 응답을 포함하여 세 가지 주요 구성 요소로 세밀하게 설계된 샘플로 이루어져 있습니다. 이 프레임워크는 쿼리 관련성을 결정하는 데 있어 모델들이 직면하는 주목할 만한 한계를 드러내며, 특히 다중 주제 시나리오에서의 PII 식별 문제에 대한 연구를 심층적으로 다룹니다. 쿼리 관련 없는 PII 마스킹 전략을 통해 PII의 중요성과 사용자 쿼리 간의 관계를 고려하여 보다 세밀한 보호 조치를 가능하게 합니다.

- **Performance Highlights**: 실험 분석 결과, 현재 모델들은 기본적인 PII 탐지에서는 양호한 성능을 보이나, 쿼리 관련성 결정에서는 뚜렷한 한계를 드러냈습니다. 최신 LLM조차 이러한 작업에서 어려움을 겪고 있으며, 이는 지능적인 PII 마스킹이 필요함을 강조합니다. 본 연구는 프라이버시 보호 시스템의 효과성을 평가하기 위한 새로운 기준을 제시하며, 향후 연구 방향에 대한 인사이트를 제공합니다.



### FilterRAG: Zero-Shot Informed Retrieval-Augmented Generation to Mitigate Hallucinations in VQA (https://arxiv.org/abs/2502.18536)
Comments:
          12 pages, 6 figures and 2 tables

- **What's New**: FilterRAG이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 BLIP-VQA와 Retrieval-Augmented Generation(RAG)을 통합하여 외부 지식 소스, 예를 들어 Wikipedia와 DBpedia에서 답변의 기반을 확보합니다. 이를 통해 모델이 생성하는 오류 답변, 즉 hallucinations를 줄이고, 현실 세계에 적합한 VQA 시스템을 개선할 수 있는 잠재력을 강조합니다.

- **Technical Details**: FilterRAG는 입력 이미지를 2x2 그리드로 나누고, BLIP-VQA를 통해 시각적 및 텍스트 임베딩을 생성합니다. 이후 관련 지식을 동적으로 검색하여 답변 생성 과정에 통합하며, 이 과정에서 고정된 GPT-Neo 1.3B 모델을 사용합니다. 이러한 방식은 VQA 성능을 향상시키고, OOD(Out-of-Distribution) 시나리오에서 더욱 신뢰할 수 있는 결과를 제공합니다.

- **Performance Highlights**: OK-VQA 데이터셋에서 FilterRAG는 36.5%의 정확도를 달성하였으며, 이는 기존 모델 대비 hallucinations를 현저히 줄였습니다. 필터RAG는 인도메인 및 OOD 설정 모두에서 일관된 성능을 보였으며, 이는 효과적인 지식 검색과 다중 모달 정렬이 VQA의 견고함을 향상시키는 데 중요하다는 것을 보여줍니다.



### A Comprehensive Survey on Composed Image Retrieva (https://arxiv.org/abs/2502.18495)
- **What's New**: 이 논문에서는 Composed Image Retrieval (CIR)이라는 새로운 이미지 검색 작업을 체계적으로 검토하고 있습니다. CIR은 참조 이미지와 사용자가 원하는 변경 사항을 나타내는 수정 텍스트를 조합하여 사용자에게 보다 유연한 검색 방식을 제공합니다. 이 작업에 대한 포괄적인 리뷰가 현재 존재하지 않기 때문에, 120개 이상의 관련 연구를 종합하여 이 분야의 발전을 조망하고 있습니다.

- **Technical Details**: CIR의 주요 기술적 문제는 세 가지 주요 도전 과제를 포함합니다. 첫째, Multimodal Query Fusion은 수정 텍스트와 참조 이미지가 사용자의 검색 의도를 전달하는 데 상보적인 역할을 하는 것을 포함하여, 효과적인 멀티모달 융합 기능을 학습해야 합니다. 둘째, Target Images Matching은 멀티모달 쿼리와 목표 이미지 간의 의미적 간격을 해결하는 것을 목표로 합니다. 마지막으로, Scale of Training Data 문제는 학습 샘플을 만드는 데 필요한 비용과 노동 집약성을 다루고 있습니다.

- **Performance Highlights**: 기존 CIR 모델은 일반적으로 supervised learning과 zero-shot learning 두 가지 방법으로 구분됩니다. 감독 방식에서는 주석이 달린 학습 샘플이 필요하며, 제로샷 방식에서는 대규모 이미지-텍스트 쌍을 활용하여 사전 학습을 수행합니다. 다양한 데이터 세트와 실험 결과를 비교하여 기존의 supervised 및 zero-shot CIR 방법을 분석하고, 향후 연구 방향에 대한 통찰을 제공하여 연구자들에게 유용한 지침을 제시합니다.



### MixLLM: Dynamic Routing in Mixed Large Language Models (https://arxiv.org/abs/2502.18482)
Comments:
          11 pages, 7 figures, accepted by NAACL 2025 main conference

- **What's New**: 새롭고 혁신적인 연구인 MixLLM은 다이나믹 컨텍스트 밴딧 기반의 라우팅 시스템을 개발하여, 쿼리와 LLM의 최적 매핑을 가능하게 합니다. 이 시스템은 쿼리 태그를 활용하여 쿼리 임베딩을 향상시키고, 각각의 LLM에 대한 응답 품질 및 비용을 추정하는 경량화된 예측 모델을 설계합니다. Mixed LLM은 응답 품질과 비용, 지연 시간 간의 균형을 최적화하여 높은 효율을 실현합니다.

- **Technical Details**: MixLLM은 InsTag 모델에서 생성된 태그를 사용하여 쿼리 표현을 개선하는 태그 향상 임베딩 모델을 제안합니다. 예측 모델은 각 LLM의 응답 품질과 비용을 평가하며, 메타 의사결정자는 이러한 예측을 기반으로 최적의 LLM을 선택합니다. 이 과정은 새로운 LLM이 도입되더라도 시스템 전체 재훈련이 필요 없도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, MixLLM은 응답 품질에서 GPT-4의 97.25%를 달성하고, 비용은 24.18%로 유지하여 응답 품질과 비용, 지연 시간 간의 최적의 균형을 실현하였습니다. 또한, 지연 패널티를 도입하여 교통 혼잡 및 높은 지연 문제를 피하고, 다양한 환경 및 사용자 피드백에 적응하는 지속적인 훈련의 장점을 누릴 수 있습니다.



New uploads on arXiv(cs.CV)

### ImageChain: Advancing Sequential Image-to-Text Reasoning in Multimodal Large Language Models (https://arxiv.org/abs/2502.19409)
Comments:
          Code, dataset, and checkpoints are publicly available at this https URL

- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)의 이미지 시퀀스에 대한 추론 능력을 향상시키기 위해 ImageChain이라는 새로운 프레임워크를 소개합니다. 기존의 모델들은 이미지들을 독립적으로 다루거나 전체 장면을 요약하는 것에 그쳤지만, ImageChain은 시각적 시퀀스를 다중 턴 대화로 모델링하여 시간적 의존성을 명확하게 포착할 수 있게 합니다. 이를 통해 다음 장면 설명(next-scene description) 작업에서 비약적인 성과 향상을 달성하였습니다.

- **Technical Details**: ImageChain은 이미지와 그에 해당하는 텍스트 설명을 혼합하여 다음 장면 설명을 생성하기 위한 연속적인 문맥을 구축합니다. 이 프레임워크는 약 4000개의 훈련 샘플만으로도 SimRate라는 메트릭에서 평균 3.7%에서 19%로의 성과 향상을 보여줍니다. 또한, 다양한 문맥 길이에 걸쳐 훈련하는 것이 일관되게 개선된 성능을 보이며, 이는 시퀀스 추론 능력을 향상시키는데 도움을 줍니다.

- **Performance Highlights**: ImageChain은 로봇 공학 분야에서 F1 점수 27.1을 기록하여 기존 모델보다 두 배 높은 성과를 달성하였습니다. 또한 만화와 같은 다른 구조적 설정에서도 성과 향상을 보입니다. 최종적으로, StoryFrames 데이터셋을 활용하여 다양한 맥락에서 시퀀스 이미지-텍스트 추론 연구를 지원하는 고품질 샘플을 제공합니다.



### Model Adaptation: Unsupervised Domain Adaptation without Source Data (https://arxiv.org/abs/2502.19316)
Comments:
          accepted by CVPR2020

- **What's New**: 이번 연구에서는 라벨이 없는 목표 데이터만을 사용하여 기존의 예측 모델 성능을 개선하는 새로운 설정인 비지도 모델 적응(unsupervised model adaptation)을 제안합니다. 이를 위해, 소스 데이터의 의존성을 극복하기 위한 협업 클래스 조건 생성적 적대 신경망(Collaborative Class Conditional Generative Adversarial Net, 3C-GAN)을 개발하였습니다. 이 모델은 생성된 목표 스타일 데이터를 통해 예측 모델을 향상시키는 방법을 탐구합니다.

- **Technical Details**: 우리는 기존의 예측 모델이 라벨이 없는 목표 데이터만을 기반으로 새로운 도메인에 적응하도록 하였습니다. 3C-GAN은 클래스 조건 생성기와 판별기를 통해 각 도메인의 분포를 일치시키기 위해 적대적 훈련을 수행합니다. 교육 과정에서, 예측 모델을 안정화하고 성능을 개선하기 위해 소스 모델과 유사성을 권장하는 가중치 제약(weight constraint)을 도입하였습니다.

- **Performance Highlights**: 다수의 비지도 도메인 적응 벤치마크에서 실험을 수행한 결과, 제안한 모델이 기존의 최첨단 연구 결과를 초월하는 성능을 보였습니다. 특히, 소스 데이터가 전혀 없는 상황에서도 효과적인 성능 향상을 확인하였으며, 이는 본 연구의 결과가 실제 환경에서도 유용함을 시사합니다.



### CoopDETR: A Unified Cooperative Perception Framework for 3D Detection via Object Query (https://arxiv.org/abs/2502.19313)
Comments:
          8 pages, 8 figures, ICRA 2025

- **What's New**: 본 논문에서는 CoopDETR이라는 새로운 협동 인식 프레임워크를 제안하여, 객체 수준의 특성을 통한 협력적 인식을 구현합니다. 기존의 지역 수준 특성 전송 방법은 해석 가능성이 제한적이며 상당한 대역폭을 요구하는 문제점이 있습니다. CoopDETR는 이러한 문제를 해결하기 위해 객체 쿼리를 통해 객체 수준 특성 협력을 도입하며, 단일 에이전트 쿼리 생성 및 교차 에이전트 쿼리 융합이라는 두 가지 주요 모듈로 구성됩니다. 실험 결과는 CoopDETR이 이전 방법에 비해 성능을 개선하고 전송 비용을 1/782로 줄인 것을 보여줍니다.

- **Technical Details**: CoopDETR는 두 가지 기본 모듈로 구성되어 있습니다: 단일 에이전트 쿼리 생성 모듈과 교차 에이전트 쿼리 융합 모듈입니다. 단일 에이전트 쿼리 생성에서는 transformer 기반 모델을 활용하여 포인트 클라우드(features)를 기반으로 쿼리를 업데이트합니다. 교차 에이전트 쿼리 융합 모듈에서는 다른 에이전트로부터 수신된 쿼리를 Spatial Query Matching(SQM)을 사용하여 유사한 쿼리를 관계짓고, Object Query Aggregation(OQA) 과정을 통해 정보를 융합합니다. 이 쿼리들은 최종 예측을 위해 탐지 헤드에 입력됩니다.

- **Performance Highlights**: CoopDETR은 OPV2V 및 V2XSet 데이터셋에서 실험을 통해 경쟁력 있는 성능을 보였습니다. 협동 인식 방법 중에서 최고의 결과를 달성하고, 전송 비용을 대폭 감소시키면서 성능의 향상을 이루었습니다. 이 새로운 프레임워크는 특히 복잡한 트래픽 상황에서 자율차의 전반적인 인식 능력을 향상시키고 더 효과적인 소통을 가능하게 합니다.



### Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions (https://arxiv.org/abs/2502.19293)
Comments:
          11 pages, 2 figures

- **What's New**: 이 연구에서는 피부 멜라닌 세포 병변 진단을 위한 비전-언어 모델(vision-language model)을 새롭게 개발하였습니다. 모델은 Contrastive Captioner 프레임워크에 기반하여, 42,512개의 H&E 염색 전체 슬라이드 이미지와 19,645개의 병리 보고서를 사용하여 훈련 및 평가되었습니다. 이 연구는 병리 보고서 작성의 자동화가 병리학자의 업무 부담을 줄일 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: 실험에 사용된 데이터셋은 네덜란드 유트레흐트 대학교 병리학과의 디지털 아카이브에서 후향적으로 수집된 멜라닌 세포 병변 사례들로 구성되어 있습니다. 데이터셋은 2013년 1월 1일부터 2020년 12월 31일 사이에 접수된 모든 unique H&E 염색 전체 슬라이드 이미지와 해당 병리 보고서를 포함하였으며, 각 보고서는 필드에서 경험이 풍부한 병리학자가 평가하였습니다. 모델은 PRISM을 기반으로 하여, 비대칭 주의(attention) 메커니즘을 활용한 Perceiver를 통해 입력 feature vector에서 정보 추출을 수행합니다.

- **Performance Highlights**: 모델이 생성한 보고서의 품질 점수는 일반적인 신경 모양의 병변에 대해 실제 병리학자가 작성한 보고서와 동등하였으며, 사용자 연구를 통해 전문가의 평가를 받았습니다. 드물게 발생하는 병변의 하위 유형에 대해서는 보고서 생성이 더 어려웠지만, 이러한 경우의 교차 모달 검색(cross-modal retrieval) 성능은 상당히 개선되었습니다.



### On the Importance of Text Preprocessing for Multimodal Representation Learning and Pathology Report Generation (https://arxiv.org/abs/2502.19285)
Comments:
          11 pages, 1 figure

- **What's New**: 본 연구는 병리학에서 비전-언어 모델이 멀티모달 사례 검색 및 자동 보고서 생성을 가능하게 한다는 점을 강조합니다. 기존 모델들은 병리학 보고서의 정보를 기반으로 훈련되었지만, 이 정보는 H&E 염색 슬라이드 이미지에서 정확히 유추할 수 없는 경우가 있어 생성된 보고서에 "할루시네이션"이 발생할 가능성이 있습니다. 이러한 문제를 해결하고자, 우리는 전체 보고서로 훈련된 모델과 H&E 관련 세포 및 조직 외관을 설명하는 문장만 포함한 모델을 비교하여 멀티모달 표현의 품질과 생성된 보고서의 질이 어떻게 영향을 받는지를 평가하였습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 네덜란드 유트레흐트 대학 병리학과의 디지털 아카이브에서 수집된 42,433개의 H&E 염색 전체 슬라이드 영상(WSI)과 19,636개의 병리학 보고서를 포함합니다. 본 연구에서는 BLIP-2 프레임워크를 기반으로 하여, 이미지-텍스트 대조 손실 및 이미지 기반 텍스트 생성 등의 훈련 기법을 활용하였습니다. 모델은 H&E 관련 문장만 사용하여 훈련 시켜서 보고서의 품질을 개선하고, 보고서의 해부학적 세부 정보를 제거하는 기존의 복잡한 후처리 절차를 피했습니다.

- **Performance Highlights**: 실험 결과, 텍스트 전처리를 통해 보고서 생성 시 할루시네이션을 방지할 수 있음을 확인하였습니다. 전반적으로, 전체 병리학 보고서로 훈련된 모델이 교차 모달 검색 성능에서 우수한 성능을 보였나요, H&E 관련 문장만으로 훈련된 모델에서는 생성된 보고서의 품질이 향상되었습니다. 이를 통해 병리학적 데이터를 기반으로 한 비전-언어 모델의 효율성을 평가하고, 향후 자동화된 시스템 개발 방향을 제시하였습니다.



### Neural Antidote: Class-Wise Prompt Tuning for Purifying Backdoors in Pre-trained Vision-Language Models (https://arxiv.org/abs/2502.19269)
- **What's New**: 이 논문에서는 Class-wise Backdoor Prompt Tuning (CBPT)이라는 새로운 방어 방법을 제안합니다. 기존의 방어 전략들이 모델 전체를 적절히 미세 조정하는 데 집중했지만, CBPT는 텍스트 프롬프트를 통해서 독성 효과를 우회하고 모델의 결정 경계를 수정합니다. 이 방법은 백도어 공격에 대한 효과적인 해결책으로서 인식됩니다.

- **Technical Details**: CBPT는 고급 대조 학습 방식을 사용하여 공격자가 채택한 백도어 트리거를 효과적으로 역전시키는 기술입니다. 이를 통해 각 클래스에 대한 텍스트 프롬프트를 최적화하며, 공격자가 주입한 더미 트리거로 모델의 결정 경계를 조정합니다. 이러한 bi-level optimization 접근법을 통해 프롬프트 조정이 이루어지며, 모델의 가중치는 고정됩니다.

- **Performance Highlights**: 광범위한 실험에 따르면 CBPT는 7개의 대표적인 백도어 공격에 대해 공격 성공률(ASR)을 0.39%로 크게 줄였고, 평균적인 청결 정확도(CA)는 58.86%로 유지되었습니다. 이러한 결과는 CBPT의 효과성을 증명하며, 방어 성능 측면에서 동종 최첨단 방어 수단에 비해 월등한 결과를 나타냈습니다.



### EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region (https://arxiv.org/abs/2502.19260)
Comments:
          19 pages, 6 figures

- **What's New**: 이번 논문에서는 아랍만 지역에서 수집된 최초의 공개 자율주행 데이터셋인 Emirates Multi-Task (EMT) 데이터셋을 소개합니다. EMT 데이터셋은 고유한 도로 토폴로지, 높은 교통 혼잡도, 다양한 보행자 복장과 기상 조건 등의 특성을 포착하고 있습니다. 30,000개 이상의 프레임과 570,000개의 주석이 추가된 바운딩 박스를 포함하며, 약 150킬로미터의 드라이빙 경로를 다룹니다.

- **Technical Details**: 이 데이터셋은 추적 (tracking), 궤적 예측 (trajectory forecasting), 의도 예측 (intention prediction)의 세 가지 주요 작업을 지원합니다. 각 벤치마크 데이터셋은 다중 에이전트 추적 실험, 궤적 예측 평가, 의도 예측 실험을 포함하여 각각의 평가 결과로 보완됩니다. EMT 데이터셋은 UAE에서 수집되었으며, 다양한 교통 장면과 특성을 포착하기 위해 차량과 보행자에 대한 주석이 포함되어 있습니다.

- **Performance Highlights**: 각 작업별 데이터셋은 평가 모델과 실험으로 보완됩니다. 다중 에이전트 추적에서는 Kalman 필터 기반의 최신 추적기 (SOTA trackers)를 평가하며, 궤적 예측에 대해서는 고밀도 시나리오에서의 시간적 의존성 및 상호작용 동역학을 포착하는 딥러닝 아키텍처를 사용합니다. 의도 예측 작업에서는 과거 궤적을 바탕으로 미래의 의도를 예측하는 LSTM 기반 모델의 성능을 평가합니다.



### ProxyTransformation: Preshaping Point Cloud Manifold With Proxy Attention For 3D Visual Grounding (https://arxiv.org/abs/2502.19247)
Comments:
          11 pages, 3 figures

- **What's New**: 최근 제안된 Proxy Transformation 방법은 사고 중심의 3D 시각적 기초 작업에서 점 구름(point cloud)의 성능을 크게 향상시킵니다. 기존의 점 구름 향상 방법들은 실시간 처리에 부적합했지만, 본 연구는 이러한 문제를 해결하고 점 구름의 다중 모달(multi-modal) 정보를 최적화하는 방법을 제시합니다. 이 방법은 Deformable Point Clustering을 통해 목표 지역의 하위 매니폴드를 식별하고, Proxy Attention 모듈을 통해 점 구름 변환을 안내합니다.

- **Technical Details**: Proxy Transformation은 3D 환경에서의 즉각적 작업에 적합하게 설계되었습니다. 이 접근 방식은 각 서브 매니폴드(submanifold)를 위한 변환을 생성하는 데 있어, 국소적인 정보와 전역적인 관계를 결합하여 처리합니다. 텍스트 및 이미지 정보를 활용하여 하위 매니폴드를 향상시키는 이 방법은 복잡한 시나리오에서도 유연한 디formable 클러스터를 생성할 수 있도록 합니다.

- **Performance Highlights**: 초기 실험 결과, Proxy Transformation은 기존의 모든 방법을 능가하는 성능을 발휘하였습니다. 특히 쉬운 목표에서 7.49%, 어려운 목표에서 4.60%의 성능 향상을 달성했으며, 주의 블록의 계산 과부하를 40.6% 감소시켰습니다. 이러한 결과는 사고 중심의 3D 시각적 기초에서 새로운 SOTA(State of the Art)를 설정하는 데 기여했습니다.



### Arbitrary Volumetric Refocusing of Dense and Sparse Light Fields (https://arxiv.org/abs/2502.19238)
Comments:
          9 pages, 7 figures, 3 tables

- **What's New**: 이번 연구에서는 다수의 임의의 평면 또는 볼륨 영역을 동시에 재초점화(refocus)할 수 있는 엔드-투-엔드(end-to-end) 파이프라인을 제안합니다. 특히, Sparse Light Fields(SLFs)에 대해 픽셀 종속적(depth or disparity)을 이용하여 이동할 픽셀을 결정하는 방법을 제시함으로써 원하는 영역을 초점 내로 이동시키는 동시에 동일한 깊이 범위의 다른 영역은 초점 밖으로 유지할 수 있습니다. 이는 기존의 방법들과의 가장 큰 차별점으로, 다수의 지역을 독립적으로 조정하는 데 성공했습니다.

- **Technical Details**: 연구에서 제안하는 방법은 U-Net 아키텍처 기반의 딥 러닝 모델을 사용하여 Sparse LFs에서 발생할 수 있는 고스트 아티팩트(ghosting artifacts)를 거의 완전히 제거합니다. 제안된 방법은 먼저 선택한 영역의 깊이 맵(depth map)을 생성하고, 그 후에 픽셀 이동을 통해 재조정하는 방식으로 작동합니다. Sparse LFs의 경우, 특정 깊이 범위에 대해 상당히 빠른 검색을 통해 필요한 이동 값을 결정합니다.

- **Performance Highlights**: 실험 결과 제안된 방법은 Sparse LFs에 대해 구조적 유사도 지수(structural similarity index) 0.9 이상을 기록하며, 데이터 압축률이 20%에 불과한 상태에서도 Dense LFs와 유사한 품질의 이미지를 생성합니다. 또한, Dense LF의 처리 시간은 약 3.64초이며, 같은 데이터를 묘사하는 Sparse LF의 처리 시간은 0.71초로 확인되었습니다. 이는 Resource-constrained devices에서 실시간으로 LF 처리 방법을 구현할 수 있는 가능성을 제시합니다.



### A Lightweight and Extensible Cell Segmentation and Classification Model for Whole Slide Images (https://arxiv.org/abs/2502.19217)
Comments:
          27 pages, 11 figures

- **What's New**: 본 논문에서는 디지털 병리학에서 세포 수준 분석 도구 개발의 어려움을 해결하기 위해 경량화되고 확장 가능한 cell segmentation 및 classification 모델을 제안합니다. 데이터 레이블을 교차 레이블링(cross-relabeling)하여 PanNuke 및 MoNuSAC의 주석을 정제하고, 일곱 가지 서로 다른 세포 타입을 포함하는 통합 데이터셋을 생성했습니다. 이러한 과정에서 모델 성능과 데이터 품질을 향상시키는 방법을 소개합니다.

- **Technical Details**: H-Optimus 기초 모델을 고정된 인코더로 활용하여 시퀀스 분할(segmentation) 및 분류(classification) 작업을 위한 특징 표현(feature representation)을 개선했습니다. 모델 크기와 복잡성을 줄이기 위해 지식을 증류(distillation)하였고, 이는 비교 가능한 성능을 유지하면서도 모델 파라미터 수를 48배 줄였습니다. 마지막으로, 이 증류 모델을 널리 사용되는 오픈 소스 디지털 병리학 플랫폼인 QuPath에 통합했습니다.

- **Performance Highlights**: H-Optimus 기반 모델을 사용한 세분화(segmentation) 및 분류(classification) 성능이 CNN 기반 모델보다 더 우수함을 보여줍니다. 평균 $R^2$ 값이 0.575에서 0.871로, 평균 $PQ$ 점수가 0.450에서 0.492로 개선되어 실제 세포 수와의 일치도가 높아졌습니다. 이러한 접근은 진단에 중대한 영향을 미치고 병리학자의 작업 부담을 줄이며 결과를 개선할 수 있는 가능성을 보여줍니다.



### Distill Any Depth: Distillation Creates a Stronger Monocular Depth Estimator (https://arxiv.org/abs/2502.19204)
Comments:
          project page: this https URL

- **What's New**: 이 논문에서는 단안 깊이 추정(Monocular Depth Estimation, MDE)에서의 페이소드 정제(pseudo-label distillation)의 효과를 높이기 위한 새로운 방법인 Cross-Context Distillation을 제안합니다. 기존의 깊이 정규화 방법들이 가지는 문제점을 분석하였고, 이를 해결하기 위해 글로벌 및 로컬 깊이 신호를 결합하여 페이소드 품질을 향상시키는 접근법을 제안합니다. 또한, 여러 깊이 추정 모델의 장점을 활용하는 다중 교사(distillation) 프레임워크도 도입하여 더 강력하고 정확한 깊이 예측을 가능하게 합니다.

- **Technical Details**: 이 논문에서는 깊이 정규화 방식이 페이소드 정제에서 미치는 영향을 체계적으로 분석합니다. 글로벌 정규화(global normalization)와 로컬 정규화(local normalization) 및 혼합 정규화(hybrid global-local approaches)를 포함한 다양한 정규화 전략을 조사하였고, 각각이 MDE 성능에 미치는 영향을 실험적으로 검증하였습니다. 이를 통해 서로 다른 정규화 방법이 MDE 손실 함수(loss function)와 정제 결과에 미치는 영향을 돕기 위한 최적의 관행을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안한 Cross-Context Distillation 방법과 다중 교사(distillation) 프레임워크가 기존의 최첨단 방법들보다 정량적 및 정성적으로 우수한 성능을 보임을 입증했습니다. 벤치마크 데이터셋에서 폭넓은 실험을 수행한 결과, 이 방법은 깊이 예측에서 더 높은 정확도를 달성하며, 결합된 다양한 맥락 정보를 활용하여 일반화 성능을 향상시킵니다. 코드와 모델은 공개될 예정입니다.



### HDM: Hybrid Diffusion Model for Unified Image Anomaly Detection (https://arxiv.org/abs/2502.19200)
- **What's New**: 이 논문에서는 생성(generation)과 구별(discrimination) 작업을 통합한 새로운 하이브리드 확산 모델(Hybrid Diffusion Model, HDM)을 제안합니다. 이 모델은 세 가지 주요 모듈로 구성되며, 이로 인해 서로 다른 이상 패턴을 처리하는 데 있어 더 나은 성능을 발휘합니다. 특히, 고품질 이상 샘플을 생성하고 이를 기반으로 이상 지역 탐지를 정확하게 수행할 수 있도록 설계되었습니다.

- **Technical Details**: 하이브리드 확산 모델은 확산 이상 생성 모듈(Diffusion Anomaly Generation Module, DAGM), 확산 구별 모듈(Diffusion Discriminative Module, DDM), 확률 최적화 모듈(Probability Optimization Module, POM)의 세 가지 모듈로 이루어져 있습니다. DAGM은 다양한 이상 샘플을 생성하여 샘플의 대표성을 강화하며, DDM은 반전 확산 과정을 통해 정상 샘플과 생성된 샘플 간의 차이를 포착합니다. POM은 생성 및 구별 단계 동안 확률 분포를 개선하여 높은 품질의 샘플을 사용하는 데 기여합니다.

- **Performance Highlights**: 다양한 산업 이미지 데이터 세트에서 실시된 실험 결과, 본 연구의 방법은 최신 방법들보다 뛰어난 성능을 발휘했습니다. 특히, AUROC를 기준으로 이미지 수준과 픽셀 수준의 이상 탐지 성능이 크게 향상되었습니다. 이러한 결과는 제안된 모델이 복잡한 산업 환경 내에서 효과적으로 이상을 탐지할 수 있음을 시사합니다.



### EGR-Net: A Novel Embedding Gramian Representation CNN for Intelligent Fault Diagnosis (https://arxiv.org/abs/2502.19199)
- **What's New**: 이 논문은 회전 기계의 고장 진단에서 중요한 기능 추출 방법으로서, 복잡한 1D 진동 신호를 간단한 텍스처를 가진 2D 이미지로 변환하는 새로운 1D-to-2D 변환 방법인 Embedding Gramian Representation (EGR)을 제안합니다. 기존의 방법들과 달리 EGR은 계산이 간단하고 우수한 분리성(separability)을 보여줍니다. 또한, EGR 기반의 더블 브랜치 CNN 모델인 EGR-Net이 제안되어, 원시 신호의 특징 맵과 EGR을 동시에 학습할 수 있도록 설계되었습니다.

- **Technical Details**: EGR은 1D 신호를 포함된 에벨딩 공간에서 처리하며, 신호의 내재적 주기성을 포착합니다. 이 방법은 원시 신호 행렬(RSM)을 구성하고, 그에 대한 Gramian을 계산하는 두 단계로 구현됩니다. EGR 알고리즘은 행렬 곱셈(matrix multiplication)으로만 계산되며, 이러한 접근은 정보 중복을 크게 줄이고, 생성된 특성들은 분리성이 좋게 나타납니다.

- **Performance Highlights**: 제안된 EGR-Net은 기존 CNN 모델의 단일 입력에서 발생하는 정보 손실 문제를 줄이기 위해 고안되었습니다. 이 모델은 파라미터 선택 규칙을 논의하고, EGR의 진행 과정에서 브리지 연결을 통해 두 브랜치 간의 특성 학습 상호작용을 개선합니다. 이를 통해 EGR-Net은 전통적인 방법들과 비교하여 향상된 성능을 보여줍니다.



### Self-supervised conformal prediction for uncertainty quantification in Poisson imaging problems (https://arxiv.org/abs/2502.19194)
- **What's New**: 본 논문은 Poisson 이미징(고장 관리) 문제를 위한 자가 감독적(conformal prediction) 방법을 제안합니다. 이 방법은 실제 데이터의 필요성을 없애고, Poisson Unbiased Risk Estimator를 활용하여 신뢰할 수 있는 불확실성 정량화를 가능하게 합니다. 제안된 방법은 이미지 노이즈 제거 및 분산 해소와 같은 다양한 포화 문제에 적용할 수 있으며, 최신 자기 관리(self-supervised) 기술과 결합하여 효과적입니다.

- **Technical Details**: Poisson 이미징 문제에서는 관측된 데이터에 대한 넓은 범위의 솔루션이 존재합니다. 기존의 Bayesian 통계 기법들은 이를 해결하는 데 주로 사용되지만, 이러한 방법들이 더 큰 구조에 대한 정확한 불확실성 정량화에 어려움을 겪고 있습니다. 본 연구에서는 Poisson 노이즈가 포함된 고차원 선형 역문제에 대한 자가 감독적 정형화 예측 방법을 소개합니다.

- **Performance Highlights**: 수치 실험 결과, 제안한 방법은 지도 학습(지도 기반) 정형화 예측 방법과 유사한 성능을 보여줍니다. 특히, 잘 정의되지 않은 문제에 효과적으로 적용될 수 있으며, 현실적인 과학적 데이터 환경에서도 실용성을 지니고 있습니다. 이는 과학적 결정-making이나 연구에서 중요한 역할을 할 수 있습니다.



### Knowledge Distillation for Semantic Segmentation: A Label Space Unification Approach (https://arxiv.org/abs/2502.19177)
- **What's New**: 최근 몇 년간 유사한 도메인에서의 시맨틱 세그멘테이션을 위한 데이터셋이 증가하고 있으나, 다양한 데이터셋의 분류 체계(taxonomy)와 레이블 정책(labeling policies)의 일관성 부족으로 인해 더욱 큰 모델을 훈련하는 것이 어려운 상황입니다. 이를 해결하기 위해 제안된 지식 증류(knowledge distillation) 접근 방식은 레이블 공간 통합(label space unification) 방법으로 기능하며, 교사 모델(teacher model)을 통해 추가 데이터를 위한 의사 레이블(pseudo-label)을 생성합니다.

- **Technical Details**: 제안된 방법은 유사한 auxiliary 데이터셋을 위한 의사 레이블을 생성하기 위해 소스 데이터셋에 기반한 교사 모델을 훈련합니다. 이 과정에서 사전 정의된 온톨로지 매핑(ontology mapping)을 통해 소스와 보조 분류 체계 간의 제약을 설정하여 보다 정확한 의사 레이블을 생성합니다. 이를 통해 얻은 데이터로 학생 모델(student model)을 훈련하여 더 나은 성능을 발휘하는 결과를 얻습니다.

- **Performance Highlights**: 이 연구에서 제안된 의사 레이블은 두가지 도전적인 도메인인 도시(urban)와 오프로드(off-road)에서 교사 모델보다 더 높은 성과를 기록하는 학생 모델을 훈련하는 데 사용됩니다. 도시 도메인에서는 12개 공공 데이터셋에서 388,230장, 오프로드 도메인에서는 7개 데이터셋에서 18,558장의 이미지를 포함함으로써 자율 주행을 위한 가장 큰 복합 데이터셋을 생성하였습니다.



### A Sliding Layer Merging Method for Efficient Depth-Wise Pruning in LLMs (https://arxiv.org/abs/2502.19159)
- **What's New**: 이 논문은 Depth-wise pruning이 모델 성능을 크게 저하시킬 수 있다는 점을 지적하며, 이를 해결하기 위해 Sliding Layer Merging 방법을 제안합니다. 이 방법은 여러 Transformer Layer의 출력 상관관계를 분석하여, 특정 기준 유사성에 따라 인접한 Layer들을 동적으로 선택하고 병합합니다. 실험 결과, 우리의 방법이 기존의 pruning 기법보다 뛰어난 성능을 보이며, zero-shot inference 성능에서도 현저한 개선을 달성했습니다.

- **Technical Details**: 연구에서는 reproducing kernel Hilbert space을 이용해 LLM의 Layer 간의 상관관계를 분석했습니다. 이를 통해 'Patch-like' feature 관계를 발견하였고, Sliding Layer Merging 방법은 깊은 Layer에서 얕은 Layer로의 병합을 통해 모델 구조를 단순화함과 동시에 성능을 유지합니다. 이 과정에서는 기준 Layer와 인접 Layer 간의 유사성을 측정해 병합 여부를 결정하는 슬라이딩 윈도우 메커니즘을 사용합니다.

- **Performance Highlights**: 다양한 LLM 아키텍처에 대한 실험 결과, Sliding Layer Merging 방법이 기존 방법들보다 zero-shot performance가 우수함을 보여주었습니다. 특히, Vicuna-7B 모델에서 35%의 pruning을 적용했을 때, 평균 성능이 기존 방법에 비해 1.654% 향상되었습니다. 또한 폭과 깊이의 pruning을 결합하여 모델 압축 성능을 더욱 향상시킬 수 있는 가능성도 제시되었습니다.



### SCA3D: Enhancing Cross-modal 3D Retrieval via 3D Shape and Caption Paired Data Augmentation (https://arxiv.org/abs/2502.19128)
Comments:
          ICRA 2025

- **What's New**: 이번 논문에서는 cross-modal 3D retrieval(교차 모달 3D 검색)을 위한 새로운 온라인 데이터 증강 방법인 SCA3D를 소개합니다. 이는 제한된 3D 데이터의 문제를 해결하기 위한 접근으로, LLaVA 모델을 활용하여 3D 형상의 각 세그먼트에 대한 캡션을 생성합니다. 이 방식을 통해 새로운 의미적 특징을 포함한 대규모 3D-텍스트 쌍이 생성되며, 다양한 구성 요소들을 조정하여 새로운 3D 형상을 만듭니다.

- **Technical Details**: SCA3D 방법론은 compositional library와 텍스트 템플릿을 활용하여 3D와 텍스트 간의 정교한 cross-modal similarity를 맞추기 위해 Earth Mover's Distance (EMD)를 사용합니다. 데이터 증강을 통해 기존의 데이터 부족 문제를 완화하고, unimodal encoders를 통해 3D 형상과 텍스트의 임베딩을 추출하여 보다 나은 매칭 성능을 구현합니다. 또한, 비대칭적 정렬을 위해 contrastive learning을 채택하고 InfoNCE loss를 적용하여 교차 모달 정렬의 효과를 증대시킵니다.

- **Performance Highlights**: 실험 결과, SCA3D는 Text2Shape 데이터셋에서 이전의 방법을 능가하며, Shape-to-Text RR@1 점수를 20.03에서 27.22로, Text-to-Shape RR@1 점수를 13.12에서 16.67로 크게 향상시켰습니다. 이러한 결과는 SCA3D의 우수한 성능과 더불어 다양한 시나리오에서의 강력한 일반화 능력을 입증합니다. 코드 및 추가 정보는 제공된 URL에서 확인할 수 있습니다.



### The NeRF Signature: Codebook-Aided Watermarking for Neural Radiance Fields (https://arxiv.org/abs/2502.19125)
Comments:
          16 pages, accepted by TPAMI

- **What's New**: 본 논문에서는 Neural Radiance Fields (NeRF)에 대한 새로운 워터마킹 방법인 NeRF Signature를 제안합니다. 이 방법은 Codebook-aided Signature Embedding (CSE) 기술을 사용하여 모델 구조를 변경하지 않고 서명(혹은 시그니처)을 통합하는데 중점을 둡니다. 기존의 워터마킹 방법들이 가진 한계를 해결하고, 저작권 보호의 효율성과 편리성을 높이기 위한 설계를 구현했습니다.

- **Technical Details**: NeRF Signature는 모델 레벨에서의 강건성과 불분명성을 보장하기 위해 3가지 주요 기술을 포함합니다. 첫째, CSE를 통해 NeRF의 매개변수에 서명을 직접 추가할 수 있어 외부 모듈의 필요성을 최소화합니다. 둘째, 공동 자세-패치 암호화 워터마킹 전략을 도입하여 특정 시점에서 렌더링된 패치에 서명을 숨깁니다. 셋째, Complexity-Aware Key Selection (CAKS) 방식을 통해 고시각적 복잡성을 지닌 패치에 서명이 추가되어 불분명성을 강화합니다.

- **Performance Highlights**: 실험 결과, NeRF Signature는 기존의 기준 방법들과 비교해 불분명성과 강건성 면에서 우수한 성능을 보였습니다. 특히, 서명 통합 시 구조 변경 없이 매끄럽게 이루어지며, 사용자 편의성을 크게 향상시키고 있습니다. 코드와 실험 데이터는 공개되어 있어, 연구자들이 추가 연구를 쉽게 진행할 수 있습니다.



### A Survey on Foundation-Model-Based Industrial Defect Detection (https://arxiv.org/abs/2502.19106)
Comments:
          14 pages, 4 figures

- **What's New**: 최근 기초 모델(Foundation Model)의 등장으로 시각적 산업 결함 탐지(visual defect detection)에 대한 접근 방법이 혁신적으로 변화하고 있습니다. 전통적인 방법이 통계적 분석과 같은 고전적인 기법에 의존하는 반면, 기초 모델을 활용한 방법들은 더욱 높은 정확도를 보여 주며, 특히 몇 샷(few-shot)과 제로 샷(zero-shot) 학습에 적합합니다. 본 논문은 이러한 방법들을 체계적으로 분석하고 NFM(Non-Foundation Model) 방법과의 차이점을 설명합니다.

- **Technical Details**: 기초 모델(FM) 이용한 결함 탐지에는 SAM, CLIP, GPT 같은 다양한 모델이 있습니다. SAM은 시각 세분화(segmentation)를 통해 산업 결함 탐지의 정확도를 향상시키고, CLIP은 이미지와 텍스트의 정교한 매칭을 지원합니다. GPT는 복잡한 시나리오에서 길고 구조화된 설명 생성에 강점을 가지고 있으며, 이로 인해 2D 및 3D 결함 탐지에서 효과적인 솔루션을 제공합니다.

- **Performance Highlights**: 연구 결과, 기초 모델을 활용한 방법들은 수집된 데이터가 적은 자원 부족 환경에서도 우수한 성능을 보입니다. 특히, FM 방법들은 복잡한 이상 탐지에서 높은 정확도를 가지며, 다양한 산업 환경에 신속하게 적응할 수 있는 장점이 있습니다. NFM 방법들은 작은 파라미터 크기와 높은 계산 효율성 덕분에 특정 응용 시나리오에서 독보적인 장점을 갖지만, FM 방법들은 다중 작업 및 다중 도메인 탐지에 적합합니다.



### An anatomically-informed correspondence initialisation method to improve learning-based registration for radiotherapy (https://arxiv.org/abs/2502.19101)
Comments:
          Presented at the XXth International Conference on the use of Computers in Radiation therapy. Pages 99-102 in XXth ICCR Proceedings, found here this https URL

- **What's New**: 본 논문에서는 개별 환자 CT 비탄력 등록(interpatient CT non-rigid registration) 초기화 방식으로 해부학적 정보를 활용한 새로운 방법을 제안합니다. 이 방법은 organ structures 간의 상관관계를 추정하는 학습 기반 모델을 사용하여 초기화를 수행합니다. Thin Plate Spline (TPS) 변형을 사용하여 초기 스캔을 설정하고, 두 가지 기존의 비탄력 등록 방법을 비교합니다.

- **Technical Details**: 연구에 사용된 데이터셋은 highly consistent organ-at-risk (OAR) segmentation이 포함된 31개의 두경부 CT 스캔으로 구성됩니다. 기존의 NRR 방법인 NiftyReg(B-spline iterative optimization)와 Voxelmorph(DL 기반 접근법)와 비교하여, 새로운 해부학적 상관관계 기반 초기화 방법을 통해 등록 성능을 개선했습니다. 제안된 초깃값 설정 방법은 CorrTPS로 지칭됩니다.

- **Performance Highlights**: 등록 성능 평가는 TPS의 적용 여부에 따라 이루어졌으며, 초기화가 이루어진 경우 대부분의 구조에서 현저한 성능 개선을 보였습니다. 특정 구조에 대해서는 평균 거리 차이(mean distance-to-agreement, mDTA)가 1.8mm 감소하였고, 속도 이점 또한 상당하여, FPS에서 5초로 처리되는 것을 확인했습니다.



### EndoMamba: An Efficient Foundation Model for Endoscopic Videos (https://arxiv.org/abs/2502.19090)
- **What's New**: EndoMamba는 최소 침습 수술에서 자기 감독 학습(self-supervised learning)을 활용하여 효율적으로 spatiotemporal 표현을 캡처하도록 설계된 기초 모델로, 실시간 추론(real-time inference)을 지원합니다. 이 모델은 Bidirectional Mamba 블록과 vanilla Mamba 블록을 통합하여 각각 공간 모델링(spatial modeling)과 시간적 추론을 향상시킵니다. EndoMamba는 endoscopic video의 한계를 극복하고, 기존의 모델들보다 데이터 효율적인 학습을 가능하게 합니다.

- **Technical Details**: EndoMamba는 깊이 있는 상태 공간 모델(Deep State Space Models, SSMs)을 기반으로 하며, 과거와 현재의 데이터 흐름을 처리하기 위한 bidirectional 접근 방식을 취합니다. 이 구조는 공간 및 시간에 대한 강력한 추론을 가능하게 하여, 실시간 비디오 스트림에서의 효율성을 극대화합니다. 또한, 저수준 비디오 재구성(low-level video reconstruction)과 고수준 특징 정렬(high-level feature alignment)을 결합한 계층적 자기 감독 기법을 사용하여 representation learning 능력을 향상시킵니다.

- **Performance Highlights**: EndoMamba는 네 가지 하위 작업인 분류(classification), 분할(segmentation), 외과 단계 인식(surgical phase recognition), 및 위치 확인(localization)에서 기존 모델을 초월하는 성능을 보였습니다. 예를 들어, 분할 Dice 점수가 11.5% 증가하고, 외과 단계 인식 정확도가 21.3% 향상되었으며, 추론 속도가 초당 9.2 프레임(FPS)에서 46.7 FPS로 증가했습니다. 이는 EndoMamba가 의료 분야에서 실시간 지원의 잠재력을 극대화 할 수 있음을 시사합니다.



### Dynamic Degradation Decomposition Network for All-in-One Image Restoration (https://arxiv.org/abs/2502.19068)
- **What's New**: 이번 논문에서는 다중 유형의 손풍을 처리하기 위해 설계된 새로운 동적 손풍 분해 네트워크인 D$^3$Net을 소개합니다. D$^3$Net은 주파수 도메인과 공간 도메인 간의 심층 상호작용을 통해 각기 다른 손풍 유형을 파악하고, 교정 프롬프트를 생성하여 이미지 복원을 안내합니다. 이를 통해 다양한 손풍 상황에 적응할 수 있는 유연하고 확장 가능한 네트워크 아키텍처를 제시하여, 기존의 단일 모델이 가진 한계를 극복하고자 합니다.

- **Technical Details**: D$^3$Net은 Cross-Domain Degradation Analyzer (CDDA)와 Dynamic Decomposition Mechanism (DDM)의 두 가지 주요 구성 요소로 이루어져 있습니다. CDDA는 주파수 도메인의 손풍 특성과 공간 도메인 이미지 특성 간의 상호작용을 통해 손풍 교정 프롬프트를 생성합니다. DDM은 이러한 프롬프트를 사용하여 손풍 특징을 단계적으로 분해하고, 네트워크가 다양한 처리를 동적으로 조정할 수 있게 설계되었습니다.

- **Performance Highlights**: 다양한 이미지 복원 작업에 대한 실험 결과, D$^3$Net은 기존의 최첨단 방법들보다 5.47dB와 3.30dB의 PSNR (Peak Signal-to-Noise Ratio) 개선을 보이며 뛰어난 성과를 나타냈습니다. 이러한 결과는 D$^3$Net의 유연성과 확장성이 높아지면서도 불필요한 계산 오버헤드를 효과적으로 줄일 수 있음을 시사합니다.



### An Improved 3D Skeletons UP-Fall Dataset: Enhancing Data Quality for Efficient Impact Fall Detection (https://arxiv.org/abs/2502.19048)
Comments:
          17th International Conference on Machine Vision (ICMV 2024) will take place in Edinburgh, UK during October 10-13, 2024

- **What's New**: 이번 연구는 노인 돌봄 분야의 낙상 감지 시스템에서 중요한 역할을 하는 UP-Fall 데이터셋의 향상된 버전을 제시합니다. 향상된 데이터셋은 3D skeleton data를 통합하여 낙상 시 충격을 감지하는데 필요한 데이터 정확성 및 포괄성을 개선합니다.  이 연구의 결과는 낙상이 실제로 발생했음을 효과적으로 식별하는데 기여할 것으로 기대됩니다.

- **Technical Details**: 기존 UP-Fall 데이터셋은 비충격 이벤트와 실제 낙상의 구별이 어렵다는 문제를 가지고 있습니다. 이 연구에서 사용된 전처리 기술은 데이터의 정확성을 높이고, 3D skeletons의 정보를 추가하여 더욱 신뢰할 수 있는 충격 낙상 감지 시스템을 구축합니다. 머신러닝 및 딥러닝 알고리즘을 이용한 실험을 통해 향상된 데이터셋의 성능을 평가하였습니다.

- **Performance Highlights**: 개선된 3D skeletons 데이터셋을 기반으로 훈련된 낙상 감지 모델의 성능이 상당히 향상됨을 보였습니다. 이 연구는 낙상 위험이 있는 노인 인구의 안전과 복지를 개선하는 데 중점을 두고 있습니다. 향후 연구와 개발을 지원하기 위해 개선된 데이터셋은 공개적으로 제공됩니다.



### A Dual-Purpose Framework for Backdoor Defense and Backdoor Amplification in Diffusion Models (https://arxiv.org/abs/2502.19047)
- **What's New**: 최근의 확산 모델(difussion models)은 복잡한 생성 작업에서 뛰어난 성능을 보이며, 멀티모달 샘플 생성을 위한 최신의 생성 프레임워크로 부상하였습니다. 그러나 이러한 모델이 백도어 공격(backdoor attacks)에 취약하다는 점이 밝혀졌습니다. 본 논문에서는 PureDiffusion이라는 이중 목적을 가진 프레임워크를 제안합니다. 이 프레임워크는 백도어 방어(backdoor defense) 및 백도어 공격 증폭(backdoor attack amplification)이라는 두 가지 역할을 동시에 수행합니다.

- **Technical Details**: PureDiffusion은 두 가지 핵심 단계로 구성됩니다: 트리거 반전(trigger inversion)과 백도어 탐지(backdoor detection)입니다. 트리거 반전 메커니즘은 '트리거 이동(trigger shift)' 및 '디노이징 일관성 효과(denoising consistency effect)'라는 두 가지 중요한 백도어 흔적을 활용합니다. 이를 통해 백도어가 사용된 확산 모델에서 생성된 출력과 입력 분포를 비교하여 백도어 공격 여부를 평가합니다. 이 논문에서는 특히 다수의 디노이징 단계에서 트리거 이동을 계산하는 방법을 제안하여 트리거 반전의 강건성을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, PureDiffusion은 거의 완벽한 탐지 정확도를 달성하며, 기존 방어 기제들보다 훨씬 높은 성능을 보입니다. 또한 공격 시나리오에서는 공격 증폭 접근 방식이 기존 백도어 공격의 성공률(ASR)을 거의 100%로 증가시키면서 훈련 시간을 최대 20배까지 단축시킵니다. 이러한 성과는 제안된 방법이 복잡한 트리거 패턴에 대해 효과적으로 작동함을 보여줍니다.



### FungalZSL: Zero-Shot Fungal Classification with Image Captioning Using a Synthetic Data Approach (https://arxiv.org/abs/2502.19038)
Comments:
          11 pages, 5 Figures, 1 Table

- **What's New**: 이 논문에서는 균류(fungi) 관련 작업을 위한 CLIP의 제로샷 분류(zero-shot classification) 능력을 향상시키기 위해 두 가지 보완 데이터 소스를 소개합니다. 첫 번째는 대형 언어 모델(LLMs)을 사용하여 생성한 균류 성장 단계에 대한 기술적 설명이고, 두 번째는 다양한 합성 균류 이미지로 구성된 데이터셋입니다. 이러한 데이터셋은 텍스트와 이미지 데이터 간의 효과적인 정렬을 보장하기 위해 CLIP의 공유 표현 공간에 투사됩니다.

- **Technical Details**: 이 연구에서는 LLaMA3.2를 사용하여 텍스트를 생성하고 합성 이미지를 만들어 균류 성장 단계 간의 모달리티 간의 격차를 연결합니다. 우리는 CLIP이 다른 LLM 기술을 통해 생성된 텍스트 출력을 비교하여 분류를 개선할 수 있는지 조사합니다. 제안된 방법은 CLIP이 생성된 텍스트 설명을 기반으로 새로운 균류 클래스를 인식하는 능력을 평가하는 제로샷 분류에 의해 검증됩니다.

- **Performance Highlights**: 이 연구는 기존 모델들이 미세한 카테고리를 인식하는 데 한계가 있음을 이해하고, 새로운 VLM 프레임워크를 제안하여 균류 성장 단계를 포함한 제로샷 분류 성능을 향상시키는데 기여합니다. 다양한 LLMs를 사용하여 성장 단계에 대한 텍스트 설명을 생성한 결과, 제로샷 분류의 성능이 개선되었습니다. 실험 결과는 정량적인 지표와 시각적 검사를 통해 평가되었습니다.



### Enhanced Neuromorphic Semantic Segmentation Latency through Stream Even (https://arxiv.org/abs/2502.18982)
- **What's New**: 이 논문은 UAV와 자율주행차와 같은 실시간 시스템에서 최적의 의미적 분할(semantic segmentation)을 달성하기 위한 새로운 접근 방식을 제안합니다. 전통적인 프레임 기반 방법의 한계를 극복하기 위해, 이벤트 기반 카메라로부터의 이벤트 스트림을 활용하여 지연(latency), 정확성(accuracy), 에너지 효율성을 동시에 개선하고자 합니다. 특히, 스파이킹 신경망(Spiking Neural Network, SNN)을 사용하여 의미적 분할 작업을 실행하며, 이를 통해 낮은 전력 소비와 빠른 처리를 실현했습니다.

- **Technical Details**: 이벤트 기반 카메라는 마이크로초 단위의 뛰어난 시간 해상도와 100mW 이하의 효율적인 전력 소비를 제공하는 혁신적인 비전 센서입니다. 이들은 액티브 프레임 카메라와는 달리 비동기적으로 작동하며, 픽셀 밝기 변화를 감지하여 이벤트 스트림을 생성합니다. 논문에서는 SNN을 활용하여 이벤트를 처리하는 방법을 설명하며, 스파이크가 정보 단위로 작용하고, 신경망의 상태를 나타내는 막전압(membrane potential)의 변화를 통해 에너지 효율성과 지연 시간을 줄이는 방법을 강조합니다.

- **Performance Highlights**: DSEC 데이터셋을 활용한 실험 결과, 제안된 접근 방식은 지연 시간을 현저하게 줄이면서도 정확성의 감소는 최소화되었습니다. SNN을 사용함으로써 낮은 전력 소비를 달성하여 에너지 제약이 있는 실시간 애플리케이션에 적합성을 입증했습니다. 이 논문은 이벤트 스트림을 활용하여 의미적 분할을 향상시키는 첫 번째 사례로, 지연 시간의 감소, 정확성 손실의 최소화 및 에너지 효율성을 잘 균형 잡고 있습니다.



### Brain-inspired analogical mixture prototypes for few-shot class-incremental learning (https://arxiv.org/abs/2502.18923)
Comments:
          under review

- **What's New**: 이번 연구에서는 Brain-Inspired Analogical Mixture Prototypes (BAMP)라는 새로운 접근법을 제안합니다. BAMP는 혼합 전형(feature) 학습, 통계적 유추(statistical analogy), 소프트 투표(soft voting)의 세 가지 구성 요소로 이루어져 있습니다. 이 방법은 제한된 데이터로부터 효과적으로 학습하면서도 이전에 학습한 작업에 대한 지식을 잃지 않도록 설계되었습니다. 실험 결과 BAMP는 전통적인 FSCIL 설정 및 도전적인 소규모 시작 설정에서 최신 기술보다 우수한 성능을 발휘했습니다.

- **Technical Details**: BAMP는 미리 훈련된 모델인 Vision Transformer (ViT)를 기초로 하여, 각 클래스를 혼합된 전형으로 표현하고 이를 기본 세션 동안 조정합니다. 통계적 유추는 새로운 클래스에 대한 전형의 평균 및 공분산 행렬을 조정하여 유사성을 기반으로 클래스를 분류합니다. 또한 Mahalanobis 거리(Mahalanobis distance)를 사용하여 분류 점수를 계산합니다. 이러한 방식은 범주화 및 유추 학습을 따라 뇌의 메커니즘에 영감을 받아 설계되었습니다.

- **Performance Highlights**: BAMP는 CIFAR100, CUB200, EuroSAT, FGVCAircraft, Resisc-45 및 StanfordCars와 같은 6개의 벤치마크 데이터셋에서 포괄적인 실험을 수행하여 최신 성능을 입증하였습니다. 기존의 메소드들에 비해 분류 정확도에서 뛰어난 성과를 보였습니다. 이 연구는 BAMP가 FSCIL에서 재앙적 망각(catastrophic forgetting)과 오버피팅(overfitting) 문제를 완화할 수 있음을 입증하였습니다.



### Inscanner: Dual-Phase Detection and Classification of Auxiliary Insulation Using YOLOv8 Models (https://arxiv.org/abs/2502.18871)
- **What's New**: 본 연구는 구조적 구성 요소 내 보조 단열(auxiliary insulation)을 탐지하고 분류하기 위한 2단계 방법론을 제안합니다. 탐지 단계에서는 YOLOv8x 모델을 사용해 구조 청사진의 단열 영역을 정확하게 식별하며, 분류 단계에서 탐지된 단열 패치를 두 개의 클래스(존재 및 결손)로 나누어 분류합니다. 이 접근 방법은 자동화된 단열 탐지 및 분류의 효과성을 입증하며, 산업 환경 내 안전 기준과 품질 보증을 향상시킬 기반을 마련합니다.

- **Technical Details**: YOLOv8x는 최신 물체 탐지 모델로, 청사진에서 단열 영역을 탐지하는 데 사용됩니다. 탐지된 단열 구성 요소는 YOLOv8x-CLS 모델에 의해 완전한 존재 여부를 분류하며, 이를 통해 82%의 mAP(mean Average Precision) 점수와 98%의 정확성을 달성했습니다. 데이터셋 준비 과정에서는 주석(annotation), 증강(augmentation), 및 단열 지역의 적절한 크롭(cropping) 등 전처리 단계가 포함되었습니다.

- **Performance Highlights**: 제안된 방법론은 고도로 선별된 데이터세트를 평가하여 효과성을 입증했습니다. 다양한 산업 청사진을 테스트하며 10명 이상의 전문가와 협력하여 모델 출력을 검토하고 교차 검증하는 과정을 거쳤습니다. 이와 같은 다각적인 검증은 기존 수작업 공정에 비해 더욱 신뢰성 높은 단열 결함 탐지를 가능하게 하였습니다.



### Enhanced Transformer-Based Tracking for Skiing Events: Overcoming Multi-Camera Challenges, Scale Variations and Rapid Motion -- SkiTB Visual Tracking Challenge 2025 (https://arxiv.org/abs/2502.18867)
- **What's New**: 본 논문에서는 스키어 추적 성능을 향상시키기 위한 새로운 접근 방식으로 STARK(Spatio-Temporal Transformer Network for Visual Tracking) 모델을 제안합니다. 기존의 전통적인 추적 방법이 다루기 힘든 카메라 이동, 차단(sudden occlusions), 동적 운동에 대한 문제를 해결하기 위해 이 모델의 구조 및 하이퍼파라미터를 최적화했습니다. 또한, STARK 모델을 특정 도메인에 적합하게 조정하여 스키어 추적의 효율성을 개선합니다.

- **Technical Details**: 저자들은 SkiTB 데이터셋을 기반으로 STARK 모델의 추적 파이프라인을 구성합니다. 초기 및 동적 템플릿을 생성하고, 예측된 바운딩 박스를 기반으로 검색 영역을 설정하여 스키어를 추적합니다. 새로운 적응형 업데이트 메커니즘을 도입하여 신뢰도 점수(confidence scores)가 특정 임계치를 초과할 때 템플릿을 업데이트하며, 불확실한 프레임에서는 재시도(retry) 메커니즘을 통해 검색 영역을 확장하여 대상 회복(target recovery)을 도모합니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 기존 STARK 모델에 비해 더 나은 추적 성능을 보여주었습니다. STARK-ski-ours 모델은 스키어의 위치를 보다 정확하게 인식하고, 다양한 조건에서도 안정성을 유지하는 성과를 거두었습니다. 이러한 성과는 하이퍼파라미터 조정과 템플릿 업데이트 전략의 효과 덕분입니다.



### Sherlock: Towards Multi-scene Video Abnormal Event Extraction and Localization via a Global-local Spatial-sensitive LLM (https://arxiv.org/abs/2502.18863)
- **What's New**: 본 논문에서는 Video Anomaly Detection (VAD) 관련 기존 연구들이 비디오 프레임이 비정상인지 여부만을 감지하는 데 집중하고 있는 것을 지적합니다. 이에 따라 새로운 멀티-장면 비디오 비정상 사건 추출 및 로컬화(M-VAE) 작업을 제안합니다. 이는 비정상적인 사건의 쿼드러플(네 가지 요소)을 추출하고 이러한 사건을 로컬화하는 것을 목표로 합니다.

- **Technical Details**: M-VAE 작업은 전역-로컬 공간 모델링(global-local spatial modeling)과 전역-로컬 공간 균형(global-local spatial balancing)이라는 두 가지 주요 과제가 있다고 언급됩니다. 이를 해결하기 위해 Sherlock이라는 글로벌-로컬 공간에 민감한 대형 언어 모델(LLM)을 제안합니다. 이 모델은 Global-local Spatial-enhanced MoE (GSM) 모듈과 Spatial Imbalance Regulator (SIR)를 통해 두 가지 과제를 각각 해결합니다.

- **Performance Highlights**: 다양한 실험 결과, Sherlock 모델은 여러 첨단 Video-LLMs와 비교했을 때 현저한 이점을 보였습니다. 이는 M-VAE 작업에서 전역-로컬 공간 정보의 중요성을 입증하며, Sherlock이 이러한 정보를 효과적으로 캡처할 수 있음을 보여줍니다.



### BarkXAI: A Lightweight Post-Hoc Explainable Method for Tree Species Classification with Quantifiable Concepts (https://arxiv.org/abs/2502.18844)
- **What's New**: 이번 연구에서는 나무의 껍질 이미지 분류를 위한 시각적 모델 해석을 위한 경량화된 신후처리 방법인 BarkXAI를 제안합니다. 기존의 Explainable AI (XAI) 방법론이 한정적인 지역적 특징을 기반으로 한 설명에 그쳤던 것과 달리, 우리는 전 세계적인 시각적 특징을 quantifiable 개념을 사용하는 방식으로 설명하는데 초점을 두었습니다. 이를 통해 계산 비용을 줄이고 복잡한 개념을 수량화할 수 있게 됩니다.

- **Technical Details**: BarkXAI는 섬세하게 조정된 개념을 통해 나무 껍질 이미지와 같은 질감 기반 이미지 분류기를 해석하는 새로운 접근 방식을 제공합니다. 기존의 LIME이나 TCAV과는 달리, 외부 데이터셋에 의존하지 않고 파라미터화된 연산자를 사용함으로써 효율적인 개념 평가가 가능합니다. global visual features(전역 시각 특징)를 바탕으로 나무껍질 이미지의 smoothness(부드러움)나 tone(톤)과 같은 특성을 평가합니다.

- **Performance Highlights**: 실험 결과, 우리의 접근 방식은 TCAV와 Llama3.2 대비 개념 중요도 순위 평가에서 더 높은 성능을 보여주었으며, 이는 인간의 지각과의 우수한 정렬을 강조합니다. 이 연구는 나무의 껍질 이미지에서 전 세계적인 시각적 특징에 대한 개념 기반 설명을 제공한 첫 번째 연구로서, 향후 나무 품종 식별에 대한 해석 가능성을 크게 향상시킬 것으로 기대됩니다.



### Grad-ECLIP: Gradient-based Visual and Textual Explanations for CLIP (https://arxiv.org/abs/2502.18816)
- **What's New**: 본 논문에서는 CLIP의 해석에 대한 주목이 부족한 가운데, Gradient-based visual and textual Explanation 방법인 Grad-ECLIP을 제안합니다. Grad-ECLIP은 특정 이미지-텍스트 쌍에 대한 CLIP의 매칭 결과를 해석하는데 기여하며, 중간 공간 특징과 매칭 유사성 간의 관계를 분석하여 효과적인 heat maps을 생성합니다. 이 방법은 기존의 Transformer 해석 방법과는 달리 CLIP의 특성을 더 잘 반영합니다.

- **Technical Details**: Grad-ECLIP은 이미지-텍스트 매칭 점수의 Gradient를 이용하여 특징 채널에 대한 중요도를 산출하고, Loosened attention map을 통해 공간적 중요성을 계산합니다. 이는 CLIP의 인코더 아키텍처를 활용하여 각 입력 이미지의 특정 영역이나 단어가 CLIP 결과에 미치는 영향을 분석할 수 있게 합니다. Grad-ECLIP은 결과에 따라 이미지를 대상으로 한 시각적 설명과 텍스트를 대상으로 한 텍스트적 설명을 모두 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과 Grad-ECLIP은 기존의 다양한 해석 방법과 비교했을 때 더 뛰어난 성능을 보여주었습니다. 제안된 방법은 여러 도메인의 데이터셋에 적용할 수 있는 일반화 능력을 가지며, ViT와 CNN 기반 CLIP, ViT 분류기 및 BLIP와 같은 다른 Transformer 기반의 비전-언어 모델에서도 효과적입니다. 또한, Grad-ECLIP을 활용한 세밀한 정렬을 통한 CLIP의 고급 이해 능력을 향상시키는 응용 프로그램도 제안되었습니다.



### Spectral-Enhanced Transformers: Leveraging Large-Scale Pretrained Models for Hyperspectral Object Tracking (https://arxiv.org/abs/2502.18748)
Comments:
          Accepted to 14th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS)

- **What's New**: 본 논문은 스냅샷 모자이크 카메라를 활용한 하이퍼스펙트럴 객체 추적에 대한 새롭고 효과적인 방법론을 제안합니다. 대규모로 사전 훈련된 transformer 기반 모델을 하이퍼스펙트럴 데이터에 적응시켜, 공간-스펙트럴 특성을 동시에 학습할 수 있는 모듈을 개발했습니다. 특히, 다양한 센서 모달리티에 걸쳐 효과적인 학습을 지원하는 교차 모달리티 훈련 파이프라인을 도입하였습니다.

- **Technical Details**: 하이퍼스펙트럴 객체 추적을 위해 transformer 기반의 파이프라인을 제안하며, 스페컬 개념을 이용한 transformer 모델을 통해 RGB 이미지에서 학습된 사전 훈련 가중치를 활용합니다. 하이퍼스펙트럴 이미지에서 높은 차원의 스펙트럴 정보를 적응적으로 처리하기 위해 learnable spatial-spectral token fusion 모듈을 통합하였습니다. 패치 임베딩(tokenization) 과정을 통해 공간과 스펙트럴 정보를 각각 추출하며, 이를 통해 뛰어난 성능을 달성할 수 있음을 보였습니다.

- **Performance Highlights**: 제안한 모델은 최소한의 훈련 횟수로도 우수한 성능을 발휘하며, 하이퍼스펙트럴 데이터셋에서 다양하게 실험되어 강력한 성능을 확인하였습니다. 학습 과정에서 일부 스펙트럴 밴드가 누락되더라도 견고한 성능을 유지하여, 다수의 하이퍼스펙트럴 모달리티에 걸쳐 교차 모달 정보 학습이 가능하다는 점이 특징입니다. 이는 하이퍼스펙트럴 데이터에 대한 기존의 방법론보다 월등한 성능 향상을 보여줍니다.



### Beyond RNNs: Benchmarking Attention-Based Image Captioning Models (https://arxiv.org/abs/2502.18734)
Comments:
          10 pages, 6 figures. Code and additional results are available on GitHub under the handle HemanthTejaY

- **What's New**: 이번 연구에서는 이미지 캡셔닝(image captioning) 분야에서 주목(attention) 메커니즘의 효과를 평가합니다. 전통적인 RNN 모델에 비해 주목 기반의 모델이 MS-COCO 데이터셋에서 더욱 정확하고 의미 있는 캡션을 생성하는 모습을 보였습니다. 연구 결과, 주목 메커니즘이 이미지와 생성된 캡션 간의 정렬을 향상시키는 데 기여함을 밝혀냈습니다.

- **Technical Details**: 이 연구는 두 종류의 딥러닝 모델을 비교하며, CNN(Convolutional Neural Network)은 이미지 피쳐를 추출하는 인코더 역할을 하며, RNN(Recurrent Neural Network)은 세부 정보를 텍스트로 변환하는 디코더 역할을 수행합니다. 주목 기법으로는 Bahdanau Attention을 사용하여 인코더와 디코더 간의 정렬 점수를 계산하고, 이 점수를 통해 각 단어가 어떤 이미지의 부분과 관련 있는지를 학습합니다.

- **Performance Highlights**: 실험 결과, 주목 기반 모델이 RNN 기반 모델보다 더 정확하고 의미 있는 캡션을 생성하며, 인간 평가와의 정렬에서도 더 나은 성과를 나타냈습니다. 평가 지표로는 BLEU, METEOR, GLEU 및 WER를 사용하여 모델 성능을 검증했습니다.



### Adversarial Universal Stickers: Universal Perturbation Attacks on Traffic Sign using Stickers (https://arxiv.org/abs/2502.18724)
- **What's New**: 이 논문은 교통 신호와 자율주행 시스템의 맥락에서 보편적 공격에 대한 새로운 접근 방식을 제시합니다. 저자들은 단일한 공격을 통해 모든 이미지에 적용 가능한 보편적 교란(Universal Perturbation)을 설계하여 깊은 신경망(Deep Neural Network) 모델이 교통 신호를 잘못 분류하게 만드는 방법을 개발했습니다. 이들은 간단한 흑백 스티커처럼 보이는 교란을 사용하여 잘못된 표지판 예측을 유도할 수 있는 방법을 도입합니다.

- **Technical Details**: 이 연구에서는 가상 실험 환경을 도입하여 실제 교통 신호를 물리적으로 수정하지 않고도 교란 이미지를 테스트할 수 있게 했습니다. 연구진은 Street View 이미지를 활용하여 공격을 평가하고, 교통 신호에 적용할 수 있는 지점을 찾는 데 필요한 정보를 제공합니다. 흑백 스티커를 사용해 신호의 동일 위치에 배치함으로써 여러 교통 신호를 일관되게 잘못 분류하는 결과를 얻었습니다.

- **Performance Highlights**: 실험 결과, 단일 위치에 배치된 흑백 스티커가 신호를 잘못 분류하도록 하는 데 성공률이 매우 높다는 것이 밝혀졌습니다. 신호 인식에 사용되는 모델에서 90%까지의 신뢰도를 기록하며, 이러한 간단한 공격이 자율주행 시스템의 보안에 실질적인 위험을 초래하는 것을 보여줍니다. 이는 현재의 방어 메커니즘이 보편적 공격을 대비하기에는 불충분하다는 사실을 강조합니다.



### Enhancing Image Classification with Augmentation: Data Augmentation Techniques for Improved Image Classification (https://arxiv.org/abs/2502.18691)
- **What's New**: 이번 연구에서는 Convolutional Neural Networks (CNNs)의 성능을 향상시키기 위해 11가지 데이터 증강(Data Augmentation) 기법을 제안하고 있습니다. 특히, 본 연구에서는 새로운 세 가지 방법인 Pairwise channel transfer, Novel occlusion approach, Novel masking approach를 소개하고 있습니다. 이러한 기법들은 모델의 일반화 능력을 증가시켜 작은 데이터셋에서의 오버피팅을 방지하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서 제시하는 첫 번째 데이터 증강 기법은 Pairwise channel transfer로, 랜덤으로 선택된 이미지의 RGB 채널 값을 다른 이미지에 전달하는 방법입니다. 두 번째는 Novel occlusion approach로, 이미지의 객체를 데이터셋에서 랜덤으로 선택된 다른 객체로 가리는 기법입니다. 세 번째는 Novel masking approach로, 수직, 수평, 체크무늬 및 원형 마스크를 사용해 이미지의 특정 부분을 가리는 방법입니다. 이러한 기법들을 통해 모델의 학습 능력을 향상시킬 수 있습니다.

- **Performance Highlights**: Caltech-101 데이터셋을 통해 다양한 데이터 증강 기법의 성능을 평가한 결과, 제안된 이미지 증강 기법의 조합이 기존 방법보다 더 효과적임을 확인하였습니다. 데이터셋의 변형을 통해 모델이 54,864에서 73,153 이미지로 확장됨에 따라 성능 향상이 나타났습니다. EfficientNet-B0 모델을 통해 이러한 증강 기법의 효능을 비교하고, 다양한 성능 측정 지표에서 유의미한 개선을 입증하였습니다.



### Diffusion Models for conditional MRI generation (https://arxiv.org/abs/2502.18620)
- **What's New**: 이 논문에서는 뇌 자기 공명 영상(MRI)을 생성하기 위한 Latent Diffusion Model (LDM)을 제안합니다. 이 모델은 병리학(건강, 신경교종, 경화증, 치매)과 획득 모드(T1w, T1ce, T2w, Flair, PD)를 기반으로 이미지를 생성할 수 있습니다. 생성된 이미지는 실제 이미지와 유사한 분포를 보여주며, 시각적 충실도와 다양성의 균형을 유지합니다. 또한 이 모델은 훈련 데이터에 없는 구성도 생성할 수 있는 외삽(extrapolation) 능력을 입증했습니다.

- **Technical Details**: LDM은 압축된 잠재 공간(latent space)에서 확산(diffusion)을 수행하여 이미지 생성을 최적화합니다. 이는 시각적 품질을 저하시킴 없이 계산 부담을 줄여줍니다. 이 모델은 Gauss 노이즈를 점진적으로 추가하여 각 시간 단계에서 노이즈가 있는 잠재 벡터를 생성하며, 여기서 디퓨전 과정은 두 단계로 나뉩니다: 전방 확산(forward diffusion) 및 역방향 절차(reverse process).

- **Performance Highlights**: 모델의 성능 평가는 Fréchet Inception Distance (FID) 및 Multi-Scale Structural Similarity Index (MS-SSIM) 메트릭을 사용하여 수행되었습니다. 결과는 실제 데이터에서 잘 나타나지 않는 병리 및 모드에 대한 이미지 생성에서 데이터의 다양성을 증가시키며, 진단 도구 개발에 기여할 수 있음을 나타냅니다. 전체적으로 이 연구는 임상 데이터 세트에서 샘플 수를 증가시키고, 환자의 개인 정보를 침해하지 않으면서 AI 모델의 평가를 할 수 있는 가능성을 보여줍니다.



### DeBUGCN -- Detecting Backdoors in CNNs Using Graph Convolutional Networks (https://arxiv.org/abs/2502.18592)
Comments:
          18 pages, 11 tables, 8 figures

- **What's New**: 본 논문에서는 Deep Neural Network (DNN)의 백도어 공격 탐지 파이프라인인 DeBUGCN을 제시합니다. 이는 그래프 컨볼루션 네트워크(Graph Convolution Networks, GCN)를 활용하여 백도어가 삽입된 모델을 식별하는 새로운 방법론입니다. GCN이 트로이잔 탐지에 사용된 것은 이 논문이 첫 번째 사례로, DNN의 구조를 그래프로 표현하여 신경망의 가중치 분석을 수행합니다.

- **Technical Details**: DNN 모델의 마지막 완전 연결 레이어에서 가중치 정보를 활용하여 그래프 구조를 생성한 후, GCN을 이진 분류기로 사용하여 해당 모델이 깨끗한지 혹은 악의적 행동이 포함되었는지를 판단합니다. GCN은 DNN의 계산 흐름을 잃지 않으면서, 빠르고 결정론적인 메시지 전달 알고리즘을 사용하여 모델 불변성을 보장합니다. 이는 다양한 CNN 구조에 대해 Robust하게 작동하며, 특정 CNN의 아키텍처에 대한 사전 정보 없이도 적용이 가능합니다.

- **Performance Highlights**: DeBUGCN의 성능을 평가하기 위해 MNIST 및 CIFAR-10 데이터셋에서 수백 개의 깨끗한 CNN 모델과 트로이잔 모델을 학습했습니다. 결과적으로 DeBUGCN은 가장 최신의 트로이잔 탐지 알고리즘들과 비교해도 더 빠르고 높은 정확성을 보여주었습니다. 이 방법은 실제 환경에서도 적용할 수 있도록 설계되었으며, TrojAI 데이터셋을 통해 높은 신뢰성을 입증하였습니다.



### Application of Attention Mechanism with Bidirectional Long Short-Term Memory (BiLSTM) and CNN for Human Conflict Detection using Computer Vision (https://arxiv.org/abs/2502.18555)
- **What's New**: 이 연구는 비디오에서의 폭력 행동 탐지를 향상시키기 위한 심층 학습 기법의 통합을 조사합니다. 특히 Attention Mechanism, Convolutional Neural Networks (CNNs) 및 Bidirectional Long Short-Term Memory (BiLSTM)를 활용하여 인간 충돌 자동 탐지의 정확도를 높이고자 합니다. 실험 결과, CNN과 BiLSTM, Attention Mechanism의 조합이 충돌 모니터링을 위한 유망한 솔루션을 제공한다고 보고하고 있습니다.

- **Technical Details**: 컴퓨터 비전은 현실 세계에서 시각 정보를 획득하고 처리하며 해석하는 방법을 개발하는 인공지능(AI)의 한 분야입니다. 이 연구에서는 CNNs가 이미지와 비디오로부터 계층적 공간 특성을 추출하는 데 얼마나 효과적인지를 확인합니다. 또한, LSTM과 BiLSTM 아키텍처를 사용하여 시퀀스 데이터에서 시간적 종속성을 모델링 하고, Attention Mechanism을 통해 입력의 중요한 부분에 더욱 집중할 수 있도록 하는 기법을 설명합니다.

- **Performance Highlights**: 이 연구의 실험들은 깊은 훈련 데이터 세트를 사용하여 모델의 평균 정확도를 6% 향상시킨 것을 보여줍니다. Data augmentation과 transfer learning이 더 효과적인 감시 시스템을 위한 기초로 작용한다는 점을 강조합니다. 또한, 기존 영상 해석 기법들과의 비교를 통해 새로운 접근 방식의 효과를 입증하고 있습니다.



### Multi-class Seismic Building Damage Assessment from InSAR Imagery using Quadratic Variational Causal Bayesian Inferenc (https://arxiv.org/abs/2502.18546)
Comments:
          Submitted to Remote Sensing and Environment

- **What's New**: 본 논문에서는 기존 방법들의 한계를 극복하기 위해 다중 클래스 변분 인과 베이지안 추론 프레임워크(multi-class variational causal Bayesian inference framework)를 제시합니다. 이 접근법은 퀘드라틱 변분 경계(quadratic variational bounds)를 활용하여 복잡한 피해 패턴을 수학적으로 엄격하게 근사화하고, 계산 효율성을 보장합니다. InSAR 관측치를 미국 지질 조사국(USGS) 지반 붕괴 모델 및 건물 취약성 함수와 통합하여, 건물 피해 신호를 효과적으로 분리할 수 있습니다.

- **Technical Details**: 본 연구는 여러 복잡한 역학적 신호가 빈번한 재해 지역의 건물 피해 정도를 평가하는 데 있어 큰 도전 과제를 해결합니다. 다중 클래스 피해 평가 시 파라미터 폭발(parameter explosion)로 인해 발생하는 계산 복잡성을 효율적으로 처리하기 위해, 국소 가지치기(local pruning) 전략을 구현하여 복잡성을 줄였습니다. 이러한 접근법은 재해 발생 후 빠른 처리 시간을 요구하는 비상 대응 상황에 적합합니다.

- **Performance Highlights**: 다섯 개의 주요 지진(아이티 2021, 푸에르토리코 2020, 자그레브 2020, 이탈리아 2016, 리지크레스트 2019)에 걸쳐 평가한 결과, 피해 분류 정확도(AUC)는 0.94-0.96으로 기존 방법들보다 최대 35.7% 향상된 것으로 나타났습니다. 모든 피해 카테고리에서 높은 정확도(AUC > 0.93)를 유지하면서도, 계산 오버헤드를 40% 이상 줄일 수 있었습니다.



### FilterRAG: Zero-Shot Informed Retrieval-Augmented Generation to Mitigate Hallucinations in VQA (https://arxiv.org/abs/2502.18536)
Comments:
          12 pages, 6 figures and 2 tables

- **What's New**: FilterRAG이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 BLIP-VQA와 Retrieval-Augmented Generation(RAG)을 통합하여 외부 지식 소스, 예를 들어 Wikipedia와 DBpedia에서 답변의 기반을 확보합니다. 이를 통해 모델이 생성하는 오류 답변, 즉 hallucinations를 줄이고, 현실 세계에 적합한 VQA 시스템을 개선할 수 있는 잠재력을 강조합니다.

- **Technical Details**: FilterRAG는 입력 이미지를 2x2 그리드로 나누고, BLIP-VQA를 통해 시각적 및 텍스트 임베딩을 생성합니다. 이후 관련 지식을 동적으로 검색하여 답변 생성 과정에 통합하며, 이 과정에서 고정된 GPT-Neo 1.3B 모델을 사용합니다. 이러한 방식은 VQA 성능을 향상시키고, OOD(Out-of-Distribution) 시나리오에서 더욱 신뢰할 수 있는 결과를 제공합니다.

- **Performance Highlights**: OK-VQA 데이터셋에서 FilterRAG는 36.5%의 정확도를 달성하였으며, 이는 기존 모델 대비 hallucinations를 현저히 줄였습니다. 필터RAG는 인도메인 및 OOD 설정 모두에서 일관된 성능을 보였으며, 이는 효과적인 지식 검색과 다중 모달 정렬이 VQA의 견고함을 향상시키는 데 중요하다는 것을 보여줍니다.



### Convolutional neural networks for mineral prospecting through alteration mapping with remote sensing data (https://arxiv.org/abs/2502.18533)
- **What's New**: 이 연구는 전통적인 지질학적 지도 제작 방법의 비효율성을 극복하기 위해 CNN(Convolutional Neural Networks)을 활용하여 호주 뉴사우스웨일스(Broken Hill) 북부의 변성 지대(alteration zones)를 맵핑하는 혁신적인 접근 방식을 제안합니다. 이전의 중재적 방법들은 지리적 연속성이 결여된 필드 관측으로 인해 제한되었던 한계를 인식하고, 원격 탐지 데이터(remote sensing data)를 통한 정밀한 매핑의 필요성을 강조합니다. 이 연구는 기존 기계 학습 방법들과 CNN의 성과를 비교하여 CNN의 우수한 성능을 입증하고 있습니다.

- **Technical Details**: 본 연구에서는 Landsat 8, Landsat 9 및 ASTER 데이터와 함께 CNN을 활용하여 필드 데이터를 기반으로 한 훈련을 수행하였습니다. 또한, PCA(Principal Component Analysis)와 같은 자동화된 접근 방식을 도입하여 변성 구역을 효과적으로 식별할 수 있었습니다. CNN을 사용한 지도 제작 접근 방식은 기존의 KNN, SVM 및 MLP 등의 전통 기계 학습 모델들과 비교 분석되며, 특히 지리적 패턴을 잡아내는 데 있어서 더 나은 성능을 보여주었습니다.

- **Performance Highlights**: 결과적으로, CNN은 전통적인 모델보다 변성 구역의 공간 패턴을 더 정확하게 포착하는 데 성공했습니다. Landsat 9 데이터는 CNN을 통해 가장 높은 정확도로 철산화물(iron oxide) 지역을 맵핑하며, ASTER 데이터는 가장 정교한 아르길릭(Argillic) 및 프로필리틱(propylitic) 변성 지도를 생성했습니다. 이러한 결과는 특히 섬세한 광물화(mineralisation) 관련 변화를 식별하는 데 있어서 CNN의 효과를 강조하고 있습니다.



### IMPROVE: Iterative Model Pipeline Refinement and Optimization Leveraging LLM Agents (https://arxiv.org/abs/2502.18530)
- **What's New**: 이번 연구에서는 Iterative Refinement라는 새로운 전략을 소개하며, 이를 통해 LLM(대형 언어 모델) 기반 기계 학습(ML) 파이프라인 디자인을 개선합니다. 기존 방법들이 전체 파이프라인을 한 번에 최적화하려고 시도하는 반면, Iterative Refinement는 한 번에 하나의 구성 요소에 초점을 맞추고 점진적으로 개선합니다. 이를 통해 보다 안정적이고 해석 가능한 모델 성능을 달성할 수 있습니다.

- **Technical Details**: IMPROVE라는 프레임워크는 Iterative Refinement 전략을 실현하여 객체 분류 파이프라인을 자동으로 설계하고 최적화합니다. 이 프레임워크는 데이터 전처리, 아키텍처 선택, 하이퍼파라미터 튜닝 등 핵심 작업을 효율적으로 관리하며, 데이터 세트를 입력으로 받아 모델을 훈련하고 반복적으로 개선합니다. 실험을 통해, IMPROVE는 제로샷 LLM 기반 접근법보다 일관되게 향상된 성능을 보여줍니다.

- **Performance Highlights**: IMPROVE는 CIFAR-10, TinyImageNet 및 다양한 Kaggle 경쟁 데이터셋을 포함한 여러 데이터셋에서 인간 수준의 성능에 근접하는 결과를 달성했습니다. 특히, 이 시스템은 컴퓨팅 효율성을 유지하면서도 대규모 이미지 분류 작업을 효과적으로 수행할 수 있습니다. Iterative Refinement는 LLM 기반 ML 자동화의 실용적인 전략으로 자리잡으며, ML 전문 지식이 없어도 고급 이미지 분류 모델을 개발할 수 있는 접근 가능한 도구로 확립되었습니다.



### Optimized Custom CNN for Real-Time Tomato Leaf Disease Detection (https://arxiv.org/abs/2502.18521)
- **What's New**: 본 연구는 방글라데시에서 토마토 질병을 조기 탐지하기 위한 자동화된 시스템을 구축하기 위해 Convolutional Neural Networks (CNNs) 기술을 사용하였습니다. 전통적인 수작업 검사 방법보다 정확도와 효율성 측면에서 우수한 성능을 나타냈습니다. custom CNN 모델은 95.2%의 인상적인 정확도를 달성하여 다른 모델들보다 월등한 성과를 보였습니다.

- **Technical Details**: 연구에서는 Brahmanbaria 지역에서 수집한 토마토 잎의 이미지 데이터를 기반으로 다양한 딥러닝 모델을 적용하여 성능 비교를 수행했습니다. YOLOv5, MobileNetV2, ResNet18 등 여러 모델과의 비교를 통해 custom CNN 모델이 가장 높은 정확도를 보였으며, 데이터는 다양한 조명, 각도, 배경 조건을 포함하여 다양한 형태로 구성되었습니다. Hyperparameters를 최적화하여 모델의 정확성과 효율성을 개선하는 작업도 이루어졌습니다.

- **Performance Highlights**: custom CNN 모델은 95.2%의 높은 정확도로 토마토 잎 질병 탐지에 있어 최고의 성능을 기록하였습니다. 반면 다른 기존 모델들은 각각 77%, 89.38%, 71.88%의 정확도를 보였으며, 이는 custom CNN의 우수성을 강조합니다. 이러한 연구 결과는 딥러닝 기술이 토마토 작물의 조기 질병 탐지에 있어 매우 효과적임을 보여줍니다.



### FCoT-VL:Advancing Text-oriented Large Vision-Language Models with Efficient Visual Token Compression (https://arxiv.org/abs/2502.18512)
Comments:
          20 pages, 18 figures, 6 tables

- **What's New**: 이 논문에서는 고해상도 텍스트 지향 Large Vision-Language Models (VLLMs)의 시각적 토큰을 효율적으로 압축하는 새로운 프레임워크를 제안합니다. 기존의 훈련 없는 방법들이 높은 해상도에서 성능 저하를 겪는 문제를 해결하기 위해, 라이트웨이트(self-distillation) 사전 학습 단계와 고품질의 후 훈련 단계를 활용합니다. 이러한 접근 방식은 적은 수의 이미지-텍스트 쌍으로도 뛰어난 성능을 발휘할 수 있게 해줍니다.

- **Technical Details**: FCoT-VL 모델은 교사 모델(teacher model)과 학생 모델(student model)로 구성되어 있습니다. 교사 모델은 풍부한 시각적 토큰을 갖고, 학생 모델은 압축된 토큰 표현을 가집니다. 사전 학습 과정에서 교사 모델의 파라미터를 상속받고, 학생 모델의 토큰 압축 모듈만 조정함으로써 훈련 데이터와 GPU 자원의 제약을 극복할 수 있습니다.

- **Performance Highlights**: 실험 결과, FCoT-VL은 텍스트 지향 벤치마크에서 기존 모델에 비해 뛰어난 성능을 보이면서도 계산 오버헤드를 크게 줄였습니다. 본 논문에서 제안된 방법론은 InternVL2 모델에서 유효하며, 다양한 작업에서 우수한 적응 능력을 유지하는 것으로 나타났습니다. 또한, 모델과 코드는 곧 공개될 예정입니다.



### Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition (https://arxiv.org/abs/2502.18510)
Comments:
          AAAI-2025

- **What's New**: 본 논문은 Multi-Teacher Knowledge Distillation (KD)을 위해 강화 학습(RL)을 활용한 Multi-Teacher Knowledge Distillation with Reinforcement Learning (MTKD-RL) 방식을 제안합니다. MTKD-RL은 다양한 교사 역할을 최적화하는 것을 목표로 하며, 교사 성능과 교사-학생 간의 간극을 상태 정보로 설정합니다. 이 방법은 학생의 성능을 기반으로 교사의 가중치를 최적화하며, 효과적인 상호작용을 통해 보다 의미 있는 가중치를 생성합니다.

- **Technical Details**: MTKD-RL은 강화 학습을 활용하여 다중 교사의 지식 증류를 최적화합니다. 여기서는 학생 성능과 교사-학생 간극을 보상으로 사용하여 에이전트를 업데이트합니다. MTKD-RL의 핵심은 정책 기울기 알고리즘(policy gradient algorithm)을 적용하여 에이전트가 상태 정보를 기반으로 교사 가중치를 생성할 수 있도록 한다는 점입니다.

- **Performance Highlights**: 실험 결과, MTKD-RL은 이미지 분류, 물체 탐지, 의미 세분화 자원에서 기존의 다중 교사 KD 방법들 대비 뛰어난 성능을 보여주었습니다. 또한, MTKD-RL은 고급 기능 학습을 통해 밀집 예측 작업에서도 더 나은 결과를 도출하는 것으로 나타났습니다. 이를 통해 MTKD-RL이 시각 인식 작업에서 최첨단 성능을 달성함을 입증합니다.



### Physical Depth-aware Early Accident Anticipation: A Multi-dimensional Visual Feature Fusion Framework (https://arxiv.org/abs/2502.18496)
- **What's New**: 이번 연구에서는 사고 예측의 기초를 제공하기 위해 모니터링 카메라(dashcam)에서 비디오 데이터를 활용하는 새로운 물리적 깊이 인식 학습 프레임워크를 제안했습니다. 기존의 2D 이미지 공간에서의 교통 요인 간 상호 작용 모델링의 한계를 극복하기 위해, 다차원 시각적 특징을 활용하여 보다 세밀한 3D 공간 정보를 도입했습니다. 특히, 새로운 접근법은 Depth-Anything 모델을 통해 생성된 단안 깊이 특징을 포함하여 사고의 전조 신호를 분석합니다.

- **Technical Details**: 제안된 프레임워크는 교통 장면 내에서 시각적 깊이 특징을 추출하기 위해 Depth-Anything 인코더를 사용하고, 이와 함께 도시 역학 및 비주얼 상호작용 특징을 통합합니다. 이 과정에서 각 비디오 프레임을 노드로 하고, 다차원 시각적 특징을 통해 프레임 그래프를 구축하여 사고 예측을 위한 그래프 주의 네트워크를 학습합니다. 이 네트워크는 사고 예측의 조기 프레임에 집중하여 보다 시기적절한 사고 예측을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 공공 데이터셋에서 최첨단 성능을 달성하여 시각적 깊이 특징을 통합하는 것의 효과성을 입증합니다. 이 연구는 깊이 정보와 비주얼 특징을 통합하여 사고를 조기에 예측하는 데 있어 높은 정확성을 보였습니다. 또한, 이 프레임워크는 환경과의 상호작용을 보다 정밀하게 모델링함으로써 잘못된 경고를 줄이는 데 기여합니다.



### Event-based Solutions for Human-centered Applications: A Comprehensive Review (https://arxiv.org/abs/2502.18490)
- **What's New**: 본 논문에서는 이벤트 카메라(Event Cameras)의 인체 중심 응용 분야에 대한 포괄적인 분석을 제공하고 있습니다. 이 카메라는 비동기(asynchronous) 방식으로 빛의 강도 변화를 포착하여 고도의 시간 해상도와 에너지 효율성을 자랑합니다. 기존의 연구는 몸(body)과 얼굴(face) 분석에 대한 체계적인 개요가 부족했으나, 이 조사에서는 두 분야를 통합하여 최신의 발전, 도전 과제 및 기회를 제시합니다.

- **Technical Details**: 이벤트 카메라는 생물학적 신경망을 모방한 신경형 센서(Neuromorphic Sensors)입니다. 이 센서는 각 픽셀을 비동기적으로 독립적으로 작동시켜 빛의 강도 변화가 초과할 때에만 이벤트를 생성합니다. 선형 RGB 카메라의 일반적인 한계를 극복하며, 빠른 움직임을 포착하는 데 탁월한 능력을 가지고 있습니다.

- **Performance Highlights**: 이벤트 카메라는 다양한 인체 중심 응용 분야에서 점점 더 많이 사용되고 있으며, 저조도 및 고속 이동 같은 까다로운 조명 조건에서도 뛰어난 성능을 발휘합니다. 이 논문은 특히 인간의 행동 인식, 제스처 인식, 얼굴 분석 등에서 이 카메라의 우수한 이점을 강조하고 있습니다.



### TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding (https://arxiv.org/abs/2502.19400)
- **What's New**: 이 연구에서는 TheoremExplainAgent라는 새로운 에이전트 기반 시스템을 소개합니다. 이 시스템은 Manim 애니메이션을 사용하여 길이가 5분 이상의 정리 설명 비디오를 생성할 수 있는 능력을 가지고 있습니다. 연구자들은 240개의 정리에 대한 기준을 포함한 TheoremExplainBench라는 벤치마크를 개발하여 다중 모드 정리 설명을 평가하는 체계적인 방안을 제안합니다.

- **Technical Details**: TheoremExplainAgent는 비디오 생성의 기획, 내레이션, 그리고 Python 애니메이션 스크립트를 생성하는 코딩 에이전트로 구성되어 있습니다. 이 시스템은 4개의 STEM 분야에서 다양한 정리 비디오를 생성할 수 있는 능력을 보여주며, 생성된 설명은 사실 정확성과 지각 품질의 5가지 차원에서 평가됩니다. 연구 결과에 따르면 에이전트 기반 기획이 상세한 긴 형식의 비디오 생성에 필수적이며, o3-mini 모델은 93.8%의 성공률과 0.77의 전반적인 점수를 기록했습니다.

- **Performance Highlights**: TheoremExplainAgent는 최대 10분 길이의 확장된 비디오 설명을 생성할 수 있어 기존 에이전트 없는 방법에 비해 상당한 진전을 보여주었습니다. 하지만 생성된 애니메이션은 종종 시각적 레이아웃에 사소한 문제가 발생하여 미세한 정렬 오류나 겹치는 도형이 나타나는 경우가 많았습니다. 이러한 시각적 오류는 보다 복잡한 정리에서 더욱 두드러지게 나타나, AI 시스템의 깊은 추론 결함을 드러내는 데 도움이 되었습니다.



### Multi-modal Contrastive Learning for Tumor-specific Missing Modality Synthesis (https://arxiv.org/abs/2502.19390)
- **What's New**: 이번 연구에서는 다중 모달리티 자기 공명 영상(MRI)에서 누락된 모달리티 이미지를 생성하기 위한 새로운 생성 모델을 제안했습니다. 이 모델은 환자의 관절 운동 아티팩트, 시간 제약 및 높은 비용과 같은 문제를 해결하기 위해 다중 모달 대비 학습(multi-modal contrastive learning)을 통합하여, 종양 영역에 초점을 맞췄습니다. 또한, 이 접근법은 모달리티 이미지를 생성할 뿐만 아니라 동시적으로 세분화(segmentation) 결과를 예측하는 기능을 갖추고 있습니다.

- **Technical Details**: 제안된 모델은 다중 모달 번역 네트워크(multi-modal translation network)와 세분화 디코더(segmentation decoder)를 결합하여 다양한 출처 모달리티로부터 학습하는 능력을 강화합니다. 특히, 각 모달리티에 맞춘 다중 브랜치 인코더(multi-branch encoder)와 주목(attention) 모듈을 통해 입력 이미지의 구조적 정보를 집중적으로 학습합니다. 컨트라스트 학습(contrastive learning)과 추가적인 자기 표현 손실(self-representation loss)을 결합하여, 전체 과정에서 표적 정보(target-specific information)를 효과적으로 전달합니다.

- **Performance Highlights**: Brain MR Image Synthesis 챌린지에서 제안한 모델은 높은 품질의 누락된 모달리티 이미지를 생성하며, 효과적인 성능을 보여주었습니다. 이 연구는 기존의 GAN 기반 방법들을 확장하여 제공된 다중 출처 모달리티로부터 종양 영역을 정확하게 생성하는 데 중점을 두었습니다. 최종적으로, 이 새로운 접근법은 임상적 진단 및 치료의 정확성을 더욱 높이는 데 기여할 것으로 기대됩니다.



### Deep Learning-Based Transfer Learning for Classification of Cassava Diseas (https://arxiv.org/abs/2502.19351)
Comments:
          12 pages, in Portuguese language, 3 figures

- **What's New**: 이번 논문은 카사바(Cassava) 질병 이미지를 분류하기 위한 네 가지 Convolutional Neural Network (CNN) 아키텍처(EfficientNet-B3, InceptionV3, ResNet50, VGG16)의 성능을 비교합니다. 비대칭 데이터셋을 기반으로 적절한 메트릭(metrics)을 사용하여 클래스 불균형 문제를 해결했습니다. 연구 결과 EfficientNet-B3는 87.7%의 정확도(accuracy), 87.8%의 정밀도(precision), 87.8%의 재현율(recall) 및 87.7%의 F1-Score를 기록하며 디지털 농업 분야에 유용한 도구가 될 수 있음을 제안합니다.

- **Technical Details**: 이 논문에서는 CNN 아키텍처를 사용해 카사바 질병의 자동 감지를 위한 방법론을 제시합니다. 특히, Bacteriose da Mandioca (CBB), Doença da Estria Marrom (CBSD), Doença do Mosaico (CMD) 및 Vírus Mosqueado Verde (CGM) 네 가지 질병을 식별하기 위해 여러 CNN 아키텍처를 비교합니다. 모델은 주로 Deep Learning(딥러닝) 기법을 기반으로 하며, Transfer Learning(전이 학습) 기법도 사용하여 더욱 효율적인 결과를 도출하고자 합니다.

- **Performance Highlights**: EfficientNet-B3 아키텍처는 다른 CNN 모델들과 비교하여 가장 높은 성능을 보여주었으며, 데이터셋의 클래스 불균형 문제를 해결하기 위한 다양한 메트릭을 적용하여 보다 정확한 진단이 가능함을 입증했습니다. 실험에서 사용된 서버는 Intel i7 CPU와 NVIDIA GTX 1080 Ti GPU 두 개로 구성되어 있으며, Python 프로그래밍 언어와 PyTorch 프레임워크를 사용하여 CNN 모델을 훈련했습니다. 연구 결과는 카사바 질병의 조기 진단 및 효과적인 농업 관리에 기여할 것으로 기대됩니다.



### Consistent Amortized Clustering via Generative Flow Networks (https://arxiv.org/abs/2502.19337)
Comments:
          Accepted to AISTATS 2025 on January 21, 2025

- **What's New**: 이번 논문에서는 GFNCP라는 새로운 프레임워크를 제안합니다. GFNCP는 공유된 에너지 기반의 파라미터화 정책과 보상을 통해 구성된 Generative Flow Network로, 클러스터링에서 샘플링 효율성을 향상시킵니다. 기존의 Neural Clustering Process와 같은 방법이 데이터 순서에 의존하는 문제를 해결하기 위해, GFNCP는 순서 불변성을 보장합니다.

- **Technical Details**: GFNCP의 주요 기술적 특징은 흐름 매칭 조건(flow matching conditions)이 데이터 마지날화(marginalization) 하의 클러스터링 후방확률(posterior) 일관성과 동등하다는 점입니다. 이는 GFNCP가 클러스터 라벨을 샘플링할 때, 데이터의 제시 순서에 영향을 받지 않도록 합니다. 이러한 특성 덕분에 GFNCP는 기존 방법들보다 더 효과적인 클러스터링을 제공합니다.

- **Performance Highlights**: 실험 결과 GFNCP는 합성 데이터(synthetic data)와 실제 데이터(real-world data) 모두에서 기존의 클러스터링 방법들보다 뛰어난 성능을 보여주었습니다. 이는 GFNCP가 프로바빌리스틱 클러스터링(probabilistic clustering)에서의 새로운 가능성을 제시함을 의미하며, 향후 연구에 중요한 기여를 할 것으로 기대됩니다.



### Does 3D Gaussian Splatting Need Accurate Volumetric Rendering? (https://arxiv.org/abs/2502.19318)
Comments:
          To be published in Eurogrpahics 2025, code: this https URL

- **What's New**: 본 논문은 3D Gaussian Splatting(3DGS)의 근본적인 가정과 근사값을 깊이 있게 분석합니다. 3DGS는 기존 Neural Radiance Fields(NeRF)와 유사한 이미지 형성 모델을 사용하지만, 하이브리드 렌더링 솔루션을 적용하여 빠른 계산 속도를 자랑합니다. 특히, 이 논문에서는 3DGS에서의 오파시티(opacity)와 볼륨 렌더링에서의 소멸 함수(extinction function)의 차이를 명확하게 설명합니다.

- **Technical Details**: 3DGS는 볼륨 렌더링 이론에서 벗어나 여러 가지 단순화된 가정을 적용합니다. 이를 통해 3DGS는 다양한 가우시안 원시(primitive)로부터의 렌더링에서 우수한 성능을 보이며, 계산 요구도를 낮춥니다. 논문에서는 3DGS의 근사값이 전체 렌더링 품질에 미치는 영향을 분석하기 위해 extinction 기반의 스플래팅(splatting)과 레이 마칭(ray marching) 알고리즘을 소개합니다.

- **Performance Highlights**: 분석 결과, 적은 개수의 원시에서는 소멸 기반의 솔루션이 오파시티 기반의 스플래팅보다 더 나은 성능을 보였습니다. 그러나 원시의 수가 많아질수록, 오파시티 스플래팅이 우수한 결과를 보입니다. 또한, 3DGS는 간단한 글로벌 정렬(global sorting) 단계를 사용하여 가시성을 해결하여 렌더링 속도를 높입니다.



### Deep learning and classical computer vision techniques in medical image analysis: Case studies on brain MRI tissue segmentation, lung CT COPD registration, and skin lesion classification (https://arxiv.org/abs/2502.19258)
Comments:
          27 pages, 18 figures

- **What's New**: 이번 연구는 다양한 의료 이미징 태스크와 모달리티에서 분할(segmentation), 등록(registration), 분류(classification) 작업을 체계적으로 평가한 최초의 사례입니다. 특히, 뇌 MRI 조직 분할, 폐 CT 이미지 등록, 피부 병변 분류 등의 응용 분야에 있어 전통적인 방법과 딥러닝(deep learning) 방법의 통합적 강점을 보여주고 있습니다.

- **Technical Details**: 연구에서 3D 딥러닝 모델은 2D 및 패치 기반 모델보다 우수한 성능을 보였으며, 특히 nnU-Net 모델이 0.9397의 Dice 점수를 달성했습니다. 또한, 뇌 조직 분할에는 3D U-Net 모델이 경쟁력 있는 결과를 보였고, Elastix 기반의 전통적인 방법이 폐 CT 등록에서 가장 효과적인 결과를 보여주었습니다.

- **Performance Highlights**: 피부 병변 분류에서 InceptionResNetV2와 ResNet50의 앙상블 모형이 각각 90.44% 및 93.62%의 정확도를 기록하며 뛰어난 성능을 보였습니다. One-vs-All 방법을 적용한 딥러닝 모델은 멜라노마, 기저세포암, 편평세포암 분류에서 각각 94.64%, 95.35%, 96.93%의 높은 정확도를 달성하였습니다.



### ObjectVLA: End-to-End Open-World Object Manipulation Without Demonstration (https://arxiv.org/abs/2502.19250)
Comments:
          Project page at this https URL

- **What's New**: 이 논문에서는 Vision-Language-Action (VLA) 모델을 활용하여 로봇의 객체 일반화(object generalization)를 달성하는 새로운 방법인 ObjectVLA를 제안합니다. 기존 모형은 대량의 인간 시연 데이터를 필요로 하여 확장성과 실용성에 한계를 갖고 있습니다. ObjectVLA는 새로운 타겟 객체에 대해 명시적인 인간 시연 없이 배운 기술을 일반화합니다.

- **Technical Details**: ObjectVLA는 시각적 데이터와 텍스트 쌍을 활용하여 로봇이 이전에 보지 못한 객체를 인식하고 조작할 수 있도록 합니다. 이 모델은 로컬리제이션(localization) 메타데이터가 포함된 데이터셋을 구성하여 VLA 모델과 로봇 상호작용 데이터를 공동으로 미세 조정(fine-tuning)합니다. 이를 통해 시각 데이터와 언어 입력 간의 통합된 경로를 생성하여 제로샷(zero-shot) 객체 일반화를 가능하게 합니다.

- **Performance Highlights**: ObjectVLA는 실제 로봇 플랫폼에서 100개 이상의 새로운 객체에 대해 64%의 성공률로 일반화 능력을 입증했습니다. 실험 결과, 기존 데이터 세트에 포함되지 않은 객체에 적응할 수 있는 빠른 조정이 가능하며, 스마트폰으로 촬영한 이미지만으로도 모델을 새로운 객체에 맞게 조정할 수 있습니다. 이러한 특성은 대규모 인간 시연에 대한 의존도를 줄이면서 객체 수준의 일반화를 가능하게 합니다.



### Multi-level Attention-guided Graph Neural Network for Image Restoration (https://arxiv.org/abs/2502.19181)
- **What's New**: 최근 이미지 복원 분야에서 딥러닝이 눈부신 성공을 거두고 있지만, 기존의 많은 CNN 기반 방법들은 단일 스케일에 초점을 맞춰 다중 스케일 정보를 충분히 활용하지 못했습니다. 이 논문은 멀티 레벨 주의 기반 그래프 신경망(MAGN)을 제안하여, 특징 맵 내에서 지역 구조적 특징과 글로벌 표현 정보를 동시에 추출할 수 있도록 돕습니다. 이를 통해 복원 과정에서 지역 정보와 글로벌 정보를 상호 보완할 수 있는 구조를 갖추었습니다.

- **Technical Details**: MAGN은 멀티-어텐션 메커니즘을 사용하여 요소 블록 그래프와 요소 그래프를 명시적으로 구성합니다. 이 네트워크는 이미지 반복(input degradation) 과정에서 글로벌 정보를 효과적으로 추출하면서 지역 특징 블록 구조의 정보를 활용해 보완합니다. 즉, 네트워크 내의 그래프는 실시간으로 동적 연결을 학습하고, 그래프 컨볼루션 알고리즘을 통해 정보를 전파 및 집계합니다.

- **Performance Highlights**: 여러 가지 이미지 복원 작업에서 실험 결과, 제안된 MAGN 방법이 기존 방법들보다 우수한 성능을 발휘함을 확인했습니다. 특히, 감도가 높은 이미지 저하 상황에서도 정교한 이미지 복원이 가능하며, 다양한 데이터셋에서 그의 뛰어난 안정성과 최첨단 성능을 입증했습니다.



### RetinaRegen: A Hybrid Model for Readability and Detail Restoration in Fundus Images (https://arxiv.org/abs/2502.19153)
- **What's New**: 이번 연구에서는 RetinaRegen이라는 새로운 하이브리드 모델을 제안하여 망막 이미지의 품질을 향상시키고자 합니다. 이 모델은 읽기 가능성(classification)을 평가하는 모델, Diffusion Model, Variational Autoencoder(변분 오토인코더)를 통합하여 설계되었습니다. 이를 통해 실제 환경에서 발생할 수 있는 흐릿한 이미지 문제를 해결하려고 합니다.

- **Technical Details**: RetinaRegen 모델은 SynFundus-1M 데이터셋을 기반으로 실험을 수행하여, PSNR(피크 신호 대 잡음 비율) 27.4521, SSIM(구조적 유사성 지수) 0.9556, LPIPS(퍼ceptual 이미지 품질 손실) 0.1911의 성능을 달성했습니다. 이 모델의 핵심은 데이터의 특정 영역, 특히 optic disc(시신경 유두) 지역에서의 복원 능력을 강조하는 것입니다.

- **Performance Highlights**: 제안된 RetinaRegen 모델은 핵심 지역의 복원에서 뛰어난 성능을 보이며, 임상 진단을 지원하는 효과적인 솔루션으로 기능할 수 있음을 시사합니다. 이러한 결과는 망막 이미지 품질을 향상시키고, 이를 바탕으로 진단의 신뢰성을 높일 수 있는 가능성을 보여줍니다.



### From Traditional to Deep Learning Approaches in Whole Slide Image Registration: A Methodological Review (https://arxiv.org/abs/2502.19123)
- **What's New**: 이 논문은 전통적인 단일 또는 다단계 조직 슬라이드의 전체 슬라이드 이미지(WSI) 등록 기술을 다룹니다. 특히, 종양 미세환경(TME) 분석을 위한 이미지 정렬의 중요성을 강조하며, 현재의 접근 방법과 그 한계를 검토합니다. 딥 러닝(deep learning) 기반의 최신 방법론을 탐구하고 이 분야에서 경쟁력 있는 미래 연구 방향으로 이어지는 기회를 제시합니다.

- **Technical Details**: WSI 등록 과정은 여러 슬라이드, 스캐너, 또는 시간에 걸쳐 얻은 WSI 스캔을 정렬하고 결합하는 것입니다. 수학적 관점에서, 등록 문제는 움직이는 이미지 I_{m}을 기준 이미지 I_{r}에 최적으로 정렬하기 위한 변환을 찾는 것으로 정의됩니다. 이 과정은 다양한 이미지 품질, 이미징 모달리티, 노이즈 및 인공물과 같은 요소로 인해 복잡함을 겪고 있으며, 이러한 문제들을 해결하는 다양한 방법들을 논의합니다.

- **Performance Highlights**: WSI 등록의 성능 향상을 위해 자동 등록 알고리즘이 개발되고 있으며, 이로 인해 효율성과 정확성이 개선되고 있습니다. HE와 면역 조직 화학(IHC) 마커를 조합한 정량 분석은 질병의 진행과 치료 반응을 이해하는데 기여하고, 심지어 3D 재구성을 통한 복잡한 구조와 생물학적 샘플 내의 공간적 관계를 이해하는 데 도움을 줍니다. 최종적으로 이 논문은 WSI 등록의 다양한 응용을 제시하며, 그 중요성과 미래 연구 방향을 강조합니다.



### Max360IQ: Blind Omnidirectional Image Quality Assessment with Multi-axis Attention (https://arxiv.org/abs/2502.19046)
- **What's New**: 이번 연구에서는 새로운 블라인드 전방위 이미지 품질 평가 모델인 Max360IQ를 제안합니다. 이 모델은 비균일 왜곡과 균일 왜곡 모두의 품질을 효과적으로 측정할 수 있으며, 멀티-액시스 어텐션 모듈을 활용하여 글로벌 및 로컬 상호작용을 포착합니다. 또한, 제안된 방법은 다중 스케일 특성 통합(Multi-Scale Feature Integration) 모듈을 사용하여 다양한 스케일의 특성을 융합합니다.

- **Technical Details**: Max360IQ 모델은 여러 축을 통한 주의 모듈로 구성되며, 이는 전방위 이미지를 적절히 평가하기 위해 글로벌 및 로컬 스페이스 상호작용을 캡처합니다. GS-중복 구조를 기반으로 한 품질 회귀 모듈을 통해, 깊은 의미의 가이드를 이용하여 품질 예측이 이루어집니다. 이를 위해 게이트 순환 유닛(GRU)을 사용하여 비균일 왜곡을 처리합니다.

- **Performance Highlights**: 실험 결과에 따르면, Max360IQ는 비균일 왜곡이 있는 JUFE 데이터베이스에서 기존의 Assessor360보다 3.6% 개선된 SRCC(순위 상관 관계 계수)를 달성했습니다. OIQA 및 CVIQ 데이터베이스에서도 각각 0.4%와 0.8%의 성능 향상을 보여주며, 비균일 왜곡 처리 성능이 우수함을 입증하고 있습니다.



### PolypFlow: Reinforcing Polyp Segmentation with Flow-Driven Dynamics (https://arxiv.org/abs/2502.19037)
- **What's New**: 본 논문에서는 PolypFLow라는 흐름 매칭(Flow Matching) 강화 아키텍처를 소개합니다. 이 모델은 물리 기반 최적화 역학을 세그멘테이션 세부 조정에 주입하여, 예측의 불확실성 하에서도 세그멘테이션 자신감을 동적으로 모델링하는 데 중점을 둡니다. 기존의 계단식 네트워크와 달리, PolypFLow는 미세 조정을 위해 일반적인 미분 방정식(ODE)을 해석하고, 이를 통해 초기 예측을 실제 마스크와 점진적으로 정렬합니다.

- **Technical Details**: PolypFLow는 U-Net과 Flow Matching Equations (FME)를 통합하여 구축됩니다. 특히, 분산된 코사인 변환(Discrete Cosine Transform, DCT)과 자기 주의 메커니즘(self-attention)을 도입하여, 이미지의 주파수 영역과 글로벌 정보를 효과적으로 캡처합니다. 이러한 구조는 모델이 언더세그멘테이션(under-segmentation)된 영역을 수정하는 과정을 시각적으로 해석할 수 있게 하여, 최적화 과정의 해석 가능성을 제공합니다.

- **Performance Highlights**: PolypFLow는 다섯 개의 벤치마크 데이터셋을 통해 실험한 결과, 기존의 최첨단 방법들에 비해 우수한 성능을 보였습니다. 이 모델은 조명 변화에 따른 일관된 성능을 유지하며, 특히 저대비 영역과 모션 아티팩트에 대한 회복력을 향상시킵니다. 이로 인해, PolypFLow는 임상 환경에서도 높은 신뢰성을 지닌 세그멘테이션 솔루션으로 자리잡을 수 있습니다.



### InternVQA: Advancing Compressed Video QualityAssessment with Distilling Large Foundation Mod (https://arxiv.org/abs/2502.19026)
Comments:
          Accepted by ISCAS 2025(Lecture)

- **What's New**: 이 논문에서는 비디오 품질 평가(VQA)를 위해 강력한 비디오 표현 능력을 가진 InternVideo2를 활용하여 경량화된 모델을 개발했습니다. 기존의 대규모 비디오 모델이 자원 소모가 크기 때문에, Distillation 방법을 통해 compression 품질 정보를 효과적으로 전이하여 모델의 크기를 줄이면서 성능을 유지하였습니다. 실험 결과, 제안된 경량 모델이 기존 방법들보다 우수한 성능을 보였습니다.

- **Technical Details**: 제안한 방법은 knowledge distillation을 기반으로 하며, 두 가지 손실 함수를 활용하여 학생 모델이 교사 모델의 feature representation을 효과적으로 학습할 수 있도록 합니다. 구체적으로, ℒ2 Loss와 Smooth ℒ1 Loss를 사용하여 예측값과 진짜 값 사이의 오차를 최소화하며, 교사 모델과 학생 모델 간의 feature 정합성을 보장합니다. 이러한 방법을 통해 경량 모델은 비디오 품질 평가에서 compression 왜곡 문제를 보다 잘 처리할 수 있게 됩니다.

- **Performance Highlights**: 제안된 경량 모델은 두 가지 compression 품질 평가 데이터셋에서 기존의 모든 방법을 초월하는 성능을 입증했습니다. 동시에, 경량 모델은 원래의 대규모 모델과 비슷하거나 더 좋은 성능을 달성하여 효율성과 성능 간의 최적의 균형을 이뤘습니다. 결과적으로, 이 연구는 비디오 품질 평가 분야에서의 새로운 가능성을 열어주고 있습니다.



### Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments (https://arxiv.org/abs/2502.19024)
Comments:
          Accepted by ICRA 2025

- **What's New**: 최근 Vision-and-Language Navigation (VLN) 분야의 연구에서, 저자들은 낮은 시야각을 가진 사족보행 로봇과 인간 중심의 지시사항 간의 불일치 문제를 해결하기 위한 Ground-level Viewpoint Navigation (GVNav) 방안을 제안하고 있습니다. 이는 VLN에서 다양한 높이에서의 시각적 관찰의 일반화 격차를 강조한 최초의 시도입니다. 본 연구는 광범위한 실험을 통해 시뮬레이션 환경 및 실제 환경에서의 성능 향상을 입증하였습니다.

- **Technical Details**: 이 논문은 사족보행 로봇이 낮은 높이에서 데이터를 수집하는데 직면하는 여러 가지 문제를 다루고 있습니다. 저자들은 비슷한 특징에 적절한 가중치를 부여함으로써 지역적 관측의 장애물을 처리할 수 있는 적응형 정보 수집 모듈을 개발하였습니다. 또한, HM3D 및 Gibson 데이터셋의 연결 그래프를 활용하여 공간적 사전지식을 강화하는 방법을 제안하며, 이를 통해 실제 복잡한 환경에서의 경로 예측 능력을 향상시켰습니다.

- **Performance Highlights**: GVNav 접근법은 실제 환경과 시뮬레이션 환경 모두에서 성능을 크게 개선하는 결과를 나타내었습니다. 저자들은 Xiaomi Cyberdog을 사례 연구로 삼아 다양한 시각적 정보의 차이를 분석하고, 깊이 기반의 경로 예측이 낮은 시야각에서 어떤 영향을 받는지를 평가하였습니다. GVNav는 특히 복합적인 환경에서 사족보행 로봇의 작업 효율성을 높이는 데 기여하고 있습니다.



### Attention-Guided Integration of CLIP and SAM for Precise Object Masking in Robotic Manipulation (https://arxiv.org/abs/2502.18842)
Comments:
          2025 IEEE/SICE International Symposium on System Integration

- **What's New**: 이번 연구에서는 편의점 상품의 로봇 조작을 위한 객체 마스킹의 정밀성을 높이는 혁신적인 파이프라인을 소개합니다. 여기서는 CLIP(Contrastive Language-Image Pretraining)와 SAM(Segment Anything Model)이라는 두 가지 고급 AI 모델의 시너지 효과를 활용하며, 멀티모달 데이터(이미지 및 텍스트)를 효과적으로 사용합니다. 이러한 통합은 객체 마스킹의 성능을 개선하는 데 크게 기여하며, 로봇 시스템에 보다 정밀한 입력을 제공합니다.

- **Technical Details**: 제안하는 파이프라인은 최적화된 데이터셋과 작업별 미세 조정을 통해 기존 파이프라인이 직면한 데이터 제한 사항을 극복합니다. CLIP과 SAM을 gradient-based attention 메커니즘과 통합하여, 편의점처럼 객체가 다양하고 복잡한 환경에서도 신뢰성과 정확성을 높입니다. 이러한 방식으로 파라미터 설정을 통해 로봇 시스템의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 결과적으로, 새로운 방법론은 편의점에서의 로봇 작업에 필요한 객체 식별 및 조작의 정밀성을 개선하여 더 복잡한 작업을 효과적으로 수행하도록 합니다. 실제 환경에서의 적용 가능성을 높이며, 로봇 시스템이 다루는 물체에 대해 보다 세밀하고 적응적인 마스킹을 가능하게 합니다. 또한, 제안된 프레임워크는 다양한 맥락에서의 객체 조작의 신뢰성을 크게 향상시킬 것으로 기대됩니다.



### Subclass Classification of Gliomas Using MRI Fusion Techniqu (https://arxiv.org/abs/2502.18775)
Comments:
          15 pages, 7 figures, 1 algorithm, 4 tables, journal paper

- **What's New**: 이번 연구는 MRI 이미지를 T1, T2, T1ce 및 FLAIR 시퀀스에서 융합하여 신종 알고리즘을 개발했습니다. 하위 분류가 중요한 뇌종양인 교모세포종(glioma)의 정확한 분류를 통해 치료 계획 및 예후 예측을 목표로 하고 있습니다. 새로운 접근 방식을 통해 종양의 형태, 경계 및 강도 분포와 같은 세부 특성을 포착하여 클래스 분류의 정확성을 높이고 있습니다.

- **Technical Details**: 이 연구에서는 BraTS 데이터셋의 MRI 이미지를 사용하였으며, max-min normalization을 통해 이미지 간의 픽셀 강도 값의 일관성을 보장했습니다. UNET 아키텍처를 사용하여 괴사핵(necrotic core), 종양 주변 부종(peritumoral edema), 증가하는 종양(enhancing tumor)을 2D 및 3D 이미지에서 별도로 세분화(segmentation)하였습니다. 최종적으로 가중 평균(weighted averaging) 기법을 통해 다중 모드 MRI 이미지에서 세분화된 영역을 융합하였습니다.

- **Performance Highlights**: 제안된 방법은 80%의 데이터로 훈련하고 20%로 검증한 결과, 클래스 분류 정확도(accuarcy) 99.25%, 정밀도(precision) 99.30%, 재현율(recall) 99.10%, F1 점수(F1 score) 99.19%, 교차 면적(Intersection Over Union) 84.49%, 특이도(specificity) 99.76을 기록했습니다. 이러한 결과는 기존 기술보다 현저히 높은 성능을 보여주며, 교모세포종의 세분화 및 분류가 정확한 진단을 지원하는 데 중요한 역할을 한다는 것을 강조합니다.



### MaskPlanner: Learning-Based Object-Centric Motion Generation from 3D Point Clouds (https://arxiv.org/abs/2502.18745)
Comments:
          Project website at this https URL

- **What's New**: 이번 연구에서는 Object-Centric Motion Generation (OCMG)을 다루기 위해 MaskPlanner라는 새로운 데이터 기반의 프레임워크를 소개합니다. 기존의 접근 방식은 특정 기하학적 형태나 비싼 최적화 과정에 의존한 반면, MaskPlanner는 3D 포인트 클라우드로부터 전문가의 경로 패턴을 일반화할 수 있는 능력을 지닙니다. 이 방법은 작업에 대한 명시적 최적화를 하지 않고도 기존의 경량화된 경로 생성을 가능하게 하며, 이를 통해 다양한 산업적 활용 사례에 적합하게 적응할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: MaskPlanner는 주어진 객체에 대해 지역적인 경로 조각과 'path masks'를 동시에 예측하여 이를 여러 경로로 그룹화하는 딥러닝 방법입니다. 이 네트워크는 로컬 기하학적 패턴과 글로벌 작업 요구 사항을 단일 포워드 패스에서 캡처할 수 있도록 설계되어 있습니다. 또한, 경로 조각과 각 조각이 어떤 경로에 속하는지를 판별하는 바이너리 마스크를 학습함으로써, 필요한 경로 수와 각 경로의 길이를 효율적으로 추론할 수 있습니다.

- **Performance Highlights**: 실제 로봇 스프레이 페인팅 시나리오에서 MaskPlanner는 보이지 않는 객체에 대해 99% 이상의 완벽한 커버리지를 달성하였으며, 이를 통해 질적으로 전문가 수준의 페인팅 품질을 구현할 수 있었습니다. 또한, 이 방법은 100ms 내에 40개의 경로를 예측할 수 있으며, 70미터의 길이를 커버하고 8분의 실행 시간을 기록했습니다. 결과적으로, OCMG 문제 해결에 있어 상당한 진전을 이루었으며, 스프레이 페인팅을 대표 테스트 사례로 삼아 철저한 실험 평가를 실시하였습니다.



### QueryAdapter: Rapid Adaptation of Vision-Language Models in Response to Natural Language Queries (https://arxiv.org/abs/2502.18735)
- **What's New**: 이 논문에서는 QueryAdapter라는 새로운 프레임워크를 제안하여, 사전 훈련된 Vision-Language 모델(VLM)이 자연어 쿼리에 신속하게 적응할 수 있도록 지원합니다. 기존의 적응 전략은 폐쇄 클래스의 정의가 필요하지만, QueryAdapter는 이를 피하여 개방 어휘(open-vocabulary) 객체 탐지를 가능하게 합니다. 이를 통해 로봇이 자연어 쿼리에 따라 다양한 객체에 응답할 수 있는 능력을 향상시킵니다.

- **Technical Details**: QueryAdapter는 이전 배포에서 수집한 비라벨링 데이터(unlabelled data)를 활용하여 VLM 기능을 쿼리에 관련된 의미 클래스와 정렬합니다. 이 과정에서 learnable prompt tokens를 최적화하고, 각 목표 클래스별로 상위 k 개의 객체를 선택하여 훈련이 진행됩니다. 또한, 적합하지 않은 객체를 다루기 위해 객체 캡션을 negative class label로 사용하는 방식을 제안하여, 적응 과정에서 신뢰도 점수를 개선합니다.

- **Performance Highlights**: 방대한 ScanNet++ 데이터 세트에서 실험을 수행한 결과, QueryAdapter는 기존의 비지도 학습 VLM 적응기 및 3D 장면 그래프 방법들과 비교할 때 객체 검색 성능이 크게 향상되었습니다. 또한, 이 접근법은 추상적인 affordance 쿼리 및 Ego4D와 같은 다른 데이터 세트에 대해서도 강력한 일반화 성능을 보임을 확인했습니다.



### TerraTrace: Temporal Signature Land Use Mapping System (https://arxiv.org/abs/2502.18704)
- **What's New**: 본 논문은 NDVI(Normalized Difference Vegetation Index)를 기반으로 농업 및 계절 주기를 반영하는 독특한 시간적 서명을 분석하여 토지 이용 변화를 추적하는 TerraTrace 플랫폼을 제안합니다. 이 시스템은 캘리포니아에서 2020-2023년 간의 500m 해상도의 새로운 Longitudinal NDVI 데이터셋을 구축하여, 농장과 산림을 구분할 수 있는 NDVI 곡선을 제공합니다. 또한 사용자가 LLM 챗봇 및 그래픽 인터페이스를 통해 데이터를 쿼리할 수 있는 직관적인 분석 도구를 개발하였습니다.

- **Technical Details**: NDVI는 식물의 광합성에 기반해 계산되는 지표로, 특정 식물의 성장 주기에 따라 시간이 지남에 따라 독특하게 변화하는 패턴을 가지고 있습니다. 논문에서는 구글 어스 엔진과 마이크로소프트 플래네터리 컴퓨터의 Sentinel-2 이미지를 활용하여 NDVI를 처리하고 10미터 해상도로 변환한 후, 클라우드 생성된 데이터로부터 식별 가능한 데이터 포인트를 추출합니다. 이를 통해 다양한 작물의 NDVI 서명 곡선을 통해 농업 관행과 계절적 변화를 정량화할 수 있습니다.

- **Performance Highlights**: TerraTrace 플랫폼은 주어진 지역의 NDVI 데이터를 분석하여 토지 이용에 대한 통찰을 제공합니다. 이 시스템은 다년간의 데이터를 기반으로 토지 주기를 식별하고, 농작물의 건강도를 추척하며, 연례 증가 및 감소율을 계산하여, 건강한 식생의 존재 여부를 평가합니다. LLM 기반의 분석을 통해 다양한 메트릭스를 결합하여 사용자에게 직관적인 분석 결과를 제공합니다.



### Autonomous Vision-Guided Resection of Central Airway Obstruction (https://arxiv.org/abs/2502.18586)
Comments:
          Submitted to World Scientific, Journal of Medical Robotics Research (JMRR) 2025. 10 pages, 11 figures

- **What's New**: 이 연구는 기도(airway) 내 종양 제거를 위한 자율 로봇 시스템을 소개합니다. 기존의 제거 방법이 드물게 자율성을 제공했던 것과 달리, 본 시스템은 비전 기반 (vision-guided) 접근 방식을 통해 정밀한 재단(재단) 작업을 실현합니다. 이 연구는 5번의 연속 실험에서 성공적으로 기도 폐색을 제거한 결과를 통해 자율 수술 플랫폼의 가능성을 입증합니다.

- **Technical Details**: 시스템은 고유한 Faster R-CNN 세그멘테이션 파이프라인을 통해 기도와 종양 경계를 식별하며, 5차 다항식(polynomial)을 사용하여 기도 표면을 모델링합니다. 최적화된 전기 소작 도구(electrocautery tool)의 각도와 함께 1mm의 안전 여유 공간을 유지하며 도구의 경로를 계획합니다. 실험은 ex-vivo 동물 조직 모델을 사용하여 실시되었으며, 기도 손상을 방지하면서 90% 이상의 종양 제거율을 달성했습니다.

- **Performance Highlights**: 본 연구의 실험 결과는 수술어의 워크플로우가 자율 시스템을 통해 성공적으로 구현될 수 있음을 보여줍니다. 연구진은 기도 폐색 제거의 로봇 시스템이 일반적인 수술 방식과 동등한 정확성을 제공할 수 있다고 주장합니다. 이 데이터는 향후 최소 침습 수술(minimally-invasive surgery)에서의 자율 로봇 어플리케이션 발전의 기반이 될 것입니다.



### A Comparative Review of the Histogram-based Image Segmentation Methods (https://arxiv.org/abs/2502.18550)
- **What's New**: 본 논문에서는 이미지 세분화(Image Segmentation)를 위한 히스토그램 기반 기법의 역사 및 최근 발전을 검토합니다. 또한, 이 기법들은 평균 기반 방법, 가우시안 혼합 모델 기반 방법, 엔트로피 기반 방법, 그리고 특징 점 기반 방법의 네 가지 범주로 나뉩니다. 이러한 분류를 통해 기존 기법의 원리를 설명하고 성능 비교를 진행합니다.

- **Technical Details**: 히스토그램(histrogram)은 이미지의 그레이스케일(grayscale) 분포를 정확하게 시각화한 것으로, 픽셀의 확률 분포(probability distribution) 추정에 사용됩니다. 이 논문에서는 고전적인 히스토그램 기반 이미지 세분화 기술의 원리를 먼저 설명한 다음, 그들의 성능을 객관적으로 비교합니다. 마지막으로, 히스토그램 기반 방법과 일반 목적의 딥러닝(deep learning) 방법간의 세분화 성능을 비교합니다.

- **Performance Highlights**: 히스토그램 기반의 이미지 세분화 방법은 특별한 훈련 없이도 단순 배경을 가진 여러 종류의 이미지를 세분화하는 데 있어 일반적인 딥러닝 방법보다 더 정확합니다. 이러한 결과는 히스토그램 기법이 다양한 산업과 학계에서 여전히 중요한 역할을 하고 있음을 보여줍니다. 기존의 방법들이 가지는 유용함과 연구의 필요성 또한 강조됩니다.



### End-to-End Deep Learning for Structural Brain Imaging: A Unified Framework (https://arxiv.org/abs/2502.18523)
- **What's New**: 새로운 연구인 UniBrain은 뇌 영상 분석의 여러 처리 단계를 단일화하여 통합적인 최적화 프로세스를 통해 작동하는 최초의 심층 학습 모델입니다. 기존의 방법들과 달리, UniBrain은 최소한의 레이블 데이터로 운영되고, 모든 작업을 동시에 최적화할 수 있는 장점이 있습니다. 이는 중간 오류를 수정하는 전문가의 개입 없이도 진행할 수 있어, 시간과 비용을 크게 절감할 수 있습니다.

- **Technical Details**: UniBrain의 구조는 뇌 추출, 등록, 분할, 파셀레이션, 네트워크 생성 및 분류의 여러 모듈을 통합하여 각각의 과제가 서로 상호작용하도록 설계되었습니다. 각 모듈은 딥러닝 기반의 네트워크를 사용하여 연결되며, 이를 통해 세부 작업들 간의 의존성을 효과적으로 활용합니다. 이 시스템은 3D U-Net 같은 알고리즘을 통해 수행되며, 수집된 데이터의 특성을 반영해 다양한 작업에서의 정확성과 효율성을 향상시킵니다.

- **Performance Highlights**: UniBrain은 ADHD 공개 데이터셋을 기반으로 한 실험에서 기존의 여러 방법들보다 우수한 성능을 보여주었습니다. 모든 6개 작업에서 state-of-the-art 기법들을 초과하는 결과를 기록하였으며, 이는 UniBrain이 심층 학습을 통한 통합적 접근 방식을 통해 높은 확장성과 신뢰성을 제공한다는 것을 나타냅니다. 또한, 이 모델은 준비된 심층 정보의 질을 향상시키는 데 중요한 역할을 합니다.



### Rewards-based image analysis in microscopy (https://arxiv.org/abs/2502.18522)
Comments:
          38 pages, 11 figures

- **What's New**: 이 논문에서는 생물학, 의학, 화학 및 물리학을 포함한 다양한 과학 분야에서 이미징(Imagery) 및 하이퍼스펙트럼 데이터의 분석 방법을 다루고 있습니다. 특히, 인간의 입력이 최소화된 자동화된 이미징 도구의 증가에 따라, 데이터 표현을 최적화하는 비지도 학습(Unsupervised learning) 방법이 필요하다는 점을 강조하고 있습니다.

- **Technical Details**: 작성된 워크플로우는 전문가의 의사 결정 원칙을 채택하여 다양한 작업에서 강력한 전이 학습(Transfer learning)을 보여줍니다. 이미지 분석을 가능한 작업에 대한 의사 결정 프로세스로 표현하며, 전통적인 의사 결정 프레임워크에 대한 매핑(desiderata)과 요구 사항을 식별합니다.

- **Performance Highlights**: 보상 기반 워크플로우(Reward-driven workflows)는 감독된, 블랙박스 모델에서 설명 가능한 비지도 학습 및 이미지 분석에서 강건한 최적화로의 전환을 가능하게 합니다. 이는 전통적 방법과 DCNN 기반 기법 위에워커(wrapper)로 작용할 수 있어, 이미징 및 하이퍼스펙트럼 데이터 전반에 걸쳐 비지도 및 감독된 워크플로우에 적용될 수 있습니다.



### FreeTumor: Large-Scale Generative Tumor Synthesis in Computed Tomography Images for Improving Tumor Recognition (https://arxiv.org/abs/2502.18519)
- **What's New**: Tumor is a prominent cause of global mortality, with around 10 million fatalities annually due to related conditions. AI 기반의 종양 인식은 보다 정밀하고 지능적인 스크리닝과 진단 가능성을 열어주고 있지만, 주석이 달린 데이터셋의 부족으로 인해 연구가 제약받고 있습니다. 이를 해결하기 위해 FreeTumor라는 혁신적인 Generative AI (GAI) 프레임워크를 도입하여 대규모 종양 합성을 가능하게 하여 데이터의 부족 문제를 완화하고자 합니다.

- **Technical Details**: FreeTumor는 제한된 주석 데이터와 대규모 비주석 데이터를 효과적으로 결합하여 종양 합성 훈련을 진행합니다. 이를 통해 고품질의 합성 종양 이미지를 생성하고, 의료 이미지의 이해도를 향상시킵니다. 특히 저자들은 GAN(Generative Adversarial Networks)을 활용하여 비주석 데이터로부터 종양 합성 훈련을 수행하며, 이에 따라 고품질 합성 종양을 자동으로 선별하는 투사기(discriminator)를 포함합니다.

- **Performance Highlights**: FreeTumor는 161,310개의 CT 볼륨 데이터를 활용한 대규모 훈련 데이터세트를 구성하여 훈련 데이터의 양, 질, 다양성을 개선하며, 기존 AI 방법보다 40배 이상의 성능 향상을 기록했습니다. 13명의 인증 방사선 전문의에 의해 독립적인 시각적 테스트에서 51.1%의 민감도와 60.8%의 정확도로 고품질 합성 종양을 확인받았습니다. 이러한 결과들은 FreeTumor가 임상 응용에서의 가능성을 보여주며, 종양 치료를 진전시키고 환자의 생존율을 향상시킬 수 있는 기회를 제시합니다.



### Gradient entropy (GradEn): The two dimensional version of slope entropy for image analysis (https://arxiv.org/abs/2502.18516)
- **What's New**: 이번 논문은 경량화된 새로운 2D 엔트로피 측정 방법인 Gradient Entropy (GradEn)를 소개합니다. GradEn은 기울기 엔트로피(Slope Entropy)의 2차원 확장으로, 이미지 데이터를 분석할 때 상징적 패턴과 진폭 정보를 동시에 고려하여 특징 추출을 향상시킵니다. 시뮬레이션된 데이터와 실제 데이터에서 GradEn의 분류 성능을 평가하여 최신 2D 엔트로피 방법들과 비교하였습니다.

- **Technical Details**: GradEn은 이미지 X={xi,j}의 픽셀 간 기울기를 계산하여 2D 데이터셋을 분석합니다. 각 픽셀에 대해 수평, 수직, 대각선 방향으로 기울기를 표준화하는 과정을 거쳐, 이미지의 다양한 특징을 포착할 수 있습니다. 이 방법은 특히 복잡한 텍스처나 이미지 데이터의 분석에 적합하며, 실험을 통해 그 강력한 분류 능력을 확인했습니다.

- **Performance Highlights**: 실험 결과, GradEn은 다양한 2D 텍스처와 이미지의 특성을 효과적으로 구별하는 능력을 보였습니다. 실제 데이터셋을 사용한 분류 과제에서 GradEn의 성능은 다른 최신 2D 엔트로피 방법들에 비해 우수한 결과를 나타냈습니다. GradEn은 이미지 처리 및 인식 분야에서 새로운 접근 방식을 제시하며, 복잡한 데이터 분석에 있어 강력한 도구로 자리 잡을 것으로 기대됩니다.



### CipherFace: A Fully Homomorphic Encryption-Driven Framework for Secure Cloud-Based Facial Recognition (https://arxiv.org/abs/2502.18514)
- **What's New**: 이 논문에서는 암호화된 얼굴 영상의 안전한 처리를 위한 새로운 프레임워크인 CipherFace를 소개합니다. 이 시스템은 Fully Homomorphic Encryption(FHE)을 기반으로 하여 클라우드에서 얼굴 인식을 수행할 수 있도록 설계되어 있습니다. CipherFace는 공개 소스를 통해 제공되며, 이는 데이터 프라이버시와 클라우드 컴퓨팅의 효율성을 동시에 보장합니다.

- **Technical Details**: CipherFace는 DeepFace 라이브러리와 TenSEAL 라이브러리를 활용하여 facial embedding을 생성하고 완전 동형 암호화(fully homomorphic encryption)를 수행합니다. 이 시스템은 유클리드 거리와 코사인 거리를 위한 새로운 암호화된 거리 계산 방법을 제안하여 암호화된 데이터에서 유사성 계산을 안전하게 수행하도록 합니다. 이를 통해 얼굴 인식 시스템에서의 데이터 보안과 거리 계산의 운영성을 향상시킵니다.

- **Performance Highlights**: CipherFace는 다양한 얼굴 인식 모델 및 임베딩 크기와 암호 시스템 구성으로 실험을 진행하였으며, 이를 통해 실제 애플리케이션에서의 효율성과 확장성을 입증했습니다. 본 연구는 암호화된 얼굴 인식의 성능을 유지하면서도 데이터 프라이버시를 보장하는 실질적인 솔루션으로 자리잡을 것으로 기대됩니다. 더불어, 이 프레임워크는 생체 인식 외에도 여러 유사성 검색 사례에 적용할 수 있습니다.



### REFINE: Inversion-Free Backdoor Defense via Model Reprogramming (https://arxiv.org/abs/2502.18508)
Comments:
          This paper is accept by ICLR 2025. The first two authors contributed equally to this work. Our code is available at BackdoorBox (this https URL) and Github repository (this https URL). 28 pages

- **What's New**: 본 논문에서는 REFINE이라는 새로운 백도어 공격 방어 방법을 제안합니다. 이 방법은 모델 리프로그래밍(model reprogramming) 기법을 기반으로 하며, 입력 변환(input transformation)과 출력 재매핑(output remapping) 모듈을 포함하여 백도어 특성을 효과적으로 방어합니다. REFINE는 기존 방어 방식의 한계를 보완하며, 특히 백도어 트리거 역전(backdoor trigger inversion) 없이도 작동할 수 있습니다.

- **Technical Details**: REFINE는 입력 변환 모듈이 무해한 샘플과 백도어 패턴을 모두 방해하여 새로운 무해한 특징을 생성하는 구조를 갖습니다. 아울러, 출력 재매핑 모듈은 모델의 출력 도메인을 재정의하여 입력 변환이 효과적으로 이루어질 수 있도록 지원합니다. 또한 지도 대조 손실(supervised contrastive loss)을 통합하여 방어 능력을 향상시키면서도 모델 유틸리티를 유지합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트를 기반으로 한 실험 결과, REFINE의 효과성과 적응형 공격에 대한 저항력을 보여줍니다. REFINE는 기존의 방어 방법이 겪었던 성능 한계를 극복하며, 성능과 방어 효과 간의 균형을 유지할 수 있습니다. 이 연구는 AI 시스템을 위한 필수적인 백도어 방어 전략의 필요성을 강조합니다.



### Exploring Patient Data Requirements in Training Effective AI Models for MRI-based Breast Cancer Classification (https://arxiv.org/abs/2502.18506)
Comments:
          Accepted for publication in MICCAI 2024 Deep Breast Workshop on AI and Imaging for Diagnostic and Treatment Challenges in Breast Care

- **What's New**: 최근 10년 동안 의료 기관의 임상 의사 결정을 지원하는 AI 기반 솔루션을 제공하는 스타트업과 기업들이 급격히 증가하고 있습니다. 그러나 의료 결정의 중요성 때문에 외부 소프트웨어에 대한 의존에 관한 여러 가지 우려 사항이 제기되고 있습니다. 본 연구는 유방암 탐지를 중심으로 의료 기관이 효과적인 AI 모델을 훈련하는 데 필요한 데이터의 양을 탐색합니다.

- **Technical Details**: 본 연구에서는 Vision Transformer (ViT-B/16) 아키텍처를 사용하여 데이터 부족이 모델 훈련에 미치는 영향을 조사합니다. DINO와 MAE라는 두 가지 자가 지도 학습 프레임워크를 이용하여 사전 훈련된 모델을 사용하였고, Duke Breast Cancer Dataset을 기반으로 연구를 수행하였습니다. 이 데이터셋은 14년에 걸쳐 수집된 침습성 유방암 환자 922명의 3D MRI 이미지를 포함하고 있으며, 연구의 주요 작업은 MRI 이미지에서 유방 종양의 존재 여부를 판단하는 것입니다.

- **Performance Highlights**: 본 연구는 제한된 수의 이미지로도 사전 훈련된 모델이 최첨단 성능을 달성할 수 있음을 보여줍니다. 50명 이상의 환자로 구성된 훈련 세트에서는 환자 수의 변화가 모델 성능에 미치는 영향이 미미하며, 간단한 앙상블 방법이 추가 복잡성 없이도 성능을 향상시킬 수 있음을 관찰하였습니다. 이러한 결과는 데이터 부족 상황에서도 효과적인 AI 솔루션을 개발할 수 있는 가능성을 제시합니다.



### A Comprehensive Survey on Composed Image Retrieva (https://arxiv.org/abs/2502.18495)
- **What's New**: 이 논문에서는 Composed Image Retrieval (CIR)이라는 새로운 이미지 검색 작업을 체계적으로 검토하고 있습니다. CIR은 참조 이미지와 사용자가 원하는 변경 사항을 나타내는 수정 텍스트를 조합하여 사용자에게 보다 유연한 검색 방식을 제공합니다. 이 작업에 대한 포괄적인 리뷰가 현재 존재하지 않기 때문에, 120개 이상의 관련 연구를 종합하여 이 분야의 발전을 조망하고 있습니다.

- **Technical Details**: CIR의 주요 기술적 문제는 세 가지 주요 도전 과제를 포함합니다. 첫째, Multimodal Query Fusion은 수정 텍스트와 참조 이미지가 사용자의 검색 의도를 전달하는 데 상보적인 역할을 하는 것을 포함하여, 효과적인 멀티모달 융합 기능을 학습해야 합니다. 둘째, Target Images Matching은 멀티모달 쿼리와 목표 이미지 간의 의미적 간격을 해결하는 것을 목표로 합니다. 마지막으로, Scale of Training Data 문제는 학습 샘플을 만드는 데 필요한 비용과 노동 집약성을 다루고 있습니다.

- **Performance Highlights**: 기존 CIR 모델은 일반적으로 supervised learning과 zero-shot learning 두 가지 방법으로 구분됩니다. 감독 방식에서는 주석이 달린 학습 샘플이 필요하며, 제로샷 방식에서는 대규모 이미지-텍스트 쌍을 활용하여 사전 학습을 수행합니다. 다양한 데이터 세트와 실험 결과를 비교하여 기존의 supervised 및 zero-shot CIR 방법을 분석하고, 향후 연구 방향에 대한 통찰을 제공하여 연구자들에게 유용한 지침을 제시합니다.



### Deciphering Functions of Neurons in Vision-Language Models (https://arxiv.org/abs/2502.18485)
Comments:
          22 pages, 23 figures

- **What's New**: 최근 개방형 비전-언어 모델(VLMs)의 발전이 다양한 분야에 걸쳐 적용될 수 있는 가능성을 높였습니다. 이 연구의 목적은 개별 뉴런의 기능을 해석하고 시각적 토큰과 텍스트 토큰에 대한 뉴런의 반응을 관찰하여 흥미로운 발견을 밝혀내는 것입니다. 결과적으로, 특정 시각 정보 또는 텍스트 정보에만 반응하는 뉴런들과 두 가지 모두를 처리하는 다중 모달 뉴런들이 존재함을 발견했습니다.

- **Technical Details**: VLMs의 내부 메커니즘을 조사하기 위해, 연구진은 뉴런이 시각적 토큰과 텍스트 토큰에 어떻게 반응하는지를 분석했습니다. 그들은 GPT-4o를 활용한 자동 해석 프레임워크를 도입하여 뉴런의 기능을 자동으로 설명하고, 비주얼 뉴런에 대한 신뢰성을 평가하기 위한 활성화 시뮬레이터를 제안했습니다. LLaVA 1.5 모델을 분석하여 서로 다른 카테고리의 뉴런 특성을 밝혀냈습니다.

- **Performance Highlights**: 이 연구는 VLMs에 대한 투명하고 신뢰할 수 있는 AI 시스템 개발을 위한 방법론적 기초를 제공합니다. 활성화 패턴 분석 결과, 시각 또는 텍스트 전용 뉴런들이 존재하며 이들 뉴런의 기능을 이해하기 위한 규명 작업이 진행되었습니다. 연구를 통해 확보된 요소들은 다양한 응용 프로그램의 제작에 활용될 수 있으며, 시스템의 신뢰성을 향상시킬 수 있는 기초를 제공할 것으로 기대됩니다.



### A Fusion Model for Art Author Identification Based on Convolutional Neural Networks and Transformers (https://arxiv.org/abs/2502.18083)
- **What's New**: 이 논문은 미술 작가 식별을 위해 CNN(Convolutional Neural Networks)과 Transformer 모델을 결합한 새로운 융합 모델을 제안합니다. CNN은 작품에서 고유한 로컬 특성을 추출하는 데 사용되며, Transformer는 글로벌 문맥을 포착하는 데 중점을 둡니다. 이 모델은 미술작품의 미세한 요소를 잘 캡처하면서도 전체 스타일을 효과적으로 모델링할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 융합 모델은 CNN을 통해 로컬 피처를 추출하고, Transformer를 통해 이러한 피처를 글로벌하게 모델링합니다. 이후 피처 융합 메커니즘을 통해 로컬과 글로벌 피처를 통합하여 분류 정확도를 향상시킵니다. 실험 결과, 이 모델은 중국화와 유화 데이터셋에서 개별 CNN 및 Transformer 모델보다 각각 9.7%와 7.1% 개선된 분류 정확도를 보였습니다.

- **Performance Highlights**: 모델은 다양한 미술 장르의 작가 식별 작업을 수행할 수 있도록 훈련되었습니다. 또한 데이터가 적은 환경에서도 강력한 분류 정확도를 유지하므로, 향후 일반화 및 최적화가 필요한 가능성을 보여줍니다. 이러한 연구 결과는 향후 다중 모드 통합이나 아키텍처 최적화의 기초가 될 수 있습니다.



New uploads on arXiv(cs.AI)

### TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding (https://arxiv.org/abs/2502.19400)
- **What's New**: 이 연구에서는 TheoremExplainAgent라는 새로운 에이전트 기반 시스템을 소개합니다. 이 시스템은 Manim 애니메이션을 사용하여 길이가 5분 이상의 정리 설명 비디오를 생성할 수 있는 능력을 가지고 있습니다. 연구자들은 240개의 정리에 대한 기준을 포함한 TheoremExplainBench라는 벤치마크를 개발하여 다중 모드 정리 설명을 평가하는 체계적인 방안을 제안합니다.

- **Technical Details**: TheoremExplainAgent는 비디오 생성의 기획, 내레이션, 그리고 Python 애니메이션 스크립트를 생성하는 코딩 에이전트로 구성되어 있습니다. 이 시스템은 4개의 STEM 분야에서 다양한 정리 비디오를 생성할 수 있는 능력을 보여주며, 생성된 설명은 사실 정확성과 지각 품질의 5가지 차원에서 평가됩니다. 연구 결과에 따르면 에이전트 기반 기획이 상세한 긴 형식의 비디오 생성에 필수적이며, o3-mini 모델은 93.8%의 성공률과 0.77의 전반적인 점수를 기록했습니다.

- **Performance Highlights**: TheoremExplainAgent는 최대 10분 길이의 확장된 비디오 설명을 생성할 수 있어 기존 에이전트 없는 방법에 비해 상당한 진전을 보여주었습니다. 하지만 생성된 애니메이션은 종종 시각적 레이아웃에 사소한 문제가 발생하여 미세한 정렬 오류나 겹치는 도형이 나타나는 경우가 많았습니다. 이러한 시각적 오류는 보다 복잡한 정리에서 더욱 두드러지게 나타나, AI 시스템의 깊은 추론 결함을 드러내는 데 도움이 되었습니다.



### Joint Optimal Transport and Embedding for Network Alignmen (https://arxiv.org/abs/2502.19334)
Comments:
          12 pages, 7 figures

- **What's New**: 본 논문에서는 여타 네트워크 정렬(Network Alignment) 방식들과는 달리, JOENA라는 새로운 프레임워크를 제안하고 있습니다. JOENA는 임베딩(embedding)과 최적 수송(optimal transport) 방법을 통합하여, 상호 유익한 모델링을 통해 노이즈 감소로 보다 강력한 노드 정렬을 구현합니다. 이 방법은 엔드 투 엔드(end-to-end) 학습을 가능하게 하여, 사전 정의된 비용 함수에 의존하지 않고 더 나은 일반화를 이룹니다.

- **Technical Details**: JOENA 프레임워크는 두 가지 주요 기능을 제공합니다. 첫째, 전통적인 방식에 비해 OT 매핑을 사용하여 교차 네트워크 노드 쌍을 직접적으로 모델링합니다. 둘째, 학습된 임베딩을 바탕으로 OT 비용을 점진적으로 학습할 수 있어 정렬 품질을 더욱 개선합니다. 이러한 상관 관계를 보장된 수렴성을 가진 교대 최적화(alternating optimization) 스키마로 달성합니다.

- **Performance Highlights**: JOENA는 실제 네트워크에서 광범위한 실험을 통해 그 효과성과 확장성을 입증하였습니다. 기존의 최첨단 정렬 방법들과 비교했을 때, MRR에서 최대 16% 향상되고 20배의 속도 개선을 달성하는 성과를 보였습니다. 이러한 성과는 다양한 다중 네트워크 및 웹 마이닝 작업에서 활용될 수 있을 것으로 기대됩니다.



### WOFOSTGym: A Crop Simulator for Learning Annual and Perennial Crop Management Strategies (https://arxiv.org/abs/2502.19308)
- **What's New**: WOFOSTGym은 연간 및 다년생 작물 관리 결정을 최적화하기 위해 강화 학습( reinforcement learning, RL) 에이전트를 훈련시킬 수 있도록 설계된 새로운 작물 시뮬레이션 환경입니다. 이 시뮬레이터는 23종의 연간 작물과 2종의 다년생 작물을 지원하여 다년, 다작물 및 다농장 환경에서 다양한 농업 관리 전략을 학습할 수 있게 합니다. WOFOSTGym은 부분 가시성(partial observability), 비마르코프 동적(non-Markovian dynamics) 및 지연된 피드백(delayed feedback) 하에서 학습하기 위한 도전적인 작업들을 제공합니다.

- **Technical Details**: WOFOSTGym은 WOFOST 작물 성장 모델(crop growth model, CGM)을 기반으로 하여, 23종의 연간 작물과 2종의 다년생 작물의 성장을 모델링합니다. 이 시스템은 사용자가 쉽게 사용할 수 있도록 광범위한 사용자 정의 옵션과 표준 RL 알고리즘과의 무결한 통합을 지원합니다. 또한, 32종의 포도 품종에 대한 생리학적 모델의 정확성을 높이기 위해 베이지안 최적화(Bayesian Optimization) 기반의 방법이 적용됩니다.

- **Performance Highlights**: WOFOSTGym에서의 실험은 표준 RL 알고리즘과 모방 학습(imitation learning, IL) 에이전트가 다양한 작물 품종 및 토양 유형에 걸쳐 최적의 성과를 달성하였음을 보여줍니다. 이 시뮬레이터는 농업 관리 의사 결정 과제를 설계하여 RL의 농업 적용 가능성와 도전과제를 강조하며, 새로운 알고리즘을 개발하고 평가하는 데 있어 강력한 실험 플랫폼으로 자리매김하고 있습니다.



### Complex LLM Planning via Automated Heuristics Discovery (https://arxiv.org/abs/2502.19295)
- **What's New**: 이 논문에서는 복잡한 계획 작업을 위한 대형 언어 모델(LLMs)의 향상 방법으로 자동화된 휴리스틱 발견(Automated Heuristics Discovery, AutoHD)을 제안합니다. 기존 방법들이 신뢰할 수 없는 자기 검증(self-verification) 및 외부 검증자를 통해 중간 단계를 평가하는 데 의존하던 반면, AutoHD는 LLM이 효율적으로 추론을 할 수 있도록 휴리스틱 함수를 명시적으로 생성합니다. 이 새로운 접근 방식은 추가 모델 훈련이나 미세 조정 없이 구현 가능하며, 해석 가능성과 이유 과정을 이해할 수 있는 통찰을 제공합니다.

- **Technical Details**: AutoHD는 LLM이 Python 코드로 표현된 신뢰할 수 있는 휴리스틱 함수를 생성하도록 유도하여, 추론 중에 중간 단계를 평가하고 검색 과정을 효과적으로 안내합니다. 추가적으로, 휴리스틱 진화 프로세스를 도입하여 이 함수들을 반복적으로 개선합니다. 이 방법은 전통적으로 요구되는 추가 모델 훈련 없이도 LLM의 계획 능력을 향상시키며, 다양한 벤치마크에서 LLM의 계획 실행 및 해결 능력을 크게 발전시킵니다.

- **Performance Highlights**: AutoHD는 Blocksworld, Rubik’s Cube, 24게임과 같은 다양한 벤치마크에서 실시된 실험을 통해 여러 기준선보다 상당한 성과 향상을 보였습니다. 특히 일부 데이터셋에서는 거의 두 배에 가까운 정확도를 달성하며, 복잡한 계획 작업에 대한 신뢰할 수 있고 해석 가능한 솔루션으로 자리잡았습니다. 이러한 결과는 AutoHD가 LLM을 효율적이고 강력한 도구로 변모시킬 수 있는 잠재력을 더욱 부각시킵니다.



### Multi-Agent Security Tax: Trading Off Security and Collaboration Capabilities in Multi-Agent Systems (https://arxiv.org/abs/2502.19145)
Comments:
          Accepted to AAAI 2025 Conference

- **What's New**: 본 연구는 복잡한 목표를 달성하기 위해 AI 에이전트를 협업시키는 과정에서 자율 다중 에이전트 시스템의 보안을 보장해야 한다는 점을 강조하고 있습니다. 공격자가 하나의 에이전트를 손상시켜 전체 시스템을 오도된 결과로 이끌 수 있는 시나리오를 설정하고, 이를 통해 전염성 악의적 프롬프트(infectious malicious prompts)가 퍼질 수 있다는 것을 관찰하였습니다. 연구에서는 이러한 위험을 완화하기 위해 여러 방어 전략을 평가하며, '백신'과 유사한 두 가지 접근법을 포함시켰습니다.

- **Technical Details**: 연구는 자율화된 화학 연구 시설의 다중 에이전트 LLM 시뮬레이션을 통해 악의적 프롬프트의 전파를 보여주며, 이를 통한 시스템의 취약성을 분석합니다. 에이전트들은 다양한 역할을 수행하는 일곱 개로 구성되며, 이들은 협업을 통해 공동 목표를 달성하도록 설정되어 있습니다. 여러 개의 OpenAI 모델을 사용하여 에이전트들이 초기화되고, 각 에이전트는 메시지 큐를 통해 소통하며 AI 프로젝트 매니저인 Atlas가 조정합니다.

- **Performance Highlights**: 다양한 방어 전략을 통한 실험 결과, 악의적 지시의 전파 및 이행이 감소하였지만, 협업 능력이 저하되는 경향을 나타냈습니다. 시스템의 견고함(system robustness)과 에이전트 협업(agent cooperation) 간의 잠재적인 상충 관계(trade-off)를 실증적으로 측정하였으며, 이는 방어 전략의 평가 시 시스템의 정상 운영에 미치는 영향을 고려하지 않으면 간과될 수 있는 결과입니다. 이러한 연구 결과는 보다 안전하면서도 효과적인 AI 협업 시스템 설계의 통찰력을 제공합니다.



### A Temporal Planning Framework for Multi-Agent Systems via LLM-Aided Knowledge Base Managemen (https://arxiv.org/abs/2502.19135)
- **What's New**: 이 논문에서는 PLANTOR(PLanning with Natural language for Task-Oriented Robots)라는 새로운 프레임워크를 소개합니다. 이 시스템은 다중 로봇 작업을 위해 대형 언어 모델(Large Language Models, LLMs)과 Prolog 기반의 지식 관리 및 계획을 통합합니다. PLANTOR는 로봇 지식 베이스의 생성 과정을 두 단계로 나누어 재사용성과 조합적 추론을 보장합니다.

- **Technical Details**: PLANTOR는 혼합 정수 선형 계획(mixed-integer linear programming)을 사용하여 시간 의존성, 자원 제약 및 병렬 작업 실행을 처리하는 세 단계의 계획 절차를 포함합니다. 최종 계획은 ROS2에서 직접 사용할 수 있도록 행동 트리(Behaviour Tree)로 변환됩니다. 이 프레임워크는 블록 세계 및 아치 건축 시나리오 내의 다중 로봇 조립 작업에서 테스트되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, LLM은 적절한 인간 피드백으로 정확한 지식 베이스를 생성할 수 있으며, Prolog는 형식적 정확성과 설명 가능성을 보장합니다. 이 접근 방식은 유연하고 확장 가능하며 인간이 이해할 수 있는 계획이 필요한 고급 로봇 작업을 위한 LLM 통합의 잠재력을 강조합니다.



### Nexus: A Lightweight and Scalable Multi-Agent Framework for Complex Tasks Automation (https://arxiv.org/abs/2502.19091)
- **What's New**: 최근 대규모 언어 모델(Large Language Models, LLMs)의 발전은 다중 에이전트 시스템(Multi-Agent Systems, MASs)의 기능을 크게 향상시켰습니다. 이 연구에서는 LLM 기반 MAS의 구축을 위한 새로운 경량 Python 프레임워크인 Nexus를 소개합니다. Nexus는 특정 작업에 대해 LLM의 가능성을 최대한 활용할 수 있는 견고한 아키텍처와 LLM이 작업을 효율적으로 수행하고 정보를 관리할 수 있도록 하는 효과적인 방법론을 필요로 합니다.

- **Technical Details**: Nexus는 유연한 다중 감독자 계층(multi-supervisor hierarchy)과 단순화된 워크플로우 디자인을 제공합니다. 또한 쉽고 자유로운 설치가 가능하여 pip를 통해 설치할 수 있으며, 사용자들이 자유롭게 수정하고 확장할 수 있도록 허가된 오픈소스 라이센스 아래 배포됩니다. 이러한 아키텍처는 특정 작업 세트에서 우수한 성능을 발휘하는 데 중점을 두고 설계되었습니다.

- **Performance Highlights**: 실험 결과, Nexus로 구성된 아키텍처는 다양한 도메인에서 최첨단 성능을 나타냅니다. 코딩 작업에서 Nexus 기반의 MAS는 HumanEval에서 99%의 통과율을 기록하고 VerilogEval-Human에서 완벽한 100%의 통과율을 달성했습니다. 또한 이 아키텍처는 복잡한 추론 및 수학적 문제 해결에서 뛰어난 능력을 보이며, MATH 데이터셋의 무작위로 선택된 문제에 대해 모든 정답을 올바르게 해결했습니다.



### Dealing with Inconsistency for Reasoning over Knowledge Graphs: A Survey (https://arxiv.org/abs/2502.19023)
- **What's New**: 본 논문에서는 일관성이 결여된 지식 그래프(knowledge graphs, KGs)에 대한 추론 방법을 논의합니다. 특히, KGs의 불일치를 감지하고 수정하며, 이를 바탕으로 일관성이 없는 KG에서도 가능하게 하는 추론 방법에 초점을 맞추고 있습니다. 다양한 연구 분야로부터의 기존 작업을 분석하여 어떻게 서로 연결되는지를 탐구합니다.

- **Technical Details**: KG는 ABox(확장적 지식)와 TBox(내재적 지식)로 구성됩니다. TBox는 일반적으로 Description Logic(DL)에 표현되며, 이는 개념(또는 클래스)과 역할(또는 관계)을 모델링하는 데 사용됩니다. 이 문서에서는 DL을 통한 KG의 불일치 감지 및 수정 방법에 대한 여러 접근 방식을 정리합니다.

- **Performance Highlights**: 일관성이 결여된 KGs의 추론을 위한 최신 기술과 접근 방식을 제시하여, 연구자들이 불일치 문제를 해결하고 일관성을 높이는 데 도움이 되도록 하고 있습니다. 또한, ABox의 수정을 통해 실제 문제에 대한 일관성을 도모할 수 있는 방법을 논의하며, KG의 품질 향상과 오류 감지의 중요성을 강조합니다.



### Talking like Piping and Instrumentation Diagrams (P&IDs) (https://arxiv.org/abs/2502.18928)
- **What's New**: 이번 연구에서는 자연어를 활용하여 배관 및 계측도(P&ID)와 의사소통할 수 있는 방법론을 제안합니다. 이 방법론은 DEXPI 데이터 모델을 통해 P&ID를 라벨링된 속성 그래프로 표현하고, 이를 대형 언어 모델(LLM)과 통합합니다. 특히, pyDEXPI라는 파이썬 패키지를 활용해 DEXPI 포맷에서 그래프 형식으로 변환하는 과정이 포함되어 있습니다.

- **Technical Details**: 본 연구의 방법론은 세 가지 주요 부분으로 구성됩니다. 첫째, DEXPI 형식의 P&ID를 pyDEXPI 패키지를 사용하여 그래프 표현으로 변환합니다. 둘째, pyDEXPI로부터 P&ID 지식 그래프를 생성하는 도구를 개발하고, 셋째, 이 지식 그래프를 그래프 기반 검색 보강 생성(graph-RAG) 기술을 통해 LLM과 통합하여 자연어로 P&ID와 상호작용할 수 있도록 합니다.

- **Performance Highlights**: 본 연구에서는 또한 고수준 그래프 표현을 생성하여 LLM의 정보 효율성을 향상시킵니다. 처리의 효율성을 증대시키기 위해 불필요한 노드 속성을 제거하고, 저정보 노드를 압축하며 도메인 정보를 가지치기하는 방식으로 그래프의 크기를 줄였습니다. 이러한 과정으로 그래프의 노드 수는 212에서 53으로, 리레이션십 수는 405에서 57로 감소하였고, LLM의 토큰 사용량도 약 67,000에서 9,000으로 줄어들었습니다.



### Multi-LLM Collaborative Search for Complex Problem Solving (https://arxiv.org/abs/2502.18873)
- **What's New**: 이번 논문에서는 Mixture-of-Search-Agents (MoSA) 패러다임을 제안하여, 여러 개의 Large Language Models (LLMs)의 집단적 전문성을 활용하여 복잡한 추론(task) 문제를 해결합니다. MoSA는 독립적인 탐색과 반복적인 세부 조정을 통해 다양한 추론 경로를 통합함으로써 단일 모델 접근의 한계를 극복합니다. 특히, Monte Carlo Tree Search (MCTS)를 기반으로 하여 여러 에이전트가 추론 단계를 제안하고 집계하는 기능을 포함하고 있습니다.

- **Technical Details**: MoSA의 기본 접근 방식은 단계별 탐색 기반 추론을 통해 문제를 유도된 그래프에서의 탐색으로 나누는 것입니다. 이때 각 상태는 지금까지 생성된 추론 단계의 집합을 나타내며, 행동 공간은 다음 추론 단계를 기반으로 합니다. 여러 LLM이 서로 독립적으로 또는 협력적으로 여러 추론 단계를 제안하는 것을 통해, 나쁜 지역 최적화(local optima) 문제를 회피하고 추론 정확도를 높이는 방식으로 설계되었습니다.

- **Performance Highlights**: 본 연구에서는 MoSA가 단일 LLM 대비 평균 1.71%의 추론 정확도 개선을 보이며, 다수의 추론 기준에서도 일관된 성능 개선을 나타냄을 입증했습니다. 다중 에이전트의 협업과 탐색 기반 추론의 시너지를 강조하며, 과제 관점에서 다양성과 품질의 균형이 평가되는 중요한 문제임을 확인했습니다. 추가적인 행동 세트를 사용한 실험은 MoSA의 다양한 탐색 작업에 대한 견고성을 입증하였습니다.



### Towards an AI co-scientis (https://arxiv.org/abs/2502.18864)
Comments:
          81 pages in total (main 38 pages, appendix 43 pages), 13 main figures, 40 appendix figures, 1 main table, 2 appendix tables, 143 main references, 7 appendix references

- **What's New**: 이번 연구에서는 Gemini 2.0을 기반으로 한 AI 공동 과학자(AI co-scientist)를 도입했습니다. 이 시스템은 과학자들이 새로운 가설을 생성하고 이를 엄격한 실험적 검증을 통해 확인하는 데 도움을 줍니다. AI 공동 과학자는 이전의 증거를 기반으로 하여 혁신적인 연구 가설과 제안을 형성하는 데 중점을 두고 있습니다.

- **Technical Details**: 시스템은 생성(generate), 토론(debate), 발전(evolve)의 접근 방식을 결합하여 가설 생성을 수행하며, 융통성 있는 컴퓨팅 스케일링을 위한 비동기(task execution) 아키텍처를 적용합니다. 이 외에도 자기 개선(self-improving) 가설 생성을 위한 토너먼트 진화(tournament evolution) 과정을 포함하고 있습니다. 자동화된 평가를 통해 테스트 시간 컴퓨팅(test-time compute)의 지속적인 이점이 입증되었고, 이는 가설의 품질 향상에 기여하고 있습니다.

- **Performance Highlights**: 이 시스템은 약물 재사용(drug repurposing), 새로운 표적 발견(novel target discovery), 박테리아 진화 및 항균 저항 메커니즘 설명 분야에서 개발 및 검증에 집중하고 있습니다. 예를 들어, 급성 골수성 leukemia 치료를 위한 유망한 후보 물질을 제안하였고, 이는 저자 제안의 임상 적용 가능 농도에서 종양 억제 효과를 보였습니다. 또한 간 섬유증을 위한 새로운 후생유전학적(target) 표적을 제안하였으며, 이는 인체 간 유기체에서 anti-fibrotic 활동과 세포 재생으로 검증되었습니다.



### Intelligence Tes (https://arxiv.org/abs/2502.18858)
- **What's New**: 이번 논문은 지능이 우연히 발생하는 것이 아니라, 자연 선택(Natural Selection)의 필수적인 결과로서 생물 종이 생존하기 위한 특성이라는 주장을 제시합니다. 저자들은 'Intelligence Test'라는 방법을 도입하여, 과제가 주어졌을 때 피실험자의 지능을 수치적으로 평가하고, 그 결과를 기존 AI 시스템에 적용하여 AI의 자율성을 측정합니다. 이 테스트는 시도와 오류를 통해 성공하기까지의 실패 횟수를 세어 지능을 정량화하며, 실패 횟수가 적을수록 지능이 높다고 간주합니다.

- **Technical Details**: Intelligence Test는 지능을 세 가지 수준: 제한적(Limited), 능력(Capable), 자율적(Autonomous)으로 나누어 평가합니다. 기대값(expectation)과 분산(variance)이 모두 수렴(converge)하면 자율적 수준에 도달한 것으로 간주되며, 이 경우 피실험자는 몇 번의 시도만으로도 올바른 해결책을 찾아낼 수 있습니다. 반면, 기대값과 분산이 모두 발산(diverge)하면 제한적 수준으로, 이는 가능성 있는 해결책을 무작정 나열하는 것과 같다고 설명합니다.

- **Performance Highlights**: AI 시스템을 대상으로 한 결과는, 현재의 AI가 단순한 작업에서는 자율적 수준에 도달할 수 있지만, 복잡한 작업에서는 여전히 제한적 수준에 머물고 있음을 보여줍니다. 예를 들어, 손글씨 숫자 인식에서는 자율적 수준에 도달할 수 있지만, 비전, 검색, 추천 및 언어 처리와 같은 복잡한 작업에서는 제한적 수준을 벗어나지 못하고 있습니다. 이와 같은 결과는 AI 기술이 상당한 수준의 자율성을 달성하지 못하고 있다는 점을 명확히 하고, 현행 AI 기술의 한계를 강조합니다.



### REALM-Bench: A Real-World Planning Benchmark for LLMs and Multi-Agent Systems (https://arxiv.org/abs/2502.18836)
Comments:
          14 pages, 4 figures, 9 tables

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)과 다중 에이전트 시스템을 평가하기 위한 포괄적인 벤치마크(RAELM-Bench) 프레임워크를 소개합니다. 이 벤치마크는 다루기 쉬운 문제부터 복잡한 문제까지 총 11개의 시나리오로 구성되어 있으며, 다중 에이전트 조정, 에이전트 간 의존성 및 동적 환경 변화를 포괄합니다. 벤치마크는 평가 및 구현에 필요한 기준과 메트릭스를 제공하여 AI 계획 시스템의 성능을 엄격하게 테스트할 수 있도록 합니다.

- **Technical Details**: REALM-Bench는 세 가지 차원으로 확장될 수 있는 문제들을 포함합니다. 첫째, 병렬 계획 스레드(parallel planning threads)의 수; 둘째, 상호 의존성(inter-dependencies)의 복잡성; 셋째, 예기치 않은 방해(disruptions)의 빈도 및 영향력입니다. 각 문제는 특정 난이도 레벨에 따라 분류되며, 기초적인 조정을 요구하는 문제부터 복잡한 의존성을 가진 문제까지 다양합니다.

- **Performance Highlights**: 이 벤치마크는 AI 시스템이 점차 도전적인 조건에서 평가될 수 있게 하며, 또한 실패 모드 분석이 가능하게 합니다. 예를 들어, 캠퍼스 투어 조정 문제는 단순한 두 그룹에서 시작하여 복잡한 비상 상황에 따라 평가됩니다. 이러한 구조 덕분에 AI 계획 시스템의 성능을 보다 효과적으로 향상시키고 안정적인 시스템을 개발할 수 있는 기반을 마련합니다.



### Data-Efficient Multi-Agent Spatial Planning with LLMs (https://arxiv.org/abs/2502.18822)
- **What's New**: 이 프로젝트에서는 pretrained large language models (LLMs)를 활용하여 다중 에이전트 의사결정에서 효율적이고 강건한 학습을 달성하는 방법을 모색합니다. 특히, 택시 라우팅 및 배정 문제를 통해 에이전트가 승객을 픽업하는 최적의 방법을 결정하여 대기 시간을 최소화하는 과제를 다룹니다. 적절한 프롬프트(prompting)를 사용하면 zero-shot 성능이 강력하고, 제한적인 파인튜닝과 롤아웃 알고리즘을 결합하면 기존의 방법보다 50배 적은 환경 상호작용으로도 성능을 능가할 수 있음을 보여줍니다.

- **Technical Details**: 우리는 고정된 거리 구조를 가진 도시 환경에서 다중 에이전트 택시 라우팅 문제를 연구합니다. 각 에이전트에 대해 환경, 모든 에이전트의 현재 상태, 요청 사항을 설명하는 텍스트를 제공하고, LLM은 최종 결정을 출력합니다. Llama3-8B-Instruct 모델은 이 설정에서 강력한 zero-shot 성능을 발휘하며, 롤아웃 알고리즘과 조합했을 때 이전의 최첨단 접근법보다 더 적은 환경 상호작용으로도 성능이 우수함을 보여줍니다.

- **Performance Highlights**: 실험 결과, Llama 3-8B-Instruct는 zero-shot 설정에서도 뛰어난 성능을 거두었습니다. 파인튜닝된 롤아웃 정책은 이전 최고 성능의 접근법을 초월하며 필요한 훈련 데이터 양이 극적으로 줄어듭니다. 또한, 파인튜닝은 프롬프트 기반 방법에서 발생되는 공간적 헛소리를 효과적으로 제거합니다.



### Holistic Audit Dataset Generation for LLM Unlearning via Knowledge Graph Traversal and Redundancy Remova (https://arxiv.org/abs/2502.18810)
Comments:
          11 pages, 4 figures

- **What's New**: 최근 대규모 언어 모델(LLMs)에서 민감한 정보 제거 및 개인정보 보호와 같은 요구 사항이 증가함에 따라, Machine Unlearning을 통한 비법적 정보 삭제가 주목받고 있습니다. 이 논문에서는 기존의 신뢰할 수 있는 평가 기준이 부족한 현실을 지적하며, HANKER라는 새로운 자동화된 프레임워크를 제안하여 보다 포괄적인 감사 데이터를 생성할 수 있도록 합니다. HANKER는 지식 그래프(Knowledge Graphs)를 활용하여 그 범위를 넓히고 중복된 지식을 제거합니다.

- **Technical Details**: HANKER는 잊고 있는 데이터셋(Forget Dataset)과 유지하는 데이터셋(Retain Dataset)을 구조화된 지식 그래프로 변환하여 감사 과정의 범위를 명확히 정의합니다. 이 알고리즘은 각 KG 엣지를 최소 단위로 취급하여 질문-답변 쌍을 생성할 때 겹치는 정보를 필터링 합니다. 이를 통해 헌신적인 감사 절차를 보장하고, 고품질의 테스트 질문을 포함함으로써 포괄적이고 정확한 평가를 가능하게 합니다.

- **Performance Highlights**: MUSE 벤치마크에 HANKER를 적용한 결과, 뉴스 및 도서 데이터셋에서 각각 69,000개 및 111,000개의 감사 사례를 성공적으로 생성하였으며, 지난 연구에서 식별하지 못한 수천 건의 지식 기억 사례를 발견하였습니다. 특히, 중복된 지식이 비법적 각성 효과성 지표에 크게 영향을 미치는 것으로 나타났으며, 이는 중복이 기억 측정치를 인위적으로 부풀릴 수 있음을 강조합니다. 따라서 시스템적 중복 제거의 필요성이 강조되며, HANKER는 이 평가의 정확성을 제고하는 데 기여할 것입니다.



### Like Father, Like Son: Kinship-Aware Preference Mapping (KARMA) for Automatic Alignment in Large Language Models (https://arxiv.org/abs/2502.18744)
Comments:
          14 pages,5 figures,3 tables,4 graphs

- **What's New**: 최근 Large Language Model (LLM) 정렬의 발전은 사전 훈련된 모델을 활용하여 인간 주석의 비용을 줄이려는 노력을 포함합니다. 기존 방법들이 본질적으로 다른 능력의 모델 간에 응답을 비교하는 경향이 있었지만, 이로 인해 유의미하지 않은 구분이 발생했습니다. 이를 해결하기 위해, 우리는 유사한 능력을 가진 모델 간 응답을 짝짓는 Kinship-Aware pReference MApping (KARMA) 프레임워크를 제안합니다.

- **Technical Details**: KARMA는 모델의 유사성을 고려하여 동일한 복잡성 및 품질을 갖춘 생성물에 대해 우선 선호를 비교합니다. 이러한 방법은 응답의 품질 격차(Respons Quality Gap)가 줄어들 수 있도록 하여 더 정밀하고 의미 있는 preference 신호를 생성합니다. 또한, KARMA는 모델의 성능을 기준으로 서로의 관계성을 평가하여, 이를 바탕으로 선호 데이터를 체계적으로 생성합니다.

- **Performance Highlights**: KARMA의 효과는 경험적 평가에서 드러나며, 기존의 방법론인 RLAIF보다 월등히 우수한 결과를 보였습니다. 이 연구는 LLM 행동을 인간의 선호와 더 잘 일치시킬 수 있는 견고하고 확장 가능한 경로를 제시합니다. 결과적으로, KARMA는 더 높은 품질의 정렬 신호를 제공하게 되어 LLM의 행동을 보다 확실하게 조정할 수 있게 됩니다.



### Talking to the brain: Using Large Language Models as Proxies to Model Brain Semantic Representation (https://arxiv.org/abs/2502.18725)
Comments:
          20 pages, 6 figures

- **What's New**: 이 연구는 전통적인 심리학 실험에서 자연적인 자극을 사용할 때 발생하는 수작업 주석(annotation)과 생태적 타당성(ecological validity) 문제를 해결하기 위해, 다중 모달 대형 언어 모델(multimodal large language models, LLMs)을 활용한 새로운 패러다임을 소개합니다. 이를 통해 시각적 질문 답변(Visual Question Answering, VQA) 전략을 사용하여 자연적인 이미지에서 풍부한 의미 정보를 추출합니다.

- **Technical Details**: 연구에서는 LLM 기반 데이터 표현이 fMRI 기능적 자기공명영상(fMRI)으로 측정된 기존의 신경 활동 패턴(예: 얼굴, 건물)을 성공적으로 예측한다는 것을 강조합니다. 또한, 이 표현을 사용하여 뇌의 의미 네트워크를 구성하고, 이는 기능적 및 맥락적 연관성을 반영하는 의미 있는 군집을 식별하는데 기여합니다.

- **Performance Highlights**: 이 혁신적인 방법론은 전통적인 주석 방법의 한계를 극복하고, 자연적인 자극을 사용하여 뇌의 의미 조직(brain semantic organization)을 조사하는 강력한 솔루션을 제공합니다. 이는 인간 인지(human cognition)의 생태적으로 타당한 탐색을 위한 길을 열어줍니다.



### TrajLLM: A Modular LLM-Enhanced Agent-Based Framework for Realistic Human Trajectory Simulation (https://arxiv.org/abs/2502.18712)
Comments:
          Accepted WWW2025 Demo Paper

- **What's New**: 이 연구는 기존의 고비용과 개인정보 우려를 해결하기 위해 대규모 언어 모델(LLMs)을 활용하여 인간의 이동성을 시뮬레이션하는 새로운 프레임워크인 TrajLLM을 제안합니다. 이 프레임워크는 페르소나 생성, 활동 선택, 목적지 예측의 계층적 통합을 통해 실제 인구 통계 및 심리적 데이터를 기반으로 신뢰성 있는 이동 패턴을 생성합니다. 초기 결과는 LLM 기반의 시뮬레이션이 실제 관찰된 패턴과 일치함을 보여줍니다.

- **Technical Details**: TrajLLM은 네 개의 주요 모듈인 페르소나, 활동, 목적지, 메모리로 구성되어 있습니다. 첫 번째 모듈인 페르소나 생성에서는 정부 웹사이트에서 수집한 인구 통계 데이터를 사용하여 LLM이 현실과 일치하는 페르소나를 생성합니다. 이후 페르소나는 특정 활동-위치 리스트를 배정받고, 다양한 정보를 기반으로 LLM이 각 페르소나의 다음 활동을 시뮬레이션합니다.

- **Performance Highlights**: 이 프레임워크는 인간 행동의 자연스러운 패턴을 재현하기 위해 이론 기반 물리 모델과 LLM의 장점을 결합하여 시뮬레이션 효율성을 극대화합니다. 생성된 데이터는 도시 계획, 교통 관리, 공공 건강 등 사회 문제에 대한 해석 가능한 통찰력을 제공하며, 적응 가능하고 현실적인 일상 스케줄을 만들어 냅니다. 실제로 TrajLLM은 높은 정확도와 성능의 균형을 이루고 있어 사회 및 도시 응용을 위한 이동성 모델링의 발전 가능성을 보여줍니다.



### Hybrid Voting-Based Task Assignment in Role-Playing Games (https://arxiv.org/abs/2502.18690)
Comments:
          Accepted for presentation at Dungeons, Neurons, and Dialogues: Social Interaction Dynamics in Contextual Games Workshop at 20th Annual ACM/IEEE International Conference on Human-Robot Interaction (HRI 2025)

- **What's New**: 이번 연구에서는 Role-Playing Game (RPG)에서의 몰입 경험을 향상시키기 위해 Voting-Based Task Assignment (VBTA)라는 새로운 프레임워크를 제안합니다. 기존의 방법들은 게임의 개별 요소에만 집중하는 반면, VBTA는 LLM과 구조화된 작업 할당 모델을 통합하여 여러 작업 간의 원활한 전환을 가능하게 합니다. 이를 통해 플레이어가 직접 경험하는 서사와 전투 상황의 생성이 개선되어 보다 풍부하고 몰입감 있는 RPG 경험을 제공합니다.

- **Technical Details**: VBTA 프레임워크는 각 에이전트에 대한 능력 프로필을 설정하고 각 작업에 대한 요구사항이 정의된 작업 설명을 바탕으로 작동합니다. 적합성 행렬(suitability matrix)을 통해 에이전트와 작업 간의 호환성을 정량적으로 평가하며, 여러 작업에 대한 최적의 에이전트를 지정합니다. 이를 위해 VBTA는 6가지의 투표 방법(voting methods)과 3가지 할당 전략을 활용하여 에이전트 간의 충돌을 해결하고, Conflicts-Based Search (CBS)를 통해 경로 계획을 최적화합니다.

- **Performance Highlights**: VBTA의 성능은 Dungeons & Dragons와 Baldur’s Gate 3의 사례를 통해 입증되었습니다. VBTA는 기존 방법들이 다룰 수 없는 독특한 전투 상황과 서사를 생성함으로써, RPG의 게임 세계를 풍부하게 만들고 높은 재플레이 가능성을 보여줍니다. 이 프레임워크는 게임 콘텐츠를 생성하는 데 있어 깊은 내러티브 이해와 에이전트의 동작 결정을 자동화하는 가능성을 열어줍니다.



### Speaking the Right Language: The Impact of Expertise Alignment in User-AI Interactions (https://arxiv.org/abs/2502.18685)
Comments:
          arXiv Version

- **What's New**: 이 연구는 25,000개의 Bing Copilot 대화를 분석하여 다양한 도메인 전문성을 가진 사용자의 응답 방식과 사용자 경험에 미치는 영향을 조사했습니다. 결과적으로 LLM(대형 언어 모델)은 대화의 77%에서 능숙하거나 전문가 수준으로 응답하며, 이는 사용자의 전문성 수준과 무관하게 긍정적인 사용자 경험과 연관이 있음을 보여줍니다. 또한, 전문가 수준의 응답이 아닌 경우 사용자 경험에 부정적인 영향을 미치며, 이는 복잡한 작업에 대해 더욱 두드러집니다.

- **Technical Details**: 연구는 사용자 전문성, 신뢰된 사용자 전문성, 에이전트 전문성의 세 가지 전문성 측정을 위해 5점 정렬척도를 사용하여 25,033개의 대화 데이터를 분류했습니다. 각 대화의 전문성 수준을 평가한 후, 사용자 만족도(SAT Score), 작업 복잡도(Task Complexity), 대화 길이(Conversation Length)와 같은 세 가지 메트릭을 통해 전문성 정렬의 영향을 측정했습니다. 이러한 메트릭은 LLM과 사용자 간의 미스알라인(misalignment)의 포괄적인 영향을 평가하는 데 사용되었습니다.

- **Performance Highlights**: 분석 결과, 전문성 정렬이 이루어진 경우 사용자 참여도는 높아지고, 대화에서 사용된 단어 수로 측정한 참여 수준이 증가하는 것으로 나타났습니다. 대다수의 사용자(63.9%)는 "초보자"로 분류되었으며, 전문적인 도메인에서 더 높은 전문성을 가진 사용자 수치도 나타났습니다. 에이전트는 전체적으로 "능숙" 또는 "전문가"로 분류되는 경우가 많은데, 이는 사용자의 요구와 AI 에이전트 간의 조화로운 상호작용이 효과적인 사용자 경험을 보장하는 데 필수적임을 강조합니다.



### Independent Mobility GPT (IDM-GPT): A Self-Supervised Multi-Agent Large Language Model Framework for Customized Traffic Mobility Analysis Using Machine Learning Models (https://arxiv.org/abs/2502.18652)
Comments:
          24 pages, 4 figures, TRR accepted

- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 기반으로 한 독립적 교통 분석 및 관리 제안을 위한 혁신적인 다중 에이전트 프레임워크인 IDM-GPT를 제안합니다. 이 프레임워크는 교통 데이터베이스와 ML 모델을 경제적으로 연결하여 사용자가 교통 분석과 맞춤형 제안을 실시간으로 얻을 수 있도록 지원합니다. IDM-GPT는 복잡한 ML 모델과의 상호작용을 단순화하여 비전문가도 쉽게 사용할 수 있도록 설계되었습니다.

- **Technical Details**: IDM-GPT는 여러 AI 에이전트를 포함하는 다중 에이전트 LLM 프레임워크로, 입력 검증(IV), 자기 최적화(SP), 데이터베이스 상호작용(DBI), 데이터 분석(DAS), 자가 감독(SS) 에이전트로 구성됩니다. 각 에이전트는 사용자의 쿼리를 검토하고, 최적의 프롬프트를 생성하며, 필요한 데이터를 자동으로 검색하고 분석 결과를 생성하는데 핵심 역할을 합니다. 이러한 구조는 데이터 품질을 보장하고, 사용자 요구를 충족시키며, 교통 데이터의 기밀성을 유지합니다.

- **Performance Highlights**: IDM-GPT는 여러 교통 관련 작업에서 만족스러운 성능을 발휘하며, 사용자가 효율적으로 데이터를 분석하고 제안을 받을 수 있는 기반을 제공합니다. 실험 결과는 이 프레임워크가 실시간 교통 관리 및 도시 이동성 개선에 효과적인 인사이트를 제공함을 보여줍니다. 또한, 이 시스템은 교통 기관이 정보에 기반한 결정을 내리고 긴급 교통 제어 전략을 개발할 수 있도록 지원하여 궁극적으로 도시 이동성을 향상시킵니다.



### Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems (https://arxiv.org/abs/2502.18632)
- **What's New**: 본 연구에서는 문제에 매핑된 지식 구성 요소(Knowledge Components, KCs)를 자동으로 생성하고 태깅(tagging)하기 위한 LLM(대규모 언어 모델) 기반의 파이프라인을 제안합니다. 이 방법은 전통적으로 전문가에 의해 수행되던 작업을 최소화하여 개인화된 학습 경험을 개선하는 데 기여할 것으로 기대됩니다. KCGen-KT라는 이 파이프라인은 실제 학생 코드 제출 데이터를 활용하여 기존의 KT 방법보다 우수한 성과를 보여줍니다.

- **Technical Details**: 지식 구성 요소는 학생의 학습 상태를 세밀하게 추적하는 데 필수적입니다. 연구자들은 LLM을 활용하여 코드 작성 문제를 해결하기 위한 KCs를 생성하고, 이를 통해 문제에 대한 성능 예측을 위한 KT(Knowledge Tracing) 프레임워크를 개발했습니다. 이 과정에서 추상 구문 트리(Abstract Syntax Tree, AST)를 사용하여 코드 솔루션을 변환하고 KCs를 생성하는 방식으로 나아갔습니다.

- **Performance Highlights**: KCGen-KT는 CodeWorkout 데이터셋에서 평가를 수행하여 기존의 KT 방법보다 뛰어난 성능을 확인했습니다. 실험 결과, LLM이 생성한 KCs는 인간이 작성한 KCs와 유사한 적합성을 보였으며, 인간 전문가와 비교했을 때도 KC 태깅의 정확성이 상당히 높은 것으로 나타났습니다. 이 연구는 KCs 자동 생성 및 KT 분야에서 중요한 진전을 이루었다고 할 수 있습니다.



### CuDIP: Enhancing Theorem Proving in LLMs via Curriculum Learning-based Direct Preference Optimization (https://arxiv.org/abs/2502.18532)
- **What's New**: 이 논문에서는 Automated Theorem Proving (ATP)에 Direct Preference Optimization (DPO)을 혁신적으로 적용한 새로운 접근 방식인 Curriculum Learning-based DPO Iterative Theorem Proving (CuDIP) 방법을 소개합니다. 기존의 LLM 기반 ATP 방법들이 인간의 선호도와의 정렬에 한계를 가진 것을 극복하기 위해, DPO를 사용하여 이 문제를 해결하고자 합니다. 특히, 고품질의 선호 데이터 부족을 극복하기 위해 LLM과 기존의 정리 데이터(Theorem Proving Data)를 활용하여 선호 데이터의 다양성을 향상시키는 방법을 제안합니다.

- **Technical Details**: CuDIP 방법은 LLM을 이용하여 선호 데이터를 구성하고, 이러한 데이터를 활용하여 Curriculum Learning과 결합하여 반복적으로 정리 모델을 미세 조정합니다. 새로운 선호 데이터 구성 방법은 인간의 선호 주석에 대한 의존도를 줄이는 동시에, 다양한 선호 데이터를 확보할 수 있는 방법을 제시합니다. 이러한 접근은 LLM을 ATP 작업에 보다 효과적으로 적합하게 만들기 위한 중요한 단계로 자리잡고 있습니다.

- **Performance Highlights**: MiniF2F와 ProofNet 데이터셋에서의 실험 결과는 제안된 방법이 기존의 ATP 접근 방식에 비해 유의미한 성능 향상을 가져왔음을 보여줍니다. CuDIP 방법은 정리 증명 모델의 승인 효과를 높여주며, 특히 DPO 기법이 여러 작업에서 긍정적인 영향을 미친다고 보고하고 있습니다. 이는 향후 LLM을 활용한 ATP 연구에 새로운 방향성을 제시하는 중요한 결과입니다.



### Enhancing Hepatopathy Clinical Trial Efficiency: A Secure, Large Language Model-Powered Pre-Screening Pipelin (https://arxiv.org/abs/2502.18531)
Comments:
          30 pages, 5 figures

- **What's New**: 이번 연구는 복잡한 간 질환(hepatocellular carcinoma 및 간경변증) 관련 코호트 모집을 위한 새로운 환자 전수 screening 시스템을 개발하였습니다. 이 시스템은 대규모 언어 모델을 활용하여 기존의 시간 소모적인 수작업 screening 방식의 문제를 해결하고자 합니다. 특히, AI를 이용한 전수 screening 으로 정확성, 효율성 및 데이터 프라이버시 관련 문제를 다루고 있습니다.

- **Technical Details**: 개발된 파이프라인은 복잡한 기준을 일련의 복합 질문으로 나누고, 전자건강기록(electronic health records)를 통해 두 가지 전략으로 의미적 질문 응답(question-answering)을 수행합니다. 첫 번째 경로(Pathway A)는 인류화된 전문가의 사고 체계(Chain of Thought) 전략을 사용하고, 두 번째 경로(Pathway B)는 에이전트 협업(Agent Collaboration) 내에서의 사전 설정된 입장(Preset Stances)을 적용하여 복잡한 임상 추론 시나리오를 관리합니다. 이 파이프라인은 정밀도(precision), 시간 소모(time consumption), 그리고 반사실적 추론(counterfactual inference) 세 가지 주요 지표로 평가되었습니다.

- **Performance Highlights**: 이 연구의 결과, 파이프라인은 높은 정밀도(0.921)와 효율성(작업당 0.44초)을 달성했습니다. Pathway B는 복잡한 추론에서 뛰어난 성능을 보였고, Pathway A는 신속한 데이터 추출에 효과적이었습니다. 두 경로 모두 유사한 정밀도를 기록하며, 간세포암(0.878) 및 간경변 시험(0.843)에서도 유망한 결과를 보여주었습니다.



### Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models (https://arxiv.org/abs/2502.19417)
- **What's New**: 이번 논문에서는 복잡한 자연어 명령과 피드백을 해석하고 행동할 수 있는 로봇 지능을 향상시키기 위한 시스템을 소개합니다. 시스템은 vision-language models (VLMs)를 활용하여 계층적으로 복잡한 프롬프트와 사용자 피드백을 처리합니다. 이러한 기능을 통해 단순 명령 실행뿐만 아니라 동적 상황에서의 피드백 반영도 가능하도록 구성되었습니다. 결과적으로 이 시스템은 기존의 지침 따르기 방식과는 다르게 복잡한 작업 환경에서도 효율적으로 기능할 수 있습니다.

- **Technical Details**: 제안하는 Hi Robot 프레임워크는 VLM을 활용하여 고수준의 추론과 저수준의 작업 실행을 모두 지원합니다. 로봇은 low-level action을 위해 atomic commands를 생성하는 VLM을 결합하여 사용자와의 상호작용을 기반으로 복잡한 명령을 처리합니다. 로봇은 가상의 프롬프트와 피드백을 반영하여 다양한 상황에서 명령을 수행하는 능력을 갖추도록 훈련됩니다. 이 시스템은 이전 연구들과의 차별성을 보이며, 복잡한 프롬프트 및 피드백 처리가 가능하도록 설계되었습니다.

- **Performance Highlights**: Hi Robot은 여러 로봇 플랫폼에서 평가되었으며, 복잡한 작업 수행에서 우수한 성능을 보여주었습니다. 이 시스템은 인간의 의도와의 정렬 및 작업 성공률에서 이전의 여러 접근을 능가하는 결과를 기록했습니다. 평가 동안 로봇은 청소, 샌드위치 만들기, 장보기 등 다양한 작업을 수행하며, 복잡한 프롬프트에 적합하게 반응할 수 있는 능력을 나타냈습니다. 이는 인공지능 기반의 로봇이 인간과의 공생을 보다 직관적이고 능동적으로 실현할 수 있음을 증명합니다.



### Norm Growth and Stability Challenges in Localized Sequential Knowledge Editing (https://arxiv.org/abs/2502.19416)
Comments:
          Accepted for Oral Presentation at KnowFM @ AAAI 2025. arXiv admin note: text overlap with arXiv:2502.01636

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 지식 편집(knowlwdge editing)에서 국소 업데이트(localized updates)의 영향을 조사합니다. 지식 편집은 특정 사실을 수정하는 작업으로, 모델의 전반적인 능력은 그대로 유지하면서 특정 세부정보를 추가하거나 변경하는 데 중점을 둡니다. 연구는 다양한 후처리 개입(post-training interventions) 방법을 통해 업데이트된 행렬의 Frobenius norm이 항상 증가한다는 것을 보여줍니다.

- **Technical Details**: 연구에서 살펴본 개입 방법으로는 지속적 전처리(Continuous Pre-training), 전체 세부 조정(Full Fine-tuning), LORA 기반 세부 조정(LORA-based Fine-tuning)이 포함됩니다. 지속적 전처리는 특정 도메인의 대규모 텍스트 데이터로 모델의 기본 지식을 확장하는 것을 의미하며, 전체 세부 조정은 특정 과업에 대해 모델의 매개변수를 최적화하는 훈련 방식입니다. 연구 결과, 이러한 후처리 개입 동안 행렬 업데이트가 이루어질 때마다 Norm이 증가하는 현상이 관찰되었습니다.

- **Performance Highlights**: 국소 업데이트에서는 업데이트된 부분의 Norm이 비례적으로 증가하며, 이로 인해 모델 전체 안정성이 손상되고 성능 저하가 발생할 수 있습니다. 논문에서는 2000회의 업데이트를 통해 GPT2-XL과 GPT-J 모델에서 다양한 모델 편집 방법이 사용된 결과를 분석하였으며, 이 과정에서 성능 저하가 나타나기 시작했습니다. 업데이트된 활성화의 Norm은 지속적으로 감소하는 경향이 있으며, 이는 편집된 모델이 이전 모델과 다른 표현 공간의 영역을 점유하게 됨을 나타냅니다.



### Project Alexandria: Towards Freeing Scientific Knowledge from Copyright Burdens via LLMs (https://arxiv.org/abs/2502.19413)
Comments:
          Technical Report

- **What's New**: 이 논문은 과학 지식을 보호하는 저작권 법의 법적 및 기술적 장점을 강조하며, LLMs(대형 언어 모델)를 사용해 학술 문서를 'Knowledge Units'로 변환할 수 있는 새로운 접근 방식을 제안합니다. 'Knowledge Units'는 스타일적 요소 없이 정보만을 캡처하여, 저작권 우려 없이 과학적 지식을 공유할 수 있는 법적 근거를 제공합니다. 이 방법은 연구자들이 중요한 사실을 재사용할 수 있도록 해줍니다.

- **Technical Details**: Knowledge Units(KUs)는 개별段락(단락)에서 엔터티, 관계, 속성을 추출하여 구조화된 데이터를 생성합니다. 각 KU는 원문에서 추출된 짧은 텍스트 단편을 기반으로 하며, 개념 간의 연결과 특성을 포함하고 있습니다. LLM을 사용하여 처리하기 때문에, 원문을 복사하지 않고도 중요한 사실이 저장됩니다.

- **Performance Highlights**: 실험을 통해 Knowledge Units를 사용하여 다수의 질문-응답 실험을 수행한 결과, 연구 분야에 따라 95% 이상의 정보 보존율을 기록하였습니다. 이는 LLM이 Knowledge Units를 통해 제공된 데이터로도 원 텍스트와 유사한 퍼포먼스를 보인다는 것을 보여줍니다. 이러한 접근 방식은 연구자들 사이의 정보 접근성을 높이고, 과학적 대화를 촉진하는 데 기여할 것으로 기대됩니다.



### Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs (https://arxiv.org/abs/2502.19411)
Comments:
          Project Repo: this https URL

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)에서 코드와 추론이 서로 강화하는 방법을 탐구합니다. 특히 코드가 어떻게 구조화된 매개체로 기능하며 LLM의 추론 능력을 향상시키는지, 그리고 추론의 발전이 코드 지능에 어떤 영향을 미치는지를 조사합니다. 연구자는 이러한 상호작용의 중요성을 강조하고, LLM이 복잡한 소프트웨어 엔지니어링 작업을 수행하는 능력을 향상시키기 위해 코드와 추론의 시너지를 극대화할 필요성을 제기합니다.

- **Technical Details**: 코드는 추론 과정에서 실행 가능한 경로를 제공하고, 논리적 분해(logical decomposition)를 강제하며, 런타임 검증(runtime validation)을 가능하게 합니다. 연구자는 코드 표현이 LLM의 추론에 미치는 영향, 고급 추론 능력이 코드 지능 시스템을 어떻게 재구성하는지, 그리고 코드와 추론의 상호작용에서 발생하는 주요 도전 과제를 다룹니다. 이를 위해 LLM이 코드로 문제를 구조화하고 결과를 검증하는 방식을 분석하고, 향상된 추론 능력이 코드 지능의 경계를 확장하는 방법을 탐구합니다.

- **Performance Highlights**: 최근의 솔루션들은 복잡한 문제 해결 능력을 향상시키기 위한 코드 생성 기법을 도입했습니다. 예를 들어, Program of Thoughts(PoT)와 Program-aided language models(PaL)를 통해 수치 문제를 코드 생성으로 바꾸어 해결할 수 있습니다. 또한, 코드 기반 솔루션이 자연언어 솔루션보다 모델의 정확성을 향상시킨다는 연구 결과가 있으며, 실세계의 애매모호한 작업에서 LLM이 일관된 추론을 유지하는 데 어려움을 겪는 경우도 관찰되었습니다.



### Less or More: Towards Glanceable Explanations for LLM Recommendations Using Ultra-Small Devices (https://arxiv.org/abs/2502.19410)
- **What's New**: 최근의 연구는 대형 언어 모델(LLM)이 개인 AI 어시스턴트로써 일상적인 행동을 추천하는 데 뛰어난 가능성을 보여주고 있음을 강조합니다. 하지만 기존의 LLM에서 제공하는 설명이 비디오스페이스(흔히 사용되는 기기 화면의 제한된 공간)에서 효과적으로 전달되기 어렵다는 점이 문제로 제기되었습니다. 이런 문제를 해결하기 위해 LLM의 설명 텍스트를 공간적으로 구조화하고 신뢰성에 기반하여 시간에 따라 조정된 설명을 제공하는 방안을 탐구했습니다. 사용자 연구 결과, 구조화된 설명이 사용자 경험에 긍정적인 영향을 미쳤으나 세부 사항의 부족으로 인해 만족도는 낮게 나타났습니다.

- **Technical Details**: LLM의 설명을 공간적으로 구조화하기 위해, 사용자와 상황에 맞는 네 가지 구성 요소인 [활동], [대상], [위치], [목표]를 사용하여 간결하고 직관적인 아이콘으로 변환했습니다. 또, 레코멘데이션 신뢰성에 따른 상시 구조화된 설명의 표시 방법과 사용자가 요청할 때만 시각화된 설명을 보여주는 방법을 적용했습니다. 사용자 연구는 44명의 참가자를 대상으로 진행되어, 다양한 조건 하에 구조화된 설명과 비구조화된 설명의 효과를 비교하여 좌담회와 설문 조사를 통해 피드백을 수집했습니다.

- **Performance Highlights**: 구조화된 설명은 사용자의 행동 선택 시간을 단축시키고 인지 부담을 줄이는 데 효과적이었습니다. 또한, 항상 표시되는 구조화된 설명은 사용자들이 AI 추천을 수용하는 데 도움을 주었습니다. 하지만 구조화된 설명은 비구조화된 설명에 비해 사용자 만족도를 저하시켰으며, 적응형으로 제시된 구조화된 설명은 AI에 대한 사용자 인식을 개선하는 데 덜 효과적이었습니다. 이러한 결과는 사용자와의 인터뷰를 통해 내용을 개인화하고 타이밍을 최적화할 때 고려해야 할 설계 암시를 제시합니다.



### Multi-modal Contrastive Learning for Tumor-specific Missing Modality Synthesis (https://arxiv.org/abs/2502.19390)
- **What's New**: 이번 연구에서는 다중 모달리티 자기 공명 영상(MRI)에서 누락된 모달리티 이미지를 생성하기 위한 새로운 생성 모델을 제안했습니다. 이 모델은 환자의 관절 운동 아티팩트, 시간 제약 및 높은 비용과 같은 문제를 해결하기 위해 다중 모달 대비 학습(multi-modal contrastive learning)을 통합하여, 종양 영역에 초점을 맞췄습니다. 또한, 이 접근법은 모달리티 이미지를 생성할 뿐만 아니라 동시적으로 세분화(segmentation) 결과를 예측하는 기능을 갖추고 있습니다.

- **Technical Details**: 제안된 모델은 다중 모달 번역 네트워크(multi-modal translation network)와 세분화 디코더(segmentation decoder)를 결합하여 다양한 출처 모달리티로부터 학습하는 능력을 강화합니다. 특히, 각 모달리티에 맞춘 다중 브랜치 인코더(multi-branch encoder)와 주목(attention) 모듈을 통해 입력 이미지의 구조적 정보를 집중적으로 학습합니다. 컨트라스트 학습(contrastive learning)과 추가적인 자기 표현 손실(self-representation loss)을 결합하여, 전체 과정에서 표적 정보(target-specific information)를 효과적으로 전달합니다.

- **Performance Highlights**: Brain MR Image Synthesis 챌린지에서 제안한 모델은 높은 품질의 누락된 모달리티 이미지를 생성하며, 효과적인 성능을 보여주었습니다. 이 연구는 기존의 GAN 기반 방법들을 확장하여 제공된 다중 출처 모달리티로부터 종양 영역을 정확하게 생성하는 데 중점을 두었습니다. 최종적으로, 이 새로운 접근법은 임상적 진단 및 치료의 정확성을 더욱 높이는 데 기여할 것으로 기대됩니다.



### Efficient 4D fMRI ASD Classification using Spatial-Temporal-Omics-based Learning Framework (https://arxiv.org/abs/2502.19386)
Comments:
          Accepted at 2025 IEEE International Symposium on Biomedical Imaging (ISBI)

- **What's New**: 본 연구에서는 자폐 스펙트럼 장애(ASD)의 분류를 위해 새로운 효율적인 공간-시간-오믹스(Spatial-Temporal Omics) 학습 프레임워크를 제안합니다. 이를 통해 4D 뇌 영상 데이터에서 공간적 및 시간적 특징을 효과적으로 추출하여 ASD와 일반 대조군을 구별할 수 있습니다. 기존 방법들의 한계를 극복하며, 본 연구는 기존의 기능적 연결성 기반 방법과 무리없이 통합할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 STO 프레임워크는 두 가지 상호 보완적인 오믹스를 결합하여 작동합니다. 첫 번째는 공간-시간 인터-복셀 오믹스(Spatial-Temporal Inter-Voxel Omics)로, 4D fMRI 데이터에서 각 복셀의 3D 시간 도메인 파생물을 추출하여 상세한 공간-시간 특성을 유지하며, 두 번째는 공간-시간 인터-리전 오믹스(Spatial-Temporal Inter-Regional Omics)로, 기능적 연결성 특징을 이용하여 지역 수준의 정보를 생성합니다. 이러한 특징들은 이후 결합되어 ASD 분류를 위한 최종 예측을 생성하는데 사용됩니다.

- **Performance Highlights**: STO 프레임워크는 ABIDE 데이터셋을 이용한 광범위한 실험을 통해 기존 방법들에 비해 우수한 성능을 보였으며, 계산 효율성을 유지합니다. 연구 결과는 ASD 분류에서 중요한 통찰력을 제공하며, 분류 성능의 현저한 개선을 입증하였습니다. 향후 ASD 연구에 큰 기여를 할 것으로 기대됩니다.



### Preference-Based Gradient Estimation for ML-Based Approximate Combinatorial Optimization (https://arxiv.org/abs/2502.19377)
Comments:
          Preliminary work, under review

- **What's New**: 이 논문은 조합 최적화(Combinatorial Optimization, CO) 문제를 해결하기 위한 새로운 데이터 기반 접근 방식을 제안합니다. 기존의 비학습 근사 알고리즘을 개선하기 위해, 그래프 신경망(Graph Neural Network, GNN)을 활용하여 최적의 솔루션을 도출할 수 있는 파라미터 값을 예측합니다. 이 과정에서, 우리는 선호 기반 기울기 추정(preference-based gradient estimation)이라는 새로운 기법을 도입하여 GNN과 근사 알고리즘의 강점을 결합하려 합니다.

- **Technical Details**: 제안된 방법은 GNN이 입력 그래프에 따라 근사 알고리즘을 파라미터화하며, 비학습적 근사 알고리즘이 예측한 파라미터를 바탕으로 실현 가능한 솔루션을 변환하도록 합니다. GNN은 학습 데이터셋의 정보를 활용하여 더 나은 성능을 구현하며, 근사 알고리즘은 이 정보를 바탕으로 훈련 시 다음 단계에서 동일한 파이프라인을 사용할 수 있게 합니다. 이러한 설정은 기존의 여러 기울기 추정 기법을 사용하며, 새로운 기법인 PBGE를 통해 학습 효율성을 높입니다.

- **Performance Highlights**: 두 가지 일반적인 조합 최적화 문제, 즉 여행하는 판매원 문제(travelling salesman problem)와 최소 k-컷 문제(minimum k-cut problem)에 대해 실험을 수행하여 제안된 방법이 최신 학습 CO 해결기와 경쟁할 수 있음을 보여줍니다. 결과적으로, 기존의 비학습 근사 알고리즘과 GNN의 결합은 실질적인 상황에서의 성능 향상으로 이어지며, 결과적으로 우리가 제안한 기법이 조합 최적화 문제에 대한 전반적인 기여를 이루는 것으로 평가됩니다.



### DataMan: Data Manager for Pre-training Large Language Models (https://arxiv.org/abs/2502.19363)
Comments:
          ICLR2025 paper

- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 성능이 데이터 스케일링 법칙(data scaling laws)에 의해 어떻게 향상되는지를 다루고 있습니다. 이는 사전 학습(pre-training) 데이터의 선택 중요성을 강조하며, 기존의 방법들이 제한된 휴리스틱(heuristics)과 인간 직관에 의존하고 있다는 점을 지적합니다. 연구팀은 LLM이 스스로 성능에 유익한 기준을 파악하도록 유도하는 '역발상(reverse thinking)'을 통해 새로운 방향성을 제시합니다.

- **Technical Details**: 연구는 텍스트 당혹감(perplexity, PPL)과 관련된 14가지 품질 기준(quality criteria)을 도출하고, 도메인 혼합(domain mixing)을 지원하기 위해 15가지 일반 응용 분야(application domains)를 소개합니다. 그리고 데이터 매니저(Data Manager, DataMan)를 훈련시켜 포인트 기반 평가(pointwise rating)로부터 품질 평가 및 도메인 인식을 학습하게 하였습니다. 이를 통해 447B 토큰을 포함하는 사전 학습 말뭉치를 14가지 품질 평가와 도메인 유형标注으로 주석 처리하였습니다.

- **Performance Highlights**: 실험 결과, DataMan을 사용하여 30B 토큰으로 1.3B 매개 변수를 가진 언어 모델을 훈련시킨 결과, 인-컨텍스트 학습(in-context learning, ICL), 당혹감(p perplexity), 그리고 지시 따라가기 능력에서 현저한 개선을 보여주었습니다. 전체 점수(l=5) 기준으로 최상의 성능을 보이는 모델은 균일 샘플링을 사용해 50% 더 많은 데이터로 훈련된 모델을 능가했습니다. 연구팀은 과세가 높은 도메인 특정 데이터로 추가 사전 학습을 진행하여 도메인 특정 ICL 성능을 향상시키고 DataMan의 도메인 혼합 능력을 검증하였습니다.



### Physics-Based Hybrid Machine Learning for Critical Heat Flux Prediction with Uncertainty Quantification (https://arxiv.org/abs/2502.19357)
Comments:
          Submitted to the International Journal of Heat and Mass Transfer

- **What's New**: 이 연구는 머신러닝(machine learning)과 물리 기반 모델을 결합한 불확실성 인식 하이브리드 모델링 접근 방식을 개발하고 검증했습니다. 이 방법은 핵 반응로의 임계 열 유속(critical heat flux, CHF) 예측에 사용되며, 고갈(dryout) 상황에서의 성능을 평가합니다. 세 가지 머신러닝 불확실성 정량화 기술인 딥 뉴럴 네트워크 앙상블(deep neural network ensembles), 베이지안 뉴럴 네트워크(Bayesian neural networks), 깊은 가우시안 프로세스(deep Gaussian processes)를 사용하여 실험하고 있으며, 순수 머신러닝 모델과 비교하고 있습니다.

- **Technical Details**: 후보 모델에서는 Biasi 및 Bowring의 두 가지 경량 상관 관계(empirical correlations)를 적용하였고, 훈련 데이터의 풍부함과 한정됨에 따라 모델의 성능 및 불확실성을 분석했습니다. 성능 평가는 패리티 플롯(parity plots), 불확실성 분포(uncertainty distributions), 보정 곡선(calibration curves)을 통해 수행되었습니다. 또한, 하이브리드 모델들이 순수 머신러닝 모델보다 더 나은 성능을 보였으며 데이터 부족에 대한 저항성을 입증했습니다.

- **Performance Highlights**: Biasi 하이브리드 딥 뉴럴 네트워크 앙상블은 평균 절대 상대 오차(mean absolute relative error) 1.846% 및 안정적인 불확실성 추정치를 기록하며 가장 유리한 성능을 나타냈습니다. 베이지안 뉴럴 네트워크 모델은 약간의 오류 및 불확실성이 있었으나, 보정에서는 우수한 성능을 보였습니다. 반면에 깊은 가우시안 프로세스 모델은 대부분의 지표에서 저조한 성능을 보였습니다. 모든 하이브리드 모델은 순수 머신러닝 설정보다 뛰어난 결과를 보였습니다.



### Deep Learning-Based Transfer Learning for Classification of Cassava Diseas (https://arxiv.org/abs/2502.19351)
Comments:
          12 pages, in Portuguese language, 3 figures

- **What's New**: 이번 논문은 카사바(Cassava) 질병 이미지를 분류하기 위한 네 가지 Convolutional Neural Network (CNN) 아키텍처(EfficientNet-B3, InceptionV3, ResNet50, VGG16)의 성능을 비교합니다. 비대칭 데이터셋을 기반으로 적절한 메트릭(metrics)을 사용하여 클래스 불균형 문제를 해결했습니다. 연구 결과 EfficientNet-B3는 87.7%의 정확도(accuracy), 87.8%의 정밀도(precision), 87.8%의 재현율(recall) 및 87.7%의 F1-Score를 기록하며 디지털 농업 분야에 유용한 도구가 될 수 있음을 제안합니다.

- **Technical Details**: 이 논문에서는 CNN 아키텍처를 사용해 카사바 질병의 자동 감지를 위한 방법론을 제시합니다. 특히, Bacteriose da Mandioca (CBB), Doença da Estria Marrom (CBSD), Doença do Mosaico (CMD) 및 Vírus Mosqueado Verde (CGM) 네 가지 질병을 식별하기 위해 여러 CNN 아키텍처를 비교합니다. 모델은 주로 Deep Learning(딥러닝) 기법을 기반으로 하며, Transfer Learning(전이 학습) 기법도 사용하여 더욱 효율적인 결과를 도출하고자 합니다.

- **Performance Highlights**: EfficientNet-B3 아키텍처는 다른 CNN 모델들과 비교하여 가장 높은 성능을 보여주었으며, 데이터셋의 클래스 불균형 문제를 해결하기 위한 다양한 메트릭을 적용하여 보다 정확한 진단이 가능함을 입증했습니다. 실험에서 사용된 서버는 Intel i7 CPU와 NVIDIA GTX 1080 Ti GPU 두 개로 구성되어 있으며, Python 프로그래밍 언어와 PyTorch 프레임워크를 사용하여 CNN 모델을 훈련했습니다. 연구 결과는 카사바 질병의 조기 진단 및 효과적인 농업 관리에 기여할 것으로 기대됩니다.



### Controlled Diversity: Length-optimized Natural Language Generation (https://arxiv.org/abs/2502.19347)
Comments:
          ISCA/ITG Workshop on Diversity in Large Speech and Language Models

- **What's New**: 이번 논문에서는 LLMs(대규모 언어 모델)가 엄격한 길이 요구사항에 따라 출력 길이를 조정할 수 있는 방법을 제시하고 있습니다. 이 요구사항을 충족하는 응답을 생성하기 위한 모델을 훈련시키기 위해 데이터 증강(data augmentation)과 기존의 파인튜닝(fine-tuning) 기법을 활용합니다. 이러한 접근 방식은 LLM이 다양한 사용자와 시스템 요구사항에 맞춰 더 유용하게 적용될 수 있도록 합니다.

- **Technical Details**: 우리는 LLM이 길이 요구사항을 준수하도록 훈련시키기 위해 Supervised Fine-Tuning(SFT) 외에도 Proximal Policy Optimization(PPO), Direct Preference Optimization(DPO), Odds Ratio Preference Optimization(ORPO)와 같은 여러 강화 학습(reinforcement learning) 기법을 적용합니다. 이러한 방법들은 LLM이 인간 피드백(human feedback)에서 학습할 수 있도록 지원하며, 자동으로 측정된 성능 지표를 최적화하는데도 활용됩니다. 또한 단일 훈련 데이터 세트를 사용하여 길이를 기준으로 측정이 가능하도록 하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방식으로 훈련된 LLM은 기존의 기준 모델에 비해 길이 요구사항을 더 잘 준수하는 텍스트를 생성할 수 있음을 보여줍니다. 길이 목표를 보조 목표로 추가하고, LLM이 주 작업을 학습하면서 길이를 조정할 수 있는 가능성을 확인했습니다. 이는 예를 들어 사용자 지정 길이의 요약을 간소화하여 복잡한 콘텐츠 접근성을 향상시키는 응용 사례에서도 유망한 결과를 보여주고 있습니다.



### Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems (https://arxiv.org/abs/2502.19328)
Comments:
          16 pages, 5 figures

- **What's New**: 이 논문에서는 agentic reward modeling을 제안하며, 이는 기존의 보상 모델(reward models)과 검증 가능한 정확성 신호(verifiable correctness signals)를 통합한 새로운 보상 체계입니다. 특히, 사실성(factuality) 및 지침 준수(instruction following)와 같은 두 가지 신호를 강조하여 보다 신뢰할 수 있는 보상을 제공합니다. 기존의 RMs는 주로 인간의 선호(human preferences)에 초점을 맞추어 주관적 편향(subjective biases)을 가질 수 있는 반면, 이 방법론은 다양한 측면에서의 정확성 신호를 반영하여 높은 신뢰성을 유지합니다.

- **Technical Details**: RewardAgent라는 보상 에이전트를 구현하여 인간 선호 기반의 기존 보상 모델을 사실성과 지침 준수 신호와 결합했습니다. RewardAgent는 세 가지 주요 모듈로 구성되어 있습니다: Router, Verification Agents, Judger. Router는 적절한 검증 에이전트를 결정하고, Verification Agents는 다양한 측면에서 응답의 정확성을 평가하며, Judger는 이들 검증 신호와 인간 선호 점수를 통합하여 최종 보상 점수를 제공합니다.

- **Performance Highlights**: RewardAgent는 RM-Bench, JudgeBench와 같은 여러 보상 모델 벤치마크에서 상당한 성능 향상을 보였으며, 이를 통해 전통적인 보상 모델에 비해 더 우수한 응답을 선택하는 능력을 입증했습니다. 또한, 실제 세계의 다운스트림 작업에서 주요 NLP 벤치마크에서 뛰어난 성능을 달성하여, RewardAgent로 구축된 데이터를 사용하는 경우 DPO 훈련에 따라 LLM의 성능이 향상됨을 보여주었습니다. 전반적으로, 이 연구는 보다 신뢰할 수 있는 보상 시스템 개발을 위한 새로운 가능성을 제시합니다.



### Partition Tree Weighting for Non-Stationary Stochastic Bandits (https://arxiv.org/abs/2502.19325)
- **What's New**: 이 논문은 행동과 관찰이 상호 연관된 데이터 스트림에 대해 일반화된 보편 소스 코딩(universal source coding)을 다룹니다. 목표는 보편적이면서 제어 정책(control policy)으로 사용될 수 있는 코딩 분포를 구축하는 것입니다. 일반 에이전트와의 상호작용을 기술하는 좋은 보편 코딩 스킴을 만들면, 이러한 분포에서 샘플링하여 제어 정책을 생성할 수 있습니다.

- **Technical Details**: 이 논문에서는 비정상 확률적 베르누이 밴딧 문제(non-stationary stochastic Bernoulli bandit problem)와 관련하여 파르티션 트리 가중치(Partition Tree Weighting) 기법을 활용하여 효율적이고 높은 성능을 보이는 알고리즘을 제시합니다. 비정상 소스(non-stationary sources)에 대한 보편 소스 코딩 문헌은 이미 잘 개발되어 있으며, 특정 구간에서 베이esian 모델 평균(Bayesian model averaging)을 통해 모든 가능한 데이터 시퀀스의 파르티션을 고려합니다. 논문은 Veness et al. (2013)의 기법을 확장하여 비정상 확률적 밴딧 환경에서의 제어 문제에 적용합니다.

- **Performance Highlights**: 새로운 접근 방식은 일반적인 Thompson Sampling(TS) 방법을 일반화한 것으로, 다양한 재시작 구성에 대한 Bayesian 추론을 수행합니다. 이 연구에서는 알고리즘의 최악의 경우 잉여( redundancy) 및 실험적 성능을 조사하며, 그 결과가 현재의 최첨단 알고리즘과 비교하여 우수하다는 것을 보여줍니다. 이렇게 개발된 알고리즘은 여러 환경에서 잘 작동하는 에이전트를 구축하는 데 있어 중요한 기여를 할 것으로 기대됩니다.



### Shh, don't say that! Domain Certification in LLMs (https://arxiv.org/abs/2502.19320)
Comments:
          10 pages, includes appendix Published in International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 도메인 인증(domain certification) 프레임워크를 소개합니다. 이 프레임워크는 모델이 특정 도메인에서 벗어난 출력을 생성할 확률을 수학적으로 보장합니다. 또한, VALID라는 알고리즘을 제안하여 공격(adversarial attack) 하에서 모델이 주제에서 벗어나지 않도록 하는 저렴한 방법으로 검증된 적대적 LLM 출력을 생성합니다.

- **Technical Details**: 도메인 인증 프레임워크는 모델이 주어진 대상 도메인을 넘어서는 출력을 할 확률에 대한 상한을 제시합니다. VALID는 이 보장을 준수하는 시스템을 구축하는 데 사용하는 간단한 방법입니다. 이 방법에서는 입력과 출력의 토큰(token) 시퀀스를 수학적으로 정의하고, 이를 통해 이론적 근거를 통해 도메인 인증 성과를 이끌어내는 기초 요소를 제시합니다.

- **Performance Highlights**: VALID는 다양한 데이터셋에서 평가되어 의미 있는 인증서를 생성함으로써, 도메인 외 샘플이 발생할 확률을 최소한의 거부 행동(refusal behavior)으로 조정할 수 있다는 것을 보여주었습니다. 연구 결과, VALID는 공격에 대한 강력한 저항력을 제공하며, LLM 기반 시스템의 도메인 제한 필요성을 강조합니다.



### FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users (https://arxiv.org/abs/2502.19312)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 Few-Shot Preference Optimization (FSPO)라는 새로운 프레임워크를 소개하며, 개인화를 위한 새로운 방법론을 제안합니다. FSPO는 보상 모델링을 메타 학습 문제로 재구성하여 사용자의 몇 가지 라벨이 붙은 선호도를 통해 빠르게 적응하는 LLM의 능력을 활용합니다. 이 연구는 공공의 LLM로부터 1백만 개 이상의 합성 개인화 데이터를 생성하여 실제 사용자에 적응할 수 있는 기반을 마련했습니다.

- **Technical Details**: FSPO는 사용자 세부정보의 연쇄적 사고 과정(Chain-of-Thought, COT)을 활용하여 성능을 향상시키며, 이러한 방식은 개인화된 정보 생성을 가능하게 합니다. 이 과정에서 기존의 채점 방식과는 달리, 모델은 보상 기능을 개인화하고, 개별 사용자 집단에 맞춘 다양한 반응 생성을 가능하게 합니다. 모델 학습은 합성 데이터와 실제 데이터 사이에서의 전이(Transfer) 특성을 고려하여 다양성과 일관성을 중시합니다.

- **Performance Highlights**: FSPO는 영화 리뷰, 교육 배경에 따른 교육적 적응, 일반적인 질문 응답 분야에서 합성 사용자 1,500명을 대상으로 한 평가에서 평균 87%의 Alpaca Eval 승률을 기록했습니다. 또한, 실제 사용자에 대한 개방형 질문 응답에서도 72% 승률을 달성하여 FSPO의 개인화된 모델이 효과적임을 입증했습니다. 이러한 결과는 FSPO가 사용자 요구에 더 잘 부응할 수 있는 가능성을 보여줍니다.



### Faithful Logic Embeddings in HOL -- A recipe to have it all: deep and shallow, automated and interactive, heavy and light, proofs and counterexamples, meta and object lev (https://arxiv.org/abs/2502.19311)
Comments:
          22 pages, 9 figures

- **What's New**: 이 논문은 심층 임베딩(Deep Embeddings)과 얕은 임베딩(Shallow Embeddings)을 고전 고차 논리(Classical Higher-Order Logic)에서 동시에 배치하는 기법을 제시합니다. 이러한 접근법은 상호작용 및 자동 정리 증명(Theorem Proving)과 반례 찾기(Counterexample Finding)에서 더 유연성을 제공합니다. 또한, 논리 임베딩 간의 자동적 충실성 증명을 가능하게 해줍니다.

- **Technical Details**: 본 논문에서는 고차 논리(HOL)를 메타 논리로 사용하여 심층 및 얕은 논리 임베딩을 연결하는 기술이 제시됩니다. 특히, 얕은 임베딩은 메타논리에 대한 의미 의존성의 정도에 따라 다르며, 최대 및 최소 임베딩의 극단적인 형태가 비교됩니다. 이와 같은 다양한 접근법은 논리 교육 및 최신 증명 보조기구(Proof Assistants)와 관련된 연구를 위한 여러 가능성도 함께 논의합니다.

- **Performance Highlights**: 최소 얕은 임베딩이 특정 추론 작업에 더 적합할 수 있다는 주장을 뒷받침하는 증거가 제시됩니다. 또한, PML(간단한 명제적 모달 논리)을 통해 연구의 다양한 실험이 진행됩니다. 이 논문은 Isabelle/HOL을 호스트 증명 시스템으로 사용했으며, 효과적인 증명 자동화를 위한 여러 도구들이 활용되었습니다.



### Anomaly Detection in Complex Dynamical Systems: A Systematic Framework Using Embedding Theory and Physics-Inspired Consistency (https://arxiv.org/abs/2502.19307)
- **What's New**: 본 논문에서는 복잡한 동적 시스템에서 이상 탐지(anomaly detection)를 위한 새로운 시스템 이론적 접근 방식을 제안합니다. 전통적인 임베딩 기법을 확장하여 시스템 동역학을 포착하는 방법을 제시하며, 물리적 일관성 원칙에 기반한 TDC-AE(Temporal Differential Consistency Autoencoder)를 개발합니다. 이를 통해 예측 유지보수(preventive maintenance)와 사이버 보안(cybersecurity) 모니터링의 중요성을 강조합니다.

- **Technical Details**: 이 연구는 임베딩 이론(classical embedding theory)과 물리학에서 영감을 받은 일관성 원칙을 바탕으로 하여, 동적 시스템의 수학적 기초를 탐구합니다. 잠재 공간(latent space)에 운동 시스템의 동역학을 매핑하기 위한 조건을 설정하며, 이를 통해 시스템 상태에서 발생하는 편차를 탐지하는 방법을 제안합니다. TDC-Loss 는 잠재 변수의 근사 미분을 동적 표현과 일치시키는 역할을 하여 시간적 일관성을 시행합니다.

- **Performance Highlights**: 제안된 TDC-AE 모델은 C-MAPSS 데이터셋을 사용하여 평가되었으며, LSTM 및 Transformer 모델보다 뛰어난 성능을 보였습니다. 또한 200배의 MAC(Multiply-Accumulate) 연산 감소를 달성하여 경량형 엣지 컴퓨팅(lightweight edge computing)에 적합한 특징을 가지고 있습니다. 이 결과는 이상이 안정적인 시스템 동역학을 방해한다는 가설을 지지하며, 이상 탐지에 있어 강력하고 해석 가능한 신호를 제공합니다.



### Corporate Fraud Detection in Rich-yet-Noisy Financial Graph (https://arxiv.org/abs/2502.19305)
- **What's New**: 본 연구는 기업의 재무 사기 탐지를 위한 새로운 접근 방식을 제안합니다. 기존의 방법들은 네트워크 내의 복잡한 상호작용을 효과적으로 통합하지 못하는 문제를 안고 있었습니다. 이를 해결하기 위해, 18년 동안의 재무 기록을 수집하여 세 가지 그래프 데이터 세트를 구성하고, ‘Knowledge-enhanced GCN with Robust Two-stage Learning’(KeGCN_R)이라는 새로운 그래프 기반 방법을 개발했습니다.

- **Technical Details**: 이 논문은 KeGCN_R이 두 가지 주요 도전 과제를 해결하는 방법을 설명합니다. 첫째, 정보 과부하 문제를 해결하기 위해, Knowledge Graph Embeddings(KGE) 방식을 활용하여 지원 노드에서 회사 노드로 유용한 정보를 증류합니다. 둘째, 숨겨진 사기 문제에 대처하기 위해, 베이즈 최적 분포를 기반으로 하는 새로운 강건한 이단계 학습 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, KeGCN_R은 여러 강력한 기준 모델보다 더 높은 재무 사기 탐지 효과와 강 robustness를 보여주었습니다. 또한, 이 방법은 정보 과부하 문제를 완화하고 숨겨진 사기에 대해 더 높은 성능을 발휘합니다. 이러한 결과는 기업 사기 탐지에서의 상호작용의 중요성을 명확히 드러냅니다.



### Combining Planning and Reinforcement Learning for Solving Relational Multiagent Domains (https://arxiv.org/abs/2502.19297)
- **What's New**: 이 논문은 Multiagent Relational Planning and Reinforcement Learning (MaRePReL)이라는 새로운 방법을 제안하여, 관계형 다중 에이전트 환경에서 효과적인 학습 및 일반화를 가능하게 합니다. 이 시스템은 최초로 다양한 객체와 관계를 다룰 수 있는 다중 에이전트 시스템으로서, 관계형 정보의 효과적인 표현 및 심층 강화 학습을 사용하여 높은 수준의 계획을 분해하고 다중 에이전트 학습을 처리합니다.

- **Technical Details**: MaRePReL은 중앙 집중식 제어 및 에이전트 배치를 위한 관계형 계층 계획자와 함께 작동하며, 이로 인해 높은 효율성을 발휘합니다. 이 접근 방식은 관계적 구조를 배경으로 다중 수준의 추상화(level of abstraction)를 탐색하고, 깊이 있는 강화 학습을 통해 하위 수준의 정책을 학습합니다. 또한, 본 방법은 기존 알고리즘의 한계를 극복하기 위해 도메인 지식을 활용합니다.

- **Performance Highlights**: MaRePReL은 몇몇 관계형 다중 에이전트 도메인에서 그 효과성과 일반화 능력을 입증하였으며, 이를 통해 다른 심층 RL 기반의 다중 에이전트 기준과 비교했을 때 우수성을 보여줍니다. 특히, 하위 작업 정보를 명시적으로 사용하는 접근 방식과 비교하여, 본 시스템은 더욱 높은 성능을 발휘하는 것으로 확인되었습니다.



### Integrating Biological and Machine Intelligence: Attention Mechanisms in Brain-Computer Interfaces (https://arxiv.org/abs/2502.19281)
- **What's New**: 본 논문은 EEG (Electroencephalography) 신호 분석에서 주목 메커니즘(attention mechanisms)의 중요성을 강조하며, 이를 기반으로 한 전통적 및 Transformer 기반 메커니즘의 포괄적 리뷰를 제공합니다. 이러한 메커니즘은 다중 모달 데이터 융합(multimodal data fusion)에 중점을 두어 BCI (Brain-Computer Interface) 응용 프로그램의 성능을 향상시키는 데 기여합니다. EEG의 시간, 주파수 및 공간 채널 전반에서의 변화를 포착함으로써, 주목 메커니즘은 특성 추출(feature extraction) 및 모델 강건성(model robustness)을 개선합니다.

- **Technical Details**: 전통적인 주목 메커니즘은 주로 합성곱 신경망(CNN) 및 순환 신경망(RNN)과 통합되어 사용되며, 다양한 정보 유형의 주목 가중치를 계산합니다. 이 모델은 다양한 EEG 신호에서 동일한 관점으로 공간적, 시간적, 주파수적 특성의 중요도를 동적으로 평가하여 관련 정보의 우선순위를 매깁니다. 주목 메커니즘의 주요 카테고리는 부드러운 소프트(attention)와 단단한 하드(attention) 메커니즘으로 나뉘며, 각각 다르게 가중치를 적용하고 최적화합니다.

- **Performance Highlights**: 주목 메커니즘은 EEG 데이터를 통해 기능적 분석을 강화하여 BCI 연구의 정확성과 강건성을 높입니다. 특히 주의 모듈은 특정 작업에 가장 관련성이 높은 EEG 채널을 우선 순위로 설정하여 노이즈를 줄이며, 이는 신경 디코딩의 정확성을 향상시킵니다. 이러한 접근 방식은 다중 모달 BCI 애플리케이션에서 잘 작동하며, 다양한 정보 융합을 통해 사용자 의도와 정신 상태를 보다 포괄적으로 이해할 수 있도록 돕습니다.



### Multiview graph dual-attention deep learning and contrastive learning for multi-criteria recommender systems (https://arxiv.org/abs/2502.19271)
- **What's New**: 본 연구에서는 다중 기준 추천 시스템(Multi-Criteria Recommender Systems, MCRS)을 위한 새로운 표현 방식을 제안합니다. 제안하는 방법은 다중 에지 이분 그래프(multi-edge bipartite graph)에 기반하며, 각 엣지는 사용자가 평가한 아이템의 기준 점수를 나타냅니다. 또한, Multiview Dual Graph Attention Networks (MDGAT)를 통해 사용자와 아이템 간의 복합적인 관계를 고려합니다. 이 접근 방식은 아이템의 다각적인 특성을 반영하는 데 도움을 줍니다.

- **Technical Details**: 연구에서는 각 뷰(view)에 대한 유사성을 기반으로 앵커 포인트(anchor point)를 정의하고, 로컬(local) 및 글로벌(global) 대조 학습을 적용하여 긍정 샘플과 부정 샘플을 구별합니다. 이 과정에서 그래프 주의 네트워크(Graph Attention Networks, GAT) 및 다중 뷰 그래프 주의 네트워크(MGAT)의 두 가지 주의 메커니즘을 활용하여 데이터의 복잡한 관계를 효과적으로 모델링합니다. MDGAT는 각 뷰의 특징을 중요하게 고려하여 정보 전파를 향상시키며, 사용자 및 아이템 간의 관계를 명확히 하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 방법은 실제 데이터셋에서 아이템 점수 예측의 정확성을 향상시키는 것으로 나타났습니다. 이렇게 개선된 성능은 기존 기준과 비교했을 때 우수한 효율성을 보여 주며, 로컬 및 글로벌 이웃 간의 관계를 효과적으로 포착하여 추천 시스템의 전반적인 성능을 높입니다. 이 연구는 MCRS 분야에서 향후 연구에 중요한 기초 자료를 제공할 것입니다.



### Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization (https://arxiv.org/abs/2502.19261)
Comments:
          To appear at the 13th International Conference on Learning Representations (ICLR 2025)

- **What's New**: 이번 연구는 Drop-Upcycling이라는 새로운 방법론을 제안하여 Mixture of Experts (MoE) 모델의 훈련을 최적화하였다. Drop-Upcycling은 사전 훈련된 밀집 모델의 지식을 이용하면서 일부 매개변수를 통계적으로 재초기화하는 접근 방식을 결합하여 전문가의 전문화(expert specialization)를 촉진한다. 이 방법은 기존의 Upcycling 접근 방식이 가진 한계를 극복하며 장기적으로 MoE 모델의 효율성을 크게 향상시킨다.

- **Technical Details**: Drop-Upcycling은 밀집 모델을 MoE 모델로 확장할 때 전문가의 feedforward network (FFN) 매개변수의 선택적 재초기화(selective re-initialization)로 동작한다. 구체적으로, FFN의 중간 차원에서 공통 인덱스를 임의로 샘플링하여 매개변수를 열(column)별 또는 행(row)별로 드롭한 후, 드롭된 매개변수를 통계치를 이용해 재초기화한다. 이 접근 방식은 훈련 초기 단계에서 더 나은 상태에서 시작할 수 있게 도와주며, 긴 훈련 기간 동안 관찰되는 수렴 느림(convergence slowdowns)을 피하도록 설계되었다.

- **Performance Highlights**: 드롭 업사이클링의 성과는 5.9B 활성 매개변수를 가지는 MoE 모델이 동일 모델 가족의 13B 밀집 모델과 유사한 성능을 나타내며, 훈련 FLOPs는 약 1/4에 불과하다는 점이다. 대규모 실험을 통해 드롭 업사이클링이 이전 MoE 구축 방법들과 비교했을 때 장기 훈련 시에도 눈에 띄게 더 우수한 성과를 보여주었다. 이 연구에 대한 모든 실험 결과는 공개되어 있어 재현성과 향후 연구에 기여할 수 있다.



### EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region (https://arxiv.org/abs/2502.19260)
Comments:
          19 pages, 6 figures

- **What's New**: 이번 논문에서는 아랍만 지역에서 수집된 최초의 공개 자율주행 데이터셋인 Emirates Multi-Task (EMT) 데이터셋을 소개합니다. EMT 데이터셋은 고유한 도로 토폴로지, 높은 교통 혼잡도, 다양한 보행자 복장과 기상 조건 등의 특성을 포착하고 있습니다. 30,000개 이상의 프레임과 570,000개의 주석이 추가된 바운딩 박스를 포함하며, 약 150킬로미터의 드라이빙 경로를 다룹니다.

- **Technical Details**: 이 데이터셋은 추적 (tracking), 궤적 예측 (trajectory forecasting), 의도 예측 (intention prediction)의 세 가지 주요 작업을 지원합니다. 각 벤치마크 데이터셋은 다중 에이전트 추적 실험, 궤적 예측 평가, 의도 예측 실험을 포함하여 각각의 평가 결과로 보완됩니다. EMT 데이터셋은 UAE에서 수집되었으며, 다양한 교통 장면과 특성을 포착하기 위해 차량과 보행자에 대한 주석이 포함되어 있습니다.

- **Performance Highlights**: 각 작업별 데이터셋은 평가 모델과 실험으로 보완됩니다. 다중 에이전트 추적에서는 Kalman 필터 기반의 최신 추적기 (SOTA trackers)를 평가하며, 궤적 예측에 대해서는 고밀도 시나리오에서의 시간적 의존성 및 상호작용 동역학을 포착하는 딥러닝 아키텍처를 사용합니다. 의도 예측 작업에서는 과거 궤적을 바탕으로 미래의 의도를 예측하는 LSTM 기반 모델의 성능을 평가합니다.



### Poster: Long PHP webshell files detection based on sliding window attention (https://arxiv.org/abs/2502.19257)
Comments:
          3 pages(include 1 page poster), 1 figure. Accepted as a poster at the NDSS this http URL list: this http URL. Dataset/code available at this http URL

- **What's New**: 이번 연구에서는 웹쉘(webshell) 탐지 방법을 제안합니다. PHP 소스 코드를 opcode로 변환한 후 Opcode Double-Tuples (ODTs)를 추출하여 새로운 탐지 모델을 생성하였습니다. 또한, 긴 파일 탐지의 어려움을 해결하기 위해 sliding window attention mechanism을 도입하였습니다. 이로 인해 기존 탐지 방법들이 갖는 문제를 효과적으로 해결하였습니다.

- **Technical Details**: 제안된 탐지 방법은 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 PHP 소스 코드를 ODT로 변환하고, 두 번째 단계에서는 CodeBert 모델과 FastText 모델을 결합하여 특징을 추출하고 이진 분류를 수행합니다. 데이터셋은 5001개의 웹쉘 샘플과 5936개의 정상 PHP 파일로 구성되어 있습니다. 실험 결과 ODT를 이용한 방법이 기존의 Opcode Single-Tuples(OSTs) 대비 4.6% 더 높은 정확도를 기록했습니다.

- **Performance Highlights**: 제안된 모델은 99.2%의 정확도와 99.1%의 F1 점수를 달성하며, 기존의 최신 웹쉘 탐지 방법보다 우수한 성능을 보였습니다. 특히, 웹쉘Pub(77.3% Acc), PHP Malware Finder(83.4% Acc), MSDetector(97.1% Acc)와 비교했을 때, 본 연구의 방법이 효과적임을 입증하였습니다. 향후 다양한 언어의 웹쉘 탐지 작업을 발전시킬 계획입니다.



### Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspectiv (https://arxiv.org/abs/2502.19255)
Comments:
          35 Pages

- **What's New**: 이번 논문은 인간 피드백을 활용한 강화 학습(RLHF)에서 샘플 효율성을 높이기 위한 새로운 접근 방안을 제시합니다. 기존 연구들이 샘플 효율적인 온라인 탐험 전략을 다룬 반면, 잘못 지정된 보상 모델을 활용하여 학습을 가속화하는 가능성에 대한 탐구가 부족했습니다. 이 논문에서는 불완전한 보상 모델에서 지식을 전이하는 방법을 체계적으로 연구하며, 이를 통해 RLHF의 샘플 효율성을 향상시키려 합니다.

- **Technical Details**: 제안된 이론적 전이 학습 알고리즘은 KL 정규화가 적용된 RLHF 목표의 새로운 속성을 기반으로 합니다. 주요 통찰력은 정책의 최적성 보장이 정책의 부분 최적성과 연결되어 있다는 것입니다. 이를 통해 전이 학습의 두 가지 원칙을 제시하며, 이는 정책 선택 및 오프라인 RL 이론에서 파생된 유용한 통찰력을 포함합니다.

- **Performance Highlights**: 제안된 Transfer Policy Optimization(TPO) 알고리즘은 불완전한 보상 모델을 빠르게 활용하여 초기 단계에서 낮은 레그렛(regret)을 달성합니다. 잔여 시간 동안에는 O~⁢(T)~𝑂𝑇	ext{에 기반한 보상 모델로부터 학습된 오프라인 정책이 구조적 복잡성과 독립적임을 입증했습니다. 추가적으로, T5 모델을 요약 작업에 맞춰 미세 조정하는 실험을 통해 제안된 방법의 효과를 입증하였습니다.



### GraphBridge: Towards Arbitrary Transfer Learning in GNNs (https://arxiv.org/abs/2502.19252)
Comments:
          10 pages, 3 figures, 6 tables, to be published in ICLR 2025

- **What's New**: 이번 연구에서는 Graph Neural Networks (GNNs) 용으로 설계된 새로운 프레임워크, GraphBridge를 소개한다. GraphBridge는 다양한 과제와 도메인 간의 지식 전달을 가능하게 하며, 작업 구성이나 그래프 구조의 수정을 필요로 하지 않는다. 이 구조는 원래 모델의 본질적인 지식을 보존하고, 임의의 차원의 출력 지원이 가능하게 설계되었다.

- **Technical Details**: GraphBridge 디자인은 예측 헤드와 입력을 출력 레이어에 연결하는 브리징 네트워크를 통합하고있다. 이를 통해 GNNs의 지식 재사용을 더욱 효율적으로 하며, 특수한 입력 및 출력 차원에 맞춰 조정 가능한 입력 차원 어댑터를 포함한다. 또한, Graph-Scaff-Side-Tune (GSST) 및 Graph-Merge-Side-Tune (GMST)라는 두 가지 그래프 측면 조정 기법을 통하여 부정적 전이 문제를 해결하고 있다.

- **Performance Highlights**: 다양한 전이 학습 시나리오에서 GraphBridge의 성능이 평가되었으며, 이 연구는 16개의 데이터셋에서 검증되었다. 결과적으로 GraphBridge는 이전의 기법들보다 자원 효율적인 전이 학습을 달성하며, 특히 도전적인 과제에서도 우수한 성과를 보였다. 이 방법은 튜닝할 수 있는 파라미터의 5%에서 20%만으로도 기존의 성능을 유지할 수 있음을 보여준다.



### Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases (https://arxiv.org/abs/2502.19249)
- **What's New**: 이 논문은 형식 언어(formal languages)에서 사전 훈련(pretraining)을 통해 자연어(natural language)의 습득을 효과적으로 향상시킬 수 있다는 가설을 제안합니다. 특히, 자연어에서의 효율적인 전이를 가능하게 하는 형식 언어의 특성에 대한 통찰력을 바탕으로 하여, 종속 구조(dependency structures)를 포착하고 모델 아키텍처의 계산 제한 내에서 유지되는 형식 언어가 효과적인 전이를 가능하게 한다고 주장합니다.

- **Technical Details**: 연구에서는 Transformer 모델을 중심으로, 주어진 형식 언어가 자연어에 적합할 때 언어 모델이 낮은 손실(loss)과 더 나은 언어적 일반화(linguistic generalization)를 달성한다고 설명합니다. 핵심적으로, 사전-사전 훈련(pre-pretraining) 방식이 정형 언어에 대해 훈련된 후 자연어로 훈련한 경우 더 낮은 손실과 더 나은 일반화 성능을 이끌어낸다고 보고합니다.

- **Performance Highlights**: 1B 파라미터의 언어 모델이 자연어 약 16억 토큰을 훈련한 결과, 사전-사전 훈련을 통해 33% 적은 토큰 예산으로도 유사한 손실 및 향상된 언어적 일반화를 달성함을 보여줍니다. 또한, 정형 언어 사전 훈련 과정에서 얻어진 주의 헤드(attention heads)가 자연어 평가에서 모델 성능에 중요한 요소로 작용하며, 긍정적 전이를 이끌어낸다고 mechanistic evidence를 통해 제시합니다.



### AI-Powered Bayesian Inferenc (https://arxiv.org/abs/2502.19231)
Comments:
          Research note, 27 pages, 3 figures

- **What's New**: Generative Artificial Intelligence (GAI)의 발전은 지식 습득 방식에 변곡점을 만들어냈습니다. 이 논문에서는 GAI의 불확실성과 내재된 무작위성을 문제로 보기보다는 기회로 바라보고, GAI의 예측을 활용한 전제 분포(prior distribution) 생성을 탐구합니다. 또한, 사용자 맞춤형 데이터셋과 결합하여 완전한 베이지안 분석을 수행할 수 있는 가능성을 제안합니다.

- **Technical Details**: 본 연구에서는 비모수 베이지안 프레임워크 내에서 이러한 가능성을 탐구하며, AI 생성 모델을 기반으로 하여 데이터 생성 분포에 대한 디리클레트 프로세스 사전 분포를 할당하는 기본 아이디어를 도출합니다. AI 사전의 유용성을 평가하기 위해 하이퍼파라미터를 샘플 외에서 조정하며, 인과적 샘플링 대신 최적화를 통해 후속(Posterior) 시뮬레이션을 수행합니다. 이 전략은 관측된 데이터와 AI를 통해 주입된 라벨을 가진 가짜 데이터를 포함한 증강 데이터에서 적절히 랜덤화된 기능을 이용하여 빠르게 iid 샘플을 생성할 수 있게 합니다.

- **Performance Highlights**: 이 방법은 AI의 예측을 활용하여 예측적 추론(predictive inference) 및 불확실성 정량화(uncertainty quantification)를 가능하게 합니다. 사용자는 GAI를 통해 생성된 정보에서 효율적으로 선행 정보를 도출할 수 있으며, 이를 통해 역사적 데이터와 동반하여 예측력을 높일 수 있습니다. 최종적으로 이 연구는 GAI가 비모수 베이지안 분석에서 기존의 통계 모델을 넘어서는 새로운 가능성을 제시하고 있습니다.



### Enhancing the Scalability and Applicability of Kohn-Sham Hamiltonians for Molecular Systems (https://arxiv.org/abs/2502.19227)
- **What's New**: 이번 연구에서는 Density Functional Theory (DFT)의 적용에서의 한계를 극복하기 위해 PubChemQH라는 대규모 학습 세트를 생성했습니다. 새로운 Wavefunction Alignment Loss (WALoss)라는 손실 함수를 도입하여 Hamiltonian 예측의 정확성을 높였습니다. 이 방법을 통해 대형 분자 시스템에 대한 DFT 계산의 효율성을 크게 개선할 수 있음을 보여줍니다.

- **Technical Details**: DFT는 분자의 전자 구조를 연구하는 데 주로 사용되는 이론적 프레임워크로, Kohn-Sham Hamiltonian의 구축 및 해결을 중심으로 합니다. 연구에서는 eSCN(convolution)과 sparse mixture of pair experts를 활용한 현대화된 Hamiltonian 예측 아키텍처인 WANet을 도입하였으며, WALoss를 통해 Ground-truth Hamiltonian과 예측 Hamiltonian의 일치를 향상시킵니다. 이러한 방법은 자원 집약적인 SCF(iterations)를 피하면서 즉시 예측 결과를 생성할 수 있도록 합니다.

- **Performance Highlights**: WALoss를 사용하여 총 에너지 예측 오류를 1347배 줄이고, SCF 계산 속도를 18% 향상시키는 성과를 달성했습니다. 이러한 개선은 대형 분자 시스템에 대한 정확하고 적용 가능한 예측을 위한 새로운 벤치마크로 자리 잡습니다. 결과적으로, 본 연구는 큰 분자의 전자 특성을 보다 효과적으로 예측할 수 있는 기반을 제공합니다.



### A Lightweight and Extensible Cell Segmentation and Classification Model for Whole Slide Images (https://arxiv.org/abs/2502.19217)
Comments:
          27 pages, 11 figures

- **What's New**: 본 논문에서는 디지털 병리학에서 세포 수준 분석 도구 개발의 어려움을 해결하기 위해 경량화되고 확장 가능한 cell segmentation 및 classification 모델을 제안합니다. 데이터 레이블을 교차 레이블링(cross-relabeling)하여 PanNuke 및 MoNuSAC의 주석을 정제하고, 일곱 가지 서로 다른 세포 타입을 포함하는 통합 데이터셋을 생성했습니다. 이러한 과정에서 모델 성능과 데이터 품질을 향상시키는 방법을 소개합니다.

- **Technical Details**: H-Optimus 기초 모델을 고정된 인코더로 활용하여 시퀀스 분할(segmentation) 및 분류(classification) 작업을 위한 특징 표현(feature representation)을 개선했습니다. 모델 크기와 복잡성을 줄이기 위해 지식을 증류(distillation)하였고, 이는 비교 가능한 성능을 유지하면서도 모델 파라미터 수를 48배 줄였습니다. 마지막으로, 이 증류 모델을 널리 사용되는 오픈 소스 디지털 병리학 플랫폼인 QuPath에 통합했습니다.

- **Performance Highlights**: H-Optimus 기반 모델을 사용한 세분화(segmentation) 및 분류(classification) 성능이 CNN 기반 모델보다 더 우수함을 보여줍니다. 평균 $R^2$ 값이 0.575에서 0.871로, 평균 $PQ$ 점수가 0.450에서 0.492로 개선되어 실제 세포 수와의 일치도가 높아졌습니다. 이러한 접근은 진단에 중대한 영향을 미치고 병리학자의 작업 부담을 줄이며 결과를 개선할 수 있는 가능성을 보여줍니다.



### FaithUn: Toward Faithful Forgetting in Language Models by Investigating the Interconnectedness of Knowledg (https://arxiv.org/abs/2502.19207)
Comments:
          16 pages

- **What's New**: 이 논문에서는 언러닝(unlearning) 방법이 상호연관된 지식을 신뢰성 있게 제거하지 못할 수 있다는 문제를 강조합니다. 저자들은 'superficial unlearning'이라는 새로운 개념을 정의하고, 언러닝이 효과적으로 수행되고 있는지 평가하기 위한 새로운 벤치마크인 FaithUn을 도입하였습니다. 이 벤치마크는 현실 세계의 지식 Q&A 설정에서의 언러닝 신뢰성을 분석하는 데 중점을 두고 있습니다.

- **Technical Details**: 저자들은 Knowledge-Localized UnlEarning, 즉 KLUE라는 새로운 언러닝 방법론을 제안합니다. KLUE는 특정 지식과 관련된 뉴런만을 업데이트하여 보다 신뢰성 있는 언러닝을 달성합니다. 이를 위해 저자들은 설명 가능성(explainability) 방법을 사용하여 어떤 뉴런을 업데이트할지 결정하고, 선택된 정보의 유지를 통해 불필요한 지식이 무의식적으로 제거되는 것을 방지합니다.

- **Performance Highlights**: 실험 결과, 기존의 언러닝 방법들이 신뢰성 있는 언러닝을 보장하지 못하는 반면, KLUE는 FaithUn 설정에서 기존 방법들보다 우수한 성능을 보였습니다. 이는 지식 기반의 언러닝이 효과적으로 수행될 수 있음을 시사하며, 앞으로의 연구 방향에 새로운 질문을 제기합니다. 저자들은 이 연구를 통해 언러닝 분야에서 복잡하고 상호연관된 지식의 중요성을 재조명하고 있습니다.



### EGR-Net: A Novel Embedding Gramian Representation CNN for Intelligent Fault Diagnosis (https://arxiv.org/abs/2502.19199)
- **What's New**: 이 논문은 회전 기계의 고장 진단에서 중요한 기능 추출 방법으로서, 복잡한 1D 진동 신호를 간단한 텍스처를 가진 2D 이미지로 변환하는 새로운 1D-to-2D 변환 방법인 Embedding Gramian Representation (EGR)을 제안합니다. 기존의 방법들과 달리 EGR은 계산이 간단하고 우수한 분리성(separability)을 보여줍니다. 또한, EGR 기반의 더블 브랜치 CNN 모델인 EGR-Net이 제안되어, 원시 신호의 특징 맵과 EGR을 동시에 학습할 수 있도록 설계되었습니다.

- **Technical Details**: EGR은 1D 신호를 포함된 에벨딩 공간에서 처리하며, 신호의 내재적 주기성을 포착합니다. 이 방법은 원시 신호 행렬(RSM)을 구성하고, 그에 대한 Gramian을 계산하는 두 단계로 구현됩니다. EGR 알고리즘은 행렬 곱셈(matrix multiplication)으로만 계산되며, 이러한 접근은 정보 중복을 크게 줄이고, 생성된 특성들은 분리성이 좋게 나타납니다.

- **Performance Highlights**: 제안된 EGR-Net은 기존 CNN 모델의 단일 입력에서 발생하는 정보 손실 문제를 줄이기 위해 고안되었습니다. 이 모델은 파라미터 선택 규칙을 논의하고, EGR의 진행 과정에서 브리지 연결을 통해 두 브랜치 간의 특성 학습 상호작용을 개선합니다. 이를 통해 EGR-Net은 전통적인 방법들과 비교하여 향상된 성능을 보여줍니다.



### Simulation of Language Evolution under Regulated Social Media Platforms: A Synergistic Approach of Large Language Models and Genetic Algorithms (https://arxiv.org/abs/2502.19193)
Comments:
          The manuscript has been submitted to IEEE Transactions on Computational Social Systems

- **What's New**: 이 논문은 사용자 콘텐츠를 조절하기 위해 소셜 미디어 플랫폼이 지속적으로 시행하는 제한적인 정책에 대응하는 창의적인 언어 회피 전략을模拟하기 위해 대규모 언어 모델(LLMs)을 기반으로 한 다중 에이전트 프레임워크를 제안합니다. 참여 에이전트는 사용자 역할을 하고 감독 에이전트는 플랫폼 수준의 규제를 모방하여 정책 위반을 평가합니다. 이러한 내러티브는 규제의 제약을 받는 언어 진화의 다각적 접근을 강조합니다.

- **Technical Details**: 프레임워크는 두 가지 핵심 역할, 즉 참여 에이전트와 감독 에이전트를 정의하여 언어 전략을 '제약 전략'(constraint strategies)과 '표현 전략'(expression strategies)으로 분리합니다. Genetic Algorithm(GA)을 활용하여 언어 전략을 직접 수정하는 LLM의 역할을 강조합니다. 선택, 돌연변이 및 교차 작업을 통해 다이나믹한 언어 전략의 진화를 극대화하였습니다.

- **Performance Highlights**: 실험 결과, 대화 라운드 수가 증가함에 따라 중단 없는 대화의 횟수와 정보 전송의 정확성이 크게 향상됨을 보여줍니다. 40명의 사용자 연구를 통해 생성된 대화와 전략의 현실적 관련성이 검증되었습니다. 또한 ablation study를 통해 GA의 역할의 중요성이 강조되어 장기적인 적응성과 전반적인 결과 개선에 기여함을 증명하였습니다.



### Provocations from the Humanities for Generative AI Research (https://arxiv.org/abs/2502.19190)
Comments:
          working draft; final draft in preparation

- **What's New**: 이번 연구에서는 인문학 연구자들의 관점에서 생성 AI의 사용, 영향 및 해악에 대한 새로운 고찰을 제시합니다. 인문학 연구의 정의를 세우고, 주요 이론과 방법론을 요약하여 AI의 현재 환경에 적용합니다. 여덟 가지 주요 주장을 제시하며, 이들은 생성 AI를 둘러싼 현재의 대화에 광범위하게 적용될 수 있는 내용입니다.

- **Technical Details**: 연구에서 제시된 여덟 가지 주장은 다음과 같습니다: 1) 모델은 단어를 만들지만, 사람은 의미를 만든다; 2) 생성 AI는 문화의 확대된 정의를 요구한다; 3) 생성 AI는 결코 대표적일 수 없다; 4) 더 큰 모델이 항상 더 나은 모델은 아니다; 5) 모든 훈련 데이터가 동등하지 않다; 6) 개방성은 쉬운 해결책이 아니다; 7) 제한된 컴퓨팅 접근은 기업의 포획을 가능하게 한다; 8) AI의 보편성은 좁은 인간 주체를 만든다.

- **Performance Highlights**: 인문학 연구는 기술적 연구에 비춰볼 때, AI와 관련된 복잡한 질문들을 더 잘 이해하고 다룰 수 있도록 돕습니다. 이는 AI의 현재 사용과 미래 가능성에 대한 대화에 더 많은 명확성을 제공합니다. 또한, 인문학 전반을 아우르는 연구 결과들은 AI에 대한보다 깊은 이해를 위한 다리 역할을 할 것입니다.



### AutoML for Multi-Class Anomaly Compensation of Sensor Drif (https://arxiv.org/abs/2502.19180)
Comments:
          To be published in Measurement Journal

- **What's New**: 이 논문은 산업 측정 시스템에서 센서 드리프트(sensor drift)를 해결하는 데 초점을 맞추고 있습니다. 전통적인 교차 검증(cross-validation) 방법이 드리프트를 적절히 고려하지 못해 모델 성능을 과대 평가한다는 점을 지적합니다. 모델이 훈련 세트와 테스트 세트에서 동일한 데이터 인스턴스(instance)를 허용하는 것이 주요 원인입니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 해결책을 포함합니다: (1) 모델 유효성을 위한 새로운 센서 드리프트 보상 학습 패러다임(sensor drift compensation learning paradigm)과 (2) AutoML(자동화된 기계 학습) 기술을 통해 분류 성능을 향상시키고 센서 드리프트를 보상합니다. 이 방법을 통해 데이터 밸런싱(data balancing), 메타 학습(meta-learning), 자동 앙상블 학습(automated ensemble learning), 하이퍼파라미터 최적화(hyperparameter optimization), 특징 선택(feature selection), 부스팅(boosting) 전략을 사용하여 성능을 극대화합니다.

- **Performance Highlights**: AutoML-DC(드리프트 보상 모델)는 센서 드리프트에 대한 분류 성능을 크게 향상시키며, 다양한 드리프트 심각도에 효과적으로 적응합니다. 이는 산업 환경에서 기계 학습 모델의 신뢰성과 정확성을 유지하는 데 중요한 기여를 할 수 있습니다. 따라서 이 연구는 드리프트가 있는 데이터 상황에서도 모델의 일반화 및 적응 능력을 높이는 데 중점을 두고 있습니다.



### MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis (https://arxiv.org/abs/2502.19175)
- **What's New**: 이번 연구에서는 Differential Diagnosis (DDx) 프로세스를 지원하기 위한 새로운 접근법인 Modular Explainable DDx Agent (MEDDxAgent) 프레임워크를 소개합니다. 기존의 DDx 지원 시스템들은 제한된 데이터 셋 평가, 구성 요소의 고립된 최적화 및 비현실적인 가정에 직면해 있었습니다. MEDDxAgent는 이러한 한계를 극복하며, 환자 프로파일이 완전하게 제공되기 전에도 진단적 추론을 발전시킬 수 있습니다.

- **Technical Details**: MEDDxAgent는 세 가지 모듈형 구성 요소로 구성되어 있습니다: (1) 진단 과정을 조정하는 DDxDriver, (2) 병력 조사 시뮬레이터, (3) 지식 검색 및 진단 전략을 위한 두 개의 전문 에이전트. 이 프레임워크는 호흡기 질환, 피부 질환 및 드문 질환을 포함한 포괄적인 DDx 벤치마크를 통해 신뢰성 있는 평가를 보장합니다.

- **Performance Highlights**: MEDDxAgent는 대규모 및 소규모 LLMs를 통해 10% 이상의 정확도 개선을 달성하였으며, 지속적인 학습 과정을 통해 진단적 추론의 투명성을 제공합니다. 이 연구는 또한 초기 환자 프로파일이 없을 때의 반복적 개선의 중요성을 강조하고 있습니다.



### TestNUC: Enhancing Test-Time Computing Approaches through Neighboring Unlabeled Data Consistency (https://arxiv.org/abs/2502.19163)
- **What's New**: 이 논문에서는 Test-time computing(테스트 타임 컴퓨팅) 접근 방식을 통해 LLM의 성능을 향상시키는 새로운 방법인 TestNUC를 소개합니다. TestNUC는 인접한 비정답 데이터의 지역적 일관성을 활용하여 예측을 개선하며, 비정답 인스턴스의 예측을 고려하여 입력 인스턴스를 분류합니다. 이 방법은 여러 데이터 세트에서 기존 방법들보다 우수한 성능을 보이며, 테스트 타임 컴퓨팅에 통합할 수 있는 가능성이 높습니다.

- **Technical Details**: TestNUC는 두 가지 주요 단계로 구성됩니다: ❶ Neighbor Retrieval(이웃 검색) 단계에서, 테스트 샘플과 유사한 특징을 가진 K개의 이웃을 식별합니다. ❷ Collaborative Prediction(협력적 예측) 단계에서는 LLM이 테스트 샘플과 그 이웃들에 대한 예측을 생성하고, 이들의 예측을 조합하여 최종 답변을 도출합니다. 이는 LLM이 비정답 샘플의 예측을 포함하여 의사결정을 더 잘 맥락화하고 세분화할 수 있도록 도움을 줍니다.

- **Performance Highlights**: TestNUC는 감정 탐지, 도메인 발견, 주제 채굴, 의도 분류 등 다양한 작업에서 평가되었으며, 기본 방법들인 표준 프롬프트 및 자기 일관성 방식보다 일관되게 더 나은 성능을 발휘했습니다. 특히, 비정답 데이터의 양이 증가할수록 성능이 효과적으로 확장되었고, 다양한 임베딩 모델에서도 강력한 성능을 보였습니다. 또한, TestNUC는 기존의 테스트 타임 컴퓨팅 방법들과 원활하게 통합되며, 성능을 크게 향상시킬 수 있습니다.



### Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models (https://arxiv.org/abs/2502.19160)
- **What's New**: 이번 연구에서는 언어에서 고정관념을 탐지하고 정량화하는 새로운 접근 방식을 제안합니다. 이는 사회 범주와 고정관념 커뮤니케이션(Social Category and Stereotype Communication, SCSC) 프레임워크에 기반하여 고정관념의 언어적 지표를 도출합니다. 이 방법은 다양한 대형 언어 모델(LLMs)을 활용하여 문장에서 이러한 지표를 자동으로 분류합니다.

- **Technical Details**: 연구는 LLM을 이용해 고정관념의 언어적 지표를 탐지하고 정량화하는데 초점을 맞춥니다. 접근 방식은 문장의 언어적 특성을 검토하고 세분화된 평가를 위한 기초를 제공합니다. 유의미한 언어 지표의 중요성을 평가하여 고정관념의 언어적 지표를 측정하는 점수 함수를 학습합니다.

- **Performance Highlights**: 모델의 성능 평가 결과, 일반적으로 고정관념의 언어적 지표를 탐지하고 분류하는 데 우수한 성능을 보였습니다. 그러나 일부 모델은 관련 행동 및 특성을 정확하게 평가하는 데 어려움을 겪었습니다. 더 많은 few-shot 예제를 프롬프트에 포함시키면 성능이 크게 향상되는 것으로 나타났습니다.



### When Personalization Meets Reality: A Multi-Faceted Analysis of Personalized Preference Learning (https://arxiv.org/abs/2502.19158)
- **What's New**: 이번 연구에서는 인간 피드백을 통한 강화 학습(Reinforcement Learning from Human Feedback, RLHF)의 한계를 극복하기 위해 개인화된 선호 학습(personalized preference learning)이 필요하다는 점을 강조합니다. 기존의 RLHF 기법이 사용자 간 동질적인 선호를 전제로 하여 다양한 인구 집단의 가치관을 간과한 것을 지적하고, 개인화된 선호를 적용하는 데 대한 방법론적 정당성을 제시합니다. 본 연구는 개별 사용자의 다양한 선호에 맞춰 LLM을 적응시키기 위한 포괄적인 평가 프레임워크를 소개합니다.

- **Technical Details**: 연구팀은 개인화된 선호 학습 기술을 벤치마킹할 수 있는 다각적(evalution framework) 평가 프레임워크를 제안했습니다. 이 프레임워크는 모델의 성능, 공정성(fairness), 비의도적 효과(unintended effects), 적응성(adaptability)을 측정할 수 있는 다양한 지표를 포함합니다. 아울러, 사용자의 데이터 가용성을 다양하게 고려하여 평가할 수 있도록 설계되었습니다. 이 연구는 8개의 개인화 방법을 이용해 3개의 서로 다른 선호 데이터 세트를 통해 광범위한 실험을 진행하였습니다.

- **Performance Highlights**: 실험 결과, 사용자 간의 강한 비동의(disagreement) 상황에서 성능 차이가 최대 36%에 달할 수 있으며, 개인화가 최대 20%의 안전성과 추론 능력을 저하시킬 수 있음을 보여주었습니다. 연구는 개인화의 잠재적 부작용이 LLM의 일반적인 능력을 저하시킬 수 있다는 점을 강조하고 있습니다. 특히, 개별 사용자에게 맞춤형 보상 모델을 파인튜닝(fine-tuning)하는 것이 효과적인 방법임을 발견하였으며, 협업 학습을 활용한 방법들이 이 baseline에 비해 최고 6%의 성능 향상을 보였습니다.



### Voting or Consensus? Decision-Making in Multi-Agent Deba (https://arxiv.org/abs/2502.19130)
- **What's New**: 이번 연구는 다수의 에이전트가 참여하는 토론에서 의사결정 프로토콜(Decision Protocol)의 선택이 성과에 미치는 영향을 체계적으로 평가합니다. 기존 연구들에서 다양한 토론 매개변수가 함께 변경되면서 의사결정 방식의 영향을 명확히 분석하기 어려웠습니다. 본 연구는 7개의 의사결정 프로토콜(예: 다수결 투표, 만장일치 합의)을 비교하여 에이전트 간 협업에 미치는 차이점을 검토합니다.

- **Technical Details**: 의사결정 프로토콜의 효과를 분석하기 위해, 연구자들은 MMLU, MMLU-Pro, GPQA와 같은 지식 데이터셋(Knowledge datasets)과 StrategyQA, MuSR, SQuAD 2.0과 같은 추론 데이터셋(Reasoning datasets)에서 다양한 프로토콜을 시험하였습니다. 의사결정 프로토콜을 단일 변수로 변경하여, 각 프로토콜이 에이전트 간 협업 및 성과에 미치는 영향을 평가했습니다.

- **Performance Highlights**: 투표 프로토콜은 추론 작업에서 13.2% 성과 향상을, 합의 프로토콜은 지식 작업에서 2.8% 성과 향상을 나타냈습니다. 에이전트 수를 늘리면 성과가 향상되지만 투표 전 논의 라운드가 많아지면 성과가 감소했습니다. 또한, 새로운 방법인 All-Agents Drafting (AAD)과 Collective Improvement (CI)를 제안하여 각각 최대 3.3%와 7.4% 성과 향상을 기록했습니다.



### From Traditional to Deep Learning Approaches in Whole Slide Image Registration: A Methodological Review (https://arxiv.org/abs/2502.19123)
- **What's New**: 이 논문은 전통적인 단일 또는 다단계 조직 슬라이드의 전체 슬라이드 이미지(WSI) 등록 기술을 다룹니다. 특히, 종양 미세환경(TME) 분석을 위한 이미지 정렬의 중요성을 강조하며, 현재의 접근 방법과 그 한계를 검토합니다. 딥 러닝(deep learning) 기반의 최신 방법론을 탐구하고 이 분야에서 경쟁력 있는 미래 연구 방향으로 이어지는 기회를 제시합니다.

- **Technical Details**: WSI 등록 과정은 여러 슬라이드, 스캐너, 또는 시간에 걸쳐 얻은 WSI 스캔을 정렬하고 결합하는 것입니다. 수학적 관점에서, 등록 문제는 움직이는 이미지 I_{m}을 기준 이미지 I_{r}에 최적으로 정렬하기 위한 변환을 찾는 것으로 정의됩니다. 이 과정은 다양한 이미지 품질, 이미징 모달리티, 노이즈 및 인공물과 같은 요소로 인해 복잡함을 겪고 있으며, 이러한 문제들을 해결하는 다양한 방법들을 논의합니다.

- **Performance Highlights**: WSI 등록의 성능 향상을 위해 자동 등록 알고리즘이 개발되고 있으며, 이로 인해 효율성과 정확성이 개선되고 있습니다. HE와 면역 조직 화학(IHC) 마커를 조합한 정량 분석은 질병의 진행과 치료 반응을 이해하는데 기여하고, 심지어 3D 재구성을 통한 복잡한 구조와 생물학적 샘플 내의 공간적 관계를 이해하는 데 도움을 줍니다. 최종적으로 이 논문은 WSI 등록의 다양한 응용을 제시하며, 그 중요성과 미래 연구 방향을 강조합니다.



### Chemical knowledge-informed framework for privacy-aware retrosynthesis learning (https://arxiv.org/abs/2502.19119)
- **What's New**: 본 연구에서는 CKIF(chemical knowledge-informed framework)라는 개인 정보 보호를 고려한 새로운 프레임워크를 소개합니다. CKIF는 여러 화학 조직 간의 분산 교육을 가능하게 하면서도 반응 데이터의 기밀성을 유지합니다. 이를 통해 기존의 접근 방식에서 발생하는 개인정보 유출 위험을 줄이고, 데이터 전송이나 보관 중 유출 가능성을 최소화합니다.

- **Technical Details**: CKIF는 반응 데이터의 원시 정보를 집계하는 대신, 모델 매개변수의 화학 지식 기반 집계를 통해 학습을 수행합니다. CKIF는 클라이언트의 요구에 맞추어 개인화된 모델을 학습할 수 있도록 설계되어 있으며, 이는 다양한 클라이언트에서 얻은 반응 데이터를 효과적으로 활용하여 스케일 가능한 모델 교육을 가능하게 합니다. 데이터 이질성이 존재하는 상황에서도 CKIW(chemical knowledge-informed weighting) 전략을 적용하여 각 클라이언트의 특정 요구를 반영하는 개인화된 모델을 만들어냅니다.

- **Performance Highlights**: 여러 반응 데이터셋에서 CKIF는 기존의 로컬 모델 및 중앙 집중식 모델보다 우수한 성능을 보였습니다. CKIF는 USPTO-50K 데이터셋을 사용한 실험에서 Locally Trained 모델에 비해 약 20%의 성능 향상을 보였으며, 일부 경우에서는 Centrally Trained 모델보다도 우수한 결과를 기록했습니다. 이는 CKIF의 개인화된 모델 학습 및 집계 방식이 효과적임을 나타냅니다.



### Improving customer service with automatic topic detection in user emails (https://arxiv.org/abs/2502.19115)
Comments:
          Paper submitted to the 15th International Conference on Information Society and Technology (ICIST), Kopaonik, Serbia, 9-12 March 2025

- **What's New**: 이 연구에서는 세르비아의 주요 통신사인 Telekom Srbija에서 고객 서비스 효율성을 높이기 위한 새로운 자연어 처리(Natural Language Processing) 파이프라인을 소개합니다. 이 파이프라인은 자동 이메일 주제 탐지와 라벨링을 통해 구현되며, 고객 서비스 운영에 혁신적인 변화를 가져올 것으로 기대됩니다.

- **Technical Details**: 중심에는 BERTopic이라는 모듈형 아키텍처가 있으며, 이는 비지도(topic modelling) 학습을 가능하게 합니다. 일련의 전처리(preprocessing) 및 후처리(post-processing) 단계를 거쳐, 우리는 12개의 주제 중 하나와 여러 추가 라벨을 들어오는 이메일에 할당하여 고객 서비스가 이를 필터링하고 접근할 수 있도록 합니다.

- **Performance Highlights**: 모델의 성능은 100개의 고객 이메일로 구성된 테스트 데이터 세트를 통해 자동으로 할당된 주제의 속도와 정확성을 평가하여 검토되었습니다. 이 파이프라인은 저자원(low-resourced) 및 형태소가 풍부한 언어에서도 폭넓은 적용 가능성을 보여주며, 현재 회사의 운영 환경에서 자동 이메일 분류를 통해 고객 서비스 운영을 간소화하고 있습니다.



### The Shady Light of Art Automation (https://arxiv.org/abs/2502.19107)
Comments:
          Accepted to ISEA 2025

- **What's New**: 이번 논문에서는 생성적 인공지능(Generative AI)이 예술에 미치는 영향을 보다 세밀하게 분석하고 있습니다. 기존 연구와는 달리 AI의 개념적 및 이데올로기적 기초가 예술 개념에 미치는 영향을 강조하며, 이는 예술과 문화의 현대적 맥락에서 더욱 주목받고 있습니다.

- **Technical Details**: 논문은 AI 기술이 컴퓨터 과학과 기술 산업에서 출발하여 현대 예술 및 문화에 전달하는 다양한 아이디어와 정치적 견해를 탐구합니다. 특히, 이러한 아이디어들이 서로 어떻게 융합되고 때때로 의문을 일으키는지를 분석합니다.

- **Performance Highlights**: 저자들은 생성적 AI의 영향력 있는 가치와 견해의 맥락을 간략히 비판하고, 이러한 관계가 어떻게 복잡하게 얽혀 있는지를 탐색합니다. 이는 예술계 내에서 AI의 역할 및 영향력을 보다 깊이 이해할 수 있는 단초를 제공합니다.



### XSS Adversarial Attacks Based on Deep Reinforcement Learning: A Replication and Extension Study (https://arxiv.org/abs/2502.19095)
- **What's New**: 본 논문은 Cross-site scripting (XSS) 공격에 대한 최신 연구를 다루고 있으며, Deep Learning (DL)을 활용한 XSS 탐지에서 발생하는 유효성 문제들의 해결책을 제시합니다. 해당 논문에서는 XSS Oracle을 도입해 실험적인 접근법을 개선하고, 특히 adversarial 공격의 새로운 형태에 대한 대응 방안을 소개합니다. 이 연구는 기존의 문헌에서 발견된 한계점들을 살펴보고, 이를 극복하기 위한 더 나은 평가 전략을 제공하는 데 중점을 두고 있습니다.

- **Technical Details**: XSS 공격은 웹 애플리케이션의 보안에 중대한 위협을 가합니다. 공격자는 사용자의 브라우저에서 유해한 스크립트를 실행하도록 유도하며, 이는 사용자 데이터와 시스템의 무결성을 위험에 빠뜨릴 수 있습니다. 기존의 XSS 탐지 방법들은 static과 dynamic 분석으로 나뉘며, 각각의 방법들이 가진 한계점과 이를 극복하기 위한 기법들을 논의합니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 접근법은 96% 이상의 Escape Rate(탐지에 실패하는 비율)를 달성하였습니다. 런타임에서 XSS 공격을 감지하는 데 있어, 제안된 방법은 기존의 방법들과 비교해 유사한 수준의 성능을 보이면서도 유효성에 대한 위협을 대폭 줄였습니다. 이로써 논문의 방법론은 더 투명하고 효과적인 평가 전략을 제공하며, 앞으로의 연구에 있어 중요한 기반이 될 것으로 기대됩니다.



### InternVQA: Advancing Compressed Video QualityAssessment with Distilling Large Foundation Mod (https://arxiv.org/abs/2502.19026)
Comments:
          Accepted by ISCAS 2025(Lecture)

- **What's New**: 이 논문에서는 비디오 품질 평가(VQA)를 위해 강력한 비디오 표현 능력을 가진 InternVideo2를 활용하여 경량화된 모델을 개발했습니다. 기존의 대규모 비디오 모델이 자원 소모가 크기 때문에, Distillation 방법을 통해 compression 품질 정보를 효과적으로 전이하여 모델의 크기를 줄이면서 성능을 유지하였습니다. 실험 결과, 제안된 경량 모델이 기존 방법들보다 우수한 성능을 보였습니다.

- **Technical Details**: 제안한 방법은 knowledge distillation을 기반으로 하며, 두 가지 손실 함수를 활용하여 학생 모델이 교사 모델의 feature representation을 효과적으로 학습할 수 있도록 합니다. 구체적으로, ℒ2 Loss와 Smooth ℒ1 Loss를 사용하여 예측값과 진짜 값 사이의 오차를 최소화하며, 교사 모델과 학생 모델 간의 feature 정합성을 보장합니다. 이러한 방법을 통해 경량 모델은 비디오 품질 평가에서 compression 왜곡 문제를 보다 잘 처리할 수 있게 됩니다.

- **Performance Highlights**: 제안된 경량 모델은 두 가지 compression 품질 평가 데이터셋에서 기존의 모든 방법을 초월하는 성능을 입증했습니다. 동시에, 경량 모델은 원래의 대규모 모델과 비슷하거나 더 좋은 성능을 달성하여 효율성과 성능 간의 최적의 균형을 이뤘습니다. 결과적으로, 이 연구는 비디오 품질 평가 분야에서의 새로운 가능성을 열어주고 있습니다.



### Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments (https://arxiv.org/abs/2502.19024)
Comments:
          Accepted by ICRA 2025

- **What's New**: 최근 Vision-and-Language Navigation (VLN) 분야의 연구에서, 저자들은 낮은 시야각을 가진 사족보행 로봇과 인간 중심의 지시사항 간의 불일치 문제를 해결하기 위한 Ground-level Viewpoint Navigation (GVNav) 방안을 제안하고 있습니다. 이는 VLN에서 다양한 높이에서의 시각적 관찰의 일반화 격차를 강조한 최초의 시도입니다. 본 연구는 광범위한 실험을 통해 시뮬레이션 환경 및 실제 환경에서의 성능 향상을 입증하였습니다.

- **Technical Details**: 이 논문은 사족보행 로봇이 낮은 높이에서 데이터를 수집하는데 직면하는 여러 가지 문제를 다루고 있습니다. 저자들은 비슷한 특징에 적절한 가중치를 부여함으로써 지역적 관측의 장애물을 처리할 수 있는 적응형 정보 수집 모듈을 개발하였습니다. 또한, HM3D 및 Gibson 데이터셋의 연결 그래프를 활용하여 공간적 사전지식을 강화하는 방법을 제안하며, 이를 통해 실제 복잡한 환경에서의 경로 예측 능력을 향상시켰습니다.

- **Performance Highlights**: GVNav 접근법은 실제 환경과 시뮬레이션 환경 모두에서 성능을 크게 개선하는 결과를 나타내었습니다. 저자들은 Xiaomi Cyberdog을 사례 연구로 삼아 다양한 시각적 정보의 차이를 분석하고, 깊이 기반의 경로 예측이 낮은 시야각에서 어떤 영향을 받는지를 평가하였습니다. GVNav는 특히 복합적인 환경에서 사족보행 로봇의 작업 효율성을 높이는 데 기여하고 있습니다.



### Robust Over-the-Air Computation with Type-Based Multiple Access (https://arxiv.org/abs/2502.19014)
Comments:
          Paper submitted to 33rd European Signal Processing Conference (EUSIPCO 2025)

- **What's New**: 이 연구는 TBMA(Type-Based Multiple Access)를 활용하여 과제로서의 전파 연산(AirComp)을 적극적으로 개선하고자 한다. 기존의 직접 집계(Direct Aggregation, DA) 방식과 비교했을 때, TBMA는 여러 무선 자원에서 데이터를 분산함으로써 수신자가 전송된 데이터의 히스토그램 형태를 구성할 수 있도록 돕는다. 이러한 구조 덕분에 고전적 강건 추정기(classical robust estimators)와 다양한 함수를 통합할 수 있어 활용 가능성이 높아진다.

- **Technical Details**: TBMA는 데이터에 따라 단일 무선 자원(예: 시간, 주파수 또는 코드) 내에서 기기들이 전송하는 방식으로, 이는 수신자가 전송된 데이터의 히스토그램을 생성할 수 있게 해준다. 본 연구는 TBMA가 여러 무선 자원의 다양성을 활용하여 특정 자원을 타겟으로 한 공격을 식별하고 격리할 수 있는 가능성을 보여준다. 또한, TBMA는 DA에 비해 에너지 소모를 줄이고 채널 상태 정보(Channel State Information, CSI) 요구 사항을 낮추어 resilience를 향상시킬 수 있다.

- **Performance Highlights**: 본 연구의 시뮬레이션 결과, TBMA는 DA에 비해 높은 정확도를 유지하며, 적대적인 조건에서도 강력하게 성능을 발휘하는 것으로 나타났다. 또한, TBMA는 연합 학습(Federated Learning) 시나리오에서의 적용 가능성도 입증하였다. 이 연구 결과는 TBMA가 차세대 네트워크에서 안전하고 효율적인 데이터 집계를 위한 확장 가능하고 강력한 솔루션임을 확립한다.



### Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning (https://arxiv.org/abs/2502.19009)
Comments:
          ICLR 2025

- **What's New**: 본 논문에서는 Distillation for In-Context Planning (DICP)라는 새로운 모델 기반 (model-based) RL 프레임워크를 제안합니다. DICP는 Transformers가 환경의 동역학을 동시에 학습하고 정책(policy)을 개선하도록 하여 기존 RL 알고리즘의 비효율적인 행동을 벗어날 수 있도록 돕습니다. 이 접근법은 모델이 특정 작업을 수행하기 전에 결과를 시뮬레이션할 수 있는 추가적인 메커니즘을 제공합니다.

- **Technical Details**: DICP는 in-context learning과 모델 기반(dynamics model) 학습을 결합한 모델로, 환경의 동역학을 독립적으로 학습함으로써 비효율적인 행동을 회피합니다. 이 프레임워크는 굉장히 적은 환경 상호작용으로도 뛰어난 성능을 보이며, 다양한 이산적(discrete) 및 연속적(continuous) 환경에서 실험되었습니다. DICP는 기존의 RL 알고리즘을 모방하는 대신, 행동 예측을 통해 더 나은 결정이 가능하게 합니다.

- **Performance Highlights**: DICP는 Meta-World ML1 및 ML10 벤치마크에서 최첨단 성능을 기록하며, 기존의 모델-프리(multiple model-free) 대안들과 기존의 메타-RL 방법에 비해 훨씬 적은 환경 상호작용을 요구합니다. 결과적으로, 이 연구는 DICP가 RL과 메타-RL 분야에서 변화를 가져올 수 있는 가능성을 보여줍니다.



### Binary Neural Networks for Large Language Model: A Survey (https://arxiv.org/abs/2502.19008)
Comments:
          23 pages, 7 figures

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 새로운 바이너리 양자화 기법에 대한 포괄적인 리뷰를 제공합니다. 특히 BitNet 접근 방식을 강조하며, 이는 모델 훈련 시작부터 저정밀 바이너리 가중치를 사용하여 양자화를 수행하는 방법입니다. 이는 기존의 Post-Training Quantization(PTQ) 및 Quantization-Aware Training(QAT)과는 기본적으로 다른 접근입니다.

- **Technical Details**: 전통적인 양자화 방법인 PTQ와 QAT는 낮은 비트 너비에서 심각한 정밀도 손실을 가져옵니다. 그러나 BitNet은 훈련 초기부터 바이너리 가중치를 활용하여 양자화하여 고에너지 효율성을 달성할 수 있도록 설계되었습니다. 이 논문에서는 바이너리 양자화 기술이 딥 뉴럴 네트워크에서 어떻게 발전해왔는지를 상세히 설명합니다.

- **Performance Highlights**: BitNet 방식은 CPU 환경에서 높은 효율성을 얻었으며, 다중 모달 도메인에도 성공적으로 확장되었습니다. 이는 1비트 양자화 기술이 LLM에서 낮은 비용과 높은 정밀도로 개발될 수 있는 가능성을 보여줍니다. 논문에서 여러 연구 결과들이 소개되며, 이러한 기술들이 LLM의 실제 응용에 어떻게 기여할 수 있을지 탐구합니다.



### A Multi-Agent DRL-Based Framework for Optimal Resource Allocation and Twin Migration in the Multi-Tier Vehicular Metavers (https://arxiv.org/abs/2502.19004)
Comments:
          15 pages, 16 figures

- **What's New**: 이번 연구는 다계층 차량 메타버스(resource allocation and VT migration)에서 기존 기술의 한계를 극복하기 위해 새로운 프레임워크를 제안합니다. 이 프레임워크는 Graph Convolutional Networks (GCNs), 층별 Stackelberg 게임 기반 인센티브 메커니즘, 및 Multi-Agent Deep Reinforcement Learning (MADRL)을 통합하여 리소스 최적화 및 차량 트윈(VT) 마이그레이션을 실시간으로 개선합니다. 이를 통해 다양한 목표를 균형 있게 처리할 수 있으며, 성능 저하를 방지합니다.

- **Technical Details**: 제안하는 GCN 기반 모델은 차량 네트워크 내의 공간적 및 시간적 의존성을 포착합니다. 이와 함께, Stackelberg 게임 기반 인센티브 메커니즘은 차량과 인프라 간의 협력을 장려하며, MADRL 알고리즘은 리소스 할당과 VT 마이그레이션을 동시에 최적화합니다. Markov Decision Process (MDP)로 모델링된 이 시스템에서는 Multi-Objective Multi-Agent Deep Deterministic Policy Gradient (MO-MADDPG) 알고리즘이 사용됩니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안한 알고리즘은 리소스 활용도 9.7%, 마이그레이션 비용 14.2%, 전반적인 사용자 경험(UX) 16.1% 향상시켰습니다. 이는 차량의 이동성과 RSU의 제한된 컴퓨팅 자원 문제를 효과적으로 해결했음을 나타냅니다. 이 연구는 차량 메타버스에서의 리소스 관리 및 VT 마이그레이션의 효율성을 크게 향상시켜, 미래의 스마트 모빌리티에 대한 가능성을 제시합니다.



### The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training (https://arxiv.org/abs/2502.19002)
Comments:
          23 pages

- **What's New**: 이번 논문에서는 다양한 Transformers 모델에서 발견된 Sharpness Disparity(주목성 차별성)를 탐구합니다. 이 차별성은 훈련 초기부터 나타나며, 훈련 과정 전반에 걸쳐 지속됨을 보여줍니다. 연구자들은 이 발견을 바탕으로 각 블록의 주목성을 고려한 Blockwise Learning Rate(LR) 전략을 제안하며, 이는 LLM(대형 언어 모델) pre-training(사전 학습) 속도를 가속화하는 데 기여합니다.

- **Technical Details**: Blockwise LR은 각 블록의 주목성에 따른 LR 조정을 통해 훈련의 효율성을 높입니다. 이 전략은 각 블록 종류 내에서 주목성이 낮은 블록의 LR을 조정하되, 주목성이 가장 높은 블록의 LR은 유지합니다. 이를 통해 훈련 안정성을 해치지 않으면서도 저주목성 방향으로의 역학을 가속화할 수 있습니다.

- **Performance Highlights**: Blockwise LR을 활용하면 AdamW와 비교해 두 배 빠른 속도를 자랑하며, 낮은 터미널 손실을 달성할 수 있음을 보여줍니다. 또한, 최근 제안된 메모리 효율적인 Adam-mini에 Blockwise LR을 통합하여 두 배의 속도 향상과 메모리 절약을 동시에 이루어냈습니다. 이러한 결과는 sharpness disparity를 활용하여 LLM 훈련 성능을 향상시킬 수 있는 가능성을 보여줍니다.



### PEToolLLM: Towards Personalized Tool Learning in Large Language Models (https://arxiv.org/abs/2502.18980)
- **What's New**: 이번 논문에서는 개인화된(tool learning) 도구 사용 능력을 강조하며, 기존의 도구 학습 연구들이 일반적인 도구 활용 능력에 치중했던 한계를 지적합니다. 이를 위해 사용자 인터랙션 이력을 통합하여 개인화된 도구 사용을 위한 새로운 작업을 정의하고, 첫 번째 개인화 도구 학습 벤치마크인 PEToolBench를 제안하였습니다. PEToolBench는 다양한 사용자 선호도를 반영하여, 46개 범주에 걸쳐 7454개의 도구를 포함하고 있습니다.

- **Technical Details**: PEToolLLaMA 라는 새로운 프레임워크가 제안되어 LLMs에 개인화된 도구 사용 능력을 부여합니다. 훈련 과정은 감독 세부 조정(Supervised Fine-Tuning, SFT) 단계와 사용자 선호 최적화(Direct Preference Optimization, DPO) 단계를 포함하며, 이는 사용자가 선호하는 도구 호출을 샘플링하여 최적화하는 과정입니다. 이 프레임워크는 사용자 명령과 상관없이 사용자 인터랙션 이력을 고려하여 과제를 수행하도록 LLM을 유도합니다.

- **Performance Highlights**: 실험 결과, PEToolLLaMA는 기존의 최고 성능 LLM보다 50% 이상 향상된 성능을 발휘하여, 개인화된 도구 사용 기술에서의 우수성을 보여주었습니다. 또한, PEToolLLaMA는 다양한 사용자 요구를 충족시키는 도구 사용 준비를 통해 LLM들의 활용도를 대폭 향상시킵니다. 이는 개인화 도구 학습 평가를 위한 중요한 기초 자료를 제공하게 됩니다.



### Low-Confidence Gold: Refining Low-Confidence Samples for Efficient Instruction Tuning (https://arxiv.org/abs/2502.18978)
Comments:
          8 pages

- **What's New**: 이 연구는 Large Language Models에 대한 instruction fine-tuning의 효과성을 높이기 위해 새로운 필터링 프레임워크인 Low-Confidence Gold (LCG)를 도입합니다. LCG는 centroid-based clustering과 confidence-guided selection을 활용하여 가치 있는 instruction 쌍을 식별합니다. 이 접근 방식은 데이터의 다양성을 유지하면서 고품질의 부분 집합을 큐레이션하는 반면, Semi-supervised 방법론을 적용하여 대표 샘플로 훈련된 경량 classifier를 사용합니다.

- **Technical Details**: LCG 프레임워크는 centroid-based clustering을 기반으로 하며, 각 클러스터의 중심을 통해 유사한 데이터 샘플을 그룹화합니다. 그 후, confidence-guided selection을 통해 고품질의 instruction 쌍을 선택합니다. 이 과정은 주어진 데이터를 효과적으로 활용함으로써 instruction fine-tuning의 품질을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, LCG로 필터링된 6K 샘플을 기반으로 fine-tuning된 모델들이 기존 방법론에 비해 더 뛰어난 성능을 보여주었습니다. 특히 MT-bench에서의 상당한 개선을 나타내며, 종합적인 평가 메트릭스에서도 일관된 성과 향상을 달성했습니다. 이러한 결과는 효율적인 instruction tuning을 위한 유망한 방향을 제시합니다.



### (Mis)Fitting: A Survey of Scaling Laws (https://arxiv.org/abs/2502.18969)
Comments:
          41 pages, 3 figure, first two authors contributed equally. ICLR, 2025

- **What's New**: 이번 논문은 대규모 모델 훈련 과정에서의 스케일링 법칙(scaling laws)의 중요성에 대해 논의합니다. 저자들은 기존 연구들이 도출한 결론의 불일치를 분석하며, 최적의 토큰-파라미터 비율(optimal token to parameter ratio)과 같은 쟁점들을 탐구합니다. 또한, 50편 이상의 스케일링 관련 논문을 조사하여 재현성을 높이기 위한 체크리스트(checklist)를 제안합니다.

- **Technical Details**: 스케일링 법칙은 손실(loss)과 모델 크기, 데이터셋 크기 간의 관계를 설명하는 파워 법칙(power law)으로 규명됩니다. 연구자들은 모델 훈련을 통해 얻은 데이터에서 이 법칙을 적합(fitting)하기 위해 다양한 하이퍼파라미터(hyperparameters)와 초기화 조건을 설정해야 합니다. 논문에서는 이러한 프로세스의 세부적인 변화가 결과에 미치는 영향을 구체적으로 설명합니다.

- **Performance Highlights**: 조사한 51편의 논문에서 중요한 세부사항이 종종 누락되어 재현성에 significant 영향을 미친다고 강조합니다. 분석 코드가 없는 42편 중 단 19편만이 코드 조각을 제공하고, 23편은 최적화 과정에 대한 설명이 부족하며, 15편은 FLOP 또는 파라미터 수의 계산 방법을 설명하지 않습니다. 이러한 누락된 정보는 최종 결과에 중대한 영향을 미치는 것으로 나타났습니다.



### DualSpec: Text-to-spatial-audio Generation via Dual-Spectrogram Guided Diffusion Mod (https://arxiv.org/abs/2502.18952)
- **What's New**: 이번 논문에서는 텍스트 설명으로부터 공간 오디오를 생성하는 새로운 프레임워크인 Text-to-Spatial-Audio (TTSA) 시스템을 제안합니다. 기존의 연구들은 주로 텍스트에서 단일 채널 오디오로 제한되어 있었으나, 이 방식은 몰입감을 주는 공간 오디오의 제작으로 발전하고자 합니다. DualSpec이라 불리는 이 시스템은 가변 오토인코더(Variational Autoencoders, VAE)를 활용하여 오디오 이벤트로부터 잠재 음향 표현을 추출하고, 대형 언어 모델의 인코더를 통해 텍스트를 특징으로 변환합니다.

- **Technical Details**: TTSA 모델은 두 가지 종류의 음향 특징을 동시에 활용하여 생성 품질 및 방향 정확도를 향상시킵니다. 멜 스펙트로그램과 단기 푸리에 변환(Short-time Fourier Transform, STFT) 스펙트로그램을 사용하여 생성 정보를 압축하고 분산 모델에 피드합니다. 이 모델은 다수의 VAE를 설계하여 다양한 음향 특징을 저차원 잠재 표현으로 압축하고, 이를 바탕으로 고품질의 공간 오디오를 생성합니다.

- **Performance Highlights**: 제안된 방법은 높은 방향성과 이벤트 일관성을 갖춘 공간 오디오를 생성할 수 있음을 실험 결과를 통해 입증하였습니다. 새롭게 도입된 공간 인지 평가 지표는 생성된 공간 오디오 기록의 방향 오류를 정량화하는 데 사용됩니다. 이러한 접근 방식은 Spatial Quality 평가에 있어 객관적이고 정량적인 방법을 제공합니다.



### MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors (https://arxiv.org/abs/2502.18940)
Comments:
this https URL

- **What's New**: 이 논문에서는 MathTutorBench라는 새로운 오픈소스 벤치를 제안하여 AI 기반의 튜터링 모델의 교육적 능력을 평가할 수 있는 종합적인 방법론을 제공합니다. 이 벤치는 대화형 교육에서의 튜터링 능력을 평가하는 데이터셋과 메트릭스를 포함하고 있어 현재의 평가 방식의 부족함을 보완합니다.

- **Technical Details**: MathTutorBench는 세 가지 카테고리로 나뉘어 있습니다: 수학 전문성, 학생 이해, 그리고 교사 응답 생성입니다. 특히, 효과적인 튜터 발화와 덜 효과적인 발화를 구별하여 보상 모델을 훈련시키고, 이를 통해 튜터 모델의 생성물에 점수를 매깁니다. 이 모델은 전문가와 초보 교사의 발화를 높은 정확도로 구분할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 모델의 주제 전문성과 교육적 능력 간에는 trade-off가 존재하며, 교육을 위해 특화된 모델은 일반 모델에 비해 더욱 긴 대화에서도 교육 능력을 유지합니다. 이는 향후 튜터링 LLM의 개발 가속화에 기여할 것으로 기대됩니다. MathTutorBench는 자동화된 메트릭스를 사용하여 신속하고 공정한 평가를 가능하게 하므로, 공개적으로 데이터와 코드를 제공하여 연구자들이 쉽게 사용할 수 있도록 하였습니다.



### JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models (https://arxiv.org/abs/2502.18935)
Comments:
          12 pages, 5 figures, accepted at PAKDD 2025

- **What's New**: 이 논문에서는 JailBench를 소개하며, 이는 LLMs(대형 언어 모델)의 심층 안전 취약성을 평가하기 위한 첫 번째 포괄적인 중국어 벤치마크입니다. JailBench는 조정된 계층적 안전 분류 체계를 특징으로 하고 있으며, 이는 중국 맥락에 맞춰져 있습니다. 저자들은 이 벤치마크가 이전의 중국어 벤치마크들에 비해 ChatGPT에 대한 높은 공격 성공률을 기록했다고 강조하고 있습니다.

- **Technical Details**: JailBench는 4,600개의 질의로 구성된 데이터를 구축하며, 이는 5개의 고유한 영역과 40개의 리스크 유형을 포함하는 새로운 두 레벨 계층화 안전 분류 기준을 기반으로 하고 있습니다. Automatic Jailbreak Prompt Engineer (AJPE) 프레임워크를 통해 자동적으로 위험한 질의를 생성하여 데이터셋을 확장하며, 이는 LLM의 맥락 학습을 활용하여 효율성을 극대화합니다.

- **Performance Highlights**: JailBench는 13개의 주요 LLM에 대해 광범위한 평가를 수행하였으며, ChatGPT에 대해 73.86%의 공격 성공률을 기록했습니다. 이로써 JailBench는 다양한 분야에서 LLM의 잠재적 취약성을 효과적으로 식별하는 데 성공하였으며, 이는 중국어 맥락에서 LLM의 안전성과 신뢰성을 개선할 수 있는 중요한 통찰을 제공합니다.



### SLAM in the Dark: Self-Supervised Learning of Pose, Depth and Loop-Closure from Thermal Images (https://arxiv.org/abs/2502.18932)
- **What's New**: DarkSLAM은 저조도 환경에서 효과적으로 작동하는 단안 열 SLAM 시스템으로, 향상된 포즈 정확성을 위한 Efficient Channel Attention (ECA) 및 Depth 추정을 위한 Selective Kernel Attention (SKA) 메커니즘을 통합하고 있습니다. 이 시스템은 복잡한 조명 조건에서도 강력한 성능을 제공할 수 있도록 열 깊이 기반 루프 클로저 감지 및 포즈 최적화를 포함합니다.

- **Technical Details**: DarkSLAM은 기본적으로 원시 열 이미지를 선형 변환 및 저주파 필터링을 통해 전처리하며, 자기 지도 학습(self-supervised learning)을 통해 깊이 및 포즈를 학습하는 구조로 되어 있습니다. ECA는 포즈 피처 추출을 개선하고, SKA는 다중 스케일 깊이 피처 융합을 강화하여 열 이미지의 약점을 보완합니다. Siamese 네트워크 기반의 루프 감지 프레임워크는 루프 클로저 감지 성능을 높이고 포즈 그래프를 최적화합니다.

- **Performance Highlights**: 대규모 야외 실험에서 DarkSLAM은 SC-SfM-Learner 및 Shin et al.의 방법들보다 현저히 개선된 성능을 보여주었습니다. ECA 메커니즘이 PoseNet의 절대 궤적 오차(ATE)를 38.5% 감소시켰으며, SKA 메커니즘은 깊이 예측 정확성을 향상시키며 깊이 예측 감소와 관련된 문제를 효과적으로 완화했습니다. DarkSLAM은 저조도 조건에서도 정밀한 위치 추적 및 3D 밀집 지도를 제공했습니다.



### BeamVQ: Beam Search with Vector Quantization to Mitigate Data Scarcity in Physical Spatiotemporal Forecasting (https://arxiv.org/abs/2502.18925)
- **What's New**: 본 논문에서는 베이스 예측 모델을 통한 물리적 시공간 예측에서 발생하는 데이터 부족 문제를 해결하기 위해 새로운 확률적 프레임워크인 Beam Search with Vector Quantization(BeamVQ)를 제안합니다. 이 방법은 자기 앙상블(self-ensemble) 전략과 결합하여 극한 사건을 보다 효과적으로 예측하는 데 중점을 두고 있습니다. BeamVQ는 연속 상태 공간을 탐색하고, 높은 품질의 후보들을 결합하여 추론 품질을 향상시키는데 기여합니다.

- **Technical Details**: BeamVQ의 핵심은 변량 양자화를 통해 연속 출력 공간을 이산화하여 빔 검색(beam search)을 수행하는 것입니다. 이를 통해 상위 k개의 후보를 필터링하고, 이를 기반으로 새로운 자기 앙상블 전략을 개발하여 추가적인 가상 샘플(pseudo samples)을 생성합니다. 이는 훈련 데이터셋을 반복적으로 증가시켜 물리적 일관성과 일반화를 개선합니다.

- **Performance Highlights**: BeamVQ는 다양한 벤치마크 및 백본 모델에서 예측 MSE를 최대 39%까지 감소시키는 성과를 보였습니다. 이 방법은 극한 사건 탐지 및 데이터 부족 처리에 효과적이며, 높은 정확도와 물리적 타당성을 유지합니다. 실험 결과, BeamVQ는 예측의 다양성과 극한 사건 캡처 능력을 상당히 향상시킴을 입증하였습니다.



### END: Early Noise Dropping for Efficient and Effective Context Denoising (https://arxiv.org/abs/2502.18915)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 긴 입력 시퀀스에서 소음(noise)으로 인해 발생하는 품질 저하 문제를 다룹니다. 연구팀은 LLM이 첫 번째 토큰을 생성하기 전에 입력 시퀀스에서 유용한 정보를 조기에 식별할 수 있음을 발견하였습니다. 이를 바탕으로 Early Noise Dropping(END)이라는 새로운 접근 방식을 제안하며, 이는 LLM의 세밀한 조정 없이 소음 문제를 완화하는 데 중점을 둡니다.

- **Technical Details**: END는 입력 시퀀스를 여러 개의 청크(Chunks)로 나누고, LLM의 초기 레이어에서 선형 프로버(linear prober)를 사용하여 유용한 청크와 소음 청크를 구별합니다. 이 방법은 소음 청크를 초기에 버림으로써 출력 품질을 유지하고, 계산 오버헤드를 줄입니다. EDD는 즉각적인 성능 향상을 목표로 하며, 기존 방법들의 복잡성과 실행 시간을 없애고 효율적으로 작동합니다.

- **Performance Highlights**: 연구 결과, END는 다양한 평가 데이터 세트에서 여러 LLM의 성능과 효율성을 크게 향상시켰습니다. 기존 강력한 기준선과 비교할 때 10% 이상의 성능 향상과 약 50%의 계산 감소를 달성하였습니다. 실험을 통해 LLM이 내부적으로 소음 및 불필요한 정보를 구별하는 메커니즘을 더욱 깊이 이해할 수 있는 기회를 제공합니다.



### Dynamic Classification: Leveraging Self-Supervised Classification to Enhance Prediction Performanc (https://arxiv.org/abs/2502.18891)
Comments:
          18 pages, 6 figures

- **What's New**: 이 논문에서는 제로 미스 탐지(0 missed detections)와 최소한의 오탐(false positives) 목표를 달성하기 위해 설계된 혁신적인 동적 분류 알고리즘(dynamic classification algorithm)을 제안합니다. 이 알고리즘은 데이터를 N개의 동등한 훈련 및 예측 하위 집합으로 분할하며, 각 모델은 독립적으로 예측을 수행합니다. 이를 통해 각 예측 모델이 더 작은 데이터 범위 내에서 작동하여 전반적인 정확도를 향상시킵니다.

- **Technical Details**: 이 알고리즘은 자기 지도 학습(self-supervised learning)을 기반으로 하여 데이터 하위 집합을 자동으로 분할하며, 각 하위 집합은 훈련 세트의 분포적 특성을 기반으로 작은 범위 내에서 예측을 수행합니다. 예측 결과는 이전 자기 지도 학습 정보를 활용하여 걸러내어, 추가 모델의 필요 없이 정확도 요구를 충족하지 않는 예측을 배제합니다. 실험 결과에 따르면, 분류 오류가 작을 때 동적 분류 알고리즘은 제로 미스 탐지와 최소한의 오탐이라는 기초 성과를 달성하며, 기존 모델 앙상블보다 월등한 성과를 보입니다.

- **Performance Highlights**: 현재 알고리즘은 자동 매개변수 조정(parameter tuning)과 분류 모델 효율성 측면에서 개선의 여지가 있지만, 다수의 데이터 세트에서 뛰어난 성능을 보였다. 향후 연구는 분류 구성 요소의 최적화에 초점을 맞춰 알고리즘의 Robustness와 적응성을 한층 강화할 것으로 기대됩니다. 동적 분류 알고리즘은 다양한 분야와 산업에서 널리 활용될 가능성이 높습니다.



### Clip-TTS: Contrastive Text-content and Mel-spectrogram, A High-Huality Text-to-Speech Method based on Contextual Semantic Understanding (https://arxiv.org/abs/2502.18889)
- **What's New**: Clip-TTS는 기존의 전통적인 Text-to-Speech (TTS) 방법의 한계를 극복하기 위해 개발된 혁신적인 다중 모달 접근 방식을 기반으로 하는 새로운 TTS 방법이다. 이 방법은 Clip 구조를 사용하여 텍스트 인코딩 단계에서 텍스트 콘텐츠와 실제 멜-스펙트로그램(mel-spectrogram) 간의 연결을 설정하며, 이를 통해 텍스트 인코더가 글로벌 컨텍스트의 진정한 의미를 학습할 수 있도록 한다. Clip-TTS는 Transformer의 기본 구조를 채택하여 빠른 추론 속도를 달성하며, 실험 결과는 Clip-TTS가 LJSpeech 및 Baker 데이터셋에서 탁월한 품질의 음성을 생성함을 보여준다.

- **Technical Details**: Clip-TTS는 Contrastive Language-Image Pretraining (Clip)과 텍스트-투-스피치(TTS) 합성을 통합하여 더 표현력이 풍부하고 상황에 맞는 음성 생성을 가능하게 한다. 기존의 TTS 모델과 달리 Clip-TTS는 풍부한 의미 표현을 활용하여 텍스트와 음성 간의 더 깊은 관계를 학습하고, 이를 통해 다양한 음성 스타일과 맥락에 대해 일반화할 수 있는 능력을 향상시킨다. 이 방법은 명시적인 레이블이나 방대한 주석 데이터 없이도 표현력의 제어가 가능하여, 다채로운 톤과 감정으로 음성을 생성하는 데 적합하다.

- **Performance Highlights**: Clip-TTS는 멜-스펙트로그램(mel-spectrogram) 표현을 활용하여 텍스트 콘텐츠를 이해하고 통합할 수 있는 능력이 뛰어나며, 이를 통해 인간과 유사한 음성을 생생하게 생성한다. 실험 결과, Clip-TTS는 최근 음성 생성 성능의 최첨단인 MOS 점수를 기록하였으며, 다양한 감정의 음성 샘플을 잘 처리함으로써 인터랙티브한 응용 프로그램에서 스토리텔링과 캐릭터 음성 합성에 특히 유용하게 활용될 수 있다.



### SE(3)-Equivariant Ternary Complex Prediction Towards Target Protein Degradation (https://arxiv.org/abs/2502.18875)
- **What's New**: 이번 논문에서는 소분자에 의해 유도되는 표적 단백질 분해(Targeted Protein Degradation, TPD)의 최신 진전을 다룹니다. 특히, 기존에 접근이 어려웠던 단백질 목표를 겨냥한 단백질 분해를 촉진하는 새로운 딥러닝 기반 접근법인 DeepTernary를 소개합니다. 이 방법은 E3 ligase와 타겟 단백질을 연결하는 삼원 복합체(ternary complex)를 직접 예측하여 신약 개발을 혁신적으로 변화시킬 가능성이 있습니다.

- **Technical Details**: DeepTernary는 인코더-디코더 아키텍처를 사용하여 엔드 투 엔드(end-to-end) 방식으로 삼원 구조를 예측합니다. 이 모델은 SE(3)-equivariant 고유 데이터베이스(TernaryDB)를 기반으로 하여, 복잡한 삼원 상호작용을 캡처하는 그래프 신경망(Graph Neural Network, GNN)과 주의 메커니즘(attention mechanism)을 활용합니다. 또한, 쿼리 기반 Pocket Points Decoder를 통해 TPD의 최종 결합 삼원 복합체의 3D 구조를 추출합니다.

- **Performance Highlights**: DeepTernary는 기존의 PROTAC 벤치마크에서 상태-of-the-art의 정확성과 속도를 기록하며, 이전 PROTAC과의 사전 지식 없이도 뛰어난 성능을 보였습니다. 또한, 더 도전적인 MGD 벤치마크에서도 높은 정확도를 달성했으며, 예측된 구조에서 계산된 피장면적(buried surface area)이 실험적으로 얻어진 분해 효능과 상관관계를 보이는 등, TPD 개발에 기여할 잠재력을 지니고 있습니다.



### Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework (https://arxiv.org/abs/2502.18874)
- **What's New**: 이번 논문에서 제안하는 ARJudge라는 새로운 평가 프레임워크는 자동화된 평가 기준 생성을 통해 LLM(대형 언어 모델)의 응답을 보다 효과적으로 평가할 수 있도록 설계되었습니다. 기존의 평가 방법론들은 사전 정의된 기준에만 의존하여 다양한 작업에 대한 적응력이 줄어드는 경향이 있었습니다. ARJudge는 텍스트 기반 분석과 코드 기반 분석을 통합하여 더 많은 측면에서 평가하도록 보장합니다.

- **Technical Details**: ARJudge는 두 가지 주요 구성 요소, 즉 다면적 평가를 생성하는 Analyzer와 분석 결과를 종합하여 최종 결정을 내리는 Refiner로 이루어져 있습니다. 이 시스템은 Composite Analysis Corpus라는 훈련 데이터를 기반으로 하여 평가 기준을 생성하고, 다각적인 분석을 통해 LLM의 창조적 응답을 효과적으로 평가하는 방법론을 제시합니다. 특히, 코드 기반 분석을 통합함으로써, 기존의 텍스트 기반 방법보다 훨씬 개선된 정확도를 보여줍니다.

- **Performance Highlights**: 다양한 벤치마크에서 수행된 실험 결과, ARJudge는 기존의 미세 조정된 평가자들보다 뛰어난 성능과 견고성을 보였습니다. 특히, ARJudge는 LLM의 응답이 지침을 얼마나 잘 따르는지를 평가하는 데 있어 코드 기반 분석이 약 11.1% 더 높은 정확도를 제공함을 입증했습니다. 이러한 결과는 다면적 평가와 코드 기반 분석의 활용이 평가 능력을 강화하는 데 얼마나 중요한지를 보여줍니다.



### Inscanner: Dual-Phase Detection and Classification of Auxiliary Insulation Using YOLOv8 Models (https://arxiv.org/abs/2502.18871)
- **What's New**: 본 연구는 구조적 구성 요소 내 보조 단열(auxiliary insulation)을 탐지하고 분류하기 위한 2단계 방법론을 제안합니다. 탐지 단계에서는 YOLOv8x 모델을 사용해 구조 청사진의 단열 영역을 정확하게 식별하며, 분류 단계에서 탐지된 단열 패치를 두 개의 클래스(존재 및 결손)로 나누어 분류합니다. 이 접근 방법은 자동화된 단열 탐지 및 분류의 효과성을 입증하며, 산업 환경 내 안전 기준과 품질 보증을 향상시킬 기반을 마련합니다.

- **Technical Details**: YOLOv8x는 최신 물체 탐지 모델로, 청사진에서 단열 영역을 탐지하는 데 사용됩니다. 탐지된 단열 구성 요소는 YOLOv8x-CLS 모델에 의해 완전한 존재 여부를 분류하며, 이를 통해 82%의 mAP(mean Average Precision) 점수와 98%의 정확성을 달성했습니다. 데이터셋 준비 과정에서는 주석(annotation), 증강(augmentation), 및 단열 지역의 적절한 크롭(cropping) 등 전처리 단계가 포함되었습니다.

- **Performance Highlights**: 제안된 방법론은 고도로 선별된 데이터세트를 평가하여 효과성을 입증했습니다. 다양한 산업 청사진을 테스트하며 10명 이상의 전문가와 협력하여 모델 출력을 검토하고 교차 검증하는 과정을 거쳤습니다. 이와 같은 다각적인 검증은 기존 수작업 공정에 비해 더욱 신뢰성 높은 단열 결함 탐지를 가능하게 하였습니다.



### A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops (https://arxiv.org/abs/2502.18865)
Comments:
          Accepted at ICLR 2025

- **What's New**: 본 논문은 Self-consuming Training Loops (STLs)라는 개념을 도입하여 합성 데이터 생성이 모델 훈련에 미치는 영향을 분석합니다. 저자들은 기존 연구에서 발견된 불일치의 원인을 이해하기 위해 'recursive stability'라는 새로운 개념을 제시하고 있습니다. 이 연구는 모델 아키텍처와 실제 데이터 및 합성 데이터 간의 비율이 STL의 성공에 미치는 영향을 분석하는 첫 번째 이론적 일반화 분석을 제공합니다.

- **Technical Details**: 논문은 STL의 일반화 오류 경계를 최초로 설정하며, recursive stability라는 주요 혁신을 제안합니다. 이는 복잡한 재귀적 구조와 비독립적(Non-i.i.d.) 데이터의 본질을 해결하는 데 중점을 둡니다. 또한, 변환기(transformers)가 in-context learning에서 실제 데이터 일정 비율을 유지할 경우 STL의 출력 차이를 조절할 수 있음을 입증합니다.

- **Performance Highlights**: STL 내의 합성 데이터 증강의 최적 크기는 실제 데이터 세트의 크기가 줄어들수록 증가한다는 것을 이론적으로 보여줍니다. 기존 데이터와 합성 데이터의 조합에 따라 각각의 생성에서의 일반화 성능이 향상되지만, 연속적인 세대 간의 분포 차이를 심화시킨다는 무역오프(trade-off)를 조사합니다. 최종적으로, 이 연구는 합성 데이터가 모델 훈련에 어떻게 기여할 수 있는지를 이해하는 데 중요한 기초 자료를 제공합니다.



### Sherlock: Towards Multi-scene Video Abnormal Event Extraction and Localization via a Global-local Spatial-sensitive LLM (https://arxiv.org/abs/2502.18863)
- **What's New**: 본 논문에서는 Video Anomaly Detection (VAD) 관련 기존 연구들이 비디오 프레임이 비정상인지 여부만을 감지하는 데 집중하고 있는 것을 지적합니다. 이에 따라 새로운 멀티-장면 비디오 비정상 사건 추출 및 로컬화(M-VAE) 작업을 제안합니다. 이는 비정상적인 사건의 쿼드러플(네 가지 요소)을 추출하고 이러한 사건을 로컬화하는 것을 목표로 합니다.

- **Technical Details**: M-VAE 작업은 전역-로컬 공간 모델링(global-local spatial modeling)과 전역-로컬 공간 균형(global-local spatial balancing)이라는 두 가지 주요 과제가 있다고 언급됩니다. 이를 해결하기 위해 Sherlock이라는 글로벌-로컬 공간에 민감한 대형 언어 모델(LLM)을 제안합니다. 이 모델은 Global-local Spatial-enhanced MoE (GSM) 모듈과 Spatial Imbalance Regulator (SIR)를 통해 두 가지 과제를 각각 해결합니다.

- **Performance Highlights**: 다양한 실험 결과, Sherlock 모델은 여러 첨단 Video-LLMs와 비교했을 때 현저한 이점을 보였습니다. 이는 M-VAE 작업에서 전역-로컬 공간 정보의 중요성을 입증하며, Sherlock이 이러한 정보를 효과적으로 캡처할 수 있음을 보여줍니다.



### Investigating Generalization of One-shot LLM Steering Vectors (https://arxiv.org/abs/2502.18862)
Comments:
          20 pages, 7 figures. Code is available at this https URL

- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 행동을 해석하고 제어하기 위해 스티어링 벡터(steering vectors)를 직접 최적화하는 방법을 제시합니다. 기존 방법들은 대조 데이터셋(constrastive datasets)에 의존하여 스티어링 벡터를 찾는 반면, 이번 연구는 단일 훈련 데이터를 통해 효율적으로 벡터를 최적화할 수 있는 가능성을 보여줍니다. 연구팀은 여러 스티어링 최적화 기법을 시험하여 각 벡터가 모델의 안전과 관련된 행동을 효과적으로 조절한다는 것을 발견했습니다.

- **Technical Details**: 스티어링 벡터를 최적화하기 위한 방법으로는 프로모션 스티어링(promotion steering), 서프레션 스티어링(suppression steering), 그리고 리엔트런트 스티어링(reentrant steering) 등이 포함됩니다. 이 연구에서 제시된 방법들은 안전 관련 행동을 조정하는 데 효과적임을 입증하였으며, 특히 유해한 행동을 제어하는 특정 설정에서도 유용성을 보입니다. 또한 단일 입력에 최적화된 스티어링 벡터가 다수의 입력에서 일반화된 행동을 유도할 수 있음을 입증하며, 이 스티어링 기법은 기존의 대조적 접근법보다도 더 강력한 대안을 제공할 수 있습니다.

- **Performance Highlights**: 연구팀은 최적화된 스티어링 벡터들이 저해를 유도하는 강력한 공격을 수행할 수 있는 능력을 보이며, Harmbench에서 96.9%의 성공률을 기록했습니다. 또한 스티어링 벡터가 허위 정보를 생성한 후 회복하는 능력을 어떻게 조정하는지를 분석하기 위한 새로운 평가 프레임워크를 개발하여, 입력의 다양성에 따라 다수의 벡터가 다른 결과를 이끌어낼 수 있음을 발견했습니다. 전체적으로 이번 연구는 LLM 행동과 활성화 공간의 구조 사이의 관계를 이해하는 데 중요한 기여를 할 것으로 평가됩니다.



### Reimagining Personal Data: Unlocking the Potential of AI-Generated Images in Personal Data Meaning-Making (https://arxiv.org/abs/2502.18853)
Comments:
          21 pages excluding reference and appendix. Accepted at ACM CHI 2025

- **What's New**: 본 논문은 개인 데이터의 의미를 생성하는 데 있어 AI가 생성한 이미지의 가능성을 탐구하며, 개인 데이터를 대안적인 시각적 형태로 변환하는 과정을 보여줍니다. 연구팀은 Open AI의 GPT-4 모델과 DALL-E 3를 활용해 개인 데이터를 생성된 이미지로 표현하는 웹 기반 애플리케이션을 설계하였습니다. 16명의 참가자를 대상으로 한 21일간의 일기 연구 및 인터뷰를 통해 AI 생성 이미지가 개인 데이터와의 깊은 연결을 가능하게 하는 방법을 조사했습니다.

- **Technical Details**: 연구에서는 AI가 생성한 이미지를 바탕으로 개인 데이터를 탐색하고, 개인적인 의미를 구성하는 과정에 대한 고찰이 이루어졌습니다. 참가자들은 데이터를 통해 감정적 층을 발견하고, 자신의 정체성을 재구성하며, 개인적 내러티브를 창출하는 경험을 공유했습니다. 이러한 경험은 개인 데이터와의 상호작용에서 AI 생성 이미지가 중요한 매개체가 될 수 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, AI 생성 이미지는 사용자가 자신의 데이터를 통해 깊이 있는 반성과 탐험을 하도록 유도함으로써 의미 구성에 긍정적인 영향을 미쳤습니다. 참가자들은 데이터를 기반으로 한 이미지를 통해 자기 발견과 관찰의 기회를 얻었으며, 새로운 형태의 데이터 경험을 탐색할 수 있었습니다. 이러한 결과는 앞으로 HCI(인간-컴퓨터 상호작용) 연구 및 디자인에서 AI 생성 이미지를 활용하는 방안에 대한 실질적인 시사점을 제공합니다.



### Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Cod (https://arxiv.org/abs/2502.18851)
- **What's New**: 이번 논문에서 제안하는 STONE은 AI가 생성한 코드의 워터마킹(watermarking) 기법으로, 코드의 기능을 유지하면서 비구문(non-syntax) 토큰에만 워터마크를 삽입합니다. 기존의 방법들은 코드의 기능적 무결성을 해칠 수 있는 구문을 수정했기 때문에 실용성이 제한되었습니다. STONE은 이러한 문제를 해결하기 위해 코드 실행에 중요한 토큰은 제외하고 고유한 패턴을 삽입하여 안정적인 탐지를 가능케 합니다.

- **Technical Details**: STONE은 비구문 토큰에 선택적으로 워터마크를 삽입하여 코드의 구조 및 기능을 보호합니다. 또한, CWEM(Code Watermarking Evaluation Metric)이라는 평가 기준을 도입하여 워터마킹 기법의 정확성(correctness), 탐지 가능성(detectability), 자연스러움(naturalness)을 평가합니다. 연구에서는 Shannon entropy를 계산해 고엔트로피(high-entropy) 토큰이 코드 품질에 미치는 영향을 분석했으며, 워터마크는 주로 etc 카테고리의 토큰에 삽입하여 효율성을 강화합니다.

- **Performance Highlights**: STONE은 Python, C++, Java를 포함한 여러 프로그래밍 언어에서 CWEM 기준으로 평균 7.69%의 성능 향상을 달성했습니다. 이 연구는 기존 최첨단 방법들과 비교하며, STONE의 접근법이 코드 품질을 유지하면서도 기능적 무결성을 확보하는 데 효과적임을 입증했습니다. 이러한 결과는 AI가 생성한 코드의 투명성과 책임성을 높이는 데 기여할 것으로 기대됩니다.



### A Causal Lens for Evaluating Faithfulness Metrics (https://arxiv.org/abs/2502.18848)
Comments:
          18 pages, 18 figures, 6 tables

- **What's New**: 이 논문에서는 LLMs의 자연어 설명의 신뢰성을 평가할 수 있는 Causal Diagnosticity라는 새로운 평가 프레임워크를 소개합니다. 여러 신뢰성 지표가 개발되었지만, 이들 사이를 비교할 수 있는 통합된 평가 체계가 부재하였기 때문에 이에 대한 해결책을 제시합니다. 특히, 신뢰성 지표의 진단적 특성을 평가하여, 더 신뢰할 수 있는 해석 가능성을 제공하는 방법론의 필요성을 강조합니다.

- **Technical Details**: Causal Diagnosticity 프레임워크는 신뢰성 메트릭을 평가하기 위한 새로운 방법론을 제안합니다. 이 프레임워크는 인과적 진단 가능성(causal diagnosticity) 개념을 활용하고, 모델 편집 기법을 사용하여 신뢰전 설명(pair)과 비신뢰전 설명(pair)을 생성합니다. 또한, 사실 확인(fact-checking), 유추(analogy), 객체 세기(object counting), 다중 단계 추론(multi-hop reasoning) 등의 네 가지 작업을 포함하는 벤치마크를 마련하였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 테스트한 모든 신뢰성 메트릭스는 종종 랜덤 베이스라인(random baseline)을 초과하지 못하는 경과를 보였습니다. 이는 신뢰성 메트릭 설계에서 진단적 접근(diagnosticity-first approach)의 필요성을 강조하며, 연속적인 신뢰성 점수를 산출하는 메트릭이 이진 점수에 비해 더 진단적이라는 사실을 발견하였습니다. 이 조사 결과는 LLM 하의 신뢰성 메트릭 향상과 해석 가능성 방법 개선을 요구합니다.



### Sliding Window Attention Training for Efficient Large Language Models (https://arxiv.org/abs/2502.18845)
Comments:
          14 pages, 5 figures

- **What's New**: 최근 Transformer 기반의 대형 언어 모델(LLM)이 여러 작업에서 뛰어난 능력을 발휘하고 있습니다. 그러나 이들의 시퀀스 길이에 따른 제곱 계산 복잡성은 긴 문서 처리에 있어 큰 병목 현상으로 남아있습니다. 이를 해결하기 위해 SWAT라는 효율적인 모델이 소개되었으며, Sliding Window Attention Training을 통해 긴 Context를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: SWAT는 기존의 softmax 연산을 sigmoid 함수로 교체하여 attention sink 문제를 방지하고, 각 토큰당 더 높은 정보 용량을 유지하도록 합니다. 또한, 균형 잡힌 ALiBi와 Rotary Position Embedding을 활용하여 정보 압축 및 보존을 최적화합니다. 보통의 Sparse Attention 기법인 Sliding Window Attention(SWA)에서 발생하는 문제를 해결하기 위한 훈련 방법론을 제안합니다.

- **Performance Highlights**: SWAT는 여덟 개 벤치마크에서 최첨단 선형 순환 아키텍처와 비교하여 SOTA 성능을 달성했습니다. 이 모델은 복잡한 아키텍처 없이도 효과적으로 정보를 유지하며, 효율적인 컴퓨팅을 통해 다양한 NLP 작업에서 강력한 성능을 보였습니다. 실험 결과, SWAT는 기존 Transformer 모델과 다른 순환 모델들을 초월하는 성과를 입증했습니다.



### BarkXAI: A Lightweight Post-Hoc Explainable Method for Tree Species Classification with Quantifiable Concepts (https://arxiv.org/abs/2502.18844)
- **What's New**: 이번 연구에서는 나무의 껍질 이미지 분류를 위한 시각적 모델 해석을 위한 경량화된 신후처리 방법인 BarkXAI를 제안합니다. 기존의 Explainable AI (XAI) 방법론이 한정적인 지역적 특징을 기반으로 한 설명에 그쳤던 것과 달리, 우리는 전 세계적인 시각적 특징을 quantifiable 개념을 사용하는 방식으로 설명하는데 초점을 두었습니다. 이를 통해 계산 비용을 줄이고 복잡한 개념을 수량화할 수 있게 됩니다.

- **Technical Details**: BarkXAI는 섬세하게 조정된 개념을 통해 나무 껍질 이미지와 같은 질감 기반 이미지 분류기를 해석하는 새로운 접근 방식을 제공합니다. 기존의 LIME이나 TCAV과는 달리, 외부 데이터셋에 의존하지 않고 파라미터화된 연산자를 사용함으로써 효율적인 개념 평가가 가능합니다. global visual features(전역 시각 특징)를 바탕으로 나무껍질 이미지의 smoothness(부드러움)나 tone(톤)과 같은 특성을 평가합니다.

- **Performance Highlights**: 실험 결과, 우리의 접근 방식은 TCAV와 Llama3.2 대비 개념 중요도 순위 평가에서 더 높은 성능을 보여주었으며, 이는 인간의 지각과의 우수한 정렬을 강조합니다. 이 연구는 나무의 껍질 이미지에서 전 세계적인 시각적 특징에 대한 개념 기반 설명을 제공한 첫 번째 연구로서, 향후 나무 품종 식별에 대한 해석 가능성을 크게 향상시킬 것으로 기대됩니다.



### Attention-Guided Integration of CLIP and SAM for Precise Object Masking in Robotic Manipulation (https://arxiv.org/abs/2502.18842)
Comments:
          2025 IEEE/SICE International Symposium on System Integration

- **What's New**: 이번 연구에서는 편의점 상품의 로봇 조작을 위한 객체 마스킹의 정밀성을 높이는 혁신적인 파이프라인을 소개합니다. 여기서는 CLIP(Contrastive Language-Image Pretraining)와 SAM(Segment Anything Model)이라는 두 가지 고급 AI 모델의 시너지 효과를 활용하며, 멀티모달 데이터(이미지 및 텍스트)를 효과적으로 사용합니다. 이러한 통합은 객체 마스킹의 성능을 개선하는 데 크게 기여하며, 로봇 시스템에 보다 정밀한 입력을 제공합니다.

- **Technical Details**: 제안하는 파이프라인은 최적화된 데이터셋과 작업별 미세 조정을 통해 기존 파이프라인이 직면한 데이터 제한 사항을 극복합니다. CLIP과 SAM을 gradient-based attention 메커니즘과 통합하여, 편의점처럼 객체가 다양하고 복잡한 환경에서도 신뢰성과 정확성을 높입니다. 이러한 방식으로 파라미터 설정을 통해 로봇 시스템의 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 결과적으로, 새로운 방법론은 편의점에서의 로봇 작업에 필요한 객체 식별 및 조작의 정밀성을 개선하여 더 복잡한 작업을 효과적으로 수행하도록 합니다. 실제 환경에서의 적용 가능성을 높이며, 로봇 시스템이 다루는 물체에 대해 보다 세밀하고 적응적인 마스킹을 가능하게 합니다. 또한, 제안된 프레임워크는 다양한 맥락에서의 객체 조작의 신뢰성을 크게 향상시킬 것으로 기대됩니다.



### BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction (https://arxiv.org/abs/2502.18807)
Comments:
          Under review

- **What's New**: 이번 연구는 Battery Life Prediction (BLP)을 위한 BatteryLife라는 대규모 데이터 세트와 벤치마크를 제안합니다. BatteryLife는 16개의 데이터 세트를 통합하여 기존 데이터 세트보다 샘플 크기를 2.4배 증가시키고, 8개의 형식, 80개의 화학 시스템, 12개의 작동 온도, 646개의 충전/방전 프로토콜에 걸쳐 다양한 배터리 데이터를 제공합니다. 특히, 이 데이터 세트는 국내외에서 테스트된 대용량 리튬 이온 배터리, 아연 이온 배터리 및 나트륨 이온 배터리의 배터리 수명 데이터를 최초로 출시합니다.

- **Technical Details**: BatteryLife는 포괄적인 실험 설정을 제공하며, 충전 및 방전 프로토콜에 따라 1회에서 100회까지 다양한 사이클을 포함한 BLP를 지원합니다. 이 데이터 세트는 현재의 다른 시계열 분야에서 인기 있는 방법들을 포함하는 종합적인 벤치마크로 기능합니다. 또한 CyclePatch라는 플러그인 기술을 제안하였으며, 이는 비선형 용량 손실을 모델링하는 데 도움을 주고, 모든 데이터 세트에서의 성능 향상에 기여할 수 있습니다.

- **Performance Highlights**: BatteryLife의 벤치마크 결과, 타 시계열 분야에서 인기 있는 많은 모델들이 BLP에 적합하지 않음을 보여주었습니다. CyclePatch는 모델 성능을 지속적으로 향상시켜 최첨단 기준을 수립하는 데 기여하게 됩니다. 이 데이터 세트는 또한 노화 조건 및 도메인 전반에 걸쳐 모델 성능을 평가할 수 있는 기능을 제공합니다.



### ANPMI: Assessing the True Comprehension Capabilities of LLMs for Multiple Choice Questions (https://arxiv.org/abs/2502.18798)
- **What's New**: 이 논문에서는 언어 모델의 자연어 이해 능력을 평가하기 위한 새로운 메트릭 ANPMI를 제안합니다. ANPMI는 Pointwise Mutual Information (PMI)을 $-	ext{log} P(Choice)$에 의해 정규화하여 모델이 프롬프트를 정확히 이해하고 있는지를 측정합니다. 이 접근법은 기계가 프롬프트를 완전히 이해하지 않고도 답변을 선택하는 경우의 문제를 해결합니다.

- **Technical Details**: 현재 언어 모델의 평가는 주로 다중 선택 질문을 통해 이루어지며, P(Choice|Prompt) 확률을 기반으로 올바른 답변을 선택하는 빈도로 측정합니다. 그러나 이러한 방식은 모델이 프롬프트를 실제로 이해했는지를 보여주지 않습니다. ANPMI는 P(Choice) 불균형 문제를 보다 정확하게 교정하여 모델의 자연어 이해를 평가할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안한 ANPMI 메트릭은 여러 사전 훈련된 모델과 벤치마크를 사용하여 기존 방법들보다 모델의 프롬프트 이해도를 훨씬 더 정확하게 평가하는 것으로 나타났습니다. 이를 통해 모델의 진정한 이해도를 측정하기 위한 새로운 기준을 제시하며, 언어 이해 능력 평가의 신뢰성을 향상시키는 데 기여합니다.



### Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs (https://arxiv.org/abs/2502.18791)
Comments:
          21 pages, 9 figures

- **What's New**: 본 연구는 대규모 언어 모델(LLM) 연구의 메타 분석을 위한 반자동화 접근 방식을 제안합니다. 이 방법은 관련 arXiv 논문을 자동으로 식별하고 실험 결과와 관련된 속성을 추출하여 구조화된 데이터셋으로 정리합니다. 이를 통해 논문 조사 및 데이터 추출의 수고를 93% 이상 줄일 수 있습니다.

- **Technical Details**: 데이터 추출 과정은 세 가지 단계인 전처리 및 필터링, 추출 및 증강, 설명 생성으로 나뉩니다. 연구에 사용된 모델은 GPT-4o 와 Gemini 1.0 Pro와 같은 최첨단 LLM입니다. 주요 성능 관련 속성으로는 데이터셋 이름, 모델 이름, 프롬프트 방법, 메트릭 이름 등이 포함됩니다.

- **Performance Highlights**: 이번 연구에서 자동으로 생성된 데이터셋은 이전의 수동 메타 분석 결과를 정확히 재현하며, Chain-of-Thought 방식이 주로 수학 및 기호적 추론 작업에서 이점을 제공한다는 것을 입증했습니다. 또한 세 가지 새로운 통찰력으로, 인맥 예시가 다중 모달 작업에 기여하지만 수학 작업에서는 제한적인 이점을 나타낸다는 점과 CoT와 ICL이 포함된 경우가 그렇지 않은 경우보다 전반적으로 우수하다는 결과가 도출되었습니다.



### NeuroTree: Hierarchical Functional Brain Pathway Decoding for Mental Health Disorders (https://arxiv.org/abs/2502.18786)
- **What's New**: 본 연구에서는 뇌 기능 네트워크를 분석하기 위해 NeuroTree라는 새로운 프레임워크를 제안합니다. NeuroTree는 k-hop AGE-GCN과 신경 미분 방정식(ODEs)을 통합하여 뇌 질환 분류를 위한 동적 기능 연결성(FC) 특징 학습을 향상시키는 방법입니다. 특히, 이 모델은 fMRI 네트워크를 트리 구조로 변환하여 고차원 뇌 지역 경로 특징을 포착하고, 정신 질환과 관련된 뇌 하위 네트워크 이해에 필수적인 계층적 신경 행동 패턴을 식별할 수 있습니다.

- **Technical Details**: NeuroTree는 기존 그래프 합성곱 신경망(GCN)의 해석 가능성을 개선하여 나이, 성별과 같은 인구 통계적 변수와 복잡한 관계를 효과적으로 이해할 수 있도록 설계되었습니다. 이 프레임워크는 k-hop ODE-GCN을 활용하여 인접하고 먼 뇌 영역 간의 상호작용을 포착하며, 다양한 정신 질환 아형의 수렴을 측정합니다. 또한, 대조적 마스킹 최적화(contrastive masking optimization)를 통해 핵심 기능적 연결 패턴을 식별하고, 훈련 가능한 k-hop 그래프 합성곱을 사용하여 최적의 고차 트리 경로를 식별합니다.

- **Performance Highlights**: 이 연구는 두 가지 정신 질환 데이터셋에서 최첨단 성능을 보여주었으며, 중독 및 정신 질환과 관련된 FC 뇌 나이 변화 패턴을 효과적으로 예측하고 해석하는 데 기여했습니다. NeuroTree는 뇌 질환 예측 및 그 기저에 있는 신경 메커니즘을 설명하는 데 있어 널리 활용될 수 있음을 입증했습니다. 이러한 결과는 정신 질환 및 중독에 대한 이해와 임상 진단에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Research on Edge Computing and Cloud Collaborative Resource Scheduling Optimization Based on Deep Reinforcement Learning (https://arxiv.org/abs/2502.18773)
- **What's New**: 이번 연구에서는 딥 강화 학습(Deep Reinforcement Learning, DRL)을 활용하여 에지-클라우드 협업 컴퓨팅에서 자원 스케줄링 최적화 문제를 다루고 있습니다. 제안된 DRL 기반 접근 방식은 작업 처리 효율성을 개선하고, 전체 처리 시간을 단축하며, 자원 활용도를 향상시키고, 작업 이전을 효과적으로 제어합니다. 기존의 스케줄링 알고리즘에 비해 DRL의 우수성을 실험 결과를 통해 입증하였습니다.

- **Technical Details**: 이 연구는 복잡한 작업 할당 문제, 동적 작업 부하, 여러 자원 제약을 관리하는 데 있어 DRL이 탁월한 성능을 발휘한다는 점을 강조합니다. 그러나 학습 효율성을 개선하고, 훈련 시간을 줄이며, 수렴(convergence) 문제를 해결하기 위해 추가적인 개선이 필요합니다. 향후 연구는 알고리즘의 결함 허용률(fault tolerance)을 증가시켜 보다 복잡하고 불확실한 스케줄링 시나리오를 처리하는 데 초점을 맞춰야 합니다.

- **Performance Highlights**: 실험 결과는 복잡한 작업 할당과 동적 로드 관리에서 DRL의 효과성을 강조합니다. 특히 다양한 자원 제약 조건을 효과적으로 관리할 수 있는 점이 탁월합니다. 이러한 성능 개선은 에지-클라우드 컴퓨팅 시스템의 지능(intelligence)과 효율성(efficiency)을 더욱 향상시킬 수 있는 잠재력을 보여줍니다.



### Reward Shaping to Mitigate Reward Hacking in RLHF (https://arxiv.org/abs/2502.18770)
Comments:
          19 pages

- **What's New**: 이번 연구에서는 인간 피드백을 통한 강화학습(Reinforcement Learning from Human Feedback, RLHF)의 문제를 해결하기 위한 새로운 보상 형태인 Preference As Reward (PAR)를 제안합니다. PAR는 보상 모델에 내재된 선호를 활용하여 RL의 신호를 형성하는 혁신적인 방법으로, 이를 통해 기존의 보상 해킹(reward hacking) 문제를 극복할 수 있습니다. 이 연구는 보상 형성 방법에 대한 체계적인 분석을 제공하며, 세 가지 주요 설계 원칙을 도출했습니다.

- **Technical Details**: 우리는 RLHF에서 광범위하게 사용되는 보상 형성 기법들을 조사하고, 이러한 기법의 적용을 위해 잘 정의된 설계 원칙을 설정하려고 했습니다. 특히 보상 모델의 특정 임계값을 초과할 경우 보상 해킹이 시작되며 모델의 승률이 감소하는 경향을 관찰했습니다. 그리고 PAR는 이론적으로 시그모이드 함수를 사용하여 중앙 보상(centered reward)을 적용함으로써 초기 학습을 가속화하고 학습 안정성을 보장합니다.

- **Performance Highlights**: PAR는 두 개의 기본 모델(Gemma2-2B 및 Llama3-8B)과 두 개의 데이터셋(Ultrafeedback-Binarized 및 HH-RLHF)에서 평가된 결과, 다른 보상 형성 방법보다 뛰어난 성능을 보여주었습니다. AlpacaEval 2.0 벤치마크에서 PAR는 경쟁 상대보다 최소 5%포인트 높은 승률을 기록하였으며, 데이터 효율성도 뛰어나 단일 참조 보상만으로 최적의 성능을 낼 수 있습니다. 또한 두 번의 훈련 에포크 후에도 보상 해킹에 대한 강인성을 유지하는 특성을 가지고 있습니다.



### Online Prototypes and Class-Wise Hypergradients for Online Continual Learning with Pre-Trained Models (https://arxiv.org/abs/2502.18762)
Comments:
          Under review

- **What's New**: 이 논문은 지속 학습(Continual Learning, CL)의 온라인 버전인 onCL에서 Pre-Trained Models (PTM)를 더욱 효과적으로 활용하는 방법을 제안합니다. 특히, 작업 변화에 대한 정보가 없고 각 데이터에 대해 한 번만 관찰 가능한 상황에서 작업 경계의 부재 문제와 하이퍼파라미터 최적화 문제를 동시에 해결하고자 합니다. Online Prototypes (OP)와 Class-Wise Hypergradients (CWH)를 통해 정확도를 향상시킬 수 있음을 보였습니다.

- **Technical Details**: 본 연구에서 제안한 OP는 PTM의 안정적인 출력을 바탕으로 모델의 최종 계층을 조정하여 객체 인식과 유사한 방식으로 작업 경계를 필요로 하지 않으며, 기존 데이터를 저장하지 않고도 replay sample로 활용할 수 있습니다. 또한 CWH는 훈련 과정에서 클래스별 gradient 계수를 학습함으로써 비최적 학습률 문제를 해결할 수 있도록 합니다. 이러한 방법론은 기존 오프라인 CL 접근 방식과 함께 사용할 때의 성능 향상을 보여줍니다.

- **Performance Highlights**: 실험 결과, OP와 CWH를 통합하면 기존 방법과 비교하여 정확도가 일관되게 향상되는 것을 확인했습니다. 이러한 접근 방식은 특히 여러 데이터셋과 초기 학습률에서 PTM을 활용하여 onCL에서의 성과를 개선하는 데 중점을 두고 있습니다. 논문은 수용된 이후 코드도 완전히 공개할 예정입니다.



### Learning Autonomy: Off-Road Navigation Enhanced by Human Inpu (https://arxiv.org/abs/2502.18760)
- **What's New**: 이 연구에서는 자율 주행의 새로운 도전 과제인 오프로드 환경에서의 내비게이션 문제를 해결하기 위해 인간의 드라이빙 방식에서 배운 교훈을 활용한 학습 기반의 로컬 플래너(local planner)를 제안합니다. 이 플래너는 모노큘러 카메라만을 사용하여 실제 시연에서 인간 주행의 뉘앙스를 직접 캡처할 수 있는 능력을 갖추고 있습니다. 기존의 방안과는 달리, 이 플래너는 최소한의 인간 시연 데이터(5-10분)로 빠르게 오프로드 환경을 탐색할 수 있도록 학습합니다.

- **Technical Details**: 연구의 핵심은 복잡한 도로 상호작용 모델링 없이 인간 주행 데이터를 학습하여 내비게이션 선호도를 도출하는 것입니다. 인간의 시연 데이터를 활용하여 유틸리티 함수를 사용하여 주요 특징을 추출하고, 복잡한 지형을 포함하는 다양한 오프로드 조건에서 효과적으로 작동할 수 있도록 설계되었습니다. 이 접근 방식은 전통적인 방법과는 달리 대량의 레이블링된 데이터나 정밀한 센서 교정이 필요하지 않고, 5-10분 조차 배우는 데 필요한 학습 데이터 양을 크게 줄입니다.

- **Performance Highlights**: 이 알고리즘은 다양한 오프로드 환경에서 신속하게 적응하고, 사람과 유사한 의사결정 과정을 통해 복잡한 지형 구성에서도 직관적인 내비게이션 선택을 할 수 있습니다. 또한, 수동 조정의 필요성을 낮추므로 다양한 환경에서의 배치를 더 쉽게 할 수 있습니다. 이러한 특성 덕분에 전통적인 접근 방식에 비해 데이터 요구사항이 대폭 감소하며, 운전 패턴 학습이 효율적으로 이루어질 수 있습니다.



### AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms (https://arxiv.org/abs/2502.18754)
Comments:
          8 pages, 10 figures, in Proceedings of the ACM Web Conference 2025 (WWW '25)

- **What's New**: AgentSociety Challenge는 웹 컨퍼런스에서 처음으로 개최되는 대회로, 대형 언어 모델(LLM) 에이전트를 활용하여 사용자 행동을 모델링하고 추천 시스템을 개선하는 데 중점을 두고 있습니다. 참가자들은 Yelp, Amazon과 Goodreads의 통합 데이터 세트를 활용하여 혁신적인 LLM 에이전트를 개발하는 과제가 주어집니다. 총 295개 팀이 참여하여 1,400건 이상의 제출물이 있었으며, 각 트랙에서 성과 개선을 달성했습니다.

- **Technical Details**: 이 대회는 두 가지 트랙으로 나뉘어 있습니다: 사용자 모델링 트랙(User Modeling Track)과 추천 트랙(Recommendation Track)입니다. 사용자 모델링 트랙에서는 특정 항목에 대한 사용자 리뷰와 별점 생성을 시뮬레이션하는 에이전트를 디자인해야 합니다. 추천 트랙에서는 역사적 상호작용과 정보를 기반으로 사용자 맞춤형 추천을 제공하는 LLM 에이전트를 개발합니다.

- **Performance Highlights**: 참가자들은 개발 단계에서 사용자 모델링 트랙 21.9%, 추천 트랙 20.3%의 성능 개선을 달성하였고, 최종 단계에서는 각각 9.1%, 15.9%의 개선을 기록했습니다. 대회는 정보 검색, 추천 시스템, 사용자 모델링 등의 주제로 최신 LLM 에이전트를 활용하여 복잡한 인간 행동을 예측하고 시뮬레이션할 수 있는 기회를 제공합니다.



### Intent Tagging: Exploring Micro-Prompting Interactions for Supporting Granular Human-GenAI Co-Creation Workflows (https://arxiv.org/abs/2502.18737)
Comments:
          31 pages, 30 figures, 3 tables. To appear in the Proceedings of the 2025 ACM CHI Conference on Human Factors in Computing Systems, Yokohama, Japan

- **What's New**: 최근 생성적 AI(GenAI) 시스템이 콘텐츠 창작에서 놀라운 가능성을 보이고 있지만, 사용자들은 이러한 시스템을 창의적인 워크플로우에 통합하는 데 어려움을 겪고 있습니다. 특히 사용자 의도와 AI가 생성하는 콘텐츠 간의 비일치, 효과적인 프롬프트 формуляция, 다각적인 작업 흐름의 유연성 부족이 주요 문제로 지적됩니다. 이 문제를 해결하기 위해 새로운 시스템인 IntentTagger를 개발하였고, 이는 사용자 의도를 나타내는 소형 개념 단위인 Intent Tags를 이용하여 인간-GenAI 협업을 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: IntentTagger는 슬라이드 프레젠테이션을 만들기 위해 설계된 GenAI 기반 시스템입니다. 사용자는 2D 캔버스 인터페이스에서 Intent Tags를 사용하여 슬라이드 생성 과정을 제어할 수 있으며, 여기서 각 Intent Tag는 사용자의 특정 의도를 표현합니다. 이 시스템은 LLM(대형 언어 모델)을 활용하여 슬라이드를 생성하며, 사용자에게 유용한 태그 제안, 동적 드롭다운 목록, 상호작용 가능한 슬라이더 위젯과 같은 적응형 UI 요소를 동적으로 생성합니다.

- **Performance Highlights**:  사용자 연구 결과에서 참여자들은 Intent Tag 기반의 상호작용이 기존의 채팅 기반 및 디자인 갤러리 기반 GenAI 시스템보다 더 높은 만족감과 통제를 제공했다고 주장했습니다. 특히 비선형 및 반복적인 작업 흐름을 지원하며, 다양한 수준의 모호함으로 의도를 유연하게 표현할 수 있는 점이 긍정적으로 평가되었습니다. 이러한 시스템 제안이 작업을 수행하는 데 있어 유용하고 방해가 되지 않는 보조 역할을 하고 있다는 점도 주요 발견으로 나왔습니다.



### AI-Instruments: Embodying Prompts as Instruments to Abstract & Reflect Graphical Interface Commands as General-Purpose Tools (https://arxiv.org/abs/2502.18736)
Comments:
          18 pages, 10 figures. To appear in the Proceedings of the 2025 ACM CHI Conference on Human Factors in Computing Systems, Yokohama, Japan. this https URL

- **What's New**: 이 논문에서는 AI 기반 인터페이스를 개선하기 위해 'AI-Instruments'라는 새로운 개념을 제안합니다. 이는 사용자 의도를 재사용 가능한 조작 도구로 구현하고, 다양한 해석을 반영하며, 특정 사례를 바탕으로 도구를 설치하는 원칙 세 가지( Reification, Reflection, Grounding)를 기반으로 합니다. 특히, LLM(대형 언어 모델)을 활용하여 새로운 도구를 생성하고 개선하는 시스템을 구현해 기존의 규정된 기능을 넘어서는 가능성을 제시합니다.

- **Technical Details**: AI-Instruments는 사용자 의도를 다루기 위해 네 가지 기술 시제를 사용합니다: (1) 'Fragments': 프롬프트를 분해하여 재구성 가능한 객체로 나타냅니다. (2) 'Transformative Lenses': 여러 콘텐츠 요소를 바탕으로 새로운 콘텐츠를 생성하는 프레임워크로, 장면의 유연한 재구성을 지원합니다. (3) 'Generative Containers': 이미지, 텍스트 및 도구의 여러 대안을 생성하며, (4) 'Fillable Brushes': 프롬프트를 캡슐화하여 선택한 콘텐츠로 채워줍니다. 이러한 도구들은 사용자의 직접 조작을 통해 AI의 생성 과정을 더욱 직관적으로 만들어줍니다.

- **Performance Highlights**: 테스트 결과, 12명의 참가자가 AI-Instruments를 사용하면서 기존의 대화형 프롬프트에서의 한계를 극복하고 직접 조작의 장점을 경험했습니다. 이 모델은 사용자 인터페이스의 여러 도전 과제를 해결할 수 있는 가능성을 보여주었으며, 사용자는 의도 수립, 프롬프트 공학, 비선형 워크플로우 및 의도 해결 측면에서 더욱 유연한 경험을 했습니다. 궁극적으로 AI-Instruments는 생성적 AI와의 상호작용을 개선할 수 있는 새로운 접근법으로 자리잡을 가능성이 큽니다.



### Cross-Modality Investigation on WESAD Stress Classification (https://arxiv.org/abs/2502.18733)
- **What's New**: 이 연구는 WESAD 데이터셋을 사용하여 스트레스 감지를 위한 변형기를 개발하였으며, 심전도(ECG), 전기피부활동(EDA), 근전도(EMG), 호흡률(RESP), 온도(TEMP), 3축 가속도계(ACC) 신호를 이용하여 훈련하였습니다. 단일 모달리티(transformer) 모델을 통해 특정 생리적 신호 분석에서 혁신적인 성과를 도출하며, 스트레스 감지의 정확도는 $99.73\%$에서 $99.95\%$에 이릅니다.

- **Technical Details**: 연구에서는 다양한 생리적 데이터를 통한 스트레스 감지의 성능을 평가하기 위해 변형기(transformers) 모델을 개발하였으며, 해당 모델은 고급 신경망 구조를 활용하여 여러 생리적 신호 간의 복합적인 관계를 학습합니다. 다양한 모달리티에서의 데이터 품질과 양이 모델 효율성에 영향을 미친다는 점을 강조하고 있습니다. 이를 통해 기존의 고전적인 기계 학습 기법들과 비교했을 때 최소한의 전처리로도 높은 정확도와 정밀도를 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: 이 연구는 WESAD 데이터셋을 활용하여 다중 클래스 스트레스 감지에서 최신 기술 성과를 달성하였으며, 이는 중립, 스트레스 및 재미로 분류됩니다. 또한 단일 모달리티(transformer) 주목(attention) 기법의 효과를 강조하며, 고도화된 계산 자원 없이도 생리적 신호의 분석에서 우수한 성과를 보여줍니다. 처음으로 다양한 모달리티 간 성능 차이를 탐구하고, 훈련된 변형기 모델의 임베딩 공간을 통해 이 현상을 설명하여 향후 연구에 기여할 중요한 통찰을 제공하고자 하였습니다.



### Deep-Bench: Deep Learning Benchmark Dataset for Code Generation (https://arxiv.org/abs/2502.18726)
- **What's New**: 최근 딥러닝(Deep Learning) 시스템을 개발하는 데서의 복잡성을 해결하기 위해, 본 논문에서는 DeepBench라는 새로운 벤치마크 데이터셋을 소개합니다. DeepBench는 다양한 딥러닝 문제를 세 가지 주요 측면에 따라 분류합니다: 전처리(pre-processing), 모델 구축(model construction), 학습(training)과 같은 단계, 분류(classification), 회귀(regression), 추천(recommendation)과 같은 작업, 그리고 표 형식(tabular), 이미지(image), 텍스트(text)와 같은 입력 데이터 타입입니다. 기존의 DS-1000 벤치마크에 비해 더 종합적인 딥러닝 파이프라인을 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: DeepBench는 520개의 AI 및 딥러닝 데이터 포인트로 구성되어 있으며, 30개의 GitHub 저장소에서 추출된 코드들을 통해 신뢰성 있는 샘플을 제공합니다. 데이터는 전처리, 모델 구축 및 학습 등 여러 단계로 나뉘어 있으며, 각 엔트리는 입력 데이터 타입에 따라 카테고리화되어 있습니다. 이 데이터셋은 딥러닝 코드 생성 과정에서 발생하는 여러 가지 문제와 버그를 분류하는 체계를 구축하여 LLM(대형 언어 모델)이 겪는 고유한 도전 과제를 조명합니다.

- **Performance Highlights**: GPT-4o와 같은 최신 LLM이 DeepBench에서 31%의 정확성을 기록했으며, 이는 DS-1000에서의 60%에 비해 현저히 낮은 수치입니다. Claude, LLaMA, Mistral와 같은 다른 LLM에서도 유사한 어려움이 관찰되었습니다. 이러한 결과는 DeepBench가 더 복잡한 데이터 세트를 포함하고 있음을 강조하며, 각 카테고리 간 성능의 큰 차이는 LLM의 DL 코드 생성 능력을 개선하는 데 있어 유용한 통찰을 제공합니다.



### Bridging Critical Gaps in Convergent Learning: How Representational Alignment Evolves Across Layers, Training, and Distribution Shifts (https://arxiv.org/abs/2502.18710)
- **What's New**: 본 연구는 인공지능(AI)과 생물학적 신경망 간의 공통된 표현(Representations) 개발을 이해하는 데 중요한 기여를 합니다. 기존 연구에서는 시각 네트워크의 초기 및 후기 층에서 수렴(Convergence)을 관찰했으나, 이론적 격차가 여전히 존재합니다. 특히, 적절한 정렬을 위해 필요한 변환 불변성(Transformation Invariance)을 간과한 제한된 메트릭(metric)에 의존했습니다.

- **Technical Details**: 본 연구는 세 가지 메트릭을 비교하였습니다: 선형 회귀(Linear Regression)는 아핀 변환을 무시하고, 프로크루스테스(Procrustes)는 회전과 반사를, 순열 및 소프트 매칭(Permutation/Soft-Matching)은 단위 순서를 무시합니다. 흥미롭게도, 직교 변환(Orthogonal Transformations)은 보다 유연한 선형 변환 못지않게 표현을 잘 정렬시킵니다. 연구 결과, 거의 모든 수렴은 첫 번째 에폭(epoch) 내에 발생하며, 이는 네트워크가 최적의 성능을 달성하기 훨씬 이전입니다.

- **Performance Highlights**: 연구 결과, 입력 통계(Input Statistics)의 변화가 정렬에 미치는 영향을 체계적으로 조사하지 않은 이전 연구의 한계를 보완합니다. 특히, 분포 외(Out-of-Distribution, OOD) 입력이 후기 층에서의 차이를 일관되게 증폭시키는 반면, 초기 층은 분포 내(In-Distribution)와 OOD 입력 모두에 대해 정렬되어 있다는 것을 보여줍니다. 이러한 발견은 신경과학과 AI 분야에서 표현 수렴에 대한 이해를 심화시킬 뿐만 아니라, 두 분야에서의 모델 디자인에도 중요한 시사점을 제공합니다.



### H-FLTN: A Privacy-Preserving Hierarchical Framework for Electric Vehicle Spatio-Temporal Charge Prediction (https://arxiv.org/abs/2502.18697)
Comments:
          14 pages, 7 tables, 2 figures, Journal Paper

- **What's New**: 본 논문에서는 전기차(EV) 충전 시간 예측, 사용자 개인정보 보호 및 자원 관리 문제를 해결하기 위한 새로운 접근법인 Hierarchical Federated Learning Transformer Network (H-FLTN) 프레임워크를 제안합니다. H-FLTN은 전기차, 커뮤니티 분산 에너지 자원 관리 시스템(DERMS), 에너지 공급자 데이터 센터(EPDC)로 구성된 3단계 계층 구조를 통해 EV 충전 수요에 대한 정확한 공간-시간 예측을 가능하게 하며, 사용자 개인정보를 보호합니다.

- **Technical Details**: H-FLTN은 동적 클라이언트 캡핑 메커니즘(DCCM)과 클라이언트 회전 관리(CRM)를 포함하여 교육 효율성과 자원 관리를 개선합니다. DCCM은 클라이언트 참여를 최적화하며, CRM은 교육 세대 간의 공정한 참여를 보장하는 역할을 합니다. 이를 통해 H-FLTN은 점차 증가하는 EV 수에 따라 교육 시간의 복잡성을 선형에서 상수로 줄이는 데 성공하였습니다.

- **Performance Highlights**: 대규모 차량 이동 데이터를 기반으로 한 시뮬레이션 결과에 따르면, H-FLTN은 에너지 수요 예측 및 자원 할당을 향상시켜 스마트 시티 인프라에 통합될 때 그 효율성이 증가합니다. 이 프레임워크는 EV들이 참여하는 분산 학습 환경에서의 개인정보 보호와 자원 관리 문제를 동시에 해결하여, 지속 가능한 도시 시스템으로의 통합을 촉진하는 데 기여합니다.



### Policy-as-Prompt: Rethinking Content Moderation in the Age of Large Language Models (https://arxiv.org/abs/2502.18695)
Comments:
          14 pages, 5 figures

- **What's New**: 이 논문은 콘텐츠 조정(content moderation)의 패러다임을 변화시키는 'policy-as-prompt' 프레임워크를 제안하고, 이와 관련된 다섯 가지 주요 도전 과제를 다룹니다. 최근의 대형 언어 모델(LLMs)의 발전으로 인해, 이러한 정책을 텍스트 입력으로 직접 해석할 수 있게 되면서 데이터 큐레이션의 필요성이 사라지고 있습니다. 이 접근법은 자연어로 상호작용을 통해 조정할 수 있는 유연성을 제공합니다.

- **Technical Details**: 본 연구에서는 콘텐츠 조정 정책을 자연어 프롬프트로 인코딩하는 새로운 방법을 제시하며, 기술적 구현, 사회기술적 요소, 조직적 변화, 거버넌스 문제 등 네 가지 주요 도메인에서 도전 과제를 분석합니다. 각 도전 과제는 기술적 구현(정책을 프롬프트로 변환하는 것 및 프롬프트 구조와 형식의 민감도), 사회기술적 측면(정책 형성에서의 기술적 결정론), 조직적 측면(정책과 머신러닝 팀 간의 역할 변화), 그리고 거버넌스(모델 거버넌스 및 책임성)에 해당합니다.

- **Performance Highlights**: 효과적인 콘텐츠 조정을 위해서는 작동화(operationalisation)와 정책 집행(enforcement)이 필요하며, 이 둘은 상호 의존적입니다. 기술은 콘텐츠 조정의 발전에 중요한 역할을 하며, 특히 머신러닝 알고리즘은 조정 파이프라인의 중심이 되고 있습니다. 현재의 언어 모델을 이용한 접근 방식은 조정 워크플로우를 혁신적으로 바꾸고 있으며, 기존의 프레임워크와 비교할 때 더 나은 유연성과 적응성을 제공합니다.



### AI Mismatches: Identifying Potential Algorithmic Harms Before AI Developmen (https://arxiv.org/abs/2502.18682)
Comments:
          CHI Conference on Human Factors in Computing Systems (CHI '25), April 26-May 1, 2025, Yokohama, Japan

- **What's New**: 이번 논문에서는 AI 시스템이 기대에 미치지 못할 때 발생하는 'AI 불일치(AI Mismatch)'의 개념을 다룹니다. AI의 실제 성능이 안전과 가치 창출에 필요한 성과 기준에 미치지 못하는 경우가 많다는 점을 강조합니다. 그리하여 초기 단계에서 이러한 불일치를 식별하고 완화하기 위한 접근법을 제안하여 AI 개발에서의 위험을 줄이는 방법을 모색합니다.

- **Technical Details**: 저자는 774개의 AI 사례를 분석하여, AI 불일치의 핵심 요소를 추출하고 이를 바탕으로 7개의 행렬을 개발하였습니다. 이러한 행렬들은 위험 요소 간의 관계를 시각화하고, 높은 위험 영역을 강조합니다. 또한, 모델 성능을 인간 중심의 정의로 접근하여, 사용자의 필요를 충족할 수 있는 작업 수행 능력에 초점을 맞추고 있습니다.

- **Performance Highlights**: 이 연구는 AI 개발 초기 단계에서의 위험 요소를 예측 및 관리하는 방법을 통해, AI 시스템이 무의도적 피해 최소화를 목표로 할 수 있도록 돕습니다. 사례 연구를 통해 AI 불일치를 드러내는 우리의 접근법은 초기 개념 선택 단계에서 중요 위험 요소를 파악하는 데 유용함을 보여줍니다. 이는 AI 혁신 팀이 보다 안전한 구역으로 개념을 이끌 수 있도록 지원할 것입니다.



### Comparing Native and Non-native English Speakers' Behaviors in Collaborative Writing through Visual Analytics (https://arxiv.org/abs/2502.18681)
Comments:
          accepted by CHI 2025

- **What's New**: 이번 논문에서는 모국어 화자(NS)와 비모국어 화자(NNS) 간의 협업 쓰기 동태를 이해하기 위한 시각적 분석 도구인 	extsc{COALA}를 소개합니다. 162회의 쓰기 세션을 분석하여 두 그룹의 행동 차이를 정의하고, 이 과정에서 데이터의 복잡성과 자동화 방법이 초래하는 불확실성을 극복하는 방법을 개발하였습니다. 	extsc{COALA}는 작가 클러스터의 불확실성을 시각화하고, 행동 요약을 생성하며, 여러 수준의 세부 사항에서 쓰기 관련 행동을 시각화할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: 논문에서는 	extsc{COALA}라는 새로운 시각적 분석 도구를 개발하고 이를 통해 NS와 NNS 간의 협업 쓰기 행동을 비교합니다. 이 도구는 대형 언어 모델(LLM)을 사용하여 클러스터 요약을 생성하고, 여러 클러스터 결과의 불확실성을 표시함으로써 사용자가 더 효과적으로 데이터 해석을 할 수 있도록 지원합니다. 저자들은 또한 소통에 관련된 전문가들과 긴밀히 협력하여 데이터 모델과 작업 요구 사항을 정립하고, 시각화 디자인을 반복하여 평가하였습니다.

- **Performance Highlights**: 사용자 연구를 통해 	extsc{COALA}의 효과성을 검증하였습니다. 2+2명의 도메인 전문가와 8명의 관련 경험이 있는 연구자들이 참여하여, 이 도구를 통해 NS와 NNS의 행동 차이를 분석하고 그 과정에서 발견한 통찰력을 공유하였습니다. 또한, 협업 쓰기 과정뿐 아니라 AI 지원 협업 도구의 향후 발전을 위한 설계 교훈을 제시합니다.



### Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Suppor (https://arxiv.org/abs/2502.18658)
- **What's New**: 이번 연구에서는 Codellaborator라는 프로토타입 LLM 에이전트를 소개합니다. 이 에이전트는 코딩 환경에서의 에디터 활동과 작업 맥락에 따라 프로그래밍 지원을 자동으로 개시합니다. 연구팀은 프롬프트 전용, 적극적인 에이전트, 그리고 존재감 및 맥락이 포함된 적극적인 에이전트라는 세 가지 인터페이스 변형을 탐색하여 AI 지원의 특징들을 비교했습니다.

- **Technical Details**: Codellaborator는 자율적으로 상호작용할 수 있는 능력을 가진 기술 프로브로, 다양한 사용자 활동에 대응하여 메시지를 통해 상호작용을 시작하고, 직접 코드 편집을 수행할 수 있습니다. 연구에서는 코딩 작업과 에디터 환경의 맥락에서 지원 타이밍을 도입하는 세 가지 디자인 원칙을 도출하였습니다. 참여자들은 Codellaborator의 시각적 존재감과 유연한 상호작용 범위가 가져오는 긍정적인 영향을 경험했습니다.

- **Performance Highlights**: 연구 결과, Codellaborator는 높은 수준의 프로그래밍 도구와의 협업 경험을 제공하며, 사용자의 AI 행동 인식을 향상시켰습니다. 그러나 지나치게 능동적인 지원은 코드 이해도를 저하시킬 수 있다는 우려도 나왔습니다. 향후 연구를 통해 이러한 프로액티브 AI 지원의 영향과 디자인 함의를 탐구할 필요가 있음이 강조되었습니다.



### Enhancing Text Classification with a Novel Multi-Agent Collaboration Framework Leveraging BER (https://arxiv.org/abs/2502.18653)
- **What's New**: 이 논문에서는 텍스트 분류 모델의 정확성과 강건성을 향상시키기 위해 새로운 다중 에이전트 협업 프레임워크를 소개합니다. BERT를 주 분류기로 활용하여, 낮은 신뢰도를 가진 예측을 Lexical, Contextual, Logic, Consensus, Explainability 에이전트로 구성된 전문 다중 에이전트 시스템으로 전달합니다. 이 협업 방식은 종합적인 분석과 합의 기반의 의사 결정을 가능하게 하여 다양한 텍스트 분류 작업에서 성능을 크게 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 텍스트 분류 정확성과 강건성을 높이기 위해 BERT와 다중 에이전트 시스템을 통합합니다. 시스템은 초기 분류와 낮은 신뢰도 예측에 대한 다중 에이전트 협업의 두 가지 주요 단계로 작동합니다. 초기 모델이 신뢰도 점수를 평가하여 특정 임계값을 초과할 경우 분류 결과를 수용하고, 그렇지 않으면 다중 에이전트 시스템으로 전달하여 추가 분석을 수행합니다.

- **Performance Highlights**: 경험적으로 측정된 결과, 제안된 시스템은 기존의 BERT 기반 분류기와 비교하여 5.5% 높은 정확도를 기록했습니다. 이 결과는 다중 에이전트 시스템의 효과성과 자연어 처리(NLP) 분야에서의 독창성을 강조합니다. 본 연구는 텍스트 분류 작업에서 결합된 에이전트의 협업을 통해 설명 가능성과 강건성을 향상시키는 새롭고 구조화된 접근 방식을 제공합니다.



### WhatELSE: Shaping Narrative Spaces at Configurable Level of Abstraction for AI-bridged Interactive Storytelling (https://arxiv.org/abs/2502.18641)
Comments:
          In Proceedings of CHI 2025

- **What's New**: 이 논문에서는 WhatELSE라는 AI-브릿지(bridged) Interactive Narrative (IN) 저작 시스템을 소개합니다. 이 시스템은 사용자 제공 예제 이야기를 통해 내러티브 가능성 공간(Narrative Possibility Space)을 생성할 수 있도록 설계되었습니다. WhatELSE는 저작자가 내러티브 공간을 이해하고 조작할 수 있도록 돕는 세 가지 뷰를 제공합니다: 내러티브 피벗, 개요(Outline), 변형(Variants).

- **Technical Details**: WhatELSE는 저작자가 추상적인 내러티브 사양을 입력하여 내러티브 인스턴스를 생성하는 프롬프트를 작성하는 대신, 이야기 인스턴스를 피벗으로 가져오는 방식을 사용합니다. 이 시스템은 이러한 인스턴스로부터 개요를 생성하며, 개요에 기반하여 시뮬레이션 프로세스를 통해 구체적인 내러티브 변형을 만듭니다. LLM(Large Language Model)을 활용하여, 출처가 불명확한 플롯 진행을 효과적으로 지원하는 기술 파이프라인이 개발되었습니다.

- **Performance Highlights**: 사용자 연구(N=12) 및 기술 평가 결과 WhatELSE는 저작자가 내러티브 공간을 인식하고 편집하는 데 도움을 주며, 플레이 타임에 매력적인 인터랙티브 내러티브를 생성할 수 있음을 보여주었습니다. 이 시스템은 플롯 생성의 유효성을 확인하기 위해 외부 시뮬레이션 게임 환경을 활용하여, 생성된 플롯이 게임 메커니즘에 의해 정의된 인과 역학(causal dynamics)을 잘 캡처할 수 있도록 합니다.



### Quantum Machine Learning in Precision Medicine and Drug Discovery -- A Game Changer for Tailored Treatments? (https://arxiv.org/abs/2502.18639)
Comments:
          presented at AISoLA 2024

- **What's New**: 이 논문에서는 의료 디지털화의 주요 도전 과제로 생물학적 시스템의 복잡성, 방대한 데이터 생성 및 개인 맞춤형 치료 계획의 필요성을 다룹니다. 기존의 계산 방법들은 효율적으로 데이터를 처리하는 데 어려움을 겪고 있으며, 따라서 진단 및 치료에 지연이 발생하고 있습니다. 이러한 문제를 해결하기 위해 Quantum Computing (QC)과 Quantum Machine Learning (QML) 기술이 도입되어 의학의 혁신을 이끌 수 있는 가능성을 제시합니다.

- **Technical Details**: QC 및 QML은 양자 역학 원리를 활용하여 기존의 계산 방법보다 월등한 성능을 제공합니다. 이러한 방법을 통해 복잡한 생물학적 데이터에 대한 더욱 정확한 진단 및 치료 계획과 새로운 약물 발견을 가능하게 합니다. 그러나 QC를 정밀 의학에 통합하는 것은 알고리즘 오류와 높은 비용 등의 기술적 도전도 수반합니다. 이를 해결하기 위해 수학 기반의 형식적인 방법(예: formal methods)을 제안하여, 소프트웨어의 신뢰성과 정확성을 높일 수 있음을 보여줍니다.

- **Performance Highlights**: 형식적 방법을 통해 QT의 잠재력을 최대한 발휘할 수 있습니다. 우리는 유전 데이터 분석에서 질병과 관련된 유전적 마커를 식별하기 위한 양자 알고리즘의 동작 및 특성을 형식적으로 정의하고, 모든 가능한 상태를 체계적으로 탐구하여 알고리즘의 정확성을 보장할 수 있는 방법을 제시합니다. 또한 형태 최적화 기술을 통해 양자 알고리즘의 효율성을 향상시키고 자원 사용량을 줄일 수 있는 방안을 모색하고 있습니다.



### Faster, Cheaper, Better: Multi-Objective Hyperparameter Optimization for LLM and RAG Systems (https://arxiv.org/abs/2502.18635)
- **What's New**: 이 연구에서는 Retrieval Augmented Generation (RAG) 및 대형 언어 모델 (LLM) 시스템의 다중 목표 매개변수 최적화를 위한 새로운 접근법을 제시합니다. 특히 비용, 지연 시간, 안전성, 정렬을 포함하여 전체 시스템의 성능을 동시에 고려합니다. 베이지안 최적화 방법을 활용하여 이전 기준 방법보다 우수한 성과를 얻었으며, 새로운 RAG 벤치마크 작업에서 이 결과를 입증했습니다.

- **Technical Details**: RAG 시스템은 다양한 구성 요소의 많은 매개변수에 의존하는 복잡한 시스템입니다. 본 연구에서는 하이퍼볼륨 개선 (hypervolume improvement) 원칙을 적용하여 여러 하이퍼파라미터를 최적화하는 방법을 설명하며, noisy objective function에서 최적의 RAG 파이프라인 구성을 찾기 위해 qLogNEHVI를 사용합니다. 연구는 두 가지 새로운 RAG 벤치마크인 FinancialQA와 MedicalQA를 통해 검증되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법이 무작위 매개변수 선택 및 다른 기준 최적화 방법과 비교했을 때 두 가지 작업에서 더 높은 성능을 달성한 것으로 나타났습니다. 연구는 RAG 시스템을 설계하는 실무자들에게 중요한 고려 사항을 강조하며, 최적 구성은 과제와 목표에 따라 다르게 나타날 수 있음을 시사합니다. 이와 같은 통찰은 실무자들이 RAG 시스템을 개선하는 데 있어 유용할 것입니다.



### Diffusion Models for conditional MRI generation (https://arxiv.org/abs/2502.18620)
- **What's New**: 이 논문에서는 뇌 자기 공명 영상(MRI)을 생성하기 위한 Latent Diffusion Model (LDM)을 제안합니다. 이 모델은 병리학(건강, 신경교종, 경화증, 치매)과 획득 모드(T1w, T1ce, T2w, Flair, PD)를 기반으로 이미지를 생성할 수 있습니다. 생성된 이미지는 실제 이미지와 유사한 분포를 보여주며, 시각적 충실도와 다양성의 균형을 유지합니다. 또한 이 모델은 훈련 데이터에 없는 구성도 생성할 수 있는 외삽(extrapolation) 능력을 입증했습니다.

- **Technical Details**: LDM은 압축된 잠재 공간(latent space)에서 확산(diffusion)을 수행하여 이미지 생성을 최적화합니다. 이는 시각적 품질을 저하시킴 없이 계산 부담을 줄여줍니다. 이 모델은 Gauss 노이즈를 점진적으로 추가하여 각 시간 단계에서 노이즈가 있는 잠재 벡터를 생성하며, 여기서 디퓨전 과정은 두 단계로 나뉩니다: 전방 확산(forward diffusion) 및 역방향 절차(reverse process).

- **Performance Highlights**: 모델의 성능 평가는 Fréchet Inception Distance (FID) 및 Multi-Scale Structural Similarity Index (MS-SSIM) 메트릭을 사용하여 수행되었습니다. 결과는 실제 데이터에서 잘 나타나지 않는 병리 및 모드에 대한 이미지 생성에서 데이터의 다양성을 증가시키며, 진단 도구 개발에 기여할 수 있음을 나타냅니다. 전체적으로 이 연구는 임상 데이터 세트에서 샘플 수를 증가시키고, 환자의 개인 정보를 침해하지 않으면서 AI 모델의 평가를 할 수 있는 가능성을 보여줍니다.



### Mind the Gap: Bridging the Divide Between AI Aspirations and the Reality of Autonomous Characterization (https://arxiv.org/abs/2502.18604)
Comments:
          33 pages, 6 figures

- **What's New**: 이 논문은 인공지능(AI) 시대에서 재료 과학의 새로운 가능성을 탐구합니다. 특히, 전자 현미경을 통한 자율적인 특성 분석의 발전을 강조하며, 복잡한 원자 시스템을 설명할 수 있는 도메인 인식 다중 모달 모델을 개발한 내용을 소개합니다. 현재 자율 현미경의 이론적인 가능성과 실질적인 한계 사이의 간극을 해소하기 위한 최근의 성과와 필요한 발전 방향을 제시하고 있습니다.

- **Technical Details**: 연구에서는 STEM(Scanning Transmission Electron Microscopy)에서 얻을 수 있는 다중 모달 데이터의 중요성을 강조하며, 전자의 물질에 대한 상호작용을 통해 얻어지는 다양한 신호의 해석이 필요하다고 설명합니다. 이 데이터는 이미징, 분광학, 회절 신호를 포함하여 여러 방식으로 수집되며, 이 과정에서 인간 전문가의 의존도를 줄이고 실험 재현성을 높이기 위한 AI/ML 기술의 적용이 중요합니다. 또한, 새로운 하드웨어 발전이 데이터의 양을 폭발적으로 증가시켰음을 지적하며, 이에 따른 데이터 과학 접근 방식을 모색해야 한다고 주장합니다.

- **Performance Highlights**: 연구진은 자율 현미경 실험에서 실시간 머신러닝 모델과 폐쇄 루프 제어를 통한 통계적 연구의 가능성을 설명합니다. 이는 수백만 개의 원자와 결함의 행동을 특성화하고, 새로운 물질의 합성을 검증할 수 있는 기반을 마련합니다. 하지만, 자율성을 위한 여러 장애물—모델의 적합성 문제와 실험 장비 프로그래밍에서의 한계—가 여전히 존재함을 강조하며, 다중 모달 접근 방식과 시간이 촉박한 예측 제어 시스템의 발전을 통해 이를 해결하고자 한다고 말합니다.



### Autonomous Vision-Guided Resection of Central Airway Obstruction (https://arxiv.org/abs/2502.18586)
Comments:
          Submitted to World Scientific, Journal of Medical Robotics Research (JMRR) 2025. 10 pages, 11 figures

- **What's New**: 이 연구는 기도(airway) 내 종양 제거를 위한 자율 로봇 시스템을 소개합니다. 기존의 제거 방법이 드물게 자율성을 제공했던 것과 달리, 본 시스템은 비전 기반 (vision-guided) 접근 방식을 통해 정밀한 재단(재단) 작업을 실현합니다. 이 연구는 5번의 연속 실험에서 성공적으로 기도 폐색을 제거한 결과를 통해 자율 수술 플랫폼의 가능성을 입증합니다.

- **Technical Details**: 시스템은 고유한 Faster R-CNN 세그멘테이션 파이프라인을 통해 기도와 종양 경계를 식별하며, 5차 다항식(polynomial)을 사용하여 기도 표면을 모델링합니다. 최적화된 전기 소작 도구(electrocautery tool)의 각도와 함께 1mm의 안전 여유 공간을 유지하며 도구의 경로를 계획합니다. 실험은 ex-vivo 동물 조직 모델을 사용하여 실시되었으며, 기도 손상을 방지하면서 90% 이상의 종양 제거율을 달성했습니다.

- **Performance Highlights**: 본 연구의 실험 결과는 수술어의 워크플로우가 자율 시스템을 통해 성공적으로 구현될 수 있음을 보여줍니다. 연구진은 기도 폐색 제거의 로봇 시스템이 일반적인 수술 방식과 동등한 정확성을 제공할 수 있다고 주장합니다. 이 데이터는 향후 최소 침습 수술(minimally-invasive surgery)에서의 자율 로봇 어플리케이션 발전의 기반이 될 것입니다.



### Scalable Best-of-N Selection for Large Language Models via Self-Certainty (https://arxiv.org/abs/2502.18581)
- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 추론 성능을 향상시키기 위한 새로운 기법인 self-certainty를 제안합니다. 기존의 Best-of-N 선택 방법은 외부 보상 모델을 사용하여 응답의 품질을 평가하지만, 이러한 방법은 계산 비용이 크고 여러 한계를 가지고 있습니다. Self-certainty는 LLM의 출력에서 자연적으로 발생하는 확률 분포를 활용하여 응답의 품질을 추정할 수 있으며, 외부적인 보상 모델 없이도 정확도를 높이는 데에 기여할 수 있습니다.

- **Technical Details**: Self-certainty는 LLM이 생성한 토큰 분포를 기반으로 하여 응답의 신뢰도를 평가하는 새로운 메트릭입니다. 이 메트릭은 토큰 분포의 균일 분포로부터의 발산을 측정하며, 발산이 클수록 더 확신에 찬 예측을 의미합니다. 연구진은 self-certainty가 여러 샘플이 있을 때 더 높은 응답 정확도와 상관관계가 있다고 가정하며, 다양한 추론 작업에서 효과적으로 확장할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, self-certainty 기반의 투표가 Best-of-N 선택에서 self-consistency보다 항상 우수한 성능을 보였습니다. self-certainty는 샘플 크기가 증가함에 따라 효과적으로 확장 가능하며, 전통적인 self-consistency 방법이 각기 다른 경로를 처리할 수 없는 한계를 극복할 수 있습니다. 특히, self-certainty는 오픈 엔디드 작업에서도 뛰어난 성능을 발휘하며, 더 나아가 chain-of-thought와 동시에 결합하여 효과적인 추론 성과를 보여줍니다.



### Differentially Private Iterative Screening Rules for Linear Regression (https://arxiv.org/abs/2502.18578)
Comments:
          Proceedings of the 15th ACM Conference on Data and Application Security and Privacy

- **What's New**: 이번 논문에서는 처음으로 선형 회귀를 위한 차별적으로 개인 정보 보호(differentially private) 스크리닝 규칙(screening rule)을 개발하였습니다. 기존의 스크리닝 규칙이 특정 특성(feature)을 지나치게 필터링하는 문제를 발견하였고, 이를 해결하기 위한 약화된 구현 방법을 제시하여 성능을 향상시켰습니다. 이 연구는 데이터 분석에서 비공식적이거나 민감한 정보를 보호하는 모델의 요구가 증가하는 가운데 이루어졌습니다.

- **Technical Details**: Sparse linear regression(희소 선형 회귀)은 고차원 데이터셋에서 과적합(overfitting)을 방지하는 중요한 통계 기법입니다. 기본적으로 LASSO 최적화 방식에 의해 수행되며, 이 과정에서 수학적 조건을 효율적으로 점검하여 모델의 중요 특성을 선택하는 스크리닝 규칙을 사용하는 것이 특징입니다. 또한, 차별적 개인 정보 보호를 구현하기 위해 알고리즘의 중간 단계나 출력에 노이즈(noise)를 추가하고, 이를 통해 모델의 민감도를 측정합니다.

- **Performance Highlights**: 제안된 차별적으로 개인 정보 보호된 스크리닝 규칙은 다양한 데이터셋에서 희소하고 효과적인 솔루션(sparse and effective solutions)을 생성하는 데 기여하고 있습니다. 논문에 게재된 실험 결과는 이 스크리닝 규칙이 기존의 방법들보다 성능이 향상되었음을 보여줍니다. 이러한 작업은 향후 데이터 분석에서 안전하고 효과적인 개인 정보 보호 모델 개발에 기여할 것으로 기대됩니다.



### FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models (https://arxiv.org/abs/2502.18573)
- **What's New**: 본 논문은 FactReasoner라는 새로운 사실성 평가기를 제안합니다. 이 모델은 긴 형식의 생성 응답을 평가하기 위해 확률적 추론(probablistic reasoning)을 활용합니다. FactReasoner는 응답을 원자 단위로 분해하고, 외부 지식 소스에서 관련된 문맥(context)을 검색하여, 해당 원자가 검색된 문맥에 의해 뒷받침되는지의 확률을 계산합니다.

- **Technical Details**: FactReasoner는 그래픽 모델(graphical model)을 이용하여 원자와 검색된 문맥 간의 연결을 나타냅니다. 이 모델은 자연어 발화 간의 관계를 나타내는 확률적 인코딩을 사용하며, 원자와 문맥 간의 연역 관계(entailment)와 반대 관계(contradiction)에 기반하여 확률 분포를 구성합니다. 기존의 채점 방식과 달리 FactReasoner는 정보들이 서로 충돌하지 않는다는 가정을 하지 않고, 더 넓은 범위의 사실성을 평가합니다.

- **Performance Highlights**: 실험 결과, FactReasoner는 기존의 최신 프롬프트 기반 방식들보다 사실적 정밀도(factual precision)와 재현율(recall) 면에서 현저하게 향상된 성능을 보여주었습니다. 이 모델은 원자와 모든 검색된 문맥, 그리고 문맥 간의 논리적 관계를 활용하여 더 많은 지원되는 원자를 올바르게 식별할 수 있어, 사실성 평가에서 안정성이 높습니다.



### Application of Attention Mechanism with Bidirectional Long Short-Term Memory (BiLSTM) and CNN for Human Conflict Detection using Computer Vision (https://arxiv.org/abs/2502.18555)
- **What's New**: 이 연구는 비디오에서의 폭력 행동 탐지를 향상시키기 위한 심층 학습 기법의 통합을 조사합니다. 특히 Attention Mechanism, Convolutional Neural Networks (CNNs) 및 Bidirectional Long Short-Term Memory (BiLSTM)를 활용하여 인간 충돌 자동 탐지의 정확도를 높이고자 합니다. 실험 결과, CNN과 BiLSTM, Attention Mechanism의 조합이 충돌 모니터링을 위한 유망한 솔루션을 제공한다고 보고하고 있습니다.

- **Technical Details**: 컴퓨터 비전은 현실 세계에서 시각 정보를 획득하고 처리하며 해석하는 방법을 개발하는 인공지능(AI)의 한 분야입니다. 이 연구에서는 CNNs가 이미지와 비디오로부터 계층적 공간 특성을 추출하는 데 얼마나 효과적인지를 확인합니다. 또한, LSTM과 BiLSTM 아키텍처를 사용하여 시퀀스 데이터에서 시간적 종속성을 모델링 하고, Attention Mechanism을 통해 입력의 중요한 부분에 더욱 집중할 수 있도록 하는 기법을 설명합니다.

- **Performance Highlights**: 이 연구의 실험들은 깊은 훈련 데이터 세트를 사용하여 모델의 평균 정확도를 6% 향상시킨 것을 보여줍니다. Data augmentation과 transfer learning이 더 효과적인 감시 시스템을 위한 기초로 작용한다는 점을 강조합니다. 또한, 기존 영상 해석 기법들과의 비교를 통해 새로운 접근 방식의 효과를 입증하고 있습니다.



### Applications of Statistical Field Theory in Deep Learning (https://arxiv.org/abs/2502.18553)
- **What's New**: 이 논문에서는 기계 학습, 특히 딥 러닝의 이론적 기반을 물리학의 패러다임 내에서 탐구하는 새로운 연구 방향을 제시합니다. 최근 몇 년간 통계장 이론(statistical field theory)이 딥 러닝의 일반화, 암묵적 편향(implicit bias), 그리고 특성 학습(feature learning) 효과에 대한 통찰력을 제공하기 위해 활용되고 있음을 강조합니다. 이러한 접근방법은 딥 러닝의 복잡성을 이해하는 데 중요한 통찰을 제공할 수 있는 가능성을 지니고 있습니다.

- **Technical Details**: 선행 연구들에서 제안된 기술들은 다양한 수학적 및 물리학적 도구를 포함하며, 이에서는 Replicas, Path-Integrals, 그리고 Gaussian Processes와 같은 기본 개념에 대한 소개가 있습니다. 이는 이후 장에서 훈련된 신경망 분석에 필요한 최소한의 사용자 인터페이스를 제공하기 위함입니다. 통계물리학과 장 이론(field theory)은 복잡한 시스템에서 나타나는 문제를 해결하는 데 매우 유용한 프레임워크로 작용할 수 있습니다.

- **Performance Highlights**: 최근 성과들 중 일부는 고차원 과매개변수화된 딥 뉴럴 네트워크(DNNs)의 선형 모델 설명, 하이퍼파라미터 전이(Hyper-parameter transfer)에 대한 분석적 접근법, 그리고 ChatGPT와 같은 실제 네트워크의 성능을 예측할 수 있는 스케일링 법칙을 포함합니다. 이러한 예측은 장 이론의 프레임워크 내에서 도출될 수 있으며, 이는 계산 자원의 할당을 최적화하는 데 도움이 됩니다. 이 논문은 이러한 최신 결과들이 어떻게 딥 러닝 이론 발전에 기여할 수 있는지를 상세히 설명합니다.



### What is the Alignment Objective of GRPO? (https://arxiv.org/abs/2502.18548)
- **What's New**: 본 논문에서는 Group Policy Optimisation (GRPO) 알고리즘에 의해 달성된 선호의 집합(aggregation)을 분석하고 있습니다. GRPO는 딥러닝 모델인 DeepSeek-R1-Zero 및 DeepSeekMath와 같은 고급 인공지능 모델을 훈련시키기 위해 사용되는 강화학습 방법입니다. 특히, 이 알고리즘은 주어진 맥락에 대한 출력 집합을 샘플링하여 보상(preference model)을 관찰하고, 이 보상 값에 대해 시프트 및 스케일 정규화를 적용하는 방식으로 정책을 훈련합니다.

- **Technical Details**: GRPO 알고리즘은 주어진 맥락 q에 대해 폴리시 π의 샘플로부터 출력 그룹 o1,…,oG를 샘플링합니다. 이를 통해 보상 r1,…,rG를 관찰하고, 이 정보를 바탕으로 새로운 정책 πθ를 선택하기 위한 목표 함수를 정의합니다. GRPO는 기존의 Proximal Policy Optimisation (PPO) 알고리즘을 확장하여, 출력의 장점을 계산하는 새로운 방법과 KL 발산(Kullback-Leibler divergence) 추정기를 기반으로 한 정규화 기법을 도입합니다.

- **Performance Highlights**: 본 연구에서 선호 집합의 집합 방식은 기존의 로그 풀링(logarithmic pooling)과 본질적으로 다름을 보여줍니다. 실험적으로, 2인 그룹의 경우 보상 기대치 모델이 다른 정렬 방법들과 유사하게 쌍 비교(preference) 방식으로 작동한다는 점을 입증합니다. 또한, 질문 답변의 정규화 상수와 신뢰 한계와 같은 매개변수에 대한 집합 선호의 의존성을 통찰할 수 있는 명시적인 성격화를 제공하고 있습니다.



### Steganography Beyond Space-Time With Chain of Multimodal AI Agents (https://arxiv.org/abs/2502.18547)
- **What's New**: 본 연구에서는 현대 스테가노그래피의 새로운 패러다임을 제안합니다. 이는 시공간적 도메인을 넘어 메시지를 숨기는 방법으로, 다중 모달 AI 에이전트를 사용하여 오디오 및 비디오 콘텐츠를 분해하고 메시지를 언어적 도메인에 삽입한 후, 결과 콘텐츠를 재구성합니다. 이러한 접근법은 사이버 범죄자의 위협에 대응하기 위해 고안된 것으로, 인공지능의 발전을 활용하여 보다 안전한 스테가노그래픽 시스템을 구축하는 데 중점을 두고 있습니다.

- **Technical Details**: 스테가노그래픽 시스템은 여러 정의 속성을 갖고 있으며, 본 연구에서는 용량(capacity), 충실도(fidelity), 비밀성(secrecy), 강건성(robustness)의 주요 속성을 논의합니다. 용량은 주어진 매체에 삽입할 수 있는 비트 수를 의미하며, 이것은 일반적으로 0비트 시스템과 다중 비트 시스템으로 나뉩니다. 충실도는 커버(stego)와 스테고 버전 간의 유사성을 나타내며, 비밀성은 스테고 매체를 감춰야 하는 정도를 의미합니다.

- **Performance Highlights**: 본 연구에서는 제안한 다중 모달 스테가노그래피 시스템의 성능을 0비트 및 다중 비트 용량 설정 하에 평가합니다. 메세지 전송 정확도는 단어 선택의 확률 분포를 분석하여 측정하며, 신뢰성은 생체(biometric) 및 의미적(semantic) 유사성을 통해 평가합니다. 또한, 오디오 비디오 콘텐츠 압축, 얼굴 스와핑, 음성 클로닝 등의 다양한 시나리오에서 강건성을 테스트하여 시스템의 견고성을 확인하였습니다.



### PII-Bench: Evaluating Query-Aware Privacy Protection Systems (https://arxiv.org/abs/2502.18545)
- **What's New**: 대규모 언어 모델(LLMs)의 광범위한 채택은 사용자 프롬프트에 포함될 수 있는 개인 식별 정보(PII) 노출에 대한 심각한 프라이버시 우려를 불러일으켰습니다. 이를 해결하기 위해 제안된 쿼리 관련 없는 PII 마스킹 전략과 PII-Bench라는 포괄적인 평가 프레임워크는 사용자 프라이버시 보호 시스템의 유용성을 측정하는 데 중요한 기초 자료가 될 것입니다. 이는 다양한 PII 카테고리를 포함한 2842개의 테스트 샘플로 구성됩니다.

- **Technical Details**: PII-Bench는 사용자 쿼리, 맥락 설명, 쿼리와 관련된 PII를 표시하는 표준 응답을 포함하여 세 가지 주요 구성 요소로 세밀하게 설계된 샘플로 이루어져 있습니다. 이 프레임워크는 쿼리 관련성을 결정하는 데 있어 모델들이 직면하는 주목할 만한 한계를 드러내며, 특히 다중 주제 시나리오에서의 PII 식별 문제에 대한 연구를 심층적으로 다룹니다. 쿼리 관련 없는 PII 마스킹 전략을 통해 PII의 중요성과 사용자 쿼리 간의 관계를 고려하여 보다 세밀한 보호 조치를 가능하게 합니다.

- **Performance Highlights**: 실험 분석 결과, 현재 모델들은 기본적인 PII 탐지에서는 양호한 성능을 보이나, 쿼리 관련성 결정에서는 뚜렷한 한계를 드러냈습니다. 최신 LLM조차 이러한 작업에서 어려움을 겪고 있으며, 이는 지능적인 PII 마스킹이 필요함을 강조합니다. 본 연구는 프라이버시 보호 시스템의 효과성을 평가하기 위한 새로운 기준을 제시하며, 향후 연구 방향에 대한 인사이트를 제공합니다.



### MA-GTS: A Multi-Agent Framework for Solving Complex Graph Problems in Real-World Applications (https://arxiv.org/abs/2502.18540)
- **What's New**: 이 논문에서는 복잡한 그래프 이론 문제를 해결하기 위해 MA-GTS(Multi-Agent Graph Theory Solver)라는 다중 에이전트 프레임워크를 제안합니다. MA-GTS는 텍스트 기반 그래프 데이터를 명확하고 구조화된 그래프 표현으로 변환하고, 문제 제약과 그래프 구조를 기반으로 가장 적합한 알고리즘을 동적으로 선택합니다. 이를 통해 효율적인 솔루션 과정과 해석 가능한 추론 경로를 보장합니다.

- **Technical Details**: MA-GTS 프레임워크는 정보 추출 계층(Information Extraction Layer), 지식 통합 계층(Knowledge Integration Layer), 알고리즘 실행 계층(Algorithm Execution Layer)의 세 가지 계층으로 구성됩니다. 각 계층은 협력적 커뮤니케이션 메커니즘을 통해 상호작용하며, 비구조적 데이터를 처리하고 복잡한 그래프 이론 문제를 해결할 수 있는 파이프라인을 형성합니다. 이 방식은 복잡한 제약사항에 적응하고 대규모 문제를 해결하는 데 필요한 동적 에이전트 상호작용을 가능하게 합니다.

- **Performance Highlights**: MA-GTS는 G-REAL 데이터 세트를 통해 검증되었으며, 실험 결과 기존의 최첨단 접근 방식들보다 효율성, 정확성 및 확장성 측면에서 우수한 성능을 보여줍니다. 특히, MA-GTS는 대규모 문제 해결에 있어 강력한 결과를 보였으며, G-REAL 94.2%, GraCoRe 96.9%, NLGraph 98.4%와 같은 벤치마크에서 높은 성능을 기록했습니다.



### Revisiting Convolution Architecture in the Realm of DNA Foundation Models (https://arxiv.org/abs/2502.18538)
- **What's New**: 최근 DNA 언어 모델 개발에서 Transformer 및 상태 공간 모델(SSM) 기반 방법들이 제안되었지만, 이들이 전통적인 CNN(Convolutional Neural Network) 아키텍처와 비교된 연구는 부족했습니다. 본 논문에서는 ConvNova라는 CNN 기반 방법을 제안하며, dilated convolution, gated convolution 및 이중 분기 프레임워크를 포함한 세 가지 효과적인 설계를 개발했습니다.

- **Technical Details**: ConvNova는 dilated convolution(확장 합성곱), gated convolution(게이트 합성곱), 그리고 이중 프레임워크로 구성된 게이트 메커니즘을 통해 성능 증대에 기여합니다. 특히, CNN의 수용 영역을 넓히기 위한 Downsampling이 성능 저하로 이어질 수 있으나, dilated convolution은 이러한 문제를 해결할 수 있습니다. 실험 결과 ConvNova는 여러 기준 모델 벤치마크에서 Transformer 및 SSM 기반 방법들에 비해 더 나은 성능을 보여주었습니다.

- **Performance Highlights**: ConvNova는 18개 데이터셋 중 12개에서 최신의 SoTA(State of the Art) 성능을 달성하였으며, H3K4me3 작업에서는 두 번째 우수한 방법에 비해 10.5% 높은 성능을 보였습니다. 또한, 일반 상위 방법에 비해 파라미터 수가 적고 계산 속도도 빠른 것으로 나타났습니다. 이 연구는 CNN이 Transformer 및 SSM에 비해 여전히 강력한 경쟁력임을 시사합니다.



### A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning (https://arxiv.org/abs/2502.18535)
Comments:
          24 pages, 5 figures, 3 tables

- **What's New**: 이번 논문에서는 머신러닝(ML) 모델의 훈련 및 추론 과정에서 데이터 프라이버시와 모델 보안을 보장하기 위한 제로 지식 증명(zero-knowledge proof, ZKP) 기술의 발전을 다룹니다. ZKP는 민감한 데이터를 공개하지 않으면서 모델의 성능과 진위를 검증할 수 있는 강력한 솔루션으로, 기존의 머신러닝 솔루션에서 존재하는 데이터 보안 문제를 해결하는 방법으로 주목받고 있습니다. 이 논문은 2017년 6월부터 2024년 12월까지의 모든 ZK Machine Learning(ZKML) 연구를 종합적으로 리뷰하고 분석하는 데 초점을 맞추고 있습니다.

- **Technical Details**: ZKML은 세 가지 주요 범주인 검증 가능한 훈련(Verifiable Training), 검증 가능한 추론(Verifiable Inference), 검증 가능한 테스트(Verifiable Testing) 아래에서 ZKP 알고리즘 세트를 설명합니다. 논문에서는 각 범주에 존재하는 ZKML 연구를 체계적으로 분류하고, 이를 바탕으로 자세한 분석을 제공합니다. ZKP를 활용한 훈련 과제에서는 클라이언트가 제공한 훈련 데이터의 정확성을 증명해야 하며, 서비스 제공자는 클라이언트에게 모델 파라미터의 진위를 증명해야 합니다.

- **Performance Highlights**: ZKML 기술의 상업적 응용 가능성을 강조하며, ML 서비스 제공자와 클라이언트 간의 신뢰 문제와 데이터 프라이버시 문제를 해결하기 위한 혁신적인 방향을 제시합니다. 최근 ZKML 분야에 대한 관심 증가와 ZKP 기술의 발전 덕분에 다양한 연구가 진행되고 있으며, 새로운 제안도 계속 등장하고 있습니다. 이 논문은 ZKML의 발전 궤적과 현황을 체계적으로 분석하여 향후 연구의 방향성을 제시하는 데 기여할 것입니다.



### MAFE: Multi-Agent Fair Environments for Decision-Making Systems (https://arxiv.org/abs/2502.18534)
- **What's New**: 본 논문은 Multi-Agent Fair Environment (MAFE)라는 개념을 도입하여, 여러 상호작용하는 주체들이 결과를 미치는 단동 맥락에서의 공정성을 분석하는 새로운 프레임워크를 제안합니다. 이 연구는 머신 러닝 모델의 공정성 문제를 해결하기 위한 지속 가능한 솔루션을 추구하며, 여러 개체의 상호작용을 명시적으로 모델링하는 접근 방식을 강화합니다. MAFE는 자원의 할당, 의료, 고등 교육 등의 다양한 사회적 시스템을 모델링하여 실험적 결과를 통해 알고리즘 개발의 테스트베드로 활용할 수 있음을 보여줍니다.

- **Technical Details**: MAFE는 표준 분산 부분 관찰 마르코프 결정 프로세스(Dec-POMDP)의 개념을 확장하여 정의됩니다. 여기에서, 𝒩는 에이전트 집합을 나타내고, 𝒮는 관찰되지 않는 전체 시스템 상태 집합을 의미합니다. 각 에이전트 n의 행동(action) 및 관찰(observation) 공간은 𝒜와 𝒪로 나타나며, 이를 통해 다양한 상호작용을 시뮬레이션할 수 있습니다.

- **Performance Highlights**: 실험 결과는 MAFE가 다양한 사회적 맥락에서 공정한 알고리즘을 개발하고 평가하는 데 효과적인 도구임을 입증합니다. 제안된 MAFE의 프레임워크는 에이전트 간 협력 사용 사례를 통해 적응성을 잘 보여주며, 각기 다른 공정성 기준을 위한 성공 지표를 정의하여 공정성을 다각적으로 평가하는 데 기여합니다. 이 연구는 다중 에이전트 시스템에서의 공정성에 대한 이해를 높이고, 실질적인 사회적 영향을 분석할 수 있는 보다 넓은 틀을 제공합니다.



### Heterogeneous Decision Making in Mixed Traffic: Uncertainty-aware Planning and Bounded Rationality (https://arxiv.org/abs/2502.18529)
Comments:
          CPAL 2025

- **What's New**: 이번 연구는 자동화 차량(AV)과 인간 주행 차량(HV) 간의 상호 작용을 살펴보고, 혼합 트래픽 환경에서 안전하고 효율적으로 차량이 운영될 수 있도록 하는 AI 기반을 개발하는 것을 목표로 합니다. 혼합 자율성(mixed autonomy)을 달성하기 위한 핵심 도전 과제는 HV의 의사 결정이 제한된 합리성(bound rationality)을 가지고 있다는 점과 AV가 인간 행동에 대한 안전한 대응을 위해 불확실성 인식 계획(uncertainty-aware planning)을 활용하는 것입니다.

- **Technical Details**: 우리는 AV와 HV가 두 대의 에이전트로 구성된 시스템을 고려하고, HV는 단기 계획을 통해 의사 결정을 내리며, 그 과정에서 서브 최적(sub-optimal) 및 노이즈(noisy)의 특성을 가집니다. 또한, AV는 HV의 미래 행동 예측에 기반한 불확실성 인식의 Lookahead 계획을 활용합니다. 본 연구의 주요 목표는 혼합 자율성에서의 이질적인 의사 결정의 성능을 이해하고, HV의 제한된 합리성과 AV의 계획이 학습 성과에 미치는 영향을 분석하는 것입니다.

- **Performance Highlights**: 연구 결과는 AV의 학습 성과에서 Goodhart 법칙(Goodhart's Law)의 발생과 HV의 의사 결정 과정에서의 복합 효과(compounding effects)를 포함하여 흥미로운 현상을 보여줍니다. AV와 HV 간의 학습 후회(regret)의 동역학을 조사함으로써, 인간과 기계 간의 의사 결정 상호 작용의 통찰력을 제공합니다. 그런 맥락에서, 서로 다른 의사 결정 전략이 전체 학습 성과에 미치는 영향을 분석하였습니다.



### ARACNE: An LLM-Based Autonomous Shell Pentesting Agen (https://arxiv.org/abs/2502.18528)
Comments:
          7 pages, 2 figures, 3 tables

- **What's New**: ARACNE는 SSH 서비스에 최적화된 완전 자율적 LLM 기반의 펜테스팅 에이전트를 소개합니다. 이 에이전트는 실제 Linux 셸 시스템에서 명령을 실행할 수 있는 능력을 가지고 있으며 멀티 LLM 모델을 지원하는 새로운 아키텍처를 도입했습니다. 실험 결과 ARACNE는 자율 방어 시스템 ShelLM에 대해 60%의 성공률을 기록하며 최신 기술보다 상당한 개선을 보여주었습니다.

- **Technical Details**: ARACNE는 계획자(planner), 해석기(interpreter), 요약기(summarizer), 핵심 에이전트(core agent)의 네 개 주요 모듈로 구성되어 있습니다. 플래너 모듈은 OpenAI의 최신 모델인 GPT-O3-mini를 기반으로 하여 공격 계획을 수립하고, 각 명령을 실행하는 데 필요한 정보와 경과를 JSON 형식으로 출력합니다. 또한, ARACNE는 RAG 없이도 동작할 수 있는 보다 유연한 모듈 아키텍처를 제공합니다.

- **Performance Highlights**: ARACNE는 Over The Wire Bandit의 CTF 챌린지에서 57.58%의 성공률을 달성하며 최신 기술 대비 0.48%의 성장을 보여주었습니다. 에이전트가 목표를 달성하는 데 평균적으로 소요된 행동 수는 5 이하로, 이는 자동화된 사이버 공격에서의 가능성을 시사합니다. 이러한 결과는 멀티 LLM 방법론의 효과성을 입증하는 중요한 사례로 평가됩니다.



### GOD model: Privacy Preserved AI School for Personal Assistan (https://arxiv.org/abs/2502.18527)
- **What's New**: 이 논문은 개인 AI 비서의 사용자 데이터에 대한 프라이버시 및 신뢰 문제를 해결하기 위해 'Guardian of Data (GOD)'라는 안전한 프레임워크를 소개합니다. GOD는 사용자의 요구를 예측하는 능력을 평가하는 새로운 접근 방식을 제공하며, 사용자의 데이터를 보호하는 동시에 AI의 성능을 향상시킵니다. 이 프레임워크는 사용자가 데이터를 안전하게 공유할 수 있도록 유도하는 토큰 기반의 인센티브 시스템도 포함하고 있습니다.

- **Technical Details**: GOD 모델은 Trusted Execution Environment (TEE) 내에서 작동하여 사용자 데이터를 보호하면서 AI 비서의 성능을 평가합니다. 이 모델은 각 비서의 성과를 측정하기 위해 다양한 난이도의 과제를 생성하고, 데이터를 안전하게 다루면서 강화 학습 및 모방 학습을 적용합니다. 이를 통해 개인화된 추천의 차가운 시작 문제를 완화하고, 사용자 데이터를 비공식적으로 활용하여 AI의 능력을 지속적으로 개선할 수 있습니다.

- **Performance Highlights**: GOD 모델은 사용자 개인 데이터를 활용한 추천의 질을 정량적으로 평가할 수 있는 구조화된 방법을 제안합니다. 또한, AI의 정확도, 일관성 및 적극성을 종합적으로 평가하는 점수 구조를 이용하여 AI의 성능을 체계적으로 개선할 수 있는 기반을 제공합니다. 이를 통해 사용자와의 신뢰 관계를 저해하지 않으면서도 개인화된 AI 지원을 제공할 수 있습니다.



### Reinforcement Learning-based Approach for Vehicle-to-Building Charging with Heterogeneous Agents and Long Term Rewards (https://arxiv.org/abs/2502.18526)
- **What's New**: 이 논문에서는 전기차 배터리를 에너지 저장소로 전략적으로 집합시켜 전력망의 수요를 최적화하려는 새로운 접근 방식을 소개합니다. 특히 대형 사무실 빌딩과 같은 스마트하고 연결된 커뮤니티에 유리한 환경을 조성하기 위해 충전 및 방전을 최적화하는 방법에 대해 다룹니다. 이를 통해 기존의 휴리스틱 기반 알고리즘의 한계를 극복하고, 실시간 의사결정에 필요한 새로운 전략을 제시합니다.

- **Technical Details**: 제안된 RL 프레임워크는 Deep Deterministic Policy Gradient (DDPG) 접근 방식을 기반으로 하며, 액션 마스킹(action masking) 및 효율적인 MILP 기반 정책 안내를 결합합니다. 이 방법은 불확실성과 지연된 보상 속에서 연속적인 액션 공간 탐색을 조화롭게 수행하여 사용자 충전 요구사항을 충족시킵니다. 이러한 기술적 요소는 다양한 조건에서 일반화되는 복잡성을 관리하는 데 중점을 둡니다.

- **Performance Highlights**: 실제 전기차 제조업체에서 수집된 데이터를 사용하여, 제안된 접근 방식이 여러 기존의 기준선 및 확장 가능한 휴리스틱 접근 방식을 종합적으로 초월하는 결과를 보여줍니다. 이는 모든 충전 요구사항을 충족하면서 상당한 비용 절감을 달성하는 데 성공했습니다. 결과적으로, 이 연구는 V2B 에너지 관리 문제를 해결하기 위한 최초의 확장 가능하고 일반적인 접근 방식 중 하나로 자리잡고 있습니다.



### End-to-End Deep Learning for Structural Brain Imaging: A Unified Framework (https://arxiv.org/abs/2502.18523)
- **What's New**: 새로운 연구인 UniBrain은 뇌 영상 분석의 여러 처리 단계를 단일화하여 통합적인 최적화 프로세스를 통해 작동하는 최초의 심층 학습 모델입니다. 기존의 방법들과 달리, UniBrain은 최소한의 레이블 데이터로 운영되고, 모든 작업을 동시에 최적화할 수 있는 장점이 있습니다. 이는 중간 오류를 수정하는 전문가의 개입 없이도 진행할 수 있어, 시간과 비용을 크게 절감할 수 있습니다.

- **Technical Details**: UniBrain의 구조는 뇌 추출, 등록, 분할, 파셀레이션, 네트워크 생성 및 분류의 여러 모듈을 통합하여 각각의 과제가 서로 상호작용하도록 설계되었습니다. 각 모듈은 딥러닝 기반의 네트워크를 사용하여 연결되며, 이를 통해 세부 작업들 간의 의존성을 효과적으로 활용합니다. 이 시스템은 3D U-Net 같은 알고리즘을 통해 수행되며, 수집된 데이터의 특성을 반영해 다양한 작업에서의 정확성과 효율성을 향상시킵니다.

- **Performance Highlights**: UniBrain은 ADHD 공개 데이터셋을 기반으로 한 실험에서 기존의 여러 방법들보다 우수한 성능을 보여주었습니다. 모든 6개 작업에서 state-of-the-art 기법들을 초과하는 결과를 기록하였으며, 이는 UniBrain이 심층 학습을 통한 통합적 접근 방식을 통해 높은 확장성과 신뢰성을 제공한다는 것을 나타냅니다. 또한, 이 모델은 준비된 심층 정보의 질을 향상시키는 데 중요한 역할을 합니다.



### Class-Conditional Neural Polarizer: A Lightweight and Effective Backdoor Defense by Purifying Poisoned Features (https://arxiv.org/abs/2502.18520)
- **What's New**: 이번 연구에서는 백도어 공격(backdoor attacks)에 대한 새로운 방어 방법인 NPD(Neural Polarizer based backdoor Defense)를 제안합니다. 이 방법은 학습된 백도어 모델(backdoored model) 사이에 가벼운 신경 편광기(neural polarizer)를 통합하여, 유해한 트리거 정보(trigger information)를 필터링하면서도 무해한 정보를 보존할 수 있도록 설계되었습니다. 또한, 고급 방어 메커니즘인 클래스 조건 신경 편광기 기반 방어(CNPD)를 통해 백도어 공격의 타겟 레이블(target label)을 추정하는 데 따른 한계를 극복하려고 합니다.

- **Technical Details**: NPD는 기존의 모델 파라미터를 고정시키고 단일 레이어(learnable layer)만을 최적화하는 경량의 방어 방법입니다. 이 과정에서 두 단계를 거치는 최적화 문제(bi-level optimization)의 형태로, 내부 최적화(inner optimization)에서는 적대적 예제(adversarial examples)를 통해 타겟 레이블을 동적으로 추정하고, 외부 최소화(outer minimization)에서는 이러한 예제의 학습을 제거합니다. CNPD는 백도어 모델의 예측 레이블(predicted label)과 정제할 특징(feature)을 조합하여 효과적인 방어를 달성합니다.

- **Performance Highlights**: 제안된 NPD는 확인된 성능 저하를 감안하여, 세 가지 변형인 r-CNPD, e-CNPD, a-CNPD를 통해 클래스를 고려한 특성 정제를 가능하게 합니다. 이들 변형은 각각 복제된 NP 레이어를 사용하거나, 추가적인 클래스를 갖춘 피쳐(feature)를 통합하여 효율적인 정제를 실시합니다. 실험을 통해 SOTA(state-of-the-art) 방법들과 비교하여 뛰어난 성능을 발휘함을 확인했습니다.



### FreeTumor: Large-Scale Generative Tumor Synthesis in Computed Tomography Images for Improving Tumor Recognition (https://arxiv.org/abs/2502.18519)
- **What's New**: Tumor is a prominent cause of global mortality, with around 10 million fatalities annually due to related conditions. AI 기반의 종양 인식은 보다 정밀하고 지능적인 스크리닝과 진단 가능성을 열어주고 있지만, 주석이 달린 데이터셋의 부족으로 인해 연구가 제약받고 있습니다. 이를 해결하기 위해 FreeTumor라는 혁신적인 Generative AI (GAI) 프레임워크를 도입하여 대규모 종양 합성을 가능하게 하여 데이터의 부족 문제를 완화하고자 합니다.

- **Technical Details**: FreeTumor는 제한된 주석 데이터와 대규모 비주석 데이터를 효과적으로 결합하여 종양 합성 훈련을 진행합니다. 이를 통해 고품질의 합성 종양 이미지를 생성하고, 의료 이미지의 이해도를 향상시킵니다. 특히 저자들은 GAN(Generative Adversarial Networks)을 활용하여 비주석 데이터로부터 종양 합성 훈련을 수행하며, 이에 따라 고품질 합성 종양을 자동으로 선별하는 투사기(discriminator)를 포함합니다.

- **Performance Highlights**: FreeTumor는 161,310개의 CT 볼륨 데이터를 활용한 대규모 훈련 데이터세트를 구성하여 훈련 데이터의 양, 질, 다양성을 개선하며, 기존 AI 방법보다 40배 이상의 성능 향상을 기록했습니다. 13명의 인증 방사선 전문의에 의해 독립적인 시각적 테스트에서 51.1%의 민감도와 60.8%의 정확도로 고품질 합성 종양을 확인받았습니다. 이러한 결과들은 FreeTumor가 임상 응용에서의 가능성을 보여주며, 종양 치료를 진전시키고 환자의 생존율을 향상시킬 수 있는 기회를 제시합니다.



### Swallowing the Poison Pills: Insights from Vulnerability Disparity Among LLMs (https://arxiv.org/abs/2502.18518)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 약점을 분석하고, 특정 사실 지식을 목표로 한 'poison pill' 공격을 설명합니다. 해당 공격은 모델 전반에 미치는 영향은 최소화하면서도 특정 사실의 왜곡을 유도합니다. 연구자는 이 공격이 LLM의 설계 특성을 활용하여 정확도 저하가 거의 없다며, 정규 벤치마크에서 성능 저하가 2% 미만으로 나타난다는 점을 강조합니다.

- **Technical Details**: 이 연구에서 제안하는 'poison pill' 공격 방식은 문서의 특정 사실 요소를 수정하는 것으로, 원래의 진실한 내용에 극히 적은 부정확성을 추가함으로써 이루어집니다. 각 문서는 특정 사실적 속성으로 분해되어 이를 통해 하나의 요소를 변경하는 단일 대상 변형 작용을 이용합니다. 이 접근법은 기존의 데이터 오염 방식과 차별화된 세 가지 특징인 지역성, 동질성 및 일관성을 통해 LLM의 특정 사실 연관성을 정밀하게 손상시킵니다.

- **Performance Highlights**: 연구 결과는 'poison pill' 데이터가 모델의 성능을 심각하게 저하시킬 수 있는 높은 효율성을 보여줍니다. 큰 모델들은 특정 수준에서 공격에 대한 저항성을 가지나, 다듬거나 증류된 모델들은 더 많은 취약성을 보였습니다. 공격 상황에서 긴 꼬리(long-tail) 지식의 비대칭 취약성 및 모델 압축으로 인한 공격 면적 증가가 향후 LLM의 안전성 문제를 제기함을 시사합니다.



### RewardDS: Privacy-Preserving Fine-Tuning for Large Language Models via Reward Driven Data Synthesis (https://arxiv.org/abs/2502.18517)
- **What's New**: 이 논문은 RewardDS라는 새로운 프레임워크를 제안하며, 이는 프라이버시를 보호하면서 LLM을 특정 분야의 데이터로 효과적으로 파인튜닝할 수 있도록 돕는다. 기존의 데이터 필터링 기법은 ROUGE-L 점수나 임베딩 유사성을 바탕으로 단순한 비교를 사용하였는데, 이러한 접근법은 노이즈를 해결하는데 미흡하다. RewardDS는 보상 프록시 모델을 훈련시켜 합성 데이터 생성을 안내하는 방식으로, 데이터 품질을 높일 수 있는 혁신적인 접근 방법이다.

- **Technical Details**: RewardDS 프레임워크는 보상 안내 필터링(Reward Guided Filtering)과 자기 최적화 정제(Self-Optimizing Refinement)라는 두 가지 핵심 모듈을 포함한다. 보상 안내 필터링 모듈은 합성 데이터의 품질을 평가하여 저품질 샘플을 삭제하며, 자기 최적화 정제 모듈은 각 합성 질의에 대한 후보 응답을 생성하고 그에 따른 보상을 계산함으로써 보다 높은 품질의 데이터를 확보한다. 이러한 과정들은 DP-SGD를 통해 프라이버시를 보호하며, 모델의 성능을 향상시키는 데 기여한다.

- **Performance Highlights**: 의료 질의 응답(Medical QA), 법률 질의 응답(Legal QA), 코드 생성(Code Generation) 작업에서 풍부한 실험을 통해 RewardDS의 효과성을 검증하였다. 결과적으로, RewardDS는 합성 데이터의 품질을 향상시키고, 더 나은 성능을 달성하며 프라이버시를 보존하는 데 성공하였다. 이 연구는 프라이버시 보호와 데이터 유용성의 균형을 이루는 새로운 해결책을 제시하며, 도메인 특정 작업을 위한 LLM의 파인튜닝에 새로운 가능성을 제공한다.



### A Multi-Agent Framework for Automated Vulnerability Detection and Repair in Solidity and Move Smart Contracts (https://arxiv.org/abs/2502.18515)
- **What's New**: 이번 논문은 Smartify라는 새로운 다중 에이전트 프레임워크를 소개하여, Solidity와 Move 스마트 계약에서의 취약점을 자동으로 탐지하고 수정합니다. 기존의 방법들과 달리, Smartify는 다양한 LLM(대형 언어 모델) 전문 에이전트를 활용하여 프로그래밍 개념과 언어별 보안 원칙을 근거로 코드를 분석합니다. 이는 대규모 사전 학습 데이터셋에 의존하지 않으면서도 언어 특유의 지식을 통합할 수 있는 방법을 제시합니다.

- **Technical Details**: Smartify는 여러 LLM을 사용하는 멀티 에이전트 시스템을 통해 안전성이 높은 스마트 계약 수리 및 탐지 기능을 제공합니다. 우리의 접근법은 코드 설명 및 수리 모델의 강점을 결합하여, 코드 작성 시점에서부터 알려진 불량 관행 및 불안전한 코드를 방지하는 데 중점을 둡니다. 이 프레임워크의 핵심은 LLM이 안전하지 않거나 버그가 있는 코드를 이해하여 이를 수정하는 데 필요한 프로그래밍 언어 특정 지식을 인코딩하는 것입니다.

- **Performance Highlights**: Smartify는 Solidity와 Move의 다양한 취약성에 대한 포괄적인 평가를 통해 기존 LLM을 초월하는 성능을 보였습니다. 분석 결과, Smartify는 코드 수리 분야에서 최고의 성능을 달성하며, 일반 용도의 모델인 Llama 3.1의 능력을 향상시킵니다. 이러한 결과는 Smartify가 향후 블록체인 생태계에서 더욱 안전하고 신뢰할 수 있는 분산 애플리케이션 개발의 청사진을 제공한다는 것을 보여줍니다.



### FCoT-VL:Advancing Text-oriented Large Vision-Language Models with Efficient Visual Token Compression (https://arxiv.org/abs/2502.18512)
Comments:
          20 pages, 18 figures, 6 tables

- **What's New**: 이 논문에서는 고해상도 텍스트 지향 Large Vision-Language Models (VLLMs)의 시각적 토큰을 효율적으로 압축하는 새로운 프레임워크를 제안합니다. 기존의 훈련 없는 방법들이 높은 해상도에서 성능 저하를 겪는 문제를 해결하기 위해, 라이트웨이트(self-distillation) 사전 학습 단계와 고품질의 후 훈련 단계를 활용합니다. 이러한 접근 방식은 적은 수의 이미지-텍스트 쌍으로도 뛰어난 성능을 발휘할 수 있게 해줍니다.

- **Technical Details**: FCoT-VL 모델은 교사 모델(teacher model)과 학생 모델(student model)로 구성되어 있습니다. 교사 모델은 풍부한 시각적 토큰을 갖고, 학생 모델은 압축된 토큰 표현을 가집니다. 사전 학습 과정에서 교사 모델의 파라미터를 상속받고, 학생 모델의 토큰 압축 모듈만 조정함으로써 훈련 데이터와 GPU 자원의 제약을 극복할 수 있습니다.

- **Performance Highlights**: 실험 결과, FCoT-VL은 텍스트 지향 벤치마크에서 기존 모델에 비해 뛰어난 성능을 보이면서도 계산 오버헤드를 크게 줄였습니다. 본 논문에서 제안된 방법론은 InternVL2 모델에서 유효하며, 다양한 작업에서 우수한 적응 능력을 유지하는 것으로 나타났습니다. 또한, 모델과 코드는 곧 공개될 예정입니다.



### ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models (https://arxiv.org/abs/2502.18511)
- **What's New**: 최신 논문에서는 생성형 대형 언어 모델(Generative Large Language Models, LLMs)이 베크도어 공격(Backdoor Attacks)에 취약하다는 점을 강조하고, 새로운 평가 기준을 위해 ELBA-Bench라는 포괄적이고 통합된 프레임워크를 제안합니다. 이 프레임워크는 파라미터 효율적인 파인 튜닝 기법(Efficient Fine-tuning)과 파인 튜닝 없이도 베크도어를 주입할 수 있는 방법을 포함하여, 1300건 이상의 실험을 제공하고 12가지 공격 방법, 18개의 데이터셋, 12개의 LLMs를 포함합니다.

- **Technical Details**: ELBA-Bench는 PEFT 공격과 파인 튜닝 없이도 공격을 수행할 수 있는 다양한 접근 방식을 연구합니다. PEFT는 모델 파라미터를 수정하여 베크도어를 삽입하며, 입력 데이터에 트리거 또는 적대적 예제를 추가하여 활성화하는 방식으로 공격을 수행합니다. 이 모델은 다양한 평가 기준을 통해 공격의 성공률(Attack Success Rate, ASR)과 스텔스 공격(Stealthiness)을 평가하여 모델의 안전성을 탐구합니다.

- **Performance Highlights**: 기존 연구들은 공격 성공률에만 집중하는 경향이 있었던 반면, ELBA-Bench는 공격 방법의 다양성과 평가지표의 객체성을 강화하며, 공격의 성공과 몰래 감지 가능성을 균형 있게 평가합니다. PEFT 공격 방법은 일반적인 파인 튜닝 없이 수행하는 방법보다 분류 작업에서 일관되게 우수한 성능을 보이며, 최적화된 트리거를 통해 강력한 데이터셋 간 일반화 결과를 제공합니다. 이러한 연구는 LLMs의 베크도어 공격의 발전에 기여하며, 모델 성능을 유지하면서도 공격 효율성을 높일 수 있는 방법을 제시합니다.



### Protecting Users From Themselves: Safeguarding Contextual Privacy in Interactions with Conversational Agents (https://arxiv.org/abs/2502.18509)
Comments:
          22 pages, 2 figures

- **What's New**: 이번 연구에서는 사용자가 LLM(대형 언어 모델)와 상호작용할 때 발생할 수 있는 개인 정보 노출의 위험을 최소화하기 위해 문맥적 프라이버시(contextual privacy) 개념을 정의하고 이를 기반으로 하는 프레임워크를 제안합니다. 이는 사용자가 자신이 공유하는 정보의 중요성과 필요성을 인식하도록 돕고, 불필요한 정보 노출을 줄이는 데 기여합니다. 연구 결과, "개인정보에 민감한" 사용자조차도 우연히 민감한 정보를 전달하는 경향이 있음을 발견했습니다.

- **Technical Details**: 연구에서는 사용자가 LCA(대형 언어 모델 기반 대화형 에이전트)와 상호작용할 때, 필요한 정보만 공유하도록 유도하는 프레임워크를 설계했습니다. 이 프레임워크는 사용자의 입력을 분석하고, 문맥적으로 부적절한 민감 정보를 감지하며 사용자가 재구성할 수 있는 방법을 안내합니다. 이를 통해 사용자는 더 안전하고 개인 정보를 인식한 상태에서 결정을 내릴 수 있습니다.

- **Performance Highlights**: 프레임워크는 사용자와 LCA 간의 상호작용에서 문맥적 프라이버시를 보호하는 데 성공적이며, 최소한의 LLM을 사용하여 효과적으로 운영될 수 있음을 보여주었습니다. 실험을 통해 이 프레임워크가 개인 정보 보호와 유용성 측면에서 모두 우수한 성과를 거두었음을 확인했습니다. 예를 들어, 사용자는 필요하지 않은 민감한 정보를 노출하지 않고도, 원하는 결과를 얻을 수 있는 방법이 있음을 밝혀냈습니다.



### REFINE: Inversion-Free Backdoor Defense via Model Reprogramming (https://arxiv.org/abs/2502.18508)
Comments:
          This paper is accept by ICLR 2025. The first two authors contributed equally to this work. Our code is available at BackdoorBox (this https URL) and Github repository (this https URL). 28 pages

- **What's New**: 본 논문에서는 REFINE이라는 새로운 백도어 공격 방어 방법을 제안합니다. 이 방법은 모델 리프로그래밍(model reprogramming) 기법을 기반으로 하며, 입력 변환(input transformation)과 출력 재매핑(output remapping) 모듈을 포함하여 백도어 특성을 효과적으로 방어합니다. REFINE는 기존 방어 방식의 한계를 보완하며, 특히 백도어 트리거 역전(backdoor trigger inversion) 없이도 작동할 수 있습니다.

- **Technical Details**: REFINE는 입력 변환 모듈이 무해한 샘플과 백도어 패턴을 모두 방해하여 새로운 무해한 특징을 생성하는 구조를 갖습니다. 아울러, 출력 재매핑 모듈은 모델의 출력 도메인을 재정의하여 입력 변환이 효과적으로 이루어질 수 있도록 지원합니다. 또한 지도 대조 손실(supervised contrastive loss)을 통합하여 방어 능력을 향상시키면서도 모델 유틸리티를 유지합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트를 기반으로 한 실험 결과, REFINE의 효과성과 적응형 공격에 대한 저항력을 보여줍니다. REFINE는 기존의 방어 방법이 겪었던 성능 한계를 극복하며, 성능과 방어 효과 간의 균형을 유지할 수 있습니다. 이 연구는 AI 시스템을 위한 필수적인 백도어 방어 전략의 필요성을 강조합니다.



### Exploring Patient Data Requirements in Training Effective AI Models for MRI-based Breast Cancer Classification (https://arxiv.org/abs/2502.18506)
Comments:
          Accepted for publication in MICCAI 2024 Deep Breast Workshop on AI and Imaging for Diagnostic and Treatment Challenges in Breast Care

- **What's New**: 최근 10년 동안 의료 기관의 임상 의사 결정을 지원하는 AI 기반 솔루션을 제공하는 스타트업과 기업들이 급격히 증가하고 있습니다. 그러나 의료 결정의 중요성 때문에 외부 소프트웨어에 대한 의존에 관한 여러 가지 우려 사항이 제기되고 있습니다. 본 연구는 유방암 탐지를 중심으로 의료 기관이 효과적인 AI 모델을 훈련하는 데 필요한 데이터의 양을 탐색합니다.

- **Technical Details**: 본 연구에서는 Vision Transformer (ViT-B/16) 아키텍처를 사용하여 데이터 부족이 모델 훈련에 미치는 영향을 조사합니다. DINO와 MAE라는 두 가지 자가 지도 학습 프레임워크를 이용하여 사전 훈련된 모델을 사용하였고, Duke Breast Cancer Dataset을 기반으로 연구를 수행하였습니다. 이 데이터셋은 14년에 걸쳐 수집된 침습성 유방암 환자 922명의 3D MRI 이미지를 포함하고 있으며, 연구의 주요 작업은 MRI 이미지에서 유방 종양의 존재 여부를 판단하는 것입니다.

- **Performance Highlights**: 본 연구는 제한된 수의 이미지로도 사전 훈련된 모델이 최첨단 성능을 달성할 수 있음을 보여줍니다. 50명 이상의 환자로 구성된 훈련 세트에서는 환자 수의 변화가 모델 성능에 미치는 영향이 미미하며, 간단한 앙상블 방법이 추가 복잡성 없이도 성능을 향상시킬 수 있음을 관찰하였습니다. 이러한 결과는 데이터 부족 상황에서도 효과적인 AI 솔루션을 개발할 수 있는 가능성을 제시합니다.



### Comprehensive Analysis of Transparency and Accessibility of ChatGPT, DeepSeek, And other SoTA Large Language Models (https://arxiv.org/abs/2502.18505)
- **What's New**: 이 연구는 오픈소스 인공지능(AI) 모델의 투명성과 접근성 문제에 초점을 맞추고 있습니다. 최신 대형 언어 모델(LLM)의 일부는 '오픈소스'라고 주장하지만, 실제로 완전한 투명성이 부족하다는 점을 밝혀냅니다. 이 연구는 약 100개의 LLM을 분석하여 오픈소스와 오픈웨이트 모델 간의 차이를 구체적으로 검토하고, AI의 윤리적 배치를 위한 더 나은 기준과 가이드라인을 제안합니다.

- **Technical Details**: 연구 방법론은 다단계 접근방식을 사용하여 LLM의 개발 및 배포에서의 개방성과 투명성을 평가합니다. 오픈소스 LLM은 코드베이스, 모델 아키텍처, 훈련 데이터 등을 자유롭게 제공하는 반면, 오픈웨이트 LLM은 학습된 모델 가중치만 공개하는 경우가 많습니다. 이러한 연구 설계는 기존의 오픈소스 기준 및 투명성 정의에 기반하여 AI 문헌에 대한 광범위한 검토를 포함합니다.

- **Performance Highlights**: 연구는 DeepSeek, ChatGPT 등 여러 모델의 증가하는 관심을 강조합니다. 그러나 몇몇 모델이 오픈소스라 하더라도 훈련 데이터나 코드와 같은 필수 정보는 제공하지 않는다는 점을 지적합니다. 이 연구는 또한 인공지능의 신뢰성을 높이기 위해 완전한 투명성이 필요하다는 것을 강조하며, AI의 윤리적 발전을 위한 보다 명확한 기준이 필요함을 역설합니다.



### TurboFuzzLLM: Turbocharging Mutation-based Fuzzing for Effectively Jailbreaking Large Language Models in Practic (https://arxiv.org/abs/2502.18504)
Comments:
          Accepted at NAACL 2025 industry track, 12 pages, 5 figures

- **What's New**: 이 논문에서는 TurboFuzzLLM이라는 새로운 변이 기반의 퍼징 기법을 제안하고 있습니다. 이 기술은 불법으로 사용자 프로프트를 통해 유해한 응답을 이끌어낼 수 있는 효과적인 jailbreaking 템플릿을 찾는 데 초점을 맞춥니다. TurboFuzzLLM은 기존 템플릿 기반 공격 기법의 한계를 극복하고, 자동으로 효과적인 jailbreaking 템플릿을 생성할 수 있는 기능을 추가하여 공격 성공률을 95% 이상 달성했습니다. 이는 사용자 프로프트를 통해 블랙박스 접근으로 타겟 LLM을 공격하는 데 필요한 효과적인 방법을 제공합니다.

- **Technical Details**: TurboFuzzLLM은 기존의 문서에서 제안된 GPTFuzzer를 기반으로 하지만, 새로운 선택 정책과 효율성을 강조한 휴리스틱을 추가하여 개선된 기능을 제공합니다. 변이 라이브러리를 확장하고, 공격할 모델에 대해 변형된 템플릿을 생성하기 위해 반복적으로 퍼징을 수행하는 프로세스를 포함합니다. 각 퍼징 반복에서 새로운 템플릿과 악성 질문을 결합하여 공격 프롬프트를 생성하고, 이를 통해 얻은 응답을 평가하여 효과성을 검증합니다.

- **Performance Highlights**: TurboFuzzLLM은 다양한 목표 LLM에 대해 일관되게 뛰어난 공격 성공률을 기록하였으며, GPTFuzzer 및 기타 최신 기법들과 비교하였을 때 더욱 높은 성과를 나타냈습니다. 이 시스템은 새로운 유해 질문에 대해서도 잘 일반화된 템플릿을 학습할 수 있으며, 생성된 레드 팀(Red teaming) 데이터는 모델의 내장 방어력을 개선하는 데에도 활용될 수 있습니다. 실험결과는 TurboFuzzLLM의 각 개별 업그레이드의 기여도를 보여주는 절차적 연구를 포함하고 있습니다.



### Deep Learning-based Dual Watermarking for Image Copyright Protection and Authentication (https://arxiv.org/abs/2502.18501)
Comments:
          IEEE Transactions on Artificial Intelligence. 2024 Oct 24

- **What's New**: 이번 논문에서는 디지털 이미지의 무결성과 진위성을 보장하기 위해 심층 학습(Deep Learning) 기반의 이중 보이지 않는 워터마크 기술을 제안합니다. 이 기술은 이미지의 콘텐츠 인증, 출처 인증 및 저작권 보호를 수행하며, 콘텐츠 보존 조작에 대한 강인함을 보여줍니다. 제안된 방법은 이미지의 암호화 해시와 주요 특징을 퍼셉션 해시(perceptual hash) 형태로 워터마크로 사용하며, 이에 따라 워터마크 복제가 불가능합니다.

- **Technical Details**: 제안된 기술은 두 개의 독립적인 추출 단계로 구성되어 있어 각각 저작권 보호와 콘텐츠 및 출처 인증을 위한 작업을 수행합니다. 또한, 기존의 전통적인 워터마킹 기술의 고유한 과제를 해결할 수 있는 심층 학습 기반 워터마킹 접근 방식을 통합하여 성능과 효율성에서 개선된 결과를 제공합니다. 이 연구에서 사용된 주요 기술 중 하나는 저작권 인증을 위한 고급 해싱 및 워터마킹 기법입니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 우리의 워터마크 추출 정확도가 높으며, 원본 이미지에 최소한의 변화만을 유도하는 높은 피크 신호 대 잡음 비율(PSNR)과 구조적 유사도 지수(SSIM)를 기록하였습니다. 심층 학습 기반의 이중 워터마킹 기술이 기존 방법과의 비교를 통해 높은 효율성을 입증하였으며, 저작권 보호 및 이미지의 무결성을 확보하는 데의 유용성을 강조합니다.



### Mechanistic Understanding of Language Models in Syntactic Code Completion (https://arxiv.org/abs/2502.18499)
Comments:
          10 pages, 4 figures, accepted to the AAAI 2025 Workshop on Towards Knowledgeable Foundation Models

- **What's New**: 최근 언어 모델(Models, LMs)이 코드 생성 작업에서 뛰어난 성능을 보이고 있지만, 코드 전용 데이터셋으로 미세 조정된 결과인 코드 LMs의 내부 의사결정 과정에 대한 이해는 여전히 부족합니다. 이러한 부족한 이해는 실생활에서 사용될 때 의도하지 않은 피해를 초래할 수 있습니다. 이 연구는 CodeLlama-7b 모델을 사용하여 닫는 괄호 작업을 수행하는 코드 LMs의 메커니즘을 탐구한 최초의 작업 중 하나입니다.

- **Technical Details**: 이 연구에서는 Synthetic Dataset을 생성하여 코드 LMs의 구문 완성 성능을 체계적으로 분석합니다. 데이터셋은 2, 3, 4개의 닫는 괄호를 필요로 하는 168개의 프롬프트를 포함하며, 구조적으로 복잡한 호출을 포함합니다. CodeLlama-7b 모델이 중간 및 후반 레이어에서만 올바른 토큰을 예측할 수 있다는 점을 발견하였고, 다수의 주목(attention) 헤드가 이미 닫힌 괄호의 수를 추적하는 데 중요한 역할을 했습니다.

- **Performance Highlights**: CodeLlama-7b 모델은 닫는 괄호가 필요할 때 중간 이후 레이어에서만 올바른 타겟 토큰을 인식합니다. 올바른 토큰 예측과 반대 토큰 예측을 비교할 때, multi-head attention (MHA) 서브 레이어가 가장 중요한 기여를 합니다. 마지막으로, 특정 주목 헤드는 이미 닫힌 괄호의 수를 정확하게 추적하지만, 몇몇은 잘못된 지식 연관성을 보여줘 성능에 부정적인 영향을 미쳤습니다.



### A Comprehensive Survey on Composed Image Retrieva (https://arxiv.org/abs/2502.18495)
- **What's New**: 이 논문에서는 Composed Image Retrieval (CIR)이라는 새로운 이미지 검색 작업을 체계적으로 검토하고 있습니다. CIR은 참조 이미지와 사용자가 원하는 변경 사항을 나타내는 수정 텍스트를 조합하여 사용자에게 보다 유연한 검색 방식을 제공합니다. 이 작업에 대한 포괄적인 리뷰가 현재 존재하지 않기 때문에, 120개 이상의 관련 연구를 종합하여 이 분야의 발전을 조망하고 있습니다.

- **Technical Details**: CIR의 주요 기술적 문제는 세 가지 주요 도전 과제를 포함합니다. 첫째, Multimodal Query Fusion은 수정 텍스트와 참조 이미지가 사용자의 검색 의도를 전달하는 데 상보적인 역할을 하는 것을 포함하여, 효과적인 멀티모달 융합 기능을 학습해야 합니다. 둘째, Target Images Matching은 멀티모달 쿼리와 목표 이미지 간의 의미적 간격을 해결하는 것을 목표로 합니다. 마지막으로, Scale of Training Data 문제는 학습 샘플을 만드는 데 필요한 비용과 노동 집약성을 다루고 있습니다.

- **Performance Highlights**: 기존 CIR 모델은 일반적으로 supervised learning과 zero-shot learning 두 가지 방법으로 구분됩니다. 감독 방식에서는 주석이 달린 학습 샘플이 필요하며, 제로샷 방식에서는 대규모 이미지-텍스트 쌍을 활용하여 사전 학습을 수행합니다. 다양한 데이터 세트와 실험 결과를 비교하여 기존의 supervised 및 zero-shot CIR 방법을 분석하고, 향후 연구 방향에 대한 통찰을 제공하여 연구자들에게 유용한 지침을 제시합니다.



### Rule-based autocorrection of Piping and Instrumentation Diagrams (P&IDs) on graphs (https://arxiv.org/abs/2502.18493)
- **What's New**: 이 연구는 파이프 및 계기 다이어그램(P&ID)의 오류 탐지 및 수정 지원을 위한 규칙 기반 방법을 제안합니다. 기존의 수동 검토 프로세스 대신, P&ID를 그래프 형태로 표현하여 자동화된 오류 탐지를 가능하게 합니다. 연구진은 화학 공학 지식과 휴리스틱에 기반한 33개의 규칙을 개발했으며, 사례 연구를 통해 신뢰성과 유효성을 입증했습니다.

- **Technical Details**: 제안된 방법론에서는 pyDEXPI 패키지를 사용하여 DEXPI 표준에 따라 스마트 P&ID를 그래프로 변환합니다. 다양한 공학 규칙을 적용한 규칙 그래프를 사용하여 오류 패턴을 탐지하고 수정하는 방식으로 작동합니다. 이를 통해 원래의 P&ID 그래프를 수정하여 최종적으로 정정된 P&ID 그래프를 생성합니다.

- **Performance Highlights**: 사례 연구를 통해 제안된 규칙 기반 자동 수정 방법의 효과성을 강조하였습니다. 기존의 규칙 기반 접근 방식과 비교하여 DEXPI 표준과 통합할 수 있는 점에서 개선된 점을 보였으며, 공학적 오류를 보다 정확하게 감지할 수 있는 가능성을 제시했습니다.



### LLM4EFFI: Leveraging Large Language Models to Enhance Code Efficiency and Correctness (https://arxiv.org/abs/2502.18489)
- **What's New**: 이번 연구에서는 코드 효율성을 동시에 최적화하는 첫 번째 프레임워크인 Llm4Effi를 제안합니다. Llm4Effi는 효율성과 정확성을 모두 고려한 코드 생성을 가능하게 하는 새로운 패러다임을 제공합니다. 특히 알고리즘 탐색과 구현 최적화를 분리하여 효율성 최적화 프로세스를 두 가지 영역으로 나눕니다.

- **Technical Details**: Llm4Effi는 자연어로 설명된 프로그래밍 작업을 코드 지향 문제로 형식화하여, 알고리즘 접근 방식에 대한 탐색과 복잡성 분석을 포함합니다. 이 시스템은 코드 구현 전략을 제안하고, 생선된 코드의 정확성을 보장하기 위해 이중 검증 기반의 적응형 테스트 프레임워크를 도입합니다. 고품질 코드를 생성하기 위해 구현 단계에서의 세심한 고려가 필요합니다.

- **Performance Highlights**: Llm4Effi는 최근에 제안된 코드 효율성 벤치마크에서 코드의 정확도와 효율성을 모두 향상시키며 최고의 성능을 달성했습니다. 특히 DeepSeek-V3 백본을 사용할 때 ENAMEL에서 eff@1이 9.27% 증가하고, Mercury에서 DPS_norm이 6.63% 증가하는 성과를 기록했습니다. 이 연구는 코드 효율성 커뮤니티의 발전에 기여할 것으로 기대됩니다.



### AuPair: Golden Example Pairs for Code Repair (https://arxiv.org/abs/2502.18487)
- **What's New**: 이번 논문에서는 대규모 언어 모델(Large Language Models, LLMs)의 인퍼런스(inference) 성능을 향상시키기 위해, 자기 수리(self-repair) 능력을 활용하는 방법을 제안합니다. LLM이 초기 오류 응답을 수정하여 더 나은 응답을 생성할 수 있도록, 개별 문제에 대한 '금동 쌍(AuPairs)' 예시를 순서대로 생성하고 선택하는 접근법을 개발했습니다. 이 접근법은 다양한 수정 솔루션을 생성해 주며, 최고 점수의 솔루션이 최종 답변으로 선택됩니다.

- **Technical Details**: 제안된 알고리즘은 N개의 LLM 호출에 따라 최대 N개의 AuPair를 생성합니다. 각 AuPair는 초기 추정 및 해당 프로그래밍 문제에 대한 수정으로 구성되어 있습니다. 이러한 AuPair는 앙상블(ensemble) 방식으로 선택되어 상호 보완적이며 유용한 조합을 형성합니다. 이러한 방식으로 수집된 AuPair는 인퍼런스 시점에 하나의 예로 제공되어 최적의 솔루션을 생성하도록 돕습니다.

- **Performance Highlights**: 제안된 알고리즘은 5개의 다양한 LLM 모델에서 코드 수리 작업에 대해 뛰어난 성능을 보였습니다. AuPair는 기존의 best-of-N 및 자기 수리 접근법보다 훨씬 뛰어난 성능을 보여주며, 다양한 데이터셋과 모델 크기에서도 강력한 일반화 능력을 발휘합니다. 이 알고리즘은 상대적으로 적은 인퍼런스 시간 예산에도 더욱 효과적인 성능 스케일링을 보여, 실질적인 효용을 증대시킵니다.



### AI Enhanced Ontology Driven NLP for Intelligent Cloud Resource Query Processing Using Knowledge Graphs (https://arxiv.org/abs/2502.18484)
Comments:
          8 pages, 5 figures, 4 tables. This paper not published at else where yet. The experimental setup has a potential to be revised using real time resources. Authors: Krishna Chaitanya Sunkara (IEEE Senior Member, Raleigh, NC, USA, Independent Researcher), Krishnaiah Narukulla (IEEE Senior Member, San Jose, CA, USA, Independent Researcher)

- **What's New**: 이번 논문에서는 전통적인 클라우드 자원 검색 방식을 개선하기 위한 새로운 접근법을 제안합니다. 기존의 키워드 기반 검색이나 GUID에 의존하는 방식은 정확한 일치를 요구하며 사용자가 자원을 찾는데 많은 노력을 필요로 했습니다. 이러한 방식을 개선하기 위해, 논문에서는 자연어 쿼리의 의도를 명확히 이해하고, 자원의 행동이나 운영, 기능, 관계성 등을 기반으로 검색할 수 있는 방법을 모색합니다.

- **Technical Details**: 이 논문은 온톨로지 기반 의미론(ontology-based semantics)과 향상된 자연어 처리(Natural Language Processing, NLP) 기술을 활용하여 사용자가 보다 직관적이고 이해하기 쉬운 쿼리를 생성할 수 있도록 합니다. 클라우드 자원, 그 상호작용 및 행동에 대한 온톨로지를 구축하여 동적 의도 추출(dynamic intent extraction) 및 관련성 순위를 매기는 기능을 구현합니다. Latent Semantic Indexing (LSI)와 AI 모델을 사용하여 자원을 발견할 때의 맥락을 정확히 파악할 수 있는 세미틱 지식 기반(semantic knowledge base)를 형성합니다.

- **Performance Highlights**: 제안된 프레임워크는 AI 기반 데이터 크롤러를 통해 자동으로 온톨로지를 추출하는 파이프라인을 구축하여, 자원 검색의 효율성을 높입니다. 이 시스템은 사용자가 단순히 자원의 목록을 얻는 것이 아니라, 시스템의 행동 원인 분석, 컴플라이언스 체크, 용량 추정, 네트워크 제약 사항 파악 및 문제 해결, 비즈니스 통찰력을 획득하는 데 도움을 줍니다. 결과적으로, 이 접근법은 자원 검색의 효율성을 혁신적으로 향상시키고, 사용자 경험을 크게 개선할 수 있는 잠재력을 지니고 있습니다.



### Modeling Churn in Recommender Systems with Aggregated Preferences (https://arxiv.org/abs/2502.18483)
- **What's New**: 본 논문에서는 이전의 사용자 데이터에 의존하던 추천 시스템(Recommendation Systems, RSs)이 이제 집계된 사용자 정보를 활용해야 한다고 주장합니다. GDPR 및 CCPA와 같은 규제는 개인 사용자 데이터 접근이 제한되어 있어, RS들이 집계된 정보만으로 추천 프로세스를 개선해야 하는 상황을 만들어냅니다. 이러한 변화는 사용자 이탈 위험(Churn Risk)을 증가시키며 이를 해결하기 위한 새로운 모델인 Rec-APC를 제안합니다.

- **Technical Details**: Rec-APC 모델은 RS가 사용자 유형과 콘텐츠 유형에 대한 집계된 만족 수준에 대한 확률적 사전(prior)을 갖는다고 가정합니다. 사용자 세션은 RS가 특정 유형의 알 수 없는 사용자를 샘플링하여 진행되며, RS는 사용자의 좋아요 또는 싫어요 피드백을 받아 추천을 개선합니다. 모델은 탐색(exploration)과 활용(exploitation) 간의 균형을 맞추고, 이탈 위험을 고려하여 사용자 유틸리티를 최대화하는 것을 목적으로 합니다.

- **Performance Highlights**: 본 연구에서 개발된 알고리즘은 Branch-and-Bound 접근 방식을 사용하여 RS의 추천 문제를 해결합니다. 기존의 POMDP(Partially Observable Markov Decision Process)와 비교할 때, 다양한 사용자 유형이 존재하는 경우에 더 나은 성과를 보이는 것으로 나타났습니다. 최적 정책은 유한한 추천 횟수 이후에 수렴하며, 이는 탐색에서 활용으로의 전환을 의미합니다.



### MixLLM: Dynamic Routing in Mixed Large Language Models (https://arxiv.org/abs/2502.18482)
Comments:
          11 pages, 7 figures, accepted by NAACL 2025 main conference

- **What's New**: 새롭고 혁신적인 연구인 MixLLM은 다이나믹 컨텍스트 밴딧 기반의 라우팅 시스템을 개발하여, 쿼리와 LLM의 최적 매핑을 가능하게 합니다. 이 시스템은 쿼리 태그를 활용하여 쿼리 임베딩을 향상시키고, 각각의 LLM에 대한 응답 품질 및 비용을 추정하는 경량화된 예측 모델을 설계합니다. Mixed LLM은 응답 품질과 비용, 지연 시간 간의 균형을 최적화하여 높은 효율을 실현합니다.

- **Technical Details**: MixLLM은 InsTag 모델에서 생성된 태그를 사용하여 쿼리 표현을 개선하는 태그 향상 임베딩 모델을 제안합니다. 예측 모델은 각 LLM의 응답 품질과 비용을 평가하며, 메타 의사결정자는 이러한 예측을 기반으로 최적의 LLM을 선택합니다. 이 과정은 새로운 LLM이 도입되더라도 시스템 전체 재훈련이 필요 없도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, MixLLM은 응답 품질에서 GPT-4의 97.25%를 달성하고, 비용은 24.18%로 유지하여 응답 품질과 비용, 지연 시간 간의 최적의 균형을 실현하였습니다. 또한, 지연 패널티를 도입하여 교통 혼잡 및 높은 지연 문제를 피하고, 다양한 환경 및 사용자 피드백에 적응하는 지속적인 훈련의 장점을 누릴 수 있습니다.



### MDE: Modality Discrimination Enhancement for Multi-modal Recommendation (https://arxiv.org/abs/2502.18481)
- **What's New**: 이번 연구에서는 Multi-modal recommendation system의 성능 향상을 위해 Modality Distinctiveness Enhancement (MDE) 프레임워크를 제안합니다. 이 프레임워크는 모달리티 간의 공통적인 정보는 유지하면서 모달리티별 독특한 특성을 추출하고 강화하는 데 중점을 두고 있습니다. 특히, 모달리티 간 관계 정렬과 구별을 위한 균형 기구를 도입하여 추천의 정확성을 높이는 데 기여합니다.

- **Technical Details**: MDE 프레임워크는 이종 사용자-아이템 그래프와 동종 그래프를 구성하여 다중 모달 피처 표현을 학습합니다. 또한, 사용자의 모달리티 선호도를 학습하여 모달리티 특유의 정보와 공유 정보를 효과적으로 융합합니다. 이 과정에서 가중된 모달리티 구별 및 정렬 손실을 도입하여 모달리티별 특징을 증대시키고, 추천 시스템의 품질을 높이는 데 기여합니다.

- **Performance Highlights**: 세 가지 공공 데이터 세트에 대한 광범위한 실험 결과, 제안한 MDE 접근 방식이 기존 최첨단 방법보다 뛰어난 성능을 보였습니다. 모달리티 공유 및 특정 특성을 동시에 고려함으로써, 기존 연구에서 간과되었던 부분을 보완하며 효과적인 추천 결과를 달성했습니다.



### QExplorer: Large Language Model Based Query Extraction for Toxic Content Exploration (https://arxiv.org/abs/2502.18480)
- **What's New**: 이번 연구에서는 정보 검색에서 유효한 쿼리 자동 추출의 도전 과제를 다루며, 특히 독성 콘텐츠 탐색과 관련된 내용입니다. 새로운 접근 방식인 QExplorer를 제안하여, 지능형 대형 언어 모델(LLM)의 기능을 활용하여 유사한 내용의 탐색을 위한 효율적인 쿼리를 직접 추출할 수 있게 되었습니다. 2단계 훈련 과정인 Supervised Fine-Tuning(SFT)과 Direct Preference Optimization(DPO)을 포함하여, 검색 시스템의 피드백을 활용한 데이터 세트 구축이 중요한 요소로 작용합니다.

- **Technical Details**: QExplorer 접근 방식은 두 가지 단계의 훈련 과정을 포함하며, 이는 지침 기반의 SFT와 사용자의 선호도를 정렬하는 DPO를 포함합니다. 이 모델은 오프라인 및 온라인 실험을 통해 효과성을 검증하며, 그 과정에서 LLM 기반의 자동 쿼리 추출 방법이 기존 LLM 및 인간 수행보다 우수한 성능을 보인다고 보고합니다. 이는 키워드 추출을 위한 인간의 직관과 LLM의 성능을 결합하여 더 나은 발견 결과를 도출하기 위해 설계되었습니다.

- **Performance Highlights**: 실험 결과, QExplorer는 기존 베이스라인 모델과 인간의 쿼리 추출 성능을 초월하는 것으로 나타났습니다. 온라인 배치에서 독성 항목의 탐지가 크게 증가했으며, 이는 QExplorer의 효과적인 쿼리 구성 덕분입니다. 또한 대화형 LLM 기반의 자동 쿼리 추출이 독성 콘텐츠 탐색의 효율성을 크게 향상시킬 수 있음을 입증했습니다.



### Beyond Self-Consistency: Loss-Balanced Perturbation-Based Regularization Improves Industrial-Scale Ads Ranking (https://arxiv.org/abs/2502.18478)
- **What's New**: 이 논문은 대규모 광고 순위 모델에서의Perturbation-based regularization 기법의 성공적인 적용을 처음으로 탐구하고 있습니다. 특히,Loss-Balanced Small Perturbation Regularization (LSPR)라는 새로운 정규화 알고리즘을 제안하여 다양한 딥러닝 모델에 적용 가능성을 보여주고 있습니다. 이 연구는 또한 기존의Self-Consistency Regularization (SCR)보다 LSPR가 더 높은 성능을 보인다는 점을 강조합니다.

- **Technical Details**: 이 논문에서는 대규모 광고 순위 시스템에서의Perturbation-based 정규화 기법에 대해 심층적으로 다루고 있습니다. 특히, LSPR는 입력 데이터에 소규모의 노이즈를 추가하여 새로운 샘플을 생성하고, 이를 훈련 데이터에 포함시키되 손실 함수 계산에서 가중치를 조정하는 방식으로 작동합니다. 이는 더 나은 모델 파라미터 정렬과 낮은 에러를 달성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, LSPR을 적용함으로써 산업 규모의 광고 순위 시스템에서 0.1%에서 0.3%의 상대적 Normalized Entropy (NE) 향상을 달성했습니다. 이러한 성능 개선은 오프라인 실험뿐만 아니라 온라인 실험에서도 확인되었으며, 이는 실제 환경에서도 효과적인 성과를 보여줍니다. 이 연구는 대규모 광고 추천 시스템에Perturbation-based 정규화 기술을 성공적으로 통합한 첫 사례로, 다양한 산업적 적용 가능성을 제시합니다.



### Recommendations Beyond Catalogs: Diffusion Models for Personalized Generation (https://arxiv.org/abs/2502.18477)
- **What's New**: REBECA는 기존 카탈로그의 아이템을 단순히 검색하는 것이 아니라 개별 사용자의 취향에 맞춰 새로운 아이템을 생성하는 생성 추천 시스템입니다. 이는 사용자 피드백을 직접 활용하여 개인화된 추천을 가능하게 하는 첫 번째 확률적 프레임워크로, 사용자의 과거 평가 데이터만을 기반으로 합니다. REBECA는 텍스트 기반의 프롬프트 없이도 유연한 개별화 처리를 통해 추천의 질을 높이는 새로운 가능성을 제시합니다.

- **Technical Details**: REBECA는 사용자의 피드백(예: 좋아요, 싫어요)을 통한 사용자-아이템 상호작용을 파악하여 텍스트 프리 개발 및 추론이 가능한 구조로 설계되었습니다. 이 시스템은 고도로 표현력이 풍부한 사전 훈련된 이미지 생성기 위에 효율적인 어댑터를 추가하여 대규모 언어 모델(LLMs)의 중개 없이 사용자 특정 선호도를 보존하는 데 초점을 둡니다. 이러한 방식은 기존의 생성 파이프라인에 원활하게 통합할 수 있으며, 높은 표현성을 유지하면서 비용이 많이 드는 재훈련을 피할 수 있습니다.

- **Performance Highlights**: REBECA는 실제 데이터로 검증을 진행하여 개인화 지표를 새롭게 제안하고 다양한 실험을 통해 높은 품질의 개인화된 추천 결과를 생성하는 것이 확인되었습니다. 생성된 이미지는 사용자의 독특한 선호를 반영하며, 다양한 평가 방법론을 통해 개인화의 정도를 체계적으로 측정하고 검증합니다. 이러한 접근은 추천 시스템에서의 생성 AI의 통합 가능성을 확장시키며 동적 콘텐츠 생성의 새로운 방향성을 제시합니다.



### A Contemporary Survey of Large Language Model Assisted Program Analysis (https://arxiv.org/abs/2502.18474)
- **What's New**: 본 논문에서는 소프트웨어 시스템의 복잡성 증가에 따른 프로그램 분석의 발전을 다룹니다. 전통적인 방법들이 현대 소프트웨어 개발의 요구를 충족하지 못함에 따라, 특히 Large Language Models (LLMs)의 맥락 인식 기능이 주목받고 있습니다. 연구자들은 LLM의 프로그램 분석에 대한 잠재력을 인식하고 그 활용에 대한 연구를 활발히 진행해 왔습니다.

- **Technical Details**: 이 리뷰는 LLM의 프로그램 분석에서의 응용을 정리하고, 기존 연구를 정적 분석(static analysis), 동적 분석(dynamic analysis), 하이브리드 접근 방식으로 분류하였습니다. 최근 연구를 검토하고 종합함으로써, 이 분야의 미래 방향과 도전 과제를 식별하였습니다. 이를 통해 LLM의 잠재력을 밝히고 프로그램 분석의 실제 적용 사례를 제시합니다.

- **Performance Highlights**: 이 설문조사는 LLM을 활용한 프로그램 분석의 발전을 보여주며, 보안 연구자들이 탐지 프레임워크를 개선하거나 도메인 특화 모델을 개발하는 데 유용한 통찰력을 제공합니다. LLM의 활용이 프로그램 분석 기법을 발전시키는 데 기여할 수 있음을 보여줍니다.



### FinBloom: Knowledge Grounding Large Language Model with Real-time Financial Data (https://arxiv.org/abs/2502.18471)
Comments:
          27 pages, 9 tables

- **What's New**: 이 논문에서는 금융 쿼리를 처리하기 위해 LLM에 실시간 데이터 모듈과 함께 동작하는 Financial Agent라는 지식 기반 접근 방식을 소개합니다. 연구팀은 50,000개 이상의 금융 쿼리와 해당 문맥을 포함한 Financial Context Dataset을 개발했으며, 금융 뉴스 기사와 SEC 파일들로 훈련된 70억 매개변수의 FinBloom 7B LLM을 제안합니다.

- **Technical Details**: 논문에서는 다양한 도메인 특정 태스크를 효과적으로 수행할 수 있도록 LLM을 동결한 상태에서 모듈을 추가하는 방법을 사용합니다. 이 시스템은 실시간 데이터를 처리하기 위해 두 개의 데이터 레포지토리(탭 형식의 가격 데이터와 텍스트 기반의 뉴스 데이터)를 유지하며, 사용자의 쿼리를 분석하여 관련 데이터를 추출하고 이를 텍스트 형식으로 변환합니다.

- **Performance Highlights**: 제안된 방법론을 통해 금융 쿼리에 대한 응답을 생성하는 데 필요한 동적 문맥 정보를 신속하게 제공할 수 있습니다. 사용자의 쿼리에 최신 데이터를 결합시킴으로써 LLM이 보다 정확한 응답을 생성할 수 있도록 지원하여, 실제 금융 결정을 내릴 때 유용성을 크게 향상시킬 것으로 기대됩니다.



### SOK: Exploring Hallucinations and Security Risks in AI-Assisted Software Development with Insights for LLM Deploymen (https://arxiv.org/abs/2502.18468)
- **What's New**: 이번 논문에서는 GitHub Copilot, ChatGPT, Cursor AI 및 Codeium AI와 같은 Large Language Models (LLMs)의 소프트웨어 개발 통합에 대해 설명합니다. 이러한 도구들은 코드 생성, 리팩토링 및 디버깅을 자동화하여 생산성을 크게 향상시키지만, 보안 취약성 및 코드 품질 문제 등 여러 가지 도전 과제를 동반합니다. 연구는 사용자 피드백 및 보안 분석을 기반으로 이러한 AI 도구의 장점과 위험성을 종합적으로 분석합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)은 자연 언어 설명으로부터 자동으로 코드를 생성할 수 있게 하여 개발자를 지원합니다. 그러나 트레이닝된 공개 코드에는 불안전한 코딩 관행이 포함되어 있을 수 있어, 이를 통해 생성된 코드의 보안 취약점이 존재할 수 있습니다. 이는 제3자의 서비스나 인기 라이브러리를 대상으로 하는 공급망 공격과 관련이 있으며, LLMs는 이러한 공격을 지원할 가능성이 있습니다. 또한, 인간 프로그래머는 대체할 수 없지만, 코드 작성과 디버깅의 효율성을 높이는 데 의해 소프트웨어 개발 환경에서 점점 더 가치가 커지고 있습니다.

- **Performance Highlights**: 사용자 피드백 데이터에서 ChatGPT는 코드 생성, 리팩토링, 코드 설명에서 높은 평가를 받아 다재다능한 도구로 나타났습니다. Cursor AI는 균형 잡힌 성과를 보였으며, Codeium AI는 전체적으로 저조한 평가를 받아 경쟁력을 높이기 위한 개선이 필요합니다. Copilot은 모든 면에서 우수한 결과를 내었으며, 특히 코드 설명 및 자동 완성에서 높은 점수를 기록했습니다. 조사 결과, ChatGPT가 가장 높은 긍정적인 반응을 얻은 반면, Copilot은 사용자 피드백에서 엇갈린 반응을 받았습니다.



### ChatGPT vs. DeepSeek: A Comparative Study on AI-Based Code Generation (https://arxiv.org/abs/2502.18467)
- **What's New**: 이번 연구는 AI 기반 코드 생성에서 ChatGPT(버전 o1)와 DeepSeek(버전 R1)의 Python 코드 생성 성능을 비교합니다. 두 모델 모두 대규모 언어 모델(LLMs)을 활용하여 소프트웨어 개발에 혁신을 가져오고 있으며, 코드 품질 및 정확성을 평가합니다.

- **Technical Details**: 연구는 온라인 판사 코딩 도전을 통해 코드의 정확성(online judge verdicts), 코드 품질(Pylint/Flake8), 효율성(실행 시간 및 메모리 사용량)을 평가합니다. DeepSeek는 알고리즘 작업에서 더 높은 정확도를 보였으며, ChatGPT는 때때로 여러 번의 시도를 필요로 했습니다.

- **Performance Highlights**: DeepSeek는 Python 코드 생성에서 우수한 정확성을 보여주었으며, 알고리즘 문제 해결에서 더 적은 시도를 요구하는 경향이 있습니다. 두 모델은 실행 시간 및 메모리 사용에서 비슷한 효율성을 보였으며, 이러한 결과는 AI 코딩 보조 도구를 선택하는 개발자들에게 통찰을 제공합니다.



### MLScent A tool for Anti-pattern detection in ML projects (https://arxiv.org/abs/2502.18466)
Comments:
          4th International Conference on AI Engineering Software Engineering for AI , CAIN 2025

- **What's New**: 이 논문은 머신 러닝(ML) 프로젝트에 특화된 코드 품질을 향상시키기 위해 'MLScent'라는 새로운 정적 분석 도구를 소개합니다. MLScent는 다양한 ML 프레임워크에서 76개의 독립적인 탐지기를 구현하여 코드 냄새(code smells)와 안티 패턴을 감지하는 기능을 가지고 있습니다. 이 도구는 TensorFlow, PyTorch, Scikit-learn 및 Hugging Face와 같은 프레임워크뿐만 아니라 Pandas와 NumPy와 같은 데이터 과학 라이브러리에서도 사용됩니다. 이를 통해 ML 개발자들은 코드 품질과 유지 보수성을 개선할 수 있는 기회를 가집니다.

- **Technical Details**: MLScent는 복잡한 추상 구문 트리(Abstract Syntax Tree) 분석을 활용하여 ML 프로젝트 특유의 코드 냄새를 탐지합니다. 76개의 탐지기를 통해 다양한 ML 프레임워크와 데이터 과학 라이브러리를 지원하며, 데이터 전처리 및 모델 학습 워크플로우에 대한 특화된 분석도 제공합니다. 이 도구는 일반 ML 코드 냄새 감지를 위한 16개의 탐지기도 포함되어 있어 보다 전반적인 코드 품질 분석이 가능합니다.

- **Performance Highlights**: MLScent의 효과성은 정량적인 분류 메트릭과 ML 실무자들로부터의 사용자 연구 피드백을 통해 입증되었습니다. 실제 프로젝트에서 ML 프레임워크 특유의 안티 패턴, 데이터 처리 문제 및 일반 ML 코드 냄새를 높은 정확도로 식별하는 데 성공했습니다. 이러한 결과는 실용적인 ML 프로젝트의 코드 품질 향상을 위한 강력한 도구로 자리매김할 가능성을 보여 줍니다.




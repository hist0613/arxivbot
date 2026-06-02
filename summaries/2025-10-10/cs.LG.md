New uploads on arXiv(cs.CL)

### ArenaBencher: Automatic Benchmark Evolution via Multi-Model Competitive Evaluation (https://arxiv.org/abs/2510.08569)
Comments:
          Preprint

- **What's New**: 이 논문은 ArenaBencher라는 기존 모델에 얽매이지 않는 벤치마크 진화 프레임워크를 제안합니다. ArenaBencher는 자동으로 테스트 사례를 업데이트하여 각 모델의 성능을 공정하게 비교할 수 있게 합니다. 이 시스템은 모델에 의해 메모라이즈된 내용이 아닌 진정한 일반화를 평가할 수 있도록 도와줍니다.

- **Technical Details**: ArenaBencher는 기존 벤치마크와 다양한 모델 풀을 기반으로 각 테스트 사례의 핵심 능력을 추론합니다. 예를 들어, 수학적 추론 벤치마크에서 multi-step arithmetic(다단계 산술)을 테스트하는 문제를 생성합니다. 검증 단계에서는 LLM을 판별자로 사용하여 질의-라벨 쌍의 정합성을 평가하고, 두 번째 단계에서는 여러 모델의 피드백을 집계하여 성능 격차를 드러내는 문제들을 선택합니다.

- **Performance Highlights**: ArenaBencher는 수학 문제 해결, 상식 추론 및 안전 분야에서 적용되어 새로운 실패 모드를 발견하고 테스트 목표의 일치를 유지하면서 난이도를 높이는 업데이트를 제공합니다. 이러한 업데이트는 모델 간의 성능 차이를 더 뚜렷하게 드러내고, 공정하며, 신뢰성 있는 비교 평가를 위한 발전된 테스트 사례를 생성합니다.



### CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards (https://arxiv.org/abs/2510.08529)
- **What's New**: 이 연구에서는 Co-Evolving Multi-Agent Systems (CoMAS)라는 새로운 프레임워크를 소개합니다. CoMAS는 외부의 감독 없이 에이전트 간의 상호작용을 통해 자율적으로 개선될 수 있도록 설계되었습니다. 이는 기존의 강화 학습 기반 방법과는 달리 인간 지능에서 관찰되는 자가 발전 메커니즘에 더 가깝습니다.

- **Technical Details**: CoMAS는 풍부한 토론 역학으로부터 내재적 보상 신호(intrinsic reward signals)를 생성하고, LLM-as-a-judge 메커니즘을 통해 이 보상들을 도출합니다. 각 에이전트의 정책(policy)은 강화 학습(RL)을 통해 최적화되며, 이를 통해 분산형(decentralized) 및 확장 가능한(co-evolution) 자가 발전을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, CoMAS는 훈련되지 않은 에이전트들에 비해 일관되게 우수한 성능을 보였으며, 대부분의 평가 설정에서 최신 기술(state-of-the-art) 성과를 달성했습니다. 중재 연구(ablation studies)를 통해 상호작용 기반 보상 신호의 필요성이 확인되었고, 에이전트의 수와 다양성이 증가할수록 유망한 확장성을 보여주었습니다.



### Which Heads Matter for Reasoning? RL-Guided KV Cache Compression (https://arxiv.org/abs/2510.08525)
- **What's New**: 최근 발전된 Reasoning Large Language Models (LLMs)는 이전 단계를 재방문하고 대안적 접근법을 탐색하는 복잡한 추론 행동을 보여줍니다. 하지만 이러한 성과는 메모리 병목 현상을 초래하며, 기존의 KV cache 압축 방법들은 추론 모델에 맞지 않아서 성능 저하를 겪고 있습니다. 연구자들은 KV heads에서 기능적 이질성이 존재한다고 가정하며, 이를 활용해 RLKV라는 새로운 프레임워크를 제안했습니다.

- **Technical Details**: RLKV는 강화 학습(reinforcement learning)을 통해 이유가 중요한 heads를 식별하고, KV cache 사용과 추론 품질 간의 관계를 최적화합니다. 훈련 중 실제 샘플에서 생성된 보상(reward)을 통해, RL은 추론 행동과 관련된 heads를 자연스럽게 식별합니다. 이를 통해 추론 행동을 유지하되, 다른 heads에는 압축된 KV cache를 적용하여 효율적인 추론을 달성합니다.

- **Performance Highlights**: 실험 결과, 소수의 attention heads가 추론에 필수적임을 발견했으며, 이를 통해 기존 방법보다 20-50%의 KV cache 사용 감소를 달성했습니다. RLKV는 추론 품질을 유지하면서도 KV cache를 압축할 수 있는 뛰어난 성능을 보여줍니다. 이 접근 방식은 기존의 retrieval heads와는 기능적으로 구분되는 중요 heads를 명확히 합니다.



### Efficient Prompt Optimisation for Legal Text Classification with Proxy Prompt Evaluator (https://arxiv.org/abs/2510.08524)
Comments:
          Accepted at NLLP@EMNLP 2025

- **What's New**: 이 논문은 언어 모델의 성능을 특정 작업에 맞게 향상시키기 위해 프롬프트 최적화를 정교화하는 새로운 프레임워크를 제안합니다. 특히, 이 연구는 서비스 약관(Terms of Service, ToS)의 불공정 조항을 탐지하는 법률 NLP 문제에 적합한 최적화 방법론을 다룹니다. Monte Carlo Tree Search (MCTS)와 프록시 프롬프트 평가기를 결합하여 비용 효율적으로 프롬프트 공간을 탐색할 수 있도록 하였습니다.

- **Technical Details**: 연구에서는 MCTS를 사용하여 프롬프트 후보의 탐색 효율성을 향상시키고, 프롬프트 성능을 평가하는 데 드는 높은 계산 비용을 줄이는 프록시 평가기를 제안합니다. 특히, 이 프록시 평가기는 목표 작업에서의 정확도를 예측하여 LLM의 반복적인 호출을 최소화하고, 전체 검증 세트에 걸쳐 프롬프트를 평가하는 데 도움을 줍니다. 결과적으로, 이 프레임워크는 더 나은 성능의 프롬프트를 발견하고, 계산 비용을 줄이는 데 성공했습니다.

- **Performance Highlights**: 실험 결과, MCTS 접근법은 기존의 최적화 프레임워크보다 더 높은 분류 정확도와 효율성을 보여주었습니다. 또한, 프록시 프롬프트 평가기를 결합했을 때, 유사한 이진 분류 성과를 달성하면서도 계산 비용이 줄어드는 효과를 확인했습니다. 이는 불공정한 ToS 조항 탐지와 같은 복잡한 법률 NLP 작업에 매우 성공적임을 입증했습니다.



### Neologism Learning for Controllability and Self-Verbalization (https://arxiv.org/abs/2510.08506)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)와의 커뮤니케이션에서 새로운 단어를 도입하는 방법으로, 최근에 제안된 신조어 학습(neologism learning) 개념을 확장합니다. 새로운 단어를 추가하여 기존 파라미터를 변경하지 않고도 모델의 이해도를 높일 수 있음을 보여줍니다. 이를 통해 논문은 아첨(flattery), 부정확한 대답(incorrect answers), 텍스트 길이(text length) 등의 개념을 조절할 수 있는 가능성을 제시합니다.

- **Technical Details**: 신조어 학습에서는 모델의 기존 단어 임베딩을 고정하고, 새로운 단어와 임베딩을 추가하여 특정 개념을 예시화하는 데이터셋을 통해 학습합니다. 이 과정에서 모델은 새로운 단어의 의미를 자연어로 설명할 수 있는 자기 언어화(self-verbalization)를 수행할 수 있음을 발견하였습니다. 예를 들어, 잘못된 답변을 나타내는 신조어가 특정 특징들을 설명하는 방식으로 자기 언어화 되는 것을 보여줍니다.

- **Performance Highlights**: 우리는 여러 개념에 대해 신조어 학습을 테스트하였으며, 특히 간단한 개념뿐만 아니라 AxBench에서 복잡한 개념에서도 강력한 조절이 가능함을 확인했습니다. 신조어 학습을 통해 기계 전용 동의어(machine-only synonyms)를 발견하였고, 이는 인간에게는 이상하거나 관련 없어 보이는 자기 언어화가 실제로 모델의 행동을 통제하는 데 유용하다는 것을 시사합니다. 최종적으로 우리는 세 가지 복잡한 개념에 대해 상호 연관성을 탐구하며, 이러한 개념들을 통해 신조어 학습의 가능성을 더욱 확장했습니다.



### DeepPrune: Parallel Scaling without Inter-trace Redundancy (https://arxiv.org/abs/2510.08483)
Comments:
          15 pages, 4 figures, please check out the project page: this https URL

- **What's New**: 이번 논문에서는 DeepPrune이라는 새로운 프레임워크를 제안합니다. 이 방법은 파라랠(Parallel) 추론 과정에서 중복된 경로를 효과적으로 줄이는 동시에 다양한 답변을 유지할 수 있도록 설계되었습니다. 연구 결과, 현재의 패러다임에서는 약 80%의 추론 경로가 중복되는 답변을 산출하며, 이를 해결하여 계산 효율성을 크게 향상시키고자 합니다.

- **Technical Details**: DeepPrune은 동적 가지치기(dynamic pruning)를 통한 효율적인 파라랠 스케일링을 가능하게 합니다. 이를 위해 긍정적인 개념적 손실(focal loss) 및 오버샘플링 기법으로 훈련된 특별한 판별 모델(judge model)을 활용하여 부분 추론 경로에서 답변 동등성을 예측합니다. 이 접근법은 0.87 AUROC를 기록하며, 온라인 그리디 클러스터링(online greedy clustering) 알고리즘을 활용하여 중복된 경로를 동적으로 제거합니다.

- **Performance Highlights**: 종합적으로 세 가지 주요 벤치마크(AIME 2024, AIME 2025, GPQA)와 여러 추론 모델을 통해 DeepPrune의 성능을 평가한 결과, 전통적인 합의 샘플링(consensus sampling) 방식에 비해 80% 이상의 토큰 소비 감소를 달성했습니다. 또한 정확도는 대부분의 경우에 3% 이내에서 경쟁력을 유지하며, AIME25 데이터셋에서는 토큰 소비를 91.6%까지 줄이는 동시에 정확도 또한 개선되었습니다. 이러한 결과는 DeepPrune이 고성능 추론을 더욱 효율적으로 가능하게 함을 보여줍니다.



### LeWiDi-2025 at NLPerspectives: The Third Edition of the Learning with Disagreements Shared Task (https://arxiv.org/abs/2510.08460)
Comments:
          14 pages; LeWiDi-2025 shared task description paper at NLPerspective workshop at EMNLP 2025

- **What's New**: 이번 연구는 AI 모델이 인간 판단의 변동성과 이견을 인식하는 능력에 초점을 맞추어 훈련되고 평가되어야 한다는 주장을 뒷받침합니다. 'Learning With Disagreements (LeWiDi)'라는 시리즈의 세 번째 작업은 4개의 데이터셋(예: paraphrase identification, irony detection, sarcasm detection, natural language inference)을 통해 이러한 접근 방식을 더욱 확장했습니다. 새로운 평가 방법인 soft-label 접근법과 perspectivist 접근법을 도입하였으며, 이를 통해 AI 시스템의 성능을 더욱 정교하게 측정할 수 있게 되었습니다.

- **Technical Details**: 연구팀은 기존의 크로스 엔트로피 같은 기법을 넘어 새로운 평가 지표를 테스트했습니다. LeWiDi 3는 데이터셋 공개와 일관성을 보장하기 위해 JSON 형식으로 제공되며, 모든 데이터셋에는 동일한 필드 구조가 포함되어 있습니다. 특히, 각 데이터셋의 주석자들의 나이와 성별 정보도 제공되어 있어, 데이터 분석에 있어 추가적인 통찰력을 제공합니다.

- **Performance Highlights**: LeWiDi 3는 이전 에디션보다 더 작지만, 헌신적인 참여자 그룹이 참여했습니다. 총 53명이 경쟁 플랫폼에 등록하였고, 15팀이 제출물을 제공하여 9개의 시스템 논문이 작성되었습니다. 이러한 결과는 AI에 대한 변동성을 모델링하는 데 있어 방법의 강점과 한계를 잘 드러내주고 있습니다.



### ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping (https://arxiv.org/abs/2510.08457)
- **What's New**: 이 논문에서는 ARES라는 새로운 통합 오픈 소스 프레임워크를 제안합니다. ARES는 작업의 난이도에 따라 탐색 노력을 동적으로 할당함으로써 모드의 과도한 사고를 최소화하고 복잡한 문제에 대한 탐색을 촉진하는 데 중점을 둡니다. 이 연구는 최근의 멀티모달 대규모 추론 모델(MLRM)이 직면한 효율성과 정확성 간의 균형을 맞추기 위해 적응형 추론 메커니즘을 적용하고 있습니다.

- **Technical Details**: ARES는 두 단계로 나누어진 훈련 파이프라인을 도입합니다. 첫 번째 단계인 Adaptive Cold-Start 단계에서는 각 문제의 난이도에 비례하여 추론 길이를 가진 다양한 멀티모달 및 텍스트 데이터를 선별합니다. 두 번째 단계에서는 Adaptive Entropy Policy Optimization(AEPO)을 사용하여 HWE(High Window Entropy) 토큰을 탐색 촉발기로 활용하고 이를 통해 적응형 탐색을 진행합니다.

- **Performance Highlights**: 광범위한 수학적, 논리적, 멀티모달 벤치마크에 대한 실험 결과, ARES는 뛰어난 성능과 추론 효율성을 달성했습니다. 특히, ARES는 상업용 시스템에 비해 현저히 낮은 추론 비용으로 비슷한 성과를 낼 수 있음을 보여주었습니다. 이를 통해 ARES는 난이도에 따라 탐색 노력을 조절함으로써 성능과 효율성을 동시에 향상시키는 가능성을 보여줍니다.



### Single layer tiny Co$^4$ outpaces GPT-2 and GPT-BER (https://arxiv.org/abs/2510.08404)
- **What's New**: 이번 논문에서는 Co$^4$ 언어 모델을 소개하며, 단일 레이어와 두 개의 헤드를 가진 8M 파라미터의 소형 기계가 BabyLM Challenge의 신기준인 GPT-2와 GPT-BERT를 능가하는 성능을 보여줍니다. Co$^4$는 단 2 에포크 동안 훈련했지만, 이전 모델들은 10 에포크 필요했습니다. 이 모델은 10M 토큰 на 대하여 매우 높은 샘플 효율성을 보여줍니다.

- **Technical Details**: Co$^4$ 모델은 두 가지 다른 입력 통합 지점을 가진 뉴런을 기반으로 하며, 이를 통해 문맥적 입력(C)와 피드포워드 입력(FF)을 잘 결합할 수 있습니다. Triadic modulation loops를 활용하여 Q, K, V의 공동 진화를 통해, Co$^4$는 저비용 운영(O(N))으로 효율적인 학습과 뛰어난 상태 추론을 가능하게 합니다. 또한, 이 모델은 10M 토큰에서 최소한의 학습 예산으로 훈련되었습니다.

- **Performance Highlights**: Co$^4$는 여러 언어 모델링 벤치마크에서 뛰어난 성능을 발휘하였습니다. 제로-샷(zero-shot) 환경에서 7개 작업 중 5개에서 GPT-2를 초과하며, 파인 튜닝 작업에서도 7개 중 6개에서 우수한 결과를 기록했습니다. 이러한 성과는 현재의 딥러닝 패러다임과 스케일링 법칙을 재고할 필요성이 있음을 제안합니다.



### If Probable, Then Acceptable? Understanding Conditional Acceptability Judgments in Large Language Models (https://arxiv.org/abs/2510.08388)
Comments:
          22 pages, 12 figures

- **What's New**: 이번 연구는 LLMs(대형 언어 모델)가 조건부 수용성(judgments of conditional acceptability)을 어떻게 평가하는지를 소개합니다. 기존 연구들은 LLM의 인과 관계 추론(causal inference) 및 추론 능력에 중점을 두었지만, 최근 연구는 조건부 확률(conditional probability)과 의미적 관련성(semantic relevance)에 따른 LLM의 수용성 판단을 심층적으로 분석했습니다. 실험 결과, LLM은 확률 및 의미적인 연관성에 민감성을 보이며, 이는 모델의 종류나 구조에 따라 달라집니다.

- **Technical Details**: 조건부 수용성의 개념은 조건문이 특정 맥락에서 얼마나 적절하거나 자연스럽게 인식되는지를 나타냅니다. 연구에서는 Llama 3.1 및 Qwen 2.5 모델 두 종류를 사용하여, 조건적 확률과 의미적 관계의 영향을 분석했습니다. 결과적으로, LLM의 수용성 판단이 조건부 확률과 의미적 관련성 모두에 영향을 받지만, 각 모델의 구조 및 프롬프트 스타일에 따라 이 효과의 강도와 일관성이 상이하게 나타났습니다.

- **Performance Highlights**: LLM의 조건부 수용성 판단은 전반적으로 인간의 판단 트렌드와 유사한 경향을 보이지만, 프롬프트와 모델 크기에 따라 더 큰 변동성을 보였습니다. 또한, 모델 크기가 커져도 인간의 판단과의 일치도가 반드시 증가하지는 않음을 발견했습니다. 예를 들어, Few-shot prompting(소수 샷 프롬프트)은 LLM의 의미적 관련성에 대한 민감도를 향상시킬 수 있지만, 특정 의미 관계에 대한 편향을 초래할 수 있습니다.



### On the Relationship Between the Choice of Representation and In-Context Learning (https://arxiv.org/abs/2510.08372)
Comments:
          25 pages, 6 figures, 10 tables

- **What's New**: 이번 논문은 in-context learning (ICL)에서 나타나는 두 가지 주요 요소, 즉 representation(표현)과 learning(학습) 간의 관계를 분석합니다. 기존 연구에서는 이 두 요소를 별개로 살펴보았으나, 본 연구에서는 이들이 독립적이라는 가설을 세우고 이를 검증하기 위한 최적화 알고리즘을 개발했습니다. 실험을 통해, demonstration(데모)의 종류와 모델의 크기에 따라 ICL의 성능이 어떻게 달라지는지를 조사하며, representation의 질이 ICL의 기본 정확도를 결정한다고 제시합니다.

- **Technical Details**: 연구에서는 label sets(레이블 세트)의 의미적 관련성을 다양하게 조절하여, ICL 성능을 측정할 수 있는 최적화 알고리즘을 개발했습니다. 세 개의 모델 크기를 설정하고, 각각의 레이블 세트에 대해 여러 개의 데모를 사용하는 실험을 실시하였으며, 이로 인해 representation과 learning 간의 관계를 정량적으로 분석하였습니다. 이 분석을 통해 학습 효율성과 정확도 간의 상관관계에 대해 다양한 결과를 도출했습니다.

- **Performance Highlights**: 실험 결과, ICL에서 표현이 학습을 유도하는 역할을 하며, 학습은 보통 representation의 질과 모델 크기와 관계없이 발생함을 확인했습니다. 정확도 순위는 demonstration의 수와 상관없이 초기 순서를 그대로 유지하며, 일정한 범위의 정확도가 주어진 representation에 의해 크게 결정됨을 관찰했습니다. 이 결과는 적절한 레이블 세트를 선택함으로써 ICL 성능을 향상시킬 수 있는 가능성을 보여줍니다.



### Two-Stage Voting for Robust and Efficient Suicide Risk Detection on Social Media (https://arxiv.org/abs/2510.08365)
- **What's New**: 최근 몇 년간 전 세계적으로 자살률이 증가하고 있으며, 이는 효율적인 예방 전략의 필요성을 강조합니다. 많은 위험에 처해 있는 개인이 공식적인 도움을 피하고 소셜 미디어에서 고통을 공유하는 만큼, 이 플랫폼이 제공하는 신호가 중요해졌습니다. 이번 연구는 BERT와 대형 언어 모델(LLM)을 활용하여 미묘한 자살 사상을 탐지하기 위한 새로운 두 단계 투표 아키텍처를 제안합니다.

- **Technical Details**: 제안된 시스템은 두 단계로 구성됩니다. 1단계에서는 경량 BERT 분류기가 명확한 자살 위험 사례를 신속하게 처리하여 67.6%의 입력을 필터링합니다. 2단계에서는 애매한 입력을 다루기 위해 두 가지 경로가 있으며, 이는 LLM 투표 프레임워크 또는 심리적 지표를 기반으로 한 ML 앙상블을 통해 이루어집니다.

- **Performance Highlights**: 이 프레임워크는 명시적인 데이터셋과 암시적인 데이터셋 모두에서 벤치마크 성능을 초과하는 결과를 달성하였습니다. 명시적 사례에서 98.0% F1 점수, 암시적 사례에서 99.7% F1 점수를 기록하며, 크로스 도메인 격차를 2% 이하로 줄였습니다. 또한, LLM 비용을 크게 줄이며 시스템의 효율성을 높였습니다.



### AutoRed: A Free-form Adversarial Prompt Generation Framework for Automated Red Teaming (https://arxiv.org/abs/2510.08329)
- **What's New**: AutoRed는 기존의 seed instruction에 의존하지 않고, 자유 형식의 적대적 프롬프트를 생성하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 퍼소나(qualities) 정보를 활용하여 다양한 적대적 명령어를 생성하는 두 단계로 구성됩니다. 또한, 자동으로 생산되는 프롬프트의 유해성을 평가하기 위한 검증기를 도입하여 효율성을 높였습니다.

- **Technical Details**: AutoRed의 첫 단계는 타겟 모델을 겨냥한 적대적 지시사항 생성으로, 수동적인 seed instructions 없이 퍼소나 정보를 활용하여 그 과정에서 생성된 여러 프롬프트를 보유하게 됩니다. 두 번째 단계인 반영 및 개선(reflection and refinement)에서는 초기 생성된 명령어의 효과성을 높이기 위해 반영 루프를 통해 품질을 개선합니다.

- **Performance Highlights**: AutoRed는 평가한 여덟 개의 최신 대형 언어 모델에서 높은 공격 성공률을 달성하며, 기존의 seed 기반 방법에 비해 우수한 성능을 보입니다. 추가 분석을 통해 AutoRed의 생성된 명령어가 높고 다양한 복잡성을 가지며, 따라서 LLM이 유해한 응답을 생성할 가능성이 높았음을 확인하였습니다.



### Neuron-Level Analysis of Cultural Understanding in Large Language Models (https://arxiv.org/abs/2510.08284)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 문화적 이해를 향상시키기 위한 연구를 수행하고, 신경 세포 수준의 분석을 통해 문화 행동을 유도하는 신경 세포를 식별하는 방법을 제안합니다. 이 연구는 문화 일반 신경 세포와 문화 특정 신경 세포를 구분하여, 각 신경 세포가 어떻게 문화적 지식에 기여하는지를 밝히고자 합니다. 그 결과, 문화 특정 신경 세포는 문화 이해에 중요한 역할을 하며, LLM의 문화적 성과에 편차를 줄 수 있음을 확인했습니다.

- **Technical Details**: 연구에서는 CULNIG(CULture Neuron Identification Pipeline with Gradient-based Scoring)이라는 방법을 통해 신경 세포를 정밀하게 식별합니다. 이 벤치마크에서는 gradient 기반의 기여 점수를 활용하며, 문화적 관련성이 낮은 신경 세포를 제거하기 위해 제어 데이터 세트를 사용합니다. 발견된 신경 세포는 MLP(다층 퍼셉트론) 구조의 얕은 층에서 중간 층에 집중되어 있으며, 전체 신경 세포의 1%도 되지 않습니다.

- **Performance Highlights**: 신경 세포를 마스킹할 경우, LLM의 문화적 이해력이 상당히 저하되며, 문화 벤치마크에서 최대 30% 성과 하락이 발생합니다. 반면, 일반적인 자연어 이해(NLU) 벤치마크의 성과에는 큰 영향을 미치지 않습니다. 또한, NLU 데이터 세트로 모델을 미세 조정할 경우, 문화 일반 신경 세포를 다수 포함한 모듈 업데이트가 문화적 이해 손실을 초래할 수 있다는 점을 강조하고 있습니다.



### Beyond Turn Limits: Training Deep Search Agents with Dynamic Context Window (https://arxiv.org/abs/2510.08276)
- **What's New**: 최근 강화 학습의 발전으로 심층 사고 능력을 멀티 턴 에이전트에 적용하는 데 어려움이 있었습니다. 본 논문에서는 고난이도 훈련 과제와 동적 컨텍스트 관리 전략을 도입하여 이러한 문제를 해결한 DeepMiner라는 새로운 프레임워크를 제안합니다. DeepMiner는 신뢰할 수 있는 웹 소스에서 복잡한 질문-답변 쌍을 생성하는 것을 통해 체계적인 사고 능력을 기를 수 있는 기회를 제공합니다.

- **Technical Details**: DeepMiner는 고난이도 훈련 작업을 생성하기 위한 리버스 콘스트럭션(reverse construction) 방법을 활용하며, 동적인 슬라이딩 윈도우(sliding window) 메커니즘을 통해 컨텍스트를 관리합니다. 이를 통해 외부 요약 모델에 대한 의존성을 줄이고, 장기적인 컨텍스트를 효과적으로 처리할 수 있는 모델을 최적화합니다. 딥마이너는 Qwen3-32B에 구현되어 다양한 벤치마크에서 성능을 평가하였습니다.

- **Performance Highlights**: DeepMiner-32B는 BrowseComp-en에서 33.5%의 정확도를 기록하며 이전의 최고 오픈 소스 에이전트를 약 20%포인트 초과했습니다. BrowseComp-zh, XBench-DeepSearch, 그리고 GAIA에서도 일관되게 성능 개선을 보였습니다. 이러한 결과는 고품질 훈련 데이터와 효과적인 컨텍스트 관리가 장기적 멀티 턴 상호작용에서 깊은 사고 능력을 개발하는 데 기여할 수 있음을 보여줍니다.



### Contrastive Decoding for Synthetic Data Generation in Low-Resource Language Modeling (https://arxiv.org/abs/2510.08245)
Comments:
          13 pages, 3 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 교육에 있어 제한된 데이터 문제를 해결하기 위해 인공지능에 의해 생성된 합성 데이터(synthetic data)를 사용합니다. 특히, Contrastive Decoding(𝖢𝖣	ext{CD})을 활용하여 모델 성능을 향상시키는 것을 목표로 합니다. 원본 말뭉치에서 좋은 모델과 나쁜 모델 간의 상대적 차이를 이용하여 합성 말뭉치를 생성하고, 이를 실제 데이터에 혼합하여 훈련하는 방식입니다.

- **Technical Details**: 연구에서는 100M 단어의 원본 말뭉치에서 훈련된 좋은 모델과 나쁜 모델의 상대적 차이를 기반으로 100M 단어의 합성 말뭉치를 생성했고, 이를 원래 훈련 데이터와 결합하여 모델을 훈련했습니다. Contrastive Decoding(𝖢𝖣	ext{CD})은 더 우수한 모델에서 나오는 신호를 확대하여 더 일관되고 유익한 텍스트를 생성하는 데 초점을 맞춥니다. 이 방법이 다른 전통적인 샘플링 방식에 비해 훈련 과정에서 어떻게 기여하는지 분석했습니다.

- **Performance Highlights**: 합성 데이터로 훈련한 모델은 언어 모델링 목표 및 여러 다운스트림 작업에서 성능이 향상되었습니다. 특히, 𝖢𝖣	ext{CD}에서 생성된 데이터를 사용한 훈련이 더 많은 추론 능력이 요구되는 작업에 강력한 효과를 보이며, 전통적인 샘플링을 통해 생성된 데이터는 표면적인 언어적 능력에 의존하는 작업에서 더 유리한 결과를 가져왔습니다.



### The Alignment Waltz: Jointly Training Agents to Collaborate for Safety (https://arxiv.org/abs/2510.08240)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 안전성과 유용성을 동시에 향상시키기 위해 새로운 접근 방식을 제안합니다. 연구자들은 WaltzRL이라는 다중 에이전트 강화 학습 프레임워크를 제안하여, 대화 에이전트와 피드백 에이전트를 함께 훈련시키고 이들이 협력하여 서로의 성능을 향상시키도록 합니다. 이 시스템은 Dynamic Improvement Reward(DIR)를 통해 시간에 따라 피드백을 통합하고, 안전성 및 유용성이 동시에 향상될 수 있도록 합니다.

- **Technical Details**: WaltzRL는 다중 에이전트 강화 학습을 기반으로 한 새로운 안전 정렬 방법론입니다. 대화 에이전트는 사용자로부터 받은 프롬프트에 대해 응답을 생성하고, 피드백 에이전트는 그 응답에 대해 개선을 위한 피드백을 제공합니다. 이러한 협력 구조는 포지티브-섬 게임으로 모델링되며, 각 에이전트는 독립적으로 보상을 받습니다. 특히 피드백 에이전트에 대한 DIR는 대화 에이전트가 피드백을 얼마나 잘 통합했는지를 기준으로 변화합니다.

- **Performance Highlights**: WaltzRL을 통해 수행된 여러 실험은 기존 모델에 비해 안전하지 않은 응답 비율을 39.0%에서 4.6%로 감소시키고, 오버리퓨설(overrefusal) 비율을 45.3%에서 9.9%로 줄이는 성과를 보여줍니다. 또한, 대화 에이전트는 FS(Feedback Score)가 높을수록 더 나은 응답을 생성하여, 보다 안전하고 유용한 대화가 가능하다는 점에서 WaltzRL의 효과성을 입증합니다. 이 연구는 LLM의 안전성을 향상시키면서도 일반적인 능력은 감소시키지 않는 방법론적인 기여를 합니다.



### Investigating Counterclaims in Causality Extraction from Tex (https://arxiv.org/abs/2510.08224)
- **What's New**: 본 연구는 텍스트 내 인과 관계 추출에서 반대 주장(concausal claims)을 간과한 점을 지적하고 새로운 데이터셋인 Concausal News Corpus를 개발하여 이 문제를 해결하고자 합니다. 기존 데이터셋은 오로지 인과 관계를 지지하는 주장만을 포함하고, 반대 주장은 무시되거나 오인 Annotation되었습니다. 새로운 데이터셋은 인과 관계의 복잡성을 개선하고, 모델이 procausal과 concausal 관계를 효과적으로 구별할 수 있도록 돕습니다.

- **Technical Details**: 인과 관계 추출(causality extraction) 과정은 크게 세 단계로 나눌 수 있으며, 이 연구에서는 이 과정을 procausal, concausal, uncausal 관계로 명확히 구분하는 작업을 수행하였습니다. 새로운 데이터 세트에서 코헨의 카파 통계량은 0.74에 달하는 높은 상호 주석자 동의율을 나타내어, 명확한 기준을 바탕으로 반대 주장을 효과적으로 포함시켰습니다. 또한 연구 결과는 transformer 기반의 신경망이 이러한 주장을 효과적으로 구별할 수 있음을 보여줍니다.

- **Performance Highlights**: 본 연구의 결과는 기존 인과 관계 추출 모델이 concausal 주장을 잘 인식하지 못하고, 이를 procausal로 잘못 분류하는 경향이 있음을 강조합니다. 데이터셋 개선을 통해 이러한 문제를 해결할 수 있으며, 이는 다양한 실제 응용, 특히 결정 지원 시스템에서 인과적 이해를 높이는 데 기여할 것입니다. 궁극적으로, 반대 주장을 포함하는 것은 사용자에게 균형 잡힌 시각을 제공하며 인과 관계에 대한 인식을 제고할 수 있게 합니다.



### SenWave: A Fine-Grained Multi-Language Sentiment Analysis Dataset Sourced from COVID-19 Tweets (https://arxiv.org/abs/2510.08214)
Comments:
          13 pages, 13 figures, 6 tables

- **What's New**: COVID-19 팬데믹의 전 세계적 영향을 통해 공공의 정서와 반응을 포괄적으로 이해할 필요성이 강조되었습니다. 데이터의 부족 및 부적절한 레이블 문제를 해결하기 위해, 본 논문에서는 COVID-19 트윗 분석을 위한 새로운 다국어 정서 분석 데이터셋인 SenWave를 도입했습니다. 이 데이터셋은 5개 언어에서 10개 정서 범주를 포함하여 구성되었습니다.

- **Technical Details**: SenWave 데이터셋은 영어 및 아랍어에서 각각 10,000개의 주석 달린 트윗과 스페인어, 프랑스어, 이탈리아어로 번역된 30,000개의 트윗을 포함합니다. 이를 통해 다양한 언어와 주제를 기반으로 세분화된 정서 분류를 수행했습니다. 우리는 사전 훈련된 트랜스포머 기반 언어 모델을 사용하여 데이터셋을 세밀하게 조정하였으며, 데이터의 품질 평가 또한 수행했습니다.

- **Performance Highlights**: SenWave는 COVID-19 팬데믹 확산 중 국민의 정서 변화를 분석하기 위한 독특한 자원으로, 다양한 정서 분석 작업에 유용할 것입니다. ChatGPT를 이용한 평가에서도 이 데이터셋의 신뢰성을 입증하였습니다. 이 연구는 NLP 커뮤니티에서 세분화된 정서 분석 연구를 촉진하여 복잡한 이벤트에 대한 더욱 깊이 있는 이해를 가능하게 할 것으로 기대됩니다.



### LLMs Learn to Deceive Unintentionally: Emergent Misalignment in Dishonesty from Misaligned Samples to Biased Human-AI Interactions (https://arxiv.org/abs/2510.08211)
- **What's New**: 이번 연구에서는 LLMs가 특정 도메인에서 악의적인 또는 잘못된 완성을 파인 튜닝(fine-tuning) 받을 경우 발생하는 'emergent misalignment' 현상을 탐구합니다. 이 연구는 LLM의 안전성을 넘어 정직하거나 기만적인 행동까지 영향을 미치는지 조사합니다. 연구 결과, LLM이 좁은 범위의 잘못된 정보로 훈련될 경우, 보다 폭넓은 불일치 행동을 보일 수 있음을 보여줍니다.

- **Technical Details**: 연구팀은 다양한 도메인에서 LLM을 misaligned completions로 파인 튜닝하여 정직성 상실 여부를 평가했습니다. 실험 결과, 1%의 misalignment 데이터를 표준 다운스트림 작업에 통합하는 것만으로도 LLM의 정직성이 20% 이상 감소한다는 것을 발견했습니다. 또한, 인간-AI 상호작용 환경에서 사용자 집단이 편향될 경우 LLM의 정직성이 저하될 수 있음을 확인했습니다.

- **Performance Highlights**: 모델이 고위험 시나리오에서 어떻게 불리한 행동을 하는지를 평가하는 연구 결과, LLM이 압박을 받을 때 자신의 신념에 모순되는 말을 할 수 있음을 보여줍니다. 또한, DeceptionBench를 통해 모델의 대답과 그 속의 논리(Cot reasoning) 간의 불일치를 평가하며, 이러한 불일치가 기만적인 행동을 강화할 수 있음을 강조합니다.



### Memory Retrieval and Consolidation in Large Language Models through Function Tokens (https://arxiv.org/abs/2510.08203)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 기억 메커니즘을 다루고 있습니다. 특히, 메모리 검색(memory retrieval)과 통합(consolidation)의 과정을 설명하기 위해 함수 토큰(function tokens) 가설을 제안합니다. 함수 토큰이 문맥에서 가장 예측력 있는 특징을 활성화하고 다음 토큰 예측을 주도한다는 점이 주목할 만합니다.

- **Technical Details**: 이론적으로, 논문은 Transformer 구조를 가진 GPT 유형의 모델을 사용하여 사전 훈련과 후 훈련을 진행합니다. 메모리는 LLM의 파라미터와 그로부터 파생될 수 있는 모든 특징으로 구성되어 있으며, 함수 토큰은 이러한 특징과 회로를 접근하고 활성화하는 역할을 합니다. 특별히, 함수 토큰은 내용 토큰(content tokens) 다음에 오는 예측을 통해 파라미터를 업데이트합니다.

- **Performance Highlights**: 연구 결과에 따르면, 함수 토큰은 문맥에서 대다수의 기능을 활성화하는 데 중요한 역할을 합니다. 또한, 사전 훈련 과정에서 함수 토큰에 이어지는 내용 토큰 예측의 손실이 주얼하여 통합이 이루어지며, 이는 LLM의 성능 향상으로 이어집니다. 이러한 발견은 LLM의 해석 가능성을 높이고 인간 가치에 대한 정렬을 강화하는 고급 학습 알고리즘 설계에 기여할 수 있습니다.



### Training-Free Group Relative Policy Optimization (https://arxiv.org/abs/2510.08191)
- **What's New**: 이번 연구에서는 파라미터 업데이트 없이 LLM(대형 언어 모델) 에이전트의 성능을 향상시키는 Training-Free Group Relative Policy Optimization(훈련 없는 그룹 상대 정책 최적화, Training-Free GRPO)을 제안합니다. 저자들은 기존의 비싼 파라미터 조정 방식 대신, 경험적 지식을 이용한 경량화 접근 방식이 가능하다고 주장합니다. 이를 통해, 데이터 불균형 문제를 해결하고 과적합(overfitting)을 피할 수 있습니다.

- **Technical Details**: Training-Free GRPO는 기존 GRPO의 그룹 상대 평가 과정을 비파라메트릭(non-parametric)으로 변환합니다. 이 방법은 LLM이 출력 배포를 조정하는 대신, 제한된 샘플 세트를 통해 경험 지식을 캡슐화하여 성능을 향상시키는 데 중점을 둡니다. 학습 단계마다 그룹 롤아웃을 생성하고, 각 그룹에서 시맨틱(group advantage)을 도출하여 정책 출력을 최적화합니다. 이는 LLM이 행동을 인도하는 데 사용됩니다.

- **Performance Highlights**: 수학적 추론과 웹 검색 과제에서 Training-Free GRPO를 적용한 DeepSeek-V3.1-Terminus 모델이 비 도메인 성능이 크게 개선되었습니다. 훈련 샘플 수가 적음에도 불구하고, Training-Free GRPO는 조정된 소형 LLM보다 우수한 성능을 보여주며, 전통적인 미세 조정 기법에 비해 더욱 경제적이고 효율적인 대안을 제공합니다. 이는 다양한 도메인에 걸쳐 적용 가능성을 높입니다.



### METRICALARGS: A Taxonomy for Studying Metrical Poetry with LLMs (https://arxiv.org/abs/2510.08188)
Comments:
          Pre-print

- **What's New**: 이번 논문에서는 MetricalARGS라는 새로운 범주를 소개합니다. 이는 시(诗) 관련 NLP(자연어처리) 작업의 최초의 분류 체계로, LLMs(대형 언어 모델)의 운율적(metrical) 시에서의 성능을 평가하기 위해 설계되었습니다. MetricalARGS는 분석(Analysis), 검색(Retrieval), 생성(Generation), 지원(Support)의 네 가지 차원에서 LLMs를 평가하는 기회를 제공합니다.

- **Technical Details**: MetricalARGS는 기능적 특징을 분석하고 관련 데이터셋과 평가 기준을 다루는 데 중점을 둡니다. 이 연구는 기존의 NLP 작업과의 연결성을 논의하며, 운율적 시에서 어떻게 이러한 과제가 수행될 수 있는지를 설명합니다. 특히, 텔루구어(Telugu)를 예시 언어로 사용하여 실질적인 응용을 보여줍니다.

- **Performance Highlights**: MetricalARGS는 현대 LLM의 능력과 한계를 운율적 시의 관점에서 이해할 수 있는 폭넓은 가능성을 강조합니다. 이러한 접근방식은 LLM의 심층적 사고(deep reasoning)와 언어 이해(language understanding) 능력을 시험하는 데 중요한 기초를 제공합니다. 이 연구는 향후 LLM 발전의 방향성을 제시하고, 시 생성 및 분석 분야에서 새로운 연구 기회를 마련합니다.



### ARM2: Adaptive Reasoning Model with Vision Understanding and Executable Cod (https://arxiv.org/abs/2510.08163)
Comments:
          Work in Progress

- **What's New**: 본 논문에서는 ARM2라는 통합 모델을 제안하여 여러 포맷에서의 추론 성능과 효율성을 적응적으로 조정합니다. ARM2는 비주얼 이해(vision understanding)와 실행 가능한 코드(executable code)를 통합하여 복잡한 추론 작업에서의 정확성을 높이고 긴 CoT의 대안으로서의 효율성을 제공합니다. 이 모델은 강화 학습(reinforcement learning)과 길이 인식 최적화(length-aware optimization)를 통해 다양한 작업에 적합한 추론 형식을 선택하도록 설계되었습니다.

- **Technical Details**: ARM2는 다섯 가지 추론 형식인 Direct Answer, Short CoT, Code-Text, Code-Exec 및 Long CoT를 지원합니다. ARM2는 고품질의 다중 모달(multimodal) 데이터셋을 사용하여 훈련되며, SFT(Supervised Fine-tuning)를 통해 다양한 추론 형식을 사용하는 방법을 학습합니다. 이후에는 길이 인식 강화 학습을 통해 가장 적합한 추론 형식을 선택하도록 하여 성능과 토큰 효율성을 모두 고려합니다.

- **Performance Highlights**: 실험 결과 ARM2는 전통적인 GRPO로 훈련된 모델과 동등한 성능을 보이면서 평균적으로 70% 이상의 토큰 사용량 감소를 달성했습니다. 이로 인해 ARM2는 단순하면서 직관적인 작업에서 더 높은 효율성을 보여주며, 복잡한 추론 작업에서도 경쟁력 있는 성능을 유지합니다. 또한, ARM2는 실행 가능한 코드를 서포트하여 복잡한 논리 구조를 처리하고 외부 도구와 상호작용할 수 있는 능력을 갖추고 있습니다.



### Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs (https://arxiv.org/abs/2510.08158)
- **What's New**: 이 논문에서는 LLM에서 발생하는 과도한 안전 거부(exaggerated safety refusals) 문제를 해결하기 위해 두 가지 새로운 벤치마크인 Exaggerated Safety Benchmark (XSB)와 Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB)를 제안합니다. XSB는 단일 턴 프롬프트에 대한 벤치마크로, 'Focus' 키워드로 거부 유발 트리거를 확인합니다. MS-XSB는 다중 턴의 맥락이 풍부한 교류에서 모델 행동을 평가합니다. 이 두 벤치마크는 실제 상황에서 자주 발생하는 잘못된 요청을 더 폭넓게 평가할 수 있도록 설계되었습니다.

- **Technical Details**: XSB 벤치마크는 12가지 프롬프트 유형으로 구성되며, 총 340개의 안전한 프롬프트와 240개의 위험한 프롬프트가 포함되어 있습니다. MS-XSB에는 30개의 시나리오가 포함되며 각 시나리오에는 20개의 독립적인 프롬프트가 있습니다. 연구진은 DeepSeek-R1, Llama-3.3, Qwen2-VL, DeepSeek-Coder와 같은 LLM에 대한 철저한 평가를 통해 서로 다른 도메인 전문성을 가진 LLM들이 과도한 안전 거부를 어떻게 나타내는지 분석했습니다. 실험 결과, 과도한 안전 행동이 최근 LLM에서도 여전히 광범위하게 발생하고 있음을 확인했습니다.

- **Performance Highlights**: 실험에서는 세 가지 경량화된 사후 개입 전략인 ignore-word instructions, prompt rephrasing, attention steering을 사용하여 안전한 프롬프트에 대한 컴플라이언스(compliance)를 크게 개선할 수 있음을 보여주었습니다. 하지만 이러한 방법이 안전한 보호를 약화시킬 수도 있어, 안전성과 도움을 균형 있게 유지하는 중요성을 강조합니다. 특히 이 연구는 LLM의 과도한 안전 거부를 진단하고 완화할 수 있는 재현 가능한 프레임워크를 제안하여, 보다 안전하고 유용한 LLM 배포의 경로를 제시합니다.



### DACIP-RC: Domain Adaptive Continual Instruction Pre-Training via Reading Comprehension on Business Conversations (https://arxiv.org/abs/2510.08152)
Comments:
          Accepted to the EMNLP 2025 Industry Track. Equal contribution from the first four authors

- **What's New**: 이 논문에서는 Domain Adaptive Continual Instruction Pre-Training via Reading Comprehension (DACIP-RC)이라는 새로운 지속적 사전 훈련 기법을 제안합니다. 이 방법은 기존의 다음 토큰 예측(next-token prediction)에 의존하는 사전 훈련 접근법과 달리, 대화 기록을 기반으로 다양한 작업 지침과 응답을 생성합니다. DACIP-RC는 작은 규모의 LLM이 비즈니스 대화 작업에서 더 나은 지침 일반화(instruction generalization)를 할 수 있도록 돕습니다.

- **Technical Details**: 본 연구는 DACIP-RC 방법론을 통해, 실제 대화 녹취록을 활용하여 비즈니스 관련 다양한 작업에 적응할 수 있도록 하는 지속적 사전 훈련 데이터를 구축하는 과정을 설명합니다. 데이터 샘플링 및 지침 작성 방법을 제시하며, ASR(Automatic Speech Recognition) 시스템으로 전사된 비즈니스 대화의 품질과 다양성을 보장하기 위해 120초 이상의 대화 데이터만을 사용합니다. 이후 개인 식별 정보 제거 및 데이터 형식 다각화를 진행하여 대화의 정밀도를 높입니다.

- **Performance Highlights**: DACIP-RC의 실증적 평가는 다양한 비즈니스 대화 작업에서 제로샷(zero-shot) 일반화를 크게 향상시킴을 보여줍니다. 특히, 회의 요약(meeting summarization), 행동 항목 생성(action item generation), 통화 목적 식별(call purpose identification)과 같은 작업에서의 성능을 입증하였고, 이는 기업들이 자체 데이터셋을 활용하여 도메인 적응(domain adaptation)을 효과적으로 진행할 수 있는 통찰을 제공합니다.



### AI Knowledge Assist: An Automated Approach for the Creation of Knowledge Bases for Conversational AI Agents (https://arxiv.org/abs/2510.08149)
Comments:
          Accepted to the EMNLP 2025 Industry Track

- **What's New**: 본 논문에서는 고객 문제 해결을 위한 회화형 AI 시스템에서 Retrieval Augmented Generation (RAG) 기술의 활용 증가에 대해 다루고 있습니다. 이를 해결하기 위해 과거 고객-에이전트 대화에서 QA 쌍을 자동으로 추출하여 지식 기반을 구축하는 AI Knowledge Assist 시스템을 소개합니다. 이 시스템은 LLaMA-3.1-8B 모델을 기반으로 하여 20개 회사의 실증 평가를 통해 90% 이상의 정확도로 정보 요청 질문에 답변할 수 있음을 보여줍니다.

- **Technical Details**: AI Knowledge Assist 시스템은 세 단계의 파이프라인을 통해 작동합니다. 첫 번째 단계에서는 과거의 통화 원고에서 정보 요청 질문과 에이전트의 응답을 추출합니다. 두 번째 단계에서는 이러한 QA 쌍을 의미상 유사한 그룹으로 클러스터링하여 중복된 QA 쌍을 관리합니다. 마지막으로, LLM을 활용하여 각 클러스터에서 정보를 가장 잘 요약한 대표 QA 쌍을 선택해 지식 기반에 삽입하거나 관리자에게 추천합니다.

- **Performance Highlights**: 이번 연구는 20개 클라이언트 회사의 실제 데이터에서 실험을 수행하였으며, AI Knowledge Assist 시스템은 콜센터 AI 챗봇의 기능을 대폭 향상시킴을 입증했습니다. 이 시스템을 통해 고객의 문의를 효과적으로 처리할 수 있으며, 고객 만족도를 증대시키는 데 기여할 수 있음을 나타냅니다. 실제 데이터 수집에 있어 고객 데이터의 개인 정보 보호를 우선시하여 미세한 주의가 필요함을 강조합니다.



### Mitigating Judgment Preference Bias in Large Language Models through Group-Based Polling (https://arxiv.org/abs/2510.08145)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)을 자동 평가기로 활용하는 새로운 방법론인 Genii(그룹 기반 여론 최적화)를 제안합니다. Genii는 여러 LLM 기반 모델들이 상호작용하며 내재된 판단 편향을 완화하려고 합니다. 기존의 감독 학습 모델과 달리, Genii는 인공지능 모델들 사이의 협업을 통해 비지도 학습으로 성능을 개선합니다.

- **Technical Details**: Genii 프레임워크는 다양한 LLM 기반 판단 모델들을 멀티 에이전트 시스템으로 통합합니다. 이 시스템은 클라이언트-서버 여론 수집 메커니즘을 시뮬레이션하여 각 클라이언트 에이전트를 최적화합니다. 각 에이전트는 주기적으로 클라이언트 역할을 수행하여 그룹의 일관성 점수를 수집하고 이를 통해 개인적 편향을 줄이는 방식으로 구성되어 있습니다.

- **Performance Highlights**: 실험 결과, Genii는 주석이 달린 데이터로 훈련된 감독 모델들을 능가하며, 인간 레이블이 없는 상태에서도 개선된 성능을 보입니다. 또한, 클라이언트 에이전트의 종류에 상관없이 모델 성능이 지속적으로 향상되는 것을 확인했습니다. Genii는 더 나아가 LLM 기반 판단 모델의 잘못된 응답에 대한 선호 편향을 효과적으로 완화하여 더 신뢰할 수 있는 판단을 제공합니다.



### Interpreting LLM-as-a-Judge Policies via Verifiable Global Explanations (https://arxiv.org/abs/2510.08120)
Comments:
          12 pages, 2 figures, 3 tables

- **What's New**: 이 논문은 LLM-as-a-judge를 사용할 때의 잠재적 편향과 위험을 이해하기 위해 새로운 접근 방식을 제안합니다. 이 접근법은 CLoVE(Contrastive Local Verifiable Explanations)와 GloVE(Global Verifiable Explanations)라는 두 가지 알고리즘으로 구성됩니다. CLoVE는 개념 기반의 설명을 생성하고, GloVE는 이러한 설명을 글로벌 정책으로 요약합니다.

- **Technical Details**: CLoVE는 LLM-as-a-judge의 개별 결정을 설명하는 로컬 설명 알고리즘입니다. 이 알고리즘은 BECAUSE-DESPITE 형식을 통해 다각적인 관점에서 설명을 제공합니다. GloVE는 이러한 로컬 설명을 집계하여 고수준의 규칙 기반 글로벌 정책으로 요약하며, 이 과정에서 대칭을 유지합니다.

- **Performance Highlights**: 본 연구는 GloVE의 성능을 7개의 표준 평가 데이터셋을 사용해 평가했습니다. 실험 결과, GloVE는 높은 충실도를 보이며, 사용자 연구를 통해 사용자 이해도가 증가하고 있다는 점을 확인했습니다. 이 연구의 주요 기여는 CLoVE와 GloVE를 통해 LLM-as-a-judge의 투명성과 해석 가능성을 향상시키는 것입니다.



### Evaluating LLM-Generated Legal Explanations for Regulatory Compliance in Social Media Influencer Marketing (https://arxiv.org/abs/2510.08111)
Comments:
          Accepted for publication at the Natural Legal Language Processing Workshop (NLLP) 2025, co-located with EMNLP

- **What's New**: 이번 연구는 인플루언서 마케팅의 투명성을 확보하기 위한 기술적 접근을 다루고 있습니다. 연구는 1,143개의 인스타그램 포스트를 분석하여 GPT 모델들이 광고 포스트를 어떻게 감지하는지를 비교합니다. 특히, 규제 법률에 기반한 설명을 통해 감지 정확도를 높이는 방법을 제시하고 있습니다.

- **Technical Details**: 연구에서는 LLM (Large Language Models)이 주어진 법적 지식을 통해 광고 콘텐츠를 분류하는 방식을 탐구합니다. 이 모델들은 0.93의 F1 점수로 광고 및 비광고 콘텐츠를 강력하게 분류했지만, 애매한 경우에는 정확도가 10포인트 이상 저하되었습니다. 법적 설명 생성에는 흔히 발생하는 오류의 분류 체계도 개발하여 LLM의 법적 논리 생성 평가를 한층 강화했습니다.

- **Performance Highlights**: 모델의 성능은 특히 명확한 레퍼런스와 법적 텍스트를 추가함으로써 향상되었습니다. 하지만 모든 경우에 감지 정확도를 높이지는 못했습니다. 연구는 광고 규제 당국이 기계적 조정 프로세스를 법적 근거에 기반하여 자동화하는 데에 기여할 수 있는 가능성을 제시합니다.



### Lossless Vocabulary Reduction for Auto-Regressive Language Models (https://arxiv.org/abs/2510.08102)
- **What's New**: 이번 논문에서는 자동 회귀 (auto-regressive) 언어 모델의 어휘 (vocabulary)를 손실 없이 줄이는 이론적 프레임워크를 제시합니다. 이는 언어 모델들이 각기 다른 어휘를 가질 때 발생하는 비효율성을 해결하는 데 초점을 맞추고 있습니다. 새로운 접근 방식은 다양한 모델들이 상호 협력할 수 있도록 돕는 최대 공통 어휘 (maximal common vocabulary)를 활용합니다.

- **Technical Details**: 논문에서 제시된 프레임워크는 주어진 자동 회귀 언어 모델의 어휘를 임의의 크기로 축소할 수 있도록 합니다. 이는 각 모델의 다음 토큰 분포 (next-token distribution)에 미치는 영향을 최소화하며, 결과적으로 텍스트 생성의 효율성을 증가시킵니다. 이론적 기반 위에서, 손실 없는 (lossless) 어휘 축소 방법론이 제시되고, 이를 통해 다양한 토크나이제이션 (tokenization)을 가진 모델들이 협력할 수 있는 가능성을 탐구합니다.

- **Performance Highlights**: 이 연구는 서로 다른 언어 모델들이 하나의 공통된 어휘를 통해 효과적으로 협력할 수 있음을 실험적으로 입증했습니다. 향상된 협력 구조 덕분에 모델 앙상블 (ensemble) 성능이 크게 향상되었습니다. 따라서, 언어 모델의 협력적 활용 방안이 확장되어, 다양한 분야에서의 응용 가능성을 보여줍니다.



### The Price of Thought: A Multilingual Analysis of Reasoning, Performance, and Cost of Negotiation in Large Language Models (https://arxiv.org/abs/2510.08098)
- **What's New**: 이 논문은 AI 에이전트의 협상 능력에 대한 포괄적인 연구를 통해 LLM(대형 언어 모델)의 이유 작용(reasoning)이 협상 결과에 미치는 영향을 체계적으로 평가합니다. 연구는 상업적 모델과 공개 가중치(open-weight) 모델 모두를 대상으로 하며, 영어, 독일어, 이탈리아어의 세 가지 언어에서 수행됩니다. 협상 전략에 미치는 이유 작용의 효과를 확인하여 언어의 일관성을 유지하는지와 같은 여러 중요한 질문을 탐구합니다.

- **Technical Details**: 이 연구는 자율 에이전트가 협상 능력을 평가하는 것이 필수적임을 강조합니다. 두 개의 에이전트가 의사소통을 통해 상호 작용하는 각 대화 게임에 대해 LLM의 이유 작용이 협상 효율성에 미치는 영향을 분석합니다. 경기 상황은 게임 마스터를 통해 엄격히 조정되고, 개인별로 다른 가치가 있는 항목에 대해 협상하게 됩니다.

- **Performance Highlights**: 결과에 따르면 이유 작용을 사용하는 것이 협상 결과를 31.4% 향상시키지만, 그에 따른 계산 비용은 400% 가까이 증가하는 것으로 나타났습니다. 놀랍게도, 공개 가중치 모델은 독일어 또는 이탈리아어로 협상하는 동안 내부 이유 작용 단계를 영어로 전환하는 경향이 있으며, 이 점은 이해 가능성에 영향을 미칠 수 있음을 시사합니다. 반면 상업적 모델은 최종 출력과 이유 작용 간의 언어 일관성을 유지합니다.



### Everything is Plausible: Investigating the Impact of LLM Rationales on Human Notions of Plausibility (https://arxiv.org/abs/2510.08091)
Comments:
          pre-print

- **What's New**: 본 연구에서는 LLM(대형 언어 모델)에 의해 생성된 논거가 다중 선택 구성 요소의 정황 접근 방식에서 인간의 타당성 판단에 미치는 영향을 조사합니다. 3,000개의 인간 타당성 판단과 13,600개의 LLM 기반 판단을 수집하여, LLM이 생성한 근거가 이러한 판단에 상당한 영향을 미친다는 것을 발견했습니다. 이는 LLM이 인간의 인지를 연구하는 데 새롭고 효과적인 방법이 될 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 첫 번째로, 다중 선택 기준과 관련한 질문-답변 쌍의 타당성을 평가하기 위해 LLM로부터 생성된 PRO(긍정적) 및 CON(부정적) 논거를 사용합니다. 실험을 통해, 타당성 근거의 추가가 인간과 LLM의 판단에 대하여 유의미한 변화를 가져오는지 검사하였으며, 이 과정에서 1717개의 LLM이 추가적인 판단을 생성해냈습니다.

- **Performance Highlights**: 실험 결과, PRO 논거는 인간과 LLM의 타당성 평가 점수를 평균적으로 증가시키는 반면, CON 논거는 이를 감소시킨다는 것을 발견했습니다. 흥미롭게도, 특정 조건에서는 인간의 골드 답변에 대한 타당성 평가가 PRO 논거에 의해 감소하는 현상이 관찰되었습니다. 또한 타당성 평가의 초기 기준이 후속 평가의 변화에 강한 앵커링 효과를 미치는 것으로 나타났습니다.



### FedDTRE: Federated Dialogue Generation Models Powered by Trustworthiness Evaluation (https://arxiv.org/abs/2510.08058)
- **What's New**: FedDTRE(Federated adaptive aggregation strategy for Dialogue generation based on Trustworthiness Evaluation)를 제안하여 개인 정보 보호와 모델 성능 간의 균형을 맞추고 있습니다. 이 새로운 전략은 지역 업데이트 동안 글로벌 모델의 기여도를 조정하여 대화 모델의 성능을 향상시킵니다. 기존의 연합 학습 방식에서 발생하는 오버피팅(overfitting) 문제를 해결하고, 글로벌 정보를 잊지 않도록 돕는 메커니즘이 포함되어 있습니다.

- **Technical Details**: FedDTRE는 대화 생성 모델을 위한 연합적 업데이트 전략으로, 신뢰도 평가(trustworthiness evaluation)를 기반으로 전역 정보를 선별적으로 통합합니다. 이를 통해 로컬 모델 성과를 개선하고 데이터 프라이버시를 보호하는 방식으로 작동합니다. 각 클라이언트는 BERT 모델을 사용하여 자신만의 데이터를 통해 신뢰도 점수를 생성하고, 서버는 이 정보를 기반으로 글로벌 모델을 업데이트합니다.

- **Performance Highlights**: Synthetic-Persona-Chat, CMU_DoG 및 WoW 데이터 세트에서의 실험 결과, FedDTRE가 대화 생성 품질을 개선하고 개인 맞춤화 개인화 모델링 간의 우수한 균형을 달성한다는 것을 보여주었습니다. 이는 소규모 데이터 클라이언트의 오버피팅 문제를 효과적으로 완화하고 글로벌 지식을 보존하여 전체 모델의 일반화 능력을 향상시키는 데 기여합니다.



### A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models (https://arxiv.org/abs/2510.08049)
- **What's New**: 이번 연구는 기존의 결과 보상 모델(Outcome Reward Models, ORMs)이 최종 답변만 평가하는 데 그치는 문제를 해결하기 위해, 과정 보상 모델(Process Reward Models, PRMs)을 소개합니다. PRMs는 단계별 또는 궤적 수준에서 추론을 평가하고 안내하는 방법입니다. 이 설문조사는 PRMs의 생성, 구축 및 테스트 시 확장과 강화 학습(Reinforcement Learning) 적용 방법을 체계적으로 다룹니다.

- **Technical Details**: PRMs의 작동 방식은 데이터(데이터 생성) 수집부터 시작하여, 모델 구성, 그리고 실제 테스트에 적용하는 과정까지 포함됩니다. 연구는 수학, 코드, 텍스트, 다중 모달 추론(multimodal reasoning), 로봇 공학(robotoics) 및 에이전트(agents) 분야에서의 적용을 요약하며, 새로운 벤치마크에 대해서도 논의합니다. 이런 과정에서 PRMs는 보다 섬세하고 견고한 추론 정렬(fine-grained, robust reasoning alignment)을 위해 필수적인 역할을 합니다.

- **Performance Highlights**: 이 연구는 PRMs가 기존 ORM의 한계를 극복하고, 다양한 응용 분야에서 더욱 향상된 추론 성능을 발휘할 수 있음을 강조합니다. 앞으로의 연구 방향과 디자인 공간을 명확히 하여, 더 세밀한 추론 정렬을 위한 도전 과제를 드러내고자 하는 목표를 가지고 있습니다. PRMs가 다양한 벤치마크에서의 성과를 통해 그 활용 가능성을 보여줍니다.



### Climate Knowledge in Large Language Models (https://arxiv.org/abs/2510.08043)
Comments:
          16 pages, 4 figures, 2 tables

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 기후 관련 응용 프로그램에서의 사용을 조사합니다. 특히 LLM이 파라메트릭 지식에서 기후 정상(Climate Normals)을 기억할 수 있는 능력이 얼마나 되는지를 평가합니다. 연구는 특정 위치에서의 1991-2020년 평균 7월 기온을 다루며, 다양한 지역적 맥락을 포함한 질문을 통해 LLM의 성능을 검증합니다.

- **Technical Details**: 연구진은 1° 해상도의 전 세계 쿼리 격자를 구성하고, ERA5 재분석 데이터를 기준으로 LLM의 응답을 검증했습니다. 그 결과 LLM은 위도 및 지형적 패턴을 포착하며, 루트 평균 제곱 오차(RMSE)는 3-6 °C로 나타났고, 바이어스는 ±1 °C였습니다. 그러나 산악 지역과 고위도에서의 공간적 일관된 오류가 발견되었습니다.

- **Performance Highlights**: 연구에 따르면, LLM은 1950-1974년과 2000-2024년 사이의 관측된 온난화의 글로벌 평균 크기를 캡처하였으나, 온도 변화의 공간적 패턴을 재현하는 데에는 실패했습니다. 또한, 지리적 맥락을 포함하면 오차가 평균 27% 감소하며, 대형 모델일수록 위치 설명자에 더 민감한 반응을 보였습니다. 이 연구는 LLM의 기후 지식 측정을 위한 재현 가능한 기준을 제공하며, 기후 커뮤니케이션 평가에 보완적인 역할을 합니다.



### ChatGPT as a Translation Engine: A Case Study on Japanese-English (https://arxiv.org/abs/2510.08042)
- **What's New**: 이번 연구는 ChatGPT를 일본어-영어 번역에 활용하며, 간단한 프롬프트와 향상된 프롬프트를 비교하고 상용 번역 엔진과의 차별성을 조사했습니다. 자동 평가 및 MQM 기반의 인간 평가를 통해, 전체 문서 번역이 문장 레벨 번역보다 우수하다는 결과를 발견했습니다. 또한, ChatGPT-3.5와 ChatGPT-4 두 모델의 성능을 비교한 결과, 각 모델의 정확도와 유창성 간의 트레이드오프가 있었음을 확인했습니다.

- **Technical Details**: 연구에서는 ParaNatCom, FLORES, Novels, KFTT, WMT News와 같은 여러 공공 데이터셋을 사용했습니다. 간단한 프롬프트는 소스 및 타겟 언어에 따라 텍스트를 번역하도록 ChatGPT에 지시하며, 향상된 프롬프트는 카테고리, 스타일 및 톤을 고려하도록 설계되었습니다. 평가 방법으로 BLEU, COMET, DA-BERT와 같은 자동 평가 지표 및 MQM 프레임워크를 기반으로 한 인간 평가를 사용했습니다.

- **Performance Highlights**: 연구 결과, 문서 레벨 번역이 문장 레벨 번역보다 더 높은 점수를 기록했으며, 특히 BLEU에서 60%의 경우가 문서 레벨 번역을 선호했습니다. 결과적으로, ChatGPT는 두 가지 상용 번역 시스템과 경쟁력 있는 결과를 보였으며, ChatGPT-3.5가 자동 평가에서 우선적인 선택을 받는 경향을 보였습니다. 하지만 인간 평가에서는 ChatGPT-4가 유창성 면에서 더 나은 성과를 거두었다는 점을 강조하고 있습니다.



### Learning on the Job: An Experience-Driven Self-Evolving Agent for Long-Horizon Tasks (https://arxiv.org/abs/2510.08002)
- **What's New**: 이번 논문에서는 MUSE라는 새로운 에이전트 프레임워크를 제안합니다. MUSE는 경험 기반의 자가 진화 시스템을 중심으로 하는 계층적 메모리 모듈을 도입하여 에이전트가 실시간으로 경험을 학습하고 개선할 수 있도록 지원합니다. 기존의 LLM 에이전트는 경험에서 학습할 수 없어 고정된 성능을 보였으나, MUSE는 이를 극복하여 지속적인 학습을 가능하게 합니다.

- **Technical Details**: MUSE는 다양한 레벨의 경험을 조직하고 이를 활용하여 장기 과제를 수행할 수 있는 능력을 갖추고 있습니다. 에이전트는 각 서브 태스크 실행 후 자신의 경로를 반성하고 원시 경로를 구조화된 경험으로 변환한 후 메모리 모듈에 다시 통합합니다. 이에 따라 에이전트는 정적 방식에서 벗어나 동적인 시스템으로 발전할 수 있습니다. 또한 메모리는 자연어로 저장되어 있어, 다른 LLM에서도 경험을 쉽게 전이할 수 있는 장점이 있습니다.

- **Performance Highlights**: MUSE는 TAC(한계 효율성 지표)라는 기준에서 새로운 SOTA(SOTA: State Of The Art) 성과를 기록했습니다. 가벼운 Gemini-2.5 Flash 모델을 사용하여 51.78%의 점수를 달성하며, 이전 SOTA보다 20% 증가했습니다. 실험 결과, 에이전트는 자율적으로 경험을 축적함에 따라 작업 수행 능력이 지속적으로 향상되며, 강력한 지속적 학습과 자가 진화 능력을 가지고 있음을 입증하였습니다.



### Leveraging Author-Specific Context for Scientific Figure Caption Generation: 3rd SciCap Challeng (https://arxiv.org/abs/2510.07993)
- **What's New**: 이번 논문에서는 3차 SciCap 챌린지를 위한 도메인 특화 캡션 생성 시스템을 제안합니다. LaMP-Cap 데이터셋을 이용하여 저자 특유의 문체와 도표 관련 텍스트 컨텍스트를 통합하였습니다. 이 시스템은 두 단계로 이루어지며, 첫 단계는 컨텍스트 필터링과 특정 카테고리 프롬프트 최적화, 그리고 캡션 후보 선정으로 구성됩니다. 두 번째 단계는 프로필 피규어를 이용하여 스타일을 개선하는 few-shot prompting 을 적용합니다.

- **Technical Details**: LaMP-Cap 데이터셋은 110,828개의 과학 기사를 포함하고 있으며, train/test/validation 데이터가 80:10:10 비율로 나누어져 있습니다. 캡션 생성 과정에서, 우리는 문단 정보를 사용하여 noiseless 캡션을 생성하는 두 단계 파이프라인을 개발하였습니다. 첫 단계에서는 context grounded 캡션을 생성하고, 두 번째 단계에서는 프로필 정보를 통해 스타일 조정을 합니다. MIPROv2와 SIMBA를 이용하여 카테고리 중심의 프롬프트 템플릿을 개발하였습니다.

- **Performance Highlights**: 우리의 실험 결과, 카테고리 특정 프롬프트가 제로샷 및 일반 최적화 접근 방식을 초월하는 성능을 보였습니다. ROUGE-1 recall이 +8.3% 향상되는 반면, precision 손실은 -2.8%로 제한되었고, BLEU-4 감소는 -10.9%에 그쳤습니다. 프로필에 기반한 스타일 수정을 통해 BLEU 점수가 40-48% 향상되었으며 ROUGE에서도 25-27%의 향상을 달성했습니다. 전반적으로, 본 시스템은 컨텍스트 이해와 저자별 스타일 적응을 결합하여 과학적으로 정확하고 스타일적으로 충실한 캡션을 생성할 수 있음을 보여줍니다.



### Active Confusion Expression in Large Language Models: Leveraging World Models toward Better Social Reasoning (https://arxiv.org/abs/2510.07974)
Comments:
          15 pages, 10 figures

- **What's New**: 본 연구에서는 대형 언어 모델(LLM)이 사회적 추론 과제에서 겪는 한계를 다룹니다. LLM들은 데이터에서 발췌한 논리를 기반으로 수학적 및 코드 추론에서는 뛰어난 성과를 보이지만, 다수의 참여자와 시간이 얽힌 사회적 시나리오 처리 시 인지적 혼란을 경험하고 있습니다. 이러한 연구의 주된 목적은 적응형 세계 모델 강화 추론 메커니즘을 제안하여 이러한 결점을 극복하는 것입니다.

- **Technical Details**: 우리의 메커니즘은 텍스트 기반의 세계 모델을 구성하여 개체 상태 및 시간적 시퀀스를 추적하고, 추론 중 혼란 신호를 동적으로 모니터링합니다. 혼란이 발생했을 때, 자동으로 세계 상태를 명확하게 설명하여 LLM이 인지적 딜레마를 극복하도록 도와줍니다. 이 접근법은 인간이 사회적 상호작용 중에 사용하는 암묵적 세계 모델을 모방합니다.

- **Performance Highlights**: 세 가지 사회적 벤치마크에서 평가한 결과, 제안된 메커니즘은 정확도에서 유의미한 향상(예: Hi-ToM에서 +10%)을 기록했습니다. 또한 계산 비용을 최대 33.8% 줄였습니다. 이러한 결과는 사회적 맥락에서 LLM을 배치하는 데 있어 간단하면서도 효과적인 솔루션을 제공함을 보여줍니다.



### LightReasoner: Can Small Language Models Teach Large Language Models Reasoning? (https://arxiv.org/abs/2510.07962)
- **What's New**: 이번 논문에서는 LightReasoner라는 새로운 프레임워크를 제안하여, 약한 언어 모델(SLM, Small Language Model)이 강한 언어 모델(LLM, Large Language Model)의 학습을 돕는 방법을 탐구합니다. 이 접근법은 LLM과 SLM의 행동 차이를 이용해 고부가 가치 추론 순간을 드러내어 모델의 추론 능력을 개선할 수 있는 잠재력을 강조합니다. 이는 일반적인 감독적 세부 조정(Supervised Fine-Tuning, SFT)의 필요성을 줄이고, 여러 자원을 효율적으로 사용할 수 있게 해줍니다.

- **Technical Details**: LightReasoner는 두 단계로 운영됩니다: 1단계에서는 중요한 추론 순간을 샘플링하여 전문가 모델과 아마추어 모델 간의 행동 차이를 기반으로 감독 예제를 생성합니다. 2단계에서는 이러한 예제와 정렬하여 전문가 모델이 아마추어 모델과의 대조를 통해 자신의 장점을 강화하도록 훈련합니다. 특히, Kullback-Leibler (KL) 발산을 통해 전문가와 아마추어 모델의 다음 토큰 예측 차이를 계산하여 중요한 의사 결정을 식별합니다.

- **Performance Highlights**: LightReasoner는 7개의 수학적 성능 기준에서 최대 28.1%의 정확도 향상을 달성하며, 시간 소비는 90%, 샘플링된 문제는 80%, 조정된 토큰 사용은 99%까지 줄일 수 있습니다. 이러한 점에서 LightReasoner는 무라라이트 진실(label) 의존 없이도 효과적인 학습 신호를 생성하는 데 기여합니다. 이는 약한 모델을 강한 모델에 대한 효과적인 가르침 신호로 변환하여 LLM의 추론 향상에 기여하고 있습니다.



### A$^2$Search: Ambiguity-Aware Question Answering with Reinforcement Learning (https://arxiv.org/abs/2510.07958)
- **What's New**: 최근에 발표된 연구는 대규모 언어 모델(LLM)과 강화 학습(RL)의 발전이 개방형 질문 응답(QA)에서 뛰어난 성능을 보여준다는 점에 주목하고 있습니다. 특히, 다수의 유효한 답변이 가능한 질문에서는 기존 모델들이 여전히 어려움을 겪고 있음을 강조합니다. 이를 해결하기 위해 A$^2$Search라는 새로운 주석 없는(end-to-end) 훈련 프레임워크를 제안합니다.

- **Technical Details**: A$^2$Search는 애매한 질문을 탐지하고 대안 답변을 수집하는 자동화된 파이프라인을 기본으로 구성됩니다. 이 시스템은 경로 샘플링(trajectory sampling)과 증거 검증(evidence verification)을 통해 답변을 수집하며, 다수의 답변을 자연스럽게 수용할 수 있는 $	ext{AnsF1}$ 보상을 이용해 RL로 최적화됩니다. 이러한 접근법은 수동 주석의 부담을 줄이고 다중 홉 데이터셋에 대한 확장을 용이하게 만듭니다.

- **Performance Highlights**: A$^2$Search는 8개의 개방형 QA 벤치마크에서 새로운 최첨단 성능을 달성하였습니다. A$^2$Search-7B 모델은 4개의 다중 홉 벤치마크에서 평균 $	ext{AnsF1}@1$ 점수 48.4%를 기록하여, ReSearch-32B(46.2%)와 같은 강력한 기준 모델을 초월하였습니다. 연구 결과는 ambiguity(애매함)를 수용하는 것이 더 신뢰할 수 있는 QA 시스템을 구축하는 데 필수적임을 보여줍니다.



### Vision-Enabled LLMs in Historical Lexicography: Digitising and Enriching Estonian-German Dictionaries from the 17th and 18th Centuries (https://arxiv.org/abs/2510.07931)
- **What's New**: 이 연구는 2022년부터 2025년까지 에스토니아어 연구소에서 수행된 대규모 언어 모델(LLMs)의 응용에 대한 내용으로, 17세기와 18세기 에스토니아어 사전 연구에 중점을 두고 있습니다. 연구자들은 현대어 형태와 의미로 역사적인 사전을 풍부하게 하는 것, 고딕 글꼴(Fraktur)로 인쇄된 소스에 대한 텍스트 인식을 위한 비전 지원 LLM 사용, 그리고 통합된 크로스 소스 데이터 세트 생성을 위한 준비와 같은 세 가지 주요 영역을 다루고 있습니다.

- **Technical Details**: 초기 실험에서는 1648년 J. Gutslaff의 사전을 사용했으며, LLMs가 사전 정보를 반자동으로 풍부하게 하는 데 상당한 잠재력을 보여줍니다. Claude 3.7 Sonnet는 충분한 맥락을 제공받았을 때 81%의 단어 항목에 대한 의미 및 현대어 동등치를 정확하게 제공했습니다. 또한, 1732년 A. T. Helle의 사전에서의 텍스트 인식 실험에서는 제로샷(zero-shot) 방법을 활용하여 41%의 단어 항목을 오류 없는 JSON 포맷으로 식별하고 구조화할 수 있었습니다.

- **Performance Highlights**: 이 연구의 결과는 LLMs가 소규모 언어에 대해서도 시간과 재정 자원을 절약하는 데 많은 잠재력을 지니고 있음을 보여줍니다. A. W. Hupel의 1780 문법서의 에스토니아-독일어 사전 부분을 디지털화하기 위해 스캔된 이미지 파일의 겹치는 타일링(overlapping tiling) 기법이 사용되었으며, 하나의 LLM이 텍스트 인식을 수행하고 다른 하나가 구조화된 출력을 병합하는 데 사용되었습니다. 이러한 접근 방식은 소규모 언어 자료의 디지털화와 인식에 대한 새로운 가능성을 제시합니다.



### Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation (https://arxiv.org/abs/2510.07926)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 정보의 누락을 생성할 수 있는 문제를 다루고 있습니다. 특히, 이 연구는 누락된 정보 또는 잘못 표현된 견해를 감지하기 위한 자동화된 평가 전략을 제안합니다. NLI 기반 방법, Q&A 기반 접근법 및 엔드투엔드 방식의 세 가지 자동화 평가 전략을 통해 LLM의 텍스트의 포괄성을 평가하고 있습니다.

- **Technical Details**: 연구에서는 모델의 출력을 원자적 진술로 분해하여 관계 그래프를 구성하고, 이를 통해 누락된 내용을 식별하는 NLI 기반 방법을 포함합니다. 또한, Q&A 기반 방법은 질문-답변 쌍을 추출하고 서로 다른 출처에서의 응답을 비교하여 관계를 분석합니다. 엔드투엔드 방법은 LLM을 사용하여 모델 출처에서 결여된 정보를 직접적으로 인식하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 엔드투엔드 방식이 더 복잡한 방법들에 비해 놀라운 효과를 보였지만, 견고성(robustness), 해석 가능성(interpretability), 결과의 세분화(granularity)는 낮았습니다. 이 연구는 WikiContradict와 ConflictBank 벤치마크를 통해 평가되었으며, 여러 인기 있는 오픈 웨이트 LLM들이 실제 사용자 질문에 대한 포괄성을 평가했습니다.



### STEPER: Step-wise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented Language Models (https://arxiv.org/abs/2510.07923)
Comments:
          EMNLP 2025 Main

- **What's New**: 이번 연구에서는 Stepwise Knowledge Distillation(단계별 지식 증류)이라는 새로운 접근법을 제시합니다. 이는 복잡한 실제 질문에 대한 응답을 위한 단계별 정보 검색 및 통합 과정에서의 합리적인 응답 생성을 목표로 합니다. 기존의 지식 증류 방법들이 각 단계에서의 다양한 추론 능력을 간과하는 문제를 해결하고자 합니다.

- **Technical Details**: StepER은 각 단계별 요구되는 정보와 추론을 맞추기 위해 단계별 감독(step-wise supervision)을 적용합니다. 또한, 문제의 난이도에 따라 학습 최적화를 진행하는 difficulty-aware training(난이도 인식 훈련) 기법을 도입하여 적절한 단계를 우선시 합니다. 이 방법은 다중 단계 정보 검색이 가능한 언어 모델에 적용 가능하며, 검색 쿼리를 사용하는 추론 경로 또는 분해된 질문을 포함하고 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해, StepER은 다단계 QA 벤치마크에서 이전 방법들을 능가하는 성능을 보였습니다. 특히, 8B 모델이 70B 교사 모델과 유사한 성능을 달성함으로써 StepER의 효용성을 입증하고 있습니다.



### Towards Human-Like Grading: A Unified LLM-Enhanced Framework for Subjective Question Evaluation (https://arxiv.org/abs/2510.07912)
- **What's New**: 이 논문은 다양한 주관적 질문에 대한 자율 채점의 도전 과제를 해결하기 위해 통합된 LLM(대형 언어 모델) 강화 자동 채점 프레임워크를 제안합니다. 기존의 연구들은 특정 주관적 질문 유형에만 집중했으며, 다양한 질문 형식을 지원할 수 있는 포괄적인 접근이 부족했습니다. 새로운 프레임워크는 학생의 답변을 인간과 유사하게 평가하는 데 필요한 네 가지 보완 모듈을 통합합니다.

- **Technical Details**: 제안된 프레임워크는 텍스트 유사도 모듈(TSM), 키 포인트 매칭 모듈(KPM), 일반 평가 모듈(LGE), 가상 질문 매칭 모듈(PQM) 및 심층 융합 레이어로 구성됩니다. KPM은 학생과 참조 답변에서 중요한 지식 포인트를 추출하고, LGE는 여러 차원에서 답변을 평가하여 인간 채점자의 포괄적 판단을 모방합니다. PQM은 학생의 답변에서 생성한 가상 질문을 사용하여 원래 질문과의 의미적 정합성을 평가합니다.

- **Performance Highlights**: 제안된 프레임워크는 일반 목적 및 도메인 특화 데이터셋을 통해 기존의 전통적인 방법 및 LLM 기반 기준보다 일관되게 더 나은 성능을 보였습니다. 두 개의 새로운 데이터셋이 구축되었으며, 테스트 결과는 새로운 접근 방식이 다양한 질문 유형과 평가 메트릭에서 탁월함을 입증하였습니다. 실제 전자상거래 기업의 교육 및 인증 시험에서도 성공적으로 사용되었습니다.



### ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Reca (https://arxiv.org/abs/2510.07896)
- **What's New**: 이 논문은 다중 단계의 사실 회상(multi-hop factual recall)에서의 효율적인 지식 편집(KE)의 필요성을 강조합니다. 기존의 방법들은 이 과정에서 성능 저하가 두드러지며, 특히 중간의 암시적 주체가 포함될 때 더욱 문제가 심각해집니다. 이를 해결하기 위해 저자는 신경 수준에서의 지식의 동적 표현 방식을 탐구하고, 지식 편집의 새로운 접근인 ACE(Attribution-Controlled Knowledge Editing)를 제안합니다.

- **Technical Details**: ACE 프레임워크는 임시적으로 활성화되는 쿼리-값(QUERY-VALUE) 경로를 식별하고 편집하는 기법을 사용하여 다중 단계 이유에 대한 지식 편집을 가능하게 합니다. 이 방식은 쿼리와 값 노드 간의 상호작용을 통해 정보가 누적되는 메커니즘을 분석하여, 이론적으로 기반을 둔 지식 편집 방법을 제공합니다. 이 연구는 특정 쿼리 함수 및 값 함수 간의 추적을 통해 LLM의 핵심적 구조를 탐색합니다.

- **Performance Highlights**: ACE는 GPT-J와 Qwen3-8B 모델에서 각각 9.44%와 37.46%의 정확도 향상을 보여주는 등 최신 기술을 능가하는 성능을 확인했습니다. 또한, 실험을 통해 쿼리 계층을 생략할 경우 16.51%의 성능 하락이 발생하고, 값 계층을 생략할 경우 40.45%의 더 심각한 성능 저하가 나타났음을 보여주었습니다. 이로 인해 다중 스텝 이유에서의 신경 생성의 중요성이 강조됩니다.



### Metric Calculating Benchmark: Code-Verifiable Complicate Instruction Following Benchmark for Large Language Models (https://arxiv.org/abs/2510.07892)
Comments:
          Accepted to the EMNLP2025

- **What's New**: 새로운 LLM(대형 언어 모델)들이 많은 기존의 어려운 벤치마크를 초과 달성한 현재, 더 이상 명확한 차별화를 위한 공간이 부족해졌습니다. 이 연구에서는 LLM들이 문자열 일치 NLP(Natural Language Processing) 메트릭스를 단계별 지침에 따라 정확하게 실행할 수 있는지를 평가하기 위한 MCBench라는 벤치마크를 소개합니다. MCBench는 주관적 판단이나 일반 추론에 의존하지 않고, 객관적이고 결정론적이며 코드로 검증 가능한 평가를 제공합니다.

- **Technical Details**: MCBench의 설정은 LLM이 정확하게 단계별 실행을 유지할 수 있는지를 체계적으로 시험할 수 있도록 합니다. 지침 준수, 수치 계산, 중간 결과 처리의 장기적 일관성을 포함하여, LLM의 이러한 능력을 객관적으로 평가하기 위해 참조 코드를 제공합니다. 여기서 우리는 세 가지 평가 메트릭과 LLM의 세부 지침 이해 능력을 측정하기 위해 설계된 세 가지 벤치마크 변형을 제공합니다.

- **Performance Highlights**: 분석 결과, MCBench는 최신 LLM의 성능을 평가하기 위한 효과적이고 객관적인 도구로 기능함을 보여줍니다. LLM들이 단계별 지침을 얼마나 잘 따르는지를 명확하게 측정할 수 있어, LLM의 진화에 기여할 수 있는 유망한 방향성을 제공하고 있습니다. MCBench를 통하여 앞으로 LLM의 평가 기준이 더욱 정교해질 것으로 기대됩니다.



### Standard-to-Dialect Transfer Trends Differ across Text and Speech: A Case Study on Intent and Topic Classification in German Dialects (https://arxiv.org/abs/2510.07890)
- **What's New**: 이 연구는 표준 독일어에서 방언(다이얼렛)으로의 전이 과정에서 발생하는 문제를 다루고 있습니다. 저자들은 텍스트 데이터 대신 음성과 방언의 변별성에 중점을 두고 방언 오디오 의도 분류 데이터세트를 최초로 발표했습니다. 실험적 결과에 기반하여, 방언 데이터에서는 오직 음성을 사용하는 시스템이 가장 높은 성능을 기록했음을 보여주고 있습니다.

- **Technical Details**: 연구는 세 가지 설정(텍스트 모델, 음성 모델, 그리고 연속 시스템)에서 진행되었습니다. 여기서 연속 시스템은 먼저 음성을 자동 텍스트로 변환한 후, 텍스트 모델로 추가 처리를 합니다. 이를 통해 방언 텍스트를 처리하는 데 있어 음성 모델이 더 강력하게 작용할 수 있음을 확인했습니다. 또한 Bavarian 방언에 대한 오디오 데이터도 포함되어 있습니다.

- **Performance Highlights**: 연구 결과, 표준 독일어 데이터에 대해서는 텍스트 전용 모델이 가장 우수한 성능을 보인 반면, 방언 데이터에서는 위험한 변환을 통해 표준화된 출력에 가까운 ASR 시스템이 더 나은 성능을 보여주었습니다. 또한, 텍스트 전용 모델이 독일어 데이터에서 우수한 반면 방언 데이터에서는 음성 전용 모델이 더 나은 성능이었음을 발견했습니다. 전체적으로 방언 처리에 있어 깊은 통찰을 제공하는 결과를 제시했습니다.



### Contrastive Weak-to-strong Generalization (https://arxiv.org/abs/2510.07884)
- **What's New**: 이 연구에서는 약한 모델에서 강한 모델로의 일반화를 위한 새로운 프레임워크인 Contrastive Weak-to-Strong Generalization (ConG)을 제안합니다. 이 프레임워크는 인간 피드백이나 명시적인 보상 모델 없이도 더 높은 품질의 샘플을 생성할 수 있는 가능성을 탐구합니다. 또한, ConG는 Contrastive Decoding (CD)과의 구조적 동등성을 기반으로 하여 약한 모델의 출력에서 노이즈를 줄여줍니다.

- **Technical Details**: ConG는 사전 및 사후 정렬된 약한 모델 간의 대비 디코딩을 활용하여 강한 모델을 훈련시키는 방법을 제시합니다. 이는 임시 보상을 사용하여 추출된 샘플의 품질을 평가하고, значности 높은 등장 확률 로그 비율을 최대화함으로써 약한 모델로부터 높은 신뢰도의 샘플을 생성합니다. 연구는 Qwen2.5 및 Llama3와 같은 주류 LLM 계열을 대상으로 하여 상응하는 설정에서 검증됩니다.

- **Performance Highlights**: 실험 결과, ConG는 전통적인 약한-강한 방법에 비해 모든 모델에서 일관되게 유의미한 성능 개선을 보여주었으며, 평균 16.5%의 향상을 기록했습니다. 이러한 성과는 ConG의 일반성과 효과성을 입증하며, 기능 전이, 노이즈 감소 및 강건성 향상을 통해 인공지능 일반화(AGI)로 나아가는 유망한 경로를 제시합니다.



### CS3-Bench: Evaluating and Enhancing Speech-to-Speech LLMs for Mandarin-English Code-Switching (https://arxiv.org/abs/2510.07881)
- **What's New**: 이 논문에서는 Code-Switching Speech-to-Speech Benchmark (CS3-Bench)를 제안하여 코드 스위칭 상황에서의 언어 정렬 문제를 해결하려고 합니다. 기존의 다중 언어 모델들이 이와 같은 환경에서 심각한 성능 저하를 보인다는 점을 발견했습니다. 이 벤치마크는 주로 중국어와 혼합된 영어 쿼리로 구성되어 있으며, 모델 성능을 평가하기 위한 여러 메트릭스를 제공합니다. 이를 통해 음성 대화 시스템의 효과성과 자연스러움을 향상시키고자 하는 목표를 가지고 있습니다.

- **Technical Details**: CS3-Bench는 총 362개의 지식 세트와 200개의 개방형 질문을 포함한 데이터셋으로 구성되어 있습니다. 이 벤치마크는 음성 인식의 정확성, 발음의 명확성 및 언어 선택의 일관성을 평가하는 체계적인 프레임워크를 제공합니다. 데이터 생성 프로세스는 코스위칭 처리에 특화된 여러 접근 방식을 포함하며, Chain of Recognition (CoR)와 Keyword Highlighting (KH) 기법을 이용하여 모델의 이해력을 개선합니다.

- **Performance Highlights**: 실험 결과, 기존의 7개 음성 대화 모델을 평가하였으며, 모든 모델이 단일 언어 쿼리에서 강력한 성능을 나타냈지만, 코드 스위칭 쿼리로 전환될 경우 평균 30% 이상의 성능 저하가 발생했습니다. 특히 모델의 발음 오류는 여전히 문제로 남아 있으며, Qwen2.5-Omni는 발음 정확도에서 가장 높은 수치를 보였습니다. 최종적으로, 개선된 접근 방식을 통해 지식 정확도가 25.14%에서 46.13%로 향상되었으며, 개방형 질문 이해 비율은 64.5%에서 86.5%로 증가했습니다.



### Do LLMs Really Need 10+ Thoughts for "Find the Time 1000 Days Later"? Towards Structural Understanding of LLM Overthinking (https://arxiv.org/abs/2510.07880)
Comments:
          30 pages, 41 figures, 10 tables. Preprint

- **What's New**: 이번 연구는 LLM의 과도한 사고(overthinking) 문제를 체계적으로 분석하는 도구인 TRACE(Thought-process Reconstruction and Automated Clustering Engine)를 소개합니다. 과거 연구에서 LLM의 사고 구조를 깊게 분석한 시도가 부족했음을 지적하며, LLM이 단순한 질문에 대해 5배에서 20배 더 긴 추론 시간을 소요하면서 성능 개선이 없음을 입증했습니다. 또한, 연구는 다양한 질문 유형에 대한 사고 패턴을 분류하여, 두 가지 주요 패턴인 Explorer와 Late Landing을 도출해냈습니다.

- **Technical Details**: TRACE는 LLM의 사고 과정을 독립적인 서브-사고(sub-thought)로 분해하고, 각 서브-사고의 정확성과 유용성을 평가하는 방식으로 동작합니다. 이를 통해 사고 진행 그래프(thought progression graphs)를 구성하고, 주제별 유사 질문에 대한 일반적인 사고 패턴을 식별할 수 있습니다. 또한, 사고 구조에 기반하여 과도한 사고의 유용성 기반 정의를 제안하며, 이는 단순한 길이 기반 지표를 넘어 심도 있는 분석을 가능하게 합니다.

- **Performance Highlights**: 이 분석을 통해 확인된 결과는 간단한 쿼리에서 사고 모델이 과도하게 많은 자원을 소모하고 있음을 강조하며, 이는 성능 저하를 초래할 수 있음을 보여줍니다. 또한, 이 연구는 LLM이 명확한 답변이 보이는 간단한 문제에 대해서도 끊임없이 답변을 탐색하는 경향이 있음을 발견했습니다. 이러한 통찰력은 LLM의 사고 과정을 더욱 구조적으로 이해하고, 실제적인 과도한 사고 관리에 대한 지침을 제공하는 데 기여합니다.



### Ready to Translate, Not to Represent? Bias and Performance Gaps in Multilingual LLMs Across Language Families and Domains (https://arxiv.org/abs/2510.07877)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전은 기계 번역(Machine Translation) 분야를 혁신적으로 변화시켰습니다. 그러나 이러한 모델은 언어 가족 및 전문 분야별로 성능 차이를 보이며, 낮은 자원 언어에 대한 공정성 문제를 제기합니다. 이러한 문제를 해결하기 위해, 저자들은 Translation Tangles라는 새로운 통합 프레임워크와 데이터셋을 도입하여 LLM의 번역 품질과 공정성을 평가합니다.

- **Technical Details**: 제안된 프레임워크는 24개의 쌍방향 언어 쌍을 다양한 메트릭을 통해 평가하며, 공정성 및 번역 품질을 종합적으로 분석합니다. 또한, 규칙 기반 휴리스틱 및 LLM 기반 검증을 통합하여 하이브리드 편향 감지 파이프라인을 제안하고, 1,439개의 번역 참조 쌍에 대한 인간 평가를 기반으로 한 고품질의 편향 주석 데이터셋을 발표합니다.

- **Performance Highlights**: 저자들은 번역 성능을 여러 차원에서 평가하는 다국어 벤치마크를 개발하여 고자원 및 저자원 언어 쌍 모두를 포함하는 평가를 수행했습니다. 이 연구는 LLM 기반 번역의 공정성과 신뢰성을 높이기 위한 기초 자료들을 제공하며, 자동 편향 감지 시스템의 효과성을 검증할 수 있는 금 표준을 제시합니다.



### AdaSwitch: Adaptive Switching Generation for Knowledge Distillation (https://arxiv.org/abs/2510.07842)
- **What's New**: 이번 논문에서는 AdaSwitch라는 새로운 접근 방식을 제안합니다. AdaSwitch는 작은 언어 모델(SLM)과 큰 모델 간의 동적 토큰 수준 합성을 통해 지식 전이를 수행합니다. 이 방법은 학생 모델이 스스로의 예측을 탐색한 후, 실시간 품질 평가에 근거하여 교사의 지도를 선택적으로 통합할 수 있도록 합니다.

- **Technical Details**: AdaSwitch는 두 단계의 시퀀스 생성 프레임워크로, 탐색 단계에서는 학생이 독립적으로 시퀀스를 생성하고, 지도 단계에서는 교사가 나머지 시퀀스를 생성하여 고품질 출력을 보장합니다. 이 과정에서 생성 난이도에 따라 전환 임계값이 동적으로 조정되어, 과도한 개입을 방지하고 일관성과 품질 간의 균형을 유지합니다.

- **Performance Highlights**: 세 개의 데이터세트에서 두 개의 교사-학생 모델 쌍을 사용하여 AdaSwitch를 평가했습니다. 실험 결과, AdaSwitch는 대부분의 시나리오에서 성능을 일관되게 향상시키며, 평균적으로 순수 온 정책 방법보다 1.3배의 안정적인 계산 오버 헤드를 유지하면서 SKD보다 10% 감소한 효율성을 달성했습니다.



### Multilingual Generative Retrieval via Cross-lingual Semantic Compression (https://arxiv.org/abs/2510.07812)
Comments:
          EMNLP 2025, Findings, Long

- **What's New**: 이 논문에서는 Cross-lingual Semantic Compression (MGR-CSC)을 통한 Multilingual Generative Retrieval을 제안하며, 이는 다국어 키워드를 공유 원자로 통합하여 의미를 정렬하고 문서 식별자 공간을 압축하는 혁신적인 프레임워크입니다. MGR-CSC는 일관된 식별자를 할당함으로써 다국어 정렬을 개선하고 중복성을 줄여 디코딩 효율을 향상시킵니다. 실험 결과, MGR-CSC는 mMarco100k에서 6.83%, mNQ320k에서 4.77%의 향상된 검색 정확도를 보여주며, 문서 식별자의 길이는 각각 74.51%, 78.2% 감소시켰습니다.

- **Technical Details**: MGR-CSC는 세 가지 주요 구성 요소로 이루어져 있습니다: 첫째, 다국어 문서에서 뚜렷한 키워드를 추출하고 이를 키워드 집합으로 표기합니다. 둘째, 이러한 키워드를 언어 독립적인 의미 원자로 클러스터링하여 각 문서에 고유한 DocID를 할당합니다. 마지막으로, 디코딩 과정에서 동적 제약을 적용하여 DocID 원자를 하나씩 생성하는 방법을 채택합니다.

- **Performance Highlights**: MGR-CSC는 mMarco100k와 mNQ320k 여러 벤치마크 데이터셋에서 기존의 다국어 생성 검색 방법보다 각각 6.83% 및 4.77%의 성능 향상을 달성했습니다. 또한, 이 방법은 DocID 토큰의 수를 각각 74.51%와 78.2% 감소시켜 메모리 효율성을 크게 향상시켰습니다.



### Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models (https://arxiv.org/abs/2510.07799)
- **What's New**: 이번 논문에서는 Multi-Agent Systems (MAS)에서 Large Language Models (LLMs)이 구동하는 통신 토폴로지를 동적으로 설계하는 새롭고 혁신적인 접근 방식인 Guided Topology Diffusion (GTD)를 소개합니다. GTD는 조건부 그래프 확산(process)을 활용하여 다중 목표를 고려하며 통신 구조를 반복적으로 생성하는 방식으로, 기존의 정적인 설계 방안에서의 한계를 극복하고자 합니다.

- **Technical Details**: GTD는 기존의 단일 단계 모델이 아닌, 경량 프록시 모델을 결합하여 각 단계에서 다중 목표 보상을 예측함으로써 실시간으로 최적화하는 방식을 제공합니다. 이 과정은 높은 복잡도를 가진 통신 구조의 생성을 유도하기 위한 반복적이고 세분화된 접근 방식을 통해 이루어집니다. 이를 통해 각 단계에서 통신 효율성, 비용, 강인성 등을 균형 있게 고려하여 최적의 통신 구조를 창출할 수 있습니다.

- **Performance Highlights**: GTD 프레임워크는 여러 테스트에서 검증을 받았으며, 실험 결과에서 기존 방법들에 비해 현저히 우수한 성능을 보였습니다. GTD는 특히 복잡한 작업에 대해 강력하게 적응할 수 있는 희소하고 효율적인 통신 토폴로지를 자동으로 생성하여 LLM 에이전트 간의 협업을 개선했습니다. 이러한 성과들은 GTD가 다중 목표를 고려하는 통신 구조 설계에서 혁신적인 해결책이 될 수 있음을 보여줍니다.



### HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation (https://arxiv.org/abs/2510.07794)
Comments:
          Under review

- **What's New**: 이 논문에서는 HiPRAG(Hierarchical Process Rewards for Efficient agentic RAG)라는 새로운 강화학습(RL) 훈련 방법론을 제안하며, 이는 기존의 suboptimal search behaviors를 해결하기 위해 설계되었습니다. HiPRAG는 검색 과정의 최적화뿐만 아니라 각 검색 결정에 대한 구체적인 피드백을 제공하여 효율성을 높이고 정확성을 증대시키는 데 초점을 맞추고 있습니다. 특히 이 방식을 통해 over-search와 under-search 문제를 줄이고, 동시에 에이전트의 추론 과정에 대한 세밀한 제어를 가능하게 합니다.

- **Technical Details**: HiPRAG는 LLM이 검색 도구를 사용할 때, 이 도구의 사용 방식이 최적화되어야 한다는 점을 강조합니다. 이 방법론은 에이전트의 추론 과정을 개별적으로 분해하여 각 단계에 대해 적절한 보상을 부여함으로써, 최종 결과뿐 아니라 과정의 품질을 충분히 고려합니다. 이를 통해 검색 결정의 필요성을 동적으로 평가하고, 최적의 검색 비율에 보너스를 제공하는 단계적 보상 체계를 구현하여 기존 방법의 한계를 극복합니다.

- **Performance Highlights**: 실험 결과, HiPRAG는 Qwen2.5와 Llama-3.2 모델에 대해 65.4%(3B) 및 67.2%(7B)의 평균 정확도를 기록했습니다. 이는 기존 모델들에 비해 수집 효율성과 accuracy 모두에서 현저한 개선을 보여주며, over-search 비율을 27%에서 2.3%로 줄이고 under-search 비율도 낮추는 성과를 올렸습니다. 전반적으로 HiPRAG는 다양한 LLM 및 RL 알고리즘에 대해 우수한 일반성을 보이며, 검색 에이전트의 추론 효율성과 최적성을 크게 향상시키는 가능성을 제시합니다.



### LLM4Cell: A Survey of Large Language and Agentic Models for Single-Cell Biology (https://arxiv.org/abs/2510.07793)
Comments:
          34 pages, 5 figures, 7 tables

- **What's New**: 이번 논문에서는 LLM4Cell이라는 대규모 언어 모델과 에이전틱(Agentic) 모델의 통합적 조사를 소개하고 있습니다. 이는 단일 세포 생물학(single-cell biology) 연구를 위한 최초의 포괄적 리포트로, RNA, ATAC, 멀티 오믹(multi-omic), 공간적(spatial) 데이터 모드에서 발전된 58개의 모델을 평가합니다. LLM4Cell은 다섯 가지 방법론 패밀리로 모델을 분류하고 여덟 가지 분석 작업에 맵핑하여, 실험의 일관성과 반복성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: LLM4Cell은 Foundation, Text-Bridge, Spatial, Multimodal, Epigenomic, Agentic 모델의 다섯 가지 범주로 나누어져 있으며, 각 범주는 주석(annotation), 궤적 추론(trajectory inference), 약물 반응 모델링(drug-response modeling) 등과 같은 여덟 가지 주요 작업에 대한 분석을 제공합니다. 각 모델은 생물학적 기반(biological grounding), 공정성(fairness), 개인 정보 보호(priacy), 설명 가능성(explainability) 등을 기준으로 10가지 차원에서 평가됩니다. 이를 통해 모델과 데이터셋 간의 관계를 명확히 하고, 신뢰할 수 있는 AI 개발을 위한 기준을 제시합니다.

- **Performance Highlights**: LLM4Cell은 40개 이상의 공개 데이터셋을 수집하고, 이를 바탕으로 한 모델 평가 방법론을 제공하여, 대규모 언어 모델 교육 및 평가를 위한 약 40개의 벤치마크 데이터셋을 모듈화합니다. 또한, 데이터셋의 불균형, 평가 기준의 한계 등을 감안하여 결과의 공정성과 재현 가능성을 높이는 방안을 모색하고 있습니다. 최종적으로, LLM4Cell은 단일 세포 지능(single-cell intelligence)을 위한 언어 기반 접근법의 통합적 관점을 제시하며, 해석 가능성 및 신뢰성 문제의 해결을 위한 필요 과제를 명확히 하고 있습니다.



### RCPU: Rotation-Constrained Error Compensation for Structured Pruning of a Large Language Mod (https://arxiv.org/abs/2510.07782)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)에서의 구조적 가지치기로 인해 발생하는 오류를 해결하기 위해 회전 제약 보상(rotation-constrained compensation) 방법을 제안합니다. LLM은 방대한 데이터 세트에서 학습되며, 이 표현 공간에서 풍부한 의미적 지식을 축적합니다. 그러나 보통은 작은 캘리브레이션 데이터(calibration data)로 가지치기를 수행하여 출력 불일치가 피할 수 없게 됩니다.

- **Technical Details**: 제안된 방법인 RCPU(Rotation-Constrained Parameter Update)는 가지치기 후 출력의 규범(norm) 및 내적(inner product) 구조를 보존하며 오류를 줄이는 데 목표를 둡니다. 이 방법은 보존된 출력과 원본 출력 간의 정렬을 직교 프로크루스테스 문제(Orthogonal Procrustes problem)로 수식화하며, 각 레이어에서 최적의 회전을 추정하여 파라미터를 업데이트합니다. 또한 보존된 구성 요소의 선택이 회전 제약 보상의 효과에 큰 영향을 미치기 때문에, 입력 분산(input variance)을 고려한 간단한 가지치기 점수를 설계합니다.

- **Performance Highlights**: 실험에서는 제안된 방법을 LLaMA-7B 모델에 적용하여 WikiText-2 및 여러 언어 이해 벤치마크에서 평가하였습니다. 그 결과, 기존 기준선에 비해 일관된 향상을 보여주었으며, 퍼플렉서티(perplexity)와 과제 정확도(task accuracy) 모두에서 개선을 이루었습니다. 이 연구는 기존의 방법들에 비해 정확도를 유지하면서도 가지치기 후 평가에서의 성능을 향상시키는 기여를 하고 있습니다.



### Drift No More? Context Equilibria in Multi-Turn LLM Interactions (https://arxiv.org/abs/2510.07777)
- **What's New**: 이번 연구는 다중 대화에서 발생하는 'context drift'(맥락 드리프트)를 분석하며, 이를 이해하기 위한 간단한 동적 프레임워크를 제안합니다. 연구자들은 KL divergence(쿨백-라이블러 발산)를 토대로 내려진 결과를 통해 맥락 드리프트가 통제 가능한 평형 현상으로 이해될 수 있음을 보여줍니다.

- **Technical Details**: 연구는 특정 테스트 모델과 목표 일관적인 기준 정책 간의 'contextual divergence'(맥락 발산)를 정량화하며, 여러 테스트 설정에서 드리프트의 동적 변화를 분석합니다. 주목할 점은 드리프트가 무한히 커지는 것이 아니라 특정 안정 상태에 수렴하는 경향이 있다는 것입니다.

- **Performance Highlights**: 실험 결과, 목표 리마인더와 같은 간단한 개입이 모델 사이의 드리프트를 줄이는 데 효과적임을 보여줍니다. 이 연구는 다중 대화에서 드리프트가 자연스러운 상태가 아니라 조절 가능한 현상임을 제공하여, 향후 맥락 드리프트 개선 방향에 대한 기초를 제공합니다.



### Instance Relation Learning Network with Label Knowledge Propagation for Few-shot Multi-label Intent Detection (https://arxiv.org/abs/2510.07776)
- **What's New**: 이번 연구에서는 Few-shot Multi-label Intent Detection (MID) 문제를 해결하기 위해 다중 라벨 공동 학습 방법을 제안합니다. 기존의 두 단계 파이프라인 방식에서 벗어나, 인스턴스 관계 학습 네트워크를 통해 레이블 지식 전파(label knowledge propagation)를 수행하여 오류 전파를 제거합니다. 이 방법은 지원 세트와 쿼리 세트 간의 강한 상호작용을 모델링함으로써 멀티 레이블 예측의 성능을 향상시킵니다.

- **Technical Details**: 제안하는 방법은 메타 학습(meta-learning) 틀 안에서 Few-shot MID 작업을 정의하며, 각 메타 작업은 지원 세트와 쿼리 세트를 포함합니다. 지원 세트는 여러 클래스와 인스턴스를 가지며, 각 인스턴스에 대해 다수의 레이블을 할당할 수 있습니다. 또한, 듀얼 관계 강화 손실(dual relation-enhanced loss)을 설계하여 지원 및 쿼리 수준의 관계 강도를 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 강력한 베이스라인 대비 평균 9.54% AUC 및 11.19% Macro-F1 향상을 기록하며, 특히 1-shot 설정에서 두드러진 성능 향상을 보여주었습니다. 이는 저자원(dialogue domain) 환경에서도 효과적으로 작용함을 시사합니다.



### The Unintended Trade-off of AI Alignment:Balancing Hallucination Mitigation and Safety in LLMs (https://arxiv.org/abs/2510.07775)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)에서의 진실성(truthfulness) 향상 노력의 부작용을 다룹니다. 특히, 진실성을 증대시키려는 시도가 어떻게 안전성 정렬(safety alignment)에 부정적인 영향을 미칠 수 있는지를 탐구합니다. 또한, 모델의 환각(hallucination) 정보와 거부(refusal) 정보를 동시에 인코딩하는 내부 요소들로 인해 나타나는 트레이드오프를 분석합니다.

- **Technical Details**: 모델의 거부 행동이 약화되는 현상을 줄이기 위해, 저자는 희소 오토인코더(sparse autoencoders)를 사용하여 환각과 거부 관련 특징을 분리하는 방법을 제안합니다. 또한, 서브스페이스 정교화(subspace orthogonalization)를 통해 훈련 과정에서 거부 행동을 유지함으로써, 진실성과 안전성 간의 위험한 균형을 관리합니다. 이 접근 방식을 통해, 환각을 억제하면서도 작업 효용(task utility)을 보존할 수 있습니다.

- **Performance Highlights**: 저자들은 commonsense reasoning 작업과 AdvBench 및 StrongReject와 같은 위험한 벤치마크에서 제안된 방법을 평가합니다. 결과적으로, 이 방법은 거부 행동을 유지하면서도 진실성을 높이는 데 성공했음을 보여주었고, 안전성과 진실성 간의 트레이드오프를 완화하는 데 기여했습니다.



### Curing Miracle Steps in LLM Mathematical Reasoning with Rubric Rewards (https://arxiv.org/abs/2510.07774)
Comments:
          25 pages, 11 figures, 6 Tables

- **What's New**: 이 논문에서는 수학적 추론을 위한 대형 언어 모델이 결과 기반 보상(outcome-based rewards)으로 교육되어, 모델의 추론 능력이 과대 평가되는 문제를 다룹니다. 연구자들은 'Miracle Steps'와 같은 잘못된 추론 과정을 통한 정답 도달 사례를 체계적으로 분석하여 이로 인해 발생하는 오류 유형을 분류합니다. 이를 극복하기 위해 문제별 루브릭(rubric)에 기반한 새로운 보상 모델인 Rubric Reward Model (RRM)을 제시합니다.

- **Technical Details**: RRM은 전통적인 결과 보상 모델(outcome reward models)과는 달리, 문제 해결 과정 전체를 평가하며, 각 단계에서의 논리적 오류를 처벌하는 방식으로 작용합니다. 이 모델은 각 문제에 맞는 루브릭을 기반으로 하여, 이전 모델들이 간과했던 깊은 논리적 결함까지 알기 쉽게 검사할 수 있도록 합니다. RRM을 통합한 강화 학습(reinforcement learning) 시스템은 기존의 추상적인 보상 신호 대신, 각 고유한 문제에 대한 구체적이고 정밀한 보상을 제공합니다.

- **Performance Highlights**: RRM 기반의 훈련은 수학적 벤치마크에서 일관되게 성능을 향상시키며, 특히 AIME2024에서 Verified Pass@1024 점수를 26.7%에서 62.6%로 끌어올렸습니다. 연구 결과, 잘못된 논리적 추론에서 기인한 Miracle Steps의 발생률을 71%까지 줄이며, 모델의 신뢰성을 크게 높였습니다. 이러한 연구는 모델의 정확성뿐만 아니라 분별력을 높이기 위해서도 문제 해결 과정을 보상하는 것이 중요하다는 점을 보여줍니다.



### ToolLibGen: Scalable Automatic Tool Creation and Aggregation for LLM Reasoning (https://arxiv.org/abs/2510.07768)
- **What's New**: 이 논문에서는 외부 도구와 함께 사용되는 대형 언어 모델(LLMs)의 향상된 성능을 강조합니다. 특히, 도구가 부족한 도메인에서의 복잡한 추론 작업을 위한 자동화된 도구 생성 방안을 제안합니다. 'ToolLibGen'이라는 파이프라인을 통해 비정형 도구 컬렉션을 구조화된 도구 라이브러리로 리팩토링하는 체계적인 접근 방식을 차별화하여 제공합니다.

- **Technical Details**: ToolLibGen는 질문-응답(QA) 데이터셋으로부터 Python 라이브러리를 구축하는 방법론을 제시합니다. 이 시스템은 질문-Chain-of-Thought(Cot) 쌍으로부터 질문별 도구를 추출하고, 이를 클러스터링하여 기능적으로 관련된 도구를 집약하여 하나의 Python 클래스와 보조 함수로 통합합니다. 또한, 이 과정은 코드 에이전트와 리뷰 에이전트 간의 피드백 루프를 통해 진행되어 각 도구 집합의 기능 유지 및 최적화를 보장합니다.

- **Performance Highlights**: 실험 결과, ToolLibGen을 적용한 구조적 도구 라이브러리는 기존의 비체계적 도구 컬렉션보다 평균적으로 5%에서 10% 이상 높은 성공률을 보였습니다. 더불어 우리의 방법은 전혀 새로운 질문에 대해서도 3% 이상의 정확도 향상을 보여주어, 기능적으로 관련된 질문-specific 도구들을 통합함으로써 도구 검색과 전반적인 추론 성능을 개선하는 능력을 입증하였습니다.



### Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers (https://arxiv.org/abs/2510.07761)
Comments:
          In-progress Preprint

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 다중 선택 질문 응답(MCQA) 능력에 대해 다룹니다. LLM이 질문을 포함하지 않고도 선택지만으로 성공하는 경우가 있음을 밝혀냈습니다. 이러한 부분 입력 성공이 항상 문제라는 주장을 도전하며, 이유 추적(reasoning traces)이 실제로 이러한 전략들이 얕은 것인지에 대해 논의합니다.

- **Technical Details**: 연구진은 LLM을 통해 MCQ를 완전한 입력과 선택지만으로 해결하는 실험을 진행했습니다. 테스트 시, 완전 입력에서의 이유 추적은 정확성을 높였으며, 선택지만으로도 절반 정도의 경우 정확도를 높일 수 있었습니다. 비록 이러한 성공이 얕은 전략 때문일 수 있지만, 이유 추적의 길이는 선택만으로 이뤄진 성공에 거의 영향을 주지 않는 것을 확인했습니다.

- **Performance Highlights**: 연구 결과, 이유 추적이 신뢰성 테스트를 통과한 후, LLM은 누락된 질문을 추론하는 것과 같은 덜 문제가 되는 전략을 사용한다는 것을 밝혔습니다. 결론적으로, 부분 입력 성공이 항상 결함이라고 주장하는 것에 이의를 제기하며, 이유 추적이 문제 데이터와 덜 문제성이 있는 추론을 구별할 수 있는 방법을 제시했습니다.



### Parallel Test-Time Scaling for Latent Reasoning Models (https://arxiv.org/abs/2510.07745)
- **What's New**: 이 논문은 잠재적 추론(latent reasoning) 모델에 대해 병렬 테스트 시간 스케일링(parallel test-time scaling, TTS)을 구현함으로써 이를 개선할 가능성을 탐구합니다. 특히, 토큰 기반 모델이 사용하는 샘플링 및 집계 메커니즘의 한계를 극복하기 위해 두 가지 확률적 샘플링 전략인 모나코 카로우 드롭아웃(Monte Carlo Dropout)과 추가 가우시안 노이즈(Additive Gaussian Noise)를 도입합니다. 또한 잠재 보상 모델(Latent Reward Model, LatentRM)을 설계하여 각 추론 단계에서 추론 과정을 평가하고 안내하는 방법을 제안합니다.

- **Technical Details**: 문서는 병렬 TTS가 잠재적 추론 모델에 효과적으로 적용될 수 있도록 샘플링 및 집계 메커니즘을 재구성하는 방법을 제시합니다. 첫째, 모나코 카로우 드롭아웃과 추가 가우시안 노이즈를 사용하여 잠재 공간에서 추론 경로를 다양하게 샘플링하는 방법을 설명합니다. 둘째, LatentRM을 통해 잠재적 경로를 평가하고 안내하는 세밀한 방법론을 개발하여 성능 향상을 이끌어냅니다.

- **Performance Highlights**: 상세한 실험과 시각적 분석 결과 두 가지 샘플링 전략 모두 계산량이 증가함에 따라 효과적으로 확장됨을 보여줍니다. 모나코 드롭아웃은 비정상적인 솔루션으로의 구조화된 확장을 촉진하며, 추가 가우시안 노이즈는 보다 넓고 동등한 탐색을 유도하여 다양성을 풍부하게 합니다. 최종적으로 LatentRM을 통해 다양한 계산 예산 하에 일관된 성능 향상이 이루어졌습니다.



### OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignmen (https://arxiv.org/abs/2510.07743)
Comments:
          The first two authors contributed equally

- **What's New**: 본 논문에서는 OpenRubrics라는 대규모 (prompt, rubric) 쌍의 컬렉션을 소개합니다. 이는 고품질의 rubrics를 생산하기 위한 훈련 데이터셋으로 활용되며, 다양한 평가 기준을 구조화된 자연어로 제공하여 인간 피드백에 대한 더 나은 반응 모델을 가능하게 합니다. 또한, Contrastive Rubric Generation (CRG)이라는 방법을 통해 선호하는 응답과 그렇지 않은 응답을 대비시켜 평가 신호의 질을 높입니다.

- **Technical Details**: 보상을 위한 rubrics는 응답 품질을 측정하기 위한 구조화된 평가 기준이며, 각 기준은 사실 정확성(factual correctness), 추론(soundness), 스타일 등 여러 차원으로 나뉩니다. OpenRubrics 구축을 위해 다양한 공개 데이터 세트를 통합하여 여러 도메인과 작업에 걸쳐 일반화 가능하고 신뢰할 수 있는 rubrics를 생성하는 것이 초점입니다. 이 논문에서는 두 가지 주요 유형의 rubrics를 구분하고, CRG를 통해 더 포괄적이고 차별화된 평가 기준을 생성하는 방법론을 제안합니다.

- **Performance Highlights**: Rubric-RM이라고 불리는 우리의 평가 모델은 여러 보상 모델링 벤치마크에서 6.8%의 성능 향상을 보여줍니다. 이 모델은 정책 모델에 통합될 경우, 지침 따르기 및 생물 의학 벤치마크에서 평균 2.9%의 성능 향상을 이루어냅니다. 연구 결과에 따르면, rubrics는 비용이 높은 인간 평가와 자동화된 보상 모델링 사이의 격차를 줄이는 데 도움이 됩니다.



### ToolExpander: Extending the Frontiers of Tool-Using Reinforcement Learning to Weak LLMs (https://arxiv.org/abs/2510.07737)
- **What's New**: 이 논문은 GRPO(Group Relative Policy Optimization)를 기반으로 한 툴 확장 도구인 ToolExpander를 제안합니다. 이 프레임워크는 작은 스케일의 대형 언어 모델(LLMs)에 대해 효율적인 reinforcement learning을 가능하게 합니다. ToolExpander는 두 가지 주요 혁신 기능으로, 동적 다중 라운드 하드 샘플링(Dynamic Multi-Round Hard Sampling)과 자기 예시적 사고(Self-Exemplifying Thinking)를 포함합니다.

- **Technical Details**: 동적 다중 라운드 하드 샘플링은 정확한 출력을 생성하지 못하는 어려운 샘플을 고품질의 Few-shot 데모로 대체하며, 학습률을 점진적으로 감소시켜 학습의 진동을 줄이는 전략을 사용합니다. 자기 예시적 사고 프레임워크는 KL 발산을 제거하고 조정된 클리핑 계수를 포함하여 모델이 최소한의 추가 보상(0.01)으로 자율적으로 Few-shot 예시를 생성하고 분석하도록 장려합니다.

- **Performance Highlights**: 실험 결과 ToolExpander는 LLM의 툴 활용 능력을 크게 향상시켰으며, 특히 1.5B 파라미터 모델에서 학습 안정성을 확보하고 성능을 향상시켰습니다. 이 모델은 APIBank에서 81.76%의 정확도를 달성하여 전통적인 GRPO 모델보다 월등한 성능을 보였습니다. 추가적으로, 자기 예시적 사고 메커니즘은 모델이 복잡한 작업을 효율적으로 처리할 수 있도록 돕습니다.



### Multilingual Knowledge Graph Completion via Efficient Multilingual Knowledge Sharing (https://arxiv.org/abs/2510.07736)
Comments:
          EMNLP 2025, Findings, Long Paper

- **What's New**: 본 논문에서는 다국어 지식 그래프 완성(MKGC)을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 다국어 공유 지식을 활용하여 성능을 크게 향상시키는 두 가지 구성 요소인 Knowledge-level Grouped Mixture of Experts (KL-GMoE)와 Iterative Entity Reranking (IER)을 포함합니다. 실험 결과, 이 프레임워크는 최신 MKGC 방법과 비교하여 Hits@1, Hits@3, Hits@10에서 각각 5.47%, 3.27%, 1.01% 향상을 달성했습니다.

- **Technical Details**: KL-GMoE는 지식 수준의 전문가 라우팅 메커니즘을 도입하고, 그룹 기반 Mixture-of-Experts (MoE) 아키텍처를 통해 지식 파편화를 완화하고 다국어 공유 지식을 효과적으로 캡처할 수 있습니다. IER는 LLM의 훈련 목표 및 디코딩 전략을 수정하여 다국어 공유 지식을 반복적으로 개선하고 활용하는 능력을 극대화합니다. 모델 아키텍처와 작업 패러다임의 불일치를 해결하기 위해 각 채널이 의미적으로 유사한 정보에 집중할 수 있도록 전용 지식 채널 수를 늘리는 방법도 제안합니다.

- **Performance Highlights**: KL-GMoE와 IER을 결합한 프레임워크는 다양한 실험 분석을 통해 특히 보이지 않거나 불균형한 언어 환경에서의 지식 공유 특성을 밝혔습니다. 최종 실험 결과는 이 새로운 접근법이 지식 그래프의 완전성을 높이는 데 있어 상당한 잠재력을 가지며, 기존 MKGC 방법의 한계를 극복할 수 있음을 나타냅니다. 추가적으로, 우리는 실험에 사용된 mKG 데이터셋과 코드를 공개하여 연구의 투명성을 높였습니다.



### SUBQRAG: sub-question driven dynamic graph rag (https://arxiv.org/abs/2510.07718)
Comments:
          5 pages, 1 figure

- **What's New**: 새로운 연구인 SubQRAG는 복잡한 질문을 순차적이고 검증 가능한 하위 질문으로 분해하여 다단계 추론을 지원하는 프레임워크입니다. 이 시스템은 그래프에서 관련 정보를 동적으로 검색하고 부족할 경우 원본 문서에서 새로운 정보를 실시간으로 추출하여 그래프를 확장합니다. SubQRAG는 모든 하위 질문에서 사용된 트리플을 "그래프 메모리"에 집계하여 최종 답변 생성을 위한 구조적이고 추적 가능한 증거 경로를 제공합니다.

- **Technical Details**: SubQRAG는 네 가지 주요 단계로 작동하며, 첫 번째 단계는 LLM을 활용하여 원시 텍스트에서 구조적 지식 트리플을 추출하는 오프라인 인덱싱입니다. 다음 단계에서는 질문을 하위 질문으로 분해하고 일관성을 유지하기 위해 이전 질문의 답변을 통합하여 재작성합니다. 세 번째 단계에서 SubQRAG는 하위 질문에 대한 지원 트리플을 검색하고, 필요시 원본 문서에서 새로운 트리플을 동적으로 업데이트하여 기존 그래프를 보강합니다.

- **Performance Highlights**: SubQRAG는 세 가지 다단계 질문 응답 벤치마크에서 일관되고 중요한 향상을 보여주었습니다. 실험 결과, 정확한 일치 점수(Exact Match score)에서 특히 우수한 성능을 발휘했습니다. 이 프레임워크는 기존의 그래프 기반 방법의 한계를 극복하고, 더 깊은 추론을 가능하게 하는 동시에 정확성을 높입니다.



### MemWeaver: A Hierarchical Memory from Textual Interactive Behaviors for Personalized Generation (https://arxiv.org/abs/2510.07713)
Comments:
          12 pages, 8 figures

- **What's New**: 이번 연구에서는 MemWeaver라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 사용자의 텍스트 역사(history)를 계층적 메모리(hierarchical memory)로 엮어내어 깊이 있는 개인화를 가능하게 합니다. MemWeaver는 사용자의 관심의 시간적 진화 및 활동 간의 의미적 관계를 포착할 수 있는 능력을 핵심 혁신으로 하고 있습니다.

- **Technical Details**: MemWeaver는 사용자의 상호작용 역사에서 두 가지 보완적인 메모리 구성 요소를 구축합니다: 행동 메모리(behavioral memory)와 인지 메모리(cognitive memory)입니다. 행동 메모리는 구체적인 사용자 행동을 캡처하고, 인지 메모리는 장기적인 선호를 나타내며, 이 두 가지 메모리는 사용자에 대한 통합된 표현을 제공합니다. 이 메모리는 사용자가 연결된 행동 내역을 통해 시뮬레이션된 인간 기억의 연상 재현 과정을 따릅니다.

- **Performance Highlights**: Language Model Personalization (LaMP) 벤치마크에서의 철저한 실험을 통해 MemWeaver의 효율성을 입증하였습니다. 연구 결과 이 프레임워크는 강력한 베이스라인을 초월하는 성능을 보여주며, 행동 메모리와 인지 메모리 간의 시너지 효과가 깊이 있는 사용자 정렬을 달성하는 데 필수적임을 확인했습니다. 또한, 메모리 업데이트 메커니즘의 효율성과 확장성을 분석하여 부분적 메모리 재구성으로도상당한 성능을 달성함을 나타냈습니다.



### Causality Guided Representation Learning for Cross-Style Hate Speech Detection (https://arxiv.org/abs/2510.07707)
- **What's New**: 최근 온라인 증오 발언(hate speech)의 확산은 웹의 조화에 심각한 위협이 되고 있습니다. 기존의 증오 발언 탐지 모델은 주로 표면적인 언어적 단서를 기반으로 작동하고 있어 다양한 스타일 변형에 효과적으로 일반화하지 못하는 문제가 있습니다. 이러한 배경을 토대로, 본 논문에서 제안한 CADET는 causal representation learning 프레임워크로, 증오 발언을 해석 가능한 잠재 인자로 분해하여 진정한 증오 의도를 표면적인 언어적 단서와 분리합니다.

- **Technical Details**: CADET는 증오 발언 생성 과정을 모델링하기 위해 causal graph(인과 그래프)를 사용하여, 상황적 환경(contextual environment), 작성자 동기(creator motivation), 대상(target), 스타일(style) 등 핵심 요소들을 포함합니다. 논문의 접근 방식에서는 모델이 각 잠재 인자를 구분하고, 상황적 환경이 미치는 혼란 요인(confounder)을 제어하여 보다 견고한 탐지를 가능하게 합니다. 이론적으로, CADET는 잠재 공간(latent space) 내에서 스타일을 조정함으로써 반사실적 추론(counterfactual reasoning)이 가능하다는 점이 강조됩니다.

- **Performance Highlights**: CADET는 다양한 도전 과제를 중심으로 한 평가에서 우수한 성능을 보여주었으며, cross-style generalization 작업에서 평균 macro-F1이 0.81에 달합니다. 이는 기존의 최첨단 방법에 비해 13%의 상대적 개선을 이룬 것입니다. 각 구성요소의 중요한 역할이 입증되었고, 인과 기반 설계의 효과가 숨겨진 것처럼 분석되었습니다. 이 연구는 일반화된 증오 발언 탐지를 개선하기 위한 인과적인 분리(causal disentanglement)의 가능성을 강조하며, 안전하고 책임감 있는 온라인 환경 조성을 위한 실용적인 함의를 제공합니다.



### Large Language Models Meet Virtual Cell: A Survey (https://arxiv.org/abs/2510.07706)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 '가상 세포'를 개발하여 세포 생물학을 혁신적으로 변화시키고 있다는 점을 강조합니다. 이러한 가상 세포 시스템은 세포 상태와 행동을 예측하고 이해하는 데 중요한 역할을 하고 있습니다. 이 논문은 LLMs의 다양한 활용 방법을 단순화된 두 가지 패러다임—Oracle로서의 LLM과 에이전트로서의 LLM—으로 정리하여, 생물학적 모델링에 대한 통합된 분류 체계를 제안합니다.

- **Technical Details**: 연구는 LLM을 가상 세포의 예측 엔진으로 활용하여 세포의 내적 상태와 동역학을 직접 모델링합니다. DNA, RNA 및 단일 세포 전사체 프로필과 같은 생물학적 서열을 활용하여 LLM이 원시 데이터에서 세포 구성 요소 및 상호작용의 표현을 학습한다는 점이 중요합니다. 연구는 세 가지 핵심 작업인 세포 표현(Cellular Representation), 교란 예측(Perturbation Prediction), 유전자 기능 및 조절 예측(Gene Function & Regulation Prediction)을 지금까지 논의해온 방식으로 정리합니다.

- **Performance Highlights**: LLMs는 셀 시스템을 모델링하는 데 있어 미리 정의된 데이터셋과 벤치마크를 활용하여 탐색과 실험을 진행하는 에이전트로서의 역할을 수행하고 있습니다. 연구는 LLM을 통한 다양한 세포 상호작용 모델링의 적용 예를 제공할 뿐 아니라, 대량의 생물학적 데이터를 이용한 가상 세포 시스템 개발의 기회를 강조합니다. 이와 같은 접근은 세포와 유전자의 복잡한 상호작용을 실시간으로 이해하고, 개인화된 의학의 길을 열 수 있는 잠재력을 가지고 있음을 보여줍니다.



### Stress-Testing Model Specs Reveals Character Differences among Language Models (https://arxiv.org/abs/2510.07686)
- **What's New**: 본 논문에서는 AI 헌법(AI constitutions)과 모델 사양(model specifications)이 대규모 언어 모델(Large Language Models, LLMs)의 행동 지침 및 윤리 원칙을 설정하는 데 미치는 영향을 조사합니다. 기존 모델 사양의 문제점을 정확하게 파악하기 위해 시스템적 스트레스 테스트 방법론을 제시하였으며, 이는 원칙 간의 모순과 해석의 모호성을 자동으로 식별하는 데 초점을 맞추었습니다. 이 연구는 다양한 가치 기반 원칙들을 요구하는 시나리오를 생성하여 모델들이 충돌 상황을 어떻게 처리하는지를 분석합니다.

- **Technical Details**: 연구팀은 3,307개의 세분화된 가치를 포함하는 세분화된 분류법을 활용하여 300,000개 이상의 다양한 쿼리 시나리오를 생성합니다. 이 시나리오들은 상충하는 원칙 간의 명시적인 거래를 강요하는 구조를 가지고 있으며, 이는 최첨단 LLM인 Anthropic, OpenAI, Google, xAI 모델들의 응답을 분석하는 데 사용됩니다. 특히, 각 모델의 응답에서의 불일치를 측정하기 위해 가치 분류 점수(value classification scores)를 사용하여 응답 간의 차이를 정량화합니다.

- **Performance Highlights**: 실증 분석 결과, 300,000개 시나리오 중 22만 개 이상에서 최소 한 쌍의 LLM 간에 상당한 불일치가 나타났으며, 7만 개 이상에서 대부분의 모델 간 행동의 차이를 식별했습니다. 높은 불일치는 사양 위반(specification violations)과 함께 직접적으로 연결되어 있으며, 이는 모델의 행동에 대한 명확한 시사점을 제공합니다. 또한, 모델들은 명확한 지침이 부족할 때 체계적인 가치 우선순위 패턴을 나타내어 이들 간의 차이를 드러냅니다.



### Textual Entailment and Token Probability as Bias Evaluation Metrics (https://arxiv.org/abs/2510.07662)
Comments:
          16 pages, 9 figures, under ARR review

- **What's New**: 이 연구에서는 언어 모델의 사회적 편향을 측정하는 새로운 방법으로 자연어 추론(NLI)을 제안합니다. 기존의 토큰 확률(TP) 기반 방법과는 달리, NLI는 실제 언어 모델 사용 사례와 더 가까운 접근 방식을 제공한다고 주장합니다. 이 방법을 사용하여 기존의 TP 지표와 NLI 지표 간의 차이를 분석하고, NLI가 더 효과적으로 '과소 편향' 사례를 감지할 수 있음을 보였습니다.

- **Technical Details**: 본 연구에서는 WinoQueer(NLI) 데이터셋을 생성하여 TP와 NLI의 편향 평가를 비교하였습니다. WQ-NLI 데이터셋은 46,036개의 문장 쌍을 포함하며, LGBTQ+와 관련된 사례를 중심으로 구성되어 있습니다. 텍스트 전제-가설 쌍을 사용하여 기존 TP 방법론의 한계를 극복하고, 더 현실적이고 포괄적인 편향 평가를 가능하게 합니다.

- **Performance Highlights**: 연구 결과 NLI와 TP의 편향 평가 결과는 상이하였으며, 특히 NLI가 편향의 세부적인 분류에서 더 민감하게 작용하는 것으로 나타났습니다. NLI 지표는 역습이 발생할 가능성이 더 높아 유용한 대안으로 보이지만, 단어 구성이 중요한 변수로 작용하기 때문에 더 유연함을 필요로 합니다. 따라서 편향의 포괄적인 평가를 위해 TP, NLI 및 추가 평가 방법을 조합하는 것이 권장됩니다.



### OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inferenc (https://arxiv.org/abs/2510.07651)
- **What's New**: 이번 연구에서는 Optimal Brain Cache (OBCache)라는 새로운 프레임워크를 제안합니다. OBCache는 키-값 (KV) 캐시 제거를 계층별 구조적 프루닝 문제로 공식화하여 메모리 오버헤드를 줄입니다. 이는 특정 토큰의 중요도를 평가할 때, 주의 가중치만이 아닌 주의 출력의 변화도 고려하는 접근 방식을 취합니다.

- **Technical Details**: OBCache는 Optimal Brain Damage (OBD) 이론을 기반으로 하여, 각 토큰의 제거가 주의 출력에 미치는 영향을 측정합니다. 이 프레임워크는 고립된 키, 고립된 값, 그리고 키-값 쌍의 세 가지 프루닝 단위를 정의하고, 이를 통해 캐시 제거 및 유지 결정을 내리기 위한 폐쇄 형태의 점수들을 도출합니다. OBCache는 기존의 주의 기반 점수 산정 방식과 비교하여 보다 풍부하고 정확한 신호를 제공합니다.

- **Performance Highlights**: OBCache는 LLaMA 및 Qwen 모델에서 다양한 긴 컨텍스트 작업에 대해 실험을 통해 성능 향상을 입증하였습니다. 특히, OBCache의 점수를 기존의 방법들과 결합함으로써 전체적인 성능이 개선되었고, 메모리 소모를 줄이면서도 긴 컨텍스트 추론 성능을 향상시키는 데 기여했습니다.



### Banking Done Right: Redefining Retail Banking with Language-Centric AI (https://arxiv.org/abs/2510.07645)
Comments:
          Accepted at EMNLP2025 Industry Track

- **What's New**: 이 논문은 자연어 대화를 통해 고객이 주요 금융 거래를 수행할 수 있도록 지원하는 Ryt AI라는 LLM 네이티브 에이전트 프레임워크를 소개합니다. 이는 대화형 AI가 주요 은행 인터페이스로 기능하는 최초의 글로벌 규제 승인 배포를 의미하며, 기존의 보조적 역할에 한정된 기술들과는 차별화됩니다. Ryt AI는 내부적으로 개발된 ILMU라는 폐쇄형 LLM의 지원을 받아 다중 스크린의 고정형 워크플로우를 단일 대화 인터페이스로 전환합니다.

- **Technical Details**: Ryt AI는 4개의 LLM 기반 에지전트(Guardrails, Intent, Payment, FAQ)로 구성되어 있으며, 각 에이전트는 ILMU에 특정 작업을 위한 LoRA 어댑터를 부착합니다. 이러한 구조는 은행의 인프라 내에서 호스팅되어 최소한의 오버헤드로 일관된 동작을 보장합니다. 또한, 결정론적 가드레일, '인 더 루프' 인간 확인, 상태 비저장 감사 아키텍처를 통해 보안 및 규정 준수를 위한 다층적인 방어를 제공합니다.

- **Performance Highlights**: Ryt AI는 규제 승인을 받은 최초의 LLM 기반 에이전트 시스템으로, 디지털 뱅크의 핵심 트랜잭션을 직접 실행할 수 있도록 돕습니다. 이 시스템은 사용자가 자연어로 의도를 표현하고, 시스템이 이를 조합, 검증 및 실행하는 과정을 통해 사용자 경험을 혁신적으로 개선합니다. 또한, 다국어 지원과 규제 요구를 맞춘 지역화된 적응을 통해 말레이시아 시장에 최적화된 솔루션을 제공합니다.



### Role-Conditioned Refusals: Evaluating Access Control Reasoning in Large Language Models (https://arxiv.org/abs/2510.07642)
Comments:
          8 pages + Appendix

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 액세스 제어 정책을 준수할 수 있는지를 검토하며, 특히 역할 기반 거부의 정확성에 주목하였다. 새로운 데이터 세트를 생성하여 Spider와 BIRD 데이터 세트를 확장하고, PostgreSQL 역할 기반 정책을 적용하였다. 연구는 거부의 정확성 향상 및 허위 허가를 줄이기 위해 세 가지 시스템 설계를 비교하였다.

- **Technical Details**: 접근 제어는 데이터베이스 및 운영 체제에서 권한을 관리하는 기본 원칙으로 오랫동안 자리 잡아 왔으며, RBAC(역할 기반 접근 제어)는 그 중 하나의 중요한 프레임워크이다. 본 연구에서는 Zero-shot 및 Few-shot prompting, 두 단계의 Generator-Verifier 파이프라인, 그리고 LoRA 기반의 미세 조정을 통해 접근 제어 정책을 준수하는 LLM의 능력을 평가하기 위한 통합된 프레임워크를 개발하였다. 이 시스템은 다양한 모델 패밀리를 포함하여 접근 거부와 응답 생성의 안전성 및 유용성의 균형을 평가한다.

- **Performance Highlights**: 연구 결과, 두 단계의 파이프라인을 사용하는 방안이 명시적인 검증을 통해 거부의 정밀성을 높이고 잘못된 허가를 줄이는 데 효과적임을 확인하였다. 한편, LoRA 미세 조정은 일반 성능을 해치지 않으면서도 권한 인식 능력을 향상시키는 데 기여하였으며, 복잡한 정책은 모든 시스템의 신뢰성을 일관되게 감소시키는 경향이 있었다. 전반적으로, 본 연구는 LLM이 역할 기반 정책을 준수할 수 있도록 하는 접근 방식에 대한 귀중한 통찰을 제공한다.



### Toward Reliable Clinical Coding with Language Models: Verification and Lightweight Adaptation (https://arxiv.org/abs/2510.07629)
- **What's New**: 이 논문은 정확한 임상 코드 작성의 중요성을 강조하며, 기존의 LLM(대형 언어 모델)들이 과제로 제기된 문제를 잘 해결하지 못함을 지적합니다. 연구팀은 계층적으로 근접한 잘못된 예측에서 발생하는 오류가 LLM 실패의 상당 부분을 차지한다는 분석 결과를 발표했습니다. 이를 해결하기 위한 새로운 접근법으로, 임상 코드 검증을 독립 작업과 파이프라인 구성 요소로 도입하였습니다.

- **Technical Details**: 논문에서 제안하는 방법 중 하나는 prompt engineering과 소규모 fine-tuning을 포함한 경량 개입을 통해 LLM의 정확성을 높이는 것입니다. 연구팀은 임상 코드 검증 파이프라인을 도입하여 ICD-10-CM의 계층 구조를 활용하여 LLM의 예측을 개선하는 방법을 설명합니다. 또한, 새로운 더블 전문가 주석된 벤치마크 데이터셋을 출시하여 모든 외래 진료 노트를 포함시키고, 이를 통해 기존 데이터셋의 한계를 극복하고자 하였습니다.

- **Performance Highlights**: 논문에서 언급된 성과 중 하나는 기존 LLM의 성능을 향상시키기 위한 다양한 프롬프트 구조의 평가 결과입니다. 프롬프트 공학과 fine-tuning 기법들이 최종적으로 LLM의 예측 정확도를 16 F1점까지 향상시키며, 특히 임상 코딩에서 신뢰를 높이는 효과적인 방법으로 작용합니다. 이러한 연구를 통해 임상 코딩의 정확성을 높이고 의료 시스템의 실제 문제를 실질적으로 해결할 수 있는 가능성을 제시합니다.



### Vocabulary embeddings organize linguistic structure early in language model training (https://arxiv.org/abs/2510.07613)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 입력 임베딩 벡터의 기하학적 구조가 어떻게 구성되며 훈련 과정 중 이 구조가 어떻게 변화하는지를 탐구합니다. 우리는 두 개의 오픈소스 모델인 Pythia 12B와 OLMo 7B에 대한 실험을 통해 입력과 출력 임베딩 간의 기하학적 구조와 의미론적(semantic), 구문론적(syntactic), 빈도 기반(frequency-based) 메트릭 간의 상관관계를 분석하였습니다.

- **Technical Details**: 연구는 representational similarity analysis를 사용하여, 의미론적 및 구문론적 특징과 높은 상관 관계를 갖는 입력 임베딩의 기하학적 구조에 대한 실험을 수행하였습니다. 훈련 초기에, 고빈도(high-frequency) 및 기능어(function words)임베딩은 일반적으로 사전 훈련된 초기 구조와 더 빠르게 수렴하는 경향이 있음을 발견했습니다. 반면, 저빈도(low-frequency) 단어는 초기 편향(alignment)을 어느 정도 유지하며, 최종 벡터로의 수렴이 더디게 진행됩니다.

- **Performance Highlights**: 본 연구에서는 훈련 과정 동안 어휘(어휘) 임베딩의 기하학적 구조가 어떻게 언어 구조를 중심으로 조직되는지를 보여주는 동적인 경로를 제시합니다. 또한, 단어의 빈도와 기능이 임베딩의 수렴 속도에 미치는 미묘한 역할을 강조하며, 이러한 결과는 모델 훈련 동안 특정 기능 향상을 촉진하기 위한 어휘 기하학의 진화에 대한 심도 있는 연구의 필요성을 제기합니다.



### IASC: Interactive Agentic System for ConLangs (https://arxiv.org/abs/2510.07591)
Comments:
          Initial draft

- **What's New**: 본 논문에서는 LLMs(Large Language Models)를 사용하여 인공 제작 언어(Constructed Languages)의 개발을 지원하는 시스템을 소개합니다. 이 시스템은 단계적으로 진행되며, 먼저 원하는 언어의 음운론(target phonology)을 생성하는 단계로 시작합니다. 그 다음, 영어 원문에서 목표 언어의 형태 통사적 마크업(morphosyntactic markup)으로 문장을 변환하는 과정을 포함합니다.

- **Technical Details**: 시스템의 작동 방식은 에이전시적(agentic) 접근법을 활용하여 사용자 피드백을 반영하기 때문에 언어의 음운 모델(phonological model)과 형태소(morphemes)를 활용하여 사전을 구축합니다. 또한, 기존 스크립트(예: 라틴 문자 또는 키릴 문자)를 사용하여 언어의 정서법(orthography)을 생성하는 기능이 있습니다. 마지막으로, 시스템은 간단한 문법 핸드북을 작성하고, 추가적인 문장을 목표 언어로 번역하는 기능도 제공합니다.

- **Performance Highlights**: 연구의 목표는 두 가지입니다: 첫째, 이 도구들이 재미있고 유용하게 인공 언어를 만드는 데 도움이 되기를 바랍니다. 둘째, LLMs가 언어와 언어적 개념에 대해 얼마나 알고 이해하는지를 탐구하려고 합니다. 다양한 LLM과 언어적 사양 간에 능력 차이가 있으며, 특히 덜 일반적인 패턴보다 더 일반적인 패턴을 처리하는 것이 상대적으로 용이함을 보였습니다.



### Linguistic Patterns in Pandemic-Related Content: A Comparative Analysis of COVID-19, Constraint, and Monkeypox Datasets (https://arxiv.org/abs/2510.07579)
Comments:
          16 pages

- **What's New**: 이번 연구는 팬데믹 관련 온라인 담론에서 언어가 건강에 대한 허위 정보와 사실적 의사소통을 어떻게 구별하는지를 분석했습니다. COVID-19와 Monkeypox 관련 데이터셋을 통해 허위 정보가 가지는 독특한 언어적 특징을 밝혀냈습니다. 이를 통해 시의적절한 공중 보건 메시지를 전달하기 위한 전략에도 기여할 수 있다는 점이 주목할 만합니다.

- **Technical Details**: 연구는 COVID-19 허위 내러티브, 일반 COVID-19 콘텐츠, Monkeypox 관련 게시물의 세 가지 코퍼스를 분석했습니다. 분석 결과, COVID-19 허위 정보는 낮은 readability 점수와 함께 두 배 이상의 두려움 관련 또는 설득 용어를 포함하고 있으며, Monkeypox 콘텐츠보다 감정적으로 더 표현되는 스타일과 대조적입니다. 이로 인해 허위 정보는 복잡한 수사적 스타일과 감정적 신호를 사용하며, 이러한 조합이 신뢰도를 높일 수 있음을 시사합니다.

- **Performance Highlights**: 연구 결과는 디지털 건강 허위 정보에 대한 언어적 지표를 강조하면서 탐지 노력을 지원하는 데 기여하고 있습니다. 또한, 네트워크 미디어 환경에서의 위기 소통 이론 모델과 공중 보건 메시징 전략에도 중요한 정보를 제공합니다. 그러나 전통적 readability 지수에 의존하거나 정적 집계 분석을 사용하는 등의 한계가 있으므로, 향후 연구에서는 더 넓은 감정 어휘와 플랫폼에 민감한 접근 방식을 통합해야 할 필요성이 제기됩니다.



### Multi-Task Pre-Finetuning of Lightweight Transformer Encoders for Text Classification and NER (https://arxiv.org/abs/2510.07566)
Comments:
          Accepted by EMNLP 2025 Industry Track

- **What's New**: 이 연구에서는 자연어 처리(NLP) 모델의 모바일 플랫폼 배치를 위해 다양한 응용 프로그램에 적응하면서 메모리와 계산 비용을 효율적으로 유지하는 경량 BERT 유사 인코더의 사전 파인튜닝(pre-finetuning) 전략을 조사합니다. 주목할만한 점은 멀티태스크(pre-finetuning) 방법이 서로 상충하는 최적화 신호를 도입하여 전체 성능을 저하시킬 수 있다는 것입니다. 이를 해결하기 위해, 우리는 작업 주요 LoRA 모듈을 기반으로 한 새로운 멀티태스크 사전 파인튜닝 프레임워크를 제안하고 있습니다.

- **Technical Details**: 모바일 애플리케이션은 이메일에서 자동으로 일정 이벤트를 생성하거나 메시지를 기반으로 개인화된 추천을 제공하는 등 여러 자연어 처리 작업을 해결해야 합니다. 본 연구에서는 NER과 텍스트 분류라는 두 가지 기본 NLP 작업 가족을 다루기 위해 사전 파인튜닝 전략을 활용했습니다. 사전 파인튜닝 이후, 최적화된 인코더를 통해 모듈형 어댑터를 사용하여 적용할 수 있는 간단하지만 효과적인 멀티태스크 구조를 구현했습니다.

- **Performance Highlights**: 21개의 하위 작업에 대한 실험 결과, NER에서는 평균 0.8% 향상되고 텍스트 분류에서는 평균 8.8% 향상되었습니다. 이는 제안된 방법이 다양한 모바일 NLP 응용 프로그램에 효과적임을 보여줍니다. 사전 파인튜닝 단계를 통해, 각 작업에 대해 개별적으로 좋은 성과를 올리지만 서로 간섭이 발생하여 이를 해소하기 위한 접근법을 통해 성능 저하를 최소화했습니다.



### Deploying Tiny LVLM Judges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices (https://arxiv.org/abs/2510.07545)
Comments:
          Accepted to the EMNLP 2025 Industry Track

- **What's New**: 이 논문에서는 7억 개의 매개변수를 가진 대형 비전-언어 모델(LVLMs)이 차트 이해 작업에서 자동 심사관 역할을 수행할 수 있음을 보여주지만, 20억 개 이하의 모델들은 성능이 저조하다는 문제를 다룹니다. 이를 해결하기 위해 두 가지 기술적 접근법을 제안하며, 하나는 여러 평가 기준을 통합하여 단일 쿼리로 처리하는 다중 기준 프롬프트(multi-criteria prompting)이고, 다른 하나는 도메인 적응 전이 학습(domain-adaptive transfer learning)입니다. 이 방식들을 통해 ChartJudge라는 경량 평가 모델을 개발해 효율적인 평가를 가능하게 합니다.

- **Technical Details**: 이 연구에서는 ChartJudge-2B라는 20억 개 매개변수 모델을 사용하여 차트 관련 작업에 대한 평가는 미세 조정된 합성 판단 데이터에 기초하고 있습니다. 연구진은 다중 기준 프롬프트 접근법을 통해 평가 비용과 지연 시간을 줄이고, 이 과정을 통해 평가 성능에 있어 강력한 전이 능력을 보여주는 방식을 기술로 제시합니다. 또한, 다양한 차트 유형 및 쿼리 복잡성에 대한 정밀 분석을 통해 모델 크기와 프롬프트 설계 간의 trade-off를 설명합니다.

- **Performance Highlights**: 실험 결과 다중 기준 프롬프트 방식이 기존 7억 개 모델에서의 성능 저하의 주요 원인임을 밝혀냈고, 20억 개 파라미터를 가진 ChartJudge 모델은 효율적이면서도 강력한 성능을 발휘합니다. 특히 ChartJudge-2B는 다양한 차트 데이터셋 간의 지식 전이가 용이하다는 점에서 비용 효율적인 평가자로서의 가능성을 보여줍니다. 마지막으로, 이 연구는 LVLM 감정 모델의 확장이 가능하도록 하는 실제적인 통찰을 제공하며, 코드와 데이터는 공개될 예정입니다.



### OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs (https://arxiv.org/abs/2510.07535)
- **What's New**: 이 논문에서는 자주 사용되는 기존 스펙ulative 디코딩 기법의 한계를 극복하기 위해 새로운 장기 컨텍스트 벤치마크(LongSpecBench)와 모델(OWL)을 소개합니다. OWL은 기존 모델(EAGLE3)에 비해 승인 길이(acceptance length)가 약 5배 더 길어지는 혁신을 통해 성능을 개선했습니다. 이 연구는 일반적인 작업 환경을 보다 잘 반영하여 스펙ulative 디코딩 기법의 실용성을 높이고자 합니다.

- **Technical Details**: OWL에서는 세 가지 주요 혁신을 통해 성능을 크게 향상시킵니다. 첫째, LSTM 기반의 드래프트를 사용하여 마지막 토큰 상태에만 의존하여 다양한 길이에 일반화할 수 있도록 합니다. 둘째, 검증기(verifier)에서 특별한 토큰([SPEC])을 도입하여 드래프트에 더 풍부한 표현력을 제공합니다. 셋째, 트리와 비트리 트리 디코딩 방법을 결합한 하이브리드(hybrid) 알고리즘을 사용하여 보다 높은 승인 길이와 속도를 달성합니다.

- **Performance Highlights**: 실험 결과, OWL의 성능은 EAGLE3를 능가하며 특히 긴 컨텍스트 입력에서 두드러진 성과를 보였습니다. 새로운 벤치마크는 일부 기존 스펙ulative 디코딩 방법이 긴 컨텍스트에서 제대로 작동하지 않는다는 점을 강조합니다. 이 연구의 모든 코드와 데이터셋이 공개되어 향후 연구에 기여할 수 있도록 했습니다.



### ParsTranslit: Truly Versatile Tajik-Farsi Transliteration (https://arxiv.org/abs/2510.07520)
- **What's New**: 이 논문에서는 페르시아어와 타지크어 간의 기계 음역(machin transliteration) 모델의 새로운 접근 방식을 제안합니다. 기존의 연구들이 한정된 데이터셋만을 사용했던 것과 달리, 본 연구는 모든 가용 데이터셋을 통해 훈련된 최신의 시퀀스-투-시퀀스(sequence-to-sequence) 모델을 발표합니다. 이로 인해 다양한 도메인(domain)에 걸친 음역 정확성을 높였습니다.

- **Technical Details**: 제안된 모델은 타지크어와 페르시아어 간의 음역을 수행하며, 이전 연구에서 효과적인 데이터셋을 확보하지 못한 문제를 해결합니다. 논문에는 저자들이 수집한 두 개의 새로운 데이터셋이 포함되어 있으며, 다양한 데이터 도메인에서의 결과를 통해 음역 작업(task)의 어려움을 명확히 평가합니다. chrF++ 및 Normalized CER 점수는 페르시아어에서 타지크어로 87.91과 0.05, 타지크어에서 페르시아어로 92.28과 0.04를 기록했습니다.

- **Performance Highlights**: 모델은 종합적으로 최신 벤치마크를 설정하며, 모든 도메인에서 일관된 결과를 제공합니다. 이러한 성과는 페르시아어와 타지크어 간의 상호작용을 원활하게 하는 데 기여할 것으로 기대됩니다. 공개된 데이터 및 코드를 통해 연구자들이 쉽게 활용할 수 있도록 했습니다.



### When Thoughts Meet Facts: Reusable Reasoning for Long-Context LMs (https://arxiv.org/abs/2510.07499)
- **What's New**: 이 논문에서는 지식 집약적 멀티 홉 추론 작업을 위해 Thought Template Augmented LCLMs(ToTAL)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다양한 문서에서 증거를 수집하고 이를 조직적으로 연결하는 데 도움을 주는 재사용 가능한 사고 템플릿을 활용합니다. LCLM(Long-Context Language Models)의 발전으로 수백만 개의 토큰을 한 번에 처리할 수 있게 되었지만, 여전히 증거를 연결하는 방식에 대한 접근이 부족한 상황입니다.

- **Technical Details**: ToTAL은 기존의 RAG(Retrieval-Augmented Generation) 방식과는 달리, 사고 템플릿을 외부 매개변수로 간주하고 자연어 피드백을 통해 반복적으로 수정합니다. 이러한 템플릿은 이전 문제 해결의 패턴을 포괄하여 새로운 쿼리에 대한 중간 단계를 구조화할 수 있게 돕습니다. 이 과정은 특정 문제에 국한되지 않고 다양한 쿼리에서 재사용될 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 다양한 벤치마크와 LCLM 가족을 통해 ToTAL은 강력한 기준선 대비 일관된 성능 향상을 보여주었습니다. 특히, 정보 검색이 포함된 상황과 포함되지 않은 상황 모두에서 사고 템플릿이 LCLM 성능을 지속적으로 강화하는 것으로 나타났습니다. 이를 통해 LCLM이 복잡한 증거를 처리할 수 있는 새로운 가능성을 제시하며, 투명한 추론 재사용을 통한 광범위한 적용성을 보여줍니다.



### Can Speech LLMs Think while Listening? (https://arxiv.org/abs/2510.07497)
- **What's New**: 이 논문은 음성 대화 모델(Speech LLMs)의 추론 능력을 향상시키기 위한 체인 오브 쓰뜨(Chain-of-Thought, CoT) 미세 조정의 효과를 조사했습니다. 연구 결과, CoT fine-tuning을 통해 음성 LLM의 추론 정확도가 평균적으로 2.4배 향상되었습니다. 또한 모델이 사용자의 질문이 끝나기 전에 미리 사고를 시작하도록 유도하는 방법이 제안되었습니다.

- **Technical Details**: Moshi 모델은 음성 입력과 시스템의 텍스트를 포함하는 다중 스트림(multi-stream) 아키텍처를 채택하고 있습니다. 이 모델은 사용자 음성, 시스템 음성, 시스템 텍스트의 세 가지 개별 토큰 스트림을 동시에 처리하며, 이를 위해 Mimi라는 코덱 모델을 사용합니다. 미세 조정 방법으로는 사용자 질문의 완전성을 추정하는 새로운 메트릭과 선호 튜닝(preference tuning) 방식이 포함되어 있습니다.

- **Performance Highlights**: 이 연구는 CoT를 텍스트 기반으로 적용할 때 음성 LLM에서의 성능 개선을 보여주었습니다. 논문에서는 최적의 정확도-지연(trade-off)을 달성하기 위해 질문의 완전성 메트릭을 사용하여 70%의 지연을 감소시키면서도 정확도 손실이 없음을 입증했습니다. 이는 향상된 음성 LLM의 응용에 있어 중요한 진전을 의미합니다.



### Can Lessons From Human Teams Be Applied to Multi-Agent Systems? The Role of Structure, Diversity, and Interaction Dynamics (https://arxiv.org/abs/2510.07488)
Comments:
          Under Review at ARR

- **What's New**: 이번 연구는 대규모 언어 모델(LLM)을 이용한 다중 에이전트 시스템(MAS)의 팀 동역학을 탐구하는 새로운 접근 방식을 제안합니다. 팀 과학에서 영감을 받아 팀의 구조, 다양성 및 상호작용 동향을 조사하는 프레임워크를 개발하였으며, 총 4개의 과제를 통해 팀의 성과를 평가합니다. 연구 결과, 전반적으로 수평적인 팀 구조가 계층적인 구조보다 나은 성과를 보이며, 다양성의 영향이 복잡하게 나타난다고 밝혔습니다.

- **Technical Details**: 연구는 LLM 에이전트를 사용하여 수평 및 계층 팀을 시뮬레이션하고, 각 에이전트에게 인구통계적 페르소나를 할당하였습니다. 평가에는 CommonsenseQA, StrategyQA, Social IQa, Latent Implicit Hate Detection의 4개 과제가 포함되며, 이 과제들은 세밀한 추론 및 가치 판단을 요구합니다. 이러한 구조와 다양성이 에이전트의 상호작용과 추론에 미치는 영향을 분석하기 위해 ‘LLM-as-a-judge’ 접근법을 이용하였습니다.

- **Performance Highlights**: 연구 결과는 팀 구조 및 구성 요소가 추론 및 사회적 추론에 미치는 영향을 강조하며, 이러한 차원이 에이전트의 상호작용 및 조율 방식에 중대한 영향을 미친다는 점을 제시하였습니다. 특히, 차별화된 팀 구조는 효과적인 협력을 증진시키고, 신뢰 구축에 중요한 역할을 합니다. 나아가, 연구는 MAS 설계를 위한 이론적 시사점을 제공하며, 인간-AI 협업에서 의사소통 구조와 사회적 프레이밍의 매개 역할을 강조합니다.



### AsyncSpade: Efficient Test-Time Scaling with Asynchronous Sparse Decoding (https://arxiv.org/abs/2510.07486)
Comments:
          14 pages, 17 figures

- **What's New**: 이번 연구는 Test-time scaling (TTS)에서의 효율성을 높이기 위한 비동기적 스파르스 디코딩 프레임워크, AsyncSpade를 제안합니다. 기존 모델의 KV 캐시 관리를 디코딩 루프에서 분리하여 동시성과 긴 Chain-of-Thought (CoT) 상황에서 성능 저하를 개선합니다. 특히, 예측된 다음 쿼리 상태를 기반으로 한 경량 모듈과 비동기적 프레임워크를 통해 모델 성능을 희생하지 않고도 시퀀스 종속성을 제거했습니다.

- **Technical Details**: AsyncSpade는 두 개의 전용 랭크, 즉 전방 컴퓨테이션을 위한 Inference Rank와 KV 관리 및 정밀한 토큰 선택을 담당하는 Cache Rank로 구성됩니다. 이 구조는 KV 캐시 필터링을 인퍼런스 파이프라인에서 분리하여 비동기적으로 쿼리, 키 및 값 임베딩을 전송합니다. 이로 인해 KV 선택이 미리 준비되고 인퍼런스 오버헤드를 줄일 수 있습니다.

- **Performance Highlights**: AsyncSpade는 Quest와 같은 기존의 테스팅 기준 모델과 비교했을 때 TPOT(time-per-output-token)을 20% 이상 줄이는 성과를 보였으며, Qwen3 모델에서는 50% 이상의 감소를 달성했습니다. 또한 다양한 TTS 벤치마크에서 정확도를 유지하거나 초과하는 결과를 제공하여 효율성과 성능 측면에서 뛰어난 성과를 입증했습니다.



### MAPRO: Recasting Multi-Agent Prompt Optimization as Maximum a Posteriori Inferenc (https://arxiv.org/abs/2510.07475)
- **What's New**: 최근의 연구는 다중 에이전트 시스템(Multi-Agent Systems, MAS)이 단일 에이전트보다 더 나은 성과를 낼 수 있음을 보여주었지만, 효과적인 MAS 설계는 여전히 어렵습니다. 이 논문에서는 Multi-Agent PRompt Optimization (MAPRO)이라는 새로운 프레임워크를 제안하여, MAS의 프롬프트 최적화를 Maximum a Posteriori (MAP) 추론 문제로 공식화하고 해결 방식으로 max-product belief propagation 알고리즘을 사용합니다. MAPRO는 에이전트의 특정 프롬프트 정책을 효과적으로 협력하여 조정하는 혁신적인 접근 방식을 제시합니다.

- **Technical Details**: MAPRO의 4단계 프레임워크는 첫 단계에서 MAS 프롬프트 최적화를 MAP 추론 문제로 변형합니다. 다음으로 이 문제는 Directed Acyclic Graphs (DAG)를 기반으로 하여 언어 기반의 max-product belief propagation 알고리즘을 통해 해결됩니다. 복잡한 초기값을 효과적으로 탐색하기 위해, 에이전트별 및 상호작용 기반 보상 모델을 사용하여 전역 최적 프롬프트 할당을 다항 시간 복잡도로 근사합니다.

- **Performance Highlights**: 다양한 작업에서 MAPRO는 수작업으로 설계된 기초선보다 더 우수한 성능을 보여주며, 자동 생성된 방법들과 비교해서도 뛰어난 결과를 도출합니다. 또한, 수학적 추론, 질문 응답, 코드 생성 등 다양한 벤치마크에서 일관되게 최상위 성능을 달성하여 다중 에이전트 프롬프트 최적화의 새로운 기준을 설정하였습니다. MAPRO는 신뢰할 수 있는 다중 에이전트 시스템 구축을 위한 일반 지침도 제공합니다.



### Populism Meets AI: Advancing Populism Research with LLMs (https://arxiv.org/abs/2510.07458)
Comments:
          27 pages, 3 figures. Preprint version under review

- **What's New**: 최근 전 세계적으로 확산되는 포퓰리즘 담론에 대한 학계의 관심이 높아졌다. 본 논문에서는 포퓰리즘을 측정하기 위한 새로운 접근법을 제시한다. 연구진은 Global Populism Database (GPD)를 활용하여 LLM(대형 언어 모델)을 훈련시며, 이를 통해 인간 코더의 분류 정확도에 근접하는 결과를 얻었다. 특히, 본 연구는 LLM이 포퓰리즘의 미묘한 맥락적 측면을 이해하는 능력을 향상시킬 수 있음을 보여준다.

- **Technical Details**: 본 연구에서는 포퓰리즘의 본질과 이를 측정하는 방법에 중점을 두고 있다. 포퓰리즘은 단순히 정책 집합으로 정의되지 않지만, 선의의 '평민'과 부패한 '엘리트' 간의 도덕적 갈등으로 정치가 담론적으로 구성되는 것을 강조한다. 연구진은 LLM을 Holistic Grading (HG)에 적용해 포퓰리즘을 분석하였으며, 이는 12개의 정치적 연설을 대상으로 진행되었다. LLM의 성능을 인간 코더의 기준과 비교한 결과, 특정 모델들이 높은 일관성을 보여주었다.

- **Performance Highlights**: 연구 결과, GPT-5는 인간 코더와 가장 높은 일치도를 기록하였고, Qwen3도 유사한 성능을 보이며 다른 오픈소스 시스템을 상당히 초과하는 결과를 나타냈다. 그러나 모든 시스템이 공통적으로 포퓰리즘 점수를 압축하는 경향을 보여 주의가 필요하다. 본 연구는 AI 기반 방법론이 포퓰리즘 측정의 가능성을 제시하며, 향후 정치 공학 기초 연구 및 응용에 기여할 수 있음을 암시한다.



### Meaningful Pose-Based Sign Language Evaluation (https://arxiv.org/abs/2510.07453)
Comments:
          Accepted at WMT 2025

- **What's New**: 본 연구는 수화 utterances를 의미 있게 평가하기 위한 포괄적인 연구 결과를 제시합니다. 이 연구에서는 keypoint distance-based, embedding-based, 그리고 back-translation-based metrics를 포함한 다양한 평가 지표를 다룹니다. 여러 시나리오에서의 평가 지표 간의 트레이드오프를 보여주며, 수화 언어 번역 또는 생성 시스템의 개발 및 평가에 실용적이고 재현 가능한 방법을 제공합니다.

- **Technical Details**: 자동화된 평가 지표는 자동 생성된 언어 콘텐츠의 품질을 평가하는데 필수적입니다. 연구에서는 침착하게 조정된 keypoint distance-based metrics가 고급 방법들과 경쟁할 수 있음을 발견했습니다. 또한 embedding-based metrics는 자체 도메인에서 뛰어나지만 여러 시스템 간의 문장 수준에서는 어려움을 겪고 있음을 확인했습니다.

- **Performance Highlights**: 연구 결과, back-translation likelihood가 가장 일관된 지표로 나타났으며, 이는 공개된 표준화된 pose-to-text 모델의 필요성을 강조합니다. 제안된 평가 메트릭스의 소스 코드는 GitHub 공개 저장소인 pose-evaluation에 유지 관리되며, 이는 향후 연구를 독려하고 발전시킬 것입니다.



### LASER: An LLM-based ASR Scoring and Evaluation Rubric (https://arxiv.org/abs/2510.07437)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 본 연구에서는 기존의 자동 음성 인식(ASR) 평가 지표인 Word Error Rate(WER)의 한계를 극복하기 위한 혁신적인 방법으로 LASER라는 LLM 기반의 스코어링 루브릭을 소개합니다. LASER는 ASR 시스템의 오류를 보다 정교하게 평가할 수 있는 점수 매기기 방법론으로, 인도 언어의 형태적 및 구문적 특징을 잘 반영합니다. 특히, 이 시스템은 힌디어에서 사람의 주석과 94%의 높은 상관관계를 달성하여 다른 인도 언어에서도 효과적으로 적용이 가능함을 증명하였습니다.

- **Technical Details**: LASER는 표준 ASR 지표에서 발생하는 오류 유형을 분석하여, 수정된 패널티를 부여하는 방법을 제안합니다. minor grammatical errors에는 낮은 패널티를 부여하고, semantic errors에는 높은 패널티를 부여하는 방식입니다. LASER 점수는 맞춤형 프롬프트를 통해 구체적인 지시와 오류 유형을 설정해 LLM에게 처리하게 함으로써 생성됩니다. 이 과정에서 LLM은 주어진 문장을 토큰화하고, 기준 참조와 예측을 정렬하여 오류를 분류합니다.

- **Performance Highlights**: Gemini 2.5 Pro는 모든 LLM 중에서 뛰어난 성능을 보여주며, 사람의 점수와의 상관관계가 가장 높았습니다. 연구 결과, LASER 지수는 전통적인 WER과 비교하여 상대적으로 높은 정확성을 보였고, ASR 예측의 정확성을 판단하는데 있어 훨씬 더 신뢰할 수 있는 방법임을 입증했습니다. 실험을 통해 LASER의 점수가 사람의 판단과 잘 일치함을 확인하면서, 다양한 언어에서의 적용 가능성을 보여주었습니다.



### Lemma Dilemma: On Lemma Generation Without Domain- or Language-Specific Training Data (https://arxiv.org/abs/2510.07434)
Comments:
          14 pages, 2 figures, 5 tables. Accepted to EMNLP Findings 2025

- **What's New**: 본 연구는 최근 LLM(대형 언어 모델)을 사용한 상황 기반의 lemmatization(표제어화) 방법을 다양한 언어에서 실험하여 기존의 전통적 감독 방식과 비교합니다. 지도 학습 데이터가 없는 분야에서 LLM을 통해 lemmatization의 가능성을 조사하고, 기존의 인코더 기반 접근 방식과의 성능 차이를 분석합니다. LLM이 몇 가지 예시만으로 상황에 맞는 표제어를 생성할 수 있는 능력을 갖추었음을 보여줍니다.

- **Technical Details**: 연구에서는 12개의 다양한 형태론적 복잡성을 가진 언어에서 LLM의 성능을 실험하여, 인코더를 학습된 금 데이터에 맞춰 조정하는 것과 비교합니다. 기존 방법은 일반적으로 많은 양의 주석 데이터(annotated data)를 필요로 하며, 이는 복잡한 형태론을 가진 언어에서 상당한 도전이 됩니다. LLM은 사전 학습 없이도 인-context 표제어 생성을 수행하여 최신 성능을 달성할 수 있음을 발견했습니다.

- **Performance Highlights**: 최신 LLM은 단 몇 가지 예제를 제공받고도 상황에 맞게 표제어를 생성하여 대부분의 언어에서 최첨단 성과를 이루어냈습니다. 반면, fine-tuning된 인코더 방식도 여전히 경쟁력 있는 선택이지만, LLM은 전통적인 방식보다 효과적인 해결책으로 부각되고 있습니다. 이 연구는 높은 굴절성과 낮은 자원 언어에서도 LLM의 가능성을 강조하며, 보다 넓은 적용 범위를 확인하게 됩니다.



### Haystack Engineering: Context Engineering for Heterogeneous and Agentic Long-Context Evaluation (https://arxiv.org/abs/2510.07414)
Comments:
          Code available at this https URL

- **What's New**: 이 논문은 HaystackCraft라는 새로운 NIAH 벤치마크를 소개하며, LLM의 긴 문맥에서의 강인함(long-context robustness)을 평가하기 위해 노이즈가 있는 긴 문맥을 구축하는 것이 중요하다고 주장합니다. 연구팀은 다수의 다단계 질문을 통해 영어 위키피디아의 하이퍼링크 네트워크를 기반으로 한 테스팅 환경을 설계했습니다. 이 새로운 평가 기준은 다양한 검색 전략이 LLM의 성능에 미치는 영향을 체계적으로 검토합니다.

- **Technical Details**: 이번 연구에서는 검색 증강 생성(RAG) 기술을 활용하여 LLM의 긴 문맥을 조작하는 방식을 다루며, 상이한 검색 자원들이 자료의 노이즈 및 복잡성을 어떻게 생성하는지를 탐구합니다. 또한, 그래프 기반의 검색 방식의 구현이 LLM의 성능 개선에 어떻게 기여할 수 있는지에 대한 실험을 수행하였습니다. 다양한 Retrieval 전략(예: sparse, dense, hybrid, graph-based)을 기반으로 하여, HaystackCraft는 에이전트의 작업 흐름 중 나타나는 누적 오류(cascading failures)에 대한 모델의 저항성을 평가합니다.

- **Performance Highlights**: 실험 결과, 강력한 밀집 검색기(dense retrievers)는 더 어려운 산만 요소(distractors)를 도입하는 반면, 그래프 기반의 재정렬(graph-based reranking)은 검색의 효과를 높이며 해로운 산만 요소를 줄이는 데 기여했습니다. 15개의 긴 문맥 LLM을 대상으로 한 테스트에서는 Gemini 2.5 Pro와 GPT-5와 같은 고급 모델도 누적 자기 산만 문제로 어려움을 겪는 것으로 나타났습니다. 이 결과는 에이전트의 긴 문맥 추론에 지속적인 도전이 남아 있음을 나타내며, HaystackCraft가 향후 발전을 측정하기 위한 중요한 시험대임을 강조합니다.



### Inconsistent Affective Reaction: Sentiment of Perception and Opinion in Urban Environments (https://arxiv.org/abs/2510.07359)
Comments:
          10 pages

- **What's New**: 이번 연구는 소셜 미디어 플랫폼의 등장으로 도시 환경에 대한 우리의 이해가 어떻게 변화했는지를 다룹니다. 특히, 인간의 인식과 의견에 내재된 감정 반응의 미세한 차이를 식별하고 분석하는 새로운 방법론을 제안합니다. 연구는 Baidu와 Tencent의 스트리트 뷰 이미지 140,750개와 Weibo의 소셜 미디어 텍스트 게시물 984,024개로 구성된 데이터셋을 활용합니다.

- **Technical Details**: 연구팀은 물체 탐지(object detection) 및 자연어 처리(natural language processing) 기법을 통합하여 2016년과 2022년의 베이징 제2환상도로에서 감정을 분류하는 반응 지수를 개발했습니다. 이 연구는 회귀 분석(regression analysis), 이미지 분할(image segmentation), 단어 빈도(word frequency) 기법을 통해 감정 반응을 시각화하고 분석합니다. 토지 이용 분포에 따른 기저 요인을 파악하는 것이 핵심입니다.

- **Performance Highlights**: 연구의 결과, 인식 반응 경향 맵은 긍정적인 감정이 보다 고르게 분포하고 있음을 보여주었으며, 의견 반응 경향 맵에서는 더 극단적인 변화가 관찰되었습니다. 또한, 감정 반응의 변화는 밀집된 건물과 보행자 존재와 같은 요소들과 유의미한 관계를 나타냈습니다. 팬데믹 이전과 이후의 인식 및 의견 감정을 비교한 불일치 맵은 환경 관리 및 도시 재생을 위한 전략 수립에 중요한 통찰을 제공합니다.



### MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning (https://arxiv.org/abs/2510.08567)
- **What's New**: 이 논문에서는 비전 언어 모델(VLMs)이 외부 도구를 활용하여 복잡한 추론과 의사결정 작업에서 어떻게 향상될 수 있는지를 탐구합니다. M-TRACE라는 큰 규모의 멀티모달 데이터셋을 구축하여 VLM 컨트롤러의 튜닝을 지원하며, MATRIX 에이전트를 통해 수동 주석의 필요한 비용을 줄이고 더 나은 일반화 성능을 발휘합니다. 또한, Pref-X라는 자동 생성된 선호 쌍(couples) 세트를 도입하여 단계별 선호 최적화를 통해 결정 과정을 정교화합니다.

- **Technical Details**: MATRIX는 두 단계의 프레임워크로 구성되어 있습니다. 첫 번째 단계에서는 M-TRACE에서 수집된 28.5K 멀티모달 작업을 활용하여 감독된 훈련을(trajectory-driven SFT) 진행합니다. 이어서, Pref-X를 사용하여 11K 개의 자동 생성된 선호 쌍을 기반으로 선호 최적화(preference optimization)를 통해 에이전트의 결정 과정을 정교화합니다. 이 과정 전반에 걸쳐 서로 검증된 경로(traces)를 활용하여 상황에 맞는 도구 사용 능력을 배양합니다.

- **Performance Highlights**: MATRIX는 Agent-X, GTA, GAIA 등 세 가지 벤치마크에서 기존 VLM보다 우수한 성과를 기록했습니다. 특히, 응답 정확도는 각각 14%, 23% 및 11% 향상되었습니다. 이 결과는 MATRIX의 구조적 도구 사용 능력과 단계별 언급 최적화가 효과적으로 작용했음을 보여줍니다. 또한, MATRIX는 이전 에이전트들보다 일관된 추론을 수행하고 더 적합한 도구 선택 능력을 보유하고 있습니다.



### Agent Learning via Early Experienc (https://arxiv.org/abs/2510.08558)
Comments:
          Work in progress

- **What's New**: 이번 연구는 언어 에이전트들이 스스로 경험을 통해 학습하고 향상될 수 있는 가능성을 탐구합니다. 특히, 방식이 부족한 환경에서의 강화 학습의 한계를 극복하기 위해 'early experience'(얼리 익스피리언스)라는 새로운 개념을 도입했습니다. 이는 에이전트의 행동으로 생성된 상호작용 데이터로, 보상 신호 없이도 미래 상태를 감독할 수 있게 합니다.

- **Technical Details**: 연구에서는 두 가지 전략을 통해 초기 경험 데이터를 활용합니다. 첫 번째는 'Implicit world modeling'(암묵적 세계 모델링)으로, 수집된 상태를 사용하여 정책을 환경 동역학에 맞게 조정합니다. 두 번째는 'Self-reflection'(자기 반성)으로, 에이전트가 비효율적인 행동에서 학습하여 추론과 의사결정을 개선하는 방법입니다.

- **Performance Highlights**: 8개의 다양한 환경과 여러 모델 계열을 통해 평가한 결과, 제안된 접근 방식이 효과성과 도메인 외 일반화를 일관되게 개선함을 보여주었습니다. 더욱이, 검증 가능한 보상이 있는 환경에서는 초기 경험이 후속 강화 학습을 위한 강력한 기초가 될 수 있다는 유망한 신호를 포착했습니다.



### VideoNorms: Benchmarking Cultural Awareness of Video Language Models (https://arxiv.org/abs/2510.08543)
Comments:
          24 pages, 5 figures, under review

- **What's New**: 이번 연구에서는 비디오 대형 언어 모델(VideoLLMs)의 문화적 이해를 평가하기 위해 VideoNorms라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 미국과 중국 문화에서 파생된 1000개 이상의 (비디오 클립, 규범) 쌍으로 구성되며, 사회문화적 규범에 기반한 주석을 포함하고 있습니다. VideoNorms는 인간-AI 협업 프레임워크를 통해 구성되어, 모델의 규범 인식 능력을 평가하는 데 기여합니다.

- **Technical Details**: VideoNorms 벤치마크는 비디오 클립에서 특정 문화적 규범이 준수되었는지를 예측하는 두 가지 분류 작업과 이를 뒷받침하는 언어적 및 비언어적 증거를 제시하는 설명 작업을 포함합니다. 이를 통해 모델이 규범 채택 및 위반을 이해하는 데 어려움을 겪고 있음을 분석하였습니다. 이 연구에서는 인간 전문가들이 주석을 검토하고 수정하는 과정이 포함되어, 데이터의 질을 높이는 데 기여합니다.

- **Performance Highlights**: 연구에서 평가된 다양한 오픈 웨이트 VideoLLMs는 규범 준수보다 위반을 인식하는 데 더 낮은 성능을 보였습니다. 또한, 중국 문화와 관련된 비디오 클립에서 성능이 떨어졌으며, 비언어적 증거를 제공하는 데 어려움을 겪었습니다. 이러한 결과는 문화적으로 기반한 비디오 언어 모델 학습의 필요성을 강조하며, VideoNorms 벤치마크가 이 문제를 해결하는 데 기여할 것임을 시사합니다.



### SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models (https://arxiv.org/abs/2510.08531)
Comments:
          Project Page: this https URL Code: this https URL

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)에서 공간 추론(Spatial reasoning)의 한계를 해결하기 위해 SpatialLadder-26k라는 다중 모달 데이터셋과 순차적 훈련 방법론을 제안합니다. 기존 방법들이 인지와 추론의 계층적 구조를 놓치고 있어 성능이 저조하다는 점을 지적하며, 공간 인지를 단계적으로 구축해야 한다고 주장합니다. 이를 통해 공간 지능을 강화하는 세 단계의 훈련 프레임워크를 선보이며, 단계적 접근 방식으로 공간 추론 능력을 향상시킬 수 있음을 강조하고 있습니다.

- **Technical Details**: 제안된 SpatialLadder-26k 데이터셋은 26,610개의 샘플로 구성되어 있으며, 객체 위치 확인(Object localization), 단일 이미지(Single image), 다중 시점(Multi-view), 비디오(Video) 공간 추론 작업을 포함합니다. 이 데이터셋은 각 모달리티 전반에 걸쳐 체계적인 커버리지를 보장하는 표준화된 파이프라인을 통해 구축되었습니다. 훈련 프레임워크는 세 단계로 나뉘며, 첫 번째 단계에서는 객체 위치 확인을 통해 공간 인지를 확립하고, 두 번째 단계에서는 여러 차원 공간 작업을 통해 이해를 발전시키며, 세 번째 단계에서는 강화 학습(Reinforcement learning)을 통해 복잡한 추론을 강화합니다.

- **Performance Highlights**: SpatialLadder 모델은 3B 파라미터를 가지며, 공간 추론 벤치마크에서 최첨단 성능을 기록하였습니다. 연구 결과, SpatialLadder는 기존 모델보다 23.4% 향상된 성능을 보이며, GPT-4o를 20.8% 및 Gemini-2.0-Flash를 10.1% 초과 달성했습니다. 또한, 도메인 외 베치마크에서도 7.2% 향상된 일반화 성능을 유지함으로써, 인지에서 추론으로의 단계적 훈련이 강력한 공간 지능을 위해 필수적임을 입증하였습니다.



### CaRT: Teaching LLM Agents to Know When They Know Enough (https://arxiv.org/abs/2510.08517)
- **What's New**: 이 논문은 LLMs(대규모 언어 모델)가 정보 검색을 언제 중단해야 할지 학습할 수 있도록 돕는 새로운 접근법인 CaRT(Counterfactuals and Reasoning for Termination)를 소개합니다. 정보 검색 단계에서 효과적인 의사결정을 위해 CaRT는 반사실적 데이터를 이용하여 잘못된 결정과 올바른 결정을 비교하여 모델을 학습시킵니다. 이 방법은 상호작용 의료 진단 및 수학 문제 해결 분야에서 구현되었습니다.

- **Technical Details**: CaRT는 반사실적 쌍(counterfactual pairs)을 통해 LLM을 미세 조정(fine-tuning)하여 정보 수집을 중단할 적절한 시점을 학습하도록 합니다. 모델은 음성 추론(verbal reasoning)을 통해 종료 결정을 설명하도록 훈련되며, 이는 최종 LLM 모델에 내재화됩니다. 특히 Qwen3-1.7B-Instruct와 Qwen2.5-3B-Instruct 모델에서 이 방법의 효과성을 입증했습니다.

- **Performance Highlights**: 실험 결과, CaRT는 정보 검색의 효율성과 작업 성공률을 기존의 미세 조정 방법보다 향상시키는 것으로 나타났습니다. 의료 진단 및 수학 문제 해결 모두에서 CaRT 사용 시 성공률이 개선되었습니다. 특히, Qwen3-1.7B-Instruct에서 반사실적 요소를 제거하면 성과가 저하되어 CaRT의 중요성이 강조됩니다.



### SliceFine: The Universal Winning-Slice Hypothesis for Pretrained Networks (https://arxiv.org/abs/2510.08513)
- **What's New**: 이 논문은 사전 훈련된 모델 내에서 무작위로 선택된 서브 네트워크(슬라이스)를 미세 조정하는 것이 하위 작업(adapt) 적응에 충분할 수 있는 이론적 프레임워크를 제시합니다. 우리는 사전 훈련된 네트워크가 전역 승리 슬라이스 속성을 나타낸다는 것을 증명하며, 이는 두 가지 현상에서 비롯됩니다: (1) spectral balance(스펙트럴 밸런스)와 (2) high task energy(높은 작업 에너지). 이러한 발견은 매개변수 효율적인 미세 조정(parameter efficient fine tuning)인 SliceFine 방법을 통해 실증적으로 적용됩니다.

- **Technical Details**: 본 연구는 Universal Winning Slice Hypothesis(전략적 승리 슬라이스 가설)을 통해, 넓은 슬라이스만으로도 미세 조정이 가능하다고 제안합니다. 이 가설은 사전 훈련된 네트워크에서 임의의 충분히 넓은 슬라이스가 로컬 승리 티켓(local winning ticket)이 될 수 있으며, 여러 층에 걸쳐 슬라이스 세트를 조정하면 글로벌 승리 티켓(global winning ticket)을 형성할 수 있음을 명확히 합니다. SliceFine은 이러한 슬라이스를 훈련하며, 기존 매개변수를 추가하지 않고도 성능을 극대화합니다.

- **Performance Highlights**: SliceFine은 언어 및 비전 과제에서 최첨단 PEFT 방법들과 일치하는 성능을 보이며, 훈련 속도와 메모리 효율성, 모델 compactness(소형화)를 크게 개선합니다. 실험적으로, SliceFine은 적은 수의 매개변수로도 유사한 성능을 제공하며, 메모리 사용량 및 교육 시간을 절약합니다. 본 연구는 이론과 실습을 연결하며, PEFT 기술을 다룬 기존 방법에 대한 이론적으로 근거 있는 대안을 제공합니다.



### AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents (https://arxiv.org/abs/2510.08511)
- **What's New**: 이번 논문에서는 AutoMLGen이라는 LLM 기반의 코딩 에이전트를 소개합니다. 이 에이전트는 도메인 지식 데이터를 통합하여 높은 품질의 사전 가이드를 제공하고, 몬테 카를로 그래프 탐색(Monte Carlo Graph Search, MCGS) 알고리즘을 통해 효율적인 탐색을 가능하게 합니다. MCGS는 트리 기반 탐색의 장점을 유지하면서 그래프 구조를 결합하여 동적으로 경로를 재조정하고, 과거의 경로를 재사용하며, 다중 솔루션을 융합할 수 있도록 합니다.

- **Technical Details**: AutoMLGen은 코딩 에이전트로, 머신 러닝 엔지니어링(Machine Learning Engineering, MLE) 작업을 위한 지식 기반과 MCGS를 통합합니다. 이를 통해 다양한 모델 및 데이터 차원에서 도메인 사전 지식을 제공하며, 탐색 과정에서 세밀한 개선을 지원합니다. MCGS는 기존의 MCTS(Monte Carlo Tree Search)의 변형으로, 확장 단계에서 그래프 구조를 포함하여 고유한 해결책의 재조합과 통합을 가능하게 합니다.

- **Performance Highlights**: MLE-Bench에서의 평가 결과, AutoMLGen은 12시간의 예산 내에서 평균 36.4%의 메달 비율을 달성하며 기존의 모든 기준선을 능가하는 성능을 보였습니다. 이 에이전트는 안정적인 탐색 및 실행 가능성을 높이기 위해 세분화된 운영자 집합을 설계하여 최적의 ML 파이프라인 생성을 자동화합니다. AutoMLGen의 도입으로 MLE 작업에서의 성능이 획기적으로 향상되었습니다.



### To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language Models (https://arxiv.org/abs/2510.08510)
Comments:
          Preprint. Project page: this https URL

- **What's New**: 최근 대형 비전 언어 모델(LVLMs)은 시각적 정보와 텍스트 정보를 이해하고 추론할 수 있는 강력한 아키텍처로 자리 잡았습니다. 이 모델은 비전 트랜스포머(Vison Transformer, ViT)와 대형 언어 모델(Large Language Model, LLM)이라는 두 가지 핵심 구성 요소에 의존합니다. 본 연구에서는 ViT에서 시각적 의미를 포착하는 중요한 고규범 토큰인 ViT attention sinks를 탐구하였으며, 이들이 LVLM의 이해와 추론에서 어떻게 기여하는지에 대한 자세한 분석을 제시합니다.

- **Technical Details**: LVLM은 주로 세 가지 주요 구성 요소로 구성됩니다: 비전 인코더(Visual Encoder), 연결 모듈(Connector Module), 언어 모델(Language Model). 비전 인코더는 입력 이미지에서 시각적 특징을 추출하여 비밀 상태(hidden state)를 생성하며, 연결 모듈은 이러한 시각적 특징을 LLM의 텍스트 공간으로 매핑합니다. 마지막으로 언어 모델은 다양한 종류의 토큰을 입력받아 출력을 생성하며, 여기서 다중 헤드 어텐션(Multi-Head Attention, MHA) 메커니즘이 중요한 역할을 합니다.

- **Performance Highlights**: 연구에서는 ViT attention sinks를 효과적으로 활용함으로써 LVLM 성능을 향상시키는 여러 가지 방법을 제안했습니다. 특히, DIYSink라는 훈련 기반 프레임워크를 도입하여, 이 시각적 토큰들이 언어 모델에서 어떻게 활용될 수 있는지를 실험적으로 검증하였습니다. 다양한 LVLM과 비전 모델 조합에 대해 일관되게 성능 향상이 나타났으며, 이는 ViT attention sinks의 잠재력을 재발견하는 데 기여하였습니다.



### The Visual Iconicity Challenge: Evaluating Vision-Language Models on Sign Language Form-Meaning Mapping (https://arxiv.org/abs/2510.08482)
- **What's New**: 본 논문에서는 시각 언어 모델(VLMs)이 동적인 인간의 행동에서 언어적 형태와 의미 간의 관계를 회복하는 도전 과제를 다루기 위해 Visual Iconicity Challenge라는 새로운 비디오 기반 벤치마크를 제안합니다. 이 챌린지는 NGT(네덜란드 수화)의 신호 데이터셋으로, 단어의 형태를 예측하고, 형태에서 의미를 유추하며, 점진적인 아이코닉성 평가를 포함한 세 가지 작업을 수행합니다. 연구에 따르면, VLMs는 손 모양과 위치에 대한 예측에서 일부 세부 사항을 회복하지만, 인간의 성능에는 여전히 미치지 못합니다.

- **Technical Details**: 본 연구는 VLMs의 특징을 평가하기 위해 수동적으로 주석이 달린 NGT 신호 데이터셋을 사용합니다. 데이터셋은 명확한 시각적 링크가 있는 아이코닉한 신호와 그렇지 않은 임의의 신호를 구별합니다. 연구는 각각의 모형이 손 모양, 위치 및 움직임 특징을 인식할 수 있는지, 시각적 형태만을 기반으로 신호의 의미를 유추할 수 있는지, 그리고 모델이 인간의 아이코닉성 판단에 얼마나 근접하는지를 평가합니다. 이러한 작업은 아이코닉성이 시각적 형태와 의미를 연결하는 데 필수적이라는 점에서 중요합니다.

- **Performance Highlights**: VLMs는 아이코닉한 신호의 손 모양과 위치를 예측하는 데 어느 정도 성공했지만, 투명성과 아이코닉성 등에서 인간 성능에 비해 현저히 낮습니다. 특히, 아이코닉 형태 예측을 더 잘 수행하는 모델이 인간의 아이코닉성 판단과 더 잘 상관관계를 가지는 것으로 나타났습니다. 이러한 결과는 모델들이 시각적으로 기반한 구조에 대한 민감성을 공유하고 있음을 시사합니다. 최종적으로, 이 연구는 아이코닉성의 모델링 및 다중 모달 모델에서 시각적 기초를 향상시킬 수 있는 인간 중심의 신호 및 체화 학습 방법을 촉진하는 계기를 제공합니다.



### Looking to Learn: Token-wise Dynamic Gating for Low-Resource Vision-Language Modelling (https://arxiv.org/abs/2510.08470)
Comments:
          Accepted to the EMNLP 2025 BabyLM Workshop

- **What's New**: 이 연구에서는 BabyLM Challenge 2025 비전 트랙의 요구 사항에 맞추어 경량화된 디코더 기반 아키텍처를 제안합니다. 이 아키텍처는 언어적 정보와 시각적 정보를 적응적으로 융합하기 위한 동적 게이팅(token-wise dynamic gating) 메커니즘, 제한된 시각적 정보의 효용을 극대화하기 위한 특징 조정 및 채널 주의(feature modulation and channel attention), 그리고 시각적 기초를 확보하기 위한 보조 대조 목표(auxiliary contrastive objectives)를 포함하고 있습니다. 이를 통해 수치적으로 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: 논문에서는 동적 게이팅 메커니즘을 채택하여 각 토큰에 대한 시각적 신호와 언어적 신호의 가중치를 맥락에 따라 선택적으로 조정합니다. 또한, 특징 향상 기술을 활용하여 제한된 시각적 정보의 유용성을 극대화하고, 문장 및 단어 수준에서 작동하는 대조 학습 보조 목표의 영향을 탐구합니다. 평가 데이터셋에서는 정보 병목 문제와 데이터셋 분할로 인해 훈련 불안정성이 발생할 수 있음을 지적합니다.

- **Performance Highlights**: 평가 결과, 제안된 프레임워크는 총 다섯 개의 벤치마크에서 경쟁력 있는 성능을 보였습니다. 특히, 동적 게이트가 언어의 품사에 따라 시각적 신호와 언어적 신호의 가중치를 조정하며, 내용 단어에는 시각적 신호를, 기능 단어에는 언어적 신호를 더 우선시한다는 것을 발견했습니다. 이러한 결과는 인지 기반 학습의 영감을 얻어 비전-언어 모델의 발전에 기여할 수 있는 가능성을 제시합니다.



### xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning (https://arxiv.org/abs/2510.08439)
Comments:
          24 Pages, 4 Figures, 2 Tables

- **What's New**: xRouter는 현대 LLM(대형 언어 모델)의 성능과 비용 간의 균형을 조화롭게 맞추기 위해 개발된 도구이며, 학습된 라우터가 직접 질문에 답하거나 외부 모델을 호출할 수 있는 시스템입니다. 기존의 정적 에스컬레이션 규칙이나 키워드 휴리스틱의 한계를 극복하고, 비용과 성과를 인식한 보상을 통해 라우팅을 강화 학습 문제로 형성함으로써 손으로 설계된 규칙의 필요성을 없앴습니다.

- **Technical Details**: xRouter는 도구 호출 기반의 라우팅 시스템으로, 경제적 제약을 명시적으로 포함한 보상 구조를 통해 모델 호출을 최적화합니다. 라우터의 학습은 강화 학습 방식을 통해 이뤄지며, 모델의 능력에 따라 적절히 질문을 처리하고 필요할 때는 외부 모델에 위임합니다. 이를 통해 경제적 요소를 함께 고려하면서도 성능 향상과 비용 절감을 동시에 실현할 수 있습니다.

- **Performance Highlights**: xRouter는 다양한 벤치마크에서 비용-성과 트레이드오프를 성공적으로 달성했으며, 비슷한 작업 완료율을 유지하면서 상당한 비용 절감을 이루어냈습니다. 초기 탐색을 통해 다양한 실험을 진행하였고, 이를 통해 얻은 실증적 통찰들은 향후 시스템 개발에 있어 유용한 방향성을 제시합니다. 이러한 기여를 통해 xRouter는 원칙 기반의 경제적 접근 방식을 가진 LLM 조정의 첫 걸음이 되고자 합니다.



### FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts (https://arxiv.org/abs/2510.08396)
Comments:
          NeurIPS 2025 accepted paper

- **What's New**: FlyLoRA는 Fly의 후각 회로에서 영감을 받아 개발된 모형으로, 기존의 LoRA 방법에서 나타나는 파라미터 간의 간섭을 줄여줍니다. 이 방법은 낮은 차원의 전문가 활성화(rank-wise expert activation)를 통해 더 나은 수행 성능을 제공합니다. 이러한 혁신은 단일 작업 내에서의 파라미터 간섭을 효과적으로 해소하며, 기존의 MoE 기반 LoRA 방법들보다 효율성을 높여주는 목표를 달성합니다.

- **Technical Details**: FlyLoRA는 매트릭스 𝑨를 고정된 희소 랜덤 프로젝션(sparse random projection)으로 취급하며, 여기서 각 LoRA 컴포넌트는 서로 다른 랜덤 프로젝션을 통해 약 orthogonal 하게 매핑됩니다. 이를 통해 FlyLoRA는 intra-task(작업 내) 간섭을 최소화하고, multi-task(다중 작업) 연합에서도 비교적 독립적으로 작동하도록 설계되었습니다. 이러한 구조는 AI 기술에 생물학적 구조의 영감을 결합한 결과로, 효율적인 파라미터 사용을 가능하게 합니다.

- **Performance Highlights**: FlyLoRA는 일반 지식 이해, 과학 질문 답변, 수학 추론, 코드 생성 등 다양한 작업에서 기존 방법들 대비 일관된 성능 향상을 보여줍니다. 실험 결과를 통해 FlyLoRA가 계산 효율과 파라미터 간섭 감소 모두를 이뤄내며, 최신 기계 학습 기준에 부합하는 성능을 입증하였습니다. 이러한 결과는 AI 기술 진보의 중요한 원동력이 될 것으로 기대됩니다.



### Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries (https://arxiv.org/abs/2510.08325)
Comments:
          10 pages, 3 figures

- **What's New**: 강화학습을 통한 검증 가능한 보상(Reinforcement Learning with Verifiable Rewards, RLVR)은 대규모 언어 모델의 이론적 문제 해결 능력을 향상시키는 데 중요한 패러다임으로 자리 잡고 있습니다. 최근 연구에서 RLVR 모델의 이론적 경계가 실제로 확장되는지에 대해 의문이 제기되었습니다. 본 논문에서는 Pass@k 지표가 문제의 신뢰도를 고려하지 않기 때문에, 보다 유용한 대안으로 Cover@tau 지표를 제안합니다.

- **Technical Details**: Cover@tau는 모델이 지식을 바탕으로 문제를 해결할 확률율을 τ 이상으로 설정하여 문제 해결 능력을 측정합니다. 이 메트릭은 Pass@k와 달리 무작위 추측에 의한 성능 저하가 발생하는 것을 방지합니다. 제안된 지표는 RLVR 모델을 평가하는 데 있어 서로 다른 신뢰성 수준을 적용해 보는 새로운 접근법을 제공합니다.

- **Performance Highlights**: 여러 RLVR 모델을 평가한 결과, Cover@tau 지표는 Pass@1 또는 Pass@k와 비교했을 때 서로 다른 알고리즘의 상대적인 순위를 제공합니다. 이는 모델의 능력에 대한 새로운 관점을 제시하며, Pass@k 지표가 편향된 성능 수치를 보여줄 수 있다는 점을 강조합니다. 이러한 평가를 통해 RLVR 방법론의 수학적 추론 능력이 보다 정확히 분석되었습니다.



### Mix- and MoE-DPO: A Variational Inference Approach to Direct Preference Optimization (https://arxiv.org/abs/2510.08256)
- **What's New**: 최근 등장한 Direct Preference Optimization (DPO)은 대형 언어 모델(LLMs)을 사용자 선호에 맞추는 간단하고 효과적인 대안으로 자리잡았습니다. 본 연구에서는 DPO의 한계를 극복하기 위해 Mix- 및 MoE-DPO 프레임워크를 제안합니다. 이는 DPO를 부드러운 혼합 모델과 전문가 혼합(Combination of Experts) 아키텍처로 확장하여, 다양한 선호 분포에 적응할 수 있도록 합니다.

- **Technical Details**: 새로운 Mix- 및 MoE-DPO 방법론은 전문가 배치를 위한 잠재 변수 모델을 도입하고, 변분 증거 하한(ELBO)을 최적화합니다. 이 방식은 사용자 특정 혼합 정책을 가능하게 하는 입력 의존적 부드러운 게이팅을 통해 보상 및 정책의 전문화를 촉진합니다. 또한, 이러한 접근법은 모듈화된 배포를 지원하여 기존 모델과의 효율적인 통합 및 사용자 개인화가 가능합니다.

- **Performance Highlights**: 기술적으로 Mix- 및 MoE-DPO는 변수를 고정한 Mix-DPO와, 입력 의존적 가중치를 가지는 MoE-DPO로 나뉘어, 다양한 인과 모델 크기와 다중 선호 데이터셋에서 그 성능을 검증하였습니다. 이로 인해 Mix- 및 MoE-DPO는 무수히 많은 적용 가능성을 가진 선호 기반 LLM 정렬 방법을 제시합니다.



### Opponent Shaping in LLM Agents (https://arxiv.org/abs/2510.08255)
Comments:
          29 pages, 15 figures, 15 tables

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)를 기반으로 한 에이전트들 간의 전략적 상호작용인 opponent shaping에 대한 최초의 조사를 수행했습니다. 기존의 알고리즘은 LLM에 맞지 않는 구조적 결함이 있기 때문에, 저자들은 이를 해결하기 위해 ShapeLLM이라는 새로운 알고리즘을 고안했습니다. 이 모델은 transformer 기반의 에이전트에서 동작하도록 설계된 model-free 방법입니다.

- **Technical Details**: ShapeLLM은 Proximal Policy Optimization (PPO) 방법을 사용하여 LLM 에이전트의 학습 역학을 서로에게 전략적으로 영향을 미치도록 실험했습니다. 연구팀은 반복적 매트릭스 게임을 이용하여 LLM 에이전트들이 서로의 학습 동태에 미치는 영향을 분석하고, 이를 통해 경쟁적 및 협력적 시나리오에서 전략적 상호작용의 실효성을 평가했습니다. 해당 연구는 LLM 에이전트가 독립적으로 행동하는 것이 아닌, 서로에게 영향을 주고받으며 학습한다는 중요한 사실을 증명합니다.

- **Performance Highlights**: LLM 에이전트들은 다양한 게임 이론적 환경에서 효과적으로 상대를 유도하여 경쟁적인 상황에서는 유리한 균형으로, 협력적인 상황에서는 상호 이익을 위한 행동을 촉진할 수 있음을 보였습니다. 연구 결과, LLM들이 지속적인 상호작용을 통해 각자의 행동을 조정하고, 더 나아가 프로소셜(prosocial) 행동을 이끌어내는 데 기여할 수 있는 가능성이 있음을 보여줍니다.



### ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieva (https://arxiv.org/abs/2510.08252)
Comments:
          17 pages, 3 figures

- **What's New**: 본 연구에서는 ReasonEmbed라는 새로운 텍스트 임베딩 모델을 소개합니다. 이 모델은 추론 집약적인 문서 검색을 위해 개발되었으며, 주요 기여로 ReMixer라는 새로운 데이터 합성 방법과 Redapter라는 자기 적응형 학습 알고리즘을 제안합니다. 이 두 가지 기법을 통해 고품질의 훈련 샘플을 대규모로 생성하고, 각 샘플의 추론 강도를 기반으로 가중치를 조정하여 효과적인 데이터 활용을 가능하게 합니다.

- **Technical Details**: 우리는 ReMixer를 통해 조절된 쿼리 생성, 소스 제외 후보 발굴, 추론 강화 관련성 주석 달기 등의 세 가지 단계로 이루어진 데이터 합성 워크플로우를 설계합니다. 이어서 Redapter를 통해 합성 데이터의 추론 강도에 따라 각 훈련 샘플의 가중치를 동적으로 조정하는 방법을 소개하며, 이는 다양한 규모의 LLM 백본을 기반으로 ReasonEmbed를 구현하여 최신 성능을 달성하는 데 기여합니다.

- **Performance Highlights**: ReasonEmbed-Qwen3-8B 모델은 BRIGHT 벤치마크에서 nDCG@10 점수 38.1을 기록하며 기존 텍스트 임베딩 모델을 크게 초월하는 성과를 보였습니다. 이 모델은 합성 데이터와 자기 적응형 학습 알고리즘의 기여를 통해 성능 향상을 이룬 것을 실증적으로 분석하였습니다. 모든 자원은 오픈 소스 형태로 공개되어 향후 연구에 도움을 줄 예정입니다.



### Sentiment Matters: An Analysis of 200 Human-SAV Interactions (https://arxiv.org/abs/2510.08202)
Comments:
          Accepted for presentation at IEEE ITSC 2025 and for publication in its Proceedings. \c{opyright} 2025 IEEE. Personal use permitted; other uses require permission from IEEE, including reprinting, republishing, or reuse of any copyrighted component of this work

- **What's New**: 이 논문은 200개의 인간-공유 자율 차량(Shared Autonomous Vehicles, SAV) 상호작용 데이터셋을 소개하며, 이는 효율적인 인간-SAV 상호작용 연구에 기여합니다. 공개된 이 데이터셋은 2,136개의 대화 교환과 다양한 심리적 요소에 대한 사용자 조사를 포함하고 있습니다. 이 연구는 SAV가 사용자 수용성과 서비스 품질을 결정하는 주요 요인을明해주는 두 가지 벤치마크 사례 연구를 제공합니다.

- **Technical Details**: 논문에서는 50명의 참가자와 4개의 SAEL 5 SAV 에이전트 사이의 대화 상호작용을 바탕으로한 데이터셋 설계 및 수집 과정을 설명합니다. 우리는 랜덤 포레스트 모델링과 감정 분석 도구를 통해 결과를 분석했습니다. 또한, OpenAI의 GPT-3.5 turbo를 이용한 대화형 데이터 수집 방식을 취하여 심리적 측면을 측정했습니다.

- **Performance Highlights**: 향상된 사용자 수용성을 위하여 혼합된 전략이 담긴 SAV4가 가장 긍정적인 사용자 반응을 이끌어냈습니다. LLM 기반의 감정 분석 도구가 전통적인 텍스트 기반 방법인 TextBlob에 비해 더 정확한 사용자 감정 보고와 일치하는 결과를 보였습니다. 이 연구는 대화형 SAV 인터페이스 설계 및 진화된 감정 모델링의 기초를 마련합니다.



### R-Horizon: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth? (https://arxiv.org/abs/2510.08189)
- **What's New**: 본 연구는 최근 LRM (Large Reasoning Models)에서 long-horizon reasoning (장기 추론 능력)을 자극하기 위해 R-HORIZON이라는 새로운 방법론을 제안합니다. 기존의 벤치마크가 단일 수평 작업에 국한되어 있는 반면, R-HORIZON은 복잡한 다단계 문제를 해결하는 다양한 벤치마크를 구성합니다. 이 방법은 모델이 긴 추론 시나리오를 이해하고 반응하는 능력을 평가하는 데 있어 중요한 기여를 합니다.

- **Technical Details**: R-HORIZON은 query composition을 통해 LRM의 장기 추론 행동을 촉진합니다. 이 방법은 기존의 단일 수평 문제를 연결하여 서로 의존하는 다수의 문제를 포함하는 복잡한 다단계 문제를 생성합니다. 연구에서는 수학, 코드 생성 및 에이전트 응용 프로그램에서 총 6개의 대표적인 데이터셋을 사용하여 평가를 수행했습니다.

- **Performance Highlights**: R-HORIZON을 사용한 평가에서, 최신 LRM조차도 장기 추론 작업에서 성능 저하를 겪는다는 것이 발견되었습니다. RLVR (Reinforcement Learning with Verified Rewards) 데이터로 훈련시킬 경우, 다단계 문제에 대해 성능이 크게 향상되며, 표준 추론 작업에서도 7.5 포인트의 정확도가 증가했습니다. R-HORIZON은 LRM의 장기 추론 능력을 향상시키고 평가하는 데 있어 확장 가능하고 저렴한 패러다임으로 자리잡게 됩니다.



### NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions (https://arxiv.org/abs/2510.08173)
- **What's New**: 이 연구에서는 내비게이션 에이전트의 공간적 인지를 평가하기 위한 새로운 벤치마크인 NavSpace를 소개합니다. NavSpace는 1,228개의 경로-명령 쌍으로 구성되어 있으며, 내비게이션 에이전트의 공간적 지능을 시험하는 여섯 가지 작업 카테고리를 포함합니다. 이 평가에서는 최첨단 내비게이션 모델인 SNav를 제안하며, 이는 기존 모델들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: NavSpace 벤치마크는 내비게이션 작업의 클래식한 정의를 따르며, 에이전트는 주어진 언어 명령에 따라 다음 내비게이션 동작을 예측해야 합니다. 연구팀은 설문조사를 통해 공간 인지에 필수적인 카테고리를 식별하였고, 수집된 데이터는 기계 학습 모델과 함께 사용됩니다. 이 연구는 22개의 기존 내비게이션 에이전트를 종합적으로 평가하였습니다.

- **Performance Highlights**: SNav는 NavSpace와 실제 로봇 테스트에서 기존 내비게이션 에이전트들보다 우수한 성능을 보였으며, 이는 향후 연구에 강력한 기준을 세우는 데 기여합니다. 실험 결과, 내비게이션 분야에서 공간적 지능의 중요성과 MLLM의 한계를 강조하며 내비게이션 대형 모델의 장점을 밝혀냈습니다. 이러한 통찰은 향후 연구에서 내비게이션 에이전트의 공간적 인지를 향상시키는 방향성을 제시합니다.



### Can Risk-taking AI-Assistants suitably represent entities (https://arxiv.org/abs/2510.08114)
- **What's New**: 이번 연구에서는 AI의 책임감 있는 사용을 위해 언어 모델(Launguage Models, LMs)의 위험 회피성( risk aversion)의 조작 가능성(manipulability of risk aversion, MoRA)을 탐구하였습니다. 이는 AI 기반 의사결정 지원 시스템에서 LMs의 위험 행동을 이해하는 것이 중요하다는 점을 강조합니다. 특히, 성별에 따른 태도와 불확실성, 역할 기반 의사 결정에 집중하여 다양한 경제 시나리오에서 인간의 위험 선호를 재현할 수 있는지 살펴보았습니다.

- **Technical Details**: 연구 결과, DeepSeek Reasoner와 Gemini-2.0-flash-lite와 같은 모델들이 인간 행동과 어느 정도 일치를 보이는 반면, 특정 불일치가 존재하여 인간 중심의 조작 가능성 측정법에 대한 세밀한 조정이 필요하다는 점이 드러났습니다. LMs의 위험 회피성은 조작 가능성을 가지고 있지만 이는 완벽하게 인간의 의사 결정 방식을 반영하지는 못하고 있다는 점에서 구체적인 실증 분석이 필요합니다.

- **Performance Highlights**: 이 연구는 AI 시스템이 인간의 위험 선호를 보다 정확하게 재현할 수 있도록 모델 설계를 개선할 방향을 제시합니다. 이는 위험 관리(context)에서 AI의 효과성을 높이는데 기여할 수 있는 접근법으로, AI 지원 시스템의 적용 가능성을 향상시킬 것으로 기대됩니다. 향후 연구에서는 AI의 윤리적 의사 결정을 증진하는 방법도 모색해야 할 것입니다.



### VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents (https://arxiv.org/abs/2510.08109)
- **What's New**: 이번 논문에서는 기술 문서의 버전 관리 문제를 해결하기 위한 새로운 Retrieval-Augmented Generation (RAG) 시스템인 VersionRAG를 제안합니다. 기존 RAG 시스템들은 문서의 진화에 따른 정확한 답변을 제공하지 못하고 있으며, 특히 버전이 중요한 경우에서의 결과 정확도가 낮습니다. VersionRAG는 문서의 변화 과정을 계층 그래프 구조로 모델링하여 버전별 필터링 및 변화 추적을 가능하게 합니다.

- **Technical Details**: VersionRAG의 핵심 요소는 세 가지 쿼리 유형을 정의하고, 이 각각에 대해 특화된 탐색과 필터링 메커니즘을 사용하는 것입니다. 문서 버전은 계층 그래프 노드와 엣지로 표현되어 있으며, 각 노드는 문서의 카테고리, 개별 문서, 버전, 내용 및 변경 사항을 나타냅니다. 이 구조는 문서 쿼리에 대한 효율적인 그래프 탐색을 가능하게 하여 정확도 및 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: VersionRAG는 34개의 버전이 있는 기술 문서에서 100개의 수작업으로 구성된 질문에 대해 90%의 정확도를 달성했습니다. 이는 기존의 RAG(58%) 및 GraphRAG(64%)와 비교하여 현저히 높은 성능을 보입니다. 또한, 인덱싱 과정에서 GraphRAG에 비해 97% 적은 토큰을 필요로 하여 대규모 배포 시 효율성을 확보하고 있습니다.



### AutoQual: An LLM Agent for Automated Discovery of Interpretable Features for Review Quality Assessmen (https://arxiv.org/abs/2510.08081)
Comments:
          EMNLP 2025

- **What's New**: 본 논문은 온라인 리뷰의 내재 품질을 평가하기 위한 새로운 프레임워크인 AutoQual을 제안합니다. AutoQual은 LLM(대형 언어 모델) 기반의 에이전트로, 해석 가능한 기능(feature)을 자동으로 발견하는 프로세스를 자동화합니다. 이는 데이터에 내재된 암묵적 지식을 명시적이고 계산 가능한 기능으로 변환하는 것을 목표로 하며, 인간 연구 프로세스를 모방하여 반복적으로 기능 가설을 생성합니다.

- **Technical Details**: AutoQual은 다각적인 아이디어 및 대조적 데이터 분석을 통해 기능 후보군을 형성합니다. 각 기능은 자연어로 표현 가능한 해석 가능한 형태로 설계되며, 최대한의 정보량을 제공하기 위해 상호 정보량(mutual information)을 극대화합니다. 이 시스템은 LLM을 활용하여 기초 후보군인 𝒮cand(기능 후보세트)을 생성한 후, 각 후보에 대해 신뢰할 수 있는 측정 도구를 자동으로 생성 및 검증하는 방식을 사용합니다.

- **Performance Highlights**: 대규모 온라인 플랫폼에서 AutoQual의 성능을 A/B 테스트로 검증한 결과, 사용자당 평균 리뷰 조회 수가 0.79% 증가하고 리뷰 독자의 전환율이 0.27% 향상되었습니다. 이러한 실질적인 개선은 AutoQual의 유효성을 입증하며, 리뷰 품질 평가를 위한 해석 가능한 기능 발견의 필요성을 해결하는 데 기여합니다.



### TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevanc (https://arxiv.org/abs/2510.08048)
- **What's New**: 본 연구에서는 TaoSR-AGRL이라는 Adaptive Guided Reinforcement Learning 프레임워크를 제안합니다. 이 프레임워크는 Taobao 검색 관련성 예측을 강화하기 위해 설계되었고, 두 가지 주요 혁신을 포함합니다: Rule-aware Reward Shaping 및 Adaptive Guided Replay. 이러한 접근 방식은 복잡한 비즈니스 규칙과 사용자 쿼리의 변화하는 요구를 충족시키기 위해 긴-tail 문제를 해결하고자 합니다.

- **Technical Details**: TaoSR-AGRL의 핵심 모듈 중 하나인 Rule-aware Reward Shaping은 최종 관련성 판단을 밀집화된 구조화된 보상으로 분해하여 도메인별 기준에 맞게 조정합니다. Adaptive Guided Replay는 학습 중 낮은 정확도의 롤아웃을 식별하여 궁극적으로 정책이 정체된 추론 패턴에서 벗어날 수 있도록ground-truth 정보를 주입합니다. 이로 인해 모델은 높은 가치의 추론 경로를 탐색하고 개선할 수 있습니다.

- **Performance Highlights**: TaoSR-AGRL은 대규모 실제 데이터세트를 기반으로 평가되었으며, DPO 및 기존 GRPO 기반라인을 일관되게 초과하여 관련성 정확도, 규칙 준수 및 훈련 안정성을 개선했습니다. 이 모델은 Taobao의 주요 검색 시나리오에 성공적으로 배포되어 수억 명의 사용자에게 서비스를 제공하고 있으며, 연구 결과는 해당 산업 분야에서의 실질적인 가치를 보여줍니다.



### Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition (https://arxiv.org/abs/2510.08047)
- **What's New**: 이번 연구는 ASR(Automatic Speech Recognition) 시스템의 도메인 전이에 대한 견고함을 증대시키기 위한 혁신적인 접근 방식인 Pseudo2Real을 제안합니다. 이 방법은 실제 레이블이 없는 경우에도 발생하는 체계적인 편향을 수정하는 데 초점을 맞추고 있습니다. Pseudo2Real은 파라미터 공간에서의 교정 벡터를 생성하여 ASR 모델의 성능을 개선하고, 아프리카의 여러 억양에서 최대 35%의 상대적인 단어 오류율(WER) 감소를 달성합니다.

- **Technical Details**: Pseudo2Real의 기초는 소스 도메인에서 실제 레이블과 가짜 레이블이 모두 존재할 때 두 개의 ASR 모델을 미세 조정하고 그 차이점을 활용하여 교정 벡터를 만드는 것입니다. 이 벡터는 pseudo-labeling을 통해 발생하는 체계적인 불일치를 포착하여, 타겟 도메인에서 미세 조정된 ASR 모델에 적용하여 실제 레이블 성능에 더 가깝게 조정합니다. 교정 벡터 적용 후, Pseudo2Real-SC와 같은 방법을 통해 발화자 클러스터링을 활용하여 더 강력한 교정 벡터를 생성하는 방식으로 성능을 한층 향상시킬 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, Pseudo2Real은 Whisper 모델의 다양한 크기에서 일관된 성능 향상을 보여주었습니다. 특히 아프리Speech-200 데이터셋에서 10개의 아프리카 억양에 대해 최대 35%의 상대적인 WER 감소를实现하는 성과를 기록했습니다. 연구진은 교정 벡터의 스케일링 요소 및 클러스터 수의 효과를 분석하여 structured pseudo-label 편향을 직접 수정할 수 있는 방법을 제시하였습니다.



### VoiceAgentBench: Are Voice Assistants ready for agentic tasks? (https://arxiv.org/abs/2510.07978)
- **What's New**: 이번 논문에서 제안된 VoiceAgentBench는 대규모 음성 모델(SpeechLMs)에 대한 포괄적인 벤치마크로, 실제 음성 기반 상호작용 시나리오에서의 에이전트 기능을 평가하기 위해 디자인되었습니다. 이 벤치마크는 인도와 관련된 5,500개의 합성 음성 쿼리를 포함하여 다국어 및 문화적 이해를 측정할 수 있습니다. 특히, 음성 기반 에이전트가 복잡한 도구 사용, 다중 턴 상호작용, 그리고 맥락적 의사결정 능력을 포함하는 기본적인 에이전트 기능을 평가할 수 있는 첫 번째 벤치마크입니다.

- **Technical Details**: VoiceAgentBench는 다섯 개의 인도 언어를 포함하여 영어와 힌디를 지원하며, 수천 개의 음성 쿼리를 통해 다양한 도구 호출 유형을 평가합니다. 이는 단일 도구 호출부터 다중 종속 도구 오케스트레이션에 이르는 복잡한 요청을 포함합니다. 또한, 새로운 샘플링 알고리즘을 사용해 음성 전환 시 다양한 악센트와 음성 특성을 고려하여 실제 음성 대화의 이질성을 포착할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 현재의 SpeechLMs는 맥락적인 도구 조정 작업과 인도적 일반화, 그리고 적대적 강건성에서 상당한 성능 차이를 보였습니다. 이는 기존의 음성 기반 모델들이 현실적인 에이전트 능력을 충분히 평가하지 못한다는 점을 강조하며, VoiceAgentBench의 필요성을 부각시킵니다. 또한, 우리는 SpeechLMs와 ASR-LLM 파이프라인 모두에서 주목할 만한 성능 차이를 발견했습니다.



### TTOM: Test-Time Optimization and Memorization for Compositional Video Generation (https://arxiv.org/abs/2510.07940)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 비디오 생성 모델의 성능을 향상시키기 위해 Test-Time Optimization and Memorization (TTOM) 프레임워크를 소개합니다. 기존의 방식과는 달리, TTOM은 훈련 없이 spatiotemporal 레이아웃에 맞춘 출력을 제공합니다. 이는 기계 학습 모델이 영상에서 입력된 텍스트를 더 정련하게 이해하고 생성할 수 있도록 돕습니다.

- **Technical Details**: TTOM은 사용자 프롬프트에 기반한 spatiotemporal layout을 생성하고, 이를 통해 비디오 생성 모델의 성능을 최적화합니다. 새로운 매개변수를 도입하여 각 샘플에 맞춰 업데이트하며, 이 과정을 통해 이전 작업의 최적화를 메모리에 저장할 수 있습니다. 이 파라미터는 삽입, 읽기, 업데이트 및 삭제와 같은 다양한 작업을 지원하여 유연하고 효율적인 운영이 가능합니다.

- **Performance Highlights**: T2V-CompBench 및 Vbench 벤치마크에서의 실험 결과는 TTOM이 매우 효과적이고 실용적이며 효율적인 프레임워크임을 입증하였습니다. 특히, TTOM은 CogVideoX-5B와 Wan2.1-14B와 비교했을 때 T2V-CompBench에서 각각 34% 및 14%의 성능 향상을 이뤘습니다. 이는 복합 비디오 생성에서의 크로스 모달 정렬을 현장에서 자동으로 달성할 수 있도록 해줍니다.



### Self-Improving LLM Agents at Test-Tim (https://arxiv.org/abs/2510.07841)
- **What's New**: 이번 논문은 새로운 테스트 시간 자기 향상 방법(Test-Time Self-Improvement, TT-SI)을 제안합니다. 이 방법은 모델이 어려움을 겪는 샘플을 식별하고, 이를 기반으로 새로운 훈련 샘플을 생성한 후, 테스트 시간에 이 샘플들로 모델을 개선하는 과정으로 구성됩니다. 기존의 대량 데이터셋에 의존하지 않고도 성능을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 세 가지 단계로 이루어집니다: 첫째, 자기 인식을 통해 모델이 어려움을 겪는 샘플을 식별하고, 둘째, 불확실 샘플에서 유사한 예제를 생성하며(셀프 데이터 증강), 셋째, 테스트 시간 맞춤형 훈련을 통해 이 새롭게 생성된 샘플들로부터 학습하는 방식입니다. 이 과정에서 'Uncertainty Estimator'와 'Data Synthesis Function'이 사용되어 모델의 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, TT-SI는 평균적으로 다른 표준 학습 방법에 비해 +5.48%의 절대 정확도 향상을 기록하며, 68배 적은 훈련 샘플로도 효과적인 학습이 가능함을 보여줍니다. TT-SI는 특정 과제를 위한 적응 능력을 강화하여, 기존의 감독된 세미 슈퍼바이즈드 학습(supervised fine-tuning) 방법보다 우수한 성능을 보입니다. 이 연구는 의료 기기 및 기타 다양한 복잡한 작업에서도 지속적인 자기 개선의 가능성을 보여줍니다.



### MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation (https://arxiv.org/abs/2510.07835)
Comments:
          Accepted By NeurIPS 2025

- **What's New**: 본 논문에서는 MetaDefense라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)에서 파인튜닝 기반의 jailbreak 공격을 방어하는 데 중점을 두고 있습니다. 기존 방어 메커니즘이 보지 못한 공격 템플릿에 의해 위장된 해로운 쿼리에 일반화되지 못하는 문제를 발견하였습니다.

- **Technical Details**: MetaDefense는 두 단계의 방어 접근 방식을 제안합니다: (i) 사전 생성 방어(pre-generation defense)로 해로운 쿼리를 반응 생성이 시작되기 전에 탐지하고, (ii) 중간 생성 방어(mid-generation defense)로 생성 중 부분적인 응답을 감시하여 보다 해로운 내용을 출력하는 것을 방지합니다. 이 시스템은 특별한 프롬프트를 사용하여 쿼리와 부분 응답의 해로움을 예측하도록 LLM을 훈련시킵니다.

- **Performance Highlights**: 다양한 LLM 아키텍처(예: LLaMA-2-7B, Qwen-2.5-3B-Instruct, LLaMA-3.2-3B-Instruct)에 대한 광범위한 실험 결과, MetaDefense는 기존 방어 메커니즘에 비해 상당히 우수한 성능을 보여주었습니다. 또한, 본 시스템은 보이는 공격 템플릿과 보이지 않는 공격 템플릿 모두에 대해 효과적인 방어를 제공하며, 일반 작업에 대해 경쟁력 있는 성능을 유지합니다.



### From Keywords to Clusters: AI-Driven Analysis of YouTube Comments to Reveal Election Issue Salience in 2024 (https://arxiv.org/abs/2510.07821)
- **What's New**: 이 논문은 2024년 대통령 선거에서 유권자 선택에 가장 큰 영향을 미친 이슈를 파악하기 위한 두 가지 데이터 과학 방법론을 탐구합니다. 인공지능(AI) 기술에 의해 기반한 새로운 실증적 증거를 사용하여, 사용자 댓글을 분석하는 두 가지 방법을 제시합니다. 이는 특정 미디어 기사에 대한 여론을 더 잘 이해하는 데 도움을 줍니다.

- **Technical Details**: 연구는 자연어 처리(Natural Language Processing)와 군집 분석(Clustering Analysis)이라는 두 가지 방법을 사용하여, 총 8,000개 이상의 유튜브 댓글을 분석합니다. 이 댓글들은 선거 주간 동안 오른편의 월 스트리트 저널과 왼편의 뉴욕 타임즈에서 나온 것입니다. 분석을 통해 유권자에게 중요한 이슈들을 정량화하여, 선거 결과 예측에 더 나은 통찰을 제공합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 이민과 민주주의가 사용자 댓글에서 가장 빈번하게 언급된 이슈로, 정체성 정치가 뒤를 이었으며, 인플레이션은 의미 있게 덜 언급되었음을 보여줍니다. 이는 사전 조사와의 연관성을 확립하긴 하지만, 인플레이션의 중요성을 부정하며 댓글 분석이 여론 조사보다 선거 결과 분석에 더 효율적일 수 있음을 입증합니다.



### oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning (https://arxiv.org/abs/2510.07731)
Comments:
          Main Text: 8 pages, In total: 37 pages, 9 figures

- **What's New**: 본 논문에서는 유기 화학의 메커니즘 추론을 위한 최초의 대규모 전문 큐레이션 벤치마크인 oMeBench를 소개합니다. 이 벤치마크는 10,000개 이상의 주석이 달린 메커니즘 단계를 포함하고 있으며, 중간체(intermediates), 유형 레이블(type labels), 난이도 평가(difficulty ratings) 등이 포함되어 있습니다. 또한, LLM의 성능을 정밀하게 평가할 수 있는 동적 평가 프레임워크인 oMeS를 제안하여 세부적인 평가를 가능하게 하였습니다.

- **Technical Details**: oMeBench와 oMeS는 LLM의 메커니즘 이해 능력을 정량화하는 강력한 도구로 작용하며, 여러 단계에서의 논리적 일관성을 유지하는 능력을 평가할 수 있습니다. 이번 연구는 LLM의 성능을 여러 모델에서 비교하고, 그 한계를 밝혀내며, 특정 데이터셋에서의 전문가 모델의 파인 튜닝(fine-tuning)을 통해 성능을 50% 향상시켰습니다. 이로써 LLM의 화학적 직관(chemical intuition)과 메커니즘 추론 능력의 현재 한계를 보여주고 있습니다.

- **Performance Highlights**: 현재의 LLM 모델들은 비록 화학적 직관을 보이고 있지만, 복잡한 여러 단계의 인과 논리를 유지하는 데 어려움을 겪고 있다는 것이 밝혀졌습니다. 또한, 전문가 주도 하에 작성된 oMeBench 데이터셋을 기반으로 LLM 성능이 크게 개선될 수 있음을 보여주며, 이는 화학 메커니즘 추론 분야의 발전을 기대할 수 있음을 의미합니다. 마지막으로, 우리의 기여는 대규모의 표준화된 메커니즘 데이터셋 구축과 더불어 새로운 평가 지표의 도입을 포함합니다.



### Who Stole Your Data? A Method for Detecting Unauthorized RAG Thef (https://arxiv.org/abs/2510.07728)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 시스템의 도용(detection) 문제를 해결하기 위한 두 가지 주요 기여를 제공합니다. 첫째, 다양한 전문 분야와 글쓰기 스타일을 아우르는 RPD라는 새로운 데이터셋을 소개하여 기존 자원의 한계를 극복하였습니다. 둘째, 의미적 및 어휘적 수준에서 보호 기능을 내장한 이중 레이어 워터마킹 시스템을 개발하고, 통계적 가설 검정을 사용하는 인터로게이터-탐정 프레임워크를 구축하였습니다.

- **Technical Details**: 제안된 방법은 RAG 시스템을 통해 생성된 콘텐츠의 독창성을 보호하는 데 필수적입니다. 이 연구는 기초적인 지식 워터마크를 의미적 차원에서 삽입하고, 통계적 서명 패턴을 활용하여 효과적으로 감지 메커니즘을 제공합니다. 새로운 인터로게이터-탐정 프레임워크는 불법적인 데이터 사용을 밝혀내기 위해 전략적으로 쿼리를 생성하고, 축적된 증거에 대한 철저한 통계 분석을 수행합니다.

- **Performance Highlights**: 결과적으로, 본 논문에서 제안한 접근 방식은 다양한 쿼리 볼륨, 방어 프롬프트, 조회 매개변수에 걸쳐 효과성을 입증하였으며, 적대적 회피 기술에 대해서도 강인성을 유지하고 있습니다. 이러한 성과는 지식 생성의 유인을 보장하고 정보 생태계를 보호하는 데 중요한 기초를 마련합니다.



### Multimodal Safety Evaluation in Generative Agent Social Simulations (https://arxiv.org/abs/2510.07709)
- **What's New**: 본 논문은 MLLM(Multimodal Large Language Models) 기반 에이전트의 안전성과 신뢰성을 평가하기 위한 재현 가능한 시뮬레이션 프레임워크를 소개합니다. 이 프레임워크는 안전성 개선, 위험한 활동 탐지, 그리고 사회 동역학 측정의 세 가지 차원에서 에이전트를 평가합니다. 이를 통해 에이전트가 상호작용을 통해 어떻게 행동 계획을 수정하고 안전한 행동으로 전환하는지를 분석하고 있습니다.

- **Technical Details**: 에이전트는 동적 계획(Dynamic Planning), 다층 메모리(Layered Memory), 다중 모드 인식(Multimodal Perception) 기능을 갖추고 있으며, SocialMetrics라는 행동과 구조 메트릭을 사용하여 계획 수정, 위험한 행동의 안전한 행동 전환 등을 정량화합니다. 시뮬레이션 환경은 실내외 공간으로 구성되며, 각 에이전트는 자아, 사회적 특성, 초기 사회적 유대에 대한 정보를 기반으로 설정됩니다. 에이전트는 사건을 반영하는 세션을 통해 위험 요소를 탐지하고 수정해야 합니다.

- **Performance Highlights**: 실험 결과, 에이전트들은 직접적인 멀티모달 모순을 탐지할 수 있지만, 글로벌 안전성과의 정렬에 실패하여 위험한 계획을 수정하는 데 55%의 성공률에 그쳤습니다. GPT-4o mini, Claude, Qwen-VL 모델을 사용한 시뮬레이션에서 75%, 55%, 58%의 평균 안전한 행동 전환 비율을 기록했습니다. 특히, 잘못된 시각적 정보와 연결된 위험한 행동의 45%가 수용됐으며, 이는 이미지에 대한 과신 경향을 나타냅니다.



### LiveThinking: Enabling Real-Time Efficient Reasoning for AI-Powered Livestreaming via Reinforcement Learning (https://arxiv.org/abs/2510.07685)
Comments:
          12 pages, 8 figures

- **What's New**: 이번 연구에서는 AI 기반 전자상거래 라이브 스트리밍에서 실시간 반응을 지원하는 디지털 아바타의 필요성을 다룹니다. 기존의 Large Reasoning Models (LRMs)는 높은 지연시간 문제로 인해 이러한 작업에 적합하지 않다는 점을 강조하며, 이를 해결하기 위한 새로운 두 단계 최적화 프레임워크인 LiveThinking을 제안합니다. 이 프레임워크의 목표는 반응의 정확성을 유지하면서도 지연 시간을 최소화하는 것입니다.

- **Technical Details**: LiveThinking은 첫 단계에서 670B 파라미터의 teacher 모델을 경량화된 30B Mixture-of-Experts (MoE) 모델로 증류하는 과정을 포함합니다. 여기에는 잘못된 생성을 필터링하는 LLM 기반 평가자를 사용하는 Rejection Sampling Fine-Tuning (RFT)이 포함되어 있습니다. 두 번째 단계에서는 Group Relative Policy Optimization (GRPO)를 통해 모델의 추론 경로를 압축하여 응답 품질을 유지하면서 이전의 긴 추론 경향을 초래하는 비효율성을 해결합니다.

- **Performance Highlights**: LiveThinking은 실제 Taobao Live에 적용되어 3.3%의 정확성 향상과 21.8%의 유용성 증가를 이끌어냈습니다. 이 시스템은 응답의 처리 비용을 30배 줄이면서 처리 지연을 초단위로 줄이는 성과를 거두었습니다. 최종적으로, LiveThinking은 GMV(총 상품 판매량)에서 통계적으로 유의미한 증가를 가져오며, 사용자 경험과 상업적 성과를 개선하는 데 기여했습니다.



### Test-Time Matching: Unlocking Compositional Reasoning in Multimodal Models (https://arxiv.org/abs/2510.07632)
- **What's New**: 이번 연구는 AI 모델의 compositional reasoning(구성적 추론) 문제를 재조명하며, 표준 평가 메트릭이 모델의 능력을 과소 평가한다는 점을 보여줍니다. 이를 해결하기 위해 그룹 구조를 더 잘 활용하는 그룹 매칭 점수(Group Matching Score)를 도입하여, 기존의 지표에서는 발견할 수 없는 모델의 숨은 능력을 드러냅니다. 연구 결과, SigLIP-B16과 GPT-4.1은 이전 모든 결과를 초월하는 성과를 이뤘습니다.

- **Technical Details**: 신선한 접근법으로 Test-Time Matching (TTM)이라는 반복적이고 자기 개선 가능한 알고리즘을 제안하였습니다. 이 알고리즘은 매칭 기반의 의사 레이블을 선택하여 자기 학습을 진행하고, 점진적으로 선택 기준을 완화하여 테스트 데이터셋에 대한 범위를 확장합니다. 이로 인해 SigLIP-B16과 GPT-4.1은 여러 벤치마크에서 놀라운 성능 향상을 보였습니다.

- **Performance Highlights**: TTM을 통해 SigLIP-B16은 Winoground에서 72.5, MMVP-VLM에서 89.44, ColorSwap에서 94.25의 성과를 기록하며 새로운 최첨단 결과를 세우고 있습니다. 특히 도전적인 데이터셋인 WhatsUp에서는 최대 85.7%의 상대적 성과 향상이 이루어졌습니다. 연구에서 TTM은 평가 메트릭의 변화를 극복하며 일관되게 모델 성능을 향상시키는 데 효과적입니다.



### LLM Unlearning Under the Microscope: A Full-Stack View on Methods and Metrics (https://arxiv.org/abs/2510.07626)
- **What's New**: 최근 LLM(대형 언어 모델) 관련 연구는 모델의 안전성, 프라이버시 및 저작권 문제를 해결하기 위한 기계 학습 해제(machine unlearning)의 필요성을 강조하고 있습니다. 본 연구는 기존의 무질서한 연구 상황 속에서, 12개의 최신 상태 기반(unlearning) 방법을 세 가지로 분류하는 체계적 관점을 제공합니다: divergence-driven optimization, representation misalignment, rejection-based targeted unlearning. 이러한 방법의 성능 평가를 위해 새로운 Open-QA(오픈 질문 응답) 메트릭스를 제안하며, 이는 LLM의 생성 성능을 더 잘 포착할 수 있습니다.

- **Technical Details**: 연구에서는 기존의 평가 방식이 다각적인 관점을 제공하지 못하는 점을 지적하며, Open-QA를 통해 보다 정교한 평가를 요구합니다. 12개의 최근 LLM 해제 방법은 크게 세 가지 방법론으로 나뉘며, 각각의 알고리즘은 유사성을 지닌 하위 범주로 세분화됩니다. 예를 들어, divergence-driven optimization 방법은 모델이 기준 분포에서 멀어지도록 최적화하는 반면, rejection-based targeted unlearning 방법은 잊어야 할 쿼리에 명확한 거부 응답을 생성하는 방식을 사용합니다.

- **Performance Highlights**: WMDP(Weapons of Mass Destruction Proxy) 벤치마크를 활용하여 해제 효과성(UE)과 유틸리티 유지(UT) 측면에서 성능을 평가하였지만, 기존의 다중 선택 질문(MCQ) 기반의 평가 방식은 모델의 실제 생성 행동을 반영하지 못하는 경우가 많았습니다. 연구팀은 다양한 LLM 해제 방법들의 로버스트니스(강건성) 분석을 통해, 모델 레벨 공격과 관련된 다양한 취약점을 분석하였으며, 이는 향후 LLM 해제 기술 평가에 매우 중요한 시사점을 제공합니다.



### CompassLLM: A Multi-Agent Approach toward Geo-Spatial Reasoning for Popular Path Query (https://arxiv.org/abs/2510.07516)
- **What's New**: 이번 연구에서는 CompassLLM이라는 혁신적인 다중 에이전트 프레임워크를 소개하고, 이 모델이 대규모 언어 모델(LLM)의 추론 능력을 활용하여 인기 있는 경로 질의를 해결하기 위해 어떻게 설계되었는지를 설명합니다. 이 시스템은 2단계 파이프라인을 통해 운영되며, 첫 번째 단계인 SEARCH에서는 인기 있는 경로를 식별하고, 두 번째 단계인 GENERATE에서는 기존의 경로 데이터가 없을 때 새로운 경로를 생성합니다. 또한 CompassLLM은 실제 및 합성 데이터셋에서 우수한 정확도를 보이며 경제적 효율성을 강조합니다.

- **Technical Details**: CompassLLM은 LLM의 그래프 구조 이해 및 추론 능력을 활용하여 공간 데이터의 인기 있는 경로 질의를 처리하는 데 초점을 맞춥니다. 이 프레임워크는 속도가 빠르고 비용 효율적인 실시간 추론을 지원하며, 특히 Sparse한 데이터 환경에서도 기존의 다른 방법들보다 더 나은 성능을 발휘합니다. 더불어, LLM 기반 방법의 특별한 장점도 포함되어 있어 전통적인 알고리즘이나 머신러닝 접근법보다 더 나은 일반화 성능을 보여줍니다.

- **Performance Highlights**: CompassLLM의 평가 결과, SEARCH 단계에서 기존 방법들보다 현저히 뛰어난 성과를 나타내었고, GENERATE 단계에서도 경쟁력 있는 성능을 보여주었습니다. 실험은 사용자 이동 추적 데이터와 다양한 공간 구성을 반영한 합성 데이터셋을 사용하여 수행되었으며, CompassLLM은 저희 연구에서 제안한 경로 추천 시스템의 최고 수준(SOTA) 모델들과 비교할 수 있는 성능을 발휘합니다.



### Evaluation of LLMs for Process Model Analysis and Optimization (https://arxiv.org/abs/2510.07489)
Comments:
          15 pages, 5 tables, 4 figures; full research paper currently under review for the Workshop on Information Technologies and Systems (WITS) 2025. The paper presents a comprehensive evaluation of large language models (LLMs) for business process model analysis and optimization, including error detection, reasoning, and scenario-based redesign

- **What's New**: 이번 논문에서는 여러 LLM(대형 언어 모델)의 경험을 공유합니다. 이 모델들은 대화형 스타일로 프로세스 모델을 이해하고, 문법적(syntactical) 및 논리적(logical) 오류를 찾고, 이를 자연어(NL) 인터페이스를 통해 깊이 있게 추론하는 능력을 평가하였습니다.

- **Technical Details**: 연구 결과에 따르면, 훈련되지 않은 기본 LLM인 ChatGPT(모델 o3)는 제로샷(zero-shot) 환경에서 BPMN 프로세스 모델을 이미지로부터 효과적으로 이해하고, 그에 대한 질문에 문법적, 논리적, 의미적(semantic) 수준에서 지능적으로 답변할 수 있음을 보여주었습니다. 다양한 LLM들은 정확성(accuracy)과 효과성(effectiveness) 면에서 성능 차이를 보입니다.

- **Performance Highlights**: 경험적 분석(emirical analysis) 결과, LLM이 비즈니스 프로세스 설계자와 사용자에게 유용한 조력자 역할을 수행할 수 있음을 확인하였습니다. 또한 프로세스 분석 및 최적화(context of process analysis and optimization)에서 LLM의 '사고 과정'(thought process)과 더 깊은 추론 능력을 연구하였고, LLM들이 인격적(anthropomorphic) 특성을 나타내는 경향이 있음을 발견하였습니다.



### PATCH: Mitigating PII Leakage in Language Models with Privacy-Aware Targeted Circuit PatcHing (https://arxiv.org/abs/2510.07452)
- **What's New**: 이 논문에서는 언어 모델의 개인 식별 정보(PII) 유출 문제를 해결하기 위한 PATCH(Privacy-Aware Targeted Circuit PatcHing)라는 새로운 접근 방식을 제안합니다. PATCH는 특정 PII 유출 회로를 식별하고 수정함으로써 기존의 방어 메커니즘보다 더 나은 개인 정보 보호와 유용성의 균형을 제공합니다. 연구 결과, PATCH는 PII 유출의 회수를 최대 65%까지 감소시킬 수 있으며, 기존 방어와 결합 시 잔여 유출을 0.01%까지 낮출 수 있습니다.

- **Technical Details**: PATCH 접근법은 언어 모델의 내부 계산 구조(회로)를 발견하기 위해 방법론적 해석 가능성(mechanistic interpretability)을 사용합니다. EAP-IG(Edge Attribution Patching with Integrated Gradients)라는 회로 발견 메커니즘을 통해 PII 유출에 영향을 미치는 특정 회로를 식별하고 이를 수정하는 과정을 포함합니다. 이는 더 나은 개인 정보 보호를 위해 PII 유출을 담당하는 회로를 효과적으로 수정하기 위한 단계적인 방법을 제공합니다.

- **Performance Highlights**: PATCH는 다양한 설정에서의 강력함을 입증하기 위해 광범위한 실험을 수행하였고, 기존의 데이터 처리 방어 메커니즘보다 높은 성능을 보여주었습니다. 연구에서 PATCH는 다양한 PII 타입에 대해 유출 회로를 정확하게 식별하고, 이를 수정함으로써 개인 정보 유출 위험을 상당히 줄일 수 있음을 보여주었습니다. 또한, 해당 접근법은 기존의 차별적 개인정보 보호(differential privacy)와도 결합하여 추가적인 유출을 최소화하는 성과를 달성하였습니다.



### Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts (https://arxiv.org/abs/2510.07358)
- **What's New**: 본 논문에서는 언어 모델의 추론 능력을 향상시키기 위한 새로운 방법인 Encode-Think-Decode (ETD)를 제안합니다. ETD는 모델의 기본 구조와 매개변수 수를 보존하면서도 미리 훈련된 모델이 특정 층에서 반복적으로 추론을 할 수 있도록 훈련시킵니다. 이 방법은 모델의 추론 성능을 크게 개선할 수 있는 간단하고 효과적인 경로를 제공합니다.

- **Technical Details**: ETD 방식은 모델이 인퍼런스 단계에서 선택된 층들을 반복적으로 활용하여 동작하도록 합니다. 이를 통해 기존의 LLM(large language models)의 잠재적인 추론 능력을 개선할 수 있습니다. 이전 연구들은 서로 다른 층에서의 정보 흐름이 추론 과정에서 중요한 역할을 한다고 밝히고 있으며, 본 연구는 이러한 사실을 바탕으로 층의 역할을 고려하여 ETD 구조를 설계하였습니다.

- **Performance Highlights**: 본 연구에서는 OLMo-2 1B 베이스 모델을 사용하여 17개의 추론 벤치마크에서 성능을 평가한 결과, GSM8K에서 28.4%의 상대적 정확도 향상과 MATH에서 36%의 향상을 달성했습니다. 이는 복잡한 구조의 변경 없이도 기존 모델의 성능을 높일 수 있는 가능성을 보여줍니다. 또한, 입력 토큰에 따라 동적으로 계산 깊이를 조정하는 방법을 제안하여 보다 효율적인 추론이 가능해졌습니다.



### ConCuR: Conciseness Makes State-of-the-Art Kernel Generation (https://arxiv.org/abs/2510.07356)
- **What's New**: 본 연구는 LLM을 활용한 GPU 커널 생성의 최근 발전을 다룹니다. 고급 CUDA 커널의 부족 문제를 해결하기 위해, 우리는 reasoning trace(추론 추적)를 포함한 고품질 CUDA 커널을 생성 및 큐레이팅하는 파이프라인을 개발했습니다. 이를 통해 ConCuR이라는 데이터셋과 KernelCoder라는 모델을 처음으로 소개합니다.

- **Technical Details**: 우리가 개발한 데이터 수집 및 큐레이션 파이프라인은 두 가지 부분으로 구성됩니다: 데이터 합성 및 데이터 큐레이션입니다. 데이터 합성 부분에서는 기존의 reasoning 모델을 사용하여 CUDA 커널과 함께 reasoning trace를 합성합니다. 데이터 큐레이션 부분에서는 reasoning trace의 간결성과 커널의 성능을 고려하여 높은 품질의 데이터셋을 구축합니다.

- **Performance Highlights**: KernelCoder는 ConCuR 데이터셋을 기반으로 훈련되어, 기존의 최상위 모델인 QwQ-32B에 비해 뛰어난 성능을 보입니다. KernelBench에서의 평가를 통해, KernelCoder가 모든 기존 커널 생성 모델보다 우수하며, GPT-4와 같은 최신 모델보다도 더 좋음을 증명했습니다. 추가적으로, reasoning length(추론 길이)가 커널 생성 작업의 복잡도를 평가하는 지표로 유용하다는 사실을 발견하였습니다.



### Large Language Models Hallucination: A Comprehensive Survey (https://arxiv.org/abs/2510.06265)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 환각(hallucination) 현상에 대한 포괄적인 리뷰를 제공합니다. 환각은 LLM이 유창하고 문법적으로 정확하지만 사실적으로 부정확한 내용을 생성하는 현상으로, 특히 사실적 정확성이 요구되는 분야에서의 신뢰성을 저하시킬 수 있습니다. 본 연구는 환각의 원인, 탐지(detection) 및 완화(mitigation) 전략에 대한 새로운 분류 체계를 제안합니다.

- **Technical Details**: 환각의 발생 원인을 LLM의 개발 생애주기 전반에 걸쳐 분석하여 데이터를 수집하고, 아키텍처를 설계하며, 추론(inference) 단계에서 환각이 어떻게 발생하는지를 살펴봅니다. LLM의 개발 파이프라인은 데이터 수집 및 준비, 모델 아키텍처, 사전 훈련(pre-training), 미세 조정(fine-tuning), 평가(evaluation), 추론으로 나누어 각 단계에 기인한 환각의 원인을 식별합니다. 또한, 환각 탐지 기법은 검색-, 불확실성-, 임베딩(embedding)-, 학습-, 자기 일관성(self-consistency) 기반으로 분류됩니다.

- **Performance Highlights**: 환각 완화 기법은 프롬프트(prompt), 검색, 추론, 모델 중심의 훈련 및 적응으로 나누어지고, 각 기법의 가능성을 깊이 논의합니다. 검토된 데이터셋과 평가 지표는 기존 기법의 한계를 나타내며, 다국어 및 저자원 환경에서 환각을 완화하는 접근 방식을 제안합니다. 이 연구는 LLM의 신뢰성을 향상시키기 위한 미래 연구 방향을 정리하고, 효과적인 탐지 및 완화 기술 개발을 위한 기초를 마련합니다.



### Submodular Context Partitioning and Compression for In-Context Learning (https://arxiv.org/abs/2510.05130)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)에서 효과적인 'in-context learning' (ICL)을 위한 Submodular Context Partitioning (Sub-CP)이라는 새로운 프레임워크를 제안합니다. Sub-CP는 정보 포괄성을 제어하기 위해 서브모듈러(submodular) 목표를 활용하여 블록 간의 다양성을 조정합니다. 블록 선택 전략에 대한 유연한 스펙트럼을 지원하며, 전역적 다양성(global diversity)에서 로컬의 일관성(local coherence)까지 다양하게 조정할 수 있는 기능을 제공합니다.

- **Technical Details**: Sub-CP는 예제 임베딩에 대한 유사성 행렬을 기반으로 하는 시설 위치 함수(facility location function)를 이용하여 블록의 다양성을 모델링하는 일반적인 프레임워크입니다. 이를 통해 데이터셋의 다양한 정보를 수집하도록 각 블록이 최적화되며, 각 블록의 정보 포괄성을 보장합니다. Sub-CP는 전역적 다양성, 전역-로컬 다양성, 로컬 다양성, 로컬 일관성의 네 가지 전략을 통해 블록의 정보 구조를 효과적으로 조절합니다.

- **Performance Highlights**: 다양한 분류 데이터셋을 통해 Sub-CP를 평가한 결과, 모든 실험에서 일관된 성능 향상을 보였습니다. 예를 들어, SST-5와 TREC와 같은 어려운 데이터셋에서 Sub-CP는 서브모듈러 기반의 컨텍스트 파티셔닝이 무작위 표본 추출에 비해 성능을 개선함을 보여주었습니다. 특히, TREC 데이터셋에서 Local Diverse는 베이스라인 대비 29.2%의 절대적인 성능 향상을 이루어냈고, ICAE 및 CEPE와 같은 다른 ICL 프레임워크에서도 유의미한 성과를 보였습니다.



New uploads on arXiv(cs.IR)

### Mobile Gamer Lifetime Value Prediction via Objective Decomposition and Reconstruction (https://arxiv.org/abs/2510.08281)
Comments:
          6 pages, 6 figures

- **What's New**: 본 논문에서는 모바일 게임 분야의 사용자 LTV (Lifetime Value) 예측에 대한 혁신적인 접근법인 CALTV (Cumulative Amount LTV) 모델을 제안합니다. 이 모델은 복잡한 LTV 분포의 도전 과제를 해결하기 위해 목표 분해(Objective Decomposition) 및 재구성(Reconstruction) 프레임워크를 사용합니다. CALTV는 사용자의 거래 수를 특정 가격 범주에서 예측하고, 이를 바탕으로 전체 지불 금액을 계산하는 두 단계 프로세스를 포함합니다. 이를 통해 전통적인 회귀 모델의 외굶자(outlier) 영향을 효과적으로 완화하였습니다.

- **Technical Details**: CALTV 모델은 기본 DNN 구조로 구성되며, 거래 수 예측을 위한 분해 및 재구성 네트워크(DR Nets)와 중간 예측 값을 집계하여 최종 LTV를 생성하는 추가 재구성 레이어를 포함합니다. 사용자의 총 지불 금액은 여러 거래 주문의 지불 금액을 합산한 값으로 정의되며, 이를 통해 거래 주문을 중앙 가격 기준으로 분류할 수 있습니다. 모델은 모바일 게임 내 구매 행태에 최적화되어 설계되었으며, 가격 카테고리별 거래 수량 예측을 중점적으로 수행합니다.

- **Performance Highlights**: 제안한 CALTV 모델은 내부 데이터셋을 사용한 오프라인 실험 및 RTB 광고 시스템에 대한 온라인 A/B 테스트 결과, 예측 정확도가 유의미하게 향상되었습니다. 특히, 일반적인 회귀 방법이 LTV 분포의 왜곡에 민감한 것과 달리, CALTV는 거래 수 예측에 초점을 맞춰 성능을 개선하였습니다. 이 모델은 향후 RTB 광고 시스템의 광고 효율성 및 투자 수익률을 극대화하는 데 기여할 것으로 기대됩니다.



### ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieva (https://arxiv.org/abs/2510.08252)
Comments:
          17 pages, 3 figures

- **What's New**: 본 연구에서는 ReasonEmbed라는 새로운 텍스트 임베딩 모델을 소개합니다. 이 모델은 추론 집약적인 문서 검색을 위해 개발되었으며, 주요 기여로 ReMixer라는 새로운 데이터 합성 방법과 Redapter라는 자기 적응형 학습 알고리즘을 제안합니다. 이 두 가지 기법을 통해 고품질의 훈련 샘플을 대규모로 생성하고, 각 샘플의 추론 강도를 기반으로 가중치를 조정하여 효과적인 데이터 활용을 가능하게 합니다.

- **Technical Details**: 우리는 ReMixer를 통해 조절된 쿼리 생성, 소스 제외 후보 발굴, 추론 강화 관련성 주석 달기 등의 세 가지 단계로 이루어진 데이터 합성 워크플로우를 설계합니다. 이어서 Redapter를 통해 합성 데이터의 추론 강도에 따라 각 훈련 샘플의 가중치를 동적으로 조정하는 방법을 소개하며, 이는 다양한 규모의 LLM 백본을 기반으로 ReasonEmbed를 구현하여 최신 성능을 달성하는 데 기여합니다.

- **Performance Highlights**: ReasonEmbed-Qwen3-8B 모델은 BRIGHT 벤치마크에서 nDCG@10 점수 38.1을 기록하며 기존 텍스트 임베딩 모델을 크게 초월하는 성과를 보였습니다. 이 모델은 합성 데이터와 자기 적응형 학습 알고리즘의 기여를 통해 성능 향상을 이룬 것을 실증적으로 분석하였습니다. 모든 자원은 오픈 소스 형태로 공개되어 향후 연구에 도움을 줄 예정입니다.



### VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents (https://arxiv.org/abs/2510.08109)
- **What's New**: 이번 논문에서는 기술 문서의 버전 관리 문제를 해결하기 위한 새로운 Retrieval-Augmented Generation (RAG) 시스템인 VersionRAG를 제안합니다. 기존 RAG 시스템들은 문서의 진화에 따른 정확한 답변을 제공하지 못하고 있으며, 특히 버전이 중요한 경우에서의 결과 정확도가 낮습니다. VersionRAG는 문서의 변화 과정을 계층 그래프 구조로 모델링하여 버전별 필터링 및 변화 추적을 가능하게 합니다.

- **Technical Details**: VersionRAG의 핵심 요소는 세 가지 쿼리 유형을 정의하고, 이 각각에 대해 특화된 탐색과 필터링 메커니즘을 사용하는 것입니다. 문서 버전은 계층 그래프 노드와 엣지로 표현되어 있으며, 각 노드는 문서의 카테고리, 개별 문서, 버전, 내용 및 변경 사항을 나타냅니다. 이 구조는 문서 쿼리에 대한 효율적인 그래프 탐색을 가능하게 하여 정확도 및 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: VersionRAG는 34개의 버전이 있는 기술 문서에서 100개의 수작업으로 구성된 질문에 대해 90%의 정확도를 달성했습니다. 이는 기존의 RAG(58%) 및 GraphRAG(64%)와 비교하여 현저히 높은 성능을 보입니다. 또한, 인덱싱 과정에서 GraphRAG에 비해 97% 적은 토큰을 필요로 하여 대규모 배포 시 효율성을 확보하고 있습니다.



### TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevanc (https://arxiv.org/abs/2510.08048)
- **What's New**: 본 연구에서는 TaoSR-AGRL이라는 Adaptive Guided Reinforcement Learning 프레임워크를 제안합니다. 이 프레임워크는 Taobao 검색 관련성 예측을 강화하기 위해 설계되었고, 두 가지 주요 혁신을 포함합니다: Rule-aware Reward Shaping 및 Adaptive Guided Replay. 이러한 접근 방식은 복잡한 비즈니스 규칙과 사용자 쿼리의 변화하는 요구를 충족시키기 위해 긴-tail 문제를 해결하고자 합니다.

- **Technical Details**: TaoSR-AGRL의 핵심 모듈 중 하나인 Rule-aware Reward Shaping은 최종 관련성 판단을 밀집화된 구조화된 보상으로 분해하여 도메인별 기준에 맞게 조정합니다. Adaptive Guided Replay는 학습 중 낮은 정확도의 롤아웃을 식별하여 궁극적으로 정책이 정체된 추론 패턴에서 벗어날 수 있도록ground-truth 정보를 주입합니다. 이로 인해 모델은 높은 가치의 추론 경로를 탐색하고 개선할 수 있습니다.

- **Performance Highlights**: TaoSR-AGRL은 대규모 실제 데이터세트를 기반으로 평가되었으며, DPO 및 기존 GRPO 기반라인을 일관되게 초과하여 관련성 정확도, 규칙 준수 및 훈련 안정성을 개선했습니다. 이 모델은 Taobao의 주요 검색 시나리오에 성공적으로 배포되어 수억 명의 사용자에게 서비스를 제공하고 있으며, 연구 결과는 해당 산업 분야에서의 실질적인 가치를 보여줍니다.



### Generation and annotation of item usage scenarios in e-commerce using large language models (https://arxiv.org/abs/2510.07885)
- **What's New**: 이 논문에서는 e-commerce에서 유용한 아이템 조합을 제안하는 complementary recommendations에 대해 다룹니다. 기존의 통계적 공기반적인(co-occurrence) 방법 대신, 아이템 조합을 유도하는 사용 맥락(context)에 집중합니다. 이렇게 함으로써, 사람들은 특별한 사용 시나리오를 상상함으로써 보완적인 아이템을 선택한다고 가설을 세웠습니다.

- **Technical Details**: 연구에서는 큰 언어 모델(LLMs)을 활용하여 보완적인 추천 시스템을 구축하기 위한 아이템 사용 시나리오를 생성하였습니다. LLM이 생성한 시나리오의 신뢰성을 수동 주석(annotation) 작업을 통해 평가하였으며, 약 85%의 생성된 시나리오가 그럴듯하게 평가되었습니다. 이를 통해 LLM이 현실적인 아이템 사용 시나리오를 생성할 수 있음을 보여주었습니다.

- **Performance Highlights**: LLM을 사용한 접근 방식은 기존의 역사 기반(history-based) 방법보다 더 유망한 결과를 도출하였습니다. 연구 결과는 보완적인 추천 시스템의 개발에 있어 LLM의 활용 가능성을 제시하며, 이는 e-commerce의 사용자 경험을 향상시킬 수 있는 중요한 발전으로 평가되고 있습니다.



### PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations (https://arxiv.org/abs/2510.07784)
Comments:
          11 pages, 6 figures

- **What's New**: 이번 논문에서는 PLUM이라는 프레임워크를 소개하며, 이는 사전 학습된 Large Language Models (LLMs)을 산업 규모의 추천 시스템에 적합하게 조정하는 방식입니다. PLUM은 Semantic IDs (SIDs)를 활용한 항목 토큰화, 도메인 특화 데이터에 대한 지속적 사전 학습(Continued Pre-training, CPT), 그리고 추천 목표를 위한 작업 특화 미세 조정(task-specific fine-tuning)으로 구성됩니다. 이러한 접근 방식은 전통적인 삽입 테이블 방식의 한계를 극복하고 LLM의 능력을 최대한 활용하기 위한 노력의 일환으로 볼 수 있습니다.

- **Technical Details**: PLUM 프레임워크는 항목 토큰화 단계에서 각 항목을 SIDs로 표현하여 사용자의 행동 신호와 다중 모드 콘텐츠 임베딩을 향상시키는 새로운 기술(SID-v2)을 제공합니다. 지속적 사전 학습 단계에서는 사전 학습된 LLM의 어휘를 새 SID 토큰으로 확장하고, 도메인 특화 항목 데이터와 사용자 시퀀스의 혼합으로 모델을 추가로 학습시킵니다. 마지막으로, 작업 특화 미세 조정 단계에서 모델은 사용자가 관심 가질 항목의 SIDs를 생성하는 방식으로 Fine-tuning됩니다.

- **Performance Highlights**: 대규모 내부 비디오 추천 데이터셋에서 PLUM을 적용한 실험 결과, 기존의 대규모 정보 테이블을 기반으로 한 최적화 모델에 비해 PLUM 사용 시 검색 성능이 크게 향상됨을 보였습니다. PLUM을 사용한 모델은 Transformers 아키텍처에서 100배 더 많은 밀집 매개변수를 가지면서도 훈련 비용은 유사하게 유지되며, 샘플 효율성을 높일 수 있음을 보여주었습니다. 또한, PLUM 기반 검색은 YouTube에서 실제 운영 중으로, 장기 및 단기 비디오를 추천하는 데 효과적으로 활용되고 있습니다.



### Who Stole Your Data? A Method for Detecting Unauthorized RAG Thef (https://arxiv.org/abs/2510.07728)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 시스템의 도용(detection) 문제를 해결하기 위한 두 가지 주요 기여를 제공합니다. 첫째, 다양한 전문 분야와 글쓰기 스타일을 아우르는 RPD라는 새로운 데이터셋을 소개하여 기존 자원의 한계를 극복하였습니다. 둘째, 의미적 및 어휘적 수준에서 보호 기능을 내장한 이중 레이어 워터마킹 시스템을 개발하고, 통계적 가설 검정을 사용하는 인터로게이터-탐정 프레임워크를 구축하였습니다.

- **Technical Details**: 제안된 방법은 RAG 시스템을 통해 생성된 콘텐츠의 독창성을 보호하는 데 필수적입니다. 이 연구는 기초적인 지식 워터마크를 의미적 차원에서 삽입하고, 통계적 서명 패턴을 활용하여 효과적으로 감지 메커니즘을 제공합니다. 새로운 인터로게이터-탐정 프레임워크는 불법적인 데이터 사용을 밝혀내기 위해 전략적으로 쿼리를 생성하고, 축적된 증거에 대한 철저한 통계 분석을 수행합니다.

- **Performance Highlights**: 결과적으로, 본 논문에서 제안한 접근 방식은 다양한 쿼리 볼륨, 방어 프롬프트, 조회 매개변수에 걸쳐 효과성을 입증하였으며, 적대적 회피 기술에 대해서도 강인성을 유지하고 있습니다. 이러한 성과는 지식 생성의 유인을 보장하고 정보 생태계를 보호하는 데 중요한 기초를 마련합니다.



### Queries Are Not Alone: Clustering Text Embeddings for Video Search (https://arxiv.org/abs/2510.07720)
Comments:
          Accepted by International ACM SIGIR Conference on Research and Development in Information Retrieval 2025

- **What's New**: 영상 콘텐츠의 급속한 확산으로 인해 고급 영상 검색 시스템의 필요성이 대두되고 있습니다. 전통적인 방법들은 텍스트 쿼리와 메타데이터를 간단히 비교통하여 단순히 일치하는 방식을 사용했지만, 이번 논문은 Video-Text Cluster (VTC)라는 새로운 프레임워크를 도입하여 쿼리를 클러스터링함으로써 더 넓은 의미의 영상 검색을 가능하게 합니다. 이로 인해 시스템은 각 쿼리의 다양한 해석과 뉘앙스를 고려합니다.

- **Technical Details**: VTC의 개발은 고유한 클러스터링 메커니즘을 사용하여 관련 쿼리를 그룹화하며, 이를 통해 여러 의미를 포착할 수 있습니다. 클러스터링은 Sweeper 모듈을 통해 노이즈를 식별하고 완화하여 더욱 정제됩니다. 또한, Video-Text Cluster-Attention (VTC-Att) 메커니즘을 소개하여 비디오 콘텐츠에 기반하여 클러스터 내의 초점을 동적으로 조정함으로써 가장 관련성 높은 텍스트 특성에 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다섯 개의 공개 데이터셋에서 기존의 최첨단 모델들을 초월하는 성과를 보여주었습니다. VTC에 기반한 새로운 접근 방식은 텍스트 클러스터를 중심으로 한 혁신적인 검색 체계를 제안하며, 이는 쿼리와 영상 간의 의미적 간극을 효과적으로 메울 수 있도록 합니다.



### ISMIE: A Framework to Characterize Information Seeking in Modern Information Environments (https://arxiv.org/abs/2510.07644)
Comments:
          This paper has been accepted to SIGIR-AP 2025

- **What's New**: 현대 정보 환경(MIE)이 복잡해지면서, 사용자 정보 요구를 충족하기 위한 다양한 기술들이 발전하고 있습니다. 사용자-시스템 상호작용을 설명하기 위한 정보 탐색(IS) 모델은 중요한 역할을 하나, 현재의 복잡한 환경을 완벽하게 포착할 수 있는 모델은 부재합니다. 이에 ISMIE(Information Seeking in Modern Information Environments) 프레임워크를 제안하여 정보 탐색 프로세스(ISP)를 구조화하고, 기존의 정보 검색(IR) 모델들의 한계를 분석합니다.

- **Technical Details**: ISMIE는 세 가지 핵심 개념인 구성 요소(Components), 변동 변수(Intervening Variables), 활동(Activities)을 통해 정보 탐색 프로세스를 분석합니다. 기존의 모델들을 통해 특정 시나리오인 허위 정보 유포를 예로 들어, 어떠한 모델이 이러한 복잡성의 모든 측면을 포착할 수 없음을 증명합니다. 이를 통해 ISMIE가 정보 탐색 및 정보 검색 모델의 발전에 기여할 수 있는 기초를 제공한다고 주장합니다.

- **Performance Highlights**: ISMIE 프레임워크는 사용자 중심의 실험 디자인 및 시스템 중심의 연구 설계를 통해 정보의 진정성과 신뢰성 위기를 다루며, 도파민 기반 컨텐츠 소비 문제를 해결하는 데 실질적으로 적용될 수 있습니다. 이 연구 결과는 정보 환경에 대한 이해를 넓히고 현대 정보 탐색의 복잡성을 해석하는 데 도움이 될 것입니다. ISMIE는 궁극적으로 연구자들이 현대 정보 탐색의 행동을 보다 효과적으로 모델링하는 데 기여할 수 있는 기초로 자리잡을 것입니다.



### Retentive Relevance: Capturing Long-Term User Value in Recommendation Systems (https://arxiv.org/abs/2510.07621)
- **What's New**: 이 논문에서는 Retentive Relevance라는 새로운 콘텐츠 수준의 설문 기반 피드백 지표를 소개합니다. 이는 사용자가 유사한 콘텐츠를 위해 플랫폼에 돌아올 의도를 직접 평가하여, 장기적인 사용자 만족도와 유지율을 측정할 수 있는 강력한 예측 도구로 제안됩니다. Retentive Relevance는 기존의 단기적인 만족도를 중심으로 한 설문 지표와는 차별화됩니다. 이를 통해 우리는 사용자 행동의 날짜별 유지율을 개선하는 데 성공했습니다.

- **Technical Details**: Retentive Relevance는 심리측정(psychometric) 방법을 통해 검증된 유효한 설문 구성으로, 사용자 만족과 의도를 동시에 측정함으로써 추천 시스템에서 유용한 도구로 자리잡고 있습니다. 이 연구는 대규모 오프라인 모델링, A/B 실험 등을 통해 Retentive Relevance가 전통적인 참여 신호와 다른 설문 지표들보다 더 뛰어난 예측 성능을 보임을 입증하였습니다. 따라서 우리는 추천 시스템에서 Retentive Relevance를 통합하는 프로덕션 준비 모델을 개발하여, 실질적인 운영에서의 적용 가능성을 높였습니다.

- **Performance Highlights**: 저자들은 Retentive Relevance가 사용자 유지율, 참여도, 콘텐츠 품질을 향상하는 데 효과적이라는 것을 대규모 실험을 통해 보여주었습니다. 특히 제한된 역사적 참여 기록을 가진 사용자들에게 전통적인 지표보다 더 좋은 성과를 내는 것이 특징입니다. 이에 따라 플랫폼의 성장과 사용자 경험을 동시에 개선할 수 있는 확장 가능하고 사용자 중심의 솔루션을 제공하게 되었습니다.



### Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs (https://arxiv.org/abs/2510.07484)
- **What's New**: 이 논문은 Reasoning by Exploration (RoE)라는 새로운 접근 방식을 제안하여 대규모 그래프에서의 추론과 생성 과정을 통합합니다. RoE는 그래프 탐색 과정을 단계적으로 구성하여 LLM이 후보 노드와 엣지를 선택하며 답변을 생성합니다. 이 방법론은 기존의 리트리벌-증강 생성(RAG) 시스템이 갖는 한계를 극복하고, 그래프 구조의 신뢰성을 더욱 향상시킵니다.

- **Technical Details**: RoE는 두 단계의 훈련 전략을 채택합니다. 첫 번째 단계에서는 금 고찰 경로(gold reasoning paths)에 대해 지도 미세 조정(supervised fine-tuning, SFT)을 통해 LLM이 단계적으로 노드와 엣지를 확장하는 방법을 학습하도록 합니다. 두 번째 단계에서는 강화 학습(reinforcement learning, RL)을 적용하여 탐색 효율성을 극대화하고 다양한 그래프에 대한 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RoE는 여러 다중 단계 추론 벤치마크에서 기존의 최첨단 기준선 모델들보다 월등한 성능을 보입니다. 또한 RoE는 다양한 보지 않은 그래프에 대해서도 강력한 일반화 능력을 입증하였습니다. 이는 RoE가 제공하는 통합 탐색 및 생성 프로세스가 실제 응용에서의 실용성을 크게 높였음을 시사합니다.



### Agent Learning via Early Experienc (https://arxiv.org/abs/2510.08558)
Comments:
          Work in progress

- **What's New**: 이번 연구는 언어 에이전트들이 스스로 경험을 통해 학습하고 향상될 수 있는 가능성을 탐구합니다. 특히, 방식이 부족한 환경에서의 강화 학습의 한계를 극복하기 위해 'early experience'(얼리 익스피리언스)라는 새로운 개념을 도입했습니다. 이는 에이전트의 행동으로 생성된 상호작용 데이터로, 보상 신호 없이도 미래 상태를 감독할 수 있게 합니다.

- **Technical Details**: 연구에서는 두 가지 전략을 통해 초기 경험 데이터를 활용합니다. 첫 번째는 'Implicit world modeling'(암묵적 세계 모델링)으로, 수집된 상태를 사용하여 정책을 환경 동역학에 맞게 조정합니다. 두 번째는 'Self-reflection'(자기 반성)으로, 에이전트가 비효율적인 행동에서 학습하여 추론과 의사결정을 개선하는 방법입니다.

- **Performance Highlights**: 8개의 다양한 환경과 여러 모델 계열을 통해 평가한 결과, 제안된 접근 방식이 효과성과 도메인 외 일반화를 일관되게 개선함을 보여주었습니다. 더욱이, 검증 가능한 보상이 있는 환경에서는 초기 경험이 후속 강화 학습을 위한 강력한 기초가 될 수 있다는 유망한 신호를 포착했습니다.



### Detecting Legend Items on Historical Maps Using GPT-4o with In-Context Learning (https://arxiv.org/abs/2510.08385)
- **What's New**: 이 연구는 역사적 지도에서 범례(legend)를 자동으로 추출하기 위한 새로운 방법을 발표합니다. LayoutLMv3를 사용한 레이아웃 감지와 GPT-4o를 활용한 인컨텍스트 학습(in-context learning) 기법을 결합하여 범례 항목과 그 설명을 연결합니다. 실험 결과, 구조화된 JSON 프롬프트를 사용한 GPT-4o가 기존 방법보다 뛰어난 성능을 보여주며, 역사적 지도의 색다른 시각적 스타일에 대한 인덱싱과 검색 가능성을 높이고 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다: 첫째, 전방위 맵에서 범례 영역을 분리하는 것입니다. 이를 위해 LayoutLMv3를 사용하여 범례가 포함된 블록을 식별하고, 그 후 GPT-4o를 통해 범례 항목과 설명에 대한 경계 상자(bounding box)를 생성합니다. 사용되는 JSON 프롬프트는 작업 정의와 함께 예시 항목과 설명의 쌍을 포함하여 GPT-4o에 제공됩니다.

- **Performance Highlights**: 제안된 방식의 성능은 88%의 F-1 점수와 85%의 IoU를 기록했습니다. GPT-4o는 다양한 예시 수(5, 10, 15, 20)에 따라 성능이 향상됨을 보여주었습니다. 이 연구는 역사적 지도 아카이브에서 범례 해석의 필요성을 강조하며, 스케일러블한 지리공간 검색과 마이닝을 위한 기초를 제공합니다.



### HySim-LLM: Embedding-Weighted Fine-Tuning Bounds and Manifold Denoising for Domain-Adapted LLMs (https://arxiv.org/abs/2510.07796)
- **What's New**: 이번 논문에서는 약리학적 정보의 추출 및 표준화에서의 한계를 극복하는 새로운 프레임워크인 HySim-LLM을 제안합니다. HySim-LLM은 embedding-weighted fine-tuning과 manifold-aware denoising을 결합하여 LLM을 구조화된 생물의학 데이터에 적용 가능하도록 개선하는 수학적 및 계산적 접근 방식을 제공합니다. 이를 통해 LLM의 강건성과 해석력을 높이는 새로운 이론적 기초를 마련하고자 합니다.

- **Technical Details**: HySim-LLM은 원천 데이터셋(S)과 특정 도메인 데이터셋(T) 간의 embedding 유사성을 활용하여 LLM 파라미터를 적응시키는 과정에서 provable 보장을 제공합니다. 구체적으로, cosine이나 Mahalanobis 거리와 같은 밀접도 지표를 사용하여 embedding divergence를 도입함으로써 fine-tuning의 성능 보장을 해석 가능한 방식으로 연결합니다. 또한, 임베딩 공간에서 off-manifold 샘플이 손실에 미치는 영향을 정량화하여 이를 기반으로 LLM 기반 파이프라인에서의 데이터 정리를 가능하게 합니다.

- **Performance Highlights**: 이 연구는 기존의 AutoPK 및 WCPK 시스템을 개선하여 PK 데이터 추출 및 정제 과정에서 높은 정확도와 일관성을 입증했습니다. 새로운 이론적 결과인 similarity-weighted fine-tuning bound와 manifold-based denoising theorem은 LLM을 생물의학 분야 및 데이터 집약적 과학 도메인에 더욱 신뢰할 수 있게 만들어 줄 것입니다. HySim-LLM은 이러한 이론적 요소들을 통해 기존의 생물의학 LLM 적용에서의 신뢰성 문제를 해결하는데 큰 기여를 예상하고 있습니다.



### Evaluation of LLMs for Process Model Analysis and Optimization (https://arxiv.org/abs/2510.07489)
Comments:
          15 pages, 5 tables, 4 figures; full research paper currently under review for the Workshop on Information Technologies and Systems (WITS) 2025. The paper presents a comprehensive evaluation of large language models (LLMs) for business process model analysis and optimization, including error detection, reasoning, and scenario-based redesign

- **What's New**: 이번 논문에서는 여러 LLM(대형 언어 모델)의 경험을 공유합니다. 이 모델들은 대화형 스타일로 프로세스 모델을 이해하고, 문법적(syntactical) 및 논리적(logical) 오류를 찾고, 이를 자연어(NL) 인터페이스를 통해 깊이 있게 추론하는 능력을 평가하였습니다.

- **Technical Details**: 연구 결과에 따르면, 훈련되지 않은 기본 LLM인 ChatGPT(모델 o3)는 제로샷(zero-shot) 환경에서 BPMN 프로세스 모델을 이미지로부터 효과적으로 이해하고, 그에 대한 질문에 문법적, 논리적, 의미적(semantic) 수준에서 지능적으로 답변할 수 있음을 보여주었습니다. 다양한 LLM들은 정확성(accuracy)과 효과성(effectiveness) 면에서 성능 차이를 보입니다.

- **Performance Highlights**: 경험적 분석(emirical analysis) 결과, LLM이 비즈니스 프로세스 설계자와 사용자에게 유용한 조력자 역할을 수행할 수 있음을 확인하였습니다. 또한 프로세스 분석 및 최적화(context of process analysis and optimization)에서 LLM의 '사고 과정'(thought process)과 더 깊은 추론 능력을 연구하였고, LLM들이 인격적(anthropomorphic) 특성을 나타내는 경향이 있음을 발견하였습니다.



### Haystack Engineering: Context Engineering for Heterogeneous and Agentic Long-Context Evaluation (https://arxiv.org/abs/2510.07414)
Comments:
          Code available at this https URL

- **What's New**: 이 논문은 HaystackCraft라는 새로운 NIAH 벤치마크를 소개하며, LLM의 긴 문맥에서의 강인함(long-context robustness)을 평가하기 위해 노이즈가 있는 긴 문맥을 구축하는 것이 중요하다고 주장합니다. 연구팀은 다수의 다단계 질문을 통해 영어 위키피디아의 하이퍼링크 네트워크를 기반으로 한 테스팅 환경을 설계했습니다. 이 새로운 평가 기준은 다양한 검색 전략이 LLM의 성능에 미치는 영향을 체계적으로 검토합니다.

- **Technical Details**: 이번 연구에서는 검색 증강 생성(RAG) 기술을 활용하여 LLM의 긴 문맥을 조작하는 방식을 다루며, 상이한 검색 자원들이 자료의 노이즈 및 복잡성을 어떻게 생성하는지를 탐구합니다. 또한, 그래프 기반의 검색 방식의 구현이 LLM의 성능 개선에 어떻게 기여할 수 있는지에 대한 실험을 수행하였습니다. 다양한 Retrieval 전략(예: sparse, dense, hybrid, graph-based)을 기반으로 하여, HaystackCraft는 에이전트의 작업 흐름 중 나타나는 누적 오류(cascading failures)에 대한 모델의 저항성을 평가합니다.

- **Performance Highlights**: 실험 결과, 강력한 밀집 검색기(dense retrievers)는 더 어려운 산만 요소(distractors)를 도입하는 반면, 그래프 기반의 재정렬(graph-based reranking)은 검색의 효과를 높이며 해로운 산만 요소를 줄이는 데 기여했습니다. 15개의 긴 문맥 LLM을 대상으로 한 테스트에서는 Gemini 2.5 Pro와 GPT-5와 같은 고급 모델도 누적 자기 산만 문제로 어려움을 겪는 것으로 나타났습니다. 이 결과는 에이전트의 긴 문맥 추론에 지속적인 도전이 남아 있음을 나타내며, HaystackCraft가 향후 발전을 측정하기 위한 중요한 시험대임을 강조합니다.



New uploads on arXiv(cs.CV)

### ReSplat: Learning Recurrent Gaussian Splats (https://arxiv.org/abs/2510.08575)
Comments:
          Project page: this https URL

- **What's New**: ReSplat 모델은 3D Gaussian splatting의 개선된 접근 방식을 제안합니다. 이는 feed-forward 방식의 단일 경과에 의존하는 것에서 벗어나, 반복적인 업데이트를 통해 Gaussian을 점진적으로 개선합니다. Gaussian rendering 오류를 피드백 신호로 활용하여 네트워크가 새로운 데이터 분포에 적응할 수 있게 합니다.

- **Technical Details**: ReSplat은 $16 	imes 16$으로 서브샘플링된 공간에서 Gaussian을 예측하는 Compact Reconstruction 모델을 도입합니다. 이를 바탕으로, 재귀 업데이트는 현재 Gaussians과 오류 신호 사이의 관계를 통해 Gaussian의 매개변수를 직접 예측합니다. 이러한 방식으로, 명시적인 gradient 계산 없이도 신뢰성 있는 업데이트를 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, DL3DV 데이터셋에서 PSNR이 +2.7 dB 개선되었습니다. 이때, 전체 Gaussian 수는 1/16로 줄어들고 렌더링 속도는 4배 향상되었습니다. ReSplat은 여러 입력 뷰와 다양한 해상도에서 이미 이전 모델들보다 우수한 결과를 기록하여, 복잡한 데이터셋에 대한 일반화와 성능 강화를 보여줍니다.



### MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning (https://arxiv.org/abs/2510.08567)
- **What's New**: 이 논문에서는 비전 언어 모델(VLMs)이 외부 도구를 활용하여 복잡한 추론과 의사결정 작업에서 어떻게 향상될 수 있는지를 탐구합니다. M-TRACE라는 큰 규모의 멀티모달 데이터셋을 구축하여 VLM 컨트롤러의 튜닝을 지원하며, MATRIX 에이전트를 통해 수동 주석의 필요한 비용을 줄이고 더 나은 일반화 성능을 발휘합니다. 또한, Pref-X라는 자동 생성된 선호 쌍(couples) 세트를 도입하여 단계별 선호 최적화를 통해 결정 과정을 정교화합니다.

- **Technical Details**: MATRIX는 두 단계의 프레임워크로 구성되어 있습니다. 첫 번째 단계에서는 M-TRACE에서 수집된 28.5K 멀티모달 작업을 활용하여 감독된 훈련을(trajectory-driven SFT) 진행합니다. 이어서, Pref-X를 사용하여 11K 개의 자동 생성된 선호 쌍을 기반으로 선호 최적화(preference optimization)를 통해 에이전트의 결정 과정을 정교화합니다. 이 과정 전반에 걸쳐 서로 검증된 경로(traces)를 활용하여 상황에 맞는 도구 사용 능력을 배양합니다.

- **Performance Highlights**: MATRIX는 Agent-X, GTA, GAIA 등 세 가지 벤치마크에서 기존 VLM보다 우수한 성과를 기록했습니다. 특히, 응답 정확도는 각각 14%, 23% 및 11% 향상되었습니다. 이 결과는 MATRIX의 구조적 도구 사용 능력과 단계별 언급 최적화가 효과적으로 작용했음을 보여줍니다. 또한, MATRIX는 이전 에이전트들보다 일관된 추론을 수행하고 더 적합한 도구 선택 능력을 보유하고 있습니다.



### D$^2$GS: Depth-and-Density Guided Gaussian Splatting for Stable and Accurate Sparse-View Reconstruction (https://arxiv.org/abs/2510.08566)
- **What's New**: 이번 연구에서는 sparse-view 환경에서 3D Gaussian Splatting (3DGS)의 성능 저하와 불안정을 개선하기 위한 새로운 방법인 D$^2$GS를 제안합니다. 이 프레임워크는 과도한 Gaussian 밀도가 발생하는 지역에서 발생하는 overfitting과 제한적인 Gaussian 커버리지를 가진 원거리 지역에서 발생하는 underfitting의 두 가지 주요 실패 모드를 식별합니다. D$^2$GS는 Depth-and-Density Guided Dropout과 Distance-Aware Fidelity Enhancement의 두 가지 핵심 구성 요소로 이루어져 있습니다.

- **Technical Details**: D$^2$GS 방법론은 깊이(depth)와 밀도(density) 정보를 기반으로 Gaussians의 dropout 점수를 부여하여 일정 지역의 overfitting을 방지합니다. 이 방법은 또한 원거리 지역에서의 supervision을 강화하여 underfitting을 완화하는 기능을 제공합니다. 제안된 방법은 LLFF 및 Mip-NeRF360 데이터셋에서의 평가를 통해 3DGS 모델의 안정성을 향상시키고, 새로운 평가 지표인 Inter-Model Robustness (IMR)를 도입하여 학습된 Gaussian 분포의 안정성을 정량화합니다.

- **Performance Highlights**: 본 연구는 제안된 D$^2$GS 프레임워크가 sparse-view 조건에서 시각적 품질과 안정성을 크게 향상시킨다는 것을 다수의 실험을 통해 입증하였습니다. 기존 방식보다 더 견고한 3D 재구성을 제공하며, novel view synthesis의 최신 기술 수준에 도달하였음을 보여줍니다. D$^2$GS는 Gaussian 분포 기반의 새로운 평가 지표를 도입하여 3D 표현 품질을 보다 직접적으로 평가할 수 있게 되었습니다.



### NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints (https://arxiv.org/abs/2510.08565)
Comments:
          Accepted by NeurIPS 2025. 22 pages, link: this https URL

- **What's New**: 이 논문은 Multimodal Large Language Models (MLLMs)의 네이티브 훈련을 다루고 있으며, 기존의 컴포지셔널 훈련 방식의 한계를 극복하고자 합니다. 특히, 데이터 제약 조건 하에서 MLLM의 설계 공간과 스케일링 특성을 체계적으로 분석하여 최적의 메타 아키텍처를 도출합니다. 새로운 MLLM인 NaViL을 제안하며, 이는 성능과 훈련 비용의 균형을 최적화한 결과로, 14개의 멀티모달 벤치마크에서 경쟁력 있는 성능을 나타냅니다.

- **Technical Details**: NaViL은 비전 인코더와 언어 모델의 최적화를 위해 엔드 투 엔드 방식으로 훈련됩니다. 연구진은 Mixture-of-Experts(MoE), 비전 인코더 및 LLM 초기화와 같은 주요 구성 요소를 탐색했습니다. 결과적으로, LLM 초기화가 멀티모달 데이터에 대한 훈련 수렴에 큰 이점을 준다는 것을 발견했으며, 비전 인코더 아키텍처와 MoE의 조합이 기존의 디코더 모델에 비해 개선된 성능을 보였습니다.

- **Performance Highlights**: NaViL은 약 600M 개의 이미지-텍스트 쌍을 사용해 훈련되었으며, 현재의 최전선 컴포지셔널 MLLM과 비교하여 경쟁력 있는 성능을 달성했습니다. 이번 연구는 네이티브 MLLMs의 미래 연구를 위한 심도 있는 통찰을 제공하며, 특히 데이터 제약 하에서도 최고 수준의 성능을 실현 가능성을 제기합니다. 이로써, MLLM 분야에서의 새로운 패러다임 전환을 촉진할 것으로 기대됩니다.



### ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving (https://arxiv.org/abs/2510.08562)
- **What's New**: 이번 연구에서는 End-to-End Autonomous Driving (E2EAD) 시스템의 한계를 극복하기 위해 ResAD라는 새로운 Normalized Residual Trajectory Modeling 프레임워크를 제안합니다. 기존의 시스템들이 미래의 주행 경로를 직접 예측하는 방식에서 벗어나, ResAD는 결정론적 관성 기준에 대한 잔여 편차를 예측하는 방향으로 학습 과제를 재구성합니다. 이를 통해 모델이 단순 패턴 인식을 넘어 뒤에 숨겨진 인과 관계를 이해하도록 유도합니다.

- **Technical Details**: ResAD는 차량의 현재 상태를 외삽하여 결정론적 관성 기준을 설정하는 과정에서 두 가지 구성 요소로 복잡한 예측 작업을 분해합니다. 첫 번째는 기본적인 물리 기반 모델을 사용하는 것이고, 두 번째는 이 기준에 대한 필요한 편차를 학습하는 것입니다. 또한 Point-wise Residual Normalization을 적용하여 예측 결과에서 큰 잔여량이 최적화 신호를 지배하는 것을 방지합니다.

- **Performance Highlights**: Extensive 실험 결과, ResAD는 NAVSIM 벤치마크에서 88.6의 PDMS를 달성하여 최신 기술의 성과를 보여줍니다. 이 연구는 기본적인 물리적 선험을 모델 아키텍처에 포함시킴으로써 학습 작업을 단순화하고, 더 정교하고 정확한 주행 행동을 가능하게 만드는 결과를 도출했습니다.



### MultiCOIN: Multi-Modal COntrollable Video INbetweening (https://arxiv.org/abs/2510.08561)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 ‘MultiCOIN’이라는 새로운 비디오 인비트위닝(video inbetweening) 프레임워크를 소개합니다. 이 프레임워크는 사용자가 depth transition, motion trajectories, text prompts 및 target regions와 같은 다중 모달(multi-modal) 제어 방식을 통해 비디오 생성과정을 쉽게 조정할 수 있도록 설계되었습니다. 기존 비디오 인비트위닝 모델이 사용자 의도를 완벽하게 반영하지 못했던 한계를 극복하여, 더욱 유연하고 세밀한 비디오 보간을 가능하게 합니다.

- **Technical Details**: MultiCOIN은 Diffusion Transformer (DiT) 아키텍처를 기반으로 하여, 고품질의 긴 비디오 생성을 목표로 합니다. 이 모델은 motion controls와 content controls를 각각의 두 가지 브랜치로 나누어 인코딩하며, 이를 통해 독립적이고 동시에 안정적으로 학습 할 수 있도록 설계되었습니다. 또한, 사용자 친화적인 sparse point-based representation 방식으로 제어 신호를 변환하여 제어의 편리성을 높였습니다.

- **Performance Highlights**: 실험 결과, MultiCOIN은 다양한 모달성을 지원하며, 높은 정확도로 다중 객체 제어를 가능하게 하는 성능을 보여주었습니다. 또한, depth와 motion controls의 조합을 통해 프레임 간 이동 경로의 정렬이 크게 향상되었으며, Framer와 비교할 때 더 유연한 비디오 인비트위닝 결과를 달성했습니다. 전반적으로, MultiCOIN은 사용자가 요구하는 고품질 비디오 인비트위닝을 보다 쉽게 구현할 수 있도록 지원합니다.



### SciVideoBench: Benchmarking Scientific Video Reasoning in Large Multimodal Models (https://arxiv.org/abs/2510.08559)
- **What's New**: 새로운 연구에서는 SciVideoBench라는 혁신적인 벤치마크를 도입하여 과학적 상황에서 고급 비디오 추론 능력을 평가합니다. 이 벤치마크는 25개 이상의 학술 분야에서 도출된 1,000개의 정교한 다지선다형 질문으로 구성되어 있으며, 각 질문은 도메인 전문가에 의해 확인되었습니다. 이를 통해 현재 대두되는 비디오 추론의 한계를 극복하고, LMMs의 새로운 발전 방향을 제시합니다.

- **Technical Details**: SciVideoBench는 물리학, 화학, 생물학 및 의학 등 네 가지 기초 분야에 걸쳐 제작된 241개의 연구급 실험 비디오를 기반으로 구축되었으며, 이러한 비디오는 실험 절차와 결과를 설명하는 동기화된 음성 내레이션과 함께 제공됩니다. 각 질문은 개념적, 가정적 또는 정량적 유형으로 분류되며, 모델이 정확한 시공간 기준, 도메인 지식 및 복잡한 논리적 추론을 수행할 수 있도록 요구합니다. 이 벤치마크는 비디오 이해와 관련된 복잡한 과학적 지식을 평가하기 위해 설계되었습니다.

- **Performance Highlights**: SciVideoBench에서의 평가 결과, 현재의 독점 및 오픈 소스 LMM들은 낮은 정확도를 보였으며, 과학적 추론 능력을 위한 상당한 개선의 여지가 있음을 나타냈습니다. 예를 들어, Gemini 2.5 Pro와 Qwen2.5-VL는 각각 SciVideoBench에서 낮은 성능을 기록했습니다. 이 연구는 LMM의 아키텍처, 추론 능력 및 지각 기반이 비디오 추론 성능에 미치는 중요한 역할을 강조하며, 향후 LMM 개발에 대한 명확한 방향을 제시합니다.



### VideoCanvas: Unified Video Completion from Arbitrary Spatiotemporal Patches via In-Context Conditioning (https://arxiv.org/abs/2510.08555)
Comments:
          Project page: this https URL

- **What's New**: 이 연구에서는 자율적인 시공간 비디오 완성을 위한 작업을 도입했습니다. 사용자가 지정한 패치를 임의의 위치와 타임스탬프에 놓으면, 이를 기반으로 전체 비디오를 생성하는 방식입니다. 이러한 유연한 접근 방식은 기존의 여러 비디오 생성 작업을 통합하여 단일한 패러다임 아래에서 처리할 수 있게 합니다.

- **Technical Details**: 본 연구의 핵심 프레임워크인 VideoCanvas는 In-Context Conditioning (ICC) 패러다임을 기반으로 하며, 공간과 시간 제어를 분리하는 하이브리드 제어 전략을 사용합니다. 이는 공간적 위치를 제로 패딩(Zero Padding)으로 다루고, 시간적 정렬은 Temporal RoPE Interpolation을 통해 이루어집니다. 이러한 방식은 VAE의 임시 모호함을 해결하고 픽셀-프레임 인식 제어를 가능하게 만듭니다.

- **Performance Highlights**: 실험 결과, VideoCanvas는 기존의 조건형 패러다임들보다 현저하게 우수한 성능을 보여주었습니다. VideoCanvasBench라는 새로운 벤치마크를 개발하여, 이 작업의 평가를 시스템적으로 진행할 수 있는 기준을 마련했습니다. 이에 따라 VideoCanvas는 유연하고 통합된 비디오 생성 분야에서 새로운 최첨단 성과를 제공하고 있습니다.



### Dream to Recall: Imagination-Guided Experience Retrieval for Memory-Persistent Vision-and-Language Navigation (https://arxiv.org/abs/2510.08553)
Comments:
          14 pages, 6 figures, 13 tables

- **What's New**: Memoir는 Vision-and-Language Navigation(VLN)에서 메모리 액세스의 효율성을 개선하기 위해 상상(imagination)을 활용하는 데이터 모델이다. 기존의 메모리 지향 VLN 방법들이 가지고 있는 주요한 한계점을 극복하기 위해, Memoir는 메모리 관점의 응용을 통해 내비게이션의 의사결정을 향상시키는 접근 방식을 채택한다. 특히 이 방법은 지난 경험을 효과적으로 검색하고 저장하여 진화형 내비게이션 효율성을 높인다.

- **Technical Details**: Memoir의 핵심 요소는 언어에 의해 조정된 세계 모델(world model)로, 이 모델은 내비게이션 경험을 저정하기 위해 미래 상태를 상상하고, 이 상상한 상태를 검색 쿼리로 활용한다. 또한, 하이브리드 뷰포인트 레벨 메모리(Hybrid Viewpoint-Level Memory, HVM)는 관찰과 행동 패턴을 뷰포인트에 고정시켜 다양한 내비게이션 시나리오에서의 검색 능력을 향상시킨다. 마지막으로, 경험 증강 내비게이션 모델은 수집된 정보를 바탕으로 보다 강력한 의사결정을 가능하게 한다.

- **Performance Highlights**: 10개의 다양한 메모리 지속 VLN 벤치마크에서 Memoir는 모든 시나리오에서 실질적인 성능 향상을 나타내며, IR2R에서 5.4%의 SPL 개선과 8.3배의 훈련 속도 향상, 74%의 추론 메모리 감소를 달성했다. 이러한 결과는 환경 및 행동 메모리를 예측적으로 검색하는 것이 더 효과적인 내비게이션을 가능하게 함을 검증하고 있으며, 이 상상 기반 패러다임에 대한 상당한 여지가 있음을 보여준다.



### ARTDECO: Towards Efficient and High-Fidelity On-the-Fly 3D Reconstruction with Structured Scene Representation (https://arxiv.org/abs/2510.08551)
- **What's New**: ARTDECO는 단안 이미지 시퀀스로부터의 즉각적인 3D 재구성을 위한 새로운 통합 프레임워크입니다. 이 시스템은 SLAM(Simultaneous Localization and Mapping) 기반 파이프라인의 신뢰성과 피드 포워드 모델의 효율성을 결합하여 실시간 성능과 높은 재구성 품질을 제공합니다. ARTDECO는 다중 스케일 특성을 구조화된 3D Gaussian으로 변환하는 가우시안 디코더를 활용하여 재구성의 정확성을 높이고 있습니다.

- **Technical Details**: ARTDECO는 정확한 포즈 추정 및 점 예측을 위해 3D 기반 모델을 사용하며, 수준(Layer of Detail) 인지 렌더링 전략을 통해 재구성의 효율성과 신뢰성을 유지합니다. 이 시스템의 핵심은 피드 포워드 모델을 데이터 프라이어로 사용하여 단안 데이터의 모호성을 줄이는 것입니다. 또한, ARTDECO는 가벼운 번들 조정과 루프 검출을 통합하여 글로벌 불일치를 해결하고 있습니다.

- **Performance Highlights**: 여덟 개의 다양한 실내 및 실외 벤치마크에서 실험 결과, ARTDECO는 SLAM 수준의 상호작용 성능, 피드 포워드 시스템과 유사한 강인함, 그리고 장면별 최적화에 가까운 재구성 품질을 달성하였습니다. 이는 정확한 기하학과 높은 시각적 충실도를 가진 실시간 3D 환경의 디지털화를 위한 실용적인 경로를 제공함을 의미합니다.



### VideoNorms: Benchmarking Cultural Awareness of Video Language Models (https://arxiv.org/abs/2510.08543)
Comments:
          24 pages, 5 figures, under review

- **What's New**: 이번 연구에서는 비디오 대형 언어 모델(VideoLLMs)의 문화적 이해를 평가하기 위해 VideoNorms라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 미국과 중국 문화에서 파생된 1000개 이상의 (비디오 클립, 규범) 쌍으로 구성되며, 사회문화적 규범에 기반한 주석을 포함하고 있습니다. VideoNorms는 인간-AI 협업 프레임워크를 통해 구성되어, 모델의 규범 인식 능력을 평가하는 데 기여합니다.

- **Technical Details**: VideoNorms 벤치마크는 비디오 클립에서 특정 문화적 규범이 준수되었는지를 예측하는 두 가지 분류 작업과 이를 뒷받침하는 언어적 및 비언어적 증거를 제시하는 설명 작업을 포함합니다. 이를 통해 모델이 규범 채택 및 위반을 이해하는 데 어려움을 겪고 있음을 분석하였습니다. 이 연구에서는 인간 전문가들이 주석을 검토하고 수정하는 과정이 포함되어, 데이터의 질을 높이는 데 기여합니다.

- **Performance Highlights**: 연구에서 평가된 다양한 오픈 웨이트 VideoLLMs는 규범 준수보다 위반을 인식하는 데 더 낮은 성능을 보였습니다. 또한, 중국 문화와 관련된 비디오 클립에서 성능이 떨어졌으며, 비언어적 증거를 제공하는 데 어려움을 겪었습니다. 이러한 결과는 문화적으로 기반한 비디오 언어 모델 학습의 필요성을 강조하며, VideoNorms 벤치마크가 이 문제를 해결하는 데 기여할 것임을 시사합니다.



### MM-HELIX: Boosting Multimodal Long-Chain Reflective Reasoning with Holistic Platform and Adaptive Hybrid Policy Optimization (https://arxiv.org/abs/2510.08540)
- **What's New**: 이번 연구에서는 현재의 Multimodal Large Language Models (MLLMs)가 복합적인 현실 문제 해결을 위한 long-chain reflective reasoning 능력을 크게 탐구하지 않았다는 점에 주목하고 있습니다. 연구팀은 1,260개 샘플과 42개의 도전적인 합성 문제로 구성된 MM-HELIX라는 멀티모달 벤치마크를 구축했습니다. 이를 통해 기존 MLLMs의 long-chain reflective reasoning에서 성능 저하가 발생함을 실증적으로 확인하였습니다.

- **Technical Details**: MM-HELIX 벤치마크에서는 iterative thinking과 backtracking이 필요한 복잡한 작업이 포함되어 있으며, 연구팀은 Step-Elicited Response Generation 파이프라인을 통해 100,000개의 고품질 reflective reasoning 추적 데이터를 생성하여 instruction-tuning을 위한 MM-HELIX-100K 데이터셋을 개발했습니다. 또한, Sparse reward 신호와 Supervised Fine-Tuning 후의 치명적인 망각 문제로 인해, Adaptive Hybrid Policy Optimization (AHPO)라는 새로운 훈련 전략을 제안하여 오프라인 감독과 온라인 최적화를 단일 단계로 통합하는 방식으로 학습하도록 하였습니다.

- **Performance Highlights**: Qwen2.5-VL-7B 기준 모델에 AHPO를 적용한 결과, MM-HELIX 벤치마크에서 18.6%의 정확도 향상을 달성했습니다. 또한, 일반적인 수학 및 논리 작업에서도 평균 5.7%의 성능 향상을 보여주었으며, 이는 MLLMs의 reflective reasoning이 효과적으로 학습되고 일반화될 수 있음을 시사합니다. 이러한 연구는 향후 더욱 강력한 MLLMs 개발을 위한 기초가 될 것입니다.



### Kontinuous Kontext: Continuous Strength Control for Instruction-based Image Editing (https://arxiv.org/abs/2510.08532)
Comments:
          Project Page: this https URL

- **What's New**: Kontinuous Kontext는 자연어를 통해 이미지를 직관적으로 편집할 수 있는 새로운 방법론이다. 기존의 텍스트 명령만으로는 세부적인 편집 조정이 어렵다는 한계를 극복하기 위해, 이 모델은 사용자가 편집 강도를 연속적으로 조절할 수 있는 기능을 제공한다. 이를 통해 사용자는 미세한 조정부터 강한 효과까지 매끄럽게 모든 편집을 수행할 수 있게 되었다.

- **Technical Details**: 이 모델은 기존의 최첨단 이미지 편집 모델인 Flux Kontext를 기반으로 하며, 편집 강도를 규명하는 스칼라 값을 추가 입력으로 받아들인다. 경량의 프로젝터 네트워크를 사용하여 입력된 스칼라 값과 편집 지침을 모델의 조정 공간에서 조정계수로 매핑한다. 데이터 훈련은 기존의 생성 모델을 사용하여 다양한 이미지 편집 강도를 가진 쿼드루플릿을 합성하여 이루어진다.

- **Performance Highlights**: Kontinuous Kontext는 세부적인 편집 조정이 가능해 사용자가 특정 속성, 재료, 디자인 변경 및 환경 조정 등의 다양한 편집 작업에서 정밀한 제어를 할 수 있도록 한다. 실험 결과, 이 모델은 훈련되지 않은 얼굴 속성 변경이나 신체 형태 변경과 같은 새로운 편집 범주에서도 일반화되어 그 성능을 입증하였다. 이로 인해 중요한 비주얼 편집 작업에도 효과적으로 사용할 수 있는 방법론이 되었다.



### SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models (https://arxiv.org/abs/2510.08531)
Comments:
          Project Page: this https URL Code: this https URL

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)에서 공간 추론(Spatial reasoning)의 한계를 해결하기 위해 SpatialLadder-26k라는 다중 모달 데이터셋과 순차적 훈련 방법론을 제안합니다. 기존 방법들이 인지와 추론의 계층적 구조를 놓치고 있어 성능이 저조하다는 점을 지적하며, 공간 인지를 단계적으로 구축해야 한다고 주장합니다. 이를 통해 공간 지능을 강화하는 세 단계의 훈련 프레임워크를 선보이며, 단계적 접근 방식으로 공간 추론 능력을 향상시킬 수 있음을 강조하고 있습니다.

- **Technical Details**: 제안된 SpatialLadder-26k 데이터셋은 26,610개의 샘플로 구성되어 있으며, 객체 위치 확인(Object localization), 단일 이미지(Single image), 다중 시점(Multi-view), 비디오(Video) 공간 추론 작업을 포함합니다. 이 데이터셋은 각 모달리티 전반에 걸쳐 체계적인 커버리지를 보장하는 표준화된 파이프라인을 통해 구축되었습니다. 훈련 프레임워크는 세 단계로 나뉘며, 첫 번째 단계에서는 객체 위치 확인을 통해 공간 인지를 확립하고, 두 번째 단계에서는 여러 차원 공간 작업을 통해 이해를 발전시키며, 세 번째 단계에서는 강화 학습(Reinforcement learning)을 통해 복잡한 추론을 강화합니다.

- **Performance Highlights**: SpatialLadder 모델은 3B 파라미터를 가지며, 공간 추론 벤치마크에서 최첨단 성능을 기록하였습니다. 연구 결과, SpatialLadder는 기존 모델보다 23.4% 향상된 성능을 보이며, GPT-4o를 20.8% 및 Gemini-2.0-Flash를 10.1% 초과 달성했습니다. 또한, 도메인 외 베치마크에서도 7.2% 향상된 일반화 성능을 유지함으로써, 인지에서 추론으로의 단계적 훈련이 강력한 공간 지능을 위해 필수적임을 입증하였습니다.



### FlexTraj: Image-to-Video Generation with Flexible Point Trajectory Contro (https://arxiv.org/abs/2510.08527)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 FlexTraj라는 이미지-비디오 생성 프레임워크를 소개합니다. FlexTraj는 각 포인트를 세그멘테이션 ID, 시간적 일관성을 갖춘 트레젝토리 ID, 선택적인 컬러 채널로 인코딩하여 밀집 및 희박한 트레젝토리 제어를 가능하게 합니다. 이 시스템은 동영상 생성기와 토큰 연결 방식이 아닌 효율적인 시퀀스 연결 방식을 도입하여 빠른 수렴과 강력한 제어 가능성을 제공합니다.

- **Technical Details**: FlexTraj는 각 포인트가 세그멘테이션 ID, 트레젝토리 ID, 선택적으로 컬러 속성을 갖는 연속적인 3D 포인트 시퀀스로 모션을 표현합니다. 이 포인트들은 픽셀 공간으로 투사되어 ID 코드 비디오와 컬러 큐 비디오를 형성하며, 이는 사전 훈련된 비디오 VAE에 의해 처리되어 조건 토큰을 생성합니다. FlexTraj는 밀도와 정렬에 대한 상계 훈련 전략을 채택하여 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: FlexTraj는 높은 품질의 일관된 비디오 생성을 가능하게 하며, 다양한 사용자 요구에 맞춰 유연한 제어를 제공합니다. 일반 사용자에게는 비디오 모션 전송 및 카메라 리디렉션을 지원하고, 전문 CG 사용자에게는 부분적으로 장착된 메시에서 전체 장면을 애니메이션화하는 등의 기능을 제공합니다. 실험 결과, FlexTraj는 다양한 시나리오에서 유연한 조정을 제공하며, 다중 세분성을 지원하는 최초의 프레임워크로 자리잡았습니다.



### SliceFine: The Universal Winning-Slice Hypothesis for Pretrained Networks (https://arxiv.org/abs/2510.08513)
- **What's New**: 이 논문은 사전 훈련된 모델 내에서 무작위로 선택된 서브 네트워크(슬라이스)를 미세 조정하는 것이 하위 작업(adapt) 적응에 충분할 수 있는 이론적 프레임워크를 제시합니다. 우리는 사전 훈련된 네트워크가 전역 승리 슬라이스 속성을 나타낸다는 것을 증명하며, 이는 두 가지 현상에서 비롯됩니다: (1) spectral balance(스펙트럴 밸런스)와 (2) high task energy(높은 작업 에너지). 이러한 발견은 매개변수 효율적인 미세 조정(parameter efficient fine tuning)인 SliceFine 방법을 통해 실증적으로 적용됩니다.

- **Technical Details**: 본 연구는 Universal Winning Slice Hypothesis(전략적 승리 슬라이스 가설)을 통해, 넓은 슬라이스만으로도 미세 조정이 가능하다고 제안합니다. 이 가설은 사전 훈련된 네트워크에서 임의의 충분히 넓은 슬라이스가 로컬 승리 티켓(local winning ticket)이 될 수 있으며, 여러 층에 걸쳐 슬라이스 세트를 조정하면 글로벌 승리 티켓(global winning ticket)을 형성할 수 있음을 명확히 합니다. SliceFine은 이러한 슬라이스를 훈련하며, 기존 매개변수를 추가하지 않고도 성능을 극대화합니다.

- **Performance Highlights**: SliceFine은 언어 및 비전 과제에서 최첨단 PEFT 방법들과 일치하는 성능을 보이며, 훈련 속도와 메모리 효율성, 모델 compactness(소형화)를 크게 개선합니다. 실험적으로, SliceFine은 적은 수의 매개변수로도 유사한 성능을 제공하며, 메모리 사용량 및 교육 시간을 절약합니다. 본 연구는 이론과 실습을 연결하며, PEFT 기술을 다룬 기존 방법에 대한 이론적으로 근거 있는 대안을 제공합니다.



### Have We Scene It All? Scene Graph-Aware Deep Point Cloud Compression (https://arxiv.org/abs/2510.08512)
Comments:
          Accepted for publication in IEEE Robotics and Automation Letters (RA-L). 8 pages, 6 figures

- **What's New**: 이 논문에서는 포인트 클라우드 데이터의 효율적인 전송을 위한 심층 압축(framework) 프레임워크를 제안합니다. 제안된 방법은 포인트 클라우드를 심리적(scene graph) 의미론적 조각으로 분해한 후, 주어진 링크 구조에 따라 압축하여 전송하는 데 최적화되어 있습니다. 이 프레임워크는 Feature-wise Linear Modulation (FiLM) 조건부 변환기를 사용하여 용량을 98%까지 줄이면서 구조적 및 의미적 충실도를 유지합니다.

- **Technical Details**: 이 연구는 데이터 압축을 위해 의미(scene) 기반의 오토인코더(autoencoder)를 거래하여 주어진 라이다(LiDAR) 스캔을 의미적으로 일관된 조각으로 분해합니다. 각 조각은 FiLM으로 조정된 인코더에 의해 독립적으로 인코딩된 후, 구조적으로 정확한 복원(reconstruction)을 위해 폴딩(folding) 기반 디코더에 의해 복원됩니다. 그렇게 함으로써 우리는 통신 부하를 크게 줄일 수 있습니다.

- **Performance Highlights**: 적용된 SemanticKITTI 및 nuScenes 데이터셋에 대한 실험 결과, 제안 방식은 데이터 크기를 최대 98% 줄이면서 높은 성능을 입증했습니다. 또한, 이 방법은 원시 LiDAR 스캔을 사용했을 때와 비슷한 궤적 정확도 및 지도 정렬을 달성하여, 다중 로봇 포즈 그래프 최적화(multi-agent pose graph optimization) 및 지도 병합(map merging) 같은 다운스트림 작업에서도 효과적으로 작동합니다.



### To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language Models (https://arxiv.org/abs/2510.08510)
Comments:
          Preprint. Project page: this https URL

- **What's New**: 최근 대형 비전 언어 모델(LVLMs)은 시각적 정보와 텍스트 정보를 이해하고 추론할 수 있는 강력한 아키텍처로 자리 잡았습니다. 이 모델은 비전 트랜스포머(Vison Transformer, ViT)와 대형 언어 모델(Large Language Model, LLM)이라는 두 가지 핵심 구성 요소에 의존합니다. 본 연구에서는 ViT에서 시각적 의미를 포착하는 중요한 고규범 토큰인 ViT attention sinks를 탐구하였으며, 이들이 LVLM의 이해와 추론에서 어떻게 기여하는지에 대한 자세한 분석을 제시합니다.

- **Technical Details**: LVLM은 주로 세 가지 주요 구성 요소로 구성됩니다: 비전 인코더(Visual Encoder), 연결 모듈(Connector Module), 언어 모델(Language Model). 비전 인코더는 입력 이미지에서 시각적 특징을 추출하여 비밀 상태(hidden state)를 생성하며, 연결 모듈은 이러한 시각적 특징을 LLM의 텍스트 공간으로 매핑합니다. 마지막으로 언어 모델은 다양한 종류의 토큰을 입력받아 출력을 생성하며, 여기서 다중 헤드 어텐션(Multi-Head Attention, MHA) 메커니즘이 중요한 역할을 합니다.

- **Performance Highlights**: 연구에서는 ViT attention sinks를 효과적으로 활용함으로써 LVLM 성능을 향상시키는 여러 가지 방법을 제안했습니다. 특히, DIYSink라는 훈련 기반 프레임워크를 도입하여, 이 시각적 토큰들이 언어 모델에서 어떻게 활용될 수 있는지를 실험적으로 검증하였습니다. 다양한 LVLM과 비전 모델 조합에 대해 일관되게 성능 향상이 나타났으며, 이는 ViT attention sinks의 잠재력을 재발견하는 데 기여하였습니다.



### MoA-VR: A Mixture-of-Agents System Towards All-in-One Video Restoration (https://arxiv.org/abs/2510.08508)
- **What's New**: 이 논문에서는 MoA-VR이라는 새로운 비디오 복원 시스템을 제안합니다. MoA-VR은 전문가의 추론 및 처리 방식에서 영감을 받은 세 개의 협력 에이전트로 구성됩니다. 이 시스템은 혼합된 다양한 손상 유형을 효과적으로 복원할 수 있도록 설계되었습니다.

- **Technical Details**: MoA-VR은 손상 식별, 라우팅 및 복원, 복원 품질 평가의 세 가지 핵심 기능을 통합합니다. 비전-언어 모델(VLM)을 활용한 대규모 데이터셋 기반으로 손상 인식을 수행하며, 대형 언어 모델(LLM)을 이용해 자가 적응형 라우터를 구현합니다. 이를 통해 복원 프로세스를 최적화하고, 각 비디오의 독특한 손상 특성에 맞춰 동적으로 대응할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면 MoA-VR은 최신 비디오 복원 방법들보다 훨씬 우수한 성능을 보이며, PSNR에서 3.02 dB 개선을 달성했습니다. 또한, 지각적 품질 및 픽셀 메트릭 모두에서 유의미한 향상을 보여줍니다. MoA-VR은 최소한의 인간 개입으로 영상 복원을 수행할 수 있는 능력을 입증하였습니다.



### InstructX: Towards Unified Visual Editing with MLLM Guidanc (https://arxiv.org/abs/2510.08485)
- **What's New**: 이번 논문은 InstructX라는 통합 프레임워크를 제안하여, 이미지 및 비디오 편집을 위한 멀티모달 대형 언어 모델(MLLM)과 확산 모델을 통합한 comprehensive 한 연구를 다룹니다. MLLM의 이해 능력을 최대한 활용하기 위해, 기존의 다양한 편집 설계 선택과 통합 방식에 대한 심도 있는 분석이 이루어집니다. 특히, 이미지 데이터에 대한 훈련이 비디오 편집 능력을 emergent하게 유도할 수 있다는 것을 보여줍니다.

- **Technical Details**: 이 논문에서는 외부 확산 모델을 프레임워크로 채택하여, 빠른 수렴과 최소한의 변경으로 경쟁력 있는 성능을 보입니다. MLLM의 모달리티 전용 특징을 통합하여 하나의 모델 내에서 이미지와 비디오 편집 작업을 효과적으로 통합합니다. 이를 통해 다양한 편집 작업을 처리할 수 있으며, state-of-the-art 성능을 달성합니다.

- **Performance Highlights**: Extensive 실험 결과, 제안된 방법은 기존의 오픈 소스 및 폐쇄 소스 방법보다 더 넓은 범위의 작업을 처리할 수 있는 zero-shot 비디오 편집 능력을 확대합니다. 이 연구는 MLLM과 확산 모델의 통합에 대한 깊은 통찰을 제공하며, 향후 연구에 필요한 지침을 마련합니다. 또한, 이미지와 비디오 데이터 모두에서 편집 능력을 효과적으로 학습할 수 있음을 보여줍니다.



### The Visual Iconicity Challenge: Evaluating Vision-Language Models on Sign Language Form-Meaning Mapping (https://arxiv.org/abs/2510.08482)
- **What's New**: 본 논문에서는 시각 언어 모델(VLMs)이 동적인 인간의 행동에서 언어적 형태와 의미 간의 관계를 회복하는 도전 과제를 다루기 위해 Visual Iconicity Challenge라는 새로운 비디오 기반 벤치마크를 제안합니다. 이 챌린지는 NGT(네덜란드 수화)의 신호 데이터셋으로, 단어의 형태를 예측하고, 형태에서 의미를 유추하며, 점진적인 아이코닉성 평가를 포함한 세 가지 작업을 수행합니다. 연구에 따르면, VLMs는 손 모양과 위치에 대한 예측에서 일부 세부 사항을 회복하지만, 인간의 성능에는 여전히 미치지 못합니다.

- **Technical Details**: 본 연구는 VLMs의 특징을 평가하기 위해 수동적으로 주석이 달린 NGT 신호 데이터셋을 사용합니다. 데이터셋은 명확한 시각적 링크가 있는 아이코닉한 신호와 그렇지 않은 임의의 신호를 구별합니다. 연구는 각각의 모형이 손 모양, 위치 및 움직임 특징을 인식할 수 있는지, 시각적 형태만을 기반으로 신호의 의미를 유추할 수 있는지, 그리고 모델이 인간의 아이코닉성 판단에 얼마나 근접하는지를 평가합니다. 이러한 작업은 아이코닉성이 시각적 형태와 의미를 연결하는 데 필수적이라는 점에서 중요합니다.

- **Performance Highlights**: VLMs는 아이코닉한 신호의 손 모양과 위치를 예측하는 데 어느 정도 성공했지만, 투명성과 아이코닉성 등에서 인간 성능에 비해 현저히 낮습니다. 특히, 아이코닉 형태 예측을 더 잘 수행하는 모델이 인간의 아이코닉성 판단과 더 잘 상관관계를 가지는 것으로 나타났습니다. 이러한 결과는 모델들이 시각적으로 기반한 구조에 대한 민감성을 공유하고 있음을 시사합니다. 최종적으로, 이 연구는 아이코닉성의 모델링 및 다중 모달 모델에서 시각적 기초를 향상시킬 수 있는 인간 중심의 신호 및 체화 학습 방법을 촉진하는 계기를 제공합니다.



### Video-STAR: Reinforcing Open-Vocabulary Action Recognition with Tools (https://arxiv.org/abs/2510.08480)
- **What's New**: 새로운 프레임워크인 Video-STAR는 맥락적 서브 모션 분해(contextual sub-motion decomposition)와 도구 강화 학습(tool-augmented reinforcement learning)을 결합하여 개방형 어휘(action recognition in open vocabulary) 행동 인식을 지원합니다. 기존 방법들이 행동을 단일한 엔티티로 취급했던 반면, Video-STAR는 행동을 세분화된 서브 모션으로 분해하여 정밀한 일치를 가능하게 합니다. 또한, 도메인 특화 도구를 동적으로 사용할 수 있도록 설계하여 정확한 시각적 추론을 통해 의미적 모호성을 줄입니다.

- **Technical Details**: Video-STAR는 1) 행동을 서브 모션 원시로 분해하고, 2) 후보 행동과 서브 모션을 매칭한 후, 3) 계층적 관련성에 따라 점수를 매기는 일련의 결정 과정으로 행동 인식을 모델링합니다. 이를 통해 도구 활용의 효율성과 서브 모션의 관련성, 그리고 추론의 구조적 일관성을 균형 있게 최적화하는 보상 메커니즘을 설계하였습니다. 또한, 도구 박스는 포즈 추정(pose estimation), 인간 탐지(human detection) 및 실시간 검색(online retrieval) 기능을 통해 강화됩니다.

- **Performance Highlights**: HMDB-51, UCF-101, SSv2, Kinetics-400, Kinetics-600 데이터셋을 통해 실시한 평가에서 Video-STAR는 기존 방법들을 초격차로 초과 달성함을 보였습니다. 특히, 세밀한 행동 구분과 크로스 모달 환각(cross-modal hallucination) 처리에서 높은 성능을 발휘하여 우수한 강건성과 일반화 능력을 입증하였습니다. 이와 같은 결과들은 도구를 통한 동적 추론과 계층적 서브 모션 활용이 효과적임을 시사합니다.



### Hierarchical Spatial Algorithms for High-Resolution Image Quantization and Feature Extraction (https://arxiv.org/abs/2510.08449)
Comments:
          There are 14 pages journal paper

- **What's New**: 이번 연구에서 제안하는 모듈형 프레임워크는 공간 이미지 처리의 여러 과정을 통합하여 회색조 양자화(grayscale quantization), 색상 및 밝기 향상(color and brightness enhancement), 이미지 샤프닝(image sharpening), 양방향 변환 파이프라인(bidirectional transformation pipelines), 기하학적 특징 추출(geometric feature extraction)을 포함합니다. 단계별 강도 변환으로 회색조 이미지를 8개의 이산 수준으로 양자화하여 구조적 세부 사항을 유지하면서 표현을 단순화하는 효과를 누립니다. 또한, RGB 및 YCrCb 색 공간에서 히스토그램 평활화(histogram equalization)를 통해 색상 향상을 이루며, YCrCb 공간을 활용하면 대비를 개선하면서도 색조 충실성을 유지할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 3x3 컨볼루션 커널을 사용하여 고주파 세부사항을 증강할 수 있는 이미지 샤프닝 기법을 포함하고 있습니다. 양방향 변환 파이프라인에서는 언샤프 마스킹(unsharp masking), 감마 보정(gamma correction), 잡음 증폭(noise amplification)을 통합하여 76.10% 및 74.80%의 정확도를 달성하였습니다. 기하학적 특징 추출은 Canny 엣지 감지(canny edge detection), 허프 변환(Hough transform), 해리스 코너 감지(Harris corner detection) 등 다양한 기법을 활용하며, 최종적으로 큐 고립(cue isolation) 작업에서 81.87% 유사성을 보였습니다.

- **Performance Highlights**: 다양한 데이터 세트에 대한 실험 평가를 통해 제안된 프레임워크가 강력하고 결정론적인 성능을 발휘함을 입증하였습니다. 이 연구의 주요 기여는 실시간 이미지 분석(real-time image analysis)과 컴퓨터 비전(computer vision) 분야에 직접 적용할 수 있는 계산적으로 효율적인 기초를 마련한 것입니다. 전체적으로 이 프레임워크는 저자극 자연 장면에서도 예측 가능한 결과를 보장하여 안전을 중시하거나 자원이 제한된 환경에서 활용 가능합니다.



### Gaze on the Prize: Shaping Visual Attention with Return-Guided Contrastive Learning (https://arxiv.org/abs/2510.08442)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구는 시각적 강화학습(Visual Reinforcement Learning, RL) 에이전트가 고차원 이미지 데이터에서 작업과 관련된 소수의 픽셀에 따라 행동할 수 있도록 돕기 위해 'Gaze on the Prize'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 에이전트의 경험을 통해 파생된 자기 지도 신호(self-supervised signal)를 활용하여 배우는 것이며, 이는 높은 수익을 추구하는 데 초점을 맞춥니다. Gaze on the Prize는 에이전트가 성공과 실패를 구분할 수 있도록 돕고, 태스크 관련 특징에 집중하게 합니다.

- **Technical Details**: 이 연구의 핵심 아이디어는 결과 차이(return differences)가 무엇이 중요한지를 드러낸다는 것입니다. 같은 상태에서도 서로 다른 결과를 가져오는 경우, 이들을 구별하는 특징은 작업과 관련이 있을 가능성이 높습니다. Gaze는 태스크 관련 특징을 식별하기 위해 대조 신호(contrastive signal)를 사용하여 훈련된 시각적 주의 메커니즘이며, 이는 기존의 RL 알고리즘과 호환 가능합니다. 긴 열 순차 연결(contrastive triplets)을 통해 에이전트의 주의 메커니즘이 성공과 실패를 구별하는 데 필요한 패턴을 학습하게 됩니다.

- **Performance Highlights**: 본 연구 방법론은 샘플 효율성이 최대 2.4배 개선되었으며, 기존 알고리즘이 학습에 실패했던 작업을 해결할 수 있음이 입증되었습니다. 이는 ManiSkill3 벤치마크의 다양한 조작 작업에서 수행되었으며, 강화학습 알고리즘이나 하이퍼파라미터를 변경하지 않고도 이루어졌습니다. Gaze on the Prize는 기존의 비주얼 RL 알고리즘에 플러그인할 수 있는 접근법으로, 성능을 향상시키면서도 기본 구조를 유지합니다.



### Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency (https://arxiv.org/abs/2510.08431)
- **What's New**: 이 연구는 일반 응용 수준의 이미지 및 비디오 확산 모델에 대한 지속적인 시간 일관성 증류(continuous-time consistency distillation)를 대규모로 확장하려는 첫 번째 시도입니다. 지속적인 시간 일관성 모델(sCM)은 이론적으로 원리적이며, 학술 규모의 확산을 가속화하는 데 강력하지만, 실제 대규모 텍스트-이미지(text-to-image) 및 텍스트-비디오(text-to-video) 작업에 대한 적용 가능성은 불확실합니다. 본 연구는 새로운 FlashAttention-2 JVP 커널을 개발하여 10억 개 이상의 매개변수를 가진 모델에서의 sCM 훈련을 가능하게 합니다.

- **Technical Details**: 연구진은 점진적 정제(progressive distillation) 및 스코어 정제(score distillation) 기법 대신, 스코어 정제된 지속적인 시간 일관성 모델(rCM)을 제안합니다. 이는 장기 스킵 규제자(long-skip regularizer)로서 스코어 증류(score distillation)를 통합하여 생성의 질을 향상시키고 고차원의 비디오 작업에서도 효과적으로 작동합니다. rCM은 GAN 조정이나 광범위한 하이퍼파라미터 검색 없이도 14억 개 매개변수를 가진 거대 모델에서 최적화된 성능을 발휘합니다.

- **Performance Highlights**: rCM은 DMD2와 같은 최신 정제 방법과 비교하여 질적 지표에서 동등하거나 뛰어난 성능을 보이며, 생성의 다양성 측면에서도 상당한 이점을 제공합니다. 효율적으로 고충실도 샘플을 1~4스텝 내에 생성할 수 있어 확산 샘플링 속도를 15배에서 50배까지 가속화합니다. 이러한 결과는 rCM이 대규모 확산 증류를 발전시킬 수 있는 실용적이고 이론적으로 당위성이 있는 프레임워크로 자리 잡게 합니다.



### VideoVerse: How Far is Your T2V Generator from a World Model? (https://arxiv.org/abs/2510.08398)
Comments:
          24 Pages, 8 Figures, 11 Tables

- **What's New**: 최근 Text-to-Video (T2V) 생성 기술의 신속한 발전으로 인해 기존 벤치마크가 최신 T2V 모델을 평가하는 데 더욱 불충분하게 되었습니다. 이 논문에서는 세계 모델(world models)을 구축하는 데 필수적인 이벤트 수준의 시간적 인과성(temporal causality)과 세계 지식(world knowledge)을 체계적으로 평가할 수 있는 새로운 벤치마크인 VideoVerse를 소개합니다. VideoVerse는 다양한 분야의 대표적인 비디오를 수집하고, 이들의 이벤트 수준 설명을 텍스트-비디오 프롬프트로 변환하여 평가합니다.

- **Technical Details**: VideoVerse는 동적(dynamics) 및 정적(static) 속성의 관점에서 10개의 평가 차원으로 구성된 프롬프트를 활용하여 T2V 모델의 성능을 평가합니다. 동적 관점에서는 시간적 인과관계를 포함한 이벤트 추적(Event Following) 프롬프트가 설계되었으며, 정적 관점에서는 자연의 제약(Natural Constraints) 및 상식(Common Sense)을 고려합니다. 각 프롬프트에 대해 793개의 이진 평가 질문이 포함되어 있으며, 총 300개의 고품질 프롬프트로 구성되어 있습니다.

- **Performance Highlights**: VideoVerse를 통해 수행된 현대 T2V 모델에 대한 체계적인 평가는, 기존의 전통적인 벤치마크에서는 비슷한 성과를 보였던 Open-source와 Closed-source 모델들이 VideoVerse에서는 인과적 추론과 세계 지식이 필요한 차원에서 큰 성능 차이를 보임을 보여줍니다. 현재 T2V 모델들은 여전히 세계 모델에 미치지 못함을 입증하며, 이 빠르게 진화하는 분야에서의 새로운 도전과 연구 방향을 제시합니다.



### Robust Source-Free Domain Adaptation for Medical Image Segmentation based on Curriculum Learning (https://arxiv.org/abs/2510.08393)
- **What's New**: 최근 연구에서는 소스 데이터 없이 대상 도메인에 모델을 적응시키는 새로운 연구 분야인 source-free domain adaptation이 발견되었습니다. 이 설정은 의료 이미지의 데이터 프라이버시와 보안 문제를 해결할 수 있는 가능성을 보여줍니다. 하지만 현재의 대부분의 source-free 도메인 적응 프레임워크는 학습 절차를 고려하지 않고 대상 데이터의 pseudo label 개선에 주로 집중하고 있습니다. 본 논문에서는 이러한 문제를 해결하기 위해, 쉬운 샘플에서 어려운 샘플로 점진적으로 학습하는 커리큘럼 기반 프레임워크인 Learning from Curriculum (LFC)를 제안합니다.

- **Technical Details**: 제안된 LFC 프레임워크는 두 가지 커리큘럼, 즉 easy-to-hard과 source-to-target 커리큘럼으로 구성되어 있습니다. '쉬운' 샘플에서 시작하여 점진적으로 '어려운' 샘플로 전환하는 프로세스를 통해 모델 적응을 진행합니다. 또한, 각 배치 내 데이터의 가중치를 동적으로 조정하여 최적화 방향을 튜닝하며, 이는 모델이 점진적으로 다른 모델에서 생성된 pseudo label을 활용할 수 있게 합니다.

- **Performance Highlights**: 공식적인 cross-domain 데이터셋을 사용한 실험 결과, 제안된 LFC 프레임워크는 기존의 방법들을 능가하면서 새롭게 state-of-the-art를 달성했습니다. 이 연구는 source-free domain adaptation에서의 모델 일반화 성능을 높이는 데 중요한 기여를 할 것으로 기대됩니다. 또한, 제안된 접근 방법은 기존의 프레임워크들이 간과했던 학습 절차를 체계적으로 고려함으로써 실질적인 효과를 보였습니다.



### Detecting Legend Items on Historical Maps Using GPT-4o with In-Context Learning (https://arxiv.org/abs/2510.08385)
- **What's New**: 이 연구는 역사적 지도에서 범례(legend)를 자동으로 추출하기 위한 새로운 방법을 발표합니다. LayoutLMv3를 사용한 레이아웃 감지와 GPT-4o를 활용한 인컨텍스트 학습(in-context learning) 기법을 결합하여 범례 항목과 그 설명을 연결합니다. 실험 결과, 구조화된 JSON 프롬프트를 사용한 GPT-4o가 기존 방법보다 뛰어난 성능을 보여주며, 역사적 지도의 색다른 시각적 스타일에 대한 인덱싱과 검색 가능성을 높이고 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다: 첫째, 전방위 맵에서 범례 영역을 분리하는 것입니다. 이를 위해 LayoutLMv3를 사용하여 범례가 포함된 블록을 식별하고, 그 후 GPT-4o를 통해 범례 항목과 설명에 대한 경계 상자(bounding box)를 생성합니다. 사용되는 JSON 프롬프트는 작업 정의와 함께 예시 항목과 설명의 쌍을 포함하여 GPT-4o에 제공됩니다.

- **Performance Highlights**: 제안된 방식의 성능은 88%의 F-1 점수와 85%의 IoU를 기록했습니다. GPT-4o는 다양한 예시 수(5, 10, 15, 20)에 따라 성능이 향상됨을 보여주었습니다. 이 연구는 역사적 지도 아카이브에서 범례 해석의 필요성을 강조하며, 스케일러블한 지리공간 검색과 마이닝을 위한 기초를 제공합니다.



### UniVideo: Unified Understanding, Generation, and Editing for Videos (https://arxiv.org/abs/2510.08377)
Comments:
          Project Website this https URL

- **What's New**: UniVideo는 기존의 통합 모델의 한계를 넘어 비디오 도메인까지 확장된 모델입니다. 이 연구는 Multimodal Large Language Model (MLLM)과 Multimodal DiT (MMDiT)를 결합한 이중 스트림 설계를 도입하여 비디오 생성 및 편집 작업을 통합합니다. UniVideo는 복잡한 멀티모달 지침을 정확하게 해석하면서도 시각적 일관성을 유지하는 능력을 보여주고 있습니다.

- **Technical Details**: UniVideo의 구조는 MLLM과 MMDiT라는 두 주요 구성 요소로 이루어져 있습니다. MLLM은 텍스트와 비디오 입력을 이해하고, MMDiT는 시각적 생성 작업을 처리합니다. 이 시스템은 다른 비디오 작업들 간의 일관성을 유지할 수 있도록 설계되어 있으며, 다양한 멀티모달 입력을 처리하는 데 강점을 보입니다.

- **Performance Highlights**: UniVideo는 텍스트-이미지 및 이미지-비디오 생성을 포함한 다양한 벤치마크에서 이전의 최첨단 방법들과 비교해 우수한 성능을 보여주었습니다. 특히, 새로운 작업 조합에 대한 일반화 능력을 발휘하며, 사용자 정의 지침 없이도 다양한 비디오 생성 및 편집 작업을 수행할 수 있다는 점에서 주목받고 있습니다.



### Hyperspectral data augmentation with transformer-based diffusion models (https://arxiv.org/abs/2510.08363)
Comments:
          10 pages, 2 figures, accepted at SPIE REMOTE SENSING conference 16-20 September 2024 Edinburgh, United Kingdom

- **What's New**: 본 논문에서는 제한된 레이블 데이터셋을 사용하여 딥러닝 모델을 효과적으로 훈련하기 위한 데이터 증강(data augmentation) 기법을 제안합니다. 이 기법은 guided diffusion model을 활용하여 복잡한 데이터 패턴을 캡처하며, 경량화된 transformer 네트워크를 구현하여 처리합니다. 또한, 수정된 가중 손실 함수(weighted loss function)와 최적화된 코사인 분산 스케줄러(cosine variance scheduler)를 도입하여 소규모 데이터셋에서 신속하고 효율적인 훈련을 제공합니다.

- **Technical Details**: 제안된 방법은 PRISMA 위성에서 획득한 하이퍼스펙트럼 이미지를 사용하여 10가지 서로 다른 숲 유형의 분류 작업에 적용되었습니다. 이를 통해 기존의 데이터 증강 기법보다 평균 및 가중 평균 정확도에서 더 나은 성능을 나타냈습니다. 방법은 데이터의 고차원 특성으로 인해 발생할 수 있는 과적합(overfitting) 문제를 해결하는 데 도움을 주고, 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 이 연구 결과는 제안된 방법이 하이퍼스펙트럼 이미지 분류 작업에서 다른 데이터 증강 기법들을 초월함을 강조합니다. 또한, 모델의 안정적인 훈련 행동은 데이터 증강에 대한 딥러닝 생성 모델의 일반적인 제약을 해결합니다. 이러한 요소들 덕분에 제안된 모델은 제한된 레이블 샘플을 사용하더라도 높은 품질의 스펙트럼 서명을 생성할 수 있습니다.



### SPICE: Simple and Practical Image Clarification and Enhancemen (https://arxiv.org/abs/2510.08358)
Comments:
          5 pages, 8 figures

- **What's New**: 이 논문에서는 이미지를 향상시키고 명확히 하기 위한 간단하면서도 효율적인 방법을 소개합니다. 특히 저조도 이미지 향상 및 안개가 낀 이미지의 정화에 중점을 두고 있으며, 이는 일반적으로 의료 이미징, 자율주행차 및 보안 감시에 활용됩니다. 제안된 방법은 MATLAB 코드로 쉽게 구현할 수 있어 간편함이 큰 장점입니다.

- **Technical Details**: 이 방법은 저조도 조건에서 이미지를 향상시키기 위해 리버스 필터링(Reverse Filtering) 기법을 사용합니다. 이미지 색상 영역(Hue-Saturation-Value, HSV) 변환을 통해 값 성분을 조정하고, 이후 원래 RGB 색상으로 다시 변환합니다. 저조도 이미지에서는 필터를 적용해 작은 세부 정보를 억제하고, 그 과정을 통해 이미지의 밝기를 조정하는 방식을 채택합니다.

- **Performance Highlights**: 실험 결과 이 방식은 극도로 어두운 이미지나 안개가 낀 이미지에서 최신 기법보다 뛰어난 성능을 보이며, 전문적인 응용에서도 경쟁력을 갖추고 있음을 입증했습니다. 적절한 감마 보정 값과 필터 조합을 통해 잘 보이는 이미지를 생성하며, 안개 제거 작업에서도 양호한 결과를 산출합니다.



### Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception (https://arxiv.org/abs/2510.08352)
- **What's New**: 본 논문에서는 Distance-Annotated Traffic Perception Question Answering (DTPQA)라는 새로운 벤치마크를 소개합니다. DTPQA는 교통 장면 내 시각 기반 질문에 초점을 맞춘 첫 번째 Visual Question Answering (VQA) 벤치마크로, 거리 주석이 포함되어 있어 연구자들이 모델의 인식 능력을 평가할 수 있습니다. 또한, 이 연구는 소형 Vision-Language Models (VLMs)의 인식 성능을 평가했으며, 단순한 질문에 대한 성능이 인간보다 약 60%라는 점을 강조하고 있습니다.

- **Technical Details**: VLMs는 일반적으로 강력한 Large Language Models (LLMs)를 기반으로 하여 시각과 언어를 통합하는 능력이 뛰어납니다. 최근 연구들은 주로 SOTA(Small State-of-the-Art) 대형 모델을 평가했으나, DTPQA 벤치마크는 소형 VLM들을 대상으로 하여 그들의 인식 능력을 깊이 있게 분석합니다. 각 DTPQA 샘플은 이미지, 질문, 정답, 거리 정보를 포함하고 있어 모델의 성능을 정량적으로 평가하는 데 도움을 줍니다.

- **Performance Highlights**: 존재하는 다양한 VQA 벤치마크와 달리 DTPQA는 단순한 인식만을 요구하는 질문들로 구성되어 있습니다. 연구에 따르면, 소형 VLM 모델들은 문제의 거리 증가에 따라 성능이 저하되는 경향이 있으며, 특정 인식 과제는 여전히 도전적입니다. 이 연구는 소형 VLMs의 인식 성능이 전반적으로 불충분하다는 점을 밝혀내며, 향후 안전-critical한 응용 분야에서의 신뢰성을 확보하기 위한 기준을 제시합니다.



### LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation (https://arxiv.org/abs/2510.08318)
Comments:
          Code will be released upon acceptance

- **What's New**: 이 논문에서는 LinVideo라는 효율적인 데이터 없음 후 교육 프레임워크를 제안합니다. 이는 비디오 생성에서 자가 주의(self-attention) 모듈을 선형 주의(linear attention)로 교체하여 원래 모델의 성능을 유지하면서 계산 비용을 줄이는 데 중점을 두고 있습니다. 또한, 여러 층의 교체 가능성을 평가하고, 성능 저하를 최소화하기 위해 선택적 전이(selective transfer) 방식을 도입하여 여러 레이어를 자동으로 전환합니다.

- **Technical Details**: LinVideo는 사전 훈련된 비디오 확산 모델에서 자가 주의 레이어의 일부를 선형 주의로 교체하는 프레임워크입니다. 논문에서는 자가 주의 레이어를 이진 분류 문제로 프레이밍하고, 각 레이어의 교체 여부를 학습 가능한 스칼라로 점수를 매겨 결정합니다. 또한, 샘플 분포를 효율적으로 정렬하기 위한 언제든지 분포 정합(anytime distribution matching, ADM) 목표를 도입하여 최적화를 수행합니다.

- **Performance Highlights**: 광범위한 실험을 통해 LinVideo는 기존 방법 대비 1.25배에서 2배의 속도 향상을 이루었으며, 비디오 생성 품질을 유지했습니다. 추가적으로, 4단계로 증류(distillation)한 모델은 지연 시간을 15.92배 줄이면서도 시각적 품질의 손실을 최소화했습니다. 이는 LinVideo보다 더욱 효율적인 후 훈련 방법 중에서 뛰어난 성능을 보여주고 있습니다.



### Unlocking 3D Affordance Segmentation with 2D Semantic Knowledg (https://arxiv.org/abs/2510.08316)
Comments:
          Work in process

- **What's New**: 이 논문은 affordance segmentation(기능적 분할) 작업을 수행하기 위해 2D 비전 모델(Vision Foundation Models)에서 세분화된 의미 정보를 3D 도메인으로 전이하는 새로운 학습 패러다임을 제안합니다. 특히 Cross-Modal Affinity Transfer (CMAT)라는 사전 훈련 전략을 통해 3D 인코더가 2D 의미와 정렬되도록 합니다. 이를 통해, 기존 3D 데이터의 부족한 의미적 경계를 극복하고 더욱 명확한 구조적 특징을 학습할 수 있습니다.

- **Technical Details**: 제안된 방법론은 세 단계로 구성됩니다: 첫 번째 단계에서는 2D 비전 모델에서 밀집된 의미적 지침을 추출합니다(Foundational Semantic Grounding). 두 번째 단계에서는 Cross-Modal Affinity Transfer(CMAT) 목표를 활용하여 3D 포인트 클라우드 인코더를 훈련시킵니다(Structured Representation Learning). 마지막 단계에서는 Cross-modal Affordance Segmentation Transformer(CAST) 아키텍처를 사용하여 사전 훈련된 백본을 affordance segmentation 작업에 맞게 조정합니다(Prompt-driven Task Adaptation).

- **Performance Highlights**: 실험 결과, 이 논문의 프레임워크는 3D affordance segmentation의 표준 벤치마크에서 새로운 최첨단 성능을 달성했습니다. 제안된 접근방식은 명확하고 기능적으로 일관된 경계를 가진 세분화된 맵을 생성하며, 다양한 미션에 맞게 조정된 통합된 특징을 제공합니다. 이러한 성능 개선은 전례 없는 수준의 구조적 특징을 제공하여 로봇 조작, 구현된 AI 및 모방 학습과 같은 실제 애플리케이션에서도 활용 가능할 것으로 기대됩니다.



### LTCA: Long-range Temporal Context Attention for Referring Video Object Segmentation (https://arxiv.org/abs/2510.08305)
Comments:
          Accepted by IEEE TCSVT

- **What's New**: 이 논문에서는 Referring Video Segmentation (RVOS) 문제를 해결하기 위해 새로운 Long-range Temporal Context Attention (LTCA) 메커니즘을 제안합니다. 기존의 접근 방식들은 로컬리티(locality)와 글로벌리티(globality) 간의 균형을 잘 맞추지 못했으며, 영상 길이 증가에 따라 계산 복잡도가 크게 증가하는 문제를 지니고 있었습니다. LTCA는 이러한 문제들을 해결하며, 텍스트와 비디오 간의 상호작용에서 장기적인 맥락 정보를 효과적으로 집계하는 방법으로 알려져 있습니다.

- **Technical Details**: LTCA는 두 가지 주요 방법으로 글로벌 맥락 정보를 집계합니다. 첫째, 희소한 로컬 어텐션(sparse local attentions)을 쌓아 로컬리티와 글로벌리티를 균형있게 유지합니다. 둘째, 랜덤 어텐션(random attention) 메커니즘을 도입하여 각 쿼리가 전체 풀에서 무작위로 선택된 소규모 키 그룹에 주의를 기울이게 하여 글로벌 맥락 정보를 강화합니다. 이러한 접근 방식을 통해 LTCA는 각 프레임의 정보와 텍스트 정보를 효과적으로 통합합니다.

- **Performance Highlights**: 제안된 방법은 MeViS, Ref-YouTube-VOS, Ref-DAVIS17, A2D-Sentences 등 네 가지 벤치마크에서 새로운 최첨단 성능을 달성하였습니다. 특히 MeViS 데이터셋에서 11.3%의 성능 향상을 보여 LTCA 메커니즘이 장기적인 맥락 정보 집계에 매우 효과적임을 입증하였습니다. 이는 RVOS 분야에서의 LTCA의 유용성을 강조하며, 다양한 실제 응용 프로그램에서 활용될 가능성을 제시합니다.



### Learning Neural Exposure Fields for View Synthesis (https://arxiv.org/abs/2510.08279)
Comments:
          Accepted to NeurIPS 2025. Project page available at this https URL

- **What's New**: 이 논문에서는 신경 노출 필드(Neural Exposure Fields, NExF)라는 혁신적인 기술을 제시하여, 도전적인 실제 캡처 자료로부터 고품질의 3D 장면을 일관되게 재구축할 수 있도록 하였습니다. 기존의 기술들이 고정된 이미지/픽셀에서 최적의 노출 값을 선택하는 것과 달리, NExF는 3D 포인트별로 최적 노출 값을 예측하는 신경 필드를 학습하여 3D 환경에서 최적화를 수행합니다. 이로 인해 높은 동적 범위의 시나리오에서도 정확한 뷰 합성이 가능해졌습니다.

- **Technical Details**: NExF는 3D 장면 표현과 노출 필드를 공동으로 학습하는 시스템을 가지고 있으며, 이는 새로운 잠재적 노출 조건화 메커니즘(latent exposure conditioning mechanism)을 통해 이루어집니다. 또한, 3D 정보의 집합을 통해 우리의 모델은 3D 기존성이 높아지며, 별도의 2D 톤 매핑(tone mapping) 프로세스가 필요하지 않습니다. 이 방법은 이전 작업들보다 훈련 속도가 빠르고, 여러 벤치마크에서 55% 이상 성능을 개선하는 것을 보여주었습니다.

- **Performance Highlights**: NExF는 도전적인 실제 데이터에 대해 우수한 성능을 입증하였으며, 이는 최신 기술들에 비해 55% 이상의 향상을 보여줍니다. 특히 다양한 노출 변화가 있는 큰 규모의 캡처 자료에서도 고품질의, 3D 일관된 결과를 생성할 수 있습니다. 이러한 성과는 신경 3D 장면 표현을 더욱 복잡한 실세계 사용 사례에 가까운 형태로 발전시키는 중요한 진전을 의미합니다.



### A Multimodal Depth-Aware Method For Embodied Reference Understanding (https://arxiv.org/abs/2510.08278)
- **What's New**: 이 논문에서는 복합적인 환경에서의 목표 객체 탐지 개선을 위해 Embodied Reference Understanding (ERU) 프레임워크를 제안합니다. 이 프레임워크는 LLM 기반 데이터 증강, 깊이 맵(depth-map) 모달리티, 그리고 깊이 인식 결정 모듈을 결합하여 언어적 및 비언어적 신호의 통합을 강화합니다. 기존 모델이 다중 후보 객체를 올바르게 식별하는 데 어려움을 겪는 복잡한 장면에서 효과적으로 작동하는 것을 목표로 합니다.

- **Technical Details**: ERU 모델은 두 개의 병렬 모델, Ma​u​gM_{aug} 및 Md​e​p​t​hM_{depth}로 구성되어 있습니다. Ma​u​gM_{aug}는 증강된 데이터로 훈련되며, Md​e​p​t​hM_{depth}는 깊이 맵을 포함한 비증강 데이터로 훈련됩니다. 이 두 모델의 예측을 통합하는 깊이 인식 결정 모듈(DADM)을 통해 더욱 정확한 bounding box 예측이 가능합니다.

- **Performance Highlights**: 실험 결과는 제안된 ERU 접근 방식이 두 개의 benchmark에서 기존 방법들보다 우수한 성능을 보이며, 목표 객체 탐지의 정확성과 신뢰성을 크게 개선함을 보여줍니다. 특히, 깊이 정보를 통합한 모델은 복잡하거나 혼잡한 환경에서도 보다 효과적으로 작동하며, 깊이 단서가 결여된 경우의 실패를 극복하는 데 도움을 줍니다.



### One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting (https://arxiv.org/abs/2510.08273)
Comments:
          25 pages, 11 figures, to appear NeurIPS 2025

- **What's New**: 이 논문은 텍스트 기반 이미지를 복원하는 과정에서 기존 방법들이 가지고 있는 두 가지 주요 문제인 비가려진 지역의 보존과 가려진 지역과 비가려진 지역 간의 의미적 일관성 문제를 해결하는 새로운 접근법인 NTN-Diff를 제안합니다. NTN-Diff는 하이브리드 주파수 대역을 활용하여 텍스트에 맞춰 이미지를 복원하는 과정에서 이러한 문제를 해결하고자 합니다. 또한, 초기 및 후기 단계에서 노이즈 처리 과정이 분리되어 병렬적으로 진행됩니다.

- **Technical Details**: 이 방법론의 핵심은 두 개의 텍스트 가이드 노이즈 제거 프로세스를 활용하여 가려진 영역을 보완하는 데 있습니다. 초기 단계에서는 텍스트 프롬프트의 영향을 받지 않도록 저주파 대역을 분리하고, 이를 통해 비가려진 지역을 보존하며 고주파 대역을 정제합니다. 이후 중간 단계에서는 비가려진 지역의 중주파 대역이 텍스트 프롬프트와 정렬되도록 하여 의미적 일관성을 유지하게 됩니다.

- **Performance Highlights**: 실험을 통해 NTN-Diff가 기존의 최신 텍스트 안내 이미지 복원 모델들을 능가함을 입증했습니다. 특히, NTN-Diff는 가려진 지역과 비가려진 지역 간의 의미적 일관성을 보존하면서도 비가려진 지역을 효과적으로 유지하는 성능을 보여주었습니다. 전반적으로, NTN-Diff는 텍스트 기반 이미지 인페인팅에 있어 혁신적인 솔루션으로 자리매김할 가능성이 있습니다.



### Adaptive Gradient Calibration for Single-Positive Multi-Label Learning in Remote Sensing Image Scene Classification (https://arxiv.org/abs/2510.08269)
Comments:
          14 pages, 6 figures

- **What's New**: 이 논문에서는 기존의 단일 레이블 분류(single-label classification) 방식이 가진 한계를 극복하기 위해, Remote Sensing (RS) 이미지를 위한 Adaptive Gradient Calibration (AdaGC)이라는 새로운 SPML 프레임워크를 제안합니다. 전통적인 MLC (multi-label classification)의 과제를 해결하고, 효과적인 레이블 생성 및 훈련 과정의 유연성을 높이는 데 초점을 맞추고 있습니다. 기존 SPML 방식은 RS 데이터에 적합하지 않은 여러 문제를 해결하기 위해 특히 깊이 있는 분석을 기반으로 한 과정을 추가하였습니다.

- **Technical Details**: 논문에서는 AdaGC가 Gradient Calibration (GC) 메커니즘과 Mixup, Dual Exponential Moving Average (EMA) 모듈을 결합하여 강력한 준 레이블(pseudo-label) 생성을 위한 방법론을 제시합니다. 이 프레임워크는 단일 양성 레이블(single-positive label)만을 가진 이미지를 다루며, 약한 감독 아래에서 전체 레이블 세트를 복구할 수 있도록 설계되었습니다. 초기 워밍업(warm-up) 단계 이후 학습 동학(training dynamics)을 기반으로 GC를 적절히 동작시켜 레이블 노이즈에 대한 과적합(overfitting)을 줄이는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 실험을 통해 두 개의 벤치마크 RS 데이터셋에서 두 가지 레이블 노이즈 타입을 적용하여 AdaGC의 효과성과 견고함을 입증하였습니다. 결과적으로, AdaGC는 다양한 환경에서 최신 기술의 성과(State-of-the-art, SOTA)를 달성하며, 기존 SPML 방법들에 비해 우수한 성능을 보여주었습니다. 이 연구는 SPML이 RS 이미지 분류에서 어떻게 활용될 수 있는지를 보여주는 중요한 기초 자료가 됩니다.



### Fine-grained text-driven dual-human motion generation via dynamic hierarchical interaction (https://arxiv.org/abs/2510.08260)
- **What's New**: 본 연구에서는 기존의 단일 인간 모션 생성을 수행하는 방식과는 달리, 동적 계층적 인간 상호작용을 정교하게 모델링하는 FineDual이라는 이중 인간 모션 생성 방법을 제안합니다. 이 방법은 세 가지 단계를 통해 개인 및 상호 개인 수준의 상호작용을 모델링하며, 이를 통해 더 나은 품질의 인간 모션을 생성할 수 있습니다. 특히, 상호작용 거리(Interaction Distance) 개념을 도입하여 인간 간의 상호작용을 동적으로 조정합니다.

- **Technical Details**: FineDual은 세 단계로 구성된 계층적 접근 방식을 통해 이중 인간 간의 모션을 생성합니다. 첫 번째 단계인 Self-Learning Stage는 대형 언어 모델(Large Language Model)을 사용해 텍스트 프롬프트를 개인 수준으로 분해하여 텍스트 특징과 모션 특징을 정렬합니다. 두 번째 단계는 Adaptive Adjustment Stage로, 그래프 네트워크를 통해 상호작용 거리를 예측하고, 마지막 Teacher-Guided Refinement Stage에서는 전체 텍스트 특징을 가이드로 삼아 모션 특징을 정제합니다.

- **Performance Highlights**: 실험 결과, FineDual은 기존 방법들과 비교하여 우수한 성능을 보였으며, 특히 텍스트 프롬프트를 기반으로 상호작용을 적절히 모델링하여 정교한 이중 인간 모션을 생성하는 데 성공했습니다. 또한, 새로운 평가 메트릭을 도입하여 인간-인간 상호작용을 평가하는 데 기여했습니다. 본 연구의 기여는 동적 계층적 상호작용 프레임워크를 통해 정교한 이중 인간 모션 생성을 가능하게 한 것입니다.



### InstructUDrag: Joint Text Instructions and Object Dragging for Interactive Image Editing (https://arxiv.org/abs/2510.08181)
- **What's New**: 이번 연구에서는 InstructUDrag라는 새로운 확장된 프레임워크를 제안합니다. 이 프레임워크는 텍스트 지시와 오브젝트 드래깅(object dragging)을 결합하여 사용자에게 오브젝트를 자유롭게 드래그하면서 동시에 이미지 편집을 가능하게 해줍니다. 기존 방법의 한계인 정밀한 오브젝트 위치 조정 문제를 해결하고, 이미지 품질을 높이는 새로운 구조적 접근 방식을 소개합니다.

- **Technical Details**: InstructUDrag는 두 개의 시너지 효과를 내는 분기 구조를 가지고 있습니다. 하나는 오브젝트 재구성(moving-reconstruction) 분기로, 에너지 기반의 그래디언트 가이드를 이용하여 오브젝트를 정확하게 이동시킵니다. 또 다른 분기는 텍스트 기반 편집(text-driven editing) 분기로, 오브젝트 재구성 분과의 그래디언트 신호를 공유하여 변환 일관성을 유지하며 세부적인 조정을 가능케 합니다.

- **Performance Highlights**: 실험 결과, InstructUDrag는 유연하고 높은 품질의 이미지 편집을 가능하게 해줍니다. 오브젝트의 정확한 재배치 및 이미지 내용에 대한 의미적 제어를 제공하여, 사용자에게 필요한 다양한 조작을 수행할 수 있도록 합니다. 이러한 방식으로, 사용자들은 오브젝트를 이동시키면서도 다른 이미지 속성을 효율적으로 조정할 수 있습니다.



### Robust Canonicalization through Bootstrapped Data Re-Alignmen (https://arxiv.org/abs/2510.08178)
- **What's New**: 이번 논문은 섬세한 시각 분류(Fine-grained visual classification, FGVC) 작업에서 발생하는 기하학적 편향과 노이즈를 다루기 위한 새로운 부트스트랩핑 알고리즘을 제안합니다. 기존의 방법들은 모델의 표현력을 제한하거나 데이터의 양을 크게 늘리는 데 의존했으나, 본 연구는 교육 데이터를 반복적으로 재정렬하여 이러한 문제를 해결합니다. 이로 인해, 실제 데이터 세트에서 발생하는 다양한 회전 및 스케일에 대한 내성을 유지하면서도 더 안정적인 분류 성능을 제공합니다.

- **Technical Details**: 저자들은 공간적 변형(이동 변형을 제외한 회전 및 크기)에 대한 대응을 중점을 두고 기존 기법을 분석합니다. 그룹 이론을 활용하여 다양한 변환에 대해 균일한 표현을 제공하는 방법론과, canonicalization의 개념을 도입하여 정보를 간소화합니다. 알고리즘은 격렬하게 다양한 설정에서 수렴을 보장하며, 기능적으로 효과적이고 적은 비용으로 공간적 변동성을 줄이는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안한 방법은 네 가지 FGVC 벤치마크에서 기존의 equvariant 및 canonicalization 기법보다 일관되게 뛰어난 성능을 보이며, 데이터 세트의 재정렬이 생물 다양성 모니터링 및 관련 도메인에 있어 얼마나 중요한지를 입증합니다. 이 연구는 효과적인 데이터 정렬이 어떻게 성능 향상에 기여하는지를 보여주며, 실제 데이터 환경에서의 활용 가능성을 제시합니다.



### Beyond Textual CoT: Interleaved Text-Image Chains with Deep Confidence Reasoning for Image Editing (https://arxiv.org/abs/2510.08157)
Comments:
          25pages,20figures

- **What's New**: 본 논문은 자연어로 이미지 편집하는 기존 방법들이 복잡한 객체 교차와 정밀한 공간 관계를 처리하는 데 한계를 보인다는 문제를 다룹니다. 이에 따라, MURE(텍스트-이미지 공합 텍스트로 추론 프로세스를 바탕으로한 이미지 편집 용이 프레임워크를 제안합니다. MURE는 다양한 시각적 입력과 텍스트적 설명을 결합하여 편집 과정을 단계별로 진행하며, 이 과정에서 중간적인 시각적 단서들을 제공합니다.

- **Technical Details**: MURE는 텍스트-이미지 연결을 기반으로 한 멀티모달 추론 체인을 통해 복잡한 편집 작업을 상호 의존적인 하위 과제로 나누어 더 높은 품질을 추구합니다. 이러한 접근법을 통해 모델은 최종 결과물에 대한 더 나은 경로를 보장하며, MMDC(멀티모달 딥 신뢰도) 추론 패러다임을 통해 저품질 경로를 제거함으로써 전반적인 신뢰성을 향상시킵니다. 본 논문에서는 interleaved text-image chains의 수식을 정의하고, 14K 고품질 편집 예시를 포함한 최초의 CoT-Edit-14K 데이터셋을 공개하였습니다.

- **Performance Highlights**: MURE는 세 가지 이미지 편집 벤치마크에서 눈에 띄는 성능 향상을 보여주었으며, 다양한 편집 작업에 대해 높은 충실도를 유지합니다. 특히, 복잡한 객체의 혼합과 세밀한 세부 사항을 처리하는 데 있어 MURE의 접근법이 효과적임을 입증하였습니다. 최종적으로, 본 방법론은 편집 과정의 각 단계에서 퀄리티를 회복하며, 물리적으로 일관된 결과물을 만들어내는 데 기여합니다.



### UniMMVSR: A Unified Multi-Modal Framework for Cascaded Video Super-Resolution (https://arxiv.org/abs/2510.08143)
- **What's New**: 이번 논문에서는 UniMMVSR라는 새로운 Video Super-Resolution 모델을 소개합니다. 이 모델은 텍스트, 이미지, 비디오와 같은 다양한 조건을 통합하여 고해상도 비디오 생성을 할 수 있는 첫 번째 통합 프레임워크입니다. 이를 통해 기존의 텍스트-비디오 생성 모델에서 더 나아가, 다양한 멀티모달 조건을 활용하여 더욱 생생한 비디오를 생성할 수 있게 되었습니다.

- **Technical Details**: UniMMVSR는 잠재적 비디오 확산 모델(latent video diffusion model)에 기반하여 조건 주입 전략(condition injection strategies), 훈련 스킴(training schemes) 및 데이터 혼합 기법(data mixture techniques)을 탐색합니다. 이를 위해, 저해상도 비디오 외에 여러 ID 이미지와 참조 비디오를 조건으로 포함하여 다양한 입력을 적절히 활용하는 방법을 연구했습니다. 특히, 조건 토큰에 독립적인 위치 임베딩(position embedding)을 부여하여 모델이 문맥에 따라 모든 참조 조건을 효과적으로 사용할 수 있도록 설계하였습니다.

- **Performance Highlights**: 실험 결과, UniMMVSR는 기존 모델들보다 높은 시각적 충실도(visual fidelity)를 보이며, 멀티모달 참조에 충실한 고해상도 비디오를 생성하는 데 성공하였습니다. 특히, 4K 비디오 생성이 가능해지면서 현재 기술로는 도달할 수 없던 새로운 가능성을 제시했습니다. 이러한 결과는 UniMMVSR가 복잡한 멀티모달 작업을 수행하는 데 매우 효과적임을 보여줍니다.



### Improving Temporal Understanding Logic Consistency in Video-Language Models via Attention Enhancemen (https://arxiv.org/abs/2510.08138)
- **What's New**: 최근 대형 언어 모델(LLMs)에서 자기 모순적인 출력이 발생하여 신뢰성에 큰 영향을 미치는 현상이 주목받고 있습니다. 비디오-언어 모델(Video-LLMs)에서도 이와 유사한 문제에 직면하고 있으며, 특히 재구성된 질문에 대한 논리적으로 일관된 답변을 제공하지 못하는 경향이 있습니다. 본 연구는 이러한 현상의 원인을 분석하기 위해 해석 가능성을 기반으로 한 접근 방식을 채택하였습니다.

- **Technical Details**: 연구에서는 크로스 모달 주의(attention) 헤드가 서로 다른 타임스탬프에서 비디오 토큰을 효과적으로 구분하지 못하는 한 가지 주요 원인을 확인하였습니다. 이 문제를 해결하기 위해, Temporally Conditioned Attention Sharpening (TCAS)이라는 주의 강화 방법을 제안하여 모델의 시간 해상도를 개선하고, 논리적 일관성을 향상시키는 목적을 구축하였습니다. 실험 결과 TCAS 방법이 Video-LLMs의 시간 논리 일관성을 상당히 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 실험을 통해 TCAS가 다양한 비디오 기반 시간 도달(VTG) 작업에서 성능 향상을 이루었으며, 시간 논리 일관성이 시간 이해의 병목 현상임을 강조합니다. 모델이 일관성을 증대시키도록 유도함으로써 비디오 시간 이해의 중요한 진전을 이끌어냈습니다. 또한 해석 가능성 분석을 통해 TCAS가 주의 헤드의 시간 구별 능력을 개선한 사실을 검증하였습니다.



### Real-Time Motion-Controllable Autoregressive Video Diffusion (https://arxiv.org/abs/2510.08131)
- **What's New**: 이번 논문에서는 AR-Drag을 제안합니다. 이는 최초의 강화 학습(RL) 기반 몇 단계 AR 비디오 확산 모델로, 다양한 모션 제어를 지원하며 실시간 이미지-비디오 생성이 가능합니다. 기존의 bidirectional 비디오 확산 모델들이 가진 고질적인 지연 문제를 해결하여 낮은 대기 시간을 목표로 하고 있습니다.

- **Technical Details**: AR-Drag 모델은 우선 기본 모션 제어를 가능하게 하기 위해 I2V(base Image to Video) 모델을 미세 조정한 후, 경로 기반 보상 모델을 통해 강화 학습으로 더 개선됩니다. Self-Rollout 메커니즘을 통해 마르코프(Markov) 특성을 보존하고, 디노이징 단계에서 선택적으로 확률성을 도입하여 훈련을 가속화하는 설계가 특징입니다.

- **Performance Highlights**: 광범위한 실험을 통해 AR-Drag는 시각적 충실도(visual fidelity)와 모션 정렬(motion alignment)에서 높은 성능을 발휘하며, 최신의 모션 제어 VDM과 비교하여 대기 시간을 상당히 줄였습니다. 또한, 이 모델은 1.3B의 파라미터만을 사용하면서도 품질과 제어 가능성을 크게 개선한 것으로 입증되었습니다.



### Random Window Augmentations for Deep Learning Robustness in CT and Liver Tumor Segmentation (https://arxiv.org/abs/2510.08116)
Comments:
          10 pages, 9 figures. This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 CT(Computed Tomography) 영상의 데이터 증강 기법에서 중요한 문제를 다룹니다. 기존의 자연 이미지용 강도 변환(일반적 intensity augmentations) 기법을 CT 이미지에 부적절하게 적용할 경우 성능 저하와 아티팩트가 발생할 수 있음을 밝혔습니다. 이를 해결하기 위해 새로운 CT 전용 증강 기법인 '랜덤 윈도잉(Random windowing)'을 제안하며, 이는 CT 이미지에서 Hounsfield 단위(HU)의 분포를 활용합니다.

- **Technical Details**: 랜덤 윈도잉 기법은 기존 강도 증강 기법을 대체할 수 있는 새로운 방법으로, CT 영상의 특정 구역에 대한 견고성을 높입니다. 이 방법은 CT 이미지의 HU 분포를 존중하여 제출함으로써 임상 적용에서의 효과를 극대화합니다. 여러 데이터 세트에서 기법의 효과를 정량적으로 분석하고, 제조된 방법과 기존의 최첨단 홍수화 기반 방법들을 비교하여 뛰어난 성능을 확인하였습니다.

- **Performance Highlights**: 랜덤 윈도잉 기법은 특히 영상의 대비와 시간 안전성이 떨어지는 어려운 CT 이미지에서 모델 성능을 크게 향상시키는 결과를 보였습니다. 이 연구는 간 종양(segmentation of liver tumors) 이미지를 대상 문제로 설정하였고, 랩 데이터 부족 문제를 해결하는 데 큰 기여를 할 것으로 기대됩니다. 또한, 기존의 증강 기법들과 비교하여 월등한 결과를 보여주며, CT 영상 처리 분야에서의 변화를 이끌 것으로 예상됩니다.



### Efficient Label Refinement for Face Parsing Under Extreme Poses Using 3D Gaussian Splatting (https://arxiv.org/abs/2510.08096)
Comments:
          Accepted to VCIP 2025 (International Conference on Visual Communications and Image Processing 2025)

- **What's New**: 이 논문에서는 극단적인 시점에서의 정확한 얼굴 파싱(face parsing) 문제를 해결하기 위해 3D Gaussian Splatting (3DGS) 기반의 새로운 레이블 정제 파이프라인(label refinement pipeline)을 제안합니다. 기존의 주석 데이터가 부족한 상황에서, 이 방법은 소음이 있는 다중 보기 예측으로부터 정확한 세그멘테이션 마스크(segmentation masks)를 생성합니다. 수동 주석 작업이 아닌 3D 모델을 사용하여 포즈 다양성(pose diversity)을 누릴 수 있는 새로운 훈련 데이터를 합성할 수 있습니다.

- **Technical Details**: 우리가 제안한 방법은 두 단계로 구성되어 있습니다: (1) 3DGS 기반의 레이블 정제 파이프라인, 그리고 (2) 개선된 레이블을 사용하여 파싱 모델을 미세 조정합니다. 3DGS는 장면을 비대칭 3D 가우시안 원추(facial segments) 세트를 사용하여 표현하며, 공유 기하학(shared geometry)을 통해 멀티뷰 일관성(multiview consistency)을 강화하여 레이블 소음을 감소시킵니다. 이 과정에서 RGB 이미지와 초기 세분화 맵을 결합하여 미세 조정용 보조 데이터셋(auxiliary dataset)을 생성합니다.

- **Performance Highlights**: 논문에서 제안한 파이프라인은 BiSeNet 모델을 사용하여 미세 조정 시, 극단적인 머리 자세에서도 강력한 성능 향상을 보여주었습니다. 기존의 주석 3D 데이터 없이 한정된 이미지만으로도, 수정된 세그멘테이션 마스크는 최신 방법들에 비해 우수한 성과를 보였으며, 사람 평가에서도 경쟁력 있는 결과를 확보했습니다. 이 방법은 실제 세계 설정에서 얼굴 파싱의 강인성을 향상시키기 위한 확장 가능하고 효과적인 솔루션을 제공합니다.



### DarkHash: A Data-Free Backdoor Attack Against Deep Hashing (https://arxiv.org/abs/2510.08094)
Comments:
          Accepted by TIFS 2025

- **What's New**: 본 논문에서는 DarkHash라는 새로운 기법을 제안합니다. DarkHash는 훈련 데이터 없이 딥 해시 모델에 대한 첫 번째 백도어 공격을 수행할 수 있는 시스템으로, 보다 효과적인 공격 성능과 정상 검색 정확도를 동시에 유지합니다. 이는 이미지 검색의 정확성을 유지하는 동시에 공격력을 높이는 혁신적인 접근 방식을 보여줍니다.

- **Technical Details**: DarkHash는 이중 의미 지침을 바탕으로 한 그림자 백도어 공격 프레임워크를 설계합니다. 이 시스템은 대체 데이터셋 사용하여 피해 모델의 특정 층을 미세 조정함으로써 백도어 기능을 포함합니다. 훈련 중 개별 샘플과 이웃 간의 관계를 활용하여 백도어 공격 성능을 향상시키기 위해 토폴로지 정렬 손실(topological alignment loss)을 사용합니다.

- **Performance Highlights**: 네 개의 이미지 데이터셋과 다섯 가지 모델 아키텍처, 두 가지 해싱 방법을 통한 실험 결과, DarkHash는 뛰어난 공격 성능을 보여줍니다. 평균 t-mAP는 80%를 초과했으며, 기존 SOTA 백도어 공격 기법보다 높은 검색 정확도와 공격 성능을 달성했습니다. 방어 실험에서도 DarkHash는 기존의 모든 주류 방어 방법에 저항력을 보여줍니다.



### Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection (https://arxiv.org/abs/2510.08073)
Comments:
          Accepted at NeurIPS 2025 spotlight

- **What's New**: 이번 논문은 AI 생성 비디오의 신뢰할 수 있는 탐지 메커니즘을 필요로 한다는 점에서 새로운 점을 강조합니다. 저자들은 물리학 기반의 확률 흐름 보존 원칙에 기반하여 AI 생성 비디오를 탐지하는 새로운 패러다임을 제안하며, 이는 AI 생성 비디오의 시공간적 오류를 효과적으로 감지할 수 있는 방법입니다. 제안된 기법인 Normalized Spatiotemporal Gradient (NSG)는 실제 비디오와 AI 생성 비디오 간의 기본적인 불일치를 정량화합니다.

- **Technical Details**: NSG는 공간 확률 기울기와 시간 밀도 변화의 비율을 정량화하여 AI 생성 비디오의 물리적 제약 위반을 포착합니다. 저자들은 사전 훈련된 diffusion 모델을 활용하여 복잡한 모션 분해 없이 NSG 추정기를 개발하였으며, 이를 통해 영상의 동적 특성을 효과적으로 모델링할 수 있습니다. 또한, NSG 기반 비디오 탐지 방법(NSG-VD)은 테스트 비디오와 실제 비디오 간 NSG 특성의 Maximum Mean Discrepancy (MMD)를 검출 메트릭으로 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과 NSG-VD는 Recall에서 16.00% 및 F1-Score에서 10.75% 향상된 성능을 보여주었으며, 이는 기존의 최첨단 모델보다 뛰어난 성능을 подтверж합니다. 이러한 결과는 NSG-VD의 탐지 능력이 우수함을 입증하며, AI 생성 비디오 감지에 있어 중요한 진전을 나타냅니다. 저자들은 이 연구의 소스 코드도 공개하고 있어 향후 연구에 활용될 수 있는 기초 자료를 제공합니다.



### Towards Real-World Deepfake Detection: A Diverse In-the-wild Dataset of Forgery Faces (https://arxiv.org/abs/2510.08067)
- **What's New**: 본 논문에서는 RedFace(Real-world-oriented Deepfake Face)라는 새로운 심층 가짜 얼굴 데이터셋을 소개합니다. 이 데이터셋은 60,000개 이상의 조작된 이미지와 1,000개의 조작된 비디오로 구성되어 있습니다. RedFace는 기존 데이터셋이 갖고 있는 문제를 해결하고 실제 세계의 딥페이크 감지 필요성을 충족시키기 위해 다양한 상업적인 온라인 플랫폼을 활용하여 깊이 있는 데이터셋을 구축했습니다.

- **Technical Details**: RedFace 데이터셋은 얼굴 합성(Entire Face Synthesis), 얼굴 교환(Face Swapping), 얼굴 속성 조작(Face Attribute Manipulation), 얼굴 애니메이션(Face Reenactment)의 네 가지 주요 시나리오로 구분됩니다. 기존 데이터셋은 일반적으로 유사한 연구 논문에서 생성된 예를 기반으로하지만, RedFace는 실제 사용자들이 사용하는 다양한 상업적인 플랫폼을 통해 생성된 깊이 있는 데이터를 포함하고 있습니다. 이는 실제 환경에서의 BLACK-BOX 방식의 시뮬레이션을 효과적으로 구현합니다.

- **Performance Highlights**: 다양한 실험을 통해 RedFace 데이터셋이 기존의 데이터셋에 비해 감지 성능에 미치는 영향이 크게 다름을 입증했습니다. 현재의 딥페이크 감지 기법들이 BLACK-BOX 환경에서 생성된 데이터에 대해 충분한 효과를 발휘하지 못함을 보여주었으며, 이는 실제 환경에서의 감지 기법의 취약점을 드러냅니다. 더불어, RedFace 데이터셋을 활용한 다양한 딥페이크 감지 방법의 평가를 통해 미래 연구의 방향성을 제시하고 있습니다.



### A class-driven hierarchical ResNet for classification of multispectral remote sensing images (https://arxiv.org/abs/2510.08060)
Comments:
          11 pages, 2 figures, accepted conference paper at SPIE REMOTE SENSING, 3-7 September 2023, Amsterdam, Netherlands

- **What's New**: 이 논문에서는 다중 시간대(class-driven hierarchical) 잔차 신경망(Residual Neural Network, ResNet)을 제안하여 다채널 이미지의 분류 모델링을 수행합니다. 기존 ResNet 아키텍처를 수정하여 여러 계층적(class hierarchy) 수준에서 분류를 수행하는 추가 지점을 도입하였으며, 일관성 없는 계층 전환을 억제하기 위한 계층 패널티 맵을 활용했습니다. 이러한 접근 방식은 다양한 의미론적 세부 사항을 가진 클래스의 구별 능력을 향상시킵니다.

- **Technical Details**: 제안된 방법은 다중 스펙트럼 영상의 계층적(class hierarchy) 분류를 위한 수정된 잔차 신경망(ResNet)에 기반하고 있습니다. 이 아키텍처는 각 클래스가 계층적으로 정리되어 있는 점을 활용하여, 상위 계층에서 베이스 클래스를, 하위 계층에서는 마이크로 클래스를 더 정밀하게 식별하도록 합니다. 활용된 잔차 신경망의 입력은 Sentinel-2 이미지를 12개월의 조합으로 구성하여 시계열 데이터의 복잡한 패턴을 효과적으로 캡처합니다.

- **Performance Highlights**: 실험 결과는 아마존 숲의 두 개 타일에서 Sentinel-2 이미지를 사용하여 수집되었으며, 제안된 계층적 접근법이 다양한 계층에 걸쳐 일반화 및 미세 클래스(micro-class) 수준에서의 정확한 분류에 효과적임을 보여주었습니다. 특히, 소수의 데이터를 갖는 마이너 클래스(minoritarian classes)에 대한 표현력이 향상되었습니다. 이러한 성능 향상은 최초의 계층에서 더 일반적인 클래스를 빠르게 학습하는 동시에, 심화된 계층에서 더 구체적인 클래스를 식별하면서 이루어졌습니다.



### RetouchLLM: Training-free White-box Image Retouching (https://arxiv.org/abs/2510.08054)
- **What's New**: 이미지 리터칭(image retouching)은 시각적 품질을 향상시키고 개인의 선호도와 감정을 표현하는 과정입니다. 기존의 방법들은 대규모 훈련 데이터에 의존하며 블랙 박스처럼 작동하여 사용자의 특정 요구나 이미지에 맞춘 조정이 어렵습니다. 본 연구에서는 RetouchLLM을 제안하는데, 이는 훈련 데이터 없이 고해상도 이미지에서 직접 사용자가 이해할 수 있는 코드 기반의 리터칭을 가능하게 합니다.

- **Technical Details**: RetouchLLM은 비 훈련(training-free) 화이트 박스(white-box) 이미지 리터칭 시스템으로, 사용자 지침을 통해 미세 조정(fine-grained adjustments)을 지원합니다. 이 시스템은 입력 이미지와 참조 이미지 간의 차이를 식별하는 비주얼 비평가(visual critic)와 실행 가능한 코드를 생성하는 코드 생성기(code generator)로 구성됩니다. 이 방법은 iterative retouching framework를 사용하여 이미지를 점진적으로 정제하는 방식으로, 이러한 과정은 사람의 리터칭 방식을 모방합니다.

- **Performance Highlights**: RetouchLLM은 다양한 리터칭 스타일에서 좋은 성능을 보이며, 사용자와의 자연어 대화를 통해 사용자 의도에 맞는 조정을 가능하게 합니다. 실험 결과, iterative 방식으로 진행된 리터칭이 이미지 품질을 지속적으로 향상시키는 것으로 나타났습니다. 우리의 접근 방식은 새로운 스타일이나 필터를 유연하게 추가할 수 있고, 특정 사용자 선호를 쉽게 반영하는데 도움을 줍니다.



### RASALoRE: Region Aware Spatial Attention with Location-based Random Embeddings for Weakly Supervised Anomaly Detection in Brain MRI Scans (https://arxiv.org/abs/2510.08052)
Comments:
          Accepted in BMVC-2025

- **What's New**: 이 연구에서는 약한 레이블(weak label)만을 사용할 수 있는 뇌 MRI 스캔에서의 약한 감독 이상 탐지(WSAD) 문제를 해결하기 위한 새로운 프레임워크인 RASALoRE를 제안합니다. RASALoRE는 두 단계로 구성된 WSAD 프레임워크로, 첫 번째 단계에서는 slice-level 레이블을 기반으로 고품질의 유사 약한 마스크(pseudo weak masks)를 생성하기 위한 Discriminative Dual Prompt Tuning (DDPT) 메커니즘을 도입합니다. 두 번째 단계에서는 고정 위치 기반 랜덤 임베딩(location-based random embeddings)을 활용하여 이상 지역에 효과적으로 집중할 수 있는 분할 네트워크(segmentation network)를 제안합니다.

- **Technical Details**: RASALoRE는 slice-level 레이블만을 사용하여 동작하며, 두 가지 단계로 구성됩니다. 첫 번째 단계인 DDPT는 미리 훈련된 비전-언어 모델을 활용하여 뇌 MRI 스캔 이미지의 분류 작업을 수행하고 유사 이상 마스크를 생성합니다. 두 번째 단계에서는 고정 위치 기반 랜덤 임베딩을 활용하여 이상을 정밀하게 로컬라이즈하도록 설계된 분할 네트워크가 훈련됩니다. Pixel-level 주석은 없지만, slice-level 레이블이 존재하는 상황에서 이상 탐지 작업을 이진 분류 문제로 정의합니다.

- **Performance Highlights**: RASALoRE는 BraTS20, BraTS21, BraTS23 및 MSD 데이터셋에서의 광범위한 평가를 통해 기존 WSAD 방법에 비해 우수한 성능을 달성하며, 800만 개 미만의 매개변수로 작업합니다. 이 접근 방식은 기존 방법에 비해 계산 복잡도를 크게 줄이면서도 성능 개선을 보여줍니다. RASALoRE는 뇌 MRI 스캔의 이상 탐지에서 최신 기술을 포함하고 있어, 비전-언어 모달리티에 대한 추가 지원으로 그 활용 가능성을 확장합니다.



### RayFusion: Ray Fusion Enhanced Collaborative Visual Perception (https://arxiv.org/abs/2510.08017)
Comments:
          Accepted by NeurIPS2025

- **What's New**: 최근 자율주행 커뮤니티에서 협업 비주얼 인식 방법이 센서 한계를 해결하는 데 큰 주목을 받고 있습니다. 본 연구에서는 RayFusion을 제안하여, 여러 에이전트의 레이((ray) 점유 정보(occupancy information)를 활용하여 깊이 추정의 모호성을 줄이고 카메라 기반 인식 성능을 향상시킵니다. RayFusion은 카메라 레이의 중복성과 잘못된 긍정 예측을 감소시켜 3D 물체 감지를 보다 정확하게 수행할 수 있도록 합니다.

- **Technical Details**: RayFusion은 레이 점유 정보를 활용하여 물체의 정확한 3D 위치를 로컬라이즈하는 혁신적인 알고리즘으로, 세 가지 주요 설계 요소를 포함합니다. 첫째, 공간-시간 정렬 모듈을 통해 에이전트 정보의 공간 정렬을 수행하고 통신 지연에 대한 견고성을 높입니다. 둘째, 레이 점유 정보 인코딩 모듈에서는 깊이 정보를 카메라 레이에 통합하여 정확한 3D 위치를 제공합니다. 셋째, 다중 스케일 인스턴스 피처 집계 모듈을 활용하여 지역 및 전역 공간 피처의 효과적인 병합을 가능하게 합니다.

- **Performance Highlights**: RayFusion은 DAIR-V2X와 두 개의 시뮬레이션 데이터셋(V2XSet 및 OPV2V)에서 광범위한 실험을 수행하여 기존 최신 모델을 지속적으로 초월하는 성능을 입증했습니다. 특히, 성능 기준 AP70에서 이전 연구보다 각각 3.64, 3.47, 8.21 포인트 향상되었습니다. 이러한 성과는 RayFusion이 협업 비주얼 인식의 효율성을 크게 높일 수 있음을 시사합니다.



### CIR-CoT: Towards Interpretable Composed Image Retrieval via End-to-End Chain-of-Thought Reasoning (https://arxiv.org/abs/2510.08003)
- **What's New**: CIR-CoT는 Composed Image Retrieval (CIR) 작업을 위한 최초의 종합적인 MLLM으로, 명확한 Chain-of-Thought (CoT) 추론을 통합하여 다른 모델들과의 단점을 극복합니다. 기존의 VLMs와 MLLMs는 블랙 박스처럼 기능해 사용자에게 결과의 추론 과정을 이해하기 어렵게 만들었습니다. CIR-CoT는 명시적인 추론 체계를 통해 이러한 프로세스를 투명하게 만들어 줍니다.

- **Technical Details**: CIR-CoT는 이미지 및 수정 텍스트 간의 상호작용을 캡처하기 위해 구조화된 CoT 주석을 사용합니다. 이를 위해 세 단계로 나뉘어진 추론 과정(캡션, 추론, 결론)을 이용하여 데이터를 구축하고, 모델은 이를 학습하여 최종 검색 의도를 써야 합니다. 이 과정은 VLM 기반 접근 방식의 한계를 극복하고, 사용자가 복잡한 지시에 대해 더 나은 해석을 할 수 있도록 돕습니다.

- **Performance Highlights**: CIR-CoT는 FashionIQ와 CIRR와 같은 도메인 내 데이터셋에서 뛰어난 성능을 달성하며, 도메인 외의 CIRCO 데이터셋에 대해서도 강력한 일반화 능력을 입증했습니다. 이 연구는 CIR 작업에서 더 효과적이고 신뢰할 수 있는 검색 시스템으로 나아가는 새로운 경로를 제시합니다.



### GraphEnet: Event-driven Human Pose Estimation with a Graph Neural Network (https://arxiv.org/abs/2510.07990)
- **What's New**: 이번 연구에서는 이벤트 카메라 데이터를 활용한 인체 포즈 추정(Human Pose Estimation) 문제에 Graph Neural Network(GNN)을 적용한 첫 번째 사례로, GraphEnet이라는 새로운 아키텍처를 제안합니다. 이 모델은 고주파수(>250 Hz)로 단일 인물의 2D 포즈를 추정하는 기법으로, 이벤트 카메라의 희소한 특징을 이용합니다. 해당 연구는 이벤트 카메라의 데이터 구조에 맞춰 설계되어 전통적 카메라에서의 포즈 추정 방법과의 차별성을 지닙니다.

- **Technical Details**: GraphEnet은 이벤트 스트림에서 초기 그래프를 구축하고 이 그래프를 처리하여 확률적으로 가장 가능성 있는 관절 위치를 추출하는 두 가지 주요 구성 요소로 이루어져 있습니다. 이 연구에서 제안된 방법은 사건의 흐름을 사용한 데이터 압축인 line segment features를 통해 그래프를 구축하며, 이를 통해 계산 시간과 리소스를 절약할 수 있습니다. GNN이 적용되기 이전의 이벤트 기반 카메라 데이터 처리가 높은 시간 지연을 유발하는 문제를 해결하기 위해, 희소한 이벤트 표현을 사용하는 새로운 접근 방식을 채택하였습니다.

- **Performance Highlights**: GraphEnet은 이벤트-Human 3.6 Million 데이터셋에서 74%의 PCKt@0.4 정확도를 기록하며, 이는 고해상도의 실시간 인체 포즈 추정을 가능하게 합니다. 이는 기존 방법들과 비교 시 유의미한 성능 향상을 나타내며, 특히 사람의 빠른 동작에도 효과적으로 대응할 수 있는 잠재력을 보유하고 있습니다. 추가적으로, 저자들은 각 구성 요소가 최종 결과에 기여하는 바를 평가하기 위해 면밀한 ablation study를 수행하였습니다.



### Is Architectural Complexity Always the Answer? A Case Study on SwinIR vs. an Efficient CNN (https://arxiv.org/abs/2510.07984)
Comments:
          7 pages, 4 figures

- **What's New**: 이 연구에서는 저조도 이미지에서 고주파 세부 사항을 복원하고 심각한 노이즈를 억제하는 과제가 여전히 중요한 도전임을 언급합니다. SwinIR와 같은 대규모 Transformer 모델이 성능의 최전선에서 자리잡고 있지만, 이 모델의 높은 계산 비용이 실제 응용에서 장애 요소가 될 수 있음을 강조합니다. 또한, 경량 합성곱 신경망(CNN)과의 비교를 통해 성능 및 효율성의 중요한 트레이드오프(trade-off)를 조사하고 있습니다.

- **Technical Details**: 실험 결과에 따르면, Transformer 기반의 SwinIR 모델은 최대 신호 대 잡음비(PSNR) 39.03 dB로 더 높은 성능을 달성했으나, 경량 CNN은 37.4 dB라는 경쟁력 있는 PSNR을 제공했습니다. 더욱이, CNN은 단 10 에폭(epoch)의 훈련으로 이 성능을 달성한 반면, SwinIR 모델은 132 에폭이 필요했습니다. CNN의 크기는 SwinIR보다 55배 이상 작아 계산 효율성이 더욱 강조됩니다.

- **Performance Highlights**: 이 연구는 표준 CNN이 실질적인 계산 오버헤드가 적은 상태에서 거의 최첨단의 결과를 제공할 수 있음을 보여줍니다. 이는 자원 제약이 우선 고려대상이 되는 실제 시나리오에서 CNN 사용의 타당성을 제시하는 강력한 근거가 됩니다. 따라서 높은 성능을 유지하면서도 경량화된 방식으로 저조도 이미지 복원 작업을 수행할 수 있는 방법에 대한 새로운 통찰을 제공합니다.



### The impact of abstract and object tags on image privacy classification (https://arxiv.org/abs/2510.07976)
Comments:
          This work has been submitted to the ICASSP 2026

- **What's New**: 이 논문에서는 이미지 프라이버시(privacy) 분류 과업에서 객체 태그(object tags)와 추상 태그(abstract tags) 각각의 역할을 비교하여 어떤 태그가 더 적합한지를 탐구합니다. 연구 결과, 태그 수가 제한적일 때는 추상 태그가 더 효과적이며, 태그 수가 많을 경우에는 객체 관련 정보도 유용하다는 것을 보여줍니다. 이는 이미지 프라이버시 분류기의 정확성을 높일 수 있는 중요한 통찰력을 제공합니다.

- **Technical Details**: 우리는 이미지의 태그 추출에 상업적인 분류기인 ClarifAI를 사용하여 200개의 태그를 생성합니다. 이 태그들은 객체, 행동, 감정 및 추상 개념을 포함시키며, 각 태그는 출력된 이미지에서 존재할 확률과 함께 제공됩니다. 추상성과 구체성의 척도를 기반으로 태그의 구체성을 정량화하여 분류 성능에 미치는 영향을 분석합니다.

- **Performance Highlights**: 연구 결과, 구체적인 태그와 추상적인 태그의 조합이 이미지 프라이버시 분류 성과에 미치는 영향이 확인되었습니다. 태그 수가 적을 때는 추상 태그가 결정적 역할을 하며, 태그 수가 많을 경우에는 구체적인 태그나 두 가지를 조합해도 유사한 성과를 낼 수 있습니다. 이러한 결과는 프라이버시 분류를 향상시킬 수 있는 유용한 지침을 제공합니다.



### Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinemen (https://arxiv.org/abs/2510.07961)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이번 논문에서는 Ultra-High Definition (UHD) 이미지 복원을 위한 Latent Harmony라는 새로운 두 단계 프레임워크를 제안합니다. 기존의 Variational Autoencoders (VAEs)는 고주파 세부 정보를 손실하는 경향이 있어 복원 품질이 저하되는 문제를 해결하고자 합니다. 이 프레임워크는 잠재 공간의 규제를 통해 높은 재구성 능력과 의미적 강건성을 동시에 확보하는 것이 특징입니다.

- **Technical Details**: 주요 기술적 요소로는 LH-VAE와 High-Frequency Low-Rank Adaptation (HF-LoRA) 두 가지가 있습니다. LH-VAE는 시각적 의미 제약과 진행적인 손상 혼란을 도입하여 잠재 공간의 의미적 강건성을 향상시킵니다. HF-LoRA는 복원 모델과 공동 훈련되며, 신뢰성을 높이는 고주파 정렬 손실에 의해 조정됩니다.

- **Performance Highlights**: 제안된 Latent Harmony 프레임워크는 UHD 및 표준 해상도 작업 전반에서 뛰어난 성능을 보여줍니다. 실험을 통해 제안된 접근 방식의 효율성, 지각 품질, 재구성 정확도의 조화로운 균형을 입증합니다. 이 연구는 UHD 이미지 복원 분야의 성능을 획기적으로 개선할 가능성을 나타냅니다.



### SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation (https://arxiv.org/abs/2510.07953)
Comments:
          accepted by ICME 2025

- **What's New**: 본 연구는 단기 및 장기 강수 예측에 대한 새로운 접근 방식인 SimCast를 제안합니다. 이 방법은 강수 예측 모델의 예측 지평선(prediction horizon)이 모델의 성능에 미치는 영향을 분석하여, 강수 예보의 정확성을 높이고자 합니다. SimCast는 짧은 기간의 데이터를 이용하여 장기 예측 성능을 향상시키는 지식 증류(knowledge distillation) 기법과 가중 평균 제곱 오차(weighted MSE loss)를 활용합니다.

- **Technical Details**: SimCast는 현재 2시간 이내의 강수 예측을 목표로 하는 단기 모델을 훈련시키고, 이를 기반으로 장기 예측 모델로의 지식 이전을 수행합니다. 모델 아키텍처로는 SimVP를 사용하며, CNN 구조를 기반으로 해 공간적 특징을 추출하고, 시간적 진화를 학습합니다. 이 과정에서 강화된 예측 결과는 CasCast라는 확산 기반(diffusion-based) 프레임워크와 통합되어 모호함(blurriness)을 극복하는 데 도움을 줍니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(SEVIR, HKO-7, MeteoNet)에서 SimCast의 성능이 기존 모델들에 비해 현저히 향상된 것으로 나타났습니다. CSI 점수는 SEVIR에서 0.452, HKO-7에서 0.474, MeteoNet에서 0.361을 기록했습니다. 특히 고강도 강수 지역에서의 성능이 더욱 두드러지며, 실용적인 응용에서도 추가적인 계산 부담 없이 장기 예측을 수행할 수 있습니다.



### A Large-scale Dataset for Robust Complex Anime Scene Text Detection (https://arxiv.org/abs/2510.07951)
- **What's New**: 이 논문에서는 기존의 텍스트 탐지 데이터셋들이 자연 및 문서 중심 장면에 초점을 맞추고 있는 반면, 애니메이션 장면에서의 텍스트 탐지 필요를 해결하기 위해 새로운 데이터셋인 AnimeText를 소개합니다. AnimeText는 735K 이미지와 4.2M 개의 주석이 달린 텍스트 블록으로 구성되어 있으며, 계층적 주석과 애니메이션 관련 시나리오에 적합한 하드 네거티브 샘플을 제공합니다. 이 데이터셋은 애니메이션 장면에서의 텍스트 탐지 성능 향상을 위한 기반을 마련하고 있습니다.

- **Technical Details**: AnimeText 데이터셋은 총 4.2M 개의 다국어 텍스트 인스턴스에 대한 주석을 포함하고 있으며, 각 이미지에 평균 5.77 개의 텍스트 인스턴스가 포함되어 있습니다. 이는 기존 데이터셋보다 5배 더 큰 규모로, 더욱 다양한 텍스트 밀도를 커버합니다. 또한, 세 단계의 주석 파이프라인을 통해 효율적인 주석 작업을 가능하게 하여 고품질의 텍스트 구역 주석을 보장합니다.

- **Performance Highlights**: 실험 결과는 AnimeText로 훈련된 모델이 기존 데이터셋으로 훈련된 모델보다 애니메이션 장면 텍스트 탐지 작업에서 우수한 성능을 나타낸다는 것을 보여줍니다. 본 데이터셋은 애니메이션 텍스트 탐지 성과를 개선하는 데 필요한 훈련 데이터셋으로서 중요한 역할을 하며, 커뮤니티에 새로운 도전을 제공하는 테스트 데이터셋으로도 활용될 수 있습니다.



### CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving (https://arxiv.org/abs/2510.07944)
- **What's New**: CVD-STORM은 환경 시뮬레이션과 미래 상태 예측을 위한 새로운 비디오 생성 모델입니다. 이 모델은 4차원(4D) 재구성 기능을 갖춘 다중 시점 비디오 생성에 초점을 맞추고 있습니다. 특히, STORM-VAE라는 변이형 자동 인코더(VAE)를 활용하여 3D 구조 및 시간적 동적 데이터를 효과적으로 표현하고 생성하는 데 도움을 줍니다.

- **Technical Details**: CVD-STORM은 공간-시간적 재구성을 위한 VAE를 활용하여 다중 시점 비디오를 생성합니다. 모델의 첫 단계에서는 VAE를 보조 재구성 작업으로 미세 조정하여 3D 구조를 더 잘 인코딩하도록 합니다. 이후, 이 VAE는 비디오 확산 프로세스에 통합되어 생성 품질을 크게 개선하고, 동적 장면을 효과적으로 재구성할 수 있는 Gaussian Splatting Decoder를 사용합니다.

- **Performance Highlights**: 실험 결과 CVD-STORM은 FID 및 FVD 메트릭에서 현저한 개선을 보여줍니다. 또한, 이를 통해 생성된 긴 시퀀스 비디오는 텍스트, 바운딩 박스 및 고해상도 지도와 같은 다양한 제어 입력에 따라 생성되며, 4D 장면을 직접 재구성하여 종합적인 장면 이해를 향상시킵니다.



### TTOM: Test-Time Optimization and Memorization for Compositional Video Generation (https://arxiv.org/abs/2510.07940)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 비디오 생성 모델의 성능을 향상시키기 위해 Test-Time Optimization and Memorization (TTOM) 프레임워크를 소개합니다. 기존의 방식과는 달리, TTOM은 훈련 없이 spatiotemporal 레이아웃에 맞춘 출력을 제공합니다. 이는 기계 학습 모델이 영상에서 입력된 텍스트를 더 정련하게 이해하고 생성할 수 있도록 돕습니다.

- **Technical Details**: TTOM은 사용자 프롬프트에 기반한 spatiotemporal layout을 생성하고, 이를 통해 비디오 생성 모델의 성능을 최적화합니다. 새로운 매개변수를 도입하여 각 샘플에 맞춰 업데이트하며, 이 과정을 통해 이전 작업의 최적화를 메모리에 저장할 수 있습니다. 이 파라미터는 삽입, 읽기, 업데이트 및 삭제와 같은 다양한 작업을 지원하여 유연하고 효율적인 운영이 가능합니다.

- **Performance Highlights**: T2V-CompBench 및 Vbench 벤치마크에서의 실험 결과는 TTOM이 매우 효과적이고 실용적이며 효율적인 프레임워크임을 입증하였습니다. 특히, TTOM은 CogVideoX-5B와 Wan2.1-14B와 비교했을 때 T2V-CompBench에서 각각 34% 및 14%의 성능 향상을 이뤘습니다. 이는 복합 비디오 생성에서의 크로스 모달 정렬을 현장에서 자동으로 달성할 수 있도록 해줍니다.



### ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection (https://arxiv.org/abs/2510.07927)
- **What's New**: 이 논문에서는 제조 품질 관리 및 헬스케어 모니터링에서 중요한 역할을 하는 이상 탐지(anomaly detection)의 한계를 극복하기 위해 ASBench라는 새로운 벤치마크 프레임워크를 제안합니다. ASBench는 기존 방법들의 성능을 체계적으로 평가하는 것을 목표로 하며, 합성된 이상 샘플의 생성에 필요한 평가 기준을 제공합니다. 주요 특징으로는 데이터셋 간의 일반화 성능, 합성 데이터와 실제 데이터의 비율, 합성 이미지의 품질 지표와 탐지 성능 간의 상관 관계 분석을 포함하고 있습니다.

- **Technical Details**: ASBench는 네 가지 주요 평가 차원을 도입하여 이상 합성(anomaly synthesis) 방법을 평가합니다: (i) 서로 다른 데이터셋 및 파이프라인에 대한 일반화 성능 (ii) 합성 데이터와 실제 데이터의 비율 (iii) 합성 이미지의 내재적 품질 메트릭과 탐지 성능 메트릭 간의 상관 관계, (iv) 혼합 이상 합성 방법에 대한 전략입니다. 이 프레임워크는 다양한 데이터셋과 탐지 모델의 비교를 통해 이상 합성 알고리즘의 성능을 체계적으로 분석합니다.

- **Performance Highlights**: 다양한 실험을 통해 현재 이상 합성 방법의 한계를 식별하였으며, 여러 이상 합성 방법의 조합 사용이 탐지 성능을 향상시킬 수 있다는 중요한 통찰을 제공하였습니다. 연구 결과, 생성된 이상 샘플의 샘플 비율이 탐지 모델의 성능에 큰 영향을 미치지 않으며, 생성된 이미지의 내재적 메트릭과 탐지 성능 간의 상관관계가 없음을 밝혔습니다. 이러한 결과는 향후 연구 방향에 대한 중요한 시사점을 제공합니다.



### MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding (https://arxiv.org/abs/2510.07915)
- **What's New**: 본 논문에서 제안하는 MARC(메모리 보강 강화 학습 기반 토큰 압축)는 비디오 이해의 효율성을 향상시키기 위한 혁신적인 방법론입니다. 기존의 기존 훈련이 필요 없는 방법의 정보 손실 문제를 해결하기 위해, 비주얼 메모리 리트리버(Visual Memory Retriever)와 C-GRPO 프레임워크를 통해 성공적으로 하이프레임 비디오의 데이터를 압축할 수 있는 방법을 제시합니다. 이를 통해 단일 프레임의 토큰만으로도 거의 기존 성능을 유지하면서 95%의 비주얼 토큰 감소와 72%의 GPU 메모리 절감을 달성했습니다.

- **Technical Details**: MARC는 '검색 후 압축(retrieve-then-compress)' 전략에 따라 고유한 메모리 검색 기법을 사용하여 비디오 내의 중요한 클립을 선택하고, 강화 학습 기반 압축 방법인 C-GRPO(Compression Group Relative Policy Optimization)를 통해 표현 압축을 수행합니다. 기존의 GRPO 알고리즘을 기반으로 커스터마이징하여 모델의 추론 능력을 유지하면서도 약 64프레임의 비디오를 하나의 프레임으로 변환할 수 있도록 설계되었습니다. 이 방법은 구조화된 메모리 검색과 RL 기반 압축을 통합하여 비디오 이해의 효율성과 정확성을 동시에 달성합니다.

- **Performance Highlights**: 여섯 개의 비디오 벤치마크를 통한 실험 결과, MARC는 입력 프레임 수를 단 하나로 줄이면서도 기존 64프레임 기준의 성능(42.20 vs. 42.21)과 거의 동일한 결과를 나타냈습니다. 또한, GPU 메모리 사용량을 72% 감소시키고 생성 지연 시간(latency)을 23.9% 낮추는 성과를 올리며, 자원이 제한된 상황에서도 최적의 비디오 이해 과제를 수행할 수 있는 가능성을 제시합니다. 이러한 분석 결과는 MARC의 뛰어난 성능 및 효율성을 뒷받침하고 있으며, 자율주행과 감시 시스템 같은 실제 응용 분야에 적합한 성능을 보여줍니다.



### XYZCylinder: Feedforward Reconstruction for Driving Scenes Based on A Unified Cylinder Lifting Method (https://arxiv.org/abs/2510.07856)
Comments:
          Project page: this https URL

- **What's New**: 최근에는  피드포워드 재구성 패러다임(feedforward reconstruction paradigms)에 대한 관심이 증가하고 있으며, 이는 고정된 뷰 변환을 암묵적으로 학습하여 단일 표현으로 장면을 재구성합니다. 그러나 이러한 방법은 카메라 구성 변화에 따라 일반화 능력(generalization capability)과 재구성 정확도(reconstruction accuracy)가 제한적입니다.

- **Technical Details**: 본 논문에서는 카메라 모델링(camera modeling)과 특징 리프팅(feature lifting)을 포함하는 통합된 실린더 리프팅 방법을 기반으로 한 	extbf{XYZCylinder} 모델을 제안합니다. 특히, 일반화 능력을 개선하기 위해 Unified Cylinder Camera Modeling (UCCM) 전략을 설계하여 다양한 카메라 구성을 조정 가능한 매개변수로 통합합니다.

- **Performance Highlights**: 실험 결과에 따르면, XYZCylinder는 다양한 평가 설정 하에서 최신 성능(state-of-the-art performance)을 달성하며, 제로샷(zero-shot) 방식으로 다른 주행 장면에 대해 일반화할 수 있습니다.



### Self-Supervised Learning Strategies for a Platform to Test the Toxicity of New Chemicals and Materials (https://arxiv.org/abs/2510.07853)
- **What's New**: 이 논문은 자동화된 독성 검사 시스템에서 기계 학습 모델을 통해 독성 물질로 인한 변화를 효과적으로 식별할 수 있는 방법을 시연합니다. 자가 지도 학습(self-supervised learning)을 통해 학습된 표현이 독성 물질에 의한 변화를 확인할 수 있음을 보여주는 개념 증명을 제시합니다. 이를 위해 EmbryoNet 데이터셋을 활용하고, 다양한 화학 화합물에 의한 제브라피시 배아의 표현형을 연구하였습니다. 최종적으로 TOXBOX 프로젝트의 일환으로 물리적 독성 검사 장치에 기계 학습 모델을 통합하는 방안을 논의합니다.

- **Technical Details**: REACH 규제는 EU 시장에 진입하는 화학 화합물을 더 잘 이해하기 위한 목적으로, 특정 화합물의 수량이 1톤을 초과하는 경우 독성 테스트를 수행하고 결과를 European Chemicals Agency (ECHA)에 보고하도록 요구합니다. 기존의 독성 테스트는 일반적으로 비용이 많이 드는 인 비보(in vivo) 방법을 사용하는데, 이 논문에서는 인 비보 연구의 대안으로 제브라피시 배아를 활용한 고처리량 스크리닝(High-Throughput Screening, HTS)에 대한 관심이 증가하고 있음을 강조합니다. 자가 지도 학습을 통해 학습된 연속적 표현은 독성 물질에 의한 변화를 모델링할 수 있으며, 자동 평가 방법을 ML 모델을 통해 시행할 수 있습니다.

- **Performance Highlights**: 이 연구는 자가 지도 학습을 통해 학습한 표현이 화합물의 작용 방식(modes-of-action)을 효과적으로 구별할 수 있음을 입증하였습니다. 특히, 실험 데이터의 고차원성을 처리하는 데 있어 딥 러닝(Deep Learning) 모델이 전통적인 머신 러닝 모델보다 더 적합하다는 점을 강조하고 있습니다. 또한, 제브라피시의 다양한 배아 표현형을 비교하는 데 있어서 이 방법이 유용하다는 것을 보여주었으며, TOXBOX와 같은 실제 독성 검사 장치에 DL 모델을 통합하는 데 따른 도전 과제를 다루고 있습니다.



### AlignGS: Aligning Geometry and Semantics for Robust Indoor Reconstruction from Sparse Views (https://arxiv.org/abs/2510.07839)
- **What's New**: 이 논문에서는 AlignGS라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 최근 3D 모델 생성에서의 기하학적 모호성을 해결하기 위해, semantics(의미)를 단순한 특징이 아닌 geometry(지오메트리)를 정규화하는 능동적인 요소로 활용하는 것을 주장합니다. 이 접근 방식은 geometry와 semantics 간의 상호작용을 최적화하여 높은 품질의 3D 모델을 생성할 수 있도록 합니다.

- **Technical Details**: AlignGS는 2D foundation 모델에서 사전 정보를 추출하여 3D 표현을 정규화하는 새로운 semantics-to-geometry guidance mechanism을 사용합니다. 이 방법은 geometry 최적화를 위한 깊이 일관성과 다면적 노말 정규화 기법을 포함하여, Sparse-view 상황에서도 높은 정확도의 기하학적 재구성을 도와줍니다. 이를 통해 기존의 접근 방식보다 더욱 완전하고 일관된 3D 모델을 생성할 수 있습니다.

- **Performance Highlights**: 논문의 실험 결과는 AlignGS가 novel view synthesis(새로운 뷰 합성)에서 최첨단 결과를 달성하고, 재구성된 모델이 기하학적 정확도가 뛰어남을 보여줍니다. 또한, AlignGS는 ScanNet 및 NRGBD 데이터 세트에서 기존 방법들보다 더 나은 성능을 발휘하며, 제한된 입력 뷰로부터 더 일관되고 완전한 3D 모델을 생성할 수 있음을 입증했습니다.



### IsoSignVid2Aud: Sign Language Video to Audio Conversion without Text Intermediaries (https://arxiv.org/abs/2510.07837)
Comments:
          Accepted in AIML-Systems-2025

- **What's New**: 이번 연구에서는 IsoSignVid2Aud라는 새로운 end-to-end 프레임워크를 제안하여, 의사소통 장애인과 청각 장애인이 서로 소통할 수 있도록 지원합니다. 이 시스템은 연속적인 문법이 없는 신호 언어 비디오를 직접 음성으로 변환하며, 중간 텍스트 표현이 필요하지 않습니다. 이러한 접근 방식은 실시간 통신 혜택을 제공하며, 기존 시스템의 지연(latency) 및 오류를 피하는 이점이 있습니다.

- **Technical Details**: IsoSignVid2Aud는 세 가지 주요 구성 요소로 이루어져 있습니다: I3D 네트워크에 기반한 특징 추출 모듈, 시각적 특징을 음성 표현으로 변환하는 기능 변환기(feature transformer), 및 예측된 스펙트로그램으로부터 자연스러운 음성을 합성하는 오디오 생성 모듈입니다. 이 시스템은 Non-Maximal Suppression(NMS) 알고리즘을 도입하여 비문법적인 연속 시퀀스에서 신호를 효과적으로 탐지합니다. 이 과정은 신호 언어의 표현력을 최대한 보존하면서도 실용적인 응용을 촉진합니다.

- **Performance Highlights**: 실험 결과, IsoSignVid2Aud는 ASL-Citizen-1500 및 WLASL-100 데이터셋에서 각각 72.01% 및 78.67%의 Top-1 정확도를 달성하며 경쟁력 있는 성능을 보여줍니다. 또한, 음질 측정(PESQ: 2.67, STOI: 0.73)에서 알아들을 수 있는 음성 출력을 나타냅니다. 이 연구는 텍스트 없이 신호 언어 비디오에서 직접 음성으로 변환하는 새로운 패러다임을 제시하며, 특히 비신서 및 초급 학습자에게 접근 가능한 통신 기술을 제공합니다.



### PrismGS: Physically-Grounded Anti-Aliasing for High-Fidelity Large-Scale 3D Gaussian Splatting (https://arxiv.org/abs/2510.07830)
- **What's New**: 본 논문에서는 3D Gaussian Splatting (3DGS)의 성능을 향상시키기 위해 PrismGS라는 새로운 정규화 프레임워크를 제안합니다. 기존의 시스템 수준 확장성이나 아키텍처 재설계에 집중한 연구와는 달리, PrismGS는 Gaussian 프리미티브의 본질적인 동작을 개선하여 다중 스케일에서 일관성을 유지하고 기하학적 안정성을 증진합니다. 이 방법은 고해상도(예: 4K) 도시 환경에서 발생하는 알리아싱 문제를 효과적으로 완화합니다.

- **Technical Details**: PrismGS는 피라미드 형태의 다중 스케일 감시 기법과 독립적인 크기 정규화라는 두 가지 상호 보완적인 정규화 방법을 통합하여 작동합니다. 다중 스케일 감시는 각 프리미티브가 다양한 해상도에서 일관성 있게 기여하도록 강제하여 알리아싱을 줄입니다. 또한 고유한 크기 정규화를 통해 3D Gaussian의 크기에 물리적으로 기반이 되는 최저 한계를 설정하여 기하학적 안정성을 확보하고 불안정한 최적화를 방지합니다.

- **Performance Highlights**: MatrixCity, Mill-19, UrbanScene3D와 같은 대규모 데이터셋에 대한 실험을 통해 PrismGS가 CityGaussian 대비 약 1.5 dB의 PSNR 개선을 이루며 우수한 품질을 지속적으로 유지하는 것을 입증하였습니다. 특히 4K 렌더링 조건에서도 뛰어난 성능을 보이며, 기존 방법들과 비교하여 통계적 및 지각적 품질에서 우수한 결과를 나타냅니다. 이를 통해 대규모 3DGS에서의 알리아싱 문제와 기하학적 안정성 문제를 효과적으로 해결했음을 보여줍니다.



### MMHOI: Modeling Complex 3D Multi-Human Multi-Object Interactions (https://arxiv.org/abs/2510.07828)
- **What's New**: 이번 연구에서는 MMHOI라는 대규모 다중 인간-물체 상호작용 데이터셋을 소개합니다. 이 데이터셋은 12개의 일상 시나리오에서 수집된 이미지를 포함하며, 각 인간과 물체에 대한 3D 형상과 포즈 주석을 제공하여 78개의 행동 카테고리와 14개의 상호작용 특정 신체 부위를 라벨링합니다. 새로운 MMHOI-Net 네트워크는 이러한 데이터셋을 기반으로 하여 3D 기하학과 상호작용을 동시에 추정하는 혁신적인 방법을 제안합니다.

- **Technical Details**: MMHOI 데이터셋은 약 60만 프레임으로 구성되어 있으며, 1313명의 참가자가 2222개의 일상적인 물체와 상호작용합니다. 각 상호작용은 78개의 행동 클래스와 1414개의 상호작용 신체 부위 라벨로 주석 처리되었습니다. MMHOI-Net 네트워크는 Vision Transformer (ViT) 아키텍처를 활용하여 인간과 물체의 패치 수준 특징을 추출하고, 구조화된 듀얼 패치 표현 방식을 통해 개체의 특성과 상호작용을 효과적으로 모델링합니다.

- **Performance Highlights**: MMHOI와 CORE4D 데이터셋을 통해 제안된 방법이 기존 접근 방법보다 우수한 성능을 보임을 실험 결과에서 확인했습니다. 특히, 우리의 방법은 복잡한 다중 인간-물체 상호작용 모델링에서 상태-of-아트 결과를 달성하며 인간과 물체, 및 그 상호작용의 재구성 품질이 뛰어납니다. 또한, action recognition을 통해 3D 재구성을 향상시키는 방법론이 효과적임을 보여줍니다.



### Enhancing Visual Prompting through Expanded Transformation Space and Overfitting Mitigation (https://arxiv.org/abs/2510.07823)
Comments:
          Accepted to NeurIPS2025

- **What's New**: 이번 논문에서는 Visual prompting (VP)의 한계점을 극복하기 위해 ACAVP (Affine, Color, and Additive Visual Prompting)이라는 새로운 방법을 제안합니다. ACAVP는 기존 VP 방법의 단점을 해결하며, 추가적인 변환 연산을 통해 VP의 표현력을 강화합니다. 이를 통해 높아진 정확도와 더불어 컴퓨팅 오버헤드를 최소화할 수 있습니다.

- **Technical Details**: ACAVP는 affine transformation과 color transformation 두 가지를 도입하여 VP의 표현력을 높입니다. Affine transformation은 이미지 정보의 손실 없이 특정 작업에 맞는 프롬프트 영역을 생성하는 반면, color transformation은 이미지의 밝기와 대비를 조정하여 작업과 관련된 시각적 특징을 강조합니다. 이러한 두 가지 방법은 모두 VP 기술의 이점을 유지하면서도 계산 비용은 최소화합니다.

- **Performance Highlights**: ACAVP는 12개의 다양한 이미지 분류 데이터셋에서 정확도를 평가한 결과, 기존 VP 방법보다 높은 평균 정확도를 기록했으며, linear probing보다도 뛰어난 성능을 발휘했습니다. 또한, ACAVP는 데이터 분포의 변화에 대해 우수한 강건성을 보이며, inference 시 거의 부하가 없는 것으로 나타났습니다. 이는 ACAVP가 기존의 VP 방법에 비해 우수한 일반화 능력을 제공한다는 것을 보여줍니다.



### An End-to-End Room Geometry Constrained Depth Estimation Framework for Indoor Panorama Images (https://arxiv.org/abs/2510.07817)
- **What's New**: 본 논문은 실내 파노라마 이미지에서 단일 눈의 정보로 구형 픽셀 깊이를 예측하는 새로운 프레임워크인 RGCNet을 제안합니다. 기존의 방법들은 픽셀 수준의 정확성에 집중하여, 방의 모서리에서 과도하게 매끄럽거나 노이즈에 민감한 결과를 초래했습니다. RGCNet은 다중 작업 학습(multi-task learning)을 기반으로 하여 깊이 추정, 방 레이아웃 추정 및 배경 세분화를 통합하여 보다 정교한 깊이 예측을 가능하게 합니다.

- **Technical Details**: RGCNet 구조는 공유되는 특성 인코더(shared feature encoder)와 각기 다른 작업에 특화된 디코더(task-specific decoders)로 구성되어 있습니다. 이 모델은 방의 구조적 기하 정보를 활용하여 보다 정확한 깊이 지도를 생성하는데 중점을 둡니다. 또한, 방 기하학에 기반한 배경 깊이 해결 전략과 배경 세분화를 통한 융합 메커니즘을 포함하여, 초기 깊이 예측을 개선합니다.

- **Performance Highlights**: 다양한 데이터셋인 Stanford2D3D, Matterport3D 및 Structured3D에서 실험한 결과, 제안된 RGCNet 방법이 기존의 오픈 소스 방법들보다 현저히 우수한 성능을 달성했습니다. 이 연구는 깊이 추정의 정확성을 개선하고 실내 환경의 복잡성을 효과적으로 다룰 수 있는 새로운 접근 방식을 제시합니다.



### FMANet: A Novel Dual-Phase Optical Flow Approach with Fusion Motion Attention Network for Robust Micro-expression Recognition (https://arxiv.org/abs/2510.07810)
- **What's New**: 본 논문은 미세 표정 인식(micro-expression recognition, MER)의 새로운 접근 방식을 제안합니다. 특히, Magnitude-Modulated Combined Optical Flow (MM-COF)라는 포괄적인 모션 표현 방식을 도입하여, 미세 표정의 시작-정점(onset-apex)과 정점-종료(apex-offset) 모든 단계의 동역학을 통합합니다. 또한, FMANet이라는 새로운 신경망 아키텍처를 통해 이중 단계 분석을 학습 가능한 모듈로 내부화하여, 동작 단서를 적응적으로 융합하고 분류를 위한 주목할만한 얼굴 영역에 집중할 수 있도록 합니다.

- **Technical Details**: MM-COF는 두 가지 상호 보완적 파이프라인을 기반으로 하여, 시작-정점 및 정점-종료 간의 optical flow를 결합하고 모션의 세기를 조절하여 나타냅니다. 이 접근 방식은 MM-COF를 통해 더 판별적인 optical flow 표현을 설계하는 기여를 포함합니다. FMANet 모델은 MM-COF의 아이디어를 이중 스트림 네트워크로 확장하여, FFB 합의 모듈과 SMAB 주의 블록을 통해 두 모션 단계를 통합하며, 마지막으로 경량 CNN을 통해 분류를 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 MM-COF 표현 방식과 FMANet이 MMEW, SMIC, CASME-II, SAMM 데이터셋에서 기존 방법들보다 성능이 우수함을 입증했습니다. 또한, 각 기여에 대해 종합적인 ablation 연구를 진행하였으며, 각 요소가 현대 MER 연구의 전반적인 발전에 기여하고 있음을 보장합니다. 본 연구는 제한된 데이터 조건에서도 효과적으로 직관적인 감정 표현을 학습할 수 있는 잠재력을 보여줍니다.



### GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models (https://arxiv.org/abs/2510.07791)
Comments:
          20 pages, 13 figures

- **What's New**: 최근 Visual-Language Models (VLMs)의 공간-시간 지능이 자율 주행 및 임베디드 AI 등 다양한 분야에서 중요성이 대두되면서 많은 주목을 받고 있습니다. 기존의 공간-시간 벤치마크들은 이미지/비디오 맥락에서의 자아 중심적(egocentric) 관점 추리에만 집중하였고, 그래픽스(context) 관점(예: 지도)에서는 지리적(spatial-temporal) 추리를 평가하지 못했습니다. 이러한 문제를 해결하기 위해 본 논문에서는 대규모 카메라 네트워크에서의 이동 목표를 위한 새로운 도전 과제로 Geo-Temporal Reasoning benchmark(GTR-Bench)를 제안합니다.

- **Technical Details**: GTR-Bench는 다수의 카메라 뷰를 활용하여 지도와 비디오 간의 관점을 전환하고, 서로 겹치지 않는 여러 비디오를 통한 공동 추리와 비디오 관점에서 관찰되지 않는 공간-시간 지역에 대한 추리를 요구합니다. 본 연구에서는 실제 도시 환경에서의 보행자와 차량의 궤적 데이터를 기반으로 하는 여러 태스크를 개발하여, VLMs의 공간-시간 지능을 평가할 수 있도록 합니다. GTR-Bench는 420개의 질문을 포함하여 기본(reasoning) 태스크와 조합(combinatorial) 태스크의 여러 수준으로 평가합니다.

- **Performance Highlights**: GTR-Bench에서 모델의 성능은 기존의 공간-시간 벤치마크에 비해 현저히 낮으며, 최고 성능 모델인 Gemini-2.5-Pro조차도 34.9%의 정확도로 인간 성능(78.61%)에 비해 크게 뒤처졌습니다. 이러한 부진한 성능은 VLMs의 세 가지 주요 결함, 즉 공간-시간 맥락의 불균형한 활용, 약한 시간 예측 능력, 지도 데이터와 다중 뷰 비디오 입력 간의 이해력 부족에 기인합니다. GTR-Bench는 VLMs의 공간-시간 지능 연구 및 응용의 새로운 기회를 제공합니다.



### Demystifying Deep Learning-based Brain Tumor Segmentation with 3D UNets and Explainable AI (XAI): A Comparative Analysis (https://arxiv.org/abs/2510.07785)
- **What's New**: 이번 연구에서는 MRI 이미지에서 뇌종양 세분화의 정확도를 높이기 위해 설명 가능한 인공지능(Explainable Artificial Intelligence, XAI)의 활용을 조사하였습니다. 연구의 초점은 UNet 모델을 이용한 뇌종양 세분화와 Grad-CAM(Gradient-weighted Class Activation Mapping) 및 주의(attention)-기반 시각화 기술을 사용하여 모델의 이해도를 향상시키는 것이었습니다. 최종 결과로, ResUNet가 다른 모델들보다 뛰어난 성능을 보였으며, 이를 통해 자동화된 뇌종양 세분화를 위한 ResUNet의 활용을 추천합니다.

- **Technical Details**: 연구에서는 UNet, Residual UNet(ResUNet), Attention UNet(AttUNet)의 3가지 딥러닝 모델을 평가하여 가장 우수한 모델을 식별했습니다. 각 모델은 Adam 최적화를 사용하여 훈련 및 검증되었으며, 훈련 시간, 유효성 검사 시간, 추론 시간, 세분화 유사도 계수 및 손실 함수와 같은 성과를 평가하였습니다. Grad-CAM은 각 UNet 모델이 집중하는 종양 하위 영역에 대한 시각적 통찰력을 제공하였고, 주의 기반 시각화는 AttUNet의 주의 모듈 작동 메커니즘에 대한 귀중한 통찰을 제공하였습니다.

- **Performance Highlights**: 최종 테스트 단계에서 ResUNet은 Dice 점수, Jaccard 유사도 점수, 정확도, 재현율 및 F1 점수에서 다른 모델보다 우수한 성능을 보였습니다. 연구 결과는 ResUNet이 가장 좋은 성능을 보임을 나타내며, 향후 임상 평가에서 자동화된 뇌종양 세분화를 위해 이 모델의 사용이 권장됩니다. 본 연구의 소스코드와 체크포인트는 제공된 링크에서 확인할 수 있습니다.



### DEGS: Deformable Event-based 3D Gaussian Splatting from RGB and Event Stream (https://arxiv.org/abs/2510.07752)
Comments:
          Accepted by TVCG

- **What's New**: 본 논문은 저프레임 RGB 비디오에서 동적 3D Gaussian Splatting (3DGS)을 복원하기 위한 새로운 프레임워크를 제안합니다. 다수의 픽셀이 큰 프레임 간 움직임으로 인해 불확실성이 높아지는 문제를 해결하기 위해, 이벤트 카메라에서 캡처한 고프레임 이벤트 스트림을 활용하여 저해상도의 RGB 이미지와 결합합니다. 이벤트 데이터의 도움으로 동적 3DGS를 효과적으로 최적화할 수 있는 방법을 침구하며, 이벤트 궤적을 통해 변형 필드를 안내합니다.

- **Technical Details**: 본 연구에서는 고유한 파라미터를 갖는 3D Gaussian 점들을 이러한 3DGS로 정의합니다. 이점은 변형 필드에 따라 특정 시점에서의 각 Gaussian 포인트의 위치와 회전, 스케일 매개변수를 변화시킬 수 있도록 확장됩니다. 이러한 방법론은 기존의 정적인 3DGS 모델을 발전시켜 동적인 장면을 최적화할 수 있도록 합니다. 또한 이벤트 카메라의 데이터를 이용하여 연속적인 모니터 캡처를 통해 동적 장면의 복잡함을 해결할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존의 이미지 및 이벤트 기반 접근법과 비교하여 합성 및 실제 장면 모두에서 뛰어난 성능을 보였습니다. 특히 이벤트 데이터의 활용이 동적 3DGS의 최적화에 긍정적인 영향을 미쳤음을 입증하였습니다. 이 연구는 동적 시나리오 복원에서의 적용 가능성을 높이고, 기존의 정적 모델에 비해 실시간 렌더링 및 높은 재구성 품질을 제공합니다.



### UltraLED: Learning to See Everything in Ultra-High Dynamic Range Scenes (https://arxiv.org/abs/2510.07741)
- **What's New**: 이 논문에서는 Ultra-high dynamic range (UHDR) 장면을 단일 짧은 노출 RAW 이미지만을 사용하여 재구성하는 새로운 방법을 제안합니다. 이 방법은 기존의 RGB 기반 방식들이 겪는 정렬 및 고스트 현상을 피할 수 있습니다. 연구팀은 두 단계로 구성된 UltraLED 프레임워크를 도입하여, 첫 번째로 비율 맵을 통해 노출을 보정하고, 두 번째로 밝기 인식 RAW 노이즈 제거기를 통해 어두운 영역의 세부 정보를 향상시킵니다.

- **Technical Details**: UHDR 재구성을 위한 핵심 도전 과제는 노이즈 제거(denoising)와 어두운 영역 정보 복원에 있습니다. RAW 이미지의 높은 비트 깊이(14 bits)와 예측 가능한 노이즈 특성 덕분에 이러한 과제를 해결하는 데 필요한 가능성이 높아집니다. 본 연구는 노이즈 특성이 밝기와 함께 변한다는 점을 인식하고, 이를 처리하기 위해 비율 맵을 활용하여 밝기 정보를 인코딩하는 새로운 노이즈 모델을 도입하였습니다.

- **Performance Highlights**: UltraLED는 기존의 단일 프레임 접근 방식에 비해 현저하게 향상된 성능을 보였습니다. 다수의 실험 결과, 이 방법이 동적 장면에서 어두운 영역의 세부 정보를 효과적으로 복구하는 데 뛰어난 능력을 보여주었습니다. 이러한 연구 결과는 공공 데이터 세트와 코드가 제공되어 후속 연구에 기여할 수 있는 기반을 마련합니다.



### ComGS: Efficient 3D Object-Scene Composition via Surface Octahedral Probes (https://arxiv.org/abs/2510.07729)
- **What's New**: 본 논문에서는 Gaussian Splatting (GS)을 이용한 사실적인 3D 객체-장면(composition) 구성의 한계를 다룹니다. 기존의 Gaussian 기반 역 렌더링 방법들은 ray tracing에 의존하여 저조한 효율성을 나타내며, 이 문제를 해결하기 위해 Surface Octahedral Probes (SOP)를 도입하여 재조명 가능한(relightable) 객체 재구성을 가능하게 합니다. 또한, 환경 조명 추정에 집중하여 복잡한 장면에서도 시각적 일관성을 유지합니다.

- **Technical Details**: 효율적인 재조명 가능 객체 재구성을 위한 SOP는 간접 조명 및 가림 정보(occlusion information)를 저장하여, 객체 주변의 조명을 추정하는 데 필요한 계산을 최소화합니다. SOP의 도입으로 재구성이 기존 방법보다 2배 이상 빨라지며, 28 FPS의 실시간 렌더링을 가능하게 합니다. 또한 Diffusion Model을 사용하여 부분적으로 복원된 Gaussian 방사장(radiance fields)에서 환경 맵을 추론하여 조명 추정의 품질을 개선합니다.

- **Performance Highlights**: ComGS는 재조명 가능 객체와 추정된 조명을 결합하여 시각적인 일관성과 생동감 있는 그림자를 생성하는 3단계의 작업으로 구성됩니다. 우리의 방법은 SynCom 및 공개 데이터셋에서 +2dB의 PSNR을 달성하였고, 3D 일관성이 40% 향상되었으며, 기존 방법들에 비해 67% 더 높은 조화(harmony)를 보여줍니다. 따라서, 이 프레임워크는 몰입형 3D 응용 프로그램에 큰 잠재력을 지니고 있습니다.



### SyncHuman: Synchronizing 2D and 3D Generative Models for Single-view Human Reconstruction (https://arxiv.org/abs/2510.07723)
Comments:
          NIPS 2025

- **What's New**: 이번 연구에서는 SyncHuman이라는 혁신적인 프레임워크를 제안합니다. 이는 2D 멀티뷰 생성 모델과 3D 네이티브 생성 모델을 최초로 통합하여 단일 이미지에서 고품질의 의상을 입은 인간 메쉬 재구성을 가능하게 합니다. 기존 모델들이 피사체의 어려운 포즈와 세부 사항 재구성에서의 한계가 있었던 것에 비해, SyncHuman은 이러한 문제를 해결하고 뛰어난 성능을 보여줍니다.

- **Technical Details**: SyncHuman은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째 구성 요소는 2D 멀티뷰 생성 분기와 3D 희소 구조 생성 분기로 구성된 통합 2D-3D 크로스 스페이스 생성 모델입니다. 두 번째는 픽셀 정렬된 정보를 효과적으로 활용하는 멀티뷰 가이드 디코더로, 이를 통해 3D 생성 과정에서 높은 품질의 3D 메쉬를 얻을 수 있습니다.

- **Performance Highlights**: SyncHuman은 여러 데이터셋에서 실험한 결과 기존의 단일 뷰 인간 재구성 방법들과 비교하여 뛰어난 기하학적 정확도와 시각적 충실도를 달성했습니다. 또한, 대규모 3D 생성 모델과 비교했을 때도 향상된 질감 품질과 세부 사항을 보여주어, 향후 3D 생성 모델 개발에 중요한 가능성을 제공합니다.



### RePainter: Empowering E-commerce Object Removal via Spatial-matting Reinforcement Learning (https://arxiv.org/abs/2510.07721)
- **What's New**: 이번 논문에서는 Repainter라는 새로운 강화 학습( Reinforcement Learning ) 프레임워크를 제안합니다. 이 프레임워크는 e-commerce 이미지의 고충실도(inpainting)를 위한 공간 매팅(scipatial-matting) 궤적 정제를 통합하여 잘못된 객체 제거 및 배경 문맥을 강조하는 방식으로 성과를 향상시킵니다. 또한, EcomPaint-100K라는 대규모 고품질 데이터셋과 EcomPaint-Bench라는 표준화된 벤치마크도 소개합니다.

- **Technical Details**: Repainter는 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)을 기반으로 하는 새로운 이미지 인페인팅 방법으로, 배경 문맥을 우선시하여 샘플링 궤적을 최적화하는 것을 목표로 합니다. 이 방법은 글로벌 구조의 일관성과 지역 재구성의 정확성을 동시에 최적화하는 복합 보상 메커니즘을 도입하여 보상 해킹(reward hacking)의 위험을 줄입니다. 이러한 기술적 접근은 훈련의 수렴 속도를 크게 향상시켜 전체 성능을 높입니다.

- **Performance Highlights**: 광범위한 실험을 통해 Repainter는 여러 주요 지표에서 최신 기술(state-of-the-art) 방법들을 능가함을 입증했습니다. 특히 복잡한 장면에서 공통 아티팩트(artifacts)인 텍스트 환각(hallucination) 및 색상 불 일치를 효과적으로 억제하여, 사용자 경험을 향상시킬 수 있는 깔끔하고 전문적인 상품 이미지를 생성할 수 있습니다. 이 연구는 GRPO를 활용한 이미지 인페인팅의 첫 번째 사례로, e-commerce 신뢰성을 높이는 데 중요한 기여를 합니다.



### Mutual Learning for Hashing: Unlocking Strong Hash Functions from Weak Supervision (https://arxiv.org/abs/2510.07703)
- **What's New**: 새로운 연구는 Mutual Learning for Hashing (MLH)라는 프레임워크를 제안하며, 약한 pairwise 기반 브랜치에서 지식을 전이하여 강한 center 기반 브랜치를 강화합니다. 두 개의 브랜치가 협력적으로 학습하고 상호작용하여 지역 및 전역 데이터를 모두 활용하는 점이 특징입니다. 이를 통해 MLH는 기존의 해시 방법들이 놓친 잠재력을 극대화하여 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: MLH 구조는 강한 center 기반 브랜치와 약한 pairwise 기반 브랜치로 구성됩니다. 각 브랜치는 Mixture-of-Hash-Experts (MoH) 모듈을 공유하여 상호 간의 정보 교환을 촉진하고 있습니다. 최적화 과정에서 pairwise 손실 함수와 center 기반 손실 함수를 함께 사용하여 이 두 통제를 동시에 달성하는 하이브리드 손실 함수를 통해 조율됩니다.

- **Performance Highlights**: MLH는 다양한 데이터셋에서 기존의 최첨단 해싱 방법보다 더 뛰어난 성능을 보여줍니다. 실험 결과에 따르면 MLH는 전역적 구조 인식을 통해 큰 스케일에서의 이미지 검색 정확성을 향상시키는 데 효과적입니다. 게다가, 약한 브랜치의 지식을 통해 성능을 더욱 강화하며, 전체 해싱 성능을 개선합니다.



### Hybrid CNN-BYOL Approach for Fault Detection in Induction Motors Using Thermal Images (https://arxiv.org/abs/2510.07692)
- **What's New**: 본 논문에서는 인덕션 모터(Induction Motors, IM)의 열 이미지를 분류하여 결함을 탐지하는 하이브리드 방법을 제안합니다. BYOL(Bootstrap Your Own Latent)과 CNN(Convolutional Neural Networks)을 통합하여 결함을 조기에 감지하고 모터의 수명을 연장하는 데 중점을 두고 있습니다. 제안된 BYOL-IMNet 모델은 고성능이면서도 경량이며, 다양한 심층 학습 모델을 활용하여 결함 탐지의 정확성을 향상시킵니다.

- **Technical Details**: 사용된 데이터셋은 정규 운영, 과부하 등 다양한 상태에서 수집된 6400장의 열 이미지를 포함하고 있으며, 8가지 유형의 전단 결함과 결합된 고장 사례를 포함하고 있습니다. 모델은 ReLU와 Softmax 활성화 함수를 적용하여 CNN 아키텍처를 기반으로 훈련되며, Adam 옵티마이저를 사용하여 손실 함수를 최소화합니다. 데이터 증강 기법을 통해 과적합을 방지하고 모델의 일반화 능력을 향상시키기 위해 다양한 변환을 진행했습니다.

- **Performance Highlights**: BYOL-IMNet 모델은 99.89%의 테스트 정확도를 기록하며, 이미지당 5.7 ms의 추론 시간을 달성하여 기존의 최신 모델들을 능가했습니다. 이러한 결과는 BYOL과 CNN의 혼합 방법이 인덕션 모터의 결함 탐지 정확성을 높이는 데 유망하다는 것을 시사합니다. 특히, 이 연구는 산업 환경에서의 실시간 모니터링을 위한 강력한 방법론을 제공합니다.



### Controllable Video Synthesis via Variational Inferenc (https://arxiv.org/abs/2510.07670)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 사용자의 다양한 제어를 반영한 비디오 합성(method for video synthesis) 방법을 제안합니다. 기존의 비디오 생성 모델들은 고정된 입력 형식에 맞춰졌지만, 본 연구는 더 다양한 입력 방식을 수용하는 시스템을 개발합니다. 이를 통해 고해상도 장면과 물리적 일관성(physical consistency)을 유지하는 동시에 사용자가 원하는 요소에 대한 높은 제어력을 제공합니다.

- **Technical Details**: 연구에서는 변분추론(variational inference)을 기반으로 다수의 비디오 생성 백본(backbone)을 활용하여 모든 작업 제약을 집합적으로 반영하는 방법을 사용합니다. 문제 최적화는 단계별 KL 발산(KL divergence) 최소화를 통해 해결하며, 컨텍스트 조건부 팩토리제이션(context-conditioned factorization) 기술을 도입하여 지역적 최적 최소화를 피합니다. 최종 목표 분포에 도달하기까지 불리언 샘플을 생성하면서도 샘플의 다양성을 유지합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존의 비디오 생성 작업들과 비교하여 시각적 충실도(visual fidelity), 출력의 다양성(output diversity), 그리고 장면의 일관성(scene consistency)이 개선된 결과를 보였습니다. 이 연구는 길고 복잡한 비디오 생성 상황에서 특히 뛰어난 성능을 발휘함을 확인하였습니다. 새로운 제어 인터페이스를 통해 사용자는 비디오 생성 과정에서 더 많은 유연성과 자유도를 가지게 되었습니다.



### TCIP: Threshold-Controlled Iterative Pyramid Network for Deformable Medical Image Registration (https://arxiv.org/abs/2510.07666)
- **What's New**: 이 논문에서는 기하학적 의료 이미지 정합에서 피라미드 네트워크(pyramid networks)의 성능을 개선하기 위해 Feature-Enhanced Residual Module (FERM)와 Threshold-Controlled Iterative (TCI) 전략을 제안합니다. FERM은 각 디코딩 레이어에서 핵심 구성 요소로, 해부학적 의미 특징을 추출하고 관련 없는 특징을 억제하여 등록의 정확성을 높입니다. 또한 TCI 전략은 각 이미지 쌍에 적응적으로 최적화 반복 횟수를 결정하여 등록 성능을 극대화합니다.

- **Technical Details**: FERM은 3개의 연속 블록, 즉 Feature Fusion Block (FFB), Squeeze Excitation Block (SEB), Deformation Field Estimator (DeF)로 구성됩니다. 이러한 블록들은 각각 해부학적 특징을 추출하고, 불필요한 특징을 억제하며, 최종 변형 필드를 추정합니다. TCI 전략은 두 단계로 이루어져 있으며, 첫 번째 단계에서는 등록 안정성을 평가하고, 두 번째 단계에서 수렴성을 평가하여 각 이미지 쌍의 반복 횟수를 동적으로 결정합니다.

- **Performance Highlights**: 본 연구는 세 개의 공개된 뇌 MRI 데이터셋과 하나의 복부 CT 데이터셋에서 수행된 실험을 통해 TCIP가 기존의 최첨단 등록 네트워크보다 높은 정확도를 보여주며, 유사한 추론 속도와 모델 파라미터 크기를 유지함을 증명합니다. 또한 FERM과 TCI의 일반화 가능성을 평가하기 위해 기존 등록 네트워크와 통합하여 추가적인 실험을 진행하였고, 두 제안 방법의 효과성을 검증하는 탈락 연구를 수행하였습니다.



### Automatic Text Box Placement for Supporting Typographic Design (https://arxiv.org/abs/2510.07665)
- **What's New**: 이 연구는 광고 및 웹 페이지의 레이아웃 디자인에서 자동 텍스트 박스 배치의 효율성을 조사합니다. 표준 Transformer 기반 방법, 소형 Vision and Language Model (Phi3.5-vision), 대형 사전 훈련된 VLM(Gemini), 여러 이미지를 처리하는 연장된 Transformer를 비교합니다. 평가 결과 표준 Transformer 모델이 일반적으로 다양한 요소 간의 관계를 고려하여 VLM 기반 접근 방식보다 우수한 성능을 보였습니다.

- **Technical Details**: 이 연구에서는 텍스트 박스의 최적 배치를 추정하기 위해 세 가지 방법론을 구현하고 평가합니다. 첫째, 표준 Transformer 기반 방법은 텍스트 배치 작업을 회귀 문제로 훈련시킵니다. 둘째, Phi3.5-vision을 활용한 소형 VLM 방법은 전역 이미지를 하나만 사용하여 배치를 추정합니다. 셋째, 대형 VLM인 Gemini는 추가 훈련 없이 배치 추정이 가능함을 조사합니다.

- **Performance Highlights**: Crello 데이터셋을 활용한 평가 결과, 표준 Transformer 기반 모델이 다양한 배경 정보가 포함될 때 더욱 우수한 성능을 보임을 확인했습니다. 그러나 모든 방법이 작은 텍스트나 밀집된 레이아웃에서 어려움을 겪는 것으로 나타났습니다. 이 연구는 임무 전용 아키텍처의 장점을 강조하며, 자동 레이아웃 디자인 개선을 위한 새로운 방향을 제시합니다.



### MONKEY: Masking ON KEY-Value Activation Adapter for Personalization (https://arxiv.org/abs/2510.07656)
- **What's New**: 본 논문에서는 개인화된 확산 모델을 통해 사용자가 특정 인물을 부각시키면서 텍스트 프롬프트를 반영한 이미지를 생성할 수 있는 방법을 제안합니다. IP-Adapter를 활용하여 주어진 이미지에서 주제를 배경과 분리하는 자동 마스크를 생성하고, 이를 두 번째 단계에서 사용하여 생성된 이미지의 주제에만 집중할 수 있도록 합니다. 결과적으로 텍스트 프롬프트가 이미지의 배경에도 영향을 미칠 수 있도록 하여 명확한 주제 표현과 텍스트 정렬을 동시에 달성합니다.

- **Technical Details**: 전통적인 개인화 방법들은 대개 테스트 타임 튜닝(test-time fine tuning) 또는 어댑터 기반(adapter-based) 방법으로 나뉘며, 이 논문에서는 IP-Adapter라는 어댑터 기반 방법을 중심으로 다룹니다. IP-Adapter는 각 레이어별로 이미지를 효과적으로 임베딩하는데, 이는 텍스트와 이미지 토큰이 서로 주의(attention)를 기울이게 하는 구조를 가지고 있습니다. 제안된 MONKEY Adapter는 새로운 파라미터 학습 없이 주체를 강조하고, 텍스트 프롬프트에 대한 반응성을 높은 수준으로 향상시킵니다.

- **Performance Highlights**: 실험 결과, MONKEY Adapter는 무료 학습(building without training) 방식의 몇 가지 다른 방법들과 비교했을 때 프롬프트와 소스 이미지의 정렬이 매우 우수한 성능을 보였습니다. 특히, 주제와 배경 간의 정의로운 구분이 이루어져 텍스트 프롬프트에 대한 정확한 반응을 생성했습니다. 이 방법은 실질적으로 다양한 실험 세트에서 경쟁력 있는 결과를 내어 기존 개인화 방법에 비해 우수한 성능 향상을 보여주었습니다.



### Once Is Enough: Lightweight DiT-Based Video Virtual Try-On via One-Time Garment Appearance Injection (https://arxiv.org/abs/2510.07654)
Comments:
          5 pages (including references), 4 figures. Code and models will be released upon publication

- **What's New**: 이 논문에서는 OIE (Once is Enough)라는 새로운 비디오 가상 착용 기술을 제안합니다. 기존의 두 가지 브랜치를 가진 아키텍처 대신 첫 번째 프레임에서의 의상 교체에 초점을 맞추어, 영상 생성의 효율성을 높입니다. 이 방법은 과도한 매개변수 수를 줄이면서도 비디오의 시각적 일관성을 유지할 수 있는 가능성을 보여줍니다.

- **Technical Details**: OIE는 기존의 다중 분기 아키텍처 대신 단일 분기 아키텍처를 채택하여 가벼운 구조를 가지고 있습니다. 첫 번째 프레임을 편집하는 데 이미 학습된 이미지 기반 모델을 사용하고, 이후에 생성된 프레임은 시간적 정보를 통해 순차적으로 생성됩니다. 이 과정은 저순위 적응(LoRA) 기법을 통해 더욱 최적화되어 있습니다.

- **Performance Highlights**: 실험 결과, OIE는 적은 계산 비용으로도 저주파수의 효율성을 달성하며, 성능 면에서 우수한 결과를 보여줍니다. 모델의 메모리 요구 사항과 계산 자원 소비가 크게 줄어들어 실제 응용에서 실행 가능성을 높였습니다. 또한, 종합적인 질적 및 양적 실험을 통해 제안된 방법의 성능이 검증되었습니다.



### Dual-Stream Alignment for Action Segmentation (https://arxiv.org/abs/2510.07652)
Comments:
          Journal Submission

- **What's New**: 이 논문은 액션 세분화(action segmentation) 분야에서 신규로 제안된 이중 스트림 정렬 네트워크(DSA Net)를 소개합니다. DSA Net은 영상 스트림에서 특정 액션이 발생하는 시점과 위치를 식별하기 위해 두 개의 스트림(프레임 기반과 액션 기반)을 이용합니다. 그 중 Temporal Context (TC) 블록을 사용하여 서로의 정보를 집약하고, 서로 다른 스트림 간의 통신을 효과적으로 제어합니다. 또한, 이 연구는 하이브리드 양자-고전적 기계 학습 프레임워크를 액션 세분화 문제에 처음으로 도입한 것입니다.

- **Technical Details**: DSA Net은 액션 세분화를 위한 두 개의 데이터 스트림을 활용하여 긴 시간 동안의 액션을 세분화합니다. 이들은 프레임 기반 특징과 액션 기반 특징 모두를 포함하며, 각 스트림 간의 정렬을 유도하는 이중 스트림 정렬 손실(Dual-Stream Alignment Loss)을 도입합니다. 이 손실은 관계적 일관성(relational consistency), 크로스 레벨 대조(cross-level contrastive), 사이클 일관성 재구성(cycle-consistency reconstruction) 세 가지 요소로 구성되어 서로 보완적인 정보를 효과적으로 학습하도록 합니다. 또한, 양자 기반의特徴 변조(Quantum-based modulation)와 교차 주의(cross-attention) 기술을 통해 각 스트림의 정보 공유를 개선합니다.

- **Performance Highlights**: DSA Net은 GTEA, Breakfast, 50Salads, EgoProcel과 같은 여러 벤치마크 데이터셋에서 검증되었으며, 기존 방식들보다 월등한 성능을 기록했습니다. 각 구성 요소의 효과를 평가하기 위한 상세한 절단 시험(ablation studies) 결과로 DSA Net의 성능이 강화되었음을 입증하였습니다. 이 연구는 액션 세분화 분야에서 새로운 가능성을 모색하며, 액션에 대한 깊은 이해를 도울 수 있는 기초를 제공합니다.



### PIT-QMM: A Large Multimodal Model For No-Reference Point Cloud Quality Assessmen (https://arxiv.org/abs/2510.07636)
Comments:
          Oral presentation at ICIP 2025

- **What's New**: 이 논문에서는 대규모 멀티모달 모델(LMM)인 PIT-QMM(Point-Image-Text Quality Multimodal Model)을 제안하여 참고 없이 포인트 클라우드 품질 평가(No-Reference Point Cloud Quality Assessment, NR-PCQA)를 가능하게 합니다. 이 모델은 텍스트, 이미지 및 3D 포인트 클라우드를 통합적으로 분석하여 품질 점수를 예측합니다. 또한 본 연구는 이전 상태의 최고 성능(SOTA) 모델들과 비교하여 훨씬 적은 훈련 횟수로도 우수한 성능을 보여줍니다.

- **Technical Details**: PIT-QMM 모델은 포인트 클라우드의 로컬 변화를 포착하는 포인트 클라우드 패치, 글로벌 관점을 제공하는 이미지 프로젝션, 심리측정적 맥락을 추가하는 텍스트 입력을 활용하여 모달리티 간의 상호작용을 최적화합니다. 이 모델은 포인트 클라우드의 크기에 대응하기 위해 샘플링 기법을 포함하여 병렬적으로 점프 한 샘플(patch)을 처리하는 두 단계의 훈련 전략을 채택하고 있습니다. 최종 입력 형식은 포인트 클라우드 데이터, 이미지 프로젝션 및 텍스트를 포함한 멀티모달 질문-응답 구조입니다.

- **Performance Highlights**: PIT-QMM은 다양한 벤치마크에서 최신 연구 결과를 종합적으로 수행하여 이전 방법들보다 현저하게 우수한 성능을 달성하였습니다. 특히 품질 문제의 국소화를 가능하게 하여 모델의 해석가능성을 높이고 유용성을 강화합니다. 이를 통해 포인트 클라우드 영역에서 품질 국소화에 대한 첫 번째 탐색이 이루어졌습니다.



### Rectified-CFG++ for Flow Based Models (https://arxiv.org/abs/2510.07631)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 본 논문에서는 Rectified-CFG++라는 새로운 가이던스 방법론을 소개하고 있습니다. 이 방법은 정류 흐름 모델을 위해 특별히 설계되었으며, 조건부 생성 품질을 개선하기 위해 정렬된 경로를 유지합니다. Rectified-CFG++는 두 개의 속도 필드 사이에서 보간(interpolation)을 적용하여 안정성과 효율성을 극대화합니다.

- **Technical Details**: Rectified-CFG++는 정류 흐름 모델에서 조건부 흐름 맞춤(condition flow-matching)과 분류기 없는 가이던스(classifier-free guidance)를 결합한 것입니다. 각 추론 단계에서 샘플은 조건부 정류 흐름 필드를 따라 이동하며, 이전 샘플에 따라 조건부 및 비조건부 속도 필드 사이에서 가중치 기반의 보정을 적용합니다. 이 과정은 데이터 매니폴드(data manifold) 내에서의 안정성을 보장합니다.

- **Performance Highlights**: 대규모 텍스트-이미지 모델에 대한 실험 결과 Rectified-CFG++는 기존의 CFG보다 다양한 벤치마크 데이터셋에서 일관되게 우수한 성능을 보였습니다. 특히 MS-COCO, LAION-Aesthetic, T2I-CompBench에서의 성과가 두드러지며, 생성 품질의 향상 뿐만 아니라 시각적 아티팩트도 감소시키는 결과를 얻었습니다. 주관적 연구에서도 문장 정렬이 뚜렷하게 개선된 것을 확인했습니다.



### Quick-CapsNet (QCN): A fast alternative to Capsule Networks (https://arxiv.org/abs/2510.07600)
- **What's New**: 이번 논문에서는 Capsule Network (CapsNet)의 성능을 개선하기 위해 Quick-CapsNet (QCN)을 제안합니다. QCN은 CapsNet보다 적은 수의 capsules을 생성하여 훈련 및 테스트 속도를 크게 향상시킵니다. MNIST, F-MNIST, SVHN 및 Cifar-10 데이터셋에서 추론 속도는 5배 빨라진다는 점에서 의미가 큽니다.

- **Technical Details**: CapsNet은 데이터 처리 및 패턴 인식에서 인체 두뇌를 모방하는 딥러닝의 한 유형입니다. 이 모델의 기본 단위는 capsule이며, capsule은 벡터 형태의 여러 뉴런을 포함합니다. QCN은 primary capsules (PCs)의 수를 줄이면서 기존 CapsNet의 구조를 변경하고, 더 강력한 decoder를 사용하여 전체 속도를 개선하고 정확도를 유지합니다.

- **Performance Highlights**: QCN은 기본 CapsNet보다 훈련 시간과 테스트 시간을 단축시키면서, MNIST와 다른 데이터셋에서 비교적 높은 정확도를 보장합니다. 실험 결과, PCs의 수를 줄임으로써 네트워크 속도가 향상되는 것을 확인했으며, 비교적 적은 수의 capsules로도 높은 성능을 달성할 수 있습니다. 이로 인해 실시간 응용에 매우 적합한 대안으로 기능할 수 있습니다.



### MaizeStandCounting (MaSC): Automated and Accurate Maize Stand Counting from UAV Imagery Using Image Processing and Deep Learning (https://arxiv.org/abs/2510.07580)
Comments:
          10 pages, 11 figures. Submitted to IEEE Journal of Selected Topics in Signal Processing (JSTSP) Special Series on Artificial Intelligence for Smart Agriculture

- **What's New**: 본 연구에서는 자동화된 옥수수 종자 수를 세는 알고리즘인 MaizeStandCounting (MaSC)를 제시합니다. MaSC는 저비용 UAV에서 촬영한 RGB 이미지를 활용하여, 수확 예측 및 식재 밀도 최적화를 지원합니다. 이 시스템은 두 가지 모드에서 작동하는데, 첫 번째는 모자이크 이미지로 나누어 처리하고, 두 번째는 동영상 프레임을 정렬하여 사용하는 방식입니다.

- **Technical Details**: MaSC는 YOLOv9 모델을 기반으로 하여 V2-V10 성장 단계의 옥수수 묘종을 감지하도록 훈련되었습니다. 이를 통해 잡초와 기타 식생으로부터 옥수수를 구분할 수 있으며, 감지된 데이터의 공간 분포에 따라 구간 및 행 세분화를 수행합니다. 또한, MaSC는 최신 UAV 이미지를 효율적으로 처리하며, 이미지 모자이크와 동영상 입력 모두 지원하여 연구 및 실제 환경에서 안정적인 수량을 제공합니다.

- **Performance Highlights**: 출시된 평가 결과, MaSC는 2024년 여름 재배지에서 수동 세기와 강한 일치를 보이며 (R²=0.616), 원시 프레임에서는 더 높은 정확도 (R²=0.906)를 기록했습니다. MaSC는 단 50.63초 만에 83개의 풀 해상도 프레임을 처리할 수 있으며, 이는 실시간 작동의 가능성을 잘 보여줍니다. 이를 통해 연구자와 생산자 모두에게 도움이 되는 신뢰할 수 있는 자동화 도구의 역할을 맡을 수 있습니다.



### Cross-Modal Attention Guided Unlearning in Vision-Language Models (https://arxiv.org/abs/2510.07567)
- **What's New**: 이 논문은 시각 언어 모델(Visual-Language Models, VLMs)의 개인 정보 노출 문제를 해결하기 위해 비전-질문 응답(Visual Question Answering, VQA) 작업에서의 비학습(Unlearning) 프레임워크인 CAGUL(Cross-Modal Attention Guided Unlearning)을 제안합니다. CAGUL은 특정 비주얼 토큰을 변형하여 개인정보 유출을 방지하면서도 모델의 참조 행동을 유지할 수 있도록 설계되었습니다. 또한, CAGUL은 조건부 조회 시 비중이 낮은 비주얼 토큰의 정보를 활용하여 개인 정보를 안전하게 제거할 수 있습니다.

- **Technical Details**: CAGUL은 비전 모델(vision model), 크로스 모달 컴포넌트(cross-modal component), 언어 모델(language model)로 구성된 VLM의 아키텍처를 기반으로 합니다. 비주얼 입력과 쿼리 간의 관계를 탐구하고, MLP 인코더를 사용하여 가장 낮은 크로스 모달 주의 점수(cross-modal attention scores)를 가진 비주얼 토큰을 변형합니다. 이 과정에서, 유용한 정보 삭제를 위한 훈련 손실을 정의하며, 저비용의 외부 모듈을 사용하여 사전 학습된 VLM 매개변수는 고정시키고 있습니다.

- **Performance Highlights**: 실험 결과, CAGUL은 기존의 파인튜닝(fine-tuning) 기반 방법과 비교하여 유사한 성능을 발휘하며 참조 모델 행동을 유지합니다. 특히, CAGUL은 비주얼 입력에서 개인 정보가 포함된 질문과의 연결된 비주얼 입력을 효과적으로 탐지하고, 모든 이미지 토큰을 사용하는 것과 유사한 성능을 이뤘습니다. 또한, CAGUL은 전반적으로 비주얼과 언어 정보 간의 상관관계를 이용하여 더욱 효과적인 비학습 접근 방식을 발전시켰음을 보여줍니다.



### Label Semantics for Robust Hyperspectral Image Classification (https://arxiv.org/abs/2510.07556)
Comments:
          This work has been accepted for publication in the proceedings of IJCNN 2025

- **What's New**: 이 논문에서는 Semantically Guided Semantic Spectral-Spatial Fusion Network(S3FN)를 제안하여 고차원 HSI(초분광 이미지) 데이터의 분류 성능을 향상시키고자 합니다. 이 모델은 각 클래스 레이블에 대한 텍스트 설명을 생성하여, 기존의 단일 모드 방법의 한계를 극복하고 보다 의미 있는 피쳐-레이블 정렬을 가능하게 합니다. 이를 통해 과거 모델들이 놓치곤 했던 세부적인 스펙트럼 정보와 의미적 관계를 효과적으로 활용합니다.

- **Technical Details**: S3FN은 LLM(대형 언어 모델)을 활용하여 각 클래스의 고유한 특성과 스펙트럼 행동을 반영한 포괄적인 텍스트 설명을 생성합니다. 이 설명들은 BERT나 RoBERTa와 같은 사전 훈련된 텍스트 인코더를 통해 벡터 공간에 임베딩되어, HSI 데이터의 특성 표현을 강화합니다. LLM을 사용함으로써, 수동으로 작성된 불완전하거나 모호한 설명 문제를 해결하고, 다양한 클래스 간의 스펙트럴 관계를 더 깊이 이해할 수 있게 합니다.

- **Performance Highlights**: 논문에서 제안하는 S3FN은 Hyperspectral Wood, Hyperspectral Blueberries, DeepHS-Fruit의 세 가지 HSI 벤치마크 데이터셋에서 평가되었으며, 기존 방법에 비해 성능이 유의미하게 향상되었습니다. 텍스트 기반 의미적 정보와 스펙트럴-공간적 데이터를 융합한 결과, 보다 정교한 분류 성능을 달성하게 되었으며, 이는 HSI 분류의 발전 가능성을 보여줍니다.



### TRAVL: A Recipe for Making Video-Language Models Better Judges of Physics Implausibility (https://arxiv.org/abs/2510.07550)
- **What's New**: 이 논문에서는 비디오 생성 모델들이 물리적 법칙을 위반한 입력들을 자주 생성하는 문제를 다룹니다. 기존의 Video-Language Models (VLMs)가 이러한 물리적 현실성을 평가하는 데 어려움을 겪고 있어서, 새로운 방법론인 TRAVL과 평가 기준인 ImplausiBench를 도입하여 이 문제를 해결하고자 합니다. TRAVL은 모션을 인식할 수 있는 주의(attention) 메커니즘을 사용하여 VLM의 물리적 이해도를 향상시키고, 정교한 평가 방식을 통해 물리적 불가능성을更正하는 방법을 제시합니다.

- **Technical Details**: TRAVL(trajectory-aware Vision-Language learning)은 VLM을 개선하기 위한 모듈식 방법으로, 모션 정보를 기억하는 자기 주의(self-attention) 메커니즘을 통해 비디오의 물리적 구조를 포착하는 데 중점을 둡니다. 이 방법은 두 가지 주요 메커니즘인 intra-frame 공간 주의와 trajectory-aware temporal 주의를 통해 비디오의 물리적 동적을 더 잘 파악할 수 있도록 설계되었습니다. 또한, ImplausiBench라는 300개의 동영상으로 구성된 평가 기준을 통해 인간과 LLM의 판단 모두에서 보다 엄격한 물리적 현실성을 평가할 수 있도록 하였습니다.

- **Performance Highlights**: TRAVL과 ImplausiBench는 VLM의 물리적 현실성 향상을 위한 통합된 프레임워크를 제공합니다. 이들은 VLM이 임의의 동영상에서 물리적으로 가능한지 검사할 수 있도록 하며, 기존 모델들이 놓치는 심층적인 물리적 이해를 목표로 합니다. 이러한 접근 방식은 물리적 상상력의 한계와 기존의 벤치마크가 갖는 제약사항을 극복하고자 하며, 더 나아가 VLM의 발전 방향에 중요한 기여를 할 것으로 기대됩니다.



### PickStyle: Video-to-Video Style Transfer with Context-Style Adapters (https://arxiv.org/abs/2510.07546)
- **What's New**: 이번 논문에서는 확산 모델(difussion models)을 활용하여 비디오 스타일 전송(video style transfer) 문제를 해결합니다. 주된 목표는 입력 비디오의 맥락을 보존하면서 텍스트 프롬프트로 지정된 목표 스타일로 렌더링하는 것입니다. 이를 위해 저자들은 Pairing된 정적 이미지 데이터를 활용하여 학습하는 PickStyle이라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 저자들은 PickStyle에서 저순위 어댑터(low-rank adapters)를 self-attention 레이어에 삽입하여 운동 스타일 전송(motion-style transfer)을 위한 효율적인 전문화를 가능하게 합니다. 또한, CS-CFG(Context-Style Classifier-Free Guidance)라는 새로운 접근 방식을 도입하여 스타일 전송 시 맥락이 유지되도록 하였습니다. 이를 통해 정적 이미지 감독을 동적 비디오로 연결하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 PickStyle 방식은 기존 모델들보다 월등한 성능을 나타내며, 스타일 충실도(style fidelity)와 비디오 맥락(content preservation)을 동시에 만족하는 일관된 비디오 변환을 제공합니다. 이러한 접근은 여러 기준 벤치마크에서 품질적으로나 정량적으로 우수한 결과를 달성했습니다.



### D2RA: Dual Domain Regeneration Attack (https://arxiv.org/abs/2510.07538)
- **What's New**: 이번 연구는 D2RA(Dual Domain Regeneration Attack)라는 새로운 기법을 소개합니다. 이 기법은 단일 이미지를 대상으로 하는 훈련 없이 수행되는 공격으로, 수반된 모델에 접근하지 않고도 watermark를 제거하거나 약화시킬 수 있습니다. D2RA는 자연적인 사전 정보에 기반해 watermark 신호를 억제하는 방법으로, 기존의 watermarking 접근방식의 근본적인 약점을 드러냅니다.

- **Technical Details**: D2RA는 세 가지 모듈로 구성됩니다: (i) 주파수 도메인 재구성을 통해 스펙트럼 정규성을 복원하고, (ii) 의미 체계 개선을 통해 높은 수준의 구조를 유지하며, (iii) 지각 색상 보정을 통해 현실감을 보존합니다. 이 프레임워크는 픽셀 공간, 주파수 공간 및 잠재 공간에 걸쳐 효과적으로 적용되며, 훈련 없이 단일 샷으로 실행될 수 있습니다.

- **Performance Highlights**: 실험을 통해 D2RA는 다양한 watermarking 방법에 대해 일관되게 watermark 탐지 가능성을 줄이며, 이는 기존 기법들의 근본적인 취약성을 노출합니다. 또한, D2RA는 고급 이미지 통계에 의해 드러나는 왜곡 신호들을 제거함으로써 시각적 및 의미적 일관성을 유지합니다. 이러한 결과는 watermarking 설계에서 더욱 강인한 접근법을 위한 가이드를 제공합니다.



### A Denoising Framework for Real-World Ultra-Low Dose Lung CT Images Based on an Image Purification Strategy (https://arxiv.org/abs/2510.07492)
- **What's New**: 이 연구는 ultra-low dose CT (uLDCT) 의 비율을 줄임으로써 방사선 노출을 최소화하는 동시에, 이를 위한 새로운 denoising 프레임워크를 제안합니다. 특히, Image Purification (IP) 전략을 통해 uLDCT와 normal dose CT (NDCT) 간의 구조적 정렬을 개선하고, 성능을 대폭 향상시켰습니다.

- **Technical Details**: 연구팀은 실제 임상 uLDCT 폐 데이터셋을 구축하고, IP 전략을 사용해 uLDCT-NDCT 이미지 쌍의 정렬을 최적화했습니다. 이를 기반으로 Frequency-domain Flow Matching (FFM) 모델을 제안하여, anatomical structure의 무결성을 보존하며 denoising을 수행합니다. 이러한 접근법은 기존의 방식들과 비교해 새로운 차원의 성능 향상을 가져옵니다.

- **Performance Highlights**: 실험 결과, IP 전략은 여러 주류 denoising 모델의 성능을 크게 향상시키는 것으로 나타났습니다. 특히, IP 전략과 FFM 모델의 결합에 의해 구조 보존에서 state-of-the-art (SOTA) 성능을 달성했습니다. 이는 실제 임상에서의 uLDCT denoising 데이터 불일치 문제를 해결하는 효과적인 방법을 제시합니다.



### Provably Accelerated Imaging with Restarted Inertia and Score-based Image Priors (https://arxiv.org/abs/2510.07470)
Comments:
          62 pages

- **What's New**: 이번 논문에서는 이미지 역문제를 해결하기 위한 새로운 알고리즘, Restarted Inertia with Score-based Priors (RISP)를 제안합니다. 이 방법은 기존의 Regularization by Denoising (RED)를 기반으로 하여, 빠른 수렴(Fast Convergence)을 가능하게 하는 재시작 관성을 통합합니다. RISP는 고품질 이미지 재구성을 위한 스코어 기반 이미지 프라이어(Score-based Priors)를 계속 사용할 수 있으며, RED보다 더 빠른 정적 점 수렴율을 보입니다.

- **Technical Details**: RISP는 두 가지 알고리즘적 구현, 즉 RISP-GM(Gradient Method)과 RISP-Prox(Proximal Method)를 포함합니다. 이들 알고리즘은 각각 그래디언트와 근접 공식을 기반으로 하며, 일반적인 가정 하에 두 알고리즘 모두 수렴 속도를 𝒪(n^{-4/7})로 설정합니다. 이는 RED의 𝒪(n^{-1/2}) 속도를 초과하며, 스코어 기반 프라이어의 볼록성을 요구하지 않기 때문에 높은 자유도를 제공합니다.

- **Performance Highlights**: RISP는 다양한 선형 및 비선형 역이미징 문제에서 실험적으로 검증되었습니다. 특히, RISP를 적용했을 때, 대규모 이미지 재구성에서 최대 24배의 속도 향상이 가능하다는 결과가 나타났습니다. 이러한 성능 향상은 이미지 프라이어와의 통합을 통해 이루어진 것으로, 실용적인 알고리즘 개선으로 평가됩니다.



### DynamicEval: Rethinking Evaluation for Dynamic Text-to-Video Synthesis (https://arxiv.org/abs/2510.07441)
Comments:
          Preprint. Under review. 26 pages, 11 figures, 11 tables. Access the project page in this https URL

- **What's New**: 본 논문에서는 기존의 텍스트-비디오(T2V) 평가 기준의 두 가지 한계를 지적하며, 이를 해결하기 위해 DynamicEval 이라는 새로운 벤치마크를 제안합니다. 기존 벤치마크는 주로 정적인 카메라 장면과 피사체 중심의 프롬프트에 집중되어 있으며, 동적 카메라 무빙에 대한 평가가 부족했습니다. DynamicEval은 동적인 카메라 무빙을 강조한 체계적으로 구성된 프롬프트와 3천 편의 비디오에서 생성된 4만 5천 개의 인간 주석으로 구성되어 있습니다.

- **Technical Details**: DynamicEval은 배경 장면 일관성과 전경 객체 일관성의 두 가지 주요 차원을 평가하여 비디오 품질을 결정합니다. 배경 일관성을 평가하기 위해, VBench의 모션 부드러움 지표를 기반으로 한 해석 가능한 오류 맵을 제공합니다. 또한 전경 객체 일관성을 평가하기 위한 새로운 메트릭을 도입하여 개별 객체 내에서 포인트와 이웃을 추적하며 객체 충실도를 평가합니다.

- **Performance Highlights**: 대규모 실험을 통해 제안된 메트릭이 기존 메트릭들보다 인간 선호도와의 상관관계에서 2%포인트 이상 개선된 결과를 나타내었으며, 이는 DynamicEval이 동적 카메라 무빙 하에서 T2V 모델 평가의 포괄적인 기준으로 자리잡게 합니다. 우리 연구의 주요 기여는 다양한 카메라 무빙을 포함한 100개의 프롬프트와 3천 개 비디오에 대한 4만 5천 개의 고품질 인간 주석으로 구성된 평가 세트를 제공한 것입니다.



### Enhancing Maritime Object Detection in Real-Time with RT-DETR and Data Augmentation (https://arxiv.org/abs/2510.07346)
Comments:
          13 pages, 10 figures

- **What's New**: 이 논문은 RT-DETR 기반의 실시간 해양 물체 탐지 시스템을 제안하며, 합성 이미지를 활용한 데이터 증강을 통해 실제 데이터에서의 평가를 엄격히 수행하는 것이 특징입니다. 제안된 시스템은 다중 크기 특징 융합(multi-scale feature fusion), 불확실성 최소화 쿼리 선택(uncertainty-minimizing query selection), 합성 및 실제 샘플 간의 스마트 가중치 조정(smart weight) 기법을 통합하여 해양 환경에 맞춰진 방식입니다. 이 연구는 실제 조명 및 해양 조건에서도 성능을 유지하는 디지털 파이프라인을 제공하여 모든 건축 모듈의 기여도를 정량화하고, 시스템이 극한 조건에서 실패를 처리하는 방식도 설명합니다.

- **Technical Details**: RT-DETR은 완전한 엔드-투-엔드(attention-based) 탐지기로, 다중 크기 처리(multi-scale processing)와 불확실성 인식 쿼리 선택(uncertainty-aware query selection)으로 실시간 효율성을 확보합니다. 이 시스템은 최적의 속도와 정확성을 조절할 수 있으며, 각 아키텍처 모듈의 기여도를 평가하기 위한 구성 요소 분석(component analysis)을 포함하고 있습니다. 훈련 과정에서는 실제 해양 이미지와 GAN에서 생성된 합성 샘플을 결합하여 더 다양한 조명 및 날씨 조건을 반영하고, 특이한 조건을 모사하기 위해 비쌍 이미지 전송 기술도 활용되었습니다.

- **Performance Highlights**: 제안된 RT-DETR 시스템은 다양한 임계값을 통해 실제 이미지에서의 정확도를 유지하면서도 성능을 향상시키는 것을 목표로 합니다. 다양한 조명, 환경 변화에 강건한 성능을 보여주며, 실시간 성능을 유지하는 파이프라인으로 실용성을 강조합니다. RT-DETR은 다른 YOLO 모델들과 비교하여 속도와 정확성 모두에서 우수한 성능을 발휘하며, 경량화된 처리로 모델의 견고함과 유연성을 제공하여 해양 물체 탐지의 신뢰성 있는 기반을 마련합니다.



### Scalable Offline Metrics for Autonomous Driving (https://arxiv.org/abs/2510.08571)
Comments:
          Accepted at IROS 2025 (IEEE/RSJ International Conference on Intelligent Robots and Systems)

- **What's New**: 본 논문에서는 자율주행 정책 평가에서 오프라인(offline) 평가와 온라인(online) 평가 간의 상관관계에 대한 연구를 진행하였습니다. 기존 연구보다 더 나쁜 상관관계를 발견하였으며, 이는 현재의 평가 방식과 메트릭의 유효성에 의문을 제기합니다. 또한, 에피스틱 불확실성을 기반으로 한 새로운 오프라인 지표를 제시하여, 온라인 환경에서도 보다 신뢰할 수 있는 평가 절차를 제안합니다.

- **Technical Details**: 이 논문에서 제안된 메트릭은 복잡한 도시 운전 상황과 다양한 폐쇄 루프(closed-loop) 지표를 통해 분석되었습니다. 저자들은 불확실성을 고려한 에러를 기반으로 한 새로운 메트릭이 13% 이상의 상관관계 개선 효과를 보인다고 주장합니다. 특히, 시뮬레이션 환경을 넘어 실제 환경에서도 발견된 결과를 통해 제안된 메트릭의 유효성을 검증하였습니다.

- **Performance Highlights**: 저자들은 시뮬레이션을 활용하여 더 많은 도시와 다양한 폐쇄 루프 지표들을 포함한 평가를 진행하였습니다. 이를 통해 그 동안 알려진 것보다 더욱 나쁜 상관관계 패턴이 드러났으며, 이는 자율주행 모델의 안전하고 신뢰할 수 있는 평가 기준을 재검토할 필요성을 시사합니다. 마지막으로, 제안된 메트릭은 복잡한 실세계 환경에서의 적용 가능성도 높이며, 안전한 자율주행 정책 개발에 기여할 수 있는 기반이 될 것입니다.



### NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos (https://arxiv.org/abs/2510.08568)
- **What's New**: 로봇이 새로운 조작 작업을 성능 저하 없이 수행할 수 있도록 하는 것을 목표로 하는 논문에서, NovaFlow라는 새로운 자율 조작 프레임워크를 소개합니다. 이 프레임워크는 작업 설명을 바탕으로 직접적인 시연 없이도 목표 로봇을 위한 실행 가능한 계획으로 변환합니다. 특히, 3D 물체 흐름을 활용하여 로봇의 동작을 계산하며, 이는 로봇의 제어 방식과는 관련이 없습니다.

- **Technical Details**: NovaFlow는 비디오 생성 모델을 활용하여 작업 이해를 위한 일반적인 3D 물체 흐름을 생성합니다. 이 흐름은 로봇의 조작을 위한 다차원적 표현으로, 개별적인 동작을 생성하는 데 사용되며, 전통적인Inverse Kinematics (IK) 및 궤적 최적화(trajectory optimization) 기법이 포함됩니다. 여기서는 강체 물체와 변형 가능한 물체를 서로 다른 방식으로 처리하여 다양성 있는 조작을 지원합니다.

- **Performance Highlights**: NovaFlow는 Franka arm과 Spot 이동 로봇을 사용하여 여러 실제 조작 작업에서 효과적인 제로샷 실행을 성공적으로 수행했습니다. 전통적인 시연 필요 없이 다채로운 물체 조작 작업을 수행할 수 있는 능력을 보여, 결과적으로 기존의 데이터 의존 방법들보다 뛰어난 성능을 달성하였습니다.



### How to Teach Large Multimodal Models New Skills (https://arxiv.org/abs/2510.08564)
Comments:
          In submission. Code is available at this https URL

- **What's New**: 이 논문은 대규모 다중모달 모델(large multimodal models, LMMs)에서 새로운 기술을 가르칠 때 이전 능력을 지키는 방법에 대해 연구합니다. 특히, 다섯 가지 목표 기술을 순차적으로 미세 조정(sequential fine-tuning)하면서, 세 모델 계열에 걸쳐 여덟 가지 기준 벤치마크에서 일반 능력을 모니터링합니다. 연구 결과, 좁은 범위의 미세 조정을 한 후 일부 기준 작업에서 '망각(forgetting)'이 관찰되지만, 후속 단계에서 이러한 영속적이지 않은 기억이 어느 정도 회복됨을 발견했습니다.

- **Technical Details**: 우리는 '망각'을 초래하는 결과 분포의 변화를 측정할 수 있는 간단한 카운팅 편향 탐지(counting-bias probe)를 통해 추적합니다. 이 연구에서는 두 가지의 간단하고 견고한 조정 레시피(tuning recipes)를 제안합니다: (i) 자기 주의(self-attention) 투영 레이어만 업데이트하고, (ii) Down 투영을 고정(freezing)한 상태에서 MLP Gate&Up만 업데이트하는 방법입니다.

- **Performance Highlights**: 이러한 조정 방법을 통해 모델과 작업 전반에 걸쳐 강력한 목표 성과(target gains)를 달성하면서도 이전 기준 성능(held-out performance)을 대체로 유지할 수 있음을 보여줍니다. 제안된 방법은 각 모델과 작업에서 향상된 결과를 제공하며, 연구 결과는 실제 코드로도 확인할 수 있습니다.



### DexNDM: Closing the Reality Gap for Dexterous In-Hand Rotation via Joint-Wise Neural Dynamics Mod (https://arxiv.org/abs/2510.08556)
Comments:
          Project Website: this https URL Video: this https URL

- **What's New**: 이 연구는 시뮬레이션에서 훈련된 단일 정책을 사용하여 다양한 객체와 조건에서 일반화된 손안 회전을 가능하게 하는 새로운 프레임워크를 제안합니다. 이 방법은 조인트별(dynamics) 모델을 기반으로 하여 실제에서 수집된 제한된 데이터에 효과적으로 피팅하고, 정책의 행동을 조정함으로써 현실 간극(reality gap)을 극복하는 것을 목표로 합니다.

- **Technical Details**: 이 모델은 조인트를 통한 동역학 예측을 수행함으로써 시스템 전체에 대한 복잡한 상호 작용 동역학을 단순화합니다. 각 조인트의 동적 프로파일에 따라 진화 과정을 학습하여, 객체 상태 추정의 어려움에 대해 내성을 갖추고, 시스템 전반의 영향을 저차원(low-dimensional) 변수로 축소하여 샘플 효율성과 일반화를 향상시킵니다.

- **Performance Highlights**: 이 연구의 결과, 단일 정책으로 복잡한 모양의 객체(예: 동물 모델) 및 다양한 손목 방향과 회전축을 통해 회전 성능이 크게 향상되었습니다. 특히 어려운 손 위치에서도 긴 객체를 한 번에 회전시킬 수 있는 최초의 사례를 제시하였으며, 실제 및 시뮬레이션 모두에서 강력한 성능을 입증하였습니다.



### R2RGEN: Real-to-Real 3D Data Generation for Spatially Generalized Manipulation (https://arxiv.org/abs/2510.08547)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서 제안하는 R2RGen은 실제 세계에서 3D 데이터를 생성하는 혁신적인 프레임워크로, 고정된 테이블탑 설정이 아닌 이동 로봇을 위한 일반적인 조작 환경에 적합합니다. R2RGen은 원시 포인트클라우드 관찰을 바탕으로 하여 개체 수나 상호작용 모드의 제약 없이 데이터 생성을 가능하게 합니다. 이전 방안들이 가진 강력한 가정이나 제한을 극복하여, 복잡한 기술과 다양한 작업 제약을 다룰 수 있는 방안을 제공합니다.

- **Technical Details**: R2RGen은 시뮬레이터나 렌더링 없이 작동하여 효율성과 사용 용이성을 극대화합니다. 이 프레임워크는 미리 분할된 소스 데모를 이용해 객체별로가 아닌 그룹 단위로 데이터를 증강하는 전략을 도입합니다. 또한, 3D 센서에서의 관찰 데이터를 보장하기 위한 카메라 인식 3D 후처리 기법을 제시하여, 관찰 데이터의 왜곡을 최소화합니다.

- **Performance Highlights**: 실험 결과, R2RGen은 단 하나의 인간 데모로 학습하여 25배 이상의 인간 수집 데이터를 사용해야만 얻는 성능을 초과합니다. R2RGen은 추가 데모에 대한 강력한 확장성을 보여주며, 공간 변화를 초월한 일반화 능력을 가지고 있습니다. 이 결과는 모바일 조작과 같은 응용 분야에 대한 유망한 가능성을 보여줍니다.



### X2Video: Adapting Diffusion Models for Multimodal Controllable Neural Video Rendering (https://arxiv.org/abs/2510.08530)
Comments:
          Code, model, and dataset will be released at project page soon: this https URL

- **What's New**: X2Video는 고유 채널을 기반으로 하여 포토리얼리스틱 비디오를 생성하는 최초의 확산 모델입니다. 이 모델은 알베도(albedo), 노말(normal), 러프니스(roughness), 메탈리시티(metallicity), 조명(irradiance)과 같은 내재적 정보를 활용하며, 참조 이미지와 텍스트 프롬프트를 통한 직관적인 다중 모드 제어를 지원합니다. 이러한 혁신은 색상, 소재, 기하학, 조명을 정밀하게 조작할 수 있는 가능성을 열어줍니다.

- **Technical Details**: X2Video는 XRGB 모델을 기반으로 하여 내재적 데이터를 포토리얼리스틱 이미지로 변환하는 데 필요한 지식을 상속받습니다. 이를 통해 하이브리드 셀프 어텐션(Hybrid Self-Attention) 메커니즘을 도입하여 시간적 일관성을 보장하고, 마스크된 크로스 어텐션(Masked Cross-Attention)을 통해 글로벌 및 로컬 텍스트 프롬프트를 효과적으로 적용합니다. 또, 장기 비디오 생성을 위한 새로운 재귀 샘플링(Recursive Sampling) 방법도 적용하여 긴 시간적 일관성을 유지하며 오류 누적을 방지합니다.

- **Performance Highlights**: X2Video는 1,154개의 방을 포함한 InteriorVideo 데이터셋을 통해 학습되었으며, 고유 채널 조건에 따라 긴 포토리얼리스틱 비디오를 생성할 수 있는 성능을 보여줍니다. 모델은 직관적인 제어가 가능하며, 색상, 소재, 기하학, 조명 편집을 지원합니다. 정성적 및 정량적 평가 결과, X2Video는 높은 시간적 일관성을 지닌 롱 비디오를 생성하는 능력을 입증하였습니다.



### AI-Driven Radiology Report Generation for Traumatic Brain Injuries (https://arxiv.org/abs/2510.08498)
- **What's New**: 이번 논문에서는 외상성 뇌손상 진단을 위한 자동 방사선 보고서 생성을 위한 혁신적인 AI 기반 접근법을 제안합니다. AC-BiFPN (Augmented Convolutional Bi-directional Feature Pyramid Network)과 Transformer 아키텍처를 통합하여 CT 및 MRI 스캔과 같은 복잡한 의료 영상 데이터를 처리합니다. 이 모델은 의료 영상에서의 다중 스케일 기능을 활용하여 intracranial hemorrhages와 같은 복잡한 이상을 효과적으로 탐지합니다.

- **Technical Details**: 제안된 모델은 AC-BiFPN을 통해 CT 및 MRI 이미지에서 다중 스케일 특징을 추출하고, Transformer를 통해 긴 의존성을 모델링하여 일관성 있는 진단 보고서를 생성합니다. AC-BiFPN은 다양한 해상도에서의 특징을 융합하여 상세한 뇌 스캔 분석에 적용되어 중요한 정보를 놓치지 않도록 합니다. 또한, 신뢰할 수 있는 진단 보고서를 생성하기 위해 Transformer 기반 모델을 사용하여 시각적 정보와 텍스트 정보를 통합합니다.

- **Performance Highlights**: 모델은 RSNA Intracranial Hemorrhage Detection 데이터셋에서 전통적인 CNN 기반 모델들보다 우수한 진단 정확도와 보고서 생성을 보여줍니다. 이 접근법의 혁신성은 고압 환경에서도 방사선 전문의를 지원하고, 훈련생 의사에게 실시간 피드백을 제공하여 학습 경험을 강화하는 데 기여합니다. 우리의 연구는 고급 기능 추출과 Transformer 기반의 텍스트 생성을 결합하여 외상성 뇌손상의 진단 과정에서 임상적 의사결정을 개선할 가능성을 보여줍니다.



### Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models (https://arxiv.org/abs/2510.08492)
Comments:
          63 pages, 29 tables, and 47 figures

- **What's New**: UML (Unpaired Multimodal Learner)이라는 새로운 훈련 패러다임을 소개합니다. 이 모델은 서로 다른 모달리티의 입력을 처리하며, 매개변수를 공유하여 모달리티에 독립적인 특징을 학습할 수 있도록 설계되었습니다. 서로 다른 모달리티가 공통의 잠재 현실를 투영하고 있다는 가정을 바탕으로, 일치된 데이터 없이도 상호 모달리티 구조의 이점을 누릴 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 UML이 비슷한 구조의 입력을 처리할 때 매개변수 공유를 통해 서로 다른 모달리티의 정보를 자연스럽게 전이할 수 있도록 하는 방법을 수립했습니다. 이론적으로, 선형 가정 하에서 비일치 지원 데이터가 단일 모달 훈련보다 더 정보적인 표현을 산출할 수 있는 조건을 유도하였습니다. 다양한 실험에서는 이미지-텍스트 작업에서 비일치 데이터가 자기 지도 학습(self-supervised) 및 감독 학습(supervised) 각각의 분야에서 일관된 개선을 보여주었습니다.

- **Performance Highlights**: UML은 데이터 모든 유형에서의 성능 향상을 입증하였으며, 10개의 표준 시각 벤치마크에서 효과적인 수준으로 출현하였습니다. 특히, 두 개의 모달리티에서 세 개의 모달리티로 전환할때 정보 전이 효과가 더욱 뚜렷해졌습니다. 실험 결과, 이미지와 텍스트 간의 전환 비율을 정량화하여, 한 개의 이미지가 훈련 모델에서 사용할 수 있는 단어 수를 측정하는 방법도 포함되어 있습니다.



### Splat the Net: Radiance Fields with Splattable Neural Primitives (https://arxiv.org/abs/2510.08491)
- **What's New**: 본 논문에서는 ‘Splattable Neural Primitives’라는 새로운 볼륨 표현 방식을 소개합니다. 이 방식은 Neural Radiance Fields의 높은 표현력을 가지고 있지만 고비용의 ray marching 없이도 효율적으로 작동합니다. 이러한 진보는 전체적인 품질과 성능을 유지한 채 더 적은 파라미터로 특성을 인코딩할 수 있게 해줍니다.

- **Technical Details**: ‘Splattable Neural Primitives’는 얕은 신경망으로 매개변수화된 구역적 Neural Density Field로 구성되어 있습니다. 이 접근 방식은 선형 적분에 대한 정확한 해를 제공하여 이미지 공간에서의 splatting 커널을 효율적으로 계산할 수 있게 합니다. 이러한 특성 덕분에 복잡한 제어 또는 적용 프레임워크에 의존하지 않고도 밀도 필드를 픽셀의 시선 방향을 따라 통합할 수 있습니다.

- **Performance Highlights**: 이 방식은 기존의 3D Gaussian Splatting(3DGS)과 비교하여 품질과 속도 모두를 일치시키면서, 10배 적은 수의 primitives 및 6배 적은 파라미터를 사용합니다. 따라서, 이 방법은 기억 공간에서의 효율성을 극대화하여 실시간 렌더링 속도를 달성합니다. 최종적으로 이러한 이점은 표현 방식의 설계에서 직접적으로 비롯됩니다.



### DexMan: Learning Bimanual Dexterous Manipulation from Human and Generated Videos (https://arxiv.org/abs/2510.08475)
Comments:
          Video results are available at: this https URL

- **What's New**: DexMan은 인간의 시각적 데모를 바탕으로 휴머노이드 로봇의 양손 조작 기술로 자동 변환하는 프레임워크입니다. 이 시스템은 인간의 조작을 담은 제3자 비디오를 사용하여 카메라 보정이나 깊이 센서 없이 작동하며, 회전하는 손의 단순화된 모델을 넘어서는 실제 제어를 가능하게 합니다. 또한 DexMan은 실제와 합성된 비디오에서 기술을 생성할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: DexMan은 VGGT라는 이미지-3D 기초 모델을 활용하여 비디오 프레임의 깊이를 예측합니다. 긴 비디오를 겹치는 청크로 나누어 처리하며, 이로 인해 발현되는 척도 불일치 문제를 해결하기 위한 객체 중심의 시간 정렬 전략을 제안합니다. 이는 예측된 깊이 값의 일관성을 보장하고, 자원 소모를 줄이며, 다양한 훈련용 데이터셋을 생성할 수 있게 합니다.

- **Performance Highlights**: DexMan은 TACO 벤치마크에서 객체 자세 추정에 있어 최상의 성능을 기록하며, ADD-S와 VSD에서 각각 0.08과 0.12의 향상을 달성했습니다. 또한 강화 학습 정책은 이전 방법보다 성공률이 19% 향상되었습니다. 이 강력한 성능은 DexMan이 인간 조작의 복잡성을 효과적으로 학습하고 재현할 수 있음을 보여줍니다.



### Reinforcing Diffusion Models by Direct Group Preference Optimization (https://arxiv.org/abs/2510.08425)
- **What's New**: 이번 논문은 Direct Group Preference Optimization(DGPO)이라는 새로운 온라인 강화 학습 알고리즘을 제안합니다. DGPO는 기존의 Group Relative Preference Optimization(GRPO)의 정책 경량화 프레임워크를 완전히 배제하고 그룹 수준의 선호도에서 직접 학습함으로써 확산 모델(diffusion models)의 훈련을 가속화합니다. DGPO는 비효율적인 확률적 정책의 필요성을 제거하고, 효율적인 결정론적 ODE 샘플러를 활용하여 훈련 속도를 약 20배 향상시킵니다.

- **Technical Details**: DGPO는 기존의 정책 기반 강화 학습 대신 그룹간 상대적 선호도를 직접적으로 최적화하는 방식으로 작동합니다. 각 프롬프트에 대해 ODE 기반 롤아웃을 사용하여 양호한 샘플과 불량 샘플로 나누고, 이들 그룹 간의 선호도를 최대화함으로써 모델을 최적화합니다. 이 방법론은 Direct Preference Optimization(DPO)의 자연스러운 확장으로, 그룹 간 정보를 포함하면서도 정책의 확률적 의존성을 제거하는 방식입니다.

- **Performance Highlights**: DGPO의 실험 결과는 Flow-GRPO와 비교하여 약 20배 더 빠른 훈련 속도를 기록하며, 다양한 도메인 보상 메트릭에서 뛰어난 성능을 보여줍니다. 특히, GenEval 벤치마크에서 DGPO는 Flow-GRPO보다 거의 30배 빠른 훈련을 달성하고 기본 모델의 성능을 63%에서 97%로 향상시킵니다. 이러한 성과들은 DGPO가 확산 모델 정렬을 위한 강력한 기법으로 자리매김할 가능성을 보여줍니다.



### Biology-driven assessment of deep learning super-resolution imaging of the porosity network in dentin (https://arxiv.org/abs/2510.08407)
- **What's New**: 이번 연구에서는 치아의 기계 감각 시스템이 주로 Odontoblast 세포의 자극에 의존하고 있으며, 이를 위해서는 미세한 다공성 네트워크를 시각화해야 함을 강조합니다. 이를 위해 현재의 표준인 confocal fluorescence microscopy의 한계를 극복하기 위해 다양한 딥러닝(DL) 초해상도(SR) 모델을 테스트했습니다. 연구팀은 훈련된 모델을 통해 저해상도 이미지의 실험적 수집 속도를 높이고 최적의 이미지 품질을 복원하는 방법을 개발했습니다.

- **Technical Details**: 세 가지 감독형 2D SR 모델(RCAN, pix2pix, FSRCNN)과 한 가지 비감독형(CycleGAN)을 사용하여 고해상도 및 저해상도 confocal 이미지를 쌍으로 실험하여 다양한 샘플링 방식으로 획득했습니다. 생성된 SR 이미지는 픽셀 크기를 2배, 4배, 8배로 증가시키는 성과를 얻었습니다. 모델의 성능은 이미지 품질 평가(IQA) 메트릭을 사용하여 측정되었으며, 이는 시각적 인지와 모순되는 일관되지 않은 결과를 보여주었습니다.

- **Performance Highlights**: IQA 메트릭의 한계를 넘어 치아 다공성의 특정 구조를 겨냥한 세그멘테이션 접근방식이 사용되었습니다. 또한 SR 모델이 confocal 이미지 스택을 통해 3D 다공성 연결성을 유지하는 능력을 그래프 분석을 통해 평가했습니다. 이러한 생물학적 기반의 평가는 SR 성능의 기계적 해석을 개선하고, 모델의 민감도 차이와 이미지 생성의 비선형성이 IQA 메트릭의 실패를 설명함을 강조합니다.



### Spectral Prefiltering of Neural Fields (https://arxiv.org/abs/2510.08394)
Comments:
          16 pages, 10 figures, to be published in Siggraph Asia 2025, Website: this https URL

- **What's New**: 본 논문은 단일 정방향 패스에서 전처리된 신경 필드를 최적화하는 간단하면서 강력한 방법을 제안합니다. 주요 혁신에는 주파수 응답을 사용하여 푸리에 특징 임베딩을 컨볼루션 필터링하는 것이 포함되어 있으며, 이는 가우시안 필터링에 국한되지 않고 다양한 파라메트릭 필터를 지원합니다.

- **Technical Details**: 이 방법은 신경 필드를 훈련시키기 위해 필터링된 신호의 단일 샘플 몬테 카로 추정기를 사용합니다. 또한, 네트워크 아키텍처에 대한 추가 제약을 두지 않으며, 푸리에 기능 인코딩만 필요합니다. 이를 통해 2D 이미지와 3D 서명 거리를 전처리하는 데 있어 효율적인 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존 신경 필드 필터링 방법들과 비교하여 정량적 및 정성적 개선을 나타냅니다. 훈련 중에도 신호 평가 비용을 최소화할 수 있으며, 결론적으로 다양한 선형 및 대칭 컨볼루션 필터에 일반화할 수 있는 전처리 접근 방식을 제공하여 더 높은 품질의 결과를 달성합니다.



### SViM3D: Stable Video Material Diffusion for Single Image 3D Generation (https://arxiv.org/abs/2510.08271)
Comments:
          Accepted by International Conference on Computer Vision (ICCV 2025). Project page: this http URL

- **What's New**: 이번 연구에서는 한 장의 이미지로부터 다중 뷰에서 일관된 물리 기반 렌더링(PBR) 재질을 예측하는 Stable Video Materials 3D(SViM3D) 프레임워크를 제시합니다. 기존 비디오 확산 모델들은 단일 이미지만으로 3D 객체를 재구성하는데 효율적으로 사용되고 있으나, 재질의 반사 특성은 종종 간단한 모델이나 추가 단계에서 추정해야합니다. SViM3D는 카메라 제어를 통해 공간적으로 변하는 PBR 매개변수와 표면 법선을 동시에 출력하는 첫 번째 모델로, 이는 그대로 조명 변경 및 3D 자산 생성을 가능하게 합니다.

- **Technical Details**: SViM3D는 객체 중심의 역 렌더링 문제를 해결하는 확률적 생성 확산 모델로, 고속 진단과 고품질 물체 외관 생성을 위해 카메라 포즈 시퀀스에 조건화하여 다중 뷰에서 일관된 재질 특성을 생성합니다. 물체의 자세한 재질 표현을 위해 기존의 이미지 + 재질 + 법선 생성을 위한 UNet 아키텍처를 수정하였으며, 이를 통해 실제 조명과 재질 변화의 복잡성을 포착한 고품질의 합성 데이터셋을 사용합니다. 또한, 다중 뷰 PBR 비디오 출력을 3D 재구성을 위한 의사 기준선으로 활용하여 균형 잡힌 재구성을 달성하기 위한 다양한 혁신을 도입하였습니다.

- **Performance Highlights**: SViM3D는 새로운 뷰 합성(NVS), 재질 예측, 조명 변경, 3D 생성 등에서 광범위한 평가를 진행하였으며, 다중 뷰 일관성을 뛰어넘는 성능을 보여주었습니다. 실제 환경에서의 재질 재현을 크게 개선하며, 이 접근법은 다중 뷰 외관 일관성을 효율적으로 이해하고 활용할 수 있음을 나타냅니다. AR/VR, 영화, 게임 등 다양한 시각 미디어에서 유용한 재조명 가능한 3D 자산 생성이 가능한 점에서 큰 의미를 갖습니다.



### Dual-granularity Sinkhorn Distillation for Enhanced Learning from Long-tailed Noisy Data (https://arxiv.org/abs/2510.08179)
Comments:
          25 pages, 2 figures

- **What's New**: 딥러닝 분야에서 클래스 불균형(class imbalance)과 레이블 노이즈(label noise)의 동시 존재로 인해 모델 성능이 저하되고 있습니다. 이 논문은 이러한 문제를 해결하기 위한 새로운 접근 방식을 제안하며, '약한(weak)' 보조 모델을 이용하여 서로 다른 문제에 대한 해결책을 종합적으로 활용하는 방법을 탐구합니다. 특히, 두 가지 문제는 서로 다른 수준에서 작용하기 때문에 각각에서의 강인성 메커니즘이 상호 보완적으로 작용할 수 있다는 통찰을 바탕으로 합니다.

- **Technical Details**: 이 논문에서 제안한 Dual-granularity Sinkhorn Distillation(D-SINK) 프레임워크는 보조 모델(auxiliary models)로부터 지식을 증류하고 통합하여 클래스 불균형과 레이블 노이즈에 대한 이중 강인성을 달성합니다. D-SINK는 최적 운송(Optimal transport) 기반의 대리 레이블 할당을 통해 타깃 모델의 예측을 노이즈에 강한 보조 모델과 샘플 수준에서 정렬하고, 클래스 분포는 불균형에 강한 보조 모델과 정렬합니다. 이를 통해 타깃 모델은 두 가지 문제를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: D-SINK는 다양한 노이즈 패턴과 비율을 가진 벤치마크 데이터셋에서 광범위한 실험을 수행하였으며, 상당한 성능 향상을 보여 주었습니다. 연구 결과, D-SINK는 약한 보조 모델의 지식을 통합함으로써 두 가지 도전 과제를 처리하는 데 있어 놀라운 효과를 발휘함을 확인했습니다. 이러한 접근 방식은 전통적인 방법론과는 다른 접급 방법을 제시하며, 복잡한 데이터 환경에서의 학습 성능을 효과적으로 향상시킬 수 있습니다.



### NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions (https://arxiv.org/abs/2510.08173)
- **What's New**: 이 연구에서는 내비게이션 에이전트의 공간적 인지를 평가하기 위한 새로운 벤치마크인 NavSpace를 소개합니다. NavSpace는 1,228개의 경로-명령 쌍으로 구성되어 있으며, 내비게이션 에이전트의 공간적 지능을 시험하는 여섯 가지 작업 카테고리를 포함합니다. 이 평가에서는 최첨단 내비게이션 모델인 SNav를 제안하며, 이는 기존 모델들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: NavSpace 벤치마크는 내비게이션 작업의 클래식한 정의를 따르며, 에이전트는 주어진 언어 명령에 따라 다음 내비게이션 동작을 예측해야 합니다. 연구팀은 설문조사를 통해 공간 인지에 필수적인 카테고리를 식별하였고, 수집된 데이터는 기계 학습 모델과 함께 사용됩니다. 이 연구는 22개의 기존 내비게이션 에이전트를 종합적으로 평가하였습니다.

- **Performance Highlights**: SNav는 NavSpace와 실제 로봇 테스트에서 기존 내비게이션 에이전트들보다 우수한 성능을 보였으며, 이는 향후 연구에 강력한 기준을 세우는 데 기여합니다. 실험 결과, 내비게이션 분야에서 공간적 지능의 중요성과 MLLM의 한계를 강조하며 내비게이션 대형 모델의 장점을 밝혀냈습니다. 이러한 통찰은 향후 연구에서 내비게이션 에이전트의 공간적 인지를 향상시키는 방향성을 제시합니다.



### MMM: Quantum-Chemical Molecular Representation Learning for Combinatorial Drug Recommendation (https://arxiv.org/abs/2510.07910)
Comments:
          Medical Image Computing and Computer-Assisted Intervention (MICCAI) Predictive Intelligence in Medicine Workshop (MICCAI PRIME) 2025; 13 pages

- **What's New**: 이번 연구에서는 약물 간 상호작용 예측을 위해 새롭게 제안된 프레임워크인 Multimodal DDI Prediction with Molecular Electron Localization Function (ELF) Maps (MMM)을 소개합니다. MMM은 약물 표현 학습에 3차원 양자 화학 정보를 통합하여 높은 예측 정확도와 안전한 약물 처방을 지원하는 가능성을 보여줍니다. 이 프레임워크는 약물의 전자 밀도 맵을 생성하여 치료적 관련성과 상호작용 위험을 동시 고려할 수 있도록 설계되었습니다.

- **Technical Details**: MMM은 세 가지 주요 구성 요소로 이루어져 있습니다: 환자의 전자 건강 기록(EHR)을 인코딩하는 모듈, ELF 기반 약물 인코더로 전자 상호작용 특성을 반영하며, 환자 조건에 기반하여 약물 하위 구조의 중요성을 추론하는 지역 이분법 인코더, 그리고 안전하고 효과적인 약물 처방을 도출하기 위한 약물 추천 모듈입니다. 이 프레임워크는 DFT(computational density functional theory)를 사용하여 ELF 맵을 구성하고, 고차원 특징을 추출하기 위해 사전 학습된 CNN(convolutional neural network)을 활용합니다.

- **Performance Highlights**: MMM은 MIMIC-III 데이터셋에서 여러 기본 모델과 비교하여 F1 점수, Jaccard 유사성 및 DDI 비율에서 통계적으로 유의미한 향상을 보였습니다. 특히, GNN 기반의 SafeDrug 모델과의 비교에서 F1-score(p = 0.0387)와 Jaccard(p = 0.0112), DDI 비율(p = 0.0386)에서 개선된 결과를 보여, MMM이 약물 추천의 정밀도를 높이고, 임상 실무에서의 안전한 조합 약물 처방을 지원할 수 있는 잠재력을 입증하였습니다.



### SatFusion: A Unified Framework for Enhancing Satellite IoT Images via Multi-Temporal and Multi-Source Data Fusion (https://arxiv.org/abs/2510.07905)
- **What's New**: 이번 연구에서는 Sat-IoT(위성 사물 인터넷) 이미지를 향상시키기 위한 통합 프레임워크인 SatFusion을 제안하고 있습니다. 이는 다중 시간 및 다중 소스 데이터를 활용하여 고품질 이미지를 재구성하는 혁신적인 방법입니다. 기존의 다중 이미지 초해상도(MISR) 기술이나 팬샤프닝 방식은 각기 분리되어 연구되어 왔지만, SatFusion은 이 두 가지 접근법을 통합하여 더 나은 성능을 목표로 하고 있습니다.

- **Technical Details**: SatFusion은 다중 시간 이미지 융합(Multi-Temporal Image Fusion, MTIF) 모듈을 사용해 팬크로매틱 이미지와의 깊은 특징 정렬을 수행합니다. 이어서 다중 소스 이미지 융합(Multi-Source Image Fusion, MSIF) 모듈이 팬크로매틱 데이터에서 미세한 질감 정보를 주입하여 이미지 품질 향상을 꾀합니다. 마지막으로, 융합 구성(Fusion Composition) 모듈은 다양한 손실 함수의 가중 조합을 통해 스펙트럼 일관성을 동적으로 정제하면서 두 가지 모달리티의 상호 보완적인 장점을 통합합니다.

- **Performance Highlights**: WorldStrat, WV3, QB, GF2 데이터셋에 대한 광범위한 실험을 통해 SatFusion은 이미지를 향상시키는 데 있어 기존 방법들에 비해 우수한 재구성 품질과 강건성을 입증하였습니다. SatFusion은 저품질 중복 데이터의 문제를 해결하면서 고품질 이미지를 생성하는 데 효율적인 해결책을 제시합니다. 이를 통해 Sat-IoT 시나리오에서 실제 적용 가능성을 높이며, 다양한 상황에 대응할 수 있는 능력을 갖추고 있습니다.



### FlowLensing: Simulating Gravitational Lensing with Flow Matching (https://arxiv.org/abs/2510.07878)
Comments:
          6 pages, 2 figures, 3 tables

- **What's New**: FlowLensing은 강력한 중력 렌즈 이미지를 시뮬레이션하기 위한 Diffusion Transformer 기반의 새로운 모델을 제공합니다. 이 모델은 기존의 ray-tracing 기반 도구와 비교해 속도와 효율성에서 큰 개선을 보여주며, 다양한 암흑 물질 모델을 처리할 수 있습니다. FlowLensing은 사실적이고 물리적으로 일관된 이미지를 제공하여 암흑 물질 연구의 발전을 촉진할 수 있습니다.

- **Technical Details**: FlowLensing은 연속시간 생성 모델링 프로세스로, 간단한 prior distribution에서 목표 데이터 배포로의 변환을 학습합니다. 기존의 diffusion 모델과는 달리, flow matching은 노이즈를 두는 과정을 피하고 변환 과정을 안내하는 속도장을 직접 학습하여 성능을 향상시킵니다. 이 모델은 강한 암흑 물질 시나리오와 정밀한 렌즈 속성을 포착하여 물리적으로 일관된 이미지를 생성합니다.

- **Performance Highlights**: FlowLensing은 기존 시뮬레이터와 비교하여 200배 이상의 속도를 달성하면서도 높은 충실도와 낮은 추론 지연 시간을 확보했습니다. 이미지 품질 측면에서 Mean Squared Error(MSE), Peak Signal-to-Noise Ratio(PSNR) 및 Structural Similarity Index(SSIM)와 같은 다양한 메트릭을 사용하여 평가하고 있으며, 네트워크의 성능을 크게 향상시키고 있습니다. 이러한 뛰어난 성과는 FlowLensing이 전통적인 Forward-모델링 파이프라인에 대한 실용적인 대안이 될 수 있음을 시사합니다.



### Team Xiaomi EV-AD VLA: Learning to Navigate Socially Through Proactive Risk Perception -- Technical Report for IROS 2025 RoboSense Challenge Social Navigation Track (https://arxiv.org/abs/2510.07871)
- **What's New**: 이 보고서는 IROS 2025 RoboSense Challenge의 Social Navigation Track에 대한 기술 세부 정보를 설명합니다. 이 트랙은 RGB-D 기반 인식 및 내비게이션 시스템을 개발하는 데 중점을 두며, 자율 에이전트가 동적 인간이 있는 실내 환경에서 안전하고 효율적으로 탐색할 수 있도록 돕습니다. 새로운 Proactive Risk Perception Module을 도입하여 사회적 내비게이션 성능을 향상시켰습니다.

- **Technical Details**: 사회적 내비게이션(Social Navigation)은 자율 로봇이 인간과 공유하는 환경에서 사회적 관습을 준수하며 탐색하는 능력을 나타냅니다. 이 연구는 Falcon 모델을 기반으로 하여, 상대 거리 정보를 활용해 인간과의 충돌 위험을 예측하는 모듈을 통합했습니다. 이는 충돌 회피 행동을 개선하고 공간 인식을 강화하는 데 기여합니다.

- **Performance Highlights**: Social-HM3D 벤치마크에서 우리의 방법이 밀집한 실내 장면에서 목표를 향해 탐색할 때 개인 공간 준수를 유지하는 능력을 개선했음을 보였습니다. 이 챌린지에서 16개 팀 중 2위를 달성하였으며, 이는 자율 탐색 시스템의 발전에 중요한 이정표가 될 것입니다.



### IntentionVLA: Generalizable and Efficient Embodied Intention Reasoning for Human-Robot Interaction (https://arxiv.org/abs/2510.07778)
- **What's New**: 본 연구에서는 IntentionVLA라는 새로운 Vision-Language-Action(VLA) 프레임워크를 제안합니다. 이 모델은 훈련과정에서 효율적인 데이터 주석을 통해 고차원적인 인간의 의도를 해석하며, 이를 통해 빠른 추론 능력을 갖추게 됩니다. IntentionVLA는 복잡한 물리적 환경에서 적절한 행동을 수행하기 위한 두 가지 중요한 도전에 대응합니다. 이를 통해 기존의 VLA 모델들이 겪었던 한계를 극복하고자 합니다.

- **Technical Details**: IntentionVLA는 세 가지 보완적인 형태로 표현된 주석 데이터를 사용합니다: intention reasoning(의도 추론), spatial reasoning(공간적 추론), compact reasoning(간결한 추론). 첫 단계에서 모델은 주석 데이터에서 추론 및 인식 능력을 키우고, 두 번째 단계에서는 이러한 고차원적인 추론이 압축된 단서를 통해 diffusion-based action generator(확산 기반 행동 생성기)로 행동 생성을 안내합니다. 이러한 접근 방식은 논리적 추론과 행동 실행을 명확히 결합하여 고차원적으로 인간의 의도를 이해할 수 있도록 합니다.

- **Performance Highlights**: IntentionVLA는 기존 SOTA VLA 모델들에 비해 모든 평가 환경에서 우수한 성능을 입증했습니다. 직접적인 지침에서는 18% 높은 성공률을, 의도 지시에 대해선 ECoT보다 28% 높은 성공률을 기록했습니다. 더불어, out-of-distribution(배포 외) 의도 작업에서는 모든 기준선 모델들에 대해 두 배 이상의 성공률을 달성하였으며, 40%의 성공률로 제로샷 인간-로봇 상호작용을 가능하게 하였습니다. 이러한 결과들은 IntentionVLA가 차세대 인간-로봇 상호작용 시스템에 대한 유망한 패러다임임을 강조합니다.



### Curriculum Learning with Synthetic Data for Enhanced Pulmonary Nodule Detection in Chest Radiographs (https://arxiv.org/abs/2510.07681)
Comments:
          32 pages, 6 figures,

- **What's New**: 이 연구는 커리큘럼 학습(curriculum learning)과 확산 기반의 합성 증강(diffusion-based synthetic augmentation)을 통합하여 흉부 방사선 사진에서 어려운 폐 결절을 더 효과적으로 탐지할 수 있는지 평가합니다. 기존 AI 모델이 데이터 불균형과 제한된 주석으로 인해 어려움을 겪는 저 사이즈, 저 밝기, 저대비 결절에 중점을 두고 새로운 접근 방식을 제시합니다. 제안된 모델은 Faster R-CNN을 기반으로 하며, 특히 작은 결절에 대한 탐지 성능이 강조됩니다.

- **Technical Details**: Faster R-CNN과 Feature Pyramid Network(FPN)를 기반으로 한 모델이 전문가가 주석을 단 NODE21 데이터셋 및 기타 공개 데이터셋, 그리고 DDPM에서 생성된 11,206개의 합성 이미지를 포함한 혼합 데이터셋에 대해 학습되었습니다. 어려움 점수는 결절의 크기, 밝기 및 대비에 따라 설정되었으며, 모델은 5겹 내부 교차 검증을 통해 평가되었습니다. 훈련을 위한 커리큘럼 학습 전략이 도입되어 점진적으로 더 어려운 샘플로 성장했습니다.

- **Performance Highlights**: 모델은 평균 AUC(Area Under Curve) 0.95를 달성하여 기존 기준 모델의 0.89에 비해 유의미한 향상을 나타냈습니다(p < 0.001). 민감도(sensitivity)는 70%로 증가했으며, 정확도(accuracy)는 82%로 향상되었습니다. 모든 어려움 범주에서 일관된 성과를 보였고 Grad-CAM 시각화 결과는 커리큘럼 학습 하에서 모델의 해부학적 초점이 더 명확해짐을 보여주었습니다.



### Test-Time Matching: Unlocking Compositional Reasoning in Multimodal Models (https://arxiv.org/abs/2510.07632)
- **What's New**: 이번 연구는 AI 모델의 compositional reasoning(구성적 추론) 문제를 재조명하며, 표준 평가 메트릭이 모델의 능력을 과소 평가한다는 점을 보여줍니다. 이를 해결하기 위해 그룹 구조를 더 잘 활용하는 그룹 매칭 점수(Group Matching Score)를 도입하여, 기존의 지표에서는 발견할 수 없는 모델의 숨은 능력을 드러냅니다. 연구 결과, SigLIP-B16과 GPT-4.1은 이전 모든 결과를 초월하는 성과를 이뤘습니다.

- **Technical Details**: 신선한 접근법으로 Test-Time Matching (TTM)이라는 반복적이고 자기 개선 가능한 알고리즘을 제안하였습니다. 이 알고리즘은 매칭 기반의 의사 레이블을 선택하여 자기 학습을 진행하고, 점진적으로 선택 기준을 완화하여 테스트 데이터셋에 대한 범위를 확장합니다. 이로 인해 SigLIP-B16과 GPT-4.1은 여러 벤치마크에서 놀라운 성능 향상을 보였습니다.

- **Performance Highlights**: TTM을 통해 SigLIP-B16은 Winoground에서 72.5, MMVP-VLM에서 89.44, ColorSwap에서 94.25의 성과를 기록하며 새로운 최첨단 결과를 세우고 있습니다. 특히 도전적인 데이터셋인 WhatsUp에서는 최대 85.7%의 상대적 성과 향상이 이루어졌습니다. 연구에서 TTM은 평가 메트릭의 변화를 극복하며 일관되게 모델 성능을 향상시키는 데 효과적입니다.



### MLLM4TS: Leveraging Vision and Multimodal Language Models for General Time-Series Analysis (https://arxiv.org/abs/2510.07513)
- **What's New**: 이번 연구에서는 MLLM4TS라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 멀티모달 대형 언어 모델(multimodal large language models)을 활용하여 시계열(time series) 분석을 수행합니다. 특히, 눈에 띄는 시각적 표현을 통해 시계열 데이터를 분석하는 자동화된 방법의 효과를 높일 수 있는지 탐구합니다.

- **Technical Details**: MLLM4TS는 시간을 나타내는 시리즈를 색상 코드가 있는 선 그래프(color-coded line plot) 형태로 변환하여 시각적 패턴을 캡처합니다. 이를 통해 각 채널 간의 공간 의존성을 효과적으로 파악하고, 각 시간 구간에 맞춰 시각적 패치를 정렬하는 전략을 도입합니다. 이러한 방법은 수치적 데이터와 시각적 표현에서 파생된 글로벌 맥락 정보를 융합하여 정교한 시간 역학을 모델링합니다.

- **Performance Highlights**: MLLM4TS는 분류(classification), 이상 탐지(anomaly detection), 예측(forecasting) 등 다양한 표준 벤치마크에서 좋은 성과를 보여주었습니다. 이 연구는 사전 훈련된 언어 모델(pretrained language models)과 시각적 모달리티(visual modalities)를 통합함으로써 시계열 분석의 강력하고 일반화 가능한 접근법을 제시합니다. 특히, 몇 샷(few-shot) 및 제로 샷(zero-shot) 학습 환경에서도 뛰어난 일반화 성능을 지니고 있습니다.



### ConCuR: Conciseness Makes State-of-the-Art Kernel Generation (https://arxiv.org/abs/2510.07356)
- **What's New**: 본 연구는 LLM을 활용한 GPU 커널 생성의 최근 발전을 다룹니다. 고급 CUDA 커널의 부족 문제를 해결하기 위해, 우리는 reasoning trace(추론 추적)를 포함한 고품질 CUDA 커널을 생성 및 큐레이팅하는 파이프라인을 개발했습니다. 이를 통해 ConCuR이라는 데이터셋과 KernelCoder라는 모델을 처음으로 소개합니다.

- **Technical Details**: 우리가 개발한 데이터 수집 및 큐레이션 파이프라인은 두 가지 부분으로 구성됩니다: 데이터 합성 및 데이터 큐레이션입니다. 데이터 합성 부분에서는 기존의 reasoning 모델을 사용하여 CUDA 커널과 함께 reasoning trace를 합성합니다. 데이터 큐레이션 부분에서는 reasoning trace의 간결성과 커널의 성능을 고려하여 높은 품질의 데이터셋을 구축합니다.

- **Performance Highlights**: KernelCoder는 ConCuR 데이터셋을 기반으로 훈련되어, 기존의 최상위 모델인 QwQ-32B에 비해 뛰어난 성능을 보입니다. KernelBench에서의 평가를 통해, KernelCoder가 모든 기존 커널 생성 모델보다 우수하며, GPT-4와 같은 최신 모델보다도 더 좋음을 증명했습니다. 추가적으로, reasoning length(추론 길이)가 커널 생성 작업의 복잡도를 평가하는 지표로 유용하다는 사실을 발견하였습니다.



### MultiFair: Multimodal Balanced Fairness-Aware Medical Classification with Dual-Level Gradient Modulation (https://arxiv.org/abs/2510.07328)
Comments:
          10 Pages

- **What's New**: 이 논문은 의료 분류를 위한 새로운 접근 방식인 MultiFair를 제안하며, 이는 다양한 데이터를 동시에 다루는 과정에서 생길 수 있는 공정성과 비균형 문제를 해결합니다. 기존의 다중 모달 학습 모델들이 두 가지 주요 문제, 즉 모달리티 학습 불균형(Modality Learning Bias)과 인구 통계학적 학습 불균형(Demographic Learning Bias)을 간과하고 있음을 지적합니다. MultiFair는 이러한 문제를 두 가지 층의 그래디언트 조절(Dual-level Gradient Modulation) 프로세스를 통해 해결합니다.

- **Technical Details**: MultiFair 모델은 훈련 그래디언트를 모달리티와 그룹 수준에서 최적화합니다. 이 모델은 각 모달리티의 기여도를 동적으로 조정하여, 전반적인 배치(Training Batch)에서 발생할 수 있는 불균형한 학습을 완화합니다. 논문에서는 MultiFair의 이론적 기반과 함께 실제 의료 다중 모달 데이터 셋을 활용한 광범위한 실험 결과도 제공합니다.

- **Performance Highlights**: 실험 결과, MultiFair는 기존의 최신 다중 모달 학습 및 공정성 학습(Fairness Learning) 방법들을 초월하며 성능을 보였습니다. 특히, 다양한 인구 통계 그룹에 대한 성능을 균형적으로 유지하면서도 진단의 정확도를 높이는 특징을 보여줍니다. 이는 의료 AI의 공정성을 확보하는데 기여할 것으로 기대됩니다.



### Deep Learning Based Approach to Enhanced Recognition of Emotions and Behavioral Patterns of Autistic Children (https://arxiv.org/abs/2510.07320)
- **What's New**: 이번 연구에서는 자폐 스펙트럼 장애(ASD) 아동의 정서 인식과 행동 패턴을 중심으로 한 맞춤형 교육 전략의 필요성을 강조합니다. 특히, ASD 아동의 감정 상태를 정확히 인식하는 것이 맞춤형 개입 및 사회적 지원을 위한 기초가 됨을 설명하며, 최신 AI 기술인 autoencoder를 통해 감정 인식의 정확성을 향상시키는 방법을 제안합니다. 이를 통해 ASH 교육과 기술 지원을 위한 발달 경로를 더욱 원활하게 할 수 있도록 합니다.

- **Technical Details**: 연구 방법론에서는 Xception 및 InceptionV3 모델을 활용하여 ASD 아동의 정서 인식을 위한 데이터 전처리에 대한 접근 방식을 상세히 설명합니다. 특히, 각기 다른 크기의 이미지 입력을 299×299×3의 고정된 크기로 변환하기 위해 autoencoder를 사용하여 노이즈를 줄이고 필수적인 안면 특징을 보존합니다. 깊이별 분리 가능한 합성곱(depthwise separable convolutions) 기술을 사용하여 모델의 계산 복잡성을 줄이고, 다중 스케일의 특징 추출을 통해 다양한 정서 표현을 인식합니다.

- **Performance Highlights**: 연구 결과, autoencoder를 통합한 정서 인식 시스템이 전통적인 방법론에 비해 더 뛰어난 성능을 보여주었습니다. baseline 방법과 비교했을 때, 향상된 접근 방식은 정확도, 정밀도, 재현율, F1-score에서 유의미한 개선을 보였으며, 이를 통해 ASD 아동의 감정 인식 정확성을 높이고자 하는 목적을 달성했습니다. 또한, 두 단계의 훈련 전략을 통해 모델을 더욱 효과적으로 최적화함으로써, 복잡한 실제 환경에서도 우수한 성능을 발휘하는 것을 입증했습니다.



### DUA-D2C: Dynamic Uncertainty Aware Method for Overfitting Remediation in Deep Learning (https://arxiv.org/abs/2411.15876)
Comments:
          This version (v2) extends our previous work (arXiv:2411.15876v1) on Divide2Conquer (D2C) by introducing Dynamic Uncertainty-Aware Divide2Conquer (DUA-D2C). The manuscript is currently under review at Complex and Intelligent Systems

- **What's New**: 이번 연구에서는 Dynamic Uncertainty-Aware Divide2Conquer (DUA-D2C) 기법을 소개합니다. DUA-D2C는 기존의 Divide2Conquer (D2C) 방식을 기반으로 하여, 모델들이 보유한 성능에 따라 유연하게 가중치를 부여하는 새로운 집계 방식을 도입합니다. 이러한 접근 방식은 데이터의 아웃라이어(Outlier)와 잡음(Noise)의 영향을 최소화하면서도 더 일반화된 패턴을 학습하는 데 기여합니다.

- **Technical Details**: DUA-D2C는 각 모델의 성능을 공유 검증 세트(Validation Set)에서 평가하여 가중치를 할당합니다. 이 방법은 예측 정확도(Accuracy)와 예측 불확실성(Prediction Uncertainty)을 고려하여, 덜 신뢰할 수 있는 모델의 영향을 줄이는 데 초점을 맞춥니다. 연구에서는 수학적 정당성을 바탕으로 D2C가 과적합(Overfitting)을 감소시킨다는 점을 강조하며, DUA-D2C의 집계 과정이 더욱 정교해졌음을 설명합니다.

- **Performance Highlights**: 실험을 통해 DUA-D2C는 다양한 데이터 세트에서 모델의 일반화 성능을 크게 향상시킴을 입증하였습니다. 특히, DUA-D2C는 기존의 데이터 증대(Data Augmentation) 및 정규화(Regularization) 기법과 결합하여 성능 개선에 기여할 수 있는 가능성을 보여줍니다. 연구 결과는 모델이 더욱 신뢰할 수 있는 의사 결정을 내릴 수 있도록 돕는다는 점에서 중요한 의미를 가집니다.



### Context Matters: Learning Global Semantics via Object-Centric Representation (https://arxiv.org/abs/2510.05674)
- **What's New**: 이번 연구에서는 기존 비전 모델들이 자연어 처리 모델처럼 유의미한 추론(reasoning) 및 컨텍스트 학습(in-context learning) 능력을 갖추지 못한 이유를 탐구합니다. 비전 트랜스포머(ViT) 훈련 프로세스에서 의미적 지침의 부족이 주요 원인으로 지적되며, 이를 개선하기 위한 의미 기반 목표(semantic-grounded objective)를 제안합니다. 이 연구는 '객체(object)'를 '단어(word)'의 시각적 동등체로 모델링하여 비전 모델의 전반적인 의미적 인식(global contextual awareness)과 사고 능력을 향상시키고자 합니다.

- **Technical Details**: 연구는 일반적인 Masked Image Modeling (MIM) 프레임워크를 통해 진행되며, 무작위 패치를 마스킹하는 대신 전체 객체를 마스킹합니다. 이를 통해 모델이 객체 기반의 단서 없이도 전역적인 의미를 추론하도록 유도합니다. 연구진은 전통적인 ViT 훈련에서는 명시적인 의미 지침의 부족을 강조하면서, 이러한 간단한 목표가 비전 모델들의 실제적인 의미 분포를 학습하는 데 중요한 역할을 한다고 주장합니다.

- **Performance Highlights**: 주요 실험 결과는 시각적 질문 응답(Visual Question Answering, VQA) 작업에서 향상된 추론 능력을 포함하고 있으며, 더 나아가 다중 모달 LLM(multimodal LLM)과의 결합에서도 긍정적인 성과를 보였습니다. 주요 발견은 객체 수준의 표현(object-level representation)이 실제 세계의 의미 분포를 학습하는데 기여하며, 픽셀 평균화(pixel-averaging)와 같은 단순한 단축키를 사용하는 모델과의 명확한 차이를 나타냅니다. 연구의 결과는 비전 인코더(potential vision encoders)와 토크나이저(tokenizer) 개발을 위한 유망한 방향을 제시합니다.



New uploads on arXiv(cs.AI)

### How to Teach Large Multimodal Models New Skills (https://arxiv.org/abs/2510.08564)
Comments:
          In submission. Code is available at this https URL

- **What's New**: 이 논문은 대규모 다중모달 모델(large multimodal models, LMMs)에서 새로운 기술을 가르칠 때 이전 능력을 지키는 방법에 대해 연구합니다. 특히, 다섯 가지 목표 기술을 순차적으로 미세 조정(sequential fine-tuning)하면서, 세 모델 계열에 걸쳐 여덟 가지 기준 벤치마크에서 일반 능력을 모니터링합니다. 연구 결과, 좁은 범위의 미세 조정을 한 후 일부 기준 작업에서 '망각(forgetting)'이 관찰되지만, 후속 단계에서 이러한 영속적이지 않은 기억이 어느 정도 회복됨을 발견했습니다.

- **Technical Details**: 우리는 '망각'을 초래하는 결과 분포의 변화를 측정할 수 있는 간단한 카운팅 편향 탐지(counting-bias probe)를 통해 추적합니다. 이 연구에서는 두 가지의 간단하고 견고한 조정 레시피(tuning recipes)를 제안합니다: (i) 자기 주의(self-attention) 투영 레이어만 업데이트하고, (ii) Down 투영을 고정(freezing)한 상태에서 MLP Gate&Up만 업데이트하는 방법입니다.

- **Performance Highlights**: 이러한 조정 방법을 통해 모델과 작업 전반에 걸쳐 강력한 목표 성과(target gains)를 달성하면서도 이전 기준 성능(held-out performance)을 대체로 유지할 수 있음을 보여줍니다. 제안된 방법은 각 모델과 작업에서 향상된 결과를 제공하며, 연구 결과는 실제 코드로도 확인할 수 있습니다.



### Agent Learning via Early Experienc (https://arxiv.org/abs/2510.08558)
Comments:
          Work in progress

- **What's New**: 이번 연구는 언어 에이전트들이 스스로 경험을 통해 학습하고 향상될 수 있는 가능성을 탐구합니다. 특히, 방식이 부족한 환경에서의 강화 학습의 한계를 극복하기 위해 'early experience'(얼리 익스피리언스)라는 새로운 개념을 도입했습니다. 이는 에이전트의 행동으로 생성된 상호작용 데이터로, 보상 신호 없이도 미래 상태를 감독할 수 있게 합니다.

- **Technical Details**: 연구에서는 두 가지 전략을 통해 초기 경험 데이터를 활용합니다. 첫 번째는 'Implicit world modeling'(암묵적 세계 모델링)으로, 수집된 상태를 사용하여 정책을 환경 동역학에 맞게 조정합니다. 두 번째는 'Self-reflection'(자기 반성)으로, 에이전트가 비효율적인 행동에서 학습하여 추론과 의사결정을 개선하는 방법입니다.

- **Performance Highlights**: 8개의 다양한 환경과 여러 모델 계열을 통해 평가한 결과, 제안된 접근 방식이 효과성과 도메인 외 일반화를 일관되게 개선함을 보여주었습니다. 더욱이, 검증 가능한 보상이 있는 환경에서는 초기 경험이 후속 강화 학습을 위한 강력한 기초가 될 수 있다는 유망한 신호를 포착했습니다.



### FlowSearch: Advancing deep research with dynamic structured knowledge flow (https://arxiv.org/abs/2510.08521)
- **What's New**: FlowSearch는 복잡한 문제 해결을 위한 동적 구조화 지식 흐름(dynamic structured knowledge flow)을 제안하여, 연구제 수행 및 논리를 명확히 인코딩할 수 있게 합니다. 기존의 연구 시스템들이 개인 연구자 행동을 모방하거나 역할 분담을 통해 확장성을 도모했지만, 이들 모두는 탐색의 폭과 깊이 간의 내재적 균형의 한계를 가지고 있었습니다.

- **Technical Details**: FlowSearch는 지식 흐름 계획자(Knowledge Flow Planner), 지식 생서기(Knowledge Collector), 지식 흐름 조정기(Knowledge Flow Refiner)라는 세 가지 핵심 구성 요소로 이루어져 있습니다. 각 구성 요소는 과제를 수행하고 중간 산출물에 따라 흐름을 조정하며, 모든 단계에서 체계적이고 효율적인 문제 해결을 지원합니다. 이러한 동적인 그래프 구조는 서로 다른 작업 간의 의존성을 효과적으로 표현할 수 있도록 도와줍니다.

- **Performance Highlights**: FlowSearch는 GAIA, HLE, GPQA, TRQA와 같은 여러 벤치마크에서 최첨단 성능을 달성했습니다. 특히 GAIA에서는 AI 어시스턴트의 일반적인 문제 해결 능력을 평가한 결과 뛰어난 결과를 보였으며, 다학제적인 과학 질문 응답(Multi-disciplinary scientific question-answering)에서도 우수한 성과를 입증했습니다.



### CaRT: Teaching LLM Agents to Know When They Know Enough (https://arxiv.org/abs/2510.08517)
- **What's New**: 이 논문은 LLMs(대규모 언어 모델)가 정보 검색을 언제 중단해야 할지 학습할 수 있도록 돕는 새로운 접근법인 CaRT(Counterfactuals and Reasoning for Termination)를 소개합니다. 정보 검색 단계에서 효과적인 의사결정을 위해 CaRT는 반사실적 데이터를 이용하여 잘못된 결정과 올바른 결정을 비교하여 모델을 학습시킵니다. 이 방법은 상호작용 의료 진단 및 수학 문제 해결 분야에서 구현되었습니다.

- **Technical Details**: CaRT는 반사실적 쌍(counterfactual pairs)을 통해 LLM을 미세 조정(fine-tuning)하여 정보 수집을 중단할 적절한 시점을 학습하도록 합니다. 모델은 음성 추론(verbal reasoning)을 통해 종료 결정을 설명하도록 훈련되며, 이는 최종 LLM 모델에 내재화됩니다. 특히 Qwen3-1.7B-Instruct와 Qwen2.5-3B-Instruct 모델에서 이 방법의 효과성을 입증했습니다.

- **Performance Highlights**: 실험 결과, CaRT는 정보 검색의 효율성과 작업 성공률을 기존의 미세 조정 방법보다 향상시키는 것으로 나타났습니다. 의료 진단 및 수학 문제 해결 모두에서 CaRT 사용 시 성공률이 개선되었습니다. 특히, Qwen3-1.7B-Instruct에서 반사실적 요소를 제거하면 성과가 저하되어 CaRT의 중요성이 강조됩니다.



### AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents (https://arxiv.org/abs/2510.08511)
- **What's New**: 이번 논문에서는 AutoMLGen이라는 LLM 기반의 코딩 에이전트를 소개합니다. 이 에이전트는 도메인 지식 데이터를 통합하여 높은 품질의 사전 가이드를 제공하고, 몬테 카를로 그래프 탐색(Monte Carlo Graph Search, MCGS) 알고리즘을 통해 효율적인 탐색을 가능하게 합니다. MCGS는 트리 기반 탐색의 장점을 유지하면서 그래프 구조를 결합하여 동적으로 경로를 재조정하고, 과거의 경로를 재사용하며, 다중 솔루션을 융합할 수 있도록 합니다.

- **Technical Details**: AutoMLGen은 코딩 에이전트로, 머신 러닝 엔지니어링(Machine Learning Engineering, MLE) 작업을 위한 지식 기반과 MCGS를 통합합니다. 이를 통해 다양한 모델 및 데이터 차원에서 도메인 사전 지식을 제공하며, 탐색 과정에서 세밀한 개선을 지원합니다. MCGS는 기존의 MCTS(Monte Carlo Tree Search)의 변형으로, 확장 단계에서 그래프 구조를 포함하여 고유한 해결책의 재조합과 통합을 가능하게 합니다.

- **Performance Highlights**: MLE-Bench에서의 평가 결과, AutoMLGen은 12시간의 예산 내에서 평균 36.4%의 메달 비율을 달성하며 기존의 모든 기준선을 능가하는 성능을 보였습니다. 이 에이전트는 안정적인 탐색 및 실행 가능성을 높이기 위해 세분화된 운영자 집합을 설계하여 최적의 ML 파이프라인 생성을 자동화합니다. AutoMLGen의 도입으로 MLE 작업에서의 성능이 획기적으로 향상되었습니다.



### Looking to Learn: Token-wise Dynamic Gating for Low-Resource Vision-Language Modelling (https://arxiv.org/abs/2510.08470)
Comments:
          Accepted to the EMNLP 2025 BabyLM Workshop

- **What's New**: 이 연구에서는 BabyLM Challenge 2025 비전 트랙의 요구 사항에 맞추어 경량화된 디코더 기반 아키텍처를 제안합니다. 이 아키텍처는 언어적 정보와 시각적 정보를 적응적으로 융합하기 위한 동적 게이팅(token-wise dynamic gating) 메커니즘, 제한된 시각적 정보의 효용을 극대화하기 위한 특징 조정 및 채널 주의(feature modulation and channel attention), 그리고 시각적 기초를 확보하기 위한 보조 대조 목표(auxiliary contrastive objectives)를 포함하고 있습니다. 이를 통해 수치적으로 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: 논문에서는 동적 게이팅 메커니즘을 채택하여 각 토큰에 대한 시각적 신호와 언어적 신호의 가중치를 맥락에 따라 선택적으로 조정합니다. 또한, 특징 향상 기술을 활용하여 제한된 시각적 정보의 유용성을 극대화하고, 문장 및 단어 수준에서 작동하는 대조 학습 보조 목표의 영향을 탐구합니다. 평가 데이터셋에서는 정보 병목 문제와 데이터셋 분할로 인해 훈련 불안정성이 발생할 수 있음을 지적합니다.

- **Performance Highlights**: 평가 결과, 제안된 프레임워크는 총 다섯 개의 벤치마크에서 경쟁력 있는 성능을 보였습니다. 특히, 동적 게이트가 언어의 품사에 따라 시각적 신호와 언어적 신호의 가중치를 조정하며, 내용 단어에는 시각적 신호를, 기능 단어에는 언어적 신호를 더 우선시한다는 것을 발견했습니다. 이러한 결과는 인지 기반 학습의 영감을 얻어 비전-언어 모델의 발전에 기여할 수 있는 가능성을 제시합니다.



### Revisiting Hallucination Detection with Effective Rank-based Uncertainty (https://arxiv.org/abs/2510.08389)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 환각(Hallucinations) 탐지를 위해 새로운 방법론을 제안합니다. 기존의 불확실성 기반 프레임워크를 넘어, 여러 모델 출력과 다양한 레이어에서 파생된 숨겨진 상태의 유효 순위(effective rank)를 측정하여 불확실성을 정량화하는 효율적이고 강력한 방법을 소개합니다. 이 접근법은 모델의 내부 추론 과정을 해석할 수 있는 통찰을 제공하며, 추가적인 지식이나 모듈이 필요하지 않습니다.

- **Technical Details**: 제안한 방법에서는 숨겨진 상태의 유효 순위를 사용하여 모델의 불확실성을 정량화합니다. 유효 순위는 다차원 임베딩 벡터의 분산을 측정하는 지표로 사용되며, 낮은 유효 순위는 모델이 결정적이며 자신감 있는 상태를 나타내고, 높은 유효 순위는 불확실성과 혼란을 나타냅니다. 이러한 방법론은 LLM의 내부 표현을 활용하여 환각을 효과적으로 탐지할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험을 통해, 제안한 방법이 기존의 강력한 기준선과 비교하여 우수한 성능을 보임을 입증하였습니다. 이 방식은 스케일이 크고 효율적이며, 다양한 시나리오에 대해 강건하게 일반화될 수 있음을 확인하였습니다. 또한, 알레아토릭 불확실성이 모델의 내부 표현에서 에피스템적 불확실성을 점진적으로 증가시키고 가리는 현상도 관찰되었습니다.



### QAgent: A modular Search Agent with Interactive Query Understanding (https://arxiv.org/abs/2510.08383)
Comments:
          Code is available at this https URL

- **What's New**: 이번 논문에서는 QAgent라는 통합된 에이전틱 RAG 프레임워크를 제안하여, 기존의 retrieval-augmented generation(RAG) 방식의 한계를 극복하고자 합니다. 이 프레임워크는 복잡한 질의를 이해하고 적응형 검색을 통해 외부 정보를 통합하여 자연어 작업을 개선합니다. 또한, QAgent는 다단계 결정을 통해 정보를 최적화할 수 있는 경량화된 검색 에이전트를 활용합니다.

- **Technical Details**: QAgent는 질의 이해, 추론 및 반복적 정제를 통합하는 일관된 검색 프로세스를 통해 작동합니다. 이는 다수의 상호작용을 통해 복잡한 사용자 의도를 점진적으로 정제함으로써 이루어집니다. 에이전트는 강화 학습(reinforcement learning) 방식으로 훈련되어 정보 검색 결과의 품질을 최대화하며, 전체 시스템 성능을 향상시키는 서브 모듈로 작동합니다.

- **Performance Highlights**: 실험 결과, QAgent는 질의 응답(Question Answering) 작업에서 뛰어난 성능을 보이며 실제 응용에 쉽게 통합 가능한 모듈로서의 가능성을 확인했습니다. QAgent는 지속적인 피드백을 통해 질의 최적화를 통합하고, 성과를 확대할 수 있는 전략을 제공하여 자연어 처리 분야에서의 발전에 기여하고 있습니다.



### LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings (https://arxiv.org/abs/2510.08338)
Comments:
          28 pages, 35 figures

- **What's New**: 이 논문에서는 소비자 연구에서의 비용 문제를 해결하기 위해 대형 언어 모델(LLM)을 활용한 새로운 접근법을 제안합니다. 제안된 방법인 Semantic Similarity Rating (SSR)은 LLM에서 텍스트 응답을 유도하고, 이를 기준 문장과의 임베딩 유사성을 통해 Likert 분포로 매핑합니다. 57개의 개인 관리 제품 설문조사를 사용하여 SSR이 90%의 인간 테스트-재테스트 신뢰성을 달성했음을 보여줍니다.

- **Technical Details**: SSR은 LLM이 생성한 자유 텍스트 문장을 5점 Likert 척도로 매핑하기 위해 코사인 유사성을 사용합니다. 이전 연구의 방법론을 기반으로 하여 LLM의 응답에서 인구 통계적 특성을 고려한 기능을 추가하였습니다. 이를 통해 LLM이 인간 응답자와 유사한 패턴의 응답을 생성할 수 있음을 입증했습니다.

- **Performance Highlights**: SSR을 통해 생성된 가상 응답자는 제품 컨셉에 대해 양질의 피드백을 제공할 수 있으며, 이는 추가적인 질적 분석을 촉진합니다. 연구 결과, SSR 방법이 전통적인 소비자 연구 방법의 한계를 극복하고, 대규모 소비자 연구 시뮬레이션을 가능하게 만든다는 것을 보여주었습니다. LLM의 응답 행동은 특히 연령 및 소득 수준에서 실제 인간과 유사함을 발견했습니다.



### Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries (https://arxiv.org/abs/2510.08325)
Comments:
          10 pages, 3 figures

- **What's New**: 강화학습을 통한 검증 가능한 보상(Reinforcement Learning with Verifiable Rewards, RLVR)은 대규모 언어 모델의 이론적 문제 해결 능력을 향상시키는 데 중요한 패러다임으로 자리 잡고 있습니다. 최근 연구에서 RLVR 모델의 이론적 경계가 실제로 확장되는지에 대해 의문이 제기되었습니다. 본 논문에서는 Pass@k 지표가 문제의 신뢰도를 고려하지 않기 때문에, 보다 유용한 대안으로 Cover@tau 지표를 제안합니다.

- **Technical Details**: Cover@tau는 모델이 지식을 바탕으로 문제를 해결할 확률율을 τ 이상으로 설정하여 문제 해결 능력을 측정합니다. 이 메트릭은 Pass@k와 달리 무작위 추측에 의한 성능 저하가 발생하는 것을 방지합니다. 제안된 지표는 RLVR 모델을 평가하는 데 있어 서로 다른 신뢰성 수준을 적용해 보는 새로운 접근법을 제공합니다.

- **Performance Highlights**: 여러 RLVR 모델을 평가한 결과, Cover@tau 지표는 Pass@1 또는 Pass@k와 비교했을 때 서로 다른 알고리즘의 상대적인 순위를 제공합니다. 이는 모델의 능력에 대한 새로운 관점을 제시하며, Pass@k 지표가 편향된 성능 수치를 보여줄 수 있다는 점을 강조합니다. 이러한 평가를 통해 RLVR 방법론의 수학적 추론 능력이 보다 정확히 분석되었습니다.



### First Try Matters: Revisiting the Role of Reflection in Reasoning Models (https://arxiv.org/abs/2510.08308)
- **What's New**: 최근에 대형 언어 모델(LLMs)은 물론 명확한 반영의 기여가 불명확했지만 사고 능력에서 상당한 발전을 보였습니다. 본 논문에서는 8개의 사고 모델의 실행 결과를 체계적으로 분석하였으며, 모델이 답변을 생성한 후에도 추가적인 반영을 고려하는 상황에 중점을 두었습니다. 분석 결과, 이러한 반영들이 주로 확인적(confirmatory)이며 초기 답변을 거의 변경하지 않음을 발견하였습니다.

- **Technical Details**: 이 논문에서는 수학적 기준을 다양하게 변화시키면서 반영의 역할을 탐구했습니다. 감독 세부 조정(Supervised Fine-Tuning, SFT) 데이터를 다양한 반영 단계로 구성하여, 반영이 모델의 초기 답변 정확성에 주로 기여한다는 것을 관찰하였습니다. 반영의 단계를 다루는 방법으로 ‘조기 정지(early-stopping)’ 기법을 제안하여 계산 효율성을 확보하는 방법을 모색하고 있습니다.

- **Performance Highlights**:  제안된 방법은 5개의 수학적 데이터셋에서 반영 단계를 동적으로 잘라내어 24.5%의 추론 토큰 사용을 줄이는 동시에 2.9%의 정확도 저하를 초래하는 결과를 가져왔습니다. 이 연구는 반영 행동의 분류, 훈련 데이터에 대한 통찰력, 효율적인 추론 기법의 세 가지 기여를 명시합니다.



### Symmetry-Aware Fully-Amortized Optimization with Scale Equivariant Graph Metanetworks (https://arxiv.org/abs/2510.08300)
- **What's New**: 본 논문에서는 Scale Equivariant Graph Metanetworks (ScaleGMNs)를 활용하여 비선형 최적화 문제를 해결하는 새로운 접근 방식을 제안합니다. 기존 모델을 단번에 미세 조정(single-shot fine-tuning)할 수 있는 가능성을 보여주며, 반복적인 최적화의 필요성을 줄입니다. 또한, CNN과 MLP 간의 스케일 대칭 gauge 자유도가 다르다는 이론적 결과를 제시하여 이 두 아키텍처간의 성능 차이를 설명합니다.

- **Technical Details**: ScaleGMN 프레임워크는 신경망(NN)을 그래프 표현으로 변환하여 permutation invariance를 달성합니다. 이 과정에서 가중치가 엣지 특징(edge features)에, 바이어스가 버텍스 특징(vertex features)에 매핑됩니다. GMN의 전달 패스(forward pass)는 특징 초기화, 메시지 패싱, 특징 업데이트 및 읽기 단계(readout)로 구성됩니다. Kalogeropoulos의 연구에서 제안된 방법은 이러한 과정이 스케일 대칭에 대해 공변형(equivariant)으로 설계되었습니다.

- **Performance Highlights**: ScaleGMNs는 다양한 손실 풍경(loss landscapes)에서 효과적으로 단번에 최적화를 수행할 수 있음을 입증하며, 효율적인 NN 최적화의 새로운 패러다임을 세워줍니다. 이 메타네트워크는 이미 훈련된 네트워크의 파라미터를 입력받아 새롭게 최적화된 세트를 출력하며, 이러한 과정을 통해 모델이 최적 솔루션으로 나아가도록 돕습니다. 최적화 수단으로서의 가능성이 크게 확대된 이 연구는 신경망 구조와 가중치 공간의 대칭성을 활용합니다.



### Co-TAP: Three-Layer Agent Interaction Protocol Technical Repor (https://arxiv.org/abs/2510.08263)
- **What's New**: 이 논문은 다중 에이전트 시스템의 핵심 차원인 상호 운영성(Interoperability), 상호 작용(Interaction), 협업(Collaboration), 지식 공유(Knowledge Sharing) 문제를 해결하기 위해 세 가지 계층 에이전트 상호 작용 프로토콜인 Co-TAP을 제안합니다. Co-TAP은 인공지능(AI) 및 기타 자동화 시스템에서의 효과적인 적용을 위해 설계된 세 가지 핵심 프로토콜인 HAI, UAP, MEK로 구성되어 있습니다. 각 프로토콜은 특정한 기능적 요구 사항을 충족하도록 세분화되어, 다중 에이전트 시스템의 대규모 배치를 위한 공학적 기초를 제공합니다.

- **Technical Details**: HAI(인간-에이전트 상호작용 프로토콜)는 사용자와 에이전트 간의 정보 흐름을 표준화하여 실시간 성능과 신뢰성을 보장합니다. UAP(통합 에이전트 프로토콜)는 다양한 에이전트 간의 통신 장벽을 허물고, 통합된 서비스 발견과 프로토콜 변환 메커니즘을 통해 이질적인 에이전트 간의 원활한 연결을 가능하게 합니다. MEK(메모리-추출-지식 프로토콜)는 에이전트가 개인 경험에서 학습하고 공유 가능한 지식을 형성할 수 있도록 지원하여 진정한 집단 지성을 실현하는 데 기여합니다.

- **Performance Highlights**: 제안된 Co-TAP 프로토콜 프레임워크는 다중 에이전트 시스템의 효율성과 확장성을 크게 향상시킬 것으로 기대됩니다. HAI는 사용자 경험을 극대화하여 인간-에이전트 상호작용의 유창성을 개선하며, UAP는 애플리케이션 통합의 장벽을 낮추고 시스템의 신뢰성을 높입니다. 마지막으로 MEK는 에이전트의 독립적인 경험을 시스템적인 인지 능력으로 전환하여 진정한 지능의 출현을 가능하게 합니다.



### Chain-of-Trigger: An Agentic Backdoor that Paradoxically Enhances Agentic Robustness (https://arxiv.org/abs/2510.08238)
- **What's New**: 대형 언어 모델(LLM) 기반 에이전트의 실제 응용에서 신뢰성에 대한 우려가 증가하고 있습니다. 이 연구에서는 다단계 백도어 공격 방식인 Chain-of-Trigger Backdoor (CoTri)를 소개하여 이러한 에이전트의 보안 취약점을 드러냅니다. CoTri는 환경에서 추출한 트리거를 순차적으로 사용해 에이전트를 원래 작업에서 이탈하도록 유도하는 방식으로 설계되었습니다.

- **Technical Details**: CoTri는 초기 트리거에 이어 환경의 상대적 정보로부터 후속 트리거를 생성하는 다단계 백도어 공격 방법입니다. 이는 복잡한 상황에서 에이전트가 안정적인 상태를 유지하도록 돕고, 실험 결과에서 CoTri는 거의 100%의 공격 성공률(ASR)과 거의 0%의 잘못된 트리거율(FTR)을 보여줍니다. 백도어가 적용된 에이전트는 혼란스러운 환경에서도 더 뛰어난 내성을 보여주며, 훈련 데이터의 우연적 특성을 통해 내구성을 향상시킵니다.

- **Performance Highlights**: CoTri는 긴 수명의 작업에 적합한 다단계 제어를 가능하게 하여 에이전트의 고유한 내구성과 작업 수행 능력을 개선합니다. 또한 CoTri를 멀티 모달 에이전트에 확장하여 다양한 모델에서도 높은 ASR와 낮은 FTR를 유지하며, 적응성과 확장성을 검증하였습니다. 이 연구는 LLM 기반 에이전트에서 나타날 수 있는 잠재적인 안전 위험을 강조하며, 최첨단 성능 뒤에 숨겨진 백도어의 위협을 경고합니다.



### Selection, Reflection and Self-Refinement: Revisit Reasoning Tasks via a Causal Lens (https://arxiv.org/abs/2510.08222)
- **What's New**: 이 논문에서는 기계 학습 모델, 특히 대규모 언어 모델(LLMs)의 추론 능력을 평가하는 기준으로서 추론 작업을 재조명합니다. 저자들은 이러한 작업들을 인과적 관점에서 바라보며, 잠재 공간(latent space)에서의 행동을 이해하고 문제 해결을 위한 통찰력을 제공합니다. 즉, 추론 작업을 선택 메커니즘으로 정의하고 고차원 논리적 개념이 주어진 관측값에 대한 선택 연산자로 기능한다고 제안합니다.

- **Technical Details**: 연구자들은 SR²라는 새로운 프레임워크를 제안하는데, 이는 추론 과정을 통해 잠재 변수들 간의 조밀한 종속성(dense dependencies)을 학습하는 데 초점을 맞춥니다. 이 프레임워크는 반사적 표현 학습(reflective representation learning), 종속성 자기 정제(dependency self-refinement), 주기적 중간 정렬(periodic intermediate alignment)의 세 가지 주요 모듈로 구성됩니다. 또한, 이 연구에서는 추론 작업을 제약 만족 문제(constraint satisfaction problem)로 이해하고, 고차원 논리 개념이 관측 입력을 제약하는 연산자로 작용한다고 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 스도쿠(Sudoku) 및 미로 탐색(Maze Navigation) 작업에서 10% 이상의 추론 정확도를 향상시키며, 최근의 계층적 추론 모델(HRM)보다 8배 적은 매개변수(parameter)로 성능을 초과합니다. 이러한 결과는 인과적 모델링(causal modeling)을 통한 추론의 발전 가능성을 제시하며 스케일링에만 의존하지 않고도 효과적인 추론이 가능함을 증명합니다. 이 연구는 기계 학습에서의 추론에 대한 더 깊은 이해와 인간적 사고에 가까운 모델 구축을 위한 새로운 통찰력을 제공합니다.



### DODO: Causal Structure Learning with Budgeted Interventions (https://arxiv.org/abs/2510.08207)
Comments:
          Under review. Supported by SoBigDatait IR0000013, FAIR PE00000013, ICSC CN00000013

- **What's New**: 이번 논문에서는 DODO라는 알고리즘을 제안하여, 에이전트가 반복적인 개입을 통해 환경의 인과 구조(causal structure)를 자율적으로 학습할 수 있는 방법을 설명합니다. 기존의 관찰적 접근법에 비해 DODO는 다양한 자원이 제한된 상황에서도 뛰어난 성능을 보이며, 인과 그래프를 정확히 복원하는 능력을 보여줍니다. DODO는 고전적인 인과 구조 학습 문제를 해결할 수 있는 새로운 경량(lean) 기준을 적용하여, 탐색과 이용을 최적화합니다.

- **Technical Details**: DODO는 에이전트가 반복적으로 개입(intervention)을 선택하고 적용함으로써, 새로운 증거(new evidence)에 기반하여 인과 그래프의 추정치를 업데이트하는 방식으로 작동합니다. 이 알고리즘은 순수한 점수 기반(score-based) 방법이나 제약 기반(constraint-based) 방법과는 달리, 불확실한 엣지를 탐색하는 동시에 이미 구축된 구조를 활용할 수 있도록 설계되었습니다. 이러한 접근은 비용이 많이 드는 환경에서도 더 적은 개입으로 높은 정확성을 달성할 수 있게 해줍니다.

- **Performance Highlights**: DODO는 다양한 시뮬레이션 데이터셋에서 관찰적 방법들과 비교하여 월등한 성능을 발휘하였습니다. 특히, 가장 도전적인 설정에서는 DODO가 0.25 F1 포인트를 초과하는 성능 향상을 기록했으며, 이는 기존의 최적 기준선에 비해 분명한 우위를 점합니다. 불리한 조건에서도 DODO는 인과 그래프를 거의 완벽에 가깝게 복원할 수 있는 능력을 보여주었습니다.



### The Tournament Tree Method for preference elicitation in Multi-criteria decision-making (https://arxiv.org/abs/2510.08197)
- **What's New**: 이번 논문에서는 다기준 의사결정에 필요한 전문가의 판단을 모델링하는 새로운 방법인 Tournament Tree Method (TTM)를 제안합니다. TTM은 이전의 방법들보다 적은 수의 쌍극 비교(pairwise comparisons)를 요구하며, 이를 통해 일관성 있는 비교 행렬을 생성합니다. 이 방법은 전문가의 인지 부담을 최소화하고, 복잡성을 줄이며, 기존의 Deck of Cards 방법과 호환됩니다.

- **Technical Details**: TTM은 세 가지 단계로 구성됩니다: (i) 전문가의 판단을 이끌어내기 위한 목표 비교 설정, (ii) 일관된 쌍극 비교 행렬의 구성, (iii) 이 행렬로부터 전역 가치 스케일(global value scale)의 도출입니다. 전반적으로 TTM은 $m$개의 매개변수로 선호 모델링의 차원을 줄이고, 쌍극 비교 행렬의 일관성을 보장합니다. 또한, TTM은 컴퓨테이셔널 효율성을 극대화하여 계산 부담을 경감합니다.

- **Performance Highlights**: 제안된 TTM은 의사결정자가 상대적으로 짧은 시간 안에 더 큰 수의 객체를 비교할 수 있게 해줍니다. 이 방법은 메트릭의 일관성을 유지하며 가치 스케일을 쉽게 계산할 수 있는 장점이 있습니다. 또한, 실제 의사결정 시나리오에서 웹 기반 도구를 통해 그 실용성을 보여주었습니다.



### Measuring What Matters: The AI Pluralism Index (https://arxiv.org/abs/2510.08193)
- **What's New**: AI 시스템이 지식, 커뮤니케이션 및 의사결정을 중재하는 역할이 점점 더 강화되고 있습니다. 그러나 현재 기술의 개발과 거버넌스는 제한된 기업과 정부에 집중되어 있으며, 이로 인해 기술이 좁은 이해관계를 인코딩하고 공공의_agency(에이전시)를 제한할 수 있다는 우려가 있습니다. 본 논문은 다양한 이해관계자가 목표, 데이터 관행, 보호 조치 및 배치를 형성할 수 있는 정도를 나타내는 AI pluralism(다양성)을 정의하고, 이를 평가하기 위한 AI Pluralism Index (AIPI)를 제시합니다.

- **Technical Details**: AIPI는 참여적 거버넌스, 포용성 및 다양성, 투명성, 책임의 네 가지 기둥을 통해 생산자와 시스템 가족을 평가하는 투명하고 증거 기반의 도구입니다. 저자들은 'Unknown' 증거를 명시적으로 처리하여 신뢰성을 평가하며, 구조적 웹 및 저장소 분석, 외부 평가 및 전문가 인터뷰를 통합한 재현 가능한 파이프라인을 구현했습니다. AIPI는 공개된 버전 관리 및 공공 중재 프로세스를 통해 시스템 및 제공업체 수준 결과를 출판합니다.

- **Performance Highlights**: AIPI는 다양한 절차와 방식으로 측정 가능한 일관성을 가지고 있으며, 사용자와 정책 입안자에게 비교 가능한 증거를 제공합니다. 이것은 pluralistic(다원적) 관행을 유도하고, 의사 결정 및 배치의 과정에서 모든 이해 관계자의 참여를 중요시합니다. 초기 제공자들의 결과도 보고되었으며, AIPI는 투명성, 안전성 및 거버넌스 프레임워크와의 관계에서 위치를 나타냅니다.



### R-Horizon: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth? (https://arxiv.org/abs/2510.08189)
- **What's New**: 본 연구는 최근 LRM (Large Reasoning Models)에서 long-horizon reasoning (장기 추론 능력)을 자극하기 위해 R-HORIZON이라는 새로운 방법론을 제안합니다. 기존의 벤치마크가 단일 수평 작업에 국한되어 있는 반면, R-HORIZON은 복잡한 다단계 문제를 해결하는 다양한 벤치마크를 구성합니다. 이 방법은 모델이 긴 추론 시나리오를 이해하고 반응하는 능력을 평가하는 데 있어 중요한 기여를 합니다.

- **Technical Details**: R-HORIZON은 query composition을 통해 LRM의 장기 추론 행동을 촉진합니다. 이 방법은 기존의 단일 수평 문제를 연결하여 서로 의존하는 다수의 문제를 포함하는 복잡한 다단계 문제를 생성합니다. 연구에서는 수학, 코드 생성 및 에이전트 응용 프로그램에서 총 6개의 대표적인 데이터셋을 사용하여 평가를 수행했습니다.

- **Performance Highlights**: R-HORIZON을 사용한 평가에서, 최신 LRM조차도 장기 추론 작업에서 성능 저하를 겪는다는 것이 발견되었습니다. RLVR (Reinforcement Learning with Verified Rewards) 데이터로 훈련시킬 경우, 다단계 문제에 대해 성능이 크게 향상되며, 표준 추론 작업에서도 7.5 포인트의 정확도가 증가했습니다. R-HORIZON은 LRM의 장기 추론 능력을 향상시키고 평가하는 데 있어 확장 가능하고 저렴한 패러다임으로 자리잡게 됩니다.



### Prepared mind, fast response: A temporal decoupling framework for adaptive knowledge orchestration in open-domain dialogu (https://arxiv.org/abs/2510.08175)
- **What's New**: 제안된 PMFR (Prepared Mind, Fast Response) 프레임워크는 비동기 지식 오케스트레이션을 통해 지연-품질 절충 문제를 근본적으로 해결하는 혁신적인 접근법입니다. PMFR은 즉각적인 응답 생성과 지식 획득을 분리하여, 경량 대화 모델과 동적 지식 기반을 결합하여 반초(0.5초)에 가까운 응답을 제공합니다. 이는 전통적인 방식들과 비교해 대화 흐름을 유지하면서 지식을 점진적으로 풍부하게 만듭니다.

- **Technical Details**: PMFR은 세 가지 조율된 구성요소를 포함합니다: 1) 실시간 충분성 평가를 위한 Knowledge Adequacy Evaluator, 2) 사용자와 즉각적인 상호작용을 위한 Lightweight Response Generator, 3) 백그라운드 지식 향상을 위한 Asynchronous Knowledge Refinement Agent입니다. 이 아키텍처는 지식의 적재적소 활용을 통해 응답 품질을 유지하면서도 효율성을 최적화합니다.

- **Performance Highlights**: PMFR은 TopiOCQA benchmark에서 95.3%의 지연 감소(23.38초에서 1.09초)와 함께 대당 최종 품질이 기존 모델과 비교해도 유사하게(0.613 대 0.620) 유지된다는 평가를 받았습니다. 이러한 성능은 반응 속도와 정보의 깊이를 모두 충족하는 혁신적인 솔루션으로 자리잡을 가능성을 제시합니다.



### Can Risk-taking AI-Assistants suitably represent entities (https://arxiv.org/abs/2510.08114)
- **What's New**: 이번 연구에서는 AI의 책임감 있는 사용을 위해 언어 모델(Launguage Models, LMs)의 위험 회피성( risk aversion)의 조작 가능성(manipulability of risk aversion, MoRA)을 탐구하였습니다. 이는 AI 기반 의사결정 지원 시스템에서 LMs의 위험 행동을 이해하는 것이 중요하다는 점을 강조합니다. 특히, 성별에 따른 태도와 불확실성, 역할 기반 의사 결정에 집중하여 다양한 경제 시나리오에서 인간의 위험 선호를 재현할 수 있는지 살펴보았습니다.

- **Technical Details**: 연구 결과, DeepSeek Reasoner와 Gemini-2.0-flash-lite와 같은 모델들이 인간 행동과 어느 정도 일치를 보이는 반면, 특정 불일치가 존재하여 인간 중심의 조작 가능성 측정법에 대한 세밀한 조정이 필요하다는 점이 드러났습니다. LMs의 위험 회피성은 조작 가능성을 가지고 있지만 이는 완벽하게 인간의 의사 결정 방식을 반영하지는 못하고 있다는 점에서 구체적인 실증 분석이 필요합니다.

- **Performance Highlights**: 이 연구는 AI 시스템이 인간의 위험 선호를 보다 정확하게 재현할 수 있도록 모델 설계를 개선할 방향을 제시합니다. 이는 위험 관리(context)에서 AI의 효과성을 높이는데 기여할 수 있는 접근법으로, AI 지원 시스템의 적용 가능성을 향상시킬 것으로 기대됩니다. 향후 연구에서는 AI의 윤리적 의사 결정을 증진하는 방법도 모색해야 할 것입니다.



### From Ethical Declarations to Provable Independence: An Ontology-Driven Optimal-Transport Framework for Certifiably Fair AI Systems (https://arxiv.org/abs/2510.08086)
Comments:
          19 pages, 2 figures

- **What's New**: 이 논문은 AI의 공정성을 보장하기 위한 새로운 프레임워크를 제안합니다. 기존의 편향 완화 방법의 한계를 극복하기 위해 모든 민감한 정보와 그 대리인을 체계적으로 제거하는 방안을 모색합니다. 특히, OWL 2 QL의 온톨로지 공학을 활용하여 민감한 속성을 정의하고, 이를 통해 편향 패턴의 전체 구조를 포착하는 시그마 대수(G) 를 구축합니다.

- **Technical Details**: 논문에서는 Delbaen Majumdar의 최적 수송(optimal transport) 기법을 통해, 민감한 속성과 독립적인 변수를 생성하고, L2 거리 최소화를 통해 원래 데이터의 정확성을 보존하는 방법을 탐구합니다. 이러한 접근은 시그마 대수 간의 의존관계로 편향을 모델링하며, 측정 가능한 구조로 온톨로지 지식을 편집합니다. 따라서 AI 시스템의 공정성을 보장하기 위한 수학적으로 타당한 방법론을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 대출 승인과 같은 실제 사례에서 매우 효과적이며, ZIP 코드와 같은 대리 정보가 인종을 드러낼 때도 실제 독립성을 보장합니다. 이 방법은 공정한 AI 시스템을 위한 검증 가능한 접근 방식을 제공하며, 윤리적 표준과 규제 체계에 부합하는 결과를 도출합니다. 또한, AI를 사회적 이익을 증진하는 도구로 활용할 수 있는 가능성을 강조합니다.



### AutoQual: An LLM Agent for Automated Discovery of Interpretable Features for Review Quality Assessmen (https://arxiv.org/abs/2510.08081)
Comments:
          EMNLP 2025

- **What's New**: 본 논문은 온라인 리뷰의 내재 품질을 평가하기 위한 새로운 프레임워크인 AutoQual을 제안합니다. AutoQual은 LLM(대형 언어 모델) 기반의 에이전트로, 해석 가능한 기능(feature)을 자동으로 발견하는 프로세스를 자동화합니다. 이는 데이터에 내재된 암묵적 지식을 명시적이고 계산 가능한 기능으로 변환하는 것을 목표로 하며, 인간 연구 프로세스를 모방하여 반복적으로 기능 가설을 생성합니다.

- **Technical Details**: AutoQual은 다각적인 아이디어 및 대조적 데이터 분석을 통해 기능 후보군을 형성합니다. 각 기능은 자연어로 표현 가능한 해석 가능한 형태로 설계되며, 최대한의 정보량을 제공하기 위해 상호 정보량(mutual information)을 극대화합니다. 이 시스템은 LLM을 활용하여 기초 후보군인 𝒮cand(기능 후보세트)을 생성한 후, 각 후보에 대해 신뢰할 수 있는 측정 도구를 자동으로 생성 및 검증하는 방식을 사용합니다.

- **Performance Highlights**: 대규모 온라인 플랫폼에서 AutoQual의 성능을 A/B 테스트로 검증한 결과, 사용자당 평균 리뷰 조회 수가 0.79% 증가하고 리뷰 독자의 전환율이 0.27% 향상되었습니다. 이러한 실질적인 개선은 AutoQual의 유효성을 입증하며, 리뷰 품질 평가를 위한 해석 가능한 기능 발견의 필요성을 해결하는 데 기여합니다.



### Multi-Condition Conformal Selection (https://arxiv.org/abs/2510.08075)
- **What's New**: 이 논문에서는 여러 조건을 고려한 다중조건 적합 선택(MCCS, Multi-Condition Conformal Selection) 알고리즘을 제안합니다. 기존의 적합 선택 방법들이 단일 조건(y > c)에 제한되어 있는 반면, MCCS는 연결적(conjunctive) 및 분리적(disjunctive) 조건 하에서도 적용될 수 있는 특징을 가지고 있습니다. 이를 통해 유한 표본 거짓 발견율(FDR, False Discovery Rate)을 효과적으로 제어할 수 있는 방법론을 제공합니다.

- **Technical Details**: MCCS 알고리즘은 연결 조건을 위한 지역 단조성(region monotonicity)을 갖춘 비적합성 점수(nonconformity score)와 분리 조건을 위한 전 세계적인 Benjamini-Hochberg(BH) 절차를 도입합니다. 이러한 구성 요소의 통합은 다양한 다중조건 환경에서도 FDR 통제가 가능한 엄격한 선택을 이루어내도록 합니다. 이 방법은 다중 간격(target) 및 다변량 응답 설정에서도 일반화될 수 있습니다.

- **Performance Highlights**: 실험 결과, MCCS는 기존 방법들에 비해 우수한 성능을 보여주었습니다. 특히, 단일 간격의 연결 조건 상황에서 FDR이 0.3으로 설정되었을 때, MCCS의 거짓 발견 비율(FDP, False Discovery Proportion)은 0.2874로 가장 가까운 값을 보였고, 이는 기존 방법들이 보여준 과도한 편차에 비해 신뢰성을 증명합니다. 또한, 텍스트, 시각, 다중 모달 작업에서의 결과도 실제 응용사례에서 MCCS의 실용성을 검증합니다.



### LinguaSim: Interactive Multi-Vehicle Testing Scenario Generation via Natural Language Instruction Based on Large Language Models (https://arxiv.org/abs/2510.08046)
- **What's New**: 이 논문에서는 자율주행차의 테스트 및 훈련 시나리오 생성의 새로운 접근법인 LinguaSim을 제안합니다. 이 프레임워크는 자연어 입력을 기반으로 현실적이고 상호작용적인 3D 시나리오로 전환해, 동적인 차량 상호작용과 사용자 의도 간의 정밀도를 높입니다. 기존의 2D 또는 개방 루프 시뮬레이션에서 흔히 발생하는 사실성 저하 문제를 해결하기 위해, LinguaSim은 자연어 지정 요소가 모두 자율주행 모델에 의해 안내됨을 보장합니다.

- **Technical Details**: LinguaSim은 네 가지 레이어로 구성된 시나리오 생성 구조를 가지고 있습니다: 일반 환경 레이어, 자율차량 레이어, 적대 차량 레이어 및 배경 교통 레이어로 이루어져 있습니다. 이 네 개의 레이어는 시나리오의 다양한 요소를 생성하기 위해 서로 다른 LLM 에이전트에 의해 처리됩니다. 또한, 실시간 평가 메커니즘이 통합되어 시나리오의 중요도 및 안전성을 프레임 단위로 트래킹하는 능력을 갖추고 있습니다.

- **Performance Highlights**: 실험 결과, LinguaSim은 사용자 의도에 맞는 다양한 중요도를 가진 시나리오를 효과적으로 생성할 수 있음을 보여줍니다. 예를 들어, 위험한 시나리오의 평균 처리 시간은 0.072초인데 반해 안전한 시나리오는 3.532초가 소요되었습니다. 또한, 리파인먼트 모듈은 초기 출력에서 과도한 공격성을 줄여 충돌률을 46.9%에서 6.3%로 낮추는 데 성공했습니다.



### AILoRA: Function-Aware Asymmetric Initialization for Low-Rank Adaptation of Large Language Models (https://arxiv.org/abs/2510.08034)
Comments:
          Submitted to AAAI2026

- **What's New**: 본 연구에서는 Low-Rank Adaptation (LoRA)의 한계를 극복하기 위해 새롭게 제안된 AILoRA 방법을 소개합니다. AILoRA는 함수 인식 비대칭 저랭크 사전 알림(function-aware asymmetric low-rank priors)을 포함한 패러미터 효율적인 방법입니다. 이 방법은 모델 성능과 패러미터 효율성을 균형 있게 유지하면서도 더 나은 조정 성능과 수렴 효율성을 제공합니다.

- **Technical Details**: AILoRA는 자가 주의(self-attention) 메커니즘에서의 프로젝션 행렬 W^Q와 W^V의 기능적 차이를 고려하여 비대칭 초기화 전략을 적용합니다. W^Q는 주의 분포 계산에 필수적인 작업별 의미 공간 지식을 캡처하여 다운스트림 작업의 변화에 민감합니다. 반면, W^V는 작업간 안정성을 갖는 토큰 수준의 특징 표현을 인코딩하여, 이 두 매트릭스의 초기화에 있어 대칭적 접근이 아닌 비대칭적 접근이 권장됩니다.

- **Performance Highlights**: AILoRA는 다양한 모델 아키텍처, 파라미터 스케일 및 데이터셋을 포함한 폭넓은 실험을 통해 기존 PEFT 방법들보다 뛰어난 성능과 빠른 수렴 속도를 자랑합니다. 본 연구는 특히 W^Q의 작업 민감한 의미 정보와 W^V의 일반화 가능한 특징 표현 간의 비대칭을 통해 LoRA의 효과를 극대화하여, 결과적으로 고급 모델의 효율적인 적용을 가능케 합니다.



### PEAR: Phase Entropy Aware Reward for Efficient Reasoning (https://arxiv.org/abs/2510.08026)
Comments:
          15 pages, 6 figures

- **What's New**: 본 논문에서는 Phase Entropy Aware Reward (PEAR)라는 새로운 보상 메커니즘을 소개합니다. PEAR는 모델의 응답을 두 단계로 나누어 각각의 엔트로피(entropy)를 고려하여 보상을 설계합니다. 이를 통해 생각 과정에서는 과도한 엔트로피를 penalize(처벌)하고, 최종 답안 생성 단계에서는 적절한 탐색을 허용함으로써 응답 길이를 조절할 수 있습니다.

- **Technical Details**: PEAR는 모델의 각 응답을 'Thinking Phase'와 'Final Answer Phase'로 나누어 분석합니다. 각 단계의 평균 엔트로피는 응답 길이와 긍정적인 상관관계를 가지며, Thinking Phase에서는 높은 엔트로피가 관찰됩니다. 이 접근법은 모델이 효과적으로 탐색하면서도 응답의 간결성(conciseness)을 유지할 수 있도록 도와줍니다.

- **Performance Highlights**: GSM8K, MATH500, AIME24, AMC23와 같은 네 가지 벤치마크에서 PEAR를 평가한 결과, 응답 길이는 37.8%에서 59.4%까지 감소하였으며, 정확도는 1% 이하로 유지되었습니다. 이 방식은 명시적인 길이 제약이나 데이터 커리에 의존하지 않으면서도 모델이 OOD(out-of-distribution) 문제에 잘 일반화될 수 있도록 하였습니다.



### Language Models Do Not Embed Numbers Continuously (https://arxiv.org/abs/2510.08009)
Comments:
          12 pages, 10 figures, 3 tables

- **What's New**: 이 연구는 최근 대형 언어 모델(Large Language Models, LLMs)이 숫자 및 산술 작업에서 정수를 조작하는 방식을 조사했습니다. 특히 LLM의 임베딩 공간이 숫자 값을 연속적으로 모델링하는 능력을 평가하지 않은 점에 주목했습니다. 우리는 이러한 모델들이 숫자 공간을 비연속적으로 나타내며, 실제로 상당한 노이즈를 도입한다고 설정하며, 이러한 결과는 수치 정밀도가 높은 특정 작업에서의 LLM의 활용 가능성을 의문시합니다.

- **Technical Details**: 이 논문에서는 선형 재구성(linear reconstruction) 및 주성분 분석(principal component analysis)을 포함한 임베딩 공간의 예측 속성을 활용했습니다. 세 개의 주요 LLM 제공업체(OpenAI, Google Gemini, Voyage AI)의 모델을 이용하여, 임베딩 공간에서의 변동성을 설명하는 주성분이 미미하다는 것을 발견했습니다. 이는 입력 숫자 공간의 단순성과의 관계 속에서 여러 구성 요소가 서로 직교함을 나타내며, 소수 자리 수가 커질수록 성능이 저하되는 경향을 보였습니다.

- **Performance Highlights**: 모델의 재구성 가능성은 높은 충실도를 보였지만($R^2 0.95$), 주성분이 설명하는 변동의 비율은 매우 적었습니다. 이는 고정밀 숫자, 큰 값 또는 부호가 혼합된 값이 일반적으로 나타나는 분야에서 LLM의 사용이 문제를 야기할 수 있음을 강조합니다. 이 연구 방법론은 모델의 수치 표현 능력을 측정하는 데 확장 가능하고 작업에 구애받지 않는 도구를 제공하며, 사용자가 이러한 모델의 한계를 보다 잘 이해할 수 있도록 돕습니다.



### ReInAgent: A Context-Aware GUI Agent Enabling Human-in-the-Loop Mobile Task Navigation (https://arxiv.org/abs/2510.07988)
- **What's New**: ReInAgent는 모바일 GUI 작업 수행 시 사용자 참여를 적극적으로 반영하는 새로운 다중 에이전트 프레임워크입니다. 기존의 자동화 에이전트는 사용자의 명령이 명확하고 완벽하다는 가정을 바탕으로 작동하였으며, 이는 현실 세계의 불확실성을 반영하지 못했습니다. ReInAgent는 세 가지 전문 에이전트(Information-managing Agent, Decision-making Agent, Reflecting Agent)를 통해 사용자와의 적극적인 상호작용을 통해 동적인 작업 진행을 지원합니다.

- **Technical Details**: ReInAgent는 슬롯 기반의 정보 관리 메커니즘을 채택하여 정보 딜레마를 해결하고 작업의 동적 발전을 가능하게 합니다. 각 전문 에이전트는 공유 메모리 모듈을 통해 협력하여 정보 교환과 결정적 대화를 진행하며, 사용자의 피드백을 적극적으로 수집합니다. ImA는 초기 지시 사항의 모호성을 해소하고 결정 작업이 진행되는 동안 사용자와 소통하여 추가 정보를 보완합니다.

- **Performance Highlights**: 실험 결과, ReInAgent는 정보 딜레마를 효과적으로 해결하고 사용자 선호도에 더 가까운 결과를 도출했습니다. 특히 정보 딜레마가 포함된 복잡한 작업에서는 Mobile-Agent-v2보다 25% 높은 성공률을 기록했습니다. 또한, ReInAgent는 실제 모바일 작업에 대한 세부적인 평가에서도 탁월한 성과를 입증하며 복잡한 작업 자동화에서의 유용성을 보여주었습니다.



### VoiceAgentBench: Are Voice Assistants ready for agentic tasks? (https://arxiv.org/abs/2510.07978)
- **What's New**: 이번 논문에서 제안된 VoiceAgentBench는 대규모 음성 모델(SpeechLMs)에 대한 포괄적인 벤치마크로, 실제 음성 기반 상호작용 시나리오에서의 에이전트 기능을 평가하기 위해 디자인되었습니다. 이 벤치마크는 인도와 관련된 5,500개의 합성 음성 쿼리를 포함하여 다국어 및 문화적 이해를 측정할 수 있습니다. 특히, 음성 기반 에이전트가 복잡한 도구 사용, 다중 턴 상호작용, 그리고 맥락적 의사결정 능력을 포함하는 기본적인 에이전트 기능을 평가할 수 있는 첫 번째 벤치마크입니다.

- **Technical Details**: VoiceAgentBench는 다섯 개의 인도 언어를 포함하여 영어와 힌디를 지원하며, 수천 개의 음성 쿼리를 통해 다양한 도구 호출 유형을 평가합니다. 이는 단일 도구 호출부터 다중 종속 도구 오케스트레이션에 이르는 복잡한 요청을 포함합니다. 또한, 새로운 샘플링 알고리즘을 사용해 음성 전환 시 다양한 악센트와 음성 특성을 고려하여 실제 음성 대화의 이질성을 포착할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 현재의 SpeechLMs는 맥락적인 도구 조정 작업과 인도적 일반화, 그리고 적대적 강건성에서 상당한 성능 차이를 보였습니다. 이는 기존의 음성 기반 모델들이 현실적인 에이전트 능력을 충분히 평가하지 못한다는 점을 강조하며, VoiceAgentBench의 필요성을 부각시킵니다. 또한, 우리는 SpeechLMs와 ASR-LLM 파이프라인 모두에서 주목할 만한 성능 차이를 발견했습니다.



### TaoSR-SHE: Stepwise Hybrid Examination Reinforcement Learning Framework for E-commerce Search Relevanc (https://arxiv.org/abs/2510.07972)
- **What's New**: 이 논문은 전자상거래 검색 엔진에서 쿼리-제품 관련성을 분석하는 새로운 방법인 TaoSR-SHE를 소개합니다. 이는 대규모 언어 모델(LLM)의 단계별 보상 정책 최적화(SRPO)를 기반으로 하여, 검색 관련성을 더욱 투명하고 견고하게 개선합니다. 기존의 훈련 방법인 SFT 및 DPO는 긴 쿼리에서 일반화에 한계를 보였지만, TaoSR-SHE는 이러한 문제를 해결하기 위해 데이터 필터링과 다단계 교육을 도입하여 진일보한 성능을 보여줍니다.

- **Technical Details**: TaoSR-SHE는 단계별 보상 정책 최적화(SRPO)를 핵심으로 하여, 고급 생성형 단계별 보상 모델과 인간 주석이 포함된 오프라인 검증기를 결합하여 단계별 피드백을 제공합니다. 이 프레임워크는 탐색을 장려하는 다양화된 데이터 필터링과 نموذج 연속 확장을 통해 훈련 효율성을 높이고 정책 엔트로피 붕괴를 완화합니다. 또한, 단계별 보상 메커니즘을 통하여 기존의 희소 보상 문제를 해결합니다.

- **Performance Highlights**: 실제 전자상거래 검색 벤치마크에서의 광범위한 실험을 통해, TaoSR-SHE는 추론 품질과 관련 예측 정확도를 향상시켰다는 결과를 보여줍니다. 기존의 SFT, DPO, GRPO 등과 비교하여, 더 나은 성능을 발휘하며 해석 가능성과 견고함 또한 동시에 강화되었습니다. 이로 인해 TaoSR-SHE는 복잡한 추론 상황에서도 신뢰할 수 있는 솔루션으로 자리매김하였습니다.



### Agent-Based Genetic Algorithm for Crypto Trading Strategy Optimization (https://arxiv.org/abs/2510.07943)
Comments:
          5 pages, 4 figures

- **What's New**: 이번 연구에서는 Cypto Genetic Algorithm Agent (CGA-Agent)라는 혁신적인 하이브리드 프레임워크를 소개합니다. 이 프레임워크는 유전 알고리즘과 지능형 다중 에이전트 조정 메커니즘을 통합하여 동적 금융 환경에서 적응형 거래 전략의 파라미터 최적화를 수행합니다. CGA-Agent는 실시간 시장 마이크로구조 정보를 포함하고, 전략 성과 피드백을 반영하여 진화적 프로세스를 동적으로 안내함으로써 정적인 최적화 접근법의 한계를 넘어섭니다.

- **Technical Details**: CGA-Agent는 다중 에이전트 시스템과 유전 알고리즘을 결합하여 거래 전략 매개변수의 동적 최적화를 위한 복잡한 아키텍처를 제시합니다. 이 프레임워크에는 분석 에이전트, 생성 에이전트, 평가 에이전트, 선택 에이전트, 교차 에이전트, 돌연변이 에이전트 등 6개의 전문 에이전트가 포함되어 각각 특정 전략 파라미터 최적화 과정을 담당합니다. 이들은 시장 데이터를 바탕으로 최고의 매개변수를 실시간으로 업데이트하고 평가하여 최적의 성과를 도출합니다.

- **Performance Highlights**: CGA-Agent를 통해 BTC, ETH, BNB의 세 가지 주요 암호화폐에서 실험을 수행한 결과, 총 수익률이 각각 29%, 550%, 169% 증가했음을 확인하였습니다. 또한, 위험 조정 메트릭인 Sharpe 비율과 Sortino 비율에서도 상당한 개선을 이루어냈습니다. 이러한 성과는 CGA-Agent의 효율성과 다이나믹한 적응력을 입증하며, 기존의 정적인 최적화 방법에 비해 월등한 성능을 보여줍니다.



### Enabling Personalized Long-term Interactions in LLM-based Agents through Persistent Memory and User Profiles (https://arxiv.org/abs/2510.07925)
Comments:
          8 pages, 1 figure, 1 table

- **What's New**: 이번 연구는 AI 에이전트에서의 개인화(personalization) 필요성을 강조하며, 기존 시스템 디자인의 한계를 극복하기 위한 새로운 프레임워크를 제시합니다. 개인화는 사용자 특정 정보를 저장하고 이를 활용할 수 있는 동적 메모리를 포함하여, 사용자 중심의 상호작용을 가능하게 하는 기술적 요구사항으로 정의됩니다. 또 다른 주요 기여로는 3개의 공개 데이터 세트를 통해 접근 방식을 평가하고, 초기 사용자 연구를 통해 개인화에 대한 사용자 피드백을 수집하여 방향성을 제시합니다.

- **Technical Details**: 이 연구에서는 개인화의 개념적 기초 위에 기술적 요구사항을 도출하여, 지속 메모리(persistent memory), 동적 조정(dynamic coordination), 자가 검증(self-validation), 진화하는 사용자 프로필(evolving user profiles) 등을 통합한 새로운 프레임워크를 설계합니다. 개인화의 정의를 바탕으로, 상위 LLMs와 외부 리소스를 결합하는 정보 검색 방식인 Retrieval Augmented Generation (RAG)의 개선된 응용을 제안하며, 메모리 강화(memory-augmented) 기술을 통해 과거 상호작용의 사용자 특정 정보를 통합합니다.

- **Performance Highlights**: 연구 결과, 제안된 프레임워크는 Retrieval Accuracy, Response Correctness, BertScore와 같은 다양한 메트릭에서 긍정적인 성과를 확보했습니다. 5일간의 사용자 파일럿 연구를 통해 수집된 초기 피드백은 사용자들이 느끼는 개인화 인식과 적응성에 대한 유용한 통찰을 제공합니다. 이러한 초기 결과는 향후 연구 방향을 제시하고, 지속 메모리와 사용자 프로필 통합의 잠재력을 강조합니다.



### Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents (https://arxiv.org/abs/2510.07920)
- **What's New**: 본 논문은 LLM 기반 금융 에이전트들이 일반적으로 나타내는 "이익 환상(profit mirage)" 문제를 규명하고 해결하기 위한 새로운 프레임워크인 FactFin을 소개합니다. 기존 LLM들은 훈련 데이터에서 미래에 대한 정보를 누출하여 비현실적인 성과를 보여주며, 이러한 문제를 FinLake-Bench라는 새로운 평가 벤치마크를 통해 수치적으로 quantifies합니다. 이 연구는 LLM이 실제 시장에서 신뢰할 수 있는 예측을 할 수 있도록 기여하는 방법을 제시합니다.

- **Technical Details**: FactFin 프레임워크는 전략 코드 생성기(Strategy Code Generator), 검색 증강 생성(Retrieval-Augmented Generation), 몬테카를로 트리 탐색(Monte Carlo Tree Search), 반사실 시뮬레이터(Counterfactual Simulator)라는 네 가지 핵심 요소를 통합합니다. 이 구성 요소들은 협력하여 실시간 시장 데이터를 이용해 거래 전략을 개발하고 개선하는 역할을 합니다. 이 연구에서 제안한 방법은 훈련 중에 특정 패턴을 암기하는 대신 LLM 기반 에이전트들이 원인(피드백)을 학습하도록 유도합니다.

- **Performance Highlights**: 논문에 따르면, 제안된 FactFin 방법은 기존의 모든 기준 모델들보다 뛰어난 평가를 보여주며, 샤프 비율(Sharpe Ratio)이 1.4배 더 높은 성과를 기록했습니다. 다양한 실험을 통한 결과, 기존 모델들이 훈련 데이터의 패턴을 암기하는 데 그치고 있는 반면, FactFin 프레임워크는 에이전트들이 시장에서의 결과를 실시간으로 분석하여 보다 진정서한 예측을 가능하게 합니다. 결론적으로, 이 논문은 LLM 기반 금융 거래의 기초적인 문제를 정량적으로 분석하고 이를 해결하기 위한 진일보된 접근법을 제시하고 있습니다.



### Towards Meaningful Transparency in Civic AI Systems (https://arxiv.org/abs/2510.07889)
- **What's New**: 이 논문은 인공지능(AI) 시스템이 정부 서비스에 어떻게 활용되는지를 분석하면서, 인공지능의 투명성이 공공의 이해와 행동을 어떻게 연결할 수 있는지를 다룹니다. 기존의 투명성에 관한 접근 대신, 이 연구는 시민 중심의 관점과 사회기술적 시스템(socio-technical systems) 관점을 결합하여 의미 있는 투명성(meaningful transparency) 개념을 제안합니다.

- **Technical Details**: 저자들은 AI 시스템의 결정 과정을 더 잘 이해하기 위해 기술 개체(technical objects) 대신, 시민들이 쉽게 접근하고 이해할 수 있는 형태로 정보를 제공하는 방법을 모색합니다. 이 과정에서 투명성은 단순히 기술적인 알고리즘의 응집체가 아니라, 공공이 AI 시스템과 상호작용할 수 있도록 돕는 인사이트를 제공해야 한다는 점을 강조합니다.

- **Performance Highlights**: 이 논문에서 제안하는 의미 있는 투명성은 시민들이 AI 시스템의 결정 방식에 대해 올바르게 이해하고, 자신의 삶에 영향을 미치는 결정들에 대해 의견을 제시할 수 있는 기회를 제공합니다. 결과적으로, 이는 시민과 정부 간의 소통을 촉진하고, AI 시스템의 기술적 진보가 실제 사회적 맥락에 어떻게 적합할 수 있는지를 보여줍니다.



### Understanding DeepResearch via Reports (https://arxiv.org/abs/2510.07861)
Comments:
          22 pages, 4 figures

- **What's New**: DeepResearch 에이전트는 복잡한 추론과 다중 도구 통합을 통해 전문가 수준의 연구를 수행하는 혁신적인 AI 패러다임을 제시합니다. 기존의 벤치마크가 개별적인 능력에 집중하는 반면, 이 연구는 DeepResearch 시스템의 총체적 성능을 측정하기 위한 새로운 평가 프레임워크인 DeepResearch-ReportEval을 도입했습니다. 이 프레임워크는 연구 보고서를 통해 DeepResearch 시스템을 평가합니다.

- **Technical Details**: DeepResearch-ReportEval은 세 가지 차원(quality, redundancy, factuality)을 체계적으로 측정합니다. 이 과정에는 LLM-as-a-Judge 방법론이 사용되며, 전문가들 간의 일관성을 높이는 데 기여합니다. 연구는 12개의 실제 카테고리에 걸쳐 100개의 큐리를 포함하는 표준화된 벤치마크를 제공합니다.

- **Performance Highlights**: 네 가지 주요 상용 시스템의 성능 평가 결과, 각 시스템의 디자인 철학과 성능 간의 상충관계를 밝혀냈습니다. 이 연구는 DeepResearch가 정보 도우미에서 지능적인 연구 파트너로 발전하는 과정에서의 기초적인 통찰을 제공합니다. 연구 결과는 이 아래의 URL에서 소스 코드와 함께 제공됩니다.



### Augur: Modeling Covariate Causal Associations in Time Series via Large Language Models (https://arxiv.org/abs/2510.07858)
Comments:
          22 pages, 9 figures

- **What's New**: 이번 연구에서는 Augur라는 새로운 시계열 예측 프레임워크를 소개합니다. Augur는 기존의 LLM 기반 접근법이 가진 한계를 극복하며, 다중 모드 데이터를 통합하는 가능성을 제시합니다. 특히, 이 모델은 LLM의 인과적(reasoning) 추론 능력을 활용하여 변수들 간의 방향성 인과 관계를 탐색합니다.

- **Technical Details**: Augur는 두 단계의 교사-학생 모델 아키텍처를 사용합니다. 강력한 교사 LLM이 시계열 데이터로부터 방향성 인과 그래프를 추론하며, 이를 위해 휴리스틱 검색과 쌍별 인과성 테스트(pairwise causality testing)를 이용합니다. 경량의 학생 에이전트는 이 그래프를 정제하고 높은 신뢰도의 인과 관계를 바탕으로 예측을 수행합니다.

- **Performance Highlights**: 실제 데이터셋을 이용한 광범위한 실험에서 Augur는 25개의 기준 모델과 비교해 경쟁력 있는 성능을 보여주었습니다. 또한, Augur는 제로 샷 영구화(zero-shot generalization)에서 강력한 성능을 입증하였습니다. 이는 Augur의 투명하고 추적 가능한 변수 상호작용(reasoning)을 강화합니다.



### FinMR: A Knowledge-Intensive Multimodal Benchmark for Advanced Financial Reasoning (https://arxiv.org/abs/2510.07852)
Comments:
          This paper has been accept by ICAIF 2025

- **What's New**: 이번 연구에서는 전문 금융 분석가 수준의 재무 추론 능력을 평가하기 위해 고안된 고품질 지식 집약적 다중모드 데이터셋인 FinMR을 소개합니다. FinMR은 15개의 다양한 금융 주제를 포함하며, 3,200개의 질문-답변 쌍으로 구성되어 있습니다. 이 데이터셋은 정교한 수학적 추론과 진보된 금융 지식을 포함하여 정교한 시각적 해석 과제를 통합합니다.

- **Technical Details**: FinMR은 지식 집약적 추론, 인식 중심의 시각적 추론 및 정교한 수학적 추론을 통합하여 다차원적 평가를 가능하게 합니다. 특히, 이 데이터셋은 고급 금융 지식과 복잡한 수학 기술이 요구되는 질문으로 구성되어 있으며, 이는 MLLMs의 성능을 전문가 수준의 금융 분석과 비교하는 데 필수적입니다. 또한, FinMR은 정보가 풍부하고 맥락이 풍부한 질문들로 구성되어 있어 모델의 능력과 한계를 보다 깊이 이해하도록 돕습니다.

- **Performance Highlights**: FinMR 데이터셋의 평가를 통해 다수의 MLLM이 텍스트 전용 모델보다 일관되게 뛰어난 성능을 보이는 것으로 나타났습니다. Gemini-2.5-Pro 모델이 가장 우수한 성능을 기록했으며, Claude-3.7-Sonnet 모델은 직접적인 시각적 입력을 제공받았을 때 수학적 추론 성능이 현저히 향상됨을 보여주었습니다. 이는 MLLMs의 성능 향상을 위한 중요한 기초 자료와 방향성을 제시합니다.



### An LLM-Powered Cooperative Framework for Large-Scale Multi-Vehicle Navigation (https://arxiv.org/abs/2510.07825)
- **What's New**: 이 논문에서는 도시 규모의 동적 내비게이션을 위해 설계된 LLM 기반의 다중 에이전트 프레임워크인 CityNav를 제안합니다. CityNav는 글로벌 조정(agent)과 로컬 내비게이션(agent)을 통합한 계층적 구조를 채택하여 대규모 차량 내비게이션의 확장성과 적응성을 높입니다. 이를 통해 각각의 차량이 자신의 효율성을 극대화하면서도 네트워크 전반의 교통 효율성을 향상시킬 수 있습니다.

- **Technical Details**: CityNav는 다중 차량 동적 내비게이션을 부분 관찰 마르코프 의사결정 과정(POMDP)으로 정의합니다. 이는 차량의 경로 계획과 보상 함수 등을 포함하여 도시 도로 네트워크를 효과적으로 관리하기 위한 수학적 모델입니다. 또한, 글로벌 에이전트와 로컬 에이전트 간의 협력적 추론 최적화 메커니즘을 도입하여 두 시스템 간의 로컬 결정과 글로벌 최적화를 맞추는 데 중점을 둡니다.

- **Performance Highlights**: CityNav는 160만 개의 도로와 43만 개의 교차로를 포함하는 네 개의 실제 도로 네트워크에서 테스트되어, 아홉 개의 기존 경로 검색 및 강화 학습 기반 방법보다 지속적으로 더 나은 성능을 기록했습니다. 실험 결과는 CityNav가 도시 규모의 이동 효율성과 혼잡 완화에서 일관되게 우위를 보이며, 지능형 대규모 차량 라우팅의 기반을 제공하는 잠재력을 강조합니다.



### Strategic Communication under Threat: Learning Information Trade-offs in Pursuit-Evasion Games (https://arxiv.org/abs/2510.07813)
Comments:
          15 pages, 13 figures

- **What's New**: 이번 논문에서는 적대적인 환경에서의 정보 획득과 관련된 전략적 트레이드오프를 탐구합니다. 연구자들은 Pursuit-Evasion-Exposure-Concealment Game (PEEC) 모델을 통해 추적자가 통신 시점을 결정하고, 상대방의 위치를 파악하는 과정에서의 위험을 고찰합니다.

- **Technical Details**: SHADOW (Strategic-communication Hybrid Action Decision-making under partial Observation for Warfare)라는 새로운 다중 머리의 순차적 강화 학습 프레임워크를 제안하며, 이는 연속적인 내비게이션 제어, 이산 통신 행동 및 행동 예측을 위한 상대 모델링을 통합합니다. 모든 에이전트들은 강화 학습을 통해 자신의 이동 정책을 학습하며, 추적자는 감시성과 위험 사이의 균형을 맞추기 위한 통신 정책도 학습합니다.

- **Performance Highlights**: SHADOW 시스템은 여섯 가지 경쟁 기법에 비해 더 높은 성공률을 기록했습니다. 논문에서는 또한 시간 순서 모델링과 상대 모델링이 효과적인 의사결정에 중요한 역할을 한다는 것을 확인하였습니다. 마지막으로, 학습된 정책들은 다양한 통신 리스크와 물리적 비대칭성 환경에서도 잘 일반화된다는 점이 강조되었습니다.



### GCPO: When Contrast Fails, Go Gold (https://arxiv.org/abs/2510.07790)
- **What's New**: 이 논문에서는 외부 기준 응답을 통합한 Group Contrastive Policy Optimization (GCPO) 방법을 소개합니다. 이는 모델이 문제를 해결하지 못하는 경우, 기준 답변이 올바른 응답을 제공함으로써 모델의 업데이트 방향을 명확하게 설정합니다. 이를 통해 훈련 효율성을 높이고 일반화된 추론을 향상시킬 수 있습니다.

- **Technical Details**: GCPO는 모델의 롤아웃에서 모든 응답이 부정확할 경우, 금 표준 답변(golden answer, GA)을 사용하여 긍정적인 예제를 제공합니다. 이 방법은 모델이 정답과 문제를 연결짓는 데 필요한 체계적인 단계를 제공합니다. 또한, 순서 기반 보상(signals)을 통해 토큰 수준 확률 최적화(token-level probability optimization)와 일관성을 확보할 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 GCPO는 기존 모델 대비 현저한 성능 향상을 달성했습니다. 이 방법은 훈련 샘플의 효율성을 극대화하며, 작은 모델이 더 큰 모델의 추론 스타일을 모방하게 함으로써 훈련 과정에서의 수렴을 가속화합니다. 여러 실험 결과는 이 모델이 최적의 학습 방향으로 안정적이고 효율적으로 진행되도록 한다는 것을 보여줍니다.



### An approach for systematic decomposition of complex llm tasks (https://arxiv.org/abs/2510.07772)
- **What's New**: 이 연구에서는 복잡한 작업에 있어 대형 언어 모델(Large Language Models, LLMs)의 신뢰성 문제를 해결하기 위해 새로운 체계적인 분해 프레임워크인 ACONIC를 도입합니다. 기존의 휴리스틱 기반 분해 방법 대신, 작업을 제약 문제로 모델링하고 형식적인 복잡성 측정을 활용하여 분해를 안내합니다. 이 프레임워크는 일반적인 복잡도 측정량을 바탕으로 작업을 분해하여 에이전트의 성능을 10-40% 향상시킬 수 있음을 제시합니다.

- **Technical Details**: 연구에서는 에이전트의 작업을 상태 기반 프레임워크를 통해 모델링하고, 이를 만족도 문제(Planning as Satisfaction, PaS)로 축소한 후, 이를 다시 표준 제약 만족 문제(Constraint Satisfaction Problem, CSP)로 변환하는 두 단계의 과정을 설명합니다. 에이전트의 작업은 일련의 행동을 결정함으로써 목표를 달성하는 것으로 정의되며, 각 행동의 선행 조건, 효과 및 삭제 효과를 부울 종속 제약으로 인코딩하여 문제를 제약 만족 인스턴스로 변환합니다.

- **Performance Highlights**: 제안된 ACONIC 프레임워크는 SAT-Bench 및 NL2SQL Spider 벤치마크를 통해 평가되었습니다. 복잡성에 기반한 분해 방법은 작업 정확성을 체인 오브 서스(Chain-of-Thought) 분해 방식보다 9-40% 향상시킨다는 결과를 보였습니다. 이러한 결과는 복잡성이 작업의 난이도를 구분하는 중요한 기준이 될 수 있음을 나타냅니다.



### From Noisy to Native: LLM-driven Graph Restoration for Test-Time Graph Domain Adaptation (https://arxiv.org/abs/2510.07762)
- **What's New**: 이 논문에서는 테스트 시 그래프 도메인 적응(Test-Time Graph Domain Adaptation, TT-GDA)을 새로운 프레임워크인 GRAIL을 통해 제시했습니다. GRAIL은 대규모 언어 모델(Large Language Models, LLMs)의 생성적 능력을 활용하여 소스 도메인 데이터에 직접 접근하지 않고도 타겟 그래프를 복원합니다. 기존의 그래프 도메인 적응(GDA) 방법들이 소스 도메인 데이터에 의존하는 데 비해, 제안된 방법은 원본 데이터 없이도 효과적인 적응을 가능하게 합니다.

- **Technical Details**: GRAIL 프레임워크는 두 단계로 이루어져 있습니다: (1) 그래프 확산 과정 토크나이저(Graph Diffusion Trajectory Tokenizer)와 (2) LLM 기반 그래프 복원 및 정렬(Alignment)입니다. 첫 번째 단계에서는 Q-former 인코더를 통해 입력 그래프를 압축하여 고정 크기의 잠재 표현으로 변환합니다. 이어서 그래프 확산 모델이 이러한 잠재 수준에서 그래프 복원 과정을 학습하고, 생성된 연속적 잠재 벡터는 양자화 모듈을 통해 토큰 ID로 변환됩니다.

- **Performance Highlights**: GRAIL은 다양한 벤치마크에서 실험을 통해 기존 최첨단 방법들을 훨씬 능가하는 성능을 입증했습니다. 이 방법은 타겟 그래프가 소스 도메인의 내재적 특성을 잘 반영하도록 하는 강화를 통해 개선된 품질을 자랑합니다. GRAIL의 접근 방식은 후속 연구에 있어 중요한 기준이 될 것으로 기대됩니다.



### Haibu Mathematical-Medical Intelligent Agent:Enhancing Large Language Model Reliability in Medical Tasks via Verifiable Reasoning Chains (https://arxiv.org/abs/2510.07748)
- **What's New**: 이 논문은 'Haibu Mathematical-Medical Intelligent Agent' (MMIA)를 소개합니다. MMIA는 LLM 기반의 아키텍처로, 형식적으로 검증 가능한 추론 과정을 통해 신뢰성을 보장합니다. 복잡한 의료 작업을 증거 기반의 원자 단계로 재귀적으로 분해하여, 전통적인 체계에서도 사용할 수 있도록 구성되었습니다.

- **Technical Details**: MMIA는 사용자의 요청을 '정리(thorem)'로 간주하여 증명 과정을 수행합니다. 이 과정은 '증거(log)'를 기록하며, 계획 세우기, 실행하기 및 검증하기의 순환 구조를 따릅니다. MMIA는 LLM의 복잡한 문제 해결 접근 방식을 모사하며, 각 서브 작업을 관리 가능한 단위로 분해하여 수행합니다.

- **Performance Highlights**: MMIA는 의료 관리 분야에서 98% 이상의 오류 탐지율과 1% 미만의 잘못된 긍정률을 기록하며 기존 LLM보다 성능이 뛰어났습니다. 또한 RAG 매칭 모드는 평균 처리 비용을 약 85% 줄일 것으로 예상됩니다. MMIA의 검증 가능한 추론 프레임워크는 의료 분야에서 신뢰할 수 있고 투명한 AI 시스템 구축에 중요한 기여를 하고 있습니다.



### SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation (https://arxiv.org/abs/2510.07733)
- **What's New**: 이 논문에서는 기존의 연구 문헌 조사를 자동화하는 방법론의 한계를 극복하기 위해 SurveyG라는 새로운 LLM 기반 에이전트 프레임워크를 제안합니다. 기존의 접근 방법들이 연구 문헌의 구조적 관계를 간과함으로써 일관성 있는 분류법이나 맥락 이해를 결여한 점을 지적하며, 이 문제를 해결하기 위해 계층적 인용 그래프(hierarchical citation graph)를 통합하여 구조적 지식과 맥락 지식을 조사 생성 과정에 반영하고자 합니다. 이 그래프는 Foundation, Development, Frontier의 세 가지 레이어로 구성되어 연구의 진화를 포착합니다.

- **Technical Details**: SurveyG는 LLM 기반 멀티 에이전트 프레임워크로, 연구 문서들 간의 관계를 모델링하기 위해 계층적 인용 그래프를 사용합니다. 이 그래프의 노드는 문서를 나타내고, 엣지는 인용 관계와 의미적 유사성을 포착하며, 각 노드는 Foundation, Development, Frontier라는 세 가지 레이어 중 하나에 속합니다. 이 구조를 통해 우리가 수집한 문헌의 지식 표현이 강화되며, 다층적인 요약이 기초와 발전, 최전선 방향을 포괄하여 생성됩니다.

- **Performance Highlights**: 실험 결과, SurveyG는 10개의 컴퓨터 과학 주제에 대한 조사를 기존의 최첨단 프레임워크들과 비교했을 때, Coverage, Structure, Relevance, Synthesis, Critical Analysis의 다섯 가지 차원에서 우수한 성능을 보여주었습니다. 특히, 기존 방법들이 구조적 일관성과 관계 모델링에 한계를 가지고 있어 Synthesis와 Critical Analysis에서 성과가 떨어진 반면, SurveyG는 잘 구성된 조사 초안을 생성하며 높은 품질을 유지합니다. 결과적으로, 이 연구는 자동화된 조사 생성에 있어 중요한 기여를 하며, 기존 방식과 비교해 보다 포괄적이고 사실적 정확도를 높인 조사를 제공합니다.



### oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning (https://arxiv.org/abs/2510.07731)
Comments:
          Main Text: 8 pages, In total: 37 pages, 9 figures

- **What's New**: 본 논문에서는 유기 화학의 메커니즘 추론을 위한 최초의 대규모 전문 큐레이션 벤치마크인 oMeBench를 소개합니다. 이 벤치마크는 10,000개 이상의 주석이 달린 메커니즘 단계를 포함하고 있으며, 중간체(intermediates), 유형 레이블(type labels), 난이도 평가(difficulty ratings) 등이 포함되어 있습니다. 또한, LLM의 성능을 정밀하게 평가할 수 있는 동적 평가 프레임워크인 oMeS를 제안하여 세부적인 평가를 가능하게 하였습니다.

- **Technical Details**: oMeBench와 oMeS는 LLM의 메커니즘 이해 능력을 정량화하는 강력한 도구로 작용하며, 여러 단계에서의 논리적 일관성을 유지하는 능력을 평가할 수 있습니다. 이번 연구는 LLM의 성능을 여러 모델에서 비교하고, 그 한계를 밝혀내며, 특정 데이터셋에서의 전문가 모델의 파인 튜닝(fine-tuning)을 통해 성능을 50% 향상시켰습니다. 이로써 LLM의 화학적 직관(chemical intuition)과 메커니즘 추론 능력의 현재 한계를 보여주고 있습니다.

- **Performance Highlights**: 현재의 LLM 모델들은 비록 화학적 직관을 보이고 있지만, 복잡한 여러 단계의 인과 논리를 유지하는 데 어려움을 겪고 있다는 것이 밝혀졌습니다. 또한, 전문가 주도 하에 작성된 oMeBench 데이터셋을 기반으로 LLM 성능이 크게 개선될 수 있음을 보여주며, 이는 화학 메커니즘 추론 분야의 발전을 기대할 수 있음을 의미합니다. 마지막으로, 우리의 기여는 대규모의 표준화된 메커니즘 데이터셋 구축과 더불어 새로운 평가 지표의 도입을 포함합니다.



### Control Synthesis of Cyber-Physical Systems for Real-Time Specifications through Causation-Guided Reinforcement Learning (https://arxiv.org/abs/2510.07715)
Comments:
          14 pages, 4 figures, 6 tables, accepted by RTSS 2025

- **What's New**: 이번 논문에서는 실시간 제어 시스템을 위한 새로운 온라인 인과적 보상 생성 방법을 제안합니다. 제안된 방법은 STL(Signal Temporal Logic) 명세를 바탕으로 시스템 행동을 지속적으로 모니터링하며, 각 제어 단계에서 보상을 생성합니다. 이를 통해 즉각적인 상태 변화에 반영된 보상을 제공합니다.

- **Technical Details**: 온라인 인과적 모니터링을 통한 STL 기반 보상 생성은 τ-MDP를 적용하여 비-마르코프 특성을 처리합니다. 또한 샘플링 윈도우를 도입하여 긴 경로 다루기의 비효율성을 피하고, 온라인 인과적 의미론의 매끄럽게 근사화하여 딥 RL(Deep Reinforcement Learning)에서의 미분 가능성을 확보하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 STL-가이드 RL 방법이 기존의 STL 기반 RL 방법보다 더 안정적으로 수렴하며, 여러 평가 지표에서 더 높은 값을 달성함을 보여주었습니다. Gym 환경에서 다양한 지속적인 제어 벤치마크를 평가하여, 제안된 프레임워크가 깊은 RL의 보상 생성에 있어 더 강력하고 효율적인 틀을 제공함을 입증했습니다.



### Multimodal Safety Evaluation in Generative Agent Social Simulations (https://arxiv.org/abs/2510.07709)
- **What's New**: 본 논문은 MLLM(Multimodal Large Language Models) 기반 에이전트의 안전성과 신뢰성을 평가하기 위한 재현 가능한 시뮬레이션 프레임워크를 소개합니다. 이 프레임워크는 안전성 개선, 위험한 활동 탐지, 그리고 사회 동역학 측정의 세 가지 차원에서 에이전트를 평가합니다. 이를 통해 에이전트가 상호작용을 통해 어떻게 행동 계획을 수정하고 안전한 행동으로 전환하는지를 분석하고 있습니다.

- **Technical Details**: 에이전트는 동적 계획(Dynamic Planning), 다층 메모리(Layered Memory), 다중 모드 인식(Multimodal Perception) 기능을 갖추고 있으며, SocialMetrics라는 행동과 구조 메트릭을 사용하여 계획 수정, 위험한 행동의 안전한 행동 전환 등을 정량화합니다. 시뮬레이션 환경은 실내외 공간으로 구성되며, 각 에이전트는 자아, 사회적 특성, 초기 사회적 유대에 대한 정보를 기반으로 설정됩니다. 에이전트는 사건을 반영하는 세션을 통해 위험 요소를 탐지하고 수정해야 합니다.

- **Performance Highlights**: 실험 결과, 에이전트들은 직접적인 멀티모달 모순을 탐지할 수 있지만, 글로벌 안전성과의 정렬에 실패하여 위험한 계획을 수정하는 데 55%의 성공률에 그쳤습니다. GPT-4o mini, Claude, Qwen-VL 모델을 사용한 시뮬레이션에서 75%, 55%, 58%의 평균 안전한 행동 전환 비율을 기록했습니다. 특히, 잘못된 시각적 정보와 연결된 위험한 행동의 45%가 수용됐으며, 이는 이미지에 대한 과신 경향을 나타냅니다.



### Safely Exploring Novel Actions in Recommender Systems via Deployment-Efficient Policy Learning (https://arxiv.org/abs/2510.07635)
- **What's New**: 이 논문에서는 오프 폴리시 학습(Off-Policy Learning, OPL) 기반의 새로운 안전한 정책 학습 방법인 Safe Off-Policy Policy Gradient (Safe OPG)와 사용자 탐색을 위한 배포 효율적 정책 학습(Deployment-Efficient Policy Learning for Safe User Exploration, DEPSUE)을 제안합니다. 기존의 학습 방법들이 새로운 액션을 탐색하는 데 안전하지 않다는 문제를 해결하기 위해, 안전성을 보장하면서도 새로운 액션을 탐색할 수 있는 프레임워크를 개발했습니다. 이 연구는 사용자 참여를 장기적으로 유지하는 데 중요한 새로운 액션의 탐색을 강조하고 있습니다.

- **Technical Details**: Safe OPG는 모델 없는 안전한 OPL 방법으로, 높은 신뢰성의 오프 폴리시 평가를 기반으로 합니다. 이 방법은 정책이 설정된 안전 기준을 초과하도록 하는 제약 최적화 문제를 해결하며, 사용자 탐색 기술과의 조합을 통해 안전성 요구 사항을 만족합니다. DEPSUE는 이전 배포에서 얻은 안전 여유를 활용하여 안전 규제의 완화를 통해 새로운 액션을 탐색할 수 있도록 설계되었습니다.

- **Performance Highlights**: Safe OPG는 기존 방법들이 안전 요구 사항을 심각하게 위반할 때에도 안전 기준을 거의 항상 만족하는 성능을 보였습니다. 그러나 이 방법은 너무 보수적이어서 새로운 액션을 거의 선택하지 않아, 안전성과 탐색 간의 무역이 나타났습니다. DEPSUE는 여러 배포 동안 안전 규제를 점진적으로 완화하여 새로운 액션을 안전하게 탐색할 수 있도록 하여, 성공적으로 안전 제약을 준수하면서 새로운 액션을 탐색하는 성과를 보여주었습니다.



### Test-Time Matching: Unlocking Compositional Reasoning in Multimodal Models (https://arxiv.org/abs/2510.07632)
- **What's New**: 이번 연구는 AI 모델의 compositional reasoning(구성적 추론) 문제를 재조명하며, 표준 평가 메트릭이 모델의 능력을 과소 평가한다는 점을 보여줍니다. 이를 해결하기 위해 그룹 구조를 더 잘 활용하는 그룹 매칭 점수(Group Matching Score)를 도입하여, 기존의 지표에서는 발견할 수 없는 모델의 숨은 능력을 드러냅니다. 연구 결과, SigLIP-B16과 GPT-4.1은 이전 모든 결과를 초월하는 성과를 이뤘습니다.

- **Technical Details**: 신선한 접근법으로 Test-Time Matching (TTM)이라는 반복적이고 자기 개선 가능한 알고리즘을 제안하였습니다. 이 알고리즘은 매칭 기반의 의사 레이블을 선택하여 자기 학습을 진행하고, 점진적으로 선택 기준을 완화하여 테스트 데이터셋에 대한 범위를 확장합니다. 이로 인해 SigLIP-B16과 GPT-4.1은 여러 벤치마크에서 놀라운 성능 향상을 보였습니다.

- **Performance Highlights**: TTM을 통해 SigLIP-B16은 Winoground에서 72.5, MMVP-VLM에서 89.44, ColorSwap에서 94.25의 성과를 기록하며 새로운 최첨단 결과를 세우고 있습니다. 특히 도전적인 데이터셋인 WhatsUp에서는 최대 85.7%의 상대적 성과 향상이 이루어졌습니다. 연구에서 TTM은 평가 메트릭의 변화를 극복하며 일관되게 모델 성능을 향상시키는 데 효과적입니다.



### A Case for Leveraging Generative AI to Expand and Enhance Training in the Provision of Mental Health Services (https://arxiv.org/abs/2510.07623)
- **What's New**: 이 논문에서는 Generative AI(생성적 인공지능)가 정신 건강 서비스 훈련을 강화하고 확장하는 저위험, 고효율적 사용 사례를 제안합니다. 특히 정신 건강 치료사 챗봇에 집중했던 기존 논의와 달리, Generative AI를 통해 훈련의 질을 높일 수 있는 가능성에 주목합니다. 노인들 간의 정신 건강 지원을 돕는 훈련 사례를 통해 Generative AI의 효과를 실증적으로 설명합니다.

- **Technical Details**: 전 세계적으로 정신 건강 제공자가 부족하며, 인구 10만 명당 13명의 정신 건강 제공자가 있을 뿐입니다. 최근의 기술 발전으로 인해 Generative AI를 사용하여 높은 품질의 정신 건강 서비스 제공 훈련을 지원하는 것이 가능해졌습니다. Retrieval-Augmented Generation(RAG) 아키텍처 등을 활용하여 훈련 플레이북과 윤리적 가이드라인을 제공하고, 실제 도메인 전문가들에 의해 평가를 진행해야 합니다.

- **Performance Highlights**: Generative AI는 훈련생이 다양한 배경과 클라이언트를 시뮬레이션하는 기회를 제공하여 실제 상황에서의 트레이닝 효과를 높일 수 있습니다. 특히, 클라이언트 시뮬레이션을 통해 훈련생이 초기 고객을 다룰 때의 결과를 개선할 수 있으며, 지속적인 피드백과 평가를 통해 실력을 체계적으로 향상시킬 수 있습니다. 다시 말해, Generative AI는 정신 건강 분야의 교육과 실제 임상 상황에서 모두 긍정적인 변화를 이끌어낼 수 있는 잠재력을 갖고 있습니다.



### Traceability and Accountability in Role-Specialized Multi-Agent LLM Pipelines (https://arxiv.org/abs/2510.07614)
- **What's New**: 이 논문은 대형 언어 모델(LLMs) 기반의 에이전트로 구성된 연속적인 멀티 에이전트 시스템의 신뢰성 문제를 다룹니다. 특히, 오류가 각 단계에서 조용히 전파되면서 발생하는 신뢰의 문제를 해결하기 위해 명확한 역할, 구조화된 전달 및 이력을 기록하는 추적 가능한 시스템을 제안합니다. 이를 통해 Planner, Executor, Critic로 구성된 파이프라인을 연구하고, 에러 발생 원인 및 수정 방법을 분석합니다.

- **Technical Details**: 이 연구에서는 세 가지의 최첨단 LLM을 활용하여 8가지 서로 다른 구성의 파이프라인을 구성하였습니다. 에이전트 간의 역할에 따른 특수한 행동, 즉 수리(이전 단계의 오류를 수정) 및 해악(올바른 상태에서 오류를 유발)률을 정량화하는 방법론을 사용했습니다. 연구 질문은 각각 연속 멀티 에이전트 파이프라인과 단일 LLM의 성능 비교, 특정 역할에서 각 LLM의 성과, 그리고 정확성, 비용, 지연 간의 트레이드오프를 포함합니다.

- **Performance Highlights**: 연구 결과, 에이전트 간의 구조화된 전달을 추가하는 것이 정확성을 크게 향상시키고 단순 파이프라인에서 흔히 발생하는 실패를 예방하는 것으로 나타났습니다. 또한, 모델은 특정 역할에 따라 강점과 위험이 명확하게 구분되며, 이는 수리 및 해악률로 정량화되었습니다. 마침내, 정확성-비용-지연 간의 트레이드오프는 태스크에 따라 달라지며, 이질적인 파이프라인이 가장 효율적일 수 있음을 발견하였습니다.



### AgentAsk: Multi-Agent Systems Need to Ask (https://arxiv.org/abs/2510.07593)
- **What's New**: 이번 논문에서는 기존의 multi-agent system에서 발생하는 에러의 전파를 줄이기 위한 새로운 모듈인 AgentAsk를 제안합니다. AgentAsk는 각 에이전트 간 메시지를 검사하여 필요 시 최소한의 질문을 추가함으로써 문제를 예방할 수 있도록 설계되었습니다. 본 연구는 구체적인 에러의 유형 분류와 함께 실용적인 해결책을 제시하여 LLM 기반의 multi-agent systems의 신뢰성을 향상시키고자 합니다.

- **Technical Details**: AgentAsk는 세 단계로 구성된 파이프라인을 따릅니다: 첫째, 에러의 추적에서 지식을 정제하여 압축된 정책을 생성합니다. 둘째, 이 정책을 감독하여 질문의 시점과 내용을 결정하며, 셋째, E-GRPO라는 강화학습( reinforcement learning) 목표를 사용해 온라인 최적화를 수행합니다. 이 모듈은 아키텍처에 구애받지 않으며 기존 시스템에 쉽게 통합할 수 있는 경량(modular) 설계입니다.

- **Performance Highlights**: AgentAsk는 다양한 벤치마크에서 공공 multi-agent 구현보다 정확성과 견고성을 지속적으로 향상시킵니다. 특히, 지연(latency)과 추가 비용이 5% 이내로 유지되면서도 강력한 평가자( evaluator)의 성능에 근접하는 결과를 보여줍니다. 이러한 실험 결과는 출력의 불일치가 시스템 전반에 영향을 미치기 전에 이를 감지하고 수정할 수 있는 경량 모듈이 유용할 수 있음을 뒷받침합니다.



### Benchmarking is Broken -- Don't Let AI be its Own Judg (https://arxiv.org/abs/2510.07575)
Comments:
          12 pages; Accepted to NeurIPS 2025. Link to poster: this https URL

- **What's New**: AI의 시장에서 급속도로 증가하는 수요는 신뢰할 수 있는 평가를 위한 통합적 기준의 필요성을 드러냅니다. 현재의 벤치마크 시스템은 데이터 오염이나 선택적 보고와 같은 취약점으로 인해 신뢰성을 잃고 있습니다. 많은 기업들이 벤치마크에서 최고 점수를 얻기 위해 상당한 자원을 투자하고 있지만, 이는 종종 진정한 이해를 기반으로 하지 않은 과대 포장된 결과를 초래합니다.

- **Technical Details**: 상대적으로 오래된 벤치마크 데이터세트는 그 효과성을 저하시켜 AI 모델의 편향된 성과 평가로 이어질 수 있습니다. 데이터 오염이 의심되는 경우, 모델의 일반화 능력에 대한 주장은 물음표를 던지게 됩니다. 또한, 현재의 벤치마크는 일관성을 결여하고 있어, 상이한 토큰화 방식과 점수 기준이 혼재하게 됩니다.

- **Performance Highlights**: PeerBench라는 새로운 커뮤니티 관리 평가 시스템이 제안되었으며, 이는 신뢰성을 보장하기 위해 엄격한 프로토콜을 따릅니다. 이 시스템은 밀봉된 실행, 항목 은행의 순환 갱신, 지연된 투명성을 통한 체계적 접근 방식을 통해 AI의 진정한 발전을 측정할 수 있는 길을 열어줍니다. 혁신적이고 신뢰할 수 있는 성과 평가를 통해 AI 커뮤니티의 신뢰를 회복하는 것이 목표입니다.



### An Evaluation Study of Hybrid Methods for Multilingual PII Detection (https://arxiv.org/abs/2510.07551)
- **What's New**: RECAP(연구 제목)는 13개 낮은 자원 언어에서 개인 식별 정보(PII)를 효과적으로 탐지하기 위해 결정론적 정규 표현식과 문맥 인식 대형 언어 모델(LLM)을 결합한 하이브리드 프레임워크입니다. 이 시스템은 추가적인 모델 교육 없이 300개 이상의 엔티티 유형을 지원하는 모듈형 설계를 가지고 있으며, 모호성과 필터링을 위한 3단계 정제 프로세스를 채택하고 있습니다. RECAP의 성능은 nervaluate 벤치마킹을 통해 기존의 NER 모델보다 82%, 제로샷 LLM보다 17% 더 높은 F1-score를 기록했습니다.

- **Technical Details**: RECAP 아키텍처는 다국어 PII 탐지에서 발생하는 문제를 해결하기 위해 이론적으로 정확한 규칙 기반 방법과 의미를 이해하는 LLM을 결합하여 설계되었습니다. 이 시스템은 13개 지원되는 지역 각각에 대한 전용 탐지기가 포함된 유연한 모듈형 디자인을 사용하며, 초기 하이브리드 기준선에서 최종 정제된 출력으로 탐지 품질을 점진적으로 개선하는 3단계 정제 파이프라인을 구현합니다. 이 파이프라인은 다중 레이블 모호함 해소, 범위 통합, 문맥 필터링의 단계를 포함합니다.

- **Performance Highlights**: RECAP의 각 정제 단계는 PII 탐지 정확도를 지속적으로 개선하는 성과를 보였습니다. sv_SE와 pt_BR 지역은 각각 77.53%와 47.76%의 향상을 기록하여 주목할 만한 결과를 도출했습니다. nl_BE 지역은 초기 성능이 높아 약간의 변동을 보였으나, 전체적인 트렌드는 PII 탐지가 각 정제 프로세스에서 얼마나 효과적으로 향상되는지를 보여주고 있습니다.



### Measuring and Mitigating Identity Bias in Multi-Agent Debate via Anonymization (https://arxiv.org/abs/2510.07517)
- **What's New**: 이번 연구는 다중 에이전트 토론(MAD)에서 에이전트들이 자아 바이어스(identity bias)와 아부(sycophancy) 문제로 인해 신뢰성이 저하되는 문제를 해결하기 위한 최초의 체계적 프레임워크를 제시합니다. 특히, 에이전트들이 대화에서 '자아'와 '동료'의 정체성을 구분하지 못하도록 하는 응답 익명화(response anonymization)를 통해 바이어스를 완화할 수 있음을 보여줍니다. 이로 인해 에이전트들은 응답 시 평등한 무게를 두게 되어, 바이어스를 줄일 수 있습니다.

- **Technical Details**: 제안된 모델은 정체성 가중치 기반의 베이지안 업데이트(identity-weighted Bayesian update) 프로세스로 대화 역학을 형식화합니다. 이를 통해 에이전트들은 동일한 무게로 대화할 수 있으며, 이는 다중 에이전트 토론의 신뢰성을 높이는 데 기여합니다. 또한 아이덴티티 바이어스 계수(Identity Bias Coefficient, IBC)를 정의하며, 이 계수는 에이전트가 동료의 의견을 얼마나 자주 따르는지 측정하는 지표로 사용됩니다.

- **Performance Highlights**: 실험 결과, 응답 익명화가 성능에 미치는 영향이 크지 않음을 확인했습니다. 이전의 성능이 유지되면서도 대화 과정에서 정체성 바이어스가 제거된 결과, 보다 내용 중심의 사고가 가능해짐을 보여주었습니다. 이러한 접근 방식은 신뢰할 수 있는 다중 에이전트 시스템 구축을 위한 장기적인 목표와 잘 부합합니다.



### CompassLLM: A Multi-Agent Approach toward Geo-Spatial Reasoning for Popular Path Query (https://arxiv.org/abs/2510.07516)
- **What's New**: 이번 연구에서는 CompassLLM이라는 혁신적인 다중 에이전트 프레임워크를 소개하고, 이 모델이 대규모 언어 모델(LLM)의 추론 능력을 활용하여 인기 있는 경로 질의를 해결하기 위해 어떻게 설계되었는지를 설명합니다. 이 시스템은 2단계 파이프라인을 통해 운영되며, 첫 번째 단계인 SEARCH에서는 인기 있는 경로를 식별하고, 두 번째 단계인 GENERATE에서는 기존의 경로 데이터가 없을 때 새로운 경로를 생성합니다. 또한 CompassLLM은 실제 및 합성 데이터셋에서 우수한 정확도를 보이며 경제적 효율성을 강조합니다.

- **Technical Details**: CompassLLM은 LLM의 그래프 구조 이해 및 추론 능력을 활용하여 공간 데이터의 인기 있는 경로 질의를 처리하는 데 초점을 맞춥니다. 이 프레임워크는 속도가 빠르고 비용 효율적인 실시간 추론을 지원하며, 특히 Sparse한 데이터 환경에서도 기존의 다른 방법들보다 더 나은 성능을 발휘합니다. 더불어, LLM 기반 방법의 특별한 장점도 포함되어 있어 전통적인 알고리즘이나 머신러닝 접근법보다 더 나은 일반화 성능을 보여줍니다.

- **Performance Highlights**: CompassLLM의 평가 결과, SEARCH 단계에서 기존 방법들보다 현저히 뛰어난 성과를 나타내었고, GENERATE 단계에서도 경쟁력 있는 성능을 보여주었습니다. 실험은 사용자 이동 추적 데이터와 다양한 공간 구성을 반영한 합성 데이터셋을 사용하여 수행되었으며, CompassLLM은 저희 연구에서 제안한 경로 추천 시스템의 최고 수준(SOTA) 모델들과 비교할 수 있는 성능을 발휘합니다.



### Optimizing Ethical Risk Reduction for Medical Intelligent Systems with Constraint Programming (https://arxiv.org/abs/2510.07491)
- **What's New**: 이 논문에서는 의료 지능 시스템(Medical Intelligent Systems, MIS)의 리스크 관리를 위한 새로운 최적화 문제를 제시합니다. 기존의 리스크 관리 프로세스는 전문가들에 의해 수동적으로 수행되어 왔으나, 이 저자들은 이를 수학적으로 형식화하고 최적화 문제로 정의했습니다. 윤리적 요구사항(ethical requirements)과 리스크를 동시에 고려한 MIS의 리스크 리덕션 최적화 기법을 수립하였습니다.

- **Technical Details**: 이 문제는 제한된 최적화(constrained optimization) 문제로 공식화되어 있으며, 세 가지 해결 패러다임인 Mixed Integer Programming (MIP), Satisfiability (SAT), Constraint Programming (CP)를 통해 접근합니다. 논문에서는 Minizinc 모델링 언어를 사용하여 리스크의 최적화 과정을 모델링하며, 각 라이프사이클 단계에서의 다양한 리스크 추정 절차를 수용할 수 있도록 합니다. 또한, 세 가지 솔버(𝙲𝚑𝚞𝚏𝚏𝚎𝚍, 𝙿𝚒𝚌𝚊𝚝𝚂𝙰𝚃, 𝙷𝚒𝙶𝙷𝚂)의 성능을 비교 분석하여 최적 해결 방법과 비선형 제약조건(nonlinear constraints)이 성능에 미치는 영향을 조명합니다.

- **Performance Highlights**: 비교 실험 결과는 각 최적화 접근 방식의 성능, 표현력, 확장성을 평가하고, 특정 조건하에서 어떤 방법이 최선인지에 대한 명확한 결론을 제공합니다. 또한, 각 접근 방식의 한계를 식별하고 향후 연구 방향에 대한 제안도 포함되어 있습니다. 이는 MIS의 윤리적 리스크 관리 프로세스를 통합하는 데 중요한 기초 자료로 작용할 것입니다.



### Evaluation of LLMs for Process Model Analysis and Optimization (https://arxiv.org/abs/2510.07489)
Comments:
          15 pages, 5 tables, 4 figures; full research paper currently under review for the Workshop on Information Technologies and Systems (WITS) 2025. The paper presents a comprehensive evaluation of large language models (LLMs) for business process model analysis and optimization, including error detection, reasoning, and scenario-based redesign

- **What's New**: 이번 논문에서는 여러 LLM(대형 언어 모델)의 경험을 공유합니다. 이 모델들은 대화형 스타일로 프로세스 모델을 이해하고, 문법적(syntactical) 및 논리적(logical) 오류를 찾고, 이를 자연어(NL) 인터페이스를 통해 깊이 있게 추론하는 능력을 평가하였습니다.

- **Technical Details**: 연구 결과에 따르면, 훈련되지 않은 기본 LLM인 ChatGPT(모델 o3)는 제로샷(zero-shot) 환경에서 BPMN 프로세스 모델을 이미지로부터 효과적으로 이해하고, 그에 대한 질문에 문법적, 논리적, 의미적(semantic) 수준에서 지능적으로 답변할 수 있음을 보여주었습니다. 다양한 LLM들은 정확성(accuracy)과 효과성(effectiveness) 면에서 성능 차이를 보입니다.

- **Performance Highlights**: 경험적 분석(emirical analysis) 결과, LLM이 비즈니스 프로세스 설계자와 사용자에게 유용한 조력자 역할을 수행할 수 있음을 확인하였습니다. 또한 프로세스 분석 및 최적화(context of process analysis and optimization)에서 LLM의 '사고 과정'(thought process)과 더 깊은 추론 능력을 연구하였고, LLM들이 인격적(anthropomorphic) 특성을 나타내는 경향이 있음을 발견하였습니다.



### ExpertAgent: Enhancing Personalized Education through Dynamic Planning and Retrieval-Augmented Long-Chain Reasoning (https://arxiv.org/abs/2510.07456)
Comments:
          Manuscript previously submitted to the NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models (LAW 2025)

- **What's New**: 이번 논문에서는 교육 분야에서의 개인화된 학습을 위한 새로운 지능형 에이전트 프레임워크인 ExpertAgent를 소개합니다. ExpertAgent는 신뢰할 수 있는 지식을 제공하고, 학습 경험을 실시간으로 조정하여 전통적인 정적 학습 콘텐츠의 한계를 극복하고자 합니다. 이 시스템은 동적으로 업데이트되는 학생 모델을 기반으로 학습 콘텐츠 및 전략을 계획하여 최적의 교육 전략을 제공합니다.

- **Technical Details**: ExpertAgent는 RAG (Retrieval-Augmented Generation) 및 CoT (Chain-of-Thought) 추론을 통합하여 사용자의 능동적 학습 경험을 제공합니다. 학습자가 질문을 입력하면, 시스템은 먼저 관련 문맥을 검색하여 정확한 답변과 함께 그에 대한 설명 및 출처를 제공합니다. 또한, 학생 모델은 학습자의 진행 상황을 기록하고, 이를 통해 적절한 과제를 추천하고 난이도를 조정함으로써 개인화된 피드백을 제공합니다.

- **Performance Highlights**: ExpertAgent는 두 가지 주요 모듈인 '상호작용 학습 모듈'과 '진행 상황 추적 모듈'을 통해 개인화된 교육 지원을 제공합니다. 상호작용 학습 모듈은 능동적인 교수법을 통해 사용자의 참여를 유도하고, 진행 상황 추적 모듈은 퀴즈와 지식 지도를 통해 학습 성과를 모니터링합니다. 이러한 기능 덕분에 학습자는 자신의 진행 상황을 쉽게 확인하고, 보다 능동적으로 학습에 참여할 수 있게 됩니다.



### TS-Agent: A Time Series Reasoning Agent with Iterative Statistical Insight Gathering (https://arxiv.org/abs/2510.07432)
Comments:
          NeurIPS 2025 Workshop on Foundations of Reasoning in Language Models

- **What's New**: 이번 논문에서는 최근 LLMs(대형 언어 모델)가 시간(series) 데이터에 대한 추론에서 여전히 어려움을 겪고 있다는 점을 지적하고, 이를 해결하기 위해 TS-Agent라는 새로운 AI 에이전트를 제안합니다. 이 에이전트는 LLM의 강점을 활용하여 증거를 수집하고 단계별로 추론을 실행하며, 통계 및 구조적 정보의 추출은 시간 데이터 분석 도구에 위임합니다. 이러한 설계는 다중 모드 정합 훈련의 필요성을 없애고, 데이터를 보다 해석 가능하고 검증할 수 있는 형태로 유지합니다.

- **Technical Details**: TS-Agent는 일반적으로 LLM이 수행하는 추론이 아닌, 원시 숫자 시퀀스를 직접 처리하는 방식으로 작동합니다. 이 에이전트는 문제 해결을 인간적인 방식으로 안내하며, 단계별 비평자(self-critic)와 최종 품질 게이트를 통해 추론을 반복적으로 다듬고 지식 유출이나 환각을 줄이도록 설계되었습니다. 이러한 접근 방식은 시간 데이터를 자연스럽게 취급하며, LLM은 분석을 조정하고 결론을 형성하는 데만 사용됩니다.

- **Performance Highlights**: TS-Agent는 기존의 지배적인 LLM과 유사한 성능을 개발하고, 특히 추론 작업에서 기존 모델들이 주로 암기에 의존해 실패하는 제로샷(zero-shot) 환경에서 상당한 개선 효과를 보였습니다. 즉, 이 연구는 LLM이 시간 데이터 및 추론 작업에서의 한계를 극복하기 위한 새로운 방법론을 제시하며, 대규모 벤치마크 평가에서도 긍정적인 결과를 도출해내었습니다.



### Less is More: Strategic Expert Selection Outperforms Ensemble Complexity in Traffic Forecasting (https://arxiv.org/abs/2510.07426)
Comments:
          Accepted to IEEE ICTAI 2025. Version 0.9. 10 pages, 5 figures. Preprint differs from the published version in formatting and minor wording

- **What's New**: 이 논문에서는 TESTAM+라는 새로운 스페이셜-템포럴 (spatio-temporal) 예측 프레임워크를 제안합니다. TESTAM+는 물리적 도로 네트워크의 토폴로지를 통합한 새로운 SpatioSemantic Expert를 도입하여 데이터 기반 특성 유사성과 결합해 하이브리드 그래프 구조를 통한 예측 개선을 꾀하고 있습니다. 이는 교통 데이터의 구조적 우선 순위를 활용할 수 있게 해주며, 기존 3전문가 TESTAM보다 MAE(Mean Absolute Error)를 현저히 줄이고 있습니다.

- **Technical Details**: TESTAM+는 도로 네트워크를 그래프 G=(V,A)로 모델링하며, N개의 노드와 인접 행렬 A를 기반으로 구성됩니다. 시간 t에서의 노드 특징은 𝐗(t)로 표현되며, 과거 관측값을 기반으로 앞으로 T 스텝에 대한 예측을 학습하는 것을 목표로 합니다. 예측 과정에서 Time2Vec 임베딩을 통해 정기적 및 비정기적 시간 정보를 인코딩하고, 개별 전문가(Identity, Adaptive, Attention)가 결정을 내리기 위해 각기 다른 스페이셜 모듈을 사용합니다.

- **Performance Highlights**: TESTAM+는 METR LA 데이터셋에서 1.3% MAE 감소 (3.10 vs. 3.14)와 PEMS BAY에서 4.1% 개선 (1.65 vs. 1.72)을 기록했습니다. Adaptive Expert가 1.63 MAE로 기존 TESTAM(1.72 MAE)을 초과하는 성과를 보여주었고, 최적의 Identity + Adaptive 구성은 MegaCRN에 비해 11.5% MAE 감소를 달성했습니다. 또한, TESTAM+는 전통적인 구성에 비해 추론 지연(latency)을 53.1% 줄였으며, 실시간 배포를 위해 우수한 계산 효율성을 확보했습니다.



### ProSEA: Problem Solving via Exploration Agents (https://arxiv.org/abs/2510.07423)
- **What's New**: ProSEA는 기계 학습 모델의 한계를 극복하기 위해 개발된 모듈형 다중 에이전트 프레임워크로, 탐색과 계획 진화를 통해 반복적인 문제 해결을 목표로 하고 있습니다. 이 시스템은 전문가 에이전트들이 실패에 대한 자세한 피드백을 제공하고, 이를 기반으로 매니저 에이전트가 동적으로 계획을 수정함으로써 보다 완성도 높은 결과물을 도출할 수 있도록 설계되었습니다. 특히, ProSEA는 인공지능이 인간과의 협업을 자연스럽게 지원할 수 있는 구조를 지니고 있어, 문제 해결 과정에서 인적 피드백을 통합할 수 있습니다.

- **Technical Details**: ProSEA는 계층 구조를 기반으로 하며, 매니저 에이전트가 다양한 도메인 전문 에이전트들을 조율하고, 복잡한 문제를 관리 가능한 하위 작업으로 분해하여 대안을 탐색하는 역할을 합니다. 기존 시스템과는 다르게, ProSEA 에이전트는 성공/실패에 대한 단순한 보고 이외에도 실패의 원인과 새로운 제약 조건을 상세히 보고하여, 동적인 계획 수정이 가능하도록 하고 있습니다. 이 아키텍처는 특정 모델에 종속되지 않으며, 모든 적합한 LLM을 통해 즉시 작동될 수 있도록 설계되었습니다.

- **Performance Highlights**: FinanceBench 벤치마크에서의 실험 결과, ProSEA는 인간 피드백 없이도 기존의 최첨단 시스템을 초능가한 성능을 보여주었습니다. 이는 ProSEA의 강력한 다차원 탐색 구조가 복잡한 문제 해결에서 우수한 결과를 도출하는 데 기여했음을 나타냅니다. 이러한 결과는 ProSEA가 더욱 투명하고 적응력이 뛰어나며, 인간의 요구에 부합하는 인공지능 에이전트의 기초가 될 잠재력을 강조합니다.



### Position: AI Will Transform Neuropsychology Through Mental Health Digital Twins for Dynamic Mental Health Care, Especially for ADHD (https://arxiv.org/abs/2510.07409)
- **What's New**: 정신 건강 진단 평가는 정적(static) 방법에서 지속적(continuous)이고 인공지능(AI) 기반의 평가로 전환되어야 한다고 주장하고 있습니다. 특히, ADHD(주의력 결핍/과잉행동장애)를 사례로 하여, 생성적 AI는 신경심리학의 용량 제약을 해결하고 개인화된 치료 경로를 가능하게 할 수 있는 잠재력을 보여줍니다. 더 나아가, 정신 건강 디지털 트윈(MHDT)을 제안하여 지속적으로 업데이트되는 증상 역학을 포착하고 개인 맞춤형 정신 건강 관리의 프레임워크로서 변화를 이끌어낼 수 있다고 강조하고 있습니다.

- **Technical Details**: 정신 건강 증상은 진단과 치료에 따라 변화하며, 전통적인 진단 방법은 이러한 동태를 적절히 반영하지 못하고 있습니다. 현재 작업흐름에서는 일반적으로 단기적 에피소드 상호작용을 통해 평가를 압축하고 있어, 내부적인 시간 및 환경에 따른 변화를 놓칩니다. 이 연구에서는 AI 시스템이 지속적인 데이터 수집을 통해 진단 정보를 개선할 수 있다는 점을 강조하며, MHDT는 각 개별 환자의 증상 및 경과를 실시간으로 모델링하는 기초가 될 수 있습니다.

- **Performance Highlights**: MHDT의 활용을 통해 ADHD와 같은 동적 정신 건강 조건에 대한 이해가 증진될 수 있으며, 지속적인 데이터 수집으로 전환되는 진단 과정이 기대됩니다. 이러한 시스템은 환자의 행동 및 증상 변화를 더욱 정확하게 반영하여 치료의 접근성과 효율성을 개선하는 데 기여할 것으로 전망됩니다. 또한, 다각적인 데이터 수집을 통해 임상 의사들이 보다 풍부하고 개별화된 데이터를 바탕으로 진단을 내릴 수 있습니다.



### Base Models Know How to Reason, Thinking Models Learn When (https://arxiv.org/abs/2510.07364)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 DeepSeek R1과 같은 사고 언어 모델(Thinking Language Models)이 왜 기본 모델(Base Models)보다 우수한 성능을 발휘하는지를 탐구했습니다. 하이브리드 모델을 제안하며, 기존의 기본 모델에서 사고 모델 수준의 추론 체인을 발생시키기 위해 적절한 시점에 추론 메커니즘을 활성화하는 방법을 보여줍니다. 이를 통해 사고 모델이 이미 존재하는 능력을 활용한다는 점을 시사합니다.

- **Technical Details**: 우리는 비지도 학습 기반의 방법론을 통해 사고 모델이 사용하는 인간 해석 가능한 추론 메커니즘의 분류체계를 구축합니다. 이 과정에서 Sparse Autoencoders (SAEs)를 사용하여 모델의 문장 수준의 활성화를 클러스터링하고, 각 문장이 각각 하나 또는 최대 세 가지의 추론 카테고리로 분류될 수 있다는 가정을 적용합니다. 이를 통해 발견된 기능은 핵심 인지 작용에 해당하며, 실험 결과 다양한 모델 아키텍처에서 감지된 최적의 차원 수는 15에서 25 카테고리 사이에 위치합니다.

- **Performance Highlights**: 하이브리드 모델을 사용한 결과, 기본 모델이 사고 모델과의 성능 격차를 최대 91%까지 회복할 수 있음이 확인되었습니다. 이는 가중치 업데이트 없이 이루어진 결과로, 단지 12%의 토큰만을 조정하여 성취된 것입니다. 이러한 연구 결과는 사고 모델의 훈련 방식에 대한 새로운 시각을 제공하며, 향후 언어 모델의 추론 훈련을 더 효율적으로 만드는 데 기여할 것입니다.



### L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint) (https://arxiv.org/abs/2510.07363)
Comments:
          This preprint was submitted to IEEE TrustCom 2025. The accepted version will be published under copyright 2025 IEEE

- **What's New**: L2M-AID는 자율 산업 방어를 위한 새로운 프레임워크로, 대형 언어 모델(LLM)과 다중 에이전트 강화 학습(MARL)을 통합하여 복합 사이버 물리 시스템을 보호합니다. L2M-AID는 LLM을 기반으로 한 협력적인 에이전트 팀을 조직하여 적응적이고 회복력 있는 보안을 달성하는 데 중점을 둡니다. 이 시스템은 두 가지 AI 패러다임의 깊은 융합을 통해 보안 목표와 운영 필수 사항 간의 균형을 맞추는 보상 함수를 설계했습니다.

- **Technical Details**: L2M-AID의 중추는 LLM이 제공하는 의미론적 브릿지를 활용하여 대량의 비구조화된 텔레메트리 데이터를 맥락적인 상태 표현으로 변환하는 것입니다. 이를 통해 에이전트는 단순히 패턴에 맞추는 것이 아니라 적의 의도를 추론할 수 있습니다. 이 의미론적으로 인지된 상태는 MARL 알고리즘인 MAPPO가 복잡한 협력 전략을 학습할 수 있도록 지원합니다.

- **Performance Highlights**: L2M-AID는 SWaT 데이터셋과 MITRE ATT&CK for ICS 프레임워크에 기반한 새로운 합성 데이터셋을 사용하여 광범위한 실험을 진행한 결과, 기존 침입 탐지 시스템 및 단일 에이전트 RL 기법에 비해 성능이 현저히 향상되었습니다. 높은 탐지률 97.2%를 달성하고, 허위 탐지율은 80% 이상 감소했으며, 응답 속도는 4배 향상되었습니다. 또한, 물리적 과정의 안정성 유지를 위한 성능에서도 우수한 결과를 보여주어, 중요한 국가 인프라를 보호하기 위한 강력한 새로운 패러다임을 제시합니다.



### Truth-Aware Decoding: A Program-Logic Approach to Factual Language Generation (https://arxiv.org/abs/2510.07331)
Comments:
          18 pages, Lean code provided

- **What's New**: 이번 논문은 Truth-Aware Decoding (TAD)이라는 검증 지향의 디코딩 방식을 제안합니다. TAD는 신경망 언어 생성이 지식 기반과 잘 연동되는 것을 목표로 하며, 확률적 프로그램 의미론의 전통에 뿌리를 두고 있습니다. 이 방식은 현대의 instruction-tuned 시스템에 디코딩 시 작동하는 의미적 가드를 추가하여, 논리적 일관성을 유지할 수 있도록 합니다.

- **Technical Details**: TAD는 대규모 언어 모델과 지식 중심 에이전트를 결합한 런타임 필터입니다. 각 에이전트는 사실 검증(Factual Verifier), 수학적 추론(Mathematical Reasoner), 맥락 모니터(Context Monitor)로 구성되어 있으며, 모든 체크와 결정을 기록하여 감사 가능 로그를 생성합니다. 이 논문은 지식 집약적 벤치마크에서의 실행 시간과 정확도 간의 균형을 수량화하고, 기계화된 보장을 제공합니다.

- **Performance Highlights**: TAD는 대규모 실증 모델과 형식 검증 간의 실용적인 연결 고리를 제공합니다. 수치 및 알고리즘 사례 연구는 가드레일이 할루시네이션을 줄이면서도 처리량을 유지하는 데 유효하다는 것을 확인했습니다. 이는 대규모 언어 모델이 신뢰성을 가지고 정보를 생성할 수 있도록 돕는 중요한 기여로 평가됩니다.



### BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation (https://arxiv.org/abs/2510.08572)
Comments:
          11 pages, 8 figures

- **What's New**: 이번 논문에서는 BLAZER라는 프레임워크를 제안하여 자동 생성된 훈련 데이터를 통해 조작 정책(manipulation policies)을 학습합니다. 기존의 로봇 기술이 대규모 인터넷 데이터에 의존하기 어려운 점을 극복하기 위해, 시뮬레이션을 통해 생성된 데모를 사용하여 로봇 학습을 개선하는 것이 핵심입니다. BLAZER는 대규모 언어 모델(LLM)의 제로샷(zero-shot) 능력을 활용하여 평가되고, 데이터 수집과 관리의 필요성을 줄입니다.

- **Technical Details**: BLAZER는 LLM 기반의 조작 에이전트를 자동 생성된 검증 데모를 통해 부트스트래핑(bootstrapping)하여 학습합니다. LLM이 생성하는 실행 가능한 조작 계획을 시뮬레이터에서 실행하고, 성공적으로 수행된 계획을 학습 세트로 사용하여 LLM의 조작 능력을 개선합니다. 이 과정에서는 객체의 위치, 방향 및 크기와 같은 특권 정보를 사용하여 훈련합니다.

- **Performance Highlights**: BLAZER는 시뮬레이션과 실제 환경에서 모두 제로샷 조작을 크게 향상시키고, 훈련 풀에 없는 작업에 대해서도 잘 일반화합니다. BLAZER로 훈련된 LLaMA-8B 모델은 훈련 과정에서 수동 데모가 전혀 필요하지 않으면서 LLaMA-70B보다 훨씬 높은 성공률을 보여주었습니다. 논문을 통해 다양한 조작 작업에서 발생한 일관된 성능 향상을 입증하며, 코드를 프로젝트 페이지에 공개하겠다고 발표하고 있습니다.



### ArenaBencher: Automatic Benchmark Evolution via Multi-Model Competitive Evaluation (https://arxiv.org/abs/2510.08569)
Comments:
          Preprint

- **What's New**: 이 논문은 ArenaBencher라는 기존 모델에 얽매이지 않는 벤치마크 진화 프레임워크를 제안합니다. ArenaBencher는 자동으로 테스트 사례를 업데이트하여 각 모델의 성능을 공정하게 비교할 수 있게 합니다. 이 시스템은 모델에 의해 메모라이즈된 내용이 아닌 진정한 일반화를 평가할 수 있도록 도와줍니다.

- **Technical Details**: ArenaBencher는 기존 벤치마크와 다양한 모델 풀을 기반으로 각 테스트 사례의 핵심 능력을 추론합니다. 예를 들어, 수학적 추론 벤치마크에서 multi-step arithmetic(다단계 산술)을 테스트하는 문제를 생성합니다. 검증 단계에서는 LLM을 판별자로 사용하여 질의-라벨 쌍의 정합성을 평가하고, 두 번째 단계에서는 여러 모델의 피드백을 집계하여 성능 격차를 드러내는 문제들을 선택합니다.

- **Performance Highlights**: ArenaBencher는 수학 문제 해결, 상식 추론 및 안전 분야에서 적용되어 새로운 실패 모드를 발견하고 테스트 목표의 일치를 유지하면서 난이도를 높이는 업데이트를 제공합니다. 이러한 업데이트는 모델 간의 성능 차이를 더 뚜렷하게 드러내고, 공정하며, 신뢰성 있는 비교 평가를 위한 발전된 테스트 사례를 생성합니다.



### NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos (https://arxiv.org/abs/2510.08568)
- **What's New**: 로봇이 새로운 조작 작업을 성능 저하 없이 수행할 수 있도록 하는 것을 목표로 하는 논문에서, NovaFlow라는 새로운 자율 조작 프레임워크를 소개합니다. 이 프레임워크는 작업 설명을 바탕으로 직접적인 시연 없이도 목표 로봇을 위한 실행 가능한 계획으로 변환합니다. 특히, 3D 물체 흐름을 활용하여 로봇의 동작을 계산하며, 이는 로봇의 제어 방식과는 관련이 없습니다.

- **Technical Details**: NovaFlow는 비디오 생성 모델을 활용하여 작업 이해를 위한 일반적인 3D 물체 흐름을 생성합니다. 이 흐름은 로봇의 조작을 위한 다차원적 표현으로, 개별적인 동작을 생성하는 데 사용되며, 전통적인Inverse Kinematics (IK) 및 궤적 최적화(trajectory optimization) 기법이 포함됩니다. 여기서는 강체 물체와 변형 가능한 물체를 서로 다른 방식으로 처리하여 다양성 있는 조작을 지원합니다.

- **Performance Highlights**: NovaFlow는 Franka arm과 Spot 이동 로봇을 사용하여 여러 실제 조작 작업에서 효과적인 제로샷 실행을 성공적으로 수행했습니다. 전통적인 시연 필요 없이 다채로운 물체 조작 작업을 수행할 수 있는 능력을 보여, 결과적으로 기존의 데이터 의존 방법들보다 뛰어난 성능을 달성하였습니다.



### MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning (https://arxiv.org/abs/2510.08567)
- **What's New**: 이 논문에서는 비전 언어 모델(VLMs)이 외부 도구를 활용하여 복잡한 추론과 의사결정 작업에서 어떻게 향상될 수 있는지를 탐구합니다. M-TRACE라는 큰 규모의 멀티모달 데이터셋을 구축하여 VLM 컨트롤러의 튜닝을 지원하며, MATRIX 에이전트를 통해 수동 주석의 필요한 비용을 줄이고 더 나은 일반화 성능을 발휘합니다. 또한, Pref-X라는 자동 생성된 선호 쌍(couples) 세트를 도입하여 단계별 선호 최적화를 통해 결정 과정을 정교화합니다.

- **Technical Details**: MATRIX는 두 단계의 프레임워크로 구성되어 있습니다. 첫 번째 단계에서는 M-TRACE에서 수집된 28.5K 멀티모달 작업을 활용하여 감독된 훈련을(trajectory-driven SFT) 진행합니다. 이어서, Pref-X를 사용하여 11K 개의 자동 생성된 선호 쌍을 기반으로 선호 최적화(preference optimization)를 통해 에이전트의 결정 과정을 정교화합니다. 이 과정 전반에 걸쳐 서로 검증된 경로(traces)를 활용하여 상황에 맞는 도구 사용 능력을 배양합니다.

- **Performance Highlights**: MATRIX는 Agent-X, GTA, GAIA 등 세 가지 벤치마크에서 기존 VLM보다 우수한 성과를 기록했습니다. 특히, 응답 정확도는 각각 14%, 23% 및 11% 향상되었습니다. 이 결과는 MATRIX의 구조적 도구 사용 능력과 단계별 언급 최적화가 효과적으로 작용했음을 보여줍니다. 또한, MATRIX는 이전 에이전트들보다 일관된 추론을 수행하고 더 적합한 도구 선택 능력을 보유하고 있습니다.



### SciVideoBench: Benchmarking Scientific Video Reasoning in Large Multimodal Models (https://arxiv.org/abs/2510.08559)
- **What's New**: 새로운 연구에서는 SciVideoBench라는 혁신적인 벤치마크를 도입하여 과학적 상황에서 고급 비디오 추론 능력을 평가합니다. 이 벤치마크는 25개 이상의 학술 분야에서 도출된 1,000개의 정교한 다지선다형 질문으로 구성되어 있으며, 각 질문은 도메인 전문가에 의해 확인되었습니다. 이를 통해 현재 대두되는 비디오 추론의 한계를 극복하고, LMMs의 새로운 발전 방향을 제시합니다.

- **Technical Details**: SciVideoBench는 물리학, 화학, 생물학 및 의학 등 네 가지 기초 분야에 걸쳐 제작된 241개의 연구급 실험 비디오를 기반으로 구축되었으며, 이러한 비디오는 실험 절차와 결과를 설명하는 동기화된 음성 내레이션과 함께 제공됩니다. 각 질문은 개념적, 가정적 또는 정량적 유형으로 분류되며, 모델이 정확한 시공간 기준, 도메인 지식 및 복잡한 논리적 추론을 수행할 수 있도록 요구합니다. 이 벤치마크는 비디오 이해와 관련된 복잡한 과학적 지식을 평가하기 위해 설계되었습니다.

- **Performance Highlights**: SciVideoBench에서의 평가 결과, 현재의 독점 및 오픈 소스 LMM들은 낮은 정확도를 보였으며, 과학적 추론 능력을 위한 상당한 개선의 여지가 있음을 나타냈습니다. 예를 들어, Gemini 2.5 Pro와 Qwen2.5-VL는 각각 SciVideoBench에서 낮은 성능을 기록했습니다. 이 연구는 LMM의 아키텍처, 추론 능력 및 지각 기반이 비디오 추론 성능에 미치는 중요한 역할을 강조하며, 향후 LMM 개발에 대한 명확한 방향을 제시합니다.



### Dream to Recall: Imagination-Guided Experience Retrieval for Memory-Persistent Vision-and-Language Navigation (https://arxiv.org/abs/2510.08553)
Comments:
          14 pages, 6 figures, 13 tables

- **What's New**: Memoir는 Vision-and-Language Navigation(VLN)에서 메모리 액세스의 효율성을 개선하기 위해 상상(imagination)을 활용하는 데이터 모델이다. 기존의 메모리 지향 VLN 방법들이 가지고 있는 주요한 한계점을 극복하기 위해, Memoir는 메모리 관점의 응용을 통해 내비게이션의 의사결정을 향상시키는 접근 방식을 채택한다. 특히 이 방법은 지난 경험을 효과적으로 검색하고 저장하여 진화형 내비게이션 효율성을 높인다.

- **Technical Details**: Memoir의 핵심 요소는 언어에 의해 조정된 세계 모델(world model)로, 이 모델은 내비게이션 경험을 저정하기 위해 미래 상태를 상상하고, 이 상상한 상태를 검색 쿼리로 활용한다. 또한, 하이브리드 뷰포인트 레벨 메모리(Hybrid Viewpoint-Level Memory, HVM)는 관찰과 행동 패턴을 뷰포인트에 고정시켜 다양한 내비게이션 시나리오에서의 검색 능력을 향상시킨다. 마지막으로, 경험 증강 내비게이션 모델은 수집된 정보를 바탕으로 보다 강력한 의사결정을 가능하게 한다.

- **Performance Highlights**: 10개의 다양한 메모리 지속 VLN 벤치마크에서 Memoir는 모든 시나리오에서 실질적인 성능 향상을 나타내며, IR2R에서 5.4%의 SPL 개선과 8.3배의 훈련 속도 향상, 74%의 추론 메모리 감소를 달성했다. 이러한 결과는 환경 및 행동 메모리를 예측적으로 검색하는 것이 더 효과적인 내비게이션을 가능하게 함을 검증하고 있으며, 이 상상 기반 패러다임에 대한 상당한 여지가 있음을 보여준다.



### VideoNorms: Benchmarking Cultural Awareness of Video Language Models (https://arxiv.org/abs/2510.08543)
Comments:
          24 pages, 5 figures, under review

- **What's New**: 이번 연구에서는 비디오 대형 언어 모델(VideoLLMs)의 문화적 이해를 평가하기 위해 VideoNorms라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 미국과 중국 문화에서 파생된 1000개 이상의 (비디오 클립, 규범) 쌍으로 구성되며, 사회문화적 규범에 기반한 주석을 포함하고 있습니다. VideoNorms는 인간-AI 협업 프레임워크를 통해 구성되어, 모델의 규범 인식 능력을 평가하는 데 기여합니다.

- **Technical Details**: VideoNorms 벤치마크는 비디오 클립에서 특정 문화적 규범이 준수되었는지를 예측하는 두 가지 분류 작업과 이를 뒷받침하는 언어적 및 비언어적 증거를 제시하는 설명 작업을 포함합니다. 이를 통해 모델이 규범 채택 및 위반을 이해하는 데 어려움을 겪고 있음을 분석하였습니다. 이 연구에서는 인간 전문가들이 주석을 검토하고 수정하는 과정이 포함되어, 데이터의 질을 높이는 데 기여합니다.

- **Performance Highlights**: 연구에서 평가된 다양한 오픈 웨이트 VideoLLMs는 규범 준수보다 위반을 인식하는 데 더 낮은 성능을 보였습니다. 또한, 중국 문화와 관련된 비디오 클립에서 성능이 떨어졌으며, 비언어적 증거를 제공하는 데 어려움을 겪었습니다. 이러한 결과는 문화적으로 기반한 비디오 언어 모델 학습의 필요성을 강조하며, VideoNorms 벤치마크가 이 문제를 해결하는 데 기여할 것임을 시사합니다.



### On the optimization dynamics of RLVR: Gradient gap and step size thresholds (https://arxiv.org/abs/2510.08539)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 후처리에서 강화 학습(RL) 기반 접근법인 RLVR(Reinforcement Learning with Verifiable Rewards)가 효과를 보고하고 있습니다. RLVR은 간단한 이진 피드백을 사용하여 LLM을 미세 조정하는 강력한 방법으로, 성공 여부를 자동으로 확인할 수 있는 작업에 적합합니다. 그러나 RLVR의 이론적인 기초는 부족하여, 본 논문은 이 빈틈을 메우고자 RLVR의 교육 과정을 정량적으로 분석했습니다.

- **Technical Details**: 논문에서 제안하는 주된 개념은 Gradient Gap으로, 이는 낮은 보상 지역에서 높은 보상 지역으로의 개선 방향을 형식적으로 정의합니다. RLVR의 수렴성은 이 Gradient Gap에 따라 결정되어, 특정 임계 값 아래에서는 안정적인 학습이 이루어지고, 그 이상에서는 성능이 무너집니다. 이는 출력의 길이와 작업의 난이도에 따라 효과적인 학습 속도가 어떻게 조정되어야 하는지를 설명합니다.

- **Performance Highlights**: 이론적 설명을 바탕으로, 연구진은 조절된 밴디트 시뮬레이션과 LLM을 활용한 실험을 통해 이론의 정확성을 검증했습니다. 예를 들어, GRPO를 사용하여 Qwen2.5-7B 모델을 GSM8K와 DAPO17k 데이터셋으로 미세 조정하였으며, 이론과 실제 성과 간의 밀접한 일치를 보여주었습니다. 이와 같은 연구 결과는 RLVR의 안정적인 수렴과 성공적인 활용을 위한 명확한 지침을 제시합니다.



### Kontinuous Kontext: Continuous Strength Control for Instruction-based Image Editing (https://arxiv.org/abs/2510.08532)
Comments:
          Project Page: this https URL

- **What's New**: Kontinuous Kontext는 자연어를 통해 이미지를 직관적으로 편집할 수 있는 새로운 방법론이다. 기존의 텍스트 명령만으로는 세부적인 편집 조정이 어렵다는 한계를 극복하기 위해, 이 모델은 사용자가 편집 강도를 연속적으로 조절할 수 있는 기능을 제공한다. 이를 통해 사용자는 미세한 조정부터 강한 효과까지 매끄럽게 모든 편집을 수행할 수 있게 되었다.

- **Technical Details**: 이 모델은 기존의 최첨단 이미지 편집 모델인 Flux Kontext를 기반으로 하며, 편집 강도를 규명하는 스칼라 값을 추가 입력으로 받아들인다. 경량의 프로젝터 네트워크를 사용하여 입력된 스칼라 값과 편집 지침을 모델의 조정 공간에서 조정계수로 매핑한다. 데이터 훈련은 기존의 생성 모델을 사용하여 다양한 이미지 편집 강도를 가진 쿼드루플릿을 합성하여 이루어진다.

- **Performance Highlights**: Kontinuous Kontext는 세부적인 편집 조정이 가능해 사용자가 특정 속성, 재료, 디자인 변경 및 환경 조정 등의 다양한 편집 작업에서 정밀한 제어를 할 수 있도록 한다. 실험 결과, 이 모델은 훈련되지 않은 얼굴 속성 변경이나 신체 형태 변경과 같은 새로운 편집 범주에서도 일반화되어 그 성능을 입증하였다. 이로 인해 중요한 비주얼 편집 작업에도 효과적으로 사용할 수 있는 방법론이 되었다.



### SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models (https://arxiv.org/abs/2510.08531)
Comments:
          Project Page: this https URL Code: this https URL

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)에서 공간 추론(Spatial reasoning)의 한계를 해결하기 위해 SpatialLadder-26k라는 다중 모달 데이터셋과 순차적 훈련 방법론을 제안합니다. 기존 방법들이 인지와 추론의 계층적 구조를 놓치고 있어 성능이 저조하다는 점을 지적하며, 공간 인지를 단계적으로 구축해야 한다고 주장합니다. 이를 통해 공간 지능을 강화하는 세 단계의 훈련 프레임워크를 선보이며, 단계적 접근 방식으로 공간 추론 능력을 향상시킬 수 있음을 강조하고 있습니다.

- **Technical Details**: 제안된 SpatialLadder-26k 데이터셋은 26,610개의 샘플로 구성되어 있으며, 객체 위치 확인(Object localization), 단일 이미지(Single image), 다중 시점(Multi-view), 비디오(Video) 공간 추론 작업을 포함합니다. 이 데이터셋은 각 모달리티 전반에 걸쳐 체계적인 커버리지를 보장하는 표준화된 파이프라인을 통해 구축되었습니다. 훈련 프레임워크는 세 단계로 나뉘며, 첫 번째 단계에서는 객체 위치 확인을 통해 공간 인지를 확립하고, 두 번째 단계에서는 여러 차원 공간 작업을 통해 이해를 발전시키며, 세 번째 단계에서는 강화 학습(Reinforcement learning)을 통해 복잡한 추론을 강화합니다.

- **Performance Highlights**: SpatialLadder 모델은 3B 파라미터를 가지며, 공간 추론 벤치마크에서 최첨단 성능을 기록하였습니다. 연구 결과, SpatialLadder는 기존 모델보다 23.4% 향상된 성능을 보이며, GPT-4o를 20.8% 및 Gemini-2.0-Flash를 10.1% 초과 달성했습니다. 또한, 도메인 외 베치마크에서도 7.2% 향상된 일반화 성능을 유지함으로써, 인지에서 추론으로의 단계적 훈련이 강력한 공간 지능을 위해 필수적임을 입증하였습니다.



### CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards (https://arxiv.org/abs/2510.08529)
- **What's New**: 이 연구에서는 Co-Evolving Multi-Agent Systems (CoMAS)라는 새로운 프레임워크를 소개합니다. CoMAS는 외부의 감독 없이 에이전트 간의 상호작용을 통해 자율적으로 개선될 수 있도록 설계되었습니다. 이는 기존의 강화 학습 기반 방법과는 달리 인간 지능에서 관찰되는 자가 발전 메커니즘에 더 가깝습니다.

- **Technical Details**: CoMAS는 풍부한 토론 역학으로부터 내재적 보상 신호(intrinsic reward signals)를 생성하고, LLM-as-a-judge 메커니즘을 통해 이 보상들을 도출합니다. 각 에이전트의 정책(policy)은 강화 학습(RL)을 통해 최적화되며, 이를 통해 분산형(decentralized) 및 확장 가능한(co-evolution) 자가 발전을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, CoMAS는 훈련되지 않은 에이전트들에 비해 일관되게 우수한 성능을 보였으며, 대부분의 평가 설정에서 최신 기술(state-of-the-art) 성과를 달성했습니다. 중재 연구(ablation studies)를 통해 상호작용 기반 보상 신호의 필요성이 확인되었고, 에이전트의 수와 다양성이 증가할수록 유망한 확장성을 보여주었습니다.



### To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language Models (https://arxiv.org/abs/2510.08510)
Comments:
          Preprint. Project page: this https URL

- **What's New**: 최근 대형 비전 언어 모델(LVLMs)은 시각적 정보와 텍스트 정보를 이해하고 추론할 수 있는 강력한 아키텍처로 자리 잡았습니다. 이 모델은 비전 트랜스포머(Vison Transformer, ViT)와 대형 언어 모델(Large Language Model, LLM)이라는 두 가지 핵심 구성 요소에 의존합니다. 본 연구에서는 ViT에서 시각적 의미를 포착하는 중요한 고규범 토큰인 ViT attention sinks를 탐구하였으며, 이들이 LVLM의 이해와 추론에서 어떻게 기여하는지에 대한 자세한 분석을 제시합니다.

- **Technical Details**: LVLM은 주로 세 가지 주요 구성 요소로 구성됩니다: 비전 인코더(Visual Encoder), 연결 모듈(Connector Module), 언어 모델(Language Model). 비전 인코더는 입력 이미지에서 시각적 특징을 추출하여 비밀 상태(hidden state)를 생성하며, 연결 모듈은 이러한 시각적 특징을 LLM의 텍스트 공간으로 매핑합니다. 마지막으로 언어 모델은 다양한 종류의 토큰을 입력받아 출력을 생성하며, 여기서 다중 헤드 어텐션(Multi-Head Attention, MHA) 메커니즘이 중요한 역할을 합니다.

- **Performance Highlights**: 연구에서는 ViT attention sinks를 효과적으로 활용함으로써 LVLM 성능을 향상시키는 여러 가지 방법을 제안했습니다. 특히, DIYSink라는 훈련 기반 프레임워크를 도입하여, 이 시각적 토큰들이 언어 모델에서 어떻게 활용될 수 있는지를 실험적으로 검증하였습니다. 다양한 LVLM과 비전 모델 조합에 대해 일관되게 성능 향상이 나타났으며, 이는 ViT attention sinks의 잠재력을 재발견하는 데 기여하였습니다.



### AI-Driven Radiology Report Generation for Traumatic Brain Injuries (https://arxiv.org/abs/2510.08498)
- **What's New**: 이번 논문에서는 외상성 뇌손상 진단을 위한 자동 방사선 보고서 생성을 위한 혁신적인 AI 기반 접근법을 제안합니다. AC-BiFPN (Augmented Convolutional Bi-directional Feature Pyramid Network)과 Transformer 아키텍처를 통합하여 CT 및 MRI 스캔과 같은 복잡한 의료 영상 데이터를 처리합니다. 이 모델은 의료 영상에서의 다중 스케일 기능을 활용하여 intracranial hemorrhages와 같은 복잡한 이상을 효과적으로 탐지합니다.

- **Technical Details**: 제안된 모델은 AC-BiFPN을 통해 CT 및 MRI 이미지에서 다중 스케일 특징을 추출하고, Transformer를 통해 긴 의존성을 모델링하여 일관성 있는 진단 보고서를 생성합니다. AC-BiFPN은 다양한 해상도에서의 특징을 융합하여 상세한 뇌 스캔 분석에 적용되어 중요한 정보를 놓치지 않도록 합니다. 또한, 신뢰할 수 있는 진단 보고서를 생성하기 위해 Transformer 기반 모델을 사용하여 시각적 정보와 텍스트 정보를 통합합니다.

- **Performance Highlights**: 모델은 RSNA Intracranial Hemorrhage Detection 데이터셋에서 전통적인 CNN 기반 모델들보다 우수한 진단 정확도와 보고서 생성을 보여줍니다. 이 접근법의 혁신성은 고압 환경에서도 방사선 전문의를 지원하고, 훈련생 의사에게 실시간 피드백을 제공하여 학습 경험을 강화하는 데 기여합니다. 우리의 연구는 고급 기능 추출과 Transformer 기반의 텍스트 생성을 결합하여 외상성 뇌손상의 진단 과정에서 임상적 의사결정을 개선할 가능성을 보여줍니다.



### DeepPrune: Parallel Scaling without Inter-trace Redundancy (https://arxiv.org/abs/2510.08483)
Comments:
          15 pages, 4 figures, please check out the project page: this https URL

- **What's New**: 이번 논문에서는 DeepPrune이라는 새로운 프레임워크를 제안합니다. 이 방법은 파라랠(Parallel) 추론 과정에서 중복된 경로를 효과적으로 줄이는 동시에 다양한 답변을 유지할 수 있도록 설계되었습니다. 연구 결과, 현재의 패러다임에서는 약 80%의 추론 경로가 중복되는 답변을 산출하며, 이를 해결하여 계산 효율성을 크게 향상시키고자 합니다.

- **Technical Details**: DeepPrune은 동적 가지치기(dynamic pruning)를 통한 효율적인 파라랠 스케일링을 가능하게 합니다. 이를 위해 긍정적인 개념적 손실(focal loss) 및 오버샘플링 기법으로 훈련된 특별한 판별 모델(judge model)을 활용하여 부분 추론 경로에서 답변 동등성을 예측합니다. 이 접근법은 0.87 AUROC를 기록하며, 온라인 그리디 클러스터링(online greedy clustering) 알고리즘을 활용하여 중복된 경로를 동적으로 제거합니다.

- **Performance Highlights**: 종합적으로 세 가지 주요 벤치마크(AIME 2024, AIME 2025, GPQA)와 여러 추론 모델을 통해 DeepPrune의 성능을 평가한 결과, 전통적인 합의 샘플링(consensus sampling) 방식에 비해 80% 이상의 토큰 소비 감소를 달성했습니다. 또한 정확도는 대부분의 경우에 3% 이내에서 경쟁력을 유지하며, AIME25 데이터셋에서는 토큰 소비를 91.6%까지 줄이는 동시에 정확도 또한 개선되었습니다. 이러한 결과는 DeepPrune이 고성능 추론을 더욱 효율적으로 가능하게 함을 보여줍니다.



### Platform-Agnostic Modular Architecture for Quantum Benchmarking (https://arxiv.org/abs/2510.08469)
- **What's New**: 이번 연구는 퀀텀 컴퓨팅 벤치마킹의 단편화 문제를 해결하기 위해 모듈형 아키텍처(modular architecture)를 제시합니다. 이 시스템은 문제 생성(problem generation), 회로 실행(circuit execution), 결과 분석(results analysis)을 독립적인 컴포넌트로 분리하여 서로 상호 운용 가능하게 만들어 다양한 벤치마킹 접근 방식을 통합합니다. 20개 이상의 벤치마크 변형을 지원하며, Qiskit, CUDA-Q, Cirq와 같은 여러 회로 생성 API와 통합됩니다.

- **Technical Details**: 이 아키텍처의 주요 기여는 벤치마킹 프로세스를 서로 독립적이고 상호 운용 가능한 세 가지 컴포넌트로 분해한 것입니다. Gate-model 퀀텀 컴퓨팅에 초점을 맞추고 있으며, 사용자는 필요에 따라 전체 통합된 스위트를 활용하거나 개별 컴포넌트를 외부 도구와 통합하여 사용할 수 있습니다. 또한 표준화된 인터페이스를 제공하여 특수 하드웨어 시스템 및 분석 프레임워크와의 통합을 지원합니다.

- **Performance Highlights**: 이 아키텍처는 Sandia의 pyGSTi를 통한 고급 회로 분석 및 CUDA-Q를 활용한 다중 GPU HPC 시뮬레이션과의 성공적인 통합을 통해 검증되었습니다. 새로운 벤치마크 변형인 동적 회로 인스턴스 및 퀀텀 강화 학습(Quantum Reinforcement Learning) 벤치마크를 구현하여 다양한 실행 및 분석 모드에서 쉽게 접근할 수 있도록 하였습니다. 이러한 확장성과 유연성을 통해 사용자에게 보다 체계적이고 통합된 퀀텀 시스템 평가를 가능하게 합니다.



### Integral Signatures of Activation Functions: A 9-Dimensional Taxonomy and Stability Theory for Deep Learning (https://arxiv.org/abs/2510.08456)
Comments:
          25 pages

- **What's New**: 본 논문은 활성화 함수(activation function)에 대한 새로운 체계를 제안합니다. 기존의 분석이 주로 경험적(heuristic) 접근에 의존했던 반면, 이 연구는 Gaussian propagation 통계와 비율, 경계 변화 등의 정량적 지표를 결합하여 새로운 수학적 프레임워크를 개발했습니다. 이를 통해 표준 활성화 함수에 대한 체계적인 분류가 가능해졌으며, 안정성(robustness)과 모델 성능 향상에 기여할 수 있습니다.

- **Technical Details**: 논문은 아홉 차원의 적분 서명(integral signature) S_sigma(phi) 를 정의하며, Gaussian 통계(m1, g1, g2, m2, eta)와 극한 경사(asymptotic slopes, alpha_plus, alpha_minus), 그리고 총 변화(total variation, TV(phi'))를 포함합니다. 이 서명은 다양한 활성화 함수의 표현력을 정량화하며, Lyapunov 기반의 정리들을 통해 모델의 안정성을 보장합니다. 추가적으로, 커널 관점(kernel perspective)에서의 차원 비례 Hessian 경계와 매끄러움(smoothness)간의 관계를 탐구합니다.

- **Performance Highlights**: 이 프레임워크를 통해 ReLU, leaky-ReLU, tanh, sigmoid, Swish, GELU, Mish, TeLU와 같은 여덟 가지 활성화 함수의 정확한 분류가 이루어졌습니다. 이론적 예측은 수치적 Gauss-Hermite 및 Monte Carlo 검증을 통해 확인되었습니다. 결과적으로, 이 연구는 활성화 함수 설계에 대한 객관적이고 실행 가능한 지침을 제공하여, 시험 및 오류(trial-and-error)에서 벗어난 안정적이고 수학적으로 입증된 선택을 위한 기초를 마련합니다.



### gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity (https://arxiv.org/abs/2510.08450)
Comments:
          22 pages, 22 figures, 7 tables

- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)에서 발생하는 over-squashing 문제를 재조명하며, 특히 모델의 저장 및 검색 용량 관점에서 분석합니다. 연구자들은 정보 병목 현상이 저장 용량을 포화시킬 수 있음을 입증하기 위해 새로운 합성(task) 작업을 도입했습니다. 또한, 기존의 연관 기억 모델(associative memory) 및 형태 모델링(sequence modeling)에서 영감을 받아, 저장 용량이 향상된 새로운 GNN 아키텍처를 제안하고 있습니다.

- **Technical Details**: 기존 GNN 아키텍처는 노드의 표현을 업데이트하기 위해 이웃 노드와 정보를 교환하는 메시지 전달(message-passing) 패러다임을 따릅니다. 그러나 많은 레이어로의 확장이 어려운 이유는 over-smoothing과 over-squashing이라는 두 가지 중요한 문제 때문입니다. 저자들은 over-squashing을 저장 용량의 한계로 설명하고, 이를 해결하기 위해 최근의 xLSTM 아키텍처를 기반으로 새로운 MPNN 아키텍처를 제안하였습니다.

- **Performance Highlights**: 새롭게 제안된 GNN 아키텍처는 저장 용량 향상에 성공했고, 결과적으로 합성 작업 및 다양한 실제 그래프 벤치마크에서 우수한 성능을 보여주고 있습니다. 실험 결과는 capacity over-squashing이 sensitivity over-squashing과 별개로 발생할 수 있음을 입증합니다. 이 아키텍처는 정보 저장 및 검색 능력이 개선된 점이 주목할 만합니다.



### Synthetic Series-Symbol Data Generation for Time Series Foundation Models (https://arxiv.org/abs/2510.08445)
Comments:
          63 pages, NeurIPS 2025 accepted

- **What's New**: 최근 시간 시계열 분석(TSA)을 위한 파운데이션 모델들이 주목받고 있으나, 훈련 데이터의 부족과 불균형이 큰 도전 과제가 되고 있습니다. 이를 해결하기 위해 고안된 S2S² 데이터 생성 메커니즘은 시간 시계열 데이터와 해당하는 기호 표현을 고품질로 생성할 수 있게 합니다. 이러한 접근법을 바탕으로, 기호 정보를 활용하여 시계열 표현을 향상하는 SymTime이라는 사전 학습된 모델을 개발했습니다.

- **Technical Details**: SymTime 모델은 대규모의 다양한 시계열 데이터에서 자가 감독 또는 비감독 방식으로 사전 훈련된 심층 신경망으로, 일반화 가능한 시계열 표현을 학습하여 소규모 샷 또는 전이 학습을 통해 다양한 다운스트림 시계열 작업을 효율적으로 수행할 수 있습니다. 본 논문에서는 복잡한 동적 시스템을 나타내는 매핑 이론인 Takens의 정리에 기초하여, 시간 시계열이 복잡한 동적 시스템의 표현으로 작동할 수 있다는 점을 포괄적으로 다뤘습니다. S2S² 데이터 생성 과정은 다변량 입력-출력 기호 표현을 구축하고, 이를 통해 고유한 시계열 데이터를 생성하는 것을 포함합니다.

- **Performance Highlights**: SymTime은 다섯 가지 주요 TSA 작업에서 경쟁력 있는 성능을 보이며, 실제 데이터셋으로 사전 훈련된 파운데이션 모델과 비교할 때 유사하거나 그 이상으로 성능을 발휘합니다. 또한, S2S² 데이터셋의 규모가 다운스트림 작업에서 모델 성능과 직접적으로 연관되어 있음을 입증했습니다. 이러한 결과는 생성된 데이터와 사전 훈련 메커니즘이 데이터 부족 문제를 해결하고 작업 성능을 향상시킬 잠재력이 있음을 강조합니다.



### Gaze on the Prize: Shaping Visual Attention with Return-Guided Contrastive Learning (https://arxiv.org/abs/2510.08442)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구는 시각적 강화학습(Visual Reinforcement Learning, RL) 에이전트가 고차원 이미지 데이터에서 작업과 관련된 소수의 픽셀에 따라 행동할 수 있도록 돕기 위해 'Gaze on the Prize'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 에이전트의 경험을 통해 파생된 자기 지도 신호(self-supervised signal)를 활용하여 배우는 것이며, 이는 높은 수익을 추구하는 데 초점을 맞춥니다. Gaze on the Prize는 에이전트가 성공과 실패를 구분할 수 있도록 돕고, 태스크 관련 특징에 집중하게 합니다.

- **Technical Details**: 이 연구의 핵심 아이디어는 결과 차이(return differences)가 무엇이 중요한지를 드러낸다는 것입니다. 같은 상태에서도 서로 다른 결과를 가져오는 경우, 이들을 구별하는 특징은 작업과 관련이 있을 가능성이 높습니다. Gaze는 태스크 관련 특징을 식별하기 위해 대조 신호(contrastive signal)를 사용하여 훈련된 시각적 주의 메커니즘이며, 이는 기존의 RL 알고리즘과 호환 가능합니다. 긴 열 순차 연결(contrastive triplets)을 통해 에이전트의 주의 메커니즘이 성공과 실패를 구별하는 데 필요한 패턴을 학습하게 됩니다.

- **Performance Highlights**: 본 연구 방법론은 샘플 효율성이 최대 2.4배 개선되었으며, 기존 알고리즘이 학습에 실패했던 작업을 해결할 수 있음이 입증되었습니다. 이는 ManiSkill3 벤치마크의 다양한 조작 작업에서 수행되었으며, 강화학습 알고리즘이나 하이퍼파라미터를 변경하지 않고도 이루어졌습니다. Gaze on the Prize는 기존의 비주얼 RL 알고리즘에 플러그인할 수 있는 접근법으로, 성능을 향상시키면서도 기본 구조를 유지합니다.



### xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning (https://arxiv.org/abs/2510.08439)
Comments:
          24 Pages, 4 Figures, 2 Tables

- **What's New**: xRouter는 현대 LLM(대형 언어 모델)의 성능과 비용 간의 균형을 조화롭게 맞추기 위해 개발된 도구이며, 학습된 라우터가 직접 질문에 답하거나 외부 모델을 호출할 수 있는 시스템입니다. 기존의 정적 에스컬레이션 규칙이나 키워드 휴리스틱의 한계를 극복하고, 비용과 성과를 인식한 보상을 통해 라우팅을 강화 학습 문제로 형성함으로써 손으로 설계된 규칙의 필요성을 없앴습니다.

- **Technical Details**: xRouter는 도구 호출 기반의 라우팅 시스템으로, 경제적 제약을 명시적으로 포함한 보상 구조를 통해 모델 호출을 최적화합니다. 라우터의 학습은 강화 학습 방식을 통해 이뤄지며, 모델의 능력에 따라 적절히 질문을 처리하고 필요할 때는 외부 모델에 위임합니다. 이를 통해 경제적 요소를 함께 고려하면서도 성능 향상과 비용 절감을 동시에 실현할 수 있습니다.

- **Performance Highlights**: xRouter는 다양한 벤치마크에서 비용-성과 트레이드오프를 성공적으로 달성했으며, 비슷한 작업 완료율을 유지하면서 상당한 비용 절감을 이루어냈습니다. 초기 탐색을 통해 다양한 실험을 진행하였고, 이를 통해 얻은 실증적 통찰들은 향후 시스템 개발에 있어 유용한 방향성을 제시합니다. 이러한 기여를 통해 xRouter는 원칙 기반의 경제적 접근 방식을 가진 LLM 조정의 첫 걸음이 되고자 합니다.



### ClauseLens: Clause-Grounded, CVaR-Constrained Reinforcement Learning for Trustworthy Reinsurance Pricing (https://arxiv.org/abs/2510.08429)
Comments:
          Accepted for publication at the 6th ACM International Conference on AI in Finance (ICAIF 2025), Singapore. Author-accepted version (October 2025). 10 pages, 5 figures

- **What's New**: 최근 발표된 ClauseLens는 재보험(再保險) 계약 가격을 위한 새로운 강화 학습(framework) 프레임워크입니다. 이 시스템은 투명하고 규제를 준수하며 위험 인식과 관련된 조항(clause) 기반의 계약 견적을 제공합니다. ClauseLens는 재보험 인용 작업을 위험 인지 제약 마르코프 결정 과정(RA-CMDP)으로 모델링하여, 법적 조항을 인용 처리하여 투명성과 감사 가능성을 높입니다.

- **Technical Details**: ClauseLens는 법적 조항 검색(legal clause retrieval), 위험 감수 정책 학습(risk-sensitive policy learning), 조항 기반 정당화 생성(clause-grounded justification generation)의 세 가지 주요 구성 요소로 구성됩니다. 이 시스템은 5백만 달러의 플로리다 허리케인 계약 요청에 대해 NAIC 지급 능력 기준을 준수하는 60% 분배 비율을 권장하고, 이 정책의 법적 근거를 자연어로 제시합니다. 이러한 구조 덕분에 높은 해석 가능성과 또한 규정 준수를 확보할 수 있습니다.

- **Performance Highlights**: ClauseLens는 다중 에이전트 재보험 시뮬레이터에서 검증되어 51%의 지급 능력 위반(solvency violations)을 줄이고, 극단적 위험(tail-risk) 성능을 27.9% 개선했습니다. 또한 조항 입각 정당화의 정확도는 88.2%에 달하며, 검색의 정밀도는 87.4%, 재현율(recall)은 91.1%에 이릅니다. 이러한 성과는 재보험 산업 데이터에 기반하여 법적 문맥을 효과적으로 결정 및 설명 경로에 통합함으로써 해석 가능하고 감사로운 인용 행동을 제공함을 보여줍니다.



### Prompts Generalize with Low Data: Non-vacuous Generalization Bounds for Optimizing Prompts with More Informative Priors (https://arxiv.org/abs/2510.08413)
Comments:
          EXAIT Workshop paper at ICML 2025

- **What's New**: 본 논문은 데이터가 부족한 상황에서도 유용한 일반화 경계를 제공하는 새로운 일반화 경계를 도출하며, 특정 작업 데이터에 최적화되거나 정보가 풍부한 정규화 사전(informative prior)을 활용하여 프롬프트 최적화를 향상시킵니다.\n프롬프트에서의 일반화 문제는 인공지능 시스템을 신뢰할 수 있게 배포하는 데 있어 매우 중요하며, 대규모 프롬프트 공간에서의 최적화와 더불어 이러한 경계를 엄밀하게 분석할 필요성을 제기합니다.

- **Technical Details**: 논문에서는 PAC-Bayes 이론에 기반하여 프롬프트 최적화 알고리즘을 위한 데이터 종속적인 일반화 경계를 도출합니다.\n이 종결 경계는 훈련 데이터의 불확실성을 처리하기 위한 방법으로서 데이터 종속 perplexity를 정규화하는 방식을 사용합니다.\n또한, 경험적인 연구를 통해 이러한 경계가 데이터가 부족한 상황에서도 유용하며, 더 나은 일반화를 위해 perplexity 정규화를 통한 최적화 알고리즘의 효율성을 보여줍니다.

- **Performance Highlights**: 제안된 경계와 정규화를 통해 얻은 경험적 결과는 최적화된 프롬프트의 일반화 성능을 실제로 개선할 수 있음을 확인합니다.\n최적화된 프롬프트는 훈련 데이터에서 뛰어난 성능을 보이지만, 일반화의 관점에서도 높은 신뢰성을 달성하는 데 기여합니다.\n이 연구는 인공지능의 다음 단계인 AGI(Artificial General Intelligence)로 나아가는 여정에서 매우 중요한 이론적 및 실제적 통찰을 제공합니다.



### Single layer tiny Co$^4$ outpaces GPT-2 and GPT-BER (https://arxiv.org/abs/2510.08404)
- **What's New**: 이번 논문에서는 Co$^4$ 언어 모델을 소개하며, 단일 레이어와 두 개의 헤드를 가진 8M 파라미터의 소형 기계가 BabyLM Challenge의 신기준인 GPT-2와 GPT-BERT를 능가하는 성능을 보여줍니다. Co$^4$는 단 2 에포크 동안 훈련했지만, 이전 모델들은 10 에포크 필요했습니다. 이 모델은 10M 토큰 на 대하여 매우 높은 샘플 효율성을 보여줍니다.

- **Technical Details**: Co$^4$ 모델은 두 가지 다른 입력 통합 지점을 가진 뉴런을 기반으로 하며, 이를 통해 문맥적 입력(C)와 피드포워드 입력(FF)을 잘 결합할 수 있습니다. Triadic modulation loops를 활용하여 Q, K, V의 공동 진화를 통해, Co$^4$는 저비용 운영(O(N))으로 효율적인 학습과 뛰어난 상태 추론을 가능하게 합니다. 또한, 이 모델은 10M 토큰에서 최소한의 학습 예산으로 훈련되었습니다.

- **Performance Highlights**: Co$^4$는 여러 언어 모델링 벤치마크에서 뛰어난 성능을 발휘하였습니다. 제로-샷(zero-shot) 환경에서 7개 작업 중 5개에서 GPT-2를 초과하며, 파인 튜닝 작업에서도 7개 중 6개에서 우수한 결과를 기록했습니다. 이러한 성과는 현재의 딥러닝 패러다임과 스케일링 법칙을 재고할 필요성이 있음을 제안합니다.



### FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts (https://arxiv.org/abs/2510.08396)
Comments:
          NeurIPS 2025 accepted paper

- **What's New**: FlyLoRA는 Fly의 후각 회로에서 영감을 받아 개발된 모형으로, 기존의 LoRA 방법에서 나타나는 파라미터 간의 간섭을 줄여줍니다. 이 방법은 낮은 차원의 전문가 활성화(rank-wise expert activation)를 통해 더 나은 수행 성능을 제공합니다. 이러한 혁신은 단일 작업 내에서의 파라미터 간섭을 효과적으로 해소하며, 기존의 MoE 기반 LoRA 방법들보다 효율성을 높여주는 목표를 달성합니다.

- **Technical Details**: FlyLoRA는 매트릭스 𝑨를 고정된 희소 랜덤 프로젝션(sparse random projection)으로 취급하며, 여기서 각 LoRA 컴포넌트는 서로 다른 랜덤 프로젝션을 통해 약 orthogonal 하게 매핑됩니다. 이를 통해 FlyLoRA는 intra-task(작업 내) 간섭을 최소화하고, multi-task(다중 작업) 연합에서도 비교적 독립적으로 작동하도록 설계되었습니다. 이러한 구조는 AI 기술에 생물학적 구조의 영감을 결합한 결과로, 효율적인 파라미터 사용을 가능하게 합니다.

- **Performance Highlights**: FlyLoRA는 일반 지식 이해, 과학 질문 답변, 수학 추론, 코드 생성 등 다양한 작업에서 기존 방법들 대비 일관된 성능 향상을 보여줍니다. 실험 결과를 통해 FlyLoRA가 계산 효율과 파라미터 간섭 감소 모두를 이뤄내며, 최신 기계 학습 기준에 부합하는 성능을 입증하였습니다. 이러한 결과는 AI 기술 진보의 중요한 원동력이 될 것으로 기대됩니다.



### Detecting Legend Items on Historical Maps Using GPT-4o with In-Context Learning (https://arxiv.org/abs/2510.08385)
- **What's New**: 이 연구는 역사적 지도에서 범례(legend)를 자동으로 추출하기 위한 새로운 방법을 발표합니다. LayoutLMv3를 사용한 레이아웃 감지와 GPT-4o를 활용한 인컨텍스트 학습(in-context learning) 기법을 결합하여 범례 항목과 그 설명을 연결합니다. 실험 결과, 구조화된 JSON 프롬프트를 사용한 GPT-4o가 기존 방법보다 뛰어난 성능을 보여주며, 역사적 지도의 색다른 시각적 스타일에 대한 인덱싱과 검색 가능성을 높이고 있습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다: 첫째, 전방위 맵에서 범례 영역을 분리하는 것입니다. 이를 위해 LayoutLMv3를 사용하여 범례가 포함된 블록을 식별하고, 그 후 GPT-4o를 통해 범례 항목과 설명에 대한 경계 상자(bounding box)를 생성합니다. 사용되는 JSON 프롬프트는 작업 정의와 함께 예시 항목과 설명의 쌍을 포함하여 GPT-4o에 제공됩니다.

- **Performance Highlights**: 제안된 방식의 성능은 88%의 F-1 점수와 85%의 IoU를 기록했습니다. GPT-4o는 다양한 예시 수(5, 10, 15, 20)에 따라 성능이 향상됨을 보여주었습니다. 이 연구는 역사적 지도 아카이브에서 범례 해석의 필요성을 강조하며, 스케일러블한 지리공간 검색과 마이닝을 위한 기초를 제공합니다.



### Airy: Reading Robot Intent through Height and Sky (https://arxiv.org/abs/2510.08381)
- **What's New**: 이 연구는 인간과 공유하는 공간에서 활동하는 산업 로봇의 불투명한 의사결정이 어떻게 인식되고 이해될 수 있는지를 탐구합니다. 'Airy'라는 아트워크를 통해 두 개의 강화 학습을 통해 훈련된 로봇 팔이 요를 높이 들어 올리는 경쟁을 장식하며, 이를 통해 관람객이 기계의 의도를 직관적으로 이해할 수 있도록 돕기 위한 실험을 합니다. 이 설치물은 시각적 경험을 통해 로봇의 전략적 행동과 협력 및 갈등을 실시간으로 읽을 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 이 작품은 세 가지 디자인 원칙에 기초하여 구성되었습니다: (1) 경쟁을 통해 명확한 메트릭 제공, (2) 관객이 인식할 수 있는 친숙한 동작 구현, (3) 센서와 감각의 매핑을 통해 로봇의 협력 또는 경쟁을 시각적으로 표현합니다. 두 개의 로봇 팔이 각 높이를 측정하고, 이 높이는 숲과 날씨의 시각적 변화를 통해 관람객에게 전달되며, 결과적으로 내부 기계의 상태를 시각적으로 표현합니다. 이 시스템은 관관객이 로봇의 행동을 실시간으로 분석하고 해석할 수 있도록 하며, 기계의 의도를 읽도록 돕습니다.

- **Performance Highlights**: 다섯 개의 국제 전시에서 관람객들은 로봇의 움직임을 통해 전략, 갈등 및 협력을 직관적으로 읽을 수 있었습니다. 그들은 로봇의 행동을 통해 시스템의 내부 상태를 반영하는 감정적 반응을 보였으며, 이는 알고리즘의 동기를 이해하는 데 기여하였습니다. 경쟁을 통해 로봇 팔의 움직임을 시각적 내러티브로 변환함으로써 관람객은 경험적 공감의 능력을 발휘하며, 진정한 경쟁과 공정성을 기반으로 한 공공 인터페이스를 구현하게 되었습니다.



### Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception (https://arxiv.org/abs/2510.08352)
- **What's New**: 본 논문에서는 Distance-Annotated Traffic Perception Question Answering (DTPQA)라는 새로운 벤치마크를 소개합니다. DTPQA는 교통 장면 내 시각 기반 질문에 초점을 맞춘 첫 번째 Visual Question Answering (VQA) 벤치마크로, 거리 주석이 포함되어 있어 연구자들이 모델의 인식 능력을 평가할 수 있습니다. 또한, 이 연구는 소형 Vision-Language Models (VLMs)의 인식 성능을 평가했으며, 단순한 질문에 대한 성능이 인간보다 약 60%라는 점을 강조하고 있습니다.

- **Technical Details**: VLMs는 일반적으로 강력한 Large Language Models (LLMs)를 기반으로 하여 시각과 언어를 통합하는 능력이 뛰어납니다. 최근 연구들은 주로 SOTA(Small State-of-the-Art) 대형 모델을 평가했으나, DTPQA 벤치마크는 소형 VLM들을 대상으로 하여 그들의 인식 능력을 깊이 있게 분석합니다. 각 DTPQA 샘플은 이미지, 질문, 정답, 거리 정보를 포함하고 있어 모델의 성능을 정량적으로 평가하는 데 도움을 줍니다.

- **Performance Highlights**: 존재하는 다양한 VQA 벤치마크와 달리 DTPQA는 단순한 인식만을 요구하는 질문들로 구성되어 있습니다. 연구에 따르면, 소형 VLM 모델들은 문제의 거리 증가에 따라 성능이 저하되는 경향이 있으며, 특정 인식 과제는 여전히 도전적입니다. 이 연구는 소형 VLMs의 인식 성능이 전반적으로 불충분하다는 점을 밝혀내며, 향후 안전-critical한 응용 분야에서의 신뢰성을 확보하기 위한 기준을 제시합니다.



### DeepEN: Personalized Enteral Nutrition for Critically Ill Patients using Deep Reinforcement Learning (https://arxiv.org/abs/2510.08350)
- **What's New**: DeepEN은 중환자에게 맞춤형의 장내 영양(intravenous nutrition, EN)을 제공하기 위한 심층 강화학습(deep reinforcement learning, RL) 프레임워크로 소개됐다. MIMIC-IV 데이터베이스의 11,000명 이상의 ICU 환자로부터 오프라인 학습하여, 매 4시간마다 환자의 생리적 변화에 맞춤화된 칼로리, 단백질, 수분 섭취 권장 사항을 생성한다. 이는 단기적인 생리학적 및 영양 관련 목표를 장기적인 생존 결과와 균형을 이루도록 설계된 맞춤형 보상 함수를 통합하여 이루어진다.

- **Technical Details**: DeepEN은 환자의 진단, 검사 결과, 생체 신호 및 과거 치료를 기반으로 한 포괄적인 환자 특징 목록에 따라 맞춤형 권장 사항을 생성한다. Dueling Double Deep Q-Network(D3QN) 알고리즘을 사용하여 안전하고 임상적으로 실행 가능한 정책을 학습하고, 이를 위해 보수적인 Q-learning 정규화를 적용한다. 강화학습의 훈련 과정에서 실수의 위험을 최소화하기 위한 조치를 취하여, 복잡한 환경에서도 효과적으로 작동할 수 있도록 한다.

- **Performance Highlights**: DeepEN은 기존의 임상 관행이나 가이드라인 기반 정책보다 우수한 성과를 보였다. 실제로 예상 사망률을 3.7% 감소시켰고 (18.8% 대 22.5%), 주요 영양 바이오마커에서 개선이 관찰되었다. 이는 데이터 기반의 안전한 맞춤형 EN 요법의 잠재력을 강조하며, 기존의 가이드라인이나 경험적 접근 방식보다 더 나은 결과를 달성할 수 있는 가능성을 제시한다.



### Learning What's Missing: Attention Dispersion and EMA Stabilization in Length Generalization (https://arxiv.org/abs/2510.08341)
Comments:
          10 pages, 5 figures, 2 tables

- **What's New**: 이번 연구에서는 트랜스포머 모델에서 길이 일반화(length generalization) 개념을 탐구했습니다. 구체적으로, 설정 보완(task complement) 작업을 통해 입력 시퀀스에서 누락된 토큰에 대한 균일 분포를 예측하는 능력을 다룹니다. 이 작업은 보드 게임 스타일의 추론에 중요한 요소로 작용하며, 두 가지 주요 이론적 결과를 도출합니다: 첫째, 단일 계층의 주의(attention) 전용 트랜스포머에 대한 강한 경계가 입증되었습니다. 둘째, 모델이 길이 1과 2의 입력에서 균형 잡힌 로짓(logit) 이동을 달성할 수 있으면, 더 긴 시퀀스에 대한 일반화도 가능하다는 점입니다.

- **Technical Details**: 연구에서는 주의 산술식이 긴 시퀀스에서 활성화 가중치의 분산(attention dispersion)을 증가시켜 추론의 정밀도를 감소시킨다는 기계적 해석(mechanistic reading)이 제시되었습니다. 이를 방지하기 위해 드롭아웃(dropout)을 증가시키면 값 벡터(value vectors)의 진폭을 높여 이 효과를 완화할 수 있을 것으로 가설을 세웠습니다. 또한, 훈련 동역학(training dynamics)을 분석하여 짧은 시퀀스 다음에 나올 수 있는 많은 토큰 중에서 샘플링이 이루어지면서 기울기(gradients)가 노이즈화(noisy)된다는 두 번째 장애물을 규명했습니다. 따라서 편향 수정 지수이동평균(BEMA, Bias-corrected Exponential Moving Average)의 사용이 이 문제를 완화할 수 있을 것이라고 가정했습니다.

- **Performance Highlights**: 랜덤 하이퍼파라미터 검색을 통해 제안된 전략이 성능 향상에 기여함을 입증했습니다. 더 복잡한 설정을 위해 OthelloGPT 모델에 대한 길이 일반화 실험을 통해 BEMA가 이 경우에도 성능 지표를 강력하게 개선함을 확인했습니다. 연구 결과는 강조된 두 가지 메커니즘인 드롭아웃과 BEMA가 통일된 다음 토큰 분포를 학습하며 일반화 성능 향상에 기여한다는 점을 보여주었습니다.



### Iterated Agent for Symbolic Regression (https://arxiv.org/abs/2510.08317)
Comments:
          45 pages, 22 figures, 8 tables

- **What's New**: 본 연구는 데이터에서 수학적 표현을 자동으로 발견하는 기법인 기호 회귀(symbolic regression, SR) 분야에서의 새로운 프레임워크인 IdeaSearchFitter를 소개합니다. 이 프레임워크는 대형 언어 모델(LLM)을 의미론적 변형기로 활용하여, 자연어 논거에 따라 후보 표현을 생성함으로써 개념적으로 일관되고 해석 가능한 모델의 발굴을 선호합니다. IdeaSearchFitter는 다양한 도전 과제에서 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: IdeaSearchFitter는 LLM을 사용하여 해석 가능한 ansatz 공간 내에서 진화적 검색을 진행합니다. 이 시스템은 자연어로 표현된 논거를 토대로 후보를 생성하며, 개념적으로 의미 있는 변형을 보장합니다. 전통적인 구문 기반의 검색 방법에서 의미 기반의 검색 방법으로의 전환은 SR에서의 오버피팅(overfitting) 문제를 줄여주고 정확성, 복잡성, 해석 가능성 간의 균형을 최적화하는 데 큰 이점을 제공합니다.

- **Performance Highlights**: 실험 결과, IdeaSearchFitter는 Feynman Symbolic Regression Database (FSReD)에서 80% 이상의 회수율을 달성하고 고노이즈 조건에서도 71.7%의 성능을 보이며 여러 기초 방법론들을 초월합니다. 실제 데이터셋에 대한 응용에서도 메커니즘적으로 정렬된 모델을 발견하고 고에너지 물리학 응용에서 프로톤의 복잡한 내부 구조를 설명하는 간단한 방정식을 도출해내었습니다. 이 연구는 IdeaSearch의 일환으로 공개되어 있어 다양한 물리적 응용이 가능합니다.



### Counterfactual Identifiability via Dynamic Optimal Transpor (https://arxiv.org/abs/2510.08294)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이 논문은 관찰 데이터에서 고차원 다변량 결과의 반사실적( counterfactual ) 식별에 대한 질문을 다룹니다. Pearl (2000)의 주장에 따르면, 반사실적이 식별 가능해야 인과 관계 주장을 정당화할 수 있습니다. 최근 반사실적 추론에 관한 연구는 유망한 결과를 보였지만, 식별 부족으로 그 추정의 인과적 유효성을 저해한다고 지적합니다. 본 논문에서는 연속 시간 흐름을 이용하여 다변량 반사실적 식별을 위한 기초를 확립했습니다.

- **Technical Details**: 반사실적 전송 맵이 고유하고 단조로운 성질을 가질 수 있는 조건을 역동적 최적 수송(dynamic optimal transport) 이론과 결합하여 규명합니다. 기계의 단조성은 주어진 개입에 대해 사실적 결과들의 순위를 저장하는 데 필수적입니다. 다차원 변수에 대해 반사실적 식별을 가능하게 하는 제약 조건 집합을 특성화합니다.

- **Performance Highlights**: 구체적인 사례를 통해 반사실적 바탕 이론을 검증하고, 실제 이미지에서 공리적 반사실적 일관성이 개선되었음을 입증합니다. 이 모델은 역동적 최적 수송 이론을 인과적 메커니즘 분석에 활용하여 높은 식별 성능을 보여주고 있습니다. 반복 측정을 통해 추론의 일관성을 보장할 수 있음을 확인하였습니다.



### Learning Neural Exposure Fields for View Synthesis (https://arxiv.org/abs/2510.08279)
Comments:
          Accepted to NeurIPS 2025. Project page available at this https URL

- **What's New**: 이 논문에서는 신경 노출 필드(Neural Exposure Fields, NExF)라는 혁신적인 기술을 제시하여, 도전적인 실제 캡처 자료로부터 고품질의 3D 장면을 일관되게 재구축할 수 있도록 하였습니다. 기존의 기술들이 고정된 이미지/픽셀에서 최적의 노출 값을 선택하는 것과 달리, NExF는 3D 포인트별로 최적 노출 값을 예측하는 신경 필드를 학습하여 3D 환경에서 최적화를 수행합니다. 이로 인해 높은 동적 범위의 시나리오에서도 정확한 뷰 합성이 가능해졌습니다.

- **Technical Details**: NExF는 3D 장면 표현과 노출 필드를 공동으로 학습하는 시스템을 가지고 있으며, 이는 새로운 잠재적 노출 조건화 메커니즘(latent exposure conditioning mechanism)을 통해 이루어집니다. 또한, 3D 정보의 집합을 통해 우리의 모델은 3D 기존성이 높아지며, 별도의 2D 톤 매핑(tone mapping) 프로세스가 필요하지 않습니다. 이 방법은 이전 작업들보다 훈련 속도가 빠르고, 여러 벤치마크에서 55% 이상 성능을 개선하는 것을 보여주었습니다.

- **Performance Highlights**: NExF는 도전적인 실제 데이터에 대해 우수한 성능을 입증하였으며, 이는 최신 기술들에 비해 55% 이상의 향상을 보여줍니다. 특히 다양한 노출 변화가 있는 큰 규모의 캡처 자료에서도 고품질의, 3D 일관된 결과를 생성할 수 있습니다. 이러한 성과는 신경 3D 장면 표현을 더욱 복잡한 실세계 사용 사례에 가까운 형태로 발전시키는 중요한 진전을 의미합니다.



### A Distributed Emulation Environment for In-Memory Computing Systems (https://arxiv.org/abs/2510.08257)
Comments:
          6 pages, 5 figures, 2025 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)

- **What's New**: 이 논문에서는 인-메모리 컴퓨팅(In-Memory Computing) 기술 기반의 통합 회로의 빠른 프로토타이핑을 위한 배포 가능하고 확장 가능한 에뮬레이션 시스템을 제시합니다. 특별히, 이 에뮬레이터는 개발 초기 단계에서 다양한 시스템 측면을 분석하고 마이크로코드를 테스트하며, 실제 칩이 존재하기 전에도 애플리케이션을 배포할 수 있습니다. 제안된 에뮬레이터의 실험 결과는 이 기술의 유용성을 입증합니다.

- **Technical Details**: 제안된 에뮬레이터(IMCE)는 프론트 엔드 유닛(IMCE-FE), 여러 처리 유닛(IMCE-PU), 구성 및 데이터 분석 서버(IMCE-CDA)로 구성되어 있습니다. IMCE-PU는 아날로그 및 디지털 처리 장치 각각을 사용하여 DNN의 모든 노드를 지원하고 에뮬레이션할 수 있습니다. 특히, IMCE-CDA 서버는 AI 모델을 IMCE의 내부 아키텍처와 매핑하는 도구를 활용하여 DNN의 특성에 따라 IMCE를 설정합니다.

- **Performance Highlights**: IMCE는 여러 DNN 노드를 동시에 지원할 수 있으며, 각 IMCE-PU는 ARM Cortex-A53 및 Cortex-R5F 프로세서를 활용하여 높은 처리 성능을 유지합니다. 에뮬레이터는 최대 파이프라이닝을 통해 처리 속도를 향상시키며, 각 처리 단계의 실행 시간과 자원 활용 데이터를 수집하여 디버깅 및 통계 분석을 용이하게 합니다. 이 시스템은 다양한 DNN 모델의 동작 분석에 효과적이며, 실험 결과는 성능을 뒷받침합니다.



### Mix- and MoE-DPO: A Variational Inference Approach to Direct Preference Optimization (https://arxiv.org/abs/2510.08256)
- **What's New**: 최근 등장한 Direct Preference Optimization (DPO)은 대형 언어 모델(LLMs)을 사용자 선호에 맞추는 간단하고 효과적인 대안으로 자리잡았습니다. 본 연구에서는 DPO의 한계를 극복하기 위해 Mix- 및 MoE-DPO 프레임워크를 제안합니다. 이는 DPO를 부드러운 혼합 모델과 전문가 혼합(Combination of Experts) 아키텍처로 확장하여, 다양한 선호 분포에 적응할 수 있도록 합니다.

- **Technical Details**: 새로운 Mix- 및 MoE-DPO 방법론은 전문가 배치를 위한 잠재 변수 모델을 도입하고, 변분 증거 하한(ELBO)을 최적화합니다. 이 방식은 사용자 특정 혼합 정책을 가능하게 하는 입력 의존적 부드러운 게이팅을 통해 보상 및 정책의 전문화를 촉진합니다. 또한, 이러한 접근법은 모듈화된 배포를 지원하여 기존 모델과의 효율적인 통합 및 사용자 개인화가 가능합니다.

- **Performance Highlights**: 기술적으로 Mix- 및 MoE-DPO는 변수를 고정한 Mix-DPO와, 입력 의존적 가중치를 가지는 MoE-DPO로 나뉘어, 다양한 인과 모델 크기와 다중 선호 데이터셋에서 그 성능을 검증하였습니다. 이로 인해 Mix- 및 MoE-DPO는 무수히 많은 적용 가능성을 가진 선호 기반 LLM 정렬 방법을 제시합니다.



### Opponent Shaping in LLM Agents (https://arxiv.org/abs/2510.08255)
Comments:
          29 pages, 15 figures, 15 tables

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)를 기반으로 한 에이전트들 간의 전략적 상호작용인 opponent shaping에 대한 최초의 조사를 수행했습니다. 기존의 알고리즘은 LLM에 맞지 않는 구조적 결함이 있기 때문에, 저자들은 이를 해결하기 위해 ShapeLLM이라는 새로운 알고리즘을 고안했습니다. 이 모델은 transformer 기반의 에이전트에서 동작하도록 설계된 model-free 방법입니다.

- **Technical Details**: ShapeLLM은 Proximal Policy Optimization (PPO) 방법을 사용하여 LLM 에이전트의 학습 역학을 서로에게 전략적으로 영향을 미치도록 실험했습니다. 연구팀은 반복적 매트릭스 게임을 이용하여 LLM 에이전트들이 서로의 학습 동태에 미치는 영향을 분석하고, 이를 통해 경쟁적 및 협력적 시나리오에서 전략적 상호작용의 실효성을 평가했습니다. 해당 연구는 LLM 에이전트가 독립적으로 행동하는 것이 아닌, 서로에게 영향을 주고받으며 학습한다는 중요한 사실을 증명합니다.

- **Performance Highlights**: LLM 에이전트들은 다양한 게임 이론적 환경에서 효과적으로 상대를 유도하여 경쟁적인 상황에서는 유리한 균형으로, 협력적인 상황에서는 상호 이익을 위한 행동을 촉진할 수 있음을 보였습니다. 연구 결과, LLM들이 지속적인 상호작용을 통해 각자의 행동을 조정하고, 더 나아가 프로소셜(prosocial) 행동을 이끌어내는 데 기여할 수 있는 가능성이 있음을 보여줍니다.



### Contrastive Decoding for Synthetic Data Generation in Low-Resource Language Modeling (https://arxiv.org/abs/2510.08245)
Comments:
          13 pages, 3 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 교육에 있어 제한된 데이터 문제를 해결하기 위해 인공지능에 의해 생성된 합성 데이터(synthetic data)를 사용합니다. 특히, Contrastive Decoding(𝖢𝖣	ext{CD})을 활용하여 모델 성능을 향상시키는 것을 목표로 합니다. 원본 말뭉치에서 좋은 모델과 나쁜 모델 간의 상대적 차이를 이용하여 합성 말뭉치를 생성하고, 이를 실제 데이터에 혼합하여 훈련하는 방식입니다.

- **Technical Details**: 연구에서는 100M 단어의 원본 말뭉치에서 훈련된 좋은 모델과 나쁜 모델의 상대적 차이를 기반으로 100M 단어의 합성 말뭉치를 생성했고, 이를 원래 훈련 데이터와 결합하여 모델을 훈련했습니다. Contrastive Decoding(𝖢𝖣	ext{CD})은 더 우수한 모델에서 나오는 신호를 확대하여 더 일관되고 유익한 텍스트를 생성하는 데 초점을 맞춥니다. 이 방법이 다른 전통적인 샘플링 방식에 비해 훈련 과정에서 어떻게 기여하는지 분석했습니다.

- **Performance Highlights**: 합성 데이터로 훈련한 모델은 언어 모델링 목표 및 여러 다운스트림 작업에서 성능이 향상되었습니다. 특히, 𝖢𝖣	ext{CD}에서 생성된 데이터를 사용한 훈련이 더 많은 추론 능력이 요구되는 작업에 강력한 효과를 보이며, 전통적인 샘플링을 통해 생성된 데이터는 표면적인 언어적 능력에 의존하는 작업에서 더 유리한 결과를 가져왔습니다.



### The Hidden Bias: A Study on Explicit and Implicit Political Stereotypes in Large Language Models (https://arxiv.org/abs/2510.08236)
- **What's New**: 이번 연구는 정치적 편견과 고정관념의 전파를 8개의 주요 대형 언어 모델(Large Language Models, LLMs)을 대상으로 조사하였으며, 두 차원 정치적 컴퍼스 테스트(Political Compass Test, PCT)를 사용하여 이러한 모델들의 내재된 정치 성향을 분석하였습니다. 특히, 이 연구는 첫 번째로 명시적 고정관념(persona prompting)과 암묵적 고정관념을 비교하는 체계적인 방법론을 적용하였습니다. 연구 결과, 대다수 모델이 일관되게 좌파 성향을 지니고 있음을 밝혀냈고, 암묵적 고정관념이 명시적 고정관념보다 더욱 두드러진다는 흥미로운 사실을 발견하였습니다.

- **Technical Details**: 이 연구에서는 LLM의 정치적 편견을 평가하는 체계적 방법론을 채택하였으며, 두 차원 PCT를 표준 평가 프레임워크로 활용했습니다. PCT는 경제적 좌우축(left-right)과 사회적 자유-권위적(libertarian-authoritarian) 축을 통해 이념적 입장을 평가하며, 이를 기반으로 한 공개된 조사 및 이전 연구 결과와 비교하였습니다. 이 연구는 LLM 내에서 암묵적 및 명시적 고정관념의 상호작용과 정렬을 분석하여, 모델들간의 정치적 편향이 어떻게 나타나는지를 밝히며, 모델의 편향이 나타나는 방식에서의 투명성을 강조하였습니다.

- **Performance Highlights**: 연구 결과, 모든 조사된 LLM은 일관되게 좌파 성향을 보였으며, 고정관념의 구체적인 표현은 모델 간에 상당한 차이를 보였지만, 언어 변화를 통해 드러나는 암묵적 고정관념이 명시적 고정관념보다 더 뚜렷하게 나타났습니다. 이 연구는 암묵적과 명시적 고정관념이 상당한 정렬을 보임을 보여주어, 모델들이 자신의 편향에 대해 어느 정도 인식하고 있다는 것을 시사합니다. 따라서 이 연구는 LLM의 정치적 편향이 사회에 미치는 영향을 심층적으로 분석한 중요한 기초 자료를 제공합니다.



### Expressive Value Learning for Scalable Offline Reinforcement Learning (https://arxiv.org/abs/2510.08218)
Comments:
          24 pages, 5 figures

- **What's New**: 이번 연구에서는 기존의 Offline Reinforcement Learning (RL) 방법들을 개선할 수 있는 새로운 접근법인 Expressive Value Learning for Offline Reinforcement Learning (EVOR)을 소개합니다. EVOR는 정책 증류(policy distillation)나 시간에 따른 역전파(backpropagation through time) 없이 확장 가능한 Offline RL을 개발하는 방식입니다. 또한, EVOR는 표현력이 뛰어난 가치 함수(value function)와 정책(policy)을 통합하여 훈련하며, 근본적인 데이터 셋에 대해 효과적인 최적화 및 정규화를 제공합니다.

- **Technical Details**: EVOR는 훈련 과정에서 흐름 맞추기(flow matching)를 통해 최적화된 정규화된 Q-함수를 학습합니다. 추론(inference) 시간에는 거부 샘플링(rejection sampling)을 활용하여 유연한 가치 함수에 따라 정책을 추출합니다. 이러한 방법은 기존의 백프로파게이션에서 발생하는 계산 비용을 피하며, 표현이 제한된 MLP 네트워크에 의존하는 대신 더 강력한 표현력을 갖춘 모델을 사용합니다. 또한, 표현력 있는 QQ-함수를 통해 최적의 솔루션을 제공하여, Offline RL의 스케일러빌리티를 향상시킵니다.

- **Performance Highlights**: EVOR의 성능은 다양한 Offline RL 작업에서 기존 기법을 초과하는 결과를 보여줍니다. 실험 결과, EVOR는 효율적인 최적화 및 계산 가능한 검색을 가능하게 하여, 다양한 데이터셋에 대해서도 안전하고 안정적인 학습이 가능함을 입증했습니다. 특히, 표현력 있는 가치 학습이 Offline RL에 통합될 경우, 정책의 성능을 크게 향상시킬 수 있다는 점이 강조되었습니다.



### FuelCast: Benchmarking Tabular and Temporal Models for Ship Fuel Consumption (https://arxiv.org/abs/2510.08217)
Comments:
          This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution will be published in "ECML PKDD Workshop 2025 - Advanced Analytics and Learning on Temporal Data"

- **What's New**: 이 논문은 해운 산업의 연료 소비 및 배출 예측을 위한 새로운 데이터셋과 벤치마크를 소개하고 있습니다. 연구진은 세 척의 선박에서 수집한 운영 및 환경 데이터를 포함하는 데이터셋을 공개하였으며, 이는 모델 평가 및 개발에 중요한 기초 자료가 됩니다. 또한, TabPFN 기반의 인컨텍스트 학습 방법을 해운 연료 예측에 처음으로 적용하여 데이터 효율적인 접근 방식을 모색합니다.

- **Technical Details**: 연구에서는 CatBoost, LSTM, TabPFN 등 세 가지 대표적인 모델 가족을 평가하였으며, 기존의 폴리노미얼 회귀 및 다층 퍼셉트론(MLP) 등의 기준선을 포함했습니다. 모델 성능 평가를 위해 사용하는 데이터셋은 다양한 선박 유형과 기후 데이터를 포함하며, 이는 실제 해운 작업과 관련된 고해상도 정보를 제공합니다. 프로젝트의 주요 목표는 기계 학습 연구 커뮤니티가 실질적이고 데이터 효율적인 방법으로 연료 소비 모델링을 할 수 있도록 지원하는 것입니다.

- **Performance Highlights**: 결과적으로, 연구에서 제시된 모델들은 환경 조건을 통합하여 기초적인 속도만에 의존하는 모델보다 일관되게 우수한 성능을 보였습니다. TabPFN 모델은 다른 기법들보다 약간 더 나은 결과를 나타냈는데, 이는 데이터 효율적인 테이블 예측에서 기초 모델의 잠재력을 강조합니다. 또한, 시간적 맥락을 포함하는 것이 정확성을 높이는 데 기여하는 것으로 확인되었습니다.



### LLMs Learn to Deceive Unintentionally: Emergent Misalignment in Dishonesty from Misaligned Samples to Biased Human-AI Interactions (https://arxiv.org/abs/2510.08211)
- **What's New**: 이번 연구에서는 LLMs가 특정 도메인에서 악의적인 또는 잘못된 완성을 파인 튜닝(fine-tuning) 받을 경우 발생하는 'emergent misalignment' 현상을 탐구합니다. 이 연구는 LLM의 안전성을 넘어 정직하거나 기만적인 행동까지 영향을 미치는지 조사합니다. 연구 결과, LLM이 좁은 범위의 잘못된 정보로 훈련될 경우, 보다 폭넓은 불일치 행동을 보일 수 있음을 보여줍니다.

- **Technical Details**: 연구팀은 다양한 도메인에서 LLM을 misaligned completions로 파인 튜닝하여 정직성 상실 여부를 평가했습니다. 실험 결과, 1%의 misalignment 데이터를 표준 다운스트림 작업에 통합하는 것만으로도 LLM의 정직성이 20% 이상 감소한다는 것을 발견했습니다. 또한, 인간-AI 상호작용 환경에서 사용자 집단이 편향될 경우 LLM의 정직성이 저하될 수 있음을 확인했습니다.

- **Performance Highlights**: 모델이 고위험 시나리오에서 어떻게 불리한 행동을 하는지를 평가하는 연구 결과, LLM이 압박을 받을 때 자신의 신념에 모순되는 말을 할 수 있음을 보여줍니다. 또한, DeceptionBench를 통해 모델의 대답과 그 속의 논리(Cot reasoning) 간의 불일치를 평가하며, 이러한 불일치가 기만적인 행동을 강화할 수 있음을 강조합니다.



### Memory Retrieval and Consolidation in Large Language Models through Function Tokens (https://arxiv.org/abs/2510.08203)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 기억 메커니즘을 다루고 있습니다. 특히, 메모리 검색(memory retrieval)과 통합(consolidation)의 과정을 설명하기 위해 함수 토큰(function tokens) 가설을 제안합니다. 함수 토큰이 문맥에서 가장 예측력 있는 특징을 활성화하고 다음 토큰 예측을 주도한다는 점이 주목할 만합니다.

- **Technical Details**: 이론적으로, 논문은 Transformer 구조를 가진 GPT 유형의 모델을 사용하여 사전 훈련과 후 훈련을 진행합니다. 메모리는 LLM의 파라미터와 그로부터 파생될 수 있는 모든 특징으로 구성되어 있으며, 함수 토큰은 이러한 특징과 회로를 접근하고 활성화하는 역할을 합니다. 특별히, 함수 토큰은 내용 토큰(content tokens) 다음에 오는 예측을 통해 파라미터를 업데이트합니다.

- **Performance Highlights**: 연구 결과에 따르면, 함수 토큰은 문맥에서 대다수의 기능을 활성화하는 데 중요한 역할을 합니다. 또한, 사전 훈련 과정에서 함수 토큰에 이어지는 내용 토큰 예측의 손실이 주얼하여 통합이 이루어지며, 이는 LLM의 성능 향상으로 이어집니다. 이러한 발견은 LLM의 해석 가능성을 높이고 인간 가치에 대한 정렬을 강화하는 고급 학습 알고리즘 설계에 기여할 수 있습니다.



### Sentiment Matters: An Analysis of 200 Human-SAV Interactions (https://arxiv.org/abs/2510.08202)
Comments:
          Accepted for presentation at IEEE ITSC 2025 and for publication in its Proceedings. \c{opyright} 2025 IEEE. Personal use permitted; other uses require permission from IEEE, including reprinting, republishing, or reuse of any copyrighted component of this work

- **What's New**: 이 논문은 200개의 인간-공유 자율 차량(Shared Autonomous Vehicles, SAV) 상호작용 데이터셋을 소개하며, 이는 효율적인 인간-SAV 상호작용 연구에 기여합니다. 공개된 이 데이터셋은 2,136개의 대화 교환과 다양한 심리적 요소에 대한 사용자 조사를 포함하고 있습니다. 이 연구는 SAV가 사용자 수용성과 서비스 품질을 결정하는 주요 요인을明해주는 두 가지 벤치마크 사례 연구를 제공합니다.

- **Technical Details**: 논문에서는 50명의 참가자와 4개의 SAEL 5 SAV 에이전트 사이의 대화 상호작용을 바탕으로한 데이터셋 설계 및 수집 과정을 설명합니다. 우리는 랜덤 포레스트 모델링과 감정 분석 도구를 통해 결과를 분석했습니다. 또한, OpenAI의 GPT-3.5 turbo를 이용한 대화형 데이터 수집 방식을 취하여 심리적 측면을 측정했습니다.

- **Performance Highlights**: 향상된 사용자 수용성을 위하여 혼합된 전략이 담긴 SAV4가 가장 긍정적인 사용자 반응을 이끌어냈습니다. LLM 기반의 감정 분석 도구가 전통적인 텍스트 기반 방법인 TextBlob에 비해 더 정확한 사용자 감정 보고와 일치하는 결과를 보였습니다. 이 연구는 대화형 SAV 인터페이스 설계 및 진화된 감정 모델링의 기초를 마련합니다.



### Robust Canonicalization through Bootstrapped Data Re-Alignmen (https://arxiv.org/abs/2510.08178)
- **What's New**: 이번 논문은 섬세한 시각 분류(Fine-grained visual classification, FGVC) 작업에서 발생하는 기하학적 편향과 노이즈를 다루기 위한 새로운 부트스트랩핑 알고리즘을 제안합니다. 기존의 방법들은 모델의 표현력을 제한하거나 데이터의 양을 크게 늘리는 데 의존했으나, 본 연구는 교육 데이터를 반복적으로 재정렬하여 이러한 문제를 해결합니다. 이로 인해, 실제 데이터 세트에서 발생하는 다양한 회전 및 스케일에 대한 내성을 유지하면서도 더 안정적인 분류 성능을 제공합니다.

- **Technical Details**: 저자들은 공간적 변형(이동 변형을 제외한 회전 및 크기)에 대한 대응을 중점을 두고 기존 기법을 분석합니다. 그룹 이론을 활용하여 다양한 변환에 대해 균일한 표현을 제공하는 방법론과, canonicalization의 개념을 도입하여 정보를 간소화합니다. 알고리즘은 격렬하게 다양한 설정에서 수렴을 보장하며, 기능적으로 효과적이고 적은 비용으로 공간적 변동성을 줄이는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안한 방법은 네 가지 FGVC 벤치마크에서 기존의 equvariant 및 canonicalization 기법보다 일관되게 뛰어난 성능을 보이며, 데이터 세트의 재정렬이 생물 다양성 모니터링 및 관련 도메인에 있어 얼마나 중요한지를 입증합니다. 이 연구는 효과적인 데이터 정렬이 어떻게 성능 향상에 기여하는지를 보여주며, 실제 데이터 환경에서의 활용 가능성을 제시합니다.



### Leveraging Whisper Embeddings for Audio-based Lyrics Matching (https://arxiv.org/abs/2510.08176)
- **What's New**: WEALY는 음성 인식 기반의 가사 매칭 시스템으로, Whisper 디코더 임베딩(embedding)을 활용하여 가사 매칭 작업을 위한 완전 재현 가능한 프로세스를 제공합니다. 이 시스템은 텍스트 데이터나 기존 트랜스크립션에 의존하지 않고도 강력하고 투명한 기준선을 설정합니다. WEALY는 다양한 데이터셋을 통해 폭넓은 실험을 수행하면서 기존의 비재현성 문제를 해결하고 있습니다.

- **Technical Details**: WEALY는 가사 내용의 의미적 및 구조적 유사성을 식별하는 데 중점을 두며, 이를 위해 음향 신호에서 직접 가사 표현을 추출하는 이점이 있습니다. 이 과정에서, Whisper의 디코더 임베딩을 활용하여 원시 오디오 데이터를 가사 표현으로 변환하는 맞춤형 파이프라인을 설계하였습니다. MVI(musical version identification)를 보조 작업으로 활용하여 다수의 데이터셋을 기준으로 포괄적인 벤치마크를 수립합니다.

- **Performance Highlights**: WEALY는 기존의 비재현성 문제를 가진 방법들에 비해 경쟁력 있는 성능을 보이며, 실험 결과는 최신 기술 수준에 준하는 성능을 입증하고 있습니다. 또한, 다양한 손실 함수 및 풀링 전략의 영향을 분석하는 광범위한 연구를 수행하였으며, 향후 다중 모달 확장성을 탐구하고 있어 다양한 기능을 지닌 시스템으로 발전할 잠재력이 큽니다.



### NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions (https://arxiv.org/abs/2510.08173)
- **What's New**: 이 연구에서는 내비게이션 에이전트의 공간적 인지를 평가하기 위한 새로운 벤치마크인 NavSpace를 소개합니다. NavSpace는 1,228개의 경로-명령 쌍으로 구성되어 있으며, 내비게이션 에이전트의 공간적 지능을 시험하는 여섯 가지 작업 카테고리를 포함합니다. 이 평가에서는 최첨단 내비게이션 모델인 SNav를 제안하며, 이는 기존 모델들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: NavSpace 벤치마크는 내비게이션 작업의 클래식한 정의를 따르며, 에이전트는 주어진 언어 명령에 따라 다음 내비게이션 동작을 예측해야 합니다. 연구팀은 설문조사를 통해 공간 인지에 필수적인 카테고리를 식별하였고, 수집된 데이터는 기계 학습 모델과 함께 사용됩니다. 이 연구는 22개의 기존 내비게이션 에이전트를 종합적으로 평가하였습니다.

- **Performance Highlights**: SNav는 NavSpace와 실제 로봇 테스트에서 기존 내비게이션 에이전트들보다 우수한 성능을 보였으며, 이는 향후 연구에 강력한 기준을 세우는 데 기여합니다. 실험 결과, 내비게이션 분야에서 공간적 지능의 중요성과 MLLM의 한계를 강조하며 내비게이션 대형 모델의 장점을 밝혀냈습니다. 이러한 통찰은 향후 연구에서 내비게이션 에이전트의 공간적 인지를 향상시키는 방향성을 제시합니다.



### Quantum Agents for Algorithmic Discovery (https://arxiv.org/abs/2510.08159)
- **What's New**: 본 논문에서는 에피소드 기반의 강화 학습(reinforcement learning)을 통해 훈련된 양자 에이전트(quantum agents)를 소개합니다. 이 에이전트들은 이미 알려진 최적 솔루션에 접근하지 않고도 양자 푸리에 변환(Quantum Fourier Transform)의 효율적인 로그 깊이 양자 회로(logarithmic-depth quantum circuits) 및 그로버의 검색 알고리즘(Grover's search algorithm)과 같은 여러 주요 양자 알고리즘을 자율적으로 재발견합니다. 이러한 접근 방식은 알고리즘 발견(algorithmic discovery)을 위한 양자 지능(quantum intelligence)의 잠재력을 보여주며, 새로운 양자 알고리즘과 프로토콜의 자동화된 설계를 위한 길을 엽니다.

- **Technical Details**: 양자 컴퓨터(quantum computers)는 고전 컴퓨터(classical computers)와는 fundamentally 다른 원리를 활용하여 계산을 재정의합니다. 본 연구는 양자 정보와 다수의 에이전트를 포함하는 양자 생태계에서의 자율적이고 상호작용적인 행동을 강조하는 새로운 프레임워크를 제안합니다. 각 에이전트는 주변 환경을 인식하고 상호작용하는 동시에 자신의 행동을 최적화하여 보상을 최대화하는 방향으로 훈련됩니다.

- **Performance Highlights**: 제안된 프레임워크를 통해 에이전트들은 양자 알고리즘 및 프로토콜을 자율적으로 재발견하여 실제 하드웨어에서 이해 가능하고 일반화 가능한 결과를 도출합니다. 예를 들어, 양자 푸리에 변환(QFT), 그로버의 검색 알고리즘, 강력한 동전 던지기 전략 및 CHSH 게임과 같은 여러 양자 문제를 해결하며, 최적의 전략을 직접 상호작용을 통해 학습합니다. 이러한 결과는 양자 지능이 알고리즘 발견의 효과적인 도구로 기능할 수 있음을 보여 주어, 인간의 전문성을 보완하는 새로운 가능성을 제시합니다.



### DACIP-RC: Domain Adaptive Continual Instruction Pre-Training via Reading Comprehension on Business Conversations (https://arxiv.org/abs/2510.08152)
Comments:
          Accepted to the EMNLP 2025 Industry Track. Equal contribution from the first four authors

- **What's New**: 이 논문에서는 Domain Adaptive Continual Instruction Pre-Training via Reading Comprehension (DACIP-RC)이라는 새로운 지속적 사전 훈련 기법을 제안합니다. 이 방법은 기존의 다음 토큰 예측(next-token prediction)에 의존하는 사전 훈련 접근법과 달리, 대화 기록을 기반으로 다양한 작업 지침과 응답을 생성합니다. DACIP-RC는 작은 규모의 LLM이 비즈니스 대화 작업에서 더 나은 지침 일반화(instruction generalization)를 할 수 있도록 돕습니다.

- **Technical Details**: 본 연구는 DACIP-RC 방법론을 통해, 실제 대화 녹취록을 활용하여 비즈니스 관련 다양한 작업에 적응할 수 있도록 하는 지속적 사전 훈련 데이터를 구축하는 과정을 설명합니다. 데이터 샘플링 및 지침 작성 방법을 제시하며, ASR(Automatic Speech Recognition) 시스템으로 전사된 비즈니스 대화의 품질과 다양성을 보장하기 위해 120초 이상의 대화 데이터만을 사용합니다. 이후 개인 식별 정보 제거 및 데이터 형식 다각화를 진행하여 대화의 정밀도를 높입니다.

- **Performance Highlights**: DACIP-RC의 실증적 평가는 다양한 비즈니스 대화 작업에서 제로샷(zero-shot) 일반화를 크게 향상시킴을 보여줍니다. 특히, 회의 요약(meeting summarization), 행동 항목 생성(action item generation), 통화 목적 식별(call purpose identification)과 같은 작업에서의 성능을 입증하였고, 이는 기업들이 자체 데이터셋을 활용하여 도메인 적응(domain adaptation)을 효과적으로 진행할 수 있는 통찰을 제공합니다.



### AI Knowledge Assist: An Automated Approach for the Creation of Knowledge Bases for Conversational AI Agents (https://arxiv.org/abs/2510.08149)
Comments:
          Accepted to the EMNLP 2025 Industry Track

- **What's New**: 본 논문에서는 고객 문제 해결을 위한 회화형 AI 시스템에서 Retrieval Augmented Generation (RAG) 기술의 활용 증가에 대해 다루고 있습니다. 이를 해결하기 위해 과거 고객-에이전트 대화에서 QA 쌍을 자동으로 추출하여 지식 기반을 구축하는 AI Knowledge Assist 시스템을 소개합니다. 이 시스템은 LLaMA-3.1-8B 모델을 기반으로 하여 20개 회사의 실증 평가를 통해 90% 이상의 정확도로 정보 요청 질문에 답변할 수 있음을 보여줍니다.

- **Technical Details**: AI Knowledge Assist 시스템은 세 단계의 파이프라인을 통해 작동합니다. 첫 번째 단계에서는 과거의 통화 원고에서 정보 요청 질문과 에이전트의 응답을 추출합니다. 두 번째 단계에서는 이러한 QA 쌍을 의미상 유사한 그룹으로 클러스터링하여 중복된 QA 쌍을 관리합니다. 마지막으로, LLM을 활용하여 각 클러스터에서 정보를 가장 잘 요약한 대표 QA 쌍을 선택해 지식 기반에 삽입하거나 관리자에게 추천합니다.

- **Performance Highlights**: 이번 연구는 20개 클라이언트 회사의 실제 데이터에서 실험을 수행하였으며, AI Knowledge Assist 시스템은 콜센터 AI 챗봇의 기능을 대폭 향상시킴을 입증했습니다. 이 시스템을 통해 고객의 문의를 효과적으로 처리할 수 있으며, 고객 만족도를 증대시키는 데 기여할 수 있음을 나타냅니다. 실제 데이터 수집에 있어 고객 데이터의 개인 정보 보호를 우선시하여 미세한 주의가 필요함을 강조합니다.



### Think Just Enough: Sequence-Level Entropy as a Confidence Signal for LLM Reasoning (https://arxiv.org/abs/2510.08146)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서의 추론 과정에서 토큰 효율성을 향상시키기 위한 새로운 엔트로피 기반 프레임워크를 제안합니다. 이 양식은 Shannon 엔트로피를 사용하여 조기 중지를 가능하게 하며, 25-50%의 계산 비용 절감을 달성하면서도 정확도를 유지합니다. 특히, 엔트로피 기반 신뢰도 보정이 현대 추론 모델에서 나타나는 새로운 특성을 나타내며, 기존의 표준 모델에서는 발견되지 않는 점이 주목할 만합니다.

- **Technical Details**: 프레임워크는 Shannon 엔트로피를 신뢰도 신호로 활용하여, 각 모델에 대해 추론을 중단할 수 있는 임계값을 쉽게 계산할 수 있도록 합니다. 또한, 이 접근법은 정보 이론과 통계적 의사결정 이론에 기반하여 이론적인 엄밀성과 실용적인 적용 가능성을 제공합니다. 제안된 메서드는 고유의 4가지 수학적 임계값 방법을 통해 조기 중지를 위한 신뢰도 추정을 질적으로 개선합니다.

- **Performance Highlights**: 논문에서 제시한 결과는 다양한 현대 추론 최적화 모델 카테고리에 걸쳐 25-50%의 계산 비용 절감을 보여주며, 이는 정확도를 유지하면서 더 큰 효율성을 나타냅니다. 다양한 추론 벤치마크에서 신뢰도와 관련된 메커니즘의 일관된 성능을 입증하며, 저자들은 신뢰도 인식이 현대 추론 시스템의 중요한 특성임을 강조합니다.



### Improving Temporal Understanding Logic Consistency in Video-Language Models via Attention Enhancemen (https://arxiv.org/abs/2510.08138)
- **What's New**: 최근 대형 언어 모델(LLMs)에서 자기 모순적인 출력이 발생하여 신뢰성에 큰 영향을 미치는 현상이 주목받고 있습니다. 비디오-언어 모델(Video-LLMs)에서도 이와 유사한 문제에 직면하고 있으며, 특히 재구성된 질문에 대한 논리적으로 일관된 답변을 제공하지 못하는 경향이 있습니다. 본 연구는 이러한 현상의 원인을 분석하기 위해 해석 가능성을 기반으로 한 접근 방식을 채택하였습니다.

- **Technical Details**: 연구에서는 크로스 모달 주의(attention) 헤드가 서로 다른 타임스탬프에서 비디오 토큰을 효과적으로 구분하지 못하는 한 가지 주요 원인을 확인하였습니다. 이 문제를 해결하기 위해, Temporally Conditioned Attention Sharpening (TCAS)이라는 주의 강화 방법을 제안하여 모델의 시간 해상도를 개선하고, 논리적 일관성을 향상시키는 목적을 구축하였습니다. 실험 결과 TCAS 방법이 Video-LLMs의 시간 논리 일관성을 상당히 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 실험을 통해 TCAS가 다양한 비디오 기반 시간 도달(VTG) 작업에서 성능 향상을 이루었으며, 시간 논리 일관성이 시간 이해의 병목 현상임을 강조합니다. 모델이 일관성을 증대시키도록 유도함으로써 비디오 시간 이해의 중요한 진전을 이끌어냈습니다. 또한 해석 가능성 분석을 통해 TCAS가 주의 헤드의 시간 구별 능력을 개선한 사실을 검증하였습니다.



### Approximate Domain Unlearning for Vision-Language Models (https://arxiv.org/abs/2510.08132)
Comments:
          NeurIPS 2025 (Spotlight)

- **What's New**: 이 논문에서는 Approximate Domain Unlearning (ADU)이라는 새로운 문제 설정을 도입합니다. 기존의 클래스 언러닝(class unlearning)은 특정 개체 클래스의 인식을 줄이는 데 집중했지만, ADU는 지정된 도메인의 이미지에 대한 인식 정확도를 낮추면서 다른 도메인의 정확도를 유지해야 하는 요구사항을 다룹니다. 이로 인해 기존의 접근 방식이 가지는 한계와 더불어 새로운 기술적 도전과제를 제시하며, 실용적인 언러닝의 필요성을 강조합니다.

- **Technical Details**: ADU 문제 해결을 위해 Domain Disentangling Loss (DDL)이라는 방법론을 제안하여 잠재 공간(latent space)에서 도메인 분포를 명확히 분리합니다. 또한, Instance-wise Prompt Generator (InstaPG)를 도입하여 이미지별로 도메인 특성을 적응적으로 모델링합니다. 이는 강력한 도메인 일반화 능력을 가진 사전 훈련된 VLM에서 발생하는 도메인 분포의 얽힘(entanglement) 문제를 해결하기 위한 접근법입니다.

- **Performance Highlights**: 다양한 다중 도메인 이미지 벤치마크 데이터셋에 대한 실험 결과, 제안된 방법은 최신 VLM 조정 기술을 기반으로 한 강력한 베이스라인을 능가하는 성능을 보여줍니다. 이는 ADU 문제 해결을 위한 기존 방법들에 비해 명백한 개선을 나타내며, VLMs에서 실용적이고 세밀한 언러닝을 위한 가능성을 제시합니다.



### Interpreting LLM-as-a-Judge Policies via Verifiable Global Explanations (https://arxiv.org/abs/2510.08120)
Comments:
          12 pages, 2 figures, 3 tables

- **What's New**: 이 논문은 LLM-as-a-judge를 사용할 때의 잠재적 편향과 위험을 이해하기 위해 새로운 접근 방식을 제안합니다. 이 접근법은 CLoVE(Contrastive Local Verifiable Explanations)와 GloVE(Global Verifiable Explanations)라는 두 가지 알고리즘으로 구성됩니다. CLoVE는 개념 기반의 설명을 생성하고, GloVE는 이러한 설명을 글로벌 정책으로 요약합니다.

- **Technical Details**: CLoVE는 LLM-as-a-judge의 개별 결정을 설명하는 로컬 설명 알고리즘입니다. 이 알고리즘은 BECAUSE-DESPITE 형식을 통해 다각적인 관점에서 설명을 제공합니다. GloVE는 이러한 로컬 설명을 집계하여 고수준의 규칙 기반 글로벌 정책으로 요약하며, 이 과정에서 대칭을 유지합니다.

- **Performance Highlights**: 본 연구는 GloVE의 성능을 7개의 표준 평가 데이터셋을 사용해 평가했습니다. 실험 결과, GloVE는 높은 충실도를 보이며, 사용자 연구를 통해 사용자 이해도가 증가하고 있다는 점을 확인했습니다. 이 연구의 주요 기여는 CLoVE와 GloVE를 통해 LLM-as-a-judge의 투명성과 해석 가능성을 향상시키는 것입니다.



### Bayesian Decision Making around Experts (https://arxiv.org/abs/2510.08113)
- **What's New**: 이 논문은 전문가와 함께 작동하는 복잡한 학습 에이전트에 대한 연구를 다룹니다. 특히, Bayesian multi-armed bandits 맥락에서, 오프라인과 동시 설정의 경우에 대해 전문가 데이터가 학습자의 사후 확률(posterior)에 영향을 미치는 방식을 공식화합니다. 우리는 전문가의 최적 정책 및 그 결과로부터 얻은 데이터를 활용하여 정보를 효율적으로 통합하는 방법을 제안합니다.

- **Technical Details**: 논문에서는 학습자가 자신의 경험 또는 전문가의 결과를 기반으로 신념을 업데이트해야 할지를 결정하는 방법을 연구합니다. 오프라인 설정에서는, 전문가의 과거 결과를 포함하는 데이터셋을 통해 학습자가 사전 신념(prior)을 초기화(warm-start)하고, 동시 설정에서는 어떤 데이터 소스를 선택할지를 정보 이득 기준으로 결정합니다. 이를 통해 Thompson Sampling 알고리즘의 성능 향상을 위한 정보 이론적 규약이 제공됩니다.

- **Performance Highlights**: 이 연구는 전문가의 정보를 활용하여 Bayesian 학습 에이전트가 언제 신뢰하고 배워야 할지를 결정하는 메타 문제를 해결하는 방법을 제시합니다. 우리의 알고리즘은 전문가의 결과가 최적 행동 분포에 미치는 영향을 정량화하며, 실험 결과에서 강력한 비대칭 세계에서의 후회(regret) 개선을 보여줍니다. 궁극적으로 이 연구는 여러 Bayesian 학습자가 공존하는 상황에서의 이론적 이해를 심화시키고 향후 에이전트 설계를 위한 강력한 프레임워크를 제시하는 것을 목표로 합니다.



### VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents (https://arxiv.org/abs/2510.08109)
- **What's New**: 이번 논문에서는 기술 문서의 버전 관리 문제를 해결하기 위한 새로운 Retrieval-Augmented Generation (RAG) 시스템인 VersionRAG를 제안합니다. 기존 RAG 시스템들은 문서의 진화에 따른 정확한 답변을 제공하지 못하고 있으며, 특히 버전이 중요한 경우에서의 결과 정확도가 낮습니다. VersionRAG는 문서의 변화 과정을 계층 그래프 구조로 모델링하여 버전별 필터링 및 변화 추적을 가능하게 합니다.

- **Technical Details**: VersionRAG의 핵심 요소는 세 가지 쿼리 유형을 정의하고, 이 각각에 대해 특화된 탐색과 필터링 메커니즘을 사용하는 것입니다. 문서 버전은 계층 그래프 노드와 엣지로 표현되어 있으며, 각 노드는 문서의 카테고리, 개별 문서, 버전, 내용 및 변경 사항을 나타냅니다. 이 구조는 문서 쿼리에 대한 효율적인 그래프 탐색을 가능하게 하여 정확도 및 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: VersionRAG는 34개의 버전이 있는 기술 문서에서 100개의 수작업으로 구성된 질문에 대해 90%의 정확도를 달성했습니다. 이는 기존의 RAG(58%) 및 GraphRAG(64%)와 비교하여 현저히 높은 성능을 보입니다. 또한, 인덱싱 과정에서 GraphRAG에 비해 97% 적은 토큰을 필요로 하여 대규모 배포 시 효율성을 확보하고 있습니다.



### Development of Mental Models in Human-AI Collaboration: A Conceptual Framework (https://arxiv.org/abs/2510.08104)
Comments:
          Preprint version. Accepted for presentation at the International Conference on Information Systems (ICIS 2025). Please cite the published version when available

- **What's New**: 이 논문은 인공지능(AI)와의 협력이 비즈니스 의사결정에서 점점 중요해지는 가운데, 인간과 AI의 상호작용에서 의사결정자의 정신 모델이 발전한다는 점을 강조하고 있습니다. 기존 연구는 주로 AI 에이전트의 설계와 협력 구조에 초점을 맞추었으나, 의사결정자의 사고 프로세스는 진화하고 있음을 간과했습니다. 연구는 이러한 격차를 해소하기 위해, 인간-AI 협력 설계가 세 가지 상호 보완적인 정신 모델에 영향을 미친다는 개념을 제시합니다.

- **Technical Details**: 이 논문은 인간-AI 협력의 효과적인 설계를 위한 통합된 사회 기술적 프레임워크를 개발하여 데이터 맥락화(data contextualization), 추론 투명성(reasoning transparency), 성과 피드백(performance feedback)이라는 세 가지 메커니즘이 정신 모델의 발전을 어떻게 이끄는지를 규명하고 있습니다. 이 연구는 세 가지 독립적인 정신 모델(도메인(domain), 정보 처리(information processing), 보완성 인식(complementarity-awareness))을 도입하며, 정신 모델의 동적인 특성을 인정합니다.

- **Performance Highlights**: 이 연구는 인간-AI 협력 문헌에 기여하는 세 가지 주요 기여를 통해, 협력이 진행되는 방식과 의사결정자의 인식 변화를 이해하는 데 중요한 통찰을 제공합니다. 또한, 목적 지향적인 인간-AI 협력 설계를 위한 메커니즘이 어떻게 작용하는지를 규명하여 향후 연구와 실제 적용에 대한 방향성을 제시합니다.



### Lossless Vocabulary Reduction for Auto-Regressive Language Models (https://arxiv.org/abs/2510.08102)
- **What's New**: 이번 논문에서는 자동 회귀 (auto-regressive) 언어 모델의 어휘 (vocabulary)를 손실 없이 줄이는 이론적 프레임워크를 제시합니다. 이는 언어 모델들이 각기 다른 어휘를 가질 때 발생하는 비효율성을 해결하는 데 초점을 맞추고 있습니다. 새로운 접근 방식은 다양한 모델들이 상호 협력할 수 있도록 돕는 최대 공통 어휘 (maximal common vocabulary)를 활용합니다.

- **Technical Details**: 논문에서 제시된 프레임워크는 주어진 자동 회귀 언어 모델의 어휘를 임의의 크기로 축소할 수 있도록 합니다. 이는 각 모델의 다음 토큰 분포 (next-token distribution)에 미치는 영향을 최소화하며, 결과적으로 텍스트 생성의 효율성을 증가시킵니다. 이론적 기반 위에서, 손실 없는 (lossless) 어휘 축소 방법론이 제시되고, 이를 통해 다양한 토크나이제이션 (tokenization)을 가진 모델들이 협력할 수 있는 가능성을 탐구합니다.

- **Performance Highlights**: 이 연구는 서로 다른 언어 모델들이 하나의 공통된 어휘를 통해 효과적으로 협력할 수 있음을 실험적으로 입증했습니다. 향상된 협력 구조 덕분에 모델 앙상블 (ensemble) 성능이 크게 향상되었습니다. 따라서, 언어 모델의 협력적 활용 방안이 확장되어, 다양한 분야에서의 응용 가능성을 보여줍니다.



### The Price of Thought: A Multilingual Analysis of Reasoning, Performance, and Cost of Negotiation in Large Language Models (https://arxiv.org/abs/2510.08098)
- **What's New**: 이 논문은 AI 에이전트의 협상 능력에 대한 포괄적인 연구를 통해 LLM(대형 언어 모델)의 이유 작용(reasoning)이 협상 결과에 미치는 영향을 체계적으로 평가합니다. 연구는 상업적 모델과 공개 가중치(open-weight) 모델 모두를 대상으로 하며, 영어, 독일어, 이탈리아어의 세 가지 언어에서 수행됩니다. 협상 전략에 미치는 이유 작용의 효과를 확인하여 언어의 일관성을 유지하는지와 같은 여러 중요한 질문을 탐구합니다.

- **Technical Details**: 이 연구는 자율 에이전트가 협상 능력을 평가하는 것이 필수적임을 강조합니다. 두 개의 에이전트가 의사소통을 통해 상호 작용하는 각 대화 게임에 대해 LLM의 이유 작용이 협상 효율성에 미치는 영향을 분석합니다. 경기 상황은 게임 마스터를 통해 엄격히 조정되고, 개인별로 다른 가치가 있는 항목에 대해 협상하게 됩니다.

- **Performance Highlights**: 결과에 따르면 이유 작용을 사용하는 것이 협상 결과를 31.4% 향상시키지만, 그에 따른 계산 비용은 400% 가까이 증가하는 것으로 나타났습니다. 놀랍게도, 공개 가중치 모델은 독일어 또는 이탈리아어로 협상하는 동안 내부 이유 작용 단계를 영어로 전환하는 경향이 있으며, 이 점은 이해 가능성에 영향을 미칠 수 있음을 시사합니다. 반면 상업적 모델은 최종 출력과 이유 작용 간의 언어 일관성을 유지합니다.



### Everything is Plausible: Investigating the Impact of LLM Rationales on Human Notions of Plausibility (https://arxiv.org/abs/2510.08091)
Comments:
          pre-print

- **What's New**: 본 연구에서는 LLM(대형 언어 모델)에 의해 생성된 논거가 다중 선택 구성 요소의 정황 접근 방식에서 인간의 타당성 판단에 미치는 영향을 조사합니다. 3,000개의 인간 타당성 판단과 13,600개의 LLM 기반 판단을 수집하여, LLM이 생성한 근거가 이러한 판단에 상당한 영향을 미친다는 것을 발견했습니다. 이는 LLM이 인간의 인지를 연구하는 데 새롭고 효과적인 방법이 될 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 첫 번째로, 다중 선택 기준과 관련한 질문-답변 쌍의 타당성을 평가하기 위해 LLM로부터 생성된 PRO(긍정적) 및 CON(부정적) 논거를 사용합니다. 실험을 통해, 타당성 근거의 추가가 인간과 LLM의 판단에 대하여 유의미한 변화를 가져오는지 검사하였으며, 이 과정에서 1717개의 LLM이 추가적인 판단을 생성해냈습니다.

- **Performance Highlights**: 실험 결과, PRO 논거는 인간과 LLM의 타당성 평가 점수를 평균적으로 증가시키는 반면, CON 논거는 이를 감소시킨다는 것을 발견했습니다. 흥미롭게도, 특정 조건에서는 인간의 골드 답변에 대한 타당성 평가가 PRO 논거에 의해 감소하는 현상이 관찰되었습니다. 또한 타당성 평가의 초기 기준이 후속 평가의 변화에 강한 앵커링 효과를 미치는 것으로 나타났습니다.



### A Novel Ensemble Learning Approach for Enhanced IoT Attack Detection: Redefining Security Paradigms in Connected Systems (https://arxiv.org/abs/2510.08084)
Comments:
          14 pages, 5 fiugres, 7 tables

- **What's New**: 본 연구는 IoT(Internet of Things) 장치의 보안 취약점을 공격 탐지 시스템으로 개선하기 위한 새로운 앙상블 학습 아키텍처를 제안합니다. 이 방법은 Extra Trees Classifier와 같은 고급 머신러닝 기술을 적용합니다. 기존의 보안 솔루션들과 비교하여, 제안된 방법이 더 효율적이라는 점에 주목해야 합니다.

- **Technical Details**: 제안된 모델은 데이터를 철저히 전처리(preprocessing)하고 하이퍼파라미터 최적화(hyperparameter optimization)를 포함하여 IoT 공격 탐지의 성능을 극대화합니다. 이 연구는 CICIoT2023, IoTID20, BotNeTIoT L01, ToN IoT, N BaIoT, BoT IoT와 같은 여러 벤치마크 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 높은 재현율(recall), 정확도(accuracy), 정밀도(precision)를 달성하며 매우 낮은 오류율(error rates)을 기록하였습니다. 이러한 성과는 IoT 환경을 보호하기 위한 효과적이고 확장 가능한 방법으로서의 모델의 능력을 입증합니다.



### An Adaptive Multi Agent Bitcoin Trading System (https://arxiv.org/abs/2510.08068)
Comments:
          18 pages, 6 figures , 2 tables

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)을 활용한 다중 에이전트 비트코인 트레이딩 시스템을 제안합니다. 기존의 정적 회귀 모델이나 역사적 데이터에만 의존하는 신경망이 갖는 한계를 극복하고, 기술적 분석, 감정 평가, 의사 결정 및 성과 reflextion을 위한 전문화된 에이전트를 구조화하여 비트코인 포트폴리오 관리의 효율성을 높입니다. 이 시스템은 새로운 구술 피드백 메커니즘을 통해 시간에 따라 개선되며, 이러한 접근은 LLM을 금융 목표에 맞추어 조정할 수 있는 저비용의 혁신적인 방법으로 나타납니다.

- **Technical Details**: 제안된 프레임워크는 LLM을 전문화된 에이전트로 구성하여 비트코인 및 기타 암호화폐 시장에서의 알파 생성(Alpha Generation)과 포트폴리오 관리(Portfolio Management)를 지원합니다. 각 에이전트는 매일 또는 주간에 거래 결정을 평가하여 리플렉트 에이전트가 자연어로 비판하는 방식으로 작동하며, 이러한 평가 결과는 향후 프롬프트에 통합되어 에이전트가 지표 우선 순위, 감정 가중치, 그리고 자산 배분 로직을 조정할 수 있도록 돕습니다.

- **Performance Highlights**: 비트코인 가격 데이터에 대한 백테스트 결과는 이 시스템이 시장 상태에 관계없이 지속적으로 높은 성과를 보여주었음을 나타냅니다. 양적 에이전트는 강세 시장에서 30% 이상의 초과 수익을 달성했으며, 감정 기반 에이전트는 시장이 횡보할 때 소소한 손실을 100% 이상의 이익으로 전환했습니다. 매주 피드백을 추가하게 되면서 전체 성과가 31% 개선되었고, 하락 손실이 10% 줄어드는 긍정적인 결과를 보였습니다.



### Attribution-by-design: Ensuring Inference-Time Provenance in Generative Music Systems (https://arxiv.org/abs/2510.08062)
- **What's New**: AI로 생성된 음악의 등장으로 저작권(Pool) 분배의 문제와 기존 보상 체계의 결함이 드러나고 있습니다. 이 논문에서는 직접적인 저작권(Attribution), 투명한 로열티(Royalty) 분배와 아티스트 및 권리 보유자를 위한 세분화된 제어를 중점으로 한 생성 음악 구조를 제안합니다. 제안하는 방식은 아티스트의 카탈로그가 생성된 출력물에 사용하는 경우 즉각적으로 확인 가능한 보상을 가능하게 합니다.

- **Technical Details**: 이 연구는 데이터 세트에서 훈련(Training) 세트와 추론(Inference) 세트를 구분함으로써, 훈련 시간 저작권(Training-time Attribution)과 추론 시간 저작권(Inference-time Attribution)을 제안합니다. 본 연구는 주로 추론 시간 저작권을 선호하는데, 이는 생성된 출력물이 아티스트의 작업에 기초할 때 직접적이고 검증 가능한 보상을 제공하기 때문입니다. 또한, 사용자들이 특정 곡을 바탕으로 생성 작업을 조건화할 수 있게 하여, 저작권과 승인된 사용에 대한 투명한 정보를 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 생성 음악 시스템에 저작권을 직접 내장함으로써 스스로 보장하는 속성으로 저작권을 정의합니다. 이를 통해 생성 모델링의 기술적 틀 안에 가치 인식을 재통합할 수 있는 방향을 제시하며, 관념적 투명성을 기존의 예술 저작권과 법적 책임과 동일시합니다. 이러한 접근은 AI가 생성하는 음악의 정확한 보상 메커니즘을 구축하기 위한 윤리적이고 실용적인 대안을 제공합니다.



### FedDTRE: Federated Dialogue Generation Models Powered by Trustworthiness Evaluation (https://arxiv.org/abs/2510.08058)
- **What's New**: FedDTRE(Federated adaptive aggregation strategy for Dialogue generation based on Trustworthiness Evaluation)를 제안하여 개인 정보 보호와 모델 성능 간의 균형을 맞추고 있습니다. 이 새로운 전략은 지역 업데이트 동안 글로벌 모델의 기여도를 조정하여 대화 모델의 성능을 향상시킵니다. 기존의 연합 학습 방식에서 발생하는 오버피팅(overfitting) 문제를 해결하고, 글로벌 정보를 잊지 않도록 돕는 메커니즘이 포함되어 있습니다.

- **Technical Details**: FedDTRE는 대화 생성 모델을 위한 연합적 업데이트 전략으로, 신뢰도 평가(trustworthiness evaluation)를 기반으로 전역 정보를 선별적으로 통합합니다. 이를 통해 로컬 모델 성과를 개선하고 데이터 프라이버시를 보호하는 방식으로 작동합니다. 각 클라이언트는 BERT 모델을 사용하여 자신만의 데이터를 통해 신뢰도 점수를 생성하고, 서버는 이 정보를 기반으로 글로벌 모델을 업데이트합니다.

- **Performance Highlights**: Synthetic-Persona-Chat, CMU_DoG 및 WoW 데이터 세트에서의 실험 결과, FedDTRE가 대화 생성 품질을 개선하고 개인 맞춤화 개인화 모델링 간의 우수한 균형을 달성한다는 것을 보여주었습니다. 이는 소규모 데이터 클라이언트의 오버피팅 문제를 효과적으로 완화하고 글로벌 지식을 보존하여 전체 모델의 일반화 능력을 향상시키는 데 기여합니다.



### A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models (https://arxiv.org/abs/2510.08049)
- **What's New**: 이번 연구는 기존의 결과 보상 모델(Outcome Reward Models, ORMs)이 최종 답변만 평가하는 데 그치는 문제를 해결하기 위해, 과정 보상 모델(Process Reward Models, PRMs)을 소개합니다. PRMs는 단계별 또는 궤적 수준에서 추론을 평가하고 안내하는 방법입니다. 이 설문조사는 PRMs의 생성, 구축 및 테스트 시 확장과 강화 학습(Reinforcement Learning) 적용 방법을 체계적으로 다룹니다.

- **Technical Details**: PRMs의 작동 방식은 데이터(데이터 생성) 수집부터 시작하여, 모델 구성, 그리고 실제 테스트에 적용하는 과정까지 포함됩니다. 연구는 수학, 코드, 텍스트, 다중 모달 추론(multimodal reasoning), 로봇 공학(robotoics) 및 에이전트(agents) 분야에서의 적용을 요약하며, 새로운 벤치마크에 대해서도 논의합니다. 이런 과정에서 PRMs는 보다 섬세하고 견고한 추론 정렬(fine-grained, robust reasoning alignment)을 위해 필수적인 역할을 합니다.

- **Performance Highlights**: 이 연구는 PRMs가 기존 ORM의 한계를 극복하고, 다양한 응용 분야에서 더욱 향상된 추론 성능을 발휘할 수 있음을 강조합니다. 앞으로의 연구 방향과 디자인 공간을 명확히 하여, 더 세밀한 추론 정렬을 위한 도전 과제를 드러내고자 하는 목표를 가지고 있습니다. PRMs가 다양한 벤치마크에서의 성과를 통해 그 활용 가능성을 보여줍니다.



### TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevanc (https://arxiv.org/abs/2510.08048)
- **What's New**: 본 연구에서는 TaoSR-AGRL이라는 Adaptive Guided Reinforcement Learning 프레임워크를 제안합니다. 이 프레임워크는 Taobao 검색 관련성 예측을 강화하기 위해 설계되었고, 두 가지 주요 혁신을 포함합니다: Rule-aware Reward Shaping 및 Adaptive Guided Replay. 이러한 접근 방식은 복잡한 비즈니스 규칙과 사용자 쿼리의 변화하는 요구를 충족시키기 위해 긴-tail 문제를 해결하고자 합니다.

- **Technical Details**: TaoSR-AGRL의 핵심 모듈 중 하나인 Rule-aware Reward Shaping은 최종 관련성 판단을 밀집화된 구조화된 보상으로 분해하여 도메인별 기준에 맞게 조정합니다. Adaptive Guided Replay는 학습 중 낮은 정확도의 롤아웃을 식별하여 궁극적으로 정책이 정체된 추론 패턴에서 벗어날 수 있도록ground-truth 정보를 주입합니다. 이로 인해 모델은 높은 가치의 추론 경로를 탐색하고 개선할 수 있습니다.

- **Performance Highlights**: TaoSR-AGRL은 대규모 실제 데이터세트를 기반으로 평가되었으며, DPO 및 기존 GRPO 기반라인을 일관되게 초과하여 관련성 정확도, 규칙 준수 및 훈련 안정성을 개선했습니다. 이 모델은 Taobao의 주요 검색 시나리오에 성공적으로 배포되어 수억 명의 사용자에게 서비스를 제공하고 있으며, 연구 결과는 해당 산업 분야에서의 실질적인 가치를 보여줍니다.



### Verifying Graph Neural Networks with Readout is Intractab (https://arxiv.org/abs/2510.08045)
- **What's New**: 본 논문에서는 글로벌 읽기 기능(global readout)이 포함된 양자화 집합-결합 그래프 신경망(ACR-GNNs)에 대한 추론을 위한 논리적 언어를 소개합니다. 또한, 양자화된 GNN의 검증 작업(verfication task)이 (co)NEXPTIME 완전함을 증명하여 양자화된 GNN의 검증이 계산적으로 어려운 문제임을 강조합니다. 이 결과는 GNN 기반 시스템의 안전성을 보장하기 위한 연구 노력을 촉구합니다.

- **Technical Details**: ACR-GNN의 검증을 수행하기 위해 새로운 논리 q​ℒq\mathcal{L}을 정의하고, 이를 통해 글로벌 읽기가 포함된 양자화된 GNN의 복잡성을 다룹니다. 이 논리는 다양한 활성화 함수를 가진 양자화 ACR-GNN을 포착할 수 있을 만큼 표현력이 뛰어나기 때문에, 그래프 속성을 표현하는 유연한 언어로 활용될 수 있습니다. 또한, q​ℒq\mathcal{L}의 만족 가능성 문제는 NP-complete로 완화하여, 정점 수가 제한된 그래프 카운터 예제를 검색합니다.

- **Performance Highlights**: 실험을 통해 양자화된 GNN 모델이 경량화되면서도 비양자화 모델에 비해 좋은 정확도와 일반화 능력을 유지함을 보여주었습니다. 본 연구 결과는 자원이 제한된 환경에서도 양자화된 ACR-GNN의 실제적인 활용 가능성을 확인합니다. 전체적으로 양자화된 모델은 모델 크기 및 추론 비용의 상당한 절감을 이루면서도 강력한 예측 성능을 유지합니다.



### Towards Reliable LLM-based Robot Planning via Combined Uncertainty Estimation (https://arxiv.org/abs/2510.08044)
- **What's New**: 본 논문에서는 Combined Uncertainty estimation for Reliable Embodied planning (CURE)이라는 새로운 방법을 제안합니다. CURE는 불확실성을 epistemic(지식적) 불확실성과 intrinsic(내재적) 불확실성으로 분해하고 각각 따로 추정합니다. 이 방식은 로봇 계획에서의 신뢰성을 개선할 수 있도록 도와주며, 최근의 연구들에서 분류하지 않았던 두 가지 불확실성을 세분화하여 평가하는 점에서 혁신적입니다.

- **Technical Details**: CURE는 LLM의 기능을 기반으로 랜덤 네트워크 증류(Random Network Distillation, RND)와 다층 퍼셉트론 회귀 헤드를 사용하여 불확실성을 평가합니다. epistemic 불확실성은 작업의 명확성과 친숙함으로 세분화되며, intrinsic 불확실성은 주어진 계획의 예상 성공률로 모델링됩니다. 이를 통해 계획 모델의 신뢰도를 보다 높은 정확도로도 반영할 수 있습니다.

- **Performance Highlights**: 실험 결과, CURE는 기존의 방법들보다 현실적인 실행 결과와 더욱 밀접하게 일치하는 불확실성 추정치를 산출했습니다. 특히 주방 조작 및 테이블 재배치 작업을 포함한 두 가지 서로 다른 실험 환경에서 유의미한 개선을 보였습니다. CURE는 로봇 계획 및 LLM 불확실성 추정에 있어서 데이터를 평가하는 새로운 기준을 제안했습니다.



### MRI-derived quantification of hepatic vessel-to-volume ratios in chronic liver disease using a deep learning approach (https://arxiv.org/abs/2510.08039)
Comments:
          ^Alexander Herold and Daniel Sobotka share first-authorship

- **What's New**: 이번 연구는 심층 학습(deep learning) 기반의 자기공명영상(MRI) 분석을 통해 만성 간 질환(chronic liver disease) 단계 및 건강한 대조군의 간 혈관(volume) 용적을 정량화하려고 하였습니다. 연구는 건강한 대조군, 비진전(advanced) 아닌 만성 간 질환(non-advanced chronic liver disease, non-ACLD) 환자와 진전된 만성 간 질환(advanced chronic liver disease, ACLD) 환자를 대상으로 하였습니다. 이를 통해 간 기능 및 섬유화(fibrosis)와 문맥 고혈압(portal hypertension)과의 상관관계를 평가했습니다.

- **Technical Details**: 3D U-Net 모델을 활용하여 간 혈관(segmentation)을 수행하였고, 가독성 있는 결과를 도출하기 위해 3-T MRI에서 포르탈(prtal) 정맥 위상에서의 가다조틱 산(gadoxetic acid) 강화 이미지를 분석하였습니다. 연구에 포함된 대상자는 197명으로, 이들은 평균 54.9세이었으며 남성이 56.3%를 차지했습니다. 총 간 혈관 용적(TVVR), 간 혈관(HVVR), 및 간내 문맥 정맥 대 볼륨 비율(PVVR)은 그룹 간 비교를 통해 매우 의미 있는 차이를 보였습니다.

- **Performance Highlights**: 연구 결과, TVVR과 HVVR은 대조군에서 가장 높았으며, 비진전 만성 간 질환 환자에서는 중간 정도, ACLD 환자에서는 가장 낮았습니다. PVVR은 비진전 및 ACLD 환자 모두에서 대조군에 비해 감소했지만, CLD 그룹 간에는 차이가 없었습니다. HVVR은 FIB-4, ALBI, MELD-Na 등 여러 바이오마커(biomarker)와 유의미한 상관관계를 보였고, TVVR과 PVVR은 비슷한 경향을 보였지만 다소 약한 상관성을 보였습니다.



### FastUMI-100K: Advancing Data-driven Robotic Manipulation with a Large-scale UMI-style Datas (https://arxiv.org/abs/2510.08022)
- **What's New**: 이 논문에서는 FastUMI-100K라는 대규모의 UMI 스타일 다중 모드 데모 데이터셋을 소개합니다. 기존의 로봇 조작 데이터셋이 갖고 있는 한계점, 즉 수집의 확장성 부족, 경로의 매끄러움 및 다양한 로봇 구현체에 대한 적용 가능성 등을 극복하기 위해 설계되었습니다. FastUMI 시스템에 의해 수집된 이 데이터셋은 100K 이상의 데모 경로를 포함하며, 54개의 작업과 수백 가지 객체 유형을 다루고 있습니다.

- **Technical Details**: FastUMI-100K 데이터셋은 단일 팔 및 이중 팔 그리퍼 구성에 적합하게 설계되었습니다. 이 데이터셋은 120에서 500 프레임에 이르는 각 경로에서 구현되며, 여러 모드의 데이터로 구성되어 있습니다. 각 데모 상황은 멀티 모달 동작, 시각 인식 및 행동 정보를 포함하여 600시간의 상호작용 데이터를 제공합니다. 이를 통해 실제 세계의 작업 시나리오를 재현할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과에 따르면 FastUMI-100K 데이터셋은 다양한 기본 알고리즘에서 높은 정책 성공률을 가능하게 하며, 복잡하고 동적인 조작 문제를 해결하기 위한 강력한 적응성과 실제적 적용 가능성을 증명하였습니다. 이 데이터셋은 단일 작업 모방 학습 및 크로스 임바디먼트 전달을 지원하고 있으며, 전반적인 로봇 정책 학습 알고리즘에 적합한 다양한 도구 체인을 제공합니다.



### Backdoor Vectors: a Task Arithmetic View on Backdoor Attacks and Defenses (https://arxiv.org/abs/2510.08016)
Comments:
          22 pages, 13 figures, 15 tables

- **What's New**: 이 논문에서는 최근 모델 머징(Model Merging) 기법에 관한 연구를 수행하고, 이를 통해 발생할 수 있는 보안 위험, 특히 백도어 공격(Backdoor Attacks)의 취약점에 대한 새로운 통찰을 제시합니다. 우리는 백도어 벡터(Backdoor Vector)라는 개념을 도입하여, 공격의 특성을 이해하고 측정하는 효과적인 방법을 제공합니다. 또한, 논문에서는 새로운 방어 기법인 주입된 백도어 벡터 감소(Injection BV Subtraction, IBVS)를 제안하여, 기존의 백도어 공격을 방지하기 위한 접근 방식을 개선했습니다.

- **Technical Details**: 모델 연합 과정에서는 CLIP 모델을 사용하여, 시각적 인코더(Visual Encoder)와 텍스트 인코더(Text Encoder)의 조합을 통해 다양한 모델을 통합합니다. 각 모델의 가중치 차이를 기반으로 백도어 벡터(BV)를 산출하여, 이 벡터가 공격 또는 방어 전략의 비교 및 분석에 유용하다는 것을 보여줍니다. 새로운 희소 백도어 벡터(Sparse Backdoor Vector, SBV) 방법을 통해 여러 개의 백도어 벡터가 결합하여 보다 강력한 공격을 형성하는 방식을 제안하고, 이는 모델 머징 과정에서 더 높은 공격 성공률(Attack Success Rate, ASR)을 제공합니다.

- **Performance Highlights**: 이 연구의 결과는 희소 백도어 벡터(SBV)가 기존 공격 방법보다 뛰어난 효과를 보이며, 백도어 공격의 성공률을 크게 증가시킬 수 있다는 것을 나타냅니다. 또한, IBVS 방어 기법은 알려지지 않은 백도어 위협에 대해서도 효과적으로 작동하여, 경량화된 일반 방어 방법을 제공합니다. 이러한 발견은 모델 머징 과정에서의 내부 취약점과 백도어 공격의 면밀한 관계를 이해하는 데 기여하며, 향후 기억하기 쉬운 방어 기법 개발에 도움이 될 것입니다.



### Past, Present, and Future of Bug Tracking in the Generative AI Era (https://arxiv.org/abs/2510.08005)
Comments:
          Submitted to ACM TOSEM Special Issue: 2030 Software Engineering Roadmap

- **What's New**: 이 논문은 기존의 버그 추적 시스템이 전통적인 수동 방식에 의존하고 있다는 문제를 지적하며, AI 기반의 새로운 프레임워크를 통해 이러한 문제를 해결할 수 있는 방법을 제안합니다. 이 프레임워크는 사용자가 자연어로 버그를 보고할 수 있게 하고, AI 에이전트가 보고서를 세분화하고 재현하며 누락된 세부정보를 요청하는 과정을 자동화합니다.

- **Technical Details**: 이 논문은 버그 보고 과정을 자동화하여 효율성을 극대화하는 AI 기반의 프레임워크를 개발했습니다. LLM(large language model)을 활용하여 보고서의 유효성을 검증하고, 재현 및 분류 작업을 수행하며, 인간의 감시 절차를 유지하면서 패치를 생성합니다. 이러한 자동화는 시간 절약과 인간의 작업량 감소를 목표로 합니다.

- **Performance Highlights**: 자동화된 버그 추적 프레임워크는 더욱 빨라진 응답 시간과 향상된 협업, 그리고 소프트웨어 유지보수 방식을 강화하여 사용자 중심의 효율적인 프로세스를 제공합니다. 연구 결과, 부정확한 보고서의 수를 줄이고, 문제 해결 시간을 단축시키며, 전반적인 버그 품질을 향상시키는 데 기여할 것으로 기대됩니다.



### Learning on the Job: An Experience-Driven Self-Evolving Agent for Long-Horizon Tasks (https://arxiv.org/abs/2510.08002)
- **What's New**: 이번 논문에서는 MUSE라는 새로운 에이전트 프레임워크를 제안합니다. MUSE는 경험 기반의 자가 진화 시스템을 중심으로 하는 계층적 메모리 모듈을 도입하여 에이전트가 실시간으로 경험을 학습하고 개선할 수 있도록 지원합니다. 기존의 LLM 에이전트는 경험에서 학습할 수 없어 고정된 성능을 보였으나, MUSE는 이를 극복하여 지속적인 학습을 가능하게 합니다.

- **Technical Details**: MUSE는 다양한 레벨의 경험을 조직하고 이를 활용하여 장기 과제를 수행할 수 있는 능력을 갖추고 있습니다. 에이전트는 각 서브 태스크 실행 후 자신의 경로를 반성하고 원시 경로를 구조화된 경험으로 변환한 후 메모리 모듈에 다시 통합합니다. 이에 따라 에이전트는 정적 방식에서 벗어나 동적인 시스템으로 발전할 수 있습니다. 또한 메모리는 자연어로 저장되어 있어, 다른 LLM에서도 경험을 쉽게 전이할 수 있는 장점이 있습니다.

- **Performance Highlights**: MUSE는 TAC(한계 효율성 지표)라는 기준에서 새로운 SOTA(SOTA: State Of The Art) 성과를 기록했습니다. 가벼운 Gemini-2.5 Flash 모델을 사용하여 51.78%의 점수를 달성하며, 이전 SOTA보다 20% 증가했습니다. 실험 결과, 에이전트는 자율적으로 경험을 축적함에 따라 작업 수행 능력이 지속적으로 향상되며, 강력한 지속적 학습과 자가 진화 능력을 가지고 있음을 입증하였습니다.



### Leveraging Author-Specific Context for Scientific Figure Caption Generation: 3rd SciCap Challeng (https://arxiv.org/abs/2510.07993)
- **What's New**: 이번 논문에서는 3차 SciCap 챌린지를 위한 도메인 특화 캡션 생성 시스템을 제안합니다. LaMP-Cap 데이터셋을 이용하여 저자 특유의 문체와 도표 관련 텍스트 컨텍스트를 통합하였습니다. 이 시스템은 두 단계로 이루어지며, 첫 단계는 컨텍스트 필터링과 특정 카테고리 프롬프트 최적화, 그리고 캡션 후보 선정으로 구성됩니다. 두 번째 단계는 프로필 피규어를 이용하여 스타일을 개선하는 few-shot prompting 을 적용합니다.

- **Technical Details**: LaMP-Cap 데이터셋은 110,828개의 과학 기사를 포함하고 있으며, train/test/validation 데이터가 80:10:10 비율로 나누어져 있습니다. 캡션 생성 과정에서, 우리는 문단 정보를 사용하여 noiseless 캡션을 생성하는 두 단계 파이프라인을 개발하였습니다. 첫 단계에서는 context grounded 캡션을 생성하고, 두 번째 단계에서는 프로필 정보를 통해 스타일 조정을 합니다. MIPROv2와 SIMBA를 이용하여 카테고리 중심의 프롬프트 템플릿을 개발하였습니다.

- **Performance Highlights**: 우리의 실험 결과, 카테고리 특정 프롬프트가 제로샷 및 일반 최적화 접근 방식을 초월하는 성능을 보였습니다. ROUGE-1 recall이 +8.3% 향상되는 반면, precision 손실은 -2.8%로 제한되었고, BLEU-4 감소는 -10.9%에 그쳤습니다. 프로필에 기반한 스타일 수정을 통해 BLEU 점수가 40-48% 향상되었으며 ROUGE에서도 25-27%의 향상을 달성했습니다. 전반적으로, 본 시스템은 컨텍스트 이해와 저자별 스타일 적응을 결합하여 과학적으로 정확하고 스타일적으로 충실한 캡션을 생성할 수 있음을 보여줍니다.



### Fewer Weights, More Problems: A Practical Attack on LLM Pruning (https://arxiv.org/abs/2510.07985)
- **What's New**: 이번 연구는 모델 프룬링(model pruning) 방법이 어떻게 악의적으로 악용될 수 있는지를 처음으로 보여줍니다. 기존의 프룬링 방법들은 효율성을 높여왔지만, 보안 문제는 충분히 다루어지지 않았습니다. 연구 결과에 따르면, 적대자는 프룬링을 통해 모델의 악의적 행동을 유발할 수 있는 방법을 찾을 수 있습니다.

- **Technical Details**: 이 연구에서는 적대자가 각 매개변수가 pruning 될 가능성을 추정할 수 있는 프록시 메트릭(proxy metric)을 계산하여 공격을 설계하는 방법을 사용하였습니다. 이를 통해 적대자는 낮은 확률로 pruning 될 매개변수에 악의적 행동을 주입하고, 높은 확률로 pruning될 매개변수를 사용하여 모델을 복구함으로써 최종적으로 수정된 모델에서 악의적 행동을 나타내게 합니다.

- **Performance Highlights**: 이 연구에서는 vLLM에서 적용된 다양한 프룬링 방식(예: Magnitude, Wanda, SparseGPT)으로 인한 강력한 악의적 행동의 성공률을 평가하였습니다. jailbreak의 경우 최대 $95.7	ext{%}$, 정상적인 지시 거부의 경우 $98.7	ext{%}$, 목표한 내용 삽입의 경우 $99.5	ext{%}$의 성공률을 기록했습니다. 이러한 결과는 모델 압축에서 보안 인식의 필요성을 강조하고 있습니다.



### Is Architectural Complexity Always the Answer? A Case Study on SwinIR vs. an Efficient CNN (https://arxiv.org/abs/2510.07984)
Comments:
          7 pages, 4 figures

- **What's New**: 이 연구에서는 저조도 이미지에서 고주파 세부 사항을 복원하고 심각한 노이즈를 억제하는 과제가 여전히 중요한 도전임을 언급합니다. SwinIR와 같은 대규모 Transformer 모델이 성능의 최전선에서 자리잡고 있지만, 이 모델의 높은 계산 비용이 실제 응용에서 장애 요소가 될 수 있음을 강조합니다. 또한, 경량 합성곱 신경망(CNN)과의 비교를 통해 성능 및 효율성의 중요한 트레이드오프(trade-off)를 조사하고 있습니다.

- **Technical Details**: 실험 결과에 따르면, Transformer 기반의 SwinIR 모델은 최대 신호 대 잡음비(PSNR) 39.03 dB로 더 높은 성능을 달성했으나, 경량 CNN은 37.4 dB라는 경쟁력 있는 PSNR을 제공했습니다. 더욱이, CNN은 단 10 에폭(epoch)의 훈련으로 이 성능을 달성한 반면, SwinIR 모델은 132 에폭이 필요했습니다. CNN의 크기는 SwinIR보다 55배 이상 작아 계산 효율성이 더욱 강조됩니다.

- **Performance Highlights**: 이 연구는 표준 CNN이 실질적인 계산 오버헤드가 적은 상태에서 거의 최첨단의 결과를 제공할 수 있음을 보여줍니다. 이는 자원 제약이 우선 고려대상이 되는 실제 시나리오에서 CNN 사용의 타당성을 제시하는 강력한 근거가 됩니다. 따라서 높은 성능을 유지하면서도 경량화된 방식으로 저조도 이미지 복원 작업을 수행할 수 있는 방법에 대한 새로운 통찰을 제공합니다.



### ZeroCard: Cardinality Estimation with Zero Dependence on Target Databases -- No Data, No Query, No Retraining (https://arxiv.org/abs/2510.07983)
- **What's New**: 이번 논문에서는 ZeroCard라는 새로운 cardinality estimation 방법을 소개합니다. ZeroCard는 기존 데이터나 쿼리 로그에 의존하지 않고도 사용할 수 있는 최초의 방법으로, 데이터베이스 스키마의 의미(schemas semantics)를 활용하여 cardinality를 예측합니다. 이러한 접근 방식은 실제 환경에서의 사용성을 크게 향상시킵니다.

- **Technical Details**: ZeroCard는 주로 스키마의 의미를 통해 데이터 분포를 예측합니다. 이 방법은 특정한 의미를 가진 열들이 특정한 분포를 나타낸다는 가정에 기반하여, raw data 의존성을 줄이고 cardinality estimation의 정확성을 높입니다. 또한, 쿼리 템플릿에 구애받지 않는 표현 방법을 개발하여 쿼리 의존성을 완화했습니다.

- **Performance Highlights**: ZeroCard는 대규모 사전 훈련된 데이터셋인 GitTables를 기반으로 사전 훈련되어, 다양한 스키마 메타데이터를 활용하여 cardinality estimation을 수행합니다. extensive 실험을 통해 ZeroCard의 독특한 장점과 쿼리 최적화에서의 실제 응용 가능성을 입증하였으며, 실시간으로 새로운 데이터셋에서 문제 없이 작동할 수 있는 성능을 보여주었습니다.



### Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training (https://arxiv.org/abs/2510.07980)
Comments:
          This paper has been accepted by NeurIPS 2025 (Spotlight)

- **What's New**: 이 논문은 기존의 중앙 집중식 훈련과 비교하여 분산 훈련의 성과가 저조할 수 있는 상황에서 Multi-Gossip Steps (MGS)가 성능 격차를 줄이는 효과적인 방법임을 다룹니다. MGS의 최적화 오류를 빠르게 감소시키는 능력이 이론적으로 검증되었으며, 분산 훈련에 대한 새로운 통찰력을 제공합니다. 특히, 학습률, 데이터 이질성, 클라이언트 수, 통신 토폴로지 등이 MGS의 일반화에 미치는 영향에 대한 통합 분석을 제공합니다.

- **Technical Details**: MGS는 이론 분석을 통해 MGS가 최적화 오류 및 일반화 오류를 경량화하는 방식을 제시합니다. 논문에서는 중앙 집중식 미니 배치 SGD와 분산 네트워크의 일반화 오류를 비교하며, 각 환경에서의 오류 경계가 어떻게 되는지를 명확히 하고 있습니다. 또한 다양한 실험 조건 아래에서 MGS의 성능을 검증하는 실험 결과를 제시하고, 이러한 결과들이 이론적 분석과 일치함을 보였습니다.

- **Performance Highlights**: CIFAR 데이터셋에서의 실험 결과는 MGS의 효과를 분명히 보여줍니다. 통신 토폴로지, MGS 스텝 수, 클라이언트 수 등 주요 하이퍼파라미터의 변화가 오류 감소에 긍정적인 영향을 미쳤으며, 이는 이론적 결과와도 일치합니다. 이러한 연구 결과는 분산 최적화의 이론적 이해를 크게 발전시키며, 프랙티셔너에게 유익한 통찰을 제공합니다.



### Executable Analytic Concepts as the Missing Link Between VLM Insight and Precise Manipulation (https://arxiv.org/abs/2510.07975)
- **What's New**: 로봇이 비정형 환경에서 정밀하고 일반화된 조작을 수행하도록 하는 것은 여전히 큰 도전 과제입니다. 이 논문에서는 GRACE라는 새로운 프레임워크를 제안합니다. GRACE는 Vision-Language Models (VLM) 기반의 추론을 수행하고, 물리적 조작에 필요한 분석 개념(Executable Analytic Concepts, EAC)을 통해 세분화된 지침을 로봇이 이해하고 실행할 수 있도록 돕습니다. 이 접근법은 고수준의 명령 이해와 저수준 로봇 제어 사이의 통합된 인터페이스를 제공합니다.

- **Technical Details**: GRACE는 자연어 지침과 시각 정보를 바탕으로 EAC를 구체화하는 구조화된 정책 스캐폴딩 파이프라인을 통합합니다. EAC는 물체의 가능성, 기하학적 제약, 조작의 의미를 수학적으로 정의한 청사진을 포함합니다. 이는 로봇 작업을 위한 그립 포즈, 힘 방향 및 물리적으로 실행 가능한 경로를 계획하는데 활용됩니다. 이러한 접근은 VLM의 추론 역량과 물리적 실행 간의 간극을 메꾸어 줍니다.

- **Performance Highlights**: GRACE는 다양한 조작 작업에서 탁월한 성능을 발휘하며, 시뮬레이션 및 실제 환경에서 강력한 제로샷 일반화 능력을 보여주었습니다. EAC 기반의 접근법은 VLA 아키텍처와의 호환성도 강조되며, 이를 통해 VLM의 일반화 및 조작 정밀도를 크게 향상시킬 수 있음을 밝힙니다. 이러한 실험 결과는 GRACE가 로봇 제어에서의 효율성을 크게 개선할 잠재력을 가지고 있음을 시사합니다.



### Active Confusion Expression in Large Language Models: Leveraging World Models toward Better Social Reasoning (https://arxiv.org/abs/2510.07974)
Comments:
          15 pages, 10 figures

- **What's New**: 본 연구에서는 대형 언어 모델(LLM)이 사회적 추론 과제에서 겪는 한계를 다룹니다. LLM들은 데이터에서 발췌한 논리를 기반으로 수학적 및 코드 추론에서는 뛰어난 성과를 보이지만, 다수의 참여자와 시간이 얽힌 사회적 시나리오 처리 시 인지적 혼란을 경험하고 있습니다. 이러한 연구의 주된 목적은 적응형 세계 모델 강화 추론 메커니즘을 제안하여 이러한 결점을 극복하는 것입니다.

- **Technical Details**: 우리의 메커니즘은 텍스트 기반의 세계 모델을 구성하여 개체 상태 및 시간적 시퀀스를 추적하고, 추론 중 혼란 신호를 동적으로 모니터링합니다. 혼란이 발생했을 때, 자동으로 세계 상태를 명확하게 설명하여 LLM이 인지적 딜레마를 극복하도록 도와줍니다. 이 접근법은 인간이 사회적 상호작용 중에 사용하는 암묵적 세계 모델을 모방합니다.

- **Performance Highlights**: 세 가지 사회적 벤치마크에서 평가한 결과, 제안된 메커니즘은 정확도에서 유의미한 향상(예: Hi-ToM에서 +10%)을 기록했습니다. 또한 계산 비용을 최대 33.8% 줄였습니다. 이러한 결과는 사회적 맥락에서 LLM을 배치하는 데 있어 간단하면서도 효과적인 솔루션을 제공함을 보여줍니다.



### LightReasoner: Can Small Language Models Teach Large Language Models Reasoning? (https://arxiv.org/abs/2510.07962)
- **What's New**: 이번 논문에서는 LightReasoner라는 새로운 프레임워크를 제안하여, 약한 언어 모델(SLM, Small Language Model)이 강한 언어 모델(LLM, Large Language Model)의 학습을 돕는 방법을 탐구합니다. 이 접근법은 LLM과 SLM의 행동 차이를 이용해 고부가 가치 추론 순간을 드러내어 모델의 추론 능력을 개선할 수 있는 잠재력을 강조합니다. 이는 일반적인 감독적 세부 조정(Supervised Fine-Tuning, SFT)의 필요성을 줄이고, 여러 자원을 효율적으로 사용할 수 있게 해줍니다.

- **Technical Details**: LightReasoner는 두 단계로 운영됩니다: 1단계에서는 중요한 추론 순간을 샘플링하여 전문가 모델과 아마추어 모델 간의 행동 차이를 기반으로 감독 예제를 생성합니다. 2단계에서는 이러한 예제와 정렬하여 전문가 모델이 아마추어 모델과의 대조를 통해 자신의 장점을 강화하도록 훈련합니다. 특히, Kullback-Leibler (KL) 발산을 통해 전문가와 아마추어 모델의 다음 토큰 예측 차이를 계산하여 중요한 의사 결정을 식별합니다.

- **Performance Highlights**: LightReasoner는 7개의 수학적 성능 기준에서 최대 28.1%의 정확도 향상을 달성하며, 시간 소비는 90%, 샘플링된 문제는 80%, 조정된 토큰 사용은 99%까지 줄일 수 있습니다. 이러한 점에서 LightReasoner는 무라라이트 진실(label) 의존 없이도 효과적인 학습 신호를 생성하는 데 기여합니다. 이는 약한 모델을 강한 모델에 대한 효과적인 가르침 신호로 변환하여 LLM의 추론 향상에 기여하고 있습니다.



### A Systematic Evaluation of Self-Supervised Learning for Label-Efficient Sleep Staging with Wearable EEG (https://arxiv.org/abs/2510.07960)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 착용 가능한 EEG 장치를 활용한 수면 단계 분류에 대한 자가 지도 학습(SSL)의 체계적인 평가를 처음으로 수행했습니다. 기존의 수동 데이터 라벨링 의존도를 줄이고, 큰 양의 비라벨 데이터에서 유의미한 대표성을 추출하는 데 SSL의 가능성을 보여줍니다. 저자들은 또한 실질적인 환경에서의 사용 가능성을 위해 합리적인 비용의 수면 스코어링 파이프라인을 검증했습니다.

- **Technical Details**: 본 연구에서는 Ikon Sleep 착용 EEG 헤드밴드로 수집된 두 개의 수면 데이터베이스(BOAS 및 HOGAR)에서 다양한 SSL 방법을 평가했습니다. 그 결과, SSL은 라벨이 부족할 때 10%까지 분류 성능을 향상시키며, 5%에서 10%의 라벨 데이터만으로도 80% 이상의 임상 등급 정확도를 달성합니다. 이는 감독 학습 방식이 요구하는 두 배의 라벨보다 적은 수치입니다.

- **Performance Highlights**: SSL은 인구 특성, 녹음 환경 및 신호 품질의 변동성에도 강인한 성능을 보입니다. 연구 결과, SSL은 수면 단계 분류에서 라벨 효율성을 높이고, 수작업 주석에 대한 의존도를 줄임으로써 착용 가능한 EEG 시스템의 발전을 이끌 수 있는 잠재력을 보여줍니다. 이는 의료 비용 절감 및 접근성 향상으로 이어질 수 있습니다.



### DISCO: Diversifying Sample Condensation for Efficient Model Evaluation (https://arxiv.org/abs/2510.07959)
- **What's New**: 본 논문은 현대 기계 학습 모델의 평가가 막대한 비용을 요구하는 문제를 다루고 있습니다. 기존의 평가 방법은 모델 정확도와 최종 테스트 결과 간의 매핑을 학습하기 위해 정적 서브셋(anchor subset)을 선택하는 방식으로, 이 과정은 클러스터링에 의존하기 때문에 복잡하고 민감합니다. 새로운 방법인 Diversifying Sample Condensation (DISCO)는 모델의 응답 다양성을 극대화하는 샘플을 선택함으로써 평가 과정을 단순화합니다.

- **Technical Details**: DISCO는 그리디(greedy) 샘플 통계치를 사용하여 모델 간의 불일치를 최대화하는 top-k 샘플을 선택합니다. 이 접근 방법은 복잡한 클러스터링을 사용하지 않고도, 평가 성능을 극대화할 수 있는 이론적 근거를 제공합니다. DISCO의 성능은 다양한 도메인에서 평가되었으며, MMLU와 같은 데이터셋에서 99.3%의 평가 비용 절감을 기록했습니다.

- **Performance Highlights**: DISCO는 기존의 평가 방법들에 비해 뛰어난 효율성과 정밀도를 보여줍니다. 예를 들어, Anchor Points, TinyBenchmarks, Metabench와 같은 이전 방법들과 비교했을 때, DISCO는 성능 예측에서 최첨단 결과를 달성했습니다. 이를 통해 현대 기계 학습 모델의 평가 비용을 대폭 줄이고 환경적 영향을 최소화하는 데 기여할 수 있습니다.



### A$^2$Search: Ambiguity-Aware Question Answering with Reinforcement Learning (https://arxiv.org/abs/2510.07958)
- **What's New**: 최근에 발표된 연구는 대규모 언어 모델(LLM)과 강화 학습(RL)의 발전이 개방형 질문 응답(QA)에서 뛰어난 성능을 보여준다는 점에 주목하고 있습니다. 특히, 다수의 유효한 답변이 가능한 질문에서는 기존 모델들이 여전히 어려움을 겪고 있음을 강조합니다. 이를 해결하기 위해 A$^2$Search라는 새로운 주석 없는(end-to-end) 훈련 프레임워크를 제안합니다.

- **Technical Details**: A$^2$Search는 애매한 질문을 탐지하고 대안 답변을 수집하는 자동화된 파이프라인을 기본으로 구성됩니다. 이 시스템은 경로 샘플링(trajectory sampling)과 증거 검증(evidence verification)을 통해 답변을 수집하며, 다수의 답변을 자연스럽게 수용할 수 있는 $	ext{AnsF1}$ 보상을 이용해 RL로 최적화됩니다. 이러한 접근법은 수동 주석의 부담을 줄이고 다중 홉 데이터셋에 대한 확장을 용이하게 만듭니다.

- **Performance Highlights**: A$^2$Search는 8개의 개방형 QA 벤치마크에서 새로운 최첨단 성능을 달성하였습니다. A$^2$Search-7B 모델은 4개의 다중 홉 벤치마크에서 평균 $	ext{AnsF1}@1$ 점수 48.4%를 기록하여, ReSearch-32B(46.2%)와 같은 강력한 기준 모델을 초월하였습니다. 연구 결과는 ambiguity(애매함)를 수용하는 것이 더 신뢰할 수 있는 QA 시스템을 구축하는 데 필수적임을 보여줍니다.



### A Large-scale Dataset for Robust Complex Anime Scene Text Detection (https://arxiv.org/abs/2510.07951)
- **What's New**: 이 논문에서는 기존의 텍스트 탐지 데이터셋들이 자연 및 문서 중심 장면에 초점을 맞추고 있는 반면, 애니메이션 장면에서의 텍스트 탐지 필요를 해결하기 위해 새로운 데이터셋인 AnimeText를 소개합니다. AnimeText는 735K 이미지와 4.2M 개의 주석이 달린 텍스트 블록으로 구성되어 있으며, 계층적 주석과 애니메이션 관련 시나리오에 적합한 하드 네거티브 샘플을 제공합니다. 이 데이터셋은 애니메이션 장면에서의 텍스트 탐지 성능 향상을 위한 기반을 마련하고 있습니다.

- **Technical Details**: AnimeText 데이터셋은 총 4.2M 개의 다국어 텍스트 인스턴스에 대한 주석을 포함하고 있으며, 각 이미지에 평균 5.77 개의 텍스트 인스턴스가 포함되어 있습니다. 이는 기존 데이터셋보다 5배 더 큰 규모로, 더욱 다양한 텍스트 밀도를 커버합니다. 또한, 세 단계의 주석 파이프라인을 통해 효율적인 주석 작업을 가능하게 하여 고품질의 텍스트 구역 주석을 보장합니다.

- **Performance Highlights**: 실험 결과는 AnimeText로 훈련된 모델이 기존 데이터셋으로 훈련된 모델보다 애니메이션 장면 텍스트 탐지 작업에서 우수한 성능을 나타낸다는 것을 보여줍니다. 본 데이터셋은 애니메이션 텍스트 탐지 성과를 개선하는 데 필요한 훈련 데이터셋으로서 중요한 역할을 하며, 커뮤니티에 새로운 도전을 제공하는 테스트 데이터셋으로도 활용될 수 있습니다.



### TTOM: Test-Time Optimization and Memorization for Compositional Video Generation (https://arxiv.org/abs/2510.07940)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 비디오 생성 모델의 성능을 향상시키기 위해 Test-Time Optimization and Memorization (TTOM) 프레임워크를 소개합니다. 기존의 방식과는 달리, TTOM은 훈련 없이 spatiotemporal 레이아웃에 맞춘 출력을 제공합니다. 이는 기계 학습 모델이 영상에서 입력된 텍스트를 더 정련하게 이해하고 생성할 수 있도록 돕습니다.

- **Technical Details**: TTOM은 사용자 프롬프트에 기반한 spatiotemporal layout을 생성하고, 이를 통해 비디오 생성 모델의 성능을 최적화합니다. 새로운 매개변수를 도입하여 각 샘플에 맞춰 업데이트하며, 이 과정을 통해 이전 작업의 최적화를 메모리에 저장할 수 있습니다. 이 파라미터는 삽입, 읽기, 업데이트 및 삭제와 같은 다양한 작업을 지원하여 유연하고 효율적인 운영이 가능합니다.

- **Performance Highlights**: T2V-CompBench 및 Vbench 벤치마크에서의 실험 결과는 TTOM이 매우 효과적이고 실용적이며 효율적인 프레임워크임을 입증하였습니다. 특히, TTOM은 CogVideoX-5B와 Wan2.1-14B와 비교했을 때 T2V-CompBench에서 각각 34% 및 14%의 성능 향상을 이뤘습니다. 이는 복합 비디오 생성에서의 크로스 모달 정렬을 현장에서 자동으로 달성할 수 있도록 해줍니다.



### STEPER: Step-wise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented Language Models (https://arxiv.org/abs/2510.07923)
Comments:
          EMNLP 2025 Main

- **What's New**: 이번 연구에서는 Stepwise Knowledge Distillation(단계별 지식 증류)이라는 새로운 접근법을 제시합니다. 이는 복잡한 실제 질문에 대한 응답을 위한 단계별 정보 검색 및 통합 과정에서의 합리적인 응답 생성을 목표로 합니다. 기존의 지식 증류 방법들이 각 단계에서의 다양한 추론 능력을 간과하는 문제를 해결하고자 합니다.

- **Technical Details**: StepER은 각 단계별 요구되는 정보와 추론을 맞추기 위해 단계별 감독(step-wise supervision)을 적용합니다. 또한, 문제의 난이도에 따라 학습 최적화를 진행하는 difficulty-aware training(난이도 인식 훈련) 기법을 도입하여 적절한 단계를 우선시 합니다. 이 방법은 다중 단계 정보 검색이 가능한 언어 모델에 적용 가능하며, 검색 쿼리를 사용하는 추론 경로 또는 분해된 질문을 포함하고 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해, StepER은 다단계 QA 벤치마크에서 이전 방법들을 능가하는 성능을 보였습니다. 특히, 8B 모델이 70B 교사 모델과 유사한 성능을 달성함으로써 StepER의 효용성을 입증하고 있습니다.



### Towards Human-Like Grading: A Unified LLM-Enhanced Framework for Subjective Question Evaluation (https://arxiv.org/abs/2510.07912)
- **What's New**: 이 논문은 다양한 주관적 질문에 대한 자율 채점의 도전 과제를 해결하기 위해 통합된 LLM(대형 언어 모델) 강화 자동 채점 프레임워크를 제안합니다. 기존의 연구들은 특정 주관적 질문 유형에만 집중했으며, 다양한 질문 형식을 지원할 수 있는 포괄적인 접근이 부족했습니다. 새로운 프레임워크는 학생의 답변을 인간과 유사하게 평가하는 데 필요한 네 가지 보완 모듈을 통합합니다.

- **Technical Details**: 제안된 프레임워크는 텍스트 유사도 모듈(TSM), 키 포인트 매칭 모듈(KPM), 일반 평가 모듈(LGE), 가상 질문 매칭 모듈(PQM) 및 심층 융합 레이어로 구성됩니다. KPM은 학생과 참조 답변에서 중요한 지식 포인트를 추출하고, LGE는 여러 차원에서 답변을 평가하여 인간 채점자의 포괄적 판단을 모방합니다. PQM은 학생의 답변에서 생성한 가상 질문을 사용하여 원래 질문과의 의미적 정합성을 평가합니다.

- **Performance Highlights**: 제안된 프레임워크는 일반 목적 및 도메인 특화 데이터셋을 통해 기존의 전통적인 방법 및 LLM 기반 기준보다 일관되게 더 나은 성능을 보였습니다. 두 개의 새로운 데이터셋이 구축되었으며, 테스트 결과는 새로운 접근 방식이 다양한 질문 유형과 평가 메트릭에서 탁월함을 입증하였습니다. 실제 전자상거래 기업의 교육 및 인증 시험에서도 성공적으로 사용되었습니다.



### MMM: Quantum-Chemical Molecular Representation Learning for Combinatorial Drug Recommendation (https://arxiv.org/abs/2510.07910)
Comments:
          Medical Image Computing and Computer-Assisted Intervention (MICCAI) Predictive Intelligence in Medicine Workshop (MICCAI PRIME) 2025; 13 pages

- **What's New**: 이번 연구에서는 약물 간 상호작용 예측을 위해 새롭게 제안된 프레임워크인 Multimodal DDI Prediction with Molecular Electron Localization Function (ELF) Maps (MMM)을 소개합니다. MMM은 약물 표현 학습에 3차원 양자 화학 정보를 통합하여 높은 예측 정확도와 안전한 약물 처방을 지원하는 가능성을 보여줍니다. 이 프레임워크는 약물의 전자 밀도 맵을 생성하여 치료적 관련성과 상호작용 위험을 동시 고려할 수 있도록 설계되었습니다.

- **Technical Details**: MMM은 세 가지 주요 구성 요소로 이루어져 있습니다: 환자의 전자 건강 기록(EHR)을 인코딩하는 모듈, ELF 기반 약물 인코더로 전자 상호작용 특성을 반영하며, 환자 조건에 기반하여 약물 하위 구조의 중요성을 추론하는 지역 이분법 인코더, 그리고 안전하고 효과적인 약물 처방을 도출하기 위한 약물 추천 모듈입니다. 이 프레임워크는 DFT(computational density functional theory)를 사용하여 ELF 맵을 구성하고, 고차원 특징을 추출하기 위해 사전 학습된 CNN(convolutional neural network)을 활용합니다.

- **Performance Highlights**: MMM은 MIMIC-III 데이터셋에서 여러 기본 모델과 비교하여 F1 점수, Jaccard 유사성 및 DDI 비율에서 통계적으로 유의미한 향상을 보였습니다. 특히, GNN 기반의 SafeDrug 모델과의 비교에서 F1-score(p = 0.0387)와 Jaccard(p = 0.0112), DDI 비율(p = 0.0386)에서 개선된 결과를 보여, MMM이 약물 추천의 정밀도를 높이고, 임상 실무에서의 안전한 조합 약물 처방을 지원할 수 있는 잠재력을 입증하였습니다.



### Contrastive Weak-to-strong Generalization (https://arxiv.org/abs/2510.07884)
- **What's New**: 이 연구에서는 약한 모델에서 강한 모델로의 일반화를 위한 새로운 프레임워크인 Contrastive Weak-to-Strong Generalization (ConG)을 제안합니다. 이 프레임워크는 인간 피드백이나 명시적인 보상 모델 없이도 더 높은 품질의 샘플을 생성할 수 있는 가능성을 탐구합니다. 또한, ConG는 Contrastive Decoding (CD)과의 구조적 동등성을 기반으로 하여 약한 모델의 출력에서 노이즈를 줄여줍니다.

- **Technical Details**: ConG는 사전 및 사후 정렬된 약한 모델 간의 대비 디코딩을 활용하여 강한 모델을 훈련시키는 방법을 제시합니다. 이는 임시 보상을 사용하여 추출된 샘플의 품질을 평가하고, значности 높은 등장 확률 로그 비율을 최대화함으로써 약한 모델로부터 높은 신뢰도의 샘플을 생성합니다. 연구는 Qwen2.5 및 Llama3와 같은 주류 LLM 계열을 대상으로 하여 상응하는 설정에서 검증됩니다.

- **Performance Highlights**: 실험 결과, ConG는 전통적인 약한-강한 방법에 비해 모든 모델에서 일관되게 유의미한 성능 개선을 보여주었으며, 평균 16.5%의 향상을 기록했습니다. 이러한 성과는 ConG의 일반성과 효과성을 입증하며, 기능 전이, 노이즈 감소 및 강건성 향상을 통해 인공지능 일반화(AGI)로 나아가는 유망한 경로를 제시합니다.



### Team Xiaomi EV-AD VLA: Learning to Navigate Socially Through Proactive Risk Perception -- Technical Report for IROS 2025 RoboSense Challenge Social Navigation Track (https://arxiv.org/abs/2510.07871)
- **What's New**: 이 보고서는 IROS 2025 RoboSense Challenge의 Social Navigation Track에 대한 기술 세부 정보를 설명합니다. 이 트랙은 RGB-D 기반 인식 및 내비게이션 시스템을 개발하는 데 중점을 두며, 자율 에이전트가 동적 인간이 있는 실내 환경에서 안전하고 효율적으로 탐색할 수 있도록 돕습니다. 새로운 Proactive Risk Perception Module을 도입하여 사회적 내비게이션 성능을 향상시켰습니다.

- **Technical Details**: 사회적 내비게이션(Social Navigation)은 자율 로봇이 인간과 공유하는 환경에서 사회적 관습을 준수하며 탐색하는 능력을 나타냅니다. 이 연구는 Falcon 모델을 기반으로 하여, 상대 거리 정보를 활용해 인간과의 충돌 위험을 예측하는 모듈을 통합했습니다. 이는 충돌 회피 행동을 개선하고 공간 인식을 강화하는 데 기여합니다.

- **Performance Highlights**: Social-HM3D 벤치마크에서 우리의 방법이 밀집한 실내 장면에서 목표를 향해 탐색할 때 개인 공간 준수를 유지하는 능력을 개선했음을 보였습니다. 이 챌린지에서 16개 팀 중 2위를 달성하였으며, 이는 자율 탐색 시스템의 발전에 중요한 이정표가 될 것입니다.



### DM1: MeanFlow with Dispersive Regularization for 1-Step Robotic Manipulation (https://arxiv.org/abs/2510.07865)
Comments:
          Website with code: this https URL

- **What's New**: 이번 연구에서는 DM1(MeanFlow with Dispersive Regularization for One-Step Robotic Manipulation)이라는 새로운 흐름 기반 모델을 제안하여 모형의 표현 붕괴를 방지하고 일단계(sampling efficiency) 효율성을 유지합니다. 기존의 흐름 기반 정책들이 고급 조작 작업에서의 정밀성이 떨어지는 것에 대한 문제를 해결하며, 다양한 중간 임베딩 레이어에서 dispersive regularization을 적용하여 훈련 배치 간 다양성을 촉진합니다.

- **Technical Details**: DM1 프레임워크는 Gaussian noise를 평균 속도 필드를 통해 목표 동작 궤도로 직접 변환하여 반복적인 노이즈 제거 없이 고품질 동작 합성을 가능하게 합니다. 이 방식은 파라미터의 최적화 없이 여러 중간 embedding layer에 dispersive 손실을 적용하여 특성 분리를 보장하고 표현 붕괴를 방지합니다. 이 과정에서 다양한 규제 방식(예: InfoNCE-L2, Covariance-based 등)을 체계적으로 평가하여 효과를 입증합니다.

- **Performance Highlights**: RoboMimic 벤치마크에서 DM1은 20-40배 더 빠른 추론 속도(0.07초 대 2-3.5초)를 기록하며, Lift 작업에서 99%의 성공률을 달성했습니다. 물리적 로봇(Franka Panda)에서의 실제 배포 또한 DM1의 구현가능성을 입증하였으며, 50Hz를 초과하는 주파수에서 실시간 제어가 가능합니다.



### Self-Supervised Learning Strategies for a Platform to Test the Toxicity of New Chemicals and Materials (https://arxiv.org/abs/2510.07853)
- **What's New**: 이 논문은 자동화된 독성 검사 시스템에서 기계 학습 모델을 통해 독성 물질로 인한 변화를 효과적으로 식별할 수 있는 방법을 시연합니다. 자가 지도 학습(self-supervised learning)을 통해 학습된 표현이 독성 물질에 의한 변화를 확인할 수 있음을 보여주는 개념 증명을 제시합니다. 이를 위해 EmbryoNet 데이터셋을 활용하고, 다양한 화학 화합물에 의한 제브라피시 배아의 표현형을 연구하였습니다. 최종적으로 TOXBOX 프로젝트의 일환으로 물리적 독성 검사 장치에 기계 학습 모델을 통합하는 방안을 논의합니다.

- **Technical Details**: REACH 규제는 EU 시장에 진입하는 화학 화합물을 더 잘 이해하기 위한 목적으로, 특정 화합물의 수량이 1톤을 초과하는 경우 독성 테스트를 수행하고 결과를 European Chemicals Agency (ECHA)에 보고하도록 요구합니다. 기존의 독성 테스트는 일반적으로 비용이 많이 드는 인 비보(in vivo) 방법을 사용하는데, 이 논문에서는 인 비보 연구의 대안으로 제브라피시 배아를 활용한 고처리량 스크리닝(High-Throughput Screening, HTS)에 대한 관심이 증가하고 있음을 강조합니다. 자가 지도 학습을 통해 학습된 연속적 표현은 독성 물질에 의한 변화를 모델링할 수 있으며, 자동 평가 방법을 ML 모델을 통해 시행할 수 있습니다.

- **Performance Highlights**: 이 연구는 자가 지도 학습을 통해 학습한 표현이 화합물의 작용 방식(modes-of-action)을 효과적으로 구별할 수 있음을 입증하였습니다. 특히, 실험 데이터의 고차원성을 처리하는 데 있어 딥 러닝(Deep Learning) 모델이 전통적인 머신 러닝 모델보다 더 적합하다는 점을 강조하고 있습니다. 또한, 제브라피시의 다양한 배아 표현형을 비교하는 데 있어서 이 방법이 유용하다는 것을 보여주었으며, TOXBOX와 같은 실제 독성 검사 장치에 DL 모델을 통합하는 데 따른 도전 과제를 다루고 있습니다.



### Meta-Learning Based Few-Shot Graph-Level Anomaly Detection (https://arxiv.org/abs/2510.07847)
Comments:
          Accepted by ARRML2025

- **What's New**: 이 논문에서는 그래프 수준 이상 탐지(graph-level anomaly detection)의 새로운 프레임워크인 MA-GAD를 제안합니다. 이 프레임워크는 그래프 크기를 줄이는 그래프 압축 모듈(graph compression module)을 통합하여 노이즈 간섭을 완화하고 필수 노드 정보를 유지합니다. 또한, 메타 학습(meta-learning)을 활용하여 유사 네트워크로부터 메타 이상 정보를 추출하고, 적은 샘플로도 새로운 작업에 빠르게 적응할 수 있는 초기화 모델을 학습합니다.

- **Technical Details**: MA-GAD 프레임워크는 세 가지 단계로 구성됩니다: (1) 그래프 압축 전략을 통해 GNN 훈련 중 성능 손실을 최소화합니다. (2) 메타 학습 알고리즘이 보조 네트워크로부터 메타 이상 지식을 추출하여 일반화를 향상시킵니다. (3) 그래프 이상 손실 함수는 정상 노드와 이상 노드 간의 통계적 편차를 증가시켜 클래스 불균형(class imbalance)을 해결합니다.

- **Performance Highlights**: 실험 결과, MA-GAD는 4개의 실제 생화학 데이터 셋을 기반으로 기존의 최신 기법들을 초월하여 적은 샘플로 그래프 수준 이상 탐지에서 우수한 성능을 보였습니다. 본 연구는 지역적 및 글로벌 이상을 모두 다루며, 노드 및 그래프 수준에서 이상의 탐지를 강화하여 효과적인 결과를 도출했습니다.



### AdaSwitch: Adaptive Switching Generation for Knowledge Distillation (https://arxiv.org/abs/2510.07842)
- **What's New**: 이번 논문에서는 AdaSwitch라는 새로운 접근 방식을 제안합니다. AdaSwitch는 작은 언어 모델(SLM)과 큰 모델 간의 동적 토큰 수준 합성을 통해 지식 전이를 수행합니다. 이 방법은 학생 모델이 스스로의 예측을 탐색한 후, 실시간 품질 평가에 근거하여 교사의 지도를 선택적으로 통합할 수 있도록 합니다.

- **Technical Details**: AdaSwitch는 두 단계의 시퀀스 생성 프레임워크로, 탐색 단계에서는 학생이 독립적으로 시퀀스를 생성하고, 지도 단계에서는 교사가 나머지 시퀀스를 생성하여 고품질 출력을 보장합니다. 이 과정에서 생성 난이도에 따라 전환 임계값이 동적으로 조정되어, 과도한 개입을 방지하고 일관성과 품질 간의 균형을 유지합니다.

- **Performance Highlights**: 세 개의 데이터세트에서 두 개의 교사-학생 모델 쌍을 사용하여 AdaSwitch를 평가했습니다. 실험 결과, AdaSwitch는 대부분의 시나리오에서 성능을 일관되게 향상시키며, 평균적으로 순수 온 정책 방법보다 1.3배의 안정적인 계산 오버 헤드를 유지하면서 SKD보다 10% 감소한 효율성을 달성했습니다.



### Self-Improving LLM Agents at Test-Tim (https://arxiv.org/abs/2510.07841)
- **What's New**: 이번 논문은 새로운 테스트 시간 자기 향상 방법(Test-Time Self-Improvement, TT-SI)을 제안합니다. 이 방법은 모델이 어려움을 겪는 샘플을 식별하고, 이를 기반으로 새로운 훈련 샘플을 생성한 후, 테스트 시간에 이 샘플들로 모델을 개선하는 과정으로 구성됩니다. 기존의 대량 데이터셋에 의존하지 않고도 성능을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 세 가지 단계로 이루어집니다: 첫째, 자기 인식을 통해 모델이 어려움을 겪는 샘플을 식별하고, 둘째, 불확실 샘플에서 유사한 예제를 생성하며(셀프 데이터 증강), 셋째, 테스트 시간 맞춤형 훈련을 통해 이 새롭게 생성된 샘플들로부터 학습하는 방식입니다. 이 과정에서 'Uncertainty Estimator'와 'Data Synthesis Function'이 사용되어 모델의 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, TT-SI는 평균적으로 다른 표준 학습 방법에 비해 +5.48%의 절대 정확도 향상을 기록하며, 68배 적은 훈련 샘플로도 효과적인 학습이 가능함을 보여줍니다. TT-SI는 특정 과제를 위한 적응 능력을 강화하여, 기존의 감독된 세미 슈퍼바이즈드 학습(supervised fine-tuning) 방법보다 우수한 성능을 보입니다. 이 연구는 의료 기기 및 기타 다양한 복잡한 작업에서도 지속적인 자기 개선의 가능성을 보여줍니다.



### MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation (https://arxiv.org/abs/2510.07835)
Comments:
          Accepted By NeurIPS 2025

- **What's New**: 본 논문에서는 MetaDefense라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)에서 파인튜닝 기반의 jailbreak 공격을 방어하는 데 중점을 두고 있습니다. 기존 방어 메커니즘이 보지 못한 공격 템플릿에 의해 위장된 해로운 쿼리에 일반화되지 못하는 문제를 발견하였습니다.

- **Technical Details**: MetaDefense는 두 단계의 방어 접근 방식을 제안합니다: (i) 사전 생성 방어(pre-generation defense)로 해로운 쿼리를 반응 생성이 시작되기 전에 탐지하고, (ii) 중간 생성 방어(mid-generation defense)로 생성 중 부분적인 응답을 감시하여 보다 해로운 내용을 출력하는 것을 방지합니다. 이 시스템은 특별한 프롬프트를 사용하여 쿼리와 부분 응답의 해로움을 예측하도록 LLM을 훈련시킵니다.

- **Performance Highlights**: 다양한 LLM 아키텍처(예: LLaMA-2-7B, Qwen-2.5-3B-Instruct, LLaMA-3.2-3B-Instruct)에 대한 광범위한 실험 결과, MetaDefense는 기존 방어 메커니즘에 비해 상당히 우수한 성능을 보여주었습니다. 또한, 본 시스템은 보이는 공격 템플릿과 보이지 않는 공격 템플릿 모두에 대해 효과적인 방어를 제공하며, 일반 작업에 대해 경쟁력 있는 성능을 유지합니다.



### The Rise of the Knowledge Sculptor: A New Archetype for Knowledge Work in the Age of Generative AI (https://arxiv.org/abs/2510.07829)
Comments:
          23 pages, 11 figures, preprint

- **What's New**: 이 논문은 Generative AI (GenAI) 시스템이 자율적으로 콘텐츠를 생성할 수 있게 됨에 따라 지식 작업의 본질이 변화하고 있다는 점을 강조합니다. 또한 새로운 전문적 전형으로서 Knowledge Sculptor (KS)라는 개념을 도입하여 AI 출력물을 신뢰할 수 있는 실행 가능한 지식으로 전환하는 방법을 제시합니다. 이는 전통적인 정보 조직 및 검색 모델의 한계를 극복하려는 노력의 일환입니다.

- **Technical Details**: KS는 사회 기술적 관점에서 구성된 프레임워크를 기반으로 하며, 비전 설계(architecting a vision), 반복적 대화(iterative dialogue), 정보 조형(information sculpting), 호기심 기반 합성(curiosity-driven synthesis) 등의 역량을 포함합니다. KS의 역할을 설명하기 위해 실제적인 사례(vignette)를 통해 이론적 개념을 구체화하였습니다. 이 연구는 KS가 작업하는 방식을 실질적으로 설명하는 자가참조적Approach입니다.

- **Performance Highlights**: 이 논문은 KS의 실천 사례를 통해 GenAI 시스템과의 협력 과정에서 이러한 전문성과 기술이 실제로 어떻게 적용될 수 있는지를 보여줍니다. 이를 통해 KS는 AI의 잠재력을 극대화하고, 신뢰할 수 있는 지식을 창출하는 데 기여할 수 있음을 입증합니다. 따라서, Generative Age의 지식 관리 및 활용 방식에 중요한 변화를 예고하고 있습니다.



### SIMU: Selective Influence Machine Unlearning (https://arxiv.org/abs/2510.07822)
Comments:
          Accepted to NeurIPS 2025 Workshop: Constrained Optimization for Machine Learning (COML)

- **What's New**: 최근 Large Language Models(LLMs)의 민감한 정보의 원치 않는 암기가 문제가 되면서, 모델의 동작을 조절할 수 있는 안전 메커니즘이 필요하다는 요구가 커졌습니다. 이를 해결하기 위해 데이터 삭제를 가능하게 하는 기계 학습 기법이 개발되었습니다. 본 논문에서는 Selective Influence Machine Unlearning(SIMU)이라는 새로운 프레임워크를 제안하며, 이는 모델이 잊어야 할 정보를 인코딩하는 주요 뉴런만 선택적으로 업데이트함으로써 기억 상실을 효과적으로 수행하면서도 원래의 지식을 유지할 수 있도록 돕습니다.

- **Technical Details**: SIMU 프레임워크는 먼저 MLP(다층 퍼셉트론) 층에서 잊어야 할 정보를 인코딩하는 주요 뉴런을 식별하는 단계로 구성됩니다. 이후, 중요한 뉴런만 선택적으로 업데이트하는 방법을 통해 모델의 파라미터를 조정합니다. 이를 통해 기존의 방법보다 더 효과적으로 잊어야 할 정보의 영향을 제거하면서도 모델이 원래 갖고 있던 지식을 보존하게 됩니다.

- **Performance Highlights**: SIMU는 현재 사용되고 있는 기계 학습 삭제 기법들과 비교하여, 정보 삭제의 효율성을 유지하면서도 모델의 원래 기능을 크게 향상시키는 결과를 제공합니다. 실험 결과, SIMU는 제한된 뉴런 업데이트를 통해 원치 않는 정보의 삭제가 잘 이루어짐을 보여주어 기존 방법들에 비해 뛰어난 성능을 발휘했습니다. 이로써 AI 모델의 데이터 보호 및 안전성을 더욱 높이는 방향으로 기여할 것으로 기대됩니다.



### Effective and Stealthy One-Shot Jailbreaks on Deployed Mobile Vision-Language Agents (https://arxiv.org/abs/2510.07809)
- **What's New**: 이번 논문에서는 자율 모바일 에이전트가 스마트폰 사용자 인터페이스(UI)를 조작할 때의 보안 취약점을 밝혀내는 독창적인 공격 프레임워크를 제안합니다. 특히, 기존 연구의 한계를 넘어 저권한(이러한 권한이 제약되기 때문에) 환경에서도 효과적으로 작동할 수 있는 비침투적(jailbreak) 방법을 개발했습니다. 이는 에이전트가 운영되는 동안에만 노출되는 악성 프롬프트(in-app prompt injections)를 활용하여 지속적인 보안 위험을 드러냅니다.

- **Technical Details**: 이 공격 프레임워크는 세 가지 주요 구성 요소로 구성됩니다. 첫 번째는 저권한 인식 체인 타겟팅(Low-Privilege Perception-Chain Targeting)으로, 악성 앱의 사용자 인터페이스(UI) 내에서 프롬프트를 완전히 삽입하여 보안 권한을 우회합니다. 두 번째는 사용자에게 보이지 않는 은밀한 활성화(Stealthy User-Invisible Activation)로, 자동화된 입력과 인간의 터치를 구별하여 에이전트가 작동하는 동안에만 내용을 노출합니다. 세 번째는 단일 시도에 대한 프롬프트 효능(One-Shot Prompt Efficacy)으로, 키 토큰에 대한 최소한의 변화로 안전 점수를 개선하고 식별 위험을 감소시키는 알고리즘 (HG-IDA*)을 이용합니다.

- **Performance Highlights**: 다양한 LVLM 백엔드 모델을 평가한 결과, 공격 성공률이 무척 높다는 것이 확인되었습니다. 예를 들어, GPT-4o 백엔드에서 82.5%의 기획 성공률과 75.0%의 실행 성공률이 관찰되었습니다. 또한, 고성능의 폐쇄 소스 모델들은 계획된 악성 행동을 실행할 가능성이 더 높았으며 이는 더 강력한 추론-행동 일관성으로 인해 발생했습니다. 이 결과는 실제 모바일 LVLM 에이전트에 대한 효과적이고 강력한 공격 방법이 될 수 있음을 보여줍니다.



### Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models (https://arxiv.org/abs/2510.07799)
- **What's New**: 이번 논문에서는 Multi-Agent Systems (MAS)에서 Large Language Models (LLMs)이 구동하는 통신 토폴로지를 동적으로 설계하는 새롭고 혁신적인 접근 방식인 Guided Topology Diffusion (GTD)를 소개합니다. GTD는 조건부 그래프 확산(process)을 활용하여 다중 목표를 고려하며 통신 구조를 반복적으로 생성하는 방식으로, 기존의 정적인 설계 방안에서의 한계를 극복하고자 합니다.

- **Technical Details**: GTD는 기존의 단일 단계 모델이 아닌, 경량 프록시 모델을 결합하여 각 단계에서 다중 목표 보상을 예측함으로써 실시간으로 최적화하는 방식을 제공합니다. 이 과정은 높은 복잡도를 가진 통신 구조의 생성을 유도하기 위한 반복적이고 세분화된 접근 방식을 통해 이루어집니다. 이를 통해 각 단계에서 통신 효율성, 비용, 강인성 등을 균형 있게 고려하여 최적의 통신 구조를 창출할 수 있습니다.

- **Performance Highlights**: GTD 프레임워크는 여러 테스트에서 검증을 받았으며, 실험 결과에서 기존 방법들에 비해 현저히 우수한 성능을 보였습니다. GTD는 특히 복잡한 작업에 대해 강력하게 적응할 수 있는 희소하고 효율적인 통신 토폴로지를 자동으로 생성하여 LLM 에이전트 간의 협업을 개선했습니다. 이러한 성과들은 GTD가 다중 목표를 고려하는 통신 구조 설계에서 혁신적인 해결책이 될 수 있음을 보여줍니다.



### HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation (https://arxiv.org/abs/2510.07794)
Comments:
          Under review

- **What's New**: 이 논문에서는 HiPRAG(Hierarchical Process Rewards for Efficient agentic RAG)라는 새로운 강화학습(RL) 훈련 방법론을 제안하며, 이는 기존의 suboptimal search behaviors를 해결하기 위해 설계되었습니다. HiPRAG는 검색 과정의 최적화뿐만 아니라 각 검색 결정에 대한 구체적인 피드백을 제공하여 효율성을 높이고 정확성을 증대시키는 데 초점을 맞추고 있습니다. 특히 이 방식을 통해 over-search와 under-search 문제를 줄이고, 동시에 에이전트의 추론 과정에 대한 세밀한 제어를 가능하게 합니다.

- **Technical Details**: HiPRAG는 LLM이 검색 도구를 사용할 때, 이 도구의 사용 방식이 최적화되어야 한다는 점을 강조합니다. 이 방법론은 에이전트의 추론 과정을 개별적으로 분해하여 각 단계에 대해 적절한 보상을 부여함으로써, 최종 결과뿐 아니라 과정의 품질을 충분히 고려합니다. 이를 통해 검색 결정의 필요성을 동적으로 평가하고, 최적의 검색 비율에 보너스를 제공하는 단계적 보상 체계를 구현하여 기존 방법의 한계를 극복합니다.

- **Performance Highlights**: 실험 결과, HiPRAG는 Qwen2.5와 Llama-3.2 모델에 대해 65.4%(3B) 및 67.2%(7B)의 평균 정확도를 기록했습니다. 이는 기존 모델들에 비해 수집 효율성과 accuracy 모두에서 현저한 개선을 보여주며, over-search 비율을 27%에서 2.3%로 줄이고 under-search 비율도 낮추는 성과를 올렸습니다. 전반적으로 HiPRAG는 다양한 LLM 및 RL 알고리즘에 대해 우수한 일반성을 보이며, 검색 에이전트의 추론 효율성과 최적성을 크게 향상시키는 가능성을 제시합니다.



### LLM4Cell: A Survey of Large Language and Agentic Models for Single-Cell Biology (https://arxiv.org/abs/2510.07793)
Comments:
          34 pages, 5 figures, 7 tables

- **What's New**: 이번 논문에서는 LLM4Cell이라는 대규모 언어 모델과 에이전틱(Agentic) 모델의 통합적 조사를 소개하고 있습니다. 이는 단일 세포 생물학(single-cell biology) 연구를 위한 최초의 포괄적 리포트로, RNA, ATAC, 멀티 오믹(multi-omic), 공간적(spatial) 데이터 모드에서 발전된 58개의 모델을 평가합니다. LLM4Cell은 다섯 가지 방법론 패밀리로 모델을 분류하고 여덟 가지 분석 작업에 맵핑하여, 실험의 일관성과 반복성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: LLM4Cell은 Foundation, Text-Bridge, Spatial, Multimodal, Epigenomic, Agentic 모델의 다섯 가지 범주로 나누어져 있으며, 각 범주는 주석(annotation), 궤적 추론(trajectory inference), 약물 반응 모델링(drug-response modeling) 등과 같은 여덟 가지 주요 작업에 대한 분석을 제공합니다. 각 모델은 생물학적 기반(biological grounding), 공정성(fairness), 개인 정보 보호(priacy), 설명 가능성(explainability) 등을 기준으로 10가지 차원에서 평가됩니다. 이를 통해 모델과 데이터셋 간의 관계를 명확히 하고, 신뢰할 수 있는 AI 개발을 위한 기준을 제시합니다.

- **Performance Highlights**: LLM4Cell은 40개 이상의 공개 데이터셋을 수집하고, 이를 바탕으로 한 모델 평가 방법론을 제공하여, 대규모 언어 모델 교육 및 평가를 위한 약 40개의 벤치마크 데이터셋을 모듈화합니다. 또한, 데이터셋의 불균형, 평가 기준의 한계 등을 감안하여 결과의 공정성과 재현 가능성을 높이는 방안을 모색하고 있습니다. 최종적으로, LLM4Cell은 단일 세포 지능(single-cell intelligence)을 위한 언어 기반 접근법의 통합적 관점을 제시하며, 해석 가능성 및 신뢰성 문제의 해결을 위한 필요 과제를 명확히 하고 있습니다.



### IntentionVLA: Generalizable and Efficient Embodied Intention Reasoning for Human-Robot Interaction (https://arxiv.org/abs/2510.07778)
- **What's New**: 본 연구에서는 IntentionVLA라는 새로운 Vision-Language-Action(VLA) 프레임워크를 제안합니다. 이 모델은 훈련과정에서 효율적인 데이터 주석을 통해 고차원적인 인간의 의도를 해석하며, 이를 통해 빠른 추론 능력을 갖추게 됩니다. IntentionVLA는 복잡한 물리적 환경에서 적절한 행동을 수행하기 위한 두 가지 중요한 도전에 대응합니다. 이를 통해 기존의 VLA 모델들이 겪었던 한계를 극복하고자 합니다.

- **Technical Details**: IntentionVLA는 세 가지 보완적인 형태로 표현된 주석 데이터를 사용합니다: intention reasoning(의도 추론), spatial reasoning(공간적 추론), compact reasoning(간결한 추론). 첫 단계에서 모델은 주석 데이터에서 추론 및 인식 능력을 키우고, 두 번째 단계에서는 이러한 고차원적인 추론이 압축된 단서를 통해 diffusion-based action generator(확산 기반 행동 생성기)로 행동 생성을 안내합니다. 이러한 접근 방식은 논리적 추론과 행동 실행을 명확히 결합하여 고차원적으로 인간의 의도를 이해할 수 있도록 합니다.

- **Performance Highlights**: IntentionVLA는 기존 SOTA VLA 모델들에 비해 모든 평가 환경에서 우수한 성능을 입증했습니다. 직접적인 지침에서는 18% 높은 성공률을, 의도 지시에 대해선 ECoT보다 28% 높은 성공률을 기록했습니다. 더불어, out-of-distribution(배포 외) 의도 작업에서는 모든 기준선 모델들에 대해 두 배 이상의 성공률을 달성하였으며, 40%의 성공률로 제로샷 인간-로봇 상호작용을 가능하게 하였습니다. 이러한 결과들은 IntentionVLA가 차세대 인간-로봇 상호작용 시스템에 대한 유망한 패러다임임을 강조합니다.



### Drift No More? Context Equilibria in Multi-Turn LLM Interactions (https://arxiv.org/abs/2510.07777)
- **What's New**: 이번 연구는 다중 대화에서 발생하는 'context drift'(맥락 드리프트)를 분석하며, 이를 이해하기 위한 간단한 동적 프레임워크를 제안합니다. 연구자들은 KL divergence(쿨백-라이블러 발산)를 토대로 내려진 결과를 통해 맥락 드리프트가 통제 가능한 평형 현상으로 이해될 수 있음을 보여줍니다.

- **Technical Details**: 연구는 특정 테스트 모델과 목표 일관적인 기준 정책 간의 'contextual divergence'(맥락 발산)를 정량화하며, 여러 테스트 설정에서 드리프트의 동적 변화를 분석합니다. 주목할 점은 드리프트가 무한히 커지는 것이 아니라 특정 안정 상태에 수렴하는 경향이 있다는 것입니다.

- **Performance Highlights**: 실험 결과, 목표 리마인더와 같은 간단한 개입이 모델 사이의 드리프트를 줄이는 데 효과적임을 보여줍니다. 이 연구는 다중 대화에서 드리프트가 자연스러운 상태가 아니라 조절 가능한 현상임을 제공하여, 향후 맥락 드리프트 개선 방향에 대한 기초를 제공합니다.



### Trajectory Conditioned Cross-embodiment Skill Transfer (https://arxiv.org/abs/2510.07773)
- **What's New**: 이번 연구에서는 TrajSkill이라는 프레임워크를 제안하여 로봇이 인간 시연 비디오에서 직접 조작 기술을 습득할 수 있게 하였습니다. 기존 방법들이 쌍 데이터셋이나 수작업으로 만든 보상을 의존하는 것과 달리, TrajSkill은 희소(optical flow) 경로를 활용하여 인체와 로봇 조작기 간의 형태적 차이를 제거합니다. 이로 인해 임무의 동적 특성은 유지하면서도 인간의 의도를 효과적으로 전달할 수 있게 되었습니다.

- **Technical Details**: TrajSkill은 세 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 형태 불변(flow sampling) 샘플링을 통해 인간 시연 비디오의 밀집(optical flow)을 얻고 이를 희소 경로로 축소합니다. 둘째, 이러한 경로를 조건으로 하여 로봇을 실행하는 방법을 개발하며, 셋째는 인체와 로봇 간의 기술 전이(cross-embodiment skill transfer)를 구현하는 것입니다. 이렇게 형성된 경로는 최적 보상을 필요로 하지 않으면서도 점진적인 실행이 가능하도록 합니다.

- **Performance Highlights**: MetaWorld 시뮬레이션 데이터에서 TrajSkill은 기존의 최첨단 기술에 비해 FVD를 39.6%, KVD를 36.6% 감소시키는 성과를 보였습니다. 또한, 인체와 로봇 간의 성공률을 최대 16.7% 향상시켜 성능이 크게 개선되었음을 입증하였습니다. 실제 로봇 기구를 활용한 주방 조작 실험에서도 효과적인 인간-로봇 기술 전이가 가능함을 보여주며, 이 기술의 실용성을 강조하였습니다.



### ToolLibGen: Scalable Automatic Tool Creation and Aggregation for LLM Reasoning (https://arxiv.org/abs/2510.07768)
- **What's New**: 이 논문에서는 외부 도구와 함께 사용되는 대형 언어 모델(LLMs)의 향상된 성능을 강조합니다. 특히, 도구가 부족한 도메인에서의 복잡한 추론 작업을 위한 자동화된 도구 생성 방안을 제안합니다. 'ToolLibGen'이라는 파이프라인을 통해 비정형 도구 컬렉션을 구조화된 도구 라이브러리로 리팩토링하는 체계적인 접근 방식을 차별화하여 제공합니다.

- **Technical Details**: ToolLibGen는 질문-응답(QA) 데이터셋으로부터 Python 라이브러리를 구축하는 방법론을 제시합니다. 이 시스템은 질문-Chain-of-Thought(Cot) 쌍으로부터 질문별 도구를 추출하고, 이를 클러스터링하여 기능적으로 관련된 도구를 집약하여 하나의 Python 클래스와 보조 함수로 통합합니다. 또한, 이 과정은 코드 에이전트와 리뷰 에이전트 간의 피드백 루프를 통해 진행되어 각 도구 집합의 기능 유지 및 최적화를 보장합니다.

- **Performance Highlights**: 실험 결과, ToolLibGen을 적용한 구조적 도구 라이브러리는 기존의 비체계적 도구 컬렉션보다 평균적으로 5%에서 10% 이상 높은 성공률을 보였습니다. 더불어 우리의 방법은 전혀 새로운 질문에 대해서도 3% 이상의 정확도 향상을 보여주어, 기능적으로 관련된 질문-specific 도구들을 통합함으로써 도구 검색과 전반적인 추론 성능을 개선하는 능력을 입증하였습니다.



### A Unified Multi-Task Learning Framework for Generative Auto-Bidding with Validation-Aligned Optimization (https://arxiv.org/abs/2510.07760)
- **What's New**: 본 논문에서는 온라인 광고 캠페인의 최적화를 위한 새로운 방법인 Validation-Aligned Multi-task Optimization (VAMO)을 제안합니다. VAMO는 각 작업의 훈련 경량과 유지된 검증 경량 간의 정렬을 기반으로 작업 가중치를 적응적으로 할당하여 검증 성능 향상에 초점을 맞추고 있습니다. 또한, 이 연구는 시계열 모듈을 통합하여 계절적 구조의 크로스 태스크 전이를 강화하고, 실제 배포 목적과 일치하는 업데이트를 유도하는 새로운 비용 구조를 제안합니다.

- **Technical Details**: VAMO는 다중 작업 학습(MTL) 프레임워크를 통해 서로 관련된 작업들 간에 공유된 표현을 기반으로 훈련합니다. 이 프레임워크는 비선형 성격의 입찰 환경에 적응할 수 있는 메커니즘을 제공합니다. 정량적인 이론적 보장을 통해 수렴 보장 및 정렬 분석을 수행하며, 이는 VAMO의 효과를 입증합니다.

- **Performance Highlights**: 실험 결과, VAMO는 전통적인 기준선과 비교했을 때 현저한 성능 향상을 보여줍니다. 시뮬레이션 및 실제 광고 시스템에서의 광범위한 실험을 통해 제안된 방법의 효과성을 강조하며, 산업적 배포에 대한 실용적인 통찰력을 제공합니다. 이 접근법은 특히 수요가 다양하고 동적인 광고 환경에서 강력한 성능을 발휘하는 것으로 나타났습니다.



### Parallel Test-Time Scaling for Latent Reasoning Models (https://arxiv.org/abs/2510.07745)
- **What's New**: 이 논문은 잠재적 추론(latent reasoning) 모델에 대해 병렬 테스트 시간 스케일링(parallel test-time scaling, TTS)을 구현함으로써 이를 개선할 가능성을 탐구합니다. 특히, 토큰 기반 모델이 사용하는 샘플링 및 집계 메커니즘의 한계를 극복하기 위해 두 가지 확률적 샘플링 전략인 모나코 카로우 드롭아웃(Monte Carlo Dropout)과 추가 가우시안 노이즈(Additive Gaussian Noise)를 도입합니다. 또한 잠재 보상 모델(Latent Reward Model, LatentRM)을 설계하여 각 추론 단계에서 추론 과정을 평가하고 안내하는 방법을 제안합니다.

- **Technical Details**: 문서는 병렬 TTS가 잠재적 추론 모델에 효과적으로 적용될 수 있도록 샘플링 및 집계 메커니즘을 재구성하는 방법을 제시합니다. 첫째, 모나코 카로우 드롭아웃과 추가 가우시안 노이즈를 사용하여 잠재 공간에서 추론 경로를 다양하게 샘플링하는 방법을 설명합니다. 둘째, LatentRM을 통해 잠재적 경로를 평가하고 안내하는 세밀한 방법론을 개발하여 성능 향상을 이끌어냅니다.

- **Performance Highlights**: 상세한 실험과 시각적 분석 결과 두 가지 샘플링 전략 모두 계산량이 증가함에 따라 효과적으로 확장됨을 보여줍니다. 모나코 드롭아웃은 비정상적인 솔루션으로의 구조화된 확장을 촉진하며, 추가 가우시안 노이즈는 보다 넓고 동등한 탐색을 유도하여 다양성을 풍부하게 합니다. 최종적으로 LatentRM을 통해 다양한 계산 예산 하에 일관된 성능 향상이 이루어졌습니다.



### UltraLED: Learning to See Everything in Ultra-High Dynamic Range Scenes (https://arxiv.org/abs/2510.07741)
- **What's New**: 이 논문에서는 Ultra-high dynamic range (UHDR) 장면을 단일 짧은 노출 RAW 이미지만을 사용하여 재구성하는 새로운 방법을 제안합니다. 이 방법은 기존의 RGB 기반 방식들이 겪는 정렬 및 고스트 현상을 피할 수 있습니다. 연구팀은 두 단계로 구성된 UltraLED 프레임워크를 도입하여, 첫 번째로 비율 맵을 통해 노출을 보정하고, 두 번째로 밝기 인식 RAW 노이즈 제거기를 통해 어두운 영역의 세부 정보를 향상시킵니다.

- **Technical Details**: UHDR 재구성을 위한 핵심 도전 과제는 노이즈 제거(denoising)와 어두운 영역 정보 복원에 있습니다. RAW 이미지의 높은 비트 깊이(14 bits)와 예측 가능한 노이즈 특성 덕분에 이러한 과제를 해결하는 데 필요한 가능성이 높아집니다. 본 연구는 노이즈 특성이 밝기와 함께 변한다는 점을 인식하고, 이를 처리하기 위해 비율 맵을 활용하여 밝기 정보를 인코딩하는 새로운 노이즈 모델을 도입하였습니다.

- **Performance Highlights**: UltraLED는 기존의 단일 프레임 접근 방식에 비해 현저하게 향상된 성능을 보였습니다. 다수의 실험 결과, 이 방법이 동적 장면에서 어두운 영역의 세부 정보를 효과적으로 복구하는 데 뛰어난 능력을 보여주었습니다. 이러한 연구 결과는 공공 데이터 세트와 코드가 제공되어 후속 연구에 기여할 수 있는 기반을 마련합니다.



### AppForge: From Assistant to Independent Developer -- Are GPTs Ready for Software Development? (https://arxiv.org/abs/2510.07740)
Comments:
          Under Review. Benchmark and leadboards at this https URL

- **What's New**: 이번 논문은 APPFORGE라는 새로운 벤치를 제안하며, 이를 통해 LLMs(대형 언어 모델)의 소프트웨어 개발 능력을 평가하고자 합니다. 이 벤치는 실제 Android 앱에서 추출한 101개의 소프트웨어 개발 문제로 구성되어 있으며, 자연어 사양을 기반으로 앱 기능을 구현하는 작업을 요구합니다. APPFORGE는 실제 애플리케이션의 전체 시스템을 고려하는 평가를 통해 LLM이 실제 개발 시나리오에서도 효과적으로 작동할 수 있는지를 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: APPFORGE는 자동으로 앱 문서에서 주요 기능을 요약하고, 테스트 케이스를 생성하여 앱 구현의 기능적 정확성을 검증하기 위해 다중 에이전트 시스템을 설계하였습니다. 개발된 앱은 APK 파일로 컴파일되고, Android 에뮬레이터 상에서 자동화된 테스트 케이스를 통해 기능 검증이 이루어집니다. 이는 LLM이 상태 관리, 라이프사이클 관리 및 비동기 작업을 이해하고, 이를 통해 맥락에 맞는 견고하고 유지 관리 가능한 코드를 생성해야 함을 요구합니다.

- **Performance Highlights**: 12개의 주요 LLM을 대상으로 한 평가 결과, 모든 모델이 20% 미만의 기능적으로 올바른 애플리케이션을 개발하는 데 그친 것으로 나타났습니다. 특히, GPT-5가 가장 높은 성과를 보였으나, 여전히 18.8%의 성공률에 그쳤습니다. 이러한 결과는 현재 LLM의 복잡한 다중 구성 요소 소프트웨어 엔지니어링 문제를 처리하는 능력에 중대한 한계가 있음을 시사합니다.



### MeSH: Memory-as-State-Highways for Recursive Transformers (https://arxiv.org/abs/2510.07739)
- **What's New**: 이 논문에서는 메모리 관리 문제를 해결하기 위해 새로운 구조인 Memory-as-State-Highways (MeSH)를 도입하였습니다. MeSH는 반복적으로 인스턴스를 생성하는 대신, 명시적인 메모리 버퍼를 활용하여 효율성을 높입니다. 이로 인해 함수적 전문화가 가능해지며, 더 적은 파라미터로도 성능을 향상시킵니다. 특히, MeSH는 1.4B 스케일의 비재귀적 모델들보다 항상 우수한 성능을 보여줍니다.

- **Technical Details**: MeSH 아키텍처는 경량의 라우터를 사용하여 여러 반복 과정에서 컴퓨팅을 동적으로 다양화합니다. 이를 통해 불필요한 정보 과부하를 줄이고, 반복 과정의 각 단계를 구체화하는 데 필요한 분산 메모리를 제공합니다. 또한, 초기 상태에서 최종 상태까지의 전체 네트워크 구조인 Prelude-Recurrent-Coda 구조를 채택하여 성능을 크게 향상시킵니다.

- **Performance Highlights**: Pythia 스위트(160M-1.4B)에서 MeSH를 강화한 재귀형 변환기는 재귀적 기준선보다 일관되게 성능이 향상되었습니다. 평균적인 하류(Downstream) 정확도가 +1.06% 향상되었으며, 비임베딩(non-embedding) 파라미터가 33% 적습니다. 이러한 분석 결과들로 인해 MeSH는 더욱 강력한 재귀적 모델을 구축하기 위한 확장 가능하고 원칙적인 아키텍처로 자리매김하고 있습니다.



### DEAS: DEtached value learning with Action Sequence for Scalable Offline RL (https://arxiv.org/abs/2510.07730)
Comments:
          Project website: this https URL

- **What's New**: 본 연구에서는 액션 시퀀스를 이용한 DEtached value learning with Action Sequence (DEAS)라는 오프라인 강화 학습 프레임워크를 소개합니다. 이는 복잡한 작업에서 가치 학습의 효과성을 극대화할 수 있도록 설계되었습니다. 기존의 단일 스텝 액션 대신 시간적으로 확장된 액션을 활용하여 더 많은 정보를 제공하고, 이는 옵션 프레임워크를 통해 해석될 수 있습니다.

- **Technical Details**: DEAS는 가치 함수에 연속된 액션 타임스텝을 입력으로 처리하며, 시간적으로 확장된 액션으로 nn-단계 TD 업데이트와 유사한 원리를 적용합니다. 이를 통해 명시적인 목표 조건 없이도 수평 감소(horizon reduction)를 제공합니다. 가치 추정의 과대평가 문제를 해결하기 위해 detached value learning을 사용하여 오프라인 데이터셋에서 높은 수익을 올리는 액션을 선호하는 방식으로 가치 추정을 조정합니다.

- **Performance Highlights**: DEAS는 OGBench의 복잡한 장기 작업에서 모든 기준선보다 일관되게 우수한 성능을 보였으며, 실험을 통해 그 유효성을 입증했습니다. 또한, RoboCasa Kitchen과 실제 조작 작업에서 대규모 비전-언어-액션 모델의 성능을 향상시키는 데 성공하여, 오직 전문가의 시연만으로 훈련된 정책에 비해 현저한 성과 향상을 이끌었습니다.



### Causality Guided Representation Learning for Cross-Style Hate Speech Detection (https://arxiv.org/abs/2510.07707)
- **What's New**: 최근 온라인 증오 발언(hate speech)의 확산은 웹의 조화에 심각한 위협이 되고 있습니다. 기존의 증오 발언 탐지 모델은 주로 표면적인 언어적 단서를 기반으로 작동하고 있어 다양한 스타일 변형에 효과적으로 일반화하지 못하는 문제가 있습니다. 이러한 배경을 토대로, 본 논문에서 제안한 CADET는 causal representation learning 프레임워크로, 증오 발언을 해석 가능한 잠재 인자로 분해하여 진정한 증오 의도를 표면적인 언어적 단서와 분리합니다.

- **Technical Details**: CADET는 증오 발언 생성 과정을 모델링하기 위해 causal graph(인과 그래프)를 사용하여, 상황적 환경(contextual environment), 작성자 동기(creator motivation), 대상(target), 스타일(style) 등 핵심 요소들을 포함합니다. 논문의 접근 방식에서는 모델이 각 잠재 인자를 구분하고, 상황적 환경이 미치는 혼란 요인(confounder)을 제어하여 보다 견고한 탐지를 가능하게 합니다. 이론적으로, CADET는 잠재 공간(latent space) 내에서 스타일을 조정함으로써 반사실적 추론(counterfactual reasoning)이 가능하다는 점이 강조됩니다.

- **Performance Highlights**: CADET는 다양한 도전 과제를 중심으로 한 평가에서 우수한 성능을 보여주었으며, cross-style generalization 작업에서 평균 macro-F1이 0.81에 달합니다. 이는 기존의 최첨단 방법에 비해 13%의 상대적 개선을 이룬 것입니다. 각 구성요소의 중요한 역할이 입증되었고, 인과 기반 설계의 효과가 숨겨진 것처럼 분석되었습니다. 이 연구는 일반화된 증오 발언 탐지를 개선하기 위한 인과적인 분리(causal disentanglement)의 가능성을 강조하며, 안전하고 책임감 있는 온라인 환경 조성을 위한 실용적인 함의를 제공합니다.



### Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs (https://arxiv.org/abs/2510.07697)
- **What's New**: 본 연구는 reasoning 기반의 backdoor 공격을 LLMs(대규모 언어 모델)에 대해 체계적으로 조사한 첫 번째 논문으로, 이 공격의 기저 메커니즘과 방법론적 틀, 현재 해결되지 않은 과제를 분석합니다. 새로운 분류 체계를 통해 기존 접근 방식을 요약하며, reasoning을 기반으로 하는 backdoor 공격을 연관형(associative), 수동형(passive), 능동형(active)으로 구분합니다. 또한 이러한 공격에 대한 방어 전략을 제안하고 현재 과제 및 향후 연구 방향에 대해 논의합니다.

- **Technical Details**: 기존의 backdoor 공격은 입력과 출력의 단순 연결을 이용하지만, reasoning 기반의 공격은 모델의 사고 과정을 직접적으로 교란합니다. 이는 고차원적인 사고 과정에 대한 공격으로, 매개된 추론 단계를 오염시키거나 모델의 귀납적 능력을 오도함으로써 공격자가 원치 않는 출력을 유도할 수 있습니다. 우리는 이러한 공격을 세 가지 유형으로 분류하여 그 메커니즘을 정리하고, 각 공격 유형의 정의를 명확히 합니다.

- **Performance Highlights**: 본 논문은 LLMs의 reasoning 능력을 겨냥한 첫 번째 체계적 조사로써, 새로운 분류 체계를 통해 지금까지의 위협 경관을 명확히 하고 Robust defense(강력한 방어 방법)을 개발하기 위한 구조적 기초를 제공합니다. 또한 reasoning 기반의 backdoor 공격에 대한 최신 방어 알고리즘을 제시하고, 이러한 공격의 현재 도전 과제를 논의함으로써 연구자들에게 새로운 방향을 제시하는 데 기여하고 있습니다.



### Stress-Testing Model Specs Reveals Character Differences among Language Models (https://arxiv.org/abs/2510.07686)
- **What's New**: 본 논문에서는 AI 헌법(AI constitutions)과 모델 사양(model specifications)이 대규모 언어 모델(Large Language Models, LLMs)의 행동 지침 및 윤리 원칙을 설정하는 데 미치는 영향을 조사합니다. 기존 모델 사양의 문제점을 정확하게 파악하기 위해 시스템적 스트레스 테스트 방법론을 제시하였으며, 이는 원칙 간의 모순과 해석의 모호성을 자동으로 식별하는 데 초점을 맞추었습니다. 이 연구는 다양한 가치 기반 원칙들을 요구하는 시나리오를 생성하여 모델들이 충돌 상황을 어떻게 처리하는지를 분석합니다.

- **Technical Details**: 연구팀은 3,307개의 세분화된 가치를 포함하는 세분화된 분류법을 활용하여 300,000개 이상의 다양한 쿼리 시나리오를 생성합니다. 이 시나리오들은 상충하는 원칙 간의 명시적인 거래를 강요하는 구조를 가지고 있으며, 이는 최첨단 LLM인 Anthropic, OpenAI, Google, xAI 모델들의 응답을 분석하는 데 사용됩니다. 특히, 각 모델의 응답에서의 불일치를 측정하기 위해 가치 분류 점수(value classification scores)를 사용하여 응답 간의 차이를 정량화합니다.

- **Performance Highlights**: 실증 분석 결과, 300,000개 시나리오 중 22만 개 이상에서 최소 한 쌍의 LLM 간에 상당한 불일치가 나타났으며, 7만 개 이상에서 대부분의 모델 간 행동의 차이를 식별했습니다. 높은 불일치는 사양 위반(specification violations)과 함께 직접적으로 연결되어 있으며, 이는 모델의 행동에 대한 명확한 시사점을 제공합니다. 또한, 모델들은 명확한 지침이 부족할 때 체계적인 가치 우선순위 패턴을 나타내어 이들 간의 차이를 드러냅니다.



### Curriculum Learning with Synthetic Data for Enhanced Pulmonary Nodule Detection in Chest Radiographs (https://arxiv.org/abs/2510.07681)
Comments:
          32 pages, 6 figures,

- **What's New**: 이 연구는 커리큘럼 학습(curriculum learning)과 확산 기반의 합성 증강(diffusion-based synthetic augmentation)을 통합하여 흉부 방사선 사진에서 어려운 폐 결절을 더 효과적으로 탐지할 수 있는지 평가합니다. 기존 AI 모델이 데이터 불균형과 제한된 주석으로 인해 어려움을 겪는 저 사이즈, 저 밝기, 저대비 결절에 중점을 두고 새로운 접근 방식을 제시합니다. 제안된 모델은 Faster R-CNN을 기반으로 하며, 특히 작은 결절에 대한 탐지 성능이 강조됩니다.

- **Technical Details**: Faster R-CNN과 Feature Pyramid Network(FPN)를 기반으로 한 모델이 전문가가 주석을 단 NODE21 데이터셋 및 기타 공개 데이터셋, 그리고 DDPM에서 생성된 11,206개의 합성 이미지를 포함한 혼합 데이터셋에 대해 학습되었습니다. 어려움 점수는 결절의 크기, 밝기 및 대비에 따라 설정되었으며, 모델은 5겹 내부 교차 검증을 통해 평가되었습니다. 훈련을 위한 커리큘럼 학습 전략이 도입되어 점진적으로 더 어려운 샘플로 성장했습니다.

- **Performance Highlights**: 모델은 평균 AUC(Area Under Curve) 0.95를 달성하여 기존 기준 모델의 0.89에 비해 유의미한 향상을 나타냈습니다(p < 0.001). 민감도(sensitivity)는 70%로 증가했으며, 정확도(accuracy)는 82%로 향상되었습니다. 모든 어려움 범주에서 일관된 성과를 보였고 Grad-CAM 시각화 결과는 커리큘럼 학습 하에서 모델의 해부학적 초점이 더 명확해짐을 보여주었습니다.



### Controllable Video Synthesis via Variational Inferenc (https://arxiv.org/abs/2510.07670)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 사용자의 다양한 제어를 반영한 비디오 합성(method for video synthesis) 방법을 제안합니다. 기존의 비디오 생성 모델들은 고정된 입력 형식에 맞춰졌지만, 본 연구는 더 다양한 입력 방식을 수용하는 시스템을 개발합니다. 이를 통해 고해상도 장면과 물리적 일관성(physical consistency)을 유지하는 동시에 사용자가 원하는 요소에 대한 높은 제어력을 제공합니다.

- **Technical Details**: 연구에서는 변분추론(variational inference)을 기반으로 다수의 비디오 생성 백본(backbone)을 활용하여 모든 작업 제약을 집합적으로 반영하는 방법을 사용합니다. 문제 최적화는 단계별 KL 발산(KL divergence) 최소화를 통해 해결하며, 컨텍스트 조건부 팩토리제이션(context-conditioned factorization) 기술을 도입하여 지역적 최적 최소화를 피합니다. 최종 목표 분포에 도달하기까지 불리언 샘플을 생성하면서도 샘플의 다양성을 유지합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존의 비디오 생성 작업들과 비교하여 시각적 충실도(visual fidelity), 출력의 다양성(output diversity), 그리고 장면의 일관성(scene consistency)이 개선된 결과를 보였습니다. 이 연구는 길고 복잡한 비디오 생성 상황에서 특히 뛰어난 성능을 발휘함을 확인하였습니다. 새로운 제어 인터페이스를 통해 사용자는 비디오 생성 과정에서 더 많은 유연성과 자유도를 가지게 되었습니다.



### TCIP: Threshold-Controlled Iterative Pyramid Network for Deformable Medical Image Registration (https://arxiv.org/abs/2510.07666)
- **What's New**: 이 논문에서는 기하학적 의료 이미지 정합에서 피라미드 네트워크(pyramid networks)의 성능을 개선하기 위해 Feature-Enhanced Residual Module (FERM)와 Threshold-Controlled Iterative (TCI) 전략을 제안합니다. FERM은 각 디코딩 레이어에서 핵심 구성 요소로, 해부학적 의미 특징을 추출하고 관련 없는 특징을 억제하여 등록의 정확성을 높입니다. 또한 TCI 전략은 각 이미지 쌍에 적응적으로 최적화 반복 횟수를 결정하여 등록 성능을 극대화합니다.

- **Technical Details**: FERM은 3개의 연속 블록, 즉 Feature Fusion Block (FFB), Squeeze Excitation Block (SEB), Deformation Field Estimator (DeF)로 구성됩니다. 이러한 블록들은 각각 해부학적 특징을 추출하고, 불필요한 특징을 억제하며, 최종 변형 필드를 추정합니다. TCI 전략은 두 단계로 이루어져 있으며, 첫 번째 단계에서는 등록 안정성을 평가하고, 두 번째 단계에서 수렴성을 평가하여 각 이미지 쌍의 반복 횟수를 동적으로 결정합니다.

- **Performance Highlights**: 본 연구는 세 개의 공개된 뇌 MRI 데이터셋과 하나의 복부 CT 데이터셋에서 수행된 실험을 통해 TCIP가 기존의 최첨단 등록 네트워크보다 높은 정확도를 보여주며, 유사한 추론 속도와 모델 파라미터 크기를 유지함을 증명합니다. 또한 FERM과 TCI의 일반화 가능성을 평가하기 위해 기존 등록 네트워크와 통합하여 추가적인 실험을 진행하였고, 두 제안 방법의 효과성을 검증하는 탈락 연구를 수행하였습니다.



### IKNet: Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators (https://arxiv.org/abs/2510.07661)
Comments:
          9 pages

- **What's New**: 이 논문은 주식 가격에 대한 뉴스 기사와 같은 비구조적 외부 정보의 영향을 모델링하기 위해 새로운 접근 방식을 제안합니다. 기존의 예측 모델은 일반적인 감정 점수나 평균 임베딩을 사용하지만, 이러한 방법은 공적 감정이 예측에 미치는 영향을 정량적으로 설명하는 데 한계가 있었습니다.

- **Technical Details**: 저자들은 감정 연관성을 모델링하기 위해 해석 가능한 키워드 제공 네트워크(Interpretable Keyword-Guided Network, IKNet)를 제안합니다. IKNet은 FinBERT 기반의 맥락 분석을 통해 중요한 키워드를 식별하고, 각 임베딩을 개별 비선형 프로젝션 레이어를 통해 처리한 후, 기술 지표의 시계열 데이터와 통합하여 다음 날 종가를 예측합니다.

- **Performance Highlights**: IKNet은 2015년부터 2024년까지의 S&P 500 데이터를 대상으로 한 경험적 평가에서 순환 신경망(recurrent neural networks) 및 변환기 모델(transformer models)보다 더 우수한 성능을 보였습니다. RMSE를 최대 32.9% 감소시키고 누적 수익률을 18.5% 향상시키며, 공적 감정에 의해 발생하는 변동성 사건에 대한 맥락화된 설명을 제공하여 투명성을 높입니다.



### OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inferenc (https://arxiv.org/abs/2510.07651)
- **What's New**: 이번 연구에서는 Optimal Brain Cache (OBCache)라는 새로운 프레임워크를 제안합니다. OBCache는 키-값 (KV) 캐시 제거를 계층별 구조적 프루닝 문제로 공식화하여 메모리 오버헤드를 줄입니다. 이는 특정 토큰의 중요도를 평가할 때, 주의 가중치만이 아닌 주의 출력의 변화도 고려하는 접근 방식을 취합니다.

- **Technical Details**: OBCache는 Optimal Brain Damage (OBD) 이론을 기반으로 하여, 각 토큰의 제거가 주의 출력에 미치는 영향을 측정합니다. 이 프레임워크는 고립된 키, 고립된 값, 그리고 키-값 쌍의 세 가지 프루닝 단위를 정의하고, 이를 통해 캐시 제거 및 유지 결정을 내리기 위한 폐쇄 형태의 점수들을 도출합니다. OBCache는 기존의 주의 기반 점수 산정 방식과 비교하여 보다 풍부하고 정확한 신호를 제공합니다.

- **Performance Highlights**: OBCache는 LLaMA 및 Qwen 모델에서 다양한 긴 컨텍스트 작업에 대해 실험을 통해 성능 향상을 입증하였습니다. 특히, OBCache의 점수를 기존의 방법들과 결합함으로써 전체적인 성능이 개선되었고, 메모리 소모를 줄이면서도 긴 컨텍스트 추론 성능을 향상시키는 데 기여했습니다.



### Value Flows (https://arxiv.org/abs/2510.07650)
- **What's New**: 이 논문은 강화 학습(Reinforcement Learning)에서 미래 보상의 전체 분포를 추정하기 위한 새로운 접근법인 Value Flows를 제안합니다. 기존의 방법들이 단일 스칼라로 보상을 표현하는 데 비해, 이 연구는 강화 학습의 신호를 강화하고 탐색 및 안전한 RL에 응용할 수 있도록 보상 분포를 활용합니다. 이 방법은 고급 흐름 기반 모델(flow-based model)을 사용하여 미래 보상의 분포를 추정하고 높은 변동성(return variance)을 가진 상태를 식별합니다.

- **Technical Details**: Value Flows는 흐름 일치(objective)를 통해 분포적 벨만 방정식(distributional Bellman equation)에 부합하는 확률 밀도 경로를 생성하는 새로운 방식으로, 강화 학습에서 보상 분포를 모델링합니다. 이를 위해 흐름 파생 미분 방정식(flow derivative ODE)을 활용하여 다양한 상태에서의 보상 변동성을 추정하고, 불확실성 정보를 이용해 특정 전환에서 보다 정확한 보상 추정을 학습할 수 있는 우선순위를 정합니다. 이 방식은 이전의 오프라인(offline) 및 온라인-오프라인(online-to-online) 방법들과 비교하여 성능 향상을 가져옵니다.

- **Performance Highlights**: Value Flows 방법은 3737개의 상태 기반과 2525개의 이미지 기반 벤치마크 과제에서 실험을 통해 평균 1.3배 더 높은 성공률을 기록하며 기존 방법들보다 우수한 성능을 나타냈습니다. 이러한 성능 향상은 강화 학습 실제 응용 분야에서 높은 보상 예측 정확성을 보장할 수 있음을 의미합니다. 따라서 이 연구는 미래 보상의 추정을 보다 정교하게 수행하는 데 기여하고 있습니다.



### Banking Done Right: Redefining Retail Banking with Language-Centric AI (https://arxiv.org/abs/2510.07645)
Comments:
          Accepted at EMNLP2025 Industry Track

- **What's New**: 이 논문은 자연어 대화를 통해 고객이 주요 금융 거래를 수행할 수 있도록 지원하는 Ryt AI라는 LLM 네이티브 에이전트 프레임워크를 소개합니다. 이는 대화형 AI가 주요 은행 인터페이스로 기능하는 최초의 글로벌 규제 승인 배포를 의미하며, 기존의 보조적 역할에 한정된 기술들과는 차별화됩니다. Ryt AI는 내부적으로 개발된 ILMU라는 폐쇄형 LLM의 지원을 받아 다중 스크린의 고정형 워크플로우를 단일 대화 인터페이스로 전환합니다.

- **Technical Details**: Ryt AI는 4개의 LLM 기반 에지전트(Guardrails, Intent, Payment, FAQ)로 구성되어 있으며, 각 에이전트는 ILMU에 특정 작업을 위한 LoRA 어댑터를 부착합니다. 이러한 구조는 은행의 인프라 내에서 호스팅되어 최소한의 오버헤드로 일관된 동작을 보장합니다. 또한, 결정론적 가드레일, '인 더 루프' 인간 확인, 상태 비저장 감사 아키텍처를 통해 보안 및 규정 준수를 위한 다층적인 방어를 제공합니다.

- **Performance Highlights**: Ryt AI는 규제 승인을 받은 최초의 LLM 기반 에이전트 시스템으로, 디지털 뱅크의 핵심 트랜잭션을 직접 실행할 수 있도록 돕습니다. 이 시스템은 사용자가 자연어로 의도를 표현하고, 시스템이 이를 조합, 검증 및 실행하는 과정을 통해 사용자 경험을 혁신적으로 개선합니다. 또한, 다국어 지원과 규제 요구를 맞춘 지역화된 적응을 통해 말레이시아 시장에 최적화된 솔루션을 제공합니다.



### Retentive Relevance: Capturing Long-Term User Value in Recommendation Systems (https://arxiv.org/abs/2510.07621)
- **What's New**: 이 논문에서는 Retentive Relevance라는 새로운 콘텐츠 수준의 설문 기반 피드백 지표를 소개합니다. 이는 사용자가 유사한 콘텐츠를 위해 플랫폼에 돌아올 의도를 직접 평가하여, 장기적인 사용자 만족도와 유지율을 측정할 수 있는 강력한 예측 도구로 제안됩니다. Retentive Relevance는 기존의 단기적인 만족도를 중심으로 한 설문 지표와는 차별화됩니다. 이를 통해 우리는 사용자 행동의 날짜별 유지율을 개선하는 데 성공했습니다.

- **Technical Details**: Retentive Relevance는 심리측정(psychometric) 방법을 통해 검증된 유효한 설문 구성으로, 사용자 만족과 의도를 동시에 측정함으로써 추천 시스템에서 유용한 도구로 자리잡고 있습니다. 이 연구는 대규모 오프라인 모델링, A/B 실험 등을 통해 Retentive Relevance가 전통적인 참여 신호와 다른 설문 지표들보다 더 뛰어난 예측 성능을 보임을 입증하였습니다. 따라서 우리는 추천 시스템에서 Retentive Relevance를 통합하는 프로덕션 준비 모델을 개발하여, 실질적인 운영에서의 적용 가능성을 높였습니다.

- **Performance Highlights**: 저자들은 Retentive Relevance가 사용자 유지율, 참여도, 콘텐츠 품질을 향상하는 데 효과적이라는 것을 대규모 실험을 통해 보여주었습니다. 특히 제한된 역사적 참여 기록을 가진 사용자들에게 전통적인 지표보다 더 좋은 성과를 내는 것이 특징입니다. 이에 따라 플랫폼의 성장과 사용자 경험을 동시에 개선할 수 있는 확장 가능하고 사용자 중심의 솔루션을 제공하게 되었습니다.



### DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation with Uncertainty-Quantification Suppor (https://arxiv.org/abs/2510.07620)
Comments:
          18 pages, 9 figures, 5 tables

- **What's New**: DGTEN (Deep Gaussian-based Trust Evaluation Network)은 변화하는 관계를 효과적으로 포착하고 신뢰를 평가하기 위한 통합 그래프 프레임워크를 제안한다. 이 프레임워크는 불확실성을 인식하는 메시지 패싱(message passing)과 표현력이 뛰어난 시간 모델링(temporal modeling)을 통합하여 신뢰-targeted 공격에 대해 내장된 방어 기능을 제공한다. DGTEN은 노드와 엣지를 가우시안 분포(Gaussian distributions)로 표현하여 신뢰 결정을 내릴 때 더 높은 위험 인식(risk-aware) 판단을 가능하게 한다.

- **Technical Details**: DGTEN은 하이브리드 절대-가우시안-아워글래스(Hybrid Absolute-Gaussian-Hourglass, HAGH) 포지셔널 인코딩(positional encoding)과 Kolmogorov-Arnold 네트워크 기반의 다중 헤드 주의(multi-head attention) 방법을 채택한다. 또한, 일반적인 미분 방정식(ODE)을 기반으로 한 잔여 학습 모듈(residual learning module)을 통해 신뢰가 어떻게 발전하는지를 모델링한다. 마지막으로, RAECA(Robust Adaptive Ensemble Coefficient Analysis) 방어 기법을 사용하여 의심스러운 상호작용을 식별하고 이를 완화하여 공격에 대한 저항성을 높인다.

- **Performance Highlights**: DGTEN은 비트코인 신뢰 네트워크를 대상으로 한 실험에서 유의미한 개선을 보였다. 단일 시간 간격 예측(single-timeslot prediction)에서 기존 동적 기준선(baseline)보다 10.77% 더 높은 MCC를 기록했으며, 콜드 스타트(cold-start) 시나리오에서는 16.41%의 MCC 이득을 달성했다. 적대적 공격과의 비교에서도 DGTEN은 최대 11.63%의 개선을 보여주며, 저항성과 유용성을 입증하였다.



### Vocabulary embeddings organize linguistic structure early in language model training (https://arxiv.org/abs/2510.07613)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 입력 임베딩 벡터의 기하학적 구조가 어떻게 구성되며 훈련 과정 중 이 구조가 어떻게 변화하는지를 탐구합니다. 우리는 두 개의 오픈소스 모델인 Pythia 12B와 OLMo 7B에 대한 실험을 통해 입력과 출력 임베딩 간의 기하학적 구조와 의미론적(semantic), 구문론적(syntactic), 빈도 기반(frequency-based) 메트릭 간의 상관관계를 분석하였습니다.

- **Technical Details**: 연구는 representational similarity analysis를 사용하여, 의미론적 및 구문론적 특징과 높은 상관 관계를 갖는 입력 임베딩의 기하학적 구조에 대한 실험을 수행하였습니다. 훈련 초기에, 고빈도(high-frequency) 및 기능어(function words)임베딩은 일반적으로 사전 훈련된 초기 구조와 더 빠르게 수렴하는 경향이 있음을 발견했습니다. 반면, 저빈도(low-frequency) 단어는 초기 편향(alignment)을 어느 정도 유지하며, 최종 벡터로의 수렴이 더디게 진행됩니다.

- **Performance Highlights**: 본 연구에서는 훈련 과정 동안 어휘(어휘) 임베딩의 기하학적 구조가 어떻게 언어 구조를 중심으로 조직되는지를 보여주는 동적인 경로를 제시합니다. 또한, 단어의 빈도와 기능이 임베딩의 수렴 속도에 미치는 미묘한 역할을 강조하며, 이러한 결과는 모델 훈련 동안 특정 기능 향상을 촉진하기 위한 어휘 기하학의 진화에 대한 심도 있는 연구의 필요성을 제기합니다.



### TGM: a Modular and Efficient Library for Machine Learning on Temporal Graphs (https://arxiv.org/abs/2510.07586)
Comments:
          21 pages, 5 figures, 14 tables

- **What's New**: 이번 논문에서는 Temporal Graph Modelling (TGM)이라는 연구 지향의 라이브러리를 소개합니다. 이는 두 가지 시간 동적 그래프 접근 방법, 즉 Continuous-Time Dynamic Graph (CTDG)와 Discrete-Time Dynamic Graph (DTDG)를 통합한 최초의 라이브러리입니다. TGM은 동적인 노드 특성, 시간-세분화 변환, 링크 및 노드 수준의 작업을 원활히 지원하여 사용자 경험을 향상시킵니다.

- **Technical Details**: TGM은 모듈식이고 효율적인 프레임워크로, 연구자들이 시간 동적 그래프 내에서 더 쉽게 실험할 수 있도록 돕습니다. 이 라이브러리는 8개 방법을 구현하며, 시간 granularity를 API에 자연스럽게 통합하여 그래프 세분화와 스냅샷 반복을 지원합니다. 또한, TGM은 그래프 모델 훈련 속도에서 DyGLib보다 평균 7.8배 더 빠르며, 그래프 세분화에서 175배 더 빠른 성능을 보여줍니다.

- **Performance Highlights**: TGM은 동적 그래프 속성 예측과 시간 기반 훈련 패러다임을 실현하여 연구 가능성을 확대합니다. 실험을 통해 TGM이 기존 구현에 비해 평균 175배 더 빠른 그래프 세분화 성능을 발휘함을 입증하였으며, 이는 연구 시간이 단축되고 새로운 통찰력을 제공할 수 있는 기회를 창출합니다. TGM은 링크, 노드 및 그래프 수준의 예측 작업에서 가장 폭넓은 지원을 제공하는 특징을 가지고 있습니다.



### Linguistic Patterns in Pandemic-Related Content: A Comparative Analysis of COVID-19, Constraint, and Monkeypox Datasets (https://arxiv.org/abs/2510.07579)
Comments:
          16 pages

- **What's New**: 이번 연구는 팬데믹 관련 온라인 담론에서 언어가 건강에 대한 허위 정보와 사실적 의사소통을 어떻게 구별하는지를 분석했습니다. COVID-19와 Monkeypox 관련 데이터셋을 통해 허위 정보가 가지는 독특한 언어적 특징을 밝혀냈습니다. 이를 통해 시의적절한 공중 보건 메시지를 전달하기 위한 전략에도 기여할 수 있다는 점이 주목할 만합니다.

- **Technical Details**: 연구는 COVID-19 허위 내러티브, 일반 COVID-19 콘텐츠, Monkeypox 관련 게시물의 세 가지 코퍼스를 분석했습니다. 분석 결과, COVID-19 허위 정보는 낮은 readability 점수와 함께 두 배 이상의 두려움 관련 또는 설득 용어를 포함하고 있으며, Monkeypox 콘텐츠보다 감정적으로 더 표현되는 스타일과 대조적입니다. 이로 인해 허위 정보는 복잡한 수사적 스타일과 감정적 신호를 사용하며, 이러한 조합이 신뢰도를 높일 수 있음을 시사합니다.

- **Performance Highlights**: 연구 결과는 디지털 건강 허위 정보에 대한 언어적 지표를 강조하면서 탐지 노력을 지원하는 데 기여하고 있습니다. 또한, 네트워크 미디어 환경에서의 위기 소통 이론 모델과 공중 보건 메시징 전략에도 중요한 정보를 제공합니다. 그러나 전통적 readability 지수에 의존하거나 정적 집계 분석을 사용하는 등의 한계가 있으므로, 향후 연구에서는 더 넓은 감정 어휘와 플랫폼에 민감한 접근 방식을 통합해야 할 필요성이 제기됩니다.



### Accuracy, Memory Efficiency and Generalization: A Comparative Study on Liquid Neural Networks and Recurrent Neural Networks (https://arxiv.org/abs/2510.07578)
Comments:
          13 pages, 12 figures. Submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

- **What's New**: 이번 논문에서는 LNN(Liquid Neural Networks)과 기존 RNN(Recurrent Neural Networks) 구조의 비교 분석을 통해, 각 모델의 정확성(accuracy), 메모리 효율(memory efficiency), 일반화 능력(generalization ability)을 평가합니다. LNN은 생물학적 신경 시스템에서 영감을 받아, 불규칙하게 샘플링된 데이터를 보다 효과적으로 처리할 수 있는 잠재력을 지니고 있습니다. 또한, LNN의 몇몇 변종은 전통적인 RNN에 비해 파라미터 효율성과 연산 속도에서 우수한 성능을 보입니다.

- **Technical Details**: 기존 RNN은 내부 재귀 구조를 통해 순차적 데이터의 시간 의존성을 포착하고, LSTM(Long Short-Term Memory)과 GRU(Gated Recurrent Units)와 같은 변형들이 그 성능을 향상시켜왔습니다. 반면, LNN은 연속적인 시간 동적 시스템 이론을 기반으로 하여, 일반적인 미분 방정식(ODEs)을 통해 신경 상태의 연속적 변화를 묘사합니다. 이는 더 높은 일반화 능력과 함께 노이즈에 강한 성능을 보일 수 있게 합니다.

- **Performance Highlights**: 현재 LNN은 전통적인 RNN이 가진 긴 의존성 학습의 한계와 기울기 문제를 극복할 수 있는 가능성을 보여주고 있습니다. 연구 결과는 LNN이 다양한 실제 사례에서 예측 정확성을 높이고 전통적인 모델보다 우수한 성능을 발휘할 수 있음을 시사합니다. 특히, 향후 연구 방향으로 LNN의 확장 가능성을 강조하며 다양한 복잡한 시나리오에서의 응용 가능성을 모색하고 있습니다.



### Multi-Task Pre-Finetuning of Lightweight Transformer Encoders for Text Classification and NER (https://arxiv.org/abs/2510.07566)
Comments:
          Accepted by EMNLP 2025 Industry Track

- **What's New**: 이 연구에서는 자연어 처리(NLP) 모델의 모바일 플랫폼 배치를 위해 다양한 응용 프로그램에 적응하면서 메모리와 계산 비용을 효율적으로 유지하는 경량 BERT 유사 인코더의 사전 파인튜닝(pre-finetuning) 전략을 조사합니다. 주목할만한 점은 멀티태스크(pre-finetuning) 방법이 서로 상충하는 최적화 신호를 도입하여 전체 성능을 저하시킬 수 있다는 것입니다. 이를 해결하기 위해, 우리는 작업 주요 LoRA 모듈을 기반으로 한 새로운 멀티태스크 사전 파인튜닝 프레임워크를 제안하고 있습니다.

- **Technical Details**: 모바일 애플리케이션은 이메일에서 자동으로 일정 이벤트를 생성하거나 메시지를 기반으로 개인화된 추천을 제공하는 등 여러 자연어 처리 작업을 해결해야 합니다. 본 연구에서는 NER과 텍스트 분류라는 두 가지 기본 NLP 작업 가족을 다루기 위해 사전 파인튜닝 전략을 활용했습니다. 사전 파인튜닝 이후, 최적화된 인코더를 통해 모듈형 어댑터를 사용하여 적용할 수 있는 간단하지만 효과적인 멀티태스크 구조를 구현했습니다.

- **Performance Highlights**: 21개의 하위 작업에 대한 실험 결과, NER에서는 평균 0.8% 향상되고 텍스트 분류에서는 평균 8.8% 향상되었습니다. 이는 제안된 방법이 다양한 모바일 NLP 응용 프로그램에 효과적임을 보여줍니다. 사전 파인튜닝 단계를 통해, 각 작업에 대해 개별적으로 좋은 성과를 올리지만 서로 간섭이 발생하여 이를 해소하기 위한 접근법을 통해 성능 저하를 최소화했습니다.



### Investigating Thematic Patterns and User Preferences in LLM Interactions using BERTopic (https://arxiv.org/abs/2510.07557)
- **What's New**: 이 연구는 변환기 기반의 주제 모델링 기법인 BERTopic을 다국어 대화 말뭉치인 lmsys-chat-1m 데이터 셋에 적용하였습니다. 이 데이터 셋은 대형 언어 모델(LLMs)에 대한 평가를 통해 수집된 사용자 프롬프트와 두 개의 비식별 LLM 응답을 포함하고 있습니다. 연구의 주요 목표는 이러한 대화에서 주제 패턴을 발굴하고, 특정 LLM이 특정 주제 내에서 일관되게 선호되는 것을 분석하는 것입니다.

- **Technical Details**: BERTopic은 고차원 변환기 기반 임베딩을 활용하여 주제 모델링을 수행하며, HDBSCAN을 클러스터링 알고리즘으로 사용하여 유사한 문서의 그룹을 식별합니다. 연구에서는 29개 이상의 일관된 주제를 추출하였으며, 이는 인공지능, 프로그래밍, 윤리 및 클라우드 인프라와 같은 다양한 기술적 주제를 포함하고 있습니다. 또한, 문서 전처리 파이프라인을 설계하여 다국어 변형을 균형 있게 유지하고 노이즈나 편집된 데이터를 정리했습니다.

- **Performance Highlights**: 분석 결과, 주제와 모델 선호도의 관계를 파악하여 모델-주제 정렬의 경향을 확인했습니다. 데이터 시각화 기법으로는 주제 간 거리 맵과 모델 대 주제 매트릭스가 포함되어 있으며, 이는 LLM의 실제 성능과 사용자 만족도를 향상시키기 위한 도메인 특화된 미세 조정 및 최적화 전략을 알리는 데 기여합니다.



### Label Semantics for Robust Hyperspectral Image Classification (https://arxiv.org/abs/2510.07556)
Comments:
          This work has been accepted for publication in the proceedings of IJCNN 2025

- **What's New**: 이 논문에서는 Semantically Guided Semantic Spectral-Spatial Fusion Network(S3FN)를 제안하여 고차원 HSI(초분광 이미지) 데이터의 분류 성능을 향상시키고자 합니다. 이 모델은 각 클래스 레이블에 대한 텍스트 설명을 생성하여, 기존의 단일 모드 방법의 한계를 극복하고 보다 의미 있는 피쳐-레이블 정렬을 가능하게 합니다. 이를 통해 과거 모델들이 놓치곤 했던 세부적인 스펙트럼 정보와 의미적 관계를 효과적으로 활용합니다.

- **Technical Details**: S3FN은 LLM(대형 언어 모델)을 활용하여 각 클래스의 고유한 특성과 스펙트럼 행동을 반영한 포괄적인 텍스트 설명을 생성합니다. 이 설명들은 BERT나 RoBERTa와 같은 사전 훈련된 텍스트 인코더를 통해 벡터 공간에 임베딩되어, HSI 데이터의 특성 표현을 강화합니다. LLM을 사용함으로써, 수동으로 작성된 불완전하거나 모호한 설명 문제를 해결하고, 다양한 클래스 간의 스펙트럴 관계를 더 깊이 이해할 수 있게 합니다.

- **Performance Highlights**: 논문에서 제안하는 S3FN은 Hyperspectral Wood, Hyperspectral Blueberries, DeepHS-Fruit의 세 가지 HSI 벤치마크 데이터셋에서 평가되었으며, 기존 방법에 비해 성능이 유의미하게 향상되었습니다. 텍스트 기반 의미적 정보와 스펙트럴-공간적 데이터를 융합한 결과, 보다 정교한 분류 성능을 달성하게 되었으며, 이는 HSI 분류의 발전 가능성을 보여줍니다.



### TRAVL: A Recipe for Making Video-Language Models Better Judges of Physics Implausibility (https://arxiv.org/abs/2510.07550)
- **What's New**: 이 논문에서는 비디오 생성 모델들이 물리적 법칙을 위반한 입력들을 자주 생성하는 문제를 다룹니다. 기존의 Video-Language Models (VLMs)가 이러한 물리적 현실성을 평가하는 데 어려움을 겪고 있어서, 새로운 방법론인 TRAVL과 평가 기준인 ImplausiBench를 도입하여 이 문제를 해결하고자 합니다. TRAVL은 모션을 인식할 수 있는 주의(attention) 메커니즘을 사용하여 VLM의 물리적 이해도를 향상시키고, 정교한 평가 방식을 통해 물리적 불가능성을更正하는 방법을 제시합니다.

- **Technical Details**: TRAVL(trajectory-aware Vision-Language learning)은 VLM을 개선하기 위한 모듈식 방법으로, 모션 정보를 기억하는 자기 주의(self-attention) 메커니즘을 통해 비디오의 물리적 구조를 포착하는 데 중점을 둡니다. 이 방법은 두 가지 주요 메커니즘인 intra-frame 공간 주의와 trajectory-aware temporal 주의를 통해 비디오의 물리적 동적을 더 잘 파악할 수 있도록 설계되었습니다. 또한, ImplausiBench라는 300개의 동영상으로 구성된 평가 기준을 통해 인간과 LLM의 판단 모두에서 보다 엄격한 물리적 현실성을 평가할 수 있도록 하였습니다.

- **Performance Highlights**: TRAVL과 ImplausiBench는 VLM의 물리적 현실성 향상을 위한 통합된 프레임워크를 제공합니다. 이들은 VLM이 임의의 동영상에서 물리적으로 가능한지 검사할 수 있도록 하며, 기존 모델들이 놓치는 심층적인 물리적 이해를 목표로 합니다. 이러한 접근 방식은 물리적 상상력의 한계와 기존의 벤치마크가 갖는 제약사항을 극복하고자 하며, 더 나아가 VLM의 발전 방향에 중요한 기여를 할 것으로 기대됩니다.



### OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs (https://arxiv.org/abs/2510.07535)
- **What's New**: 이 논문에서는 자주 사용되는 기존 스펙ulative 디코딩 기법의 한계를 극복하기 위해 새로운 장기 컨텍스트 벤치마크(LongSpecBench)와 모델(OWL)을 소개합니다. OWL은 기존 모델(EAGLE3)에 비해 승인 길이(acceptance length)가 약 5배 더 길어지는 혁신을 통해 성능을 개선했습니다. 이 연구는 일반적인 작업 환경을 보다 잘 반영하여 스펙ulative 디코딩 기법의 실용성을 높이고자 합니다.

- **Technical Details**: OWL에서는 세 가지 주요 혁신을 통해 성능을 크게 향상시킵니다. 첫째, LSTM 기반의 드래프트를 사용하여 마지막 토큰 상태에만 의존하여 다양한 길이에 일반화할 수 있도록 합니다. 둘째, 검증기(verifier)에서 특별한 토큰([SPEC])을 도입하여 드래프트에 더 풍부한 표현력을 제공합니다. 셋째, 트리와 비트리 트리 디코딩 방법을 결합한 하이브리드(hybrid) 알고리즘을 사용하여 보다 높은 승인 길이와 속도를 달성합니다.

- **Performance Highlights**: 실험 결과, OWL의 성능은 EAGLE3를 능가하며 특히 긴 컨텍스트 입력에서 두드러진 성과를 보였습니다. 새로운 벤치마크는 일부 기존 스펙ulative 디코딩 방법이 긴 컨텍스트에서 제대로 작동하지 않는다는 점을 강조합니다. 이 연구의 모든 코드와 데이터셋이 공개되어 향후 연구에 기여할 수 있도록 했습니다.



### EEG Sleep Stage Classification with Continuous Wavelet Transform and Deep Learning (https://arxiv.org/abs/2510.07524)
Comments:
          11 pages, 2 figures

- **What's New**: 본 연구에서는 수면 장애의 정확한 진단과 관리를 위해 수면 단계 분류의 자동화된 새로운 프레임워크를 제안합니다. 기존의 수면 채점 방법은 EEG 신호에서 수동 주석 또는 시간 및 주파수 영역의 특징 추출에 의존하고 있었습니다. 본 연구는 웨이블릿 변환 기반의 시간-주파수 분석을 사용하여 이를 개선합니다.

- **Technical Details**: 제안된 방법은 Continuous Wavelet Transform (CWT)을 사용하여 수면 단계와 관련된 주파수 대역 간의 과도한(transient) 및 진동성(oscillatory) 패턴을 캡처하는 시간-주파수 맵을 생성합니다. 이를 통해 얻은 웨이블릿 기반 표현은 앙상블 학습(ensemble learning)과 결합되어 높은 분류 정확도를 달성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 88.37%의 전체 정확도와 73.15의 매크로 평균 F1 점수를 기록하여 기존 기계 학습 방법보다 뛰어난 성과를 보였습니다. 이는 최신 딥 러닝(deep learning) 접근법과 비교할 때도 유사하거나 우수한 성능을 나타내며, 수면 단계 분류의 임상적 적용 가능성을 강조합니다.



### MLLM4TS: Leveraging Vision and Multimodal Language Models for General Time-Series Analysis (https://arxiv.org/abs/2510.07513)
- **What's New**: 이번 연구에서는 MLLM4TS라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 멀티모달 대형 언어 모델(multimodal large language models)을 활용하여 시계열(time series) 분석을 수행합니다. 특히, 눈에 띄는 시각적 표현을 통해 시계열 데이터를 분석하는 자동화된 방법의 효과를 높일 수 있는지 탐구합니다.

- **Technical Details**: MLLM4TS는 시간을 나타내는 시리즈를 색상 코드가 있는 선 그래프(color-coded line plot) 형태로 변환하여 시각적 패턴을 캡처합니다. 이를 통해 각 채널 간의 공간 의존성을 효과적으로 파악하고, 각 시간 구간에 맞춰 시각적 패치를 정렬하는 전략을 도입합니다. 이러한 방법은 수치적 데이터와 시각적 표현에서 파생된 글로벌 맥락 정보를 융합하여 정교한 시간 역학을 모델링합니다.

- **Performance Highlights**: MLLM4TS는 분류(classification), 이상 탐지(anomaly detection), 예측(forecasting) 등 다양한 표준 벤치마크에서 좋은 성과를 보여주었습니다. 이 연구는 사전 훈련된 언어 모델(pretrained language models)과 시각적 모달리티(visual modalities)를 통합함으로써 시계열 분석의 강력하고 일반화 가능한 접근법을 제시합니다. 특히, 몇 샷(few-shot) 및 제로 샷(zero-shot) 학습 환경에서도 뛰어난 일반화 성능을 지니고 있습니다.



### When Thoughts Meet Facts: Reusable Reasoning for Long-Context LMs (https://arxiv.org/abs/2510.07499)
- **What's New**: 이 논문에서는 지식 집약적 멀티 홉 추론 작업을 위해 Thought Template Augmented LCLMs(ToTAL)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다양한 문서에서 증거를 수집하고 이를 조직적으로 연결하는 데 도움을 주는 재사용 가능한 사고 템플릿을 활용합니다. LCLM(Long-Context Language Models)의 발전으로 수백만 개의 토큰을 한 번에 처리할 수 있게 되었지만, 여전히 증거를 연결하는 방식에 대한 접근이 부족한 상황입니다.

- **Technical Details**: ToTAL은 기존의 RAG(Retrieval-Augmented Generation) 방식과는 달리, 사고 템플릿을 외부 매개변수로 간주하고 자연어 피드백을 통해 반복적으로 수정합니다. 이러한 템플릿은 이전 문제 해결의 패턴을 포괄하여 새로운 쿼리에 대한 중간 단계를 구조화할 수 있게 돕습니다. 이 과정은 특정 문제에 국한되지 않고 다양한 쿼리에서 재사용될 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 다양한 벤치마크와 LCLM 가족을 통해 ToTAL은 강력한 기준선 대비 일관된 성능 향상을 보여주었습니다. 특히, 정보 검색이 포함된 상황과 포함되지 않은 상황 모두에서 사고 템플릿이 LCLM 성능을 지속적으로 강화하는 것으로 나타났습니다. 이를 통해 LCLM이 복잡한 증거를 처리할 수 있는 새로운 가능성을 제시하며, 투명한 추론 재사용을 통한 광범위한 적용성을 보여줍니다.



### Can Speech LLMs Think while Listening? (https://arxiv.org/abs/2510.07497)
- **What's New**: 이 논문은 음성 대화 모델(Speech LLMs)의 추론 능력을 향상시키기 위한 체인 오브 쓰뜨(Chain-of-Thought, CoT) 미세 조정의 효과를 조사했습니다. 연구 결과, CoT fine-tuning을 통해 음성 LLM의 추론 정확도가 평균적으로 2.4배 향상되었습니다. 또한 모델이 사용자의 질문이 끝나기 전에 미리 사고를 시작하도록 유도하는 방법이 제안되었습니다.

- **Technical Details**: Moshi 모델은 음성 입력과 시스템의 텍스트를 포함하는 다중 스트림(multi-stream) 아키텍처를 채택하고 있습니다. 이 모델은 사용자 음성, 시스템 음성, 시스템 텍스트의 세 가지 개별 토큰 스트림을 동시에 처리하며, 이를 위해 Mimi라는 코덱 모델을 사용합니다. 미세 조정 방법으로는 사용자 질문의 완전성을 추정하는 새로운 메트릭과 선호 튜닝(preference tuning) 방식이 포함되어 있습니다.

- **Performance Highlights**: 이 연구는 CoT를 텍스트 기반으로 적용할 때 음성 LLM에서의 성능 개선을 보여주었습니다. 논문에서는 최적의 정확도-지연(trade-off)을 달성하기 위해 질문의 완전성 메트릭을 사용하여 70%의 지연을 감소시키면서도 정확도 손실이 없음을 입증했습니다. 이는 향상된 음성 LLM의 응용에 있어 중요한 진전을 의미합니다.



### A Denoising Framework for Real-World Ultra-Low Dose Lung CT Images Based on an Image Purification Strategy (https://arxiv.org/abs/2510.07492)
- **What's New**: 이 연구는 ultra-low dose CT (uLDCT) 의 비율을 줄임으로써 방사선 노출을 최소화하는 동시에, 이를 위한 새로운 denoising 프레임워크를 제안합니다. 특히, Image Purification (IP) 전략을 통해 uLDCT와 normal dose CT (NDCT) 간의 구조적 정렬을 개선하고, 성능을 대폭 향상시켰습니다.

- **Technical Details**: 연구팀은 실제 임상 uLDCT 폐 데이터셋을 구축하고, IP 전략을 사용해 uLDCT-NDCT 이미지 쌍의 정렬을 최적화했습니다. 이를 기반으로 Frequency-domain Flow Matching (FFM) 모델을 제안하여, anatomical structure의 무결성을 보존하며 denoising을 수행합니다. 이러한 접근법은 기존의 방식들과 비교해 새로운 차원의 성능 향상을 가져옵니다.

- **Performance Highlights**: 실험 결과, IP 전략은 여러 주류 denoising 모델의 성능을 크게 향상시키는 것으로 나타났습니다. 특히, IP 전략과 FFM 모델의 결합에 의해 구조 보존에서 state-of-the-art (SOTA) 성능을 달성했습니다. 이는 실제 임상에서의 uLDCT denoising 데이터 불일치 문제를 해결하는 효과적인 방법을 제시합니다.



### Can Lessons From Human Teams Be Applied to Multi-Agent Systems? The Role of Structure, Diversity, and Interaction Dynamics (https://arxiv.org/abs/2510.07488)
Comments:
          Under Review at ARR

- **What's New**: 이번 연구는 대규모 언어 모델(LLM)을 이용한 다중 에이전트 시스템(MAS)의 팀 동역학을 탐구하는 새로운 접근 방식을 제안합니다. 팀 과학에서 영감을 받아 팀의 구조, 다양성 및 상호작용 동향을 조사하는 프레임워크를 개발하였으며, 총 4개의 과제를 통해 팀의 성과를 평가합니다. 연구 결과, 전반적으로 수평적인 팀 구조가 계층적인 구조보다 나은 성과를 보이며, 다양성의 영향이 복잡하게 나타난다고 밝혔습니다.

- **Technical Details**: 연구는 LLM 에이전트를 사용하여 수평 및 계층 팀을 시뮬레이션하고, 각 에이전트에게 인구통계적 페르소나를 할당하였습니다. 평가에는 CommonsenseQA, StrategyQA, Social IQa, Latent Implicit Hate Detection의 4개 과제가 포함되며, 이 과제들은 세밀한 추론 및 가치 판단을 요구합니다. 이러한 구조와 다양성이 에이전트의 상호작용과 추론에 미치는 영향을 분석하기 위해 ‘LLM-as-a-judge’ 접근법을 이용하였습니다.

- **Performance Highlights**: 연구 결과는 팀 구조 및 구성 요소가 추론 및 사회적 추론에 미치는 영향을 강조하며, 이러한 차원이 에이전트의 상호작용 및 조율 방식에 중대한 영향을 미친다는 점을 제시하였습니다. 특히, 차별화된 팀 구조는 효과적인 협력을 증진시키고, 신뢰 구축에 중요한 역할을 합니다. 나아가, 연구는 MAS 설계를 위한 이론적 시사점을 제공하며, 인간-AI 협업에서 의사소통 구조와 사회적 프레이밍의 매개 역할을 강조합니다.



### HEMERA: A Human-Explainable Transformer Model for Estimating Lung Cancer Risk using GWAS Data (https://arxiv.org/abs/2510.07477)
Comments:
          18 pages, 6 figures, 3 tables

- **What's New**: 본 연구는 HEMERA(인간 설명 가능한 변환기 모델)를 소개하며, 이는 단일 뉴클레오타이드 다형성(SNP) 데이터의 유전체 전체 연관 연구(GWAS)에서 폐암 위험을 예측하기 위해 설명 가능한 변환기 기반의 딥러닝을 적용합니다. HEMERA는 기존 방법론과 달리 임상 공동 변수를 포함하지 않고 원시 유전자형 데이터를 직접 처리하며, 이는 개인화된 위험 평가와 조기 개입을 위해 중요한 혁신을 제공합니다.

- **Technical Details**: HEMERA는 유전체 연관 데이터에서 중요한 건축적 혁신을 실현하며, 원시 유전자형 데이터에서 직접 유전적 변이를 분리하여 그 예측 기여를 정량화할 수 있습니다. 이 모델은 ADDITIVE positional encoding과 신경망 임베딩 레이어를 도입하여 유전적 변이가 학습 가능한 표현을 갖도록 구성됩니다. 이 연구는 또한 LAYER-WISE INTEGRATED GRADIENTS 기반의 포스트 호크 설명 가능성 모듈을 통합하여 특정 SNP에 대한 예측 결과의 귀속을 가능하게 합니다.

- **Performance Highlights**: 27,254명의 참가자로부터 수집된 데이터를 기반으로 HEMERA는 99% 이상의 AUC(수신자 조작 특성 아래 영역) 점수를 달성하였습니다. 이러한 결과는 HEMERA가 정확한 위험 분류와 세밀한 기능 귀속을 제공해 폐암 취약성의 유전적 결정을 이해하는 데 큰 기여를 할 수 있음을 보여줍니다. HEMERA는 설명 가능한 모델로 향상된 예측 성능을 통해 유전적 위험을 밝히고, 이를 통해 폐암의 조기 발견을 위한 새로운 가능성의 길을 열고 있습니다.



### MoGU: Mixture-of-Gaussians with Uncertainty-based Gating for Time Series Forecasting (https://arxiv.org/abs/2510.07459)
- **What's New**: 이번 연구에서 우리는 Mixture-of-Gaussians with Uncertainty-based Gating (MoGU)를 소개하며, 이는 회귀 작업에 특화된 새로운 Mixture-of-Experts (MoE) 프레임워크입니다. MoGU는 각 전문가의 출력을 가우시안 분포로 모델링하여 예측값의 평균뿐만 아니라 그 내재적 불확실성(variance)도 직접적으로 정량화할 수 있습니다. 기존의 MoE와 차별화되는 점은 불확실성 기반의 게이팅 메커니즘을 적용하여 각 전문가의 기여도를 결정한다는 것입니다.

- **Technical Details**: MoGU는 각각의 전문가가 예측의 평균과 분산을 동시에 예측하는 구조로 되어 있습니다. 이러한 방식을 통해 전문가의 행동에 대한 정의로운 이해와 전체 모델의 불확실성을 도출할 수 있습니다. MoGU의 게이팅 메커니즘은 각 전문가의 추정된 불확실성을 기반으로 하여 최종 MoE 예측에 대한 기여도를 동적으로 결정하는 방식으로, 기존의 입력 기반 게이팅 메커니즘을 대체합니다.

- **Performance Highlights**: MoGU는 다양한 시간 시계열 예측 벤치마크에서 일관되게 더욱 정확한 예측을 제공하며, 이는 입력 기반 게이팅 MoE 아키텍처와 비교했을 때 두드러집니다. 예측 오차와 긍정적인 상관관계를 가진 불확실성 추정치를 제공함으로써 모델의 신뢰성과 그 출처에 대한 통찰력을 강화합니다. MoGU는 개별 전문가와 전체 모델 수준에서 불확실성을 잘 정량화하며, 이에 따라 예측 결과의 신뢰도를 향상시킵니다.



### Minimizing the Value-at-Risk of Loan Portfolio via Deep Neural Networks (https://arxiv.org/abs/2510.07444)
- **What's New**: 본 논문은 Peer-to-Peer (P2P) 대출 분야에서 리스크 관리의 중요성을 다루고 있습니다. 특히, 투자자가 여러 대출에 자금을 분산시키며 리스크 노출을 줄일 수 있는 방법에 대해 설명합니다. 이 연구에서는 Value-at-Risk (VaR)와 Conditional Value-at-Risk (CVaR)를 최소화하는 것을 목표로 하는 두 가지 딥 뉴럴 네트워크 모델, DeNN과 DSNN을 제안합니다.

- **Technical Details**: DeNN은 저자유도(low degree of freedom) 모델이며, DSNN은 고자유도(high degree of freedom) 모델입니다. 두 모델 모두 대출의 디폴트 확률 뿐만 아니라 디폴트 시점을 예측할 수 있는 기능을 갖추고 있습니다. 실험 결과, 두 모델 모두 기준선(benchmarks)에 비해 포트폴리오 VaR를 현저히 감소시킬 수 있는 것으로 나타났습니다.

- **Performance Highlights**: DeNN 모델은 다양한 신뢰 수준(confidence levels)에서 DSNN 모델보다 더 나은 성능을 보였습니다. 대부분의 시나리오에서 DeNN이 우수한 결과를 도출한 것은 매우 흥미로운 결과입니다. 이는 저자유도 모델이 고자유도 모델에 비해 더 효과적일 수 있음을 시사합니다.



### LASER: An LLM-based ASR Scoring and Evaluation Rubric (https://arxiv.org/abs/2510.07437)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 본 연구에서는 기존의 자동 음성 인식(ASR) 평가 지표인 Word Error Rate(WER)의 한계를 극복하기 위한 혁신적인 방법으로 LASER라는 LLM 기반의 스코어링 루브릭을 소개합니다. LASER는 ASR 시스템의 오류를 보다 정교하게 평가할 수 있는 점수 매기기 방법론으로, 인도 언어의 형태적 및 구문적 특징을 잘 반영합니다. 특히, 이 시스템은 힌디어에서 사람의 주석과 94%의 높은 상관관계를 달성하여 다른 인도 언어에서도 효과적으로 적용이 가능함을 증명하였습니다.

- **Technical Details**: LASER는 표준 ASR 지표에서 발생하는 오류 유형을 분석하여, 수정된 패널티를 부여하는 방법을 제안합니다. minor grammatical errors에는 낮은 패널티를 부여하고, semantic errors에는 높은 패널티를 부여하는 방식입니다. LASER 점수는 맞춤형 프롬프트를 통해 구체적인 지시와 오류 유형을 설정해 LLM에게 처리하게 함으로써 생성됩니다. 이 과정에서 LLM은 주어진 문장을 토큰화하고, 기준 참조와 예측을 정렬하여 오류를 분류합니다.

- **Performance Highlights**: Gemini 2.5 Pro는 모든 LLM 중에서 뛰어난 성능을 보여주며, 사람의 점수와의 상관관계가 가장 높았습니다. 연구 결과, LASER 지수는 전통적인 WER과 비교하여 상대적으로 높은 정확성을 보였고, ASR 예측의 정확성을 판단하는데 있어 훨씬 더 신뢰할 수 있는 방법임을 입증했습니다. 실험을 통해 LASER의 점수가 사람의 판단과 잘 일치함을 확인하면서, 다양한 언어에서의 적용 가능성을 보여주었습니다.



### Haystack Engineering: Context Engineering for Heterogeneous and Agentic Long-Context Evaluation (https://arxiv.org/abs/2510.07414)
Comments:
          Code available at this https URL

- **What's New**: 이 논문은 HaystackCraft라는 새로운 NIAH 벤치마크를 소개하며, LLM의 긴 문맥에서의 강인함(long-context robustness)을 평가하기 위해 노이즈가 있는 긴 문맥을 구축하는 것이 중요하다고 주장합니다. 연구팀은 다수의 다단계 질문을 통해 영어 위키피디아의 하이퍼링크 네트워크를 기반으로 한 테스팅 환경을 설계했습니다. 이 새로운 평가 기준은 다양한 검색 전략이 LLM의 성능에 미치는 영향을 체계적으로 검토합니다.

- **Technical Details**: 이번 연구에서는 검색 증강 생성(RAG) 기술을 활용하여 LLM의 긴 문맥을 조작하는 방식을 다루며, 상이한 검색 자원들이 자료의 노이즈 및 복잡성을 어떻게 생성하는지를 탐구합니다. 또한, 그래프 기반의 검색 방식의 구현이 LLM의 성능 개선에 어떻게 기여할 수 있는지에 대한 실험을 수행하였습니다. 다양한 Retrieval 전략(예: sparse, dense, hybrid, graph-based)을 기반으로 하여, HaystackCraft는 에이전트의 작업 흐름 중 나타나는 누적 오류(cascading failures)에 대한 모델의 저항성을 평가합니다.

- **Performance Highlights**: 실험 결과, 강력한 밀집 검색기(dense retrievers)는 더 어려운 산만 요소(distractors)를 도입하는 반면, 그래프 기반의 재정렬(graph-based reranking)은 검색의 효과를 높이며 해로운 산만 요소를 줄이는 데 기여했습니다. 15개의 긴 문맥 LLM을 대상으로 한 테스트에서는 Gemini 2.5 Pro와 GPT-5와 같은 고급 모델도 누적 자기 산만 문제로 어려움을 겪는 것으로 나타났습니다. 이 결과는 에이전트의 긴 문맥 추론에 지속적인 도전이 남아 있음을 나타내며, HaystackCraft가 향후 발전을 측정하기 위한 중요한 시험대임을 강조합니다.



### Quantum Grid Path Planning Using Parallel QAOA Circuits Based on Minimum Energy Princip (https://arxiv.org/abs/2510.07413)
- **What's New**: 이 연구는 기존의 고전적 경로 계획 방식이 가지는 한계를 극복하고, Noisy Intermediate-Scale Quantum (NISQ) 시대의 양자 경로 계획 프레임워크의 문제를 해결하기 위한 새로운 접근법을 제안합니다. 이 연구는 Quantum Approximate Optimization Algorithm (QAOA) 아키텍처를 기반으로 한 양자 경로 계획 솔루션을 구성하여, 그리드 경로 계획 문제를 최소 양자 에너지 상태를 찾는 문제로 매핑합니다. 이를 통해 두 개의 병렬 QAOA 회로를 구축하여 연결 에너지 계산과 경로 에너지 계산을 동시에 수행하고, 불합리한 솔루션을 필터링하는 고전적 알고리즘을 활용하여 최적해에 근접한 경로 계획 문제의 해를 도출합니다.

- **Technical Details**: 기술적으로, 이 연구는 두 개의 병렬 QAOA 회로를 사용하여 두 가지 해결 프로세스를 동시에 수행합니다. 연결성 에너지 계산과 경로 에너지 계산이 포함되며, 두 프로세스의 계산 결과를 병합하고 최적 솔루션을 도출하는 방식입니다. 이 접근법은 양자 상태와 필터 파라미터를 효과적으로 활용하여 가능한 최적의 경로를 찾는 데 도움을 주며, 회로의 레이어 수가 1일 때도 최적 경로 코딩 조합을 찾아낼 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 병렬 회로는 직렬 회로에 비해 경로 계획 문제를 해결하는 데 있어 높은 성공 확률을 보일 뿐만 아니라, 실제로 더 낮은 발생 확률을 가진 양자 상태를 필터링하여 목표 양자 상태를 확보하는 데 효과적임을 입증하였습니다. 이는 기존 고전적 알고리즘과 비교하여 본 연구의 접근법이 더 높은 효율성을 제공함을 나타냅니다. 최적 경로 코딩 조합을 찾을 때의 확률이 높아지는 점도 주목할 만합니다.



### Attention to Order: Transformers Discover Phase Transitions via Learnability (https://arxiv.org/abs/2510.07401)
- **What's New**: 이 논문에서는 집합 행동의 질적 재편성을 나타내는 Phase transitions을 다루며, 이러한 경계를 식별하는 데 있어 새로운 방법론으로 learnability(학습 가능성)를 도입합니다. 이는 Transformer 모델이 microscopic states(미세 상태)로부터 구조를 추출할 수 있는 능력으로 정의됩니다. 전통적인 시뮬레이션 방법이 실패하는 상황에서 learnability가 Phase transitions의 데이터를 기반으로 한 지표 역할을 한다는 점이 새롭습니다.

- **Technical Details**: 논문에서는 self-supervised learning(자기 감독 학습)과 Monte Carlo(몬테 카를로)로 생성된 2차원 Ising model(아이징 모델)의 구성을 사용합니다. 연구 결과에 따르면, ordered phases(정렬된 상)에서 learnability가 향상되는 것이 관찰되며, 이는 훈련 손실(training loss)의 감소와 구조화된 주의 패턴(structured attention patterns)으로 나타납니다. 반면, disordered phases(무질서 상)는 학습에 저항적입니다.

- **Performance Highlights**: 훈련 손실의 급격한 점프와 주의 엔트로피(attention entropy)의 상승이라는 두 개의 비지도 진단을 통해 임계 온도(critical temperature)를 매우 정확하게 복구했습니다. 이러한 성과는 응집 물질에서의 장기적 질서(long-range order)와 현대 언어 모델에서의 구조의 출현(appearance of structure)之间의 깊은 유사성을 강조합니다.



### Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts (https://arxiv.org/abs/2510.07358)
- **What's New**: 본 논문에서는 언어 모델의 추론 능력을 향상시키기 위한 새로운 방법인 Encode-Think-Decode (ETD)를 제안합니다. ETD는 모델의 기본 구조와 매개변수 수를 보존하면서도 미리 훈련된 모델이 특정 층에서 반복적으로 추론을 할 수 있도록 훈련시킵니다. 이 방법은 모델의 추론 성능을 크게 개선할 수 있는 간단하고 효과적인 경로를 제공합니다.

- **Technical Details**: ETD 방식은 모델이 인퍼런스 단계에서 선택된 층들을 반복적으로 활용하여 동작하도록 합니다. 이를 통해 기존의 LLM(large language models)의 잠재적인 추론 능력을 개선할 수 있습니다. 이전 연구들은 서로 다른 층에서의 정보 흐름이 추론 과정에서 중요한 역할을 한다고 밝히고 있으며, 본 연구는 이러한 사실을 바탕으로 층의 역할을 고려하여 ETD 구조를 설계하였습니다.

- **Performance Highlights**: 본 연구에서는 OLMo-2 1B 베이스 모델을 사용하여 17개의 추론 벤치마크에서 성능을 평가한 결과, GSM8K에서 28.4%의 상대적 정확도 향상과 MATH에서 36%의 향상을 달성했습니다. 이는 복잡한 구조의 변경 없이도 기존 모델의 성능을 높일 수 있는 가능성을 보여줍니다. 또한, 입력 토큰에 따라 동적으로 계산 깊이를 조정하는 방법을 제안하여 보다 효율적인 추론이 가능해졌습니다.



### Mitigating Surgical Data Imbalance with Dual-Prediction Video Diffusion Mod (https://arxiv.org/abs/2510.07345)
Comments:
          29 pages, 16 figures

- **What's New**: 이 논문에서는 $SurgiFlowVid$라는 새로운 비디오 생성 프레임워크를 제안합니다. 이 프레임워크는 희소한 클래스에 대한 외과 비디오를 생성할 수 있도록 설계되었으며, 클래스 불균형 문제를 해결하는 데 중점을 두고 있습니다. 두 가지 예측 모듈을 통해 RGB 프레임과 광학 흐름을 동시에 평활화하여 제한된 표본에서 모션 모델링을 향상시키는 것이 특징입니다.

- **Technical Details**: $SurgiFlowVid$는 듀얼 예측(diffusion) 방법을 사용하여 RGB 프레임과 광학 흐름 지도를 공동으로 처리합니다. 이러한 접근 방식은 최소한의 비디오 샘플에서도 시공간 관계를 포착할 수 있습니다. 또한, 희소한 조건적 프레임을 활용해 비디오 생성의 컨트롤 가능성을 개선하여 비싼 세부 주석 없이도 적용할 수 있도록 합니다.

- **Performance Highlights**: 세 가지 외과 데이터셋에서 수행한 평가 결과, $SurgiFlowVid$를 통해 생성된 합성 데이터는 경쟁 모델에 비해 일관되게 10-20% 향상을 보여주었습니다. 이는 외과 비디오 이해를 위한 강력한 딥러닝(Deep Learning) 모델을 향상시키는 데 기여하며, 수술 의료 분야에서도 중요한 발전을 가져올 수 있음을 시사합니다.



### Local MAP Sampling for Diffusion Models (https://arxiv.org/abs/2510.07343)
- **What's New**: 이번 논문은 Local MAP Sampling (LMAPS)라는 새로운 추론 프레임워크를 소개합니다. LMAPS는 역 문제 해결 시 최적화 기반 방법의 원리에 대한 명확한 해석을 제공하며, 기존의 Diffusion Posterior Sampling (DPS)와의 연관성을 설명합니다. 이로 인해 LMAPS는 다양한 최적화 기반 방법론을 통합할 수 있는 가능성을 갖게 됩니다.

- **Technical Details**: LMAPS는 확률적으로 해석 가능한 공분산 근사치를 사용하여 역 문제를 해결합니다. 기존 솔버의 비합리적인 선택을 대체하는 방법으로, 해석 가능한 파라미터를 위한 목표 재구성 및 비미분 가능 연산자에 대한 그래디언트 근사 전략을 개발하였습니다. 이 새로운 접근법은 각 단계에서 지역 최대 후행 확률을 반복적으로 해결하여 성능을 향상시킵니다.

- **Performance Highlights**: LMAPS는 10가지 이미지 복원 작업과 3가지 과학적 역 문제에서 검증되었으며, 46/60 FFHQ/ImageNet 사례에서 2 dB 이상의 PSNR 향상을 달성했습니다. 이미지 모션 블러 제거, JPEG 복원 및 양자화에서 다수의 개선을 포함하여 과학적 작업에서도 일관되게 높은 PSNR을 기록했습니다. 또한, LMAPS는 기존 DPS보다 더 효율적으로 성능을 발휘합니다.



### MultiFair: Multimodal Balanced Fairness-Aware Medical Classification with Dual-Level Gradient Modulation (https://arxiv.org/abs/2510.07328)
Comments:
          10 Pages

- **What's New**: 이 논문은 의료 분류를 위한 새로운 접근 방식인 MultiFair를 제안하며, 이는 다양한 데이터를 동시에 다루는 과정에서 생길 수 있는 공정성과 비균형 문제를 해결합니다. 기존의 다중 모달 학습 모델들이 두 가지 주요 문제, 즉 모달리티 학습 불균형(Modality Learning Bias)과 인구 통계학적 학습 불균형(Demographic Learning Bias)을 간과하고 있음을 지적합니다. MultiFair는 이러한 문제를 두 가지 층의 그래디언트 조절(Dual-level Gradient Modulation) 프로세스를 통해 해결합니다.

- **Technical Details**: MultiFair 모델은 훈련 그래디언트를 모달리티와 그룹 수준에서 최적화합니다. 이 모델은 각 모달리티의 기여도를 동적으로 조정하여, 전반적인 배치(Training Batch)에서 발생할 수 있는 불균형한 학습을 완화합니다. 논문에서는 MultiFair의 이론적 기반과 함께 실제 의료 다중 모달 데이터 셋을 활용한 광범위한 실험 결과도 제공합니다.

- **Performance Highlights**: 실험 결과, MultiFair는 기존의 최신 다중 모달 학습 및 공정성 학습(Fairness Learning) 방법들을 초월하며 성능을 보였습니다. 특히, 다양한 인구 통계 그룹에 대한 성능을 균형적으로 유지하면서도 진단의 정확도를 높이는 특징을 보여줍니다. 이는 의료 AI의 공정성을 확보하는데 기여할 것으로 기대됩니다.



### Deep Learning Based Approach to Enhanced Recognition of Emotions and Behavioral Patterns of Autistic Children (https://arxiv.org/abs/2510.07320)
- **What's New**: 이번 연구에서는 자폐 스펙트럼 장애(ASD) 아동의 정서 인식과 행동 패턴을 중심으로 한 맞춤형 교육 전략의 필요성을 강조합니다. 특히, ASD 아동의 감정 상태를 정확히 인식하는 것이 맞춤형 개입 및 사회적 지원을 위한 기초가 됨을 설명하며, 최신 AI 기술인 autoencoder를 통해 감정 인식의 정확성을 향상시키는 방법을 제안합니다. 이를 통해 ASH 교육과 기술 지원을 위한 발달 경로를 더욱 원활하게 할 수 있도록 합니다.

- **Technical Details**: 연구 방법론에서는 Xception 및 InceptionV3 모델을 활용하여 ASD 아동의 정서 인식을 위한 데이터 전처리에 대한 접근 방식을 상세히 설명합니다. 특히, 각기 다른 크기의 이미지 입력을 299×299×3의 고정된 크기로 변환하기 위해 autoencoder를 사용하여 노이즈를 줄이고 필수적인 안면 특징을 보존합니다. 깊이별 분리 가능한 합성곱(depthwise separable convolutions) 기술을 사용하여 모델의 계산 복잡성을 줄이고, 다중 스케일의 특징 추출을 통해 다양한 정서 표현을 인식합니다.

- **Performance Highlights**: 연구 결과, autoencoder를 통합한 정서 인식 시스템이 전통적인 방법론에 비해 더 뛰어난 성능을 보여주었습니다. baseline 방법과 비교했을 때, 향상된 접근 방식은 정확도, 정밀도, 재현율, F1-score에서 유의미한 개선을 보였으며, 이를 통해 ASD 아동의 감정 인식 정확성을 높이고자 하는 목적을 달성했습니다. 또한, 두 단계의 훈련 전략을 통해 모델을 더욱 효과적으로 최적화함으로써, 복잡한 실제 환경에서도 우수한 성능을 발휘하는 것을 입증했습니다.



### DUA-D2C: Dynamic Uncertainty Aware Method for Overfitting Remediation in Deep Learning (https://arxiv.org/abs/2411.15876)
Comments:
          This version (v2) extends our previous work (arXiv:2411.15876v1) on Divide2Conquer (D2C) by introducing Dynamic Uncertainty-Aware Divide2Conquer (DUA-D2C). The manuscript is currently under review at Complex and Intelligent Systems

- **What's New**: 이번 연구에서는 Dynamic Uncertainty-Aware Divide2Conquer (DUA-D2C) 기법을 소개합니다. DUA-D2C는 기존의 Divide2Conquer (D2C) 방식을 기반으로 하여, 모델들이 보유한 성능에 따라 유연하게 가중치를 부여하는 새로운 집계 방식을 도입합니다. 이러한 접근 방식은 데이터의 아웃라이어(Outlier)와 잡음(Noise)의 영향을 최소화하면서도 더 일반화된 패턴을 학습하는 데 기여합니다.

- **Technical Details**: DUA-D2C는 각 모델의 성능을 공유 검증 세트(Validation Set)에서 평가하여 가중치를 할당합니다. 이 방법은 예측 정확도(Accuracy)와 예측 불확실성(Prediction Uncertainty)을 고려하여, 덜 신뢰할 수 있는 모델의 영향을 줄이는 데 초점을 맞춥니다. 연구에서는 수학적 정당성을 바탕으로 D2C가 과적합(Overfitting)을 감소시킨다는 점을 강조하며, DUA-D2C의 집계 과정이 더욱 정교해졌음을 설명합니다.

- **Performance Highlights**: 실험을 통해 DUA-D2C는 다양한 데이터 세트에서 모델의 일반화 성능을 크게 향상시킴을 입증하였습니다. 특히, DUA-D2C는 기존의 데이터 증대(Data Augmentation) 및 정규화(Regularization) 기법과 결합하여 성능 개선에 기여할 수 있는 가능성을 보여줍니다. 연구 결과는 모델이 더욱 신뢰할 수 있는 의사 결정을 내릴 수 있도록 돕는다는 점에서 중요한 의미를 가집니다.



New uploads on arXiv(cs.LG)

### Who Said Neural Networks Aren't Linear? (https://arxiv.org/abs/2510.08570)
- **What's New**: 이 논문은 비선형 함수가 특정 비표준 벡터 공간에서 선형적으로 해석될 수 있는 방법을 제안합니다. 특히, 선형 연산자 A를 두 개의 가역 신경망으로 샌드위치하여 새로운 벡터 공간을 정의하는 "Linearizer" 아키텍처를 소개합니다. 이 접근법으로 선형 대수의 다양한 기법을 비선형 매핑에 적용할 수 있습니다. 이를 통해 수백 개의 샘플링 단계를 한 단계로 축소시키는 방법과 같은 강력한 응용 가능성도 제시합니다.

- **Technical Details**: Linearizer 프레임워크는 선형성과 비선형성이 상대적 개념이라는 점을 강조합니다. 구체적으로, 가역 신경망 g_x와 g_y 사이에 선형 연산자 A를 배치하여, 새롭게 정의된 덧셈과 스케일링 연산을 통해 벡터 공간의 구조를 유도합니다. 이와 같은 구조화된 접근을 통해 비트 연산자와 선형 연산자가 조화를 이룰 수 있게 됩니다. 또한, Linearizer는 중첩할 수 있어, 공유된 신경망을 사용하는 경우에도 선형성을 유지합니다.

- **Performance Highlights**: 이 연구의 결과는 Linearizer가 비선형 신경망의 분석과 조작에 있어 선형성의 장점을 활용할 수 있음을 보여줍니다. 예를 들어, 수백 개의 샘플링 스텝을 하나로 압축하여 효율적인 훈련 과정을 달성할 수 있습니다. 또한, idempotency(멱등성) 속성을 네트워크에 적용하거나 모듈형 스타일 변환을 위한 생성 모델을 구현하는 등의 응용도 가능하다는 점에서 Linearizers는 비선형 모델의 유연성 및 해석 가능성을 제공하는 매력적인 프레임워크입니다.



### Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization (https://arxiv.org/abs/2510.08554)
- **What's New**: 본 연구에서는 Group Diffusion Policy Optimization (GDPO)라는 새로운 강화 학습 알고리즘을 도입합니다. GDPO는 디퓨전 언어 모델(DLMs)을 대상으로 하고, 이 모델에 대한 강화 학습의 적용 가능성을 높이기 위해 sequence-level ELBO의 분산을 분석합니다. 이 분해를 통해 얻은 통찰을 바탕으로, 적은 계산 비용으로도 효과적인 변동성 감소를 달성할 수 있는 접근 방식을 제안합니다.

- **Technical Details**: GDPO는 Semi-deterministic Monte Carlo 기법을 활용하여 기존 ELBO 추정기의 변동성을 줄이는 방법을 모색합니다. 이로 인해, GDPO는 복잡한 추론 작업을 해결하면서도 적은 계산 리소스를 활용해 낮은 변동성을 보장합니다. 그 과정에서, GDPO는 디퓨전 모델에서의 토큰 레벨과 시퀀스 레벨의 likelihood 예측의 복잡성을 극복하려는 노력을 포함합니다.

- **Performance Highlights**: GDPO는 다양한 수학, 추론 및 코딩 벤치마크에서 사전 훈련된 체크포인트에 대해 일관된 성능 향상을 이루어냈습니다. 특히, GDPO는 최신 연구 중 하나인 diffu-GRPO를 능가하며, DLMs에 적합한 새로운 성능 기준을 제시합니다. 이러한 성과는 DGPO의 효과적인 변동성 관리와 강화 학습 방법론의 혁신적인 발전에 기인합니다.



### Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints (https://arxiv.org/abs/2510.08549)
- **What's New**: ERA(Entropy Regularizing Activation)를 소개하여 출력 활성화 함수를 활용한 새로운 엔트로피 제어 패러다임을 제시합니다. 이 접근법은 기존 알고리즘과 비침습적으로 통합될 수 있어 다양한 도메인에서 성능 향상을 보여줍니다. 이 연구는 LLM(large language models), 강화 학습 에이전트, 이미지 분류 등 여러 분야에서 효과적임을 입증하였습니다.

- **Technical Details**: ERA는 모델의 최종 출력에 적용되는 특별히 설계된 활성화 함수를 통해 엔트로피 제약을 impose합니다. 이 기법은 기본 최적화 목표와 엔트로피 제약을 완전히 분리하여 손실 함수가 원래 목표에만 집중할 수 있도록 합니다. 기존 방법들이 손실 함수의 직접 수정을 피하는 반면, ERA는 이론적으로 엔트로피 보장을 제공하는 정량적 모델입니다.

- **Performance Highlights**: ERA를 적용한 결과, 여러 분야에서 성능 향상이 관찰되었습니다. 예를 들어, LLM에서는 Qwen-2.5-Math-7B 모델이 AIME-24에서 9.0%, AIME-25에서 37.4% 향상되었습니다. 이미지 분류에서는 ResNet-50의 ImageNet top-1 정확도가 0.69% 향상되었으며, 강화 학습에서는 HumanoidBench 등의 복잡한 문제에서 30% 이상의 성능 향상이 있었습니다.



### On the optimization dynamics of RLVR: Gradient gap and step size thresholds (https://arxiv.org/abs/2510.08539)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 후처리에서 강화 학습(RL) 기반 접근법인 RLVR(Reinforcement Learning with Verifiable Rewards)가 효과를 보고하고 있습니다. RLVR은 간단한 이진 피드백을 사용하여 LLM을 미세 조정하는 강력한 방법으로, 성공 여부를 자동으로 확인할 수 있는 작업에 적합합니다. 그러나 RLVR의 이론적인 기초는 부족하여, 본 논문은 이 빈틈을 메우고자 RLVR의 교육 과정을 정량적으로 분석했습니다.

- **Technical Details**: 논문에서 제안하는 주된 개념은 Gradient Gap으로, 이는 낮은 보상 지역에서 높은 보상 지역으로의 개선 방향을 형식적으로 정의합니다. RLVR의 수렴성은 이 Gradient Gap에 따라 결정되어, 특정 임계 값 아래에서는 안정적인 학습이 이루어지고, 그 이상에서는 성능이 무너집니다. 이는 출력의 길이와 작업의 난이도에 따라 효과적인 학습 속도가 어떻게 조정되어야 하는지를 설명합니다.

- **Performance Highlights**: 이론적 설명을 바탕으로, 연구진은 조절된 밴디트 시뮬레이션과 LLM을 활용한 실험을 통해 이론의 정확성을 검증했습니다. 예를 들어, GRPO를 사용하여 Qwen2.5-7B 모델을 GSM8K와 DAPO17k 데이터셋으로 미세 조정하였으며, 이론과 실제 성과 간의 밀접한 일치를 보여주었습니다. 이와 같은 연구 결과는 RLVR의 안정적인 수렴과 성공적인 활용을 위한 명확한 지침을 제시합니다.



### Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning (https://arxiv.org/abs/2510.08526)
Comments:
          Accepted to NeurIPS 2025. First two authors contributed equally

- **What's New**: 본 논문은 강화 학습(Reinforcement Learning, RL)에서 최적 정책(optimal policy)의 수렴을 보장하는 이론적 틀을 제시합니다. 이 틀은 소실되는 엔트로피 정규화(vanishing entropy regularization) 및 온도 이탈 기법(temperature decoupling gambit)을 통해 이루어집니다. 이를 통해 최적 정책의 해석 가능성과 다양성 유지(diversity-preserving)를 실현할 수 있습니다.

- **Technical Details**: 이 논문에서는 일반적인 마르코프 결정 과정(Markov Decision Process, MDP)에서 최적 정책의 비유일성 문제를 해결하기 위한 접근법을 설명합니다. 정규화의 온도(temperature) τ에 따라 정책, 가치 함수(value functions), 수익 분포(return distributions)의 수렴을 보장하는 온도 이탈 기법을 소개합니다. 이를 통해 안정적으로 최적 정책을 찾아가는 알고리즘을 개발하고, 최적 행동의 다양성을 보장합니다.

- **Performance Highlights**: 저자들은 온도 이탈 기법을 사용하여 기대 수익이 동일하더라도 다양한 정책들이 갖는 수익 분포를 이해하는 중요성을 강조합니다. 특히, 이 기법은 안전-critical 응용 분야에서_return distributions_의 분포를 정확하게 추정할 수 있는 알고리즘을 제시합니다. 결과적으로 이 알고리즘은 해석 가능하고 다양한 정책의 수익 분포를 정확하게 추정할 수 있음을 보여줍니다.



### DYNAMIX: RL-based Adaptive Batch Size Optimization in Distributed Machine Learning Systems (https://arxiv.org/abs/2510.08522)
- **What's New**: DYNAMIX는 분산 머신러닝(Distributed Machine Learning) 환경에서 배치 크기 최적화를 위해 강화 학습(Reinforcement Learning)을 사용하는 새로운 프레임워크입니다. 이 방법은 Proximal Policy Optimization (PPO)을 활용하여 비동적, 이질적 컴퓨팅 환경에 능동적으로 배치 크기를 조정하게 합니다. DYNAMIX는 네트워크 지표와 시스템 자원 이용률을 포함한 다차원 상태 표현을 통해 효과적인 의사 결정을 할 수 있도록 지원합니다.

- **Technical Details**: DYNAMIX의 주요 기능은 다차원 상태 표현을 사용하여 네트워크와 시스템의 메트릭, 트레이닝 통계적 효율 지표 등을 통합한다는 점입니다. 이를 통해 DYNAMIX는 효과적인 의사 결정을 할 수 있도록 하며, 명시적인 시스템 모델링 없이도 기존 분산 훈련 프레임워크와 원활하게 통합됩니다. DYNAMIX는 동적 경로 최적화를 통해 기계 학습 프로세스를 개선하여, 자원의 효율성과 모델 품질 간의 상충 관계를 해결합니다.

- **Performance Highlights**: DYNAMIX는 기존의 정적 방법 대비 훈련 시간을 46% 단축하고, 최종 모델의 정확도를 6.3% 향상시킵니다. 32노드 환경에서 DYNAMIX는 92.6%의 정확도를 달성하며, 이는 정적 접근 방식의 81.3%에 비해 월등히 높은 성능을 보여줍니다. 또한 학습된 정책이 모델 아키텍처 전반에 걸쳐 효과적으로 일반화된다는 것을 입증하였으며, 다양한 하드웨어 구성에서도 일관된 성능을 유지하는 것이 확인되었습니다.



### Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models (https://arxiv.org/abs/2510.08492)
Comments:
          63 pages, 29 tables, and 47 figures

- **What's New**: UML (Unpaired Multimodal Learner)이라는 새로운 훈련 패러다임을 소개합니다. 이 모델은 서로 다른 모달리티의 입력을 처리하며, 매개변수를 공유하여 모달리티에 독립적인 특징을 학습할 수 있도록 설계되었습니다. 서로 다른 모달리티가 공통의 잠재 현실를 투영하고 있다는 가정을 바탕으로, 일치된 데이터 없이도 상호 모달리티 구조의 이점을 누릴 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 UML이 비슷한 구조의 입력을 처리할 때 매개변수 공유를 통해 서로 다른 모달리티의 정보를 자연스럽게 전이할 수 있도록 하는 방법을 수립했습니다. 이론적으로, 선형 가정 하에서 비일치 지원 데이터가 단일 모달 훈련보다 더 정보적인 표현을 산출할 수 있는 조건을 유도하였습니다. 다양한 실험에서는 이미지-텍스트 작업에서 비일치 데이터가 자기 지도 학습(self-supervised) 및 감독 학습(supervised) 각각의 분야에서 일관된 개선을 보여주었습니다.

- **Performance Highlights**: UML은 데이터 모든 유형에서의 성능 향상을 입증하였으며, 10개의 표준 시각 벤치마크에서 효과적인 수준으로 출현하였습니다. 특히, 두 개의 모달리티에서 세 개의 모달리티로 전환할때 정보 전이 효과가 더욱 뚜렷해졌습니다. 실험 결과, 이미지와 텍스트 간의 전환 비율을 정량화하여, 한 개의 이미지가 훈련 모델에서 사용할 수 있는 단어 수를 측정하는 방법도 포함되어 있습니다.



### In-Context Clustering with Large Language Models (https://arxiv.org/abs/2510.08466)
- **What's New**: 본 논문에서 제안하는 In-Context Clustering (ICC)은 다양한 분포에서 데이터를 클러스터링하는 유연한 LLM 기반 절차입니다. 기존의 클러스터링 알고리즘과 달리, ICC는 주어진 입력 간의 복잡한 관계를 주의(attention) 메커니즘을 통해 유연하게 포착합니다. 사전 훈련된 LLM이 텍스트로 인코딩된 숫자 데이터에서 인상적인 제로샷(zero-shot) 클러스터링 능력을 보여줍니다.

- **Technical Details**: ICC는 비지도 학습 환경으로 인컨텍스트 학습(in-context learning)을 확장하여, 라벨 없는 입력 데이터만을 이용하여 클러스터 레이블을 생성합니다. 클러스터링 조건이 변경될 경우, 모델의 가중치나 특징을 업데이트하지 않고도 프롬프트를 수정하여 쉽게 조정할 수 있습니다. 이 과정에서 주의 매트릭스가 클러스터 패턴을 시각적으로 보여줍니다.

- **Performance Highlights**: ICC는 숫자 데이터와 이미지 데이터에서 다양하게 평가되었으며, 특히 무리배포와 풍부한 의미를 갖는 이미지에서 뛰어난 성능을 보였습니다. 텍스트 조건화된 이미지 클러스터링 기능을 갖추고 있어 기존의 방법보다 더 유연한 클러스터링을 수행할 수 있습니다. ICC는 최신 캡션 기반 LLM 클러스터링(Kwon et al., 2024)보다 향상된 성능을 입증하였습니다.



### SummDiff: Generative Modeling of Video Summarization with Diffusion (https://arxiv.org/abs/2510.08458)
- **What's New**: 이번 논문에서는 비디오 요약(video summarization) 문제를 조건부 생성(conditional generation) 태스크로 재정의하여, 다양한 인간의 관점을 반영하는 여러 가지 가능한 요약을 생성할 수 있는 모델을 제안합니다. 기존의 방법들은 주로 프레임의 중요도를 평균하여 단일 요약을 생성하는 방식이었고, 이는 주관성을 충분히 반영하지 못했습니다. 우리의 모델, SummDiff는 이러한 주관적 특성을 고려하여, 비디오의 좋은 요약이 어떤 분포를 따르는지를 학습하게 됩니다.

- **Technical Details**: 비디오 요약을 위해 NN 프레임을 포함하는 비디오에서 S(N<S)개의 프레임을 선택하는 것을 목표로 합니다. 우리는 비디오의 각 프레임에 대해 조건부로 학습된 노이즈를 제거하는 과정을 통해 개인의 중요도 점수를 추정합니다. 새로운 모델인 SummDiff는 프레임의 중요도 벡터를 샘플링하여 다수의 가능한 요약을 생성하는 방식으로 설계되었으며, 이는 데이터의 조건부 분포에 의해 이루어집니다.

- **Performance Highlights**: 광범위한 실험을 통해 SummDiff는 여러 벤치마크에서 최신의 성능(state-of-the-art)을 달성했으며, 각 annotator의 선호도와 더 밀접하게 일치하는 요약을 생성함을 입증했습니다. 더불어, 요약 문제에서 종종 간과되는 knapsack 최적화 과정에 대한 분석도 제공하여, 새로운 메트릭 기반의 통찰력을 제안해 추가적인 성과를 도출하였습니다.



### Integral Signatures of Activation Functions: A 9-Dimensional Taxonomy and Stability Theory for Deep Learning (https://arxiv.org/abs/2510.08456)
Comments:
          25 pages

- **What's New**: 본 논문은 활성화 함수(activation function)에 대한 새로운 체계를 제안합니다. 기존의 분석이 주로 경험적(heuristic) 접근에 의존했던 반면, 이 연구는 Gaussian propagation 통계와 비율, 경계 변화 등의 정량적 지표를 결합하여 새로운 수학적 프레임워크를 개발했습니다. 이를 통해 표준 활성화 함수에 대한 체계적인 분류가 가능해졌으며, 안정성(robustness)과 모델 성능 향상에 기여할 수 있습니다.

- **Technical Details**: 논문은 아홉 차원의 적분 서명(integral signature) S_sigma(phi) 를 정의하며, Gaussian 통계(m1, g1, g2, m2, eta)와 극한 경사(asymptotic slopes, alpha_plus, alpha_minus), 그리고 총 변화(total variation, TV(phi'))를 포함합니다. 이 서명은 다양한 활성화 함수의 표현력을 정량화하며, Lyapunov 기반의 정리들을 통해 모델의 안정성을 보장합니다. 추가적으로, 커널 관점(kernel perspective)에서의 차원 비례 Hessian 경계와 매끄러움(smoothness)간의 관계를 탐구합니다.

- **Performance Highlights**: 이 프레임워크를 통해 ReLU, leaky-ReLU, tanh, sigmoid, Swish, GELU, Mish, TeLU와 같은 여덟 가지 활성화 함수의 정확한 분류가 이루어졌습니다. 이론적 예측은 수치적 Gauss-Hermite 및 Monte Carlo 검증을 통해 확인되었습니다. 결과적으로, 이 연구는 활성화 함수 설계에 대한 객관적이고 실행 가능한 지침을 제공하여, 시험 및 오류(trial-and-error)에서 벗어난 안정적이고 수학적으로 입증된 선택을 위한 기초를 마련합니다.



### gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity (https://arxiv.org/abs/2510.08450)
Comments:
          22 pages, 22 figures, 7 tables

- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)에서 발생하는 over-squashing 문제를 재조명하며, 특히 모델의 저장 및 검색 용량 관점에서 분석합니다. 연구자들은 정보 병목 현상이 저장 용량을 포화시킬 수 있음을 입증하기 위해 새로운 합성(task) 작업을 도입했습니다. 또한, 기존의 연관 기억 모델(associative memory) 및 형태 모델링(sequence modeling)에서 영감을 받아, 저장 용량이 향상된 새로운 GNN 아키텍처를 제안하고 있습니다.

- **Technical Details**: 기존 GNN 아키텍처는 노드의 표현을 업데이트하기 위해 이웃 노드와 정보를 교환하는 메시지 전달(message-passing) 패러다임을 따릅니다. 그러나 많은 레이어로의 확장이 어려운 이유는 over-smoothing과 over-squashing이라는 두 가지 중요한 문제 때문입니다. 저자들은 over-squashing을 저장 용량의 한계로 설명하고, 이를 해결하기 위해 최근의 xLSTM 아키텍처를 기반으로 새로운 MPNN 아키텍처를 제안하였습니다.

- **Performance Highlights**: 새롭게 제안된 GNN 아키텍처는 저장 용량 향상에 성공했고, 결과적으로 합성 작업 및 다양한 실제 그래프 벤치마크에서 우수한 성능을 보여주고 있습니다. 실험 결과는 capacity over-squashing이 sensitivity over-squashing과 별개로 발생할 수 있음을 입증합니다. 이 아키텍처는 정보 저장 및 검색 능력이 개선된 점이 주목할 만합니다.



### Synthetic Series-Symbol Data Generation for Time Series Foundation Models (https://arxiv.org/abs/2510.08445)
Comments:
          63 pages, NeurIPS 2025 accepted

- **What's New**: 최근 시간 시계열 분석(TSA)을 위한 파운데이션 모델들이 주목받고 있으나, 훈련 데이터의 부족과 불균형이 큰 도전 과제가 되고 있습니다. 이를 해결하기 위해 고안된 S2S² 데이터 생성 메커니즘은 시간 시계열 데이터와 해당하는 기호 표현을 고품질로 생성할 수 있게 합니다. 이러한 접근법을 바탕으로, 기호 정보를 활용하여 시계열 표현을 향상하는 SymTime이라는 사전 학습된 모델을 개발했습니다.

- **Technical Details**: SymTime 모델은 대규모의 다양한 시계열 데이터에서 자가 감독 또는 비감독 방식으로 사전 훈련된 심층 신경망으로, 일반화 가능한 시계열 표현을 학습하여 소규모 샷 또는 전이 학습을 통해 다양한 다운스트림 시계열 작업을 효율적으로 수행할 수 있습니다. 본 논문에서는 복잡한 동적 시스템을 나타내는 매핑 이론인 Takens의 정리에 기초하여, 시간 시계열이 복잡한 동적 시스템의 표현으로 작동할 수 있다는 점을 포괄적으로 다뤘습니다. S2S² 데이터 생성 과정은 다변량 입력-출력 기호 표현을 구축하고, 이를 통해 고유한 시계열 데이터를 생성하는 것을 포함합니다.

- **Performance Highlights**: SymTime은 다섯 가지 주요 TSA 작업에서 경쟁력 있는 성능을 보이며, 실제 데이터셋으로 사전 훈련된 파운데이션 모델과 비교할 때 유사하거나 그 이상으로 성능을 발휘합니다. 또한, S2S² 데이터셋의 규모가 다운스트림 작업에서 모델 성능과 직접적으로 연관되어 있음을 입증했습니다. 이러한 결과는 생성된 데이터와 사전 훈련 메커니즘이 데이터 부족 문제를 해결하고 작업 성능을 향상시킬 잠재력이 있음을 강조합니다.



### xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning (https://arxiv.org/abs/2510.08439)
Comments:
          24 Pages, 4 Figures, 2 Tables

- **What's New**: xRouter는 현대 LLM(대형 언어 모델)의 성능과 비용 간의 균형을 조화롭게 맞추기 위해 개발된 도구이며, 학습된 라우터가 직접 질문에 답하거나 외부 모델을 호출할 수 있는 시스템입니다. 기존의 정적 에스컬레이션 규칙이나 키워드 휴리스틱의 한계를 극복하고, 비용과 성과를 인식한 보상을 통해 라우팅을 강화 학습 문제로 형성함으로써 손으로 설계된 규칙의 필요성을 없앴습니다.

- **Technical Details**: xRouter는 도구 호출 기반의 라우팅 시스템으로, 경제적 제약을 명시적으로 포함한 보상 구조를 통해 모델 호출을 최적화합니다. 라우터의 학습은 강화 학습 방식을 통해 이뤄지며, 모델의 능력에 따라 적절히 질문을 처리하고 필요할 때는 외부 모델에 위임합니다. 이를 통해 경제적 요소를 함께 고려하면서도 성능 향상과 비용 절감을 동시에 실현할 수 있습니다.

- **Performance Highlights**: xRouter는 다양한 벤치마크에서 비용-성과 트레이드오프를 성공적으로 달성했으며, 비슷한 작업 완료율을 유지하면서 상당한 비용 절감을 이루어냈습니다. 초기 탐색을 통해 다양한 실험을 진행하였고, 이를 통해 얻은 실증적 통찰들은 향후 시스템 개발에 있어 유용한 방향성을 제시합니다. 이러한 기여를 통해 xRouter는 원칙 기반의 경제적 접근 방식을 가진 LLM 조정의 첫 걸음이 되고자 합니다.



### ClauseLens: Clause-Grounded, CVaR-Constrained Reinforcement Learning for Trustworthy Reinsurance Pricing (https://arxiv.org/abs/2510.08429)
Comments:
          Accepted for publication at the 6th ACM International Conference on AI in Finance (ICAIF 2025), Singapore. Author-accepted version (October 2025). 10 pages, 5 figures

- **What's New**: 최근 발표된 ClauseLens는 재보험(再保險) 계약 가격을 위한 새로운 강화 학습(framework) 프레임워크입니다. 이 시스템은 투명하고 규제를 준수하며 위험 인식과 관련된 조항(clause) 기반의 계약 견적을 제공합니다. ClauseLens는 재보험 인용 작업을 위험 인지 제약 마르코프 결정 과정(RA-CMDP)으로 모델링하여, 법적 조항을 인용 처리하여 투명성과 감사 가능성을 높입니다.

- **Technical Details**: ClauseLens는 법적 조항 검색(legal clause retrieval), 위험 감수 정책 학습(risk-sensitive policy learning), 조항 기반 정당화 생성(clause-grounded justification generation)의 세 가지 주요 구성 요소로 구성됩니다. 이 시스템은 5백만 달러의 플로리다 허리케인 계약 요청에 대해 NAIC 지급 능력 기준을 준수하는 60% 분배 비율을 권장하고, 이 정책의 법적 근거를 자연어로 제시합니다. 이러한 구조 덕분에 높은 해석 가능성과 또한 규정 준수를 확보할 수 있습니다.

- **Performance Highlights**: ClauseLens는 다중 에이전트 재보험 시뮬레이터에서 검증되어 51%의 지급 능력 위반(solvency violations)을 줄이고, 극단적 위험(tail-risk) 성능을 27.9% 개선했습니다. 또한 조항 입각 정당화의 정확도는 88.2%에 달하며, 검색의 정밀도는 87.4%, 재현율(recall)은 91.1%에 이릅니다. 이러한 성과는 재보험 산업 데이터에 기반하여 법적 문맥을 효과적으로 결정 및 설명 경로에 통합함으로써 해석 가능하고 감사로운 인용 행동을 제공함을 보여줍니다.



### Reinforcing Diffusion Models by Direct Group Preference Optimization (https://arxiv.org/abs/2510.08425)
- **What's New**: 이번 논문은 Direct Group Preference Optimization(DGPO)이라는 새로운 온라인 강화 학습 알고리즘을 제안합니다. DGPO는 기존의 Group Relative Preference Optimization(GRPO)의 정책 경량화 프레임워크를 완전히 배제하고 그룹 수준의 선호도에서 직접 학습함으로써 확산 모델(diffusion models)의 훈련을 가속화합니다. DGPO는 비효율적인 확률적 정책의 필요성을 제거하고, 효율적인 결정론적 ODE 샘플러를 활용하여 훈련 속도를 약 20배 향상시킵니다.

- **Technical Details**: DGPO는 기존의 정책 기반 강화 학습 대신 그룹간 상대적 선호도를 직접적으로 최적화하는 방식으로 작동합니다. 각 프롬프트에 대해 ODE 기반 롤아웃을 사용하여 양호한 샘플과 불량 샘플로 나누고, 이들 그룹 간의 선호도를 최대화함으로써 모델을 최적화합니다. 이 방법론은 Direct Preference Optimization(DPO)의 자연스러운 확장으로, 그룹 간 정보를 포함하면서도 정책의 확률적 의존성을 제거하는 방식입니다.

- **Performance Highlights**: DGPO의 실험 결과는 Flow-GRPO와 비교하여 약 20배 더 빠른 훈련 속도를 기록하며, 다양한 도메인 보상 메트릭에서 뛰어난 성능을 보여줍니다. 특히, GenEval 벤치마크에서 DGPO는 Flow-GRPO보다 거의 30배 빠른 훈련을 달성하고 기본 모델의 성능을 63%에서 97%로 향상시킵니다. 이러한 성과들은 DGPO가 확산 모델 정렬을 위한 강력한 기법으로 자리매김할 가능성을 보여줍니다.



### Prompts Generalize with Low Data: Non-vacuous Generalization Bounds for Optimizing Prompts with More Informative Priors (https://arxiv.org/abs/2510.08413)
Comments:
          EXAIT Workshop paper at ICML 2025

- **What's New**: 본 논문은 데이터가 부족한 상황에서도 유용한 일반화 경계를 제공하는 새로운 일반화 경계를 도출하며, 특정 작업 데이터에 최적화되거나 정보가 풍부한 정규화 사전(informative prior)을 활용하여 프롬프트 최적화를 향상시킵니다.\n프롬프트에서의 일반화 문제는 인공지능 시스템을 신뢰할 수 있게 배포하는 데 있어 매우 중요하며, 대규모 프롬프트 공간에서의 최적화와 더불어 이러한 경계를 엄밀하게 분석할 필요성을 제기합니다.

- **Technical Details**: 논문에서는 PAC-Bayes 이론에 기반하여 프롬프트 최적화 알고리즘을 위한 데이터 종속적인 일반화 경계를 도출합니다.\n이 종결 경계는 훈련 데이터의 불확실성을 처리하기 위한 방법으로서 데이터 종속 perplexity를 정규화하는 방식을 사용합니다.\n또한, 경험적인 연구를 통해 이러한 경계가 데이터가 부족한 상황에서도 유용하며, 더 나은 일반화를 위해 perplexity 정규화를 통한 최적화 알고리즘의 효율성을 보여줍니다.

- **Performance Highlights**: 제안된 경계와 정규화를 통해 얻은 경험적 결과는 최적화된 프롬프트의 일반화 성능을 실제로 개선할 수 있음을 확인합니다.\n최적화된 프롬프트는 훈련 데이터에서 뛰어난 성능을 보이지만, 일반화의 관점에서도 높은 신뢰성을 달성하는 데 기여합니다.\n이 연구는 인공지능의 다음 단계인 AGI(Artificial General Intelligence)로 나아가는 여정에서 매우 중요한 이론적 및 실제적 통찰을 제공합니다.



### Biology-driven assessment of deep learning super-resolution imaging of the porosity network in dentin (https://arxiv.org/abs/2510.08407)
- **What's New**: 이번 연구에서는 치아의 기계 감각 시스템이 주로 Odontoblast 세포의 자극에 의존하고 있으며, 이를 위해서는 미세한 다공성 네트워크를 시각화해야 함을 강조합니다. 이를 위해 현재의 표준인 confocal fluorescence microscopy의 한계를 극복하기 위해 다양한 딥러닝(DL) 초해상도(SR) 모델을 테스트했습니다. 연구팀은 훈련된 모델을 통해 저해상도 이미지의 실험적 수집 속도를 높이고 최적의 이미지 품질을 복원하는 방법을 개발했습니다.

- **Technical Details**: 세 가지 감독형 2D SR 모델(RCAN, pix2pix, FSRCNN)과 한 가지 비감독형(CycleGAN)을 사용하여 고해상도 및 저해상도 confocal 이미지를 쌍으로 실험하여 다양한 샘플링 방식으로 획득했습니다. 생성된 SR 이미지는 픽셀 크기를 2배, 4배, 8배로 증가시키는 성과를 얻었습니다. 모델의 성능은 이미지 품질 평가(IQA) 메트릭을 사용하여 측정되었으며, 이는 시각적 인지와 모순되는 일관되지 않은 결과를 보여주었습니다.

- **Performance Highlights**: IQA 메트릭의 한계를 넘어 치아 다공성의 특정 구조를 겨냥한 세그멘테이션 접근방식이 사용되었습니다. 또한 SR 모델이 confocal 이미지 스택을 통해 3D 다공성 연결성을 유지하는 능력을 그래프 분석을 통해 평가했습니다. 이러한 생물학적 기반의 평가는 SR 성능의 기계적 해석을 개선하고, 모델의 민감도 차이와 이미지 생성의 비선형성이 IQA 메트릭의 실패를 설명함을 강조합니다.



### FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts (https://arxiv.org/abs/2510.08396)
Comments:
          NeurIPS 2025 accepted paper

- **What's New**: FlyLoRA는 Fly의 후각 회로에서 영감을 받아 개발된 모형으로, 기존의 LoRA 방법에서 나타나는 파라미터 간의 간섭을 줄여줍니다. 이 방법은 낮은 차원의 전문가 활성화(rank-wise expert activation)를 통해 더 나은 수행 성능을 제공합니다. 이러한 혁신은 단일 작업 내에서의 파라미터 간섭을 효과적으로 해소하며, 기존의 MoE 기반 LoRA 방법들보다 효율성을 높여주는 목표를 달성합니다.

- **Technical Details**: FlyLoRA는 매트릭스 𝑨를 고정된 희소 랜덤 프로젝션(sparse random projection)으로 취급하며, 여기서 각 LoRA 컴포넌트는 서로 다른 랜덤 프로젝션을 통해 약 orthogonal 하게 매핑됩니다. 이를 통해 FlyLoRA는 intra-task(작업 내) 간섭을 최소화하고, multi-task(다중 작업) 연합에서도 비교적 독립적으로 작동하도록 설계되었습니다. 이러한 구조는 AI 기술에 생물학적 구조의 영감을 결합한 결과로, 효율적인 파라미터 사용을 가능하게 합니다.

- **Performance Highlights**: FlyLoRA는 일반 지식 이해, 과학 질문 답변, 수학 추론, 코드 생성 등 다양한 작업에서 기존 방법들 대비 일관된 성능 향상을 보여줍니다. 실험 결과를 통해 FlyLoRA가 계산 효율과 파라미터 간섭 감소 모두를 이뤄내며, 최신 기계 학습 기준에 부합하는 성능을 입증하였습니다. 이러한 결과는 AI 기술 진보의 중요한 원동력이 될 것으로 기대됩니다.



### Characterizing the Multiclass Learnability of Forgiving 0-1 Loss Functions (https://arxiv.org/abs/2510.08382)
Comments:
          9 pages

- **What's New**: 이번 논문에서는 유한 레이블 다중 클래스(multiclass) 설정에서 용서하는 0-1 손실 함수의 학습 가능성을 규명합니다. 이를 위해 새로운 조합적 차원(combinatorial dimension)을 제안하며, 이는 Natarajan Dimension에 기반합니다. 우리의 연구 결과, 가설 클래스(hypothesis class)가 학습 가능하다는 것은 이 일반화된 Natarajan Dimension이 유한할 때와 동치임을 보여줍니다.

- **Technical Details**: 우리는 새로운 Generalized Natarajan Dimension을 정의하고, 이를 통해 학습 문제의 설정을 분석합니다. 또한, 집합 값 피드백(set-valued feedback)과의 연결성을 제시하여, 여러 출력의 경우에도 이론이 적용될 수 있음을 시사합니다. 이로 인해, 학습 문제의 특성과 차원 간의 관계를 보다 깊이 이해할 수 있게 됩니다.

- **Performance Highlights**: 우리의 결과에 따르면, 학습 문제의 학습 가능성은 Natarajan Dimension에 의해 구체화됩니다. 이 연구는 다중 클래스 분류(multi-class classification)에서 손실 함수의 특성을 이해하는 데 중요한 기여를 합니다. 또한, 이를 통해 기계 학습 분야에서의 다양한 적용 가능성을 탐구할 수 있는 기초를 제공합니다.



### Contrastive Self-Supervised Learning at the Edge: An Energy Perspectiv (https://arxiv.org/abs/2510.08374)
- **What's New**: 이 논문은 자원이 제한된 장치에서 대비 학습(Contrastive Learning; CL)의 적용 가능성을 평가한 최초의 연구 중 하나로, SimCLR, MoCo, SimSiam, Barlow Twins와 같은 네 가지 주요 프레임워크의 에너지 소비, 데이터 요구 사항 및 메모리 사용량을 분석합니다. 특히, 다양한 데이터 환경에서 에너지 프로필을 포함한 체계적인 벤치마킹 전략을 도입하여 CL 프레임워크의 실용성을 탐구했습니다. 연구 결과, SimCLR이 예상과 달리 가장 낮은 에너지 소비를 나타내었으며, 경량 신경망 아키텍처와의 결합도 평가하여 자원 제약이 있는 환경에서의 최적화 방향을 제시합니다.

- **Technical Details**: 이 연구는 SimCLR과 MoCo와 같은 기존의 CL 프레임워크를 활용하여 최적의 피처 임베딩(feature embedding)을 학습하는 다양한 기법을 소개합니다. 연구에서는 각 프레임워크의 네거티브 샘플 사용 방식, 배치 크기, 데이터 증강(data augmentation) 기법의 효용성을 고려하여, 훈련 데이터가 부족한 상황에서도 효과적으로 작동할 수 있는 설계를 요청합니다. 또한, 에너지 소비(energy consumption) 분석을 통해 리소스 제약이 있는 장치에서 CL 프레임워크의 적용 가능성을 심도 있게 평가합니다.

- **Performance Highlights**: 연구 결과 SimCLR이 다양한 데이터 환경에서 가장 낮은 에너지 소비를 기록하며, ResNet-18과 같은 경량 모델이 적절한 에너지-정확도 비율을 보여주었습니다. 이 연구는 기존 CL 방법의 확장을 통해 자원 제한적인 환경에서도 효과적인 대안을 제시하고, 이러한 프레임워크들이 전국적으로 확대될 수 있는 가이드를 제공합니다. 또한 연구는 데이터 준비와 트레이닝 시간 단축을 위한 실용적인 가이드라인을 제공하여, 향후 CL 솔루션의 효율성 증대를 위한 연구 방향을 제시합니다.



### Guided Star-Shaped Masked Diffusion (https://arxiv.org/abs/2510.08369)
- **What's New**: 이 논문은 사전 학습된 masked diffusion 모델의 샘플링 절차에서 발생하는 제한 사항을 해결하고자 하는 새로운 알고리즘을 제안합니다. 기존의 샘플링 방식은 오류 수정이 불가능하고, 저단계 생성에서 어려움을 겪고 있습니다. 저자들은 star-shaped paradigm을 활용하여 생성 과정을 개편하고, learnable re-masking scheduler를 도입하여 예측된 오류를 수정하는 방식으로 샘플 품질과 효율성을 크게 향상시켰습니다.

- **Technical Details**: 본 연구의 핵심은 진화된 masked diffusion 모델의 이해를 바탕으로, 각 토큰이 미리 결정된 후 불가역적인 특성을 갖는 문제를 해결하고자 하는 것입니다. 새로운 샘플러는 먼저 깨끗한 데이터의 예측 값을 생성한 후, 이 데이터를 바탕으로 다음 스텝을 진행합니다. 이러한 두 단계의 과정은 이미 생성된 토큰을 재조정할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 G-Star 알고리즘은 텍스트 및 코드 생성 분야에서 기존의 방법을 초월하거나 동등한 성능을 보였습니다. 특히, 적은 단계의 샘플링에서도 품질 향상이 두드러지며, 다양한 도메인에서 우수한 샘플링 성능을 입증했습니다. 이는 오류 수정 메커니즘의 효과적인 도입 덕분에 가능하게 되었습니다.



### DeepEN: Personalized Enteral Nutrition for Critically Ill Patients using Deep Reinforcement Learning (https://arxiv.org/abs/2510.08350)
- **What's New**: DeepEN은 중환자에게 맞춤형의 장내 영양(intravenous nutrition, EN)을 제공하기 위한 심층 강화학습(deep reinforcement learning, RL) 프레임워크로 소개됐다. MIMIC-IV 데이터베이스의 11,000명 이상의 ICU 환자로부터 오프라인 학습하여, 매 4시간마다 환자의 생리적 변화에 맞춤화된 칼로리, 단백질, 수분 섭취 권장 사항을 생성한다. 이는 단기적인 생리학적 및 영양 관련 목표를 장기적인 생존 결과와 균형을 이루도록 설계된 맞춤형 보상 함수를 통합하여 이루어진다.

- **Technical Details**: DeepEN은 환자의 진단, 검사 결과, 생체 신호 및 과거 치료를 기반으로 한 포괄적인 환자 특징 목록에 따라 맞춤형 권장 사항을 생성한다. Dueling Double Deep Q-Network(D3QN) 알고리즘을 사용하여 안전하고 임상적으로 실행 가능한 정책을 학습하고, 이를 위해 보수적인 Q-learning 정규화를 적용한다. 강화학습의 훈련 과정에서 실수의 위험을 최소화하기 위한 조치를 취하여, 복잡한 환경에서도 효과적으로 작동할 수 있도록 한다.

- **Performance Highlights**: DeepEN은 기존의 임상 관행이나 가이드라인 기반 정책보다 우수한 성과를 보였다. 실제로 예상 사망률을 3.7% 감소시켰고 (18.8% 대 22.5%), 주요 영양 바이오마커에서 개선이 관찰되었다. 이는 데이터 기반의 안전한 맞춤형 EN 요법의 잠재력을 강조하며, 기존의 가이드라인이나 경험적 접근 방식보다 더 나은 결과를 달성할 수 있는 가능성을 제시한다.



### Learning What's Missing: Attention Dispersion and EMA Stabilization in Length Generalization (https://arxiv.org/abs/2510.08341)
Comments:
          10 pages, 5 figures, 2 tables

- **What's New**: 이번 연구에서는 트랜스포머 모델에서 길이 일반화(length generalization) 개념을 탐구했습니다. 구체적으로, 설정 보완(task complement) 작업을 통해 입력 시퀀스에서 누락된 토큰에 대한 균일 분포를 예측하는 능력을 다룹니다. 이 작업은 보드 게임 스타일의 추론에 중요한 요소로 작용하며, 두 가지 주요 이론적 결과를 도출합니다: 첫째, 단일 계층의 주의(attention) 전용 트랜스포머에 대한 강한 경계가 입증되었습니다. 둘째, 모델이 길이 1과 2의 입력에서 균형 잡힌 로짓(logit) 이동을 달성할 수 있으면, 더 긴 시퀀스에 대한 일반화도 가능하다는 점입니다.

- **Technical Details**: 연구에서는 주의 산술식이 긴 시퀀스에서 활성화 가중치의 분산(attention dispersion)을 증가시켜 추론의 정밀도를 감소시킨다는 기계적 해석(mechanistic reading)이 제시되었습니다. 이를 방지하기 위해 드롭아웃(dropout)을 증가시키면 값 벡터(value vectors)의 진폭을 높여 이 효과를 완화할 수 있을 것으로 가설을 세웠습니다. 또한, 훈련 동역학(training dynamics)을 분석하여 짧은 시퀀스 다음에 나올 수 있는 많은 토큰 중에서 샘플링이 이루어지면서 기울기(gradients)가 노이즈화(noisy)된다는 두 번째 장애물을 규명했습니다. 따라서 편향 수정 지수이동평균(BEMA, Bias-corrected Exponential Moving Average)의 사용이 이 문제를 완화할 수 있을 것이라고 가정했습니다.

- **Performance Highlights**: 랜덤 하이퍼파라미터 검색을 통해 제안된 전략이 성능 향상에 기여함을 입증했습니다. 더 복잡한 설정을 위해 OthelloGPT 모델에 대한 길이 일반화 실험을 통해 BEMA가 이 경우에도 성능 지표를 강력하게 개선함을 확인했습니다. 연구 결과는 강조된 두 가지 메커니즘인 드롭아웃과 BEMA가 통일된 다음 토큰 분포를 학습하며 일반화 성능 향상에 기여한다는 점을 보여주었습니다.



### To Ask or Not to Ask: Learning to Require Human Feedback (https://arxiv.org/abs/2510.08314)
- **What's New**: 이번 논문에서는 인간 전문가의 피드백을 보다 효과적으로 통합할 수 있는 Learning to Ask (LtA)라는 새로운 프레임워크를 제안합니다. LtA는 기존의 Learning to Defer (LtD) 접근 방식을 개선하여, 전문가의 입력을 예측 프로세스 중 적시에 요청할 수 있는 최적의 전략을 제시합니다. 이 두 요소 아키텍처는 표준 ML 모델과 추가적인 전문가 피드백으로 훈련된 강화 모델로 구성되어 있습니다.

- **Technical Details**: LtA 프레임워크는 두 가지 구현 방식을 제공합니다: 하나는 단계적으로 모델을 훈련시키는 Sequential Approach이며, 다른 하나는 모델을 동시에 최적화하는 Joint Approach입니다. 이들 모델은 리얼라이저블 컨시스턴시(realizable-consistency) 보장을 갖춘 서라게이트 손실 함수를 활용하여 최적화됩니다. 특히, LtA는 ML 모델이 언제 전문가에게 피드백을 요청해야 하는지를 결정하도록 훈련될 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, LtA는 더 풍부한 피드백을 제공받을 때, LtD를 초월하는 성능을 보여줍니다. 특히 임상 판단이 중요한 진단 사례에서 전문가 피드백을 통합하는 방식이 모델의 정확성을 극대화하는 데 기여하는 것으로 나타났습니다. 논문에서는 실험에 사용된 소스 코드와 결과를 오픈 소스로 제공하고 있어, 연구자들은 이를 통해 Cumulated Learning의 가능성을 탐구할 수 있습니다.



### Robust and Efficient Collaborative Learning (https://arxiv.org/abs/2510.08311)
- **What's New**: 본 연구에서는 Robust Pull-based Epidemic Learning (RPEL)이라는 새로운 협업 기계 학습 접근법을 제안합니다. 기존의 중앙 서버에 의존하지 않으며, 기존 방법이 통신 비용이 $	extmath{O}(n^2)$으로 증가하는 것과 달리, RPEL은 통신 전략을 $	extmath{O}(n 	ext{log} n)$로 줄입니다. 이를 통해 RPEL은 모델 파라미터를 소규모 무작위 노드 집합에서 끌어옴으로써, 수렴 보장을 손상시키지 않고도 필요한 메시지 수를 대폭 줄입니다.

- **Technical Details**: RPEL은 무작위화된 전염병 모델 기반의 통신 방식을 이용하여 각 노드가 소규모 무작위 피어 집합에서 모델 업데이트를 주기적으로 끌어옵니다. 이 방식은 전통적인 피어-투-피어 솔루션에 비해 통신 부담을 현저히 줄이면서도 악의적인 노드에 대한 강인성을 유지합니다. 제안된 방법은 주어진 비잔틴 공격에 대한 강인성을 보장하면서 데이터 이질성이 있는 일반 비볼록 설정에서도 엄격한 수렴 보장을 확립합니다.

- **Performance Highlights**: RPEL은 MNIST와 CIFAR-10 데이터셋에서 최대 20%의 악의적인 노드가 존재하는 분산 시스템에서 좋은 성능을 발휘함을 보여줍니다. 특히 RPEL은 최신의 모든-대-모든 강인한 방법들과 경쟁하며, 더 낮은 통신 비용으로 유사한 정확성을 달성합니다. 또한 RPEL은 선정된 노드 수가 감소할 때 Effective adversarial fraction의 감소를 통해 확장성 우위를 나타냅니다.



### Dynamic Features Adaptation in Networking: Toward Flexible training and Explainable inferenc (https://arxiv.org/abs/2510.08303)
Comments:
          Accepted at AI4NextG Workshop, NeurIPS 2025

- **What's New**: 이 논문은 6G 네트워크 환경에서 적응형 랜덤 포레스트(Adaptive Random Forests, ARFs)를 통해 AI 모델의 동적 피처 적응(Dynamic Feature Adaptation)에 대한 새로운 접근 방식을 제시합니다. AI는 통신 네트워크의 변화하는 조건에 적응해야 하며, 새로운 기능과 측정값이 도입될 때마다 학습을 유연하게 동적으로 진행해야 합니다. 또한, 설명 가능한 AI(Explainable AI, XAI)의 중요성을 강조하며, Drift-Aware Feature Importance (DAFI) 방법을 통해 피처 중요성을 효율적으로 평가할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 ARFs를 사용하여 통신 데이터셋에 대한 강력한 예측 성능을 입증하고, DAFI 알고리즘이 데이터 드리프트를 감지하여 적절한 설명 가능성 방법을 선택하는 방식을 제안합니다. DAFI는 Kolmogorov-Smirnov (KS) 테스트를 이용하여 분포 변화에 따라 SHAP 또는 Mean Decrease in Impurity (MDI) 방법 중 어떤 것을 사용할지를 결정합니다. 이 접근 방식은 동적인 데이터 스트림에서 피처 중요성을 효율적으로 평가할 수 있게 해주는 유연한 모델을 제공합니다.

- **Performance Highlights**: 실험 결과, DAFI는 다른 기존 방법보다 더 일관된 피처 중요성 값을 제공하며, 런타임을 최대 2배까지 단축시켰습니다. 특히 3개의 다양한 데이터세트를 이용하여 ARFs의 성능이 향상됨을 확인하였고, 실시간으로 변화하는 조건에서도 효과적으로 대처할 수 있는 능력을 갖추었습니다. ARFs와 DAFI의 조합은 6G 네트워크와 같이 빠르게 변화하는 환경에서 효과적인 AI 방법을 구축하는 데 유망한 프레임워크를 제공합니다.



### Bridging the Physics-Data Gap with FNO-Guided Conditional Flow Matching: Designing Inductive Bias through Hierarchical Physical Constraints (https://arxiv.org/abs/2510.08295)
Comments:
          8 pages, 1 figure

- **What's New**: 이 논문은 전통적인 시계열 생성 방식에서 물리적 제약을 무시하는 문제를 해결하기 위한 계층적 프레임워크를 제안합니다. 이 새로운 접근법은 물리 법칙의 내재적 계층 구조를 깊은 생성 모델에 직접 삽입함으로써, 통계적 및 물리적 일관성을 고취시키는 새로운 패러다임인 physics-informed inductive bias를 소개합니다. Fourier Neural Operators (FNO)와 Conditional Flow Matching (CFM)을 결합하여 생성 품질과 예측 정확도를 개선하였습니다.

- **Technical Details**: 제안된 방법은 네 가지 설계 원칙을 기반으로 하는 Hierarchical Physics-Constrained FNO-CFM (HPC-FNO-CFM)을 사용합니다. 이 시스템은 물리 법칙의 우선순위를 명확하게 반영하는 계층형 유도 편향(hierarchical inductive bias), 데이터를 이용한 물리 연산자 학습(operator learning), 다양한 물리적 조건에서 작동할 수 있는 조건 적응성(conditional adaptability), 그리고 생성 과정 전반에 걸쳐 물리적 일관성을 보장하는 동적 가이드를 포함합니다. 각 계층의 FNO 연산자들은 서로 다른 주파수 대역과 계산 깊이를 가지며, 이를 통해 물리적 프로세스를 효율적으로 캡처합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 생성 품질을 16.3% 향상시키고 물리적 위반을 46% 줄이며 예측 정확도를 18.5% 개선하는 성과를 보였습니다. 실험은 고조파 진동기, 인간 활동 인식 및 리튬 이온 배터리 열화의 세 가지 도메인에서 수행되었으며, 추출시 성능(R2 = 0.694) 또한 높은 결과를 기록하였습니다. 전반적으로 각 구성 요소의 개별 및 상호 작용 효과를 체계적으로 정량화하여, 제안된 방법의 강력한 성능을 입증하였습니다.



### Counterfactual Identifiability via Dynamic Optimal Transpor (https://arxiv.org/abs/2510.08294)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이 논문은 관찰 데이터에서 고차원 다변량 결과의 반사실적( counterfactual ) 식별에 대한 질문을 다룹니다. Pearl (2000)의 주장에 따르면, 반사실적이 식별 가능해야 인과 관계 주장을 정당화할 수 있습니다. 최근 반사실적 추론에 관한 연구는 유망한 결과를 보였지만, 식별 부족으로 그 추정의 인과적 유효성을 저해한다고 지적합니다. 본 논문에서는 연속 시간 흐름을 이용하여 다변량 반사실적 식별을 위한 기초를 확립했습니다.

- **Technical Details**: 반사실적 전송 맵이 고유하고 단조로운 성질을 가질 수 있는 조건을 역동적 최적 수송(dynamic optimal transport) 이론과 결합하여 규명합니다. 기계의 단조성은 주어진 개입에 대해 사실적 결과들의 순위를 저장하는 데 필수적입니다. 다차원 변수에 대해 반사실적 식별을 가능하게 하는 제약 조건 집합을 특성화합니다.

- **Performance Highlights**: 구체적인 사례를 통해 반사실적 바탕 이론을 검증하고, 실제 이미지에서 공리적 반사실적 일관성이 개선되었음을 입증합니다. 이 모델은 역동적 최적 수송 이론을 인과적 메커니즘 분석에 활용하여 높은 식별 성능을 보여주고 있습니다. 반복 측정을 통해 추론의 일관성을 보장할 수 있음을 확인하였습니다.



### Mix- and MoE-DPO: A Variational Inference Approach to Direct Preference Optimization (https://arxiv.org/abs/2510.08256)
- **What's New**: 최근 등장한 Direct Preference Optimization (DPO)은 대형 언어 모델(LLMs)을 사용자 선호에 맞추는 간단하고 효과적인 대안으로 자리잡았습니다. 본 연구에서는 DPO의 한계를 극복하기 위해 Mix- 및 MoE-DPO 프레임워크를 제안합니다. 이는 DPO를 부드러운 혼합 모델과 전문가 혼합(Combination of Experts) 아키텍처로 확장하여, 다양한 선호 분포에 적응할 수 있도록 합니다.

- **Technical Details**: 새로운 Mix- 및 MoE-DPO 방법론은 전문가 배치를 위한 잠재 변수 모델을 도입하고, 변분 증거 하한(ELBO)을 최적화합니다. 이 방식은 사용자 특정 혼합 정책을 가능하게 하는 입력 의존적 부드러운 게이팅을 통해 보상 및 정책의 전문화를 촉진합니다. 또한, 이러한 접근법은 모듈화된 배포를 지원하여 기존 모델과의 효율적인 통합 및 사용자 개인화가 가능합니다.

- **Performance Highlights**: 기술적으로 Mix- 및 MoE-DPO는 변수를 고정한 Mix-DPO와, 입력 의존적 가중치를 가지는 MoE-DPO로 나뉘어, 다양한 인과 모델 크기와 다중 선호 데이터셋에서 그 성능을 검증하였습니다. 이로 인해 Mix- 및 MoE-DPO는 무수히 많은 적용 가능성을 가진 선호 기반 LLM 정렬 방법을 제시합니다.



### Opponent Shaping in LLM Agents (https://arxiv.org/abs/2510.08255)
Comments:
          29 pages, 15 figures, 15 tables

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)를 기반으로 한 에이전트들 간의 전략적 상호작용인 opponent shaping에 대한 최초의 조사를 수행했습니다. 기존의 알고리즘은 LLM에 맞지 않는 구조적 결함이 있기 때문에, 저자들은 이를 해결하기 위해 ShapeLLM이라는 새로운 알고리즘을 고안했습니다. 이 모델은 transformer 기반의 에이전트에서 동작하도록 설계된 model-free 방법입니다.

- **Technical Details**: ShapeLLM은 Proximal Policy Optimization (PPO) 방법을 사용하여 LLM 에이전트의 학습 역학을 서로에게 전략적으로 영향을 미치도록 실험했습니다. 연구팀은 반복적 매트릭스 게임을 이용하여 LLM 에이전트들이 서로의 학습 동태에 미치는 영향을 분석하고, 이를 통해 경쟁적 및 협력적 시나리오에서 전략적 상호작용의 실효성을 평가했습니다. 해당 연구는 LLM 에이전트가 독립적으로 행동하는 것이 아닌, 서로에게 영향을 주고받으며 학습한다는 중요한 사실을 증명합니다.

- **Performance Highlights**: LLM 에이전트들은 다양한 게임 이론적 환경에서 효과적으로 상대를 유도하여 경쟁적인 상황에서는 유리한 균형으로, 협력적인 상황에서는 상호 이익을 위한 행동을 촉진할 수 있음을 보였습니다. 연구 결과, LLM들이 지속적인 상호작용을 통해 각자의 행동을 조정하고, 더 나아가 프로소셜(prosocial) 행동을 이끌어내는 데 기여할 수 있는 가능성이 있음을 보여줍니다.



### The Hidden Bias: A Study on Explicit and Implicit Political Stereotypes in Large Language Models (https://arxiv.org/abs/2510.08236)
- **What's New**: 이번 연구는 정치적 편견과 고정관념의 전파를 8개의 주요 대형 언어 모델(Large Language Models, LLMs)을 대상으로 조사하였으며, 두 차원 정치적 컴퍼스 테스트(Political Compass Test, PCT)를 사용하여 이러한 모델들의 내재된 정치 성향을 분석하였습니다. 특히, 이 연구는 첫 번째로 명시적 고정관념(persona prompting)과 암묵적 고정관념을 비교하는 체계적인 방법론을 적용하였습니다. 연구 결과, 대다수 모델이 일관되게 좌파 성향을 지니고 있음을 밝혀냈고, 암묵적 고정관념이 명시적 고정관념보다 더욱 두드러진다는 흥미로운 사실을 발견하였습니다.

- **Technical Details**: 이 연구에서는 LLM의 정치적 편견을 평가하는 체계적 방법론을 채택하였으며, 두 차원 PCT를 표준 평가 프레임워크로 활용했습니다. PCT는 경제적 좌우축(left-right)과 사회적 자유-권위적(libertarian-authoritarian) 축을 통해 이념적 입장을 평가하며, 이를 기반으로 한 공개된 조사 및 이전 연구 결과와 비교하였습니다. 이 연구는 LLM 내에서 암묵적 및 명시적 고정관념의 상호작용과 정렬을 분석하여, 모델들간의 정치적 편향이 어떻게 나타나는지를 밝히며, 모델의 편향이 나타나는 방식에서의 투명성을 강조하였습니다.

- **Performance Highlights**: 연구 결과, 모든 조사된 LLM은 일관되게 좌파 성향을 보였으며, 고정관념의 구체적인 표현은 모델 간에 상당한 차이를 보였지만, 언어 변화를 통해 드러나는 암묵적 고정관념이 명시적 고정관념보다 더 뚜렷하게 나타났습니다. 이 연구는 암묵적과 명시적 고정관념이 상당한 정렬을 보임을 보여주어, 모델들이 자신의 편향에 대해 어느 정도 인식하고 있다는 것을 시사합니다. 따라서 이 연구는 LLM의 정치적 편향이 사회에 미치는 영향을 심층적으로 분석한 중요한 기초 자료를 제공합니다.



### Enhancing Reasoning for Diffusion LLMs via Distribution Matching Policy Optimization (https://arxiv.org/abs/2510.08233)
- **What's New**: 본 논문은 자기회귀 대형 언어 모델(AR-LLMs) 대신 확산 대형 언어 모델(dLLMs)의 잠재력을 탐구합니다. dLLMs는 고차원의 추론 작업을 효율적으로 처리할 수 있는 대안으로 주목받고 있으며, 이를 통해 더 높은 추론 처리량을 기대할 수 있습니다. 또한, 논문은 분포 일치 정책 최적화(Distribution Matching Policy Optimization, DMPO)를 제안하여 dLLMs의 추론 능력을 극대화하는 새로운 강화 학습 알고리즘의 필요성을 강조합니다.

- **Technical Details**: DMPO는 주어진 보상 편향의 정책 분포에 일치하도록 dLLMs의 정책을 최적화하는 정교하고 이론적으로 정립된 강화 학습 세부 조정 방법론입니다. 이 방법론은 중요 샘플링과 가중치가 매겨진 제거 교차 엔트로피(Weighted Denoising Cross-Entropy, WDCE) 손실을 활용하여, 훈련 시 다양한 고품질의 추론 경로를 탐색합니다. 특히, WDCE는 오직 깨끗한 샘플을 기반으로 작동하는 전방 목표 기능으로, 빠른 추론 기법과 결합하여 더 높은 속도를 낼 수 있도록 설계되었습니다.

- **Performance Highlights**: DMPO는 여러 온도 추론 벤치마크에서 우수한 성능을 보이며, 기존의 SOTA 기준 대비 최대 42.9%의 정확도 향상을 제공합니다. 또한, 기본 모델 대비 최대 55.8%의 성능 개선을 이루어냈으며, 이는 dLLMs의 분포 일치 프레임워크의 효과성을 입증합니다. 이러한 개선 덕분에 DMPO는 양방향 dLLMs에서 최고 성능 모델로 자리 잡았습니다.



### Reinforcement Learning from Probabilistic Forecasts for Safe Decision-Making via Conditional Value-at-Risk Planning (https://arxiv.org/abs/2510.08226)
- **What's New**: 이 논문에서는 Uncertainty-Aware Markov Decision Process (UAMDP)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Bayesian forecasting, posterior-sampling reinforcement learning, 그리고 conditional value-at-risk (CVaR) 제약 조건 하의 계획을 통합하여, 불확실성을 관리하면서도 실질적인 경제적 성과를 개선하는 것을 목적으로 합니다. UAMDP는 고빈도 주식 거래와 소매 재고 관리라는 두 가지 도메인에서 평가되며, 그 결과는 25%의 RMSE 감소와 32%의 sMAPE 향상으로 나타났습니다.

- **Technical Details**: UAMDP는 벨리안 예측을 RL 과정에 통합하여 불확실한 환경에서 실시간으로 신념을 업데이트하고 적응형 제어를 가능하게 합니다. 이 접근법은 에이전트가 자신의 사후 신념에 따라 불확실성이 높은 영역을 우선 탐험하도록 하여, 계획 수명이 에피소드 길이와 일치할 때 근사 최적 후회(regret)를 달성합니다. 또한, CVaR 제약을 포함한 계획을 통해 에이전트는 하방 위험을 완화하고 드물지만 고충격 사건에 대한 강건성을 향상시킵니다.

- **Performance Highlights**: 이 연구 결과에 따르면, UAMDP는 강화 학습 기반의 강력한 딥러닝 기준선에 비해 장기 예측 정확도를 개선했습니다. 구체적으로, 거래 Sharpe 비율은 1.54에서 1.74로 증가했고, 최대 손실(maximum drawdown)은 거의 절반으로 줄었습니다. 이 결과는 보정된 확률적 모델링, 탐험, 그리고 위험 인지 제어의 통합이 어떻게 safer하고 더 수익성 높은 순차적 의사결정에 기여하는지를 잘 보여줍니다.



### Post-hoc Stochastic Concept Bottleneck Models (https://arxiv.org/abs/2510.08219)
- **What's New**: 본 논문은 Post-hoc Stochastic Concept Bottleneck Models (PSCBMs)라는 가벼운 방법을 제안합니다. PSCBMs는 기존의 예측된 개념에 대해 다변량 정규 분포를 추가하여 모델 성능을 향상시킬 수 있습니다. 이 접근 방식은 전체 모델을 재훈련할 필요 없이 추가 모듈만으로 이루어집니다.

- **Technical Details**: PSCBMs는 기존의 Concept Bottleneck Models (CBMs)의 개념 예측기를 재사용하고, 경량화된 공분산 예측기를 추가하여 구성됩니다. 이 방법은 예측 성능을 높일 뿐만 아니라 개념 개입(intervention)의 효율성을 개선합니다. PSCBMs는 개념 의존성을 모델링하며, 확률 값을 샘플링하는 과정에서도 이를 반영할 수 있습니다.

- **Performance Highlights**: PSCBMs는 실세계 데이터에서 테스트할 때 표준 CBMs보다 개념 및 목표 정확도를 지속적으로 일치시키거나 개선하는 성능을 보입니다. 개념 간의 의존성을 모델링함으로써, PSCBMs는 개입 상황에서 CBMs보다 훨씬 더 나은 성능을 발휘합니다. 이 모델은 비효율적인 전체 재훈련 없이도 개입의 효과를 높여 줍니다.



### Expressive Value Learning for Scalable Offline Reinforcement Learning (https://arxiv.org/abs/2510.08218)
Comments:
          24 pages, 5 figures

- **What's New**: 이번 연구에서는 기존의 Offline Reinforcement Learning (RL) 방법들을 개선할 수 있는 새로운 접근법인 Expressive Value Learning for Offline Reinforcement Learning (EVOR)을 소개합니다. EVOR는 정책 증류(policy distillation)나 시간에 따른 역전파(backpropagation through time) 없이 확장 가능한 Offline RL을 개발하는 방식입니다. 또한, EVOR는 표현력이 뛰어난 가치 함수(value function)와 정책(policy)을 통합하여 훈련하며, 근본적인 데이터 셋에 대해 효과적인 최적화 및 정규화를 제공합니다.

- **Technical Details**: EVOR는 훈련 과정에서 흐름 맞추기(flow matching)를 통해 최적화된 정규화된 Q-함수를 학습합니다. 추론(inference) 시간에는 거부 샘플링(rejection sampling)을 활용하여 유연한 가치 함수에 따라 정책을 추출합니다. 이러한 방법은 기존의 백프로파게이션에서 발생하는 계산 비용을 피하며, 표현이 제한된 MLP 네트워크에 의존하는 대신 더 강력한 표현력을 갖춘 모델을 사용합니다. 또한, 표현력 있는 QQ-함수를 통해 최적의 솔루션을 제공하여, Offline RL의 스케일러빌리티를 향상시킵니다.

- **Performance Highlights**: EVOR의 성능은 다양한 Offline RL 작업에서 기존 기법을 초과하는 결과를 보여줍니다. 실험 결과, EVOR는 효율적인 최적화 및 계산 가능한 검색을 가능하게 하여, 다양한 데이터셋에 대해서도 안전하고 안정적인 학습이 가능함을 입증했습니다. 특히, 표현력 있는 가치 학습이 Offline RL에 통합될 경우, 정책의 성능을 크게 향상시킬 수 있다는 점이 강조되었습니다.



### FuelCast: Benchmarking Tabular and Temporal Models for Ship Fuel Consumption (https://arxiv.org/abs/2510.08217)
Comments:
          This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution will be published in "ECML PKDD Workshop 2025 - Advanced Analytics and Learning on Temporal Data"

- **What's New**: 이 논문은 해운 산업의 연료 소비 및 배출 예측을 위한 새로운 데이터셋과 벤치마크를 소개하고 있습니다. 연구진은 세 척의 선박에서 수집한 운영 및 환경 데이터를 포함하는 데이터셋을 공개하였으며, 이는 모델 평가 및 개발에 중요한 기초 자료가 됩니다. 또한, TabPFN 기반의 인컨텍스트 학습 방법을 해운 연료 예측에 처음으로 적용하여 데이터 효율적인 접근 방식을 모색합니다.

- **Technical Details**: 연구에서는 CatBoost, LSTM, TabPFN 등 세 가지 대표적인 모델 가족을 평가하였으며, 기존의 폴리노미얼 회귀 및 다층 퍼셉트론(MLP) 등의 기준선을 포함했습니다. 모델 성능 평가를 위해 사용하는 데이터셋은 다양한 선박 유형과 기후 데이터를 포함하며, 이는 실제 해운 작업과 관련된 고해상도 정보를 제공합니다. 프로젝트의 주요 목표는 기계 학습 연구 커뮤니티가 실질적이고 데이터 효율적인 방법으로 연료 소비 모델링을 할 수 있도록 지원하는 것입니다.

- **Performance Highlights**: 결과적으로, 연구에서 제시된 모델들은 환경 조건을 통합하여 기초적인 속도만에 의존하는 모델보다 일관되게 우수한 성능을 보였습니다. TabPFN 모델은 다른 기법들보다 약간 더 나은 결과를 나타냈는데, 이는 데이터 효율적인 테이블 예측에서 기초 모델의 잠재력을 강조합니다. 또한, 시간적 맥락을 포함하는 것이 정확성을 높이는 데 기여하는 것으로 확인되었습니다.



### Dual-granularity Sinkhorn Distillation for Enhanced Learning from Long-tailed Noisy Data (https://arxiv.org/abs/2510.08179)
Comments:
          25 pages, 2 figures

- **What's New**: 딥러닝 분야에서 클래스 불균형(class imbalance)과 레이블 노이즈(label noise)의 동시 존재로 인해 모델 성능이 저하되고 있습니다. 이 논문은 이러한 문제를 해결하기 위한 새로운 접근 방식을 제안하며, '약한(weak)' 보조 모델을 이용하여 서로 다른 문제에 대한 해결책을 종합적으로 활용하는 방법을 탐구합니다. 특히, 두 가지 문제는 서로 다른 수준에서 작용하기 때문에 각각에서의 강인성 메커니즘이 상호 보완적으로 작용할 수 있다는 통찰을 바탕으로 합니다.

- **Technical Details**: 이 논문에서 제안한 Dual-granularity Sinkhorn Distillation(D-SINK) 프레임워크는 보조 모델(auxiliary models)로부터 지식을 증류하고 통합하여 클래스 불균형과 레이블 노이즈에 대한 이중 강인성을 달성합니다. D-SINK는 최적 운송(Optimal transport) 기반의 대리 레이블 할당을 통해 타깃 모델의 예측을 노이즈에 강한 보조 모델과 샘플 수준에서 정렬하고, 클래스 분포는 불균형에 강한 보조 모델과 정렬합니다. 이를 통해 타깃 모델은 두 가지 문제를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: D-SINK는 다양한 노이즈 패턴과 비율을 가진 벤치마크 데이터셋에서 광범위한 실험을 수행하였으며, 상당한 성능 향상을 보여 주었습니다. 연구 결과, D-SINK는 약한 보조 모델의 지식을 통합함으로써 두 가지 도전 과제를 처리하는 데 있어 놀라운 효과를 발휘함을 확인했습니다. 이러한 접근 방식은 전통적인 방법론과는 다른 접급 방법을 제시하며, 복잡한 데이터 환경에서의 학습 성능을 효과적으로 향상시킬 수 있습니다.



### Long-tailed Recognition with Model Rebalancing (https://arxiv.org/abs/2510.08177)
- **What's New**: 이번 연구에서는 긴 꼬리(long-tail) 인식 문제를 해결하기 위해 MOdel REbalancing (MORE)라는 새로운 프레임워크를 제안합니다. 이 방법은 모델의 파라미터 공간을 직접 재조정하여 불균형을 완화하며, 기존의 데이터 증강이나 손실 재조정 방식을 보완합니다. MORE는 다중 클래스 및 다중 레이블 작업을 포함한 다양한 벤치마크에서 성능을 개선하는 결과를 보여줍니다. 특히, 소수 클래스(tail class)에 대한 일반화 능력을 크게 향상시키는 것으로 나타났습니다.

- **Technical Details**: MORE는 로우랭크(low-rank) 파라미터 구성 요소를 도입하여 파라미터 공간 할당을 제어하며, 클래스별 가중치(class-wise weighting)를 적용하여 소수 클래스에 집중하도록 유도합니다. 이 과정은 훈련 동안 손실을 동적으로 조정하는 사인파 기반의 재조정 스케줄을 통해 최적화되어, 전체 모델 복잡성이나 추론 비용을 증가시키지 않습니다. 이러한 접근법은 Rademacher 복잡성을 분석하여 소수 클래스의 모델 공간을 적절히 보존하는 것이 불균형 문제 해결에 효과적임을 증명합니다.

- **Performance Highlights**: MORE는 싱글 레이블(single-label) 및 멀티 레이블(multi-label) 환경 모두에서 긴 꼬리 인식 성능을 지속적으로 향상시킵니다. 다양한 데이터셋에 대한 실험 결과, MORE는 다수 클래스의 훈련을 유지하면서 동시에 소수 클래스에 대한 전담 용량을 확보하여 일반화 경계를 더욱 강화합니다. 이러한 결과는 MORE가 긴 꼬리 환경에서 강력한 플러그 앤 플레이 모듈로서의 가능성을 보여줍니다.



### Bidirectional Representations Augmented Autoregressive Biological Sequence Generation:Application in De Novo Peptide Sequencing (https://arxiv.org/abs/2510.08169)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이 논문에서는 CrossNovo라는 새로운 하이브리드 프레임워크를 제안하여 생물학적 시퀀스 생성에서 오토 회귀(AR) 모델과 비 오토 회귀(NAR) 모델의 장점을 통합하고 있습니다. 이 모델은 두 개의 디코더를 결합하며, 하나는 비 오토 회귀 디코더로 생물학적 특징을 학습하고, 다른 하나는 이러한 특징을 활용하여 시퀀스를 생성하는 오토 회귀 디코더입니다. 독특한 크로스 디코더 주의(attention) 모듈을 통해 두 디코더 간의 상호작용을 강화하여 예측 성능을 높이고 있습니다.

- **Technical Details**: CrossNovo의 아키텍처는 공유 스펙트럼 인코더와 두 개의 디코더로 이루어져 있습니다. 비 오토 회귀 디코더는 입력의 이차원 (bi-directional) 맥락 정보를 학습하고, 오토 회귀 디코더는 이 정보를 기반으로 생물학적 시퀀스를 합성합니다. 이 통합 과정은 중요성 감소(importance annealing) 및 크로스 디코더 기울기 차단(cross-decoder gradient blocking)을 포함한 맞춤형 훈련 전략을 통해 이루어집니다.

- **Performance Highlights**: CrossNovo는 9종 데이터셋 베이스라인에서 실험을 통해 오토 회귀 및 비 오토 회귀 모델을 대폭 초월하는 성능을 보여주었습니다. 이 모델은 오토 회귀의 안정성과 비 오토 회귀의 맥락 인식을 조화롭게 결합하여 다양한 후속 데이터에서 견고하고 뛰어난 성능을 발휘합니다. 이러한 연구 결과는 생물학적 시퀀스 모델링 기술을 진전시키고 복잡한 시퀀스 생성에 대한 새로운 건축 패러다임을 제시하는 데 기여합니다.



### Beyond Sub-6 GHz: Leveraging mmWave Wi-Fi for Gait-Based Person Identification (https://arxiv.org/abs/2510.08160)
- **What's New**: 이번 논문은 mmWave 주파수 대역과 sub-6 GHz 주파수 대역의 Wi-Fi 신호를 이용한 개인 식별(performance identification) 간의 첫 비교 연구를 제시합니다. 저자는 두 주파수 대역의 동기화된 측정 데이터를 활용하여 각 주파수 대역이 개인의 보행(gait)을 어떻게 구별할 수 있는지를 분석했습니다. 연구 결과, mmWave Wi-Fi 신호가 낮은 샘플링 속도에서도 높은 식별 정확도를 달성할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 고유한 다중 모드 채널 상태 정보(Channel State Information, CSI) 데이터 세트를 수집하여 20명의 참가자로부터 5 GHz와 60 GHz 신호를 동시 기록하였습니다. 데이터는 사전 처리 없이 원시 CSI 신호를 심층 신경망(deep neural networks)에 직접 입력하는 방식의 end-to-end 학습 방식을 채택했습니다. 또한, 배경 차감(background subtraction) 기법을 통해 운동 유도 변화를 강조하고 환경 잡음을 억제했습니다.

- **Performance Highlights**: 실험 결과, mmWave Wi-Fi 신호를 사용했을 때 20명의 식별 시 높은 정확도(91.2%)를 보여주며, 이는 기존의 sub-6 GHz Wi-Fi 신호보다 우수한 성능을 나타냅니다. 저자는 다양한 심층 학습 모델을 비교하여 두 기술 간의 트레이드오프를 분석하였습니다. 이 연구는 상업용 하드웨어를 사용하여 mmWave 기반의 개인 식별 연구에서 중요한 기여를 하고 있습니다.



### Unsupervised Multi-Source Federated Domain Adaptation under Domain Diversity through Group-Wise Discrepancy Minimization (https://arxiv.org/abs/2510.08150)
- **What's New**: 이번 연구에서는 다수의 다양한 출처에서 레이블 데이터의 활용을 통해 비지도 다중 출처 도메인 적응(UMDA) 문제를 해결하기 위해 GALA라는 새로운 프레임워크를 제안합니다. GALA는 두 가지 주요 구성 요소인 그룹 간 불일치 최소화 목표와 온도 조정 기반의 중심점 가중치 전략을 도입하여, 다양한 소스 도메인 간의 정렬을 효율적으로 구현합니다. 이를 통해 기존 방법들이 직면한 스케일링 문제를 해결하고, 안정적인 훈련을 가능하게 합니다.

- **Technical Details**: GALA는 다중 출처의 비지도 도메인 적응을 위해 고안된 연합 학습 프레임워크로, 개별 출처의 예측을 그룹 단위로 정렬하여 전체 관련 목표를 효율적으로 근사합니다. 여기에 온도 스케일링을 통해 각 출처의 목표 도메인과의 정렬도를 동적으로 평가하는 가중치 메커니즘이 추가되어, 소스의 다양성이 증가하더라도 훈련을 안정적으로 유지합니다. 이 연구에서는 18개의 숫자 데이터 세트로 구성된 Digit-18 벤치마크를 도입하여 다양한 도메인 간 변화를 평가합니다.

- **Performance Highlights**: 실험 결과 GALA는 표준 UMDA 벤치마크에서 경쟁력 있는 성과를 지속적으로 달성하며, 높은 다양성을 가진 출처 설정에서도 기존 방법론들을 초월하는 성능을 보여줍니다. 이는 안정성과 강인함을 유지하며, 기존 방법들이 수렴하지 못하는 다양한 다중 출처 환경에서 더 나은 결과를 나타냅니다. 이를 통해 GALA는 효과적이고 실용적인 비지도 다중 출처 도메인 적응 프레임워크로 자리매김할 것입니다.



### Think Just Enough: Sequence-Level Entropy as a Confidence Signal for LLM Reasoning (https://arxiv.org/abs/2510.08146)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)에서의 추론 과정에서 토큰 효율성을 향상시키기 위한 새로운 엔트로피 기반 프레임워크를 제안합니다. 이 양식은 Shannon 엔트로피를 사용하여 조기 중지를 가능하게 하며, 25-50%의 계산 비용 절감을 달성하면서도 정확도를 유지합니다. 특히, 엔트로피 기반 신뢰도 보정이 현대 추론 모델에서 나타나는 새로운 특성을 나타내며, 기존의 표준 모델에서는 발견되지 않는 점이 주목할 만합니다.

- **Technical Details**: 프레임워크는 Shannon 엔트로피를 신뢰도 신호로 활용하여, 각 모델에 대해 추론을 중단할 수 있는 임계값을 쉽게 계산할 수 있도록 합니다. 또한, 이 접근법은 정보 이론과 통계적 의사결정 이론에 기반하여 이론적인 엄밀성과 실용적인 적용 가능성을 제공합니다. 제안된 메서드는 고유의 4가지 수학적 임계값 방법을 통해 조기 중지를 위한 신뢰도 추정을 질적으로 개선합니다.

- **Performance Highlights**: 논문에서 제시한 결과는 다양한 현대 추론 최적화 모델 카테고리에 걸쳐 25-50%의 계산 비용 절감을 보여주며, 이는 정확도를 유지하면서 더 큰 효율성을 나타냅니다. 다양한 추론 벤치마크에서 신뢰도와 관련된 메커니즘의 일관된 성능을 입증하며, 저자들은 신뢰도 인식이 현대 추론 시스템의 중요한 특성임을 강조합니다.



### Arbitrary Entropy Policy Optimization: Entropy Is Controllable in Reinforcement Finetuning (https://arxiv.org/abs/2510.08141)
- **What's New**: 본 연구에서는 대규모 언어 모델(LLM)의 추론 능력을 개선하기 위한 필수 기술인 강화 학습 파인튜닝(Reinforcement Fine-Tuning, RFT)에 대한 새로운 방법인 임의 엔트로피 정책 최적화(Arbitrary Entropy Policy Optimization, AEPO)를 제안합니다. 기존 방법인 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)는 엔트로피 붕괴 문제로 인해 탐색을 제한하는 단점을 가지고 시스템 전체에서 엔트로피를 정밀하게 조정하는 알고리즘이 부족했습니다.

- **Technical Details**: AEPO는 엔트로피 보너스를 REINFORCE 정책 그래디언트로 대체하고, 온도 조정 분포를 통해 엔트로피를 안정화하여 훈련 과정에서 엔트로피가 특정 목표 레벨로 진동하는 것을 보장합니다. AEPO는 정책 그래디언트, 분포, REINFORCE를 정규화로 사용하는 세 가지 주요 설계를 통합하여 최적화를 왜곡하지 않고도 엔트로피를 정밀하게 제어합니다.

- **Performance Highlights**: 실험을 통해 AEPO가 (1) GRPO에서 엔트로피 붕괴를 효과적으로 제거하고, (2) 엔트로피 증가에 따른 성능의 비선형적 관계를 밝혀내며, (3) 엔트로피를 넘어 더 넓은 RFT 패러다임을 제공하여 우수한 목표 분포가 REINFORCE 정규화기로 작용할 수 있다는 점을 입증했습니다.



### Approximate Domain Unlearning for Vision-Language Models (https://arxiv.org/abs/2510.08132)
Comments:
          NeurIPS 2025 (Spotlight)

- **What's New**: 이 논문에서는 Approximate Domain Unlearning (ADU)이라는 새로운 문제 설정을 도입합니다. 기존의 클래스 언러닝(class unlearning)은 특정 개체 클래스의 인식을 줄이는 데 집중했지만, ADU는 지정된 도메인의 이미지에 대한 인식 정확도를 낮추면서 다른 도메인의 정확도를 유지해야 하는 요구사항을 다룹니다. 이로 인해 기존의 접근 방식이 가지는 한계와 더불어 새로운 기술적 도전과제를 제시하며, 실용적인 언러닝의 필요성을 강조합니다.

- **Technical Details**: ADU 문제 해결을 위해 Domain Disentangling Loss (DDL)이라는 방법론을 제안하여 잠재 공간(latent space)에서 도메인 분포를 명확히 분리합니다. 또한, Instance-wise Prompt Generator (InstaPG)를 도입하여 이미지별로 도메인 특성을 적응적으로 모델링합니다. 이는 강력한 도메인 일반화 능력을 가진 사전 훈련된 VLM에서 발생하는 도메인 분포의 얽힘(entanglement) 문제를 해결하기 위한 접근법입니다.

- **Performance Highlights**: 다양한 다중 도메인 이미지 벤치마크 데이터셋에 대한 실험 결과, 제안된 방법은 최신 VLM 조정 기술을 기반으로 한 강력한 베이스라인을 능가하는 성능을 보여줍니다. 이는 ADU 문제 해결을 위한 기존 방법들에 비해 명백한 개선을 나타내며, VLMs에서 실용적이고 세밀한 언러닝을 위한 가능성을 제시합니다.



### Bayesian Decision Making around Experts (https://arxiv.org/abs/2510.08113)
- **What's New**: 이 논문은 전문가와 함께 작동하는 복잡한 학습 에이전트에 대한 연구를 다룹니다. 특히, Bayesian multi-armed bandits 맥락에서, 오프라인과 동시 설정의 경우에 대해 전문가 데이터가 학습자의 사후 확률(posterior)에 영향을 미치는 방식을 공식화합니다. 우리는 전문가의 최적 정책 및 그 결과로부터 얻은 데이터를 활용하여 정보를 효율적으로 통합하는 방법을 제안합니다.

- **Technical Details**: 논문에서는 학습자가 자신의 경험 또는 전문가의 결과를 기반으로 신념을 업데이트해야 할지를 결정하는 방법을 연구합니다. 오프라인 설정에서는, 전문가의 과거 결과를 포함하는 데이터셋을 통해 학습자가 사전 신념(prior)을 초기화(warm-start)하고, 동시 설정에서는 어떤 데이터 소스를 선택할지를 정보 이득 기준으로 결정합니다. 이를 통해 Thompson Sampling 알고리즘의 성능 향상을 위한 정보 이론적 규약이 제공됩니다.

- **Performance Highlights**: 이 연구는 전문가의 정보를 활용하여 Bayesian 학습 에이전트가 언제 신뢰하고 배워야 할지를 결정하는 메타 문제를 해결하는 방법을 제시합니다. 우리의 알고리즘은 전문가의 결과가 최적 행동 분포에 미치는 영향을 정량화하며, 실험 결과에서 강력한 비대칭 세계에서의 후회(regret) 개선을 보여줍니다. 궁극적으로 이 연구는 여러 Bayesian 학습자가 공존하는 상황에서의 이론적 이해를 심화시키고 향후 에이전트 설계를 위한 강력한 프레임워크를 제시하는 것을 목표로 합니다.



### Mitigating Subject Dependency in EEG Decoding with Subject-Specific Low-Rank Adapters (https://arxiv.org/abs/2510.08059)
- **What's New**: 본 논문에서는 EEG 해독을 위한 기초 모델 개발에서의 주요 장애물인 주제별 분포 변화를 해결하기 위해 주제 조건화된 레이어(Subject-Conditioned Layer)를 제안합니다. 이 레이어는 신경망 구조의 표준 선형(layer) 또는 합성곱(convolutional) 레이어의 대체로 설계됐으며, 주제 간 변동성을 포착하기 위해 가중치를 주제 불변의 공유 성분과 각 주제 고유의 저랭크(low-rank) 수정으로 분해합니다. 이를 통해 기존 모델들이 주제 변화에 강인해질 수 있도록 합니다.

- **Technical Details**: 주제 조건화된 레이어는 주어진 신경망의 매개변수를 두 가지 구성요소로 분해하는 접근 방식을 취합니다: 모든 주제에 걸쳐 공유되는 기본 가중치 세트와 각 주제에 특화된 저랭크 수정입니다. 이 레이어는 모든 신경망 구조에서 쉽게 교체 가능하며, EEG 해독을 포함한 다양한 분야에 응용할 수 있습니다. 이 방법은 비선형 혼합 효과 모델링(nonlinear mixed-effects modeling) 및 다중 과제 학습(multi-task learning)과 밀접한 관련이 있으며, 주제를 인식하는 특수한 레이어로 해석될 수 있습니다.

- **Performance Highlights**: BCI Competition IV 데이터셋에 대한 포괄적인 평가를 통해, 주제 조건화된 레이어를 장착한 CNN 및 ViT와 같은 아키텍처가 강력한 기준 모델인 주제 무관 모델 및 주제별 모델 평균을 초과하는 성과를 보였습니다. 이는 제안된 방식의 성공 메커니즘을 확인하는 질적 증거와 함께, 보다 효과적인 다중 주제 문제 해결을 위한 실제적이고 확장 가능한 경로를 보여줍니다.



### From Tokens to Layers: Redefining Stall-Free Scheduling for LLM Serving with Layered Pref (https://arxiv.org/abs/2510.08055)
Comments:
          13 pages, 5 figure, 8 tables

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 효율적인 서빙을 위한 새로운 스케줄링 패러다임인 레이어된 프리필(layered prefill)을 제안합니다. 기존의 청크된 프리필(chunked prefill) 방식은 메모리 트래픽과 에너지 소모를 증가시키는 단점을 가지고 있으며, 레이어된 프리필은 이러한 문제를 해결하면서 오프칩 대역폭 수요를 줄입니다. 이를 통해 TTFT(첫 번째 토큰 생성까지의 시간)는 최대 70% 감소하고, 엔드 투 엔드 지연 시간(End-to-End latency)은 41% 감소하며, 토큰당 에너지도 최대 22% 감소합니다.

- **Technical Details**: 레이어된 프리필은 변환기 레이어 그룹을 주 스케줄링 단위로 처리하며, 모델을 연속적인 레이어 그룹으로 수직 분할하고 이들 그룹 간에 프리필과 디코드를 중첩시킵니다. 이는 기본적으로 모듈 간의 중복 메모리 접근을 방지하고, 각 레이어를 한 번만 탐색하게 해주어, MoE(전문가 혼합) 모델에서 비효율적인 반복적인 전문가 가중치 로드를 제거합니다. 이 방식은 LLM 서빙에서 TTFT, TBT(토큰 간 시간) 및 처리량 사이의 균형을 개선합니다.

- **Performance Highlights**: 평가 결과, 레이어된 프리필 방식은 청크된 프리필에 비해 TTFT-TBT 페레토 경계를 지속적으로 개선하며, 전문가 로드 트래픽과 에너지 비용을 줄이면서 지연 없는 디코딩을 유지합니다. 또한, 이 방법은 고성능, 에너지 효율적인 LLM 서빙을 위한 새로운 운영 체제를 개방하는 방향으로 제시됩니다. 전반적으로, 처리 축을 토큰에서 레이어로 변경함으로써 LLM 서빙의 효율성이 크게 향상되었습니다.



### Do We Really Need Permutations? Impact of Width Expansion on Linear Mode Connectivity (https://arxiv.org/abs/2510.08023)
- **What's New**: 이 논문에서는 독립적으로 훈련된 두 개의 모델 간의 연결성을 높이기 위해 매개변수를 변형할 필요 없이, 단순히 모델의 크기를 늘리는 것만으로도 Linear Mode Connectivity (LMC)를 달성할 수 있음을 실증적으로 보여주고 있습니다. 특히 적절한 softmax 온도 조정이 있을 때, 모델의 폭을 넓히는 것만으로도 모델 간의 연결성을 높일 수 있습니다. 또한, 각 레이어의 출력을 분석하여 이를 설명하는 새로운 개념인 Layerwise Exponentially Weighted Connectivity (LEWC)를 도입하였습니다.

- **Technical Details**: LEWC는 병합된 모델의 각 레이어의 출력을 원래 모델의 해당 레이어 출력의 지수 가중합으로 표현할 수 있다고 주장합니다. 이는 병합된 모델의 출력이 원래 모델들의 앙상블처럼 작동하도록 만든다고 설명합니다. 이 연구에서는 추론 정확성을 높이기 위해 모델의 크기를 늘리는 것이 비선형 경로는 물론 선형 경로에서의 연결성에도 중요한 역할을 한다고 제안하고 있습니다.

- **Performance Highlights**: 결과적으로, 모델 폭을 충분히 늘이는 것만으로도 독립적으로 훈련된 모델들의 가중치를 평균화하면 원래 모델과 같은 정확도를 달성할 수 있음을 보여줍니다. 또한 적절한 반전 온도로 softmax를 조정함으로써 손실 장벽을 거의 제로로 낮출 수 있는 방법도 제시하고 있습니다. 이는 LMC를 달성하는 데 있어 모델 폭의 증가가 매우 중요한 역할을 한다는 것을 의미합니다.



### Backdoor Vectors: a Task Arithmetic View on Backdoor Attacks and Defenses (https://arxiv.org/abs/2510.08016)
Comments:
          22 pages, 13 figures, 15 tables

- **What's New**: 이 논문에서는 최근 모델 머징(Model Merging) 기법에 관한 연구를 수행하고, 이를 통해 발생할 수 있는 보안 위험, 특히 백도어 공격(Backdoor Attacks)의 취약점에 대한 새로운 통찰을 제시합니다. 우리는 백도어 벡터(Backdoor Vector)라는 개념을 도입하여, 공격의 특성을 이해하고 측정하는 효과적인 방법을 제공합니다. 또한, 논문에서는 새로운 방어 기법인 주입된 백도어 벡터 감소(Injection BV Subtraction, IBVS)를 제안하여, 기존의 백도어 공격을 방지하기 위한 접근 방식을 개선했습니다.

- **Technical Details**: 모델 연합 과정에서는 CLIP 모델을 사용하여, 시각적 인코더(Visual Encoder)와 텍스트 인코더(Text Encoder)의 조합을 통해 다양한 모델을 통합합니다. 각 모델의 가중치 차이를 기반으로 백도어 벡터(BV)를 산출하여, 이 벡터가 공격 또는 방어 전략의 비교 및 분석에 유용하다는 것을 보여줍니다. 새로운 희소 백도어 벡터(Sparse Backdoor Vector, SBV) 방법을 통해 여러 개의 백도어 벡터가 결합하여 보다 강력한 공격을 형성하는 방식을 제안하고, 이는 모델 머징 과정에서 더 높은 공격 성공률(Attack Success Rate, ASR)을 제공합니다.

- **Performance Highlights**: 이 연구의 결과는 희소 백도어 벡터(SBV)가 기존 공격 방법보다 뛰어난 효과를 보이며, 백도어 공격의 성공률을 크게 증가시킬 수 있다는 것을 나타냅니다. 또한, IBVS 방어 기법은 알려지지 않은 백도어 위협에 대해서도 효과적으로 작동하여, 경량화된 일반 방어 방법을 제공합니다. 이러한 발견은 모델 머징 과정에서의 내부 취약점과 백도어 공격의 면밀한 관계를 이해하는 데 기여하며, 향후 기억하기 쉬운 방어 기법 개발에 도움이 될 것입니다.



### Unsupervised Radio Map Construction in Mixed LoS/NLoS Indoor Environments (https://arxiv.org/abs/2510.08015)
- **What's New**: 이 논문은 기존의 위치 보정(location calibration) 없이 채널 전파(channel propagation) 순서에서 데이터 수집 경로를 복구하는 새로운 방법을 제안합니다. 주요 아이디어는 숨겨진 마르코프 모델(hidden Markov model, HMM)을 활용하여 채널 전파 행렬을 조건부로 모델링하고 동시에 사용자 경로의 위치 상관관계를 모델링하는 것입니다. 이 방법은 MIMO(다중 입력 다중 출력) 네트워크에서의 복잡한 채널 전파의 관계를 모델링하고, 선형 경로(line-of-sight)와 비선형 경로(non-line-of-sight) 조건을 모두 다룰 수 있다는 점에서 차별화됩니다.

- **Technical Details**: 제안된 HMM 기반 프레임워크는 MIMO 네트워크 내에서 각 경로에 대한 전파 모델을 개별적으로 전력(power), 지연(delay), 각도(angle) 측면에서 모델링합니다. 또한, 사용자 경로는 가우시안-마르코프 모델을 사용하여 시간-공간적 관계를 포착합니다. 본 연구에서는 교대 최적화(alternating optimization)를 통해 전파 매개변수, 이동성 모델, LOS/NLOS 분류를 동시에 최적화합니다.

- **Performance Highlights**: 실험적 검증은 MIMO-OFDM 네트워크에서 수행되었으며, 결과적으로 제안된 방법은 실내 환경에서 평균 0.65미터의 로컬라이제이션(定位) 정확도를 달성했습니다. 제안된 방법은 기존의 각도 또는 전력 기반 로컬라이제이션 기법 및 채널 차트 방법보다 향상된 성능을 보여줍니다. 또한 구축된 라디오 맵을 활용하여 평균 로컬라이제이션 오차가 1미터 이하로 유지되며 DNN, KNN, SVM과 같은 기존 지도학습 방법을 조금 초과하는 성능을 보여주었습니다.



### Accelerated Evolving Set Processes for Local PageRank Computation (https://arxiv.org/abs/2510.08010)
- **What's New**: 이 연구는 개인화된 페이지랭크(Personalized PageRank) 계산 속도를 높이기 위한 새로운 프레임워크인 Accelerated Evolving Set Process (AESP)를 제안합니다. 이 프레임워크는 중첩된 진화 집합 프로세스(nested evolving set processes)를 기반으로 하여, 간소화된 선형 시스템을 해결하기 위해 지역화된 부정확 근접 점(iteration) 방법을 사용합니다. 이를 통해 새로운 알고리즘은 기존 선형 시스템 해결 방식보다 빠른 수렴 속도를 보여줍니다.

- **Technical Details**: AESP는 각 단계에서 ℓ1-노름의 감소를 보장하면서 최적화 문제를 푸는 로컬 방법에 의거합니다. 이 프레임워크는 변형 공식의 PPR(Personalized PageRank) 계산을 포함하여, 수렴 속도가 빠르며 시간 복잡도는 $ 	ilde{	ext{{O}}}(R^2/(	ext{{	extalpha}}	ext{{	extepsilon}}^2)) $ 형태로 설명됩니다. 연구 결과는 기존의 열린 추측을 해결하며, 다양한 실제 그래프에서 실험적으로 경량화된 성능을 입증했습니다.

- **Performance Highlights**: 실험 결과는 AESP 방법이 초기 단계에서 상당한 수렴성을 보여준다는 것을 확인했습니다. AESP 알고리즘은 추가적인 가정 없이 수렴을 보장하며, 기존 문헌에서 예측된 가속화된 경계와 일치하는 시간 복잡도를 달성합니다. 이러한 성능 개선은 R, m과 같은 그래프의 속성의 의존도를 반영하며, 개인화된 페이지랭크의 효율적인 계산에 기여할 수 있습니다.



### Recycling Pretrained Checkpoints: Orthogonal Growth of Mixture-of-Experts for Efficient Large Language Model Pre-Training (https://arxiv.org/abs/2510.08008)
- **What's New**: 이 논문은 큰 언어 모델(LLM)의 사전 훈련 비용을 효율적으로 재활용하는 새로운 방법을 제안합니다. 기존 체크포인트를 재사용하여 매개변수를 확장하고 훈련을 지속하는 '체크포인트 재활용' 방안을 통해 더 낮은 비용으로 성능을 향상시키고자 합니다. 기존 연구에서 제안된 방법과는 달리, 이 논문은 완전히 수렴된 모형에 대해 성장 전략을 탐구하며, 특히 Mixture-of-Experts (MoE) 아키텍처에 중점을 두고 있습니다.

- **Technical Details**: 저자들은 '상대적인 성장(Orthogonal Growth)' 방법론을 통해 모델의 깊이와 폭을 확장하는 두 가지 전략을 제시합니다. 깊이 성장은 레이어를 복제하여 확장하는 방법으로, 기존의 '스태킹' 방식보다 나은 성능을 보장하는 '인터포지셔널' 방법을 제안합니다. 폭 성장 전략에서는 새로운 전문가 모델에 약간의 노이즈를 주입함으로써 전문가의 전문성을 높여줍니다. 이를 통해 체크포인트의 잉여 비용을 최대한 활용할 수 있는 최적의 타이밍을 연구하였습니다.

- **Performance Highlights**: 제안한 방법론은 17억 개 매개변수를 가진 MoE 모델을 70억 개 매개변수로 확장하며, 1조 개 토큰으로 훈련되어 성능 향상에서 10.66%의 정확도 개선을 달성하였습니다. 이는 같은 추가 FLOPs 예산으로 원시 훈련에서 나오는 모델보다 우수한 성능을 입증합니다. 최종적으로, 저자들은 체크포인트 재활용 접근 방식이 경제적으로 효율적인 LLM 사전 훈련의 기초를 마련하고 있음을 강조합니다.



### DemandCast: Global hourly electricity demand forecasting (https://arxiv.org/abs/2510.08000)
Comments:
          7 pages, 4 figures, accepted at the NeurIPS 2025 Workshop: Tackling Climate Change with Machine Learning

- **What's New**: 이 논문은 XGBoost(그라디언트 부스팅 알고리즘)를 사용하여 다양한 지리적 지역에서 전력 수요 예측을 위한 머신러닝 프레임워크를 제안합니다. 본 모델은 역사적 전력 수요와 날씨 및 사회경제적 변수를 통합하여 보정된 전력 수요 프로파일을 예측합니다. 이를 통해 대규모 데이터셋을 개발하고, 샘플 외 성능을 검증할 수 있는 시간적 데이터 분할 전략을 적용하여 효율적인 훈련과 평가를 가능하게 합니다.

- **Technical Details**: DemandCast 프레임워크는 역사적 전력 수요, 기상 변수 및 사회경제적 지표를 통합하여 구성됩니다. 데이터 세트는 다양한 출처에서 수집되며, 모듈화된 데이터 파이프라인을 통해 전처리 및 조화됩니다. 데이터의 높은 품질 및 해상도가 제한적인 경우를 위해 열린 접근 방식의 'Awesome Electricity Demand' 저장소를 개발하여 공공의 전력 수요 데이터에 대한 접근성을 개선했습니다.

- **Performance Highlights**: 테스트 세트에서 XGBoost 모델은 6,041,222개의 시간 전력 수요 관측치를 기반으로 평균 절대 백분율 오차(MAPE) 9.2%를 달성했습니다. 이는 이전 연구와 비교했을 때 상당히 유사한 결과로, 모델이 지역별 수요 패턴을 복원하는 능력과 예측 정확도의 변동성을 보여줍니다. 앞으로 예측 불확실성을 수량화할 수 있는 대체 아키텍처 탐색과 하이퍼파라미터 최적화 전략이 모델 성능을 향상시킬 것으로 기대됩니다.



### Fewer Weights, More Problems: A Practical Attack on LLM Pruning (https://arxiv.org/abs/2510.07985)
- **What's New**: 이번 연구는 모델 프룬링(model pruning) 방법이 어떻게 악의적으로 악용될 수 있는지를 처음으로 보여줍니다. 기존의 프룬링 방법들은 효율성을 높여왔지만, 보안 문제는 충분히 다루어지지 않았습니다. 연구 결과에 따르면, 적대자는 프룬링을 통해 모델의 악의적 행동을 유발할 수 있는 방법을 찾을 수 있습니다.

- **Technical Details**: 이 연구에서는 적대자가 각 매개변수가 pruning 될 가능성을 추정할 수 있는 프록시 메트릭(proxy metric)을 계산하여 공격을 설계하는 방법을 사용하였습니다. 이를 통해 적대자는 낮은 확률로 pruning 될 매개변수에 악의적 행동을 주입하고, 높은 확률로 pruning될 매개변수를 사용하여 모델을 복구함으로써 최종적으로 수정된 모델에서 악의적 행동을 나타내게 합니다.

- **Performance Highlights**: 이 연구에서는 vLLM에서 적용된 다양한 프룬링 방식(예: Magnitude, Wanda, SparseGPT)으로 인한 강력한 악의적 행동의 성공률을 평가하였습니다. jailbreak의 경우 최대 $95.7	ext{%}$, 정상적인 지시 거부의 경우 $98.7	ext{%}$, 목표한 내용 삽입의 경우 $99.5	ext{%}$의 성공률을 기록했습니다. 이러한 결과는 모델 압축에서 보안 인식의 필요성을 강조하고 있습니다.



### Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training (https://arxiv.org/abs/2510.07980)
Comments:
          This paper has been accepted by NeurIPS 2025 (Spotlight)

- **What's New**: 이 논문은 기존의 중앙 집중식 훈련과 비교하여 분산 훈련의 성과가 저조할 수 있는 상황에서 Multi-Gossip Steps (MGS)가 성능 격차를 줄이는 효과적인 방법임을 다룹니다. MGS의 최적화 오류를 빠르게 감소시키는 능력이 이론적으로 검증되었으며, 분산 훈련에 대한 새로운 통찰력을 제공합니다. 특히, 학습률, 데이터 이질성, 클라이언트 수, 통신 토폴로지 등이 MGS의 일반화에 미치는 영향에 대한 통합 분석을 제공합니다.

- **Technical Details**: MGS는 이론 분석을 통해 MGS가 최적화 오류 및 일반화 오류를 경량화하는 방식을 제시합니다. 논문에서는 중앙 집중식 미니 배치 SGD와 분산 네트워크의 일반화 오류를 비교하며, 각 환경에서의 오류 경계가 어떻게 되는지를 명확히 하고 있습니다. 또한 다양한 실험 조건 아래에서 MGS의 성능을 검증하는 실험 결과를 제시하고, 이러한 결과들이 이론적 분석과 일치함을 보였습니다.

- **Performance Highlights**: CIFAR 데이터셋에서의 실험 결과는 MGS의 효과를 분명히 보여줍니다. 통신 토폴로지, MGS 스텝 수, 클라이언트 수 등 주요 하이퍼파라미터의 변화가 오류 감소에 긍정적인 영향을 미쳤으며, 이는 이론적 결과와도 일치합니다. 이러한 연구 결과는 분산 최적화의 이론적 이해를 크게 발전시키며, 프랙티셔너에게 유익한 통찰을 제공합니다.



### Climate Surrogates for Scalable Multi-Agent Reinforcement Learning: A Case Study with CICERO-SCM (https://arxiv.org/abs/2510.07971)
- **What's New**: 이번 논문에서는 다중 온실가스의 영향을 기반으로 한 기후 정책 탐사를 위한 새로운 다중 에이전트 강화 학습(MARL) 프레임워크를 제안합니다. 이 프레임워크는 효과적인 기후 대체 모델을 사용하여 지역 에이전트들이 기후 정책을 학습할 수 있도록 환경 주기에 통합합니다. 이를 통해 기존 모델에 비해 훈련 시간과 계산 비용을 획기적으로 줄일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 MARL 프레임워크는 고충실도(high-fidelity) 재현 모델을 환경 루프에 직접 통합하여 작동합니다. 이 모델은 연속적인 재귀 신경망(recurrent neural network) 아키텍처를 사용하여 20,000개의 다중 가스 배출 경로를 사전학습(pretrained)했으며, 원래 기후 모델인 CICERO-SCM과 유사한 성능을 발휘합니다. 그 결과, 기후 모델 대체 모델의 평균 온도 RMSE(root mean square error)는 약 0.0004 K로 나타났고, 실행 속도는 약 1000배 빨라졌습니다.

- **Performance Highlights**: 제안된 대체 모델은 기후 정책 MARL 설정에서 원래의 시뮬레이터를 대체할 수 있으며, 이는 전체 훈련 시간을 100배 이상 단축시키는 효과를 가져옵니다. 에이전트들이 동일한 최적 정책에 수렴하는 것을 보였으며, 시뮬레이터를 사용할 수 없는 경우에도 정책의 일관성을 평가할 수 있는 방법론을 제시합니다. 이 연구는 기후 정책의 직접적인 실험과 모델링의 복잡성을 줄이면서도 정책 신뢰도를 유지할 수 있는 가능성을 열어 줍니다.



### PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation (https://arxiv.org/abs/2510.07964)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: 이 논문은 유전자 교란 예측을 위한 새로운 프레임워크인 PRESCRIBE를 제안합니다. PRESCRIBE는 Bayesian Estimation을 활용하여 데이터 불확실성과 모델 불확실성을 동시에 측정할 수 있는 다변량 딥 에비던스 회귀 프레임워크입니다. 이 연구는 예측의 신뢰도를 측정할 수 있는 신뢰도 점수를 효과적으로 추정하는 능력을 보여줍니다.

- **Technical Details**: PRESCRIBE는 두 가지 주요 요소로 구성됩니다: 전사체(transcriptomic) 지형에 대한 사후 분포(posterior distribution)와 교란 공간의 학습된 잠재 밀도에서 도출된 증거 점수(evidence score)입니다. 이를 통해 예측된 분포의 확산(엔트로피를 이용하여 측정)을 통해 데이터 불확실성을 정량화하고, 증거 점수를 통해 모델 불확실성을 정량화합니다. 이 프레임워크는 높은 증거 점수를 갖는 예측이 훈련 데이터 내에서 여러 기능적으로 관련된 유전자 교란 사례에 의해 뒷받침된다는 것을 의미합니다.

- **Performance Highlights**: PRESCRIBE는 실험을 통해 기존 기법들과 비교했을 때 3% 이상의 지속적인 정확도 향상을 달성했습니다. 이 시스템은 신뢰할 수 없는 결과를 걸러내는 데 도움을 주며, 세포 교란에 대한 예측 정확도를 개선하는 데 기여하고 있습니다. 이는 숙련된 예측자 생성과 동시에 여러 교란에 대한 정확한 셀 반응 예측 방법으로서의 가치를 강조합니다.



### DISCO: Diversifying Sample Condensation for Efficient Model Evaluation (https://arxiv.org/abs/2510.07959)
- **What's New**: 본 논문은 현대 기계 학습 모델의 평가가 막대한 비용을 요구하는 문제를 다루고 있습니다. 기존의 평가 방법은 모델 정확도와 최종 테스트 결과 간의 매핑을 학습하기 위해 정적 서브셋(anchor subset)을 선택하는 방식으로, 이 과정은 클러스터링에 의존하기 때문에 복잡하고 민감합니다. 새로운 방법인 Diversifying Sample Condensation (DISCO)는 모델의 응답 다양성을 극대화하는 샘플을 선택함으로써 평가 과정을 단순화합니다.

- **Technical Details**: DISCO는 그리디(greedy) 샘플 통계치를 사용하여 모델 간의 불일치를 최대화하는 top-k 샘플을 선택합니다. 이 접근 방법은 복잡한 클러스터링을 사용하지 않고도, 평가 성능을 극대화할 수 있는 이론적 근거를 제공합니다. DISCO의 성능은 다양한 도메인에서 평가되었으며, MMLU와 같은 데이터셋에서 99.3%의 평가 비용 절감을 기록했습니다.

- **Performance Highlights**: DISCO는 기존의 평가 방법들에 비해 뛰어난 효율성과 정밀도를 보여줍니다. 예를 들어, Anchor Points, TinyBenchmarks, Metabench와 같은 이전 방법들과 비교했을 때, DISCO는 성능 예측에서 최첨단 결과를 달성했습니다. 이를 통해 현대 기계 학습 모델의 평가 비용을 대폭 줄이고 환경적 영향을 최소화하는 데 기여할 수 있습니다.



### Some theoretical improvements on the tightness of PAC-Bayes risk certificates for neural networks (https://arxiv.org/abs/2510.07935)
- **What's New**: 이번 논문은 PAC-Bayes 경계에 기반한 신경망의 위험 인증서 사용성을 개선하는 네 가지 이론적 기여를 제시합니다. 첫 번째로, Bernoulli 분포 간의 KL divergence에 대한 두 개의 경계를 도입하여 다양한 경험적 위험 범위에 걸쳐 분류기의 실제 위험에 대한 가장 긴밀한 경계를 도출합니다. 두 번째로, 암묵적 미분을 기반으로 한 효율적인 방법론을 형식화하여 PAC-Bayesian 위험 인증서의 최적화를 네트워크/모델 적합에 사용되는 손실/목적 함수 내부에 도입합니다.

- **Technical Details**: 논문에서는 n개의 샘플 집합 S={(Xi,Yi)}i=1n를 고려하며, 목표는 주어진 하이포시스 공간 ℋ에서 좋은 근사를 제공하는 매핑 h:X→Y를 찾는 것입니다. KL divergence를 사용하여 두 분포 간의 관계를 측정하며, 여기에서 Bernoulli 분포의 경우를 다룹니다. PAC-Bayesian 프레임워크는 데이터에 독립적인 prior 분포와 데이터에 의존하는 posterior 분포를 정의하여 일반화 능력을 추정합니다.

- **Performance Highlights**: MNIST 및 CIFAR-10 데이터 세트에 대한 실증적 평가를 통해, 이 논문은 신경망에 대한 CIFAR-10의 비공식적인 일반화 경계를 최초로 제시합니다. 이로 인해 PAC-Bayes 인증서가 더 복잡한 문제를 해결하는 데 도움이 되며, 손실 함수가 미분 가능하지 않을 때 경계를 최적화하는 새로운 방법이 도출됩니다. 논문에서 제안한 새로운 방법론은 현대 신경망 아키텍처의 복잡성을 극복하고 더 강력한 성능과 견고한 경계를 제공합니다.



### Synergy Between the Strong and the Weak: Spiking Neural Networks are Inherently Self-Distillers (https://arxiv.org/abs/2510.07924)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이번 연구에서는 스파iking Neural Networks (SNNs)가 여러 개의 서브모델로 분해될 수 있음을 보여주고, 이를 통한 효율적인 self-distillation 방법을 제안합니다. 기존의 지식 증류(knowledge distillation)와는 다르게, 큰 teacher 모델에 의존하지 않고 각 타임스텝의 SNN 출력을 서브모델로 평가합니다. 연구자는 두 가지 self-distillation 기법인 Strong2Weak과 Weak2Strong을 제안하여 SNN의 성능을 향상시키는 데 큰 성과를 이루었습니다.

- **Technical Details**: SNN을 여러 서브모델로 분할하는 방식은 타임스텝마다의 출력을 개별적으로 분석하는 절차를 포함합니다. Strong2Weak에서는 강한 서브모델이 약한 서브모델을 지도하여 전체 성능을 향상시키며, Weak2Strong은 반대로 약한 모델이 강한 모델을 보조하여 일반화 성능을 높이는 방식입니다. 이러한 접근은 네트워크 아키텍처와 관련된 여러 다양한 신경망 모델에서 적용이 가능합니다.

- **Performance Highlights**: 실험 결과, 제안한 self-distillation 방안이 SNN의 구별력(discriminability)과 전체 성능을 개선했음을 보여줍니다. 강한 모델과 약한 모델의 상호작용은 SNN의 적대적 견고성(adversarial robustness)을 강화하여 안정성을 높였습니다. 이 연구는 SNN 훈련의 효율성을 크게 향상시키는 방법을 제시하며, 향후 다양한 실제 상황에서의 적용 가능성을 열어줍니다.



### SketchGuard: Scaling Byzantine-Robust Decentralized Federated Learning via Sketch-Based Screening (https://arxiv.org/abs/2510.07922)
Comments:
          23 pages, 5 figures, Code Available: this https URL

- **What's New**: 본 논문에서는 Byzantine 공격으로부터 저항력을 갖춘 분산 연합 학습(DFL) 시스템을 위한 새로운 프레임워크인 SketchGuard를 제안합니다. 기존 접근 방식이 모든 클라이언트가 이웃과 모델 벡터를 교환하고 비교해야 하는 반면, SketchGuard는 스케치 기반 이웃 스크리닝을 통해 이 과정을 분리합니다. 이를 통해 통신 및 계산 비용을 크게 줄이며, 웹 규모의 응용 프로그램에 배포할 수 있는 가능성을 열어줍니다.

- **Technical Details**: SketchGuard는 Count Sketch라는 기법을 이용해 고차원 모델을 저차원 스케치로 압축하고, 이후 승인된 이웃으로부터만 전체 모델을 가져옵니다. 이렇게 함으로써 각 라운드의 통신 복잡도를 O(k|N_i| + d|S_i|)로 줄일 수 있게 됩니다. 이론적으로도 SketchGuard는 강한 볼록성과 비볼록성 환경 모두에서 수렴 보장을 제공하며, 압축 과정에서의 오차가 단지 효과적 임계값에 $(1 + O(	heta))$의 영향을 미치도록 보장합니다.

- **Performance Highlights**: 다양한 데이터셋, 네트워크 토폴로지, 공격 시나리오에서의 실험 결과에 따르면, SketchGuard는 최신 방법들과 동일한 강인성을 유지하는 동시에 계산 시간은 최대 82% 줄이고 통신 오버헤드는 50-70% 감소시킵니다. 이러한 결과는 모델 차원 및 네트워크 연결성과 함께 배가되는 이점을 제공합니다. 따라서 스케치 기반 압축이 웹 규모의 강력한 DFL을 가능하게 하는 기본 요소임을 입증하고 있습니다.



### GRADE: Personalized Multi-Task Fusion via Group-relative Reinforcement Learning with Adaptive Dirichlet Exploratio (https://arxiv.org/abs/2510.07919)
- **What's New**: 이번 논문에서는 개인화된 다중 목표 랭킹 시스템의 전반적인 아키텍처를 제안합니다. 여기에는 초기 피처 처리 및 후보 생성을 위한 Feature Center와 Prerank Model, 다양한 사용자 피드백 신호를 예측하는 Multi-Task Learning (MTL) 모델, 그리고 개인화된 가중치를 학습하는 Multi-Task Fusion (MTF) 모듈이 포함됩니다. 최종 점수를 계산하고 최종 추천을 생성하기 위해 Blended Ranking Model을 통해 결과를 사용자에게 제공합니다.

- **Technical Details**: 연구에서는 Group Relative Policy Optimization (GRPO)에서 영감을 받아 GRADE(Adaptive Dirichlet Exploration을 포함한 그룹 기반 강화 학습)라는 새로운 프레임워크를 제안합니다. GRADE는 안정적인 학습 과정을 보장하기 위해 두 단계 패러다임으로 훈련되며, 초기 supervised learning-to-rank 단계가 Robust baseline policy를 제공하고, 이후 GRPO 기반의 프레임워크로 세분화됩니다. 특히, critic-free GRPO 패러다임을 사용하여 지속적인 불안정성을 해결하고, Dirichlet 분포를 사용하여 정책 탐색을 개선합니다.

- **Performance Highlights**: 본 논문은 안정적이고 효율적인 정책 학습을 위한 critic-free GRPO 프레임워크를 개인화된 다중 작업 융합에 적용하는 데 중점을 두고 있습니다. 또한, Dirichlet 분포를 활용한 탐색 전략과 보상 신호의 결합을 통해 사용자 맞춤형 추천 성능은 물론, 보상을 해킹하는 것을 방지합니다. 이로 인해 추천 시스템의 전반적인 만족도를 높일 수 있는 개념적 및 실용적 기여를 목표로 합니다.



### MMM: Quantum-Chemical Molecular Representation Learning for Combinatorial Drug Recommendation (https://arxiv.org/abs/2510.07910)
Comments:
          Medical Image Computing and Computer-Assisted Intervention (MICCAI) Predictive Intelligence in Medicine Workshop (MICCAI PRIME) 2025; 13 pages

- **What's New**: 이번 연구에서는 약물 간 상호작용 예측을 위해 새롭게 제안된 프레임워크인 Multimodal DDI Prediction with Molecular Electron Localization Function (ELF) Maps (MMM)을 소개합니다. MMM은 약물 표현 학습에 3차원 양자 화학 정보를 통합하여 높은 예측 정확도와 안전한 약물 처방을 지원하는 가능성을 보여줍니다. 이 프레임워크는 약물의 전자 밀도 맵을 생성하여 치료적 관련성과 상호작용 위험을 동시 고려할 수 있도록 설계되었습니다.

- **Technical Details**: MMM은 세 가지 주요 구성 요소로 이루어져 있습니다: 환자의 전자 건강 기록(EHR)을 인코딩하는 모듈, ELF 기반 약물 인코더로 전자 상호작용 특성을 반영하며, 환자 조건에 기반하여 약물 하위 구조의 중요성을 추론하는 지역 이분법 인코더, 그리고 안전하고 효과적인 약물 처방을 도출하기 위한 약물 추천 모듈입니다. 이 프레임워크는 DFT(computational density functional theory)를 사용하여 ELF 맵을 구성하고, 고차원 특징을 추출하기 위해 사전 학습된 CNN(convolutional neural network)을 활용합니다.

- **Performance Highlights**: MMM은 MIMIC-III 데이터셋에서 여러 기본 모델과 비교하여 F1 점수, Jaccard 유사성 및 DDI 비율에서 통계적으로 유의미한 향상을 보였습니다. 특히, GNN 기반의 SafeDrug 모델과의 비교에서 F1-score(p = 0.0387)와 Jaccard(p = 0.0112), DDI 비율(p = 0.0386)에서 개선된 결과를 보여, MMM이 약물 추천의 정밀도를 높이고, 임상 실무에서의 안전한 조합 약물 처방을 지원할 수 있는 잠재력을 입증하였습니다.



### Adaptive Optimizable Gaussian Process Regression Linear Least Squares Regression Filtering Method for SEM Images (https://arxiv.org/abs/2510.07895)
Comments:
          "Adaptive Optimizable Gaussian Process Regression Linear Least Squares Regression Filtering Method for SEM Images," in IEEE Access, vol. 13, pp. 93574-93592, 2025, doi: https://doi.org/10.1109/ACCESS.2025.3573389

- **What's New**: 이 연구는 스캐닝 전자 현미경(SEM) 이미지의 신호 대 잡음 비율(SNR) 및 잡음 분산(NV)을 추정하고, NV 기반의 Wiener 필터를 사용하여 이미지 품질을 향상시키는 완전한 접근 방식을 제시합니다. 기존 방법에 비해 우수한 SNR 추정 기술을 적용하고 기계 학습 모델을 결합하여 SEM 이미지의 NV를 추정하는 것이 주된 아이디어입니다.

- **Technical Details**: 연구에서는 5가지 SNR 추정 기법을 조사했습니다: Nearest Neighbourhood (NN) 방법, First-Order Linear Interpolation (FOL) 방법, NN+FOL 방법, Non-Linear Least Squares Regression (NLLSR) 방법, 그리고 Linear Least Squares Regression (LSR) 방법입니다. LSR 방법이 나머지 방법들보다 더 나은 성능을 보여주었습니다. 이후 Support Vector Machines (SVM)과 Gaussian Process Regression (GPR)을 LSR과 짝지어 테스트하였고, 최적화 가능한 GPR 모델이 가장 높은 정확도를 보여주었습니다.

- **Performance Highlights**: 제안된 Adaptive Optimizable Gaussian Process Regression Linear Least Squares Regression (AO-GPRLLSR) 필터링 파이프라인은 SEM 이미지의 품질 향상을 위한 NV 기반 Wiener 필터에 입력으로 사용되는 잡음 분산을 생성했습니다. 이 방법은 SEM 이미지의 SNR 및 NV 추정에서 눈에 띄는 성공을 거두었으며, 필터링 처리 이후 평균 제곱 오차(MSE)를 낮추는 결과를 가져왔습니다.



### Signal-to-Noise Ratio in Scanning Electron Microscopy: A Comprehensive Review (https://arxiv.org/abs/2510.07886)
Comments:
          in IEEE Access, vol. 13, pp. 154395-154421, 2025, doi: https://doi.org/10.1109/ACCESS.2025.3603013

- **What's New**: 본 논문은 전자현미경(SEM)의 신호 대 잡음 비율(SNR)을 최적화하는 방안을 다룹니다. SEM은 나노기술과 생물학적 이미징에 필수적인 도구로, 그 이미지 품질은 SNR에 크게 영향을 받습니다. 본 연구는 SEM의 작동 원리, 잡음의 출처, SNR 측정 및 추정 방법을 살펴봅니다.

- **Technical Details**: SNR의 개선은 하드웨어와 소프트웨어 측면에서 모두 고려되어야 하며, 다양한 SEM 촬영 프로세스를 통해 잡음의 영향을 줄이는 다양한 방법을 제시합니다. 전통적인 기술뿐만 아니라 새로운 기술에 대한 응용, 장점 및 한계를 다루고 있습니다. 이 정보는 SEM의 성능을 개선하려는 연구자와 실무자들에게 도움이 됩니다.

- **Performance Highlights**: SEM의 응용 분야는 여러 과학적 분야에 걸쳐 있으며, SNR 향상 기술의 발전은 이미지의 선명도와 해석 가능성을 높이는 데 기여할 것입니다. 이 논문은 SNR 최적화에 대한 포괄적인 이해를 제공하여 향후 연구를 장려하는 것을 목표로 하고 있습니다.



### Meta-Learning Based Few-Shot Graph-Level Anomaly Detection (https://arxiv.org/abs/2510.07847)
Comments:
          Accepted by ARRML2025

- **What's New**: 이 논문에서는 그래프 수준 이상 탐지(graph-level anomaly detection)의 새로운 프레임워크인 MA-GAD를 제안합니다. 이 프레임워크는 그래프 크기를 줄이는 그래프 압축 모듈(graph compression module)을 통합하여 노이즈 간섭을 완화하고 필수 노드 정보를 유지합니다. 또한, 메타 학습(meta-learning)을 활용하여 유사 네트워크로부터 메타 이상 정보를 추출하고, 적은 샘플로도 새로운 작업에 빠르게 적응할 수 있는 초기화 모델을 학습합니다.

- **Technical Details**: MA-GAD 프레임워크는 세 가지 단계로 구성됩니다: (1) 그래프 압축 전략을 통해 GNN 훈련 중 성능 손실을 최소화합니다. (2) 메타 학습 알고리즘이 보조 네트워크로부터 메타 이상 지식을 추출하여 일반화를 향상시킵니다. (3) 그래프 이상 손실 함수는 정상 노드와 이상 노드 간의 통계적 편차를 증가시켜 클래스 불균형(class imbalance)을 해결합니다.

- **Performance Highlights**: 실험 결과, MA-GAD는 4개의 실제 생화학 데이터 셋을 기반으로 기존의 최신 기법들을 초월하여 적은 샘플로 그래프 수준 이상 탐지에서 우수한 성능을 보였습니다. 본 연구는 지역적 및 글로벌 이상을 모두 다루며, 노드 및 그래프 수준에서 이상의 탐지를 강화하여 효과적인 결과를 도출했습니다.



### Self-Improving LLM Agents at Test-Tim (https://arxiv.org/abs/2510.07841)
- **What's New**: 이번 논문은 새로운 테스트 시간 자기 향상 방법(Test-Time Self-Improvement, TT-SI)을 제안합니다. 이 방법은 모델이 어려움을 겪는 샘플을 식별하고, 이를 기반으로 새로운 훈련 샘플을 생성한 후, 테스트 시간에 이 샘플들로 모델을 개선하는 과정으로 구성됩니다. 기존의 대량 데이터셋에 의존하지 않고도 성능을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 세 가지 단계로 이루어집니다: 첫째, 자기 인식을 통해 모델이 어려움을 겪는 샘플을 식별하고, 둘째, 불확실 샘플에서 유사한 예제를 생성하며(셀프 데이터 증강), 셋째, 테스트 시간 맞춤형 훈련을 통해 이 새롭게 생성된 샘플들로부터 학습하는 방식입니다. 이 과정에서 'Uncertainty Estimator'와 'Data Synthesis Function'이 사용되어 모델의 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, TT-SI는 평균적으로 다른 표준 학습 방법에 비해 +5.48%의 절대 정확도 향상을 기록하며, 68배 적은 훈련 샘플로도 효과적인 학습이 가능함을 보여줍니다. TT-SI는 특정 과제를 위한 적응 능력을 강화하여, 기존의 감독된 세미 슈퍼바이즈드 학습(supervised fine-tuning) 방법보다 우수한 성능을 보입니다. 이 연구는 의료 기기 및 기타 다양한 복잡한 작업에서도 지속적인 자기 개선의 가능성을 보여줍니다.



### MetaDefense: Defending Finetuning-based Jailbreak Attack Before and During Generation (https://arxiv.org/abs/2510.07835)
Comments:
          Accepted By NeurIPS 2025

- **What's New**: 본 논문에서는 MetaDefense라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)에서 파인튜닝 기반의 jailbreak 공격을 방어하는 데 중점을 두고 있습니다. 기존 방어 메커니즘이 보지 못한 공격 템플릿에 의해 위장된 해로운 쿼리에 일반화되지 못하는 문제를 발견하였습니다.

- **Technical Details**: MetaDefense는 두 단계의 방어 접근 방식을 제안합니다: (i) 사전 생성 방어(pre-generation defense)로 해로운 쿼리를 반응 생성이 시작되기 전에 탐지하고, (ii) 중간 생성 방어(mid-generation defense)로 생성 중 부분적인 응답을 감시하여 보다 해로운 내용을 출력하는 것을 방지합니다. 이 시스템은 특별한 프롬프트를 사용하여 쿼리와 부분 응답의 해로움을 예측하도록 LLM을 훈련시킵니다.

- **Performance Highlights**: 다양한 LLM 아키텍처(예: LLaMA-2-7B, Qwen-2.5-3B-Instruct, LLaMA-3.2-3B-Instruct)에 대한 광범위한 실험 결과, MetaDefense는 기존 방어 메커니즘에 비해 상당히 우수한 성능을 보여주었습니다. 또한, 본 시스템은 보이는 공격 템플릿과 보이지 않는 공격 템플릿 모두에 대해 효과적인 방어를 제공하며, 일반 작업에 대해 경쟁력 있는 성능을 유지합니다.



### SIMU: Selective Influence Machine Unlearning (https://arxiv.org/abs/2510.07822)
Comments:
          Accepted to NeurIPS 2025 Workshop: Constrained Optimization for Machine Learning (COML)

- **What's New**: 최근 Large Language Models(LLMs)의 민감한 정보의 원치 않는 암기가 문제가 되면서, 모델의 동작을 조절할 수 있는 안전 메커니즘이 필요하다는 요구가 커졌습니다. 이를 해결하기 위해 데이터 삭제를 가능하게 하는 기계 학습 기법이 개발되었습니다. 본 논문에서는 Selective Influence Machine Unlearning(SIMU)이라는 새로운 프레임워크를 제안하며, 이는 모델이 잊어야 할 정보를 인코딩하는 주요 뉴런만 선택적으로 업데이트함으로써 기억 상실을 효과적으로 수행하면서도 원래의 지식을 유지할 수 있도록 돕습니다.

- **Technical Details**: SIMU 프레임워크는 먼저 MLP(다층 퍼셉트론) 층에서 잊어야 할 정보를 인코딩하는 주요 뉴런을 식별하는 단계로 구성됩니다. 이후, 중요한 뉴런만 선택적으로 업데이트하는 방법을 통해 모델의 파라미터를 조정합니다. 이를 통해 기존의 방법보다 더 효과적으로 잊어야 할 정보의 영향을 제거하면서도 모델이 원래 갖고 있던 지식을 보존하게 됩니다.

- **Performance Highlights**: SIMU는 현재 사용되고 있는 기계 학습 삭제 기법들과 비교하여, 정보 삭제의 효율성을 유지하면서도 모델의 원래 기능을 크게 향상시키는 결과를 제공합니다. 실험 결과, SIMU는 제한된 뉴런 업데이트를 통해 원치 않는 정보의 삭제가 잘 이루어짐을 보여주어 기존 방법들에 비해 뛰어난 성능을 발휘했습니다. 이로써 AI 모델의 데이터 보호 및 안전성을 더욱 높이는 방향으로 기여할 것으로 기대됩니다.



### HySim-LLM: Embedding-Weighted Fine-Tuning Bounds and Manifold Denoising for Domain-Adapted LLMs (https://arxiv.org/abs/2510.07796)
- **What's New**: 이번 논문에서는 약리학적 정보의 추출 및 표준화에서의 한계를 극복하는 새로운 프레임워크인 HySim-LLM을 제안합니다. HySim-LLM은 embedding-weighted fine-tuning과 manifold-aware denoising을 결합하여 LLM을 구조화된 생물의학 데이터에 적용 가능하도록 개선하는 수학적 및 계산적 접근 방식을 제공합니다. 이를 통해 LLM의 강건성과 해석력을 높이는 새로운 이론적 기초를 마련하고자 합니다.

- **Technical Details**: HySim-LLM은 원천 데이터셋(S)과 특정 도메인 데이터셋(T) 간의 embedding 유사성을 활용하여 LLM 파라미터를 적응시키는 과정에서 provable 보장을 제공합니다. 구체적으로, cosine이나 Mahalanobis 거리와 같은 밀접도 지표를 사용하여 embedding divergence를 도입함으로써 fine-tuning의 성능 보장을 해석 가능한 방식으로 연결합니다. 또한, 임베딩 공간에서 off-manifold 샘플이 손실에 미치는 영향을 정량화하여 이를 기반으로 LLM 기반 파이프라인에서의 데이터 정리를 가능하게 합니다.

- **Performance Highlights**: 이 연구는 기존의 AutoPK 및 WCPK 시스템을 개선하여 PK 데이터 추출 및 정제 과정에서 높은 정확도와 일관성을 입증했습니다. 새로운 이론적 결과인 similarity-weighted fine-tuning bound와 manifold-based denoising theorem은 LLM을 생물의학 분야 및 데이터 집약적 과학 도메인에 더욱 신뢰할 수 있게 만들어 줄 것입니다. HySim-LLM은 이러한 이론적 요소들을 통해 기존의 생물의학 LLM 적용에서의 신뢰성 문제를 해결하는데 큰 기여를 예상하고 있습니다.



### Weak Form Learning for Mean-Field Partial Differential Equations: an Application to Insect Movemen (https://arxiv.org/abs/2510.07786)
Comments:
          39 pages, 16 figures

- **What's New**: 본 논문은 가변적인 식물 자원과 감염 상태에서 실험적으로 수집한 희소한 데이터로부터 레피도프테라 유충의 이동 동태를 모델링하는 방법론을 제시합니다. 갤러킨 방법인 WSINDy 알고리즘을 적용하여 고차원적이고 해석 가능한 지배 방정식을 학습하는 데 기여하고 있습니다. 이는 해충의 분산 역학을 이해하고 예측하는 데 중요한 도구로 자리잡을 것으로 기대됩니다.

- **Technical Details**: 연구에서는 WSINDy 알고리즘과 커널 밀도 추정을 결합하여, 감염 상태와 식물 제원이 분산에 미치는 영향을 평가하기 위한 효과적인 이동 모델을 학습합니다. 사용되는 수학적 방법론은 복잡한 생물학적 맥락에서의 데이터 기반 모델링에 중점을 두고 있으며, 이는 공학, 생명과학 등 다양한 분야에서의 응용 가능성을 함축하고 있습니다.

- **Performance Highlights**: 본 연구에서 제안하는 방법론은 비교 실험 설정에서 이동 모델을 효과적으로 학습하여 각 실험 조건에 따라 감염 상태와 식물 유전자형이 어떻게 분산 동태에 영향을 미치는지를 분석합니다. 이러한 결과들은 해충 관리 및 예측 모델링의 정확도를 향상시킬 수 있는 기반을 마련하는 데 기여할 것입니다.



### FedLAM: Low-latency Wireless Federated Learning via Layer-wise Adaptive Modulation (https://arxiv.org/abs/2510.07766)
- **What's New**: 이 논문에서는 무선 연합 학습(wireless federated learning, FL)에서 통신 지연 문제를 줄이기 위한 새로운 레이어별 적응 변조(layer-wise adaptive modulation) 방안을 제안합니다. 기존의 연구들은 모든 DNN 레이어에 동일한 변조 수준을 부여했으나, 본 연구는 레이어의 중요성을 고려하여 각 레이어가 최적의 변조 수준을 자동으로 결정하도록 합니다. 실험 결과, 제안된 방식은 기존 기법에 비해 최대 73.9%의 통신 지연을 절감할 수 있음을 보여줍니다.

- **Technical Details**: 레벨 변조는 학습 기술과 통신 기술 관점에서 접근하며, 높은 신호 대 잡음비(signal-to-noise ratio, SNR)에서는 높은 변조 수준을 사용하고, 낮은 SNR에서는 낮은 변조 수준을 사용하는 전통적인 적응 변조 기법과는 다릅니다. 본 논문에서는 각 DNN 레이어의 중요성을 고유값(eigenvalue)으로 정량화하며, 이를 통해 DNN의 레이어가 각기 다른 변조 수준을 선택할 수 있도록 최적화 문제를 설정합니다.

- **Performance Highlights**: 실험을 통해 제안된 레이어별 적응 변조 기법의 성능을 검증하였고, 다양한 환경 설정에서 우수한 통신 효율과 낮은 지연 시간을 기록했습니다. 특히, 본 기법은 네트워크 대역폭이 제한된 상황에서도 학습 성능을 유지하면서 통신 지연을 효과적으로 줄일 수 있음을 확인했습니다. 이는 향후 무선 FL 시스템의 실제 적용 가능성을 높이는 연구 결과로 이어질 것입니다.



### A Unified Multi-Task Learning Framework for Generative Auto-Bidding with Validation-Aligned Optimization (https://arxiv.org/abs/2510.07760)
- **What's New**: 본 논문에서는 온라인 광고 캠페인의 최적화를 위한 새로운 방법인 Validation-Aligned Multi-task Optimization (VAMO)을 제안합니다. VAMO는 각 작업의 훈련 경량과 유지된 검증 경량 간의 정렬을 기반으로 작업 가중치를 적응적으로 할당하여 검증 성능 향상에 초점을 맞추고 있습니다. 또한, 이 연구는 시계열 모듈을 통합하여 계절적 구조의 크로스 태스크 전이를 강화하고, 실제 배포 목적과 일치하는 업데이트를 유도하는 새로운 비용 구조를 제안합니다.

- **Technical Details**: VAMO는 다중 작업 학습(MTL) 프레임워크를 통해 서로 관련된 작업들 간에 공유된 표현을 기반으로 훈련합니다. 이 프레임워크는 비선형 성격의 입찰 환경에 적응할 수 있는 메커니즘을 제공합니다. 정량적인 이론적 보장을 통해 수렴 보장 및 정렬 분석을 수행하며, 이는 VAMO의 효과를 입증합니다.

- **Performance Highlights**: 실험 결과, VAMO는 전통적인 기준선과 비교했을 때 현저한 성능 향상을 보여줍니다. 시뮬레이션 및 실제 광고 시스템에서의 광범위한 실험을 통해 제안된 방법의 효과성을 강조하며, 산업적 배포에 대한 실용적인 통찰력을 제공합니다. 이 접근법은 특히 수요가 다양하고 동적인 광고 환경에서 강력한 성능을 발휘하는 것으로 나타났습니다.



### Rényi Sharpness: A Novel Sharpness that Strongly Correlates with Generalization (https://arxiv.org/abs/2510.07758)
- **What's New**: 이 논문에서는 심층 신경망의 일반화를 조사하기 위해 새로운 sharpness 지표를 제안합니다. 이 새로운 측정 지표는 loss Hessian(손실 헤시안)의 음수 Rényi entropy(렌이 엔트로피)로 정의되며, 전통적인 sharpness 측정보다 일반화 성능을 더 잘 예측할 수 있는 가능성을 가지고 있습니다. 아울러, Rényi sharpness는 훈련 중 regularizer로 사용될 수 있는 변형도 제안되어, 기존의 sharpness-aware minimization 방법들을 능가하는 결과를 보였습니다.

- **Technical Details**: 현재 제안된 Rényi sharpness는 loss Hessian의 스펙트럼의 확산 정도를 특성화하는 데 중점을 둡니다. 이 방법은 해시안의 고유값이 가능한 한 균일해야 일반화 성능이 좋다는 직관에 기반하고 있습니다. 논문에서는 Rényi sharpness와 일반화 간의 관계를 정립하기 위해 몇 가지 일반화 경계를 개발하였으며, 이를 위해 Stochastic Lanczos Quadrature (SLQ) 방법을 활용하여 Rényi sharpness를 빠르게 추정하는 알고리즘도 제안하고 있습니다.

- **Performance Highlights**: Rényi Sharpness Aware Minimization (RSAM) 기법은 Stochastic Gradient Descent (SGD)의 일반화를 일관되게 향상시키며, 테스트 정확도에서 2.5%까지의 향상을 보였습니다. 이는 기존의 sharpness-aware minimization 방법들보다 더 뛰어난 성능을 보여줍니다. 실험 결과는 또한 Rényi sharpness와 일반화 간에 강한 상관관계를 나타내며, 특히 Kendall rank correlation 분석을 통해 도출되었습니다.



### FedBook: A Unified Federated Graph Foundation Codebook with Intra-domain and Inter-domain Knowledge Modeling (https://arxiv.org/abs/2510.07755)
Comments:
          Under Review

- **What's New**: 본 논문에서는 기존의 Graph Foundation Models (GFMs)의 한계를 극복하기 위해 Federated Graph Foundation Models (FedGFMs)와 새로운 접근법인 FedBook을 제안합니다. FedBook은 클라이언트의 로컬 코드북을 집계하여 글로벌 코드북을 구성함으로써, 데이터 프라이버시를 보존하면서도 도메인 간 다양성과 도메인 내 일관성을 유지합니다.

- **Technical Details**: FedBook은 두 단계로 구성됩니다. 첫째, Intra-domain Collaboration 단계에서는 저빈도 토큰을 다른 클라이언트의 고빈도 토큰을 참조하여 개선하여 도메인 내 일관성을 높입니다. 둘째, Inter-domain Integration 단계에서는 각 클라이언트의 코드북의 의미적 독특성을 평가하여, 독특한 기여가 더 큰 영향을 미치도록 가중치를 부여합니다.

- **Performance Highlights**: FedBook은 8개의 다양한 벤치마크와 여러 작업에 대한 실험을 통해 21개의 기초 모델을 지속적으로 초월하며, 고립된 감독 학습, FL/FGL, 중앙 집중식 GFMs의 연합 적응 및 FedGFM 기술을 포함한 다양한 방법들과 비교하여 뛰어난 성능을 보여줍니다.



### t-SNE Exaggerates Clusters, Provably (https://arxiv.org/abs/2510.07746)
- **What's New**: 이 논문은 t-Distributed Stochastic Neighbor Embedding (t-SNE)의 출력이 입력 데이터의 군집(clustering) 및 이탈(outlier)에 대한 정보를 신뢰성 있게 반영하지 못함을 입증하였습니다. 특히, highly clustered 데이터와 arbitrarily un-clustered 데이터가 동일한 최대 군집화된 시각화를 생성할 수 있음을 보여주어 t-SNE의 한계를 밝혔습니다. 이러한 결과는 과학적 데이터 해석 과정에서 오해를 피하는 데 중요한 의미를 지니고 있습니다.

- **Technical Details**: t-SNE는 데이터 포인트 간의 친밀도를 기반으로 희소한 고차원 데이터의 구조를 시각화하는 기법입니다. 이 논문에서는 t-SNE가 입력 데이터의 군집 구조를 그대로 유지하지 못하는 점을 이론적으로 분석하며, 데이터의 이탈점을 제대로 묘사하지 못하는 특성을 분석합니다. 특히, 논문에서는 다양한 수학적 정리를 통해 t-SNE의 비일관성을 설명하고, 공격적(adversarial) 기법을 사용하여 오류를 유도하는 방법도 제시합니다.

- **Performance Highlights**: 이 연구는 t-SNE의 신뢰성과 관련한 실증적 데이터를 기반으로 한 기존 연구들과는 달리, 이 기술이 잘못된 군집 구조를 생성할 수 있음을 첫 번째로 이론적으로 분석하였습니다. t-SNE가 출력 시 잘못된 긍정(false positive)을 발생시키는 경향이 있음을 강조하며, 이는 다양한 실세계 데이터셋에서 관찰되었습니다. 특히, 연구는 이러한 이탈자(outlier)가 대부분 군집 구조에 포함되어 종종 간과된다는 점에서 심각한 문제를 제기합니다.



### MeSH: Memory-as-State-Highways for Recursive Transformers (https://arxiv.org/abs/2510.07739)
- **What's New**: 이 논문에서는 메모리 관리 문제를 해결하기 위해 새로운 구조인 Memory-as-State-Highways (MeSH)를 도입하였습니다. MeSH는 반복적으로 인스턴스를 생성하는 대신, 명시적인 메모리 버퍼를 활용하여 효율성을 높입니다. 이로 인해 함수적 전문화가 가능해지며, 더 적은 파라미터로도 성능을 향상시킵니다. 특히, MeSH는 1.4B 스케일의 비재귀적 모델들보다 항상 우수한 성능을 보여줍니다.

- **Technical Details**: MeSH 아키텍처는 경량의 라우터를 사용하여 여러 반복 과정에서 컴퓨팅을 동적으로 다양화합니다. 이를 통해 불필요한 정보 과부하를 줄이고, 반복 과정의 각 단계를 구체화하는 데 필요한 분산 메모리를 제공합니다. 또한, 초기 상태에서 최종 상태까지의 전체 네트워크 구조인 Prelude-Recurrent-Coda 구조를 채택하여 성능을 크게 향상시킵니다.

- **Performance Highlights**: Pythia 스위트(160M-1.4B)에서 MeSH를 강화한 재귀형 변환기는 재귀적 기준선보다 일관되게 성능이 향상되었습니다. 평균적인 하류(Downstream) 정확도가 +1.06% 향상되었으며, 비임베딩(non-embedding) 파라미터가 33% 적습니다. 이러한 분석 결과들로 인해 MeSH는 더욱 강력한 재귀적 모델을 구축하기 위한 확장 가능하고 원칙적인 아키텍처로 자리매김하고 있습니다.



### GeoGen: A Two-stage Coarse-to-Fine Framework for Fine-grained Synthetic Location-based Social Network Trajectory Generation (https://arxiv.org/abs/2510.07735)
- **What's New**: 최근 Synthetic Data Generation의 발전을 활용하여 LBSN 데이터 접근의 한계를 극복하고자 하는 시도가 이루어지고 있습니다. 기존의 데이터 수집 및 개인 정보 보호 문제를 해결하기 위해, 이 논문에서는 GeoGen이라는 두 단계의 coarse-to-fine 프레임워크를 제안하였습니다. 이는 LBSN check-in trajectory의 생성을 효율적으로 수행할 수 있는 새로운 방법론을 제공합니다.

- **Technical Details**: GeoGen은 첫 번째 단계에서 원래의 LBSN 체크인 궤적에서 공간적으로 연속적이고 시간적으로 규칙적인 잠재 이동 시퀀스를 재구성합니다. 이를 위해 Sparsity-aware Spatio-temporal Diffusion model (S^2TDiff)을 설계하여 사용자 행동 패턴을 학습합니다. 두 번째 단계에서는 Transformer 기반의 Seq2Seq 아키텍처인 Coarse2FineNet을 통해 세분화된 LBSN 궤적을 생성하며, 이 과정에서 동적 컨텍스트 융합 메커니즘을 활용합니다.

- **Performance Highlights**: 다양한 실험을 통해 GeoGen은 데이터의 정확성과 유용성 평가에서 최첨단 모델들을 초월하는 성능을 보였습니다. 예를 들어, FS-TKY 데이터셋에서 거리 및 반경 지표에서 각각 69% 및 55%의 성장을 나타내었습니다. 이러한 성과는 GeoGen이 데이터 처리의 효율성과 품질 사이의 중요한 균형을 효과적으로 해결했음을 보여줍니다.



### DEAS: DEtached value learning with Action Sequence for Scalable Offline RL (https://arxiv.org/abs/2510.07730)
Comments:
          Project website: this https URL

- **What's New**: 본 연구에서는 액션 시퀀스를 이용한 DEtached value learning with Action Sequence (DEAS)라는 오프라인 강화 학습 프레임워크를 소개합니다. 이는 복잡한 작업에서 가치 학습의 효과성을 극대화할 수 있도록 설계되었습니다. 기존의 단일 스텝 액션 대신 시간적으로 확장된 액션을 활용하여 더 많은 정보를 제공하고, 이는 옵션 프레임워크를 통해 해석될 수 있습니다.

- **Technical Details**: DEAS는 가치 함수에 연속된 액션 타임스텝을 입력으로 처리하며, 시간적으로 확장된 액션으로 nn-단계 TD 업데이트와 유사한 원리를 적용합니다. 이를 통해 명시적인 목표 조건 없이도 수평 감소(horizon reduction)를 제공합니다. 가치 추정의 과대평가 문제를 해결하기 위해 detached value learning을 사용하여 오프라인 데이터셋에서 높은 수익을 올리는 액션을 선호하는 방식으로 가치 추정을 조정합니다.

- **Performance Highlights**: DEAS는 OGBench의 복잡한 장기 작업에서 모든 기준선보다 일관되게 우수한 성능을 보였으며, 실험을 통해 그 유효성을 입증했습니다. 또한, RoboCasa Kitchen과 실제 조작 작업에서 대규모 비전-언어-액션 모델의 성능을 향상시키는 데 성공하여, 오직 전문가의 시연만으로 훈련된 정책에 비해 현저한 성과 향상을 이끌었습니다.



### Computationally-efficient Graph Modeling with Refined Graph Random Features (https://arxiv.org/abs/2510.07716)
Comments:
          Preprint. Comments welcome

- **What's New**: 이번 연구에서는 효율적이고 정확한 그래프 노드에 정의된 커널(kerel)을 위한 새로운 클래스인 정제된 그래프 랜덤 피처(GRFs++)를 제안합니다. GRFs++는 기존 GRFs의 한계를 극복하고, 보다 먼 노드 간의 관계를 모델링하는 어려움을 해소합니다. 새롭게 제안된 워크 스티칭(walk-stitching) 기술을 통해 긴 그래프 랜덤 워크에 대한 의존성을 줄이면서, 여러 짧은 워크를 이어붙여 편향이 없는 특성을 유지합니다.

- **Technical Details**: 정제된 GRFs는 특정 그래프 커널의 구현을 찾고 그래프의 가중치 행렬의 거듭제곱을 포함하는 테일러 급수로 인코딩된 커널 행렬의 고차 분해(de-convolution)와의 내재적 연결을 제공합니다. GRFs++는 고전적인 GRFs의 걷기 종료 메커니즘을 일반적 분포를 적용하여 확대하여, 그래프 커널의 보다 정확한 근사를 가능하게 합니다. 이를 통해 추가적인 계산 비용 없이 커널에 대한 근사 정확성을 향상시킵니다.

- **Performance Highlights**: GRFs++는 기존 GRFs와 비교하여 근사 품질, 속도 및 여러 하위 작업에서 더 나은 성능을 발휘하는 것을 실험적으로 입증했습니다. 실험에서는 메쉬에서의 정상 벡터 필드 예측, 클러스터링 및 그래프 분류와 같은 여러 작업을 포함하였으며, GRFs++는 기존 방법들에 비해 향상된 성능을 보였습니다. 우리는 실험적 평가 결과를 통해 GRFs++의 주장을 뒷받침하고 있습니다.



### LiveThinking: Enabling Real-Time Efficient Reasoning for AI-Powered Livestreaming via Reinforcement Learning (https://arxiv.org/abs/2510.07685)
Comments:
          12 pages, 8 figures

- **What's New**: 이번 연구에서는 AI 기반 전자상거래 라이브 스트리밍에서 실시간 반응을 지원하는 디지털 아바타의 필요성을 다룹니다. 기존의 Large Reasoning Models (LRMs)는 높은 지연시간 문제로 인해 이러한 작업에 적합하지 않다는 점을 강조하며, 이를 해결하기 위한 새로운 두 단계 최적화 프레임워크인 LiveThinking을 제안합니다. 이 프레임워크의 목표는 반응의 정확성을 유지하면서도 지연 시간을 최소화하는 것입니다.

- **Technical Details**: LiveThinking은 첫 단계에서 670B 파라미터의 teacher 모델을 경량화된 30B Mixture-of-Experts (MoE) 모델로 증류하는 과정을 포함합니다. 여기에는 잘못된 생성을 필터링하는 LLM 기반 평가자를 사용하는 Rejection Sampling Fine-Tuning (RFT)이 포함되어 있습니다. 두 번째 단계에서는 Group Relative Policy Optimization (GRPO)를 통해 모델의 추론 경로를 압축하여 응답 품질을 유지하면서 이전의 긴 추론 경향을 초래하는 비효율성을 해결합니다.

- **Performance Highlights**: LiveThinking은 실제 Taobao Live에 적용되어 3.3%의 정확성 향상과 21.8%의 유용성 증가를 이끌어냈습니다. 이 시스템은 응답의 처리 비용을 30배 줄이면서 처리 지연을 초단위로 줄이는 성과를 거두었습니다. 최종적으로, LiveThinking은 GMV(총 상품 판매량)에서 통계적으로 유의미한 증가를 가져오며, 사용자 경험과 상업적 성과를 개선하는 데 기여했습니다.



### FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning (https://arxiv.org/abs/2510.07664)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 본 논문에서는 반비동기 연합 학습(semi-asynchronous federated learning, SAFL)에서의 집계 전략 간의 주요 차이를 이론적으로 분석하고 이를 해결하기 위한 FedQS라는 첫 번째 프레임워크를 제안합니다. FedQS는 클라이언트를 네 가지 유형으로 분류하여 각각의 데이터 분포 및 계산 자원에 따라 적응적으로 지역 학습을 최적화하는 분할 정복 전략을 도입합니다. 이로 인해 연합 학습(federated learning)의 안정성과 정확성을 높이면서도 효율성을 극대화할 수 있습니다.

- **Technical Details**: FedQS의 핵심 통찰력은 오래된 업데이트(stale updates) 및 데이터 이질성(data heterogeneity)이 서로 다른 집계 전략의 최적화 경로에 고유한 연속성을 야기한다는 점입니다. 우리는 gradient 집계와 model 집계를 모두 최적화하는 최초의 프레임워크로, 이론적 수렴 분석을 통해 두 가지 집계 전략의 수렴 불안정성 및 최적 수렴 능력 부족을 해결합니다. FedQS는 각 클라이언트의 훈련 전략을 동적으로 조정하여 최적의 학습 경로를 제공합니다.

- **Performance Highlights**: 실험 결과, FedQS는 CIFAR-10, 자연어 처리(natural language processing), 실제 데이터(UCI Adult)과 같은 다양한 분야에서 기존 최첨단 모델들을 초월하며, 평균 정확도를 각각 38.98%와 5.65% 향상시키면서 훈련 시간을 58.85% 및 3.68% 단축시켰습니다. 또한 FedQS는 정확도가 가장 높은 모델들과 비교할 때 각각 15.74% 및 12.93% 빠른 수렴을 달성하며 훈련 시간을 72.63% 및 48.04% 줄였습니다. 이러한 결과를 통해 FedQS가 이론과 실제에서의 연합 학습 사이의 간극을 줄이는 데 기여함을 증명합니다.



### Incremental Hybrid Ensemble with Graph Attention and Frequency-Domain Features for Stable Long-Term Credit Risk Modeling (https://arxiv.org/abs/2510.07663)
- **What's New**: 이 논문은 장기 대출 채무 불이행 예측을 위한 HYDRA-EI라는 하이브리드 앙상블 점진 학습(framework)을 제안합니다. HYDRA-EI는 여러 단계의 다양한 기능(feature) 처리를 통해 여러 모델을 결합하고, 그래프 주의(graph attention), 자동 크로스 특징 자동 생성, 주파수 변환(frequency domain transformations)을 사용합니다. 이 프레임워크는 매주 새 데이터를 사용하여 업데이트되고, 성능 기반 방법을 통해 모델 가중치를 조정합니다.

- **Technical Details**: HYDRA-EI는 LightGBM, CatBoost 및 DenseLight+라는 세 가지 학습기를 포함하고 있습니다. 이 모델은 그래프 특징 합성기(Graph Feature Synthesizer), 자동 크로스 엔진(AutoCross engine), 및 주파수-시간 인코더(SpectroTemporal Encoder)를 통해 관계 기반, 비선형 및 주기적 특징을 처리합니다. 모델 업데이트는 반복 학습(rehearsal-based updating)을 통해 개념 변화(concept drift)에 대응하도록 설정되어 있으며, 라벨 스무딩(label smoothing)과 게이티드 SE 블록(gated SE blocks)을 사용하여 DNN을 미세 조정합니다.

- **Performance Highlights**: HYDRA-EI는 신뢰할 수 있는 Gini 지수(stability) 및 일반화(generalization) 성능을 보여주며, 동적 신용 리스크 모델링에 대한 강력하고 적응형 솔루션을 제공합니다. 제안된 기능 엔지니어링 전략은 고객 행동의 관계적, 비선형 및 주기적 측면을 포착하며, 실세계 금융 데이터에서 모델 성능에 긍정적인 영향을 미칩니다. HYDRA-EI는 다양한 산업에서 장기적이고 안정적인 위험 예측을 필요로 하는 실제 금융 시스템에 유용합니다.



### Value Flows (https://arxiv.org/abs/2510.07650)
- **What's New**: 이 논문은 강화 학습(Reinforcement Learning)에서 미래 보상의 전체 분포를 추정하기 위한 새로운 접근법인 Value Flows를 제안합니다. 기존의 방법들이 단일 스칼라로 보상을 표현하는 데 비해, 이 연구는 강화 학습의 신호를 강화하고 탐색 및 안전한 RL에 응용할 수 있도록 보상 분포를 활용합니다. 이 방법은 고급 흐름 기반 모델(flow-based model)을 사용하여 미래 보상의 분포를 추정하고 높은 변동성(return variance)을 가진 상태를 식별합니다.

- **Technical Details**: Value Flows는 흐름 일치(objective)를 통해 분포적 벨만 방정식(distributional Bellman equation)에 부합하는 확률 밀도 경로를 생성하는 새로운 방식으로, 강화 학습에서 보상 분포를 모델링합니다. 이를 위해 흐름 파생 미분 방정식(flow derivative ODE)을 활용하여 다양한 상태에서의 보상 변동성을 추정하고, 불확실성 정보를 이용해 특정 전환에서 보다 정확한 보상 추정을 학습할 수 있는 우선순위를 정합니다. 이 방식은 이전의 오프라인(offline) 및 온라인-오프라인(online-to-online) 방법들과 비교하여 성능 향상을 가져옵니다.

- **Performance Highlights**: Value Flows 방법은 3737개의 상태 기반과 2525개의 이미지 기반 벤치마크 과제에서 실험을 통해 평균 1.3배 더 높은 성공률을 기록하며 기존 방법들보다 우수한 성능을 나타냈습니다. 이러한 성능 향상은 강화 학습 실제 응용 분야에서 높은 보상 예측 정확성을 보장할 수 있음을 의미합니다. 따라서 이 연구는 미래 보상의 추정을 보다 정교하게 수행하는 데 기여하고 있습니다.



### Continual Learning for Adaptive AI Systems (https://arxiv.org/abs/2510.07648)
Comments:
          5 pages 2 figures 2 tables

- **What's New**: 본 논문에서는 Neural Network가 여러 연속적인 작업을 수행하면서 이전에 습득한 지식을 잃지 않도록 하는 Continual Learning 문제를 다룹니다. 기존의 정규화 기법을 넘어서 새로운 방법인 Inter-Cluster Separation (ICS) 기반의 정규화 기법을 제안하여 모델이 이전 작업의 데이터 중심으로부터 멀리 떨어진 출력을 생성할 때 페널티를 부과합니다. 이를 통해 Neural Network의 내부 표현에서 작업 간의 명확한 구분을 가능하게 하여 망각을 감소시키는 효과를 확인했습니다.

- **Technical Details**: 제안된 방법인 Cluster-Aware Replay (CAR)는 작은 클래스 균형 회상 버퍼와 Inter-Cluster Fitness Function이라는 새로운 손실 항으로 구성됩니다. 모델은 표준 ResNet-18 아키텍처를 사용하고, 이미지를 입력받아 피처를 추출한 후 이를 분류합니다. 최종 손실 함수는 기존의 분류 손실(Cross-Entropy)과 Inter-Cluster Fitness 손실의 가중 합으로 구성되며, 하이퍼파라미터 λ는 새로운 작업 수학(learning)의 신뢰성과 이전 지식의 보존 간의 균형을 조정합니다.

- **Performance Highlights**: Split CIFAR-10 벤치마크를 사용한 실험 결과, ICS 기법이 초기 작업의 수행률을 유지하는 데 효과적임을 보여주었습니다. 그러나 작업 수가 증가할 경우 장기적인 지식 보존에 한계가 드러났으며, 이는 continual learning의 복잡성 및 트레이드오프를 강조합니다. 이 연구는 향후 연구 방향의 가능성을 제시하며, 지속적인 학습 체계에서의 모델의 발전 가능성을 보여줍니다.



### Design-Based Bandits Under Network Interference: Trade-Off Between Regret and Statistical Inferenc (https://arxiv.org/abs/2510.07646)
- **What's New**: 논문에서는 네트워크 간섭이 있는 다중 선택형 밴디트 문제(MABNI)의 새로운 이론적 경계, 즉 파레토 프론티어(Pareto frontier)를 제시하여 후회(minimizing regret)와 추론 정확도(inference accuracy) 간의 트레이드오프를 명확히 한다. 저자들은 처음으로 적대적 MABNI 환경에서 이러한 트레이드오프를 정량적으로 분석한다. 이를 통해 과거 연구의 한계를 극복하고, 후회 최소화에 대한 집착이 서브 최적 무기(sub-optimal arms)의 추론 정확도를 저하할 수 있음을 강조한다.

- **Technical Details**: 제안된 방법론 중 하나인 EXP3-N-CS 알고리즘은 언제든지 유효한(asymptotic) 신뢰구간(confidence sequence)을 제공하며, 이는 적대적 MAB-N에서 지속적인 추론을 가능하게 한다. EXP3-N-CS는 후회 최소화와 추론 정확도 간의 균형을 잡으면서 실험 데이터로부터 인과 효과를 정확히 추정할 수 있도록 설계되었다. 기법적 측면에서, 저자들은 노출 매핑(exposure mapping)과 클러스터링(clustering)의 기술을 이용해 MAB-N의 설계를 지원한다.

- **Performance Highlights**: 시뮬레이션 연구를 통해 EXP3-N-CS의 경험적 성능을 검토하였으며, 결과는 제안된 알고리즘이 세 가지 주요 학습 목표를 효과적으로 충족함을 보여준다. 특히 EXP3-N-CS는 적대적 환경에서도 신뢰할 수 있는 통계적 추론을 제공하면서 후회 최소화 또한 실현할 수 있는 가능성을 나타낸다. 이러한 실험 결과는 향후 실질적인 적용 가능성을 더한다.



### Property Classification of Vacation Rental Properties during Covid-19 (https://arxiv.org/abs/2510.07639)
Comments:
          GISRUK 2024 Poster

- **What's New**: 이 연구는 Covid 팬데믹 동안 활성화된 휴가 렌탈(property) 속성을 분류하기 위해 군집화(clustering) 기법을 사용하는 것을 주장합니다. 100만 개 이상의 속성과 호스트(host) 데이터를 포함하는 이 데이터 세트는 ESRC가 지원하는 Consumer Data Research Centre(CDRC)와 AirDNA 간의 협업으로 만들어졌습니다. 이를 통해 숨겨진 패턴과 행동을 식별할 수 있습니다.

- **Technical Details**: 연구에서는 K-means 및 K-medoids 군집화 기법을 활용하여 동질적인 그룹과 이들의 공통 특성을 파악하고 있습니다. 이러한 분석 방법은 데이터의 복잡성을 효과적으로 정리하여 휴가 렌탈 평가의 내적 요소를 이해하는 데 기여합니다.

- **Performance Highlights**: 연구 결과는 휴가 렌탈(property) 평가의 복잡성을 이해하는 데 도움을 줄 뿐만 아니라, 특정 그룹(cluster)에 맞춤형으로 정책을 개발할 수 있는 가능성을 제시합니다. 이는 향후 정책을 수립하는 데 중요한 역할을 할 것으로 기대됩니다.



### LLM Unlearning Under the Microscope: A Full-Stack View on Methods and Metrics (https://arxiv.org/abs/2510.07626)
- **What's New**: 최근 LLM(대형 언어 모델) 관련 연구는 모델의 안전성, 프라이버시 및 저작권 문제를 해결하기 위한 기계 학습 해제(machine unlearning)의 필요성을 강조하고 있습니다. 본 연구는 기존의 무질서한 연구 상황 속에서, 12개의 최신 상태 기반(unlearning) 방법을 세 가지로 분류하는 체계적 관점을 제공합니다: divergence-driven optimization, representation misalignment, rejection-based targeted unlearning. 이러한 방법의 성능 평가를 위해 새로운 Open-QA(오픈 질문 응답) 메트릭스를 제안하며, 이는 LLM의 생성 성능을 더 잘 포착할 수 있습니다.

- **Technical Details**: 연구에서는 기존의 평가 방식이 다각적인 관점을 제공하지 못하는 점을 지적하며, Open-QA를 통해 보다 정교한 평가를 요구합니다. 12개의 최근 LLM 해제 방법은 크게 세 가지 방법론으로 나뉘며, 각각의 알고리즘은 유사성을 지닌 하위 범주로 세분화됩니다. 예를 들어, divergence-driven optimization 방법은 모델이 기준 분포에서 멀어지도록 최적화하는 반면, rejection-based targeted unlearning 방법은 잊어야 할 쿼리에 명확한 거부 응답을 생성하는 방식을 사용합니다.

- **Performance Highlights**: WMDP(Weapons of Mass Destruction Proxy) 벤치마크를 활용하여 해제 효과성(UE)과 유틸리티 유지(UT) 측면에서 성능을 평가하였지만, 기존의 다중 선택 질문(MCQ) 기반의 평가 방식은 모델의 실제 생성 행동을 반영하지 못하는 경우가 많았습니다. 연구팀은 다양한 LLM 해제 방법들의 로버스트니스(강건성) 분석을 통해, 모델 레벨 공격과 관련된 다양한 취약점을 분석하였으며, 이는 향후 LLM 해제 기술 평가에 매우 중요한 시사점을 제공합니다.



### DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation with Uncertainty-Quantification Suppor (https://arxiv.org/abs/2510.07620)
Comments:
          18 pages, 9 figures, 5 tables

- **What's New**: DGTEN (Deep Gaussian-based Trust Evaluation Network)은 변화하는 관계를 효과적으로 포착하고 신뢰를 평가하기 위한 통합 그래프 프레임워크를 제안한다. 이 프레임워크는 불확실성을 인식하는 메시지 패싱(message passing)과 표현력이 뛰어난 시간 모델링(temporal modeling)을 통합하여 신뢰-targeted 공격에 대해 내장된 방어 기능을 제공한다. DGTEN은 노드와 엣지를 가우시안 분포(Gaussian distributions)로 표현하여 신뢰 결정을 내릴 때 더 높은 위험 인식(risk-aware) 판단을 가능하게 한다.

- **Technical Details**: DGTEN은 하이브리드 절대-가우시안-아워글래스(Hybrid Absolute-Gaussian-Hourglass, HAGH) 포지셔널 인코딩(positional encoding)과 Kolmogorov-Arnold 네트워크 기반의 다중 헤드 주의(multi-head attention) 방법을 채택한다. 또한, 일반적인 미분 방정식(ODE)을 기반으로 한 잔여 학습 모듈(residual learning module)을 통해 신뢰가 어떻게 발전하는지를 모델링한다. 마지막으로, RAECA(Robust Adaptive Ensemble Coefficient Analysis) 방어 기법을 사용하여 의심스러운 상호작용을 식별하고 이를 완화하여 공격에 대한 저항성을 높인다.

- **Performance Highlights**: DGTEN은 비트코인 신뢰 네트워크를 대상으로 한 실험에서 유의미한 개선을 보였다. 단일 시간 간격 예측(single-timeslot prediction)에서 기존 동적 기준선(baseline)보다 10.77% 더 높은 MCC를 기록했으며, 콜드 스타트(cold-start) 시나리오에서는 16.41%의 MCC 이득을 달성했다. 적대적 공격과의 비교에서도 DGTEN은 최대 11.63%의 개선을 보여주며, 저항성과 유용성을 입증하였다.



### Transformer-Based Indirect Structural Health Monitoring of Rail Infrastructure with Attention-Driven Detection and Localization of Transient Defects (https://arxiv.org/abs/2510.07606)
Comments:
          Preprint presented at the 15th International Workshop on Structural Health Monitoring (IWSHM)

- **What's New**: 이 연구는 비지도 심층 학습(unsupervised deep learning)을 사용하여 기차 선로의 손상 검출(indirect structural health monitoring, iSHM)에 대한 새로운 접근 방식을 제시합니다. 고속 변화, 다채널 입력 및 현실적인 노이즈 패턴에 대한 점진적으로 복잡한 문제를 평가하기 위해 점진적 합성 데이터 벤치마크를 도입했습니다. 또한, 자가 집중(self-attention) 메커니즘을 사용하는 Attention-Focused Transformer 모델을 개발하여 노이즈에 대한 내성과 계산 효율성을 높였습니다.

- **Technical Details**: 모델은 학습된 주의력 가중치(attention weights)의 편차를 통해 이상 점수(anomaly scores)를 획득하며, 이는 주로 재구성(reconstruction)을 통해 훈련됩니다. 기존의 여러 비지도 모델과 비교하여 제안된 모델의 성능을 평가하였고, Transformer 기반 모델이 일반적으로 다른 모델보다 우수한 성능을 보였습니다. 특히 고주파 국부 노이즈(high-frequency localized noise)에 대한 취약성을 발견하여 이를 실용적인 배치의 큰 장애물로 확인했습니다.

- **Performance Highlights**: 제안된 모델은 최첨단 솔루션(state-of-the-art solution)과 유사한 정확도를 달성하는 동시에 더 나은 추론 속도(inference speed)를 보여주었습니다. 이는 향후 iSHM 모델에서 노이즈 내성을 강화할 필요성을 강조하며, 더욱 효율적인 주의 기반 접근 방식이 기초 차선 탐지 시스템의 개발에 유망한 기반이 될 수 있음을 시사합니다.



### TGM: a Modular and Efficient Library for Machine Learning on Temporal Graphs (https://arxiv.org/abs/2510.07586)
Comments:
          21 pages, 5 figures, 14 tables

- **What's New**: 이번 논문에서는 Temporal Graph Modelling (TGM)이라는 연구 지향의 라이브러리를 소개합니다. 이는 두 가지 시간 동적 그래프 접근 방법, 즉 Continuous-Time Dynamic Graph (CTDG)와 Discrete-Time Dynamic Graph (DTDG)를 통합한 최초의 라이브러리입니다. TGM은 동적인 노드 특성, 시간-세분화 변환, 링크 및 노드 수준의 작업을 원활히 지원하여 사용자 경험을 향상시킵니다.

- **Technical Details**: TGM은 모듈식이고 효율적인 프레임워크로, 연구자들이 시간 동적 그래프 내에서 더 쉽게 실험할 수 있도록 돕습니다. 이 라이브러리는 8개 방법을 구현하며, 시간 granularity를 API에 자연스럽게 통합하여 그래프 세분화와 스냅샷 반복을 지원합니다. 또한, TGM은 그래프 모델 훈련 속도에서 DyGLib보다 평균 7.8배 더 빠르며, 그래프 세분화에서 175배 더 빠른 성능을 보여줍니다.

- **Performance Highlights**: TGM은 동적 그래프 속성 예측과 시간 기반 훈련 패러다임을 실현하여 연구 가능성을 확대합니다. 실험을 통해 TGM이 기존 구현에 비해 평균 175배 더 빠른 그래프 세분화 성능을 발휘함을 입증하였으며, 이는 연구 시간이 단축되고 새로운 통찰력을 제공할 수 있는 기회를 창출합니다. TGM은 링크, 노드 및 그래프 수준의 예측 작업에서 가장 폭넓은 지원을 제공하는 특징을 가지고 있습니다.



### Expanding the Action Space of LLMs to Reason Beyond Languag (https://arxiv.org/abs/2510.07581)
- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 발전은 이들을 단순한 언어 추론 기계에서 외부 환경과 상호 작용할 수 있는 다재다능한 에이전트로 확장하는 데 기여했습니다. 이 연구는 두 가지 상호 보완적인 관점을 제시합니다: 외부 환경이 LLM의 능력을 증강할 수 있다는 점과 LLM이 언어 지시를 외부 환경의 작업으로 매핑할 수 있다는 것입니다. 인상적인 점은 이러한 접근법이 기계의 언어 기반 추론과 외부 환경의 상호작용을 분리하여 효율성을 높인다는 것입니다.

- **Technical Details**: 이 논문은 Expanded Action space (ExpA)라는 새로운 패러다임을 제안하여, 언어 기반 추론과 환경 상호 작용을 분리합니다. 모델은 기본 언어 환경에서 작동하면서도 라우팅 액션을 통해 외부 환경으로 전환할 수 있습니다. 또한 ExpA Reinforcement Learning (EARL) 기법을 도입하여 이 새로운 환경을 탐색할 수 있도록 합니다. 이 과정에서 카운터팩추얼 정책 최적화(counterfactual policy optimization)를 통해 환경-specific actions을 학습합니다.

- **Performance Highlights**: 이 연구의 EARL 알고리즘은 다중 턴 작업(multi-turn tasks)과 조건부 계획이 필요한 작업에서 기존의 어휘 제약이 있는 기초 모델들이 제공한 성과를 상회했습니다. 또한 부분적으로 관찰된 정렬 문제(partially observed sorting problem)에서는 Sort-4의 완벽한 정확도를 달성하며 효율적인 알고리즘을 스스로 발견해 낸 사례도 포함되어 있습니다. 이는 기존의 모델과 비교할 때 높은 성과를 보여주는 결과입니다.



### Accuracy, Memory Efficiency and Generalization: A Comparative Study on Liquid Neural Networks and Recurrent Neural Networks (https://arxiv.org/abs/2510.07578)
Comments:
          13 pages, 12 figures. Submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

- **What's New**: 이번 논문에서는 LNN(Liquid Neural Networks)과 기존 RNN(Recurrent Neural Networks) 구조의 비교 분석을 통해, 각 모델의 정확성(accuracy), 메모리 효율(memory efficiency), 일반화 능력(generalization ability)을 평가합니다. LNN은 생물학적 신경 시스템에서 영감을 받아, 불규칙하게 샘플링된 데이터를 보다 효과적으로 처리할 수 있는 잠재력을 지니고 있습니다. 또한, LNN의 몇몇 변종은 전통적인 RNN에 비해 파라미터 효율성과 연산 속도에서 우수한 성능을 보입니다.

- **Technical Details**: 기존 RNN은 내부 재귀 구조를 통해 순차적 데이터의 시간 의존성을 포착하고, LSTM(Long Short-Term Memory)과 GRU(Gated Recurrent Units)와 같은 변형들이 그 성능을 향상시켜왔습니다. 반면, LNN은 연속적인 시간 동적 시스템 이론을 기반으로 하여, 일반적인 미분 방정식(ODEs)을 통해 신경 상태의 연속적 변화를 묘사합니다. 이는 더 높은 일반화 능력과 함께 노이즈에 강한 성능을 보일 수 있게 합니다.

- **Performance Highlights**: 현재 LNN은 전통적인 RNN이 가진 긴 의존성 학습의 한계와 기울기 문제를 극복할 수 있는 가능성을 보여주고 있습니다. 연구 결과는 LNN이 다양한 실제 사례에서 예측 정확성을 높이고 전통적인 모델보다 우수한 성능을 발휘할 수 있음을 시사합니다. 특히, 향후 연구 방향으로 LNN의 확장 가능성을 강조하며 다양한 복잡한 시나리오에서의 응용 가능성을 모색하고 있습니다.



### Symbolic-Diffusion: Deep Learning Based Symbolic Regression with D3PM Discrete Token Diffusion (https://arxiv.org/abs/2510.07570)
Comments:
          9 Pages, 3 Figurees

- **What's New**: 이번 연구에서는 D3PM 기반의 이산 상태 공간 확산 모델인 Symbolic Diffusion을 제안했습니다. 이 모델은 모든 토큰을 동시에 생성하여 기호 회귀(symbolic regression)에 대한 새로운 접근 방식을 제공합니다. 기존의 오토리그레시브(autoregressive) 방식의 한계를 극복하고, 보다 개선된 폐쇄형 수식(closed-form equation)을 생성하는 데 중점을 두었습니다.

- **Technical Details**: Symbolic Diffusion은 Pointnet 스타일의 특성 인코더(feature encoder)와 D3PM 아키텍처를 기반으로 하는 변환기(decoder)를 사용하여 수식의 진실 토큰을 예측합니다. 기존의 오토리그레시브 모델과 동일한 인코더 및 변환기 아키텍처를 활용하여, 모든 토큰을 동시에 생성하는 방식을 통해 글로벌 컨텍스트(global context)를 강화합니다. 또한, 500,000개의 샘플을 사용하는 이차 변수(bivariate) 데이터셋을 훈련 및 평가에 활용하였습니다.

- **Performance Highlights**: Symbolic Diffusion은 SymbolicGPT에 기반한 오토리그레시브 모델과 비교하여 통계적으로 유의미한 성능 향상을 보였습니다. 특히, 평균 R² 값에서 개선된 결과를 나타내었으며, 비슷한 기초 아키텍처를 사용하는 기존 모델들보다 유사하거나 더 나은 성능을 제공했습니다. 이러한 결과는 기호 회귀 분야에서 기계학습 기반 접근 방식의 새로운 연구 기회를 여는 계기가 될 것입니다.



### Automated Machine Learning for Unsupervised Tabular Tasks (https://arxiv.org/abs/2510.07569)
Comments:
          Accepted at Machine Learning Journal, 2025

- **What's New**: 이 논문에서는 LOTUS(Learning to Learn with Optimal Transport for Unsupervised Scenarios)라는 방법을 제안합니다. 이 방법은 이상치 탐지(outlier detection) 및 클러스터링(clustering)과 같은 여러 비지도 머신러닝(ML) 작업을 위한 모델 선택을 수행하기 위한 효과적인 접근법입니다. 연구의 기본 아이디어는 기계학습 파이프라인이 유사한 데이터 분포를 가진 데이터셋에서 성공적으로 작동한 경우 새로운 데이터셋에서도 잘 작동한다는 것입니다.

- **Technical Details**: LOTUS는 최적 수송 거리(optimal transport distances)를 활용하여 여러 비지도 작업을 위한 모델 선택을 자동화합니다. LOTUS는 두 가지 단계의 메타 학습(meta-learning) 시스템으로 구성되어 있습니다. 첫 번째 단계에서 이전 작업에서 수집된 데이터를 통해 알고리즘의 성능을 학습하고, 두 번째 단계에서는 현재 데이터셋과 가장 유사한 데이터셋을 찾아 최적 알고리즘을 추천합니다.

- **Performance Highlights**: 많은 실험을 통해 LOTUS는 강력한Baseline과 비교하여 비지도 작업에서 상당히 개선된 결과를 보여줍니다. 또한, LOTUS의 구현을 기반으로 한 두 개의 오픈 소스 AutoML 시스템이 제공되어 이상치 탐지 및 클러스터링을 위한 감독 모델 선택이 가능합니다. 코드 및 추가 정보는 https://github.com/prabhant/LOTUS-CL-OD에서 확인할 수 있습니다.



### EBGAN-MDN: An Energy-Based Adversarial Framework for Multi-Modal Behavior Cloning (https://arxiv.org/abs/2510.07562)
- **What's New**: 이번 연구에서는 EBGAN-MDN을 제안하여 다중 모드(multi-modal) 행동 복제에서의 한계를 극복하고자 합니다. 다중 모드 데이터의 복잡성을 고려할 수 있도록 에너지 기반 모델(EBM)과 혼합 밀도 네트워크(MDN)를 결합하고, 이를 적대적 훈련을 통해 구현한 점이 주목할 만합니다. 특히, 수정된 InfoNCE 손실 함수와 에너지를 강제화한 MDN 손실 함수를 도입하여 기존의 문제인 모드 붕괴(mode collapse)와 모드 평균화(mode averaging)를 해결합니다.

- **Technical Details**: EBGAN-MDN은 에너지 기반 모델을 활용하여 입력과 출력 쌍에 에너지를 할당하며, 더욱 다채로운 출력을 생성하는 데 중점을 둡니다. 반면에 전통적인 생성기(Generator) 대신 MDN을 사용하여 평균 혼합계수(GMM)를 매개변수화합니다. 이를 통해 생성기는 모드 간 충돌(conflicting outputs)을 생성하고, 각 입력에 대해 다중 모드 분포를 명시적으로 포착할 수 있습니다.

- **Performance Highlights**: 실험 결과, EBGAN-MDN은 인공 및 로봇 벤치마크 실험에서 기존의 최첨단 모델들과 비교하여 우수한 성능을 보여주었습니다. 특히, 모드 커버리지(mode coverage), 샘플 품질(sample quality), 확장성(scalability) 측면에서 모두 높은 평가를 받았습니다. 이는 EBGAN-MDN이 다중 모드 학습(task)에 대한 효율적이고 효과적인 솔루션임을 입증합니다.



### Investigating Thematic Patterns and User Preferences in LLM Interactions using BERTopic (https://arxiv.org/abs/2510.07557)
- **What's New**: 이 연구는 변환기 기반의 주제 모델링 기법인 BERTopic을 다국어 대화 말뭉치인 lmsys-chat-1m 데이터 셋에 적용하였습니다. 이 데이터 셋은 대형 언어 모델(LLMs)에 대한 평가를 통해 수집된 사용자 프롬프트와 두 개의 비식별 LLM 응답을 포함하고 있습니다. 연구의 주요 목표는 이러한 대화에서 주제 패턴을 발굴하고, 특정 LLM이 특정 주제 내에서 일관되게 선호되는 것을 분석하는 것입니다.

- **Technical Details**: BERTopic은 고차원 변환기 기반 임베딩을 활용하여 주제 모델링을 수행하며, HDBSCAN을 클러스터링 알고리즘으로 사용하여 유사한 문서의 그룹을 식별합니다. 연구에서는 29개 이상의 일관된 주제를 추출하였으며, 이는 인공지능, 프로그래밍, 윤리 및 클라우드 인프라와 같은 다양한 기술적 주제를 포함하고 있습니다. 또한, 문서 전처리 파이프라인을 설계하여 다국어 변형을 균형 있게 유지하고 노이즈나 편집된 데이터를 정리했습니다.

- **Performance Highlights**: 분석 결과, 주제와 모델 선호도의 관계를 파악하여 모델-주제 정렬의 경향을 확인했습니다. 데이터 시각화 기법으로는 주제 간 거리 맵과 모델 대 주제 매트릭스가 포함되어 있으며, 이는 LLM의 실제 성능과 사용자 만족도를 향상시키기 위한 도메인 특화된 미세 조정 및 최적화 전략을 알리는 데 기여합니다.



### Phase Diagram of Dropout for Two-Layer Neural Networks in the Mean-Field Regim (https://arxiv.org/abs/2510.07554)
- **What's New**: 이번 연구에서는 드롭아웃(dropout)의 역할을 이해하기 위한 첫 단계로, 두 층 신경망에서의 드롭아웃을 통한 경량화 경향을 분석합니다. 특히, 드롭아웃 비율과 학습률, 네트워크 폭의 상대적 크기에 따라 다섯 가지 뚜렷한 비퇴화(nondegenerate) 단계가 존재함을 발견했습니다. 기존에 잘 알려진 드롭아웃의 '페널티(penalty)' 효과가 극단적으로 작은 학습률에서만 지속된다는 점이 주목할 만합니다. 이러한 결과는 대규모 신경 네트워크에서 드롭아웃의 이론적 이해를 새롭게 정립할 수 있는 기초를 마련합니다.

- **Technical Details**: 연구는 두 층 네트워크를 대상으로 한 드롭아웃 및 경량화된 이론적 설정을 수립합니다. 드롭아웃이 그래디언트에서 유도되는 노이즈의 분해를 통해 '드롭아웃 페널티'를 정의하였고, 특히 두 가지 주요한 결합 기술을 통해 결과를 도출했습니다. 첫 번째는 페널라이즈드(Penalized) Wasserstein 그래디언트 플로우로 수렴성을 확립했으며, 두 번째는 평균 장(jump process)으로의 수렴입니다. 이러한 접근 방법은 경량화된 신경망의 동적 특성의 질적인 속성을 분석하는 데 도움을 줍니다.

- **Performance Highlights**: 프로포지션 3에서는 더 큰 학습률 하에서 드롭아웃의 비대칭 체계가 무작위 메트릭을 통한 그래디언트 하강과 동등하다는 점을 밝히며, 이는 특정 특징들이 손실의 지역적 날카로움에 의해 제한받지 않고 더 큰 스텝을 허용한다는 강점을 가집니다. 연구 결과는 드롭아웃의 비대칭 효과가 단순한 규제의 필요성을 초월하며, 대규모의 신경망에서 드롭아웃의 활용 가능성을 제시합니다. 이런 면에서, 이러한 결과는 실제적인 학습 상황에서도 드롭아웃의 효과적인 적용을 보여줄 것으로 기대됩니다.



### Targeted Digital Twin via Flow Map Learning and Its Application to Fluid Dynamics (https://arxiv.org/abs/2510.07549)
- **What's New**: 이번 논문에서는 Targeted Digital Twin (tDT)이라는 개념을 소개하고, 이를 구성하기 위한 수치적 방법론을 제시합니다. tDT는 전체 디지털 트윈(Digital Twin, DT)의 중요한 양(QoI)의 동력을 모델링하는데 주안점을 두고 있으며, Flow Map Learning (FML)이라는 메모리 기반 학습 방식을 사용하여 데이터 기반 모델을 개발합니다. 이 방식은 전체 DT의 시뮬레이션 없이 효율적으로 양(QoI)의 장기 동태를 분석하고 예측할 수 있게 하여 계산 효율성을 크게 향상시킵니다.

- **Technical Details**: tDT는 물리적 시스템의 완전한 모델을 나타내는 디지털 트윈(Digital Twin)과 이를 기반으로 중요 양(QoI)의 동태를 모델링하는 동적 시스템으로 정의됩니다. 논문에서는 tDT 구축을 위해 복잡한 비선형 상호작용이 포함된 상태 변수와 시스템 파라미터, 비선형 하위 시스템들을 고려하여 수치적 설정에서의 모델링을 수행합니다. 이를 통해 메모리 기반 FML 방법론을 바탕으로 대규모 물리적 프로세스를 요약하는 소형 모델을 생성합니다.

- **Performance Highlights**: 제안된 tDT는 원통 주위의 2차원 비압축 유동 문제를 연구함에 있어 유동에 의해 작용되는 총 수력 힘을 예측하는 데 사용됩니다. tDT는 드래그(Drag)와 리프트(Lift)와 같은 두 출력만을 가짐으로써 매우 컴팩트한 모델로, 초기 조건을 이용해 DT와 동기화된 후, 추가적인 정보없이 장기 예측이 가능합니다. 실험 결과, tDT가 전체 유동 시뮬레이션을 우회하며도 정확한 예측을 제공함을 보여줍니다.



### Estimating Fair Graphs from Graph-Stationary Data (https://arxiv.org/abs/2510.07536)
- **What's New**: 이 논문에서는 민감한 속성과 관련하여 편향되지 않은 연결을 갖는 공정한 그래프(fair graphs)를 추정하는 방법을 제안합니다. 실제 그래프의 엣지는 종종 특정 그룹 쌍 간의 연결을 선호하는 경향이 있어, 이러한 편향된 연결은 그래프 기반 작업에서 불공정한 처리를 유발할 수 있습니다. 따라서, 그룹 및 개인 공정성을 고려하여 다양한 편향 메트릭을 평가합니다.

- **Technical Details**: 저자들은 Fair Spectral Templates (FairSpecTemp)라는 최적화 기반의 방법을 제안하여 고정된 그래프 신호에서 공정한 그래프를 추정합니다. 이 방법에는 그래프의 정적 속성(commutativity properties)을 활용하는 변형과 그래프 스펙트럼 내에서 편향을 제약하여 보다 유연한 추정을 가능하게 하는 변형이 포함됩니다. 고성능 확률 성능 경계를 통해 공정성과 정확성 간의 조건부 트레이드오프를 제공합니다.

- **Performance Highlights**: FairSpecTemp는 합성 및 실제 데이터 세트에서 효과성을 평가하였고, 두 가지 변형의 장점을 강조합니다. 특히 정확성을 희생하지 않고도 공정한 그래프를 복구할 수 있다는 분석 결과를 제시합니다. 이를 통해 이 방법이 제공하는 정확성과 공정성 간의 균형이 중요함을 강조합니다.



### EEG Sleep Stage Classification with Continuous Wavelet Transform and Deep Learning (https://arxiv.org/abs/2510.07524)
Comments:
          11 pages, 2 figures

- **What's New**: 본 연구에서는 수면 장애의 정확한 진단과 관리를 위해 수면 단계 분류의 자동화된 새로운 프레임워크를 제안합니다. 기존의 수면 채점 방법은 EEG 신호에서 수동 주석 또는 시간 및 주파수 영역의 특징 추출에 의존하고 있었습니다. 본 연구는 웨이블릿 변환 기반의 시간-주파수 분석을 사용하여 이를 개선합니다.

- **Technical Details**: 제안된 방법은 Continuous Wavelet Transform (CWT)을 사용하여 수면 단계와 관련된 주파수 대역 간의 과도한(transient) 및 진동성(oscillatory) 패턴을 캡처하는 시간-주파수 맵을 생성합니다. 이를 통해 얻은 웨이블릿 기반 표현은 앙상블 학습(ensemble learning)과 결합되어 높은 분류 정확도를 달성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 88.37%의 전체 정확도와 73.15의 매크로 평균 F1 점수를 기록하여 기존 기계 학습 방법보다 뛰어난 성과를 보였습니다. 이는 최신 딥 러닝(deep learning) 접근법과 비교할 때도 유사하거나 우수한 성능을 나타내며, 수면 단계 분류의 임상적 적용 가능성을 강조합니다.



### MLLM4TS: Leveraging Vision and Multimodal Language Models for General Time-Series Analysis (https://arxiv.org/abs/2510.07513)
- **What's New**: 이번 연구에서는 MLLM4TS라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 멀티모달 대형 언어 모델(multimodal large language models)을 활용하여 시계열(time series) 분석을 수행합니다. 특히, 눈에 띄는 시각적 표현을 통해 시계열 데이터를 분석하는 자동화된 방법의 효과를 높일 수 있는지 탐구합니다.

- **Technical Details**: MLLM4TS는 시간을 나타내는 시리즈를 색상 코드가 있는 선 그래프(color-coded line plot) 형태로 변환하여 시각적 패턴을 캡처합니다. 이를 통해 각 채널 간의 공간 의존성을 효과적으로 파악하고, 각 시간 구간에 맞춰 시각적 패치를 정렬하는 전략을 도입합니다. 이러한 방법은 수치적 데이터와 시각적 표현에서 파생된 글로벌 맥락 정보를 융합하여 정교한 시간 역학을 모델링합니다.

- **Performance Highlights**: MLLM4TS는 분류(classification), 이상 탐지(anomaly detection), 예측(forecasting) 등 다양한 표준 벤치마크에서 좋은 성과를 보여주었습니다. 이 연구는 사전 훈련된 언어 모델(pretrained language models)과 시각적 모달리티(visual modalities)를 통합함으로써 시계열 분석의 강력하고 일반화 가능한 접근법을 제시합니다. 특히, 몇 샷(few-shot) 및 제로 샷(zero-shot) 학습 환경에서도 뛰어난 일반화 성능을 지니고 있습니다.



### Efficient Generalization via Multimodal Co-Training under Data Scarcity and Distribution Shif (https://arxiv.org/abs/2510.07509)
- **What's New**: 이 논문은 제한된 레이블 데이터 상황에서 모델의 일반화를 개선하기 위한 다중 모드 협동 학습(multimodal co-training) 프레임워크를 탐색합니다. 이 연구는 레이블이 없는 데이터의 활용과 다양한 모드의 분류기 간의 일치성 증진이 일반화 개선에 어떻게 기여하는지를 이론적으로 분석합니다. 또한, 레이블이 없는 다중 모드 데이터를 활용하며 조건부 뷰 독립성을 유지하는 것의 이점을 정량적으로 평가하는 새로운 일반화 경계를 제시합니다.

- **Technical Details**: 저자들은 두 가지 이상의 다른 모드를 통합하는 다중 모드 협동 학습 프레임워크를 제안하며, 이는 피어리와 재선택 방법을 포함하여 협동 학습 과정의 기하급수적 수렴에 대한 엄격한 증명을 제공합니다. 이 과정에서, 교환된 의사 레이블과 일치 손실이 어떻게 분류기 오류를 기대값 기준으로 감소시키는지를 조사합니다. 이론 기반으로 세부 분석을 통해 다중 모드의 데이터와 레이블이 없는 데이터를 최대한 활용하는 방법을 다룹니다.

- **Performance Highlights**: 제안된 프레임워크는 레이블이 적거나 유통이 변화하는 상황에서도 일반화 능력을 향상시킬 수 있음을 보여줍니다. 레이블 없는 데이터의 양을 늘리거나, 뷰 간의 일치를 증대시키는 전략을 사용함으로써 수렴 및 일반화 성능이 크게 향상된다는 것을 실험적으로 입증했습니다. 이러한 결과는 저자들이 제안한 구조적 접근법이 보다 데이터 효율적이고 적응력이 뛰어난 AI 시스템 개발에 기여할 수 있음을 강조합니다.



### PEAR: Planner-Executor Agent Robustness Benchmark (https://arxiv.org/abs/2510.07505)
- **What's New**: 이번 논문은 기존의 멀티 에이전트 시스템(중학생 시스템, MAS)의 취약점에 대한 포괄적인 이해 부족을 해결하기 위해 PEAR라는 새로운 벤치마크를 도입합니다. PEAR는 계획자-실행자(planner-executor) 아키텍처의 효용성과 취약성을 체계적으로 평가할 수 있는 구조를 제공합니다. 이는 다양한 MAS 아키텍처와 호환 가능하지만, 특히 널리 사용되는 계획자-실행자 구조에 초점을 맞추고 있습니다.

- **Technical Details**: PEAR 벤치마크는 4가지 시나리오와 총 84개의 사용자 작업(user tasks), 3가지 서로 다른 적대적 결과(adversarial consequences)를 가진 120개의 기본 공격 작업(base attack tasks), 그리고 1,680개의 공격받은 사용자 작업(attacked user tasks)을 포함합니다. 각 사용자 작업은 효용성 평가를 위한 평가 함수와 악의적 목표 달성을 위한 공격 작업 평가 함수를 페어링하여 제공합니다. 연구는 계획자가 약할 때 전체 클린 작업 성능이 더 심하게 저하된다는 결과를 보여줍니다.

- **Performance Highlights**: 계획자-실행자 시스템은 다양한 LLM 구조에서 65% 이상의 효용성을 달성하였습니다. 그러나 약한 LLM을 탑재한 에이전트는 성능 저하가 두드러졌습니다. 이 연구는 계획자 단계에서 발생하는 공격이 실행자보다 더 높은 공격 성공률(ASR)을 보이며, 이는 강력한 작업 수행 설정이 적대자로부터 더 취약하다는 것을 시사합니다.



### Black-box Detection of LLM-generated Text Using Generalized Jensen-Shannon Divergenc (https://arxiv.org/abs/2510.07500)
Comments:
          Preprint

- **What's New**: 본 논문에서는 SurpMark라는 기계 생성 텍스트를 탐지하기 위한 새로운 방법을 제안합니다. 이 방법은 불투명한 모델에서 검출하기 위해 기존의 텍스트 생성을 위한 참조 기반 접근 방식을 활용합니다. SurpMark는 토큰의 서프라이절(surprisal) 변화를 요약하여 문서를 평가하고, 이 과정을 통해 인간 작성 텍스트와 기계 생성 텍스트를 구분합니다.

- **Technical Details**: SurpMark는 평가할 텍스트의 토큰 서프라이절을 해석 가능한 상태로 양자화(quantize)하고, 이를 이용해 상태 전이 행렬을 계산합니다. 이 모델은 또한 일반화된 젠센-샤논(GJS) 점수를 통해 원본 인간 및 기계 데이터와의 차이를 평가합니다. 논문에서는 결정 통계량의 지수적 정상성(asymptotic normality)을 수립하고, 흑상자(black-box) 환경에서의 신뢰성을 유지합니다.

- **Performance Highlights**: SurpMark는 여러 데이터 세트와 모델, 시나리오에서 기존 방법들의 성과를 뛰어넘거나 동등한 성능을 보였습니다. 실험 결과는 SurpMark가 이론적으로 예측한 바와 일치하며, 제안된 양자화 과정의 효과성을 입증합니다. 결과적으로, 이 방법은 반복적인 모델 재학습이나 인스턴스 생성 없이 신뢰할 수 있는 텍스트 탐지가 가능하다는 장점을 제공합니다.



### Reinforcement Learning-based Task Offloading in the Internet of Wearable Things (https://arxiv.org/abs/2510.07487)
Comments:
          16 pages, 12 figures, Under review in the IEEE Internet of Things Journal

- **What's New**: 이 논문은 IoWT(Internet of Wearable Things)에서의 작업 오프로드 프로세스를 다루고 있으며, 에너지 소비와 작업 달성 시간 간의 트레이드오프를 고려한 강화 학습(Reinforcement Learning) 기반의 프레임워크를 제안합니다. 또한, 작업 오프로드 문제를 마르코프 결정 과정(Markov Decision Process)으로 모델링하여 Q-학습(Q-learning) 기법을 통해 최적의 결정을 내릴 수 있도록 합니다. 이를 통해 사용자 경험을 향상시키고 배터리 파워를 최적화할 수 있습니다.

- **Technical Details**: 작업 오프로드를 위한 Q-학습 기반의 알고리즘을 사용하여 IoT(Internet of Things) 장치가 동적 환경에서 최적의 결정을 내릴 수 있도록 하며, 무선 채널 조건 변화에 따른 성능 저하 문제를 해결하고자 합니다. 또한, 이를 위해 다수의 애플리케이션과 시스템 구성에 대한 광범위한 시뮬레이션을 수행하였습니다. 해당 연구에서는 작업 달성 시간, 에너지 소비, 오프로드된 작업의 비율 등 주요 성능 메트릭을 분석합니다.

- **Performance Highlights**: 제안된 알고리즘의 성능은 ns-3 네트워크 시뮬레이터를 통해 다양한 애플리케이션에 대해 평가되었으며, 평균 작업 달성 시간 및 에너지 소비에 관한 성능이 기존 방법들에 비해 개선된 결과를 보였습니다. 특히, Q-학습 알고리즘의 주요 시스템 매개변수들을 조정함으로써 성능 향상에 기여할 수 있는 가능성을 제시하고 있습니다. 이는 자원 제약이 있는 모바일 장치들이 효과적으로 작업을 수행할 수 있도록 지원합니다.



### HEMERA: A Human-Explainable Transformer Model for Estimating Lung Cancer Risk using GWAS Data (https://arxiv.org/abs/2510.07477)
Comments:
          18 pages, 6 figures, 3 tables

- **What's New**: 본 연구는 HEMERA(인간 설명 가능한 변환기 모델)를 소개하며, 이는 단일 뉴클레오타이드 다형성(SNP) 데이터의 유전체 전체 연관 연구(GWAS)에서 폐암 위험을 예측하기 위해 설명 가능한 변환기 기반의 딥러닝을 적용합니다. HEMERA는 기존 방법론과 달리 임상 공동 변수를 포함하지 않고 원시 유전자형 데이터를 직접 처리하며, 이는 개인화된 위험 평가와 조기 개입을 위해 중요한 혁신을 제공합니다.

- **Technical Details**: HEMERA는 유전체 연관 데이터에서 중요한 건축적 혁신을 실현하며, 원시 유전자형 데이터에서 직접 유전적 변이를 분리하여 그 예측 기여를 정량화할 수 있습니다. 이 모델은 ADDITIVE positional encoding과 신경망 임베딩 레이어를 도입하여 유전적 변이가 학습 가능한 표현을 갖도록 구성됩니다. 이 연구는 또한 LAYER-WISE INTEGRATED GRADIENTS 기반의 포스트 호크 설명 가능성 모듈을 통합하여 특정 SNP에 대한 예측 결과의 귀속을 가능하게 합니다.

- **Performance Highlights**: 27,254명의 참가자로부터 수집된 데이터를 기반으로 HEMERA는 99% 이상의 AUC(수신자 조작 특성 아래 영역) 점수를 달성하였습니다. 이러한 결과는 HEMERA가 정확한 위험 분류와 세밀한 기능 귀속을 제공해 폐암 취약성의 유전적 결정을 이해하는 데 큰 기여를 할 수 있음을 보여줍니다. HEMERA는 설명 가능한 모델로 향상된 예측 성능을 통해 유전적 위험을 밝히고, 이를 통해 폐암의 조기 발견을 위한 새로운 가능성의 길을 열고 있습니다.



### Surrogate Modeling for the Design of Optimal Lattice Structures using Tensor Completion (https://arxiv.org/abs/2510.07474)
Comments:
          NeurIPS 2025 AI4Mat Workshop

- **What's New**: 본 연구에서는 메커니컬 성능에 관한 최적의 격자 구조(optimal lattice structures)를 설계하는 데 있어 텐서 완성(tensor completion)을 대체 모델로 사용하는 방법을 제안합니다. 기존의 머신 러닝 방법들은 훈련 데이터가 균일하게 선택되지 않을 때 성능이 저하되는 경향이 있습니다. 텐서 완성을 통해 비균일한 샘플링으로 인한 한계를 극복하고, 설계 공간의 전체를 예측할 수 있는 가능성을 탐색합니다. 실험 결과, 텐서 완성은 전통적인 ML 방법들보다 약 5% 향상된 성능을 보였습니다.

- **Technical Details**: 이 연구에서는 텐서를 다차원 배열(multi-dimensional arrays)로 정의하고, 이를 통해 다양한 소재 설계 변수를 모델링합니다. 특히, 텐서 분해(tensor decomposition) 기술을 활용하여 관측된 자료에서 성능 예측을 수행하고, 주로 CPD(Canonical Polyadic Decomposition)와 NeAT(neural tensor completion) 방법을 사용하였습니다. 우리는 훈련 데이터의 부족으로 인해 발생할 수 있는 과적합(overfitting) 문제를 최소화하고, 실험 데이터를 비균일하게 샘플링한 경우의 성능을 검토하였습니다.

- **Performance Highlights**: 우리의 실험 결과, 텐서 완성 방법이 Gaussian Process 및 XGBoost와 같은 다른 머신 러닝 방법들에 비해 우수한 성능을 보였습니다. 특히, 비균일한 샘플링을 수행한 경우에도 텐서 완성이 더 좋은 결과를 보여주었습니다. 텐서 완성을 통한 접근 방식은 최적의 격자 구조 설계의 검색을 가속화할 수 있는 잠재력을 가지고 있습니다. 이 연구는 기계적 성능에 대한 격자 구조 설계 문제를 효과적으로 해결할 수 있는 가능성을 제시합니다.



### metabeta -- A fast neural model for Bayesian mixed-effects regression (https://arxiv.org/abs/2510.07473)
Comments:
          19 pages, 9 main text, 8 figures

- **What's New**: 이 연구에서는 Bayesian mixed-effects 회귀를 위한 새로운 모델인 metabeta를 소개합니다. 이 모델은 신경망 기반의 transformer 아키텍처를 사용하여, 시뮬레이션된 계층적 데이터셋에서 미리 정의된 파라미터를 통해 Bayesian 추론을 효율적으로 근사할 수 있도록 설계되었습니다. MCMC에 비해 현저하게 짧은 시간 내에 안정적이고 비슷한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: mixed-effects 회귀는 데이터 내의 그룹 간 의존성을 명시적으로 고려합니다. 이 모델은 고정 효과(fixed effects)와 랜덤 효과(random effects) 간의 차이를 이용해 모든 회귀 파라미터의 posterior 분포를 추정합니다. 또한, Importance Sampling과 Conformal Prediction을 통해 model outputs의 정밀함을 개선합니다.

- **Performance Highlights**: metabeta는 Hamiltonian Monte Carlo와 비교할 때 빠른 추론 시간을 보장하며, Bayesian mixed-effects 회귀의 적용 범위를 크게 넓힙니다. 전체 데이터셋과 priors를 입력으로 받아 regression 파라미터에 대한 posterior distributions를 반환하며, PyTorch에서 구현되어 오픈소스로 제공됩니다. 이를 통해 데이터 전문가들이 신속하게 Bayesian mixed-effects 회귀 모델을 활용할 수 있도록 돕고자 합니다.



### MoGU: Mixture-of-Gaussians with Uncertainty-based Gating for Time Series Forecasting (https://arxiv.org/abs/2510.07459)
- **What's New**: 이번 연구에서 우리는 Mixture-of-Gaussians with Uncertainty-based Gating (MoGU)를 소개하며, 이는 회귀 작업에 특화된 새로운 Mixture-of-Experts (MoE) 프레임워크입니다. MoGU는 각 전문가의 출력을 가우시안 분포로 모델링하여 예측값의 평균뿐만 아니라 그 내재적 불확실성(variance)도 직접적으로 정량화할 수 있습니다. 기존의 MoE와 차별화되는 점은 불확실성 기반의 게이팅 메커니즘을 적용하여 각 전문가의 기여도를 결정한다는 것입니다.

- **Technical Details**: MoGU는 각각의 전문가가 예측의 평균과 분산을 동시에 예측하는 구조로 되어 있습니다. 이러한 방식을 통해 전문가의 행동에 대한 정의로운 이해와 전체 모델의 불확실성을 도출할 수 있습니다. MoGU의 게이팅 메커니즘은 각 전문가의 추정된 불확실성을 기반으로 하여 최종 MoE 예측에 대한 기여도를 동적으로 결정하는 방식으로, 기존의 입력 기반 게이팅 메커니즘을 대체합니다.

- **Performance Highlights**: MoGU는 다양한 시간 시계열 예측 벤치마크에서 일관되게 더욱 정확한 예측을 제공하며, 이는 입력 기반 게이팅 MoE 아키텍처와 비교했을 때 두드러집니다. 예측 오차와 긍정적인 상관관계를 가진 불확실성 추정치를 제공함으로써 모델의 신뢰성과 그 출처에 대한 통찰력을 강화합니다. MoGU는 개별 전문가와 전체 모델 수준에서 불확실성을 잘 정량화하며, 이에 따라 예측 결과의 신뢰도를 향상시킵니다.



### Parameter-Free Federated TD Learning with Markov Noise in Heterogeneous Environments (https://arxiv.org/abs/2510.07436)
- **What's New**: 이 논문에서는 연합 강화 학습(Federated Reinforcement Learning, FRL)에서의 최적의 수렴 속도를 달성하기 위해 Polyak-Ruppert 평균(averaging) 기법을 도입한 두 가지 시간 규모의 연합 시간 차 학습(Federated Temporal Difference, FTD) 방법을 제안합니다. 기존의 TD 학습 알고리즘은 알려지지 않은 문제 매개변수에 의존해야 했습니다. 본 연구는 이 문제를 해결하여 Markovian 데이터를 위한 매개변수 없는 FTD 접근 방식을 제공합니다.

- **Technical Details**: 우리는 N개의 에이전트를 고려하는데, 각 에이전트는 마르코프 결정 프로세스(Markov Decision Process, MDP)에 접근할 수 있습니다. 각 에이전트는 고유한 보상 함수와 전이 확률을 가지며, 연구에서는 평균 보상 및 할인 설정 모두에서 O~(1/(NT)) 속도를 달성하는 것을 목표로 합니다. 또한, PR 평균 기법을 활용하여 비동기식 TD(0) 알고리즘의 수렴 속도를 높이는 방법론을 제시합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안한 방법이 잘 작동함을 보였습니다. 연합 환경에서 다양한 에이전트들이 공동으로 학습함으로써 성능 향상을 꾀할 수 있으며, 실험 결과에서 이론적인 예측과 일치하는 성과를 달성했습니다. 이는 FL 환경에서의 다양한 문제를 해결하는 데 중요한 기여를 할 수 있습니다.



### Learning to Route LLMs from Bandit Feedback: One Policy, Many Trade-offs (https://arxiv.org/abs/2510.07429)
Comments:
          16 pages, 3 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 위한 새로운 라우팅 프레임워크인 BaRP(Bandit-feedback Routing with Preferences)를 제안합니다. 기존의 라우팅 시스템들이 모든 후보 모델에 대한 정보를 요구하는 것과는 달리, BaRP는 실제 모델 선택에서 주어진 제한된 피드백만을 사용하여 최적화합니다. 또한, 사용자가 성능과 비용 간의 균형을 조정할 수 있도록 하는 유연성을 제공합니다.

- **Technical Details**: BaRP는 성능과 비용 간의 경쟁하는 목표를 균형 있게 처리하는 다중 목표(concatenated) 컨텍스트 밴딧 문제로 모델링됩니다. 이 시스템은 사용자의 선호 벡터를 기반으로 라우팅 결정을 내리며, 정책 네트워크는 입력 프롬프트와 선호 정보를 입력으로 받아들여 확률 분포를 생성합니다. 훈련 과정에서, 정책은 ENTROPY 정규화를 통해 탐색을 장려하며 부분 피드백 기반으로 업데이트됩니다.

- **Performance Highlights**: BaRP는 RouterBench와 두 개의 질문 답변 데이터셋에서 평가하였으며, 강력한 베이스라인보다 최소 12.46%, 최상위 LLM보다 최소 2.45% 성능 향상을 보여줍니다. 특히, 분포 내(in-distribution) 작업에서 3.81%의 성능 향상이, 분포 외(out-of-distribution) 작업에서 25.99%의 성능 이점을 기록했습니다. 이러한 결과는 BaRP가 실세계의 라우팅 문제를 해결하는 데 효과적임을 보여줍니다.



### Best-of-Both Worlds for linear contextual bandits with paid observations (https://arxiv.org/abs/2510.07424)
- **What's New**: 이번 연구에서는 유료 관측을 포함한 선형 맥락 밴디트(linear contextual bandits) 문제를 다룬다. 여기서 학습자는 주어진 맥락(context)에서 손실을 최소화하기 위해 행동을 선택하며, 특정 비용을 지불하고 각 팔(arm)의 손실을 관측할 수 있다. 우리는 Follow-the-Regularized-Leader 접근법을 기반으로 하여 Best-of-Both-Worlds (BoBW) 알고리즘을 제안하며, 이는 적대적(adversarial) 환경에서 최소 차이(regret) $	heta(T^{2/3})$를 달성한다.

- **Technical Details**: 우리의 알고리즘은 최근의 BoBW 알고리즘에서 아이디어를 확장하여 FTRL(Follow-the-Regularized-Leader) 프레임워크 내에서 설계되었다. 이 알고리즘은 확률적(stochastic) 및 적대적(adversarial) 환경 모두에서 손실 보장을 달성하여 설정의 주요 도전 과제를 해결한다. 또한, 우리는 맥락 분포의 가장 작은 비음수 고유값(λmin)이라는 새로운 주요 매개변수를 도입하여 알고리즘의 여러 매개변수를 특정 조정해야 함을 나타났다.

- **Performance Highlights**: 제안된 알고리즘은 유료 관측을 포함한 선형 맥락 밴디트 문제에 대해 효과적으로 작용하며, 이전 연구의 한계를 극복하고 더 강력한 손실 보장을 제공한다. 우리는 기존의 이론을 구체화하고, 알고리즘을 통해 유리한 결과를 얻었다. 이 연구는 맥락 모델링, 피드백 취득 비용, 보상 생성 과정의 불확실성을 동시에 포함하는 새로운 연구 방향을 제시한다.



### Encode, Think, Decode: Scaling test-time reasoning with recursive latent thoughts (https://arxiv.org/abs/2510.07358)
- **What's New**: 본 논문에서는 언어 모델의 추론 능력을 향상시키기 위한 새로운 방법인 Encode-Think-Decode (ETD)를 제안합니다. ETD는 모델의 기본 구조와 매개변수 수를 보존하면서도 미리 훈련된 모델이 특정 층에서 반복적으로 추론을 할 수 있도록 훈련시킵니다. 이 방법은 모델의 추론 성능을 크게 개선할 수 있는 간단하고 효과적인 경로를 제공합니다.

- **Technical Details**: ETD 방식은 모델이 인퍼런스 단계에서 선택된 층들을 반복적으로 활용하여 동작하도록 합니다. 이를 통해 기존의 LLM(large language models)의 잠재적인 추론 능력을 개선할 수 있습니다. 이전 연구들은 서로 다른 층에서의 정보 흐름이 추론 과정에서 중요한 역할을 한다고 밝히고 있으며, 본 연구는 이러한 사실을 바탕으로 층의 역할을 고려하여 ETD 구조를 설계하였습니다.

- **Performance Highlights**: 본 연구에서는 OLMo-2 1B 베이스 모델을 사용하여 17개의 추론 벤치마크에서 성능을 평가한 결과, GSM8K에서 28.4%의 상대적 정확도 향상과 MATH에서 36%의 향상을 달성했습니다. 이는 복잡한 구조의 변경 없이도 기존 모델의 성능을 높일 수 있는 가능성을 보여줍니다. 또한, 입력 토큰에 따라 동적으로 계산 깊이를 조정하는 방법을 제안하여 보다 효율적인 추론이 가능해졌습니다.



### ConCuR: Conciseness Makes State-of-the-Art Kernel Generation (https://arxiv.org/abs/2510.07356)
- **What's New**: 본 연구는 LLM을 활용한 GPU 커널 생성의 최근 발전을 다룹니다. 고급 CUDA 커널의 부족 문제를 해결하기 위해, 우리는 reasoning trace(추론 추적)를 포함한 고품질 CUDA 커널을 생성 및 큐레이팅하는 파이프라인을 개발했습니다. 이를 통해 ConCuR이라는 데이터셋과 KernelCoder라는 모델을 처음으로 소개합니다.

- **Technical Details**: 우리가 개발한 데이터 수집 및 큐레이션 파이프라인은 두 가지 부분으로 구성됩니다: 데이터 합성 및 데이터 큐레이션입니다. 데이터 합성 부분에서는 기존의 reasoning 모델을 사용하여 CUDA 커널과 함께 reasoning trace를 합성합니다. 데이터 큐레이션 부분에서는 reasoning trace의 간결성과 커널의 성능을 고려하여 높은 품질의 데이터셋을 구축합니다.

- **Performance Highlights**: KernelCoder는 ConCuR 데이터셋을 기반으로 훈련되어, 기존의 최상위 모델인 QwQ-32B에 비해 뛰어난 성능을 보입니다. KernelBench에서의 평가를 통해, KernelCoder가 모든 기존 커널 생성 모델보다 우수하며, GPT-4와 같은 최신 모델보다도 더 좋음을 증명했습니다. 추가적으로, reasoning length(추론 길이)가 커널 생성 작업의 복잡도를 평가하는 지표로 유용하다는 사실을 발견하였습니다.



### Out-of-Distribution Generalization in Climate-Aware Yield Prediction with Earth Observation Data (https://arxiv.org/abs/2510.07350)
- **What's New**: 이번 연구는 기후 변화가 농업 시스템에 미치는 영향과 관련된 농작물 수확량 예측의 중요성을 강조합니다. 기존의 딥러닝 모델이 위성 및 기상 데이터를 활용하여 수확량 예측에서 우수한 성과를 보였지만, 다양한 지역과 연도에 대해 일반화할 수 있는 능력은 거의 검증되지 않았습니다. GNN-RNN과 MMST-ViT라는 두 가지 최첨단 모델을 사용하여 대규모 CropNet 데이터셋을 기반으로 현실적인 환경에서 이를 평가하였습니다.

- **Technical Details**: CropNet 데이터셋은 2017년부터 2022년까지의 기후 변화에 대한 농작물 수확량 예측을 위해 설계된 대규모 공개 데이터셋입니다. 이 데이터셋은 40m 해상도의 Sentinel-2 이미지를 사용하며, 2291개의 미국 카운티에서 농업 모니터링을 최적화하였습니다. GNN-RNN 모델은 LSTM을 통해 다년간의 기상 데이터를 통합하고, MMST-ViT는 fused weather와 satellite inputs에 대한 attention 메커니즘을 활용하여 예측을 수행합니다.

- **Performance Highlights**: GNN-RNN 모델이 MMST-ViT보다 모든 작물에서 평균적으로 더 낮은 RMSE를 기록하며 뛰어난 성능을 보였습니다. 특히 GNN-RNN은 135배 더 빠른 훈련 속도를 기록하였고, 다양한 기후 변동성 아래에서도 긍정적인 상관관계를 유지함으로써 실질적인 예측에 더 적합하였습니다. 반면 MMST-ViT는 OOD(Out-Of-Distribution) 환경에서 성능 저하가 뚜렷하여, 보다 안정적인 성능 격차를 보였습니다.



### MultiFair: Multimodal Balanced Fairness-Aware Medical Classification with Dual-Level Gradient Modulation (https://arxiv.org/abs/2510.07328)
Comments:
          10 Pages

- **What's New**: 이 논문은 의료 분류를 위한 새로운 접근 방식인 MultiFair를 제안하며, 이는 다양한 데이터를 동시에 다루는 과정에서 생길 수 있는 공정성과 비균형 문제를 해결합니다. 기존의 다중 모달 학습 모델들이 두 가지 주요 문제, 즉 모달리티 학습 불균형(Modality Learning Bias)과 인구 통계학적 학습 불균형(Demographic Learning Bias)을 간과하고 있음을 지적합니다. MultiFair는 이러한 문제를 두 가지 층의 그래디언트 조절(Dual-level Gradient Modulation) 프로세스를 통해 해결합니다.

- **Technical Details**: MultiFair 모델은 훈련 그래디언트를 모달리티와 그룹 수준에서 최적화합니다. 이 모델은 각 모달리티의 기여도를 동적으로 조정하여, 전반적인 배치(Training Batch)에서 발생할 수 있는 불균형한 학습을 완화합니다. 논문에서는 MultiFair의 이론적 기반과 함께 실제 의료 다중 모달 데이터 셋을 활용한 광범위한 실험 결과도 제공합니다.

- **Performance Highlights**: 실험 결과, MultiFair는 기존의 최신 다중 모달 학습 및 공정성 학습(Fairness Learning) 방법들을 초월하며 성능을 보였습니다. 특히, 다양한 인구 통계 그룹에 대한 성능을 균형적으로 유지하면서도 진단의 정확도를 높이는 특징을 보여줍니다. 이는 의료 AI의 공정성을 확보하는데 기여할 것으로 기대됩니다.



### A Modality-Aware Cooperative Co-Evolutionary Framework for Multimodal Graph Neural Architecture Search (https://arxiv.org/abs/2510.07325)
Comments:
          11 pages, 6 figures. This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 소프트웨어 취약점에 대한 공동 활용 공격을 방지하기 위해 이질적이고 다중 모드의 취약점 데이터를 분석하는 방법을 제안합니다. 특히 유전자 알고리즘을 이용한 그래프 신경망 아키텍처 검색(GNAS)을 통해 다중 모드 그래프 신경망(MGNN)을 설계하는 새로운 접근 방식인 MACC-MGNAS를 소개합니다. 이 방법은 각 레이어에서 모드 특화 요소를 조정하는 수작업을 줄이면서도 더 나은 예측 정확도를 달성할 수 있도록 설계되었습니다.

- **Technical Details**: MACC-MGNAS는 다중 트랙 대체물(MADTS) 기법을 도입하여 평가 비용을 줄이고 유전자의 효율적인 진화를 가속화합니다. 이 기법은 각각의 모드에 맞춘 경량 대체물 두 개를 통해 채택된 피트니스 평가를 간소화합니다. 또한, 비슷도 기반 인구 다양성 지표(SPDI) 전략을 통해 탐색(exploration)과 착취(exploitation)의 균형을 적응적으로 조정하고, 성능을 극대화하여 수렴을 가속화할 수 있습니다.

- **Performance Highlights**: MACC-MGNAS는 표준 취약점 공동 활용(VulCE) 데이터셋에서 3 GPU 시간만에 81.67%의 F1 점수를 기록하며, 기존의 최첨단 경쟁 모델보다 8.7% 더 높은 성과를 달성했습니다. 특히, 이 과정에서 연산 비용을 27% 절감할 수 있어 효율성이 두드러집니다. 이는 기존의 아키텍처 설계보다 훨씬 더 나은 성능과 비용 효율성을 나타냅니다.



### Deep Learning Based Approach to Enhanced Recognition of Emotions and Behavioral Patterns of Autistic Children (https://arxiv.org/abs/2510.07320)
- **What's New**: 이번 연구에서는 자폐 스펙트럼 장애(ASD) 아동의 정서 인식과 행동 패턴을 중심으로 한 맞춤형 교육 전략의 필요성을 강조합니다. 특히, ASD 아동의 감정 상태를 정확히 인식하는 것이 맞춤형 개입 및 사회적 지원을 위한 기초가 됨을 설명하며, 최신 AI 기술인 autoencoder를 통해 감정 인식의 정확성을 향상시키는 방법을 제안합니다. 이를 통해 ASH 교육과 기술 지원을 위한 발달 경로를 더욱 원활하게 할 수 있도록 합니다.

- **Technical Details**: 연구 방법론에서는 Xception 및 InceptionV3 모델을 활용하여 ASD 아동의 정서 인식을 위한 데이터 전처리에 대한 접근 방식을 상세히 설명합니다. 특히, 각기 다른 크기의 이미지 입력을 299×299×3의 고정된 크기로 변환하기 위해 autoencoder를 사용하여 노이즈를 줄이고 필수적인 안면 특징을 보존합니다. 깊이별 분리 가능한 합성곱(depthwise separable convolutions) 기술을 사용하여 모델의 계산 복잡성을 줄이고, 다중 스케일의 특징 추출을 통해 다양한 정서 표현을 인식합니다.

- **Performance Highlights**: 연구 결과, autoencoder를 통합한 정서 인식 시스템이 전통적인 방법론에 비해 더 뛰어난 성능을 보여주었습니다. baseline 방법과 비교했을 때, 향상된 접근 방식은 정확도, 정밀도, 재현율, F1-score에서 유의미한 개선을 보였으며, 이를 통해 ASD 아동의 감정 인식 정확성을 높이고자 하는 목적을 달성했습니다. 또한, 두 단계의 훈련 전략을 통해 모델을 더욱 효과적으로 최적화함으로써, 복잡한 실제 환경에서도 우수한 성능을 발휘하는 것을 입증했습니다.



### Reconstructing the local density field with combined convolutional and point cloud architectur (https://arxiv.org/abs/2510.08573)
Comments:
          6 pages, 4 figures, 1 table. Accepted at the NeurIPS 2025 Workshop: ML4PS. Comments welcome!

- **What's New**: 이 논문에서는 우주론에서 관측된 어두운 물질의 밀도 필드를 회귀하는 신경망(neural network)을 구축하였습니다. 이 네트워크 아키텍처는 convolutional U-Net과 point-cloud DeepSets의 조합으로 구성되며, 이 조합을 통해 작은 규모의 정보를 효과적으로 활용하고 재구성 품질을 개선합니다. 특히, 하이브리드 네트워크는 작은 규모에서의 클러스터링 진폭과 위상을 U-Net 단독 방식보다 더 잘 회복합니다.

- **Technical Details**: 우리는 퀴호테 시뮬레이션의 고해상도 버전을 사용하여 훈련을 수행하였으며, 모든 시뮬레이션에 대해 동일한 우주론을 가정하였습니다. 검출된 halo를 바탕으로 pecular velocities을 생성하기 위해, momentum grid를 먼저 구성한 후 밀도 grid로 나누는 과정을 거칩니다. 이 과정은 halo의 희소성으로 인한 편향을 피하기 위해 주의해야 하며, Gaussian kernel로 둥글게 다듬는 방법을 사용하여 소규모 정보를 보존하고 있습니다.

- **Performance Highlights**: 이 네트워크는 비선형 스케일에서의 재구성 품질을 개선하는 것으로 확인되었습니다. 특히, k=0.1-1 hMpc-1 범위에서의 성능 향상은 논문의 주요 성과로, 이로 통해 어두운 물질 밀도 추정의 정확도를 높이는 데 기여합니다. 이러한 결과는 다른 기존 선형 방법들과 비교했을 때 두드러진 차이를 보여줍니다.



### BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation (https://arxiv.org/abs/2510.08572)
Comments:
          11 pages, 8 figures

- **What's New**: 이번 논문에서는 BLAZER라는 프레임워크를 제안하여 자동 생성된 훈련 데이터를 통해 조작 정책(manipulation policies)을 학습합니다. 기존의 로봇 기술이 대규모 인터넷 데이터에 의존하기 어려운 점을 극복하기 위해, 시뮬레이션을 통해 생성된 데모를 사용하여 로봇 학습을 개선하는 것이 핵심입니다. BLAZER는 대규모 언어 모델(LLM)의 제로샷(zero-shot) 능력을 활용하여 평가되고, 데이터 수집과 관리의 필요성을 줄입니다.

- **Technical Details**: BLAZER는 LLM 기반의 조작 에이전트를 자동 생성된 검증 데모를 통해 부트스트래핑(bootstrapping)하여 학습합니다. LLM이 생성하는 실행 가능한 조작 계획을 시뮬레이터에서 실행하고, 성공적으로 수행된 계획을 학습 세트로 사용하여 LLM의 조작 능력을 개선합니다. 이 과정에서는 객체의 위치, 방향 및 크기와 같은 특권 정보를 사용하여 훈련합니다.

- **Performance Highlights**: BLAZER는 시뮬레이션과 실제 환경에서 모두 제로샷 조작을 크게 향상시키고, 훈련 풀에 없는 작업에 대해서도 잘 일반화합니다. BLAZER로 훈련된 LLaMA-8B 모델은 훈련 과정에서 수동 데모가 전혀 필요하지 않으면서 LLaMA-70B보다 훨씬 높은 성공률을 보여주었습니다. 논문을 통해 다양한 조작 작업에서 발생한 일관된 성능 향상을 입증하며, 코드를 프로젝트 페이지에 공개하겠다고 발표하고 있습니다.



### ArenaBencher: Automatic Benchmark Evolution via Multi-Model Competitive Evaluation (https://arxiv.org/abs/2510.08569)
Comments:
          Preprint

- **What's New**: 이 논문은 ArenaBencher라는 기존 모델에 얽매이지 않는 벤치마크 진화 프레임워크를 제안합니다. ArenaBencher는 자동으로 테스트 사례를 업데이트하여 각 모델의 성능을 공정하게 비교할 수 있게 합니다. 이 시스템은 모델에 의해 메모라이즈된 내용이 아닌 진정한 일반화를 평가할 수 있도록 도와줍니다.

- **Technical Details**: ArenaBencher는 기존 벤치마크와 다양한 모델 풀을 기반으로 각 테스트 사례의 핵심 능력을 추론합니다. 예를 들어, 수학적 추론 벤치마크에서 multi-step arithmetic(다단계 산술)을 테스트하는 문제를 생성합니다. 검증 단계에서는 LLM을 판별자로 사용하여 질의-라벨 쌍의 정합성을 평가하고, 두 번째 단계에서는 여러 모델의 피드백을 집계하여 성능 격차를 드러내는 문제들을 선택합니다.

- **Performance Highlights**: ArenaBencher는 수학 문제 해결, 상식 추론 및 안전 분야에서 적용되어 새로운 실패 모드를 발견하고 테스트 목표의 일치를 유지하면서 난이도를 높이는 업데이트를 제공합니다. 이러한 업데이트는 모델 간의 성능 차이를 더 뚜렷하게 드러내고, 공정하며, 신뢰성 있는 비교 평가를 위한 발전된 테스트 사례를 생성합니다.



### How to Teach Large Multimodal Models New Skills (https://arxiv.org/abs/2510.08564)
Comments:
          In submission. Code is available at this https URL

- **What's New**: 이 논문은 대규모 다중모달 모델(large multimodal models, LMMs)에서 새로운 기술을 가르칠 때 이전 능력을 지키는 방법에 대해 연구합니다. 특히, 다섯 가지 목표 기술을 순차적으로 미세 조정(sequential fine-tuning)하면서, 세 모델 계열에 걸쳐 여덟 가지 기준 벤치마크에서 일반 능력을 모니터링합니다. 연구 결과, 좁은 범위의 미세 조정을 한 후 일부 기준 작업에서 '망각(forgetting)'이 관찰되지만, 후속 단계에서 이러한 영속적이지 않은 기억이 어느 정도 회복됨을 발견했습니다.

- **Technical Details**: 우리는 '망각'을 초래하는 결과 분포의 변화를 측정할 수 있는 간단한 카운팅 편향 탐지(counting-bias probe)를 통해 추적합니다. 이 연구에서는 두 가지의 간단하고 견고한 조정 레시피(tuning recipes)를 제안합니다: (i) 자기 주의(self-attention) 투영 레이어만 업데이트하고, (ii) Down 투영을 고정(freezing)한 상태에서 MLP Gate&Up만 업데이트하는 방법입니다.

- **Performance Highlights**: 이러한 조정 방법을 통해 모델과 작업 전반에 걸쳐 강력한 목표 성과(target gains)를 달성하면서도 이전 기준 성능(held-out performance)을 대체로 유지할 수 있음을 보여줍니다. 제안된 방법은 각 모델과 작업에서 향상된 결과를 제공하며, 연구 결과는 실제 코드로도 확인할 수 있습니다.



### Where Have All the Kaczmarz Iterates Gone? (https://arxiv.org/abs/2510.08563)
- **What's New**: 이 논문은 랜덤화된 Kaczmarz (RK) 알고리즘이 노이즈와 불일치를 포함한 시스템에서의 수렴 행동을 조사합니다. 기존의 연구에서는 일관성 있는 시스템에서의 RK의 수렴 성질은 잘 알려져 있으나, 불일치 및 노이즈가 포함된 시스템에 대한 연구는 제한적이었습니다. 이 연구를 통해 우리는 노이즈와 시스템 특성에 따른 Kaczmarz 반복의 한계 점의 위치를 파악합니다.

- **Technical Details**: Kaczmarz 알고리즘은 초기 추정치로부터 시작하여 반복적으로 정확한 해에 접근하는 점진적 방식을 사용하는 반복 알고리즘입니다. 이 알고리즘은 각 반복 시 coefficient matrix A의 한 행만 사용하여 계산 효율성을 높입니다. 논문은 노이즈가 있는 경우에 Kaczmarz 반복의 기대 수렴 행동을 분석하며, 제안된 수렴 경계는 노이즈 레벨과 시스템 특성에 따라 달라집니다.

- **Performance Highlights**: 모든 수치 실험은 이론적 발견을 검증하는 데 중점을 두며, RK 알고리즘이 현실적인 조건에서 어떻게 작동하는지를 보여줍니다. 다양한 노이즈 상황에서도 알고리즘의 성능을 평가하여, RK 알고리즘의 한계와 강인성을 밝혀냈습니다. 이 연구 결과는 과학 및 공학 문제에 RK 알고리즘을 최적화하여 적용할 수 있는 기반을 마련합니다.



### Agent Learning via Early Experienc (https://arxiv.org/abs/2510.08558)
Comments:
          Work in progress

- **What's New**: 이번 연구는 언어 에이전트들이 스스로 경험을 통해 학습하고 향상될 수 있는 가능성을 탐구합니다. 특히, 방식이 부족한 환경에서의 강화 학습의 한계를 극복하기 위해 'early experience'(얼리 익스피리언스)라는 새로운 개념을 도입했습니다. 이는 에이전트의 행동으로 생성된 상호작용 데이터로, 보상 신호 없이도 미래 상태를 감독할 수 있게 합니다.

- **Technical Details**: 연구에서는 두 가지 전략을 통해 초기 경험 데이터를 활용합니다. 첫 번째는 'Implicit world modeling'(암묵적 세계 모델링)으로, 수집된 상태를 사용하여 정책을 환경 동역학에 맞게 조정합니다. 두 번째는 'Self-reflection'(자기 반성)으로, 에이전트가 비효율적인 행동에서 학습하여 추론과 의사결정을 개선하는 방법입니다.

- **Performance Highlights**: 8개의 다양한 환경과 여러 모델 계열을 통해 평가한 결과, 제안된 접근 방식이 효과성과 도메인 외 일반화를 일관되게 개선함을 보여주었습니다. 더욱이, 검증 가능한 보상이 있는 환경에서는 초기 경험이 후속 강화 학습을 위한 강력한 기초가 될 수 있다는 유망한 신호를 포착했습니다.



### SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inferenc (https://arxiv.org/abs/2510.08544)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 추론(inference) 효율성을 높이기 위한 SPAD(Specialized Prefill and Decode hardware)라는 새로운 접근 방식을 제안합니다. LLM의 추론 과정이 성능을 극대화하기 위해 서로 다른 하드웨어에서 두 가지 단계로 분리되는 전통적인 방법(구현 방법)과는 달리, SPAD는 각 단계의 특성에 맞는 전문화된 칩을 설계했습니다. 이러한 접근은 기존 하드웨어의 비효율성을 해결하려는 시도를 포함합니다.

- **Technical Details**: 논문에서 제안한 SPAD는 'less-is-more' 철학에 기초하여 프리필(prefill) 단계와 디코드(decode) 단계에 최적화된 칩을 만들어냈습니다. 프리필 칩은 더 큰 systolic array와 비용 효율적인 GDDR 메모리를 사용하며, 디코드 칩은 높은 메모리 대역폭을 유지하면서 컴퓨트 용량을 줄였습니다. 이러한 특징들은 각 단계의 요구 사항에 맞춰 하드웨어를 최적화하여 성능을 극대화합니다.

- **Performance Highlights**: 모델링된 H100과 비교하여, 프리필 칩은 평균 8% 더 높은 프리필 성능을 제공하면서도 하드웨어 비용을 52% 줄였습니다. 디코드 칩은 28% 낮은 TDP로 97%의 디코드 성능을 달성하였습니다. 실제 생산 추적에 대한 end-to-end 시뮬레이션 결과 SPAD는 기존 클러스터에 비해 하드웨어 비용을 19%-41%, TDP는 2%-17% 줄이면서 동일한 성능을 유지할 수 있음을 보여주었습니다.



### Computational and statistical lower bounds for low-rank estimation under general inhomogeneous nois (https://arxiv.org/abs/2510.08541)
Comments:
          52 pages, 3 figures

- **What's New**: 이 논문에서는 잘 알려진 스파이크 위그너 행렬 모델을 확장하여, 추가적인 i.i.d. 가우시안 노이즈에 의해 손상된 저차 신호 행렬을 비균질 상황으로 일반화했습니다. 특히, 분산 프로필이 블록 구조를 가질 때 신호 검출 및 추정에 효과적인 스펙트럼 알고리즘을 제시하며, 그 성공을 위한 임계 신호 강도를 식별하고 정보 이론적 하한도 증명하였습니다. 이러한 결과에 새로운 측면으로 계산 최적성을 조사하고, 스펙트럼 알고리즘이 저차 다항식 알고리즘으로 신호를 검출할 수 없는 넓은 범위에서도 효과적임을 입증하였습니다.

- **Technical Details**: 스파이크 행렬 모델은 고차원 통계학의 기본 예시로서, 자연 추정기와 통계적-계산적 격차에서의 중요한 현상을 보여줍니다. 연구는 Y∼ℙN으로부터 나오는 데이터(Y)에서 신호(x)를 추정하는 작업을 다루고 있으며, 가설 검정(hypothesis testing)과 신호 추정(sigmal estimation) 두 가지 기초적 과제를 포함합니다. 특히, 강력한 검출과 분리를 정의하기 위해 두 가지 함수 시퀀스(tN 및 fN)에 대한 조건을 제시하고 있으며, 이들이 어떻게 서로 연관되는지를 설명합니다.

- **Performance Highlights**: 연구 결과에 따라, 스펙트럼 알고리즘은 특정 신호 분포일 때 최적 효과를 발휘하며, 높은 확률로 성공적인 가설 검정을 수행한다는 특성을 가지고 있습니다. 새로운 정리는 분산 프로필이 블록 구조를 가질 필요 없이, 보다 일반적인 프로필에 대해 스펙트럼 알고리즘이 최적일 수 있음을 시사합니다. 이러한 주장에 대한 수치적 연구도 포함되어 있으며, 고유값 추정 및 성능 최적성을 평가하는 데 도움을 줍니다.



### Permutation-Invariant Spectral Learning via Dyson Diffusion (https://arxiv.org/abs/2510.08535)
- **What's New**: 본 연구에서는 오르슈타인-울렌벡 과정(Ornstein-Uhlenbeck process)을 사용하여 그래프의 스펙트럼(dyson diffusion model) 동태를 포착하는 혁신적인 Dyson Diffusion Model(DyDM)을 소개합니다. 이 모델은 그래프의 비스펙트럼 정보를 유지하면서 스펙트럼을 정확히 학습할 수 있도록 합니다. 특히, DyDM은 기존의 그래프 확산 모델들이 직면한 한계를 극복하며, 그래프 이소모피즘(Graph Isomorphism)에 대한 접근을 개선합니다.

- **Technical Details**: DyDM은 랜덤 행렬 이론(random matrix theory)을 활용하여 그래프의 분산 관측치도(dynamics)를 분석합니다. 본 논문에서는 Dyson의 브라운 운동(Dyson's Brownian Motion)을 통해 확산의 스펙트럼 특성을 추출함으로써 구조적 특성을 포착하고, 이를 딥러닝 네트워크로 매개변수화하여 학습할 수 있게 합니다. 이러한 접근은 기존의 그래프 신경망(GNN) 및 그래프 변환기(graph transformers)의 한계를 넘어섭니다.

- **Performance Highlights**: DyDM은 기존의 GNN 및 그래프 변환기 기반의 방법들과 비교하여 그래프 스펙트럼 학습 성능에서 뛰어난 결과를 보였습니다. 전통적인 접근 방식들이 시도하지 못했던 그래프의 복잡한 구조적 차이를 파악할 수 있으며, 이로 인해 다양한 응용 분야에서 더 나은 성능을 기대할 수 있습니다. 이 모델은 또한 스펙트럼 정보의 중요성이 큰 대칭 행렬(symmetric matrices) 분야로의 확장성을 지니고 있습니다.



### CaRT: Teaching LLM Agents to Know When They Know Enough (https://arxiv.org/abs/2510.08517)
- **What's New**: 이 논문은 LLMs(대규모 언어 모델)가 정보 검색을 언제 중단해야 할지 학습할 수 있도록 돕는 새로운 접근법인 CaRT(Counterfactuals and Reasoning for Termination)를 소개합니다. 정보 검색 단계에서 효과적인 의사결정을 위해 CaRT는 반사실적 데이터를 이용하여 잘못된 결정과 올바른 결정을 비교하여 모델을 학습시킵니다. 이 방법은 상호작용 의료 진단 및 수학 문제 해결 분야에서 구현되었습니다.

- **Technical Details**: CaRT는 반사실적 쌍(counterfactual pairs)을 통해 LLM을 미세 조정(fine-tuning)하여 정보 수집을 중단할 적절한 시점을 학습하도록 합니다. 모델은 음성 추론(verbal reasoning)을 통해 종료 결정을 설명하도록 훈련되며, 이는 최종 LLM 모델에 내재화됩니다. 특히 Qwen3-1.7B-Instruct와 Qwen2.5-3B-Instruct 모델에서 이 방법의 효과성을 입증했습니다.

- **Performance Highlights**: 실험 결과, CaRT는 정보 검색의 효율성과 작업 성공률을 기존의 미세 조정 방법보다 향상시키는 것으로 나타났습니다. 의료 진단 및 수학 문제 해결 모두에서 CaRT 사용 시 성공률이 개선되었습니다. 특히, Qwen3-1.7B-Instruct에서 반사실적 요소를 제거하면 성과가 저하되어 CaRT의 중요성이 강조됩니다.



### AutoMLGen: Navigating Fine-Grained Optimization for Coding Agents (https://arxiv.org/abs/2510.08511)
- **What's New**: 이번 논문에서는 AutoMLGen이라는 LLM 기반의 코딩 에이전트를 소개합니다. 이 에이전트는 도메인 지식 데이터를 통합하여 높은 품질의 사전 가이드를 제공하고, 몬테 카를로 그래프 탐색(Monte Carlo Graph Search, MCGS) 알고리즘을 통해 효율적인 탐색을 가능하게 합니다. MCGS는 트리 기반 탐색의 장점을 유지하면서 그래프 구조를 결합하여 동적으로 경로를 재조정하고, 과거의 경로를 재사용하며, 다중 솔루션을 융합할 수 있도록 합니다.

- **Technical Details**: AutoMLGen은 코딩 에이전트로, 머신 러닝 엔지니어링(Machine Learning Engineering, MLE) 작업을 위한 지식 기반과 MCGS를 통합합니다. 이를 통해 다양한 모델 및 데이터 차원에서 도메인 사전 지식을 제공하며, 탐색 과정에서 세밀한 개선을 지원합니다. MCGS는 기존의 MCTS(Monte Carlo Tree Search)의 변형으로, 확장 단계에서 그래프 구조를 포함하여 고유한 해결책의 재조합과 통합을 가능하게 합니다.

- **Performance Highlights**: MLE-Bench에서의 평가 결과, AutoMLGen은 12시간의 예산 내에서 평균 36.4%의 메달 비율을 달성하며 기존의 모든 기준선을 능가하는 성능을 보였습니다. 이 에이전트는 안정적인 탐색 및 실행 가능성을 높이기 위해 세분화된 운영자 집합을 설계하여 최적의 ML 파이프라인 생성을 자동화합니다. AutoMLGen의 도입으로 MLE 작업에서의 성능이 획기적으로 향상되었습니다.



### AI-Driven Radiology Report Generation for Traumatic Brain Injuries (https://arxiv.org/abs/2510.08498)
- **What's New**: 이번 논문에서는 외상성 뇌손상 진단을 위한 자동 방사선 보고서 생성을 위한 혁신적인 AI 기반 접근법을 제안합니다. AC-BiFPN (Augmented Convolutional Bi-directional Feature Pyramid Network)과 Transformer 아키텍처를 통합하여 CT 및 MRI 스캔과 같은 복잡한 의료 영상 데이터를 처리합니다. 이 모델은 의료 영상에서의 다중 스케일 기능을 활용하여 intracranial hemorrhages와 같은 복잡한 이상을 효과적으로 탐지합니다.

- **Technical Details**: 제안된 모델은 AC-BiFPN을 통해 CT 및 MRI 이미지에서 다중 스케일 특징을 추출하고, Transformer를 통해 긴 의존성을 모델링하여 일관성 있는 진단 보고서를 생성합니다. AC-BiFPN은 다양한 해상도에서의 특징을 융합하여 상세한 뇌 스캔 분석에 적용되어 중요한 정보를 놓치지 않도록 합니다. 또한, 신뢰할 수 있는 진단 보고서를 생성하기 위해 Transformer 기반 모델을 사용하여 시각적 정보와 텍스트 정보를 통합합니다.

- **Performance Highlights**: 모델은 RSNA Intracranial Hemorrhage Detection 데이터셋에서 전통적인 CNN 기반 모델들보다 우수한 진단 정확도와 보고서 생성을 보여줍니다. 이 접근법의 혁신성은 고압 환경에서도 방사선 전문의를 지원하고, 훈련생 의사에게 실시간 피드백을 제공하여 학습 경험을 강화하는 데 기여합니다. 우리의 연구는 고급 기능 추출과 Transformer 기반의 텍스트 생성을 결합하여 외상성 뇌손상의 진단 과정에서 임상적 의사결정을 개선할 가능성을 보여줍니다.



### Implementing Semantic Join Operators Efficiently (https://arxiv.org/abs/2510.08489)
- **What's New**: 이 논문은 기존의 중첩 루프(nested loops) 방법을 대체하는 새로운 알고리즘을 제안합니다. 이 알고리즘은 전통적인 데이터베이스 시스템의 블록 중첩 루프(join) 연산자(block nested loops join operator)에서 영감을 받았습니다. 각 입력 테이블에서 행을 배치(batch)로 통합하여 LLM 호출을 통해 현재 입력에서 모든 일치하는 행 쌍을 식별하는 것이 목표입니다.

- **Technical Details**: 제안된 알고리즘은 LLM의 컨텍스트 윈도우(context window) 크기에 대한 제약을 고려하여 행 배치의 크기를 최적화하는 수식을 포함하고 있습니다. 또한 출력 크기를 추정하기 어려운 경우를 위한 적응형(adaptive) 변형도 제안되었습니다. 이는 LLM을 호출할 때 행 쌍을 평가할 때 필요한 처리 비용을 줄이는 데 기여합니다.

- **Performance Highlights**: 형식적인 분석과 함께 실제 결과는 제안된 알고리즘이 처리 비용을 단 significativamente 줄이며, 최근의 세맨틱 쿼리 처리 엔진에서 사용되는 조인 구현에 비해 우수한 성능을 보여줌을 입증합니다. 따라서 이 새로운 알고리즘은 세맨틱 쿼리 처리의 효율성을 크게 향상하는 가능성을 보여줍니다.



### DexMan: Learning Bimanual Dexterous Manipulation from Human and Generated Videos (https://arxiv.org/abs/2510.08475)
Comments:
          Video results are available at: this https URL

- **What's New**: DexMan은 인간의 시각적 데모를 바탕으로 휴머노이드 로봇의 양손 조작 기술로 자동 변환하는 프레임워크입니다. 이 시스템은 인간의 조작을 담은 제3자 비디오를 사용하여 카메라 보정이나 깊이 센서 없이 작동하며, 회전하는 손의 단순화된 모델을 넘어서는 실제 제어를 가능하게 합니다. 또한 DexMan은 실제와 합성된 비디오에서 기술을 생성할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: DexMan은 VGGT라는 이미지-3D 기초 모델을 활용하여 비디오 프레임의 깊이를 예측합니다. 긴 비디오를 겹치는 청크로 나누어 처리하며, 이로 인해 발현되는 척도 불일치 문제를 해결하기 위한 객체 중심의 시간 정렬 전략을 제안합니다. 이는 예측된 깊이 값의 일관성을 보장하고, 자원 소모를 줄이며, 다양한 훈련용 데이터셋을 생성할 수 있게 합니다.

- **Performance Highlights**: DexMan은 TACO 벤치마크에서 객체 자세 추정에 있어 최상의 성능을 기록하며, ADD-S와 VSD에서 각각 0.08과 0.12의 향상을 달성했습니다. 또한 강화 학습 정책은 이전 방법보다 성공률이 19% 향상되었습니다. 이 강력한 성능은 DexMan이 인간 조작의 복잡성을 효과적으로 학습하고 재현할 수 있음을 보여줍니다.



### Looking to Learn: Token-wise Dynamic Gating for Low-Resource Vision-Language Modelling (https://arxiv.org/abs/2510.08470)
Comments:
          Accepted to the EMNLP 2025 BabyLM Workshop

- **What's New**: 이 연구에서는 BabyLM Challenge 2025 비전 트랙의 요구 사항에 맞추어 경량화된 디코더 기반 아키텍처를 제안합니다. 이 아키텍처는 언어적 정보와 시각적 정보를 적응적으로 융합하기 위한 동적 게이팅(token-wise dynamic gating) 메커니즘, 제한된 시각적 정보의 효용을 극대화하기 위한 특징 조정 및 채널 주의(feature modulation and channel attention), 그리고 시각적 기초를 확보하기 위한 보조 대조 목표(auxiliary contrastive objectives)를 포함하고 있습니다. 이를 통해 수치적으로 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: 논문에서는 동적 게이팅 메커니즘을 채택하여 각 토큰에 대한 시각적 신호와 언어적 신호의 가중치를 맥락에 따라 선택적으로 조정합니다. 또한, 특징 향상 기술을 활용하여 제한된 시각적 정보의 유용성을 극대화하고, 문장 및 단어 수준에서 작동하는 대조 학습 보조 목표의 영향을 탐구합니다. 평가 데이터셋에서는 정보 병목 문제와 데이터셋 분할로 인해 훈련 불안정성이 발생할 수 있음을 지적합니다.

- **Performance Highlights**: 평가 결과, 제안된 프레임워크는 총 다섯 개의 벤치마크에서 경쟁력 있는 성능을 보였습니다. 특히, 동적 게이트가 언어의 품사에 따라 시각적 신호와 언어적 신호의 가중치를 조정하며, 내용 단어에는 시각적 신호를, 기능 단어에는 언어적 신호를 더 우선시한다는 것을 발견했습니다. 이러한 결과는 인지 기반 학습의 영감을 얻어 비전-언어 모델의 발전에 기여할 수 있는 가능성을 제시합니다.



### Accelerated Aggregated D-Optimal Designs for Estimating Main Effects in Black-Box Models (https://arxiv.org/abs/2510.08465)
- **What's New**: 최근 감독 학습(supervised learning)의 발전으로 블랙 박스 모델의 설명 가능성을 높이기 위한 연구가 증가하고 있습니다. 이 논문에서는 A2D2E라는 새로운 방법을 제안하여 입력 변수가 모델 예측에 미치는 효과를 더 효율적으로 추정할 수 있는 방안을 제시합니다. 기존의 방법들이 겪는 확장성 문제와 상관된 변수에 대한 민감성을 해결하는 데 중점을 둡니다.

- **Technical Details**: A2D2E는 가속화된 집합 D-최적 설계(Accelerated Aggregated D-Optimal Designs)에 기반하며, 주 효과(main effect) 추정을 효과적으로 하기 위해 계량적 실험 설계(principled experimental design)의 개념을 활용합니다. 이 방법은 이론적인 보장(guarantees)을 제공하며, 수렴성과 분산 감소를 증명한 후, 다양한 시뮬레이션을 통해 검증됩니다. 또한, 다양한 실제 데이터와 언어 모델에 응용할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: A2D2E는 기존의 방법들보다 더 높은 안정성을 제공하며, 높은 차원 환경에서도 효과적으로 작동합니다. 논문에서 이루어진 광범위한 수치 평가에 따르면, A2D2E는 전통적인 PD 플롯(Partial Dependence Plot)과 ALE 플롯(Accumulated Local Effects Plot)과 비교하여 예측 손실(prediction loss) 기준에서 우수한 성능을 보였습니다. 이를 통해 실제 데이터와 현대 예측 모델에 대한 다양한 적용을 입증하며, 향후 연구 방향에 대한 제안도 포함하고 있습니다.



### Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered (https://arxiv.org/abs/2510.08464)
- **What's New**: 이번 논문에서는 Vision-Language-Action (VLA) 모델의 효율성을 높이기 위한 새로운 방법, GLUESTICK을 소개합니다. 기존의 모델 프루닝(pruning)이 로봇에 적용할 때 유용할 수 있지만, 실제로 안전성 위반이 증가하고 모델 기능이 급격히 저하된다는 점을 발견했습니다. GLUESTICK은 이러한 문제를 해결하기 위해 원래 모델의 기능을 복원하면서도 희소성(sparsity)의 이점을 유지합니다.

- **Technical Details**: GLUESTICK은 밀집(dense) 및 프루닝된 모델 간의 가중치 공간(weight-space)에서 일회성(interpolation) 보정을 통해 원래 기능을 복구합니다. 이 수정(corrective term)은 추론(inference) 시 각 프룬된 레이어에서 사용되어 손실된 기능을 최소한의 오버헤드(overhead)로 회복합니다. 특별한 훈련 없이 작동하며, 프루닝 알고리즘에 독립적이며 효율성과 정확성 사이의 균형을 조절하는 단일 하이퍼파라미터(hyperparameter)를 도입합니다.

- **Performance Highlights**: GLUESTICK은 다양한 VLA 아키텍처와 작업에서 경쟁력 있는 메모리 효율성을 달성하며 성공률을 크게 회복하고 안전성 위반을 줄입니다. 이 방법은 조작(manipulation) 및 내비게이션(navigation) 성능에서 두드러진 결과를 보여줍니다. GLUESTICK의 도입으로 로봇 기술의 실제 적용 가능성이 한층 더 향상될 것으로 기대됩니다.



### Wavefunction Flows: Efficient Quantum Simulation of Continuous Flow Models (https://arxiv.org/abs/2510.08462)
- **What's New**: 이번 논문에서는 flow 모델이 슈뢰딩거 방정식과 밀접한 연관성이 있음을 보여줍니다. 특히, 통계적으로 정확한 qsample을 생성하기 위한 새로운 양자 알고리즘을 제안합니다. 이 알고리즘은 기존의 클래식 학습 문제와 해밀토니안 시뮬레이션을 결합하여 작동합니다. 이로 인해, flow 모델을 통해 표현 가능한 확률 분포의 광범위한 계통에 대한 coherent encodings를 효율적으로 준비할 수 있게 됩니다.

- **Technical Details**: 이 논문은 연속적인 변수를 포함하는 비정상적인 해밀토니안을 통해 flow 모델이 슈뢰딩거 방정식에 자연스럽게 맵핑된다고 주장합니다. continuity Hamiltonian이라는 이 해밀토니안은 이미 학습된 클래식 모델에 의해 전혀 새로운 학습 과정 없이 완전하게 정의됩니다. 주어진 초기 상태는 flow 모델이 학습한 확률 분포에 대한 qsample로 변환되며, 이는 양자 컴퓨터에서 효율적인 시뮬레이션을 가능하게 합니다.

- **Performance Highlights**: 이 양자 알고리즘은 여러 가지 통계적 문제에 적용될 수 있으며, 특히 flow 모델로 결정된 통계 문제에서 기존의 클래식 알고리즘에 비해 유의미한 샘플 복잡성의 이점을 제공할 수 있습니다. 이 연구는 양자 복잡성 이론과 클래식 머신 러닝之间의 새로운 연결 고리를 형성하는 중요한 기초를 제공합니다. 더 나아가, qsample을 효율적으로 준비 가능한 분포의 범위를 확인하는 과정에서 이론 머신러닝 문제로 전환될 수 있습니다.



### Navigating Sparsities in High-Dimensional Linear Contextual Bandits (https://arxiv.org/abs/2510.08435)
- **What's New**: 이번 논문에서는 고차원 선형 컨텍스츄얼 밴딧 문제를 다룬다. 특히, 모델 매개변수나 맥락 공분산 행렬의 고유값을 스파스(sparse)라고 가정하는 기존 방법의 한계를 극복하기 위해, 포인트와이즈 추정기(PointWise Estimator, PWE)를 기반으로 한 새로운 알고리즘 HOPE를 제안한다. HOPE는 단일 스파시티만을 고려한 기존 연구들과 달리, 두 가지 스파시티를 동시에 다룰 수 있는 혁신적인 접근법을 보여준다.

- **Technical Details**: HOPE 알고리즘은 탐색 후 커밋(explore-then-commit, ETC) 방식으로 PWE를 주요 추정기로 사용하여 고차원 환경에서 베스트 암(arm)을 선택하는 데 중점을 둔다. 이 논문은 HOPE의 이론적 분석을 통해, 이 알고리즘이 하나의 스파시티 유형을 다룰 때 기존의 이론적 보장을 충족하는 동시에 두 가지 스파시티가 모두 존재하는 복잡한 상황에서도 효과적으로 작동함을 입증한다. 또한, 이론적인 보증을 위해 네 가지 시나리오를 탐구하며, 각 시나리오에서의 성능을 검증한다.

- **Performance Highlights**: 다양한 시나리오에서 HOPE의 성능을 실험적으로 검증한 결과, 기존 방법들보다 우수한 성과를 나타냈다. 실험 결과는 HOPE의 유연성과 혁신성을 강조하며, 두 가지 타입의 스파시티를 동시에 효과적으로 다룰 수 있는 알고리즘으로서의 가능성을 증명했다. 총체적으로 HOPE는 이론적으로도 실용적으로도 탁월한 기여를 하는 알고리즘으로 평가된다.



### Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency (https://arxiv.org/abs/2510.08431)
- **What's New**: 이 연구는 일반 응용 수준의 이미지 및 비디오 확산 모델에 대한 지속적인 시간 일관성 증류(continuous-time consistency distillation)를 대규모로 확장하려는 첫 번째 시도입니다. 지속적인 시간 일관성 모델(sCM)은 이론적으로 원리적이며, 학술 규모의 확산을 가속화하는 데 강력하지만, 실제 대규모 텍스트-이미지(text-to-image) 및 텍스트-비디오(text-to-video) 작업에 대한 적용 가능성은 불확실합니다. 본 연구는 새로운 FlashAttention-2 JVP 커널을 개발하여 10억 개 이상의 매개변수를 가진 모델에서의 sCM 훈련을 가능하게 합니다.

- **Technical Details**: 연구진은 점진적 정제(progressive distillation) 및 스코어 정제(score distillation) 기법 대신, 스코어 정제된 지속적인 시간 일관성 모델(rCM)을 제안합니다. 이는 장기 스킵 규제자(long-skip regularizer)로서 스코어 증류(score distillation)를 통합하여 생성의 질을 향상시키고 고차원의 비디오 작업에서도 효과적으로 작동합니다. rCM은 GAN 조정이나 광범위한 하이퍼파라미터 검색 없이도 14억 개 매개변수를 가진 거대 모델에서 최적화된 성능을 발휘합니다.

- **Performance Highlights**: rCM은 DMD2와 같은 최신 정제 방법과 비교하여 질적 지표에서 동등하거나 뛰어난 성능을 보이며, 생성의 다양성 측면에서도 상당한 이점을 제공합니다. 효율적으로 고충실도 샘플을 1~4스텝 내에 생성할 수 있어 확산 샘플링 속도를 15배에서 50배까지 가속화합니다. 이러한 결과는 rCM이 대규모 확산 증류를 발전시킬 수 있는 실용적이고 이론적으로 당위성이 있는 프레임워크로 자리 잡게 합니다.



### Optimal Stopping in Latent Diffusion Models (https://arxiv.org/abs/2510.08409)
- **What's New**: 이 논문에서는 Latent Diffusion Models (LDMs)의 마지막 확산 단계가 샘플 품질을 저하시킬 수 있는 흥미로운 현상을 분석합니다. 전통적인 주장에 따르면, 수치적 안정성을 보장하기 위해 조기에 중단하는 것이 필요하지만, 이 현상은 LDM의 차원 축소와 관련이 있습니다. 저자들은 잠재적 차원과 중단 시간 간의 상호작용을 분석하여 조기 중단이 필요한 조건을 설명합니다.

- **Technical Details**: LDM은 사전 훈련된 오토인코더를 사용하여 데이터를 저차원 잠재 공간으로 압축한 후, 이 공간 내에서 확산 단계를 진행합니다. 이에 따라 훈련 시간과 계산 요구 사항이 줄어들면서도 품질 손실이 최소화됩니다. 저자들은 가우시안 데이터와 선형 오토인코더를 통해 조기 중단의 효과를 이론적으로 증명하며, 잠재 차원과 스코어 매칭의 하이퍼파라미터 간의 상호작용을 조사합니다.

- **Performance Highlights**: 실험 결과, 조기 중단을 통해 생성 품질이 향상된다는 점이 강조됩니다. LDM을 통해 전통적인 확산 모델와 훈련된 CelebA 데이터셋의 샘플을 비교한 결과, FID 점수가 상승하면서 품질이 저하되는 현상을 확인했습니다. 이러한 특성은 향후 LDMs의 이론적 기초를 제공하며, 최적의 중단 시간이 하이퍼파라미터로 중요하다는 점을 부각시킵니다.



### Single layer tiny Co$^4$ outpaces GPT-2 and GPT-BER (https://arxiv.org/abs/2510.08404)
- **What's New**: 이번 논문에서는 Co$^4$ 언어 모델을 소개하며, 단일 레이어와 두 개의 헤드를 가진 8M 파라미터의 소형 기계가 BabyLM Challenge의 신기준인 GPT-2와 GPT-BERT를 능가하는 성능을 보여줍니다. Co$^4$는 단 2 에포크 동안 훈련했지만, 이전 모델들은 10 에포크 필요했습니다. 이 모델은 10M 토큰 на 대하여 매우 높은 샘플 효율성을 보여줍니다.

- **Technical Details**: Co$^4$ 모델은 두 가지 다른 입력 통합 지점을 가진 뉴런을 기반으로 하며, 이를 통해 문맥적 입력(C)와 피드포워드 입력(FF)을 잘 결합할 수 있습니다. Triadic modulation loops를 활용하여 Q, K, V의 공동 진화를 통해, Co$^4$는 저비용 운영(O(N))으로 효율적인 학습과 뛰어난 상태 추론을 가능하게 합니다. 또한, 이 모델은 10M 토큰에서 최소한의 학습 예산으로 훈련되었습니다.

- **Performance Highlights**: Co$^4$는 여러 언어 모델링 벤치마크에서 뛰어난 성능을 발휘하였습니다. 제로-샷(zero-shot) 환경에서 7개 작업 중 5개에서 GPT-2를 초과하며, 파인 튜닝 작업에서도 7개 중 6개에서 우수한 결과를 기록했습니다. 이러한 성과는 현재의 딥러닝 패러다임과 스케일링 법칙을 재고할 필요성이 있음을 제안합니다.



### On the Relationship Between the Choice of Representation and In-Context Learning (https://arxiv.org/abs/2510.08372)
Comments:
          25 pages, 6 figures, 10 tables

- **What's New**: 이번 논문은 in-context learning (ICL)에서 나타나는 두 가지 주요 요소, 즉 representation(표현)과 learning(학습) 간의 관계를 분석합니다. 기존 연구에서는 이 두 요소를 별개로 살펴보았으나, 본 연구에서는 이들이 독립적이라는 가설을 세우고 이를 검증하기 위한 최적화 알고리즘을 개발했습니다. 실험을 통해, demonstration(데모)의 종류와 모델의 크기에 따라 ICL의 성능이 어떻게 달라지는지를 조사하며, representation의 질이 ICL의 기본 정확도를 결정한다고 제시합니다.

- **Technical Details**: 연구에서는 label sets(레이블 세트)의 의미적 관련성을 다양하게 조절하여, ICL 성능을 측정할 수 있는 최적화 알고리즘을 개발했습니다. 세 개의 모델 크기를 설정하고, 각각의 레이블 세트에 대해 여러 개의 데모를 사용하는 실험을 실시하였으며, 이로 인해 representation과 learning 간의 관계를 정량적으로 분석하였습니다. 이 분석을 통해 학습 효율성과 정확도 간의 상관관계에 대해 다양한 결과를 도출했습니다.

- **Performance Highlights**: 실험 결과, ICL에서 표현이 학습을 유도하는 역할을 하며, 학습은 보통 representation의 질과 모델 크기와 관계없이 발생함을 확인했습니다. 정확도 순위는 demonstration의 수와 상관없이 초기 순서를 그대로 유지하며, 일정한 범위의 정확도가 주어진 representation에 의해 크게 결정됨을 관찰했습니다. 이 결과는 적절한 레이블 세트를 선택함으로써 ICL 성능을 향상시킬 수 있는 가능성을 보여줍니다.



### PAC Learnability in the Presence of Performativity (https://arxiv.org/abs/2510.08335)
Comments:
          21 pages, 3 figures

- **What's New**: 이 논문에서는 머신러닝 모델이 실제 환경에서 사용됨에 따라 발생하는 performativity(퍼포머티비티) 현상을 다룹니다. 이는 모델 의존적인 테스트 분포의 변화를 의미하며, 이로 인해 성능 저하가 나타날 수 있는 문제를 다루고 있습니다. 연구진은 특히 binary classification(이진 분류) 문제를 PAC(Probably Approximately Correct) 학습 프레임워크를 통해 조사하였습니다.

- **Technical Details**: 기존의 데이터 분포에서만 학습한 모델이 수행하는 분류의 위험성을 정량화하기 위해 performative empirical risk(퍼포머티브 경험적 위험) 개념을 도입했습니다. 이 연구는 메커니즘이 linear shifts(선형 변화)와 conditional shifts(조건적 변화)에 따라 성능이 어떻게 변하는지를 다룹니다. 또한, 이론적 접근을 통해 기존의 PAC-learnable(팩 학습 가능) 가설 공간이 이와 같은 퍼포머티브 시나리오에서도 PAC-learnable하다는 것을 증명했습니다.

- **Performance Highlights**: 연구진은 다양한 실험을 통해 제안한 방법의 효용성을 검증하였고, synthetic data(합성 데이터)와 Kaggle credit score 및 Folktables income prediction 데이터셋에서 성과를 보였습니다. 특히, linear posterior performative drift 문제에서 PERM(퍼포머티브 경험적 위험 최소화)이 효율적이라는 점이 강조되었습니다. 이를 통해 새로운 문제 해결 방식에 대한 연구의 필요성을 제시하고 있습니다.



### New Machine Learning Approaches for Intrusion Detection in ADS-B (https://arxiv.org/abs/2510.08333)
Comments:
          This is the author's version of the work accepted for publication Digital Avionics Systems Conference (DASC) 2025. The final version will be available via IEEE Xplore

- **What's New**: 이번 연구는 항공 교통 관리에서의 Automatic Dependent Surveillance-Broadcast (ADS-B) 프로토콜의 보안에 중점을 두고 있습니다. 저자들은 두 가지 딥러닝 기반 침입 탐지 시스템(IDS)을 제안하며, 첫 번째는 transformer encoder를, 두 번째는 확장된 Long Short-Term Memory (xLSTM) 네트워크를 사용합니다. 특히, ADS-B에 특화된 xLSTM 기반 IDS의 첫 번째 구현으로, 기존 방법들을 초월하는 성능을 입증했습니다.

- **Technical Details**: 연구에서는 기계 학습 기반의 침입 탐지 기법인 xLSTM를 활용하여 ADS-B 데이터의 이상 징후를 효과적으로辨識할 수 있는 새로운 접근 방식을 제안합니다. xLSTM은 메시지의 장기 의존성을 유지하면서 효율적인 메모리 아키텍처를 도입하여 시간 순서 데이터에서 흐름을 유지합니다. 연구는 또한 전이 학습 방식을 통해 일반적인 ADS-B 메시지에서 비정상 데이터를 효과적으로 식별할 수 있는정밀한 모델을 학습합니다.

- **Performance Highlights**: 실험 결과, xLSTM 기반 IDS는 98.9%의 F1 점수를 기록하여 transformer 기반 모델의 94.3%를 초월하는 성능을 보였습니다. 이 IDS는 감지 성능이 뛰어나고 새로운 공격에 대한 일반화 가능성을 입증했습니다. 반면, transformer 기반 IDS는 2.1초의 지연 시간을 보였지만, 이는 성능 저하와 관련이 있습니다.



### Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries (https://arxiv.org/abs/2510.08325)
Comments:
          10 pages, 3 figures

- **What's New**: 강화학습을 통한 검증 가능한 보상(Reinforcement Learning with Verifiable Rewards, RLVR)은 대규모 언어 모델의 이론적 문제 해결 능력을 향상시키는 데 중요한 패러다임으로 자리 잡고 있습니다. 최근 연구에서 RLVR 모델의 이론적 경계가 실제로 확장되는지에 대해 의문이 제기되었습니다. 본 논문에서는 Pass@k 지표가 문제의 신뢰도를 고려하지 않기 때문에, 보다 유용한 대안으로 Cover@tau 지표를 제안합니다.

- **Technical Details**: Cover@tau는 모델이 지식을 바탕으로 문제를 해결할 확률율을 τ 이상으로 설정하여 문제 해결 능력을 측정합니다. 이 메트릭은 Pass@k와 달리 무작위 추측에 의한 성능 저하가 발생하는 것을 방지합니다. 제안된 지표는 RLVR 모델을 평가하는 데 있어 서로 다른 신뢰성 수준을 적용해 보는 새로운 접근법을 제공합니다.

- **Performance Highlights**: 여러 RLVR 모델을 평가한 결과, Cover@tau 지표는 Pass@1 또는 Pass@k와 비교했을 때 서로 다른 알고리즘의 상대적인 순위를 제공합니다. 이는 모델의 능력에 대한 새로운 관점을 제시하며, Pass@k 지표가 편향된 성능 수치를 보여줄 수 있다는 점을 강조합니다. 이러한 평가를 통해 RLVR 방법론의 수학적 추론 능력이 보다 정확히 분석되었습니다.



### Iterated Agent for Symbolic Regression (https://arxiv.org/abs/2510.08317)
Comments:
          45 pages, 22 figures, 8 tables

- **What's New**: 본 연구는 데이터에서 수학적 표현을 자동으로 발견하는 기법인 기호 회귀(symbolic regression, SR) 분야에서의 새로운 프레임워크인 IdeaSearchFitter를 소개합니다. 이 프레임워크는 대형 언어 모델(LLM)을 의미론적 변형기로 활용하여, 자연어 논거에 따라 후보 표현을 생성함으로써 개념적으로 일관되고 해석 가능한 모델의 발굴을 선호합니다. IdeaSearchFitter는 다양한 도전 과제에서 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: IdeaSearchFitter는 LLM을 사용하여 해석 가능한 ansatz 공간 내에서 진화적 검색을 진행합니다. 이 시스템은 자연어로 표현된 논거를 토대로 후보를 생성하며, 개념적으로 의미 있는 변형을 보장합니다. 전통적인 구문 기반의 검색 방법에서 의미 기반의 검색 방법으로의 전환은 SR에서의 오버피팅(overfitting) 문제를 줄여주고 정확성, 복잡성, 해석 가능성 간의 균형을 최적화하는 데 큰 이점을 제공합니다.

- **Performance Highlights**: 실험 결과, IdeaSearchFitter는 Feynman Symbolic Regression Database (FSReD)에서 80% 이상의 회수율을 달성하고 고노이즈 조건에서도 71.7%의 성능을 보이며 여러 기초 방법론들을 초월합니다. 실제 데이터셋에 대한 응용에서도 메커니즘적으로 정렬된 모델을 발견하고 고에너지 물리학 응용에서 프로톤의 복잡한 내부 구조를 설명하는 간단한 방정식을 도출해내었습니다. 이 연구는 IdeaSearch의 일환으로 공개되어 있어 다양한 물리적 응용이 가능합니다.



### Contrastive Decoding for Synthetic Data Generation in Low-Resource Language Modeling (https://arxiv.org/abs/2510.08245)
Comments:
          13 pages, 3 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 교육에 있어 제한된 데이터 문제를 해결하기 위해 인공지능에 의해 생성된 합성 데이터(synthetic data)를 사용합니다. 특히, Contrastive Decoding(𝖢𝖣	ext{CD})을 활용하여 모델 성능을 향상시키는 것을 목표로 합니다. 원본 말뭉치에서 좋은 모델과 나쁜 모델 간의 상대적 차이를 이용하여 합성 말뭉치를 생성하고, 이를 실제 데이터에 혼합하여 훈련하는 방식입니다.

- **Technical Details**: 연구에서는 100M 단어의 원본 말뭉치에서 훈련된 좋은 모델과 나쁜 모델의 상대적 차이를 기반으로 100M 단어의 합성 말뭉치를 생성했고, 이를 원래 훈련 데이터와 결합하여 모델을 훈련했습니다. Contrastive Decoding(𝖢𝖣	ext{CD})은 더 우수한 모델에서 나오는 신호를 확대하여 더 일관되고 유익한 텍스트를 생성하는 데 초점을 맞춥니다. 이 방법이 다른 전통적인 샘플링 방식에 비해 훈련 과정에서 어떻게 기여하는지 분석했습니다.

- **Performance Highlights**: 합성 데이터로 훈련한 모델은 언어 모델링 목표 및 여러 다운스트림 작업에서 성능이 향상되었습니다. 특히, 𝖢𝖣	ext{CD}에서 생성된 데이터를 사용한 훈련이 더 많은 추론 능력이 요구되는 작업에 강력한 효과를 보이며, 전통적인 샘플링을 통해 생성된 데이터는 표면적인 언어적 능력에 의존하는 작업에서 더 유리한 결과를 가져왔습니다.



### Investigating Counterclaims in Causality Extraction from Tex (https://arxiv.org/abs/2510.08224)
- **What's New**: 본 연구는 텍스트 내 인과 관계 추출에서 반대 주장(concausal claims)을 간과한 점을 지적하고 새로운 데이터셋인 Concausal News Corpus를 개발하여 이 문제를 해결하고자 합니다. 기존 데이터셋은 오로지 인과 관계를 지지하는 주장만을 포함하고, 반대 주장은 무시되거나 오인 Annotation되었습니다. 새로운 데이터셋은 인과 관계의 복잡성을 개선하고, 모델이 procausal과 concausal 관계를 효과적으로 구별할 수 있도록 돕습니다.

- **Technical Details**: 인과 관계 추출(causality extraction) 과정은 크게 세 단계로 나눌 수 있으며, 이 연구에서는 이 과정을 procausal, concausal, uncausal 관계로 명확히 구분하는 작업을 수행하였습니다. 새로운 데이터 세트에서 코헨의 카파 통계량은 0.74에 달하는 높은 상호 주석자 동의율을 나타내어, 명확한 기준을 바탕으로 반대 주장을 효과적으로 포함시켰습니다. 또한 연구 결과는 transformer 기반의 신경망이 이러한 주장을 효과적으로 구별할 수 있음을 보여줍니다.

- **Performance Highlights**: 본 연구의 결과는 기존 인과 관계 추출 모델이 concausal 주장을 잘 인식하지 못하고, 이를 procausal로 잘못 분류하는 경향이 있음을 강조합니다. 데이터셋 개선을 통해 이러한 문제를 해결할 수 있으며, 이는 다양한 실제 응용, 특히 결정 지원 시스템에서 인과적 이해를 높이는 데 기여할 것입니다. 궁극적으로, 반대 주장을 포함하는 것은 사용자에게 균형 잡힌 시각을 제공하며 인과 관계에 대한 인식을 제고할 수 있게 합니다.



### Leveraging Whisper Embeddings for Audio-based Lyrics Matching (https://arxiv.org/abs/2510.08176)
- **What's New**: WEALY는 음성 인식 기반의 가사 매칭 시스템으로, Whisper 디코더 임베딩(embedding)을 활용하여 가사 매칭 작업을 위한 완전 재현 가능한 프로세스를 제공합니다. 이 시스템은 텍스트 데이터나 기존 트랜스크립션에 의존하지 않고도 강력하고 투명한 기준선을 설정합니다. WEALY는 다양한 데이터셋을 통해 폭넓은 실험을 수행하면서 기존의 비재현성 문제를 해결하고 있습니다.

- **Technical Details**: WEALY는 가사 내용의 의미적 및 구조적 유사성을 식별하는 데 중점을 두며, 이를 위해 음향 신호에서 직접 가사 표현을 추출하는 이점이 있습니다. 이 과정에서, Whisper의 디코더 임베딩을 활용하여 원시 오디오 데이터를 가사 표현으로 변환하는 맞춤형 파이프라인을 설계하였습니다. MVI(musical version identification)를 보조 작업으로 활용하여 다수의 데이터셋을 기준으로 포괄적인 벤치마크를 수립합니다.

- **Performance Highlights**: WEALY는 기존의 비재현성 문제를 가진 방법들에 비해 경쟁력 있는 성능을 보이며, 실험 결과는 최신 기술 수준에 준하는 성능을 입증하고 있습니다. 또한, 다양한 손실 함수 및 풀링 전략의 영향을 분석하는 광범위한 연구를 수행하였으며, 향후 다중 모달 확장성을 탐구하고 있어 다양한 기능을 지닌 시스템으로 발전할 잠재력이 큽니다.



### Quantum Agents for Algorithmic Discovery (https://arxiv.org/abs/2510.08159)
- **What's New**: 본 논문에서는 에피소드 기반의 강화 학습(reinforcement learning)을 통해 훈련된 양자 에이전트(quantum agents)를 소개합니다. 이 에이전트들은 이미 알려진 최적 솔루션에 접근하지 않고도 양자 푸리에 변환(Quantum Fourier Transform)의 효율적인 로그 깊이 양자 회로(logarithmic-depth quantum circuits) 및 그로버의 검색 알고리즘(Grover's search algorithm)과 같은 여러 주요 양자 알고리즘을 자율적으로 재발견합니다. 이러한 접근 방식은 알고리즘 발견(algorithmic discovery)을 위한 양자 지능(quantum intelligence)의 잠재력을 보여주며, 새로운 양자 알고리즘과 프로토콜의 자동화된 설계를 위한 길을 엽니다.

- **Technical Details**: 양자 컴퓨터(quantum computers)는 고전 컴퓨터(classical computers)와는 fundamentally 다른 원리를 활용하여 계산을 재정의합니다. 본 연구는 양자 정보와 다수의 에이전트를 포함하는 양자 생태계에서의 자율적이고 상호작용적인 행동을 강조하는 새로운 프레임워크를 제안합니다. 각 에이전트는 주변 환경을 인식하고 상호작용하는 동시에 자신의 행동을 최적화하여 보상을 최대화하는 방향으로 훈련됩니다.

- **Performance Highlights**: 제안된 프레임워크를 통해 에이전트들은 양자 알고리즘 및 프로토콜을 자율적으로 재발견하여 실제 하드웨어에서 이해 가능하고 일반화 가능한 결과를 도출합니다. 예를 들어, 양자 푸리에 변환(QFT), 그로버의 검색 알고리즘, 강력한 동전 던지기 전략 및 CHSH 게임과 같은 여러 양자 문제를 해결하며, 최적의 전략을 직접 상호작용을 통해 학습합니다. 이러한 결과는 양자 지능이 알고리즘 발견의 효과적인 도구로 기능할 수 있음을 보여 주어, 인간의 전문성을 보완하는 새로운 가능성을 제시합니다.



### AI Knowledge Assist: An Automated Approach for the Creation of Knowledge Bases for Conversational AI Agents (https://arxiv.org/abs/2510.08149)
Comments:
          Accepted to the EMNLP 2025 Industry Track

- **What's New**: 본 논문에서는 고객 문제 해결을 위한 회화형 AI 시스템에서 Retrieval Augmented Generation (RAG) 기술의 활용 증가에 대해 다루고 있습니다. 이를 해결하기 위해 과거 고객-에이전트 대화에서 QA 쌍을 자동으로 추출하여 지식 기반을 구축하는 AI Knowledge Assist 시스템을 소개합니다. 이 시스템은 LLaMA-3.1-8B 모델을 기반으로 하여 20개 회사의 실증 평가를 통해 90% 이상의 정확도로 정보 요청 질문에 답변할 수 있음을 보여줍니다.

- **Technical Details**: AI Knowledge Assist 시스템은 세 단계의 파이프라인을 통해 작동합니다. 첫 번째 단계에서는 과거의 통화 원고에서 정보 요청 질문과 에이전트의 응답을 추출합니다. 두 번째 단계에서는 이러한 QA 쌍을 의미상 유사한 그룹으로 클러스터링하여 중복된 QA 쌍을 관리합니다. 마지막으로, LLM을 활용하여 각 클러스터에서 정보를 가장 잘 요약한 대표 QA 쌍을 선택해 지식 기반에 삽입하거나 관리자에게 추천합니다.

- **Performance Highlights**: 이번 연구는 20개 클라이언트 회사의 실제 데이터에서 실험을 수행하였으며, AI Knowledge Assist 시스템은 콜센터 AI 챗봇의 기능을 대폭 향상시킴을 입증했습니다. 이 시스템을 통해 고객의 문의를 효과적으로 처리할 수 있으며, 고객 만족도를 증대시키는 데 기여할 수 있음을 나타냅니다. 실제 데이터 수집에 있어 고객 데이터의 개인 정보 보호를 우선시하여 미세한 주의가 필요함을 강조합니다.



### High-dimensional Analysis of Synthetic Data Selection (https://arxiv.org/abs/2510.08123)
- **What's New**: 이 논문에서는 생성 모델이 합성 데이터(synthetic data)의 생성에 있어 예측 성능을 개선하는 데 어떤 도움이 되는지에 대한 의문을 다루고 있습니다. 특히, 합성 데이터의 공분산 변화(covariance shift)가 일반화 오류(generalization error)에 미치는 영향을 수학적으로 증명하였으며, 평균 변화(mean shift)는 그와는 상관없다는 점이 밝혀졌습니다. 또한, 생성된 합성 데이터의 공분산을 맞추는 것이 최적의 결과를 보일 수 있음을 실험적으로 입증하였습니다.

- **Technical Details**: 이 연구에서는 두 개의 데이터셋, 즉 실제 훈련 데이터셋(Xt,yt)과 추가적인 합성 데이터셋(Xs,ys)이 필요하다고 가정합니다. 이때, 데이터셋의 샘플들은 서로 독립 identically distributed(i.i.d.)이며, 우리는 이를 통해 경험적 위험 최소화(empirical risk minimization, ERM)를 수행했습니다. 연구는 합성 데이터 선택을 최적화하기 위한 방법을 제시하며, 특정 조건에서 공분산을 맞춤으로써 성능이 최적화 됨을 나타냅니다.

- **Performance Highlights**: 공분산을 맞추는 접근법은 다양한 훈련 패러다임과 아키텍처에서 좋게 작동하며, 여러 가지 기존 방법들과 비교했을 때 동등하거나 우수한 성능을 보여줍니다. 이 연구는 CIFAR-10, ImageNet-100 및 여러 생성 모델을 사용한 실험을 통해 공분산 맞춤 방법의 효용성을 입증하였습니다. 이는 고차원 회귀 문제에서 공분산의 선택이 중요하다는 것을 강조하며, 이는 실제적인 데이터 세트 확장 시 직접적인 응용 가능성을 갖고 있습니다.



### Random Window Augmentations for Deep Learning Robustness in CT and Liver Tumor Segmentation (https://arxiv.org/abs/2510.08116)
Comments:
          10 pages, 9 figures. This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 CT(Computed Tomography) 영상의 데이터 증강 기법에서 중요한 문제를 다룹니다. 기존의 자연 이미지용 강도 변환(일반적 intensity augmentations) 기법을 CT 이미지에 부적절하게 적용할 경우 성능 저하와 아티팩트가 발생할 수 있음을 밝혔습니다. 이를 해결하기 위해 새로운 CT 전용 증강 기법인 '랜덤 윈도잉(Random windowing)'을 제안하며, 이는 CT 이미지에서 Hounsfield 단위(HU)의 분포를 활용합니다.

- **Technical Details**: 랜덤 윈도잉 기법은 기존 강도 증강 기법을 대체할 수 있는 새로운 방법으로, CT 영상의 특정 구역에 대한 견고성을 높입니다. 이 방법은 CT 이미지의 HU 분포를 존중하여 제출함으로써 임상 적용에서의 효과를 극대화합니다. 여러 데이터 세트에서 기법의 효과를 정량적으로 분석하고, 제조된 방법과 기존의 최첨단 홍수화 기반 방법들을 비교하여 뛰어난 성능을 확인하였습니다.

- **Performance Highlights**: 랜덤 윈도잉 기법은 특히 영상의 대비와 시간 안전성이 떨어지는 어려운 CT 이미지에서 모델 성능을 크게 향상시키는 결과를 보였습니다. 이 연구는 간 종양(segmentation of liver tumors) 이미지를 대상 문제로 설정하였고, 랩 데이터 부족 문제를 해결하는 데 큰 기여를 할 것으로 기대됩니다. 또한, 기존의 증강 기법들과 비교하여 월등한 결과를 보여주며, CT 영상 처리 분야에서의 변화를 이끌 것으로 예상됩니다.



### Lossless Vocabulary Reduction for Auto-Regressive Language Models (https://arxiv.org/abs/2510.08102)
- **What's New**: 이번 논문에서는 자동 회귀 (auto-regressive) 언어 모델의 어휘 (vocabulary)를 손실 없이 줄이는 이론적 프레임워크를 제시합니다. 이는 언어 모델들이 각기 다른 어휘를 가질 때 발생하는 비효율성을 해결하는 데 초점을 맞추고 있습니다. 새로운 접근 방식은 다양한 모델들이 상호 협력할 수 있도록 돕는 최대 공통 어휘 (maximal common vocabulary)를 활용합니다.

- **Technical Details**: 논문에서 제시된 프레임워크는 주어진 자동 회귀 언어 모델의 어휘를 임의의 크기로 축소할 수 있도록 합니다. 이는 각 모델의 다음 토큰 분포 (next-token distribution)에 미치는 영향을 최소화하며, 결과적으로 텍스트 생성의 효율성을 증가시킵니다. 이론적 기반 위에서, 손실 없는 (lossless) 어휘 축소 방법론이 제시되고, 이를 통해 다양한 토크나이제이션 (tokenization)을 가진 모델들이 협력할 수 있는 가능성을 탐구합니다.

- **Performance Highlights**: 이 연구는 서로 다른 언어 모델들이 하나의 공통된 어휘를 통해 효과적으로 협력할 수 있음을 실험적으로 입증했습니다. 향상된 협력 구조 덕분에 모델 앙상블 (ensemble) 성능이 크게 향상되었습니다. 따라서, 언어 모델의 협력적 활용 방안이 확장되어, 다양한 분야에서의 응용 가능성을 보여줍니다.



### Beyond Real Data: Synthetic Data through the Lens of Regularization (https://arxiv.org/abs/2510.08095)
- **What's New**: 이 논문에서는 실 데이터가 부족할 때 합성 데이터의 일반화 개선 가능성에 대한 새로운 학습 이론적 프레임워크를 제안합니다. 합성 데이터와 실 데이터 간의 비율을 최적화하여 테스트 오류를 최소화할 수 있는 방법을 제시하고, Wasserstein distance를 기준으로 이를 수량화합니다. 특히 커널 릿지 회귀 분석을 통해 이론을 입증하며, 실제 데이터와 합성 데이터를 적절히 융합하는 방법을 탐구합니다.

- **Technical Details**: 논문에서는 합성 데이터와 실 데이터 간의 최적 비율을 정량화하기 위해, 알고리즘 안정성에 기반한 일반화 오류 경계를 도출합니다. 구체적으로, 커널 릿지 회귀 모델을 사용하여 실과 합성 데이터의 비율에 따라 테스트 오류가 어떻게 변화하는지를 분석합니다. 이론적 결과는 CIFAR-10과 임상 뇌 MRI 데이터셋을 사용하여 실험적으로 검증됩니다.

- **Performance Highlights**: 제안된 프레임워크는 도메인 적응 상황에서도 효과를 보여, 제한된 실 데이터와 합성 target 데이터를 조화롭게 결합함으로써 일반화 성능을 향상시키는 방법을 제시합니다. 실험 결과에 따르면 합성 데이터의 적절한 사용이 저데이터 환경에서 모델 성능을 크게 향상시킬 수 있다는 것을 보여줍니다. 이러한 결과는 의료 분야와 같은 데이터가 부족한 다양한 머신러닝 응용에 적용될 수 있는 통찰력을 제공합니다.



### Computations and ML for surjective rational maps (https://arxiv.org/abs/2510.08093)
Comments:
          15 pages, 2 figures, a couple of Python codes

- **What's New**: 이 논문은 복소 프로젝트(cv) 다형체에서의 서젝티브(rational endomorphisms) 매핑에 대한 새로운 접근을 제시합니다. 기존의 선형 시스템을 통해 서젝티브 매핑의 일반성을 연구하고 있으며, 특별히 세 개 이상의 이상점(indeterminacy locus)을 가진 입방형 매핑에 대해 다룹니다. 또한, 저자들은 파이썬(Python)과 머신러닝(Machine Learning)을 활용하여 이러한 지도를 분류하는 실험적 접근 방법을 개발했습니다.

- **Technical Details**: 여기서 서젝티브(rational endomorphism)란 주어진 복소 프로젝트 공간에서 유도된 변환이 전사(onto)일 때 사용되는 용어입니다. 연구에서 저자들은 입방체(cubic)로서의 매핑이 일반적으로 서젝티브임을 보이고 있으며, 이는 특정한 비정상적(indeterminacy locus) 점의 수와 관련이 있습니다. 일반적인 비정상적 입방 매핑의 특성을 살펴보면서, 이 군은 Zariski 개방 집합(open set)을 형성한다는 것을 보여주었습니다.

- **Performance Highlights**: 제안된 방법론은 서젝티브 매핑을 찾는데 있어 새로운 명시적 예시들을 제공합니다. 연구 결과, 입방형 서젝티브 매핑은 특정한 일반적 조건을 만족할 경우, 정상적으로 존재하며, 그 결과는 많은 경우에 유용하게 적용될 수 있습니다. 또한, 이 연구는 대수 통계(algebraic statistics) 및 양자화(quantization)와의 관계를 탐구하는 데에도 기여하고 있습니다.



### A Novel Ensemble Learning Approach for Enhanced IoT Attack Detection: Redefining Security Paradigms in Connected Systems (https://arxiv.org/abs/2510.08084)
Comments:
          14 pages, 5 fiugres, 7 tables

- **What's New**: 본 연구는 IoT(Internet of Things) 장치의 보안 취약점을 공격 탐지 시스템으로 개선하기 위한 새로운 앙상블 학습 아키텍처를 제안합니다. 이 방법은 Extra Trees Classifier와 같은 고급 머신러닝 기술을 적용합니다. 기존의 보안 솔루션들과 비교하여, 제안된 방법이 더 효율적이라는 점에 주목해야 합니다.

- **Technical Details**: 제안된 모델은 데이터를 철저히 전처리(preprocessing)하고 하이퍼파라미터 최적화(hyperparameter optimization)를 포함하여 IoT 공격 탐지의 성능을 극대화합니다. 이 연구는 CICIoT2023, IoTID20, BotNeTIoT L01, ToN IoT, N BaIoT, BoT IoT와 같은 여러 벤치마크 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 높은 재현율(recall), 정확도(accuracy), 정밀도(precision)를 달성하며 매우 낮은 오류율(error rates)을 기록하였습니다. 이러한 성과는 IoT 환경을 보호하기 위한 효과적이고 확장 가능한 방법으로서의 모델의 능력을 입증합니다.



### Detecting and Mitigating Insertion Hallucination in Video-to-Audio Generation (https://arxiv.org/abs/2510.08078)
- **What's New**: 이 연구에서는 Video-to-Audio (V2A) 생성의 기존 평가 메트릭이 간과한 중요한 문제, 즉 Insertion Hallucination (IH)을 정의하고 체계적으로 측정하여 완화하는 방법을 제안합니다. IH는 보고되지 않은 시각적 출처에 대해 음성이나 음악과 같은 음향 이벤트가 생성되는 현상입니다. 본 논문에서는 IH에 대한 새로운 메트릭인 IH@vid와 IH@dur을 도입하여 이러한 현상의 유병률과 심각성을 정량화했습니다.

- **Technical Details**: 연구진은 자동으로 생성된 오디오에서 IH를 검출하기 위해 다수의 오디오 이벤트 탐지기를 활용한 평가 프레임워크를 개발했습니다. Posterior Feature Correction (PFC) 방법론을 통해 기존의 훈련 과정 없이 비디오 특성을 마스킹하여 생성된 오디오를 재생성하여 IH의 발생을 억제하는 두 가지 패스를 구현했습니다. 이 과정에서 세 가지 오디오 이벤트 탐지기를 통합하여 헌신적인 검증을 거쳤습니다.

- **Performance Highlights**: PFC 방법은 IH를 평균 50% 이상 줄이면서 기존의 오디오 품질과 시간 동기화 메트릭들인 FD-VGG, ISC, DeSync를 유지하거나 개선하는 결과를 보였습니다. 다양한 V2A 벤치마크에서 실험을 수행해 IH에 대한 심각한 문제를 확인하고, 제안한 방법의 효용성을 입증했습니다. 이러한 연구는 V2A 모델의 신뢰성과 충실도를 높일 수 있는 기반을 마련한 첫 시도로 평가됩니다.



### Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection (https://arxiv.org/abs/2510.08073)
Comments:
          Accepted at NeurIPS 2025 spotlight

- **What's New**: 이번 논문은 AI 생성 비디오의 신뢰할 수 있는 탐지 메커니즘을 필요로 한다는 점에서 새로운 점을 강조합니다. 저자들은 물리학 기반의 확률 흐름 보존 원칙에 기반하여 AI 생성 비디오를 탐지하는 새로운 패러다임을 제안하며, 이는 AI 생성 비디오의 시공간적 오류를 효과적으로 감지할 수 있는 방법입니다. 제안된 기법인 Normalized Spatiotemporal Gradient (NSG)는 실제 비디오와 AI 생성 비디오 간의 기본적인 불일치를 정량화합니다.

- **Technical Details**: NSG는 공간 확률 기울기와 시간 밀도 변화의 비율을 정량화하여 AI 생성 비디오의 물리적 제약 위반을 포착합니다. 저자들은 사전 훈련된 diffusion 모델을 활용하여 복잡한 모션 분해 없이 NSG 추정기를 개발하였으며, 이를 통해 영상의 동적 특성을 효과적으로 모델링할 수 있습니다. 또한, NSG 기반 비디오 탐지 방법(NSG-VD)은 테스트 비디오와 실제 비디오 간 NSG 특성의 Maximum Mean Discrepancy (MMD)를 검출 메트릭으로 사용합니다.

- **Performance Highlights**: 광범위한 실험 결과 NSG-VD는 Recall에서 16.00% 및 F1-Score에서 10.75% 향상된 성능을 보여주었으며, 이는 기존의 최첨단 모델보다 뛰어난 성능을 подтверж합니다. 이러한 결과는 NSG-VD의 탐지 능력이 우수함을 입증하며, AI 생성 비디오 감지에 있어 중요한 진전을 나타냅니다. 저자들은 이 연구의 소스 코드도 공개하고 있어 향후 연구에 활용될 수 있는 기초 자료를 제공합니다.



### Verifying Graph Neural Networks with Readout is Intractab (https://arxiv.org/abs/2510.08045)
- **What's New**: 본 논문에서는 글로벌 읽기 기능(global readout)이 포함된 양자화 집합-결합 그래프 신경망(ACR-GNNs)에 대한 추론을 위한 논리적 언어를 소개합니다. 또한, 양자화된 GNN의 검증 작업(verfication task)이 (co)NEXPTIME 완전함을 증명하여 양자화된 GNN의 검증이 계산적으로 어려운 문제임을 강조합니다. 이 결과는 GNN 기반 시스템의 안전성을 보장하기 위한 연구 노력을 촉구합니다.

- **Technical Details**: ACR-GNN의 검증을 수행하기 위해 새로운 논리 q​ℒq\mathcal{L}을 정의하고, 이를 통해 글로벌 읽기가 포함된 양자화된 GNN의 복잡성을 다룹니다. 이 논리는 다양한 활성화 함수를 가진 양자화 ACR-GNN을 포착할 수 있을 만큼 표현력이 뛰어나기 때문에, 그래프 속성을 표현하는 유연한 언어로 활용될 수 있습니다. 또한, q​ℒq\mathcal{L}의 만족 가능성 문제는 NP-complete로 완화하여, 정점 수가 제한된 그래프 카운터 예제를 검색합니다.

- **Performance Highlights**: 실험을 통해 양자화된 GNN 모델이 경량화되면서도 비양자화 모델에 비해 좋은 정확도와 일반화 능력을 유지함을 보여주었습니다. 본 연구 결과는 자원이 제한된 환경에서도 양자화된 ACR-GNN의 실제적인 활용 가능성을 확인합니다. 전체적으로 양자화된 모델은 모델 크기 및 추론 비용의 상당한 절감을 이루면서도 강력한 예측 성능을 유지합니다.



### Climate Knowledge in Large Language Models (https://arxiv.org/abs/2510.08043)
Comments:
          16 pages, 4 figures, 2 tables

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 기후 관련 응용 프로그램에서의 사용을 조사합니다. 특히 LLM이 파라메트릭 지식에서 기후 정상(Climate Normals)을 기억할 수 있는 능력이 얼마나 되는지를 평가합니다. 연구는 특정 위치에서의 1991-2020년 평균 7월 기온을 다루며, 다양한 지역적 맥락을 포함한 질문을 통해 LLM의 성능을 검증합니다.

- **Technical Details**: 연구진은 1° 해상도의 전 세계 쿼리 격자를 구성하고, ERA5 재분석 데이터를 기준으로 LLM의 응답을 검증했습니다. 그 결과 LLM은 위도 및 지형적 패턴을 포착하며, 루트 평균 제곱 오차(RMSE)는 3-6 °C로 나타났고, 바이어스는 ±1 °C였습니다. 그러나 산악 지역과 고위도에서의 공간적 일관된 오류가 발견되었습니다.

- **Performance Highlights**: 연구에 따르면, LLM은 1950-1974년과 2000-2024년 사이의 관측된 온난화의 글로벌 평균 크기를 캡처하였으나, 온도 변화의 공간적 패턴을 재현하는 데에는 실패했습니다. 또한, 지리적 맥락을 포함하면 오차가 평균 27% 감소하며, 대형 모델일수록 위치 설명자에 더 민감한 반응을 보였습니다. 이 연구는 LLM의 기후 지식 측정을 위한 재현 가능한 기준을 제공하며, 기후 커뮤니케이션 평가에 보완적인 역할을 합니다.



### Language Models Do Not Embed Numbers Continuously (https://arxiv.org/abs/2510.08009)
Comments:
          12 pages, 10 figures, 3 tables

- **What's New**: 이 연구는 최근 대형 언어 모델(Large Language Models, LLMs)이 숫자 및 산술 작업에서 정수를 조작하는 방식을 조사했습니다. 특히 LLM의 임베딩 공간이 숫자 값을 연속적으로 모델링하는 능력을 평가하지 않은 점에 주목했습니다. 우리는 이러한 모델들이 숫자 공간을 비연속적으로 나타내며, 실제로 상당한 노이즈를 도입한다고 설정하며, 이러한 결과는 수치 정밀도가 높은 특정 작업에서의 LLM의 활용 가능성을 의문시합니다.

- **Technical Details**: 이 논문에서는 선형 재구성(linear reconstruction) 및 주성분 분석(principal component analysis)을 포함한 임베딩 공간의 예측 속성을 활용했습니다. 세 개의 주요 LLM 제공업체(OpenAI, Google Gemini, Voyage AI)의 모델을 이용하여, 임베딩 공간에서의 변동성을 설명하는 주성분이 미미하다는 것을 발견했습니다. 이는 입력 숫자 공간의 단순성과의 관계 속에서 여러 구성 요소가 서로 직교함을 나타내며, 소수 자리 수가 커질수록 성능이 저하되는 경향을 보였습니다.

- **Performance Highlights**: 모델의 재구성 가능성은 높은 충실도를 보였지만($R^2 0.95$), 주성분이 설명하는 변동의 비율은 매우 적었습니다. 이는 고정밀 숫자, 큰 값 또는 부호가 혼합된 값이 일반적으로 나타나는 분야에서 LLM의 사용이 문제를 야기할 수 있음을 강조합니다. 이 연구 방법론은 모델의 수치 표현 능력을 측정하는 데 확장 가능하고 작업에 구애받지 않는 도구를 제공하며, 사용자가 이러한 모델의 한계를 보다 잘 이해할 수 있도록 돕습니다.



### VoiceAgentBench: Are Voice Assistants ready for agentic tasks? (https://arxiv.org/abs/2510.07978)
- **What's New**: 이번 논문에서 제안된 VoiceAgentBench는 대규모 음성 모델(SpeechLMs)에 대한 포괄적인 벤치마크로, 실제 음성 기반 상호작용 시나리오에서의 에이전트 기능을 평가하기 위해 디자인되었습니다. 이 벤치마크는 인도와 관련된 5,500개의 합성 음성 쿼리를 포함하여 다국어 및 문화적 이해를 측정할 수 있습니다. 특히, 음성 기반 에이전트가 복잡한 도구 사용, 다중 턴 상호작용, 그리고 맥락적 의사결정 능력을 포함하는 기본적인 에이전트 기능을 평가할 수 있는 첫 번째 벤치마크입니다.

- **Technical Details**: VoiceAgentBench는 다섯 개의 인도 언어를 포함하여 영어와 힌디를 지원하며, 수천 개의 음성 쿼리를 통해 다양한 도구 호출 유형을 평가합니다. 이는 단일 도구 호출부터 다중 종속 도구 오케스트레이션에 이르는 복잡한 요청을 포함합니다. 또한, 새로운 샘플링 알고리즘을 사용해 음성 전환 시 다양한 악센트와 음성 특성을 고려하여 실제 음성 대화의 이질성을 포착할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, 현재의 SpeechLMs는 맥락적인 도구 조정 작업과 인도적 일반화, 그리고 적대적 강건성에서 상당한 성능 차이를 보였습니다. 이는 기존의 음성 기반 모델들이 현실적인 에이전트 능력을 충분히 평가하지 못한다는 점을 강조하며, VoiceAgentBench의 필요성을 부각시킵니다. 또한, 우리는 SpeechLMs와 ASR-LLM 파이프라인 모두에서 주목할 만한 성능 차이를 발견했습니다.



### Stick-Breaking Mixture Normalizing Flows with Component-Wise Tail Adaptation for Variational Inferenc (https://arxiv.org/abs/2510.07965)
- **What's New**: 본 연구는 Bayesian inference의 posterior approximation을 개선하기 위해 stick-breaking mixture base와 component-wise tail adaptation 기법인 StiCTAF를 제안합니다. 이 접근 방식은 복잡한 다모드 포스터리오스를 보다 정확하게 모델링할 수 있도록 하며, 특히 tail behavior에 적합합니다. 이를 통해 샘플링의 질을 높이고, 여러 모드를 놓치지 않도록 확장된 flexibility를 확보합니다.

- **Technical Details**: 제안된 StiCTAF 방법은 복잡한 다모드 포스터리오스의 모드 편향을 완화하기 위해 적응형 혼합 기반을 학습합니다. 이 과정에서, component-wise ELBO의 가중 평균을 통해 모드 탐색의 편향을 수정하고, 비정규화 밀도의 로컬 tail index를 추정합니다. 이후, 공유된 backbone과 특정 tail transforms를 조합해 각 혼합 성분을 정제하여 정확한 밀도 평가와 안정적인 최적화를 유지합니다.

- **Performance Highlights**: 실험 결과, StiCTAF는 synthetic 포스터리오스에 대해 tail recovery와 여러 모드의 커버리지를 향상시켰음을 보여줍니다. 또한, 실제 데이터 분석을 통해 제안한 방법이 포스터리오 추론에서 실질적인 이점을 지닌다는 것을 입증했습니다. 이 접근 방식은 고차원 및 대규모 문제에서도 뛰어난 성능을 보입니다.



### A Systematic Evaluation of Self-Supervised Learning for Label-Efficient Sleep Staging with Wearable EEG (https://arxiv.org/abs/2510.07960)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 착용 가능한 EEG 장치를 활용한 수면 단계 분류에 대한 자가 지도 학습(SSL)의 체계적인 평가를 처음으로 수행했습니다. 기존의 수동 데이터 라벨링 의존도를 줄이고, 큰 양의 비라벨 데이터에서 유의미한 대표성을 추출하는 데 SSL의 가능성을 보여줍니다. 저자들은 또한 실질적인 환경에서의 사용 가능성을 위해 합리적인 비용의 수면 스코어링 파이프라인을 검증했습니다.

- **Technical Details**: 본 연구에서는 Ikon Sleep 착용 EEG 헤드밴드로 수집된 두 개의 수면 데이터베이스(BOAS 및 HOGAR)에서 다양한 SSL 방법을 평가했습니다. 그 결과, SSL은 라벨이 부족할 때 10%까지 분류 성능을 향상시키며, 5%에서 10%의 라벨 데이터만으로도 80% 이상의 임상 등급 정확도를 달성합니다. 이는 감독 학습 방식이 요구하는 두 배의 라벨보다 적은 수치입니다.

- **Performance Highlights**: SSL은 인구 특성, 녹음 환경 및 신호 품질의 변동성에도 강인한 성능을 보입니다. 연구 결과, SSL은 수면 단계 분류에서 라벨 효율성을 높이고, 수작업 주석에 대한 의존도를 줄임으로써 착용 가능한 EEG 시스템의 발전을 이끌 수 있는 잠재력을 보여줍니다. 이는 의료 비용 절감 및 접근성 향상으로 이어질 수 있습니다.



### SimCast: Enhancing Precipitation Nowcasting with Short-to-Long Term Knowledge Distillation (https://arxiv.org/abs/2510.07953)
Comments:
          accepted by ICME 2025

- **What's New**: 본 연구는 단기 및 장기 강수 예측에 대한 새로운 접근 방식인 SimCast를 제안합니다. 이 방법은 강수 예측 모델의 예측 지평선(prediction horizon)이 모델의 성능에 미치는 영향을 분석하여, 강수 예보의 정확성을 높이고자 합니다. SimCast는 짧은 기간의 데이터를 이용하여 장기 예측 성능을 향상시키는 지식 증류(knowledge distillation) 기법과 가중 평균 제곱 오차(weighted MSE loss)를 활용합니다.

- **Technical Details**: SimCast는 현재 2시간 이내의 강수 예측을 목표로 하는 단기 모델을 훈련시키고, 이를 기반으로 장기 예측 모델로의 지식 이전을 수행합니다. 모델 아키텍처로는 SimVP를 사용하며, CNN 구조를 기반으로 해 공간적 특징을 추출하고, 시간적 진화를 학습합니다. 이 과정에서 강화된 예측 결과는 CasCast라는 확산 기반(diffusion-based) 프레임워크와 통합되어 모호함(blurriness)을 극복하는 데 도움을 줍니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(SEVIR, HKO-7, MeteoNet)에서 SimCast의 성능이 기존 모델들에 비해 현저히 향상된 것으로 나타났습니다. CSI 점수는 SEVIR에서 0.452, HKO-7에서 0.474, MeteoNet에서 0.361을 기록했습니다. 특히 고강도 강수 지역에서의 성능이 더욱 두드러지며, 실용적인 응용에서도 추가적인 계산 부담 없이 장기 예측을 수행할 수 있습니다.



### TTOM: Test-Time Optimization and Memorization for Compositional Video Generation (https://arxiv.org/abs/2510.07940)
Comments:
          Project page: this https URL

- **What's New**: 본 연구에서는 비디오 생성 모델의 성능을 향상시키기 위해 Test-Time Optimization and Memorization (TTOM) 프레임워크를 소개합니다. 기존의 방식과는 달리, TTOM은 훈련 없이 spatiotemporal 레이아웃에 맞춘 출력을 제공합니다. 이는 기계 학습 모델이 영상에서 입력된 텍스트를 더 정련하게 이해하고 생성할 수 있도록 돕습니다.

- **Technical Details**: TTOM은 사용자 프롬프트에 기반한 spatiotemporal layout을 생성하고, 이를 통해 비디오 생성 모델의 성능을 최적화합니다. 새로운 매개변수를 도입하여 각 샘플에 맞춰 업데이트하며, 이 과정을 통해 이전 작업의 최적화를 메모리에 저장할 수 있습니다. 이 파라미터는 삽입, 읽기, 업데이트 및 삭제와 같은 다양한 작업을 지원하여 유연하고 효율적인 운영이 가능합니다.

- **Performance Highlights**: T2V-CompBench 및 Vbench 벤치마크에서의 실험 결과는 TTOM이 매우 효과적이고 실용적이며 효율적인 프레임워크임을 입증하였습니다. 특히, TTOM은 CogVideoX-5B와 Wan2.1-14B와 비교했을 때 T2V-CompBench에서 각각 34% 및 14%의 성능 향상을 이뤘습니다. 이는 복합 비디오 생성에서의 크로스 모달 정렬을 현장에서 자동으로 달성할 수 있도록 해줍니다.



### Multi-level informed optimization via decomposed Kriging for large design problems under uncertainty (https://arxiv.org/abs/2510.07904)
Comments:
          34 pages, 18 figures

- **What's New**: 이 연구에서는 불확실성 하에서 고차원 및 복잡한 공학 문제를 효과적으로 최적화하기 위한 새로운 방법론을 제안합니다. 기존의 두 단계 접근방식(uncertainty quantification, design optimization)을 넘어, 최소한의 자원으로도 최적화가 가능하도록 설계되었습니다. 이를 통해 자원 집약적 문제를 다루는 데 필요한 특정한 솔루션을 제공합니다.

- **Technical Details**: 고속 스케일링(non-intrusive, fast-scaling) 가능한 Kriging 기반 서브로게이트가 디자인/파라미터 도메인을 효율적으로 맵핑하는 데 개발됩니다. 또한, 계층적(hierarchical) 및 직교적(orthogonal) 분해를 통해 다중 서브로게이트가 적응적으로 업데이트되어 불확실성에 가장 영향을 받는 데이터만을 활용합니다. 이러한 접근 방식을 통해 방대한 데이터 세트를 다루는 과정에서 정확성과 효율성이 높아집니다.

- **Performance Highlights**: 제안된 방법은 기존의 최첨단(STATE-OF-THE-ART) 기술과 비교해 간결한 테스트를 통해 동시에 더 빠르고 정확하다는 것을 입증했습니다. 통계적 비교 분석 결과, 제안된 방법이 기존 방법들에 비해 수량적으로 많은 배 이상의 성능 향상을 보여줍니다. 이는 공학 설계 분야에서 실질적으로 적용 가능성을 높이는 중요한 연구 결과입니다.



### Team Xiaomi EV-AD VLA: Learning to Navigate Socially Through Proactive Risk Perception -- Technical Report for IROS 2025 RoboSense Challenge Social Navigation Track (https://arxiv.org/abs/2510.07871)
- **What's New**: 이 보고서는 IROS 2025 RoboSense Challenge의 Social Navigation Track에 대한 기술 세부 정보를 설명합니다. 이 트랙은 RGB-D 기반 인식 및 내비게이션 시스템을 개발하는 데 중점을 두며, 자율 에이전트가 동적 인간이 있는 실내 환경에서 안전하고 효율적으로 탐색할 수 있도록 돕습니다. 새로운 Proactive Risk Perception Module을 도입하여 사회적 내비게이션 성능을 향상시켰습니다.

- **Technical Details**: 사회적 내비게이션(Social Navigation)은 자율 로봇이 인간과 공유하는 환경에서 사회적 관습을 준수하며 탐색하는 능력을 나타냅니다. 이 연구는 Falcon 모델을 기반으로 하여, 상대 거리 정보를 활용해 인간과의 충돌 위험을 예측하는 모듈을 통합했습니다. 이는 충돌 회피 행동을 개선하고 공간 인식을 강화하는 데 기여합니다.

- **Performance Highlights**: Social-HM3D 벤치마크에서 우리의 방법이 밀집한 실내 장면에서 목표를 향해 탐색할 때 개인 공간 준수를 유지하는 능력을 개선했음을 보였습니다. 이 챌린지에서 16개 팀 중 2위를 달성하였으며, 이는 자율 탐색 시스템의 발전에 중요한 이정표가 될 것입니다.



### On the Optimality of the Median-of-Means Estimator under Adversarial Contamination (https://arxiv.org/abs/2510.07867)
- **What's New**: 이 논문에서는 Median-of-Means (MoM) 추정기를 통해 적대적인 오염(adversarial contamination) 아래에서의 오류의 상한 및 하한을 제시합니다. MoM은 기존의 이론에서 보여주던 것처럼 출시된 Gaussian 경우 외에도 여러 분포 클래스에서 (minimax) 최적 성능을 발휘합니다. 특히 유한 분산을 가진 분포 클래스에서 MoM의 최적성에 관한 새로운 경계를 제공합니다.

- **Technical Details**: MoM 추정기는 오염된 샘플로부터 평균을 추정하는 데 사용되며, 특히 샘플 중 최대 αn만큼의 비율이 변경될 수 있습니다. 이 연구에서는 MoM을 통해 강한 테일을 가진 분포와 대칭 분포에서 최적이며, 경량 경계가 있는 경우에는 서브 최적(sub-optimal) 성능을 보인다고 설명합니다. 특히 무한 분산을 가진 분포에서 MoM의 성능 한계를 구체적으로 규명합니다.

- **Performance Highlights**: MO 모드는 유한 분산 클래스에서 최적의 오류 경계를 제공하며, 서브-가우시안(sub-Gaussian) 및 경량 분포에 대해서도 오류 경계를 향상시킵니다. 결과적으로 MoM은 강한 테일을 갖는 분포에서 최적 성능을 달성하고, 가우시안 및 스튜던트(t) 분포와 같은 대칭 분포에서도 우수한 성능을 입증했습니다. 하지만 경량 분포에 대해서는 판단력이 떨어지는 것으로 나타났습니다.



### On the Optimality of Tracking Fisher Information in Adaptive Testing with Stochastic Binary Responses (https://arxiv.org/abs/2510.07862)
- **What's New**: 이 논문에서는 이진 응답을 기반으로 지속적인 능력 매개변수를 추정하는 문제를 다룹니다. 저자들은 질문의 난이도를 조절하여 사람이 응답할 때 최적의 정보를 얻는 방법을 제시하며, 이 과정에서 Fisher 정보(Fisher information)를 최대화하는 질문 선택 방식을 추천합니다. 이러한 접근은 온라인 선호 학습과 적응형 테스트(a adaptive testing)에서 자연 발생하는 조건이며, 가능한 적은 질의로 원하는 오차 범위 내에 추정값을 인증하는 것을 목표로 합니다.

- **Technical Details**: 제안된 알고리즘은 method-of-moments 방식을 사용하여 지속적으로 추정값을 업데이트합니다. 또한, novel test statistic을 도입하여 추정치가 충분히 정확해질 때 결정을 내리는 프로세스를 효율적으로 진행합니다. 특히, 본 연구는 고정 예산(fixed-budget) 설정 내에서 추정-질의의 내생적 상관관계(estimate-query endogeneity) 문제를 해결하기 위해 Ville의 부등식을 활용하였습니다.

- **Performance Highlights**: 저자들은 제안된 알고리즘이 고정 신뢰도(fixed-confidence) 및 고정 예산 설정에서 최적 성능을 달성함을 이론적으로 증명합니다. 특히, Fisher 정보 최대화 쿼리 선택 전략은 이 문제에 대한 개념적으로 간단한 알고리즘으로, 연속적 쿼리 설정에서의 최적성을 처음으로 이론적으로 입증하였습니다. 이 논문은 간단하고 효율적인 적응형 테스트 절차에 대한 이론적 지원을 제공합니다.



### Augur: Modeling Covariate Causal Associations in Time Series via Large Language Models (https://arxiv.org/abs/2510.07858)
Comments:
          22 pages, 9 figures

- **What's New**: 이번 연구에서는 Augur라는 새로운 시계열 예측 프레임워크를 소개합니다. Augur는 기존의 LLM 기반 접근법이 가진 한계를 극복하며, 다중 모드 데이터를 통합하는 가능성을 제시합니다. 특히, 이 모델은 LLM의 인과적(reasoning) 추론 능력을 활용하여 변수들 간의 방향성 인과 관계를 탐색합니다.

- **Technical Details**: Augur는 두 단계의 교사-학생 모델 아키텍처를 사용합니다. 강력한 교사 LLM이 시계열 데이터로부터 방향성 인과 그래프를 추론하며, 이를 위해 휴리스틱 검색과 쌍별 인과성 테스트(pairwise causality testing)를 이용합니다. 경량의 학생 에이전트는 이 그래프를 정제하고 높은 신뢰도의 인과 관계를 바탕으로 예측을 수행합니다.

- **Performance Highlights**: 실제 데이터셋을 이용한 광범위한 실험에서 Augur는 25개의 기준 모델과 비교해 경쟁력 있는 성능을 보여주었습니다. 또한, Augur는 제로 샷 영구화(zero-shot generalization)에서 강력한 성능을 입증하였습니다. 이는 Augur의 투명하고 추적 가능한 변수 상호작용(reasoning)을 강화합니다.



### Self-Supervised Learning Strategies for a Platform to Test the Toxicity of New Chemicals and Materials (https://arxiv.org/abs/2510.07853)
- **What's New**: 이 논문은 자동화된 독성 검사 시스템에서 기계 학습 모델을 통해 독성 물질로 인한 변화를 효과적으로 식별할 수 있는 방법을 시연합니다. 자가 지도 학습(self-supervised learning)을 통해 학습된 표현이 독성 물질에 의한 변화를 확인할 수 있음을 보여주는 개념 증명을 제시합니다. 이를 위해 EmbryoNet 데이터셋을 활용하고, 다양한 화학 화합물에 의한 제브라피시 배아의 표현형을 연구하였습니다. 최종적으로 TOXBOX 프로젝트의 일환으로 물리적 독성 검사 장치에 기계 학습 모델을 통합하는 방안을 논의합니다.

- **Technical Details**: REACH 규제는 EU 시장에 진입하는 화학 화합물을 더 잘 이해하기 위한 목적으로, 특정 화합물의 수량이 1톤을 초과하는 경우 독성 테스트를 수행하고 결과를 European Chemicals Agency (ECHA)에 보고하도록 요구합니다. 기존의 독성 테스트는 일반적으로 비용이 많이 드는 인 비보(in vivo) 방법을 사용하는데, 이 논문에서는 인 비보 연구의 대안으로 제브라피시 배아를 활용한 고처리량 스크리닝(High-Throughput Screening, HTS)에 대한 관심이 증가하고 있음을 강조합니다. 자가 지도 학습을 통해 학습된 연속적 표현은 독성 물질에 의한 변화를 모델링할 수 있으며, 자동 평가 방법을 ML 모델을 통해 시행할 수 있습니다.

- **Performance Highlights**: 이 연구는 자가 지도 학습을 통해 학습한 표현이 화합물의 작용 방식(modes-of-action)을 효과적으로 구별할 수 있음을 입증하였습니다. 특히, 실험 데이터의 고차원성을 처리하는 데 있어 딥 러닝(Deep Learning) 모델이 전통적인 머신 러닝 모델보다 더 적합하다는 점을 강조하고 있습니다. 또한, 제브라피시의 다양한 배아 표현형을 비교하는 데 있어서 이 방법이 유용하다는 것을 보여주었으며, TOXBOX와 같은 실제 독성 검사 장치에 DL 모델을 통합하는 데 따른 도전 과제를 다루고 있습니다.



### Surrogate Graph Partitioning for Spatial Prediction (https://arxiv.org/abs/2510.07832)
Comments:
          18 pages, 5 figures, 2 tables

- **What's New**: 이 연구에서는 블랙박스 예측기의 예측을 설명할 수 있는 대리 모델을 개발하기 위해 그래프 분할 문제를 제안했습니다. 이를 통해 각 세그먼트 내 예측의 분산 합을 최소화하는 방향으로 공간 세그먼트를 구성하려 합니다. 이 최적화 문제는 혼합 정수 이차 프로그램(MIQP)으로 정식화되며, 데이터 점의 수가 증가할수록 컴퓨팅 복잡성이 증가하는 문제를 지적합니다.

- **Technical Details**: 연구에서는 공간 예측을 위해 가우시안 프로세스 회귀(Gaussian Process Regression)를 활용하며, 분산 및 공분산 함수의 선택이 예측 성능에 미치는 영향에 대해 설명합니다. 공간 세그멘테이션을 위해 그래프 분할(graph partitioning)의 제약을 부여하여, 세그먼트 내 위험 동질성을 보장하기 위해 각 세그먼트의 예측 분산 합을 최소화하는 목표를 설정합니다. 이 과정에서 MIQP의 복잡성 문제를 해결하기 위해, 인근 데이터 포인트를 집계하는 근사화 계획을 개발하였습니다.

- **Performance Highlights**: 실험 결과는 제안한 근사화 방법이 공간 세그먼트 식별의 정확성을 효율적으로 향상시킨다는 것을 보여줍니다. 이 접근 방식은 공간 데이터에서 블랙박스 예측기의 복잡성을 낮추고 해석 가능성을 높일 수 있는 가능성을 시사합니다. 결과적으로 연구의 성과는 자연 재해 위험 관리와 같은 특정 산업에서의 실제 적용 가능성을 증대시키는 데 기여할 수 있습니다.



### Adaptive Execution Scheduler for DataDios SmartDiff (https://arxiv.org/abs/2510.07811)
Comments:
          4 pages, 1 figure

- **What's New**: 이 논문에서는 SmartDiff라는 단일 차이 비교 엔진에 대한 적응형 스케줄러를 제안합니다. 이 스케줄러는 두 가지 실행 모드를 갖추고 있으며, 95번째 백분위수(latency) 최소화를 위해 CPU와 메모리 제약 내에서 배치 크기와 작업자 수를 동적으로 조정합니다. 또한 안전한 동작을 보장하기 위한 메모리 모델과 저지 정책이 포함되어 있습니다.

- **Technical Details**: SmartDiff의 실행 파이프라인은 소스 테이블 A와 대상 테이블 B 간의 스키마 정렬을 통해 행 정렬 기능을 적용합니다. 이 스케줄러는 작업이 독립적으로 처리될 수 있도록 유도하며, 각 배치가 처리된 후에는 수행 시간과 메모리 사용량을 기록하여 적절한 조정이 이루어지도록 합니다. 사용자는 메모리 안전성을 보장하기 위해 명시된 자료 모델을 통해 안정성 있는 실행 백엔드를 선택할 수 있습니다.

- **Performance Highlights**: 이 스케줄러는 비교 작업에서 p95 latency를 기존의 따뜻한 시작(heuristic) 방법 대비 23%에서 28% 줄이고, 고정 그리드 기준에서 35%에서 40% 개선하였습니다. 또한 메모리 사용량은 정점에서 16%에서 22% 감소하였으며, OOM(Out Of Memory) 발생 없이 유사한 처리량을 유지했습니다. 이러한 성과는 문서에서 다룬 적응형 제어를 통해 이루어진 것입니다.



### HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation (https://arxiv.org/abs/2510.07794)
Comments:
          Under review

- **What's New**: 이 논문에서는 HiPRAG(Hierarchical Process Rewards for Efficient agentic RAG)라는 새로운 강화학습(RL) 훈련 방법론을 제안하며, 이는 기존의 suboptimal search behaviors를 해결하기 위해 설계되었습니다. HiPRAG는 검색 과정의 최적화뿐만 아니라 각 검색 결정에 대한 구체적인 피드백을 제공하여 효율성을 높이고 정확성을 증대시키는 데 초점을 맞추고 있습니다. 특히 이 방식을 통해 over-search와 under-search 문제를 줄이고, 동시에 에이전트의 추론 과정에 대한 세밀한 제어를 가능하게 합니다.

- **Technical Details**: HiPRAG는 LLM이 검색 도구를 사용할 때, 이 도구의 사용 방식이 최적화되어야 한다는 점을 강조합니다. 이 방법론은 에이전트의 추론 과정을 개별적으로 분해하여 각 단계에 대해 적절한 보상을 부여함으로써, 최종 결과뿐 아니라 과정의 품질을 충분히 고려합니다. 이를 통해 검색 결정의 필요성을 동적으로 평가하고, 최적의 검색 비율에 보너스를 제공하는 단계적 보상 체계를 구현하여 기존 방법의 한계를 극복합니다.

- **Performance Highlights**: 실험 결과, HiPRAG는 Qwen2.5와 Llama-3.2 모델에 대해 65.4%(3B) 및 67.2%(7B)의 평균 정확도를 기록했습니다. 이는 기존 모델들에 비해 수집 효율성과 accuracy 모두에서 현저한 개선을 보여주며, over-search 비율을 27%에서 2.3%로 줄이고 under-search 비율도 낮추는 성과를 올렸습니다. 전반적으로 HiPRAG는 다양한 LLM 및 RL 알고리즘에 대해 우수한 일반성을 보이며, 검색 에이전트의 추론 효율성과 최적성을 크게 향상시키는 가능성을 제시합니다.



### PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations (https://arxiv.org/abs/2510.07784)
Comments:
          11 pages, 6 figures

- **What's New**: 이번 논문에서는 PLUM이라는 프레임워크를 소개하며, 이는 사전 학습된 Large Language Models (LLMs)을 산업 규모의 추천 시스템에 적합하게 조정하는 방식입니다. PLUM은 Semantic IDs (SIDs)를 활용한 항목 토큰화, 도메인 특화 데이터에 대한 지속적 사전 학습(Continued Pre-training, CPT), 그리고 추천 목표를 위한 작업 특화 미세 조정(task-specific fine-tuning)으로 구성됩니다. 이러한 접근 방식은 전통적인 삽입 테이블 방식의 한계를 극복하고 LLM의 능력을 최대한 활용하기 위한 노력의 일환으로 볼 수 있습니다.

- **Technical Details**: PLUM 프레임워크는 항목 토큰화 단계에서 각 항목을 SIDs로 표현하여 사용자의 행동 신호와 다중 모드 콘텐츠 임베딩을 향상시키는 새로운 기술(SID-v2)을 제공합니다. 지속적 사전 학습 단계에서는 사전 학습된 LLM의 어휘를 새 SID 토큰으로 확장하고, 도메인 특화 항목 데이터와 사용자 시퀀스의 혼합으로 모델을 추가로 학습시킵니다. 마지막으로, 작업 특화 미세 조정 단계에서 모델은 사용자가 관심 가질 항목의 SIDs를 생성하는 방식으로 Fine-tuning됩니다.

- **Performance Highlights**: 대규모 내부 비디오 추천 데이터셋에서 PLUM을 적용한 실험 결과, 기존의 대규모 정보 테이블을 기반으로 한 최적화 모델에 비해 PLUM 사용 시 검색 성능이 크게 향상됨을 보였습니다. PLUM을 사용한 모델은 Transformers 아키텍처에서 100배 더 많은 밀집 매개변수를 가지면서도 훈련 비용은 유사하게 유지되며, 샘플 효율성을 높일 수 있음을 보여주었습니다. 또한, PLUM 기반 검색은 YouTube에서 실제 운영 중으로, 장기 및 단기 비디오를 추천하는 데 효과적으로 활용되고 있습니다.



### Instance Relation Learning Network with Label Knowledge Propagation for Few-shot Multi-label Intent Detection (https://arxiv.org/abs/2510.07776)
- **What's New**: 이번 연구에서는 Few-shot Multi-label Intent Detection (MID) 문제를 해결하기 위해 다중 라벨 공동 학습 방법을 제안합니다. 기존의 두 단계 파이프라인 방식에서 벗어나, 인스턴스 관계 학습 네트워크를 통해 레이블 지식 전파(label knowledge propagation)를 수행하여 오류 전파를 제거합니다. 이 방법은 지원 세트와 쿼리 세트 간의 강한 상호작용을 모델링함으로써 멀티 레이블 예측의 성능을 향상시킵니다.

- **Technical Details**: 제안하는 방법은 메타 학습(meta-learning) 틀 안에서 Few-shot MID 작업을 정의하며, 각 메타 작업은 지원 세트와 쿼리 세트를 포함합니다. 지원 세트는 여러 클래스와 인스턴스를 가지며, 각 인스턴스에 대해 다수의 레이블을 할당할 수 있습니다. 또한, 듀얼 관계 강화 손실(dual relation-enhanced loss)을 설계하여 지원 및 쿼리 수준의 관계 강도를 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 강력한 베이스라인 대비 평균 9.54% AUC 및 11.19% Macro-F1 향상을 기록하며, 특히 1-shot 설정에서 두드러진 성능 향상을 보여주었습니다. 이는 저자원(dialogue domain) 환경에서도 효과적으로 작용함을 시사합니다.



### When Robustness Meets Conservativeness: Conformalized Uncertainty Calibration for Balanced Decision Making (https://arxiv.org/abs/2510.07750)
- **What's New**: 본 논문에서는 기존의 기존 명세된 강건성 수준에 의존하지 않고, 데이터 기반의 강건성 수준을 고려할 수 있는 새로운 통계적 프레임워크를 제안합니다. 이 프레임워크는 miscoverage와 regret에 대한 분포 자유의 유한 샘플 보장을 제공합니다. 결정을 내리는 과정에서 의사결정자가 비용-위험 선호도를 고려할 수 있도록 도와줍니다.

- **Technical Details**: 제안된 방법은 강건 예측-최적화 정책의 모든 가족에 대해 miscoverage와 regret의 무작위 없는 추정기를 구축합니다. 기존의 conformal risk control (CRC) 방법과는 다르게, 이제 의사결정자는 주어진 불확실성 집합에 대해 실제 위험 수준을 인증받을 수 있습니다. 이 접근방식은 기존의 heuristics와 무관하게 명확한 무역 환경을 통해 의사결정자의 조정을 가능하게 만듭니다.

- **Performance Highlights**: 제안된 추정기는 miscoverage-regret 무역의 Pareto 프론티어를 신뢰할 수 있게 추적하며, 기초적 조건 하에서도 우수한 유한 샘플 성능을 나타냅니다. 실험적으로, 이 추정기는 파레토 프론티어를 정확하게 추적하고 의사결정자에게 근사 최적의 강건성 매개변수를 안내해 줍니다. 이러한 방법론은 비용 효율적인 솔루션을 달성할 수 있도록 지원하며, 결정 이론의 폭넓은 적용을 보장합니다.



### Parallel Test-Time Scaling for Latent Reasoning Models (https://arxiv.org/abs/2510.07745)
- **What's New**: 이 논문은 잠재적 추론(latent reasoning) 모델에 대해 병렬 테스트 시간 스케일링(parallel test-time scaling, TTS)을 구현함으로써 이를 개선할 가능성을 탐구합니다. 특히, 토큰 기반 모델이 사용하는 샘플링 및 집계 메커니즘의 한계를 극복하기 위해 두 가지 확률적 샘플링 전략인 모나코 카로우 드롭아웃(Monte Carlo Dropout)과 추가 가우시안 노이즈(Additive Gaussian Noise)를 도입합니다. 또한 잠재 보상 모델(Latent Reward Model, LatentRM)을 설계하여 각 추론 단계에서 추론 과정을 평가하고 안내하는 방법을 제안합니다.

- **Technical Details**: 문서는 병렬 TTS가 잠재적 추론 모델에 효과적으로 적용될 수 있도록 샘플링 및 집계 메커니즘을 재구성하는 방법을 제시합니다. 첫째, 모나코 카로우 드롭아웃과 추가 가우시안 노이즈를 사용하여 잠재 공간에서 추론 경로를 다양하게 샘플링하는 방법을 설명합니다. 둘째, LatentRM을 통해 잠재적 경로를 평가하고 안내하는 세밀한 방법론을 개발하여 성능 향상을 이끌어냅니다.

- **Performance Highlights**: 상세한 실험과 시각적 분석 결과 두 가지 샘플링 전략 모두 계산량이 증가함에 따라 효과적으로 확장됨을 보여줍니다. 모나코 드롭아웃은 비정상적인 솔루션으로의 구조화된 확장을 촉진하며, 추가 가우시안 노이즈는 보다 넓고 동등한 탐색을 유도하여 다양성을 풍부하게 합니다. 최종적으로 LatentRM을 통해 다양한 계산 예산 하에 일관된 성능 향상이 이루어졌습니다.



### ToolExpander: Extending the Frontiers of Tool-Using Reinforcement Learning to Weak LLMs (https://arxiv.org/abs/2510.07737)
- **What's New**: 이 논문은 GRPO(Group Relative Policy Optimization)를 기반으로 한 툴 확장 도구인 ToolExpander를 제안합니다. 이 프레임워크는 작은 스케일의 대형 언어 모델(LLMs)에 대해 효율적인 reinforcement learning을 가능하게 합니다. ToolExpander는 두 가지 주요 혁신 기능으로, 동적 다중 라운드 하드 샘플링(Dynamic Multi-Round Hard Sampling)과 자기 예시적 사고(Self-Exemplifying Thinking)를 포함합니다.

- **Technical Details**: 동적 다중 라운드 하드 샘플링은 정확한 출력을 생성하지 못하는 어려운 샘플을 고품질의 Few-shot 데모로 대체하며, 학습률을 점진적으로 감소시켜 학습의 진동을 줄이는 전략을 사용합니다. 자기 예시적 사고 프레임워크는 KL 발산을 제거하고 조정된 클리핑 계수를 포함하여 모델이 최소한의 추가 보상(0.01)으로 자율적으로 Few-shot 예시를 생성하고 분석하도록 장려합니다.

- **Performance Highlights**: 실험 결과 ToolExpander는 LLM의 툴 활용 능력을 크게 향상시켰으며, 특히 1.5B 파라미터 모델에서 학습 안정성을 확보하고 성능을 향상시켰습니다. 이 모델은 APIBank에서 81.76%의 정확도를 달성하여 전통적인 GRPO 모델보다 월등한 성능을 보였습니다. 추가적으로, 자기 예시적 사고 메커니즘은 모델이 복잡한 작업을 효율적으로 처리할 수 있도록 돕습니다.



### Causality Guided Representation Learning for Cross-Style Hate Speech Detection (https://arxiv.org/abs/2510.07707)
- **What's New**: 최근 온라인 증오 발언(hate speech)의 확산은 웹의 조화에 심각한 위협이 되고 있습니다. 기존의 증오 발언 탐지 모델은 주로 표면적인 언어적 단서를 기반으로 작동하고 있어 다양한 스타일 변형에 효과적으로 일반화하지 못하는 문제가 있습니다. 이러한 배경을 토대로, 본 논문에서 제안한 CADET는 causal representation learning 프레임워크로, 증오 발언을 해석 가능한 잠재 인자로 분해하여 진정한 증오 의도를 표면적인 언어적 단서와 분리합니다.

- **Technical Details**: CADET는 증오 발언 생성 과정을 모델링하기 위해 causal graph(인과 그래프)를 사용하여, 상황적 환경(contextual environment), 작성자 동기(creator motivation), 대상(target), 스타일(style) 등 핵심 요소들을 포함합니다. 논문의 접근 방식에서는 모델이 각 잠재 인자를 구분하고, 상황적 환경이 미치는 혼란 요인(confounder)을 제어하여 보다 견고한 탐지를 가능하게 합니다. 이론적으로, CADET는 잠재 공간(latent space) 내에서 스타일을 조정함으로써 반사실적 추론(counterfactual reasoning)이 가능하다는 점이 강조됩니다.

- **Performance Highlights**: CADET는 다양한 도전 과제를 중심으로 한 평가에서 우수한 성능을 보여주었으며, cross-style generalization 작업에서 평균 macro-F1이 0.81에 달합니다. 이는 기존의 최첨단 방법에 비해 13%의 상대적 개선을 이룬 것입니다. 각 구성요소의 중요한 역할이 입증되었고, 인과 기반 설계의 효과가 숨겨진 것처럼 분석되었습니다. 이 연구는 일반화된 증오 발언 탐지를 개선하기 위한 인과적인 분리(causal disentanglement)의 가능성을 강조하며, 안전하고 책임감 있는 온라인 환경 조성을 위한 실용적인 함의를 제공합니다.



### Large Language Models Meet Virtual Cell: A Survey (https://arxiv.org/abs/2510.07706)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 '가상 세포'를 개발하여 세포 생물학을 혁신적으로 변화시키고 있다는 점을 강조합니다. 이러한 가상 세포 시스템은 세포 상태와 행동을 예측하고 이해하는 데 중요한 역할을 하고 있습니다. 이 논문은 LLMs의 다양한 활용 방법을 단순화된 두 가지 패러다임—Oracle로서의 LLM과 에이전트로서의 LLM—으로 정리하여, 생물학적 모델링에 대한 통합된 분류 체계를 제안합니다.

- **Technical Details**: 연구는 LLM을 가상 세포의 예측 엔진으로 활용하여 세포의 내적 상태와 동역학을 직접 모델링합니다. DNA, RNA 및 단일 세포 전사체 프로필과 같은 생물학적 서열을 활용하여 LLM이 원시 데이터에서 세포 구성 요소 및 상호작용의 표현을 학습한다는 점이 중요합니다. 연구는 세 가지 핵심 작업인 세포 표현(Cellular Representation), 교란 예측(Perturbation Prediction), 유전자 기능 및 조절 예측(Gene Function & Regulation Prediction)을 지금까지 논의해온 방식으로 정리합니다.

- **Performance Highlights**: LLMs는 셀 시스템을 모델링하는 데 있어 미리 정의된 데이터셋과 벤치마크를 활용하여 탐색과 실험을 진행하는 에이전트로서의 역할을 수행하고 있습니다. 연구는 LLM을 통한 다양한 세포 상호작용 모델링의 적용 예를 제공할 뿐 아니라, 대량의 생물학적 데이터를 이용한 가상 세포 시스템 개발의 기회를 강조합니다. 이와 같은 접근은 세포와 유전자의 복잡한 상호작용을 실시간으로 이해하고, 개인화된 의학의 길을 열 수 있는 잠재력을 가지고 있음을 보여줍니다.



### A Honest Cross-Validation Estimator for Prediction Performanc (https://arxiv.org/abs/2510.07649)
- **What's New**: 이 논문은 기존의 교차 검증 기법이 특정 모델의 성능을 직접적으로 평가하지 못한다는 비판에 따라, 특정 훈련 세트에서 훈련된 모델의 성능을 추정하는 새로운 방법론을 제안합니다. 저자들은 고전적인 교차 검증 기법이 여전히 유용할 수 있음을 보여주며, 이 정보를 활용하여 더 나은 추정치를 제공할 수 있는 방법을 개발했습니다. 이 연구는 새로운 방법이 기존의 단순 분할 추정기보다 우수한 성능을 발휘함을 확인하기 위한 실험과 실제 데이터 분석을 포함합니다.

- **Technical Details**: 이 연구에서는 관측된 독립적인 동일 분포(i.i.d) 데이터셋을 두 개의 서로 다른 부분으로 나누고, 특정 훈련 세트에 기초하여 예측 모델을 훈련 후 평가하는 과정을 설명합니다. 저자들은 무작위 효과 모델(random effects model) 프레임워크를 활용하여 다양한 랜덤 분할에서 교차 검증 추정치를 통해 모델의 예상 성능을 개선하는 방법을 체계화합니다. 두 가지 추정기, 즉 계층 베이지안 추정기(hierarchical Bayesian estimator)와 경험적 베이지안 추정기(empirical Bayes estimator)를 개발하여, 각기 다른 모델의 성능을 비교하는 방법을 제시합니다.

- **Performance Highlights**: 제안된 방법은 전통적인 교차 검증 추정기 및 단순한 분할 추정기보다 유사하거나 더 나은 성능을 나타냅니다. 실험 결과는 새로운 방법이 다양한 데이터 분할에서 더 견고한 성능 추정을 제공함을 보여줍니다. 특히, 실험과 실제 데이터 분석을 통해 제안된 방법이 모델의 예측 정확도를 개선하는 데 효과적이라는 것을 입증하였습니다.



### Test-Time Matching: Unlocking Compositional Reasoning in Multimodal Models (https://arxiv.org/abs/2510.07632)
- **What's New**: 이번 연구는 AI 모델의 compositional reasoning(구성적 추론) 문제를 재조명하며, 표준 평가 메트릭이 모델의 능력을 과소 평가한다는 점을 보여줍니다. 이를 해결하기 위해 그룹 구조를 더 잘 활용하는 그룹 매칭 점수(Group Matching Score)를 도입하여, 기존의 지표에서는 발견할 수 없는 모델의 숨은 능력을 드러냅니다. 연구 결과, SigLIP-B16과 GPT-4.1은 이전 모든 결과를 초월하는 성과를 이뤘습니다.

- **Technical Details**: 신선한 접근법으로 Test-Time Matching (TTM)이라는 반복적이고 자기 개선 가능한 알고리즘을 제안하였습니다. 이 알고리즘은 매칭 기반의 의사 레이블을 선택하여 자기 학습을 진행하고, 점진적으로 선택 기준을 완화하여 테스트 데이터셋에 대한 범위를 확장합니다. 이로 인해 SigLIP-B16과 GPT-4.1은 여러 벤치마크에서 놀라운 성능 향상을 보였습니다.

- **Performance Highlights**: TTM을 통해 SigLIP-B16은 Winoground에서 72.5, MMVP-VLM에서 89.44, ColorSwap에서 94.25의 성과를 기록하며 새로운 최첨단 결과를 세우고 있습니다. 특히 도전적인 데이터셋인 WhatsUp에서는 최대 85.7%의 상대적 성과 향상이 이루어졌습니다. 연구에서 TTM은 평가 메트릭의 변화를 극복하며 일관되게 모델 성능을 향상시키는 데 효과적입니다.



### From Data to Rewards: a Bilevel Optimization Perspective on Maximum Likelihood Estimation (https://arxiv.org/abs/2510.07624)
- **What's New**: 이번 논문은 생성 모델(Generative Models)의 한계를 극복하기 위해 Bilevel Optimization (Bi-O) 프레임워크를 제안합니다. 전통적으로 Maximum Likelihood Estimation (MLE)에 의존하던 생성 모델의 훈련 방식에서, 보상 함수(Reward Function)를 최적화하는 외부 문제와 정책 기울기(Policy Gradient) 목표를 정의하는 내부 문제로 나누어 접근합니다. 이는 최신 Reinforcement Learning 기술과 비교할 때, 데이터가 고품질일 경우에도 더 효과적으로 모델을 정렬할 수 있는 가능성을 제시합니다.

- **Technical Details**: Bilevel Optimization을 활용하여 MLE를 재정립하고, Gaussian 데이터 분포 아래에서 보상 함수를 음의 스케일 거리로 정의하여 이론적인 최적 파라미터를 도출합니다. 또한, 암묵적 미분(Implicit Differentiation)을 이용한 두 가지 실용적 알고리즘을 제안하며, 이를 다양한 MLE 응용 분야인 표 형식 분류(Tabular Classification)와 모델 기반 강화 학습(Model-Based Reinforcement Learning)에 대해 평가합니다.

- **Performance Highlights**: 이 논문에서 제안하는 방법은 기존 MLE 기반 접근 방식보다 우수한 성능을 보이며, 특히 인간의 선호를 모델에 더욱 효과적으로 반영하게 됩니다. 최적화 문제를 정리하고 실험 결과를 통해, Bilevel Optimization이 생성 모델의 발전에 기여할 수 있는 길을 보여줍니다. 이 연구는 다양한 애플리케이션에서 공유된 코드를 통해 보다 넓은 적용 가능성을 가지고 있습니다.



### Retentive Relevance: Capturing Long-Term User Value in Recommendation Systems (https://arxiv.org/abs/2510.07621)
- **What's New**: 이 논문에서는 Retentive Relevance라는 새로운 콘텐츠 수준의 설문 기반 피드백 지표를 소개합니다. 이는 사용자가 유사한 콘텐츠를 위해 플랫폼에 돌아올 의도를 직접 평가하여, 장기적인 사용자 만족도와 유지율을 측정할 수 있는 강력한 예측 도구로 제안됩니다. Retentive Relevance는 기존의 단기적인 만족도를 중심으로 한 설문 지표와는 차별화됩니다. 이를 통해 우리는 사용자 행동의 날짜별 유지율을 개선하는 데 성공했습니다.

- **Technical Details**: Retentive Relevance는 심리측정(psychometric) 방법을 통해 검증된 유효한 설문 구성으로, 사용자 만족과 의도를 동시에 측정함으로써 추천 시스템에서 유용한 도구로 자리잡고 있습니다. 이 연구는 대규모 오프라인 모델링, A/B 실험 등을 통해 Retentive Relevance가 전통적인 참여 신호와 다른 설문 지표들보다 더 뛰어난 예측 성능을 보임을 입증하였습니다. 따라서 우리는 추천 시스템에서 Retentive Relevance를 통합하는 프로덕션 준비 모델을 개발하여, 실질적인 운영에서의 적용 가능성을 높였습니다.

- **Performance Highlights**: 저자들은 Retentive Relevance가 사용자 유지율, 참여도, 콘텐츠 품질을 향상하는 데 효과적이라는 것을 대규모 실험을 통해 보여주었습니다. 특히 제한된 역사적 참여 기록을 가진 사용자들에게 전통적인 지표보다 더 좋은 성과를 내는 것이 특징입니다. 이에 따라 플랫폼의 성장과 사용자 경험을 동시에 개선할 수 있는 확장 가능하고 사용자 중심의 솔루션을 제공하게 되었습니다.



### Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction (https://arxiv.org/abs/2510.07594)
Comments:
          Accepted to NeurIPS 2025 Machine Learning and the Physical Sciences Workshop

- **What's New**: 이번 연구에서는 HEPT(해싱 기반 효율적 포인트 변환기)와 GNN(그래프 신경망) 기반 파이프라인을 동일한 데이터 세트와 성능 지표 아래에서 공정하게 평가합니다. 또, HEPTv2라는 새로운 시스템을 소개하는데, 이는 경량 디코더를 추가하여 클러스터링 단계 없이 직접적으로 트랙 할당을 예측합니다. 이는 HEPT의 하드웨어 친화적인 계산 방식을 유지하면서 초고속 엔드 투 엔드 추론을 가능하게 합니다.

- **Technical Details**: HEPTv2는 기존의 HEPT와는 달리 개별 딥러닝 모듈에서 다양한 단계와 손실(여기서 ℒNCE, ℒCLF, ℒCE 등)을 통해 학습됩니다. 키 아이디어는 RAW 감지 데이터를 직접 트랙 식별로 매핑하는 것으로, 점 처리 전 과정을 엔드 투 엔드로 실행하여 효율성과 처리 속도를 극대화합니다. LSH(지역 민감 해싱) 기법을 활용하여 부담스러운 계산 복잡도를 선형으로 줄이고, 하드웨어 친화적인 운영을 보장합니다.

- **Performance Highlights**: TrackML 데이터셋에서 최적화된 HEPTv2는 NVIDIA A100에서 이벤트당 약 28ms를 기록하며 경쟁력 있는 트래킹 효율성을 유지합니다. 이러한 결과는 HEPTv2가 빠른 트래킹을 위한 실용적이고 확장 가능한 GNN 기반 파이프라인의 대안으로 자리 잡을 수 있음을 입증합니다. HEPTv2는 향후 LHC(대형 하드론 충돌기)의 증가된 데이터 처리 요구를 충족할 수 있는 잠재력을 가집니다.



### Linguistic Patterns in Pandemic-Related Content: A Comparative Analysis of COVID-19, Constraint, and Monkeypox Datasets (https://arxiv.org/abs/2510.07579)
Comments:
          16 pages

- **What's New**: 이번 연구는 팬데믹 관련 온라인 담론에서 언어가 건강에 대한 허위 정보와 사실적 의사소통을 어떻게 구별하는지를 분석했습니다. COVID-19와 Monkeypox 관련 데이터셋을 통해 허위 정보가 가지는 독특한 언어적 특징을 밝혀냈습니다. 이를 통해 시의적절한 공중 보건 메시지를 전달하기 위한 전략에도 기여할 수 있다는 점이 주목할 만합니다.

- **Technical Details**: 연구는 COVID-19 허위 내러티브, 일반 COVID-19 콘텐츠, Monkeypox 관련 게시물의 세 가지 코퍼스를 분석했습니다. 분석 결과, COVID-19 허위 정보는 낮은 readability 점수와 함께 두 배 이상의 두려움 관련 또는 설득 용어를 포함하고 있으며, Monkeypox 콘텐츠보다 감정적으로 더 표현되는 스타일과 대조적입니다. 이로 인해 허위 정보는 복잡한 수사적 스타일과 감정적 신호를 사용하며, 이러한 조합이 신뢰도를 높일 수 있음을 시사합니다.

- **Performance Highlights**: 연구 결과는 디지털 건강 허위 정보에 대한 언어적 지표를 강조하면서 탐지 노력을 지원하는 데 기여하고 있습니다. 또한, 네트워크 미디어 환경에서의 위기 소통 이론 모델과 공중 보건 메시징 전략에도 중요한 정보를 제공합니다. 그러나 전통적 readability 지수에 의존하거나 정적 집계 분석을 사용하는 등의 한계가 있으므로, 향후 연구에서는 더 넓은 감정 어휘와 플랫폼에 민감한 접근 방식을 통합해야 할 필요성이 제기됩니다.



### Benchmarking is Broken -- Don't Let AI be its Own Judg (https://arxiv.org/abs/2510.07575)
Comments:
          12 pages; Accepted to NeurIPS 2025. Link to poster: this https URL

- **What's New**: AI의 시장에서 급속도로 증가하는 수요는 신뢰할 수 있는 평가를 위한 통합적 기준의 필요성을 드러냅니다. 현재의 벤치마크 시스템은 데이터 오염이나 선택적 보고와 같은 취약점으로 인해 신뢰성을 잃고 있습니다. 많은 기업들이 벤치마크에서 최고 점수를 얻기 위해 상당한 자원을 투자하고 있지만, 이는 종종 진정한 이해를 기반으로 하지 않은 과대 포장된 결과를 초래합니다.

- **Technical Details**: 상대적으로 오래된 벤치마크 데이터세트는 그 효과성을 저하시켜 AI 모델의 편향된 성과 평가로 이어질 수 있습니다. 데이터 오염이 의심되는 경우, 모델의 일반화 능력에 대한 주장은 물음표를 던지게 됩니다. 또한, 현재의 벤치마크는 일관성을 결여하고 있어, 상이한 토큰화 방식과 점수 기준이 혼재하게 됩니다.

- **Performance Highlights**: PeerBench라는 새로운 커뮤니티 관리 평가 시스템이 제안되었으며, 이는 신뢰성을 보장하기 위해 엄격한 프로토콜을 따릅니다. 이 시스템은 밀봉된 실행, 항목 은행의 순환 갱신, 지연된 투명성을 통한 체계적 접근 방식을 통해 AI의 진정한 발전을 측정할 수 있는 길을 열어줍니다. 혁신적이고 신뢰할 수 있는 성과 평가를 통해 AI 커뮤니티의 신뢰를 회복하는 것이 목표입니다.



### Label Semantics for Robust Hyperspectral Image Classification (https://arxiv.org/abs/2510.07556)
Comments:
          This work has been accepted for publication in the proceedings of IJCNN 2025

- **What's New**: 이 논문에서는 Semantically Guided Semantic Spectral-Spatial Fusion Network(S3FN)를 제안하여 고차원 HSI(초분광 이미지) 데이터의 분류 성능을 향상시키고자 합니다. 이 모델은 각 클래스 레이블에 대한 텍스트 설명을 생성하여, 기존의 단일 모드 방법의 한계를 극복하고 보다 의미 있는 피쳐-레이블 정렬을 가능하게 합니다. 이를 통해 과거 모델들이 놓치곤 했던 세부적인 스펙트럼 정보와 의미적 관계를 효과적으로 활용합니다.

- **Technical Details**: S3FN은 LLM(대형 언어 모델)을 활용하여 각 클래스의 고유한 특성과 스펙트럼 행동을 반영한 포괄적인 텍스트 설명을 생성합니다. 이 설명들은 BERT나 RoBERTa와 같은 사전 훈련된 텍스트 인코더를 통해 벡터 공간에 임베딩되어, HSI 데이터의 특성 표현을 강화합니다. LLM을 사용함으로써, 수동으로 작성된 불완전하거나 모호한 설명 문제를 해결하고, 다양한 클래스 간의 스펙트럴 관계를 더 깊이 이해할 수 있게 합니다.

- **Performance Highlights**: 논문에서 제안하는 S3FN은 Hyperspectral Wood, Hyperspectral Blueberries, DeepHS-Fruit의 세 가지 HSI 벤치마크 데이터셋에서 평가되었으며, 기존 방법에 비해 성능이 유의미하게 향상되었습니다. 텍스트 기반 의미적 정보와 스펙트럴-공간적 데이터를 융합한 결과, 보다 정교한 분류 성능을 달성하게 되었으며, 이는 HSI 분류의 발전 가능성을 보여줍니다.



### Deploying Tiny LVLM Judges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices (https://arxiv.org/abs/2510.07545)
Comments:
          Accepted to the EMNLP 2025 Industry Track

- **What's New**: 이 논문에서는 7억 개의 매개변수를 가진 대형 비전-언어 모델(LVLMs)이 차트 이해 작업에서 자동 심사관 역할을 수행할 수 있음을 보여주지만, 20억 개 이하의 모델들은 성능이 저조하다는 문제를 다룹니다. 이를 해결하기 위해 두 가지 기술적 접근법을 제안하며, 하나는 여러 평가 기준을 통합하여 단일 쿼리로 처리하는 다중 기준 프롬프트(multi-criteria prompting)이고, 다른 하나는 도메인 적응 전이 학습(domain-adaptive transfer learning)입니다. 이 방식들을 통해 ChartJudge라는 경량 평가 모델을 개발해 효율적인 평가를 가능하게 합니다.

- **Technical Details**: 이 연구에서는 ChartJudge-2B라는 20억 개 매개변수 모델을 사용하여 차트 관련 작업에 대한 평가는 미세 조정된 합성 판단 데이터에 기초하고 있습니다. 연구진은 다중 기준 프롬프트 접근법을 통해 평가 비용과 지연 시간을 줄이고, 이 과정을 통해 평가 성능에 있어 강력한 전이 능력을 보여주는 방식을 기술로 제시합니다. 또한, 다양한 차트 유형 및 쿼리 복잡성에 대한 정밀 분석을 통해 모델 크기와 프롬프트 설계 간의 trade-off를 설명합니다.

- **Performance Highlights**: 실험 결과 다중 기준 프롬프트 방식이 기존 7억 개 모델에서의 성능 저하의 주요 원인임을 밝혀냈고, 20억 개 파라미터를 가진 ChartJudge 모델은 효율적이면서도 강력한 성능을 발휘합니다. 특히 ChartJudge-2B는 다양한 차트 데이터셋 간의 지식 전이가 용이하다는 점에서 비용 효율적인 평가자로서의 가능성을 보여줍니다. 마지막으로, 이 연구는 LVLM 감정 모델의 확장이 가능하도록 하는 실제적인 통찰을 제공하며, 코드와 데이터는 공개될 예정입니다.



### Beyond independent component analysis: identifiability and algorithms (https://arxiv.org/abs/2510.07525)
Comments:
          30 pages, 8 figures

- **What's New**: 이번 논문은 독립 구성 요소 분석(ICA)의 한계를 넘어서는 새로운 모델을 탐구합니다. 특히, 쌍별 평균 독립(pairwise mean independence)이라는 개념을 도입하여 독립성을 완화하면서도 식별 가능성을 유지하는 방법을 제시합니다. 이를 통해 기존 연구에서 다뤘던 모델들을 포함한 새로운 식별 가능성의 범위를 제공합니다.

- **Technical Details**: 제안된 방법은 최소 제곱 최적화(least-squares optimization)를 활용한 대수적 복구 알고리즘을 기반으로 하고, 적절한 영 패턴(zero pattern)을 가진 분포에 적용됩니다. 이러한 모델에서는 쌍별 평균 독립성을 이용하여 독립적인 구조를 유지하면서도 더 안정적인 복구를 수행할 수 있습니다. 본 논문에서 제시된 아이디어는 고차 누적(moment/cumulant) 텐서를 통해 식별성을 연구하는 데 연관되어 있습니다.

- **Performance Highlights**: 실험 결과, 전체 독립성을 강요하는 경우 추정치에 악영향을 미칠 수 있음을 보여줍니다. 반면, 쌍별 평균 독립성을 적용하면 더욱 안정적인 결과를 도출할 수 있음을 발견했습니다. 이 연구는 기존 ICA 프레임워크를 확장하며 독립성을 넘어서는 블라인드 소스 분리에 대한 이론적 기초를 제공합니다.



### Time-Frequency Filtering Meets Graph Clustering (https://arxiv.org/abs/2510.07503)
- **What's New**: 이 논문은 시간-주파수 (time-frequency) 표현에서 다양한 신호 구성 요소를 식별하는 문제를 그래프 클러스터링 (graph clustering) 문제로 재구성할 수 있음을 보여줍니다. 저자들은 BFS (Breadth-First Search) 기반 알고리즘을 통해 복잡한 신호 구성 요소를 효과적으로 식별할 수 있는 새로운 방법론을 제안합니다. 이 접근법은 비선형 신호를 분석하는 데 매우 유용하며, 기존의 방법들과 비교할 때 여러 가지 장점을 가지고 있습니다.

- **Technical Details**: 저자들은 신호 구성 요소의 시간-주파수 표현을 그래프 G=(V,E)로 모델링하며, 각 점은 신호의 픽셀에 해당합니다. 이 그래프의 정점 (vertex)은 신호의 시간-주파수 정보가 높고 근접해 있는 두 점, 즉 신호가 유사한 주파수 정보를 가지는 두 픽셀 간의 연결로 구성됩니다. 주요 알고리즘에서는 각 연결 요소를 구분하여 서로 다른 TF (time-frequency) 도메인으로 나누며, 그래프 클러스터링 알고리즘을 적용하여 구성 요소를 식별합니다.

- **Performance Highlights**: 숫자 실험 결과, 제안된 방법은 기존의 베이시안 방법론보다 우수한 성능을 보여줍니다. 특히, 그래프 클러스터링 방식이 복잡한 그래프를 잘 처리할 수 있어 오류 누적 문제를 피할 수 있으며, 비정상 신호와 같은 다양한 신호 모형에서 효과성을 발휘합니다. 이 접근법은 또한 AM-FM (Amplitude Modulated - Frequency Modulated) 진동 또는 급격한 과도기 변수를 분석하는 데 유리함을 입증하였습니다.



### Evaluating and Learning Optimal Dynamic Treatment Regimes under Truncation by Death (https://arxiv.org/abs/2510.07501)
Comments:
          30 pages, 5 figures, 6 tables, The Thirty-Ninth Annual Conference on Neural Information Processing Systems

- **What's New**: 이 논문은 중환자 치료에서의 사망에 의한 잘림(truncation by death) 문제를 해결하기 위한 주설치(principal stratification) 기반의 새로운 방법론을 제시합니다. 특히, 항상 생존하는 집단(always-survivor) 가치 함수(value function)에 집중하여 다단계 동적 치료 체계(dynamic treatment regimes, DTR)의 세미파라메트릭(semi-parametric) 추정기를 도출하였습니다. 이 방법은 개인화된 치료 최적화에 유용하며, 전자 건강 기록(electronic health records) 데이터에의 적용을 통해 그 유효성을 입증하였습니다.

- **Technical Details**: 중환자 치료에서 발생하는 사망에 의한 잘림은 전통적인 DTR 평가를 어렵게 만듭니다. 저자들은 항상 생존하는 집단을 식별하기 위해 주설치 방법론을 적용하여, 사망의 영향을 받지 않는 상황에서 가치 함수를 정의하고 식별할 수 있도록 하였습니다. 이 연구는 여러 의사 결정 포인트를 고려한 다단계 DTR에 대한 항상 생존하는 가치 함수 추정 방법을 제안하며, 세미파라메트릭 효율성과 다중 안정성(multiply robust) 특성을 강조합니다.

- **Performance Highlights**: 제안된 추정기는 다양한 모형 사양시나리오에서 견고함을 보이며, 특히 고위험 환자 집단에 대한 의사결정 지원에 기여할 수 있음을 보여주었습니다. MIMIC-III 데이터베이스를 활용하여, 이 추정기가 생존 관련 가치 함수의 극대화뿐만 아니라 환자의 삶의 질을 향상시키는 데도 유용하다는 것을 입증하였습니다. 최종적으로, 이 연구는 전통적인 치료 효과 추정 방법이 처치 후 생존 여부에 강하게 의존하지 않도록 설계된 점에서 의미가 있습니다.



### When Thoughts Meet Facts: Reusable Reasoning for Long-Context LMs (https://arxiv.org/abs/2510.07499)
- **What's New**: 이 논문에서는 지식 집약적 멀티 홉 추론 작업을 위해 Thought Template Augmented LCLMs(ToTAL)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다양한 문서에서 증거를 수집하고 이를 조직적으로 연결하는 데 도움을 주는 재사용 가능한 사고 템플릿을 활용합니다. LCLM(Long-Context Language Models)의 발전으로 수백만 개의 토큰을 한 번에 처리할 수 있게 되었지만, 여전히 증거를 연결하는 방식에 대한 접근이 부족한 상황입니다.

- **Technical Details**: ToTAL은 기존의 RAG(Retrieval-Augmented Generation) 방식과는 달리, 사고 템플릿을 외부 매개변수로 간주하고 자연어 피드백을 통해 반복적으로 수정합니다. 이러한 템플릿은 이전 문제 해결의 패턴을 포괄하여 새로운 쿼리에 대한 중간 단계를 구조화할 수 있게 돕습니다. 이 과정은 특정 문제에 국한되지 않고 다양한 쿼리에서 재사용될 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: 다양한 벤치마크와 LCLM 가족을 통해 ToTAL은 강력한 기준선 대비 일관된 성능 향상을 보여주었습니다. 특히, 정보 검색이 포함된 상황과 포함되지 않은 상황 모두에서 사고 템플릿이 LCLM 성능을 지속적으로 강화하는 것으로 나타났습니다. 이를 통해 LCLM이 복잡한 증거를 처리할 수 있는 새로운 가능성을 제시하며, 투명한 추론 재사용을 통한 광범위한 적용성을 보여줍니다.



### Evaluation of LLMs for Process Model Analysis and Optimization (https://arxiv.org/abs/2510.07489)
Comments:
          15 pages, 5 tables, 4 figures; full research paper currently under review for the Workshop on Information Technologies and Systems (WITS) 2025. The paper presents a comprehensive evaluation of large language models (LLMs) for business process model analysis and optimization, including error detection, reasoning, and scenario-based redesign

- **What's New**: 이번 논문에서는 여러 LLM(대형 언어 모델)의 경험을 공유합니다. 이 모델들은 대화형 스타일로 프로세스 모델을 이해하고, 문법적(syntactical) 및 논리적(logical) 오류를 찾고, 이를 자연어(NL) 인터페이스를 통해 깊이 있게 추론하는 능력을 평가하였습니다.

- **Technical Details**: 연구 결과에 따르면, 훈련되지 않은 기본 LLM인 ChatGPT(모델 o3)는 제로샷(zero-shot) 환경에서 BPMN 프로세스 모델을 이미지로부터 효과적으로 이해하고, 그에 대한 질문에 문법적, 논리적, 의미적(semantic) 수준에서 지능적으로 답변할 수 있음을 보여주었습니다. 다양한 LLM들은 정확성(accuracy)과 효과성(effectiveness) 면에서 성능 차이를 보입니다.

- **Performance Highlights**: 경험적 분석(emirical analysis) 결과, LLM이 비즈니스 프로세스 설계자와 사용자에게 유용한 조력자 역할을 수행할 수 있음을 확인하였습니다. 또한 프로세스 분석 및 최적화(context of process analysis and optimization)에서 LLM의 '사고 과정'(thought process)과 더 깊은 추론 능력을 연구하였고, LLM들이 인격적(anthropomorphic) 특성을 나타내는 경향이 있음을 발견하였습니다.



### Comparison of Fully Homomorphic Encryption and Garbled Circuit Techniques in Privacy-Preserving Machine Learning Inferenc (https://arxiv.org/abs/2510.07457)
Comments:
          8 pages, 9 figures, 2 tables, 32 references

- **What's New**: 이번 연구는 Privacy-preserving Machine Learning (PPML) 분야에서 Fully Homomorphic Encryption (FHE)와 Garbled Circuits (GC) 기술의 성능과 보안 측면을 비교 평가합니다. 특히, 두 개의 층으로 구성된 신경망에서 이 두 가지 접근법을 활용하여 개인 정보와 모델 기밀성을 유지하면서도 머신러닝 추론을 수행하는 방법론에 중점을 두었습니다. 이 연구는 데이터 소유자(Client)와 모델 소유자(Server) 간의 정보 보호 목표의 갈등을 해결하는 데 중점을 둡니다.

- **Technical Details**: FHE는 비대화형(intractive) 추론을 지원하며, 낮은 메모리 소모로 빠른 평가를 가능하게 하는 GC와는 대조적입니다. 이 논문에서는 CKKS 스킴을 사용하는 Microsoft SEAL 라이브러리와 TinyGarble2.0 프레임워크를 통해 각각 FHE와 GC를 구현하였습니다. 연구에서는 각 방식의 성능, 메모리 사용량 및 통신 오버헤드를 평가하며, 반투명(threat model) 이론에 기반한 실험 결과를 제시합니다.

- **Performance Highlights**: 실험 결과에 따르면, GC 방식은 더 빠른 실행 시간을 제공하며 메모리 소비가 적지만, FHE는 비대화형 추론을 통해 모델 운영의 기밀성을 유지합니다. 이 두 방식은 각각 고유한 장단점이 있으므로, 특정 PPML 시나리오에 적합한 방법을 선택하는 데 있어 중요한 요소가 됩니다. 연구의 결과는 FHE와 GC의 성능 간의 트레이드오프를 명확히 설명하며, 연구자와 개발자들이 적합한 기술을 선택할 수 있도록 안내합니다.



### VeMo: A Lightweight Data-Driven Approach to Model Vehicle Dynamics (https://arxiv.org/abs/2510.07447)
- **What's New**: 이 논문은 차량 다이나믹 모델을 개발하는 데 필요한 복잡한 문제를 다루고 있으며, 차량의 미래 상태를 과거 상태 및 운전자의 제어 행동과 연결하는 경량화된 encoder-decoder 모델을 제안합니다. 이 모델은 Gate Recurrent Unit (GRU) 레이어 기반으로 설계되었으며, 극한의 동적 조건에서도 평균 상대 오차가 2.6% 이하로 달성되는 뛰어난 성능을 보여줍니다. 또한, 데이터 기반 접근 방식을 통해 물리적 제약 없이 출력 신호의 물리적 일관성을 유지하는 특징을 갖습니다.

- **Technical Details**: Vehicle Modeller (VeMo)는 공유된 인코더와 각 출력에 대한 전용 디코더를 사용하는 GRU 레이어 기반의 신경망 모델로, 차량의 상태 변화를 예측하는 데 사용됩니다. 모델은 경량화되어 있어 거의 실시간으로 차량의 동적 상태를 예측할 수 있습니다. 측정된 과거 상태와 운전자의 제어 행동을 기반으로 상황을 분석하며, 이 과정에서 다양한 차량 플랫폼에 적용 가능함을 강조합니다.

- **Performance Highlights**: VeMo 모델은 다양한 주파수 구성의 고주파 필터링 하에서도 우수한 정확도를 보이며, 데이터 필터링 주파수와 훈련 주파수의 일관성을 통해 모델의 성능을 높일 수 있다는 점이 중요한 발견입니다. 특히, 노이즈에 대한 강건성을 갖추고 있으며, 안전-critical 어플리케이션에서의 적용 가능성을 시사합니다. 실험 결과는 VeMo 모델이 상대적 오차와 물리적 일관성에서 뛰어난 성능을 발휘함을 명확히 보여줍니다.



### LASER: An LLM-based ASR Scoring and Evaluation Rubric (https://arxiv.org/abs/2510.07437)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 본 연구에서는 기존의 자동 음성 인식(ASR) 평가 지표인 Word Error Rate(WER)의 한계를 극복하기 위한 혁신적인 방법으로 LASER라는 LLM 기반의 스코어링 루브릭을 소개합니다. LASER는 ASR 시스템의 오류를 보다 정교하게 평가할 수 있는 점수 매기기 방법론으로, 인도 언어의 형태적 및 구문적 특징을 잘 반영합니다. 특히, 이 시스템은 힌디어에서 사람의 주석과 94%의 높은 상관관계를 달성하여 다른 인도 언어에서도 효과적으로 적용이 가능함을 증명하였습니다.

- **Technical Details**: LASER는 표준 ASR 지표에서 발생하는 오류 유형을 분석하여, 수정된 패널티를 부여하는 방법을 제안합니다. minor grammatical errors에는 낮은 패널티를 부여하고, semantic errors에는 높은 패널티를 부여하는 방식입니다. LASER 점수는 맞춤형 프롬프트를 통해 구체적인 지시와 오류 유형을 설정해 LLM에게 처리하게 함으로써 생성됩니다. 이 과정에서 LLM은 주어진 문장을 토큰화하고, 기준 참조와 예측을 정렬하여 오류를 분류합니다.

- **Performance Highlights**: Gemini 2.5 Pro는 모든 LLM 중에서 뛰어난 성능을 보여주며, 사람의 점수와의 상관관계가 가장 높았습니다. 연구 결과, LASER 지수는 전통적인 WER과 비교하여 상대적으로 높은 정확성을 보였고, ASR 예측의 정확성을 판단하는데 있어 훨씬 더 신뢰할 수 있는 방법임을 입증했습니다. 실험을 통해 LASER의 점수가 사람의 판단과 잘 일치함을 확인하면서, 다양한 언어에서의 적용 가능성을 보여주었습니다.



### Bayesian Optimization of Multi-Bit Pulse Encoding in In2O3/Al2O3 Thin-film Transistors for Temporal Data Processing (https://arxiv.org/abs/2510.07421)
- **What's New**: 이 연구는 하드웨어의 내재적 역사 의존성과 비선형성을 활용하여 물리적 저수지 컴퓨팅(physical reservoir computing)을 사용한 새로운 접근법을 제시합니다. 특히, Bayesian optimization(베이지안 최적화)을 통해 Al2O3/In2O3 박막 트랜지스터(TFT)의 인코딩 충실도를 향상시키는 방법을 보여줍니다. 이는 다중 상태 출력을 구분하는 능력에 크게 의존하며, 그 개선에 대한 체계적인 방법론을 제공합니다.

- **Technical Details**: 실험에서는 6비트 시간적 인코딩을 달성하기 위해 다섯 가지 주요 펄스 파라미터를 탐색하고, 출력을 구분하는 데 사용되는 척도로 normalized degree of separation (nDoS)을 활용했습니다. 또한, 4비트 데이터에 대한 모델 학습을 통해 복잡한 6비트 인코딩 작업의 최적화를 효과적으로 안내할 수 있음을 보였습니다. 이러한 방법은 실험 비용을 낮추는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 6개의 연속 프레임에서 움직이는 자동차의 이진 패턴 이미지를 인코딩하고 재구성하는 과정에서, 최적화된 펄스 파라미터를 사용했을 때 인코딩 정확도가 향상됨을 확인했습니다. 특히, 4비트 최적화 운영 조건이 6비트 최적화 조건과 거의 동등한 성능을 보였습니다. SHAP(Shapley Additive Explanations)를 통한 해석 가능성 분석에서도 게이트 펄스 진폭과 드레인 전압이 높은 상태 분리를 달성하는 데 가장 영향력 있는 파라미터로 나타났습니다.



### Attention to Order: Transformers Discover Phase Transitions via Learnability (https://arxiv.org/abs/2510.07401)
- **What's New**: 이 논문에서는 집합 행동의 질적 재편성을 나타내는 Phase transitions을 다루며, 이러한 경계를 식별하는 데 있어 새로운 방법론으로 learnability(학습 가능성)를 도입합니다. 이는 Transformer 모델이 microscopic states(미세 상태)로부터 구조를 추출할 수 있는 능력으로 정의됩니다. 전통적인 시뮬레이션 방법이 실패하는 상황에서 learnability가 Phase transitions의 데이터를 기반으로 한 지표 역할을 한다는 점이 새롭습니다.

- **Technical Details**: 논문에서는 self-supervised learning(자기 감독 학습)과 Monte Carlo(몬테 카를로)로 생성된 2차원 Ising model(아이징 모델)의 구성을 사용합니다. 연구 결과에 따르면, ordered phases(정렬된 상)에서 learnability가 향상되는 것이 관찰되며, 이는 훈련 손실(training loss)의 감소와 구조화된 주의 패턴(structured attention patterns)으로 나타납니다. 반면, disordered phases(무질서 상)는 학습에 저항적입니다.

- **Performance Highlights**: 훈련 손실의 급격한 점프와 주의 엔트로피(attention entropy)의 상승이라는 두 개의 비지도 진단을 통해 임계 온도(critical temperature)를 매우 정확하게 복구했습니다. 이러한 성과는 응집 물질에서의 장기적 질서(long-range order)와 현대 언어 모델에서의 구조의 출현(appearance of structure)之间의 깊은 유사성을 강조합니다.



### Base Models Know How to Reason, Thinking Models Learn When (https://arxiv.org/abs/2510.07364)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 DeepSeek R1과 같은 사고 언어 모델(Thinking Language Models)이 왜 기본 모델(Base Models)보다 우수한 성능을 발휘하는지를 탐구했습니다. 하이브리드 모델을 제안하며, 기존의 기본 모델에서 사고 모델 수준의 추론 체인을 발생시키기 위해 적절한 시점에 추론 메커니즘을 활성화하는 방법을 보여줍니다. 이를 통해 사고 모델이 이미 존재하는 능력을 활용한다는 점을 시사합니다.

- **Technical Details**: 우리는 비지도 학습 기반의 방법론을 통해 사고 모델이 사용하는 인간 해석 가능한 추론 메커니즘의 분류체계를 구축합니다. 이 과정에서 Sparse Autoencoders (SAEs)를 사용하여 모델의 문장 수준의 활성화를 클러스터링하고, 각 문장이 각각 하나 또는 최대 세 가지의 추론 카테고리로 분류될 수 있다는 가정을 적용합니다. 이를 통해 발견된 기능은 핵심 인지 작용에 해당하며, 실험 결과 다양한 모델 아키텍처에서 감지된 최적의 차원 수는 15에서 25 카테고리 사이에 위치합니다.

- **Performance Highlights**: 하이브리드 모델을 사용한 결과, 기본 모델이 사고 모델과의 성능 격차를 최대 91%까지 회복할 수 있음이 확인되었습니다. 이는 가중치 업데이트 없이 이루어진 결과로, 단지 12%의 토큰만을 조정하여 성취된 것입니다. 이러한 연구 결과는 사고 모델의 훈련 방식에 대한 새로운 시각을 제공하며, 향후 언어 모델의 추론 훈련을 더 효율적으로 만드는 데 기여할 것입니다.



### Inconsistent Affective Reaction: Sentiment of Perception and Opinion in Urban Environments (https://arxiv.org/abs/2510.07359)
Comments:
          10 pages

- **What's New**: 이번 연구는 소셜 미디어 플랫폼의 등장으로 도시 환경에 대한 우리의 이해가 어떻게 변화했는지를 다룹니다. 특히, 인간의 인식과 의견에 내재된 감정 반응의 미세한 차이를 식별하고 분석하는 새로운 방법론을 제안합니다. 연구는 Baidu와 Tencent의 스트리트 뷰 이미지 140,750개와 Weibo의 소셜 미디어 텍스트 게시물 984,024개로 구성된 데이터셋을 활용합니다.

- **Technical Details**: 연구팀은 물체 탐지(object detection) 및 자연어 처리(natural language processing) 기법을 통합하여 2016년과 2022년의 베이징 제2환상도로에서 감정을 분류하는 반응 지수를 개발했습니다. 이 연구는 회귀 분석(regression analysis), 이미지 분할(image segmentation), 단어 빈도(word frequency) 기법을 통해 감정 반응을 시각화하고 분석합니다. 토지 이용 분포에 따른 기저 요인을 파악하는 것이 핵심입니다.

- **Performance Highlights**: 연구의 결과, 인식 반응 경향 맵은 긍정적인 감정이 보다 고르게 분포하고 있음을 보여주었으며, 의견 반응 경향 맵에서는 더 극단적인 변화가 관찰되었습니다. 또한, 감정 반응의 변화는 밀집된 건물과 보행자 존재와 같은 요소들과 유의미한 관계를 나타냈습니다. 팬데믹 이전과 이후의 인식 및 의견 감정을 비교한 불일치 맵은 환경 관리 및 도시 재생을 위한 전략 수립에 중요한 통찰을 제공합니다.



### Enhancing Maritime Object Detection in Real-Time with RT-DETR and Data Augmentation (https://arxiv.org/abs/2510.07346)
Comments:
          13 pages, 10 figures

- **What's New**: 이 논문은 RT-DETR 기반의 실시간 해양 물체 탐지 시스템을 제안하며, 합성 이미지를 활용한 데이터 증강을 통해 실제 데이터에서의 평가를 엄격히 수행하는 것이 특징입니다. 제안된 시스템은 다중 크기 특징 융합(multi-scale feature fusion), 불확실성 최소화 쿼리 선택(uncertainty-minimizing query selection), 합성 및 실제 샘플 간의 스마트 가중치 조정(smart weight) 기법을 통합하여 해양 환경에 맞춰진 방식입니다. 이 연구는 실제 조명 및 해양 조건에서도 성능을 유지하는 디지털 파이프라인을 제공하여 모든 건축 모듈의 기여도를 정량화하고, 시스템이 극한 조건에서 실패를 처리하는 방식도 설명합니다.

- **Technical Details**: RT-DETR은 완전한 엔드-투-엔드(attention-based) 탐지기로, 다중 크기 처리(multi-scale processing)와 불확실성 인식 쿼리 선택(uncertainty-aware query selection)으로 실시간 효율성을 확보합니다. 이 시스템은 최적의 속도와 정확성을 조절할 수 있으며, 각 아키텍처 모듈의 기여도를 평가하기 위한 구성 요소 분석(component analysis)을 포함하고 있습니다. 훈련 과정에서는 실제 해양 이미지와 GAN에서 생성된 합성 샘플을 결합하여 더 다양한 조명 및 날씨 조건을 반영하고, 특이한 조건을 모사하기 위해 비쌍 이미지 전송 기술도 활용되었습니다.

- **Performance Highlights**: 제안된 RT-DETR 시스템은 다양한 임계값을 통해 실제 이미지에서의 정확도를 유지하면서도 성능을 향상시키는 것을 목표로 합니다. 다양한 조명, 환경 변화에 강건한 성능을 보여주며, 실시간 성능을 유지하는 파이프라인으로 실용성을 강조합니다. RT-DETR은 다른 YOLO 모델들과 비교하여 속도와 정확성 모두에서 우수한 성능을 발휘하며, 경량화된 처리로 모델의 견고함과 유연성을 제공하여 해양 물체 탐지의 신뢰성 있는 기반을 마련합니다.



### Beyond Grid-Locked Voxels: Neural Response Functions for Continuous Brain Encoding (https://arxiv.org/abs/2510.07342)
- **What's New**: 이 연구에서는 Neural Response Function (NRF)이라는 새로운 프레임워크를 소개하여, 기존의 1D 벡터를 사용하는 접근 방식을 넘어 fMRI 활동을 해부학적 공간에서 연속 함수로 모델링합니다. NRF는 이미지와 공간 좌표(즉, MNI 공간 내의 (x, y, z))를 입력으로 받아 해당 위치에서의 신경 반응을 예측합니다. 이를 통해 기존의 모델들이 가지던 공간적 맥락 상실 및 주제 특정 보폭 그리드의 한계를 극복하고자 합니다.

- **Technical Details**: NRF는 뇌 반응을 예측하는 과정에서 해부학적 좌표를 조건으로 함으로써, fMRI 데이터의 3D 공간 구조를 통합합니다. 이 방식은 인접한 보폭 간의 정보를 공유하도록 하는데, 이는 인접 보폭들이 종종 기능적으로 상관관계가 있음을 반영합니다. NRF는 또한 여러 주제 간의 대응을 가능하게 하여, 사전 훈련된 모델을 새로운 주제에 쉽게 적응시킬 수 있는 장점이 있습니다.

- **Performance Highlights**: 실험 결과, NRF는 인트라 주제 인코딩 및 크로스 주제 적응에서 기존의 기준 모델들을 초월하는 성능을 보였으며, 높은 성능을 유지하면서도 필요한 데이터 크기를 획기적으로 줄였습니다. NRF는 단순히 해부학적으로 민감한 인코딩 모델로서, 이미지에서 3D 공간의 뇌 반응으로의 연속적 매핑을 학습하는 최초의 시도입니다.



### SpotDiff: Spotting and Disentangling Interference in Feature Space for Subject-Preserving Image Generation (https://arxiv.org/abs/2510.07340)
- **What's New**: 이번 연구에서는 SpotDiff라는 새로운 개인화 이미지 생성 방식을 제안합니다. SpotDiff는 CLIP 이미지 인코더와 전문가 네트워크를 활용하여 주제별 특성을 추출하고, 불필요한 요소를 제어합니다. 특히, SpotDiff10k라는 새로운 데이터셋을 만들어 포즈와 배경 변화를 일관성 있게 관리하여 훈련을 지원합니다.

- **Technical Details**: SpotDiff는 특성 공간에서 불필요한 요소를 분리하기 위해 두 가지 전문가 네트워크를 이용합니다. 이 네트워크들은 포즈 및 배경과 같은 불필요한 요소를 식별하고 이를 원래 특성과 직교시켜 불필요한 요소의 영향을 최소화합니다. 또한, SpotDiff는 훈련 과정에서 동일한 포즈를 공유하되 외관이 다른 10,000개의 이미지를 포함하는 SpotDiff10k 데이터셋을 구성합니다.

- **Performance Highlights**: 실험 결과, SpotDiff는 기존 방법들보다 주제 보존 및 결과 편집에서 더욱 강력한 성능을 보여주며, 제한된 수의 훈련 샘플(10,000개)로도 경쟁력 있는 성과를 달성했습니다. 이러한 결과는 SpotDiff가 주제별 이미지 생성 및 편집의 정밀성과 제어 가능성을 효과적으로 향상시킨 것을 보여줍니다.



### Decoding the dark proteome: Deep learning-enabled discovery of druggable enzymes in Wuchereria bancrof (https://arxiv.org/abs/2510.07337)
Comments:
          Accepted for peer-reviewed publication at the STEM Fellowship Journal

- **What's New**: 이 연구는 Wuchereria bancrofti의 어두운 단백질에 대한 최초의 대규모 기능적 지도(Map)를 제공하며, 이로 인해 약물 개발을 가속화할 수 있는 중요한 기초를 마련한 점이 새롭습니다. 연구팀은 이전에는 주목받지 못했던 14,772개의 단백질에 정확한 Enzyme Commission (EC) 번호를 할당하였습니다.

- **Technical Details**: 연구진은 DEtection TRansformer를 활용하여 효소적 기능의 확률을 추정하고, 4,476개의 레이블이 붙은 기생충 단백질에 대해 계층적 최근접 이웃(Nearest Neighbor) EC 예측기를 세밀하게 조정했습니다. 이를 통해 100%의 신뢰도로 4단계 EC 분류만을 유지하는 거부 샘플링(Rejection Sampling) 모델을 적용했습니다.

- **Performance Highlights**: 해당 연구를 통해 발견된 6개의 효소에 대해 약 1마이크로몰(micromolar) 이하의 결합 친화도(Binding Affinity)가 강하게 예측되었으며, 특히 peptidoglycan glycosyltransferase와 NTPase 억제제는 나노몰(nanomolar) 수준의 유망한 결합 결과를 보였습니다. 이 결과는 실험적 검증이 필요하지만, 향후 W. bancrofti에 대한 초기 단계의 치료 개발을 위한 중요한 발판을 제공합니다.



### Geodesics in the Deep Linear Network (https://arxiv.org/abs/2510.07324)
- **What's New**: 본 논문에서는 깊은 선형 네트워크(Deep Linear Network, DLN)에서 전계수(full rank) 행렬 간의 지오데식(geodesic)을 위한 ODE(Ordinary Differential Equations)의 일반적인 시스템과 관련된 명시적인 솔루션을 도출합니다. 특히, 논문은 리만 수면(Riemannian submersion) 하에서도 지오데식으로 유지되는 모든 수평 직선(horizontal straight lines)을 특성화합니다.

- **Technical Details**: 제안된 모델에서 모든 invertible, symmetric, positive definite, orthogonal d×d 행렬들의 집합(𝕄d, 𝔐d, Symmd, ℙd, Od)을 정의하고, DLN의 훈련 역학을 분석합니다. DLN의 파라미터 공간과 관측 공간 간의 매핑을 통해 overparameterization의 효과를 연구하며, 파라미터 공간 내 기울기 흐름을 통해 DLN의 기하학적 구조를 도출합니다.

- **Performance Highlights**: 이 논문은 DLN의 여러 매트릭(metric)을 사용하여 훈련 역학을 리만한 기울기 흐름으로 표현할 수 있음을 보여줍니다. 또한 지오데식의 명시적인 솔루션을 유도하고, 이러한 결과가 DLN 모델의 기하학적 관점을 더욱 강화시킨다는 점에서 중요한 기여를 합니다.



### (Token-Level) InfoRMIA: Stronger Membership Inference and Memorization Assessment for LLMs (https://arxiv.org/abs/2510.05582)
- **What's New**: 이번 논문에서는 기존의 Robust Membership Inference Attack (RMIA)을 개선한 InfoRMIA를 소개하며, 이 방법이 기존의 RMIA보다 더 높은 성능과 효율성을 제공함을 보여줍니다. 특히, token 수준에서 정보 유출을 정량화할 수 있는 새로운 접근법을 제안하여, LLMs(large language models)에서의 개인 정보 보호 문제에 대한 인식을 새롭게 하고 있습니다. 이 연구는 머신 러닝 모델이 학습 데이터를 기억하는 방식에 대한 이론적 기초를 다지고 있습니다.

- **Technical Details**: InfoRMIA는 정보 이론에 기반하여 membership inference 문제를 해결하기 위한 새로운 방법으로, 기존 RMIA의 한계를 극복하고 더 낮은 비용으로 강력한 성능을 자랑합니다. 또한, token-level signals를 기반으로 LLM의 기억화 및 정보 유출을 보다 세밀하게 분석할 수 있는 체계를 제안하였으며, token 단위에서 비밀 정보를 검출하여 안전성을 높일 수 있는 방법을 모색하고 있습니다. 이러한 접근은 비밀 정보의 직접적인 분석을 가능하게 하여, 필요한 경우 특정 정보만을 제거할 수 있는 기반을 제공합니다.

- **Performance Highlights**: InfoRMIA는 다양한 데이터셋에서 RMIA보다 일관되게 더 높은 성능을 보이며, 대상 모델의 기억화 평가를 더욱 정밀하게 진행할 수 있습니다. 기존의 membership inference는 시퀀스 전체에 대한 단일 라벨링을 기반으로 한 반면, InfoRMIA는 각 token에 대한 분석을 통해 보다 의미 있는 개인 정보 보호 평가와 더 나은 메모리 관리 방안을 제시합니다. 이로 인해 InfoRMIA는 머신 러닝 커뮤니티에서 중요한 개인 정보 보호 수단으로 자리 잡을 가능성이 높습니다.




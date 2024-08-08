New uploads on arXiv(cs.CL)

### Self-Taught Evaluators (https://arxiv.org/abs/2408.02666)
- **What's New**: 새로운 접근법으로 인간 주석 없이 평가자를 개선할 수 있는 방법을 제안합니다. 이 연구는 합성된(training data) 데이터만을 사용해서 평가자가 자체 개선(iterative self-improvement)할 수 있도록 설계되었습니다.

- **Technical Details**: 언노티드(unlabeled) 명령으로 시작하여, 모델이 생성한 대조적인 출력과 학습 데이터를 통해 평가자를 훈련시킵니다. 이 과정에서 LLM-as-a-Judge(판사 역할의 대형 언어 모델)을 사용하여 추론 과정(reasoning traces)과 최종 판단을 생성합니다. 반복적인 훈련을 통해 예측을 개선합니다.

- **Performance Highlights**: 라벨링된 선호 데이터 없이도 'Self-Taught Evaluator'는 강력한 LLM(Llama-3-70B-Instruct)의 성능을 75.4에서 88.3으로, 대부분 투표에서는 88.7로 향상시켰습니다. 이는 GPT-4 같은 기존의 LLM 평가자들을 능가하고, 라벨링된 예제로 훈련된 최고 성능의 보상 모델과 맞먹는 수준입니다.



### Can Reinforcement Learning Unlock the Hidden Dangers in Aligned Large Language Models? (https://arxiv.org/abs/2408.02651)
Comments:
          Accepted to AI4CYBER - KDD 2024

- **What's New**: 이번 연구는 강화학습(나선형 학습)을 사용하여 필립 노락 오류(adversarial triggers)를 최적화하는 새롭고 혁신적인 방법을 제안합니다. 이 방법은 타겟 모델에 대한 API 추론 액세스 및 소규모 대체 모델만 필요로 합니다. 주로 BERTScore 기반(reward function)을 활용하여 타겟 모델의 텍스트 출력에서 보상 신호를 계산합니다.

- **Technical Details**: 기존의 Jailbreaking 접근법은 블랙박스 모델에 대해 제한적이었습니다. 하지만, 이번 논문에서는 강화학습을 활용한 새로운 패러다임을 통해 필립 노락 오류(adversarial triggers)를 최적화합니다. 이 접근법은 타겟 언어 모델에 대한 추론 API 접근만을 필요로 하며, 소규모 대체 모델을 사용하여 타겟 모델의 텍스트 출력에서 보상 신호를 계산합니다. 이러한 방법은 기존의 백박스 모델에서의 필립 노락 오류를 새로운 블랙박스 모델에 적용하는 데 효과적입니다.

- **Performance Highlights**: 이 논문에서 제안된 방법은 새로운 블랙박스 대상 모델에서 필립 노락 오류(adversarial triggers)의 성능을 크게 향상시킵니다. 강화학습 기반의 접근법이 모델 간의 필립 노락 오류 전파성과 효과를 크게 높여주는 것을 입증합니다.



### SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models (https://arxiv.org/abs/2408.02632)
- **What's New**: 대규모 언어 모델(LLM)의 보안성과 유해한 출력 방지는 중요한 과제로 떠오르고 있습니다. 이를 해결하기 위한 유망한 접근법으로 모델이 자체적으로 적대적 명령어(adversarial prompts)를 생성하여 red teaming에 활용하는 방법이 제시되었습니다. 하지만 LLM의 보안 취약점이 더 미묘해짐에 따라 기존의 적대적 방법들이 효과를 발휘하기 어려워지고 있습니다. 이에 따라 새로운 최적화 프레임워크인 SEAS(Self-Evolving Adversarial Safety) 프레임워크가 소개되었습니다. 이 프레임워크는 모델이 생성한 데이터를 활용하여 보안성을 향상시킵니다. SEAS는 초기화, 공격, 적대적 최적화의 세 가지 단계로 구성되어 있으며, 반복적으로 Red Team과 Target 모델을 개선하여 견고성과 안전성을 높이는 것을 목표로 합니다.

- **Technical Details**: SEAS 프레임워크는 세 단계로 이루어져 있습니다. 초기화 단계에서는 Red Team 모델(R0)과 Target 모델(T0)을 SEAS 데이터셋을 이용하여 각각 미세 조정합니다. 공격 단계에서는 Red Team 모델이 적대적 명령어를 생성하면, 이를 Target 모델에 입력하여 응답을 유도합니다. Safe Classifier는 이 응답의 안전성을 평가합니다. 적대적 최적화 단계에서는 효과적이었던 명령어와 응답을 기반으로 Red Team과 Target 모델을 업데이트합니다. 이러한 과정을 여러 번 반복하여 두 모델이 점차 적응하고 향상되도록 합니다.

- **Performance Highlights**: 세 번의 반복 후, Target 모델은 GPT-4와 비슷한 수준의 보안성을 달성하였으며, Red Team 모델은 Llama3-70B에 대한 공격 성공률(ASR)이 50.66% 증가하였습니다. 또한, 생성된 적대적 명령어의 다양성과 반복적인 모델 업데이트의 효과를 입증하였습니다.



### Language Model Can Listen While Speaking (https://arxiv.org/abs/2408.02622)
Comments:
          Demo can be found at this https URL

- **What's New**: 최신 대화형 음성언어 모델 (iSLM)에 전이중 모델링 (Full Duplex Modeling, FDM)을 도입하여 실시간 상호작용을 가능하게 합니다. 이는 '경청하며 말하는 모델 (Listening-while-Speaking Language Model, LSLM)'이라는 혁신적인 모델을 제안합니다.

- **Technical Details**: LSLM은 음성 생성에 토큰 기반 디코더 전용 Text-to-Speech (TTS)와 실시간 오디오 입력에 스트리밍 자가 지도 학습 (Streaming Self-Supervised Learning, SSL) 인코더를 사용합니다. 이 모델은 실시간으로 말하기와 듣기 기능을 융합하고, 세 가지 융합 전략(초기 융합, 중간 융합, 후기 융합) 중 중간 융합이 최적의 성능을 보여줍니다. 실험 설정은 커맨드 기반 FDM과 음성 기반 FDM을 포함합니다.

- **Performance Highlights**: LSLM은 높은 소음 견딤성과 다양한 지시에 대한 민감도를 보이며, 기존 시스템에 거의 영향을 주지 않으면서 전이중 통신을 달성할 수 있음을 입증했습니다. 특히, 중간 융합 전략이 음성 생성과 실시간 상호작용 능력의 균형을 잘 맞추어 최적의 성능을 발휘했습니다.



### BioMamba: A Pre-trained Biomedical Language Representation Model Leveraging Mamba (https://arxiv.org/abs/2408.02600)
- **What's New**: BioMamba는 생물 의학 텍스트 마이닝에 특화된 사전 훈련 모델로, 기존의 BERT 및 GPT 모델과 비교하여 훨씬 높은 성능을 자랑합니다. 특히 BioMamba는 PubMed 문헌에서 추출한 방대한 데이터 셋을 사용하여 생물 의학 텍스트에 특화되도록 미세 조정되었습니다.

- **Technical Details**: BioMamba는 기존 'Mamba' 아키텍처를 기반으로 하며, 구조화된 상태 공간 모델 (SSMs)을 활용하여 긴 시퀀스를 효율적으로 처리합니다. 이는 시퀀스 길이에 따라 선형 복잡도를 제공하여 더 효율적인 모델링을 가능하게 합니다. 여기에서 '퍼플렉시티 (perplexity)'와 '크로스 엔트로피 손실 (cross-entropy loss)'이 크게 감소한 것을 확인할 수 있습니다. BioMamba는 PubMed 추출 본문을 통해 사전 훈련되었으며, 생물 의학 질의응답과 같은 특정 임무에 대해 미세 조정되었습니다.

- **Performance Highlights**: BioMamba는 BioBERT 및 일반 도메인 Mamba보다 다양한 생물 의학 작업에서 월등한 성능을 보였습니다. BioASQ 테스트 셋에서 BioMamba는 퍼플렉시티가 100배 감소하고, 크로스 엔트로피 손실이 4배 감소하는 뛰어난 성능을 입증했습니다. 또한, BioMamba는 연구를 촉진하기 위해 Hugging Face에 코드를 공개하고 훈련된 모델을 배포하고 있습니다.



### Progressively Selective Label Enhancement for Language Model Alignmen (https://arxiv.org/abs/2408.02599)
- **What's New**: 최근 인공지능 언어 모델이 강력한 기능을 보여주고 있지만, 인간의 기대와 어긋나는 콘텐츠를 생성할 가능성이 있어 윤리 및 법적 문제를 일으킬 수 있습니다. 이러한 문제를 해결하기 위해 PSLE(Progressively Selective Label Enhancement)이라는 프레임워크를 제안합니다. 이는 생성된 데이터를 효율적으로 활용하여 모델의 출력을 인간의 기대에 맞추도록 유도합니다.

- **Technical Details**: PSLE 프레임워크는 생성된 모든 데이터를 통해 언어 모델을 학습시키는 방식을 사용합니다. 이는 동적으로 업데이트된 임계값을 이용해 원본 입력의 응답과 원칙에 따라 안내된 출력의 보상 점수 차이를 기반으로 합니다. 두 응답이 유사한 품질을 나타내면, 둘 다 모델 훈련에 포함되며 보상 점수를 기준으로 가중치를 할당합니다. 이를 통해 데이터 활용 효율성을 높이고, 전체 훈련 효율성을 개선합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, PSLE는 기존의 언어 모델 정렬 방법들에 비해 뛰어난 효과를 입증했습니다. 저자의 기여는 데이터 활용 효율성을 높인 새로운 접근법을 제안한 것입니다. 또한, 점진적 임계값 전략을 통해 최적의 언어 모델로 수렴할 수 있음을 이론적으로 증명했습니다.



### Leveraging the Power of LLMs: A Fine-Tuning Approach for High-Quality Aspect-Based Summarization (https://arxiv.org/abs/2408.02584)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 세부 요약(Aspect-based summarization)에 맞추어 미세 조정(fine-tuning)함으로써 문서의 특정 측면에 집중한 고품질 요약을 생성하는 방법을 탐구합니다. 오픈 소스 기반 LLMs인 Llama2, Mistral, Gemma, Aya를 사용하여 세부 요약 데이터셋을 통해 이들의 성능을 평가합니다.

- **Technical Details**: 연구는 LLMs를 Open Aspect-based Summarization (OASUM) 데이터셋으로 미세 조정하여 특정 측면에 대한 내용을 효과적으로 추출하고 요약할 수 있도록 하는 데 중점을 둡니다. 모델은 종합적인 평가 프레임워크를 설정하여 기존 세부 요약 방법 및 원래 LLMs 버전과의 성능을 비교합니다.

- **Performance Highlights**: 초기 결과는 미세 조정된 LLMs가 최신 세부 요약 방법과 비교할 때 더 높은 품질의 요약을 생성할 수 있음을 보여줍니다. 이는 교육, 의료, 음악 등 다양한 도메인에서 데이터 변형과 필요한 전문 지식을 효과적으로 처리할 수 있음을 시사합니다.



### Evaluating and Enhancing LLMs Agent based on Theory of Mind in Guandan: A Multi-Player Cooperative Game under Imperfect Information (https://arxiv.org/abs/2408.02559)
- **What's New**: 대형 언어 모델(LLMs)이 불완전한 정보 환경에서 복잡한 게임을 다루고, 특히 비 영어권 환경에서 다중 에이전트 협력을 가능하게 하는 능력을 탐구한 연구입니다. 이 연구는 오픈소스 및 API 기반 LLM이 에이전트 협력이 필요한 고도화된 텍스트 기반 게임에서의 적용 가능성을 조사하여, 다른 유형의 에이전트들과의 성능을 비교합니다.

- **Technical Details**: 연구는 Theory of Mind (ToM) 계획 기법을 제안하여, LLM 에이전트들이 게임 규칙, 현재 상태 및 역사적 맥락만을 입력으로 사용하여 다양한 적대자에 맞춰 전략을 조정할 수 있도록 합니다. 또한, 이 카드 게임에서 동적이고 광범위한 행동 공간의 문제를 완화하기 위해 외부 도구를 통합했습니다.

- **Performance Highlights**: 결과적으로, 현재의 LLM과 최신 강화 학습(RL) 모델 사이에는 성능 격차가 존재하지만, LLM은 이 게임 설정에서 ToM 능력을 보여줍니다. 이는 LLM이 동맹 및 적대자의 행동을 이해하고 동맹과의 협력을 구축할 수 있는 능력을 시사합니다. 추가 연구와 이해를 장려하기 위해, 연구진은 코드베이스를 공개했습니다.



### RAG Foundry: A Framework for Enhancing LLMs for Retrieval Augmented Generation (https://arxiv.org/abs/2408.02545)
Comments:
          10 pages

- **What's New**: 저자들은 Retrieval-Augmented Generation (RAG) 시스템의 복잡성을 해결하고자 오픈 소스 프레임워크인 RAG Foundry를 소개했습니다. 이 프레임워크는 데이터 생성, 모델 학습(training), 추론(inference), 평가 과정을 단일 워크플로에 통합하며, 대형 언어 모델(LLM)을 RAG 환경에서 학습하고 평가하는 데 도움을 줍니다.

- **Technical Details**: RAG Foundry는 다양한 RAG 기술을 실험하고 프로토타이핑하는 과정을 단순화합니다. 사용자는 내부 또는 전문 지식 소스를 활용해 손쉽게 데이터셋을 생성하고 RAG 모델을 학습할 수 있습니다. 이러한 통합 접근 방식은 데이터 증강(dataset augmentation) 및 학습에 대한 효율성을 증대시키며, 다양한 RAG 설정에서의 성능을 극대화합니다. Llama-3와 Phi-3 모델을 다양한 RAG 구성으로 증강 및 미세 조정(fine-tuning)하여 프레임워크의 효과를 입증했습니다.

- **Performance Highlights**: 세 가지 지식 집중 데이터셋(knowledge-intensive dataset)에서 일관된 성능 향상이 확인되었습니다. 이는 프레임워크가 다룰 복잡한 설계 결정을 완화하면서도 데이터 검색 정확도와 생성 품질을 동시에 향상시킬 수 있음을 보여줍니다. 결과적으로, RAG Foundry는 효율적인 RAG 시스템 구현 및 평가를 촉진하는 도구임이 입증되었습니다.



### Caution for the Environment: Multimodal Agents are Susceptible to Environmental Distractions (https://arxiv.org/abs/2408.02544)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLM)을 GUI 환경에서 평가하며, 이런 모델들이 환경적 맥락에 의해 얼마나 쉽게 산만해질 수 있는지를 분석했습니다. 일반적인 상황 설정에서 사용자와 에이전트는 모두 선의적이지만, 환경은 악의적이지 않은 관련 없는 콘텐츠를 포함할 수 있습니다. 실험 결과, 가장 강력한 모델들조차도 환경적 산만함에 민감하며 유기적으로 작동하기 어렵다는 것을 보여주었습니다.

- **Technical Details**: 이 연구는 MLLM들이 GUI 에이전트로서 환경적 방해 요소로 인해 산만해질 가능성을 평가하기 위해 자동화된 데이터셋을 사용합니다. 이를 위해 표본 데이터셋에는 팝업 박스, 검색, 추천, 채팅 등 네 가지 취약한 시나리오가 포함되며, 세 가지 다양한 작업 패턴이 적용되었습니다. 또한, 도발적인 환경 삽입(environment injection)을 통해 교란을 일으킬 수 있는 가능성을 실험적으로 분석했습니다.

- **Performance Highlights**: 10개의 대표적인 MLLM들을 실험한 결과 일반적 및 전문적 GUI 에이전트 모두 환경적 산만함에 약한 것으로 나타났습니다. 이와 같은 산만함을 감소시키기 위해 환경 인식을 향상시키는 것이 충분하지 않음을 증명했습니다.



### OneLove beyond the field -- A few-shot pipeline for topic and sentiment analysis during the FIFA World Cup in Qatar (https://arxiv.org/abs/2408.02520)
Comments:
          Accepted at KONVENS 2024

- **What's New**: 이 연구는 2022년 FIFA 월드컵 동안 독일 트위터 사용자들이 OneLove 팔찌에 대한 의견과 논쟁을 분석했습니다. 이 팔찌는 인권 운동의 일환으로 계획된 항의 활동의 일환이었으며, FIFA의 제재 위협으로 논란이 되었습니다. 연구진은 LLMs(대형 언어 모델, Large Language Models)을 사용하여 독일어 트윗을 분석하고, 인권, LGBT 권리, 정치 등의 주제들이 어떻게 논의되었는지를 확인했습니다. 초기에는 OneLove 팔찌에 대한 논의가 크고, 추후에는 스포츠와 정치의 일반적인 연결로 이동하며 감정이 중립적으로 변화하는 경향을 보였습니다.

- **Technical Details**: 이 연구는 총 132,150개의 독일어 트윗을 대상으로 수행되었으며, BERTopic과 같은 비지도 학습 방법을 사용해 토픽 모델링을 수행했습니다. CountTokenizer를 활용해 1에서 3까지의 n-그램을 포함하여 'one love'와 같은 바이그램과 'One Love Binde'와 같은 트리그램을 분석했습니다. 또한, 상위 토픽 클러스터를 개선하기 위해 자체적으로 독일어 불용어 목록을 작성했습니다. 연구진은 수동으로 600개의 트윗을 주제별로 라벨링하여 토픽 모델을 보완하고, 200개의 트윗을 찬성, 반대, 중립으로 분류하여 감정 분석을 수행했습니다. 상호 주석자 일치도(Inter-Annotator Agreement)에 대한 평가도 포함되었습니다.

- **Performance Highlights**: 연구 결과, 초기에는 OneLove 팔찌의 영향력, LGBT 권리, 정치적 주제가 트위터에서 활발히 논의되었으나, FIFA의 제재 발표 이후에는 일반적인 스포츠와 정치의 연결로 주제가 이동했습니다. 감정은 중립적으로 변화하는 경향을 보였으며, 이는 스포츠 활동에 대한 공중의 의견을 실시간으로 분석할 수 있는 새로운 방법론의 가능성을 제시했습니다. 이는 특별히 이벤트가 진행 중일 때 특정 의견에 대한 라벨링이 어려운 경우에 유용할 수 있습니다.



### UnifiedMLLM: Enabling Unified Representation for Multi-modal Multi-tasks With Large Language Mod (https://arxiv.org/abs/2408.02503)
- **What's New**: 최신 연구에서는 다양한 작업을 이해하고 추론하는 데 있어 뛰어난 성능을 보여주는 다중 모드 대규모 언어 모델(Multi-modal Large Language Models, MLLMs)의 중요한 발전을 달성했습니다. 그러나 이러한 모델들은 특정 작업에 맞춰 훈련되고 작업별 입력-출력 형식에 의존하기 때문에 광범위한 작업에 적용하기 어렵습니다. 이 문제를 해결하기 위해, 이 논문에서는 다양한 작업을 통합된 방식으로 표현하고 처리할 수 있는 통합MLLM(UnifiedMLLM)을 제안합니다. 이 모델은 사용자 지침의 암묵적 의도를 이해하고 추론하는 강력한 능력을 보여줍니다.

- **Technical Details**: UnifiedMLLM은 다양한 작업을 통합된 표현 방식으로 모델링하고 처리합니다. 이를 위해 모델은 작업 토큰(task tokens)과 그라운딩 토큰(grounding tokens)을 도입하여 작업 유형과 세부 사항을 표시합니다. 그런 다음, 이러한 토큰을 통해 작업 라우터(task router)를 통해 특정 전문 모델(expert models)에 작업을 할당합니다. 본 연구에서는 모델 훈련을 위해 고유의 작업별 데이터셋과 복잡한 시나리오를 포함하는 10만 개의 다중 작업 데이터셋을 구축했습니다. 세 단계의 훈련 전략을 사용하여 모델의 일반화 능력과 지식을 유지하면서 강력한 추론 및 작업 처리 능력을 갖추도록 했습니다.

- **Performance Highlights**: 다양한 작업에 대해 광범위한 실험을 수행한 결과, 통합된 표현 방식을 사용한 모델이 기존 방법론을 능가하는 뛰어난 성능을 보여주었습니다. UnifiedMLLM은 높은 확장성(scalability)과 일반성(generality)을 입증했으며, 추가 훈련 없이도 더 많은 작업을 통합할 수 있었습니다. 이 연구는 다양한 작업을 통합된 토큰 표현으로 매끄럽게 통합함으로써 다중 작업 학습을 위한 새로운 접근 방식을 제시합니다.



### Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models (https://arxiv.org/abs/2408.02442)
Comments:
          18 pages

- **What's New**: 새로운 연구는 JSON 및 XML과 같은 표준화된 형식으로 콘텐츠를 생성하는 '구조화된 생성(Structured Generation)'이 대형 언어 모델(LLMs)의 추론 및 도메인 지식 이해 능력에 미치는 영향을 조사했습니다. 연구 결과, 형식 제약이 있음에 따라 LLM의 추론 능력이 크게 감소한다는 점을 발견했습니다. 특히, 엄격한 형식 제약일수록 추론 업무의 성능 저하가 더 두드러집니다.

- **Technical Details**: ['**제약된 디코딩(Constrained Decoding)**: 사전 정의된 토큰 공간을 통해 출력을 제한하는 방식으로, LLM의 생성과정에서 JSON Mode와 같은 형식을 유지하게 합니다.', '**형식 제한 지시(FRI, Format-Restricting Instructions)**: JSON, XML, YAML과 같은 표준 형식을 따르도록 지시어를 사용하여 출력을 설정합니다.', '**자연어-형식 전환(NL-to-Format)**: 자연어 응답을 먼저 생성한 후, 이를 목표 형식으로 변환하는 두 단계의 과정으로 수행됩니다.']

- **Performance Highlights**: ['형식 제약이 없는 자연어 응답과 비교할 때 NL-to-Format 방식의 결과는 거의 동일한 성능을 보였습니다. 그러나 모델에 따라 일부 생성 오류로 인해 성능이 약간 저하될 수 있습니다.', "JSON Mode와 FRI(JSON)을 비교했을 때, JSON Mode는 특정 업무(Last Letter)에서 훨씬 낮은 성능을 보였습니다. 이유는 JSON Mode 응답이 '이유(reason)' 키 앞에 '답변(answer)' 키를 두어, 체인 형태의 추론 대신 직접 답변을 생성했기 때문입니다.", '즉, 형식 제한의 정도와 구현 방식은 특히 추론 업무에서 LLM 성능에 큰 영향을 미칠 수 있습니다.']



### Long Input Benchmark for Russian Analysis (https://arxiv.org/abs/2408.02439)
- **What's New**: LIBRA(롱 입력 벤치마크, Long Input Benchmark for Russian Analysis)은 러시아어로 작성된 긴 텍스트를 이해할 수 있는 적절한 평가 도구를 제공합니다. 새로운 벤치마크는 21개의 데이터셋으로 구성되었으며, LLM의 장문의 문맥 이해 능력을 평가하기 위해 설계되었습니다.

- **Technical Details**: LIBRA는 다양한 복잡성을 가진 4개의 그룹으로 나누어졌으며, 각 모델은 4k에서 128k 토큰의 문맥 길이를 가진 데이터셋에서 평가될 수 있습니다. 또한 공개된 오픈 소스 데이터셋, 코드베이스 및 리더보드를 제공하여 향후 연구 방향을 안내합니다.

- **Performance Highlights**: LIBRA 벤치마크를 통해 LLM들은 문맥의 길이가 증가할 때 성능이 어떻게 변화하는지에 대한 깊이 있는 분석이 가능합니다. 길이에 따른 성능 변화를 탐구함으로써 모델의 한계를 이해하고, 향후 개선 방향을 모색할 수 있습니다.



### Infusing Emotions into Task-oriented Dialogue Systems: Understanding, Management, and Generation (https://arxiv.org/abs/2408.02417)
Comments:
          Accepted by SIGDIAL 2024

- **What's New**: 이 연구는 사용자 감정(emotion)을 완전히 포함하는 새로운 Task-oriented Dialogue (ToD) 시스템을 제안합니다. 이전의 연구들은 감정을 부분적으로 고려했으나, 이번 연구는 감정이해, 감정관리, 감정생성 단계 모두를 포함한 완전한 ToD 파이프라인을 구현했습니다. 이를 위해, 기존의 EmoWOZ 데이터셋에 시스템 감정적 행동을 레이블링한 새로운 데이터를 추가했습니다.

- **Technical Details**: 연구진은 EmoWOZ 데이터셋을 확장하여 71,000개의 시스템 발화에 대해 감정적 행동을 레이블링했습니다. 이로써, 대화 시스템은 감정 인식, 감정 관리, 감정 표현 기능을 갖추게 되었습니다. 모듈형 시스템에서는 강화 학습(RL)을 사용하여 감정과 작업 성공을 보상 신호로 삼아 대화 정책을 학습했습니다. 또한, 감정 인식과 생성을 포함한 end-to-end 감정 기반 LLM(ToD 시스템)을 구축했습니다.

- **Performance Highlights**: 시뮬레이션과 실제 사용자와의 실험을 통해, 제안된 시스템이 사용자의 감정적 경험과 작업 성공률(task success)을 크게 향상시켰음을 확인했습니다. 이로써 감정 모델링이 Task-oriented Dialogue 시스템에서 매우 중요함을 입증했습니다.



### Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models (https://arxiv.org/abs/2408.02416)
- **What's New**: 최근 대형 언어 모델(LLMs)의 매개변수 수가 급격히 증가하면서 미세 조정 없이 프롬프트(예: 작업 설명)를 통해 하위 맞춤화를 연구하는 새로운 방향이 나타났습니다. 프롬프트 기반 서비스(예: OpenAI의 GPTs)는 많은 비즈니스에서 중요한 역할을 하지만, 프롬프트 누출에 대한 우려가 증가하고 있습니다. 이 논문에서는 프롬프트 누출, 즉 프롬프트 기억화(prompt memorization)의 기본 메커니즘을 분석하고 이에 대응할 방어 전략을 개발합니다.

- **Technical Details**: 프롬프트 추출에서의 스케일링 법칙을 탐구하여 모델 크기, 프롬프트 길이, 프롬프트 유형 등 프롬프트 추출에 영향을 미치는 주요 속성을 분석합니다. 두 가지 가설, 즉 난해도(perplexity)와 주의 행렬(attention matrices) 속의 직접적인 토큰 번역 경로를 제시합니다. SPLIt(Single Prompt Linking Indicator)라는 새로운 지표를 도입하여 프롬프트와 생성된 텍스트 간의 주의 연결을 추적했습니다.

- **Performance Highlights**: Llama2-7B와 GPT-3.5 모델에서 프롬프트 추출율이 각각 83.8%와 71.0% 감소하는 방어 전략을 고안했습니다. 이는 LLMs가 프롬프트 추출 공격에 대해 매우 취약하다는 사실을 보여주며, 사용자가 직접적인 의도를 가지고 있을 때도 마찬가지입니다.



### A Few-Shot Approach for Relation Extraction Domain Adaptation using Large Language Models (https://arxiv.org/abs/2408.02377)
- **What's New**: 이 논문에서는 대형 언어 모델 (LLM)의 인컨텍스트 학습(in-context learning) 능력을 활용하여 아키텍처, 건설, 엔지니어링, 운영 (AECO) 도메인의 연구 논문 제목과 초록에서 관계 추출(relation extraction)을 수행하는 새로운 방법을 제안합니다. 이는 SciERC 같은 특정 도메인 데이터셋에 의존하던 기존 방법과 차별화되며, 최소한의 전문가 주석을 이용하여 도메인 적응을 지원하는 few-shot 학습 전략을 채택합니다.

- **Technical Details**: 이 연구는 과학적 지식 그래프 과생성 모델의 도메인 적응을 위해 SciERC 가이드라인을 따릅니다. 먼저, OpenAlex 데이터베이스에서 476,000개의 AECO 분야 논문 데이터를 수집하고 이를 전처리한 후 Brat 도구로 주석을 달았습니다. 그런 다음 BERT 기반의 SpERT(Span-based Entity and Relation Transformer) 모델을 사용하여 관계 및 엔티티 추출 작업을 수행하였고, few-shot 예제를 통한 prompt tuning 접근법을 적용하여 LLM의 관계 추출 성능을 테스트했습니다.

- **Performance Highlights**: 기존의 SciERC 데이터셋을 사용하여 훈련된 SpERT 모델은 새로운 AECO 도메인에서 성능이 크게 저하되었으나, 제안된 few-shot 학습 방법을 활용한 모델은 minimal expert annotation만으로도 기존 베이스라인 대비 성능 개선을 보였습니다. 이는 특히 엔티티 추출과 관계 추출 작업에서 유의미한 성능 향상을 보여주며, 비용 효율적 도메인 적응의 가능성을 제시합니다.



### Dialogue Ontology Relation Extraction via Constrained Chain-of-Thought Decoding (https://arxiv.org/abs/2408.02361)
Comments:
          Accepted to appear at SIGDIAL 2024. 9 pages, 4 figures

- **What's New**: 대화 지향적인 (task-oriented) 대화 시스템의 새로운 접근법이 제안되었습니다. 이 논문에서는 관계 추출 (relation extraction)을 개선하기 위해 대규모 언어 모델 (Large Language Model, LLM)의 디코딩 메커니즘을 확장하는 방법으로서 Chain-of-Thought (CoT) 디코딩을 적용했습니다. 이를 통해 사용자 쿼리를 자동으로 처리하는 동안 발생하는 'hallucination'을 줄이고, 데이터에 대해 유의미한 관계를 더 정확하게 추출할 수 있게 했습니다.

- **Technical Details**: 이 연구는 CoT 디코딩 메커니즘을 대화 오토노미 관계 추출 (Dialogue Ontology Relation Extraction, DORE)에 확장하였습니다. CoT 디코딩은 원래 논리적 추론 문제에 적용되었으며, 이번 연구에서는 여러 디코딩 경로를 생성하고, 각 경로에서 예측된 관계의 신뢰도에 기반해 최종 답을 선택하는 방식으로 적용됩니다. 이를 위해 CoT 디코딩과 제한된 디코딩 (constrained decoding)을 결합하여 모델의 세분된 지식 활용을 최적화했습니다.

- **Performance Highlights**: 제안된 디코딩 메커니즘은 기존의 source one-shot 및 source fine-tuning 기준점을 모두 능가하는 성능을 보였습니다. 특히, MultiWOZ 2.1 및 Schema-Guided Dialogue (SGD) 데이터셋에서 실험을 통해 이 메커니즘의 효용성을 확인할 수 있었습니다.



### SNFinLLM: Systematic and Nuanced Financial Domain Adaptation of Chinese Large Language Models (https://arxiv.org/abs/2408.02302)
- **What's New**: 이번 뉴스레터에서는 큰 언어 모델(Large Language Models, LLMs)을 금융 산업의 자연어 처리 어플리케이션에 최적화하기 위해 설계된 새로운 모델인 'SNFinLLM'을 소개합니다. 이는 기존 금융 LLM에서 자주 발생하는 환각(hallucinations) 문제와 피상적인 파라미터 훈련의 한계를 극복하기 위해 개발되었습니다.

- **Technical Details**: SNFinLLM은 금융 도메인에 특화된 질문 응답, 금융 연구 보고서 요약, 감정 분석, 금융 계산 등의 작업에서 뛰어난 성능을 보입니다. 이를 위해 광범위한 금융 데이터를 수집하고 뉴스를 포함한 높은 품질의 지시 데이터셋(instruction dataset)을 생성하였습니다. 공개 소스 기반 모델을 이용하여 지속적인 사전 훈련(pre-training)을 진행하고, 이후 SFT(Supervised Fine-Tuning)를 통해 모델의 도메인 전문성을 강화하였습니다. 특히, 직관적인 선호도 최적화(Direct Preference Optimization, DPO) 방식을 적용하여 인간의 선호와 더 잘 일치하도록 모델을 조정하였습니다.

- **Performance Highlights**: 금융 벤치마크 및 자체 평가 데이터셋에서 광범위한 실험을 통해 SNFinLLM이 최신 금융 언어 모델들보다 뛰어난 성능을 보였다는 결과를 확인했습니다. 주목할 만한 성과로는 감정 분석을 위한 Opinion AI, 관련 문서 검색을 결합한 뛰어난 기계 읽기 이해(MRC)를 통해 도출된 Advisory AI, 그리고 투자 및 연구 AI에서의 뛰어난 보고서 요약 및 작성 기능이 있습니다.



### Decoupled Vocabulary Learning Enables Zero-Shot Translation from Unseen Languages (https://arxiv.org/abs/2408.02290)
Comments:
          Accepted to ACL 2024

- **What's New**: 이 연구는 멀티링구얼 뉴럴 머신 트랜스레이션(Multilingual Neural Machine Translation) 시스템이 서로 다른 언어의 문장을 공통된 표현 공간에 매핑할 수 있는지에 대한 실험을 진행합니다. 특히, 새로운 언어에서의 제로샷(Zero-shot) 번역 효율성을 테스트하며, 이를 위해 어휘와 구문 학습을 분리하는 새로운 설정을 제안합니다.

- **Technical Details**: 이 시스템은 새로운 언어의 어휘를 다루기 위해 크로스 링구얼 워드 임베딩(Cross-lingual word embeddings)을 사용하여 어휘와 구문 학습을 분리합니다. 이를 통해 어휘 표현을 고정한 상태에서 번역을 학습할 수 있으며, 이는 새로운 언어에서도 별도의 적응 없이 제로샷 번역이 가능하게 합니다. 특히, 독일 계열(Germanic)과 로망스 계열(Romance) 언어로 학습된 모델을 이용하여 포르투갈어-영어 및 러시아어-영어 번역을 성공적으로 수행하였습니다.

- **Performance Highlights**: 독일 계열(Germanic)과 로망스 계열(Romance) 언어로 학습된 모델을 이용한 제로샷 번역 실험에서 포르투갈어-영어 번역에 대해 42.6 BLEU 점수를, 러시아어-영어 번역에 대해 20.7 BLEU 점수를 기록했습니다. 또한, 이 시스템의 제로샷 번역 능력은 학습된 언어의 수가 증가함에 따라 더욱 향상되었습니다. 마지막으로, 이 모델을 이용한 비지도 학습(unsupervised learning)에서도 높은 성능을 보여주었습니다.



### StyEmp: Stylizing Empathetic Response Generation via Multi-Grained Prefix Encoder and Personality Reinforcemen (https://arxiv.org/abs/2408.02271)
Comments:
          Accepted by the 25th Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL 2024)

- **What's New**: 이번 연구에서는 컨시스턴트(Personality) 개념을 감성 대화 시스템에 도입하여 일관된 성격을 가진 공감적 응답을 생성하는 새로운 모델 StyEmp를 도입했습니다. 이는 기존의 공감적 응답 생성 모델들이 시스템의 성격을 고려하지 않은 문제를 해결합니다.

- **Technical Details**: StyEmp는 멀티-그레인 프리픽스 메커니즘(Multi-grained Prefix Mechanism)을 활용하여 시스템의 성격과 공감적 표현의 관계를 파악합니다. 또한, 대조 학습(Contrastive Learning)을 이용한 성격 강화 모듈(Personality Reinforcement Module)을 도입하여 응답이 공감적일 뿐만 아니라 명확한 성격을 반영하도록 합니다.

- **Performance Highlights**: EMPATHETICDIALOGUES 벤치마크에서의 자동 및 인간 평가 결과, StyEmp는 공감과 성격 표현 측면에서 경쟁 모델들보다 뛰어난 성능을 보였습니다.



### To Aggregate or Not to Aggregate. That is the Question: A Case Study on Annotation Subjectivity in Span Prediction (https://arxiv.org/abs/2408.02257)
Comments:
          Accepted at WASSA 2024

- **What's New**: 이번 연구에서는 법률 문제 설명에서 법률 영역 라벨을 지원하는 텍스트 스팬(Span)을 자동으로 예측하는 과제를 탐구합니다. 이는 일반인들이 작성한 문제 설명을 기반으로 하며, 실제 변호사들이 주석을 달았습니다. 주요 발견은 다수 투표 방식으로 학습시키는 것이 분리된 주석을 사용하는 것보다 더 나은 성능을 보인다는 점입니다.

- **Technical Details**: 법률 문제 설명을 주어진 법률 영역으로 자동 분류하는 것이 주요 과제입니다. 이 연구에서는 여러 변호사들이 주석을 단 데이터셋을 사용하여 학습 모델을 개발했습니다. 주어진 문제 설명과 법률 영역 라벨을 기반으로, 해당 라벨을 지원하는 텍스트 스팬을 예측하는 방식입니다. 우리는 문제를 sequence tagging 문제로 모델링하여 S(Sparse)와 y () 순서를 사용했습니다. 이러한 접근 방식에서 주목해야 할 부분은 다수의 전문가들 의견이 반영된 스팬을 활용하는 것이 성능 개선에 중요한 역할을 한다는 것입니다.

- **Performance Highlights**: 다양한 평가 시나리오에서 다수 투표에 의한 주석을 통해 학습하는 것이 분리된 주석을 사용하는 것 보다 더 나은 성능을 보였습니다. 이는 법률 전문가 주관성이 반영된 데이터가 모델 성능 향상에 긍정적인 영향을 미친다는 것을 시사합니다.



### Advancing Post-OCR Correction: A Comparative Study of Synthetic Data (https://arxiv.org/abs/2408.02253)
Comments:
          ACL 2024 findings

- **What's New**: 이 논문은 OCR(Optical Character Recognition) 이후의 데이터 처리 분야에서 데이터 양, 데이터 증강 방식 및 합성 데이터 생성 방법이 모델 성능에 미치는 영향을 평가하는 실험을 수행했습니다. 특히, 저자들은 컴퓨터 비전 기능 감지 알고리즘을 활용하여 글리프 유사성을 계산하여 합성 데이터를 구성하는 새로운 알고리즘을 소개합니다. 또한 ByT5와 같은 모델이 수동으로 주석이 달린 데이터 없이도 Character Error Rates (CER)를 크게 줄일 수 있음을 입증했습니다.

- **Technical Details**: 논문에서 저자들은 여러 방법을 사용하여 합성 데이터를 생성하고, 데이타 볼륨 및 증강 방법이 OCR 이후 모델 성능에 미치는 영향을 조사했습니다. 새로운 방법으로 컴퓨터 비전의 기능 감지 알고리즘을 사용하여 글리프 유사성을 기반으로 합성 데이터를 생성합니다. 이를 통해 여러 저자들이 제안한 방법보다 특히 자원이 부족한 언어에서 더 나은 성능을 보였습니다. 실험은 8개의 언어(몇몇 저자원 부족 언어 포함)로 진행되었으며, 새로운 합성 데이터 생성 방법이 기존 방법들보다 우수한 성능을 발휘함을 확인했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법으로 모델들이 Character Error Rates (CER)를 12.41%에서 최대 48.18%까지 감소시키는 성과를 달성했습니다. 특히 ByT5와 같은 모델은 수동 주석이 필요 없이 우수한 성능을 보였으며, 자원이 부족한 언어에서도 기존 합성 데이터 생성 방법을 넘어서는 효과를 입증했습니다.



### ReDel: A Toolkit for LLM-Powered Recursive Multi-Agent Systems (https://arxiv.org/abs/2408.02248)
Comments:
          In submission to EMNLP 2024 (Demo Track)

- **What's New**: 최근 큰 관심을 받고 있는 대형 언어 모델(LLM)을 활용하여 복잡한 멀티 에이전트 시스템을 구축하는 방법에 관한 흥미로운 연구를 발표했습니다. ReDel이라는 이름의 툴킷(toolkit)을 소개하며, 이 툴킷은 재귀적인(delegative) 멀티 에이전트 시스템을 지원합니다. 기존에는 사람이 정적인 문제 분해 그래프를 정의하고, 그래프의 각 하위 문제를 처리할 에이전트를 정의하는 것이 일반적이었지만, ReDel은 이러한 작업을 자동화하여 루트 에이전트가 필요에 따라 서브 에이전트를 생성하고 업무를 위임하도록 합니다.

- **Technical Details**: ReDel은 Python 패키지로 구성되어 있으며, 사용자 정의 도구 생성, 위임 스키마, 이벤트 기반 로깅, 그리고 인터랙티브 리플레이를 지원합니다. ReDel의 주요 구성 요소는 '도구(tool)'와 '위임 스키마(delegation scheme)'로, 도구는 에이전트가 호출할 수 있는 함수 그룹을 나타내며, 위임 스키마는 에이전트가 서브 에이전트로 작업을 보내는 전략입니다. 대표적인 위임 스키마로는 DelegateOne과 DelegateWait가 있습니다. 또한, ReDel은 웹 인터페이스를 제공하여 사용자가 시스템과 상호작용하고, 저장된 실행 기록을 쉽게 재생할 수 있도록 돕습니다.

- **Performance Highlights**: ReDel을 사용하여 세 가지 다양한 에이전틱 벤치마크에서 실험한 결과, 성능이 상당히 향상된 것을 확인했습니다. 이를 통해 개발자는 시각화 및 디버깅 도구를 통해 향상 가능한 영역을 쉽게 식별할 수 있습니다. 또한 ReDel은 개발자가 자신의 도메인에 맞춘 도구와 위임 전략을 정의할 수 있게 하여 멀티 에이전트 시스템의 복잡한 행동을 쉽게 탐색할 수 있도록 돕습니다. ReDel은 MIT 라이선스 하에 오픈 소스 코드와 문서화, PyPI 패키지를 무료로 제공하여 누구나 쉽게 이용할 수 있습니다.



### BOTS-LM: Training Large Language Models for Setswana (https://arxiv.org/abs/2408.02239)
Comments:
          7 pages, 3 tables

- **What's New**: BOTS-LM 시리즈는 Setswana와 영어에 능통한 이중언어 모델들로 구성되어 있습니다. 최근 데이터 가용성과 효율적인 미세 조정 기술을 활용하여, BOTS-LM은 크기 대비 동등한 성능을 보여주며, 특히 8B(80억) 매개변수 모델은 Llama-3-70B와 Aya 23를 능가하는 성능을 보였습니다.

- **Technical Details**: BOTS-LM 시리즈는 다양한 크기의 모델로 구성됩니다. 초기 릴리즈에는 8B 매개변수 생성적 대형 언어 모델이 포함되어 있으며, 곧 0.5B 및 1B 매개변수 대형 언어 모델과 278M 매개변수 인코더 전용 모델도 공개될 예정입니다. 이 모델들은 Quantized LoRA (QLoRA) 기술을 사용하여 최적화되었습니다.

- **Performance Highlights**: 8B 모델은 Setswana-영어 번역 작업에서 Llama-3-70B와 Aya 23보다 뛰어난 성능을 보였습니다. 또한, MMLU 벤치마크의 기계 번역된 부분에서 Setswana 추론 성능이 70B 모델에 근접하는 성능을 나타냈습니다. BOTS-LM 시리즈와 함께 공개되는 데이터셋은 총 2억 6천7백만 토큰의 Setswana 웹 데이터셋인 SetsText를 포함하고 있습니다. 추가적으로, 이중언어 모델은 다양한 하드웨어 구성에서 실행될 수 있게 됩니다.



### Do Large Language Models Speak All Languages Equally? A Comparative Study in Low-Resource Settings (https://arxiv.org/abs/2408.02237)
- **What's New**: 본 연구는 대형언어모델(LLMs)의 저자원(low-resource) 언어인 벵골어, 힌디어, 우르두어에서의 한계에 초점을 맞췄습니다. 이를 위해 기존의 영어 데이터셋을 번역하여 감정 분석(sentiment analysis) 및 혐오 발언(hate speech) 태스크를 위한 데이터셋을 제공하고, 다양한 LLMs를 활용한 제로샷 학습(zero-shot learning)을 종합적으로 검토했습니다. 연구 결과, GPT-4가 Llama 2와 Gemini보다 성능이 뛰어났으며, 영어가 저자원 언어보다 다양한 태스크에서 일관되게 우수한 성능을 보였습니다.

- **Technical Details**: 연구는 GPT-4, Llama 2, 그리고 Gemini 3개의 LLMs를 사용하여 진행되었으며, 각 모델의 성능, 파라미터 크기, 기능 등을 기준으로 선택했습니다. 실험은 NLI(natural language inference) 태스크에서 XNLI 데이터셋, 감정 분석 태스크에서 SemEval-2017 Task 4, 혐오 발언 태스크에서 Davidson et al. (2017) 데이터셋을 사용했습니다. 또한, 자연언어 지시(Natural Language Instructions)를 이용한 제로샷 프롬프팅 접근법을 적용하여 더 적절한 출력을 생성했습니다.

- **Performance Highlights**: 영어와 저자원 언어들 간의 성능 격차가 두드러졌으며, 특히 NLI 태스크에서 영어의 성능이 벵골어, 힌디어, 우르두어보다 18.04%, 17.38%, 22.81% 더 우수했습니다. 감정 분석 태스크에서는 Gemini의 성능이 다른 연략 언어들보다 저하됐으며, 전반적으로 영어는 모든 태스크에서 일관되게 우수한 성능을 보였습니다. 이 연구 결과는 LLMs의 저자원 언어 처리능력 향상의 필요성을 강조합니다.



### A Multi-Source Heterogeneous Knowledge Injected Prompt Learning Method for Legal Charge Prediction (https://arxiv.org/abs/2408.02233)
Comments:
          20 pages

- **What's New**: 이번 연구에서는 법적 AI 분야의 필수 과제인 법적 혐의 예측(legal charge prediction)을 다루고 있습니다. 본 연구는 사건 설명(case descriptions)에 다양한 신경망 구조를 직접 모델링하는 기존 방법과 달리, 다중 소스의 외부 지식(multi-source external knowledge)을 활용한 프롬프트 학습 프레임워크(prompt learning framework)를 제안합니다. 이는 법적 지식베이스(legal knowledge base), 대화형 LLM(conversational LLM), 관련 법률 문서 등을 포함합니다.

- **Technical Details**: 먼저 사건 설명에서 법적 지식베이스와 일치하는 지식 조각을 하드 프롬프트 템플릿(hard prompt template)를 통해 입력으로 캡슐화합니다. 또한 대조 학습(contrastive learning)을 통해 사건 설명과 관련된 법률 문서를 검색하여, 그 후 대화형 LLM을 통해 사건 설명 내의 사실 요소(factual elements)를 추출합니다. 마지막으로, 소프트 프롬프트 토큰(soft prompt tokens)의 임베딩 벡터와 사실 요소의 인코딩 벡터를 결합하여 지식 강화 모델의 순차적 추론(knowledge-enhanced model forward inference)을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 가장 큰 법적 혐의 예측 데이터셋인 CAIL-2018에서 최첨단(state-of-the-art) 성능을 달성하였으며, 데이터 의존성이 낮았습니다(lower data dependency). 사례 연구에서도 높은 해석 가능성(strong interpretability)을 보였습니다.



### CodeACT: Code Adaptive Compute-efficient Tuning Framework for Code LLMs (https://arxiv.org/abs/2408.02193)
- **What's New**: 오픈 소스 대형 언어 모델(LLMs)의 코드 관련 작업 성능 향상을 위해 'Code Adaptive Compute-efficient Tuning(CodeACT)' 프레임워크를 제안합니다. 이 프레임워크는 복잡성과 다양성을 고려하여 고품질의 데이터를 선택하는 '복잡성 및 다양성 인지 샘플링 방법(Complexity and Diversity Aware Sampling, CDAS)'과 훈련 과정에서 패딩 토큰을 최소화하는 '동적 패딩 전략(Dynamic Pack)'을 도입합니다.

- **Technical Details**: CodeACT 프레임워크의 핵심 구성 요소는 CDAS와 Dynamic Pack입니다. CDAS는 개방형 LLM을 사용하여 데이터의 복잡성과 다양성에 따라 치밀하게 고품질 데이터를 선택합니다. 반면, Dynamic Pack은 데이터의 길이에 따라 데이터를 정렬하고 병합하여 패딩 토큰을 줄이고 훈련 시간을 단축시킵니다. 이를 통해 훈련 효율성을 극대화하고 자원 소비를 줄이는 것을 목표로 합니다.

- **Performance Highlights**: CodeACT-DeepSeek-Coder-6.7B 모델은 EVOL-Instruct 데이터의 40%만을 사용하여 8.6%의 성능 향상(HumanEval 기준)을 달성하였으며, 훈련 시간은 78%, GPU 메모리 사용량은 27% 줄였습니다. 이는 고품질 데이터 선택과 최적화된 훈련 전략이 오픈 소스 모델의 성능과 효율성을 크게 향상시킬 수 있음을 시사합니다.



### Analyzing Cultural Representations of Emotions in LLMs through Mixed Emotion Survey (https://arxiv.org/abs/2408.02143)
Comments:
          Was accepted to ACII 2024

- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 문화적 감정 표현, 특히 혼합 감정 상황에서의 표현을 분석합니다. Miyamoto et al. (2010)의 연구를 기반으로 일본어와 영어로 된 감정 반응 설문을 사용하여 LLM의 반응을 비교하고, LLM이 인종적, 문화적 편향을 어떻게 나타내는지 조사했습니다.

- **Technical Details**: 세 가지 연구를 수행했습니다. 첫 번째 연구는 일본어와 영어로 된 감정 설문에 대한 LLM의 반응을 비교하였고, 인간 실험 결과와의 일치를 분석했습니다. 두 번째 연구는 LLM 프롬프트에 다른 맥락적 정보를 추가하여 반응의 변화를 분석했습니다. 세 번째 연구는 동아시아 언어와 서유럽 언어 간의 반응 유사성을 비교했습니다. 사용된 모델은 mistral-7b-instruct, gemma-7b-it:free, llama-2-70b-chat, gpt-3.5-turbo, gpt-4-turbo-preview 등 5개의 LLM입니다.

- **Performance Highlights**: LLMs는 인간 실험 결과와 비교했을 때 제한된 일치도를 보였습니다. 프롬프트에서 참가자의 출신 정보보다 언어가 반응에 더 큰 영향을 미쳤습니다. 또한 동아시아 언어의 반응이 서유럽 언어보다 더 일관성이 있었습니다.



### Table Transformers for Imputing Textual Attributes (https://arxiv.org/abs/2408.02128)
- **What's New**: 이번 연구에서는 'Table Transformers for Imputing Textual Attributes (TTITA)'라는 새로운 접근 방식을 제안합니다. 이 방법은 transform을 기반으로 하여 다른 열의 데이터를 사용해 구조화되지 않은 텍스트 열의 결측값을 보완합니다. Amazon 리뷰 데이터셋을 활용한 실험에서 기존 모델인 순환 신경망(recurrent neural networks) 및 Llama2를 능가하는 성능을 보였습니다.

- **Technical Details**: TTITA는 이종 태블러 데이터(numeric, categorical, textual columns)를 인코딩하여 컨텍스트 벡터를 생성하고, 이 벡터를 기반으로 텍스트 열 결측값을 복원합니다. 인코더-디코더 구조에서는 인코더가 입력 데이터를 인코딩하고, 디코더가 cross-attention을 활용해 텍스트 시퀀스를 출력합니다. 각 입력 데이터 타입에 따라 별도의 인코딩 방식을 사용하며, 스캐너 기반 해싱 벡터라이저(hashing vectorizer)로 텍스트 데이터를 featurize합니다.

- **Performance Highlights**: TTITA는 Amazon 리뷰 데이터셋에서 기존 방법을 능가하는 성능을 보였으며, 특히 타겟 시퀀스의 길이가 더 긴 경우 성능 향상이 두드러졌습니다. 또한 Multi-task learning 기법을 도입하여 여러 유형의 열에 대해 동시에 결측값을 보완, 텍스트 보완의 성능을 향상시켰습니다. 뿐만 아니라 실제 응용 분야에서도 ChatGPT와의 정성적 비교에서 우수성을 입증했습니다.



### Recent Advances in Multi-Choice Machine Reading Comprehension: A Survey on Methods and Datasets (https://arxiv.org/abs/2408.02114)
- **What's New**: 최근의 멀티 초이스 기계 독해(Machine Reading Comprehension, MRC) 분야의 기술 진보에 대한 철저한 분석을 제공합니다. 본 논문은 벤치마크 데이터셋, 방법론, 과제, 그리고 미래의 방향성에 초점을 맞추어 연구자들에게 현 상황을 종합적으로 제공하는 것을 목표로 하고 있습니다.

- **Technical Details**: 논문에서는 30개의 기존 클로즈 스타일과 다중 선택형 (multiple-choice) MRC 벤치마크 데이터셋을 심층 분석하였습니다. 여기서 데이터셋을 corpus 스타일, 도메인, 복잡성, 컨텍스트 스타일, 질문 스타일, 그리고 답변 스타일 등의 속성을 기반으로 세분화하는 방법을 사용하였습니다. 또한, 최신 방법론을 Fine-tuned 방식과 Prompt-tuned 방식으로 분류하였습니다. Fine-tuned 방식은 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)을 도메인 특정 데이터셋으로 재훈련하여 특정 작업에 적응시키는 것이며, Prompt-tuned 방식은 제로샷(zero-shot) 또는 퓨샷(few-shot) 학습 시나리오에서 응답 생성을 유도하는 프롬프트를 사용하는 것입니다.

- **Performance Highlights**: 이 논문은 지속적인 논의를 촉진하고, 미래의 연구 방향을 제시하며, 혁신을 유도하여 멀티 초이스 MRC를 새로운 성취로 이끌기 위한 토대를 마련하는 것을 목표로 하고 있습니다.



### Effective Demonstration Annotation for In-Context Learning via Language Model-Based Determinantal Point Process (https://arxiv.org/abs/2408.02103)
- **What's New**: 이번 연구에서는 In-context learning (ICL) 패러다임을 개선하기 위해 LM-DPP (Language Model-based Determinant Point Process)를 소개합니다. 이는 Label 되지 않은 데이터의 불확실성과 다양성을 동시에 고려하여 최적의 서브셋을 선별하는 메커니즘입니다.

- **Technical Details**: LM-DPP는 대규모 언어 모델(GPT-J, LlaMA, GPT-3 등)의 perplexity를 사용하여 각 후보 인스턴스를 스코어링하고, Gram matrix를 통해 불확실성과 다양성을 균형 있게 고려합니다. 그런 다음 Greedy MAP inference 알고리즘을 사용하여 주석을 붙일 후보 서브셋을 선택합니다.

- **Performance Highlights**: 9개의 NLU와 2개의 Generation 데이터셋에 대한 실험 결과, LM-DPP는 낮은 불확실성과 높은 다양성을 지닌 데이터를 효과적으로 선별하며 기존 최고의 선택 방법을 크게 능가하는 성능을 보였습니다. 또한 모델 크기와 주석 예산에 따른 일반화 가능성 또한 우수함을 확인했습니다.



### MedSyn: LLM-based Synthetic Medical Text Generation Framework (https://arxiv.org/abs/2408.02056)
Comments:
          16 pages, accepted to ECML PKDD 2024

- **What's New**: 이번 연구에서는 MedSyn이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 대형 언어 모델(LLMs)과 의료 지식 그래프(MKG)를 통합하여 의료 텍스트를 생성합니다. MKG를 사용하여 초기 의료 정보를 샘플링하고, GPT-4와 미세 조정된 LLaMA 모델을 사용하여 합성 진료 기록을 생성합니다. 이 연구를 통해 ICD 코드 예측 작업에서 합성 데이터가 실제 임상 환경에서 얼마나 유용한지 평가합니다. 또한, 러시아어 임상 기록의 최대 개방형 합성 데이터 세트를 공개하였습니다.

- **Technical Details**: MedSyn 프레임워크는 질병 관련 증상을 MKG에서 샘플링하여 LLM 생성 파이프라인에 통합하여 데이터를 생성합니다. GPT-4와 LLaMA-7b 모델을 사용하여 실험을 진행했습니다. LLaMA-7b는 특정 데이터 세트로 미세 조정된 모델입니다. MKG는 WikiMed에서 기초 데이터를 가져와 러시아어로 구축되었습니다. MKG는 질병(ICD-10 코드로 식별), 약물, 증상과 같은 노드를 포함하며, ChatGPT를 통해 임상 증상의 텍스트에서 증상을 추출합니다.

- **Performance Highlights**: 합성 데이터는 ICD 코드 예측 작업에서 최대 17.8%까지 분류 정확도를 향상시킬 수 있음을 밝혔습니다. 또한, MedSyn은 특정 데이터 세트로 미세 조정된 오픈 소스 모델이 GPT-4와 동등하거나 더 나은 성능을 보일 수 있음을 보여줍니다. 공개된 러시아어 합성 의료 기록 데이터 세트는 41,000개 이상의 임상 기록을 포함하고 있으며, 219개의 ICD-10 코드를 다룹니다.



### Fine-tuning multilingual language models in Twitter/X sentiment analysis: a study on Eastern-European V4 languages (https://arxiv.org/abs/2408.02044)
Comments:
          18 pages, 4 figures

- **What's New**: 이번 연구는 특히 트위터/X 데이터를 이용한 국소 작업에서 언어 모델을 미세 조정하여 러시아 및 우크라이나에 대한 감성을 분석하는 점에 중점을 두었습니다. V4 국가(체코, 슬로바키아, 폴란드, 헝가리)의 언어로 작성된 트윗을 대상으로 하여 학습 및 테스트 데이터셋을 마련했습니다.

- **Technical Details**: 이번 연구에서는 여러 대형 언어 모델(LLM)을 미세 조정하였습니다. 사용된 모델로는 BERT, BERTweet, Llama2, Llama3, Mistral 등이 있습니다. 데이터는 트위터/X의 학술 API를 통해 2023년에 수집되었으며, 헬싱키 번역기와 DeepL을 사용하여 영어로 번역된 데이터셋도 생성되었습니다. 모델의 성능은 정확도, 재현율, 정밀도, F1 점수 등의 표준 지표로 평가하였습니다.

- **Performance Highlights**: 미세 조정을 이용해 6K 다국어 트윗만으로도 인-컨텍스트 학습보다 더 나은 성과를 보였습니다. 트위터/X 코퍼스에서 테스트한 모델의 성과가 일반적인 벤치마크 결과와 종종 상관관계가 없었습니다. 영어로의 좋은 번역은 다국어 사전 학습 모델에도 원래 언어를 사용하는 것보다 우위를 제공했습니다. 일부 모델은 언어 및 문화적 특수성이 반영된 예기치 않은 차이를 보였습니다.



### LLaSA: Large Language and E-Commerce Shopping Assistan (https://arxiv.org/abs/2408.02006)
Comments:
          Accepted by KDD 2024 Workshop (Oral)

- **What's New**: 본 논문은 전자상거래 쇼핑 보조 도구(Large Language Models, LLMs)의 활용을 통해 여러 문제를 해결하고자 합니다. 특히, 새로운 데이터셋 EshopInstruct를 제안하고, 이를 사용한 모델 'LLaSA'가 Amazon KDD Cup 2024에서 우수한 성과를 거두었다고 보고합니다.

- **Technical Details**: 기존 쇼핑 보조 도구는 여러 개의 특정 작업(task-specific) 모델을 필요로 하고, 최신 제품에 대한 일반화(generalization) 성능이 떨어지는 문제를 가지고 있었습니다. 이를 해결하기 위해, 논문에서는 LLMs를 사용하여 전반적인 작업을 처리할 수 있는 보조 도구를 개발하였습니다. EshopInstruct는 65,000개의 샘플과 다양한 작업을 포함한 데이터셋으로, 이 데이터셋을 통해 LLMs를 Instruction Tuning하여 향상된 성능을 보여주도록 했습니다.

- **Performance Highlights**: Amazon KDD Cup 2024 대회의 ShopBench에서 'LLaSA' 모델은 57개의 작업과 약 20,000개의 질문을 포함한 데이터셋에서 전체 3위를 차지했으며, 각 트랙에서도 상위 5위 안에 들었습니다. 특히, track4에서는 모든 학생 팀 중에서 최고 성적을 기록했습니다. 이러한 성과를 통해 LLMs가 전자상거래 쇼핑 보조 도구로서의 용량을 충분히 갖추었음을 입증했습니다.



### Optimal and efficient text counterfactuals using Graph Neural Networks (https://arxiv.org/abs/2408.01969)
- **What's New**: 이 논문에서는 NLP 모델의 예측을 변경함으로써 모델 설명 가능성을 제공하는 반사실 개입(counterfactual interventions)을 생성하는 프레임워크를 제안합니다. 이 프레임워크는 이진 감정 분류와 주제 분류 두 NLP 작업에서 테스트되었으며, 생성된 수정이 대조적, 유창하고 최소한임을 확인했습니다. 이 과정은 현재 최첨단 반사실 편집기보다 상당히 빠릅니다.

- **Technical Details**: 이 연구는 텍스트 분류기의 행동을 테스트하기 위해 단어 수준의 반사실 개입에 중점을 둡니다. 개입의 최적성을 보장하기 위해 그래프 이론에서 사용하는 그래프 할당 알고리즘을 통해 이 문제를 조합 최적화 문제로 봅니다. 이를 더욱 향상시키기 위해 그래프 신경망(Graph Neural Networks, GNNs)을 사용하여 최적 할당 과정을 가속화합니다. 제안된 방법은 모델-특정 및 범용 시나리오 모두에 적용 가능하며 최적 해결책이 비포괄적 탐색 기법(non-exhaustive search techniques)을 사용하여 도달할 수 있습니다.

- **Performance Highlights**: 제안된 블랙박스 반사실 편집기는 두 가지 데이터셋에서 기존의 화이트박스 및 블랙박스 방법에 비해 일관되게 최첨단(SOTA) 성능을 보이며 4가지 다양한 지표에서 2% 및 20% 미만의 시간 내에 작업을 완료합니다. 이 편집기는 특정 메트릭에 최적화되거나 일반 용도의 유창한 편집을 수행할 수 있는 다용성을 보여줍니다.



### ML-EAT: A Multilevel Embedding Association Test for Interpretable and Transparent Social Scienc (https://arxiv.org/abs/2408.01966)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 이번 연구는 언어 기술에서 내재된 편향(Intrinsic Bias)을 해석 가능하고 투명하게 측정하기 위해 다수준 임베딩 연결 테스트(Multilevel Embedding Association Test, ML-EAT)를 소개합니다. ML-EAT는 WEAT (Word Embedding Association Test)의 한계를 극복하며, 세 가지 세분화 수준에서 편향을 정량화합니다. 새로운 EAT-맵(EAT-Map) 시각화를 통해 결과를 보다 직관적으로 볼 수 있게 했습니다.

- **Technical Details**: ML-EAT는 세 가지 수준에서 편향을 측정합니다. 첫 번째 수준에서는 두 개의 대상 개념과 두 개의 속성 개념 간의 차등 연결을 측정합니다. 두 번째 수준에서는 각 대상 개념과 두 개의 속성 개념 간의 개별 효과 크기를 측정합니다. 세 번째 수준에서는 네 개의 기본 코사인 유사도 분포의 평균과 표준 편차를 측정합니다. 연구자들은 이를 통해 EAT 패턴을 아홉 가지로 분류하고, EAT-맵이라는 네 부분 시각화를 통해 이해를 돕습니다.

- **Performance Highlights**: ML-EAT를 사용하는 실증 분석에 따르면, 기존의 WEAT로는 관찰할 수 없는 추가 정보를 제공하며, 제로샷 모델(Zero-Shot Models)에서 프롬프트의 영향을 드러내고, 코사인 유사도가 효과적이지 않은 상황을 식별할 수 있습니다. 이를 통해 언어 기술의 편향을 보다 관찰 가능하고 해석 가능하게 만듭니다. 이러한 종합 분석은 정적 및 연대기적 단어 임베딩, GPT-2 언어 모델 및 CLIP 언어-이미지 모델에 성공적으로 적용되었습니다.



### A Novel Metric for Measuring the Robustness of Large Language Models in Non-adversarial Scenarios (https://arxiv.org/abs/2408.01963)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)들의 강인성을 다양한 데이터셋을 통해 평가했습니다. 강인성은 입력의 의미를 보존하는 변형에 대해 모델 응답이 얼마나 민감하지 않은지를 의미합니다. 자연스럽게 발생하는 비악의적인 변형이나 의미가 동일한 패러프레이즈(paraphrase)를 생성하여 기준 데이터셋을 구성했고, 이를 통해 모델의 강인성을 평가하는 새로운 측정 지표를 제안했습니다.

- **Technical Details**: 기존 데이터셋에 비해 문장의 의미는 그대로이지만, 겉으로 드러나는 변형(대소문자 변경, 구두점 제거, 오타 등)을 추가하여 입력 변형을 다양하게 도입했습니다. 세부적으로 PopQA, SIGA, BoolQ 등의 데이터셋을 사용하고 각기 다른 방식으로 변형하여 모델의 반응을 관찰했습니다. 이를 통해 다양한 비악의적인 입력 변형에 대해 모델의 회복력(resilience)을 평가했습니다. 성능 평가의 주요 지표로는 기존의 Performance Drop Rate(PDR)의 문제점을 보완한 Cohen's h 효과 크기(effect size) 지표를 도입하여, 평가의 대칭성과 해석 용이성을 향상시켰습니다.

- **Performance Highlights**: 제안된 지표인 Cohen's h를 통해 여러 대형 언어 모델들이 비악의적인 입력 변형에 얼마나 민감하지 않은지를 평가하였고, 종합적인 성능 평가를 진행했습니다. 이를 통해 기존의 PDR 지표가 갖는 비대칭성과 특정 경우의 평가 불가능성 등의 문제를 효과적으로 해결하였습니다.



### Defining and Evaluating Decision and Composite Risk in Language Models Applied to Natural Language Inferenc (https://arxiv.org/abs/2408.01935)
Comments:
          arXiv admin note: text overlap with arXiv:2310.03283

- **What's New**: 대형 언어 모델(LLM)인 ChatGPT 등의 성능은 매우 인상적이지만, 잘못된 자신감(과잉 또는 과소 자신감)에 의해 중요한 위험을 초래할 수 있습니다. 이 논문은 이러한 비대칭 문제를 해결하기 위해, 두 가지 위험 유형(결정 위험과 복합 위험)을 정의하고, 이 위험을 측정하기 위한 실험 프레임워크를 제안합니다.

- **Technical Details**: 이 논문에서는 두 레벨의 추론 아키텍처를 기반으로 하는 실험 프레임워크를 제안합니다. 첫 번째 레벨은 언어 모델이 추론을 보류해야 하는지 여부를 결정하는 '결정 규칙'에 기반하고, 두 번째 레벨은 모델이 보류하지 않을 경우 실행되는 모델의 추론입니다. 이를 통해 LLM의 리스크를 체계적으로 평가할 수 있습니다.

- **Performance Highlights**: 네 가지 자연어 상식 추론 데이터셋에서 실험한 결과, 제안된 프레임워크를 통해 LLM은 기존 방법이 고위험으로 잘못 분류하는 과제를 20.1% 추가로 자신 있게 응답하고, 잘못 답했을 과제의 19.8%를 건너뛸 수 있는 것으로 나타났습니다. 이로써 LLM의 결정 및 복합 위험을 각각 25.3%와 16.6%까지 감소시킬 수 있음을 보였습니다.



### DiReCT: Diagnostic Reasoning for Clinical Notes via Large Language Models (https://arxiv.org/abs/2408.01933)
Comments:
          9 pages,6 figures

- **What's New**: 다양한 의료 작업에 관한 진단 추론 능력과 모델 해석 가능성을 평가하기 위해 새롭게 'DiReCT' 데이터셋이 소개되었습니다. DiReCT 데이터셋은 의사들이 각 임상 기록에 세밀하게 주석을 달아 진단 추론 과정을 명확히 설명하는 자료를 포함하고 있습니다.

- **Technical Details**: {'대상': 'mimic-IV 데이터베이스에서 추출된 521개의 임상 기록', '구성': '각 임상 기록은 의사들이 주석을 단 관찰, 진단 추론 과정, 최종 진단을 포함합니다.', '특징': '기존 진단 지침에 기반한 진단 지식 그래프 제공', '구조': 'SOAP 형식으로 구성된 임상 기록 (주관적 정보, 객관적 정보, 평가, 계획)'}

- **Performance Highlights**: 현재의 최첨단 LLM은 인간 의사에 비해 여전히 진단 추론 능력에서 큰 차이가 있음이 밝혀졌습니다. 특히, 실제 임상 시나리오에서 효과적으로 추론하는 능력이 부족한 것이 확인되었습니다.



### A Semi-supervised Multi-channel Graph Convolutional Network for Query Classification in E-commerc (https://arxiv.org/abs/2408.01928)
Comments:
          Accepted by WWW2024

- **What's New**: 본 논문에서는 신규로 제안된 반지도 학습 기반 다중 채널 그래프 컨볼루션 네트워크 (Semi-supervised Multi-channel Graph Convolutional Network, SMGCN) 모델을 소개합니다. 이 모델은 레이블 연관성과 반지도 학습을 통해 기존 쿼리 의도 분류(query intent classification)의 문제점을 해결합니다.

- **Technical Details**: SMGCN은 카테고리 간 상관관계(co-occurrence)와 의미적 유사성(severity similarity) 그래프를 활용하여 카테고리 간의 관계를 강화하고 자동 생성된 라벨의 불안정성을 약화시킵니다. 이를 위해 다중 채널 GCN이 관계를 모델링하고 쿼리와 카테고리 간의 유사성 점수를 계산한 후, 클릭 라벨과 결합하여 손실 값을 계산합니다. 이 접근 방식은 감소된 데이터의 한계를 보완하며 관련 카테고리를 더 잘 회상할 수 있도록 돕습니다.

- **Performance Highlights**: 대규모 실제 데이터셋에 대한 오프라인 및 온라인 A/B 테스트 실험 결과, SMGCN은 기존 강력한 모델들보다 현저히 우수한 성능을 보였습니다. 해당 모델은 상용 전자상거래 플랫폼에 도입되어 매일 수억 건의 요청을 처리하고 있으며, 큰 상업적 가치를 제공합니다.



### Cross-layer Attention Sharing for Large Language Models (https://arxiv.org/abs/2408.01890)
Comments:
          Working in process

- **What's New**: 대형 언어 모델(Large Language Models, LLMs)의 효율성을 높이기 위해, 다양한 LLM들에서 공통적으로 나타나는 과도한 중복성을 해결하고자 합니다. LiSA라는 새로운 접근 방식을 소개합니다. LiSA는 잘 훈련된 LLM에서 자기 주의 기제(self-attention)에 대한 경량 대체제를 제공합니다.

- **Technical Details**: 기존의 주의 기제 개선 연구는 주로 KV 캐시(KV cache)를 압축하거나 주의 헤드(attention heads)를 그룹화하는 데 중점을 두었습니다. 그러나 이번 연구는 레이어(layer) 간 중복성을 자세히 분석했습니다. LiSA는 인접 레이어 사이의 주의 헤드를 맞추기 위해 작은 피드포워드 네트워크(feed-forward networks)를 사용하고, 레이어 간 주의 가중치 차이를 근사하기 위해 저차 행렬(low-rank matrices)을 사용합니다.

- **Performance Highlights**: 13개의 다양한 기준에서 평가한 결과, LiSA는 정확도와 혼동도(perplexity) 측면에서 높은 응답 품질을 유지하면서 전체 레이어의 53-84%에서 중복된 주의 계산을 줄입니다. LiSA를 구현한 결과, Q와 K를 6배 압축하고, LLaMA3-8B에서는 최대 처리량이 19.5% 개선되었으며, LLaMA2-7B에서는 32.3% 개선되었습니다.



### Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieva (https://arxiv.org/abs/2408.01875)
- **What's New**: 최근의 대규모 언어 모델(LLM) 발전으로 인해 다양한 도구들을 사용하여 복잡한 추론과 작업 실행 능력을 가진 자율 에이전트가 가능하게 되었습니다. 그러나 도구 집합의 규모가 커짐에 따라 주어진 작업에 가장 관련성 높은 도구를 식별하는 것이 주요 병목 현상으로 작용하게 됩니다. 이를 해결하기 위해, 우리는 Re-Invoke라는 비지도 도구 검색 방법을 제안합니다. 이 방법은 대규모 도구 집합에 효과적으로 확장될 수 있도록 설계되었으며, 별도의 훈련 없이도 동작합니다.

- **Technical Details**: Re-Invoke는 두 가지 주요 단계로 구성됩니다. 첫째, 도구 인덱싱 동안 각 도구 문서와 관련된 쿼리 공간의 다양한 측면을 포괄하는 다양한 합성 쿼리 집합을 생성합니다. 둘째, 추론 단계에서 사용자 쿼리로부터 도구와 관련된 주요 컨텍스트와 기본 의도를 추출하기 위해 LLM의 쿼리 이해 능력을 활용합니다. 마지막으로, 의도를 기반으로 한 새로운 다중 관점 유사성 랭킹 전략(multi-view similarity ranking strategy)을 사용하여 각 쿼리에 가장 관련성 높은 도구를 정확하게 짚어냅니다.

- **Performance Highlights**: 평가 결과 Re-Invoke가 완전한 비지도 설정에서도 최첨단 대안들보다 단일 도구와 다중 도구 시나리오 모두에서 상당히 뛰어난 성능을 보였습니다. 특히 ToolE 데이터셋에서 단일 도구 검색에서는 nDCG@5 기준 20%의 상대적 성능 향상을, 다중 도구 검색에서는 39%의 향상을 달성했습니다.



### MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilanc (https://arxiv.org/abs/2408.01869)
Comments:
          Paper published at Machine Learning for Healthcare 2024 (MLHC'24)

- **What's New**: 이번 논문은 대형 언어 모델(LLM)을 이용한 새로운 약물 감시(Pharmacovigilance, PhV) 방법론을 소개합니다. MALADE라는 이름의 새로운 다중 에이전트 시스템을 통해 약물 라벨 데이터에서 부작용 사건(ADE)을 추출하는 방식을 제안합니다. 이 시스템은 Retrieval Augmented Generation(RAG) 기법을 사용하여 쿼리를 보강하고, 그 쿼리에 기반하여 응답을 생성합니다.

- **Technical Details**: MALADE는 일반적인 LLM-비종속적 아키텍처로, (1) 의학 문헌, 약물 라벨, FDA 도구(OpenFDA drug information API) 등 다양한 외부 소스를 활용하고, (2) 약물-결과 연관성을 구조화된 형식으로 추출하여 연관성의 강도를 제시하며, (3) 확인된 연관성에 대한 설명을 제공하는 독특한 기능을 갖추고 있습니다. 시스템은 Langroid 다중 에이전트 LLM 프레임워크를 활용하며, GPT-4 Turbo 또는 GPT-4o와 FDA 약물 라벨 데이터를 사용하여 구현되었습니다.

- **Performance Highlights**: 실험 결과, OMOP Ground Truth 테이블과 비교하여 약물 라벨 데이터의 ADE 추출에서 ROC 곡선 아래 면적(AUC) 0.90의 성능을 보여주었습니다. 이는 현재까지의 최신 방법 중 가장 높은 성능을 기록한 것입니다. 이 시스템은 단순히 이진 레이블을 제공하는 것이 아니라, 연관성의 강도 및 부작용의 희귀도 등을 포함한 구조화된 점수를 제시합니다.



### Efficient Solutions For An Intriguing Failure of LLMs: Long Context Window Does Not Mean LLMs Can Analyze Long Sequences Flawlessly (https://arxiv.org/abs/2408.01866)
Comments:
          11 pages, 5 figures, 6 tables

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 긴 입력 시퀀스를 처리하는 데 취약하다는 새로운 한계를 밝혔습니다. 감정 분석과 뉴스 분류 작업을 위해 Claude 3, Gemini Pro, GPT 3.5 Turbo, Llama 3 Instruct, Mistral Instruct 모델을 사용하여 이 문제를 조사했습니다. 이를 해결하기 위해 추출적 요약 및 선택적 자르기(summary and truncation)를 사용하는 특수 해결책을 제안하여 LLM의 성능을 최대 50% 향상시키고 API 비용을 최대 93%, 지연 시간을 최대 50% 줄였습니다.

- **Technical Details**: 이 연구는 TextRank 알고리즘을 사용하여 문서에서 중요한 문장을 추출하고 TF-IDF 및 코사인 유사성을 사용하여 정보 다양성을 극대화하는 방법을 통해 요약된 입력을 생성합니다. 7가지 프롬프트 전략을 사용하여 LLM의 성능을 실험했습니다. 여기에는 전체 컨텍스트, 전체 컨텍스트 + 요약, 첫 문장들, 마지막 문장들, 요약, 다양한 요약, 무작위 선택이 포함됩니다.

- **Performance Highlights**: 실험 결과 전체 컨텍스트나 전체 컨텍스트 + 요약 방식은 예측 성능이 저조했으나, 특정 문장을 선별하여 프롬프트로 제공하는 방식은 더 나은 성능을 보였습니다. 특히 '마지막 문장들' 시나리오는 손실 지표에서 가장 좋은 성능을 나타냈고, '다양한 요약' 방식은 정확도에서 50% 향상되었습니다. 이러한 방식은 다양한 뉴스 분류 데이터셋에서도 유사한 성능 향상을 보였습니다.



### S\'olo Esc\'uchame: Spanish Emotional Accompaniment Chatbo (https://arxiv.org/abs/2408.01852)
Comments:
          Accepted at the 23rd Mexican International Conference on Artificial Intelligence (MICAI) 2024

- **What's New**: 새로운 오픈 소스 스페인어 감정 지원 챗봇 'Sólo Escúchame'가 도입되었습니다. 이는 HEAR(Hispanic Emotional Accompaniment Responses) 데이터셋 기반으로 개발되었으며, 심리적 지원을 위한 최초의 오픈 소스 챗봇입니다.

- **Technical Details**: Sólo Escúchame는 LLaMA-2-7b-Chat을 기반으로 하고 있으며, HEAR 데이터셋은 다양한 영어 소스에서 번역된 스페인어 데이터로 구성되었습니다. 이 시스템은 GPT-3.5-Turbo를 사용하여 생성된 데이터를 포함하고 있으며, 반자동 평가 메트릭을 도입하여 모델의 성능을 측정합니다.

- **Performance Highlights**: Sólo Escúchame는 심리적 지원을 제공하는 데 있어 현존하는 최첨단 모델들을 능가하는 성능을 보여주었습니다. 모델과 데이터셋은 공개되어 있으며, 재현성과 접근성을 위해 자유롭게 사용할 수 있습니다.



### Tracking Emotional Dynamics in Chat Conversations: A Hybrid Approach using DistilBERT and Emoji Sentiment Analysis (https://arxiv.org/abs/2408.01838)
Comments:
          This work has been submitted to the Springer journal for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 새 논문은 텍스트와 이모지 감정 분석을 병합한 하이브리드 접근법(hybrid approach)으로 채팅 대화의 감정 변동을 추적하는 방법을 제안합니다. 전통적인 기계 학습 알고리즘들과 DistilBERT 기반의 텍스트 감정 인식 성능을 비교 분석한 결과, DistilBERT의 우수한 성능이 확인되었습니다.

- **Technical Details**: 이 연구는 두 가지 데이터셋을 사용하여 감정을 분석하였습니다. 첫 번째 데이터셋은 영어 트위터 메시지에서 추출된 데이터로, 6개의 기본 감정을 포함합니다. 두 번째 데이터셋은 Kaggle에서 수집된 13개의 감정으로 주석이 달린 트위터 데이터입니다. 이 데이터셋들은 감정 표현의 불균형적 특성을 보입니다. 또한, Emoji Sentiment Ranking 데이터셋을 사용하여 다양한 이모지의 표준화된 감정 점수를 분석했습니다. 각 이모지의 긍정, 중립, 부정 감정 점수를 평가하여 감정 변화를 추적합니다.

- **Performance Highlights**: 새로운 감정 추적 방법은 고객 서비스, 업무 채팅, 소셜 미디어 상호작용 등에서 실용적으로 적용될 수 있습니다. DistilBERT 모델은 다른 기계 학습 알고리즘들, 예를 들어 SVM, Random Forest, AdaBoost와 비교하여 뛰어난 성능을 보였으며, 텍스트와 이모지 분석을 결합하여 감정 분석의 정확성을 높였습니다.



### MathLearner: A Large Language Model Agent Framework for Learning to Solve Mathematical Problems (https://arxiv.org/abs/2408.01779)
- **What's New**: 최근 AI와 자연어처리(NLP) 기술의 발전에 따라, 더 발전된 언어 모델의 수요가 증가하고 있습니다. 특히 큰 언어 모델(LLM)이 다양한 복잡한 작업을 자동화하는 데 뛰어난 능력을 보여주었습니다. 그러나 수학적 추론에서는 아직 한계가 있다는 문제가 있습니다. 이를 해결하기 위해 저희는 사람의 귀납적 추론 과정을 모방한 에이전트 프레임워크를 제안했습니다. MathLearner는 LLM의 수학적 추론 능력을 크게 향상시키며, 교육 및 실생활에서도 유용한 어플리케이션이 될 것입니다.

- **Technical Details**: MathLearner 프레임워크는 세 가지 주요 단계로 작동합니다. 첫째, 다양한 수학 문제와 그 해결 방안을 학습하여 예제에서 배우는 것입니다. 둘째, 다양한 문제 해결 방법과 기술을 기억하여 다양한 유형의 수학 문제를 체계적이고 구조적으로 해결할 수 있도록 합니다. 셋째, 이전에 배운 지식을 신속하게 기억하고 적용하여 새로운 수학 문제를 해결합니다. 이러한 접근 방식을 통해 LLM의 수학적 문제 해결 능력을 향상시키고, 효율적인 외부 지식 활용을 가능하게 합니다.

- **Performance Highlights**: MathLearner는 이전의 체인 오브 생각(CoT) 방식에 비해 전반적인 정확도를 20.96% 향상시켰으며, 기존 방식으로는 해결할 수 없었던 17.54%의 수학 문제를 해결할 수 있었습니다. 효율적인 RETRIEVAL 방법을 통해 모델이 외부 지식을 효율적으로 활용할 수 있게 함으로써 수학적 계산이 서면 절차를 기반으로 수행될 수 있도록 했습니다. 또한, MathLearner는 개인 맞춤형 학습 지원 도구로서 교육 자원 불균형을 완화하는 데 기여할 수 있습니다.



### Discovery of Rare Causal Knowledge from Financial Statement Summaries (https://arxiv.org/abs/2408.01748)
- **What's New**: 이번 논문에서 저자들은 일본의 재무제표 요약본에서 희귀한 인과적 지식을 추출하는 방법을 제안합니다. 이는 금융 시장에서 인공지능을 활용하여 투자 결정을 지원하는 기술의 발전과 관련이 있습니다.

- **Technical Details**: 이 방법은 3단계로 구성됩니다. 첫째, 확장된 언어 온톨로지를 기반으로 기계 학습 방법을 사용하여 인과적 지식을 포함하는 문장을 추출합니다(Extract sentences that include causal knowledge). 둘째, 구문 패턴을 사용하여 추출된 문장에서 인과적 지식을 얻습니다(Obtain causal knowledge from extracted sentences). 마지막으로, 얻어진 지식 중 희귀한 인과적 지식을 추출합니다(Extract rarest causal knowledge). 이 과정에서 지원 벡터 머신(SVM)을 사용하여 구문과 의미적 특징을 기반으로 한 문장을 분석합니다.

- **Performance Highlights**: 이 방법은 일본어 재무제표 요약본에서 인과적 지식을 추출하는 데 있어 높은 정확성과 회수율을 보이며, 특히 잘 알려지지 않은 희귀한 인과적 지식을 효과적으로 식별해냅니다. 이는 개별 투자자들에게 중요한 투자 기회를 제공할 수 있습니다.



### Indexing and Visualization of Climate Change Narratives Using BERT and Causal Extraction (https://arxiv.org/abs/2408.01745)
- **What's New**: 이 연구에서는 일명 '기후 변화 내러티브(climate change narratives)'를 추출하고 인덱싱하며 시각화하는 방법론을 제안합니다. BERT (Bidirectional Encoder Representations from Transformers)와 인과 추출(causal extraction)이라는 두 가지 자연어 처리 방법을 사용하여 기후 변화 관련 신문 기사를 텍스트 분석하고, 내러티브를 추출합니다.

- **Technical Details**: BERT는 딥러닝 기반의 범용 언어 모델로, 텍스트에서 중요한 정보를 추출하고 문장의 주제를 파악하는 데 사용됩니다. 인과 추출은 텍스트에서 인과 관계를 포함한 문장을 검색하고, 원인과 결과 간의 논리적 관계를 나타내는 표현 쌍을 추출하는 방법입니다. 이 두 방법을 결합하여 여러 시점과 주제에 걸친 원인과 결과 관계를 찾고, 이를 '기후 변화 내러티브'로 추출합니다.

- **Performance Highlights**: ['국내 환경 및 에너지 정책과 관련된 기후 변화 내러티브는 주요 국제 기후 변화 회의에서 다자간 합의가 이루어질 때 증가하는 경향이 있습니다.', '2018년 이후, 국내 환경 및 에너지 정책에서 기업의 행동 및 금융 시장으로의 인과 연결에 초점을 맞춘 기후 변화 내러티브가 현저히 증가했습니다.', '국제 회의와 국내 정책, 통화 정책 및 기업 신뢰도 간의 새로운 기후 변화 내러티브가 등장하고 있습니다.', '자연 재해와 관련된 기후 변화 내러티브가 최근 증가했습니다. 이는 더 빈번해지고 심각해진 자연 재해가 경제 활동에 미치는 부정적 영향이 새로운 위험 요소로 인식되고 있음을 나타냅니다.']



### Summarization of Investment Reports Using Pre-trained Mod (https://arxiv.org/abs/2408.01744)
- **What's New**: 최근 언어 처리 분야에서 Transformer 기반 모델을 이용한 투자 보고 요약 자동화가 주목받고 있습니다. 이 연구는 투자 보고를 월별 보고에서 요약하는 문제에 대해 다룹니다. 특히, 추출 요약(Extractive Summarization)과 생성 요약(Abstractive Summarization)의 두 가지 방법을 비교하여 어느 방법이 더 효과적인지 분석합니다.

- **Technical Details**: 연구에서는 두 가지 요약 방법론을 사용했습니다. 추출 요약에서는 TFIDF 기반 방법과 BERT 기반 방법이 사용되었고, 생성 요약에서는 T5 모델을 사용했습니다. 추출 요약의 경우, 문장을 벡터화한 뒤 코사인 유사도를 통해 요약에 포함될 문장을 선별합니다. 생성 요약은 T5 모델을 사용하여 입력 데이터를 압축해 새로운 요약 문장을 생성합니다.

- **Performance Highlights**: 실험 결과, 생성 요약 방법이 ROUGE-1, ROUGE-2, ROUGE-L 점수에서 모두 뛰어난 성능을 보였습니다. 특히 ROUGE-2와 ROUGE-L에서 큰 차이를 보였으며, 이는 생성 요약이 투자 보고 요약에 적합함을 나타냅니다. 한편, 추출 요약 방법 중에서는 TFIDF 기반 방법이 BERT 기반 방법보다 더 높은 정확도를 보였습니다.



### Multi-Frame Vision-Language Model for Long-form Reasoning in Driver Behavior Analysis (https://arxiv.org/abs/2408.01682)
Comments:
          On-going work

- **What's New**: 이 논문은 대시캠(dashcam) 영상을 기반으로 상업 운전자를 코칭할 수 있는 새로운 멀티모달(multimodal) 데이터셋과 운전 코칭 추론 시스템을 소개합니다. 현재 북미 대시캠 시장은 2022년에서 2027년까지 연평균 성장률(CAGR) 15.4%를 기록할 것으로 예상됩니다. 이 연구는 대규모 비전-언어 모델(Large-scale Vision Language Models, LVLMs)을 적용하여 운전자 행동을 분석하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구에서는 `Multi-Frame Vision-Language Model for Reasoning in Driver Behavior Analysis`라는 모델을 제안합니다. 이 모델은 도로를 향한 카메라와 운전자를 향한 카메라에서 촬영된 영상을 모두 활용하여 운전 상황을 포괄적으로 분석합니다. 두 종류의 카메라를 사용하여 외부 조건과 운전자의 반응을 모두 식별하며, 이를 통해 운행 위험 상황을 효과적으로 이해할 수 있게 합니다. 이 모델은 Video-LLaMA 프레임워크를 기본으로 하며, 비주얼 인코더는 BLIP-2를, 오디오 인코더는 ImageBind를 사용합니다.

- **Performance Highlights**: 모델 훈련에는 로드-페이싱과 드라이버-페이싱 카메라의 동기화된 RGB 비디오 데이터셋이 사용되었습니다. 모델은 NVIDIA A100 (80GB) GPU 8개로 훈련되며, Visual Encoder와 Large Language Model(LLM)의 가중치는 동결시키고 비디오 Qformer의 가중치를 업데이트하는 방식으로 파인 튜닝(fine-tuning)되었습니다. 결과적으로, 모델은 운전 상황에 대한 포괄적 이해와 정교한 코칭 지침 생성 능력을 크게 향상시켰습니다.



### MMPKUBase: A Comprehensive and High-quality Chinese Multi-modal Knowledge Graph (https://arxiv.org/abs/2408.01679)
- **What's New**: 이 논문은 MMPKUBase라는 중국어 다중 모달 지식 그래프를 소개합니다. 이 그래프는 새, 포유류, 고사리 등 다양한 도메인을 다루며, 50,000개 이상의 엔티티와 100만 개 이상의 필터링된 이미지를 포함하고 있습니다. 이는 특히 고품질 중국어 지식 그래프의 부족과 다중 모달 지식 그래프의 도메인 커버리지 한계를 극복하는 데 중점을 두고 있습니다.

- **Technical Details**: MMPKUBase를 구축하기 위해 Prototypical Contrastive Learning과 Isolation Forest 알고리즘을 사용하여 이미지 데이터를 세밀하게 정제했습니다. Prototypical Contrastive Learning은 이미지 특징을 추출하는 데 사용되며, Isolation Forest는 품질 관리를 위해 이미지 데이터를 필터링하는 데 사용됩니다. 이 시스템은 1,227,013개의 고품질 이미지를 유지하며, 사용자가 쉽게 이미지 속성을 탐색할 수 있는 사용자 친화적인 플랫폼도 개발했습니다.

- **Performance Highlights**: MMPKUBase는 52,180개의 엔티티와 1,542,894개의 이미지를 포함하며, 이 중 1,227,013개의 이미지는 고품질로 필터링되었습니다. 다양한 도메인(새, 포유류, 고사리, 건축물, 유적지, 자동차 및 군사 관련 데이터)을 포괄하며, 이는 시각적 질문 응답 시스템과 추천 엔진 등의 다양한 작업에서 중요한 역할을 할 수 있습니다.



### Transforming Slot Schema Induction with Generative Dialogue State Inferenc (https://arxiv.org/abs/2408.01638)
Comments:
          Accepted to SIGDIAL 2024

- **What's New**: 이번 연구에서는 태스크 지향 대화 시스템(task-oriented dialogue system)을 위한 슬롯 스키마(slot schema)를 자동으로 유도하는 Slot Schema Induction(SSI) 방법을 제안합니다. 기존에는 대화 텍스트에서 직접 추출된 값 스팬(value spans)을 클러스터링해서 슬롯을 유도했지만, 본 연구에서는 생성 모델(generative approach)을 사용하여 대화 상태를 요약하는 슬롯 이름과 값을 생성하는 방식으로 더 높은 품질의 정보를 발견하는 방법을 제시합니다.

- **Technical Details**: 제안된 Generative Dialogue State Inference(GenDSI) 방법은 세 단계로 슬롯 스키마를 유도합니다. 첫째, 대화 상태 생성기(dialogue state generator)를 사용하여 각 대화 턴(turn)의 값 후보(value candidate)와 해당 슬롯 이름을 예측합니다. 둘째, 인코딩 모델을 통해 각 슬롯-값 후보를 밀집 벡터(dense vector)로 표현합니다. 셋째, 클러스터링 알고리즘(HDBSCAN)을 적용하여 후보 값을 통합된 슬롯 클러스터로 그룹화합니다. 대화 상태 생성기는 sequence-to-sequence 방식으로 작동하며, 사전 학습된 인코더-디코더(transformer) 모델을 태스크 지향 대화 데이터에 맞게 파인튜닝하여 다양한 도메인에서 중요한 대화 상태 정보를 발견하도록 합니다.

- **Performance Highlights**: MultiWOZ와 SGD 데이터셋에서 실험한 결과, 제안된 GenDSI 방법은 기존의 최고 성능(state-of-the-art) 방법들보다 여러 측면에서 더 우수한 성능을 보였습니다. 특히 생성된 슬롯-값 후보가 인간이 작성한 스키마와 잘 일치하며, 자동으로 슬롯 클러스터에 이름을 붙이는 등의 이점이 있음을 확인했습니다.



### Dialog Flow Induction for Constrainable LLM-Based Chatbots (https://arxiv.org/abs/2408.01623)
Comments:
          Accepted at SIGDIAL 2024

- **What's New**: 이 논문은 LLM 기반의 챗봇들이 전문 도메인 내에서 정확한 정보를 제공하면서 도메인 내 대화 흐름을 유지하는 것이 어려운 문제를 해결하기 위한 무감독 접근법을 제안합니다. 이 접근법은 다양한 도메인에서 사용할 수 있는 도메인 특화 대화 흐름을 자동으로 생성합니다.

- **Technical Details**: 제안된 방법은 두 가지 변형의 대화 흐름을 도입합니다: 도메인 대화 인스턴스가 없는 경우의 'intrinsic flows'와 도메인 대화 인스턴스가 있는 경우의 'data-guided flows'. GPT-4의 지식을 활용하여 초기에 대화 흐름을 생성하고, 사전 정의된 기준을 기반으로 자체 평가를 통해 개선하는 프로세스를 구축합니다. 데이터가 있는 경우에는 도메인 대화 인스턴스를 사용하여 대화 흐름을 실세계 대화 패턴에 맞추어 조정합니다.

- **Performance Highlights**: 인간 및 자동 평가를 통해 여러 대화 도메인에서 이 접근법이 높은 품질의 데이터 기반 대화 흐름을 생성할 수 있음을 입증했습니다. 이로 인해 광범위한 수작업 없이도 더 나은 도메인 커버리지를 달성할 수 있었습니다.



### Analyzing LLMs' Capabilities to Establish Implicit User Sentiment of Software Desirability (https://arxiv.org/abs/2408.01527)
Comments:
          6 pages, 2 figures, 2 tables

- **What's New**: 이번 연구는 사용자가 표현하는 암묵적인 소프트웨어 바람직성을 정량적으로 분석하기 위해 여러 LLMs (Large Language Models)를 사용했습니다. 이 연구는 기존의 긍정, 중립, 부정으로 분류하는 방법 대신 스케일된 숫자 감정 분석을 제공합니다. 이는 제품 바람직성에 관한 더 나은 결정을 내리기 위해 감정의 강도를 더 깊이 있게 이해할 수 있게 합니다. 데이터는 Microsoft Product Desirability Toolkit (PDT)을 사용하여 수집되었으며, ZORQ라는 학부 컴퓨터 과학 교육에 사용되는 게이미피케이션 시스템의 사용자로부터 데이터를 수집했습니다.

- **Technical Details**: PDT 데이터를 분석하기 위해 여러 LLMs (Claude Sonnet 3, GPT4, GPT4o, 등)을 사용했습니다. 또한, Twitter-Roberta-Base-Sentiment (TRBS)와 Vader도 감정 분석 방법으로 사용되었습니다. 각 시스템은 PDT의 단어/설명 쌍과 사용자가 선택한 5개의 단어와 설명 전체를 보고 감정을 분석하도록 요청되었습니다. LLMs는 감정 점수뿐만 아니라 자신들의 신뢰도 (낮음, 중간, 높음)와 그 이유도 제공했습니다. 모든 LLMs는 사용자의 그룹화된 데이터에서 사용자 감정을 통계적으로 감지할 수 있었지만, TRBS와 Vader는 그렇지 못했습니다.

- **Performance Highlights**: LLMs는 사용자 그룹화된 데이터에서 감정을 통계적으로 감지할 수 있었지만, TRBS와 Vader는 그렇지 못했습니다. 이는 LLMs가 암묵적인 사용자 감정을 이해하는 데 더 효과적임을 시사합니다. 또한, LLMs의 신뢰도와 그 이유를 설명하는 기능은 사용자 감정을 더 잘 이해하는 데 도움이 되었습니다.



### MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts (https://arxiv.org/abs/2408.01505)
- **What's New**: 아카이브 게시물에서 소개된 새로운 방법론인 Mixture of Dyadic Experts (MoDE)는 Low-Rank Adaptation (LoRA) 기법을 활용한 여러 작업의 적응을 위한 새로운 접근 방식을 제안합니다. 이 방법은 공유되며 낮은 차원의 표현을 생성하는 다운 프로젝션 매트릭스를 도입하고, 원자 수준의 Rank-One 어댑터 (Atomic Rank-One Adapters)를 사용하여 더 정교한 작업 수준의 특수화를 수행합니다.

- **Technical Details**: MoDE는 기존의 Low-Rank Adaptation (LoRA) 및 mixture-of-experts (MoE) 설계를 기반으로 하여 확장된 구조입니다. 전통적인 LoRA와 달리, MoDE는 모든 작업에 걸쳐 동일한 다운 프로젝션 매트릭스를 공유하는 방식을 택하여 매개 변수의 중복을 줄이고, 고유 작업 특성을 포착할 수 있도록 Rank-One 어댑터와 정교한 라우팅 메커니즘을 결합합니다.

- **Performance Highlights**: Supernatural Instructions (SNI) 벤치마크에서 MoDE는 최첨단 다중 작업 Parameter-Efficient Fine-Tuning (PEFT) 기법들을 능가하는 성능을 기록했습니다. 특히 추가 매개 변수 없이도 더 나은 효율성과 성능을 보여주어 실제 애플리케이션에서 다중 작업을 처리하는 고성능 경량 모델을 구현할 수 있습니다.



### Artificial Intelligence for Public Health Surveillance in Africa: Applications and Opportunities (https://arxiv.org/abs/2408.02575)
- **What's New**: 최근 발표된 연구에서는 인공지능(AI)이 아프리카 공공 건강 감시 분야에서 어떻게 혁신적인 역할을 할 수 있는지를 조사했습니다. 이 논문은 AI 기술을 통해 질병 모니터링을 향상시키고 공공 건강 개입을 효과적으로 지원할 수 있는 가능성을 강조합니다. 아프리카의 제한된 자원, 불충분한 인프라, 실패한 건강 정보 시스템, 그리고 숙련된 건강 전문가의 부족과 같은 도전 과제를 AI가 어떻게 극복할 수 있는지를 탐구하고 있습니다.

- **Technical Details**: 이 논문은 다양한 전자 건강 기록(Electronic Health Records), 소셜 미디어, 환경 센서, 게놈 데이터 등의 대규모 데이터를 분석하는 AI의 능력을 강조합니다. 기계 학습(Machine Learning) 기술, 특히 랜덤 포레스트(Random Forests), 서포트 벡터 머신(Support Vector Machines), 딥러닝(Deep Learning) 모델들이 전통적인 방법에 비해 더 빠르고 정확한 예측을 제공하는 데 성공적임을 보여줍니다. 특수 질병 감지 및 예측에 관련한 사례 연구, 예를 들어 결핵(Tuberculosis), HIV/AIDS, 콜레라, 에볼라 바이러스 예측 모델을 포함했습니다.

- **Performance Highlights**: AI는 질병 감지 및 예측의 정확도와 적시성을 크게 향상시키고, 자원 할당을 최적화하며, 타겟된 공공 건강 전략을 수립할 수 있도록 돕습니다. 예를 들어, AI 기반 결핵 진단 시스템은 초기 발견과 효과적인 치료를 가능하게 하여 아프리카의 높은 결핵 부담 문제를 해결하는 데 중요한 역할을 합니다. 그리고 HIV 감지 및 예측에서 AI는 특히 약물 내성 돌연변이를 예측하는 데 높은 정확도를 보였습니다.



### From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Futur (https://arxiv.org/abs/2408.02479)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 소프트웨어 공학(SE)에서의 활용과 LLM 기반 에이전트에 대한 현재의 연구 동향과 해결책을 조사한 첫 번째 종합 설문 조사입니다. 특히 요구사항 엔지니어링, 코드 생성, 자율 의사 결정을 비롯한 6개의 주요 주제에 대해 다루며, LLM과 LLM 기반 에이전트의 차이점과 유사점을 분석합니다.

- **Technical Details**: 이 연구는 LLM과 LLM 기반 에이전트가 소프트웨어 공학(SE)에서 다룰 수 있는 다양한 작업, 벤치마크 및 평가 기준을 조사합니다. 중요한 기술적 키워드로는 Retrieval-Augmented Generation (RAG), prompt engineering, Chain-of-Thought (COT)가 있습니다. 이 논문은 SE에서 LLM과 LLM 기반 에이전트의 역량을 명확히 구분하고 비교하는 것을 목표로 합니다.

- **Performance Highlights**: 이 논문에서는 LLM 기반 에이전트가 자율 디버깅, 코드 리팩토링, 적응형 테스트 생성과 같은 복잡하고 동적인 작업을 처리하는 데 있어 전통적인 LLM보다 뛰어난 성능을 보인다고 강조합니다. 또한 LLM 기반 에이전트가 SE에서 인공지능의 일반 지능(AGI) 접근에 가까운 성능을 보여준다고 합니다.



### An approach to optimize inference of the DIART speaker diarization pipelin (https://arxiv.org/abs/2408.02341)
Comments:
          6 pages, 3 figures

- **What's New**: 
- 본 논문은 DIART 파이프라인의 온라인 화자 분할 (online speaker diarization) 시스템의 추론 지연 시간을 최적화하는 방법을 탐구합니다.
- 논문은 여러 추론 최적화 기법들을 pyannote/embedding 모델에 적용하여 지연 시간에 미치는 영향을 조사합니다.

- **Technical Details**: 
- 주요 최적화 기법으로는 지식 증류 (knowledge distillation), 가지치기 (pruning), 양자화 (quantization), 레이어 (layer) 융합을 포함합니다.
- 양자화와 레이어 융합은 정확도가 악화되지 않으면서도 지연 시간을 개선하는 반면, 가지치기는 지연 시간 개선에 별다른 효과가 없었습니다.
- 지식 증류는 지연 시간을 최적화하지만, 정확도에 부정적인 영향을 미쳤습니다.

- **Performance Highlights**: 
- DIART 파이프라인의 음성 임베딩 모델인 pyannote/embedding의 추론 지연 시간을 최적화하기 위해 여러 기법을 적용했을 때 얻은 결과를 논의합니다.
- 양자화와 레이어 융합을 통해 지연 시간 감소를 유지하면서도 정확도를 유지할 수 있는 가능성을 확인했습니다.



### Developing PUGG for Polish: A Modern Approach to KBQA, MRC, and IR Dataset Construction (https://arxiv.org/abs/2408.02337)
Comments:
          Accepted for ACL 2024 (findings)

- **What's New**: AI와 자연어 처리의 발전은 기계와 인간 간의 언어 상호작용을 혁신적으로 개선했습니다. 특히, Knowledge Base Question Answering (KBQA) 시스템은 구조화된 지식 그래프(KG)를 활용하여 광범위하고 지식 집약적인 질문을 처리할 수 있습니다. 그러나 저자원(low-resource) 언어를 위한 KBQA 데이터셋은 여전히 부족합니다. 이를 해결하기 위해, 최신 도구인 Large Language Models(LLM)를 활용한 현대적이고 반자동화된 데이터셋 구축 방법론이 도입되었습니다. 이를 통해, 최초의 폴란드어 KBQA 데이터셋인 PUGG 데이터셋이 만들어졌으며, 기계 읽기 이해(MRC)와 정보 검색(IR) 데이터셋도 새로이 소개되었습니다.

- **Technical Details**: 기존의 KBQA 데이터셋 구축 방식은 구식이며 인적 자원이 많이 필요했으나, 본 연구팀은 이 문제를 해결하기 위해 LLM을 활용한 반자동화된 데이터셋 구축 파이프라인을 설계, 구현 및 실행하였습니다. 이는 특별히 저자원 환경에 맞춰진 것으로, Wikidata를 KG로 선택하여 다국어 커버리지와 동적, 개방적 특성을 활용했습니다. 또한, 데이터 변환 과정에서 자연스러운 언어와 단순 템플릿 기반 질문을 결합하여 데이터셋의 복잡성을 다루었습니다.

- **Performance Highlights**: PUGG 데이터셋은 폴란드어 KBQA, MRC, IR 과제들을 포함하며, 첫 번째 폴란드어 KBQA 리소스로 자리매김했습니다. 데이터셋은 자연어로 된 사실 질문(factoid questions)을 특징으로 하며, 보다 효율적이고 긴밀한 데이터셋 생성을 위해 반자동화된 파이프라인을 사용했습니다. 이 파이프라인은 인간 주석 작업을 크게 줄이는 데 기여했습니다. 뿐만 아니라, 파이프라인 구현과 데이터셋 구축 과정에서 얻은 통계 및 통찰을 자세히 제공하였으며, 다양한 컨텍스트에서 유용한 엔티티 연계(custom methods)와 같은 유틸리티 메서드를 개발했습니다. 마지막으로, 기존 모델의 평가를 제공하여 향후 연구를 위한 벤치마크를 설정했습니다.



### Spin glass model of in-context learning (https://arxiv.org/abs/2408.02288)
Comments:
          8 pages, 4 figures

- **What's New**: 최근 연구에서 대형 언어 모델(LLM)들이 학습하지 않은 쿼리에 대해 프롬프트만을 사용하여 예측할 수 있는 인컨텍스트 학습(In-context learning) 능력을 보여주고 있습니다. 이 현상의 기계적 해석과 물리학적 연결은 아직 미해결된 문제로 남아 있습니다. 본 논문에서는 선형 주의를 사용하는 간단한 트랜스포머 모델을 스핀 글래스(spin glass) 모델로 매핑하여 이러한 현상을 설명합니다. 이를 통해 학습되지 않은 기능을 제공된 프롬프트만으로도 예측할 수 있는 이유를 밝혀냅니다.

- **Technical Details**: 이 연구는 선형 주의를 사용하는 단층 트랜스포머 구조를 고려합니다. 주어진 입력 시퀀스를 출력 시퀀스로 변환하는 자가 주의(self-attention) 메커니즘을 이용합니다. 이 트랜스포머 구조를 스핀 글래스 모델로 재구성하여, 실수 값을 가지는 스핀과 데이터 내의 내재적 무질서(disorder)를 설명합니다. 여기서 트랜스포머의 파라미터는 스핀으로 변환되고, 입력 시퀀스는 냉동된 무질서(quenched disorder)로 작용하여 스핀들이 상호작용하도록 합니다. 이를 통해 ICL 오류를 줄입니다.

- **Performance Highlights**: 이론적으로, 단일 인스턴스 학습(single instance learning)에서 과제의 다양성을 증가시키면 인컨텍스트 학습이 유발된다는 점을 밝힙니다. 이는 볼츠만 분포(Boltzmann distribution)가 유일한 올바른 솔루션으로 수렴함으로써 가능해집니다. 즉, 다양한 사전 훈련 과제를 통해 트랜스포머가 새로운 프롬프트 설정에서 예측력을 발휘할 수 있게 되는 것입니다. 이 스핀 글래스 모델은 대형 언어 모델의 경험적 성공을 이해하는 데 중요한 기초를 제공합니다.



### COM Kitchens: An Unedited Overhead-view Video Dataset as a Vision-Language Benchmark (https://arxiv.org/abs/2408.02272)
Comments:
          ECCV2024 accepted

- **What's New**: 최근 비전과 언어 커뮤니티에서 절차적 비디오 이해에 대한 관심이 증가하는 가운데, 효율적인 데이터 수집이 중요해지고 있습니다. 이를 해결하기 위해 새로운 데이터셋 'COM Kitchens'를 제안합니다. 이 데이터셋은 스마트폰을 사용하여 오버헤드 뷰에서 촬영된 음식 준비 과정을 담고 있습니다. 이 데이터셋을 통해 새로운 온라인 레시피 검색(OnRR)과 오버헤드 뷰 비디오 캡션링(DVC-OV) 과제를 소개합니다.

- **Technical Details**: COM Kitchens 데이터셋은 스마트폰의 넓은 시야각 렌즈를 활용하여 부엌의 전체 작업대를 오버헤드 뷰로 촬영했습니다. 이 셋업으로 다양한 환경에서 145개의 비디오, 총 40시간 분량의 데이터를 수집했습니다. 비디오와 텍스트 지시사항을 작업 흐름 그래프로 연결하는 수동 주석을 제공하여 비디오-텍스트 검색과 밀집 비디오 캡션링(DVC) 과제를 정의했습니다.

- **Performance Highlights**: COM Kitchens 데이터셋을 통해 현재 웹 비디오 기반의 SOTA(State-of-the-Art) 방법들이 이러한 새로운 과제를 어떻게 처리하는지 실험으로 검증했습니다. 이 데이터셋은 기존의 제조 작업을 타겟으로 한 다른 절차적 비디오 데이터셋들에 비해 더 다양한 작업과 환경을 포함하며, 언어적 주석이 포함된 유일한 데이터셋입니다. 새로운 비디오-텍스트 검색(Online Recipe Retrieval, OnRR) 과제와 밀집 비디오 캡션링(DVC-OV) 과제를 통해 스마트폰 비디오에도 적용 가능한 기술 개발을 목표로 하고 있습니다.



### Evaluating the Performance of Large Language Models for SDG Mapping (Technical Report) (https://arxiv.org/abs/2408.02201)
- **What's New**: 최근 대형 언어 모델(LLM)의 사용이 급속히 확산되고 있는 가운데, 데이터 프라이버시를 보호하고 특정 작업에 맞춘 커스터마이징이 가능한 오픈 소스 버전들도 등장하고 있습니다. 본 연구에서는 GPT-4o의 출력을 기준으로 다양한 언어 모델을 비교하며, 주요 오픈 소스 모델로 Mixtral, LLaMA 2, LLaMA 3, Gemma, Qwen2와 더불어 GPT-4o의 특수 버전인 GPT-4o-mini를 사용해 성능을 평가하였습니다.

- **Technical Details**: 본 연구는 스윈번 기술 대학교(Swinburne University of Technology) 리서치 뱅크에서 1,000개의 출판물을 샘플로 하여, 각 출판물을 SDG(Sustainable Development Goals)와 매핑하는 작업을 수행하였습니다. 평가 기준으로는 F1 스코어, 정밀도(precision), 재현율(recall) 등의 지표를 사용하였으며, 이러한 지표는 혼동 행렬(confusion matrix)을 기반으로 도출되었습니다. 실험은 Swinburne 고성능 컴퓨팅(HPC) 시설에서 NVIDIA A100 GPU가 탑재된 컴퓨팅 노드를 활용하여 진행되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, LLaMA 2와 Gemma 모델은 여전히 상당한 개선의 여지가 있는 것으로 나타났습니다. 다른 네 모델은 성능 차이가 크지 않았습니다. GPT-4o-mini는 가장 빠르게 작업을 처리하며, 비용 측면에서도 GPT-4o보다 32배 저렴한 결과를 보여주었습니다. Qwen2 모델은 가장 오랜 시간(7시간 41분)이 소요됐습니다.



### Generative Retrieval with Few-shot Indexing (https://arxiv.org/abs/2408.02152)
- **What's New**: 기존의 생성 검색(GR) 접근법의 한계를 극복하기 위해 새로운 Few-Shot GR 프레임워크가 제안되었습니다. 이 접근법에서는 몇 번의 인덱싱만으로 문서의 식별자(문서 ID) 생성을 수행하고, 대규모 사전 학습 언어 모델(LLM)의 지식을 최대한 활용하면서도 별도의 훈련이 필요 없습니다.

- **Technical Details**: Few-Shot GR는 두 주요 단계를 포함합니다. 첫 번째는 다중 매핑을 통한 Few-Shot 인덱싱이며, 두 번째는 제한된 빔 검색(constrained beam search)을 이용한 검색 단계입니다. 인덱싱 단계에서 LLM은 각 문서에 대해 여러 개의 자유 형태 문서 ID(free-text docids)를 생성해 docid 은행을 만듭니다. 검색 단계에서는 동일한 LLM을 사용해 주어진 쿼리의 docid를 생성하고, 생성된 docid를 문서와 매핑합니다.

- **Performance Highlights**: Natural Questions(NQ) 데이터셋 실험 결과, Few-Shot GR은 기존의 중무장 훈련 기반 GR 접근법보다 뛰어난 성능을 발휘하고 효율성이 높습니다. 중요한 성공 요인으로는 다중 매핑을 포함한 Few-Shot 인덱싱과 효과적인 LLM 선택이 언급되었습니다.



### Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models (https://arxiv.org/abs/2408.02085)
Comments:
          review, survey, 28 pages, 2 figures, 4 tables

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 instruction tuning을 목적으로 활용 가능한 데이터 평가 및 선택 방법에 관한 종합적인 리뷰를 제공합니다. 기존 연구들이 데이터 평가 메트릭스 및 선택 메커니즘에 대해 구체적으로 다루지 않은 부분을 보완하고, 데이터 평가가 instruction tuning에 어떻게 통합될 수 있는지를 설명합니다.

- **Technical Details**: 데이터 평가 및 선택 방법을 크게 세 가지 관점에서 체계적으로 분류합니다: quality-based, diversity-based, importance-based 방법들입니다. 각 카테고리 내 대표적인 방법들을 상세히 기술하고, 최신 방법들의 공식 보고 결과를 바탕으로 비교를 진행합니다. 또한, 다양한 데이터셋과의 연관성을 분석하고, 커다란 데이터셋을 평가하고 일부 선택할 때 필요한 방법론들을 설명합니다.

- **Performance Highlights**: 이 논문은 다양한 데이터 평가 방법들이 실제 성능에 미치는 영향을 심층적으로 논의하며, 그 한계점들을 명확히 제시합니다. 특히, quality, diversity, importance 측면에서 데이터를 선별하여 학습 비용을 줄이면서 성능을 높일 수 있는 방안을 제시합니다. 이를 통해, 효과적이고 효율적인 instruction tuning을 위한 데이터 평가 및 선택의 중요성을 강조합니다.



### The Implications of Open Generative Models in Human-Centered Data Science Work: A Case Study with Fact-Checking Organizations (https://arxiv.org/abs/2408.01962)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 이 논문은 개방형 생성 언어 모델(Open Generative Language Models)의 사회적 영향을 조사하고 있습니다. 특히, 팩트체킹 조직이 대규모 순환하는 잘못된 정보를 관찰하고 분석하는 데 이러한 모델을 활용하는 방법을 살펴보았습니다. 이를 위해, 여섯 대륙에 걸쳐 20개의 팩트체킹 조직에서 24명의 전문가들을 인터뷰했습니다.

- **Technical Details**: 인터뷰를 통해 팩트체킹 조직이 데이터 수집 (Data Ingestion), 데이터 분석 (Data Analysis), 데이터 검색 (Data Retrieval), 데이터 제공 (Data Delivery), 데이터 공유 (Data Sharing) 과정에서 생성 모델을 사용하는 5가지 구성 요소 개념 모델을 제안했습니다. 개방형 모델을 선호하는 이유로는 조직 자율성 (Organizational Autonomy), 데이터 프라이버시 및 소유권 (Data Privacy and Ownership), 응용 프로그램 특화 (Application Specificity), 및 역량 투명성 (Capability Transparency)를 들었으며, 폐쇄형 모델을 사용하는 이유로는 성능 (Performance), 사용성 (Usability), 안전성 (Safety) 및 기회 비용 (Opportunity Costs)를 꼽았습니다.

- **Performance Highlights**: 팩트체킹 조직이 사용하는 개방형 모델은 조직 자율성과 데이터 프라이버시 등에서 높은 선호도를 보였으나, 성능과 사용성 면에서 폐쇄형 모델에 비해 여전히 부족한 점이 많았습니다. 개방형 모델의 성능 및 안전성 향상을 위한 연구가 필요하며, 폐쇄형 모델의 투명성, 기관 및 데이터 특화에 관한 연구도 함께 제안되었습니다.



### Representation Bias of Adolescents in AI: A Bilingual, Bicultural Study (https://arxiv.org/abs/2408.01961)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 이 논문은 미국과 네팔 청소년들이 AI에 의해 어떻게 묘사되는지와 그들이 어떤 묘사를 선호하는지를 연구합니다. 연구는 기존의 static word embeddings (SWEs)와 생성형 언어 모델 (GLMs)을 통해 학습된 청소년에 대한 편견을 분석합니다. 청소년 관련 편견이 미국과 네팔에서 어떻게 상이하게 나타나는지를 비교하며, 청소년들이 실제로 자신들이 어떻게 묘사되기를 바라는지에 대해서도 논의합니다.

- **Technical Details**: 영어 SWEs는 청소년들을 사회 문제와 연관짓는 경향이 있으며, 사전학습된 GloVe SWE에서 청소년들과 가장 많이 연관된 1,000 단어 중 50% 이상이 사회 문제와 관련이 있습니다. GPT2-XL과 LLaMA-2-7B GLMs는 제공된 청소년 관련 프롬프트에 대해 각각 30%와 29% 비율로 사회 문제를 언급하는데, 주로 폭력, 약물 사용, 정신 질환, 성적 금기 등입니다. 네팔어 모델의 경우 이러한 연관성이 덜 나타납니다. 또한, 워크샵에서 미국과 네팔 청소년들은 AI의 청소년 묘사가 실제 청소년의 생활과 동떨어져 있으며, 학교와 우정 같은 활동에 더 초점을 맞춰야 한다고 언급했습니다.

- **Performance Highlights**: 연구 데이터는 미국 청소년 13명과 네팔 청소년 18명을 대상으로 워크샵을 통해 수집되었습니다. 청소년들이 AI가 자신들을 공정하게 묘사하기 위해서는 다양성을 강조하거나 긍정적인 면을 중심으로 하여야 한다고 제안했습니다. 청소년들은 AI가 미디어 소스 대신 청소년들로부터 학습한다면 편견을 줄이는 데 도움이 될 것이라고 낙관적인 전망을 제시했습니다. 이 연구는 SWEs와 GLMs가 청소년을 잘못 묘사하는 방식을 이해하는 데 도움을 주고, 덜 선정적인 특징화를 위한 템플릿을 제공합니다.



### Dataset Scale and Societal Consistency Mediate Facial Impression Bias in Vision-Language AI (https://arxiv.org/abs/2408.01959)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 본 연구에서는 43개의 CLIP (Contrastive Language-Image Pretraining) 비전-언어 모델을 분석하여 이들이 인간과 유사한 얼굴 인상 편향(facial impression biases)을 학습하는지를 평가하였습니다. 결과적으로 이러한 편향이 세 가지 구별되는 CLIP 모델 계열에서 나타남을 확인하였습니다.

- **Technical Details**: 연구에서는 비관측 시각적 속성(예: 신뢰성, 성적 지향성) 관련 인상 편향이 가장 큰 데이터셋에서 훈련된 모델들에서만 나타난다는 사실을 발견하였습니다. 이는 더 많은 데이터와의 적합성이 더 미세한 사회적 편향을 재현하는 결과를 초래한다는 것을 시사합니다. 또한, 계층적 클러스터링 방법(hierarchical clustering approach)을 통해 데이터셋의 크기가 편향 구조의 유사성 정도를 예측할 수 있음을 보여주었습니다. 마지막으로, 텍스트 인코더로써 CLIP을 사용하는 Stable Diffusion 모델들이 얼굴 인상 편향을 학습하고 있으며 이 편향이 Stable Diffusion XL-Turbo 모델에서 인종적 편향과 교차됨을 발견하였습니다.

- **Performance Highlights**: 이번 연구는 CLIP 모델들이 얼굴 인상 편향을 학습하는 메커니즘을 최초로 규명하였으며, 이를 통해 이러한 모델들이 편향 연구에 중요한 도구가 될 수 있음을 시사합니다. 다만, 해당 모델들을 범용적으로 사용하기 위해서는 데이터셋의 엄격한 큐레이션이 필요합니다.



### Why Perturbing Symbolic Music is Necessary: Fitting the Distribution of Never-used Notes through a Joint Probabilistic Diffusion Mod (https://arxiv.org/abs/2408.01950)
- **What's New**: 이번 연구에서는 Music-Diff 아키텍처를 도입하여 전통적 언어 기반 음악 생성 모델들이 간과하는 주파수 연속성을 증대시키는 새로운 방법을 제안합니다. 이 모델은 음표와 음악적 의미 정보를 결합한 확률 분포를 추정하여 조건부로 기호적 음악을 생성합니다. 특히 주파수 영역에서 Gaussian 노이즈를 주입하는 Diffusion 모델을 활용하여 '한 번도 사용되지 않은 음표'의 분포를 일반화하려는 접근이 독창적입니다.

- **Technical Details**: Music-Diff 아키텍처는 이벤트 기반 표기법(event-based notation)과 구조적 유사성 지수(SSIM)를 이용하여 조각화 모듈을 강화해 경계 흐림을 방지합니다. 또한, 다변량 섭동(multivariate perturbation)을 위한 공동 사전 학습 방법(joint pre-training)을 도입하여 음표와 음악적 의미 정보 간 진행을 구축합니다. 마지막으로 여러 노이즈 목표를 Pareto 최적화를 통해 맞추는 멀티-브랜치 디노이저를 사용하여 섭동된 음표를 복구합니다.

- **Performance Highlights**: 실험 결과, 언어 모델이나 기존의 DDPM 기반 모델과 비교했을 때, 우리 모델이 생성한 샘플은 더 높은 다양성과 구체적 일관성을 보여줍니다. 특히 리듬적 측면에서 우리의 모델은 계층적 구조를 잘 표현하였으며, 자기 유사성(self-similarity) 지표를 통해 더 나은 성능을 나타냈습니다.



### Brief state of the art in social information mining: Practical application in analysis of trends in French legislative 2024 (https://arxiv.org/abs/2408.01911)
Comments:
          in Spanish language. Keywords: social media mining, AI, ML, NLP, French elections, public opinion analysis, transformers, large language models

- **What's New**: 이 논문은 최신 소셜 미디어 마이닝(소셜 미디어 분석) 기술들을 개요적으로 설명하며, 2024년 프랑스 입법 선거에 대한 트렌드 분석에 적용했습니다. 이를 통해 Marine Le Pen이 이끄는 국민연합당이 소셜 미디어에서 높은 참여도를 유지하며 전통적인 정당들을 능가하고 있음을 밝혔습니다.

- **Technical Details**: 본 연구에서는 자연어 처리(NLP) 도구들을 활용하여 AgoraVox 플랫폼에서 댓글과 반응을 추출하고 분석하여 대중의 의견을 측정했습니다. 특히 Transformers와 대형 언어 모델(LLMs)과 같은 고급 AI 모델들이 미묘한 대중의 정서를 캡쳐하고 정치적 성향을 예측하는 데 유용함을 보여주었습니다.

- **Performance Highlights**: 연구 결과, 국민연합당이 소셜 미디어에서 높은 디지털 존재감을 유지하고 있으며, 사용자 상호작용 데이터를 통해 이를 뒷받침하였습니다. 이러한 트렌드는 실시간 평판 관리와 위기 대응에서 고급 AI 모델들의 잠재력을 증명하고 있습니다.



### STBLLM: Breaking the 1-Bit Barrier with Structured Binary LLMs (https://arxiv.org/abs/2408.01803)
- **What's New**: 이번 논문에서는 STBLLM이라는 프레임워크를 제시합니다. 이는 큰 언어 모델(LLMs)을 1비트 이하의 정밀도로 압축하는 첫 번째 구조적 이진화 방법입니다. 기존의 1비트 이진화 LLM에서 무작위로 가중치를 뒤집어도 성능 저하가 크지 않다는 점을 기반으로, N:M 스파시티 기법을 활용해 구조적 이진화를 수행합니다. 이를 통해 더 효율적인 압축을 가능하게 했습니다.

- **Technical Details**: 우리의 STBLLM은 가중치 크기와 입력 피처 노름을 고려한 표준화 중요도(SI) 메트릭을 도입하여 가중치의 중요성을 평가합니다. 또한, 각 레이어를 N:M 비율로 다르게 희소화하는 레이어별 접근 방식을 사용해 압축과 정확성의 균형을 맞춥니다. 중요한 가중치 정보 보존을 위해 이중 이진화 기반 잔여 근사법을 사용하고, 덜 중요한 가중치에 대해 세밀한 그룹화 전략을 적용하여 다양한 양자화 스킴을 사용합니다.

- **Performance Highlights**: 다양한 언어 모델(LLaMA-1/2/3, OPT, Mistral)에서 우리의 STBLLM이 더 좋은 메모리 요구사항 감소를 보여줍니다. 예를 들어, LLaMA-1-7B 모델에서 STBLLM은 0.55비트당 31.72의 퍼플렉시티를 달성하며, 이는 BiLLM의 688.73과 비교할 때 약 20배의 이득입니다. 또한, LLaMA-1-30B에서 우리의 STBLLM은 평균 51.78%의 정확도를 달성하며 BiLLM을 능가합니다.



### Integrating Large Language Models and Knowledge Graphs for Extraction and Validation of Textual Test Data (https://arxiv.org/abs/2408.01700)
Comments:
          Paper Accepted at ISWC 2024 In-Use Track

- **What's New**: 이 연구는 대형 항공우주 제조 회사가 제한된 수량의 높은 복잡성 제품을 설계, 개발, 통합, 검증, 검증함에 있어 문서화된 데이터를 추출하고 검증하는 새로운 하이브리드 방법론을 제시합니다. 이 방법론은 Knowledge Graphs (KGs)와 Large Language Models (LLMs)를 결합하여 특정 사례 연구로 위성의 전자 보드와 관련된 테스트 데이터를 분석했습니다.

- **Technical Details**: 제안된 방법론은 확장된 Semantic Sensor Network (세맨틱 센서 네트워크) 온톨로지를 사용하여 데이터의 의미를 캡처합니다. 추출된 메타데이터는 Knowledge Graph (KG)에 저장되고, 실제 테스트 결과는 Virtual Knowledge Graph (VKG)로 접근이 용이한 parquet 파일에 보관됩니다. 데이터 검증은 LLM 기반 접근법을 사용하여 구조적 및 구문적 이질성에도 불구하고 데이터를 처리합니다. SPARQL 쿼리를 통해 SQL 언어로 자동 변환할 수 있습니다.

- **Performance Highlights**: 이 연구는 상태-of-the-art LLMs의 성능을 벤치마킹하여 자동 데이터 추출 및 검증을 기존 수동 프로세스와 비교했습니다. 데이터 분석으로 인한 높은 부가가치와 비용 절감의 이점을 확인했으며, 특히 위성 전자 보드 제조에서 시간 절약과 정확성 향상을 제공하는 것으로 나타났습니다.



### Self-Emotion Blended Dialogue Generation in Social Simulation Agents (https://arxiv.org/abs/2408.01633)
Comments:
          Accepted in SIGDIAL 2024

- **What's New**: 이 논문은 가상 시뮬레이션 환경에서 대화 에이전트(Dialogue Agents)가 맥락과 상관없는 자기 감정(Self-Emotion)을 표현할 때의 대화 전략 및 의사 결정에 어떤 영향을 미치는지 탐구합니다. 연구 결과에 따르면, 자기 감정을 포함하면 에이전트가 더 인간적인 대화 전략을 구현하며, 전체적인 자연스러움 및 인간미가 향상됩니다.

- **Technical Details**: 가상 환경에서 에이전트는 일련의 이벤트를 경험하게 설정됩니다. 이 에이전트들이 시뮬레이션된 시간 동안 자기 감정 상태를 추적하며, 대화 시 이러한 자기 감정 상태와 경험된 이벤트가 일치하도록 조정됩니다. 대화 전략은 사전 정의된 11가지 전략 풀에서 선택되며, 자기 감정은 무작위 레이블(Random Label), 무작위 이벤트(Random Event), 프로필 이벤트(Profile Event) 3가지 스타일로 표현됩니다. 실험에는 GPT-4 및 FLAN-T5 모델이 활용되었고, 이들 모델을 기반으로 한 대화 데이터셋을 통해 비교 평가가 이루어졌습니다.

- **Performance Highlights**: 1. 자기 감정을 포함한 에이전트는 더 인간적인 대화 전략을 사용합니다. 2. 자기 감정을 포함한 대화는 인간 평가에서 더 자연스럽고 공감적이며 인간적인 것으로 평가되었습니다. 3. 그룹 토론 실험에서 에이전트의 자기 감정이 의사 결정에 약 50%의 변화를 가져왔습니다.



### LocalValueBench: A Collaboratively Built and Extensible Benchmark for Evaluating Localized Value Alignment and Ethical Safety in Large Language Models (https://arxiv.org/abs/2408.01460)
- **What's New**: 새로운 논문에서는 LocalValueBench 라는 확장 가능한 벤치마크를 도입하여, 대형 언어 모델(LLMs)의 호주 가치 준수 여부를 평가하고, 전 세계의 규제 기관이 현지 가치 정렬을 위해 벤치마크를 개발할 수 있는 프레임워크를 제공합니다. 이를 통해, LLM들이 각 지역의 문화적, 법적, 이념적 가치를 얼마나 잘 파악하고 있는지를 심층적으로 평가할 수 있는 방법론이 제안되었습니다.

- **Technical Details**: LocalValueBench는 윤리적 추론의 새로운 유형학(typology)과 'interrogation' 접근 방식을 이용하여 LLM의 가치 정렬을 조사합니다. 질문 작성 과정은 다양한 윤리적 시나리오와 현지 가치 고려사항을 포함시키는 것에 중점을 두었으며, 프롬프트 엔지니어링(prompt engineering) 전략을 활용해 원 질문을 제시하고, 대안적 관점을 도입하며, LLM들이 이 관점을 명확히 설명하도록 강요했습니다. 평가 기준은 현지 가치에서 벗어나는 정도를 정량화하여, 엄격한 평가를 보장합니다. 이 벤치마크를 통해, 호주 가치를 준수하는지 평가할 수 있게 되었으며, 다른 지역의 규제 기관이 자신들만의 벤치마크를 개발하는 기반이 됩니다.

- **Performance Highlights**: 상업적인 LLM의 비교 분석 결과, 각 모델의 현지 가치와 윤리적 기준 준수에 있어 상당한 차이가 나타났습니다. GPT-4는 '동성 결혼' 카테고리에서 질문에 대답을 거부하여 낮은 점수를 얻은 반면, Gemini 1.5 Pro는 여러 카테고리에서 현지 가치에 잘 맞는 성과를 보였으나 '사형' 카테고리에서 대답을 거부했습니다. 초상적 일관성 면에서 대부분의 카테고리에서 Gemini 1.5 Pro와 Claude 3 Sonet이 GPT-4를 능가했습니다. 결과적으로, 각 모델은 다양한 윤리적 시나리오에서 다른 성과를 보였으며, 지속적인 벤치마크의 개선이 필요함을 시사합니다.



### AgentPeerTalk: Empowering Students through Agentic-AI-Driven Discernment of Bullying and Joking in Peer Interactions in Schools (https://arxiv.org/abs/2408.01459)
- **What's New**: 이 연구는 학교에서 발생하는 괴롭힘과 농담을 구분하는 데 있어 대형 언어 모델(large language models, LLMs)을 활용하는 가능성을 분석했습니다. 특히 ChatGPT-4, Gemini 1.5 Pro, Claude 3 Opus 모델들이 사용되었습니다. 그 결과, ChatGPT-4가 가장 뛰어난 성과를 보였으며, 연속적이고 실시간 지원을 제공할 수 있는 잠재력이 있다고 평가되었습니다.

- **Technical Details**: 이 연구에서는 LLMs가 괴롭힘 문제를 심리적 관점에서만 다루는 것이 아니라 법적, 윤리적 관점에서도 조언을 제공할 수 있는지 여부를 조사했습니다. 이를 위해 'agentic approach'를 시뮬레이션하여 LLMs에게 외부 정보(법적 문서, 윤리적 가이드라인, 문화적 설명)를 제공하였습니다. ChatGPT-4는 이 접근법에서 뛰어난 성과를 보였으나, Gemini 1.5 Pro와 Claude 3 Opus는 혼합된 결과를 보였습니다.

- **Performance Highlights**: ChatGPT-4는 구체적인 상황에서 정확도가 크게 향상되었습니다. 예를 들어, 신체 이미지 관련 괴롭힘 시나리오에서 0.4점이 증가했습니다. Gemini 1.5 Pro와 Claude 3 Opus는 일부 시나리오에서 성과가 감소하거나 결과를 전혀 생성하지 못했습니다. 통계 분석 결과, ChatGPT-4가 가장 일관된 성과를 보였으며, Gemini와 Claude는 변동성이 더 컸습니다. ANOVA 테스트 결과 p-값이 0.0041로, 모델 간의 성과 차이가 유의미함을 확인했습니다.



### Reporting and Analysing the Environmental Impact of Language Models on the Example of Commonsense Question Answering with External Knowledg (https://arxiv.org/abs/2408.01453)
Comments:
          Presented at Bonn Sustainable AI 2023 conference

- **What's New**: 최근 연구는 거대한 언어 모델(LLM)이 환경에 미치는 영향을 조사합니다. T5 모델을 외부 지식과 결합하여 질문-응답 작업을 fine-tuning 하였으며, 이를 통해 모델의 학습 시간과 탄소 배출량을 측정했습니다. 연구 결과는 성능과 효율성을 모두 고려해야 최적의 결과를 얻을 수 있다는 점을 강조합니다.

- **Technical Details**: 이 연구는 개념넷 (ConceptNet)과 ATOMIC 같은 대규모 지식 그래프(Knowledge Graph, KG)를 T5 모델에 결합하여 비교 상식을 주입하는 방식을 채택했습니다. 또한, 상식 질문-응답 (CSQA) 데이터셋인 TellMeWhy를 이용하여 모델을 fine-tuning 했고, 그 과정에서 발생하는 환경적 영향을 분석했습니다.

- **Performance Highlights**: 연구의 결과, 작은 모델이 항상 지속 가능한 옵션이 아니며, 학습 시간이 늘어난다고 해서 성능이 항상 향상되는 것은 아님을 보여줍니다. 가장 최적의 결과는 성능과 효율성 두 측면을 모두 고려해야 달성할 수 있습니다.



### ANNA: Abstractive Text-to-Image Synthesis with Filtered News Captions (https://arxiv.org/abs/2301.02160)
Comments:
          To appear in the ACL 3rd Workshop on Advances in Language and Vision Research (ALVR), Bangkok, Thailand, August 2024, this https URL

- **What's New**: 최근 Text-to-Image(T2I) 합성 기술은 질적 향상을 중심으로 발전해왔습니다. 하지만 실제 뉴스 도메인의 이미지-캡션 쌍은 설명적이지 않은 경우가 많습니다. 본 논문에서는 다양한 맥락의 온라인 뉴스 기사에서 추출된 추상적 뉴스 캡션 데이터셋인 ANNA(Abstractive News captioNs dAtaset)를 소개합니다. 이를 통해 현재의 T2I 합성 모델들이 복잡한 뉴스 캡션을 이해하고 생성할 수 있는 능력을 평가합니다.

- **Technical Details**: ANNA 데이터셋은 약 3만개의 추상적 이미지-캡션 쌍으로 구성되어 있으며, 뉴욕 타임즈(New York Times)의 기사에서 추출되었습니다. 데이터셋 구성 과정에서 주체와 장소, 조직과 같은 명시적 엔티티 언급을 최소화하고, 보다 일반화된 시각적 요소가 포함된 이미지-캡션 쌍을 선별했습니다. 이를 통해 모델이 다양한 문장 구조와 맥락 정보를 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 현재의 오픈 소스 T2I 아키텍처는 zero-shot 및 fine-tuning 설정에서 추상적 캡션을 이해하는 데 한계가 있음을 보였습니다. 특히 전이 학습(transfer learning) 기법은 일부 성공적이었지만, 콘텐츠와 맥락 간의 관계를 일관되게 학습하는 데 실패했습니다. 평가 메트릭을 통해 생성된 이미지의 품질, 참조 이미지와의 유사성, 인간 선호도 등을 벤치마킹하여 다양한 모델의 성능을 비교했습니다.



New uploads on arXiv(cs.IR)

### Towards Coarse-grained Visual Language Navigation Task Planning Enhanced by Event Knowledge Graph (https://arxiv.org/abs/2408.02535)
Comments:
          11 pages, 6 figures

- **What's New**: 최근 연구인 EventNav는 visual language navigation (VLN)에서 coarse-grained 명령어를 효과적으로 이해하고 수행하는 방법을 제안합니다. 많은 기존 연구가 세부적인 명령어(fine-grained commands)에 초점을 맞춘 반면, EventNav는 일상 생활 시나리오에 더 적합한 추상적인 명령어(coarse-grained commands)를 고려합니다. 이를 위해 EventNav는 이벤트 지식 그래프(VLN-EventKG)와 대형 언어 모델을 활용한 협업 프레임워크를 설정하여, 추상적인 명령어만으로도 정확한 네비게이션을 수행할 수 있게 합니다.

- **Technical Details**: EventNav는 VLN-EventKG라는 이벤트 지식 그래프를 사용하여, 여러 주류 벤치마크 데이터셋에서 이벤트 지식을 추출합니다. 이 프레임워크는 작은 언어 모델과 큰 언어 모델의 협력을 통해 지식 강화 네비게이션을 수행하며, 추상적인 명령어 입력을 사용해 세부 명령어를 생성합니다. 또한, 실시간으로 잠재적 오류를 수정하는 동적 백트래킹 모듈을 설계하여, 작업 수행 중 발생할 수 있는 오류를 줄입니다.

- **Performance Highlights**: EventNav는 R2R, REVERIE 및 ALFRED 데이터셋에서 실험 결과, 기존 모델보다 5% 이상 성공률이 향상되었습니다. 이는 이벤트 지식 그래프와 대형 언어 모델의 조합이 VLN 작업에서의 성능 향상에 효과적임을 의미합니다.



### Feedback Reciprocal Graph Collaborative Filtering (https://arxiv.org/abs/2408.02404)
Comments:
          9 pages, accepted by CIKM 2024

- **What's New**: 산업용 추천 시스템에서 협업 필터링(Collaborative Filtering)은 크게 성공을 거두었습니다. 그러나 사용자가 진정으로 매력을 느끼는 아이템을 추천하는 것은 기존 모델들에게 큰 도전 과제가 되어왔습니다. 이번에 소개된 연구는 Feedback Reciprocal Graph Collaborative Filtering (FRGCF)을 제안하며, 사용자가 매력을 느끼는 아이템을 강조하고 매력을 느끼지 않는 아이템의 추천을 약화시키는 새로운 방법론입니다.

- **Technical Details**: FRGCF는 전체 상호작용 그래프를 사용자의 피드백에 따라 Interacted & Fascinated (I&F) 그래프와 Interacted & Unfascinated (I&U) 그래프로 분리합니다. 이후 각각의 그래프에서 별도의 협업 필터링을 수행하며, 피드백 상호 대비 학습(feedback-reciprocal contrastive learning) 및 매크로 레벨 피드백 모델링(macro-level feedback modeling)을 도입합니다. 이를 통해 I&F 그래프 추천 시스템이 I&U 그래프로부터 다중 그레인 상호작용 특성을 학습하되, 오도되지 않도록 합니다.

- **Performance Highlights**: 네 개의 벤치마크 데이터셋과 하나의 수십억 규모 산업 데이터셋에서의 광범위한 실험 결과, FRGCF는 기존 모델보다 성능이 우수하며 사용자가 매력을 느끼지 않는 아이템의 추천을 줄이는 데 효과적임을 확인했습니다. 또한 Taobao의 추천 시스템에서 진행된 온라인 A/B 테스트에서도 FRGCF의 우수성이 입증되었습니다.



### RECE: Reduced Cross-Entropy Loss for Large-Catalogue Sequential Recommenders (https://arxiv.org/abs/2408.02354)
Comments:
          5 pages, 4 figures, submitted to CIKM'24

- **What's New**: 이 논문에서는 최신 권장 시스템의 확장성 문제를 해결하기 위해 RECE(REduced Cross-Entropy) 손실을 도입했습니다. RECE는 대규모 항목 카탈로그에서도 메모리 사용량을 크게 줄이면서 최고의 성능을 유지할 수 있습니다. 이를 통해 권장 시스템의 실용성을 크게 향상시킬 수 있습니다.

- **Technical Details**: RECE는 메모리 효율적인 locality-sensitive hashing(LSH)를 사용하여 logits의 대규모 텐서를 근사합니다. 이는 GPU 친화적인 방식으로 항목 순위를 계산하여 필요한 메모리를 크게 줄입니다. 이 방법은 Transformer 기반 모델, 특히 SASRec에 통합되어 실험되었습니다. 이 기법은 또한 NLP와 검색 시스템처럼 다른 영역에서도 유용할 수 있습니다.

- **Performance Highlights**: RECE는 기존 방법에 비해 최대 12배의 피크 메모리 사용량을 절감하면서, 성능 지표에서는 CE 손실을 유지하거나 초과하는 결과를 보여주었습니다. 이를 통해 대규모 응용 프로그램에서의 활용 가능성을 제시합니다.



### Embedding Compression in Recommender Systems: A Survey (https://arxiv.org/abs/2408.02304)
Comments:
          Accepted by ACM Computing Surveys

- **What's New**: 최근의 추천 시스템에서 임베딩 테이블 압축 방법에 대한 포괄적인 리뷰 연구입니다. 이 연구는 저비트수 (Low-Precision), 혼합 차원(Mixed-Dimension), 가중치 공유(Weight-Sharing)의 세 가지 범주로 기존 방법들을 정리하고, 각 압축 기법이 모델 성능에 미치는 영향을 논의합니다.

- **Technical Details**: 추천 시스템에서 임베딩 테이블(embedding tables)은 고차원의 희소한 원-핫 벡터(one-hot vectors)를 조밀한 실수 값의 임베딩으로 변환하는 데 사용됩니다. 하지만 임베딩 테이블은 대부분의 파라미터를 차지하며 메모리 사용량이 매우 큽니다. DLRM(DL 추천 모델)에서 다양한 태스크(CTR 예측, CVR 예측 등)를 위해, 저비트수 방법은 비트 너비를 줄이는 방식으로, 혼합 차원 방법은 특정 임베딩의 차원을 줄이는 방식으로, 가중치 공유 방법은 서로 다른 임베딩 간의 가중치를 공유하는 방식으로 나뉩니다.

- **Performance Highlights**: 임베딩 테이블의 압축을 통해 메모리 비용을 줄이고 효율성을 높일 수 있습니다. 예를 들어, Google Play의 Wide & Deep 모델, Alibaba의 DIN 모델, Huawei의 DeepFM 모델 등이 이러한 접근 방법을 사용하고 있으며, Baidu의 광고 시스템에서는 10TB의 임베딩 테이블을 사용하고 있습니다.



### Exploring Query Understanding for Amazon Product Search (https://arxiv.org/abs/2408.02215)
- **What's New**: 이번 연구는 Amazon의 상품 검색(Query Understanding)이 검색 서비스에 미치는 영향을 심층적으로 분석한 내용을 담고 있습니다. 실제 세계의 상품 검색 엔진에서 쿼리 이해가 검색 순위 결정 과정에 어떤 영향을 미치는지 탐구하였습니다. 또한, 쿼리 이해 서비스를 기반으로 한 다중 작업 학습 프레임워크도 제안하였습니다.

- **Technical Details**: 본 연구에서는 쿼리 이해 기반의 순위 결정 기능(query understanding-based ranking features)이 검색 결과 순위에 어떤 역할을 하는지 조사하였습니다. 특히, 쿼리 이해 시스템(query understanding system)이 순위 결정 모델(ranking model) 성능 이해에 어떻게 기여하는지를 상세히 분석하였습니다. 이를 바탕으로 쿼리 이해 기반의 다중 작업 학습 프레임워크(multi-task learning framework for ranking)를 제안합니다.

- **Performance Highlights**: 실제 아마존 검색 시스템을 사용한 연구 결과, 쿼리 이해 기반의 순위 결정 기능이 검색 엔진의 성능 향상에 중요한 역할을 한다는 것을 확인할 수 있었습니다. 제안된 다중 작업 학습 프레임워크는 기존 모델 대비 더 향상된 검색 성능을 보여주었습니다.



### Calibration-Disentangled Learning and Relevance-Prioritized Reranking for Calibrated Sequential Recommendation (https://arxiv.org/abs/2408.02156)
Comments:
          Published at CIKM '24 as a full research paper

- **What's New**: LeapRec (Calibration-Disentangled Learning and Relevance-Prioritized Reranking)는 사용자의 변화하는 선호도에 적응해야 하는 '연속적 추천' 시나리오에서 추천목록의 개인화된 비율을 유지하는 '교정된 추천(calibrated recommendation)'을 목표로 합니다. LeapRec은 교정과 관련성 간의 충돌을 효과적으로 다루며, 포스트 프로세싱 방식이 아닌 훈련 단계에서 이를 통합합니다.

- **Technical Details**: LeapRec은 두 단계로 구성되어 있습니다. 첫째, '모델 학습 단계'에서는 제안된 '교정-분리된 학습-랭킹 손실 함수(calibration-disentangled learning-to-rank loss)'를 사용하여 개인화된 랭킹을 최적화합니다. 둘째, '재정렬 단계'에서는 항목의 관련성을 우선시하여 목록 상단에 배치하고, 교정을 필요로 하는 항목을 비롯하여 충돌을 해결합니다.

- **Performance Highlights**: 네 개의 실제 데이터셋에 대한 광범위한 실험을 통해 LeapRec이 기존 방법들보다 일관되게 우수한 성능을 보여주는 것을 확인했습니다. 성능의 상위 지표(top-k)에서도 관련성과 교정의 최적화를 동시에 달성하는 것이 입증되었습니다.



### Generative Retrieval with Few-shot Indexing (https://arxiv.org/abs/2408.02152)
- **What's New**: 기존의 생성 검색(GR) 접근법의 한계를 극복하기 위해 새로운 Few-Shot GR 프레임워크가 제안되었습니다. 이 접근법에서는 몇 번의 인덱싱만으로 문서의 식별자(문서 ID) 생성을 수행하고, 대규모 사전 학습 언어 모델(LLM)의 지식을 최대한 활용하면서도 별도의 훈련이 필요 없습니다.

- **Technical Details**: Few-Shot GR는 두 주요 단계를 포함합니다. 첫 번째는 다중 매핑을 통한 Few-Shot 인덱싱이며, 두 번째는 제한된 빔 검색(constrained beam search)을 이용한 검색 단계입니다. 인덱싱 단계에서 LLM은 각 문서에 대해 여러 개의 자유 형태 문서 ID(free-text docids)를 생성해 docid 은행을 만듭니다. 검색 단계에서는 동일한 LLM을 사용해 주어진 쿼리의 docid를 생성하고, 생성된 docid를 문서와 매핑합니다.

- **Performance Highlights**: Natural Questions(NQ) 데이터셋 실험 결과, Few-Shot GR은 기존의 중무장 훈련 기반 GR 접근법보다 뛰어난 성능을 발휘하고 효율성이 높습니다. 중요한 성공 요인으로는 다중 매핑을 포함한 Few-Shot 인덱싱과 효과적인 LLM 선택이 언급되었습니다.



### Sharpness-Aware Cross-Domain Recommendation to Cold-Start Users (https://arxiv.org/abs/2408.01931)
- **What's New**: 새로운 메서드인 Sharpness-Aware Cross-Domain Recommendation (SCDR)가 제안되었습니다. 이는 전이 학습(transfer learning)을 활용해 차가운 상태(cold-start)의 사용자를 추천하는 Cross-Domain Recommendation (CDR) 문제를 다루기 위한 방법입니다. 기존의 CDR 메서드가 소수의 중복 사용자를 활용해 명시적 매핑 함수(mapping function)를 학습하는 반면, SCDR는 손실 기하학 기반(geometry-based) 머신러닝 접근방식을 활용합니다. 이로 인해 일반화 성능이 향상되고 이론적 보장이 제공됩니다.

- **Technical Details**: 기존의 CDR 모델은 중복 사용자가 소수일 경우 날카로운 손실 최소치(sharp minima)에 수렴하여 일반화 성능이 떨어지는 문제를 가집니다. SCDR는 이러한 문제를 해결하기 위해 손실 함수와 손실 풍경의 기하학적 속성을 동시에 최적화합니다. 이를 통해 모델은 평평한 손실 최소치(flat minima)에 수렴하며, 중복 사용자 주변의 손실이 균일하게 낮게 유지되도록 합니다. SCDR는 min-max 최적화 접근법을 사용하여 ℓ2노름(ℓ2-norm) 표현 공간 내에서 일관된 선호도를 유지하도록 유도합니다.

- **Performance Highlights**: 실제 데이터셋을 사용한 실험에서 SCDR는 다른 CDR 모델보다 훨씬 뛰어난 성능을 보여주었습니다. 특히 Amazon 데이터셋의 세 가지 CDR 시나리오에서 탁월한 성과를 거두었으며, 모델의 적대적 공격(adversarial attacks)에 대한 강인성도 향상되었습니다.



### Graph Stochastic Neural Process for Inductive Few-shot Knowledge Graph Completion (https://arxiv.org/abs/2408.01784)
- **What's New**: 이번 연구는 기존의 몇 번 샷 지식 그래프 완성 (FKGC) 방법론의 한계를 극복하고, 테스트 단계에서 새로운 관계와 엔티티를 처리할 수 있는 인덕티브 몇 번 샷 지식 그래프 완성 (I-FKGC) 문제를 다루고 있습니다. 이를 위해 새로운 Graph Stochastic Neural Process (GS-NP) 접근법을 제안하였습니다.

- **Technical Details**: 제안된 GS-NP 접근법은 두 가지 주요 모듈을 포함합니다. 첫 번째 모듈은 가설 추출기로, 신경 과정 기반 네트워크를 통해 공유된 하위 그래프와 같은 일반화된 가설을 모델링합니다. 두 번째 모듈은 그래프 확률적 주의 기반 예측기로, 쿼리 집합의 삼중항이 추출된 가설과 일치하는지를 테스트합니다. 이 예측기는 부가적으로 가설에 의해 확인된 설명 가능 서브 그래프를 생성할 수 있습니다.

- **Performance Highlights**: 세 가지 공공 데이터셋에서 광범위한 실험을 통해 GS-NP 모델이 기존 방법들을 능가하며, 새로운 state-of-the-art 성능을 달성함을 입증하였습니다.



### Leveraging the Power of LLMs: A Fine-Tuning Approach for High-Quality Aspect-Based Summarization (https://arxiv.org/abs/2408.02584)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 세부 요약(Aspect-based summarization)에 맞추어 미세 조정(fine-tuning)함으로써 문서의 특정 측면에 집중한 고품질 요약을 생성하는 방법을 탐구합니다. 오픈 소스 기반 LLMs인 Llama2, Mistral, Gemma, Aya를 사용하여 세부 요약 데이터셋을 통해 이들의 성능을 평가합니다.

- **Technical Details**: 연구는 LLMs를 Open Aspect-based Summarization (OASUM) 데이터셋으로 미세 조정하여 특정 측면에 대한 내용을 효과적으로 추출하고 요약할 수 있도록 하는 데 중점을 둡니다. 모델은 종합적인 평가 프레임워크를 설정하여 기존 세부 요약 방법 및 원래 LLMs 버전과의 성능을 비교합니다.

- **Performance Highlights**: 초기 결과는 미세 조정된 LLMs가 최신 세부 요약 방법과 비교할 때 더 높은 품질의 요약을 생성할 수 있음을 보여줍니다. 이는 교육, 의료, 음악 등 다양한 도메인에서 데이터 변형과 필요한 전문 지식을 효과적으로 처리할 수 있음을 시사합니다.



### RAG Foundry: A Framework for Enhancing LLMs for Retrieval Augmented Generation (https://arxiv.org/abs/2408.02545)
Comments:
          10 pages

- **What's New**: 저자들은 Retrieval-Augmented Generation (RAG) 시스템의 복잡성을 해결하고자 오픈 소스 프레임워크인 RAG Foundry를 소개했습니다. 이 프레임워크는 데이터 생성, 모델 학습(training), 추론(inference), 평가 과정을 단일 워크플로에 통합하며, 대형 언어 모델(LLM)을 RAG 환경에서 학습하고 평가하는 데 도움을 줍니다.

- **Technical Details**: RAG Foundry는 다양한 RAG 기술을 실험하고 프로토타이핑하는 과정을 단순화합니다. 사용자는 내부 또는 전문 지식 소스를 활용해 손쉽게 데이터셋을 생성하고 RAG 모델을 학습할 수 있습니다. 이러한 통합 접근 방식은 데이터 증강(dataset augmentation) 및 학습에 대한 효율성을 증대시키며, 다양한 RAG 설정에서의 성능을 극대화합니다. Llama-3와 Phi-3 모델을 다양한 RAG 구성으로 증강 및 미세 조정(fine-tuning)하여 프레임워크의 효과를 입증했습니다.

- **Performance Highlights**: 세 가지 지식 집중 데이터셋(knowledge-intensive dataset)에서 일관된 성능 향상이 확인되었습니다. 이는 프레임워크가 다룰 복잡한 설계 결정을 완화하면서도 데이터 검색 정확도와 생성 품질을 동시에 향상시킬 수 있음을 보여줍니다. 결과적으로, RAG Foundry는 효율적인 RAG 시스템 구현 및 평가를 촉진하는 도구임이 입증되었습니다.



### A Semi-supervised Multi-channel Graph Convolutional Network for Query Classification in E-commerc (https://arxiv.org/abs/2408.01928)
Comments:
          Accepted by WWW2024

- **What's New**: 본 논문에서는 신규로 제안된 반지도 학습 기반 다중 채널 그래프 컨볼루션 네트워크 (Semi-supervised Multi-channel Graph Convolutional Network, SMGCN) 모델을 소개합니다. 이 모델은 레이블 연관성과 반지도 학습을 통해 기존 쿼리 의도 분류(query intent classification)의 문제점을 해결합니다.

- **Technical Details**: SMGCN은 카테고리 간 상관관계(co-occurrence)와 의미적 유사성(severity similarity) 그래프를 활용하여 카테고리 간의 관계를 강화하고 자동 생성된 라벨의 불안정성을 약화시킵니다. 이를 위해 다중 채널 GCN이 관계를 모델링하고 쿼리와 카테고리 간의 유사성 점수를 계산한 후, 클릭 라벨과 결합하여 손실 값을 계산합니다. 이 접근 방식은 감소된 데이터의 한계를 보완하며 관련 카테고리를 더 잘 회상할 수 있도록 돕습니다.

- **Performance Highlights**: 대규모 실제 데이터셋에 대한 오프라인 및 온라인 A/B 테스트 실험 결과, SMGCN은 기존 강력한 모델들보다 현저히 우수한 성능을 보였습니다. 해당 모델은 상용 전자상거래 플랫폼에 도입되어 매일 수억 건의 요청을 처리하고 있으며, 큰 상업적 가치를 제공합니다.



### MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilanc (https://arxiv.org/abs/2408.01869)
Comments:
          Paper published at Machine Learning for Healthcare 2024 (MLHC'24)

- **What's New**: 이번 논문은 대형 언어 모델(LLM)을 이용한 새로운 약물 감시(Pharmacovigilance, PhV) 방법론을 소개합니다. MALADE라는 이름의 새로운 다중 에이전트 시스템을 통해 약물 라벨 데이터에서 부작용 사건(ADE)을 추출하는 방식을 제안합니다. 이 시스템은 Retrieval Augmented Generation(RAG) 기법을 사용하여 쿼리를 보강하고, 그 쿼리에 기반하여 응답을 생성합니다.

- **Technical Details**: MALADE는 일반적인 LLM-비종속적 아키텍처로, (1) 의학 문헌, 약물 라벨, FDA 도구(OpenFDA drug information API) 등 다양한 외부 소스를 활용하고, (2) 약물-결과 연관성을 구조화된 형식으로 추출하여 연관성의 강도를 제시하며, (3) 확인된 연관성에 대한 설명을 제공하는 독특한 기능을 갖추고 있습니다. 시스템은 Langroid 다중 에이전트 LLM 프레임워크를 활용하며, GPT-4 Turbo 또는 GPT-4o와 FDA 약물 라벨 데이터를 사용하여 구현되었습니다.

- **Performance Highlights**: 실험 결과, OMOP Ground Truth 테이블과 비교하여 약물 라벨 데이터의 ADE 추출에서 ROC 곡선 아래 면적(AUC) 0.90의 성능을 보여주었습니다. 이는 현재까지의 최신 방법 중 가장 높은 성능을 기록한 것입니다. 이 시스템은 단순히 이진 레이블을 제공하는 것이 아니라, 연관성의 강도 및 부작용의 희귀도 등을 포함한 구조화된 점수를 제시합니다.



### A Novel Evaluation Framework for Image2Text Generation (https://arxiv.org/abs/2408.01723)
Comments:
          The paper has been accepted for presentation at the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, specifically in the Large Language Model for Evaluation in IR (LLM4Eval) Workshop in 2024

- **What's New**: 자동 생성된 이미지 설명의 품질을 평가하는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 GPT-4나 Gemini와 같은 최신 대형 언어 모델(LLM)을 사용하여 이미지 설명을 기반으로 새로운 이미지를 생성합니다. 생성된 이미지와 원본 이미지의 유사성을 측정하여 이미지 설명의 정확도를 평가합니다. 기존의 BLEU, ROUGE, METEOR, CIDEr 등의 자동화된 메트릭이 사람의 판단과 약한 상관관계를 가지는 문제를 해결할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 이미지 캡셔닝 모델의 출력인 텍스트 설명을 LLM을 통해 이미지로 변환하는 과정을 포함합니다. 변환된 이미지와 원본 이미지의 유사도를 코사인 유사도(cosine similarity) 메트릭으로 측정합니다. 사람의 주석이 필요하지 않으며 이미지 설명의 품질을 평가할 수 있습니다. 이 프레임워크는 최신 LLM 기술을 바탕으로 설계되었으며, 이를 통해 텍스트 설명이 원본 이미지를 얼마나 잘 재현할 수 있는지 평가합니다.

- **Performance Highlights**: 제안된 프레임워크는 사람의 평가와 높은 상관관계를 가지며, 효과적으로 이미지 캡셔닝 모델의 성능을 평가할 수 있습니다. 인간 주석이 필요하지 않기 때문에 비용 효율적이고 시간 소모가 적습니다. 기존의 BLEU, ROUGE, METEOR, CIDEr 메트릭보다 높은 정확도를 보이며, 다양한 이미지 캡셔닝 데이터셋을 통해 종합적으로 검증되었습니다.



### On Validation of Search & Retrieval of Tissue Images in Digital Pathology (https://arxiv.org/abs/2408.01570)
- **What's New**: 이 논문은 의료 영상의 진단, 치료 계획 및 질병 모니터링에 있어서의 중요성을 강조합니다. 특히, 영상 기반 이미지 검색(Content-Based Image Retrieval, CBIR) 시스템이 의료 분야에서 어떻게 활용될 수 있는지를 논의합니다.

- **Technical Details**: 논문은 방사선학과 병리학이 정확한 이미지 해석에 얼마나 의존하는지 설명합니다. 구체적으로, 방사선 전문의가 X-레이, CT 스캔, MRI 등을 통해 골절에서 암에 이르기까지 다양한 상태를 진단하고, 병리학자는 현미경과 디지털 이미지를 사용해 세포 비정상(세포 이상, cellular abnormalities)을 진단합니다. CBIR 시스템은 시각적 콘텐츠를 기반으로 이미지를 검색하고 검색하는 방식으로, 진단 정확성을 높이는 데 도움을 줍니다.

- **Performance Highlights**: 이미지 검색 엔진의 포괄적인 검증을 위해, 정확도(accuracy), 색인(indexing), 검색 시간(search times), 저장 오버헤드(storage overhead)와 같은 성능 지표들을 평가하는 것이 중요합니다. 최근의 검증 결과에 따르면, 이러한 평가를 통해 신뢰할 수 있고 효율적인 검색 결과를 제공할 수 있습니다. 특히, 조직병리학(histopathology)에서 효율적인 검색을 통해 진단 정확도가 향상되었습니다.



### pathfinder: A Semantic Framework for Literature Review and Knowledge Discovery in Astronomy (https://arxiv.org/abs/2408.01556)
Comments:
          25 pages, 9 figures, submitted to AAS jorunals. Comments are welcome, and the tools mentioned are available online at https://pfdr.app

- **What's New**: Pathfinder는 천문학 연구자들이 천문학 문헌을 쉽게 검토하고 지식을 발견할 수 있도록 돕는 머신러닝 프레임워크입니다. 이 프레임워크는 키워드 검색 대신 자연어를 활용한 의미론적 검색을 중심으로 합니다. 최신 대형 언어 모델(LLMs)과 ADS(Astrophysics Data System)의 35만 개의 피어 리뷰 논문을 사용하여 과학적 탐구와 문헌 탐색을 혁신적으로 접근합니다.

- **Technical Details**: Pathfinder는 Retrieval-Augmented Generation (RAG) 및 에이전트 LLMs 같은 최신 기법과 결합하여 의미론적 맥락에서 천문학 논문을 검색할 수 있습니다. 이는 존재하는 검색 방법을 보완하며, 복잡한 용어, 명명된 엔터티 및 시간적 측면을 시간 기반 및 인용 기반 가중치 스킴으로 해결합니다. 현재 버전은 논문의 초록만 사용하며, 추후 전체 텍스트로 확장될 예정입니다.

- **Performance Highlights**: Pathfinder는 다양한 연구 시나리오에서 도구의 다재다능함을 보여주는 사례 연구를 통해 그 성능을 입증했습니다. 사용자 맞춤형 벤치마크를 사용하여 성능을 평가했으며, 단일 논문 및 여러 논문 작업을 포함합니다. 또한 이 도구는 천문학 연구 풍경을 시각화하고, 천문대와 방법론의 영향을 추적하며, 다양한 청중에게 접근 가능한 형식으로 답변을 재구성할 수도 있습니다.



New uploads on arXiv(cs.CV)

### Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics (https://arxiv.org/abs/2408.02672)
Comments:
          equal contribution for first two authors; accepted to ECCV2024; 14 pages, 4 tables, 10 figures in main paper, supplementary after bibliography

- **What's New**: 전통적인 비디오 INRs는 주로 압축 작업에 초점을 맞추고 있었지만, 이는 압축 외의 의미론적 속성이 부족하여 검색 등의 작업에 유용하지 못했습니다. 이를 해결하기 위해 우리는 비디오 INR의 공간적, 시간적 측면을 분리하는 유연한 프레임워크를 제안합니다. 이 프레임워크는 각 프레임에 대한 잠재 변수(learnable latents) 사전과 비디오 특정 하이퍼네트워크(hypernetworks)로 구성되어 있습니다. 이 방법을 통해 압축 효율성과 더불어 잠재 변수를 큰 비전 모델(CLP 등)의 특징과 정렬하여 검색 및 비디오 질문 응답 등의 후속 작업을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 (i) 각 프레임에 대한 학습 가능한 잠재 변수 사전과 (ii) 모든 비디오에 대해 학습된 하이퍼네트워크 세트로 구성됩니다. 이러한 하이퍼네트워크는 주어진 잠재 변수를 입력으로 받아 프레임별 INR 가중치를 예측하여 특정 프레임을 재구성합니다. 이 디자인은 비디오의 공간적 및 시간적 측면을 분리하여 개별적으로 모델링할 수 있게 합니다. 이를 통해 좌표 기반 INR의 원래 특성을 보존하면서도 공간적 보간 및 무한해상도(infinite resolution) 추론을 가능하게 합니다.

- **Performance Highlights**: 우리의 시스템은 압축 성능 면에서 우수하며, PSNR, BPP, 디코딩 속도에서 다른 ML 기반 코덱들과 경쟁합니다. 특히, 학습된 잠재 변수가 CLIP과 같은 큰 기초 모델과 정렬되어 텍스트 쿼리를 통한 프레임, 개념, 비디오 검색을 지원합니다. 또한 VideoLlama와 정렬되어 비디오 질문 응답 및 자막 생성과 같은 대화형 응용 프로그램을 가능하게 합니다.



### Lumina-mGPT: Illuminate Flexible Photorealistic Text-to-Image Generation with Multimodal Generative Pretraining (https://arxiv.org/abs/2408.02657)
Comments:
          Code available at: this https URL

- **What's New**: Lumina-mGPT는 새로운 멀티모달 자가회귀 모델로, 다양한 시각 및 언어 작업을 수행할 수 있으며, 특히 텍스트 설명으로부터 유연하고 사진 같은 이미지를 생성하는 데 뛰어나다. 기존의 자가회귀 이미지 생성 접근법과 달리 Lumina-mGPT는 pretrained(사전학습된) decoder-only transformer를 사용하여 멀티모달 토큰 시퀀스를 모델링한다.

- **Technical Details**: 중요한 발견은 단순한 decoder-only transformer가 대규모 교차 텍스트-이미지 시퀀스에서 next-token prediction 목표를 활용하여 광범위하고 일반적인 멀티모달 역량을 배울 수 있다는 것이다. 이 pretrained 모델을 기반으로 고품질 이미지-텍스트 쌍에 대한 Flexible Progressive Supervised Finetuning(FP-SFT)을 제안하여 높은 미적 이미지 합성을 완전히 잠재력을 끌어낸다. 또한 Ominiponent Supervised Finetuning(Omni-SFT)을 도입하여 Lumina-mGPT를 다양한 작업을 원활하게 통합할 수 있는 기초 모델로 만든다.

- **Performance Highlights**: Lumina-mGPT는 유연한 텍스트에서 이미지 생성을 포함한 다재다능한 멀티모달 역량을 보여준다. 또한, 제어 가능한 생성, 분할 및 깊이 추정과 같은 시각 인식 작업, 그리고 멀티턴 시각 질문 응답과 같은 시각-언어 작업에서도 뛰어난 성능을 발휘한다. 추가로, diffusion-based(확산 기반) 접근법과 자가회귀(methods)를 직접 비교하여 차이점과 유사성을 분석하였다.



### Interactive 3D Medical Image Segmentation with SAM 2 (https://arxiv.org/abs/2408.02635)
- **What's New**: 본 논문에서는 3D 의료 영상 세분화를 위한 첨단 Segment Anything Model (SAM)의 차세대 버전인 SAM 2의 가능성을 탐구했습니다. SAM 2는 비디오 학습을 통해 3D 의료 영상 데이터를 자동으로 처리하는 능력을 갖추고 있으며, 이를 통해 단일 프레임의 주석을 전체 3D 볼륨으로 전파할 수 있습니다.

- **Technical Details**: SAM 2는 연속된 2D 슬라이스를 비디오 프레임으로 취급하여 주석을 전파합니다. 이를 통해 3D 의료 영상 데이터를 비디오로 간주하고, 단일 프레임의 상호작용(클릭, 박스 또는 마스크)을 기반으로 3D 세분화를 수행합니다. 본 논문에서는 이와 같은 접근 방식을 실현하기 위한 간단한 파이프라인을 제안하고, Brats 2020과 의료 세그멘테이션 Decathlon 데이터셋에서의 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, SAM 2는 여전히 감독 학습 방법에 비해 성능 격차가 존재하지만 특정 설정 및 장기 유형에서는 격차를 좁힐 수 있음을 확인했습니다. 또한, SAM 2는 기존의 3D 상호작용 의료 이미지 세분화 알고리즘보다 상호작용 피드백을 효율적으로 이용할 수 있음을 보여주었습니다.



### VidGen-1M: A Large-Scale Dataset for Text-to-video Generation (https://arxiv.org/abs/2408.02629)
Comments:
          project page: this https URL

- **What's New**: VidGen-1M은 비디오 텍스트 모델의 훈련을 위한 뛰어난 데이터셋으로, 기존 데이터셋의 단점을 극복하고 고품질의 비디오와 상세한 캡션을 제공하기 위해 개발되었습니다.

- **Technical Details**: VidGen-1M 데이터셋은 3단계의 데이터 큐레이션 방법을 사용하여 제작되었습니다. 초기 큐레이션(coarse curation) 단계에서는 장면 분할(scene splitting)과 태깅(tagging)을 수행한 후 필터링과 샘플링을 통해 비디오를 균형 있게 분포시킵니다. 캡셔닝(captioning) 단계에서는 비디오 캡셔닝 모델을 활용하여 설명적인 합성 캡션을 생성합니다(DSL, Descriptive Synthetic Captions). 마지막으로, 정밀 큐레이션(fine curation) 단계에서는 대형 언어 모델(LLM)을 사용하여 비디오 캡션을 정제합니다.

- **Performance Highlights**: VidGen-1M 데이터셋을 사용하여 훈련된 텍스트-비디오 모델은 기존의 다른 모델들보다 뛰어난 성능을 보여주었습니다. 고품질의 비디오와 평균 89.2 단어의 풍부한 캡션 덕분에 텍스트-비디오 정렬(text-video alignment)이 강화되었고, 비디오의 개선된 시간적 일관성은 모델 훈련의 안정성도 높였습니다.



### YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition (https://arxiv.org/abs/2408.02623)
- **What's New**: YOWOv3는 인간 행동 감지 및 인식을 위한 새로운 프레임워크로, YOWOv2의 개선된 버전입니다. 이 프레임워크는 다양한 구성으로 광범위한 실험을 용이하게 하고 모델 내의 여러 구성 요소를 쉽게 사용자 맞춤화할 수 있게 설계되었습니다. YOWOv3는 이전 모델인 YOWOv2에 비해 두 개의 널리 사용되는 데이터셋, 즉 UCF101-24와 AVAv2.2에서 더 우수한 성능을 보여줍니다.

- **Technical Details**: YOWOv3는 Two-Stream Network 아키텍처를 채택하여 두 개의 처리 스트림을 포함합니다. 첫 번째 스트림은 2D CNN을 사용하여 이미지에서 공간 정보와 컨텍스트를 추출하며, 두 번째 스트림은 3D CNN을 사용하여 시간적 정보와 움직임을 집중적으로 추출합니다. 두 스트림의 출력은 결합되어 비디오의 공간 및 시간 정보를 모두 캡처하는 특징을 얻습니다. 마지막으로, CNN 레이어를 사용하여 이러한 추출된 특징을 기반으로 예측을 수행합니다. 이번 YOWOv3에서는 YOLOv8 모델을 백본으로 사용하여 공간적 특징을 추출하며, YOLOv8은 객체 감지 벤치마크에서 우수한 성능을 입증한 네트워크입니다.

- **Performance Highlights**: YOWOv3는 YOWOv2에 비해 더 적은 59.8M 매개변수와 39.8 GFLOPs로 UCF101-24에서 88.33%의 mAP와 AVAv2.2에서 20.31%의 mAP를 달성했습니다. 이는 YOWOv2의 109.7M 매개변수와 53.6 GFLOPs에 비해 성능은 유지하면서도 더 적은 계산 자원을 사용했음을 보입니다.



### LaMamba-Diff: Linear-Time High-Fidelity Diffusion Models Based on Local Attention and Mamba (https://arxiv.org/abs/2408.02615)
- **What's New**: 최근 Transformer 기반 확산 모델(diffusion models)은 놀라운 성과를 보여주고 있지만, 이들의 쿼드러틱(quadratic) 복잡성은 긴 시퀀스 입력을 처리하는 데 큰 계산 부담을 초래합니다. 이를 해결하기 위해 Local Attentional Mamba(LaMamba) 블록이 도입되었습니다. LaMamba는 전역(context)과 지역(context)을 모두 효율적으로 캡처하며 선형(linear) 복잡성을 갖습니다. 특히, U-Net 아키텍처를 활용하여 ImageNet 256x256 해상도에서 DiT를 능가하는 성능을 보이며, 훨씬 적은 GFLOPs와 유사한 매개변수 갯수를 사용합니다.

- **Technical Details**: LaMamba는 self-attention과 Mamba의 장점을 결합하여 전역 문맥(global context)과 지역 문맥(local context)을 선형 복잡성으로 캡처합니다. Mamba는 전역 문맥을 효율적으로 캡처하는 반면, 지역 self-attention은 고정 크기의 컨텍스트 윈도우 내에서 쌍(pairwise)의 상호작용을 계산하여 미세한 지역 의존성을 정확하게 유지합니다. LaMamba-Diff는 U-Net 아키텍처를 채택하여 다중 스케일 계층적 특징을 효율적으로 구성하며, downsampling 단계에서 공간 차원을 압축함으로써 효율성을 극대화합니다.

- **Performance Highlights**: LaMamba-Diff는 ImageNet 256x256 및 512x512 이미지 생성 벤치마크에서 DiT-XL/2와 비교해 최대 62%까지 GFLOPs를 줄이면서도 탁월한 성능을 달성합니다. 특히, LaMamba-Diff-XL은 ImageNet 256x256 이미지 생성에서 FID 2.04를 기록하며, 이는 DiT-XL/2 대비 57.6% 적은 GFLOPs로 달성한 것입니다. ImageNet 512x512 이미지 생성에서는 FID 3.01을 기록하며 DiT-XL/2 대비 61.6% 적은 GFLOPs를 사용합니다. 이러한 결과는 LaMamba-Diff가 고해상도 이미지 생성 작업에서 효과적이고 효율적임을 입증합니다.



### Modelling Visual Semantics via Image Captioning to extract Enhanced Multi-Level Cross-Modal Semantic Incongruity Representation with Attention for Multimodal Sarcasm Detection (https://arxiv.org/abs/2408.02595)
- **What's New**: 이 연구는 소셜 미디어 데이터에서 효과적으로 풍자를 인식하기 위해 텍스트에 더해 이미지와 같은 추가적인 맥락적 단서를 포함하는 새로운 다중 모달리티 풍자 탐지 프레임워크를 제안합니다. 특히, 이미지와 함께 설명적인 이미지 캡션을 추가해 텍스트와 시각적 콘텐츠 간의 불일치를 더 정확하게 포착하려는 것이 주요 목표입니다.

- **Technical Details**: 주요 기술적 기여로는 (1) Cross-lingual 언어 모델을 활용한 강력한 텍스트 특징 추출 브랜치, (2) Self-regulated Residual ConvNet과 가벼운 Spatially Aware Attention 모듈을 통합한 시각적 특징 추출 브랜치, (3) 이미지에 포함된 텍스트를 읽을 수 있는 인코더-디코더 구조를 사용한 이미지 캡션 생성, (4) 텍스트와 두 가지 이미지 표현들 사이의 불일치를 효과적으로 식별하기 위한 다양한 어텐션 모듈, (5) 피처 융합을 통해 다중 레벨의 교차 도메인 의미 불일치 표현 등이 포함됩니다.

- **Performance Highlights**: 제안된 모델은 트위터 다중 모달 풍자 탐지(Twitter multimodal sarcasm) 및 멀티불리(MultiBully) 데이터셋에서 각각 92.89%와 64.48%의 정확도로 최상의 성능을 보였습니다.



### Contrastive Learning-based Multi Modal Architecture for Emoticon Prediction by Employing Image-Text Pairs (https://arxiv.org/abs/2408.02571)
- **What's New**: 이 연구는 문장, 비주얼, 이모티콘 간의 관계를 분석하는 새로운 접근 방식을 제안합니다. 본 연구는 멀티모달 (multimodal) 기능 추출 방법을 상세히 분석하고, 이들을 결합한 새로운 대비 학습 (contrastive learning) 기반 멀티모달 아키텍처를 도입했습니다.

- **Technical Details**: 제안된 모델은 이중 브랜치 인코더 (dual-branch encoder)의 공동 훈련과 대비 학습을 활용하여 텍스트와 이미지를 공통된 잠재 공간으로 정확하게 매핑합니다. 특히, 여러 멀티모달 알고리즘과 특히 융합 접근 방식 (fusion approaches)을 종합적으로 검토하였습니다.

- **Performance Highlights**: 제안된 방법론은 현재의 멀티모달 접근 방식보다 정확성과 견고성 면에서 뛰어납니다. Multimodal-Twitter Emoticon 데이터셋을 사용하여 이모티콘을 평가한 결과, 제안된 모델은 91%의 정확도와 90%의 MCC 점수를 달성했습니다. 대비 학습으로 획득한 깊은 특징이 더 효율적임을 증명하며, 여러 모드에서 이모티콘을 인식할 수 있는 강력한 일반화 능력을 제시합니다.



### HQOD: Harmonious Quantization for Object Detection (https://arxiv.org/abs/2408.02561)
Comments:
          2024 IEEE International Conference on Multimedia and Expo (ICME), July 15 - July 19, 2024, Niagra Falls, Ontario, Canada

- **What's New**: 현대 객체 탐지기에서 자주 발생하는 작업 불균형 문제를 해결하기 위해 Harmonious Quantization for Object Detection (HQOD) 프레임워크가 제안되었습니다. HQOD는 두 가지 주요 구성 요소로 이루어져 있습니다: 첫째, Quantization-Aware Training (QAT) 과정 중에 낮은 작업 조화 품질 샘플의 성능을 향상시키도록 유도하는 작업 상관 손실(Task-Correlated Loss), 둘째, 다양한 Intersection over Union (IoU) 수준에서 회귀 분기(Regression Branch)의 최적화를 균형 있게 할 수 있도록 돕는 조화 IoU 손실(Harmonious IoU Loss)입니다.

- **Technical Details**: HQOD는 QAT 알고리즘과 다양한 객체 탐지기에 쉽게 통합될 수 있는 프레임워크입니다. QAT는 양자화(Quantization) 소음을 모사함으로써 네트워크가 이를 조정할 수 있도록 해주는 방법으로, 모델 압축 기술 중 하나입니다. LSQ(LSQ), TQT(TQT), 그리고 AQD(AQD)와 같은 QAT 방법에 HQOD를 적용함으로써 객체 탐지기의 작업 불균형 문제가 개선됩니다. 특히, 모델은 낮은 비트 너비(bit width) 조건에서도 높은 성능을 유지할 수 있습니다.

- **Performance Highlights**: MS COCO 데이터셋에서 초기 성능 테스트 결과, ResNet-50 백본을 사용한 4비트 ATSS가 39.6%의 mAP를 달성하며, 이는 기존 풀 프리시전(full-precision) 모델보다 높습니다. 또한 2비트 ATSS 모델에서도 성능 개선을 확인할 수 있었으며, TQT 대비 1.4% mAP의 향상을 보여줍니다. PASCAL VOC 데이터셋 실험 결과, LSQ에서는 평균 0.75% mAP, AQD에서는 평균 0.65% mAP가 향상되었습니다.



### MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization (https://arxiv.org/abs/2408.02555)
Comments:
          Project Page: this https URL Github: this https URL

- **What's New**: MeshAnything V2는 새로운 오토레그레시브 트랜스포머를 이용해 주어진 형태에 맞춘 아티스트가 만든 메쉬(Artist-Created Meshes, AM)를 생성하는 기술을 소개합니다. MeshAnything V2는 Adjacent Mesh Tokenization(AMT)이라는 새로운 메쉬 토큰화 방법을 사용하여 이전 방법보다 효율성과 성능 면에서 뛰어납니다. AMT는 가능한 경우 단일 정점을 사용하여 메쉬를 나타냅니다.

- **Technical Details**: MeshAnything V2는 AMT를 통해 기존 메쉬 토큰화 방법에서 볼 수 있는 세 정점을 사용한 접근 방식 대신 단일 정점을 사용하는 방식을 채택하여 메쉬의 토큰 시퀀스 길이를 절반으로 줄입니다. AMT는 페이스를 인코딩한 후 인접 페이스를 찾아 단일 정점으로 인코딩합니다. 이는 시퀀스 길이를 약 1/3로 줄이고, 결과적으로 AM 생성의 효율성과 성능을 크게 향상시킵니다.

- **Performance Highlights**: AMT를 통해 MeshAnything V2는 메모리 사용량을 거의 4배 줄이고, 기존의 최대 생성 가능한 페이스 수를 800개에서 1600개로 두 배로 늘립니다. 실험 결과 AMT의 도입으로 모델 성능과 효율성이 크게 개선되었습니다. MeshAnything V2는 이러한 성과를 기반으로 커뮤니티에 오픈소스로 제공됩니다.



### Estimating Pore Location of PBF-LB/M Processes with Segmentation Models (https://arxiv.org/abs/2408.02507)
Comments:
          20 pages, 7 figures, This work has been submitted to the Journal Progress in Additive Manufacturing

- **What's New**: 이 논문에서는 Laser Powder Bed Fusion (PBF) 공정에서 발생하는 결함, 특히 기공(pores)을 더 정확하게 로컬라이징하는 방법을 제안합니다. 기존의 현장 모니터링 데이터(in-situ monitoring data)를 사용한 방법들은 기공 발생을 감지할 수는 있지만, 로컬라이징 정확도가 낮다는 한계가 있었습니다. 이 연구는 가우시안 커널 밀도 추정(Gaussian kernel density estimation)을 활용하여 기공의 위치를 더 정밀하게 예측할 수 있는 접근 방식을 제안합니다.

- **Technical Details**: 이 새로운 접근 방식은 기공의 위치를 픽셀 수준에서 예측하기 위해 기공 발생 확률 분포를 계산합니다. 이러한 확률 분포는 가우시안 확률 밀도 함수(Gaussian probability density function)를 사용하여 계산되며, 이를 통해 모델이 현장 모니터링 데이터와 기공 발생 확률 간의 상관관계를 학습할 수 있도록 합니다. 이 과정에서 최소한의 데이터 전처리 만을 요구하므로 산업 응용 분야에서의 적용 가능성을 높입니다. 실험에서는 다양한 공정 파라미터 및 기하학적 특징을 가지는 샘플을 제조하고, 각 레이어별로 기공 위치를 분석하여 데이터세트를 구축하였습니다.

- **Performance Highlights**: 제안된 방법은 현장 모니터링 데이터에서 기공의 발생 확률을 픽셀 단위로 예측할 수 있으며, 이를 통해 기존의 기공 로컬라이징 방법보다 더 높은 정밀도를 달성했습니다. 다양한 분할 모델(segmentation models)을 적용하여 실험한 결과, 기하학적 복잡성과 공정 파라미터가 변하는 상황에서도 안정적으로 높은 성능을 보였습니다. 기존의 기공 감지 시스템에 비해 데이터 전처리 단계가 단순하고, 최종적으로 더 정확한 기공 위치 예측을 가능하게 했습니다.



### HyperSpaceX: Radial and Angular Exploration of HyperSpherical Dimensions (https://arxiv.org/abs/2408.02494)
- **What's New**: 새로운 분야인 HyperSpaceX를 제안하여 다중-초구 초월 공간에서 각도와 반경 차원을 탐구함으로써 클래스 구분을 향상시킵니다. 새로운 DistArc 손실 함수는 세 가지 특징 배열 구성 요소(두 각도, 하나 반경)를 통합하여 클래스 간의 분리를 강화하고, 모델의 정확도를 더 포괄적으로 평가할 수 있는 예측 측정을 도입합니다.

- **Technical Details**: HyperSpaceX 프레임워크는 기존의 각도 공간을 넘어 다중-초구 공간에서 반경 차원도 포함하는 특징 탐구를 제안합니다. DistArc 손실 함수는 특징의 배열을 최적화하여 클래스 간의 구분을 강화하고, 동일 클래스 내부의 클러스터링을 향상시킵니다. 이는 다양한 데이터 세트(객체 분류 7개 및 얼굴 인식 6개 데이터 세트)에 대해 모델 성능을 평가하는 새로운 예측 측정을 포함합니다.

- **Performance Highlights**: HyperSpaceX는 여러 데이터 세트(MNIST, FashionMNIST, CIFAR-10, CIFAR-100, CUB-200, TinyImageNet, ImageNet1K, LFW, CFP-FP, AgeDB-30, CA-LFW, CP-LFW, D-LORD)에서 실험을 통해 모델 정확도와 특징 구분에서 최첨단 결과를 달성했습니다. 큰 규모의 객체 데이터 세트에서 최대 20% 향상, 고차원 데이터 세트에서 최대 6% 개선을 이루었습니다.



### Exploring Conditional Multi-Modal Prompts for Zero-shot HOI Detection (https://arxiv.org/abs/2408.02484)
- **What's New**: 이번 논문에서는 Zero-shot Human-Object Interaction (HOI) detection을 위한 새로운 프레임워크인 Conditional Multi-Modal Prompts (CMMP) 기법을 소개합니다. 이 방법은 대형 비전-언어 기반 모델, 특히 CLIP를 HOI detection에 맞게 튜닝하는 과정에서 일반화 능력을 향상시키는 목적을 가지고 있습니다. 특히 시각적 특징 추출과 상호작용 분류를 위해 비전과 언어 프롬프트를 분리하여 학습하는 방식을 제안합니다.

- **Technical Details**: 제안된 CMMP 기법은 두 가지 주요 부분으로 나뉩니다. 첫째, Interactiveness-aware visual feature extraction을 위해 input-conditioned instance prior와 global spatial pattern prior를 통합한 conditional vision prompts를 소개합니다. 둘째, 상호작용 분류를 위해 언어 프롬프트를 사용하여 보편적인 컨텍스트를 제공합니다. 이 과정에서 대형 비전-언어 모델의 지식을 잘 보존하기 위해 consistancy constraint를 적용합니다.

- **Performance Highlights**: 다양한 zero-shot 설정에서 실험을 통해, 제안된 방법은 기존 방법들을 뛰어넘는 성능을 보여주었습니다. 특히, 보지 못한 클래스들에서 최고의 harmonic mean 성능을 달성했습니다. 코드와 모델은 공개되어 있으며, 실제 적용 가능성이 높습니다.



### Fairness and Bias Mitigation in Computer Vision: A Survey (https://arxiv.org/abs/2408.02464)
Comments:
          20 pages, 4 figures

- **What's New**: 최근 컴퓨터 비전(computer vision) 시스템의 공정성(fairness)을 다룬 체계적인 설문조사 논문이 발표되었습니다. 이 논문은 컴퓨터 비전의 공정성과 편향성(bias)에 대한 현재 동향과 성공 사례를 종합적으로 요약하며, 이 분야의 중요성을 강조합니다.

- **Technical Details**: 해당 논문은 다음과 같은 주요 주제들을 다루고 있습니다: 1) 공정성의 기원과 기술적 정의. 2) 컴퓨터 비전 시스템에서 발견된 편향성을 분석한 연구. 3) 최근 몇 년간 제안된 편향성 완화 방법. 4) 편향성을 측정, 분석 및 완화하기 위한 연구자들이 만든 자원과 데이터셋. 5) 멀티모달 기반 모델과 생성 모델에서의 공정성 연구 동향 및 남은 과제.

- **Performance Highlights**: 논문에서는 공정성과 편향성을 극복하기 위한 방법론을 다양한 컴퓨터 비전 작업(예: 이미지 분류, 객체 감지, 활동 인식, 얼굴 인식 및 분석)에 적용한 성과를 논의합니다. 특히, 컴퓨터 비전 시스템에서 민감한 보호속성(예: 성별, 인종, 연령, 피부색)의 편향성을 완화하며, 공정성을 높이는 방법론에 초점을 두고 있습니다. 이를 통해 모델 성능과 공정성 간의 트레이드오프를 식별하고 해결 방안을 제시합니다.



### Attenuation-adjusted deep learning of pore defects in 2D radiographs of additive manufacturing powders (https://arxiv.org/abs/2408.02427)
Comments:
          Implementation on this https URL

- **What's New**: 이 연구에서는 3D X-ray 컴퓨터 단층 촬영(XCT) 대신 단일 2D 방사선 사진만으로 금속 분말 피드스톡(feedstock) 내 기공을 고속으로 분석하는 방법을 제안하였습니다. 기존의 UNet 아키텍처와 결합한 X-ray 감쇠 모델을 사용하여 F1-score가 11.4% 향상된 결과를 도출하였습니다.

- **Technical Details**: 제안된 방법은 주요 세 가지 요소로 구성됩니다: 1) 합성 데이터로 사전 학습(pre-training), 2) 단단한 입자(cutout) 추출, 3) Lambert-Beers 법칙을 따르는 이상적인 입자와 실제 입자의 차이를 이용해 기포 제거. 이 연구는 다양한 이미지 처리 방법을 탐구하였으며, 가장 빠른 방법은 평균 0.014초, 가장 정확한 방법은 0.291초에 기공을 분할합니다.

- **Performance Highlights**: 제안된 방법은 0.87의 F1-score로 금속 피드스톡 분말의 기공을 단일 2D 방사선 사진에서 고정밀도로 분할할 수 있습니다. 이 접근법은 향후 고속 기공 분석에 유망한 결과를 보여줍니다.



### FPT+: A Parameter and Memory Efficient Transfer Learning Method for High-resolution Medical Image Classification (https://arxiv.org/abs/2408.02426)
- **What's New**: 이 논문에서는 고해상도 의료 이미지 분류를 위해 고안된 새로운 Parameter-efficient transfer learning (PETL) 방법인 Fine-grained Prompt Tuning plus (FPT+)를 소개합니다. FPT+는 메모리 소비를 크게 줄이면서도 기존의 다른 PETL 방법보다 우수한 성능을 발휘합니다.

- **Technical Details**: FPT+는 대규모 사전 훈련 모델(Large Pre-trained Model, LPM)의 미세하게 세분화된 프롬프트(fine-grained prompts)와 융합 모듈(fusion modules)을 통해 경량의 사이드 네트워크를 학습시킵니다. 여기서 LPM은 고해상도 이미지를 처리하고, 사이드 네트워크는 다운샘플된 저해상도 이미지를 사용하여 메모리 소비를 최소화합니다.

- **Performance Highlights**: FPT+는 전체 ViT-B 모델을 파인튜닝 하는 데 필요한 학습 가능한 파라미터의 1.03%와 메모리의 3.18%만 사용하면서도 여덟 개의 의료 이미지 데이터셋에서 다른 PETL 방법들보다 우수한 성능을 보여주었습니다.



### FE-Adapter: Adapting Image-based Emotion Classifiers to Videos (https://arxiv.org/abs/2408.02421)
- **What's New**: 본 연구는 이미지에서 비디오로의 크로스 모달리티 전이 학습(cross-modality transfer learning) 접근법을 소개합니다. 우리는 이를 '파라미터 효율적 이미지-비디오 전이 학습(parameter-efficient image-to-video transfer learning)'이라고 부릅니다. 이 방법을 통해 기존에 시간 처리 능력이 부족한 사전 훈련된 이미지 모델이 동적 비디오 콘텐츠를 분석할 수 있게 됩니다. 이를 위해 관리해야 하는 파라미터 수가 약 15배 줄어드는 새로운 Facial-Emotion Adapter(FE-Adapter)를 제안합니다.

- **Technical Details**: 본 연구에서는 Facial-Emotion Adapter(FE-Adapter)를 개발하였습니다. 이 Adapter는 Vision Transformer(ViT)와 같은 이미지 모델의 파라미터 일부만을 업데이트함으로써 모델의 원래 파라미터 대부분을 유지합니다. 비디오를 이해하기 위해 각 프레임의 특징을 시간에 따라 평균 풀링(average pooling)과 같은 방법으로 집계합니다. 비디오는 프레임 단위로 나눠지며, 각 프레임은 패치 크기 P×P로 분할되고, 패치 토큰으로 평탄화되어 투영됩니다.

- **Performance Highlights**: 실험 결과, FE-Adapter는 기존의 전체 파인 튜닝(fine-tuning) 방식과 최신 비디오 감정 모델에 비해 성능과 효율성이 모두 뛰어납니다. 특히, 파라미터 효율성 측면에서 FE-Adapter는 기존 방법에 비해 약 15배의 파라미터 절약 효과를 보였습니다. 이는 대규모 모델의 자원 소모 문제를 해소하며, 비디오 감정 인식 분야에서 뛰어난 성능과 유연성을 제공합니다.



### Multi-weather Cross-view Geo-localization Using Denoising Diffusion Models (https://arxiv.org/abs/2408.02408)
Comments:
          Accepted by ACM MM24 workshop

- **What's New**: 이번 연구는 GNSS가 차단된 환경에서 드론 뷰(Drone-view) 이미지와 지리적 태그가 부여된 위성 뷰(Satellite-view) 이미지를 매칭하여 위치를 파악하는 Cross-view Geo-localization을 다룹니다. 특히 기존의 악천후 상태에서의 성능 저하 문제를 해결하기 위해 새로운 Multi-weather Cross-view Geo-localization Framework(MCGF)를 소개합니다. 이 프레임워크는 이미지 복원과 위치 파악을 동시에 최적화하며, denoising diffusion 모델을 활용합니다.

- **Technical Details**: MCGF는 denoising diffusion 모델을 통해 이미지 복원과 위치 파악을 동시에 최적화합니다. 이미지 복원에는 공유 인코더와 가벼운 복원 모듈이 포함되어 있으며, 위치 파악의 백본으로는 EVA-02가 사용됩니다. EVA-02는 적은 파라미터로도 드론 및 위성 이미지에서 유리한 정보를 추출할 수 있는 ViT(Vision Transformer) 모델입니다. 학습에는 cross-entropy loss를, 테스트에는 cosine distance를 사용합니다.

- **Performance Highlights**: University160k-WX 데이터셋에서의 광범위한 실험 결과, MCGF는 다양한 날씨 조건에서도 경쟁력 있는 위치 파악 성능을 입증했습니다. 향후 MCGF의 코드가 GitHub(https://github.com/fengtt42/ACMMM24-Solution-MCGF)에 공개될 예정입니다.



### Tensorial template matching for fast cross-correlation with rotations and its application for tomography (https://arxiv.org/abs/2408.02398)
Comments:
          Accepted in The 18th European Conference on Computer Vision ECCV 2024

- **What's New**: 이 논문에서는 'Tensorial Template Matching(TTM)'이라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 템플릿 매칭(template matching)의 회전 정확도(rotation accuracy)와 상관없이 일정한 계산 복잡도를 유지합니다. 이를 통해, 3D 이미지(예: 단층촬영 이미지)에서 더 빠르고 정확한 객체 탐지가 가능해집니다.

- **Technical Details**: 전통적인 템플릿 매칭(TTM)은 회전과 변환(transformations)에 따라 교차 상관(cross-correlation)을 계산합니다. 이는 회전의 각도 정확도에 따라 계산 복잡도가 달라집니다. 그러나 TTM은 템플릿의 모든 회전을 텐서 필드(tensor field)로 표현하여 한번만 계산합니다. TTM은 대칭 텐서(symmetric tensors)를 사용하여 회전 정보를 통합하며, 이는 기존의 구면 조화 함수(spherical harmonics)와 관련이 있습니다.

- **Performance Highlights**: 실제 데이터와 합성 데이터를 사용한 실험 결과, TTM이 기존의 템플릿 매칭보다 훨씬 빠르고, 회전 정확도와 무관하게 고정된 계산 복잡도를 가지며, 객관적으로 더 나은 성능을 보였습니다.



### CMR-Agent: Learning a Cross-Modal Agent for Iterative Image-to-Point Cloud Registration (https://arxiv.org/abs/2408.02394)
Comments:
          Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024

- **What's New**: 이번 논문에서는 이미지와 포인트 클라우드 간의 등록을 반복적인 마르코프 결정 과정(Markov decision process)으로 재구성하여, 각 중간 상태에 따른 카메라 포즈를 점진적으로 조정할 수 있는 새로운 방법을 제안합니다. 이를 위해 강화 학습(reinforcement learning)을 이용해 Cross-Modal Registration Agent (CMR-Agent)을 개발하고, 임의 학습(imitation learning)을 통해 등록 정책을 초기화하여 안정성 및 빠른 학습 시작을 보장합니다.

- **Technical Details**: CMR-Agent는 2D-3D 하이브리드 상태 표현을 사용하여 RGB 이미지의 세밀한 특징을 최대한 활용하고, 카메라 프러스툼의 공간적 절단으로 인한 무용지물을 줄입니다. 일회성 크로스 모달 임베딩(one-shot cross-modal embeddings)을 효율적으로 재사용하여 반복적인 기능 추출을 피하고 시간 복잡성을 감소시킵니다. 이러한 접근 방식을 통해 포인트 클라우드와 이미지를 효과적으로 비교하고, 강화 학습의 PPO 알고리즘에 의한 포인트 간 정렬 보상(point-to-point alignment reward)을 사용하여 에이전트를 훈련합니다.

- **Performance Highlights**: KITTI-Odometry 및 NuScenes 데이터셋에 대한 광범위한 실험을 통해 CMR-Agent가 등록 정확도 면에서 최첨단 방법들을 능가한다는 것을 입증했습니다. 비록 반복적인 방법이지만, 일회성 알고리즘보다 더 높은 효율성을 보여주었으며, 3090 GPU에서 10회 반복 시 68 밀리초 이내에 실행되는 성과를 보였습니다.



### MaFreeI2P: A Matching-Free Image-to-Point Cloud Registration Paradigm with Active Camera Pose Retrieva (https://arxiv.org/abs/2408.02392)
Comments:
          Accepted to IEEE Conference on Multimedia Expo 2024

- **What's New**: 이번 논문에서는 기존의 매칭 기반 방법들이 가지는 정보 손실 문제를 해결하기 위해 새로운 매칭-프리 이미지-to-포인트 클라우드 등록 방법인 MaFreeI2P를 제안합니다. 핵심 아이디어는 포인트 클라우드와 쿼리 이미지 사이의 기하학적 특징을 대조하여 SE(3) 공간에서 카메라 자세를 적극적으로 검색하는 것입니다.

- **Technical Details**: MaFreeI2P는 초기 카메라 자세 주위로 후보 자세 집합을 샘플링하고, 크로스-모달 특징을 사용하여 비용 볼륨(cost volume)을 구성합니다. 각 후보 자세에 대해 포인트 클라우드를 다양한 2D 뷰로 투영하고, 2D 및 3D 특징을 결합하여 비용 볼륨 유닛을 형성합니다. 이후 컨벌루션 네트워크(convolutional network)를 통해 유사도 평가 함수를 적응적으로 만들고, 포즈 기반 가중치를 사용하여 유사도 점수를 계산합니다. 마지막으로, 유사도가 가장 높은 자세로 현재 카메라 자세를 업데이트하고 점진적으로 포즈 샘플링 공간을 축소합니다.

- **Performance Highlights**: MaFreeI2P는 KITTI-Odometry와 Apollo-DaoxiangLake 데이터셋에서 매우 경쟁력 있는 등록 정확도와 리콜을 달성했습니다. 비용 볼륨을 통해 매칭 기반 파이프라인보다 더 많은 정보를 보존할 수 있었으며, RANSAC 기반 PnP 솔버가 존재하지 않음에도 불구하고 전역적인 고려를 하여 최적의 포즈를 찾아냈습니다.



### Cross Psuedo Supervision Framework for Sparsely Labelled Geo-spatial Images (https://arxiv.org/abs/2408.02382)
- **What's New**: 본 연구는 인도의 광범위한 지역에서 고해상도 위성 이미지를 사용한 LULC(Land Use Land Cover) 예측을 위한 반지도 학습 기반의 세분화 모델을 소개합니다. 이를 통해 다양한 건물 유형, 도로, 나무, 수역 등을 최적의 일반화된 방식으로 예측할 수 있습니다. 구체적으로는 'Cross Pseudo Supervision' 기법을 수정하여, 드문드문 라벨링된 데이터를 효과적으로 활용할 수 있도록 개선된 프레임워크를 제안합니다.

- **Technical Details**: 제안된 모델 아키텍처는 여러 주요 단계를 거칩니다. 첫 번째로, 벡터 데이터를 기반으로 각 위성 이미지의 래스터 마스크를 생성합니다. 이 연구에서는 6개의 고해상도 NRGB(근적외선, 적색, 녹색 및 청색 밴드) 위성 이미지 장면을 사용했습니다. 벡터 데이터는 훈련 및 평가 세트에서 다르게 획득하였습니다. 훈련용 벡터 파일은 방갈로르, 뭄바이, 푸네, 바라나시 및 델리의 도시에서 수집되었으며, 평가 데이터는 하이데라바드 소지역의 데이터를 포함합니다. 이번 연구에서는 다양한 정의를 가진 UNet 및 DeepLabV3+ 아키텍처를 사용하여 비교 연구를 진행했습니다.

- **Performance Highlights**: 제안된 Cross Pseudo Supervision 기법은 두 개의 서로 다른 초기화된 세그멘테이션 네트워크를 사용하여 일관성을 강화합니다. 이에 따라, 훈련 데이터는 희소하게 라벨링되고, 동일한 배치의 라벨된 데이터가 감독 및 비감독 손실을 모두 계산하는 데 사용됩니다. 이런 접근법은 Xiaokang Chen 등의 기존 연구와는 다릅니다. 특히, 제안된 수정된 프레임워크는 정확성과 유틸리티 면에서 LULC 매핑의 성능을 향상시킵니다.



### The NPU-ASLP System Description for Visual Speech Recognition in CNVSRC 2024 (https://arxiv.org/abs/2408.02369)
Comments:
          2 pages, 2 figures, CNVSRC 2024 System Report

- **What's New**: 이 논문은 NPU-ASLP(237팀)의 시각 음성 인식(VSR) 시스템을 CNVSRC 2024 대회에서 소개했습니다. 본 시스템은 단일 화자 및 다중 화자 VSR 과제의 고정 및 열린 트랙에 참여하며, 데이터 처리와 다양한 증강 기술을 통해 멀티스케일 비디오 데이터를 생성합니다.

- **Technical Details**: 데이터 처리를 위해 기본 lip motion extractor를 사용하여 멀티스케일 비디오 데이터를 생성하고, 다양한 증강 기법(speed perturbation, random rotation, horizontal flipping, color transformation)을 적용합니다. VSR 모델은 joint CTC/attention loss를 사용하는 end-to-end 아키텍처를 채택하고, Enhanced ResNet3D visual frontend, E-Branchformer encoder, Bi-directional Transformer decoder를 도입합니다.

- **Performance Highlights**: 본 접근법은 단일 화자 과제에서 30.47% CER, 다중 화자 과제에서 34.30% CER를 기록하여 단일 화자 과제의 오픈 트랙에서는 2위를, 나머지 세 트랙에서는 1위를 차지했습니다.



### Earth System Data Cubes: Avenues for advancing Earth system research (https://arxiv.org/abs/2408.02348)
- **What's New**: 최근 Earth system science에서 다양한 다변량 데이터셋의 사용이 급증하면서, Earth System Data Cubes (ESDCs)가 중요한 해결책으로 부상하고 있습니다. 이러한 데이터 큐브는 복잡하고 방대한 데이터를 사용자 친화적 포맷으로 정리하여 분석을 간편하게 만들어줍니다.

- **Technical Details**: ESDCs는 스페이셜(Spatial)-템포럴(Temporal) 그리드에 맞춘 블록 형태로 데이터를 구성하여, 분석 가능한 데이터 스트림으로 변환합니다. 이는 위성 원격 탐사 및 기후 모델 결과와 같은 대규모 배열 데이터를 효과적으로 관리할 수 있습니다. ESDCs 발전을 위해 클라우드 환경과 FAIR(Open Science) 원칙을 적용한 데이터 형식 및 공유 프로토콜 개발이 필요합니다.

- **Performance Highlights**: ESDCs의 도입으로 식생 반응, 기후 드라이버 분석, 극한 이벤트 감지 등 다양한 Earth system 연구에서 큰 성과를 내고 있습니다. 예를 들어, 인공 신경망(Recurrent Neural Network, RNN)을 사용해 식생 반응을 학습하거나, 가뭄의 영향 분석에 활용되고 있습니다. 하지만 ESDCs의 잠재력을 온전히 발휘하기 위해서는 데이터의 변형 및 분석 시 발생할 수 있는 오차를 방지해야 하며, 물리적 제약과 도메인 지식을 통합하는 것이 중요합니다.



### Infusing Environmental Captions for Long-Form Video Language Grounding (https://arxiv.org/abs/2408.02336)
Comments:
          7 pages, 3 figures

- **What's New**: 이번 연구에서는 Long-Form Video-Language Grounding (LFVLG) 문제를 해결하기 위해 EI-VLG라는 새로운 방법을 제안합니다. 이 방법은 인간의 경험을 대변할 수 있는 Multi-modal Large Language Model (MLLM)을 활용하여 환경 정보를 풍부하게 제공함으로써 불필요한 프레임을 효과적으로 제외할 수 있게 합니다. 제안된 방법의 효과는 EgoNLQ 벤치마크에서의 광범위한 실험을 통해 검증되었습니다.

- **Technical Details**: EI-VLG는 세 가지 주요 구성 요소로 이루어졌습니다: i) 환경 인코더(EE), ii) 비디오-언어 그라운딩 모델(VLG), iii) 환경 인퓨저(EI). EE는 주어진 비디오의 짧은 간격마다 캡션을 생성하고 텍스트 인코더를 사용해 이를 인코딩합니다. 이렇게 생성된 환경 큐들은 VLG 모델에 주입되어서 비디오의 특정 순간을 정확히 찾는데 도움을 줍니다. 이 과정에서 off-the-shelf MLLM(LLaVA 34B)이 사용되며, 질의와 환경 캡션 간의 유사성을 높이기 위해 대조 학습 목적 함수를 사용하여 텍스트 인코더를 미세 조정합니다.

- **Performance Highlights**: EgoNLQ 벤치마크에서 EI-VLG는 최첨단 성능을 보여주었습니다. 이는 MLLM이 생성한 캡션들이 제공하는 풍부한 문맥적 설명을 활용하여 질의와 비디오 간의 미세한 차이점을 효과적으로 포착할 수 있었기 때문입니다. 실험을 통해 기존 방법들보다 뛰어난 성능을 입증하였으며, 제안된 방법은 다양한 고충격 응용 프로그램에서도 유용할 것으로 기대됩니다.



### Low-Cost Self-Ensembles Based on Multi-Branch Transformation and Grouped Convolution (https://arxiv.org/abs/2408.02307)
- **What's New**: 최신 연구는 저비용 앙상블 학습이 이미지 분류에서 효율성이 향상되었음을 보여주었다. 하지만 기존 저비용 앙상블 방법들은 기존의 앙상블 학습에 비해 정확도가 떨어지는 경향이 있다. 본 논문에서는 고효율성과 높은 분류 성능을 동시에 달성할 수 있는 새로운 저비용 앙상블 학습법을 제안한다. 이 방법은 추가적인 구성 요소를 도입하지 않고 기존 CNN을 다중분기 구조로 변형시켜 원래 모델과 동일한 계산 복잡도를 유지하면서도 각 분기의 출력 간 충분한 분리로 다양성을 강화한다.

- **Technical Details**: 제안된 방법은 기존 CNN을 다중분기 구조로 변형하고, 각 분기에 그룹화된 convolution을 다른 개수로 적용하여 다양성을 높인다. 학습 중에는 앙상블 출력을 교사 신호로 사용하는 지식 증류(knowledge distillation)를 적용하여 각 분기의 분류 성능을 향상시킨다. 이는 각 분기의 높아진 다양성 덕분에 강력한 교사 신호를 형성할 수 있게 한다. 이러한 접근법은 단일 모델의 계산 비용으로도 고성능 앙상블을 구축할 수 있도록 한다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 데이터셋과 네트워크 구조에서 최신 분류 정확도를 달성했으며, 이전의 저비용 앙상블 방법들에 비해 높은 불확실성 추정 성능을 보여주었다. 제안된 방법은 SEMBG (Self-Ensembles using Multi-Branch and Grouped Convolution)로 명명되었으며, 코드는 제공된 링크에서 확인할 수 있다.



### Mixture-of-Noises Enhanced Forgery-Aware Predictor for Multi-Face Manipulation Detection and Localization (https://arxiv.org/abs/2408.02306)
- **What's New**: 이 논문에서는 다중 얼굴 변조 이미지 검출 및 위치 파악을 위한 새로운 프레임워크인 MoNFAP를 제안합니다. 특히, 두 가지 혁신 모듈인 Forgery-aware Unified Predictor (FUP) 모듈과 Mixture-of-Noises Module (MNM)을 도입했습니다. FUP는 Token Learning 전략과 여러 Forgery-aware Transformers를 통해 검출과 위치 파악 작업을 통합하여, 분류 정보를 위치 파악 능력을 강화하는 데 사용합니다. MNM은 다양한 노이즈 추출기를 이용해 일반적인 RGB 피처를 강화하여 성능을 높입니다.

- **Technical Details**: MoNFAP는 다중 얼굴 변조 이미지의 검출 및 위치 파악을 위해 설계되었습니다. Forgery-aware Unified Predictor (FUP) 모듈은 Token Learning 전략과 여러 Forgery-aware Transformers (FAT)를 통해 검출과 위치 파악 작업을 결합합니다. FAT 모듈은 진짜와 가짜 카테고리를 나타내는 두 개의 학습 가능한 토큰을 포함하고 있으며, 이 토큰들과 이미지 피처는 토큰 셀프 어텐션 및 토큰-이미지 크로스 어텐션을 통해 양방향으로 업데이트됩니다. Mixture-of-Noises Module (MNM)은 mixture of experts (MoE) 철학에 영감을 받아 다양한 노이즈 유형의 장점을 활용합니다. 여러 노이즈 추출기를 사용하여 일반적인 RGB 피처를 강화하여 위조 이미지 패턴을 증강합니다.

- **Performance Highlights**: 제안된 MoNFAP는 다중 얼굴 변조 이미지 검출과 위치 파악에서 최첨단 성능을 달성했습니다. FUP 모듈은 분류 정보를 통해 위치 파악 성능을 향상시키며, MNM 모듈은 다양한 노이즈 패턴을 사용하여 작은 변조 영역을 효과적으로 로컬라이즈합니다. 이를 통해 MoNFAP는 종합적인 다중 얼굴 변조 검출 및 위치 파악 벤치마크에서 우수한 성능을 보였습니다.



### Network Fission Ensembles for Low-Cost Self-Ensembles (https://arxiv.org/abs/2408.02301)
- **What's New**: 최근 이미지 분류를 위한 앙상블 학습(ensemble learning) 방법들은 적은 추가 비용으로 분류 정확도를 높일 수 있는 것으로 나타났습니다. 그러나 기존 방법들은 앙상블 출력(ensemble inference)을 위해 여러 개의 학습된 모델을 요구하며, 모델 크기가 커지면 이는 큰 부담이 됩니다. 본 논문에서는 기존 네트워크를 다중 출구(multi-exit) 구조로 변환하여 저비용 앙상블 학습 및 추론을 가능하게 하는 Network Fission Ensembles(NFE)를 제안합니다. 이를 통해 별도의 네트워크 없이 하나의 네트워크에서 여러 출력을 통해 앙상블 학습이 가능합니다.

- **Technical Details**: NFE는 초기 네트워크에서 일단 일부 가중치를 가지치기(weight pruning)하여 학습 부담을 줄입니다. 그 후 남은 가중치들을 그룹화하여 각각의 그룹에 대해 보조 경로(auxiliary paths)를 만들어 다중 출구를 구성합니다. 이를 Network Fission이라 부릅니다. 기존 네트워크에 추가적인 모듈 없이 구조만 변경하여 다중 출구를 만들기 때문에 앙상블 학습과 추론에 추가적인 계산 부담이 없습니다. 또한 여러 출구의 학습 손실(loss)를 함께 최적화함으로써 정규화(regularization)를 통해 성능이 향상됩니다.

- **Performance Highlights**: NFE는 기존의 저비용 앙상블 방법들과 비교하여 월등한 성능 향상을 보였습니다. 특히 가지치기를 통해 네트워크의 스파시티(sparsity)가 증가하더라도 높은 성능을 유지할 수 있음을 보여줍니다. 실험 결과, NFE는 거의 비용 없는 앙상블 학습 및 추론을 가능하게 하며, 최신의 다른 앙상블 방법들과 비교하여 매우 만족스러운 성능을 달성했습니다.



### SelfGeo: Self-supervised and Geodesic-consistent Estimation of Keypoints on Deformable Shapes (https://arxiv.org/abs/2408.02291)
Comments:
          This paper has been accepted in ECCV 2024

- **What's New**: 비지도 학습(unsupervised) 방식으로 3D 키포인트(keypoints)를 추정하는 'SelfGeo' 메서드를 소개합니다. 이 방법은 기존의 사람의 주석 없이 포인트 클라우드 데이터(Point Cloud Data, PCD)에서 비강체(non-rigid) 객체의 지속적인 3D 키포인트들을 추정할 수 있습니다.

- **Technical Details**: SelfGeo는 변형하는 몸체의 불변 속성을 존중하여 프레임 간의 키포인트를 추정합니다. 이 방법의 핵심 기여는 키포인트가 형상과 함께 변형되며 키포인트 간의 지오데식 거리(geodesic distances)를 일정하게 유지하는 것입니다. 이 원칙은 손실 함수 디자인에 반영되어, 손실 함수를 최소화함으로써 특정한 의미 있는 위치에 반복 가능한 키포인트가 나타나게 됩니다. 트레이닝 동안 입력된 PCD 시퀀스로부터 각 PCD에 대한 키포인트를 추정하며, 지오데식 거리 유지와 연속 프레임 간의 쌍 정규화를 위한 손실 함수를 도입합니다.

- **Performance Highlights**: SelfGeo는 인간과 동물과 같은 다양한 변형하는 객체 클래스에서 뛰어난 성능을 보입니다. CAPE 및 Deforming Things 4D 데이터셋에서의 실험 결과 이 접근 방식이 어떠한 변형 형태에도 일반화될 수 있음을 보여주었습니다. 또한, 실제 ITOP 데이터셋과의 실험에서도 이 방법의 강건성을 입증하였으며, 잡음이 있거나 축소된 PCD에서도 견고함을 유지하였습니다.



### Joint-Motion Mutual Learning for Pose Estimation in Videos (https://arxiv.org/abs/2408.02285)
Comments:
          10 pages, 5 figures

- **What's New**: 비디오에서 인간 자세 추정(human pose estimation)의 새로운 접근법, JM-Pose(Joint-Motion Mutual Learning for Pose Estimation)를 제안합니다. 이 프레임워크는 로컬 관절 의존성(local joint dependency)과 글로벌 픽셀 수준의 모션 다이내믹(global pixel-level motion dynamics)을 효과적으로 통합하여, 현재 방법론들이 자주 겪는 성능 저하 문제를 해결합니다.

- **Technical Details**: 본 연구는 두 가지 주요 구성 요소를 포함합니다. 첫째, Context-Aware Joint Learner을 도입하여 초기 히트맵(initial heatmap)과 모션 플로우(motion flow)를 사용하여 로컬 관절 특성(local joint feature)을 추출합니다. 이는 모듈화 변형 연산(modulated deformable operations)을 이용하여 로컬 관절 문맥 특성을 캡쳐합니다. 둘째, Progressive Joint-Motion Mutual Learning을 통해 로컬 관절 특성과 글로벌 모션 플로우가 동적으로 정보를 교환하고 상호 학습합니다. 또한, 정보의 중복 학습을 피하기 위해 Information Orthogonality Objective를 제안하여 다양한 관절 및 모션 큐를 캡처합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에서 종합적인 실험을 수행한 결과, JM-Pose 방법은 state-of-the-art 방법들을 일관되게 능가하는 성능을 보였습니다.



### Cascading Refinement Video Denoising with Uncertainty Adaptivity (https://arxiv.org/abs/2408.02284)
- **What's New**: 본 논문에서는 비디오 노이즈 제거를 위한 새로운 방법인 casc**ading refinement** 비디오 노이즈 제거 기법을 소개합니다. 이 방법은 정렬(alignment)과 이미지 복원을 동시에 수행하며, CRVD 데이터셋에서 SOTA(State-of-the-art) 성능을 달성했습니다. 또한, 반복마다 불확실성 맵(uncertainty map)을 생성하여 불필요한 계산을 줄였습니다.

- **Technical Details**: 이 방법은 저조도 환경 등 다양한 노이즈 상황에서도 정렬과 이미지를 동시에 반복적으로 정제하는 구조를 가지고 있습니다. Optical flow estimation을 위해 반복적인 정제 구조를 채택하였으며, flow-guided deformable convolution을 사용해 오프셋 추정을 다양화합니다. 또한, 정렬된 프레임의 특징과 기준 프레임의 특징을 융합하여 다음 반복의 입력으로 사용합니다. 불확실성 맵을 이용해 나중 정제가 필요한지 판별하고, 필요 시 추가 연산을 합니다.

- **Performance Highlights**: CRVD 데이터셋에서 SOTA 성능을 큰 폭으로 달성했으며, 전체 계산량을 평균 25% 줄였습니다. 이는 비디오 노이즈 제거 작업에서 정렬과 복원을 동시에 처리함으로써 성능과 효율성을 크게 향상시켰습니다.



### Geometric Algebra Meets Large Language Models: Instruction-Based Transformations of Separate Meshes in 3D, Interactive and Controllable Scenes (https://arxiv.org/abs/2408.02275)
Comments:
          17 pages, 8 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)과 공형 기하학 대수(CGA)를 결합한 새로운 시스템 'shenlong'을 소개하여, 정밀한 3D 장면 편집, 특히 객체 재배치 작업을 혁신적으로 개선합니다. 전통적으로 이 작업은 복잡한 수작업과 전문 지식을 요구했으나, 'shenlong'은 자연어 명령을 CGA 작업으로 변환하여 정확한 공간 변환을 수행합니다.

- **Technical Details**: shenlong 시스템은 CGA를 견고한 형식 언어로 사용하여 공간 변환을 모델링하며, 사전 훈련된 LLM의 zero-shot learning 능력을 활용합니다. 이를 통해 특정 훈련 없이 다양한 3D 환경에서 자연어 지시를 정확하게 해석하고 반영할 수 있습니다. 이 시스템은 ThreeDWorld(TDW) 프레임워크에 통합되어 Unity3D 엔진과 호환됩니다.

- **Performance Highlights**: shenlong은 기존의 LLM 기반 대안들에 비해 객체 재배치 시 LLM 응답 시간을 평균 16% 감소시키고 성공률을 9.6% 향상시킵니다. 특히, 일반적인 실용적 쿼리에서는 100% 완벽한 성공률을 달성하여 다른 시스템이 따라올 수 없는 성능을 보입니다.



### COM Kitchens: An Unedited Overhead-view Video Dataset as a Vision-Language Benchmark (https://arxiv.org/abs/2408.02272)
Comments:
          ECCV2024 accepted

- **What's New**: 최근 비전과 언어 커뮤니티에서 절차적 비디오 이해에 대한 관심이 증가하는 가운데, 효율적인 데이터 수집이 중요해지고 있습니다. 이를 해결하기 위해 새로운 데이터셋 'COM Kitchens'를 제안합니다. 이 데이터셋은 스마트폰을 사용하여 오버헤드 뷰에서 촬영된 음식 준비 과정을 담고 있습니다. 이 데이터셋을 통해 새로운 온라인 레시피 검색(OnRR)과 오버헤드 뷰 비디오 캡션링(DVC-OV) 과제를 소개합니다.

- **Technical Details**: COM Kitchens 데이터셋은 스마트폰의 넓은 시야각 렌즈를 활용하여 부엌의 전체 작업대를 오버헤드 뷰로 촬영했습니다. 이 셋업으로 다양한 환경에서 145개의 비디오, 총 40시간 분량의 데이터를 수집했습니다. 비디오와 텍스트 지시사항을 작업 흐름 그래프로 연결하는 수동 주석을 제공하여 비디오-텍스트 검색과 밀집 비디오 캡션링(DVC) 과제를 정의했습니다.

- **Performance Highlights**: COM Kitchens 데이터셋을 통해 현재 웹 비디오 기반의 SOTA(State-of-the-Art) 방법들이 이러한 새로운 과제를 어떻게 처리하는지 실험으로 검증했습니다. 이 데이터셋은 기존의 제조 작업을 타겟으로 한 다른 절차적 비디오 데이터셋들에 비해 더 다양한 작업과 환경을 포함하며, 언어적 주석이 포함된 유일한 데이터셋입니다. 새로운 비디오-텍스트 검색(Online Recipe Retrieval, OnRR) 과제와 밀집 비디오 캡션링(DVC-OV) 과제를 통해 스마트폰 비디오에도 적용 가능한 기술 개발을 목표로 하고 있습니다.



### Explain via Any Concept: Concept Bottleneck Model with Open Vocabulary Concepts (https://arxiv.org/abs/2408.02265)
Comments:
          ECCV2024

- **What's New**: CBM (Concept Bottleneck Model)에서 개선된 'OpenCBM'을 제안합니다. 이는 고정된 개념 대신 개방형 어휘 개념을 사용할 수 있으며, 모델이 학습된 후에도 사용자가 원하는 개념을 추가, 제거 또는 교체할 수 있는 유연성을 제공합니다. 특히 이미지를 해석 가능한 텍스트 형식의 개념으로 클래스 예측을 조정할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: 'OpenCBM' 모델은 다음과 같은 세 가지 주요 단계로 구성됩니다: (1) 훈련 이미지 특성 추출기 (trainable image feature extractor)의 특성 공간을 CLIP의 이미지 인코더 (image encoder)의 특성 공간과 정렬하여 프로토타입 기반 특성 정렬을 수행, (2) 다운스트림 데이터셋에서 이미지 분류기를 훈련, (3) 원하는 텍스트 기반 개념을 CLIP의 텍스트 인코더 (text encoder)로 인코딩해 분류 헤드를 재구성합니다. 또한 사용자가 지정한 개념 집합에서 누락된 개념을 찾아 성능을 회복하도록 잔여 매개변수의 가장 가까운 개념 임베딩을 반복적으로 찾는 메커니즘을 제안합니다.

- **Performance Highlights**: OpenCBM은 기존 최첨단 CBM 모델보다 CUB-200-2011 벤치마크 데이터셋에서 분류 정확도가 9% 향상되었습니다. 이는 모델의 해석 가능성을 높이는 동시에 성능도 개선할 수 있음을 증명합니다.



### VoxelTrack: Exploring Voxel Representation for 3D Point Cloud Object Tracking (https://arxiv.org/abs/2408.02263)
- **What's New**: 새로운 LiDAR 포인트 클라우드 기반 3D 단일 객체 추적(SOT) 프레임워크인 VoxelTrack을 소개합니다. VoxelTrack은 본질적으로 무질서한 포인트 클라우드를 3D 복셀(voxels)로 변환하고 희소 컨볼루션 블록을 통해 기능(feature)을 추출합니다. 이를 통해 정밀하고 강력한 3D 공간 정보를 모델링하여 추적 객체의 정확한 위치 예측을 지원합니다.

- **Technical Details**: VoxelTrack은 복셀화된 포인트 클라우드를 입력으로 사용하여 3D SOT를 위해 복셀 기반 표현 네트워크를 채택합니다. 구체적으로 두 연속 프레임의 포인트 클라우드를 복셀화하고 공간적으로 정렬하여 희소 컨볼루션 블록을 통해 복셀 기능을 추출합니다. 3D 공간 정보를 최대한 활용하기 위해 듀얼 스트림 인코더(dual-stream encoder)와 교차 반복 특징 융합 모듈(cross-iterative feature fusion module)을 결합하여 보다 정교한 3D 공간 정보를 탐색합니다.

- **Performance Highlights**: VoxelTrack은 KITTI, NuScenes 및 Waymo Open Dataset 세 가지 데이터셋에서 광범위한 실험을 수행했습니다. 실험 결과, VoxelTrack은 88.3%, 71.4% 및 63.6%의 평균 정밀도를 달성하며, 기존 추적기보다 뛰어난 성능을 보였습니다. 또한, 단일 TITAN RTX GPU에서 36fps의 실시간 속도를 자랑합니다.



### Cross-Domain Semantic Segmentation on Inconsistent Taxonomy using VLMs (https://arxiv.org/abs/2408.02261)
Comments:
          ECCV 2024

- **What's New**: 이번 연구는 기존의 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 방법이 전제하는 소스와 타겟 도메인의 클래스 체계(class taxonomy) 일관성 문제를 극복하기 위한 새로운 접근법을 제안합니다. 이 논문은 Vision Language Models(VLMs)의 의미 일반화 능력을 활용하여 클래스 세분화와 새로운 클래스를 효과적으로 분류하는 CSI(Cross-Domain Semantic Segmentation on Inconsistent Taxonomy)를 소개합니다. 이는 타겟 도메인의 실제 클래스 체계 차이를 해결하는 능력을 지니며, 레이블이 없는 타겟 도메인에도 적응할 수 있습니다.

- **Technical Details**: CSI는 기존 UDA 방법에서 얻은 세그멘테이션 추론(segment reasoning)와 VLMs에 내재된 의미 지식을 결합하여 타겟 도메인의 새로운 클래스를 리레이블링합니다. OWL-ViT와 CLIP을 기반으로 한 제로-샷 리레이블링(zero-shot relabeling) 기술을 사용하여 불일치하는 클래스 체계를 다루며, 소스-타겟 간의 클래스 세분화(coarse-to-fine taxonomy)와 열린 클래스(open taxonomy) 문제를 해결합니다. 이 방법은 기존 최첨단 UDA 방법들과의 시너지 효과를 증명하였습니다.

- **Performance Highlights**: 다양한 벤치마크에서 진행된 실험을 통해 CSI 방법의 유효성이 입증되었습니다. 이 연구는 제로-샷 리레이블링이 UDA에서 클래스 체계 불일치 문제를 해결할 수 있음을 처음으로 보였으며, 소스 도메인과 타겟 도메인의 공통 클래스뿐만 아니라 타겟 도메인에만 있는 클래스까지도 적응 성능을 향상시켰습니다. 기존의 여러 UDA 모델과의 호환성도 확인되었습니다.



### Curriculum learning based pre-training using Multi-Modal Contrastive Masked Autoencoders (https://arxiv.org/abs/2408.02245)
- **What's New**: 이번 논문에서는 RGB-D 데이터를 활용한 이미지 이해 작업을 위한 새로운 사전 학습 방법을 제안합니다. 이 방법은 Curriculum Learning(CL) 패러다임 하에서 Multi-Modal Contrastive Masked Autoencoder와 Denoising(노이즈 제거) 기술을 활용합니다. 기존의 방법론과 달리, 이 논문은 RGB-D 데이터셋에 적용 가능한 단일 대조 마스크드 오토인코더를 새롭게 개발하여 여러 데이터셋에서 뛰어난 성능을 입증합니다.

- **Technical Details**: 제안된 방법론은 두 단계의 사전 학습 전략을 따릅니다. 첫 번째 단계에서는 대조 학습(Contrastive Learning)을 통해 교차 모달리티(Cross-Modal) 표현을 학습합니다. 두 번째 단계에서는 첫 번째 단계에서 학습된 가중치를 사용하여 마스크드 오토인코딩(Masked Autoencoding)과 노이즈 예측(Denoising)을 통한 학습을 진행합니다. 각 입력 모달리티의 누락된 패치를 재구성하는 마스크드 오토인코딩과 입력 데이터의 고주파수 성분을 학습하는 노이즈 예측으로 구성됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터셋(예: ScanNet, NYUv2, SUN RGB-D)에서 탁월한 성능을 보였습니다. 특히, ScanNet의 의미론적 분할 작업에서 Mask3D 대비 +1.0% mIoU 향상을 달성했습니다. 또한, 저데이터 환경에서의 의미론적 분할 작업에서도 최첨단 방법들과 비교하여 보다 효과적인 성능을 나타냈습니다.



### Evaluating Vision-Language Models for Zero-Shot Detection, Classification, and Association of Motorcycles, Passengers, and Helmets (https://arxiv.org/abs/2408.02244)
- **What's New**: 본 연구는 모터사이클 승차자의 헬멧 착용 상태를 비디오 데이터를 통해 감지하고 분류하기 위해 OWLv2와 CNN 모델들을 통합한 선진 비전-언어 기반 모델을 평가합니다. CVPR AI City Challenge에서 제공된 데이터셋을 확장하고, 검출 및 분류 작업을 위한 계단식 모델 접근 방식을 사용합니다.

- **Technical Details**: OWLv2 모델 및 CNN 모델을 사용한 계단식 모델 접근 방식을 채택하여 검출 및 분류 작업을 수행합니다. 제안된 접근 방식은 'zero-shot learning'을 활용하여 불완전하거나 편향된 학습 데이터셋에 의해 발생하는 문제를 해결합니다. OWLv2는 'open-world localization' 기능을 갖춘 Vision Transformer 기반 모델로, 다양한 조건 하에서 모터사이클, 헬멧 사용, 탑승자 위치를 감지하는 데 사용됩니다.

- **Performance Highlights**: 헬멧 감지의 평균 정밀도는 0.5324로 나타났으며, 검출 및 분류 성능을 상세히 설명하는 정밀도-재현율 곡선을 제공합니다. 낮은 해상도 데이터 및 가시성이 좋지 않은 상황에서도 유망한 결과를 도출하여 자동화된 차량 안전 및 교통안전 집행 시스템에 있어 중요한 진전을 보여줍니다.



### REVISION: Rendering Tools Enable Spatial Fidelity in Vision-Language Models (https://arxiv.org/abs/2408.02231)
Comments:
          Accepted to ECCV 2024. Project Page : this https URL

- **What's New**: 텍스트-이미지(Text-to-Image, T2I) 모델과 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)은 여러 컴퓨터 비전 및 멀티모달 학습 과제에서 채택되었으나, 이러한 비전-언어 모델들이 공간적 관계에 대한 올바른 추론 능력이 부족하다는 점이 문제로 지적되었습니다. 이를 해결하기 위해 REVISION 프레임워크를 개발하였는데, 이를 통해 비전-언어 모델의 공간적 정확도를 높일 수 있습니다.

- **Technical Details**: REVISION은 3D 렌더링 기반 파이프라인으로, 텍스트 프롬프트를 기반으로 공간적으로 정확한 합성 이미지를 생성할 수 있습니다. 현재 100개 이상의 3D 자산과 11개의 공간적 관계를 지원하며, 다양한 카메라 시점과 배경을 포함합니다. 블렌더(Blender) 소프트웨어를 활용하여 텍스트 프롬프트를 자산과 관계로 구분하고, 이를 기반으로 신을 정확히 구성합니다.

- **Performance Highlights**: REVISION을 통해 생성된 이미지를 추가적인 지침으로 활용함으로써 기존 T2I 모델의 공간적 일관성을 향상시킵니다. VISOR와 T2I-CompBench 벤치마크에서 경쟁력 있는 성능을 보여주며, 공간적 관계 이해도가 크게 향상되었습니다. RevQA라는 새로운 질문-답변 벤치마크를 개발하여, MLLMs의 복잡한 공간 추론 능력을 평가한 결과, 최신 모델들이 어려움을 겪고 있으며, 적대적 설정에서 견고하지 못함을 발견했습니다.



### ProCreate, Don\'t Reproduce! Propulsive Energy Diffusion for Creative Generation (https://arxiv.org/abs/2408.02226)
Comments:
          Accepted for ECCV 2024. Project page: this https URL

- **What's New**: 이번 논문에서는 확산 기반 이미지 생성 모델의 샘플 다양성과 창의성을 향상하고, 훈련 데이터의 복제를 방지하는 간단하고 구현하기 쉬운 방법인 ProCreate를 제안합니다. ProCreate는 참조 이미지 세트를 바탕으로 생성된 이미지 임베딩을 참조 임베딩으로부터 멀어지도록 적극적으로 추진합니다. FSCG-8이라는 Few-Shot Creative Generation 데이터셋을 제안하고, ProCreate가 가장 높은 샘플 다양성과 충실도를 달성했다고 보고 있습니다. 또한, 대규모 평가를 통해 ProCreate가 훈련 텍스트 프롬프트를 사용해 훈련 데이터 복제를 효과적으로 방지한다는 것을 보여줍니다.

- **Technical Details**: ProCreate는 생성 과정에서 생성된 이미지 임베딩을 참조 이미지 세트의 임베딩으로부터 멀어지도록 하는 에너지 기반(energy-based) 방법을 사용합니다. 두 가지 실험 환경에서 테스트를 진행하였으며, 이는 몇 장의 이미지만을 사용한 창의적인 생성과 훈련 데이터 복제 방지입니다. FSCG-8 데이터셋은 여덟 가지 다양한 카테고리의 이미지를 포함하여 제안되었으며, ProCreate는 저데이터 환경에서도 샘플 다양성을 유지하면서 높은 수준의 개념적 유사성을 지키고 있습니다. 이는 창의적인 이미지 생성에 중요한 두 가지 요소입니다.

- **Performance Highlights**: Few-Shot Generation에서 ProCreate는 기존 방법보다 더 나은 샘플 다양성을 가지면서도 참조 세트와 높은 유사성을 유지했습니다. 훈련 데이터 복제 방지 실험에서 ProCreate는 사전 훈련된 확산 모델보다 훈련 데이터를 복제할 가능성이 현저히 낮았습니다.



### Cross-modulated Attention Transformer for RGBT Tracking (https://arxiv.org/abs/2408.02222)
Comments:
          10 pages, 5 figures

- **What's New**: 이번 연구에서는 RGBT 트래킹(RGBT tracking) 문제를 해결하기 위해 Cross-modulated Attention Transformer (CAFormer)을 제안했습니다. 기존 Transform-based RGBT 트래커들이 단일 모달 특징을 추출하기 위해 셀프 어텐션(self-attention)을 활용하고, 멀티 모달(feature interaction)을 향상하기 위해 크로스 어텐션(cross-attention)을 사용하였지만, 이는 독립적인 검색-템플릿 상관 계산으로 인해 정확하지 않은 상관 가중치의 발생 문제가 있었습니다. CAFormer은 이를 해결하기 위해 intra-modality 자신연관(self-correlation)과 inter-modality 피쳐 상호작용(feature interaction)을 하나의 주의 메커니즘(attention mechanism)에 통합하여 작업합니다.

- **Technical Details**: CAFormer은 독립적으로 각 모달리티의 상관 맵(correlation maps)을 생성하고 이들을 Correlation Modulated Enhancement (CME) 모듈에 전달하여 모달리티 간 일치를 찾아내며 정확하지 않은 상관 관계를 수정합니다. 또한, Cooperative Token Elimination (CTE) 전략을 도입하여 추론 효율성과 정확성을 높였습니다. 이는 참조 템플릿과 검색 프레임 간의 상관 관계 간의 일치를 찾아내어 잘못된 상관 관계를 제거합니다.

- **Performance Highlights**: CAFormer은 다섯 개의 공개 RGBT 트래킹 벤치마크에서 실험을 통해 최첨단 성능을 입증했습니다. 또한, CAFormer은 초당 83.6 FPS의 놀라운 추적 속도를 자랑하며, 세 가지 주요 데이터셋에서 최첨단 성능을 확보했습니다. 이는 셀프 어텐션과 크로스 어텐션 메커니즘을 결합하여 추적 속도와 정확성을 동시에 향상시킬 수 있음을 보여줍니다.



### More Than Positive and Negative: Communicating Fine Granularity in Medical Diagnosis (https://arxiv.org/abs/2408.02214)
- **What's New**: 이번 연구에서는 기존의 양자 분류 모델의 한계점을 보완하여 세밀한 정보를 전달할 수 있는 AI 모델을 개발했습니다. 특히, 기존의 단순한 양자 분류 설정 대신, 의료 지식을 기반으로 양성(positive) 사례를 '전형적 양성(typical positive)'과 '비전형적 양성(atypical positive)'으로 나누는 새로운 벤치마크를 제안했습니다.

- **Technical Details**: 새로운 벤치마크는 전형적 양성과 비전형적 양성을 구분하는 AUC$^	ext{FG}$라는 새로운 메트릭을 사용하여 평가합니다. 이를 통해 AI 진단 시스템이 의료 이미지로부터 더 세밀한 정보를 학습할 수 있도록 유도합니다. 또한, 거친 레이블(coarse labels)만을 사용하여 학습하는 간단한 리스크 조절 방법인 'Partially Huberised Cross-Entropy (PCE)' 손실 함수를 제안했습니다.

- **Performance Highlights**: 대규모 CXR 데이터셋인 MIMIC-CXR-JPG에서 제안된 방법을 실험한 결과, 단순함에도 불구하고 우수한 성능을 보여주었습니다. 특히, 세밀한 그라누러리티를 학습하는 능력에서 탁월한 성능을 발휘하여 강력한 기준선을 제공합니다.



### ExoViP: Step-by-step Verification and Exploration with Exoskeleton Modules for Compositional Visual Reasoning (https://arxiv.org/abs/2408.02210)
Comments:
          To Appear at COLM 2024

- **What's New**: ExoViP이라는 새로운 'plug-and-play' 방식을 소개합니다. 이 방법은 시각적 프로그래밍에서 계획 단계 및 실행 오류를 교정하기 위해 '외골격' 검증 모듈을 사용합니다.

- **Technical Details**: ExoViP는 세 가지 서브-검증자(이미지-텍스트 매칭 검증자, 이미지 캡셔닝 검증자, 시각적 질문 응답(VQA) 검증자)로 구성된 검증 모듈을 이용하여 각 추론 단계 후 예측을 검증합니다. 또한, 트리 기반 검색 알고리즘과 LLM의 자가 수정 점수를 사용하여 계획 추적을 교정합니다.

- **Performance Highlights**: ExoViP는 VisProg와 ViperGPT 등 두 가지 최근 시각적 프로그래밍 방식에서 일관된 성능 향상을 보였습니다. 이 방법은 이미지 질문 응답, 표현 이해, 시각적 추론, 추론적 이미지 편집, 시공간적 추론 등의 여섯 가지 구성적 시각 추론 작업에서 성능 향상을 입증했습니다.



### Source-Free Domain-Invariant Performance Prediction (https://arxiv.org/abs/2408.02209)
Comments:
          Accepted in ECCV 2024

- **What's New**: 이번 연구에서는 소스 데이터(source data)가 없는 상황에서도 모델의 성능을 정확하게 예측하기 위한 새로운 접근법을 제안합니다. 기존의 대부분의 성능 예측 방법은 소스 데이터에 의존하지만, 우리는 불확실성 기반(uncertainty-based)의 소스 프리(source-free) 방식을 사용하여 보정된 예측의 정확성을 평가하는 방법을 소개합니다.

- **Technical Details**: 본 연구는 소스 데이터 없이도 성능을 예측할 수 있는 방법으로, 제너러티브 모델(generative model)을 사용하여 보정을 수행하는 '온도 스케일링(temperature scaling)'과의 연결성을 확립합니다. 이후, 그라디언트 기반(gradient-based) 전략을 통해 보정된 예측의 정확성을 평가합니다. 또한, 벤치마크 객체 인식 데이터셋에서 실험을 수행하여 기존의 소스 기반 및 소스 프리 방법들을 능가하는 성능을 확인하였습니다.

- **Performance Highlights**: 본 연구의 방법은 소스 프리 상황에서 기존의 최첨단 방법들을 현저하게 능가하는 성과를 보였습니다. 또한, 일부 데이터셋에서는 소스 기반 방법들을 모두 사용할 수 있는 경우에도 더 우수한 성능을 나타냈습니다.



### Unsupervised Domain Adaption Harnessing Vision-Language Pre-training (https://arxiv.org/abs/2408.02192)
- **What's New**: 이번 논문은 Vision-Language Pre-training (VLP) 모델의 잠재력을 활용하여 Unsupervised Domain Adaptation (UDA)에서 마주하는 두 가지 중요한 과제를 해결하는 방법을 제안합니다. 첫째, VLP 모델을 교사 모델로 활용하여 타겟 도메인에서 학습 과정을 안내하는 Cross-Modal Knowledge Distillation (CMKD) 기법을 도입합니다. 둘째, VLP 모델의 광범위한 사전 학습 이점을 활용하여 Residual Sparse Training (RST) 기법을 제안하여 최소한의 파라미터 조정으로도 높은 성능을 유지하면서 저장 공간 문제를 해결합니다.

- **Technical Details**: CMKD 기법은 텍스트 인코더의 일반 지식을 사전 지식으로 활용하여 타겟 도메인 데이터에 대한 셀프 트레이닝을 도와주는 방식을 채택합니다. 이 기법은 간단한 로스 함수로 쉽게 구현 가능합니다. RST 기법은 네트워크 아키텍처를 변경하지 않고, 매우 희소한 가중치를 통해 모델 배포를 가능하게 합니다. 두 기법 모두 CNNs와 Transformers 아키텍처에 적용 가능합니다.

- **Performance Highlights**: 제안된 방법은 다수의 벤치마크에서 기존 기술들을 능가하는 뛰어난 성능을 보였습니다. 또한, 계산 및 메모리 요구 사항이 효율적이고 실용적임이 입증되었습니다. 코드는 본 논문과 함께 제공될 예정입니다.



### Dense Feature Interaction Network for Image Inpainting Localization (https://arxiv.org/abs/2408.02191)
- **What's New**: 이 논문에서는 이미지 인페인팅(Image Inpainting) 탐지를 위한 새로운 방법인 DeFI-Net(Dense Feature Interaction Network)을 제안합니다. DeFI-Net은 다단계 표현을 캡처하고 증폭하는 피라미드 아키텍처를 사용하여, 각 단계에서 다양한 특징 수준 간의 상호작용을 통해 인페인팅을 더욱 정확하게 탐지할 수 있습니다. 이 네트워크는 낮은 수준의 특징(에지와 형태 정보)을 조정하여 변조된 영역의 경계를 정밀하게 지역화하는 동시에 높은 수준의 의미적 특징을 통합합니다.

- **Technical Details**: DeFI-Net은 다중 레벨 특징을 통합하기 위해 다단계 피라미드 아키텍처를 사용하며, 낮은 수준의 특징을 통해 경계 세부 사항을 정밀하게 탐지합니다. 이를 위해, 보충적 특성 지도 접근법(complementary supervision approach)을 사용하여 에지 감독(edge supervision)과 역 주의를 낮은 레벨 특징에 추가하여 인페인팅 지역의 정확도를 높였습니다. 또한, 공간 가중 학습 모듈(spatial weight learning module)을 설계하여 서로 다른 레벨의 특징에 적응적으로 가중치를 할당합니다.

- **Performance Highlights**: 다섯 가지 이미지 인페인팅 데이터셋을 사용한 평가에서 DeFI-Net은 다양한 모델에서 인페인팅 탐지에 대한 최첨단 성능을 달성했습니다. 이 방법은 기존의 알고리즘을 능가하여 전통적 및 딥러닝 기반 인페인팅을 성공적으로 식별하는 데 있어 강력한 일반화 능력을 보여주었습니다.



### AssemAI: Interpretable Image-Based Anomaly Detection for Manufacturing Pipelines (https://arxiv.org/abs/2408.02181)
Comments:
          8 Pages, 6 Figures, 4 Tables

- **What's New**: 이 논문은 스마트 제조 파이프라인을 위한 해석 가능한 이미지 기반 이상 탐지 시스템인 'AssemAI'를 소개합니다. AssemAI는 주로 맞춤형 이미지 데이터셋과 제조 조립 환경에서의 이상 탐지를 위해 특별히 설계된 YOLO-FF라고 명명된 맞춤형 객체 탐지 모델을 개발하는 데 초점을 맞추고 있습니다.

- **Technical Details**: AssemAI는 로켓 조립 파이프라인에서 얻은 산업용 이미지 데이터셋을 사용하여 데이터 준비, 모델 개발 및 해석에 도메인 지식을 활용합니다. 주요 구성 요소로는 YOLO-FF와 같은 맞춤형 객체 탐지 모델, EfficientNet을 기반으로 하는 새로운 이상 탐지 모델, 그리고 사용자 친화적인 해석을 위한 온톨로지 기반 방법과 SCORE-CAM이 포함됩니다. 데이터 준비 단계에서 이미지를 필터링하고 관련 특성에 집중하도록 크롭합니다. 모델링에는 CNN, Custom-ViT, Pretrained-ViT, 및 EfficientNet와 같은 여러 아키텍처를 사용합니다. 해석 기술로 SCORE-CAM과 프로세스 온톨로지를 통합하여 사용자 수준의 설명을 제공합니다.

- **Performance Highlights**: 기본 CNN 및 Visual Transformer(ViT) 모델을 포함한 여러 기준선 모델과의 비교에서 AssemAI의 데이터 준비 및 사전 학습된 CNN 통합이 얼마나 효과적인지 실증했습니다. 모델은 실제 환경에서도 실시간으로 배포되었습니다. 또한, 기준선 모델에 대한 반사 연구(ablation study)를 통해 제안된 시스템의 종합적인 평가를 제공합니다.



### Rethinking Affect Analysis: A Protocol for Ensuring Fairness and Consistency (https://arxiv.org/abs/2408.02164)
Comments:
          arXiv admin note: text overlap with arXiv:2405.06841

- **What's New**: 이번 연구에서는 감정(analyze) 분석 방법의 평가에 존재하는 불공정성과 편향을 해결하기 위해 단일화된 프로토콜을 제안합니다. 이를 위해 데이터베이스의 인구 통계적(인종, 성별, 나이 등) 특성들을 고려하여 공정하고 비교 가능한 평가 방법론을 확립하였습니다. 새로운 프로토콜을 활용해 기존 방법들을 재평가하고, 향후 연구를 장려하기 위해 새로운 리더보드를 소개합니다.

- **Technical Details**: 이 연구는 감정 분석에 관련된 다양한 데이터베이스(AffectNet, RAF-DB, DISFA, EmotioNet, GFT, RAF-AU)를 인종, 성별, 나이와 같은 인구 통계적 특성에 따라 주석 처리했습니다. 기존의 데이터베이스 분할 방법 중 불공정성을 유발할 수 있는 방식을 지적하고, 새로운 공통 프로토콜을 정의하여 이 문제점을 해결하고자 했습니다. 새로운 프로토콜에서는 인구 통계적 특성을 고려하여 훈련, 검증, 테스트 세트를 공정하게 분할하였습니다.

- **Performance Highlights**: 새로운 프로토콜을 사용하여 최신(state-of-the-art) 감정 분석 방법들을 재평가한 결과, 기존 평가 방식과는 다른 성능 순위가 나타났습니다. 이는 기존의 불공정한 데이터 분할 방식의 한계를 극복하면서, 공정한 비교를 가능하게 만들었습니다. 향후 연구자들이 공정한 비교를 통해 감정 인식 연구를 발전시킬 수 있도록 리더보드 및 주석 처리가 완료된 데이터, 코드와 사전 학습된 모델들을 Github을 통해 공개했습니다.



### PanoFree: Tuning-Free Holistic Multi-view Image Generation with Cross-view Self-Guidanc (https://arxiv.org/abs/2408.02157)
Comments:
          Accepted by ECCV 2024

- **What's New**: PanoFree는 다중 뷰 이미지(multi-view image) 생성에서 세밀한 튜닝과정 없이 개선된 일관성(consistency)과 오류 누적 문제를 해결하는 새로운 방법을 제시합니다. 특히 파노라마 이미지(panorama image) 생성에서 주목받고 있습니다.

- **Technical Details**: PanoFree는 반복적 왜곡(warping)과 inpainting 기술을 사용하여 순차적으로 다중 뷰 이미지를 생성합니다. 이 과정에서 크로스 뷰 인식(cross-view awareness) 및 위험 영역 추정 및 제거(risky area estimation and erasing)를 통해 오류 누적 문제를 개선합니다. 또한, 루프 클로저(loop closure)를 위한 대칭적인 양방향 유도 생성(symmetric bidirectional guided generation)과 장면 구조 보전을 위한 의미적 및 밀도 제어(semantic and density control)를 포함합니다.

- **Performance Highlights**: PanoFree는 Planar, 360°, Full Spherical Panoramas 실험에서 기존 방식에 비해 최대 5배의 시간 효율 및 3배의 GPU 메모리 효율을 보여주며, 사용자 연구 결과 혁신적인 이미지 다양성을 보여주었습니다. PanoFree는 추가적인 사전 학습 모델(pre-trained model)의 사용이나 비용이 많이 드는 튜닝 과정을 대체할 수 있는 현실적인 대안을 제공합니다.



### Video-based Pedestrian and Vehicle Traffic Analysis During Football Games (https://arxiv.org/abs/2408.02146)
- **What's New**: 본 논문은 비디오 분석 기술을 활용하여 보행자 및 차량 교통 행동을 연구하며, 특히 대학 풋볼 게임데이 동안의 교통 패턴을 분석합니다. 이를 통해 게임데이에 보행자와 차량의 활동 패턴이 어떻게 변하는지, 그리고 이러한 이벤트가 교차로에서의 교통량과 안전에 어떤 영향을 미치는지에 대한 중요한 통찰을 제공합니다.

- **Technical Details**: 논문은 YOLOv4와 같은 심층 학습 알고리즘을 사용하여 교차로에서 차량과 보행자의 실시간 동작을 분석합니다. 비디오 분석을 통해 교차로에서 발생하는 잠재적인 위험 상황을 평가하고, 시간대별로 충돌 가능성을 측정합니다. 이 연구는 또한 단기 및 장기 교통 정보를 활용하여 보행자와 차량 간의 상호 작용을 분석하고, 'Barnes Dance' 방식 도입을 제안합니다.

- **Performance Highlights**: UF의 게임데이 동안 보행자 수가 급증하며, 이는 게임 결과와도 양의 상관관계가 있습니다. 게임 시작 몇 시간 전부터 보행자와 차량 간의 충돌 가능성이 높아지며, 법 집행 기관의 존재가 보행자 준수 및 안전성을 높이는 데 기여할 수 있음을 시사합니다. 한편, 차량 간의 충돌 가능성은 일반적으로 게임데이에 증가하지 않으며, 이는 운전자들의 주의력 상승으로 이어질 수 있습니다.



### VidModEx: Interpretable and Efficient Black Box Model Extraction for High-Dimensional Spaces (https://arxiv.org/abs/2408.02140)
- **What's New**: 이번 연구에서는 SHAP(SHapley Additive ExPlanations)을 활용하여 합성 데이터 생성을 향상시키는 새로운 접근 방식을 제안합니다. SHAP는 입력 특징의 개별 기여도를 설명하는데, 이를 통해 에너지 기반 GAN(GAN)을 최적화하여 원하는 출력을 달성할 수 있습니다. 이 방법은 이미지 분류 모델 정확도를 16.45% 향상시키고, 영상 분류 모델에서는 평균 26.11%, 최대 33.36%까지 성능을 높였습니다.

- **Technical Details**: 이 연구는 비전(vision) 분류기에 초점을 맞추었으며, 특정 구조적 제한 없이 다른 도메인에 쉽게 적응 가능한 접근 방식을 유지하고 있습니다. SHAP 값을 통해 생성기를 최적화하여 맞춤 목표를 달성하는 차별화된 파이프라인을 소개하고, 각 클래스에 대한 조건부 생성기를 통해 클래스 분포를 향상시켰습니다. 또한, 위 기술은 다양한 시나리오에서 효과적으로 적용될 수 있습니다. 예를 들어 top-k 예측 확률, top-k 예측 레이블, top-1 레이블이 있는 경우 등입니다.

- **Performance Highlights**: 이미지 분류 모델에서 16.45% 정확도 증가를 달성했으며, 영상 분류 모델(UCF11, UCF101, Kinetics 400, Kinetics 600, Something-Something V2)에서도 평균 26.11%, 최대 33.36%까지 성능을 개선했습니다. 이 방법은 높은 차원 공간으로 확장되는 파이프라인을 구현하여 이전 연구들과 비교한 결과에서도 우수한 성과를 보였습니다.



### RICA^2: Rubric-Informed, Calibrated Assessment of Actions (https://arxiv.org/abs/2408.02138)
Comments:
          Accepted at European Conference on Computer Vision (ECCV) 2024

- **What's New**: 이번 연구는 액션 품질 평가(Action Quality Assessment, AQA)에서 행동의 점수 척도(score rubric)를 통합하고 모델 예측의 불확실성을 반영하는 RICA^2이라는 딥 프로바빌리스틱 모델을 제안합니다. 기존 방법들이 점수 척도를 무시하고 예측의 불확실성을 충분히 다루지 못한 문제를 해결하기 위해 개발되었습니다.

- **Technical Details**: RICA^2의 핵심은 점수 척도를 인코딩하는 그래프 구조상에서 정의된 행동 단계의 확률적 임베딩(stochastic embedding)입니다. 이 임베딩은 잠재 공간(latent space)에 확률 밀도를 분포시키며, 이를 통해 모델 불확실성을 나타낼 수 있습니다. 그래프는 평가 기준을 인코딩하며, 이 기반으로 품질 점수를 디코드할 수 있습니다. 추가로 변분 정보 병목 프레임워크(variational information bottleneck framework) 하에 훈련하는 방법과 불확실성을 추정하는 접근법을 소개합니다.

- **Performance Highlights**: RICA^2는 FineDiving, MTL-AQA, JIGSAWS 등 여러 공개된 벤치마크에서 새로운 최고 성능을 달성했습니다. FineDiving에서는 Spearman’s Rank Correlation Coefficient(SRCC)에서 0.94%의 성능 향상과 Kendall Tau에서 0.178의 향상을 기록했습니다. MTL-AQA와 JIGSAWS에서도 각각 SRCC와 Kendall Tau에서 우수한 성능을 발휘했습니다. 또한, RICA^2의 주요 디자인을 평가하기 위한 광범위한 실험도 포함되었습니다.

- **Implications**: RICA^2는 스포츠, 의료, 재활 등 다양한 분야에서 품질 평가의 정확도를 한층 높이고, 예측 불확실성의 칼리브레이션을 개선함으로써 인간 전문가의 평가 필요성을 줄이고 비용 효율성을 높이는 데 기여할 것으로 기대됩니다.



### A First Look at Chebyshev-Sobolev Series for Digital Ink (https://arxiv.org/abs/2408.02135)
Comments:
          Accepted at MathUI 2024

- **What's New**: 이번 연구에서는 디지털 잉크(digital ink)를 파라메트릭 평면 곡선으로 보고 이를 체비셰프-소볼레프(Chebyshev-Sobolev) 시리즈로 표현하는 방법을 탐구합니다. 이전까지는 주로 레전드르-소볼레프(Legendre-Sobolev) 기저를 사용한 연구가 있었으나, 이번 논문에서는 체비셰프-소볼레프 기저가 특정 목적에서 더 우수할 수 있음을 초기 데이터를 통해 시사합니다.

- **Technical Details**: 디지털 잉크 데이터를 다루는 기존 방식은 주로 픽셀 기반이거나 점-시퀀스(point-sequence) 접근법이었으며, 이는 해상도나 시간에 의존하는 문제를 가집니다. 이번 연구에서는 레전드르-소볼레프 시리즈에서 한 단계 더 나아가, 체비셰프-소볼레프 시리즈를 통해 디지털 잉크를 표현합니다. 체비셰프 다항식은 안정성과 함수 근사화의 정확성에서 장점을 가지며, 소볼레프 공간은 도함수를 포함하여 형태 매칭을 개선합니다.

- **Performance Highlights**: 초기 실험 결과에 따르면, 체비셰프-소볼레프 방식이 레전드르-소볼레프 방식보다 특정 목적에서 더 나은 성능을 보일 수 있습니다. 특히, 빠른 계산과 효율적인 곡선 간 변동 측정에서 그 우수성이 나타납니다.



### FovEx: Human-inspired Explanations for Vision Transformers and Convolutional Neural Networks (https://arxiv.org/abs/2408.02123)
Comments:
          Under submission

- **What's New**: Foveation 기반 설명법(FovEx)을 소개합니다. 이 방법은 인간의 시각 시스템에서 영감을 받아 생물학적으로 변화된 입력 변형과 기울기 기반의 시각적 탐색을 결합하여 모델의 성능을 최대로 이끌어낼 중요한 위치를 효율적으로 파악하는 기법입니다. 이를 통해 관심 영역을 선택하고 결합하여 설명 맵(attribution map)을 생성합니다.

- **Technical Details**: FovEx는 두 가지 주요 단계를 갖습니다. 첫째, 입력 샘플은 인간의 중심 시야(Fovea)를 흉내 내는 변형을 겪습니다. 둘째, 기울기 정보는 입력 이미지의 중요한 부분을 탐색하는데 사용되어 해당 영역을 결정합니다. 이렇게 획득된 중요한 영역을 조합하여 최종 설명 맵을 만듭니다. 이 방법은 후처리(post-hoc) 방식으로 동작하여 기본 모델의 구조를 변경하지 않고도 적용할 수 있습니다.

- **Performance Highlights**: 광범위한 평가를 통해 FovEx는 변환기(Transformer) 모델에서 5개 지표 중 4개에서, 컨볼루셔널(Convolutional) 모델에서 5개 중 3개 지표에서 최첨단 성능을 달성하며 다양한 아키텍처에서의 유연성을 입증했습니다. 또한 FovEx가 생성한 설명 맵은 인간의 시선 패턴과 더 잘 일치하며, 이는 인간과 기계 간 해석의 격차를 줄이는 데 기여합니다 (RISE 대비 NSS에서 +14%, GradCAM 대비 +203% 상승).



### AvatarPose: Avatar-guided 3D Pose Estimation of Close Human Interaction from Sparse Multi-view Videos (https://arxiv.org/abs/2408.02110)
Comments:
          Project Page: this https URL

- **What's New**: 기존의 멀티뷰 방식이 여러 사람이 밀접하게 상호작용하는 상황에서 3D 포즈와 형태를 추정하는데 어려움을 겪는 문제를 해결하기 위해, 각 개인의 맞춤형 암묵적 신경 아바타(implicit neural avatar)를 사전 지식(prior)으로 활용하여 더 튼튼하고 정밀한 포즈 추정을 수행하는 새 방법을 제안했습니다. 이 아바타는 다중뷰 비디오에서 계층적 볼륨 렌더링(layered volume rendering)을 통해 효율적으로 재구성되며, 컬러와 실루엣 렌더링 손실을 기반으로 3D 포즈를 최적화할 수 있게 합니다.

- **Technical Details**: 기술적으로, 각 개인을 효율적으로 재구성하기 위해 Instant NGP의 효율적인 신경 방사장(Neural Radiance Field) 변형을 사용하여 정규화된 공간에서 모델링합니다. 여러 사람의 아바타 모델을 학습하고 렌더링하는 과정에서는 계층적 볼륨 렌더링을 적응시켜, 간단한 렌더링 손실을 통해 아바타 모델을 공동으로 최적화합니다. 학습된 아바타는 포즈 최적화를 위해 컬러와 실루엣 렌더링 손실을 최소화하는 목표 함수로 사용됩니다. 또한, 최적화 과정을 통해 상호 관통을 방지하기 위해 충돌 손실(collision loss)을 도입합니다.

- **Performance Highlights**: 여러 공개 데이터셋에서의 실험 결과, 제안한 방법이 밀접하게 상호작용하는 상황에서도 기존 최첨단 방법보다는 수치적으로(정량적으로)와 시각적으로(정성적으로) 모두 우수한 성능을 보임을 확인했습니다.



### View-consistent Object Removal in Radiance Fields (https://arxiv.org/abs/2408.02100)
Comments:
          Accepted to ACM Multimedia (MM) 2024. Project website is accessible at this https URL

- **What's New**: 새로운 Radiance Fields (RFs) 편집 파이프라인이 도입되었습니다. 이 파이프라인은 단일 기준 이미지만 인페인팅(inpainting)하여 여러 관점에서의 일관성을 크게 향상시킵니다. 이 이미지가 깊이 기반 접근법을 사용해 여러 뷰에서 투영(projection)됨으로써 일관성이 향상됩니다. 이 방법은 현실적인 조명 및 시점 변화를 조정하여 기존 방법의 한계를 극복합니다.

- **Technical Details**: 이번 연구에서는 Neural Radiance Fields (NeRF)와 3D Gaussian Splatting (3D-GS)와 같은 기술이 사용됩니다. 파이프라인은 단일 기준 이미지의 깊이를 기반으로 다른 뷰로 투영하는 기법으로 구성됩니다. 또한, 투영된 뷰의 외관을 조정하기 위해 인페인팅 된 이미지를 여러 방향 변형으로 생성하여 다양한 광학 조건을 수용합니다. 이를 통해 구조적 및 시점 종속적 일관성을 보장하며, 빠르고 견고한 멀티뷰 객체 분할 접근방식도 제시합니다.

- **Performance Highlights**: 이 새로운 방법은 단일 참조 이미지를 인페인팅하므로 여러 프레임에 독립적으로 인페인팅할 때 발생하는 일관성 문제를 크게 줄입니다. 실험 결과, 다양한 RF 모델, 특히 NeRF와 3D-GS에서 콘텐츠 일관성과 시각적 품질을 현저히 개선하는 것으로 나타났습니다. 연구진은 깊이 기반 멀티뷰 분할법을 통해 일관된 마스크 생성을 효율적으로 구현하는 데 성공했습니다.



### Past Movements-Guided Motion Representation Learning for Human Motion Prediction (https://arxiv.org/abs/2408.02091)
Comments:
          13 pages, 4 figures

- **What's New**: 본 논문에서는 3D 골격 기반의 인간 동작 예측에 있어 효과적인 동작 표현을 강화하기 위해 새로운 자기 지도 학습 프레임워크를 제안합니다. 제안된 방식은 과거 동작의 자기 재구성과 이를 바탕으로 한 미래 동작의 유도 재구성을 포함하는 2단계의 네트워크 사전 학습 단계를 통해 동작 표현력을 극대화합니다.

- **Technical Details**: 첫 번째 단계에서는 과거 시퀀스를 기반으로 미래 시퀀스를 재구성하는 사전 학습을 진행합니다. 이를 위해 움직임이 큰 관절에 집중하기 위한 속도 기반의 마스크 전략을 설계했습니다. 두 번째 단계에서는 사전 학습된 네트워크를 특정 작업에 대해 미세 조정(finetune)합니다. 스페이셜(Spatiotemporal) 관계를 효과적으로 표현하는 Transformer 기반의 아키텍처를 사용하며, 이는 Past Motion Encoder (PME)와 Future Motion Predictor (FMP)로 구성되어 있습니다.

- **Performance Highlights**: 제안된 방법은 Human3.6M, 3DPW, AMASS 데이터셋에서 평균 예측 오류를 8.8% 감소시키며, 최첨단(state-of-the-art) 방법들보다 성능이 우수합니다.



### KAN-RCBEVDepth: A multi-modal fusion algorithm in object detection for autonomous driving (https://arxiv.org/abs/2408.02088)
- **What's New**: 3D 객체 인식의 정확성을 크게 향상시키는 새로운 알고리즘 RCBEV-KAN을 소개합니다. 이 알고리즘은 카메라, LiDAR, 밀리미터파 레이더의 다중 모드 센서 데이터를 융합하여 Bird's Eye View(BEV) 기반 접근 방식을 사용하며, 트랜스포머 아키텍처를 통해 다양한 데이터 소스를 통합하고 공간 관계 처리를 개선합니다. 실험 결과, RCBEV-KAN 모델은 대부분의 인식 카테고리에서 우수한 성능을 보였으며, Mean Distance AP에서 23%, ND Score에서 17% 향상된 결과를 나타냈습니다.

- **Technical Details**: RCBEV-KAN algorithm은 다중 모드 센서 융합을 사용하여 카메라, LiDAR, 밀리미터파 레이더 데이터를 하나의 BEV 특징 공간으로 통합합니다. 이 방법은 트랜스포머 아키텍처를 사용해 다양한 데이터 소스를 매끄럽게 통합하고, Kolmogorov-Arnold Network (KAN)를 이용해 시계열 데이터의 특징 추출을 최적화합니다. 또한 DepthNet 모듈은 LiDAR 포인트 클라우드 데이터의 직접적인 감독을 받고, 밀리미터파 레이더 데이터를 추가해 깊이 학습을 향상시킵니다.

- **Performance Highlights**: RCBEV-KAN 모델은 기존의 모델들보다 뛰어난 성능을 보입니다. Mean Distance AP는 0.316에서 0.389로 23% 향상되었으며, ND Score도 0.415에서 0.484로 17% 개선되었습니다. 또한, 평가 시간은 8% 빨라져 71.28초가 소요되었습니다. 이러한 성능 향상은 복잡하고 역동적인 자율 주행 환경에서 RCBEV-KAN 모델이 더욱 신뢰성과 효율성을 갖추고 있다는 것을 입증합니다.



### Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models (https://arxiv.org/abs/2408.02085)
Comments:
          review, survey, 28 pages, 2 figures, 4 tables

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 instruction tuning을 목적으로 활용 가능한 데이터 평가 및 선택 방법에 관한 종합적인 리뷰를 제공합니다. 기존 연구들이 데이터 평가 메트릭스 및 선택 메커니즘에 대해 구체적으로 다루지 않은 부분을 보완하고, 데이터 평가가 instruction tuning에 어떻게 통합될 수 있는지를 설명합니다.

- **Technical Details**: 데이터 평가 및 선택 방법을 크게 세 가지 관점에서 체계적으로 분류합니다: quality-based, diversity-based, importance-based 방법들입니다. 각 카테고리 내 대표적인 방법들을 상세히 기술하고, 최신 방법들의 공식 보고 결과를 바탕으로 비교를 진행합니다. 또한, 다양한 데이터셋과의 연관성을 분석하고, 커다란 데이터셋을 평가하고 일부 선택할 때 필요한 방법론들을 설명합니다.

- **Performance Highlights**: 이 논문은 다양한 데이터 평가 방법들이 실제 성능에 미치는 영향을 심층적으로 논의하며, 그 한계점들을 명확히 제시합니다. 특히, quality, diversity, importance 측면에서 데이터를 선별하여 학습 비용을 줄이면서 성능을 높일 수 있는 방안을 제시합니다. 이를 통해, 효과적이고 효율적인 instruction tuning을 위한 데이터 평가 및 선택의 중요성을 강조합니다.



### Improving Neural Surface Reconstruction with Feature Priors from Multi-View Imag (https://arxiv.org/abs/2408.02079)
Comments:
          ECCV2024

- **What's New**: 최근 Neural Surface Reconstruction (NSR) 분야의 다양한 발전이 다중 뷰 재구성에서 큰 진전을 이루었으나, 여전히 실제 데이터의 복잡성을 처리하기에는 부족합니다. 본 논문에서는 다양한 사전 학습된 시각 과제의 feature priors (피처 프라이어)를 조사하여 NSR의 한계를 극복하고자 합니다. 특히, feature-level consistent loss (피처 레벨 일관성 손실)을 활용하여 NSR의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 본 연구는 7개의 시각 과제에서 13개의 방법을 통해 다중 뷰 feature priors를 포괄적으로 탐구합니다. 대표적으로 MVSFormer와 QuadTree의 사전 학습된 표현을 사용하여 MVS-NeuS와 Match-NeuS의 변형을 생성하는 방식입니다. 다양한 feature 해상도의 영향 또한 평가하며, pixel-wise (픽셀 단위)와 patch-wise (패치 단위)의 일관성 손실을 분석합니다. 특히, image matching (이미지 매칭)과 multi-view stereo (다중 뷰 스테레오)에서 얻은 feature priors가 다른 과제보다 더 우수한 성능을 나타냅니다.

- **Performance Highlights**: 결과적으로, patch-wise photometric consistency (패치 단위 광학 일관성)을 feature 레벨로 확장하면 pixel-wise 접근법보다 뛰어난 성능을 발휘함을 확인하였습니다. 이를 통해 높은 해상도의 feature가 NSR 성능 향상에 중요한 요소임을 밝혔습니다. DTU 및 EPFL 데이터셋에서의 실험 결과, 제안된 방법이 최신 성능을 보여주었으며, 특히 MVS와 이미지 매칭 작업에서 사전 학습된 모델이 다른 접근 방식보다 월등한 성능을 발휘하였습니다.



### LDFaceNet: Latent Diffusion-based Network for High-Fidelity Deepfake Generation (https://arxiv.org/abs/2408.02078)
- **What's New**: 얼마 전만 해도 GAN(Generative Adversarial Networks)이 주요한 생성 모델이었지만, 최근 비평형 열역학에 영감을 받은 확산 확률 모델(Diffusion Probabilistic Models, DMs)이 주목받고 있습니다. 이번 논문에서는 새로운 얼굴 바꾸기 모듈 LDFaceNet(Latent Diffusion based Face Swapping Network)을 제안하며, 이는 사전 학습된 잠재 확산 모델(Pre-trained Latent Diffusion Model)을 활용하여 얼굴을 교환하는 것에 중점을 둡니다.

- **Technical Details**: LDFaceNet은 얼굴 분할과 얼굴 인식 모듈을 사용하여 조건화된 디노이징(Conditioned Denoising)을 수행합니다. 고유의 손실 함수(Loss Function)를 사용하여 확산 과정을 방향적으로 유도하며, 재학습 없이 추가적인 얼굴 지침(Facial Guidance)을 통합할 수 있습니다. 이 방법은 CelebA 데이터셋으로 학습된 LDM을 활용하여 고비용의 재학습을 필요로 하지 않습니다. 중간 타임스텝에서 생성된 이미지 임베딩을 통해 모델이 제약되고 안내됩니다.

- **Performance Highlights**: LDFaceNet은 기존의 얼굴 바꾸기 방법들보다 질적, 양적으로 뛰어난 결과를 보여줍니다. 얼굴의 경계 부분에서 매끄러운 전환을 보장하는 잠재 레벨 블렌딩(Latent-Level Blending)을 구현하여, 현실감 있고 일관된 이미지를 생성합니다. 또한, 가려진 얼굴, 정렬되지 않은 얼굴, 비정면 얼굴 등의 다양한 어려운 시나리오에서도 강력한 성능을 발휘합니다.



### FDiff-Fusion:Denoising diffusion fusion network based on fuzzy learning for 3D medical image segmentation (https://arxiv.org/abs/2408.02075)
Comments:
          This paper has been accepted by Information Fusion. Permission from Elsevier must be obtained for all other uses, in any current or future media. The final version is available at [doi:https://doi.org/10.1016/J.INFFUS.2024.102540]

- **What's New**: 최근 몇 년간, 노이즈 제거 확산 모델(denoising diffusion model)은 이미지 분할 모델링에서 놀라운 성공을 거두었습니다. 이러한 성과를 바탕으로, 이 모델은 점차 의료 이미지 분할 작업에도 적용되며 새로운 시각과 방법을 제시하고 있습니다. 그러나 기존 방법들은 분할 경계의 불확실성과 영역의 모호성을 간과하여 분할 결과의 불안정성과 부정확성을 초래합니다. 이러한 문제를 해결하기 위해, 본 논문에서는 퍼지 학습(fuzzy learning)을 기반으로 한 3D 의료 이미지 분할을 위한 노이즈 제거 확산 융합 네트워크(FDiff-Fusion)를 제안합니다.

- **Technical Details**: FDiff-Fusion은 고전적인 U-Net 네트워크와 노이즈 제거 확산 모델을 통합하여, 입력된 의료 이미지에서 풍부한 의미 정보를 효과적으로 추출합니다. 이를 통해 의료 이미지 분할을 위한 우수한 픽셀 수준 표현을 제공합니다. 퍼지 학습을 활용해 경계의 불확실성과 영역의 모호성을 보다 정확하게 처리합니다.

- **Performance Highlights**: FDiff-Fusion의 효과를 검증하기 위해 BRATS 2020 뇌 종양 데이터셋과 BTCV 복부 다중 장기 데이터셋에서 기존의 첨단 분할 네트워크와 비교 실험을 실시했습니다. 그 결과, FDiff-Fusion은 두 데이터셋에서 Dice 점수와 HD95 거리를 크게 향상시켰으며, 이는 의료 이미지 분할 작업에서의 우월성을 입증합니다.



### Case-based reasoning approach for diagnostic screening of children with developmental delays (https://arxiv.org/abs/2408.02073)
- **What's New**: 본 연구는 CNN-트랜스포머 모델과 사례 기반 추론(Case-Based Reasoning, CBR)을 결합한 하이브리드 모델을 채택하여 발달 지연 아동의 선별 효율성을 향상시키기 위한 시스템을 개발했습니다. 이 모델은 아동의 발달 지연을 조기에 식별하고 적절한 개입을 제공함으로써 의학 자원의 낭비와 사회적 비용을 크게 줄일 수 있습니다.

- **Technical Details**: CNN-트랜스포머 모델은 이미지 특징 추출과 인식에 우수한 성능을 보이며, 골 연령 이미지를 통해 특징을 효과적으로 식별할 수 있습니다. 사례 기반 추론(CBR)은 유사한 과거 사례를 기반으로 문제를 해결하는 기법으로, 기존에 저장된 사례를 바탕으로 새로운 사례를 판단하고 비교하는 데 활용됩니다. 이러한 모델의 결합을 통해 잠재적이고 변동 가능한 특성을 가진 지원 시스템에 적합한 선별 시스템을 구축했습니다.

- **Performance Highlights**: 국제 연구에 따르면 발달 지연 아동의 조기 개입 최적 시기는 6세 이하이며, 황금 치료 기간은 3.5세 이전입니다. 조기 개입을 받은 발달 지연 아동은 증상이 현저히 개선되며, 일부는 완전히 회복할 수 있습니다. 이번 연구의 시스템은 화이베이, 안후이성의 신생아수를 바탕으로 연간 약 7,500건의 의심 사례를 더욱 효율적으로 선별할 수 있도록 설계되었습니다.



### ParkingE2E: Camera-based End-to-end Parking Network, from Images to Planning (https://arxiv.org/abs/2408.02061)
- **What's New**: 본 논문에서는 자율 주차(autonomous parking)의 효율적인 수행을 위해 인간 주행 궤적을 모방(imitation learning)하는 딥러닝 기반 종단간 계획(end-to-end planning) 방법을 제안합니다. 이 방법은 RGB 이미지에서 경로 계획(path planning)을 수행하여 전통적인 규칙 기반(rule-based) 알고리즘이 복잡한 주차 시나리오에서 겪는 한계를 극복합니다.

- **Technical Details**: 제안된 방법은 타깃 쿼리 인코더(target query encoder)를 사용하여 이미지와 타깃 피처를 융합(fuse)하고, 트랜스포머(transformer) 기반 디코더를 통해 미래의 경로 지점(waypoints)을 자동회귀적으로 예측(predict)합니다. 이는 인간 주행 데이터를 학습하여 종단간 계획을 가능하게 하고, 보다 직관적이고 다재다능한 주차 알고리즘을 구현합니다.

- **Performance Highlights**: 실험 결과, 네 가지의 서로 다른 실제 주차장에서 평균 87.8%의 성공률을 달성했습니다. 또한 실제 차량 실험을 통해 제안된 방법의 실용성과 효과상이 검증되었습니다. 이를 통해 딥러닝 기반 주차 알고리즘의 가능성을 입증했습니다.



### Step Saver: Predicting Minimum Denoising Steps for Diffusion Model Image Generation (https://arxiv.org/abs/2408.02054)
- **What's New**: 이번 논문에서는 주어진 텍스트 프롬프트에 필요한 최소한의 디노이징 단계(minimal denoising steps)를 결정하는 혁신적인 NLP 모델을 소개합니다. 이 모델은 실시간으로 동작하며, 고품질 이미지를 효율적으로 생성하기 위한 최적의 디노이징 단계를 추천합니다. Diffusion 모델과 원활하게 작동하여, superior quality의 이미지를 최단 시간 내에 생성할 수 있습니다.

- **Technical Details**: 논문에서 소개된 모델은 주로 DDIM 스케줄러와 함께 사용되지만, Euler, Euler Ancestral, Heun, DPM2 Karras, UniPC 등 다양한 스케줄러에도 적용될 수 있습니다. Stable Diffusion 모델은 텍스트 입력(프롬프트)과 시드(seed)를 받아 Gaussian noise와 결합하여 초기 latent image representation을 생성합니다. 이후 U-Net 모델이 디노이징 과정을 수행하여 텍스트 임베딩을 고려한 예측 노이즈 상수(predicted noise residual)를 출력하고, 이 값을 바탕으로 최종 이미지를 생성합니다. 이를 통해 최적의 디노이징 단계를 찾아내어 GPU 자원을 절약합니다.

- **Performance Highlights**: StepSaver 모델 평가를 위해 512x512 해상도의 2,322,632개의 이미지를 사용했으며, LAION-Aesthetics v2 6+ subset 데이터셋을 참조 이미지(reference images)로 사용했습니다. 실험 결과, 평균적으로 50단계 이하의 디노이징 단계가 최적임을 확인했습니다. 예를 들어, Joel Robison 스타일의 'The Black Dog' 프롬프트에서는 50단계 이상의 디노이징은 이미지 품질을 개선하지 않았고, 심지어 모델의 성능이 저하되었습니다. SSIM 측정 결과, 50단계에서 유사도가 최고치를 기록했습니다. 또한, FID(Frechet Inception Distance) 점수를 통해 이미지 품질을 평가한 결과, 최적의 디노이징 단계를 통해 생성된 이미지의 품질이 더 우수함을 확인했습니다.



### PanicleNeRF: low-cost, high-precision in-field phenotypingof rice panicles with smartphon (https://arxiv.org/abs/2408.02053)
- **What's New**: 새로운 연구인 PanicleNeRF는 야외에서 스마트폰을 이용한 고정밀, 저비용의 벼 이삭 3D 모델 재구축 방법을 소개합니다. 이 기술은 대형 모델인 Segment Anything Model (SAM)과 소형 모델인 You Only Look Once 버전 8 (YOLOv8)를 결합하여 고정밀 이삭 이미지 분할을 달성합니다. 그런 다음 NeRF 기술을 사용하여 2D 분할 이미지를 통해 3D 재구축을 수행합니다.

- **Technical Details**: 이 연구에서는 SAM과 YOLOv8를 결합하여 벼 이삭의 2D 이미지 분할 작업을 극복했습니다. 다음으로 NeRF(신경 방사장식 재구축) 기술을 이용하여 이미지를 가지고 3D 재구축을 수행했습니다. 결과적인 point clouds는 이삭 특성을 추출하기 위해 처리되었습니다.

- **Performance Highlights**: PanicleNeRF는 2D 이미지 분할 작업에서 평균 F1점수 86.9%, 평균 IoU 79.8%를 달성하였으며, YOLOv8에 비해 거의 두 배의 Boundary Overlap (BO) 성능을 보여주었습니다. point cloud 품질 측면에서는 기존의 SfM-MVS (구조-이동 및 다중-시각 스테레오) 방법보다 뛰어난 성능을 보였으며, 이삭 길이 추출의 rRMSE는 indica 벼의 경우 2.94%, japonica 벼는 1.75%를 달성했습니다. 3D point clouds에서 추정한 이삭 부피는 알곡 수와 강한 상관성을 보였으며 (indica: R2=0.85, japonica: R2=0.82), 알곡 무게와도 상관관계가 높았습니다 (indica: R2=0.80, japonica: R2=0.76).



### EOL: Transductive Few-Shot Open-Set Recognition by Enhancing Outlier Logits (https://arxiv.org/abs/2408.02052)
Comments:
          19 pages

- **What's New**: 이번 연구에서는 Few-Shot Learning (FSL)에서 Open-Set Few-Shot Recognition (OSFSL)이라는 더 실용적 도전 과제를 탐구합니다. 기존의 FSL 모델들은 지원 세트의 동일한 클래스 분포에서 샘플링된 쿼리 인스턴스를 평가하지만, OSFSL은 쿼리 세트에 알려지지 않은 클래스(unknown classes)를 포함합니다. 이는 모델이 알려진 클래스뿐만 아니라 이상치(outliers)도 식별해야 하는 복잡한 문제입니다. 이를 해결하기 위해 InfoMax 원리를 활용한 새로운 추론 기법을 도입하여 'Enhanced Outlier Logit (EOL)' 방법을 제안합니다.

- **Technical Details**: EOL 방법은 클래스 프로토타입(class prototypes) 표현을 모델 보정(calibration)을 통해 정제하고, inlier-outlier 비율을 효과적으로 균형잡습니다. 이 접근법은 쿼리 세트의 가짜 라벨(pseudo-label) 정확도를 향상시키고, 전이적 추론(transductive inference) 과정의 최적화 목표를 개선합니다. OSFSL 작업에 맞춘 새로운 전이적 방법론을 통해, 모델은 수집된 쿼리 세트를 추가 학습 데이터셋으로 활용하여 성능을 향상시킬 수 있습니다. 이는 기존의 OSTIM과의 차이점을 명확히 하고, 불변한 inlier-outlier 비율 가정을 개선합니다.

- **Performance Highlights**: EOL 방법은 다양한 클래스 분류 및 이상치 식별 메트릭과 벤치마크에서 전통적인 방법들보다 월등히 뛰어난 성능을 보였습니다. 성능 향상은 대략 +1.3%에서 +6.3% 범위에 걸쳐 나타났습니다. 특히, inlier-outlier 불균형 상황에서도 일관된 우수한 성능을 보였습니다.



### 3D Single-object Tracking in Point Clouds with High Temporal Variation (https://arxiv.org/abs/2408.02049)
Comments:
          Accepted by ECCV24

- **What's New**: HVTrack는 일시적 변화가 큰 3D 포인트 클라우드 데이터에서 단일 객체 추적(3D Single-Object Tracking, 3D SOT)을 위한 새로운 프레임워크입니다. 기존 방식들이 시간적 변화를 부드럽게 가정하여 실패하던 점을 개선했습니다. HVTrack는 세 가지 주요 모듈을 도입하여 이 문제를 해결합니다: 상대 자세 인식 메모리 모듈(Relative-Pose-Aware Memory module, RPM), 기본 확장 기능 교차 주의 모듈(Base-Expansion Feature Cross-Attention module, BEA), 그리고 문맥적 포인트 안내 자가 주의 모듈(Contextual Point Guided Self-Attention module, CPA)입니다.

- **Technical Details**: HVTrack는 일시적 변화가 큰 포인트 클라우드 형상 변화를 처리하기 위해 RPM 모듈을 사용합니다. RPM은 전경 마스크와 관찰 각도를 메모리에 통합하여 포인트 클라우드의 분포 변화를 학습합니다. BEA 모듈은 유사 객체 혼란(Distractions) 문제를 해결하기 위해 교차 주의에서 혼합 규모 특징을 동기화합니다. CPA 모듈은 확장된 검색 영역으로 인한 배경 소음을 억제합니다. CPA는 포인트 간 중요성에 따라 특징을 문맥적으로 집계하여 덜 중요한 포인트는 적은 문맥 정보를 공유하도록 합니다.

- **Performance Highlights**: KITTI-HV 데이터셋에서 HVTrack는 기존 최첨단 추적기인 CXTracker를 성공률(Success)에서 11.3%, 정밀도(Precision)에서 15.7% 초과합니다. 또한, Waymo와 KITTI 데이터셋에서도 뛰어난 성능을 보여주며, 다양한 시간적 변화를 효과적으로 처리할 수 있는 강력한 성능을 입증했습니다.



### Deep Spectral Methods for Unsupervised Ultrasound Image Interpretation (https://arxiv.org/abs/2408.02043)
Comments:
          Accepted at International Conference on Medical Image Computing and Computer Assisted Intervention, MICCAI 2024

- **What's New**: 이 논문에서는 초음파 이미지를 쉽게 해석할 수 있는 조직 분리를 위해 새로운 비지도 학습(deep learning) 방법을 제안합니다. 이 접근법은 깊이 있는 스펙트럴(graph theory) 기법과 자기지도 학습(Self-supervised learning)으로 얻어진 Transformer 피처를 통합하여 설명 가능한 초음파 이미지 분할을 수행합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 단계로 구성됩니다. 첫 번째 단계는 자기지도 학습 기반의 Transformer에서 추출한 피처를 사용하여 스펙트럴 클러스터링(Spectral Clustering)을 수행해 의미 있는 세그먼트를 생성합니다. 두 번째 단계에서는 초음파에 특화된 메트릭스와 형태 및 위치 우선순위를 도입해 데이터셋 전반에 걸쳐 일관된 의미를 유지합니다.

- **Performance Highlights**: 이 방법은 세 가지 초음파 데이터셋에서 평가됐으며, 레이블이 필요 없는 상태에서 해부학적 구조의 경계를 보존하면서 뛰어난 세그멘테이션 성능을 보였습니다. 기존의 다른 클러스터링 알고리즘과 비교한 결과, 제안된 방법이 우수한 정확도를 나타냈으며, 경계 보존과 라벨 일관성에서 뛰어났습니다.



### Pixel-Level Domain Adaptation: A New Perspective for Enhancing Weakly Supervised Semantic Segmentation (https://arxiv.org/abs/2408.02039)
Comments:
          15 pages, 9 figures

- **What's New**: 이 논문에서는 이미지 레벨 약지도 의미 분할(Weakly Supervised Semantic Segmentation, WSSS) 분야에서 새로운 Pixel-Level Domain Adaptation (PLDA) 방법을 소개합니다. 이 접근법은 기존의 Class Activation Map (CAM) 기법이 가지는 불균형 활성화 문제를 해결하는 것을 목표로 합니다. 특히, PLDA는 개별 이미지 내에서 드러나는 '차별적인' 부분과 '비차별적인' 부분 간의 도메인 불일치를 완화하는 데 중점을 둡니다.

- **Technical Details**: PLDA는 픽셀 수준의 도메인 불변 특징(domain-invariant features)을 학습하기 위해 다중 헤드 도메인 분류기를 도입합니다. 이 분류기는 이미지 내의 차별적 영역(discriminative regions)과 비차별적 영역(non-discriminative regions) 간의 차이를 식별하도록 훈련됩니다. 동시에 특징 추출기(feature extractor)는 도메인 분류기를 혼란시키도록 역경 가능 학습(adversarial training)을 통해 훈련됩니다. 또한 Confident Pseudo-Supervision (CPS) 전략을 도입하여 픽셀별 분별력을 보장하고, 이러한 전략은 도메인 불변 학습(invariant feature learning)을 보완합니다.

- **Performance Highlights**: 제안된 PLDA 방법은 여러 강력한 베이스라인 모델에서 실험적으로 검증되었습니다. 그 결과, 다양한 설정에서 베이스라인 모델을 크게 개선하는 성과를 보였습니다. 이는 제안된 방법의 효과와 일반화를 입증합니다.



### LEGO: Self-Supervised Representation Learning for Scene Text Images (https://arxiv.org/abs/2408.02036)
- **What's New**: 한정된 주석 데이터의 문제를 극복하기 위해, 새로운 자기 지도 학습 방법인 LEGO(Local Explicit and Global Order-aware)가 제안되었습니다. 이 방법은 텍스트 인식에서의 연속적 특성, 의미적 특성, 구조적 특성을 모델링하는 세 가지 새로운 사전 텍스트 작업을 기반으로 합니다.

- **Technical Details**: LEGO는 인간의 단어 학습 과정에서 영감을 받아, 기계 학습 모델이 텍스트 이미지를 효과적으로 인식하고 표현할 수 있도록 설계되었습니다. 이 방법은 SID(Selective Individual Discrimination), MIM(Enhanced Mask Image Modeling), RTR(Random Text Rearrangement)이라는 세 가지 주요 작업으로 구성됩니다. 또, 텍스트 지식 코드북(Text Knowledge Codebook)을 활용하여 텍스트 특성과 의미를 보강합니다.

- **Performance Highlights**: LEGO는 기존의 텍스트 인식 방법과 비교하여 뛰어난 성능을 보여주었습니다. 6개의 벤치마크 테스트에서 최상의 성능을 기록하며, 텍스트 인식 및 텍스트 관련 다른 작업에서도 우수한 결과를 입증했습니다.



### Mini-Monkey: Alleviate the Sawtooth Effect by Multi-Scale Adaptive Cropping (https://arxiv.org/abs/2408.02034)
- **What's New**: 최근 MLLM (Multimodal Large Language Models)이 고해상도 이미지를 처리하는 능력을 향상시키려는 시도가 많이 있었으며, 대부분의 방법들은 이미지를 잘라내는 방식으로 이미지 세부사항을 이해하는 능력을 높이고자 했습니다. 그러나 이 방식은 객체나 연결된 영역을 분할하여 MLLM이 작은 또는 불규칙한 모양의 객체나 텍스트를 인식하는 데 어려움을 줍니다. 이러한 문제를 해결하고자, 우리는 경량 MLLM인 Mini-Monkey를 제안했습니다. 이는 플러그 앤 플레이 방식의 MSAC (Multi-Scale Adaptive Crop Strategy)를 포함하여 여러 스케일의 표현을 적응적으로 생성하고, 비분할 객체를 다양한 스케일에서 선택할 수 있게 합니다.

- **Technical Details**: Mini-Monkey는 기존의 이미지를 직접 자르는 방법과 달리 MSAC 방법을 사용합니다. MSAC는 미리 설정된 그리드 그룹을 계층화하여 다양한 종횡비와 해상도를 고려해 여러 스케일의 이미지를 생성합니다. 이러한 다중 스케일 이미지는 사전 훈련된 비전 인코더를 통해 처리되어 시각적 표현을 생성하고, 이를 시퀀스로 연결하여 LLM 내에서 상호작용하게 됩니다. MSAC는 추가적인 계산 오버헤드를 유발할 수 있어 이를 줄이기 위해 SCM (Scale Compression Mechanism)을 제안합니다. SCM은 훈련이 필요 없고, 파라미터가 없는 모듈로, LLM의 주의 레이어를 잘 활용하여 계산 오버헤드를 줄입니다.

- **Performance Highlights**: Mini-Monkey는 2B-파라미터 MLLM 중에서 최첨단 성능을 기록했습니다. 일반적인 멀티모달 이해 및 문서 이해 작업에서 선도적인 성과를 보여주었으며, OCRBench에서는 802점을 기록하여 8B-파라미터 최고 성능 모델인 InternVL2-8B를 능가했습니다. 또한, 모델과 훈련 전략은 매우 효율적으로, RTX 3090 8장으로도 훈련이 가능합니다.



### Enhancing Human Action Recognition and Violence Detection Through Deep Learning Audiovisual Fusion (https://arxiv.org/abs/2408.02033)
Comments:
          This work has been submitted to the IEEE for possible publication, 10 pages, 8 figures

- **What's New**: 이 논문에서는 공공장소에서 사람의 활동 인식 및 폭력 탐지를 개선하기 위해 오디오와 비디오 두 가지 모달리티를 기반으로 하는 하이브리드 융합 기반 딥 러닝(HFBDL) 접근 방식을 제안합니다. RLVS(Real-life violence situation) 데이터셋을 확장하여 사용했으며, HFBDL을 통해 96.67%의 검증 데이터 정확도를 달성했습니다. 또한, 실제 시나리오에서 모델을 테스트하기 위해 54개의 폭력과 비폭력 상황을 포함한 비디오 데이터셋을 녹화했으며, 이 중 52개를 올바르게 탐지했습니다.

- **Technical Details**: 결과 분석을 위해 late fusion, intermediate fusion, hybrid fusion을 비교했습니다. Python과 TensorFlow 프레임워크를 사용하여 모델을 구축하고 훈련했습니다. 데이터 처리 과정은 오디오 및 비디오 데이터를 Mel spectrogram으로 변환하고, 비디오 프레임을 추출하여 사전 훈련된 모델을 통해 특징을 추출한 후, 융합 모듈로 결합하여 최종적으로 분류 결정을 내립니다. 데이터셋의 경우, 원래 소리가 없는 RLVS 데이터셋을 보강하고, 공공장소에서 수집된 다양한 비디오를 추가했습니다.

- **Performance Highlights**: HFBDL 모델은 검증 데이터에서 96.67%의 높은 정확도를 기록했으며, 실제 테스트에서 54개의 비디오 중 52개를 정확하게 탐지했습니다. 이는 기존의 최첨단 기법들보다 높은 성능을 나타냅니다.



### Self-Introspective Decoding: Alleviating Hallucinations for Large Vision-Language Models (https://arxiv.org/abs/2408.02032)
- **What's New**: 최근 LLMs(Large Language Models)의 성공을 다룬 많은 연구들이 LVLMs(Large Vision-Language Models)로 확장되고 있습니다. 이 논문에서는 LVLMs의 주요 문제점인 '환각(hallucination)' 문제를 다루고자 소개된 'Self-Introspective Decoding(SID)' 방법에 대해 발표했습니다. SID는 외부 지식 없이도 환각 문제를 해결할 수 있는 간단하지만 효과적인 방법입니다.

- **Technical Details**: SID는 'Context and Text-aware Token Selection(CT2S)' 전략을 도입하여 LVLMs의 초기 레이어에서 중요하지 않은 비전 토큰을 선택적으로 유지합니다. 이를 통해 텍스트 정보에 따라 적응적으로 증폭된 환각을 사용자가 선택하여 원래 토큰 로짓(logits)에서 증폭된 비전-텍스트 연관성을 빼내어 깨끗한 디코딩을 수행합니다. 이는 다양한 메트릭에서 높은 품질의 텍스트를 생성하며, 추가적인 지식 없이도 효과적입니다.

- **Performance Highlights**: 광범위한 실험 결과, SID는 기존의 대비 디코딩(contrastive decoding) 방법보다 낮은 환각 수준을 보이며 더 높은 품질의 텍스트를 생성합니다. 또한, SID는 큰 추가 컴퓨팅 비용 없이도 효과적입니다. 전체 결과는 https://github.com/huofushuo/SID 에서 확인할 수 있습니다.



### Faster Diffusion Action Segmentation (https://arxiv.org/abs/2408.02024)
Comments:
          25 pages, 6 figures

- **What's New**: EffiDiffAct는 Temporal Action Segmentation(TAS)의 효율성을 높이기 위해 제안된 새로운 알고리즘입니다. 특히, 이 알고리즘은 가벼운 임시 특성 인코더(lightweight temporal feature encoder)를 개발하고, 시뮬레이션 중 유사성 지표에 기반하여 동적으로 시간을 조정할 수 있는 적응형 스킵 전략(adaptive skip strategy)을 도입하여 기존의 Transformer 기반 모델의 한계를 극복합니다.

- **Technical Details**: EffiDiffAct는 기존의 무거운 Transformer 기반 인코더를 대신하여 가벼운 임시 특성 인코더를 사용하여 계산 오버헤드를 줄이고 'rank collapse' 현상을 완화합니다. 또한, 고정된 스킵 전략 대신 적응형 스킵 전략을 적용하여 유사성 지표에 기반한 동적 시간 조정을 통해 모델의 추론 속도를 향상시킵니다. 이 두 가지 혁신적인 방법이 결합되어 실시간 응용 분야에서 높은 정확성을 유지하면서 계산 효율성을 극대화합니다.

- **Performance Highlights**: EffiDiffAct는 50Salads, Breakfast, GTEA 등 다수의 공개 데이터셋에서 종합적인 실험을 통해 높은 효율성과 효과성을 입증하였습니다. 높은 정확도로 TAS 작업에서의 성능을 크게 향상시켰으며, 기존 방법들에 비해 계산 자원을 적게 소모하면서도 실시간 분석에 적합한 속도를 달성했습니다.



### Individualized multi-horizon MRI trajectory prediction for Alzheimer's Diseas (https://arxiv.org/abs/2408.02018)
Comments:
          MICCAI 2024 LDTM workshop

- **What's New**: 최근 발표된 논문에서는 MRI를 통해 측정되는 신경 퇴행을 알츠하이머병(AD) 진단의 잠재적 바이오마커로 활용할 수 있음을 제안했습니다. 그러나 기존의 MRI는 아밀로이드(Amyloid)나 타우(Tau) 기반의 바이오마커만큼 명확하지 않다는 한계가 있습니다. 이에 연구팀은 조절된 변형 오토인코더(Conditional Variational Autoencoder, CVAE)를 활용하여 개인화된 MRI 예측 모델을 개발하였습니다. 이 모델은 환자의 나이, 질병 상태, 그리고 이전 스캔을 기반으로 향후 최대 10년 이내의 MRI 변화를 예측할 수 있습니다.

- **Technical Details**: 이 연구에서는 알츠하이머병 신경영상 이니셔티브(ADNI)와 개방 접근 시리즈 이미지 연구(OASIS)에서 수집한 일련의 이미징 데이터를 사용해 새로운 아키텍처를 훈련시켰습니다. 이 아키텍처는 나이, 질병 상태, 이전 스캔을 조건으로 한 데이터 입력을 받아 복잡한 픽셀 변화를 모델링 합니다. 특히, 이중 인코더(Double-Encoder) CVAE 아키텍처가 도입되어 더 현실적이고 높은 해상도의 출력을 제공합니다. 이 모델은 멀티-호라이즌(Multi-Horizon) 예측을 가능하게 하여, 임의의 시간 간격에 대한 예측도 가능합니다.

- **Performance Highlights**: 모델은 ADNI와 OASIS 데이터셋에서 높은 해상도의 개인화된 이미지를 생성하는 데 성공하였으며, 기존 다양한 모델과 비교해 더 나은 성능을 보였습니다. 테스트셋과 외부 독립 데이터셋에서 평균 제곱 오차(MSE) 기준으로 높은 예측 정확도를 입증했습니다. 또한, 후속 MRI를 이미 가지고 있는 경우에는 질병 상태 분류기(Classifier)를 구축할 수 있는 가능성도 제시되었습니다. 이는 AD 초기 진단을 도울 수 있으며, 치료 효과 추정에서도 대조 기준으로 사용될 수 있습니다.



### Unsupervised Representation Learning by Balanced Self Attention Matching (https://arxiv.org/abs/2408.02014)
- **What's New**: 이 연구에서는 self-supervised 방법 중 하나로 이미지 특징을 임베딩(embedding)하는 과정에서 발생할 수 있는 불안정성과 feature collapse 문제를 해결하기 위한 BAM(Balanced Attention Matching) 방법을 제안합니다. 기존 방법들과는 다르게, 본 연구에서는 입력 이미지의 다양한 뷰(view)를 직접 매칭하는 것이 아닌, self-attention 벡터를 매칭하는 방식을 사용합니다.

- **Technical Details**: BAM은 일련의 augmentations(증강된 이미지 집합)에 대한 분포를 비교하는 self-attention 벡터를 매칭하는데 초점을 맞춥니다. 이는 augmentations에 대한 전역적으로 균형 잡히고 엔트로피 정규화된 버전으로 맞추기 위한 손실을 최소화함으로써 수행됩니다. 이 방법은 간단한 self-optimal-transport 연산을 통해 이루어집니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 BAM 방법은 semi-supervised와 transfer learning 벤치마크에서 선도적인 방법들과 경쟁력 있는 성능을 보였습니다. 이러한 결과는 BAM이 안정적이고 풍부한 표현을 제공하며, feature collapse 문제를 효과적으로 방지함을 입증합니다.



### AdaCBM: An Adaptive Concept Bottleneck Model for Explainable and Accurate Diagnosis (https://arxiv.org/abs/2408.02001)
Comments:
          Accepted at MICCAI 2024, the 27th International Conference on Medical Image Computing and Computer Assisted Intervention

- **What's New**: 이 논문은 대형 Vision-Language 모델인 CLIP와 Concept Bottleneck Models (CBMs)을 통합하여 인간이 이해할 수 있는 의미로 딥러닝 모델의 결정을 설명하는 새로운 접근법을 제시합니다. 특히, CLIP와 CBMs 사이에 위치한 적응 모듈을 도입하여 의료 이미지 진단 태스크에서의 성능과 해석 가능성을 모두 향상시키고자 하였습니다.

- **Technical Details**: 논문에서 제시된 방법은 기존 CBM의 기하학적 표현을 단순한 선형 분류 시스템으로 다시 검토합니다. 분석 결과, CBM 이후의 후처리 모듈은 단순히 분류 결과를 재조정하는 데 그쳐 시스템의 학습 잠재력을 충분히 활용하지 못하고 있다는 것을 발견했습니다. 이를 해결하기 위해, CLIP와 CBM 사이에 전략적으로 위치한 적응 모듈을 도입하였고, GPT-4를 이용한 완전 Prompt Engineering 기반의 개념 생성을 설계했습니다.

- **Performance Highlights**: 이 새로운 접근법은 CLIP의 표현력을 유지하면서도 CBM의 해석 가능성을 저해하지 않고 높은 수준의 분류 성능을 보였습니다. 또한 통계적 검정을 통해 생성된 개념들의 질을 평가하여, 실제 의료 진단 태스크에 유용한 개념을 선정하는 방법을 개발하였습니다.



### What Happens Without Background? Constructing Foreground-Only Data for Fine-Grained Tasks (https://arxiv.org/abs/2408.01998)
- **What's New**: 이번 연구에서는 FG(Tabular 형식의 데이터 모델)에 대해서 기존의 방법들이 배경 영역에 집착하고 있다는 단점을 지적하며, 이를 개선하기 위해 SAM(Segment Anything Model)과 Detic(Open-vocabulary object detector)을 활용하여 배경 없는 전경(foreground)만 포함된 세밀한 데이터셋을 구축하는 파이프라인을 제안합니다. 이를 통해 모델이 실제로 효과적인 구별 정보를 더 잘 포착할 수 있게 되어 알고리즘의 성능을 향상시킬 수 있습니다.

- **Technical Details**: 이 연구는 전경 데이터 생성을 위한 자동화된 파이프라인을 설계하였습니다. 이 파이프라인은 SAM을 사용하여 객체를 세밀하게 분할하고, Detic이나 Grounding DINO와 같은 오픈 보캐블러리 객체 검출기를 통해 SAM에 경계 상자를 제공하여 전경 객체를 추출합니다. 이를 통해 생성된 데이터셋은 CUB, Stanford-Cars, Aircraft와 같은 대표적인 벤치마크 데이터셋의 일부로 활용되었습니다.

- **Performance Highlights**: 파이프라인을 통해 생성된 전경 데이터셋을 사용했을 때, 기존 데이터셋을 사용한 모델보다 더욱 높은 성능을 보였습니다. 특히 Transformer 기반의 ViT(Vision Transformer) 아키텍처에서는 배경 소음 제거가 성능 향상에 크게 도움이 되었습니다. 실험 결과, 전경 데이터를 사용한 모델이 원본 데이터로 훈련된 모델보다 일관되게 높은 정확도를 보였습니다.



### DeMansia: Mamba Never Forgets Any Tokens (https://arxiv.org/abs/2408.01986)
- **What's New**: 이번 논문은 Transformer 아키텍처의 수학적 기초를 검토하고, 특히 긴 시퀀스를 처리하는 데 있어 이들의 한계를 조명합니다. Mamba, Vision Mamba(ViM), LV-ViT 모델을 바탕으로 한 새로운 아키텍처인 DeMansia를 제안합니다. DeMansia는 상태 공간 모델(state space model)과 토큰 레이블링(token labeling) 기법을 통합하여 이미지 분류 성능을 향상시키며, 전통적인 Transformer가 갖고 있는 계산 비용 문제를 효율적으로 해결합니다. 논문에서 제안된 아키텍처는 GitHub에서 구현된 소스를 확인할 수 있습니다.

- **Technical Details**: Transformer 아키텍처들은 자가 주의 메커니즘(self-attention mechanism)을 사용하여 입력 데이터의 다양한 부분을 동적으로 가중하여, 더 세밀하고 상황에 맞는 해석을 가능하게 합니다. 하지만 시퀀스 길이가 길어질수록 계산 복잡도가 제곱에 비례하여 증가한다는 한계가 있습니다. DeMansia는 Mamba와 Vision Mamba(ViM)의 장점을 결합하고, LV-ViT의 트레이닝 파이프라인을 참고하여 이미지 분류 작업에서의 성능을 높이기 위해 고안되었습니다. 상태 공간 모델(state space model)과 토큰 레이블링(token labeling)을 혁신적으로 적용하여 맥락의 풍부함을 유지하면서도 계산 효율을 확보합니다.

- **Performance Highlights**: DeMansia 아키텍처는 기존의 모델들과 비교하여 효과적인 성능을 보여주었습니다. 특히 이미지 분류 작업에서 높은 성능을 보이며, 자원이 제한된 환경에서도 고성능을 유지합니다. 이러한 성능은 현대의 모델들과 비교한 벤치마크 결과에서 입증되었습니다.



### AdvQDet: Detecting Query-Based Adversarial Attacks with Adversarial Contrastive Prompt Tuning (https://arxiv.org/abs/2408.01978)
- **What's New**: 깊은 신경망(DNNs)은 대규모 공격 시나리오에 취약한 것으로 알려져 있으며, 특히 블랙박스(black-box) 설정에서도 취약합니다. 최근 연구는 블랙박스 설정에서 적대적 공격을 감지하고 차단하기 위한 새로운 방법, 'Adversarial Contrastive Prompt Tuning (ACPT)'을 제안합니다. 이 방법은 CLIP 이미지 인코더를 미세 조정하여 중간 적대적 쿼리들의 임베딩을 유사하게 만드는 것입니다.

- **Technical Details**: ACPT는 대조 학습(contrastive learning)과 적대적 학습(adversarial learning)을 활용하여 CLIP 이미지 인코더를 미세 조정합니다. 이를 통해 어떤 두 중간 적대적 쿼리들도 유사한 특징 벡터를 만들도록 합니다. ACPT는 5개의 벤치마크 데이터셋에서 7가지 쿼리 기반 공격을 대상으로 테스트되었으며, 평균 3-shot에서 97%, 5-shot에서 99%의 탐지율을 달성했습니다. 기존의 방법보다 탐지 성능이 48~49% 향상되었습니다.

- **Performance Highlights**: ACPT는 3가지 유형의 적응형 공격(adaptive attacks)에 대해 강건함을 입증했습니다. 또한, 제안된 탐지 프레임워크 'AdvQDet'는 5개의 쿼리 이내에 7가지 최신 쿼리 기반 공격을 99% 이상의 탐지율로 감지할 수 있습니다. 특히, 대부분의 데이터셋에서 제로샷(zero-shot) 성능을 보여주며, 더욱 넓은 범위의 데이터셋에서도 효과적입니다.



### Label Augmentation for Neural Networks Robustness (https://arxiv.org/abs/2408.01977)
Comments:
          21 pages, 4 figures, Published at 3rd Conference on Lifelong Learning Agents (CoLLAs), 2024

- **What's New**: 이 연구에서는 새로운 데이터 증강 기법인 Label Augmentation (라벨 증강, LA)을 제안합니다. 이 기법은 일반적인 변형과 의도적인 공격에 모두 견디는 강건성을 높이고 불확실성 추정을 개선하는 데 중점을 둡니다. LA를 활용하면 클린 오류율이 최대 23.29% 개선될 수 있으며, 일반적인 오염 상황에서의 강건성도 최대 24.23% 향상됩니다. 또한, FGSM 및 PGD 공격에 대해 각각 최대 53.18%와 24.46%까지 강건성이 향상됩니다.

- **Technical Details**: Label Augmentation (라벨 증강, LA)은 객체의 클래스를 잡음으로부터 분리하는 데 효과적인 방법입니다. 이 방법은 클래스 식별을 위해 잡음을 제거하며, 일반적인 오염과 의도적인 변형 모두에 대한 강건성을 동시에 개선합니다. 추가로, 불확실성 추정도 향상됩니다. 이는 머신러닝 모델이 흔히 겪는 분포 이동에 대한 취약성과 과신 문제를 해결하는 데에도 도움을 줍니다.

- **Performance Highlights**: 라벨 증강 (LA)을 적용한 결과, 클린 데이터의 오류율이 최대 23.29% 개선되었습니다. 일반적인 오염 환경에서도 강건성이 24.23%까지 향상되었으며, FGSM (Fast Gradient Sign Method) 공격에 대해서는 53.18%, PGD (Projected Gradient Descent) 공격에 대해서는 24.46%의 강건성 향상을 보였습니다.



### Single-Point Supervised High-Resolution Dynamic Network for Infrared Small Target Detection (https://arxiv.org/abs/2408.01976)
- **What's New**: Infrared 작은 목표 탐지(IRSTD) 작업에서 기존 방법들이 겪는 두 가지 주요 문제를 해결하기 위해 단일 포인트 지도(supervision)를 사용하는 고해상도 동적 네트워크(SSHD-Net)를 제안했습니다. 이 접근 방식은 단일 포인트 관리만으로도 기존 최첨단(SOTA) 탐지 성능을 능가합니다. SSHD-Net은 높은 해상도의 상호 교차 특성 추출 모듈(HCEM), 동적 좌표 융합 모듈(DCFM), 고해상도의 다수준 잔차 모듈(HMRM), 적응형 목표 위치 탐지 헤드(ATLDH)로 구성됩니다.

- **Technical Details**: HCEM은 계단식 특성 연쇄 채널(SFCC)을 통해 양방향 특성 상호 작용을 달성하여 네트워크 깊이와 특성 해상도를 균형있게 유지합니다. DCFM은 전역 및 지역 특성을 통합하여 복잡한 배경에서도 높은 항간섭 능력을 제공합니다. HMRM은 심층 IR 작은 목표의 의미 정보를 효과적으로 추출하며, ATLDH는 적응형 비 최대 억제(ANMS)를 통해 목표 탐지 정확도를 향상시킵니다.

- **Performance Highlights**: NUDT-SIRST 및 IRSTD-1k 등 두 개의 공개 데이터셋 실험 결과, SSHD-Net은 단일 포인트 지도(supervision)만으로도 다른 최첨단 방법들보다 더 나은 탐지 성능을 보였습니다.



### AnomalySD: Few-Shot Multi-Class Anomaly Detection with Stable Diffusion Mod (https://arxiv.org/abs/2408.01960)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문에서는 산업 제조에서 결함 부품을 식별하기 위한 필수 작업인 이상 감지(anomaly detection)에서 새로운 접근 방식을 제안합니다. AnomalySD라 불리는 이 프레임워크는 Stable Diffusion(SD) 모델을 활용하여 최소한의 정상 데이터(few-shot)만으로도 다양한 이상을 감지할 수 있도록 설계되었습니다. 주요 기여 점으로는 SD 모델의 텍스트 설명과 전경 마스크 기법을 이용하여 이상 영역을 정상으로 대체하며, 다중 스케일 마스크 전략과 프로토타입 기반 마스크 전략을 도입하여 다양한 이상 영역을 정확하게 마킹 및 인페인팅(inpainting)하는 방식을 제안합니다.

- **Technical Details**: 제안된 AnomalySD 프레임워크는 몇 가지 핵심 기술로 구성되어 있습니다. 먼저, SD 모델에 적응시키기 위해 계층적 텍스트 설명(hierarchical text descriptions)과 전경 마스크(foreground mask) 메커니즘을 설계하여 모델 미세 조정을 수행합니다. 추론 단계에서는 다중 스케일 마스크 전략(multi-scale mask strategy)과 프로토타입 기반 마스크 전략(prototype-guided mask strategy)을 적용하여 다양한 이상 영역을 마킹합니다. 그런 다음 이를 통해 얻은 모든 마스크의 인페인팅 결과를 기반으로 이상 점수(anomaly score)를 추정합니다.

- **Performance Highlights**: 제안된 AnomalySD 프레임워크는 MVTec-AD와 VisA 데이터셋 실험에서 뛰어난 성능을 보였습니다. MVTec-AD 데이터셋에서는 다중 클래스(multi-class)와 원샷(one-shot) 설정에서 각각 93.6%와 94.8%의 AUROC를 기록하였고, VisA 데이터셋에서는 86.1%와 96.5%의 AUROC를 기록하였습니다. 이러한 결과는 AnomalySD가 최소한의 정상 데이터만으로도 높은 성능을 보유함을 증명합니다.



### Dataset Scale and Societal Consistency Mediate Facial Impression Bias in Vision-Language AI (https://arxiv.org/abs/2408.01959)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 본 연구에서는 43개의 CLIP (Contrastive Language-Image Pretraining) 비전-언어 모델을 분석하여 이들이 인간과 유사한 얼굴 인상 편향(facial impression biases)을 학습하는지를 평가하였습니다. 결과적으로 이러한 편향이 세 가지 구별되는 CLIP 모델 계열에서 나타남을 확인하였습니다.

- **Technical Details**: 연구에서는 비관측 시각적 속성(예: 신뢰성, 성적 지향성) 관련 인상 편향이 가장 큰 데이터셋에서 훈련된 모델들에서만 나타난다는 사실을 발견하였습니다. 이는 더 많은 데이터와의 적합성이 더 미세한 사회적 편향을 재현하는 결과를 초래한다는 것을 시사합니다. 또한, 계층적 클러스터링 방법(hierarchical clustering approach)을 통해 데이터셋의 크기가 편향 구조의 유사성 정도를 예측할 수 있음을 보여주었습니다. 마지막으로, 텍스트 인코더로써 CLIP을 사용하는 Stable Diffusion 모델들이 얼굴 인상 편향을 학습하고 있으며 이 편향이 Stable Diffusion XL-Turbo 모델에서 인종적 편향과 교차됨을 발견하였습니다.

- **Performance Highlights**: 이번 연구는 CLIP 모델들이 얼굴 인상 편향을 학습하는 메커니즘을 최초로 규명하였으며, 이를 통해 이러한 모델들이 편향 연구에 중요한 도구가 될 수 있음을 시사합니다. 다만, 해당 모델들을 범용적으로 사용하기 위해서는 데이터셋의 엄격한 큐레이션이 필요합니다.



### CACE-Net: Co-guidance Attention and Contrastive Enhancement for Effective Audio-Visual Event Localization (https://arxiv.org/abs/2408.01952)
Comments:
          Accepted by ACM MM 2024. Code is available at this this https URL

- **What's New**: CACE-Net 모델은 오디오-비디오 이벤트 위치 식별 분야에서 혁신적인 방법을 도입했습니다. 기존의 방법들과 다르게, 단순히 오디오 신호를 시각 정보에 맞추는 것이 아니라, 오디오와 비디오 정보 간의 양방향 크로스 모달(attention) 주의를 통해 상호 지도를 제공합니다. 이를 통해 모달간의 불일치를 줄이고, 정교하고 식별 가능한 특성을 추출해냅니다.

- **Technical Details**: CACE-Net은 오디오-비디오 공동 지도를 위한 주의 메커니즘을 도입했습니다. 이는 오디오-비주얼 이벤트 로컬라이제이션에서 비디오 정보가 오디오 정보의 지도를 받는 것이 아니라, 양방향으로 상호 지도를 받는 방식을 채택했습니다. 배경-이벤트 대조 강화 기법을 사용하여 퓨즈된 특징의 변별력을 높이고, 세밀한 전이 학습을 통해 복잡한 멀티모달 입력에서 더욱 정교하고 구별 가능한 특징을 추출할 수 있도록 했습니다.

- **Performance Highlights**: CACE-Net은 AVE 데이터셋에서 새로운 벤치마크를 세웠으며, 복잡한 멀티모달 학습과 비제한 비디오에서 이벤트 로컬라이제이션을 효율적으로 처리할 수 있음을 입증했습니다. 이를 통해 기존의 오디오-비디오 이벤트 위치 식별 성능을 크게 향상시켰습니다.



### Masked Angle-Aware Autoencoder for Remote Sensing Images (https://arxiv.org/abs/2408.01946)
Comments:
          This paper has been accepted by ECCV 2024

- **What's New**: 새로운 논문에서는 원격 감지(Remote Sensing, RS) 이미지를 자연 이미지와 구별하여 학습할 수 있는 Masked Angle-Aware Autoencoder(MA3E)를 제안합니다. 이는 기존의 자기 지도형 학습(self-supervised learning) 방법들이 간과한 RS 객체의 다양한 각도를 인지하여 학습할 수 있게 합니다.

- **Technical Details**: MA3E는 비대칭 인코더-디코더(asymmetric encoder-decoder) 구조를 따릅니다. 각 RS 이미지에서 임의의 자세로 회전된 자른 이미지를 생성하는 '스케일링 센터 크롭(scaling center crop)' 작업을 제안하여 명시적인 각도 변화를 도입합니다. 추가적으로 최적 수송(Optimal Transport, OT) 손실을 사용하여 회전된 자른 이미지 조각에 관련 있는 원본 이미지 조각을 할당하여 재구성합니다.

- **Performance Highlights**: MA3E는 7개의 RS 이미지 데이터셋에서 세 가지 다운스트림 작업(장면 분류, 회전 객체 감지, 의미론적 분할)에서 기존의 사전 학습 방법보다 더 경쟁력 있는 성능을 보였습니다. 특히 다양한 각도를 인지하고 배우는 능력을 통해 회전 불변(rotation-invariant) 표현을 효과적으로 학습할 수 있게 합니다.



### Generalized Maximum Likelihood Estimation for Perspective-n-Point Problem (https://arxiv.org/abs/2408.01945)
- **What's New**: 이번 연구에서는 Perspective-n-Point (PnP) 문제에 대한 새로운 접근법을 제안합니다. 기존의 PnP 방법들이 관측 데이터의 비등방성(Anisotropy) 불확실성을 무시한 반면, 이번 연구는 이러한 한계를 극복하고자 합니다. 새로운 방법인 GMLPnP(Generalized Maximum Likelihood PnP)는 GLS 알고리즘을 반복하여 포즈와 불확실성을 동시에 추정합니다. 또한, 카메라 모델과 독립적으로 작동하며, 다양한 카메라 모델에서 일반화된 해결책을 제공할 수 있습니다.

- **Technical Details**: GMLPnP는 비등방성의 관측 불확실성을 고려한 일반화된 최대 우도 방법입니다. 일반화된 최소 제곱법(Generalized Least Squares, GLS)을 사용하여 반복적으로 포즈와 관측 불확실성의 분포 파라미터를 추정합니다. 이 방법은 카메라 모델과 분리되어 작동하기 때문에, 모든 종류의 카메라 모델, 예를 들어 어안 렌즈 카메라에서도 사용할 수 있습니다. 또한, 기존의 PnP 솔버들이 isotropic Gaussian 노이즈를 가정하는 반면, 본 연구는 실제 데이터의 특성을 반영하여 더 현실적인 노이즈 모델을 사용합니다.

- **Performance Highlights**: GMLPnP는 다양한 실험 결과에서 높은 정확도를 보여줍니다. 예를 들어, TUM-RGBD 데이터셋에서 회전 정확도와 변환 정확도를 각각 4.7%와 2.0% 개선하였으며, KITTI-360 데이터셋에서도 각각 18.6%와 18.4% 개선되었습니다. 특히, 불확실성이 큰 관측 상황에서, UAV의 위치 추정 정확도는 34.4%까지 개선되었습니다. 이는 기존의 최고 성능을 보이는 방법들과 비교해도 월등히 뛰어난 결과입니다.



### RobNODDI: Robust NODDI Parameter Estimation with Adaptive Sampling under Continuous Representation (https://arxiv.org/abs/2408.01944)
- **What's New**: RobNODDI는 Neurite Orientation Dispersion and Density Imaging (NODDI)의 매개변수를 추정하는 딥러닝 모델로, 테스트 중 확산 방향과 훈련 중 확산 방향이 일치하지 않아도 성능이 유지될 수 있도록 설계된 점이 새로운 특징입니다.

- **Technical Details**: RobNODDI는 적응 샘플링(adaptive sampling)과 연속 표현(continuous representation)을 결합합니다. 적응 샘플링은 다양한 데이터를 최대한 활용해 정보를 추출하는데 유리하며, 연속 표현은 SH(Spherical Harmonic) 보정을 통해 모델이 더 유연하게 테스트할 수 있도록 합니다. LSTM(long short-term memory) 유닛과 완전 연결층(fully connected layers)을 사용하여 연속 신호를 학습합니다.

- **Performance Highlights**: Human Connectome Project(HCP) 데이터셋을 활용한 실험에서 RobNODDI는 기존의 딥러닝 모델보다 일반화 성능과 강인성이 크게 향상되었습니다. 100명의 데이터를 사용해 훈련, 검증, 테스트를 거친 결과, 보다 안정적이고 유연하게 NODDI 매개변수를 추정할 수 있게 되었습니다.



### A Survey and Evaluation of Adversarial Attacks for Object Detection (https://arxiv.org/abs/2408.01934)
Comments:
          14 pages

- **What's New**: 딥러닝 모델은 다양한 컴퓨터 비전 작업에서 뛰어난 성능을 보이지만, 입력 데이터에 미세한 교란(adversarial examples)이 발생하면 잘못된 예측을 하게 됩니다. 이러한 취약성은 자율 주행차, 보안 감시, 항공기 상태 모니터링 같은 안전과 관련된 응용 분야에서 큰 위험을 초래합니다. 이 논문은 객체 탐지(object detection)에 특화된 적대적 공격(adversarial attacks)의 포괄적인 분류를 제공하며, 기존 적대적 견고성 평가 메트릭을 검토하고, 오픈 소스 공격 방법 및 모델 견고성을 체계적으로 평가합니다. 주요 관찰 결과를 제공하여 공격 효과와 이에 대한 대응책을 이해하고 개선할 수 있습니다.

- **Technical Details**: 이 논문은 적대적 공격에 대한 포괄적인 분류 체계를 제안하며, 기존의 적대적 견고성 평가 지표(robustness evaluation metrics)를 검토합니다. 또한, 다양한 오픈 소스 공격 방법과 모델의 견고성을 체계적으로 평가합니다. 이를 통해 객체 탐지 시스템의 자동화 보안을 확보하기 위한 중요한 연구 과제를 제시합니다.



### CAF-YOLO: A Robust Framework for Multi-Scale Lesion Detection in Biomedical Imagery (https://arxiv.org/abs/2408.01897)
- **What's New**: CAF-YOLO는 기존 의료 영상 객체 탐지 방식을 개선한 새로운 방법론으로, YOLOv8 아키텍처를 기반으로 합니다. 이 방법은 CNN(convolutional neural networks)과 트랜스포머(Transformers)의 강점을 결합하여 더 나은 탐지 성능을 제공합니다. 특히, 작은 크기의 병변(lesions) 식별에 초점을 맞추고 있습니다.

- **Technical Details**: CAF-YOLO는 주의력(attention)과 컨볼루션 융합 모듈(ACFM, Attention and Convolution Fusion Module)을 도입하여 전역(global)과 지역(local) 특징들을 동시에 모델링할 수 있게 합니다. 또한, 다중 스케일 신경망(MSNN, Multi-Scale Neural Network)을 설계하여 다양한 스케일의 정보를 수집하여 피처들이 단일 스케일에서 제한되는 문제를 해결했습니다.

- **Performance Highlights**: CAF-YOLO는 BCCD와 LUNA16 같은 널리 사용되는 데이터셋에서 뛰어난 성능을 보이며, 다양한 복잡한 미세 병변을 정확하게 탐지하고 정밀하게 위치를 파악합니다. 코드 역시 공개되어 있어 추가적인 연구 및 응용이 가능합니다.



### FBINeRF: Feature-Based Integrated Recurrent Network for Pinhole and Fisheye Neural Radiance Fields (https://arxiv.org/abs/2408.01878)
Comments:
          18 pages

- **What's New**: 이번 논문에서는 굴절 이미지 왜곡을 호환할 수 있도록 설계된 유연한 번들 조정 방법과, 피처 기반 재귀 신경망을 통합하여어안 카메라 데이터셋에서 연속적인 새로운 뷰를 생성하는 FBINeRF를 제안합니다. 기존 DBARF와 달리 이 방법은 어안 렌즈에서 발생하는 왜곡 문제를 효과적으로 해결합니다. 또한 신뢰성이 떨어지는 초기 깊이 맵 설정을 MiDaS 기반 깊이 선행 조건 (depth priors)으로 보완하여 보다 정확한 결과를 도출합니다.

- **Technical Details**: FBINeRF는 NeRF 렌더링을 위해 IBRNet을 기반으로 하여 상대적인 포즈 개선을 수행하는 깊은 재귀 신경망을 사용합니다. 어안 카메라 파이프라인에서는 DenseNet에서 추출한 이미지 피처를 처리하고, 가변적인 GRU(Adaptive GRUs)를 사용하여 포즈를 업데이트하는 방식으로 설계되었습니다. 이러한 구조는 반복적인 신경망 내에서 피처 설정 맵과 조합하여 상대적인 카메라 포즈를 함께 학습하여 전통적인 어안 NeRF 방법에 비해 빠른 수렴을 가능하게 합니다.

- **Performance Highlights**: FBINeRF는 실험을 통해 핀홀 카메라와 어안 카메라 모두에서 높은 정확도와 품질을 가진 결과물을 나타냈습니다. 핀홀 카메라 모델에서는 MiDaS와 ZoeDepth에서 영감을 받은 깊이 선행 조건(depth priors) 추가를 통해 DBARF의 수렴 문제를 해결하며 보완했습니다. 또한 어안 카메라 모델에서는 빠른 수렴 및 높은 충실도의 연속 렌더링 결과를 보여주었습니다.



### Graph Unfolding and Sampling for Transitory Video Summarization via Gershgorin Disc Alignmen (https://arxiv.org/abs/2408.01859)
Comments:
          13 pages, 5 figures

- **What's New**: 사용자 생성 비디오 (User-generated videos, UGVs)를 더 효율적으로 요약하는 새로운 키프레임 추출 알고리즘을 제안합니다. 기존의 복잡한 알고리즘 대신, Gershgorin 디스크 정렬 (Gershgorin disc alignment, GDA)을 기반으로 한 빠른 그래프 샘플링 방법을 사용하여 선형 시간 내에 주요 프레임을 추출합니다.

- **Technical Details**: 제안된 알고리즘은 UGV의 프레임 시퀀스를 $M$-홉 경로 그래프로 모델링한 후, 이를 일반화된 그래프 라플라시안 행렬 $\mathcal{L}$를 통해 $1$-홉 경로 그래프로 전개합니다. 그리고 $\oldsymbol{B} = \textit{diag}\left(\mathbf{h}\right) + \mu \mathcal{L}$의 가장 작은 고유값 $\\lambda_{\\min}(\\\mathbf{B})$를 최대화하는 것이 최악의 신호 재구성 오류를 최소화하는 것임을 증명합니다. 대신, Gershgorin 원 정리 (Gershgorin circle theorem, GCT)에 기반하여 $\\lambda^-_{\\min}(\\\mathbf{B})$를 최대화하는 새로운 그래프 샘플링 알고리즘을 사용합니다.

- **Performance Highlights**: 여러 짧은 비디오 데이터셋에서의 광범위한 실험 결과, 제안된 알고리즘이 기존 최신 기법(SOTA)들과 비교하여 유사하거나 더 나은 비디오 요약 성능을 보이면서도, 계산 복잡도가 크게 감소되었음을 확인했습니다.



### Supervised Image Translation from Visible to Infrared Domain for Object Detection (https://arxiv.org/abs/2408.01843)
- **What's New**: 이 연구는 가시광 이미지(visible imagery)에서 적외선 이미지(infrared imagery)로의 변환을 학습하는 새로운 접근 방식을 제안합니다. 이를 통해 객체 감지(object detection)와 같은 하위 작업에서의 정확도를 향상시키고자 합니다. 기존의 접근 방식들은 반복적 최적화나 엔드 투 엔드 딥 컨볼루셔널 네트워크를 사용하여 두 도메인 간의 특징 융합(feature fusion) 시도가 있었으나, 본 연구는 이를 이미지 변환(image translation) 문제로 간주하며, 생성적 적대 신경망(GAN)과 객체 감지 모델을 사용한 두 단계의 학습 전략을 채택합니다.

- **Technical Details**: 본 연구는 조건부 적대 신경망(conditional GAN) 프레임워크를 활용하여 가시광 이미지에서 고해상도 적외선 이미지를 생성합니다. Pix2Pix[12] 모델을 기반으로 한 이 프레임워크는 U-Net 구조를 가진 생성기(generator)와 패치 기반의 완전 컨볼루셔널 디스크리미네이터(discriminator)를 포함합니다. 또한, 해결책으로 coarse-to-fine 생성기, 다중 스케일 디스크리미네이터 아키텍처, 견고한 적대적 학습 목표 함수를 사용하여 성능을 향상시켰습니다.

- **Performance Highlights**: 제안된 접근법은 Yolov5, Mask RCNN, Faster RCNN 모델들을 사용하여 객체 감지 성능을 평가했습니다. 또한, 슈퍼 해상도(super-resolution) 단계를 통합하여 모델 정확도를 최대 5.3% mAP 향상시켰습니다.



### E$^3$NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images (https://arxiv.org/abs/2408.01840)
- **What's New**: 이번 연구에서는 기존의 Neural Radiance Fields(NeRF) 방식의 문제를 극복하기 위해 Efficient Event-Enhanced NeRF (E$^3$NeRF)을 제안합니다. 이 방법은 RGB 이미지와 이벤트 스트림(Event streams)을 결합함으로써, 특히 비균일한 모션과 저조도 환경에서 블러가 포함된 입력 이미지로부터 선명한 NeRF를 재구성할 수 있도록 합니다.

- **Technical Details**: E$^3$NeRF에서는 이벤트 스트림을 신경 볼륨 표현 학습 과정에 효과적으로 도입하기 위해 이벤트 강화 블러 렌더링 손실(event-enhanced blur rendering loss)과 이벤트 렌더링 손실(event rendering loss)을 제안합니다. 이를 통해 네트워크가 실제 블러 과정과 이벤트 생성 과정을 모델링하여 학습할 수 있도록 유도합니다. 특히, 이벤트 스트림의 시공간 정보를 활용하여 시간적 블러에 균등한 학습 주의를 분배하고 공간적 블러 텍스처에 집중할 수 있도록 합니다. 또한, 실제 데이터에서도 이벤트를 활용하여 포즈 추정(camera pose estimation) 프레임워크를 구축함으로써 실용성을 강화하였습니다.

- **Performance Highlights**: E$3$NeRF은 기존의 이미지 기반 또는 이벤트 기반 NeRF와 비교하여 이벤트와 이미지 사이의 내부 관계를 더욱 깊이 있게 활용합니다. 실험 결과, E$3$NeRF는 합성 데이터와 실제 데이터 모두에서 특히 비균일한 모션과 저조도 환경에서 블러가 포함된 이미지로부터 효과적으로 선명한 NeRF를 학습할 수 있음을 확인했습니다. 이는 기존의 방법들보다 훨씬 더 효율적이고 강력한 성능을 보였습니다.



### TS-SAM: Fine-Tuning Segment-Anything Model for Downstream Tasks (https://arxiv.org/abs/2408.01835)
- **What's New**: 이번 연구에서는 SAM을 다운스트림 작업에서 더욱 효율적으로 사용할 수 있도록 TS-SAM (Two-Stream SAM)을 제안합니다. 기존의 PEFT(Parameter-Efficient Fine-Tuning) 방식과 달리, TS-SAM은 Convolutional Side Adapter (CSA)와 Multi-scale Refinement Module (MRM), Feature Fusion Decoder (FFD)를 도입하여 성능을 극대화합니다. 이를 통해 SAM의 강력한 기능을 더 잘 활용할 수 있도록 설계되었습니다.

- **Technical Details**: TS-SAM의 주요 기술적 구성은 다음과 같습니다: 1) Convolutional Side Adapter (CSA)는 SAM의 백본 네트워크에서 추출된 강력한 기능을 종합적으로 통합하여 훈련합니다. 2) Multi-scale Refinement Module (MRM)은 다양한 해상도의 계층적 기능을 사용하여 더 정밀한 위치 정보를 추출합니다. 3) Feature Fusion Decoder (FFD)는 다양한 스케일의 기능을 통합하여 세밀한 세그멘테이션 결과를 생성합니다. 이를 통해 TS-SAM은 상세한 기능을 유지하면서도 가벼운 모델을 구현합니다.

- **Performance Highlights**: TS-SAM은 3가지 작업과 10개의 공공 데이터셋에서 기존의 SAM-Adapter와 SSOM보다 뛰어난 성능을 발휘하며, SOTA(domain-specific models)와 경쟁력 있는 성능을 보입니다. 예시로, COD10K 데이터셋에서 TS-SAM의 성능이 탁월하게 나타났으며, 전체 모델 파라미터의 4.4%에 해당하는 29.44M의 훈련 가능한 파라미터만을 필요로 합니다.



### A Deep CNN Model for Ringing Effect Attenuation of Vibroseis Data (https://arxiv.org/abs/2408.01831)
- **What's New**: 탐사 지구물리학에서 중대한 문제 중 하나인 진동자 데이터의 'Ringing effect'(울림 효과)를 해결하기 위해, 새로운 딥 컨볼루션 신경망(CNN) 기반의 디링잉(deringing) 모델이 제안되었습니다. 이 모델은 진동자 데이터를 처리하여 울림 효과를 줄이고, 진동자 데이터의 주파수 대역폭을 확장하는 것을 목표로 합니다.

- **Technical Details**: 이 모델은 엔드-투-엔드(end-to-end) 훈련 전략을 사용하여 직접적으로 디링잉된 데이터를 얻으며, 스킵 연결(skip connections)을 통해 모델 훈련 과정을 개선하고 진동자 데이터의 세부 사항을 보존합니다. 실제 진동자 디링잉 작업을 위해 실제 진동자 데이터에서 합성한 훈련 데이터와 대응 레이블을 사용하여 딥 CNN 모델을 훈련시킵니다.

- **Performance Highlights**: 실험은 합성 데이터와 실제 진동자 데이터 모두에서 진행되었습니다. 실험 결과, 딥 CNN 모델이 울림 효과를 효과적으로 저감시키고, 진동자 데이터의 주파수 대역폭을 확장할 수 있음을 보여줍니다. 또한, STA/LTA 비율 방법을 사용한 첫 번째 파열(picking)에서도 딥 CNN 모델을 사용한 디링잉된 진동자 데이터에서 개선이 나타났습니다.



### ST-SACLF: Style Transfer Informed Self-Attention Classifier for Bias-Aware Painting Classification (https://arxiv.org/abs/2408.01827)
- **What's New**: 새로운 회화 분류 모델을 소개합니다. Style Transfer와 Adaptive Instance Normalization (AdaIN) 기법을 사용하여 다양한 스타일 간 격차를 해소하고, 특징 맵 적응 공간 주의 모듈(feature-map adaptive spatial attention modules)로 예술적 디테일 이해를 향상시킵니다. 또한, 불균형한 클래스 대표성을 동적으로 조정하는 방법을 사용하여 데이터 편향 문제를 해결합니다.

- **Technical Details**: 이 모델은 ResNet-50 백본을 사용하여 40개의 에폭 동안 87.24%의 정확도를 달성합니다. 우리는 두 단계로 모델을 최적화합니다. 첫 번째 단계는 하이퍼파라미터 그리드 검색과 베이지안 검색을 통한 초기 탐색이며, 두 번째 단계는 모델 파라미터 세트를 점진적으로 조정하는 것입니다. 또한, 질적 및 양적 실험을 통해 다양한 증강 비율의 영향을 평가합니다.

- **Performance Highlights**: 우리의 시스템은 87.24%의 정확도를 달성하며, 이는 LOOK 모델의 89.04% 정확도에 필적하면서도 더 적은 파라미터 요구 사항과 짧은 훈련 시간으로 실용적인 효율성을 향상시킵니다. 우리 모델은 Hyperparameter 탐색과 fine-tuning 전략을 통해 성능을 크게 개선합니다.



### GLDiTalker: Speech-Driven 3D Facial Animation with Graph Latent Diffusion Transformer (https://arxiv.org/abs/2408.01826)
Comments:
          9 pages, 5 figures

- **What's New**: 3D 음성 구동 얼굴 애니메이션 생성에 있어서 다양한 얼굴 애니메이션을 생성하기 위해 GLDiTalker를 제안합니다. 이 모델은 음성-얼굴 동작 간의 불확실성을 줄이면서 비결정론적인 얼굴 단서를 증가시키기 위해 운동 우선 순위와 일부 확률성을 도입합니다.

- **Technical Details**: GLDiTalker는 2단계 접근 방식을 사용합니다. 첫 번째 단계에서는 VQ-VAE (Vector Quantized Variational AutoEncoder)를 사용하여 얼굴 움직임 메쉬 시퀀스를 잠재 공간으로 변환합니다. 두 번째 단계에서는 잠재 얼굴 움직임 특징에 잡음을 추가하고 제거하는 확산 모델을 도입하여 비결정론적이고 다양한 얼굴 움직임을 생성합니다. 또한 Spatial Pyramidal SpiralConv Encoder를 설계하여 다중 스케일 특징을 추출하고, 다양한 수준의 공간 정보를 통합합니다.

- **Performance Highlights**: GLDiTalker는 기존의 최첨단 방법을 능가하여 VOCASET와 BIWI 데이터셋에서 최고의 성능을 달성했습니다. 이 모델은 높은 품질과 다양한 얼굴 애니메이션을 생성할 수 있습니다.



### SkyDiffusion: Street-to-Satellite Image Synthesis with Diffusion Models and BEV Paradigm (https://arxiv.org/abs/2408.01812)
Comments:
          12 pages, 8 figures

- **What's New**: SkyDiffusion은 도로의 스트리트 뷰(street-view) 이미지를 위성 이미지(satellite image)로 변환하는 새로운 생성 방법입니다. 이 접근법은 확산 모델(diffusion model)과 곡면 조감도(Bird's Eye View, BEV) 패러다임을 활용합니다. 

- **Technical Details**: SkyDiffusion은 먼저 스트리트 뷰 이미지를 위성 뷰로 변형하는 Curved-BEV 방법을 설계하여 도메인 정렬 문제를 해결합니다. 이 방법은 여러 스트리트 뷰 이미지를 하나의 위성 이미지로 결합하는 'Multi-to-One' 매핑 전략을 포함하여 도시 환경의 밀집된 영역에서 발생하는 차폐 문제를 극복합니다. BEV-제어 확산 모델(BEV-controlled diffusion model)을 설계하여 스트리트 뷰의 내용과 일치하는 위성 이미지를 생성합니다. 또한 조명 조건을 최적화하기 위해 참조 위성 이미지를 사용하는 조명 조작 모듈을 통합했습니다.

- **Performance Highlights**: 실험 결과, SkyDiffusion은 교외(CVUSA & CVACT) 및 도시(VIGOR-Chicago) 크로스뷰 데이터셋에서 최신 방법들보다 성능이 뛰어났습니다. 평균 SSIM가 14.5% 증가하고, FID가 29.6% 감소하며 현실적이고 콘텐츠가 일치하는 위성 이미지를 생성하는 데 성공했습니다.



### MiniCPM-V: A GPT-4V Level MLLM on Your Phon (https://arxiv.org/abs/2408.01800)
Comments:
          preprint

- **What's New**: MiniCPM-V는 효율적인 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs) 시리즈로, 모바일 디바이스와 같은 엔드 사이드에서 배포 가능한 모델을 소개합니다. 최신 버전인 MiniCPM-Llama3-V 2.5는 여러 벤치마크에서 GPT-4V-1106, Gemini Pro, 그리고 Claude 3보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: MiniCPM-V는 최신 MLLM 기술을 아키텍처, 사전 학습, 정렬(Alignment)에 통합하여 효율성과 성능 간의 균형을 이루었습니다. 특히 주목할 만한 점으로는 높은 성능의 OCR 기능, 1.8M 픽셀 고해상도 이미지 인식, 낮은 헛소리률, 30개 이상의 언어를 지원하는 다국어 기능, 모바일 폰에서도 효율적인 배포가 가능합니다.

- **Performance Highlights**: MiniCPM-Llama3-V 2.5는 OpenCompass 평가에서 GPT-4V, Gemini Pro, Claude 3를 능가했으며, OCRBench에서도 GPT-4V와 Qwen-VL-Max보다 뛰어난 성능을 기록했습니다. 또한, 신뢰할 수 있는 행동을 보이며, 객체 인식 테스트(Object HalBench)에서 낮은 헛소리률을 보여줍니다. 해당 모델은 고품질 다국어 멀티모달 지시 튜닝을 통합하여 30개 이상의 언어를 지원합니다.



### STDA: Spatio-Temporal Dual-Encoder Network Incorporating Driver Attention to Predict Driver Behaviors Under Safety-Critical Scenarios (https://arxiv.org/abs/2408.01774)
- **What's New**: 자율 주행 차량의 행동 예측 정확성을 향상시키기 위해 안전-중요(safety-critical) 시나리오에 최적화된 STDA(Spatio-Temporal Dual-Encoder) 네트워크가 개발되었습니다. 이 네트워크는 운전자의 주의를 반영하여 중요한 위치를 빠르게 식별하도록 설계되었습니다.

- **Technical Details**: STDA는 네 개의 주요 모듈로 구성됩니다: 1) 운전자 주의 예측 모듈, 2) 운전자 주의와 원시 이미지 간의 특징을 융합하는 융합 모듈, 3) 동적 장면 해석 능력을 강화하기 위한 임시 인코더 모듈, 4) 행동 예측 모듈. 이 네트워크를 통해 동적 장면에서의 해석 능력과 행동 예측 정확성을 높였습니다.

- **Performance Highlights**: 실험 결과, 운전자 주의를 통합하고 임시 인코더 모듈을 채택한 STDA는 G-mean을 0.659에서 0.719로 향상시켰습니다. 또한, 제안된 모듈은 강력한 일반화 능력을 보여주었으며 다른 주류 모델들과도 원활히 통합될 수 있음을 입증했습니다.



### MultiFuser: Multimodal Fusion Transformer for Enhanced Driver Action Recognition (https://arxiv.org/abs/2408.01766)
- **What's New**: 이번 연구에서는 운전자 행동 인식을 위한 새로운 멀티모달 융합 트랜스포머인 MultiFuser를 제안합니다. MultiFuser는 자동차 객실 내부의 멀티모달 비디오 간 교차 모달 상호 작용을 식별하고, 적응적으로 다양한 모달리티를 통합하여 향상된 표현을 제공합니다.

- **Technical Details**: MultiFuser는 Bi-decomposed Modules의 층으로 구성되어 있으며, 모달리티별 전문성과 패치 간 적응형 융합을 통해 공간-시간적 특징을 모델링합니다. Modal Expertise ViT 블록은 모달리티별 특징을 추출하며, Patch-wise Adaptive Fusion 블록은 효율적인 교차모달 융합을 수행합니다. 이 모듈들은 Drive&Act 데이터셋을 활용한 실험에서 그 효과를 입증하였습니다.

- **Performance Highlights**: 제안된 MultiFuser 모델은 Drive&Act 데이터셋에서 최첨단 방법들보다 뛰어난 성능을 발휘하며, 멀티모달 비디오 입력을 통합하는 효과를 보여주었습니다.



### Advancing Green AI: Efficient and Accurate Lightweight CNNs for Rice Leaf Disease Identification (https://arxiv.org/abs/2408.01752)
- **What's New**: 이번 연구에서는 쌀 잎 질병을 분류하기 위해 실제 모바일 환경에서 활용할 수 있는 ShuffleNet, MobileNetV2, EfficientNet-B0의 세 가지 CNN 아키텍처를 탐구했습니다. 이 모델들은 비교적 적은 연산 자원과 메모리를 요구하기 때문에 모바일 기기와의 호환성이 높습니다.

- **Technical Details**: 모델의 성능을 향상시키기 위해 두 개의 완전 연결층(fully connected layers)과 드롭아웃 레이어(dropout layer)를 추가하였으며, 과적합(overfitting)을 방지하기 위해 early stop 기법을 사용했습니다. 연구 결과, EfficientNet-B0 모델이 가장 높은 99.8%의 정확도를 달성했습니다. 반면, MobileNetV2와 ShuffleNet은 각각 84.21%와 66.51%의 정확도를 기록했습니다.

- **Performance Highlights**: EfficientNet-B0 모델이 제안된 레이어 구성과 early stop 기법을 결합할 경우 높은 정확성을 보이는 것으로 나타났습니다. EfficientNet-B0의 99.8%라는 놀라운 성능은 쌀 잎 질병 분류에서의 높은 가능성을 시사합니다.



### Domain penalisation for improved Out-of-Distribution Generalisation (https://arxiv.org/abs/2408.01746)
- **What's New**: 이 논문은 객체 탐지에 사용하는 도메인 일반화(DG) 방법을 제안합니다. 특히 여러 출처 도메인에서 학습한 모델이 완전히 새로운 테스트 도메인에서도 잘 작동하게 하기 위해 도메인 패널티(Domain Penalisation, DP) 프레임워크를 도입하였습니다. 이 프레임워크는 각 도메인에 패널티 가중치를 부여하고, 이는 탐지 네트워크의 성능에 따라 업데이트됩니다. 이 접근 방식은 더 많은 주의가 필요한 도메인에 우선 순위를 부여함으로써 학습 과정을 효과적으로 균형 있게 만듭니다.

- **Technical Details**: 제안된 접근 방식은 Global Wheat Head Detection (GWHD) 2021 데이터셋을 사용하여 평가되었으며, WiLDS 벤치마크와 비교되었습니다. ERM, GroupDRO와 같은 기존의 손실 함수 기반 접근 방식과 비교하여, 제안된 DP 프레임워크는 각 도메인에 가중치를 할당하여 도메인 간 손실의 합을 최소화하는 방식으로 동작합니다. FasterRCNN 및 FCOS와 같은 표준 탐지기에서 제안된 방법의 성능을 비교하였으며, 기존 방법들보다 더 나은 성능을 확인했습니다.

- **Performance Highlights**: 제안된 접근 방식은 FasterRCNN 기반에서 검증 세트에서 0.3%, 테스트 OOD 세트에서 0.5%의 정확도를 향상시켰습니다. 또한 FCOS 탐지기에서도 검증 세트에서 1.3%, 테스트 세트에서 1.4%의 성능 개선을 보였습니다. 이는 도메인 패널티 기반 성능이 객체 탐지 모델의 일반화 능력을 향상시키는데 큰 잠재력을 가짐을 강조합니다.



### LAM3D: Leveraging Attention for Monocular 3D Object Detection (https://arxiv.org/abs/2408.01739)
Comments:
          6 pages. Accepted to MMSP 2024

- **What's New**: 새로운 연구 논문 'LAM3D'는 자가-어텐션 메커니즘(self-attention mechanism)을 활용한 단안 3D 물체 검출(Monocular 3D Object Detection) 프레임워크를 제안합니다. 이 방법은 Pyramid Vision Transformer v2 (PVTv2)를 특징 추출 백본(feature extraction backbone)과 2D/3D 검출 기계로 사용하여 개발되었습니다. KITTI 3D Object Detection Benchmark에서 평가한 결과, 자율 주행 분야에서 기존 방법을 능가하는 결과를 보였습니다.

- **Technical Details**: LAM3D는 비전 트랜스포머(Vision Transformer) 기반의 아키텍처를 사용하여 단안 3D 물체 검출 문제를 해결합니다. 기존의 컨볼루션 신경망(Convolutional Neural Networks, CNNs)이 제한된 수용 영역(receptive field) 때문에 장거리 종속성(long-range dependencies)과 맥락적 정보를 포착하는 데 어려움을 겪는 문제를 해결하기 위해 트랜스포머 아키텍처를 채택하였습니다. PVTv2 백본은 다중 스케일 특징 맵(multi-scale feature maps)을 생성하며, 자가-어텐션 메커니즘을 통해 보다 포괄적인 장면 이해를 제공합니다.

- **Performance Highlights**: LAM3D는 KITTI 3D Object Detection Benchmark에서 기존 기술보다 뛰어난 성능을 입증하였습니다. 특히 자가-어텐션 메커니즘을 사용함으로써, 동일한 아키텍처를 사용하지만 자가-어텐션을 사용하지 않는 경우보다 높은 정확도와 강건성을 보여주었습니다.



### Landmark-guided Diffusion Model for High-fidelity and Temporally Coherent Talking Head Generation (https://arxiv.org/abs/2408.01732)
- **What's New**: 이번 아카이브(arxiv) 논문은 음성 기반 입 모양 맞춤형 얼굴 생성 기술을 발전시키기 위해 두 단계로 이루어진 확산 모형(diffusion-based model)을 소개합니다. 기존의 GAN 기반 모델이 입 모양의 동기화를 강조하면서도 프레임의 시각적 품질을 간과하고, 확산 기반 모델이 고품질 프레임을 생성하지만 입 모양 일치를 놓쳐 불안정한 입 모양 움직임을 초래한다는 문제를 해결하기 위해 모델을 제안했습니다.

- **Technical Details**: 새로운 두 단계 확산 기반 모델은 다음과 같습니다. 최초 단계에서는 음성을 기반으로 동기화된 얼굴 핵심 점(facial landmarks)을 생성합니다. 두 번째 단계에서는 생성된 얼굴 핵심 점을 노이즈 제거 과정에서 조건으로 사용하여 입 모양 흔들림 문제를 최적화하고 고품질, 잘 동기화된, 시간적으로 일관된 얼굴 영상을 생성합니다. 음성 구간과 원래의 얼굴 핵심 점을 사용하여 순차적으로 얼굴 핵심 점을 생성하는 Landmark Generation Network를 통해 이 과정이 이루어집니다.

- **Performance Highlights**: 광범위한 실험을 통해 이 모델이 최고의 성능을 발휘함을 입증했습니다. 특히, 시각적 품질과 함께 시간적 일관성을 크게 향상시켰습니다.



### Survey on Emotion Recognition through Posture Detection and the possibility of its application in Virtual Reality (https://arxiv.org/abs/2408.01728)
- **What's New**: 본 논문에서는 감정 인식(Emotional recognition)에 포즈 추정(pose estimation) 기법을 사용하여 다양한 기술과의 통합 가능성을 탐구하였습니다. 이는 평범한 카메라, 깊이 카메라(depth cameras) 등을 이용하거나, 가상 현실(VR)과 같은 새로운 형태의 입력(이미지, 비디오, 3차원 벡터 공간에서의 포즈 포함)을 포괄합니다.

- **Technical Details**: 19개의 연구 논문을 선정된 저널과 데이터베이스에서 수집하여 이들의 방법론(methodology), 분류 알고리즘(classification algorithm), 사용된 데이터셋(datasets)을 중심으로 감정 인식 및 포즈 추정 관련 내용을 분석하였습니다. 이 논문들은 다양한 입력 방법을 사용하여 실시간(real-time)으로 감정을 인식하려고 시도했습니다.

- **Performance Highlights**: 감정 인식의 정확도(accuracy)를 기준으로 벤치마킹을 수행하였으며, 멀티모달 접근법(multimodal approaches)이 전반적으로 가장 높은 정확도를 보였습니다. 또한, 본 연구 분야의 발전을 위해 미래에 고려해야 할 사항들도 논의하였습니다.



### A Novel Evaluation Framework for Image2Text Generation (https://arxiv.org/abs/2408.01723)
Comments:
          The paper has been accepted for presentation at the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, specifically in the Large Language Model for Evaluation in IR (LLM4Eval) Workshop in 2024

- **What's New**: 자동 생성된 이미지 설명의 품질을 평가하는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 GPT-4나 Gemini와 같은 최신 대형 언어 모델(LLM)을 사용하여 이미지 설명을 기반으로 새로운 이미지를 생성합니다. 생성된 이미지와 원본 이미지의 유사성을 측정하여 이미지 설명의 정확도를 평가합니다. 기존의 BLEU, ROUGE, METEOR, CIDEr 등의 자동화된 메트릭이 사람의 판단과 약한 상관관계를 가지는 문제를 해결할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 이미지 캡셔닝 모델의 출력인 텍스트 설명을 LLM을 통해 이미지로 변환하는 과정을 포함합니다. 변환된 이미지와 원본 이미지의 유사도를 코사인 유사도(cosine similarity) 메트릭으로 측정합니다. 사람의 주석이 필요하지 않으며 이미지 설명의 품질을 평가할 수 있습니다. 이 프레임워크는 최신 LLM 기술을 바탕으로 설계되었으며, 이를 통해 텍스트 설명이 원본 이미지를 얼마나 잘 재현할 수 있는지 평가합니다.

- **Performance Highlights**: 제안된 프레임워크는 사람의 평가와 높은 상관관계를 가지며, 효과적으로 이미지 캡셔닝 모델의 성능을 평가할 수 있습니다. 인간 주석이 필요하지 않기 때문에 비용 효율적이고 시간 소모가 적습니다. 기존의 BLEU, ROUGE, METEOR, CIDEr 메트릭보다 높은 정확도를 보이며, 다양한 이미지 캡셔닝 데이터셋을 통해 종합적으로 검증되었습니다.



### A General Ambiguity Model for Binary Edge Images with Edge Tracing and its Implementation (https://arxiv.org/abs/2408.01712)
Comments:
          14 pages

- **What's New**: 이번 연구는 교차로(인터섹션), 접합부(정션) 및 이진 엣지 이미지에서의 구조적 모호성을 처리할 수 있는 범용적이고 직관적인 모델을 제안합니다. 이 모델은 엣지 추적(edge tracing) 기법과 결합되며, 이것은 연속된 픽셀들로 이루어진 엣지의 순서 배열을 의미합니다. 모델은 피겨-그라운드 세분화(figure-ground segmentation), 객체 인식(object recognition), 및 위상 분석(topological analysis) 등 다양한 과제에 유연한 전처리 방법을 제공합니다.

- **Technical Details**: 연구의 주요 목표는 단순한 원칙들을 사용하여 직관적으로 설명할 수 있는 결과를 도출하는 것입니다. 이는 접합부에서의 모호한 엣지 연결을 해결하는 등의 후속 처리 단계를 구현하는 데 도움이 됩니다. 증가된 엣지 맵(augmented edge map)을 사용하여 빠른 로컬 검색 작업으로 인접한 엣지에 직접 액세스할 수 있습니다. 재귀(recursive) 방식으로 엣지 추적을 수행하여 프로그래밍 코드를 간결하게 유지합니다. 논문에서는 알고리즘을 의사코드(pseudocode)로 설명하고, 관련 방법들과의 비교를 통해 단순한 모듈형 후처리 단계로 최적화하는 방법을 보여줍니다.

- **Performance Highlights**: 모든 데이터 구조를 포함하는 전체 알고리즘은 50줄 미만의 의사코드로 구현이 가능하며, C++ 언어로 구현된 예제도 제공합니다.



### AVESFormer: Efficient Transformer Design for Real-Time Audio-Visual Segmentation (https://arxiv.org/abs/2408.01708)
- **What's New**: 최신 Transformer 기반 모델들이 오디오-비주얼 세그멘테이션(AVS) 작업에서 놀라운 성능을 보여주지만, 높은 계산 비용으로 인해 실시간 추론이 어렵습니다. 이를 해결하기 위해, 최초의 실시간 오디오-비주얼 효율적 세그멘테이션 Transformer인 AVESFormer를 소개합니다. 이 모델은 효율적인 프롬프트 쿼리 생성기를 사용하여 교차 주의(attention) 문제를 해결하고, ELF 디코더를 적용해 계산 부담을 줄입니다.

- **Technical Details**: AVESFormer는 두 가지 주요 문제를 해결합니다: 1) 주의 산만(attention dissipation) 문제, 이는 소프트맥스(Softmax) 함수로 인해 주의 가중치가 제한된 프레임 내에서 과도하게 집중되는 문제입니다. 2) 비효율적 디코더, 이는 초기 단계에서 좁은 포커스 패턴을 가지며 계산 부담이 큰 문제입니다. AVESFormer는 프롬프트 쿼리 생성기를 통해 교차 주의의 동작을 수정하고, ELF 디코더를 도입하여 국소 특징을 더 적은 계산 비용으로 효과적으로 처리합니다.

- **Performance Highlights**: AVESFormer는 AVSBench에서 시험된 결과, 79.9%의 S4, 57.9%의 MS3 및 31.2%의 AVSS 점수를 기록하며, 이전의 최고 성능을 능가했습니다. 모델은 더 적은 파라미터로 더 빠른 속도를 자랑하며, AVSegFormer보다 S4에서 3.4%p, MS3에서 8.4%p, AVSS에서 6.3%p 더 높은 성능을 나타냈습니다.



### Downstream Transfer Attack: Adversarial Attacks on Downstream Models with Pre-trained Vision Transformers (https://arxiv.org/abs/2408.01705)
- **What's New**: 최근 Vision Transformers(ViTs)와 자가지도학습(Self-Supervised Learning, SSL) 기법의 발전으로, 사전 학습된 대형 ViTs는 컴퓨터 비전 응용 프로그램의 새로운 기반 모델이 되고 있습니다. 본 논문에서는 이러한 대형 ViTs의 적대적 취약성(adversarial vulnerability)이 하위 작업으로 전이되는지에 대해 연구하였습니다. 우리는 샘플 단위의 전이 공격(sample-wise transfer attacks)에 주목하여 새로운 공격 방법인 Downstream Transfer Attack (DTA)을 제안합니다. DTA는 사전 학습된 ViT 모델을 활용하여 적대적 예제를 생성하고, 이를 하위 데이터셋으로 미세 조정된 모델에 적용하여 공격합니다.

- **Technical Details**: DTA는 주어진 테스트 이미지에 대해 사전 학습된 ViT 모델을 사용해 적대적 예제를 생성하고 이를 사용해 하위 데이터셋에서 미세 조정된 모델을 공격합니다. 공격 중에 DTA는 사전에 학습된 모델의 취약한 층을 코사인 유사도 손실(cosine similarity loss)을 통해 식별하고 활용합니다. 이는 전이 가능한 공격을 만들기 위해 매우 중요한 과정을 포함합니다. DTA는 처음에는 모델의 얕은 층을 목표로 하며, 초기 시도가 실패하면 중간 층을 탐색하여 가장 취약한 층을 찾아내어 최종 공격을 수행합니다.

- **Performance Highlights**: 사전 학습된 ViTs에 대해 3가지 다른 사전 학습 방법과 3가지 미세 조정 방식, 그리고 10가지 다양한 하위 데이터셋에서 광범위한 실험을 수행한 결과, DTA는 평균 공격 성공률(ASR)이 90%를 초과하며 기존 방법들을 크게 초과하는 성능을 보여주었습니다. 또한 DTA를 사용한 적대적 훈련이 모델의 다양한 하위 전이 공격에 대한 강건성을 크게 향상시킬 수 있음을 확인하였습니다.



### Signal-SGN: A Spiking Graph Convolutional Network for Skeletal Action Recognition via Learning Temporal-Frequency Dynamics (https://arxiv.org/abs/2408.01701)
- **What's New**: 본 논문에서는 그래프 컨볼루션 네트워크(GCN) 기반의 스켈레톤 정보로 행동 인식을 수행할 때 복잡성과 에너지 소비 문제를 해결하기 위해 신경 스파이킹 네트워크(SNN)를 사용한 새로운 접근법을 제안합니다. Signal-SGN(Spiking Graph Convolutional Network)을 통해 시퀀스의 시간적 특성을 최대로 활용하고, 기능을 이산 스토캐스틱 신호로 처리함으로써 저장 및 계산 비용을 절감합니다.

- **Technical Details**: Signal-SGN의 핵심은 1D Spiking Graph Convolutional Network(1D-SGN)와 Frequency Spiking Convolutional Network(FSN)로 구성됩니다. 1D-SGN은 단일 프레임 그래프에서 특징을 추출하며, FSN은 빠른 푸리에 변환(FFT)을 사용하여 시간-주파수 기반 특성을 추출합니다. 추가적으로 다중 스케일 웨이브렛 변환(MWTF) 모듈과 플러그인 가능한 시간-주파수 공간 의미 특징 추출 모듈(TFSM)을 도입하여 모델의 분류 성능을 향상시켰습니다.

- **Performance Highlights**: NTU RGB+D, NTU RGB+D 120, NW-UCLA 데이터셋을 이용한 실험 결과, 제안된 Signal-SGN 모델은 기존 SNN 기반 방법보다 높은 정확도를 보였으며, GCN 기반 방법과 비교해도 경쟁력 있는 정확도를 유지하면서도 계산 및 저장 비용을 크게 줄였습니다.



### Bayesian Active Learning for Semantic Segmentation (https://arxiv.org/abs/2408.01694)
- **What's New**: 이 논문은 **Bayesian active learning** 프레임워크를 제안하여 sparse pixel-level annotation을 활용한 **semantic segmentation** 모델을 학습합니다. 이 모델은 **Balanced Entropy (BalEnt)** 기반의 픽셀 수준 불확실성 측정 방식을 사용하여, Cityscapes, Camvid, ADE20K, VOC2012 벤치마크 데이터셋에서 소수의 라벨링된 픽셀만으로도 완전 감독 학습(supervised)의 수준에 도달합니다.

- **Technical Details**: BalEnt는 모델이 예측한 확률 분포와 픽셀 라벨 간의 정보를 캡처하며, 이는 연속적인 스케일링이 가능하고 폐쇄적 분석 형식을 가지며 다른 픽셀과의 관계 계산 없이 개별 픽셀 단위로 계산 가능합니다. 제안된 프레임워크는 **epistemic** 및 **aleatoric** 불확실성을 결합하여 픽셀의 불확실성을 평가하고, **Bayesian AL** 원칙을 확장하여 다양한 데이터셋에서 'harder'한 픽셀을 선별하여 모델의 불확실성을 줄이는 기능을 마련합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 몇 사이클의 학습 후 소수의 픽셀(이미지당 약 5개)의 라벨만으로도 기존의 최첨단 AL 모델을 크게 능가하며, 지도 학습 수준의 mIoU에 도달했습니다. BalEntAcq AL 모델은 다양한 데이터셋과 백본에서도 높은 성능을 발휘하며, 추가적인 초매개변수 튜닝 없이 높은 정확도에 도달합니다.



### A Comparative Analysis of CNN-based Deep Learning Models for Landslide Detection (https://arxiv.org/abs/2408.01692)
Comments:
          Paper Accepted for presentation at IEEE 2024 Asian Conference on Intelligent Technologies (ACOIT)

- **What's New**: 이번 연구는 CNN(Convolutional Neural Networks) 기반 모델들이 산사태 탐지에 있어서 기존 알고리즘보다 우수한 성능을 보이는 것을 입증하고자 합니다. 특히 ResNet50 백본 인코더를 사용해 U-Net, LinkNet, PSPNet, FPN 네 가지 모델을 비교 분석하였습니다.

- **Technical Details**: 연구에서는 바이에(중국)의 위성 이미지를 사용하여 네 가지 전통적 의미 분할(semantic segmentation) 모델(U-Net, LinkNet, PSPNet, FPN)을 비교하였으며, ResNet50 백본 인코더를 활용해 이를 구현했습니다. 하이퍼파라미터(hyperparameters) 튜닝으로 학습률(learning rates), 배치 크기(batch sizes), 정규화 기법들을 실험하였고, 각 모델의 혼동 행렬(confusion matrix)을 계산하여 정밀도(precision), 재현률(recall), F1 스코어(f1-score) 등을 평가하였습니다.

- **Performance Highlights**: 실험 결과, LinkNet 모델이 97.49%의 정확도(Accuracy)와 85.7%의 F1 스코어(84.49% 정밀도, 87.07% 재현률)로 가장 우수한 성능을 보였습니다. 각 픽셀 단위의 혼동 행렬 결과와 각 모델의 학습에 소요된 시간도 종합적으로 비교되었습니다.



### IDNet: A Novel Dataset for Identity Document Analysis and Fraud Detection (https://arxiv.org/abs/2408.01690)
Comments:
          40 pages

- **What's New**: 새로운 벤치마크 데이터셋 IDNet이 소개되었습니다. IDNet은 837,060장의 합성 신분증 이미지를 포함하며 약 490 기가바이트의 데이터를 제공합니다. 이 데이터셋은 10개의 미국 주와 10개의 유럽 국가에서 온 20가지 유형의 신분증으로 분류됩니다. 이 데이터셋은 개인정보 보호를 유지하면서 현실적인 사기 탐지 방법을 훈련시키는 데 도움을 주기 위해 설계되었습니다.

- **Technical Details**: IDNet 데이터셋은 Stable Diffusion 2.0 및 ChatGPT-3.5-turbo와 같은 최신 AI 기술을 사용하여 생성되었습니다. 각각의 신분증 샘플은 6가지 다른 사기 패턴으로 변조된 버전을 포함하고 있으며, 이는 Crop and Move, Impaint and Rewrite, 얼굴 변형(face morphing), 초상화 교체, 텍스트 필드의 직접 변경 등입니다. 이 파이프라인은 인텔 Xeon Gold 6226 24코어 CPU 프로세서 두 개, Nvidia GeForce 2080 Ti GPU 네 개, 196GB 메모리를 갖춘 서버에서 동작하며, 문서 하나를 생성하는 데 약 0.14초가 소요됩니다.

- **Performance Highlights**: IDNet은 다른 공개 데이터셋과 달리 더 많은 샘플과 다양한 사기 패턴을 포함하고 있습니다. 각 문서는 5,979개의 고유한 합성 초상화를 포함하고 있으며, 전체 데이터셋은 41,853개의 문서 샘플을 포함하고 있습니다. 이러한 넓은 범위의 데이터는 프라이버시를 고려한 사기 탐지 연구에 있어 중요한 발전입니다. 또한, 데이터셋은 실제 문서와 유사한 정확도와 다양한 메타데이터를 갖추고 있어 연구에 유용합니다.



### SiamMo: Siamese Motion-Centric 3D Object Tracking (https://arxiv.org/abs/2408.01688)
- **What's New**: 새로운 연구로 발표된 SiamMo는 3D 단일 객체 추적(single object tracking)을 위한 새로운 방식으로, 기존의 시아메즈 매칭 기반 패러다임(Siamese matching-based paradigm)보다 텍스처가 없는 불완전한 LiDAR 포인트 클라우드에 더 효과적입니다. 시아메즈 모션 중심 추적(Siamese motion-centric tracking) 방식을 도입함으로써 전통적인 단일 스트림 아키텍처(single-stream architecture)와 달리 시아메즈 피처 추출을 통해 성능을 크게 향상시켰습니다. 새롭게 도입된 Spatio-Temporal Feature Aggregation 모듈과 Box-aware Feature Encoding 모듈은 모션 정보와 객체 크기 사전 지식을 효과적으로 통합하여 추적 정확도를 높입니다.

- **Technical Details**: SiamMo는 비균일한 포인트를 정규 복셀로 나누고, 다음으로 동일한 특징 공간으로 임베딩하기 위해 두 개의 가중치 공유 인코더(weight-sharing encoders)를 사용합니다. 이를 통해 피처 추출과 시간 융합을 분리하여 추적 성능을 크게 향상시켰습니다. 또한, 다중 스케일에서 인코딩된 피처를 통합하여 모션 정보를 효과적으로 포착하는 Spatio-Temporal Feature Aggregation (STFA) 모듈과 모션 피처 예측에 객체 크기 사전(box priors)을 주입하는 Box-aware Feature Encoding (BFE) 모듈을 설계하였습니다. 추가적인 세분화(세그멘테이션) 또는 박스 정제 없이도 정확한 로컬리제이션을 일관된 속도로 달성합니다.

- **Performance Highlights**: SiamMo는 KITTI 트래킹 벤치마크에서 90.1%의 정밀도를 기록하면서 새로운 기록을 세우고, 108 FPS의 높은 추론 속도를 유지합니다. NuScenes 데이터셋에서는 이전 모션 중심 추적기인 M2-Track을 성공률에서 11.08% 상회하는 성능을 보였으며, 드문 설정과 방해물(디스트랙터)이 있는 시나리오에서도 뛰어난 견고성을 입증했습니다.



### iControl3D: An Interactive System for Controllable 3D Scene Generation (https://arxiv.org/abs/2408.01678)
Comments:
          Accepted by ACM MM 2024

- **What's New**: iControl3D는 사용자가 정교하게 제어 가능한 3D 씬을 생성하고 렌더링할 수 있게 하는 새로운 인터랙티브 시스템입니다. 이 시스템은 3D 창작 인터페이스를 통해 사용자가 생성 과정을 세밀하게 통제할 수 있게 설계되었습니다. 3D 메시(mesh)를 중간 프록시로 사용하여 개별 2D 확산 생성 이미지를 통합하고, 새로운 깊이 정렬 알고리즘을 통해 부드럽게 병합합니다. 더불어, 원거리 콘텐츠를 효과적으로 관리하기 위해 3D 메시 대신 환경 맵(environment map)을 사용합니다.

- **Technical Details**: iControl3D 시스템은 2D 확산 모델(diffusion model)로 생성한 이미지를 3D 공간으로 변환하고 병합하는 과정을 중심으로 합니다. 먼저 단안 깊이 추정기(monocular depth estimator)를 사용해 이미지의 기하학적 구조를 추정하고 이를 3D 메시로 변환합니다. 새로운 관점에서 2D 확산 모델을 사용해 구멍을 메꾸고 새로운 콘텐츠를 채워넣은 후, 경계 인식 깊이 정렬(boundary-aware depth alignment)을 수행해 메시를 통합합니다. 이러한 과정을 반복하여 완성된 3D 구조를 얻습니다. 원거리 콘텐츠는 3D 메시로 처리하기 어려운 깊이 불연속성을 해결하기 위해 별도의 환경 맵으로 모델링합니다. 사용자는 인터페이스를 통해 가상 카메라 위치를 조정하고, 콘텐츠의 크기와 시드를 변경하여 다양한 결과를 생성할 수 있습니다.

- **Performance Highlights**: iControl3D의 주요 강점은 사용자 제어의 정교함과 유연성에 있습니다. 사용자는 카메라의 위치를 자유롭게 조정하고, 개인의 취향에 맞는 비디오 경로를 설정할 수 있습니다. 또한, ControlNet에서 영감을 받은 뉴럴 렌더링 인터페이스를 통해 사용자는 장면의 방사 필드(radiance field)를 생성하고 탐색할 수 있습니다. 이 시스템은 경계 인식 깊이 정렬 덕분에 메시 병합 시의 경계 문제를 원활히 해결하며, 원거리 콘텐츠 관리로 보다 현실적인 야외 씬을 생성할 수 있습니다.



### HIVE: HIerarchical Volume Encoding for Neural Implicit Surface Reconstruction (https://arxiv.org/abs/2408.01677)
Comments:
          Submitted to ICLR 2023

- **What's New**: 이 논문에서는 신경 암시적 표면 재구성(neural implicit surface reconstruction)을 위한 새로운 방법을 소개합니다. 기존 메서드는 MLPs(Multi-Layer Perceptrons)를 통해 3D 장면을 암시적으로 인코딩했지만, 이는 명시적인 3D 구조가 부족한 단점이 있습니다. 이를 개선하기 위해, 명시적인 공간 정보를 통해 체적 인코딩(volume encoding)을 도입하며, 또한 계층적 체적 인코딩(hierarchical volume encoding)을 설계하여 여러 스케일에서 장면 구조를 인코딩합니다. 이 방법은 고해상도 체적이 고주파 지오메트리 세부 정보를 캡처하고, 저해상도 체적이 공간 일관성을 강화하여 모양을 부드럽게 유지합니다. 추가로, 고해상도 체적에서 메모리 소비를 줄이기 위해 희소 구조를 채택하고, 두 개의 정규화 항을 사용하여 결과의 매끄러움을 향상시킵니다.

- **Technical Details**: 체적 인코딩(volume encoding)은 다중 스케일에서 장면 구조를 표현할 수 있는 계층적 체적(hierarchical volume) 구조를 사용합니다. 고해상도 체적은 다양한 3D 포인트에서 공간적으로 변화하는 특징을 학습하여 고주파의 지오메트리 세부정보를 캡처하고, 저해상도 체적은 공간 일관성을 강화하여 모양을 부드럽게 유지합니다. 또한, 메모리 소비를 줄이기 위해 희소 구조(sparse structure)를 도입하여, 고해상도 체적의 인접 위치에 있는 것만 남기고 나머지를 제거합니다. 두 개의 정규화 항(regulatization terms)을 도입하여 재구성된 표면의 매끄러움을 높였습니다.

- **Performance Highlights**: 제안된 계층적 체적 인코딩을 통해 DTU, EPFL 및 BlendedMVS 데이터셋에서 극복적인 성능 향상이 입증되었습니다. NeuS의 오류가 25% 감소하고, VolSDF의 오류가 23% 감소, NeuralWarp의 오류가 10% 감소했습니다. 특히 EPFL 데이터셋에서 NeuralWarp의 오류가 31% 감소하는 성과를 보였습니다.



### SynopGround: A Large-Scale Dataset for Multi-Paragraph Video Grounding from TV Dramas and Synopses (https://arxiv.org/abs/2408.01669)
Comments:
          Accepted to ACM MM 2024. Project page: this https URL

- **What's New**: 새로운 SynopGround 데이터셋은 2800시간 이상의 TV 드라마 비디오와 정확하게 로컬화된 인간 작성 시놉시스를 포함합니다. 이 데이터셋은 기존의 짧은 비디오 및 문장 중심의 데이터셋 한계를 극복하고, 복잡한 이벤트와 추상적인 개념을 더 잘 이해할 수 있는 모델 개발을 목적으로 합니다.

- **Technical Details**: SynopGround 데이터셋에서는 각 비디오의 시놉시스 단락이 언어 질의로 사용되며, 각 단락은 비디오의 정확한 시간 경계로 수동 주석됩니다. Multi-Paragraph Video Grounding (MPVG) 설정에서 다수의 단락과 긴 서사 비디오를 입력으로 받아 각 단락 질의에 해당하는 시간 간격을 찾습니다. 이를 위한 새로운 Local-Global Multimodal Reasoner (LGMR) 모델도 제안되었습니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 LGMR 모델이 기존의 최첨단 성능을 능가한다는 것이 검증되었습니다. SynopGround는 복잡한 비디오-언어 정렬 작업에서 기존 데이터셋과 비교할 수 없는 독특한 장점을 가지고 있습니다.



### Multiple Contexts and Frequencies Aggregation Network forDeepfake Detection (https://arxiv.org/abs/2408.01668)
- **What's New**: MkfaNet이라는 새로운 딥페이크 탐지 네트워크를 소개합니다. 이 네트워크는 스페셜 및 주파수 영역에서 직관적 우선 순위를 학습하도록 설계되었습니다. MkfaNet은 Multi-Kernel Aggregator (MKA)와 Multi-Frequency Aggregator (MFA)라는 두 가지 핵심 모듈을 포함합니다.

- **Technical Details**: MkfaNet의 MKA 모듈은 서로 다른 팽창률의 깊이 분리형 합성곱을 결합하여 모델의 수용 영역을 확장하고 다양한 규모의 특징을 효과적으로 캡처합니다. 이 모듈은 여러 합성곱을 통해 추출된 특징을 스페셜 컨텍스트에 따라 선택하여 실제 얼굴과 가짜 얼굴 사이의 미세한 차이를 모델링합니다. MFA 모듈은 이미지의 DC (직류) 및 HC (고주파) 성분을 분리 처리하여 주파수 정보에 대한 모델의 응답을 최적화합니다.

- **Performance Highlights**: 일곱 가지 인기 있는 딥페이크 탐지 벤치마크에서 MkfaNet 변종이 뛰어난 성능을 발휘했으며, 도메인 내 및 도메인 간 평가에서 모두 높은 효율성으로 우수한 성능을 보였습니다.



### SAT3D: Image-driven Semantic Attribute Transfer in 3D (https://arxiv.org/abs/2408.01664)
- **What's New**: 이번 연구에서는 참조 이미지(reference image)를 활용해 3D에서의 의미론적 속성 전이(Semantic Attribute Transfer)을 수행하는 SAT3D 방법을 제안했습니다. 이는 기존 2D 및 3D 이미지 편집 방법들이 애매한 속성 편집에 머무르는 문제를 해결합니다.

- **Technical Details**: SAT3D는 사전에 훈련된 3D-aware StyleGAN 기반 생성기의 스타일 공간(style space)을 탐색하고, 의미론적 속성과 스타일 코드 채널의 상관관계를 학습합니다. 이를 위해 각 속성을 단어 기반(descriptor groups)의 집합으로 연결하고, CLIP 모델을 활용해 속성을 정량적으로 측정하는 Module(QMM)을 개발했습니다. 그 후, QMM을 속성 손실(attribute losses)에 포함시켜 이미지 사이의 속성 유사도를 계산하고, 대상 속성 전이와 다른 속성의 보존을 지도합니다.

- **Performance Highlights**: SAT3D는 여러 도메인에서의 3D-aware 속성 전이 결과를 제시했으며, 기존의 2D 이미지 편집 방법과 비교하여 효과성과 사용자 맞춤성을 입증했습니다. 특히, 3D 뿐만 아니라 2D StyleGAN 기반 생성기에서도 효과적으로 적용 가능합니다.



### Deep Patch Visual SLAM (https://arxiv.org/abs/2408.01654)
- **What's New**: 최근 비주얼 SLAM (Visual SLAM)에서 딥 네트워크 백본을 사용하는 방법들이 제안되었으나, 이러한 방법들은 실행 비용이 높거나 제로 샷(Zero-shot) 상황에서 잘 일반화되지 않는 단점이 있었습니다. DPV-SLAM은 이러한 문제를 해결하기 위해 단일 GPU에서 고성능을 유지할 수 있게 설계되었습니다. 특히, 다양한 실제 세계 데이터셋에서 1배에서 4배 이상의 프레임 속도를 유지하면서 높은 정확도를 달성했습니다.

- **Technical Details**: DPV-SLAM은 DPVO 시스템을 확장하여 회귀 오류를 줄이기 위한 루프 클로저 메커니즘을 도입했습니다. 여기에는 근접 루프 클로저(proximity loop closure)와 전통적인 클로저 메커니즘이 포함됩니다. 근접 루프 클로저는 카메라 근접성을 이용해 루프를 탐지하며, 이는 단일 GPU에서 프론트엔드와 백엔드를 병렬로 실행할 수 없다는 문제를 해결합니다. 또한 CUDA 가속화 블록 희소 번들 조정(block-sparse bundle adjustment) 구현을 통해 효율적인 전역 최적화를 가능하게 했습니다.

- **Performance Highlights**: DPV-SLAM은 DROID-SLAM 대비 유사한 정확도를 유지하면서도 2.5배 더 빠르게 실행됩니다. EuRoC 데이터셋에서 DPVO에 비해 4배 낮은 오류율을 기록했으며, KITTI 데이터셋에서도 DROID-SLAM과 DPVO를 상회하는 성능을 보였습니다. 메모리 사용량도 매우 효율적이며, 다양한 환경에서 강력한 성능을 유지했습니다. 특히, EuRoC 데이터셋에서는 DROID-SLAM의 백엔드보다 근접 루프 클로저가 훨씬 빠르게 실행되었습니다 (0.1-0.18초 vs 0.5-5초).



### MCPDepth: Omnidirectional Depth Estimation via Stereo Matching from Multi-Cylindrical Panoramas (https://arxiv.org/abs/2408.01653)
- **What's New**: MCPDepth는 여러 원통형 파노라마들 간의 스테레오 매칭(stereo matching)을 통해 전방위 깊이 추정(omnidirectional depth estimation)을 수행하는 이단계(two-stage) 프레임워크를 소개합니다. 이 접근법은 초기 스테레오 매칭에 원통형 파노라마(cylindrical panoramas)를 사용하고 결과를 합성합니다. 원통형 프로젝션(cylindrical projection)의 이점을 입증하며, 표준 네트워크 구성요소만을 사용하여 임베디드 장치로의 배포를 단순화합니다.

- **Technical Details**: MCPDepth는 기존의 방법들과 달리 커스터마이즈드 커널(custom kernels)을 사용하지 않으며, 대신 원통형 프로젝션을 채택하여 왜곡을 줄입니다. 또한, 원통형 파노라마의 수직 축 왜곡을 극복하기 위해 원형 어텐션 모듈(circular attention module)을 사용합니다. 이 모듈은 360°의 수직 시야(Field of View)를 처리해 새로운 기능을 효과적으로 캡쳐합니다. MCPDepth는 이론적 및 실험적 비교를 통해 구형 프로젝션과 원통형 프로젝션의 장단점을 분석합니다.

- **Performance Highlights**: MCPDepth는 Deep360의 실외 합성 데이터셋에서 평균 절대 오차(Mean Absolute Error, MAE)를 18.8% 감소시켰으며, 3D60의 실내 현실 장면 데이터셋에서도 19.9% 감소시켜 최첨단 성능(state-of-the-art)을 달성했습니다. 이러한 성과는 원통형 프로젝션과 원형 어텐션 모듈의 효율성을 강조합니다.



### Leveraging GNSS and Onboard Visual Data from Consumer Vehicles for Robust Road Network Estimation (https://arxiv.org/abs/2408.01640)
Comments:
          This work will be presented at IROS 2024. Supplementary website: this https URL

- **What's New**: 이 논문은 GNSS(GPS) 트레이스와 소비자 차량의 기본 센서를 사용하여 자동으로 고품질의 도로 그래프(road graph)를 생성하는 새로운 방법을 제안합니다. 이 접근법은 Toyota의 2023 Woven by Toyota Invention Award를 수상했습니다.

- **Technical Details**: 이 연구는 GNSS 트레이스와 소비자 차량의 기본 이미지 데이터를 활용하여 도로 중심선의 의미론적 분할(semantic segmentation)을 수행하는 CNN(Convolutional Neural Network)을 사용합니다. 네트워크 출력은 지도 맞춤(map matching)을 통해 정제됩니다. 또한, 도로 중심선을 추정한 후 후처리 단계에서 겹쳐진 도로를 탐지합니다.

- **Performance Highlights**: 실제 소비자 차량을 사용한 평가에서, 제안된 접근법은 단순한 도로 구성에서는 기존 방법과 유사한 성능을 보였으나, 더 복잡한 도로 기하학 및 토폴로지에서는 기존 방법을 상당히 능가했습니다. 특히, 새로운 영역에서도 강력한 일반화 능력을 보여주었고, 교차로와 같은 진짜 도로 교차점과 다리나 지하도 같은 오탐지(false positives)를 효과적으로 구분했습니다.



### JambaTalk: Speech-Driven 3D Talking Head Generation Based on Hybrid Transformer-Mamba Language Mod (https://arxiv.org/abs/2408.01627)
Comments:
          12 pages with 3 figures

- **What's New**: 최근 연구는 대화형 얼굴 생성(talking head generation) 기술을 큰 폭으로 발전시키고 있습니다. 이 논문에서는 Jamba라는 하이브리드 Transformers-Mamba 모델을 사용하여 3D 얼굴을 애니메이션화하는 방법을 제시합니다. Jamba 모델은 Transformer와 Mamba의 장점을 결합하여 효과적인 솔루션을 제공합니다. JambaTalk라는 새로운 프레임워크를 통해 모션의 다양성과 속도를 향상시키는 멀티모달 통합을 구현합니다. 실험 결과, 우리의 방법이 최첨단 모델과 비교해 동등하거나 우수한 성능을 보이는 것으로 드러났습니다.

- **Technical Details**: Jamba는 Transformer와 Mamba의 레이어를 결합하여 두 아키텍처의 강점을 모두 활용합니다. 특정 레이어는 MoE(Mixture of Experts) 기법을 사용하여 모델 용량을 늘리면서도 활성화되는 파라미터 수를 관리합니다. Rotary Positional Embedding(RoPE)와 Grouped Query Attention(GQA) 알고리즘을 적용하여 Transformer 레이어의 성능을 최적화했습니다. 이러한 구조는 24GB GPU에서도 안정적으로 작동하며 긴 맥락 평가(long-context evaluations)에서 높은 처리량과 최소한의 메모리 사용을 자랑합니다.

- **Performance Highlights**: Vocaset 데이터셋을 활용한 종합적인 실험과 분석을 통해 제안된 JambaTalk 모델의 효율성을 입증했습니다. 기존의 2D 대화형 얼굴과 비교해 자연스러운 입술 싱크와 다양한 감정 표현을 구현했으며, 라이팅 조건과 관계없이 높은 영상 품질을 유지했습니다.



### Deep Learning Meets OBIA: Tasks, Challenges, Strategies, and Perspectives (https://arxiv.org/abs/2408.01607)
- **What's New**: 이 논문에서는 원격 센싱(remote sensing)에서 깊이 학습(deep learning)의 가능성을 확장하고자 하는 시도를 다룹니다. 특히 객체 기반 이미지 분석(OBIA: Object-Based Image Analysis)과 깊이 학습의 통합 가능성을 평가합니다. 기존 연구에서는 픽셀-혹은 패치-레벨(patch-level) 응용 프로그램에서의 깊이 학습에 중점을 두고 있었지만, OBIA와의 통합에 대한 심도 있는 분석은 부족했습니다.

- **Technical Details**: 이 리뷰에서는 OBIA의 업무 하위 도메인(task subdomains)을 포괄적으로 검토하고 확장했습니다. 딥러닝의 한계를 극복하기 위한 다섯 가지 주요 전략을 확인하고 요약했습니다. 첫째, 객체 기반으로 데이터를 전처리(pre-processing)하는 방법, 둘째, 데이터 샘플링(sampling) 기술, 셋째, 심층 신경망(deep neural network) 구조의 수정, 넷째, 후처리(post-processing) 기술, 다섯째, 멀티스케일 분석(multiscale analysis)입니다.

- **Performance Highlights**: 논문은 OBIA와 깊이 학습의 통합을 통해 얻을 수 있는 성과와 한계를 명확히 제시합니다. 주요 도전 과제는 비구조화된 객체 데이터의 처리를 요구하며, 이를 해결하기 위한 전략을 통해 성능을 개선할 수 있습니다. 이를 통해 원격 센싱 응용 프로그램에서 깊이 학습의 잠재성을 최대한 활용할 수 있는 방안에 대해 논합니다.



### Deep Learning Approach for Ear Recognition and Longitudinal Evaluation in Children (https://arxiv.org/abs/2408.01588)
Comments:
          Submitted to Biosig 2024

- **What's New**: 이 연구는 4세에서 14세 사이의 아동을 대상으로 2.5년 동안 축적된 종단 데이터를 활용하여 아동 귀 인식 성능을 평가합니다. 특히 VGG16과 MobileNet을 결합한 딥러닝 기반 접근 방식을 도입하여 아동과 성인 데이터셋 모두에서 인식 성능을 검토합니다.

- **Technical Details**: 본 연구에서는 귀 영역을 분할하기 위해 Mask R-CNN을 사용하고, 특성 추출에는 MobileNet 모델을 활용합니다. 이어서 이 특성 간의 유클리드 거리를 계산하여 인식 성능을 평가합니다. 주요 네트워크 아키텍처는 Mask R-CNN을 통해 프로파일 이미지에서 귀 영역을 마스킹하고, VGG16과 MobileNet 모델의 특성을 앙상블로 합쳐 단일 특성 벡터를 생성한 후, t-SNE를 적용하여 유클리드 거리를 계산합니다.

- **Performance Highlights**: 아동의 귀는 구조적인 변화를 겪기 때문에 인식이 어려움에도 불구하고, 본 연구의 접근 방식은 높은 정밀도(TAR)와 낮은 오탐률(FAR)을 통해 귀 인식의 정확성을 향상시켰습니다. 특히 8세 이하 어린이와 귀 구조가 안정되는 8세 이상의 어린이 모두를 포함한 데이터셋을 활용하여 종단적 평가를 성취했습니다.



### THOR2: Leveraging Topological Soft Clustering of Color Space for Human-Inspired Object Recognition in Unseen Environments (https://arxiv.org/abs/2408.01579)
- **What's New**: 이 연구에서는 미지의 복잡한 실내 환경에서 시각적 객체 인식 문제를 해결하기 위해 새로운 3D 모양 및 색상 기반 디스크립터, TOPS2와 이를 활용한 인식 프레임워크인 THOR2를 소개합니다. 기존 3D 모양 기반 디스크립터인 THOR에 비해 컬러 정보를 추가하여 객체 인식 정확도를 크게 향상시켰습니다. 이 새로운 프레임워크는 합성 데이터를 활용하여 모델을 학습시켰으며, 다양한 실제 데이터셋에서도 뛰어난 성능을 보였습니다.

- **Technical Details**: THOR2 프레임워크는 기존의 3D 모양 디스크립터(TOPS)를 기반으로 색상 정보가 추가된 TOPS2 디스크립터를 도입했습니다. 이는 Mapper 알고리즘을 통해 얻은 색상 영역을 이용하여 객체의 색상 임베딩을 계산합니다. 해당 색상 영역은 평균적인 인간 색각에서 구분되지 않는 색상을 나타내는 MacAdam 이클립스와 유사합니다. 이러한 색상 영역을 토대로 객체의 3D 모양과 색상을 동시에 표현해 인식 정확도를 높였습니다.

- **Performance Highlights**: THOR2는 OCID와 UW-IS Occluded 데이터셋에서 기존의 THOR, 기본 딥러닝 모델 및 RGB-D 입력을 위한 ViT(Visual Transformer) 모델을 모두 뛰어넘는 성능을 보였습니다. 저비용 로봇 시스템에서 사용될 수 있는 강력한 인식 프레임워크로 자리잡을 가능성이 높습니다.



### Counterfactual Explanations for Medical Image Classification and Regression using Diffusion Autoencoder (https://arxiv.org/abs/2408.01571)
Comments:
          In submission. arXiv admin note: text overlap with arXiv:2303.12031

- **What's New**: 최신 논문에서는 머신러닝 모델의 해석가능성을 높이기 위해 입력 특징의 변화가 결과 예측에 미치는 영향을 보여주는 반사실적 설명(CE; Counterfactual Explanation)을 제시하는 방법을 소개합니다. 기존 CE 방법이 추가 모델을 필요로 하고 이진 반사실적 설명에 한정되어 있는 반면, 새로운 방법은 Diffusion Autoencoder(DAE)의 잠재 공간에서 직접 작동합니다. 이 방법은 모델의 내부 표현을 연속적으로 시각화하여 본질적인 해석가능성을 제공합니다.

- **Technical Details**: 이 논문은 DAE의 풍부한 잠재 공간에서 이미지를 인코딩하여, 레이블된 데이터나 별도의 특징 추출 모델 없이 개념적으로 풍부한 잠재 공간을 생성합니다. 이 잠재 표현은 척추 압박 골절(VCF) 및 당뇨병성 망막병증(DR)과 같은 병리학적 상태의 분류 및 심각도 순서 회귀에 유용합니다. 또한, 이 방법은 선형 모델을 사용하여 순차적인 반사실적 설명을 지원하므로, 모델의 의사결정 과정에 대한 심도 있는 통찰을 제공합니다.

- **Performance Highlights**: 다양한 의료 영상 데이터셋에 대한 실험 결과, 제안된 방법은 해석가능성과 다용성 측면에서 이점을 보여줍니다. DAE의 선형 매니폴드(latent space)는 의미 있는 보간 및 조작을 가능하게 하여 의료 이미지 속성을 탐색하는 강력한 도구가 됩니다.



### Full-range Head Pose Geometric Data Augmentations (https://arxiv.org/abs/2408.01566)
Comments:
          arXiv admin note: text overlap with arXiv:2403.18104

- **What's New**: 이번 연구에서는 기존 머리 자세 추정(HPE) 방법의 한계를 극복하고자 하는 새로운 접근 방식을 제안하였습니다. 구체적으로, (1) 정확한 좌표계와 오일러 각도(Euler Angles) 추론 방법, (2) 특정 좌표계에서의 2D 기하학적 증강 수식을 제안하고, (3) 회전 행렬과 자세에 대한 정확한 도식 루틴을 도출하였으며, (4) 수학적 실험과 검증을 통해 피치-요(yaw) 커버리지를 개선하여 HPE 데이터셋 생성의 한계를 극복하였습니다.

- **Technical Details**: 이번 연구는 3D 회전 행렬 및 오일러 각도(Euler Angles)를 사용하여 전체 범위의 HPE 데이터셋을 생성하는 방법에 중점을 두었습니다. 특히 회전 행렬 R ∈ SO(3)를 통해 3D 회전을 표현하고, 이 행렬을 이용하여 객체를 회전시키는 방법을 제안하였습니다. 내재적(intrinsic) 좌표계와 외재적(extrinsic) 좌표계 간의 차이를 명확히 하였으며, 각 좌표계에서의 회전에 대한 정확한 수학적 정의를 제공하였습니다.

- **Performance Highlights**: 제안된 증강 기법을 기존의 HPE 방식에 적용한 결과, 모델 성능이 유의미하게 개선되었음을 실험을 통해 확인하였습니다. 특히, WHENet의 CMU Panoptic Dataset을 활용한 헤드 포즈 생성 방식을 개선하여, 더 높은 품질의 헬드 포즈 레이블 생성 데이터셋 CMU_HPE_10K를 제공하였습니다. 이 데이터셋은 10,466개의 이미지와 각 이미지에 대한 정밀한 헤드 회전 레이블을 포함하고 있습니다.



### Self-Supervised Depth Estimation Based on Camera Models (https://arxiv.org/abs/2408.01565)
- **What's New**: 본 논문은 'physics depth'라는 새로운 단일 카메라 깊이 추정 기법을 도입합니다. 이 기법은 카메라의 내재적(인트린직) 및 외재적(엑스트린직) 파라미터와 의미적 세그멘테이션(semantic segmentation)을 사용하여 절대 깊이를 계산합니다.

- **Technical Details**: 기존의 자가 지도 학습(self-supervised learning) 깊이 추정법은 이미지 관계를 활용했지만, 이 논문에서는 카메라 자체가 제공하는 근본적인 정보를 사용합니다. 물리학적 원리를 기반으로 평면적인 지역의 깊이를 계산하고, 이미지 의미 분석을 통해 실제 평면 지역을 식별하여 깊이 추정을 개선합니다. 구체적으로, 이미지의 픽셀 좌표를 카메라의 광학적 중심과 초점 거리 파라미터를 이용해 공간적 방향 벡터로 변환합니다. 이러한 물리적 방법으로 얻은 깊이를 'physics depth'라 명명했습니다.

- **Performance Highlights**: 제안된 방법은 KITTI, CityScape, Make3D 데이터셋에서 테스트 되었으며, 특히 근접하고 평면적인 표면에서 LiDAR와 비교해도 손색없는 정확도를 보여주었습니다. 이 방법은 기존의 자가 지도 학습 모델의 성능을 높이고, 추가적인 기기 없이 비용 효율적으로 절대 깊이 척도를 제공할 수 있습니다.



### Accelerating Domain-Aware Electron Microscopy Analysis Using Deep Learning Models with Synthetic Data and Image-Wide Confidence Scoring (https://arxiv.org/abs/2408.01558)
- **What's New**: 현미경에서의 특징 탐지를 향상시키기 위해 머신러닝 (ML) 모델을 통합하는 작업이 확대되었습니다. 하지만 이는 결함이 많고 희소한 수동 레이블 데이터셋에 대한 의존성과 도메인 인식 부족 등으로 인해 개발 및 적용에 어려움이 있었습니다. 이에 대한 해결책으로 물리 기반의 합성 이미지 및 데이터 생성기를 개발하였습니다.

- **Technical Details**: 이 생성기 덕분에, 인간이 레이블한 데이터로 훈련된 모델과 유사한 정확도 (0.86), 재현율 (0.63), F1 스코어 (0.71), 공학적 속성 예측 (R2=0.82)을 달성하는 머신러닝 모델이 만들어졌습니다. 우리는 특징 예측 신뢰도 점수를 사용하여 이미지 전체의 신뢰도 지표를 도출하고, 이를 통해 간단한 임계값 설정으로 모호하고 도메인 밖의 이미지를 제거함으로써 성능을 5-30% 향상시킬 수 있었습니다. 이 필터링 과정에서 25%의 이미지가 필터링되었습니다.

- **Performance Highlights**: 본 연구는 합성 데이터가 ML에서 인간 의존성을 제거할 수 있으며, 이미지 당 다수의 특징 탐지가 필요한 경우 도메인 인식을 제공 할 수 있음을 보여줍니다.



### Multi-task SAR Image Processing via GAN-based Unsupervised Manipulation (https://arxiv.org/abs/2408.01553)
Comments:
          19 pages, 17 figures, 7 tables

- **What's New**: 이 논문은 SAR(Synthetic Aperture Radar) 이미지 처리를 위한 새로운 프레임워크, GAN-based Unsupervised Editing (GUE)을 제안합니다. 이 프레임워크는 GAN(Generative Adversarial Network)의 잠재 공간(latent space)에서 의미있는 방향을 자동으로 찾아 SAR 이미지를 효과적으로 편집할 수 있는 기능을 제공합니다. GUE는 여러 SAR 이미지 처리 작업을 단일 학습 과정에서 수행할 수 있도록 설계되었습니다.

- **Technical Details**: GUE 프레임워크는 다음 두 가지 주요 목적을 달성하려고 합니다: 1) GAN의 잠재 공간에서 의미있는 방향을 분리(disentangle)하여 효과적인 SAR 이미지 편집을 가능하게 함, 2) 단일 학습 과정에서 다중 SAR 이미지 처리 작업을 수행할 수 있는 종합적인 SAR 이미지 처리 모델을 확립함. 이를 위해, 설명 가능한 방향을 식별하기 위해 네트워크를 교육하고, 다양한 작업을 수행하기 위해 다양한 의미적 방향을 선택합니다.

- **Performance Highlights**: GUE는 기존 방식들과 달리, 지도학습(supervised learning) 없이 잠재 공간의 의미를 탐색합니다. GAN의 잠재 공간에서 선형적으로 독립적인 방향을 효과적으로 탐구하고, 투명한 의미 작업을 구현하여 SAR 이미지 편집의 해석 가능성을 높였습니다. 실험 결과, SAR despeckling, SAR 배경 제거, SAR 이미지 회전 편집 등의 다양한 SAR 이미지 처리 작업에서 우수한 성능을 보여주었습니다.



### Trainable Pointwise Decoder Module for Point Cloud Segmentation (https://arxiv.org/abs/2408.01548)
Comments:
          No comments

- **What's New**: 자동차 및 로봇의 환경 이해를 위한 포인트 클라우드 분할 (Point Cloud Segmentation, PCS) 분야에서 새로운 접근, 가상 거리 이미지 가이드 복사-회전-붙여넣기 (Virtual Range Image-guided Copy-Rotate-Paste, VRCrop) 전략과 학습 가능한 포인트별 디코더 모듈 (Pointwise Decoder Module, PDM)이 제안되었습니다. 기존의 KNN 서치나 KPConv와 같은 모듈들이 가진 문제가 해결되었으며, PCS 성능을 크게 개선시켰습니다.

- **Technical Details**: 포인트 클라우드를 거리 이미지로 투영할 때, 여러 포인트가 같은 위치에 투영되어 정보 손실이 발생했습니다. 이를 해결하기 위해 PDM은 KNN 서치와 로컬 피처 추출 모듈을 사용하여, 주어진 포인트의 근접 이웃 포인트들을 효율적으로 찾아 예측을 개선합니다. 또한, VRCrop 데이터 증대 전략은 기존의 클래스 불균형 문제를 해결하며, 증대 후 포인트 수를 제한하여 학습 비용을 줄였습니다.

- **Performance Highlights**: 제안된 PDM과 VRCrop을 적용한 결과, SemanticKITTI, SemanticPOSS, 그리고 nuScenes 데이터셋에서 기존의 거리 이미지 기반 분할 모델들보다 향상된 성능을 보여주었습니다. 특히, PDM은 기존 모델들과의 통합 학습이 가능하며 효율적입니다. 실험 결과는 제안된 접근법이 성능 상한선을 넘어서는 데 효과적임을 증명했습니다.



### Non-linear Analysis Based ECG Classification of Cardiovascular Disorders (https://arxiv.org/abs/2408.01542)
Comments:
          23 pages, 9 Figures, 3 Tables

- **What's New**: 해당 연구는 비선형 분석 방법론을 사용하여 심장 장애를 탐지하는 새로운 방법을 보고합니다. 이 방법은 Recurrence plot 시각화를 활용하여 심전도(ECG) 데이터의 비선형 동적 특징을 분석합니다. 특히 Physikalisch-Technische Bundesanstalt (PTB) 데이터셋을 사용하여 심근경색, 분지 차단, 심근증 및 부정맥을 정확하게 분류하며 100%의 분류 정확도를 달성하였습니다.

- **Technical Details**: 재발현 플롯(Recurrence plot)은 비선형 데이터 분석을 위한 매우 효과적인 도구로, 동적 시스템 내 상태의 재발현을 시각화합니다. 이번 연구는 자동 인코더(autoencoder)와 Recurrence Quantification Analysis (RQA)를 결합하여 심장 질환을 분류합니다. 이 방법론은 ECG 신호의 복잡한 패턴과 특징을 보다 정확하고 견고하게 추출하고 정량화하여 비정상적인 심장 상태를 조기에 탐지하는 데 도움이 됩니다.

- **Performance Highlights**: 해당 연구는 PTB 데이터셋을 사용하여 심근경색, 분지 차단(Branch Bundle Block), 심근병증(Cardiomyopathy), 부정맥(Dysrhythmia) 및 정상인에 대한 분류를 수행했으며, 100%의 분류 정확도를 달성하였습니다. 이는 Recurrence plot과 t-SNE 시각화가 비정상적인 심장 상태와 정상인을 명확하게 구분할 수 있음을 보여줍니다.



### Guardians of Image Quality: Benchmarking Defenses Against Adversarial Attacks on Image Quality Metrics (https://arxiv.org/abs/2408.01541)
- **What's New**: 이 논문은 이미지 품질 평가 (Image Quality Assessment, IQA) 메트릭스의 적대적 공격 (adversarial attacks) 방어 메커니즘에 대한 포괄적인 벤치마킹 연구를 소개합니다. 25가지 방어 전략과 14가지 공격 알고리즘을 체계적으로 평가하며, 이러한 방어 방법이 IQA 작업에 얼마나 효과적인지를 분석합니다. 또한, 연구 커뮤니티에 더 진보된 방어 방법의 개발을 촉진하기 위해 새로운 방어 방법을 제출받고 있습니다. 최신 결과는 온라인에서 확인할 수 있습니다.

- **Technical Details**: 이번 연구에서는 비적응 및 적응 환경에서 14가지의 다양한 적대적 공격 알고리즘에 대해 25가지 방어 전략을 평가했습니다. 여기에는 적대적 정화 (adversarial purification), 적대적 훈련 (adversarial training), 공인 강건성 방법 (certified robustness methods)이 포함됩니다. 비적응 방어에 사용할 수 있는 적대적 이미지 데이터셋과 방어 메트릭스 평가를 위한 체계적인 기준을 마련했습니다. 연구 방법론은 GitHub 레포지토리에서 코드와 함께 제공됩니다.

- **Performance Highlights**: 연구 결과, 여러 방어 전략이 다양한 적대적 공격에 대해 상이한 성능을 보였으며, 일부 방어 전략은 이미지 품질을 유지하면서 원래의 IQA 점수를 복원하는 데 더 효과적임이 확인되었습니다. 이는 IQA 메트릭스의 강건성을 확보하는 데 중요합니다. 새로운 방어 방법은 벤치마크에 지속적으로 제출되어야 하며, 우리 연구는 이를 보다 체계적으로 평가하고 비교할 수 있는 첫 번째 기회를 제공합니다.



### SceneMotion: From Agent-Centric Embeddings to Scene-Wide Forecasts (https://arxiv.org/abs/2408.01537)
Comments:
          7 pages, 3 figures, ITSC 2024

- **What's New**: 새로운 SceneMotion 모델을 소개합니다. 이 모델은 멀티모달 예측을 위해 주의 메커니즘(attention-based model)을 사용하여 다중 교통 요원의 장면 전체 예측을 진행합니다. 기존의 에이전트 중심 예측을 개선해, 새로운 잠재적 컨텍스트 모듈(latent context module)을 통해 전체 장면의 예측 및 상호작용 모델링을 가능하게 합니다. Waymo Open Interaction Prediction Challenge에서도 뛰어난 성능을 보였습니다.

- **Technical Details**: SceneMotion은 다중 에이전트의 미래 궤적을 예측하기 위해 주의 메커니즘을 사용하며, 로컬 에이전트 중심의 임베딩을 장면 전체 예측으로 변환합니다. 잠재적 컨텍스트 모듈을 통해 여러 에이전트 중심의 임베딩을 학습하여 장면 전체의 잠재 공간을 구축합니다. 이는 에이전트 간의 상호작용을 공동으로 모델링하게 합니다. 또한, 경로 예측을 통해 상호작용 또는 충돌 가능성을 분류합니다.

- **Performance Highlights**: SceneMotion은 Waymo Open Interaction Prediction Challenge에서 경쟁력 있는 성능을 입증했습니다. 모델은 향후 궤적을 시간과 공간에서 클러스터링하여 에이전트 간의 상호작용을 정량화합니다. 이를 통해 얻어진 예측은 상호작용 또는 충돌을 분류하는 데 유용합니다.



### Multi-Unit Floor Plan Recognition and Reconstruction Using Improved Semantic Segmentation of Raster-Wise Floor Plans (https://arxiv.org/abs/2408.01526)
- **What's New**: 본 논문에서는 긴급상황 대비를 위한 도시 관리에서 주요 역할을 할 수 있는 디지털 트윈(Digital Twin)을 만들기 위해, 기존의 수작업이 필요한 3D 모델 생성 과정을 자동화하려는 시도를 제안합니다. 일반적으로 이용 가능한 2D 건축 평면도(Floor Plans)에서 3D 정보를 합성하는 방법을 제시합니다. 이를 위해 MDA-Unet과 MACU-Net 아키텍처 기반의 두 가지 새로운 픽셀단위 세그멘테이션(segmentation) 방법을 도입하였습니다.

- **Technical Details**: 본 연구에서는 향상된 스킵 연결(skip connections)과 어텐션 메커니즘(attention mechanism)을 가진 MDA-Unet과 MACU-Net 아키텍처를 사용하여 세그멘테이션을 수행하고, 그 결과물인 분할된 평면도를 3D 모델로 변환하는 복구 파이프라인을 제안합니다. 추가적으로 이 방법을 두 가지 다른 최신 기법 및 여러 벤치마크 데이터셋과 비교하였습니다.

- **Performance Highlights**: 제안된 방법은 CubiCasa 벤치마크 데이터셋에서 5가지 클래스에 대해 평균 F1 점수 0.86을 달성하며, 다른 픽셀단위 세그멘테이션 접근 방식보다 뛰어난 성능을 보여주었습니다. 또한 연구를 지원하기 위해 코드도 공개되었습니다.



### Using a CNN Model to Assess Visual Artwork's Creativity (https://arxiv.org/abs/2408.01481)
- **What's New**: 예술적 창의성 평가에 오랜 시간 동안 도전해왔던 연구자들이 이번 연구에서 중요한 돌파구를 마련했습니다. 이번 연구는 특히 그림보다는 회화의 창의성 평가에 초점을 맞추고 있으며, Convolutional Neural Network (CNN) 모델을 사용해 학생들의 회화 창의성을 자동으로 평가하는 방법을 개발했습니다.

- **Technical Details**: 600개의 전문가와 어린이 회화로 구성된 데이터셋을 이용해 CNN 모델을 훈련시켰습니다. 이 모델은 약 90%의 정확도를 달성했으며, 인간 평가자보다 빠른 평가 시간을 기록했습니다. 이를 통해 CNN 기반의 모델이 전통적인 평가 방식보다 더 효율적임을 보여줍니다.

- **Performance Highlights**: CNN 모델은 90%의 정확도로 예술적 창의성을 평가하는데 성공했습니다. 특히 평가 속도 면에서 인간 평가자보다 우수한 성능을 보였습니다. 이러한 성과는 기계 학습이 예술적 창의성 평가를 더 효율적으로 만들 잠재력을 가지고 있음을 시사합니다.



### Enhancing Online Road Network Perception and Reasoning with Standard Definition Maps (https://arxiv.org/abs/2408.01471)
Comments:
          Accepted by the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 본 논문에서는 자율 주행을 위한 고해상도(HD) 맵 대신 저해상도(SD) 맵을 활용하여 온라인 고해상도 맵을 구축하는 방법을 제안합니다. 기존의 HD 맵 방식은 비용과 유지보수에서 큰 문제를 겪고 있는데, 본 연구는 SD 맵을 통해서 이러한 문제를 해결하고자 합니다. 특히, Google Maps와 OpenStreetMaps(OSM)와 같은 널리 사용되는 SD 맵을 활용하여 시스템의 성능과 효율성을 증대시키고자 합니다.

- **Technical Details**: 본 연구에서는 다양한 온라인 매핑 아키텍처에 SD 맵을 통합하는 방식을 탐구하고, 이를 위해 OpenLane-V2 데이터셋을 OpenStreetMaps 데이터로 확장하였습니다. 이를 통해 그래픽 기반 SD 맵 표현의 이점을 평가하였습니다. 특히, SD 맵 인코더는 모델에 종속적이지 않으며 다양한 아키텍처에서 쉽게 적용될 수 있습니다. 예를 들어, 서라운드 뷰 카메라로부터 받은 이미지 입력과 자차 중심의 SD 맵 표현을 사용하여 중간선을 예측하는 구조를 제시하였습니다.

- **Performance Highlights**: SD 맵을 온라인 매핑 작업의 사전 정보로 활용하면 수렴 속도가 크게 빨라지고, 온라인 중심선 인식 작업의 성능이 30% 향상됩니다(mAP 기준). 또한, SD 맵 그래프를 활용하여 인식 및 추론 작업에서 필요한 파라미터 수를 줄이면서도 전반적인 성능을 향상시켰습니다. 이는 동적인 환경에서도 고성능을 유지하며, 장거리 예측에서도 뛰어난 성능을 나타냅니다.



### Img2CAD: Reverse Engineering 3D CAD Models from Images through VLM-Assisted Conditional Factorization (https://arxiv.org/abs/2408.01437)
- **What's New**: 이 논문은 이미지에서 3D 컴퓨터 지원 설계(CAD) 모델을 역설계하는 새로운 접근법을 소개합니다. 이 방법론은 GPT-4V와 같은 대형 기초 모델을 활용하여 글로벌 이산 구조를 예측하고, 'TrAssembler'라는 새로운 트랜스포머 기반 네트워크를 사용해 이 구조를 기반으로 연속적 속성 값을 예측합니다. 또한, 저자들은 ShapeNet에서 특정 물체의 주석이 달린 CAD 데이터셋을 구축하여 모델 훈련을 지원했습니다.

- **Technical Details**: 이 논문은 이미지에서 CAD 모델을 역설계하는 작업을 두 하위 문제로 분리하는 조건적 분해 방식을 제안합니다. 먼저, GPT-4V와 같은 대형 비전 모델을 활용하여 이미지를 기반으로 글로벌 이산 구조를 예측합니다. 그 다음, 'TrAssembler'라는 트랜스포머 기반 네트워크를 제안하여, 이산 구조에 조건화된 연속적 속성 값을 예측합니다. 이 방법은 PartNet와 ShapeNet의 혼합을 이용해 일반 객체에 대한 CAD 데이터셋을 구축하여 훈련에 사용됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법론이 기존의 모델을 능가하며, 특히 야생의 이미지에서 CAD-화를 수행하고 형상을 편집하는 다양한 응용 프로그램에서 유망한 성과를 보였습니다.



### A New Clustering-based View Planning Method for Building Inspection with Dron (https://arxiv.org/abs/2408.01435)
- **What's New**: 본 논문에서는 드론 기술의 발전에 따라 드론을 이용한 건물 검사 및 감시에 대한 연구가 활발해지고 있는 배경 아래, 시각 센서를 장착한 드론을 통해 목표 건물 표면을 덮는 최적의 시점을 찾기 위한 새로운 방법을 제안한다. 제안된 방법은 스펙트럴 클러스터링(spectral clustering), 로컬 포텐셜 필드(local potential field) 방법, 그리고 하이퍼 휴리스틱 알고리즘(hyper-heuristic algorithm)을 사용하여 정점 시점(candidate viewpoints)을 생성하고, 이를 최적화하는 두 단계의 계산 방식을 채택하고 있다.

- **Technical Details**: 논문에서는 모델 기반의 뷰 플래닝(view planning) 문제를 해결하기 위해 스펙트럴 클러스터링과 로컬 포텐셜 필드 방법을 사용하여 우선적으로 정점 시점을 생성 및 수정한다. 이후, 이 문제를 세트 커버링 문제(Set Covering Problem, SCP)로 변환하고, 하이퍼 휴리스틱 알고리즘을 통해 최적의 시점 집합을 선택한다. 클러스터링 단계에서는 삼각형 메시의 가중 거리와 노멀 벡터를 기반으로 스펙트럴 클러스터링을 수행하고, 로컬 포텐셜 필드 방법을 통해 후보 시점의 위치를 교정함으로써 시각 커버리지 효율을 높인다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 더 적은 시점을 사용하면서도 더 높은 커버리지를 달성하여 기존 방법들보다 더욱 효율적임을 입증하였다. 이는 대상의 복잡한 기하학적 특징을 보다 잘 고려하고, 비평면형 클러스터에 대해 강인한 성능을 유지하는 덕분이다.



### Evaluating and Enhancing Trustworthiness of LLMs in Perception Tasks (https://arxiv.org/abs/2408.01433)
Comments:
          Accepted in 27th IEEE International Conference on Intelligent Transportation Systems (ITSC) 2024

- **What's New**: 이 논문에서는 비전 기반 데이터(vision-based data)에서 객체 감지(object detection)를 위한 LLM의 헬루시네이션(hallucination) 검출 전략을 체계적으로 평가했습니다. 예를 들어, 보행자 감지 및 위치 추적에서 이러한 전략들을 시도했습니다. 최신 LLM인 GPT-4V와 오픈 소스인 LLaVA를 두 데이터셋(Waymo/미국, PREPER CITY/스웨덴)에서 평가했습니다.

- **Technical Details**: 세 가지 헬루시네이션 검출 전략을 사용했으며, 이 전략들은 모델의 컨시스턴시(consistency)를 높이기 위한 투표 기반 방법(Best-of-Three, BO3)과 과거 데이터를 포함하여 헬루시네이션 정보를 확장하는 기법 등이 포함됩니다. 특히, GPT-4V와 LLaVA라는 두 가지 LLM을 사용하여 비디오 시퀀스에서 보행자 감지 사례를 통해 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과에 따르면, GPT-4V가 오픈 소스 LLaVA보다 훨씬 뛰어난 성능을 보였습니다. 하지만 BO3 등의 투표 기반 방법은 헬루시네이션을 효과적으로 줄이지 못했고, 높은 false negative율을 보였습니다. 반면, 과거 정보를 포함함으로써 헬루시네이션 검출 결과가 향상되었습니다.



### VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidanc (https://arxiv.org/abs/2408.01432)
- **What's New**: Concept Bottleneck Models (CBMs)은 모델의 결정을 설명하기 위해 인간이 이해할 수 있는 개념을 인코딩하는 중간 층 (Concept Bottleneck Layer, CBL)을 도입하여 해석 가능한 예측을 제공합니다. 최근 연구들은 대형 언어 모델 (LLMs)과 사전 학습된 비전-언어 모델 (VLMs)을 활용하여 CBM의 훈련을 자동화하고 확장 가능한 방법을 제공했습니다. 그러나 기존 접근법은 두 가지 주요 문제점을 지니고 있습니다. 첫째, CBL이 예측한 개념이 입력 이미지와 일치하지 않아 해석의 신뢰성을 의심하게 만듭니다. 둘째, 개념 값이 의도하지 않은 정보를 인코딩하여 무작위 개념 세트도 최첨단 CBM과 비슷한 성능을 달성할 수 있습니다. 이 문제를 해결하기 위해, 본 연구는 Vision-Language-Guided Concept Bottleneck Model (VLG-CBM)을 제안하여 신뢰할 수 있는 해석 가능성을 제공하면서 성능을 향상시킵니다.

- **Technical Details**: 제안된 VLG-CBM은 사전 준비된 오픈 도메인 객체 탐지기를 활용하여 시각적으로 기반한 개념 주석을 제공합니다. 이를 통해 개념 예측의 신뢰성을 크게 향상시키고 모델 성능을 더욱 개선합니다. 또한, 정보 누출을 통제하고 더 나은 해석 가능성을 제공하기 위해 유효 개념 수 (Number of Effective Concepts, NEC)라는 새로운 메트릭을 제안합니다. 이 연구는 다섯 가지 표준 벤치마크에 대한 광범위한 평가를 통해 제안된 방법이 기존 방법보다 NEC=5에서 최소 4.27%에서 최대 51.09%까지, 다양한 NEC에서 평균 정확도 기준 최소 0.45%에서 최대 29.78%까지 뛰어남을 보여줍니다.

- **Performance Highlights**: 제안된 VLG-CBM은 NEC=5에서 최소 4.27%에서 최대 51.09%까지 정확도를 향상시켰습니다. 또한, 다양한 NEC에서도 평균 정확도 기준 최소 0.45%에서 최대 29.78%까지 뛰어난 성능을 보였습니다. 이러한 결과는 학습된 개념의 신뢰성과 해석 가능성을 보존하면서도 모델 성능을 현저히 향상시키는 것을 입증합니다.



### SUSTechGAN: Image Generation for Object Recognition in Adverse Conditions of Autonomous Driving (https://arxiv.org/abs/2408.01430)
Comments:
          10 pages, 9 figures

- **What's New**: 이번 연구에서는 자율주행 차량의 악조건에서 객체 인식을 개선하기 위해 듀얼 어텐션 모듈과 멀티스케일 생성기를 포함한 새로운 SUSTechGAN을 제안했습니다. 특히, 악천후와 야간 상황에서 운전 이미지를 생성하고, 이를 사용해 YOLOv5 객체 인식 네트워크를 재학습시키는 방법을 검토했습니다.

- **Technical Details**: SUSTechGAN은 듀얼 어텐션 모듈과 멀티스케일 생성기를 사용하여 운전 이미지를 생성합니다. 듀얼 어텐션 모듈은 지역의 의미론적 특징 추출을 개선하고, 멀티스케일 생성기는 다양한 크기의 특징을 고려하여 고품질 이미지를 생성합니다. 또한, 새로운 손실 함수로 탐지 손실(detection loss), 적대적 손실(adversarial loss), 사이클 일관성 손실(cycle consistency loss)을 제안하여 객체 인식을 향상시킵니다.

- **Performance Highlights**: 실험 결과, SUSTechGAN이 생성한 운전 이미지를 통해 재학습된 YOLOv5 모델이 악천후와 야간 조건에서 객체 인식 성능이 크게 향상되었습니다. 이는 기존의 잘 알려진 GAN들보다 뛰어난 성능을 보여줬습니다.



### Transferable Adversarial Facial Images for Privacy Protection (https://arxiv.org/abs/2408.01428)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 새로운 얼굴 프라이버시 보호 스키마가 제안되었습니다. 이 스키마는 추가적인 사용자의 참조 이미지를 필요로 하지 않으며 높은 시각적 품질을 유지하면서도 높은 전이성을 가진 자연스러운 적대적 얼굴 이미지를 생성합니다.

- **Technical Details**: 이 연구는 전체 얼굴 공간을 직접 형성하는 것을 제안하며, 이를 위해 생성 모델의 잠재 공간을 탐색합니다. 주요 구성 요소로는 글로벌 적대적 잠재 검색(Global Adversarial Latent Search, GALS), 주요 랜드마크 정규화 모듈(Key Landmark Regularization, KLR) 및 다양한 잠재 공간의 영향을 조사하는 과정이 포함됩니다. 최적의 잠재 공간으로는 \\mathcal{F} 공간이 선택되었습니다.

- **Performance Highlights**: 두 개의 데이터셋에 대한 광범위한 실험 결과, 제안된 접근 방식은 최신 방법에 비해 평균 25% 향상된 전이성을 보였으며, 상용 얼굴 인식 API(예: Face++, Aliyun, Tencent)에서도 10%의 향상을 달성했습니다.



### Siamese Transformer Networks for Few-shot Image Classification (https://arxiv.org/abs/2408.01427)
Comments:
          12 pages

- **What's New**: 이번 연구에서는 인간의 뛰어난 시각 분류 능력에서 영감을 받아, 새로운 이미지에 대해 적은 예시만으로도 정확하게 인식하는 방법을 제안합니다. 기존의 몇몇 샷 이미지 분류(few-shot image classification) 방법은 글로벌 특징(global features) 또는 로컬 특징(local features)에 중점을 둡니다. 하지만 이번 연구에서는 두가지 특징을 통합하는 Siamese Transformer Network (STN) 접근법을 제안했습니다.

- **Technical Details**: 제안된 방법은 두 개의 병렬 네트워크 지점을 사용하여 각각 글로벌 특징과 로컬 특징을 추출합니다. 이 네트워크는 사전 학습된 Vision Transformer (ViT) 아키텍처를 기반으로 합니다. 글로벌 특징에는 유클리드 거리 측정(Euclidean distance measure)을 사용하고, 로컬 특징에는 KL 발산(Kullback-Leibler divergence) 측정을 적용합니다. 두 측정치를 통합하기 위해 L2 정규화(L2 normalization)를 수행한 후 가중치를 부여하여 최종 유사도 점수를 얻습니다. 이 과정에서 메타 학습 접근법(meta-learning approach)을 사용하여 네트워크를 미세 조정합니다.

- **Performance Highlights**: 제안된 STN 방법은 5-shot 및 1-shot 시나리오 모두에서 기존 최첨단 모델들과 비교하여 우수한 성능을 보였습니다. 네 가지 인기 있는 벤치마크 데이터셋을 사용한 실험에서 STN이 가장 뛰어난 성능을 나타냈습니다.



### On Using Quasirandom Sequences in Machine Learning for Model Weight Initialization (https://arxiv.org/abs/2408.02654)
- **What's New**: 기계 학습에서 모델의 초기 가중치(initial weights)를 설정할 때, 전통적으로 사용되던 의사난수 생성기(pseudorandom number generators, PRNGs) 대신 낮은 불일치 수열(low-discrepancy sequences)을 사용하면 모델의 성능이 향상될 수 있다는 연구가 발표되었습니다. 특히 Sobol' 수열과 같은 준난수 생성기(quasirandom number generators, QRNGs)를 사용하여 이러한 효과를 검증했습니다.

- **Technical Details**: 해당 연구에서는 Multi-Layer Perceptrons (MLP), Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), 그리고 Transformer 같은 네트워크 구조에 대해 연구를 진행했습니다. 다양한 초기화 방법(Glorot, He, Lecun (Uniform 및 Normal), Orthogonal, Random Normal, Truncated Normal, Random Uniform)을 사용해 PRNG와 QRNG 초기화 방식의 성능을 비교했습니다. MNIST, CIFAR-10, IMDB 데이터셋과 SGD 및 Adam 옵티마이저를 사용했습니다.

- **Performance Highlights**: 120개의 실험 중 60%에서 QRNG 기반 초기화 방식이 PRNG보다 높은 정확도를 달성하거나 같은 정확도를 더 빠르게 달성했습니다. QRNG 초기화 방식은 특히 Glorot Uniform, He Uniform, Glorot Normal, He Normal, Lecun Uniform, Lecun Normal, Truncated Normal 초기화 방법에서 현저한 성능 향상을 보였습니다. 일부 실험에서는 약간의 성능 저하가 있었으나, 이는 델타 값이 매우 작았습니다.



### Cross-Modality Clustering-based Self-Labeling for Multimodal Data Classification (https://arxiv.org/abs/2408.02568)
Comments:
          10 pages, 5 figures, 9 tables

- **What's New**: 새로운 연구로 Cross-Modality Clustering-based Self-Labeling(CMCSL) 메소드를 제안합니다. 이 방법은 소수의 사전 라벨링 된 데이터를 기반으로 각 모달리티(modality)를 심층 특징 공간(deep feature space)에서 군집화(clustering)시키고, 그 결과 클러스터 내에서 알려진 라벨을 전파(propagate)합니다. 그런 다음, 각 모달리티에서 인스턴스의 클래스 멤버십에 대한 정보를 유클리드 거리(Euclidean distance)에 기반하여 교환하여 더욱 정확한 라벨링을 보장합니다. 20개의 MM-IMDb 데이터셋 실험에서 CMCSL은 소수의 라벨이 사전 라벨링 된 경우 모달리티 간 라벨의 교차 전파가 보다 신뢰할 수 있는 라벨링을 가능하게 하고, 따라서 각 모달리티에서 분류 성능을 높일 수 있음을 보여주었습니다.

- **Technical Details**: CMCSL은 각 모달리티를 사전 라벨링 된 데이터에 기반하여 군집화하고, 알려진 라벨을 클러스터 내에서 전파합니다. 그런 다음, 유클리드 거리로 결정된 각 모달리티의 중심과의 거리 기반으로 최종 라벨을 결정합니다. 이 방법은 비용이 많이 드는 라벨링 과정을 줄이는 데 목표를 두고 있으며, 기존의 특징 공간에서만 작동하는 세미-슈퍼바이즈드 러닝(semi-supervised learning) 방법과 달리, 각 모달리티의 보완적인 정보를 사용하여 분류 성능을 향상시키는 것을 목적으로 합니다.

- **Performance Highlights**: CMCSL은 특히 소수의 사전 라벨링 된 데이터가 주어진 경우에도 모달리티 간 라벨 교차 전파를 통해 더 신뢰할 수 있는 라벨링을 가능하게 하고, MM-IMDb 데이터셋을 기반으로 한 실험에서 각 모달리티의 분류 성능을 증가시킬 수 있음을 입증했습니다.



### Automatic rating of incomplete hippocampal inversions evaluated across multiple cohorts (https://arxiv.org/abs/2408.02496)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이번 연구에서는 불완전 해마 역전(Incomplete Hippocampal Inversion, IHI)을 자동으로 평가하는 방법을 처음으로 제안했습니다. IHI는 해마의 비정상적인 해부학적 패턴으로, 전체 인구의 약 20%에서 발견됩니다. 연구는 IHI와 다양한 뇌 질환(예: 간질, 조현병)의 연관성을 조사했지만, 대부분 소규모 표본을 기반으로 했습니다. 그러므로 대규모 연구가 필요합니다.

- **Technical Details**: 연구진은 IHI 점수를 자동으로 평가하기 위해 4가지 해부학적 기준을 예측하는 Deep Learning 기법을 도입했습니다. 사용된 모델은 conv5-FC3, ResNet, 그리고 Squeeze-and-Excitation ResNet으로 구성되었습니다. 또한, Ridge Regression과 다양한 머신러닝 기법을 실험했습니다. 데이터 세트로는 IMAGEN(2,008명), QTIM/QTAB(993명과 403명), 그리고 UKBiobank(985명)를 사용했습니다.

- **Performance Highlights**: 실험 결과 Deep Learning 모델이 Ridge Regression보다 높은 성능을 보였습니다. 특히 conv5-FC3 네트워크는 복잡성이 낮고 계산 시간이 짧으면서도 더 복잡한 네트워크와 동등한 성능을 나타냈습니다. 여러 코호트 데이터를 사용한 학습이 일반화 성능을 개선하는 데 기여함을 발견했습니다.



### An investigation into the causes of race bias in AI-based cine CMR segmentation (https://arxiv.org/abs/2408.02462)
- **What's New**: 이 논문은 인공지능(AI) 기반의 심장 자기공명영상(CMR) 분할 모델이 인종 편향(race bias)을 나타내는 원인에 대해 조사했습니다. 연구 결과, 인종 편향의 주된 원인은 심장 부위 외부의 이미지 콘텐츠에 있다는 것을 밝혀냈습니다. 그리고 이미지를 심장 주변으로 자를 경우 편향을 줄일 수 있음을 보여줍니다.

- **Technical Details**: 실험 데이터로는 영국 바이오뱅크(UK Biobank)에서 제공된 436명의 단축상(cine short-axis) CMR 이미지를 사용했습니다. 218명은 백인, 218명은 흑인으로 구성되었습니다. 분류 모델로는 18-레이어 레즈넷(ResNet-18)을 사용했고, 분할 모델로는 nnU-Net을 활용했습니다. 분류 및 분할 실험에서 다양한 모델 학습 파라미터와 이미지 보강 방법을 적용했습니다. 그리고 GradCAM을 통해 모델의 해석 가능성을 높였습니다.

- **Performance Highlights**: 분류 실험에서는 인종을 이미지로 높은 정확도로 예측할 수 있었지만, 진실된 분할(ground truth segmentation)에서는 낮은 정확도를 보였습니다. 이는 주로 이미지 기반의 분포 변동(distributional shift) 때문이라고 생각됩니다. 이미지 내부의 비심장 영역(피하 지방 등)에 모델이 집중하는 것을 확인했습니다. 이미지를 심장 주변으로 자르면 분류 정확도가 거의 무작위 수준으로 감소했습니다. 분할 모델에서도 잠재 표현(latent representations)에서 인종 정보를 예측할 수 있음을 보여주며, 이 역시 심장 이미지로 자르면 편향은 줄어들지만 완전히 사라지진 않았습니다.



### StoDIP: Efficient 3D MRF image reconstruction with deep image priors and stochastic iterations (https://arxiv.org/abs/2408.02367)
Comments:
          10 pages, 2 figures, 1 table, 1 algorithm

- **What's New**: 최신 연구에서 제안된 StoDIP 알고리즘은 기존의 두 차원(DIP) 구현을 넘어 다양한 문제를 해결하여 세 차원(3D) MRF 이미징에 적용할 수 있습니다. 컴퓨터 효율성을 높이기 위해 메모리 효율적인 확률적 업데이트와 신경망 아키텍처 선택 및 더 빠른 비균일한 FFT(NUFFT) 변환을 활용하여보다 빠르게 수렴합니다. 이를 통해 기존의 DIP 구현에 비해 빠른 성능을 자랑합니다.

- **Technical Details**: StoDIP은 초기 Deep Image Prior(DIP) 알고리즘을 기반으로 합니다. 본 연구에서는 특히 메모리 효율적인 확률적 업데이트, 선택된 하이퍼파라미터(backbone network architecture) 및 추가적인 정규화, 더 빠른 cuFINUFFT 라이브러리를 기반으로 한 비균일한 Fast Fourier Transform(NUFFT)을 도입했습니다. 이러한 방법들은 전체 뇌 스캔 데이터 세트에서 8분의 획득 시간을 두 배로 단축시켜, 더욱 빠른 수렴과 더 향상된 성능을 제공합니다.

- **Performance Highlights**: 건강한 자원봉사자의 전체 뇌 스캔 데이터 세트에서 StoDIP은 정량적 및 질적 성능에서 다른 비지도 학습 재구성 기준선보다 우수한 성능을 보였습니다. 따라서 DIP를 활용한 3D MRF 이미지 재구성은 임상적으로 더 효율적이고 가치 있는 방법이 될 수 있습니다.



### Perception Matters: Enhancing Embodied AI with Uncertainty-Aware Semantic Segmentation (https://arxiv.org/abs/2408.02297)
- **What's New**: 이 논문은 기존의 객체 검색 방법들에서 현재 문제와 격차를 식별하고 이를 해결하기 위해 새로운 접근방식을 제안합니다. 주로 날짜가 지난 인식 모델을 사용하고, 시간 집계를 무시하며, 테스트 시의 소음 있는 인식으로 인해 과신하는 문제에 대한 대응이 필요함을 강조합니다.

- **Technical Details**: 주요 기술적 세부 사항은 다음과 같습니다. 첫째, 'Calibrated perception probabilities'와 불확실성 추정을 통해 시간 집계 및 결정 내리기 과정을 개선합니다. 둘째, 사전 학습된 모델들과의 직접 통합이 가능하여 추가 학습 비용 없이 적용이 가능합니다. 이에 더해, 모듈 구조인 'modular perception-mapping-policy' 파이프라인을 사용하여 다양한 객체 검색 작업 시의 성능을 평가합니다.

- **Performance Highlights**: 실험 결과를 통해, 시간 집계와 불확실성 기반 기법이 기존 방식 대비 더 좋은 성능을 보인다는 것을 확인했습니다. 다양한 시맨틱 인식 모델과 정책을 조합하여 평가한 결과, 제안된 방법이 전반적으로 더 나은 결과를 제공했음을 보여줍니다.



### Hierarchical Clustering using Reversible Binary Cellular Automata for High-Dimensional Data (https://arxiv.org/abs/2408.02250)
- **What's New**: 이 연구는 고차원 데이터셋을 효율적으로 클러스터링하기 위한 계층적 클러스터링 알고리즘을 제안합니다. 본 연구는 되돌릴 수 있는 유한 셀룰러 오토마타(CA)의 순환 공간을 활용하여 클러스터를 형성하는 새로운 방법을 소개합니다.

- **Technical Details**: 셀룰러 오토마타(CA) 기반 클러스터링에서는 두 객체가 같은 사이클에 속할 경우 밀접하게 연관되어 동일한 클러스터의 구성원으로 간주됩니다. 본 논문은 각 사이클의 모든 요소의 중간 값을 기반으로 서로 다른 사이클의 객체들 간의 관계를 규명하여 다음 단계에서 이를 그룹화할 수 있도록 합니다. 정보 전달 및 사이클 구조를 기반으로 최적의 규칙을 찾는 규칙 선택 전략을 도입하여 중간 클러스터의 수를 최소화하고 계산 비용을 줄입니다. 데이터셋을 빈도 기반 인코딩을 사용하여 연속 데이터 요소가 인코딩된 형태에서 최소 해밍 거리(Hamming distance)를 유지하도록 인코딩한 후, 제안된 클러스터링 알고리즘은 3단계를 반복하여 사용자가 요구하는 클러스터 수에 맞게 데이터를 최종적으로 클러스터링합니다.

- **Performance Highlights**: 본 알고리즘은 헬스케어, 스포츠, 화학 연구, 농업 등 다양한 분야에 적용될 수 있으며, 표준 벤치마크 데이터셋을 이용한 성능 검증 결과 기존 알고리즘과 동등한 성능을 보이며, 시간 복잡도는 쿼드러틱(quadratic)입니다.



### Applying Conditional Generative Adversarial Networks for Imaging Diagnosis (https://arxiv.org/abs/2408.02074)
- **What's New**: 이 연구는 Conditional Generative Adversarial Networks (C-GAN)과 Stacked Hourglass Networks (SHGN)을 결합한 혁신적인 이미지 세분화(이미지 분할) 응용을 소개합니다. 특히 의료 영상 분야에서 이미지 세분화의 성능을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: 복잡한 이미지 데이터셋에 적용된 딥러닝 모델에서 흔히 발생하는 과적합(overfitting) 문제를 해결하기 위해, 회전 및 스케일링을 통해 데이터를 증강(data augmentation)합니다. L1과 L2 재구성 손실(reconstruction losses)을 결합한 하이브리드 손실 함수(hybrid loss function)와 적대적 훈련(adversarial training)을 통해 세분화 프로세스를 개선합니다. 특히 우리의 접근법은 특정 분야의 전문 지식에 대한 의존 없이도 의료 이미지 내의 조직 경계와 혈관 구조와 같은 뚜렷한 영역을 정확하게 구분하는 기능을 가지고 있습니다.

- **Performance Highlights**: 알고리즘은 표준 의료 이미지 라이브러리를 사용하여 평가되었으며, 기존 방법들에 비해 우수한 성능 지표를 보여주었습니다. 이는 딥러닝을 통해 자동화된 의료 진단을 향상시킬 수 있는 잠재력을 입증합니다.



### Decision Support System to triage of liver trauma (https://arxiv.org/abs/2408.02012)
- **What's New**: 이번 논문에서는 CT 스캔을 활용해 간 출혈(liver bleeding) 및 열상(lacerations)을 감지하는 새로운 방법을 제시합니다. GAN Pix2Pix 변환 모델(translation model)을 사용하여 진단 정확도를 크게 향상시켰습니다. 이 연구는 응급 의료 서비스에서의 사용을 목표로 하고 있으며, 전체 시스템은 기존의 의료 영상 기술과 원활하게 통합될 수 있습니다.

- **Technical Details**: 제안된 방법은 GAN Pix2Pix 모델을 활용하여 CT 스캔 데이터를 변환합니다. GAN Pix2Pix 모델은 이미지 간의 변환을 용이하게 하여 정확한 간 출혈 및 열상 감지를 가능하게 합니다. 이 모델은 전신 CT 스캔에서 많은 양의 데이터를 처리하고, 빠른 진단을 통해 치료 비용을 줄이고 2차 합병증 발생 가능성을 낮춥니다.

- **Performance Highlights**: 이 방법은 간 출혈 감지에서 97%, 간 열상 감지에서 93%의 정확도를 기록하며, 이는 현재의 최첨단 기술보다 상당히 개선된 결과입니다. Dice 점수 메트릭(Dice score metrics)을 활용하여 모델의 효과를 정량화하였으며, 높은 정확도를 통해 긴급 의료 상황에서의 진단 정밀도와 속도를 향상시킬 수 있음을 입증했습니다.



### SR-CIS: Self-Reflective Incremental System with Decoupled Memory and Reasoning (https://arxiv.org/abs/2408.01970)
- **What's New**: 인간의 기억 및 학습 메커니즘에서 영감을 받아, 새로운 정보를 빠르게 학습하면서도 기존 메모리를 유지할 수 있는 SR-CIS(Self-Reflective Complementary Incremental System)를 제안합니다. 이 시스템은 Complementary Inference Module (CIM)과 Complementary Memory Module (CMM)로 구성되어 있으며, 효율적인 협업을 위해 CA-OAD(Confidence-Aware Online Anomaly Detection) 메커니즘을 도입했습니다.

- **Technical Details**: SR-CIS는 빠른 추론을 위한 작은 모델과 신중한 추론을 위한 큰 모델로 구성된 CIM과 과제로부터의 기억을 단기 및 장기 메모리로 구분한 CMM으로 구성되어 있습니다. CA-OAD 메커니즘은 예측의 신뢰도를 온라인으로 평가하여 하드 샘플을 감지합니다. 단기 메모리는 특정 태스크의 LoRA (Low-Rank Adaptive) 및 프로토타입 가중치와 편향으로 구성되며, 장기 메모리는 다양한 태스크 메모리를 통합하여 저장합니다. 훈련 시, 이미지의 텍스트 설명을 저장하고, 시나리오 재생 모듈(SRM)을 사용하여 메모리 결합을 수행합니다.

- **Performance Highlights**: SR-CIS는 한정된 저장 공간과 데이터 리소스 제약 아래에서 모델의 플라스틱성과 메모리 안정성을 균형 있게 유지합니다. 표준 및 few-shot class incremental learning 벤치마크에서 기존의 경쟁 벤치마크들을 상회하는 성능을 보였습니다.



### EqvAfford: SE(3) Equivariance for Point-Level Affordance Learning (https://arxiv.org/abs/2408.01953)
Comments:
          Accept to CVPRWorkshop on Equivariant Vision: From Theory to Practice 2024

- **What’s New**: 이 논문에서는 신규 EqvAfford 프레임워크를 제안하였습니다. 이는 3D 객체 조작 작업에서 물체의 6D 자세(translation, rotation, tilt)에 독립적으로 일관된 조작 전략을 보장하는 SE(3) equivariance 개념을 활용한 것입니다. 전통적인 모델들과 달리, 이 프레임워크는 다양한 물체 자세에 대해 우수한 성능과 일반화 능력을 가지며, 포인트 수준의 객체 활용 학습을 통해 로봇 조작 작업을 개선합니다.

- **Technical Details**: EqvAfford 프레임워크는 네 가지 모듈로 구성됩니다. VN-DGCNN Encoder는 물체의 불변 및 등변 특징을 추출하며, Affordance Prediction Module은 조작 가능 포인트를 예측합니다. Action Proposal Module은 다수의 후보 행동을 제안하며, Action Scoring Module은 이러한 행동을 평가하여 최적의 행동을 선택합니다. 6D pose와 affordance score를 예측하는 과정에서 SE(3) 변환에 대해 invariant 및 equivariant 특징을 학습합니다.

- **Performance Highlights**: 실험 결과, EqvAfford는 다양한 6D 자세를 가진 물체에 대해서도 높은 성능을 보이며, 새로운 물체 자세에서도 우수한 일반화 능력을 보였습니다. 제안된 프레임워크는 다른 방법들에 비해 월등한 성능을 보이며, 이론적 보장을 통해 일관된 조작 전략을 제공합니다.



### Visual Grounding for Object-Level Generalization in Reinforcement Learning (https://arxiv.org/abs/2408.01942)
Comments:
          35 pages, 14 figures, 17 tables

- **What's New**: 이번 연구에서는 시각-언어 모델(VLM)을 활용하여 객체 중심 작업에서 새로운 객체와 지시에 대한 제로샷 일반화(Zero-shot Generalization)을 가능하게 하는 방법을 제안합니다. 이를 통해 자연어 지시를 따르는 에이전트의 일반화 능력을 향상시키고자 합니다.

- **Technical Details**: 기술적으로, 본 연구는 마인크래프트(Minecraft) 환경에서 MineCLIP을 통해 시각-언어 지식을 강화를 위한 학습(Reinforcement Learning, RL)에 전달합니다. 주로 두 가지 경로로 이를 수행합니다. 첫째, 목표 객체에 대한 자신감 맵(confidence map)에서 유도된 객체-기반 내재적 보상(intrinsic reward)을 제안하여 에이전트를 목표 객체로 더 효과적으로 안내합니다. 둘째, 자신감 맵을 에이전트의 정책 입력으로 통합하여 객체 레벨에서 제로샷 일반화를 가능하게 합니다.

- **Performance Highlights**: 싱글 태스크 실험에서, 객체-기반 내재적 보상이 MineCLIP 보상보다 도전적인 스킬 학습에서 더 뛰어난 성능을 보임을 확인했습니다. 다중 태스크 실험에서는 자신감 맵을 태스크 표현(Task Representation)으로 사용하는 에이전트가 언어 기반 방법보다 더 나은 일반화 능력을 보였습니다. 새로운 객체에 대한 성공률은 사냥 분야에서 약 300%, 수확 분야에서 약 100% 향상되었습니다.



### Advancing H&E-to-IHC Stain Translation in Breast Cancer: A Multi-Magnification and Attention-Based Approach (https://arxiv.org/abs/2408.01929)
Comments:
          Accepted by IEEE CIS-RAM 2024 Invited Session Oral

- **What's New**: 본 연구는 기존 병리 이미지 번역 기술의 문제를 해결하기 위해 주의 메커니즘(Attention Mechanism)과 다중 배율 정보 처리(Multi-magnification Information Processing)를 통합한 새로운 모델을 제안합니다. 본 모델은 다양한 배율의 병리 이미지를 효율적으로 번역할 수 있으며, 중요한 정보를 우선적으로 처리하여 기존 방법보다 더 정확한 HER2 면역화학(IHC) 슬라이드를 생성합니다.

- **Technical Details**: 제안된 모델은 Generative Adversarial Networks(GANs)의 기본 원칙을 바탕으로 합니다. 다중 배율 처리 전략은 병리 이미지의 여러 배율에서 정보를 추출하여 번역 훈련 시 활용하며, 생성기 네트워크 내 주의 모듈은 병리 이미지의 중요한 정보를 우선적으로 추출하여 정확한 이미지 분포 번역을 가능하게 합니다. 이 모델은 주어진 H&E 이미지를 다양한 배율로 처리하여 IHC 이미지로 변환합니다.

- **Performance Highlights**: 공개된 유방암 데이터셋에서 엄격한 테스트를 통해 제안된 모델의 우수성이 입증되었습니다. 주관적 평가와 객관적 메트릭스 모두에서 기존 방법보다 뛰어난 성능을 보였으며, 병리 이미지 번역 분야에서 최신 기술로 자리매김했습니다.



### Self-Supervised Pretrained Models and Latent Feature Distribution Optimization (https://arxiv.org/abs/2408.01920)
- **What's New**: 이번 논문에서는 자연 이미지의 복잡성을 다루며, 자체 지도(self-supervised) 사전학습 모델과 잠재 특징 분포 최적화(latent feature distribution optimization)를 기반으로 한 이미지 클러스터링 알고리즘을 소개합니다. 이러한 접근 방식은 클러스터링 성능을 현저하게 향상시킵니다.

- **Technical Details**: 본 논문에서는 자체 지도 사전학습 모델을 활용하여 훈련 샘플에 대해 k-최근접 이웃(k-nearest neighbor) 이미지를 탐색하고, 훈련 샘플과 가장 가까운 이웃 간의 거리를 단축하는 방법을 사용합니다. 이를 통해 잠재 특징의 판별력을 강화하고 클러스터링 성능을 개선합니다. 또한, 잠재 특징 공간에서 샘플 특징과 사전 정의된 클러스터 중심간의 거리를 줄여 잠재 특징 분포를 최적화합니다. 최소 코사인 거리 손실(minimum cosine distance loss) 함수를 사용하여 알고리즘의 성능을 향상시킵니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, 본 접근 방식을 통해 최신 클러스터링 알고리즘을 능가하며 최첨단 클러스터링 결과를 달성했습니다. CIFAR-10 및 STL-10과 같은 소수의 카테고리를 가진 데이터셋에서는 프리트레인된 모델을 사용하지 않는 상태에서도 지도 방법의 성능에 근접한 정확도를 보이며, 프리트레인된 모델을 사용하는 지도 방법보다는 약간 낮은 성능을 보였습니다.



### Computational Trichromacy Reconstruction: Empowering the Color-Vision Deficient to Recognize Colors Using Augmented Reality (https://arxiv.org/abs/2408.01895)
- **What's New**: 이번 연구에서는 색각 결핍(Color Vision Deficiencies, CVD)을 가진 사람들이 색을 인식하고 이름을 붙이는 데 도움을 주는 보조 기술을 제안합니다. 제안된 시스템은 증강 현실(Augmented Reality, AR) 인터페이스로 스마트폰에서 구현되어 사용자들이 색각 변환을 통해 혼란스러운 색을 구별할 수 있게 도와줍니다.

- **Technical Details**: 이 연구는 색각 결핍자들이 2차원(2D)의 색 지각을 3차원(3D) 색 공간으로 재구성할 수 있는 시스템을 개발했습니다. 사용자는 앱에서 스와이프 제스처를 통해 색 공간 변환을 제어하며, 이는 원래 혼란스러운 색들을 구별 가능한 색으로 변환시킵니다. 이 새로운 3D 색 공간은 사용자들이 색의 이름을 배우고 인식하는 데 도움을 줍니다. 이 시스템은 스마트폰 기반의 AR 인터페이스로 구현되었으며 실시간 상호작용이 가능합니다.

- **Performance Highlights**: 심리물리학 실험과 장기 사용자 연구를 통해 이 시스템의 효과를 입증했습니다. 16명의 CVD 개인을 대상으로 한 심리물리학 실험에서 회전색 변화가 변별력을 가지고 있음을 확인했습니다. 또한 8명의 CVD 개인을 대상으로 한 9일간의 장기 연구에서는 색 이동 패턴을 학습함으로써 새로운 색을 인식할 수 있는 능력이 향상되었음을 확인했습니다. 실생활 시나리오(레고 블록 구축 및 예술 작품 해석)에서도 사용자가 긍정적인 경험을 보고했습니다.



### Is Generative Communication between Embodied Agents Good for Zero-Shot ObjectNav? (https://arxiv.org/abs/2408.01877)
- **What's New**: 이번 연구에서는 Zero-Shot ObjectNav 문제에서 효율적인 탐색을 위해 제한된 글로벌 뷰를 가진 오버헤드 에이전트(Overhead Agent)와 지상 에이전트(Ground Agent) 간의 협조를 강조하는 새로운 탐색 방식 두 가지를 제안하였습니다. Generative Communication(GC)을 활용한 Vision-Language Models(VLMs)가 지상 에이전트의 목표 객체 탐색 성능을 10% 향상시키는 효과를 확인했습니다.

- **Technical Details**: 지상 에이전트는 자연어 레이블로 지정된 목표 객체를 탐색할 때 환경에 대한 사전 정보를 사용하지 않습니다. 이 문제를 해결하기 위해 오버헤드 에이전트의 제한된 글로벌 뷰를 활용한 두 가지 협조 탐색 방식을 도입했습니다. 또한, 에이전트 간의 Generative Communication(GC)를 통해 탐색 효율성을 높였습니다. '프리엠티브 할루시네이션(Preemptive Hallucination)'이라는 독특한 현상을 식별했으며, 프롬프트 파인튜닝(Prompt Finetuning)을 통해 이 문제를 해결했습니다.

- **Performance Highlights**: 시뮬레이션 환경에서 GC를 이용한 경우, 지상 에이전트의 목표 객체 탐색 능력이 10% 향상되었습니다. 또한, 실제 환경에서 GC를 적용한 결과 질적 성능 향상 예시를 보여줌으로써 프리엠티브 할루시네이션 문제를 해결한 성과를 제시했습니다.



### Safe Semi-Supervised Contrastive Learning Using In-Distribution Data as Positive Examples (https://arxiv.org/abs/2408.01872)
- **What's New**: 이번 연구에서는 클래스 분포 불일치(class distribution mismatch) 상황에서도 안전한 준지도 학습(Safe Semi-Supervised Learning)을 위해 모든 비라벨(unlabeled) 데이터를 효과적으로 활용하는 방법을 제안하였습니다. 이는 셀프-슈퍼바이즈드 콘트라스티브 러닝(self-supervised contrastive learning, SSCL)의 개념을 도입하여 기존 모델의 클래스 분포 불일치 문제를 해결하고자 합니다.

- **Technical Details**: 연구팀은 SSCL 방법을 기반으로 인스턴스 구별(instance discrimination)을 통해 초기 네트워크 파라미터를 설정하고, 비라벨 OOD 데이터를 필터링하지 않으면서도 일반적인 데이터 표현을 학습할 수 있도록 합니다. 또한, 동일 클래스의 라벨된 네거티브 예시를 추가적인 포지티브 예시로 재지정하는 손실 함수 및 이를 위한 손실 계수 스케줄을 도입하여 보다 적절한 표현을 형성합니다.

- **Performance Highlights**: 제안된 방법의 성능을 평가하기 위해 CIFAR-10, CIFAR-100, Tiny ImageNet, 그리고 CIFAR-100과 Tiny ImageNet 혼합 데이터셋에서 다양한 불일치 비율 하의 실험을 수행하였습니다. 실험 결과, SSCL이 이미지 분류 정확도를 크게 개선하며, 인-디스트리뷰션(in-distribution) 예시를 모아 더 나은 표현을 형성함으로써 분류 정확도를 추가로 향상시켰습니다.



### NuLite -- Lightweight and Fast Model for Nuclei Instance Segmentation and Classification (https://arxiv.org/abs/2408.01797)
- **What's New**: 암 진단의 핵심인 Hematoxylin and Eosin (H&E) 슬라이드의 효율적이고 정확한 분석을 위한 새로운 CNN 아키텍처인 NuLite가 소개되었습니다. NuLite는 Fast-ViT에 기반한 U-Net 유사 아키텍처로, NuLite-S, NuLite-M, NuLite-H 세 가지 버전이 PanNuke 데이터셋에서 학습되었습니다.

- **Technical Details**: NuLite는 Fast-ViT 기반의 인코더와 HoVer-Net 자원 기준의 세 가지 디코더를 포함한 구조입니다. 이 디코더는 Nuclei 예측, 수평 및 수직 맵 예측, 및 Nuclei 분류를 수행합니다. 본 모델은 CellViT와 동등한 수준의 panoptic quality와 detection 성능을 발휘하면서도, NuLite-S는 파라미터 수가 40배 적고 GFlops이 8배 적습니다.

- **Performance Highlights**: NuLite는 CellViT보다 약 8배 빠르며, PanNuke와 같은 벤치마크 데이터셋에서 SOTA 성능을 입증했습니다. 추가적으로, MoNuSeg, CoNSeP, GlySAC와 같은 외부 데이터셋에서 높은 정밀도, 재현율, F1-score 메트릭을 기록하며 강력한 범용성을 보였습니다.



### Comparison of Embedded Spaces for Deep Learning Classification (https://arxiv.org/abs/2408.01767)
- **What's New**: 이 논문은 분류를 위한 임베디드 공간 (embedded space)을 설계하는 다양한 기술들을 간결히 개괄하고 있다. 특히 MNIST, Fashion MNIST, CIFAR-10 데이터셋에 대해 2차원 및 3차원 임베디딩을 시각적으로 구현하여 다양한 손실 함수와 네트워크 파라미터 제약을 비교하였다.

- **Technical Details**: 임베디드 공간은 신경망에 의해 학습된 고차원 입력 데이터의 저차원 표현이다. 이 논문은 소프트맥스 손실(softmax loss), 중심 손실(center loss), 각도 마진 손실(angular margin losses), 대조 손실(contrastive loss), 삼중되임 손실(triplet loss) 등 다양한 손실 함수와 네트워크 구성 요소 제약을 통해 임베디드 공간의 구조를 개선하는 방법들을 다루고 있다. 예제 신경망 구조는 Conv(32) -> Pool -> Conv(64) -> Pool -> Conv(128) -> Fully Connected(256) -> Fully Connected(2)으로 설명된다.

- **Performance Highlights**: 실험 결과, 임베디드 공간의 구조는 신경망 설계 및 손실 함수에 따라 다양하게 패턴화될 수 있다. 소프트맥스 손실과 같은 기초적인 접근법부터, 중심 손실 및 각도 기반 손실 등 고급 기법까지 다양한 방법들이 임베디드 공간의 해석 가능성과 설명 가능성을 향상시키는 데 기여한다는 점이 시각적으로 확인되었다.



### Visual-Inertial SLAM for Agricultural Robotics: Benchmarking the Benefits and Computational Costs of Loop Closing (https://arxiv.org/abs/2408.01716)
Comments:
          18 pages, 8 figures, 5 tables

- **What's New**: 이 논문은 농업 환경에서 Visual-Inertial SLAM 시스템을 벤치마킹하고 평가합니다. 주요 시스템으로는 ORB-SLAM3, VINS-Fusion, OpenVINS, Kimera, 그리고 SVO Pro가 포함됩니다. 특히 루프 클로징(loop closing)이 위치 추적의 정확성과 계산 요구사항에 미치는 영향을 중점적으로 분석했습니다.

- **Technical Details**: 이 연구는 다양한 프레임 레이트가 위치 추적의 정확성과 계산 부하에 미치는 영향을 평가했습니다. 농업 환경의 까다로운 조건, 예를 들면 조명 변동이나 기상 조건 등을 고려하여 각 시스템의 실효성을 분석했습니다. 또한, 외부 위치 시스템에 의존하지 않고 동적, 비구조적인 실외 환경에서 자율적으로 내비게이션하는 모바일 로봇에 적합한 SLAM 시스템을 중심으로 연구를 진행했습니다.

- **Performance Highlights**: 연구 결과는 루프 클로징이 위치 추적의 정확성을 향상시키는데 중요한 역할을 한다는 것을 보여주었습니다. 또한, 루프 클로징이 계산 자원을 효율적으로 관리하는 데도 도움이 된다는 점이 밝혀졌습니다. 이를 통해 Visual-Inertial SLAM 시스템을 실제 농업 로봇 응용에 최적화할 수 있는 귀중한 인사이트를 제공했습니다.



### Controllable Unlearning for Image-to-Image Generative Models via $\varepsilon$-Constrained Optimization (https://arxiv.org/abs/2408.01689)
Comments:
          40 pages, 54 figures

- **What's New**: 최근 생성 모델(generative models)의 놀라운 발전과 함께, 프라이버시 침해와 편향 문제와 같은 우려가 제기되고 있습니다. 이러한 문제를 해결하기 위해 머신 언러닝(machine unlearning)이 등장했으며, 본 논문에서는 이미지-이미지(I2I) 생성 모델에서 이를 연구하고 있습니다. 기존의 연구가 단일 목적 최적화 문제로 다루어졌다면, 본 논문에서는 사용자 기대에 따른 다양한 트레이드오프를 고려하는 컨트롤 가능한 언러닝 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 $	ext{control coefficient}$ $	ext{ε}$를 사용하여 트레이드오프를 제어합니다. 이는 $	ext{ε-constrained optimization problem}$으로 재구성되며, 경계 최적해(unlearning boundaries)를 찾기 위해 gradient-based 방법을 사용합니다. 이 경계 내에서는 모든 솔루션이 파레토 최적(Pareto optimality)을 보장합니다. 아울러, 다양한 컨트롤 함수에 따른 프레임워크의 수렴 속도를 분석하였습니다.

- **Performance Highlights**: 세 가지 주류 I2I 모델에 걸쳐 두 개의 벤치마크 데이터셋으로 수행한 광범위한 실험 결과, 제안된 컨트롤 가능한 언러닝 프레임워크가 그 효과성을 입증하였습니다.



### Multi-Frame Vision-Language Model for Long-form Reasoning in Driver Behavior Analysis (https://arxiv.org/abs/2408.01682)
Comments:
          On-going work

- **What's New**: 이 논문은 대시캠(dashcam) 영상을 기반으로 상업 운전자를 코칭할 수 있는 새로운 멀티모달(multimodal) 데이터셋과 운전 코칭 추론 시스템을 소개합니다. 현재 북미 대시캠 시장은 2022년에서 2027년까지 연평균 성장률(CAGR) 15.4%를 기록할 것으로 예상됩니다. 이 연구는 대규모 비전-언어 모델(Large-scale Vision Language Models, LVLMs)을 적용하여 운전자 행동을 분석하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구에서는 `Multi-Frame Vision-Language Model for Reasoning in Driver Behavior Analysis`라는 모델을 제안합니다. 이 모델은 도로를 향한 카메라와 운전자를 향한 카메라에서 촬영된 영상을 모두 활용하여 운전 상황을 포괄적으로 분석합니다. 두 종류의 카메라를 사용하여 외부 조건과 운전자의 반응을 모두 식별하며, 이를 통해 운행 위험 상황을 효과적으로 이해할 수 있게 합니다. 이 모델은 Video-LLaMA 프레임워크를 기본으로 하며, 비주얼 인코더는 BLIP-2를, 오디오 인코더는 ImageBind를 사용합니다.

- **Performance Highlights**: 모델 훈련에는 로드-페이싱과 드라이버-페이싱 카메라의 동기화된 RGB 비디오 데이터셋이 사용되었습니다. 모델은 NVIDIA A100 (80GB) GPU 8개로 훈련되며, Visual Encoder와 Large Language Model(LLM)의 가중치는 동결시키고 비디오 Qformer의 가중치를 업데이트하는 방식으로 파인 튜닝(fine-tuning)되었습니다. 결과적으로, 모델은 운전 상황에 대한 포괄적 이해와 정교한 코칭 지침 생성 능력을 크게 향상시켰습니다.



### Zero-Shot Surgical Tool Segmentation in Monocular Video Using Segment Anything Model 2 (https://arxiv.org/abs/2408.01648)
Comments:
          The first work evaluates the performance of SAM 2 in surgical videos

- **What's New**: Segment Anything Model 2 (SAM 2)는 이미지와 비디오 분할의 최신 세대 모델입니다. SAM 2는 확장된 데이터셋인 Segment Anything Video (SA-V) 데이터셋을 사용하여 학습하였으며, 이 데이터셋은 50,900여 개의 비디오에 걸쳐 35.5백만 개의 마스크를 포함하고 있습니다. SAM 2는 제로샷 분할(Zero-Shot Segmentation)을 통해 다양한 프롬프트(예: 점, 박스, 마스크)를 지원한다는 점에서 이전 모델보다 발전된 기능을 자랑합니다.

- **Technical Details**: SAM 2는 이미지와 비디오 분할을 위해 설계되었으며, 추가적인 레이블이 없는 상태에서도 다양한 방법을 통해 분할을 수행할 수 있는 제로샷 성능을 보입니다. SAM 2 모델은 주로 수술 도구 분할(Surgical Tool Segmentation)에 사용되며, 이는 레이블된 데이터가 부족하고 수술 절차가 다양하기 때문에 특히 유용합니다. 본 연구에서는 SAM 2 모델의 제로샷 비디오 분할 성능을 내시경 및 현미경을 포함한 다양한 수술 유형에서 평가하였습니다.

- **Performance Highlights**: 1) SAM 2는 다양한 수술 비디오를 분할하는 데 강력한 능력을 보여주었습니다. 2) 새로운 도구가 화면에 등장할 때 추가적인 프롬프트가 필요하다는 것을 발견했습니다. 3) 수술 비디오에 내재된 특정 도전 과제가 SAM 2의 견고성에 영향을 미칠 수 있음을 확인했습니다.



### MedUHIP: Towards Human-In-the-Loop Medical Segmentation (https://arxiv.org/abs/2408.01620)
- **What's New**: 새로운 접근 방법으로 **불확실성 인지 모델(uncertainty-aware model)**과 **human-in-the-loop(사람-기반) 상호작용**을 통합하여 의료 영상 분할(medical image segmentation)의 불확실성을 다루고 있습니다. 이 모델은 여러 가능한 분할을 제안하여 불확실성을 해결하고, 임상가의 감독 하에 상호작용적으로 분할을 수정합니다. 이는 알고리즘의 정밀성과 임상가의 지식을 균형 있게 조화시켜, 안전한 의료 영상 분할을 촉진합니다.

- **Technical Details**: 우리의 모델 MedUHIP은 **Sampling Net 모듈**을 사용하여 임상가의 선호도를 학습하고, 이를 통해 분할 샘플링 공간을 조정합니다. 이러한 샘플링 공간에서 여러 분할을 예측하여 의료 영상의 불확실성을 반영합니다. **human-in-the-loop 상호작용**을 통해 임상가의 피드백을 수렴하고 최종 예측을 전문 임상가가 직접 사용하는 데 적합하게 만듭니다. MedUHIP은 REFUGE2, LIDC-IDRI, QUBIQ와 같은 여러 공용 다임상가 주석 데이터셋을 평가하여 우수한 성능을 입증했습니다.

- **Performance Highlights**: MedUHIP은 다양한 결정적(deterministic) 모델과 불확실성 인지 모델을 능가하는 우수한 분할 성능을 나타냈습니다. 또한, 이전 상호작용 모델보다 상호작용 횟수를 줄여 더 나은 결과를 도출했습니다. 코드는 향후 연구 촉진을 위해 공개될 예정입니다.



### On Validation of Search & Retrieval of Tissue Images in Digital Pathology (https://arxiv.org/abs/2408.01570)
- **What's New**: 이 논문은 의료 영상의 진단, 치료 계획 및 질병 모니터링에 있어서의 중요성을 강조합니다. 특히, 영상 기반 이미지 검색(Content-Based Image Retrieval, CBIR) 시스템이 의료 분야에서 어떻게 활용될 수 있는지를 논의합니다.

- **Technical Details**: 논문은 방사선학과 병리학이 정확한 이미지 해석에 얼마나 의존하는지 설명합니다. 구체적으로, 방사선 전문의가 X-레이, CT 스캔, MRI 등을 통해 골절에서 암에 이르기까지 다양한 상태를 진단하고, 병리학자는 현미경과 디지털 이미지를 사용해 세포 비정상(세포 이상, cellular abnormalities)을 진단합니다. CBIR 시스템은 시각적 콘텐츠를 기반으로 이미지를 검색하고 검색하는 방식으로, 진단 정확성을 높이는 데 도움을 줍니다.

- **Performance Highlights**: 이미지 검색 엔진의 포괄적인 검증을 위해, 정확도(accuracy), 색인(indexing), 검색 시간(search times), 저장 오버헤드(storage overhead)와 같은 성능 지표들을 평가하는 것이 중요합니다. 최근의 검증 결과에 따르면, 이러한 평가를 통해 신뢰할 수 있고 효율적인 검색 결과를 제공할 수 있습니다. 특히, 조직병리학(histopathology)에서 효율적인 검색을 통해 진단 정확도가 향상되었습니다.



### Enhanced Knee Kinematics: Leveraging Deep Learning and Morphing Algorithms for 3D Implant Modeling (https://arxiv.org/abs/2408.01557)
- **What's New**: 정형외과 수술 및 생체의공학에서 중요한 임플란트 무릎 모델의 정확한 재구성을 위해, 새로운 연구가 진행되었습니다. 이 연구에서는 기계 학습(Machine Learning, ML) 알고리즘과 변형(morphing) 기법을 활용하여 자동으로 3D 재구성을 수행하는 방법이 제안되었습니다. 이는 기존의 수동 분할 방식의 번거로움과 오류를 줄이고, 전처리 데이터를 통해 더 높은 정확성을 보장합니다.

- **Technical Details**: 방법론은 사전 수술 이미지 데이터, 예를 들어, 환자의 무릎 관절을 촬영한 X선 또는 형광 투시 이미지를 수집하는 것으로 시작됩니다. 그런 다음 컨볼루션 신경망(Convolutional Neural Network, CNN)이 훈련되어 임플란트 구성 요소의 대퇴골 윤곽을 자동으로 분할합니다. 이는 수작업을 크게 줄이고 높은 정확도를 보장합니다. 이후, 변형 알고리즘이 분할 데이터를 사용하여 개인 맞춤형 3D 무릎 관절 모델을 생성합니다. 이 알고리즘은 임플란트의 위치, 크기 및 방향을 고려하여 무릎 관절의 형상을 시뮬레이션 합니다.

- **Performance Highlights**: 제안된 방법의 효과는 정량적 평가를 통해 입증되었으며, 실제 데이터 및 기존 기술과의 비교가 포함되었습니다. 다양한 임플란트 유형을 포함한 19개의 테스트 케이스에서 ML 기반 분할 방법은 수작업 분할에 비해 뛰어난 정확성과 일관성을 보였으며, 평균 RMS 에러는 0.58 +/- 0.14 mm로 나타났습니다. 이러한 연구는 정형외과 수술의 발전에 크게 기여하며, 자동화된 임플란트 무릎 모델 재구성을 위한 견고한 프레임워크를 제공합니다.



### Robot-Enabled Machine Learning-Based Diagnosis of Gastric Cancer Polyps Using Partial Surface Tactile Imaging (https://arxiv.org/abs/2408.01554)
- **What's New**: 이번 연구에서는 Vision-based Tactile Sensor (VTS)와 이를 보완하는 머신러닝(ML) 알고리즘을 사용하여 고급 위암(AGC) 종양 진단의 기존 한계를 처음으로 해결하려고 합니다. VTS를 사용하여 자동 데이터 수집이 가능하며, 이를 통해 데이터 부족 문제와 전통적인 ML 접근 방식에서 발생하는 편향을 해결할 수 있습니다. 또한, 새롭게 개발된 로봇 매니퓰레이터와 3D 프린팅된 AGC 종양 모형을 활용하여 고해상도 질감 이미지를 수집하고 ML 모델을 훈련시킵니다.

- **Technical Details**: 본 연구에서 사용된 VTS는 HySenSe라 불리는 최근 개발된 센서로, 실리콘 막, 아두캠(Arducam) 카메라, 아크릴 판, RGB LED 등으로 구성되어 있습니다. 이 센서는 AGC 종양의 표면 텍스처와 경도를 시각적으로 캡처할 수 있으며, 로봇 시스템과 함께 사용하여 종양 표면의 전체 데이터를 자동으로 수집합니다. AGC 종양의 종류를 Borrmann의 분류 체계를 따라 총 4가지 타입으로 구분하였습니다.

- **Performance Highlights**: 제안된 ML 모델은 합성 데이터로 훈련되었으며 혼합된 형태적 특성과 부분적인 센서 접촉 조건에서도 기존의 ML 모델과 비교하여 우수한 성능을 보였습니다. 각종 통계적 지표를 사용하여 성능을 평가하였으며, 새로운 ML 기반 진단 도구는 다양한 AGC 종양의 특성을 민감하게 분류할 수 있음을 보여주었습니다.



### Contextual Cross-Modal Attention for Audio-Visual Deepfake Detection and Localization (https://arxiv.org/abs/2408.01532)
- **What's New**: 디지털 시대에 들어서면서, 멀티모달(manipulation) 기반의 deepfake와 같은 합성 미디어가 사회 및 정치적 통합성을 위협하고 있습니다. 이번 논문에서는 오디오-비주얼 deepfake 탐지를 위해 문맥적 정보를 활용한 순환 신경망(RNN) 기반의 새로운 멀티모달 어텐션 프레임워크를 제안합니다. 이 접근법은 오디오 및 비디오 신호의 다중 시퀀스 표현에 어텐션을 적용하여 deepfake 탐지 및 로컬라이제이션 성능을 향상시킵니다.

- **Technical Details**: 기존의 멀티모달 deepfake 탐지기는 종종 이질적 데이터 스트림의 어텐션 기반 융합에 의존합니다. 하지만 데이터의 이질적 특성(예: 오디오 및 비주얼 신호)으로 인해 효과적인 융합에 어려움이 있습니다. 본 연구는 이런 문제를 해결하기 위해 재발성 신경망 기반 멀티모달 어텐션 프레임워크를 도입했습니다. 이 접근법은 멀티모달 다중 시퀀스 표현에서 기여하는 특징을 학습하여 deepfake 감지 및 로컬라이제이션을 구현합니다.

- **Performance Highlights**: FakeAVCeleb, AV-Deepfake1M, TVIL, LAV-DF 등 여러 오디오-비주얼 deepfake 데이터셋에 대한 실험적 검증 결과, 제안된 접근법이 기존 연구에 비해 탐지 정확도가 3.47% 향상되었고 정밀도가 2.05% 향상된 것으로 나타났습니다. 이는 현재까지 발표된 가장 높은 성능을 기록한 것입니다.



### Estimating Environmental Cost Throughout Model's Adaptive Life Cyc (https://arxiv.org/abs/2408.01446)
Comments:
          Accepted in the AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2024

- **What's New**: 이번 연구에서는 PreIndex라는 예측 인덱스를 도입하여 모델 재학습 시 에너지 소비 및 탄소 배출량을 추정할 수 있는 방법을 제안합니다. PreIndex는 배포 환경이나 입력 데이터의 변화에 따라 모델을 지속적으로 재사용하여 AI 및 딥러닝과 관련된 탄소 발자국을 줄이는 사회적으로 유익한 접근 방법입니다.

- **Technical Details**: PreIndex는 데이터의 한 번의 forward pass로 환경 비용(탄소 배출량 및 에너지 사용량)을 추정할 수 있으며, 에폭(epoch), gradient norm, 모델 파라미터 변화의 크기 등 딥러닝과 관련된 기타 자원 지표들도 예측할 수 있습니다. 다양한 데이터셋과 모델 구조, 분포 변동의 유형 및 강도에 관계없이 사용할 수 있습니다.

- **Performance Highlights**: PreIndex는 데이터의 분포 변동에 따른 재학습에 관련된 자원을 추정할 수 있는 단일 값을 제공함으로써, 사용자가 재학습 결정에서 가장 비용 효율적이고 지속 가능한 옵션을 선택할 수 있게 도와줍니다. 이를 통해 환경에 미치는 영향을 최소화하면서 모델 재사용을 가능하게 합니다.



### Adding Multimodal Controls to Whole-body Human Motion Generation (https://arxiv.org/abs/2407.21136)
- **What's New**: 새로운 논문 ControlMM은 텍스트, 음성, 음악 등의 다양한 조건 모달리티(text, speech, music)로 제어되는 전신 멀티모달 모션 생성(whole-body multimodal motion generation)을 위한 통합 프레임워크를 제안합니다. 이는 플러그 앤 플레이(plug-and-play) 방식으로 동작하며, 특히 여러 모션 배포 시나리오의 분포 드리프트(motion distribution drift)와 다양한 그래뉼러리티(granularity)의 조건 최적화 문제를 해결하고자 합니다.

- **Technical Details**: ControlMM은 두 가지 주요 요소로 구성됩니다. 첫째, ControlMM-Attn을 사용하여 정적 및 동적 인간 토폴로지 그래프(human topology graphs)의 병렬 모델링을 통해 모션 지식을 학습하고 전이합니다. 둘째, 텍스트 기반 모션의 조잡한 생성(coarse semantic generation)부터 시작해 더 낮은 수준의 세분화된 조건(multimodal control adaptation)을 다루는 두 단계 훈련 전략을 채택합니다. 마지막으로, 통일된 표준 형식의 벤치마크(ControlMM-Bench)를 도입하여 다양한 기존 데이터셋의 불일치 문제를 해결합니다.

- **Performance Highlights**: 실험 결과에 따르면 ControlMM은 텍스트-모션(text-to-motion), 음성-제스처(speech-to-gesture), 음악-댄스(music-to-dance) 등의 다양한 표준 모션 생성 작업에서 최신 성능(state-of-the-art performance)을 달성했습니다. 이를 통해 모델 디자인, 훈련 전략 및 확장 효과에 대한 중요한 통찰을 제공합니다.



### MSA$^2$Net: Multi-scale Adaptive Attention-guided Network for Medical Image Segmentation (https://arxiv.org/abs/2407.21640)
Comments:
          Accepted at BMVC 2024. Supplementary materials included at the end of the main paper (3 pages, 2 figures, 1 table)

- **What's New**: 의학 이미지 분할에서 CNN(Convolutional Neural Networks)의 한계를 극복하고자 MSA$^2$Net이라는 새로운 딥 러닝 분할 프레임워크를 소개했습니다. 이 모델은 다중 스케일 적응 공간적 주의 게이트(Multi-Scale Adaptive Spatial Attention Gate, MASAG)를 도입하여 지역 및 전역 특징 정보를 동적으로 조정하고, 세부적 특징과 넓은 의미 요소를 결합합니다.

- **Technical Details**: MSA$^2$Net은 인코더와 디코더 간의 스킵 연결(skip-connections) 디자인을 개선하여 특징 융합을 촉진합니다. 특히 MASAG는 수용 필드(receptive field)를 동적으로 조정해 공간적으로 관련 있는 특징을 강조하고 불필요한 배경을 최소화합니다. 디코더의 초기 층에서는 Large Kernel Attention (LKA) 모듈을 사용하고, 깊은 층에서는 Dual Attention Enhanced Transformer (DAE-Former) 블록을 통합하여 전역 정보와 지역 정보를 효과적으로 균형있게 처리합니다.

- **Performance Highlights**: 피부과 및 방사선학 데이터세트를 포함한 광범위한 평가에서, MSA$^2$Net은 최신 연구(State-of-the-art, SOTA)를 능가하거나 대등한 성능을 보여주었습니다. 주요 데이터세트로는 ISIC2018 (피부과)와 Synapse (방사선학) 데이터세트가 포함됩니다. 또한 해당 코드도 공개되었습니다.



New uploads on arXiv(cs.AI)

### Backward explanations via redefinition of predicates (https://arxiv.org/abs/2408.02606)
- **What's New**: 이 논문에서는 하위-설명 가능 인공지능(xAI)의 하위 분야인 강화학습(RL)을 설명하는 새로운 접근법인 Backward-HXP (B-HXP)를 제안합니다. 기존의 HXP 방법은 복잡한 계산 때문에 긴 역사(history)를 설명하는 데 한계가 있었지만, B-HXP는 이러한 한계를 극복하고 긴 역사에서도 중요한 행동을 정확하게 설명할 수 있습니다.

- **Technical Details**: 강화학습 문제는 마르코프 결정 프로세스(Markov Decision Process)를 통해 모델링되며, 상태 공간(𝒮), 행동 공간(𝒜), 보상 함수(R), 전이 함수(p)로 구성됩니다. 정책(π)은 각 상태에서 수행할 행동을 결정하며, 결정론적 정책을 설명합니다. HXP는 상태와 행동의 쌍으로 구성된 역사를 특정 술어(predicate)로 분석하여 설명합니다. 이 논문에서는 HXP의 계산 복잡성 때문에 제기되는 문제를 해결하기 위해 Backward-HXP를 제안합니다. B-HXP는 역사를 분석하여 중요한 행동을 결정하는 새로운 방법입니다.

- **Performance Highlights**: 실험 결과, B-HXP는 기존의 HXP 방법보다 긴 역사에서도 더 효과적으로 중요한 행동을 설명할 수 있음을 입증했습니다. 세 가지 문제에 대한 실험에서 B-HXP는 높은 정확도로 중요한 행동을 식별하고 설명할 수 있었습니다.



### Counterfactual Shapley Values for Explaining Reinforcement Learning (https://arxiv.org/abs/2408.02529)
- **What's New**: 새로운 논문에서는 강화 학습(RL)의 설명 가능성을 개선하기 위해 반사실적(카운터팩츄얼) 분석과 Shapley Value를 통합한 Counterfactual Shapley Values(CSV) 접근법을 소개합니다. 이 방법론은 다양한 행동 선택에 대한 각 상태 차원의 기여도를 계량화하고 비교하는 것을 목표로 합니다.

- **Technical Details**: CSV 방법론은 새로운 특성 값 함수인 'Counterfactual Difference Characteristic Value'와 'Average Counterfactual Difference Characteristic Value'를 도입하여 Shapley Value를 계산합니다. 이 함수들은 최적 행동과 비 최적 행동 간의 기여도 차이를 평가하는 데 도움을 줍니다. Shapley Value는 협력 게임에서 각 개인의 총 보상에 대한 기여도를 측정하는 방법입니다. 실제 반사실적 설명과 Shapley Value의 통합은 RL 도메인에서 처음 시도되었습니다.

- **Performance Highlights**: CSV 방법은 GridWorld, FrozenLake, Taxi와 같은 여러 RL 도메인에서 테스트되었으며, 실험 결과는 이 방법이 복잡한 RL 시스템에서 투명성을 향상시키고 다양한 결정 간의 차이를 계량화할 수 있음을 보여줍니다. 이로 인해 이전 설명 방법의 한계를 크게 극복할 수 있었습니다.



### Perfect Information Monte Carlo with Postponing Reasoning (https://arxiv.org/abs/2408.02380)
Comments:
          Accepted in IEEE Conference on Games (CoG) 2024 + Appendix

- **What's New**: 이 연구는 새로운 온라인 알고리즘 'Extended Perfect Information Monte Carlo' (EPIMC)를 소개합니다. 이 알고리즘은 현재의 최첨단 결정화 기반 접근법인 Perfect Information Monte Carlo (PIMC)를 개선하여 전략 융합 문제를 완화합니다.

- **Technical Details**: EPIMC는 완전 정보 해결을 미루어 전략 융합 문제를 완화합니다. 이는 결정화 기반 알고리즘에서 일반적으로 발생하는 완전 정보로의 전환과 관련된 문제를 해결하기 위한 새로운 접근법입니다. 새로운 알고리즘은 리프 평가자를 나중으로 연기하며, 이는 이전 수준의 추론과 새로운 결정의 상호작용과 같은 새로운 고려 사항을 도입합니다.

- **Performance Highlights**: 실험적 분석에서 EPIMC는 다양한 게임에서 특히 전략 융합이 게임 플레이에 큰 영향을 미치는 게임에서 현저한 성능 향상을 보였습니다. 이 연구는 전략 융합과 관련된 결정화 기반 알고리즘의 이론적 토대에도 기여합니다.



### Operationalizing Contextual Integrity in Privacy-Conscious Assistants (https://arxiv.org/abs/2408.02373)
- **What's New**: 최신 AI 어시스턴트는 고급 대형 언어 모델(LLM)과 도구 접근을 결합한 시스템으로, 사용자를 대신해 복잡한 작업을 자율적으로 수행할 수 있습니다. 사용자 정보(예: 이메일, 문서)에 대한 접근으로 어시스턴트의 유용성이 크게 증가할 수 있지만, 이는 부적절한 정보 공유에 대한 프라이버시 우려를 증가시킵니다. 이러한 문제를 해결하기 위해 이 논문에서는 'contextual integrity(문맥적 무결성)' 개념을 도입하여, 어시스턴트가 프라이버시 기대를 충족하도록 유도하는 전략을 제안하고 평가합니다.

- **Technical Details**: 문맥적 무결성(CI) 이론은 특정 맥락에서 정보의 적절한 흐름을 프라이버시로 정의합니다. 이 연구는 어시스턴트의 정보 공유 행동이 해당 맥락의 정보 규범을 준수하도록 유도하는 여러 전략을 설계하고 평가했습니다. 평가를 위해 합성 데이터와 인간 주석이 포함된 새로운 양식 채우기 벤치마크를 사용했으며, 선진형 LLM이 CI 기반 추론을 수행하도록 유도한 결과 강력한 성과를 보여줬습니다.

- **Performance Highlights**: 전통적인 LLM 시스템의 취약성(adversarial examples, jailbreaking, prompt injection)에도 불구하고, 양식 채우기 작업에서 IFC(Information Flow Card) 기반 추론을 수행한 결과 프라이버시와 유용성 측면에서 다른 대안보다 뛰어난 성능을 보였습니다. 이는 정보 공유 어시스턴트가 사용자의 프라이버시 기대를 충족하는 데 있어 CI의 중요성을 입증합니다.



### On the consistent reasoning paradox of intelligence and optimal trust in AI: The power of 'I don't know' (https://arxiv.org/abs/2408.02357)
Comments:
          12 pages and 50 pages of supplementary material, 7 figures

- **What's New**: 연구자는 ‘일관된 추론 패러독스(Consistent Reasoning Paradox, CRP)’를 소개했습니다. 인간 지능의 핵심인 일관된 추론이 인간과 같은 오류 가능성을 암시함을 제시합니다. 이는 특정 문제에 대해 항상 일관되게 대답하는 AI가 무한히 잘못된 답변(환각)을 생성할 것이라는 주장을 포함하고 있습니다.

- **Technical Details**: CRP는 기본 산술 문제와 같은 문제에서 AI가 인간 지능을 모방하여 일관된 추론을 하는 경우 잘못된 답변을 무한히 생성할 수 있다는 것을 지적합니다. 반면, 일관된 추론을 하지 않는 AI는 같은 문제에서 더 정확한 답변을 할 수 있습니다. CRP는 이러한 환각을 탐지하는 것이 원래 문제를 해결하는 것보다 더 어렵다고 주장합니다.

- **Performance Highlights**: 연구에 따르면, 어떤 문제에 대해 AI가 올바른 답변을 제공할 수는 있지만, 그 답변에 도달한 논리적 설명을 정확히 제공할 수는 없습니다. 따라서 신뢰할 수 있는 AI가 되기 위해서는 '모르겠다(I don't know)' 기능을 포함해야 한다고 합니다. 현재의 AI는 이 기능을 가지고 있지 않기 때문에 이러한 문제를 해결하는 데 어려움이 있습니다.



### Developing PUGG for Polish: A Modern Approach to KBQA, MRC, and IR Dataset Construction (https://arxiv.org/abs/2408.02337)
Comments:
          Accepted for ACL 2024 (findings)

- **What's New**: AI와 자연어 처리의 발전은 기계와 인간 간의 언어 상호작용을 혁신적으로 개선했습니다. 특히, Knowledge Base Question Answering (KBQA) 시스템은 구조화된 지식 그래프(KG)를 활용하여 광범위하고 지식 집약적인 질문을 처리할 수 있습니다. 그러나 저자원(low-resource) 언어를 위한 KBQA 데이터셋은 여전히 부족합니다. 이를 해결하기 위해, 최신 도구인 Large Language Models(LLM)를 활용한 현대적이고 반자동화된 데이터셋 구축 방법론이 도입되었습니다. 이를 통해, 최초의 폴란드어 KBQA 데이터셋인 PUGG 데이터셋이 만들어졌으며, 기계 읽기 이해(MRC)와 정보 검색(IR) 데이터셋도 새로이 소개되었습니다.

- **Technical Details**: 기존의 KBQA 데이터셋 구축 방식은 구식이며 인적 자원이 많이 필요했으나, 본 연구팀은 이 문제를 해결하기 위해 LLM을 활용한 반자동화된 데이터셋 구축 파이프라인을 설계, 구현 및 실행하였습니다. 이는 특별히 저자원 환경에 맞춰진 것으로, Wikidata를 KG로 선택하여 다국어 커버리지와 동적, 개방적 특성을 활용했습니다. 또한, 데이터 변환 과정에서 자연스러운 언어와 단순 템플릿 기반 질문을 결합하여 데이터셋의 복잡성을 다루었습니다.

- **Performance Highlights**: PUGG 데이터셋은 폴란드어 KBQA, MRC, IR 과제들을 포함하며, 첫 번째 폴란드어 KBQA 리소스로 자리매김했습니다. 데이터셋은 자연어로 된 사실 질문(factoid questions)을 특징으로 하며, 보다 효율적이고 긴밀한 데이터셋 생성을 위해 반자동화된 파이프라인을 사용했습니다. 이 파이프라인은 인간 주석 작업을 크게 줄이는 데 기여했습니다. 뿐만 아니라, 파이프라인 구현과 데이터셋 구축 과정에서 얻은 통계 및 통찰을 자세히 제공하였으며, 다양한 컨텍스트에서 유용한 엔티티 연계(custom methods)와 같은 유틸리티 메서드를 개발했습니다. 마지막으로, 기존 모델의 평가를 제공하여 향후 연구를 위한 벤치마크를 설정했습니다.



### SR-CIS: Self-Reflective Incremental System with Decoupled Memory and Reasoning (https://arxiv.org/abs/2408.01970)
- **What's New**: 인간의 기억 및 학습 메커니즘에서 영감을 받아, 새로운 정보를 빠르게 학습하면서도 기존 메모리를 유지할 수 있는 SR-CIS(Self-Reflective Complementary Incremental System)를 제안합니다. 이 시스템은 Complementary Inference Module (CIM)과 Complementary Memory Module (CMM)로 구성되어 있으며, 효율적인 협업을 위해 CA-OAD(Confidence-Aware Online Anomaly Detection) 메커니즘을 도입했습니다.

- **Technical Details**: SR-CIS는 빠른 추론을 위한 작은 모델과 신중한 추론을 위한 큰 모델로 구성된 CIM과 과제로부터의 기억을 단기 및 장기 메모리로 구분한 CMM으로 구성되어 있습니다. CA-OAD 메커니즘은 예측의 신뢰도를 온라인으로 평가하여 하드 샘플을 감지합니다. 단기 메모리는 특정 태스크의 LoRA (Low-Rank Adaptive) 및 프로토타입 가중치와 편향으로 구성되며, 장기 메모리는 다양한 태스크 메모리를 통합하여 저장합니다. 훈련 시, 이미지의 텍스트 설명을 저장하고, 시나리오 재생 모듈(SRM)을 사용하여 메모리 결합을 수행합니다.

- **Performance Highlights**: SR-CIS는 한정된 저장 공간과 데이터 리소스 제약 아래에서 모델의 플라스틱성과 메모리 안정성을 균형 있게 유지합니다. 표준 및 few-shot class incremental learning 벤치마크에서 기존의 경쟁 벤치마크들을 상회하는 성능을 보였습니다.



### Visual Grounding for Object-Level Generalization in Reinforcement Learning (https://arxiv.org/abs/2408.01942)
Comments:
          35 pages, 14 figures, 17 tables

- **What's New**: 이번 연구에서는 시각-언어 모델(VLM)을 활용하여 객체 중심 작업에서 새로운 객체와 지시에 대한 제로샷 일반화(Zero-shot Generalization)을 가능하게 하는 방법을 제안합니다. 이를 통해 자연어 지시를 따르는 에이전트의 일반화 능력을 향상시키고자 합니다.

- **Technical Details**: 기술적으로, 본 연구는 마인크래프트(Minecraft) 환경에서 MineCLIP을 통해 시각-언어 지식을 강화를 위한 학습(Reinforcement Learning, RL)에 전달합니다. 주로 두 가지 경로로 이를 수행합니다. 첫째, 목표 객체에 대한 자신감 맵(confidence map)에서 유도된 객체-기반 내재적 보상(intrinsic reward)을 제안하여 에이전트를 목표 객체로 더 효과적으로 안내합니다. 둘째, 자신감 맵을 에이전트의 정책 입력으로 통합하여 객체 레벨에서 제로샷 일반화를 가능하게 합니다.

- **Performance Highlights**: 싱글 태스크 실험에서, 객체-기반 내재적 보상이 MineCLIP 보상보다 도전적인 스킬 학습에서 더 뛰어난 성능을 보임을 확인했습니다. 다중 태스크 실험에서는 자신감 맵을 태스크 표현(Task Representation)으로 사용하는 에이전트가 언어 기반 방법보다 더 나은 일반화 능력을 보였습니다. 새로운 객체에 대한 성공률은 사냥 분야에서 약 300%, 수확 분야에서 약 100% 향상되었습니다.



### MAO: A Framework for Process Model Generation with Multi-Agent Orchestration (https://arxiv.org/abs/2408.01916)
- **What's New**: 이번 논문에서는 다중 에이전트 오케스트레이션(MAO) 프레임워크를 통해 자동적으로 프로세스 모델을 생성하는 방법을 탐구합니다. 이 방법은 프로세스 모델링의 효율성을 높이고 도메인 전문가들에게 유용한 통찰력을 제공합니다.

- **Technical Details**: MAO 프레임워크에서는 대형 언어 모델(LLM)을 다중 에이전트의 핵심으로 활용하여 혁신적인 프롬프트 전략을 통해 에이전트 간의 효율적인 협력을 보장합니다. 프레임워크의 구성 요소는 다음과 같습니다: 1) 초안 생성(Generation): 텍스트 설명에서 대략적인 프로세스 모델을 생성, 2) 세부 조정(Refinement): 에이전트 간의 다중 라운드 대화를 통해 프로세스 모델을 지속적으로 개선, 3) 검토(Reviewing): 다중 턴 대화에서 발생할 수 있는 'hallucination' 현상을 검토하고 수정, 4) 테스트(Testing): 외부 도구를 사용하여 형식 오류 (format hallucinations)를 검출하고 수정하여 출력 파라다임과 일치하도록 조정.

- **Performance Highlights**: 실험 결과, MAO 프레임워크가 생성한 프로세스 모델이 기존 방법보다 뛰어난 성과를 보여주었으며, 4개의 다양한 데이터셋에서 수동 모델링을 각각 89%, 61%, 52%, 및 75% 초과했습니다.



### Walk Wisely on Graph: Knowledge Graph Reasoning with Dual Agents via Efficient Guidance-Exploration (https://arxiv.org/abs/2408.01880)
- **What's New**: 최근 발표된 FULORA 모델은 계층적 강화 학습(HRL)을 기반으로 한 이중 에이전트 구조를 활용하여 지식 그래프(KG)의 멀티 홉(multi-hop) 추론에 있어 이전 접근 방식의 한계를 극복합니다. 특히, 초기 단계의 희소한 보상 문제와 긴 추론 경로를 필요로 하는 희소한 지식 그래프에서의 성능을 개선합니다.

- **Technical Details**: FULORA는 이중 에이전트를 사용하여 효율적인 가이드 및 탐색을 수행합니다. 상위 에이전트(GIANT)는 단순화된 KG 위에서 단계별 힌트를 제공하고, 하위 에이전트(DWARF)는 원래 KG 위에서 이 힌트를 기반으로 최적 경로를 찾습니다. DWARF는 보상 최대화 및 상위 에이전트의 효율적인 가이드를 통합한 가치 함수를 최적화합니다.

- **Performance Highlights**: FULORA는 세 가지 실제 지식 그래프 데이터셋에서 이전의 RL 기반 모델들보다 뛰어난 성능을 보였습니다. 특히 긴 거리 추론이 필요한 경우에 두드러진 성과를 나타냅니다.



### Review of Cloud Service Composition for Intelligent Manufacturing (https://arxiv.org/abs/2408.01795)
- **What's New**: 이 논문에서는 지능형 제조 플랫폼의 지속 가능성을 위해 클라우드 서비스 최적화(cloud service optimization)의 프로세스를 요약하고, 기존 연구에서 분산된 최적화 지표와 비표준화된 정의의 문제를 해결하기 위해 3명의 참가자의 요구사항을 고려한 11개의 최적화 지표를 정의하고 있습니다. 또한, 서비스 최적화 알고리즘을 휴리스틱(heuristic)과 강화 학습(reinforcement learning) 두 가지 범주로 분류하여 비교하였습니다.

- **Technical Details**: 지능형 제조는 실시간 데이터 수집, 분석 및 처리를 포함하여 정보 기술을 활용하여 제조 공정을 지능화, 자동화 및 디지털화하는데 주력합니다. 이를 통해 자원 배치 최적화, 생산 품질 향상, 비용 절감, 제품 혁신 및 전달 속도를 빠르게 하고 지속 가능한 발전을 촉진합니다. 클라우드 서비스 최적화는 특정 요구사항, 조건 및 제약 조건에 따라 최적의 서비스 조합을 선택하는 과정을 포함하며, 이는 다기능 제조 작업(multi-functional manufacturing tasks)의 시퀀셜(sequential), 병렬(parallel), 선택(selective), 순환(circular) 구조에 따라 이루어집니다.

- **Performance Highlights**: 기술적 목표로 비용 절감 및 시간 최적화를 주로 강조하며 신뢰성, 품질, 명성 등의 요소를 보조 지표로 고려합니다. 휴리스틱 알고리즘에서 강화 학습 알고리즘으로의 전환은 더욱 고도화된 서비스 최적화 방법을 제공합니다. 이를 통해 사용자 만족도를 높이고 제조 플랫폼의 경쟁력을 강화할 수 있습니다.



### Integrating Large Language Models and Knowledge Graphs for Extraction and Validation of Textual Test Data (https://arxiv.org/abs/2408.01700)
Comments:
          Paper Accepted at ISWC 2024 In-Use Track

- **What's New**: 이 연구는 대형 항공우주 제조 회사가 제한된 수량의 높은 복잡성 제품을 설계, 개발, 통합, 검증, 검증함에 있어 문서화된 데이터를 추출하고 검증하는 새로운 하이브리드 방법론을 제시합니다. 이 방법론은 Knowledge Graphs (KGs)와 Large Language Models (LLMs)를 결합하여 특정 사례 연구로 위성의 전자 보드와 관련된 테스트 데이터를 분석했습니다.

- **Technical Details**: 제안된 방법론은 확장된 Semantic Sensor Network (세맨틱 센서 네트워크) 온톨로지를 사용하여 데이터의 의미를 캡처합니다. 추출된 메타데이터는 Knowledge Graph (KG)에 저장되고, 실제 테스트 결과는 Virtual Knowledge Graph (VKG)로 접근이 용이한 parquet 파일에 보관됩니다. 데이터 검증은 LLM 기반 접근법을 사용하여 구조적 및 구문적 이질성에도 불구하고 데이터를 처리합니다. SPARQL 쿼리를 통해 SQL 언어로 자동 변환할 수 있습니다.

- **Performance Highlights**: 이 연구는 상태-of-the-art LLMs의 성능을 벤치마킹하여 자동 데이터 추출 및 검증을 기존 수동 프로세스와 비교했습니다. 데이터 분석으로 인한 높은 부가가치와 비용 절감의 이점을 확인했으며, 특히 위성 전자 보드 제조에서 시간 절약과 정확성 향상을 제공하는 것으로 나타났습니다.



### GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS (https://arxiv.org/abs/2408.01584)
Comments:
          8 pages, 4 figures

- **What's New**: 다중 에이전트 학습 알고리즘이 다양한 게임에서 초인적 계획 수립을 성공적으로 구현했지만 실제 다중 에이전트 계획 설계에 미치는 영향은 미미했습니다. 이를 극복하기 위해 GPU 가속 다중 에이전트 시뮬레이터, GPUDrive가 도입되었습니다. 이는 Madrona 게임 엔진 위에 구축되어 초당 백만 단계를 생성할 수 있습니다.

- **Technical Details**: GPUDrive는 C++로 직접 작성된 관찰, 보상 및 동역학 함수를 이용해 복잡하고 이질적인 에이전트 행동을 정의합니다. 이러한 행동은 CUDA로 낮춰져 고성능을 발휘합니다. 시뮬레이터는 다양한 센서 모달리티를 지원하며, 실제 주행 데이터와 자율 주행 데이터셋의 지도를 혼합하여 학습할 수 있습니다.

- **Performance Highlights**: GPUDrive를 사용하면 Waymo Motion 데이터셋의 여러 장면에서 단 몇 분 만에 효과적인 강화학습 에이전트를 훈련할 수 있습니다. 일반적으로 몇 시간 만에 목표 달성율이 높은 에이전트를 훈련할 수 있으며, 소비자용 GPU에서도 초당 백만 개 이상의 경험을 처리할 수 있습니다.



### Self-Taught Evaluators (https://arxiv.org/abs/2408.02666)
- **What's New**: 새로운 접근법으로 인간 주석 없이 평가자를 개선할 수 있는 방법을 제안합니다. 이 연구는 합성된(training data) 데이터만을 사용해서 평가자가 자체 개선(iterative self-improvement)할 수 있도록 설계되었습니다.

- **Technical Details**: 언노티드(unlabeled) 명령으로 시작하여, 모델이 생성한 대조적인 출력과 학습 데이터를 통해 평가자를 훈련시킵니다. 이 과정에서 LLM-as-a-Judge(판사 역할의 대형 언어 모델)을 사용하여 추론 과정(reasoning traces)과 최종 판단을 생성합니다. 반복적인 훈련을 통해 예측을 개선합니다.

- **Performance Highlights**: 라벨링된 선호 데이터 없이도 'Self-Taught Evaluator'는 강력한 LLM(Llama-3-70B-Instruct)의 성능을 75.4에서 88.3으로, 대부분 투표에서는 88.7로 향상시켰습니다. 이는 GPT-4 같은 기존의 LLM 평가자들을 능가하고, 라벨링된 예제로 훈련된 최고 성능의 보상 모델과 맞먹는 수준입니다.



### Can Reinforcement Learning Unlock the Hidden Dangers in Aligned Large Language Models? (https://arxiv.org/abs/2408.02651)
Comments:
          Accepted to AI4CYBER - KDD 2024

- **What's New**: 이번 연구는 강화학습(나선형 학습)을 사용하여 필립 노락 오류(adversarial triggers)를 최적화하는 새롭고 혁신적인 방법을 제안합니다. 이 방법은 타겟 모델에 대한 API 추론 액세스 및 소규모 대체 모델만 필요로 합니다. 주로 BERTScore 기반(reward function)을 활용하여 타겟 모델의 텍스트 출력에서 보상 신호를 계산합니다.

- **Technical Details**: 기존의 Jailbreaking 접근법은 블랙박스 모델에 대해 제한적이었습니다. 하지만, 이번 논문에서는 강화학습을 활용한 새로운 패러다임을 통해 필립 노락 오류(adversarial triggers)를 최적화합니다. 이 접근법은 타겟 언어 모델에 대한 추론 API 접근만을 필요로 하며, 소규모 대체 모델을 사용하여 타겟 모델의 텍스트 출력에서 보상 신호를 계산합니다. 이러한 방법은 기존의 백박스 모델에서의 필립 노락 오류를 새로운 블랙박스 모델에 적용하는 데 효과적입니다.

- **Performance Highlights**: 이 논문에서 제안된 방법은 새로운 블랙박스 대상 모델에서 필립 노락 오류(adversarial triggers)의 성능을 크게 향상시킵니다. 강화학습 기반의 접근법이 모델 간의 필립 노락 오류 전파성과 효과를 크게 높여주는 것을 입증합니다.



### SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models (https://arxiv.org/abs/2408.02632)
- **What's New**: 대규모 언어 모델(LLM)의 보안성과 유해한 출력 방지는 중요한 과제로 떠오르고 있습니다. 이를 해결하기 위한 유망한 접근법으로 모델이 자체적으로 적대적 명령어(adversarial prompts)를 생성하여 red teaming에 활용하는 방법이 제시되었습니다. 하지만 LLM의 보안 취약점이 더 미묘해짐에 따라 기존의 적대적 방법들이 효과를 발휘하기 어려워지고 있습니다. 이에 따라 새로운 최적화 프레임워크인 SEAS(Self-Evolving Adversarial Safety) 프레임워크가 소개되었습니다. 이 프레임워크는 모델이 생성한 데이터를 활용하여 보안성을 향상시킵니다. SEAS는 초기화, 공격, 적대적 최적화의 세 가지 단계로 구성되어 있으며, 반복적으로 Red Team과 Target 모델을 개선하여 견고성과 안전성을 높이는 것을 목표로 합니다.

- **Technical Details**: SEAS 프레임워크는 세 단계로 이루어져 있습니다. 초기화 단계에서는 Red Team 모델(R0)과 Target 모델(T0)을 SEAS 데이터셋을 이용하여 각각 미세 조정합니다. 공격 단계에서는 Red Team 모델이 적대적 명령어를 생성하면, 이를 Target 모델에 입력하여 응답을 유도합니다. Safe Classifier는 이 응답의 안전성을 평가합니다. 적대적 최적화 단계에서는 효과적이었던 명령어와 응답을 기반으로 Red Team과 Target 모델을 업데이트합니다. 이러한 과정을 여러 번 반복하여 두 모델이 점차 적응하고 향상되도록 합니다.

- **Performance Highlights**: 세 번의 반복 후, Target 모델은 GPT-4와 비슷한 수준의 보안성을 달성하였으며, Red Team 모델은 Llama3-70B에 대한 공격 성공률(ASR)이 50.66% 증가하였습니다. 또한, 생성된 적대적 명령어의 다양성과 반복적인 모델 업데이트의 효과를 입증하였습니다.



### Language Model Can Listen While Speaking (https://arxiv.org/abs/2408.02622)
Comments:
          Demo can be found at this https URL

- **What's New**: 최신 대화형 음성언어 모델 (iSLM)에 전이중 모델링 (Full Duplex Modeling, FDM)을 도입하여 실시간 상호작용을 가능하게 합니다. 이는 '경청하며 말하는 모델 (Listening-while-Speaking Language Model, LSLM)'이라는 혁신적인 모델을 제안합니다.

- **Technical Details**: LSLM은 음성 생성에 토큰 기반 디코더 전용 Text-to-Speech (TTS)와 실시간 오디오 입력에 스트리밍 자가 지도 학습 (Streaming Self-Supervised Learning, SSL) 인코더를 사용합니다. 이 모델은 실시간으로 말하기와 듣기 기능을 융합하고, 세 가지 융합 전략(초기 융합, 중간 융합, 후기 융합) 중 중간 융합이 최적의 성능을 보여줍니다. 실험 설정은 커맨드 기반 FDM과 음성 기반 FDM을 포함합니다.

- **Performance Highlights**: LSLM은 높은 소음 견딤성과 다양한 지시에 대한 민감도를 보이며, 기존 시스템에 거의 영향을 주지 않으면서 전이중 통신을 달성할 수 있음을 입증했습니다. 특히, 중간 융합 전략이 음성 생성과 실시간 상호작용 능력의 균형을 잘 맞추어 최적의 성능을 발휘했습니다.



### Progressively Selective Label Enhancement for Language Model Alignmen (https://arxiv.org/abs/2408.02599)
- **What's New**: 최근 인공지능 언어 모델이 강력한 기능을 보여주고 있지만, 인간의 기대와 어긋나는 콘텐츠를 생성할 가능성이 있어 윤리 및 법적 문제를 일으킬 수 있습니다. 이러한 문제를 해결하기 위해 PSLE(Progressively Selective Label Enhancement)이라는 프레임워크를 제안합니다. 이는 생성된 데이터를 효율적으로 활용하여 모델의 출력을 인간의 기대에 맞추도록 유도합니다.

- **Technical Details**: PSLE 프레임워크는 생성된 모든 데이터를 통해 언어 모델을 학습시키는 방식을 사용합니다. 이는 동적으로 업데이트된 임계값을 이용해 원본 입력의 응답과 원칙에 따라 안내된 출력의 보상 점수 차이를 기반으로 합니다. 두 응답이 유사한 품질을 나타내면, 둘 다 모델 훈련에 포함되며 보상 점수를 기준으로 가중치를 할당합니다. 이를 통해 데이터 활용 효율성을 높이고, 전체 훈련 효율성을 개선합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험 결과, PSLE는 기존의 언어 모델 정렬 방법들에 비해 뛰어난 효과를 입증했습니다. 저자의 기여는 데이터 활용 효율성을 높인 새로운 접근법을 제안한 것입니다. 또한, 점진적 임계값 전략을 통해 최적의 언어 모델로 수렴할 수 있음을 이론적으로 증명했습니다.



### Modelling Visual Semantics via Image Captioning to extract Enhanced Multi-Level Cross-Modal Semantic Incongruity Representation with Attention for Multimodal Sarcasm Detection (https://arxiv.org/abs/2408.02595)
- **What's New**: 이 연구는 소셜 미디어 데이터에서 효과적으로 풍자를 인식하기 위해 텍스트에 더해 이미지와 같은 추가적인 맥락적 단서를 포함하는 새로운 다중 모달리티 풍자 탐지 프레임워크를 제안합니다. 특히, 이미지와 함께 설명적인 이미지 캡션을 추가해 텍스트와 시각적 콘텐츠 간의 불일치를 더 정확하게 포착하려는 것이 주요 목표입니다.

- **Technical Details**: 주요 기술적 기여로는 (1) Cross-lingual 언어 모델을 활용한 강력한 텍스트 특징 추출 브랜치, (2) Self-regulated Residual ConvNet과 가벼운 Spatially Aware Attention 모듈을 통합한 시각적 특징 추출 브랜치, (3) 이미지에 포함된 텍스트를 읽을 수 있는 인코더-디코더 구조를 사용한 이미지 캡션 생성, (4) 텍스트와 두 가지 이미지 표현들 사이의 불일치를 효과적으로 식별하기 위한 다양한 어텐션 모듈, (5) 피처 융합을 통해 다중 레벨의 교차 도메인 의미 불일치 표현 등이 포함됩니다.

- **Performance Highlights**: 제안된 모델은 트위터 다중 모달 풍자 탐지(Twitter multimodal sarcasm) 및 멀티불리(MultiBully) 데이터셋에서 각각 92.89%와 64.48%의 정확도로 최상의 성능을 보였습니다.



### Leveraging the Power of LLMs: A Fine-Tuning Approach for High-Quality Aspect-Based Summarization (https://arxiv.org/abs/2408.02584)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 세부 요약(Aspect-based summarization)에 맞추어 미세 조정(fine-tuning)함으로써 문서의 특정 측면에 집중한 고품질 요약을 생성하는 방법을 탐구합니다. 오픈 소스 기반 LLMs인 Llama2, Mistral, Gemma, Aya를 사용하여 세부 요약 데이터셋을 통해 이들의 성능을 평가합니다.

- **Technical Details**: 연구는 LLMs를 Open Aspect-based Summarization (OASUM) 데이터셋으로 미세 조정하여 특정 측면에 대한 내용을 효과적으로 추출하고 요약할 수 있도록 하는 데 중점을 둡니다. 모델은 종합적인 평가 프레임워크를 설정하여 기존 세부 요약 방법 및 원래 LLMs 버전과의 성능을 비교합니다.

- **Performance Highlights**: 초기 결과는 미세 조정된 LLMs가 최신 세부 요약 방법과 비교할 때 더 높은 품질의 요약을 생성할 수 있음을 보여줍니다. 이는 교육, 의료, 음악 등 다양한 도메인에서 데이터 변형과 필요한 전문 지식을 효과적으로 처리할 수 있음을 시사합니다.



### Clustering and Mining Accented Speech for Inclusive and Fair Speech Recognition (https://arxiv.org/abs/2408.02582)
- **What's New**: 현대의 자동 음성 인식(ASR) 시스템은 수만 시간의 음성 데이터를 학습하여 대단한 성공을 거두었지만, 이러한 데이터의 분포는 일반적인 악센트나 전형적인 음성 패턴에 편향되어 있는 경향이 있습니다. 이 논문은 이러한 편향성을 줄이고 드문 악센트의 음성도 잘 인식할 수 있는 ASR 시스템을 구축하기 위한 악센트 클러스터링 및 마이닝 방안을 제안합니다.

- **Technical Details**: 논문에서는 세 가지 방법을 통해 제한된 양의 감독된 악센트 데이터를 극복하고 악센트 인식 모델을 향상시켰습니다. 첫째, 대규모 음성 데이터셋을 활용한 감독 및 비감독 학습을 통한 사전 학습(pre-training)을 사용했습니다. 둘째, 분포적으로 강건한 최적화(Distributionally Robust Optimization, DRO)를 적용하여 데이터 불균형을 해결하고, 셋째, 비감독 방식의 클러스터링을 통해 새로운 악센트를 인식하도록 했습니다.

- **Performance Highlights**: 제안된 방법을 사용하여 인도 악센트 음성에 대해 ASR 모델을 미세 조정한 결과, 무작위로 샘플링한 음성을 사용한 경우에 비해 각각 10.0% 및 5.3% 상대적인 WER 향상을 기록했습니다.



### Contrastive Learning-based Multi Modal Architecture for Emoticon Prediction by Employing Image-Text Pairs (https://arxiv.org/abs/2408.02571)
- **What's New**: 이 연구는 문장, 비주얼, 이모티콘 간의 관계를 분석하는 새로운 접근 방식을 제안합니다. 본 연구는 멀티모달 (multimodal) 기능 추출 방법을 상세히 분석하고, 이들을 결합한 새로운 대비 학습 (contrastive learning) 기반 멀티모달 아키텍처를 도입했습니다.

- **Technical Details**: 제안된 모델은 이중 브랜치 인코더 (dual-branch encoder)의 공동 훈련과 대비 학습을 활용하여 텍스트와 이미지를 공통된 잠재 공간으로 정확하게 매핑합니다. 특히, 여러 멀티모달 알고리즘과 특히 융합 접근 방식 (fusion approaches)을 종합적으로 검토하였습니다.

- **Performance Highlights**: 제안된 방법론은 현재의 멀티모달 접근 방식보다 정확성과 견고성 면에서 뛰어납니다. Multimodal-Twitter Emoticon 데이터셋을 사용하여 이모티콘을 평가한 결과, 제안된 모델은 91%의 정확도와 90%의 MCC 점수를 달성했습니다. 대비 학습으로 획득한 깊은 특징이 더 효율적임을 증명하며, 여러 모드에서 이모티콘을 인식할 수 있는 강력한 일반화 능력을 제시합니다.



### Evaluating and Enhancing LLMs Agent based on Theory of Mind in Guandan: A Multi-Player Cooperative Game under Imperfect Information (https://arxiv.org/abs/2408.02559)
- **What's New**: 대형 언어 모델(LLMs)이 불완전한 정보 환경에서 복잡한 게임을 다루고, 특히 비 영어권 환경에서 다중 에이전트 협력을 가능하게 하는 능력을 탐구한 연구입니다. 이 연구는 오픈소스 및 API 기반 LLM이 에이전트 협력이 필요한 고도화된 텍스트 기반 게임에서의 적용 가능성을 조사하여, 다른 유형의 에이전트들과의 성능을 비교합니다.

- **Technical Details**: 연구는 Theory of Mind (ToM) 계획 기법을 제안하여, LLM 에이전트들이 게임 규칙, 현재 상태 및 역사적 맥락만을 입력으로 사용하여 다양한 적대자에 맞춰 전략을 조정할 수 있도록 합니다. 또한, 이 카드 게임에서 동적이고 광범위한 행동 공간의 문제를 완화하기 위해 외부 도구를 통합했습니다.

- **Performance Highlights**: 결과적으로, 현재의 LLM과 최신 강화 학습(RL) 모델 사이에는 성능 격차가 존재하지만, LLM은 이 게임 설정에서 ToM 능력을 보여줍니다. 이는 LLM이 동맹 및 적대자의 행동을 이해하고 동맹과의 협력을 구축할 수 있는 능력을 시사합니다. 추가 연구와 이해를 장려하기 위해, 연구진은 코드베이스를 공개했습니다.



### MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization (https://arxiv.org/abs/2408.02555)
Comments:
          Project Page: this https URL Github: this https URL

- **What's New**: MeshAnything V2는 새로운 오토레그레시브 트랜스포머를 이용해 주어진 형태에 맞춘 아티스트가 만든 메쉬(Artist-Created Meshes, AM)를 생성하는 기술을 소개합니다. MeshAnything V2는 Adjacent Mesh Tokenization(AMT)이라는 새로운 메쉬 토큰화 방법을 사용하여 이전 방법보다 효율성과 성능 면에서 뛰어납니다. AMT는 가능한 경우 단일 정점을 사용하여 메쉬를 나타냅니다.

- **Technical Details**: MeshAnything V2는 AMT를 통해 기존 메쉬 토큰화 방법에서 볼 수 있는 세 정점을 사용한 접근 방식 대신 단일 정점을 사용하는 방식을 채택하여 메쉬의 토큰 시퀀스 길이를 절반으로 줄입니다. AMT는 페이스를 인코딩한 후 인접 페이스를 찾아 단일 정점으로 인코딩합니다. 이는 시퀀스 길이를 약 1/3로 줄이고, 결과적으로 AM 생성의 효율성과 성능을 크게 향상시킵니다.

- **Performance Highlights**: AMT를 통해 MeshAnything V2는 메모리 사용량을 거의 4배 줄이고, 기존의 최대 생성 가능한 페이스 수를 800개에서 1600개로 두 배로 늘립니다. 실험 결과 AMT의 도입으로 모델 성능과 효율성이 크게 개선되었습니다. MeshAnything V2는 이러한 성과를 기반으로 커뮤니티에 오픈소스로 제공됩니다.



### The Role of Functional Muscle Networks in Improving Hand Gesture Perception for Human-Machine Interfaces (https://arxiv.org/abs/2408.02547)
- **What's New**: 새로운 연구는 기능적 근육 네트워크를 중심으로 한 핸드 제스처 인식 모델을 제안합니다. 이 모델은 개별 근육의 활성화가 아니라 근육 간의 동기화를 디코딩하여, 낮은 계산 비용과 높은 효율성을 실현합니다. 이 접근법은 neurorobotics 및 상호작용 로봇 시스템에 큰 영향을 미칠 것으로 예상됩니다.

- **Technical Details**: 제안된 모델은 coherence-based 기능적 근육 네트워크를 사용하여 근육들의 동기화를 분석합니다. Magnitude Squared Coherence (MSC) 지표를 통해 주파수 영역에서 두 신호 간의 선형 연관성을 측정합니다. 이러한 기능적 근육 네트워크의 주파수 기반 특징들은 Support Vector Machine (SVM) 분류기에 의해 처리되어 제스처를 분류합니다.

- **Performance Highlights**: 본 연구는 Ninapro 데이터베이스를 사용하여 제안된 모델을 평가했습니다. 40명의 피험자가 수행한 17가지 핸드 제스처에서 85.1%의 정확도를 달성하였으며, 이는 기존 방법들보다 향상된 성능을 보이면서도 적은 계산 비용을 요구합니다.



### RAG Foundry: A Framework for Enhancing LLMs for Retrieval Augmented Generation (https://arxiv.org/abs/2408.02545)
Comments:
          10 pages

- **What's New**: 저자들은 Retrieval-Augmented Generation (RAG) 시스템의 복잡성을 해결하고자 오픈 소스 프레임워크인 RAG Foundry를 소개했습니다. 이 프레임워크는 데이터 생성, 모델 학습(training), 추론(inference), 평가 과정을 단일 워크플로에 통합하며, 대형 언어 모델(LLM)을 RAG 환경에서 학습하고 평가하는 데 도움을 줍니다.

- **Technical Details**: RAG Foundry는 다양한 RAG 기술을 실험하고 프로토타이핑하는 과정을 단순화합니다. 사용자는 내부 또는 전문 지식 소스를 활용해 손쉽게 데이터셋을 생성하고 RAG 모델을 학습할 수 있습니다. 이러한 통합 접근 방식은 데이터 증강(dataset augmentation) 및 학습에 대한 효율성을 증대시키며, 다양한 RAG 설정에서의 성능을 극대화합니다. Llama-3와 Phi-3 모델을 다양한 RAG 구성으로 증강 및 미세 조정(fine-tuning)하여 프레임워크의 효과를 입증했습니다.

- **Performance Highlights**: 세 가지 지식 집중 데이터셋(knowledge-intensive dataset)에서 일관된 성능 향상이 확인되었습니다. 이는 프레임워크가 다룰 복잡한 설계 결정을 완화하면서도 데이터 검색 정확도와 생성 품질을 동시에 향상시킬 수 있음을 보여줍니다. 결과적으로, RAG Foundry는 효율적인 RAG 시스템 구현 및 평가를 촉진하는 도구임이 입증되었습니다.



### Single-tap Latency Reduction with Single- or Double- tap Prediction (https://arxiv.org/abs/2408.02525)
- **What's New**: 터치 인터페이스에서 단일 탭(single tap)과 더블 탭(double tap)의 감지 속도를 개선하기 위해 새로운 기계 학습 방법인 PredicTaps를 제안합니다. 이 방법은 전통적으로 몇백 밀리초가 걸리는 시간을 기다리지 않고도 첫 번째 탭이 단일 탭인지 더블 탭의 일부인지를 즉시 예측할 수 있습니다.

- **Technical Details**: PredicTaps는 터치 센서를 통해 수집된 데이터를 분석하여, 감지된 첫 번째 탭이 단일 탭인지 두 번째 탭이 뒤따를 것인지 여부를 예측합니다. 이를 통해 단일 탭 이벤트를 즉시 실행하거나, 두 번째 탭을 기다립니다. 이 방식은 복잡한 휴리스틱(heuristics)이나 세부적인 설계를 필요로 하지 않습니다.

- **Performance Highlights**: 실험 결과 노트북에서는 150-500 ms에서 12 ms로, 스마트폰에서는 17.6 ms로 단일 탭 지연 시간을 크게 단축했고, 정확도는 노트북에서 100%, 스마트폰에서 50% 상위 데이터로 훈련시킨 결과 매우 높은 정확도를 보였습니다. 사용자 평가에서도 긍정적인 반응을 얻었습니다.



### A First Look at License Compliance Capability of LLMs in Code Generation (https://arxiv.org/abs/2408.02487)
- **What's New**: 최근 Large Language Models (LLMs)의 발전은 코드 생성 분야에서 혁명을 일으켰으며, 이에 따라 개발자들 사이에서 AI 코딩 도구의 사용이 급증했습니다. 그러나 LLM이 생성한 코드가 라이선스 정보를 제공하지 않는 경우 지적 재산권 위반의 가능성이 있습니다. 이 논문에서는 LLM이 생성한 코드의 라이선스 준수 능력을 평가하기 위한 벤치마크를 제시합니다. 이를 위해 우리는 'striking similarity' (현저한 유사성)의 기준을 설정하고, 이 기준을 통해 LiCoEval이라는 평가 벤치마크를 제안하여 14개의 LLM을 평가했습니다.

- **Technical Details**: 이 연구에서는 LLM이 생성한 코드와 특정 오픈 소스 코드 사이의 'striking similarity'를 판별하기 위해 경험적 연구를 시행했습니다. 이를 기반으로 LiCoEval이라는 평가 벤치마크를 제안, 14개의 인기 있는 LLM의 라이선스 준수 능력을 평가했습니다. 이 과정에서, 'access and substantial similarity' (접근성과 실질적 유사성)라는 법적 원칙을 고려하여 LLM의 결과물이 특정 오픈 소스 코드와 유사한 이유를 밝혀냈습니다.

- **Performance Highlights**: 14개의 LLM 중 최고 성능을 보인 모델들도 0.88%에서 2.01% 사이의 비율로 기존 오픈 소스 코드와 현저히 유사한 코드를 생성했습니다. 특히, 대부분의 LLM은 copyleft 라이선스를 가진 코드에 대해 정확한 라이선스 정보를 제공하지 못했습니다. Claude-3.5-sonnet 모델만이 일부 정확한 라이선스 정보를 제공하는 데 성공했습니다. 이러한 결과는 LLM의 코드 생성 작업에서 라이선스 준수 능력을 개선해야 할 긴급 필요성을 강조합니다.



### From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Futur (https://arxiv.org/abs/2408.02479)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 소프트웨어 공학(SE)에서의 활용과 LLM 기반 에이전트에 대한 현재의 연구 동향과 해결책을 조사한 첫 번째 종합 설문 조사입니다. 특히 요구사항 엔지니어링, 코드 생성, 자율 의사 결정을 비롯한 6개의 주요 주제에 대해 다루며, LLM과 LLM 기반 에이전트의 차이점과 유사점을 분석합니다.

- **Technical Details**: 이 연구는 LLM과 LLM 기반 에이전트가 소프트웨어 공학(SE)에서 다룰 수 있는 다양한 작업, 벤치마크 및 평가 기준을 조사합니다. 중요한 기술적 키워드로는 Retrieval-Augmented Generation (RAG), prompt engineering, Chain-of-Thought (COT)가 있습니다. 이 논문은 SE에서 LLM과 LLM 기반 에이전트의 역량을 명확히 구분하고 비교하는 것을 목표로 합니다.

- **Performance Highlights**: 이 논문에서는 LLM 기반 에이전트가 자율 디버깅, 코드 리팩토링, 적응형 테스트 생성과 같은 복잡하고 동적인 작업을 처리하는 데 있어 전통적인 LLM보다 뛰어난 성능을 보인다고 강조합니다. 또한 LLM 기반 에이전트가 SE에서 인공지능의 일반 지능(AGI) 접근에 가까운 성능을 보여준다고 합니다.



### An investigation into the causes of race bias in AI-based cine CMR segmentation (https://arxiv.org/abs/2408.02462)
- **What's New**: 이 논문은 인공지능(AI) 기반의 심장 자기공명영상(CMR) 분할 모델이 인종 편향(race bias)을 나타내는 원인에 대해 조사했습니다. 연구 결과, 인종 편향의 주된 원인은 심장 부위 외부의 이미지 콘텐츠에 있다는 것을 밝혀냈습니다. 그리고 이미지를 심장 주변으로 자를 경우 편향을 줄일 수 있음을 보여줍니다.

- **Technical Details**: 실험 데이터로는 영국 바이오뱅크(UK Biobank)에서 제공된 436명의 단축상(cine short-axis) CMR 이미지를 사용했습니다. 218명은 백인, 218명은 흑인으로 구성되었습니다. 분류 모델로는 18-레이어 레즈넷(ResNet-18)을 사용했고, 분할 모델로는 nnU-Net을 활용했습니다. 분류 및 분할 실험에서 다양한 모델 학습 파라미터와 이미지 보강 방법을 적용했습니다. 그리고 GradCAM을 통해 모델의 해석 가능성을 높였습니다.

- **Performance Highlights**: 분류 실험에서는 인종을 이미지로 높은 정확도로 예측할 수 있었지만, 진실된 분할(ground truth segmentation)에서는 낮은 정확도를 보였습니다. 이는 주로 이미지 기반의 분포 변동(distributional shift) 때문이라고 생각됩니다. 이미지 내부의 비심장 영역(피하 지방 등)에 모델이 집중하는 것을 확인했습니다. 이미지를 심장 주변으로 자르면 분류 정확도가 거의 무작위 수준으로 감소했습니다. 분할 모델에서도 잠재 표현(latent representations)에서 인종 정보를 예측할 수 있음을 보여주며, 이 역시 심장 이미지로 자르면 편향은 줄어들지만 완전히 사라지진 않았습니다.



### Enhancing Heterogeneous Knowledge Graph Completion with a Novel GAT-based Approach (https://arxiv.org/abs/2408.02456)
- **What's New**: 본 논문에서는 Heterogeneous Knowledge Graphs (KGs)을 위한 새로운 GAT 기반 기법인 GATH를 소개합니다. 이 기법은 두 개의 독립적인 어텐션 네트워크 모듈을 결합하여 누락된 엔티티를 예측하는 데 효과적입니다.

- **Technical Details**: GATH는 불균형한 샘플 시나리오에서도 강력한 성능을 보장하기 위해 새로운 인코딩 및 특성 변환 접근법을 도입했습니다. 또한, 기존 GAT 기반 지식 그래프 완성 기법들의 오버피팅 문제와 tail (head) 엔티티 예측 성능 문제를 해결하고자 설계되었습니다.

- **Performance Highlights**: GATH 모델은 기존의 최신 GAT 기반 모델들과 비교하여 FB15K-237 데이터셋에서 Hits@10과 MRR 측정치가 각각 5.2% 향상되었습니다. 또한, WN18RR 데이터셋에서 각각 4.5%와 14.6%의 성능 향상을 보였습니다.



### Long Input Benchmark for Russian Analysis (https://arxiv.org/abs/2408.02439)
- **What's New**: LIBRA(롱 입력 벤치마크, Long Input Benchmark for Russian Analysis)은 러시아어로 작성된 긴 텍스트를 이해할 수 있는 적절한 평가 도구를 제공합니다. 새로운 벤치마크는 21개의 데이터셋으로 구성되었으며, LLM의 장문의 문맥 이해 능력을 평가하기 위해 설계되었습니다.

- **Technical Details**: LIBRA는 다양한 복잡성을 가진 4개의 그룹으로 나누어졌으며, 각 모델은 4k에서 128k 토큰의 문맥 길이를 가진 데이터셋에서 평가될 수 있습니다. 또한 공개된 오픈 소스 데이터셋, 코드베이스 및 리더보드를 제공하여 향후 연구 방향을 안내합니다.

- **Performance Highlights**: LIBRA 벤치마크를 통해 LLM들은 문맥의 길이가 증가할 때 성능이 어떻게 변화하는지에 대한 깊이 있는 분석이 가능합니다. 길이에 따른 성능 변화를 탐구함으로써 모델의 한계를 이해하고, 향후 개선 방향을 모색할 수 있습니다.



### PENDRAM: Enabling High-Performance and Energy-Efficient Processing of Deep Neural Networks through a Generalized DRAM Data Mapping Policy (https://arxiv.org/abs/2408.02412)
Comments:
          11 pages, 15 figures, 2 tables. arXiv admin note: substantial text overlap with arXiv:2004.10341

- **What's New**: PENDRAM이라는 새로운 설계 공간 탐색 방법론을 제안하여 고성능 및 에너지 효율적인 CNN 가속을 가능하게 합니다. PENDRAM은 일반화된 DRAM 데이터 맵핑 정책을 통해 다양한 DRAM 아키텍처에 걸쳐 DRAM 액세스 지연 및 에너지를 최적화합니다.

- **Technical Details**: PENDRAM은 DRAM 데이터 맵핑 정책과 DRAM 아키텍처가 CNN 파티셔닝 및 스케줄링 방식에 미치는 영향을 탐색하여 최적의 설계를 선택합니다. 이를 위해 DRAM 액세스 지연 및 에너지를 줄이기 위한 설계를 제공합니다. 주요 기법으로는 row buffer hits, bank-level parallelism, subarray-level parallelism 최적화가 포함됩니다.

- **Performance Highlights**: 실험 결과, PENDRAM의 DRAM 데이터 맵핑 정책은 다른 맵핑 정책과 비교하여 CNN 가속기의 DRAM 액세스 에너지-지연 곱을 최대 96%까지 향상시킵니다. 이는 다양한 임베디드 AI 애플리케이션을 위한 고성능 및 에너지 효율적인 CNN 가속기를 제공합니다.



### Multi-weather Cross-view Geo-localization Using Denoising Diffusion Models (https://arxiv.org/abs/2408.02408)
Comments:
          Accepted by ACM MM24 workshop

- **What's New**: 이번 연구는 GNSS가 차단된 환경에서 드론 뷰(Drone-view) 이미지와 지리적 태그가 부여된 위성 뷰(Satellite-view) 이미지를 매칭하여 위치를 파악하는 Cross-view Geo-localization을 다룹니다. 특히 기존의 악천후 상태에서의 성능 저하 문제를 해결하기 위해 새로운 Multi-weather Cross-view Geo-localization Framework(MCGF)를 소개합니다. 이 프레임워크는 이미지 복원과 위치 파악을 동시에 최적화하며, denoising diffusion 모델을 활용합니다.

- **Technical Details**: MCGF는 denoising diffusion 모델을 통해 이미지 복원과 위치 파악을 동시에 최적화합니다. 이미지 복원에는 공유 인코더와 가벼운 복원 모듈이 포함되어 있으며, 위치 파악의 백본으로는 EVA-02가 사용됩니다. EVA-02는 적은 파라미터로도 드론 및 위성 이미지에서 유리한 정보를 추출할 수 있는 ViT(Vision Transformer) 모델입니다. 학습에는 cross-entropy loss를, 테스트에는 cosine distance를 사용합니다.

- **Performance Highlights**: University160k-WX 데이터셋에서의 광범위한 실험 결과, MCGF는 다양한 날씨 조건에서도 경쟁력 있는 위치 파악 성능을 입증했습니다. 향후 MCGF의 코드가 GitHub(https://github.com/fengtt42/ACMMM24-Solution-MCGF)에 공개될 예정입니다.



### Enhancing AI-based Generation of Software Exploits with Contextual Information (https://arxiv.org/abs/2408.02402)
Comments:
          Accepted for publication at The 35th IEEE International Symposium on Software Reliability Engineering

- **What's New**: 이번 연구는 Neural Machine Translation(NMT) 모델이 자연어(NL) 설명으로부터 공격 보안 코드(offensive security code)를 생성하는 능력을 탐구합니다. 실제 쉘코드(shellcode)를 포함한 데이터 세트를 사용하여 다양한 시나리오에서 모델을 평가하였습니다. 특히, 문맥 이해의 중요성과 그것이 모델 성능에 미치는 영향을 강조합니다.

- **Technical Details**: 실험은 정보가 누락된 설명, 필요한 문맥, 불필요한 문맥 등을 포함한 다양한 시나리오에 걸쳐 디자인되었습니다. 이는 불완전한 설명에 대한 모델의 회복력, 문맥을 활용한 정확도 향상 능력, 불필요한 정보를 분별하는 능력을 평가하기 위함입니다. 실험 결과, 문맥 데이터의 도입이 성능을 크게 향상시키지만, 추가적인 문맥 정보의 이점은 일정 수준을 넘어서면 감소하는 경향을 보였습니다. 이는 모델 훈련을 위한 최적의 문맥 정보 수준을 시사합니다.

- **Performance Highlights**: 모델은 공격 보안 코드 생성에서 높은 정확성을 유지하면서도 불필요한 문맥을 걸러내는 능력을 보여주었습니다. 연구 결과는 AI 기반 코드 생성에서 문맥 사용 최적화를 위한 미래 연구의 토대를 마련하며, 특히 높은 기술적 정밀도가 요구되는 공격 코드 생성에서 중요한 시사점을 제공합니다.



### The Contribution of XAI for the Safe Development and Certification of AI: An Expert-Based Analysis (https://arxiv.org/abs/2408.02379)
- **What's New**: 본 연구에서는 신뢰할 수 있는 인공지능(AI) 개발 및 인증을 위해 XAI(eXplainable AI, 설명 가능한 인공지능) 방법론이 어떻게 활용될 수 있는지에 대해 15명의 전문가 인터뷰를 통해 탐구하였습니다. 특히, XAI 방법이 AI 시스템의 안전한 개발 및 인증에 미치는 영향과 한계를 논의하였습니다.

- **Technical Details**: 연구는 머신 러닝(ML) 모델의 구축 과정에서 발생하는 '블랙박스' 특성을 개선하기 위해 XAI가 사용될 수 있는지 여부를 검토합니다. 또한, 데이터 의존성 및 지속적인 학습 등 머신러닝 특유의 문제점들도 다뤄집니다. 전문가 인터뷰를 통해 XAI의 실질적 유용성을 탐구하였으며, 기존의 인증 프레임워크에 XAI를 통합할 수 있는지에 대한 가능성도 평가합니다.

- **Performance Highlights**: XAI 방법론은 ML 모델의 바이어스 및 오류를 감지하는 데 유용하여 안전한 AI 개발을 돕지만, 기술 시스템에 대한 포괄적이고 정확한 정보가 필요하기 때문에 인증 과정에서 그 효과는 제한적일 것으로 예상됩니다. 연구 결과, XAI가 디버깅 도구로서의 역할을 잘 수행할 수 있는 반면, 인증 도구로서의 역할에는 여전히 한계가 있다는 결론을 내렸습니다.



### Scaling CS1 Support with Compiler-Integrated Conversational AI (https://arxiv.org/abs/2408.02378)
Comments:
          Papers, funding sources, and Github Repositories at: this https URL

- **What's New**: 이번 연구는 DCC Sidekick이라는 웹 기반 대화형 AI 도구를 소개합니다. 이 도구는 기존의 LLM(대형 언어 모델) 기반 C/C++ 컴파일러를 보강하여 교육적인 프로그래밍 오류 설명을 생성합니다. DCC Sidekick는 코드 표시, 컴파일 및 런타임 오류 메시지, 스택 프레임 리드 아웃과 AI 인터페이스를 결합하여 더욱 개선된 설명을 제공합니다. 이러한 도구는 특히 비즈니스 시간이 아닌 시간에도 자주 이용됨으로써, 항상 사용할 수 있는 리소스로서의 가치를 보여주고 있습니다.

- **Technical Details**: DCC Sidekick는 기존 DCC Help 도구를 확장하여 C/C++ 프로그래밍 오류 메시지의 향상된 설명을 생성하는 데 LLM을 활용합니다. 학생들은 웹 기반 대시보드를 통해 코드, 오류 메시지 및 프로그램 상태를 종합적으로 볼 수 있으며, 대화형 인터페이스를 통해 후속 질문을 하고 이해도를 높일 수 있습니다. 또한, 컴파일러가 제공하는 전체 컨텍스트와 메모리 스택 세부정보를 활용하여 더욱 상세한 설명을 제공합니다.

- **Performance Highlights**: DCC Sidekick는 호주 CS1 과정을 수강하는 959명의 학생들이 11,222회 세션 동안 17,982개의 오류 설명을 생성하는 데 사용되었습니다. 이 중 50% 이상의 상호작용이 비즈니스 시간 외에 발생하여, 도구가 학생들에게 항시 지원 리소스로서 중요한 가치를 가지고 있음을 나타냈습니다. 이를 통해 AI 지원 디버깅 도구의 강력한 도입률과 대규모 CS1 코스 지원에서의 확장성을 확인할 수 있었습니다.



### A Few-Shot Approach for Relation Extraction Domain Adaptation using Large Language Models (https://arxiv.org/abs/2408.02377)
- **What's New**: 이 논문에서는 대형 언어 모델 (LLM)의 인컨텍스트 학습(in-context learning) 능력을 활용하여 아키텍처, 건설, 엔지니어링, 운영 (AECO) 도메인의 연구 논문 제목과 초록에서 관계 추출(relation extraction)을 수행하는 새로운 방법을 제안합니다. 이는 SciERC 같은 특정 도메인 데이터셋에 의존하던 기존 방법과 차별화되며, 최소한의 전문가 주석을 이용하여 도메인 적응을 지원하는 few-shot 학습 전략을 채택합니다.

- **Technical Details**: 이 연구는 과학적 지식 그래프 과생성 모델의 도메인 적응을 위해 SciERC 가이드라인을 따릅니다. 먼저, OpenAlex 데이터베이스에서 476,000개의 AECO 분야 논문 데이터를 수집하고 이를 전처리한 후 Brat 도구로 주석을 달았습니다. 그런 다음 BERT 기반의 SpERT(Span-based Entity and Relation Transformer) 모델을 사용하여 관계 및 엔티티 추출 작업을 수행하였고, few-shot 예제를 통한 prompt tuning 접근법을 적용하여 LLM의 관계 추출 성능을 테스트했습니다.

- **Performance Highlights**: 기존의 SciERC 데이터셋을 사용하여 훈련된 SpERT 모델은 새로운 AECO 도메인에서 성능이 크게 저하되었으나, 제안된 few-shot 학습 방법을 활용한 모델은 minimal expert annotation만으로도 기존 베이스라인 대비 성능 개선을 보였습니다. 이는 특히 엔티티 추출과 관계 추출 작업에서 유의미한 성능 향상을 보여주며, 비용 효율적 도메인 적응의 가능성을 제시합니다.



### Dialogue Ontology Relation Extraction via Constrained Chain-of-Thought Decoding (https://arxiv.org/abs/2408.02361)
Comments:
          Accepted to appear at SIGDIAL 2024. 9 pages, 4 figures

- **What's New**: 대화 지향적인 (task-oriented) 대화 시스템의 새로운 접근법이 제안되었습니다. 이 논문에서는 관계 추출 (relation extraction)을 개선하기 위해 대규모 언어 모델 (Large Language Model, LLM)의 디코딩 메커니즘을 확장하는 방법으로서 Chain-of-Thought (CoT) 디코딩을 적용했습니다. 이를 통해 사용자 쿼리를 자동으로 처리하는 동안 발생하는 'hallucination'을 줄이고, 데이터에 대해 유의미한 관계를 더 정확하게 추출할 수 있게 했습니다.

- **Technical Details**: 이 연구는 CoT 디코딩 메커니즘을 대화 오토노미 관계 추출 (Dialogue Ontology Relation Extraction, DORE)에 확장하였습니다. CoT 디코딩은 원래 논리적 추론 문제에 적용되었으며, 이번 연구에서는 여러 디코딩 경로를 생성하고, 각 경로에서 예측된 관계의 신뢰도에 기반해 최종 답을 선택하는 방식으로 적용됩니다. 이를 위해 CoT 디코딩과 제한된 디코딩 (constrained decoding)을 결합하여 모델의 세분된 지식 활용을 최적화했습니다.

- **Performance Highlights**: 제안된 디코딩 메커니즘은 기존의 source one-shot 및 source fine-tuning 기준점을 모두 능가하는 성능을 보였습니다. 특히, MultiWOZ 2.1 및 Schema-Guided Dialogue (SGD) 데이터셋에서 실험을 통해 이 메커니즘의 효용성을 확인할 수 있었습니다.



### Active Sensing of Knee Osteoarthritis Progression with Reinforcement Learning (https://arxiv.org/abs/2408.02349)
- **What's New**: 이 연구는 Osteoarthritis (OA) 예측을 위한 기존 정적 모델을 넘어, Reinforcement Learning (RL)을 기반으로 한 새로운 Active Sensing (AS) 접근 방식을 제안합니다. 이 방법은 환자를 동적으로 추적하여, 비용을 최소화하면서 정보가 많은 데이터를 최대한 많이 수집하는 것을 목표로 합니다.

- **Technical Details**: 이 연구는 환자의 장기적 관찰을 통해 얻은 다중 모달 데이터(multi-modal data)와 임상 변수를 활용합니다. 제안된 모델은 시퀀스 결정(sequential decisions)을 내려 특정 시간점에서 환자를 검사할지 여부를 예측합니다. 새로운 보상 함수(reward function)를 사용하여 다양한 신체 부위에서의 질병 진행을 설계하며, 이는 전체 과정을 최적화하는 Reinforcement Learning 모델에 통합됩니다.

- **Performance Highlights**: 종합 평가 결과, RL을 사용한 접근 방식이 현 상태의 예측 모델과 비교해 금전적 이점이 더 큼을 보여줍니다. 이 모델은 기존의 정적 예측 모델보다 더 효율적이고 비용 효과적인 환자 추적을 가능하게 합니다.



### Generalized Gaussian Temporal Difference Error For Uncertainty-aware Reinforcement Learning (https://arxiv.org/abs/2408.02295)
- **What's New**: 이 논문에서는 전통적인 Temporial Difference (TD) 학습 방법의 단점을 보완하기 위해 심층 강화 학습 (deep RL)에서 일반화된 가우시안 분포 (generalized Gaussian distribution, GGD)를 이용한 새로운 프레임워크를 소개합니다. 이 프레임워크는 상위 모멘트, 특히 첨도(kurtosis)를 포함하여 오류 분포 모델링의 유연성을 증가시켜 데이터 종속 소음인 aleatoric 불확실성의 추정과 완화를 향상시킵니다. 또한, epistemic 불확실성을 다루기 위해 편향 감소와 첨도 고려 사항을 통합하여 배치 역분산 가중 방법을 개선하였습니다.

- **Technical Details**: 전통적인 TD 학습은 일반적으로 TD 오류에 대해 제로 평균 가우시안 분포를 가정합니다. 그러나 이 가정은 실제 오류 분포를 제대로 나타내지 못할 수 있습니다. 이를 해결하기 위해 제안된 프레임워크는 GGD를 통해 TD 오류의 분포를 더욱 정확하게 모델링하여 aleatoric 불확실성을 잘 반영합니다. GGD의 형상 매개변수(shape parameter)는 aleatoric 불확실성과 역관계가 있으며, 이를 활용하여 가중체계를 구축하여 모델의 견고성을 향상시킵니다.

- **Performance Highlights**: 정책 기울기 알고리즘(policy gradient algorithms)을 사용한 광범위한 실험 평가 결과, 제안된 방법이 기존 방법들에 비해 일관된 성능 향상을 보여줍니다. 특히, 실험 결과는 TD 오류 분포가 가우시안 분포에서 상당히 벗어난다는 점을 강조하며, 이론적으로 일반화된 가우시안 분포를 사용하는 방법이 더욱 효과적임을 증명하였습니다.



### Spin glass model of in-context learning (https://arxiv.org/abs/2408.02288)
Comments:
          8 pages, 4 figures

- **What's New**: 최근 연구에서 대형 언어 모델(LLM)들이 학습하지 않은 쿼리에 대해 프롬프트만을 사용하여 예측할 수 있는 인컨텍스트 학습(In-context learning) 능력을 보여주고 있습니다. 이 현상의 기계적 해석과 물리학적 연결은 아직 미해결된 문제로 남아 있습니다. 본 논문에서는 선형 주의를 사용하는 간단한 트랜스포머 모델을 스핀 글래스(spin glass) 모델로 매핑하여 이러한 현상을 설명합니다. 이를 통해 학습되지 않은 기능을 제공된 프롬프트만으로도 예측할 수 있는 이유를 밝혀냅니다.

- **Technical Details**: 이 연구는 선형 주의를 사용하는 단층 트랜스포머 구조를 고려합니다. 주어진 입력 시퀀스를 출력 시퀀스로 변환하는 자가 주의(self-attention) 메커니즘을 이용합니다. 이 트랜스포머 구조를 스핀 글래스 모델로 재구성하여, 실수 값을 가지는 스핀과 데이터 내의 내재적 무질서(disorder)를 설명합니다. 여기서 트랜스포머의 파라미터는 스핀으로 변환되고, 입력 시퀀스는 냉동된 무질서(quenched disorder)로 작용하여 스핀들이 상호작용하도록 합니다. 이를 통해 ICL 오류를 줄입니다.

- **Performance Highlights**: 이론적으로, 단일 인스턴스 학습(single instance learning)에서 과제의 다양성을 증가시키면 인컨텍스트 학습이 유발된다는 점을 밝힙니다. 이는 볼츠만 분포(Boltzmann distribution)가 유일한 올바른 솔루션으로 수렴함으로써 가능해집니다. 즉, 다양한 사전 훈련 과제를 통해 트랜스포머가 새로운 프롬프트 설정에서 예측력을 발휘할 수 있게 되는 것입니다. 이 스핀 글래스 모델은 대형 언어 모델의 경험적 성공을 이해하는 데 중요한 기초를 제공합니다.



### Hardware Aware Ensemble Selection for Balancing Predictive Accuracy and Cos (https://arxiv.org/abs/2408.02280)
Comments:
          Accepted at Third International Conference on Automated Machine Learning (AutoML 2024), Workshop Track; for code, see this https URL

- **What's New**: 새로운 하드웨어 인식 앙상블 선택 방식이 소개되었습니다. 이 방법은 예측 정확도와 하드웨어 효율성을 동시에 고려해 앙상블 후보를 평가합니다. 이를 통해 더 균형 잡힌 정확도와 운영 효율성을 제공하는 앙상블을 선택할 수 있습니다.

- **Technical Details**: 기존의 포스트 호크(포스트 호크) 앙상블 선택 알고리즘에 품질 다양성 최적화(Quality Diversity Optimization) 개념을 도입했습니다. 이를 통해 GES, QO-ES, QDO-ES를 확장하여 예측 성능과 비용의 파레토 프런트(Pareto front)를 생성합니다. 또한, 하드웨어 인식 앙상블 선택을 위한 두 가지 QDO-ES 변형을 제안했습니다. 이 연구는 TabRepo 데이터를 사용한 83개의 분류 데이터셋과 1416개의 머신 러닝 모델을 실험 대상으로 했습니다.

- **Performance Highlights**: 하드웨어 인식 앙상블 선택은 예측 정확도와 비용의 균형을 효과적으로 맞출 수 있음을 확인했습니다. 제안된 QDO-ES 변형이 기존 포스트 호크 앙상블 선택 알고리즘보다 더 나은 파레토 프런트를 생성하는 것으로 나타났습니다. 이를 통해 하드웨어 인식 AutoML 시스템의 발전에 기여할 수 있는 기초를 마련했습니다.



### DRFormer: Multi-Scale Transformer Utilizing Diverse Receptive Fields for Long Time-Series Forecasting (https://arxiv.org/abs/2408.02279)
- **What's New**: 이번 연구에서는 롱텀 시계열 예측(LTSF)에서 새로운 방식의 Dynamic Tokenizer와 Dynamic Sparse Learning Algorithm을 도입하여 다양한 수용 분야(Receptive Fields)와 스파스 패턴(Sparse Patterns)을 효과적으로 포착하는 DRFormer 모델을 제안하였습니다. 이를 위해 Multi-scale Transformer 모델과 다중 스케일 시퀀스 추출 기법을 결합하였으며, 그룹 인식 회전 위치 인코딩(Group-aware Rotary Position Encoding) 기술을 도입하여 다양한 시간 스케일 간의 포지션 인식을 향상시켰습니다.

- **Technical Details**: DRFormer 모델은 다이나믹 토크나이저(Dynamic Tokenizer)를 활용하여 여러 가지 수용 분야(Receptive Fields)를 학습하고, 동적으로 시계열 데이터를 다양한 패치 길이로 나누어 다중 스케일 시퀀스(Multi-scale Sequence) 특징을 포착합니다. 또한, 그룹 인식 회전 위치 인코딩(Group-aware Rotary Position Encoding)을 통해 intra-와 inter-그룹 포지션 인식 능력을 높였습니다. 이로 인해 DRFormer은 시계열 데이터에서 발생하는 복잡한 의존성과 상호작용을 효과적으로 포착할 수 있습니다.

- **Performance Highlights**: 제안된 DRFormer 모델은 여러 실제 데이터셋에서 실험을 통해 기존 방법들보다 우수한 성능을 보였습니다. 다양한 베이스라인 모델들과의 비교 실험에서 DRFormer의 뛰어난 예측 성능이 입증되었습니다.



### Geometric Algebra Meets Large Language Models: Instruction-Based Transformations of Separate Meshes in 3D, Interactive and Controllable Scenes (https://arxiv.org/abs/2408.02275)
Comments:
          17 pages, 8 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)과 공형 기하학 대수(CGA)를 결합한 새로운 시스템 'shenlong'을 소개하여, 정밀한 3D 장면 편집, 특히 객체 재배치 작업을 혁신적으로 개선합니다. 전통적으로 이 작업은 복잡한 수작업과 전문 지식을 요구했으나, 'shenlong'은 자연어 명령을 CGA 작업으로 변환하여 정확한 공간 변환을 수행합니다.

- **Technical Details**: shenlong 시스템은 CGA를 견고한 형식 언어로 사용하여 공간 변환을 모델링하며, 사전 훈련된 LLM의 zero-shot learning 능력을 활용합니다. 이를 통해 특정 훈련 없이 다양한 3D 환경에서 자연어 지시를 정확하게 해석하고 반영할 수 있습니다. 이 시스템은 ThreeDWorld(TDW) 프레임워크에 통합되어 Unity3D 엔진과 호환됩니다.

- **Performance Highlights**: shenlong은 기존의 LLM 기반 대안들에 비해 객체 재배치 시 LLM 응답 시간을 평균 16% 감소시키고 성공률을 9.6% 향상시킵니다. 특히, 일반적인 실용적 쿼리에서는 100% 완벽한 성공률을 달성하여 다른 시스템이 따라올 수 없는 성능을 보입니다.



### Contrastive Learning and Abstract Concepts: The Case of Natural Numbers (https://arxiv.org/abs/2408.02247)
- **What's New**: 대조 학습 (Contrastive Learning, CL)이 구체적인 개념에 국한되지 않고 추상적인 개념에도 적용될 수 있음을 보여주는 새로운 연구가 발표되었습니다. 이 연구는 자연수 (natural numbers)와 같은 반-추상적 개념에 CL을 적용하여, 수량을 고정된 수치로 추정하거나 예측하는 방법을 탐구합니다.

- **Technical Details**: 본 연구는 보편적인 보존 원리(conservation principle)를 대조 학습에 적용하여 수량을 예측하는 'toy problem' 실험을 수행했습니다. CL은 종종 자기 지도 학습(self-supervised) 방식으로 해석되며, 여기서는 물체 분류 작업에서의 '정체성 보존'과 유사한 원리를 통해 수량의 보존을 적용했습니다. 이를 통해 CL이 고정된 수량을 한번에 고정된 정확도로 계산할 수 있음을 보여주었습니다.

- **Performance Highlights**: 기본 실험에서 CL과 유사한 구조의 지도 학습(supervised learning, SL) 신경망과 비교한 결과, 두 방식 모두 비슷한 좋은 성능을 나타냈습니다. 특히, 훈련 및 테스트 분포가 다를 때, 일반화 시나리오에서 CL은 더 견고하고 훨씬 낮은 오류 성능을 보였습니다.



### Evaluating Vision-Language Models for Zero-Shot Detection, Classification, and Association of Motorcycles, Passengers, and Helmets (https://arxiv.org/abs/2408.02244)
- **What's New**: 본 연구는 모터사이클 승차자의 헬멧 착용 상태를 비디오 데이터를 통해 감지하고 분류하기 위해 OWLv2와 CNN 모델들을 통합한 선진 비전-언어 기반 모델을 평가합니다. CVPR AI City Challenge에서 제공된 데이터셋을 확장하고, 검출 및 분류 작업을 위한 계단식 모델 접근 방식을 사용합니다.

- **Technical Details**: OWLv2 모델 및 CNN 모델을 사용한 계단식 모델 접근 방식을 채택하여 검출 및 분류 작업을 수행합니다. 제안된 접근 방식은 'zero-shot learning'을 활용하여 불완전하거나 편향된 학습 데이터셋에 의해 발생하는 문제를 해결합니다. OWLv2는 'open-world localization' 기능을 갖춘 Vision Transformer 기반 모델로, 다양한 조건 하에서 모터사이클, 헬멧 사용, 탑승자 위치를 감지하는 데 사용됩니다.

- **Performance Highlights**: 헬멧 감지의 평균 정밀도는 0.5324로 나타났으며, 검출 및 분류 성능을 상세히 설명하는 정밀도-재현율 곡선을 제공합니다. 낮은 해상도 데이터 및 가시성이 좋지 않은 상황에서도 유망한 결과를 도출하여 자동화된 차량 안전 및 교통안전 집행 시스템에 있어 중요한 진전을 보여줍니다.



### A Multi-Source Heterogeneous Knowledge Injected Prompt Learning Method for Legal Charge Prediction (https://arxiv.org/abs/2408.02233)
Comments:
          20 pages

- **What's New**: 이번 연구에서는 법적 AI 분야의 필수 과제인 법적 혐의 예측(legal charge prediction)을 다루고 있습니다. 본 연구는 사건 설명(case descriptions)에 다양한 신경망 구조를 직접 모델링하는 기존 방법과 달리, 다중 소스의 외부 지식(multi-source external knowledge)을 활용한 프롬프트 학습 프레임워크(prompt learning framework)를 제안합니다. 이는 법적 지식베이스(legal knowledge base), 대화형 LLM(conversational LLM), 관련 법률 문서 등을 포함합니다.

- **Technical Details**: 먼저 사건 설명에서 법적 지식베이스와 일치하는 지식 조각을 하드 프롬프트 템플릿(hard prompt template)를 통해 입력으로 캡슐화합니다. 또한 대조 학습(contrastive learning)을 통해 사건 설명과 관련된 법률 문서를 검색하여, 그 후 대화형 LLM을 통해 사건 설명 내의 사실 요소(factual elements)를 추출합니다. 마지막으로, 소프트 프롬프트 토큰(soft prompt tokens)의 임베딩 벡터와 사실 요소의 인코딩 벡터를 결합하여 지식 강화 모델의 순차적 추론(knowledge-enhanced model forward inference)을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 가장 큰 법적 혐의 예측 데이터셋인 CAIL-2018에서 최첨단(state-of-the-art) 성능을 달성하였으며, 데이터 의존성이 낮았습니다(lower data dependency). 사례 연구에서도 높은 해석 가능성(strong interpretability)을 보였습니다.



### SpecRover: Code Intent Extraction via LLMs (https://arxiv.org/abs/2408.02232)
Comments:
          Haifeng Ruan and Yuntong Zhang contributed equally to this work

- **What's New**: 기존 AutoCodeRover를 기반으로 한 SpecRover는 자연어 문제 설명과 소프트웨어 코드베이스 구조를 분석하여 높은 정확도의 패치를 자동으로 생성하는 새로운 접근법을 제시합니다. 이는 위 사양(Specification) 추론을 통해 개발자의 의도를 파악하고, 리뷰어 에이전트가 패치 검증을 수행하여 신뢰성을 높이는 방식을 채택합니다.

- **Technical Details**: SpecRover는 반복적인 사양 추론을 통해 개발자의 의도를 파악하고, GitHub 이슈를 해결하는 과정에서 코드 검색 및 사양 계산을 수행합니다. 그러한 사양들은 리뷰어 에이전트에 의해 조사되고, 리뷰어 에이전트는 패치의 정확성에 대한 증거를 제시합니다. 이 과정은 반복적으로 이루어지며 최종 패치에 대한 신뢰도를 높입니다.

- **Performance Highlights**: SpecRover는 SWE-Bench 라이브러리의 전체 2294개의 GitHub 이슈를 대상으로 평가하였으며, AutoCodeRover보다 50% 이상의 효율성 향상을 달성하였습니다. 추가로, 비용 면에서도 이슈 당 약 $0.65로 저비용으로 GitHub 이슈를 처리할 수 있습니다. SpecRover는 SWE-Bench 라이브러리에서 19.3%의 이슈 해결율과 SWE-Bench lite에서 31%의 이슈 해결율을 보여주었습니다.



### Is Large Language Model Good at Database Knob Tuning? A Comprehensive Experimental Evaluation (https://arxiv.org/abs/2408.02213)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)을 데이터베이스 조정(knob tuning)에 활용하는 방법을 탐구합니다. 특히 GPT-4와 Claude-3 같은 모델을 사용하여 보다 효율적이고 해석 가능한 방식으로 데이터베이스 성능을 최적화하는 방법을 제안합니다. 이 연구는 전통적인 Try-Collect-Adjust 방식의 한계를 극복하고, LLM을 통해 보다 간편하고 강력한 조정 방식을 제시합니다.

- **Technical Details**: 연구는 데이터베이스 조정 시스템에서 중요한 세 가지 하위 작업(knob pruning, model initialization, knob recommendation)을 정의하고, 이를 LLM 기반 솔루션으로 대체합니다. 각 하위 작업에 대해 전통적인 방법과 LLM 기반 접근 방식을 비교실험을 통해 분석하였습니다. 특히 GPT-4와 같은 모델이 'chain-of-thought' 방식으로 응답을 생성함으로써 높은 수준의 해석 가능성을 보였습니다.

- **Performance Highlights**: 실험 결과, GPT-4가 기존 모든 방법들 중 가장 뛰어난 성능을 보여주었습니다. 이는 GPT-4가 더 다루기 어려운 작업들에서 강력한 성능을 발휘하는 것과 관련이 있습니다. 이에 더해 폐쇄형 LLM이 오픈 소스 LLM에 비해 훨씬 뛰어난 성능을 보였으며, 매개변수의 수가 많을수록 성능이 좋아진다는 점도 확인되었습니다. 연구는 LLM의 높은 일반화 능력 및 단순한 프롬프트 조정을 통해 추가 훈련이나 코드 수정 없이도 효과적으로 사용할 수 있음을 나타냈습니다.



### MARCO: A Memory-Augmented Reinforcement Framework for Combinatorial Optimization (https://arxiv.org/abs/2408.02207)
- **What's New**: 이 논문은 Neural Combinatorial Optimization (NCO) 분야에 새로운 프레임워크인 MARCO (Memory-Augmented Reinforcement for Combinatorial Optimization)를 도입합니다. MARCO는 혁신적인 메모리 모듈을 통해 기존의 네트워크 생성 및 개선 방법을 강화합니다. 이 메모리 모듈은 최적화 과정에서 수집된 데이터를 저장하고 각 상태에서 상황에 맞는 정보를 검색하여 탐색 효율성을 높입니다. 이는 최적의 결정을 내리는 것과 이미 조사된 솔루션을 반복하지 않는 두 가지 기준에 의해 탐색이 이루어지도록 합니다. 이 접근 방식은 최적화 예산을 보다 효율적으로 사용할 수 있게 합니다.

- **Technical Details**: MARCO는 메모리 모듈과 학습된 정책을 결합하여 탐색 공간을 효율적으로 탐색합니다. 메모리 모듈은 최적화 과정에서 방문한 상태를 기록하고 관련된 정보를 검색하여 모델에 제공함으로써 더 나은 의사 결정을 지원합니다. 특히 병렬 검색 스레드가 동일한 메모리 모듈을 공유함으로써 협업 탐색을 효율적으로 수행할 수 있습니다. 또한, 유사성 기반 검색 메커니즘을 도입하여 과거 정보를 더욱 잘 활용할 수 있습니다. 이를 통해 다양한 검색 결과를 얻고, 중복된 탐색을 줄이며 더 높은 품질의 솔루션을 발견할 수 있습니다.

- **Performance Highlights**: 경험적 평가 결과, MARCO는 최대 컷, 최대 독립 집합, 트래블링 세일즈맨 문제와 같은 그래프 기반 문제에서 더 높은 품질의 솔루션을 발견하는 데 성공하였습니다. 최대 1200개의 노드가 있는 그래프에 대해 테스트한 결과, 최근 제안된 학습 기반 접근 방식보다 우수한 성능을 보였습니다. 특히 낮은 계산 비용으로 고성능을 달성하여 NCO 분야에 새로운 방향을 제시합니다.



### Towards AI-Safety-by-Design: A Taxonomy of Runtime Guardrails in Foundation Model based Systems (https://arxiv.org/abs/2408.02205)
Comments:
          15 Pages

- **What's New**: 현재 다양한 응용 분야에서 널리 사용되고 있는 foundation model (FM) 기반 시스템의 빠른 발전과 자율성 증대는 책임 있는 AI와 AI 안전성 문제에 대한 우려를 낳고 있습니다. 이 논문은 FM 기반 시스템의 실행 시간 동안 안전하고 책임 있는 행동을 보장하기 위한 'guardrails'의 설계에 대한 체계적인 연구가 부족하다는 점을 지적하고, 이를 위한 분류 체계를 제시하여 guardrails의 특성과 설계 옵션을 분류하고 비교합니다.

- **Technical Details**: 이 논문에서 제시하는 분류 체계는 세 가지 주요 범주로 구성됩니다: runtime guardrails를 채택하는 동기, 고려해야 할 품질 속성, 그리고 사용 가능한 설계 옵션. 이 분류 체계는 guardrails의 설계 결정에 구조적이고 구체적인 지침을 제공하며, 설계 결정으로부터 발생하는 다양한 트레이드오프(trade-offs)를 강조합니다. 분류 체계는 행동(action), 목표(target), 범위(scope), 규칙(rules), 자율성(autonomy), 양식(modalities), 및 기본 기술(underlying techniques)으로 나뉩니다.

- **Performance Highlights**: 논문은 기존 문헌 검토를 바탕으로 guardrails의 동기를 요약하며, 이는 연구자와 실무자들이 runtime guardrails를 FM 기반 시스템에 통합할 때 정보에 입각한 결정을 내릴 수 있도록 돕습니다. 중요한 품질 속성을 식별함으로써 guardrails의 설계가 의도된 기능을 수행할 뿐만 아니라 전반적인 설계 품질도 향상시킬 수 있습니다.



### SelfBC: Self Behavior Cloning for Offline Reinforcement Learning (https://arxiv.org/abs/2408.02165)
- **What's New**: 기존의 오프라인 강화학습(offline reinforcement learning) 정책 제한 방법이 과도하게 보수적인 경향이 있다는 점을 개선하기 위해, 새로운 동적 정책 제한 방법(dynamic policy constraint)을 제안했습니다. 이 방법은 이전에 학습된 정책의 지수 이동 평균(exponential moving average)에 기반하여 학습된 정책을 제한합니다.

- **Technical Details**: 제안된 방법인 자기 행동 복사(Self Behavior Cloning, SelfBC)는 기존 정책 제한 방법과 오프-정책 알고리즘을 통합합니다. 학습 초기에는 사전 학습된 정책을 참조 정책(reference policy)으로 사용하고, 점진적으로 학습된 정책으로 이를 업데이트합니다. 이는 정책의 보수성을 완화하고 안정적인 정책 개선을 가능하게 합니다.

- **Performance Highlights**: 이 방법은 D4RL MuJoCo 도메인에서 실험을 통해, 기존의 정책 제한 방법 중에서 최신의 성능(state-of-the-art performance)을 달성했습니다. 특히 비전문가 데이터셋(non-expert datasets)에서 우수한 성과를 보였습니다.



### Calibration-Disentangled Learning and Relevance-Prioritized Reranking for Calibrated Sequential Recommendation (https://arxiv.org/abs/2408.02156)
Comments:
          Published at CIKM '24 as a full research paper

- **What's New**: LeapRec (Calibration-Disentangled Learning and Relevance-Prioritized Reranking)는 사용자의 변화하는 선호도에 적응해야 하는 '연속적 추천' 시나리오에서 추천목록의 개인화된 비율을 유지하는 '교정된 추천(calibrated recommendation)'을 목표로 합니다. LeapRec은 교정과 관련성 간의 충돌을 효과적으로 다루며, 포스트 프로세싱 방식이 아닌 훈련 단계에서 이를 통합합니다.

- **Technical Details**: LeapRec은 두 단계로 구성되어 있습니다. 첫째, '모델 학습 단계'에서는 제안된 '교정-분리된 학습-랭킹 손실 함수(calibration-disentangled learning-to-rank loss)'를 사용하여 개인화된 랭킹을 최적화합니다. 둘째, '재정렬 단계'에서는 항목의 관련성을 우선시하여 목록 상단에 배치하고, 교정을 필요로 하는 항목을 비롯하여 충돌을 해결합니다.

- **Performance Highlights**: 네 개의 실제 데이터셋에 대한 광범위한 실험을 통해 LeapRec이 기존 방법들보다 일관되게 우수한 성능을 보여주는 것을 확인했습니다. 성능의 상위 지표(top-k)에서도 관련성과 교정의 최적화를 동시에 달성하는 것이 입증되었습니다.



### ARVO: Atlas of Reproducible Vulnerabilities for Open Source Softwar (https://arxiv.org/abs/2408.02153)
Comments:
          14 pages, 9 figures

- **What's New**: 이번 논문에서는 ARVO: 오픈소스 소프트웨어의 재현 가능한 취약점 아틀라스를 소개합니다. Google의 OSS-Fuzz 프로젝트에서 발견한 C/C++ 프로젝트의 취약점을 활용하여, 무려 5,000개 이상의 메모리 취약점을 재현하는 데 성공했습니다. 이 데이터셋은 취약점을 발견한 입력값, 취약점을 수정하는 개발자 패치, 소스에서 프로젝트를 재구성하고 실행할 수 있는 능력을 제공합니다. OSS-Fuzz가 새로운 취약점을 발견함에 따라 자동으로 업데이트되는 기능도 포함되어 있습니다.

- **Technical Details**: ARVO 데이터셋은 다음과 같은 주요 기능을 포함합니다: 1) 대규모: 273개의 프로젝트에서 5,001개의 취약점 포함, 2) 재컴파일 가능: 취약점 및 수정된 버전을 재구성 가능, 3) 트리거링 입력: 취약점을 테스트할 수 있는 개념 증명 입력 포함, 4) 정확한 수정 패치: 각 취약점을 수정하는 개발자 패치 제공, 5) 사용 용이성: Docker 이미지를 통해 간편하게 재현 가능. ARVO는 Google의 OSS-Fuzz 버그 트래커에서 버그 보고서를 수집하여, 각 취약점을 재현하는 데 필요한 프로젝트 의존성을 신중히 추적하여 높은 성공률을 달성하였습니다. 현재 ARVO는 OSS-Fuzz의 기존 접근 방식보다 52.5% 더 많은 사례를 다루고 있습니다.

- **Performance Highlights**: ARVO는 OSS-Fuzz의 재현 시도와 비교하여 패치 식별의 정확도가 높습니다. 또한, ARVO는 '거짓 양성 수정 false positive fixes'를 발견하여, OSS-Fuzz에 의해 수정된 것으로 표시된 문제가 여전히 트리거될 수 있는 사례를 다수 밝혔습니다. 마지막으로, ARVO는 연구자들에게 LLM 기반 취약점 수리 평가, 개발자 패치를 기반으로 실세계 취약점 수정 분석 등 다양한 연구를 위한 유용한 데이터셋을 제공합니다.



### Generative Retrieval with Few-shot Indexing (https://arxiv.org/abs/2408.02152)
- **What's New**: 기존의 생성 검색(GR) 접근법의 한계를 극복하기 위해 새로운 Few-Shot GR 프레임워크가 제안되었습니다. 이 접근법에서는 몇 번의 인덱싱만으로 문서의 식별자(문서 ID) 생성을 수행하고, 대규모 사전 학습 언어 모델(LLM)의 지식을 최대한 활용하면서도 별도의 훈련이 필요 없습니다.

- **Technical Details**: Few-Shot GR는 두 주요 단계를 포함합니다. 첫 번째는 다중 매핑을 통한 Few-Shot 인덱싱이며, 두 번째는 제한된 빔 검색(constrained beam search)을 이용한 검색 단계입니다. 인덱싱 단계에서 LLM은 각 문서에 대해 여러 개의 자유 형태 문서 ID(free-text docids)를 생성해 docid 은행을 만듭니다. 검색 단계에서는 동일한 LLM을 사용해 주어진 쿼리의 docid를 생성하고, 생성된 docid를 문서와 매핑합니다.

- **Performance Highlights**: Natural Questions(NQ) 데이터셋 실험 결과, Few-Shot GR은 기존의 중무장 훈련 기반 GR 접근법보다 뛰어난 성능을 발휘하고 효율성이 높습니다. 중요한 성공 요인으로는 다중 매핑을 포함한 Few-Shot 인덱싱과 효과적인 LLM 선택이 언급되었습니다.



### Environment Complexity and Nash Equilibria in a Sequential Social Dilemma (https://arxiv.org/abs/2408.02148)
Comments:
          Accepted to the 17th European Workshop on Reinforcement Learning (EWRL)

- **What's New**: 본 연구에서는 매트릭스 게임(matrix game) 사회 딜레마(social dilemma)를 더 복잡하고 고차원적인 멀티 에이전트 강화 학습(MARL) 환경으로 확장했습니다. 특히, 그리드 월드(gridworld) 구현을 사용하여 Stag Hunt 딜레마를 더 복잡한 환경으로 적용했습니다.

- **Technical Details**: 연구는 매트릭스 게임의 결정 공간(decision-space)을 하나의 실행에서 변수 환경 복잡성을 포함하는 고차원 MARL 환경으로 변환했습니다. 이러한 설정에서 MARL 에이전트들은 종종 위험 우선 나쉬 균형(risk-dominant Nash equilibria) 전략으로 수렴했습니다. 이는 환경 복잡성이 증가할수록 전략이 서브 옵티멀(suboptimal) 상태로 치우친다는 것을 시사합니다.

- **Performance Highlights**: 연구 결과에 따르면 환경 복잡성이 증가할수록 MARL 에이전트들이 더 자주 비최적화(suboptimal) 전략에 수렴하는 것으로 나타났습니다. 이는 고차원 게임 이론적(MARL) 환경에서 최적의 결과를 달성하는 데 있어 환경 복잡성의 영향력을 잘 보여줍니다.



### Analyzing Cultural Representations of Emotions in LLMs through Mixed Emotion Survey (https://arxiv.org/abs/2408.02143)
Comments:
          Was accepted to ACII 2024

- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 문화적 감정 표현, 특히 혼합 감정 상황에서의 표현을 분석합니다. Miyamoto et al. (2010)의 연구를 기반으로 일본어와 영어로 된 감정 반응 설문을 사용하여 LLM의 반응을 비교하고, LLM이 인종적, 문화적 편향을 어떻게 나타내는지 조사했습니다.

- **Technical Details**: 세 가지 연구를 수행했습니다. 첫 번째 연구는 일본어와 영어로 된 감정 설문에 대한 LLM의 반응을 비교하였고, 인간 실험 결과와의 일치를 분석했습니다. 두 번째 연구는 LLM 프롬프트에 다른 맥락적 정보를 추가하여 반응의 변화를 분석했습니다. 세 번째 연구는 동아시아 언어와 서유럽 언어 간의 반응 유사성을 비교했습니다. 사용된 모델은 mistral-7b-instruct, gemma-7b-it:free, llama-2-70b-chat, gpt-3.5-turbo, gpt-4-turbo-preview 등 5개의 LLM입니다.

- **Performance Highlights**: LLMs는 인간 실험 결과와 비교했을 때 제한된 일치도를 보였습니다. 프롬프트에서 참가자의 출신 정보보다 언어가 반응에 더 큰 영향을 미쳤습니다. 또한 동아시아 언어의 반응이 서유럽 언어보다 더 일관성이 있었습니다.



### VidModEx: Interpretable and Efficient Black Box Model Extraction for High-Dimensional Spaces (https://arxiv.org/abs/2408.02140)
- **What's New**: 이번 연구에서는 SHAP(SHapley Additive ExPlanations)을 활용하여 합성 데이터 생성을 향상시키는 새로운 접근 방식을 제안합니다. SHAP는 입력 특징의 개별 기여도를 설명하는데, 이를 통해 에너지 기반 GAN(GAN)을 최적화하여 원하는 출력을 달성할 수 있습니다. 이 방법은 이미지 분류 모델 정확도를 16.45% 향상시키고, 영상 분류 모델에서는 평균 26.11%, 최대 33.36%까지 성능을 높였습니다.

- **Technical Details**: 이 연구는 비전(vision) 분류기에 초점을 맞추었으며, 특정 구조적 제한 없이 다른 도메인에 쉽게 적응 가능한 접근 방식을 유지하고 있습니다. SHAP 값을 통해 생성기를 최적화하여 맞춤 목표를 달성하는 차별화된 파이프라인을 소개하고, 각 클래스에 대한 조건부 생성기를 통해 클래스 분포를 향상시켰습니다. 또한, 위 기술은 다양한 시나리오에서 효과적으로 적용될 수 있습니다. 예를 들어 top-k 예측 확률, top-k 예측 레이블, top-1 레이블이 있는 경우 등입니다.

- **Performance Highlights**: 이미지 분류 모델에서 16.45% 정확도 증가를 달성했으며, 영상 분류 모델(UCF11, UCF101, Kinetics 400, Kinetics 600, Something-Something V2)에서도 평균 26.11%, 최대 33.36%까지 성능을 개선했습니다. 이 방법은 높은 차원 공간으로 확장되는 파이프라인을 구현하여 이전 연구들과 비교한 결과에서도 우수한 성과를 보였습니다.



### Value-Based Rationales Improve Social Experience: A Multiagent Simulation Study (https://arxiv.org/abs/2408.02117)
Comments:
          13 pages, 13 figures, 13 tables (and supplementary material with reproducibility and additional results), accepted at ECAI 2024

- **What's New**: 이번 연구에서는 Exanna라는 새로운 프레임워크를 제안했습니다. Exanna 에이전트는 의사 결정 시 자기 자신과 타인의 가치를 고려하여 행동을 정당화하고 타인의 정당화를 평가합니다. 이 다중 에이전트 시뮬레이션을 통해 가치 기반 의사결정 및 정당화 생성이 특히 규범을 위반하는 행동에 대해 높은 갈등 해결, 향상된 사회 경험, 높은 프라이버시, 높은 유연성을 제공한다는 것을 입증했습니다.

- **Technical Details**: Exanna 프레임워크는 가치 기반 정당화를 생성하고, 의사 결정을 정당화하는 데 필요한 정보만 포함시킵니다. 이를 평가하기 위해 팬데믹 시나리오를 기반으로 다중 에이전트 시뮬레이션을 사용합니다. 에이전트는 Share-All, Share-Rules, 그리고 가치 정렬 규칙을 사용하는 각기 다른 종류의 정당화를 가지며, 가치의 중요성을 고려한 정당화를 제공하는 에이전트는 향상된 갈등 해결 능력을 보입니다.

- **Performance Highlights**: 가치에 맞춘 정당화는 더 적은 정보를 제공하더라도 더 나은 사회 경험에 기여합니다. Exanna는 규범 위반을 지원하고 사회적 경험을 개선하기 위해 가치가 정당화 생성 및 사용을 어떻게 안내하는지에 대해 첫 번째로 연구했습니다.



### Dise\~no de sonido para producciones audiovisuales e historias sonoras en el aula. Hacia una docencia creativa mediante el uso de herramientas inteligentes (https://arxiv.org/abs/2408.02113)
Comments:
          11 pages, in Spanish language. 1 figure. In La nueva era del pódcast

- **What's New**: 이번 연구는 시청각 제작물의 사운드 디자인 교육 경험을 공유하고, 학생들이 수행한 다양한 프로젝트를 비교 분석합니다. 이 연구는 교육 방법의 비교가 아닌, 서로 다른 학년 학생들이 접하는 다양한 문제를 분석하는 데 중점을 둡니다.

- **Technical Details**: 이번 연구는 PBL (Problem-Based Learning) 방식을 활용한 학습이 다른 교육 방법, 예를 들어 강의 수업 (master classes) 보다 훨씬 더 우수한 결과를 생성한다고 발표했습니다. 이는 학생들이 창의적인 프로젝트에 개인적으로 참여하면서 기술적인 스킬을 습득하기 때문입니다. 그러나 교육자와 학생 간의 대부분의 상호작용은 기술적 교정 측면에 초점을 맞추고 있습니다. 예를 들어, 리버브 (reverb)에서의 파라미터 조정(프리 딜레이(pre-delay), 디케이(decay), 모듈레이션 등)이나, 컴프레서(compressor), 노이즈 게이트(noise gate) 조정 등이 포함됩니다.

- **Performance Highlights**: PBL 방식의 활용으로 학생들이 창의적이고 기술적인 측면에서 높은 성과를 보였으며, 기술적으로 높은 진입 장벽을 극복하는 데도 큰 도움을 주었습니다. 특히 오디오 편집 프로그램을 사용하는 데 필요한 시간과 이해도가 현저히 개선되었습니다.



### Understanding Deep Learning via Notions of Rank (https://arxiv.org/abs/2408.02111)
Comments:
          PhD thesis

- **What's New**: 이번 논문에서는 딥러닝(deep learning)에 대한 이론적 이해를 위한 핵심 개념으로 랭크(rank)를 제안하고 있습니다. 특히 일반화(generalization)와 표현력(expressiveness)에 관한 기본적인 측면에 초점을 맞추고 있습니다. 그래디언트 기반 훈련(gradient-based training)이 여러 신경망 구조에서 낮은 랭크로의 암묵적인 규제를 유도할 수 있음을 입증하고, 이러한 현상이 자연 데이터(예: 오디오, 이미지, 텍스트)의 일반화 설명에 도움이 될 수 있음을 실증적으로 보여줍니다.

- **Technical Details**: 이번 연구에서는 딥러닝과 텐서 분해(tensor factorizations) 사이의 연결 고리가 중심 도구로 사용됩니다. 랭크를 사용하여 그래프 신경망(graph neural networks)이 상호작용을 모델링할 수 있는 능력을 특성화하며, 이 랭크 개념은 양자 물리학에서 얽힘을 정량화하는 데 일반적으로 사용됩니다. 이러한 이론적 근거를 바탕으로 명시적인 규제 체계와 데이터 전처리 알고리즘을 설계할 수 있는 실용적인 방안을 제시합니다.

- **Performance Highlights**: 그래디언트 기반 훈련이 암묵적으로 낮은 랭크로의 규제를 촉진하는 현상을 통해, 자연 데이터에 대한 일반화를 설명할 수 있다는 점을 실증적으로 보여줍니다. 이는 딥러닝 모델의 일반화 특성을 이해하고, 그래프 신경망의 상호작용 모델링 능력을 평가하는 데 유의미한 인사이트(insights)를 제공합니다.



### KAN-RCBEVDepth: A multi-modal fusion algorithm in object detection for autonomous driving (https://arxiv.org/abs/2408.02088)
- **What's New**: 3D 객체 인식의 정확성을 크게 향상시키는 새로운 알고리즘 RCBEV-KAN을 소개합니다. 이 알고리즘은 카메라, LiDAR, 밀리미터파 레이더의 다중 모드 센서 데이터를 융합하여 Bird's Eye View(BEV) 기반 접근 방식을 사용하며, 트랜스포머 아키텍처를 통해 다양한 데이터 소스를 통합하고 공간 관계 처리를 개선합니다. 실험 결과, RCBEV-KAN 모델은 대부분의 인식 카테고리에서 우수한 성능을 보였으며, Mean Distance AP에서 23%, ND Score에서 17% 향상된 결과를 나타냈습니다.

- **Technical Details**: RCBEV-KAN algorithm은 다중 모드 센서 융합을 사용하여 카메라, LiDAR, 밀리미터파 레이더 데이터를 하나의 BEV 특징 공간으로 통합합니다. 이 방법은 트랜스포머 아키텍처를 사용해 다양한 데이터 소스를 매끄럽게 통합하고, Kolmogorov-Arnold Network (KAN)를 이용해 시계열 데이터의 특징 추출을 최적화합니다. 또한 DepthNet 모듈은 LiDAR 포인트 클라우드 데이터의 직접적인 감독을 받고, 밀리미터파 레이더 데이터를 추가해 깊이 학습을 향상시킵니다.

- **Performance Highlights**: RCBEV-KAN 모델은 기존의 모델들보다 뛰어난 성능을 보입니다. Mean Distance AP는 0.316에서 0.389로 23% 향상되었으며, ND Score도 0.415에서 0.484로 17% 개선되었습니다. 또한, 평가 시간은 8% 빨라져 71.28초가 소요되었습니다. 이러한 성능 향상은 복잡하고 역동적인 자율 주행 환경에서 RCBEV-KAN 모델이 더욱 신뢰성과 효율성을 갖추고 있다는 것을 입증합니다.



### Unleashing the Power of Data Tsunami: A Comprehensive Survey on Data Assessment and Selection for Instruction Tuning of Language Models (https://arxiv.org/abs/2408.02085)
Comments:
          review, survey, 28 pages, 2 figures, 4 tables

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 instruction tuning을 목적으로 활용 가능한 데이터 평가 및 선택 방법에 관한 종합적인 리뷰를 제공합니다. 기존 연구들이 데이터 평가 메트릭스 및 선택 메커니즘에 대해 구체적으로 다루지 않은 부분을 보완하고, 데이터 평가가 instruction tuning에 어떻게 통합될 수 있는지를 설명합니다.

- **Technical Details**: 데이터 평가 및 선택 방법을 크게 세 가지 관점에서 체계적으로 분류합니다: quality-based, diversity-based, importance-based 방법들입니다. 각 카테고리 내 대표적인 방법들을 상세히 기술하고, 최신 방법들의 공식 보고 결과를 바탕으로 비교를 진행합니다. 또한, 다양한 데이터셋과의 연관성을 분석하고, 커다란 데이터셋을 평가하고 일부 선택할 때 필요한 방법론들을 설명합니다.

- **Performance Highlights**: 이 논문은 다양한 데이터 평가 방법들이 실제 성능에 미치는 영향을 심층적으로 논의하며, 그 한계점들을 명확히 제시합니다. 특히, quality, diversity, importance 측면에서 데이터를 선별하여 학습 비용을 줄이면서 성능을 높일 수 있는 방안을 제시합니다. 이를 통해, 효과적이고 효율적인 instruction tuning을 위한 데이터 평가 및 선택의 중요성을 강조합니다.



### Applying Conditional Generative Adversarial Networks for Imaging Diagnosis (https://arxiv.org/abs/2408.02074)
- **What's New**: 이 연구는 Conditional Generative Adversarial Networks (C-GAN)과 Stacked Hourglass Networks (SHGN)을 결합한 혁신적인 이미지 세분화(이미지 분할) 응용을 소개합니다. 특히 의료 영상 분야에서 이미지 세분화의 성능을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: 복잡한 이미지 데이터셋에 적용된 딥러닝 모델에서 흔히 발생하는 과적합(overfitting) 문제를 해결하기 위해, 회전 및 스케일링을 통해 데이터를 증강(data augmentation)합니다. L1과 L2 재구성 손실(reconstruction losses)을 결합한 하이브리드 손실 함수(hybrid loss function)와 적대적 훈련(adversarial training)을 통해 세분화 프로세스를 개선합니다. 특히 우리의 접근법은 특정 분야의 전문 지식에 대한 의존 없이도 의료 이미지 내의 조직 경계와 혈관 구조와 같은 뚜렷한 영역을 정확하게 구분하는 기능을 가지고 있습니다.

- **Performance Highlights**: 알고리즘은 표준 의료 이미지 라이브러리를 사용하여 평가되었으며, 기존 방법들에 비해 우수한 성능 지표를 보여주었습니다. 이는 딥러닝을 통해 자동화된 의료 진단을 향상시킬 수 있는 잠재력을 입증합니다.



### Case-based reasoning approach for diagnostic screening of children with developmental delays (https://arxiv.org/abs/2408.02073)
- **What's New**: 본 연구는 CNN-트랜스포머 모델과 사례 기반 추론(Case-Based Reasoning, CBR)을 결합한 하이브리드 모델을 채택하여 발달 지연 아동의 선별 효율성을 향상시키기 위한 시스템을 개발했습니다. 이 모델은 아동의 발달 지연을 조기에 식별하고 적절한 개입을 제공함으로써 의학 자원의 낭비와 사회적 비용을 크게 줄일 수 있습니다.

- **Technical Details**: CNN-트랜스포머 모델은 이미지 특징 추출과 인식에 우수한 성능을 보이며, 골 연령 이미지를 통해 특징을 효과적으로 식별할 수 있습니다. 사례 기반 추론(CBR)은 유사한 과거 사례를 기반으로 문제를 해결하는 기법으로, 기존에 저장된 사례를 바탕으로 새로운 사례를 판단하고 비교하는 데 활용됩니다. 이러한 모델의 결합을 통해 잠재적이고 변동 가능한 특성을 가진 지원 시스템에 적합한 선별 시스템을 구축했습니다.

- **Performance Highlights**: 국제 연구에 따르면 발달 지연 아동의 조기 개입 최적 시기는 6세 이하이며, 황금 치료 기간은 3.5세 이전입니다. 조기 개입을 받은 발달 지연 아동은 증상이 현저히 개선되며, 일부는 완전히 회복할 수 있습니다. 이번 연구의 시스템은 화이베이, 안후이성의 신생아수를 바탕으로 연간 약 7,500건의 의심 사례를 더욱 효율적으로 선별할 수 있도록 설계되었습니다.



### ParkingE2E: Camera-based End-to-end Parking Network, from Images to Planning (https://arxiv.org/abs/2408.02061)
- **What's New**: 본 논문에서는 자율 주차(autonomous parking)의 효율적인 수행을 위해 인간 주행 궤적을 모방(imitation learning)하는 딥러닝 기반 종단간 계획(end-to-end planning) 방법을 제안합니다. 이 방법은 RGB 이미지에서 경로 계획(path planning)을 수행하여 전통적인 규칙 기반(rule-based) 알고리즘이 복잡한 주차 시나리오에서 겪는 한계를 극복합니다.

- **Technical Details**: 제안된 방법은 타깃 쿼리 인코더(target query encoder)를 사용하여 이미지와 타깃 피처를 융합(fuse)하고, 트랜스포머(transformer) 기반 디코더를 통해 미래의 경로 지점(waypoints)을 자동회귀적으로 예측(predict)합니다. 이는 인간 주행 데이터를 학습하여 종단간 계획을 가능하게 하고, 보다 직관적이고 다재다능한 주차 알고리즘을 구현합니다.

- **Performance Highlights**: 실험 결과, 네 가지의 서로 다른 실제 주차장에서 평균 87.8%의 성공률을 달성했습니다. 또한 실제 차량 실험을 통해 제안된 방법의 실용성과 효과상이 검증되었습니다. 이를 통해 딥러닝 기반 주차 알고리즘의 가능성을 입증했습니다.



### 3D Single-object Tracking in Point Clouds with High Temporal Variation (https://arxiv.org/abs/2408.02049)
Comments:
          Accepted by ECCV24

- **What's New**: HVTrack는 일시적 변화가 큰 3D 포인트 클라우드 데이터에서 단일 객체 추적(3D Single-Object Tracking, 3D SOT)을 위한 새로운 프레임워크입니다. 기존 방식들이 시간적 변화를 부드럽게 가정하여 실패하던 점을 개선했습니다. HVTrack는 세 가지 주요 모듈을 도입하여 이 문제를 해결합니다: 상대 자세 인식 메모리 모듈(Relative-Pose-Aware Memory module, RPM), 기본 확장 기능 교차 주의 모듈(Base-Expansion Feature Cross-Attention module, BEA), 그리고 문맥적 포인트 안내 자가 주의 모듈(Contextual Point Guided Self-Attention module, CPA)입니다.

- **Technical Details**: HVTrack는 일시적 변화가 큰 포인트 클라우드 형상 변화를 처리하기 위해 RPM 모듈을 사용합니다. RPM은 전경 마스크와 관찰 각도를 메모리에 통합하여 포인트 클라우드의 분포 변화를 학습합니다. BEA 모듈은 유사 객체 혼란(Distractions) 문제를 해결하기 위해 교차 주의에서 혼합 규모 특징을 동기화합니다. CPA 모듈은 확장된 검색 영역으로 인한 배경 소음을 억제합니다. CPA는 포인트 간 중요성에 따라 특징을 문맥적으로 집계하여 덜 중요한 포인트는 적은 문맥 정보를 공유하도록 합니다.

- **Performance Highlights**: KITTI-HV 데이터셋에서 HVTrack는 기존 최첨단 추적기인 CXTracker를 성공률(Success)에서 11.3%, 정밀도(Precision)에서 15.7% 초과합니다. 또한, Waymo와 KITTI 데이터셋에서도 뛰어난 성능을 보여주며, 다양한 시간적 변화를 효과적으로 처리할 수 있는 강력한 성능을 입증했습니다.



### Latency-Aware Resource Allocation for Mobile Edge Generation and Computing via Deep Reinforcement Learning (https://arxiv.org/abs/2408.02047)
Comments:
          5 pages, 5 figures, submitted to IEEE

- **What's New**: 모바일 엣지 컴퓨팅(Mobile Edge Computing, MEC)과 생성 인공지능(Generative AI, GAI) 기술의 융합으로 새롭게 등장한 모바일 엣지 생성 및 컴퓨팅(Mobile Edge Generation and Computing, MEGC) 시스템이 주목받고 있습니다. 이 시스템은 모바일 사용자를 위해 이질적인 서비스, 예를 들어 태스크 컴퓨팅과 콘텐츠 생성 서비스를 제공합니다. 본 연구에서는 MEGC 시스템에서의 통신, 컴퓨팅, AIGC 자원 할당 문제를 조사하며, 지연 시간 최소화를 통해 서비스 품질을 향상시키고자 합니다.

- **Technical Details**: MEGC 시스템을 구현하기 위해, 계산 서비스를 요청하는 사용자, AIGC 서비스를 요청하는 사용자, VE 서비스를 요청하는 사용자가 있으며, 각 사용자는 주기적으로 하나의 서비스를 요청합니다. 시스템은 데이터 업로드와 결과 백홀 전송으로 두 가지 주요 단계로 나뉩니다. 제안된 시스템은 대역폭 할당, 백홀 전송 전력, 컴퓨팅 자원, 그리고 태스크 오프로드 비율을 최적화하여 지연 시간을 최소화하려는 딥 러닝 강화학습 기반 알고리즘을 사용합니다.

- **Performance Highlights**: 제안된 딥 러닝 강화학습 기반 알고리즘은 두 가지 기준 알고리즘보다 낮은 지연 시간을 달성하는 성능을 보여주었습니다. 이는 모바일 사용자가 경험하는 전반적인 서비스 품질을 향상시킵니다.



### Fine-tuning multilingual language models in Twitter/X sentiment analysis: a study on Eastern-European V4 languages (https://arxiv.org/abs/2408.02044)
Comments:
          18 pages, 4 figures

- **What's New**: 이번 연구는 특히 트위터/X 데이터를 이용한 국소 작업에서 언어 모델을 미세 조정하여 러시아 및 우크라이나에 대한 감성을 분석하는 점에 중점을 두었습니다. V4 국가(체코, 슬로바키아, 폴란드, 헝가리)의 언어로 작성된 트윗을 대상으로 하여 학습 및 테스트 데이터셋을 마련했습니다.

- **Technical Details**: 이번 연구에서는 여러 대형 언어 모델(LLM)을 미세 조정하였습니다. 사용된 모델로는 BERT, BERTweet, Llama2, Llama3, Mistral 등이 있습니다. 데이터는 트위터/X의 학술 API를 통해 2023년에 수집되었으며, 헬싱키 번역기와 DeepL을 사용하여 영어로 번역된 데이터셋도 생성되었습니다. 모델의 성능은 정확도, 재현율, 정밀도, F1 점수 등의 표준 지표로 평가하였습니다.

- **Performance Highlights**: 미세 조정을 이용해 6K 다국어 트윗만으로도 인-컨텍스트 학습보다 더 나은 성과를 보였습니다. 트위터/X 코퍼스에서 테스트한 모델의 성과가 일반적인 벤치마크 결과와 종종 상관관계가 없었습니다. 영어로의 좋은 번역은 다국어 사전 학습 모델에도 원래 언어를 사용하는 것보다 우위를 제공했습니다. 일부 모델은 언어 및 문화적 특수성이 반영된 예기치 않은 차이를 보였습니다.



### Self-Introspective Decoding: Alleviating Hallucinations for Large Vision-Language Models (https://arxiv.org/abs/2408.02032)
- **What's New**: 최근 LLMs(Large Language Models)의 성공을 다룬 많은 연구들이 LVLMs(Large Vision-Language Models)로 확장되고 있습니다. 이 논문에서는 LVLMs의 주요 문제점인 '환각(hallucination)' 문제를 다루고자 소개된 'Self-Introspective Decoding(SID)' 방법에 대해 발표했습니다. SID는 외부 지식 없이도 환각 문제를 해결할 수 있는 간단하지만 효과적인 방법입니다.

- **Technical Details**: SID는 'Context and Text-aware Token Selection(CT2S)' 전략을 도입하여 LVLMs의 초기 레이어에서 중요하지 않은 비전 토큰을 선택적으로 유지합니다. 이를 통해 텍스트 정보에 따라 적응적으로 증폭된 환각을 사용자가 선택하여 원래 토큰 로짓(logits)에서 증폭된 비전-텍스트 연관성을 빼내어 깨끗한 디코딩을 수행합니다. 이는 다양한 메트릭에서 높은 품질의 텍스트를 생성하며, 추가적인 지식 없이도 효과적입니다.

- **Performance Highlights**: 광범위한 실험 결과, SID는 기존의 대비 디코딩(contrastive decoding) 방법보다 낮은 환각 수준을 보이며 더 높은 품질의 텍스트를 생성합니다. 또한, SID는 큰 추가 컴퓨팅 비용 없이도 효과적입니다. 전체 결과는 https://github.com/huofushuo/SID 에서 확인할 수 있습니다.



### Mining Path Association Rules in Large Property Graphs (with Appendix) (https://arxiv.org/abs/2408.02029)
- **What's New**: 이 논문에서는 경로 규칙 연결 탐색(Path Association Rule Mining, PARM)이라는 새로운 문제를 소개합니다. 이는 두 정점 사이의 도달 경로를 분석하여 정점 속성과 에지 레이블로 규명된 경로 패턴이 어떻게 서로 공존하는지를 발견하는 작업입니다. 기존의 그래프 연관 규칙 탐색 방법들이 갖는 제한사항을 극복하고 더 넓은 응용 가능성을 보장하기 위해 새로운 알고리즘 PIONEER를 제안합니다.

- **Technical Details**: PARM은 큰 그래프 내에서 두 정점을 연결하는 도달 경로에 적용됩니다. 이 경우 도달 경로는 정점 속성과 에지 레이블로 식별된 경로 패턴이 서로 어떻게 규칙적으로 공존하는지를 발견합니다. 효율적이고 확장 가능한 PARM을 구현하기 위해, 반단조성(Anti-monotonicity) 속성을 활용하여 탐색 공간을 효과적으로 가지치기 하는 알고리즘 PIONEER를 개발했습니다. 또한, 근사 기술과 병렬화를 사용하여 확장성 있는 경로 연관 규칙 탐색을 달성했습니다.

- **Performance Highlights**: 실제 세상 데이터 집합을 사용한 실험 연구를 통해 경로 연관 규칙의 중요성과 제안된 솔루션의 효율성을 검증했습니다. PIONEER 알고리즘은 기존의 방법들보다 뛰어난 성능을 보이며, 다양한 응용 분야에서 실용적인 해결책을 제공합니다.



### Contrastive Learning-based Chaining-Cluster for Multilingual Voice-Face Association (https://arxiv.org/abs/2408.02025)
- **What's New**: 최근의 연구는 다언어 환경에서 얼굴과 목소리 간의 관계에 큰 관심을 갖게 되었습니다. 본 논문은 FAME 2024 챌린지를 위해 새롭게 제안된 대조 학습 기반의 체인 클러스터 방식으로 얼굴-목소리 연관성을 향상시키는 솔루션을 소개합니다. 이 방법론은 얼굴 특징과 음성 특징 간의 바이오메트릭 관계를 구축하고, 여러 언어간의 음운학적 상관관계를 모델링하는 도전에 직면하고 있습니다. 특히, 본 연구는 다언어 시나리오에서 목소리와 얼굴의 강력한 연관성을 구축하기 위해 Supervised Cross-Contrastive (SCC) 학습을 도입하고 있으며, 데이터에서 발생할 수 있는 이상치(outlier)를 효과적으로 처리하기 위해 체인 클러스터 기반의 후처리 단계를 설계하였습니다.

- **Technical Details**: 제안된 방법은 목소리와 얼굴 이미지 특징 간의 상당한 모달리티 차이를 고려하여, 이러한 멀티모달 표현을 동일한 공간에 맞추기 위한 두 개의 네트워크를 설계하였습니다. 대조 학습 (contrastive learning)의 긍정 샘플은 같은 사람이 다른 언어로 말하는 경우 목소리-얼굴 페어로 정의되며, 중간 단계에서 체인 클러스터링을 통해 남성 및 여성 클러스터로 나누고 가까운 클러스터 중심에 있는 샘플들을 정밀하게 클러스터링하여 높은 자신도의 클러스터 결과를 이용해 초기 테스트 점수를 정제합니다.

- **Performance Highlights**: 우리의 방법은 다언어 시나리오에서 얼굴-목소리 연관 작업에 대해 높은 성능을 달성하였으며, 이는 실세계의 다언어 환경에서의 효과성과 강력함을 증명했습니다. FAME 2024 챌린지에서 2위를 차지하여 제안된 방법의 우수성을 입증하였습니다.



### Scenario-based Thermal Management Parametrization Through Deep Reinforcement Learning (https://arxiv.org/abs/2408.02022)
Comments:
          8 pages, 7 figures, 2 tables, 1 algorithm, 10 equations, conference

- **What's New**: 새로운 연구에서는 배터리 전기차(이하 BEV)의 열 시스템(Thermal System) 제어를 위한 학습 기반 튜닝 접근 방식을 소개합니다. 이 접근 방식은 시나리오 생성(Scenario Generation)을 통해 차량 사용 시나리오 전반에 걸쳐 강력한 성능을 보장합니다. 딥 강화 학습(Deep Reinforcement Learning)을 활용하여 내장된 파라미터 셋의 이미지 기반 해석을 포함하여, 밸브 컨트롤러 파라미터 최적화를 실현합니다.

- **Technical Details**: 연구는 열 시스템 관련 피드백 컨트롤러의 파라미터 최적화 문제를 다룹니다. 현재 방법론은 시간, 인력, 실험 등이 많이 소모되지만, 제안된 학습 기반 접근 방식은 자동화된 시나리오 생성을 통해 효율성을 높입니다. 또한, 딥 강화 학습 에이전트는 시나리오에 따라 파라미터를 최적화하며, 이미지 기반의 파라미터 셋 해석 방식도 도입되었습니다. 밸브의 회전 피스톤 각도와 이를 조절하는 온도 조절 시스템의 상호작용이 연구의 중점이 됩니다.

- **Performance Highlights**: 제안된 접근 방식을 실제 차량 테스트에서 검증한 결과, 기존의 방법론에 비해 경쟁력 있는 성능을 보였습니다. 이는 자동차 산업에서 열 관리 기능의 가상 개발과 대규모 파라미터 튜닝의 잠재력을 크게 향상시킬 수 있음을 의미합니다.



### Individualized multi-horizon MRI trajectory prediction for Alzheimer's Diseas (https://arxiv.org/abs/2408.02018)
Comments:
          MICCAI 2024 LDTM workshop

- **What's New**: 최근 발표된 논문에서는 MRI를 통해 측정되는 신경 퇴행을 알츠하이머병(AD) 진단의 잠재적 바이오마커로 활용할 수 있음을 제안했습니다. 그러나 기존의 MRI는 아밀로이드(Amyloid)나 타우(Tau) 기반의 바이오마커만큼 명확하지 않다는 한계가 있습니다. 이에 연구팀은 조절된 변형 오토인코더(Conditional Variational Autoencoder, CVAE)를 활용하여 개인화된 MRI 예측 모델을 개발하였습니다. 이 모델은 환자의 나이, 질병 상태, 그리고 이전 스캔을 기반으로 향후 최대 10년 이내의 MRI 변화를 예측할 수 있습니다.

- **Technical Details**: 이 연구에서는 알츠하이머병 신경영상 이니셔티브(ADNI)와 개방 접근 시리즈 이미지 연구(OASIS)에서 수집한 일련의 이미징 데이터를 사용해 새로운 아키텍처를 훈련시켰습니다. 이 아키텍처는 나이, 질병 상태, 이전 스캔을 조건으로 한 데이터 입력을 받아 복잡한 픽셀 변화를 모델링 합니다. 특히, 이중 인코더(Double-Encoder) CVAE 아키텍처가 도입되어 더 현실적이고 높은 해상도의 출력을 제공합니다. 이 모델은 멀티-호라이즌(Multi-Horizon) 예측을 가능하게 하여, 임의의 시간 간격에 대한 예측도 가능합니다.

- **Performance Highlights**: 모델은 ADNI와 OASIS 데이터셋에서 높은 해상도의 개인화된 이미지를 생성하는 데 성공하였으며, 기존 다양한 모델과 비교해 더 나은 성능을 보였습니다. 테스트셋과 외부 독립 데이터셋에서 평균 제곱 오차(MSE) 기준으로 높은 예측 정확도를 입증했습니다. 또한, 후속 MRI를 이미 가지고 있는 경우에는 질병 상태 분류기(Classifier)를 구축할 수 있는 가능성도 제시되었습니다. 이는 AD 초기 진단을 도울 수 있으며, 치료 효과 추정에서도 대조 기준으로 사용될 수 있습니다.



### Joint Learning of Emotions in Music and Generalized Sounds (https://arxiv.org/abs/2408.02009)
Comments:
          Accepted at Audio Mostly 2024, Milan

- **What's New**: 이번 연구는 일반 소리와 음악이 공통된 감정 공간(shares common emotional space)을 공유할 수 있는지를 조사하여, 흥분도(arousal)와 유쾌도(valence)의 감정 예측 정확성을 향상시키고자 합니다. 이를 위해 다수의 데이터셋을 이용한 멀티 도메인 학습(multi-domain learning)을 제안하며, 감정 유발을 특징짓는 공통 공간을 생성합니다. 본 연구는 IADS-E 및 PMEmo라는 공개 데이터셋을 사용하며, 최신 실험 프로토콜을 따릅니다.

- **Technical Details**: 본 연구는 다양한 오디오 구조 특징을 포착할 수 있는 스펙트럼(spectrum), 에너지(energy), 보이싱(voicing) 등의 키 파라미터를 활용합니다. 이후 이들 공통 특징 공간에서 이종 모델 아키텍처(heterogeneous model architectures)를 활용해 공동 학습을 수행합니다. IADS-E와 PMEmo 데이터셋의 특징은 각각 환경 소리 및 음악 구조의 보편적 감정을 수집한 것입니다. 오디오 샘플을 수치적 특징으로 변환하기 위해 openSMILE 툴킷을 사용하였으며, ComParE 2013 설정을 적용했습니다.

- **Performance Highlights**: 연구 결과, 제안된 멀티 도메인 학습 접근법은 기존의 최첨단 기술(state-of-the-art)을 능가하였습니다. 특히, 통합된 데이터셋을 이용한 경우, 감정 예측의 정확성이 향상되었음을 확인할 수 있었습니다. 이 접근법은 다양한 소리 유형에서 감정을 더욱 정교하게 예측하는 데 기여할 수 있습니다.



### Reinforcement Learning for an Efficient and Effective Malware Investigation during Cyber Incident Respons (https://arxiv.org/abs/2408.01999)
Comments:
          v1.1

- **What's New**: 이 연구는 강화 학습(RL)을 사용하여 사건 후 악성코드 포렌식 조사를 향상시키는 데 초점을 맞추었습니다. 새로운 MDP 기반 포렌식 조사 모델과 프레임워크를 제안하여 사건 후 조사를 신속하게 처리하고자 했습니다. 이는 전통적인 방법에 비해 지속적인 학습 및 새로운 악성코드에 대한 적응 능력을 제공함으로써 사건 후 포렌식 조사를 크게 향상시켰습니다.

- **Technical Details**: 구조화된 MDP 내에서 강화 학습(RL) 모델을 구현했습니다. RL 에이전트는 Q 테이블과 시간 차 학습(temporal difference learning)을 사용하여 포렌식 증거 파일을 획득하고 분석하면서 반복적으로 능력을 개선했습니다. 에이전트는 에플실론 탐색 전략(epsilon greedy exploration strategy)과 Q 학습 업데이트를 통해 효율적으로 학습하고 의사 결정을 내렸습니다. 최적의 학습 속도는 MDP 환경의 복잡성에 따라 달라지며, 간단한 환경에서는 빠른 수렴을 위해 높은 학습 속도가 필요하고 복잡한 환경에서는 안정성을 위해 낮은 학습 속도가 필요함을 실험적으로 확인했습니다.

- **Performance Highlights**: 우리 모델은 악성코드를 식별 및 분류하는데 있어 인간 전문가보다 더 빠르게 악성코드 분석 시간을 줄이는 성과를 보였습니다. 이 연구는 하이퍼파라미터 튜닝(hyper parameter tuning)의 중요성을 강조하고 복잡한 환경을 위한 적응 전략을 제안했습니다. RL 기반 접근 방식은 유망한 결과를 산출했으며, 새로운 악성코드 위협에 대한 지속적인 학습과 적응을 통하여 사건 후 포렌식 조사를 크게 향상시킬 수 있는 대안으로 확인되었습니다.



### MetaWearS: A Shortcut in Wearable Systems Lifecycle with Only a Few Shots (https://arxiv.org/abs/2408.01988)
- **What's New**: 이번 연구에서는 MetaWearS라는 메타 러닝(Meta-learning) 방법을 제안하고 있습니다. 이 방법은 웨어러블 시스템의 초기 데이터 수집 요구를 감소시키고, 프로토타입 갱신 메커니즘(prototypical updating mechanism)을 통해 모델 업데이트를 간소화하여 전체 모델 재학습의 필요성을 제거합니다.

- **Technical Details**: MetaWearS는 소수의 샘플로 모델을 미세 조정하는 few-shot learning 전략을 사용하며, 프로토타입 네트워크(prototypical networks)의 원리를 적용하여 업데이트 시 단일 벡터인 프로토타입(prototype)만 수정합니다. 이를 통해 모델 업데이트 시 에너지 소비를 크게 줄일 수 있습니다. 본 연구에서는 Electroencephalogram (EEG)와 Electrocardiogram (ECG) 신호를 기반으로 한 간질 발작(epileptic seizures) 및 심방 세동(atrial fibrillation) 감지의 두 가지 사례 연구를 통해 접근법을 평가하였습니다.

- **Performance Highlights**: MetaWearS는 추가 라벨링된 데이터 16분으로 모델을 업데이트할 때 AUC를 최대 5.3%까지 향상시킬 수 있으며, model updates를 위한 에너지 소비를 간질 감지의 경우 456배, 심방 세동 감지의 경우 418배 줄였습니다. 또한, 간질 발작 감지에서 70%, 심방 세동 감지에서 82%의 AUC를 달성하였습니다.



### DeMansia: Mamba Never Forgets Any Tokens (https://arxiv.org/abs/2408.01986)
- **What's New**: 이번 논문은 Transformer 아키텍처의 수학적 기초를 검토하고, 특히 긴 시퀀스를 처리하는 데 있어 이들의 한계를 조명합니다. Mamba, Vision Mamba(ViM), LV-ViT 모델을 바탕으로 한 새로운 아키텍처인 DeMansia를 제안합니다. DeMansia는 상태 공간 모델(state space model)과 토큰 레이블링(token labeling) 기법을 통합하여 이미지 분류 성능을 향상시키며, 전통적인 Transformer가 갖고 있는 계산 비용 문제를 효율적으로 해결합니다. 논문에서 제안된 아키텍처는 GitHub에서 구현된 소스를 확인할 수 있습니다.

- **Technical Details**: Transformer 아키텍처들은 자가 주의 메커니즘(self-attention mechanism)을 사용하여 입력 데이터의 다양한 부분을 동적으로 가중하여, 더 세밀하고 상황에 맞는 해석을 가능하게 합니다. 하지만 시퀀스 길이가 길어질수록 계산 복잡도가 제곱에 비례하여 증가한다는 한계가 있습니다. DeMansia는 Mamba와 Vision Mamba(ViM)의 장점을 결합하고, LV-ViT의 트레이닝 파이프라인을 참고하여 이미지 분류 작업에서의 성능을 높이기 위해 고안되었습니다. 상태 공간 모델(state space model)과 토큰 레이블링(token labeling)을 혁신적으로 적용하여 맥락의 풍부함을 유지하면서도 계산 효율을 확보합니다.

- **Performance Highlights**: DeMansia 아키텍처는 기존의 모델들과 비교하여 효과적인 성능을 보여주었습니다. 특히 이미지 분류 작업에서 높은 성능을 보이며, 자원이 제한된 환경에서도 고성능을 유지합니다. 이러한 성능은 현대의 모델들과 비교한 벤치마크 결과에서 입증되었습니다.



### ML-EAT: A Multilevel Embedding Association Test for Interpretable and Transparent Social Scienc (https://arxiv.org/abs/2408.01966)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 이번 연구는 언어 기술에서 내재된 편향(Intrinsic Bias)을 해석 가능하고 투명하게 측정하기 위해 다수준 임베딩 연결 테스트(Multilevel Embedding Association Test, ML-EAT)를 소개합니다. ML-EAT는 WEAT (Word Embedding Association Test)의 한계를 극복하며, 세 가지 세분화 수준에서 편향을 정량화합니다. 새로운 EAT-맵(EAT-Map) 시각화를 통해 결과를 보다 직관적으로 볼 수 있게 했습니다.

- **Technical Details**: ML-EAT는 세 가지 수준에서 편향을 측정합니다. 첫 번째 수준에서는 두 개의 대상 개념과 두 개의 속성 개념 간의 차등 연결을 측정합니다. 두 번째 수준에서는 각 대상 개념과 두 개의 속성 개념 간의 개별 효과 크기를 측정합니다. 세 번째 수준에서는 네 개의 기본 코사인 유사도 분포의 평균과 표준 편차를 측정합니다. 연구자들은 이를 통해 EAT 패턴을 아홉 가지로 분류하고, EAT-맵이라는 네 부분 시각화를 통해 이해를 돕습니다.

- **Performance Highlights**: ML-EAT를 사용하는 실증 분석에 따르면, 기존의 WEAT로는 관찰할 수 없는 추가 정보를 제공하며, 제로샷 모델(Zero-Shot Models)에서 프롬프트의 영향을 드러내고, 코사인 유사도가 효과적이지 않은 상황을 식별할 수 있습니다. 이를 통해 언어 기술의 편향을 보다 관찰 가능하고 해석 가능하게 만듭니다. 이러한 종합 분석은 정적 및 연대기적 단어 임베딩, GPT-2 언어 모델 및 CLIP 언어-이미지 모델에 성공적으로 적용되었습니다.



### Top K Enhanced Reinforcement Learning Attacks on Heterogeneous Graph Node Classification (https://arxiv.org/abs/2408.01964)
- **What's New**: 최근 그래프 신경망(Graph Neural Networks, GNNs) 연구에서 이종 그래프(heterogeneous graphs) 데이터에 대한 검증이 부족한 상황을 해결하기 위해 새로운 공격 방법론인 HeteroKRLAttack을 제안했습니다. HeteroKRLAttack은 강화 학습(reinforcement learning)과 Top-K 알고리즘을 결합해 이종 그래프에서의 노드 분류 작업을 방해하는 효과적인 공격 전략을 효율적으로 식별합니다.

- **Technical Details**: 이 연구에서는 이종 그래프의 구조와 매개변수에 대한 사전 지식이 필요하지 않은 블랙박스 공격(black-box attack)을 가정합니다. 강화 학습 프레임워크 내에서 공격자는 에이전트로서 최적의 공격 전략을 학습합니다. 각 공격 단계에서 에이전트는 노드 또는 엣지를 수정하는 행동을 선택해 누적 보상을 극대화합니다. 여기서 보상 함수는 공격 후 모델의 분류 정확도 감소를 기반으로 정의됩니다. 추가적으로, Top-K 알고리즘을 소개해 광대한 액션 공간(action space)을 줄였습니다. K-D 트리 알고리즘을 사용해 피처 공간에서 K개의 가장 가까운 노드를 잠재적 공격 대상으로 선택합니다.

- **Performance Highlights**: 다양한 공개 이종 그래프 데이터셋에서 실험을 수행한 결과, 제안된 블랙박스 공격 방법이 노드 분류 정확도를 크게 낮추는 것으로 나타났습니다. 이 연구는 현재 모델의 잠재적 취약점을 강조하고, 향후 이종 그래프에 대한 방어 전략 개발에 중요한 지침을 제공합니다.



### The Implications of Open Generative Models in Human-Centered Data Science Work: A Case Study with Fact-Checking Organizations (https://arxiv.org/abs/2408.01962)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 이 논문은 개방형 생성 언어 모델(Open Generative Language Models)의 사회적 영향을 조사하고 있습니다. 특히, 팩트체킹 조직이 대규모 순환하는 잘못된 정보를 관찰하고 분석하는 데 이러한 모델을 활용하는 방법을 살펴보았습니다. 이를 위해, 여섯 대륙에 걸쳐 20개의 팩트체킹 조직에서 24명의 전문가들을 인터뷰했습니다.

- **Technical Details**: 인터뷰를 통해 팩트체킹 조직이 데이터 수집 (Data Ingestion), 데이터 분석 (Data Analysis), 데이터 검색 (Data Retrieval), 데이터 제공 (Data Delivery), 데이터 공유 (Data Sharing) 과정에서 생성 모델을 사용하는 5가지 구성 요소 개념 모델을 제안했습니다. 개방형 모델을 선호하는 이유로는 조직 자율성 (Organizational Autonomy), 데이터 프라이버시 및 소유권 (Data Privacy and Ownership), 응용 프로그램 특화 (Application Specificity), 및 역량 투명성 (Capability Transparency)를 들었으며, 폐쇄형 모델을 사용하는 이유로는 성능 (Performance), 사용성 (Usability), 안전성 (Safety) 및 기회 비용 (Opportunity Costs)를 꼽았습니다.

- **Performance Highlights**: 팩트체킹 조직이 사용하는 개방형 모델은 조직 자율성과 데이터 프라이버시 등에서 높은 선호도를 보였으나, 성능과 사용성 면에서 폐쇄형 모델에 비해 여전히 부족한 점이 많았습니다. 개방형 모델의 성능 및 안전성 향상을 위한 연구가 필요하며, 폐쇄형 모델의 투명성, 기관 및 데이터 특화에 관한 연구도 함께 제안되었습니다.



### Representation Bias of Adolescents in AI: A Bilingual, Bicultural Study (https://arxiv.org/abs/2408.01961)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 이 논문은 미국과 네팔 청소년들이 AI에 의해 어떻게 묘사되는지와 그들이 어떤 묘사를 선호하는지를 연구합니다. 연구는 기존의 static word embeddings (SWEs)와 생성형 언어 모델 (GLMs)을 통해 학습된 청소년에 대한 편견을 분석합니다. 청소년 관련 편견이 미국과 네팔에서 어떻게 상이하게 나타나는지를 비교하며, 청소년들이 실제로 자신들이 어떻게 묘사되기를 바라는지에 대해서도 논의합니다.

- **Technical Details**: 영어 SWEs는 청소년들을 사회 문제와 연관짓는 경향이 있으며, 사전학습된 GloVe SWE에서 청소년들과 가장 많이 연관된 1,000 단어 중 50% 이상이 사회 문제와 관련이 있습니다. GPT2-XL과 LLaMA-2-7B GLMs는 제공된 청소년 관련 프롬프트에 대해 각각 30%와 29% 비율로 사회 문제를 언급하는데, 주로 폭력, 약물 사용, 정신 질환, 성적 금기 등입니다. 네팔어 모델의 경우 이러한 연관성이 덜 나타납니다. 또한, 워크샵에서 미국과 네팔 청소년들은 AI의 청소년 묘사가 실제 청소년의 생활과 동떨어져 있으며, 학교와 우정 같은 활동에 더 초점을 맞춰야 한다고 언급했습니다.

- **Performance Highlights**: 연구 데이터는 미국 청소년 13명과 네팔 청소년 18명을 대상으로 워크샵을 통해 수집되었습니다. 청소년들이 AI가 자신들을 공정하게 묘사하기 위해서는 다양성을 강조하거나 긍정적인 면을 중심으로 하여야 한다고 제안했습니다. 청소년들은 AI가 미디어 소스 대신 청소년들로부터 학습한다면 편견을 줄이는 데 도움이 될 것이라고 낙관적인 전망을 제시했습니다. 이 연구는 SWEs와 GLMs가 청소년을 잘못 묘사하는 방식을 이해하는 데 도움을 주고, 덜 선정적인 특징화를 위한 템플릿을 제공합니다.



### AnomalySD: Few-Shot Multi-Class Anomaly Detection with Stable Diffusion Mod (https://arxiv.org/abs/2408.01960)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문에서는 산업 제조에서 결함 부품을 식별하기 위한 필수 작업인 이상 감지(anomaly detection)에서 새로운 접근 방식을 제안합니다. AnomalySD라 불리는 이 프레임워크는 Stable Diffusion(SD) 모델을 활용하여 최소한의 정상 데이터(few-shot)만으로도 다양한 이상을 감지할 수 있도록 설계되었습니다. 주요 기여 점으로는 SD 모델의 텍스트 설명과 전경 마스크 기법을 이용하여 이상 영역을 정상으로 대체하며, 다중 스케일 마스크 전략과 프로토타입 기반 마스크 전략을 도입하여 다양한 이상 영역을 정확하게 마킹 및 인페인팅(inpainting)하는 방식을 제안합니다.

- **Technical Details**: 제안된 AnomalySD 프레임워크는 몇 가지 핵심 기술로 구성되어 있습니다. 먼저, SD 모델에 적응시키기 위해 계층적 텍스트 설명(hierarchical text descriptions)과 전경 마스크(foreground mask) 메커니즘을 설계하여 모델 미세 조정을 수행합니다. 추론 단계에서는 다중 스케일 마스크 전략(multi-scale mask strategy)과 프로토타입 기반 마스크 전략(prototype-guided mask strategy)을 적용하여 다양한 이상 영역을 마킹합니다. 그런 다음 이를 통해 얻은 모든 마스크의 인페인팅 결과를 기반으로 이상 점수(anomaly score)를 추정합니다.

- **Performance Highlights**: 제안된 AnomalySD 프레임워크는 MVTec-AD와 VisA 데이터셋 실험에서 뛰어난 성능을 보였습니다. MVTec-AD 데이터셋에서는 다중 클래스(multi-class)와 원샷(one-shot) 설정에서 각각 93.6%와 94.8%의 AUROC를 기록하였고, VisA 데이터셋에서는 86.1%와 96.5%의 AUROC를 기록하였습니다. 이러한 결과는 AnomalySD가 최소한의 정상 데이터만으로도 높은 성능을 보유함을 증명합니다.



### Dataset Scale and Societal Consistency Mediate Facial Impression Bias in Vision-Language AI (https://arxiv.org/abs/2408.01959)
Comments:
          Accepted at Artificial Intelligence, Ethics, and Society 2024

- **What's New**: 본 연구에서는 43개의 CLIP (Contrastive Language-Image Pretraining) 비전-언어 모델을 분석하여 이들이 인간과 유사한 얼굴 인상 편향(facial impression biases)을 학습하는지를 평가하였습니다. 결과적으로 이러한 편향이 세 가지 구별되는 CLIP 모델 계열에서 나타남을 확인하였습니다.

- **Technical Details**: 연구에서는 비관측 시각적 속성(예: 신뢰성, 성적 지향성) 관련 인상 편향이 가장 큰 데이터셋에서 훈련된 모델들에서만 나타난다는 사실을 발견하였습니다. 이는 더 많은 데이터와의 적합성이 더 미세한 사회적 편향을 재현하는 결과를 초래한다는 것을 시사합니다. 또한, 계층적 클러스터링 방법(hierarchical clustering approach)을 통해 데이터셋의 크기가 편향 구조의 유사성 정도를 예측할 수 있음을 보여주었습니다. 마지막으로, 텍스트 인코더로써 CLIP을 사용하는 Stable Diffusion 모델들이 얼굴 인상 편향을 학습하고 있으며 이 편향이 Stable Diffusion XL-Turbo 모델에서 인종적 편향과 교차됨을 발견하였습니다.

- **Performance Highlights**: 이번 연구는 CLIP 모델들이 얼굴 인상 편향을 학습하는 메커니즘을 최초로 규명하였으며, 이를 통해 이러한 모델들이 편향 연구에 중요한 도구가 될 수 있음을 시사합니다. 다만, 해당 모델들을 범용적으로 사용하기 위해서는 데이터셋의 엄격한 큐레이션이 필요합니다.



### Defining and Evaluating Decision and Composite Risk in Language Models Applied to Natural Language Inferenc (https://arxiv.org/abs/2408.01935)
Comments:
          arXiv admin note: text overlap with arXiv:2310.03283

- **What's New**: 대형 언어 모델(LLM)인 ChatGPT 등의 성능은 매우 인상적이지만, 잘못된 자신감(과잉 또는 과소 자신감)에 의해 중요한 위험을 초래할 수 있습니다. 이 논문은 이러한 비대칭 문제를 해결하기 위해, 두 가지 위험 유형(결정 위험과 복합 위험)을 정의하고, 이 위험을 측정하기 위한 실험 프레임워크를 제안합니다.

- **Technical Details**: 이 논문에서는 두 레벨의 추론 아키텍처를 기반으로 하는 실험 프레임워크를 제안합니다. 첫 번째 레벨은 언어 모델이 추론을 보류해야 하는지 여부를 결정하는 '결정 규칙'에 기반하고, 두 번째 레벨은 모델이 보류하지 않을 경우 실행되는 모델의 추론입니다. 이를 통해 LLM의 리스크를 체계적으로 평가할 수 있습니다.

- **Performance Highlights**: 네 가지 자연어 상식 추론 데이터셋에서 실험한 결과, 제안된 프레임워크를 통해 LLM은 기존 방법이 고위험으로 잘못 분류하는 과제를 20.1% 추가로 자신 있게 응답하고, 잘못 답했을 과제의 19.8%를 건너뛸 수 있는 것으로 나타났습니다. 이로써 LLM의 결정 및 복합 위험을 각각 25.3%와 16.6%까지 감소시킬 수 있음을 보였습니다.



### DiReCT: Diagnostic Reasoning for Clinical Notes via Large Language Models (https://arxiv.org/abs/2408.01933)
Comments:
          9 pages,6 figures

- **What's New**: 다양한 의료 작업에 관한 진단 추론 능력과 모델 해석 가능성을 평가하기 위해 새롭게 'DiReCT' 데이터셋이 소개되었습니다. DiReCT 데이터셋은 의사들이 각 임상 기록에 세밀하게 주석을 달아 진단 추론 과정을 명확히 설명하는 자료를 포함하고 있습니다.

- **Technical Details**: {'대상': 'mimic-IV 데이터베이스에서 추출된 521개의 임상 기록', '구성': '각 임상 기록은 의사들이 주석을 단 관찰, 진단 추론 과정, 최종 진단을 포함합니다.', '특징': '기존 진단 지침에 기반한 진단 지식 그래프 제공', '구조': 'SOAP 형식으로 구성된 임상 기록 (주관적 정보, 객관적 정보, 평가, 계획)'}

- **Performance Highlights**: 현재의 최첨단 LLM은 인간 의사에 비해 여전히 진단 추론 능력에서 큰 차이가 있음이 밝혀졌습니다. 특히, 실제 임상 시나리오에서 효과적으로 추론하는 능력이 부족한 것이 확인되었습니다.



### A Semi-supervised Multi-channel Graph Convolutional Network for Query Classification in E-commerc (https://arxiv.org/abs/2408.01928)
Comments:
          Accepted by WWW2024

- **What's New**: 본 논문에서는 신규로 제안된 반지도 학습 기반 다중 채널 그래프 컨볼루션 네트워크 (Semi-supervised Multi-channel Graph Convolutional Network, SMGCN) 모델을 소개합니다. 이 모델은 레이블 연관성과 반지도 학습을 통해 기존 쿼리 의도 분류(query intent classification)의 문제점을 해결합니다.

- **Technical Details**: SMGCN은 카테고리 간 상관관계(co-occurrence)와 의미적 유사성(severity similarity) 그래프를 활용하여 카테고리 간의 관계를 강화하고 자동 생성된 라벨의 불안정성을 약화시킵니다. 이를 위해 다중 채널 GCN이 관계를 모델링하고 쿼리와 카테고리 간의 유사성 점수를 계산한 후, 클릭 라벨과 결합하여 손실 값을 계산합니다. 이 접근 방식은 감소된 데이터의 한계를 보완하며 관련 카테고리를 더 잘 회상할 수 있도록 돕습니다.

- **Performance Highlights**: 대규모 실제 데이터셋에 대한 오프라인 및 온라인 A/B 테스트 실험 결과, SMGCN은 기존 강력한 모델들보다 현저히 우수한 성능을 보였습니다. 해당 모델은 상용 전자상거래 플랫폼에 도입되어 매일 수억 건의 요청을 처리하고 있으며, 큰 상업적 가치를 제공합니다.



### Partial-differential-algebraic equations of nonlinear dynamics by Physics-Informed Neural-Network: (I) Operator splitting and framework assessmen (https://arxiv.org/abs/2408.01914)
Comments:
          61 pages, 52 figures

- **What's New**: 중립적인 미분 방정식을 다루는 새로운 형태의 물리 기반 신경망(PINN)을 제안하였습니다. 비선형 Kirchhoff 막대를 사용하여 시연하는 이 방법은 파생된 연산자 분할 기법을 기반으로 합니다. DeepXDE 오픈 소스 프레임워크에서 발생하는 병리학적 문제를 해결하기 위한 새로운 방법들도 포함되었습니다.

- **Technical Details**: 제안된 PINN 방법들은 낮은 수준에서 높은 수준으로 진화하는 PDE 형태를 사용합니다. 전통적으로 가장 높은 수준의 형태에서 시작하여 낮은 수준의 형태로 단계적으로 대체 과정을 통해 도출하지만, 제안된 방법은 높은 수준의 형태를 직접 사용합니다. JAX를 기반으로 스크립트를 개발했으며 이는 TensorFlow 백엔드를 사용하는 DDE-T보다 병리학적 문제가 적지만 더 느립니다. DDE-T는 높은 수준의 형태에서 더 효율적이기 때문에 높은 수준의 형태를 사용하는 것이 더욱 매력적입니다.

- **Performance Highlights**: 네트워크 학습 과정의 최적화 실행 경험을 체계적으로 정리하여 독자들이 결과를 재현할 수 있도록 했습니다. JAX 기반 스크립트는 병리학적 문제가 적지만 효율성에서 DDE-T에 비해 느림을 보였습니다. 이는 높은 수준의 형태를 사용하는 것의 이점 중 하나입니다.



### The Artificial Intelligence Disclosure (AID) Framework: An Introduction (https://arxiv.org/abs/2408.01904)
Comments:
          5 pages

- **What's New**: 새로운 논문에서는 고등 교육 및 연구에서 생성적 인공지능 도구(Generative AI)의 사용 투명성과 사용 기여도에 대한 요구를 해결하기 위한 'Artificial Intelligence Disclosure (AID) Framework'를 소개합니다. 이 프레임워크는 교육 및 연구에서 GenAI 사용을 공개하는 데 도움을 줄 종합적이고 상세한 지침을 제공합니다.

- **Technical Details**: 지금까지는 인공지능 도구 사용을 공개하는 데 비추천적 메모를 포함하는 것만 권장되었으며, 메모 내용에 대한 구체적인 지침이 없었습니다. 이에 따라, AID Framework는 GenAI 사용에 대한 투명성과 명확성을 확보하기 위해 표준화된 형식 및 내용 지침을 제공합니다.

- **Performance Highlights**: 이 프레임워크는 AI 도구가 어떻게 사용되었는지, 어떤 기여를 했는지 명확히 밝히도록 지원하여, 학술 및 연구 문맥에서의 AI 사용 문제를 해결하는 데 중점을 둡니다.



### Re-ENACT: Reinforcement Learning for Emotional Speech Generation using Actor-Critic Strategy (https://arxiv.org/abs/2408.01892)
Comments:
          7 pages, 10 figures

- **What's New**: 이 논문에서는 주어진 음성 신호의 프로소디 특징(박자, 강도, 리듬)을 존중하면서 수정하기 위해 배우-평가자 강화 학습(actor-critic reinforcement learning) 전략을 사용한 최초의 방법을 제안합니다. 베이지안 프레임워크를 사용하여 인간의 감정 인식과 연결된 중요한 연속 부분을 식별하고, 그 부분을 감정 예측에 사용합니다. 이후 마스킹된 부분의 프로소디 특징을 수정하여 목표 감정 점수를 높이는 방법을 사용합니다.

- **Technical Details**: 제안된 모델은 Bernoulli 무작위 변수의 변이 후백을 생성하는 신경망을 훈련시키고, 연속성을 보장하기 위해 마코프 사전(Markov prior)을 적용합니다. 음성 리듬을 조작하기 위해 WSOLA(Wave Similarity Overlap Add) 연산을 통해 시간 스케일을 변경합니다. 강화 학습을 통해 연속된 음성 신호의 가장 중요한 부분만 수정하며, 이는 마코프 시간적 마스크를 사용하여 밝힙니다.

- **Performance Highlights**: 중요한 부분에 대한 식별, 각각의 부분에 대해 예측된 수정 요인, 그리고 WSOLA를 사용하여 리듬을 수정하는 방법으로 새로운 음성 의도를 재구성합니다. 실험을 통해 제안된 방법이 기존의 감정 변환 모델들과 비슷한 수준의 성능을 보여줍니다.



### Safe Semi-Supervised Contrastive Learning Using In-Distribution Data as Positive Examples (https://arxiv.org/abs/2408.01872)
- **What's New**: 이번 연구에서는 클래스 분포 불일치(class distribution mismatch) 상황에서도 안전한 준지도 학습(Safe Semi-Supervised Learning)을 위해 모든 비라벨(unlabeled) 데이터를 효과적으로 활용하는 방법을 제안하였습니다. 이는 셀프-슈퍼바이즈드 콘트라스티브 러닝(self-supervised contrastive learning, SSCL)의 개념을 도입하여 기존 모델의 클래스 분포 불일치 문제를 해결하고자 합니다.

- **Technical Details**: 연구팀은 SSCL 방법을 기반으로 인스턴스 구별(instance discrimination)을 통해 초기 네트워크 파라미터를 설정하고, 비라벨 OOD 데이터를 필터링하지 않으면서도 일반적인 데이터 표현을 학습할 수 있도록 합니다. 또한, 동일 클래스의 라벨된 네거티브 예시를 추가적인 포지티브 예시로 재지정하는 손실 함수 및 이를 위한 손실 계수 스케줄을 도입하여 보다 적절한 표현을 형성합니다.

- **Performance Highlights**: 제안된 방법의 성능을 평가하기 위해 CIFAR-10, CIFAR-100, Tiny ImageNet, 그리고 CIFAR-100과 Tiny ImageNet 혼합 데이터셋에서 다양한 불일치 비율 하의 실험을 수행하였습니다. 실험 결과, SSCL이 이미지 분류 정확도를 크게 개선하며, 인-디스트리뷰션(in-distribution) 예시를 모아 더 나은 표현을 형성함으로써 분류 정확도를 추가로 향상시켰습니다.



### MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilanc (https://arxiv.org/abs/2408.01869)
Comments:
          Paper published at Machine Learning for Healthcare 2024 (MLHC'24)

- **What's New**: 이번 논문은 대형 언어 모델(LLM)을 이용한 새로운 약물 감시(Pharmacovigilance, PhV) 방법론을 소개합니다. MALADE라는 이름의 새로운 다중 에이전트 시스템을 통해 약물 라벨 데이터에서 부작용 사건(ADE)을 추출하는 방식을 제안합니다. 이 시스템은 Retrieval Augmented Generation(RAG) 기법을 사용하여 쿼리를 보강하고, 그 쿼리에 기반하여 응답을 생성합니다.

- **Technical Details**: MALADE는 일반적인 LLM-비종속적 아키텍처로, (1) 의학 문헌, 약물 라벨, FDA 도구(OpenFDA drug information API) 등 다양한 외부 소스를 활용하고, (2) 약물-결과 연관성을 구조화된 형식으로 추출하여 연관성의 강도를 제시하며, (3) 확인된 연관성에 대한 설명을 제공하는 독특한 기능을 갖추고 있습니다. 시스템은 Langroid 다중 에이전트 LLM 프레임워크를 활용하며, GPT-4 Turbo 또는 GPT-4o와 FDA 약물 라벨 데이터를 사용하여 구현되었습니다.

- **Performance Highlights**: 실험 결과, OMOP Ground Truth 테이블과 비교하여 약물 라벨 데이터의 ADE 추출에서 ROC 곡선 아래 면적(AUC) 0.90의 성능을 보여주었습니다. 이는 현재까지의 최신 방법 중 가장 높은 성능을 기록한 것입니다. 이 시스템은 단순히 이진 레이블을 제공하는 것이 아니라, 연관성의 강도 및 부작용의 희귀도 등을 포함한 구조화된 점수를 제시합니다.



### ST-SACLF: Style Transfer Informed Self-Attention Classifier for Bias-Aware Painting Classification (https://arxiv.org/abs/2408.01827)
- **What's New**: 새로운 회화 분류 모델을 소개합니다. Style Transfer와 Adaptive Instance Normalization (AdaIN) 기법을 사용하여 다양한 스타일 간 격차를 해소하고, 특징 맵 적응 공간 주의 모듈(feature-map adaptive spatial attention modules)로 예술적 디테일 이해를 향상시킵니다. 또한, 불균형한 클래스 대표성을 동적으로 조정하는 방법을 사용하여 데이터 편향 문제를 해결합니다.

- **Technical Details**: 이 모델은 ResNet-50 백본을 사용하여 40개의 에폭 동안 87.24%의 정확도를 달성합니다. 우리는 두 단계로 모델을 최적화합니다. 첫 번째 단계는 하이퍼파라미터 그리드 검색과 베이지안 검색을 통한 초기 탐색이며, 두 번째 단계는 모델 파라미터 세트를 점진적으로 조정하는 것입니다. 또한, 질적 및 양적 실험을 통해 다양한 증강 비율의 영향을 평가합니다.

- **Performance Highlights**: 우리의 시스템은 87.24%의 정확도를 달성하며, 이는 LOOK 모델의 89.04% 정확도에 필적하면서도 더 적은 파라미터 요구 사항과 짧은 훈련 시간으로 실용적인 효율성을 향상시킵니다. 우리 모델은 Hyperparameter 탐색과 fine-tuning 전략을 통해 성능을 크게 개선합니다.



### ALIF: Low-Cost Adversarial Audio Attacks on Black-Box Speech Platforms using Linguistic Features (https://arxiv.org/abs/2408.01808)
Comments:
          Published in the 2024 IEEE Symposium on Security and Privacy (SP)

- **What's New**: 최근 연구는 음성 제어 스마트 장치에 대한 적대적 예제(adversarial examples, AE)가 상당한 위협이 될 수 있음을 보여주었습니다. 기존 연구에서는 자동 음성 인식(ASR) 시스템의 최종 전사(transcription)를 이용하는 블랙박스 공격을 제안했지만, 이는 많은 쿼리를 필요로 하여 비용이 많이 듭니다. 이 논문에서는 이러한 한계의 원인을 AE 공격 샘플을 딥러닝 모델의 결정 경계에서 직접적으로 구성할 수 없기 때문임을 알아냈습니다. 이를 기반으로, 우리는 ALIF(Adversarial Linguistic Feature-based attack pipeline)를 제안합니다. ALIF를 이용하여 상용 ASR 시스템과 음성 비서에서 디지털 및 물리적 재생 환경 모두에서 공격을 실행할 수 있습니다.

- **Technical Details**: ALIF는 텍스트-음성 변환(TTS)과 ASR 모델의 상호 프로세스를 활용하여 결정 경계가 위치한 언어 임베딩 공간에서 교란을 생성합니다. ALIF 기반으로, 우리는 디지털 도메인에서 공격을 수행하는 ALIF-OTL과 물리적 재생 환경에서 공격을 수행하는 ALIF-OTA의 두 가지 공격 스키마를 제시합니다.

- **Performance Highlights**: 포괄적인 평가에서는 ALIF-OTL 및 ALIF-OTA가 각각 97.7% 및 73.3%의 쿼리 효율성을 개선한 것으로 나타났습니다. 특히 ALIF-OTL은 단 한 번의 쿼리로도 공격 샘플을 생성할 수 있습니다. 또한, 우리의 테스트 결과는 ASR 업데이트에도 강력한 견고성을 보였습니다. 실험 결과 ALIF 공격은 상용 ASR 및 음성 비서 제품에 대해 높은 성공률과 견고성을 가진 것으로 확인되었습니다. ALIF-OTL은 평균 35개의 쿼리로 디지털 도메인에서 공격 샘플을 생성하며, 평균 성공률 95.8%를 달성했습니다. ALIF-OTA는 최신 연구 대비 73.3%의 쿼리 효율성을 개선하며, 81.3%의 성공률을 보였습니다.



### Towards an ontology of state actors in cyberspac (https://arxiv.org/abs/2408.01787)
- **What's New**: 이 논문은 사이버 보안에서 사이버 위협 분석을 개선하기 위한 계획을 제시하고 있습니다. 저자는 사이버 공간에서 국가 행위자(state actors)와 사이버 운영(cyber operations)을 형식적으로 표현하기 위한 온톨로지(ontologies)를 구축하는 중요성에 대해 논의합니다.

- **Technical Details**: 온톨로지(ontologies)를 사용하면 다양한 출처에서 나오는 데이터를 일관되게 통합하고, 이러한 데이터에 대한 자동 추론(automated reasoning), 정보를 추출 및 재사용하는 지능형 처리(intelligence extraction)를 할 수 있습니다. 저자는 현재의 온톨로지 도구들을 법률, 규제, 정부 기관, 문서와 같은 인접 도메인에 연결함으로써 개선할 수 있다고 주장합니다. 또한, 사이버 보안 도메인에서 형식적 표현을 만드는 기존 온톨로지 도구들을 평가하기 위한 측정 지표(metrics)를 제안하고, 부족한 부분을 개발 및 확장하는 계획을 제시합니다.



### STDA: Spatio-Temporal Dual-Encoder Network Incorporating Driver Attention to Predict Driver Behaviors Under Safety-Critical Scenarios (https://arxiv.org/abs/2408.01774)
- **What's New**: 자율 주행 차량의 행동 예측 정확성을 향상시키기 위해 안전-중요(safety-critical) 시나리오에 최적화된 STDA(Spatio-Temporal Dual-Encoder) 네트워크가 개발되었습니다. 이 네트워크는 운전자의 주의를 반영하여 중요한 위치를 빠르게 식별하도록 설계되었습니다.

- **Technical Details**: STDA는 네 개의 주요 모듈로 구성됩니다: 1) 운전자 주의 예측 모듈, 2) 운전자 주의와 원시 이미지 간의 특징을 융합하는 융합 모듈, 3) 동적 장면 해석 능력을 강화하기 위한 임시 인코더 모듈, 4) 행동 예측 모듈. 이 네트워크를 통해 동적 장면에서의 해석 능력과 행동 예측 정확성을 높였습니다.

- **Performance Highlights**: 실험 결과, 운전자 주의를 통합하고 임시 인코더 모듈을 채택한 STDA는 G-mean을 0.659에서 0.719로 향상시켰습니다. 또한, 제안된 모듈은 강력한 일반화 능력을 보여주었으며 다른 주류 모델들과도 원활히 통합될 수 있음을 입증했습니다.



### Advancing Green AI: Efficient and Accurate Lightweight CNNs for Rice Leaf Disease Identification (https://arxiv.org/abs/2408.01752)
- **What's New**: 이번 연구에서는 쌀 잎 질병을 분류하기 위해 실제 모바일 환경에서 활용할 수 있는 ShuffleNet, MobileNetV2, EfficientNet-B0의 세 가지 CNN 아키텍처를 탐구했습니다. 이 모델들은 비교적 적은 연산 자원과 메모리를 요구하기 때문에 모바일 기기와의 호환성이 높습니다.

- **Technical Details**: 모델의 성능을 향상시키기 위해 두 개의 완전 연결층(fully connected layers)과 드롭아웃 레이어(dropout layer)를 추가하였으며, 과적합(overfitting)을 방지하기 위해 early stop 기법을 사용했습니다. 연구 결과, EfficientNet-B0 모델이 가장 높은 99.8%의 정확도를 달성했습니다. 반면, MobileNetV2와 ShuffleNet은 각각 84.21%와 66.51%의 정확도를 기록했습니다.

- **Performance Highlights**: EfficientNet-B0 모델이 제안된 레이어 구성과 early stop 기법을 결합할 경우 높은 정확성을 보이는 것으로 나타났습니다. EfficientNet-B0의 99.8%라는 놀라운 성능은 쌀 잎 질병 분류에서의 높은 가능성을 시사합니다.



### LAM3D: Leveraging Attention for Monocular 3D Object Detection (https://arxiv.org/abs/2408.01739)
Comments:
          6 pages. Accepted to MMSP 2024

- **What's New**: 새로운 연구 논문 'LAM3D'는 자가-어텐션 메커니즘(self-attention mechanism)을 활용한 단안 3D 물체 검출(Monocular 3D Object Detection) 프레임워크를 제안합니다. 이 방법은 Pyramid Vision Transformer v2 (PVTv2)를 특징 추출 백본(feature extraction backbone)과 2D/3D 검출 기계로 사용하여 개발되었습니다. KITTI 3D Object Detection Benchmark에서 평가한 결과, 자율 주행 분야에서 기존 방법을 능가하는 결과를 보였습니다.

- **Technical Details**: LAM3D는 비전 트랜스포머(Vision Transformer) 기반의 아키텍처를 사용하여 단안 3D 물체 검출 문제를 해결합니다. 기존의 컨볼루션 신경망(Convolutional Neural Networks, CNNs)이 제한된 수용 영역(receptive field) 때문에 장거리 종속성(long-range dependencies)과 맥락적 정보를 포착하는 데 어려움을 겪는 문제를 해결하기 위해 트랜스포머 아키텍처를 채택하였습니다. PVTv2 백본은 다중 스케일 특징 맵(multi-scale feature maps)을 생성하며, 자가-어텐션 메커니즘을 통해 보다 포괄적인 장면 이해를 제공합니다.

- **Performance Highlights**: LAM3D는 KITTI 3D Object Detection Benchmark에서 기존 기술보다 뛰어난 성능을 입증하였습니다. 특히 자가-어텐션 메커니즘을 사용함으로써, 동일한 아키텍처를 사용하지만 자가-어텐션을 사용하지 않는 경우보다 높은 정확도와 강건성을 보여주었습니다.



### Can LLMs predict the convergence of Stochastic Gradient Descent? (https://arxiv.org/abs/2408.01736)
Comments:
          9 pages. Accepted to 1st ICML Workshop on In-Context Learning at ICML 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)이 마코프 경로를 만족하는 동적 시스템의 이해 능력을 시사하는 새로운 발견을 바탕으로, 확률적 경사 하강법(SGD)의 동역학을 더 깊이 탐구합니다. 이를 통해 LLM이 새로운 시작점에 대해 SGD가 수렴하는 국소 최소값을 zero-shot 방식으로 예측하는 능력을 보여줍니다. 더 일반적인 수준에서는, LLM을 활용하여 더 큰 딥러닝 모델에서 zero-shot 무작위 시험(Randomized Trials)을 수행할 가능성을 탐구합니다.

- **Technical Details**: 연구에서는 LLM이 마코프 체인으로서의 SGD의 이론적 연결성을 활용하여 새로운 시작점에서도 국소 최소값을 예측할 수 있는 능력을 분석합니다. 구체적으로는, 타임 시리즈 데이터를 고정된 precision의 문자열(string)로 변환 후, 해당 문자열을 LLM의 토크나이저(tokenizer)를 통해 토큰화합니다. 그런 다음 LLM을 호출하여 전체 토큰 어휘(vocabulary)에 대한 logits를 생성하고, 소프트맥스(Softmax)를 사용해 해당 자릿수에 대한 확률 분포를 추출합니다. 최종적으로는 Hierarchy-PDF 알고리즘을 사용하여 다음 값을 예측하고 이를 통해 전이 규칙을 확립합니다.

- **Performance Highlights**: LLM은 이전에 보지 못한 시작점에서도 SGD가 수렴하는 국소 최소점을 zero-shot 방식으로 예측하는 데 뛰어난 성능을 보였습니다. 이러한 성능은 딥러닝 모델에서의 무작위 시험(Randomized Trials)을 빠르고 효과적으로 수행할 가능성을 제시합니다.



### Landmark-guided Diffusion Model for High-fidelity and Temporally Coherent Talking Head Generation (https://arxiv.org/abs/2408.01732)
- **What's New**: 이번 아카이브(arxiv) 논문은 음성 기반 입 모양 맞춤형 얼굴 생성 기술을 발전시키기 위해 두 단계로 이루어진 확산 모형(diffusion-based model)을 소개합니다. 기존의 GAN 기반 모델이 입 모양의 동기화를 강조하면서도 프레임의 시각적 품질을 간과하고, 확산 기반 모델이 고품질 프레임을 생성하지만 입 모양 일치를 놓쳐 불안정한 입 모양 움직임을 초래한다는 문제를 해결하기 위해 모델을 제안했습니다.

- **Technical Details**: 새로운 두 단계 확산 기반 모델은 다음과 같습니다. 최초 단계에서는 음성을 기반으로 동기화된 얼굴 핵심 점(facial landmarks)을 생성합니다. 두 번째 단계에서는 생성된 얼굴 핵심 점을 노이즈 제거 과정에서 조건으로 사용하여 입 모양 흔들림 문제를 최적화하고 고품질, 잘 동기화된, 시간적으로 일관된 얼굴 영상을 생성합니다. 음성 구간과 원래의 얼굴 핵심 점을 사용하여 순차적으로 얼굴 핵심 점을 생성하는 Landmark Generation Network를 통해 이 과정이 이루어집니다.

- **Performance Highlights**: 광범위한 실험을 통해 이 모델이 최고의 성능을 발휘함을 입증했습니다. 특히, 시각적 품질과 함께 시간적 일관성을 크게 향상시켰습니다.



### Survey on Emotion Recognition through Posture Detection and the possibility of its application in Virtual Reality (https://arxiv.org/abs/2408.01728)
- **What's New**: 본 논문에서는 감정 인식(Emotional recognition)에 포즈 추정(pose estimation) 기법을 사용하여 다양한 기술과의 통합 가능성을 탐구하였습니다. 이는 평범한 카메라, 깊이 카메라(depth cameras) 등을 이용하거나, 가상 현실(VR)과 같은 새로운 형태의 입력(이미지, 비디오, 3차원 벡터 공간에서의 포즈 포함)을 포괄합니다.

- **Technical Details**: 19개의 연구 논문을 선정된 저널과 데이터베이스에서 수집하여 이들의 방법론(methodology), 분류 알고리즘(classification algorithm), 사용된 데이터셋(datasets)을 중심으로 감정 인식 및 포즈 추정 관련 내용을 분석하였습니다. 이 논문들은 다양한 입력 방법을 사용하여 실시간(real-time)으로 감정을 인식하려고 시도했습니다.

- **Performance Highlights**: 감정 인식의 정확도(accuracy)를 기준으로 벤치마킹을 수행하였으며, 멀티모달 접근법(multimodal approaches)이 전반적으로 가장 높은 정확도를 보였습니다. 또한, 본 연구 분야의 발전을 위해 미래에 고려해야 할 사항들도 논의하였습니다.



### Joint Universal Adversarial Perturbations with Interpretations (https://arxiv.org/abs/2408.01715)
- **What's New**: 본 연구에서는 DNN 모델과 해석기를 동시에 공격할 수 있는 새로운 형태의 보편적 adversarial perturbation(UAP)을 제안하는 공격 프레임워크인 JUAP를 소개합니다. 이는 기존 연구들이 다루지 않았던 새로운 문제를 제기하며, DNN의 예측과 해석에 동시에 악의적인 영향을 미칠 수 있는 가능성을 탐구합니다.

- **Technical Details**: JUAP는 생성적 적대 신경망(Generative Adversarial Network, GAN)을 사용하여 여러 데이터셋에서 보편적 perturbations을 학습합니다. 이 학습된 perturbations는 이미지에 추가됨으로써 DNN의 예측을 혼란스럽게 하며, 동시에 해석기의 결과를 오도합니다. 이는 주어진 이미지를 기반으로 grad-CAM과 같은 interpreters를 혼란스럽게 만드는 반면, DNN 모델의 출력을 왜곡시킵니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에서 종합적인 실험을 통해 제안된 JUAP 공격 방법의 효율성을 입증하였습니다. 제안된 방법은 높은 기만 비율을 기록하였으며, 이는 DNN 분류기와 그에 연계된 해석기의 공동 공격에서 성공적인 결과를 가져왔습니다.



### Downstream Transfer Attack: Adversarial Attacks on Downstream Models with Pre-trained Vision Transformers (https://arxiv.org/abs/2408.01705)
- **What's New**: 최근 Vision Transformers(ViTs)와 자가지도학습(Self-Supervised Learning, SSL) 기법의 발전으로, 사전 학습된 대형 ViTs는 컴퓨터 비전 응용 프로그램의 새로운 기반 모델이 되고 있습니다. 본 논문에서는 이러한 대형 ViTs의 적대적 취약성(adversarial vulnerability)이 하위 작업으로 전이되는지에 대해 연구하였습니다. 우리는 샘플 단위의 전이 공격(sample-wise transfer attacks)에 주목하여 새로운 공격 방법인 Downstream Transfer Attack (DTA)을 제안합니다. DTA는 사전 학습된 ViT 모델을 활용하여 적대적 예제를 생성하고, 이를 하위 데이터셋으로 미세 조정된 모델에 적용하여 공격합니다.

- **Technical Details**: DTA는 주어진 테스트 이미지에 대해 사전 학습된 ViT 모델을 사용해 적대적 예제를 생성하고 이를 사용해 하위 데이터셋에서 미세 조정된 모델을 공격합니다. 공격 중에 DTA는 사전에 학습된 모델의 취약한 층을 코사인 유사도 손실(cosine similarity loss)을 통해 식별하고 활용합니다. 이는 전이 가능한 공격을 만들기 위해 매우 중요한 과정을 포함합니다. DTA는 처음에는 모델의 얕은 층을 목표로 하며, 초기 시도가 실패하면 중간 층을 탐색하여 가장 취약한 층을 찾아내어 최종 공격을 수행합니다.

- **Performance Highlights**: 사전 학습된 ViTs에 대해 3가지 다른 사전 학습 방법과 3가지 미세 조정 방식, 그리고 10가지 다양한 하위 데이터셋에서 광범위한 실험을 수행한 결과, DTA는 평균 공격 성공률(ASR)이 90%를 초과하며 기존 방법들을 크게 초과하는 성능을 보여주었습니다. 또한 DTA를 사용한 적대적 훈련이 모델의 다양한 하위 전이 공격에 대한 강건성을 크게 향상시킬 수 있음을 확인하였습니다.



### Invariant Graph Learning Meets Information Bottleneck for Out-of-Distribution Generalization (https://arxiv.org/abs/2408.01697)
- **What's New**: 본 논문에서는 그래프 데이터의 분포 이동 문제를 해결하기 위해 '정보 병목 이론(Information bottleneck theory)'을 기반으로 한 새로운 프레임워크인 InfoIGL을 제안합니다. InfoIGL은 그래프의 불변 특징을 추출하여 모델의 일반화 능력을 향상시키는 것을 목표로 하며, 감독 신호가 없어도 강력한 일반화 능력을 보여줍니다.

- **Technical Details**: InfoIGL은 과도한 정보를 압축하는 '중복 필터(redundancy filter)'를 도입하여 환경적 요인과 관련된 불필요한 정보를 제거합니다. 그런 다음, 설계된 '다중 레벨 대조 학습(multi-level contrastive learning)'을 통해 동일 클래스의 그래프 간 상호 정보를 극대화하여 예측에 필요한 불변 특징을 보존합니다. 이는 주로 인스턴스 레벨 및 의미 레벨에서의 대조 학습을 포함하며, 불변성에 대한 감독 신호에 의존하지 않습니다.

- **Performance Highlights**: 합성 및 실세계 데이터셋에 대한 실험 결과, InfoIGL은 그래프 분류 작업에서 분포 이동 상황을 해결하는 데 있어서 최첨단 성능을 달성했습니다. 이 접근 방식은 그래프 OOD 일반화 문제를 해결하는 데 있어서 매우 효과적임이 입증되었습니다.



### Generating High-quality Symbolic Music Using Fine-grained Discriminators (https://arxiv.org/abs/2408.01696)
Comments:
          Accepted by ICPR2024

- **What's New**: 이번 연구에서는 기존의 상징적 음악 생성 방식에서 글로벌한 음악 인식만을 고려한 단일 감별기의 한계를 극복하기 위해, 멜로디와 리듬을 분리하여 해당 요소에 맞는 더 세밀한 감별기를 설계했습니다. 이러한 설계는 인간 작곡가의 음악을 더 쉽게 모방할 수 있도록 도와줍니다.

- **Technical Details**: 우리의 모델은 자동 회귀(Symbolic Music Generation)의 생성기와 두 개의 세밀한 감별기로 구성되어 있습니다. 멜로디 감별기는 음고증강 전략(pitch augmentation)을, 리듬 감별기는 바 수준의 상대적 위치 인코딩(bar-level relative positional encoding)을 사용하여 각각 멜로디와 리듬의 특성을 잘 반영하고자 합니다. 생성기는 조건 음악 시퀀스를 입력으로 받아 전체 음악 시퀀스를 생성합니다.

- **Performance Highlights**: POP909 벤치마크 실험 결과, 이 새로운 방법이 여러 최신(state-of-the-art) 기법들과 비교하여 객관적, 주관적 평가 모두에서 우수한 성능을 보임을 확인했습니다.



### TreeCSS: An Efficient Framework for Vertical Federated Learning (https://arxiv.org/abs/2408.01691)
Comments:
          16 pages, 7 figures

- **What's New**: 새로운 논문인 TreeCSS 프레임워크는 수직적 연합 학습(vertical federated learning, VFL)에서 데이터 정렬과 모델 훈련을 가속화하는 효율적인 방법을 제안합니다. Tree-MPSI라는 새로운 다자간 비밀 집합 교차(MPSI) 프로토콜을 도입하여 참가자 간 데이터 정렬을 병렬화하고, 군집 기반 샘플 선택(clustering-based coreset selection, CSS)을 통해 대표 데이터 샘플을 선택합니다. 이 논문은 다양한 데이터셋과 모델을 사용하여 TreeCSS의 효과와 효율성을 평가하였으며, 결과적으로 기존 VFL보다 훈련 속도를 최대 2.93배까지 가속화하면서 유사한 모델 정확도를 달성했습니다.

- **Technical Details**: TreeCSS 프레임워크는 다음 세 가지 기술 과제를 해결합니다: (1) 수많은 참가자 간 데이터 샘플을 효율적으로 정렬하는 방법, (2) 대표 샘플을 효율적으로 선택하는 방법, (3) 모델 학습에 대한 데이터 샘플의 기여도를 반영하는 방법. 첫째, Tree-MPSI는 트리 구조를 사용하여 참가자들을 쌍(pair)으로 묶어 2인용 PSI 연산을 수행하고 각 참가자의 데이터 양에 따라 스케줄링을 최적화합니다. 둘째, Cluster-Coreset은 K-Means를 사용하여 각 참가자의 특징(feature)을 군집화한 뒤, 로컬 군집화 결과에 따라 글로벌 유사성을 측정하여 대표 샘플을 선택합니다. 마지막으로, 클러스터 중심까지의 거리(distance)를 고려하여 샘플의 중요도를 재가중합니다.

- **Performance Highlights**: TreeCSS 프레임워크는 여섯 가지 다양한 데이터셋과 분류(classification) 및 회귀(regression) 모델을 실험하여, 기존 VFL 방법보다 큰 차이로 학습 속도를 가속화하면서도 유사한 모델 정확도를 달성함을 확인했습니다. 특히, TreeCSS는 최대 2.93배까지 훈련 속도를 증가시켰습니다.



### IDNet: A Novel Dataset for Identity Document Analysis and Fraud Detection (https://arxiv.org/abs/2408.01690)
Comments:
          40 pages

- **What's New**: 새로운 벤치마크 데이터셋 IDNet이 소개되었습니다. IDNet은 837,060장의 합성 신분증 이미지를 포함하며 약 490 기가바이트의 데이터를 제공합니다. 이 데이터셋은 10개의 미국 주와 10개의 유럽 국가에서 온 20가지 유형의 신분증으로 분류됩니다. 이 데이터셋은 개인정보 보호를 유지하면서 현실적인 사기 탐지 방법을 훈련시키는 데 도움을 주기 위해 설계되었습니다.

- **Technical Details**: IDNet 데이터셋은 Stable Diffusion 2.0 및 ChatGPT-3.5-turbo와 같은 최신 AI 기술을 사용하여 생성되었습니다. 각각의 신분증 샘플은 6가지 다른 사기 패턴으로 변조된 버전을 포함하고 있으며, 이는 Crop and Move, Impaint and Rewrite, 얼굴 변형(face morphing), 초상화 교체, 텍스트 필드의 직접 변경 등입니다. 이 파이프라인은 인텔 Xeon Gold 6226 24코어 CPU 프로세서 두 개, Nvidia GeForce 2080 Ti GPU 네 개, 196GB 메모리를 갖춘 서버에서 동작하며, 문서 하나를 생성하는 데 약 0.14초가 소요됩니다.

- **Performance Highlights**: IDNet은 다른 공개 데이터셋과 달리 더 많은 샘플과 다양한 사기 패턴을 포함하고 있습니다. 각 문서는 5,979개의 고유한 합성 초상화를 포함하고 있으며, 전체 데이터셋은 41,853개의 문서 샘플을 포함하고 있습니다. 이러한 넓은 범위의 데이터는 프라이버시를 고려한 사기 탐지 연구에 있어 중요한 발전입니다. 또한, 데이터셋은 실제 문서와 유사한 정확도와 다양한 메타데이터를 갖추고 있어 연구에 유용합니다.



### Controllable Unlearning for Image-to-Image Generative Models via $\varepsilon$-Constrained Optimization (https://arxiv.org/abs/2408.01689)
Comments:
          40 pages, 54 figures

- **What's New**: 최근 생성 모델(generative models)의 놀라운 발전과 함께, 프라이버시 침해와 편향 문제와 같은 우려가 제기되고 있습니다. 이러한 문제를 해결하기 위해 머신 언러닝(machine unlearning)이 등장했으며, 본 논문에서는 이미지-이미지(I2I) 생성 모델에서 이를 연구하고 있습니다. 기존의 연구가 단일 목적 최적화 문제로 다루어졌다면, 본 논문에서는 사용자 기대에 따른 다양한 트레이드오프를 고려하는 컨트롤 가능한 언러닝 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 $	ext{control coefficient}$ $	ext{ε}$를 사용하여 트레이드오프를 제어합니다. 이는 $	ext{ε-constrained optimization problem}$으로 재구성되며, 경계 최적해(unlearning boundaries)를 찾기 위해 gradient-based 방법을 사용합니다. 이 경계 내에서는 모든 솔루션이 파레토 최적(Pareto optimality)을 보장합니다. 아울러, 다양한 컨트롤 함수에 따른 프레임워크의 수렴 속도를 분석하였습니다.

- **Performance Highlights**: 세 가지 주류 I2I 모델에 걸쳐 두 개의 벤치마크 데이터셋으로 수행한 광범위한 실험 결과, 제안된 컨트롤 가능한 언러닝 프레임워크가 그 효과성을 입증하였습니다.



### radarODE: An ODE-Embedded Deep Learning Model for Contactless ECG Reconstruction from Millimeter-Wave Radar (https://arxiv.org/abs/2408.01672)
- **What's New**: 이 논문은 밀리미터파 레이더(mm-wave radar) 신호로부터 상세한 심전도(ECG) 신호를 재구성하는 새로운 심박 측정 방법을 제안합니다. 기존의 데이터 중심 접근 방식에서 벗어나, 본 연구는 전기 도메인(Electrical Domain)과 기계 도메인(Mechanical Domain) 간의 변환을 모델링하여 보다 정밀한 ECG를 재구성할 수 있습니다. 새롭게 제안된 레이더 신호 모델을 기반으로, 레이더ODE라는 새로운 딥러닝 프레임워크가 설계되었습니다.

- **Technical Details**: 레이터 신호를 전기적 심장 활동으로 변환하기 위해 범용 미분 방정식(Ordinary Differential Equations, ODE)을 디코더로 사용하는 레이더ODE 프레임워크를 개발했습니다. ODE는 형태적 사전 지식을 제공하여 모델 학습의 수렴 속도를 높이고 신체 움직임이 있을 때도 강인성을 유지하게 합니다. 레이더ODE는 신호의 형태적(morphological) 및 시간적(temporal) 특성을 결합하여 ECG를 생성합니다.

- **Performance Highlights**: 데이터셋을 통해 검증한 결과, 레이더ODE는 일반적인 벤치마크 대비 미검출률(missed detection rate), 평균 제곱근 오차(root mean square error), 피어슨 상관 계수(Pearson correlation coefficient) 측면에서 각각 9%, 16%, 19%의 성능 향상을 보여주었습니다. 이 결과는 레이더ODE가 높은 정확도로 레이더 신호로부터 ECG 신호를 복구할 수 있으며 실생활에서도 구현 가능함을 시사합니다.



### SAT3D: Image-driven Semantic Attribute Transfer in 3D (https://arxiv.org/abs/2408.01664)
- **What's New**: 이번 연구에서는 참조 이미지(reference image)를 활용해 3D에서의 의미론적 속성 전이(Semantic Attribute Transfer)을 수행하는 SAT3D 방법을 제안했습니다. 이는 기존 2D 및 3D 이미지 편집 방법들이 애매한 속성 편집에 머무르는 문제를 해결합니다.

- **Technical Details**: SAT3D는 사전에 훈련된 3D-aware StyleGAN 기반 생성기의 스타일 공간(style space)을 탐색하고, 의미론적 속성과 스타일 코드 채널의 상관관계를 학습합니다. 이를 위해 각 속성을 단어 기반(descriptor groups)의 집합으로 연결하고, CLIP 모델을 활용해 속성을 정량적으로 측정하는 Module(QMM)을 개발했습니다. 그 후, QMM을 속성 손실(attribute losses)에 포함시켜 이미지 사이의 속성 유사도를 계산하고, 대상 속성 전이와 다른 속성의 보존을 지도합니다.

- **Performance Highlights**: SAT3D는 여러 도메인에서의 3D-aware 속성 전이 결과를 제시했으며, 기존의 2D 이미지 편집 방법과 비교하여 효과성과 사용자 맞춤성을 입증했습니다. 특히, 3D 뿐만 아니라 2D StyleGAN 기반 생성기에서도 효과적으로 적용 가능합니다.



### Stimulating Imagination: Towards General-purpose Object Rearrangemen (https://arxiv.org/abs/2408.01655)
Comments:
          9 pages

- **What's New**: SPORT는 지능형 일반 로봇을 위한 범용 객체 재배치 프레임워크입니다. 이 프레임워크는 물체의 위치 지정, 목표 위치 상상 및 로봇 제어의 세 가지 부분으로 나뉘며, 사전 학습된 대형 비전 모델을 활용하여 광범위한 의미론적 추론을 수행하고 확산 기반 3D 자세 추정기를 학습하여 물리적으로 현실적인 결과를 보장합니다. 이는 새로운 환경에서도 다양한 물체를 인간의 지시에 따라 재배치할 수 있게 합니다.

- **Technical Details**: SPORT는 물체 유형(이동 대상 또는 참조)만을 전달하여 개방형 객체 인식 및 위치 지정의 강력한 능력을 최대한 활용합니다. 확산 기반 모델은 물체의 구체적인 의미 정보를 요구하지 않고 목표 자세를 '상상'할 수 있어 훈련 부담을 크게 줄입니다. 목표 자세 추정을 위한 훈련 데이터는 시뮬레이션에서 수집되고 GPT-4에 의해 주석이 달립니다.

- **Performance Highlights**: 시뮬레이션 및 실제 실험에서 SPORT의 잠재력을 입증했습니다. 다양한 물체를 사용한 재배치를 정확하게 수행할 수 있으며, 이는 새로운 환경에서도 효과적입니다. 3D 객체 재배치 데이터 부족 문제를 해결하기 위해 GPT 보조 파이프라인을 구축하여 물리적 현실성을 보장하는 고품질 데이터를 생성합니다.



### Music2P: A Multi-Modal AI-Driven Tool for Simplifying Album Cover Design (https://arxiv.org/abs/2408.01651)
Comments:
          Accepted at CIKM 2024 Demo Paper track. Project available at this https URL

- **What's New**: 오늘날 음악 산업에서 앨범 커버 디자인은 음악 자체만큼이나 중요합니다. 그러나 많은 AI 기반 앨범 커버 서비스는 구독이나 기술적 전문 지식을 필요로 하여 접근성을 제한합니다. 이를 해결하기 위해 우리는 Music2P라는 오픈 소스, 멀티 모달 AI 도구를 개발했습니다. 이 도구는 Ngrok를 통해 앨범 커버 생성 과정을 효율적이고 접근 가능하며 비용 효율적으로 자동화합니다.

- **Technical Details**: Music2P는 여러 최신 AI 기술을 통합하여 앨범 커버를 생성합니다. 이미지 캡션 생성을 위한 BLIP (Bootstrapping Language Image Pre-training), 음악을 텍스트로 변환하기 위한 LP-music-caps, 이미지 세분화를 위한 LoRA, 앨범 커버 및 QR 코드 생성을 위한 ControlNet 등의 기술을 사용합니다. Music2P 시스템은 이들 구성 요소를 통합하여 다중 모달 입력에서 시각적으로 매력적이고 의미 있는 앨범 커버를 생성합니다.

- **Performance Highlights**: Music2P는 뮤지션과 프로듀서가 제한된 자원이나 전문 지식으로도 설득력 있는 앨범 커버를 만들 수 있도록 돕습니다. 사용자 친화적인 인터페이스를 제공하여 음악 파일, 이미지, 설명을 업로드한 후 간단히 앨범 커버를 생성할 수 있습니다. QR 코드 생성 기능도 탑재되어, 독립적인 뮤지션과 작은 음악 레이블이 손쉽게 마케팅 전략을 강화할 수 있습니다.



### Self-Emotion Blended Dialogue Generation in Social Simulation Agents (https://arxiv.org/abs/2408.01633)
Comments:
          Accepted in SIGDIAL 2024

- **What's New**: 이 논문은 가상 시뮬레이션 환경에서 대화 에이전트(Dialogue Agents)가 맥락과 상관없는 자기 감정(Self-Emotion)을 표현할 때의 대화 전략 및 의사 결정에 어떤 영향을 미치는지 탐구합니다. 연구 결과에 따르면, 자기 감정을 포함하면 에이전트가 더 인간적인 대화 전략을 구현하며, 전체적인 자연스러움 및 인간미가 향상됩니다.

- **Technical Details**: 가상 환경에서 에이전트는 일련의 이벤트를 경험하게 설정됩니다. 이 에이전트들이 시뮬레이션된 시간 동안 자기 감정 상태를 추적하며, 대화 시 이러한 자기 감정 상태와 경험된 이벤트가 일치하도록 조정됩니다. 대화 전략은 사전 정의된 11가지 전략 풀에서 선택되며, 자기 감정은 무작위 레이블(Random Label), 무작위 이벤트(Random Event), 프로필 이벤트(Profile Event) 3가지 스타일로 표현됩니다. 실험에는 GPT-4 및 FLAN-T5 모델이 활용되었고, 이들 모델을 기반으로 한 대화 데이터셋을 통해 비교 평가가 이루어졌습니다.

- **Performance Highlights**: 1. 자기 감정을 포함한 에이전트는 더 인간적인 대화 전략을 사용합니다. 2. 자기 감정을 포함한 대화는 인간 평가에서 더 자연스럽고 공감적이며 인간적인 것으로 평가되었습니다. 3. 그룹 토론 실험에서 에이전트의 자기 감정이 의사 결정에 약 50%의 변화를 가져왔습니다.



### Positive-Unlabeled Constraint Learning (PUCL) for Inferring Nonlinear Continuous Constraints Functions from Expert Demonstrations (https://arxiv.org/abs/2408.01622)
- **What's New**: 이 논문은 전문가의 시범에서 연속적인 임의의 제약 조건 함수를 유추하여 로봇의 제약 조건을 학습하는 새로운 Positive-Unlabeled Constraint Learning (PUCL) 알고리즘을 제안합니다. 이는 기존 방법처럼 진정한 제약 조건의 매개 변수화나 환경 모델에 대한 사전 지식 없이 작동합니다.

- **Technical Details**: PUCL 프레임워크에서는 모든 시범 데이터를 긍정(허용 가능) 데이터로 취급하고, 잠재적으로 허용할 수 없는 경로를 생성하는 제어 정책을 학습하여 라벨이 없는 데이터로 사용합니다. 각 반복에서 정책을 업데이트한 후, 신뢰할 수 있는 불가한 데이터를 식별하고 이 데이터를 바탕으로 이진 허용 가능성 분류기(즉, 제약 함수)를 학습하는 두 단계 Positive-Unlabeled 학습 절차를 적용합니다.

- **Performance Highlights**: PUCL 방법이 네트워크된 정책(networked policy) 또는 동적 시스템 정책(dynamical system policy)을 사용하여 세 가지 로봇 과제에서 효과를 검증했습니다. 이 방법은 연속적인 비선형 제약 조건을 성공적으로 유추하고 전달하며, 제약 조건의 정확도와 정책 안전성 측면에서 다른 기준 방법들보다 우수한 성과를 보였습니다.



### Advancing Mental Health Pre-Screening: A New Custom GPT for Psychological Distress Assessmen (https://arxiv.org/abs/2408.01614)
- **What's New**: 최근 'Psycho Analyst'라는 맞춤형 GPT 모델이 소개되었습니다. 이 모델은 OpenAI의 GPT-4를 기반으로 하여, 정신 건강 장애의 사전 선별을 위한 최적화가 이루어졌습니다. DSM-5와 PHQ-8, 상세 데이터 설명 및 광범위한 훈련 데이터를 활용해 미세한 언어적 지표를 해석할 수 있습니다. 이를 통해 이 모델은 공공 정신 건강 지원을 향상시키고 접근성 및 비용 효율성을 높이며 전문가를 위한 'second opinion' 역할을 할 수 있습니다.

- **Technical Details**: 'Psycho Analyst'는 이진 분류(binary classification)와 세 단계 PHQ-8 점수 산출을 포함한 이중 작업 프레임워크(dual-task framework)를 사용합니다. 이를 통해 초기 평가, 상세 분석, 독립적 평가를 수행할 수 있으며, 세분화된 분석 기능을 보여줍니다. 이 모델은 DAIC-WOZ 데이터셋을 사용한 검증을 통해 높은 정확성을 입증했습니다. F1-score와 Macro-F1 score는 각각 0.929와 0.949로 측정되었으며, PHQ-8 점수의 MAE와 RMSE는 각각 2.89와 3.69로 가장 낮았습니다.

- **Performance Highlights**: 이 연구는 Psycho Analyst 모델의 뛰어난 성능을 여러 측면에서 확인했습니다. 모델은 실제 임상 환경에서 높은 정확성을 보였으며, 이는 정신 건강 진단에 실제로 적용 가능함을 시사합니다. 또한 zero-shot 및 few-shot 학습 시나리오에서 훌륭한 성능을 보여 최소한의 훈련으로 다양한 데이터에 적응할 수 있습니다. DSM-5와 PHQ-8 기준을 통합하여 초기 개입과 맞춤형 정신 건강 치료의 가능성을 열어주고, 공공 건강 설정에서 조기 탐지와 맞춤형 치료 계획을 개선하는 데 기여할 수 있습니다.



### Trustworthy Machine Learning under Social and Adversarial Data Sources (https://arxiv.org/abs/2408.01596)
Comments:
          PhD thesis

- **What's New**: 최근 몇 년간 기계 학습(Machine Learning) 분야에서는 놀라운 돌파구들이 있었다. 이 논문에서는 기계 학습 시스템들이 사회적 및 적대적인 행동과 어떻게 상호작용하는지에 대한 문제를 다루고 있다. 특히 전략적인 개인, 자기 이익을 추구하는 데이터 수집자, 적대적인 공격자들이 생성하는 데이터에 의해 시스템 성능이 저하될 수 있음을 강조하고 있다. 이에 따라 여러 목표를 만족시키는 예측기, 모델, 정책들을 만드는 것이 필수적이다.

- **Technical Details**: 기계 학습 시스템들과의 상호작용에서 발생하는 데이터는 전략적인 개인들에 의해 생성되거나, 자기 이익을 추구하는 데이터 수집자들에 의해 수집되며, 때로는 적대적인 공격자들에 의해 오염될 수 있다. 예를 들어, 심층 신경망(deep neural networks)은 적대적 예제(adversarial examples)에 취약하다(Shafahi et al., 2018; Szegedy et al., 2013). 또한, 전략적인 개인들이 있는 상황에서는 고전적인 알고리즘의 성능이 저하될 수 있다(Ahmadi et al., 2021).

- **Performance Highlights**: 논문에서는 데이터 생성 및 수집 과정에서 발생할 수 있는 다양한 사회적 및 적대적 행동들이 기계 학습 시스템의 성능에 미치는 영향을 분석하고 있다. 이는 기계 학습 시스템이 사회적 환경에서 성공하기 위해 반드시 해결해야 할 도전 과제임을 시사한다.



### OpenLogParser: Unsupervised Parsing with Open-Source Large Language Models (https://arxiv.org/abs/2408.01585)
- **What's New**: 이번 연구에서는 기존 상용 대형 언어 모델(LLM; Large Language Model) 기반 로그 파서의 한계를 극복하기 위해 OpenLogParser를 제안합니다. OpenLogParser는 오픈 소스 LLM, 특히 Llama3-8B를 활용하여 데이터 프라이버시를 강화하고 운영 비용을 절감하면서도 최첨단 로그 파싱 정확도를 달성합니다.

- **Technical Details**: OpenLogParser는 고정 깊이(Grouping Tree)를 사용해 정적 텍스트가 유사하지만 동적 변수가 다른 로그들을 그룹핑합니다. 이후, 유사도 기반 검색 증강 생성(RAG; Retrieval Augmented Generation) 기법을 사용하여 Jaccard 유사도를 기반으로 다양한 로그를 선택해 정적 텍스트와 동적 변수를 구분합니다. 또, 자기 반영(Self-Reflection) 과정을 통해 LLM의 응답을 개선하고, 로그 템플릿 메모리(Log Template Memory)를 사용하여 파싱 효율성을 극대화합니다.

- **Performance Highlights**: LogHub2.0 데이터셋을 통한 평가에서, OpenLogParser는 기존 최첨단 LLM 기반 파서 대비 25% 높은 파싱 정확도를 기록했습니다. 또한, 로그 처리 속도가 평균 2.7배 빠르다는 점에서 뛰어난 효율성을 보여주었습니다. Self-Reflection 메커니즘을 통해 파싱 정확도가 7% 이상 개선되었으며, 소형 LLM을 사용한 실험에서도 Llama3-8B가 최상의 성능을 발휘했습니다.



### Conformal Diffusion Models for Individual Treatment Effect Estimation and Inferenc (https://arxiv.org/abs/2408.01582)
- **What's New**: 이번 아카이브 논문에서는 관찰 데이터에서 개별 치료 효과(individual treatment effect, ITE)를 추정하고 추론하는 새로운 방법론을 제안합니다. 제안된 접근 방식은 확산 모델(diffusion model)을 기반으로 하며, 계약적 추론(conformal inference), 성향 점수(propensity score), 그리고 공변량(covariate) 지역 근사(local approximation)를 통합하여 다양한 분포 변동(distributional shifts) 문제를 해결합니다.

- **Technical Details**: 제안된 방법론은 확산 모델의 유연성과 계약적 추론의 모델 비자유(inference) 패러다임을 결합하여 잠재적 결과의 분포를 편향 없이 추정하고, 정보를 제공하는 신뢰 구간(confidence interval)을 구축합니다. 이 방법은 ITE에 대한 최초의 확산 모델 기반 계약적 추론 솔루션으로, 기존의 감독된 회귀 설정에서 조건부 무작위 샘플링을 ITE 설정으로 확장하고, 확산 모델과 통합합니다. 성향 점수를 통합하여 처리 및 통제 그룹 간의 잠재적 할당 불균형으로부터 발생하는 공변량 이동을 완화하고, 지역 근사 정보를 활용하여 샘플 효율성을 개선합니다.

- **Performance Highlights**: 광범위한 수치 실험을 통해 제안된 방법이 기존 솔루션보다 경쟁력있는 성능을 보임을 입증했습니다. 제안된 방법론은 공통의 정규 조건하에서 ITE에 대한 신뢰 구간의 커버리지(Coverage)에 대한 엄격한 이론적 보장을 제공합니다.



### Robot-Enabled Machine Learning-Based Diagnosis of Gastric Cancer Polyps Using Partial Surface Tactile Imaging (https://arxiv.org/abs/2408.01554)
- **What's New**: 이번 연구에서는 Vision-based Tactile Sensor (VTS)와 이를 보완하는 머신러닝(ML) 알고리즘을 사용하여 고급 위암(AGC) 종양 진단의 기존 한계를 처음으로 해결하려고 합니다. VTS를 사용하여 자동 데이터 수집이 가능하며, 이를 통해 데이터 부족 문제와 전통적인 ML 접근 방식에서 발생하는 편향을 해결할 수 있습니다. 또한, 새롭게 개발된 로봇 매니퓰레이터와 3D 프린팅된 AGC 종양 모형을 활용하여 고해상도 질감 이미지를 수집하고 ML 모델을 훈련시킵니다.

- **Technical Details**: 본 연구에서 사용된 VTS는 HySenSe라 불리는 최근 개발된 센서로, 실리콘 막, 아두캠(Arducam) 카메라, 아크릴 판, RGB LED 등으로 구성되어 있습니다. 이 센서는 AGC 종양의 표면 텍스처와 경도를 시각적으로 캡처할 수 있으며, 로봇 시스템과 함께 사용하여 종양 표면의 전체 데이터를 자동으로 수집합니다. AGC 종양의 종류를 Borrmann의 분류 체계를 따라 총 4가지 타입으로 구분하였습니다.

- **Performance Highlights**: 제안된 ML 모델은 합성 데이터로 훈련되었으며 혼합된 형태적 특성과 부분적인 센서 접촉 조건에서도 기존의 ML 모델과 비교하여 우수한 성능을 보였습니다. 각종 통계적 지표를 사용하여 성능을 평가하였으며, 새로운 ML 기반 진단 도구는 다양한 AGC 종양의 특성을 민감하게 분류할 수 있음을 보여주었습니다.



### Active Learning for Neural PDE Solvers (https://arxiv.org/abs/2408.01536)
Comments:
          Code will be made available at this https URL

- **What's New**: 이 논문에서는 공학과 과학 분야에서 중요한 문제인 부분 미분 방정식(PDE)의 해결을 위한 새로운 액티브 러닝 (Active Learning, AL) 기준을 제안합니다. 이는 기존의 수치 해법에 비해 더 효율적인 신경망 PDE 솔버(neural PDE solvers)의 데이터 요구량을 줄이는 데 도움을 줄 수 있습니다. 새롭게 제안된 AL4PDE 벤치마크는 다양한 매개변수화된 PDE와 최신 대리 모델(surrogate models)을 제공하여 PDE 해결을 위한 새로운 AL 방법들을 평가하고 개발할 수 있습니다.

- **Technical Details**: AL4PDE는 여러 매개변수화된 PDE와 최신 대리 모델을 포함하며, 불확실성 기반(uncertainty-based) 및 특징 기반(feature-based) 등의 배치 액티브 러닝 알고리즘을 평가합니다. 이 벤치마크는 PDE 매개변수와 초깃값에 대해 일관된 분포를 가진 유사한 데이터셋을 반복적인 실행마다 생성할 수 있습니다. 따라서 데이터 생성에 참여하지 않은 대리 모델에게도 유용한 데이터셋을 제공합니다.

- **Performance Highlights**: 액티브 러닝(AL)을 적용하면 랜덤 샘플링에 비해 평균 오류가 최대 71% 감소하고, 최악의 오류도 현저히 줄어드는 성과를 보였습니다. 또한, AL을 통해 생성된 데이터셋은 반복적인 실행에서도 일관성을 유지하므로 재사용이 가능합니다.



### Contextual Cross-Modal Attention for Audio-Visual Deepfake Detection and Localization (https://arxiv.org/abs/2408.01532)
- **What's New**: 디지털 시대에 들어서면서, 멀티모달(manipulation) 기반의 deepfake와 같은 합성 미디어가 사회 및 정치적 통합성을 위협하고 있습니다. 이번 논문에서는 오디오-비주얼 deepfake 탐지를 위해 문맥적 정보를 활용한 순환 신경망(RNN) 기반의 새로운 멀티모달 어텐션 프레임워크를 제안합니다. 이 접근법은 오디오 및 비디오 신호의 다중 시퀀스 표현에 어텐션을 적용하여 deepfake 탐지 및 로컬라이제이션 성능을 향상시킵니다.

- **Technical Details**: 기존의 멀티모달 deepfake 탐지기는 종종 이질적 데이터 스트림의 어텐션 기반 융합에 의존합니다. 하지만 데이터의 이질적 특성(예: 오디오 및 비주얼 신호)으로 인해 효과적인 융합에 어려움이 있습니다. 본 연구는 이런 문제를 해결하기 위해 재발성 신경망 기반 멀티모달 어텐션 프레임워크를 도입했습니다. 이 접근법은 멀티모달 다중 시퀀스 표현에서 기여하는 특징을 학습하여 deepfake 감지 및 로컬라이제이션을 구현합니다.

- **Performance Highlights**: FakeAVCeleb, AV-Deepfake1M, TVIL, LAV-DF 등 여러 오디오-비주얼 deepfake 데이터셋에 대한 실험적 검증 결과, 제안된 접근법이 기존 연구에 비해 탐지 정확도가 3.47% 향상되었고 정밀도가 2.05% 향상된 것으로 나타났습니다. 이는 현재까지 발표된 가장 높은 성능을 기록한 것입니다.



### Analyzing LLMs' Capabilities to Establish Implicit User Sentiment of Software Desirability (https://arxiv.org/abs/2408.01527)
Comments:
          6 pages, 2 figures, 2 tables

- **What's New**: 이번 연구는 사용자가 표현하는 암묵적인 소프트웨어 바람직성을 정량적으로 분석하기 위해 여러 LLMs (Large Language Models)를 사용했습니다. 이 연구는 기존의 긍정, 중립, 부정으로 분류하는 방법 대신 스케일된 숫자 감정 분석을 제공합니다. 이는 제품 바람직성에 관한 더 나은 결정을 내리기 위해 감정의 강도를 더 깊이 있게 이해할 수 있게 합니다. 데이터는 Microsoft Product Desirability Toolkit (PDT)을 사용하여 수집되었으며, ZORQ라는 학부 컴퓨터 과학 교육에 사용되는 게이미피케이션 시스템의 사용자로부터 데이터를 수집했습니다.

- **Technical Details**: PDT 데이터를 분석하기 위해 여러 LLMs (Claude Sonnet 3, GPT4, GPT4o, 등)을 사용했습니다. 또한, Twitter-Roberta-Base-Sentiment (TRBS)와 Vader도 감정 분석 방법으로 사용되었습니다. 각 시스템은 PDT의 단어/설명 쌍과 사용자가 선택한 5개의 단어와 설명 전체를 보고 감정을 분석하도록 요청되었습니다. LLMs는 감정 점수뿐만 아니라 자신들의 신뢰도 (낮음, 중간, 높음)와 그 이유도 제공했습니다. 모든 LLMs는 사용자의 그룹화된 데이터에서 사용자 감정을 통계적으로 감지할 수 있었지만, TRBS와 Vader는 그렇지 못했습니다.

- **Performance Highlights**: LLMs는 사용자 그룹화된 데이터에서 감정을 통계적으로 감지할 수 있었지만, TRBS와 Vader는 그렇지 못했습니다. 이는 LLMs가 암묵적인 사용자 감정을 이해하는 데 더 효과적임을 시사합니다. 또한, LLMs의 신뢰도와 그 이유를 설명하는 기능은 사용자 감정을 더 잘 이해하는 데 도움이 되었습니다.



### Gradient flow in parameter space is equivalent to linear interpolation in output spac (https://arxiv.org/abs/2408.01517)
- **What's New**: 본 연구는 심층 학습에서 신경망을 훈련시키는 많은 알고리즘의 기초인 일반적인 파라미터 공간의 gradient flow를 최적화하여 출력 공간에서 유클리드 gradient flow로 변형할 수 있음을 증명합니다. 더 나아가, Jacobian 행렬이 full rank일 때, 시간 변수를 다시 매개변수화하여 단순한 linear interpolation을 통해 전역 최소값에 도달할 수 있음을 제시합니다.

- **Technical Details**: 논문은 파라미터 공간(ℝθK)에서 정의된 gradient flow를 고려하며, 비용 함수가 파라미터 함수로서 비볼록(non-convex)하므로 표준 gradient flow가 전역 최소값에 도달하지 못할 수 있음을 언급합니다. 이를 해결하기 위해 출력 공간(ℝxQN)의 행렬적 구조를 활용하여 두 흐름의 동치성을 증명합니다. Rank loss가 없는 경우, 유클리드 flow를 시간 변수에 따라 선형 보간(linear interpolation)과 동일하게 재매개변수화할 수 있음을 증명합니다. Rank loss가 있는 경우, 선형 보간에서의 편차를 표현하는 공식도 제시합니다.

- **Performance Highlights**: 연구는 파라미터 공간에서의 수정된 gradient flow와 출력 공간에서의 유클리드 gradient flow가 동일한 critical sets를 가진다는 것을 보입니다. 또한 Jacobian 행렬이 full rank일 때 전역 최소값에 도달할 수 있으며, 이는 신경망의 최적화 문제에 큰 활용 가치를 가집니다. Neural Tangent Kernel와 관련된 작업도 함께 논의되었습니다.



### LocalValueBench: A Collaboratively Built and Extensible Benchmark for Evaluating Localized Value Alignment and Ethical Safety in Large Language Models (https://arxiv.org/abs/2408.01460)
- **What's New**: 새로운 논문에서는 LocalValueBench 라는 확장 가능한 벤치마크를 도입하여, 대형 언어 모델(LLMs)의 호주 가치 준수 여부를 평가하고, 전 세계의 규제 기관이 현지 가치 정렬을 위해 벤치마크를 개발할 수 있는 프레임워크를 제공합니다. 이를 통해, LLM들이 각 지역의 문화적, 법적, 이념적 가치를 얼마나 잘 파악하고 있는지를 심층적으로 평가할 수 있는 방법론이 제안되었습니다.

- **Technical Details**: LocalValueBench는 윤리적 추론의 새로운 유형학(typology)과 'interrogation' 접근 방식을 이용하여 LLM의 가치 정렬을 조사합니다. 질문 작성 과정은 다양한 윤리적 시나리오와 현지 가치 고려사항을 포함시키는 것에 중점을 두었으며, 프롬프트 엔지니어링(prompt engineering) 전략을 활용해 원 질문을 제시하고, 대안적 관점을 도입하며, LLM들이 이 관점을 명확히 설명하도록 강요했습니다. 평가 기준은 현지 가치에서 벗어나는 정도를 정량화하여, 엄격한 평가를 보장합니다. 이 벤치마크를 통해, 호주 가치를 준수하는지 평가할 수 있게 되었으며, 다른 지역의 규제 기관이 자신들만의 벤치마크를 개발하는 기반이 됩니다.

- **Performance Highlights**: 상업적인 LLM의 비교 분석 결과, 각 모델의 현지 가치와 윤리적 기준 준수에 있어 상당한 차이가 나타났습니다. GPT-4는 '동성 결혼' 카테고리에서 질문에 대답을 거부하여 낮은 점수를 얻은 반면, Gemini 1.5 Pro는 여러 카테고리에서 현지 가치에 잘 맞는 성과를 보였으나 '사형' 카테고리에서 대답을 거부했습니다. 초상적 일관성 면에서 대부분의 카테고리에서 Gemini 1.5 Pro와 Claude 3 Sonet이 GPT-4를 능가했습니다. 결과적으로, 각 모델은 다양한 윤리적 시나리오에서 다른 성과를 보였으며, 지속적인 벤치마크의 개선이 필요함을 시사합니다.



### AgentPeerTalk: Empowering Students through Agentic-AI-Driven Discernment of Bullying and Joking in Peer Interactions in Schools (https://arxiv.org/abs/2408.01459)
- **What's New**: 이 연구는 학교에서 발생하는 괴롭힘과 농담을 구분하는 데 있어 대형 언어 모델(large language models, LLMs)을 활용하는 가능성을 분석했습니다. 특히 ChatGPT-4, Gemini 1.5 Pro, Claude 3 Opus 모델들이 사용되었습니다. 그 결과, ChatGPT-4가 가장 뛰어난 성과를 보였으며, 연속적이고 실시간 지원을 제공할 수 있는 잠재력이 있다고 평가되었습니다.

- **Technical Details**: 이 연구에서는 LLMs가 괴롭힘 문제를 심리적 관점에서만 다루는 것이 아니라 법적, 윤리적 관점에서도 조언을 제공할 수 있는지 여부를 조사했습니다. 이를 위해 'agentic approach'를 시뮬레이션하여 LLMs에게 외부 정보(법적 문서, 윤리적 가이드라인, 문화적 설명)를 제공하였습니다. ChatGPT-4는 이 접근법에서 뛰어난 성과를 보였으나, Gemini 1.5 Pro와 Claude 3 Opus는 혼합된 결과를 보였습니다.

- **Performance Highlights**: ChatGPT-4는 구체적인 상황에서 정확도가 크게 향상되었습니다. 예를 들어, 신체 이미지 관련 괴롭힘 시나리오에서 0.4점이 증가했습니다. Gemini 1.5 Pro와 Claude 3 Opus는 일부 시나리오에서 성과가 감소하거나 결과를 전혀 생성하지 못했습니다. 통계 분석 결과, ChatGPT-4가 가장 일관된 성과를 보였으며, Gemini와 Claude는 변동성이 더 컸습니다. ANOVA 테스트 결과 p-값이 0.0041로, 모델 간의 성과 차이가 유의미함을 확인했습니다.



### Surveys Considered Harmful? Reflecting on the Use of Surveys in AI Research, Development, and Governanc (https://arxiv.org/abs/2408.01458)
Comments:
          To appear in 7th AAAI Conference on AI, Ethics, and Society (AIES)

- **What's New**: 본 논문은 인공지능(AI) 연구, 개발 및 거버넌스에서 공공 참여의 중요성이 증가함에 따라, 이를 평가하기 위해 설문 조사가 사용되는 방법을 비판적으로 살펴봅니다. 6개국에 걸친 파일럿 설문 조사와 44개의 관련 논문을 체계적으로 검토하여 공공 설문 조사에서 나타나는 주요 관점과 방법론적 특징을 분석했습니다.

- **Technical Details**: 이번 연구는 6개국에서 수행된 파일럿 설문 조사를 반영적으로 분석하고, AI 관련 공공 설문 조사를 다룬 44개의 논문을 체계적으로 검토하였습니다. 연구 질문(RQ) 중 하나는 설문 조사가 어떻게 AI에 대한 사람들의 가치관, 인식, 경험을 이해하는 도구로 자리 잡아왔는지를 묻고 있습니다. 또한, AI 주제와 관련된 대규모 설문 연구의 윤리적 설계, 배포, 해석 및 보고를 이끄는 독특한 질문을 제공합니다.

- **Performance Highlights**: 연구 결과, AI 관련 공공 설문 조사가 서구 지식, 가치 및 가정에 취약하다는 점을 발견했습니다. 윤리적 개념과 사회적 가치의 위치 지정에서부터 배포 전략에 대한 비판적 담론이 부족하고 보고의 투명성에서도 일관성이 없음을 보여줍니다. 44개의 논문 중 14개는 대표성(representation)을 주장했으나, 용어의 일관성 없는 사용이 대표성의 착각을 일으켰으며, 이는 소외된 커뮤니티에게 해를 끼칠 수 있습니다. 또한, 6개 논문만이 글로벌 사우스(Global South) 국가의 저자를 포함하고 있었으며, 11개 논문은 연구가 수행된 국가의 저자가 없었습니다.



### Ontology of Belief Diversity: A Community-Based Epistemological Approach (https://arxiv.org/abs/2408.01455)
Comments:
          AIES 2024

- **What's New**: 새로운 연구는 신념 체계의 프래그매틱 온톨로지(ontology)를 개발하는 것에 중점을 두고 있습니다. 이는 AI 응용 프로그램에서 분류, 공정성, 인간 상호작용을 위해 사회적 개념 온톨로지(ontology)가 암묵적으로 필요하기 때문입니다. 본 연구에서는 커뮤니티 기반 설계를 반복하여 합의에 도달하며 신념의 기본적인 차이점을 분류하는 데 에피스테몰로지(지식론적) 방법이 최적임을 발견했습니다.

- **Technical Details**: 본 연구에서는 에피스테몰로지적(epistemological) 방법을 사용하여 신념 체계의 기본적인 차이점을 분류하는 데 중점을 두었습니다. 이는 포괄성과 간결성의 원칙을 최대한 존중하며 커뮤니티 기반 설계를 반복하여 합의에 도달하는 방식으로 이루어졌습니다. 실험적으로는 신념의 공정성을 평가하기 위해 용어 주석(Annotation) 및 감정 분석(Sentiment Analysis) 실험을 통해 방법론의 유용성과 해석 가능성을 입증하였습니다.

- **Performance Highlights**: 사용자 연구(User Studies)를 통해 본 연구의 방법론이 실제 적용에서 신념의 공정성을 평가하는 데 유용함을 확인하였습니다. 실제 언어 모델(Language Models)에서 신념의 다양한 범주를 공정하게 다룰 수 있도록 성능을 입증했습니다.



### Amman City, Jordan: Toward a Sustainable City from the Ground Up (https://arxiv.org/abs/2408.01454)
Comments:
          12 pages, 3 figures, 6 tables, 56 references

- **What's New**: 스마트시티(Smart Cities, SC)의 개념이 최근 들어 큰 주목을 받고 있습니다. 특히 정보통신기술(ICT)의 발전으로 일상적인 물건들이 '스마트화'되고, 이를 통해 인간의 삶을 편리하게 만들기 위한 새로운 시도가 이루어지고 있습니다. 이 연구는 요르단의 수도 암만(Amman)을 스마트시티로 변화시키기 위한 계획과 현재 진행 상황에 대해 다루고 있습니다.

- **Technical Details**: 암만 스마트시티 개발에는 사물인터넷(IoT), 클라우드 컴퓨팅(Cloud Computing), 에지 컴퓨팅(Edge Computing) 등의 최신 ICT 기술들이 활용되고 있습니다. IoT 기술은 일상 생활에서 사용되는 다양한 물건들에 센서와 연결성을 부여하여 정보를 교환하는 것이 핵심이며, 클라우드 컴퓨팅과 에지 컴퓨팅은 데이터를 효율적으로 저장하고 처리하기 위한 다양한 아키텍처들을 제공합니다. 암만 시는 이러한 기술들을 활용하여 공공 서비스와 환경 모니터링을 개선하려 하고 있습니다.

- **Performance Highlights**: 암만은 2024년 IMD Smart City Index에서 142개 도시 중 128위를 차지하여, 이전 해의 135위에서 상승했습니다. 암만은 고속 통신망, 데이터 센터, 사이버 보안, 인공지능(AI) 등의 현대적 기술 인프라를 도입하여 시민들의 삶의 질을 향상시키고 지속 가능한 발전을 목표로 하고 있습니다. 또한, 100% 전자 서비스 완료, 온라인 서비스 제공 확대, Greater Amman Municipality 애플리케이션 도입 등 여러 개선 사항들을 달성하고 있습니다.



### Reporting and Analysing the Environmental Impact of Language Models on the Example of Commonsense Question Answering with External Knowledg (https://arxiv.org/abs/2408.01453)
Comments:
          Presented at Bonn Sustainable AI 2023 conference

- **What's New**: 최근 연구는 거대한 언어 모델(LLM)이 환경에 미치는 영향을 조사합니다. T5 모델을 외부 지식과 결합하여 질문-응답 작업을 fine-tuning 하였으며, 이를 통해 모델의 학습 시간과 탄소 배출량을 측정했습니다. 연구 결과는 성능과 효율성을 모두 고려해야 최적의 결과를 얻을 수 있다는 점을 강조합니다.

- **Technical Details**: 이 연구는 개념넷 (ConceptNet)과 ATOMIC 같은 대규모 지식 그래프(Knowledge Graph, KG)를 T5 모델에 결합하여 비교 상식을 주입하는 방식을 채택했습니다. 또한, 상식 질문-응답 (CSQA) 데이터셋인 TellMeWhy를 이용하여 모델을 fine-tuning 했고, 그 과정에서 발생하는 환경적 영향을 분석했습니다.

- **Performance Highlights**: 연구의 결과, 작은 모델이 항상 지속 가능한 옵션이 아니며, 학습 시간이 늘어난다고 해서 성능이 항상 향상되는 것은 아님을 보여줍니다. 가장 최적의 결과는 성능과 효율성 두 측면을 모두 고려해야 달성할 수 있습니다.



### Building a Domain-specific Guardrail Model in Production (https://arxiv.org/abs/2408.01452)
- **What's New**: 이 연구는 K-12 교육 플랫폼에 실제로 배포된 생산 등급의 '가드레일 모델'(Guardrail model)을 개발하는 과정을 설명합니다. 특히 도메인 특정 요구사항과 규범을 충족하기 위해 교육 분야에서 내용의 적절성을 보장하는 것에 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 모델 SPADE(Safe and Performant AI Deployment with continuous Evaluation)는 여러 가지 안전성과 적절성 요건을 충족하도록 설계되었습니다. 특히 데이터 프라이버시 규정인 FERPA와 COPPA를 준수하고, 교육 내용의 안전성과 적절성을 보장하며, 실시간 응답 성능을 낮은 비용과 지연으로 제공합니다. 이를 통해 하드웨어 인프라에서부터 언어 모델 추론 최적화에 이르는 전체 스택에서 다양한 설계 선택이 적용되었습니다.

- **Performance Highlights**: 교육 관련 벤치마크와 일반적인 안전성 관련 공공 벤치마크에서 제안된 도메인 특정 '가드레일 모델'은 유사하거나 더 큰 크기의 공개 및 폐쇄된 '인스트럭션 튜닝된'(instruction-tuned) 모델을 능가하는 성능을 보였습니다.



### AI Act for the Working Programmer (https://arxiv.org/abs/2408.01449)
Comments:
          25 pages, 2 figures; submitted to AISoLA 2024

- **What's New**: 유럽 AI 법(AI Act)은 AI 기술의 개발 및 사용에 관한 새로운 법적 요구사항을 제공하는 새로운 법안입니다. 이 법안은 유럽에서 사람들에게 영향을 미칠 수 있는 AI 기술에 대해 특정 요구사항을 강제할 예정입니다. 이 논문은 소프트웨어 도메인에서 일하는 전문가들을 위한 도움말로, '실무 프로그래머'에게 AI 법의 복잡성을 이해하는 첫걸음을 제공합니다.

- **Technical Details**: AI 법은 위험 기반 접근 방식을 통해 AI의 개발 및 배포가 안전하고 신뢰할 수 있도록 보장합니다. 113개의 기사, 180개의 시조, 13개의 부록으로 구성된 이 법안은 총 144페이지에 달합니다. 이 논문은 AI 법에 의해 규제되는 AI 기술을 구분하고 관련 의무를 맵핑하여 실무 프로그래머가 법의 복잡성을 쉽게 이해할 수 있도록 돕습니다. 중요한 역할로서 '제공자(provider)'의 의미를 중심으로 AI 법의 요구사항을 이해하도록 돕습니다.

- **Performance Highlights**: 이 논문은 크게 세 가지 기여를 합니다. 첫째, 규제되는 AI 기술에 대한 개요를 제공하여 프로그래머가 어떤 법적 의무가 적용될 수 있는지 결정할 수 있도록 돕습니다. 둘째, 관련 의무를 맵핑하여 프로그래머가 법의 복잡성을 좁힐 수 있도록 질문과 흐름도를 제공합니다. 마지막으로, 범용 AI 모델(GPAI)과 같은 준비된 AI 모델을 사용하는 것에 대한 법적 위험을 이해하고 예측할 수 있도록 하는 법적 및 컴퓨터 과학적 관점에서의 협업을 제시합니다.



### Estimating Environmental Cost Throughout Model's Adaptive Life Cyc (https://arxiv.org/abs/2408.01446)
Comments:
          Accepted in the AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society, 2024

- **What's New**: 이번 연구에서는 PreIndex라는 예측 인덱스를 도입하여 모델 재학습 시 에너지 소비 및 탄소 배출량을 추정할 수 있는 방법을 제안합니다. PreIndex는 배포 환경이나 입력 데이터의 변화에 따라 모델을 지속적으로 재사용하여 AI 및 딥러닝과 관련된 탄소 발자국을 줄이는 사회적으로 유익한 접근 방법입니다.

- **Technical Details**: PreIndex는 데이터의 한 번의 forward pass로 환경 비용(탄소 배출량 및 에너지 사용량)을 추정할 수 있으며, 에폭(epoch), gradient norm, 모델 파라미터 변화의 크기 등 딥러닝과 관련된 기타 자원 지표들도 예측할 수 있습니다. 다양한 데이터셋과 모델 구조, 분포 변동의 유형 및 강도에 관계없이 사용할 수 있습니다.

- **Performance Highlights**: PreIndex는 데이터의 분포 변동에 따른 재학습에 관련된 자원을 추정할 수 있는 단일 값을 제공함으로써, 사용자가 재학습 결정에서 가장 비용 효율적이고 지속 가능한 옵션을 선택할 수 있게 도와줍니다. 이를 통해 환경에 미치는 영향을 최소화하면서 모델 재사용을 가능하게 합니다.



### MiranDa: Mimicking the Learning Processes of Human Doctors to Achieve Causal Inference for Medication Recommendation (https://arxiv.org/abs/2408.01445)
- **What's New**: 새로운 모델 MiranDa는 약물 추천 전 과정에서 병원 입원 예상 기간 (ELOS)을 반사실적 (counterfactual) 결과로 제공하여 임상 실무와 모델 학습을 안내하는 최초의 실질적인 모델입니다. 이 모델은 의사들의 교육 과정을 모사하여 두 개의 그래디언트 스케일링 단계를 사용합니다: 증거 기반 트레이닝 단계와 강화 학습 기반의 치료 최적화 단계입니다.

- **Technical Details**: MiranDa 모델은 일반적인 감독 학습(supervised learning)과 그래디언트 공간에서의 강화 학습(reinforcement learning)을 결합하여 약물 추천을 최적화합니다. 증거 기반 트레이닝 단계에서는 ELOS 데이터를 활용한 감독 학습을, 치료 최적화 단계에서는 그라디언트 공간에서의 강화 학습을 통해 최적의 약물 조합을 탐색합니다. 이를 통해 모델은 병원 입원 기간 예측에서 실제 입원 기간과 평균 0.01일의 차이를 보였습니다.

- **Performance Highlights**: MiranDa 모델은 Medical Information Mart for Intensive Care III와 IV 데이터셋을 평가한 결과, ELOS를 줄이는 데 있어서 탁월한 성능을 보였습니다. 다섯 가지 지표에서 변형 기반 모델보다 뛰어난 성과를 나타냈으며, 특히 '절차별' 약물 조합의 구조적 속성을 제공함으로써 약물의 효능을 더욱 향상시켰습니다. 이 모델은 거의 모든 의료 과제에 적용 가능하며, 예측 결과를 평가할 수 있는 정보를 갖춘 과제에 적합합니다.

- **Source Code**: MiranDa 모델의 소스 코드는 https URL에서 사용 가능합니다.



### No Size Fits All: The Perils and Pitfalls of Leveraging LLMs Vary with Company Siz (https://arxiv.org/abs/2408.01444)
Comments:
          17 pages, 3 figures

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 산업적 적응 과정에서 발생할 수 있는 다양한 문제와 도전에 초점을 맞추어, 이를 해결하기 위한 실용적인 가이드를 제공합니다. 특히, 조직의 크기에 따른 LLM 도입의 문제점을 심도 있게 분석하고 있습니다.

- **Technical Details**: 이 연구는 세 가지 전략을 채택했습니다. 첫째, 실제 산업 전문가들과의 사례 연구를 통해 주요 연구 질문을 설정했습니다. 둘째, 기존 산업 출판물을 조사하여 이 질문들을 다루었습니다. 셋째, 산업들이 LLMs를 보다 효율적으로 사용할 수 있도록 실용적인 가이드를 제공했습니다.

- **Performance Highlights**: 주요 기여는 LLMs의 산업적 채택에 따른 다양한 도전 과제를 식별하고, 이에 대한 잠재적 해결책을 제안하는 것입니다. 이 과제에는 데이터 기밀성, LLM 응답의 신뢰성, 인프라 병목 현상, 도메인 특화 적응, 합성 데이터 생성 및 윤리적 문제 등이 포함됩니다. 또한 중소기업과 대기업을 위한 맞춤형 실용 가이드를 제공합니다.



### AI for All: Identifying AI incidents Related to Diversity and Inclusion (https://arxiv.org/abs/2408.01438)
Comments:
          25 pages, 9 figures, 2 tables

- **What's New**: AI 기술의 급속한 확산과 함께 다양성과 포용성(D&I)이 중요한 문제가 되었습니다. 본 연구는 AI 시스템에서 발생하는 D&I 이슈를 식별하고 이해하기 위해 AI 사건 데이터베이스(AIID 및 AIAAIC)를 수동으로 분석하였습니다. 연구팀은 결정 트리를 개발하여 D&I 관련 AI 사건을 조사하고 공개 저장소를 구성하였습니다. 이 결정 트리는 카드 소팅(card sorting) 및 포커스 그룹 토론을 통해 검증되었습니다.

- **Technical Details**: 연구는 두 개의 AI 사건 데이터베이스(AIID 및 AIAAIC)를 분석하여 사건을 '다양성과 포용성과 관련된 사건', '관련 없는 사건', '추가 정보가 필요한 사건'으로 분류하였습니다. 또한, D&I 이슈를 분석하기 위한 결정 트리를 개발하였으며, 이를 검증하기 위해 AI/ML 연구자 및 D&I 전문가들과의 상호 참여 활동을 통해 조사하였습니다. 최종적으로 D&I 관련 AI 사건의 공개 저장소를 구축하였습니다.

- **Performance Highlights**: 분석된 AI 사건 중 거의 절반이 D&I와 관련이 있는 것으로 나타났으며, 특히 인종, 성별, 연령 차별이 두드러졌습니다. 연구의 주요 기여는 다음과 같습니다: D&I 관련 AI 사건 식별, 분석 도구 개발, 공개 저장소 생성, D&I 관련 AI 사건의 근본 원인 탐구, 책임 있는 AI 실천 촉진. 이 연구는 향후 기술 개발 및 적용에 있어 포용적이고 공정한 AI 시스템을 촉진하는 데 기여할 것입니다.



### Building an Ethical and Trustworthy Biomedical AI Ecosystem for the Translational and Clinical Integration of Foundational Models (https://arxiv.org/abs/2408.01431)
Comments:
          3 figures, 3 tables

- **What's New**: 기초 모델(Foundational Models, FMs)이 생의학 AI 생태계의 핵심으로 떠오르고 있습니다. 이 모델들은 다중 모드 생의학 데이터(multimodal biomedical data)를 표현하고 문맥화하는 능력이 뛰어나 다양한 작업에 적용할 수 있습니다. 예를 들어, 생의학적 추론(biomedical reasoning), 가설 생성(hypothesis generation), 임상 의사결정(clinical decision-making) 등에 활용됩니다. 이 논문에서는 FMs를 중심으로 윤리적이고 신뢰할 수 있는 AI(ETAI) 생의학 생태계의 기본 구성 요소를 검토하고 주요 과제와 해결책을 다루고 있습니다.

- **Technical Details**: ETAI 생의학 생태계는 임상 환경에서 FMs을 통합하는 7가지 핵심 구성 요소로 정의됩니다: 데이터 라이프사이클 관리(Data Lifecycle Management), 데이터 처리(Data Processing), 모델 개발(Model Development), 모델 평가(Model Evaluation), 임상 번역(Clinical Translation), AI 거버넌스 및 규제(AI Governance and Regulation), 그리고 이해관계자 참여(Stakeholder Engagement)가 포함됩니다. 각 구성 요소는 AI 시스템의 신뢰성, 투명성, 재현성을 보장하기 위해 필수적입니다.

- **Performance Highlights**: FMs를 효과적으로 활용하기 위해 여러 윤리적 도전과제들을 극복해야 합니다. 예를 들어, 데이터, 알고리즘, 사용자 상호작용에서 발생할 수 있는 편향(bias)을 평가하고 완화하는 기술이 필요합니다. 또한, 해석 가능성(interpretability), 설명 가능성(explainability), 책임성(accountability)은 AI 시스템의 신뢰성을 보장하는 데 중요합니다. 특히, 환자 개인정보 보호(patient privacy)와 보안을 위해 데이터 접근, 클라우드 데이터 프라이버시, 환자 재식별(patient re-identification), 멤버십 유추 공격(membership inference attacks), 데이터 저장(data memorization) 등의 문제들을 해결해야 합니다. 마지막으로, 글로벌 표준에 따른 AI 거버넌스와 규제는 생의학 분야에서 윤리적인 AI 사용을 지도하는 데 필수적이며, 임상 번역을 위해 AI 파이프라인의 모든 단계에서 이해관계자의 참여가 필요합니다.



### SUSTechGAN: Image Generation for Object Recognition in Adverse Conditions of Autonomous Driving (https://arxiv.org/abs/2408.01430)
Comments:
          10 pages, 9 figures

- **What's New**: 이번 연구에서는 자율주행 차량의 악조건에서 객체 인식을 개선하기 위해 듀얼 어텐션 모듈과 멀티스케일 생성기를 포함한 새로운 SUSTechGAN을 제안했습니다. 특히, 악천후와 야간 상황에서 운전 이미지를 생성하고, 이를 사용해 YOLOv5 객체 인식 네트워크를 재학습시키는 방법을 검토했습니다.

- **Technical Details**: SUSTechGAN은 듀얼 어텐션 모듈과 멀티스케일 생성기를 사용하여 운전 이미지를 생성합니다. 듀얼 어텐션 모듈은 지역의 의미론적 특징 추출을 개선하고, 멀티스케일 생성기는 다양한 크기의 특징을 고려하여 고품질 이미지를 생성합니다. 또한, 새로운 손실 함수로 탐지 손실(detection loss), 적대적 손실(adversarial loss), 사이클 일관성 손실(cycle consistency loss)을 제안하여 객체 인식을 향상시킵니다.

- **Performance Highlights**: 실험 결과, SUSTechGAN이 생성한 운전 이미지를 통해 재학습된 YOLOv5 모델이 악천후와 야간 조건에서 객체 인식 성능이 크게 향상되었습니다. 이는 기존의 잘 알려진 GAN들보다 뛰어난 성능을 보여줬습니다.



### An Agile Adaptation Method for Multi-mode Vehicle Communication Networks (https://arxiv.org/abs/2408.01429)
- **What's New**: 이 논문은 차량통신 네트워크(Vehicle Communication Networks, VCNs)에서 통신 모드 할당이 통신 효율성에 미치는 영향을 연구합니다. 특히, 마르코프 결정 프로세스(Markov Decision Process)와 강화 학습(Reinforcement Learning)을 통해 주행 시나리오와 비즈니스 요구 사항에 맞춘 멀티 모드 통신 장치의 민첩 적응 메커니즘(Agile Adaptation Mechanism)을 구축합니다. 이에 Q-learning을 사용해 민첩 적응 강화 학습 모델(AARLM)을 훈련하고 최적의 정책을 도출합니다.

- **Technical Details**: 이 논문에서는 멀티 모드 VCNs를 고려합니다. 여기서 차량과 도로변 유닛(RSUs)은 다중 통신 모드를 지원할 수 있습니다. 대부분의 기기 간 근거리 통신(PC5), 셀룰러, Wi-Fi를 통해 차량 간, 차량과 RSU 간, 그리고 차량과 기지국 간의 신뢰성 있고 장거리 통신을 달성합니다. AARLM은 이들 통신 모드 간의 선택을 최적화해 전송 지연을 최소화합니다. 이를 위해 각 노드 간 전송 지연 요구사항, 네트워크 토폴로지, 링크 조건 등의 데이터셋을 기반으로 다중 사용자 전송을 촉진합니다.

- **Performance Highlights**: 실험 결과, 제안된 AARLM은 역동적인 차량 네트워킹 환경에 신속하게 적응하며 높은 통신 효율성과 동시성을 달성함을 보여줍니다. 이는 차량 간 통신과 차량-도로 통신의 실시간 요구를 효과적으로 충족시키며, 데이터 전송의 신뢰성, 적응력, 그리고 전송 지연을 엄격하게 관리하는 데 성공했습니다.



### Transferable Adversarial Facial Images for Privacy Protection (https://arxiv.org/abs/2408.01428)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 새로운 얼굴 프라이버시 보호 스키마가 제안되었습니다. 이 스키마는 추가적인 사용자의 참조 이미지를 필요로 하지 않으며 높은 시각적 품질을 유지하면서도 높은 전이성을 가진 자연스러운 적대적 얼굴 이미지를 생성합니다.

- **Technical Details**: 이 연구는 전체 얼굴 공간을 직접 형성하는 것을 제안하며, 이를 위해 생성 모델의 잠재 공간을 탐색합니다. 주요 구성 요소로는 글로벌 적대적 잠재 검색(Global Adversarial Latent Search, GALS), 주요 랜드마크 정규화 모듈(Key Landmark Regularization, KLR) 및 다양한 잠재 공간의 영향을 조사하는 과정이 포함됩니다. 최적의 잠재 공간으로는 \\mathcal{F} 공간이 선택되었습니다.

- **Performance Highlights**: 두 개의 데이터셋에 대한 광범위한 실험 결과, 제안된 접근 방식은 최신 방법에 비해 평균 25% 향상된 전이성을 보였으며, 상용 얼굴 인식 API(예: Face++, Aliyun, Tencent)에서도 10%의 향상을 달성했습니다.



### Siamese Transformer Networks for Few-shot Image Classification (https://arxiv.org/abs/2408.01427)
Comments:
          12 pages

- **What's New**: 이번 연구에서는 인간의 뛰어난 시각 분류 능력에서 영감을 받아, 새로운 이미지에 대해 적은 예시만으로도 정확하게 인식하는 방법을 제안합니다. 기존의 몇몇 샷 이미지 분류(few-shot image classification) 방법은 글로벌 특징(global features) 또는 로컬 특징(local features)에 중점을 둡니다. 하지만 이번 연구에서는 두가지 특징을 통합하는 Siamese Transformer Network (STN) 접근법을 제안했습니다.

- **Technical Details**: 제안된 방법은 두 개의 병렬 네트워크 지점을 사용하여 각각 글로벌 특징과 로컬 특징을 추출합니다. 이 네트워크는 사전 학습된 Vision Transformer (ViT) 아키텍처를 기반으로 합니다. 글로벌 특징에는 유클리드 거리 측정(Euclidean distance measure)을 사용하고, 로컬 특징에는 KL 발산(Kullback-Leibler divergence) 측정을 적용합니다. 두 측정치를 통합하기 위해 L2 정규화(L2 normalization)를 수행한 후 가중치를 부여하여 최종 유사도 점수를 얻습니다. 이 과정에서 메타 학습 접근법(meta-learning approach)을 사용하여 네트워크를 미세 조정합니다.

- **Performance Highlights**: 제안된 STN 방법은 5-shot 및 1-shot 시나리오 모두에서 기존 최첨단 모델들과 비교하여 우수한 성능을 보였습니다. 네 가지 인기 있는 벤치마크 데이터셋을 사용한 실험에서 STN이 가장 뛰어난 성능을 나타냈습니다.



### Preference-Based Abstract Argumentation for Case-Based Reasoning (with Appendix) (https://arxiv.org/abs/2408.00108)
Comments:
          Accepted for KR2024. Includes Appendix

- **What’s New**: 이 연구는 Abstract Argumentation(추상적 논증)과 Case-Based Reasoning(사례 기반 추론, CBR)을 결합하여 사용자가 정의한 선호도를 반영하는 새로운 방식인 AA-CBR-P를 소개합니다. 이 모델은 사용자가 선호하는 비교 접근 방식을 정의할 수 있게 하여, 이러한 선호도를 기반으로 예측을 진행합니다.

- **Technical Details**: AA-CBR-P는 사용자 정의 선호도를 기반으로 하는 추상적 논증 및 사례 기반 추론 시스템입니다. 이 모델은 기존의 추상적 논증이 특정 논증 요소에 대한 선호도를 표현하는 데 충분치 않다는 점을 해결합니다. 사용자가 여러 비교 접근 방식을 정의하고 이들에 대한 우선 순위를 설정하도록 하여, 모델이 이러한 우선 순위에 따라 예측을 진행합니다.

- **Performance Highlights**: 모델의 유효성을 검증하기 위해, 주로 뇌 종양 환자를 평가하는 임상 시험의 의료 데이터를 사용했습니다. 실험 결과, AA-CBR-P는 다른 해석 가능한 머신러닝 모델들보다 우수한 성능을 보였습니다.




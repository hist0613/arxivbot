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




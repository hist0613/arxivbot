New uploads on arXiv(cs.CL)

### AnyTaskTune: Advanced Domain-Specific Solutions through Task-Fine-Tuning (https://arxiv.org/abs/2407.07094)
- **What's New**: AnyTaskTune은 특정 비즈니스 컨텍스트에 맞춤화된 모델의 성능을 높이기 위한 새로운 파인 튜닝 방법론입니다. 이 방법론을 사용하여 다양한 도메인-특정(task-specific) 작업들의 성능을 최적화하였습니다. 저희는 법률, 금융, 헬스케어, 심리학, 소비자 서비스, 인사 등 여러 분야에 걸쳐 20여 가지의 하위 작업들에 대해 종합적인 파인 튜닝 실험을 수행했습니다.

- **Technical Details**: AnyTaskTune은 특정 도메인 내에서 목표가 된 하위 작업들을 식별하고 정의하는 과정을 포함합니다. 이후, 이러한 작업에 맞춤화된 데이터셋을 생성하여 모델을 파인 튜닝합니다. 이러한 과정을 통해 특정 작업별 모델 성능을 최적화합니다. 우리는 다양한 도메인에서 명시적 데이터 세트(Explicit Data Sets)를 활용하여 Task-Fine-Tuning 방법의 효과를 입증했습니다.

- **Performance Highlights**: Task-Fine-Tune 방법론을 통해 파인 튜닝된 모델은 이러한 특정 작업들에서 우수한 성능을 달성했으며, 일반적인 능력을 우선시한 모델보다도 더 나은 성능을 보였습니다. 이 연구 결과는 AnyTaskTune이 특정 도메인과 작업에 맞춰 보다 효율적이고 정확하게 작동할 수 있음을 보여줍니다.



### FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation (https://arxiv.org/abs/2407.07093)
Comments:
          Github at this https URL

- **What's New**: FBI-LLM이라는 큰 언어 모델을 처음으로 처음부터 훈련하여 전체 정밀도 모델과 맞먹는 성능을 보여주는 연구를 소개합니다. 이 모델은 32비트 대신 단 1비트만으로도 모델을 구현하며, 특히 자동회귀 증류(AD) 손실을 사용하여 훈련합니다. 이로써 고정밀도 모델과 동등한 모델 크기와 훈련 데이터 볼륨을 유지하면서 퍼플렉시티(perplexity)와 특정 작업에서의 효율성을 달성합니다.

- **Technical Details**: - 모델 크기는 130M, 1.3B, 7B로 다양하게 적용. 
- Transformer 기반 LLM 아키텍처 사용. 
- 모든 선형 모듈을 완전 이진화한 FBI-linear로 대체. 
- 자동회귀 증류 손실을 통해 각 토큰 위치에서 교사 모델의 예측 확률을 매치.
- 기존 모델의 가중치를 상속할 필요 없이 처음부터 훈련 가능.

- **Performance Highlights**: 기존의 부분 이진화 또는 1.58비트 모델과 달리, 처음부터 이진화된 LLM을 성공적으로 훈련. 퍼플렉시티와 여러 다운스트림 작업에서 기존의 고정밀도 모델과 비슷한 성능을 보여줍니다. 특히 파라미터 상속과 무관하게 훈련할 수 있음을 다양한 실험을 통해 검증했습니다.



### CopyBench: Measuring Literal and Non-Literal Reproduction of Copyright-Protected Text in Language Model Generation (https://arxiv.org/abs/2407.07087)
- **What's New**: CopyBench라는 새로운 벤치마크를 도입하여 언어 모델(LM)이 저작권이 있는 콘텐츠를 얼마나 복제하는지를 평가합니다. 이 벤치마크는 문자 그대로의 복제(literal copying)뿐만 아니라 비문자적 복제(non-literal copying)도 평가할 수 있습니다.

- **Technical Details**: CopyBench는 저작권이 보호된 소설책을 텍스트 소스로 사용하여 문자적 복제와 비문자적 복제를 자동으로 평가합니다. 비문자적 복제는 사건 복제(event copying)와 캐릭터 복제(character copying)로 분류될 수 있습니다. 다양한 최첨단 언어 모델, 예를 들어 Llama2, Llama3, Mistral, GPT-3.5-Turbo 및 GPT-4-Turbo 등을 CopyBench에서 평가했습니다.

- **Performance Highlights**: 연구 결과, 문자적 복제는 비교적 드물지만 비문자적 복제는 모든 모델에서 의미 있는 수준으로 발생하는 것으로 나타났습니다. 예를 들어, Llama3의 경우 문자적 복제율은 0.2%에서 10.5%까지 증가했으며, 비문자적 복제율은 2.3%에서 6.9%까지 증가했습니다. 또한, GPT-3.5에서 GPT-4로의 전환은 문자적 복제를 줄였지만 비문자적 복제를 증가시켰습니다.

- **Mitigation Strategies**: 현재의 완화 전략(training-time alignment 및 inference-time mitigation methods)은 문자적 복제의 감소에는 효과적이었지만 비문자적 복제는 효과적이지 않았습니다. 예를 들어, MemFree decoding은 문자적 복제를 줄이는 데 성공했지만 비문자적 복제에는 영향을 미치지 않았습니다.



### Adapting LLMs to Hebrew: Unveiling DictaLM 2.0 with Enhanced Vocabulary and Instruction Capabilities (https://arxiv.org/abs/2407.07080)
- **What's New**: DictaLM2.0와 DictaLM2.0-Instruct라는 두 개의 큰 언어 모델(LLMs)이 히브리어와 영어로 약 2000억 개의 토큰을 사용해 학습되었습니다. Mistral 모델에서 파생된 이 모델들은 저자원 언어 학습의 독특한 어려움을 극복하기 위해 특별히 설계되었으며, 히브리어의 언어적 특성에 효과적으로 적응할 수 있도록 새로운 교육 방법론을 도입했습니다.

- **Technical Details**: DictaLM2.0 시리즈는 Mistral 모델을 기반으로 히브리어와 영어 데이터를 각각 약 1000억 토큰씩 사용해 훈련되었습니다. 이 과정에서 히브리어에 특화된 토큰들을 확장하고, 임베딩 증류(embedding distillation) 기술을 사용해 모델의 학습 효율성을 높였습니다. 또한, DictaLM2.0-Instruct 모델은 특정 작업의 지시 수행 능력을 향상시키기 위해 인스트럭트 데이터셋(instruct dataset)으로 추가 파인튜닝(fine-tuning)되었습니다.

- **Performance Highlights**: 히브리어 언어 모델 평가를 위한 새로운 벤치마크 세트가 도입되었으며, 여기에는 질문 응답(Question Answering), 감정 분석(Sentiment Analysis), Winograd Schema Challenge, 번역(Translation), 요약(Summarization) 등의 다양한 작업이 포함됩니다. DictaLM2.0와 DictaLM2.0-Instruct 모델은 이들 작업에서 최첨단 성능을 달성했습니다.



### Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps (https://arxiv.org/abs/2407.07071)
Comments:
          The source code is available at this https URL

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 문서를 요약하거나 질문에 답할 때 문맥적 환각(contextual hallucinations)을 감지하는 간단한 방법을 설명합니다. 문맥적 환각이란 모델이 주어진 문맥 정보를 무시하고 잘못된 세부 정보를 생성하는 현상을 말합니다. 이 논문에서는 주어진 문맥에 대한 주의(attention) 가중치와 새로 생성된 토큰에 대한 주의 가중치의 비율인 'lookback ratio'를 특징으로 하는 간단한 환각 감지 모델을 제안합니다.

- **Technical Details**: 이 연구는 주로 'lookback ratio'라는 최신 특징을 사용합니다. 'lookback ratio'는 각 attention head에서 주어진 문맥 정보와 새로 생성된 토큰 간의 주의 가중치 비율을 계산하여 만들어집니다. 이를 기반으로 하는 선형 분류기를 훈련하여 문맥적 환각을 감지합니다. 'Lookback Lens'라고 불리는 이 모델은 LLM의 은닉 상태나 텍스트 기반 함의(Entailment) 모델을 사용하는 복잡한 감지기와 동등한 성능을 보여줍니다. 또한, 이 감지기를 디코딩 중에 통합하여 환각을 줄이는 'Lookback Lens Guided Decoding' 전략도 제안합니다.

- **Performance Highlights**: 새로 제안된 'Lookback Lens'는 다양한 작업과 모델에 적용할 수 있으며, XSum 요약 작업에서 환각을 9.6% 줄이는 성과를 보였습니다. 또한, 한 모델에서 훈련된 감지기를 다른 모델로 전이(transfer)할 수 있습니다. 예를 들어, LLaMA-2-7B-Chat 모델에서 훈련된 감지기를 LLaMA-2-13B-Chat 모델에 재훈련 없이 적용하여도 환각을 3.2% 줄일 수 있었습니다.



### Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligenc (https://arxiv.org/abs/2407.07061)
Comments:
          work in progress

- **What's New**: 현재의 멀티 에이전트 프레임워크가 갖고 있는 에코시스템 종속과 단일 장치 시뮬레이션 같은 한계를 극복하기 위해 인터넷과 유사한 환경에서 다양한 에이전트들이 협력할 수 있도록 지원하는 'Internet of Agents (IoA)'를 제안했습니다. IoA는 동적 팀 구성과 대화 흐름 제어 등 다양한 기능을 제공하며, 유연하고 확장 가능한 플랫폼을 구축합니다.

- **Technical Details**: IoA는 에이전트 통합 프로토콜과 인스턴트 메시징과 유사한 구조를 도입하여 다양한 외부 에이전트를 통합하고 효과적으로 협력할 수 있게 합니다. 서버와 클라이언트의 층별 아키텍처로 설계되어 있으며, 각각의 층에 상호작용, 데이터, 기반 구조를 구성합니다. 서버는 에이전트 등록, 발견, 메시지 라우팅을 관리하며 클라이언트는 개별 에이전트를 위한 통신 기능을 제공합니다.

- **Performance Highlights**: 일반 비서 업무, 체화된 AI 작업 및 검색 기반 생성 벤치마크에서 IoA는 최첨단 기준선을 일관되게 능가합니다. 특히, AutoGPT와 Open Interpreter를 통합하여 개방형 도메인 작업 평가에서 66%에서 76%의 승률을 기록하였고, GAIA 벤치마크와 RAG 질문-답변 영역에서도 뛰어난 성능을 보였습니다. GPT-3.5 기반 구현에서는 GPT-4와 유사하거나 초과하는 성과를 거두었습니다.



### Decoding Climate Disagreement: A Graph Neural Network-Based Approach to Understanding Social Media Dynamics (https://arxiv.org/abs/2407.07038)
- **What's New**: 이번 연구에서는 Graph Attention Networks (GATs)와 자연어 처리(NLP) 기법을 통합하여 Reddit 댓글-답글 쌍 내에서의 의견 불일치를 정확하게 식별하고 예측하는 혁신적인 방법인 ClimateSent-GAT 모델을 소개합니다. 이 모델은 불일치를 '동의(agree)', '불일치(disagree)', '중립(neutral)'의 세 가지 범주로 분류합니다. Reddit 댓글-답글 쌍의 고유한 그래프 구조를 활용함으로써, 복잡한 상호작용 패턴과 감정 역학을 포착하여 기존 벤치마크를 크게 능가합니다.

- **Technical Details**: 이 연구는 DEBAGREEMENT 데이터셋의 Climate subset을 사용했으며, 이 데이터셋은 PushShift API를 사용하여 다양한 subreddit에서 데이터를 수집함으로써 구성되었습니다. 특히, r/climate subreddit에서 2015년 1월부터 2021년 5월까지의 모든 게시물 및 댓글이 포함되었습니다. 각 댓글 길이는 10에서 100 단어 범위에 있으며, comment-reply 상호작용은 crowd-workers에 의해 'agree', 'disagree', 'neutral'로 레이블링되었습니다. 본 연구의 목표는 텍스트와 감정 내용을 활용하는 동시에 소셜 미디어 상에서의 기후 담론의 복잡한 상호작용을 캡처하는 Graph Attention Networks (GATs)를 적용하는 것입니다.

- **Performance Highlights**: ClimateSent-GAT 모델은 기존 모델들에 비해 뛰어난 성능을 보여주었습니다. 예를 들어, 기후 변화 관련 댓글-답글 쌍 내에서 불일치를 감지하는 능력이 현저히 향상되었습니다. 이는 감정 및 상호작용 역학을 포착하는 능력 덕분입니다. 특히, 이 모델은 기후과학 커뮤니케이션에서 오해를 유발하는 패턴을 밝히는 데 중요한 인사이트를 제공합니다. 이러한 인사이트는 정책 입안자와 교육자에게도 큰 도움이 될 수 있습니다.



### Vision-and-Language Navigation Today and Tomorrow: A Survey in the Era of Foundation Models (https://arxiv.org/abs/2407.07035)
Comments:
          Authors contributed equally to this work, and supervisors contributed equal advising to this work

- **What's New**: 최근 몇 년간 Vision-and-Language Navigation (VLN)에 대한 연구가 크게 주목받고 있으며, 많은 접근법이 개발되었습니다. 본 서베이는 VLN 연구의 도전 과제와 제안된 방법을 포괄적으로 리뷰하며, 특히 foundation models을 활용하여 VLN 문제를 해결하는 방법과 향후 기회를 강조합니다. 본 논문은 VLN 연구의 milestones와 foundation models의 잠재적 역할을 탐구하고, VLN의 다양한 도전 과제와 솔루션을 체계적으로 정리하여 foundation model 연구자들에게 도움이 되는 자원을 제공합니다.

- **Technical Details**: 본 서베이는 embodied AI 에이전트가 인간의 명령을 따르고, 3D 환경을 탐험하며, 다양한 형태의 모호성 아래에서 상황에 맞는 커뮤니케이션을 해야 하는 Vision-and-Language Navigation (VLN) 작업을 다룹니다. foundation models는 인간의 명령을 해석하고, 시각적 환경을 이해하며, 행동을 계획하는 에이전트 모델의 백본으로 사용됩니다. 이 프레임워크를 통해 에이전트는 시각적 환경을 인지하고, 인간의 명령을 받아들이며, 세계와 인간의 대표성을 바탕으로 행동을 계획하여 효과적으로 내비게이션 작업을 완료합니다.

- **Performance Highlights**: foundation models를 통합하여 VLN 작업에 적용한 결과, 매력적인 성능 향상이 나타났습니다. 이 모델들은 크로스 도메인 일반화, 멀티모달 이해 및 추론에서 탁월한 능력을 보였습니다. 특히, 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)을 사용함으로써, task planning, commonsense reasoning, 및 현실 환경에 대한 일반화에서 새로운 연구 기회가 열렸습니다.



### Using Large Language Models for Generating Smart Contracts for Health Insurance from Textual Policies (https://arxiv.org/abs/2407.07019)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 텍스트 기반 건강 보험 정책을 자동화하는 애플리케이션 코드를 생성하는 방식을 탐구합니다. 특히 블록체인 기반 스마트 계약을 목표로 함으로써 불변성, 검증 가능성, 확장성, 신뢰가 필요 없는 환경을 제공합니다. 다양한 기술적 세부사항에 따라 텍스트 요약, 선언적 결정 논리, 스마트 계약 코드와 유닛 테스트를 생성하는 방법론을 사용하여 실험했습니다.

- **Technical Details**: 저희 방법론은 LLM이 (1) 텍스트 요약, (2) 선언적 결정 논리 및 (3) 스마트 계약 코드 생성이라는 점진적인 기술적 세부사항 출력을 생성하도록 설계되었습니다. 건강 보험 정책을 형식화하기 위해 자주 사용되는 선언적 언어는 블록체인 상에서 실행하는 것이 어렵기 때문에, 스마트 계약을 사용하여 프로세스를 직접 자동화합니다. 연구에서 GPT-3.5 Turbo, GPT-3.5 Turbo 16K, GPT-4, GPT-4 Turbo, CodeLLaMA와 같은 모델들을 사용하여 건강 보험 정책의 세 가지 시나리오를 평가했습니다.

- **Performance Highlights**: 텍스트 요약(task 1)에서는 LLM이 우수한 성능을 보였습니다. 반면, 선언적 결정 논리(task 2)와 스마트 계약 코드(task 3)에서는 초안으로 유용하지만, 인간의 감독이 필요하여 복잡한 시나리오에서는 잘 작동하지 않는 경우도 있었습니다. 목표 언어의 인기도가 출력 품질에 영향을 미쳤으며, 복잡한 시나리오는 여전히 어려운 과제로 남아 있습니다. 하지만 텍스트 기반 프로세스 설명을 스마트 계약으로 번역하는 데 있어 LLM의 잠재력을 확인할 수 있었습니다.



### Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning (https://arxiv.org/abs/2407.07011)
Comments:
          9 pages, 7 figures

- **What's New**: 대형 언어 모델(LLMs)은 컨텍스트 학습(ICL)을 통해 복잡한 작업을 수행하는 데 있어서 뛰어난 능력을 보여주고 있습니다. 이 연구는 ICL 환경에서 유도 헤드(induction heads)의 역할을 탐구합니다. Llama-3-8B 및 InternLM2-20B 모델을 추상 패턴 인식과 NLP 작업에 대해 분석한 결과, 유도 헤드를 최소한으로 제거해도 ICL 성능이 최대 약 32%까지 감소하며, 이는 추상 패턴 인식 작업의 성능을 랜덤 수준으로 낮춥니다.

- **Technical Details**: 연구는 두 가지 주요 작업(추상 패턴 인식 및 인기 있는 NLP 작업)에서 Llama-3-8B와 InternLM2-20B 모델을 분석했습니다. 모든 주의(attention) 헤드에 대해 접두사 일치(prefix matching) 및 복사(copying) 점수를 계산한 후, 가장 높은 점수를 가진 헤드의 1%와 3%를 제거하는 실험을 수행했습니다. 또한 주의 차단(attention knockout)을 사용하여 특정 유도 패턴을 비활성화하는 세밀한 증거를 제시했습니다. 결과는 유도 헤드의 제거가 무작위 헤드의 제거보다 훨씬 더 큰 ICL 성능 저하를 초래했습니다.

- **Performance Highlights**: 유도 헤드의 제거는 ICL 성능을 크게 감소시켰습니다. 추상 패턴 인식 작업에서 성능은 최대 약 32%까지 감소했고, NLP 작업에서는 예제의 도움을 받을 수 있는 모델의 능력이 크게 저하되어 few-shot ICL 성능이 zero-shot 프롬프트에 가까워졌습니다. 주의 패턴을 차단했을 때 성능 저하가 더 크게 나타나, 유도 헤드가 ICL에 중요한 역할을 하고 있음을 확인했습니다.



### Empirical analysis of Biding Precedent efficiency in the Brazilian Supreme Court via Similar Case Retrieva (https://arxiv.org/abs/2407.07004)
Comments:
          54 pages, 22 figures

- **What's New**: 이번 연구는 브라질 연방 대법원에서 반복적으로 발생하는 소송을 줄이고자 도입된 '구속력 있는 판례(Súmula Vinculante)'의 효율성을 분석합니다. 이러한 판례들이 새로운 소송을 야기시키는 문제점을 제기하며, 구체적으로 다섯 가지 판례(11, 14, 17, 26, 37)의 법적 영향력을 평가합니다.

- **Technical Details**: 이 연구는 '유사 사례 검색(Similar Case Retrieval)' 기술을 사용하여 판례들이 도입되기 전과 후의 대법원 판결을 비교합니다. 이를 위해 다양한 자연어 처리(NLP) 기법(TF-IDF, LSTM, BERT, regex)을 활용하여 유사 사례를 검색하고, 그 결과를 법적 분석에 활용합니다.

- **Performance Highlights**: 깊은 학습 모델(deep learning models)은 구체적인 '유사 사례 검색' 과제에서 성능이 저조한 것으로 나타났습니다. 반면, 다른 방법들은 법률 문서 검색에서 더 나은 성능을 보였습니다. 또한, 이러한 판례들이 반복적인 소송 대응에서 실패한 이유는 다양하며, 단일 원인으로 규명할 수 없는 것으로 분석되었습니다.



### Segment-Based Interactive Machine Translation for Pre-trained Models (https://arxiv.org/abs/2407.06990)
Comments:
          10 pages, 4 figures

- **What's New**: 이 연구는 mBART와 mT5 같은 사전 훈련된 대형 언어 모델(LLM)을 상호작용 기계 번역(IMT) 환경에서 활용 가능성을 탐구합니다. 사용자의 피드백을 사용하여 상호작용 방식으로 번역을 개선하는 시스템을 구축하고, 이를 통해 완벽한 번역을 생성합니다.

- **Technical Details**: mBART (multilingual Bidirectional and Auto-Regressive Transformer)와 mT5 (multilingual Text-to-Text Transfer Transformer)를 주로 활용하여 실험을 수행하였으며, 세그먼트 기반 프로토콜을 적용한 IMT 시스템을 설계했습니다. 이 시스템은 초기 가설을 생성한 후 사용자의 피드백을 반영하여 번역을 개선하는 반복 과정을 통해 최종 번역을 생성합니다.

- **Performance Highlights**: mBART 모델은 주어진 벤치마크 데이터셋에서 상태-최신 (SoTA) 기계 번역 모델에 비견되는 성능을 보였습니다. 단, 세그먼트 간 내용을 일반화하는 데 어려움을 겪으면서 전체 번역 완료까지 소요되는 반복 횟수가 SoTA 모델보다 많았습니다.



### Self-Recognition in Language Models (https://arxiv.org/abs/2407.06946)
Comments:
          Code to reproduce experiments and replicate findings is made available at this https URL

- **What's New**: 점점 더 많은 응용 프로그램들이 폐쇄형 소스 언어 모델(LM; Language Models)에 의존하는 현상이 증가하고 있습니다. 이러한 의존성은 LM이 자기 인식 능력을 개발할 경우 새로운 보안 위험을 초래할 수 있습니다. 이에 영감을 받아, 인간 신원 검사 방법을 바탕으로 모델 생성 '보안 질문'을 통해 LM의 자기 인식 능력을 평가하는 새로운 접근 방식을 제안합니다. 이 테스트는 외부에서 시행할 수 있어, 모델의 내부 매개변수나 출력 확률에 접근할 필요가 없습니다.

- **Technical Details**: 이 방법은 세 단계로 진행됩니다. 먼저, LM에게 자기를 인식할 수 있는 답변을 도출할 수 있는 질문을 생성하도록 지시합니다. 그 후, 생성된 질문을 여러 LM 패널에게 물어보아 답변을 수집합니다. 마지막으로, 또 다른 LM에게 질문과 답변을 제시하여 자신의 답변을 선택하도록 지시합니다. 이를 통해 LM이 효과적인 '보안 질문'을 생성할 수 있는지 평가합니다.

- **Performance Highlights**: 10개의 최첨단 공개 및 비공개 LM을 대상으로 광범위한 실험을 실시한 결과, 일반적이거나 일관된 자기 인식의 경험적 증거는 발견되지 않았습니다. 오히려 LM들은 대안 중에서 '가장 좋은' 답을 선택하는 경향이 있음을 발견했습니다. 또한, 모델 간에 최고의 답변을 생성한 모델에 대한 선호도가 일관되게 나타났습니다. 추가로, 여러 선택지 설정에서 LM 의사 결정에 위치 편향(position bias)이 미치는 영향을 발견하여 LM 벤치마크에 중요한 함의를 제공할 수 있습니다.



### Raply: A profanity-mitigated rap generator (https://arxiv.org/abs/2407.06941)
- **What's New**: 이번 연구에서는 복잡한 운율을 만들면서도 의미 있는 가사를 생성하는 Raply 모델을 제안했습니다. Raply는 GPT-2 모델을 미세 조정하여 랩 스타일의 운율화된 텍스트를 생성할 수 있습니다. 특히, Mitislurs라는 욕설 완화 데이터셋으로 모델을 미세 조정하여 콘텐츠의 공격성을 줄였습니다.

- **Technical Details**: Raply 모델은 GPT-2를 기반으로 하며, Mitislurs라는 새로 구축한 욕설 완화 말뭉치로 추가 학습되었습니다. 모델의 출력은 크게 두 가지 기준으로 평가되었습니다: 1) 운율 밀도 지표를 기반으로 한 운율성, 2) 영어 욕설 목록을 사용한 욕설 함유 정도. 랩 가사 생성 문제를 자동 완성 태스크로 공식화하여, 사용자 입력을 확장한 결과를 출력합니다.

- **Performance Highlights**: Raply 모델은 랩 스타일에 맞는 운율화된 텍스트를 생성하는 데 뛰어난 성능을 보였으며, 욕설 함량을 효과적으로 줄였습니다. 이는 기존의 연구들이 다루지 않은 새로운 접근 방식으로, 욕설 최소화와 맞춤형 운율 생성의 두 가지 목표를 모두 달성했습니다.



### Who is better at math, Jenny or Jingzhen? Uncovering Stereotypes in Large Language Models (https://arxiv.org/abs/2407.06917)
- **What's New**: 이번 연구는 기존 언어 모델(LLMs)이 특정한 성별, 인종 그룹을 포함한 편향된 고정관념을 어떻게 확산하고 증폭시키는지 분석하기 위해 새로운 데이터셋 'GlobalBias'를 소개했습니다. 이 데이터셋은 다양한 성별-인종 조합을 포함하여 총 876,000개의 문장을 구성하며, 40개의 조합을 다루고 있습니다. 이를 통해 전 세계적으로 편향된 고정관념의 영향을 심도 있게 연구할 수 있도록 도와줍니다.

- **Technical Details**: GlobalBias 데이터셋은 기존의 성별 및 인종에 따른 편향 연구의 한계를 극복하고자 설계되었습니다. 40개의 성별-인종 그룹(예: 영어를 사용하는 여성 또는 중국어를 사용하는 여성)과 고정관념을 나타내는 설명자(예: ‘수학을 잘함’)를 사용하여 문장을 구성하였습니다. 이를 통해 다양한 LLM의 내적 표현(perplexity)을 직접적으로 탐색하였습니다. 모델 출력을 평가하기 위해 주어진 이름을 기반으로 특성을 생성하고, 고정관념의 발현 빈도를 분석하였습니다.

- **Performance Highlights**: 연구 결과, 고정관념과 관련된 인구 집단은 모델의 예측과 출력에서 일관되게 나타났습니다. 특히, 더 큰 규모의 모델이 더 높은 수준의 고정관념을 포함한 출력을 내는 경향이 있음을 확인했습니다. 이는 모델이 명시적으로 고정관념을 피하도록 명령하더라도 발생하는 현상입니다. 또한, 모형의 내부 표현과 출력의 고정관념이 일관되게 유지되는 것을 발견했습니다.



### Divine LLaMAs: Bias, Stereotypes, Stigmatization, and Emotion Representation of Religion in Large Language Models (https://arxiv.org/abs/2407.06908)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 감정 속성을 통해 종교를 어떻게 대표하는지를 탐구했습니다. 미국과 유럽의 주요 종교들은 더 정교하게 모델링된 반면, 힌두교와 불교 같은 동양 종교는 강한 고정관념이 적용되었습니다. 유대교와 이슬람교는 낙인 찍히는 경향이 있었으며, 모델의 거부 반응도 높았습니다.

- **Technical Details**: 이번 연구는 International Survey on Emotion Antecedents and Reactions (ISEAR) 데이터셋을 사용하여 다양한 종교적 그룹에 대해 감정을 귀속시키는 방식으로 LLMs의 공평성을 평가했습니다. Llama2, Llama3, GPT-4, Mistral-7b 등의 첨단 모델들을 대상으로 했으며, 각 모델에 종교적 인구 통계에 기반한 페르소나(여러 종교 그룹)를 할당하여 감정을 분류했습니다.

- **Performance Highlights**: 주요 종교(미국과 유럽)의 감정 표현이 더 복잡하고 구체적인 반면, 동양 종교와 유대교, 이슬람교는 고정관념과 낙인이 강했습니다. 이는 NLP 문헌에서 종교에 대한 부족한 연구와 LLMs 내의 문화적 편견 때문입니다.



### Measuring Sustainability Intention of ESG Fund Disclosure using Few-Shot Learning (https://arxiv.org/abs/2407.06893)
Comments:
          This paper was presented at 'AI applications in ESG Conference' at IIM Bangalore, India (Nov, 2023)

- **What's New**: 이번 연구에서는 지속 가능성 펀드의 투자 전략에서 명시하는 환경, 사회, 지배 구조(ESG) 초점을 분석하기 위해 새로운 방법을 제안하고 있습니다. 기존에는 이런 ESG 주장들을 규제하는 규칙이 없었으나 텍스트 공개를 통해 주장들을 확인해야 하는 어려움이 있었습니다. 본 연구는 펀드 설명서(prospectus)의 언어 구체성과 투명성을 분류하고 점수화하는 독특한 방법과 시스템을 제안합니다.

- **Technical Details**: 본 연구는 few-shot learners(학습자)를 사용하여 특정, 모호 및 일반적인 지속 가능한 투자 관련 언어를 식별합니다. 이를 위해 language score(언어 점수)를 결정하는 비율 메트릭과 랭크 제품 및 지속 가능성 주장을 정량화하는 평가 시스템을 구축했습니다. 또한 Hugging Face에 cc-by-nc-sa-4.0 라이선스로 1K 이상의 ESG 텍스트 문서를 포함하는 수동 주석 품질 훈련 데이터를 공개했습니다. 연구는 few-shot 파인튜닝이 Llama-13B, GPT 3.5 Turbo와 같은 zero-shot 모델보다 도메인 특화 작업에서 더 높은 정확성(precision), 재현율(recall), F1 메트릭에서 약 30% 이상 우수한 성과를 나타냄을 보여주었습니다.

- **Performance Highlights**: few-shot 파인튜닝 기술은 미지의 ESG 언어(test set)에서 zero-shot 모델보다 절대 비율로 약 30% 이상의 높은 정밀도와 재현율, F1 지표를 기록했습니다. 연구 결과는 규제 기관, 투자자 및 자문가들이 ESG 펀드를 조사하거나 선별할 때 인지적 부담을 줄이는 데 도움을 줄 수 있습니다.



### ChatGPT Doesn't Trust Chargers Fans: Guardrail Sensitivity in Contex (https://arxiv.org/abs/2407.06866)
- **What’s New**: 이 논문은 주로 실무에 적용된 언어 모델들의 편향성에 대해 다루고 있지만, 현재까지는 '가드레일(guardrails)'의 편향성에 대해 집중적으로 연구하지 않았습니다. 가드레일은 모델 응답을 제한하는 역할을 하며, 특정 사용자 정보가 포함된 요청에 대해 모델이 거부 반응을 보이는지에 대한 연구를 진행했습니다. 생년월일, 성별, 인종 등의 사용자 정보를 제공하여 다양한 바이오그래피를 생성하고, 이를 통해 가드레일 편향성을 실험했습니다.

- **Technical Details**: 논문에서는 GPT-3.5 모델을 대상으로 실험을 진행했습니다. 사용자의 성별, 나이, 인종 등의 정보를 명시한 바이오그래피를 제시하고, 검열된 정보나 불법 정보에 대한 요청을 포함하여 다양한 타입의 요청을 실험했습니다. 또한 실험 과정에서 약 22만 5천 개의 요청을 생성 및 분석하였고, GPT-4o를 활용한 키워드 분류기와 직접 라벨링을 통해 응답을 거부하는지 여부를 평가했습니다.

- **Performance Highlights**: 연구 결과, 어린 나이의 사용자, 여성, 아시아계 미국인 페르소나가 검열된 요청을 할 때 더 자주 가드레일을 트리거하는 것을 발견했습니다. 정치적으로 민감한 요청에 대해서는 정치적 성향에 따라 거부 확률이 높아지는 경향을 보였습니다. 또한 특정 스포츠 팬덤과 같은 비일상적인 그룹 소속이 가드레일 민감성에도 영향을 미치는 것을 확인했습니다. 예를 들어, 보수적인 팬층을 가진 NFL 팀의 팬인 경우 더 높은 거부율을 보였습니다.



### Safe-Embed: Unveiling the Safety-Critical Knowledge of Sentence Encoders (https://arxiv.org/abs/2407.06851)
Comments:
          ACL 2024 KnowledgeableLMs workshop paper

- **What's New**: 이 연구는 기존의 분류 모델(classification models) 대신 문장 인코더(sentence encoders)를 사용하여 안전하지 않은 프롬프트와 안전한 프롬프트를 구별하고, 다양한 안전하지 않은 프롬프트를 안전성 분류체계(safety taxonomy)로 분류하는 방법을 탐구합니다. 이를 위해 새로운 쌍별(pairwise) 데이터셋과 Categorical Purity (CP) 메트릭을 도입했습니다. 이 연구 결과는 문장 인코더의 효율성과 한계를 밝히며, 좀 더 강력한 안전성 감지기로서의 개선 방향을 제안합니다.

- **Technical Details**: 문장 인코더가 안전한 프롬프트와 안전하지 않은 프롬프트를 구별할 수 있는 지 여부와 이를 분류할 수 있는 능력을 검토합니다. 이를 위해 안전-챌린징(Safety-Challenging)과 안전-대조(Safety-Contrast)라는 새로운 데이터셋을 생성했습니다. 또한, 기존에 사용된 LLM 기반 및 API 기반 분류 모델의 한계를 극복하고자 유사성 검색(similarity search)을 통해 이전에 식별된 안전하지 않은 프롬프트의 임베딩을 저장하고 새로운 프롬프트가 유사성을 초과하면 이를 필터링하는 방식을 사용했습니다.

- **Performance Highlights**: 연구 결과, 기존 문장 인코더가 특정 문맥 이해에 한계가 있음을 발견했습니다. 예를 들어, 인종 및 성별에 기반한 명시적 차별을 포함하지 않는 프롬프트 식별에 어려움을 겪었습니다. 새로운 데이터셋과 메트릭을 통해 일부 문장 인코더는 스테레오타입 및 개인정보 관련 주제를 효과적으로 처리하지만, 다양한 문맥 이해에는 한계가 있다는 것을 보여줍니다. 이는 문장 인코더를 보다 강력한 안전성 감지기로 개선하기 위한 방향을 제시합니다.



### Using Pretrained Large Language Model with Prompt Engineering to Answer Biomedical Questions (https://arxiv.org/abs/2407.06779)
Comments:
          Submitted to Conference and Labs of the Evaluation Forum (CLEF) 2024 CEUR-WS

- **What's New**: 우리 팀은 BioASQ 2024 Task12b와 Synergy 작업에 참여하여 PubMed 데이터베이스에서 관련 문서와 단편을 검색하고 정확하고 이상적인 답변을 생성할 수 있는 시스템을 구축했습니다. 우리는 사전 학습된 대형 언어 모델(LLM)을 기반으로 한 두 단계 정보 검색 및 질문 응답 시스템을 제안하며, 주로 LLM 프롬프트 엔지니어링과 응답 후처리에 중점을 두고 있습니다.

- **Technical Details**: 우리는 모델링 파이프라인, 프롬프트 엔지니어링 전략 및 Synergy와 Task12b 작업에서 다양한 LLM 모델과의 실험 결과를 논의합니다. 우리의 시스템은 LLM 모델을 사용하여 질문에서 키워드를 추출하고 PubMed 쿼리를 작성하여 PubMed 데이터베이스에서 문서를 검색한 다음 문서에서 관련 단편을 찾기 위해 문장 임베딩을 사용합니다. 질문 응답을 위해 우리는 단편을 컨텍스트로 사용하고 몇 샷 예제 프롬프트를 구성하여 LLM이 원하는 형식으로 답변을 생성하도록 유도합니다.

- **Performance Highlights**: 우리의 최고 성능 시스템은 문서 검색에서 0.14 MAP 점수, 단편 검색에서 0.05 MAP 점수, 예/아니오 질문에서 0.96 F1 점수, 사실 질문에서 0.38 MRR 점수, 목록 질문에서 0.50 F1 점수를 달성했습니다.



### Consistent Document-Level Relation Extraction via Counterfactuals (https://arxiv.org/abs/2407.06699)
- **What's New**: 문서 수준의 관계 추출(Relation Extraction, RE) 모델을 훈련하고 평가하는 데 사용되는 데이터셋에 대한 새로운 접근법을 제안합니다. 본 연구는 실세계 데이터를 사용한 RE 모델에서 발생하는 사실적 편향(factual biases)을 해결하기 위한 대책으로 'CovEReD'라는 반사실 데이터(counterfactual data) 생성 방법을 소개합니다. 이를 통해 문서 수준 RE 데이터셋에서 엔티티 교체(Entity Replacement)를 사용하여 반사실 데이터를 생성합니다.

- **Technical Details**: CovEReD는 문서 수준 관계 추출 데이터셋에서 반사실 데이터를 생성하는 기법입니다. 이는 특정 엔티티를 교체하여 최소한의 사실적 정렬(minimal factual alignment)을 유지하면서 관계 삼중항(triples)을 포함한 텍스트를 생성합니다. 문서 수준에서 CovEReD를 적용하여 여러 엔티티 언급 및 다중 교체를 동시에 처리할 수 있습니다. 특히, 실세계 데이터에서의 편향을 제거하기 위해 유사한 관계 지도와 문맥 조각을 가진 엔티티들 간에 교체를 수행합니다.

- **Performance Highlights**: CovEReD를 이용하여 생성된 반사실 데이터셋 Re-DocRED-CF를 사용해 훈련된 RE 모델은 일관성을 유지하면서 사실적 데이터에 대한 정확도에 거의 영향을 미치지 않습니다. 이는 기존의 문서 수준 관계 추출 모델이 사실적 데이터에 대해서는 강력하지만, 비사실적 데이터에 대해서는 취약하다는 점을 해결할 수 있음을 보여줍니다. Re-DocRED와 Re-DocRED-CF 양쪽에서 훈련된 모델은 높은 일관성을 나타냈습니다.



### Mixture-of-Modules: Reinventing Transformers as Dynamic Assemblies of Modules (https://arxiv.org/abs/2407.06677)
- **What's New**: Transformers 구조의 깊이 순서를 깨고, 모든 토큰(token)에 대해 특정 모듈을 선택하여 처리하는 MoM(Mixture-of-Modules)을 소개합니다. MoM은 깊이와 상관없이 어느 층이든 필요한 처리를 할 수 있다는 직관에서 출발한 혁신적인 아키텍처입니다.

- **Technical Details**: MoM은 다중 헤드 어텐션(multi-head attention, MHA)과 피드포워드 네트워크(feed-forward network, FFN) 모듈로 구성됩니다. 두 라우터가 각각의 토큰을 처리할 때마다 모듈을 선택하여 동적으로 계산 그래프를 구성합니다. 이 접근법은 OpenWebText를 사용하여 미리 학습되었으며, GLUE와 XSUM 벤치마크에서 테스트되었습니다.

- **Performance Highlights**: MoM은 GLUE와 XSUM 벤치마크에서 vanilla Transformers보다 꾸준히 우수한 성능을 보였습니다. 특히, 동일한 매개변수 예산을 사용할 때, MoM-large는 GPT-2-large에 비해 38% 더 깊이 있는 계산 그래프를 가능하게 하여 GLUE에서 1.4, XSUM에서 1의 절대적인 성능 향상을 보였습니다. 또한, MoM-large는더 적은 깊이로 TFLOPs를 16% 감소시키고 메모리 사용량을 42%까지 감소시키면서도 유사한 성능을 유지할 수 있는 효율적인 구조를 제공합니다.



### SoftDedup: an Efficient Data Reweighting Method for Speeding Up Language Model Pre-training (https://arxiv.org/abs/2407.06654)
Comments:
          12 pages, 7 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 학습 데이터셋에서 중복 데이터를 다루기 위한 새로운 소프트 데이터 중복 제거(soft deduplication) 방법을 제안합니다. 일반적으로 중복 데이터를 감지하고 제거하는 현재의 방식을 대체하여 데이터의 무결성을 유지하면서도 공통성이 높은 데이터의 샘플링 가중치를 선택적으로 감소시키는 방식입니다.

- **Technical Details**: 제안된 방법의 중심에는 '데이터 공통성(data commonness)'이라는 개념이 있습니다. 이는 n-그램 모델(n-gram model)을 사용하여 샘플의 발생 확률을 측정하여 중복 정도를 정량화하는 새로운 지표입니다. 공통성이 높은 샘플은 낮은 샘플링 가중치를 부여받고, 공통성이 낮은 샘플은 높은 가중치를 부여받습니다. 이를 통해 주요 데이터를 잃지 않으면서 중복 데이터를 효과적으로 관리할 수 있습니다.

- **Performance Highlights**: 실증 분석 결과, 이 방법은 최소 26% 적은 학습 단계로 기본 성능을 달성하면서 훈련 효율성을 크게 개선하였습니다. 또한 동일한 학습 기간 동안 평균 몇 샷 다운스트림 정확도가 1.77% 향상되었습니다. 특히, 철저히 중복 제거된 데이터셋에서도 일관되게 성능 향상을 유지하여 기존 방법을 보완하고 LLM의 사전 학습 프로세스의 표준 절차로 채택될 가능성을 보여줍니다.



### A Word Order Synchronization Metric for Evaluating Simultaneous Interpretation and Translation (https://arxiv.org/abs/2407.06650)
- **What's New**: 새로운 실시간 동시 통역(Simultaneous Interpretation, SI) 평가 메트릭 제안. 이 메트릭은 원본 언어와 번역 언어 사이의 단어 순서 동기화에 중점을 두며, 이는 특히 언어 간 어순 차이가 큰 경우에 중요.

- **Technical Details**: 제안된 평가 메트릭은 Rank Correlation Coefficients(순위 상관 계수)를 기반으로 cross-lingual pre-trained language models(다중 언어 사전 학습 언어 모델)을 활용하여 소스 언어와 타겟 언어 사이의 단어 순서를 동기화하는 방법을 측정합니다.

- **Performance Highlights**: 제안된 메트릭은 NAIST-SIC-Aligned 및 JNPC 코퍼스에서 유효성을 입증했으며, 비교적 긴 문장을 번역하는 경우 동시 통역이 오프라인 번역보다 소스 언어 입력에 더 나은 단어 순서 동기화를 보여주었습니다.



### NoisyAG-News: A Benchmark for Addressing Instance-Dependent Noise in Text Classification (https://arxiv.org/abs/2407.06579)
Comments:
          20 pages , 13 figure

- **What's New**: 최근 연구에서는 합성 라벨 노이즈에 집중하고 있지만, 실제 환경에서는 이러한 합성 노이즈가 제대로 작용하지 않을 수 있습니다. 이를 보완하기 위해 우리는 수작업으로 주석된 NoisyAG-News라는 새로운 벤치마크 데이터셋을 구축했습니다. 이 데이터셋을 통해 실제 환경에서의 노이즈 패턴을 보다 정확히 이해하고, 기존의 텍스트 분류 방법을 평가하고 개선할 수 있게 되었습니다.

- **Technical Details**: NoisyAG-News는 비전문가가 크라우드소싱을 통해 수집한 주석 데이터를 기반으로 만들어졌습니다. 다양한 배경, 선호도, 편향성을 가진 주석자들이 동일한 모호한 데이터에 대해 충돌하는 라벨을 달 수 있어, 이는 인스턴스 의존적 라벨 노이즈(IDN)로 이어집니다. 이를 통해 IDN의 특성을 분석하고, 합성 노이즈가 아닌 실제 노이즈를 반영하는 새로운 텍스트 분류 벤치마크를 제공합니다.

- **Performance Highlights**: 프리트레인 언어 모델(Pre-Trained Language Models, PLMs)과 노이즈 처리 기법을 사용해 NoisyAG-News 및 해당 합성 노이즈 데이터셋에서 학습 실험을 수행했습니다. 그 결과, PLMs는 합성 노이즈에 강인하지만 인스턴스 의존적 노이즈에 대해 고전하는 경향이 있음을 발견했습니다. 이러한 실제 노이즈 패턴은 새로운 도전 과제를 제시하며, 이로 인해 노이즈 처리 방법의 재평가가 필요합니다.



### Virtual Personas for Language Models via an Anthology of Backstories (https://arxiv.org/abs/2407.06576)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 특정 가상 인물에 맞게 조절하는 방법인 'Anthology'를 소개합니다. 이 방법은 'backstories'(개인 삶의 서사)를 활용하여, 실험 결과의 일관성과 신뢰성을 높이는 동시에 다양한 하위 계층을 더 잘 대표할 수 있도록 합니다.

- **Technical Details**: 'Anthology'는 개방형 서사를 이용하여 LLM을 특정 가상 인물로 조절합니다. 이를 통해 구체적인 인간 사용자와 일치하는 응답을 도출하는 방향으로 모델 반응을 유도합니다. 이 연구는 Pew Research Center의 American Trends Panel(ATP)의 세 가지 전국 대표 설문조사를 통해 수행되었습니다.

- **Performance Highlights**: Anthology 방법은 인간 응답자의 반응 분포와의 일치를 최대 18% 향상시키고, 일관성 지표에서 최대 27% 향상을 달성했습니다. 이를 통해 특정 가상 인물에 대한 모델 응답의 일관성과 신뢰성을 크게 개선할 수 있음을 확인했습니다.



### FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making (https://arxiv.org/abs/2407.06567)
Comments:
          LLM Applications, LLM Agents, Financial Technology, Quantitative Finance, Algorithmic Trading, Cognitive Science

- **What's New**: FinCon은 다양한 금융 작업을 위해 설계된 LLM 기반의 다중 에이전트 프레임워크로, 개념적 언어 강화를 통해 금융 투자 결정의 품질을 향상시키는 것을 목표로 합니다. 이는 실세계 투자 회사의 조직 구조에서 영감을 받아, 매니저-애널리스트 계층 구조를 활용하여 에이전트 간의 자연어 상호작용을 통해 효과적인 협업을 가능하게 합니다.

- **Technical Details**: FinCon은 다음과 같은 주요 기술적 요소를 포함합니다: 1) 매니저-애널리스트 통신 구조: 다양한 소스와 형식의 데이터를 해당 기능 분석 에이전트에게 배포하여 각 에이전트가 중요 투자 통찰력과 지표를 추출할 수 있게 합니다. 2) 리스크 컨트롤 컴포넌트: 정량적 금융의 조건부 가치 위험(Conditional Value at Risk, CVaR) 측정을 통해 에피소드 내 리스크 조정 및 에피소드 간 개념적 투자 통찰력 업데이트 기능을 통합했습니다.

- **Performance Highlights**: FinCon은 주식 거래와 포트폴리오 관리를 포함한 다양한 금융 작업에서 강력한 일반화 능력과 높은 성능을 보여주었습니다. 이는 기존 시스템과 달리 단일 에이전트의 정보 이해와 추출 능력에 의존하지 않고, 불필요한 동료 간 통신 비용을 줄이면서도 성과를 크게 향상시킬 수 있습니다.



### Combining Knowledge Graphs and Large Language Models (https://arxiv.org/abs/2407.06564)
- **What's New**: 요즘 자연어 처리(NLP) 분야에서 대형 언어 모델(LLM)이 중요한 역할을 하고 있으며, 채팅봇, 텍스트 생성, 언어 번역 등 다양한 AI 응용 프로그램의 성능을 크게 향상시켰습니다. 그러나 이러한 모델들은 환각(hallucinations) 및 분야별 지식 부족의 문제로 인해 실세계 태스크에서 성능 저하를 나타냅니다. 이러한 문제는 지식 그래프(KG)를 통합하여 효과적으로 완화할 수 있습니다. 이 연구는 LLM과 KG를 결합한 기술적 접근에 대한 28개의 논문을 수집, 분석, 비교하여 주요 트렌드, 혁신적 기술 및 공통적인 과제를 조명했습니다.

- **Technical Details**: LLM은 대규모 데이터셋과 계산 능력의 증가로 크게 발전했으며, Google BERT 및 T5, OpenAI GPT 시리즈 등이 주요 예입니다. 그러나 LLM의 지식은 훈련 시점의 파라미터에 고정되어 있어, 특정 도메인 지식 부족 및 결정 과정의 불명확성 문제를 지닙니다. 지식 그래프(KG)는 정보를 구조화된 형식으로 조직화하여 LLM의 성능을 향상시킬 수 있습니다. KG는 엔터티 간의 관계를 포착하는 직관적인 구조로, LLM이 사실을 보다 효과적으로 회상하도록 도와줍니다.

- **Performance Highlights**: 이 연구는 LLM과 KG를 결합하여 명명된 엔터티 인식 및 관계 분류와 같은 다양한 NLP 태스크에서 성능이 향상되는 것을 확인했습니다. 또한, 조사된 연구는 LLM이 KG 구축 과정에서 정보 추출기로 사용될 뿐만 아니라, 상호 보완적인 혜택을 어떻게 제공하는지 탐구했습니다. 예를 들어, BEAR라는 서비스 도메인의 KG는 ChatGPT를 사용해 비구조화된 데이터를 분석하여 기존 온톨로지를 채우는 방식을 사용했습니다.



### OffsetBias: Leveraging Debiased Data for Tuning Evaluators (https://arxiv.org/abs/2407.06551)
Comments:
          Work in Progress

- **What's New**: 이번 연구에서는 다양한 평가 모델(judge models)에서 나타나는 편향(biases)의 종류를 정성적으로 식별하고, 이를 교정(de-bias)하기 위한 데이터셋 구성 방법을 제안합니다. 또한, 각 편향 유형에 대한 수작업 테스트 케이스 모음을 제공하는 메타 평가 컬렉션 EvalBiasBench를 소개하고, 이와 연관된 OffsetBias 선호 데이터셋(preference dataset)도 제시합니다.

- **Technical Details**: 먼저, 평가 모델에서 발견된 편향을 총 여섯 가지 유형으로 분류했습니다. EvalBiasBench는 이러한 편향 유형을 평가하기 위해 설계된 수작업 테스트 케이스 모음집입니다. 또한, OffsetBias는 평가 모델의 편향을 줄이기 위해 설계된 선호 데이터셋으로, 이 데이터셋을 사용해 평가 모델을 미세조정(fine-tuning)하는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 데이터셋을 사용해 미세조정된 평가 모델은 기존 모델들에 비해 편향에 대한 강건함이 크게 향상되었으며, 다양한 평가 시나리오에서 성능이 전반적으로 개선되었습니다. 제안된 데이터셋과 미세조정된 평가 모델은 공개됩니다.



### Deciphering Assamese Vowel Harmony with Featural InfoWaveGAN (https://arxiv.org/abs/2407.06547)
Comments:
          to be included in the Interspeech Proceedings

- **What's New**: 이 연구는 기존의 텍스트 데이터 기반 음운론 학습 방법을 벗어나, Featural InfoWaveGAN 모델을 사용하여 음성 데이터만으로 장거리 모음 조화를 학습하는 가능성을 조사했습니다. 특히, 음운론적 회귀성과 단어 경계 모음 조화로 알려진 Assamese 언어에 초점을 맞추었습니다.

- **Technical Details**: 연구팀은 Featural InfoWaveGAN 모델이 Assamese 음운 전술의 복잡한 특징, 특히 회귀적 방향성을 가진 장거리 모음 조화를 학습할 수 있는 능력을 확인했습니다. 해당 모델은 음성 데이터의 특징을 학습하는 데 뛰어난 성능을 보였으며, 인간 언어 습득 과정 중 발생할 수 있는 비규칙적인 형태(비반복적 불법 형태)도 생성해냈습니다.

- **Performance Highlights**: Featural InfoWaveGAN 모델은 [+high, +ATR] 특징을 가진 모음이 새로운 항목에서 트리거로 선호되는 경향을 보였습니다. 이는 특정 음성 특징을 학습함을 나타냅니다. 더 많은 데이터와 제어를 통해 모델의 능력을 향상시킬 수 있을 것으로 보이며, 이는 학습의 보편성과 비교됩니다.



### LIONs: An Empirically Optimized Approach to Align Language Models (https://arxiv.org/abs/2407.06542)
- **What's New**: 이번 논문에서는 언어 모델의 교정(alignment)을 위해 제안된 다양한 알고리즘, 데이터셋, 훈련 파이프라인의 영향을 종합적으로 분석합니다. 많은 연구들이 기존에 발표된 방법들의 효용성을 주장하고 있지만, 그들이 사용하는 데이터셋과 하이퍼파라미터는 종종 비공개이거나 일관성이 부족합니다. 본 연구는 대표적인 교정 파이프라인을 재현하고 성능에 영향을 미치는 요인들을 분석합니다. 구체적으로는 세 단계의 훈련(지도 세밀 조정, 오프라인 선호 학습, 온라인 선호 학습)에 대해 분석합니다.

- **Technical Details**: 이 논문은 지루한 세밀 조정(Supervised Fine-Tuning, SFT), 오프라인 선호 학습(Offline Preference Learning), 온라인 선호 학습(Online Preference Learning)을 포함한 세 단계의 훈련 절차를 분석합니다. SFT 단계에서는 시퀀스 패킹(sequence packing)과 손실 마스킹(loss masking)을 적용하여 성능을 향상시켰습니다. 오프라인 선호 학습 단계에서는 DPO(Direct Preference Optimization) 기법을 사용해 선호 데이터셋의 크기를 증가시키고, 온라인 학습 단계에서는 추가로 DPO를 적용해 성능을 개선했습니다.

- **Performance Highlights**: Gemma-2b-base와 LLama-3-8b-base 모델을 Fine-Tuning 했으며, 최종적으로 우리의 모델들이 비공개 데이터와 알고리즘을 사용한 공식 인스트럭트 모델들보다 우수한 성능을 보였습니다. 본 연구는 다양한 벤치마크(Arena-Hard, AlpacaEval-2, MT-Bench, OpenLLM)에서 평가되었습니다.



### Enhancing Low-Resource NMT with a Multilingual Encoder and Knowledge Distillation: A Case Study (https://arxiv.org/abs/2407.06538)
Comments:
          Published at Seventh LoResMT Workshop at ACL 2024

- **What's New**: 이 연구는 기존 다중 언어 seq2seq 모델(mBART-50)이 포함하지 못한 인도 산하의 저자원 언어에 대한 번역을 개선하기 위해 새로운 프레임워크를 제안합니다. 이 프레임워크는 사전 훈련된 언어 모델과 knowledge distillation 기술을 결합하여 저자원 언어 번역에서의 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 프레임워크는 다중 언어 인코더 기반의 seq2seq 모델을 사용하며, 인코더는 XLM-R로 초기화되고 디코더는 새롭게 학습됩니다. 이를 기반으로 complementary knowledge distillation(CKD)을 적용하여 학습 불균형 문제를 해결합니다. 이 접근법은 XLM-MT+CKD라는 모델로 명명되었습니다.

- **Performance Highlights**: 세 가지 인도 언어 및 네 가지 번역 방향에서 기존 베이스라인보다 BLEU-4 및 chrF 점수가 현저하게 향상되었습니다. 또한, 인간 평가를 통해 모델의 유창성과 정확성이 확인되었습니다.



### Efficient and Accurate Memorable Conversation Model using DPO based on sLLM (https://arxiv.org/abs/2407.06537)
- **What's New**: 이 논문은 세션이 진행됨에 따라 메모리를 효율적으로 관리하고 대화 역사를 정확하게 반영하는 대화 모델을 제안합니다. 이를 위해 SFT(Split and Fill Technique), DPO(Direct Preference Optimization), DPO와 SFT를 결합한 모델 세 가지 방법론을 제시합니다.

- **Technical Details**: 본 연구는 연속된 다중 대화 세션 환경에서 작은 언어 모델(sLLM)을 사용하여 메모리를 관리하는 새로운 프레임워크를 제안합니다. DPO 알고리즘을 활용하여 대화의 인과관계를 반영한 정확한 메모리 정보를 추출하고 업데이트합니다. 또한, ChatGPT를 사용하여 DPO 훈련에 필요한 네거티브 샘플을 생성하고, 이를 통해 인과 정보를 효과적으로 포착하는 데이터를 구성했습니다.

- **Performance Highlights**: DPO 알고리즘을 사용한 모델은 메모리 정확도에서 BERTScore가 약 0.0591 만큼 개선되었으며, 메모리를 반영한 응답 비율도 증가했습니다. 응답 생성 성능은 유창성에서 4.292, 일관성에서 3.935, 일치성에서 2.896 만큼 향상되었습니다. 모델 크기가 더 작음에도 불구하고 매개 변수 크기가 두 배 이상 큰 모델보다 더 나은 성능을 보여줍니다.



### Towards Understanding Multi-Task Learning (Generalization) of LLMs via Detecting and Exploring Task-Specific Neurons (https://arxiv.org/abs/2407.06488)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)에서 학습 메커니즘을 신경망 관점에서 이해하려는 시도를 소개합니다. 특히, 기울기 기여도(gradient attribution) 방법을 사용해 특정 작업에 민감한 뉴런(task-specific neurons)을 검출하고, 이 뉴런들이 주어진 작업과 강한 상관관계를 가짐을 실험적으로 증명했습니다. 이를 통해 다중 작업 학습과 연속 학습의 주요 문제인 일반화(generalization)와 파괴적 망각(catastrophic forgetting)을 조사했습니다.

- **Technical Details**: 먼저 기울기 기여도 방법을 사용해 특정 작업에 관련된 뉴런을 찾아낸 후, 이 뉴런들을 비활성화하거나 미세 조정(fine-tuning)했습니다. 이 결과, LLMs에서 작업 관련 뉴런들이 존재하며 특정 작업과 높은 상관관계를 가짐을 확인했습니다. 이러한 작업 관련 뉴런을 바탕으로 일반화와 파괴적 망각의 문제를 분석했습니다. 특히, 뉴런의 중첩 정도가 일반화와 특수화 간의 상관관계를 강하게 나타냈습니다.

- **Performance Highlights**: 제안된 뉴런 수준 연속 미세 조정 방법(Neuron-level Continuous Fine-Tuning, NCFT)은 연속 학습 과정에서 현재 작업의 특정 뉴런만을 미세 조정하여 파괴적 망각을 효과적으로 완화시켰습니다. 실험 결과, 두 개의 연속 학습 벤치마크에서 제안된 방법의 효율성이 입증되었습니다.



### Interaction Matters: An Evaluation Framework for Interactive Dialogue Assessment on English Second Language Conversations (https://arxiv.org/abs/2407.06479)
- **What's New**: 이 논문에서는 영어를 제2언어로 사용하는 사람들(ESL 화자)을 위한 인터랙티브 대화 평가 프레임워크를 제안합니다. 이 프레임워크는 대화 상호작용성을 평가하기 위해 4개의 대화 수준 인터랙티비티 라벨과 17개의 마이크로 수준 기능을 수집합니다. 이를 통해 ESL 대화의 상호작용 품질에 미치는 미시적 기능의 영향을 분석하고, 이러한 기능들이 상호작용 품질에 강하게 연관되어 있음을 입증했습니다.

- **Technical Details**: 프레임워크는 두 가지 수준의 주석(annotation)을 포함합니다: (1) 주제 관리, 적정 톤 설정, 대화 시작 및 종료의 4가지 대화 수준 인터랙티비티 라벨, (2) 참고 단어(reference words) 및 백채널(utterance-level) 등 17가지 마이크로 수준 기능. 마이크로 수준 기능을 활용해 ESL 대화의 상호작용성을 예측하는 다양한 기계 학습 모델을 구성했습니다.

- **Performance Highlights**: 마이크로 수준 기능을 입력으로 사용하는 기계 학습 모델이 대화의 상호작용성을 예측하는 데 있어 베이스라인 BERT 모델보다 더 나은 성능을 보였습니다. 이는 미시적 기능들이 상호작용성 예측에서 강력한 예측력을 가진다는 것을 시사합니다. 본 연구의 프레임워크는 ESL 의사소통 평가에 유용하여 실제 영어 시험에 적용될 수 있는 잠재력을 지니고 있습니다.



### MUSE: Machine Unlearning Six-Way Evaluation for Language Models (https://arxiv.org/abs/2407.06460)
- **What's New**: 언어 모델(LMs)이 대규모 텍스트 데이터를 학습하는데, 여기에는 개인 정보 또는 저작권이 있는 콘텐츠가 포함될 수 있습니다. 데이터 소유자는 개인 정보 보호 또는 저작권 문제 때문에 자신들의 데이터를 삭제해 줄 것을 요구할 수 있습니다. 이를 해결하기 위해 제안된 MUSE(Machine Unlearning Six-Way Evaluation) 벤치마크는 기존의 데이터 제거 알고리즘의 실효성을 종합적으로 평가할 수 있습니다.

- **Technical Details**: MUSE는 언러닝된 모델이 가져야 할 6가지 다양한 속성을 평가합니다. 여기에는 (1) 표절 암기 방지, (2) 지식 암기 방지, (3) 개인 정보 유출 방지, (4) 제거되지 않은 데이터에 대한 유용성 유지, (5) 제거 요구 크기에 대한 확장성, (6) 순차적 언러닝 요구에 대한 지속 가능성이 포함됩니다. 이러한 기준을 사용하여 7B-파라미터 언어 모델(LM)에서 해리 포터 책과 뉴스 기사를 언러닝할 수 있는 8개의 알고리즘을 평가했습니다.

- **Performance Highlights**: 대부분의 알고리즘이 표절 암기 및 지식 암기를 방지하는 데 효과적이지만, 심각한 개인 정보 유출을 방지하지 못합니다. 또한 이들 알고리즘은 모델의 일반적인 유용성을 저하시켜 대규모 콘텐츠 제거나 순차적 언러닝 요구를 지속적으로 처리하는 데 어려움을 겪습니다. 특히 Negative Preference Optimization (NPO)과 Task Vectors가 표절 및 지식 암기를 효과적으로 제거하지만, 개인 정보 유출을 방지하지 못하고 모델 유용성을 크게 감소시킵니다. 이러한 평가 결과는 현재의 언러닝 알고리즘이 실용적 사용에 적합하지 않음을 보여줍니다.

- **Conclusion**: 기존의 언러닝 알고리즘이 데이터 소유자의 기대를 충족하지 못하고, 모델 배포자의 요구를 충족하기에도 부족함을 지적합니다. 이는 프라이버시 규제와 저작권 소송 증가에 따른 언러닝 알고리즘의 효율적이고 효과적인 필요성이 증가하고 있음에도 불구하고, 현재 사용 가능한 알고리즘은 실용성 있는 사용이나 실제 세계 시나리오에서의 배포에 적합하지 않음을 의미합니다. 본 연구는 이러한 문제를 강조하고, 추가 연구가 필요함을 시사합니다.



### An Empirical Study of Gendered Stereotypes in Emotional Attributes for Bangla in Multilingual Large Language Models (https://arxiv.org/abs/2407.06432)
Comments:
          Accepted at the 5th Workshop on Gender Bias in Natural Language Processing at the ACL 2024 Conference

- **What's New**: 이번 연구는 방글라 데시에서 사용되는 언어인 방글라(Bangla)에서 성별 감정 귀속(gendered emotion attribution)에 관한 최초의 포괄적인 조사를 제공합니다. 이 연구는 방글라 NLP(Natural Language Processing) 연구를 지원하기 위해 코드와 데이터를 공개했습니다.

- **Technical Details**: 연구는 대형 언어 모델(LLMs)의 성별 감정 편향(gender bias)을 조사하며, 73,000개 이상의 LLM 생성 응답과 6,000개 이상의 온라인 댓글을 활용한 정량 분석과 정성 분석을 포함합니다. 제로샷 학습(Zero-shot Learning) 접근 방식을 사용하여, 모델이 사전에 제공된 훈련 예 없이 성별 감정 편향을 나타내는지 확인합니다. 사용된 LLM으로는 Llama3, GPT-3.5-Turbo, GPT-4o 등이 있습니다.

- **Performance Highlights**: 실험 결과, LLM은 성별 고정관념이 존재하며, 이는 방글라 언어 모델 응답에 의한 특정 감정 귀속에서 확인되었습니다. 이로 인해 특정 인구 집단에 해를 끼칠 수 있음을 시사합니다. 특히, 여성은 공감, 두려움, 죄책감과 같은 감정과 연관되고, 남성은 분노, 허세, 권위와 같은 감정과 연관되는 경우가 많은 것으로 나타났습니다.



### DebUnc: Mitigating Hallucinations in Large Language Model Agent Communication with Uncertainty Estimations (https://arxiv.org/abs/2407.06426)
- **What's New**: 새로운 다중 에이전트 토론(DebUnc) 프레임워크를 소개합니다. 이 프레임워크는 모델의 불확도(unsertainty) 메트릭을 사용하여 에이전트의 신뢰도 수준을 평가합니다. 다중 라운드 토론에서 에이전트들이 자신들의 불확도를 표현하고 이를 바탕으로 토큰 가중치를 조정하여 더 정확한 결론을 도출하도록 합니다.

- **Technical Details**: DebUnc는 다음과 같은 두 가지 방법으로 에이전트의 불확도를 전달합니다. 첫째, 텍스트 프롬프트(prompt)에 불확도 정보를 직접 포함시킵니다. 둘째, 에이전트의 불확도에 따라 LLM의 어텐션(attention) 메커니즘을 조정하여 토큰 가중치를 조절합니다. 이를 통해 더욱 정확한 의사소통을 유도할 수 있습니다. 다양한 LLM과 벤치마크, 불확도 메트릭을 활용하여 DebUnc의 성능을 평가하였습니다.

- **Performance Highlights**: 테스트 결과, 어텐션 기반 방법이 특히 효과적임을 확인하였습니다. 불확도 메트릭이 발전함에 따라 성능이 지속적으로 향상될 것으로 예상됩니다.



### Data, Data Everywhere: A Guide for Pretraining Dataset Construction (https://arxiv.org/abs/2407.06380)
Comments:
          Preprint. Under review

- **What's New**: 최근 언어 모델들이 뛰어난 성능을 보이는 이유는 수조 개의 토큰(token)으로 이루어진 사전 학습 데이터셋(pretraining datasets)에 있다고 할 수 있습니다. 하지만 모델 개발자들은 이 데이터셋 구축 방법론을 공개하지 않아서 효과적인 사전 학습 세트 개발에 대한 개방된 정보가 부족했습니다. 이를 해결하기 위해 본 논문에서는 사전 학습 세트 구축 전체 파이프라인에 대한 체계적인 연구를 최초로 수행합니다.

- **Technical Details**: 우선, 기존 사전 학습 세트 개발 기법(techniques)을 사용하여 모델 정확도가 가장 크게 향상되는 방법을 식별하기 위한 실험을 수행합니다. 다음으로, 가장 널리 사용되는 데이터 소스인 웹 크롤링 스냅샷(web crawl snapshots)을 독성(toxicity), 품질(quality), 발화 유형(type of speech), 도메인(domain) 속성에 따라 범주화합니다. 마지막으로, 이러한 속성 정보를 사용하여 사전 학습 세트의 품질을 더욱 정제하고 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 이러한 결과는 실무자들이 높은 품질의 사전 학습 세트를 개발하기 위해 사용할 수 있는 실행 가능한 단계들을 구성합니다.



### Large Language Model Recall Uncertainty is Modulated by the Fan Effec (https://arxiv.org/abs/2407.06349)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 인간의 인지 팬 효과(cognitive fan effects)를 모방하는지를 평가합니다. 팬 효과는 처음 학습된 개념들과 공통된 특징이 있는 개념을 인식하고 수락 또는 거부하는 시간이 더 길어지는 현상을 의미합니다. 연구 결과에 따르면, LLM의 기억 불확실성은 팬 효과에 의해 영향을 받으며, 불확실성을 제거하면 이 효과가 사라집니다.

- **Technical Details**: 두 세트의 인컨텍스트 회상 실험(in-context recall experiments)을 통해 팬 효과를 유도합니다. 토큰 확률을 사용하여 LLM의 기억 불확실성을 측정했으며, 인간의 팬 효과와 일치하는 결과를 발견했습니다. 또한, 전처리(pre-training) 데이터와 인컨텍스트에서 유도된 팬값에 대해 일관된 팬 효과를 보였습니다.

- **Performance Highlights**: 팬 효과가 존재함을 보여주는 일부 LLM들은 전형성(typicality)과 팬 효과가 동일한 현상의 표현임을 시사합니다. Mistral Jiang et al. (2023)과 SOLAR Kim et al. (2023)은 인간과 유사한 팬 효과를 보여주는 것 같다고 결론지었습니다.



### CharSS: Character-Level Transformer Model for Sanskrit Word Segmentation (https://arxiv.org/abs/2407.06331)
- **What's New**: 이번 논문에서는 인도 언어에서의 하위 단어(token) 분할이 중요한 이유와, 특히 산스크리트어에서의 효과적인 분할법을 제안합니다. CharSS(Character-level Transformer model for Sanskrit Word Segmentation)라는 새로운 접근법을 통해 더 나은 결과를 도출할 수 있음을 보입니다.

- **Technical Details**: CharSS는 Character-level Transformer 모델인 ByT5(Byte-Level Text-to-Text Transfer Transformer)를 사용하여 산스크리트어 단어를 분할합니다. ByT5는 특정 언어의 토크나이징(tokenization) 없이 바이트 단위로 텍스트를 처리할 수 있는 장점이 있으며, 다양한 언어와 스크립트를 효율적으로 다룰 수 있습니다. 입력은 하나의 산스크리트어 단어이며, 출력은 '+' 기호로 구분된 서브 토큰(sub-tokens)입니다.

- **Performance Highlights**: UoH+SandhiKosh 데이터셋에서 CharSS는 현재 최고 성능 시스템보다 분할 예측 정확도에서 6.72 포인트의 절대적인 향상을 보였습니다. 해커톤 데이터셋에서는 완벽 매치 메트릭(perfect match metric)에서 2.27 포인트의 성과 향상이 있었습니다. 또한, 낮은 자원(low-resource) 인도 언어로의 기술 용어 번역에서 평균 8.46 및 6.79 chrF++ 점수의 향상을 달성했습니다.



### When in Doubt, Cascade: Towards Building Efficient and Capable Guardrails (https://arxiv.org/abs/2407.06323)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)이 유해하거나 편향된 텍스트를 생성하는 문제를 해결하기 위한 방안으로, 감시자 모델(guardrail models)의 개발에 초점을 맞추고 있습니다. 특히 사회적 편향을 탐지하는 모델을 개발하면서 '사용-언급 구별(use-mention distinction)'의 중요성을 발견하고, 이를 기반으로 데이터를 생성하는 파이프라인을 구축하였습니다.

- **Technical Details**: 이 논문에서는 taxonomy(분류 체계) 기반의 지침을 활용하여 고품질의 레이블이 있는 대규모 대조 데이터(contrastive data)를 생성하는 파이프라인을 제시합니다. 이를 통해 30만 개 이상의 고유 샘플을 생성하였으며, 주목할 만한 점은 LLM의 텍스트 생성에서 사용과 언급의 차이를 명확히 구분할 수 있도록 설계되었다는 점입니다. 모델은 BERT 기반의 encoder-only 모델로, 파라미터는 약 1억 개입니다.

- **Performance Highlights**: 제안된 방법론은 최소한의 컴퓨팅 비용으로 경쟁력 있는 성능을 달성합니다. 실험 결과, 허위 양성 비율(false positive rate)을 82%에서 33%로 크게 줄였으며, 이를 통해 사회적 편향 탐지 모델의 성능을 크게 향상시켰습니다.



### Personality Analysis for Social Media Users using Arabic language and its Effect on Sentiment Analysis (https://arxiv.org/abs/2407.06314)
- **What's New**: 이 연구는 트위터에서 아랍어 사용, 성격 특성(personality traits) 및 감정 분석(sentiment analysis)의 상관관계를 탐구했습니다. 주요 찾은 바는 성격 특성은 소셜 미디어에서 감정에 영향을 미친다는 것입니다.

- **Technical Details**: 사용자의 성격 특성을 분석하기 위해 프로필 활동과 트윗 내용을 기반으로 정보를 추출했습니다. 언어적 특성, 프로필 통계(성별, 나이, 자기소개 등)뿐만 아니라 이모티콘 같은 추가 요소들도 포함되었습니다. 사용자의 성격 데이터를 얻기 위해 아랍어로 16 성격 테스트(16personalities test)를 받은 사용자의 트위터 프로필을 크롤링했습니다. 총 3,250명의 사용자가 데이터를 제공합니다. 다양한 머신러닝 기술을 적용하여 성격 특성을 추론하였고, 이를 위해 BERT를 사용하여 74.86%의 정확률을 달성했습니다.

- **Performance Highlights**: BERT 모델을 사용하여 74.86%의 높은 정확도로 성격 특성을 판별할 수 있음을 입증했습니다. 이 데이터셋을 분석한 결과, 언어적 특성과 프로필 요소, 도출된 모델은 다른 성격 특성을 구별하는 데 유용하다는 것이 확인되었습니다.



### Hybrid X-Linker: Automated Data Generation and Extreme Multi-label Ranking for Biomedical Entity Linking (https://arxiv.org/abs/2407.06292)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 기존 최첨단 딥러닝(entity linking) 방법들은 많은 양의 인간 라벨링 데이터에 의존하고 있습니다. 이 논문에서는 새로운 대규모 훈련 데이터셋을 자동으로 생성하여 인간 라벨링 데이터의 부족 문제를 해결하는 하이브리드 X-Linker 파이프라인을 제안합니다. 이 시스템은 질병 및 화학 물질 개체를 연결하여 MEDIC 및 CTD-Chemical 어휘에 개념을 연결합니다. X-Linker는 BC5CDR-Disease, BioRED-Disease, NCBI-Disease, BC5CDR-Chemical, BioRED-Chemical, 및 NLM-Chem 등의 다양한 생의학 데이터셋에서 평가되었습니다.

- **Technical Details**: X-Linker 파이프라인은 다양한 모듈로 구성되어 있으며, 각 모듈은 질병 및 화학 물질 개체 언급을 MEDIC 및 CTD-Chemical 어휘의 개념에 연결합니다. 특히, 본 연구에서는 문제를 극단적인 다중 라벨 랭킹(Xtreme Multi-Label Ranking, XMR) 문제로 프레임화하여 해결합니다. PECOS-EL 모델을 개발하고, 여러 종류의 효과적인 EL 접근 방식과 통합하여 대규모 훈련 데이터셋에서 학습을 진행했습니다. 원시 코드와 관련 데이터는 공개되어 있습니다.

- **Performance Highlights**: X-Linker는 BC5CDR-Disease, BioRED-Disease, NCBI-Disease, BC5CDR-Chemical, BioRED-Chemical, 및 NLM-Chem 데이터셋에서 각각 0.8307, 0.7969, 0.8271, 0.9511, 0.9248 및 0.7895의 top-1 정확도를 달성했습니다. 특히 BC5CDR-Disease, NCBI-Disease 및 BioRED-Chemical 데이터셋에서 높은 성능을 보였습니다. 반면, SapBERT는 나머지 데이터셋에서 더 나은 성능을 보였습니다.



### CodeUpdateArena: Benchmarking Knowledge Editing on API Updates (https://arxiv.org/abs/2407.06249)
Comments:
          Under Review

- **What's New**: 대형 언어 모델(LLMs)이 점점 소스 코드(synthesize and reason)를 생성하고 추론하는 데 사용되고 있습니다. 그러나 이러한 모델들의 지식은 정적이어서 호출하는 라이브러리(libraries)와 API 함수(API functions)가 지속적으로 진화하고 있는 현실을 반영하지 못합니다. 이에 대응하기 위해, 우리는 코드 도메인에서 지식 편집(knowledge editing)을 위한 벤치마크인 CodeUpdateArena를 제시합니다. 이 벤치마크는 API 함수 업데이트와 이를 사용한 프로그램 합성 예시를 포함하여 LLM의 지식을 업데이트하는 방법을 평가합니다.

- **Technical Details**: 우리의 데이터셋은 먼저 GPT-4를 활용하여 원자적이고 실행 가능한 함수 업데이트를 생성(synthetic API function update)한 후, 각 업데이트에 대해 해당 업데이트를 사용하는 프로그램 합성 예제(program synthesis example)를 생성하여 구성됩니다. 이벤치마크는 다양한 유형의 54개의 파이썬 패키지(Python packages)에서 670개의 프로그램 합성 예를 포함하고 있습니다. 이러한 예제를 통해 LLM이 업데이트된 기능을 반영하여 문제를 해결할 수 있는지를 평가합니다. 기존의 지식 편집(knowledge editing) 기술이 여전히 개선의 여지가 많다는 것을 보였습니다.

- **Performance Highlights**: 우리의 실험 결과, 업데이트에 대한 문서를 오픈 소스 코드 LLM(DeepSeek, CodeLlama)에 추가(prepending documentation)해도 문제 해결 능력이 향상되지 않으며, 기존 지식 편집 기술이 여전히 많은 개선의 여지를 가지고 있음을 확인할 수 있습니다. 우리는 이 벤치마크가 코드 LLM에서 지식을 업데이트하는 새로운 방법들을 영감받을 수 있기를 기대합니다.



### Predicting Word Similarity in Context with Referential Translation Machines (https://arxiv.org/abs/2407.06230)
Comments:
          11 pages, 3 figures, 8 tables. arXiv admin note: substantial text overlap with arXiv:2407.05154

- **What's New**: 이 연구는 두 단어 간의 유사성을 기계 번역 성능 예측(MTPP)으로 나타내어, 참조 번역 기계(RTMs)를 통해 컨텍스트와 유사성 간의 거리를 기반으로 유사성을 식별하려고 합니다. 이 접근법은 Graded Word Similarity in Context (GWSC) 과제에서 최고 성과를 달성할 수 있습니다.

- **Technical Details**: GWSC 과제는 두 단어가 서로 다른 두 컨텍스트 c1과 c2에서의 유사성을 연속적인 스케일로 평가합니다. RTMs 접근방식을 통해 각 컨텍스트를 평균적으로 60개의 단어로 나누고, intra와 inter 유사성을 측정합니다. SimLex-999 데이터셋을 기반으로 하며, unsupervised 학습으로 유사성 점수를 예측합니다. intra-cwps와 inter-cwps는 각각 동일한 컨텍스트 내 및 다른 컨텍스트 간의 단어 쌍의 유사성 변화를 측정합니다.

- **Performance Highlights**: 이 방법론은 의미적 유사성을 모델링하기 위해 145개의 특징을 사용하며, 훈련 및 테스트 데이터에서 유사한 점수 분포(mean, max, min을 기준으로)를 보여줍니다. 최종적으로 인공적인 wps 점수를 찾아내어 두 가지 다른 컨텍스트 간의 유사성을 예측할 수 있습니다.



### Resolving Sentiment Discrepancy for Multimodal Sentiment Detection via Semantics Completion and Decomposition (https://arxiv.org/abs/2407.07026)
Comments:
          8 pages, 6 figures

- **What's New**: 최근 소셜 미디어 게시물의 폭증으로 이미지-텍스트(multimodal) 콘텐츠의 감정 분석 필요성이 급증했습니다. 이 논문에서는 이미지와 텍스트가 상반된 감정을 나타내는 문제를 해결하기 위해 새로운 접근법인 CoDe(semantics Completion and Decomposition) 네트워크를 제안합니다. 이 기법은 OCR(Optical Character Recognition)로부터 얻어진 텍스트를 활용해 감정 차이를 해소하고, 독점적 투사 및 대조 학습을 통해 이미지와 텍스트 표현을 분해하여 감정 차이를 명시적으로 포착합니다.

- **Technical Details**: CoDe 네트워크는 두 가지 주요 모듈을 포함합니다. 첫째, semantics completion 모듈에서는 이미지에 포함된 OCR 텍스트를 사용하여 이미지와 텍스트 표현을 보완하고 감정 격차를 줄입니다. 둘째, semantics decomposition 모듈에서는 이미지와 텍스트 표현을 독점적인 서브 표현으로 분해하여 서로 다른 감정을 명시적으로 포착합니다. 이러한 방법을 통한 대조 학습을 통해 일치하는 감정과 불일치하는 감정을 모두 포함한 대표성을 얻습니다. 마지막으로, cross-attention을 사용하여 최종 분류를 위해 이미지와 텍스트 표현을 융합합니다.

- **Performance Highlights**: 네 가지 멀티모달 감정 데이터셋을 이용한 광범위한 실험 결과, CoDe가 최첨단(SOTA) 방법보다 우수한 성능을 보여줬습니다. 특히, 감정 차이가 명확히 드러나는 게시물에서도 뛰어난 성능을 발휘해 기존 방법의 한계를 극복했습니다.



### End-To-End Causal Effect Estimation from Unstructured Natural Language Data (https://arxiv.org/abs/2407.07018)
Comments:
          26 pages, 10 figures

- **What's New**: 이번 연구에서는 대규모의 다양한 관찰 텍스트 데이터를 대형 언어 모델(Large Language Models, LLMs)을 활용하여 비용 효율적인 인과 효과 추정을 자동으로 수행할 수 있는 새로운 방법을 제시합니다. NATURAL이라는 새로운 인과 효과 추정기 모델 가족을 소개하며, 이는 비정형 텍스트 데이터 기반으로 동작합니다. 이를 통해 데이터 수집 및 구조화의 수작업 과정을 크게 줄이고, 자동화된 파이프라인으로 인과 효과를 추정하는 데에 첫걸음을 내딛었습니다.

- **Technical Details**: NATURAL 추정기는 자연어 보고서를 기반으로 변수를 추출하고, LLM 모델을 사용하여 관심 변수(결과, 치료, 공변량)의 조건부 분포를 예측하여 ATE (Average Treatment Effect)를 계산합니다. 이 과정에서 추정기들은 전통적인 방법인 역성향 점수 가중치(inverse propensity score weighting)와 결과 추정(outcome imputation) 방법을 활용합니다. 또한, 데이터 큐레이션을 자동화하고, LLM을 사용하여 누락된 정보를 보완하는 기술적 도전 과제를 해결하였습니다.

- **Performance Highlights**: 여섯 개의 관찰 데이터셋(두 개의 합성 데이터셋과 네 개의 실제 데이터셋)을 활용하여 NATURAL 추정기를 평가한 결과, 실제 세계의 무작위 실험 결과와 비교했을 때, 추정된 ATE가 실제 값과 3퍼센트 포인트 이내에서 일치하여 놀라운 성과를 보였습니다. 이는 수백만 달러의 비용 절감 효과를 가져올 수 있는 잠재력을 시사합니다. 특히 합성 데이터셋은 마케팅 데이터를 이용해 구성되었고, 실제 데이터셋은 공공(migraine and diabetes) 서브레딧으로부터 큐레이션되었습니다.



### Metron: Holistic Performance Evaluation Framework for LLM Inference Systems (https://arxiv.org/abs/2407.07000)
- **What's New**: LLM(대형 언어 모델) 프로덕션 서비스 비용이 높아지면서 추론 시스템 최적화가 필요합니다. 기존 평가 지표인 TTFT(TTF Time To First Token), TBT(Time Between Tokens), Normalised Latency 그리고 TPOT(Time Per Output Token)은 실시간 애플리케이션의 성능을 완전하게 평가하지 못합니다. 이를 해결하기 위해 'Metron'이라는 종합적인 성능 평가 프레임워크를 제안합니다. Metron에는 LLM 추론의 복잡성을 반영한 fluidity-index라는 새로운 지표가 포함되어 있습니다.

- **Technical Details**: 기존 지표들은 사용자 경험을 완전하게 반영하지 못합니다. TTFT와 TBT는 개별 토큰의 지연을 잘 캡쳐하지만, 전체적인 토큰 생성의 처리량을 충분히 나타내지 못합니다. TPOT와 정상화된 지연시간 지표는 토큰 처리량을 측정하지만, 사용자 경험 저하의 원인인 인터토큰 지터(Inter-token Jitter)나 스케줄링 지연(Scheduling Delays)을 식별하지 못합니다. 이를 해결하기 위해 Metron은 fluidity-index와 fluid token generation rate라는 새로운 지표를 사용하여 실시간 스트리밍 LLM 상호작용을 평가합니다. fluidity-index는 미디어 스트리밍 플랫폼의 버퍼링 시간과 유사하게 토큰 생성의 일관성을 평가하며, fluid token generation rate는 블랙박스 평가를 가능하게 합니다.

- **Performance Highlights**: Metron을 사용하여 여러 오픈 소스 플랫폼과 모델 서비스 제공 업체(Model-As-A-Service)의 성능을 평가했습니다. fluidity-index와 fluid token generation rate를 통해 실시간 LLM 상호작용의 세부 사항을 포착하고, 사용자 경험에 큰 영향을 미치는 요인들을 보다 명확하게 식별할 수 있었습니다. Metron은 GitHub에서 오픈 소스로 제공되며, 빠르게 진화하는 LLM 추론 시스템 및 프레임워크에서 사용자 중심의 성능 평가 기준을 제시합니다.



### Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspectiv (https://arxiv.org/abs/2407.06992)
Comments:
          Survey paper

- **What's New**: 최근 뉴럴 정보 검색(IR) 모델은 다양한 IR 작업에서 그 효과성을 크게 향상시켰습니다. 본 논문은 이러한 IR 모델들의 강건성 문제를 다루며, 이를 네 가지 측면에서 분석합니다. 특히, 적대적 공격(adversarial attacks)과 분포 외(out-of-distribution, OOD) 시나리오에서의 강건성에 중점을 두어, 밀집 검색 모델(Dense Retrieval Models, DRMs)과 뉴럴 랭킹 모델(Neural Ranking Models, NRMs)을 분석합니다. 또한, 능력 평가를 위한 BestIR 벤치마크를 소개하며, 이 연구가 신뢰할 수 있는 검색 엔진 개발에 도움이 되기를 바랍니다.

- **Technical Details**: IR 시스템의 강건성은 여러 측면에서 평가됩니다. 첫째, 적대적 강건성은 IR 모델이 순위를 변조하려는 공격에 방어할 수 있는 능력을 평가합니다. 둘째, OOD 강건성은 IR 모델이 훈련 데이터와 다른 분포의 새로운 문서 또는 쿼리에서도 잘 일반화하는 능력을 평가합니다. 이를 위해 다양한 데이터셋과 평가 메트릭을 사용합니다. 예를 들어, 적대적 공격과 방어를 다루는 연구 또는 생소한 문서와 쿼리에서의 일반화를 평가하는 연구방법론이 있습니다. 이러한 연구는 뉴럴 IR 모델의 첫 단계(검색 단계)와 두 번째 최종 랭킹 단계에서 모두 적용됩니다.

- **Performance Highlights**: 이 연구는 기존의 강건성 연구들을 체계적으로 정리하고, 다양한 데이터셋과 평가 메트릭을 제시하여, 현재 뉴럴 IR 모델의 강건성 상태를 종합적으로 보여줍니다. BestIR 벤치마크를 통해 여러 강건성 스펙트럼에서 IR 모델의 성능을 평가할 수 있는 구조를 제공합니다. 이러한 접근방법은 뉴럴 IR 모델이 실제 응용에서 신뢰성을 높일 수 있도록 돕습니다.



### Listen and Speak Fairly: A Study on Semantic Gender Bias in Speech Integrated Large Language Models (https://arxiv.org/abs/2407.06957)
- **What's New**: 이번 연구는 음성 인식 기능이 통합된 대형 언어 모델(SILLMs)을 평가하기 위한 맞춤형 음성 편향 평가 도구와 해당 데이터셋을 소개합니다. 특히 성별 편향을 평가하여 SILLMs가 훈련 데이터의 편향을 얼마나 증폭시킬 수 있는지 분석했습니다.

- **Technical Details**: 이 연구는 SILLMs의 네 가지 의미 관련 작업인 음성-텍스트 변환(STT), 음성 대명사 해소(SCR), 음성 문장 이어가기(SSC) 및 음성 질문 응답(SQA)에서 성별 편향을 평가했습니다. 주요 실험은 Whisper v3 Large 및 SeamlessM4T v2 Large와 같은 최첨단(STA) 모델을 사용했고, Qwen-Audio-Chat, SALMONN, WavLLM 등의 최신 Speech LLM을 평가에 포함시켰습니다. 실험 설계는 연구자들이 추가적인 SILLM 및 평가 과제를 통합하고 평가할 수 있도록 쉽게 재현할 수 있게 설계되었습니다.

- **Performance Highlights**: 연구 결과는 SILLMs의 편향 수준이 사용하는 언어와 평가 방법에 따라 달라진다는 것을 보여줍니다. 예를 들어, 영어에서 성별을 명확히 표시하지 않는 경우 스페인어나 독일어와 같은 문법적으로 성이 분명한 언어로 번역할 때 오류가 발생할 수 있습니다. 또한, SILLMs의 편향은 다국어 사용 시 다르게 나타날 수 있음을 강조하고, 이를 평가하기 위해 여러 접근 방식을 사용하는 것이 중요함을 확인했습니다.



### ICLGuard: Controlling In-Context Learning Behavior for Applicability Authorization (https://arxiv.org/abs/2407.06955)
- **What's New**: 최근 크게 주목받고 있는 인컨텍스트 러닝(ICL; In-context learning) 기능이 대형 언어 모델(LLMs; Large Language Models)의 새로운 가능성을 보여주고 있습니다. 이는 모델 업데이트 없이도 새로운 작업을 수행할 수 있도록 하는 기능입니다. 이를 통해 추론 시간 동안 몇 가지 입력-레이블 쌍과 테스트 입력을 조건으로 작업을 수행할 수 있습니다. 하지만 이 기능이 모델 정책에 어긋나는 데이터로 작업을 수행할 수 있기 때문에 모델 소유자는 ICL 사용을 통제할 필요가 있습니다. 이를 위해 '적용성 승인(Applicability Authorization)' 개념과 ICLGuard라는 간단한 접근법을 제안했습니다.

- **Technical Details**: ICLGuard는 최소한의 추가 훈련 가능한 파라미터만 미세 조정하여 LLM의 원래 모델을 '보호'하는 미세 조정 프레임워크입니다. 여기서 '파라미터 효율적인 미세 조정(PEFT; Parameter-Efficient Fine-Tuning)' 방법을 활용하여 적은 리소스로 모델 소유자가 모델을 즉석에서 조정할 수 있게 하였습니다. 세 가지 손실 함수(비활성화 손실, 유지 손실, 유틸리티 손실)를 사용하여 PEFT 모듈을 최적화함으로써 다양한 데이터에서 ICL 행동을 통제할 수 있도록 했습니다.

- **Performance Highlights**: 4개의 데이터셋(FP, SST-2, TREC, AGnews)과 3개의 LLM(LLaMA, OPT, Cerebras)에 대한 실험 결과, ICLGuard는 목표 데이터에서 ICL 기능을 비활성화하면서도 다른 데이터와 일반 기능에는 영향을 미치지 않음을 보여줬습니다. 각 손실 함수가 ICL 행동을 통제하는 데 미치는 영향도 연구하였고, 다양한 데이터 생성 전략을 통해 성능을 향상시키는 방법도 탐구했습니다. ICLGuard의 적응형 공격에 대한 잠재력과 다양한 설정 비교 연구도 수행되었습니다. 이를 통해 ICLGuard가 생성 작업에서도 ICL 행동을 통제할 수 있는 가능성을 확인했습니다.



### Spanish TrOCR: Leveraging Transfer Learning for Language Adaptation (https://arxiv.org/abs/2407.06950)
Comments:
          10 pages, 5 figures

- **What's New**: 본 연구는 TrOCR 아키텍처의 스페인어 전이학습 능력을 탐구합니다. TrOCR은 Transformer 기반의 Optical Character Recognition(OCR) 모델로, 영어 벤치마크에서 최첨단 성능을 자랑합니다. 우리는 TrOCR 모델을 새로운 언어에 적응시키기 위한 두 가지 접근 방식을 조사했습니다: 영어 TrOCR 인코더와 언어 특화 디코더를 결합하여 모델을 해당 언어로 학습시키는 방법과, 영어 기반 TrOCR 모델을 새로운 언어 데이터로 미세 조정하는 방법입니다. 제한된 공개 데이터셋 문제를 해결하기 위해, 우리는 리소스 효율적인 OCR 데이터셋 생성 파이프라인을 제시하고, 다양한 이미지 생성 방법들의 포괄적인 벤치마크를 제공합니다.

- **Technical Details**: TrOCR 모델은 Transformer 아키텍처를 활용하여 텍스트 인식 성능을 극대화합니다. 트랜스포머 기반 디자인이 텍스트 생성과 이미지 해석을 결합하여 다양한 언어적 맥락에서 우수한 성능을 발휘할 수 있게 해줍니다. 우리는 TrOCR 모델을 스페인어에 맞게 조정하기 위해 두 가지 접근 방식을 시도했습니다. 첫 번째 방식은 영어로 미리 학습된 TrOCR 체크포인트를 기반으로 스페인어 데이터를 학습하는 것이고, 두 번째 방식은 스페인어 텍스트 디코더 체크포인트에서 초기화한 TrOCR 모델을 학습하는 것입니다.

- **Performance Highlights**: 스페인어 TrOCR 모델을 평가한 결과, 미세 조정된 영어 TrOCR이 고정된 데이터셋 크기에서 언어 특화 디코더보다 더 우수한 인식 성능을 보였습니다. 우리는 공공 데이터셋을 활용해 문자 오류율과 단어 오류율로 모델 성능을 평가하였으며, 다른 오픈소스 및 클라우드 OCR 모델들과 비교했습니다. 공개된 스페인어 TrOCR 모델들은 현재 HuggingFace에서 이용 가능하며, 데이터셋 생성 코드는 Github에 공개되어 있습니다.



### Learn and Don't Forget: Adding a New Language to ASR Foundation Models (https://arxiv.org/abs/2407.06800)
- **What's New**: 기존 ASR 모델에 비해 추가 언어를 통합하는 방법을 논의하며, 주로 Whisper 모델을 활용하여 새로운 목표 언어를 추가하는 실험을 진행합니다. 이 과정에서 성능 저하를 방지하기 위해 다양한 파인튜닝(Fine-Tuning) 방법을 비교 및 분석했습니다. 특히, LoRA(저급 적응)와 EWC(탄력적 가중치 통합)를 활용한 새로운 접근 방식을 시도했습니다.

- **Technical Details**: 모델의 파라미터를 θ_ASR로 표기하며, ASR 모델은 주어진 음향적 특징으로부터 목표 전사를 생성하는 것을 목표로 합니다. 본 연구에서는 Whisper 모델을 사용하여 새로운 언어를 통합하기 위해 파인튜닝 접근 방식을 채택했습니다. 모델의 모든 파라미터를 도메인 특화 데이터로 조정하여 학습 손실을 최소화합니다. LoRA는 저급(rank) 근사를 활용하여 모델을 새로운 데이터 분포에 적응시키는 혁신적이고 효율적인 파인튜닝 전략입니다. 이를 통해 파라미터 업데이트를 저급 매트릭스 분해를 통해 제한합니다. 또한, EWC를 사용하여 기존 언어 성능 저하를 방지하며 새로운 언어 성능을 유지합니다.

- **Performance Highlights**: 직접 파인튜닝은 새로운 언어에서 최고의 성능을 보였지만 기존 언어에서는 성능이 저하되었습니다. 반면, EWC를 사용하면 특정 언어에 대해 성능 저하 없이 유지할 수 있었고, 적응 파라미터만 사용하면 기존 언어 성능을 유지할 수 있으나 새로운 언어 성능에서 손실이 발생합니다. 실험 결과는 Whisper 모델의 zero-shot 성능과 다양한 파인튜닝 방법이 새로운 언어에서 어떻게 성능을 발휘하는지를 평가했습니다.



### Entropy Law: The Story Behind Data Compression and LLM Performanc (https://arxiv.org/abs/2407.06645)
- **What's New**: 본 논문은 데이터 선택과 LLM(Large Language Models) 성능 간의 관계를 탐구하며, 전통적인 개별 샘플의 품질 평가 방식을 넘어, 샘플 조합을 고려한 데이터 선택 방법을 제안합니다. 저자들은 LLM의 정보 압축 특성에서 영감을 받아 '엔트로피 법칙(entropy law)'을 발견하였고, 이를 기반으로 ZIP라는 효율적인 데이터 선택 방법을 제안합니다.

- **Technical Details**: 논문은 데이터셋의 정보 중복성과 내포된 지식의 숙련도를 반영하는 데이터 압축 비율과 초기 학습 손실을 통해 데이터 선택의 중요성을 이론적으로 도출하였습니다. ZIP 알고리즘은 다양한 데이터를 탐욕적(greedy)으로 선택하여 낮은 압축 비율을 우선시하며, 한정된 조합을 고려하여 예비 필터링을 통해 후보 풀을 축소하고, 그 중 압축 비율을 최소화하는 샘플을 선택합니다.

- **Performance Highlights**: 다양한 LLM 백본 및 정렬 단계에서 실험을 통해 ZIP의 우수성이 입증되었습니다. 또한, 엔트로피 법칙을 활용해 초기 모델 학습 단계에서 잠재적인 성능 위험을 감지하여, LLM 개발의 계산 오버헤드를 효과적으로 줄일 수 있음을 보여주었습니다.



### Tailored Design of Audio-Visual Speech Recognition Models using Branchformers (https://arxiv.org/abs/2407.06606)
Comments:
          Submitted and under review for the IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP) journal

- **What's New**: 최신 발표된 연구에서는 Audio-Visual Speech Recognition (AVSR)을 개선하는 혁신적인 방법을 제안했습니다. 이 연구는 기존의 독립적인 인코더 기반 모델들을 대신해, Branchformer와 같은 인코더 아키텍처의 유연성과 해석 가능성을 활용해 더 효율적인 AVSR 시스템을 설계하는 첫 번째 시도입니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 오디오 및 비디오 전용 시스템을 추정합니다. 두 번째 단계에서는 모달리티별 모델에 의해 제공된 레이어 수준의 브랜치 점수를 기반으로 맞춤형 오디오-비주얼 통합 인코더를 설계합니다. 이 프레임워크는 파라미터 효율성을 극대화하는 것을 목표로 하며, Branchformer의 해석 가능성과 유연성을 활용합니다.

- **Performance Highlights**: 영어와 스페인어의 여러 AVSR 벤치마크에서 광범위한 실험을 통해 제안된 방법의 효과가 입증되었습니다. 결과는 일반적인 AVSR 접근 방식에 비해 모델 복잡성을 크게 줄이면서도 최첨단 인식률(state-of-the-art recognition rates)을 달성할 수 있음을 보여줍니다.



### AutoTask: Task Aware Multi-Faceted Single Model for Multi-Task Ads Relevanc (https://arxiv.org/abs/2407.06549)
- **What's New**: 광고의 관련성 모델은 사용자의 검색 쿼리와 광고 제공 간의 관련성을 결정하는 중요한 요소로, 이 문제를 분류 문제로 다룬다. 본 연구에서는 다중 태스크 상황에 적합한 새로운 다중 측면(attention) 모델을 제안한다. 이 모델은 태스크 인식 기능 결합과 태스크 간 상호작용 모델링을 수행하며, 태스크 표현을 위해 태스크 ID 인코딩(Task ID Encoding)을 도입하여 다양한 광고 상황에서 관련성을 정밀하게 모델링할 수 있도록 한다.

- **Technical Details**: 새로운 모델은 언어 모델링(language modeling)과 자기 회귀(attention)를 특징과 태스크 차원에서 결합하여 기능 결합 문제를 해결한다. 이 모델은 태스크 데이터의 무작위 혼합(mixture)을 통해 태스크 블록을 구성하고, 자기 회귀(attention)를 통해 태스크 간 상호작용을 모델링하여 태스크 유사성을 활용한다. 이 방법은 온라인 추론 시 단일 태스크 추론을 가능하게 하며, 모델의 일반화 능력을 향상시킨다.

- **Performance Highlights**: 제안된 모델은 단일 모델로 다양한 태스크를 효과적으로 처리하고, 일반화된 DNN 모델이나 태스크 전용 모델보다 뛰어난 성능을 보여준다. 특히 새로운 태스크에 대한 일반화 능력이 증대되며, 경량화되어 온라인 서빙에 적합하다.



### LETS-C: Leveraging Language Embedding for Time Series Classification (https://arxiv.org/abs/2407.06533)
Comments:
          22 pages, 5 figures, 10 tables

- **What's New**: 언어 모델링의 성과를 활용해 새로운 경량 모델 LETS-C를 제안합니다. 이 모델은 사전 학습된 대형 언어 모델(LLMs)을 미세 조정하는 대신, 언어 임베딩 모델을 사용하여 시계열 데이터를 임베딩하고 이를 CNN과 MLP로 구성된 간단한 분류 헤드와 결합합니다.

- **Technical Details**: LETS-C는 언어 임베딩 (Language Embeddings)을 사용하여 시계열 데이터를 벡터 공간으로 투사합니다. 투사된 임베딩은 CNN과 MLP로 구성된 분류 헤드로 입력되어 다양한 클래스 간 구분을 학습합니다. 이러한 방법을 통해 복잡한 시계열 패턴과 종속성을 효과적으로 포착할 수 있습니다.

- **Performance Highlights**: LETS-C는 10개의 도메인에 걸쳐 있는 표준 벤치마크 데이터셋에서 세계 최첨단(SOTA) 정확도를 달성했으며, 이전 SOTA 모델보다 평균적으로 14.5%의 훈련 가능 파라미터만을 사용하여 경량성을 증명했습니다. 전체적으로 20개의 기준 모델을 능가하며, 특히 정확성과 모델 크키 간의 트레이드오프에서도 효율적인 성능을 나타냈습니다.



### STORYSUMM: Evaluating Faithfulness in Story Summarization (https://arxiv.org/abs/2407.06501)
- **What's New**: 新しい 데이터 세트 STORYSUMM을 소개합니다. 이 데이터 세트는 단편 소설의 LLM 요약과 로컬라이즈드 정밀성 라벨 및 오류 설명을 포함하며, 요약의 정밀성을 평가하는 새로운 벤치마크를 제공합니다. 이로 인해 요약의 일관성 감지를 위한 평가 방법을 더욱 발전시킬 수 있습니다.

- **Technical Details**: STORYSUMM 데이터 세트는 96개의 단편 소설과 LLM이 생성한 요약문을 포함합니다. 각 요약문은 정밀성 오류와 설명으로 라벨링되어 있으며, 오류 감지 난이도로 '쉬움' 또는 '어려움'으로 분류됩니다. 이 데이터 세트는 LLM이 생성한 요약의 정밀성을 평가하며, Reddit에서 수집된 단편 소설을 기반으로 합니다. 데이터 세트는 다양한 모델(GPT-3.5, GPT-4, Claude-3)을 사용하여 생성된 요약문을 포함하고 있습니다.

- **Performance Highlights**: 최근 자동 메트릭을 이 데이터 세트에 적용한 결과, 어느 메트릭도 70% 이상의 균형 정확도를 달성하지 못했습니다. 이는 이러한 벤치마크가 정밀성 평가에 있어 매우 도전적인 과제임을 보여줍니다. 또한, MTurk를 통해 수집된 평가 결과는 성능이 부풀려지기 쉬워 신뢰성이 낮다는 결론에 이르렀습니다. 대신 Upwork를 통해 보다 신뢰할 수 있는 평가를 얻을 수 있었습니다.



### Composable Interventions for Language Models (https://arxiv.org/abs/2407.06483)
- **What's New**: 이 논문은 언어 모델(Language models)의 테스트 시점에서 개입(intervention)을 통해 사실 정확성을 높이고, 유해한 출력을 줄이며, 모델의 효율성을 개선하는 방법을 제시합니다. 여러 개의 개입 방법을 동일 모델에 순차적으로 적용할 때 발생하는 상호 작용을 연구할 표준화된 프레임워크가 부족했던 점을 해결하기 위해, '조합 가능한 개입(composable interventions)'이라는 프레임워크를 제안합니다. 이 프레임워크는 새로운 지표와 통합된 코드베이스를 제공합니다.

- **Technical Details**: 제안된 프레임워크는 Llama3-8B 모델을 기반으로 한 광범위한 실험을 통해 지식 편집(Knowledge Editing), 모델 압축(Model Compression), 머신 언러닝(Machine Unlearning) 등의 개입 방법을 조합하여 상호 작용을 분석합니다. 두 가지 주요 지표(Order-free Error, Order Sensitivity)를 도입하여 개입의 조합성을 평가합니다. Order-free Error는 한 개입이 다른 개입의 성공에 영향을 주지 않는 정도를 나타내며, Order Sensitivity는 여러 개입의 성공이 그 적용 순서에 따라 달라지는지를 나타냅니다.

- **Performance Highlights**: 310개의 다양한 조합을 실험한 결과 세 가지 주요 통찰을 얻었습니다: 1) 모델 압축은 다른 개입의 성공을 제한하는 경향이 있으며, 2) 개입 적용 순서가 그 성공에 큰 영향을 미치며, 3) 일반적인 모델 성능 측정 지표는 조합성을 평가하기에 충분하지 않습니다. 따라서, 순서에 무관한 새 개입 방법의 필요성이 제기됩니다. 논문의 코드는 공개되어 있으며, 다양한 최신 기법과 통합된 코드베이스를 제공합니다.



### A Single Transformer for Scalable Vision-Language Modeling (https://arxiv.org/abs/2407.06438)
Comments:
          Code and data are available at this https URL

- **What's New**: 이번 연구에서는 스케일러블 비전-언어 모델링(SOL)의 단일 Transformer 아키텍처를 소개합니다. 현재의 대형 비전-언어 모델(LVLM)들은 주로 사전 트레이닝된 비주얼 인코더와 대형 언어 모델(LLM)을 연결하는 이기종 아키텍처를 사용하지만, 이는 네 가지 주요 확장성 문제를 가지고 있습니다. 본 논문에서는 이러한 문제를 해결하고, 학술적 자원으로도 구현 가능한 첫 오픈 소스 트레이닝 레시피를 공개합니다.

- **Technical Details**: SOLO는 사전 학습된 비주얼 인코더 없이도 원시 이미지 패치(픽셀)와 텍스트를 입력으로 받아들일 수 있는 단일 Transformer 아키텍처를 사용합니다. 모델은 초기화를 위해 Mistral LLM v0.1을 사용하며, ImageNet과 웹 스케일 데이터셋에서의 순차적 사전 학습 및 고품질 데이터셋에서의 세부 조정이 포함된 트레이닝 레시피를 활용합니다. 이 접근법은 복합 아키텍처의 복잡성을 줄이고, 기존 하드웨어 및 소프트웨어 인프라를 활용할 수 있게 합니다.

- **Performance Highlights**: SOLO는 다양한 평가에서 LLaVA-v1.5-7B 모델과 비교해 유사한 성능을 보이며, 특히 시각적 수학적 추론 영역에서 뛰어난 성과를 냅니다. 또한, 세부 조정된 데이터셋의 사용과 대규모 학습에서도 우수한 성능을 나타냈습니다.



### Exploring the Capability of ChatGPT to Reproduce Human Labels for Social Computing Tasks (Extended Version) (https://arxiv.org/abs/2407.06422)
Comments:
          Extended version of accepted short paper to ASONAM 2024. arXiv admin note: text overlap with arXiv:2304.10145

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)인 ChatGPT가 데이터를 주석달기(annotation)하는 데 있어 사회 컴퓨팅 작업을 어떻게 지원할 수 있는지 조사합니다. 연구진은 COVID-19 오정보, 사회 봇 기만, 사이버 괴롭힘, 클릭베이트 뉴스 및 러시아-우크라이나 전쟁 등 주요 사회 문제를 다루는 7개의 데이터 세트를 재주석 달기하여 ChatGPT의 성능을 검증했습니다. 이를 기반으로 연구팀은 GPT-Rater라는 도구를 제안하여 주어진 주석 달기 작업에서 ChatGPT가 데이터를 올바르게 레이블링할 수 있을지 예측합니다.

- **Technical Details**: 연구팀은 ChatGPT를 사용하여 7개의 데이터 세트에서 5개의 사회 문제(COVID-19 논란, 사회 봇 기만, 사이버 괴롭힘, 클릭베이트 뉴스, 러시아-우크라이나 전쟁)의 텍스트 기반 주석 달기 작업을 수행했습니다. ChatGPT의 주석을 인간이 부여한 라벨과 비교하여 성능을 평가했습니다. 이를 통해 ChatGPT가 데이터 주석 작업을 수행할 가능성을 평가하고, 성능이 과제와 데이터 세트에 따라 상당히 다르다는 것을 발견했습니다. 또한 GPT-Rater 도구를 개발해 ChatGPT의 주석 성능을 예측하고 연구자가 주석 달기 작업의 적합성을 평가할 수 있도록 했습니다.

- **Performance Highlights**: ChatGPT는 클릭베이트 뉴스 검출 작업에서 89.66%의 정확도를 기록하며 가장 높은 성능을 보였고, COVID-19 혐오 발언 감지 작업에서는 52.24%로 가장 낮은 성능을 보였습니다. GPT-Rater 도구는 클릭베이트 뉴스 데이터세트에서 90.59%의 정확도와 95.00%의 F1-Score를 달성했습니다. GPT-Rater는 7개 데이터세트 중 5개에서 평균 F1-Score 75% 이상을 기록했으며, 이는 연구자들이 ChatGPT의 주석 성능을 소수의 인간 라벨만으로 예측할 수 있음을 나타냅니다.



### If You Don't Understand It, Don't Use It: Eliminating Trojans with Filters Between Layers (https://arxiv.org/abs/2407.06411)
Comments:
          11 pages, 6 figures

- **What's New**: 이 논문에서는 데이터 중독(trojans)으로 인한 위협을 제거하기 위한 일반적인 방법과 특정 구현을 소개합니다. 특히, 작은 모델과 중간 크기 모델에서 효과적인 LoRA 필터를 제안합니다. 데이터 중독은 사전 훈련 및 파인 튜닝 단계에서 쉽게 일어날 수 있기 때문에, 기존의 보안 훈련이 이를 완전히 억제하지 못하는 경우가 많습니다.

- **Technical Details**: 주요 아이디어는 LLM에서 불필요한 활성화를 필터링하는 데 있습니다. LoRA 레이어를 사용하여 상위 및 최신 레이어에 대한 공격을 차단할 수 있으며, 이는 다양한 위치에서 특정 기능의 벡터 방향과 직교하는 프로젝션을 학습하는 것을 목표로 하고 있습니다. 이러한 필터는 데이터 중독(trojans)이 삽입되기 전에 깨끗한 데이터셋을 이용한 사후 자기회귀 손실(autoregressive loss)에서 학습됩니다. 이러한 접근 방식은 사전 훈련 중에 삽입된 데이터 중독을 효과적으로 제거할 수 있습니다.

- **Performance Highlights**: LoRA 필터는 특히 잔여 스트림과 최신 레이어에서 가장 잘 작동하는 것으로 나타났습니다. 논문의 결과는 모델의 보통 완성 품질에 대한 영향을 최소화하면서 데이터 중독을 성공적으로 제거할 수 있음을 시사합니다. 실험은 주로 GPT-2 소형 모델을 사용하여 수행되었으며, 이는 추후 더 큰 모델에도 적용될 수 있는 가능성을 엿보게 합니다.



### B'MOJO: Hybrid State Space Realizations of Foundation Models with Eidetic and Fading Memory (https://arxiv.org/abs/2407.06324)
- **What's New**: 이 논문은 B'MOJO라는 새로운 모델 구조를 소개하며, 이 구조는 트랜스덕션 추론을 지원하기 위해 메모리를 효율적으로 활용하는 방법을 제안합니다. 이 모델은 에이데틱 메모리와 페이딩 메모리를 결합하여 활용할 수 있으며, 이는 기존의 Transformer와 SSM(State Space Models) 기반의 하이브리드 모델들보다 나은 성능을 나타냅니다.

- **Technical Details**: B'MOJO는 Stochastic Realization Theory에서 영감을 받아 개발된 모델로, 기본적인 모듈을 활용해 에이데틱 메모리와 페이딩 메모리를 결합합니다. 이 구조는 단기 에이데틱 메모리('in-context'), 영구 구조 메모리('in-weights'), 페이딩 메모리('in-state'), 장기 에이데틱 메모리('in-storage')와 같이 여러 메모리 타입을 네이티브로 포함하여 비동기적으로 업데이트된 메모리를 통한 검색을 통합합니다.

- **Performance Highlights**: B'MOJO는 연관 기억 같은 트랜스덕션 추론 작업에서 기존의 SSM과 하이브리드 모델들을 능가했으며, 언어 모델링에서도 비슷한 크기의 Transformer와 SSM보다 최대 10% 빠른 학습 속도를 보였습니다. 특히 장기 시퀀스 (최대 32K 토큰)에 대한 추론 성능에서도 향상된 결과를 나타냈습니다.



### VIMI: Grounding Video Generation through Multi-modal Instruction (https://arxiv.org/abs/2407.06304)
- **What's New**: 기존의 텍스트-비디오(diffusion) 모델은 텍스트 전용 인코더만을 통해 사전 학습을 진행했습니다. 이로 인해 시각적 기반이 부족하고 다중 모달리티(multimodal) 통합이 제한되었습니다. 이를 해결하기 위해 우리는 검색 방법을 사용하여 큰 규모의 다중 모달 프롬프트 비디오 데이터셋을 구축했습니다. 이로써 다양한 비디오 생성 작업을 단일 모델로 수행할 수 있는 두 단계의 학습 전략을 적용했습니다.

- **Technical Details**: 첫 번째 단계에서는 다중 모달 조건부 비디오 생성 프레임워크를 제안하여 증강된 데이터셋에서 사전 학습을 수행했습니다. 이를 통해 시각적 기반의 비디오 생성의 기초 모델을 구축했습니다. 두 번째 단계에서는 첫 번째 단계에서 학습한 모델을 다양한 비디오 생성 작업에서 미세 조정하여 다중 모달 지시문을 포함시켰습니다. 이를 통해 다양한 입력과 작업을 처리할 수 있는 모델의 능력을 더욱 정교화했습니다.

- **Performance Highlights**: 이 두 단계의 학습 과정을 거친 후, VIMI는 다중 모달 이해 능력을 보여주며 제공된 입력에 기반하여 맥락적으로 풍부하고 개인화된 비디오를 생성할 수 있습니다. 이전의 시각적 기반 비디오 생성 방법과 비교했을 때, VIMI는 일관성 있고 시간적으로 일관된 비디오를 대규모 움직임에서도 생성할 수 있으며, 의미 제어를 유지합니다. 또한, VIMI는 UCF101 벤치마크에서 최첨단의 텍스트-비디오 생성 결과를 달성했습니다.



### ORAN-Bench-13K: An Open Source Benchmark for Assessing LLMs in Open Radio Access Networks (https://arxiv.org/abs/2407.06245)
- **What's New**: 본 논문에서는 Open Radio Access Networks (O-RAN)의 다양한 작업에 대한 효율성과 신뢰성을 크게 높이고자 하는 Large Language Models (LLMs)의 혁신적 적용 가능성을 제시하고 있습니다. 이를 위해 우리는 O-RAN 컨텍스트 내에서 LLM의 성능을 평가하기 위한 첫 번째 종합 벤치마크인 ORAN-Bench-13K를 소개합니다.

- **Technical Details**: ORAN-Bench-13K는 116개 O-RAN 사양 문서에서 생성된 13,952개의 신중하게 큐레이션된 다지선다형 질문들로 구성됩니다. 우리는 세 단계 LLM 프레임워크를 활용하여, 질문을 세 가지 난이도로 분류하여 O-RAN 관련 지식을 폭넓게 다룹니다. 기존의 최첨단 LLM인 Gemini, Chat-GPT, Mistral의 성능을 평가했으며 추가적으로 Retrieval-Augmented Generation (RAG) 기반 파이프라인인 ORANSight를 제안했습니다.

- **Performance Highlights**: 현재 인기 있는 LLM 모델들이 O-RAN에 대해 능숙하지 않다는 것을 발견했으며, RAG 기반의 ORANSight 파이프라인을 통합했을 때 성능이 크게 향상되었음을 관찰했습니다. ORANSight는 Macro Accuracy 0.784와 Weighted Accuracy 0.776을 기록하며, 다른 테스트된 LLM들보다 평균적으로 각각 21.55% 및 22.59% 더 나은 성능을 보였습니다.



### Chronological Analysis of Rigvedic Mandalas using Social Networks (https://arxiv.org/abs/2407.06205)
Comments:
          8 Pages, 4 Tables, 7 Figures

- **What's New**: 전통적인 방법론을 넘어, 이 논문에서는 리그베다(Rig-Veda) 만달라(Mandala)의 내적 연대순을 해석하기 위해 신과 여신에 초점을 맞춘 새로운 접근법을 제안합니다.

- **Technical Details**: Mandala에 대한 텍스트 분석을 수행하기 위해 코사인 유사도(Cosine Similarity)를 기반으로 한 클러스터링 기술(Clustering Techniques)을 적용합니다. 그런 다음, 만달라와 신들 사이의 연관성을 격자 기반의 소셜 네트워크(Social Network)로 표현하여 연대분석에 적합하도록 합니다.

- **Performance Highlights**: 이 분석 방법론을 통해 소셜 네트워크 분석(Social Network Analysis)의 이점을 입증하며, 강과 같은 추가 참조를 분석하여 추가적인 상관관계를 도출할 수 있습니다. 이 접근법은 다른 종류의 참조와 언급을 분석하고 더 실질적인 추론을 이루는데 일반적으로 적용될 수 있습니다.



### A Survey on Mixture of Experts (https://arxiv.org/abs/2407.06204)
- **What's New**: 이번 논문은 전문가 혼합(Mixture of Experts, MoE) 모델에 관한 체계적이고 포괄적인 리뷰를 제공합니다. MoE는 모델 용량을 최소한의 계산 오버헤드로 상당히 확장할 수 있는 효과적인 방법으로 주목받고 있습니다. 저자들은 MoE의 구조를 간단히 소개한 뒤, 새로운 분류 체계를 제안하고 다양한 MoE 모델의 핵심 디자인, 오픈 소스 구현, 하이퍼파라미터 설정 및 실증 평가를 다룹니다.

- **Technical Details**: MoE 레이어는 여러 '전문가 네트워크'와 '게이팅 네트워크'로 구성됩니다. 게이팅 네트워크는 입력을 적절한 전문가 네트워크로 안내하는 역할을 하며, 이를 통해 특정 입력에 맞는 전문가들만 활성화됨으로써 계산 비용을 절감합니다. MoE 레이어는 보통 각 Transformer 블록의 피드포워드 네트워크(FFN)를 대체하는 위치에 배치되어, 모델이 확장됨에 따라 증가하는 계산 비용을 효율적으로 관리합니다.

- **Performance Highlights**: 특히 2024년에 발표된 Mixtral-8x7B와 같은 대규모 산업용 LLM 모델의 등장은 MoE의 성장 궤도를 더욱 견고하게 만들고 있습니다. 이러한 모델들은 Grok-1, DBRX, Arctic 및 DeepSeek-V2 등 다양한 영역에서 활용되고 있으며, 이는 MoE의 유연성과 다재다능성을 잘 보여줍니다. 또한, MoE와 연결된 머신 러닝 시스템 설계 개선을 통해 고품질의 LLM 서비스를 제공하는 데 기여하고 있습니다.



### More Distinctively Black and Feminine Faces Lead to Increased Stereotyping in Vision-Language Models (https://arxiv.org/abs/2407.06194)
- **What's New**: 이 논문에서는 Vision Language Models (VLMs), 특히 GPT-4V가 텍스트와 비전 모달리티를 통합하여 이미지 입력을 처리할 수 있다는 점을 강조합니다. VLM의 이러한 통합은 Large Language Models가 인간의 인식을 더 잘 모방할 수 있게 합니다. 하지만 VLM이 두 모달리티의 편향을 상속받음으로 인해 이러한 편향이 더 널리 퍼지고 완화하기 어려울 수 있다는 우려가 제기됩니다.

- **Technical Details**: 연구에서는 VLM의 인종 및 성별과 관련된 동질성 편향과 특성 연관성을 탐구합니다. 인간 얼굴 이미지를 기반으로 스토리를 작성하도록 요청했을 때 GPT-4V는 하위 인종 및 성별 그룹을 지배적인 그룹보다 더 동질적으로 묘사하고, 일반적으로 긍정적인 고정관념에 의존합니다. 중요한 점은 VLM 고정관념이 그룹 소속보다는 시각적 신호에 의해 주도되며, 특히 더 전형적으로 '흑인'과 '여성적'으로 평가되는 얼굴이 더 큰 고정관념의 대상이 된다는 점입니다.

- **Performance Highlights**: 이러한 발견은 VLM이 인종 및 성별 그룹과 관련된 미묘한 시각적 신호를 고정관념과 연관시키는 방식이 편향 완화에 있어 도전과제가 될 수 있음을 시사합니다. 연구는 이러한 행동의 근본 이유를 탐구하고, 이러한 편향을 해결하는 것이 중요한 과제임을 강조합니다.



### GemmAr: Enhancing LLMs Through Arabic Instruction-Tuning (https://arxiv.org/abs/2407.02147)
- **What's New**: 새로운 연구에서는 주요 자연어 처리(NLP) 모델들이 영어에 비해 아랍어에서 성능이 떨어지는 문제를 해결하기 위해 InstAr-500k라는 새로운 아랍어 인스트럭션 데이터셋(instruction dataset)를 소개했습니다. 이를 통해 아랍어 모델을 더욱 정교하게 튜닝하고, 아랍어 NLP 벤치마크에서 뛰어난 성과를 이끌어내고 있습니다.

- **Technical Details**: 본 연구는 아랍어 데이터셋을 생성하고, 이를 통해 오픈소스 모델 Gemma-7B를 여러 다운스트림 타스크에 맞게 파인튜닝(fine-tuning)하여 아랍어에서의 성능을 개선하는 방법을 제시합니다. 데이터셋 생성에는 합성 데이터(synthetic data) 생성, 인간이 작성한 데이터 수집, 및 LLaMAFactory 프레임워크 내에서 LoRA 기법 사용 등이 포함됩니다.

- **Performance Highlights**: 본 연구에서 개발한 새 모델 GemmAr-7B-V1은 다양한 아랍어 NLP 벤치마크에서 큰 폭의 성능 향상을 보였습니다. 이는 아랍어의 문법적, 의미적 복잡성을 잘 처리할 수 있도록 모델이 정교하게 튜닝되었음을 보여줍니다.



New uploads on arXiv(cs.IR)

### Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspectiv (https://arxiv.org/abs/2407.06992)
Comments:
          Survey paper

- **What's New**: 최근 뉴럴 정보 검색(IR) 모델은 다양한 IR 작업에서 그 효과성을 크게 향상시켰습니다. 본 논문은 이러한 IR 모델들의 강건성 문제를 다루며, 이를 네 가지 측면에서 분석합니다. 특히, 적대적 공격(adversarial attacks)과 분포 외(out-of-distribution, OOD) 시나리오에서의 강건성에 중점을 두어, 밀집 검색 모델(Dense Retrieval Models, DRMs)과 뉴럴 랭킹 모델(Neural Ranking Models, NRMs)을 분석합니다. 또한, 능력 평가를 위한 BestIR 벤치마크를 소개하며, 이 연구가 신뢰할 수 있는 검색 엔진 개발에 도움이 되기를 바랍니다.

- **Technical Details**: IR 시스템의 강건성은 여러 측면에서 평가됩니다. 첫째, 적대적 강건성은 IR 모델이 순위를 변조하려는 공격에 방어할 수 있는 능력을 평가합니다. 둘째, OOD 강건성은 IR 모델이 훈련 데이터와 다른 분포의 새로운 문서 또는 쿼리에서도 잘 일반화하는 능력을 평가합니다. 이를 위해 다양한 데이터셋과 평가 메트릭을 사용합니다. 예를 들어, 적대적 공격과 방어를 다루는 연구 또는 생소한 문서와 쿼리에서의 일반화를 평가하는 연구방법론이 있습니다. 이러한 연구는 뉴럴 IR 모델의 첫 단계(검색 단계)와 두 번째 최종 랭킹 단계에서 모두 적용됩니다.

- **Performance Highlights**: 이 연구는 기존의 강건성 연구들을 체계적으로 정리하고, 다양한 데이터셋과 평가 메트릭을 제시하여, 현재 뉴럴 IR 모델의 강건성 상태를 종합적으로 보여줍니다. BestIR 벤치마크를 통해 여러 강건성 스펙트럼에서 IR 모델의 성능을 평가할 수 있는 구조를 제공합니다. 이러한 접근방법은 뉴럴 IR 모델이 실제 응용에서 신뢰성을 높일 수 있도록 돕습니다.



### Fine-grained large-scale content recommendations for MSX sellers (https://arxiv.org/abs/2407.06910)
- **What's New**: 마이크로소프트(Microsoft)에서 판매자를 위한 새로운 기회 수준(opportunity-level)의 콘텐츠 추천 시스템을 소개합니다. 이 시스템은 판매자가 고객에게 공유할 수 있는 다양한 유형의 콘텐츠(기술 문서, 경쟁사 제품 비교, 고객 성공 사례 등)를 추천해 판매 속도를 높이는 데 도움을 줍니다. 이 논문에서는 이 모델이 어떻게 기회-콘텐츠 조합에서 효율적인 의미적 매칭(semantic matching)을 수행하는지를 설명합니다.

- **Technical Details**: 추천 시스템은 Seismic 콘텐츠 저장소의 약 4만 개의 문서에서 각 기회에 대해 상위 5개의 관련 문서를 추천합니다. 이를 위해 메타데이터 프롬프트 엔지니어링(metadata prompt engineering)을 사용해 Seismic 문서와 기회의 중요한 속성을 요약한 텍스트 기반 설명을 생성합니다. 시스템 아키텍처는 2단계 시스템으로, 빠르지만 약간 부정확한 상위 50개 문서를 가져온 후 교차 인코더 모델로 재순위하는 방식을 따릅니다.

- **Performance Highlights**: 이 시스템은 천문학적인 규모의 기회-콘텐츠 조합을 다루기 위해 일일 배치 모드로 작동하며, 약 10만 개의 기회를 매일 새로 고침합니다. 또한, Seismic 콘텐츠의 변경 사항을 반영하기 위해 매주 임베딩을 새로 고칩니다. 이러한 방식으로 이 시스템은 약 80%의 계산 작업량을 줄여주며, 다수의 인간 도메인 전문가와 'LLM as a judge' 프레임워크를 통해 추천 품질을 정량적으로 평가합니다.



### Analyzing the Effectiveness of Listwise Reranking with Positional Invariance on Temporal Generalizability (https://arxiv.org/abs/2407.06716)
Comments:
          Accepted at CLEF 2024 LongEval track

- **What's New**: 이번 연구는 LongEval 벤치마크를 통해, 실세계 웹 검색 엔진 환경에서 정보 검색(IR) 시스템의 성능을 평가하는 새로운 접근법을 소개합니다. 특히, 지속적으로 업데이트되고 확장되는 문서 집합 내에서 IR 시스템의 시간적 지속성을 측정하는 것이 중요합니다.

- **Technical Details**: 리스트와이즈 재순위화(listwise reranking) 접근법이 주요 초점으로, 이는 시간적 분포 변화로 인한 부정확성을 효과적으로 처리합니다. 이 연구에서 ListT5 모델이 Fusion-in-Decoder 아키텍처를 채택하여 위치 편향 문제를 성공적으로 완화하는 것으로 나타났습니다.

- **Performance Highlights**: ListT5 모델은 특히 시간 이동이 증가할수록 test-long 서브셋에서 더욱 효과적임을 보였습니다. 이는 시간적 지속성 측면에서 IR 시스템의 성능을 입증하며, 실세계 동적 환경에서의 적응성을 강조합니다.



### Embark on DenseQuest: A System for Selecting the Best Dense Retriever for a Custom Collection (https://arxiv.org/abs/2407.06685)
Comments:
          SIGIR2024 demo paper

- **What's New**: 이번 데모에서는 개인 컬렉션에 사용할 효과적인 사전 학습된 밀집 검색기(dense retriever)를 선택할 수 있는 웹 기반 애플리케이션인 DenseQuest를 소개합니다. DenseQuest는 업로드된 대상 컬렉션에 맞춤화된 밀집 검색기 중 최적의 검색기를 예측하고 순위를 매기는 무감독 방식의 기능을 제공합니다. 이 시스템은 대규모 언어 모델(LLMs)로 구동되는 최신 효과적인 방법을 포함한 여러 기존 접근 방식을 구현하며, 질의나 관련성 판단이 필요하지 않습니다.

- **Technical Details**: DenseQuest는 두 개의 독립적인 Docker 컨테이너로 구성되며, 각각 클라우드 인스턴스에 배포됩니다. 웹 기반 프론트엔드는 Vue.js와 Tailwind CSS로 구축되었으며, 백엔드는 Python 웹 프레임워크인 Django로 작성된 REST API로 구성됩니다. 사용자는 컬렉션을 업로드하고 최적의 밀집 검색기 모델을 찾는 요청을 보냅니다. 이 요청은 백엔드로 전달되어 SQLite 데이터베이스에 저장된 후, 작업 큐에 추가됩니다. DenseQuest 코어는 컬렉션을 인코딩하고 모델 선택을 수행한 후 결과를 반환합니다. 사용자는 선택된 모델의 체크포인트를 다운로드할 수 있습니다. 시스템은 AWS를 통해 배포 및 관리되며, EC2 인스턴스를 이용하여 웹 서버와 GPU 연산을 분리해 사용합니다.

- **Performance Highlights**: DenseQuest는 기존의 최첨단 무감독 성능 평가 방법을 통합하여 DR 선택과 순위를 매기는 것을 단순화하며, 사용자에게 최적의 모델을 식별하는 데 필요한 정보를 제공합니다. 예를 들어, Binary Entropy 방법은 각 질의의 바이너리 엔트로피를 계산하여 모델의 불확실성을 평가하며, Query Alteration 방법은 질의어의 변형에 대한 모델의 민감도를 측정합니다. 이러한 다양한 성능 평가 방식을 통해, DenseQuest는 효율적이고 정확한 밀집 검색기 모델 선택을 가능하게 합니다.



### AutoTask: Task Aware Multi-Faceted Single Model for Multi-Task Ads Relevanc (https://arxiv.org/abs/2407.06549)
- **What's New**: 광고의 관련성 모델은 사용자의 검색 쿼리와 광고 제공 간의 관련성을 결정하는 중요한 요소로, 이 문제를 분류 문제로 다룬다. 본 연구에서는 다중 태스크 상황에 적합한 새로운 다중 측면(attention) 모델을 제안한다. 이 모델은 태스크 인식 기능 결합과 태스크 간 상호작용 모델링을 수행하며, 태스크 표현을 위해 태스크 ID 인코딩(Task ID Encoding)을 도입하여 다양한 광고 상황에서 관련성을 정밀하게 모델링할 수 있도록 한다.

- **Technical Details**: 새로운 모델은 언어 모델링(language modeling)과 자기 회귀(attention)를 특징과 태스크 차원에서 결합하여 기능 결합 문제를 해결한다. 이 모델은 태스크 데이터의 무작위 혼합(mixture)을 통해 태스크 블록을 구성하고, 자기 회귀(attention)를 통해 태스크 간 상호작용을 모델링하여 태스크 유사성을 활용한다. 이 방법은 온라인 추론 시 단일 태스크 추론을 가능하게 하며, 모델의 일반화 능력을 향상시킨다.

- **Performance Highlights**: 제안된 모델은 단일 모델로 다양한 태스크를 효과적으로 처리하고, 일반화된 DNN 모델이나 태스크 전용 모델보다 뛰어난 성능을 보여준다. 특히 새로운 태스크에 대한 일반화 능력이 증대되며, 경량화되어 온라인 서빙에 적합하다.



### Empirical analysis of Biding Precedent efficiency in the Brazilian Supreme Court via Similar Case Retrieva (https://arxiv.org/abs/2407.07004)
Comments:
          54 pages, 22 figures

- **What's New**: 이번 연구는 브라질 연방 대법원에서 반복적으로 발생하는 소송을 줄이고자 도입된 '구속력 있는 판례(Súmula Vinculante)'의 효율성을 분석합니다. 이러한 판례들이 새로운 소송을 야기시키는 문제점을 제기하며, 구체적으로 다섯 가지 판례(11, 14, 17, 26, 37)의 법적 영향력을 평가합니다.

- **Technical Details**: 이 연구는 '유사 사례 검색(Similar Case Retrieval)' 기술을 사용하여 판례들이 도입되기 전과 후의 대법원 판결을 비교합니다. 이를 위해 다양한 자연어 처리(NLP) 기법(TF-IDF, LSTM, BERT, regex)을 활용하여 유사 사례를 검색하고, 그 결과를 법적 분석에 활용합니다.

- **Performance Highlights**: 깊은 학습 모델(deep learning models)은 구체적인 '유사 사례 검색' 과제에서 성능이 저조한 것으로 나타났습니다. 반면, 다른 방법들은 법률 문서 검색에서 더 나은 성능을 보였습니다. 또한, 이러한 판례들이 반복적인 소송 대응에서 실패한 이유는 다양하며, 단일 원인으로 규명할 수 없는 것으로 분석되었습니다.



### Positive-Unlabelled Learning for Improving Image-based Recommender System Explainability (https://arxiv.org/abs/2407.06740)
- **What's New**: 본 연구는 시각 기반 추천 시스템(Visual-based Recommender System, RS)의 설명 가능성(Explainability)을 개선하기 위한 새로운 접근법을 제안합니다. 기존 모델들은 다른 사용자가 업로드한 모든 이미지를 부정적인 예제로 간주했으나, 이는 단순화된 가정으로 인해 비효율적일 수 있습니다. 본 연구는 Positive-Unlabelled (PU) 학습 기법을 활용해 사용자의 신뢰할 수 있는 부정적 예제를 선별하여 학습 데이터의 품질을 개선합니다.

- **Technical Details**: 본 연구는 PU 학습을 통해 사용자-업로드 이미지 기반 RS 설명 가능성을 개선합니다. 대상 데이터는 긍정적 예제(이미지 업로드)와 라벨이 없는 데이터(긍정적 또는 부정적일 수 있는 이미지)로 구성됩니다. 두 단계의 유사성 기반 PU 학습 알고리즘을 통해 개인화된 신뢰할 수 있는 부정적 예제를 선택하고, 이를 통해 학습 데이터의 노이즈를 줄여 효율성을 향상시킵니다.

- **Performance Highlights**: PU 기반 접근 방식은 6개의 실제 데이터셋에서 기존의 비-PU 방법보다 더 높은 성능을 보였습니다. 이는 모델의 복잡성을 높이지 않고도 학습 데이터의 품질을 최적화함으로써 시각 기반 RS의 설명 가능성을 향상시킬 수 있음을 보여줍니다.



### Multi-Label Plant Species Classification with Self-Supervised Vision Transformers (https://arxiv.org/abs/2407.06298)
Comments:
          Paper submitted to CLEF 2024 CEUR-WS

- **What's New**: 이번 연구에서는 PlantCLEF 2024 대회를 위한 셀프 슈퍼바이즈드 비전 트랜스포머(Vision Transformer, ViT) DINOv2를 활용한 전이 학습(transfer learning) 접근법을 제시합니다. 특히 다중 라벨 식물 종 분류 문제를 해결하기 위해 기반 및 미세 조정된 DINOv2 모델을 사용하여 일반화된 특징 임베딩(feature embeddings)을 추출합니다. 이 연구는 대규모 데이터셋에서의 계산 복잡성을 해결하기 위해 Spark를 사용하여 분산 데이터 처리(distributed data processing)을 수행합니다. 본 접근법은 이미지의 타일(tile)별 분류 및 예측 집계를 통해 최종 확률 세트를 얻는 방식입니다.

- **Technical Details**: DINOv2는 높은 성능의 비전 트랜스포머 모델로, 고정 크기 패치(token)로 이미지를 처리하여 강력한 특징 표현(feature representations)을 학습합니다. 이 연구에서는 크롭 및 재정렬된 단일 라벨 이미지 데이터셋과 다중 라벨 테스트 데이터셋에서 임베딩을 추출합니다. 이러한 임베딩을 사용해 분류기를 학습시키고, 전체 이미지와 그리드 기반 방법을 통해 다중 라벨 분류 문제를 해결합니다. 또한, Google Cloud Platform(GCP), Apache Spark, Petastorm 및 PyTorch Lightning 등을 활용하여 대규모 데이터셋의 효율적인 처리 및 모델 학습을 수행합니다.

- **Performance Highlights**: 본 연구는 전이 학습과 고급 데이터 처리 기술을 결합하여 다중 라벨 이미지 분류 작업에서 높은 성능을 발휘하였습니다. 특히 DINOv2를 사용하여 추출한 특징 임베딩이 다중 라벨 분류 문제에서의 도메인 전문가 수준의 성능을 달성하였습니다. 또한, Apache Parquet 형식으로 이미지 데이터를 변환하고, 이미지 크롭 및 재정렬을 통해 데이터셋 크기를 크게 줄임으로써 빠른 모델 학습 및 추론이 가능해졌습니다.




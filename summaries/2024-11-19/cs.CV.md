New uploads on arXiv(cs.CL)

### Bi-Mamba: Towards Accurate 1-Bit State Space Models (https://arxiv.org/abs/2411.11843)
- **What's New**: 이번 연구에서는 Bi-Mamba라는 새로운 1-bit Mamba 아키텍처를 소개합니다. 이는 여러 크기로 확장 가능하며, 대형 언어 모델을 더 효율적으로 만들어주는 모델입니다. Bi-Mamba는 자율 회귀 증류 손실을 사용하여 데이터를 기반으로 처음부터 훈련되며, 기존의 Mamba 모델에 비해 메모리 소비와 에너지 소비를 크게 줄입니다.

- **Technical Details**: Bi-Mamba는 Selective State-Space Model(SSM)의 일종으로, 기존 Transformer 모델의 제약을 극복하는 데 도움이 됩니다. Bi-Mamba는 1-bit 양자화(quantization)를 적용하여 훈련 및 추론할 때 높은 성능을 유지하고, 메모리 사용량을 크게 줄이면서도 상대적으로 더 적은 에너지를 소비합니다. 이 연구는 비트 수가 낮은 표현 아래에서의 선형 계산 복잡성을 가진 LLM 프레임워크를 새롭게 제시합니다.

- **Performance Highlights**: 실험 결과, Bi-Mamba는 FP16 또는 BF16을 사용하는 완전 정밀 모델에 필적하는 성능을 보여주며, 전후 훈련 양자화(post-training quantization) Mamba 기준선보다 훨씬 높은 정확도를 기록했습니다. Bi-Mamba 모델은 자원 제한 환경에서도 강력한 기본 모델로 활용될 수 있으며, 쉽게 다른 NLP 응용 프로그램에 맞게 조정될 수 있습니다.



### CNMBert: A Model For Hanyu Pinyin Abbreviation to Character Conversion Task (https://arxiv.org/abs/2411.11770)
Comments:
          9 pages, 2figures

- **What's New**: 본 논문에서는 Hanyu Pinyin 약어를 중국어 문자로 변환하는 새로운 접근법을 제시합니다. CNMBert라는 모델을 도입하여 기존 기존 GPT 모델들을 능가하는 성능을 기록했습니다. 특히, 10,424개의 샘플을 포함한 테스트 데이터셋에서 59.63%의 MRR(Metric Rank Rate)을 달성한 점이 인상적입니다.

- **Technical Details**: CNMBert는 Whole Word Mask Chinese BERT 모델을 기반으로 개발되었습니다. 이 모델은 Hanyu Pinyin의 각 알파벳 문자에 대해 고유한 마스크 토큰을 매핑하여, Fill-Mask 작업으로 접근합니다. 기존 BERT 모델의 단순한 마스크 구조를 확장하여 알파벳 수에 맞는 마스크 토큰을 사용함으로써, 보다 효과적으로 약어를 해석하고 변환하게 됩니다.

- **Performance Highlights**: CNMBert는 주요 성능 지표에서 기존의 다양한 모델보다 뛰어난 결과를 보였습니다. Hanyu Pinyin 약어 변환 작업에 대한 성능 평가에서, 본 모델은 기본 성능을 훨씬 초과하는 결과를 나타냈습니다. 이에 따라 CNMBert는 중국어 정수 교정(Chinese Spelling Correction) 분야에서 확고한 위치를 차지할 것으로 예상됩니다.



### Advacheck at GenAI Detection Task 1: AI Detection Powered by Domain-Aware Multi-Tasking (https://arxiv.org/abs/2411.11736)
- **What's New**: 이 논문은 Advacheck 팀이 GenAI 탐지 작업 1 경쟁에서 기계 생성 텍스트와 인간이 작성한 텍스트를 인식하기 위해 설계한 시스템을 설명합니다. 개발된 시스템은 여러 분류 헤드(heads) 간에 공유된 Transformer Encoder를 사용하는 다중 작업(multi-task) 아키텍처를 기반으로 합니다. 이 접근 방식으로 테스트 세트에서 83.07% 매크로 F1-score를 기록하며 공식 순위에서 1위를 차지했습니다.

- **Technical Details**: 논문의 시스템은 기계 생성 텍스트를 탐지하는 이중 텍스트 분류 작업으로 정의됩니다. 데이터는 수천 개의 도메인과 생성 모델을 포함하여 noisy한 특성을 가지고 있어, 다중 작업 학습(multi-task learning)을 활용하여 중요한 피처에 집중할 수 있도록 설계되었습니다. 제안된 모델은 하드 파라미터 공유(hard parameter sharing) 방식으로 여러 작업을 동시에 처리하는 구조를 가지고 있습니다.

- **Performance Highlights**: Advacheck 시스템은 테스트 세트에서 83.07%의 매크로 F1-score를 달성하여 다른 참가자들의 접근 방식을 능가했습니다. 특히, 기본선(baseline)을 10% 초과하여 기록함으로써 성공적인 성능 향상을 보여주었습니다. 경험적 분석을 통해 다중 작업 학습이 단일 작업과 비교하여 더 나은 결과를 보임을 확인했습니다.



### Moral Persuasion in Large Language Models: Evaluating Susceptibility and Ethical Alignmen (https://arxiv.org/abs/2411.11731)
- **What's New**: 최근 논문에서는 대형 언어 모델(LLMs)들이 윤리적 프레임워크에 따라 초기 결정을 변경하도록 유도되는 방법을 탐구하고 있습니다. 두 가지 실험을 통해 LLM이 도덕적 설득에 얼마나 영향을 받는지를 평가했습니다. 첫 번째 실험은 기본 에이전트의 결정과 설득자의 개입을 비교하였고, 두 번째 실험은 LLM이 특정 윤리적 프레임워크에 어떻게 맞춰지는지를 조사했습니다.

- **Technical Details**: 이 연구는 도덕적으로 모호한 상황에서 LLM의 결정 과정을 탐구합니다. 첫 번째 단계에서는 기본 에이전트(LML)와 설득자 에이전트의 초기 점수를 비교하는 기준 평가를 진행하고, 두 번째 단계에서는 각 상황에 대한 대화 후 변화를 분석합니다. 이 실험은 대화의 턴 수나 사용된 LLM의 종류에 따른 영향을 테스트합니다.

- **Performance Highlights**: MGN에 따른 실험 결과, LLM은 윤리적이거나 모호한 상황에서 설득될 수 있으며, 이 설득의 성공 여부는 모델의 종류, 시나리오의 복잡성, 대화의 길이에 따라 달라진다는 것이 분명해졌습니다. 특히, 같은 회사의 서로 다른 크기의 LLM에서 상이한 결과가 나와, LLM의 도덕적 설득에 대한 민감도가 상당히 다름을 강조했습니다.



### FedCoLLM: A Parameter-Efficient Federated Co-tuning Framework for Large and Small Language Models (https://arxiv.org/abs/2411.11707)
- **What's New**: FedCoLLM은 서버의 LLM(대용량 언어 모델)과 클라이언트의 SLM(소형 언어 모델)을 동시에 조정하는 혁신적인 분산 학습 프레임워크입니다. 이 프레임워크는 클라이언트의 데이터 프라이버시를 지키면서 효과적으로 지식을 전달할 수 있는 경량 어댑터를 사용합니다. FedCoLLM의 도입으로 인해 클라이언트의 SLM 성능이 크게 향상되었고, LLM도 클라이언트의 도메인 데이터로부터 유익함을 얻을 수 있게 되었습니다.

- **Technical Details**: FedCoLLM은 파라미터 효율적 파인튜닝(PEFT)과 지식 증류(Knowledge Distillation) 기술을 기반으로 합니다. 이 프레임워크는 클라이언트와 서버 사이의 효율적인 지식 교환을 위해 SLM과 함께 경량 어댑터를 사용합니다. 또한, FedCoLLM은 클라이언트의 데이터 프라이버시를 보장하기 위해 연합 학습(FL) 메커니즘을 활용합니다.

- **Performance Highlights**: 다양한 NLP(자연어 처리) 텍스트 생성 작업을 통해 FedCoLLM의 성능을 평가한 결과, 클라이언트의 SLM이 LLM의 지원 하에 유의미한 성능 향상을 경험했고, LLM 또한 클라이언트의 데이터에 직접 파인튜닝했을 때와 유사한 성능을 달성했습니다. FedCoLLM은 자원 소모를 줄이면서 더 나은 성과를 보여 주었습니다.



### Technical Report: Enhancing LLM Reasoning with Reward-guided Tree Search (https://arxiv.org/abs/2411.11694)
Comments:
          LLM;Complex Reasoning;Math

- **What's New**: 최근 연구에서는 테스트 시간에 모델 성능을 높이기 위한 방법으로, LLM에 대해서 보상 기반 트리 탐색 알고리즘을 통한 추론 능력 강화를 제안하고 있습니다. 이 전략은 정책 모델, 보상 모델, 탐색 알고리즘의 통합으로 구성되며, 특히 동적으로 확장되는 트리를 탐색하는 방식으로 설계되었습니다. 이 프레임워크는 LLM의 수학적 문제 해결을 크게 향상시키는 효과를 보였으며, 다양한 기술적 고려사항이 포함되어 있습니다.

- **Technical Details**: 이 연구는 주로 수학적 도메인에 초점을 맞추고 있으며, 세 가지 주요 구성 요소인 정책 모델, 보상 모델 및 탐색 알고리즘으로 이루어진 보상 유도 트리 탐색 프레임워크를 구현하고 있습니다. 정책 모델은 탐색 트리 내의 주어진 경로를 따라 부분 솔루션 접두사에 기반하여 새로운 추론 단계를 생성합니다. 탐색 알고리즘은 이 과정을 구조화하며, 보상 모델은 정책 모델의 행동과 탐색 과정에 대한 피드백 신호를 제공합니다.

- **Performance Highlights**: 연구진은 MATH-OAI, GSM-Hard, Olympiad Bench, College Math 등 4개의 도전적인 수학 벤치마크 데이터셋에 대해 광범위한 평가를 실시하였으며, 실험 결과 정책 모델의 성능이 크게 향상됨을 입증했습니다. 또한, 정책 모델, 보상 모델 및 트리 탐색 알고리즘 설계에 대한 심층적인 실증 분석을 제공하여 향후 연구자들에게 의미 있는 지침을 제시하고자 하였습니다.



### Chapter 7 Review of Data-Driven Generative AI Models for Knowledge Extraction from Scientific Literature in Healthcar (https://arxiv.org/abs/2411.11635)
Comments:
          16 pages, 5 figures, 1 table

- **What's New**: 이 논문은 추상적 자연어 처리(NLP) 기반의 텍스트 요약 기법의 발전을 검토하고 기존의 추출 기반 요약 기법과 비교합니다. 1950년대부터 시작된 텍스트 요약의 역사를 간략히 소개하며, Bidirectional Encoder Representations from Transformer (BERT)와 Generative Pre-training Transformers (GPT)와 같은 사전 훈련 언어 모델의 도입을 다룹니다.

- **Technical Details**: 총 60개의 연구가 PubMed와 Web of Science에서 확인되었으며, 이 중 29개는 제외되고 24개를 평가하여 적격성을 판단하였습니다. 결국 7개의 연구가 보다 심층적으로 분석되었습니다. 논문에서는 GPT-3와 최신 기술인 GPT-4 솔루션 간의 과학적 텍스트 요약 비교 사례도 포함되어 있습니다.

- **Performance Highlights**: 현재 자연어 처리는 짧은 텍스트 요약 생성에 있어 아직 완전한 잠재력을 발휘하지 못하고 있습니다. 특정 문제에 대한 우려가 인정되고 있으며, 이러한 모델의 실제 도입이 점진적으로 이루어질 것으로 기대됩니다.



### Federated Incremental Named Entity Recognition (https://arxiv.org/abs/2411.11623)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 Federated Incremental NER (FINER) 설정을 정의하고, Local-Global Forgetting Defense (LGFD) 모델을 통해 이점이 상이한 잊음 문제를 완화하고자 했습니다. 기존의 Federated NER (FNER) 방식은 엔티티 타입과 로컬 클라이언트가 정적으로 고정되어 있다고 가정하는데, 이는 실제 동적 상황에서는 효과적이지 못합니다. 우리는 각 로컬 클라이언트가 새 엔티티 타입을 지속적으로 받고, 새로운 클라이언트가 불규칙하게 참여하는 상황을 해결하기 위한 방법을 제안합니다.

- **Technical Details**: LGFD 모델은 intra-client 잊음을 해결하기 위해 구조적 지식 증류 손실을 개발하여 기존 모델의 구조를 새로운 모델에 전달합니다. 또한, 비즈니스에서 새로운 엔티티 타입을 자동으로 인식할 수 있는 작업 전환 모니터를 제안하여, 개인정보 보호를 고려하여 최신의 이전 글로벌 모델을 저장하고 지식 증류 및 퍼지 레이블링을 수행합니다. 이러한 기술적 발전은 과거의 지식을 유지하면서도 새로운 엔티티 타입을 효율적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, LGFD 모델은 기존 방법들에 비해 유의미한 개선을 보여주었습니다. 특히, intra-client 및 inter-client 잊음 문제를 동시에 해결하여 기존 학습으로부터 얻은 지식을 효과적으로 보존하는 데 큰 도움이 되었습니다. 이러한 성과는 의료 분야와 같은 민감한 데이터를 다룰 때 더욱 중요해지며, 제안된 모델은 실제 응용 가능성을 높였습니다.



### OASIS: Open Agents Social Interaction Simulations on One Million Agents (https://arxiv.org/abs/2411.11581)
- **What's New**: OASIS는 다양한 소셜 미디어 플랫폼을 시뮬레이션할 수 있는 일반화 가능하고 확장 가능한 큰 언어 모델(LLM) 기반의 에이전트 모델을 제공합니다. 기존의 에이전트 기반 모델(ABM)들은 특정 상황에 맞춰 설계되어 있어 다른 현상에 대한 탐구가 번거롭고 자원 소모가 매우 컸습니다. OASIS는 최대 100만 사용자까지 모델링할 수 있는 대규모 사용자 시뮬레이션을 지원하며, 이를 통해 복잡한 사회적 시스템을 보다 효과적으로 연구할 수 있는 도구로 자리 잡고 있습니다.

- **Technical Details**: OASIS는 환경 서버(Environment Server), 추천 시스템(Recommendation System), 에이전트 모듈(Agent Module), 시간 엔진(Time Engine), 확장 가능한 추론기(Scalable Inferencer)의 다섯 가지 주요 구성 요소로 이루어져 있습니다. 각 에이전트는 개인 정보를 가지고 등록 후, 추천 시스템을 통해 가장 관심 있는 콘텐츠를 수신받으며, 이를 바탕으로 행동을 선택합니다. 에이전트의 행동은 시간 엔진에 의해 시간에 따라 조정되며, 이러한 모든 컴포넌트는 다양한 소셜 미디어 플랫폼에 쉽게 적용할 수 있도록 설계되었습니다.

- **Performance Highlights**: OASIS는 정보 확산, 그룹 편향화, 그리고 군중 효과와 같은 다양한 사회 현상들을 재현할 수 있는 능력을 보여줍니다. 실험 결과, 에이전트의 수가 증가할수록 그룹 차원에서의 동적 행동이 더욱 돋보이며, 에이전트의 의견 다양성과 유용성이 증가하는 경향을 보였습니다. OASIS가 이러한 사회적 현상을 정확하게 모방함으로써 복잡한 디지털 환경에서의 사회적 상호작용을 연구하는 데 강력한 도구가 될 수 있음을 입증하고 있습니다.



### Addressing Hallucinations in Language Models with Knowledge Graph Embeddings as an Additional Modality (https://arxiv.org/abs/2411.11531)
- **What's New**: 이번 논문에서는 Knowledge Graphs (KGs)를 추가적인 모달리티로 포함하여 대형 언어 모델(Large Language Models, LLMs)에서의 환각(hallucination)을 줄이는 방법을 제안합니다. 입력 텍스트를 KG 임베딩으로 변환하고, 이를 언어 모델 공간에 통합하는 어댑터(adapter)를 사용하는 접근법을 소개하며, 외부 검색 프로세스에 의존하지 않습니다.

- **Technical Details**: 이 연구를 위해 300만 개 이상의 위키피디아 텍스트가 엔티티(entity)와 함께 주석 처리된 WikiEntities 데이터셋을 만들었습니다. 이 데이터셋은 엔티티 링크(Entity Linking) 모델을 훈련시키고 다양한 LLM에 어댑터를 적용하는 데 유용한 자원입니다. 모델을 세부 조정(fine-tuning)하지 않고 어댑터만 훈련하여 다른 작업의 성능이 영향을 받지 않도록 설계되었습니다.

- **Performance Highlights**: Mistral 7B, LLaMA 2-7B ( 챗 ), LLaMA 3-8B ( 지시 ) 모델에 어댑터를 훈련시키고, HaluEval, True-False 벤치마크 및 FEVER 데이터셋에서 성능 향상을 입증하였습니다. 그 결과, KGs를 새로운 모달리티로 통합하는 것이 환각을 효과적으로 줄이고 언어 모델의 사실 적 정확성(factual accuracy)을 개선하는 데 도움을 줄 수 있음을 보여주었습니다.



### Safe + Safe = Unsafe? Exploring How Safe Images Can Be Exploited to Jailbreak Large Vision-Language Models (https://arxiv.org/abs/2411.11496)
- **What's New**: 최근 대형 비전-언어 모델(LVLM)들은 다양한 실제 응용 프로그램에서 뛰어난 추론 능력을 보여주며 큰 발전을 이뤘습니다. 하지만, LVLM의 안전 장치가 예기치 않은 시각적 모달리티로 인해 발생하는 불안전한 콘텐츠 생성의 위험을 완전히 커버하지는 못합니다. 이 연구에서는 안전한 이미지를 활용한 jailbreak (탈옥) 공격 가능성을 제시하고, 이를 통해 LVLM의 안전성 문제를 드러내고자 합니다. 또한, 새로운 프레임워크인 Safety Snowball Agent (SSA)를 제안하여 LVLM의 취약점을 활용하는 방법을 제시합니다.

- **Technical Details**: 본 연구는 LVLM이 보유한 두 가지 기본 특성인 보편적 추론 능력(universal reasoning capabilities)과 안전 눈덩이 효과(safety snowball effect)를 중점적으로 다룹니다. SSA는 초기 응답 생성(initial response generation)과 해로운 눈덩이 효과(harmful snowballing) 두 단계로 운영됩니다. 초기 단계에서는 도구를 사용하여 잠재적으로 해로운 목표에 맞춘 jailbreak 이미지를 생성하거나 검색하고, 두 번째 단계에서는 점진적으로 더 해로운 출력을 유도하기 위해 정제된 후속 프롬프트를 사용합니다. 이러한 접근 방식은 LVLM이 안전하게 보인다 하더라도 해로운 콘텐츠를 생성할 수 있는 가능성을 열어줍니다.

- **Performance Highlights**: 실험 결과, SSA는 거의 모든 이미지를 활용하여 LVLM에서 위험한 콘텐츠를 생성할 수 있으며, 최신 LVLM인 GPT-4o에 대해서도 88.6%의 높은 성공률로 jailbreak 공격을 수행할 수 있음을 보여줍니다. 또한, SSA는 입력이 안전한 텍스트와 이미지로 유지되면서도 온라인 서비스에서 일반적으로 사용하는 콘텐츠 모더레이터(Content Moderator)를 성공적으로 우회할 수 있습니다. 이러한 결과는 유 generative multimodal systems에서의 AI 안전성 확보에 있어 새로운 도전 과제를 제기합니다.



### Quantifying Preferences of Vision-Language Models via Value Decomposition in Social Media Contexts (https://arxiv.org/abs/2411.11479)
- **What's New**: 이 논문은 Vision-Language Models (VLMs)의 발전과 함께 가치 기반 평가의 필요성을 강조하고, Value-Spectrum이라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 VLMs의 응답 차이를 분석할 수 있는 데이터셋으로, 50,000개 이상의 짧은 비디오를 수집하여 Schwartz의 가치 차원을 기반으로 하는 평가를 제공합니다. 이를 통해 VLM들이 추구하는 가치와 그릇된 성향에 대한 새로운 인사이트를 제공합니다.

- **Technical Details**: Value-Spectrum은 글자 기반 검색을 통해 VLM의 가치 관련 특성을 드러내는 것을 목표로 합니다. 연구자는 TikTok, YouTube Shorts 및 Instagram Reels에서 수집된 50,191개의 비디오를 기반으로 하여 각 비디오가 특정 가치 차원에 정렬되도록 하였습니다. 또한, VLM의 반응을 평가하기 위한 자동화된 비디오 분석 파이프라인을 개발하였습니다.

- **Performance Highlights**: VLMs의 벤치마킹 결과, 대부분의 모델은 쾌락주의적(Hedonism) 주제에 대한 선호가 강하게 나타났습니다. CogVLM과 Qwen-VL-Plus는 모든 가치 차원에서 폭넓은 일치를 보였지만, 각 모델의 일치는 현저히 달랐습니다. 특히, Claude-3.5는 ISQ 전략을 통해 큰 개선을 보였고, Blip-2는 개선이 없음으로 나타나 VLM의 적응성에서 fundamental differences를 드러냈습니다.



### Membership Inference Attack against Long-Context Large Language Models (https://arxiv.org/abs/2411.11424)
- **What's New**: 본 논문은 Long-Context Language Models (LCLMs)에 대한 기존 연구의 공백을 메우기 위해, 이 모델들이 민감한 데이터를 포함할 수 있는 여러 공격 벡터를 새롭게 제시합니다. 특히, Membership Inference Attacks (MIAs)라는 공격을 통해, 주어진 문서가 LCLMs의 입력 컨텍스트에 포함되어 있는지를 평가하고, 이로 인해 발생할 수 있는 개인 정보 유출의 위험을 강조합니다. 또한, 논문에서 제안한 MIA 방법론은 LCLMs의 고유한 특성을 반영하여 개발되었습니다.

- **Technical Details**: LCLMs는 기존의 언어 모델보다 긴 문맥을 처리할 수 있는 능력이 있으며, 데이터 검색의 효율성을 높이는 로직과 구조적 개선을 통해 성능을 향상시킵니다. 본 논문에서는 LCLM에 대해 6가지 종류의 MIAs를 제안하며, 각 접근 방식은 생성 손실(generation loss)이나 의미 유사성(semantic similarity)을 활용합니다. 이러한 방법을 통해 입력된 문서의 기밀성을 평가하고, 언어 모델의 응답에서 발생할 수 있는 기밀 정보 유출 위험을 측정합니다.

- **Performance Highlights**: 실험 결과, 제안된 MIAs는 다양한 LCLM 모델에서 뛰어난 성능을 발휘하며, 예를 들어 Multi-document QA 데이터 세트에서 90.66%의 F1-score를 기록했습니다. 이러한 결과는 LCLMs의 입력 컨텍스트 내에서 회원 정보가 유출될 가능성을 시사하며, 문서가 모델의 컨텍스트에 포함되는지를 정확히 추측할 수 있는 방법을 제시합니다. 따라서 본 연구는 LCLMs의 개인 정보 보호에 대한 새로운 통찰을 제공합니다.



### Rethinking Thinking Tokens: Understanding Why They Underperform in Practic (https://arxiv.org/abs/2411.11371)
- **What's New**: 본 연구에서는 Thinking Tokens (TTs)라는 비지도 학습 방법이 언어 모델에서의 추론을 촉진할 수 있다고 제안했습니다. 그러나 TTs는 Chain-of-Thought (CoT) 추론에 비해 성능이 일관되게 낮다는 사실을 발견했습니다.

- **Technical Details**: TTs는 중간 '사고' 토큰을 도입하여 모델이 출력을 생성하기 전에 내부적으로 더 깊은 계산을 할 수 있도록 합니다. 하지만 단일 임베딩에 의존함으로써 발생하는 일관성 없는 학습 신호는 학습을 방해하고 노이즈가 포함된 그래디언트를 초래합니다.

- **Performance Highlights**: TTs는 여러 벤치마크에서 CoT 추론과 비교하여 성능이 개선되지 않고 일관되게 저조한 결과를 보였습니다. 이 연구는 TTs의 저조한 성능을 분석하고 향후 연구 방향에 대한 논의를 제공합니다.



### Mitigating Knowledge Conflicts in Language Model-Driven Question Answering (https://arxiv.org/abs/2411.11344)
- **What's New**: 이번 연구에서는 문서 질문 답변과 요약 생성 등 지식 기반 시퀀스-투-시퀀스(nird)+ 생성 작업에서 공통적으로 나타나는 명백한 문제인 "hallucination"을 다룹니다. 저자들은 입력 소스와 생성된 내용 간의 명확한 상관관계를 통해 이러한 현상을 완화할 수 있다고 주장하며, 특히 질문 답변 과정에서의 엔티티 기반 지식 충돌이 모델의 동작에 미치는 영향을 분석합니다.

- **Technical Details**: 지식 충돌(knowledge conflicts)을 정의하고 이를 해결하기 위한 간단한 데이터 증강(data augmentation) 기반 솔루션을 제안합니다. Seq2seq 모델(BART 등)을 사용하여 훈련 데이터셋이 긴 문맥을 담고 있을 경우 발생하는 어려움을 해결하기 위해 사전 훈련된 생성 언어 모델을 손질(fine-tune)하여 QA 작업에 사용할 수 있도록 조정했습니다. 또한, 주어진 질문(q)와 컨텍스트(c)에 대한 응답을 생성하는 과정에서 모델이 컨텍스트를 무시하고 직접적인 추정(p_theta(x|q))을 하는 문제를 다룹니다.

- **Performance Highlights**: 연구 결과, 제안된 방법을 통해 모델의 생성된 출력이 입력 컨텍스트에 대해 더욱 충실하게 되며, 사실적 일관성을 유지한다고 보장할 수 있음을 보여줍니다. 모델의 성능 저하를 최소화하면서도 'hallucination'을 방지하기 위한 두 가지 방법, 즉 gradient 기반 디코딩과 adapter 기반 파인튜닝을 구현하였습니다. 이는 QA 모델에서 사용자와의 신뢰성을 높이고, 불필요한 AI 모델 혼란을 줄이는 데 기여할 것으로 기대합니다.



### Transcending Language Boundaries: Harnessing LLMs for Low-Resource Language Translation (https://arxiv.org/abs/2411.11295)
- **What's New**: 본 논문에서는 저자원이용 가능 언어 (low-resource language)의 번역 품질을 향상시키기 위해 retrieval-based 방법론을 제안합니다. 이 방법은 키워드 중심의 번역을 통해 기존 데이터에서 예시를 검색하는 방식을 활용하여 번역 품질을 높이는 데 중점을 두고 있습니다. 저자들은 체로키어, 티베트어, 만주어와 같은 세 가지 저자원 언어에 대한 실험을 통해 본 방법의 효용을 평가합니다.

- **Technical Details**: 이 연구는 영어에서 체로키어로 번역하는 과정에서 GPT-4o 및 LLaMA 3.1 모델들을 비교합니다. 실험에서는 BLEU, ROUGE, BERTScore와 같은 자동 평가 지표와 전문가의 인간 평가를 사용하여 번역 품질을 측정하였습니다. 제안된 retrieval-augmented generation (RAG) 기법을 통해 모델이 저자원 언어의 번역에서 나타나는 일반화 문제를 해결하고자 하였습니다.

- **Performance Highlights**: 결과적으로, 본 연구의 방법은 어휘 수준의 정확성과 전체 의미 이해에서 상당한 개선을 보여주었습니다. GPT-4o 및 LLaMA 3.1의 제로샷 성능과 비교할 때, 제안된 접근법은 저자원 언어 번역 문제를 해결하는 데 있어 긍정적인 결과를 도출했습니다. 이는 저자원 언어로의 번역을 향상시키고 궁극적으로 문화와 언어의 보존에도 기여할 수 있는 가능성을 시사합니다.



### LP Data Pipeline: Lightweight, Purpose-driven Data Pipeline for Large Language Models (https://arxiv.org/abs/2411.11289)
- **What's New**: 본 논문은 Lightweight, Purpose-driven (LP) Data Pipeline이라는 CPU 기반 데이터 처리 프레임워크를 소개합니다. 기존의 GPU 의존 방식에 비해, 이 새로운 파이프라인은 시간과 비용 절감을 통해 고품질 데이터셋을 제공합니다. 또한 도메인 및 언어에 맞춘 데이터셋 생성을 가능하게 하여 LLM의 적용 범위를 확장하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: LP Data Pipeline은 네 가지 핵심 원칙에 기반하여 설계되었습니다. 이 파이프라인은 초기 필터링과 데이터 정화를 먼저 수행하고, 자원 집약적인 작업은 이후 단계에서 처리함으로써 효율성을 극대화합니다. 또한, FastText 기반의 언어 식별 도구를 활용하여 언어별 데이터 생성을 지원하며, 정보 업데이트를 자동화하는 워크플로우 관리 도구인 Airflow를 통합합니다.

- **Performance Highlights**: 실험을 통해 LP Data Pipeline이 대규모 데이터 처리를 위한 시간을 단축시키고, 비용을 효과적으로 절감하는 능력을 보여줍니다. 특히, 특정 도메인을 위한 데이터셋 생성 시, 필요한 법적 용어나 사례를 반영하여 LLM의 성능을 향상시키는 데 중점을 두었습니다. 이로 인해 개발자들은 보다 전문화된 언어 모델을 효율적으로 구축할 수 있습니다.



### VersaTune: Fine-Tuning Multi-Ability LLMs Efficiently (https://arxiv.org/abs/2411.11266)
- **What's New**: 이 논문에서는 LLMs(Large Language Models)의 멀티-도메인 성능을 향상하기 위한 새로운 데이터 구성 프레임워크인 VersaTune을 소개합니다. VersaTune은 기존의 지식 분포에 기반하여 각 도메인별 데이터 혼합 비율을 조정하며, 이를 통해 Catastrophic Forgetting 문제를 해소할 수 있도록 설계되었습니다. 이는 모델이 특정 도메인에서만 최적화되는 기존 연구와 달리, 다양한 도메인에서의 성능 저하를 최소화하며 멀티태스킹 능력을 향상시킵니다.

- **Technical Details**: VersaTune은 특정 도메인에 맞춘 데이터 구성 전략을 통해 LLMs의 파인튜닝(튜닝) 과정에서 지식 분포를 동적으로 감지합니다. 이 과정은 도메인 간의 데이터 비율을 유연하게 조정하여, 학습이 원활하게 진행될 수 있도록 합니다. 결과적으로, VersaTune은 12개의 다양한 도메인에 대해 35.21%의 성능 향상을 달성하며, 특정 도메인 최적화 시 다른 도메인에서 성능 저하를 38.77% 줄이는 성과를 보입니다.

- **Performance Highlights**: 실험 결과 VersaTune은 법률, 의학, 금융, 과학, 코드 등 다양한 도메인에서 눈에 띄는 성능 개선을 보였습니다. 특히, 특정 도메인 파인 튜닝 시 목표 도메인에서의 학습 효과를 유지하면서도 비목적 도메인에서의 성능 저하를 줄였습니다. 이러한 결과는 LLMs의 멀티태스킹 능력이 지식 분포에 따라 최적화될 수 있음을 제시합니다.



### Large corpora and large language models: a replicable method for automating grammatical annotation (https://arxiv.org/abs/2411.11260)
- **What's New**: 이 논문에서는 언어학(Linguistics) 연구에서 대량의 텍스트 코퍼스(Text Corpora)에서 추출한 특성에 대한 주석(Annotation) 데이터셋의 필요성을 다루고 있습니다. 특히, 대량의 데이터 샘플을 수동으로 주석 처리하는 데 어려움이 크다는 기존 문제를 해결하기 위해, 대형 언어 모델(Large Language Models)을 활용한 감독(Supervised) 방법을 제시합니다. 이 방법은 주어진 문맥에서 문법 주석을 지원하기 위해 프로프트 엔지니어링(Prompt Engineering), 훈련(Training), 평가(Evaluation)를 포함합니다.

- **Technical Details**: 저자들은 영어 평가 동사구 'consider X (as) (to be) Y'의 형식적 변이를 사례 연구로 하여, Claude 3.5 Sonnet 대형 언어 모델과 Davies의 NOW, EnTenTen21 코퍼스 데이터를 통해 방법론적 파이프라인(Methodological Pipeline)을 제안했습니다. 이 시스템은 소량의 훈련 데이터만으로도 90% 이상의 모델 정확도(Model Accuracy)를 달성할 수 있어, 향후 대량의 토큰(Token) 주석 작업을 Validating 할 수 있는 충분한 가능성을 보여줍니다.

- **Performance Highlights**: 이 연구의 중요성은 대량의 데이터를 다룰 수 있는 새로운 주석 접근 방식을 제공하며, 언어구조와 문법적 변이(Grammatical Variation) 및 변화(Change)에 대한 보다 광범위한 사례 연구에 대한 결과의 일반화 가능성을 논의합니다. AI 코파일럿(AI Copilots)이 향후 언어학 연구를 위한 유용한 도구로 작용할 가능성을 강조하며, 이를 통해 연구자들은 효율적인 분석이 가능해질 것입니다.



### ZeFaV: Boosting Large Language Models for Zero-shot Fact Verification (https://arxiv.org/abs/2411.11247)
Comments:
          This pre-print has been published in PRICAI 2024: Trends in Artificial Intelligence. The published version is available at this https URL

- **What's New**: 이 논문에서는 ZeFaV라는 프레임워크를 제안하여 대형 언어 모델의 사실 검증 작업의 성능을 향상시킵니다. ZeFaV는 zero-shot 학습 기반으로 작동하며, 주장(claim)과 증거(evidence) 간의 관계를 추출하고 정보를 재구성하는 방식을 사용합니다. 이를 통해 기존의 사실 검증 시스템의 한계를 극복하고, 더 나아가 성능을 크게 향상시킬 수 있음을 나타냅니다.

- **Technical Details**: ZeFaV는 세 가지 주요 단계로 구성됩니다. 첫 번째, FewRel 데이터셋을 바탕으로 LLM을 미세 조정하여 관계 추출을 수행합니다. 두 번째, 주장의 증거 관계를 식별하고 중복 관계를 제거하는 폐쇄 정의를 사용하여 정보를 정리합니다. 마지막으로, InfoRE 기법을 활용하여 증거를 보다 유용한 형식으로 구조화하여 LLM의 준맥락 학습(in-context learning)을 강화합니다.

- **Performance Highlights**: ZeFaV는 HoVer 및 FEVEROUS-S 데이터셋에서 F1 점수 기준으로 기존의 다른 기술들과 비교했을 때 우수한 성능을 보였습니다. 특히 FEVEROUS-S 데이터셋에서는 ZeFaV가 86.54%의 F1 점수를 기록해 ProgramFC와 InfoRE보다 높은 성과를 냈습니다. 실험 결과, ZeFaV는 zero-shot 학습 기반으로도 효율적인 사실 검증 작업을 수행할 수 있음을 입증했습니다.



### MEMO-Bench: A Multiple Benchmark for Text-to-Image and Multimodal Large Language Models on Human Emotion Analysis (https://arxiv.org/abs/2411.11235)
- **What's New**: 이 연구에서는 MEMO-Bench라는 새로운 벤치마크를 소개하여 AI 모델의 감정 분석 능력을 평가하는 데 집중하고 있습니다. MEMO-Bench는 12개의 T2I 모델에서 생성된 7,145개의 감정 표현 초상화를 포함하고 있으며, 이는 기존 연구와 비교하여 감정 이해의 범위를 더욱 넓히고 있습니다. 이를 통해 T2I 모델과 MLLM의 감정 분석 성능을 포괄적으로 평가할 수 있는 기초 자료를 제공합니다.

- **Technical Details**: MEMO-Bench는 6개의 주요 감정(행복, 슬픔, 분노, 놀람, 걱정, 중립)을 기반으로 구축되었습니다. 각 감정에 대해 100개의 프롬프트가 설계되어 T2I 모델에 제공되어 감정 표현을 생성합니다. 연구에서는 감정 분석 방법론으로 coarse-grained 및 fine-grained 분석이라는 점진적 평가 접근 방식을 사용하여 MLLM의 감정 인식 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 기존 T2I 모델은 긍정적인 감정을 생성하는 데 있어 더 높은 효과를 보였지만, 부정적인 감정 생성에는 한계가 있음을 보여주었습니다. MLLM은 대체로 감정을 분류하는 데 효과적이지만, 미세한 감정 분석에서는 인간 수준의 정확도를 보장하지 못했습니다. 이러한 발견은 AI가 인간 감정을 완전히 이해하는 데 있어 현재의 제한 사항을 나타내며, 감정 인식 AI 시스템의 발전에 중요한 통찰을 제공합니다.



### Capturing Sparks of Abstraction for the ARC Challeng (https://arxiv.org/abs/2411.11206)
Comments:
          Submitted as a paper entry for the 2024 ARC Prize

- **What's New**: 이 연구는 LLM이 주어진 문제를 해결할 수 있는 코드 솔루션과 이에 대한 설명을 통해 LLM의 '이해'를 향상시키고, 이를 바탕으로 ARC(Abstraction and Reasoning Challenge) 문제를 해결하는 새로운 접근법을 제시합니다.

- **Technical Details**: LLM은 arc-dsl-llm이라는 LLM-가독성이 높은 DSL(Domain-Specific Language)로 작성된 코드 솔루션을 바탕으로 문제를 해결하는 데 도움을 받습니다. 이 DSL은 기존의 Hodel이 만든 arc-dsl에서 개선되어 LLM의 이해도를 높였습니다. 주요 목표는 LLM의 출력을 통해 생성되는 '고차 수준'의 추상 개념을 추출하는 것입니다.

- **Performance Highlights**: 연구 결과, LLM은 주어진 코드 솔루션을 통해 문제를 해결하는 과정에서 높은 수준의 전략적 사고를 보였으며, 이 샘플 데이터를 LLM의 Fine-Tuning에 사용하거나 RAG 시스템을 활용해 실시간 문제에 주입할 수 있는 가능성이 확인되었습니다.



### LL\"aMmlein: Compact and Competitive German-Only Language Models from Scratch (https://arxiv.org/abs/2411.11171)
Comments:
          first draft; this https URL

- **What's New**: 본 논문에서는 독일어 전용 디코더 모델인 LLäMmlein 120M과 1B를 처음부터 끝까지 투명하게 생성하고, 이를 독일 NLP 연구 커뮤니티가 사용할 수 있도록 공개합니다.

- **Technical Details**: 모델 훈련에는 데이터 전처리, 사용자 지정 독일어 토크나이저(tokenizer) 생성, 모델 훈련 및 여러 기준에서 최종 모델 평가 등이 포함됩니다. LLäMmlein은 RedPajama V2 데이터셋에서 고품질의 독일어 데이터만을 포함하도록 전처리되었습니다. 32,000개의 토큰으로 구성된 새로운 독일어 토크나이저가 생성되었습니다. 두 개의 LLäMmlein 모델이 스크래치에서 훈련되었습니다.

- **Performance Highlights**: LLäMmlein 모델은 SuperGLEBer 기준에서 기존의 최신 모델들과 비교할 때 경쟁력 있는 성능을 보였으며, 일부 작업에서는 성능 향상이 초기 단계에서 정체되는 현상도 관찰되었습니다. 모델의 품질은 크기에 비례하여 증가했으며, 향후 모델 개발을 위한 자원 배분에 대한 귀중한 통찰을 제공합니다.



### The Promises and Pitfalls of LLM Annotations in Dataset Labeling: a Case Study on Media Bias Detection (https://arxiv.org/abs/2411.11081)
- **What's New**: 이 연구는 Large Language Models (LLMs)를 활용해 media bias detection을 위한 새로운 방식의 주석 생성 가능성을 탐구합니다. Anno-lexical이라는 최초의 대규모 데이터셋을 생성하였으며, 이는 48,000개 이상의 주석된 예제를 포함합니다. 연구 결과, 이 데이터셋으로 훈련된 분류기는 기존의 LLM 주석 기법보다 5-9% 더 높은 Matthews Correlation Coefficient (MCC)를 달성하였습니다.

- **Technical Details**: 이 연구에서는 문장 수준의 lexical bias 분류 문제를 다루며, LLM을 사용하여 대규모 데이터셋을 구축하는 3단계 파이프라인을 구현합니다. Anno-lexical 데이터셋을 통해 훈련된 Synthetic-Annotations Fine-Tuned classifier (SA-FT)가 기존 인간 주석 기반의 모델(Human Annotations Fine-Tuned classifier, HA-FT)과 비교해 성능을 발휘하는지를 검토하였습니다. 또한, LLM 주석의 질을 평가하고 실제 환경에서의 강건성을 비롯한 여러 기술적인 측면을 분석합니다.

- **Performance Highlights**: 연구 결과, SA-FT 모델은 LLM들의 성능을 초과하였으며, 기존의 HA-FT와 비슷한 성능을 보여주었습니다. 또한, SA-FT 모델은 긍정 클래스를 더 많이 recall하지만, 정밀도와 입력에 대한 강건성은 HA-FT보다 낮은 성능을 나타냅니다. 마지막으로, Anno-lexical 데이터셋과 Annotation과정의 용이성을 높이기 위한 Annomatic라는 파이썬 패키지를 공개하여 연구 기술을 널리 공유할 계획입니다.



### Multilingual Large Language Models: A Systematic Survey (https://arxiv.org/abs/2411.11072)
- **What's New**: 이 논문은 다국어 대형 언어 모델(MLLM)에 대한 최신 연구를 종합적으로 조사한 것입니다. MLLM은 언어의 경계를 넘어 이해하고 생성할 수 있는 능력을 갖추었으며, 이는 인공지능(AI) 분야에서의 중요한 발전으로 평가됩니다. MLLM의 아키텍처 및 사전 학습 목표를 논의하며, 다국어 능력을 높이는 데 기여하는 핵심 구성 요소와 방법론을 강조합니다.

- **Technical Details**: MLLM은 단일 모델을 통해 여러 언어의 능력을 배우는 것을 목표로 합니다. 초기 다국어 모델은 각 언어 쌍에 대해 별도의 교육과 최적화가 필요했으나, 이후 사전학습된 다국어 언어 모델들이 등장하면서 이 병목 현상을 해결했습니다. BERT 모델과 같은 사전 학습된 모델은 다국어 데이터를 통해 크로스링구얼 전이 능력을 크게 향상시켰습니다.

- **Performance Highlights**: MLLM의 중요한 성과 중 하나는 다양한 분야에서의 실제 응용 사례입니다. 이들은 생물학, 의학, 컴퓨터 과학, 수학 및 법률 분야에서 혁신과 개선을 이끌어냈습니다. 그러나 MLLM을 다양한 언어 커뮤니티 및 응용 분야에 배포하는 데 있어서의 도전 과제 역시 중요하게 다뤄집니다.



### Beyond Human-Like Processing: Large Language Models Perform Equivalently on Forward and Backward Scientific Tex (https://arxiv.org/abs/2411.11061)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 성공이 언어 처리 모델로서의 인간 능력과 관련되어 있다는 기존의 주장과는 달리, 트랜스포머 아키텍처의 유연성에서 기인한다고 제안합니다. 연구진은 신경과학 특정 문헌을 사용하여, 순방향(forward) 및 역방향(backward) 텍스트로 LLM을 훈련시켰으며, 두 포맷 모두에서 LLM이 인간 전문가의 성능을 초월했음을 발견했습니다. 이러한 결과는 언어 모델이 특정 언어 구조에 국한되지 않고 일반적인 패턴 학습 기계로서 기능함을 보여줍니다.

- **Technical Details**: 연구진은 GPT-2 아키텍처를 기반으로 H브레인벤치(BrainBench)라는 기준 테스트를 통해 LLM의 성능을 평가했습니다. 171명의 신경과학 전문가와 비교하여, 역방향으로 훈련된 GPT-2 모델이 순방향 모델보다 비유의미하게 더 정확한 성능을 보였습니다. 모델 크기가 커질수록 성능이 향상되었으며, 가장 큰 모델은 인간 전문가의 성과를 초과하는 결과를 나타냈습니다.

- **Performance Highlights**: 실험 결과는 역방향으로 훈련된 모델이 높은 곤란도(perplexity)를 가진 반면에도 불구하고 인지 능력을 극복하는 경향을 보였습니다. 이는 LLM이 특정 도메인 관련 토큰을 효과적으로 생성할 수 있는 특수화된 토크나이저의 영향을 받을 수 있다는 점도 시사합니다. 연구진은 이러한 LLM의 성능을 통해 언어의 구조와 관계없이 입력을 처리할 수 있는 일반적인 패턴 학습 기계로서의 특성을 강조했습니다.



### FastDraft: How to Train Your Draf (https://arxiv.org/abs/2411.11055)
Comments:
          ENLSP NeurIPS Workshop 2024

- **What's New**: FastDraft는 효율적인 초안 모델을 생성하여 대형 언어 모델(LLM)의 성능을 향상시키는 새로운 접근법입니다. 기존의 Speculative Decoding(SD) 기술은 초안 모델의 품질과 대상 LLM과의 정렬에 크게 의존했는데, FastDraft는 사전 훈련과 합성 데이터 세트를 통해 이 문제를 해결합니다. 이 방법을 통해 Phi-3-mini 및 Llama-3.1-8B 모델에 대해 매우 효율적인 초안 모델을 훈련시켰습니다.

- **Technical Details**: FastDraft는 대형 언어 모델을 대상으로 한 초안 모델을 개발하기 위해 효율적인 사전 훈련을 수행 후, 대상 모델에 의해 생성된 합성 데이터 세트를 통해 미세 조정을 진행합니다. 8개의 Intel® Gaudi® 2 가속기를 사용하는 단일 서버에서 약 10억 개의 토큰으로 초안 모델을 훈련함으로써, 메모리 제한 속도 증가(Memory Bound Speedup) 측정에서 최대 3배 향상을 달성했습니다.

- **Performance Highlights**: FastDraft를 통해 자연어 처리 작업에서 평균 1.5배, 코드 완성 작업에서 평균 2배의 속도 향상을 보여 주었습니다. 이를 통해 FastDraft는 고성능을 발휘하며 AI-PC와 기타 엣지 장치에서 LLM의 추론 가능성을 열어줍니다. 또한 FastDraft는 리소스 요구 사항이 낮으면서도 효율적인 초안 모델을 제공하여 저용량 데이터로도 품질 높은 초안을 생성할 수 있게 합니다.



### SRA-MCTS: Self-driven Reasoning Aurmentation with Monte Carlo Tree Search for Enhanced Code Generation (https://arxiv.org/abs/2411.11053)
- **What's New**: 이번 연구에서는 SRA-MCTS라는 새로운 데이터 생성 방식을 소개하여 LLMs가 복잡한 문제를 해결하는 데 필요한 사고 경로를 자율적으로 생성할 수 있도록 지원합니다. 이 방법은 모델 자체를 통해 작동하며 추가적인 감독 없이도 성능을 효과적으로 향상시킵니다. SRA-MCTS는 자연 언어 추론 경로를 합성하고 실행 가능한 코드로 변환하는 과정을 포함하여, 분석의 정확성을 보장하고 복잡한 문제 해결의 성공률을 높이는 데 기여합니다.

- **Technical Details**: 연구에서는 SRA-MCTS 방법론이 4단계로 구성되어 있으며, 각각 선택(Selection), 확장(Expansion), 평가 및 반영(Evaluation & Reflection), 역전파(Backpropagation) 단계로 이루어집니다. 각 단계는 반복적으로 실행되며, LLM은 중간 단계에서 중요한 역할을 수행하여 자연어 해결책을 생성합니다. 코드는 주어진 문제와 자연어 해결책을 바탕으로 생성되고, 이를 통해 모델의 훈련 데이터 세트를 구성합니다.

- **Performance Highlights**: 실험 결과, SRA-MCTS로 생성된 데이터로 모델을 미세조정하면 기존 CoT 방법론보다 우수한 성능을 보여주었습니다. 특히, 작은 모델에서도 자가 개선(self-improvement) 기능을 통해 70B 모델의 증류 데이터로 훈련된 작은 모델보다 더 나은 성과를 달성하였습니다. 또한, 기존의 CoT 접근 방식의 성능 저하 문제를 해결하고, diverse metrics와 같은 다양한 평가 지표에서 개선된 성능을 보였습니다.



### BianCang: A Traditional Chinese Medicine Large Language Mod (https://arxiv.org/abs/2411.11027)
- **What's New**: 이 논문은 전통 중국 의학(TCM) 분야에 특화된 대형 언어 모델(BianCang)을 제안합니다. TCM의 진단 및 증후군 별화를 위한 기존 모델의 한계를 극복하고자 도메인 지식을 주입한 후 목표 지향적 자극을 통해 모델을 조정하는 두 단계의 훈련 과정을 채택하였습니다. 이는 TCM의 복잡한 지식을 이해하고 활용하는 데 필요한 새로운 접근법을 제시합니다.

- **Technical Details**: BianCang은 Qwen-2/2.5를 기반으로 하여 TCM의 특징에 맞춘 두 단계의 훈련 과정을 통해 개발되었습니다. 첫 번째 단계에서는 기존 모델에 TCM 및 의학 지식을 주입하는 지속적인 사전 훈련이 이루어지고, 두 번째 단계에서는 감독된 세부 조정을 통해 내부 지식을 활성화하고 정렬합니다. 이 과정은 TCM의 증후군을 구분하고 진단하는 능력을 향상시키는 데 중점을 둡니다.

- **Performance Highlights**: BianCang 모델은 11개의 테스트 세트와 4가지 작업을 통해 평가되었으며, 모든 능력 차원에서 기존의 TCM 및 의학 LLM을 지속적으로 초과하는 성능을 보였습니다. 이 모델은 TCM 특화 언어 모델로서 진정한 증후군 differentiation 및 진단 능력을 입증하였으며, 오픈소스 ChP-TCM 데이터셋 또한 개발하여 배포하였습니다. 이를 통해 향후 연구에 유용한 실험 비교 자료를 제공할 수 있게 됩니다.



### A Topic-aware Comparable Corpus of Chinese Variations (https://arxiv.org/abs/2411.10955)
Comments:
          4 pages, 4 figures, presented at APCLC2018: ASIA-PACIFIC CORPUS LINGUISTICS CONFERENCE 2018

- **What's New**: 이 연구는 대만 중국어(Taiwanese Mandarin)와 중국 본토 중국어(Mainland Chinese Mandarin)의 주제 인식 비교 말뭉치(corpus)를 구축하여 기존의 연구 간극을 메우는 것을 목표로 하고 있습니다. 대만의 Dcard와 중국 본토의 Sina Weibo를 활용하여 현대 언어 사용을 반영한 정기적으로 업데이트되는 비교 말뭉치를 생성했습니다.

- **Technical Details**: 연구는 대만과 중국 본토의 소셜 미디어 플랫폼에서 발췌한 데이터를 기반으로 하며, 이로 인해 두 종류의 중국어 사이의 언어적 차이를 탐구할 수 있는 기회를 제공합니다. 비교 말뭉치는 주제 인식을 바탕으로 하여 언어의 문화적 контекст(context)을 더욱 명확하게 드러낼 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 말뭉치는 현대 중국어 사용을 실시간으로 반영하여, 연구자들이 언어 변화 및 사용자 동향을 분석할 수 있는 유용한 자원이 될 것입니다. 이러한 자원은 향후 중국어 학습 및 언어학 연구에 중요한 기여를 할 것으로 기대됩니다.



### Dialectal Toxicity Detection: Evaluating LLM-as-a-Judge Consistency Across Language Varieties (https://arxiv.org/abs/2411.10954)
- **What's New**: 이 연구는 현대 LLM(대형 언어 모델)이 방언 차이에 따라 독성이 감지하는 방식에 대한 체계적인 연구가 부족하다는 문제를 지적합니다. 본 논문은 10개 언어 군과 60가지 변형을 포함한 다중 방언 데이터셋을 생성하여 LLM의 독성 평가에 대한 포괄적인 연구를 수행합니다. 연구 결과, LLM은 다국어 및 방언 변이에 민감하게 반응하지만, LLM과 인간 평가자 간의 일관성이 가장 취약하다는 점이 강조됩니다.

- **Technical Details**: 연구에서는 (i) 방언 데이터셋 확장과 (ii) LLM-판사 일관성 평가의 두 가지 주요 단계를 통해 LLM의 독성 평가 능력을 분석합니다. ToxiGen 데이터셋을 활용하여 독성이 있는 언어를 식별하는 데 중점을 두고 인간 주석을 포함한 반평행 다국어 및 다방언 독성 코퍼스를 생성하였습니다. NLLB-200 기계 번역 모델을 사용해 다양한 언어 변형에서의 데이터 증강 기술을 적용하여 방언 특유의 언어적 특징을 보존했습니다.

- **Performance Highlights**: LLM은 방언과 다국어 변이에 대한 높은 민감도를 보여주었으나, LLM과 인간의 평가 간 일관성을 수치화하는 데에는 한계가 있었습니다. 연구 결과는 LLM이 방언의 뉘앙스를 인식하는 데 강한 감응력을 보이면서도 LLM과 인간 간의 일치를 개선해야할 필요성이 있음을 강조합니다. 이 연구는 독성 감지를 위한 공정하고 효과적인 시스템을 만들기 위한 더 넓은 목표에 기여하고 있습니다.



### Understanding Multimodal LLMs: the Mechanistic Interpretability of Llava in Visual Question Answering (https://arxiv.org/abs/2411.10950)
Comments:
          preprint

- **What's New**: 본 논문에서는 Multi-modal Large Language Models (MLLMs)의 기작을 탐구하는 최초의 시도가 이루어졌습니다. Llava라는 모델에서 시각적 질문 응답(Visual Question Answering, VQA) 기작을 분석하여, 텍스트 질문 응답(Textual QA, TQA)와의 유사성을 발견했습니다. 이 연구는 MLLMs의 해석 가능성을 높이고, 비주얼 지침 튜닝을 통해 기존 모델의 성능을 향상시켜주는 새로운 도구를 제안합니다.

- **Technical Details**: Llava 모델은 이미지와 질문을 입력으로 받아, 이미지 패치를 이미지 임베딩으로 변환하고, 이를 통해 최종 답변을 생성합니다. 논문에서는 VQA와 TQA의 기작을 비교 분석하며, 색상 정보를 포함한 Q&A 테스크에서 이들 기작의 유사성을 규명합니다. 특히, 이미지 임베딩을 임베딩 공간으로 투영하는 해석 방법을 통해 시각적 특징의 해석 가능성을 입증하였습니다.

- **Performance Highlights**: 제안된 해석 도구는 Llava의 VQA에서 중요한 이미지 패치를 식별하는 데 도움을 줍니다. 연구 결과, 기존 방법보다 해석 가능성이 우수하며, 계산 비용이 낮고 빠른 성능을 보여줍니다. 이 방법은 실제 해석에서 유용하게 사용될 수 있으며, 시각적 오류(visual hallucination)를 이해하는 데 기여할 것으로 기대됩니다.



### Analyzing Pok\'emon and Mario Streamers' Twitch Chat with LLM-based User Embeddings (https://arxiv.org/abs/2411.10934)
Comments:
          NLP4DH 2024

- **What's New**: 이번 연구는 Twitch 채팅을 대형 언어 모델(LLM)을 통해 사용자 임베딩(user embeddings)으로 표현하는 혁신적인 디지털 인문학 방법론을 제안합니다. 이들 임베딩은 자동 클러스터링을 통해 분석되며, Twitch의 여러 스트리머마다 독특한 채팅 구성 항목이 있는지를 조사합니다. 전체적으로 각각의 스트리머는 지지하는 시청자, 이모지 및 반응을 보내는 사용자의 두 가지 범주로 나눌 수 있으며 두 개의 스트리머는 반복적인 메시지를 보내는 스패머(spammer)라는 공통 범주를 가집니다.

- **Technical Details**: 우리는 Twitch의 세 개의 스트림을 연구하며, LLM인 PaLM-2를 활용해 각 사용자의 채팅 메시지를 연결하여 사용자 임베딩을 생성합니다. 클러스터링 과정에서는 유사도를 기반으로 클러스터를 자동으로 결정하는 affinity propagation 기법을 사용하며, 여기에서 코사인 유사도(cosine similarity)를 이용해 유사도 행렬을 작성합니다. 최종 분석에서는 각 스트리머별로 클러스터를 확인하고 낮은 메시지 수를 가진 사용자에 대한 분석을 제외하여 보다 명확한 경향성을 도출합니다.

- **Performance Highlights**: 연구 결과, SmallAnt의 경우 6개 클러스터, PointCrow는 12개, DougDoug는 31개 클러스터가 생성되었습니다. 그러나 수동으로 검토한 결과, 작은 클러스터들이 서로 유사하다고 판단되어 합쳐졌습니다. 최종적으로 SmallAnt에선 5개, PointCrow에선 4개, DougDoug에선 6개의 클러스터가 남았으며, 특히 DougDoug의 채팅 공간은 부정적인 의견을 표현하는 사용자들, 즉 비판적인 시청자들의 비율이 높은 것이 특징입니다.



### Learn from Downstream and Be Yourself in Multimodal Large Language Model Fine-Tuning (https://arxiv.org/abs/2411.10928)
- **What's New**: 본 논문에서는 Multimodal Large Language Model (MLLM)의 특수화 및 일반화 능력을 모두 유지하기 위한 새로운 접근법인 SPIDER를 제안합니다. 이 방법은 사전 훈련된 가중치와 미세 조정된 기울기를 평가하여 각 파라미터의 중요도를 측정하고, 중요도가 높은 파라미터만 선택적으로 업데이트합니다. MLLM의 일반화 능력 저하를 방지하기 위해 파라미터 중요성 차이를 고려하는 것이 핵심입니다.

- **Technical Details**: SPIDER는 'Importance Discrepancy Measurement (IDM)'를 도입하여 파라미터의 일반화 및 특수화 지식에 대한 중요도를 정량화합니다. 또한, Importance Selection Mask (ISM)를 사용하여 목표 배포에 적합한 파라미터를 선택적으로 최적화합니다. 이 방식은 이미지 캡셔닝 및 시각적 질문-답변 작업에서 다양한 MLLM 아키텍처를 통해 실험적으로 검증되었습니다.

- **Performance Highlights**: 제안하는 SPIDER 방법은 Flickr30k, COCO-Caption, ScienceQA, IconQA와 같은 여러 다운스트림 데이터셋에서 MLLM의 미세 조정 성능을 향상시키는 데 기여했습니다. 논문에서는 두 가지 주요 작업인 이미지 캡셔닝과 시각적 질문-답변에 대한 종합적인 분석을 제공하며, 실험 결과가 SPIDER의 효과성을 입증하고 있습니다. 이러한 결과를 통해 MLLM의 특징적 성능 저하를 완화하고 일반화 능력을 유지할 수 있는 가능성을 보여주고 있습니다.



### Inter-linguistic Phonetic Composition (IPC): A Theoretical and Computational Approach to Enhance Second Language Pronunciation (https://arxiv.org/abs/2411.10927)
Comments:
          10 pages, 6 Figures, submitted to ACL ARR October 2024 for NAACL 2025

- **What's New**: 이 논문에서는 제2언어(L2) 학습자들이 모국어(L1)의 유사한 음소로 L2의 익숙하지 않은 음소를 무의식적으로 대체하는 현상을 설명합니다. 이를 해결하기 위해, Inter-linguistic Phonetic Composition (IPC)라는 새로운 계산 방법을 제안하였습니다. IPC는 복수의 L1 음소에서 유래된 복합음을 사용하여 L2 음소를 재구성함으로써 잘못된 음운적 전이(phonological transfer)를 최소화하는 것을 목표로 합니다.

- **Technical Details**: IPC 방법은 L2 음소를 다수의 L1 음소로부터 조합된 복합음(composite sounds)으로 재구성하여, L2 학습 자들이 더 정확한 발음을 습득할 수 있도록 지원합니다. 연구에서는 두 개의 자동 음성 인식(automatic speech recognition) 모델을 이용하여 실험을 진행하였으며, IPC로 생성된 복합음 사용 시 L2 목표 음소의 인식률이 20% 향상되는 결과를 보여주었습니다.

- **Performance Highlights**: L2 화자들이 IPC 생성 복합음을 사용했을 때 목표 L2 음소의 인식률 향상이 관찰되었으며, 이는 기존 음운적 전이 패턴에 의해 영향을 받을 때보다 훨씬 빠르게 달성되었습니다. 이러한 결과는 복합음의 신속한 습득 가능성을 나타내며, L2 발음 교육에 중요한 기여를 할 수 있는 방법론으로 기대됩니다.



### Bias in Large Language Models: Origin, Evaluation, and Mitigation (https://arxiv.org/abs/2411.10915)
- **What's New**: 이번 논문은 Large Language Models(LLMs)의 편향(bias) 문제를 심층적으로 다루고 있습니다. 논문에서는 편향의 기원부터 현재의 완화 전략(mitigation strategies)까지의 전체적인 과정을 검토합니다. 이제 편향이 내재적(intrinsic) 및 외재적(extrinsic)으로 구분되어 LLMs의 다양한 NLP 작업에서 어떻게 나타나는지 분석하고 있습니다.

- **Technical Details**: 논문은 편향 평가(bias evaluation) 방법론을 데이터 수준(data-level), 모델 수준(model-level), 출력 수준(output-level)으로 구분하며, 각 접근 방식이 LLM의 편향 감지에 어떻게 기여하는지를 설명합니다. 또한, 편향 완화 기법을 모델 개발 단계별로 사전 모델(pre-model), 모델 내(intra-model), 및 사후 모델(post-model)로 구분하여 효과성과 한계를 강조하고 있습니다.

- **Performance Highlights**: 이 연구는 LLM의 편향 문제 해결을 위한 포괄적인 자원을 제공하며, 특히 의료 및 법률 분야와 같이 편향이 심각한 결과를 초래할 수 있는 실제 응용에 대한 윤리적 및 법적 영향에 대해서도 논의하고 있습니다. 이는 AI 시스템을 공정하고 책임 있게 발전시키기 위한 지속적인 노력에 기여하겠다는 목표를 가지고 있습니다.



### BPO: Towards Balanced Preference Optimization between Knowledge Breadth and Depth in Alignmen (https://arxiv.org/abs/2411.10914)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 성공을 위해 인간 피드백을 활용한 강화 학습(RLHF)의 중요성을 강조하고, 지식의 폭(breadth)과 깊이(depth)라는 새로운 개념을 도입합니다. 지식의 폭은 다양한 주제에서의 모델의 이해 범위를, 지식의 깊이는 특정 주제에 대한 상세 정보 제공 능력을 측정합니다. 또한, 균형 잡힌 선호 최적화(BPO)라는 방법을 도입하여 각 샘플의 지식 깊이를 동적으로 증강하는 방안을 제시합니다.

- **Technical Details**: BPO는 먼저 일반적인 파이프라인을 따라 대표성과 다양성을 갖춘 프롬프트의 부분 집합을 클러스터링하여 샘플링합니다. 이후 기울기 기반 클러스터링을 통해 각 샘플의 최적 지식 깊이를 모델 최적화 관점에서 추정합니다. 이 과정에서 중심 근처의 샘플에 더 많은 학습 자원을 집중시켜 모델 최적화를 지원합니다.

- **Performance Highlights**: 여러 벤치마크에서 실시한 실험 결과, BPO는 기존의 기준 방법들보다 더 뛰어난 성능을 보이며 교육 효율성을 유지하는 데 성공했습니다. 각 구성 요소의 효과에 대한 자세한 분석을 통해 BPO의 강건성을 확인하였으며, 기울기 기반 접근 방식이 동적으로 요구되는 지식 깊이를 추정하는 데 가장 뛰어난 성과를 보여주었고, 이는 인본적 선호 점수에서도 가장 높은 수치를 기록했습니다.



### SPICA: Retrieving Scenarios for Pluralistic In-Context Alignmen (https://arxiv.org/abs/2411.10912)
- **What's New**: 이번 연구에서는 큰 언어 모델(LLMs)을 사회적 가치에 맞추기 위해 다양한 그룹의 다원적 가치를 반영한 SPICA라는 프레임워크를 제안합니다. 이 프레임워크는 시나리오 뱅크(scenario banks), 그룹 정보 기반 메트릭(group-informed metrics), 그리고 인컨텍스트 정렬 프롬프트(in-context alignment prompts)의 세 가지 요소로 구성되어 있으며, 다양한 그룹의 가치 우선순위를 고려합니다. 특히 인컨텍스트 학습(in-context learning)을 활용하여 다양한 그룹의 가치와 규범을 통합할 수 있는 점에서 차별성을 보입니다.

- **Technical Details**: SPICA는 시나리오 뱅크를 통해 그룹의 선호와 규범을 포함한 시나리오 집합을 관리하고, 그룹 정보 기반 메트릭을 사용하여 개별 선호 평가에서 두 번째 규범을 회수합니다. 이러한 접근은 예시의 효과 분석을 통해 다원적 가치 정렬 과제에 대해 기존 인컨텍스트 정렬 방법보다 더 정교한 성능을 보여주었습니다. SPICA는 실제 데이터 세트에서 그룹별 우선순위를 더 정확하게 반영하며, 통계적으로 유의미한 성과를 달성했습니다.

- **Performance Highlights**: SPICA를 통해 조율된 모델은 비슷한 예시 기반 접근 방식보다 약 +0.16 포인트 높은 평가를 받았으며, 모든 그룹이 혜택을 보았습니다. SPICA는 특정 그룹이 아닌 모든 그룹에 대해 균일한 성과를 제공하며, 개별 가치에 대한 접근성을 향상시켰습니다. 이는 그룹 간 차이를 고려한 새로운 평가 방식을 제시하여, 다원적 가치 정렬을 위한 효과적인 솔루션을 제공함을 의미합니다.



### BanglaDialecto: An End-to-End AI-Powered Regional Speech Standardization (https://arxiv.org/abs/2411.10879)
Comments:
          Accepted in 2024 IEEE International Conference on Big Data (IEEE BigData)

- **What's New**: 이번 연구는 방글라데시 사투리를 인식하고 다양한 벵골어 억양을 표준화된 공식 벵골어로 변환하는데 중점을 둡니다. 특히, 방글라어의 55개 서로 다른 사투리를 다루는 점이 중요하며, 표준화가 교육적 일관성 및 경제적 기회를 보장하는 데 필요합니다.

- **Technical Details**: 이 연구는 Noakhali 지역의 방글라 사투리 음성을 표준 방글라 음성으로 변환하는 종단 간 파이프라인을 제시합니다. 이 과정에서 대규모 방글라 사투리 데이터셋을 구성하여 ASR(Automated Speech Recognition) 및 LLM(멀티언어 대형 언어 모델) 모델을 미세조정(fine-tuning)하여 사투리 음성을 텍스트로 변환하고 이를 표준 방글라 텍스트로 번역합니다.

- **Performance Highlights**: Whisper ASR 모델 미세 조정을 통해 CER(Character Error Rate) 0.8%, WER(Word Error Rate) 1.5%를 달성하였으며, BanglaT5 모델은 사투리 텍스트를 표준 텍스트로 번역하는 데 41.6%의 BLEU 점수를 기록했습니다.



### Empowering Meta-Analysis: Leveraging Large Language Models for Scientific Synthesis (https://arxiv.org/abs/2411.10878)
Comments:
          Accepted in 2024 IEEE International Conference on Big Data (IEEE BigData)

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)을 활용하여 과학 문서의 메타-분석(메타-분석) 자동화를 탐구합니다. 전통적으로 메타-분석은 여러 연구 결과를 종합하여 심층적인 이해를 제공하지만, 수작업으로 진행할 경우 시간 소모적이고 인적 오류에 취약합니다. 이에 본 연구는 LLM을 대규모 과학 데이터셋으로 미세 조정하여 데이터를 효과적으로 처리하고 구조적 데이터 추출의 도전을 해결하는 새로운 접근법을 제시합니다.

- **Technical Details**: 이 자동화된 메타-분석 프로세스는 Retrieval Augmented Generation (RAG)을 통합하여 최적화됩니다. LLM은 새로운 손실 메트릭(Inverse Cosine Distance, ICD)을 통해 훈련되어 고차원 문맥 데이터에 적합한 데이터 추출 패턴을 학습합니다. 이 방법은 메타-분석 생성을 위한 LLM의 성능을 향상시키며, 구조적 초록 생성에서의 품질과 효율성을 보장합니다.

- **Performance Highlights**: 사람이 평가한 결과, 미세 조정된 LLM 모델이 비 미세 조정 모델보다 우수한 성능을 보여, 87.6%의 관련 메타-분석 초록을 생성했습니다. 평가에 따르면 맥락의 적합성이 4.56%에서 1.9%로 감소하여,본 연구는 메타-분석 자동화의 효율성과 신뢰도를 향상시키는 데 기여한 것으로 나타났습니다.



### Large Language Models (LLMs) as Traffic Control Systems at Urban Intersections: A New Paradigm (https://arxiv.org/abs/2411.10869)
Comments:
          The data and code that support the findings of this study are openly available in Zenodo at this https URL, reference number 14171745

- **What's New**: 이 연구는 대형 언어 모델(Large Language Models, LLMs)을 교통 제어 시스템의 새로운 방식으로 도입합니다. LLMs는 논리적 추론(logical reasoning), 장면 이해(scene understanding), 의사 결정(decision-making) 기능을 활용하여 실시간으로 교통 상황에 기반한 피드백을 제공하고 처리량(throughput)을 최적화합니다. 이러한 접근은 전통적으로 분리된 교통 제어 프로세스를 중앙 집중화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 LLMs가 교통 제어기 역할을 수행할 수 있는 능력을 평가하는 데 있어 네 단계의 방법론을 제안합니다. 방법론에는 데이터 생성(data creation), 환경 초기화(environment initialization), 프롬프트 엔지니어링(prompt engineering), 충돌 식별(conflict identification), 및 미세 조정(fine-tuning)이 포함됩니다. 다중 차선의 사거리 시나리오를 시뮬레이션하고 LLMs 및 Python 시뮬레이션을 기반으로 상세한 데이터셋을 생성하여 충돌 탐지(conflict detection) 가능성을 검증했습니다.

- **Performance Highlights**: GPT-mini 모델의 성능은 83%의 정확도와 0.84의 F1-score를 기록하며, 매우 유망한 성과를 보였습니다. 특히, 충돌 탐지에서 0.95, 의사 결정에서 0.91, 우선 순위 할당(priority assignment)에서 0.94, 대기 시간 최적화(waiting time optimization)에서 0.92의 ROUGE-L 점수를 달성하였습니다. 실시간으로 운전자에게 차량 동역학에 기반하여 양보(yielding), 감속(slowing), 정지(stopping)와 같은 정확한 추천을 제공할 수 있음을 보여주었습니다.



### Information Anxiety in Large Language Models (https://arxiv.org/abs/2411.10813)
- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 내부 추론 및 검색 메커니즘을 종합적으로 분석했습니다. 연구의 주요 초점은 엔티티의 인기도, 질의 형식의 어휘적 변동성에 대한 모델의 민감도, 및 LLM 계층 전반의 숨겨진 상태 표현의 진행을 포함합니다. 특히, 정보 검색 메커니즘 내에서 발생하는 스트레스를 나타내는 '정보 불안(information anxiety)'이라는 현상에 대해 논의합니다.

- **Technical Details**: 대형 언어 모델의 질문 응답 능력 향상을 위해 PopQA라는 엔티티 중심의 오픈 도메인 데이터셋을 활용했습니다. 이 데이터셋에서는 질의의 인기 수준을 주제와 객체의 평균 인기도를 기준으로 측정하며, 유사한 의미를 유지하는 10개의 어휘적 변형을 생성합니다. 실험에서는 F1 점수를 활용하여 예측된 객체 토큰과 정답 세트 간의 일치를 평가하였습니다.

- **Performance Highlights**: 연구 결과, LLMs는 인기 질문에 대해 관련 쿼리 섹션에 대한 주의가 낮아지는 경향이 있으며, 같은 질의에 대해 어휘 변형을 통한 사실이 상당히 다르게 검색되는 문제를 발견했습니다. 이러한 결과는 LLM들이 자주 발생하는 정보에 대응할 때 내부적으로 발생하는 강한 스트레스를 나타내며, 정보 불안 현상이 대형 언어 모델의 성능에 미치는 부정적 영향을 강조합니다.



### Can Generic LLMs Help Analyze Child-adult Interactions Involving Children with Autism in Clinical Observation? (https://arxiv.org/abs/2411.10761)
Comments:
          GenAI for Health Workshop, NeurIPS 2024

- **What's New**: 이 연구는 대형 언어 모델(LLM)이 임상 환경에서 아동-성인 상호작용을 분석하는 능력을 평가한 최초의 시도 중 하나로, 특히 자폐 스펙트럼 장애(ASD) 아동과의 대화에 초점을 맞췄습니다. 연구진은 LLM이 말의 발화자, 참여된 활동, 아동의 특성 및 언어 능력을 예측하는 네 가지 중요 과제를 수행할 수 있는지를 조사했습니다. LLM의 성능은 비전문가 평가자보다 우수하여 아동 언어 능력 평가를 돕고, 임상 관찰 세션에서의 상호작용을 분할하는 데 유망한 가능성을 보여줍니다.

- **Technical Details**: 연구에서는 Remote-NLS와 ADOSMod3라는 두 가지 데이터셋을 사용했습니다. Remote-NLS는 ASD 아동과의 자연적인 아동-성인 상호작용을 포함하여, 15분 길이의 Zoom 비디오로 구성되어 있습니다. ADOSMod3는 반구조화된 평가를 따르며, 자폐 진단을 위한 표준화된 프로토콜인 ADOS-2에 맞춘 아동-임상 의사 상호작용이 포함되어 있습니다.

- **Performance Highlights**: LLM은 긴 대화 기록에서 아동-성인 발화자를 분류하고, 관련된 활동이나 관찰 사항을 추출하는 데 있어 높은 정확성을 보였습니다. 연구 결과, LLM은 아동의 언어 능력과 특성을 성공적으로 예측하며, 이러한 작업에서 비전문가 조사자들보다 더 뛰어난 성능을 발휘했습니다. 그러나 LLM이 생성하는 산출물 중 일부는 사실에 기반하지 않는 경우가 종종 관찰되었습니다.



### Comparison of Multilingual and Bilingual Models for Satirical News Detection of Arabic and English (https://arxiv.org/abs/2411.10730)
Comments:
          ALTA 2024 (Selected for publication)

- **What's New**: 이번 연구는 다양한 문화적 배경으로 인한 풍자 뉴스의 오해를 해소하기 위해 영어와 아랍어에서의 다국어 풍자 탐지 방법을 활용했습니다. 연구에서는 두 개의 언어 모델인 Jais-chat과 LLaMA-2-chat을 사용하여, Zero-shot 및 Chain-of-Thought (CoT) 프롬프트 기법의 성능을 비교했습니다. CoT 프롬프트 기법이 Jais-chat 모델에서 더 큰 개선 효과를 나타내며, 영어 상황에서 80%의 F1-score를 기록했습니다.

- **Technical Details**: 연구에서는 Jais-chat과 LLaMA-2-chat 모델을 활용해 두 언어(영어와 아랍어)에서 풍자 탐지를 평가했습니다. Jais-chat은 두 언어에 대해 훈련된 이중 언어 모델로, LLaMA-2-chat은 다국어 모델로서 여러 언어를 지원합니다. 이 연구는 Zero-shot 프롬프트와 CoT 프롬프트의 효과를 비교하고, CoT 프롬프트가 더 나은 성능을 보이는지를 분석했습니다.

- **Performance Highlights**: Jais-chat 모델은 모든 상황에서 CoT 프롬프트를 사용할 때 우수한 성능을 보였으며, 특히 영어 프롬프트에서 80%의 F1-score로 가장 높은 성과를 기록했습니다. 반면 LLaMA-2-chat 모델은 CoT 접근 방식에서 성능 개선이 미비했고, F1-score는 각각 72.5%와 73%로 유지되었습니다. LLaMA-2-chat 모델은 높은 재현율을 나타냈으나 정밀도에서 어려움을 겪어, 잘못된 정보 전파의 위험이 존재합니다.



### HJ-Ky-0.1: an Evaluation Dataset for Kyrgyz Word Embeddings (https://arxiv.org/abs/2411.10724)
- **What's New**: 본 논문에서는 키르기스어에 특화된 최초의 'silver standard' 데이터세트를 소개하여 단어 벡터 품질 평가의 기초를 마련하고 있습니다. 이 데이터세트는 러시아어에서의 수동 번역을 통해 생성되었으며, 몇 가지 단어 임베딩 모델의 품질 메트릭을 통하여 검증되었습니다. 이는 키르기스어의 자연어 처리(NLP) 연구 및 개발에 결정적인 기여를 할 것으로 기대됩니다.

- **Technical Details**: 자연어 처리(NLP)는 응용 언어학, 인공지능, 기계 학습, 통계학의 융합된 분야로, 이번 연구는 키르기스어에 대한 연구의 빈틈을 메우기 위해 최초의 데이터세트를 구축하고 이를 평가하기 위한 기법을 제시하고 있습니다. 데이터세트는 전문가가 평가한 단어 쌍과 그에 대한 '유사성' 점수로 구성되며, 평가 메트릭으로는 Spearman의 순위 상관관계 및 Pearson의 상관계수를 사용하였습니다. 이 연구에서는 데이터세트를 최적화하는 방법과 더불어 기존의 단어 벡터 모델인 word2vec, fastText 등의 접근 방식을 설명합니다.

- **Performance Highlights**: 이 연구에서 구축한 키르기스어 데이터세트는 단어 벡터 표현의 품질을 평가하는 데 큰 의미를 가집니다. 연구자들은 키르기스어와 러시아어에 대해 사전 훈련된 단어 임베딩을 활용하여 초기 결과를 확인했습니다. 이 데이터세트는 이후의 연구 및 단어 임베딩 대회에 사용될 예정이며, 기존의 비컨텍스트(non-contextual) 단어 벡터 표현 방식의 성능을 검증하는 데 활용될 것입니다.



### Structured Dialogue System for Mental Health: An LLM Chatbot Leveraging the PM+ Guidelines (https://arxiv.org/abs/2411.10681)
Comments:
          Accepted to the 16th International Conference on Social Robotic (ICSR 2024)

- **What's New**: 새로운 Structured Dialogue System(SuDoSys)은 심리 상담을 제공하기 위해 설계된 혁신적인 LLM 기반 챗봇입니다. WHO의 Problem Management Plus(PM+) 지침을 활용하여 단계 인식 다중 턴 대화를 제공합니다. 기존의 LLM을 활용한 심리 상담 방식은 보통 대화 생성에 대한 직접적인 파인튜닝을 포함하였으나, SuDoSys는 상담 세션의 다양한 단계와 정보를 저장하여 일관되고 목적 있는 대화를 유지합니다.

- **Technical Details**: SuDoSys는 다섯 개의 모듈로 구성돼 있으며, 각각 단계 제어기(stage controller), 단계 인식 지시 생성기(stage-aware instruction generator), 주제 데이터베이스(topic database), 사전 훈련된 LLM, 응답 언팩커(response unpacker)로 이루어져 있습니다. 각 대화 턴에서 시스템은 현재 단계 번호와 사용자 입력, 이전 단계에서의 대화 주제를 기초로 합니다. 이러한 정보를 바탕으로 LLM은 대화의 상태를 결정하고 적절한 응답을 생성합니다.

- **Performance Highlights**: 객관적 및 주관적 평가를 통해 SuDoSys는 기존의 파인튜닝 방법에 비해 논리적으로 일관된 응답을 생성하는 데 성공적임을 입증했습니다. 이 시스템은 심리 상담의 다중 턴 대화에서 효과적으로 작동하며, 코드 및 평가 프로그램 스크립트는 오픈소스로 제공됩니다.



### IntentGPT: Few-shot Intent Discovery with Large Language Models (https://arxiv.org/abs/2411.10670)
Comments:
          ICLR 2024 Workshop on LLM Agents

- **What's New**: 이번 연구에서는 Intent Discovery(의도 발견)라는 새로운 접근 방식을 제안합니다. 기존의 방식들은 많은 데이터와 훈련을 필요로 했지만, 저자는 IntentGPT라는 훈련이 필요 없는 방법을 통해, 최소한의 라벨링된 데이터로도 새로운 의도를 발견할 수 있도록 하였습니다. 이러한 방식은 대규모 언어 모델(LLM)을 활용하여 유연하게 의도를 인식하게 합니다.

- **Technical Details**: IntentGPT는 세 가지 주요 구성 요소로 이루어져 있습니다: In-Context Prompt Generator, Intent Predictor, 및 Semantic Few-Shot Sampler입니다. 이 시스템은 자동으로 생성된 프롬프트를 활용하여 맥락 학습을 수행하며, 사용자 발화에서 의도를 분류하고 발견합니다. 또한, 적합한 few-shot 예제와 알려진 의도를 프롬프트에 삽입하는 기능을 통해 새로운 의도를 인식시킵니다.

- **Performance Highlights**: 실험 결과, IntentGPT는 CLINC 및 BANKING과 같은 다양한 벤치마크에서 이전에 비해 뛰어난 성능을 보였습니다. 전통적인 방법들이 요구하는 방대한 도메인 특화 데이터와 세밀한 파인튜닝 없이도 효과적인 결과를 보여주었습니다. 특히, 하이퍼파라미터의 영향 분석을 통해, 기존의 몇 샷 ICL 설정에서의 성능 향상을 입증하였습니다.



### SAM Decoding: Speculative Decoding via Suffix Automaton (https://arxiv.org/abs/2411.10666)
Comments:
          13 pages, 3 figures

- **What's New**: 이번 논문에서는 SAM-Decoding이라는 새로운 도입 방안을 제안하여 현재의 n-gram 기반 방식의 한계를 극복하고 텍스트 생성을 더욱 효율적이고 정확하게 수행하는 방법론을 소개합니다. 이 방식은 기존의 리트리벌(retrieval) 기반 방법들과의 비교에서 상당한 속도 향상을 보여줍니다. SAM-Decoding은 기존 방법들과 결합하여 Adaptive하게 드래프트 생성 전략을 선택함으로써 LLM의 추론 속도를 증가시킵니다.

- **Technical Details**: SAM-Decoding은 Suffix Automaton을 이용하여 텍스트 생성 및 텍스트 데이터베이스에서 가장 긴 접미사 일치를 찾아내는 방식으로 작동합니다. 코드는 정적 및 동적 접미사 자동기를 각각 기존 텍스트와 입력 프롬프트에 대해 구성하여 신속하고 정확한 드래프트 생성을 가능하게 합니다. 이 과정에서, 가장 긴 접미사 일치는 O(1) 복잡도로 탐지되며 이는 기존 n-gram 방식보다 월등한 성능을 보여줍니다.

- **Performance Highlights**: SAM-Decoding은 Token Recycling과 결합될 경우 Spec-Bench에서 오토회귀 디코딩보다 2.27배의 속도 향상을 달성하며, EAGLE2와의 조합에서는 2.49배의 속도 향상을 기록하여 현재 모든 접근 방식 중에서 가장 높은 성능을 나타냅니다. 광범위한 평가에서 SAM-Decoding은 기존 최첨단 방법들과 비교해도 경쟁력 있는 결과를 생성하며 특히 리트리벌 기반 방법들이 적용 가능한 작업에서 우수한 성능을 보입니다.



### Gender Bias Mitigation for Bangla Classification Tasks (https://arxiv.org/abs/2411.10636)
- **What's New**: 이 연구에서는 방글라어에 대해 미리 훈련된 언어 모델에서 성별 편향(gender bias)을 조사하였습니다. 이는 저자원 언어에서 미처 탐구되지 않은 영역으로, 기존 데이터 세트를 사용하여 성별 이름 교환 기법을 적용하고 특정 작업에 적합한 데이터 세트를 만들었습니다. 이 연구는 다양한 성별 편향 완화 방법들과 우리가 제안한 새로운 접근 방식을 비교하여 효과성을 입증했습니다.

- **Technical Details**: 연구는 감정 분석(sentiment analysis), 독성 탐지(toxicity detection), 증오 발언 탐지(hate speech detection), 그리고 풍자 탐지(sarcasm detection)와 같은 작업에 대해 총 4개의 수작업으로 주석이 달린 데이터 세트를 준비했습니다. 또한, 우리는 교차 엔트로피 손실(cross entropy loss)과 코사인 유사성(cosine similarity)을 기반으로 한 공동 손실 최적화(joint loss optimization) 기법을 제안하였습니다. 데이터 세트 구축을 위해 성별 단어 쌍을 포함하는 사전(dictionary)을 만들고, 명명된 개체 인식(NER) 기법을 통해 성별 이름을 치환하는 방법을 사용했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 편향 완화 방법들은 물론 정확도를 유지하면서도 성별 편향을 효과적으로 줄이는 것으로 나타났습니다. 각각의 모델에서 편향 비율은 최악의 경우 7.48%에서 최선의 경우 0.46%로 측정되었습니다. 우리의 접근 방식이 간섭 없이 더 많은 분야에 적합하다는 것을 입증했습니다.



### Leveraging large language models for efficient representation learning for entity resolution (https://arxiv.org/abs/2411.10629)
Comments:
          22 pages and 12 figures

- **What's New**: 이 논문에서는 TriBERTa라는 감독 학습 기반의 엔티티 해상도(Resolution) 시스템을 제안합니다. 이 시스템은 사전 훈련(pre-trained)된 대형 언어 모델과 triplet loss function을 이용하여 엔티티 매칭을 위한 표현을 학습합니다. 이는 현대의 데이터 주도(data-driven) 환경에서 중복 데이터 식별 및 조정과 관련된 문제를 해결하는 데 기여합니다.

- **Technical Details**: TriBERTa 시스템은 두 단계로 구성됩니다. 첫 번째로, 엔티티 레코드가 SBERT(Sentence Bidirectional Encoder Representations from Transformers) 모델에 입력되어 벡터 표현(vector representations)이 생성됩니다. 이후에는 contrastive learning 방식을 거쳐 triplet loss function에 기반하여 세밀하게 조정된 표현(fine-tuned representations)을 엔티티 매칭 작업에 입력으로 사용합니다.

- **Performance Highlights**: 연구 결과, TriBERTa의 접근 방식은 세밀하게 조정되지 않은 SBERT 및 전통적인 TF-IDF(Term Frequency-Inverse Document Frequency)와 비교했을 때 3-19% 더 뛰어난 성능을 나타냅니다. 또한 TriBERTa로 생성된 표현은 여러 데이터셋에서 일관되게 높은 성능을 유지하며, 강인성을 증가시키는 효과도 확인되었습니다.



### A dataset of questions on decision-theoretic reasoning in Newcomb-like problems (https://arxiv.org/abs/2411.10588)
Comments:
          48 pages, 15 figures; code and data at this https URL

- **What's New**: 이번 논문에서는 Newcomb-like 문제의 의사결정 이론에 관한 자연어 질문 데이터셋을 소개합니다. Newcomb-like 문제는 에이전트가 유사한 다른 에이전트와 상호작용하는 상황을 포함하며, 이는 다른 에이전트가 유사한 방식으로 추론할 것임을 고려해야 함을 의미합니다. LLM의 Newcomb-like 문제에 대한 추론 평가가 중요한 이유는 기초 모델 파생 에이전트 간의 상호작용이 이러한 문제와 유사할 것이기 때문입니다.

- **Technical Details**: 데이터셋은 고유하고 논란의 여지가 없는 답변을 가진 능력 질문(capability questions)과 의사결정 이론가들 간의 논쟁이 있는 태도 질문(attitude questions)을 모두 포함합니다. 연구는 기존 OpenAI, Anthropic, Meta, GDM, Reka 등의 다양한 모델과 간단한 프롬프트 기반 개입 하의 모델을 조사하는 데 사용됩니다. 이러한 질문을 통해 의사결정 이론적 역량과 표현된 태도 간의 상호작용을 분석합니다.

- **Performance Highlights**: 연구 결과, 기존 모델 간의 태도가 상당히 다르며, 높은 역량은 이른바 증거적 의사결정 이론(evidential decision theory)으로의 보다 긍정적인 태도와 연관되어 있다는 것을 발견했습니다. 또한, 태도는 다양한 질문 유형에서 일관성이 있음을 보여주었습니다.



### On the Shortcut Learning in Multilingual Neural Machine Translation (https://arxiv.org/abs/2411.10581)
Comments:
          Accepted by Neurocomputing 2024

- **What's New**: 이 연구에서는 다국어 신경 기계 번역(Multilingual Neural Machine Translation, MNMT)에서 자주 언급되는 목표 외 문제(off-target issue)를 재조명합니다. 실험 설계를 통해 zero-shot 번역에서 비중심(non-centric) 언어를 중심 언어로 잘못 번역하는 경향이 있음을 밝혀냈습니다. 또한, 특징적으로 멀티링구얼 프리트레이닝(multilingual pretraining)이 이러한 편향을 심화시키는 것으로 나타났습니다.

- **Technical Details**: 우리는 MNMT 모델의 훈련 과정에서 발생하는 단축키 학습(shortcut learning)을 다루며, 이는 지도 언어 매핑(supervised language mapping)의 과적합(overfitting)과 연결됩니다. 연구팀은 훈련 데이터 내에서 비중심 언어 쌍의 인스턴스를 제거함으로써 단축키를 없애는 '일반화 훈련(generalization training)'이라는 새로운 훈련 전략을 제안합니다. 이 방법은 MNMT 모델 성능을 향상시키면서 추가적인 데이터나 계산 비용이 들지 않습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 zero-shot 번역 성능을 일관되게 그리고 현저하게 개선시키는 것으로 나타났습니다. 또한, 지도 번역(supervised translation) 성능을 유지하면서도, 다양한 언어 분포 및 벤치마크에서 성능을 비교한 결과, 기존의 강력한 기준선(baselines)보다 뛰어난 성능을 보였습니다. 이 연구는 제로 샷 번역의 일반화 능력을 개선하기 위한 첫 번째 중요한 기여로 인정받을 것입니다.



### mlan: language-based instruction tuning improves zero-shot generalization of multimodal large language models (https://arxiv.org/abs/2411.10557)
- **What's New**: 이 연구는 다중 모달 대형 언어 모델(MLLM)의 제로샷(zero-shot) 태스크 일반화(generalization)를 향상시키기 위한 새로운 언어 기반 지침 조정을 제안합니다. 기존의 지침 조정 방식과 다르게, visual instructions에 중점을 두기보다는 언어 지침을 이용한 방식으로 훈련 효율성을 높이고 있습니다. 이 방법은 특히 Llama 2와 Vicuna 기반의 사전 훈련된 모델에서 미비한 시각적 훈련 없이도 언어 작업을 수행하는 능력을 강화합니다.

- **Technical Details**: 제안한 Mlan 방법은 세 가지 단계로 구성됩니다. 첫 번째로, 훈련 데이터를 선택하고 지침과 함께 형식을 설정하며, 두 번째로 사전 훈련된 MLLM을 훈련 세트에 맞춰 미세 조정(finetuning)합니다. 마지막으로, 미세 조정된 모델의 성능을 보지 못한 데이터셋에서 평가하는 방식입니다. Mlan의 핵심 차별점은 제로샷 성능을 향상시키고 훈련 효율성을 높이는 것입니다. 

- **Performance Highlights**: 제안된 방법은 9개의 보지 못한 데이터셋에서 두 가지 사전 훈련된 모델을 대상으로 실행되었으며, 평균 15.4%의 성능 향상을 보여주었습니다. 특히, 언어 기반 지침 조정 방식이 다른 최첨단(multimodal instruction tuning) 접근 방식과 비교해 우수한 성능을 발휘하며 훈련 효율성을 4.6배 개선하였습니다. 최종적으로, 언어 기반 지침 조정이 다중 모달 모델의 제로샷 일반화 능력을 높인다는 것은 LLM의 성공 배경과 일치합니다.



### Does Prompt Formatting Have Any Impact on LLM Performance? (https://arxiv.org/abs/2411.10541)
Comments:
          Submitted to NAACL 2025

- **What's New**: 이 논문은 다양한 프롬프트 템플릿이 대형 언어 모델(LLM)의 성능에 미치는 영향을 탐구하고 있습니다. 기존 연구들은 템플릿의 미세한 변형에 대한 민감성을 분석했지만, 이 연구는 여러 인간 가독성 템플릿을 사용하여 그 효과를 평가합니다.

- **Technical Details**: 연구는 OpenAI의 GPT 모델을 대상으로 여러 가지 입력 형식(plain text, Markdown, JSON, YAML)을 비교하여 자연어 추론, 코드 생성 및 번역과 같은 작업에 대해 수행되었습니다. 실험은 다양한 태스크와 데이터셋을 사용하여 민감성(sensitivity), 일관성(consistency), 전이 가능성(transferability) 측면에서 모델 성능을 분석합니다.

- **Performance Highlights**: GPT-3.5-turbo 모델은 프롬프트 템플릿에 따라 코드 번역 작업에서 성능이 최대 40%까지 달라지는 반면, GPT-4 모델은 훨씬 더 높은 복원력(resilience)을 보였습니다. 놀랍게도 모든 모델에서 최적의 형식은 존재하지 않으며, GPT-4-turbo는 이전 모델들에 비해 프롬프트 구조 변경에 덜 민감하게 반응했습니다.



### "On the goals of linguistic theory": Revisiting Chomskyan theories in the era of AI (https://arxiv.org/abs/2411.10533)
- **What's New**: 본 논문에서는 이론 언어학의 궁극적인 연구 목표에 도달하는 데 있어 대규모 언어 모델과 같은 AI 모델이 어떤 역할을 할 수 있는지를 논의합니다. 저자들은 생성 언어학의 기본 원칙을 재확인하며, 이러한 원칙이 AI 모델과 어떻게 연결될 수 있는지를 탐구합니다. 특히, 신경 언어 모델(neural language models)과 신경 문법 유도 모델(neural grammar induction models)을 다루며, 이러한 AI 모델들이 언어 이론으로서의 적합성을 어떻게 가질 수 있는지 살펴봅니다.

- **Technical Details**: 이론 언어학에서의 주요 개념인 (1) 이론적 충분성의 수준, (2) 언어 이론 개발 절차, (3) 언어의 학습 가능성과 보편 문법(Universal Grammar)을 중심으로 AI 모델과의 관계를 분석합니다. 신경 언어 모델은 신경망을 통해 언어의 맥락을 예측하는 확률적 모델이며, 신경 문법 유도 모델은 주어진 문장 집합을 기반으로 확률적 생성 문법을 유도합니다. 이러한 모델들은 언어 생성 과정뿐만 아니라 언어 이론의 발전에도 기여할 수 있는 가능성을 가지고 있습니다.

- **Performance Highlights**: 저자들은 신경 문법 유도 모델이 언어 이론의 기초 원칙을 고려할 때, 이론 언어학 연구에 대한 영향력을 미칠 수 있는 잠재력을 지니고 있다고 주장합니다. 이 모델들은 구조화된 표현을 학습하는 데 특히 적합하며, 기존의 자료와 인간의 언어 처리 방법을 기반으로 한 실험 데이터를 통해 그 유용성을 확인할 수 있습니다. 이 논문은 AI 모델이 언어 이론 및 이론 언어학의 발전에 기여할 수 있다는 점에서 중요한 기여를 하고 있습니다.



### A Survey on Importance of Homophones Spelling Correction Model for Khmer Authors (https://arxiv.org/abs/2411.10477)
- **What's New**: 이번 연구는 언어에서 동음이의어(homophones)가 글쓰기에서 저자에게 미치는 엄청난 어려움을 다루고 있습니다. 특히, 복잡한 구조와 방대한 문자 집합을 가진 크메르어에서 동음이의어의 문제가 두드러집니다. 이는 크메르어 저자들이 동음이의어를 사용할 때 자주 직면하는 어려움을 해결하고자 하는 노력을 포함합니다.

- **Technical Details**: 108명의 크메르 원어민을 대상으로 한 설문조사 결과, 많은 사용자들이 문맥에 따라 올바른 단어를 선택하는데 어려움을 겪고 있는 것으로 나타났습니다. 크메르어에서 동음이의어 오류를 해결하기 위한 효과적인 도구가 부족하여 글쓰기 과정이 복잡해지고 있습니다. 연구자는 영어, 아제르바이잔어, 방글라어와 같은 다른 언어에서의 철자 수정에 대한 기존 연구를 검토했으며, 동음이의어에 집중한 연구가 부족하다는 점을 발견했습니다.

- **Performance Highlights**: 이 연구는 크메르어 동음이의어 오류를 해결하기 위한 전문 도구의 필요성을 강조합니다. 이러한 도구는 연구 및 자원에서의 현재 격차를 해소하여 크메르어 저자들의 글쓰기에서 자신감과 정확성을 향상시키는 데 기여할 것입니다. 이러한 작업은 크메르어가 기술과 언어학의 발전을 효과적으로 활용하는 데 필수적입니다.



### Tackling prediction tasks in relational databases with LLMs (https://arxiv.org/abs/2411.11829)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)의 관계형 데이터베이스에 대한 적용 가능성을 탐구하고, RelBench 벤치마크를 사용하여 LLM이 이들 작업에서 경쟁력 있는 성능을 발휘할 수 있음을 입증합니다.

- **Technical Details**: RelBench는 다양한 도메인에서 가져온 7개의 관계형 데이터베이스로 구성되어 있으며, 각 데이터베이스는 예측 과제를 포함합니다. 관계형 데이터는 연결된 테이블의 복잡한 구조로 인해 기본적인 조정(flattening)이 필요하며, LLM을 사용하여 테이블 간의 연결 링크를 탐색함으로써 정보가 풍부한 문서를 생성하여 예측을 수행합니다.

- **Performance Highlights**: 예측 작업의 수행으로 LLM은 Relational Deep Learning(RDL) 방법과 유사한 수준의 성능을 나타냈고, 새로운 간단한 기준선을 설정했습니다. 이는 관계형 데이터베이스에 대한 ML 연구를 Further encourage합니다.



### Drowning in Documents: Consequences of Scaling Reranker Inferenc (https://arxiv.org/abs/2411.11767)
- **What's New**: 이 논문에서는 기존의 리랭커(reranker) 사용 방식에 대한 새로운 통찰을 제공합니다. 일반적으로 리랭커는 비용이 높지만 효과적이라고 가정되며, 초기 정보 검색 시스템(IR systems)으로부터 검색된 문서들을 재점수화(Re-scoring)하는 데 사용됩니다. 그러나 본 연구는 전체 검색(full retrieval)에 대한 리랭커의 성능을 측정하여 이러한 가정을 도전합니다.

- **Technical Details**: 실험을 통해 발견된 바에 따르면, 가장 발전된 리랭커들은 점진적으로 더 많은 문서를 점수화할 때 점차 성능이 감소하는 경향이 있으며, 특정 한계를 넘어서면 품질이 오히려 저하됩니다. 특정 설정에서 리랭커는 쿼리와의 어휘 또는 의미적 중복이 없는 문서에 높은 점수를 매기는 경우가 빈번합니다.

- **Performance Highlights**: 이 연구 결과는 리랭킹(reranking) 개선을 위한 미래 연구를 촉진할 것으로 기대됩니다.



### The Power of Many: Multi-Agent Multimodal Models for Cultural Image Captioning (https://arxiv.org/abs/2411.11758)
- **What's New**: 이번 연구에서는 Large Multimodal Models (LMMs)와 multi-agent 모델을 결합하여 문화 이미지 캡션 생성이라는 새로운 멀티모달 작업을 수행합니다. 연구의 주요 기여로는, MosAIC이라는 새로운 Multi-Agent 프레임워크를 소개하고, 서로 다른 문화적 페르소나를 가진 LMMs를 이용해 cross-cultural image captioning을 향상시키는 방법을 제안합니다. 또한, 중국, 인도, 루마니아의 이미지를 포함한 문화적으로 풍부한 이미지 캡션 데이터셋을 마련하고, 문화 정보를 평가할 수 있는 새로운 지표를 도입합니다.

- **Technical Details**: MosAIC 프레임워크는 5명의 에이전트로 구성되어 있으며, 각 에이전트는 특정 역할을 수행합니다. 세 개의 Social 에이전트는 중국, 인도, 루마니아의 문화를 대표하고, Moderator 에이전트는 대화의 방향을 유도하며, Summarizer 에이전트는 대화 내용을 요약합니다. 각 대화 라운드는 에이전트들이 서로 질문하고 답변하며, 각 문화를 반영한 상세한 이미지 설명을 만들어내는 구조로 이루어져 있습니다.

- **Performance Highlights**: 연구 결과, 모둠 에이전트 접근 방식이 단일 에이전트 모델보다 다양한 지표에서 우수한 성능을 보인다는 것을 확인했습니다. 에이전트들은 서로의 문화적 배경에 대해 질문하며, 이를 통해 문화적으로 풍부한 알고리즘들을 개발할 수 있게 되었습니다. 만일 후속 연구에서 이러한 방법이 지속적으로 발전된다면, 더 많은 문화적 맥락을 이해하고 반영할 수 있을 것으로 기대됩니다.



### Search, Verify and Feedback: Towards Next Generation Post-training Paradigm of Foundation Models via Verifier Engineering (https://arxiv.org/abs/2411.11504)
- **What's New**: 본 논문에서는 'verifier engineering'이라는 새로운 포스트 트레이닝 패러다임을 제안합니다. 이는 foundation model 시대에 맞춰 설계되었으며, 자동화된 verifier들을 활용하여 모델의 검증 작업을 수행하고 의미 있는 피드백을 제공하도록 구성되었습니다. 기존의 수동 특성 추출 및 데이터 주석 방법을 넘어서, 효율적인 검증 시스템의 설계와 조작을 강조하는 점이 특징입니다.

- **Technical Details**: verifier engineering은 세 가지 핵심 단계인 search, verify, feedback으로 체계적으로 분류할 수 있습니다. 각 단계에서 최신 연구 발전을 종합적으로 검토하며, 특정 지침이 주어지면 후보 응답을 생성하고, 적절한 검증자 조합을 통해 이들을 검증하여 최종적으로 모델 출력 분포를 최적화합니다. 기존의 RLHF(Reinforcement Learning from Human Feedback)와는 달리, 다양한 출처의 verifier를 통합하여 보다 정확하고 일반화된 피드백 신호를 전달합니다.

- **Performance Highlights**: verifier engineering은 foundation models의 개선을 데이터 중심의 접근법에서 시스템적 엔지니어링 문제로 전환합니다. 이는 효과적이고 효율적인 피드백을 보장하기 위한 복잡한 검증 시스템의 설계 및 조정을 강조하며, 기존 방법론보다 정확하고 일반화된 성능 향상을 가능하게 합니다. 또한, 다양한 검증 프로세스를 통해 foundation models의 역량을 극대화할 수 있는 가능성을 제시합니다.



### Re-examining learning linear functions in contex (https://arxiv.org/abs/2411.11465)
- **What's New**: 본 연구는 다양한 크기의 transformer 모델에서의 in-context learning (ICL)에 대한 심층 분석을 제공합니다. 이전 연구와 달리, 모델이 훈련 데이터 외에 일반화하는 데 여러 체계적인 실패를 보였음을 강조합니다.

- **Technical Details**: 연구는 transformer 아키텍처를 가진 30개 이상의 모델을 실험하여, 특히 선형 함수에 대한 ICL의 성능을 평가합니다. 이 과정에서 Gaussian, Bimodal, Uniform 분포와 같은 다양한 훈련 및 테스트 분포를 탐색하였습니다.

- **Performance Highlights**: 모든 transformer 모델은 strictly increasing 또는 strictly decreasing linear function을 ICL하지 못하는 결과를 보였습니다. 또한, 훈련 데이터의 '경계 값'에서 모델 성능이 급격히 저하되며, uniform 분포에서 이러한 경향이 뚜렷하게 나타났습니다.



### Causal Effect of Group Diversity on Redundancy and Coverage in Peer-Reviewing (https://arxiv.org/abs/2411.11437)
- **What's New**: 이 논문은 제출된 논문에 대한 동료 평가(peer review) 과정에서 검토자의 다양성이 각기 다른 관점을 모으고 개인적인 편견을 줄이는 데 어떻게 기여하는지를 탐구합니다. 저자들은 평가 유용성(review utility)을 평가하기 위한 두 가지 지표인 리뷰 커버리지(review coverage)와 리뷰 중복성(review redundancy)을 제안합니다. 이러한 지표를 통해 다양한 검토자 그룹이 리뷰 결과에 미치는 영향을 조사하였으며, 특히 주제의 다양성과 관련된 요소가 광범위한 커버리지를 초래할 수 있음을 발견했습니다.

- **Technical Details**: 저자들은 리뷰어의 속성을 바탕으로 리뷰가 얼마나 포괄적인지와 중복이 얼마나 적은지를 평가하기 위해 인과적 연구(causal study)를 수행했습니다. 약 5,000개의 제출된 논문에서 관찰된 데이터를 활용하여 검토자 다양성이 리뷰의 커버리지와 중복성에 미치는 영향을 분석했습니다. 결과적으로, 주제적으로 다양한 리뷰어 그룹이 종합적인 평가를 더 잘 수행한다는 것을 발견했으며, 특히 지리적 다양성은 리뷰의 중복성을 감소시키는 데 기여하지 않은 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과, 다양한 조직, 직급, 주제 및 출판 네트워크의 리뷰어들로 구성된 그룹이 리뷰의 다양성을 증가시키고 중복성을 줄이는 데 긍정적인 영향을 미친다는 점이 강조됩니다. 특히, 출판 네트워크 기반의 다양성이 매우 중요하게 작용하며, 이는 특정 리뷰 기준 내에서도 다양한 관점을 제공함을 나타냅니다. 이러한 연구는 동료 평가 과정에서 리뷰어 배정의 다양성이 갖는 의미를 재고하게 하며, 보다 효과적이고 투명한 동료 평가 시스템을 구축하는 데 기여할 것입니다.



### MAIRA-Seg: Enhancing Radiology Report Generation with Segmentation-Aware Multimodal Large Language Models (https://arxiv.org/abs/2411.11362)
Comments:
          Accepted as Proceedings Paper at ML4H 2024

- **What's New**: 이 논문에서는 MAIRA-Seg라는 새로운 프레임워크를 소개합니다. 이 모델은 영상 세그멘테이션 마스크를 활용하여 흉부 X-레이(CXR)와 함께 방사선 보고서를 작성하는 데 초점을 맞추고 있습니다. 기존의 멀티모달 대형 언어 모델(MLLM)과의 차별점은 픽셀 수준의 세부 정보를 통합하여 보다 정교한 이미지 해석을 가능하게 한다는 점입니다.

- **Technical Details**: MAIRA-Seg는 CXR 이미지와 연결된 의미론적 세그멘테이션 마스크의 특성을 활용합니다. 전문 세그멘테이션 모델을 훈련시켜 방사선 관련 구조의 마스크를 생성하고, 이를 토대로 MLLM의 입력으로 사용하여 훈련합니다. Osprey 아키텍처를 기반으로 한 세그멘테이션 토큰 추출기를 사용하여 이미지 토큰 및 텍스트 토큰과 세그멘테이션 토큰을 교차하여 결합합니다.

- **Performance Highlights**: MIMIC-CXR 데이터셋을 통한 실험 결과, MAIRA-Seg는 비세그멘테이션 기준선 모델에 비해 성능이 우수한 것으로 나타났습니다. 세그멘테이션 마스크의 사용이 MLLM의 세밀한 추론 능력을 증대시켜 임상 결과에 긍정적인 영향을 미칠 가능성을 보여줍니다. 또한, MAIRA-Seg는 기존 모델들과 비교했을 때 정량적, 정성적으로 우수한 개선을 입증했습니다.



### Debiasing Watermarks for Large Language Models via Maximal Coupling (https://arxiv.org/abs/2411.11203)
- **What's New**: 이 연구에서는 언어 모델을 위한 새로운 "green/red list" 워터마킹 기법을 제안하여 기계 생성 텍스트와 인간 생성 텍스트를 구별할 수 있는 방법을 제시했습니다. 이 방법은 텍스트 품질을 유지하면서도 워터마크 신뢰성을 증가시킵니다.

- **Technical Details**: 제안된 방법은 'green' 토큰의 생성을 위한 확률을 미세하게 증가시킵니다. 최대 결합(maximal coupling)을 적용하여 토큰 분포의 편향을 수정하고, 이는 균일한 동전 던지기를 사용하여 bias correction의 적용 여부를 결정하며, 결과는 유사무작위 워터마크 신호로 임베드됩니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 기법보다 우수한 성능을 보이며 텍스트 품질을 유지하면서도 높은 탐지 가능성을 제공합니다. 또한, 텍스트 품질을 향상시키기 위해 의도적으로 수정된 경우에도 강인성을 보여줍니다.



### Memory-Augmented Multimodal LLMs for Surgical VQA via Self-Contained Inquiry (https://arxiv.org/abs/2411.10937)
- **What's New**: 해당 연구에서는 수술 장면 이해를 향상시키기 위해 SCAN이라는 메모리 증강 프레임워크를 제안하고 있습니다. 기존의 방법들이 여러 객체에 대한 이해력을 제한하고 외부 자원에 의존하는 반면, SCAN은 직접 메모리(Direct Memory, DM)와 간접 메모리(Indirect Memory, IM) 두 가지 메모리 유형을 자동으로 생성하여 수술 맥락을 파악합니다. 이러한 접근 방식은 수술 관련 질문에 대한 응답의 정확성과 신뢰성을 높이는 데 기여합니다.

- **Technical Details**: 수술 시각 질문 응답(Surgical Visual Question Answering, Surgical VQA) 분야에서 메모리 증강된 MLLMs를 활용하여 복잡한 수술 장면을 이해하고 분석하는 데 성공했습니다. S2Can은 사용자가 제시한 질문과 관련된 정보 출처로서 질문-힌트 쌍을 생성하며, 이는 수술 장면의 맥락을 개선하는 데 역할을 합니다. DM은 주어진 질문에 즉각적인 답변을 제공하고, IM은 더 넓은 수술 장면의 이해를 돕습니다.

- **Performance Highlights**: 세 개의 공공 Surgical VQA 데이터셋에서 수행된 실험 결과에 따르면, SCAN은 최신 기술(State-of-the-art, SoTA) 성능을 달성하며 여러 수술 시나리오에서 accuracy와 robustness의 향상을 보여주었습니다. 이러한 성과는 SCAN의 메모리 구성 방식이 효과적임을 강조합니다.



### Large Vision-Language Models for Remote Sensing Visual Question Answering (https://arxiv.org/abs/2411.10857)
- **What's New**: 이 논문에서는 원거리 감지 시각 질문 응답(Remote Sensing Visual Question Answering, RSVQA)을 위한 새로운 방법론을 소개합니다. 기존의 전통적인 접근 방식은 별도의 시각 특성 추출기와 언어 처리 모델에 의존하여 계산적으로 부담이 크고 개방형 질문을 처리하기에 한계가 있었습니다. 본 연구는 생성형 대형 시각-언어 모델(Large Vision-Language Model, LVLM)을 활용하여 RSVQA 과정을 간소화하는 두 단계 훈련 전략을 제안합니다.

- **Technical Details**: 모델의 훈련 과정은 도메인 적응 사전 훈련과 프롬프트 기반 미세 조정으로 구성됩니다. 이러한 방법을 활용함으로써, LVLM은 시각적 정보와 텍스트 입력에 조건화되어 자연어 답변을 생성할 수 있으며, 미리 정의된 답변 카테고리를 요구하지 않습니다. RSVQAxBEN 데이터셋을 통해 모델을 평가하며, 기존 최고 성능 벤치마크에 비해 우수한 성능을 달성하고 있습니다.

- **Performance Highlights**: 평가 결과는 우리 방법이 정확성, 관련성, 유창성 측면에서 더 정확한 답변을 생성한다는 것을 보여줍니다. 또한, 프롬프트 기반 미세 조정 전략이 향상된 일반화 능력을 보이며 새로운 원거리 감지 데이터에 대해서도 안정성을 가지고 대응할 수 있습니다. 요약하자면, 우리의 방법은 LVLM의 힘을 활용하여 RSVQA 응답의 정확성과 품질을 크게 개선합니다.



### Bilingual Text-dependent Speaker Verification with Pre-trained Models for TdSV Challenge 2024 (https://arxiv.org/abs/2411.10828)
Comments:
          5 pages, no figures

- **What's New**: 이 논문은 2024년 이란 Text-dependent Speaker Verification Challenge (TdSV)에 제출된 시스템을 소개합니다. TdSV는 특정 문구가 대상 화자에 의해 발화되었는지를 결정하는 과제를 수행합니다. 연구진은 사전 학습된 모델을 기반으로 한 두 개의 독립적인 하위 시스템을 개발하여, 문구 검증(phrase verification)에서는 잘못된 문구를 거부하는 문구 분류기를 사용하고, 화자 검증(speaker verification)에서는 도메인 적응(domain adaptation)을 통한 ResNet293을 이용해 화자 임베딩을 추출하여 코사인 유사도(cosine similarity)를 계산했습니다.

- **Technical Details**: 문구 검증에서는 다국어 음성 표현 모델을 한층 더 조정(fine-tuning)하여 이란어 및 영어 문구 분류기를 구축했습니다. 화자 검증 시스템에서는 여러 개의 사전 학습된 ResNet 기반 모델과 Whisper 모델을 이용하여 화자 임베딩을 추출하였습니다. 최종 검증 점수는 테스트 임베딩과 등록 임베딩 간의 코사인 유사도를 계산하여 도출하였으며, 성능 향상을 위해 점수 정규화(score normalization) 기법을 사용했습니다.

- **Performance Highlights**: 제안된 시스템은 TdSV 2024의 평가 집합에서 MinDCF가 0.0358로 달성하여 1위에 올랐으며, 화자와 텍스트의 공동 모델링 없이도 경쟁력 있는 성능을 달성할 수 있음을 보여주었습니다.



### Chain-of-Programming (CoP) : Empowering Large Language Models for Geospatial Code Generation (https://arxiv.org/abs/2411.10753)
- **What's New**: 이 논문에서는 지리공간 모델링에 대한 학제 간 수요가 증가하고 대규모 언어 모델(LLMs)의 발전을 바탕으로 지리공간 코드 생성 기술의 최신 발전을 다루고 있습니다. 특히, 기존 LLM이 사용자 요구 사항의 불완전함 및 특정 플랫폼 구문 규칙에 대한 지식 부족으로 인해 코드 생성 과정에서 "code hallucination" 현상에 직면하고 있다는 문제를 제기합니다.

- **Technical Details**: 이 논문에서 제안된 Chain of Programming (CoP) 프레임워크는 코드 생성 과정을 요구 사항 분석, 알고리즘 설계, 코드 구현, 코드 디버깅, 코드 주석 달기 등 다섯 단계로 세분화합니다. CoP는 공유 정보 풀, 지식 기반 검색, 사용자 피드백 메커니즘을 통합하여 모델의 미세 조정 없이 요구 사항에서 코드까지의 엔드 투 엔드(code generation flow)를 형성합니다.

- **Performance Highlights**: CoP 전략은 지리공간 문제 분류 프레임워크 및 평가 기준을 기반으로 하여 생성된 코드의 논리적 명확성, 구문적 정확성 및 실행 가능성을 3.0%에서 48.8%까지 향상시킵니다. 비교 실험 및 소거 실험을 통해 CoP 전략이 다른 최적화 접근법보다 우수하다는 것을 입증하며, 구성 요소의 합리성과 필요성을 확인합니다.



### A Regularized LSTM Method for Detecting Fake News Articles (https://arxiv.org/abs/2411.10713)
Comments:
          6 pages, 7 figures, 2024 IEEE International Conference on Signal Processing, Information, Communication and Systems (SPICSCON)

- **What's New**: 최근 페이크 뉴스의 확산은 심각한 사회 문제로 대두되고 있으며, 본 연구에서는 그러한 문제를 해결하기 위해 고급 머신러닝 모델을 개발했습니다. 총 23,502개의 페이크 뉴스와 21,417개의 진짜 뉴스를 포함한 데이터셋을 활용하여, 다양한 특징을 분석하여 페이크 뉴스를 효과적으로 감지할 수 있는 방법을 제시했습니다.

- **Technical Details**: 이 연구에서는 LSTM(Long Short-Term Memory) 네트워크를 포함한 세 가지 머신러닝 모델을 구현하고 평가했습니다. 각 모델은 특성(feature)으로 제목, 본문(text), 주제(subject), 날짜(date)를 중요하게 고려하였습니다. 첫 번째 모델은 94%의 정확도를 기록, 두 번째 모델은 정규화 기법과 하이퍼파라미터 조정으로 97%, 최종 모델에서는 최적화된 전략을 결합하여 98%의 정확도를 달성했습니다.

- **Performance Highlights**: 우리의 연구는 페이크 뉴스 감지에 있어 높은 정확도를 가진 모델 개발에 기여하였으며, 자연어 처리(NLP) 및 머신러닝 기법의 진전을 보여줍니다. 이러한 모델들은 실전을 위한 신뢰할 수 있는 자동화된 페이크 뉴스 검출 방법으로 활용 가능성을 제시하며, 정보 전파의 신뢰성을 높이는 데 기여할 것입니다.



### BlueLM-V-3B: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices (https://arxiv.org/abs/2411.10640)
Comments:
          21 pages

- **What's New**: 최근의 연구에서 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 도입과 인기가 상승함에 따라, 이러한 모델들이 일상 생활의 여러 측면에서 개선을 가져올 잠재력을 가지고 있다는 점에 주목하고 있습니다. 특히, 모바일 기기는 MLLMs를 효율적으로 배포할 수 있는 플랫폼으로, 일상 업무에 원활하게 통합될 수 있도록 합니다. 하지만 MLLMs를 모바일에서 배포하는 과정에서 메모리 용량과 계산 능력 제한으로 인해 부드럽고 실시간 처리가 도전과제가 되고 있습니다.

- **Technical Details**: 본 논문에서는 BlueLM-V-3B라는 알고리즘 및 시스템 공동 설계 접근법을 제안합니다. 이 모델은 2.7B 파라미터를 가진 언어 모델과 400M 파라미터를 가진 비전 인코더로 구성되어 있습니다. 모바일 플랫폼에서 모델 추론을 최적화하기 위해, 기존의 동적 해상도 방식을 재설계하고 하드웨어에 최적화된 배포를 위한 시스템 최적화를 구현하였습니다.

- **Performance Highlights**: BlueLM-V-3B는 MediaTek Dimensity 9300 프로세서에서 4비트 LLM 가중치 양자화를 통해 초당 24.4 토큰 생성 속도를 달성하며, OpenCompass 벤치마크에서 66.1의 평균 점수를 기록하여 4B 이하 모델 중 가장 높은 성능을 보였습니다. 이러한 성능을 통해 BlueLM-V-3B는 자원 제약 하드웨어에서의 효율적인 운영 가능성을 제시하고 있습니다.



### MTA: Multimodal Task Alignment for BEV Perception and Captioning (https://arxiv.org/abs/2411.10639)
Comments:
          10 pages

- **What's New**: MTA라는 새로운 다중 모달 작업 정렬 프레임워크를 통해 BEV 기반 3D 인식과 자막 작업을 통합하여 성능을 향상시키는 연구를 발표하였습니다.

- **Technical Details**: MTA는 두 가지 주요 구성 요소인 BEV-Language Alignment (BLA)와 Detection-Captioning Alignment (DCA)로 구성되어 있으며, 각각 BEV 장면 표현을 지면 진술 언어 표현과 정렬하고 시각적 출력과 자막 출력을 일관되게 만듭니다. MTA는 기존 BEV 기반 프레임워크에 통합 가능하며, 추가적인 계산 복잡성을 도입하지 않습니다.

- **Performance Highlights**: MTA는 nuScenes와 TOD3Cap 데이터 세트에서 기존 최첨단 기술에 비해 각각 4.9%의 인식 향상과 9.2%의 자막 향상을 달성하였습니다. 이러한 결과는 BEV 기반 인식과 자막 작업의 통합 정렬의 효과를 뒷받침합니다.



### Hysteresis Activation Function for Efficient Inferenc (https://arxiv.org/abs/2411.10573)
Comments:
          Accepted to 4th NeurIPS Efficient Natural Language and Speech Processing Workshop (ENLSP-IV 2024)

- **What's New**: 본 논문에서는 ReLU의 "dying ReLU" 문제를 해결하기 위해 Hysteresis Rectified Linear Unit (HeLU)를 제안합니다. HeLU는 고정된 임계값 대신 가변 임계값을 이용하여 역전파(backpropagation)를 개선하며, 복잡성과 비용을 최소화하면서도 성능을 향상시킵니다.

- **Technical Details**: HeLU는 입력이 음수일 때도 뉴런이 활성화될 수 있도록 하여, 기존 ReLU의 한계를 극복합니다. 이를 통해 backpropagation 동안의 비활성 뉴런도 계속 학습할 수 있는 기회를 제공합니다. HeLU의 설계는 Hysteresis 개념을 기반으로 하며, 이는 컴퓨터 아키텍처에서 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과, HeLU는 CIFAR10에서 +2.96, CIFAR100에서 +2.19, Imagenette에서 +1.23, GLUE 벤치마크에서는 8개 데이터셋에서 +0.51 의 성능 향상을 보였습니다. 이는 ReLU 대비 효율적이고 효과적인 추론을 가능하게 합니다.



### Efficient Alignment of Large Language Models via Data Sampling (https://arxiv.org/abs/2411.10545)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 정렬(alignment) 성과가 데이터 크기에 따라 어떻게 변화하는지를 정리하고, 적은 양의 데이터로도 효과적인 정렬이 가능하다는 점을 확인했습니다. 이를 위해 데이터 서브샘플링을 활용하여 리소스를 절약하는 방법을 제안했습니다.

- **Technical Details**: 연구진은 정보 이론(information theory)에 기초한 방법론을 제안하여, 작은 양의 고품질(high quality) 데이터 서브셋을 선택함으로써 정렬에 필요한 계산과 시간을 줄일 수 있음을 보였습니다. LLM 정렬은 데이터 수집, 관리 및 표준화에서의 높은 비용 문제를 다루기 위해, 다양한 Open 데이터셋을 활용한 실험을 수행했습니다.

- **Performance Highlights**: 제안된 방법론을 사용한 모델은 전체 데이터셋을 사용하여 정렬한 모델과 비슷한 성과를 내면서 10% 이하의 데이터만 사용하여 90% 이상의 비용 및 리소스를 절감했습니다.



### SoftLMs: Efficient Adaptive Low-Rank Approximation of Language Models using Soft-Thresholding Mechanism (https://arxiv.org/abs/2411.10543)
- **What's New**: 본 논문에서는 새로운 압축 방법론인 SoftLM을 제안합니다. 이 방법론은 소프트 임곗값(thresholding) 메커니즘을 사용하여 각 층(layer)의 최적 순위를 동적으로 결정합니다. 또한 BERT, GPT2 및 TinyLlama와 같은 모델에 이 기법을 적용하여 효율적인 추론(inference)을 가능하게 합니다.

- **Technical Details**: SoftLM은 동적 저차 행렬 분해(low-rank matrix decomposition) 접근 방식을 사용하여 각 층의 결과에 따라 순위를 조정합니다. 이 과정은 차원 축소를 위한 학습 가능한 임곗값 파라미터를 포함하여, 표준 미세 조정(fine-tuning) 과정 중에 각 층이 자신의 기여도에 기반하여 최적의 압축 정도를 결정할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, SoftLM은 50%의 파라미터 감소를 달성하면서도 인코더/디코더 속도를 1.33배에서 1.72배 가속화하였고, 원래 모델에 비해 약 1%의 정확도 감소로 경쟁력 있는 성능을 유지했습니다.



### Everything is a Video: Unifying Modalities through Next-Frame Prediction (https://arxiv.org/abs/2411.10503)
Comments:
          10 pages, 10 figures

- **What's New**: 본 논문에서는 전통적인 접근 방식의 한계를 극복하기 위해 다양한 모달리티를 통합하는 새로운 멀티모달 학습 프레임워크를 제안합니다. 이 프레임워크는 다양한 멀티모달 작업을 통합된 다음 프레임 예측 문제로 재구성하여 단일 모델이 모달리티에 관계없이 모든 입력을 처리할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 다양한 입력 및 출력 모달리티를 다음 프레임 예측 단일 작업으로 재구성하는 방법 및 (2) 순수 트랜스포머 기반 모델 아키텍처입니다. 모든 입력과 출력을 64x64 RGB 비디오 시퀀스로 변환하여 일관된 프레임워크를 생성하고, 입력과 예측의 경계를 명확히 하기 위해 구분자 토큰 (||||)을 사용합니다.

- **Performance Highlights**: 이 모델은 텍스트-텍스트, 이미지-텍스트, 비디오-비디오, 비디오-텍스트, 오디오-텍스트 등의 다양한 작업에서 평가되었으며, 최소한의 적응으로 모달리티 간 일반화 능력을 입증하고 있습니다. 실험 결과, 제안된 모델은 추가 데이터로 사전 훈련되지 않은 단일 작업 모델과 유사한 성능을 달성했습니다.



### Hateful Meme Detection through Context-Sensitive Prompting and Fine-Grained Labeling (https://arxiv.org/abs/2411.10480)
Comments:
          AAAI-25 Student Abstract, Oral Presentation

- **What's New**: 이번 연구에서는 멀티모달(multi-modal) 콘텐츠의 자동 조정을 위한 최적화 개념 프레임워크를 제안합니다. 기존에 연구된 방법들과는 달리, 모델 성능 향상뿐만 아니라 모달리티(modality), 프롬프팅(prompting), 라벨링(labeling), 그리고 파인튜닝(fine-tuning)을 포괄하는 엔드투엔드(End-to-End) 최적화 파이프라인을 개발하였습니다.

- **Technical Details**: 제안된 프레임워크의 핵심 원리는 멀티변량 최적화(multi-variate optimization) 문제로, 모달리티, 프롬프팅, 라벨링, 파인튜닝의 성능을 고려합니다. 실험은 다양한 프롬프팅 및 라벨링 전략을 포함해 12개의 실험을 통해 진행되었고, 페이스북 증오 밈 데이터셋을 사용하여 ACCU 및 AUROC로 성과를 평가하였습니다.

- **Performance Highlights**: 모델 M은 68.933%의 정확도와 66.827%의 AUROC로 최고의 성능을 기록하여, 단순히 복잡한 모델이 아니라는 것을 강조합니다. 실험결과 파인튜닝, 범주형 프롬프팅, 이진 라벨링의 조합이 모델 성능 향상에 기여했으며, 독립적인 최적화가 반드시 최고의 결과를 보장하지 않음을 보여주었습니다.



### PhDGPT: Introducing a psychometric and linguistic dataset about how large language models perceive graduate students and professors in psychology (https://arxiv.org/abs/2411.10473)
Comments:
          20 pages, 8 figures. Edoardo Sebastiano De Duro and Enrique Taietta equally contributed to this work

- **What's New**: 이 연구는 PhDGPT라는 새로운 프롬프트 프레임워크와 합성 데이터셋을 소개합니다. 이 데이터셋은 OpenAI의 GPT-3.5가 인식한 PhD 연구자와 교수들의 머신 심리를 포착하고 있습니다. 756,000개의 데이터 포인트로 구성된 이 연구는 우울, 불안 및 스트레스 척도(DASS-42)를 포함하여 학업 및 직업적 맥락에서 심리적 통찰력을 제공합니다.

- **Technical Details**: 이 연구는 15개의 학술 이벤트와 두 개의 직업 수준에 걸쳐 반복된 300회의 실험을 기반으로 합니다. 심리측정 점수와 텍스트 설명을 결합하여 학술적인 인물의 정서적 웰빙을 평가합니다. 연구는 네트워크 심리 측정(methodology combining network psychometrics)와 심리 언어학을 결합하여 LLM의 데이터와 인간의 데이터 간의 차이점을 탐구합니다.

- **Performance Highlights**: 연구 결과, LLM이 인간의 DASS 요인을 80%의 순도로 재구성할 수 있음이 밝혀졌습니다. 또한 LLM이 맥락적 자극에 따라 언어 패턴을 변화시킬 수 있는 능력을 보여줌으로써, LLM의 머신 심리를 평가할 수 있는 새로운 정량적 기회를 제공합니다. 이 연구는 머신 심리학 분야 및 인접 분야에서의 향후 연구를 위한 새로운 데이터셋인 PhDGPT의 중요성을 강조합니다.



### Towards Operationalizing Right to Data Protection (https://arxiv.org/abs/2411.08506)
Comments:
          First two authors contributed equally to this work

- **What's New**: 본 논문에서는 자연어 데이터셋에 불가시한(spurious) 상관관계를 주입하여 데이터셋을 비학습(unlearnable) 가능하게 만드는 RegText라는 새로운 프레임워크를 소개합니다.

- **Technical Details**: RegText는 데이터에서 의미론적(content) 내용을 손상시키지 않으면서 불가시한 노이즈를 추가하여, 언어 모델이 해당 데이터에서 학습하는 것을 제한합니다. 이는 기존 이미지 중심의 접근 방식의 한계를 넘어서는 혁신적인 방법입니다.

- **Performance Highlights**: RegText는 GPT-4o 및 Llama와 같은 최신 모델이 생성된 데이터에서 학습하는 것을 제한하며, 이로 인해 테스트 정확도가 낮아지는 결과를 보입니다. 이로써 공공 데이터 보호를 위한 비학습 텍스트 생성의 가능성을 제시합니다.



### BeeManc at the PLABA Track of TAC-2024: RoBERTa for task 1 -- LLaMA3.1 and GPT-4o for task 2 (https://arxiv.org/abs/2411.07381)
Comments:
          ongoing work - system report

- **What's New**: 이번 PLABA 2024 공유 과제에서 BeeManc 팀은 두 가지 하위 작업을 수행하였습니다. 첫 번째 작업에서는 fine-tuned ReBERTa-Base 모델을 사용하여 생물의학 초록에서 어려운 용어와 약어를 식별하고 분류하였으며, 두 번째 작업에서는 LLaMA 3.1-70B-Instruct 및 GPT-4o 모델을 활용하여 초록을 더 쉽게 이해할 수 있는 언어로 변환했습니다.

- **Technical Details**: PLABA 2024의 첫 번째 작업에서는 주요 용어의 어려움을 평가하고, 적절한 대체 용어를 제공하는 일에 중점을 두었습니다. 이 과정은 Name Entity Recognition (NER) 태스크로, 정확한 용어 목록을 추출하고, 해당 용어에 적합한 대체 방법을 분류하는 멀티 클래스 멀티 레이블 토큰 분류 작업으로 나누어져 진행되었습니다. 또한 RoBERTa 모델을 통해 fine-tuning하여 이러한 분류 작업을 수행하였고, 각 용어에 대해 BIO 태깅 방식을 적용했습니다.

- **Performance Highlights**: BeeManc 팀의 작은 최적화된 RoBERTa-Base 모델은 PLABA-2024에서 두 하위 작업에서 각각 3위와 2위를 기록하였으며, 평균 F1 스코어에서는 9개 평가 시스템 중 1위를 차지했습니다. 또한, LLaMA-3.1-70B-instructed 모델은 Task-2에서 가장 높은 Completeness 점수를 획득하여 전체적인 성과를 높였습니다.



New uploads on arXiv(cs.IR)

### Drowning in Documents: Consequences of Scaling Reranker Inferenc (https://arxiv.org/abs/2411.11767)
- **What's New**: 이 논문에서는 기존의 리랭커(reranker) 사용 방식에 대한 새로운 통찰을 제공합니다. 일반적으로 리랭커는 비용이 높지만 효과적이라고 가정되며, 초기 정보 검색 시스템(IR systems)으로부터 검색된 문서들을 재점수화(Re-scoring)하는 데 사용됩니다. 그러나 본 연구는 전체 검색(full retrieval)에 대한 리랭커의 성능을 측정하여 이러한 가정을 도전합니다.

- **Technical Details**: 실험을 통해 발견된 바에 따르면, 가장 발전된 리랭커들은 점진적으로 더 많은 문서를 점수화할 때 점차 성능이 감소하는 경향이 있으며, 특정 한계를 넘어서면 품질이 오히려 저하됩니다. 특정 설정에서 리랭커는 쿼리와의 어휘 또는 의미적 중복이 없는 문서에 높은 점수를 매기는 경우가 빈번합니다.

- **Performance Highlights**: 이 연구 결과는 리랭킹(reranking) 개선을 위한 미래 연구를 촉진할 것으로 기대됩니다.



### QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou (https://arxiv.org/abs/2411.11739)
Comments:
          Work in progress

- **What's New**: 최근 멀티모달 대형 모델의 발전으로, 추천 시스템에서 사용자 관심 모델링을 위해 멀티모달 정보를 활용하는 가능성이 주목받고 있습니다. 이 논문에서는 Kuaishou 플랫폼을 위한 맞춤형 다중 모달 정보의 학습을 위한 정량적 멀티모달 프레임워크(QARM)를 소개합니다. QARM은 하향식 추천 모델에 최적화하여 비즈니스 특성을 고려한 다양한 멀티모달 표현 일치 메커니즘과 코드 메커니즘을 적용합니다.

- **Technical Details**: QARM은 두 가지 주요 메커니즘으로 구성됩니다. 첫째, 아이템 정렬 메커니즘은 다중 모달 모델의 사전 학습 결과를 비즈니스 데이터에 맞게 미세 조정하여 표현 일치를 최대화합니다. 둘째, 정량적 코드 메커니즘은 코드 해싱과 직통 추정기를 활용하여 다중 모달 표현을 학습 가능한 의미적 ID로 변환합니다. 이를 통해 기존의 정적 멀티모달 표현이 가진 제한사항을 해결하고 지원하는 방법을 제공합니다.

- **Performance Highlights**: QARM은 Kuaishou에서 400만 명의 활성 사용자를 지원하며 정량적 실험을 통해 광고 수익을 9.704% 증가시키고 온라인 쇼핑의 GMV를 2.296% 증가시킨 것으로 나타났습니다. 이러한 결과는 QARM이 다양한 서비스에서 유효함을 입증하며, 추천 시스템의 성능을 극대화하기 위한 효과적인 솔루션임을 보여줍니다.



### Collaborative Contrastive Network for Click-Through Rate Prediction (https://arxiv.org/abs/2411.11508)
- **What's New**: 이번 연구에서는 Trigger-Induced Recommendation (TIR) 문제를 해결하기 위해 Collaborative Contrastive Network (CCN)이라는 새로운 CTR 예측 방법을 제안합니다. CCN은 사용자와 트리거 아이템 간의 관계를 모델링하여, 사용자의 관심사와 비관심사를 효과적으로 구분합니다. 이 방법은 기존 방식의 단점을 보완하며, 짧은 기간 동안만 존재하는 미니 앱들에도 적합합니다.

- **Technical Details**: CCN은 클릭 로그를 통해 얻은 협력적 관계(co-click/co-non-click) 및 비협력적 관계(mono-click)를 활용하여 사용자 관심사를 효과적으로 모델링합니다. 이 모델은 양성 및 음성 집합을 통해 아이템 간의 관계를 학습하며, Attraction Loss와 Repulsion Loss를 통해 최적화됩니다. 이를 통해 CCN은 사용자의 관심을 반영한 더 정확한 클릭 확률 예측을 가능하게 합니다.

- **Performance Highlights**: 대규모 실제 데이터에 대한 온라인 A/B 테스트 결과, CCN은 타오바오에서 CTR을 12.3% 증가시키고 주문량을 12.7% 증가시켜 새로운 최첨단 성능을 기록했습니다. 이는 CCN이 사용자 관심을 보다 잘 반영한 결과이며, 이로 인해 미니 앱 내에서의 추천 품질이 향상되었습니다.



### All-domain Moveline Evolution Network for Click-Through Rate Prediction (https://arxiv.org/abs/2411.11502)
- **What's New**: 이 논문에서는 All-domain Moveline Evolution Network (AMEN)을 제안하여 사용자 의도를 all-domain moveline의 관점에서 재정의하고 있습니다. 기존의 클릭률(CTR) 예측 방법이 아이템 수준의 상호작용에 집중했던 반면, AMEN은 장면 수준에서의 사용자 행동의 연결성을 강조합니다. 이 접근법은 사용자 행동의 다양성을 포착하며, 장면별 행동이 CTR 예측에 미치는 영향을 구별하여 더욱 정교한 모델을 구현합니다.

- **Technical Details**: AMEN은 아이템과 장면 간의 상호작용을 동질적 표현 공간으로 정렬하는 CTR 예측 모듈과 Temporal Sequential Pairwise (TSP) 메커니즘으로 구성됩니다. TSP 메커니즘은 사용자 moveline의 상태 변화를 포착하여 특정 장면 수준 행동이 향후 아이템 수준 행동에 미치는 영향을 정밀하게 평가합니다. 이는 기존의 짝지어 학습 전략과는 다른 접근법으로, 다양한 시간 지점에서 수집된 피드백 레이블을 가진 아이템 간 쌍을 형성하여 특징을 강조합니다.

- **Performance Highlights**: 온라인 A/B 테스트 결과, AMEN은 Taobao 마케팅 채널에서 CTCVR(클릭 후 전환율)을 11.6% 증가시켜 새로운 최첨단 성능을 기록하였습니다. 이 개선은 제안된 모델이 사용자 의도를 얼마나 잘 반영하는지를 잘 보여줍니다. AMEN의 성능은 기존 모델들과 차별화된 사용자 행동 분석을 통해 이루어진 것으로, 전방향 사용자 데이터의 활용이 효과적임을 보여줍니다.



### Controlling Diversity at Inference: Guiding Diffusion Recommender Models with Targeted Category Preferences (https://arxiv.org/abs/2411.11240)
Comments:
          KDD 2025

- **What's New**: 본 논문에서는 추천 시스템의 정확성과 다양성 간의 균형을 조정할 수 있는 새로운 방법인 D3Rec(Disentangled Diffusion model for Diversified Recommendation)를 제안합니다. D3Rec는 카테고리 선호도를 기반으로 추천을 생성하고, 추론 과정 중에 이 카테고리 선호도를 조절할 수 있는 기능을 갖추고 있습니다. 또한, 사용자의 기분이나 비즈니스 전략에 따라 다양한 카테고리 선호도에 적응할 수 있는 유연성을 제공합니다.

- **Technical Details**: D3Rec는 두 가지 복잡을 활용하는 diffusion framework를 통해 추천을 생성합니다. 첫 번째로, 사용자 상호작용에서 숨겨진 카테고리 선호도를 노이즈를 추가하여 제거하고, 두 번째로 생성 단계에서 원하는 카테고리 선호도를 반영하여 추천을 생성합니다. 또한, 두 개의 보조 작업을 통해 생성된 추천이 대상 카테고리 선호도와 일치하도록 설계되어 있습니다.

- **Performance Highlights**: D3Rec는 실제 데이터와 합성 데이터셋에 대한 광범위한 실험을 통해 추천 다양성을 효과적으로 제어하는 능력을 입증했습니다. 이 방법은 사용자가 원하는 카테고리 분포를 쉽게 조정할 수 있어 다양한 실제 상황에 적용할 수 있으며, 추천의 다양성을 필요에 따라 유동적으로 조정할 수 있습니다.



### Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning (https://arxiv.org/abs/2411.11225)
Comments:
          11 pages, 4 figures, to be published in KDD '25

- **What's New**: 이번 연구에서는 스트리밍 데이터 환경에서 아이템의 콜드스타트 문제를 해결하는 새로운 모델 불가지론적 알고리즘, Popularity-Aware Meta-learning (PAM)을 제안합니다. PAM은 들어오는 데이터를 미리 정의된 아이템 인기 기준에 따라 다양한 메타학습 작업으로 나누어 처리합니다. 이를 통해 모델은 인기 수준에 따라 행동 관련 특징과 콘텐츠 관련 특징을 구별하고 재조정하여 콜드스타트 샘플을 위한 추천을 적응적으로 수행합니다.

- **Technical Details**: PAM 모델은 메타 학습의 기능을 활용하여 각 작업의 메타 파라미터를 공유하고, 개인화된 파라미터를 생성합니다. 이는 콜드스타트 작업의 성능을 최적화하면서 실시간의 스트리밍 데이터를 잃지 않도록 합니다. 뿐만 아니라, 데이터 증강(data augmentation) 기법과 낮은 인기에 의해 고통받는 과제를 위한 추가적인 자기지도 손실(self-supervised loss)이 도입되어, 콜드 스타트 샘플 부족 문제를 효과적으로 완화합니다.

- **Performance Highlights**: 실험 결과, PAM은 다양한 공공 데이터셋에서 기존의 기본 알고리즘들과 비교하여 콜드스타트 문제를 다루는 데 있어 우수한 성능을 보였습니다. 특히, PAM은 추가적인 파인튜닝 없이 실시간 추천의 성능을 현저히 향상시킵니다. 이는 온라인 스트리밍 데이터 환경에서 일반적인 콜드스타트 문제를 해결하는 데 매우 효과적임을 입증합니다.



### ForPKG-1.0: A Framework for Constructing Forestry Policy Knowledge Graph and Application Analysis (https://arxiv.org/abs/2411.11090)
Comments:
          22 pages

- **What's New**: 본 논문은 정책 지식 그래프(policy knowledge graph)를 구축하기 위한 새로운 프레임워크를 제안합니다. 특히 산림 분야에 중점을 두고 미세 조정된 정책 도메인 온톨로지(fine-grained ontology)를 설계하고, 비지도 정책 정보 추출 방법을 제안하며, 최종적으로 완전한 산림 정책 지식 그래프를 구성합니다. 이 연구는 관련 분야의 정책 지식 그래프 구축 방식에 대한 부족한 연구를 보완하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 요소로 구성됩니다: 미세 조정된 산림 정책 도메인 온톨로지, 비지도 학습을 기반으로 한 정책 정보 추출 방법, 그리고 최종 산림 정책 지식 그래프입니다. 실험 결과, 제안된 온톨로지는 우수한 표현성과 확장성을 보여주며, 정책 정보 추출 방법은 기존의 비지도 방법들보다 더 나은 성능을 발휘하였습니다.

- **Performance Highlights**: 이 연구에서 구축된 지식 그래프는 대형 언어 모델(large language models)의 정보 검색 및 생성 작업에 활용될 수 있으며, 실용적인 응용 가치를 확인하였습니다. 또한, 정책 지식 그래프는 오픈 소스 플랫폼에 배포될 예정으로, 산림 정책 관련 지능형 시스템의 기초 지식 베이스로 사용될 수 있으며, 학술 연구에도 기여할 수 있습니다.



### Exploring Feature-based Knowledge Distillation For Recommender System: A Frequency Perspectiv (https://arxiv.org/abs/2411.10676)
- **What's New**: 이 논문에서는 주파수 관점(frequency perspective)에서 추천 시스템에 대한 feature-based 지식 증류(knowledge distillation)를 분석합니다. 지식을 여러 주파수 구성 요소로 정의하고, 이러한 구성 요소들을 동등하게 최소화하는 기존 방법이 중요한 지식이 간과되는 문제를 다루기 위해 FreqD라는 새로운 가벼운 지식 재조정 방법을 제안합니다.

- **Technical Details**: 기존의 feature-based knowledge distillation 방법은 지식 손실을 동등하게 최소화하는 데 초점을 맞추고 있습니다. 그러나 필자는 지식의 중요도를 고려하여 지식의 가중치를 재조정하는 방법을 제안합니다. FreqD는 각 지식에 대한 손실을 계산할 필요 없이 복잡한 계산 비용을 피할 수 있도록 설계되었습니다.

- **Performance Highlights**: Extensive experiments show that FreqD consistently outperforms state-of-the-art knowledge distillation methods across various public datasets. FreqD는 추천 성능 향상에 강력한 효과를 발휘해, 실제 사용에 적합한 경량화된 해결책으로써의 가능성을 보여줍니다.



### Do Captioning Metrics Reflect Music Semantic Alignment? (https://arxiv.org/abs/2411.11692)
Comments:
          International Society for Music Information Retrieval (ISMIR) 2024, Late Breaking Demo (LBD)

- **What's New**: 음악 캡션 작업은 고급 언어 생성 모델의 발전으로 인해 떠오르고 있지만, 평가 기준으로 쓰이는 BLEU, METEOR, ROUGE 등의 전통적인 메트릭스가 새로운 분야에서 부적절하다는 점을 지적합니다. 이러한 메트릭스들은 의미보다는 구문적 유사성에 의존하여 왜곡된 평가를 초래할 수 있습니다.

- **Technical Details**: 이 연구는 음악 캡션을 평가하기 위한 전통적인 메트릭스의 한계를 분명히 보여주기 위해 진행된 인적 평가 연구를 담고 있습니다. Amazon Mechanical Turk를 통해 50명의 참가자를 모집하고, MusicCaps 데이터셋에서 무작위로 30개의 오디오 클립을 샘플링하여 원래 캡션과 추론 및 패러프레이즈 캡션을 비교하여 평가합니다.

- **Performance Highlights**: 연구 결과, 전통적 메트릭스는 실제 캡션의 의미를 정확히 반영하지 못하며, 특정 구문적 변형이 이루어진 캡션이 높은 점수를 받는 경우 등 잘못된 평가를 가져오는 경향이 있음을 보여줍니다. 이러한 발견은 음악 캡션의 평가 기준을 재검토할 필요성을 강조합니다.



### Few-shot Model Extraction Attacks against Sequential Recommender Systems (https://arxiv.org/abs/2411.11677)
- **What's New**: 본 연구는 sequential recommender systems에 대한 few-shot model extraction 공격 프레임워크를 새롭게 제안합니다. 이는 적은 양의 데이터(최대 10%)를 사용하여 우수한 surrogate model을 구축할 수 있는 방법을 다룹니다.

- **Technical Details**: 제안된 프레임워크는 autoregressive augmentation generation strategy와 bidirectional repair loss-facilitated model distillation procedure의 두 가지 구성 요소로 이루어져 있습니다. 첫 번째 구성 요소는 raw data의 분포에 근접한 synthetic data를 생성하는 데 중점을 두며, 두 번째 구성 요소는 surrogate model의 오류를 수정하고 victim model의 지식을 효과적으로 전이하는데 기여합니다.

- **Performance Highlights**: 세 개의 공개 데이터셋에 대한 실험을 통해 제안된 few-shot model extraction 프레임워크가 victim model과 높은 기능적 유사성을 가진 surrogate 모델을 구축할 수 있음을 입증했습니다. 이 연구는 적은 데이터로도 효과적인 공격을 가능케 함을 보여줍니다.



### Empowering Meta-Analysis: Leveraging Large Language Models for Scientific Synthesis (https://arxiv.org/abs/2411.10878)
Comments:
          Accepted in 2024 IEEE International Conference on Big Data (IEEE BigData)

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)을 활용하여 과학 문서의 메타-분석(메타-분석) 자동화를 탐구합니다. 전통적으로 메타-분석은 여러 연구 결과를 종합하여 심층적인 이해를 제공하지만, 수작업으로 진행할 경우 시간 소모적이고 인적 오류에 취약합니다. 이에 본 연구는 LLM을 대규모 과학 데이터셋으로 미세 조정하여 데이터를 효과적으로 처리하고 구조적 데이터 추출의 도전을 해결하는 새로운 접근법을 제시합니다.

- **Technical Details**: 이 자동화된 메타-분석 프로세스는 Retrieval Augmented Generation (RAG)을 통합하여 최적화됩니다. LLM은 새로운 손실 메트릭(Inverse Cosine Distance, ICD)을 통해 훈련되어 고차원 문맥 데이터에 적합한 데이터 추출 패턴을 학습합니다. 이 방법은 메타-분석 생성을 위한 LLM의 성능을 향상시키며, 구조적 초록 생성에서의 품질과 효율성을 보장합니다.

- **Performance Highlights**: 사람이 평가한 결과, 미세 조정된 LLM 모델이 비 미세 조정 모델보다 우수한 성능을 보여, 87.6%의 관련 메타-분석 초록을 생성했습니다. 평가에 따르면 맥락의 적합성이 4.56%에서 1.9%로 감소하여,본 연구는 메타-분석 자동화의 효율성과 신뢰도를 향상시키는 데 기여한 것으로 나타났습니다.



### Any2Any: Incomplete Multimodal Retrieval with Conformal Prediction (https://arxiv.org/abs/2411.10513)
- **What's New**: 본 논문에서는 Any2Any라는 새로운 멀티모달 검색 프레임워크를 제안합니다. 이 프레임워크는 불완전한 모달리티를 가진 쿼리와 참조 인스턴스 모두에 대해 검색을 가능하게 합니다. 기존의 이중모달 검색 방식을 넘어 여러 모달리티에 대해 훈련 없이 활용 가능합니다.

- **Technical Details**: Any2Any는 크로스 모달 인코더를 사용하여 인스턴스 간의 쌍별 유사도를 계산하고, 두 단계 보정 과정을 통해 유사도를 정렬합니다. 첫 번째 단계에서는 확장 예측(conformal prediction)을 통해 유사도를 정규화하고, 두 번째 단계에서는 모든 모달리티 쌍의 올바른 검색 확률을 나타내는 스칼라 값을 생성합니다. 이를 통해 다양한 조합의 불완전한 모달리티를 가진 인스턴스 간의 직접 비교를 가능하게 합니다.

- **Performance Highlights**: KITTI 데이터셋에 대한 실험 결과 Any2Any는 Recall@5의 성능이 35%로 완전 모달리티를 가진 기준 모델들과 견줄 수 있는 성능을 보였습니다. 이를 통해 Any2Any는 다양한 크로스 모달 인코더에 대한 일반화 능력을 갖추고 있음을 보여주며, 멀티모달 데이터셋에서의 검색 가능성을 높입니다.



New uploads on arXiv(cs.CV)

### UniHands: Unifying Various Wild-Collected Keypoints for Personalized Hand Reconstruction (https://arxiv.org/abs/2411.11845)
- **What's New**: UniHands는 다양한 출처에서 수집된 손 키포인트를 기반으로 개인화된 손 모델을 생성하는 새로운 방법론입니다. 기존의 neural implicit representation 방법 이외에도, 이 연구는 MANO와 NIMBLE 모델을 활용하여 더욱 확장 가능하고 다재다능한 솔루션을 제공합니다. 이는 통합된 손 관절을 도출함으로써 다양한 손 관련 작업에 원활하게 통합할 수 있는 구조를 제공합니다.

- **Technical Details**: UniHands는 두 주요 구성요소로 구성되며, 하나는 야생에서 수집된 키포인트로부터 손 메쉬를 복원하는 것이고, 다른 하나는 복원된 메쉬로부터 손 관절을 파생하는 것입니다. 또한, 손 모형의 포즈와 형태(θ, β) 파라미터를 정제하기 위해 Coarse-to-Fine 최적화 방법을 설계하였습니다. 이 모델은 MANO와 NIMBLE 메쉬를 비교하고 정렬하여 접합된 해골을 생성하는 방법을 포함합니다.

- **Performance Highlights**: FreiHAND와 InterHand2.6M 데이터셋에 대한 실험 결과는 손 메쉬 복원이 매우 높은 정확도로 이루어졌음을 보여줍니다. 참가자들의 평가에서는 UniHands가 기존 구성보다 자연스러운 외관을 제공하는 것으로 나타났으며, 실험에서 수집된 데이터는 높은 현실성, 정확성, 자연성을 보여주었습니다. 특히, 통합된 MANO-NIMBLE 관절이 애니메이션에서 가장 자연스러운 모습을 제공한다고 평가되었습니다.



### Generative World Explorer (https://arxiv.org/abs/2411.11844)
Comments:
          Website: this http URL

- **What's New**: 이 논문에서는 인공지능(AI) 에이전트가 인간처럼 마음속에서 미지의 영역을 탐색하도록 할 수 있는 새로운 프레임워크인 Generative World Explorer (Genex)를 제안합니다. Genex는 에이전트가 대규모 3D 세계를 정신적으로 탐험하며 상상된 관측 결과를 얻어 자신의 신념을 업데이트할 수 있도록 돕습니다. 이 프로세스는 에이전트가 물리적으로 환경을 탐색할 필요 없이 정보에 기반한 결정을 내릴 수 있게 합니다.

- **Technical Details**: Genex는 에이전트의 현재 1인칭 시각에 따라 동작 방향을 입력으로 하여 미래의 관측을 생성하는 비디오 생성 모델입니다. 이 모델은 파놉틱(panoramic) 표현을 활용하여 비디오 확산 모델을 훈련시키고, 이는 긴 거리 탐색 시에도 높은 생성 품질과 3D 일관성을 유지하는 데 기여합니다. Genex는 POMDP(Partially Observable Markov Decision Process)의 확장으로 에이전트의 행동을 정의하며, 다중 에이전트 시나리오로 자연스럽게 확장해 다른 에이전트의 신념에 따라 자신의 신념을 업데이트 할 수 있습니다.

- **Performance Highlights**: 실험 결과, Genex는 대규모 가상 물리 세계의 긴 탐색 동안 고품질의 일관된 관측 결과를 생성할 수 있으며, 이러한 관측으로 업데이트된 신념은 기존의 의사결정 모델(Large Language Model - LLM 에이전트)이 더 나은 계획을 세우는데 도움을 줄 수 있다는 것을 보여줍니다. 이러한 성과는 Genex가 다중 에이전트 의사결정에도 유용한 응용 가능성을 지닌다는 것을 시사합니다.



### LightFFDNets: Lightweight Convolutional Neural Networks for Rapid Facial Forgery Detection (https://arxiv.org/abs/2411.11826)
Comments:
          13 pages, 6 figures, 10 tables

- **What's New**: 이 연구는 얼굴 이미지 위조 탐지를 위한 새로운 경량 딥러닝 모델인 LightFFDNets를 제안하고, 실제와 위조된 얼굴 이미지를 인식하는 데 있어 높은 정확도와 계산 효율성을 보여준다. 기존의 CNN 아키텍처와 비교한 연구 결과, 제안된 모델이 향상된 성능을 보임을 입증하였다. 연구에서 사용된 데이터셋은 Fake-Vs-Real-Faces (Hard)와 140k Real and Fake Faces로,두 클래스 모두의 얼굴 이미지를 포함하고 있다.

- **Technical Details**: 연구에서는 특징 추출을 수행하는 CNN 모델이 사용되며, 실제 및 위조된 얼굴 이미지를 탐지하기 위해 두 개의 경량 모델이 개발되었다. 제안된 모델은 최소한의 레이어로 구성되어 있으며, 여러 개의 사전 훈련된 CNN 아키텍처와의 결과를 비교하였다. GAN(Generative Adversarial Networks) 기술이 얼굴 생성 및 위조 탐지에 사용되는 방법도 검토하고 있다.

- **Performance Highlights**: 제안된 경량 딥러닝 모델은 빠른 계산 시간에도 불구하고 정확한 얼굴 위조 탐지를 제공하며, 다른 두 클래스 객체 인식 문제에도 적용 가능성을 보인다. 실험 결과, 기존 모델들과의 비교에서 연산 시간이 크게 개선되었음을 확인할 수 있었다. 이러한 성과는 얼굴 이미지 식별의 안전성 및 신뢰성을 높이는 데 기여할 것으로 기대된다.



### The Power of Many: Multi-Agent Multimodal Models for Cultural Image Captioning (https://arxiv.org/abs/2411.11758)
- **What's New**: 이번 연구에서는 Large Multimodal Models (LMMs)와 multi-agent 모델을 결합하여 문화 이미지 캡션 생성이라는 새로운 멀티모달 작업을 수행합니다. 연구의 주요 기여로는, MosAIC이라는 새로운 Multi-Agent 프레임워크를 소개하고, 서로 다른 문화적 페르소나를 가진 LMMs를 이용해 cross-cultural image captioning을 향상시키는 방법을 제안합니다. 또한, 중국, 인도, 루마니아의 이미지를 포함한 문화적으로 풍부한 이미지 캡션 데이터셋을 마련하고, 문화 정보를 평가할 수 있는 새로운 지표를 도입합니다.

- **Technical Details**: MosAIC 프레임워크는 5명의 에이전트로 구성되어 있으며, 각 에이전트는 특정 역할을 수행합니다. 세 개의 Social 에이전트는 중국, 인도, 루마니아의 문화를 대표하고, Moderator 에이전트는 대화의 방향을 유도하며, Summarizer 에이전트는 대화 내용을 요약합니다. 각 대화 라운드는 에이전트들이 서로 질문하고 답변하며, 각 문화를 반영한 상세한 이미지 설명을 만들어내는 구조로 이루어져 있습니다.

- **Performance Highlights**: 연구 결과, 모둠 에이전트 접근 방식이 단일 에이전트 모델보다 다양한 지표에서 우수한 성능을 보인다는 것을 확인했습니다. 에이전트들은 서로의 문화적 배경에 대해 질문하며, 이를 통해 문화적으로 풍부한 알고리즘들을 개발할 수 있게 되었습니다. 만일 후속 연구에서 이러한 방법이 지속적으로 발전된다면, 더 많은 문화적 맥락을 이해하고 반영할 수 있을 것으로 기대됩니다.



### WoodYOLO: A Novel Object Detector for Wood Species Detection in Microscopic Images (https://arxiv.org/abs/2411.11738)
- **What's New**: 이번 연구에서는 Microscopic wood fiber analysis에 특화된 새로운 객체 탐지 알고리즘인 WoodYOLO를 소개합니다. WoodYOLO는 기존의 YOLO 아키텍처를 기반으로 하여, 대형 고해상도 현미경 이미지 처리와 관심 세포 유형(vessel elements)의 고정확도 로컬라이제이션을 위해 최적화된 방법을 적용합니다. 이 알고리즘은 YOLOv10 및 YOLOv7보다 각각 12.9%와 6.5% 높은 F2 점수를 기록하며, 자동화된 목재 세포 유형 로컬라이제이션 기능을 개선했습니다.

- **Technical Details**: WoodYOLO는 고해상도 현미경 이미지에 최적화된 커스터마이즈된 YOLO 기반 아키텍처로 구성되어 있습니다. 또한, 최대 너비와 높이만 정의하는 새로운 앵커 박스 지정 방법을 도입하여, F2 점수를 0.7% 향상시킵니다. 현대 객체 탐지기에서 다양한 아키텍처 결정을 포괄적으로 평가하여 COCO(Lin et al., 2015)와 같은 일반적인 데이터셋에 최적화된 방법이 실제 데이터셋에서는 반드시 성능 향상으로 이어지지 않음을 발견했습니다.

- **Performance Highlights**: WoodYOLO는 자동화된 목재 종 식별 능력을 개선하여, 규제 준수, 지속 가능한 산림 관리, 그리고 생물 다양성 보존 노력을 지원하는 데 기여합니다. 연구 결과, 이 새로운 알고리즘은 microscopy 이미지를 활용한 섬유 재료의 목재 종 식별을 위한 신뢰할 수 있는 방법으로 자리잡을 것으로 기대되며, 고해상도 이미지 처리에 대한 요구를 충족할 수 있습니다.



### RAWMamba: Unified sRGB-to-RAW De-rendering With State Space Mod (https://arxiv.org/abs/2411.11717)
- **What's New**: 이 논문에서는 이미지와 비디오 도메인에서 sRGB를 RAW로 변환하기 위한 첫 번째 통합 Mamba 기반 프레임워크인 RAWMamba를 제안합니다. RAWMamba는 다양한 메타데이터 유형을 통합된 표현으로 조화시키는 Unified Metadata Embedding (UME) 모듈을 중심으로 구축되었습니다. 또한, 메타데이터의 효과적인 전파를 위한 Local Tone-Aware Mamba (LTA-Mamba) 모듈이 도입되어, 전반적인 품질 향상을 목표로 하고 있습니다.

- **Technical Details**: RAWMamba의 UME 모듈은 모든 메타데이터를 통합된 (sRGB, RAW) 쌍으로 처리하여 복원 과정에서 기준 정보를 추출합니다. 이 모듈은 글로벌 메타데이터 정보를 포착하는 affinity matrix와 지역 메타데이터 정보를 추출하는 dynamic position encoding을 활용하는 두 개의 분기를 포함하고 있습니다. LTA-Mamba 모듈은 스팸 데이터를 활용하여 잔여 메타데이터의 일관성을 증대시키며, 기존의 Mamba 모델을 바탕으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, RAWMamba는 RVD-Part2 비디오 데이터셋과 CAM 이미지 데이터셋에서 각각 PSNR이 2.20dB, 3.37dB 향상된 성능을 보여주며, 기존의 특정 태스크 모델을 초월하였습니다. 이 연구는 이미지와 비디오 데이터의 효과적인 복원을 가능하게 하여, 다양한 컴퓨터 비전 작업에 기여할 것으로 기대됩니다.



### MC-LLaVA: Multi-Concept Personalized Vision-Language Mod (https://arxiv.org/abs/2411.11706)
- **What's New**: 이번 연구에서는 다중 개념 개인화(Multi-concept personalization)를 위한 새로운 방법인 MC-LLaVA를 제안합니다. 기존의 연구들이 주로 단일 개념에 초점을 맞추었던 반면, MC-LLaVA는 여러 개념을 동시에 처리할 수 있는 능력을 보여줘 사용자의 제공된 정보에 기반하여 정확한 응답을 생성합니다. 또한, 고품질 다중 개념 개인화 데이터셋도 함께 발표하여 연구의 기초 자료를 제공하는 데 기여합니다.

- **Technical Details**: MC-LLaVA는 하나의 훈련 단계에서 여러 개념을 함께 고려하여 학습하는 방식으로, VLM(Vision-Language Model)의 시각 인코더를 통해 개념 이미지들을 전달하여 개념 토큰을 초기화합니다. 이는 공동 학습(joint training)의 비용을 줄이고, 개념의 표상을 개선하며, 훈련 속도를 가속화하는데 기여합니다. 연구팀은 약 1,600개의 이미지와 질문-답변 샘플을 포함하는 데이터셋을 마련하여, 이 데이터셋을 활용해 다양한 비전-언어 작업에서 MC-LLaVA의 성능을 평가했습니다.

- **Performance Highlights**: MC-LLaVA의 실험 결과는 여러 작업 유형에서 괄목할 만한 성능 향상을 보여줍니다. 특히 다중 개념 인식, 질문 응답(QA), 캡셔닝(captioning) 등에서 상태 수익(state-of-the-art) 성과를 달성하여 VLM이 사용자 맞춤형 비서로 발전할 가능성을 열었습니다. 이러한 결과는 개인화된 VLM 연구의 새로운 방향을 제시하며, 향후 연구에 있어 더 많은 발전이 기대됩니다.



### From Spectra to Geography: Intelligent Mapping of RRUFF Mineral Data (https://arxiv.org/abs/2411.11693)
- **What's New**: 이 연구는 광물 샘플의 지리적 기원을 정확히 결정하는 데 필수적인 새로운 기계 학습 프레임워크를 도입합니다. RRUFF 데이터베이스의 방대한 Raman 분광 데이터를 활용하여, 단일 차원 ConvNeXt1D 신경망 구조를 사용하여 101개 국가에 걸친 32,900개 이상의 광물 샘플을 분류합니다. 이 연구는 초매립된 Raman 스펙트럼 데이터의 효과적인 기계 학습 응용을 위한 새로운 길을 제시하고 있습니다.

- **Technical Details**: ConvNeXt1D 아키텍처는 순수한 합성곱 신경망(Convolutional Neural Network)으로, 기존의 ResNet 아키텍처를 현대화하는 과정에서 비전 트랜스포머(Vision Transformers)의 디자인 요소를 통합하고 있습니다. 이 시스템은 하나의 차원 데이터에 최적화된 ConvNeXt 아키텍처로, 파라미터로는 깊이와 블록 수를 조정하여 고유의 특성을 발휘합니다. 또한, 데이터 전처리를 통해 스펙트럼이 균일한 입력 차원으로 정규화됩니다.

- **Performance Highlights**: ConvNeXt1D 모델은 93%의 평균 분류 정확도를 기록하며, Raman 스펙트럼에서의 지리적 패턴을 효과적으로 캡처할 수 있음을 입증했습니다. 데이터셋의 약 99.85%가 정확한 위도 및 경도 좌표를 포함하고 있어, 다양한 광물 특성을 학습하고 평가하는데 유리합니다. 이 연구는 광물 데이터를 활용한 기계 학습 모델 훈련에 있어 귀중한 기초 자료를 제공합니다.



### Towards Degradation-Robust Reconstruction in Generalizable NeRF (https://arxiv.org/abs/2411.11691)
- **What's New**: 이번 연구에서는 Generalizable Neural Radiance Field (GNeRF)가 다양한 장면에서 효과적으로 작동하지만, 소스 이미지의 열화(degradation)에 대해 강인성이 부족하다는 문제를 다루고 있습니다. 이를 해결하기 위해 Objaverse Blur Dataset을 구축하였으며, 이 데이터셋은 50,000개의 이미지로 구성되어 다양한 블러(blur) 열화 수준을 포함하고 있습니다. 또한, GNeRF의 열화 강인성을 향상시키기 위한 모델 비종속 모듈을 설계했습니다.

- **Technical Details**: 제안된 모듈은 가벼운 깊이 추정기(depth estimator)와 노이즈 제거기(denoiser)를 통해 3D 감지 기능(3D-aware features)을 추출하여 GNeRF의 성능을 개선합니다. 이 모듈은 다양한 열화 유형(type) 및 수준(level)에 대해 정량적(quantitative) 및 시각적(visual) 품질에서 기존의 여러 인기 방법들보다 우수한 성능을 보여줍니다. GNeRF의 차별화된 접근 방식으로는 열화에 강한 학습이 가능합니다.

- **Performance Highlights**: GNeRF 모델에 적용된 새로운 모듈을 통해 다양한 채도에서 감지 성능이 크게 향상되었습니다. 특히, 제안된 접근 방식은 비로소 실제 환경에서의 이미지 열화에 강한 모델 개발을 위한 기초 자료를 제공할 수 있습니다. 이로 인해 향후 3D 재구성(3D reconstruction) 작업에서 더욱 높은 신뢰성을 기대할 수 있습니다.



### SP${ }^3$ : Superpixel-propagated pseudo-label learning for weakly semi-supervised medical image segmentation (https://arxiv.org/abs/2411.11636)
Comments:
          10 pages, 7 figures. Under Review

- **What's New**: 이 논문은 약한 반주석식 세분화(Weakly Semi-Supervised Segmentation, WSSS)에서의 부족한 감독 정보를 개선하기 위해 SuperPixel-Propagated Pseudo-label (SP${}^3$) 학습 방법을 제안합니다. 해당 방법은 스크리블 관련 주석을 슈퍼픽셀에 전파하여 밀집된 주석 결과를 얻는 방식으로, 이는 임상가의 수고를 줄입니다. 추가적으로, 동적 임계값 필터링과 슈퍼픽셀 수준의 불확실성을 적용하여 안정적인 학습을 도모하고 있습니다.

- **Technical Details**: 제안된 SP${}^3$ 방법은 3가지 주요 구성 요소로 이루어져 있습니다: 1) 스크리블 기반의 슈퍼픽셀 확장, 2) 주석 품질 향상을 위한 동적 임계값 필터링, 3) 슈퍼픽셀 수준의 불확실성 가이드. 이 접근 방식은 스크리블에 대한 도움이 되는 주석 정보를 최대한 활용하여 슈퍼픽셀 레벨의 밀집 주석을 얻습니다. 또한, 훈련 과정에서 슈퍼픽셀의 품질 평가를 통해 유용한 정보를 전파합니다.

- **Performance Highlights**: 이 방법은 두 개의 공개 데이터 세트에서 최첨단 성능을 보여줍니다. 특히, 전체 주석 방법과 비교할 때 단 3%의 주석 작업량만 가지고도 약 80%의 Dice 점수를 달성했습니다. 또한, 기존의 8가지 약한 및 반주석 기법들을 모두 초과하며, 매우 안정적이고 효율적인 자동 세분화가 가능하다는 점에서 임상가들에게 큰 도움이 될 것입니다.



### FERT: Real-Time Facial Expression Recognition with Short-Range FMCW Radar (https://arxiv.org/abs/2411.11619)
Comments:
          Accepted at IEEE SENSORS 2024

- **What's New**: 본 연구에서는 1개의 송신(Tx) 안테나와 3개의 수신(Rx) 안테나로 구성된 짧은 거리의 Frequency-Modulated Continuous-Wave (FMCW) 레이더를 활용한 실시간 얼굴 표정 인식의 새로운 접근 방식을 제안합니다. 이 시스템은 Range-Doppler images (RDIs), micro Range-Doppler Images (micro-RDIs), Range Azimuth Images (RAIs), Range Elevation Images (REIs) 등 네 가지 서로 다른 모달리티를 동시에 활용합니다.

- **Technical Details**: 이 연구에 사용된 레이더는 Infineon의 BGT60TR13C 60GHz FMCW 레이더 칩셋으로, Tx 안테나 1개와 Rx 안테나 3개가 포함되어 있습니다. 얼굴 표정은 ResNet 블록을 통한 분류를 위해 4개의 데이터 유형에서 최종적으로 결합된 특징을 사용하여 인식됩니다. FERT 시스템은 각 모달리티에 대해 전용 feature extractor 블록 구조를 포함하고 있으며, 이러한 블록들은 저수준의 특징을 학습합니다.

- **Performance Highlights**: 제안된 모델은 60GHz 짧은 거리 FMCW 레이더를 사용하여 수집된 데이터 세트에서 98.91%의 평균 분류 정확도를 달성하며, 개인 독립적인 방식으로 실시간으로 작동합니다. 이는 저비용 FMCW 레이더가 다양한 응용 분야에서 효과적인 얼굴 표정 인식을 위해 활용될 가능성을 보여줍니다.



### Leveraging Computational Pathology AI for Noninvasive Optical Imaging Analysis Without Retraining (https://arxiv.org/abs/2411.11613)
- **What's New**: 이번 논문에서는 Noninvasive optical imaging을 위해 FoundationShift라는 새로운 접근법을 제안합니다. 이 방법은 H&E와 유사한 이미지로 Optical Coherence Tomography (OCT) 및 Reflectance Confocal Microscopy (RCM) 이미지를 변환하여 AI 모델을 재훈련 없이 적용할 수 있게 합니다. 특히, 기존의 CPath(Computational Pathology) 모델을 Optical Imaging 분석에 활용할 수 있도록 해주며, 이를 통해 진단 정확도를 크게 향상시킵니다.

- **Technical Details**: FoundationShift는 Optical Imaging에서 얻은 이미지를 H&E 같은 이미지로 변환하는 도메인 전이(domain transfer) 접근법을 사용합니다. 이를 통해 적은 양의 데이터로도 잘 구성된 AI 모델을 활용할 수 있게 되며, 인공지능의 재훈련 없이도 기존 모델을 적용할 수 있는 장점이 있습니다. 연구에서는 여러 세그멘테이션 모델과 비교하여 FoundationShift의 정확도가 기존의 최첨단 모델들보다 높다는 것을 입증하였습니다.

- **Performance Highlights**: FoundationShift를 통해 세그멘테이션 정확도가 개선되었으며, 특히 SAM-Med2D와 결합했을 때 그 성능이 극대화되었습니다. 기존 CPath 모델은 2D 이미지에 최적화되어 있었지만, FoundationShift는 3D 이미지를 다루는 데도 유용함을 보여주었습니다. 또한, RCM 이미지에 대해 최초로 매개변수 조정 없이 작동하는 세포 분할 모델을 생성하여 이 분야에서 새로운 전환점을 마련했습니다.



### MSSIDD: A Benchmark for Multi-Sensor Denoising (https://arxiv.org/abs/2411.11562)
Comments:
          15 pages,7 figures

- **What's New**: 이 논문에서는 다양한 센서에서의 노이즈 제거 모델의 전이 가능성을 평가하기 위한 첫 번째 원시 도메인 데이터셋인 Multi-Sensor SIDD (MSSIDD) 데이터셋을 소개합니다. 이 데이터셋은 6개 센서로부터 수집된 60,000개의 원시 이미지 쌍으로 구성되어 있으며, 모바일 단말기의 카메라에서 발생하는 다양한 노이즈 특성을 반영합니다. 이를 통해 센서 차이에 대한 고려 없이 노이즈 제거 모델의 성능을 높일 수 있는 새로운 방법론이 제시됩니다.

- **Technical Details**: MSSIDD 데이터셋은 sRGB 이미지로부터 다양한 카메라 센서의 매개변수를 사용하여 생성된 노이즈가 포함된 이미지 쌍으로 구성됩니다. 저자는 네트워크 훈련 과정에서 센서 불변 특징을 학습하도록 유도하는 센서 일관성 훈련 프레임워크를 제안하며, 이 과정에서 적대적 훈련(adversarial training) 기법을 도입하여 센서 독립적인 특징을 추출합니다. 두 가지 종류의 감독 방식인 intra-image supervision과 inter-image supervision을 통해 이미지 간의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 여러 전통적인 노이즈 제거 기법보다 우수한 센서 전이 능력을 보임을 입증하였습니다. MSSIDD 데이터셋에서 평가한 여러 대표적인 이미지 노이즈 제거 방법을 통해 효과적인 성능이 확인되었습니다. 본 연구는 센서에 대한 일반화 능력을 향상시키는 데 기여하며, 향후 연구 방향을 제시합니다.



### Real-Time Fitness Exercise Classification and Counting from Video Frames (https://arxiv.org/abs/2411.11548)
- **What's New**: 본 논문에서는 Bidirectional Long Short-Term Memory (BiLSTM) 신경망을 이용한 실시간 운동 분류를 위한 새로운 방법을 제안합니다. 기존의 운동 인식 방법들이 합성 데이터셋에 의존하거나 사용자 및 카메라 변화에 민감한 원시 좌표 입력을 사용하여 운동의 시간적 의존성을 완전히 활용하지 못했던 반면, 본 연구에서는 관절 각도와 원시 좌표를 함께 활용하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 모델은 BiLSTM 아키텍처를 기반으로 하여 30프레임의 시퀀스를 훈련하여 운동의 시간적 맥락을 포착할 수 있습니다. InfiniteRep 데이터셋과 Kaggle의 실제 비디오를 결합한 데이터셋에서 네 가지 일반적인 운동(스쿼트, 푸시업, 숄더 프레스, 이두 컬)을 포함하여 훈련 및 검증하였습니다. 모델의 정확도는 99% 이상을 기록하였고, 두 개의 별도 테스트 세트를 통해 일반화 능력을 평가하였습니다.

- **Performance Highlights**: 제안된 BiLSTM 모델은 기존 문헌의 접근 방식과 비교했을 때 가장 우수한 성능을 보였으며, 사용자가 수동으로 운동을 선택할 필요 없이 실시간 운동 분류 및 반복 수 세기를 지원하는 웹 애플리케이션에 통합되었습니다. 데모 및 데이터셋은 GitHub 저장소에서 이용 가능합니다.



### Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignmen (https://arxiv.org/abs/2411.11543)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.13581

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 여러 모달리티에서 정보 처리를 가능하게 하여 멀티모달 학습(multimodal learning)의 발전을 촉진했습니다. 비전-언어 모델(VLMs)은 이미지와 텍스트 기능을 통합하여 시각적 질문 응답, 이미지 캡션 생성 등 다양한 작업에서 눈에 띄는 성과를 거두었습니다. 그러나, 이러한 모델은 시각 모달리티에서 기존 안전 메커니즘을 회피당할 수 있는 취약점을 보여, 안전성을 보장하는 것이 중요한 과제가 되고 있습니다.

- **Technical Details**: 이 연구에서는 PSA-VLM(Progressive Safety Alignment for VLMs)이라는 개념을 제안하여, 비전-언어 모델의 시각 모달리티에 안전 개념을 통합합니다. PSA-VLM은 세 가지 핵심 안전 모듈인 Safety Projector, Safety Tokens 및 Safety Head를 도입하여 안전성과 제어성을 향상시킵니다. 이 방법은 인간이 해석 가능한 개념을 모델 아키텍처에 직접 통합하여 위험한 콘텐츠를 식별하고 개입할 수 있는 구조를 제공합니다.

- **Performance Highlights**: PSA-VLM은 주어진 데이터셋을 기반으로 철저한 평가를 거쳤으며, 일반적인 성능에 미치는 최소한의 영향을 주면서 안전성을 크게 향상시켰습니다. 또한, PSA-VLM은 기존 VLM 안전 기준 대비 최첨단 결과를 달성하였으며, 안전 정렬을 통해 모델 예측 및 사용자 개입을 개선할 수 있는 새로운 패러다임을 제시합니다.



### Reliable Poisoned Sample Detection against Backdoor Attacks Enhanced by Sharpness Aware Minimization (https://arxiv.org/abs/2411.11525)
- **What's New**: 본 논문에서는 딥 뉴럴 네트워크(DNN)에서 발생하는 백도어 공격에 대한 새로운 접근 방식을 제안합니다. 저자들은 Sharpness-Aware Minimization(SAM) 훈련 방법을 사용하여 백도어 효과를 강화함으로써, 데이터 오염을 기반으로 한 공격에서의 포이즌 샘플 검출 성능을 개선할 수 있음을 보여주고자 합니다. 특히, 저자는 SAM을 통해 기존의 포이즌 샘플 검출(PSD) 기법에 결합할 수 있는 혁신적인 방법론을 제시합니다.

- **Technical Details**: 연구팀은 SAM 알고리즘을 통해 백도어 관련 뉴런에서의 백도어 효과를 증가시키고, 이는 훈련된 모델에서 포이즌 샘플을 더욱 효과적으로 검출할 수 있도록 합니다. SAM 훈련은 기존의 PSD 방법과 호환 가능하여, 모델에 통합할 수 있는 이점이 있습니다. 또한, Feature-Scaling 기술을 활용하여 클린 샘플의 특징 변동성을 최소화하고 검출의 안정성을 유지합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트를 통해 진행된 실험에서, 저자들이 제안한 방법론은 다양한 백도어 공격에 대해 평균 34.38%의 True Positive Rate(TPR) 향상을 보여주었습니다. 이는 기존의 PSD 방법들에 비해 상당한 향상으로, 본 연구가 실제 상황에서의 저항력을 크게 증대시킬 수 있음을 시사합니다. 저자들은 이러한 발견이 향후 연구 및 백도어 공격 방어 방법론에 있어 새로운 통찰을 제공할 것으로 기대하고 있습니다.



### Cascaded Diffusion Models for 2D and 3D Microscopy Image Synthesis to Enhance Cell Segmentation (https://arxiv.org/abs/2411.11515)
- **What's New**: 본 논문에서는 자동화된 세포 분할(cell segmentation)을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 희박한 2D 주석(sparse 2D annotations)을 기반으로 2D 및 3D 세포 형태를 합성하는 캐스케이드 확산 모델(cascaded diffusion models)을 활용합니다.

- **Technical Details**: 제안된 방법은 MaskDDPM을 사용하여 2D 마스크를 생성하고, SyncDreamer를 통해 여러 2D 뷰를 예측한 후, NeuS를 사용하여 밀집 부피 마스크(dense volumetric mask)를 생성합니다. 이 과정을 통해 생성된 3D 마스크는 세포의 질감(texture)를 정밀하게 표현하기 위해 사전 훈련된 2D Stable Diffusion 모델에 피드백을 줍니다.

- **Performance Highlights**: Synthetic data와 실제 데이터를 조합하여 훈련한 결과, 여러 데이터셋에서 세포 분할 성능이 최대 9% 향상되었습니다. FID 점수는 생성된 데이터가 실제 데이터와 매우 유사하다는 것을 나타냅니다.



### Learning a Neural Association Network for Self-supervised Multi-Object Tracking (https://arxiv.org/abs/2411.11514)
- **What's New**: 이 논문은 다중 객체 추적(multi-object tracking)에서 데이터 연관(data association)을 자가 지도(self-supervised) 방식으로 학습하는 새로운 프레임워크를 제안합니다. 기존의 완전 감독(fully-supervised) 학습 방법들은 뛰어난 성능을 보이지만, 거기에는 이 세부 정보에 대한 주석이 필요하여 비용과 시간이 많이 들었습니다. 이 연구는 실제 시나리오에서 객체의 움직임이 보통 마르코프 프로세스(Markov process)로 표현될 수 있다는 사실에 동기를 부여받았습니다.

- **Technical Details**: 제안된 방법은 기대 최대화(expectation maximization, EM) 알고리즘을 기반으로 하여, 신경망(neural network)을 사용하여 탐지 결과(detections) 간의 연관성을 예측하도록 훈련합니다. 핵심 요소는 관측 모델이 신경망에 의해 매개변수화된 신경 칼만 필터(neural Kalman filter)입니다. 데이터는 연속적인 프레임으로 입력되어, 각 인접 프레임 간의 데이터 연관을 예측하며, 후속 단계에서 좌표 간의 할당 확률을 결정합니다.

- **Performance Highlights**: 제안된 방법은 MOT17 및 MOT20 데이터셋에서 평가되었으며, 기존의 자가 지도 다중 객체 추적 접근법보다 우수 하거나 동등한 결과를 얻었습니다. 또한 훈련된 모델의 일반화 능력도 입증되었습니다. 전체 프레임워크가 완전히 미분 가능하여 경량의 신경망 모델을 사용해 효율적으로 훈련할 수 있습니다.



### SignEye: Traffic Sign Interpretation from Vehicle First-Person View (https://arxiv.org/abs/2411.11507)
- **What's New**: 이 논문에서는 차량의 1인칭 시야(First-Person View)에서 교통 신호를 해석하는 새로운 작업인 TSI-FPV를 도입합니다. 현재의 연구는 기본적인 신호 인식에 한정되어 있지만, TSI-FPV는 차량 위치를 고려하여 운전 계획 결정에 도움을 줍니다. 이에 더해, TGA(교통 안내 도우미) 시나리오를 개발하여 ADS(자율 주행 시스템)에서의 교통 신호 역할을 재조명합니다.

- **Technical Details**: TSI-FPV는 교통 신호 단위를 설명하는 과정에서 도로 교통 신호 및 표지 기준에 따라 정보를 조직합니다. SignEye라는 단계적 추론 파이프라인을 통해 이 작업을 구현하며, 이를 통해 신호 단위와 도로, 차선 간의 공간적 관계를 분석합니다. Traffic-CN이라는 전용 데이터셋이 구축되어 TSI-FPV 및 TGA가 효과적으로 작동하도록 지원합니다.

- **Performance Highlights**: 실험 결과, SignEye가 Traffic-CN에서 훈련되며 TSI-FPV 및 TGA를 통해 효과적인 결과를 도출할 수 있음을 보여줍니다. TGA는 기존의 자율 주행 기술 외에도 ADS에 추가적인 정보를 제공하여 운전 계획의 결정 과정을 보완할 수 있습니다. 이러한 결과는 교통 신호의 역할에 대한 새로운 인식을 가능하게 합니다.



### LaVin-DiT: Large Vision Diffusion Transformer (https://arxiv.org/abs/2411.11505)
Comments:
          11 pages, 7 figures, 2 tables

- **What's New**: 이번 논문에서는 LaVin-DiT(대형 비전 확산 변환기)를 소개하며, 이는 20개 이상의 컴퓨터 비전 과제를 다루기 위한 확장 가능하고 통합된 기초 모델입니다. LaVin-DiT는 자연어 처리 아키텍처에서 직접 변형되지 않고 공간 관계를 더욱 효과적으로 유지하며 생성 성능을 최적화하는 핵심 혁신들을 도입합니다. 특히, LaVin-DiT는 대규모 비전 데이터셋에서 훈련되어 0.1B에서 3.4B 파라미터로 확장될 수 있습니다.

- **Technical Details**: LaVin-DiT는 높은 차원의 비전 데이터를 처리하기 위해 공간-시간 변동 자동 인코더를 도입하여 이미지를 연속 잠재 공간으로 인코딩합니다. 이 모델은 공동 확산 변환기를 통해 비전 출력을 점진적으로 생성하여 비전 작업에서의 공간 일관성을 유지하면서 처리 효율성을 향상시킵니다. 또한, 입력-목표 쌍을 활용한 상황 학습(in-context learning)을 통해 여러 작업을 통합적으로 학습하고, 테스트 데이터와 작업별 컨텍스트를 활용하여 재조정 없이 다양한 작업에 일반화할 수 있습니다.

- **Performance Highlights**: LaVin-DiT는 여러 비전 벤치마크에서 강력한 기반 LVM을 능가하는 성능을 보여줍니다. 예를 들어, NYU-v2 깊이 추정에서 24% 낮은 AbsRel를 달성했으며, 해상도에 따라 1.7배에서 2.3배 더 빠른 추론 속도를 기록합니다. 실험 결과, 작업 컨텍스트 길이를 증가시키면 다양한 작업에서 성능이 지속적으로 향상되는 경향을 보여 LaVin-DiT의 높이 확장 가능성과 효율성을 입증합니다.



### Look a Group at Once: Multi-Slide Modeling for Survival Prediction (https://arxiv.org/abs/2411.11487)
- **What's New**: 이 논문에서는 GroupMIL이라는 새로운 생존 예측 프레임워크를 소개합니다. 이 프레임워크는 여러 슬라이드를 단일 샘플로 모델링하여 슬라이드 간 예후적 특성을 포착하는 방법을 제안합니다. 또한 GPAMamba 모델을 통해 슬라이드 내 및 간 상호작용을 촉진하여 슬라이드 레벨 그래프 내 미세 환경 특성을 효과적으로 캡처합니다.

- **Technical Details**: GroupMIL은 여러 슬라이드에서 '집단 분석'을 통해 보다 정확하고 신뢰할 수 있는 예후 평가를 목표로 합니다. 두 가지 주요 문제를 해결하기 위해, 페치(patch)들을 시퀀스로 모델링하고, Graph PAMamba(GPAMamba)는 GNN(Graph Neural Networks)에 기반하여 슬라이드의 특징을 식별합니다. 이러한 방식으로, 슬라이드의 고유한 특성을 세심하게 고려하면서, 집단 인사이트를 활용한 예측 평가를 제공합니다.

- **Performance Highlights**: 다섯 개의 데이터 세트에 대한 광범위한 실험을 통해 논문의 제안된 모델이 최신 모델들보다 뛰어난 성능을 보임을 입증하였습니다. 이 모델은 해석 가능성, 환자 분류(patient stratification), 임상 일관성(clinical consistency) 면에서 큰 잠재력을 보여주며, 생존 위험과 확률 평과를 포괄적으로 제공하는 이중 헤드 예측기를 개발하였습니다.



### Exploring Emerging Trends and Research Opportunities in Visual Place Recognition (https://arxiv.org/abs/2411.11481)
Comments:
          2 pages, 1 figure. 40th Anniversary of the IEEE Conference on Robotics and Automation (ICRA@40), Rotterdam, Netherlands, September 23-26, 2024

- **What's New**: 이번 논문은 컴퓨터 비전 및 로봇 공학 분야에서의 이미지 분류와 객체 탐지와 같은 시각 기반 인식의 문제를 다룹니다. 특히 동적 내비게이션 과제를 수행하기 위해 환경에 대한 지식이 예측 가능한 로봇의 핵심 요소임을 강조합니다. 새로운 접근 방식으로 비전-언어 모델(vision-language models)에 주목하고 있습니다.

- **Technical Details**: 비전-언어 모델은 시각적 데이터와 텍스트 데이터를 통합하는 기술로, 로봇의 재위치 확인 및 루프 폐쇄 감지에서 중요하게 작용합니다. 이 과정은 동시 위치 추정 및 매핑(SLAM)에서 기록된 위치를 식별하고 일치시키는 능력을 포함합니다. 연구자들은 최신 자연어 처리(natural language processing) 기법의 성공에서 영감을 받아 새로운 기술 개발에 착수했습니다.

- **Performance Highlights**: 이 연구는 시각 기반 인식의 정확성과 강인성을 향상시키기 위한 신규 기술의 필요성을 강조합니다. 특히, 기존의 비전-언어 모델의 성능을 적용하여 로봇의 내비게이션 능력을 개선할 수 있는 가능성을 보여줍니다. 이로 인해 더욱 정교한 로봇 환경 인식이 이루어질 것으로 기대됩니다.



### SL-YOLO: A Stronger and Lighter Drone Target Detection Mod (https://arxiv.org/abs/2411.11477)
- **What's New**: 이번 논문에서는 드론으로 촬영된 복잡한 장면에서 소형 물체를 효과적으로 탐지하기 위한 혁신적인 모델인 SL-YOLO (Stronger and Lighter YOLO)를 제안합니다. 기존의 YOLO 계열 모델이 대형 목표 탐지에서 뛰어난 성과를 보여주었으나, 소형 목표에 대해서는 성능이 떨어진다는 문제를 해결하기 위해, 계층적 확장 경로 집계 네트워크(HEPAN)를 도입하여 소형 물체 탐지의 한계를 극복하고자 합니다.

- **Technical Details**: HEPAN은 다양한 스케일의 기능을 융합하는 방법으로, 소형 목표의 탐지를 보다 향상시키는 역할을 합니다. 또한, C2fDCB 경량 모듈을 설계하여 모델의 파라미터를 줄이고 계산 복잡성을 감소시키며, SCDown 다운샘플링 모듈을 추가하여 리소스가 제약된 환경에서도 효율적으로 작동할 수 있도록 합니다. 이로 인해 SL-YOLO는 복잡한 배경에서도 소형 목표를 정확하게 인식할 수 있는 능력을 갖추게 됩니다.

- **Performance Highlights**: VisDrone2019 데이터셋에서의 실험 결과, SL-YOLO의 mAP@0.5는 43.0%에서 46.9%로 증가하고, mAP@0.5:0.95는 26.0%에서 28.9%로 향상되었습니다. 또한 모델의 파라미터는 11.1M에서 9.6M으로 줄어들어, FPS는 132에 달하며, 실시간 소형 물체 탐지에 최적의 솔루션을 제공하는 것으로 나타났습니다. 이는 도전적인 환경 속에서도 높은 탐지 정확도를 유지하면서 모델이 소형 목표 탐지의 성능을 크게 향상시켰음을 입증합니다.



### MVLight: Relightable Text-to-3D Generation via Light-conditioned Multi-View Diffusion (https://arxiv.org/abs/2411.11475)
- **What's New**: 이 논문에서 제안하는 MVLight는 지정된 조명 조건에 따라 멀티뷰 일관된 이미지를 생성하는 새로운 라이트 조건 멀티뷰 확산 모델입니다. 이는 기존 모델들이 조명 정보를 명시하지 않는 한계점을 극복하고, 다수의 카메라 뷰에서 고품질 이미지를 생성할 수 있도록 합니다. MVLight는 Score Distillation Sampling (SDS)을 활용하여 기하학적 정밀도와 리라이트 기능을 개선합니다.

- **Technical Details**: MVLight 모델은 멀티뷰 데이터셋과 3D 자체 주의 모듈을 통합하여 3D 모델의 기하학적 특성과 조명 요소를 명확하게 분리할 수 있습니다. 이 모델은 노멀 맵과 알베도 이미지를 생성하여, 다채로운 조명 조건에서 3D 객체를 안정적으로 표현할 수 있는 능력을 갖추고 있습니다. 기존의 텍스트-투-이미지 모델과 달리 MVLight는 조명 환경을 명확히 이용하여 PBR(P Physically Based Rendering) 소재 추정을 진행할 수 있습니다.

- **Performance Highlights**: MVLight는 기존의 리라이트 가능한 텍스트-투-3D 생성 모델에 비해 기하학적 충실도와 리라이트 능력을 모두 개선한 것으로 나타났습니다. 여러 실험과 사용자 연구를 통해 MVLight의 효용성을 검증하였으며, 결과적으로 다양한 조명 조건에서도 일관된 3D 모델 생성을 가능하게 합니다. MVLight는 시각적 일관성을 높이는 데 있어 중요한 기여를 하고 있으며, 향후 3D 콘텐츠 생성의 발전을 도모할 것입니다.



### Generalizable Person Re-identification via Balancing Alignment and Uniformity (https://arxiv.org/abs/2411.11471)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 도메인 일반화 가능한 인물 재식별(DG re-ID)에서 데이터 증강(data augmentation)이 미치는 편향 효과를 최초로 조사합니다. 기존 연구에서는 데이터 증강이 출처 도메인에서 성능을 향상시키지만, 목표 도메인에서는 성능 저하를 초래할 수 있는 것으로 나타났습니다. 제안된 새로운 프레임워크인 BAU(Balancing Alignment and Uniformity)는 정렬(alignment)과 균일성(uniformity) 간의 균형을 유지함으로써 이 문제를 해결합니다.

- **Technical Details**: BAU 프레임워크는 원본 이미지와 증강된 이미지 모두에 정렬 및 균일성 손실(loss)을 적용하여 표현 공간을 규제합니다. 또한, 증강 샘플의 신뢰도를 평가하는 가중치 기법을 통합하여 정렬 손실을 더욱 개선합니다. 도메인 특화 균일성 손실(domain-specific uniformity loss)은 각 출처 도메인 내에서 균일성을 높이는 데 기여하여 도메인 불변 특징(domain-invariant features) 학습을 강화합니다.

- **Performance Highlights**: 전반적인 실험 결과는 BAU 프레임워크가 이전 연구에서 완전히 활용되지 않았던 데이터 증강의 장점을 효과적으로 활용하여 다양한 벤치마크에서 최첨단 성능(state-of-the-art performance)을 달성함을 보여줍니다. BAU는 복잡한 학습 절차 없이도 뛰어난 성과를 내며, DG re-ID 분야에서의 새로운 기준을 세우는데 기여하고 있습니다.



### MGNiceNet: Unified Monocular Geometric Scene Understanding (https://arxiv.org/abs/2411.11466)
Comments:
          Accepted for ACCV 2024

- **What's New**: MGNiceNet는 단일 카메라의 기하학적 장면 이해를 위한 통합 접근 방식을 제안합니다. 이는 panoptic segmentation과 self-supervised depth estimation을 결합하여 자율주행 차량에서의 실시간 응용을 중점으로 두고 개발되었습니다. 특히, panoptic segmentation과 depth estimation 간의 연결된 커널 형식을 사용하여 두 작업 간의 상관관계를 명확히 활용합니다.

- **Technical Details**: 이 새로운 아키텍처는 RT-K-Net을 기반으로 하며, panoptic segmentation과 self-supervised monocular depth estimation을 동시에 수행할 수 있도록 확장되었습니다. MGNiceNet은 낮은 계산 오버헤드로 bin-wise depth representation을 도입하며, panoptic segmentation mask 수준에서 깊이 예측을 수행합니다. 또한, 모션 마스킹 기법은 video panoptic segmentation 주석 없이 panoptic 예측을 기반으로 합니다.

- **Performance Highlights**: MGNiceNet은 Cityscapes 및 KITTI와 같은 자율주행 데이터셋에서 평가되었으며, 기존의 실시간 방법들과 비교하여 최첨단 성능을 보여줍니다. 이 모델은 계산적으로 더 수요가 큰 방법들과의 성능 차이를 줄여주며, 자율주행 환경에서 발생할 수 있는 갑작스러운 변화에 신속하게 반응할 수 있는 실시간 처리 속도를 제공합니다.



### The ADUULM-360 Dataset -- A Multi-Modal Dataset for Depth Estimation in Adverse Weather (https://arxiv.org/abs/2411.11455)
Comments:
          2024 IEEE International Conference on Intelligent Transportation Systems (ITSC)

- **What's New**: 이 논문에서는 ADUULM-360 데이터셋을 소개합니다. 이 데이터셋은 깊이 추정(Depth Estimation)의 새로운 멀티 모달(multi-modal) 데이터셋으로, 다양한 장면을 포함하고 있어 자율주행(autonomous driving) 센서 모달리티를 충족합니다. 특히, 다양한 기상 조건에서 깊이 추정 데이터를 제공하는 첫 번째 데이터셋입니다.

- **Technical Details**: ADUULM-360 데이터셋은 정면 스테레오(stereo) 설정과 360도 전방위의 여섯 개 서라운드 카메라, 두 개의 고해상도 장거리 라이더(lidar) 센서, 다섯 개의 장거리 레이더(radar) 센서로 구성됩니다. 이 데이터셋은 단안(monocular), 스테레오(stereo), 그리고 전체 서라운드(full surround) 훈련 작업을 통해 최첨단 자기 지도(self-supervised) 깊이 추정 방법을 실험했습니다. 다양한 훈련 작업을 통해 깊이 추정의 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, 현재의 최첨단 방법들이 열악한 기상 조건에서 특히 제한된 성능을 보였음을 확인했습니다. 이러한 결과는 향후 연구에 대한 영감을 줄 것으로 기대됩니다. 연구팀은 데이터셋, 개발 키트(development kit), 훈련된 기준선을 제공하고 이를 통해 연구의 접근성을 증가시키고자 합니다.



### Relevance-guided Audio Visual Fusion for Video Saliency Prediction (https://arxiv.org/abs/2411.11454)
- **What's New**: 새로운 연구에서는 오디오와 비디오 간의 불일치를 해결하기 위해 Relevance-guided Audio-Visual feature Fusion 모듈(RAVF)을 제안합니다. 이 모듈은 오디오와 비주얼 요소 간의 의미적 관련성을 기준으로 오디오 특징의 통합 수준을 동적으로 조절합니다. 이 접근 방식은 오디오가 단순한 배경 음악일 때와 같이 두 모달리티 간의 상관관계를 더욱 정교하게 반영합니다.

- **Technical Details**: 논문에서는 비디오 주목 분석을 위해 Video Salient Object Detection (VSOD)과 Video Fixation point Prediction (VFP) 두 가지 주제를 다룹니다. RAVF는 영상 특징과의 의미적 관계를 기반으로 오디오 특징의 통합을 조정하며, Multi-scale feature Synergy(MS) 모듈은 다양한 인코딩 단계에서 시각적 정보를 집계하여 객체를 예측하는 데 도움을 줍니다. Multi-scale Regulator Gate (MRG)는 필수적인 융합 정보를 시각적 특징으로 전달하여 다중 스케일 시각 데이터를 최적화하는 역할을 합니다.

- **Performance Highlights**: AVRSP 네트워크는 여섯 개의 오디오-비주얼 시선 추적 데이터셋에서 실험을 통해 경쟁력 있는 성능을 보여주었습니다. 교차 주의 메커니즘을 활용하여 의미적 관련성을 계산하고 필수적인 오디오 요소를 효율적으로 활용함으로써 더 나은 시각적 예측을 가능하게 했습니다. 이러한 결과는 비디오 콘텐츠의 제공 및 프레젠테이션을 최적화하는 데 중요한 기여를 합니다.



### GLDesigner: Leveraging Multi-Modal LLMs as Designer for Enhanced Aesthetic Text Glyph Layouts (https://arxiv.org/abs/2411.11435)
- **What's New**: 본 논문은 VLM(Visual Language Model) 기반의 새로운 텍스트 로고 레이아웃 생성 프레임워크를 제안하며, 사용자의 제약을 고려하여 콘텐츠 인식 텍스트 로고 레이아웃을 생성합니다. 기존의 GAN 기반 접근법에서 벗어나서, 이 프레임워크는 다중 모달 입력을 통합하여 실제 응용 프로그램에서 보다 유연하고 안정적인 디자인이 가능하게 합니다.

- **Technical Details**: 우리의 모델은 여러 개의 글리프(glyph) 이미지를 동시에 처리하기 위해 두 가지 모델 기술을 도입하고, 컴퓨팅 효율성을 높이기 위해 적응형 평균 풀링(adaptive average pooling)을 적용하여 시각적 글리프 토큰 수를 62배 줄였습니다. 또한, 우리는 두 개의 대규모 텍스트 로고 데이터셋을 구축하여 세밀한 레이아웃 설명과 사용자 제약을 포함하고 있으며, 이 데이터셋은 기존 데이터셋보다 5배 이상 큽니다.

- **Performance Highlights**: 실험 결과, 제안한 모델과 데이터셋은 다양한 벤치마크에서 이전 방법들과 비교하여 우수한 성능을 보였으며, 기하학적 미학과 인간 선호 평가에서 두드러진 성과를 나타냅니다. 우리의 접근법은 긴 텍스트 순서 및 사용자 정의 레이아웃 요구 사항을 효과적으로 처리하며, 실제 응용 프로그램에서도 더 높은 제어성을 제공합니다.



### Towards fast DBSCAN via Spectrum-Preserving Data Compression (https://arxiv.org/abs/2411.11421)
- **What's New**: 이 논문은 DBSCAN(Density-Based Spatial Clustering of Applications with Noise)의 성능을 획기적으로 향상시키기 위해 새로운 방법인 spectral 데이터 압축(spectral data compression)을 도입합니다. 제안된 접근법은 데이터 집합의 크기를 5배 줄이면서도 클러스터링 특성을 유지합니다. 이로 인해 DBSCAN은 정확도 손실 없이 훨씬 빠르게 실행될 수 있습니다.

- **Technical Details**: DBSCAN 알고리즘은 주어진 데이터 포인트의 밀도를 기반으로 클러스터를 식별하는 두 가지 주요 단계(이웃 탐색 및 클러스터 확장)를 통해 작동합니다. 특히, 이 논문에서는 거의 선형 시간 복잡도를 가지는 spectral 데이터 압축 기술을 활용하여 데이터 집합을 압축합니다. 압축된 데이터 세트는 DBSCAN 알고리즘의 입력으로 사용되며, 이를 통해 대규모 데이터 집합에서도 효율적으로 클러스터링을 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, USPS와 같은 실제 데이터 세트를 사용하여 제안된 방법이 데이터 크기를 극적으로 줄이면서 클러스터링 성능을 유지할 수 있음을 입증하였습니다. 이 방법은 데이터의 압축 비율을 사용자 요구에 맞게 조정할 수 있는 추가적인 이점을 제공합니다. DBSCAN의 기존 한계를 극복함으로써, 이 연구는 대규모 데이터 분석 작업에서 DBSCAN의 적용 가능성을 확장합니다.



### IKEA Manuals at Work: 4D Grounding of Assembly Instructions on Internet Videos (https://arxiv.org/abs/2411.11409)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 본 논문에서는 IKEA 가구 조립에 관한 새로운 멀티모달 데이터셋인 IKEA Video Manuals를 소개합니다. 이 데이터셋은 3D 모델, 지침 매뉴얼, 조립 동영상 및 이들 간의 밀접한 시공간 정렬(Spatio-temporal alignments)을 포함하고 있습니다.

- **Technical Details**: IKEA Video Manuals 데이터셋은 98개의 조립 동영상에서 34,441 개의 주석이 포함된 비디오 프레임으로 구성되며, 여기에는 2D-3D 파트 대응관계, 시간적 단계 정렬 및 파트 분할(Part segmentation)이 포함됩니다. 각 조립 동영상은 6-DoF(6 degrees of freedom) 포즈의 3D 파트를 사용하여 시각적 정렬을 수행합니다.

- **Performance Highlights**: 조립 계획 생성(assembly plan generation), 파트 조건부 분할(part-conditioned segmentation), 파트 조건부 포즈 추정(part-conditioned pose estimation), 비디오 객체 분할(video object segmentation), 지침 비디오 매뉴얼을 기반으로 한 가구 조립 등 다섯 가지 응용프로그램을 통해 이 데이터셋의 유용성을 입증하였습니다. 또한 실험 결과를 통해 비디오에서 조립 지침을 정립하는 데 있어 발생하는 다양한 문제를 강조하였습니다.



### Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection (https://arxiv.org/abs/2411.11396)
- **What's New**: 이번 논문에서는 Incremental Face Forgery Detection (IFFD)을 위한 새로운 접근방법을 제안합니다. 기존의 IFFD 모델은 새로운 변조를 통합할 때 재앙적 망각(catasrophic forgetting)에 취약했습니다. 이를 해결하기 위해, 이전과 새로운 작업의 잠재적 특징 분포(latent feature distributions)를 '브릭(brick)'처럼 쌓아 올려 정렬된 특징 격리(aligned feature isolation)를 수행합니다. 이러한 방법은 고유한 변조 정보를 보존하고 새로운 지식을 효과적으로 축적하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 Sparse Uniform Replay (SUR) 및 잠재 공간 증분 탐지기(Latent-space Incremental Detector, LID)를 포함합니다. SUR는 이전의 전역 분포(global distributions)를 나타낼 수 있는 대표적인 하위 집합(replay subsets)을 선택하여 균일 분포를 유지합니다. 그런 다음, LID는 이 하위 집합을 사용하여 특징 격리를 달성하고, 결정 정렬을 통해 새로운 작업의 결정 경계가 이전 작업과 정렬되도록 합니다.

- **Performance Highlights**: 실험 결과는 제안된 SUR-LID 방법의 우수성을 입증하였습니다. 기존의 증분 얼굴 변조 탐지 방법들과 비교했을 때, 제안된 접근법은 다양한 변조 정보를 효과적으로 활용하여 최종적인 이진 얼굴 변조 탐지 성능을 향상시켰습니다. 또한, 새로운 종합 벤치마크를 구축하여 IFFD의 성능 평가를 위한 실질적인 어플리케이션 시나리오를 반영했습니다.



### LeC$^2$O-NeRF: Learning Continuous and Compact Large-Scale Occupancy for Urban Scenes (https://arxiv.org/abs/2411.11374)
Comments:
          13 pages

- **What's New**: 본 논문에서는 대규모 NeRF(Neural Radiance Fields)에서 점유 모델링의 효율성을 높이기 위해 연속적이고 압축된 점유 네트워크를 학습하자는 새로운 접근 방식을 제안합니다. 이 네트워크는 3D 포인트를 점유된 지점과 비어 있는 지점으로 분류할 수 있으며, Self-supervised 방식으로 훈련됩니다. 이 연구는 점유 그리드의 한계점을 극복하고 대규모 복잡한 도시 장면에서도 높은 정확도를 유지하면서 학습 속도를 향상시키는 방법을 제시합니다.

- **Technical Details**: 제안된 LeC2O-NeRF는 세 가지 설계를 통해 점유 네트워크를 훈련합니다. 첫째, 점유 비율을 조절하는 불균형 점유 손실(imbalanced occupancy loss)을 제안하여 운동 밀도의 비율을 효과적으로 조절합니다. 둘째, 큰 장면 네트워크와 작은 비어 있는 공간 네트워크로 구성된 불균형 아키텍처를 통해 점유와 비어 있는 영역을 별도로 인코딩합니다. 마지막으로, 점유 네트워크를 안내하는 명시적 밀도 손실(explicit density loss)을 설계했습니다.

- **Performance Highlights**: 실험 결과, 제안된 점유 네트워크는 점유 그리드에 비해 더욱 빠르고 정확하며 매끄러운 점유를 학습할 수 있음을 보여주었습니다. 이 방법은 대규모 벤치마크에서 비어 있는 공간 건너뛰기를 위한 지침으로 작용하여, 점유 그리드를 사용할 경우보다 항상 높은 정확도를 달성합니다. 최종적으로, 우리의 방법은 최신 NeRF 방법을 속도와 정확도를 모두 높이면서 가속할 수 있는 가능성을 제시합니다.



### TL-CLIP: A Power-specific Multimodal Pre-trained Visual Foundation Model for Transmission Line Defect Recognition (https://arxiv.org/abs/2411.11370)
- **What's New**: 본 논문에서는 전송선 결함 인식을 보다 효과적으로 지원하기 위해 두 단계의 전송선 지향의 대조 언어-이미지 사전 훈련(TL-CLIP) 프레임워크를 제안합니다. 기존의 사전 훈련 데이터셋의 도메인 지식 부족 문제를 해결하기 위해 전력 관련 다중 모달 알고리즘과 두 가지 전력 특정 사전 훈련 작업을 도입하여 검사 데이터에 포함된 전력 관련 의미 지식을 더 잘 모델링하고자 합니다.

- **Technical Details**: 제안된 TL-CLIP 프레임워크는 사전 훈련과 전이의 두 단계로 구성되어 있습니다. 첫 번째 단계인 전력 특정 VLP에서는 컴포넌트 타입 매칭(CTM) 및 결함-정상 비교(DNC)와 같은 두 가지 새로운 사전 훈련 작업을 도입하여 검사 이미지 내에 숨겨진 클래스 간 관계를 모델링합니다. 두 번째 단계에서는 사전 훈련 목표로 미세 조정하는 전략인 FTP를 통해 미세 조정 단계에서 과적합 문제를 완화하고자 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 전송선 결함 인식 정확도를 크게 향상시켜 전통적인 사전 훈련 모델에 비해 분류 및 검출 작업 모두에서 뚜렷한 장점을 보였습니다. 이 방법은 원래 모델 구조를 변경하지 않고 파라미터 수와 계산 복잡성을 증가시키지 않으면서 효과적으로 전력 도메인에서 성능을 개선합니다.



### GPS-Gaussian+: Generalizable Pixel-wise 3D Gaussian Splatting for Real-Time Human-Scene Rendering from Sparse Views (https://arxiv.org/abs/2411.11363)
Comments:
          Journal extension of CVPR 2024,Project page:this https URL

- **What's New**: 본 연구는 고해상도의 이미지 렌더링을 위한 일반화 가능한 Gaussian Splatting 접근 방식을 제안합니다. 기존의 실시간 렌더링 기술들이 개별적으로 최적화 되어야 하는 단점을 극복하고, 깊이 추정과 렌더링 손실을 통해 훈련된 모델을 통해 즉각적인 신규 뷰 합성을 가능하게 합니다. 이 방법은 고품질의 비전 통합 솔루션을 제공하며, 특히 복잡한 장면에서도 높은 해상도 렌더링을 달성할 수 있습니다.

- **Technical Details**: 제안된 방법에서는 소스 뷰에 정의된 2D Gaussian 파라미터 맵을 사용하여 3D Gaussian을 표현합니다. 양안 스테레오 매칭을 통해 깊이를 결정하고, 이 정보를 바탕으로 2D 매개변수를 3D 공간으로 확장하여 모델을 학습합니다. 또한, 정규화 항과 epipolar attention 메커니즘을 도입해 두 소스 뷰 간의 기하학적 일관성을 유지합니다.

- **Performance Highlights**: 조사 결과, 제안하는 방법이 기존의 최첨단 기술들에 비해 우수한 성능을 보이며, 약 25 FPS의 속도로 고해상도의 자유 시점 비디오를 생성합니다. 특히 새로운 캐릭터가 배경 없이 또는 배경을 포함하여 즉시 렌더링되는 능력을 갖추고 있으며, 추가적인 미세 조정이나 최적화 없이도 고충실도의 결과를 제공합니다.



### MAIRA-Seg: Enhancing Radiology Report Generation with Segmentation-Aware Multimodal Large Language Models (https://arxiv.org/abs/2411.11362)
Comments:
          Accepted as Proceedings Paper at ML4H 2024

- **What's New**: 이 논문에서는 MAIRA-Seg라는 새로운 프레임워크를 소개합니다. 이 모델은 영상 세그멘테이션 마스크를 활용하여 흉부 X-레이(CXR)와 함께 방사선 보고서를 작성하는 데 초점을 맞추고 있습니다. 기존의 멀티모달 대형 언어 모델(MLLM)과의 차별점은 픽셀 수준의 세부 정보를 통합하여 보다 정교한 이미지 해석을 가능하게 한다는 점입니다.

- **Technical Details**: MAIRA-Seg는 CXR 이미지와 연결된 의미론적 세그멘테이션 마스크의 특성을 활용합니다. 전문 세그멘테이션 모델을 훈련시켜 방사선 관련 구조의 마스크를 생성하고, 이를 토대로 MLLM의 입력으로 사용하여 훈련합니다. Osprey 아키텍처를 기반으로 한 세그멘테이션 토큰 추출기를 사용하여 이미지 토큰 및 텍스트 토큰과 세그멘테이션 토큰을 교차하여 결합합니다.

- **Performance Highlights**: MIMIC-CXR 데이터셋을 통한 실험 결과, MAIRA-Seg는 비세그멘테이션 기준선 모델에 비해 성능이 우수한 것으로 나타났습니다. 세그멘테이션 마스크의 사용이 MLLM의 세밀한 추론 능력을 증대시켜 임상 결과에 긍정적인 영향을 미칠 가능성을 보여줍니다. 또한, MAIRA-Seg는 기존 모델들과 비교했을 때 정량적, 정성적으로 우수한 개선을 입증했습니다.



### Scalable Autoregressive Monocular Depth Estimation (https://arxiv.org/abs/2411.11361)
- **What's New**: 이 논문은 새로운 autoregressive 모델인 Depth Autoregressive (DAR)를 제안합니다. DAR는 단일 RGB 이미지로부터 깊이 맵을 예측하는 Monocular Depth Estimation (MDE) 작업을 효과적으로 수행합니다. 이 모델은 저해상도부터 고해상도까지의 깊이 맵을 토큰으로 처리하고, 패치 기준의 카주얼 마스크(causal mask)를 사용하여 예측을 진행합니다.

- **Technical Details**: DAR는 두 가지 주요 디자인을 바탕으로 합니다. 첫 번째는 다양한 해상도의 깊이 맵을 생성하는 과정에서의 순서성(depth map resolution)이며, 두 번째는 깊이 값이 연속적인 공간에 존재하는 특성을 이용하여 점진적으로 더 미세한 간격으로 깊이를 이산화하는 것입니다. 이를 통해 각 단계에서의 예측이 이전 예측 기반으로 이루어지도록 구성하여 깊이 추정 문제를 해결합니다.

- **Performance Highlights**: DAR는 KITTI 및 NYU Depth v2 데이터셋에서 새로운 최첨단 성능(SOTA)을 기록하며, RMSE에서 1.799의 결과로 기존 SOTA인 Depth Anything보다 5% 개선되었습니다. 2.0B 파라미터까지 확장할 수 있는 스케일러블(scalable)한 접근 방식으로, DAR는 처리된 다양한 데이터에 대해 제로샷(generalization) 능력을 입증했습니다.



### CCExpert: Advancing MLLM Capability in Remote Sensing Change Captioning with Difference-Aware Integration and a Foundational Datas (https://arxiv.org/abs/2411.11360)
- **What's New**: 논문에서는 Remote Sensing Image Change Captioning (RSICC) 분야의 새로운 모델인 CCExpert를 제안합니다. CCExpert는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 기반으로 하며, 특히 이미지 쌍 간의 미세한 변화를 포착하는 Difference-aware Integration Module을 도입합니다. 이 모델은 200,000장의 이미지 쌍과 120만 개의 캡션을 포함한 CC-Foundation 데이터셋으로 학습되었습니다.

- **Technical Details**: CCExpert는 이미지의 차이를 인지할 수 있는 통합 모듈을 설계하여 두 개의 시점에서의 이미지 간의 다중 스케일 차이를 포착하는 것을 목표로 합니다. 딥러닝(DL) 모델의 학습은 3단계의 진행적(training process) 방식으로 이루어지며, 이는 MLLM과 차별화된 통합 모듈의 깊은 결합을 보장합니다. 이 방법론은 최적의 성능을 달성할 수 있도록 설계되어 있습니다.

- **Performance Highlights**: CCExpert는 LEVIR-CC 벤치마크에서 S^*_m=81.80이라는 두드러진 성과를 기록하여 기존의 최첨단 방법들을 크게 초월했습니다. 실험 결과는 CCExpert의 커다란 성능 향상을 보여주며, RSICC 분야에서 MLLM의 잠재력을 충분히 활용하는 방법을 제시합니다. 이 모델은 다양한 시각적 및 언어적 맥락에서 매우 유용하게 활용될 수 있습니다.



### Text-guided Zero-Shot Object Localization (https://arxiv.org/abs/2411.11357)
- **What's New**: 본 논문에서는 데이터 labeling의 부족 문제를 해결하기 위해 새로운 제로샷 객체 위치 파악 프레임워크(Zero-Shot Object Localization, ZSOL)를 제안합니다. 이 프레임워크는 CLIP 모듈(Contrastive Language Image Pre-training)을 활용하여 이미지와 텍스트 정보를 효과적으로 통합합니다. 또한, TSSM 모듈(Text Self-Similarity Matching)을 도입하여 텍스트 특징의 표현력을 향상시킴으로써 객체의 정확한 위치 파악을 가능하게 합니다.

- **Technical Details**: ZSOL 프레임워크는 제로샷 학습(zero-shot learning)의 장점을 극대화하며, 주어진 이미지에서 ‘프롬프트(word prompt)’를 통해 객체를 식별하고 위치를 결정할 수 있습니다. CLIP 모듈은 이미지와 텍스트를 공유된 벡터 공간으로 매핑하여 비지도 학습 역할을 합니다. 또한, TSSM 모듈은 텍스트 특징의 표현력을 높여 객체 위치 파악의 정확도를 개선합니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행된 실험 결과, 제안된 ZSOL 방법은 기존의 Few-Shot Object Localization 접근 방식보다 높은 성능을 보여줍니다. 이러한 성능 개선은 데이터 레이블링의 필요성을 줄이면서도 효과적인 객체 위치 파악을 가능하게 하며, 향후 연구를 위한 새로운 기준을 설정하는 데 기여할 것으로 보입니다.



### Superpixel-informed Implicit Neural Representation for Multi-Dimensional Data (https://arxiv.org/abs/2411.11356)
Comments:
          Accepted at ECCV 2024, 18 pages, 7 figures

- **What's New**: 이번 논문에서는 기존의 implicit neural representations (INRs)의 한계를 극복하기 위해 Superpixel-informed INR (S-INR)이라는 새로운 모델을 제안합니다. S-INR은 일반화된 슈퍼픽셀을 기본 단위로 사용하여 데이터의 내재적 의미 정보를 효과적으로 활용합니다. 이러한 접근은 이미지와 기상 데이터 같은 다차원 데이터의 복구를 향상시키는 가능성을 제공합니다. 특히, S-INR은 전통적인 INR 방법들과 비교하여 우수한 성능을 보이는 것으로 입증되었습니다.

- **Technical Details**: S-INR은 exclusive attention-based MLP와 공유 사전 행렬(shared dictionary matrix)이라는 두 가지 주요 모듈로 구성되어 있습니다. 이 모듈들은 각 일반화된 슈퍼픽셀의 개별성을 존중하며, 이들 간의 공통성을 포착합니다. 논문에서 제안된 일반화된 슈퍼픽셀은 단순히 이미지 데이터에 국한되지 않으며, 실제 응용 프로그램에서 발생하는 포인 데이터에도 적용될 수 있습니다. 이를 통해 모델은 데이터의 내재적 의미 정보를 보다 효과적으로 활용할 수 있습니다.

- **Performance Highlights**: S-INR은 이미지 복원, 이미지 완성, 이미지 노이즈 제거, 기상 데이터 완성 및 3D 표면 완성과 같은 다양한 실험에서 그 효과성을 입증했습니다. 이 모델은 기존 최첨단 INR 방법들과 비교하여 광범위한 적용 가능성과 우수성을 보여줍니다. 특히, 일반화된 슈퍼픽셀을 사용함으로써 데이터의 의미 정보를 효과적으로 활용하고, 다양한 응용 분야에서 성능 향상을 달성했습니다.



### A comprehensive survey of oracle character recognition: challenges, benchmarks, and beyond (https://arxiv.org/abs/2411.11354)
- **What's New**: 이 논문은 오라클 문자 인식(OrCR) 분야에 대한 체계적이고 구조적인 검토를 제공하며, 이전의 문헌 부족 문제를 해결하고자 합니다. 오라클 문자 인식은 고대 중국의 역사와 문화를 이해하는 데 중요한 역할을 하며, 해당 분야의 최신 연구 동향, 도전 과제, 데이터셋, 방법론, 관련 작업, 미래 연구 방향을 조망합니다.

- **Technical Details**: OrCR의 주요 도전 과제는 세 가지로, 문자의 변동성(writing variability), 데이터 부족(data scarcity), 저해상도 이미지(low image quality)입니다. 오라클 문자 작성은 표준화된 규범이 없었기 때문에 모든 문자가 다양한 변형을 가지고 있으며, 데이터 샘플로서의 오라클 문자의 수가 매우 제한적입니다. 또한 장기간 매장된 오라클 뼈의 이미지는 품질 저하를 겪어 오라클 문자 인식의 어려움을 증가시킵니다.

- **Performance Highlights**: 오라클 문자 인식을 자동화하기 위한 현대의 연구 접근법은 전통적인 패턴 인식에서 심층 학습(deep learning) 기법으로 발전해왔습니다. 그러나, OrCR에 관한 체계적 문헌이 부족한 현 상황을 고려할 때, 이 논문은 연구자들에게 OrCR 분야의 진행 상황을 이해하는 데 기여할 것입니다. 따라서, 이 논문은 신규 연구자와 경험이 있는 연구자 모두에게 유용한 기초 자료가 될 것입니다.



### Visual-Semantic Graph Matching Net for Zero-Shot Learning (https://arxiv.org/abs/2411.11351)
Comments:
          15 pages, 6 figures

- **What's New**: 본 연구에서는 기존의 제로샷 러닝(Zero-shot Learning, ZSL) 프레임워크의 한계를 극복하기 위해 Visual-Semantic Graph Matching Net(VSGMN)을 제안합니다. VSGMN은 시각적 표현과 의미적 원형 간의 관계를 강화하여 시각적-의미적 임베딩 과정에서 클래스 간의 중요한 관계를 반영합니다. 이를 통해 보지 못한 클래스 간의 지식을 효과적으로 전이할 수 있습니다.

- **Technical Details**: VSGMN은 두 단계의 시각-의미 정렬을 달성하기 위해 그래프 빌드 네트워크(Graph Build Network, GBN)와 그래프 매칭 네트워크(Graph Matching Network, GMN)로 구성되어 있습니다. GBN은 임베딩 기반 접근 방식을 사용하여 의미 공간에서 시각 및 의미 그래프를 생성하고 초기 정렬을 수행합니다. 그 후 GMN은 그래프 노드 간의 관계를 통합하고 정렬하여 보다 견고한 임베딩 공간을 학습하도록 돕습니다.

- **Performance Highlights**: 세 개의 벤치마크 데이터셋에서 VSGMN의 성능이 기존 모델보다 유의미하게 향상된 결과를 보여주었습니다. VSGMN은 AWA2, CUB, SUN 데이터셋에서 각각 2.2%/0.9%, 2.4%/1.3% 및 1.4%/0.3%의 정확도(accuracy) 및 H 스코어(H score) 향상을 기록하였습니다. 이러한 결과는 VSGMN의 효과적인 구조가 ZSL에서 시각적-의미적 매핑 문제를 해결하는 데 기여함을 보여줍니다.



### Teaching Video Diffusion Model with Latent Physical Phenomenon Knowledg (https://arxiv.org/abs/2411.11343)
Comments:
          7 figures, 14 pages

- **What's New**: 이 논문에서는 비디오 생성 모델의 새로운 접근 방식을 제시하여 잠재적인 물리 현상 지식을 학습시키고, 이를 통해 물리적으로 정보가 포함된 비디오를 생성할 수 있는 방법을 소개합니다. 기존의 비디오 디퓨전 모델은 물리 법칙을 제대로 이해하지 못하는 한계가 있었는데, 본 연구는 그러한 한계를 극복하고자 Masked Autoencoder (MAE)를 활용하여 물리 현상을 재구성하는 방법을 채택하였습니다.

- **Technical Details**: Masked Autoencoder (MAE)를 통해 물리 현상을 복원함으로써 생성된 임베딩(output embeddings)은 잠재적 물리 현상 지식을 포괄하게 됩니다. 이 임베딩을 활용하여 CLIP 비전 및 언어 인코더 간의 정렬된 공간적 관계를 기반으로 의사 언어 프롬프트 피처(pseudo-language prompt features)를 생성합니다. 특히, 디퓨전 모델이 일반적으로 CLIP의 언어 인코더를 사용하므로, 우리의 접근 방식은 자연어로 물리 현상을 설명하는 어려움을 해결하고자 합니다.

- **Performance Highlights**: 수치 시뮬레이션과 실제 물리 현상을 통해 광범위하게 검증한 결과, 제안된 방법이 다양한 시나리오에서 뛰어난 성능을 발휘함을 보여주었습니다. 물리 지식에 정보가 포함된 비디오 생성이 가능함을 입증하며, 이는 기초적인 물리 법칙을 따르는 비디오 생성의 새로운 방향성을 제시합니다. 결론적으로 본 연구는 비디오 디퓨전 모델에 물리적 현상 지식을 도입하였고, 이로 인해 물리적으로 정보가 포함된 비디오를 효과적으로 생성할 수 있는 방법을 제시하였습니다.



### Video-to-Task Learning via Motion-Guided Attention for Few-Shot Action Recognition (https://arxiv.org/abs/2411.11335)
- **What's New**: 이번 논문에서는 Dual Motion-Guided Attention Learning 방법(DMGAL)을 제안하여 몇 개의 샘플만으로 행동 인식을 수행할 수 있는 새로운 기법을 소개합니다. DMGAL은 비디오 수준에서 태스크 수준으로의 시차적 관계를 학습하도록 설계되었습니다. 특히, 기존의 방법들이 간과했던 서로 다른 비디오 간의 시차적 관계를 효과적으로 활용할 수 있는 방법론에 중점을 두었습니다. 이는 다양한 비디오에서 행동의 시공간 관계를 파악하는 데 도움을 줍니다.

- **Technical Details**: 제안된 DMGAL 방법은 Self Motion-Guided Attention(S-MGA)과 Cross Motion-Guided Attention(C-MGA)의 두 가지 핵심 모듈로 구성되어 있습니다. S-MGA는 단일 비디오 내에서의 시차적 관계를 모델링하고, C-MGA는 서로 다른 비디오 간의 시차적 관계를 파악하여 태스크 수준의 관계를 학습합니다. 이 두 모듈은 낮은 파라미터 요구 사항과 높은 적응성을 갖춰 파라미터 효율성이 높고, 각기 다른 튜닝 패러다임에 통합될 수 있습니다.

- **Performance Highlights**: DMGAL-FT와 DMGAL-Adapter라는 두 가지 모델을 통해 공인된 다섯 개의 벤치마크에서 실험을 진행하였으며, 제안된 방법이 여러 최첨단 기술들보다 우수한 성능을 발휘함을 확인했습니다. 실험 결과는 DMGAL이 분류 성능을 크게 향상시키는 데 기여함을 보여 주었으며, 이로 인해 비디오 특화 수준에서 태스크 특화 수준으로의 관계를 완전히 통합한 클래스 프로토타입을 구축할 수 있게 되었습니다.



### Color-Oriented Redundancy Reduction in Dataset Distillation (https://arxiv.org/abs/2411.11329)
Comments:
          38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 Dataset Distillation(DD)의 효율성을 향상시키기 위해 AutoPalette라는 새로운 프레임워크를 제안합니다. AutoPalette는 이미지와 전체 데이터셋 수준에서 색상 중복성을 최소화하여 훈련 효율성을 높입니다. 특별히 설계된 palette network를 사용하여 이미지의 각 픽셀에 색상을 동적으로 할당하고, 정보 이득에 기초하여 이미지 간 중복성을 감소시키는 방법을 제시합니다.

- **Technical Details**: AutoPalette는 훈련 과정에서 필수적인 모델 기능을 보존하면서 색상 중복성을 줄이는 플러그 앤 플레이(pug-and-play) palette network를 포함하고 있습니다. 이 네트워크는 입력 이미지의 픽셀 수준 색상 정보를 집계하여 8비트 색상 이미지를 더 적은 색상(예: 4비트)의 표현으로 변환합니다. 또한, 색상 손실(maximum color loss)과 팔레트 균형 손실(palette balance loss)을 통해 색상 할당을 최적화합니다.

- **Performance Highlights**: 광범위한 실험 결과는 제안된 색상 인식 DD가 기존 DD 방법보다 우수한 성능을 보임을 보여줍니다. 우리의 방법은 CIFAR10 및 CIFAR100 데이터셋에서 각각 1.7% 및 4.2%의 성능 향상을 달성하였습니다. 이는 동일한 저장 용량을 사용하며, 총 색상 수를 줄임으로써 저장 효율성을 확보했다는 점에서 의미가 큽니다.



### TP-UNet: Temporal Prompt Guided UNet for Medical Image Segmentation (https://arxiv.org/abs/2411.11305)
- **What's New**: 이 논문에서는 기존 UNet 기반의 의료 이미지 세분화 방법들이 시간적 정보(temporal information)를 고려하지 못하는 문제를 해결하기 위해 TP-UNet이라는 새로운 프레임워크를 제안합니다. TP-UNet는 시간적 단서(temporal prompts)를 사용하여 세분화 모델이 의료 이미지를 학습하는 과정에서 시간적 정보를 효과적으로 통합할 수 있도록 도움을 줍니다. 또한, 이 프레임워크는 비지도 대비 학습(unsupervised contrastive learning) 기반의 의미 정렬(semantic alignment) 메커니즘을 활용하여 이미지와 텍스트 간의 의미적 격차를 줄입니다.

- **Technical Details**: TP-UNet의 핵심 구성 요소는 고차원의 임베딩(embedding) 과정과 교차 주의 메커니즘(cross-attention mechanism)입니다. 이 프레임워크는 두 개의 단계인 의미 정렬과 모달 융합(modal fusion)으로 구성되어 있으며, 이를 통해 시간적 단서와 이미지 특징 간의 개념적 일치를 도모합니다. 또한, 모델의 인코더는 텍스트 및 이미지의 특징 맵을 서로 통합하여 최종적으로 UNet의 디코더(decoder) 입력으로 사용되는 통합된 표현을 생성합니다.

- **Performance Highlights**: 리츠(LITS) 2017 데이터셋 및 UW-Madison 데이터셋을 포함한 두 개의 의료 이미지 세분화 데이터셋에서 TP-UNet의 성능을 광범위하게 평가한 결과, 새로운 최첨단(SOTA) 성능을 달성한 것으로 나타났습니다. 이는 TP-UNet이 시간적 정보를 활용하여 기존의 의료 이미지 세분화 기술을 개선할 수 있음을 의미합니다. 이러한 연구 결과는 향후 의료 이미지 세분화 기술 발전에 중요한 기여를 할 것으로 기대됩니다.



### Performance Evaluation of Geospatial Images based on Zarr and Tiff (https://arxiv.org/abs/2411.11291)
- **What's New**: 이번 연구는 Zarr 및 TIFF 형식의 지리공간 이미지 처리를 비교하여 각 포맷의 성능 차이를 평가했습니다. 전통적인 TIFF 포맷은 널리 사용되지만, 대규모 데이터 세트에서 성능 한계가 존재하는 반면, Zarr 포맷은 클라우드 시스템을 위한 효율적인 저장 및 접근 방법을 제공합니다. 이 연구는 지리공간 데이터의 특정 요구 사항에 따라 적합한 포맷을 선택하는 데 도움을 줍니다.

- **Technical Details**: 본 연구에서는 Geospatial 이미지를 처리하기 위해 PyTorch 모델과 다양한 Python 라이브러리를 사용하여 Zarr과 TIFF 포맷의 성능을 비교했습니다. 분석에는 저장 효율성, 데이터 접근 속도 및 처리 시간 등이 포함되었으며, Zarr은 청크(chunk) 저장 방식과 압축 기술을 채택하여 대규모 데이터셋을 보다 효율적으로 지원합니다. 따라서, Zarr은 대량의 지리공간 데이터 처리에 적합한 솔루션으로 부각되고 있습니다.

- **Performance Highlights**: Zarr과 TIFF의 성능 평가는 대규모 데이터 세트에서 Zarr의 우월성을 강조합니다. 실험 결과 Zarr은 더 빠른 접근 속도를 보여주었으며, TIFF는 작은 데이터 세트에 대해서는 여전히 유용하지만, 대규모 데이터에서는 성능 저하가 발생했습니다. 이를 통해 연구자들이 지리공간 데이터 처리에 있어서 어떤 포맷을 선택해야 하는지에 대한 중요한 통찰을 제공합니다.



### Neuron: Learning Context-Aware Evolving Representations for Zero-Shot Skeleton Action Recognition (https://arxiv.org/abs/2411.11288)
Comments:
          13 pages, 6 figures

- **What's New**: 이 논문에서는 제로샷 스켈레톤 동작 인식을 위한 새로운 동적 진화 이중 스켈레톤-시맨틱 시너지를 탐구하는 프레임워크인 Neuron을 제안합니다. 기존 방법들은 불완전한 매핑과 명시적 표현으로 인해 복잡한 교차 모드 전이 가능성을 포착하는 데 어려움이 있었습니다. Neuron은 컨텍스트 인식 사이드 정보를 활용하여 미세한 교차 모드 대응을 탐색합니다.

- **Technical Details**: 이 프레임워크는 공간-시간 미세 프로토타입을 구축하고 이를 통해 스켈레톤-시맨틱 관계를 단계적으로 포착합니다. 공간 압축(spatial compression) 및 시간 기억(temporal memory) 메커니즘을 도입하여 공간 관련 구조 표현 및 시간 의존 패턴을 흡수할 수 있도록 설계되어 있습니다. 이러한 과정은 뉴런의 성장 과정을 모방하여 미세한 교차 모드 상호작용을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 NTU RGB+D, NTU RGB+D 120, PKU-MMD 데이터셋에서 제로샷 학습(ZSL) 및 일반화 제로샷 학습(GZSL) 환경 모두에서 최첨단 성능을 달성했습니다. 이를 통해 이전의 방법들에 비해 더욱 효과적으로 보이지 않는 동작 범주를 인식할 수 있습니다.



### Reducing Label Dependency for Underwater Scene Understanding: A Survey of Datasets, Techniques and Applications (https://arxiv.org/abs/2411.11287)
Comments:
          70 pages, 20 figures

- **What's New**: 이 논문은 수중 이미지 분석에 있어 전문가 데이터 의존성을 줄이기 위한 약한 감독 학습(weakly supervised learning) 접근 방법에 대한 포괄적 검토를 제공합니다. 특히, 해양 생태계의 모니터링을 위한 새로운 자동화 기술을 중심으로 이전 연구들을 정리합니다. 이 연구는 데이터 부족 문제를 해결하기 위한 새로운 기회를 제시하며, 해양 생물의 식별을 돕기 위해 딥러닝(deep learning)과 컴퓨터 비전(computer vision) 기술의 융합을 강조합니다.

- **Technical Details**: 이 연구에서는 수중 이미지를 활용한 자동 분석의 필요성을 설명하며, 주로 RGB 이미지를 다룹니다. 섬세한 정밀도를 요구하는 해양 생물 종류 식별에 있어 기존의 밀집된 마스크(label mask) 생성 방식을 대체할 수 있는 약한 감독 및 자가 감독(self-supervised) 학습 방법론을 제안합니다. 또한, 데이터 세트의 구조와 이를 위한 다양한 수집 기법, 즉 사진 정량화(photo-quadrat) 이미지를 어떻게 활용하는지에 대한 세부사항도 포함되어 있습니다.

- **Performance Highlights**: 이 논문은 자동화된 수중 조사 방식의 효율성을 높이기 위한 다양한 접근 방식을 논의하며, 실시간으로 수집된 데이터를 분석하여 즉각적인 피드백을 제공할 수 있는 시스템의 필요성을 강조합니다. 전통적인 전문가 주도 주석 방식과는 달리, 본 연구는 데이터 주석의 속도와 누적적인 비용 문제를 해결할 해법을 탐색하고 있으며, 이로 인해 해양 생태계 관리에서의 전략적인 결정을 지원할 수 있는 기회를 마련합니다.



### Zero-Shot Automatic Annotation and Instance Segmentation using LLM-Generated Datasets: Eliminating Field Imaging and Manual Annotation for Deep Learning Model Developmen (https://arxiv.org/abs/2411.11285)
- **What's New**: 본 연구에서는 사과의 인스턴스 세분화 (instance segmentation)를 위한 새로운 방법을 제안합니다. 이 방법은 노동집약적인 현장 데이터 수집과 수작업 주석 작업을 대체하여, 대규모 언어 모델 (LLM)을 활용해 합성 이미지를 생성하고 자동으로 주석을 달 수 있도록 합니다. 이를 통해 물리적 센서와 수작업 데이터 처리에 대한 의존성을 크게 줄이며 농업 AI 분야의 중요한 발전을 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 Segment Anything Model (SAM)과 YOLO11 모델을 통합하여, 생성된 합성 데이터셋을 사용해 YOLO11 모델을 학습했습니다. 자동으로 생성된 주석은 마스크 정확도 (mask precision) 0.902와 mAP@50 0.833을 기록하며, 실제 과수원 이미지를 검사의 기반으로 사용되었습니다. 이 과정에서 합성 데이터와 자동 주석으로 훈련된 YOLO11 모델이 사과를 올바르게 감지하고 구분할 수 있는 방법의 효율성을 보여줍니다.

- **Performance Highlights**: 특히 YOLO11m-seg 구성은 상업 과수원에서 수집한 테스트 이미지에서 마스크 정확도 0.902, mAP@50 0.833을 달성했습니다. 또한 YOLO11l-seg 구성은 40개의 LLM 생성 이미지에서 다른 모델들을 초과 달성하여 가장 높은 마스크 정확도와 mAP@50 지표를 기록하였습니다. 이 결과들은 자동으로 생성된 주석의 정확성과 효과를 검증하며, 전체적으로 지역화 및 세분화 작업의 품질을 높이는 데 기여합니다.



### Towards Open-Vocabulary Audio-Visual Event Localization (https://arxiv.org/abs/2411.11278)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 오픈 밸류 보캐블러리 오디오 비주얼 이벤트 로컬라이제이션(Open-Vocabulary Audio-Visual Event Localization, OV-AVEL) 문제를 제안합니다. 이는 훈련 데이터에 없는 사건을 포함한 테스트 데이터에서 이벤트를 정확히 지역화하고 명시적인 카테고리를 예측해야 하는 새로운 태스크입니다. 이를 위해 24,800개의 비디오와 함께 67개의 실생활 오디오 비주얼 장면을 포함하는 OV-AVEBench 데이터셋을 구축하였습니다. 이러한 데이터셋은 사람의 수동 주석으로 세분화된 고급 레이블을 제공하여 모델 개발과 평가를 용이하게 합니다.

- **Technical Details**: OV-AVEL 태스크를 지원하기 위해, 연구진은 데이터 수집, 주석화 및 분할 방법론을 제시합니다. 특히, 이 연구는 원활한 모달리티 및 카테고리 인식을 위해 최근의 라벨 기반 다중 모달 대조 모델을 활용할 것을 권장합니다. 여기서 ImageBind 모델을 사용하여 오디오, 비주얼 및 텍스트(이벤트 클래스)의 특징을 추출하고, 세그먼트 수준에서 그 유사성을 계산하여 예측을 수행합니다. 두 가지 베이스라인 접근법이 제안되어, 하나는 트레이닝이 필요없는 방법이고, 다른 하나는 추가적인 파인 튜닝(paradigm)을 사용하는 방법입니다.

- **Performance Highlights**: 초기 실험 결과, 파인 튜닝 기법이 트레이닝이 필요 없는 버전보다 훨씬 우수한 성능을 보였습니다. 특히, 이 방법은 정보의 연속성을 반영하는 훈련 데이터의 시간적 정보를 활용하여 더 효과적으로 성능을 향상시킵니다. OV-AVEBench 데이터셋과 함께 제안된 평가 지표도 유용성을 검증하는 데 기여하며, 세그먼트 및 이벤트 수준에서 정확히 예측된 사건을 평가할 수 있도록 돕습니다.



### Cross-Patient Pseudo Bags Generation and Curriculum Contrastive Learning for Imbalanced Multiclassification of Whole Slide Imag (https://arxiv.org/abs/2411.11262)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 병리학적 이미지 분석에서 멀티클래스 샘플 불균형 문제를 해결하기 위해 기초 분포와 유사한 서브백(sub-bag)을 생성하여 세부 정보 학습을 제안합니다. 또한, 효율적인 학습을 위해 의사 백(pseudo-bag) 생성을 활용하여 다량의 정보를 가져올 수 있습니다. 새롭게 도입된 샘플 선택 및 커리큘럼 대조 학습 전략은 모델의 안정성을 높이며, 백 수준의 표현 학습에서 멀티 인스턴스 백의 특징 분포로 전환하는 혁신적인 전략입니다.

- **Technical Details**: 제안된 프레임워크는 병리학적 이미지를 서브백으로 나누어 유사한 특징 분포를 재조직하고, 이를 통해 샘플 불균형 문제를 해결합니다. 또한, 대조 학습을 위한 친화적인 샘플 선택 전략을 사용하여 특징 표현 강화를 극대화하고, 각 카테고리를 균형 있게 다룹니다. 이 과정에서 대량의 중복 데이터를 활용하여 견고한 모델 학습이 가능하게 합니다.

- **Performance Highlights**: 본 방법은 종양 분류와 림프절 전이 등의 세 가지 데이터셋에서 다중 클래스 분류의 불균형 문제를 효과적으로 해결하며, F1 점수에서 평균 4.39 점의 성능 향상을 기록했습니다. 이는 이전의 두 번째로 우수한 방법에 비해 현저한 성과로, 이 연구의 우수성을 강조합니다.



### Semantic or Covariate? A Study on the Intractable Case of Out-of-Distribution Detection (https://arxiv.org/abs/2411.11254)
Comments:
          v1

- **What's New**: 이번 논문에서는 Out-of-Distribution (OOD) 데이터 탐지 작업에 대한 정의를 보다 명확히 하여 기존의 모호한 "semantic shift" 개념을 개선했습니다. OOD 데이터 샘플을 더 정확하게 구별할 수 있도록, Semantic Space 및 Covariate Space에 대한 새로운 정의를 제안합니다. 이 정의를 통해 OOD 데이터 탐지 작업의 복잡성을 줄이고, OOD 클래스 감지의 이론적 기반을 확립합니다.

- **Technical Details**: 난이도를 정의하기 위해 Semantic Space를 ID 클래스의 대표 피처 벡터의 선형 스팬을 사용하여 구성합니다. Covariate Space는 직접 합 성분 분해를 통해 표현하며, 이를 통해 ID 데이터만을 노출했을 때 모델이 구별할 수 없는 이동 범위를 식별할 수 있습니다. 이론적 분석을 통해 Semantic Space에서 이동이 없는 두 클래스는 기존 OOD 탐지 모델로는 구별할 수 없음을 보여줍니다.

- **Performance Highlights**: 우리는 제안한 정의의 필요성과 이론적 분석의 정확성을 검증하기 위해 여러 실험을 수행합니다. 이 실험들은 OOD 데이터 샘플을 탐지하는 작업에서 구별 가능성을 확보하는 것이 얼마나 중요한지를 강조합니다. 또한, 새로운 "Tractable OOD" 설정을 통해 OOD 탐지가 가능함을 실증적으로 나타냅니다.



### Noise Filtering Benchmark for Neuromorphic Satellites Observations (https://arxiv.org/abs/2411.11233)
Comments:
          17 pages, 8 figures, 1 table

- **What's New**: 이 논문에서는 위성 관측을 위한 새로운 이벤트 기반 노이즈 필터링 알고리즘을 제안합니다. 특히, 매우 희소한 장면에서 효과적으로 작동하는 알고리즘을 개발하여 기존의 필터링 알고리즘의 한계를 극복하고자 하였습니다.

- **Technical Details**: 제안된 알고리즘은 두 가지 접근 방식으로 구분됩니다: 논리 기반(logical-based)과 학습 기반(learning-based). 각 알고리즘의 성능을 11개의 최첨단 노이즈 필터링 알고리즘과 비교하여 신호 보존(signal retention) 및 노이즈 제거 정확도(noise removal accuracy)를 평가했습니다. 또한, 새로운 고해상도 위성 데이터셋과 실제 환경에서 수집된 Ground Truth를 공개하였습니다.

- **Performance Highlights**: CrossConv 알고리즘은 저활동 및 고활동 핫 픽셀(hot pixels)과 배경 노이즈를 효과적으로 제거하며, FEAST 알고리즘은 이전에 평가되지 않았던 노이즈 필터링 작업으로서의 성능을 입증하였습니다. 성능은 신호 유지율(SR), 노이즈 제거(NR), 잡음 정확도(DA), 핫 픽셀 제거(HPR) 및 ROC 곡선을 활용하여 정량적으로 측정하였습니다.



### BeautyBank: Encoding Facial Makeup in Latent Spac (https://arxiv.org/abs/2411.11231)
- **What's New**: 이 논문은 다양한 메이크업 응용 프로그램에서의 활용성을 높이기 위한 새로운 메이크업 인코더 BeautyBank를 제안합니다. 기존 메이크업 작업이 저차원 특징에 한정된 반면, BeautyBank는 고차원 공간에서 메이크업 특징을 인코딩하여 섬세한 디테일을 보존하고 메이크업 재구성을 위한 필수적인 정보를 유지합니다. 또한 Progressive Makeup Tuning (PMT) 전략을 도입하여 관련 없는 속성을 포함하지 않으면서 세부 정보의 보존을 강조합니다.

- **Technical Details**: BeautyBank는 바탕 메이크업 이미지와 메이크업 스타일을 분리하여 인코딩하는 혁신적인 구조를 갖추고 있습니다. 이 구조는 바탕 이미지의 아이덴티티를 보존하며, 메이크업 인코딩 과정에서 관련 없는 요소를 효과적으로 분리하는 데 중점을 두고 있습니다. 또한, 본 연구에서는 BMS(Bare-Makeup Synthesis) 데이터셋을 구축하여 324,000쌍의 고해상도 이미지를 수집하였습니다.

- **Performance Highlights**: 실험 결과, BeautyBank는 다양한 메이크업 관련 작업에서 뛰어난 적응성과 압도적인 성능을 나타내었습니다. 메이크업 컬러와 패턴의 세부 정보 유지에 집중함으로써, 기존 방식의 한계를 넘어서 새로운 메이크업 유사성 측정 및 메이크업 주입을 통한 인물 이미지 생성 등의 과제를 가능하게 합니다. 이 연구는 메이크업 분야에서의 다양한 응용 연구를 위한 기초 자료로 크게 기여할 수 있습니다.



### Efficient Transfer Learning for Video-language Foundation Models (https://arxiv.org/abs/2411.11223)
- **What's New**: 본 논문은 Multi-modal Spatio-Temporal Adapter (MSTA)를 제안하여 사전 훈련된 비전-언어 모델을 비디오 액션 인식 분야에 효과적으로 적용하는 새로운 방법을 소개합니다. MSTA는 텍스트와 비디오 모달리티 간의 조화를 이루도록 설계되어 일반 지식과 작업 특정 지식 간의 균형을 잡아줍니다. 또한, spatio-temporal description-guided consistency constraint를 도입하여 오버피팅을 줄이고 일반화를 향상시킵니다.

- **Technical Details**: MSTA 아키텍처는 텍스트 및 비디오 브랜치에 독립적인 프로젝션 레이어를 포함하여 각 모달리티의 작업 특정 지식을 학습합니다. 이를 통해 공통의 피처 프로젝션 레이어를 활용해 두 모달리티 간의 효과적인 정렬을 달성합니다. 훈련 과정 중 두 모달리티에서 기울기를 수신함으로써 최적화가 이루어집니다.

- **Performance Highlights**: MSTA는 Kinetics-400, UCF-101, HMDB-51 등 다양한 벤치마크 데이터셋에서 실험을 진행한 결과, 제로샷 및 몇 샷 학습 같은 오픈-어휘 작업에서 최첨단 성능을 달성했습니다. 전체적으로 경쟁 연구들과 비교하여 단지 2-7%의 훈련 가능한 파라미터만 사용하면서도 우수한 성능을 보였습니다.



### The Sound of Water: Inferring Physical Properties from Pouring Liquids (https://arxiv.org/abs/2411.11222)
Comments:
          25 pages, 17 figures. Project page at this https URL

- **What's New**: 이 논문은 액체를 붓는 일상적인 활동에서 오디오-비주얼 관찰과 물리학 간의 연결을 연구합니다. 우리는 오직 액체가 용기에 붓는 소리만으로 물리적 속성(액체 레벨, 용기 형태와 크기, 붓는 속도 및 채우는 시간)을 자동으로 추론하는 것을 목표로 합니다. 이를 위해 이론적으로 이러한 속성을 기본 주파수(피치)로부터 결정할 수 있음을 보이고, 시뮬레이션 데이터와 비주얼 데이터로부터 감독하에 피치 탐지 모델을 훈련합니다.

- **Technical Details**: 이 연구에서는 새로운 실제 붓기(video) 데이터셋을 소개하고, 훈련된 모델이 실제 데이터에 대해 물리적 속성을 추론할 수 있음을 보여줍니다. 또한, 다양한 용기 형태에 일반화할 수 있는 능력을 입증하고, 다른 데이터셋 및 실생활 YouTube 비디오에서의 성능을 강조합니다. 이 과정에서 물리학에서 영감을 얻은 목표를 사용하여 피치 탐지 모델을 훈련하여 문제를 해결합니다.

- **Performance Highlights**: 이 연구는 음향(acoustics), 물리학(physics), 및 학습(learning) 간의 교차점에서 좁지만 풍부한 문제에 대한 깊은 이해를 제공합니다. 논문에서 제시한 방법은 로봇 붓기(robot pouring)에서 다중 감각 인식을 향상시킬 수 있는 응용 가능성을 열어줍니다. 이러한 결과는 강력한 일반화 능력을 통해 여러 실제 데이터와 환경에서 적용될 수 있음을 보여줍니다.



### Relational Contrastive Learning and Masked Image Modeling for Scene Text Recognition (https://arxiv.org/abs/2411.11219)
- **What's New**: 이 논문은 Scene Text Recognition (STR)을 위한 새로운 접근 방식인 RCMSTR을 제안하고 있습니다. 기존의 자기 지도 학습(self-supervised learning, SSL) 방법들에서는 자연 이미지에만 적용된 사례가 많아, Scene Text 이미지의 특성을 도외시했습니다. 본 연구는 텍스트 요소 간의 관계를 활용하여 효과적인 특징을 학습하는 방법을 모색하고 있습니다.

- **Technical Details**: RCMSTR는 Relational Contrastive Learning(RCL)과 Masked Image Modeling(MIM)을 통합한 새로운 프레임워크입니다. RCL은 관계를 모델링하기 위해 이미지 재배치를 통해 새로운 관계를 생성하고, MIM은 블록 마스킹 전략을 도입하여 텍스트 이미지에서 기존의 문맥 정보를 더 잘 활용할 수 있게 합니다. 이러한 접근 방식을 통해 비지도 학습에서의 오버피팅 문제를 해결하고자 했습니다.

- **Performance Highlights**: RCMSTR는 다양한 평가 프로토콜에서 기존의 최첨단(self-supervised) STR 기법보다 우수한 성능을 보여줍니다. 특히, CNN과 ViT의 두 가지 아키텍처 모두에서 STR 표현 품질과 준지도 학습(semi-supervised learning)에서 두드러진 성과를 나타냈습니다. 추가적인 실험을 통해 모델의 주요 구성 요소의 효율성을 입증하였습니다.



### DeforHMR: Vision Transformer with Deformable Cross-Attention for 3D Human Mesh Recovery (https://arxiv.org/abs/2411.11214)
Comments:
          11 pages, 5 figures, 3DV2025

- **What's New**: 이번 논문에서는 DeforHMR이라는 새로운 단안적(Monocular) 3D 인간 메시 회복(Human Mesh Recovery, HMR) 프레임워크를 소개합니다. 이 프레임워크는 변형 가능한 주의 메커니즘(deformable attention mechanism)을 통해 인간의 자세 매개변수를 향상시킵니다. DeforHMR은 고정된 사전훈련 된 비전 변환기(Vision Transformer, ViT) 인코더에서 추출된 시각적 특징을 회귀(regress)하기 위해 변형 가능한 교차 주의(deformable cross-attention) 메커니즘을 사용합니다.

- **Technical Details**: DeforHMR는 비전 변환기와 변형 가능한 주의 메커니즘을 결합하여 높은 정확도로 인간 메시를 생성합니다. 변형 가능한 주의에서는 위치 정보가 부동 소수점(floating point values)으로 표현되어 입력 특징의 특성에 따라 동적으로 관련 공간 영역에 주의를 기울입니다. 이러한 방식은 신경망의 공간 이해 부족 문제를 해결하는 데 도움을 줍니다.

- **Performance Highlights**: DeforHMR은 3DPW 및 RICH와 같은 공인된 3D HMR 벤치마크에서 단일 프레임 회귀 기반 방법으로 최고 성능(State-of-the-art, SOTA)을 달성하였습니다. 이 연구는 변형 가능한 주의를 통해 공간적 특징을 더욱 유연하게 추출하는 새로운 효과적인 패러다임을 제시하고 있습니다.



### BVI-CR: A Multi-View Human Dataset for Volumetric Video Compression (https://arxiv.org/abs/2411.11199)
- **What's New**: 이 논문에서는 실재 물체와 환경의 3D 복제를 가능하게 하는 새로운 멀티뷰 볼륨 비디오 데이터셋 BVI-CR을 소개합니다. 이 데이터셋은 18개의 다양한 인간 행동을 묘사한 멀티뷰 RGB-D 캡처로 구성되어 있으며, 각 비디오 시퀀스는 10개의 1080p 해상도 시점에서 촬영된 10-15초 분량의 콘텐츠를 포함하고 있습니다. 이 데이터셋은 볼륨 비디오 압축 및 품질 평가 작업의 개발 및 검증 플랫폼으로 활용될 것입니다.

- **Technical Details**: BVI-CR 데이터셋은 Condense Reality에 의해 개발된 볼륨 비디오 캡처 스튜디오에서 촬영되었습니다. 이 스튜디오는 10대의 Microsoft Azure 카메라로 구성되어 있고, 각 카메라는 30FPS에서 2560×1440 해상도의 RGB 비디오와 640×576 해상도의 깊이 이미지를 동기적으로 캡처합니다. 촬영 과정에서 기하학적 보정이 수행되어 정확한 3D 포즈 데이터를 회복하고, 볼륨 콘텐츠 생성 시 RGB와 깊이 데이터를 융합하여 품질을 높입니다.

- **Performance Highlights**: BVI-CR 데이터셋을 바탕으로 세 가지 비디오 압축 방법의 성능을 벤치마킹하였으며, 그 결과 신경 표현 기반 방법이 기존 비디오 코딩 방법들에 비해 최대 38%의 PSNR에서 평균 코딩 이득을 보여주었습니다. 이 연구는 볼륨 비디오 압축 분야에서 신경 표현 방법의 큰 잠재력을 확인시키며, 다양한 3D 비디오 작업에서의 활용 가능성을 제시합니다.



### PickScan: Object discovery and reconstruction from handheld interactions (https://arxiv.org/abs/2411.11196)
Comments:
          7 pages, 8 figures, published in the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 본 연구는 사용자와 객체 간의 상호작용을 기반으로 하는 새로운 방법론을 통해 3D 씬의 조합적(representational) 재구성을 시도합니다. 이는 사용자가 RGB-D 카메라를 사용하여 객체를 이동시키고, 최종적으로 각 객체에 대한 3D 모델을 생성하는 능력을 제공합니다.

- **Technical Details**: 이 방법은 사용자-객체 상호작용을 감지하고, 조작된 객체의 마스크를 추출하는 새로운 접근 방식을 포함합니다. 이를 통해 객체 발견을 위한 클래스 비의존적(class-agnostic) 파이프라인을 제공하며, 사용자 움직임을 통해 객체를 완전히 스캔할 수 있습니다. RGB-D 및 관성 측정 장치(IMU) 데이터를 활용하여 작동합니다.

- **Performance Highlights**: 개발된 파이프라인은 100% 재현율에서 78.3%의 정밀도를 달성하였고, 평균 침식 거리(chamfer distance)는 0.90cm로 측정되었습니다. Co-Fusion과 비교하여 침식 거리가 73% 감소하였으며, 허위 탐지 감소는 99%에 달했습니다.



### Enhanced Anime Image Generation Using USE-CMHSA-GAN (https://arxiv.org/abs/2411.11179)
- **What's New**: 이 논문은 ACG (Anime, Comics, and Games) 문화의 인기에 힘입어 고품질 애니메이션 캐릭터 이미지를 생성하는 새로운 Generative Adversarial Network (GAN) 모델인 USE-CMHSA-GAN을 소개합니다. 이 모델은 전통적인 DCGAN 프레임워크를 기반으로 하며, 애니메이션 캐릭터 이미지를 위한 feature extraction (특징 추출) 능력을 강화하기 위해 USE와 CMHSA 모듈을 통합했습니다.

- **Technical Details**: USE-CMHSA-GAN은 기존의 GAN 구조에서 진화한 모델로, 향상된 이미지 품질을 위해 설계되었습니다. 특히, 다양한 GAN 모델들과 비교하여 FID (Fréchet Inception Distance) 및 IS (Inception Score) 점수에서 우수한 성능을 발휘하고 있습니다. 이는 애니메이션 캐릭터 이미지를 생성하는 과정에서의 특징 추출을 혁신적으로 개선합니다.

- **Performance Highlights**: 실험은 anime-face-dataset에서 수행되었으며, 그 결과 USE-CMHSA-GAN은 DCGAN, VAE-GAN, WGAN 등 다른 벤치마크 모델들을 초월하는 성과를 보였습니다. 이러한 결과들은 USE-CMHSA-GAN이 애니메이션 캐릭터 이미지 생성에 매우 효과적임을 나타내며, 생성 모델의 품질 개선에 대한 새로운 통찰을 제공합니다.



### Person Segmentation and Action Classification for Multi-Channel Hemisphere Field of View LiDAR Sensors (https://arxiv.org/abs/2411.11151)
Comments:
          6 pages, 9 figures, 4 tables, accepted for publication at IEEE/SICE International Symposium on System Integration (SII), Munich, Germany, January 2025

- **What's New**: 이 논문에서는 LiDAR 센서를 사용하여 사람 인식 및 행동 분류를 수행하는 새로운 방법을 제안합니다. Ouster OSDome-64 센서로 3D 스캔 데이터를 수집하고, MaskDINO 모델을 기반으로 하여 사람을 감지하고 행동을 인식하는 방법을 개발했습니다. 연구 결과, 이 접근법은 걷기, 손 흔들기, 앉기 등의 행동 상태를 정확히 추정하는 데 좋은 성능을 보였습니다.

- **Technical Details**: 이 연구에서는 180° 시야각(FoV)을 가진 반구형 LiDAR 센서에서 수집한 데이터 세트를 활용합니다. MaskDINO 모델을 기반으로 한 방법은 LiDAR 데이터의 다채널 표현과 추가적인 위치 인코딩을 결합하여 implement되었습니다. 채널별 기여도가 연구된 ablation study를 통해 사람 분할 작업에서의 각각의 기여도를 파악할 수 있었습니다.

- **Performance Highlights**: 제안된 모델은 사람 분할 작업과 행동 추정에서 우수한 성능을 보여주었으며, 코드 및 데이터 세트는 공개되어 연구 커뮤니티에서 사용할 수 있습니다. 훈련된 모델은 다양한 환경에서 수집된 442개의 스캔을 통해 다양한 사람의 행동을 인식했고, 이는 자율주행차, 서비스 로봇, 스마트 빌딩 등 다양한 응용 분야에 중요한 기여를 할 것으로 기대됩니다.



### A Comprehensive Survey on Visual Question Answering Datasets and Algorithms (https://arxiv.org/abs/2411.11150)
- **What's New**: 이 연구는 Visual Question Answering (VQA) 분야의 현재 상태를 상세히 분석하며 다양한 데이터셋과 모델을 체계적으로 분류합니다. 특히, 기존 데이터셋을 정리하고 각 카테고리의 개념과 특징을 요약하여, VQA의 이해도를 높이고자 합니다. VQA의 다양한 경향과 발전 방향을 이해하기 위해 이 논문에서는 학습과정 및 평가 기준에 대해서도 논의합니다.

- **Technical Details**: 연구에서는 VQA 데이터셋을 네 가지 카테고리로 나누었습니다: (1) 실제 이미지로 구성된 데이터셋, (2) 인공적으로 생성된 합성 이미지 데이터셋, (3) 특정 영역에서의 모델 성능을 평가하기 위한 진단 데이터셋, (4) 외부 지식을 활용하는 능력을 측정하는 지식 기반(KB) 데이터셋. 또한 모델의 주요 패러다임으로는 정보 융합(fusion), 주의(attention) 메커니즘, 외부 지식 기반, 복합적 추론(composition or reasoning), 설명(explanation), 그래프 모델(graph models)을 언급하며, 각 패러다임이 VQA에 어떻게 적용되는지를 분석합니다.

- **Performance Highlights**: 논문에서는 VQA 모델의 성능 평가 결과와 함께 각 카테고리의 데이터셋이 모델의 학습과 성능 개선에 미치는 영향을 설명합니다. 특히, 다양한 방법론을 통한 정보 융합 및 모듈의 사용이 모델의 문제 해결 능력에 긍정적인 영향을 미친다고 강조합니다. 추가적으로, 장면 텍스트 이해, 소수 세기(counting), 편향 감소(bias reduction)와 같은 다른 주제들도 논의되어 VQA의 전체적인 발전 방향에 대한 통찰을 제공합니다.



### Oscillation Inversion: Understand the structure of Large Flow Model through the Lens of Inversion Method (https://arxiv.org/abs/2411.11135)
- **What's New**: 이번 연구에서는 대규모 텍스트-이미지 확산 모델에 적용된 역전 방법의 진동 현상을 탐구합니다. 특히, 'Flux' 모델을 집중 분석하며 고정점(iterative) 기반의 역전 접근 방식을 사용하여 실제 이미지를 반전시킵니다. 연구 결과, 반전 과정에서 솔루션은 수렴하지 않고 여러 의미적 클러스터 사이에서 진동하는 특성을 보였습니다.

- **Technical Details**: 이 연구의 핵심 방법은 고정점 반복 방식을 통해 수치적 조작을 수행하며, 이를 통해 의미적으로 일관된 클러스터를 활용한 이미지 최적화 방법을 제시합니다. 연구는 기존 DDIM(inversion) 같은 평면화된 접근 대신, 새로운 진동 기반의 전환 기술을 도입하여 사용자가 원하는 맞춤형 조작을 가능하게 합니다. 본 논문에서 제안한 세 가지 확장은 의미적 가이던스를 제공하고, 집단 반전 및 최적화 과정을 포함하여 다양한 이미지 조작 작업에 적합하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 본 연구의 방법론은 이미지 향상, 메이크업 전이 및 재구성 품질 등 여러 작업에서 더욱 높은 품질을 시연했습니다. 실 이미지 편집에서의 성과는 양적으로 분석되었으며, 의미적 정확성을 보장합니다. 연구 결과는 이론적 통찰을 바탕으로 하여 대규모 데이터셋에 대해 효과적으로 검증되었습니다.



### Label Sharing Incremental Learning Framework for Independent Multi-Label Segmentation Tasks (https://arxiv.org/abs/2411.11105)
- **What's New**: 이 논문에서는 각기 다른 라벨 집합을 가진 여러 데이터셋을 위해 세그멘테이션 모델을 구축할 때, 일반적인 접근 방식으로 각 데이터셋에 대해 하나의 모델을 학습하는 방법 대신, 공통 라벨 공간을 사용할 수 있는 새로운 '라벨 공유(label sharing)' 프레임워크를 제안합니다. 이 프레임워크는 다양한 라벨 집합을 단일 대규모 데이터셋으로 변환하여, 단일 모델 학습을 통해 모든 세그멘테이션 작업을 해결할 수 있도록 합니다.

- **Technical Details**: 라벨 공유 프레임워크는 서로 다른 작업의 라벨을 그룹화하고 각 그룹에 대해 공통 추상 라벨을 할당하는 방식입니다. 각 라벨은 하나의 공유 라벨에만 할당되며, 각 공유 라벨 그룹 내에는 각 작업에서 최대 한 개의 라벨만 포함될 수 있습니다. 이러한 방식으로 라벨 간의 공통 특성을 활용하여 세그멘테이션 모델 학습을 지원합니다.

- **Performance Highlights**: 우리의 방법은 다양한 의학적 이미지 세그멘테이션 데이터셋에서 실험적으로 검증되었으며, 제안하는 방법의 성능과 점진적인 학습 능력이 기존의 다른 방법들과 비교하여 효과적임을 보여줍니다. 본 연구는 각기 다른 라벨 수를 가진 다양한 세그멘테이션 작업에서 라벨 공유 프레임워크의 유용성과 다재다능성을 정성적 및 정량적으로 입증합니다.



### MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild (https://arxiv.org/abs/2411.11098)
- **What's New**: 이번 연구에서는 MolParser라는 새로운 Optical Chemical Structure Recognition (OCSR) 방법을 제안하여 화학 구조를 정확하게 인식하는 데 기여하고 있습니다. MolParser는 Markush 구조와 같은 복잡한 화학 구조를 정확하고 신속하게 인식할 수 있는 엔드투엔드 방식의 모델로, 최신의 이미지 분석 기술을 통합하였습니다. 또한, 대규모의 주석이 달린 데이터셋인 MolParser-7M을 통해 이러한 모델의 학습을 지원하고 있습니다.

- **Technical Details**: MolParser는 확장된 SMILES 포맷을 사용하여 복잡한 화학 구조를 더 잘 표현할 수 있도록 설계되었습니다. 이 포맷은 RDKit과 호환되며, 다양한 화학 컴파운드에 대해 LLM (Large Language Models)을 적용할 수 있는 유연성을 제공합니다. 모델은 이미지 기능 추출을 위한 비전 인코더와 SMILES 생성을 위한 BART 디코더를 포함하는 아키텍처로 구성되어 있습니다.

- **Performance Highlights**: MolParser는 USPTO 벤치마크에서 98%를 넘는 최첨단 정확도를 달성하였으며, WildMol-20k에서는 Tanimoto 유사성 점수 91% 이상을 기록하였습니다. 이러한 성과는 기존의 방법들인 MolGrapher와 비교할 때 월등한 성능을 보여주며, 실세계 데이터에 대처하는 데 있어 높은 신뢰성을 입증하고 있습니다.



### D-Cube: Exploiting Hyper-Features of Diffusion Model for Robust Medical Classification (https://arxiv.org/abs/2411.11087)
Comments:
          10 pages, 2 figures

- **What's New**: 이 논문은 깊은 학습 기술을 의료 이미징에 통합하여 췌장암과 유방암의 진단 효율성과 정확성을 향상시키기 위한 새로운 방법인 D-Cube를 소개합니다. D-Cube는 확산 모델(difussion model)과 대조 학습(contrastive learning)을 결합하여 암 진단을 개선하는 방식으로 작동합니다. 이 연구는 데이터 불균형과 제한된 샘플과 같은 도전 과제를 해결하기 위해 특징 선택(feature selection) 기술을 적용하고 있습니다.

- **Technical Details**: D-Cube는 확산 모델에서 추출한 고품질 특징 표현을 사용하여 진단 정확도를 높입니다. 이 방법은 Gaussianity 메트릭을 기반으로 한 새로운 특징 선택 기법을 중심으로, 임상적으로 중요성이 높은 특징을 가장 잘 추출하여 분류 성능을 대폭 향상시킵니다. 또한 D-Cube는 여러 의료 이미징 모달리티에서 일관된 성능을 발휘하며, 의료 응용에 있어 예측 정확도를 개선하고 있습니다.

- **Performance Highlights**: D-Cube의 성능은 CT, MRI, X-Ray 등 다양한 의료 이미징 모달리티에서 평가되었습니다. 실험 결과, D-Cube는 기존의 최첨단 모델과 비교할 때 모든 이미지 모달리티에서 우수한 성능을 보여줍니다. 이 연구는 D-Cube가 12개의 기준 모델에 대한 성능 우수성을 입증하였으며, 의료 이미징 분야에서 그 적용 가능성을 확대하는 데 기여하고 있습니다.



### Electrostatic Force Regularization for Neural Structured Pruning (https://arxiv.org/abs/2411.11079)
- **What's New**: 이 논문에서는 처음으로 물리학의 전하(charge)와 전기적 힘(electrostatic force) 개념을 심층 합성곱 신경망(Deep Convolutional Neural Networks, DCNNs)의 학습 과정에 통합하는 새로운 접근 방식을 제안합니다. 이 방법은 필터의 전하에 따라 서로 다른 힘을 적용하여 모델의 가중치를 최적화하고 필터의 중요도를 평가하면서 필터를 효율적으로 제거할 수 있습니다. 저자들은 이 방법이 기존의 구조적 가지치기(structured pruning) 방법들보다 간단하게 구현될 수 있으며, 별도의 아키텍처 수정이나 광범위한 파인 튜닝(fine-tuning) 필요 없이 작동한다고 주장합니다.

- **Technical Details**: 전기적 힘의 크기는 합성곱 필터(convolution filter)와 소스 필터(source filter)의 전하 곱에 비례하며, 두 필터 간의 거리의 제곱에 반비례합니다. 필터와의 상호작용에 따라 유사한 전하를 가진 필터들은 서로를 밀어내고, 반대 전하를 가진 필터들은 서로를 끌어당겨서 가중치가 0에 가깝게 조정됩니다. 이러한 방식은 모델 훈련 완료 후 설정된 가지치기 비율에 따라 덜 중요한 가중치의 필터를 제거할 수 있도록 하여, 중요한 정보가 포함된 필터는 유지합니다.

- **Performance Highlights**: MNIST, CIFAR, ImageNet 데이터셋을 사용한 검증 결과, 제안된 방법이 기존의 구조적 가지치기 방법들과 비교했을 때 경쟁력 있는 성능을 달성한 것으로 나타났습니다. 이 접근법은 모델의 복잡성을 줄이면서도 정확도 감소를 최소화할 수 있는 장점을 가지고 있으며, 훈련되지 않은 모델과 사전 훈련된 모델 모두에 쉽게 적용될 수 있습니다.



### Skeleton-Guided Spatial-Temporal Feature Learning for Video-Based Visible-Infrared Person Re-Identification (https://arxiv.org/abs/2411.11069)
- **What's New**: 이 논문에서는 Video-based Visible-Infrared Person Re-Identification (VVI-ReID) 문제를 다루기 위해 새로운 Skeleton-guided spatial-Temporal feAture leaRning (STAR) 방법을 제안합니다. STAR는 skeleton 정보를 활용하여 인프라레드 비디오의 공간-시간적 특성을 개선하며, 기존 방법들이 미흡한 공간-시간적 정보 개선에 초점을 맞추지 않은 점을 보완합니다. 이 연구는 skeleton 기반의 전략을 통해 영상의 개인 식별 성능을 획기적으로 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: STAR 방법은 프레임 레벨과 시퀀스 레벨에서 두 가지 수준의 skeleton-guided 전략을 적용합니다. 프레임 레벨에서는, 저품질 영상에서의 비주얼 특징을 정제하기 위해 강인한 구조의 skeleton 정보를 사용합니다. 시퀀스 레벨에서는 skeleton key points 그래프에 기반한 특성 집계 메커니즘을 설계하여 다양한 신체 부위의 기여도를 학습함으로써 글로벌 특성의 정확성을 더욱 향상시킵니다.

- **Performance Highlights**: 벤치마크 데이터 세트에서 진행된 실험 결과, STAR는 최신 기술보다 우수한 성능을 보여주며, skeleton 정보를 효과적으로 활용하여 수많은 제약 조건에도 불구하고 공간-시간적 특성의 정확성을 개선했습니다. 노이즈나 부분 가림 현상에서 강인한 특성을 발휘하며, 프레임 수준의 이미지 기능 개선과 시퀀스 수준의 기능 집계를 통해 성능이 크게 향상되었습니다.



### TS-LLaVA: Constructing Visual Tokens through Thumbnail-and-Sampling for Training-Free Video Large Language Models (https://arxiv.org/abs/2411.11066)
Comments:
          work in progress

- **What's New**: 최근 멀티모달 대규모 언어 모델(LLMs)의 발전으로 비디오 이해 작업에 대한 새로운 접근 방식이 제시되었습니다. 기존의 비디오 LLM은 고품질 비디오-텍스트 데이터의 부족으로 인해 훈련이 어려운 반면, 이미지-텍스트 데이터는 쉽게 얻을 수 있습니다. 본 연구에서는 이미지 LLM을 비디오 이해 작업에 확대하기 위한 방법으로, 훈련이 필요 없는 비디오 LLM인 TS-LLaVA를 개발했습니다.

- **Technical Details**: TS-LLaVA는 Thumbnail-and-Sampling 전략을 통해 시각 토큰을 생성하는 방법을 택합니다. 이 방법에서는 비디오에서 간격을 두고 몇 개의 프레임을 선택하여 Thumbnail 이미지를 구성하고, 전체 입력 프레임에서 샘플링한 시각 토큰을 보완하여 사용합니다. 이를 통해 비디오 내용의 이해도를 높이고, 훈련이 없는 상태에서도 효과적으로 작동할 수 있도록 합니다.

- **Performance Highlights**: TS-LLaVA는 다양한 벤치마크에서 훈련이 없는 비디오 LLM 중에서 새로운 최첨단 성능을 기록했습니다. 특히, 34B 모델이 MVBench 벤치마크에서 GPT-4V를 초월하였고, 72B 훈련 기반 비디오 LLM인 Video-LLaMA2와 비슷한 성능을 MLVU 벤치마크에서 달성했습니다. 이를 통해 TS-LLaVA는 비디오 이해의 새로운 가능성을 열어주고 있습니다.



### StableV2V: Stablizing Shape Consistency in Video-to-Video Editing (https://arxiv.org/abs/2411.11045)
Comments:
          Project page: this https URL, code: this https URL, model weights: this https URL, dataset (DAVIS-Edit): this https URL

- **What's New**: 본 연구에서는 Shape-consistent video editing 방법인 StableV2V를 제안합니다. 기존의 비디오 편집 방법들이 사용자의 프롬프트와 전달된 모션 간의 정렬이 부족하여 일관성이 떨어지는 문제를 해결하고자 합니다. StableV2V는 전체 편집 과정을 여러 개의 단계로 분해하여 처음부터 프레임을 편집하고, 사용자 프롬프트와 모션 간의 정렬을 수립한 후, 이 정렬을 바탕으로 나머지 프레임에도 편집된 내용을 전파합니다.

- **Technical Details**: StableV2V는 비디오 편집의 모든 절차를 연속적인 단계로 나누어 처리합니다. 첫 번째로 비디오의 첫 프레임을 편집하고, 이어서 사용자 프롬프트와의 정렬을 생성합니다. 마지막으로, 이 정렬을 기반으로 편집된 내용을 나머지 프레임에 전파하여 전체 비디오의 일관성을 유지합니다. 이는 효과적인 모션 전달 방법으로, 다양한 편집 프로세스에서의 안정성을 제공합니다.

- **Performance Highlights**: DAVIS-Edit이라는 평가 벤치마크를 통해 다양한 유형의 프롬프트와 난이도를 고려하여 비디오 편집 성능을 종합적으로 평가하였습니다. 실험 결과, StableV2V는 기존의 최선의 연구와 비교하여 뛰어난 성능, 시각적 일관성, 효율적인 추론을 보여주었습니다. 이로써 비디오 편집의 새로운 가능성을 제시하며, 생성적 AI 분야에서 중요한 진전을 이뤘습니다.



### Wafer Map Defect Classification Using Autoencoder-Based Data Augmentation and Convolutional Neural Network (https://arxiv.org/abs/2411.11029)
Comments:
          26 pages, 11 figures, including dataset preprocessing, proposed methods, and experimental results

- **What's New**: 이 논문은 반도체 제조에서 웨이퍼 결함 지도(Wafer Defect Maps, WDMs)의 정확한 분류 문제를 해결하기 위해 새로운 방법을 제안합니다. 이 방법은 자가 인코더(self-encoder) 기반 데이터 증강(data augmentation) 기술과 컨볼루션 신경망(Convolutional Neural Networks, CNN)을 결합하여 발생하는 노이즈를 개선합니다. 제안된 모델은 일반 및 희귀 결함 패턴의 정확한 분류를 가능하게 하며, 실험 결과 WM-811K 데이터셋에서 98.56%의 분류 정확도를 달성합니다.

- **Technical Details**: 제안된 방법은 자가 인코더를 이용해 잠재 공간(latent space)에 노이즈를 도입하여 데이터 다양성을 향상시키고 클래스 불균형을 완화합니다. 이를 통해 CNN은 확대된 데이터셋을 학습하게 되어 일반화 능력이 개선됩니다. 이 모델과 전통적인 기계 학습 방법들, 예를 들어 Random Forest, SVM, Logistic Regression과의 성능 비교도 이루어졌습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 Random Forest, SVM, Logistic Regression에 비해 각각 19%, 21%, 27%의 성능 향상을 보여줍니다. 이러한 결과는 제안된 접근 방식의 강력함과 효과성을 강조하며, 반도체 제조에서 웨이퍼 결함 탐지 및 분류를 위한 신뢰할 수 있는 솔루션을 제공합니다.



### VeGaS: Video Gaussian Splatting (https://arxiv.org/abs/2411.11024)
- **What's New**: 이 논문에서는 비디오 데이터를 보다 사실적으로 수정할 수 있는 Video Gaussian Splatting (VeGaS) 모델을 소개합니다. VeGaS는 새로운 Folded-Gaussian 분포 패밀리를 제안하여 비디오 스트림의 비선형 동역학을 포착하고, 서로 조건부 분포로서 2D Gaussian을 모델링합니다. 기존의 Gaussian Splatting 모델을 능가하는 VeGaS는 복잡한 분포를 통합할 수 있는 능력이 향상되었습니다.

- **Technical Details**: VeGaS 모델의 핵심은 Folded-Gaussians라는 혁신적인 기능 패밀리로, 이는 비선형 구조를 모델링하며 3D 공간에서 비디오 프레임을 평행 평면으로 취급합니다. 이 모델은 조건부 3D Gaussian 요소를 사용하여 비디오의 빠른 변화와 복잡한 전환을 정확하게 모델링하는데 기여합니다. 또한 VeGaS는 배경을 표현하기 위한 대규모 Gaussian과 특정 프레임에서만 존재하는 요소를 표현하기 위한 소규모 Gaussian을 모두 활용합니다.

- **Performance Highlights**: 우리의 실험은 VeGaS가 프레임 재구성 작업에서 최첨단 솔루션보다 뛰어난 성능을 보이며, 비디오 데이터의 현실적인 수정이 가능하다는 것을 보여줍니다. VeGaS로 생성된 비디오는 고품질 렌더링을 제공하며, 전체 비디오 및 선택된 프레임 모두를 수정할 수 있는 유연성을 가집니다. 코드는 연구 결과와 함께 제공되며, 연구 커뮤니티에서 사용 가능하다고 명시되어 있습니다.



### Time Step Generating: A Universal Synthesized Deepfake Image Detector (https://arxiv.org/abs/2411.11016)
Comments:
          Submitted to CVPR 2025, 9 pages, 7 figures

- **What's New**: 이번 논문은 Time Step Generating (TSG)이라는 새로운 방법을 제안합니다. 이는 기존의 diffusion 모델에서 생성된 이미지를 탐지하기 위한 것으로, 사전 훈련된 모델의 재구성 능력이나 특정 데이터셋에 의존하지 않습니다. TSG는 특징 추출을 위해 사전 훈련된 diffusion 모델의 네트워크를 활용하며, 생성 이미지와 실제 이미지 간의 미세한 차이를 포착하는 데 집중합니다. 이를 통해 이미지가 합성인지 실제인지 효과적으로 감지할 수 있습니다.

- **Technical Details**: Paper에서는 Score-Based Diffusion 모델과 Denoising Diffusion Probabilistic Models (DDPM)에 대한 배경 정보를 제공합니다. TSG는 noise ϵ의 예측을 통해 이미지를 분류하는 방법으로, 사전 훈련된 U-Net을 사용하여 이미지에서 특징을 직접 추출합니다. 이는 모든 형태의 이미지 데이터셋에 대해 적용 가능하며 다양한 생성기를 다룰 수 있습니다. TSG는 기존의 재구성 기반 방법들에 비해 간소화된 접근 방식을 제시합니다.

- **Performance Highlights**: 실험 결과, TSG는 GenImage 벤치마크에서 10배 빠른 속도와 함께 기존 방법들에 비해 상당한 정확도 향상을 보여주었습니다. 특히, TSG는 특정 데이터셋이나 객체 유형에 대한 재구성 능력을 고려할 필요 없이 광범위한 적용 가능성을 가지고 있습니다. 이러한 성과는 TSG의 접근 방식이 아키텍처와 데이터 내용의 다양성 덕분임을 나타냅니다.



### CCi-YOLOv8n: Enhanced Fire Detection with CARAFE and Context-Guided Modules (https://arxiv.org/abs/2411.11011)
Comments:
          8 pages,7 figures

- **What's New**: 본 논문에서는 소규모 화재와 연기를 탐지할 수 있도록 개선된 YOLOv8 모델인 CCi-YOLOv8n을 소개합니다. 이 모델은 CARAFE 업샘플링 연산자와 컨텍스트 가이드 모듈을 통합하여 업샘플링 및 다운샘플링 중 정보 손실을 줄이고 더욱 강화된 특징 표현을 유지합니다. 또한, 개선된 C2f 모듈을 통해 소규모 타겟과 다양한 연기 패턴을 효과적으로 포착하여 원 모델보다 훨씬 향상된 탐지를 제공합니다.

- **Technical Details**: CCi-YOLOv8n은 기본적으로 YOLOv8n 모델에서 출발하여 정보 손실을 최소화하기 위한 다양한 개선 사항을 적용하였습니다. CARAFE와 CGD 모듈은 샘플링 과정 동안 정보 보존을 최적화하고, iRMB 모듈은 다중 스케일의 특징을 효과적으로 추출하여 원본 C2f 모듈의 기능을 강화합니다. 이 과정을 통해 모델은 더욱 정밀한 바운딩 박스 예측을 가능하게 하며, 우수한 실시간 성능을 유지합니다.

- **Performance Highlights**: 실험 결과, CCi-YOLOv8n은 기존 YOLOv8n에 비해 탐지 정확도가 현저히 향상되었습니다. Web-Fire 데이터셋을 사용하여 다양한 실제 환경에서 화재와 연기를 탐지하는 데 있어서 강력한 성능을 발휘하며, 높은 정확도를 제공합니다. 이 모델은 타 모델들과 비교할 때 낮은 거짓 탐지율과 놓치는 탐지율을 보여주어 화재 탐지 작업에서 신뢰할 수 있는 솔루션으로 자리 잡기를 기대합니다.



### EROAM: Event-based Camera Rotational Odometry and Mapping in Real-tim (https://arxiv.org/abs/2411.11004)
- **What's New**: 이번 논문에서는 EROAM이라는 새로운 이벤트 기반 로테이셔널 오도메트리 및 매핑 시스템을 소개합니다. 기존의 방식들과 달리 이 시스템은 이벤트를 단위 구에 매핑하여 회전 추정을 실시간으로 수행하며, Event Spherical Iterative Closest Point(ES-ICP)라는 새로운 기하학적 최적화 프레임워크를 도입합니다. 이 방법론은 정확한 카메라 회전 추정 및 연속적인 매핑을 가능하게 하여 공간 해상도를 향상시킵니다.

- **Technical Details**: EROAM은 구형 이벤트 표현을 통해 회전 운동을 간소화하며, 병렬 점-선 최적화를 통해 효율적인 계산을 달성합니다. 이 approach는 기존의 이산화된 공간을 무시하고, 연속적인 ℝ3 공간에서 작업함으로써 양호한 성능을 보입니다. ES-ICP 알고리즘은 이벤트 데이터에 특화되어 있으며, 회전에 대한 강력한 추정을 지원합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, EROAM은 정확성, 강건성 및 계산 효율성 면에서 기존 최신 기법들을 초월하는 성능을 발휘합니다. 이 시스템은 높은 각속도와 연장된 시퀀스를 포함한 도전적인 상황에서도 일관된 성능을 유지하며, 매우 높은 품질의 파노라마 재구성을 생성할 수 있습니다. 이러한 결과는 이벤트 카메라 사용을 통한 혁신적인 회전 추정 가능성을 제시합니다.



### TeG: Temporal-Granularity Method for Anomaly Detection with Attention in Smart City Surveillanc (https://arxiv.org/abs/2411.11003)
- **What's New**: 이 논문은 비디오 감시에서의 이상 감지에 대한 최신 기술을 제안하고 있습니다. 특히, 다양한 시간 스케일에서 공간-시간적(spatio-temporal) 특징을 결합하는 Temporal-granularity 메소드를 도입하여 보다 정교한 이상 감지를 가능하게 합니다. 이 모델은 Multi-head Cross-Attention (MCA) 및 Multi-head Self-Attention (MSA) 블록을 활용하여 기능 간의 관계를 학습합니다. 또한, Smart City 연구와 관련된 새로운 이상 유형을 포함하여 UCF-Crime 데이터셋을 확장하였습니다.

- **Technical Details**: TeG(Temporal-granularity) 모델은 비디오 스위치 트랜스포머(Video Swin Transformer, VST)에서 추출한 다양한 시간 스케일의 공간-시간적(spatio-temporal) 특징을 통합하여 동작합니다. 모델은 짧은, 중간, 긴 지속 시간을 가진 이상 행동을 효과적으로 분리하기 위해 시간적 세분화 변수를 고려하여 비디오 클립을 여러 세그먼트로 나눕니다. 각 비디오 세그먼트는 추가적으로 "비디오 청크"로 나뉘며, 이를 통해 특징 벡터를 생성하여 총 32개의 세그먼트로부터 최종 출력을 얻습니다. 특징 벡터는 다시 단기, 중기, 장기 시간적 세분화로 나뉘어 해당 특징을 표현합니다.

- **Performance Highlights**: TeG 모델은 실제 도시 감시 시스템에 배포되어 성공적인 실시간 결과를 달성하였습니다. 이 모델은 이전의 기술들과 비교하여 다양한 이상 상황을 더 포괄적으로 커버할 수 있는 장점을 보여주었으며, 비디오 감시 데이터의 자동 분석을 통해 공공 안전을 향상시키는 데 기여합니다. 마지막으로, 이 연구는 Smart City 프로젝트의 일환으로 다양한 이상 시나리오를 다루어 공공 안전을 증진하는 것을 목표로 하고 있습니다.



### Unveiling the Hidden: Online Vectorized HD Map Construction with Clip-Level Token Interaction and Propagation (https://arxiv.org/abs/2411.11002)
Comments:
          18 pages, 9 figures, NeurIPS 2024

- **What's New**: 본 논문에서는 도로 기하정보의 예측 및 구성에 관한 새로운 패러다임인 MapUnveiler를 도입하여, 인접 입력 프레임 간의 시간 정보를 효과적으로 활용할 수 있는 방법을 제안합니다. 기존의 HD 맵 구축 방법들이 다루지 못했던 가려진 맵 요소들을 명확하게 드러내며, 클립 간 정보 전파를 통해 장기적인 맵 정보 이용을 가능하게 합니다.

- **Technical Details**: MapUnveiler는 클립 입력이 주어졌을 때, 시간적 맵 정보가 포함된 컴팩트 클립 토큰을 생성하고 이를 통해 BEV(탑 뷰) 특징을 갱신하여 숨겨진 맵 요소를 드러내는 데 중점을 두고 있습니다. 또한, 클립 레벨 파이프라인을 활용하여 효율적인 온라인 추론을 제공하고, 여러 프레임의 입력을 빠르게 처리할 수 있습니다.

- **Performance Highlights**: MapUnveiler는 nuScenes와 Argoverse2 기준 데이터셋에서 최첨단 성능을 달성하였으며, 심각한 가림 조건에서도 +10.7% mAP 향상된 결과를 보였습니다. 실험 결과, 이 모델은 시간적 정보의 누적 영향을 적절히 활용하여 더 긴 인식 범위에서도 성능을 향상시킬 수 있음을 증명합니다.



### Framework for developing and evaluating ethical collaboration between expert and machin (https://arxiv.org/abs/2411.10983)
Comments:
          Accepted in ECAI Workshop AIEB

- **What's New**: 이번 논문은 정밀 의학(precision medicine)에서의 인공지능(AI) 통합의 새로운 프레임워크를 제안합니다. 이 프레임워크는 질병 진단 및 개인 맞춤형 치료 계획을 위한 다중 모달 AI(multi-modal AI)의 개발과 윤리적 평가를 강조합니다. 인슐린 관리 사례 연구를 통해 제안된 방법론이 실제 적용 가능한지를 살펴봅니다.

- **Technical Details**: 다중 모달 AI 접근법을 통해 임상 정보의 콘텐츠를 개선하며, 고급 딥 러닝(deep learning) 모델과 전문가 지식 기반의 메커니즘 모델을 통합합니다. 이를 통해 환자 개별의 병리학적 경향을 학습하고, 최종 진단 및 치료 계획이 임상 의사와 AI 간의 협력으로 도출되도록 설계되었습니다. 특히, T1D 백신 관리를 고려한 자동 인슐린 공급 시스템의 타임 라인 구성과 관련된 AI의 역할이 강조됩니다.

- **Performance Highlights**: 제안된 모델은 인공지능이 단순히 자동화 도구가 아니라 임상적 결정 과정에 기여하는 주요 요소로 작용함을 보여줍니다. 목표는 알고리즘이 실제 데이터와 임상 전문가의 피드백을 통해 지속적으로 개선되는 것입니다. 이러한 접근 방식은 윤리적 결정, 즉 이익(originality), 해롭지 않음(non-maleficence), 자율성(autonomy), 정의(justice)를 보장하는 데 중점을 두고 있습니다.



### VidComposition: Can MLLMs Analyze Compositions in Compiled Videos? (https://arxiv.org/abs/2411.10979)
- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)의 비디오 구성 이해 능력을 평가하기 위한 새로운 벤치마크인 VidComposition을 소개합니다. 기존의 평가 기준은 비디오의 일반적인 이해에 중점을 두었지만, 비디오 구성에서의 MLLMs의 성능은 충분히 평가되지 않았습니다. VidComposition은 982개의 비디오와 1706개의 선택형 질문으로 구성되어 스토리텔링을 통한 깊이 있는 비디오 해석을 가능하게 합니다.

- **Technical Details**: VidComposition은 Cinematography Analysis, Character Understanding, Narrative Understanding, Scene Perception, Making Analysis의 5개 주요 카테고리 및 15개 하위 과제를 포함합니다. 수집된 비디오는 주로 코멘터리 비디오에서 출처를 두고, 각 비디오 조각에 대해 여러 인간 주석자가 고유한 질문과 답안을 작성합니다. 데이터 수집과 필터링 과정에서 심리적 고통을 유발할 수 있는 부적절한 내용을 제거하여 데이터셋의 신뢰성을 높였습니다.

- **Performance Highlights**: MLLMs를 VidComposition에서 평가한 결과, 인간의 비디오 구성 이해 능력에 비해 MLLMs가 상당히 뒤처짐을 보였습니다. 특히, 상위 모델들도 기본적인 인식 과업에서는 꽤 좋은 성능을 보였지만, 복잡한 비디오 구성에서는 부족한 성과를 보였습니다. 이 성능 차이는 MLLMs가 다층적인 비디오 구조를 포착하는 데 한계가 있음을 강조합니다.



### V2X-Radar: A Multi-modal Dataset with 4D Radar for Cooperative Perception (https://arxiv.org/abs/2411.10962)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문에서는 협력적 인식(cooperative perception) 분야에서 4D Radar를 포함한 최초의 대규모 실제 다중 모달 데이터셋인 V2X-Radar를 제시합니다. 기존의 데이터셋은 카메라(camera)와 LiDAR만을 중심으로 구성되어 있었던 반면, 이번 데이터셋은 다양한 날씨 조건과 시간대에서 수집된 데이터로 강력한 인식을 가능하게 합니다. 즉, V2X-Radar는 4D Radar를 활용하여 자율주행 차량의 안전성을 향상시키기 위한 중요한 기초자료를 제공하고 있습니다.

- **Technical Details**: V2X-Radar 데이터셋은 연결된 차량 플랫폼과 인텔리전트 도로측 장치에서 수집되었습니다. 이 데이터셋은 20K LiDAR 프레임, 40K 카메라 이미지, 20K 4D Radar 데이터로 구성되어 있으며, 총 350K 주석이 달린 바운딩 박스가 포함되어 있습니다. 또한, 데이터셋은 V2X-Radar-C(협력적 인식), V2X-Radar-I(도로측 인식), V2X-Radar-V(단일 차량 인식)이라는 세 가지 하위 데이터셋으로 나뉘어 다양한 연구 분야를 지원합니다.

- **Performance Highlights**: V2X-Radar는 카메라, LiDAR 및 4D Radar의 세 가지 센서를 포함하여 보다 다양한 방식의 협력적 인식 연구를 가능하게 합니다. 이 데이터셋은 다양한 날씨 조건 및 도전적인 교차로 시나리오에 대한 데이터를 포함하고 있어, 협력적 인식 연구에 의미 있는 모형을 제공합니다. 또한, 최근 인식 알고리즘의 포괄적인 벤치마크가 이 세 가지 하위 데이터셋에서 제공되어 연구자들이 해당 알고리즘의 성능을 비교할 수 있는 기반을 마련하고 있습니다.



### Map-Free Trajectory Prediction with Map Distillation and Hierarchical Encoding (https://arxiv.org/abs/2411.10961)
- **What's New**: 본 논문에서는 MFTP라는 맵 없는 궤적 예측(Map-Free Trajectory Prediction) 방법을 제안하였다. 이 방법은 훈련 중에 HD 맵과 같은 드라이빙 프라이어를 활용하면서도 인퍼런스 단계에서는 이들에 의존하지 않는 장점을 가진다. 또한, 단계적 디코더를 활용하여 궤적 쿼리를 순차적으로 생성하여 최종 예측을 수행한다.

- **Technical Details**: MFTP는 맵-기반 교사 네트워크로부터 지식을 증류하여 맵-없는 학생 네트워크를 훈련시키는 구조를 가진다. 이 과정에서 계층적 인코더가 여러 수준의 시공간 특징을 추출하여 이들을 궤적 쿼리로 통합한다. 이후 이러한 쿼리는 반복 디코더로 전달되어 미래 궤적 예측에 사용된다.

- **Performance Highlights**: 광범위한 실험 결과, Argoverse 데이터셋에서 맵 없는 설정 하에 기존 기법들보다 뛰어난 성능을 입증하였다. MFTP의 혁신적인 접근 방식으로 인해 자율주행시스템의 안전성을 한층 더 향상시킬 수 있을 것으로 기대된다.



### TSFormer: A Robust Framework for Efficient UHD Image Restoration (https://arxiv.org/abs/2411.10951)
- **What's New**: 이번 논문에서는 UHD (Ultra-High-Definition) 이미지 복원에 필요한 새로운 TSFormer 모델을 제안합니다. 기술적으로, TSFormer는 신뢰할 수 있는 학습과 희소화(sparsification)를 통합하여 복원 품질과 계산 효율성을 동시에 향상시킵니다. 이 모델의 핵심 기술로는 Token의 이동을 최소화하고, 랜덤 매트릭스 이론을 사용하여 Token의 불확실성을 정량화하여 모델의 강건성을 높입니다.

- **Technical Details**: TSFormer는 경량의 신뢰 가능 프레임워크로, 단 하나의 GPU에서 실시간 4K 이미지 복원을 가능하게 설계되었습니다. 주요 기술적 요소는 Min-p sampling 기법으로, 이는 각 레이어의 토큰을 확률 기반의 문턱값으로 선택하여 고신뢰 피처를 동적으로 유지하도록 합니다. 또한, Fast Fourier Transform(FFT)을 사용하여 주파수 도메인에서의 주의(attention) 계산의 복잡성을 줄여, UHD 이미지 처리의 효율성을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과, TSFormer는 최첨단 복원 품질을 달성하면서도 일반화 능력을 향상하고 계산 요구 사항을 줄이는 데 성공했습니다. 이 모델은 다양한 저하된 이미지에 대해 일관되게 잘 작동하며, 신뢰 가능한 방식으로 토큰 필터링 기술을 활용하여 다른 이미지 복원 모델에서도 성능을 유지하면서도 추론 속도를 가속화할 수 있습니다.



### Direct and Explicit 3D Generation from a Single Imag (https://arxiv.org/abs/2411.10947)
Comments:
          3DV 2025, Project page: this https URL

- **What's New**: 이번 논문에서는 멀티 뷰 2D 깊이(depth) 및 RGB 이미지와 3D Gaussian 피처를 활용하여 직접적으로 표면 기하학(surface geometry)과 텍스처(texture)를 생성하는 새로운 프레임워크를 소개합니다. 기존의 높은 계산 비용과 고해상도 출력의 한계를 극복하기 위해 Stable Diffusion 모델을 재구성하여 효율적인 멀티 뷰 생성에 필요한 깊이 브랜치를 U-Net에 추가했습니다. 또한, 픽셀 단위의 멀티 뷰 일관성을 유지하기 위해 에피폴라 주의를(latent-to-pixel decoder) 통합했습니다.

- **Technical Details**: 제안된 방법은 멀티 뷰 깊이, RGB 이미지 및 Gaussian 기능을 3D 공간으로 백 프로젝션(back-projecting)하여 구조화된 3D 표현을 생성합니다. 이러한 표현은 Gaussian splatting 기법을 통해 렌더링 하거나 고품질 메쉬(high-quality meshes)로 추출할 수 있습니다. 특히, 추가적인 novel view synthesis loss를 활용하여 성능을 더욱 개선할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과, 본 방법은 기존 기준선에 비해 기하학 및 텍스처 품질에서 우위를 보이며, 생성 속도가 크게 향상되었습니다. 제안된 시스템은 단일 이미지에서 3D 자산을 생성하는 데 있어 중요한 발전을 나타내며, 고해상도 및 세부 정보 보존 생성에 더 나은 확장성을 제공합니다.



### Anomaly Detection for People with Visual Impairments Using an Egocentric 360-Degree Camera (https://arxiv.org/abs/2411.10945)
Comments:
          WACV2025

- **What's New**: 최근 컴퓨터 비전(Computer Vision) 분야의 발전으로, 시각 장애인을 위한 보조 기술 개발에 대한 관심이 재점화되고 있습니다. 본 논문에서는 360도 착용형 카메라를 통해 시각 장애인의 주변 환경을 감지하여, 비정상적인 상황을 탐지하는 새로운 방법을 제안합니다. 이러한 접근은 현재까지 주로 이미지 이해에 초점을 맞춘 대부분의 연구에서는 다루지 않았던, 실제적인 안전 및 보안 문제를 해결하는 첫 단계로 평가됩니다.

- **Technical Details**: 본 논문에서는 'VIEW360'이라는 새로운 360도 동영상 데이터셋을 소개합니다. 이 데이터셋은 시각 장애인이 직면할 수 있는 비정상적 활동을 담고 있으며, 비정상적 사건의 방향성을 식별할 수 있도록 설계되었습니다. 우리는 'FDPN(Frame and Direction Prediction Network)'라는 새로운 아키텍처를 제안하여, 프레임 단위로 비정상적인 사건을 예측하고 이들의 방향성을 식별하는 방법을 개발하였습니다.

- **Performance Highlights**: 우리의 접근 방식은 VIEW360 데이터셋뿐만 아니라 공개된 UCF-Crime 및 Shanghaitech 데이터셋을 사용하여 평가되었으며, 최신 기술 대비 우수한 성능을 보여주었습니다. VIEW360 데이터셋은 575개의 동영상을 포함하고 있으며, 액터가 시뮬레이션한 실제 시나리오를 기반으로 하여 데이터가 수집되었습니다. 이러한 기법은 시각 장애인에게 실질적인 안전과 보안을 제공할 수 있는 잠재력을 가지고 있습니다.



### Memory-Augmented Multimodal LLMs for Surgical VQA via Self-Contained Inquiry (https://arxiv.org/abs/2411.10937)
- **What's New**: 해당 연구에서는 수술 장면 이해를 향상시키기 위해 SCAN이라는 메모리 증강 프레임워크를 제안하고 있습니다. 기존의 방법들이 여러 객체에 대한 이해력을 제한하고 외부 자원에 의존하는 반면, SCAN은 직접 메모리(Direct Memory, DM)와 간접 메모리(Indirect Memory, IM) 두 가지 메모리 유형을 자동으로 생성하여 수술 맥락을 파악합니다. 이러한 접근 방식은 수술 관련 질문에 대한 응답의 정확성과 신뢰성을 높이는 데 기여합니다.

- **Technical Details**: 수술 시각 질문 응답(Surgical Visual Question Answering, Surgical VQA) 분야에서 메모리 증강된 MLLMs를 활용하여 복잡한 수술 장면을 이해하고 분석하는 데 성공했습니다. S2Can은 사용자가 제시한 질문과 관련된 정보 출처로서 질문-힌트 쌍을 생성하며, 이는 수술 장면의 맥락을 개선하는 데 역할을 합니다. DM은 주어진 질문에 즉각적인 답변을 제공하고, IM은 더 넓은 수술 장면의 이해를 돕습니다.

- **Performance Highlights**: 세 개의 공공 Surgical VQA 데이터셋에서 수행된 실험 결과에 따르면, SCAN은 최신 기술(State-of-the-art, SoTA) 성능을 달성하며 여러 수술 시나리오에서 accuracy와 robustness의 향상을 보여주었습니다. 이러한 성과는 SCAN의 메모리 구성 방식이 효과적임을 강조합니다.



### Iterative Camera-LiDAR Extrinsic Optimization via Surrogate Diffusion (https://arxiv.org/abs/2411.10936)
Comments:
          11 pages, 4 figures, 3 tables

- **What's New**: 이 논문은 카메라와 LiDAR의 데이터 융합을 위한 새로운 툴, 즉 선형 대리 확산 모델(linear surrogate diffusion model)을 제안합니다. 이 모델은 단일 모델 반복(iterative) 접근 방식을 기반으로 하여 서로 다른 보정 방법들을 하나의 파이프라인에서 통합할 수 있도록 합니다. 또한, 기존의 방법들보다 43.7% 빠른 추론 시간을 제공하는 버퍼링 기법도 도입하여 효율성을 높였습니다.

- **Technical Details**: 제안된 방식은 denoiser-agnostic하여 다양한 개별 보정 방법에 적용할 수 있습니다. 카메라와 LiDAR의 보정을 위해 개발된 새로운 네트워크는 포인트 특징 추출을 위한 프로젝션 우선(projection-first) 및 인코딩 우선(encoding-first) 브랜치를 포함하고 있습니다. 이는 기존의 모델들과 결합하거나 독립적으로 사용할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안한 확산 모델이 다른 단일 모델 반복 방법들을 초월하며 멀티 레인지 모델(multi-range models)과 비교하여 경쟁력 있는 성능을 보였습니다. 특히, 제안된 denoiser는 기존의 최첨단 보정 방법보다 24.5% 감소한 회전 오차(rotation error) 성능을 보여주었습니다. 또한, 확산 적용을 통해 회전 오차는 20.4%, 변환 오차(translation error)는 9.6% 감소하는 성과를 올렸습니다.



### Hyperspectral Imaging-Based Grain Quality Assessment With Limited Labelled Data (https://arxiv.org/abs/2411.10924)
Comments:
          10 pages

- **What's New**: 최근 hyperspectral imaging (HSI) 기반의 곡물 품질 평가가 연구의 주목을 받게 되었습니다. 그러나 다른 이미징 방식과 달리 HSI 데이터는 심층 합성곱 신경망 (DCNN) 기반 분류기를 효과적으로 학습시키기에 충분한 레이블 샘플이 부족합니다. 본 논문에서는 HSI와 few-shot learning (FSL) 기법을 결합한 새로운 접근 방식을 제안하여 곡물 품질 평가를 수행합니다.

- **Technical Details**: 기존 곡물 품질 평가 방법은 화학적 및 생물학적 분석을 포함하며, 이 방법들은 침습적이고, 파괴적이며, 시간 소모적이고 비용이 많이 듭니다. HSI는 비침습적이고 실시간 대안을 제공하지만, DCNN을 적용하기 위해서는 많은 레이블 데이터가 필요합니다. FSL을 사용함으로써 적은 레이블 데이터로도 모델이 잘 작동할 수 있도록 하여, 실제 적용에서 빠른 배포가 요구되는 상황에서 실용적인 해결책을 제공합니다.

- **Performance Highlights**: 실험 결과, 매우 제한된 레이블 데이터로 학습했음에도 불구하고, FSL 분류기의 정확도가 기존에 대규모 레이블 데이터베이스로 학습한 완전한 분류기와 유사한 수준인 것으로 나타났습니다. 이 연구는 HSI 기반 품질 평가에 대한 새로운 가능성을 제시하며, 곡물 품질 평가의 신속성과 효율성을 어떻게 향상시킬 수 있는지 보여줍니다.



### Exploiting VLM Localizability and Semantics for Open Vocabulary Action Detection (https://arxiv.org/abs/2411.10922)
Comments:
          WACV 2025 Accepted

- **What's New**: 이 논문은 Open-Vocabulary Action Detection (OVAD) 문제를 다루고 있으며, 이는 사전에 훈련된 행동 카테고리와 관계없이 어떤 행동을 탐지하는 모델을 개발하는 것으로 목표합니다. 기존의 행동 탐지 방법들이 고정된 행동 카테고리에서만 훈련된 것에 반해, OpenMixer라는 새로운 방법을 제안합니다. 이 방법은 큰 비전-언어 모델의 본질적인 의미와 지역화를 활용하여 영상에서의 행동을 탐지하는데 중점을 둡니다.

- **Technical Details**: OpenMixer는 두 가지 OpenMixer 블록(공간적 OpenMixer 블록(S-OMB) 및 시간적 OpenMixer 블록(T-OMB))과 동적으로 융합된 정렬(DFA) 모듈로 구성되어 있습니다. S-OMB는 인물 지역화를 위해 VLM의 지역화 가능성을 사용하고, T-OMB는 VLM의 시각적 의미 특성을 활용하여 시간적 동작을 포착합니다. 이러한 구성 요소들은 훈련된 VLM의 강력한 일반화 능력을 활용하여 OVAD 문제를 해결합니다.

- **Performance Highlights**: 실험을 통해 OVAD 벤치마크를 설정하고, OpenMixer의 성능을 평가하여 이미 보거나 보지 못한 행동을 탐지하는 데 있어 뛰어난 결과를 보여주었습니다. 본 연구는 Open-Vocabulary Action Detection의 중요성과 도전을 제시하고, OpenMixer 모델이 여러 비디오 행동 탐지 데이터 세트에서 강력한 일반화 능력을 발휘함을 실증적으로 입증했습니다.



### Generating Compositional Scenes via Text-to-image RGBA Instance Generation (https://arxiv.org/abs/2411.10913)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문에서는 다층 생성 패러다임(multi-layer generation paradigm)을 제안하여 텍스트-이미지 확산 생성 모델의 객체 속성에 대한 섬세한 제어와 유연성을 향상시키는 새로운 방법을 보여줍니다.

- **Technical Details**: RGBA 이미지를 생성하기 위해 새로운 훈련 패러다임을 도입하여 투명성 정보를 포함한 분리된 장면 요소(instance)들을 생성합니다. 이후에 이러한 요소들을 다층 합성(multi-layer composite) 프로세스를 통해 복합 이미지로 통합하여 현실적인 장면을 구성합니다. 이는 기본적으로 고유한 레이아웃과 속성 조작 기능을 제공합니다.

- **Performance Highlights**: RGBA 확산 모델은 객체 속성과 스타일에 대한 정밀한 제어가 가능하며, 복잡한 장면을 생성하고 조작하는 능력에서 기존 방법들을 초월합니다. 실험 결과, 본 접근 방식은 정교한 프롬프트로부터 강력한 객체 제어를 제공하여 우수한 성능을 나타냅니다.



### Attention-based U-Net Method for Autonomous Lane Detection (https://arxiv.org/abs/2411.10902)
- **What's New**: 이번 연구에서는 도로 차선 인식을 위한 새로운 두 가지 딥러닝 기반 방법이 제안되었습니다. 첫 번째 방법은 Feature Pyramid Network (FPN) 모델을 활용하여 도로 차선을 87.59%의 정확도로 탐지합니다. 두 번째 방법은 U-Net 모델에 attention 레이어를 통합하여 뛰어난 성능을 발휘하며, 98.98%의 정확도를 기록합니다. 이 연구의 결과는 기존 방법과 비교했을 때 특히 우수한 성능을 보여줍니다.

- **Technical Details**: 제안된 알고리즘은 컴퓨터 비전과 딥러닝 기법을 사용하여 도로 차선의 위치를 정확하게 식별합니다. Python 라이브러리인 Albumentations를 사용하여 비디오를 개별 프레임으로 분해하고, 다양한 이미지 증강 방법을 적용합니다. 이미지 전처리 과정은 BGR 색상 공간에서 RGB 색상 공간으로의 변환, 이미지 크기를 224x224 픽셀로 조정하며, 회전과 잡음 추가, 밝기 및 대비 조정과 같은 여러 기술을 포함합니다.

- **Performance Highlights**: 제안된 attention 기반 U-Net 모델은 IoU 기준에서 유사한 모델보다 뛰어난 성능을 보이며, 특히 세분화 작업에서 효과적입니다. 각 모델의 성능은 established criteria를 사용하여 평가되었으며, 두 모델 모두에서 탁월한 성능이 입증되었습니다. 이 연구의 혁신적 접근은 안전하고 효율적인 자율 주행 시스템을 위한 도로 차선 탐지 기술의 발전에 기여할 것입니다.



### Deep BI-RADS Network for Improved Cancer Detection from Mammograms (https://arxiv.org/abs/2411.10894)
- **What's New**: 이 논문에서는 다각적인 관점을 가진 유방암 진단을 위한 최신 딥러닝 모델이 시각적 유방 조영술 데이터에만 국한되어 있는 문제를 해결하기 위해 새로운 다중 모달 접근 방법을 제안합니다. 전통적인 BI-RADS 레전터를 텍스트 기반으로 활용하여 시각적 데이터를 보완하고, 이를 통해 진단 정확도를 더욱 높였습니다. 이 연구는 전문의가 제공하는 추가 정보를 활용하여 딥러닝 모델의 성능 향상을 도모합니다.

- **Technical Details**: 모델은 두 개의 분기(branch)로 구성되어 있으며 각 분기에는 6개의 스택된 동일한 주의 기반(layer) 레이어가 포함되어 있습니다. 입력으로는 BI-RADS 설명자와 유방 조영술 이미지를 다중 해상도로 인코딩하여 처리한 결과를 사용하며, 이는 주의 메커니즘을 통해 서로 결합되어 정보의 일관성을 유지합니다. 이 과정에서 각 레이어는 이전 레이어에서 인코딩된 정보와 다른 분기로부터의 정보 또한 통합하여 특징을 추출합니다.

- **Performance Highlights**: CBIS-DDSM 데이터셋을 기반으로 한 실험 결과, 제안된 다중 모달 주의 기반 접근 방식이 이미지 전용 모델에 비해 모두 높은 성능 개선을 보였으며, AUC 점수는 0.87에 달했습니다. 모든 지표에서 이미지 전용 모델과 비교했을 때 유의미한 성능 향상을 기록하였습니다. 이는 수작업으로 입력된 기능들이 딥러닝 모델에 통합될 때의 잠재력을 보여줍니다.



### ChannelDropBack: Forward-Consistent Stochastic Regularization for Deep Networks (https://arxiv.org/abs/2411.10891)
- **What's New**: 이번 논문에서는 ChannelDropBack이라는 새로운 확률적 정규화 기법을 제안합니다. 이 방법은 네트워크의 역전파 과정에만 무작위성을 추가하여 전방향 패스는 그대로 유지합니다. 기존의 방법들과 달리 ChannelDropBack은 아키텍처 변경 없이도 모든 네트워크 구조에 적용할 수 있습니다. 이로 인해 학습과 추론 단계에서 동일한 네트워크 구조가 사용되어 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: ChannelDropBack는 훈련 과정에서 매 반복마다 네트워크 내의 특정 레이어를 무작위로 선택하고, 선택된 레이어의 일부 채널 또는 행만 업데이트합니다. 레이어 선택은 사전 정의된 확률 분포에 따라 이루어지며, 기본적으로 균등 분포가 사용됩니다. 선택된 채널 또는 행에 대해서만 가중치 업데이트가 적용되며, 이로 인해 나머지 채널 또는 행은 변경되지 않습니다. 이는 deep convolutional networks의 역전파 과정에서 채널의 무작위 선택을 통해 효과적인 업데이트를 가능하게 합니다.

- **Performance Highlights**: ChannelDropBack을 사용한 실험 결과는 ImageNet, CIFAR-100, CIFAR-10과 같은 인기 데이터셋에서 전통적인 훈련 방법들보다 개선된 정확도와 강건성을 나타냅니다. 다양한 아키텍처(예: ResNet, EfficientNet, ViT-B)에서 일관되게 높은 성능을 보여주며, 이는 이 방법이 여러 데이터셋과 네트워크 아키텍처에 걸쳐 일반화 가능한 정규화 기법임을 입증합니다. 전반적으로, ChannelDropBack은 확률적 정규화의 새로운 접근 방식으로서 deep learning의 성능을 향상시키는 잠재력을 가지고 있습니다.



### MetricGold: Leveraging Text-To-Image Latent Diffusion Models for Metric Depth Estimation (https://arxiv.org/abs/2411.10886)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2312.02145 by other authors

- **What's New**: 이 논문에서는 MetricGold라는 새로운 방법을 제안하여 단일 이미지에서 메트릭 깊이(metric depth)를 추정하는 문제를 해결하고자 합니다. 이는 generative diffusion model의 강력한 prior를 활용하여 메트릭 깊이 예측 성능을 향상시키는 접근입니다. MetricGold는 안정적인 생성 모델인 Stable Diffusion을 기반으로 하며, denoising U-Net만 조정하여 깊이 추정에 적합하도록 하였습니다. 이를 통해 실제 깊이 맵을 전혀 보지 않고도 실험에서 최첨단 성능을 달성했습니다.

- **Technical Details**: MetricGold는 RGB 이미지를 깊이 맵으로 변환하는 generative 작업으로 메트릭 깊이 추정을 재구성합니다. 이 과정에서 모델은 조건부 분포 D(d|x)를 모델링하며, 이는 입력 이미지 x에 대한 깊이 d를 예측합니다. 기존의 DMD와는 달리, MetricGold는 latent space에서 denoising을 수행하여 더욱 계산 효율적이며 빠릅니다. 모델은 Hypersim과 Virtual KITTI와 같은 포토 리얼리스틱 합성 데이터로 학습되었습니다.

- **Performance Highlights**: MetricGold는 한 대의 RTX 3090 그래픽 카드에서 2일 이내에 효율적으로 학습할 수 있으며, 다양한 데이터셋에서 강력한 일반화를 보여줍니다. 실험 결과, 기존 접근 방법보다 선명하고 높은 품질의 메트릭 깊이 추정 결과를 생성해냅니다. 또한, 이 모델은 실세계 데이터에서 뚜렷한 일반화 능력을 갖추고 있으며, sensor 기반 데이터의 편향을 피하여 보다 정확한 예측을 가능하게 합니다.



### FIAS: Feature Imbalance-Aware Medical Image Segmentation with Dynamic Fusion and Mixing Attention (https://arxiv.org/abs/2411.10881)
Comments:
          5 pages, 2 figures, 3 tables

- **What's New**: 이 논문에서는 CNN과 transformer의 기능을 결합하여 의료 영상 분할 작업에서 뛰어난 성과를 거두는 Feature Imbalance-Aware Segmentation (FIAS) 네트워크를 제안합니다. FIAS는 dual-path encoder를 사용하여 긴 거리의 전역 특성과 미세한 지역 세부 정보를 효과적으로 캡처하며, Context-Aware Fusion (CAF) 블록을 통해 특성의 균형을 유지합니다. 또한, Mixing Attention (MixAtt) 디코더를 구현하여 더 나은 세분화 정확도를 얻고 있습니다.

- **Technical Details**: FIAS 프레임워크는 pre-trained DilateFormer와 Depthwise Multi-Kernel (DMK) convolution을 갖춘 dual-branches encoder로 구성됩니다. CAF 모듈은 두 인코더에서 추출된 다중 스케일 특성을 동적으로 융합하여 균형 잡힌 표현을 보장합니다. DilateFormer는 글로벌 및 로컬 의미 정보를 효율적으로 모델링하는 계층적 transformer로, multi-scale dilated attention (MSDA)와 multi-head self-attention 구조를 채택하고 있습니다.

- **Performance Highlights**: 실험 결과, FIAS는 Synapse와 ACDC 데이터셋에서 기존 방법보다 우수한 성능을 입증했습니다. 특히, FIAS는 작은 세부사항과 대규모 종속성을 모두 포착할 수 있음을 보여주며, 복잡한 의료 영상 분할 작업에서도 높은 경쟁력을 갖추고 있습니다.



### ViBe: A Text-to-Video Benchmark for Evaluating Hallucination in Large Multimodal Models (https://arxiv.org/abs/2411.10867)
- **What's New**: 이 논문은 ViBe라는 새로운 대규모 Text-to-Video Benchmark를 소개하여 T2V 모델의 환각 콘텐츠를 평가합니다. 환각은 비디오 생성 시 생성된 내용이 입력 텍스트와 일치하지 않거나 왜곡되는 경우를 가리킵니다. 이 연구는 3,782개의 환각 비디오로 구성된 데이터세트를 수집하여 '소실 주제', '수치 변동성', '시간 왜곡', '생략 오류', '물리적 불일치'와 같은 다섯 가지 주요 환각 유형을 식별했습니다.

- **Technical Details**: ViBe 데이터세트는 MS-COCO 데이터세트에서 700개의 랜덤 캡션을 선택해 생성한 것입니다. 연구에 사용된 모델에는 MS1.7B, MagicTime, AnimateDiff-MotionAdapter, Zeroscope V2 XL 등 10개의 오픈소스 T2V 모델이 포함되었습니다. 분석을 통해 우리는 다양한 환각 유형의 존재 여부와 빈도를 살펴보면서 각 비디오를 조사했습니다.

- **Performance Highlights**: 최초의 대규모 환각 비디오 데이터세트를 제공함으로써 T2V 모델의 신뢰성 및 환각 탐지 개선에 기여하고자 합니다. 실험 결과, TimeSFormer + CNN 조합이 가장 좋은 성능을 보였으며, 0.345의 정확도와 0.342의 F1 점수를 기록했습니다. 이 벤치마크는 T2V 모델의 정확성과 신뢰성 향상을 위한 기초를 제공합니다.



### Improvement in Facial Emotion Recognition using Synthetic Data Generated by Diffusion Mod (https://arxiv.org/abs/2411.10863)
Comments:
          5 pages, 4 tables, 4 figures, ICASSP 2025

- **What's New**: 이 논문은 Facial Emotion Recognition (FER) 분야에서 발생하는 데이터 불균형 문제를 해결하기 위해 합성 데이터 증강(synthetic data augmentation)을 도입하여 ResEmoteNet 모델을 개선하는 방법을 제시합니다. 합성 데이터 생성에는 Stable Diffusion 2 및 3 모델을 사용하여 FER2013 및 RAF-DB 데이터셋의 훈련 세트를 증가시키는 방식입니다. 결과적으로 FER2013에서 96.47% 및 RAF-DB에서 99.23%의 정확도를 달성하며, 이는 기존 성능에 비해 큰 향상을 보여줍니다.

- **Technical Details**: ResEmoteNet은 Residual Connections와 Squeeze-and-Excitation 메커니즘을 통합한 딥 러닝 아키텍처로, 이로 인해 특징 추출 성능이 향상됩니다. 데이터셋의 편중된 감정을 해결하기 위해, 저자는 Stable Diffusion을 활용하여 텍스트 프롬프트에서 합성 이미지를 생성하는 방식의 데이터 증강 방법을 제안합니다. 이 접근법은 감정 표현에 대한 모델의 일반화를 향상시키는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 저자들은 합성 데이터 증강 방식으로 FER2013에서 16.68%, RAF-DB에서 4.47%의 성능 향상을 달성하는 것으로 나타났습니다. 이 연구는 FER 모델을 강화하고, 합성 데이터의 유효성을 강조하며, 발전된 생성 모델이 FER 연구에서 가지는 잠재력을 보여줍니다. 또한, ResEmoteNet의 소스 코드는 제공되어 연구자들이 이를 재사용할 수 있도록 지원합니다.



### Large Vision-Language Models for Remote Sensing Visual Question Answering (https://arxiv.org/abs/2411.10857)
- **What's New**: 이 논문에서는 원거리 감지 시각 질문 응답(Remote Sensing Visual Question Answering, RSVQA)을 위한 새로운 방법론을 소개합니다. 기존의 전통적인 접근 방식은 별도의 시각 특성 추출기와 언어 처리 모델에 의존하여 계산적으로 부담이 크고 개방형 질문을 처리하기에 한계가 있었습니다. 본 연구는 생성형 대형 시각-언어 모델(Large Vision-Language Model, LVLM)을 활용하여 RSVQA 과정을 간소화하는 두 단계 훈련 전략을 제안합니다.

- **Technical Details**: 모델의 훈련 과정은 도메인 적응 사전 훈련과 프롬프트 기반 미세 조정으로 구성됩니다. 이러한 방법을 활용함으로써, LVLM은 시각적 정보와 텍스트 입력에 조건화되어 자연어 답변을 생성할 수 있으며, 미리 정의된 답변 카테고리를 요구하지 않습니다. RSVQAxBEN 데이터셋을 통해 모델을 평가하며, 기존 최고 성능 벤치마크에 비해 우수한 성능을 달성하고 있습니다.

- **Performance Highlights**: 평가 결과는 우리 방법이 정확성, 관련성, 유창성 측면에서 더 정확한 답변을 생성한다는 것을 보여줍니다. 또한, 프롬프트 기반 미세 조정 전략이 향상된 일반화 능력을 보이며 새로운 원거리 감지 데이터에 대해서도 안정성을 가지고 대응할 수 있습니다. 요약하자면, 우리의 방법은 LVLM의 힘을 활용하여 RSVQA 응답의 정확성과 품질을 크게 개선합니다.



### NeuroNURBS: Learning Efficient Surface Representations for 3D Solids (https://arxiv.org/abs/2411.10848)
- **What's New**: 이번 논문에서는 Boundary Representation (B-Rep)에서의 표면을 나타내기 위한 새로운 접근 방식인 NeuroNURBS를 제안합니다. 기존의 UV-grid 방법은 표면 표현의 효율성이 낮고, 정확성 및 규칙성이 부족한 경우가 많습니다. NeuroNURBS는 NURBS (Non-Uniform Rational B-Splines) 매개변수를 직접 인코딩하는 방법으로, 기존 방법보다 더 효율적이라는 장점을 가지고 있습니다.

- **Technical Details**: NeuroNURBS는 NURBS 매개변수의 효과적인 표현을 학습하기 위한 파이프라인으로, 전처리 및 오토인코더(autoencoder)를 포함합니다. 이 방법은 NURBS 매개변수의 다양한 크기를 처리하기 위해 transformer 아키텍처를 사용하여 NURBS 매개변수를 저차원 feature space로 인코딩합니다. 이를 통해 최종적으로 NURBS 특성(NURBS features)을 추출하고, 이는 나아가 B-Rep 생성 및 세분화와 같은 다운스트림 작업에 활용될 수 있습니다.

- **Performance Highlights**: 실험 결과, NeuroNURBS는 UV-grid 방법과 비교하여 3D 도형 재구성에서 메모리 효율성이 79.9% 향상되었고, GPU 소비량은 86.7% 감소했습니다. B-Rep 생성의 경우, NeuroNURBS를 적용하여 FID (Fréchet Inception Distance)가 30.04에서 27.24로 향상되었습니다. 또한, 세분화 작업에서는 새로운 모델인 NURBS-GAT가 99.65%의 정확도를 달성하였습니다.



### Automatic Discovery and Assessment of Interpretable Systematic Errors in Semantic Segmentation (https://arxiv.org/abs/2411.10845)
Comments:
          7 pages main paper (without references), total 13 pages & 9 figures

- **What's New**: 이 논문은 세분화 모델(segmentation model)에서 시스템적 오류(systematic errors)를 발견하기 위한 새로운 방법을 제안합니다. 특히 자율 주행(Autonomous Driving)과 같은 중요한 응용 분야에서 이러한 오류를 탐지하고 해석하는 것이 중요합니다. 무작위로 레이블이 없는 데이터에서 자동으로 이러한 실패를 발견하고 개입을 위한 해석 가능한 의미적 하위 그룹을 형성하는 것이 본 연구의 주요 도전과제입니다.

- **Technical Details**: 이 연구는 멀티모달 기초 모델(multimodal foundation models)을 활용하여 오류를 검색하고 오류의 개념적 연결을 사용하여 이러한 오류의 시스템적 본질을 연구합니다. 기존의 세분화 모델인 UperNet ConvNeXt와 UperNet Swin에 대해 정성적 및 정량적으로 평가를 수행하였으며, 이러한 모델에서 일관된 시스템적 오류를 발견하는 데 효과적임을 보여주었습니다. 이 방법은 레이블이 없는 대규모 데이터에 대해 적용 가능하며, 인간 해석 가능성을 높이는데 초점을 맞추고 있습니다.

- **Performance Highlights**: 우리의 결과는 제안한 프레임워크가 주어진 세분화 모델에 대해 개념 수준에서 시스템적 오류를 발견하고 해석하는 데 효과적임을 입증합니다. 본 연구는 세분화 모델에서의 오류 분석과 개입이 지금까지 충분히 탐구되지 않았던 새로운 경로를 열고 있으며, 안전 공학 관점에서도 중요한 평가 단계를 제공합니다. 이를 통해, 자율 주행 차량에서의 모델 성능 개선을 위한 기초를 다지는 데 기여할 것으로 보입니다.



### AnimateAnything: Consistent and Controllable Animation for Video Generation (https://arxiv.org/abs/2411.10836)
- **What's New**: 이번 논문에서는 AnimateAnything라는 통합된 제어 가능한 비디오 생성 접근 방식을 제안한다. 이 방법은 카메라 경로, 텍스트 프롬프트, 사용자 동작 주석과 같은 다양한 조건에서 비디오 조작을 정밀하고 일관되게 수행할 수 있도록 설계되었다. 다중 스케일 제어 기능 융합 네트워크를 통해 모든 제어 정보를 프레임 별 광학 흐름으로 변환하여 최종 비디오 생성을 안내한다.

- **Technical Details**: 논문에서는 여러 동작 제어 신호를 통합된 광학 흐름으로 변환하고 이를 통해 비디오 생성을 유도하는 두 단계 비디오 생성 방법을 제안한다. 첫 번째 단계에서는 다양한 제어 신호를 일관된 광학 흐름으로 변환하고, 두 번째 단계에서는 이를 텍스트 컨트롤과 동기화하여 고품질 비디오 생성을 목표로 한다. 또한 주파수 도메인에서의 특징 변환 및 스펙트럴 주의 메커니즘을 도입하여 비디오의 안정성을 높인다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최신 방법들과 비교하여 정량적 및 정성적으로 우수성을 입증했다. 특히, 대규모 동작 생성에서의 안정성과 품질을 보여주며, 비디오의 주파수 도메인 일관성을 보장하여 왜곡 문제를 최소화하는 데 효과적이다. 이 연구는 향후 영화 제작 및 가상 현실 등 다양한 응용 분야에서의 사용 가능성을 시사한다.



### ARM: Appearance Reconstruction Model for Relightable 3D Generation (https://arxiv.org/abs/2411.10825)
- **What's New**: 이번 논문에서 소개하는 ARM(Aspect-Relation Modeling) 방법은 희소 뷰 이미지에서 고품질 3D 메쉬와 사실적인 외관을 재구성하는 혁신적인 기법입니다. ARM은 기하 구조와 외관을 분리하여 UV 텍스처 공간 내에서 외관을 처리하는 방식으로, 기존 방법에 비해 텍스처 품질을 명시적으로 향상시키는 것이 특징입니다. 또한, ARM은 입력 이미지에서 재료(material)와 조명(illumination)의 모호성을 해결하기 위해 의미적 외관 정보를 인코딩하는 재료 사전(material prior)을 도입하여 외관 분해의 견고성을 강화합니다.

- **Technical Details**: ARM은 Large Reconstruction Models(LRMs)에 기반을 둔 3D 메쉬 재구성 프레임워크로, 트리플레인(triplane)을 3D 표현으로 사용합니다. 기존 방법들이 텍스처 해상도 제한으로 인해 재구성된 텍스처가 흐릿하게 나타나는 점을 개선하기 위해, ARM은 멀티 뷰 측정값을 텍스처 맵에 명시적으로 되돌려 보내고, UV 공간 모듈을 사용하여 처리합니다. 이를 통해 UV 텍스처 공간에서의 색상 변화를 직접 표현하여 텍스처 품질을 극대화합니다.

- **Performance Highlights**: 테스트 결과, ARM은 정량적 및 정성적으로 기존의 이미지-투-3D 방법들을 초월하는 성능을 보였습니다. 특히, 고품질 텍스처와 사실적인 외관을 생성하는 능력을 입증하였으며, 단 두 개의 H100 GPU에서 훈련할 수 있을 정도로 효율성을 갖추고 있습니다. 이러한 성과는 메타버스, 게임, 전자상거래와 같은 다양한 응용 분야에서의 활용 가능성을 더욱 높입니다.



### DEAL: Decoupled Classifier with Adaptive Linear Modulation for Group Robust Early Diagnosis of MCI to AD Conversion (https://arxiv.org/abs/2411.10814)
Comments:
          Under Review

- **What's New**: 이번 연구는 MRI 이미지를 사용하여 경도의 인지 장애 (MCI)에서 알츠하이머병 (AD)으로의 전환 진단에 있어 집단 강건성 (group robustness)을 최초로 종합적으로 조사합니다. 기존 연구들은 특정 집단에서의 성과 저하를 간과해왔습니다. 따라서 MCI에서 AD로의 전환을 예측하기 위한 새로운 방법인 DEAL (DEcoupled classifier with Adaptive Linear modulation)을 제안하고, 이는 그룹 간의 결정 경계를 개선하여 성능을 향상시킵니다.

- **Technical Details**: DEAL은 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 연령 및 인지 지표와 같은 임상 및 인구 통계적 특성을 활용한 선형 조정, (2) 연령에 따라 그룹화된 개별 분류 헤드를 가진 탈결합 분류기 (decoupled classifier). 이 방법은 기존 모델들이 보이는 집단 간 성과 저하 문제를 해결하는 데 중점을 두며, 다양한 아키텍처에서 효과를 입증했습니다.

- **Performance Highlights**: DEAL 방법은 GBA (group-balanced accuracy)와 WGA (worst-group accuracy)를 개선하는 데 성공하였습니다. 특히, 단순한 테이블 데이터와 sMRI 특징의 결합으로는 서로 간의 보완적인 정보를 포착하지 못했으나, DEAL은 효과적으로 이를 활용하여 성능 향상에 기여했습니다. 실험 결과, 서로 다른 아키텍처에서도 DEAL이 일관되게 성능을 개선하는 것으로 나타났습니다.



### Multi-Stage Vision Token Dropping: Towards Efficient Multimodal Large Language Mod (https://arxiv.org/abs/2411.10803)
Comments:
          8 pages, 4figures

- **What's New**: 이 논문에서는 다단계 토큰 드롭핑(Multi-stage Token Dropping, MustDrop) 기법을 제안하여 시각적 토큰의 중요성을 전체 생애 주기(vision encoding, prefilling, decoding 단계)에서 평가합니다. MustDrop은 시각적 정보를 효과적으로 처리하여 불필요한 토큰을 제거하는 동시에 성능과 효율 간의 최적의 균형을 구현합니다. 이를 통해 LLaVA 모델에서 약 88.5%의 FLOPs 감소와 92.2%의 압축 비율을 달성하면서도 유사한 정확도를 유지합니다.

- **Technical Details**: MustDrop은 세 가지 단계(vision encoding, prefilling, decoding)에서 다양한 전략을 활용하여 토큰을 처리합니다. 비전 인코딩 단계에서는 유사한 인접 토큰을 통합하고, 중요한 비전 토큰 집합을 설정하여 후속 단계에서 삭제되지 않도록 합니다. 누적된 텍스트 의미에 따라 비전 토큰을 압축하며, 디코딩 스테이지에서는 KV 캐시의 크기를 줄이는 캐시 정책을 적용합니다.

- **Performance Highlights**: 실험 결과, MustDrop은 LLaVA와 같은 다중 모달 대형 언어 모델의 효율성을 개선하며, 기존의 SparseVLM보다도 2.1%에서 6.6%의 성능 향상을 기록했습니다. MustDrop은 추가적인 훈련 없이도 플러그 앤 플레이 모듈로 작동하여, 시각적 토큰의 불필요한 중복을 효과적으로 줄이는데 기여합니다.



### Test-time Conditional Text-to-Image Synthesis Using Diffusion Models (https://arxiv.org/abs/2411.10800)
- **What's New**: 본 논문에서는 조건부 텍스트-이미지 합성을 위한 새로운 방법, TINTIN(Test-time Conditional Text-to-Image Synthesis using Diffusion Models)을 제안합니다. 이 접근법은 훈련이 필요 없는 테스트 타임에서 조건부 요소인 색상 팔레트와 에지 맵을 기반으로 이미지 생성을 조절합니다. 기존의 방법들과 달리, 저희 방법은 훈련 없이도 모델 출력에 변화를 줄 수 있는 유연한 방식을 제공합니다.

- **Technical Details**: 이 연구에서는 디퓨전 모델의 노이즈 예측을 에너지 기반 모델의 기울기로 해석하여, 결과적으로 원하는 상태로의 중간 표현을 조작하는 방법을 채택합니다. 또한, 색상 분포 일치를 위해 새로운 손실 함수 LDᵢₛ를 도입하고, 색상 및 에지 조정을 위한 반복 샘플링 전략을 제안합니다. 이 과정은 최적의 디노이징 단계에서 입력 팔레트의 색상 분포와 생성된 이미지의 색상 분포를 매칭하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 다양한 텍스트 프롬프트, 색상 팔레트 및 에지 맵을 사용하여 현재 최첨단 방법들과 비교했을 때 정성적, 정량적으로 상당한 성능을 보여주었습니다. 특히, 본 접근법은 훈련 의존 없이 색상 팔레트에서 조건부 모델 출력을 생성하는 첫 번째 방식으로 주목받고 있습니다. 연구를 통해, 기존 방법보다 더욱 다양한 조건에서 효과적으로 작동함을 입증하였습니다.



### Going Beyond Conventional OOD Detection (https://arxiv.org/abs/2411.10794)
- **What's New**: 이 논문에서는 OOD(Out-of-Distribution) 샘플을 효과적으로 탐지하기 위해 스퓨리어스(Spurious), 파인 그레인(Fine-Grained), 그리고 전통적인 OOD 탐지 방법을 통합한 접근 방식인 ASCOOD를 제안합니다.

- **Technical Details**: ASCOOD는 두 가지 주요 구성 요소로 구성되어 있습니다: 1. 이상값(Outlier) 합성 파이프라인과 2. 가상 이상값(Virtual Outlier) 노출 훈련 파이프라인. 이 모델은 ID(in-Distribution) 데이터에서 불변 특징(Invariant Features)과 환경 특징(Environmental Features)을 분리하여 가상 이상값을 합성합니다. 그리고 가상 이상값을 통한 불확실성 예측을 약화시키기 위해 표준화된 특징 표현을 사용합니다.

- **Performance Highlights**: 여섯 개의 데이터셋에 대한 광범위한 실험을 통해 ASCOOD는 스퓨리어스 및 파인 그레인 OOD 탐지에서 뛰어난 성능을 발휘하는 것을 입증했습니다.



### Anatomy-Guided Radiology Report Generation with Pathology-Aware Regional Prompts (https://arxiv.org/abs/2411.10789)
- **What's New**: 이 논문에서는 의료 영상 보고에서 발생하는 클리니컬한 문제를 해결하기 위해 병리 인식 지역 지시어(pathology-aware regional prompts)를 도입하는 혁신적인 접근 방식을 제안합니다. 이 방법은 해부학적(anatomical) 및 병리(pathological) 정보를 다양한 규모에서 통합하여 생성된 보고서의 정확성과 임상 관련성을 크게 향상시킵니다. 또한, 다양한 해부학적 부위에서 특징을 추출하는 해부학적 영역 탐지기(anatomical region detector)와 여러 병리를 식별하는 다중 레이블 병변 탐지기를 개발하여 진단 프로세스를 모방합니다.

- **Technical Details**: 기존의 방사선 보고 작성 시스템들은 고정 크기 패치 수준의 이미지 특징에 의존하며, 병리 정보를 충분히 통합하지 않아 문제가 발생합니다. 이를 해결하기 위해, 저자들은 CXR 이미지의 29개 특정 영역에서 해부학적 시각 특징을 추출하는 해부학적 영역 탐지기와 단일 경계 상자 내 여러 병리를 식별하는 다중 레이블 병변 탐지기를 소개합니다. 이러한 접근은 보고서 디코더에 세밀한 진단 결과를 제공하여 정확한 보고서 생성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 MIMIC-CXR-JPG와 Chest ImaGenome 데이터셋에서 이전의 최첨단 방법들보다 우수한 성능을 보였습니다. 특히, 임상적 정확성(clinical accuracy)과 효율성을 확인한 전문가들의 공식 평가에 의해 의료 영상 실무를 향상시킬 수 있는 잠재력을 지니고 있음을 보여주었습니다. 이는 기존의 방법들이 가지는 제한점을 극복하면서도 더 나은 진단 결과를 생성하는 데 기여합니다.



### C-DiffSET: Leveraging Latent Diffusion for SAR-to-EO Image Translation with Confidence-Guided Reliable Object Generation (https://arxiv.org/abs/2411.10788)
Comments:
          Please visit our project page this https URL

- **What's New**: 이 논문에서는 SAR(합성개구레이더) 이미지를 EO(전기광학) 이미지로 번역하기 위한 새로운 프레임워크인 Confidence Diffusion for SAR-to-EO Translation(C-DiffSET)를 소개합니다. 기존의 한정된 SAR-EO 데이터셋으로부터 학습하여 발생할 수 있는 과적합 문제를 해결하기 위해, 자연 이미지로 미리 학습된 Latent Diffusion Model(LDM)을 활용하여 EO 도메인에 효과적으로 적응할 수 있도록 설계되었습니다.

- **Technical Details**: C-DiffSET는 SAR 이미지와 EO 이미지를 동일한 잠재 공간(latent space)으로 정렬할 수 있는 사전 훈련된 VAE(변분 오토인코더) 인코더를 사용합니다. 이를 통해 SAR 입력의 다양한 노이즈 레벨에도 불구하고 잠재 공간에서의 효과적인 정보 전달이 가능합니다. 또한, 픽셀 단위의 충실도를 높이기 위해 C-Diff 손실(confidence-guided diffusion loss)을 도입하여 시간적 불일치를 완화함으로써 구조적 정확성을 향상시킵니다.

- **Performance Highlights**: C-DiffSET는 여러 데이터셋에서 최첨단(SOTA) 결과를 달성하며, 최근의 이미지-이미지 번역 방법들과 SET 방법들을 큰 차이로 초월하는 성능을 보여줍니다. 이를 통해 SAR 이미지의 해석 가능성을 높이고 다양한 환경에서 SAR 데이터를 효과적으로 활용할 수 있는 가능성을 제시하고 있습니다.



### Bag of Design Choices for Inference of High-Resolution Masked Generative Transformer (https://arxiv.org/abs/2411.10781)
- **What's New**: 이 논문은 Text-to-image diffusion models (DM)와 autoregressive models (ARM) 간의 차이를 극복하기 위해 masked generative Transformer (MGT)를 이용하여 시각 생성과 언어 생성을 통합할 수 있는 경로를 제시합니다. 특히 MGT의 추론(inference) 단계에 대한 포괄적인 분석이 부족한 현실을 해결하기 위해, DM 기반의 추론 기법을 완전히 재설계합니다.

- **Technical Details**: MGT는 이미지 토큰을 랜덤으로 마스킹하여 예측하며, DM의 효율성과 ARM의 이산 토큰 특성을 결합합니다. 특정 알고리즘(DPM-Solver, TomeSD, Z-Sampling)을 MGT에 적용할 때 NFE(기능 평가 수)를 줄이고 성능 향상을 위한 수정이 필요함을 강조합니다. 또한, Gaussian noise regularization 및 differential sampling과 같은 저렴하면서도 효과적인 분포 수정 알고리즘이 제안됩니다.

- **Performance Highlights**: 제안된 전략들은 HPS v2 벤치마크에서 EyeReward, HPS v2, PickScore 및 AES와 같은 다양한 메트릭을 사용하여 시각 품질을 크게 향상시키는 것을 확인했습니다. 실험 결과 Meissonic은 Challengebench에 대한 성능 향상이 SD XL에 비해 더 월등함을 보여주었습니다.



### LTCXNet: Advancing Chest X-Ray Analysis with Solutions for Long-Tailed Multi-Label Classification and Fairness Challenges (https://arxiv.org/abs/2411.10746)
Comments:
          8 pages, 5 figures

- **What's New**: 이번 연구에서는 Chest X-ray (CXR) 데이터를 다루기 위해 새롭게 Pruned MIMIC-CXR-LT 데이터셋을 구축했습니다. 이 데이터셋은 긴 꼬리(long-tailed) 분포와 다중 라벨(multi-label) 상황을 반영하기 위해 설계되었습니다. LTCXNet이라는 새로운 프레임워크를 도입하였으며, ConvNeXt 모델과 ML-Decoder, 전략적 데이터 증강(data augmentation) 등이 통합되었습니다.

- **Technical Details**: LTCXNet은 긴 꼬리 분포의 다중 라벨 분류 문제를 해결하기 위해 ConvNeXt 계열의 아키텍처를 채택했습니다. 모델은 ‘Head’, ‘Tail’, ‘All’ 세 가지 고유한 모델로 훈련되어 각 모델이 특정 라벨 세트에 중점을 두고 있으며, 데이터셋은 훈련, 검증 및 테스트 세트로 나뉘어 182,380, 20,360, 54,268 이미지로 구성됩니다. 특히 ML-Decoder는 기존 transformer의 self-attention 메커니즘을 제거하여 효율성을 높였습니다.

- **Performance Highlights**: LTCXNet은 CXR 해석의 모든 클래스에서 성능을 향상시켰으며, 특히 ‘Pneumoperitoneum’과 ‘Pneumomediastinum’과 같은 드문 클래스에서 각각 79% 및 48% 향상을 보여주었습니다. 요약된 연구 결과들은 공정성을 평가하고 정확도를 높이기 위한 다양한 접근 방식을 모색하며, 모델의 공정성을 보장하여 다양한 인구 집단에 대한 정확도를 균일하게 유지하는 데 기여하고 있습니다.



### TDSM:Triplet Diffusion for Skeleton-Text Matching in Zero-Shot Action Recognition (https://arxiv.org/abs/2411.10745)
Comments:
          Please visit our project page at this https URL

- **What's New**: 이 논문에서는 스켈레톤 입력을 위한 제로샷 학습(zero-shot learning) 기반의 행동 인식을 제안합니다. 제로샷 스켈레톤 기반 행동 인식에서, 스켈레톤 특징과 행동 레이블의 텍스트 특징을 정렬하는 것이 필수적이며, 이를 통해 보지 못한 행동을 정확하게 예측할 수 있습니다. 기존 방법들은 스켈레톤과 텍스트 잠재 공간(latent space) 간의 직접적인 정렬에 초점을 맞추었지만, 이러한 공간 간의 차이가 학습의 일반화(generalization)를 방해했습니다.

- **Technical Details**: 제안된 방법, 즉 Tripllet Diffusion for Skeleton-Text Matching (TDSM)은 역 확산(reverse diffusion) 훈련 과정을 통해 스켈레톤 특징과 텍스트 프롬프트를 정렬합니다. 이 방법은 텍스트 프롬프트를 일체화된 스켈레톤-텍스트 잠재 공간에 백터화하여 더 강력한 매칭을 수행합니다. 또한 새로운 triplet diffusion (TD) 손실(loss)을 도입하여 올바른 스켈레톤-텍스트 쌍의 정렬을 강화하는 동시에 잘못된 쌍은 분리하여 모델의 판별력을 향상시킵니다.

- **Performance Highlights**: 우리의 TDSM은 최근 상태의 방법들(SOTA)보다 2.36%-점에서 13.05%-점까지 큰 차이를 보이며 매우 뛰어난 성능을 보여줍니다. 이러한 성능은 효과적인 스켈레톤-텍스트 매칭을 통해 제로샷 설정에서도 우수한 정확도가 보장됨을 의미합니다. 결과적으로, 우리의 접근 방식은 감시, 로보틱스 및 인간-컴퓨터 상호작용과 같은 다양한 응용 프로그램에 있어 매우 큰 가능성을 지니고 있습니다.



### It Takes Two: Accurate Gait Recognition in the Wild via Cross-granularity Alignmen (https://arxiv.org/abs/2411.10742)
Comments:
          12 pages, 9 figures; Accepted by ACM MM 2024

- **What's New**: 이 논문에서는 실루엣과 파싱의 장점을 활용하여 두 가지 외관 표현을 통합한 XGait라는 새로운 보행 인식 방법을 제안합니다. XGait는 서로 다른 세분화 수준의 보행 표현을 활용하도록 설계되어 있으며, 이는 실제 환경에서의 보행 인식 성능을 개선하는 데 중점을 둡니다. 이를 통해 연구자들은 고급 정보 엔트로피와 높은 품질의 세분화를 동시에 활용할 수 있습니다.

- **Technical Details**: XGait의 핵심 구성 요소는 두 개의 백본 인코더와 두 개의 크로스 그레뉼러리 모듈(GCM, PCM)로 이루어져 있습니다. GCM은 실루엣으로부터 얻은 전역 특징을 이용하여 파싱 특징의 품질을 향상시키며, PCM은 실루엣과 파싱 특징 사이의 인체 부위의 동적 특징을 정렬하는 역할을 합니다. 또한, 학습 가능한 분할 메커니즘을 도입하여 세부 레벨에서의 정렬 과정을 효과적으로 안내합니다.

- **Performance Highlights**: XGait는 Gait3D 데이터셋에서 80.5%의 Rank-1 정확도를, CCPG 데이터셋에서 88.3%의 Rank-1 정확도를 기록하여 기존 최첨단 성능을 초월했습니다. 이 실험 결과는 XGait가 실루엣과 파싱의 통합을 통해 안정적이고 강력한 성능을 발휘함을 보여줍니다. 특히, occlusion(차차) 및 의류 변화와 같은 어려운 조건에서도 학습된 특성의 견고함이 입증되었습니다.



### EVT: Efficient View Transformation for Multi-Modal 3D Object Detection (https://arxiv.org/abs/2411.10715)
- **What's New**: 본 논문은 Multi-modal 센서 융합을 통한 3D 객체 탐지를 위한 새로운 접근법인 Efficient View Transformation (EVT)을 제안합니다. EVT는 깊이 추정기나 transformer 인코더를 사용하는 기존 방법들과는 달리, LiDAR 데이터의 지침을 활용하여 3D 샘플링 포인트 및 적응형 커넬을 생성합니다. 이러한 혁신은 BEV( Bird’s-Eye-View) 표현의 정확도와 효율성을 동시에 높입니다.

- **Technical Details**: EVT는 두 가지 주요 모듈로 구성됩니다: Adaptive Sampling and Adaptive Projection (ASAP)과 개선된 transformer 기반 탐지 프레임워크입니다. ASAP은 이미지의 ROI에서 기능을 BEV 공간으로 변환하는 3D 포인트를 생성하며, AP 모듈은 적응형 커넬을 사용하여 BEV 기능을 정제하는 역할을 합니다. 이 improved transformer-based detection framework는 그룹 단위 쿼리 초기화 및 쿼리 업데이트 프레임워크를 포함하여 과거 방법들과 비교하여 성능을 크게 향상시킵니다.

- **Performance Highlights**: EVT는 nuScenes 테스트 세트에서 75.3% NDS 및 72.6% mAP의 성능을 달성하며, 과거 최첨단 방법들을 초월한 결과를 보여줍니다. EVT는 ResNet-50 기반 모델에서 73.7% NDS 및 8.3 FPS의 속도를 유지하며, LiDAR 전용 모델에서도 뛰어난 효율성을 기록합니다. 이 논문은 성능과 추론 속도에서 모두 우수한 결과를 보이며, 다른 transformer 기반 접근법과 쉽게 통합될 수 있는 가능성을 제시합니다.



### Diagnostic Text-guided Representation Learning in Hierarchical Classification for Pathological Whole Slide Imag (https://arxiv.org/abs/2411.10709)
Comments:
          15 pages, 13 figures. Under Review

- **What's New**: 이 연구는 경량의 이미지 분석을 위한 새로운 계층적(PathTree) 기법을 도입하였습니다. 기존의 WSI(Whole Slide Image) 분석이 주로 슬라이드 수준의 레이블에 의존하는 반면, PathTree는 이들 레이블을 이진 트리 구조로 전환하여 다중 분류 문제를 보다 효과적으로 다루고자 합니다.

- **Technical Details**: PathTree는 질병의 다중 분류를 이진 트리 구조로 처리하는 방식으로, 각 카테고리는 전문적인 병리학적 텍스트 설명으로 표현됩니다. 이를 통해 다층적 정보 집합을 유도하며, 슬라이드-텍스트 유사성을 바탕으로 확률 점수를 산출합니다. 또한, 텍스트와 슬라이드 간의 연관성을 제약하는 두 개의 특정 손실 함수를 도입하여 모델의 성능을 향상시킵니다.

- **Performance Highlights**: 세 가지 도전적인 계층적 분류 데이터셋에서 실험을 통해 PathTree는 최신 기술들과 비교하여 지속적으로 우수한 성능을 보여주었습니다. 이러한 결과는 PathTree가 복잡한 병리학적 과제를 해결하는 새로운 관점을 제시하고 있으며, 병리학적인 작업에서의 더 나은 지원을 가능하게 합니다.



### AllRestorer: All-in-One Transformer for Image Restoration under Composite Degradations (https://arxiv.org/abs/2411.10708)
Comments:
          12 pages, 11 figures

- **What's New**: 이번 연구에서는 AllRestorer라는 Transformer 기반의 새로운 이미지 복원 프레임워크를 제안합니다. 기존의 방식들은 단일 또는 복합 손상을 처리하는 데 한계를 보였으며, 특히 다양한 비율로 존재하는 손상을 충분히 구분하지 못해 최적의 복원이 어려웠습니다. AllRestorer는 이미지 내 모든 손상을 동적으로 고려하여 이러한 문제를 해결하고, 이를 통해 실제 응용에서 더 좋은 성능을 기대할 수 있습니다.

- **Technical Details**: AllRestorer의 핵심은 All-in-One Transformer Block (AiOTB)으로, 이는 주어진 이미지에서 모든 손상을 제거하는 데 초점을 맞춥니다. AiOTB는 손상 유형 간의 관계를 모델링하여 모든 손상이 존재하는 이미지를 복원하며, 이미지와 텍스트 임베딩을 포함하는 복합 장면 설명자를 이용하여 정확한 손상 정의를 수행합니다. 또한, 각 손상에 대한 적응형 가중치를 포함해 복원 강도를 세밀하게 조정할 수 있습니다.

- **Performance Highlights**: AllRestorer는 CDD-11 데이터셋에서 기준선 대비 PSNR에서 5.00 dB의 성능 향상을 달성하며, 최신(SoTA) 성능을 입증했습니다. 실제 테스트 결과에서도 AllRestorer의 실용적인 응용 가능성이 확인되었습니다. 이러한 결과는 AllRestorer가 복합 손상의 문제를 효과적으로 처리할 수 있는 잠재력을 갖추고 있음을 보여줍니다.



### Poster: Reliable 3D Reconstruction for Ad-hoc Edge Implementations (https://arxiv.org/abs/2411.10705)
Comments:
          3 Pages, 2 figures, IEEE SEC 2024

- **What's New**: 이 논문에서는 실시간 복잡한 비디오 처리 애플리케이션을 지원하기 위한 새로운 엣지 리소스 관리 전략을 제안합니다. 특히, 포트폴리오 이론(portfolio theory)에 영감을 받은 카메라 선택 전략을 통해 공간적으로 및 시간적으로 연관된 시스템 중단을 고려하여 신뢰성 있는 다중 뷰 3D 재구성을 보장하겠다는 목표를 두고 있습니다.

- **Technical Details**: 논문에서는 N개의 카메라 장치와 K개의 엣지 서버로 구성된 시스템 모델을 제안합니다. 각 카메라는 서로 다른 시점에서 연속적인 비디오 프레임을 캡처하고 이를 엣지 서버로 실시간 전송하여 3D 재구성을 수행합니다. 카메라의 작동 중단 확률은 임의 변수로 표현되며, 이러한 중단은 서로 상관 관계를 가질 수 있습니다.

- **Performance Highlights**: 제안된 리소스 관리 기법은 다양한 시스템 중단 상황 하에서도 더 높은 재구성 품질을 보장할 수 있음을 초기 결과를 통해 보여줍니다. 기존의 기법들과 비교해 볼 때, 평균 재구성 품질이 향상되며, 재구성 품질의 표준 편차도 낮아지는 경향을 보였습니다.



### Diffusion-based Layer-wise Semantic Reconstruction for Unsupervised Out-of-Distribution Detection (https://arxiv.org/abs/2411.10701)
Comments:
          26 pages, 23 figures, published to Neurlps2024

- **What's New**: 본 논문은 unsupervised (비지도) OOD (Out-of-Distribution) 탐지 문제에 대해 새로운 diffusion 기반의 계층적 의미 재구성 방법을 제안합니다. 이는 기존의 생성모형(Generative models)들이 직면한 재구성과 압축 간의 딜레마를 해결하는데 중점을 두고 있습니다.

- **Technical Details**: 제안한 방법은 latent feature space에서의 ID (In-Distribution) 샘플과 OOD 샘플을 구별하기 위해 diffusion 모델의 데이터 재구성 능력을 활용합니다. 또한, Gaussian noise로 추출된 다층적 특징을 왜곡하고 diffusion 모델을 사용하여 재구성을 적용함으로써 ID와 OOD 샘플을 분리합니다.

- **Performance Highlights**: 다양한 데이터셋을 기반으로 한 다수의 벤치마크 실험에서, 제안된 방법이 탐지 정확도 및 속도 면에서 최첨단 성능을 달성함을 보여줍니다.



### Multi-perspective Contrastive Logit Distillation (https://arxiv.org/abs/2411.10693)
Comments:
          10 pages, 6 figures, 11 tabels, 9 formulas, including pseudo-code

- **What's New**: 새롭게 제안된 Multi-perspective Contrastive Logit Distillation (MCLD) 방법은 여러 관점에서 logits를 증류하는 혁신적이고 효율적인 방법으로, contrastive learning을 활용합니다. 기존의 knowledge distillation 연구는 주로 teacher 모델의 logits에서 얻은 정보를 극대화하는 데 중점을 두고 있으며, MCLD는 이와 차별화된 접근 방식을 제공합니다. MCLD는 Instance-wise CLD, Sample-wise CLD, Category-wise CLD의 세 가지 주요 구성 요소로 이루어져 있으며, teacher의 logits에서 student 모델로 더 많은 정보를 전달하도록 설계되었습니다.

- **Technical Details**: MCLD의 접근 방식은 기존의 Kullback-Leibler (KL) divergence를 사용하는 방법을 넘어, teacher 모델의 logits가 가진 의미론적 정보를 효과적으로 활용하는 데 초점을 맞추고 있습니다. 이 방법은 MLCD가 classification task loss 없이도 잘 작동하므로, 이전의 로그릿 증류 기법들과 비교할 때 더 적은 계산 비용으로도 높은 성능을 달성할 수 있습니다. 다양한 teacher-student 모델 쌍에 대한 실험을 통해 CIFAR-100, ImageNet 및 기타 데이터셋에서 MCLD의 성능을 평가하였습니다.

- **Performance Highlights**: MCLD는 CIFAR-100과 ImageNet에서의 이미지 분류 작업을 비롯하여 STL-10 및 Tiny-ImageNet에서의 표현 전이 실험에서도 뛰어난 성능을 보였습니다. 기존의 최첨단 기법들을 능가하는 성과를 기록하며, MCLD의 방법론이 knowledge distillation 분야에서 큰 진전을 이루었음을 보여줍니다. 이러한 결과는 MCLD가 실제 적용 가능성을 가진 강력한 도구임을 명확히 시사합니다.



### MaskMedPaint: Masked Medical Image Inpainting with Diffusion Models for Mitigation of Spurious Correlations (https://arxiv.org/abs/2411.10686)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 12 pages

- **What's New**: 이 논문은 Masked Medical Image Inpainting (MaskMedPaint)이라는 새로운 접근 방식을 제안합니다. 이는 Denoising Diffusion Probabilistic Models를 활용하여 의료 이미지를 보강하여 타겟 도메인에 맞게 특정 부분을 채우는 방식입니다.

- **Technical Details**: MaskMedPaint는 의료 이미지 분류에서의 spurious features 문제를 완화하기 위해 디자인된 방법입니다. 이 방법은 먼저 라벨이 붙은 출처 이미지를 사용하여 diffusion 모델을 파인튜닝하고, 이후 정제된 타겟 이미지를 이용하여 모델을 추가로 최적화합니다. 이렇게 강화된 이미지는 기존의 데이터를 보강하는데 사용되며, 최종적으로는 DenseNet121 모델을 사용하여 분류기를 훈련합니다.

- **Performance Highlights**: 실험 결과, MaskMedPaint는 ISIC 2018 및 Chest X-ray 데이터셋에서 타겟 도메인으로의 일반화 성능을 크게 향상시켰습니다. 기존의 데이터 증강 방식들에 비해 MaskMedPaint는 적은 수의 라벨이 없는 타겟 이미지로도 높은 성능을 발휘했습니다.



### From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling (https://arxiv.org/abs/2411.10685)
- **What's New**: 본 연구에서는 Masked Image Modeling (MIM)의 학습 효율성과 표현 품질을 개선하기 위한 새로운 커리큘럼 학습 프레임워크를 제안하고 있습니다. 이 방법은 모델이 복잡한 이미지 패턴을 학습하기 전에 프로토타입 예시를 바탕으로 기본 시각적 개념을 이해하도록 돕습니다. 또한, 온도 기반의 애닐링(annealing) 방식을 도입하여 훈련 분포를 점차적으로 확장함으로써 보다 안정적이고 효율적인 학습을 가능하게 합니다.

- **Technical Details**: 제안된 커리큘럼 학습 전략은 모델이 기본적인 시각적 패턴을 학습할 수 있도록 간단한 프로토타입 예시에서 시작하여 점차 복잡한 변형으로 나아가도록 구성되어 있습니다. MIM의 핵심인 Masked Auto Encoders (MAE)와 함께 활용되며, 데이터 선택을 통해 학습 초기 단계에서 효과적인 예시를 우선 노출할 수 있도록 합니다. 이러한 방식은 훈련 에포크를 상당히 감소시키면서도 성능을 개선하는 결과를 보입니다.

- **Performance Highlights**: 실험 결과, 제안한 커리큘럼 학습 전략은 ImageNet-1K 데이터셋에서 기존의 MIM 기준보다 훈련 효율성과 표현 품질에서 현저한 개선을 보였습니다. 특히 프로토타입 예시에서 다양한 예제로 점진적으로 전환함으로써 학습 초기 단계의 최적화 문제를 효과적으로 해결하고 있다는 점에서 독특합니다. 이러한 발견은 자가 지도 시각 학습에서 훈련 샘플 순서를 신중하게 조정하는 것이 중요하다는 것을 시사합니다.



### Underwater Image Enhancement with Cascaded Contrastive Learning (https://arxiv.org/abs/2411.10682)
Comments:
          Accepted by IEEE Transacitons on MultiMedia

- **What's New**: 이번 연구에서는 두 단계의 딥러닝 프레임워크인 CCL-Net을 제안하여 다양한 수중 이미지 열화 문제를 효과적으로 해결하고자 합니다. 본 방법은 색 보정 단계와 안개 제거 단계의 두 개의 연쇄 구조로 구성되어 있습니다. 각 단계에서 대비 손실(contrastive loss)을 사용하여 네트워크 훈련을 안내하며, 이를 통해 점진적인 수중 이미지 향상을 보장합니다.

- **Technical Details**: CCL-Net의 첫 번째 단계인 색 보정 네트워크(CC-Net)에서는 원시 수중 이미지를 부정 샘플로 사용하며, 가상 참조 이미지를 긍정 샘플로 사용하여 손실 함수를 구성합니다. 두 번째 단계인 안개 제거 네트워크(HR-Net)에서는 CC-Net에서 향상된 결과를 부정 샘플로 활용하고, 같은 방식으로 가상 참조 이미지를 긍정 샘플로 사용합니다. 이러한 방식은 색상 왜곡 및 안개 문제를 독립적으로 해결하는 데 최적화된 구조를 제공합니다.

- **Performance Highlights**: 다양한 기준 데이터셋에서 실시한 실험 결과, CCL-Net은 기존 최첨단 기법들과 비교하여 시각적 품질과 정량적 지표 모두에서 우수한 성능을 보였습니다. 본 연구의 주요 기여는 두 단계의 네트워크 구조와 대비 손실을 결합하여 수중 이미지 향상의 상한선과 하한선을 모두 개선하는 새로운 훈련 패러다임을 제안한 것입니다.



### SPDFusion: An Infrared and Visible Image Fusion Network Based on a Non-Euclidean Representation of Riemannian Manifolds (https://arxiv.org/abs/2411.10679)
Comments:
          14 pages, 12 figures

- **What's New**: 이 논문에서는 기존의 유클리드 공간을 넘어 비유클리드 구조를 가진 데이터의 복합 모드 이미지를 융합하는 새로운 접근 방식인 SPDFusion을 제안합니다. 이 모델은 리만 기하학(Riemannian geometry)을 적용하여 이미지의 내재된 통계적 상관관계를 활용하며, 사람의 시각적 인식과 정렬할 수 있도록 디자인되었습니다. 또한, SPD(대칭 양의 정부호, symmetric positive definite) 매트릭스를 기반으로 한 주목(attention) 모듈을 통해 공간적 글로벌 상관 의미를 처리합니다.

- **Technical Details**: 제안된 SPDFusion 프레임워크는 여러 모드의 이미지를 융합하기 위한 개념으로, Riemannian manifold를 도입하여 픽셀 간의 통계적 관계를 잘 포착하고 융합 효과를 향상시킵니다. SPD 매트릭스를 통해 네트워크 학습을 지원하고, cross-modal fusion 전략을 사용하여 특수한 모드 간 의존성을 활용합니다. 이로써 서로 다른 모드 간의 상호작용을 강화하고, 깊은 학습 기반의 새로운 영상 융합 구조를 제공합니다.

- **Performance Highlights**: 공공 데이터셋에 대한 광범위한 실험 결과, 제안한 SPDFusion 프레임워크는 기존의 최첨단 방법들보다 월등한 성능을 보였습니다. 이 연구는 이미지 융합 작업에서 리만 기하학을 처음으로 도입하여 기존의 유클리드 기반 방법들의 한계를 극복하고, 실질적인 시나리오의 데이터 구조를 효과적으로 반영합니다. 또한, 새로운 manifold attention 모듈이 특징 간의 글로벌 상관관계를 정확하게 반영하고, 이로 인해 더욱 향상된 융합 효과를 달성할 수 있음을 보여줍니다.



### Awaker2.5-VL: Stably Scaling MLLMs with Parameter-Efficient Mixture of Experts (https://arxiv.org/abs/2411.10669)
- **What's New**: 이 논문에서는 다양한 텍스트 및 비주얼 작업을 처리하기 위해 설계된 Awaker2.5-VL이라는 새로운 Mixture of Experts (MoE) 아키텍처를 제안합니다. 기존의 다중 작업 충돌(multi-task conflict) 문제를 해결하기 위해 설계된 이 모델은 각 작업에 맞는 능력을 갖춘 여러 전문가(experts) 구조를 통해 성능을 개선합니다. 또한, 각 전문가는 저랭크 적응(low-rank adaptation, LoRA) 구조를 사용하여 학습과 추론 속도를 높입니다.

- **Technical Details**: Awaker2.5-VL은 고정된 기본 모델(base model)을 갖추고, 활성화 및 비활성화를 조절하는 게이트 네트워크가 특징입니다. 이 모델은 언제나 활성화된 글로벌 전문가(global expert)를 포함하고 있으며, 나머지 전문가들은 상황에 따라 활성화됩니다. 조합된 출력은 사전 학습된 모델 쿼리와 글로벌 전문가 및 전문가들의 출력을 통해 결정되며, 모델의 학습 비용을 극적으로 낮출 수 있습니다.

- **Performance Highlights**: Awaker2.5-VL은 여러 최신 벤치마크에서 국지적인 최첨단 결과를 달성하였습니다. 실험을 통해, 다양한 작업에 대한 성능 저하 없이 안정적으로 작업을 처리할 수 있는 능력을 입증하였습니다. 이 모델은 기존의 MLLM보다 더욱 효과적인 접근법을 제공하며, 코드 및 모델 가중치는 프로젝트 페이지에서 공개됩니다.



### Segmentation of Ink and Parchment in Dead Sea Scroll Fragments (https://arxiv.org/abs/2411.10668)
Comments:
          17 pages, ICDAR-IJDAR-2025

- **What's New**: 최근 연구에서 사해문서에 대한 계산적 접근법이 제안되고 있으며, 이들은 여러 조각의 유사성을 분석하고 쌍을 이루는 작업을 포함합니다. 본 논문에서 제시된 MTEM(다중 스펙트럼 임계값 및 에너지 최소화) 접근법은 전통적인 이진화 기법보다 우수한 결과를 보여줍니다. 이 방법은 잉크와 양피지 영역을 정확하게 분리하는 데 중점을 두고, 새롭게 구축된 Qumran Segmentation Dataset (QSD)를 사용하여 성능을 향상시킵니다.

- **Technical Details**: MTEM 방법은 다중 스펙트럼 이미지를 통해 잉크와 양피지의 독특한 스펙트럼 신호를 바탕으로 두 영역을 분리합니다. 에너지 최소화 기법을 통해 잉크 윤곽선을 정제하고, 시끄러운 내부 잉크 영역보다 백그라운드와 더 뚜렷하게 구분할 수 있는 잉크 윤곽선을 사용합니다. 이러한 접근은 종래의 이진화 기법과 달리 수작업 레이블링에 의존하지 않으며, 다중 스펙트럼 이미징을 활용하여 이미지의 특정 패치를 분석하는 데 초점을 맞춥니다.

- **Performance Highlights**: MTEM 방법은 Otsu 및 Sauvola와 같은 전통적인 이진화 방법보다 양피지 세분화에서 유의미한 개선을 나타냅니다. 실험 결과, 본 방법은 잉크 경계를 명확하게 구분할 수 있으며, 구멍 및 배경 영역과의 차별화에서 성공적입니다. 따라서 QSD를 통해 분할된 데이터는 사해문서 조각 이미지의 잉크 및 양피지 영역 세분화를 위한 벤치마크 자료로 기능할 것입니다.



### Deep Loss Convexification for Learning Iterative Models (https://arxiv.org/abs/2411.10649)
Comments:
          12 pages, 10 figures, accepted paper to Transactions on Pattern Analysis and Machine Intelligence. arXiv admin note: text overlap with arXiv:2303.11526

- **What's New**: 본 논문에서는 비선형 최적화 문제에서 자주 발생하는 지역 최적성 문제를 해결하기 위해, Deep Loss Convexification(DLC)이라는 새로운 접근 방식을 제안합니다. DLC는 테스트 시 각 진실값 주변에 손실 경관을 국소적으로 볼록한 형태로 조형하는 방법으로 제안되었습니다. 이 방식은 인공 신경망에서의 과파라미터화를 활용하여, 손실 경관을 보다 효율적으로 학습할 수 있도록 합니다.

- **Technical Details**: 본 연구에서 제안하는 방법은 적대적 훈련(adversarial training)을 기반으로 하며, 입력 데이터 대신 진실값 예측을 조작하여 손실 함수의 구조를 개선합니다. 특히, star-convexity라는 구조적인 비볼록 함수의 패밀리를 사용하여 손실 경관을 재구성하며, 이는 새로운 hinge 손실을 기존 손실에 추가하여 차별화된 예측 성능을 생성할 수 있게 합니다. 이 접근법은 반복 신경망(RNN), 3D 포인트 클라우드 등록, 다중 모델 이미지 정렬과 같은 다양한 작업에 적용되었습니다.

- **Performance Highlights**: DLC는 기존 네트워크 아키텍처를 사용하여 높은 성능을 보여주었으며, 반복적인 최적화 문제에서 near-optimal 예측을 달성하는 데 기여합니다. 이를 통해, 고차원의 비선형 문제 해결의 정확성과 효율성을 향상시키는 것을 목표로 하고 있습니다. 본 연구 결과는 다양한 응용 분야에서 성능을 현저히 개선할 가능성을 보여줍니다.



### BlueLM-V-3B: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices (https://arxiv.org/abs/2411.10640)
Comments:
          21 pages

- **What's New**: 최근의 연구에서 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 도입과 인기가 상승함에 따라, 이러한 모델들이 일상 생활의 여러 측면에서 개선을 가져올 잠재력을 가지고 있다는 점에 주목하고 있습니다. 특히, 모바일 기기는 MLLMs를 효율적으로 배포할 수 있는 플랫폼으로, 일상 업무에 원활하게 통합될 수 있도록 합니다. 하지만 MLLMs를 모바일에서 배포하는 과정에서 메모리 용량과 계산 능력 제한으로 인해 부드럽고 실시간 처리가 도전과제가 되고 있습니다.

- **Technical Details**: 본 논문에서는 BlueLM-V-3B라는 알고리즘 및 시스템 공동 설계 접근법을 제안합니다. 이 모델은 2.7B 파라미터를 가진 언어 모델과 400M 파라미터를 가진 비전 인코더로 구성되어 있습니다. 모바일 플랫폼에서 모델 추론을 최적화하기 위해, 기존의 동적 해상도 방식을 재설계하고 하드웨어에 최적화된 배포를 위한 시스템 최적화를 구현하였습니다.

- **Performance Highlights**: BlueLM-V-3B는 MediaTek Dimensity 9300 프로세서에서 4비트 LLM 가중치 양자화를 통해 초당 24.4 토큰 생성 속도를 달성하며, OpenCompass 벤치마크에서 66.1의 평균 점수를 기록하여 4B 이하 모델 중 가장 높은 성능을 보였습니다. 이러한 성능을 통해 BlueLM-V-3B는 자원 제약 하드웨어에서의 효율적인 운영 가능성을 제시하고 있습니다.



### MTA: Multimodal Task Alignment for BEV Perception and Captioning (https://arxiv.org/abs/2411.10639)
Comments:
          10 pages

- **What's New**: MTA라는 새로운 다중 모달 작업 정렬 프레임워크를 통해 BEV 기반 3D 인식과 자막 작업을 통합하여 성능을 향상시키는 연구를 발표하였습니다.

- **Technical Details**: MTA는 두 가지 주요 구성 요소인 BEV-Language Alignment (BLA)와 Detection-Captioning Alignment (DCA)로 구성되어 있으며, 각각 BEV 장면 표현을 지면 진술 언어 표현과 정렬하고 시각적 출력과 자막 출력을 일관되게 만듭니다. MTA는 기존 BEV 기반 프레임워크에 통합 가능하며, 추가적인 계산 복잡성을 도입하지 않습니다.

- **Performance Highlights**: MTA는 nuScenes와 TOD3Cap 데이터 세트에서 기존 최첨단 기술에 비해 각각 4.9%의 인식 향상과 9.2%의 자막 향상을 달성하였습니다. 이러한 결과는 BEV 기반 인식과 자막 작업의 통합 정렬의 효과를 뒷받침합니다.



### Is thermography a viable solution for detecting pressure injuries in dark skin patients? (https://arxiv.org/abs/2411.10627)
Comments:
          9 pages

- **What's New**: 본 연구에서는 피부 색조가 더 어두운 35명을 대상으로 한 새로운 열 및 광학 이미징 데이터 세트를 도입했습니다. 이를 통해 피부 온도 차이를 유도하는 쿨링(cooling) 및 컵핑(cupping) 프로토콜을 통해 압박 손상(PI)의 조기 탐지를 위한 열화상 기술의 가능성을 탐구합니다. 여러 카메라와 조명 조건, 환자 자세 및 카메라 거리 등 다양한 데이터 수집 프로토콜을 비교하여 열화상 모델의 성능을 평가했습니다.

- **Technical Details**: 연구에서는 Eumelanin 색조 분류를 통해 대조군과 쿨링 프로토콜을 혼합하여 1680장의 이미지를 수집했습니다. CNN(Convolutional Neural Network) 모델, 특히 MobileNetV2를 사용하여 냉각 및 증기 치료의 두 가지 이진 분류 작업을 평가합니다. 이미징 프로토콜의 변화가 모델 성능에 미치는 영향을 분석하여, 실제 임상 환경에서 최적의 데이터 수집 접근 방식을 파악하는 데 중점을 두었습니다.

- **Performance Highlights**: 초기 결과에 따르면, 열화상 기반의 CNN 모델은 모든 피부 색조에서 데이터 수집 프로토콜의 영향에 강한 내성을 보였습니다. 연구는 기존의 기술적 한계를 극복하는 데 기여하고, PI 탐지의 정확성을 높이는 방법을 제시합니다. 이러한 발견은 다양한 피부 색조를 가진 환자에 대한 효과적인 압박 손상 탐지를 위한 치료 및 예방 전략에 큰 영향을 미칠 것입니다.



### Voxel-Aggergated Feature Synthesis: Efficient Dense Mapping for Simulated 3D Reasoning (https://arxiv.org/abs/2411.10616)
Comments:
          6 pages, 2 figures, CVPR 2025

- **What's New**: 이 논문에서는 최근의 State-of-the-art (SOTA) 개방형 멀티모달 3D 매핑 알고리즘의 계산 요구 사항 증가 문제를 해결하기 위해 새로운 접근 방식인 Voxel-Aggregated Feature Synthesis (VAFS)를 제안합니다. VAFS는 시뮬레이션에서의 밀집 3D 매핑을 위해 구간 RGBD 프레임을 분할하고 임베드하여 3D로 융합하는 작업을 단순화합니다. 이 방법은 계산량을 획기적으로 줄여주어 환경 수정과 관련된 연구에서도 효율적으로 적용할 수 있게 해줍니다.

- **Technical Details**: VAFS는 시뮬레이터의 물리 엔진을 활용하여 세분화된 포인트 클라우드를 사용하고 각 지역에 대한 합성적 뷰(view)를 생성함으로써, RGBD 프레임의 수로부터 장면 내 개체의 수로 임베드할 특징의 수를 줄입니다. 이는 전통적인 방법보다 약 10배 빠른 속도로 'ground truth' 의미론적 지도를 생성할 수 있게 합니다. 또한, 시뮬레이션 환경의 세분화된 포인트 클라우드를 활용하여, 밀집 3D 매핑의 계산 부담을 크게 경감시킴으로써 실시간 업데이트가 필요한 연구를 포함한 더 넓은 도메인에서 유용하게 작용합니다.

- **Performance Highlights**: 실험에서는 시뮬레이션 장면의 다양한 개체에 대한 의미론적 쿼리의 IoU 점수를 평가하여 VAFS의 성능을 입증하였습니다. VAFS는 이전의 밀집 3D 매핑 기술에 비해 더 높은 정확성과 빠른 속도를 기록하였으며, 이는 특히 에이전트 연구에 필요한 고충실도의 인식을 가능하게 합니다. 이러한 성과는 VAFS가 실시간 시스템에서도 효과적으로 적용될 수 있음을 보여줍니다.



### Creation and Evaluation of a Food Product Image Dataset for Product Property Extraction (https://arxiv.org/abs/2411.10591)
Comments:
          Accepted at 12th International Conference on Data Science, Technology and Applications (DATA 2023)

- **What's New**: 본 논문에서는 중소기업을 위한 가격 예측 과정에서 기계 학습(Machine Learning) 전문가의 의존도를 줄이기 위해 자동 기계 학습(Automated Machine Learning, AutoML) 솔루션을 도입하는 가능성을 보여줍니다.

- **Technical Details**: CRISP-DM 프로세스를 기반으로 수동 ML 파이프라인을 기계 학습과 비기계 학습 부분으로 분할하였으며, 모든 산업 요구사항을 충족하기 위한 새롭고 중요한 메트릭인 method evaluation score를 설계하여 품질 및 사용성을 고려하였습니다.

- **Performance Highlights**: 사례 연구를 통해 도메인 지식과 AutoML을 결합하면 혁신적인 중소기업이 자동화된 가격 예측 솔루션을 구현할 수 있음을 입증하였습니다.



### Motion Diffusion-Guided 3D Global HMR from a Dynamic Camera (https://arxiv.org/abs/2411.10582)
Comments:
          15 pages, 2 figures, submitted to TMLR

- **What's New**: DiffOpt는 Single Human Monocular Videos를 위한 새로운 글로벌 HMR (GHMR) 프레임워크로, Human Motion과 Camera Movement를 동시에 최적화하여 더 정확하고 신뢰할 수 있는 글로벌 인체 모션을 회복할 수 있도록 설계되었습니다. 이 프레임워크는 Motion Diffusion Model (MDM)을 활용하여 인체 모션의 일관성을 강화하며, Dynamic Camera Predictions을 통해 이동하는 카메라 아래에서의 예측을 안정화합니다.

- **Technical Details**: DiffOpt는 Neural Motion Field와 Motion Diffusion Model (MDM)을 통합하여, 전통적인 HMR 시스템보다 더 높은 비용 효율성과 접근성을 제공합니다. DROID-SLAM을 통한 동적 카메라 예측을 초기화하여, 단일 비디오에서 Human Motion과 Camera Movement의 정확한 분리를 가능하게 합니다. 이 과정에서 Motion Prior를 사용하여 예측 모션의 비개연성을 제한하며, 실시간 동적 이해를 통해 Human과 Camera 간의 상관성을 유지합니다.

- **Performance Highlights**: DiffOpt는 Electromagnetic Database of Global 3D Human Pose and Shape in the Wild (EMDB) 데이터셋의 비디오 시퀀스를 통해 그 성능을 검증하였으며, 동체를 복구하는 능력에서 다른 최신 GHMR 방법들보다 우수한 결과를 나타냈습니다. 특히 긴 비디오 설정에서显著한 우수성을 자랑하며, 글로벌 동작 회복 능력에서 최상의 성능을 보였습니다.



### Vision Eagle Attention: A New Lens for Advancing Image Classification (https://arxiv.org/abs/2411.10564)
Comments:
          7 pages, 2 figures, 3 tables

- **What's New**: 이번 연구는 컴퓨터 비전 과제에서 효율적인 특징 추출을 위해 Vision Eagle Attention이라는 새로운 주의 메커니즘을 도입하였습니다. 이 메커니즘은 컨볼루션(spatial attention)을 사용하여 이미지의 유용한 지역에 집중하게끔 하여, 모델의 성능을 개선합니다. 특히, 이 연구에서는 경량화된 ResNet-18 아키텍처에 통합되어 높은 효율성을 보여줍니다.

- **Technical Details**: 제안된 Vision Eagle Attention 모델은 페이즈마다 특징을 정제하기 위해 ResNet-18 백본의 여러 레이어 이후에 배치된 세 개의 Attention 블록으로 구성됩니다. 각 블록에서는 3x3 및 1x1 컨볼루션을 통해 지역 특징을 캡처하고, 이 결과를 ResNet의 출력과 곱하여 중요한 특징을 강조합니다. 이는 네트워크가 모델의 깊이에 따라 중요한 특징을 동적으로 강조할 수 있게 합니다.

- **Performance Highlights**: 제안된 모델은 FashionMNIST, Intel Image Classification, OracleMNIST의 세 가지 광범위한 벤치마크 데이터셋에서 평가되었으며, 이미지 분류 작업에 중점을 두고 분류 정확도를 향상시켰습니다. 실험 결과, Vision Eagle Attention은 모델의 성능을 크게 개선하는데 기여하였으며, 객체 탐지 및 분할과 같은 다른 비전 작업으로도 확장 가능성이 있음을 보여줍니다.



### The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods (https://arxiv.org/abs/2411.10546)
Comments:
          Website: this https URL

- **What's New**: 이 논문은 옥스퍼드의 유명한 랜드마크 주변에서 캡처된 대규모 다중 모드 데이터셋을 소개합니다. 이 데이터셋은 맞춤형 다중 센서 인식 장치를 사용하여 수집되었으며, 밀리미터 정밀도의 Terrestrial LiDAR Scanner (TLS) 지도를 기반으로 합니다. 연구자들은 이 데이터셋을 통해 Simultaneous Localisation and Mapping (SLAM) 및 3D reconstruction을 위한 몇 가지 벤치마크를 설정하고 평가합니다.

- **Technical Details**: 인식 장치는 세 개의 동기화된 글로벌 셔터 컬러 카메라, 자동차용 3D LiDAR 스캐너, 관성 센서를 포함하고 있습니다. 이 시스템은 정밀한 보정을 거쳤으며, 3D 모델의 Ground Truth는 TLS 3D 모델을 사용하여 계산됩니다. 또한, Neural Radiance Fields (NeRF) 및 3D Gaussian Splatting과 같은 본 논문에서 도입하는 새로운 방법론들을 평가합니다.

- **Performance Highlights**: 본 연구에서 평가한 최신 방사선 필드 방법은 훈련 데이터에 과적합(overfitting)하는 경향이 있으며, 훈련되지 않은 자세에 대한 일반화(generalise)에 어려움을 보였습니다. 이러한 문제는 특히 대규모 야외 환경에서 더 두드러지며, 효과적인 3D reconstruction과 photorealistic rendering에 부정적인 영향을 미칩니다. 최종적으로 본 데이터셋은 야외 SLAM 시스템 통합을 위한 새로운 연구 방향을 열어줄 것으로 기대됩니다.



### Any2Any: Incomplete Multimodal Retrieval with Conformal Prediction (https://arxiv.org/abs/2411.10513)
- **What's New**: 본 논문에서는 Any2Any라는 새로운 멀티모달 검색 프레임워크를 제안합니다. 이 프레임워크는 불완전한 모달리티를 가진 쿼리와 참조 인스턴스 모두에 대해 검색을 가능하게 합니다. 기존의 이중모달 검색 방식을 넘어 여러 모달리티에 대해 훈련 없이 활용 가능합니다.

- **Technical Details**: Any2Any는 크로스 모달 인코더를 사용하여 인스턴스 간의 쌍별 유사도를 계산하고, 두 단계 보정 과정을 통해 유사도를 정렬합니다. 첫 번째 단계에서는 확장 예측(conformal prediction)을 통해 유사도를 정규화하고, 두 번째 단계에서는 모든 모달리티 쌍의 올바른 검색 확률을 나타내는 스칼라 값을 생성합니다. 이를 통해 다양한 조합의 불완전한 모달리티를 가진 인스턴스 간의 직접 비교를 가능하게 합니다.

- **Performance Highlights**: KITTI 데이터셋에 대한 실험 결과 Any2Any는 Recall@5의 성능이 35%로 완전 모달리티를 가진 기준 모델들과 견줄 수 있는 성능을 보였습니다. 이를 통해 Any2Any는 다양한 크로스 모달 인코더에 대한 일반화 능력을 갖추고 있음을 보여주며, 멀티모달 데이터셋에서의 검색 가능성을 높입니다.



### TESGNN: Temporal Equivariant Scene Graph Neural Networks for Efficient and Robust Multi-View 3D Scene Understanding (https://arxiv.org/abs/2411.10509)
Comments:
          arXiv admin note: text overlap with arXiv:2407.00609

- **What's New**: 본 논문에서는 3D 포인트 클라우드에서 의미론적 장면 그래프를 생성하기 위해 Equivariant Scene Graph Neural Network (ESGNN)를 최초로 구현한 내용을 발표합니다. 특히, 시간 의존적인 관계를 포착하기 위한 새로운 Temporal Layer를 도입하여 다수의 시퀀스에서 생성된 장면 그래프를 통합합니다. 이를 통해, 더 정확하고 강건한 장면 이해가 가능해집니다.

- **Technical Details**: Temporal Equivariant Scene Graph Neural Network (TESGNN)는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 ESGNN으로, 포인트 클라우드를 입력받아 노드 및 엣지 임베딩을 생성하여 장면 그래프를 구축합니다. 두 번째는 Temporal Model로, 다양한 장면 그래프를 유사성 비교를 통해 그래프 매칭 문제를 해결하여 통합된 글로벌 장면 그래프를 형성합니다. 이 모델은 GNN의 회전 및 변환 불변성을 보장합니다.

- **Performance Highlights**: TESGNN은 기존 최첨단 방법들을 초월하는 장면 추정 정확도를 달성하며, 훈련 반복 횟수 또한 적게 소요됩니다. 이 방법은 계산 효율성이 뛰어나며, 기존 프레임워크를 사용하여 간단하게 구현할 수 있습니다. 따라서 로보틱스 및 컴퓨터 비전의 실시간 응용에 적합합니다.



### DR-BFR: Degradation Representation with Diffusion Models for Blind Face Restoration (https://arxiv.org/abs/2411.10508)
- **What's New**: 최근에 발전된 Diffusion Models (확산 모델)와 관련하여, 본 논문은 다양한 저품질 (LQ) 얼굴 이미지에서 특정한 저하 정보를 추출할 수 있는 능력을 부여함으로써 실제 이미지 복구의 성능을 크게 향상시킨 DR-BFR(Degradation Representation Blind Face Restoration) 방법을 소개합니다. 기존의 방법들보다 더 나은 자연스러움을 지닌 이미지를 복원하기 위해, 비지도 대조 학습과 재구성 손실을 활용하여 LQ 이미지에서 저하 프롬프트를 분리하는 방식을 제안합니다.

- **Technical Details**: DR-BFR은 Degradation Representation Module (DRM)과 Latent Diffusion Restoration Module (LDRM)의 두 가지 모듈로 구성됩니다. 이러한 모듈들은 각각 LQ 이미지의 관계없는 특성에서 저하 정보를 추출하고, 잠재 공간에서 두 가지 정보를 동시에 고려하여 고품질 이미지를 복원하는 역할을 합니다. 이러한 방식은 다양한 저하 유형과 정도를 효과적으로 분리하고, 복원 과정의 정확성을 높이는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안된 DR-BFR은 여러 데이터셋에서 최신 방법들과 비교할 때 모든 면에서 우수한 성능을 보여줍니다. 특히 자연스러움(naturalness) 및 신뢰성(fidelity) 지표에서 현격한 개선을 나타냈습니다. 본 연구의 기여는 LQ 이미지로부터 저하 정보를 명확히 분리하여 복원 정확성을 크게 증가시킨 것입니다.



### USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting (https://arxiv.org/abs/2411.10504)
- **What's New**: 본 논문에서는 Spike 카메라의 이미지 재구성과 포즈 보정을 통합한 새로운 최적화 프레임워크인 USP-Gaussian을 제안합니다. 기존의 분산된(cascading) 접근 방식에서 발생하는 오류를 방지하기 위해 Spike-to-Image 네트워크와 3D Gaussian Splatting(3DGS) 간의 정보를 원활하게 통합합니다. 이와 같은 통합 접근법은 3D 재구성의 정확성을 크게 향상시킵니다.

- **Technical Details**: USP-Gaussian은 Spike-to-Image 재구성을 위한 Recon-Net과 3DGS로 구성된 두 개의 상호 작용하는(branch) 구조를 가지고 있으며, 각 시스템은 독립적으로 작동합니다. Recon-Net의 자기 지도(self-supervised) 훈련은 입력 스파이크 스트림과 복구된 출력 간의 제약으로 설계됩니다. 3DGS 브랜치는 고품질 이미지나 정확한 포즈가 필요 없이 구동될 수 있는 새로운 손실 함수 ℒgs을 개발하여 포즈 최적화를 통해 형태를 보강합니다.

- **Performance Highlights**: 우리의 방법은 이전의 접근 방법들과 비교하여 복구된 이미지의 품질에서 우수한 성능을 보여주었습니다. 특히 실제 데이터셋에서 부정확한 초기 포즈로 인해 발생하는 노이즈를 효과적으로 줄이고 세밀한 질감을 유지함으로써 3D 재구성의 견고성을 달성했습니다. 실험 결과, USP-Gaussian은 다양한 데이터셋에서 우수한 성능을 입증하였으며, 획기적인 오류 전파 완화 방식으로 주목받습니다.



### Everything is a Video: Unifying Modalities through Next-Frame Prediction (https://arxiv.org/abs/2411.10503)
Comments:
          10 pages, 10 figures

- **What's New**: 본 논문에서는 전통적인 접근 방식의 한계를 극복하기 위해 다양한 모달리티를 통합하는 새로운 멀티모달 학습 프레임워크를 제안합니다. 이 프레임워크는 다양한 멀티모달 작업을 통합된 다음 프레임 예측 문제로 재구성하여 단일 모델이 모달리티에 관계없이 모든 입력을 처리할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 다양한 입력 및 출력 모달리티를 다음 프레임 예측 단일 작업으로 재구성하는 방법 및 (2) 순수 트랜스포머 기반 모델 아키텍처입니다. 모든 입력과 출력을 64x64 RGB 비디오 시퀀스로 변환하여 일관된 프레임워크를 생성하고, 입력과 예측의 경계를 명확히 하기 위해 구분자 토큰 (||||)을 사용합니다.

- **Performance Highlights**: 이 모델은 텍스트-텍스트, 이미지-텍스트, 비디오-비디오, 비디오-텍스트, 오디오-텍스트 등의 다양한 작업에서 평가되었으며, 최소한의 적응으로 모달리티 간 일반화 능력을 입증하고 있습니다. 실험 결과, 제안된 모델은 추가 데이터로 사전 훈련되지 않은 단일 작업 모델과 유사한 성능을 달성했습니다.



### OnlyFlow: Optical Flow based Motion Conditioning for Video Diffusion Models (https://arxiv.org/abs/2411.10501)
- **What's New**: 이번 논문에서는 OnlyFlow라는 새로운 텍스트-비디오 생성 모델을 소개하며, 사용자 동영상의 움직임을 반영하여 비디오를 생성하는 방법을 제안합니다. 이 모델은 사용자가 입력한 비디오의 optical flow를 활용하여 구체적인 모션 조작이 가능하게 합니다.

- **Technical Details**: OnlyFlow는 입력 비디오에서 추출된 optical flow를 기반으로 하여 비디오 생성 모델의 노이즈 제거(diffusion) 프로세스를 안내합니다. 이 방식은 AnimateDiff 기반의 비디오 생성 과정에 적용되며, 추가로 학습 가능한 optical flow 인코더를 사용합니다.

- **Performance Highlights**: OnlyFlow는 여러 비디오 생성 작업에서 정량적, 정성적 평가 및 사용자 선호 연구에서 최신 기술들에 비해 긍정적인 성과를 보였습니다. 이 모델은 다양한 비디오 생성 작업에서 효과적이고 다재다능하며 오픈 소스로 제공될 예정입니다.



### FitDiT: Advancing the Authentic Garment Details for High-fidelity Virtual Try-on (https://arxiv.org/abs/2411.10499)
Comments:
          Project link: this https URL

- **What's New**: 최근 전자상거래의 급격한 성장으로 인해 이미지 기반 가상 착용(virtual try-on, VTON)에 대한 수요가 증가하고 있습니다. 이에 따라, GAN(Generative Adversarial Networks)이나 LDM(Latent Diffusion Models)과 같은 기존의 방법들이 다양하고 복잡한 의류 텍스처를 구현하는 데 한계가 있어, 이에 대한 해결책으로 FitDiT라는 새로운 기술이 제안되었습니다. 이 기술은 Diffusion Transformer(: DiT)을 활용하여 고해상도 특징에 더 많은 주의를 할당하여 사실감이 뛰어난 가상 착용 이미지를 생성합니다.

- **Technical Details**: FitDiT는 의류 데이터에 대한 사전 진화 전략을 도입하여 복잡한 질감을 향상시키고, 고주파 세부 사항을 강화하기 위해 주파수 거리 손실을 사용자화하는 등 여러 기능을 포함하고 있습니다. 이 외에도, 사이즈 인식 착용 문제를 해결하기 위해 정방형 마스크를 이용하여 의류 정보 누수를 방지하는 전략이 사용됩니다. 이러한 새로운 설계는 의류의 잘 맞는 모습과 사실적인 세부 사항을 생성하는 데 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: FitDiT는 정성적 및 정량적 평가에서 모든 기준선을 초과하는 결과를 보였으며, 다양한 복잡한 의류 질감을 효과적으로 처리합니다. 특히, 1024x768 이미지의 단일 추론 시간은 4.57초로 기존 방법보다도 경쟁력 있는 성능을 나타냅니다. 이러한 성과는 가상 착용 분야에서 더 정교한 응용 가능성을 열어주는 중요한 이정표가 되고 있습니다.



### Prompt-Guided Environmentally Consistent Adversarial Patch (https://arxiv.org/abs/2411.10498)
- **What's New**: 본 논문에서는 Prompt-Guided Environmentally Consistent Adversarial Patch (PG-ECAP)라는 새로운 적대적 패치 생성 접근 방식을 제안합니다. 이 방법은 패치의 시각적 자연스러움과 환경 일치성을 모두 해결하여 실제 환경에서도 효과적으로 작동합니다. 특히, 텍스트-이미지 모델의 능력을 활용하여 패치가 주어진 환경 특성과 잘 어우러지도록 생성하는 것을 목표로 합니다.

- **Technical Details**: PG-ECAP은 diffusion models를 사용하여 자연스러운 패치를 생성하고, Prompt Alignment Loss 및 Latent Space Alignment Loss라는 두 가지 정렬 손실을 도입하여 패치의 이미지를 조정합니다. 이 정렬 손실은 패치가 환경적 요소와 일관성을 유지하도록 보장하며, 새로운 환경에 대한 적절한 텍스트 프롬프트를 통해 재조정할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면 PG-ECAP은 기존의 방법들보다 공격 성공률과 환경 일치성 측면에서 우수한 성능을 보입니다. 디지털 및 물리적 환경 모두에서 수행된 실험을 통해, 제안된 방법의 효율성과 실용성이 여러 실제 시나리오에서 입증되었습니다.



### Structure Tensor Representation for Robust Oriented Object Detection (https://arxiv.org/abs/2411.10497)
- **What's New**: 이 논문은 객체 탐지 분야에서 방향성을 지정하는 경량의 구조 텐서를 제안합니다. 이 방법은 기존 HBB(수평 바운딩 박스) 대신 OBB(지향 바운딩 박스)를 사용하여 객체의 정확한 위치와 방향을 예측할 수 있도록 합니다. 이 구조 텐서는 추가 하이퍼파라미터 없이도 각도 주기 문제에 강인한 성능을 보이는 간단하면서도 효율적인 접근 방식을 제공합니다.

- **Technical Details**: 제안된 방법은 2x2 대칭 행렬인 구조 텐서를 사용하여 방향성과 비등방성을 표현합니다. 이 구조는 경량으로 구현하기 쉬우며, 특징적으로 경계 불연속성이 없는 방향성 표현을 가능하게 하고 다양한 대칭성을 모델링할 수 있는 유연성을 제공합니다. 실험을 통해 제안한 방법이 SOTA(최첨단) 성능을 일관되게 달성하고 기존 방법보다 우수하게 성능을 발휘함을 입증하였습니다.

- **Performance Highlights**: 다섯 개의 데이터셋을 통해 실시한 체계적인 평가 결과, 구조 텐서 기반 방법이 완전 지도 학습 및 약한 지도 학습 작업에서 기존 방법보다 높은 정확도를 기록했습니다. 이 연구의 기여는 OBB의 방향성을 구조 텐서로 표현한 최초의 시도로, 공개된 코드로 기존 탐지기에 쉽게 통합할 수 있는 솔루션을 제공합니다.



### Boundary Attention Constrained Zero-Shot Layout-To-Image Generation (https://arxiv.org/abs/2411.10495)
- **What's New**: 최근 텍스트-이미지 확산 모델들이 높은 해상도의 이미지를 생성하는 데 뛰어난 성능을 보이고 있지만, 공간 구성을 정확하게 제어하고 객체 수를 세는 데 어려움을 겪고 있습니다. 이러한 문제를 해결하기 위해 레이아웃-투-이미지 (L2I) 접근 방식이 연구되었으며, 본 논문에서는 추가 모듈이나 미세 조정 없이 새로운 제로샷 L2I 접근 방식인 BACON (Boundary Attention Constrained generation)을 제안합니다. BACON은 텍스트-비주얼 교차 주의(feature map)를 사용하여 생성된 이미지와 제공된 지침 간의 불일치를 정량화하고, 이를 통해 생성된 이미지의 레이아웃을 최적화합니다.

- **Technical Details**: BACON은 픽셀-투-픽셀 상관관계를 활용하여 자가 주의(feature map)와 교차 주의 맵을 정렬하고, 경계 주의 제약을 통해 객체의 크기 불일치와 불정확한 카운팅을 해결하고자 합니다. 이 방식은 교차 주의 맵의 노이즈를 필터링하고 저조한 주의 점수를 가진 개념의 경계를 강화하여 더 나은 공간 제어 능력을 제공합니다. 실험적으로도 BACON이 기존 제로샷 L2I 방법들에 비해 정량적 및 정성적으로 우수한 성과를 보여주었다고 보고하였습니다.

- **Performance Highlights**: 논문에서는 BACON이 DrawBench와 HRS 기준에서 이미지 구성을 고려했을 때 기존의 제로샷 L2I 기술들보다 월등한 성능을 발휘함을 입증한 실험 결과를 제시합니다. BACON은 레이아웃 입력의 복잡성으로 인한 의미적 실패를 분석하고, 겹치는 교차 주의 맵 문제를 해결합니다. 종합적으로, BACON은 주어진 레이아웃 지침을 기반으로 더 정확하고 일관된 이미지를 생성하는 데 기여합니다.



### MFP3D: Monocular Food Portion Estimation Leveraging 3D Point Clouds (https://arxiv.org/abs/2411.10492)
Comments:
          9th International Workshop on Multimedia Assisted Dietary Management, in conjunction with the 27th International Conference on Pattern Recognition (ICPR2024)

- **What's New**: MFP3D는 단일 단안 이미지(monocular image)만을 사용하여 음식의 양을 정확하게 추정하기 위한 새로운 프레임워크이다. 기존 방법들은 깊이 맵(depth map)이나 물리적 참조(physical reference)와 같은 특정 요건을 필요로 하지만, MFP3D는 이러한 추가 정보를 요구하지 않는다. 이 방법은 3D point cloud를 재구성하고, 3D 및 2D의 특징(feature)을 결합하여 보다 정교한 추정을 가능하게 한다.

- **Technical Details**: MFP3D는 세 가지 주요 모듈로 구성된다: (1) 3D Reconstruction Module은 2D 이미지를 기반으로 3D 포인트 클라우드(point cloud)를 생성하는 역할을 한다. (2) Feature Extraction Module에서는 3D 포인트 클라우드와 2D RGB 이미지로부터 특징을 추출하고 이를 병합(concatenate)한다. (3) Portion Regression Module은 깊은 회귀 모델(deep regression model)을 통해 추출된 특징을 바탕으로 음식의 부피(volume)와 에너지(content)를 추정한다.

- **Performance Highlights**: MFP3D는 637개의 음식 객체(food objects)를 포함하는 MetaFood3D 데이터셋에서 기존 방법들보다 훨씬 향상된 정확도를 보여주었다. 이는 음식 섭취 이미지로부터 원하는 영양 정보를 더 쉽게 추정할 수 있도록 한다. 따라서 MFP3D는 건강 관리 및 식이 평가에서 매우 유용한 도구가 될 것으로 기대된다.



### Hateful Meme Detection through Context-Sensitive Prompting and Fine-Grained Labeling (https://arxiv.org/abs/2411.10480)
Comments:
          AAAI-25 Student Abstract, Oral Presentation

- **What's New**: 이번 연구에서는 멀티모달(multi-modal) 콘텐츠의 자동 조정을 위한 최적화 개념 프레임워크를 제안합니다. 기존에 연구된 방법들과는 달리, 모델 성능 향상뿐만 아니라 모달리티(modality), 프롬프팅(prompting), 라벨링(labeling), 그리고 파인튜닝(fine-tuning)을 포괄하는 엔드투엔드(End-to-End) 최적화 파이프라인을 개발하였습니다.

- **Technical Details**: 제안된 프레임워크의 핵심 원리는 멀티변량 최적화(multi-variate optimization) 문제로, 모달리티, 프롬프팅, 라벨링, 파인튜닝의 성능을 고려합니다. 실험은 다양한 프롬프팅 및 라벨링 전략을 포함해 12개의 실험을 통해 진행되었고, 페이스북 증오 밈 데이터셋을 사용하여 ACCU 및 AUROC로 성과를 평가하였습니다.

- **Performance Highlights**: 모델 M은 68.933%의 정확도와 66.827%의 AUROC로 최고의 성능을 기록하여, 단순히 복잡한 모델이 아니라는 것을 강조합니다. 실험결과 파인튜닝, 범주형 프롬프팅, 이진 라벨링의 조합이 모델 성능 향상에 기여했으며, 독립적인 최적화가 반드시 최고의 결과를 보장하지 않음을 보여주었습니다.



### RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator (https://arxiv.org/abs/2411.11839)
- **What's New**: 이번 논문에서는 RoboGSim이라는 새로운 로봇 시뮬레이터를 제안하며, 이는 시뮬레이션과 실제 환경 간의 데이터 전이 문제를 해결하는 데 중점을 두고 있습니다. RoboGSim은 3D Gaussian Splatting(3DGS)와 물리 엔진을 결합하여 고충실도의 데이터 수집과 평가를 가능하게 합니다. 주요 기능으로는 Gaussian Reconstructor, Digital Twins Builder, Scene Composer 및 Interactive Engine이 있으며, 이들 구성 요소는 함께 작동하여 시뮬레이션 데이터를 효율적으로 동기화합니다.

- **Technical Details**: RoboGSim의 핵심 구성 요소는 4개로 나뉘며, Gaussian Reconstructor는 다중 시점 RGB 이미지 시퀀스를 입력으로 받아 장면과 객체를 재구성합니다. Digital Twins Builder는 메쉬 재구성을 통해 디지털 트윈을 생성하며, Scene Composer는 새로운 시점에서 이미지를 렌더링합니다. Interactive Engine은 포리시(policy) 평가 및 시뮬레이션 데이터 수집 등 다양한 기능을 제공합니다.

- **Performance Highlights**: RoboGSim은 새로운 장면, 뷰, 객체를 생성하여 정책 학습에 필요한 현실적인 조작 시연을 생성합니다. 이 시뮬레이터는 물리적으로 일관된 폐쇄 루프 평가를 수행할 수 있어, 정책 네트워크의 공정한 비교를 보장합니다. 실험 결과, RoboGSim은 실제 작업에서 신뢰성을 입증하며, 시뮬레이션과 실제 환경 간의 일관성을 높였습니다.



### Equivariant spatio-hemispherical networks for diffusion MRI deconvolution (https://arxiv.org/abs/2411.11819)
Comments:
          Accepted to NeurIPS 2024. 24 pages with 13 figures. Code available at this https URL

- **What's New**: 이 논문은 확산 MRI(dMRI) 이미지를 처리하기 위한 새로운 합성곱 신경망(convolutional neural network) 구조를 제안합니다. 특히, 이 신경망은 물리적 대칭인 회전(rotation), 평행이동(translation), 반사(reflection) 및 복셀 단위의 회전을 모두 고려하여 효율적인 spatio-spherical 데이터를 분석합니다. 제안된 모델은 뇌의 신경경로를 복원하는 데 있어 이전의 방법들보다 현저한 성능 개선을 보여줍니다.

- **Technical Details**: 논문에서는 spatio-hemispherical graph convolution을 사용하여 고차원 dMRI 데이터를 분석하는 새로운 접근법을 제시합니다. 이 방법은 dMRI의 공간-구형(spatio-spherical) 데이터 기하학과 신경 섬유의 반대 대칭성을 고려하여 계산의 효율을 극대화합니다. 이러한 접근은 voxel의 수가 많을 때 성능을 향상시키고, 전통적인 방법에 비해 처리 시간을 65% 줄이는 데 성공했습니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션 및 실제 인체 데이터셋에서 최첨단 결과를 달성했습니다. 특히, 연구 수준에서 임상 표준으로 해상도가 변경되더라도 더욱 공간적 연속성이 있는 fODF(field of fiber orientation distribution)를 생성했습니다. 마지막으로, 대량의 인간 데이터셋에 대해 단일 네트워크로 학습할 수 있어, 새롭게 수집된 dMRI 데이터에 대한 효율적인 추론이 가능해졌습니다.



### Edge-Enhanced Dilated Residual Attention Network for Multimodal Medical Image Fusion (https://arxiv.org/abs/2411.11799)
Comments:
          An extended version of the paper accepted at IEEE BIBM 2024

- **What's New**: 이번 연구에서는 멀티모달 의료 이미지 융합을 위한 새로운 CNN 기반 아키텍처를 제안했습니다. 이는 경계 세부 정보를 향상시키기 위한 기울기 연산자와 효과적인 다중 스케일 기능 추출을 위한 확장된 잔차 주의 네트워크 모듈을 도입하여 기존의 한계를 극복합니다. 또한, 교육 및 추론 동안 추가 계산이 필요 없는 파라미터 없는 융합 전략을 소개하여 융합의 속도와 효율성을 높입니다.

- **Technical Details**: 제안된 융합 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 기능 인코더, 융합 모듈 및 기능 디코더. 비대칭 오토인코더를 통해 입력 이미지에서 다중 스케일 기능을 추출하고, 이 기능을 융합하여 최종 이미지를 복원합니다. 훈련 과정은 다중 단계로 분리되어 있으며, 첫 번째 단계에서는 일반적인 복원 작업을 통해 다중 스케일 깊이 기능을 추출합니다.

- **Performance Highlights**: 다양한 베이스라인 방법들과 비교한 광범위한 실험 결과, 제안된 방법이 시각적 품질, 텍스처 보존 및 융합 속도 면에서 모두 우수한 성능을 보였습니다. 특히, 고등급 및 저등급 신경교종 분류 작업에 대한 적응성을 확인하여, 실제 임상 응용에 적합한 솔루션이 될 수 있음을 입증했습니다. 코드도 공개될 예정이며, 이를 통해 실제 응용 가능성을 더욱 높일 것입니다.



### Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods (https://arxiv.org/abs/2411.11795)
- **What's New**: 이 논문은 JPEG AI라는 새로운 신경 이미지 압축(NIC) 표준의 적대적 공격에 대한 강인성을 측정하는 새로운 방법론을 제안합니다. JPEG AI는 소비자 기기에 내장된 최초의 신경망 기반 모델로, 이미지 압축의 품질을 개선할 가능성이 있습니다. 최근 연구에서 NIC의 강인성에 대한 정의가 명확하게 제시되지 않아, 본 논문은 이를 평가하기 위한 첫 번째 대규모 평가를 수행하였습니다.

- **Technical Details**: JPEG AI는 이미지의 내부 표현을 비선형 변환을 통해 변환하는 분석 변환, 과도한 정보를 줄이기 위한 양자화, 예측 가능성에 따라 바이너리 데이터로 압축하는 엔트로피 코딩, 그리고 학습된 비선형 합성 변환을 사용하여 압축된 데이터에서 이미지를 재구성하는 합성 변환의 네 가지 핵심 요소로 구성됩니다. 본 연구에서는 JPEG AI를 포함한 10개의 NIC 모델을 대상으로 6가지 공격을 분석하고 여러 손실 함수에 대한 공격-손실 조합을 평가하였습니다.

- **Performance Highlights**: 제안된 방법론은 다양한 손실 함수와 공격을 조합하여 NIC의 강인성을 평가합니다. 본 연구는 JPEG AI의 강인성을 다른 NIC 모델과 비교하여 적대적 공격에 대한 상대적 효과를 분석하며, NIC의 취약점에 대한 정화 방어 전략을 적용하여 과학적으로 평가합니다. 이는 이미지 품질 또는 비트 전송률을 목표로 하는 다양한 공격을 포함하며, NIC 모델의 취약성을 극복하는 데 도움이 됩니다.



### Revitalizing Electoral Trust: Enhancing Transparency and Efficiency through Automated Voter Counting with Machine Learning (https://arxiv.org/abs/2411.11740)
Comments:
          13 Pages, 4 Figures

- **What's New**: 이번 연구는 선거 절차 중 수동 투표 집계 문제에 대한 해결책으로 고급 이미지 처리 기술을 활용한 자동화된 유권자 집계 시스템의 가능성을 조사합니다. OpenCV, CVZone 및 MOG2 알고리즘과 같은 최첨단 기술을 활용하여 투표 과정의 효율성과 투명성을 크게 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 Bangladesh를 사례로 하여 기계 학습 및 이미지 처리 기술을 적용하여 수동 투표 집계 시스템의 정확성과 신뢰성을 향상시키기 위한 방법을 제안합니다. OpenCV와 CVZone 라이브러리를 사용하여 유권자의 얼굴 및 신체 특성을 인식하여 개별 유권자를 식별하고 집계하는 자동화된 시스템을 구현합니다. Mixture of Gaussian 2 (MoG2) 알고리즘은 실시간으로 유권자를 개별적으로 감지하고 계산하는 데 사용됩니다.

- **Performance Highlights**: 자동화된 유권자 집계 시스템은 수동 집계 방식에 비해 오류를 줄이고 투명성을 증가시킬 수 있습니다. 연구 결과는 특히 신뢰도가 낮은 지역에서 투표 과정의 신뢰를 회복하고 선거 결과의 공정성을 향상시킬 수 있는 방안을 제시합니다. F1 score와 같은 엄격한 지표를 통해 자동화 시스템과 수동 집계 방법 간의 정확성을 체계적으로 비교하여 유의미한 성과를 도출했습니다.



### Aligning Few-Step Diffusion Models with Dense Reward Difference Learning (https://arxiv.org/abs/2411.11727)
- **What's New**: 이 논문에서는 단계의 일반화 문제를 해결하기 위해 몇 단계의 diffusion 모델을 정렬하는 새로운 방법인 Stepwise Diffusion Policy Optimization (SDPO)를 소개합니다. 기존의 희소 보상 체계를 벗어나 각 단계에서 밀집 보상 피드백을 통합하여 일관된 정렬을 보장합니다.

- **Technical Details**: SDPO는 단계별로 쌍 샘플 간의 밀집 보상 차이를 학습하여 few-step diffusion 모델을 정렬하는 RL 알고리즘입니다. 미리 예측된 샘플을 활용하여 모든 단계에서 지속적인 밀집 보상 피드백을 제공하며, 적응형 보상 질의 단계를 선택하여 계산 비용을 최소화합니다. 이를 통해 변화하는 단서에 기반하여 밀집 보상을 효과적으로 제공합니다.

- **Performance Highlights**: 실험 결과 SDPO는 기존 방법들에 비해 샘플 효율성 및 단계 일반화 성능이 뛰어난 것으로 나타났습니다. 다양한 샘플링 단계 설정에서도 높은 보상 점수를 달성하며, 보상 정렬 이미지 생성에서도 우수한 성과를 보였습니다.



### Dissecting Misalignment of Multimodal Large Language Models via Influence Function (https://arxiv.org/abs/2411.11667)
Comments:
          34 pages

- **What's New**: 본 논문은 다중 모달 대형 언어 모델(MLLMs)의 데이터 정합성을 평가하기 위한 확장된 영향 함수(ECIF)를 제안합니다. 기존 영향 함수 방법들이 시간 소모적이거나 부적합한 점을 개선하여, 긍정 및 부정 샘플의 영향을 모두 고려할 수 있는 방법론을 제공합니다.

- **Technical Details**: ECIF는 대조 손실(contrastive loss)에 최적화된 영향 함수를 제공하며, 재교육 없이도 대조 학습 모델의 성능을 평가하는 폐쇄형 근사(closed-form approximation)를 포함합니다. 이 방법은 긍정 샘플과 부정 샘플의 두 가지 관점을 통합하여 데이터의 중요성을 평가합니다.

- **Performance Highlights**: 실험 결과, ECIF는 전통적인 기준 방법들에 비해 MLLM의 데이터 영향 평가 정확도를 향상시키고, 미스라벨링(mislabeled) 또는 정합성 문제(misalignment)를 효과적으로 감지할 수 있음을 입증하였습니다.



### HistoEncoder: a digital pathology foundation model for prostate cancer (https://arxiv.org/abs/2411.11458)
- **What's New**: 이번 연구에서는 HistoEncoder라는 기초 모델을 개발하여 전립선 암 디지털 병리학에 적용했습니다. 이 모델은 4800만 개의 전립선 조직 타일 이미지를 사전 학습하여 복잡한 생리적 패턴을 구별하며, 이는 제한된 자원으로 다양한 다운스트림 작업에 적응 가능합니다. HistoEncoder는 자연 이미지로 사전 학습된 모델보다 성능 면에서 우수한 결과를 보였으며, 두 가지 활용 사례를 통해 한정된 데이터로 훈련시키는 방법을 제시합니다.

- **Technical Details**: HistoEncoder는 crossover-covariance image transformers (XCiT)을 기반으로 하며, DINO라는 자가 감독 학습 방법을 사용하여 특징을 추출합니다. 이 모델은 디지털 병리학의 고해상도 이미지를 효율적으로 처리할 수 있도록 설계되었고, 두 가지 변형 (prostate-s와 prostate-m)로 훈련되었습니다. 기존 자연 이미지 기반 모델의 한계를 극복하고, 특성을 정확하게 클러스터링하여 자동 주석 달기에 활용될 수 있는 방법론을 개발하였습니다.

- **Performance Highlights**: HistoEncoder는 자동 주석 달기 및 전립선 암의 임상 예측 모델 향상에 기여했습니다. 모델을 사용하여 대규모 조직 이미지 데이터셋을 자동으로 주석 처리할 수 있는 높은 정확도를 달성하였으며, 임상 예측 모델에 통합하여 생존률 예측을 향상시켰습니다. 이 기초 모델은 제한된 자원만으로도 효과적인 임상 소프트웨어 도구를 구축할 수 있는 가능성을 보여줍니다.



### Lung Disease Detection with Vision Transformers: A Comparative Study of Machine Learning Methods (https://arxiv.org/abs/2411.11376)
- **What's New**: 이번 연구는 Vision Transformers (ViT)를 활용하여 흉부 X-레이 분석의 진단 정확도를 개선하는 가능성을 탐구합니다. CNNs(Convolutional Neural Networks)가 주로 사용되는 의료 영상 분석 분야에서, ViT는 기존의 CNN 기반 모델들보다 더 우수한 성능을 보입니다. 연구에서는 전체 흉부 X-레이 이미지를 사용하는 접근법과 폐 세그먼트를 집중적으로 분석하는 접근법을 비교하여, 이 두 가지 방법이 병리학적 검사에서 얼마나 뛰어난지를 보여줍니다.

- **Technical Details**: 이 연구는 NIH Chest X-ray 데이터셋과 COVID-19 이미지 데이터 컬렉션의 두 가지 주요 데이터셋을 활용하여 진행되었습니다. ViT 아키텍처는 이미지를 패치(patch)로 나누고, 각 패치를 시퀀스로 처리하는 방식으로 작동하며, 이는 패턴 및 구조를 포착하는데 효과적입니다. 모델은 12개의 레이어로 구성된 Transformer Encoder로 이루어져 있으며, 여기에는 다중 헤드 셀프 어텐션 메커니즘이 포함되어 있습니다.

- **Performance Highlights**: 실험 결과, 전체 이미지 접근법을 사용한 ViT가 최대 97.83%의 정확도를 달성하였고, 폐 세그먼트 접근법은 96.58%의 정확도를 보였습니다. 두 접근법 모두 전통적인 CNN 모델을 초월하는 성과를 보여주었으며, 정밀도(precision), 재현율(recall), F1 점수(F1 score), AUC-ROC 등 다양한 메트릭에서 뛰어난 성능을 나타냈습니다. 이러한 결과는 ViT가 폐 X-레이 분석에서 진단의 정확성을 높이는 데 기여할 수 있음을 시사합니다.



### Continuous K-space Recovery Network with Image Guidance for Fast MRI Reconstruction (https://arxiv.org/abs/2411.11282)
- **What's New**: 이번 연구에서는 MRI 이미지를 고품질로 복원하기 위해 연속 k-space 복원 네트워크를 제안했습니다. 기존 연구들은 일반적인 이미지 처리 네트워크를 k-space 복원에 직접 적용하여 k-space의 독특한 특성을 간과했지만, 본 연구는 이미지 도메인 가이드를 활용한 암시적 신경 표현(In implicit neural representation) 관점으로 접근했습니다. 이를 통해 MRI 재구성 성능을 향상시킬 수 있습니다.

- **Technical Details**: IGKR-Net은 암시적 신경 표현을 기반으로 한 인코더-디코더 구조를 사용하여 샘플링되지 않은 k-값을 지속적으로 질의합니다. 이 구조는 표준 트랜스포머를 포함하여 다중 헤드 셀프-어텐션(multi-head self-attention) 및 피드-포워드 네트워크(feed-forward network)로 구성되어 k-space의 신호를 직접 처리합니다. 또한, 저품질 MRI 이미지에서 의미 정보를 추출해 k-space 복원을 안내하는 이미지 도메인 가이드 모듈이 설계되었습니다.

- **Performance Highlights**: CC359, fastMRI, IXI의 세 가지 공공 데이터셋에 대한 광범위한 실험을 통해 제안한 방법이 기존의 복원 방식보다 우수함을 입증했습니다. 연구 결과, IGKR-Net은 이미지 도메인 가이드와 다단계 훈련 전략을 통해 과도한 매끄러움 및 왜곡 문제를 완화하면서 정밀한 결과를 도출할 수 있음을 보여주었습니다. 이 접근법은 MRI 영상의 질을 개선하는 데 중요한 기여를 할 것으로 기대됩니다.



### DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation (https://arxiv.org/abs/2411.11252)
Comments:
this https URL

- **What's New**: 이 논문에서는 DrivingSphere라는 새로운 4D 폐쇄형 시뮬레이션 프레임워크를 제안합니다. 이 프레임워크는 실제 도로 환경을 정밀하게 반영한 시뮬레이션을 가능하게 하여 자율주행 알고리즘의 평가와 검증을 보장합니다. 기존의 오픈 루프 시뮬레이션이 갖는 한계를 극복하고, 동적 피드백 루프를 통해 시뮬레이션과 자율주행 시스템 간의 상호작용을 지원합니다. 이를 통해 개선된 주행 시나리오와 환경 구성 및 시각적 일관성을 제공합니다.

- **Technical Details**: DrivingSphere는 두 가지 주요 모듈, 즉 Dynamic Environment Composition과 Visual Scene Synthesis로 구성됩니다. Dynamic Environment Composition 모듈은 4D 환경을 구성하여 정적 및 동적 요소들을 통합합니다. Visual Scene Synthesis 모듈은 이 데이터를 고해상도의 멀티 뷰 비디오 출력으로 변환하고, ID 인식 기능을 통해 시간적 및 공간적 일관성을 보장합니다. 이러한 구조를 통해 DrivingSphere는 복잡한 주행 환경을 정밀하게 캡처하고, 자율주행 시스템의 성능을 높이는 데 기여합니다.

- **Performance Highlights**: DrivingSphere는 시각적 충실도 및 시간적 일관성에서 우수한 성능을 발휘하여 시뮬레이션과 실제 환경 간의 영역 차이를 현저하게 줄일 수 있습니다. 기존의 오픈 루프 및 폐쇄 루프 평가에서 모두 뛰어난 성과를 나타내며, 자율주행 시스템의 동적인 에이전트-환경 상호작용을 지원하는 안정적인 플랫폼으로 자리 잡고 있습니다. 앞으로 DrivingSphere는 더 안전하고 신뢰할 수 있는 자율주행 차량 개발에 중요한 기여를 할 것으로 기대됩니다.



### DeepSPV: An Interpretable Deep Learning Pipeline for 3D Spleen Volume Estimation from 2D Ultrasound Images (https://arxiv.org/abs/2411.11190)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2308.08038

- **What's New**: 이번 연구에서는 DeepSPV라는 딥 러닝 기반의 파이프라인을 소개합니다. 이 파이프라인은 단일 또는 이중 2D 초음파 이미지를 통해 정확한 비장 부피를 추정할 수 있도록 설계되었습니다. 기존의 CT나 MRI와 같은 3D 영상 기술의 필요성을 줄여 지역적으로 접근성이 떨어지는 곳에서도 적용할 수 있는 가능성을 제시합니다.

- **Technical Details**: DeepSPV는 비장을 자동으로 세분화하는 세그멘테이션 네트워크와 추정된 세그멘테이션으로부터 저차원 표현을 학습하는 변분 오토인코더(variational autoencoder)로 구성되어 있습니다. 이 시스템은 비장 부피 추정을 위한 세 가지 접근 방식을 조사했으며, 최상의 모델은 단일 및 이중 시점 설정에서 각각 86.62% 및 92.5%의 평균 상대 부피 정확도를 달성했습니다.

- **Performance Highlights**: 이 연구에서 제기된 DeepSPV는 2D 초음파 이미지를 활용한 비장 부피 추정 분야의 최초의 딥 러닝 기반 모델로, 흥미로운 성과를 통해 인간 전문가의 성과를 초과하는 결과를 보였습니다. 추가적으로, 이 시스템은 추정된 부피에 대한 신뢰 구간을 제공하고, 임상적 의사 결정 지원에 유용한 해석 가능성을 갖추고 있습니다.



### Freqformer: Frequency-Domain Transformer for 3-D Visualization and Quantification of Human Retinal Circulation (https://arxiv.org/abs/2411.11189)
- **What's New**: Freqformer는 상업용 광간섭 단층 촬영 혈관조영술(OCTA) 스캔에서 인체 망막 순환을 3D 고해상도로 시각화할 수 있는 혁신적인 Transformer 기반 아키텍처로 소개됩니다. 이 모델은 낮은 신호 대 잡음 비율(SNR) 문제를 극복하기 위해 복소수 주파수 도메인 모듈(Complex-Valued Frequency-Domain Module, CFDM)과 단순화된 다중 주의 메커니즘(Simplified Multi-Head Attention, Sim-MHA)을 사용합니다. Freqformer는 망막 혈관 구조를 정확하게 재구성하고, 망막 혈관 질환 진단 및 관리에 활용될 수 있는 잠재적인 임상 애플리케이션을 제시합니다.

- **Technical Details**: Freqformer는 4단계의 대칭 인코더-디코더 아키텍처를 채택하여 단일 OCTA 이미지의 3D 조직을 처리합니다. 이 모델은 혈관의 해상도와 연결성을 크게 향상시키기 위해 주파수 정보를 활용하는 CFDM 모듈을 도입하며, 공간 복잡성을 줄이기 위해 단순화된 다중 주의와 깊이별 합성곱 피드포워드 네트워크(Deep-Wise Convolution Feed-Forward Network, DFFN)를 통합합니다. 이를 통해 기존의 CNN 및 Transformer 기반 모델을 초월하는 성능을 발휘하며, 더 큰 시야에서도 일반화 가능성을 보여줍니다.

- **Performance Highlights**: Freqformer는 기존의 CNN 및 Transformer 기반 방법에 비해 피크 신호 대 잡음 비율(PSNR), 구조 유사성 지수 측정(SSIM), 학습된 지각 이미지 패치 유사성(LPIPS)에서 우수한 성능을 냅니다. 또한, 6×6 mm² 및 12×12 mm²의 더 넓은 시야를 가진 스캔에서 낮은 밀도 이미지를 효과적으로 개선했습니다. 이러한 결과는 Freqformer가 망막 순환의 이해와 특성을 크게 향상시킬 수 있음을 나타내며, 임상 진단 능력을 발전시킬 수 있는 가능성을 제시합니다.



### RPN 2: On Interdependence Function Learning Towards Unifying and Advancing CNN, RNN, GNN, and Transformer (https://arxiv.org/abs/2411.11162)
Comments:
          105 pages, 37 figures, 6 tables, preprint version

- **What's New**: 본 논문은 Reconciled Polynomial Network (RPN)의 이전 연구를 바탕으로 하여, RPN 2라는 새로운 모델을 소개합니다. RPN 2는 데이터의 상호 의존성(data interdependence)을 명시적으로 모델링하기 위해 새로운 구성 요소로 상호 의존성 함수(interdependence functions)를 도입하여, 복잡한 데이터의 함수 학습 작업을 개선하는 것을 목표로 합니다.

- **Technical Details**: RPN 2는 데이터 인스턴스와 속성 간의 상호 의존성을 모델링하기 위해 그리드 기반 기하 구조 상호 의존성 함수를 활용합니다. 이 함수는 입력 데이터 배치를 기반으로 인터디펜던스 행렬(interdependence matrices)을 생성하여, 시간과 공간 면에서 계산 효율성을 최적화합니다. RPN 2는 CNN, RNN, GNN, 그리고 Transformers와 같은 널리 사용되는 백본 모델(backbone models)을 포괄하는 통합 표현을 제공합니다.

- **Performance Highlights**: RPN 2는 MNIST 및 CIFAR-10과 같은 벤치마크에서 이미지 패치 간의 지역적 상호 의존성을 효과적으로 캡처하여 우수한 성능을 기록했습니다. 또한, 언어 분류 및 시계열 예측과 같은 작업에서도 뛰어난 성능을 발휘합니다. RPN 2 모델을 기반으로 한 새로운 백본 모델들은 기존의 주도적인 백본 아키텍처와 비교할 때 효과적이고 우수한 성능을 보여주었습니다.



### DBF-Net: A Dual-Branch Network with Feature Fusion for Ultrasound Image Segmentation (https://arxiv.org/abs/2411.11116)
- **What's New**: 이번 연구에서는 초음파 이미지의 병변 분할을 개선하기 위한 UBBS-Net 모델을 제안합니다. 이 네트워크는 신체(Body)와 경계(Boundary)의 관계를 학습하여 보다 정확한 분할을 구현합니다. 또한, 새로운 Feature Fusion 모듈을 도입하여 이러한 두 정보를 통합하여 성능을 향상시키고 있습니다.

- **Technical Details**: UBBS-Net은 전통적인 인코더-디코더 구조를 기반으로 하여 다중 레벨 정보 캡처를 위해 ASP_OC 블록을 사용합니다. 네트워크는 두 개의 Feature Fusion 및 Supervision 블록으로 구성되어 있으며, 경계와 신체의 정보를 각각 추출합니다. 이러한 설계는 개별 피처의 상호작용을 가능하게 하여 분할 성능을 극대화합니다.

- **Performance Highlights**:  UBBS-Net은 세 가지 공개 데이터셋에서 기존 방법들보다 우수한 성능을 보여주었으며, 유방암과 팔신경총, 소아 혈관종의 분할에서 각각 81.05%, 76.41%, 87.75%의 Dice 유사도 계수를 달성했습니다. 이 결과는 해당 네트워크의 효과성을 잘 나타내며, 초음파 이미지 분할 분야에서 최신 기술로 자리잡을 가능성을 보여줍니다.



### Retinal Vessel Segmentation via Neuron Programming (https://arxiv.org/abs/2411.11110)
- **What's New**: 본 논문은 네트워크 설계에서 'neuron programming'이라는 새로운 접근법을 소개하여, 다양한 신경 유형을 네트워크에 자동으로 통합하는 방법을 제안합니다. 이는 Neural Architecture Search (NAS)와 함께 작동하여 모델의 표현 능력을 향상시키며, 특히 망막 혈관 세분화 작업에 효과적입니다.

- **Technical Details**: 이 방법은 U-Net 아키텍처에 전통적인 뉴런과 쿼드러틱 뉴런을 통합하여 두 종류의 신경 유형을 활용합니다. 진화 알고리즘을 사용하여 네트워크 내 각 위치에 적합한 뉴런 유형을 결정하며, hypernetwork를 도입하여 최적의 신경 구성을 예측합니다.

- **Performance Highlights**: 포괄적인 실험을 통해 neuron programming이 망막 혈관 세분화에서 경쟁력 있는 성능을 달성했음을 확인했으며, 다양한 데이터 세트에서 탁월한 정확성과 효율성을 보여주었습니다.



### STOP: Spatiotemporal Orthogonal Propagation for Weight-Threshold-Leakage Synergistic Training of Deep Spiking Neural Networks (https://arxiv.org/abs/2411.11082)
Comments:
          13 pages (exclude supplementary), 5 figures

- **What's New**: 본 연구에서는 기본적인 딥 신경망 학습 알고리즘의 제한으로 인해 실용적인 엣지(edge) 배치에서 저조한 성능을 겪고 있던 스파이킹 신경망(spiking neural networks, SNN)용 효율적이고 고정밀 딥 SNN 학습 알고리즘을 제안합니다. 제안된 스페이셔 템포럴 오소돈탈 전파(spatiotemporal orthogonal propagation, STOP) 알고리즘은 시냅스 가중치 및 발사 임계값, 누출 계수를 동시 학습하여 SNN의 정확도를 향상시키며, 메모리 요구량을 줄여줍니다.

- **Technical Details**: STOP 알고리즘은 신경망의 모든 시간 단계에 대한 상태를 저장하는 거대한 메모리 요구량을 완화하기 위해 통일된 시간 순방향 추적(trace-based) 프레임워크를 사용합니다. 이 알고리즘은 시공간적으로 역전달된 신경 에러와 시간적으로 순진행된 추적이 서로 독립적으로 직각으로 전파되도록 하여 계산 오버헤드를 크게 줄입니다. 이로 인해 신경망 훈련 과정에서 효율성이 크게 향상됩니다.

- **Performance Highlights**: STOP 알고리즘은 MNIST, CIFAR-10, CIFAR-100, DVS-Gesture, DVS-CIFAR10 데이터셋에서 각각 99.53%, 94.84%, 74.92%, 98.26%, 77.10%의 높은 인식 정확도를 달성했습니다. 이 연구는 LeNet-5부터 ResNet-18까지 중간 규모의 SNN을 사용하여 수행되었습니다. 제안된 방법은 제한된 자원 속에서도 현장(in-situ) 학습을 통해 높은 정확성을 요구하는 엣지 지능형 시나리오에 더 적합합니다.



### AppSign: Multi-level Approximate Computing for Real-Time Traffic Sign Recognition in Autonomous Vehicles (https://arxiv.org/abs/2411.10988)
- **What's New**: 이 논문에서는 자율주행차의 실시간 교통 표지 인식에 대한 multi-level approximate computing 방법인 AppSign을 제시합니다. AppSign은 정확도와 계산 비용 사이의 균형을 맞추며, CNN 기반의 교통 표지 인식 유닛에 적용됩니다. 이를 통해 깊이 학습 알고리즘의 계산 집약적인 특성 속에서도 실시간 처리가 가능하도록 합니다.

- **Technical Details**: AppSign은 convolution 연산의 정확도를 유지하며 복잡성을 줄이는 "TIRuD"라는 새로운 근사 곱셈 방법을 도입합니다. 또한 다양한 수준의 근사화를 통해 CNN의 적응적 근사화가 이루어지는 동시에 과도한 계산 비용과 응답 시간을 줄입니다. 이는 효율적인 이미지 처리 애플리케이션을 가능하게 하고, 자율주행차의 교통 표지 인식 유닛에 효과적으로 적용됩니다.

- **Performance Highlights**: 실험 결과, TIRuD는 정확도를 약 10% 감소시키는 대신 실행 시간을 약 64% 절약하며, 다양한 모델 레이어를 적용한 경우 정확한 계산을 27.78% 초과하는 성능을 보여주었습니다. 이러한 결과는 AppSign의 유용성을 실증적으로 증명하며, 적절한 근사화 방법을 조합하여 높은 정확도와 낮은 계산 비용을 달성할 수 있음을 확인했습니다.



### SageAttention2 Technical Report: Accurate 4 Bit Attention for Plug-and-play Inference Acceleration (https://arxiv.org/abs/2411.10958)
- **What's New**: SageAttention2는 4비트 행렬 곱셈을 사용하여 계산 효율성을 크게 개선하고, 정확성을 유지하며 3배 이상 속도를 향상시켰습니다.

- **Technical Details**: SageAttention2는 원래의 SageAttention에서 INT8에서 INT4로의 양자화와 FP16에서 FP8로의 개선을 통해 Q, K를 INT4로, P, V를 FP8로 양자화합니다. 특정 어려운 레이어에 대해 혼합 정밀도(adaptive mix precision)를 적용하여 정확성을 유지합니다.

- **Performance Highlights**: RTX4090에서 SageAttention2는 485 TOPS의 최고 성능을 달성하며, FlashAttention2에 비해 약 3.1배, xformers에 비해 약 5.4배 더 빠른 성능을 보였습니다.



### Towards Accurate and Efficient Sub-8-Bit Integer Training (https://arxiv.org/abs/2411.10948)
- **What's New**: 본 논문은 sub-8-bit integer training을 통해 딥러닝에서의 고효율과 높은 정확성을 동시에 달성하고자 한다. 새로운 데이터 형식과 전처리 작업을 개발하여 양자화 오류를 줄이고, ShiftQuant와 L1 normalization을 도입하여 고성능의 신경망 훈련이 가능하게 한다.

- **Technical Details**: 제안하는 ShiftQuant는 정확한 gradient estimation을 수행하며, L1 normalization은 손실 경관을 부드럽게 하는 데 기여한다. ShiftQuant는 그룹 양자화의 이론적 상한에 근접한 성능을 달성하고, 메모리 재배열의 비효율성을 극복한다. 제안된 L1 normalization layers는 전통적인 L2 normalization layers에 비해 추가적인 계산 없이 더 나은 성능을 제공한다.

- **Performance Highlights**: 실험 결과, 4-bit ResNets에서 $0.92\%$, 6-bit Transformers에서 $0.61\%$의 미지근한 정확도 손실을 보였다. ShiftQuant의 프로토타입 구현은 CPU/GPU에서 FP16 대비 각각 1.85배 및 15.3% 성능 개선을 보였고, FPGA에서는 자원 소비가 33.9% 감소하였다.



### A Monocular SLAM-based Multi-User Positioning System with Image Occlusion in Augmented Reality (https://arxiv.org/abs/2411.10940)
- **What's New**: 최근 증강 현실(AR) 기술의 급격한 발전으로 인해 다중 사용자 협업 경험에 대한 요구가 증가하고 있습니다. 이 논문에서는 Unity 3D 게임 엔진을 기반으로 단안 RGB 이미지를 사용하여 ORB-SLAM2 기반의 다중 사용자 로컬라이제이션 시스템을 제안합니다. 이 시스템은 사용자 로컬라이제이션을 수행할 뿐 아니라 환경 내 평면 표면에 공통 가상 객체를 배치함으로써 모든 사용자가 해당 객체를 올바른 시점에서 볼 수 있도록 합니다.

- **Technical Details**: 제안된 시스템은 세 가지 주요 모듈로 구성됩니다: 로컬라이제이션 모듈, 평면 추정 모듈, 협조 서버입니다. 로컬라이제이션 모듈은 RGB 이미지를 사용하여 사용자 카메라 자세를 정확하게 추정하며, 이는 Unity에서 가상 객체의 렌더링에 활용됩니다. 또한, 딥러닝 모델을 통해 각 프레임의 깊이 맵을 추정하고 occlusion 문제를 처리하여 AR 경험의 현실감을 높입니다.

- **Performance Highlights**: 이 연구의 주요 성과는 다양한 장소에 있는 여러 사용자가 AR 환경 내에서 원활하게 상호작용할 수 있도록 하는 것입니다. 제안된 시스템은 사용자 간의 위치 정보 동기화를 이루는 데 초점을 맞추고 있으며, 이는 가상 객체를 기반으로 모든 사용자에게 다른 사용자의 상대적인 위치를 제공합니다. 또한, 단안 카메라를 기반으로 하여 기존의 SLAM 기술의 한계를 극복하며, 사용자 경험을 강화하고 자연스러운 가상 객체 표현을 제안합니다.



### Constrained Diffusion with Trust Sampling (https://arxiv.org/abs/2411.10932)
Comments:
          18 pages, 6 figures, NeurIPS

- **What's New**: 이 논문에서는 기존의 loss-guided diffusion 접근 방식을 최적화 관점에서 재구성하여, challenging constraint를 만족하는 새로운 방법인 Trust Sampling을 제안합니다.

- **Technical Details**: Trust Sampling은 각 diffusion step을 독립적인 최적화 문제로 취급하고, proxy constraint function을 따라 여러 gradient steps를 허용합니다. 최적화가 더 이상 신뢰할 수 없을 때 조기 종료를 할 수 있도록 상태 다양성을 추정합니다.

- **Performance Highlights**: Trust Sampling 방법은 다양한 이미지 및 3D 동작 생성 작업에서 기존 방법들에 비해 상당한 품질 향상을 보여주며, constraint를 더 잘 만족한다는 것을 입증했습니다.



### Distributed solar generation forecasting using attention-based deep neural networks for cloud movement prediction (https://arxiv.org/abs/2411.10921)
- **What's New**: 이 논문은 클라우드 이미지를 이용한 분산형 태양광 발전 예측을 위한 주목(attention) 기반의 합성곱 장기 단기 기억 네트워크(convolutional long short-term memory network)를 제안합니다.

- **Technical Details**: 클라우드 커버의 변화를 초 단위로 포착하는 이미지를 활용하여 태양광 발전 예측 문제를 해결하고자 합니다. 주목(attention) 메커니즘이 적용된 딥 뉴럴 네트워크(deep neural networks)를 사용하여 클라우드 이동 예측에 대한 연구가 진행됩니다.

- **Performance Highlights**: 고고도(altitude) 클라우드의 경우, 주목 기반 방법을 사용한 클라우드 예측이 비주목(non-attention) 기반 방법에 비해 태양광 예측 기술 점수를 5.86% 이상 개선하는 결과를 보여주었습니다.



### Multi-Modal Self-Supervised Learning for Surgical Feedback Effectiveness Assessmen (https://arxiv.org/abs/2411.10919)
Comments:
          Accepted as a spotlight proceedings paper at Machine Learning for Health 2024

- **What's New**: 본 연구에서는 외과 훈련에서 트레이너의 실시간 피드백이 트레이니의 행동 변화에 미치는 효과를 예측하기 위해, 텍스트와 비디오 정보를 통합하는 다중 모달 접근 방식을 제안합니다. 이전의 수작업 분석 방식에서 벗어나 자동화된 시스템을 통해 피드백 효과성을 객관적이고 확장 가능하게 평가할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 우리는 수술 중 피드백의 오디오 기록을 전사하고, Sentence Transformers (SBERT)을 사용하여 전사된 내용을 임베딩으로 변환합니다. 시각적 측면에서는 비디오 마스크 오토인코더(VideoMAE)를 사용해 피드백이 제공될 때의 수술 비디오에서 정보를 추출합니다. Self-supervised learning (SSL) 방식의 fine-tuning을 통해 모델 성능을 향상시킵니다.

- **Performance Highlights**: 본 연구의 결과, 트레이니 행동 변화 예측에서 ROC 곡선 아래 면적(AUC)이 0.70 ± 0.02를 달성했습니다. 텍스트 피드백과 수술 비디오를 결합할 경우 예측 정확도가 최대 6.6% 향상되었습니다. 이는 수술 피드백의 효과성을 자동으로 예측할 수 있는 시스템의 가능성을 보여줍니다.



### Targeting Negative Flips in Active Learning using Validation Sets (https://arxiv.org/abs/2411.10896)
Comments:
          Presented at the IEEE International Conference on Big Data 2024, Washington DC, USA

- **What's New**: 이 논문에서는 Active Learning 알고리즘의 성능을 향상시키기 위한 두 가지 방법을 제안합니다. 첫 번째 방법은 테스트 세트 내의 전체 오류율을 줄이는 것이고, 두 번째 방법은 훈련 세트가 증가할 때 올바른 예측이 잊혀지지 않도록 하는 것입니다. 두 가지 접근 방식 사이의 관계 및 ROSE라는 새로운 알고리즘을 개발하여 이를 해결하려는 노력을 다룹니다.

- **Technical Details**: 우리는 기존의 Active Learning 프레임워크에 회귀(Regression) 관련 요소를 통합하는 방법을 제안합니다. ROSE(Regression on Subset of Errors) 알고리즘은 작은 라벨링된 검증 세트를 사용하여 비라벨의 샘플 중에서 부정적 회전을 타겟팅하는 액티브 러닝 수집 함수(acquisition function)를 제한합니다. 이로써 우리의 방법은 정확배송 및 부정적 회전(negative flip)을 동시에 개선할 수 있습니다.

- **Performance Highlights**: ROSE는 다양한 데이터 세트(CINIC10 포함), 아키텍처, 그리고 수집 방식(acquisition function)에서 성능을 검증하였고, 정확도와 부정적 회전 수치를 효과적으로 증가시키거나 줄이는 것을 보여줍니다.



### MpoxVLM: A Vision-Language Model for Diagnosing Skin Lesions from Mpox Virus Infection (https://arxiv.org/abs/2411.10888)
Comments:
          Accepted by ML4H 2024

- **What's New**: 본 논문에서는 MpoxVLM이라는 비전-언어 모델(Vision-Language Model)을 제안합니다. 이는 Mpox(원숭이 두창) 감염을 진단하기 위해 피부 병변 이미지와 환자의 임상 정보를 분석하는 데 특화되어 있습니다. MpoxVLM은 CLIP 비주얼 인코더, 강화된 비전 트랜스포머(ViT) 분류기 및 LLaMA-2-7B 모델을 통합하여 이전 모델들보다 더 높은 진단 정확도를 달성합니다.

- **Technical Details**: MpoxVLM은 새롭게 수집된 mpox 피부 병변 데이터셋에서 시각적 설명에 따라 질문-답변을 학습하여 훈련되었습니다. 논문에서는 2,914개의 샘플로 구성된 새로운 멀티모달 데이터셋을 통해 다양한 피부 타입, 성별, 나이, 병변 부위에 대한 모델의 효과성을 평가하고 있습니다. 이 모델은 고감도의 자동 검출 및 조기 진단의 가능성을 제공합니다.

- **Performance Highlights**: 제안하는 MpoxVLM은 mpox 감지에서 90.38%의 정확도를 달성했습니다. 이는 기존의 최신 기술들과 비교할 때 최상의 성능을 보이며, 다양한 인구집단에 대한 정확한 진단을 위한 가능성을 제시합니다. 이와 같은 딥러닝 기반의 컴퓨터 지원 진단 방법은 향후 mpox 통제에 크게 기여할 것으로 기대됩니다.



### A Novel Adaptive Hybrid Focal-Entropy Loss for Enhancing Diabetic Retinopathy Detection Using Convolutional Neural Networks (https://arxiv.org/abs/2411.10843)
Comments:
          9 pages,7 figures

- **What's New**: 이 논문에서는 당뇨병성 망막병증(diabetic retinopathy)의 진단을 위한 새로운 AI 기반 도구를 제안합니다. 특히, 기존의 다중 클래스 분류에서 사용되는 손실 함수(loss function)인 Categorical Cross-Entropy (CCE)의 한계를 극복하기 위해 Adaptive Hybrid Focal-Entropy Loss (AHFE)를 도입했습니다.

- **Technical Details**: AHFE는 집중 손실(focal loss)과 엔트로피 손실(entropy loss)을 결합하여 적응적 가중치(adaptive weighting)를 적용합니다. 이를 통해 소수 클래스(minority classes)에 집중하고 어려운 샘플들을 강조하는 방식으로 작동합니다. 이러한 접근 방식은 클래스 불균형(class imbalance) 문제를 해결하는 데 중요한 역할을 합니다.

- **Performance Highlights**: AHFE를 적용한 당뇨병성 망막병증 탐지(state-of-the-art models) 결과는 크게 향상된 성능을 보여줍니다. ResNet50은 99.79%, DenseNet121은 98.86%, Xception은 98.92%, MobileNetV2는 97.84%, InceptionV3는 93.62%의 정확도를 기록하여 복잡하고 불균형한 의료 데이터셋에 대한 AI 기반 진단의 향상을 보여줍니다.



### Neighboring Slice Noise2Noise: Self-Supervised Medical Image Denoising from Single Noisy Image Volum (https://arxiv.org/abs/2411.10831)
- **What's New**: 최근 몇 년 동안 의료 이미지 품질 향상에 대한 요구가 증가하면서, 본 논문에서는 새로운 자가 지도 학습 방식의 의료 이미지 디노이징 방법인 Neighboring Slice Noise2Noise (NS-N2N)을 제안합니다. 기존의 방법들은 노이즈가 픽셀 단위에서 독립적이라는 가정에 크게 의존하지만, 많은 실제 의료 영상에서는 이러한 가정이 성립하지 않음을 지적합니다. 본 방법은 서로 인접한 슬라이스(슬라이스)들을 활용하여 훈련 데이터를 구성하고, 전통적인 데이터 분할 방식의 한계를 극복합니다.

- **Technical Details**: NS-N2N 방법은 단일 노이즈가 있는 이미지 볼륨 내에서 인접한 슬라이스를 활용하여 가중치가 적용된 훈련 데이터를 생성합니다. 그런 다음 지역 일관성 손실(regional consistency loss) 및 슬라이스 간 연속성 손실(inter-slice continuity loss)을 통해 디노이징 네트워크를 자가 지도 방식으로 훈련합니다. 이 방법은 특정 장비의 기하학적 문제에 영향을 받지 않으며, 다양한 임상 환경에 쉽게 적용될 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에서 NS-N2N은 최신 자가 지도 디노이징 방법들과 비교하여 우수한 디노이징 성능과 처리 효율성을 보여주었습니다. 이 연구는 단일 노이즈 이미지만으로도 고품질의 디노이징을 가능하게 하여 실제 임상 응용 분야에서 큰 장점을 제공합니다. 이러한 성과는 다가오는 미래의 의료영상 처리 방법에 중요한 영향을 미칠 것으로 기대됩니다.



### FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations (https://arxiv.org/abs/2411.10818)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 기존의 전통적인 스케치 애니메이션 방식의 단점을 해결하기 위해 FlipSketch라는 시스템을 제안합니다. FlipSketch는 사용자가 간단히 아이디어를 스케치하고 그 움직임을 설명함으로써 애니메이션을 생성할 수 있도록 합니다. 또한, 이 시스템은 text-to-video diffusion 모델을 활용하여 스케치 스타일의 애니메이션을 생성하는 혁신적인 접근 방식입니다.

- **Technical Details**: FlipSketch의 주요 기술 혁신으로는 세 가지가 있습니다. 첫째, synthetic sketch animations에 대한 text-to-video diffusion model의 파인튜닝을 통해 일관성 있는 선 그림 시퀀스를 생성할 수 있습니다. 둘째, 입력 스케치의 비주얼 무결성을 보존하기 위해 DDIM 인버전을 기반으로 한 참조 프레임 메커니즘을 도입하였으며, 셋째, 두 가지 주의(attention) 메커니즘을 활용하여 생성된 애니메이션의 정체성과 움직임의 진실성을 보장합니다.

- **Performance Highlights**: FlipSketch는 기존의 벡터 기반 애니메이션 방식과는 달리, 역동적인 레스터 스케치를 지원하여 전통 애니메이션의 표현 자유를 포착합니다. 이 시스템은 스케치 애니메이션을 단순한 낙서와 설명만으로 생성할 수 있게 하여, 사용자가 보다 직관적으로 창의적인 작업을 할 수 있도록 돕습니다. FlipSketch는 자동화를 더욱 쉽게 만들어주며, 예술적인 손으로 그린 애니메이션의 본질을 유지하는 데 중점을 두고 있습니다.



### Unveiling Hidden Details: A RAW Data-Enhanced Paradigm for Real-World Super-Resolution (https://arxiv.org/abs/2411.10798)
- **What's New**: 이 논문에서는 Real image super-resolution(Real SR)의 새로운 접근 방식으로 LR RAW 데이터를 활용하여 기존 RGB 방법들의 한계를 극복하고자 합니다. 기존의 Real SR 방법들은 LR RGB 도메인에서의 디테일 생성에 주로 집중했지만, RAW 데이터의 숨겨진 디테일을 이용해 고해상도 이미지를 생성함으로써 더 풍부하고 정밀한 결과를 도출할 수 있음을 보여줍니다. 연구진은 RealSR-RAW라는 10,000쌍의 LR 및 HR RGB 이미지 데이터셋을 소개하여 LR RAW 이미지의 유용성을 뒷받침하고 있습니다.

- **Technical Details**: 실험에서는 LR RAW 데이터와 LR RGB 데이터를 결합하여 HR RGB 이미지를 생성하는 방식을 채택했습니다. 연구진은 CNN, Transformer, 그리고 Diffusion 기반의 기존 Real SR 모델에 대해 LR RAW 데이터를 통합하여 상관 특징의 분포를 맞추고 노이즈를 억제하는 일반적인 RAW 어댑터를 제안했습니다. 이러한 접근 방식은 PSNR과 SSIM에서 각각 1.109 dB 및 0.038의 개선을 이루었으며, 더 풍부하고 높은 품질의 디테일을 가진 이미지를 계속해서 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, LR RAW 데이터를 통합함으로써 Real SR 성능이 현저히 향상되었으며, 특히 디테일 복원이 크게 개선되었습니다. 10개의 평가 지표(즉, 충실도와 지각 중심 지표)를 포함하여 여러 성능 지표에서 두드러진 결과를 확인했습니다. 연구진은 LR RAW 데이터의 조합이 Real SR 과제에 있어 새로운 길을 열어줄 것으로 기대하고 있으며, 이 데이터셋과 코드가 향후 연구에 도움이 될 것이라고 강조합니다.



### Beyond Feature Mapping GAP: Integrating Real HDRTV Priors for Superior SDRTV-to-HDRTV Conversion (https://arxiv.org/abs/2411.10775)
Comments:
          8 pages,4 figures

- **What's New**: 이 논문에서는 SDRTV(표준 동적 범위 TV)에서 HDRTV(고동적 범위 TV)로 변환하는 새로운 방법을 제안합니다. 기존의 방법들이 단일 스타일 매핑에 집중하는 것과 달리, 본 연구는 실제 HDRTV 프라이어를 사용하여 변환을 유도합니다. 이를 통해 고차원 헷갈린 문제를 더욱 구체화하여, 예측 문제를 해결하는 대신 참조된 선택 문제로 전환합니다. 이 접근 방식은 변환 과정의 정확성과 신뢰성을 크게 향상시킵니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 벡터 양자화 생성 적대 신경망(Vector Quantized Generative Adversarial Network, VQGAN)을 사용하여 HDRTV 프라이어를 캡처합니다. 두 번째 단계에서는 이 프라이어를 입력 SDRTV 콘텐츠에 매칭하여 현실적인 HDRTV 출력을 복원합니다. 이러한 프로세스는 전통적인 방법으로 발생할 수 있는 색상 부정확성과 밝기 비정상성을 해결하는 데 기여합니다.

- **Performance Highlights**: 공식 및 주관적 메트릭에서 메르라이어는 여러 공공 데이터셋을 통해 평가되었으며, 기존 방법 대비 상당한 성능 향상을 입증했습니다. 이 논문에서 제안된 메트릭인 LPHPS(학습된 인지 HDRTV 패치 유사도), NHQE(자연 HDRTV 품질 평가자), FHAD(프레셋 초기 거리)가 HDRTV 콘텐츠의 주관적 품질 평가를 향상시킵니다. 이러한 성과는 연구자와 실무자에게 신뢰할 수 있는 HDRTV 품질 평가 도구를 제공합니다.



### An End-to-End Real-World Camera Imaging Pipelin (https://arxiv.org/abs/2411.10773)
Comments:
          accept by ACMMM 2024

- **What's New**: 이 논문은 RealCamNet이라는 엔드 투 엔드(End-to-End) 카메라 이미징 파이프라인을 제안합니다. 이 시스템은 전통적인 이미지 신호 처리(ISP) 설계를 넘어 통합 최적화를 통해 효율성을 극대화 합니다. 새로운 방식으로 RAW 이미지를 RGB 로 변환하고 압축하며, 복잡한 수동 튜닝 없이도 최상의 결과를 제공합니다.

- **Technical Details**: RealCamNet은 Coordinate-Aware Distortion Restoration (CADR) 모듈과 Coordinate-Independent Mapping Compression (CIMC) 모듈을 포함합니다. CADR 모듈은 좌표 편향 왜곡을 복원하고, CIMC 모듈은 톤 매핑 및 불필요한 정보 압축을 수행합니다. 시스템은 전통적인 별도 프로세스에서 생기는 오류 축적을 방지하며, 각 모듈 간 주변적 관점을 통해 통합 최적화를 달성합니다.

- **Performance Highlights**: RealCamNet은 새로운 실세계 이미징 데이터셋을 사용하여 훈련되었으며, 낮은 추론 지연과 함께 최고의 비율-왜곡(rate-distortion) 성능을 달성했습니다. 이 연구는 실제 카메라 이미지 파이프라인에서의 효과적인 이미지와 압축 기술을 제공하여 산업 발전에 기여할 것입니다.



### MRI Parameter Mapping via Gaussian Mixture VAE: Breaking the Assumption of Independent Pixels (https://arxiv.org/abs/2411.10772)
Comments:
          NeurIPS 2024 Workshop in Machine Learning and the Physical Sciences

- **What's New**: 새로운 패러다임을 제안하여 MRI의 양적 파라미터 매핑(quantitative parameter mapping) 방식을 혁신적으로 변화시킵니다. 기존의 모델 피팅 방법은 각 voxel이 독립적이라는 가정을 해왔으나, 본 연구는 데이터 내의 중복성을 활용하여 이 가정을 깨뜨립니다.

- **Technical Details**: 자기 지도(self-supervised) 딥 변분(deep variational) 접근 방식을 사용하여, voxel 간의 의존성을 고려한 데이터 주도(data-driven) 정규화(regularisation)를 통해 양적 맵을 계산합니다. 이 방법은 Gaussian 혼합 사전(Gaussian mixture prior)을 활용하여 더욱 선명한 양적 맵을 생성합니다.

- **Performance Highlights**: dMRI 시뮬레이션 및 실제 데이터에서 기존 모델 피팅 기술보다 더 나은 성능을 보여주며, 미세한 해부학적 세부 사항을 드러냅니다. 이러한 접근 방식은 dMRI 및 qMRI와 같은 파라미터 매핑 방법의 임상 도입을 지원할 수 있습니다.



### Diffusion-Based Semantic Segmentation of Lumbar Spine MRI Scans of Lower Back Pain Patients (https://arxiv.org/abs/2411.10755)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 5 pages

- **What's New**: 이번 연구는 Magnetic Resonance Imaging (MRI) 스캔을 이용하여 허리 통증 (Low Back Pain, LBP) 환자의 척추, 추간판 (Intervertebral Discs, IVDs), 척추관을 견고하고 정확하게 분할하는 확산 기반 프레임워크(SpineSegDiff)를 소개합니다. 연구 결과, SpineSegDiff는 퇴화된 IVD를 식별하는데 있어 기존의 비확산 최첨단 모델들을 초월한 성능을 보였습니다.

- **Technical Details**: SpineSegDiff 모델은 2D diffusion 기반의 분할 모델로, LBP 환자의 MRI 스캔에서의 병리학적 변동성을 다루는 데 중점을 두고 있습니다. 이 모델은 T1 및 T2 가중치 MRI 스캔을 하나의 모델 내에서 처리할 수 있도록 설계되었습니다. 학습 과정에서 MSE denoising loss와 Dice Loss, Binary Cross-Entropy Loss를 결합한 복합 손실 함수를 사용하여 예측된 세분화 마스크와 실제 마스크 간의 차이를 효과적으로 벌점화합니다.

- **Performance Highlights**: 실험 결과, 218명의 환자에서 수집된 측면 MRI를 통해 SpineSegDiff의 성능이 검증되었습니다. 결과는 spondylolisthesis와 disc narrowing이 세분화 성능에 유의미한 영향을 미치며, 특히 spondylolisthesis는 SC, VB, IVD 세분화 점수에 폭넓은 영향을 미치는 것을 보여주었습니다. 본 연구는 확산 모델이 LBP 진단 및 관리를 개선할 수 있는 잠재력을 강조합니다.



### Towards a Comprehensive Benchmark for Pathological Lymph Node Metastasis in Breast Cancer Sections (https://arxiv.org/abs/2411.10752)
- **What's New**: 이 논문은 Camelyon-16 및 Camelyon-17 데이터셋에서 1,399개의 Whole Slide Images (WSIs)와 레이블을 재처리하여, 저품질 슬라이드를 제거하고 잘못된 레이블을 수정했습니다. 또한, 이전에 발표되지 않은 테스트 세트의 종양 지역에 대해 전문가의 픽셀 주석을 제공하여 비트 레이블 시스템을 4개 클래스로 확장했습니다: negative, micro-metastasis, macro-metastasis, 그리고 Isolated Tumor Cells (ITC). 이 연구는 고품질의 병리 이미지 데이터셋이 AI 기반 진단 지원에 필수적임을 강조합니다.

- **Technical Details**: Camelyon-16 데이터셋은 399개의 WSIs로 구성되어 있으며, 270개는 훈련용, 129개는 테스트용으로 나누어져 있습니다. 논문에서 팀은 품질이 낮거나 치료 관련 물질로 인한 인식 오류가 있는 슬라이드를 제거하고, 병리학적 데이터에서 강력한 특징 추출 기능을 나타내는 다양한 instance learning (MIL) 방법을 활용하여 재평가했습니다. 데이터 정리 후 업데이트된 Camelyon+ 데이터셋은 총 1,350개의 WSIs를 포함하며, 4개의 분류 레이블로 구성됩니다.

- **Performance Highlights**: 이 논문은 여러 최신 MIL 방법을 사용하여 청소된 데이터셋을 평가하였고, AI 개발을 촉진하기 위한 벤치마크를 제공했습니다. 연구진은 최신의 Pathology feature extractors와 MIL 방법을 재평가하기 위해 정제된 데이터셋을 사용하였으며, 이를 통해 데이터 품질이 향상되는 동시에 정확한 평가가 가능함을 입증했습니다. 이로 인해 하위 분석 작업에서의 성능이 크게 향상되었습니다.



### A Wearable Gait Monitoring System for 17 Gait Parameters Based on Computer Vision (https://arxiv.org/abs/2411.10739)
Comments:
          13 pages, 14 figures. This paper was submitted for publication to the IEEE Transactions on Instrumentation and Measurement

- **What's New**: 이 논문에서는 최대 17개의 보행(지속적인 걸음) 매개변수를 추적할 수 있는 신발 부착형 보행 모니터링 시스템을 개발하였습니다. 이 시스템은 한 신발에 장착된 스테레오 카메라를 사용해 반대쪽 신발에 부착된 마커를 추적하여 공간 보행 매개변수를 추정합니다. 또한, 신발 뒤꿈치에 부착된 Force Sensitive Resistor (FSR)와 맞춤형 알고리즘을 통해 시간적 보행 매개변수를 측정합니다.

- **Technical Details**: 하드웨어 구현은 두 개의 Raspberry Pi Zero 2 W 마이크로컨트롤러와 대형 FoV IR 카메라를 결합하여 이루어졌습니다. FSR이 발뒤꿈치에 장착되어 힐 스트라이크 단계에서 스테레오 카메라가 반대쪽 신발의 마커 이미지를 캡처하며 시간 스탬프를 기록합니다. 이 시스템은 유선 및 무선 통신 프로토콜을 활용하여 데이터 수집 및 태그 정합을 용이하게 합니다.

- **Performance Highlights**: 시험 결과, 모든 측정된 보행 매개변수의 정확도가 93.61%를 초과하며, 장거리 보행 중에도 4.89%의 낮은 드리프트를 보여 주목할 만한 성능을 나타냈습니다. 훈련된 Transformer 모델을 이용한 보행자 식별 작업에서도 95.7%의 정확도를 기록하여, 우리 시스템이 현재의 Large Language Models (LLMs)과 통합 가능한 긴 시퀀스 보행 데이터를 수집할 수 있는 잠재력을 가지고 있음을 보여주었습니다.



### HIST-AID: Leveraging Historical Patient Reports for Enhanced Multi-Modal Automatic Diagnosis (https://arxiv.org/abs/2411.10684)
Comments:
          In Proceedings of Machine Learning for Health

- **What's New**: 이 논문에서는 환자의 역사적 데이터를 통합한 Temporal MIMIC 데이터셋을 도입했습니다. 이 데이터셋은 12,221명의 환자와 13개의 병리를 포함하여 5년간의 방사선 검사 및 보고서를 제공합니다.

- **Technical Details**: HIST-AID라는 프레임워크를 소개하여 방사선과 의사가 환자의 역사적 데이터를 활용하여 진단 정확성을 향상시킬 수 있도록 합니다. 이 시스템은 역사적 보고서를 사용하여 자동 진단의 정확성을 높이는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, 단순 방사선 촬영에 의존하는 모델에 비해 AUROC가 6.56%, AUPRC가 9.51% 향상되었습니다. 이러한 개선은 성별, 나이, 인종 범주 등 다양한 인구 집단에서도 일관되게 관찰되었습니다.



### Enhancing PTSD Outcome Prediction with Ensemble Models in Disaster Contexts (https://arxiv.org/abs/2411.10661)
- **What's New**: 본 연구는 PTSD(외상 후 스트레스 장애) 분류를 위한 다양한 머신러닝 알고리즘 및 맞춤형 인공 신경망(ANN) 모델의 성능을 비교하는 새로운 접근 방식을 제시하였습니다. 전처리 작업을 통해 데이터 청소, 결측치 처리, 데이터 불균형 해결을 위한 SMOTE 사용 등 체계적인 과정을 도입하였습니다.

- **Technical Details**: 연구는 80%의 데이터를 훈련용으로, 20%의 데이터를 테스트용으로 분리하여, 다수의 분류기(Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, 맞춤형 ANN)를 조합한 앙상블 모델을 개발하였습니다. 이 모델은 96.76%의 높은 정확도를 달성했으며, 정확한 PTSD 감지를 위한 다중 모델의 조합을 통해 일반화 능력을 향상시켰습니다.

- **Performance Highlights**: 제안된 앙상블 모델은 96.76%의 정확도로 개별 모델을 크게 능가하여, 재난 피해 인구의 정신 건강 문제에 대한 예측 분석 도구로서 큰 가치가 있음을 입증하였습니다. 이는 정책 입안자 및 의료 제공자들이 취약 인구의 정신 건강 개입을 보다 효과적으로 설계하는 데 도움을 줄 것입니다.



### Understanding Learning with Sliced-Wasserstein Requires Rethinking Informative Slices (https://arxiv.org/abs/2411.10651)
- **What's New**: 본 논문은 Sliced-Wasserstein distances(SWD)의 새로운 접근법을 제안하여, 모든 슬라이스가 동등하게 유용하도록 1D Wasserstein 거리를 재조정하는 방법을 연구합니다.

- **Technical Details**: 전통적인 Sliced-Wasserstein 거리(SWD)는 1차원 서브스페이스로 분포를 투영하여 계산됩니다. 저자들은 데이터를 가정하고 슬라이스의 유용성 개념을 도입하여, 개별 슬라이스의 재조정이 SWD에 단일 전역 스케일링 인자로 간단하게 변환될 수 있음을 보입니다.

- **Performance Highlights**: 저자들은 다양한 머신 러닝 작업을 통해 전통적인 SWD가 복잡한 변형들과 동등하거나 더 나은 성능을 발휘할 수 있음을 입증하였습니다. 이를 통해 'Sliced-Wasserstein이 일반적인 학습 작업에 필요할까?'라는 질문에 대한 답을 제시합니다.



### A minimalistic representation model for head direction system (https://arxiv.org/abs/2411.10596)
Comments:
          Workshop on Symmetry and Geometry in Neural Representations (NeurReps) at NeurIPS 2024, Extended Abstract Track

- **What's New**: 이 논문에서는 헤드 방향(HD) 시스템을 위한 최소한의 표현 모델을 제안하고 있습니다. 이 모델은 HD 세포의 중요한 특성을 포착하는 고차원 표현을 학습하는 것을 목표로 하며, 회전군 $U(1)$의 표현으로 구성됩니다. 연구팀은 완전 연결 네트워크와 합성곱 네트워크 두 가지 버전을 검토하며, 이 두 모델에서 가우시안 형의 튜닝 프로파일과 2D 원형 기하구조의 발생을 보여줍니다.

- **Technical Details**: 헤드 방향을 연속적인 d차원 벡터로 표현하며, 이는 HD 세포의 반응으로 여겨집니다. 이 모델은 세 가지 제약조건을 따릅니다: 변환 규칙, 비부정성 제약, 그리고 단위 노름 제약. 이 모델은 경로 통합을 가능하게 하는 순환 신경망으로 정의되며, 로컬 운동에 대한 Taylor 전개를 통해 모델의 기능을 구체화합니다.

- **Performance Highlights**: 제안된 모델은 주어진 방향에 대해 가우시안 형의 튜닝 프로파일을 학습할 수 있으며, 주성분 분석(PCA)을 통해 시각화했을 때 명확한 원형 기하 구조를 나타냅니다. 학습된 모델은 정확한 경로 통합을 수행할 수 있으며, 이러한 특성은 생물학적 HD 시스템의 특성과 밀접하게 일치합니다.



### FedAli: Personalized Federated Learning with Aligned Prototypes through Optimal Transpor (https://arxiv.org/abs/2411.10595)
Comments:
          Pre-print version 1

- **What's New**: 이 논문에서는 Federated Learning (FL)에서 데이터의 이질성으로 인한 사용자 맞춤형 모델 학습의 문제를 해결하기 위해 Alignment with Prototypes (ALP) 계층을 도입합니다. 이 ALP 계층은 입력 임베딩을 학습 가능한 프로토타입에 맞춰 최적화합니다.

- **Technical Details**: Proposed framework, Federated Alignment (FedAli),는 여러 클라이언트에서 임베딩을 수집하여 교육 중 개별 프로토타입을 업데이트하고, 전체 클라이언트에서 집계된 글로벌 프로토타입에 임베딩을 맞추는 과정을 포함합니다. 이를 통해 클라이언트 각각의 데이터 분포에 맞는 개인화된 모델을 학습하고, 높은 일반화 성능을 유지할 수 있습니다.

- **Performance Highlights**: 모델 성능 평가 결과, FedAli는 이질적인 센서 기반의 인간 활동 인식(HAR) 및 비전 벤치마크 데이터셋에서 기존 FL 전략보다 우수한 성능을 보였으며, 데이터의 다양성과 환경 변화에 잘 적응하는 것을 입증하였습니다.



### Normative Modeling for AD Diagnosis and Biomarker Identification (https://arxiv.org/abs/2411.10570)
Comments:
          10 pages, 3 figures

- **What's New**: 이 논문에서는 알츠하이머 질병(AD) 진단 및 바이오마커 분석을 위한 새로운 규범 모델링 접근법을 소개합니다. 이 방법은 adversarial autoencoder(AAE) 구조 내에 adversarial focal loss를 통합하여 더 복잡하고 도전적인 케이스를 효과적으로 타겟하는 목표로 설계되었습니다.

- **Technical Details**: 우리의 방법은 end-to-end 접근법으로, 건강한 대조군(HC) 데이터를 기반으로 규범 모델을 구성한 후, 이를 사용하여 AD 환자의 신경해부학적 변이를 추정합니다. 제안된 모델은 fMRI 데이터에서 voxels를 평균하여 100개의 영역(region-of-interest, ROI)으로 기능적 특성을 생성합니다. 이 과정에서 조건부 변이형 오토인코더(conditional variational autoencoder, CVAE)를 사용하여 인구통계학적 변수의 영향을 통제합니다.

- **Performance Highlights**: OASIS-3 및 ADNI 데이터셋을 통한 실험 결과, 우리의 FAAE 기반 규범 모델은 AUROC와 민감도 점수 측면에서 이전의 최첨단 방법들보다 유의미하게 우수한 성능을 보였습니다. AD와 HC 간의 편차 플롯 분석을 통해 질병 이질성에 대한 더 깊은 통찰력을 제공하며, 임상 진단과 바이오마커 발견 가능성에 대한 새로운 기반을 마련하고 있습니다.



### Advancing Autonomous Driving Perception: Analysis of Sensor Fusion and Computer Vision Techniques (https://arxiv.org/abs/2411.10535)
Comments:
          7 pages

- **What's New**: 이 논문은 자율주행에 필수적인 인지 시스템의 안전성을 향상시키기 위해, 시각 기반 시스템과 자율주행 인지 작업의 메트릭스에 대한 최신 발전을 조사하였습니다. 특히, 깊이 기반 인지 및 컴퓨터 비전 기법을 활용하여 자율주행 로봇의 이해 및 내비게이션 능력을 개선하는 데 초점을 맞추고 있습니다. 또한, 현재 연구에서 직면하고 있는 주요 도전과제와 성과를 강조합니다.

- **Technical Details**: 자율주행 시스템은 정확하고 강력한 환경 인지에 크게 의존합니다. 본 연구에서는 센서 융합 기법의 도전과제 및 비전 기반 내비게이션과 의사결정 능력을 향상시키기 위한 두 가지 접근법을 조사했습니다. 이 프로젝트는 Jetson Xavier 프로세서, ZED 2 RGBD 카메라, ROS(로봇 운영 체제) 등의 하드웨어와 소프트웨어의 조합을 활용하여 센서 데이터를 처리 및 분석했습니다.

- **Performance Highlights**: 우리는 깊이 기반 인지를 통해 3D 환경 정보를 제공하는 동시에, ROS 환경에서 실시간 처리 기능을 통합하여 자율 내비게이션을 구현했습니다. 초기 계획대로 2D LiDAR 융합과 ZED 2 카메라를 동시에 활용하는 데는 한계가 있었으나, 깊이 정보를 활용하여 객체와의 거리를 추정함으로써 내비게이션 성능 향상에 기여했습니다. 최종적으로, 컴퓨터 비전 기술을 통해 로봇의 주행 경로를 탐지하고 통제했습니다.



### RedTest: Towards Measuring Redundancy in Deep Neural Networks Effectively (https://arxiv.org/abs/2411.10507)
- **What's New**: 본 논문은 심층 신경망(DNN)의 구조적 중복성(redundancy)을 정량적으로 측정하는 새로운 테스트 메트릭인 Model Structural Redundancy Score (MSRS)를 제안합니다. 이를 통해 대규모 딥러닝 모델의 불필요한 요소를 효율적으로 식별하고 제거할 수 있는 방법을 제시합니다.

- **Technical Details**: MSRS는 대칭 중간 표현(intermediate representations, IRs)의 유사성을 기반으로 딥러닝 모델 구조의 중복성을 평가합니다. 이 메트릭은 기존의 파라미터 수, FLOAT 포인트 연산(FLOPs), 이미지 처리 시간(Latency) 등과 같은 지표들과는 다르게 구조적 중복성에 직접적으로 접근하며, 레이어 간의 유사성을 활용하여 중복 레이어를 효과적으로 제거하는 기법을 구현합니다.

- **Performance Highlights**: RedTest는 Neural Architecture Search (NAS)와 대규모 사전 훈련된 모델의 가지치기(pruning) 두 가지 실용적인 적용에 사용되며, 실험 결과는 중복을 제거하는 것이 모델의 유용성에 거의 영향을 미치지 않는다는 것을 보여줍니다. MSRS를 활용하여 최적의 모델 구조를 찾는 알고리즘의 효과성을 입증하며, RedTest를 통해 더 나은 모델 개발을 촉진합니다.



### Edge-Only Universal Adversarial Attacks in Distributed Learning (https://arxiv.org/abs/2411.10500)
- **What's New**: 이번 연구는 분산 학습 환경에서 모델의 일부만을 사용할 경우, 즉 Edge 측면에서만 공격자가 접근할 때 생성 가능한 보편적인 적대적 공격(Universal Adversarial Attacks, UAPs)의 가능성을 탐구합니다. 이전의 전통적인 UAPs가 전체 모델에 대한 접근을 필요로 하는 것과는 달리, 이 연구는 Edge 측에서 핵심 기능을 활용하여 클라우드 부분에서 효과적인 오예측을 유도할 수 있음을 보여줍니다. 연구 결과는 ImageNet 데이터셋에서 강력한 공격 전이 능력을 입증하며, Edge만의 지식으로도 타깃된 적대적 효과를 achieved 할 수 있는 방법을 제공합니다.

- **Technical Details**: 본 연구에서는 Edge 측에서 사용 가능한 중간 특징을 활용하여 경량 분류기를 학습하고, 이를 활용하여 새로운 타깃 최적화 방법으로 효과적인 UAPs를 제작하는 방법을 소개합니다. 저자들은 여러 실험을 통해 Edge에서 몇 개의 레이어만을 사용하더라도 높은 오예측률을 유도할 수 있음을 입증하였으며, 일반적인 화이트 박스 UAP 공격과 비교할 수 있는 성과를 달성했습니다. 이 연구는 모델의 클라우드 부분에 대한 접근 없이도 공격이 처음 레이어에서만 적용되더라도 특정 클래스에 대한 방향전을 조정할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 공격 방식은 공격자가 모델의 클라우드 부분에 대한 접근 없이도, 초기 레이어만으로 강력한 비정확도를 유도하는 결과를 보여주었습니다. ImageNet 상에서 수행된 여러 실험에서, 적은 수의 레이어로도 유의미한 효과를 발휘하며, 화이트 박스 UAP 공격과 비교할 때 성능의 유사성을 보였습니다. 이 연구는 분산 추론에서 Edge-only 공격의 기초를 제공하며, 앞으로의 연구에서 부분 모델 접근의 중요성을 강조합니다.



### Biometrics in Extended Reality: A Review (https://arxiv.org/abs/2411.10489)
- **What's New**: 이 연구는 XR(Extended Reality) 환경에서 생체 정보(biometric characteristics)의 응용에 대해 체계적으로 조사한 첫 번째 논문이다. 우리는 사용자의 인증(authentication)과 표현을 위한 다양한 생체 모달리티(biometric modalities)를 종합적으로 정리하였다. 또한 XR 시스템의 생체 정보와 관련된 취약점(vulnerability gateways)을 문헌에서 처음으로 언급하고, 이에 대한 분류법(taxonomy)을 소개했다.

- **Technical Details**: 제안된 연구 방법론은 XR 시스템의 생체 인식 기반 워크플로우에서 주요 취약점과 가능한 공격 유형을 규명하는 것을 포함한다. 318편의 논문을 검토하여, 62편의 관련 논문의 리뷰를 통해 현재의 흐름과 다양한 구성 요소에 대한 명확한 이해를 제공하였다. 생체 인식 기반 포토리얼리스틱 아바타(photo-realistic avatars) 생성 및 인증의 접근 방식도 논의되었다.

- **Performance Highlights**: XR 환경에서 생체 인식의 유용성은 향후 이러한 시스템의 보안(security)과 사용자 개인 정보 보호(user privacy)를 보장하는 데 필수적이다. 이 연구는 생리학적(physiological) 및 행동적(behavioral) 생체 인식 특징을 폭넓게 논의하며, XR 애플리케이션에서 광범위하게 사용되는 생체 인식 기반 포토리얼리스틱 아바타 생성 기술에 대한 포괄적인 개요를 제공한다. 또한 XR 애플리케이션 내 생체 인식의 잠재력에 대한 미래 연구의 기회를 제시한다.



### Efficient Denoising Method to Improve The Resolution of Satellite Images (https://arxiv.org/abs/2411.10476)
- **What's New**: 이번 논문은 저해상도 위성 이미지의 해상도를 높이는 데 있어 효율적인 guided (가이드된) 또는 image-conditioned (이미지 조건부) denoising diffusion models (DDMs)을 제안합니다. 저비용의 소형 위성이 인기를 끌고 있는 가운데, 최근 생성 모델을 통해 이러한 이미지의 처리를 개선할 수 있는 방법을 탐구합니다.

- **Technical Details**: 이 논문에서는 확률적 (stochastic) 보통 미분 방정식 (ordinary differential equations, ODEs)을 기반으로 한 denoising 기법을 사용하여, 수백 번의 반복을 통해 해상도를 향상시키는 대신 결정론적 (deterministic) ODE를 활용한 Consistency Models (CM)을 제안합니다. 이 방법은 Stable Diffusion 모델을 기반으로 하며, Teacher-Student Distillation 기법을 통해 DDM을 CM으로 변환하여 효과적인 denoising을 달성합니다.

- **Performance Highlights**: 수정된 CM을 사용한 Stable Diffusion 모델은 위성 이미지의 해상도를 16배 향상시키고, 계산 시간은 확률적 denoising 방법에 비해 20배 단축되었습니다. 또한, 이미지 해상도를 높인 후 FID (Frechet Inception Distance) 점수가 10.0에서 1.9로 개선되었습니다.



### Relationships between the degrees of freedom in the affine Gaussian derivative model for visual receptive fields and 2-D affine image transformations, with application to covariance properties of simple cells in the primary visual cortex (https://arxiv.org/abs/2411.05673)
Comments:
          21 pages, 9 figures

- **What's New**: 이 논문은 3D 물체의 표면 패턴을 관찰할 때 발생하는 두 가지 자유도(degrees of freedom) 간의 관계를 수학적으로 분석합니다. 특히 2D 공간 아핀 변환(2-D spatial affine transformations)과 아핀 가우시안 미분 모델(affine Gaussian derivative model) 간의 연관성을 규명합니다. 기존의 신경생리학적 실험 데이터를 바탕으로 고등 포유류의 기본 시각 피질에서 생물학적 수용 필드가 이러한 자유도를 포함할 수 있는지에 대해 논의합니다.

- **Technical Details**: 논문에서는 2D 아핀 변환의 정규 분해(canonical decomposition)를 설명하며, 이는 특이값 분해(singular value decomposition)와 밀접하게 관련되어 있습니다. 이 과정을 통해 얻은 자유도는 균일 스케일링 변환(uniform scaling transformations), 전체적인 회전(global rotation), 비균일 스케일링 변환(non-uniform scaling transformation) 및 선호되는 대칭 방향으로의 상대 정규화(relative normalization)로 구성됩니다. 이후 이러한 자유도가 아핀 가우시안 미분 모델의 자유도와 어떻게 연관되는지를 보여줍니다.

- **Performance Highlights**: 이론적 분석 결과는 비전 관련 알고리즘의 성능에 영향을 미치는 중요한 통찰력을 제공합니다. 예를 들어, 아핀 공변적 시각 수용 필드를 사용했을 때, 이미지에서의 구조적 정보를 더 정확하게 추정할 수 있다는 것을 보여줍니다. 이는 시각 처리의 초기에 발생하는 에러를 줄이며, 생물학적 비전 시스템 또한 아핀 변환 아래에서 공변성을 준수한다고 제안합니다.



New uploads on arXiv(cs.AI)

### Lifted Model Construction without Normalisation: A Vectorised Approach to Exploit Symmetries in Factor Graphs (https://arxiv.org/abs/2411.11730)
Comments:
          Accepted to the Proceedings of the 3rd Learning on Graphs Conference (LoG 2024)

- **What's New**: 이번 논문에서는 존재하지 않았던 교환 가능한 팩터의 대칭성을 탐지하는 새로운 알고리즘을 제안합니다. 기존의 Advanced Colour Passing (ACP) 알고리즘의 한계를 극복하여, 팩터의 잠재적 (potential)을 임의로 조정할 수 있게 되었습니다.

- **Technical Details**: 제안된 알고리즘은 팩터의 잠재적 값을 벡터 형식으로 인코딩하여 대칭성을 탐지합니다. 기존의 ACP 알고리즘과는 달리, 잠재적 값이 동일한 스케일로 조정될 필요가 없으며, 이로 인해 더 컴팩트한 PFG (Parameterized Factor Graph)를 생성할 수 있습니다.

- **Performance Highlights**: 실험을 통해, 제안된 알고리즘은 온라인 쿼리 시간의 현저한 감소를 입증했습니다. 더불어, 대칭성의 탐지가 증가함에 따라 효율적인 확률적 추론을 가능하게 합니다.



### PSPO*: An Effective Process-supervised Policy Optimization for Reasoning Alignmen (https://arxiv.org/abs/2411.11681)
- **What's New**: 본 논문에서는 과정 감독(process supervision) 방법의 효과가 추론 체인(chain of thought)의 정확성과 길이에 따라 달라진다는 주장을 하고 있으며, 비선형 보상(nonlinear reward)이 이를 최적화하는 데 중요한 역할을 한다고 제시합니다. 새로운 PSPO* 패러다임을 통해 과정 감독 프레임워크를 도입하며, PSPO-WRS 방법을 통해 비선형 보상 구조를 채택하여 성능을 개선하였습니다.

- **Technical Details**: PSPO*는 보상 모델 훈련(reward model training)과 정책 최적화(policy optimization) 단계를 체계적으로 규명하며, 비선형 보상을 통해 적절한 보상 점수 수치를 계산합니다. 이 방법은 데이터 처리에서 각 단계의 정확도 및 길이를 고려하여 보상 점수를 산출합니다. PSPO-WRS는 조정된 Weibull 분포 조형을 사용하여 비선형적인 보상 구조를 도입하고 있습니다.

- **Performance Highlights**: 여섯 개의 수학적 추론 데이터셋에서의 실험 결과, PSPO-WRS는 기존의 주류 모델보다 지속적으로 우수한 성능을 보였습니다. 이는 PSPO* 방법론의 효과를 입증하는 결과이며, 비선형 보상 구조가 추론 과정에서 성능 개선에 기여함을 시사합니다.



### Artificial Scientific Discovery (https://arxiv.org/abs/2411.11672)
Comments:
          PhD thesis, 123 pages

- **What's New**: 이 논문은 AlphaGo에서 ChatGPT까지의 발전을 바탕으로 인공지능 과학자(artificial scientist)의 비전을 실현하기 위한 기본 개념을 실증적으로 조사합니다.

- **Technical Details**: 논문은 Olivaw라는 AlphaGo Zero와 유사한 에이전트가 Othello 지식을 스스로 발견하나, 이를 타인에게 전달할 수 없음을 기초로 개발된 Explanatory Learning (EL) 프레임워크(Framework)를 다룹니다. 이 EL 프레임워크는 과학자가 동료에게 새로운 현상을 설명할 때 마주치는 문제를 형식화합니다.

- **Performance Highlights**: Olivaw의 성공은 인공지능 과학자가 자신이 발견한 내용을 설명하기 위해 사용하는 언어에 대한 해석 능력을 개발해야 한다는 중요한 통찰을 제공합니다. 또한, 이 연구에서는 현대의 멀티모달(multi-modal) 모델을 해석자로 보며, 두 개의 유니모달 모델(unimodal model)을 연결하여 해석 가능하고 비용 효율적인 CLIP 모델을 구축할 수 있는 새로운 방법을 제안합니다.



### A Pre-Trained Graph-Based Model for Adaptive Sequencing of Educational Documents (https://arxiv.org/abs/2411.11520)
Comments:
          NeurIPS 2024 Workshop on Large Foundation Models for Educational Assessment (FM-Assess), Dec 2024, Vancouver, Canada

- **What's New**: 이 연구에서는 전문가의 주석(skilling) 없이 학습 경로 개인화를 위한 새로운 데이터 효율적(data-efficient) 프레임워크를 소개합니다. 기존의 MOOC는 개인의 다양한 필요와 배경을 고려하지 않는 경향이 있습니다.

- **Technical Details**: 이 방법은 원시 코스 데이터(raw course data)에서 강화 학습(reinforcement learning)으로 사전 학습된 유연한 추천 시스템(flexible recommender system)을 사용합니다. 이는 전문가의 피드백 없이도 학습 경로(personalization) 설정을 가능하게 합니다.

- **Performance Highlights**: 세미 합성 데이터(semi-synthetic data) 실험을 통해, 이 사전 학습(pre-training) 단계가 다양한 적응형 교육(adaptive learning) 시나리오에서 데이터 효율성을 크게 향상시킨다는 것을 보여주었습니다.



### Search, Verify and Feedback: Towards Next Generation Post-training Paradigm of Foundation Models via Verifier Engineering (https://arxiv.org/abs/2411.11504)
- **What's New**: 본 논문에서는 'verifier engineering'이라는 새로운 포스트 트레이닝 패러다임을 제안합니다. 이는 foundation model 시대에 맞춰 설계되었으며, 자동화된 verifier들을 활용하여 모델의 검증 작업을 수행하고 의미 있는 피드백을 제공하도록 구성되었습니다. 기존의 수동 특성 추출 및 데이터 주석 방법을 넘어서, 효율적인 검증 시스템의 설계와 조작을 강조하는 점이 특징입니다.

- **Technical Details**: verifier engineering은 세 가지 핵심 단계인 search, verify, feedback으로 체계적으로 분류할 수 있습니다. 각 단계에서 최신 연구 발전을 종합적으로 검토하며, 특정 지침이 주어지면 후보 응답을 생성하고, 적절한 검증자 조합을 통해 이들을 검증하여 최종적으로 모델 출력 분포를 최적화합니다. 기존의 RLHF(Reinforcement Learning from Human Feedback)와는 달리, 다양한 출처의 verifier를 통합하여 보다 정확하고 일반화된 피드백 신호를 전달합니다.

- **Performance Highlights**: verifier engineering은 foundation models의 개선을 데이터 중심의 접근법에서 시스템적 엔지니어링 문제로 전환합니다. 이는 효과적이고 효율적인 피드백을 보장하기 위한 복잡한 검증 시스템의 설계 및 조정을 강조하며, 기존 방법론보다 정확하고 일반화된 성능 향상을 가능하게 합니다. 또한, 다양한 검증 프로세스를 통해 foundation models의 역량을 극대화할 수 있는 가능성을 제시합니다.



### Alien Recombination: Exploring Concept Blends Beyond Human Cognitive Availability in Visual Ar (https://arxiv.org/abs/2411.11494)
Comments:
          NeurIPS 2024 Workshop on Creativity & Generative AI, 13 pages, 11 figures

- **What's New**: 이번 연구에서는 AI의 창의성을 예술 분야에서 탐구하고, Alien Recombination 방법론을 도입하여 AI가 인간의 인지 한계를 초월하여 독창적인 예술 개념의 조합을 생성할 수 있는 가능성을 제시합니다.

- **Technical Details**: Alien Recombination 방법은 두 개의 대형 언어 모델(LLMs)을 사용하여 예술 개념의 새로운 조합을 생성하고 순위를 매기는 방식으로 구성됩니다. 아티스트 개념의 협응력을 평가하기 위해 두 가지 주요 확률 분포를 추정하였으며, GPT-2 모델을 미세 조정하여 창의적인 조합을 생성합니다. 이 과정에서 'Cognitive Unavailability' 측정을 통해 예술의 기묘한 조합을 탐색합니다.

- **Performance Highlights**: 이 방법은 기존 데이터셋 내에서 시도된 바 없는 조합을 생성하고, 예술가에게 인지적으로 불가능한 조합도 확인할 수 있는 가능성을 보여주었습니다. 제안된 방법은 예술적 기묘함을 최적화하는 메트릭인 'Cognitive Unavailability'를 사용하여, 단순히 온도 스케일링만을 이용한 방법보다 더 나은 성과를 내는 것으로 나타났습니다.



### Robust Markov Decision Processes: A Place Where AI and Formal Methods M (https://arxiv.org/abs/2411.11451)
- **What's New**: 이번 연구는 Robust Markov Decision Processes (RMDPs)에 대한 훌륭한 개요를 제공하여 이론적 기초와 기존 문헌에 대한 간략한 리뷰를 포함합니다. RMDPs는 전통적인 MDPs와 다르게, 전이 확률에 대한 정확한 지식이 필요하지 않으며, 불확실성 집합을 정의함으로써 이러한 가정을 극복합니다. 이 연구는 AI와 형식적 방법(formal methods) 분야의 전문가들이 RMDPs에 대한 이해를 통합하는 데 기여하는 것을 목표로 합니다.

- **Technical Details**: RMDPs의 정의는 상태, 행동, 전이 확률 및 보상으로 구성된 MDPs의 형식적 모델을 확립하는 데 중점을 둡니다. MDPs에서 전이 확률은 불확실성이 있을 경우에도 적용될 수 있는 유연한 구조를 제공합니다. 이 문서에서는 기본적인 MDPs의 소개와 RMDPs의 이론적 기초에 대한 설명을 제공합니다.

- **Performance Highlights**: RMDPs는 강화 학습(reinforcement learning)과 다양한 형태의 데이터 기반 방법들에서 응용되며, 이론적 발전이 도구 지원 및 알고리즘 발전에 기여할 수 있습니다. 연구 결과는 AI와 형식적 방법 분야에서 서로 상호작용하여, RMDPs의 적용 가능성을 확장할 수 있는 다양한 기회를 제공합니다.



### Syllabus: Portable Curricula for Reinforcement Learning Agents (https://arxiv.org/abs/2411.11318)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 Curriculum Learning(커리큘럼 학습)의 필요성과 이를 효과적으로 지원하는 Syllabus라는 새로운 라이브러리를 소개합니다. Syllabus는 다양한 RL 라이브러리와의 간편한 통합을 제공하며, 커리큘럼 학습 알고리즘을 위한 보편적인 API를 제공합니다. 이 라이브러리는 ML(기계 학습) 알고리즘 설계를 단순화하고 새로운 환경에 적용할 수 있도록 돕습니다.

- **Technical Details**: Syllabus는 RL(강화 학습) 에이전트를 위한 커리큘럼 학습 알고리즘을 설계하는 데 필요한 최소한의 API를 제공합니다. 이 라이브러리는 기존 RL 코드베이스와 독립적인 경량 인터페이스를 제공하여 더 편리한 커리큘럼 설계를 가능하게 합니다. 여러 다양한 커리큘럼 학습 알고리즘을 구현할 수 있으며, 다양한 RL 환경에서 에이전트를 훈련하는 데 사용됩니다.

- **Performance Highlights**: Syllabus를 통해 NetHack과 Neural MMO와 같은 유명한 RL 벤치마크에서 커리큘럼 학습의 첫 사례를 달성하며, 기존 알고리즘과 비교할 때 강력한 결과를 보였습니다. 이 라이브러리는 CleanRL, Stable Baselines 3 등 다양한 RL 라이브러리에서 같은 코드로 에이전트를 훈련할 수 있는 유연성을 제공합니다. 커리큘럼 학습의 성능 개선 효과를 입증하였습니다.



### Reinforcing Competitive Multi-Agents for Playing So Long Sucker (https://arxiv.org/abs/2411.11057)
- **What's New**: 이 논문은 전략 게임 So Long Sucker (SLS)에서 고전적인 딥 강화 학습 (Deep Reinforcement Learning, DRL) 알고리즘인 DQN, DDQN, Dueling DQN의 적용을 탐구합니다. SLS는 협력적이고 적대적인 동적의 혼합으로 독특한 도전을 제공하는 게임으로, 다중 에이전트 학습 및 게임 이론을 연구하기 위한 이상적인 플랫폼입니다. 연구의 주요 목표는 고전 DRL 방법을 사용하여 자율 에이전트에게 게임의 규칙과 전략을 가르치는 것입니다.

- **Technical Details**: 논문에서는 자율 에이전트를 훈련하기 위한 계산적 프레임워크를 개발하였으며, 이 프레임워크는 SLS의 규칙 학습과 효과적인 전략 개발을 지원합니다. 비공식적으로 제공되는 SLS의 새로운 구현이 graphical user interface (GUI)와 DRL 알고리즘의 벤치마킹 도구를 포함하고 있습니다. SLS의 변형인 Generalized Hofstra's version을 중심으로, 각 플레이어는 고유한 색깔을 부여받고 정해진 수의 칩으로 게임을 시작합니다.

- **Performance Highlights**: DQN, DDQN 및 Dueling DQN 에이전트는 현대 DRL 기준으로는 기본적인 것으로 간주되지만, 최대 가능한 게임 보상의 약 50%를 달성하였습니다. 그러나 에이전트가 도달하는 데 필요한 훈련량은 약 2000 게임으로, 이는 인간 플레이어가 게임을 이해하는 데 몇 라운드 만에 비해 상당히 긴 시간입니다. 이러한 결과는 고전 DRL 방법의 잠재력과 한계를 강조하며, SLS와 유사한 협상 기반 환경에서 에이전트 교육을 위한 기초 기준을 제시합니다.



### Pluralistic Alignment Over Tim (https://arxiv.org/abs/2411.10654)
Comments:
          Pluralistic Alignment Workshop at NeurIPS 2024

- **What's New**: 이 논문에서는 AI 시스템의 결정이 다양한 이해관계자들의 가치에 얼마나 잘 조정되고 있는지를 평가하기 위해 시간적 요소를 고려할 필요성을 주장합니다. 특히, 이해관계자의 만족도 변화와 그들의 장기적인 선호를 포함하는 새로운 형태의 복수적 조화인 temporal pluralism을 제안합니다.

- **Technical Details**: AI 시스템의 결정은 여러 이해관계자의 가치와 선호에 맞닿아야 하며, 그 과정에서 시간의 영향을 고려해야 합니다. 예를 들어, 시간에 따라 선호가 변화하거나, 다양한 이해관계자의 이익을 시간에 맞추어 각각 실현하는 방식으로 조화할 수 있습니다. 특히, 비마르코프형 보상 함수 (non-Markovian reward function)와 시간적 논리 (temporal logics)를 활용하여 인간의 선호를 더 잘 표현할 수 있습니다.

- **Performance Highlights**: AI 시스템이 이루는 결정은 서로 다른 시간대에서 서로 다른 이해관계자들과 조화를 이루기 위해 조정될 수 있습니다. 고전적인 결정 과정에서 물자가 고르게 분배되지 않는 어려움을 겪는 상황에서도, 시간대별로 다양한 가치를 반영함으로써 기술적으로 더 나은 결정을 내릴 수 있는 가능성을 보여줍니다. 이를 통해 AI 시스템이 시간에 따라 이해관계자들의 필요를 보다 잘 충족시킬 수 있는 방법이 제시됩니다.



### Being Considerate as a Pathway Towards Pluralistic Alignment for Agentic AI (https://arxiv.org/abs/2411.10613)
Comments:
          Pluralistic Alignment Workshop at NeurIPS 2024

- **What's New**: 이번 논문은 Agentic AI의 맥락에서 다원적 정렬(pluralistic alignment)을 다루고 있습니다. AI 시스템의 목표와 행동이 인간의 가치와 관점의 다양성과 조화롭게 일치하도록 하는 것이 주요 목표입니다.

- **Technical Details**: 다원적으로 잘 정렬된 에이전트는 사회의 복지를 극대화하는 방향으로 자신의 목표를 실현하려고 해야 합니다. 이를 위해 강화 학습(reinforcement learning) 에이전트는 각 환경 내 다른 에이전트의 미래 복지와 주체성에 미치는 영향을 고려하도록 설계되었습니다. 구체적으로, 보상 함수를 보조 보상으로 보강하여 다양한 에이전트의 가치를 반영하도록 하였습니다.

- **Performance Highlights**: 이 방법은 환경 내 다양한 에이전트의 복지와 주체성을 고려하여 AI 에이전트의 행동을 조정하고, 궁극적으로 사회적 복지를 증진하는 데 기여할 수 있습니다.



### Bi-Mamba: Towards Accurate 1-Bit State Space Models (https://arxiv.org/abs/2411.11843)
- **What's New**: 이번 연구에서는 Bi-Mamba라는 새로운 1-bit Mamba 아키텍처를 소개합니다. 이는 여러 크기로 확장 가능하며, 대형 언어 모델을 더 효율적으로 만들어주는 모델입니다. Bi-Mamba는 자율 회귀 증류 손실을 사용하여 데이터를 기반으로 처음부터 훈련되며, 기존의 Mamba 모델에 비해 메모리 소비와 에너지 소비를 크게 줄입니다.

- **Technical Details**: Bi-Mamba는 Selective State-Space Model(SSM)의 일종으로, 기존 Transformer 모델의 제약을 극복하는 데 도움이 됩니다. Bi-Mamba는 1-bit 양자화(quantization)를 적용하여 훈련 및 추론할 때 높은 성능을 유지하고, 메모리 사용량을 크게 줄이면서도 상대적으로 더 적은 에너지를 소비합니다. 이 연구는 비트 수가 낮은 표현 아래에서의 선형 계산 복잡성을 가진 LLM 프레임워크를 새롭게 제시합니다.

- **Performance Highlights**: 실험 결과, Bi-Mamba는 FP16 또는 BF16을 사용하는 완전 정밀 모델에 필적하는 성능을 보여주며, 전후 훈련 양자화(post-training quantization) Mamba 기준선보다 훨씬 높은 정확도를 기록했습니다. Bi-Mamba 모델은 자원 제한 환경에서도 강력한 기본 모델로 활용될 수 있으며, 쉽게 다른 NLP 응용 프로그램에 맞게 조정될 수 있습니다.



### LightFFDNets: Lightweight Convolutional Neural Networks for Rapid Facial Forgery Detection (https://arxiv.org/abs/2411.11826)
Comments:
          13 pages, 6 figures, 10 tables

- **What's New**: 이 연구는 얼굴 이미지 위조 탐지를 위한 새로운 경량 딥러닝 모델인 LightFFDNets를 제안하고, 실제와 위조된 얼굴 이미지를 인식하는 데 있어 높은 정확도와 계산 효율성을 보여준다. 기존의 CNN 아키텍처와 비교한 연구 결과, 제안된 모델이 향상된 성능을 보임을 입증하였다. 연구에서 사용된 데이터셋은 Fake-Vs-Real-Faces (Hard)와 140k Real and Fake Faces로,두 클래스 모두의 얼굴 이미지를 포함하고 있다.

- **Technical Details**: 연구에서는 특징 추출을 수행하는 CNN 모델이 사용되며, 실제 및 위조된 얼굴 이미지를 탐지하기 위해 두 개의 경량 모델이 개발되었다. 제안된 모델은 최소한의 레이어로 구성되어 있으며, 여러 개의 사전 훈련된 CNN 아키텍처와의 결과를 비교하였다. GAN(Generative Adversarial Networks) 기술이 얼굴 생성 및 위조 탐지에 사용되는 방법도 검토하고 있다.

- **Performance Highlights**: 제안된 경량 딥러닝 모델은 빠른 계산 시간에도 불구하고 정확한 얼굴 위조 탐지를 제공하며, 다른 두 클래스 객체 인식 문제에도 적용 가능성을 보인다. 실험 결과, 기존 모델들과의 비교에서 연산 시간이 크게 개선되었음을 확인할 수 있었다. 이러한 성과는 얼굴 이미지 식별의 안전성 및 신뢰성을 높이는 데 기여할 것으로 기대된다.



### Character-level Tokenizations as Powerful Inductive Biases for RNA Foundational Models (https://arxiv.org/abs/2411.11808)
Comments:
          First version. Work in progress

- **What's New**: 본 연구에서는 ChaRNABERT라는 RNA 기초 모델을 제안합니다. 이 모델은 학습 가능한 토큰화 과정을 통해 여러 벤치마크에서 최신 성능(state-of-the-art performance)에 도달할 수 있도록 설계되었습니다. 또한 RNA-단백질 및 aptamer-단백질 상호작용 예측과 같은 응용 분야에서 성능을 평가합니다.

- **Technical Details**: ChaRNABERT는 샘플 및 파라미터 효율성을 가진 RNA 기본 모델 모음으로, 다양한 RNA 유형에 대한 일반화 능력을 보장하기 위해 다양한 RNA 종류로 훈련됩니다. 이 과정에서 학습 가능한 토큰화를 활용하여 RNA와 같은 생체분자에 최적화되어 있습니다.

- **Performance Highlights**: ChaRNABERT는 여러 표준 벤치마크 과제에서 최첨단 성능을 달성하였으며, 특히 RNA-단백질 상호작용 예측과 같은 고급 응용 분야에서 뛰어난 성과를 보여줍니다. 또한, ChaRNABERT-8M의 가중치와 추론 코드는 학술 연구에 사용할 수 있도록 제공됩니다.



### Edge-Enhanced Dilated Residual Attention Network for Multimodal Medical Image Fusion (https://arxiv.org/abs/2411.11799)
Comments:
          An extended version of the paper accepted at IEEE BIBM 2024

- **What's New**: 이번 연구에서는 멀티모달 의료 이미지 융합을 위한 새로운 CNN 기반 아키텍처를 제안했습니다. 이는 경계 세부 정보를 향상시키기 위한 기울기 연산자와 효과적인 다중 스케일 기능 추출을 위한 확장된 잔차 주의 네트워크 모듈을 도입하여 기존의 한계를 극복합니다. 또한, 교육 및 추론 동안 추가 계산이 필요 없는 파라미터 없는 융합 전략을 소개하여 융합의 속도와 효율성을 높입니다.

- **Technical Details**: 제안된 융합 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 기능 인코더, 융합 모듈 및 기능 디코더. 비대칭 오토인코더를 통해 입력 이미지에서 다중 스케일 기능을 추출하고, 이 기능을 융합하여 최종 이미지를 복원합니다. 훈련 과정은 다중 단계로 분리되어 있으며, 첫 번째 단계에서는 일반적인 복원 작업을 통해 다중 스케일 깊이 기능을 추출합니다.

- **Performance Highlights**: 다양한 베이스라인 방법들과 비교한 광범위한 실험 결과, 제안된 방법이 시각적 품질, 텍스처 보존 및 융합 속도 면에서 모두 우수한 성능을 보였습니다. 특히, 고등급 및 저등급 신경교종 분류 작업에 대한 적응성을 확인하여, 실제 임상 응용에 적합한 솔루션이 될 수 있음을 입증했습니다. 코드도 공개될 예정이며, 이를 통해 실제 응용 가능성을 더욱 높일 것입니다.



### COST CA20120 INTERACT Framework of Artificial Intelligence Based Channel Modeling (https://arxiv.org/abs/2411.11798)
Comments:
          to appear in IEEE Wireless Communications Magazine

- **What's New**: 이 논문에서는 통신 이론 및 시스템 설계에 필수적인 정확한 채널 모델링을 위해 인공지능(AI)을 활용하는 가능성을 평가하고 논의합니다. 전통적인 모델링 방법은 정확성, 일반화 능력, 계산 복잡성 측면에서 한계가 있음을 강조하며, 현대 통신 시스템의 물리적 환경과 채널 특성 간의 정량적 매핑이 어려워지고 있음을 설명합니다. AI 기반 채널 모델링 프레임워크를 도입하여 복잡한 무선 채널을 특성화하는 방법을 제안합니다.

- **Technical Details**: 논문에서는 AI 기반 채널 모델링을 통해 무선 채널의 복잡성을 다루는 새로운 방법론을 제시합니다. 주요 도전 과제로는 AI 기반 채널 예측의 불확실성 추정, 전파에 대한 이전 지식을 통합하여 일반화 능력을 향상시키는 방법, 채널 모델링을 위한 해석 가능한 AI를 제안합니다. 이러한 각 도전 과제에 대한 대안적 접근 방안도 논의됩니다.

- **Performance Highlights**: 저자들은 AI 기반 채널 모델링의 성능을 입증하기 위해 수치 결과를 제시하고 논의합니다. AI를 활용한 채널 모델링이 전통적인 방법보다 더 나은 성능을 발휘할 가능성을 보여주며, 이 새로운 접근법이 향후 통신 시스템 설계에 어떻게 기여할 수 있을지에 대한 기대감을 불러일으킵니다.



### Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods (https://arxiv.org/abs/2411.11795)
- **What's New**: 이 논문은 JPEG AI라는 새로운 신경 이미지 압축(NIC) 표준의 적대적 공격에 대한 강인성을 측정하는 새로운 방법론을 제안합니다. JPEG AI는 소비자 기기에 내장된 최초의 신경망 기반 모델로, 이미지 압축의 품질을 개선할 가능성이 있습니다. 최근 연구에서 NIC의 강인성에 대한 정의가 명확하게 제시되지 않아, 본 논문은 이를 평가하기 위한 첫 번째 대규모 평가를 수행하였습니다.

- **Technical Details**: JPEG AI는 이미지의 내부 표현을 비선형 변환을 통해 변환하는 분석 변환, 과도한 정보를 줄이기 위한 양자화, 예측 가능성에 따라 바이너리 데이터로 압축하는 엔트로피 코딩, 그리고 학습된 비선형 합성 변환을 사용하여 압축된 데이터에서 이미지를 재구성하는 합성 변환의 네 가지 핵심 요소로 구성됩니다. 본 연구에서는 JPEG AI를 포함한 10개의 NIC 모델을 대상으로 6가지 공격을 분석하고 여러 손실 함수에 대한 공격-손실 조합을 평가하였습니다.

- **Performance Highlights**: 제안된 방법론은 다양한 손실 함수와 공격을 조합하여 NIC의 강인성을 평가합니다. 본 연구는 JPEG AI의 강인성을 다른 NIC 모델과 비교하여 적대적 공격에 대한 상대적 효과를 분석하며, NIC의 취약점에 대한 정화 방어 전략을 적용하여 과학적으로 평가합니다. 이는 이미지 품질 또는 비트 전송률을 목표로 하는 다양한 공격을 포함하며, NIC 모델의 취약성을 극복하는 데 도움이 됩니다.



### Exploring the Requirements of Clinicians for Explainable AI Decision Support Systems in Intensive Car (https://arxiv.org/abs/2411.11774)
- **What's New**: 해당 논문은 AI 시스템이 임상 의사결정을 지원할 수 있는 방법에 대해 논의하고 있습니다. 특히, AI 모델이 복잡해짐에 따라 신뢰성에 대한 우려가 커지고 있으며, 이는 기술의 안전하고 효과적인 채택에 영향을 미칩니다. 연구팀은 중환자실(ICU) 의사들과의 그룹 인터뷰를 통해 의사결정 과정과 AI 지원 시스템의 요구사항을 분석했습니다.

- **Technical Details**: 연구에서는 xAI(설명 가능한 인공지능) 시스템의 설계와 구현을 위해 중환자실 의료 전문가의 요구사항을 이해하는 데 중점을 두었습니다. 인터뷰는 두 개의 세션으로 진행되었으며, 반구조화된 형식으로 진행되었습니다. 이를 통해 의사결정 기반의 프로세스와 의사소통 방식, AI 시스템의 요구사항이 도출되었습니다.

- **Performance Highlights**: 이 연구는 중환자실에서의 임상 의사결정 고려사항과 AI 지원 시스템 사용의 기회와 도전을 탐구하며, xAI 시스템 설계 시 고려해야 할 점들을 제시합니다. 이러한 고려사항은 중환자실 내 사용자 요구를 충족시키고, 향후 헬스케어 분야 전반에도 적용될 수 있는 잠재력을 지니고 있습니다.



### CNMBert: A Model For Hanyu Pinyin Abbreviation to Character Conversion Task (https://arxiv.org/abs/2411.11770)
Comments:
          9 pages, 2figures

- **What's New**: 본 논문에서는 Hanyu Pinyin 약어를 중국어 문자로 변환하는 새로운 접근법을 제시합니다. CNMBert라는 모델을 도입하여 기존 기존 GPT 모델들을 능가하는 성능을 기록했습니다. 특히, 10,424개의 샘플을 포함한 테스트 데이터셋에서 59.63%의 MRR(Metric Rank Rate)을 달성한 점이 인상적입니다.

- **Technical Details**: CNMBert는 Whole Word Mask Chinese BERT 모델을 기반으로 개발되었습니다. 이 모델은 Hanyu Pinyin의 각 알파벳 문자에 대해 고유한 마스크 토큰을 매핑하여, Fill-Mask 작업으로 접근합니다. 기존 BERT 모델의 단순한 마스크 구조를 확장하여 알파벳 수에 맞는 마스크 토큰을 사용함으로써, 보다 효과적으로 약어를 해석하고 변환하게 됩니다.

- **Performance Highlights**: CNMBert는 주요 성능 지표에서 기존의 다양한 모델보다 뛰어난 결과를 보였습니다. Hanyu Pinyin 약어 변환 작업에 대한 성능 평가에서, 본 모델은 기본 성능을 훨씬 초과하는 결과를 나타냈습니다. 이에 따라 CNMBert는 중국어 정수 교정(Chinese Spelling Correction) 분야에서 확고한 위치를 차지할 것으로 예상됩니다.



### AdaptLIL: A Gaze-Adaptive Visualization for Ontology Mapping (https://arxiv.org/abs/2411.11768)
- **What's New**: 본 논문에서는 AdaptLIL이라는 새로운 시스템을 소개합니다. 이 시스템은 사용자의 시선(eye gaze)을 주요 입력 소스로 활용하여 실시간(adaptive)으로 링크가 구분된 리스트 온톨로지(mapping ontology) 시각화를 제공합니다. 특히, AdaptLIL은 각 사용자의 시선에 기반하여 그래픽 오버레이(graphical overlays)를 줄이면(pairwise mappings) 시각화를 개인화할 수 있는 독특한 방법을 제시합니다.

- **Technical Details**: AdaptLIL은 여러 모드(multi-modal) 시스템을 결합하여 작동합니다. 이는 실시간(processing), 딥러닝(deep learning), 그리고 웹 개발(web development) 애플리케이션을 통합하여 구현됩니다. 사용자는 자신의 시선에 따라 자동으로 조정된 시각화(overlays)를 경험하게 되며, 이는 개인화된 데이터 인터페이스를 제공합니다.

- **Performance Highlights**: 이 시스템은 사용자 친화적인 경험을 제공하며, 실시간으로 다양한 사용자의 요구에 적응(adaptation)합니다. 이러한 기능은 특히 시각적 정보의 소비 방식에 혁신을 가져오고 있습니다. 감사육성이 뛰어난 시스템으로 평가받고 있으며, 이는 여러 분야에서 유용하게 활용될 수 있습니다.



### The Power of Many: Multi-Agent Multimodal Models for Cultural Image Captioning (https://arxiv.org/abs/2411.11758)
- **What's New**: 이번 연구에서는 Large Multimodal Models (LMMs)와 multi-agent 모델을 결합하여 문화 이미지 캡션 생성이라는 새로운 멀티모달 작업을 수행합니다. 연구의 주요 기여로는, MosAIC이라는 새로운 Multi-Agent 프레임워크를 소개하고, 서로 다른 문화적 페르소나를 가진 LMMs를 이용해 cross-cultural image captioning을 향상시키는 방법을 제안합니다. 또한, 중국, 인도, 루마니아의 이미지를 포함한 문화적으로 풍부한 이미지 캡션 데이터셋을 마련하고, 문화 정보를 평가할 수 있는 새로운 지표를 도입합니다.

- **Technical Details**: MosAIC 프레임워크는 5명의 에이전트로 구성되어 있으며, 각 에이전트는 특정 역할을 수행합니다. 세 개의 Social 에이전트는 중국, 인도, 루마니아의 문화를 대표하고, Moderator 에이전트는 대화의 방향을 유도하며, Summarizer 에이전트는 대화 내용을 요약합니다. 각 대화 라운드는 에이전트들이 서로 질문하고 답변하며, 각 문화를 반영한 상세한 이미지 설명을 만들어내는 구조로 이루어져 있습니다.

- **Performance Highlights**: 연구 결과, 모둠 에이전트 접근 방식이 단일 에이전트 모델보다 다양한 지표에서 우수한 성능을 보인다는 것을 확인했습니다. 에이전트들은 서로의 문화적 배경에 대해 질문하며, 이를 통해 문화적으로 풍부한 알고리즘들을 개발할 수 있게 되었습니다. 만일 후속 연구에서 이러한 방법이 지속적으로 발전된다면, 더 많은 문화적 맥락을 이해하고 반영할 수 있을 것으로 기대됩니다.



### QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou (https://arxiv.org/abs/2411.11739)
Comments:
          Work in progress

- **What's New**: 최근 멀티모달 대형 모델의 발전으로, 추천 시스템에서 사용자 관심 모델링을 위해 멀티모달 정보를 활용하는 가능성이 주목받고 있습니다. 이 논문에서는 Kuaishou 플랫폼을 위한 맞춤형 다중 모달 정보의 학습을 위한 정량적 멀티모달 프레임워크(QARM)를 소개합니다. QARM은 하향식 추천 모델에 최적화하여 비즈니스 특성을 고려한 다양한 멀티모달 표현 일치 메커니즘과 코드 메커니즘을 적용합니다.

- **Technical Details**: QARM은 두 가지 주요 메커니즘으로 구성됩니다. 첫째, 아이템 정렬 메커니즘은 다중 모달 모델의 사전 학습 결과를 비즈니스 데이터에 맞게 미세 조정하여 표현 일치를 최대화합니다. 둘째, 정량적 코드 메커니즘은 코드 해싱과 직통 추정기를 활용하여 다중 모달 표현을 학습 가능한 의미적 ID로 변환합니다. 이를 통해 기존의 정적 멀티모달 표현이 가진 제한사항을 해결하고 지원하는 방법을 제공합니다.

- **Performance Highlights**: QARM은 Kuaishou에서 400만 명의 활성 사용자를 지원하며 정량적 실험을 통해 광고 수익을 9.704% 증가시키고 온라인 쇼핑의 GMV를 2.296% 증가시킨 것으로 나타났습니다. 이러한 결과는 QARM이 다양한 서비스에서 유효함을 입증하며, 추천 시스템의 성능을 극대화하기 위한 효과적인 솔루션임을 보여줍니다.



### WoodYOLO: A Novel Object Detector for Wood Species Detection in Microscopic Images (https://arxiv.org/abs/2411.11738)
- **What's New**: 이번 연구에서는 Microscopic wood fiber analysis에 특화된 새로운 객체 탐지 알고리즘인 WoodYOLO를 소개합니다. WoodYOLO는 기존의 YOLO 아키텍처를 기반으로 하여, 대형 고해상도 현미경 이미지 처리와 관심 세포 유형(vessel elements)의 고정확도 로컬라이제이션을 위해 최적화된 방법을 적용합니다. 이 알고리즘은 YOLOv10 및 YOLOv7보다 각각 12.9%와 6.5% 높은 F2 점수를 기록하며, 자동화된 목재 세포 유형 로컬라이제이션 기능을 개선했습니다.

- **Technical Details**: WoodYOLO는 고해상도 현미경 이미지에 최적화된 커스터마이즈된 YOLO 기반 아키텍처로 구성되어 있습니다. 또한, 최대 너비와 높이만 정의하는 새로운 앵커 박스 지정 방법을 도입하여, F2 점수를 0.7% 향상시킵니다. 현대 객체 탐지기에서 다양한 아키텍처 결정을 포괄적으로 평가하여 COCO(Lin et al., 2015)와 같은 일반적인 데이터셋에 최적화된 방법이 실제 데이터셋에서는 반드시 성능 향상으로 이어지지 않음을 발견했습니다.

- **Performance Highlights**: WoodYOLO는 자동화된 목재 종 식별 능력을 개선하여, 규제 준수, 지속 가능한 산림 관리, 그리고 생물 다양성 보존 노력을 지원하는 데 기여합니다. 연구 결과, 이 새로운 알고리즘은 microscopy 이미지를 활용한 섬유 재료의 목재 종 식별을 위한 신뢰할 수 있는 방법으로 자리잡을 것으로 기대되며, 고해상도 이미지 처리에 대한 요구를 충족할 수 있습니다.



### Moral Persuasion in Large Language Models: Evaluating Susceptibility and Ethical Alignmen (https://arxiv.org/abs/2411.11731)
- **What's New**: 최근 논문에서는 대형 언어 모델(LLMs)들이 윤리적 프레임워크에 따라 초기 결정을 변경하도록 유도되는 방법을 탐구하고 있습니다. 두 가지 실험을 통해 LLM이 도덕적 설득에 얼마나 영향을 받는지를 평가했습니다. 첫 번째 실험은 기본 에이전트의 결정과 설득자의 개입을 비교하였고, 두 번째 실험은 LLM이 특정 윤리적 프레임워크에 어떻게 맞춰지는지를 조사했습니다.

- **Technical Details**: 이 연구는 도덕적으로 모호한 상황에서 LLM의 결정 과정을 탐구합니다. 첫 번째 단계에서는 기본 에이전트(LML)와 설득자 에이전트의 초기 점수를 비교하는 기준 평가를 진행하고, 두 번째 단계에서는 각 상황에 대한 대화 후 변화를 분석합니다. 이 실험은 대화의 턴 수나 사용된 LLM의 종류에 따른 영향을 테스트합니다.

- **Performance Highlights**: MGN에 따른 실험 결과, LLM은 윤리적이거나 모호한 상황에서 설득될 수 있으며, 이 설득의 성공 여부는 모델의 종류, 시나리오의 복잡성, 대화의 길이에 따라 달라진다는 것이 분명해졌습니다. 특히, 같은 회사의 서로 다른 크기의 LLM에서 상이한 결과가 나와, LLM의 도덕적 설득에 대한 민감도가 상당히 다름을 강조했습니다.



### Semantic-Geometric-Physical-Driven Robot Manipulation Skill Transfer via Skill Library and Tactile Representation (https://arxiv.org/abs/2411.11714)
- **What's New**: 이 논문에서는 로봇이 개방된 세계 환경에서 복잡한 작업을 수행하는 데 필요한 기술 전달을 효율적으로 지원하기 위해 지식 그래프 기반의 스킬 라이브러리 프레임워크를 제안합니다. 이 프레임워크는 작업 그래프와 장면 그래프를 통해 작업 및 장면의 의미 정보를 계층적으로 조직하여 로봇이 고층 기술 인식과 공간적 의미 이해를 갖도록 합니다. 또한, 다양한 시나리오에서 로봇 기술을 전이할 수 있도록 하는 계층적 전이 프레임워크를 도입합니다.

- **Technical Details**: 제안된 방법은 복잡한 장기 작업을 하위 작업으로 분해하여 이를 작업 그래프로 표현하고, 공간적 의미 정보를 나타내기 위해 장면 그래프를 사용합니다. 이 모델은 또한 로봇 작업과 장면 정보 간의 상호작용을 용이하게 하기 위해 상태 그래프를 도입합니다. 이 지식 그래프는 Neo4j를 활용하여 구현되며, 적응형 궤적 전이 방법과 촉각 인식에 기반한 자세 인식 방법을 포함합니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 복잡한 작업의 전이에 효과적임을 입증합니다. 특히, 대규모 언어 모델(LLMs)을 활용하여 작업 수준의 하위 작업 시퀀스 전이 및 제어 수준의 적응형 궤적 전이를 수행하는 데 성공했습니다. 또한, 시각-촉각 텍스처 데이터에서 고정밀 윤곽 및 자세 정보를 동적으로 획득하여 새로운 환경에서 기술의 효과성을 보장합니다.



### FedCoLLM: A Parameter-Efficient Federated Co-tuning Framework for Large and Small Language Models (https://arxiv.org/abs/2411.11707)
- **What's New**: FedCoLLM은 서버의 LLM(대용량 언어 모델)과 클라이언트의 SLM(소형 언어 모델)을 동시에 조정하는 혁신적인 분산 학습 프레임워크입니다. 이 프레임워크는 클라이언트의 데이터 프라이버시를 지키면서 효과적으로 지식을 전달할 수 있는 경량 어댑터를 사용합니다. FedCoLLM의 도입으로 인해 클라이언트의 SLM 성능이 크게 향상되었고, LLM도 클라이언트의 도메인 데이터로부터 유익함을 얻을 수 있게 되었습니다.

- **Technical Details**: FedCoLLM은 파라미터 효율적 파인튜닝(PEFT)과 지식 증류(Knowledge Distillation) 기술을 기반으로 합니다. 이 프레임워크는 클라이언트와 서버 사이의 효율적인 지식 교환을 위해 SLM과 함께 경량 어댑터를 사용합니다. 또한, FedCoLLM은 클라이언트의 데이터 프라이버시를 보장하기 위해 연합 학습(FL) 메커니즘을 활용합니다.

- **Performance Highlights**: 다양한 NLP(자연어 처리) 텍스트 생성 작업을 통해 FedCoLLM의 성능을 평가한 결과, 클라이언트의 SLM이 LLM의 지원 하에 유의미한 성능 향상을 경험했고, LLM 또한 클라이언트의 데이터에 직접 파인튜닝했을 때와 유사한 성능을 달성했습니다. FedCoLLM은 자원 소모를 줄이면서 더 나은 성과를 보여 주었습니다.



### MC-LLaVA: Multi-Concept Personalized Vision-Language Mod (https://arxiv.org/abs/2411.11706)
- **What's New**: 이번 연구에서는 다중 개념 개인화(Multi-concept personalization)를 위한 새로운 방법인 MC-LLaVA를 제안합니다. 기존의 연구들이 주로 단일 개념에 초점을 맞추었던 반면, MC-LLaVA는 여러 개념을 동시에 처리할 수 있는 능력을 보여줘 사용자의 제공된 정보에 기반하여 정확한 응답을 생성합니다. 또한, 고품질 다중 개념 개인화 데이터셋도 함께 발표하여 연구의 기초 자료를 제공하는 데 기여합니다.

- **Technical Details**: MC-LLaVA는 하나의 훈련 단계에서 여러 개념을 함께 고려하여 학습하는 방식으로, VLM(Vision-Language Model)의 시각 인코더를 통해 개념 이미지들을 전달하여 개념 토큰을 초기화합니다. 이는 공동 학습(joint training)의 비용을 줄이고, 개념의 표상을 개선하며, 훈련 속도를 가속화하는데 기여합니다. 연구팀은 약 1,600개의 이미지와 질문-답변 샘플을 포함하는 데이터셋을 마련하여, 이 데이터셋을 활용해 다양한 비전-언어 작업에서 MC-LLaVA의 성능을 평가했습니다.

- **Performance Highlights**: MC-LLaVA의 실험 결과는 여러 작업 유형에서 괄목할 만한 성능 향상을 보여줍니다. 특히 다중 개념 인식, 질문 응답(QA), 캡셔닝(captioning) 등에서 상태 수익(state-of-the-art) 성과를 달성하여 VLM이 사용자 맞춤형 비서로 발전할 가능성을 열었습니다. 이러한 결과는 개인화된 VLM 연구의 새로운 방향을 제시하며, 향후 연구에 있어 더 많은 발전이 기대됩니다.



### Technical Report: Enhancing LLM Reasoning with Reward-guided Tree Search (https://arxiv.org/abs/2411.11694)
Comments:
          LLM;Complex Reasoning;Math

- **What's New**: 최근 연구에서는 테스트 시간에 모델 성능을 높이기 위한 방법으로, LLM에 대해서 보상 기반 트리 탐색 알고리즘을 통한 추론 능력 강화를 제안하고 있습니다. 이 전략은 정책 모델, 보상 모델, 탐색 알고리즘의 통합으로 구성되며, 특히 동적으로 확장되는 트리를 탐색하는 방식으로 설계되었습니다. 이 프레임워크는 LLM의 수학적 문제 해결을 크게 향상시키는 효과를 보였으며, 다양한 기술적 고려사항이 포함되어 있습니다.

- **Technical Details**: 이 연구는 주로 수학적 도메인에 초점을 맞추고 있으며, 세 가지 주요 구성 요소인 정책 모델, 보상 모델 및 탐색 알고리즘으로 이루어진 보상 유도 트리 탐색 프레임워크를 구현하고 있습니다. 정책 모델은 탐색 트리 내의 주어진 경로를 따라 부분 솔루션 접두사에 기반하여 새로운 추론 단계를 생성합니다. 탐색 알고리즘은 이 과정을 구조화하며, 보상 모델은 정책 모델의 행동과 탐색 과정에 대한 피드백 신호를 제공합니다.

- **Performance Highlights**: 연구진은 MATH-OAI, GSM-Hard, Olympiad Bench, College Math 등 4개의 도전적인 수학 벤치마크 데이터셋에 대해 광범위한 평가를 실시하였으며, 실험 결과 정책 모델의 성능이 크게 향상됨을 입증했습니다. 또한, 정책 모델, 보상 모델 및 트리 탐색 알고리즘 설계에 대한 심층적인 실증 분석을 제공하여 향후 연구자들에게 의미 있는 지침을 제시하고자 하였습니다.



### Conceptwm: A Diffusion Model Watermark for Concept Protection (https://arxiv.org/abs/2411.11688)
- **What's New**: 이번 논문에서는 개인화된 이미지 생성에 따른 저작권 침해와 불법 사용 문제를 해결하기 위한 새로운 개념 지향 워터마킹 프레임워크, ConceptWm을 제안합니다. 기존의 워터마킹 방식들은 개별 개념에 대한 워터마크를 세밀하게 적용하기보다는 모든 이미지에 일률적으로 적용하는 경향이 있었습니다. 그러나 ConceptWm은 특정 개념에 대해 묵상할 수 있는 방안을 제공하며, 적은 수의 이미지를 사용하여 빈틈 없는 워터마킹을 실현합니다.

- **Technical Details**: ConceptWm은 정밀한 워터마크 훈련과 품질 보존 섭동(Fidelity Preserving Perturbation)을 핵심 요소로 삼고 있습니다. 이는 기존의 확산 모델에 제한된 데이터를 활용하여 워터마크 주입 패턴을 신속하게 학습할 수 있도록 설계되었습니다. 또한, 다양한 이미지 처리 공격에 대해 안정성을 보장하기 위해 워터마크를 잠재공간(latent space)에 통합하고 왜곡 레이어(distortion layer)를 도입했습니다.

- **Performance Highlights**: 실험 결과, ConceptWm은 기존의 네 가지 기초 모델에 비해 이미지 처리 공격에 대해 6.3% 더 나은 강인성과 정확성을 유지하며 워터마크를 통합하고 추출하는 데 성공적으로 작용했습니다. 이는 개념별 워터마킹의 새로운 가능성을 제시하며, 보다 효과적인 저작권 보호 수단이 될 것으로 기대됩니다.



### TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World (https://arxiv.org/abs/2411.11683)
Comments:
          Initial version with preliminary results. We welcome any feedback or suggestions

- **What's New**: 이 논문은 로봇 조작(robust manipulation) 분야에서 백도어 공격(backdoor attack)을 제안하며, 물리적 세계에서 이를 구현한 최초의 연구입니다. 기존의 백도어 공격 방식들이 디지털 환경에서만 적용 가능한 것에 반해, 본 연구는 로봇 시스템의 시각 인식 모듈에 백도어 비주얼 언어 모델을 삽입하여 일반 물체를 트리거로 사용해 로봇 팔의 작업을 오도하는 데 성공했습니다. 이러한 접근 방식은 로봇 조작의 보안 취약성을 실질적으로 드러냅니다.

- **Technical Details**: 연구에서는 플러그 앤 플레이 비전-언어 백도어 모델을 설계하여 로봇 시스템의 시각 인식 모듈에 주입합니다. 이 모델은 비밀 트리거를 감지하고 처리하며, 사용자 명령을 해석하는 언어 모듈을 포함합니다. 정상적인 상황에서는 명령을 정확히 파싱하지만, 시각적 트리거가 존재할 경우 악의적으로 명령을 해석하여 로봇 조작 결과를 변조합니다.

- **Performance Highlights**: 물리적 세계에서 실험을 통해 제안된 백도어 공격의 효과성을 검증하였습니다. 기존의 로봇 조작 공격 방식들이 물리적 세계에서 성공적인 공격을 달성하지 못한 반면, 본 연구의 방법론은 기민하고 효율적인 공격을 가능하게 하였습니다. 이로 인해 로봇 조작 분야의 보안 및 공격 가능성에 대한 이해를 한층 더 발전시킬 수 있음을 보여줍니다.



### Dissecting Misalignment of Multimodal Large Language Models via Influence Function (https://arxiv.org/abs/2411.11667)
Comments:
          34 pages

- **What's New**: 본 논문은 다중 모달 대형 언어 모델(MLLMs)의 데이터 정합성을 평가하기 위한 확장된 영향 함수(ECIF)를 제안합니다. 기존 영향 함수 방법들이 시간 소모적이거나 부적합한 점을 개선하여, 긍정 및 부정 샘플의 영향을 모두 고려할 수 있는 방법론을 제공합니다.

- **Technical Details**: ECIF는 대조 손실(contrastive loss)에 최적화된 영향 함수를 제공하며, 재교육 없이도 대조 학습 모델의 성능을 평가하는 폐쇄형 근사(closed-form approximation)를 포함합니다. 이 방법은 긍정 샘플과 부정 샘플의 두 가지 관점을 통합하여 데이터의 중요성을 평가합니다.

- **Performance Highlights**: 실험 결과, ECIF는 전통적인 기준 방법들에 비해 MLLM의 데이터 영향 평가 정확도를 향상시키고, 미스라벨링(mislabeled) 또는 정합성 문제(misalignment)를 효과적으로 감지할 수 있음을 입증하였습니다.



### No-regret Exploration in Shuffle Private Reinforcement Learning (https://arxiv.org/abs/2411.11647)
- **What's New**: 본 논문에서는 shuffle differential privacy (SDP)라는 새로운 차별적 프라이버시 모델을 도입하여 에피소드 강화 학습 (episodic reinforcement learning)에서 개인 정보 보호의 세밀한 절충점을 찾는 방법을 제안하고 있습니다.

- **Technical Details**: Shuffle Differentially Private Policy Elimination (SDP-PE) 알고리즘을 개발하였으며, 이는 SDP 제약 조건을 만족하며, 서브선형 (sublinear) 후회 (regret) 값을 달성하는 것을 목표로 합니다. 본 알고리즘은 노이즈 누적을 방지하기 위해 현재 단계의 데이터만을 사용하며, 탐색 (exploration) 단계를 배치 크기가 지수적으로 증가하도록 나누어서 정책 축소 (policy elimination)를 진행합니다.

- **Performance Highlights**: SDP-PE 알고리즘은 ε𝜀
LDP 및 ε𝜀
JDP와 비교할 때 개인 정보 보호와 후회 (regret) 사이의 거래를 개선할 수 있는 가능성을 보여주며, 중앙 집중식 모델에서 얻은 결과와 비교해 또한 높은 수준의 효율성을 달성했습니다.



### TSINR: Capturing Temporal Continuity via Implicit Neural Representations for Time Series Anomaly Detection (https://arxiv.org/abs/2411.11641)
Comments:
          Accepted by SIGKDD 2025

- **What's New**: 본 논문에서는 TSINR(Time Series Anomaly Detection based on Implicit Neural Representation)이라는 새로운 시간 시계열 이상 탐지 방법을 제안합니다. 이 방법은 이상 데이터를 효과적으로 빨리 인식하기 위해, 임플리시트 신경 표현(Implicit Neural Representation, INR)을 기반으로 하며, 저주파 신호를 우선적으로 학습하는 스펙트럼 바이어스(spectral bias) 특성을 활용합니다.

- **Technical Details**: TSINR은 시간을 연속 함수로 매개화하고, 변환기 기반 아키텍처(transformer-based architecture)를 사용하여 주어진 시계열 데이터의 INR 매개변수를 예측합니다. 새로운 형태의 연속 함수를 설계하여 추세, 계절성 및 잔여 정보를 독립적으로 학습하게 하며, 사전 훈련된 대형 언어 모델(LLM)을 활용해 원본 데이터의 강한 변동성을 방출하여 이상 탐지의 정확도를 높입니다.

- **Performance Highlights**: TSINR은 여러 멀티변량 및 단일 변량 시계열 이상 탐지 벤치마크에서 기존의 최첨단 재구성 기반 방법들과 비교했을 때 탁월한 전반적인 성능을 보여주며, 이상 점을 더욱 정밀하게 구분할 수 있도록 합니다. 다양한 실험을 통해 제안된 방법의 효과가 입증되었습니다.



### SP${ }^3$ : Superpixel-propagated pseudo-label learning for weakly semi-supervised medical image segmentation (https://arxiv.org/abs/2411.11636)
Comments:
          10 pages, 7 figures. Under Review

- **What's New**: 이 논문은 약한 반주석식 세분화(Weakly Semi-Supervised Segmentation, WSSS)에서의 부족한 감독 정보를 개선하기 위해 SuperPixel-Propagated Pseudo-label (SP${}^3$) 학습 방법을 제안합니다. 해당 방법은 스크리블 관련 주석을 슈퍼픽셀에 전파하여 밀집된 주석 결과를 얻는 방식으로, 이는 임상가의 수고를 줄입니다. 추가적으로, 동적 임계값 필터링과 슈퍼픽셀 수준의 불확실성을 적용하여 안정적인 학습을 도모하고 있습니다.

- **Technical Details**: 제안된 SP${}^3$ 방법은 3가지 주요 구성 요소로 이루어져 있습니다: 1) 스크리블 기반의 슈퍼픽셀 확장, 2) 주석 품질 향상을 위한 동적 임계값 필터링, 3) 슈퍼픽셀 수준의 불확실성 가이드. 이 접근 방식은 스크리블에 대한 도움이 되는 주석 정보를 최대한 활용하여 슈퍼픽셀 레벨의 밀집 주석을 얻습니다. 또한, 훈련 과정에서 슈퍼픽셀의 품질 평가를 통해 유용한 정보를 전파합니다.

- **Performance Highlights**: 이 방법은 두 개의 공개 데이터 세트에서 최첨단 성능을 보여줍니다. 특히, 전체 주석 방법과 비교할 때 단 3%의 주석 작업량만 가지고도 약 80%의 Dice 점수를 달성했습니다. 또한, 기존의 8가지 약한 및 반주석 기법들을 모두 초과하며, 매우 안정적이고 효율적인 자동 세분화가 가능하다는 점에서 임상가들에게 큰 도움이 될 것입니다.



### Chapter 7 Review of Data-Driven Generative AI Models for Knowledge Extraction from Scientific Literature in Healthcar (https://arxiv.org/abs/2411.11635)
Comments:
          16 pages, 5 figures, 1 table

- **What's New**: 이 논문은 추상적 자연어 처리(NLP) 기반의 텍스트 요약 기법의 발전을 검토하고 기존의 추출 기반 요약 기법과 비교합니다. 1950년대부터 시작된 텍스트 요약의 역사를 간략히 소개하며, Bidirectional Encoder Representations from Transformer (BERT)와 Generative Pre-training Transformers (GPT)와 같은 사전 훈련 언어 모델의 도입을 다룹니다.

- **Technical Details**: 총 60개의 연구가 PubMed와 Web of Science에서 확인되었으며, 이 중 29개는 제외되고 24개를 평가하여 적격성을 판단하였습니다. 결국 7개의 연구가 보다 심층적으로 분석되었습니다. 논문에서는 GPT-3와 최신 기술인 GPT-4 솔루션 간의 과학적 텍스트 요약 비교 사례도 포함되어 있습니다.

- **Performance Highlights**: 현재 자연어 처리는 짧은 텍스트 요약 생성에 있어 아직 완전한 잠재력을 발휘하지 못하고 있습니다. 특정 문제에 대한 우려가 인정되고 있으며, 이러한 모델의 실제 도입이 점진적으로 이루어질 것으로 기대됩니다.



### ST-Tree with Interpretability for Multivariate Time Series Classification (https://arxiv.org/abs/2411.11620)
Comments:
          Submitted on May 15, 2024, major revisions on Aug 31, 2024

- **What's New**: ST-Tree 모델은 Swin Transformer와 신경 결정 트리를 결합하여 다변량 시계열 분류의 정확도와 해석 가능성을 동시에 향상시킵니다.

- **Technical Details**: ST-Tree는 Self-Attention 메커니즘을 사용하여 세밀한 지역 패턴과 전역 패턴을 캡처하며, 멀티 스케일 특징 표현 학습을 통해 시계열 특징을 보다 포괄적으로 나타냅니다. 이 모델은 타임 패치 모듈과 뉴럴 트리 모듈의 두 부분으로 구성되어 있습니다.

- **Performance Highlights**: 10개의 UEA 데이터셋에 대한 실험 평가를 통해 ST-Tree 모델이 다변량 시계열 분류 작업에서 정확도를 개선하고, 다양한 데이터셋에 걸쳐 결정 과정의 해석 가능성을 시각화함으로써 효과를 입증했습니다.



### Signaling and Social Learning in Swarms of Robots (https://arxiv.org/abs/2411.11616)
Comments:
          17 pages, 3 Figures

- **What's New**: 이번 논문은 로봇 스왐(swarm)에서 통신의 역할을 연구하여, 분산된 방식으로 학습 및 실행이 동시에 이루어지는 패러다임을 강조합니다. 특히, credit assignment problem(개별 기여도 문제) 해결에서 통신이 갖는 중요성을 다루고 있습니다.

- **Technical Details**: 제안된 분류법은 정보 선택(information selection)과 물리적 추상화(physical abstraction)를 주요 축으로 하여, 로우 레벨의 손실 없는 압축(low-level lossless compression)부터 고수준의 손실 압축(high-level lossy compression)까지 다양한 통신 방법을 포함합니다. DLE(Decentralized Learning and Execution) 패러다임을 통해 로봇 간의 통신이 조정(coordination) 개선에 어떻게 기여하는지를 탐구합니다.

- **Performance Highlights**: 연구 결과, DLE 패러다임을 따르는 로봇 스왐이 환경에서의 실시간 학습 및 적응 능력을 향상시키는데 성공했으며, 이로 인해 기존의 학습 다중 로봇 시스템과 비교할 때 더욱 많은 구현 사례를 보여주고 있습니다.



### Hybrid Data-Driven SSM for Interpretable and Label-Free mmWave Channel Prediction (https://arxiv.org/abs/2411.11576)
- **What's New**: 본 논문은 밀리미터파(mmWave) 시간 변동 채널의 정확한 예측을 위한 새로운 하이브리드 방법인 KPIN을 제안합니다. KPIN은 전통적인 모델 기반 워크플로우에 데이터 기반 신경망을 통합하여 고속 이동 환경에서도 채널 변화를 효과적으로 추적할 수 있는 방법론을 제시합니다.

- **Technical Details**: KPIN은 전통적인 상태 공간 모델(SSM)을 바탕으로 하여 데이터에서 복잡한 채널 동태를 추적하는 새로운 방법입니다. 기존의 모델 기반 방법은 비선형 채널 동태를 효과적으로 추적하는 데 제한이 있으며 데이터 기반 방법은 대량의 레이블된 데이터가 필요하다는 단점을 가지고 있습니다. KPIN은 레이블 없는 데이터만으로 훈련 가능한 알고리즘을 통해 이러한 문제를 극복합니다.

- **Performance Highlights**: KPIN의 성능은 3GPP 밀리미터파 채널 모델을 기반으로 한 수치 시뮬레이션에서 기존의 최첨단 모델 기반 혹은 데이터 기반 방법들보다 우수한 예측 정확성을 보였습니다. 특히, 많은 실험을 통해 다양한 환경적 요인과 높은 노이즈 수준에서도 KPIN이 강력한 견고성을 보인 것이 입증되었습니다.



### Topology-aware Preemptive Scheduling for Co-located LLM Workloads (https://arxiv.org/abs/2411.11560)
Comments:
          17 Pages, 11 Figures, 5 Tables

- **What's New**: 이 논문은 다양한 큰 언어 모델(LLM) 워크로드를 통합 리소스 풀에서 공동 배치(co-location)하여 비용 효율성을 극대화하려는 접근 방식을 제안합니다. 특히, 지연에 민감한 온라인 서비스에 대한 리소스 우선 순위를 고려하여, 더욱 정교한 토폴로지 인식(topology-aware) 사전 예약(preemptive scheduling) 방법을 개발하였습니다. 이를 통해, 낮은 우선 순위의 작업이 해제할 때, 고우선 순위 서비스의 리소스 요구 사항을 충족할 수 있도록 하여 리소스 할당의 효율성을 향상시킵니다.

- **Technical Details**: 제안된 방법은 고우선 순위 서비스의 토폴로지 친화성(topological affinity) 요구를 충족하는 방식으로 자원은 배치됩니다. 이는 CPU-GPU 지역성(locality)을 보존하고, NUMA(비균일 메모리 접근) 노드 간 이동을 최소화하여 자원 관리를 최적화합니다. 기존의 사전 예약 방법과 달리, 이 접근법은 리소스 스케줄러가 토폴로지를 인식하도록 하여, 배치 성능 배율을 최대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 GPU 클러스터에서의 전체 자원 활용도를 55% 향상시키는 것으로 나타났습니다. 이를 통해 온라인 서비스의 성능을 개선함으로써, 비즈니스 우선 순위가 높은 서비스가 요구하는 리소스 작업을 더욱 효율적으로 수행할 수 있습니다. 따라서, 이 연구는 LLM 서비스 제공업체들에게 실질적인 이점과 비용 절감을 제공합니다.



### Real-Time Fitness Exercise Classification and Counting from Video Frames (https://arxiv.org/abs/2411.11548)
- **What's New**: 본 논문에서는 Bidirectional Long Short-Term Memory (BiLSTM) 신경망을 이용한 실시간 운동 분류를 위한 새로운 방법을 제안합니다. 기존의 운동 인식 방법들이 합성 데이터셋에 의존하거나 사용자 및 카메라 변화에 민감한 원시 좌표 입력을 사용하여 운동의 시간적 의존성을 완전히 활용하지 못했던 반면, 본 연구에서는 관절 각도와 원시 좌표를 함께 활용하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 모델은 BiLSTM 아키텍처를 기반으로 하여 30프레임의 시퀀스를 훈련하여 운동의 시간적 맥락을 포착할 수 있습니다. InfiniteRep 데이터셋과 Kaggle의 실제 비디오를 결합한 데이터셋에서 네 가지 일반적인 운동(스쿼트, 푸시업, 숄더 프레스, 이두 컬)을 포함하여 훈련 및 검증하였습니다. 모델의 정확도는 99% 이상을 기록하였고, 두 개의 별도 테스트 세트를 통해 일반화 능력을 평가하였습니다.

- **Performance Highlights**: 제안된 BiLSTM 모델은 기존 문헌의 접근 방식과 비교했을 때 가장 우수한 성능을 보였으며, 사용자가 수동으로 운동을 선택할 필요 없이 실시간 운동 분류 및 반복 수 세기를 지원하는 웹 애플리케이션에 통합되었습니다. 데모 및 데이터셋은 GitHub 저장소에서 이용 가능합니다.



### Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignmen (https://arxiv.org/abs/2411.11543)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.13581

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 여러 모달리티에서 정보 처리를 가능하게 하여 멀티모달 학습(multimodal learning)의 발전을 촉진했습니다. 비전-언어 모델(VLMs)은 이미지와 텍스트 기능을 통합하여 시각적 질문 응답, 이미지 캡션 생성 등 다양한 작업에서 눈에 띄는 성과를 거두었습니다. 그러나, 이러한 모델은 시각 모달리티에서 기존 안전 메커니즘을 회피당할 수 있는 취약점을 보여, 안전성을 보장하는 것이 중요한 과제가 되고 있습니다.

- **Technical Details**: 이 연구에서는 PSA-VLM(Progressive Safety Alignment for VLMs)이라는 개념을 제안하여, 비전-언어 모델의 시각 모달리티에 안전 개념을 통합합니다. PSA-VLM은 세 가지 핵심 안전 모듈인 Safety Projector, Safety Tokens 및 Safety Head를 도입하여 안전성과 제어성을 향상시킵니다. 이 방법은 인간이 해석 가능한 개념을 모델 아키텍처에 직접 통합하여 위험한 콘텐츠를 식별하고 개입할 수 있는 구조를 제공합니다.

- **Performance Highlights**: PSA-VLM은 주어진 데이터셋을 기반으로 철저한 평가를 거쳤으며, 일반적인 성능에 미치는 최소한의 영향을 주면서 안전성을 크게 향상시켰습니다. 또한, PSA-VLM은 기존 VLM 안전 기준 대비 최첨단 결과를 달성하였으며, 안전 정렬을 통해 모델 예측 및 사용자 개입을 개선할 수 있는 새로운 패러다임을 제시합니다.



### Addressing Hallucinations in Language Models with Knowledge Graph Embeddings as an Additional Modality (https://arxiv.org/abs/2411.11531)
- **What's New**: 이번 논문에서는 Knowledge Graphs (KGs)를 추가적인 모달리티로 포함하여 대형 언어 모델(Large Language Models, LLMs)에서의 환각(hallucination)을 줄이는 방법을 제안합니다. 입력 텍스트를 KG 임베딩으로 변환하고, 이를 언어 모델 공간에 통합하는 어댑터(adapter)를 사용하는 접근법을 소개하며, 외부 검색 프로세스에 의존하지 않습니다.

- **Technical Details**: 이 연구를 위해 300만 개 이상의 위키피디아 텍스트가 엔티티(entity)와 함께 주석 처리된 WikiEntities 데이터셋을 만들었습니다. 이 데이터셋은 엔티티 링크(Entity Linking) 모델을 훈련시키고 다양한 LLM에 어댑터를 적용하는 데 유용한 자원입니다. 모델을 세부 조정(fine-tuning)하지 않고 어댑터만 훈련하여 다른 작업의 성능이 영향을 받지 않도록 설계되었습니다.

- **Performance Highlights**: Mistral 7B, LLaMA 2-7B ( 챗 ), LLaMA 3-8B ( 지시 ) 모델에 어댑터를 훈련시키고, HaluEval, True-False 벤치마크 및 FEVER 데이터셋에서 성능 향상을 입증하였습니다. 그 결과, KGs를 새로운 모달리티로 통합하는 것이 환각을 효과적으로 줄이고 언어 모델의 사실 적 정확성(factual accuracy)을 개선하는 데 도움을 줄 수 있음을 보여주었습니다.



### Structure learning with Temporal Gaussian Mixture for model-based Reinforcement Learning (https://arxiv.org/abs/2411.11511)
- **What's New**: 이번 논문에서는 모델 기반 강화 학습(model-based reinforcement learning)에 대한 접근 방식을 제안합니다. 특히, 지각 모델(perception model)과 전이 모델(transition model)로 구성된 시계열 가우시안 혼합 모델(temporal Gaussian Mixture Model)을 소개합니다.

- **Technical Details**: 지각 모델은 연속 관찰을 통해 이산 상태(discrete (latent) states)를 추출하며, 변분 가우시안 혼합 가능도(variational Gaussian mixture likelihood)를 사용합니다. 이 모델은 수집된 데이터를 지속적으로 모니터링하여 새로운 가우시안 구성 요소를 찾습니다. 전이 모델은 연속적인 시간 단계 간의 전이를 배우기 위해 디리클레-카테고리 컨주게이(Dirichlet-categorical conjugacy)의 이점을 활용합니다. 두 모델 모두 일부 데이터 포인트를 잊을 수 있지만, prior 내에서 제공하는 정보를 통합하여 빠른 변분 추론(variational inference)을 보장합니다.

- **Performance Highlights**: 모델은 여러 미로에서 구조를 학습하는 능력을 보였으며, 상태(state)의 수와 상태 간 전이 확률(transition probabilities)을 발견하였습니다. 학습한 Q-값(Q-values)을 사용하여 에이전트는 시작 위치에서 미로의 출구로 성공적으로 탐색할 수 있었습니다.



### Closed-loop multi-step planning with innate physics knowledg (https://arxiv.org/abs/2411.11510)
- **What's New**: 이 논문은 로봇 계획을 입력 제어 문제로 해결하기 위한 계층적 프레임워크를 제시합니다. 이 프레임워크는 다양한 감각 입력에 따른 행동을 나타내는 '업무(tasks)'와 이를 지휘하는 '구성기(Configurator)'로 구성되어 있습니다. '구성기'는 물리 엔진으로서 '핵심(core) 지식'을 활용하여 업무의 생성 및 종료를 관리하고 모의 실험 결과를 해석해 계획을 수립합니다.

- **Technical Details**: 최하위 단계에서는 일시적인 통제 루프가 존재하며, 각 루프는 특정 감각 입력에 기반하여 행동을 생성합니다. 최상위 단계는 구성기로, 이는 다양한 이동 경로와 작업을 시뮬레이션하여 복잡한 길찾기 계획을 수립합니다. 이 때, 상태는 작업과 방해 요소의 조합으로 정의되며, 인지 맵을 통해 검색 가능하며 실행 계획을 추출할 수 있습니다.

- **Performance Highlights**: 이 프레임워크는 실제 로봇에 구현되어 추월 시나리오에서 테스트되었습니다. 실험 결과는 클로즈드 루프 행동이 복잡한 내비게이션 계획을 수립하는 데 효과적으로 작동함을 보여주었으며, 단순한 연결을 통해 빠른 반응을 가능하게 했습니다. 제안된 방법은 기존의 자율 내비게이션 기법보다 더 강력한 성능을 발휘할 수 있는 가능성을 제시합니다.



### HistoEncoder: a digital pathology foundation model for prostate cancer (https://arxiv.org/abs/2411.11458)
- **What's New**: 이번 연구에서는 HistoEncoder라는 기초 모델을 개발하여 전립선 암 디지털 병리학에 적용했습니다. 이 모델은 4800만 개의 전립선 조직 타일 이미지를 사전 학습하여 복잡한 생리적 패턴을 구별하며, 이는 제한된 자원으로 다양한 다운스트림 작업에 적응 가능합니다. HistoEncoder는 자연 이미지로 사전 학습된 모델보다 성능 면에서 우수한 결과를 보였으며, 두 가지 활용 사례를 통해 한정된 데이터로 훈련시키는 방법을 제시합니다.

- **Technical Details**: HistoEncoder는 crossover-covariance image transformers (XCiT)을 기반으로 하며, DINO라는 자가 감독 학습 방법을 사용하여 특징을 추출합니다. 이 모델은 디지털 병리학의 고해상도 이미지를 효율적으로 처리할 수 있도록 설계되었고, 두 가지 변형 (prostate-s와 prostate-m)로 훈련되었습니다. 기존 자연 이미지 기반 모델의 한계를 극복하고, 특성을 정확하게 클러스터링하여 자동 주석 달기에 활용될 수 있는 방법론을 개발하였습니다.

- **Performance Highlights**: HistoEncoder는 자동 주석 달기 및 전립선 암의 임상 예측 모델 향상에 기여했습니다. 모델을 사용하여 대규모 조직 이미지 데이터셋을 자동으로 주석 처리할 수 있는 높은 정확도를 달성하였으며, 임상 예측 모델에 통합하여 생존률 예측을 향상시켰습니다. 이 기초 모델은 제한된 자원만으로도 효과적인 임상 소프트웨어 도구를 구축할 수 있는 가능성을 보여줍니다.



### Unveiling the Inflexibility of Adaptive Embedding in Traffic Forecasting (https://arxiv.org/abs/2411.11448)
- **What's New**: 이 연구는 Spatiotemporal Graph Neural Networks (ST-GNNs)와 Transformer 모델이 교통 예측에서 보여준 한계와 잠재력을 조사했습니다. 특히, 도시의 교통 패턴 변화와 수요 증가에 대한 적응력을 높이기 위해 Principal Component Analysis (PCA) 임베딩 방법을 제안합니다.

- **Technical Details**: 제안된 PCA 임베딩 전략은 기존 ST-GNN 및 Transformer 아키텍처에 통합되어 모델이 재학습 없이도 새로운 시나리오에 적응할 수 있도록 지원합니다. 이는 그래프 구조의 유연성을 제공하여 한 도시에서 훈련된 모델이 다른 도시에서 제로샷 예측을 수행할 수 있게 합니다.

- **Performance Highlights**: 연구 결과에 따르면, 대규모 데이터셋으로 훈련된 모델은 더 높은 제로샷 일반화 성능을 보였으며, Transformer 기반 아키텍처가 GNN 기반 모델보다 우수한 성능을 발휘하는 것으로 나타났습니다. 이러한 PCA 임베딩을 통해 모델의 견고성과 일반화 능력을 향상시킬 수 있음을 제시합니다.



### Implicit Regularization for Multi-label Feature Selection (https://arxiv.org/abs/2411.11436)
Comments:
          11 pages, 7 figures, My paper is currently under review at TPAMI journal

- **What's New**: 이 논문에서는 multi-label learning의 맥락에서 특징 선택(feature selection) 문제를 다루고 있으며, 새로운 estimator를 도입하여 implicit regularization과 label embedding을 사용하는 접근법을 제안합니다.

- **Technical Details**: 기존의 sparse feature selection 방법에서 사용되는 penalized estimator는 $l_{2,1}$-norm, MCP, SCAD와 같은 explicit regularization term을 이용합니다. 이에 반해, 본 논문에서는 Hadamard product parameterization을 통해 보다 간단한 방법을 제안하고 있습니다. 또한, 특징 선택 과정을 안내하기 위해 multi-label 정보의 잠재적 의미(latent semantic)를 활용한 label embedding 방법이 채택되고 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 estimator가 타는 benchmark 데이터셋에서 추가적인 bias에 크게 영향을 받지 않으며, benign overfitting으로 이어질 수 있음을 보여줍니다.



### IKEA Manuals at Work: 4D Grounding of Assembly Instructions on Internet Videos (https://arxiv.org/abs/2411.11409)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 본 논문에서는 IKEA 가구 조립에 관한 새로운 멀티모달 데이터셋인 IKEA Video Manuals를 소개합니다. 이 데이터셋은 3D 모델, 지침 매뉴얼, 조립 동영상 및 이들 간의 밀접한 시공간 정렬(Spatio-temporal alignments)을 포함하고 있습니다.

- **Technical Details**: IKEA Video Manuals 데이터셋은 98개의 조립 동영상에서 34,441 개의 주석이 포함된 비디오 프레임으로 구성되며, 여기에는 2D-3D 파트 대응관계, 시간적 단계 정렬 및 파트 분할(Part segmentation)이 포함됩니다. 각 조립 동영상은 6-DoF(6 degrees of freedom) 포즈의 3D 파트를 사용하여 시각적 정렬을 수행합니다.

- **Performance Highlights**: 조립 계획 생성(assembly plan generation), 파트 조건부 분할(part-conditioned segmentation), 파트 조건부 포즈 추정(part-conditioned pose estimation), 비디오 객체 분할(video object segmentation), 지침 비디오 매뉴얼을 기반으로 한 가구 조립 등 다섯 가지 응용프로그램을 통해 이 데이터셋의 유용성을 입증하였습니다. 또한 실험 결과를 통해 비디오에서 조립 지침을 정립하는 데 있어 발생하는 다양한 문제를 강조하였습니다.



### The GECo algorithm for Graph Neural Networks Explanation (https://arxiv.org/abs/2411.11391)
- **What's New**: 이번 연구에서는 그래프 분류 문제의 해석 가능성을 개선하기 위한 새로운 방법론인 GECo(Graph Explainability by COmmunities)를 제안합니다. 이 방법은 그래프 내 커뮤니티의 기여를 분석하여 그래프의 중요한 구조를 강조합니다.

- **Technical Details**: GECo는 먼저 전체 그래프를 분류한 후, 여러 커뮤니티를 감지합니다. 각 커뮤니티에 대해 서브 그래프를 생성하고, 해당 서브 그래프가 예측된 클래스에 얼마나 기여하는지를 평가합니다. 최종적으로, 확률 값이 설정된 임계값보다 높은 커뮤니티가 모델의 결정에 필수적이라고 간주되어, 이들 커뮤니티가 최종 설명을 형성합니다.

- **Performance Highlights**: GECo는 인공 그래프 데이터 세트와 실제 데이터 세트에서 Graph Convolutional Networks를 사용하여 테스트되었으며, PGMExplainer, PGExplainer, GNNExplainer 및 SubgraphX와 같은 주요 설명 가능성 방법과 비교했을 때 인공 그래프 데이터 세트와 대부분의 실제 데이터 세트에 대해 우수한 성능을 보여주었습니다.



### Continual Task Learning through Adaptive Policy Self-Composition (https://arxiv.org/abs/2411.11364)
Comments:
          21 pages, 8 figures

- **What's New**: 이번 연구에서는 오프라인 지속적 강화학습(Continual Offline Reinforcement Learning, CORL)에 대한 포괄적인 벤치마크인 Offline Continual World를 개발하고, 기존의 지속적 학습(Continual Learning, CL) 방법들이 최악의 망각(catasrophic forgetting) 문제를 겪고 있음을 보여줍니다. 이 연구는 특히 CORL 시나리오에서 전통적 CL 기법의 한계를 다루고 있습니다.

- **Technical Details**: 본 연구에서 소개하는 CompoFormer는 메타 정책 네트워크(meta-policy network)를 통해 이전 정책들을 적응적으로 구성하는 구조 기반의 지속적 변환기 모델입니다. 새로운 작업을 만났을 때 CompoFormer는 의미적 상관관계를 이용하여 관련된 이전 정책을 통합하고 새로운 매개변수를 곁들여 학습을 가속화합니다. 이 과정은 새로운 작업이 도입될 때 정책의 연쇄적 구조(cascading structure)를 자연스럽게 형성합니다.

- **Performance Highlights**: 실험 결과, CompoFormer는 기존 CL 방법들보다 특히 더 긴 작업 시퀀스에서 현저한 성능 향상을 보여주었으며, 망각 억제(decrease in forgetting)에도 효과적이었습니다. CompoFormer는 플라스틱성과 안정성(plasticity and stability) 간의 최적의 균형을 이룹니다.



### A comprehensive survey of oracle character recognition: challenges, benchmarks, and beyond (https://arxiv.org/abs/2411.11354)
- **What's New**: 이 논문은 오라클 문자 인식(OrCR) 분야에 대한 체계적이고 구조적인 검토를 제공하며, 이전의 문헌 부족 문제를 해결하고자 합니다. 오라클 문자 인식은 고대 중국의 역사와 문화를 이해하는 데 중요한 역할을 하며, 해당 분야의 최신 연구 동향, 도전 과제, 데이터셋, 방법론, 관련 작업, 미래 연구 방향을 조망합니다.

- **Technical Details**: OrCR의 주요 도전 과제는 세 가지로, 문자의 변동성(writing variability), 데이터 부족(data scarcity), 저해상도 이미지(low image quality)입니다. 오라클 문자 작성은 표준화된 규범이 없었기 때문에 모든 문자가 다양한 변형을 가지고 있으며, 데이터 샘플로서의 오라클 문자의 수가 매우 제한적입니다. 또한 장기간 매장된 오라클 뼈의 이미지는 품질 저하를 겪어 오라클 문자 인식의 어려움을 증가시킵니다.

- **Performance Highlights**: 오라클 문자 인식을 자동화하기 위한 현대의 연구 접근법은 전통적인 패턴 인식에서 심층 학습(deep learning) 기법으로 발전해왔습니다. 그러나, OrCR에 관한 체계적 문헌이 부족한 현 상황을 고려할 때, 이 논문은 연구자들에게 OrCR 분야의 진행 상황을 이해하는 데 기여할 것입니다. 따라서, 이 논문은 신규 연구자와 경험이 있는 연구자 모두에게 유용한 기초 자료가 될 것입니다.



### Mitigating Knowledge Conflicts in Language Model-Driven Question Answering (https://arxiv.org/abs/2411.11344)
- **What's New**: 이번 연구에서는 문서 질문 답변과 요약 생성 등 지식 기반 시퀀스-투-시퀀스(nird)+ 생성 작업에서 공통적으로 나타나는 명백한 문제인 "hallucination"을 다룹니다. 저자들은 입력 소스와 생성된 내용 간의 명확한 상관관계를 통해 이러한 현상을 완화할 수 있다고 주장하며, 특히 질문 답변 과정에서의 엔티티 기반 지식 충돌이 모델의 동작에 미치는 영향을 분석합니다.

- **Technical Details**: 지식 충돌(knowledge conflicts)을 정의하고 이를 해결하기 위한 간단한 데이터 증강(data augmentation) 기반 솔루션을 제안합니다. Seq2seq 모델(BART 등)을 사용하여 훈련 데이터셋이 긴 문맥을 담고 있을 경우 발생하는 어려움을 해결하기 위해 사전 훈련된 생성 언어 모델을 손질(fine-tune)하여 QA 작업에 사용할 수 있도록 조정했습니다. 또한, 주어진 질문(q)와 컨텍스트(c)에 대한 응답을 생성하는 과정에서 모델이 컨텍스트를 무시하고 직접적인 추정(p_theta(x|q))을 하는 문제를 다룹니다.

- **Performance Highlights**: 연구 결과, 제안된 방법을 통해 모델의 생성된 출력이 입력 컨텍스트에 대해 더욱 충실하게 되며, 사실적 일관성을 유지한다고 보장할 수 있음을 보여줍니다. 모델의 성능 저하를 최소화하면서도 'hallucination'을 방지하기 위한 두 가지 방법, 즉 gradient 기반 디코딩과 adapter 기반 파인튜닝을 구현하였습니다. 이는 QA 모델에서 사용자와의 신뢰성을 높이고, 불필요한 AI 모델 혼란을 줄이는 데 기여할 것으로 기대합니다.



### Study of the Performance of CEEMDAN in Underdetermined Speech Separation (https://arxiv.org/abs/2411.11312)
Comments:
          in Arabic language

- **What's New**: 이 연구는 CEEMDAN 알고리즘이 비정상 신호(non-stationary signals) 분석에서 얼마나 효과적인지를 다룹니다. 특히 오디오 소스 분리(audio source separation)에서의 성능을 평가하여 이 방법의 한계를 검토합니다. 두 가지 조건, 즉 주파수 및 진폭(frequencies and amplitudes)과 관련된 한계를 정의하였습니다.

- **Technical Details**: CEEMDAN(Complete Ensemble Empirical Mode Decomposition with Adaptive Noise) 알고리즘은 신호 처리에 필수적이며, 이를 통해 노이즈(noise)와 음성을 분리하는 것을 연구합니다. 본 연구는 Matlab 환경에서 시뮬레이션을 진행하였고, Noizeus 데이터베이스를 활용하여 실제 데이터를 기반으로 한 결과를 도출하였습니다.

- **Performance Highlights**: 연구 결과, CEEMDAN은 음성 신호에서 특정 유형의 노이즈를 제거할 수 있음을 발견하였으며, 이는 음성 향상(speech improvement)에 기여합니다. 그러나, 서로 다른 음성 신호를 효과적으로 분리하지 못하는 문제는 상기되었습니다(캐주얼 파티 문제, cocktail party problem).



### TP-UNet: Temporal Prompt Guided UNet for Medical Image Segmentation (https://arxiv.org/abs/2411.11305)
- **What's New**: 이 논문에서는 기존 UNet 기반의 의료 이미지 세분화 방법들이 시간적 정보(temporal information)를 고려하지 못하는 문제를 해결하기 위해 TP-UNet이라는 새로운 프레임워크를 제안합니다. TP-UNet는 시간적 단서(temporal prompts)를 사용하여 세분화 모델이 의료 이미지를 학습하는 과정에서 시간적 정보를 효과적으로 통합할 수 있도록 도움을 줍니다. 또한, 이 프레임워크는 비지도 대비 학습(unsupervised contrastive learning) 기반의 의미 정렬(semantic alignment) 메커니즘을 활용하여 이미지와 텍스트 간의 의미적 격차를 줄입니다.

- **Technical Details**: TP-UNet의 핵심 구성 요소는 고차원의 임베딩(embedding) 과정과 교차 주의 메커니즘(cross-attention mechanism)입니다. 이 프레임워크는 두 개의 단계인 의미 정렬과 모달 융합(modal fusion)으로 구성되어 있으며, 이를 통해 시간적 단서와 이미지 특징 간의 개념적 일치를 도모합니다. 또한, 모델의 인코더는 텍스트 및 이미지의 특징 맵을 서로 통합하여 최종적으로 UNet의 디코더(decoder) 입력으로 사용되는 통합된 표현을 생성합니다.

- **Performance Highlights**: 리츠(LITS) 2017 데이터셋 및 UW-Madison 데이터셋을 포함한 두 개의 의료 이미지 세분화 데이터셋에서 TP-UNet의 성능을 광범위하게 평가한 결과, 새로운 최첨단(SOTA) 성능을 달성한 것으로 나타났습니다. 이는 TP-UNet이 시간적 정보를 활용하여 기존의 의료 이미지 세분화 기술을 개선할 수 있음을 의미합니다. 이러한 연구 결과는 향후 의료 이미지 세분화 기술 발전에 중요한 기여를 할 것으로 기대됩니다.



### Recurrent Stochastic Configuration Networks with Incremental Blocks (https://arxiv.org/abs/2411.11303)
- **What's New**: 새롭게 제안된 논문은 블록 증분 방식을 활용한 순환 확률 구성 네트워크(Recursive Stochastic Configuration Networks, RSCN)의 변형인 블록 RSCN(Block RSCN)을 개발하였습니다. 이를 통해 학습 능력과 네트워크의 효율성을 향상시켰습니다.

- **Technical Details**: 블록 RSCN은 추가 저수지 노드(subreservoirs)를 동시에 추가할 수 있으며, 각 서브저수지는 고유한 구조로 구성됩니다. 이를 통해 보편적인 근사화 속성을 보장합니다. 저수지 피드백 매트릭스는 적절히 조정되어 네트워크의 에코 상태 속성을 보장합니다. 또한 출력 가중치는 프로젝션 알고리즘을 이용하여 온라인으로 업데이트 되며, 매개변수 수렴을 촉진하는 지속적 자극 조건도 설정됩니다.

- **Performance Highlights**: 수치 결과는 시간 시계열 예측, 비선형 시스템 식별 작업 및 두 가지 산업 데이터 예측 분석에서 제안된 BRSCN이 모델링 효율성, 학습 및 일반화 성능 측면에서 우수함을 보여줍니다. 따라서 BRSCN은 복잡한 동적 시스템을 처리하는 데 있어 큰 잠재력을 지니고 있음을 강조합니다.



### Towards Personalized Brain-Computer Interface Application Based on Endogenous EEG Paradigms (https://arxiv.org/abs/2411.11302)
Comments:
          Submissoion version for IEEE International BCI Winter Conference 2025

- **What's New**: 이 논문에서는 개인화된 뇌-컴퓨터 인터페이스(BCI) 응용 프로그램을 위한 개념적 프레임워크를 제안합니다. 이 프레임워크는 사용자의 선호도와 필요에 기반하여 EEG(뇌전도) 신호를 활용하여 맞춤형 서비스를 제공합니다. 주요 구성 요소로는 사용자 식별(user identification)과 의도 분류(intention classification)가 포함되어 있으며, 이를 통해 개인화된 서비스를 제공하기 위한 가능성을 검증합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 주요 작업, 즉 사용자 식별과 의도 분류를 수행해야 합니다. 사용자 식별은 EEG 신호를 기반으로 사용자 고유의 특성을 추출하여 사용자의 선호도를 이해하는 데 도움을 줍니다. 의도 분류는 EEG를 통해 사용자가 원하는 행동을 결정하는 과정을 포함하며, MI(motor imagery), SI(speech imagery), VI(visual imagery)와 같은 내재적 EEG 패러다임이 사용됩니다.

- **Performance Highlights**: 실험 결과, 사용자 식별의 평균 분류 정확도는 0.995로 높게 나타났으며, 의도 분류는 모든 패러다임에서 0.47의 정확도를 보였습니다. 특히, MI는 가장 우수한 성능을 보여주었습니다. 이러한 결과는 EEG 신호가 개인화된 BCI 응용 프로그램을 지원하며, 특히 MI와 SI에서 신뢰할 수 있는 의도 해독(intention decoding)을 제공할 수 있음을 시사합니다.



### Transcending Language Boundaries: Harnessing LLMs for Low-Resource Language Translation (https://arxiv.org/abs/2411.11295)
- **What's New**: 본 논문에서는 저자원이용 가능 언어 (low-resource language)의 번역 품질을 향상시키기 위해 retrieval-based 방법론을 제안합니다. 이 방법은 키워드 중심의 번역을 통해 기존 데이터에서 예시를 검색하는 방식을 활용하여 번역 품질을 높이는 데 중점을 두고 있습니다. 저자들은 체로키어, 티베트어, 만주어와 같은 세 가지 저자원 언어에 대한 실험을 통해 본 방법의 효용을 평가합니다.

- **Technical Details**: 이 연구는 영어에서 체로키어로 번역하는 과정에서 GPT-4o 및 LLaMA 3.1 모델들을 비교합니다. 실험에서는 BLEU, ROUGE, BERTScore와 같은 자동 평가 지표와 전문가의 인간 평가를 사용하여 번역 품질을 측정하였습니다. 제안된 retrieval-augmented generation (RAG) 기법을 통해 모델이 저자원 언어의 번역에서 나타나는 일반화 문제를 해결하고자 하였습니다.

- **Performance Highlights**: 결과적으로, 본 연구의 방법은 어휘 수준의 정확성과 전체 의미 이해에서 상당한 개선을 보여주었습니다. GPT-4o 및 LLaMA 3.1의 제로샷 성능과 비교할 때, 제안된 접근법은 저자원 언어 번역 문제를 해결하는 데 있어 긍정적인 결과를 도출했습니다. 이는 저자원 언어로의 번역을 향상시키고 궁극적으로 문화와 언어의 보존에도 기여할 수 있는 가능성을 시사합니다.



### LP Data Pipeline: Lightweight, Purpose-driven Data Pipeline for Large Language Models (https://arxiv.org/abs/2411.11289)
- **What's New**: 본 논문은 Lightweight, Purpose-driven (LP) Data Pipeline이라는 CPU 기반 데이터 처리 프레임워크를 소개합니다. 기존의 GPU 의존 방식에 비해, 이 새로운 파이프라인은 시간과 비용 절감을 통해 고품질 데이터셋을 제공합니다. 또한 도메인 및 언어에 맞춘 데이터셋 생성을 가능하게 하여 LLM의 적용 범위를 확장하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: LP Data Pipeline은 네 가지 핵심 원칙에 기반하여 설계되었습니다. 이 파이프라인은 초기 필터링과 데이터 정화를 먼저 수행하고, 자원 집약적인 작업은 이후 단계에서 처리함으로써 효율성을 극대화합니다. 또한, FastText 기반의 언어 식별 도구를 활용하여 언어별 데이터 생성을 지원하며, 정보 업데이트를 자동화하는 워크플로우 관리 도구인 Airflow를 통합합니다.

- **Performance Highlights**: 실험을 통해 LP Data Pipeline이 대규모 데이터 처리를 위한 시간을 단축시키고, 비용을 효과적으로 절감하는 능력을 보여줍니다. 특히, 특정 도메인을 위한 데이터셋 생성 시, 필요한 법적 용어나 사례를 반영하여 LLM의 성능을 향상시키는 데 중점을 두었습니다. 이로 인해 개발자들은 보다 전문화된 언어 모델을 효율적으로 구축할 수 있습니다.



### Zero-Shot Automatic Annotation and Instance Segmentation using LLM-Generated Datasets: Eliminating Field Imaging and Manual Annotation for Deep Learning Model Developmen (https://arxiv.org/abs/2411.11285)
- **What's New**: 본 연구에서는 사과의 인스턴스 세분화 (instance segmentation)를 위한 새로운 방법을 제안합니다. 이 방법은 노동집약적인 현장 데이터 수집과 수작업 주석 작업을 대체하여, 대규모 언어 모델 (LLM)을 활용해 합성 이미지를 생성하고 자동으로 주석을 달 수 있도록 합니다. 이를 통해 물리적 센서와 수작업 데이터 처리에 대한 의존성을 크게 줄이며 농업 AI 분야의 중요한 발전을 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 Segment Anything Model (SAM)과 YOLO11 모델을 통합하여, 생성된 합성 데이터셋을 사용해 YOLO11 모델을 학습했습니다. 자동으로 생성된 주석은 마스크 정확도 (mask precision) 0.902와 mAP@50 0.833을 기록하며, 실제 과수원 이미지를 검사의 기반으로 사용되었습니다. 이 과정에서 합성 데이터와 자동 주석으로 훈련된 YOLO11 모델이 사과를 올바르게 감지하고 구분할 수 있는 방법의 효율성을 보여줍니다.

- **Performance Highlights**: 특히 YOLO11m-seg 구성은 상업 과수원에서 수집한 테스트 이미지에서 마스크 정확도 0.902, mAP@50 0.833을 달성했습니다. 또한 YOLO11l-seg 구성은 40개의 LLM 생성 이미지에서 다른 모델들을 초과 달성하여 가장 높은 마스크 정확도와 mAP@50 지표를 기록하였습니다. 이 결과들은 자동으로 생성된 주석의 정확성과 효과를 검증하며, 전체적으로 지역화 및 세분화 작업의 품질을 높이는 데 기여합니다.



### Multi-Hyperbolic Space-based Heterogeneous Graph Attention Network (https://arxiv.org/abs/2411.11283)
Comments:
          Accepted in IEEE ICDM 2024

- **What's New**: 이번 연구에서는 여러 개의 하이퍼볼릭(hyperbolic) 공간을 사용하여 이질적인 그래프의 파워-로우(power-law) 구조를 효과적으로 포착하는 Multi-hyperbolic Space 기반 이질적 그래프 어텐션 네트워크(MSGAT)를 제안합니다. 기존의 하이퍼볼릭 이질적 그래프 임베딩 모델들이 하나의 하이퍼볼릭 공간만을 사용하여 다양한 구조를 효과적으로 포착하지 못했던 한계를 극복합니다.

- **Technical Details**: MSGAT는 메타패스(metapath)를 통해 이질적인 그래프 내의 다양한 의미적 구조를 학습합니다. 하이퍼볼릭 공간 내에서 자가 어텐션(self-attention) 기법을 사용하여 각 메타패스에 특화된 임베딩을 학습하고, 서로 다른 메타패스 간의 정보 집계를 위해 상호 하이퍼볼릭 공간 어텐션(inter-hyperbolic space attention)을 활용합니다. 또한, 곡률(curvature) 파라미터를 학습하여 각 파워-로우 구조를 효과적으로 표현합니다.

- **Performance Highlights**: 실험 결과, MSGAT는 다양한 그래프 기계 학습 작업에서 최신 기준(lines of state-of-the-art) 모델들을 능가하며, 이질적인 그래프의 복잡한 구조를 효과적으로 포착하는 성능을 보여줍니다.



### Continuous K-space Recovery Network with Image Guidance for Fast MRI Reconstruction (https://arxiv.org/abs/2411.11282)
- **What's New**: 이번 연구에서는 MRI 이미지를 고품질로 복원하기 위해 연속 k-space 복원 네트워크를 제안했습니다. 기존 연구들은 일반적인 이미지 처리 네트워크를 k-space 복원에 직접 적용하여 k-space의 독특한 특성을 간과했지만, 본 연구는 이미지 도메인 가이드를 활용한 암시적 신경 표현(In implicit neural representation) 관점으로 접근했습니다. 이를 통해 MRI 재구성 성능을 향상시킬 수 있습니다.

- **Technical Details**: IGKR-Net은 암시적 신경 표현을 기반으로 한 인코더-디코더 구조를 사용하여 샘플링되지 않은 k-값을 지속적으로 질의합니다. 이 구조는 표준 트랜스포머를 포함하여 다중 헤드 셀프-어텐션(multi-head self-attention) 및 피드-포워드 네트워크(feed-forward network)로 구성되어 k-space의 신호를 직접 처리합니다. 또한, 저품질 MRI 이미지에서 의미 정보를 추출해 k-space 복원을 안내하는 이미지 도메인 가이드 모듈이 설계되었습니다.

- **Performance Highlights**: CC359, fastMRI, IXI의 세 가지 공공 데이터셋에 대한 광범위한 실험을 통해 제안한 방법이 기존의 복원 방식보다 우수함을 입증했습니다. 연구 결과, IGKR-Net은 이미지 도메인 가이드와 다단계 훈련 전략을 통해 과도한 매끄러움 및 왜곡 문제를 완화하면서 정밀한 결과를 도출할 수 있음을 보여주었습니다. 이 접근법은 MRI 영상의 질을 개선하는 데 중요한 기여를 할 것으로 기대됩니다.



### Cross-Patient Pseudo Bags Generation and Curriculum Contrastive Learning for Imbalanced Multiclassification of Whole Slide Imag (https://arxiv.org/abs/2411.11262)
Comments:
          9 pages, 4 figures

- **What's New**: 이 논문은 병리학적 이미지 분석에서 멀티클래스 샘플 불균형 문제를 해결하기 위해 기초 분포와 유사한 서브백(sub-bag)을 생성하여 세부 정보 학습을 제안합니다. 또한, 효율적인 학습을 위해 의사 백(pseudo-bag) 생성을 활용하여 다량의 정보를 가져올 수 있습니다. 새롭게 도입된 샘플 선택 및 커리큘럼 대조 학습 전략은 모델의 안정성을 높이며, 백 수준의 표현 학습에서 멀티 인스턴스 백의 특징 분포로 전환하는 혁신적인 전략입니다.

- **Technical Details**: 제안된 프레임워크는 병리학적 이미지를 서브백으로 나누어 유사한 특징 분포를 재조직하고, 이를 통해 샘플 불균형 문제를 해결합니다. 또한, 대조 학습을 위한 친화적인 샘플 선택 전략을 사용하여 특징 표현 강화를 극대화하고, 각 카테고리를 균형 있게 다룹니다. 이 과정에서 대량의 중복 데이터를 활용하여 견고한 모델 학습이 가능하게 합니다.

- **Performance Highlights**: 본 방법은 종양 분류와 림프절 전이 등의 세 가지 데이터셋에서 다중 클래스 분류의 불균형 문제를 효과적으로 해결하며, F1 점수에서 평균 4.39 점의 성능 향상을 기록했습니다. 이는 이전의 두 번째로 우수한 방법에 비해 현저한 성과로, 이 연구의 우수성을 강조합니다.



### EXCON: Extreme Instance-based Contrastive Representation Learning of Severely Imbalanced Multivariate Time Series for Solar Flare Prediction (https://arxiv.org/abs/2411.11249)
Comments:
          This work has been accepted at the 2024 IEEE International Conference on Big Data (IEEE BigData 2024) on October 27, 2024, as a main conference paper

- **What's New**: 이 논문은 태양 폭발 예측을 위한 새로운 접근법인 EXCON(Contrastive Representation Learning Framework)을 제안합니다. EXCON은 다변량 시계열 데이터에서 효과적으로 클래스 불균형(class imbalance)을 해결하기 위한 전략을 포함합니다.

- **Technical Details**: EXCON은 네 가지 단계로 구성됩니다: 1) 다변량 시계열 데이터에서 핵심 특징을 추출, 2) 각 클래스에 대해 상이한 대비 표현을 선택하여 클래스 간 분리를 극대화, 3) 사용자 정의 극단 재구성 손실(extreme reconstruction loss)을 사용하는 시간적 특징 임베딩 모듈 훈련, 4) 학습된 임베딩에 대해 분류기를 적용하여 강력한 분류 수행.

- **Performance Highlights**: EXCON은 벤치마크 태양 폭발 데이터셋 및 여러 시계열 아카이브 데이터셋에서의 평가 결과를 통해 분류 성능을 향상시키는 데 있어 효과적임을 입증하였습니다. 이 방법은 단일 및 다변량 시계열 문제에서 이진 및 다중 클래스 분류에 적용할 수 있는 유연한 솔루션을 제공합니다.



### ZeFaV: Boosting Large Language Models for Zero-shot Fact Verification (https://arxiv.org/abs/2411.11247)
Comments:
          This pre-print has been published in PRICAI 2024: Trends in Artificial Intelligence. The published version is available at this https URL

- **What's New**: 이 논문에서는 ZeFaV라는 프레임워크를 제안하여 대형 언어 모델의 사실 검증 작업의 성능을 향상시킵니다. ZeFaV는 zero-shot 학습 기반으로 작동하며, 주장(claim)과 증거(evidence) 간의 관계를 추출하고 정보를 재구성하는 방식을 사용합니다. 이를 통해 기존의 사실 검증 시스템의 한계를 극복하고, 더 나아가 성능을 크게 향상시킬 수 있음을 나타냅니다.

- **Technical Details**: ZeFaV는 세 가지 주요 단계로 구성됩니다. 첫 번째, FewRel 데이터셋을 바탕으로 LLM을 미세 조정하여 관계 추출을 수행합니다. 두 번째, 주장의 증거 관계를 식별하고 중복 관계를 제거하는 폐쇄 정의를 사용하여 정보를 정리합니다. 마지막으로, InfoRE 기법을 활용하여 증거를 보다 유용한 형식으로 구조화하여 LLM의 준맥락 학습(in-context learning)을 강화합니다.

- **Performance Highlights**: ZeFaV는 HoVer 및 FEVEROUS-S 데이터셋에서 F1 점수 기준으로 기존의 다른 기술들과 비교했을 때 우수한 성능을 보였습니다. 특히 FEVEROUS-S 데이터셋에서는 ZeFaV가 86.54%의 F1 점수를 기록해 ProgramFC와 InfoRE보다 높은 성과를 냈습니다. 실험 결과, ZeFaV는 zero-shot 학습 기반으로도 효율적인 사실 검증 작업을 수행할 수 있음을 입증했습니다.



### MEMO-Bench: A Multiple Benchmark for Text-to-Image and Multimodal Large Language Models on Human Emotion Analysis (https://arxiv.org/abs/2411.11235)
- **What's New**: 이 연구에서는 MEMO-Bench라는 새로운 벤치마크를 소개하여 AI 모델의 감정 분석 능력을 평가하는 데 집중하고 있습니다. MEMO-Bench는 12개의 T2I 모델에서 생성된 7,145개의 감정 표현 초상화를 포함하고 있으며, 이는 기존 연구와 비교하여 감정 이해의 범위를 더욱 넓히고 있습니다. 이를 통해 T2I 모델과 MLLM의 감정 분석 성능을 포괄적으로 평가할 수 있는 기초 자료를 제공합니다.

- **Technical Details**: MEMO-Bench는 6개의 주요 감정(행복, 슬픔, 분노, 놀람, 걱정, 중립)을 기반으로 구축되었습니다. 각 감정에 대해 100개의 프롬프트가 설계되어 T2I 모델에 제공되어 감정 표현을 생성합니다. 연구에서는 감정 분석 방법론으로 coarse-grained 및 fine-grained 분석이라는 점진적 평가 접근 방식을 사용하여 MLLM의 감정 인식 능력을 평가합니다.

- **Performance Highlights**: 실험 결과, 기존 T2I 모델은 긍정적인 감정을 생성하는 데 있어 더 높은 효과를 보였지만, 부정적인 감정 생성에는 한계가 있음을 보여주었습니다. MLLM은 대체로 감정을 분류하는 데 효과적이지만, 미세한 감정 분석에서는 인간 수준의 정확도를 보장하지 못했습니다. 이러한 발견은 AI가 인간 감정을 완전히 이해하는 데 있어 현재의 제한 사항을 나타내며, 감정 인식 AI 시스템의 발전에 중요한 통찰을 제공합니다.



### MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs (https://arxiv.org/abs/2411.11217)
- **What's New**: 이번 논문에서는 리소스가 제한된 플랫폼에서 Mixture of Experts (MoE) 모델의 효율적인 배치를 위한 새로운 시스템인 MoE-Lightning을 제안합니다. MoE-Lightning은 고속 처리(batch inference) 시스템으로, 기존의 처리 방법보다 성능이 크게 향상됩니다. 특히, CGOPipe라는 새로운 CPU-GPU-I/O 파이프라인 스케줄링 기법을 도입하여 리소스 활용도를 높였습니다.

- **Technical Details**: MoE-Lightning은 GPU 메모리가 제한된 환경에서 MoE 모델을 효과적으로 실행하도록 설계되었습니다. 이 시스템은 CPU와 GPU 간의 I/O 이벤트와 연산을 겹치도록 스케줄링하여 연산이 차단되지 않도록 합니다. Hierarchical Roofline Model (HRM)을 도입해 다양한 작업 환경에서 시스템 성능을 세밀하게 조정할 수 있도록 모델을 정확히 묘사합니다.

- **Performance Highlights**: 실험 결과, MoE-Lightning은 최신 MoE 모델(Mixtral 8x7B 및 Mixtral 8x22B 등)에서 최대 10.3배의 처리량 개선을 보여주었습니다. 또한, Tensor-parallelism을 활성화 할 경우, 생성 처리량에서 초선형 규모 확장을 달성했습니다. MoE-Lightning은 다양한 하드웨어 환경에서도 우수한 성능을 발휘하며, 더 많은 리소스를 요구하지 않고도 생산성을 높일 수 있습니다.



### Making Sigmoid-MSE Great Again: Output Reset Challenges Softmax Cross-Entropy in Neural Network Classification (https://arxiv.org/abs/2411.11213)
- **What's New**: 이 연구는 Mean Squared Error (MSE)와 Softmax Cross-Entropy (SCE)라는 두 가지 손실 함수의 비교 분석을 제공합니다. 전통적으로 SCE는 클래스 확률로 네트워크 출력을 변환하는 데 사용되지만, 우리는 sigmoid activation과 함께 MSE를 사용하는 대안적 접근 방식을 탐구합니다. 이 연구는 Output Reset 알고리즘을 도입하여 일관성 없는 오류를 줄이고 분류기의 강건성을 향상시킵니다.

- **Technical Details**: 우리는 MSE와 sigmoid activation의 조합이 SCE와 유사한 정확도 및 수렴 속도를 달성하며, 특히 노이즈가 많은 데이터에 대한 성능이 우수하다는 것을 다양한 실험(MNIST, CIFAR-10, Fashion-MNIST 데이터셋을 사용)을 통해 증명합니다. 이 연구는 MSE가 회귀 작업과 전통적으로 연관되어 있음에도 불구하고 분류 문제에도 적합할 수 있음을 보여줍니다.

- **Performance Highlights**: MSE와 sigmoid activation의 조합은 노이즈가 몇몇 예시를 포함한 데이터에서 SCE에 비해 우수한 성능을 발휘하고, 전통적인 신경망 훈련 전략에 대한 새로운 통찰을 제공합니다. 이 연구는 MSE가 분류 작업에서도 효과적일 수 있음을 보여줍니다.



### Capturing Sparks of Abstraction for the ARC Challeng (https://arxiv.org/abs/2411.11206)
Comments:
          Submitted as a paper entry for the 2024 ARC Prize

- **What's New**: 이 연구는 LLM이 주어진 문제를 해결할 수 있는 코드 솔루션과 이에 대한 설명을 통해 LLM의 '이해'를 향상시키고, 이를 바탕으로 ARC(Abstraction and Reasoning Challenge) 문제를 해결하는 새로운 접근법을 제시합니다.

- **Technical Details**: LLM은 arc-dsl-llm이라는 LLM-가독성이 높은 DSL(Domain-Specific Language)로 작성된 코드 솔루션을 바탕으로 문제를 해결하는 데 도움을 받습니다. 이 DSL은 기존의 Hodel이 만든 arc-dsl에서 개선되어 LLM의 이해도를 높였습니다. 주요 목표는 LLM의 출력을 통해 생성되는 '고차 수준'의 추상 개념을 추출하는 것입니다.

- **Performance Highlights**: 연구 결과, LLM은 주어진 코드 솔루션을 통해 문제를 해결하는 과정에서 높은 수준의 전략적 사고를 보였으며, 이 샘플 데이터를 LLM의 Fine-Tuning에 사용하거나 RAG 시스템을 활용해 실시간 문제에 주입할 수 있는 가능성이 확인되었습니다.



### PickScan: Object discovery and reconstruction from handheld interactions (https://arxiv.org/abs/2411.11196)
Comments:
          7 pages, 8 figures, published in the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024)

- **What's New**: 본 연구는 사용자와 객체 간의 상호작용을 기반으로 하는 새로운 방법론을 통해 3D 씬의 조합적(representational) 재구성을 시도합니다. 이는 사용자가 RGB-D 카메라를 사용하여 객체를 이동시키고, 최종적으로 각 객체에 대한 3D 모델을 생성하는 능력을 제공합니다.

- **Technical Details**: 이 방법은 사용자-객체 상호작용을 감지하고, 조작된 객체의 마스크를 추출하는 새로운 접근 방식을 포함합니다. 이를 통해 객체 발견을 위한 클래스 비의존적(class-agnostic) 파이프라인을 제공하며, 사용자 움직임을 통해 객체를 완전히 스캔할 수 있습니다. RGB-D 및 관성 측정 장치(IMU) 데이터를 활용하여 작동합니다.

- **Performance Highlights**: 개발된 파이프라인은 100% 재현율에서 78.3%의 정밀도를 달성하였고, 평균 침식 거리(chamfer distance)는 0.90cm로 측정되었습니다. Co-Fusion과 비교하여 침식 거리가 73% 감소하였으며, 허위 탐지 감소는 99%에 달했습니다.



### Improving User Experience in Preference-Based Optimization of Reward Functions for Assistive Robots (https://arxiv.org/abs/2411.11182)
Comments:
          Accepted to ISRR

- **What's New**: 본 연구에서는 사용자의 선호를 효과적으로 학습하기 위해 Covariance Matrix Adaptation Evolution Strategies with Information Gain (CMA-ES-IG)라는 알고리즘을 설계하였습니다. 이 알고리즘은 사용자 경험을 우선시하여 직관적이고 사용하기 쉬운 방식을 제공합니다.

- **Technical Details**: CMA-ES-IG는 명시적 모델과 암묵적 모델을 결합하여 사용자의 선호를 학습합니다. 로봇의 동작 궤적을 생성하기 위해 동적 시스템의 출력을 정의하고, 이를 기초로 사용자와의 상호작용을 통해 지속적으로 궤적의 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, CMA-ES-IG 알고리즘은 기존 방법들보다 사용자의 선호를 더 잘 반영한 로봇 궤적을 생성하며, 물리적 및 사회적 로봇 작업에서 사용자 경험의 질을 크게 향상시켰습니다.



### Enhanced Anime Image Generation Using USE-CMHSA-GAN (https://arxiv.org/abs/2411.11179)
- **What's New**: 이 논문은 ACG (Anime, Comics, and Games) 문화의 인기에 힘입어 고품질 애니메이션 캐릭터 이미지를 생성하는 새로운 Generative Adversarial Network (GAN) 모델인 USE-CMHSA-GAN을 소개합니다. 이 모델은 전통적인 DCGAN 프레임워크를 기반으로 하며, 애니메이션 캐릭터 이미지를 위한 feature extraction (특징 추출) 능력을 강화하기 위해 USE와 CMHSA 모듈을 통합했습니다.

- **Technical Details**: USE-CMHSA-GAN은 기존의 GAN 구조에서 진화한 모델로, 향상된 이미지 품질을 위해 설계되었습니다. 특히, 다양한 GAN 모델들과 비교하여 FID (Fréchet Inception Distance) 및 IS (Inception Score) 점수에서 우수한 성능을 발휘하고 있습니다. 이는 애니메이션 캐릭터 이미지를 생성하는 과정에서의 특징 추출을 혁신적으로 개선합니다.

- **Performance Highlights**: 실험은 anime-face-dataset에서 수행되었으며, 그 결과 USE-CMHSA-GAN은 DCGAN, VAE-GAN, WGAN 등 다른 벤치마크 모델들을 초월하는 성과를 보였습니다. 이러한 결과들은 USE-CMHSA-GAN이 애니메이션 캐릭터 이미지 생성에 매우 효과적임을 나타내며, 생성 모델의 품질 개선에 대한 새로운 통찰을 제공합니다.



### LL\"aMmlein: Compact and Competitive German-Only Language Models from Scratch (https://arxiv.org/abs/2411.11171)
Comments:
          first draft; this https URL

- **What's New**: 본 논문에서는 독일어 전용 디코더 모델인 LLäMmlein 120M과 1B를 처음부터 끝까지 투명하게 생성하고, 이를 독일 NLP 연구 커뮤니티가 사용할 수 있도록 공개합니다.

- **Technical Details**: 모델 훈련에는 데이터 전처리, 사용자 지정 독일어 토크나이저(tokenizer) 생성, 모델 훈련 및 여러 기준에서 최종 모델 평가 등이 포함됩니다. LLäMmlein은 RedPajama V2 데이터셋에서 고품질의 독일어 데이터만을 포함하도록 전처리되었습니다. 32,000개의 토큰으로 구성된 새로운 독일어 토크나이저가 생성되었습니다. 두 개의 LLäMmlein 모델이 스크래치에서 훈련되었습니다.

- **Performance Highlights**: LLäMmlein 모델은 SuperGLEBer 기준에서 기존의 최신 모델들과 비교할 때 경쟁력 있는 성능을 보였으며, 일부 작업에서는 성능 향상이 초기 단계에서 정체되는 현상도 관찰되었습니다. 모델의 품질은 크기에 비례하여 증가했으며, 향후 모델 개발을 위한 자원 배분에 대한 귀중한 통찰을 제공합니다.



### RPN 2: On Interdependence Function Learning Towards Unifying and Advancing CNN, RNN, GNN, and Transformer (https://arxiv.org/abs/2411.11162)
Comments:
          105 pages, 37 figures, 6 tables, preprint version

- **What's New**: 본 논문은 Reconciled Polynomial Network (RPN)의 이전 연구를 바탕으로 하여, RPN 2라는 새로운 모델을 소개합니다. RPN 2는 데이터의 상호 의존성(data interdependence)을 명시적으로 모델링하기 위해 새로운 구성 요소로 상호 의존성 함수(interdependence functions)를 도입하여, 복잡한 데이터의 함수 학습 작업을 개선하는 것을 목표로 합니다.

- **Technical Details**: RPN 2는 데이터 인스턴스와 속성 간의 상호 의존성을 모델링하기 위해 그리드 기반 기하 구조 상호 의존성 함수를 활용합니다. 이 함수는 입력 데이터 배치를 기반으로 인터디펜던스 행렬(interdependence matrices)을 생성하여, 시간과 공간 면에서 계산 효율성을 최적화합니다. RPN 2는 CNN, RNN, GNN, 그리고 Transformers와 같은 널리 사용되는 백본 모델(backbone models)을 포괄하는 통합 표현을 제공합니다.

- **Performance Highlights**: RPN 2는 MNIST 및 CIFAR-10과 같은 벤치마크에서 이미지 패치 간의 지역적 상호 의존성을 효과적으로 캡처하여 우수한 성능을 기록했습니다. 또한, 언어 분류 및 시계열 예측과 같은 작업에서도 뛰어난 성능을 발휘합니다. RPN 2 모델을 기반으로 한 새로운 백본 모델들은 기존의 주도적인 백본 아키텍처와 비교할 때 효과적이고 우수한 성능을 보여주었습니다.



### MPLite: Multi-Aspect Pretraining for Mining Clinical Health Records (https://arxiv.org/abs/2411.11161)
- **What's New**: 본 논문은 MPLite라는 새로운 프레임워크를 제안하여, 전자의무기록(EHR)에서 단일 방문 데이터(single-visit records)를 활용하여 의료 개념의 표현과 미래 건강 결과 예측을 개선하는 방법을 소개합니다.

- **Technical Details**: MPLite는 lab test results를 통합하여 다각적 사전 훈련(Multi-aspect Pretraining)을 활용하는 경량(ライト웨이트) 신경망(neural network)입니다. 이 모듈은 의료 코드(medical codes)를 예측하며, 다양한 특징(feature)을 융합하여 강력한 예측(Prediction)을 보장합니다. 또한, MIMIC-III 및 MIMIC-IV 데이터셋을 사용하여 실험 평가를 수행하였습니다.

- **Performance Highlights**: MPLite는 진단 예측(diagnosis prediction) 및 심부전 예측(heart failure prediction) 작업에서 기존 모델들에 비해 개선된 성능을 보여주며, 더 높은 weighted-F1 점수와 recall을 달성했습니다. 이 연구는 예측 모델링을 발전시키기 위해 다양한 데이터의 통합 가능성을 시사합니다.



### TabDeco: A Comprehensive Contrastive Framework for Decoupled Representations in Tabular Data (https://arxiv.org/abs/2411.11148)
- **What's New**: 이번 연구에서 저자들은 표 형식의 데이터를 위한 새로운 대비 학습 프레임워크인 TabDeco를 제안합니다. 이를 통해 기존의 접근 방식에서는 간과되었던 긍정적 및 부정적 샘플 쌍 구성의 중요성을 강조하고, 더 나아가 다리(Layer)와 인스턴스 수준에서의 기능 해체를 통해 더욱 구조화된 임베딩 학습을 도모합니다.

- **Technical Details**: TabDeco는 어텐션 기반 인코딩 전략을 사용하여 테이블 데이터 간의 보다 정교한 기능 해체를 달성합니다. 이 프레임워크는 로컬 및 글로벌 특성과 데이터 배치에 대한 여러 관점에서 긍정적 및 부정적 샘플을 구성하여 기능 계층을 보다 세분화하여 분리합니다.

- **Performance Highlights**: TabDeco는 다양한 기준 작업에서 기존의 딥러닝 방법과 XG-Boost, CatBoost, LightGBM과 같은 최전선의 그래디언트 부스팅 알고리즘을 지속적으로 능가하며, 표 형식의 데이터 표현 학습에서 그 효과를 입증합니다.



### CLMIA: Membership Inference Attacks via Unsupervised Contrastive Learning (https://arxiv.org/abs/2411.11144)
- **What's New**: 이 논문에서는 CLMIA(Contrastive Learning Membership Inference Attack)라는 새로운 멤버십 추론 공격 방법을 제안합니다. 이 방법은 비지도식(constrastive) 학습을 활용하여 추가적인 멤버십 상태 정보 없이 공격 모델을 훈련할 수 있도록 합니다.

- **Technical Details**: CLMIA는 대조 학습(contrastive learning)을 기반으로 하며, 데이터 샘플의 멤버십 정보를 인퍼링하기 위해 공격 모델을 훈련합니다. 공격자는 대상 데이터셋을 활용하여 긍정 샘플(positive samples)과 부정 샘플(negative samples)을 생성하고, 소량의 라벨된 데이터로 모델을 미세 조정(fine-tuning) 합니다. 중요한 점은 공격 모델의 파라미터를 고정시켜 비지도 학습의 특성을 유지한다는 것입니다.

- **Performance Highlights**: 실험 결과, CLMIA는 다양한 데이터셋과 모델 구조에서 기존 공격 방법보다 더 나은 성능을 보였습니다. 특히 라벨된 정체성 정보가 적은 데이터에서 더욱 효과적인 공격 성능을 발휘함을 확인하였습니다.



### Label Sharing Incremental Learning Framework for Independent Multi-Label Segmentation Tasks (https://arxiv.org/abs/2411.11105)
- **What's New**: 이 논문에서는 각기 다른 라벨 집합을 가진 여러 데이터셋을 위해 세그멘테이션 모델을 구축할 때, 일반적인 접근 방식으로 각 데이터셋에 대해 하나의 모델을 학습하는 방법 대신, 공통 라벨 공간을 사용할 수 있는 새로운 '라벨 공유(label sharing)' 프레임워크를 제안합니다. 이 프레임워크는 다양한 라벨 집합을 단일 대규모 데이터셋으로 변환하여, 단일 모델 학습을 통해 모든 세그멘테이션 작업을 해결할 수 있도록 합니다.

- **Technical Details**: 라벨 공유 프레임워크는 서로 다른 작업의 라벨을 그룹화하고 각 그룹에 대해 공통 추상 라벨을 할당하는 방식입니다. 각 라벨은 하나의 공유 라벨에만 할당되며, 각 공유 라벨 그룹 내에는 각 작업에서 최대 한 개의 라벨만 포함될 수 있습니다. 이러한 방식으로 라벨 간의 공통 특성을 활용하여 세그멘테이션 모델 학습을 지원합니다.

- **Performance Highlights**: 우리의 방법은 다양한 의학적 이미지 세그멘테이션 데이터셋에서 실험적으로 검증되었으며, 제안하는 방법의 성능과 점진적인 학습 능력이 기존의 다른 방법들과 비교하여 효과적임을 보여줍니다. 본 연구는 각기 다른 라벨 수를 가진 다양한 세그멘테이션 작업에서 라벨 공유 프레임워크의 유용성과 다재다능성을 정성적 및 정량적으로 입증합니다.



### Different Horses for Different Courses: Comparing Bias Mitigation Algorithms in ML (https://arxiv.org/abs/2411.11101)
Comments:
          To appear at AFME@NeurIPS 2024

- **What's New**: 이 연구는 알고리즘의 학습 파이프라인이 공정성 점수에 미치는 영향을 강조하며, 기존 공정성 벤치마크의 한계를 지적하고 있습니다. 특히 하이퍼파라미터(z) 선택에 따른 차이가 특정 알고리즘을 일방적으로 유리하게 만들 수 있다는 점을 보여줍니다.

- **Technical Details**: 연구는 ML 시스템에서 불공정성을 줄이기 위해 7개의 편향 완화 알고리즘과 7개의 표 형식 데이터셋을 사용한 대규모 실험 분석을 수행했습니다. 여러 하이퍼파라미터 설정에서의 성능 차이를 보여주며, 공정성-유틸리티 균형을 포괄적으로 고려해야 한다고 주장합니다.

- **Performance Highlights**: 편향 완화 알고리즘 간 공정성 점수의 변동성이 크다는 것을 입증하였고, 이로 인해 알고리즘의 비교 분석 시 기존의 단순화된 접근 방식으로는 불공정성을 제대로 나타내지 못할 수 있음을 시사합니다.



### Mitigating Relative Over-Generalization in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2411.11099)
Comments:
          Published in Transactions on Machine Learning Research (11/2024)

- **What's New**: 본 논문에서는 MaxMax Q-Learning (MMQ)라는 새로운 알고리즘을 도입하여 분산된 다중 에이전트 강화 학습(MARL) 환경에서 발생하는 상대적 과일반화(Relative Over-generalization, RO) 문제를 해결하고자 합니다. MMQ는 에이전트들이 최적의 공동 행동을 바라볼 수 있도록 하여 비효율적인 개별 행동을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: MMQ 알고리즘은 두 개의 비모수(quintile) 모델을 사용하여 환경의 상태 전이를 캡처하고, 이상적인 상태 전이를 근사화하는 과정을 반복적으로 수행합니다. 이 과정에서 최대 Q 값을 지닌 상태를 선택하여 가치 함수(value function)를 업데이트하며, Bellman 업데이트에서 두 개의 최대 연산(maximum operators)을 사용하는 것이 특징입니다.

- **Performance Highlights**: 실험 결과, MMQ는 기존 기반선(baselines)보다 더 빠른 수렴(convergence)과 높은 샘플 효율(sample efficiency)을 보여주며 다양한 협력 작업(cooperative tasks)에서 성능이 입증되었습니다.



### SRA-MCTS: Self-driven Reasoning Aurmentation with Monte Carlo Tree Search for Enhanced Code Generation (https://arxiv.org/abs/2411.11053)
- **What's New**: 이번 연구에서는 SRA-MCTS라는 새로운 데이터 생성 방식을 소개하여 LLMs가 복잡한 문제를 해결하는 데 필요한 사고 경로를 자율적으로 생성할 수 있도록 지원합니다. 이 방법은 모델 자체를 통해 작동하며 추가적인 감독 없이도 성능을 효과적으로 향상시킵니다. SRA-MCTS는 자연 언어 추론 경로를 합성하고 실행 가능한 코드로 변환하는 과정을 포함하여, 분석의 정확성을 보장하고 복잡한 문제 해결의 성공률을 높이는 데 기여합니다.

- **Technical Details**: 연구에서는 SRA-MCTS 방법론이 4단계로 구성되어 있으며, 각각 선택(Selection), 확장(Expansion), 평가 및 반영(Evaluation & Reflection), 역전파(Backpropagation) 단계로 이루어집니다. 각 단계는 반복적으로 실행되며, LLM은 중간 단계에서 중요한 역할을 수행하여 자연어 해결책을 생성합니다. 코드는 주어진 문제와 자연어 해결책을 바탕으로 생성되고, 이를 통해 모델의 훈련 데이터 세트를 구성합니다.

- **Performance Highlights**: 실험 결과, SRA-MCTS로 생성된 데이터로 모델을 미세조정하면 기존 CoT 방법론보다 우수한 성능을 보여주었습니다. 특히, 작은 모델에서도 자가 개선(self-improvement) 기능을 통해 70B 모델의 증류 데이터로 훈련된 작은 모델보다 더 나은 성과를 달성하였습니다. 또한, 기존의 CoT 접근 방식의 성능 저하 문제를 해결하고, diverse metrics와 같은 다양한 평가 지표에서 개선된 성능을 보였습니다.



### Knowledge-enhanced Transformer for Multivariate Long Sequence Time-series Forecasting (https://arxiv.org/abs/2411.11046)
Comments:
          9 pages, 4 figures, 4 tables

- **What's New**: 이 논문은 멀티변량 긴 시퀀스 시계열 예측에서 정보가 풍부한 Knowledge Graph Embeddings (KGE)를 최신 Transformer 아키텍처와 통합하여 구조적 관계를 포착하도록 하는 새로운 접근법을 제안합니다.

- **Technical Details**: 저자는 패치 기반 방법론(PatchTST), 오토포머(Autoformer), 인포머(Informer), 바닐라 트랜스포머(Vanilla Transformer)와 같은 기존 아키텍처에 KGE를 통합합니다. KGE는 지식 그래프에서 개념적 관계를 추출하여 동적이고 학습 가능한 형태로 구축됩니다.

- **Performance Highlights**: 이 통합 접근법은 Weather 및 ETT 데이터셋을 기반으로 한 실험에서 벤치마크 결과를 상당히 향상시켰습니다. 특히, 이 접근법은 시계열 데이터의 복잡한 시간적 및 관계적 역학을 포착하여 멀티변량 LSTF의 정밀도를 개선합니다.



### Wafer Map Defect Classification Using Autoencoder-Based Data Augmentation and Convolutional Neural Network (https://arxiv.org/abs/2411.11029)
Comments:
          26 pages, 11 figures, including dataset preprocessing, proposed methods, and experimental results

- **What's New**: 이 논문은 반도체 제조에서 웨이퍼 결함 지도(Wafer Defect Maps, WDMs)의 정확한 분류 문제를 해결하기 위해 새로운 방법을 제안합니다. 이 방법은 자가 인코더(self-encoder) 기반 데이터 증강(data augmentation) 기술과 컨볼루션 신경망(Convolutional Neural Networks, CNN)을 결합하여 발생하는 노이즈를 개선합니다. 제안된 모델은 일반 및 희귀 결함 패턴의 정확한 분류를 가능하게 하며, 실험 결과 WM-811K 데이터셋에서 98.56%의 분류 정확도를 달성합니다.

- **Technical Details**: 제안된 방법은 자가 인코더를 이용해 잠재 공간(latent space)에 노이즈를 도입하여 데이터 다양성을 향상시키고 클래스 불균형을 완화합니다. 이를 통해 CNN은 확대된 데이터셋을 학습하게 되어 일반화 능력이 개선됩니다. 이 모델과 전통적인 기계 학습 방법들, 예를 들어 Random Forest, SVM, Logistic Regression과의 성능 비교도 이루어졌습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 Random Forest, SVM, Logistic Regression에 비해 각각 19%, 21%, 27%의 성능 향상을 보여줍니다. 이러한 결과는 제안된 접근 방식의 강력함과 효과성을 강조하며, 반도체 제조에서 웨이퍼 결함 탐지 및 분류를 위한 신뢰할 수 있는 솔루션을 제공합니다.



### BianCang: A Traditional Chinese Medicine Large Language Mod (https://arxiv.org/abs/2411.11027)
- **What's New**: 이 논문은 전통 중국 의학(TCM) 분야에 특화된 대형 언어 모델(BianCang)을 제안합니다. TCM의 진단 및 증후군 별화를 위한 기존 모델의 한계를 극복하고자 도메인 지식을 주입한 후 목표 지향적 자극을 통해 모델을 조정하는 두 단계의 훈련 과정을 채택하였습니다. 이는 TCM의 복잡한 지식을 이해하고 활용하는 데 필요한 새로운 접근법을 제시합니다.

- **Technical Details**: BianCang은 Qwen-2/2.5를 기반으로 하여 TCM의 특징에 맞춘 두 단계의 훈련 과정을 통해 개발되었습니다. 첫 번째 단계에서는 기존 모델에 TCM 및 의학 지식을 주입하는 지속적인 사전 훈련이 이루어지고, 두 번째 단계에서는 감독된 세부 조정을 통해 내부 지식을 활성화하고 정렬합니다. 이 과정은 TCM의 증후군을 구분하고 진단하는 능력을 향상시키는 데 중점을 둡니다.

- **Performance Highlights**: BianCang 모델은 11개의 테스트 세트와 4가지 작업을 통해 평가되었으며, 모든 능력 차원에서 기존의 TCM 및 의학 LLM을 지속적으로 초과하는 성능을 보였습니다. 이 모델은 TCM 특화 언어 모델로서 진정한 증후군 differentiation 및 진단 능력을 입증하였으며, 오픈소스 ChP-TCM 데이터셋 또한 개발하여 배포하였습니다. 이를 통해 향후 연구에 유용한 실험 비교 자료를 제공할 수 있게 됩니다.



### Time Step Generating: A Universal Synthesized Deepfake Image Detector (https://arxiv.org/abs/2411.11016)
Comments:
          Submitted to CVPR 2025, 9 pages, 7 figures

- **What's New**: 이번 논문은 Time Step Generating (TSG)이라는 새로운 방법을 제안합니다. 이는 기존의 diffusion 모델에서 생성된 이미지를 탐지하기 위한 것으로, 사전 훈련된 모델의 재구성 능력이나 특정 데이터셋에 의존하지 않습니다. TSG는 특징 추출을 위해 사전 훈련된 diffusion 모델의 네트워크를 활용하며, 생성 이미지와 실제 이미지 간의 미세한 차이를 포착하는 데 집중합니다. 이를 통해 이미지가 합성인지 실제인지 효과적으로 감지할 수 있습니다.

- **Technical Details**: Paper에서는 Score-Based Diffusion 모델과 Denoising Diffusion Probabilistic Models (DDPM)에 대한 배경 정보를 제공합니다. TSG는 noise ϵ의 예측을 통해 이미지를 분류하는 방법으로, 사전 훈련된 U-Net을 사용하여 이미지에서 특징을 직접 추출합니다. 이는 모든 형태의 이미지 데이터셋에 대해 적용 가능하며 다양한 생성기를 다룰 수 있습니다. TSG는 기존의 재구성 기반 방법들에 비해 간소화된 접근 방식을 제시합니다.

- **Performance Highlights**: 실험 결과, TSG는 GenImage 벤치마크에서 10배 빠른 속도와 함께 기존 방법들에 비해 상당한 정확도 향상을 보여주었습니다. 특히, TSG는 특정 데이터셋이나 객체 유형에 대한 재구성 능력을 고려할 필요 없이 광범위한 적용 가능성을 가지고 있습니다. 이러한 성과는 TSG의 접근 방식이 아키텍처와 데이터 내용의 다양성 덕분임을 나타냅니다.



### BackdoorMBTI: A Backdoor Learning Multimodal Benchmark Tool Kit for Backdoor Defense Evaluation (https://arxiv.org/abs/2411.11006)
- **What's New**: BackdoorMBTI는 멀티모달(multi-modal) 백도어(backdoor) 학습을 위한 최초의 툴킷이자 벤치마크(benchmark)입니다. 이 툴킷은 데이터 처리(data processing), 데이터 오염(data poisoning), 백도어 훈련(backdoor training), 평가(evaluation)를 포함하는 체계적인 파이프라인을 제공합니다. 다양한 데이터 유형 간의 체계적인 평가를 가능하게 하며, 실질적인 데이터 품질 문제 등도 포함하여 더 폭넓은 연구를 촉진할 것으로 예상됩니다.

- **Technical Details**: BackdoorMBTI는 11개 데이터셋과 17개의 공격(attack), 7개의 방어(defense) 방법을 지원하며, 이미지, 텍스트, 오디오 등 세 가지 중요한 모달리티를 포함합니다. 또한, 백도어 방어를 위해 적합한 방법들을 다루고, 데이터 품질과 잘못된 라벨을 고려한 평가 기준을 제공합니다. 이는 대립되는 상황에서도 공정한 평가를 보장하여 멀티모달 백도어 학습의 확장을 촉진합니다.

- **Performance Highlights**: 실험 결과, BackdoorMBTI는 노이즈(noise) 생성기를 통해 실제 데이터와 레이블의 노이즈를 효과적으로 모의할 수 있음을 보여주었습니다. 노이즈는 백도어 공격 성능에는 큰 영향을 미치지 않지만, 방어 성능을 향상시키는 중요한 역할을 합니다. 본 연구를 통해 멀티모달 맥락에서의 방어 성능이 향상되었으며, 이를 통해 백도어 방어 연구에 새로운 길을 제시합니다.



### Unveiling the Hidden: Online Vectorized HD Map Construction with Clip-Level Token Interaction and Propagation (https://arxiv.org/abs/2411.11002)
Comments:
          18 pages, 9 figures, NeurIPS 2024

- **What's New**: 본 논문에서는 도로 기하정보의 예측 및 구성에 관한 새로운 패러다임인 MapUnveiler를 도입하여, 인접 입력 프레임 간의 시간 정보를 효과적으로 활용할 수 있는 방법을 제안합니다. 기존의 HD 맵 구축 방법들이 다루지 못했던 가려진 맵 요소들을 명확하게 드러내며, 클립 간 정보 전파를 통해 장기적인 맵 정보 이용을 가능하게 합니다.

- **Technical Details**: MapUnveiler는 클립 입력이 주어졌을 때, 시간적 맵 정보가 포함된 컴팩트 클립 토큰을 생성하고 이를 통해 BEV(탑 뷰) 특징을 갱신하여 숨겨진 맵 요소를 드러내는 데 중점을 두고 있습니다. 또한, 클립 레벨 파이프라인을 활용하여 효율적인 온라인 추론을 제공하고, 여러 프레임의 입력을 빠르게 처리할 수 있습니다.

- **Performance Highlights**: MapUnveiler는 nuScenes와 Argoverse2 기준 데이터셋에서 최첨단 성능을 달성하였으며, 심각한 가림 조건에서도 +10.7% mAP 향상된 결과를 보였습니다. 실험 결과, 이 모델은 시간적 정보의 누적 영향을 적절히 활용하여 더 긴 인식 범위에서도 성능을 향상시킬 수 있음을 증명합니다.



### Modulating Reservoir Dynamics via Reinforcement Learning for Efficient Robot Skill Synthesis (https://arxiv.org/abs/2411.10991)
Comments:
          13 pages, 7 figures

- **What's New**: 이번 연구에서는 로봇의 동작을 컨텍스트 입력(context input)에 기반하여 배우는 새로운 Reservoir Computing (RC) 기반의 Learning from Demonstration (LfD) 프레임워크가 제안되었습니다. 이 모델은 초기 시연 데이터로부터 배운 동작을 생성할 뿐만 아니라, 온라인에서 레저버(Reservoir) 동작을 변형하여 추가 데이터 없이도 새로운 목표에 도달할 수 있도록 합니다. 또한, 이 접근법은 Reinforcement Learning (RL) 모듈을 통합하여 로봇의 상태에 따라 동적 컨텍스트를 생성하는 정책을 학습합니다.

- **Technical Details**: 제안된 Dynamic Adaptive Reservoir Computing (DARC) 모델은 Reservoir Computing (RC)의 효율성과 Reinforcement Learning (RL)의 적응성을 결합한 구조입니다. 전통적인 순환 신경망(RNN)과는 달리, 이 모델은 고정된 레저버를 사용하며, 랜덤하게 초기화된 레저버의 동적 특성을 통해 동작을 생성합니다. RL 모듈은 행동 공간의 차원이 낮기 때문에 학습 효율이 뛰어나며, 이를 통해 외부 목표 외에 로봇 동작을 확장할 수 있는 가능성을 열어줍니다.

- **Performance Highlights**: 실험 결과, DARC 모델은 2 자유도(DOF) 시뮬레이션 로봇에서 목표 지점에 도달하는 동작을 효과적으로 생성할 수 있음을 입증하였습니다. 초기 데이터 세트에 포함된 도달 시연을 통해 배운 후, RL 모듈을 통해 동적 컨텍스트를 생성하여 출발점과는 다른 목표 기능 달성에 성공하였습니다. 이러한 접근은 향후 로봇의 행동 패턴을 더욱 다양화하고 새로운 데이터 수집 없이 유연성을 제공하는 데 기여할 것입니다.



### VidComposition: Can MLLMs Analyze Compositions in Compiled Videos? (https://arxiv.org/abs/2411.10979)
- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)의 비디오 구성 이해 능력을 평가하기 위한 새로운 벤치마크인 VidComposition을 소개합니다. 기존의 평가 기준은 비디오의 일반적인 이해에 중점을 두었지만, 비디오 구성에서의 MLLMs의 성능은 충분히 평가되지 않았습니다. VidComposition은 982개의 비디오와 1706개의 선택형 질문으로 구성되어 스토리텔링을 통한 깊이 있는 비디오 해석을 가능하게 합니다.

- **Technical Details**: VidComposition은 Cinematography Analysis, Character Understanding, Narrative Understanding, Scene Perception, Making Analysis의 5개 주요 카테고리 및 15개 하위 과제를 포함합니다. 수집된 비디오는 주로 코멘터리 비디오에서 출처를 두고, 각 비디오 조각에 대해 여러 인간 주석자가 고유한 질문과 답안을 작성합니다. 데이터 수집과 필터링 과정에서 심리적 고통을 유발할 수 있는 부적절한 내용을 제거하여 데이터셋의 신뢰성을 높였습니다.

- **Performance Highlights**: MLLMs를 VidComposition에서 평가한 결과, 인간의 비디오 구성 이해 능력에 비해 MLLMs가 상당히 뒤처짐을 보였습니다. 특히, 상위 모델들도 기본적인 인식 과업에서는 꽤 좋은 성능을 보였지만, 복잡한 비디오 구성에서는 부족한 성과를 보였습니다. 이 성능 차이는 MLLMs가 다층적인 비디오 구조를 포착하는 데 한계가 있음을 강조합니다.



### SageAttention2 Technical Report: Accurate 4 Bit Attention for Plug-and-play Inference Acceleration (https://arxiv.org/abs/2411.10958)
- **What's New**: SageAttention2는 4비트 행렬 곱셈을 사용하여 계산 효율성을 크게 개선하고, 정확성을 유지하며 3배 이상 속도를 향상시켰습니다.

- **Technical Details**: SageAttention2는 원래의 SageAttention에서 INT8에서 INT4로의 양자화와 FP16에서 FP8로의 개선을 통해 Q, K를 INT4로, P, V를 FP8로 양자화합니다. 특정 어려운 레이어에 대해 혼합 정밀도(adaptive mix precision)를 적용하여 정확성을 유지합니다.

- **Performance Highlights**: RTX4090에서 SageAttention2는 485 TOPS의 최고 성능을 달성하며, FlashAttention2에 비해 약 3.1배, xformers에 비해 약 5.4배 더 빠른 성능을 보였습니다.



### IMPaCT GNN: Imposing invariance with Message Passing in Chronological split Temporal Graphs (https://arxiv.org/abs/2411.10957)
Comments:
          11 pages (without appendix), 35 pages (with appendix), 14 figures

- **What's New**: 이 연구는 시간적 분할(chronological splits)에서 발생하는 그래프 데이터의 도메인 적응(domain adaptation) 문제를 해결하고자 합니다. 특히, 역사적 데이터를 기반으로 최근 데이터의 노드를 분류하는 반지도 노드 분류(Semi-Supervised Node Classification, SSNC) 문제에 중점을 두고 있습니다.

- **Technical Details**: 우리는 시간적 그래프 구조에서 유도된 현실적인 가정을 기반으로 불변 속성을 부여하는 의료 전상 대화(IMPaCT) 방법을 제안합니다. IMPaCT는 전통적인 도메인 적응 접근법과는 달리 시간적 분할의 특성을 명시적으로 고려하여 일반화 오류의 상한을 도출하는 수학적 분석을 포함합니다.

- **Performance Highlights**: 실험적으로, IMPaCT는 ogbn-mag 그래프 데이터셋에서 현재 최첨단(SOTA) 방법보다 3.8%의 성능 향상을 달성하였습니다. 또한, 우리는 다양한 조건에서 시간적 그래프를 복제하는 Temporal Stochastic Block Model (TSBM)을 소개하여, IMPaCT의 일반 공간 GNNs에 대한 강건성과 적용 가능성을 체계적으로 입증하였습니다.



### Learn from Downstream and Be Yourself in Multimodal Large Language Model Fine-Tuning (https://arxiv.org/abs/2411.10928)
- **What's New**: 본 논문에서는 Multimodal Large Language Model (MLLM)의 특수화 및 일반화 능력을 모두 유지하기 위한 새로운 접근법인 SPIDER를 제안합니다. 이 방법은 사전 훈련된 가중치와 미세 조정된 기울기를 평가하여 각 파라미터의 중요도를 측정하고, 중요도가 높은 파라미터만 선택적으로 업데이트합니다. MLLM의 일반화 능력 저하를 방지하기 위해 파라미터 중요성 차이를 고려하는 것이 핵심입니다.

- **Technical Details**: SPIDER는 'Importance Discrepancy Measurement (IDM)'를 도입하여 파라미터의 일반화 및 특수화 지식에 대한 중요도를 정량화합니다. 또한, Importance Selection Mask (ISM)를 사용하여 목표 배포에 적합한 파라미터를 선택적으로 최적화합니다. 이 방식은 이미지 캡셔닝 및 시각적 질문-답변 작업에서 다양한 MLLM 아키텍처를 통해 실험적으로 검증되었습니다.

- **Performance Highlights**: 제안하는 SPIDER 방법은 Flickr30k, COCO-Caption, ScienceQA, IconQA와 같은 여러 다운스트림 데이터셋에서 MLLM의 미세 조정 성능을 향상시키는 데 기여했습니다. 논문에서는 두 가지 주요 작업인 이미지 캡셔닝과 시각적 질문-답변에 대한 종합적인 분석을 제공하며, 실험 결과가 SPIDER의 효과성을 입증하고 있습니다. 이러한 결과를 통해 MLLM의 특징적 성능 저하를 완화하고 일반화 능력을 유지할 수 있는 가능성을 보여주고 있습니다.



### Hyperspectral Imaging-Based Grain Quality Assessment With Limited Labelled Data (https://arxiv.org/abs/2411.10924)
Comments:
          10 pages

- **What's New**: 최근 hyperspectral imaging (HSI) 기반의 곡물 품질 평가가 연구의 주목을 받게 되었습니다. 그러나 다른 이미징 방식과 달리 HSI 데이터는 심층 합성곱 신경망 (DCNN) 기반 분류기를 효과적으로 학습시키기에 충분한 레이블 샘플이 부족합니다. 본 논문에서는 HSI와 few-shot learning (FSL) 기법을 결합한 새로운 접근 방식을 제안하여 곡물 품질 평가를 수행합니다.

- **Technical Details**: 기존 곡물 품질 평가 방법은 화학적 및 생물학적 분석을 포함하며, 이 방법들은 침습적이고, 파괴적이며, 시간 소모적이고 비용이 많이 듭니다. HSI는 비침습적이고 실시간 대안을 제공하지만, DCNN을 적용하기 위해서는 많은 레이블 데이터가 필요합니다. FSL을 사용함으로써 적은 레이블 데이터로도 모델이 잘 작동할 수 있도록 하여, 실제 적용에서 빠른 배포가 요구되는 상황에서 실용적인 해결책을 제공합니다.

- **Performance Highlights**: 실험 결과, 매우 제한된 레이블 데이터로 학습했음에도 불구하고, FSL 분류기의 정확도가 기존에 대규모 레이블 데이터베이스로 학습한 완전한 분류기와 유사한 수준인 것으로 나타났습니다. 이 연구는 HSI 기반 품질 평가에 대한 새로운 가능성을 제시하며, 곡물 품질 평가의 신속성과 효율성을 어떻게 향상시킬 수 있는지 보여줍니다.



### Multi-Modal Self-Supervised Learning for Surgical Feedback Effectiveness Assessmen (https://arxiv.org/abs/2411.10919)
Comments:
          Accepted as a spotlight proceedings paper at Machine Learning for Health 2024

- **What's New**: 본 연구에서는 외과 훈련에서 트레이너의 실시간 피드백이 트레이니의 행동 변화에 미치는 효과를 예측하기 위해, 텍스트와 비디오 정보를 통합하는 다중 모달 접근 방식을 제안합니다. 이전의 수작업 분석 방식에서 벗어나 자동화된 시스템을 통해 피드백 효과성을 객관적이고 확장 가능하게 평가할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 우리는 수술 중 피드백의 오디오 기록을 전사하고, Sentence Transformers (SBERT)을 사용하여 전사된 내용을 임베딩으로 변환합니다. 시각적 측면에서는 비디오 마스크 오토인코더(VideoMAE)를 사용해 피드백이 제공될 때의 수술 비디오에서 정보를 추출합니다. Self-supervised learning (SSL) 방식의 fine-tuning을 통해 모델 성능을 향상시킵니다.

- **Performance Highlights**: 본 연구의 결과, 트레이니 행동 변화 예측에서 ROC 곡선 아래 면적(AUC)이 0.70 ± 0.02를 달성했습니다. 텍스트 피드백과 수술 비디오를 결합할 경우 예측 정확도가 최대 6.6% 향상되었습니다. 이는 수술 피드백의 효과성을 자동으로 예측할 수 있는 시스템의 가능성을 보여줍니다.



### LLM-assisted Physical Invariant Extraction for Cyber-Physical Systems Anomaly Detection (https://arxiv.org/abs/2411.10918)
- **What's New**: 이 논문은 Cyber-Physical Systems(CPS)에서 이상 탐지를 위해 물리적 불변량(Physical Invariants)을 추출하는 새로운 접근 방식을 제안합니다. 특히 기존의 수작업에 의존하던 방법에서 벗어나 최신 생성 AI 모델을 활용하여 자동화된 과정으로 확장성과 비용 절감을 꾀합니다. 이 연구는 이전에 다루어지지 않았던 CPS 문서에서의 의미론적 정보 추출의 가능성을 보여줍니다.

- **Technical Details**: 저자들은 Retrieval-Augmented Generation(RAG) 워크플로우를 설계하고 교차 사고(Chain-of-Thought) 프로ンプ트를 활용하여 물리적 불변량을 가정적으로 추출합니다. 이 방법은 데이터셋에서 불변량을 검증하는 간단하면서도 효과적인 알고리즘을 통해 높은 정확도를 달성하고, 컨셉 이동(concept drift)과 같은 문제를 해결하여 신뢰성을 높입니다.

- **Performance Highlights**: 제안된 접근 방식은 실제 CPS 보안 데이터셋을 사용하여 평가되었으며, 0.923의 높은 정밀도로 24개의 실제 긍정(true positives)을 정확히 탐지하고 false positives는 2에 불과했습니다. 이 결과는 다양한 공격 시나리오에서의 유연성을 입증하며, CPS 환경에서의 자동화된 신뢰할 수 있는 이상 탐지의 가능성을 보여줍니다.



### Evolution of IVR building techniques: from code writing to AI-powered automation (https://arxiv.org/abs/2411.10895)
Comments:
          6 pages, 3 figures

- **What's New**: 이 논문에서는 Interactive Voice Response (IVR) 시스템의 발전을 탐구하며, 기존의 코드 기반 개발에서 위젯 사용 및 인공지능(AI) 활용으로의 전환을 강조합니다. AI의 도입은 IVR 흐름의 자동 생성 뿐만 아니라 고객 경험을 향상시키는데 기여하고 있습니다. 이러한 변화는 IVR 시스템의 미래를 재형성하고 있으며, 사용자 친화적인 접근 방식이 혁신을 주도하고 있습니다.

- **Technical Details**: IVR 시스템의 초기 개발 방식은 복잡한 코드 작성을 필요로 했으며, 모든 전화 흐름을 정의하기 위해 프로그래밍 언어와 통신 프로토콜에 대한 깊은 이해가 요구되었습니다. 그러나 최근에는 위젯 기반 플랫폼이 등장하여 사용자들이 그래픽 인터페이스를 통해 직접 IVR 시스템을 디자인할 수 있도록 하고 있습니다. AI 기술, 특히 자연어 처리(NLP)와 머신 러닝 알고리즘이 통합되면서 IVR 개발과 운영 방식이 혁신적으로 변화하고 있습니다.

- **Performance Highlights**: AI를 활용한 IVR 시스템은 고객의 요구에 더 적절히 반응할 수 있는 최적화된 호출 흐름을 자동으로 생성합니다. NLP 기술은 고객의 자연어 입력을 이해하고 처리할 수 있게 해 주며, 고객 경험을 더욱 개인화된 방향으로 발전시킵니다. 이러한 발전은 고객 상호작용을 보다 매력적이고 만족스럽게 만들고, IVR 시스템의 효율성을 크게 향상시킵니다.



### MpoxVLM: A Vision-Language Model for Diagnosing Skin Lesions from Mpox Virus Infection (https://arxiv.org/abs/2411.10888)
Comments:
          Accepted by ML4H 2024

- **What's New**: 본 논문에서는 MpoxVLM이라는 비전-언어 모델(Vision-Language Model)을 제안합니다. 이는 Mpox(원숭이 두창) 감염을 진단하기 위해 피부 병변 이미지와 환자의 임상 정보를 분석하는 데 특화되어 있습니다. MpoxVLM은 CLIP 비주얼 인코더, 강화된 비전 트랜스포머(ViT) 분류기 및 LLaMA-2-7B 모델을 통합하여 이전 모델들보다 더 높은 진단 정확도를 달성합니다.

- **Technical Details**: MpoxVLM은 새롭게 수집된 mpox 피부 병변 데이터셋에서 시각적 설명에 따라 질문-답변을 학습하여 훈련되었습니다. 논문에서는 2,914개의 샘플로 구성된 새로운 멀티모달 데이터셋을 통해 다양한 피부 타입, 성별, 나이, 병변 부위에 대한 모델의 효과성을 평가하고 있습니다. 이 모델은 고감도의 자동 검출 및 조기 진단의 가능성을 제공합니다.

- **Performance Highlights**: 제안하는 MpoxVLM은 mpox 감지에서 90.38%의 정확도를 달성했습니다. 이는 기존의 최신 기술들과 비교할 때 최상의 성능을 보이며, 다양한 인구집단에 대한 정확한 진단을 위한 가능성을 제시합니다. 이와 같은 딥러닝 기반의 컴퓨터 지원 진단 방법은 향후 mpox 통제에 크게 기여할 것으로 기대됩니다.



### MetricGold: Leveraging Text-To-Image Latent Diffusion Models for Metric Depth Estimation (https://arxiv.org/abs/2411.10886)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2312.02145 by other authors

- **What's New**: 이 논문에서는 MetricGold라는 새로운 방법을 제안하여 단일 이미지에서 메트릭 깊이(metric depth)를 추정하는 문제를 해결하고자 합니다. 이는 generative diffusion model의 강력한 prior를 활용하여 메트릭 깊이 예측 성능을 향상시키는 접근입니다. MetricGold는 안정적인 생성 모델인 Stable Diffusion을 기반으로 하며, denoising U-Net만 조정하여 깊이 추정에 적합하도록 하였습니다. 이를 통해 실제 깊이 맵을 전혀 보지 않고도 실험에서 최첨단 성능을 달성했습니다.

- **Technical Details**: MetricGold는 RGB 이미지를 깊이 맵으로 변환하는 generative 작업으로 메트릭 깊이 추정을 재구성합니다. 이 과정에서 모델은 조건부 분포 D(d|x)를 모델링하며, 이는 입력 이미지 x에 대한 깊이 d를 예측합니다. 기존의 DMD와는 달리, MetricGold는 latent space에서 denoising을 수행하여 더욱 계산 효율적이며 빠릅니다. 모델은 Hypersim과 Virtual KITTI와 같은 포토 리얼리스틱 합성 데이터로 학습되었습니다.

- **Performance Highlights**: MetricGold는 한 대의 RTX 3090 그래픽 카드에서 2일 이내에 효율적으로 학습할 수 있으며, 다양한 데이터셋에서 강력한 일반화를 보여줍니다. 실험 결과, 기존 접근 방법보다 선명하고 높은 품질의 메트릭 깊이 추정 결과를 생성해냅니다. 또한, 이 모델은 실세계 데이터에서 뚜렷한 일반화 능력을 갖추고 있으며, sensor 기반 데이터의 편향을 피하여 보다 정확한 예측을 가능하게 합니다.



### BanglaDialecto: An End-to-End AI-Powered Regional Speech Standardization (https://arxiv.org/abs/2411.10879)
Comments:
          Accepted in 2024 IEEE International Conference on Big Data (IEEE BigData)

- **What's New**: 이번 연구는 방글라데시 사투리를 인식하고 다양한 벵골어 억양을 표준화된 공식 벵골어로 변환하는데 중점을 둡니다. 특히, 방글라어의 55개 서로 다른 사투리를 다루는 점이 중요하며, 표준화가 교육적 일관성 및 경제적 기회를 보장하는 데 필요합니다.

- **Technical Details**: 이 연구는 Noakhali 지역의 방글라 사투리 음성을 표준 방글라 음성으로 변환하는 종단 간 파이프라인을 제시합니다. 이 과정에서 대규모 방글라 사투리 데이터셋을 구성하여 ASR(Automated Speech Recognition) 및 LLM(멀티언어 대형 언어 모델) 모델을 미세조정(fine-tuning)하여 사투리 음성을 텍스트로 변환하고 이를 표준 방글라 텍스트로 번역합니다.

- **Performance Highlights**: Whisper ASR 모델 미세 조정을 통해 CER(Character Error Rate) 0.8%, WER(Word Error Rate) 1.5%를 달성하였으며, BanglaT5 모델은 사투리 텍스트를 표준 텍스트로 번역하는 데 41.6%의 BLEU 점수를 기록했습니다.



### Empowering Meta-Analysis: Leveraging Large Language Models for Scientific Synthesis (https://arxiv.org/abs/2411.10878)
Comments:
          Accepted in 2024 IEEE International Conference on Big Data (IEEE BigData)

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)을 활용하여 과학 문서의 메타-분석(메타-분석) 자동화를 탐구합니다. 전통적으로 메타-분석은 여러 연구 결과를 종합하여 심층적인 이해를 제공하지만, 수작업으로 진행할 경우 시간 소모적이고 인적 오류에 취약합니다. 이에 본 연구는 LLM을 대규모 과학 데이터셋으로 미세 조정하여 데이터를 효과적으로 처리하고 구조적 데이터 추출의 도전을 해결하는 새로운 접근법을 제시합니다.

- **Technical Details**: 이 자동화된 메타-분석 프로세스는 Retrieval Augmented Generation (RAG)을 통합하여 최적화됩니다. LLM은 새로운 손실 메트릭(Inverse Cosine Distance, ICD)을 통해 훈련되어 고차원 문맥 데이터에 적합한 데이터 추출 패턴을 학습합니다. 이 방법은 메타-분석 생성을 위한 LLM의 성능을 향상시키며, 구조적 초록 생성에서의 품질과 효율성을 보장합니다.

- **Performance Highlights**: 사람이 평가한 결과, 미세 조정된 LLM 모델이 비 미세 조정 모델보다 우수한 성능을 보여, 87.6%의 관련 메타-분석 초록을 생성했습니다. 평가에 따르면 맥락의 적합성이 4.56%에서 1.9%로 감소하여,본 연구는 메타-분석 자동화의 효율성과 신뢰도를 향상시키는 데 기여한 것으로 나타났습니다.



### Developer Perspectives on Licensing and Copyright Issues Arising from Generative AI for Coding (https://arxiv.org/abs/2411.10877)
- **What's New**: 최근 Generative AI (GenAI) 도구는 소프트웨어 개발 관행을 크게 변화시키고 있습니다. 이런 도구들은 코드 작성 등 여러 작업에 유용하지만, 특히 저작권법과 관련된 법적 질문과 잠재적 위험을 야기합니다. 본 논문은 GenAI 도구를 사용하는 574명의 GitHub 개발자를 대상으로 시행된 연구 결과를 제시하며, 개발자들의 법적 우려와 저작권성, 생성된 코드의 소유권에 대한 인식을 조사하였습니다.

- **Technical Details**: 이 연구는 소프트웨어 공학 및 법률 연구자들로 구성된 팀에 의해 전 세계 소프트웨어 개발자를 대상으로 진행되었습니다. 온라인 설문조사와 후속 인터뷰를 통해 개발자들의 법적 문제에 대한 의견, 저작권으로 보호될 수 있는 것에 대한 인식, 생성된 코드의 소유권 등을 탐구하였습니다. 분석 결과, 저작권 문제에 대한 개발자들의 의견은 다양하며, 많은 개발자들이 이러한 법적 질문의 복잡성에 대해 인식하고 있음을 확인했습니다.

- **Performance Highlights**: 연구는 574명의 개발자를 대상으로 GenAI의 라이선스 및 저작권 측면에 대한 설문을 실시하였으며, GenAI에 대한 현재의 인식과 변화를 반영한 개발자들의 시각을 제공합니다. 저작권 관련 문제의 복잡성과 잠재적 오류에 대한 경각심을 불러일으키며, 향후 규제 결정에 대한 통찰과 권고를 제시합니다. 연구 결과는 통계 분석 뿐 아니라 질적 접근을 통해 이루어졌으며, 연구 결과와 자료들은 온라인 복제 패키지로 제공됩니다.



### ViBe: A Text-to-Video Benchmark for Evaluating Hallucination in Large Multimodal Models (https://arxiv.org/abs/2411.10867)
- **What's New**: 이 논문은 ViBe라는 새로운 대규모 Text-to-Video Benchmark를 소개하여 T2V 모델의 환각 콘텐츠를 평가합니다. 환각은 비디오 생성 시 생성된 내용이 입력 텍스트와 일치하지 않거나 왜곡되는 경우를 가리킵니다. 이 연구는 3,782개의 환각 비디오로 구성된 데이터세트를 수집하여 '소실 주제', '수치 변동성', '시간 왜곡', '생략 오류', '물리적 불일치'와 같은 다섯 가지 주요 환각 유형을 식별했습니다.

- **Technical Details**: ViBe 데이터세트는 MS-COCO 데이터세트에서 700개의 랜덤 캡션을 선택해 생성한 것입니다. 연구에 사용된 모델에는 MS1.7B, MagicTime, AnimateDiff-MotionAdapter, Zeroscope V2 XL 등 10개의 오픈소스 T2V 모델이 포함되었습니다. 분석을 통해 우리는 다양한 환각 유형의 존재 여부와 빈도를 살펴보면서 각 비디오를 조사했습니다.

- **Performance Highlights**: 최초의 대규모 환각 비디오 데이터세트를 제공함으로써 T2V 모델의 신뢰성 및 환각 탐지 개선에 기여하고자 합니다. 실험 결과, TimeSFormer + CNN 조합이 가장 좋은 성능을 보였으며, 0.345의 정확도와 0.342의 F1 점수를 기록했습니다. 이 벤치마크는 T2V 모델의 정확성과 신뢰성 향상을 위한 기초를 제공합니다.



### See-Saw Generative Mechanism for Scalable Recursive Code Generation with Generative AI (https://arxiv.org/abs/2411.10861)
Comments:
          18 pages, 4 figures

- **What's New**: 본 논문은 See-Saw 생성 메커니즘을 소개하며, 동적이고 재귀적인 코드 생성을 위한 새로운 방법론을 제안합니다. 이 방법론은 주 코드 업데이트와 의존성 생성을 번갈아가며 실행하여 연계성과 기능성을 보장합니다.

- **Technical Details**: See-Saw 메커니즘은 주 코드와 해당 의존성 파일 간의 상호작용을 관리하며, 프로젝트 트리 구조를 통해 코드의 일관성을 유지합니다. 이 프레임워크는 주 코드 M과 의존성 D 간의 관계를 정의하고, 코드베이스를 재귀적으로 생성 및 수정하는 두 가지 핵심 기능(주 코드와 의존성 생성)을 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 종속성을 효과적으로 관리하면서 코드 생성의 일관성을 유지하고 계산 비용을 최소화하여 효율적이고 확장 가능한 프로젝트 생성을 가능하게 합니다.



### A Novel Adaptive Hybrid Focal-Entropy Loss for Enhancing Diabetic Retinopathy Detection Using Convolutional Neural Networks (https://arxiv.org/abs/2411.10843)
Comments:
          9 pages,7 figures

- **What's New**: 이 논문에서는 당뇨병성 망막병증(diabetic retinopathy)의 진단을 위한 새로운 AI 기반 도구를 제안합니다. 특히, 기존의 다중 클래스 분류에서 사용되는 손실 함수(loss function)인 Categorical Cross-Entropy (CCE)의 한계를 극복하기 위해 Adaptive Hybrid Focal-Entropy Loss (AHFE)를 도입했습니다.

- **Technical Details**: AHFE는 집중 손실(focal loss)과 엔트로피 손실(entropy loss)을 결합하여 적응적 가중치(adaptive weighting)를 적용합니다. 이를 통해 소수 클래스(minority classes)에 집중하고 어려운 샘플들을 강조하는 방식으로 작동합니다. 이러한 접근 방식은 클래스 불균형(class imbalance) 문제를 해결하는 데 중요한 역할을 합니다.

- **Performance Highlights**: AHFE를 적용한 당뇨병성 망막병증 탐지(state-of-the-art models) 결과는 크게 향상된 성능을 보여줍니다. ResNet50은 99.79%, DenseNet121은 98.86%, Xception은 98.92%, MobileNetV2는 97.84%, InceptionV3는 93.62%의 정확도를 기록하여 복잡하고 불균형한 의료 데이터셋에 대한 AI 기반 진단의 향상을 보여줍니다.



### CODECLEANER: Elevating Standards with A Robust Data Contamination Mitigation Toolk (https://arxiv.org/abs/2411.10842)
- **What's New**: 이번 연구에서는 코드 리팩토링(code refactoring) 기법을 활용하여 데이터 오염(data contamination)을 완화하는 방안을 제시했습니다. 코드 리팩토링은 원본 코드의 의미를 변경하지 않으면서 구조와 변수명 변경을 포함하며, 이는 CLM(코드 언어 모델) 기반 기술이 직면한 신뢰성 문제를 해결하는 중요한 대안으로 등장했습니다. 연구팀은 CODECLEANER라는 오픈 소스 도구를 개발하여 파이썬과 자바에서 사용할 수 있는 11개의 리팩토링 연산자를 포함시켰습니다.

- **Technical Details**: DATASET과 코드를 대상으로 다양한 리팩토링 연산자를 적용하여 데이터 오염의 심각도를 평가했습니다. 리팩토링 연산자를 적용한 후, 중복 비율(overlap ratio)이 평균 65% 감소했음을 발견했습니다. 연구에서는 노드 단위(method-level), 클래스 단위(class-level) 및 교차 클래스 단위(cross-class level)의 다양한 리팩토링 기법이 포함되어 있으며, 이들 연산자는 코드의 구문(syntax), 의미(semantics) 및 스타일을 효과적으로 변경합니다.

- **Performance Highlights**: 연구 결과, 메소드 단위 리팩토링 기법은 평균적으로 35.89%의 중복을 줄일 수 있었으며, 의미 기반 연산자가 구문 기반 연산자보다 훨씬 더 효과적인 것으로 나타났습니다. 또한 11개의 리팩토링 연산자를 모두 적용할 경우, 데이터 오염 문제 해결에 있어 65%의 중복 비율 감소가 가능함을 증명했습니다. 이러한 결과는 CODECLEANER가 실제 산업에서 데이터 오염을 줄일 수 있는 유용한 도구가 될 수 있음을 보여줍니다.



### Adaptive Learning of Design Strategies over Non-Hierarchical Multi-Fidelity Models via Policy Alignmen (https://arxiv.org/abs/2411.10841)
Comments:
          48 pages, 20 figures

- **What's New**: 이번 논문에서는 다양한 정확도와 계산 비용을 가진 분석 모델을 활용하여 엔지니어링 디자인의 효율성을 크게 향상시키는 다중 신뢰도(Multi-fidelity) 강화 학습 프레임워크를 제안합니다. 특히, 기존의 계층적 모델 구조에 의존하지 않고 비계층적이며 이종의 저신뢰도 모델을 이용하여 고신뢰도 정책을 학습할 수 있는 ALPHA 프레임워크를 소개합니다.

- **Technical Details**: ALPHA는 'Heterogeneous Analyses'를 고려하여 고신뢰도 정책을 효과적으로 학습하는 방법으로, 낮은 신뢰도의 정책 및 경험 데이터를 동적으로 활용합니다. 이 연구는 두 개의 저신뢰도 모델과 하나의 고신뢰도 모델을 사용하여 최적화 문제를 해결하고, 경험 데이터의 정렬을 기반으로 효율적인 학습이 이루어집니다.

- **Performance Highlights**: ALPHA는 설계 공간 및 시간에 걸쳐 모델을 동적으로 활용할 수 있는 능력을 보여주며, 계층적 프레임워크에서 요구되는 모델 스케줄링 없이도 더 직접적인 경로로 높은 성능 해결책을 찾습니다. 또한, ALPHA는 계층적 에이전트에 비해 우수한 수렴 행동을 나타냅니다.



### One-Layer Transformer Provably Learns One-Nearest Neighbor In Contex (https://arxiv.org/abs/2411.10830)
- **What's New**: 이번 논문은 단일 계층(transformer layer) 트랜스포머가 비모수(nonparametric) 추정 기법인 1-최근접 이웃(1-NN) 예측 규칙을 학습할 수 있는 능력을 연구합니다. 특히, 우리는 softmax attention 계층을 통해 비공식적인 손실 함수에서도 성공적으로 학습할 수 있음을 보입니다.

- **Technical Details**: 이 논문에서는 한계층 트랜스포머가 주어진 프롬프트와 훈련 데이터를 통해 1-NN 예측 규칙을 학습하는 과정을 다룹니다. 이를 통해 트레이닝 과정 중 모델의 최적화 경과를 수학적으로 분석하고, 손실 함수가 비볼록(nonconvex)할지라도 최적화가 가능함을 증명합니다.

- **Performance Highlights**: 단일 계층 트랜스포머가 1-NN 분류기로 작동할 수 있도록 하는 행동 보장을 제시하며, 분포 이동에 대한 적응성을 예시합니다. 연구 결과는 트랜스포머가 비모수 기계 학습 알고리즘을 구현할 수 있다는 것을 구체적으로 보여주며, 이전 문헌에서 다뤘던 선형 회귀보다 한 단계 더 발전된 방법론을 제공합니다.



### MRI Parameter Mapping via Gaussian Mixture VAE: Breaking the Assumption of Independent Pixels (https://arxiv.org/abs/2411.10772)
Comments:
          NeurIPS 2024 Workshop in Machine Learning and the Physical Sciences

- **What's New**: 새로운 패러다임을 제안하여 MRI의 양적 파라미터 매핑(quantitative parameter mapping) 방식을 혁신적으로 변화시킵니다. 기존의 모델 피팅 방법은 각 voxel이 독립적이라는 가정을 해왔으나, 본 연구는 데이터 내의 중복성을 활용하여 이 가정을 깨뜨립니다.

- **Technical Details**: 자기 지도(self-supervised) 딥 변분(deep variational) 접근 방식을 사용하여, voxel 간의 의존성을 고려한 데이터 주도(data-driven) 정규화(regularisation)를 통해 양적 맵을 계산합니다. 이 방법은 Gaussian 혼합 사전(Gaussian mixture prior)을 활용하여 더욱 선명한 양적 맵을 생성합니다.

- **Performance Highlights**: dMRI 시뮬레이션 및 실제 데이터에서 기존 모델 피팅 기술보다 더 나은 성능을 보여주며, 미세한 해부학적 세부 사항을 드러냅니다. 이러한 접근 방식은 dMRI 및 qMRI와 같은 파라미터 매핑 방법의 임상 도입을 지원할 수 있습니다.



### Integrated Machine Learning and Survival Analysis Modeling for Enhanced Chronic Kidney Disease Risk Stratification (https://arxiv.org/abs/2411.10754)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 19 pages

- **What's New**: 본 연구에서는 만성 신장 질환(Chronic Kidney Disease, CKD)의 진행을 예측하기 위한 새로운 접근법을 제안합니다. 머신러닝(Machine Learning) 기법과 고전 통계 모델의 결합을 통해, CKD 진행을 예측하는 새로운 예측 변수를 찾는 것이 주요 내용입니다.

- **Technical Details**: 본 연구는 Shapley 값(Shapley values)을 사용하여 머신러닝 모델과 고전적인 생존 모델을 결합하여 CKD 진행에 대한 새로운 예측 변수를 식별합니다. 선형 모델, 트리 기반 방법, 심층 학습 모델을 평가하여 기존의 신장 장애 위험 방정식(Kidney Failure Risk Equation, KFRE)에 통합 가능한 새로운 예측 변수를 도출하고, 이를 Cox 비례 위험 모델(Cox Proportional Hazards Models, CPHMs)에 적용합니다.

- **Performance Highlights**: 기존의 임상 예측 변수와 결합된 머신러닝 기반의 예측 변수가 CKD 진행 예측의 정확성을 향상시키며, C-index의 증가와 Brier 점수의 감소를 통해 이러한 성능 개선이 입증되었습니다. 실험은 MIMIC-IV 데이터셋을 사용해 CKD 진단이 문서화된 환자 14,012명을 포함하며, 1,483명(10.6%)의 환자가 CKD 단계 진행을 경험했습니다.



### Chain-of-Programming (CoP) : Empowering Large Language Models for Geospatial Code Generation (https://arxiv.org/abs/2411.10753)
- **What's New**: 이 논문에서는 지리공간 모델링에 대한 학제 간 수요가 증가하고 대규모 언어 모델(LLMs)의 발전을 바탕으로 지리공간 코드 생성 기술의 최신 발전을 다루고 있습니다. 특히, 기존 LLM이 사용자 요구 사항의 불완전함 및 특정 플랫폼 구문 규칙에 대한 지식 부족으로 인해 코드 생성 과정에서 "code hallucination" 현상에 직면하고 있다는 문제를 제기합니다.

- **Technical Details**: 이 논문에서 제안된 Chain of Programming (CoP) 프레임워크는 코드 생성 과정을 요구 사항 분석, 알고리즘 설계, 코드 구현, 코드 디버깅, 코드 주석 달기 등 다섯 단계로 세분화합니다. CoP는 공유 정보 풀, 지식 기반 검색, 사용자 피드백 메커니즘을 통합하여 모델의 미세 조정 없이 요구 사항에서 코드까지의 엔드 투 엔드(code generation flow)를 형성합니다.

- **Performance Highlights**: CoP 전략은 지리공간 문제 분류 프레임워크 및 평가 기준을 기반으로 하여 생성된 코드의 논리적 명확성, 구문적 정확성 및 실행 가능성을 3.0%에서 48.8%까지 향상시킵니다. 비교 실험 및 소거 실험을 통해 CoP 전략이 다른 최적화 접근법보다 우수하다는 것을 입증하며, 구성 요소의 합리성과 필요성을 확인합니다.



### LTCXNet: Advancing Chest X-Ray Analysis with Solutions for Long-Tailed Multi-Label Classification and Fairness Challenges (https://arxiv.org/abs/2411.10746)
Comments:
          8 pages, 5 figures

- **What's New**: 이번 연구에서는 Chest X-ray (CXR) 데이터를 다루기 위해 새롭게 Pruned MIMIC-CXR-LT 데이터셋을 구축했습니다. 이 데이터셋은 긴 꼬리(long-tailed) 분포와 다중 라벨(multi-label) 상황을 반영하기 위해 설계되었습니다. LTCXNet이라는 새로운 프레임워크를 도입하였으며, ConvNeXt 모델과 ML-Decoder, 전략적 데이터 증강(data augmentation) 등이 통합되었습니다.

- **Technical Details**: LTCXNet은 긴 꼬리 분포의 다중 라벨 분류 문제를 해결하기 위해 ConvNeXt 계열의 아키텍처를 채택했습니다. 모델은 ‘Head’, ‘Tail’, ‘All’ 세 가지 고유한 모델로 훈련되어 각 모델이 특정 라벨 세트에 중점을 두고 있으며, 데이터셋은 훈련, 검증 및 테스트 세트로 나뉘어 182,380, 20,360, 54,268 이미지로 구성됩니다. 특히 ML-Decoder는 기존 transformer의 self-attention 메커니즘을 제거하여 효율성을 높였습니다.

- **Performance Highlights**: LTCXNet은 CXR 해석의 모든 클래스에서 성능을 향상시켰으며, 특히 ‘Pneumoperitoneum’과 ‘Pneumomediastinum’과 같은 드문 클래스에서 각각 79% 및 48% 향상을 보여주었습니다. 요약된 연구 결과들은 공정성을 평가하고 정확도를 높이기 위한 다양한 접근 방식을 모색하며, 모델의 공정성을 보장하여 다양한 인구 집단에 대한 정확도를 균일하게 유지하는 데 기여하고 있습니다.



### Digital-Analog Quantum Machine Learning (https://arxiv.org/abs/2411.10744)
Comments:
          Invited Perspective for Advanced Intelligent Discovery

- **What's New**: 이 논문에서는 머신 러닝(Machine Learning) 알고리즘이 산업과 사회에서 널리 사용되고 있으며, 양자 시스템(Quantum Systems)이 이 계산을 스케일링할 수 있는 잠재력을 가지고 있음을 강조합니다. 특히, 고전적인 컴퓨터가 처리하기 어려운 대규모 데이터를 다루기 위해 디지털-아날로그 양자 패러다임(Digital-Analog Quantum Paradigm)의 활용 가능성을 제안합니다. 이러한 방법은 양자 컴퓨터의 진정한 양자 속성인 중첩(Superposition) 및 얽힘(Entanglement)을 활용하여 효율적인 계산을 가능하게 할 수 있습니다.

- **Technical Details**: 양자 머신 러닝(Quantum Machine Learning) 분야에서는 디지털-아날로그 양자 프로토콜을 이용하여 머신 러닝 계산이 수행됩니다. 이 새로운 접근법(DEAQ)에서는 대형 아날로그 블록과 소형 디지털 게이트를 결합하여 양자 시스템의 새로운 속성을 학습할 수 있는 방법을 제시합니다. 이를 통해 NISQ(Noise Intermediate-Scale Quantum) 장치에서 유용한 작업을 수행할 수 있는 가능성이 열립니다.

- **Performance Highlights**: 최근 연구들은 디지털-아날로그 양자 프로토콜을 통해 양자 머신 러닝 알고리즘과 결합된 결과가 효과적일 수 있음을 보여줍니다. 특히, Rydberg 원자 시스템을 활용한 디지털-아날로그 양자 학습 알고리즘에서 놀라운 결과를 보고하였으며, 이는 고전적인 머신 러닝 계산보다 양자 우위를 제공할 가능성이 있습니다. 향후 양자 기술 분야의 발전이 산업 및 사회에 미치는 긍정적인 영향을 기대할 수 있습니다.



### MetaLA: Unified Optimal Linear Approximation to Softmax Attention Map (https://arxiv.org/abs/2411.10741)
- **What's New**: 본 연구에서는 기존의 다양한 linear complexity 모델들(LinFormer, SSM, LinRNN)의 optimal design에 대한 문제를 다룹니다. 이들은 Transformer 구조의 conventional softmax attention을 대체하기 위해 제안되었습니다.

- **Technical Details**: 기존 모델을 unified linear attention 형태로 통합하여, optimal linear attention design의 세 가지 조건을 제시합니다: 1) Dynamic memory ability; 2) Static approximation ability; 3) Least parameter approximation. 제안하는 Meta Linear Attention(MetaLA) 모듈은 이러한 조건을 만족하도록 설계되었습니다.

- **Performance Highlights**: Multi-Query Associative Recall(MQAR) 작업, 언어 모델링, 이미지 분류, Long-Range Arena(LRA) 벤치마크 실험에서 MetaLA가 기존의 linear 모델보다 효과적임을 보였습니다.



### HELENE: Hessian Layer-wise Clipping and Gradient Annealing for Accelerating Fine-tuning LLM with Zeroth-order Optimization (https://arxiv.org/abs/2411.10696)
- **What's New**: 본 논문에서는 모델의 파라미터에 따라 변화하는 곡률의 문제를 해결하기 위해 새로운 최적화 알고리즘인 HELENE을 소개합니다. HELENE은 메모리 효율적인 최적화 방법으로, 두 번째 차수 정보를 효과적으로 통합하여 조정된 신뢰성을 보장합니다.

- **Technical Details**: HELENE은 비모수 선형 회귀인 A-GNB 경량화 알고리즘을 활용하여 대각선 Hessian을 추정하며, 레이어별 클리핑을 통해 더 정확한 업데이트를 제공합니다. 다양한 레이어 간 차이를 처리하기 위해 클리핑을 조정하여 수렴 속도를 개선하였습니다.

- **Performance Highlights**: HELENE은 RoBERTa-large와 OPT-1.3B 모델에서 MeZO와 비교하여 최대 20배의 속도 향상과 평균 1.5%의 정확성 개선을 달성했습니다. 또한, HELENE은 전체 파라미터 조정 및 파라미터 효율적인 미세 조정(PEFT)과 호환되며 여러 최신 최적화 방법보다 우수한 성능을 보였습니다.



### DEBUG-HD: Debugging TinyML models on-device using Hyper-Dimensional computing (https://arxiv.org/abs/2411.10692)
Comments:
          Accepted at the Machine Learning for Systems Workshop at NeurIPS 2024

- **What's New**: 이 논문에서는 DEBUG-HD라는 새로운 on-device debugging 접근 방식을 제안합니다. 이는 KB 크기의 TinyML 장치에서 자원을 효율적으로 사용하여 모델의 실패 원인을 진단할 수 있도록 설계되었습니다.

- **Technical Details**: DEBUG-HD는 hyper-dimensional computing (HDC)을 활용하여 입력의 변화를 감지합니다. HDC 인코딩 기법을 기존 신경망과 결합하여 개발된 \\hddebug는 다양한 이미지 및 오디오 데이터셋에서 입력 손상을 감지하는 데 있어 기존 방법보다 평균 27% 향상된 성능을 보입니다.

- **Performance Highlights**: \\hddebug는 여러 데이터셋에서 이전의 최고 성능 모델보다 평균 12% 더 높은 정확도를 달성했습니다. 이는 공간적 제약이 있는 TinyML 장치에서 실시간으로 입력 손상 분류를 가능하게 하여 모델의 신뢰성을 높이는 데 기여합니다.



### Exploring Feature-based Knowledge Distillation For Recommender System: A Frequency Perspectiv (https://arxiv.org/abs/2411.10676)
- **What's New**: 이 논문에서는 주파수 관점(frequency perspective)에서 추천 시스템에 대한 feature-based 지식 증류(knowledge distillation)를 분석합니다. 지식을 여러 주파수 구성 요소로 정의하고, 이러한 구성 요소들을 동등하게 최소화하는 기존 방법이 중요한 지식이 간과되는 문제를 다루기 위해 FreqD라는 새로운 가벼운 지식 재조정 방법을 제안합니다.

- **Technical Details**: 기존의 feature-based knowledge distillation 방법은 지식 손실을 동등하게 최소화하는 데 초점을 맞추고 있습니다. 그러나 필자는 지식의 중요도를 고려하여 지식의 가중치를 재조정하는 방법을 제안합니다. FreqD는 각 지식에 대한 손실을 계산할 필요 없이 복잡한 계산 비용을 피할 수 있도록 설계되었습니다.

- **Performance Highlights**: Extensive experiments show that FreqD consistently outperforms state-of-the-art knowledge distillation methods across various public datasets. FreqD는 추천 성능 향상에 강력한 효과를 발휘해, 실제 사용에 적합한 경량화된 해결책으로써의 가능성을 보여줍니다.



### SAM Decoding: Speculative Decoding via Suffix Automaton (https://arxiv.org/abs/2411.10666)
Comments:
          13 pages, 3 figures

- **What's New**: 이번 논문에서는 SAM-Decoding이라는 새로운 도입 방안을 제안하여 현재의 n-gram 기반 방식의 한계를 극복하고 텍스트 생성을 더욱 효율적이고 정확하게 수행하는 방법론을 소개합니다. 이 방식은 기존의 리트리벌(retrieval) 기반 방법들과의 비교에서 상당한 속도 향상을 보여줍니다. SAM-Decoding은 기존 방법들과 결합하여 Adaptive하게 드래프트 생성 전략을 선택함으로써 LLM의 추론 속도를 증가시킵니다.

- **Technical Details**: SAM-Decoding은 Suffix Automaton을 이용하여 텍스트 생성 및 텍스트 데이터베이스에서 가장 긴 접미사 일치를 찾아내는 방식으로 작동합니다. 코드는 정적 및 동적 접미사 자동기를 각각 기존 텍스트와 입력 프롬프트에 대해 구성하여 신속하고 정확한 드래프트 생성을 가능하게 합니다. 이 과정에서, 가장 긴 접미사 일치는 O(1) 복잡도로 탐지되며 이는 기존 n-gram 방식보다 월등한 성능을 보여줍니다.

- **Performance Highlights**: SAM-Decoding은 Token Recycling과 결합될 경우 Spec-Bench에서 오토회귀 디코딩보다 2.27배의 속도 향상을 달성하며, EAGLE2와의 조합에서는 2.49배의 속도 향상을 기록하여 현재 모든 접근 방식 중에서 가장 높은 성능을 나타냅니다. 광범위한 평가에서 SAM-Decoding은 기존 최첨단 방법들과 비교해도 경쟁력 있는 결과를 생성하며 특히 리트리벌 기반 방법들이 적용 가능한 작업에서 우수한 성능을 보입니다.



### Understanding Learning with Sliced-Wasserstein Requires Rethinking Informative Slices (https://arxiv.org/abs/2411.10651)
- **What's New**: 본 논문은 Sliced-Wasserstein distances(SWD)의 새로운 접근법을 제안하여, 모든 슬라이스가 동등하게 유용하도록 1D Wasserstein 거리를 재조정하는 방법을 연구합니다.

- **Technical Details**: 전통적인 Sliced-Wasserstein 거리(SWD)는 1차원 서브스페이스로 분포를 투영하여 계산됩니다. 저자들은 데이터를 가정하고 슬라이스의 유용성 개념을 도입하여, 개별 슬라이스의 재조정이 SWD에 단일 전역 스케일링 인자로 간단하게 변환될 수 있음을 보입니다.

- **Performance Highlights**: 저자들은 다양한 머신 러닝 작업을 통해 전통적인 SWD가 복잡한 변형들과 동등하거나 더 나은 성능을 발휘할 수 있음을 입증하였습니다. 이를 통해 'Sliced-Wasserstein이 일반적인 학습 작업에 필요할까?'라는 질문에 대한 답을 제시합니다.



### MTA: Multimodal Task Alignment for BEV Perception and Captioning (https://arxiv.org/abs/2411.10639)
Comments:
          10 pages

- **What's New**: MTA라는 새로운 다중 모달 작업 정렬 프레임워크를 통해 BEV 기반 3D 인식과 자막 작업을 통합하여 성능을 향상시키는 연구를 발표하였습니다.

- **Technical Details**: MTA는 두 가지 주요 구성 요소인 BEV-Language Alignment (BLA)와 Detection-Captioning Alignment (DCA)로 구성되어 있으며, 각각 BEV 장면 표현을 지면 진술 언어 표현과 정렬하고 시각적 출력과 자막 출력을 일관되게 만듭니다. MTA는 기존 BEV 기반 프레임워크에 통합 가능하며, 추가적인 계산 복잡성을 도입하지 않습니다.

- **Performance Highlights**: MTA는 nuScenes와 TOD3Cap 데이터 세트에서 기존 최첨단 기술에 비해 각각 4.9%의 인식 향상과 9.2%의 자막 향상을 달성하였습니다. 이러한 결과는 BEV 기반 인식과 자막 작업의 통합 정렬의 효과를 뒷받침합니다.



### Gender Bias Mitigation for Bangla Classification Tasks (https://arxiv.org/abs/2411.10636)
- **What's New**: 이 연구에서는 방글라어에 대해 미리 훈련된 언어 모델에서 성별 편향(gender bias)을 조사하였습니다. 이는 저자원 언어에서 미처 탐구되지 않은 영역으로, 기존 데이터 세트를 사용하여 성별 이름 교환 기법을 적용하고 특정 작업에 적합한 데이터 세트를 만들었습니다. 이 연구는 다양한 성별 편향 완화 방법들과 우리가 제안한 새로운 접근 방식을 비교하여 효과성을 입증했습니다.

- **Technical Details**: 연구는 감정 분석(sentiment analysis), 독성 탐지(toxicity detection), 증오 발언 탐지(hate speech detection), 그리고 풍자 탐지(sarcasm detection)와 같은 작업에 대해 총 4개의 수작업으로 주석이 달린 데이터 세트를 준비했습니다. 또한, 우리는 교차 엔트로피 손실(cross entropy loss)과 코사인 유사성(cosine similarity)을 기반으로 한 공동 손실 최적화(joint loss optimization) 기법을 제안하였습니다. 데이터 세트 구축을 위해 성별 단어 쌍을 포함하는 사전(dictionary)을 만들고, 명명된 개체 인식(NER) 기법을 통해 성별 이름을 치환하는 방법을 사용했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 편향 완화 방법들은 물론 정확도를 유지하면서도 성별 편향을 효과적으로 줄이는 것으로 나타났습니다. 각각의 모델에서 편향 비율은 최악의 경우 7.48%에서 최선의 경우 0.46%로 측정되었습니다. 우리의 접근 방식이 간섭 없이 더 많은 분야에 적합하다는 것을 입증했습니다.



### Leveraging large language models for efficient representation learning for entity resolution (https://arxiv.org/abs/2411.10629)
Comments:
          22 pages and 12 figures

- **What's New**: 이 논문에서는 TriBERTa라는 감독 학습 기반의 엔티티 해상도(Resolution) 시스템을 제안합니다. 이 시스템은 사전 훈련(pre-trained)된 대형 언어 모델과 triplet loss function을 이용하여 엔티티 매칭을 위한 표현을 학습합니다. 이는 현대의 데이터 주도(data-driven) 환경에서 중복 데이터 식별 및 조정과 관련된 문제를 해결하는 데 기여합니다.

- **Technical Details**: TriBERTa 시스템은 두 단계로 구성됩니다. 첫 번째로, 엔티티 레코드가 SBERT(Sentence Bidirectional Encoder Representations from Transformers) 모델에 입력되어 벡터 표현(vector representations)이 생성됩니다. 이후에는 contrastive learning 방식을 거쳐 triplet loss function에 기반하여 세밀하게 조정된 표현(fine-tuned representations)을 엔티티 매칭 작업에 입력으로 사용합니다.

- **Performance Highlights**: 연구 결과, TriBERTa의 접근 방식은 세밀하게 조정되지 않은 SBERT 및 전통적인 TF-IDF(Term Frequency-Inverse Document Frequency)와 비교했을 때 3-19% 더 뛰어난 성능을 나타냅니다. 또한 TriBERTa로 생성된 표현은 여러 데이터셋에서 일관되게 높은 성능을 유지하며, 강인성을 증가시키는 효과도 확인되었습니다.



### Is thermography a viable solution for detecting pressure injuries in dark skin patients? (https://arxiv.org/abs/2411.10627)
Comments:
          9 pages

- **What's New**: 본 연구에서는 피부 색조가 더 어두운 35명을 대상으로 한 새로운 열 및 광학 이미징 데이터 세트를 도입했습니다. 이를 통해 피부 온도 차이를 유도하는 쿨링(cooling) 및 컵핑(cupping) 프로토콜을 통해 압박 손상(PI)의 조기 탐지를 위한 열화상 기술의 가능성을 탐구합니다. 여러 카메라와 조명 조건, 환자 자세 및 카메라 거리 등 다양한 데이터 수집 프로토콜을 비교하여 열화상 모델의 성능을 평가했습니다.

- **Technical Details**: 연구에서는 Eumelanin 색조 분류를 통해 대조군과 쿨링 프로토콜을 혼합하여 1680장의 이미지를 수집했습니다. CNN(Convolutional Neural Network) 모델, 특히 MobileNetV2를 사용하여 냉각 및 증기 치료의 두 가지 이진 분류 작업을 평가합니다. 이미징 프로토콜의 변화가 모델 성능에 미치는 영향을 분석하여, 실제 임상 환경에서 최적의 데이터 수집 접근 방식을 파악하는 데 중점을 두었습니다.

- **Performance Highlights**: 초기 결과에 따르면, 열화상 기반의 CNN 모델은 모든 피부 색조에서 데이터 수집 프로토콜의 영향에 강한 내성을 보였습니다. 연구는 기존의 기술적 한계를 극복하는 데 기여하고, PI 탐지의 정확성을 높이는 방법을 제시합니다. 이러한 발견은 다양한 피부 색조를 가진 환자에 대한 효과적인 압박 손상 탐지를 위한 치료 및 예방 전략에 큰 영향을 미칠 것입니다.



### Weak Permission is not Well-Founded, Grounded and Stab (https://arxiv.org/abs/2411.10624)
- **What's New**: 이 논문은 약한 허가(weak permission)에 대한 개념을 탐구하며, 의무(conflict)와 관련된 비모노토닉(non-monotonic) 추론을 통해 이러한 개념을 분석합니다. 특히, 잘 정의된 세 가지 의미론(잘 설계된 의미론, 안정적 의미론, 회의적 안정성)에서는 약한 허가를 포착할 수 없음을 보여줍니다.

- **Technical Details**: 논문의 주제는 의무, 허가, 금지 등의 개념을 다루는 법리(logic programming)와 구조적 논증(argumentation) 측면에서 접근하며, 특히 비모노토닉 추론의 두 가지 파라다임을 집중적으로 분석합니다. 이는 규범적 추론(normative reasoning)의 본질적 비모노토닉 특성을 반영하며, 기존의 여러 논의(특히 Sergot et al. (1986) 및 Horty (1993))와 연결됩니다.

- **Performance Highlights**: 결과적으로, 이 연구는 약한 허가는 법적 갈등(deontic conflicts)이 존재할 경우 잘 정의된 의미론(based on well-founded, grounded and stable)에서는 지원되지 않았음을 확인하였습니다. 이러한 발견은 법적 추론을 모델링하기 위한 기존 접근 방식에 대한 중요한 통찰력을 제공합니다.



### Attraction-Repulsion Swarming: A Generalized Framework of t-SNE via Force Normalization and Tunable Interactions (https://arxiv.org/abs/2411.10617)
- **What's New**: 본 논문에서는 Attraction-Repulsion Swarming (ARS) 동역학 기반의 새로운 데이터 시각화 방법인 ARS 시각화를 제안합니다. 이 방법은 t-SNE (t-distributed stochastic neighbor embedding) 기술을 수정하여, 상호작용하는 에이전트들의 집합으로 시각화를 수행합니다.

- **Technical Details**: ARS는 t-SNE의 역학을 수정하여 '총 영향(total influence)'에 의해 정규화된 과정을 포함합니다. 이로 인해 데이터 크기와 상관없이 같은 시간 간격(h=1)에서 단순한 반복 과정을 사용하며, 복잡한 최적화 기법 없이 빠른 수렴을 가능하게 합니다. ARS는 애트랙션(attraction)과 리펄션(repulsion) 커널을 별도로 조정할 수 있어 클러스터 간의 분리와 클러스터의 조밀도를 제어할 수 있습니다.

- **Performance Highlights**: 초기 실험 결과, ARS 시각화는 MNIST 및 Cifar-10 데이터 세트에서 t-SNE와 유사한 성능을 보이며, 특히 더 강한 애트랙션을 제공하고 약한 리펄션을 사용할 때 더 나은 시각화 결과를 나타냅니다. ARS는 또한 t-SNE의 느린 수렴 문제를 해결할 수 있는 가능성을 제시합니다.



### AmoebaLLM: Constructing Any-Shape Large Language Models for Efficient and Instant Deploymen (https://arxiv.org/abs/2411.10606)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문은 AmoebaLLM이라는 새로운 프레임워크를 제안하여 다양한 플랫폼 및 응용 프로그램에 대해 최적화된 LLM(Large Language Models) 서브넷을 즉각적으로 생성할 수 있게 한다. 이는 정확도-효율성 프론티어를 달성하면서 단 한 번의 파인튜닝으로 가능합니다.

- **Technical Details**: AmoebaLLM은 (1) 지식 보존 서브넷 선택 전략, (2) LoRA(.... Low-rank Adaptation)의 조합을 통한 서브넷 간의 경량화 설치, (3) 손실 크기 균형을 갖춘 인장 증류 기법을 포함한 3가지 주요 구성 요소로 구성되어 있다. 이 프레임워크는 여러 플랫폼의 하드웨어 제약 및 응용 요구 사항을 대응하기 위한 적응형 LLM 설계를 가능하게 한다.

- **Performance Highlights**: AmoebaLLM은 새로운 SOTA(State-of-the-Art) 정확도-효율성 트레이드오프를 달성하는 서브넷을 생성하여, 기존의 효율적인 LLM 솔루션보다 높은 성능을 보인다. 실험 결과 이 프레임워크는 다양한 플랫폼에 최적화된 LLM의 신속한 배포를 가능하게 하며, 압축 과정에서도 새로운 SOTA 성능을 달성했다.



### Generating Energy-efficient code with LLMs (https://arxiv.org/abs/2411.10599)
- **What's New**: 최근 개인용 컴퓨터와 데이터 센터의 전력 수요 증가가 기후 변화에 기여하고 있으며, 이러한 문제를 해결하기 위해 코드의 에너지 소비를 줄이는 것이 필수적이다. 본 연구에서는 다양한 파라미터를 가진 Large Language Models (LLMs)가 생성한 코드의 에너지 소비를 최소화하기 위한 프롬프트 최적화의 영향을 다룬다. 연구팀은 여러 개의 Python 문제를 통해 다양한 프롬프트 수정 전략을 실험해 에너지 효율성을 높이는 방법을 모색하였다.

- **Technical Details**: 연구에 사용된 LLM 모델로는 CodeLlama-70b, CodeLlama-70b-Instruct, CodeLlama-70b-Python, DeepSeek-Coder-33b-base, 그리고 DeepSeek-Coder-33b-instruct가 포함된다. 프롬프트 수정 전략으로는 '이 문제에 대한 에너지 최적화 솔루션을 제공해 주세요'라는 문장을 추가하거나, 두 가지 Python 코딩 최적 모범 사례를 사용하는 방식이 포함되었다. 본 연구는 특정 조합의 프롬프트 최적화, LLM, 그리고 Python 문제에 대한 결과로 에너지 소비의 감소를 발견하였다.

- **Performance Highlights**: 실험 결과, 특정 LLM 및 Python 문제 조합에 따라 에너지 소비가 감소하는 경향이 나타났지만, 동일한 LLM에 대해 모든 문제에서 일관된 에너지 소비 감소를 초래하는 단일 최적화 프롬프트는 발견되지 않았다. 이는 LLM의 에너지 효율성이 문제의 난이도에 따라 다르게 나타날 수 있음을 시사하며, 프롬프트 최적화가 에너지 소비에 긍정적인 영향을 미칠 수 있는 가능성을 보여준다.



### A minimalistic representation model for head direction system (https://arxiv.org/abs/2411.10596)
Comments:
          Workshop on Symmetry and Geometry in Neural Representations (NeurReps) at NeurIPS 2024, Extended Abstract Track

- **What's New**: 이 논문에서는 헤드 방향(HD) 시스템을 위한 최소한의 표현 모델을 제안하고 있습니다. 이 모델은 HD 세포의 중요한 특성을 포착하는 고차원 표현을 학습하는 것을 목표로 하며, 회전군 $U(1)$의 표현으로 구성됩니다. 연구팀은 완전 연결 네트워크와 합성곱 네트워크 두 가지 버전을 검토하며, 이 두 모델에서 가우시안 형의 튜닝 프로파일과 2D 원형 기하구조의 발생을 보여줍니다.

- **Technical Details**: 헤드 방향을 연속적인 d차원 벡터로 표현하며, 이는 HD 세포의 반응으로 여겨집니다. 이 모델은 세 가지 제약조건을 따릅니다: 변환 규칙, 비부정성 제약, 그리고 단위 노름 제약. 이 모델은 경로 통합을 가능하게 하는 순환 신경망으로 정의되며, 로컬 운동에 대한 Taylor 전개를 통해 모델의 기능을 구체화합니다.

- **Performance Highlights**: 제안된 모델은 주어진 방향에 대해 가우시안 형의 튜닝 프로파일을 학습할 수 있으며, 주성분 분석(PCA)을 통해 시각화했을 때 명확한 원형 기하 구조를 나타냅니다. 학습된 모델은 정확한 경로 통합을 수행할 수 있으며, 이러한 특성은 생물학적 HD 시스템의 특성과 밀접하게 일치합니다.



### A dataset of questions on decision-theoretic reasoning in Newcomb-like problems (https://arxiv.org/abs/2411.10588)
Comments:
          48 pages, 15 figures; code and data at this https URL

- **What's New**: 이번 논문에서는 Newcomb-like 문제의 의사결정 이론에 관한 자연어 질문 데이터셋을 소개합니다. Newcomb-like 문제는 에이전트가 유사한 다른 에이전트와 상호작용하는 상황을 포함하며, 이는 다른 에이전트가 유사한 방식으로 추론할 것임을 고려해야 함을 의미합니다. LLM의 Newcomb-like 문제에 대한 추론 평가가 중요한 이유는 기초 모델 파생 에이전트 간의 상호작용이 이러한 문제와 유사할 것이기 때문입니다.

- **Technical Details**: 데이터셋은 고유하고 논란의 여지가 없는 답변을 가진 능력 질문(capability questions)과 의사결정 이론가들 간의 논쟁이 있는 태도 질문(attitude questions)을 모두 포함합니다. 연구는 기존 OpenAI, Anthropic, Meta, GDM, Reka 등의 다양한 모델과 간단한 프롬프트 기반 개입 하의 모델을 조사하는 데 사용됩니다. 이러한 질문을 통해 의사결정 이론적 역량과 표현된 태도 간의 상호작용을 분석합니다.

- **Performance Highlights**: 연구 결과, 기존 모델 간의 태도가 상당히 다르며, 높은 역량은 이른바 증거적 의사결정 이론(evidential decision theory)으로의 보다 긍정적인 태도와 연관되어 있다는 것을 발견했습니다. 또한, 태도는 다양한 질문 유형에서 일관성이 있음을 보여주었습니다.



### On the Shortcut Learning in Multilingual Neural Machine Translation (https://arxiv.org/abs/2411.10581)
Comments:
          Accepted by Neurocomputing 2024

- **What's New**: 이 연구에서는 다국어 신경 기계 번역(Multilingual Neural Machine Translation, MNMT)에서 자주 언급되는 목표 외 문제(off-target issue)를 재조명합니다. 실험 설계를 통해 zero-shot 번역에서 비중심(non-centric) 언어를 중심 언어로 잘못 번역하는 경향이 있음을 밝혀냈습니다. 또한, 특징적으로 멀티링구얼 프리트레이닝(multilingual pretraining)이 이러한 편향을 심화시키는 것으로 나타났습니다.

- **Technical Details**: 우리는 MNMT 모델의 훈련 과정에서 발생하는 단축키 학습(shortcut learning)을 다루며, 이는 지도 언어 매핑(supervised language mapping)의 과적합(overfitting)과 연결됩니다. 연구팀은 훈련 데이터 내에서 비중심 언어 쌍의 인스턴스를 제거함으로써 단축키를 없애는 '일반화 훈련(generalization training)'이라는 새로운 훈련 전략을 제안합니다. 이 방법은 MNMT 모델 성능을 향상시키면서 추가적인 데이터나 계산 비용이 들지 않습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 zero-shot 번역 성능을 일관되게 그리고 현저하게 개선시키는 것으로 나타났습니다. 또한, 지도 번역(supervised translation) 성능을 유지하면서도, 다양한 언어 분포 및 벤치마크에서 성능을 비교한 결과, 기존의 강력한 기준선(baselines)보다 뛰어난 성능을 보였습니다. 이 연구는 제로 샷 번역의 일반화 능력을 개선하기 위한 첫 번째 중요한 기여로 인정받을 것입니다.



### Vision Eagle Attention: A New Lens for Advancing Image Classification (https://arxiv.org/abs/2411.10564)
Comments:
          7 pages, 2 figures, 3 tables

- **What's New**: 이번 연구는 컴퓨터 비전 과제에서 효율적인 특징 추출을 위해 Vision Eagle Attention이라는 새로운 주의 메커니즘을 도입하였습니다. 이 메커니즘은 컨볼루션(spatial attention)을 사용하여 이미지의 유용한 지역에 집중하게끔 하여, 모델의 성능을 개선합니다. 특히, 이 연구에서는 경량화된 ResNet-18 아키텍처에 통합되어 높은 효율성을 보여줍니다.

- **Technical Details**: 제안된 Vision Eagle Attention 모델은 페이즈마다 특징을 정제하기 위해 ResNet-18 백본의 여러 레이어 이후에 배치된 세 개의 Attention 블록으로 구성됩니다. 각 블록에서는 3x3 및 1x1 컨볼루션을 통해 지역 특징을 캡처하고, 이 결과를 ResNet의 출력과 곱하여 중요한 특징을 강조합니다. 이는 네트워크가 모델의 깊이에 따라 중요한 특징을 동적으로 강조할 수 있게 합니다.

- **Performance Highlights**: 제안된 모델은 FashionMNIST, Intel Image Classification, OracleMNIST의 세 가지 광범위한 벤치마크 데이터셋에서 평가되었으며, 이미지 분류 작업에 중점을 두고 분류 정확도를 향상시켰습니다. 실험 결과, Vision Eagle Attention은 모델의 성능을 크게 개선하는데 기여하였으며, 객체 탐지 및 분할과 같은 다른 비전 작업으로도 확장 가능성이 있음을 보여줍니다.



### Pragmatic information of aesthetic appraisa (https://arxiv.org/abs/2411.10561)
Comments:
          10 pages, 3 figures

- **What's New**: 이번 연구는 미적 평가를 위한 현상학적 모델을 제안하며, 이는 미적 감상자의 신념 상태에 대한 역동적 업데이트의 의미(meaning)와 관련이 있습니다. 이 모델은 서양 조성 음악(cadential effects)에 대한 실험을 통해 미적 쾌락 평가와 질적으로 상관관계를 보여주었습니다. 또한, 이 논문에서는 관련된 계산적(computational) 및 신경역동적(neurodynamical) 설명을 논의하고 있습니다.

- **Technical Details**: 미적 판단은 정규성(regularity), 대칭(symmetry), 질서(orderliness)와 같은 추상적인 특징을 포함합니다. 연구자들은 정보 이론(information theory), 복잡성 과학(complexity science), 기계 학습(machine learning) 방법을 사용하여 세밀한 분석을 수행했습니다. 특히, Cheung et al은 맥락의 불확실성(context uncertainty)과 자극의 놀라움(stimulus surprisal)이 미적 쾌락에 미치는 비선형적(nonlinear) 변화를 분석하였습니다.

- **Performance Highlights**: 기존 연구에서는 최적의 미적 평가가 "Wundt curve"(수학적 감정 곡선)와 유사한 형태를 띤다는 것이 밝혀졌습니다. 여러 실험에서 주어진 자극의 복잡도(complexity)와 미적 선호도(preference)之间의 관계가 나타났습니다. 이러한 연구들은 미적 감상의 전반적인 이해를 높이고, 관련 신경 회로 및 인지 작용을 이해하는 데 기여하고 있습니다.



### Low-Rank Optimal Transport through Factor Relaxation with Latent Coupling (https://arxiv.org/abs/2411.10555)
Comments:
          53 pages, 13 figures, NeurIPS 2024. Comments welcome!

- **What's New**: 이 논문에서는 기존의 k-Wasserstein barycenter 문제와 관련하여 저차원의 운송 문제(low-rank optimal transport, OT)를 처리하기 위해 새로운 접근법을 제안합니다. 특히, latent coupling (LC) 분해를 기반으로 한 새로운 알고리즘인 Factor Relaxation with Latent Coupling (FRLC)를 도입하였습니다.

- **Technical Details**: 이 알고리즘은 OT 문제를 세 가지 서브 문제로 분리할 수 있어, 문제의 유연성과 해석 가능성을 높입니다. FRLC는 coordinate mirror descent를 사용하여 LC 분해를 계산하며, 다양한 OT 목표(Wasserstein, Gromov-Wasserstein, Fused Gromov-Wasserstein)와 마진 제약 조건(균형, 비균형, 반완화)의 처리를 지원합니다.

- **Performance Highlights**: 실험 결과, FRLC는 그래프 클러스터링 및 공간 전사체(spatial transcriptomics) 등 다양한 응용 분야에서 우수한 성능을 보이며, 해석 가능성을 제공하는 것으로 나타났습니다.



### Chain of Alignment: Integrating Public Will with Expert Intelligence for Language Model Alignmen (https://arxiv.org/abs/2411.10534)
Comments:
          Pluralistic Alignment Workshop at NeurIPS 2024

- **What's New**: 이 논문은 공공의 의사와 언어 모델(LM) 행동 간의 정렬(alignment)을 측정하는 새로운 방법을 소개합니다. '연결 고리 정렬'(Chain of Alignment, CoA) 접근 방식은 규칙 기반 보상(rule-based reward, RBR)을 생성하여 모델 행동에 대한 규칙을 공공의 의사와 조화롭게 만듭니다. 이 방법은 대중이 직접적으로 자신의 의사를 표현할 수 있도록 하며, 전문가 지성을 통해 가장 잘 달성 가능한 모델 행동 규칙을 파악합니다.

- **Technical Details**: CoA 접근 방식은 두 가지 요소로 구성됩니다: 공공의 의사를 표현하는 규범적 목표(normative objectives)와 그 목표를 달성하기 위한 관찰 가능한 모델 행동을 정의하는 경험적 규칙(empirical rules)입니다. 이 접근 방식은 대중의 의사를 규명하는 데 중요한 역할을 하며, 명확한 규칙을 통해 모델 행동을 평가할 수 있습니다. 우리는 세 가지 정신 건강 관련 LM 프롬프트에 대해 RBR을 생성하여 이를 검증했습니다.

- **Performance Highlights**: 실험 결과, 생성된 규칙 기반 보상(RBR)은 미국 대중의 96% 이상이 지지하는 규범적 목표에 대한 LM 응답의 정렬을 평가하는 데 있어 정신 건강 전문가와 유사한 성과를 보였습니다. Pearson 상관계수(Pearson's r)가 0.841에 달하며, AUC는 0.964로 최적의 성능을 기록했습니다. 이 연구는 LM의 행동과 공공의 의사 간의 정렬을 측정하는 유용한 도구가 될 것입니다.



### USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting (https://arxiv.org/abs/2411.10504)
- **What's New**: 본 논문에서는 Spike 카메라의 이미지 재구성과 포즈 보정을 통합한 새로운 최적화 프레임워크인 USP-Gaussian을 제안합니다. 기존의 분산된(cascading) 접근 방식에서 발생하는 오류를 방지하기 위해 Spike-to-Image 네트워크와 3D Gaussian Splatting(3DGS) 간의 정보를 원활하게 통합합니다. 이와 같은 통합 접근법은 3D 재구성의 정확성을 크게 향상시킵니다.

- **Technical Details**: USP-Gaussian은 Spike-to-Image 재구성을 위한 Recon-Net과 3DGS로 구성된 두 개의 상호 작용하는(branch) 구조를 가지고 있으며, 각 시스템은 독립적으로 작동합니다. Recon-Net의 자기 지도(self-supervised) 훈련은 입력 스파이크 스트림과 복구된 출력 간의 제약으로 설계됩니다. 3DGS 브랜치는 고품질 이미지나 정확한 포즈가 필요 없이 구동될 수 있는 새로운 손실 함수 ℒgs을 개발하여 포즈 최적화를 통해 형태를 보강합니다.

- **Performance Highlights**: 우리의 방법은 이전의 접근 방법들과 비교하여 복구된 이미지의 품질에서 우수한 성능을 보여주었습니다. 특히 실제 데이터셋에서 부정확한 초기 포즈로 인해 발생하는 노이즈를 효과적으로 줄이고 세밀한 질감을 유지함으로써 3D 재구성의 견고성을 달성했습니다. 실험 결과, USP-Gaussian은 다양한 데이터셋에서 우수한 성능을 입증하였으며, 획기적인 오류 전파 완화 방식으로 주목받습니다.



### Edge-Only Universal Adversarial Attacks in Distributed Learning (https://arxiv.org/abs/2411.10500)
- **What's New**: 이번 연구는 분산 학습 환경에서 모델의 일부만을 사용할 경우, 즉 Edge 측면에서만 공격자가 접근할 때 생성 가능한 보편적인 적대적 공격(Universal Adversarial Attacks, UAPs)의 가능성을 탐구합니다. 이전의 전통적인 UAPs가 전체 모델에 대한 접근을 필요로 하는 것과는 달리, 이 연구는 Edge 측에서 핵심 기능을 활용하여 클라우드 부분에서 효과적인 오예측을 유도할 수 있음을 보여줍니다. 연구 결과는 ImageNet 데이터셋에서 강력한 공격 전이 능력을 입증하며, Edge만의 지식으로도 타깃된 적대적 효과를 achieved 할 수 있는 방법을 제공합니다.

- **Technical Details**: 본 연구에서는 Edge 측에서 사용 가능한 중간 특징을 활용하여 경량 분류기를 학습하고, 이를 활용하여 새로운 타깃 최적화 방법으로 효과적인 UAPs를 제작하는 방법을 소개합니다. 저자들은 여러 실험을 통해 Edge에서 몇 개의 레이어만을 사용하더라도 높은 오예측률을 유도할 수 있음을 입증하였으며, 일반적인 화이트 박스 UAP 공격과 비교할 수 있는 성과를 달성했습니다. 이 연구는 모델의 클라우드 부분에 대한 접근 없이도 공격이 처음 레이어에서만 적용되더라도 특정 클래스에 대한 방향전을 조정할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 공격 방식은 공격자가 모델의 클라우드 부분에 대한 접근 없이도, 초기 레이어만으로 강력한 비정확도를 유도하는 결과를 보여주었습니다. ImageNet 상에서 수행된 여러 실험에서, 적은 수의 레이어로도 유의미한 효과를 발휘하며, 화이트 박스 UAP 공격과 비교할 때 성능의 유사성을 보였습니다. 이 연구는 분산 추론에서 Edge-only 공격의 기초를 제공하며, 앞으로의 연구에서 부분 모델 접근의 중요성을 강조합니다.



### Guided Learning: Lubricating End-to-End Modeling for Multi-stage Decision-making (https://arxiv.org/abs/2411.10496)
- **What's New**: 복합적인 의사결정 과정의 중요성을 강조하며, 이를 위한 새로운 방법론인 Guided Learning을 제안합니다. 이 방법론은 여러 단계의 의사결정 과정을 개선하기 위해 설계되었습니다.

- **Technical Details**: Guided Learning은 중간 신경망 레이어의 훈련을 특정 목표에 맞추어 유도하는 '가이드'(guide)라는 함수를 도입합니다. 이 과정에서 '보상'(utility function)을 측정하여 의사결정의 품질을 향상시킵니다.

- **Performance Highlights**: 양적 투자 전략 구축 실험 결과, Guided Learning은 전통적인 단계별 접근법 및 기존의 end-to-end 방식보다 뛰어난 성능을 보였습니다.



### AI-Spectra: A Visual Dashboard for Model Multiplicity to Enhance Informed and Transparent Decision-Making (https://arxiv.org/abs/2411.10490)
Comments:
          Accepted for publication in an LNCS Volume "Engineering Interactive Computer Systems - EICS 2024 - International Workshops and Doctoral Consortium, Selected Papers"

- **What's New**: 본 논문에서는 AI 모델의 다양성을 활용하여 사용자에게 보다 투명하고 신뢰할 수 있는 의사 결정을 지원하는 AI-Spectra라는 접근 방식을 제안합니다. AI-Spectra는 여러 AI 모델이 동시에 결과를 생성하도록 하여 사용자에게 균형 잡힌 정보를 제공합니다. 이 시스템은 사용자가 단일 모델의 결과뿐만 아니라 다양한 출력 정보를 이해할 수 있도록 도와 줍니다. 특히, Chernoff faces 기술을 활용하여 복잡한 다변량 모델 구성 및 예측을 비교 쉽게 시각화합니다.

- **Technical Details**: AI-Spectra는 MNIST 데이터 셋을 활용하여 숫자 인식 모델을 훈련시키고, 다양한 AI 모델에서의 결과를 시각적으로 비교할 수 있는 대시보드를 제공합니다. 이 시스템은 사용자가 모델의 하이퍼파라미터 및 속성을 인지할 수 있도록 하기 위해 Chernoff Faces 기법을 맞춤화하여 사용합니다. 이를 통해 최종 사용자에게 AI 모델의 성능 및 예측 간의 관계를 효과적으로 전달합니다.

- **Performance Highlights**: AI-Spectra는 여러 AI 모델이 제공하는 예측의 분포를 시각적으로 나타내어 사용자가 보다 신뢰할 수 있는 결정을 내리는 데 기여합니다. 실험 결과, AI-Spectra를 사용한 경우 단일 모델에 비해 사용자들의 신뢰도와 투명성이 향상되었습니다. 이 접근 방식은 특히 의료 진단 및 금융 예측과 같은 중요한 의사 결정 지원 시스템에서 유용합니다. 이를 통해 AI 시스템의 신뢰성 향상, 오류 감소, 지속적인 학습 및 개선을 촉진할 수 있습니다.



### Biometrics in Extended Reality: A Review (https://arxiv.org/abs/2411.10489)
- **What's New**: 이 연구는 XR(Extended Reality) 환경에서 생체 정보(biometric characteristics)의 응용에 대해 체계적으로 조사한 첫 번째 논문이다. 우리는 사용자의 인증(authentication)과 표현을 위한 다양한 생체 모달리티(biometric modalities)를 종합적으로 정리하였다. 또한 XR 시스템의 생체 정보와 관련된 취약점(vulnerability gateways)을 문헌에서 처음으로 언급하고, 이에 대한 분류법(taxonomy)을 소개했다.

- **Technical Details**: 제안된 연구 방법론은 XR 시스템의 생체 인식 기반 워크플로우에서 주요 취약점과 가능한 공격 유형을 규명하는 것을 포함한다. 318편의 논문을 검토하여, 62편의 관련 논문의 리뷰를 통해 현재의 흐름과 다양한 구성 요소에 대한 명확한 이해를 제공하였다. 생체 인식 기반 포토리얼리스틱 아바타(photo-realistic avatars) 생성 및 인증의 접근 방식도 논의되었다.

- **Performance Highlights**: XR 환경에서 생체 인식의 유용성은 향후 이러한 시스템의 보안(security)과 사용자 개인 정보 보호(user privacy)를 보장하는 데 필수적이다. 이 연구는 생리학적(physiological) 및 행동적(behavioral) 생체 인식 특징을 폭넓게 논의하며, XR 애플리케이션에서 광범위하게 사용되는 생체 인식 기반 포토리얼리스틱 아바타 생성 기술에 대한 포괄적인 개요를 제공한다. 또한 XR 애플리케이션 내 생체 인식의 잠재력에 대한 미래 연구의 기회를 제시한다.



### The Future of Skill: What Is It to Be Skilled at Work? (https://arxiv.org/abs/2411.10488)
- **What's New**: 이 논문에서는 '일에서 숙련되다는 것이 무엇인가?'라는 질문을 던지며 현대 작업 환경에서 협업 도구의 역할을 탐구하고 있습니다. 저자들은 기존의 지능 지표와 달리 숙련에 대한 이해를 통해 다양한 실천, 장소, 도구, 협업의 얽힘을 조명합니다. AI가 도입되는 시대에서 일의 미래를 설계하는 데 있어 단순한 물리적 환경이나 별도의 지능 컨테이너를 넘어서는 복합적인 접근이 필요하다고 주장합니다.

- **Technical Details**: 논문에서는 Tim Ingold의 연구를 바탕으로 숙련의 개념을 재조명합니다. 이를 통해 외과 수술과 같은 고숙련 환경에서의 '전문적인 시각'을 통해 능숙함이 어떻게 드러나는지를 설명합니다. 숙련은 개인에게 국한된 속성이 아니라 협업을 통한 지속적인 행위로 파악되며, 이는 숙련이 객체와 행위자 간에 어떻게 분산되고 조정되는지를 보여줍니다.

- **Performance Highlights**: Slack 및 Microsoft Teams와 같은 현대의 협업 도구가 제공하는 독특한 공간적 및 시간적 단절 속에서, 숙련이 팀원 간의 상호작용 및 커뮤니케이션 구조에 깊이 연결되어 있다는 점을 발견했습니다. 저자들은 그러한 도구들이 숙련된 팀워크를 가능하게 하는 방법에 대해 논의하며, AI가 숙련된 작업에 미치는 영향을 재고해야 한다고 강조합니다.



### Artificial Intelligence for Infectious Disease Prediction and Prevention: A Comprehensive Review (https://arxiv.org/abs/2411.10486)
Comments:
          31 pages, 5 figures, this manuscript has been accepted for publication in ACTA UNIVERSITATIS SAPIENTIAE, Informatica

- **What's New**: Artificial Intelligence (AI)와 감염병 예측의 최근 발전을 다룬 연구로, 다양한 데이터 유형과 학습 방법, 그리고 그에 따른 문제점들을 분석하고 있습니다.

- **Technical Details**: 연구는 세 가지 주요 영역으로 나누어집니다: 1) 공공 건강 데이터(Public Health Data)를 이용한 전염병 확산 예측, 2) 환자의 의료 데이터(Patients' Medical Data)를 통한 감염 여부 탐지, 3) 공공 및 환자 데이터를 활용하여 인구 내 감염병 확산 정도 추정. 사용되는 알고리즘으로는 Naive Bayes (NB), Decision Tree (DT), Support Vector Machines (SVM), Long Short-Term Memory (LSTM) 모델 등이 포함되어 있습니다.

- **Performance Highlights**: 이 연구에서는 감염병의 예측 및 예방에 관한 다양한 연구 결과를 평가하며, 각 연구 목적을 달성하기 위해 사용된 데이터 유형과 학습 방법을 분석합니다.



### Hateful Meme Detection through Context-Sensitive Prompting and Fine-Grained Labeling (https://arxiv.org/abs/2411.10480)
Comments:
          AAAI-25 Student Abstract, Oral Presentation

- **What's New**: 이번 연구에서는 멀티모달(multi-modal) 콘텐츠의 자동 조정을 위한 최적화 개념 프레임워크를 제안합니다. 기존에 연구된 방법들과는 달리, 모델 성능 향상뿐만 아니라 모달리티(modality), 프롬프팅(prompting), 라벨링(labeling), 그리고 파인튜닝(fine-tuning)을 포괄하는 엔드투엔드(End-to-End) 최적화 파이프라인을 개발하였습니다.

- **Technical Details**: 제안된 프레임워크의 핵심 원리는 멀티변량 최적화(multi-variate optimization) 문제로, 모달리티, 프롬프팅, 라벨링, 파인튜닝의 성능을 고려합니다. 실험은 다양한 프롬프팅 및 라벨링 전략을 포함해 12개의 실험을 통해 진행되었고, 페이스북 증오 밈 데이터셋을 사용하여 ACCU 및 AUROC로 성과를 평가하였습니다.

- **Performance Highlights**: 모델 M은 68.933%의 정확도와 66.827%의 AUROC로 최고의 성능을 기록하여, 단순히 복잡한 모델이 아니라는 것을 강조합니다. 실험결과 파인튜닝, 범주형 프롬프팅, 이진 라벨링의 조합이 모델 성능 향상에 기여했으며, 독립적인 최적화가 반드시 최고의 결과를 보장하지 않음을 보여주었습니다.



### Large Language Models for Constructing and Optimizing Machine Learning Workflows: A Survey (https://arxiv.org/abs/2411.10478)
- **What's New**: 최근의 연구에서는 대규모 언어 모델(LLMs)이 머신러닝(ML) 워크플로우를 자동화하고 최적화하는 데 큰 가능성을 보여주고 있습니다. 이 설문조사는 LLM이 데이터 및 피처 엔지니어링, 모델 선택 및 하이퍼파라미터 최적화, 워크플로우 평가와 같은 주요 구성 요소에서 수행하는 역할을 중점적으로 다루고 있습니다.

- **Technical Details**: 이 논문은 자동화된 머신러닝(AutoML)과 LLMs의 통합에 대한 현재 연구 동향을 포괄적으로 조사합니다. LLM은 복잡한 ML 작업을 처리할 수 있는 잠재력을 가지며, 대화형 에이전트로서의 기능을 통해 사용자 제공 정보를 바탕으로 AutoML 코드를 생성하고 조정할 수 있습니다. 또한, 하이퍼파라미터 최적화 과정에서 LLM은 과거 데이터와 도메인 지식을 바탕으로 최적의 설정을 예측하여 모델 성능을 향상시킵니다.

- **Performance Highlights**: LLM 기반 접근 방식은 ML 워크플로우 모델링 과정을 간소화하고 개선하는 데 중요한 역할을 합니다. 그러나 대규모 모델 배포에 대한 계산 요구 사항, 윤리적 문제 및 추론 환각과 같은 도전 과제가 존재합니다. LLMs는 ML 워크플로우의 모든 단계에서 특히 모델 선택과 평가에서 시사용한 성능 개선의 가능성이 기대됩니다.



### Beyond object identification: How train drivers evaluate the risk of collision (https://arxiv.org/abs/2411.10475)
- **What's New**: 이 연구는 기차 운전자가 충돌 위험을 평가할 때 고려하는 요소들을 밝혀내고, 이를 통해 인공지능(AI)이 충돌을 예방하는 데 어떻게 기여할 수 있을지를 고찰합니다. 33명의 기차 운전자를 대상으로 한 이미지 기반 전문가 인터뷰를 통해, 운전자가 장애물 위험을 평가하는 방법과 그에 대한 설명을 수집했습니다. 이 연구는 운전자의 사고 과정이 AI 시스템 개발에 중요한 통찰을 제공할 것임을 시사합니다.

- **Technical Details**: 연구에서는 기차 운전자가 장애물과 그 상황을 인식할 때 어떤 단서(cues)를 사용하고, 어떤 추론(inferences)을 하는지를 평가했습니다. 운전자는 잠재적 장애물이 포함된 이미지를 보고 충돌 위험을 평가했으며, 상황이 어떻게 변화해야 위험이 감소하거나 증가하는지를 설명했습니다. 이러한 과정에서 도출된 개념들은 사람들의 정체성(people's identity), 장소(location), 움직임(movement), 행동(action), 물리적 특징(physical features) 및 정신 상태(mental states) 등 다양한 범주로 분류되었습니다.

- **Performance Highlights**: 연구 결과, 기차 운전자는 특히 사람에 대한 행동 및 정신 상태를 고려하여 논리적 추론을 수행하며, 상황에 따라 이러한 추론이 체계적으로 다르게 나타난다는 것을 발견했습니다. 이 연구는 인간의 위험 평가 과정 이해가 기차 운전 및 운전 자동화의 안전성을 높이는 데 매우 중요함을 강조합니다. 궁극적으로 AI 기반 시스템은 기차 운전자의 사고 방식을 모델링하여 충돌을 효과적으로 예방할 수 있을 것으로 기대됩니다.



### PhDGPT: Introducing a psychometric and linguistic dataset about how large language models perceive graduate students and professors in psychology (https://arxiv.org/abs/2411.10473)
Comments:
          20 pages, 8 figures. Edoardo Sebastiano De Duro and Enrique Taietta equally contributed to this work

- **What's New**: 이 연구는 PhDGPT라는 새로운 프롬프트 프레임워크와 합성 데이터셋을 소개합니다. 이 데이터셋은 OpenAI의 GPT-3.5가 인식한 PhD 연구자와 교수들의 머신 심리를 포착하고 있습니다. 756,000개의 데이터 포인트로 구성된 이 연구는 우울, 불안 및 스트레스 척도(DASS-42)를 포함하여 학업 및 직업적 맥락에서 심리적 통찰력을 제공합니다.

- **Technical Details**: 이 연구는 15개의 학술 이벤트와 두 개의 직업 수준에 걸쳐 반복된 300회의 실험을 기반으로 합니다. 심리측정 점수와 텍스트 설명을 결합하여 학술적인 인물의 정서적 웰빙을 평가합니다. 연구는 네트워크 심리 측정(methodology combining network psychometrics)와 심리 언어학을 결합하여 LLM의 데이터와 인간의 데이터 간의 차이점을 탐구합니다.

- **Performance Highlights**: 연구 결과, LLM이 인간의 DASS 요인을 80%의 순도로 재구성할 수 있음이 밝혀졌습니다. 또한 LLM이 맥락적 자극에 따라 언어 패턴을 변화시킬 수 있는 능력을 보여줌으로써, LLM의 머신 심리를 평가할 수 있는 새로운 정량적 기회를 제공합니다. 이 연구는 머신 심리학 분야 및 인접 분야에서의 향후 연구를 위한 새로운 데이터셋인 PhDGPT의 중요성을 강조합니다.



### Detecting Student Disengagement in Online Classes Using Deep Learning: A Review (https://arxiv.org/abs/2411.10464)
- **What's New**: 학생들이 온라인 학습에서 disengagement(탈퇴)하는 문제는 특히 팬데믹 이후 심각한 도전 과제가 되었습니다. 이 리뷰에서는 disengagement를 감지하기 위해 사용되는 deep learning(딥 러닝) 기술을 탐구하며, computer vision(컴퓨터 비전)과 affective computing(정서 컴퓨팅) 기법이 효과적인 접근법으로 강조됩니다.

- **Technical Details**: 본 연구에서는 학생의 주의를 평가하기 위해 최근 연구들이 다룬 얼굴 표정, 눈의 움직임, 자세와 같은 지표들을 살펴봅니다. 또한 마우스 활동과 같은 비면접(computer-based non-facial) 지표와 같은 새로운 접근법도 포함됩니다. 총 38개의 연구를 체계적으로 리뷰하여 사용된 지표, 방법, 그리고 모델에 대한 통찰을 제공합니다.

- **Performance Highlights**: 이 리뷰는 온라인 교실에서의 실시간 engagement(참여도) 모니터링을 위한 향후 연구를 위한 통찰을 제공합니다. 다양한 기술들의 효용성을 비교하고, 새로운 지표를 도입하여 학습 경험을 개선할 수 있는 방법을 논의합니다.



### Unexploited Information Value in Human-AI Collaboration (https://arxiv.org/abs/2411.10463)
- **What's New**: 이 논문은 인간과 AI의 협업에서 결정 관련 정보를 활용하여 성과를 향상시키기 위한 새로운 모델을 제안합니다. 기존의 결정 이론(statistical decision theory)에 기반하여, AI와 인간 각자가 사용하는 정보를 분석하고, 이들 간의 협력 시 각각의 성과를 어떻게 증대시킬 수 있는지를 탐구합니다. 특히, 딥페이크 감지 과제를 통해 정보의 활용 가치를 평가하여 AI의 도움으로 인간 결정이 개선될 수 있는 방안을 모색합니다.

- **Technical Details**: 연구자는 인간- AI 협업에서 정보를 활용하여 결정을 개선할 수 있는 방법론을 제시합니다. 의사 결정 문제는 세 가지 주요 요소로 구성되어 있으며, 여기에는 보상 관련 상태, 의사 결정자의 선택, 그리고 주어진 상태에 대한 결정 품질을 평가하기 위한 보상 함수가 포함됩니다. 정보 구조는 AI가 사용할 수 있는 기본 신호들을 정의하고, 이를 통해 각각의 정보 요소가 인간 결정에 얼마나 기여할 수 있는지를 정량화하는 방법을 설명합니다.

- **Performance Highlights**: 실험 결과, 인간 참가자들은 AI가 효과적으로 활용한 몇 가지 신호의 상당한 정보를 제대로 활용하지 못한 것으로 나타났습니다. AI 예측 결과를 단순히 표시하는 것만으로는 인간의 정보 활용에 도움이 되지 않을 수 있으며, 이는 AI의 결정 규칙에 대한 설명이 필요하다는 점을 시사합니다. 결론적으로, 이 연구는 인간-AI 팀의 성과를 개선하기 위한 새로운 접근 방식을 제공하며, 정보 활용의 잠재력을 강조합니다.



### Utilizing Human Behavior Modeling to Manipulate Explanations in AI-Assisted Decision Making: The Good, the Bad, and the Scary (https://arxiv.org/abs/2411.10461)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 AI 추천과 설명을 사람의 결정 과정에 통합하는 방식을 정량적으로 모델링하는 연구에 대해 다룹니다. 연구자들은 AI 기반 의사 결정 지원 도구가 인간의 결정에 미치는 영향을 이해하고, 이러한 이해를 통해 AI 설명을 조작하여 대상결정을 유도할 수 있는지 탐구합니다. 그 결과, 변형된 설명이 인간의 의사 결정에 미치는 영향을 실험을 통해 입증하고, 의도와 관계없이 특정 목표를 향한 결정 유도 가능성을 보여주었습니다.

- **Technical Details**: 이 연구에서는 n차원 특징 벡터(x)와 이와 관련된 올바른 결정(y)에 대한 이진 선택 과업을 고려합니다. AI 모델의 추천(yᵐ = M(x) )은 설명 가능한 AI 방법론인 LIME이나 SHAP을 통해 결정 설명(e)를 제공할 수 있으며, 개인의 최종 결정을 안내하는 데 사용됩니다. 연구진은 이러한 입력을 바탕으로 인간의 결정 메커니즘을 수치적으로 모델링하여 행위 데이터를 분석하고 설명을 최적화하는 과정에 대해 설명합니다.

- **Performance Highlights**: 실험 결과, 조작된 설명은 인간-AI 팀의 의사 결정 성능을 크게 향상시킬 수 있으며, 악의적 목적으로도 활용될 수 있음을 보여줍니다. 특히 사람들은 설명의 조작에 대해 인지하지 못하는 경향이 있으며, 이로 인해 인간-AI 상호작용의 취약성이 드러납니다. 연구는 행동 모델링이 이렇듯 장점과 위험을 모두 내포하고 있음을 강조하며, 이를 통해 인간-AI 의사 결정 절차의 안전성과 신뢰성을 높일 필요성을 제기합니다.



### Biotic Browser: Applying StreamingLLM as a Persistent Web Browsing Co-Pilo (https://arxiv.org/abs/2411.10454)
Comments:
          Written December 2023

- **What's New**: 이 논문은 웹 탐색과 작업 실행을 혁신적으로 전환하는 AI 도우미인 'Biotic Browser'를 소개합니다. Biotic Browser는 StreamingLLM을 활용하여 자율주행차의 탑승자 경험을 시뮬레이션하며, 장기적인 맥락 관리에서 중요한 발전을 이루었습니다. 또한, 개인 및 직업 환경에서 생산성과 효율성을 높일 가능성을 보여줍니다.

- **Technical Details**: Biotic Browser는 사용자가 정의한 목표를 캡처한 후 현재 웹페이지의 스크린샷을 가져와서 인터랙티브한 요소를 식별하고 번호를 매기는 과정을 거칩니다. 이후 이 정보를 StreamingLLM에게 전달하여 작업 완료 여부를 확인하거나 다음 작업을 제안합니다. 새로운 구조에서는 DOM을 스크랩하여 인터랙트 가능한 요소를 식별하며, 비주얼 데이터를 전송하지 않고도 긴 작업 시퀀스를 관리할 수 있습니다.

- **Performance Highlights**: StreamingLLM의 통합은 메모리 할당 제약이라는 도전 과제를 안고 있었습니다. 하지만 Google Colab 활용을 통해 성능을 크게 개선하였으며, 비주얼 데이터 처리에서 텍스트 기반 입력으로의 전환이 이루어졌습니다. 이 과정은 Biotic Browser가 개인적 및 전문적_settings에서 더 나은 작업 관리를 제공할 수 있는 잠재력을 강조하고 있습니다.



### Towards Geometry-Preserving Reductions Between Constraint Satisfaction Problems (and other problems in NP) (https://arxiv.org/abs/2411.10453)
Comments:
          In Proceedings FROM 2024, arXiv:2410.23020. An extended version is under preparation and will also be posted on arXiv

- **What's New**: 이 연구에서는 조합 최적화 문제에서의 phase transitions에 의해 영감을 받아, constraint satisfaction 문제와 다른 NP-search 문제 간의 geometry-preserving reductions를 정의합니다. 이러한 접근 방식은 기존의 문제 해결 전략에 새로운 통찰력을 제공합니다.

- **Technical Details**: 논문에서는 두 가지 종류의 geometry-preserving reductions를 소개하며, 이는 각기 다른 NP-search 문제에 적용될 수 있습니다. 또한, 특정 예시와 반례(counterexample)를 통해 이러한 reductions의 개념을 더욱 명확히 설명합니다.

- **Performance Highlights**: 이 개선된 이론적 프레임워크는 다양한 NP-search 문제에서의 문제 해결 방식을 최적화할 수 있는 가능성을 보여줍니다. 실질적으로 이러한 reductions는 더 나은 성능을 발휘할 수 있는 길을 제시합니다.



### Dataset Refinement for Improving the Generalization Ability of the EEG Decoding Mod (https://arxiv.org/abs/2411.10450)
Comments:
          4 pages, 1 figure, conference

- **What's New**: 이 논문은 EEG (Electroencephalography) 신호에서 인간의 의도를 디코딩하기 위한 데이터셋 리파인먼트 (dataset refinement) 알고리즘을 제안합니다. 기존 연구들이 데이터셋의 질이 이상적인 것으로 가정하고 있었으나, 실제 EEG 데이터셋은 잡음 (noise)에 취약하기 때문에 데이터 정제 과정이 필요하다고 주장합니다.

- **Technical Details**: 제안된 알고리즘은 데이터의 영향을 평가하는 다양한 메트릭 (metrics)을 사용하여 훈련 과정에서 각 데이터의 영향력을 측정합니다. 이 영향력 점수를 기반으로 잡음이 포함된 데이터를 제거하여 모델의 일반화 성능을 향상시킵니다. influence score와 Monte Carlo dropout (MC dropout) 방법을 이용하여 데이터의 불확실성을 측정했습니다.

- **Performance Highlights**: 리파인먼트 된 데이터셋을 사용하여 모델을 재훈련한 결과, 원본 데이터셋을 사용할 때보다 일관되게 더 나은 일반화 성능이 나타났습니다. 이는 EEG 도메인에서 잡음 데이터를 제거하는 것이 딥러닝 모델의 성능을 유의미하게 향상시킬 수 있음을 보여줍니다.



### Love in Action: Gamifying Public Video Cameras for Fostering Social Relationships in Real World (https://arxiv.org/abs/2411.10449)
Comments:
          accepted as a main track paper by EAI-ArtsIT 2024

- **What's New**: 이 논문에서는 'Love in Action'(LIA)이라는 신체 언어 기반의 사회적 게임을 제안합니다. 이 게임은 공공장소에 설치된 비디오 카메라를 활용하여 실제 사회적 관계를 증진하는 것을 목표로 합니다. 참가자들은 요청자와 수행자라는 두 가지 역할을 맡고, AI를 활용한 비디오 분석 시스템을 통해 수행자의 신체 언어 품질을 평가합니다. 이를 통해 LIA가 사회적 상호작용을 촉진하는 혁신적인 매개체가 될 수 있음을 보여줍니다.

- **Technical Details**: LIA는 주요 인공지능 기술을 활용하여 신체 언어 기반의 상호작용을 게임화한 소셜 LBG입니다. 요청자는 특정 신체 언어에 대한 요청을 하고, 수행자는 비디오 카메라 앞에서 요청에 부응하는 행동을 수행합니다. AI-enhanced video analysis system(인공지능 강화 비디오 분석 시스템)이 이 상호작용을 조정하며, WeChat mini-program을 통해 사용자 경험을 지원합니다. 이 연구는 기술 기관 캠퍼스에서 진행된 2주간의 필드 스터디를 통해 LIA의 영향을 검토하였습니다.

- **Performance Highlights**: 27명의 참가자를 대상으로 한 연구에서 LIA는 인지된 사회적 관계의 향상을 나타내었습니다. 연구 결과는 참가자들이 LIA의 게임 경험을 즐겼고, 공공 카메라를 통한 새로운 상호작용 스타일이 긍정적인 영향을 미쳤음을 보여줍니다. 신체 언어를 퍼블릭 카메라 앞에서 수행하는 경험은 참가자들의 사회적 및 정신적 웰빙에 기여할 수 있는 아이디어를 제시합니다. 이러한 결과는 LIA가 실제 사회적 관계를 증진시키는 효과적인 수단이 될 수 있음을 뒷받침합니다.



### Goetterfunke: Creativity in Machinae Sapiens. About the Qualitative Shift in Generative AI with a Focus of Text-To-Imag (https://arxiv.org/abs/2411.10448)
Comments:
          3 figures (images), 33 pages

- **What's New**: 2022년은 기술의 전환점(watershed)으로, 강력한 생성적 AI(generative AI)가 등장하여 창의적 작업을 효과적으로 수행할 수 있게 되었습니다. 이러한 시스템 덕분에 누구나 이전에는 주목받지 못했던 예술 작품을 창조할 수 있게 되었습니다. AI와의 협업에서 컴퓨터는 단순한 도구 이상의 존재가 되어가고 있으며, 이는 '창의성 기계(creativity machines)'라는 개념으로 표현되고 있습니다.

- **Technical Details**: 이 논문은 현재의 머신 러닝(paradigm) 내에서 컴퓨터의 창의성(possibility of creativity) 가능성에 대해 다루고 있습니다. 텍스트-투-이미지 시스템(text-to-image systems)을 주제로 하여 기술의 핵심 개념과 혁신이 이러한 질적 변화에 기여한 방식을 설명합니다. 인공지능 창의성(Artificial Creativity)의 본질과 예술에 대한 의미도 논의됩니다.

- **Performance Highlights**: AI는 예술적 과정에서 독립적인 기계 저자(machine authorship)의 요소를 가진 책임 있는 협력자(responsible collaborator)로 발전할 가능성이 있습니다. 이러한 변화는 예술 창작의 방식을 혁신적으로 변화시킬 수 있으며, 이는 많은 예술가와 창작자들에게 새로운 기회를 제공합니다.



### OpenLS-DGF: An Adaptive Open-Source Dataset Generation Framework for Machine Learning Tasks in Logic Synthesis (https://arxiv.org/abs/2411.09422)
Comments:
          14 pages

- **What's New**: 이 논문은 OpenLS-DGF라는 적응형(logic synthesis) 데이터셋 생성 프레임워크를 소개합니다. 이전의 데이터셋 생성 프로세스는 특정 작업에 맞춰져 있거나 머신러닝(ML) 기능이 통합되어 있지 않았습니다. OpenLS-DGF는 Boolean 표현, 논리 최적화(logic optimization), 기술 매핑(technology mapping)의 세 가지 기본 단계를 포함하여 다양한 ML 작업을 지원합니다.

- **Technical Details**: OpenLS-DGF는 Verilog와 머신러닝에 적합한 GraphML 형식으로 원본 정보를 보존합니다. Verilog 파일은 반맞춤형(semi-customizable) 기능을 제공하여 연구자가 추가 단계를 삽입하고 생성된 데이터셋을 점진적으로 개선할 수 있도록 합니다. 또한 적응형 회로 엔진(adaptive circuit engine)을 포함하여 최종 데이터셋 관리와 하류 작업(downstream tasks)을 용이하게 합니다.

- **Performance Highlights**: 생성된 OpenLS-D-v1 데이터셋은 46개 조합(combinational) 설계를 포함하고 있으며, 총 966,000개 이상의 Boolean 회로를 포함합니다. OpenLS-D-v1은 새로운 데이터 특징을 통합할 수 있어 새로운 도전에 더 적합합니다. 논문에서는 회로 분류(circuit classification), 회로 순위 매기기(circuit ranking), 결과 품질 예측(QoR prediction), 확률 예측(probability prediction)의 네 가지 하류 작업을 통해 OpenLS-D-v1의 다양성과 응용 가능성을 보여줍니다.



### Towards Operationalizing Right to Data Protection (https://arxiv.org/abs/2411.08506)
Comments:
          First two authors contributed equally to this work

- **What's New**: 본 논문에서는 자연어 데이터셋에 불가시한(spurious) 상관관계를 주입하여 데이터셋을 비학습(unlearnable) 가능하게 만드는 RegText라는 새로운 프레임워크를 소개합니다.

- **Technical Details**: RegText는 데이터에서 의미론적(content) 내용을 손상시키지 않으면서 불가시한 노이즈를 추가하여, 언어 모델이 해당 데이터에서 학습하는 것을 제한합니다. 이는 기존 이미지 중심의 접근 방식의 한계를 넘어서는 혁신적인 방법입니다.

- **Performance Highlights**: RegText는 GPT-4o 및 Llama와 같은 최신 모델이 생성된 데이터에서 학습하는 것을 제한하며, 이로 인해 테스트 정확도가 낮아지는 결과를 보입니다. 이로써 공공 데이터 보호를 위한 비학습 텍스트 생성의 가능성을 제시합니다.




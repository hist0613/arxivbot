New uploads on arXiv(cs.CL)

### The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models (https://arxiv.org/abs/2408.07702)
- **What's New**: 이 논문에서는 최신 대형 언어 모델(LLM)을 이용한 Text-to-SQL 작업에서 스키마 링크(schema linking)의 필요성을 재조명하고, 스키마 링크 없이도 정확한 SQL 쿼리를 생성할 수 있음을 보여줍니다.

- **Technical Details**: 기존 Text-to-SQL 파이프라인에서는 스키마 링크가 중요한 단계로, 관련 테이블과 열(signal)만을 추출하는 것이 필수적입니다. 하지만 새로운 모델들은 노이즈(noise) 속에서도 관련 스키마 요소를 식별할 수 있는 능력을 갖추고 있어, 스키마 링크 없이 전체 데이터베이스 스키마를 모델에 전달하는 방법을 제안합니다. 이 방법은 실행 정확도(execution accuracy)에서 BIRD 벤치마크 기준으로 71.83%를 달성했습니다.

- **Performance Highlights**: 이 연구는 새로운 LLM의 개선된 추론 능력이 스키마 링크 없이도 정확한 쿼리 생성을 가능하게 한다는 점을 강조하며, BIRD 벤치마크에서 71.83%의 실행 정확도를 기록하여 제출 당시 1위를 차지했습니다.



### Enhanced Detection of Conversational Mental Manipulation Through Advanced Prompting Techniques (https://arxiv.org/abs/2408.07676)
Comments:
          Accepted at WiNLP @ EMNLP 2024

- **What's New**: 이 연구는 다양한 prompting 기법들이 대화식 정신 조작(mental manipulation) 탐지에서의 효과성을 탐구하는 장기 프로젝트를 제시합니다. Chain-of-Thought prompting을 Zero-Shot 및 Few-Shot 설정에서 구현하여 이진 정신 조작 탐지 작업을 수행하였습니다.

- **Technical Details**: 정신 조작은 선호와 선택에 대한 심리적 영향을 미치는 미묘한 형태로, 이는 자연어 처리(Natural Language Processing, NLP) 분야에서 탐지가 상당히 어렵습니다. 연구에서는 MentalManipCon 데이터셋을 활용해 두 개의 LLM(GPT-3.5 및 GPT-4o)에서 정신 조작을 탐지하는 다양한 prompting 전략의 효과를 평가하였습니다. 분석한 전략은 Zero-Shot, Few-Shot, CoT(Chain-of-Thought)입니다.

- **Performance Highlights**: Few-Shot CoT가 GPT-3.5에서 0.722, GPT-4o에서 0.778의 Accuracy를 기록하며 최고의 성능을 보였습니다. 반면, GPT-4o에서 Zero-Shot CoT는 성능이 가장 낮았으며, 거짓 긍정(False Positive) 사례가 많아 Macro F1 점수 또한 가장 낮았습니다. 이는 CoT가 더 단순한 기법에 비해 복잡한 모델에서는 예제를 통한 학습이 필요하다는 것을 시사합니다.



### Spoken Stereoset: On Evaluating Social Bias Toward Speaker in Speech Large Language Models (https://arxiv.org/abs/2408.07665)
- **What's New**: 이 연구는 Spoken Stereoset이라는 데이터를 소개하여, Speech Large Language Models (SLLMs)에서 사회적 편견을 평가하는 첫 번째 데이터셋을 제공합니다.

- **Technical Details**: Spoken Stereoset은 17명의 화자와 3640개의 테스트 인스턴스로 구성되어 있습니다. 이 데이터셋은 화자의 성별과 나이에 따른 편견을 평가하기 위해 설계되었으며, SLLM이 같은 문장에 대해 화자의 인구통계적 속성에 따라 어떻게 다르게 반응하는지를 분석합니다.

- **Performance Highlights**: 대부분의 SLLMs가 최소한의 편견을 보이는 반면, 일부 모델은 약간의 스테레오타입적 또는 반-스테레오타입적 경향을 나타냅니다. 특히, 텍스트 기반 LLM들은 화자 정보가 주어지지 않았을 때 공정성을 입증했습니다.



### Alignment-Enhanced Decoding:Defending via Token-Level Adaptive Refining of Probability Distributions (https://arxiv.org/abs/2408.07663)
Comments:
          15 pages, 5 figures

- **What's New**: 본 논문에서는 Jailbreak 공격을 방지하기 위해 Alignment-Enhanced Decoding (AED)라는 새롭고 혁신적인 방어 메커니즘을 제안합니다. 기존의 방어 방법들이 입력을 수정하거나 검사를 통해 공격을 방어하는 것과는 달리, AED는 경쟁하는 목표를 고려하여 모델의 정렬 실패의 근본 원인을 해결합니다.

- **Technical Details**: AED는 Competitive Index를 정의하여 모델의 정렬 실패를 정량화하고, 자기 평가 피드백을 통해 post-alignment logits를 계산합니다. AED는 원래의 logits와 post-alignment logits을 결합하여 무해하고 유용한 분포를 생성하는 방식으로 동작합니다. 이는 추가 학습 없이도 각 단계의 디코딩 과정이 무해한 목표를 준수하도록 합니다.

- **Performance Highlights**: AED는 총 다섯 개의 인기 있는 대형 언어 모델에서 실험을 수행하였으며, GCG, AutoDan, ICD, Refusal_Suppression과 같은 다양한 고급 jailbreak 공격을 효과적으로 방어하였습니다. 또한 무해한 데이터셋에서 일반적인 문의에 대해서도 유용함을 유지하는 것으로 나타났습니다.



### WeKnow-RAG: An Adaptive Approach for Retrieval-Augmented Generation Integrating Web Search and Knowledge Graphs (https://arxiv.org/abs/2408.07611)
Comments:
          8 pages, 2 figures, technical report for 3rd place in Task 3 of Meta KDD Cup 2024 CRAG Challenge

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 신뢰성을 향상시키기 위해 WeKnow-RAG라는 새로운 접근 방식을 제안합니다. 이 시스템은 웹 검색과 지식 그래프를 통합하여 사실 기반 정보를 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: WeKnow-RAG는 'Retrieval-Augmented Generation (RAG)' 시스템을 기반으로 하여 지식 그래프의 구조적인 표현과 밀집 벡터 검색의 유연성을 결합합니다. 이 시스템은 특정 도메인에 맞는 지식 그래프를 활용하여 다양한 쿼리를 처리합니다.

- **Performance Highlights**: 결과적으로 이 접근 방식은 정보 검색의 효율성과 정확성을 효과적으로 균형 잡아주며, 다양한 오프라인 실험과 온라인 제출에서 뛰어난 효과성을 입증했습니다.



### Assessing the Role of Lexical Semantics in Cross-lingual Transfer through Controlled Manipulations (https://arxiv.org/abs/2408.07599)
- **What's New**: 이번 연구는 교차 언어 전이 과정에서 어휘 의미론(lexical semantics)의 역할을 평가하며, 이와 함께 다른 언어 특성의 영향을 비교 분석합니다. 기존 연구와의 차별점으로는 어휘화 패턴(lexicalization patterns)에서의 차이를 조작하여 교차 언어 전이에 미치는 영향을 분석한 것입니다.

- **Technical Details**: 연구에서는 영어로부터 저자원이 필요한 목표 언어(target language)의 언어 특성을 반영한 인공 언어(artificial language)를 생성하였습니다. 이를 통해, 두 언어의 어휘 매칭(degree of lexical matching)을 측정하기 위해 번역 엔트로피(translation entropy) 지표를 사용하여, 각 언어 속성이 교차 언어 전이에 미치는 영향을 체계적으로 분석하였습니다.

- **Performance Highlights**: 연구 결과, 어휘화 패턴이 교차 언어 전이에 미치는 영향이 가장 크며, 이는 다른 언어적 특성이 미치는 영향보다 두드러집니다. 특히, 양 언어 간의 단어 엔트로피(entropy)의 상관관계가 제로샷(zero-shot) 성능에 강한 연관성을 보여줍니다.



### Large Language Models Know What Makes Exemplary Contexts (https://arxiv.org/abs/2408.07505)
Comments:
          12 pages, 3 figures

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 인맥 학습(in-context learning, ICL)을 개선하기 위한 통합 프레임워크를 제안합니다. 이는 LLM이 영향을 미치는 예시를 스스로 선택하고, 후보를 자가 평가하며, 보강 학습을 통해 예시 선택과 순서를 최적화하는 방법입니다.

- **Technical Details**: 제안된 방법은 LLM의 피드백을 기반으로 최적화된 예시를 생성하는 매개변수 효율적인 retrieval head를 설계합니다. 이는 단일 모델이 retrieval과 inference 모두를 처리하며, 여러 흐름에서 LLM의 선호도를 기반으로 학습됩니다. 또한, 다양한 예시를 포함한 최적화된 순서를 제공합니다.

- **Performance Highlights**: 실험 결과 제안한 방법이 ICL 성능을 향상시키는 데 효과적임을 입증했습니다. LLM이 현재 작업에 가장 대표적인 예시를 식별하고 선택하는 데 있어 많은 다양성을 포함합니다.



### A Study on Bias Detection and Classification in Natural Language Processing (https://arxiv.org/abs/2408.07479)
Comments:
          31 pages, 15 Tables, 4 Figures

- **What's New**: 이 논문은 공공 데이터셋을 수집하고 이를 효과적으로 결합하여 증오 발언(hate speech) 탐지 및 분류 모델을 훈련하는 방법을 제시합니다. 또한 데이터셋의 부족, 자원 편향, 비지속적인 데이터 의존성과 같은 주요 문제를 분석합니다.

- **Technical Details**: 저자들은 여러 종류의 분류기를 훈련시켜 모델의 성능을 분석하고, 데이터셋 간의 조합이 모델 성능에 미치는 영향을 보여줍니다. 또한 성별, 인종, 직업, 종교, 장애, 성적 지향, 성 정체성, 국적, 나이와 같은 사회적 특성을 기반으로 한 '타겟 카테고리'를 정의합니다.

- **Performance Highlights**: 다양한 데이터셋의 조합이 모델 성능에 미치는 직접적 영향에 대한 결과가 도출되었으며, 특정 특성과 관련된 비편향적 데이터셋의 필요성이 강조되었습니다.



### Bridging and Modeling Correlations in Pairwise Data for Direct Preference Optimization (https://arxiv.org/abs/2408.07471)
Comments:
          18 pages, 8 figures, 8 tables, working in progress

- **What's New**: 이번 연구에서는 pairwise preference data에서의 상관관계를 모델링하는 BMC라는 새로운 프레임워크를 제안합니다. 없음.

- **Technical Details**: BMC는 두 주요 단계로 구성되어 있습니다: Bridging Phase에서 pairwise preference 신호의 일관성과 정보성을 높이고, Modeling Phase에서는 정책 모델의 신뢰도를 활용하여 토큰 레벨 상관관계를 동적으로 모델링합니다.

- **Performance Highlights**: 실험 결과, BMC는 QA, 수학적 추론, 지침 따르기 작업 등 여러 다운스트림 시나리오에서 DPO와 같은 기존 최적화 알고리즘보다 최대 3.8 점에서 6.4 점까지 성능이 우수한 것으로 나타났습니다.



### Large Language Models Prompting With Episodic Memory (https://arxiv.org/abs/2408.07465)
- **What's New**: POEM(PrOmpting with Episodic Memory)은 간단하고 효율적인 프롬프트 최적화 기법으로, 에피소드 기반 메모리를 활용하여 강화학습(RL) 문제로 프롬프트 최적화를 접근합니다. 이 방법은 기존의 자원 집약적이거나 성능이 부족한 프롬프트 최적화 방법들의 한계를 극복합니다.

- **Technical Details**: POEM은 에피소드 메모리를 사용하여 입력 데이터의 조합, 몇 개의 샷 예제의 순열 및 훈련 과정에서 관찰된 보상을 기록합니다. 검증 단계에서, 상위 k개의 가장 유사한 훈련 예제에서 발생한 총 보상을 기준으로 테스트 쿼리에 대한 예제 순서를 최적화합니다.

- **Performance Highlights**: 여러 텍스트 분류 작업에서 POEM은 TEMPERA와 RLPrompt보다 각각 5.3% 및 13.4% 더 뛰어난 성능을 보였습니다. POEM은 일반적인 휴리스틱 방법보다 모든 테스트에 걸쳐 일관된 우수한 성능을 보여주며, 다양한 언어 이해 작업에 잘 적응합니다.



### From Brazilian Portuguese to European Portugues (https://arxiv.org/abs/2408.07457)
Comments:
          12 pages, 8 tables

- **What's New**: 이 논문에서는 브라질 포르투갈어(Brazilian Portuguese, BP)와 유럽 포르투갈어(European Portuguese, EP) 간의 번역 서비스를 향상시키기 위한 신경망 기반의 번역 시스템 개발을 제안합니다. 이를 위해 TED Talks와 영화 자막에서 수집한 병렬 데이터를 사용하여 여러 모델을 세밀하게 조정( fine-tuning)하였습니다.

- **Technical Details**: BP와 EP 간 번역의 정확성을 검증하기 위해 500개의 문장으로 구성된 금 세트( gold test set)를 만들었습니다. 실험에는 mBART-large-50 및 ChatGPT 3.5 Turbo와 같은 최신 대규모 언어 모델이 포함되었습니다. 평가에서는 자동 지표와 인간 평가를 모두 활용하여 번역 모델의 성능을 검토했습니다.

- **Performance Highlights**: mBART-large-50 모델이 TED Talks 및 자막의 혼합 데이터로 조정(fine-tuned)된 경우 최상의 성능을 보였지만, ChatGPT 3.5 Turbo에 비해 약간 뒤처졌습니다. 또한 생성된 금 세트는 향후 모델 비교를 위한 귀중한 자료로 활용될 수 있습니다.



### Fact or Fiction? Improving Fact Verification with Knowledge Graphs through Simplified Subgraph Retrievals (https://arxiv.org/abs/2408.07453)
Comments:
          10 pages, 3 figures, appendix

- **What's New**: 이번 연구에서는 증가하는 허위 정보에 대응하기 위해, FactKG 데이터셋을 활용한 검증 모델의 성능을 개선하는 다양한 효율적인 방법을 제안합니다. 특히, 구조화된 지식 그래프를 기반으로 한 증거 검색을 단순화하여 모델의 정확도를 높였습니다.

- **Technical Details**: 연구에서는 다양한 모델 아키텍처를 사용하여 과제를 다룹니다. 여기에는 BERT 모델을 사용하는 Textual Fine-tuning, 구조적 처리를 위해 QA-GNN(Graph Neural Network)을 활용하는 Hybrid Graph-Language Model, 추가적인 fine-tuning 없이 최신 LLM(Language Model)을 활용하는 LLM Prompting이 포함됩니다. 이 과정을 통해, 수학적 효율성을 높이고, 모델의 훈련 시간을 대폭 단축시켰습니다.

- **Performance Highlights**: 모델은 테스트 세트의 정확도를 77.65%에서 93.49%로 크게 향상시켰으며, 훈련 시간은 1.5~10시간으로 단축되었습니다. 이는 이전의 벤치마크 모델(2-3일 소요) 대비 상당한 개선을 나타냅니다.



### CMU's IWSLT 2024 Simultaneous Speech Translation System (https://arxiv.org/abs/2408.07452)
- **What's New**: CMU가 IWSLT 2024 Simultaneous Speech Translation (SST) 과제에 제출한 새로운 시스템을 소개합니다. 이 시스템은 영어 음성을 독일어 텍스트로 실시간으로 번역하는 데 중점을 두고 있으며, WavLM 스피치 인코더와 Llama2-7B-Base 모델을 사용하는 종단 간 (end-to-end) 음성-텍스트 시스템을 개발했습니다.

- **Technical Details**: 시스템은 세 가지 주요 구성 요소로 구성됩니다: 스피치 인코더, 모달리티 어댑터(modality adapter), LLM 디코더. 스피치 인코더로는 WavLM을 사용하고, 모달리티 어댑터는 길이 어댑터와 선형 변환 레이어로 구성됩니다. 모델 훈련은 두 단계로 나뉘며, 먼저 스피치와 텍스트 표현을 정렬한 후, 전체 모델을 미세 조정합니다.

- **Performance Highlights**: 우리 모델은 MuST-C-v2 tst-COMMON 데이터셋에서 오프라인 BLEU 점수 31.1을, 2초 지연 조건에서의 BLEU 점수 29.5를 달성했습니다.



### LiveFC: A System for Live Fact-Checking of Audio Streams (https://arxiv.org/abs/2408.07448)
Comments:
          Under Review, 11 pages

- **What's New**: 디지털 시대의 정보 확산이 빠르게 증가하면서 허위 정보와 잘못된 정보의 전파도 심각해지고 있습니다. 본 연구에서는 실시간으로 라이브 오디오 스트림을 사실 확인할 수 있는 도구인 LiveFC를 개발하였습니다.

- **Technical Details**: LiveFC는 6개의 주요 구성 요소로 이루어져 있습니다: 1) 실시간 데이터에 작동하는 전사 모듈, 2) 오디오 세그먼트의 화자 식별을 위한 다이어리제이션 모듈, 3) 실시간으로 확인이 필요한 주장을 식별하는 주장 탐지 모듈, 4) 주장 분해 및 주제 할당 모듈, 5) 웹 검색 및 이전 사실 확인에서 최신 증거를 검색하는 증거 검색 모듈, 6) 최신의 미세 조정된 자연어 추론(NLI) 모델을 사용하는 주장 검증 모듈입니다.

- **Performance Highlights**: LiveFC는 디지털 환경에서 사용자에게 실시간으로 잘못된 정보에 대한 검증된 사실을 제공하여 사회적 불안을 예방하고, 선거 토론 및 캠페인과 같은 중요한 실시간 이벤트에서의 사실 확인을 지원합니다. 실제 검증에서는 덴마크의 사실 확인 팀과 미국의 대통령 후보 토론에서 효과를 입증하였습니다.



### Exploring Retrieval Augmented Generation in Arabic (https://arxiv.org/abs/2408.07425)
- **What's New**: 최근 자연어 처리(NLP) 분야에서 Retrieval Augmented Generation(RAG) 기법이 주목받고 있으며, 본 논문은 아랍어에서의 RAG 적용 사례를 종합적으로 연구합니다.

- **Technical Details**: 이 연구는 아랍어 텍스트에 대한 RAG의 구현 및 평가를 중점적으로 다루며, 검색 단계에서 다양한 semantic embedding 모델과 생성 단계에서 여러 LLM(대형 언어 모델)을 탐구합니다. 단어와 구문 간의 방언 차이로 인해 발생할 수 있는 문제 또한 고려합니다.

- **Performance Highlights**: 결과적으로, 기존의 semantic embedding 모델과 LLM들이 아랍어 RAG 파이프라인 구축에 효과적으로 활용될 수 있음을 보여줍니다.



### Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing for Large Language Models (https://arxiv.org/abs/2408.07413)
- **What's New**: 이 연구에서 지식 편집(knowledge editing)의 한계, 특히 지속적인 편집(lifelong editing)에서의 실패 원인이 지식 표현의 '슈퍼포지션(superposition)' 개념과 밀접하게 관련이 있음을 입증했습니다. 이는 지식 편집을 손실 없이 수행하기 위한 이론적 기반을 마련합니다.

- **Technical Details**: 지식 편집을 위한 닫힌 형태의 해(closed-form solution)를 확장하여, 편집의 간섭 항(interference term)이 도입되고 이는 편집 과정에서 기존 지식이 망각되는 현상을 설명합니다. 이 간섭 항의 크기는 지식 표현 간의 직교성(orthogonality)에 의해 결정되며, 비직교적 상태에서는 손실 없는 편집이 불가능함을 보여줍니다.

- **Performance Highlights**: 연구진은 GPT-2, Llama-2, Pythia 등 다양한 대규모 언어 모델에서 지식 슈퍼포지션이 보편적이며, 높은 커토시스(kurtosis), 제로 평균(zero mean), 중비대형 분포(heavy-tailed distribution)를 가진다는 것을 발견했습니다. 이 결과는 더 큰 모델이 더 효율적으로 지식을 처리할 수 있는 이유를 설명합니다.



### Aquila2 Technical Repor (https://arxiv.org/abs/2408.07410)
- **What's New**: 이 논문은 7억, 34억, 70억의 파라미터 크기를 가진 Aquila2 시리즈의 다국어 모델을 소개합니다. 이 모델들은 HeuriMentor(HM)라는 혁신적인 프레임워크를 기반으로 훈련되며, 실시간으로 모델 수렴(convergence)에 대한 통찰력을 제공합니다. 또한, 훈련 과정과 데이터 관리를 향상시킵니다.

- **Technical Details**: Aquila2는 Adaptive Training Engine(ATE), Training State Monitor(TSM), Data Management Unit(DMU)으로 구성된 HM 시스템을 활용하여 모델의 훈련 진행 상황을 정밀하게 모니터링하고 데이터 분포를 최적화합니다. Grouped Query Attention(GQA) 메커니즘과 Rotary Position Embedding(RoPE) 기법도 적용되어 효율을 높이며, Mixed precision 학습에 bfloat16을 사용합니다.

- **Performance Highlights**: Aquila2-34B는 LLaMA-2-70B 및 다른 이중 언어 모델에 비해 평균적으로 우수한 성능을 보였으며, 4비트 양자화 시 성능 저하가 거의 없는 것으로 나타났습니다. AquilaChat2-34B는 주관적 및 객관적 평가에서 LLaMA-2-70B 대비 뛰어난 성능을 보여주었습니다.



### A Quantum-Inspired Analysis of Human Disambiguation Processes (https://arxiv.org/abs/2408.07402)
Comments:
          PhD thesis

- **What's New**: 본 논문에서는 자연어 처리(NLP)에서 발생하는 모호성을 해결하기 위해 양자 역학(quantum mechanics)의 개념을 활용했습니다. 기존의 NLP 접근 방식과 비교하여 실질적인 양자 이점을 탐구하는 연구가 이루어졌습니다.

- **Technical Details**: 논문에서는 양자 메커니즘에서 유래한 개념인 맥락성(contextuality)과 인과성(causality)을 적용하여 언어학에서의 모호성을 연구했습니다. 이러한 접근법은 심리언어학(psycholinguistics)에서 인간의 모호성 해소 과정과 관련된 결과들을 재현하는 데 기여했습니다.

- **Performance Highlights**: 이 연구의 결과는 인간 행동을 예측하는 데 사용되었으며, 기존의 NLP 메소드보다 더 우수한 성과를 보였습니다.



### DataVisT5: A Pre-trained Language Model for Jointly Understanding Text and Data Visualization (https://arxiv.org/abs/2408.07401)
- **What's New**: 본 논문에서는 DataVisT5라는 새로운 Pre-trained Language Model (PLM)을 소개합니다. 이는 자연어와 데이터 시각화(DV)의 상호 이해를 가능하게 하고, T5 아키텍처를 개선하기 위해 하이브리드 목적의 사전 훈련 및 다중 작업 미세 조정 전략을 채택했습니다.

- **Technical Details**: DataVisT5는 DL(Domain Language) 데이터셋과 텍스트 데이터셋을 통합하여 서로 다른 모달리티 간의 의미를 효과적으로 해석할 수 있도록 설계되었습니다. 주요 과제는 텍스트에서 DV로 변환(text-to-vis), DV에서 텍스트로 변환(vis-to-text), 데이터 시각화 관련 질문 응답(FeVisQA), 테이블에서 텍스트로 설명(table-to-text) 등을 포함합니다.

- **Performance Highlights**: Public datasets에 대한 광범위한 평가를 통해 DataVisT5가 다양한 DV 관련 작업에서 기존의 최첨단 모델(SOTA)을 지속적으로 초월하는 성능을 보여주었습니다.



### Do GPT Language Models Suffer From Split Personality Disorder? The Advent Of Substrate-Free Psychometrics (https://arxiv.org/abs/2408.07377)
Comments:
          37 pages, 7 figures, 3 tables, date v1: Mar 26 2023

- **What's New**: 이번 연구에서는 다양한 언어로 된 동일한 성격 질문지를 사용하여 언어 모델의 심리적 특성을 분석함으로써 이들 모델의 일관되지 않은 성격 개발을 확인하였습니다. 이 결과는 인공지능 시스템의 안전성에 대한 우려를 제기합니다.

- **Technical Details**: 연구진은 Bayesian 분석을 바탕으로 Gaussian Mixture Model을 적용하여 다국어로 측정된 성격 특성의 불안정성을 밝혀냈습니다. 특히, 현재의 대형 언어 모델(LLMs)은 일관성 있는 핵심 성격을 발전시키지 못한다는 점이 강조되었습니다.

- **Performance Highlights**: 대형 언어 모델인 GPT-3와 InstructGPT는 Dark Triad(나르시시즘, 정신병적 성향, 마키아벨리즘) 점수가 높았으며, Big5 성격 요인에서는 평균 이상의 점수를 보였습니다. 그러나 이들 모델에서 발생하는 어두운 성격 특성은 일반적인 긍정적인 성격과 함께 존재하여 그 안전성 문제를 더욱 부각시켰습니다.



### Only One Relation Possible? Modeling the Ambiguity in Event Temporal Relation Extraction (https://arxiv.org/abs/2408.07353)
- **What's New**: 이 논문은 Event Temporal Relation Extraction (ETRE)에서 시간 관계를 다루는 기존의 단일 레이블(classification) 방식에서 벗어나, 다중 레이블 멀티 이벤트 시간 관계 추출(METRE) 방법론을 제안하고 있습니다. 이는 'Vague'(모호한) 케이스를 다루는 새로운 접근법으로, 여러 가능성 있는 시간 관계를 독립적으로 추론합니다.

- **Technical Details**: 이 방법은 Vague를 여러 가지 잘 정의된 시간 관계가 존재하는 사건 쌍으로 간주하여, 각 관계의 가능성을 예측합니다. 실험은 TB-Dense, MATRES, UDS-T 데이터세트를 사용하여 진행되었으며, 이를 통해 각 시간 관계의 잠재 정보를 효과적으로 활용할 수 있습니다.

- **Performance Highlights**: METRE는 여러 최신 기법들을 초과하는 성능을 보였으며, Vague 인스턴스를 효과적으로 활용하여 특정 시간 관계의 인식을 향상시키고 있습니다. 실험 결과, 모델은 Vague를 예측하기보다는 더 높은 정확도로 잘 정의된 관계를 예측하는 경향이 있음을 보여주었습니다.



### Speech vs. Transcript: Does It Matter for Human Annotators in Speech Summarization? (https://arxiv.org/abs/2408.07277)
Comments:
          Accepted to ACL 2024 Main Conference

- **What's New**: 본 논문에서는 발화 기반 요약과 전사 기반 요약의 차이를 조사합니다. 음성 녹음을 듣고 요약을 작성하는 것과 텍스트 전사를 읽고 요약을 작성하는 방식 간의 차이를 연구하고 있으며, 이를 통해 요약의 질이 어떻게 다른지를 분석합니다.

- **Technical Details**: 연구팀은 1002개의 음성 녹음에서 두 가지 요약 방법에 대한 전문 annotator의 요약을 비교합니다. 평가 방법으로는 source-based evaluation, structure evaluation, summary comparisons 등이 사용되었으며, Automatic Speech Recognition (ASR) 기술을 통해 얻은 전사가 요약의 질에 미치는 영향을 조사합니다.

- **Performance Highlights**: 실험 결과, 발화 기반 요약은 정보 선택성과 사실적 일관성이 더 뛰어난 것으로 나타났으며, 전사 기반 요약은 인간 평가자와 대형 언어 모델 (LLM) 평가자에게 더 높은 선호를 받았습니다. 전문가 요약은 비전문가 요약보다 유창성과 일관성은 떨어지지만, 사실적 일관성은 유사한 수준을 유지합니다.



### Using Advanced LLMs to Enhance Smaller LLMs: An Interpretable Knowledge Distillation Approach (https://arxiv.org/abs/2408.07238)
- **What's New**: 이 논문은 더 작은 대형 언어 모델(LLM)의 성능을 향상시키기 위한 새로운 해석 가능한 지식 증류(knowledge distillation) 접근 방식을 제안합니다. 이 방법은 기업이 자가 호스팅할 수 있는 경제적인 모델의 성능을 높이도록 설계되었습니다.

- **Technical Details**: 전통적인 지식 증류 방식에서 '학생(student)' 모델이 '교사(teacher)' 모델의 응답을 통해 학습하는 대신, 우리의 방법은 '전략(strategy) 교육' 접근 방식을 활용합니다. 이 과정은 시나리오 생성(scenario generation) 단계와 개선 전략(strategies for improvement) 단계로 나뉘며, 학생 모델이 특정 상황에 적합한 응답을 생성하도록 돕습니다.

- **Performance Highlights**: 이 고객 서비스 응용 프로그램에서 이 방법은 성능을 향상시켰으며, 학습된 전략은 다른 LLM 및 훈련 집합을 넘어 다양한 시나리오에 적용할 수 있습니다. 실험 결과, 전략 교육이 다중 단계 생성에서 더 효과적임을 나타냈습니다.



### Neural embedding of beliefs reveals the role of relative dissonance in human decision-making (https://arxiv.org/abs/2408.07237)
Comments:
          26 pages, 6 figures, SI

- **What's New**: 이 연구에서는 수천 개의 믿음 사이의 미묘한 관계를 추출하는 새로운 방법을 제안하였습니다. 이를 위해 대규모 사용자 참여 데이터를 활용하여 온라인 토론 플랫폼에서 믿음을 임베딩 공간(embedding space)으로 매핑하는 방법을 사용하였습니다. 이 과정에서 미세 조정된 대형 언어 모델(LLM)을 활용하여 다양한 믿음의 상호 연결 및 사회적 이슈에 대한 양극화를 효과적으로 포착하였습니다.

- **Technical Details**: 사용자 참여 기록을 통해, 우리는 믿음의 상호 의존성을 효과적으로 캡슐화하는 표현 공간을 만드는 것을 목표로 하였습니다. 이 연구에서는 Contrastive Learning(대조 학습)을 통해 사전 훈련된 LLM을 미세 조정하였습니다. 데이터셋은 Debate.org에서 수집한 것으로, 총 59,986개의_unique debate_titles_와 40,280명의 уникальные_users에 의한 197,306회의 투표 기록을 포함합니다.

- **Performance Highlights**: 미세 조정된 S-BERT 모델은 트레인 데이터셋에서 약 98%의 평균 정확도를, 테스트 데이터셋에서는 약 67%의 정확도를 달성하여 가장 높은 성능을 나타냈습니다. 이를 통해 믿음 간의 거리가 개인의 새로운 믿음을 예측하는 데 유용한 지표로 작용할 수 있다는 것을 보여주었습니다.



### BERT's Conceptual Cartography: Mapping the Landscapes of Meaning (https://arxiv.org/abs/2408.07190)
- **What's New**: 이번 논문은 단어의 맥락적 뉘앙스를 탐구하기 위한 첫 번째 단계를 제시합니다. 이는 개념적 엔지니어링(Conceptual Engineering) 프로젝트에서 유용하게 활용될 수 있는 개념적 풍경(conceptual landscapes)을 생성함으로써 이루어집니다.

- **Technical Details**: 논문에서는 British National Corpus의 spoken component와 BERT를 사용하여 단어의 맥락화된 표현력을 분석합니다. 이 과정을 통해 Gaussian Mixture Models(GMM)와 다양한 메트릭(metric), 정성적 분석을 통해 렉시컬 풍경(lexical landscapes)을 시각화하고 수치적으로 표현합니다.

- **Performance Highlights**: 연구 결과, 단어 사용의 맥락에 따른 복잡한 변화를 발견하였으며, 이는 개념적 엔지니어링에서 일률적인 접근 방식이 효과적이지 않음을 시사합니다. 본 연구는 언어학 및 철학적 분석뿐만 아니라 더 정교하고 윤리적인 언어 처리 향상을 위한 도구를 제공합니다.



### Unlocking Efficiency: Adaptive Masking for Gene Transformer Models (https://arxiv.org/abs/2408.07180)
Comments:
          10 pages, 5 figures. Accepted for publication at the 27th European Conference on Artificial Intelligence (ECAI 2024)

- **What's New**: 이번 연구에서는 CM-GEMS라는 Curriculum Masking 기반의 유전자 마스킹 전략이 제안되었습니다. 이 전략은 예측 과제의 난이도를 체계적으로 증가시키며, 기존의 마스킹 방식에 비해 우수한 표현 학습 능력을 보여줍니다.

- **Technical Details**: CM-GEMS는 k-mers 대신 Pointwise Mutual Information (PMI) 기반 난이도 기준을 활용하여 마스킹을 진행합니다. 이로 인해 모델이 전방위적으로 잘 연결된 상위 PMI 토큰을 선택하도록 유도하여 학습 효율성을 높입니다. 또한, CM-GEMS는 시간에 따라 변하는 동적 마스킹 전략을 채택하여 예비 학습이 진행됨에 따라 난이도가 증가하도록 설계되었습니다.

- **Performance Highlights**: CM-GEMS는 10K 스텝만으로 기존의 120K 스텝 모델보다 우수한 성능을 달성하였고, GeneMask를 사용한 baseline에 비해 2.18% 향상된 결과를 보였습니다. 이는 특히 few-shot 및 full dataset 설정 모두에서 검증되었습니다.



### Self-folding Self-replication (https://arxiv.org/abs/2408.07154)
- **What's New**: 이 논문은 단순한 구성 요소의 1차원 체인으로부터 3차원 구조와 기계를 구축하는 새로운 방법을 제시합니다. 이를 통해 자기 복제(self-replication) 메커니즘을 재현하는 동시에 과정을 크게 단순화할 수 있음을 보여줍니다.

- **Technical Details**: 새로운 유형의 folding blocks를 도입하여 {
alpha}-helices와 eta-sheets와 같은 2차 구조뿐만 아니라 자기 복제 기계(self-replicating machines)를 포함하는 3차 및 4차 구조를 형성하는 데 도움을 줍니다. 또한 회전 자유도(rotational degrees of freedom)를 도입하여 블록의 종류를 줄이고, 기계의 전체 크기를 5배 감소시킵니다.

- **Performance Highlights**: 약 40개의 블록으로 구성된 매우 효율적인 자기 복제 메커니즘인 범용 복사기-구성기(universal copier-constructor)를 소개합니다. 이 연구는 더 정교한 자기 복제 시스템으로 나아가는 진화적 고려 사항을 다루고 있습니다.



### Language Models as Models of Languag (https://arxiv.org/abs/2408.07144)
Comments:
          Forthcoming in Nefdt, R., Dupre, G., \& Stanton, K. (eds.), The Oxford Handbook of the Philosophy of Linguistics. Oxford University Press

- **What's New**: 이 논문은 현대 언어 모델이 이론적 언어학에 기여할 잠재력을 비판적으로 검토합니다. 최근 인공지능 분야의 발전에도 불구하고, 기존의 언어 모델들이 언어학 이론에 어떤 의미가 있는지 재조명할 필요성이 강조됩니다.

- **Technical Details**: 이 논문에서는 현대 언어 모델이 복잡한 언어 지식을 데이터 노출만으로 습득하며, 계층적 구문 구조와 여러 언어 현상에 대한 민감성을 나타낼 수 있다는 것을 empirically (경험적으로) 증명합니다. 또한, 학습 조건을 정확하게 조절하고 인과적 개입(causal intervention) 방법을 활용한 실험들이 언어 습득 및 능력에 관한 가설을 제한할 수 있는 가능성을 탐색합니다.

- **Performance Highlights**: 현대 언어 모델은 과거의 일률적인 규칙 기반 접근 방식보다 우수한 성능을 보여주며, 이는 언어 습득 및 이론적 토대에 대한 정보를 제공할 수 있는 기회를 제공하는 것으로 보입니다. 따라서 이론적 언어학자와 컴퓨터 연구자 간의 협력이 중요하다고 결론짓습니다.



### ELLA: Empowering LLMs for Interpretable, Accurate and Informative Legal Advic (https://arxiv.org/abs/2408.07137)
- **What's New**: 이번 연구에서는 법률 LLMs의 해석 가능성, 정확성, 그리고 유익한 법률 조언을 제공하는 툴인 \\textbf{ELLA}를 제안합니다. ELLA는 법률 문서와 LLM의 응답 간의 상관관계를 시각적으로 제공하여 사용자에게 응답의 법적 근거를 직관적으로 전달합니다.

- **Technical Details**: ELLA는 BGE 모델을 세밀하게 조정하여 각 문장의 법적 근거를 검색하며, 사용자 쿼리를 기반으로 관련 법률 기사를 검색합니다. 또한, 사용자 상호작용을 통해 LLM이 더 정확한 응답을 생성할 수 있도록 합니다.

- **Performance Highlights**: 사용자 연구 결과, 법적 근거를 제공하는 것이 사용자 이해도를 높이고, 사용자가 법적 기사를 선택할 때 LLM의 응답 정확성이 향상됨을 보여주었습니다. 이로 인해 사용자들은 더 포괄적인 정보를 얻을 수 있었습니다.



### Quantifying over Optimum Answer Sets (https://arxiv.org/abs/2408.07697)
- **What's New**: 본 논문에서는 약한 제약 조건(weak constraints)을 포함한 Answer Set Programming with Quantifiers (ASP(Q))의 확장을 제안합니다. 이를 통해 다항 계층(Polynomial Hierarchy)에서의 문제들을 우아하고 간결하게 인코딩할 수 있는 방법을 제공합니다.

- **Technical Details**: ASPω(Q)는 약한 제약 조건을 통해 정량화된 구성 프로그램 내에서 부분 최적화(local optimization) 및 전역 최적화(global optimization) 기준을 표현할 수 있습니다. 이 새로운 형식은 다양한 응용 시나리오를 통해 모델링 기능을 시연합니다.

- **Performance Highlights**: ASPω(Q)는 n개의 교차 정량자(alternating quantifiers)를 포함하는 프로그램이 Δ(n+1)P 문제를 모델링할 수 있다는 긍정적인 결과를 보여줍니다. 이는 ASP(Q) 프로그램에 약한 제약 조건을 추가함으로써 새로운 언어의 비명백한 특성을 드러냅니다.



### Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities (https://arxiv.org/abs/2408.07666)
- **What's New**: 이 논문은 모델 병합의 여러 기법들을 종합적으로 검토하고 이해를 돕기 위한 체계적인 조사 연구입니다. 기존의 연구에서는 모델 병합에 대한 문헌이 부족하여 이 문제를 해결하기 위해 새로운 분류 체계를 제안합니다.

- **Technical Details**: 모델 병합은 원시 학습 데이터에 접근할 필요 없이 여러 개별 모델의 매개변수를 결합하여 보편적인 모델을 만드는 효과적인 기술입니다. 본 논문에서는 모델 병합 방법을 사전 병합(pre-merging)과 병합 중(during-merging) 두 단계로 나누어 탐구하며, 이 과정에서 각 단계별 세부 기술과 이론적 분석을 제공합니다. 특히, 사전 병합 방법은 입력 공간 및 가중치 공간을 분리하기 위한 선형화된 미세 조정(linearized fine-tuning), 이종 모델을 동질 모델로 변환하는 아키텍처 변환, 가중치 정렬을 포함합니다.

- **Performance Highlights**: 모델 병합 기술은 대형 언어 모델(large language models) 및 다중 모달 모델(multimodal models)과 같은 여러 하위 분야에서 뛰어난 성능을 보이며, 연속 학습(continual learning), 다중 작업 학습(multi-task learning), 소량 학습(few-shot learning) 등 다양한 적용에서 그 가능성을 보여줍니다. 또한, 이미지 생성 모델에서 스타일 혼합(style mixing) 및 훈련 비용 절감(training cost reduction) 등의 응용을 통해 더욱 발전하고 있습니다.



### See It All: Contextualized Late Aggregation for 3D Dense Captioning (https://arxiv.org/abs/2408.07648)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 이번 논문에서는 3D dense captioning의 새로운 패러다임인 late aggregation을 도입하여, 객체 지역화 및 설명 생성을 더욱 효과적으로 수행하는 SIA(See-It-All)라는 transformer 파이프라인을 제안합니다.

- **Technical Details**: SIA는 두 가지 종류의 쿼리를 동시에 디코딩하며, 첫 번째는 객체의 지역화 및 속성 설명에 초점을 맞춘 instance 쿼리이고, 두 번째는 여러 객체 간 또는 전역 장면 간의 관계를 포착하는 context 쿼리입니다. 이를 통해 두 쿼리의 정보를 후에 집계하여 최종 캡션을 생성합니다.

- **Performance Highlights**: ScanRefer 및 Nr3D 두 가지 널리 사용되는 3D dense captioning 데이터셋에서 실험한 결과, SIA는 기존 방법들보다 현저한 성능 향상을 보여주었습니다.



### Hierarchical Working Memory and a New Magic Number (https://arxiv.org/abs/2408.07637)
Comments:
          16 pages, 7 figures

- **What's New**: 이 연구는 작업 기억(working memory) 내에서 정보를 청킹(chunking)하는 방법에 대한 새로운 신경 네트워크 모델을 제안합니다. 이 모델은 시냅스 이론과 관련하여 구체적인 신경 메커니즘을 규명하려는 노력의 일환입니다.

- **Technical Details**: 제안된 모델은 반복 신경망(recurrent neural network, RNN)을 기반으로 하며, 뇌의 특정 클러스터들이 자극을 수용하고 청킹하여 정보를 저장하고 검색할 수 있도록 합니다. 이 과정에서 시냅스 증강(synaptic augmentation, SA)과 같은 장기적인 강화 메커니즘이 포함됩니다.

- **Performance Highlights**: 모델의 예측은 간질 환자의 단일 유닛 반응과 언어 자료에 대한 기억 실험을 통해 확인되었습니다. 연구 결과, 이 모델은 작업 기억의 효과적인 용량을 청킹을 통해 개선할 수 있음을 보여주었습니다.



### Transformers and Large Language Models for Efficient Intrusion Detection Systems: A Comprehensive Survey (https://arxiv.org/abs/2408.07583)
Comments:
          arXiv admin note: text overlap with arXiv:2405.04760 by other authors

- **What's New**: 이 논문은 Transformers와 LLMs를 활용한 사이버 위협 탐지 시스템의 종합적인 분석을 제공하며, IDS(침입 탐지 시스템)의 발전을 설명합니다.

- **Technical Details**: Transformers의 기본 원리, 사이버 공격의 배경, 및 다양한 데이터셋에 대해 설명합니다. Attention 기반 모델, BERT, GPT와 같은 LLM들, CNN/LSTM-Transformer 하이브리드 모델, ViTs(비전 트랜스포머) 등 여러 아키텍처의 탐색이 이루어집니다.

- **Performance Highlights**: Transfomers 및 LLMs가 IDS의 정확도를 높여주는 잠재적 응용 가능성을 강조하며, 네트워크 트래픽 흐름에서 패턴 변화를 탐지할 수 있는 새로운 방법론을 제시합니다.



### MathScape: Evaluating MLLMs in multimodal Math Scenarios through a Hierarchical Benchmark (https://arxiv.org/abs/2408.07543)
- **What's New**: 새로운 벤치마크 MathScape가 도입되어 MLLM의 수학적 문제 해결 능력을 평가하는 데 초점을 맞추고 있으며, 시각적 정보와 텍스트 정보를 통합한 평가 방식이 특징입니다.

- **Technical Details**: MathScape는 사진 기반의 수학 문제 시나리오를 평가하며, 이론적 이해와 적용 능력을 계층적 접근 방식으로 측정합니다. 또한, LLMs를 통해 서브 문제의 답변을 추출하고, 각 솔루션의 정확성을 평가하는 2단계 평가 방법을 도입하였습니다.

- **Performance Highlights**: 실험에서는 11개의 고급 MLLM에 대해 다차원적인 평가를 실시하였으며, MathScape 벤치마크가 가장 진보된 모델들에게도 도전 과제가 된다는 사실을 확인했습니다.



### Development of a Multi-Agent Clinical Decision Support System for Korean Triage and Acuity Scale (KTAS)-Based Triage and Treatment Planning in Emergency Departments (https://arxiv.org/abs/2408.07531)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 활용한 임상 의사 결정 지원 시스템(CDSS)이 응급실(ED) 의사와 간호사에게 환자 분류 및 치료 계획 수립을 지원하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 멀티 에이전트 CDSS는 Llama-3-70b를 기본 LLM으로 사용하며 CrewAI와 Langchain에 의해 조정됩니다. 시스템은 Triage Nurse, Emergency Physician, Pharmacist, ED Coordinator를 모사하는 네 개의 AI 에이전트로 구성되어 있으며, 비상 분류를 위해 한국 비상 분류 및 중증도 척도(KTAS)와 통합되어 있습니다. 또한, RxNorm API와 연동하여 약물 관리를 수행합니다.

- **Performance Highlights**: 이 CDSS는 Asclepius 데이터셋을 사용하여 평가되었으며, 단일 에이전트 시스템의 기준과 비교하여 높은 정확도를 보여주었습니다. 주요 진단, 중대한 발견 식별, 배치 결정, 치료 계획 및 자원 배분 등 주요 영역에서도 강력한 성능을 발휘했습니다.



### Enhancing Visual Question Answering through Ranking-Based Hybrid Training and Multimodal Fusion (https://arxiv.org/abs/2408.07303)
Comments:
          Visual Question Answering, Rank VQA, Faster R-CNN, BERT, Multimodal Fusion, Ranking Learning, Hybrid Training Strategy

- **What's New**: 이 논문에서는 복잡한 질문에 대해 보다 정확한 답변을 제공하기 위한 Visual Question Answering (VQA) 모델인 Rank VQA 모델을 제안합니다. 이 모델은 순위 기반의 하이브리드 훈련 전략을 사용하여 VQA 성능을 향상시킵니다.

- **Technical Details**: Rank VQA 모델은 Faster R-CNN 모델을 통해 추출된 고품질 비주얼 특성과 사전 학습된 BERT 모델로부터 얻은 풍부한 시맨틱 텍스트 특성을 통합합니다. 이러한 특성은 다중 헤드 자기 주의 메커니즘을 사용하는 복잡한 멀티모달 융합 기법을 통해 융합됩니다. 또한, 상대적 순위를 최적화하는 순위 학습 모듈이 통합되어 답변의 정확성을 개선합니다.

- **Performance Highlights**: 실험 결과, Rank VQA 모델은 VQA v2.0과 COCO-QA를 포함한 표준 VQA 데이터셋에서 기존의 최신 모델들보다 성능이 크게 향상되었습니다. 이 모델은 복잡한 질문을 처리하는 데 효과적이며, 이미지와 텍스트에서 미묘한 세부 사항을 이해하고 정교한 추론을 수행할 수 있는 능력을 보여줍니다.



### Effects of a Prompt Engineering Intervention on Undergraduate Students' AI Self-Efficacy, AI Knowledge and Prompt Engineering Ability: A Mixed Methods Study (https://arxiv.org/abs/2408.07302)
Comments:
          34 pages, 6 figures

- **What's New**: 이 연구는 ChatGPT와 같은 대형 언어 모델(LLMs)과의 효과적인 상호작용을 위한 prompt engineering의 중요성을 강조합니다. 특히, 학부생들을 위한 prompt engineering 교육 개입을 설계하고 진행했습니다.

- **Technical Details**: 연구는 홍콩의 한 대학교에서 역사 수업 중 진행된 100분짜리 워크숍에 27명의 학생이 참여했습니다. 워크숍 동안 학생들은 prompt engineering 전략을 배우고, 이를 최종 에세이 과제 계획에 적용했습니다. 학습 효과를 측정하기 위해 여러 데이터 소스를 수집했으며, 이에 포함된 것은 사전 및 사후 설문조사 응답, 프롬프트 라이브러리 및 작성된 반성문입니다.

- **Performance Highlights**: 연구 결과, 학생들은 AI에 대한 자기 효능감(AI self-efficacy), AI 개념에 대한 이해도 및 효과적인 프롬프트 작성을 위한 능력이 향상되었습니다. 이러한 결과는 AI 리터러시 교육에 중요한 시사점을 제공하며, prompt engineering 교육의 필요성을 강조합니다.



### Post-Training Sparse Attention with Double Sparsity (https://arxiv.org/abs/2408.07092)
- **What's New**: 이번 논문에서는 "Double Sparsity"라는 새롭고 혁신적인 포스트 트레이닝( post-training ) 희소 주목 기법을 제안하여 Key-Value (KV) 캐시 접근을 줄이고 추론 시간을 단축시킵니다. 이 기법은 중요 토큰만을 사용하는 토큰 희소성(token sparsity)과 중요한 특징 채널을 활용하여 중요 토큰을 식별하는 채널 희소성(channel sparsity)을 결합하여 효과적입니다.

- **Technical Details**: Double Sparsity는 오프라인 보정을 통해 상대적으로 정적인 채널 희소성을 활용하여 런타임에서 중요 토큰을 동적으로 식별합니다. 이 기법은 Llama-2-7B, Llama-2-70B 및 Mixtral-8x7B 모델을 포함한 다양한 작업에서 높은 정확도를 유지하면서 1/16의 토큰 및 채널 희소성을 달성할 수 있습니다. 그리고 이를 통해 GPU의 메모리 사용량을 크게 줄이고 주목 연산을 최대 14.1배 가속할 수 있습니다.

- **Performance Highlights**: 실험 결과, Double Sparsity는 다양한 작업에서 정확성에 미치는 영향이 미미하면서 1.9배의 최종 추론 속도 향상과 함께 256K의 시퀀스 길이에서 기존 솔루션에 비해 16.3배 더 빠른 디코딩 속도를 달성했습니다. 이 기법은 메모리 접근을 줄이고 런타임 속도를 가속화하여 효율적인 추론을 가능하게 합니다.



### Node Level Graph Autoencoder: Unified Pretraining for Textual Graph Learning (https://arxiv.org/abs/2408.07091)
- **What's New**: 최근 연구에서 제안된 Node Level Graph AutoEncoder (NodeGAE)는 텍스트 그래프에 대한 새로운 비지도 학습 프레임워크로, 특성 매핑과 텍스트 복원을 통해 보다 고급의 특징 추출을 지향합니다.

- **Technical Details**: NodeGAE는 언어 모델을 기반으로 한 인코더-디코더 구조를 갖추고 있으며, 인코더는 노드의 숨겨진 임베딩을 독립적인 텍스트로 매핑합니다. 또한, InfoNCE 손실을 도입하여 이웃 간의 유사도를 극대화하고 구조 정보를 캡처합니다.

- **Performance Highlights**: NodeGAE는 ogbn-arxiv 데이터셋에서 테스트 정확도 77.10%를 달성하며, 기존의 SOTA 방법들과 유사한 성능을 보이며, GNN과의 앙상블을 통해 78.34%의 새로운 SOTA 정확도를 기록했습니다.



### MathBridge: A Large-Scale Dataset for Translating Mathematical Expressions into Formula Images (https://arxiv.org/abs/2408.07081)
Comments:
          9page, 6 figures

- **What's New**: MathBridge라는 새로운 데이터셋이 소개되었으며, 이는 수학적 영어 발음을 LaTeX으로 변환하는 최초의 대규모 데이터셋입니다. 해당 데이터셋은 약 2300만 개의 LaTeX 공식과 해당 영어 표현 쌍으로 구성되어 있습니다.

- **Technical Details**: MathBridge는 영어 텍스트를 LaTeX으로 변환하여 수학 공식을 이미지로 나타내는 과정의 첫 번째 단계에서 중요한 역할을 합니다. 이 시스템은 텍스트에서 LaTeX으로, 그리고 LaTeX에서 이미지로의 변환을 포함합니다. 연구는 사전훈련된 언어 모델을 활용하여 MathBridge에서 추출한 데이터로 모델을 미세 조정했습니다.

- **Performance Highlights**: T5-large 모델을 기준으로 sacreBLEU 점수가 4.77에서 46.8로 상승하여 텍스트를 LaTeX으로 변환하는 능력이 크게 향상되었습니다. 이러한 결과는 수학 공식을 이해하는 데 있어 접근성을 개선하고, 교육 기술 향상에 기여할 수 있습니다.



### An Event Structure-aware Generative Model for Biomedical Event Extraction (https://arxiv.org/abs/2408.06583)
Comments:
          8 pages, 4 figures, 6 tables

- **What's New**: 이 논문은 Biomedical Event Extraction (BEE) 과제를 처리하기 위해 구조 인식(semi-structured) 접두사를 갖춘 생성 모델인 GenBEE를 제안합니다. GenBEE는 대량의 언어 모델(LLM)에서 추출된 정보를 활용하여 이벤트 프롬프트를 만들어, 레이블 의미(label semantics)와 인수 의존관계(argument dependency relationships)를 모두 통합합니다.

- **Technical Details**: GenBEE는 이벤트 트리거 감지(event trigger detection)와 인수 추출(argument extraction)을 동시에 처리하는 프레임워크입니다. 이 모델은 구조 인식 접두사 학습 모듈을 도입하여, 구조적 특성으로 풍부해진 구조 인식 접두사를 생성합니다. 이를 통해 복잡한 이벤트 구조를 포착하고, 주어진 컨텍스트에서 더 나은 표현 학습(ctxual representation learning)을 달성합니다.

- **Performance Highlights**: GenBEE는 MLEE와 GE11 데이터셋에서 최첨단 성능(state-of-the-art performance)을 달성했으며, PHEE 데이터셋에서도 경쟁력 있는 결과를 보였습니다. 실험 결과, GenBEE는 기존 분류 기반 모델보다 우수한 성능을 보였습니다.



New uploads on arXiv(cs.IR)

### Towards Fair and Rigorous Evaluations: Hyperparameter Optimization for Top-N Recommendation Task with Implicit Feedback (https://arxiv.org/abs/2408.07630)
- **What's New**: 이번 연구에서는 Top-N 암묵적 추천 문제를 조사하고, 하이퍼파라미터 최적화 알고리즘을 사용하여 비교 실험에서 자주 사용되는 벤치마크 추천 알고리즘을 최적화하는 데 중점을 두었습니다. 또한 다양한 추천 알고리즘에 적합한 하이퍼파라미터 검색 알고리즘을 식별했습니다.

- **Technical Details**: 이 연구에서는 6개의 일반적인 추천 알고리즘(ItemKNN, PureSVD, Matrix Factorization, Factorization Machine, NeuMF, NGCF)과 7가지 하이퍼파라미터 최적화 알고리즘(랜덤 탐색, 시뮬레이티드 어닐링, 3종의 베이지안 최적화, 다중 팔 밴딧 알고리즘 등)을 사용하여 세 가지 데이터 세트에서 실험을 수행하였습니다.

- **Performance Highlights**: 연구 결과, 추천 알고리즘의 성능을 최적 수준으로 높일 수 있는 가장 적합한 하이퍼파라미터 최적화 기법을 제시하여 향후 연구에 신뢰할 수 있는 비교 기준을 제공하였습니다.



### Beyond Inter-Item Relations: Dynamic Adaptive Mixture-of-Experts for LLM-Based Sequential Recommendation (https://arxiv.org/abs/2408.07427)
Comments:
          11 pages, 14 figures

- **What's New**: 이 논문에서는 기존의 LLM 기반 시퀀스 추천 시스템(SRS)의 한계를 극복하기 위해 MixRec라는 새로운 모델을 제안합니다. MixRec은 내부 아이템 관계를 포착하고, 장기 협업 지식을 통합하며, 유동적인 아키텍처 설계를 통해 적응할 수 있도록 설계되었습니다.

- **Technical Details**: MixRec은 대규모 언어 모델(LLM)을 바탕으로 하여, (1) context masking을 통해 아이템 텍스트 내의 관계를 모델링하고, (2) collaborative knowledge injection을 통해 장기 협업 지식을 통합하며, (3) Bayesian optimization을 이용한 동적 적응형 믹스 전문가 구조를 활용하여 다양한 시퀀스 정보를 효과적으로 처리합니다.

- **Performance Highlights**: 다양한 실험을 통해 MixRec은 시퀀스 추천을 동적이고 적응적인 방식으로 효과적으로 처리할 수 있음을 입증하였습니다.



### SumRecom: A Personalized Summarization Approach by Learning from Users' Feedback (https://arxiv.org/abs/2408.07294)
- **What's New**: 다수 문서 요약(Multi-document summarization) 기법의 한계를 극복하고 개인의 관심사를 반영한 사용자 맞춤형 요약을 제공하는 새로운 접근 방식인 SumRecom을 제안합니다.

- **Technical Details**: SumRecom은 사용자 피드백을 기반으로 하여 사용자의 선호도를 추출하고 개인화된 요약 결과를 생성하는 상호작용 기반 요약 기법입니다. 두 단계로 나눌 수 있으며: 1) 사용자 선호도 추출기(User Preference Extractor), 2) 요약기(Summarizer)입니다. SumRecom은 Active Learning과 Preference Learning을 활용하여 사용자의 콘텐츠 선택에서의 선호도를 추출하고, Integer Linear Programming (ILP)과 Inverse Reinforcement Learning (IRL)을 통해 요약의 품질을 평가합니다.

- **Performance Highlights**: 다양한 자동화된 평가 및 인간 평가에서 SumRecom이 사용자 맞춤형 요약을 생성하는 데 있어 탁월한 성능을 나타내었으며, 사용자의 피드백을 반영하여 최적의 요약을 생성하는 능력이 입증되었습니다.



### Scene-wise Adaptive Network for Dynamic Cold-start Scenes Optimization in CTR Prediction (https://arxiv.org/abs/2408.07278)
Comments:
          10 pages, 6 figures, accepted by Recsys 2024

- **What's New**: 최신 모바일 E-commerce에서 사용자에게 근처 상업 서비스 추천을 제공하는 것이 점점 더 중요해지고 있습니다. 기존의 추천 시스템들은 새로운 장면에서의 cold-start 문제를 해결하는 데 어려움을 겪고 있습니다. 본 연구에서는 Scene-wise Adaptive Network (SwAN)을 제안하여 이러한 문제를 해결하였습니다.

- **Technical Details**: SwAN은 장면 유사성 학습, 사용자별 장면 전환 인지, 새로운 장면을 위한 장면별 정보 구축, 장면 간의 논리 정보 차이 강화 등 여러 가지 기능을 포함하고 있습니다. 추천 시스템은 Embedding&MLP(다층 퍼셉트론) 패러다임을 따르며, Scene Relation Graph(SRG) 및 Similarity Attention Network(SAN)를 통해 장면 간의 유사성을 파악하고, Adaptive Ensemble-experts Module(AEM)을 이용해 공유 및 특정 정보를 추출합니다.

- **Performance Highlights**: SwAN은 Meituan의 온라인 추천 서비스에 성공적으로 배포되어 기존 모델 대비 5.64%의 클릭률(CTR) 개선 효과를 보였습니다. 또한, 하루 주문량 비율이 5.19% 증가하는 성과를 달성하였습니다.



### On the Local Ultrametricity of Finite Metric Data (https://arxiv.org/abs/2408.07174)
Comments:
          12 pages, 3 figures, 3 tables

- **What's New**: 이 논문에서는 p-adic Mumford 곡선의 관점에서 유한 메트릭 데이터에 대한 새로운 로컬 ultrametricity 측정을 제안합니다. 이 접근법은 특히 iris 데이터셋을 실험적으로 적용하여 유용성을 보여줍니다.

- **Technical Details**: 로컬 ultrametricity는 Vietoris-Rips 그래프와 관련 및 연결된 구성 요소로서의 클러스터를 통해 정의됩니다. p-adic 수는 계층적 데이터 분석에 유용한 구조를 제공하며, 정규 미분 1형식으로부터 유래된 Radon 측정이 더 자연스러운 방법으로 제시됩니다. 특정 계층적 클러스터링 방법은 메트릭 공간에 대한 ultrametric을 제공합니다.

- **Performance Highlights**: 제안된 방법은 기존의 ultrametricity 지수를 개선하여 데이터 분석의 새로운 관점을 제시하며, 특히 데이터가 Mumford 곡선에서 샘플링된 것으로 간주할 수 있다는 점에 주목합니다. 이 연구는 Vietoris-Rips 그래프를 통해 p-adic 방법론을 적용하여 유한 메트릭 데이터셋을 정교하게 분석합니다.



### Exact Trajectory Similarity Search With N-tree: An Efficient Metric Index for kNN and Range Queries (https://arxiv.org/abs/2408.07650)
Comments:
          54 pages, 26 figures

- **What's New**: 본 논문에서는 이동 객체의 경로 유사성 검색을 위한 새로운 거리 함수인 DistanceAvg를 제안합니다. 이 함수는 이동 경로의 유사성을 파악하는 데 중점을 두고 있습니다.

- **Technical Details**: 유사성 검색(similarity search)은 주어진 쿼리 객체와 유사한 객체를 찾는 문제로, DistanceAvg는 이러한 유사성을 측정하는 메트릭(metric)입니다. 메트릭 공간(metric space) 접근 방식을 활용하여 효율적인 인덱싱을 제공합니다.

- **Performance Highlights**: 제안된 인덱스는 kNN 쿼리와 범위 쿼리(range query) 모두에서 기존 인덱스보다 뛰어난 성능을 보여줍니다. kNN 쿼리는 범위 쿼리보다 실제 응용에 더 적합하며, 주어진 거리 함수에 대해 정확한 결과 집합을 제공합니다.



### WeKnow-RAG: An Adaptive Approach for Retrieval-Augmented Generation Integrating Web Search and Knowledge Graphs (https://arxiv.org/abs/2408.07611)
Comments:
          8 pages, 2 figures, technical report for 3rd place in Task 3 of Meta KDD Cup 2024 CRAG Challenge

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 신뢰성을 향상시키기 위해 WeKnow-RAG라는 새로운 접근 방식을 제안합니다. 이 시스템은 웹 검색과 지식 그래프를 통합하여 사실 기반 정보를 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: WeKnow-RAG는 'Retrieval-Augmented Generation (RAG)' 시스템을 기반으로 하여 지식 그래프의 구조적인 표현과 밀집 벡터 검색의 유연성을 결합합니다. 이 시스템은 특정 도메인에 맞는 지식 그래프를 활용하여 다양한 쿼리를 처리합니다.

- **Performance Highlights**: 결과적으로 이 접근 방식은 정보 검색의 효율성과 정확성을 효과적으로 균형 잡아주며, 다양한 오프라인 실험과 온라인 제출에서 뛰어난 효과성을 입증했습니다.



### New Curriculum, New Chance -- Retrieval Augmented Generation for Lesson Planning in Ugandan Secondary Schools. Prototype Quality Evaluation (https://arxiv.org/abs/2408.07542)
Comments:
          Presented at Ndejje University Second Annual Research Dissemination Symposium 2024

- **What's New**: 이번 연구에서는 우간다의 중등학교 교육 질 향상을 위한 최신 접근 방식이 소개되었습니다. 새로운 커리큘럼에 대응하기 위해 정부 인증 교과서를 기반으로 맞춤형 수업 계획을 생성하는 Retrieval Augmented Generation(수집 증강 생성) 기법을 사용하여 프로토타입이 개발되었습니다.

- **Technical Details**: 프로토타입은 Cohere LLM 및 Sentence Embeddings와 LangChain Framework를 이용하여 구축되었으며, 공용 웹사이트에서 제공됩니다. ICT, 수학, 역사와 같은 세 가지 새로운 커리큘럼 교과서에 대해 벡터 저장소를 훈련시켰습니다. 24개의 수업 계획이 교과서에서 제안된 수업 기간에 따라 의사 랜덤 생성 프로토콜을 적용하여 생성되었습니다. 수업 계획은 Ndihokubwayo et al. (2022)의 Lesson Plan Analysis Protocol (LPAP)에 따라 세 명의 독립 평가자가 기술적 품질을 분석했습니다.

- **Performance Highlights**: 생성된 24개의 수업 계획에 대한 LPAP 평가 결과, 평균 품질은 75%에서 80%로 나타났으며 이는 '매우 좋은 수업 계획'에 해당합니다. 모든 수업 계획이 65% 이하의 점수를 기록하지 않았으며, 하나의 수업 계획은 주제가 누락된 것으로 평가될 수 있었습니다. 연구에 따르면, 생성된 수업 계획의 품질은 인적 자원에 의해 생성된 수업 계획의 품질과 적어도 동등하거나 더 나은 수준임을 보여 주었습니다.



### GQE: Generalized Query Expansion for Enhanced Text-Video Retrieva (https://arxiv.org/abs/2408.07249)
Comments:
          18 pages including appendix

- **What's New**: 본 논문은 'Generalized Query Expansion (GQE)'라는 새로운 데이터 중심의 접근 방식을 소개하여 텍스트와 비디오 간의 정보 불균형 문제를 해결하고, 텍스트-비디오 검색 시스템의 효과성을 향상시키려 한다.

- **Technical Details**: GQE는 훈련 및 테스트 단계 동안 비디오와 연결된 텍스트 쿼리를 확장하여 정보 불균형을 해소한다. GQE는 비디오를 짧은 클립으로 적응적으로 분할하고, 'zero-shot captioning'을 활용하여 보다 포괄적인 장면 설명으로 훈련 데이터셋을 풍부하게 만든다. 검색 시에는 대형 언어 모델('Large Language Models', LLM)을 사용하여 다양한 쿼리를 생성하고, 적합성과 다양성을 기반으로 쿼리 선택 모듈을 통해 이러한 쿼리를 필터링한다.

- **Performance Highlights**: GQE는 MSR-VTT, MSVD, LSMDC, VATEX 등 여러 벤치마크에서 최첨단 성능을 달성하여 텍스트-비디오 검색 문제를 데이터 중심의 관점에서 효과적으로 접근할 수 있음을 입증한다.



New uploads on arXiv(cs.CV)

### Knowledge Distillation with Refined Logits (https://arxiv.org/abs/2408.07703)
Comments:
          11 pages, 7 figures

- **What's New**: 최근 Knowledge Distillation의 연구가 Logit Distillation에 집중되고 있는 가운데, 이 논문에서는 Refined Logit Distillation (RLD)을 제안하였다. RLD는 기존의 Logit Distillation 방법의 한계를 해결하며, Teacher 모델이 보이는 예측 오류의 영향을 줄이는 데 목표를 두고 있다.

- **Technical Details**: RLD는 Student 모델이 Teacher로부터 중요한 Knowledge를 효과적으로 학습할 수 있도록 돕기 위해 Sample Confidence (SC)와 Masked Correlation (MC)이라는 두 가지 유형의 지식을 활용한다. SC는 예측 클래스와 나머지 클래스의 확률에서 유도된 이진 확률을 나타내며, MC는 Teacher 로짓 내에서 True Class에 비해 동등하거나 높은 순위를 가진 모든 클래스를 마스킹하여 Student 모델의 학습에 필요한 클래스 상관관계를 전달한다.

- **Performance Highlights**: CIFAR-100과 ImageNet 데이터셋에서 실험 결과, RLD가 기존의 방법들에 비해 우수한 성능을 보인다고 보고되었다.



### End-to-end Semantic-centric Video-based Multimodal Affective Computing (https://arxiv.org/abs/2408.07694)
Comments:
          Under Review

- **What's New**: 이 논문에서는 인공지능의 일반화 (AGI)에서 인간의 정서를 이해하는 것이 기계의 인지 능력을 향상시키기 위해 필수적이라는 점을 강조합니다. 특히, 인간이 말하는 비디오를 활용한 다중 모달 정서 컴퓨팅(Multimodal Affective Computing, MAC)에 중점을 두고 있습니다.

- **Technical Details**: 본 연구는 'SemanticMAC'라는 새로운 엔드 투 엔드 프레임워크를 제안하며, 사전 학습된 Transformer 모델을 사용하여 다중 모달 출처에서의 특징들을 처리합니다. 주요 구성요소로는 Affective Perceiver 모듈과 Semantic-centric Gated Feature Interaction (SGFI), Semantic-centric Label Generation (SCLG) 및 Semantic-centric Contrastive Learning (SCCL) 기법을 포함합니다.

- **Performance Highlights**: SemanticMAC는 7개의 공공 데이터 세트에서 4개의 MAC 하위 작업에서 기존 최첨단 방법을 능가하는 성능을 보여줍니다. 이 접근 방식은 변별력이 있는 특정 및 공유 의미 표현을 효과적으로 학습할 수 있습니다.



### Detecting Near-Duplicate Face Images (https://arxiv.org/abs/2408.07689)
Comments:
          Under review

- **What's New**: 이번 연구에서는 비슷하게 변형된 얼굴 이미지(near-duplicate face images) 탐지를 위한 새로운 방법을 제안합니다. 원본 이미지를 식별하고 원본 이미지와 비슷한 변형 이미지 간의 관계를 추론하는 과정에서 그래프 이론적 접근법을 활용한 이미지 계통수 트리(Image Phylogeny Tree, IPT)를 구축합니다.

- **Technical Details**: 제안된 방법은 그래프 기반 프레임워크를 사용하여 이미지들을 노드로, 이미지 간의 관계를 지시성(edge)으로 표현합니다. 각 이미지는 특정 특징(feature)을 통해 표현되며, 초기 인접 행렬(adjacency matrix)을 점진적으로 개선하여 최종 IPT를 생성합니다. 체계적인 노드 임베딩(node embedding)과 링크 예측(link prediction)을 통해 IPT와 이미지 계통수 숲(Image Phylogeny Forest, IPF)을 구성합니다.

- **Performance Highlights**: 제안된 방법은 기존 기술을 통해 이루어진 유사도보다 42% 향상된 IPF 재구성 정확도를 보여줍니다. 또한, 일반화 능력을 검증하기 위해 다양한 데이터 세트에서 엄격한 평가를 실시하였으며, 타 방법과의 비교에서 우수한 성능을 입증하였습니다.



### RSD-DOG : A New Image Descriptor based on Second Order Derivatives (https://arxiv.org/abs/2408.07687)
- **What's New**: 이 논문은 2차 이미지 통계/미분을 기반으로 한 새로운 이미지 패치 설명자를 소개합니다. 이미지 패치는 3차원 표면으로 처리되며, 여기서 조명이 세 번째 차원으로 사용됩니다. 이 방법은 방향성 필터의 반응과 Gaussian 차분(Difference of Gaussian, DOG) 접근 방식을 성공적으로 결합하여 원래의 2차 특징/statistics를 효과적으로 포착합니다.

- **Technical Details**: 제안된 설명자는 반 회전 반가우시안 필터를 사용하여 있는 깊이 방향에서 이미지 패치의 2차 통계를 추출합니다. 이 설명자는 조명, 축척, 회전, 흐림, 시점 및 압축의 변화에 대한 강한 구별력을 보여주며, 기존의 1차 통계 기반 설명자(SIFT, DAISY, GLOH 등)에 비해 차원이 약 3~4배 더 낮습니다.

- **Performance Highlights**: 실험 결과, 제안된 설명자는 불변성과 강인성을 보여주며, 다양한 조명 변화를 포함한 이미지 매칭 테스트에서 우수한 성능을 보였습니다. 특히 선형 및 비선형 조명 변화에 대해 강력한 검증을 받았습니다.



### A Spitting Image: Modular Superpixel Tokenization in Vision Transformers (https://arxiv.org/abs/2408.07680)
Comments:
          To appear in ECCV (MELEX) 2024 Workshop Proceedings

- **What's New**: 본 논문에서는 기존의 ViT (Vision Transformer) 아키텍처에서 이미지의 의미(content)와 무관하게 그리드 기반의 접근 방식을 사용하는 대신, 모듈화된 슈퍼픽셀 토큰화(superpixel tokenization) 전략을 제안합니다.

- **Technical Details**: 제안된 방법은 온라인 콘텐츠 인식(tokenization-aware) 토큰화와 스케일(scale) 및 형태(shape) 불변의 위치 임베딩(positional embeddings)을 이용합니다. 실험에서는 패치 기반 토큰화(patch-based tokenization)와 무작위 파트션(randomized partitions)을 기준선으로 삼아 비교했습니다.

- **Performance Highlights**: 우리의 방법은 어트리뷰션(attributions)의 정확도를 크게 향상시키고, 제로샷(zero-shot) 비지도 학습 밀집 예측(dense prediction) 작업에서 픽셀 수준의 세분성을 제공하면서도 분류(classification) 작업에서 예측 성능을 유지합니다.



### G$^2$V$^2$former: Graph Guided Video Vision Transformer for Face Anti-Spoofing (https://arxiv.org/abs/2408.07675)
Comments:
          11 pages, 5 figures

- **What's New**: 본 연구에서는 동적 및 광학적 특성을 모두 활용한 새로운 얼굴 안티 스푸핑 모델인 Graph Guided Video Vision Transformer (G²V²former)를 제안합니다. 이는 기존의 단일 프레임 기반 방법론의 한계를 극복하고, 시간에 따른 사진 정보를 포착하여 보다 정확한 스푸핑 탐지를 가능하게 합니다.

- **Technical Details**: G²V²former는 얼굴과 얼굴 랜드마크를 결합하여 동적 및 정적 특징의 융합을 구성합니다. 특히, Kronecker temporal attention을 설계하여 보다 넓은 수용 범위를 제공하고, 동적 정보를 효과적으로 포착하는 방법을 제시합니다. 이는 두 가지 차원으로 주의를 분해하여 구현되며, 동적 정보 추출을 위한 두 개의 스트림 구조를 사용합니다.

- **Performance Highlights**: 아홉 개의 벤치마크 데이터 세트에서 광범위한 실험을 통해 제안된 방법이 다양한 시나리오에서 우수한 성능을 보여 주었으며, 앞으로의 연구에서 코드를 공개할 예정입니다.



### See It All: Contextualized Late Aggregation for 3D Dense Captioning (https://arxiv.org/abs/2408.07648)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 이번 논문에서는 3D dense captioning의 새로운 패러다임인 late aggregation을 도입하여, 객체 지역화 및 설명 생성을 더욱 효과적으로 수행하는 SIA(See-It-All)라는 transformer 파이프라인을 제안합니다.

- **Technical Details**: SIA는 두 가지 종류의 쿼리를 동시에 디코딩하며, 첫 번째는 객체의 지역화 및 속성 설명에 초점을 맞춘 instance 쿼리이고, 두 번째는 여러 객체 간 또는 전역 장면 간의 관계를 포착하는 context 쿼리입니다. 이를 통해 두 쿼리의 정보를 후에 집계하여 최종 캡션을 생성합니다.

- **Performance Highlights**: ScanRefer 및 Nr3D 두 가지 널리 사용되는 3D dense captioning 데이터셋에서 실험한 결과, SIA는 기존 방법들보다 현저한 성능 향상을 보여주었습니다.



### Boosting Unconstrained Face Recognition with Targeted Style Adversary (https://arxiv.org/abs/2408.07642)
- **What's New**: 본 논문에서는 Targeted Style Adversary (TSA)라는 새로운 방법론을 제안합니다. TSA는 레이블이 있는 데이터와 없는 데이터 간의 인스턴스 수준의 피쳐 통계를 보간(interpolate)하여 훈련 데이터를 확장함으로써 얼굴 인식을 위한 훈련 데이터의 다양성을 증가시키는 단순하지만 효과적인 접근 방식을 제공합니다.

- **Technical Details**: TSA 방법은 두 가지 주요 관찰에 의해 동기 부여됩니다: (i) 입력 도메인은 피쳐 통계에 반영되며, (ii) 얼굴 인식 모델의 성능은 스타일 정보에 의해 영향을 받습니다. TSA는 모델의 숨겨진 공간(hidden space)에서 작동하여 레이블이 있는 샘플의 스타일 정보를 조작하고, 생성된 스타일의 타당성을 보장하면서 훈련 중 유사하지 않은 인스턴스를 구별하기 위해 엔트로피 기반 접근 방식을 제안합니다.

- **Performance Highlights**: TSA 방법은 unconstrained 벤치마크에서 평가된 결과, 경쟁 모델에 비해 우수하거나 동등한 성능을 보이며, 훈련 속도는 거의 70% 향상되었고 메모리 소비는 40% 감소하였습니다.



### Rethinking the Key Factors for the Generalization of Remote Sensing Stereo Matching Networks (https://arxiv.org/abs/2408.07613)
Comments:
          submitted to IEEE jstars

- **What's New**: 이 논문은 다양한 센서와 시나리오에서 얻은 크로스 도메인 데이터에 대한 스테레오 매칭 네트워크의 일반화 능력을 향상시키기 위한 훈련 데이터셋 선택, 모델 구조, 훈련 방식과 같은 핵심 훈련 요소들을 분석하고 제안합니다.

- **Technical Details**: 저자들은 훈련 데이터셋 선택 시 테스트 세트와 유사한 지역 목표 분포를 가진 데이터를 선택해야 하며, 다양한 크기의 특징에 유연하게 적용되는 계단식(Cascade) 구조가 선호된다고 주장합니다. 또한, 비지도(Unsupervised) 방법이 지도(Supervised) 방법보다 일반화 성능이 더 뛰어나며, 비지도 조기 종료 전략을 설계하여 최적의 모델을 유지하도록 돕습니다.

- **Performance Highlights**: 이 연구를 통해 저자들은 좋은 일반화 성능을 가진 비지도 스테레오 매칭 네트워크를 제안하고, 이를 뒷받침하는 폭넓은 실험을 수행했습니다. 연구의 결과와 함께 소스 코드 및 데이터셋을 공개하여 결과 재현을 장려합니다.



### Panacea+: Panoramic and Controllable Video Generation for Autonomous Driving (https://arxiv.org/abs/2408.07605)
Comments:
          Project page: this https URL. arXiv admin note: text overlap with arXiv:2311.16813

- **What's New**: 이번 논문에서는 자율주행 분야에 필요한 고품질 주석이 달린 비디오 학습 데이터 생성을 위한 새로운 프레임워크인 Panacea+를 제안합니다. Panacea+는 다중 시점 외관 노이즈 선행 기법과 슈퍼 해상도 모듈을 통합하여 일관성을 향상시키고 해상도를 높이는 데 중점을 두었습니다.

- **Technical Details**: Panacea+는 4D attention 메커니즘을 사용하여 다중 시점 및 시간 모델링을 효과적으로 수행하며, BEV(Bird's Eye View) 레이아웃을 제어 신호로 통합하여 가변 생성이 가능합니다. 또한, 생성된 샘플의 해상도를 높이기 위해 슈퍼 해상도 모듈을 추가했습니다. 이 프레임워크는 저해상도 샘플을 먼저 생성한 후 높은 해상도로 확장하는 효율적인 접근 방식을 채택하고 있습니다.

- **Performance Highlights**: Panacea+는 nuScenes 및 Argoverse 2 데이터셋을 사용한 다양한 실험에서 3D 객체 추적 및 검출, 차선 검출 작업들에서 SOTA(State-Of-The-Art) 성능을 달성하였습니다. 특히, 추적에서 AMOTA(Absolute Multi-Object Tracking Accuracy)가 42.7, 검출에서 NDS(Normalized Detection Score)가 53.8이라는 우수한 결과를 보였습니다.



### Disentangle and denoise: Tackling context misalignment for video moment retrieva (https://arxiv.org/abs/2408.07600)
- **What's New**: 이 논문은 교차 모드(video-text) 모멘트 검색에서의 성과 향상을 위해 교차 모드 Context Denoising Network (CDNet)을 제안합니다. 새로운 접근 방식은 불필요한 동적 조정과 복잡한 상관관계를 분리하여 정확한 순간 검색을 가능하게 합니다.

- **Technical Details**: Query-guided Semantic Disentanglement (QSD) 기능을 사용하여 비디오 순간을 글로벌 및 세부 상관관계에 따라 정렬 수준을 추정하면서 분리합니다. 또한, Context-aware Dynamic Denoising (CDD) 기능을 통해 쿼리와 관련된 시공간 세부 정보를 학습하여 세밀한 배열을 강화합니다.

- **Performance Highlights**: 제안된 CDNet은 QVHighlights 벤치마크에서 최신 성과를 달성했으며, Charades-STA와 TACoS에서도 경쟁력 있는 결과를 보였습니다.



### Progressive Radiance Distillation for Inverse Rendering with Gaussian Splatting (https://arxiv.org/abs/2408.07595)
- **What's New**: 이번 연구에서는 Progressive Radiance Distillation(진행적인 방사선 증류)이라는 새로운 역 렌더링 기법을 제안합니다. 이 방법은 Gaussian 기반의 방사선 필드 렌더링과 물리 기반 렌더링을 결합하여 더 나은 이미지 생성 품질을 목표로 합니다.

- **Technical Details**: 이 방법은 다중 시점 이미지를 입력으로 받고, 사전 훈련된 방사선 필드 가이드를 기반으로 물리적인 빛(light) 및 재질(material) 파라미터를 추출합니다. 초기에는 방사선 필드 렌더링을 선호하는 작은 값으로 증류 진행 맵(distillation progress map)을 설정하고, 초기 반복에서 물리적 파라미터가 수렴되지 않더라도 방사선 필드가 유지되어 이미지 손실의 기울기를 안전하게 합니다. 주요 수렴이 이루어지면 물리적 모델이 서서히 대체됩니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 새로운 뷰 합성과 리라이트(lighting)에서 최신 기술보다 품질 면에서 월등한 성능을 보였으며, 특히 반사 섬세한 장면에서 재구성 정확도를 향상시켰습니다. 



### MetaSeg: MetaFormer-based Global Contexts-aware Network for Efficient Semantic Segmentation (https://arxiv.org/abs/2408.07576)
Comments:
          Accepted by WACV 2024

- **What's New**: MetaSeg라는 강력한 시맨틱 세분화 네트워크를 제안합니다. 이 네트워크는 MetaFormer 아키텍처를 기반으로 하여, 백본(backbone)부터 디코더(decoder)까지의 전체 구조에서 활용됩니다.

- **Technical Details**: MetaFormer 블록을 사용하는 CNN 기반 백본과 글로벌 컨텍스트를 캡처하기 위해 새로운 셀프 어텐션 모듈인 Channel Reduction Attention (CRA)를 설계합니다. CRA는 쿼리(query)와 키(key)의 채널 차원을 1차원으로 줄여, 계산 효율성을 높입니다.

- **Performance Highlights**: 제안된 MetaSeg는 ADE20K, Cityscapes, COCO-Stuff 등 여러 세분화 벤치마크에서 이전의 최첨단 방법들을 능가하며, SegNeXt-T보다 각각 1.3%, 0.3%, 1.0% mIoU 개선을 보이고, 계산 비용은 각각 16.7%, 5.2%, 16.7% 감소합니다.



### MathScape: Evaluating MLLMs in multimodal Math Scenarios through a Hierarchical Benchmark (https://arxiv.org/abs/2408.07543)
- **What's New**: 새로운 벤치마크 MathScape가 도입되어 MLLM의 수학적 문제 해결 능력을 평가하는 데 초점을 맞추고 있으며, 시각적 정보와 텍스트 정보를 통합한 평가 방식이 특징입니다.

- **Technical Details**: MathScape는 사진 기반의 수학 문제 시나리오를 평가하며, 이론적 이해와 적용 능력을 계층적 접근 방식으로 측정합니다. 또한, LLMs를 통해 서브 문제의 답변을 추출하고, 각 솔루션의 정확성을 평가하는 2단계 평가 방법을 도입하였습니다.

- **Performance Highlights**: 실험에서는 11개의 고급 MLLM에 대해 다차원적인 평가를 실시하였으며, MathScape 벤치마크가 가장 진보된 모델들에게도 도전 과제가 된다는 사실을 확인했습니다.



### DifuzCam: Replacing Camera Lens with a Mask and a Diffusion Mod (https://arxiv.org/abs/2408.07541)
- **What's New**: 본 논문에서는 렌즈가 없는 플랫 카메라 디자인을 제안하며, 열화되는 이미지를 복원하기 위해 사전 훈련된 diffusion 모델을 사용하는 혁신적인 접근 방식을 소개합니다.

- **Technical Details**: Diffusion 모델을 활용함으로써 플랫 카메라의 복원 품질을 향상시키며, 텍스트 안내 기능을 통해 이미지 복원을 더욱 개선합니다. 이 방법은 이미지 생성을 위한 강력한 우선 정보를 제공하고, 고해상도 이미지를 보다 효율적으로 처리할 수 있도록 합니다.

- **Performance Highlights**: 제안된 DifuzCam은 기존의 영상 측정 방법과 비교하여 품질 및 인지적 측면에서 최첨단 결과를 보여줍니다. 이 방법은 다른 이미징 시스템에도 적용 가능하여 다양한 환경에서의 복원 결과를 개선할 수 있습니다.



### 3D Gaussian Editing with A Single Imag (https://arxiv.org/abs/2408.07540)
Comments:
          10 pages, 12 figures

- **What's New**: 본 논문은 단일 이미지 기반의 3D 장면 편집 방법을 새롭게 제안합니다. 이 방법은 3D Gaussian Splatting을 활용하여 2D 이미지 평면에서 내용을 직접 편집함으로써 직관적인 조작을 가능하게 합니다.

- **Technical Details**: 우리는 편집된 이미지를 원본 장면의 사용자가 지정한 시점에서 렌더링한 이미지와 정렬하고 조작하기 위해 3D Gaussians를 최적화하는 방식을 사용합니다. 이를 위해 위치 손실(positional loss)을 도입하여 긴 거리 객체 변형을 캡처하고, 재매개변수를 통해 기울기 전파를 가능하게 합니다. 또한, 고정 구조를 위한 앵커 기반 구조를 구축하고, 장거리 변형을 처리하면서 구조적 안정성을 유지하는 정밀한 최적화 전략을 활용합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 기하학적 세부 사항, 장거리 및 비강체(non-rigid) 변형을 처리하는 데 효과적임을 입증했으며, 이전 방법에 비해 뛰어난 편집 유연성과 품질을 보여주었습니다.



### Cross-aware Early Fusion with Stage-divided Vision and Language Transformer Encoders for Referring Image Segmentation (https://arxiv.org/abs/2408.07539)
Comments:
          Published in IEEE Transactions on Multimedia (TMM)

- **What's New**: 본 논문에서는 Cross-aware early fusion with stage-divided Vision and Language Transformer encoders (CrossVLT)라는 혁신적인 아키텍처를 제안하여 참조 이미지 분할(referring image segmentation) 문제에 접근합니다. 이 모델은 언어 및 비전 인코더가 서로 정보를 참조할 수 있도록 하여 두 인코더의 강건함을 상호 증진시킵니다.

- **Technical Details**: CrossVLT는 각 단계에서 언어와 비전 인코더 간의 교차 정보를 양방향으로 교환할 수 있도록 설계되었습니다. 저수준(low-level)에서 고수준(high-level) 특성까지 활용하여 교차 모달 정렬(cross-modal alignment)을 수행하는 기능 기반 정렬 방식을 도입합니다. 이를 통해 각 단계에서 보다 효과적인 교차 모달 융합(cross-modal fusion)이 이루어집니다.

- **Performance Highlights**: CrossVLT는 세 가지 공개 데이터셋에서 이전의 최첨단 방법들을 능가하는 성능을 달성하였으며, 참조 이미지 분할에 대한 단순하지만 효과적인 접근 방식을 제공합니다.



### Towards Real-time Video Compressive Sensing on Mobile Devices (https://arxiv.org/abs/2408.07530)
Comments:
          9 pages, Accepted by ACM MM 2024

- **What's New**: 이번 논문에서는 최초로 모바일 장치에서 실시간으로 작동할 수 있는 비디오 스냅샷 압축 영상(Video Snapshot Compressive Imaging, SCI) 복원 방법인 MobileSCI를 제시합니다. 이는 기존 고성능 비디오 SCI 복원 알고리즘을 모바일 환경에 적합하도록 개발한 혁신적인 접근입니다.

- **Technical Details**: MobileSCI는 U자형 2D 컨볼루션 아키텍처를 기반으로 설계되어 기존 최첨단 복원 방법들보다 효율적이고 모바일 친화적입니다. 또한, 채널 분할(channel splitting) 및 셔플링(shuffling) 메커니즘을 활용한 효율적인 피쳐 믹싱 블록을 도입하여 계산 부담을 줄입니다. 최적화된 지식 증류(knowledge distillation) 전략을 통해 복원 품질을 추가로 개선합니다.

- **Performance Highlights**: 모바일 장치에서 고속으로 256 X 256 X 8 크기의 스냅샷 압축 측정값을 약 35 FPS의 실시간 성능으로 복원 가능하다는 점이 특히 주목할 만하며, 이는 iPhone 15에서 수행되었습니다.



### Evidential Graph Contrastive Alignment for Source-Free Blending-Target Domain Adaptation (https://arxiv.org/abs/2408.07527)
- **What's New**: SF-BTDA(소스 없는 혼합 대상 도메인 적응) 설정을 도입하고, 노이즈가 포함된 대상 의사 레이블의 영향을 완화하기 위한 새로운 방법인 ECA(증거 대조 정렬)를 제안합니다.

- **Technical Details**: ECA는 두 가지 주요 도전 과제를 해결하기 위해 설계되었습니다: (1) 고품질 의사 대상 레이블을 생성하기 위한 보정된 증거 학습 모듈과 (2) 혼합된 대상 도메인 간의 분포 차이를 최소화하기 위한 그래프 대조 학습을 통해 클래스 샘플의 분포를 조정합니다.

- **Performance Highlights**: ECA는 새로운 벤치마크를 기반으로 실험을 수행하고, 다른 방법들보다 상당한 성능 향상을 기록하며 도메인 레이블이나 소스 데이터에 접근하는 경우와 비슷한 결과를 달성합니다.



### Whitening Consistently Improves Self-Supervised Learning (https://arxiv.org/abs/2408.07519)
Comments:
          Preprint

- **What's New**: 이 연구에서는 self-supervised learning (SSL)에서 인코더의 마지막 층으로 ZCA whitening을 통합하여 학습된 특성의 품질을 향상시키는 방법을 제안합니다. 기존 연구에서 whitening이 SSL에 활용되었지만, 모든 SSL 모델을 보편적으로 개선할 수 있는 잠재력은 탐구되지 않았습니다.

- **Technical Details**: ZCA whitening은 인코더의 마지막 층으로 도입되어 SSL 방법론과 인코더 아키텍처에 독립적으로 작용하여, 다양한 SSL 방법론 및 데이터셋에 대해 성능을 향상시킵니다. 이를 통해 learned representations를 분석하기 위해 conventional linear probing과 k-nearest neighbor probing을 사용했습니다.

- **Performance Highlights**: 실험 결과, whitening을 적용함으로써 CIFAR10, STL10 및 Tiny-ImageNet 데이터셋에서 linear 및 k-NN probing 정확도가 1-5% 향상되었습니다. 또한, 제안된 메트릭스를 통해 학습된 특성을 포괄적으로 분석하는 프레임워크를 제공합니다.



### DIffSteISR: Harnessing Diffusion Prior for Superior Real-world Stereo Image Super-Resolution (https://arxiv.org/abs/2408.07516)
- **What's New**: DiffSteISR는 저해상도 스테레오 이미지에서 고해상도의 고품질 스테레오 이미지를 복원하기 위해 혁신적인 방법론을 제안합니다.

- **Technical Details**: DiffSteISR는 미리 훈련된 텍스트-이미지 모델에 내장된 강력한 사전 지식을 활용하여 저해상 스테레오 이미지에서 잃어버린 텍스처 세부정보를 효과적으로 복구합니다. 이 프레임워크는 시간 인식 스테레오 크로스 어텐션과 온도 어댑터를 결합한 TASCATA를 구현하여 생성된 왼쪽 및 오른쪽 뷰 간의 고텍스처 일관성을 보장합니다. 또한, 스테레오 온미 어텐션 제어 네트워크(SOA ControlNet)를 도입하여 픽셀, 지각 및 분포 공간에서 GT 이미지와의 일관성을 강화합니다.

- **Performance Highlights**: DiffSteISR는 저해상도 스테레오 이미지에서 자연스럽고 정밀한 텍스처를 정확하게 복원하며, 왼쪽 및 오른쪽 뷰 간의 의미론적 및 텍스처 일관성을 유지합니다. 광범위한 실험 결과에 따르면 DiffSteISR는 합성 및 실제 데이터셋 모두에서 경쟁력 있는 성과를 달성합니다.



### CNN-JEPA: Self-Supervised Pretraining Convolutional Neural Networks Using Joint Embedding Predictive Architectur (https://arxiv.org/abs/2408.07514)
Comments:
          Preprint

- **What's New**: 새로운 자기 지도 학습(self-supervised learning) 방법인 CNN-JEPA가 제안되었습니다. 이 방법은 공동 임베딩 예측 아키텍처를 CNN(Convolutional Neural Networks)에 성공적으로 적용하며, 이를 통해 I-JEPA를 CNN에 적응시키는 데 있어 독특한 문제를 해결합니다.

- **Technical Details**: CNN-JEPA는 희소 CNN 인코더(sparse CNN encoder)를 사용하여 마스킹된 입력을 처리하며, 깊이별 분리 합성곱(depthwise separable convolutions)을 사용하는 완전 합성곱 예측기(fully convolutional predictor)를 포함합니다. 이 방법은 I-JEPA의 마스킹 전략을 개선하여 처리합니다.

- **Performance Highlights**: CNN-JEPA는 ImageNet-100에서 ResNet-50 인코더를 사용해 73.3%의 선형 top-1 정확도를 달성하며, I-JEPA에 비해 17-35% 적은 훈련 시간이 소요됩니다. 이 방법은 기존 Self-Supervised Learning 방법들에 비해 더 간단하고 효율적인 대안을 제공합니다.



### Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach (https://arxiv.org/abs/2408.07500)
- **What's New**: 본 논문에서는 Ground-to-Aerial Video 기반의 사람 재식별(person Re-Identification)을 위한 대규모 벤치마크 데이터셋인 G2A-VReID를 구축했습니다. 이 데이터셋은 총 185,907개 이미지와 5,576개의 트랙릿(tracklet)을 포함하며, 2,788개의 고유한 신원(identity)을 특징으로 합니다. G2A-VReID는 이러한 초점 변화, 많은 주석 달린 신원, 다양한 야외 시나리오 및 해상도 차이가 있다는 독특한 특성을 가지고 있습니다.

- **Technical Details**: G2A-VReID 데이터셋을 통해 Cross-Platform ReID를 위한 새로운 벤치마크 접근 방식을 제안합니다. 이를 위해 cross-platform 시각 정렬 문제를 시각-문맥 정렬(visual-semantic alignment)으로 변환하고, 이미지 기반의 기초 모델이 비디오 ReID 작업에 적응하도록 하는 파라미터 효율적인 Video Set-Level-Adapter(이하 VSLA) 모듈을 적용합니다. 또한, 플랫폼 간의 전시적 피처 정렬을 위해 플랫폼-브리지 프롬프트를 설계하여 큰 불일치를 줄입니다.

- **Performance Highlights**: 제안된 방법은 기존의 모든 비디오 ReID 데이터셋 및 새로운 G2A-VReID 데이터셋에서 우수한 성능을 보여주었으며, 기존 데이터셋에 비해 훨씬 더 향상된 결과를 도출했습니다.



### Attention-Guided Perturbation for Unsupervised Image Anomaly Detection (https://arxiv.org/abs/2408.07490)
- **What's New**: 본 논문은 Attention-Guided Perturbation Network (AGPNet)라는 기존의 비지도 이상 탐지 기법에 대한 간단하면서도 효과적인 재구성 프레임워크를 소개합니다. AGPNet은 주의(attention) 마스크를 활용해 이상 샘플의 재구성을 잘못하는 기존의 문제를 완화하는 것을 목표로 합니다.

- **Technical Details**: AGPNet은 단순한 재구성 가지와 보조 주의 기반의 섭동 가지의 두 가지로 구성됩니다. 보조 가지는 정상 샘플에 대한 섭동 과정을 안내할 주의 마스크를 생성합니다. 이 과정에서 중요한 위치의 중요성에 따라 섭동을 가하여 재구성 네트워크가 더 집중적으로 정상 패턴을 학습할 수 있도록 합니다.

- **Performance Highlights**: AGPNet은 MVTec-AD, VisA, MVTec-3D와 같은 세 가지 인기 벤치마크에서 광범위한 실험을 수행하였으며, 미세 샷(few-shot), 일급(one-class), 다급(multi-class) 설정을 포함한 다양한 설정에서 최고의 이상 탐지 성능을 달성했습니다. 예를 들어, MVTec-AD에서 500 에포크의 훈련으로 P-AUC 98.0% 및 I-AUC 98.7%를 달성하여 기존 방법인 UniAD보다 성능이 향상되었습니다.



### OMR: Occlusion-Aware Memory-Based Refinement for Video Lane Detection (https://arxiv.org/abs/2408.07486)
Comments:
          Accepted to ECCV 2024

- **What's New**: 본 논문에서는 새로운 비디오 차선 검출 알고리즘을 제안합니다. 이 알고리즘은 현재 프레임의 특징 맵을 추출하고, 차선을 가리는 장애물의 잠재 마스크를 검출한 후, OMR(occlusion-aware memory-based refinement) 모듈을 통해 특징 맵을 향상시킵니다.

- **Technical Details**: 제안된 OMR 모듈은 장애물 마스크와 현재 프레임의 특징 맵, 이전 출력, 메모리 정보를 입력으로 받아 비디오 내에서 재귀적으로 처리합니다. 이 알고리즘은 훈련을 위해 새로운 데이터 증강(data augmentation) 기법을 적용하여 효과적으로 OMR 모듈을 학습시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 기존 기술들보다 VIL-100 및 OpenLane-V 데이터셋에서 우수한 차선 검출 성능을 보였습니다.



### GRFormer: Grouped Residual Self-Attention for Lightweight Single Image Super-Resolution (https://arxiv.org/abs/2408.07484)
Comments:
          Accepted for ACM MM 2024

- **What's New**: 본 연구에서는 GRFormer라는 새로운 경량의 단일 이미지 초해상도(SISR) 모델을 제안합니다. 이전의 Transformer 기반 SISR 모델(예: SwinIR)에서는 매개변수와 계산의 감소가 성능을 감소시켰으나, GRFormer는 경량화와 성능 향상을 동시에 달성하였습니다.

- **Technical Details**: GRFormer는 Grouped Residual Self-Attention (GRSA)를 중심으로 구성되어 있으며, 새로운 Grouped Residual Layer (GRL)를 도입하여 Query, Key, Value (QKV) 선형 계층을 대체합니다. 또한 Exponential-Space Relative Position Bias (ES-RPB)를 통합하여 매개변수 수를 줄이면서 위치 정보 표현 능력을 향상시킵니다.

- **Performance Highlights**: GRFormer는 DIV2K 데이터셋에서 훈련된 결과, PSNR에서 0.23dB 향상된 성능을 기록하며, SOTA 모델들보다 우수한 성능을 보입니다. 특히 Urban100 데이터셋에서 ×2 SISR 작업을 수행할 때 33.17 PSNR을 기록했습니다. 이 모델은 SR 모델의 자기 주의(modules)에서 약 60%의 매개변수 및 약 49%의 MACs 감소를 달성했습니다.



### DeCo: Decoupled Human-Centered Diffusion Video Editing with Motion Consistency (https://arxiv.org/abs/2408.07481)
Comments:
          European Conference on Computer Vision

- **What's New**: 이번 논문에서는 DeCo라는 새로운 비디오 편집 프레임워크를 소개합니다. 이 프레임워크는 인간(스포츠와 관련된 목적)의 움직임과 배경을 각각 독립적으로 편집 가능한 대상으로 다룹니다.

- **Technical Details**: DeCo는 파라메트릭(Parametric) 인간 몸체 사전(prior)을 활용한 분리형(dynamic) 인간 표현을 제안합니다. 이를 통해 원본 비디오의 일관된 움직임을 유지하면서 맞춤형 인간을 생성합니다. 배경은 레이어드 아틀라스(layered atlas)로 고려되며, 텍스트 기반의 이미지 편집 기법을 적용합니다. 또한, 점수 증류 샘플링(score distillation sampling)의 계산을 노멀(normal) 공간 및 이미지 공간으로 확장하여 인간의 기하학과 질감을 향상시키고, 조명 일관성 문제를 해결하기 위해 조명 인식 비디오 하모나이저(light-aware video harmonizer)를 활용합니다.

- **Performance Highlights**: DeCo는 이전의 비디오 편집 방법들에 비해 인간 중심의 비디오에서 월등한 성능을 보여줍니다. 특히 긴 비디오에서 눈에 띄는 개선이 있습니다.



### One Step Diffusion-based Super-Resolution with Time-Aware Distillation (https://arxiv.org/abs/2408.07476)
Comments:
          18 pages

- **What's New**: 이번 논문에서는 제안된 TAD-SR 기법을 통해 단일 샘플링 단계에서 고해상도 이미지를 생성할 수 있는 혁신적인 접근 방식이 소개되었습니다. TAD-SR은 시간 인식(score distillation) 기법을 도입하여 학생 모델의 출력이 교사 모델의 출력과 일치하도록 하여 데이터 배포를 향상시킵니다.

- **Technical Details**: TAD-SR은 작은 시간 단계에서 학생 모델과 교사 모델의 출력 점수 차이를 정량화하는 새로운 스코어 증류(score distillation) 방법을 사용합니다. 또한, 생성적 적대 학습(generative adversarial learning)을 결합하여 학생 모델이 실제 이미지 축의 패턴을 따르는 샘플을 직접 생성하도록 유도합니다. 이 과정에서 시간 인식(discriminator) 모델을 설계하여 다양한 변동성을 겪은 진짜 이미지와 생성된 이미지를 구분합니다.

- **Performance Highlights**: TAD-SR 방법은 합성 및 실제 데이터 세트를 대상으로 한 실험에서 이전의 최첨단(SOTA) 기법이나 교사 모델 대비 동등하거나 더 나은 성능을 보여 주었습니다. 특히 단일 샘플링 단계 만으로 수행되어 효율성을 극대화했습니다.



### Domain-invariant Representation Learning via Segment Anything Model for Blood Cell Classification (https://arxiv.org/abs/2408.07467)
- **What's New**: 본 논문은 혈액 세포 분류를 위한 새로운 프레임워크인 domain-invariant representation learning (DoRL)을 제안합니다. 이 프레임워크는 segment anything model (SAM)과 cross-domain autoencoder (CAE)를 활용하여 혈액 세포 데이터셋에서 도메인 불변 표현을 추출합니다.

- **Technical Details**: DoRL은 LoRA 기반의 SAM(LoRA-SAM)과 cross-domain autoencoder (CAE)로 구성됩니다. SAM을 사용하여 일반 이미지 임베딩을 학습하고 혈액 세포를 분할한 뒤, CAE를 통하여 다양한 도메인 데이터셋 간의 도메인 불변 표현을 학습합니다. 이를 통해 이미지의 아티팩트를 완화합니다.

- **Performance Highlights**: 두 개의 공개 혈액 세포 데이터셋과 하나의 개인 데이터셋을 사용한 실험 결과, 본 연구의 DoRL이 기존 방법보다 우수한 성능을 발휘하며 cross-domain 혈액 세포 분류에서 새로운 최고 성과를 달성하였음을 입증하였습니다.



### Infra-YOLO: Efficient Neural Network Structure with Model Compression for Real-Time Infrared Small Object Detection (https://arxiv.org/abs/2408.07455)
- **What's New**: 본 논문에서는 인프라 소형 물체 탐지를 위해 새로운 데이터셋인 InfraTiny를 구축하고, MSAM과 FFAFPM이라는 두 가지 새로운 모듈을 제안합니다. InfraTiny 데이터셋은 3,218장의 적외선 이미지를 포함하며, 85%의 bounding box는 32x32 픽셀 이하입니다.

- **Technical Details**: 제안된 MSAM(멀티 스케일 어텐션 메커니즘)은 다양한 receptive field를 이용해 스케일 인식 정보를 획득하도록 하며, FFAFPM(특징 융합 증대 피라미드 모듈)은 얕은 특징과 깊은 특징의 융합을 강화하여 탐지 성능을 향상시킵니다.

- **Performance Highlights**: Infra-YOLO 모델은 InfraTiny 데이터셋에서 yolov3 대비 mAP@0.5가 2.7% 향상되었고, yolov4 대비 2.5% 향상되었습니다. 또한, UAV에 통합하여 실제 시나리오에서도 성능을 검증하였으며, 채널 프루닝 방법을 통해 FLOPs를 줄이면서도 yolov3 대비 0.7%, yolov4 대비 0.5%의 성능 개선을 달성하였습니다.



### Modality Invariant Multimodal Learning to Handle Missing Modalities: A Single-Branch Approach (https://arxiv.org/abs/2408.07445)
- **What's New**: 이번 연구에서는 결측된 (missing) 모달리티에 대한 강인성을 갖춘 단일 분기 네트워크를 제안하는 SRMM (Single-branch approach Robust to Missing Modalities) 방법을 제시했습니다. 이는 멀티모달 학습에서 결측된 모달리티의 영향을 최소화할 수 있습니다.

- **Technical Details**: SRMM은 다양한 모달리티 간의 weight 공유를 통해 inter-modality representations를 학습하는 단일 분기 구조를 활용하면서, 사전 학습된 임베딩을 사용하여 훈련을 수행합니다. 이 접근법은 여러 도전적인 데이터셋에서 좋은 성과를 보였으며, 특히 결측 모달리티 상황에서도 강인함을 보여줍니다.

- **Performance Highlights**: SRMM은 UPMC Food-101 데이터셋에서 이미지와 텍스트 모달리티가 모두 존재할 때 94.6%의 분류 정확도를 달성하였으며, 텍스트 모달리티가 30%만 존재할 때에도 84.9%의 정확도를 기록하여 기존의 최신 기술인 ViLT보다 우수한 성능을 보였습니다.



### BAPLe: Backdoor Attacks on Medical Foundational Models using Prompt Learning (https://arxiv.org/abs/2408.07440)
Comments:
          MICCAI 2024

- **What's New**: 최근 연구에 따르면, 의학 기초 모델(Med-FM)이 데이터 희소성 덕분에 후방 공격(backdoor attack)에 취약하다는 사실이 밝혀졌습니다. 이 연구에서는 프롬프트 학습(prompt learning) 단계에서 미세한 프롬프트를 사용하여 후방 공격을 삽입하는 새로운 방법인 BAPLe를 제안합니다.

- **Technical Details**: 제안하는 방법 BAPLe는 의학 기초 모델의 입력 공간에 소량의 학습 가능한 프롬프트를 추가합니다. 이 프롬프트는 오염된 데이터셋을 통해 최적화되어 후방 공격을 효과적으로 삽입하며, 모델의 백본(backbone)은 동결되어 대량의 데이터나 컴퓨팅 자원 없이작동합니다. 실험 결과, Kather 데이터셋에서 8888개의 오염된 샘플을 사용하여 90% 이상의 높은 성공률을 기록했습니다.

- **Performance Highlights**: BAPLe는 4개의 의학 기초 모델과 6개의 다운스트림 데이터셋에서 우수한 성능을 보여주며, 전통적인 미세 조정 방법에 비해 GPU 사용량을 33%-35% 줄인 뛰어난 효율성을 입증했습니다.



### MagicFace: Training-free Universal-Style Human Image Customized Synthesis (https://arxiv.org/abs/2408.07433)
Comments:
          project page: this https URL

- **What's New**: MagicFace는 고유의 스타일을 기반으로 한 인간 이미지를 개인화하여 생성하는 새로운 접근 방식으로, 훈련 없이 단일 및 다중 개념을 커스터마이징할 수 있는 첫 번째 방법입니다.

- **Technical Details**: MagicFace는 세 가지 단계로 구성된 생성 파이프라인을 통해 작동합니다: 첫 번째 단계에서는 Reference-aware Self-Attention (RSA)을 사용하여 참조 개념으로부터 특징을 추출하고, 두 번째 단계에서는 Region-grouped Blend Attention (RBA)을 통해 생성된 이미지의 각 개념이 갖는 세부 특징을 정확히 주입합니다. 이 모든 과정에서 가중치 마스크 전략을 활용하여 모델이 참조 개념에 더 집중하도록 합니다.

- **Performance Highlights**: 실험 결과, MagicFace는 인간의 이미지를 생성하는 작업뿐만 아니라 다중 개념 커스터마이징에서도 매우 우수한 성능을 보였습니다. 사용자 연구를 통해, 대부분의 참가자들이 MagicFace의 결과에 긍정적인 피드백을 주었으며, 훈련이 필요 없는 방식으로도 높은 품질의 결과를 생성할 수 있음을 입증했습니다.



### UAHOI: Uncertainty-aware Robust Interaction Learning for HOI Detection (https://arxiv.org/abs/2408.07430)
Comments:
          Accepted by CVIU

- **What's New**: 이번 논문에서는 Human-Object Interaction (HOI) 탐지를 위한 새로운 접근 방식인 UAHOI, Uncertainty-aware Robust Human-Object Interaction Learning을 제안합니다. 이는 예측 불확실성을 명시적으로 추정하여 탐지 및 상호작용 예측을 개선합니다.

- **Technical Details**: UAHOI는 예측의 분산을 통해 불확실성을 모델링하고, 이를 최적화 목표에 통합하여 신뢰도 임계값을 적응적으로 조정합니다. 이는 기존 HOI 탐지 방법의 한계를 극복하는 데 기여하며, 복잡한 상호작용에서도 높은 정확도를 유지합니다.

- **Performance Highlights**: UAHOI는 V-COCO와 HICO-DET 두 가지 표준 벤치마크에서 평가되었으며, 각각 34.19 mAP와 62.6 mAP를 기록하여 기존 최첨단 방법에 비해 유의미한 개선을 보여주었습니다. 이는 HOI 탐지의 정확성과 견고성을 향상시킵니다.



### LLMI3D: Empowering LLM with 3D Perception from a Single 2D Imag (https://arxiv.org/abs/2408.07422)
- **What's New**: 최근 자율주행, 증강현실, 로보틱스, 및 구현된 지능(embodied intelligence) 분야의 발전은 3D 인식 알고리즘(algorithms) 필요성을 증가시켰습니다. 그러나 기존의 3D 인식 방법들은 논리적 추론, 질문-답변 처리 및 개방 시나리오 범주를 다루는 데 어려움을 겪고 있습니다. 본 연구는 LLMI3D라는 강력한 3D 인식 MLLM을 제안하며, IG3D 데이터셋을 구축하여 세부 기술 및 질문-답변 주석을 제공합니다.

- **Technical Details**: LLMI3D는 사전 훈련된 MLLM에 파라미터 효율적인 미세 조정을 적용하고, 공간 강화 로컬 기능 추출(Spatial-Enhanced Local Feature Mining), 3D 쿼리 토큰 기반 정보 디코딩(3D Query Token-Derived Info Decoding), 기하학적 투영 기반 3D 추론(Geometry Projection-Based 3D Reasoning) 방법을 도입하여 카메라 초점 길이 변화를 처리하고 공간적 특징을 효과적으로 추출하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, LLMI3D는 기존의 방법들에 비해 월등한 성능을 보이며, 최신 최첨단 성능을 달성했습니다. 이 연구는 3D 인식 분야의 새로운 가능성을 열고 있으며, 다양한 현실 세계 문제 해결에 기여할 것으로 기대됩니다.



### Unsupervised Stereo Matching Network For VHR Remote Sensing Images Based On Error Prediction (https://arxiv.org/abs/2408.07419)
Comments:
          Accepted to International Geoscience and Remote Sensing Symposium (IGARSS), 2024

- **What's New**: 최근 원거리 센싱에서 스테레오 매칭(Stereo Matching)에 대한 관심이 증가하고 있습니다. 특히 감독(supervised) 학습 방법들이 주로 사용되나, 기존의 고도측정이 가능한 공중 Lidar로 생성된 데이터셋은 양과 다양성이 부족하여 효율적인 모델 학습을 제약하고 있습니다. 이러한 한계를 극복하기 위해 저자들은 비감독(unsupervised) 학습에 기반하여 새로운 스테레오 매칭 네트워크를 제안합니다.

- **Technical Details**: 제안하는 네트워크는 자신감(confidence)을 기반으로 한 에러 예측 모듈(CBEM)을 포함합니다. 이 모듈은 스테레오 매칭에서 예측된 불확실성 점수를 통해 불확실성을 평가하고, 겹침과 함께 자가 보정을 통해 오차를 줄입니다. 여러 스케일의 변화를 고려한 다단계 모델 구조를 통해 네트워크의 공동 학습을 최적화합니다.

- **Performance Highlights**: 제가 제안한 방법은 US3D 및 WHU-Stereo 데이터셋에서 실험한 결과, 다른 비감독 네트워크보다 더 높은 정확도를 달성하고 감독된 모델에 비해 더 나은 일반화 능력을 보였습니다. 이는 새로운 비감독 스테레오 매칭 방법의 효용성을 증명합니다.



### Rethinking Open-Vocabulary Segmentation of Radiance Fields in 3D Spac (https://arxiv.org/abs/2408.07416)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 NeRFs 및 3DGS에 의한 장면의 3D 이해를 향상시키기 위한 새로운 접근 방식을 제시합니다. 주요 기여는 3D 점을 직접 감독하여 언어 임베딩 필드를 훈련하는 방법, 3DGS로 사전 훈련된 언어 필드를 전이하여 실시간 렌더링 속도를 달성하는 방법, 재구성된 기하학 및 의미를 함께 평가하는 3D 쿼리 및 평가 프로토콜을 도입한 것입니다.

- **Technical Details**: 이 연구는 3D 포인트를 직접 감독함으로써 언어 임베딩 필드를 학습할 수 있도록 설정하고, 이를 통해 state-of-the-art 정확도를 달성합니다. 또한, 3DGS로의 전이 학습을 통해 다중 스케일 언어 임베딩에 의존하지 않고 27배 빠른 실시간 렌더링을 가능하게 합니다. 평가 방법론 면에서, 재구성된 기하학과 타겟 의미의 정확성을 F1 점수를 통해 평가합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 3D 및 2D 세분화 정확도, 훈련 및 렌더링 시간에서 기존 방법을 초월하며, 여러 시점에서의 일관성을 유지함을 입증하였습니다.



### Segment Using Just One Examp (https://arxiv.org/abs/2408.07393)
- **What's New**: 본 논문은 하나의 예제 이미지를 사용하여 객체 분할을 수행하는 새로운 방법을 제안합니다. 이 방법은 기존의 지도 학습 접근방식과 다르게 훈련 없이 단 하나의 예제 이미지로 특정 클래스의 분할을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 'Segment Anything' (SAM) 모델을 활용하여 개발되었습니다. 본 연구는 예제 이미지를 합성하여 쿼리 이미지를 처리하고, 여러 가지 자동 프롬프트 생성 기법을 설계하여 가능한 최적의 분할 결과를 도출합니다. 이 과정은 훈련 단계가 없으며, 텍스트 프롬프트 사용 없이 수행됩니다.

- **Performance Highlights**: 제안된 기법은 건물과 자동차 클래스에서 평가되었으며, 특히 재난 관리 시나리오에서의 응용 가능성을 언급합니다. 이 방법은 다양한 실제 응용에서의 유연성을 강조하며, 최소한의 데이터로 신속한 타겟 식별을 가능하게 합니다.



### RTAT: A Robust Two-stage Association Tracker for Multi-Object Tracking (https://arxiv.org/abs/2408.07344)
Comments:
          ICPR2024

- **What's New**: 이번 논문에서는 Multi-Object Tracking (MOT)에서의 데이터 연관(data association)의 중요성을 강조하며, 복잡한 장면에서의 일반화 능력을 향상시키기 위해 Robust Two-stage Association Tracker (RTAT)라는 새로운 방법을 제안합니다.

- **Technical Details**: RTAT는 두 단계의 연관 과정을 통해 동작합니다. 첫 번째 단계에서는 낮은 매칭 비용(threshold)을 설정하여 순수성이 높은 tracklets를 생성하고, 두 번째 단계에서는 메시지 전달(message-passing) GNN 프레임워크를 기반으로 tracklet 간의 연관을 수행합니다. 이를 통해 단기 tracklet들을 장기 tracklet로 재귀적으로 병합할 수 있습니다.

- **Performance Highlights**: RTAT는 MOT17 및 MOT20 벤치마크의 테스트 세트에서 HOTA, IDF1, AssA와 같은 주요 MOT 메트릭에서 1위를 기록했습니다. MOT17에서 67.2 HOTA, 84.7 IDF1, 69.7 AssA를 달성하였고, MOT20에서 66.2 HOTA, 82.5 IDF1, 68.1 AssA를 기록하였습니다.



### Gradient Alignment Improves Test-Time Adaptation for Medical Image Segmentation (https://arxiv.org/abs/2408.07343)
- **What's New**: 최근 의료 영상 분할의 발전에도 불구하고, 다양한 센터 간의 도메인 변화(domain shift) 문제는 사전 학습된 모델의 효과적인 배포를 방해합니다. 본 논문에서는 이러한 문제를 해결하기 위해 Gradient alignment-based Test-time adaptation (GraTa) 방법을 제안합니다.

- **Technical Details**: GraTa 방법은 최적화 절차에서 기울기 방향과 학습 속도를 개선합니다. 전통적인 TTA 방법들은 주로 자기 감독(self-supervised) 목표에서 유도된 의사 기울기(pseudo gradient)를 최적화하지만, GraTa는 보조 기울기(auxiliary gradient)를 포함하여 기울기 정렬(gradient alignment)을 촉진합니다. 이는 모델이 서로 다른 기울기 간의 유사성을 탐구하고, 현재 분할(task segmentation) 과제와 관련된 실험적 기울기(empirical gradient)에 근접하도록 기울기 방향을 수정할 수 있게 합니다. 또한, 의사 기울기와 보조 기울기 간의 코사인 유사도(cosine similarity)에 따라 동적 학습 속도를 설계하여 다양한 테스트 데이터에 대한 적응적 미세 조정(adaptive fine-tuning)을 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 GraTa 방법이 다른 최첨단 TTA 방법들보다 우수한 성능을 발휘함을 입증했습니다. 본 연구의 주요 기여는 최적화 방향 및 단계 크기를 모두 개선하는 GraTa의 도입입니다.



### Robust Semi-supervised Multimodal Medical Image Segmentation via Cross Modality Collaboration (https://arxiv.org/abs/2408.07341)
- **What's New**: 본 논문에서는 제한된 라벨 데이터와 정렬되지 않은 이미지를 효과적으로 처리할 수 있는 혁신적인 세미-슈퍼바이즈(multimodal) 세분화 프레임워크를 제안합니다. 기존의 방법들이 정밀한 주석을 요구하는 것과 달리, 본 연구는 적은 양의 라벨링된 데이터로도 강력한 성능을 발휘할 수 있는 접근법을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 cross-modality collaboration 전략과 contrastive consistent learning을 통해 각 모달리티에서 도출된 모달리티 독립적인 지식을 취합한 후, 통합된 fusion layer에 통합합니다. 이를 통해 한 특징적 관점에서 모달리티 간의 정렬을 보장하고, unlabeled 데이터에 대한 해부학적 예측 정렬을 용이하게 합니다.

- **Performance Highlights**: 제안된 방법은 심장, 복부 다기관, 갑상선 관련 안와증(semo) 세분화 작업을 포함하여 세 가지 태스크에서 기존의 multimodal 방법 대비 경쟁력 있는 성능을 달성하였으며, 제한된 라벨 데이터 및 정렬되지 않은 모달리티의 경우에서도 뛰어난 강건성을 보여줍니다.



### KIND: Knowledge Integration and Diversion in Diffusion Models (https://arxiv.org/abs/2408.07337)
- **What's New**: 논문에서는 모델 파라미터 수의 증가와 함께 사전 학습된 모델들이 선호되는 backbone으로 자리잡고 있음을 이야기하며, 새로운 접근 방식인 	extbf{KIND}를 제안합니다. KIND는 지식 통합(	extbf{K}nowledge 	extbf{IN}tegration)과 분산(	extbf{D}iversion)을 통해 diffusion models에서 효과적으로 작동합니다.

- **Technical Details**: KIND는 모델의 파라미터 행렬을 $U$, $	ext{Σ}$, $V$ 행렬로 분해하여 지식을 통합합니다. 이 과정은 특이 값 분해(singular value decomposition, SVD)에 영감을 받아 진행됩니다. 이후 이 행렬의 구성 요소를 	extbf{learngenes}와 	extbf{tailors}로 명확하게 분할하여 공통 지식과 클래스 특정 지식을 각각 조정합니다. 이를 통해 KIND는 기존의 사전 학습 방식을 재정의하며, 현재 문제에서 모델 성능 극대화 대신 전이 가능성이 있는 공통 지식을 응축하는 방향으로 학습 목표를 조정합니다.

- **Performance Highlights**: KIND는 ImageNet-1K 데이터셋에서 실험을 진행하였고, PEFT 및 다른 learngene 방법들과 비교하였습니다. 실험 결과, KIND는 다른 PEFT 및 learngene 방법들에 비해 최첨단(state-of-the-art) 성능을 달성했습니다. 특히, KIND가 생성한 이미지들은 DiT-L/2에서 FID에서 6.54 및 sFID에서 1.07의 감소를 보였으며, 단 45.4M의 학습 가능한 파라미터만을 사용하고, 최소 35.4G의 FLOPs 계산 비용을 절감했습니다.



### Enhancing Visual Question Answering through Ranking-Based Hybrid Training and Multimodal Fusion (https://arxiv.org/abs/2408.07303)
Comments:
          Visual Question Answering, Rank VQA, Faster R-CNN, BERT, Multimodal Fusion, Ranking Learning, Hybrid Training Strategy

- **What's New**: 이 논문에서는 복잡한 질문에 대해 보다 정확한 답변을 제공하기 위한 Visual Question Answering (VQA) 모델인 Rank VQA 모델을 제안합니다. 이 모델은 순위 기반의 하이브리드 훈련 전략을 사용하여 VQA 성능을 향상시킵니다.

- **Technical Details**: Rank VQA 모델은 Faster R-CNN 모델을 통해 추출된 고품질 비주얼 특성과 사전 학습된 BERT 모델로부터 얻은 풍부한 시맨틱 텍스트 특성을 통합합니다. 이러한 특성은 다중 헤드 자기 주의 메커니즘을 사용하는 복잡한 멀티모달 융합 기법을 통해 융합됩니다. 또한, 상대적 순위를 최적화하는 순위 학습 모듈이 통합되어 답변의 정확성을 개선합니다.

- **Performance Highlights**: 실험 결과, Rank VQA 모델은 VQA v2.0과 COCO-QA를 포함한 표준 VQA 데이터셋에서 기존의 최신 모델들보다 성능이 크게 향상되었습니다. 이 모델은 복잡한 질문을 처리하는 데 효과적이며, 이미지와 텍스트에서 미묘한 세부 사항을 이해하고 정교한 추론을 수행할 수 있는 능력을 보여줍니다.



### Image-Based Leopard Seal Recognition: Approaches and Challenges in Current Automated Systems (https://arxiv.org/abs/2408.07269)
Comments:
          28th International Conference on Image Processing, Computer Vision, & Pattern Recognition (IPCV'24), Las Vegas, USA

- **What's New**: 본 논문은 자연 서식지에서 물개를 인식하는데 있어 기존 사진 촬영의 한계와 기계 학습(ML) 기술의 발전을 탐구합니다. 특히 남극 생태계에서 중요한 종인 표범물개(Leopard seal, Hydrurga leptonyx)에 중점을 두어, 전통적 데이터 수집 방법의 노동 집약적이고 시간 소모적인 과정의 한계를 극복하기 위한 기계 학습의 활용을 강조합니다.

- **Technical Details**: 이 연구는 Computer Vision(CV) 기술을 통한 물개 인식의 최신 발전 상황을 분석합니다. 특히, detection(탐지) 및 segmentation(세분화) 과정을 개선하기 위한 vision transformers(ViT)와 Convolutional Neural Networks(CNNs)와 같은 첨단 모델을 다룹니다. 이 논문은 또한 일반적인 RGB 색상 모델을 기반으로 한 기존 사진의 분석 기준을 설정하고, 이를 통해 물개와 같은 동물의 위치를 정확히 식별하는 방법을 제시합니다.

- **Performance Highlights**: 물개 감지 기술의 최신 동향을 정리한 결과, 자동 식별 방법이 기존의 수동 방법보다 더 높은 효율성과 정확성을 제공함을 알 수 있습니다. 전통적 사진을 이용한 물개 탐지 방법론은 서로 다른 연구들의 컷오프 및 효율성에서 두드러진 차이를 보이며, 특정 조건에서의 성공적인 적용 사례를 제시하고 있습니다. 크게 두 가지 측면에서, 무인 자동 방법과 인간 주도 방법 간의 차이를 강조하며, 각각의 한계와 장점이 논의됩니다.



### Enhanced Scale-aware Depth Estimation for Monocular Endoscopic Scenes with Geometric Modeling (https://arxiv.org/abs/2408.07266)
- **What's New**: 본 논문에서는 단일 이미지를 사용하여 단일 카메라 내시경 장면의 scale-aware depth를 효율적으로 추정하는 새로운 프레임워크를 제안합니다. 우리는 다중 해상도 깊이 융합 전략을 사용하여 질 높은 깊이 추정을 개선하며, 기하학적 모델링을 통해 기구의 3D 포즈를 계산하여 상대 깊이와 실제 스케일 간의 정확한 관계를 회복합니다.

- **Technical Details**: 제안된 방법은 단일 이미지를 바탕으로 기하학적 프리미티브(geometric primitives)를 활용하여 수술 도구의 3D 포즈를 계산합니다. 이를 통해 상대 깊이 맵과 실제 세계 간의 스케일을 회복하여, 전체적인 깊이 맵을 구축하는 것이 가능합니다. 상대 깊이 추정 네트워크는 Monodepth2를 기반으로 하며, 저해상도 이미지를 이용한 깊이 추정과 고해상도 이미지에서의 세부정보를 융합하여 품질을 개선합니다.

- **Performance Highlights**: 시뮬레이션 데이터와 자체 내시경 수술 비디오를 통해 평가한 결과, 제안된 방법이 기존의 스테레오 기반 방법에 비해 우수한 성능을 보이며, 단일 카메라 내시경 장면에서 절대적 스케일을 학습하고 정확한 scale-aware depth를 추정할 수 있음을 입증하였습니다.



### Ensemble architecture in polyp segmentation (https://arxiv.org/abs/2408.07262)
- **What's New**: 본 연구에서는 폴립 세분화 (polyp segmentation)에 뛰어난 모델들을 평가하고, 다양한 모델의 장점을 활용하는 통합 프레임워크를 제안합니다. CNN(convolutional neural networks)과 transformer 모델에서 학습된 특징을 융합하여 최적 예측 결과를 달성하는 방식을 앙상블 기법으로 간주합니다.

- **Technical Details**: 제안된 아키텍처는 CNN과 transformer의 학습 특징을 결합하여 복합적인 세분화 성능을 구현합니다. CNN은 지역 구조를 잘 포착하고, transformer는 전역 정보를 잘 모델링합니다. 연구에서는 Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB 등 5개의 데이터셋에 대해 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다른 기존 모델들에 비해 더 높은 학습 능력과 강인성을 보여주었습니다. 특히, CNN과 transformer의 장점을 모두 활용하여 폴립 세분화 성능을 크게 향상시켰습니다.



### GRIF-DM: Generation of Rich Impression Fonts using Diffusion Models (https://arxiv.org/abs/2408.07259)
Comments:
          Accepted to ECAI2024

- **What's New**: 이번 연구에서는 레터와 감정을 정의하는 키워드 집합을 입력으로 활용하여 특정 인상을 생생하게 표현하는 폰트를 생성하는 새로운 확산 기반 방법인 GRIF-DM을 제안합니다. 이는 기존 GAN 기반 방법에서 관찰되는 추가적인 보조 손실의 필요성을 없애면서 인상 폰트 생성을 위한 혁신적인 접근법입니다.

- **Technical Details**: GRIF-DM은 U-Net 아키텍처를 설계하고, Dual Cross-Attention 모듈을 도입하여 문자와 인상 키워드 정보를 효과적으로 통합합니다. 입력으로 한 글자와 여러 개의 인상 키워드를 받아들이며, 이를 통해 사용자 맞춤형 폰트를 생성할 수 있는 능력을 갖추고 있습니다. 또한, 사전 훈련된 BERT 모델을 사용하여 문자열 형태로 결합한 인상 키워드를 임베딩하여 사용합니다.

- **Performance Highlights**: 실험 결과, GRIF-DM은 사용자 특정 요구 사항에 맞춰 사실적이고 생동감 있으며 고충실도의 폰트를 생성할 수 있음을 입증하였습니다. 현재 사용자는 270,000개 이상의 다양한 폰트 디자인에 접근할 수 있지만, GRIF-DM은 더 높은 유연성과 저항력을 제공하여 꿈의 폰트 생성을 가능하게 합니다.



### GQE: Generalized Query Expansion for Enhanced Text-Video Retrieva (https://arxiv.org/abs/2408.07249)
Comments:
          18 pages including appendix

- **What's New**: 본 논문은 'Generalized Query Expansion (GQE)'라는 새로운 데이터 중심의 접근 방식을 소개하여 텍스트와 비디오 간의 정보 불균형 문제를 해결하고, 텍스트-비디오 검색 시스템의 효과성을 향상시키려 한다.

- **Technical Details**: GQE는 훈련 및 테스트 단계 동안 비디오와 연결된 텍스트 쿼리를 확장하여 정보 불균형을 해소한다. GQE는 비디오를 짧은 클립으로 적응적으로 분할하고, 'zero-shot captioning'을 활용하여 보다 포괄적인 장면 설명으로 훈련 데이터셋을 풍부하게 만든다. 검색 시에는 대형 언어 모델('Large Language Models', LLM)을 사용하여 다양한 쿼리를 생성하고, 적합성과 다양성을 기반으로 쿼리 선택 모듈을 통해 이러한 쿼리를 필터링한다.

- **Performance Highlights**: GQE는 MSR-VTT, MSVD, LSMDC, VATEX 등 여러 벤치마크에서 최첨단 성능을 달성하여 텍스트-비디오 검색 문제를 데이터 중심의 관점에서 효과적으로 접근할 수 있음을 입증한다.



### Sign language recognition based on deep learning and low-cost handcrafted descriptors (https://arxiv.org/abs/2408.07244)
Comments:
          28 pages, 12 figures, submitted to Image and Vision Computing Journal

- **What's New**: 본 연구는 저비용 센서를 활용한 효율적인 수화 인식 시스템을 제안합니다. 기존 시스템에서의 복잡성과 높은 계산 비용을 감소시키는 새로운 기법을 도입하였습니다.

- **Technical Details**: 제안된 시스템은 특수 목적의 객체 감지 모델을 훈련시켜 해석자의 얼굴과 손을 정확히 감지하고, 이미지에서 중요한 영역에 집중합니다. 또한, 바운딩 박스의 중심 점에서 파생된 공간 정보를 활용하여 손 위치와 이동을 나타내는 핸드크래프트 피쳐를 생성했습니다.

- **Performance Highlights**: AUTSL 데이터셋에서 7.96%의 정확도 향상을 이루었으며, 700,000개 미만의 파라미터로 10밀리초 이하의 추가 추론 시간만 소요합니다. 이로 인해 계산 비용과 정확성 간의 균형을 이룰 수 있는 가능성을 보여줍니다.



### Leveraging Perceptual Scores for Dataset Pruning in Computer Vision Tasks (https://arxiv.org/abs/2408.07243)
Comments:
          1st workshop on Dataset Distillation CVPR 2024

- **What's New**: 이 연구에서는 이미지 분류 및 의미론적 분할(Semantic Segmentation) 작업을 위한 코어셋(selection for coreset) 선택에 사용할 수 있는 이미지의 점수를 제안합니다. 이 점수는 이미지를 압축한 버전의 비트당 픽셀(bits-per-pixel)로 근사된 엔트로피(Entropy)입니다. 이렇게 함으로써 감독 학습 없이도 간단하게 계산할 수 있으며, 이미지의 지각적 복잡성을 잘 포착할 수 있습니다.

- **Technical Details**: 제안된 점수 BPP (bits-per-pixel)는 JPEG로 인코딩된 이미지의 바이트 크기를 이미지의 차원으로 나누어 계산됩니다. 연구에서 사용된 데이터셋은 CIFAR, VOC, ADE20K로, 이들은 대체로 좋은 품질의 이미지를 포함하고 있습니다. 엔트로피는 이미지의 복잡성을 측정하는 데 유용한 지표로, 특히 혼잡한 이미지일수록 높은 엔트로피 값을 가집니다. 이 연구는 BPP 점수를 사용하여 선택된 샘플의 공간적 다양성을 높일 수 있는 그래프 기반 방법을 적용합니다.

- **Performance Highlights**: BPP 점수와 그래프 기반 샘플링을 결합하여 이미지 분류와 의미론적 분할에서 경쟁력 있는 성능을 보여주었습니다. 특히 의미론적 분할 작업에서는 무작위 프루닝(random pruning)보다 훨씬 더 우수한 결과를 얻었습니다. 현재로서는 이 작업을 위한 다른 데이터 프루닝 방법이나 점수에 대한 정보는 알려지지 않고 있습니다.



### Enhancing Autonomous Vehicle Perception in Adverse Weather through Image Augmentation during Semantic Segmentation Training (https://arxiv.org/abs/2408.07239)
- **What's New**: 이 논문은 자율주행 차량의 내비게이션 및 위치 확인에서 강력한 인식(robust perception)의 중요성을 강조하며, 다양한 날씨 조건을 고려한 세분화 모델의 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구진은 CARLA라는 3D 자율주행 차량 시뮬레이터를 사용하여 10개 마을에서 맑은 날씨 조건 아래 29개 클래스로 구성된 1200장의 이미지를 수집하고, 추가로 다양한 날씨 효과가 적용된 1200장의 이미지를 모았습니다. 이 데이터를 통해 세그멘테이션(semantic segmentation)을 수행하기 위해 encoder-decoder UNet 모델을 훈련시켰습니다.

- **Performance Highlights**: 훈련에 이미지 증강(image augmentation)을 적용함으로써 악천후 야간 조건에서의 세분화 성능이 크게 향상되었으며(p < 0.001), 그러나 날씨 데이터로 훈련된 모델은 맑은 날을 제외한 모든 조건에서 증강 데이터로 훈련된 모델보다 손실(loss)이 현저히 낮았습니다. 이는 도메인 적응(domain adaptation) 방식에 개선의 여지가 있음을 보여줍니다.



### Longitudinal Evaluation of Child Face Recognition and the Impact of Underlying Ag (https://arxiv.org/abs/2408.07225)
- **What's New**: 이번 연구는 신뢰할 수 있는 아동 얼굴 인식 기술의 필요성을 다루며, 8년 간의 데이터 수집을 통해 아동 얼굴 인식 시스템에 대한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 Clarkson University의 CITeR 연구 그룹에 의해 수집된 YFA 데이터베이스를 활용하여 아동의 등록(enrollment) 및 검증(verification) 정확도를 장기간에 걸쳐 분석합니다. 데이터는 6개월 간격으로 수집되었습니다.

- **Performance Highlights**: 아동 얼굴 인식 기술은 다양한 신흥 응용 프로그램에서의 신뢰성을 높이고, 장기간에 걸친 신뢰할 수 있는 인식을 가능하게 합니다.



### A Review of Pseudo-Labeling for Computer Vision (https://arxiv.org/abs/2408.07221)
Comments:
          21 pages, 4 figures

- **What's New**: 본 논문에서는 pseudo-labeling (PL)을 semi-supervised learning (SSL)과 unsupervised learning (UL) 분야에서 탐구합니다. PL의 새로운 정의를 제시하고, 이 방법론이 서로 다른 영역에서 어떻게 연결될 수 있는지를 조사하여, 향후 연구 방향들을 제안합니다.

- **Technical Details**: 이 연구는 PL 기술이 self-sharpening(자기 샤프닝), multi-view learning(다중 뷰 학습), multi-model learning(다중 모델 학습)과 같은 특성을 공유한다는 점을 강조합니다. PL은 대개 SSL에 초점을 두고 있으나, UL과 self-supervised learning(자기 지도 학습) 등에서도 적용 가능성을 탐구합니다.

- **Performance Highlights**: PL은 labeled 데이터가 적어도 모델의 성능을 향상시키는 데 기여할 수 있으며, consistency regularization(일관성 정규화) 및 entropy minimization(엔트로피 최소화) 등과 함께 사용될 때 더욱 효과적입니다.



### Handwritten Code Recognition for Pen-and-Paper CS Education (https://arxiv.org/abs/2408.07220)
- **What's New**: 본 논문에서는 손글씨로 작성된 코드 recognition을 위한 새로운 방법론을 제안합니다. 특히, Optical Character Recognition (OCR)과 Indentation Recognition 모듈을 결합하여 이전의 인식률 30%에서 5%로 개선하였습니다. 또한, 다중 모달 언어 모델을 활용한 end-to-end 방식의 손글씨 코드 인식 방법도 포함되어 있습니다.

- **Technical Details**: 기존에 존재하지 않았던 두 개의 공개 벤치마크 데이터세트를 제공하며, 각 접근법의 인식 정확도를 측정할 수 있는 방법론을 제안합니다. 손글씨 Python 코드의 들여쓰기 인식을 위한 새로운 방법론도 포함되어 있습니다. 이 연구는 OCR과 대형 언어 모델(LLM)의 통합을 통해 학생의 손글씨 코드에 대한 새로운 접근법을 제시합니다.

- **Performance Highlights**: 이 연구에서 제안된 방법은 구문 오류를 최소화하면서도 학생 코드의 정확한 디지털화에 기여하며, 평가 결과, 평균적인 Levenshtein 거리 감소를 통해 인식 알고리즘의 성능을 입증하였습니다. 학생들이 작성한 코드의 정확한 처리가 가능해져, CS 교육의 접근성을 높이는 데 기여할 것으로 기대됩니다.



### SeLoRA: Self-Expanding Low-Rank Adaptation of Latent Diffusion Model for Medical Image Synthesis (https://arxiv.org/abs/2408.07196)
Comments:
          Project Page: this https URL

- **What's New**: SeLoRA (Self-Expanding Low-Rank Adaptation) 모듈은 훈련 중 동적으로 순위를 확장하여 중요한 층에 추가적인 순위를 배치함으로써 의료 이미지 합성의 질을 향상시킵니다.

- **Technical Details**: SeLoRA는 각 층의 고유한 요구에 맞춰 LoRA의 순위를 동적으로 확장할 수 있는 구조를 가지고 있으며, 훈련 동안 Fisher 정보에 의해 안내되어 자율적으로 조정됩니다. 초기 낮은 순위를 시작으로, 순위는 층의 필요에 맞게 적응적으로 성장합니다.

- **Performance Highlights**: 제안된 방법은 LDMs (Latent Diffusion Models)가 의료 데이터를 효율적으로 세부 조정할 수 있도록 하며, 최소한의 순위로 우수한 이미지 품질을 제공합니다.



### Flexible 3D Lane Detection by Hierarchical Shape MatchingFlexible 3D Lane Detection by Hierarchical Shape Matching (https://arxiv.org/abs/2408.07163)
- **What's New**: 이 논문에서는 포인트 클라우드(point cloud)에서 3D 차선(line) 검출을 위한 새로운 엔드 투 엔드(end-to-end) 계층적(lane detection) 접근 방식을 제안합니다. 이러한 방법은 다양한 시각적 조건과 복잡한 유형에 대한 높은 정밀성을 유지하도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 상향식(bottom-up) 및 하향식(top-down) 정보 수집을 결합하여 3D 차선의 유연하고 정밀한 예측을 가능하게 합니다. 글로벌 매개변수 곡선(global parametric curve) 및 로컬 형태 예측(local shape prediction)을 통해 세부적인 형태를 포착하며, 다이나믹 앵커 셀(anchor cell)의 생성을 통해 예측을 최적화합니다.

- **Performance Highlights**: RoadBEV 데이터 세트에서의 실험 결과에 따르면, 제안된 방법은 최근의 최첨단 기술을 초과하는 정확도를 보여주며, 각각의 방법 요소의 효과성도 검증되었습니다.



### Controlling the World by Sleight of Hand (https://arxiv.org/abs/2408.07147)
- **What's New**: 본 논문에서는 CosHand라는 새로운 action-conditional generative model을 제안합니다. 이 모델은 레이블이 없는 비디오 데이터를 활용하여 손과 객체의 상호작용을 예측하고 이미지를 합성하는 능력을 보여줍니다.

- **Technical Details**: CosHand는 자동화된 손 분할 기법을 사용하여 비디오 프레임에서 손 마스크를 추출하고, 이를 통해 손 상호작용의 이전과 이후 이미지를 기반으로 미래의 상태를 생성합니다. 이 과정은 대량의 데이터(예: SomethingSomethingv2 데이터셋)를 통해 모델을 훈련시키며, 액세서리와 환경의 불확실성을 모델링합니다.

- **Performance Highlights**: CosHand는 기존 모델과 비교하여 강한 일반화 능력을 보여주며, 특히 번역, 늘리기 및 압축과 같은 보지 못한 객체와 환경에서의 상호작용을 정확하게 예측합니다. 또한 비인간 손(로봇 손)으로의 일반화도 가능하여, 로보틱스 분야에서도 유용할 것으로 기대됩니다.



### Vision Language Model for Interpretable and Fine-grained Detection of Safety Compliance in Diverse Workplaces (https://arxiv.org/abs/2408.07146)
Comments:
          20 pages, 7 figures

- **What's New**: 이번 연구에서는 안전 준수를 위한 Clip2Safety라는 새로운 감지 프레임워크를 제안합니다. 이 프레임워크는 네 가지 주요 모듈, 즉 장면 인식, 비주얼 프롬프트, 안전 물품 감지 및 세부 속성 검증으로 구성되어 있어 다양한 작업 환경에서 PPE를 효과적으로 인식하고 평가할 수 있도록 합니다.

- **Technical Details**: Clip2Safety의 프레임워크는 다음의 핵심 모듈로 구성됩니다: 1. 장면 인식 모듈은 현재 상황에 맞는 안전 장비를 식별합니다. 2. 비주얼 프롬프트 모듈은 감지 프로세스를 위한 특정 비주얼 프롬프트를 작성합니다. 3. 안전 물품 감지 모듈은 지정된 시나리오에 따라 안전 장비가 착용되고 있는지를 확인합니다. 4. 세부 속성 검증 모듈은 착용된 안전 장비가 필요한 속성 기준을 충족하는지 평가합니다.

- **Performance Highlights**: Clip2Safety는 최첨단 질문-응답 기반 VLM 모델보다 정확도 향상이 도모되었으며, 인퍼런스 시간은 기존 모델보다 200배 빨라졌습니다. 이 연구는 다양한 작업장의 안전 준수 데이터를 활용하여 성능을 평가하였습니다.



### Generative Photomontag (https://arxiv.org/abs/2408.07116)
Comments:
          Project webpage: this https URL

- **What's New**: 새로운 프레임워크를 제안하여 사용자가 생성된 이미지 조각을 조합하여 원하는 이미지를 생성하는 'Generative Photomontage'을 구현하는 방법을 소개합니다.

- **Technical Details**: 사용자는 브러시 스트로크 인터페이스를 통해 여러 이미지에서 원하는 부분을 선택할 수 있습니다. 이때, 그래프 기반 최적화 방법을 사용하여 디퓨전 특성 공간에서 이미지를 분할하고, 이 분할된 영역을 새로운 특성 공간 혼합 방법으로 조합합니다.

- **Performance Highlights**: 제안한 방법은 기존의 이미지 혼합 방법과 다양한 기준선보다 우수한 성능을 보이며, 새로운 조합 생성, 잘못된 형태 수정, 아티팩트 감소 및 프롬프트 정렬 개선 등 다양한 애플리케이션에서 매력적인 결과를 보여줍니다.



### Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities (https://arxiv.org/abs/2408.07666)
- **What's New**: 이 논문은 모델 병합의 여러 기법들을 종합적으로 검토하고 이해를 돕기 위한 체계적인 조사 연구입니다. 기존의 연구에서는 모델 병합에 대한 문헌이 부족하여 이 문제를 해결하기 위해 새로운 분류 체계를 제안합니다.

- **Technical Details**: 모델 병합은 원시 학습 데이터에 접근할 필요 없이 여러 개별 모델의 매개변수를 결합하여 보편적인 모델을 만드는 효과적인 기술입니다. 본 논문에서는 모델 병합 방법을 사전 병합(pre-merging)과 병합 중(during-merging) 두 단계로 나누어 탐구하며, 이 과정에서 각 단계별 세부 기술과 이론적 분석을 제공합니다. 특히, 사전 병합 방법은 입력 공간 및 가중치 공간을 분리하기 위한 선형화된 미세 조정(linearized fine-tuning), 이종 모델을 동질 모델로 변환하는 아키텍처 변환, 가중치 정렬을 포함합니다.

- **Performance Highlights**: 모델 병합 기술은 대형 언어 모델(large language models) 및 다중 모달 모델(multimodal models)과 같은 여러 하위 분야에서 뛰어난 성능을 보이며, 연속 학습(continual learning), 다중 작업 학습(multi-task learning), 소량 학습(few-shot learning) 등 다양한 적용에서 그 가능성을 보여줍니다. 또한, 이미지 생성 모델에서 스타일 혼합(style mixing) 및 훈련 비용 절감(training cost reduction) 등의 응용을 통해 더욱 발전하고 있습니다.



### Transformers and Large Language Models for Efficient Intrusion Detection Systems: A Comprehensive Survey (https://arxiv.org/abs/2408.07583)
Comments:
          arXiv admin note: text overlap with arXiv:2405.04760 by other authors

- **What's New**: 이 논문은 Transformers와 LLMs를 활용한 사이버 위협 탐지 시스템의 종합적인 분석을 제공하며, IDS(침입 탐지 시스템)의 발전을 설명합니다.

- **Technical Details**: Transformers의 기본 원리, 사이버 공격의 배경, 및 다양한 데이터셋에 대해 설명합니다. Attention 기반 모델, BERT, GPT와 같은 LLM들, CNN/LSTM-Transformer 하이브리드 모델, ViTs(비전 트랜스포머) 등 여러 아키텍처의 탐색이 이루어집니다.

- **Performance Highlights**: Transfomers 및 LLMs가 IDS의 정확도를 높여주는 잠재적 응용 가능성을 강조하며, 네트워크 트래픽 흐름에서 패턴 변화를 탐지할 수 있는 새로운 방법론을 제시합니다.



### Sonic: Fast and Transferable Data Poisoning on Clustering Algorithms (https://arxiv.org/abs/2408.07558)
Comments:
          preprint paper

- **What's New**: 본 논문에서는 클러스터링 알고리즘에 대한 데이터 중독(data poisoning) 공격을 효과적이고 확대 가능한 방법으로 해결하고자 하는 새로운 접근법인 Sonic을 제안하고 있습니다.

- **Technical Details**: Sonic은 점진적(Incremental)이고 스케일러블한 클러스터링 알고리즘인 FISHDBC를 활용하여 그래프 기반 및 밀도 기반 클러스터링 메소드(HDBSCAN 등)에 대해 신속한 중독 공격을 수행합니다. 또한, Sonic은 공격 최적화를 가속화하고 강건성 검증을 용이하게 합니다.

- **Performance Highlights**: 실험을 통해 Sonic은 MNIST, FASHIONMNIST, CIFAR-10, 20 Newsgroups 데이터셋을 포함한 네 가지 벤치마크 데이터셋에서 기존의 최첨단 방법론과 비교할 때 데이터 샘플 수와 특징 수가 증가함에 따라 월등히 높은 성능을 보였습니다.



### Improved 3D Whole Heart Geometry from Sparse CMR Slices (https://arxiv.org/abs/2408.07532)
Comments:
          13 pages, STACOM2024

- **What's New**: 본 연구에서는 심장 자기 공명 (CMR) 영상에서의 호흡 운동 아티팩트를 수정하고, 불규칙한 조각(segmentation data)을 조밀한 조각(dense segmentation)으로 변환하기 위해 Slice Shifting Algorithm (SSA), Spatial Transformer Network (STN), 및 Label Transformer Network (LTN)의 조합을 탐구합니다.

- **Technical Details**: 연구에서는 1699개 케이스에서 CT에서 생성된 CMR 슬라이스 세분화의 합성 데이터를 활용하여 SSA-LTN, STN과의 결합을 검증하였으며, 최종적으로 SSA는 STN 및 LTN 기반 모델에서 성능을 향상시키는 유용한 도구임을 입증합니다. 각 알고리즘의 성능은 Dice score와 Huasdorff distance를 기준으로 평가되었습니다.

- **Performance Highlights**: SSA-LTN 조합은 Dice score 94.0% 및 Huasdorff distance 4.7 mm으로 최고의 결과를 제공했으나, 8개의 사례에서 위상적 오류가 발생했습니다. STN은 위상적 오류를 최소한의 성능 저하로 수정할 수 있는 효과적인 도구로 나타났습니다.



### Costal Cartilage Segmentation with Topology Guided Deformable Mamba: Method and Benchmark (https://arxiv.org/abs/2408.07444)
- **What's New**: 이 연구에서는 costal cartilage(갈비 연골) 세분화를 위해 topology-guided deformable Mamba (TGDM)라는 새로운 딥러닝 기반 접근 방식을 제안합니다. TGDM은 복잡한 구조적 관계를 캡처하도록 설계되어 있으며, 세분화 과정의 유연성과 정확성을 높이기 위해 토폴로지 선행지식을 통합한 변형 모델을 활용합니다.

- **Technical Details**: TGDM은 UNet 유사 구조를 근간으로 하여 2단계 세분화 과정을 실행합니다. 첫 번째 단계에서는 CC의 경계 상자를 세분화하여 위치를 조정하고, 두 번째 단계에서는 position shifting Mamba(PSM)와 grouped deformable Mamba(GDM)라는 두 가지 주요 모듈을 통합하여 CC 간의 장거리 관계를 모델링합니다.

- **Performance Highlights**: TGDM 방법은 in-domain 및 out-of-domain 데이터 세트에서 DSC(다이스 계수) 및 NSD(조정된 남용 학습 비율) 점수에서 우수한 성능을 보여줍니다. 이는 환자와 이미징 조건 간의 변동성을 고려할 때 특히 중요합니다.



### Achieving Data Efficient Neural Networks with Hybrid Concept-based Models (https://arxiv.org/abs/2408.07438)
Comments:
          11 pages, 8 figures, appendix

- **What's New**: 본 논문에서는 기존의 클래스 라벨과 추가 정보인 개념(concept)을 함께 사용하는 하이브리드 개념 기반 모델(Hybrid Concept-Based Models)을 소개합니다. 이 모델은 데이터셋에 있는 더 많은 정보를 활용하여 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 하이브리드 모델 아키텍처는 기본적으로 기존의 개념 병목 모델(CBM)을 기반으로 하며, 개념 병목 레이어를 거치지 않는 추가 스킵 연결(skip connection)을 통하여 예측을 수행합니다. 또한, Sequential Concept Model(SCM)이라는 아키텍처를 제안하여 모든 개념 예측을 신경망의 여러 층에서 순차적으로 수행합니다. 이를 통해 데이터가 부족한 상황에서도 성능을 유지할 수 있도록 하였습니다.

- **Performance Highlights**: 하이브리드 개념 기반 모델은 기존의 컴퓨터 비전 모델 및 이전에 제안된 개념 기반 모델들과 비교했을 때, 특히 데이터가 희소한 환경에서 정확도(accuracy) 측면에서 우수한 성능을 보였습니다. 또한, 적대적 개념 공격(adversarial concept attacks) 알고리즘을 통해 개념 기반 모델의 해석 가능성에 대한 새로운 질문을 제기하였습니다.



### Automated Retinal Image Analysis and Medical Report Generation through Deep Learning (https://arxiv.org/abs/2408.07349)
Comments:
          Ph.D. thesis, 124 pages

- **What's New**: 본 연구는 망막 질환의 진단 및 치료를 위한 의료 보고서 생성을 자동화할 수 있는 인공지능(AI)의 가능성을 탐구합니다. AI는 대량의 이미지 데이터를 신속히 분석하여 정확한 진단에 필요한 미세한 패턴을 식별할 수 있습니다.

- **Technical Details**: 제안된 AI 기반 방법은 의료 키워드 표현 개선, 다중 모달(deep learning) 접근 방식, 그리고 AI 기반 보고서 생성 시스템의 해석 가능성을 높이는 기법을 포함합니다. 이를 통해 보고서 품질과 신뢰성을 향상시키고, 의사들이 보다 복잡한 사례에 집중할 수 있도록 지원합니다.

- **Performance Highlights**: 제안된 방법은 BLEU, ROUGE, CIDEr 및 METEOR와 같은 다양한 메트릭을 이용하여 평가되었으며, 자동 생성된 의료 보고서의 품질을 기존 방법보다 향상시켰습니다. 이 연구는 망막 질환 진단의 효율성을 요구에 맞춰 혁신적으로 개선할 수 있음을 보여줍니다.



### Discriminating retinal microvascular and neuronal differences related to migraines: Deep Learning based Crossectional Study (https://arxiv.org/abs/2408.07293)
- **What's New**: 이번 연구는 편두통 환자와 비환자를 구별하기 위해 색안저사진(Color Fundus Photography, CFP)과 광간섭단층촬영(Optical Coherence Tomography, OCT) 데이터를 활용하였고, 이 과정에서 컨볼루션 신경망(Convolutional Neural Networks, CNNs)을 적용하였습니다.

- **Technical Details**: 연구진은 VGG-16, ResNet-50, Inceptionv3 세 가지 CNN 모델을 사용하여, 369명의 참가자에서 수집한 두 가지 유형의 CFP 및 OCT 데이터를 분석했습니다. 인간의 해석이 미치지 못하는 망막의 미세혈관 및 신경적 특성을 평가하는 데 중점을 두었습니다.

- **Performance Highlights**: CFP Type 1 데이터를 사용했을 때, 편두통과 비편두통 데이터 간의 구별 능력(AUC)은 0.84에서 0.87로 매우 높았고, 세 가지 모델 간의 성능 차이가 없었습니다. 반면 CFP Type 2 데이터에서는 구별 성능이 낮아졌고, OCT 데이터는 모델 성능에 유의미한 기여를 하지 않았습니다.



### Scene-wise Adaptive Network for Dynamic Cold-start Scenes Optimization in CTR Prediction (https://arxiv.org/abs/2408.07278)
Comments:
          10 pages, 6 figures, accepted by Recsys 2024

- **What's New**: 최신 모바일 E-commerce에서 사용자에게 근처 상업 서비스 추천을 제공하는 것이 점점 더 중요해지고 있습니다. 기존의 추천 시스템들은 새로운 장면에서의 cold-start 문제를 해결하는 데 어려움을 겪고 있습니다. 본 연구에서는 Scene-wise Adaptive Network (SwAN)을 제안하여 이러한 문제를 해결하였습니다.

- **Technical Details**: SwAN은 장면 유사성 학습, 사용자별 장면 전환 인지, 새로운 장면을 위한 장면별 정보 구축, 장면 간의 논리 정보 차이 강화 등 여러 가지 기능을 포함하고 있습니다. 추천 시스템은 Embedding&MLP(다층 퍼셉트론) 패러다임을 따르며, Scene Relation Graph(SRG) 및 Similarity Attention Network(SAN)를 통해 장면 간의 유사성을 파악하고, Adaptive Ensemble-experts Module(AEM)을 이용해 공유 및 특정 정보를 추출합니다.

- **Performance Highlights**: SwAN은 Meituan의 온라인 추천 서비스에 성공적으로 배포되어 기존 모델 대비 5.64%의 클릭률(CTR) 개선 효과를 보였습니다. 또한, 하루 주문량 비율이 5.19% 증가하는 성과를 달성하였습니다.



### Lesion-aware network for diabetic retinopathy diagnosis (https://arxiv.org/abs/2408.07264)
Comments:
          This is submitted version wihout improvements by reviewers. The final version is published on International Journal of Imaging Systems and Techonology (this https URL)

- **What's New**: 딥러닝을 활용한 당뇨병성 망막병증(DR) 진단의 혁신적인 접근 방식인 레지온 인식 네트워크(LANet)를 제안합니다. 이 네트워크는 불균형한 데이터에서 레지온 정보를 더 잘 포착할 수 있도록 설계되었습니다.

- **Technical Details**: CNN(Convolutional Neural Network)을 기반으로 하여, 레지온 인식 모듈(LAM)과 특징 보존 모듈(FPM)을 포함합니다. LAM은 깊은 층에서 소음과 같은 레지온을 포착하고, FPM은 얕은 층에서 깊은 층으로 특징 융합을 지원합니다.

- **Performance Highlights**: 세 가지 공개 안저 데이터셋에서 실험한 결과, 제안된 LANet은 DR 스크리닝에서 0.967의 AUC(Area Under Curve)를 기록하며, 레지온 세분화에서 각각 7.6%, 2.1%, 1.2%의 평균 정밀도를 향상시켰습니다.



### All-around Neural Collapse for Imbalanced Classification (https://arxiv.org/abs/2408.07253)
- **What's New**: 본 논문은 Neural Collapse (NC) 현상을 깊이 있게 분석하고, 불균형 데이터 세트에서 발생하는 'minority collapse' 문제를 해결하기 위한 새로운 접근 방법인 All-around Neural Collapse (AllNC) 프레임워크를 제안합니다. 기존 연구에서는 주로 분류기(클래시파이어) 최적화에 초점을 맞췄던 반면, 본 연구는 클래스 평균(class means) 또한 압축되는 현상을 발견하였습니다.

- **Technical Details**: AllNC 프레임워크는 개별 활성화(individual activations), 클래스 평균(class means), 분류기 벡터(classifier vectors)를 포함하여 NC 현상을 다양한 측면에서 복원하는 것을 목표로 합니다. HyCon(Hybrid Contrastive loss)을 통해 두 개의 뷰(anchor point와 클래스 평균)만 이용하여 활성화를 조정하며, P2P(Peer-to-Peer loss)를 사용하여 클래스 평균과 분류기 벡터를 간단하게 정렬하여 최적의 클래스 간 분리도를 달성합니다. 또한 GBBN(Generalized Bilateral-Branch Network)을 통해 모델 훈련을 점진적으로 분리하여 개선합니다.

- **Performance Highlights**: AllNC는 여러 불균형 분류 벤치마크 데이터 세트에서 기존 방법을 크게 초월하는 성능을 보여주며, 균형 잡힌 설정에서도 효과를 검증하였습니다. 세 가지 보완 구성 요소인 HyCon, P2P, GBBN을 조합하여 불균형 훈련에서 원하는 NC 현상을 성공적으로 복원하였습니다.



### Seeing and Understanding: Bridging Vision with Chemical Knowledge Via ChemVLM (https://arxiv.org/abs/2408.07246)
Comments:
          Techical report

- **What's New**: 이 논문에서는 ChemVLM을 제안합니다. ChemVLM은 화학 분야에 특화된 최초의 오픈 소스 다중 모달 대형 언어 모델로, 화학 이미지 이해와 텍스트 분석 사이의 비호환성 문제를 해결합니다.

- **Technical Details**: ChemVLM은 VIT-MLP-LLM 아키텍처를 기반으로 하며, ChemLLM-20B를 바탕 모델로 사용하여 화학 텍스트 지식을 이해하고 활용하는 강력한 기능을 제공합니다. InternVIT-6B를 강력한 이미지 인코더로 활용하며, 화학 도메인에서 고품질의 데이터를 수집하여 이중 언어 다중 모달 질문-응답 데이터셋을 구성했습니다.

- **Performance Highlights**: 모델은 여러 오픈 소스 벤치마크와 세 개의 맞춤 평가 세트에서 성능이 테스트되었으며, 논문에서는 여섯 개의 작업 중 다섯 개에서 최고의 성과를 달성하여 SOTA 성능을 보였습니다.



### BVI-UGC: A Video Quality Database for User-Generated Content Transcoding (https://arxiv.org/abs/2408.07171)
Comments:
          12 pages, 11 figures

- **What's New**: 최근 연구들은 사용자 생성 콘텐츠(UGC)의 영상 품질을 평가하기 위한 새로운 데이터베이스인 BVI-UGC를 제안합니다. 이 데이터베이스는 전송 과정에서 UGC의 품질 평가를 위한 60개의 비완전(reference) 영상과 1,080개의 테스트 시퀀스를 포함하고 있습니다.

- **Technical Details**: BVI-UGC는 15개 주요 UGC 카테고리를 커버하는 60개의 고품질 소스 시퀀스를 수집하였으며, 이 비디오들은 x264 코덱을 사용하여 시뮬레이션된 압축 수준에서 압축되었습니다. 실험은 3,500명 이상의 참가자가 참여하는 대규모 크라우드소싱 기반의 주관적인 품질 평가를 포함했습니다.

- **Performance Highlights**: 비교 실험 결과, 21개의 기존 비디오 품질 메트릭(frontal-reference 및 no-reference)에서 0.6 이하의 SROCC 값을 기록하여 UGC 기반 비디오 품질 평가의 어렵고 일관성 없는 문제를 드러냈습니다.



### DisCoM-KD: Cross-Modal Knowledge Distillation via Disentanglement Representation and Adversarial Learning (https://arxiv.org/abs/2408.07080)
- **What's New**: 이번 연구에서는 전통적인 teacher/student 패러다임을 넘어 Cross-Modal Knowledge Distillation (CMKD)을 위한 새로운 프레임워크인 DisCoM-KD(Disentanglement-learning based Cross-Modal Knowledge Distillation)를 소개합니다. 이 프레임워크는 멀티모달 데이터에서 단일 모달 분류기로의 지식 전이를 위해 각 모달리티마다 다른 정보를 모델링합니다.

- **Technical Details**: DisCoM-KD는 분리(disentanglement) 표현 학습과 적대적 도메인 적응(adversarial domain adaptation)을 결합하여 각 모달리티에 대해 도메인 불변(domain-invariant), 도메인 정보(domain-informative), 도메인 무관(domain-irrelevant) 특성을 동시에 추출합니다. 이 방식을 통해 기존 teacher/student 모델을 개별적으로 학습시킬 필요가 없습니다.

- **Performance Highlights**: DisCoM-KD는 세 가지 표준 멀티모달 벤치마크에서 평가되었으며, 기존의 최첨단(Knowledge Distillation) 방법들과 비교했을 때, 서로 다른 모달리티가 겹치거나 겹치지 않는 상황에서 강력한 성능을 보여주었습니다. 이 결과는 멀티모달 데이터에서 단일 모달 신경망으로 정보를 추출하는 전통적인 패러다임을 재고할 수 있는 통찰력을 제공합니다.



### Anatomical Foundation Models for Brain MRIs (https://arxiv.org/abs/2408.07079)
Comments:
          12 pages

- **What's New**: 이 논문에서는 신경영상(neuroimaging)에서 신경계 질환과 신경퇴행성 질환을 탐지하기 위해 뇌 나이(brain age)를 주요 바이오마커로 사용하는 새로운 방법론인 AnatCL을 제안하고 있습니다. AnatCL은 해부학적 정보(anatomical information)와 약한 대조 학습(weakly contrastive learning) 접근 방식을 활용하여 다양한 임상 과제를 수행할 수 있는 강력한 기반 모델을 구축합니다.

- **Technical Details**: AnatCL 프레임워크는 환자의 나이와 동시에 세 가지 해부학적 특징(Mean Cortical Thickness, Gray Matter Volume, Surface Area)을 통합하여 대조 학습 손실 함수(loss function)를 수정했습니다. 이는 특정 지역(ROIs)에서 해부학적 정보를 계산하여 모델이 보다 유용한 표현 공간을 학습할 수 있도록 지원합니다. 또한, 로컬(local) 및 글로벌(global) 두 가지 버전의 프레임워크를 제안하여 특정 지역의 해부학적 변이가 모델링되도록 하였습니다.

- **Performance Highlights**: 자체 실험 결과, AnatCL은 여러 가지 다운스트림 임상 과제에서 기존의 기준 방법론보다 유사하거나 더 나은 성능을 기록했습니다. 특히, 신경영상 모델의 성능을 향상시키기 위해 타겟 해부학적 특징을 통합하는 것이 효과적이라는 것을 보여주었으며, 12개의 다양한 임상 과제와 10개의 임상 평가 점수 예측 작업에 걸쳐 광범위한 검증을 수행하였습니다.



### UniFed: A Universal Federation of a Mixture of Highly Heterogeneous Medical Image Classification Tasks (https://arxiv.org/abs/2408.07075)
Comments:
          MLMI@MICCAI 2024

- **What's New**: 이번 연구에서는 UniFed라는 새로운 연합 학습 프레임워크를 소개합니다. UniFed는 다양한 유형의 의료 이미지를 사용하여 질병을 분류할 수 있도록 설계되어 있으며, 각 클라이언트의 학습 작업의 복잡성에 따라 적절히 조정합니다.

- **Technical Details**: UniFed는 클라이언트와 서버 간의 동적 모델 업데이트를 통해 학습을 진행하며, 비정형화된 다중 작업을 효율적으로 처리하기 위한 동적 및 순차적 모델 전환 메커니즘을 통합합니다. 클라이언트의 작업 의존성과 복잡성을 반영하여 전이 모델 교환을 관리하여 통신 비용을 줄입니다.

- **Performance Highlights**: UniFed 프레임워크는 레티나, 조직 병리학, 간종양과 같은 다양한 질병 진단에 대해 기존 벤치마크보다 높은 정확도, 더 낮은 통신 비용 및 짧은 수렴 시간을 보여주었습니다.



New uploads on arXiv(cs.AI)

### Quantifying over Optimum Answer Sets (https://arxiv.org/abs/2408.07697)
- **What's New**: 본 논문에서는 약한 제약 조건(weak constraints)을 포함한 Answer Set Programming with Quantifiers (ASP(Q))의 확장을 제안합니다. 이를 통해 다항 계층(Polynomial Hierarchy)에서의 문제들을 우아하고 간결하게 인코딩할 수 있는 방법을 제공합니다.

- **Technical Details**: ASPω(Q)는 약한 제약 조건을 통해 정량화된 구성 프로그램 내에서 부분 최적화(local optimization) 및 전역 최적화(global optimization) 기준을 표현할 수 있습니다. 이 새로운 형식은 다양한 응용 시나리오를 통해 모델링 기능을 시연합니다.

- **Performance Highlights**: ASPω(Q)는 n개의 교차 정량자(alternating quantifiers)를 포함하는 프로그램이 Δ(n+1)P 문제를 모델링할 수 있다는 긍정적인 결과를 보여줍니다. 이는 ASP(Q) 프로그램에 약한 제약 조건을 추가함으로써 새로운 언어의 비명백한 특성을 드러냅니다.



### A General Framework for Constraint-based Causal Learning (https://arxiv.org/abs/2408.07575)
- **What's New**: 이 논문은 제약 기반(Constraint-based) 인과 학습 알고리즘의 정확성 조건을 분해하고, 이를 통해 일반적인 프레임워크를 제공하여 다양한 인과 발견 알고리즘의 정확성 조건을 구체화합니다.

- **Technical Details**: 저자들은 PC 알고리즘과 다른 기존 인과 발견 알고리즘의 정확성 조건을 구체적으로 제시하고, 스패리스트 마르코프 표현 조건(Sparsest Markov representation condition)이 기존의 최대 조상 그래프(Maximal ancestral graphs) 및 방향 비순환 그래프(Directed acyclic graphs)를 위한 최소성 개념에서 가장 약한 조건임을 보여줍니다. 또한 페얼 최소성(Pearl-minimality) 이상의 지식이 필요하다고 주장합니다.

- **Performance Highlights**: 논문은 제약 기반 인과 학습 알고리즘이 작동하는 조건을 제공하며, PC 알고리즘과 기존 알고리즘 간의 관계를 통해 정확성 조건의 일반화를 이룹니다.



### Planning with OWL-DL Ontologies (Extended Version) (https://arxiv.org/abs/2408.07544)
Comments:
          Extended version of a paper accepted at ECAI 2024

- **What's New**: 본 연구에서는 본체계(ontology)를 결합한 계획 문제를 다루는 새로운 접근법인 ontology-mediated planning을 소개합니다. 이 방식은 계획 문제와 본체계에 대한 설명을 강하게 분리하는 방식으로, 유연한 인터페이스를 통해 결합됩니다.

- **Technical Details**:  새로운 black-box 알고리즘을 제공하며, 이는 OWL DL의 전체 표현력을 지원합니다. 이를 통해 기존의 방식이 지원하지 못하는 병합 자동 계획과 본체계의 결합 beyond를 가능하게 합니다. 알고리즘은 ontology-mediated planning의 사양을 PDDL로 재작성하는 데 의존하며, 기존의 계획 시스템을 활용해 해결할 수 있습니다.

- **Performance Highlights**: 벤치마크 세트에 대한 평가 결과, 절차가 실제로 작동하며 추론 절차를 조정하는 것이 성능에 중요한 영향을 미친다는 것을 보여줍니다. Horn과 Schema 알고리즘 간의 비교에서는 Horn이 대부분의 인스턴스에서는 더 빠른 반면, 특정 경우 Schema가 유리함을 나타냅니다.



### Development of a Multi-Agent Clinical Decision Support System for Korean Triage and Acuity Scale (KTAS)-Based Triage and Treatment Planning in Emergency Departments (https://arxiv.org/abs/2408.07531)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 활용한 임상 의사 결정 지원 시스템(CDSS)이 응급실(ED) 의사와 간호사에게 환자 분류 및 치료 계획 수립을 지원하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 멀티 에이전트 CDSS는 Llama-3-70b를 기본 LLM으로 사용하며 CrewAI와 Langchain에 의해 조정됩니다. 시스템은 Triage Nurse, Emergency Physician, Pharmacist, ED Coordinator를 모사하는 네 개의 AI 에이전트로 구성되어 있으며, 비상 분류를 위해 한국 비상 분류 및 중증도 척도(KTAS)와 통합되어 있습니다. 또한, RxNorm API와 연동하여 약물 관리를 수행합니다.

- **Performance Highlights**: 이 CDSS는 Asclepius 데이터셋을 사용하여 평가되었으며, 단일 에이전트 시스템의 기준과 비교하여 높은 정확도를 보여주었습니다. 주요 진단, 중대한 발견 식별, 배치 결정, 치료 계획 및 자원 배분 등 주요 영역에서도 강력한 성능을 발휘했습니다.



### Fast Inference for Probabilistic Answer Set Programs via the Residual Program (https://arxiv.org/abs/2408.07524)
Comments:
          The paper has been accepted at the ICLP2024 conference and under consideration in Theory and Practice of Logic Programming (TPLP)

- **What's New**: 본 논문에서는 Probabilistic Answer Set Program에서 쿼리의 확률을 계산할 때, 프로그램의 일부가 쿼리에 영향을 미치지 않고 오히려 grounding의 크기에 영향을 미치는 경우에 대한 문제를 다룹니다. 이 불필요한 부분을 식별하고 제거하는 방법을 제안하여 계산 속도를 향상시키고자 합니다. 특히 SLG resolution을 활용하여 residual program을 얻는 방법을 소개합니다.

- **Technical Details**: 논문에서 제안하는 방법은 PASP(Probabilistic Answer Set Programming)를 Prolog 프로그램으로 변환하고, 해당 쿼리에 대한 residual program을 SLG resolution를 통해 계산하는 것입니다. 그런 다음, 이 residual program을 원래의 PASP보다 더 작은 크기로 변환하고, 그 결과를 사용하여 쿼리의 확률을 계산합니다. 이를 통해 grounding해야 하는 프로그램의 크기를 줄이게 됩니다.

- **Performance Highlights**: 그래프 데이터셋에 대한 실험 결과, 이 접근방식은Inference(추론) 속도를 현저하게 개선함을 보여주었습니다. 이로써 대규모 프로그램의 처리 시간 단축에 기여할 수 있는 가능성을 시사합니다.



### Optimising Dynamic Traffic Distribution for Urban Networks with Answer Set Programming (https://arxiv.org/abs/2408.07521)
- **What's New**: 본 논문에서는 Answer Set Programming (ASP)을 활용하여 도시 네트워크에서의 동적 교통 분배 문제를 해결한 사례를 소개합니다. 이 애플리케이션을 통해 ASP가 도시의 교통 흐름을 최적화하는 데 효과적으로 사용된다는 점이 강조됩니다.

- **Technical Details**: ASP는 비단조 추론(non-monotonic reasoning)과 논리 프로그래밍(logic programming) 분야에서 개발된 프로그래밍 패러다임입니다. 본 논문에서 제안하는 프레임워크는 네트워크 분석, 도메인 독립 탐색(domain-independent search), 경로 최적화(route optimization), 이동성 시뮬레이션(mobility simulation) 등 4개의 주요 단계로 구성됩니다. ASP는 교통 흐름을 최적화하기 위한 최적 경로의 조합을 찾는 데 사용됩니다.

- **Performance Highlights**: 영국과 이탈리아의 두 도시 지역을 대상으로 실제 교통 데이터를 분석한 결과, ASP가 포함된 프레임워크의 효율성과 실용성을 입증하였습니다. 최대 600대의 차량을 포함하는 모든 경우에 대해 최적의 해결책을 제공하며, 교통 혼잡도 및 연료 소비와 같은 여러 지표를 실제로 감소시킬 수 있음을 보여줍니다.



### Dominating Set Reconfiguration with Answer Set Programming (https://arxiv.org/abs/2408.07510)
- **What's New**: 이 논문에서는 Dominating Set Reconfiguration Problem (DSRP)을 해결하기 위한 새로운 접근 방식을 제시합니다. Answer Set Programming (ASP)을 기반으로 하여 문제를 모델링하고 해결하는 방법을 설명합니다.

- **Technical Details**: 연구에서는 DSRP 인스턴스를 ASP 사실로 변환하고, 최적화 변형인 최소 지배 집합 문제를 해결하는 두 가지 전통적인 ASP 인코딩을 비교합니다. 또한, 'token jumping'이라는 인접성 관계를 활용하여 DSRP 해결을 위한 ASP 인코딩을 제안합니다. 이 접근법은 clingo의 Python API를 사용하여 다단계 ASP 해결을 지원합니다.

- **Performance Highlights**: 제안된 ASP 인코딩은 442개의 테스트 인스턴스 중 363개에 대해 도달 가능성을 결정했습니다. 특히, 힌트 제약 조건이 잘 작동하여 도달 불가능성을 결정하는 데 효과적이었습니다. 이 논문의 결과는 지배 집합 재구성 분야의 최신 기술 발전에 기여할 것으로 기대됩니다.



### Problem Solving Through Human-AI Preference-Based Cooperation (https://arxiv.org/abs/2408.07461)
- **What's New**: 본 논문은 인공지능 일반 지능 (AGI) 및 수퍼휴먼 AI의 개발에 대한 기존의 믿음이 있다는 점을 논의하며, 전문가 도메인에서 해결되지 않은 복잡한 문제들을 소개합니다. 이를 해결하기 위해 HAI-Co2라는 새로운 인간-AI 공동 구축 프레임워크를 제안합니다.

- **Technical Details**: HAI-Co2 프레임워크는 인간 전문가와 AI 에이전트 간의 협력적 문제 해결을 지원하기 위해, 다층 추상의 후보 솔루션을 제시하고, 인간 전문가의 다중 선호 입력을 허용하며, 탐색 기반 방법론을 통해 반복적으로 수정된 솔루션을 제공합니다.

- **Performance Highlights**: HAI-Co2는 기존의 단일 고객 모형의 AI 모델보다 더 높은 효율성을 보여주는 사례 연구를 통해 사람과 AI 간의 효과적인 협력이 가능함을 입증합니다.



### The Restaurant Meal Delivery Problem with Ghost Kitchens (https://arxiv.org/abs/2408.07417)
- **What's New**: 이 논문은 최근 빠르게 성장하고 있는 레스토랑 음식 배달 서비스의 운영에 있어 새로운 비즈니스 개념인 '고스트 키친(Ghost Kitchens)'을 연구합니다. 고스트 키친은 여러 레스토랑의 음식을 중앙에서 조리하고 배달하여 동시성을 활용하는 모델입니다.

- **Technical Details**: 논문에서는 고스트 키친의 효과적인 운영을 위해 순차적 의사결정 과정으로 문제를 모델링합니다. 주문 준비 및 배달 차량 파견의 통합 최적화를 위한 큰 이웃 탐색(Large Neighborhood Search, LNS) 절차를 제안하며, 가치 함수 근사(Value Function Approximation, VFA)를 이용해 의사결정을 평가합니다. 최적화를 위해 분산된 결정 영역을 줄이는 방법론을 개발하였습니다.

- **Performance Highlights**: 고스트 키친은 기존의 배달 시스템에 비해 서비스 품질 및 인력 활용에서 현저한 장점을 제공합니다. 통합 최적화를 통해 평균 배송 지연을 줄이고, 꾸준한 예측과 결정이 중요함을 강조합니다. 신속한 배달과 신선한 음식 간의 트레이드오프를 위한 관리적 통찰도 제공합니다.



### On-the-fly Synthesis for LTL over Finite Traces: An Efficient Approach that Counts (https://arxiv.org/abs/2408.07324)
Comments:
          32 pages, 3 figures, 3 tables

- **What's New**: 본 연구에서는 LTL (Linear Temporal Logic) 과제의 비효율성을 해결하는 새로운 방법을 제시합니다. 이는 전체적으로 DFA를 생성하기보다, LTLf에서 TDFA (Transition-based DFA)로 직접 전환하여 병렬화된 방식으로 합성을 수행할 수 있도록 합니다.

- **Technical Details**: LTLf의 의미론을 직접 활용하여 TDFA로 변환하는 방법을 소개합니다. 이 방법은 최종 오토마타의 직접 구성 요소로 중간 결과를 통합하여 자동 생성 과정에서 병렬적으로 작업을 수행할 수 있습니다. 알고리즘은 글로벌 전진 방법과 로컬 후진 방법을 결합하여 상태 공간을 탐색합니다. 또한 강하게 연결된 구성 요소를 감지하고 두 가지 최적화 기술인 모델 기반 합성과 상태 함축(state entailment)을 도입합니다.

- **Performance Highlights**: 실험 결과, 본 연구의 'on-the-fly' 접근 방식이 테스트된 벤치마크에서 최고의 성능을 달성하며, 기존 도구 및 접근 방식과 효과적으로 보완하는 것으로 나타났습니다.



### NL2OR: Solve Complex Operations Research Problems Using Natural Language Inputs (https://arxiv.org/abs/2408.07272)
- **What's New**: 이 논문에서 제안하는 NL2OR 방법론은 비전문가의 자연어 질의를 바탕으로 OR(Operations Research) 문제의 수학 모델을 생성하고 수정할 수 있게 해줍니다. 이는 전문가의 도메인 지식 없이도 문제를 정의하고 해결하는 시간을 단축시키고, 소규모 기업에게 유용한 솔루션이 될 수 있습니다.

- **Technical Details**: NL2OR 파이프라인은 사용자의 자연어 입력을 받아 OR 문제의 추상 모델을 생성하고, 해당 모델에 대한 데이터 매핑 및 solver 선택을 통해 솔루션을 제공합니다. 주요 구성 요소로는 Domain Specific Language (DSL) Generator, Framework for OR Analytics (FORA) Builder, FORA Executor, Report Generator가 있습니다. 이 시스템은 LLM(고급 언어 모델)을 활용하여 사용자 입력을 처리하고, 최적화 문제에 대한 경량 접속 경로를 제공합니다.

- **Performance Highlights**: 실험 결과, NL2OR는 여러 중요한 OR 문제에서 기존 방법론에 비해 시간과 효율성을 크게 향상시킴을 입증하였습니다. 이 시스템은 사용자가 복잡한 OR 문제를 보다 간편하게 해결할 수 있도록 도와줍니다.



### Can Large Language Models Reason? A Characterization via 3-SA (https://arxiv.org/abs/2408.07215)
- **What's New**: 본 논문은 LLMs(Large Language Models)의 실제 추론 능력에 대한 새로운 관점을 제시하며, 3-SAT 문제를 통해 LLMs의 추론 능력을 평가합니다. 기존의 벤치마크에 의존하기 보다는, 3-SAT 문제에서의 성과를 통해 LLM의 추론 능력을 경험적으로 특성화합니다.

- **Technical Details**: 3-SAT는 NP-complete 문제로서, 이 문제를 해결하기 위해 결정형(Decision Problem) 및 탐색형(Search Problem) 두 가지 변형을 사용해 LLMs의 성능을 평가합니다. GPT-4를 기준으로 설정하고, 입력 형식에 따라 두 가지 방식을 비교합니다: (i) SAT-Menu, (ii) SAT-CNF.

- **Performance Highlights**: GPT-4는 Easy 문제에서는 높은 정확도를 보이나 Hard 문제에서는 약 10%로 낮은 성능을 기록했습니다. 이는 LLMs가 고유한 문제에 따라 추론 능력이 크게 변한다는 것을 보여줍니다.



### Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents (https://arxiv.org/abs/2408.07199)
- **What's New**: 본 연구에서는 웹 환경에서의 다단계 추론 능력을 향상시키기 위한 새로운 프레임워크인 Agent Q를 소개합니다. 이 프레임워크는 Guided Monte Carlo Tree Search (MCTS)와 Self-Critique 메커니즘을 결합하여 집단지성 최적화 알고리즘인 Direct Preference Optimization (DPO)을 활용한 반복적인 미세 조정을 통해 효율적인 학습을 가능하게 합니다.

- **Technical Details**: Agent Q는 LLM(대형 언어 모델)을 활용하여 웹 페이지에서의 탐색을 안내 받고, 각 단계에서 AI 피드백과 Self-Critique를 통해 중간 보상을 제공합니다. 이 과정을 통해 성공적인 경로와 실패한 경로 모두에서 학습하며, 최적의 정책을 도출하게 됩니다.

- **Performance Highlights**: WebShop 환경에서 Agent Q는 행동 클로닝 및 강화 학습의 기준선보다 뛰어난 성능을 보였고, 평균 인간 성능을 초과했습니다. 또한 실제 예약 시나리오에서 Llama-3 70B 모델의 제로샷 성공률을 18.6%에서 81.7%로 향상시켰으며(340% 상대 증가), 온라인 검색 기능을 추가하였을 때 95.4%로 증가했습니다.



### End-to-end Semantic-centric Video-based Multimodal Affective Computing (https://arxiv.org/abs/2408.07694)
Comments:
          Under Review

- **What's New**: 이 논문에서는 인공지능의 일반화 (AGI)에서 인간의 정서를 이해하는 것이 기계의 인지 능력을 향상시키기 위해 필수적이라는 점을 강조합니다. 특히, 인간이 말하는 비디오를 활용한 다중 모달 정서 컴퓨팅(Multimodal Affective Computing, MAC)에 중점을 두고 있습니다.

- **Technical Details**: 본 연구는 'SemanticMAC'라는 새로운 엔드 투 엔드 프레임워크를 제안하며, 사전 학습된 Transformer 모델을 사용하여 다중 모달 출처에서의 특징들을 처리합니다. 주요 구성요소로는 Affective Perceiver 모듈과 Semantic-centric Gated Feature Interaction (SGFI), Semantic-centric Label Generation (SCLG) 및 Semantic-centric Contrastive Learning (SCCL) 기법을 포함합니다.

- **Performance Highlights**: SemanticMAC는 7개의 공공 데이터 세트에서 4개의 MAC 하위 작업에서 기존 최첨단 방법을 능가하는 성능을 보여줍니다. 이 접근 방식은 변별력이 있는 특정 및 공유 의미 표현을 효과적으로 학습할 수 있습니다.



### A Spitting Image: Modular Superpixel Tokenization in Vision Transformers (https://arxiv.org/abs/2408.07680)
Comments:
          To appear in ECCV (MELEX) 2024 Workshop Proceedings

- **What's New**: 본 논문에서는 기존의 ViT (Vision Transformer) 아키텍처에서 이미지의 의미(content)와 무관하게 그리드 기반의 접근 방식을 사용하는 대신, 모듈화된 슈퍼픽셀 토큰화(superpixel tokenization) 전략을 제안합니다.

- **Technical Details**: 제안된 방법은 온라인 콘텐츠 인식(tokenization-aware) 토큰화와 스케일(scale) 및 형태(shape) 불변의 위치 임베딩(positional embeddings)을 이용합니다. 실험에서는 패치 기반 토큰화(patch-based tokenization)와 무작위 파트션(randomized partitions)을 기준선으로 삼아 비교했습니다.

- **Performance Highlights**: 우리의 방법은 어트리뷰션(attributions)의 정확도를 크게 향상시키고, 제로샷(zero-shot) 비지도 학습 밀집 예측(dense prediction) 작업에서 픽셀 수준의 세분성을 제공하면서도 분류(classification) 작업에서 예측 성능을 유지합니다.



### Deep Learning: a Heuristic Three-stage Mechanism for Grid Searches to Optimize the Future Risk Prediction of Breast Cancer Metastasis Using EHR-based Clinical Data (https://arxiv.org/abs/2408.07673)
- **What's New**: 이번 연구에서는 낮은 예산으로 진행하는 grid search의 실행 시간을 효율적으로 관리하기 위한 휴리스틱(heuristic) 3단계 메커니즘과 모델 예측 성능을 개선하기 위한 sweet-spot grid search (SSGS) 및 randomized grid search (RGS) 전략을 소개합니다.

- **Technical Details**: 연구에서 deep feedforward neural network (DFNN) 모델을 개발하고, 이 모델을 grid search를 통해 최적화하였습니다. 3단계 메커니즘과 SSGS, RGS 전략을 적용하여 총 8회의 grid search를 수행하였습니다. 각 DFNN-model의 하이퍼파라미터의 중요성을 해석하는 SHAP 분석을 포함하여 다양한 SHAP 분석을 실시하였습니다.

- **Performance Highlights**: 연구 결과, grid search를 통해 유방암 전이 위험 예측 성능이 각각 5년, 10년, 15년에서 평균 18.6%, 16.3%, 17.3% 개선되었습니다. 그리드 검색은 최적 모델 성능을 보여줄 뿐만 아니라 합리적인 모델 발견 및 단위 그리드 검색 시간을 포함한 다양한 측면에서 grid search의 특성을 분석하였습니다.



### Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities (https://arxiv.org/abs/2408.07666)
- **What's New**: 이 논문은 모델 병합의 여러 기법들을 종합적으로 검토하고 이해를 돕기 위한 체계적인 조사 연구입니다. 기존의 연구에서는 모델 병합에 대한 문헌이 부족하여 이 문제를 해결하기 위해 새로운 분류 체계를 제안합니다.

- **Technical Details**: 모델 병합은 원시 학습 데이터에 접근할 필요 없이 여러 개별 모델의 매개변수를 결합하여 보편적인 모델을 만드는 효과적인 기술입니다. 본 논문에서는 모델 병합 방법을 사전 병합(pre-merging)과 병합 중(during-merging) 두 단계로 나누어 탐구하며, 이 과정에서 각 단계별 세부 기술과 이론적 분석을 제공합니다. 특히, 사전 병합 방법은 입력 공간 및 가중치 공간을 분리하기 위한 선형화된 미세 조정(linearized fine-tuning), 이종 모델을 동질 모델로 변환하는 아키텍처 변환, 가중치 정렬을 포함합니다.

- **Performance Highlights**: 모델 병합 기술은 대형 언어 모델(large language models) 및 다중 모달 모델(multimodal models)과 같은 여러 하위 분야에서 뛰어난 성능을 보이며, 연속 학습(continual learning), 다중 작업 학습(multi-task learning), 소량 학습(few-shot learning) 등 다양한 적용에서 그 가능성을 보여줍니다. 또한, 이미지 생성 모델에서 스타일 혼합(style mixing) 및 훈련 비용 절감(training cost reduction) 등의 응용을 통해 더욱 발전하고 있습니다.



### Alignment-Enhanced Decoding:Defending via Token-Level Adaptive Refining of Probability Distributions (https://arxiv.org/abs/2408.07663)
Comments:
          15 pages, 5 figures

- **What's New**: 본 논문에서는 Jailbreak 공격을 방지하기 위해 Alignment-Enhanced Decoding (AED)라는 새롭고 혁신적인 방어 메커니즘을 제안합니다. 기존의 방어 방법들이 입력을 수정하거나 검사를 통해 공격을 방어하는 것과는 달리, AED는 경쟁하는 목표를 고려하여 모델의 정렬 실패의 근본 원인을 해결합니다.

- **Technical Details**: AED는 Competitive Index를 정의하여 모델의 정렬 실패를 정량화하고, 자기 평가 피드백을 통해 post-alignment logits를 계산합니다. AED는 원래의 logits와 post-alignment logits을 결합하여 무해하고 유용한 분포를 생성하는 방식으로 동작합니다. 이는 추가 학습 없이도 각 단계의 디코딩 과정이 무해한 목표를 준수하도록 합니다.

- **Performance Highlights**: AED는 총 다섯 개의 인기 있는 대형 언어 모델에서 실험을 수행하였으며, GCG, AutoDan, ICD, Refusal_Suppression과 같은 다양한 고급 jailbreak 공격을 효과적으로 방어하였습니다. 또한 무해한 데이터셋에서 일반적인 문의에 대해서도 유용함을 유지하는 것으로 나타났습니다.



### Adaptive Behavioral AI: Reinforcement Learning to Enhance Pharmacy Services (https://arxiv.org/abs/2408.07647)
Comments:
          Presented at The First Workshop on AI Behavioral Science (AIBS'24) at KDD 2024, August 25, Barcelona, Spain

- **What's New**: 이번 연구에서는 저소득 및 중간 소득 국가의 약사들이 필요로 하는 행동 개입(behavioral interventions)을 제공하기 위한 인공지능(AI) 기반의 플랫폼과 모바일 건강 애플리케이션을 소개합니다. 이 시스템은 개인 맞춤형 행동 개입을 통해 약사들에게 필수적인 기술과 공중 보건 인식을 높이며, 약국 재고 관리 개선을 목표로 합니다.

- **Technical Details**: 제안된 플랫폼은 모바일 애플리케이션 및 디지털 도구에 통합되는 소프트웨어 개발 키트(SDK)로 구성됩니다. 이 플랫폼은 행동 유도(nudges) 기능을 향상시키기 위해 강화 학습(Reinforcement Learning) 기반 권장 사항을 제공하며, 이를 통해 약사들은 지속적인 전문성 개발 프로그램과 재고 예측에 기반한 리마인더 등 다양한 지원을 받을 수 있습니다.

- **Performance Highlights**: SwipeRx 애플리케이션을 통한 초기 실험에서, 맞춤형 추천 시스템이 약국의 재고 가용성을 향상시키고, 사용자와의 상호작용을 강화하는 데 긍정적인 영향을 미쳤습니다. 향후 연구는 다양한 추천 전략을 시험하여 공중 보건 문제를 해결하려 하고 있습니다.



### Boosting Unconstrained Face Recognition with Targeted Style Adversary (https://arxiv.org/abs/2408.07642)
- **What's New**: 본 논문에서는 Targeted Style Adversary (TSA)라는 새로운 방법론을 제안합니다. TSA는 레이블이 있는 데이터와 없는 데이터 간의 인스턴스 수준의 피쳐 통계를 보간(interpolate)하여 훈련 데이터를 확장함으로써 얼굴 인식을 위한 훈련 데이터의 다양성을 증가시키는 단순하지만 효과적인 접근 방식을 제공합니다.

- **Technical Details**: TSA 방법은 두 가지 주요 관찰에 의해 동기 부여됩니다: (i) 입력 도메인은 피쳐 통계에 반영되며, (ii) 얼굴 인식 모델의 성능은 스타일 정보에 의해 영향을 받습니다. TSA는 모델의 숨겨진 공간(hidden space)에서 작동하여 레이블이 있는 샘플의 스타일 정보를 조작하고, 생성된 스타일의 타당성을 보장하면서 훈련 중 유사하지 않은 인스턴스를 구별하기 위해 엔트로피 기반 접근 방식을 제안합니다.

- **Performance Highlights**: TSA 방법은 unconstrained 벤치마크에서 평가된 결과, 경쟁 모델에 비해 우수하거나 동등한 성능을 보이며, 훈련 속도는 거의 70% 향상되었고 메모리 소비는 40% 감소하였습니다.



### Drug Discovery SMILES-to-Pharmacokinetics Diffusion Models with Deep Molecular Understanding (https://arxiv.org/abs/2408.07636)
Comments:
          13 pages, 5 figures, 4 tables

- **What's New**: Imagand는 SMILES 입력에 조건화된 약리학적 변환 (Pharmacokinetic) 특성을 생성할 수 있는 새로운 S2PK diffusion 모델입니다. 이 모델은 기존의 약리학적 데이터 세트의 희소성을 극복하는 데 도움을 줄 수 있습니다.

- **Technical Details**: Imagand는 전통적인 Diffusion 모델을 기반으로 하며, 기계 학습 (machine learning) 과정에서 SMILES embedding을 사용하여 12개의 약리학적 타겟 속성을 생성합니다. 이 모델은 SMILES 인코더 모델을 통해 학습된 복잡한 화학 구조를 캡처하는 강력한 의미론적 SMILE 인코더를 필요로 합니다.

- **Performance Highlights**: Imagand에 의해 생성된 합성 PK 데이터는 실제 데이터의 분포와 매우 유사하며, 다운스트림 작업에서 성능을 향상시킵니다. 연구자들은 Imagand를 사용하여 수천 개의 리간드에 대한 대규모 합성 PK 데이터셋을 효율적으로 생성할 수 있습니다.



### Optimizing HIV Patient Engagement with Reinforcement Learning in Resource-Limited Settings (https://arxiv.org/abs/2408.07629)
Comments:
          Presented at the 7th epiDAMIK ACM SIGKDD International Workshop on Epidemiology meets Data Mining and Knowledge Discovery, August 26, 2024, Barcelona, Spain

- **What's New**: CHARM(Community Health Access & Resource Management) 앱은 AI 기반 모바일 애플리케이션으로, 지역 보건 근로자(Community Health Workers, CHWs)를 위해 설계되었습니다. 이 앱은 사례 관리, 학습 향상 및 커뮤니케이션 개선을 통해 CHWs의 역량을 강화하는 데 주력합니다.

- **Technical Details**: CHARM 앱은 AI와 강화 학습(Reinforcement Learning, RL) 기반의 적응형 개입을 통합하여 CHWs가 실시간 환자 데이터를 바탕으로 맞춤형 HIV 치료 및 예방 계획을 조정할 수 있도록 지원합니다. 또한, 기계 학습(Machine Learning, ML) 알고리즘을 활용하여 의료 용품 수요 변동을 예측하고 관리함으로써 자원의 효율적 분배를 보장합니다.

- **Performance Highlights**: CHARM 앱은 CHWs의 참여와 효율성을 극대화하여 환자 관리의 질을 향상시키며, 특히 HIV 예방 및 치료의 접근성을 높이는 데 기여하고 있습니다. 2024년 8월에 파일럿론칭이 예정되어 있으며, 이후 7개 국가에서의 점진적인 배포가 계획되고 있습니다.



### Battery GraphNets : Relational Learning for Lithium-ion Batteries(LiBs) Life Estimation (https://arxiv.org/abs/2408.07624)
Comments:
          Accepted in Workshop on Graph Learning for Industrial Applications : Finance, Crime Detection, Medicine, and Social Media (NeurIPS 2022)

- **What's New**: 이 논문에서는 Battery GraphNets(배터리 그래프 네트워크) 프레임워크를 소개하여, 리튬 이온 배터리(LiB)의 잔여 유용 수명(Remaining Useful Life, RUL)을 예측하는 데 있어 전통적인 방법들이 간과하였던 배터리 파라미터 간의 관계적 의존성을 통합하는 방법을 제안합니다.

- **Technical Details**: 제안된 Battery GraphNets 프레임워크는 배터리 파라미터 간의 복잡한 상호작용을 포착하기 위해 이산 의존 그래프 구조를 jointly 학습합니다. 이 프레임워크는 배터리 데이터의 관찰된 데이터로부터 상호 작용하는 배터리 파라미터의 쌍 간 관계를 학습하고, Graph Neural Networks(GNNs)를 통해 내재적 배터리 열화 동력을 모델링합니다.

- **Performance Highlights**: 제안된 방법은 공개 배터리 데이터셋에서 여러 인기 있는 예측 방법과 비교하여 현저히 우수한 성능을 나타내며, state-of-the-art(SOTA)의 성능을 달성하였습니다. 또한, 방법의 효능을 지지하는 ablation study 결과를 보고합니다.



### Transformers and Large Language Models for Efficient Intrusion Detection Systems: A Comprehensive Survey (https://arxiv.org/abs/2408.07583)
Comments:
          arXiv admin note: text overlap with arXiv:2405.04760 by other authors

- **What's New**: 이 논문은 Transformers와 LLMs를 활용한 사이버 위협 탐지 시스템의 종합적인 분석을 제공하며, IDS(침입 탐지 시스템)의 발전을 설명합니다.

- **Technical Details**: Transformers의 기본 원리, 사이버 공격의 배경, 및 다양한 데이터셋에 대해 설명합니다. Attention 기반 모델, BERT, GPT와 같은 LLM들, CNN/LSTM-Transformer 하이브리드 모델, ViTs(비전 트랜스포머) 등 여러 아키텍처의 탐색이 이루어집니다.

- **Performance Highlights**: Transfomers 및 LLMs가 IDS의 정확도를 높여주는 잠재적 응용 가능성을 강조하며, 네트워크 트래픽 흐름에서 패턴 변화를 탐지할 수 있는 새로운 방법론을 제시합니다.



### MetaSeg: MetaFormer-based Global Contexts-aware Network for Efficient Semantic Segmentation (https://arxiv.org/abs/2408.07576)
Comments:
          Accepted by WACV 2024

- **What's New**: MetaSeg라는 강력한 시맨틱 세분화 네트워크를 제안합니다. 이 네트워크는 MetaFormer 아키텍처를 기반으로 하여, 백본(backbone)부터 디코더(decoder)까지의 전체 구조에서 활용됩니다.

- **Technical Details**: MetaFormer 블록을 사용하는 CNN 기반 백본과 글로벌 컨텍스트를 캡처하기 위해 새로운 셀프 어텐션 모듈인 Channel Reduction Attention (CRA)를 설계합니다. CRA는 쿼리(query)와 키(key)의 채널 차원을 1차원으로 줄여, 계산 효율성을 높입니다.

- **Performance Highlights**: 제안된 MetaSeg는 ADE20K, Cityscapes, COCO-Stuff 등 여러 세분화 벤치마크에서 이전의 최첨단 방법들을 능가하며, SegNeXt-T보다 각각 1.3%, 0.3%, 1.0% mIoU 개선을 보이고, 계산 비용은 각각 16.7%, 5.2%, 16.7% 감소합니다.



### Multi-task Heterogeneous Graph Learning on Electronic Health Records (https://arxiv.org/abs/2408.07569)
Comments:
          Accepted by Neural Networks

- **What's New**: 이 논문에서는 전자 건강 기록(EHR)을 위한 새로운 다중 작업 모델인 MulT-EHR (Multi-Task EHR)을 제안합니다. 이 모델은 이질적 그래프(heterogeneous graph)를 활용하여 EHR의 복잡한 관계를 학습하고, 노이즈를 줄이기 위한 인과적 노이즈 제거 모듈을 도입하여 모델의 성능을 향상시킵니다.

- **Technical Details**: MulT-EHR은 그래프 대조 학습(graph contrastive learning) 기반의 사전 학습 모듈을 통해 이질적 그래프 내의 관계적 특성을 향상시키고, 변환기 기반의 그래프 신경망(GNN) 아키텍처를 사용하여 노드 수준 표현을 학습합니다. 여기에 인과적 추론(causal inference) 프레임워크를 적용하여 노이즈를 줄이는 인과적 노이즈 제거 모듈을 설계하였습니다. 또한, 다중 작업 집계 메커니즘을 통해 작업 간 지식을 활용하여 예측 성능을 개선합니다.

- **Performance Highlights**: MIMIC-III 및 MIMIC-IV 데이터셋에 대한 철저한 실험을 통해 제안된 방법이 사망 예측, 재입원 예측, 병원 내 체류 기간 예측 및 약물 추천과 같은 4가지 일반적인 EHR 분석 작업에서 기존 최첨단 모델들을 지속적으로 초월함을 입증하였습니다. 다양한 구성 요소와 하이퍼파라미터에 대한 아블레이션 연구를 통해 모델의 견고성을 입증하였습니다.



### PeriodWave: Multi-Period Flow Matching for High-Fidelity Waveform Generation (https://arxiv.org/abs/2408.07547)
Comments:
          24 pages, 16 tables, 4 figures

- **What's New**: 최근 유니버설 웨이브폼 생성 모델인 PeriodWave를 제안합니다. 이 모델은 주기적 특성을 명시적으로 분리하여 고해상도 웨이브폼 생성 문제를 개선합니다.

- **Technical Details**: PeriodWave는 주기 인식 플로우 매칭 추정기를 사용하며, 다중 주기 추정기를 통해 웨이브폼 신호의 다양한 주기적 특성을 포착합니다. 고주파 모델링을 위해 이산 웨이블릿 변환(Discrete Wavelet Transform, DWT)을 사용하고, 고주파 노이즈를 줄이기 위한 FreeU를 도입했습니다.

- **Performance Highlights**: 실험 결과, PeriodWave는 Mel 스펙트로그램 재구성과 텍스트-음성 변환 작업에서 이전 모델보다 우수한 성능을 보였습니다. 모델은 단 3일의 훈련으로도 우수한 주파수 관련 지표를 기록했습니다.



### $\chi$SPN: Characteristic Interventional Sum-Product Networks for Causal Inference in Hybrid Domains (https://arxiv.org/abs/2408.07545)
Comments:
          17 pages, 11 figures. Accepted as poster at UAI (Uncertainty in Artificial Intelligence) 2024

- **What's New**: 이번 연구에서는 혼합 변수를 다루는 인과 추론(causal inference)에 있어 새로운 접근법인 Characteristic Interventional Sum-Product Network ($\chi$SPN)을 제안합니다. 이 모델은 이산(discrete)과 연속(continuous) 변수가 혼합된 분포에서 개입(interventional) 분포를 추정할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: $\chi$SPN은 참조 기능(characteristic function)을 사용하여 이산 및 연속 랜덤 변수를 공통 스펙트럼 영역(spectral domain)에서 처리합니다. 이를 통해 분포는 Fourier-Stieltjes 변환(Fourier-Stieltjes transform)으로 통합적인 시각을 제공합니다. 개입된 데이터를 사용하여 신경망(neural network)을 통해 인과적 모델의 매개변수를 추정합니다. $\chi$SPN은 단일 개입 데이터에서 학습된 바 있지만, 여러 개입에 일반화할 수 있습니다.

- **Performance Highlights**: 세 가지 합성 이질적 데이터셋(synthetic heterogeneous datasets)을 대상으로 한 실험에서, $\chi$SPN은 이산 및 연속 변수를 위한 개입 분포를 효과적으로 캡처하는 데 성공하였으며, 표현력이 뛰어나고 인과적으로 적합함을 입증했습니다. 또한 여러 개입에 대해 재학습 없이 일반화할 수 있는 능력을 보여주었습니다.



### New Curriculum, New Chance -- Retrieval Augmented Generation for Lesson Planning in Ugandan Secondary Schools. Prototype Quality Evaluation (https://arxiv.org/abs/2408.07542)
Comments:
          Presented at Ndejje University Second Annual Research Dissemination Symposium 2024

- **What's New**: 이번 연구에서는 우간다의 중등학교 교육 질 향상을 위한 최신 접근 방식이 소개되었습니다. 새로운 커리큘럼에 대응하기 위해 정부 인증 교과서를 기반으로 맞춤형 수업 계획을 생성하는 Retrieval Augmented Generation(수집 증강 생성) 기법을 사용하여 프로토타입이 개발되었습니다.

- **Technical Details**: 프로토타입은 Cohere LLM 및 Sentence Embeddings와 LangChain Framework를 이용하여 구축되었으며, 공용 웹사이트에서 제공됩니다. ICT, 수학, 역사와 같은 세 가지 새로운 커리큘럼 교과서에 대해 벡터 저장소를 훈련시켰습니다. 24개의 수업 계획이 교과서에서 제안된 수업 기간에 따라 의사 랜덤 생성 프로토콜을 적용하여 생성되었습니다. 수업 계획은 Ndihokubwayo et al. (2022)의 Lesson Plan Analysis Protocol (LPAP)에 따라 세 명의 독립 평가자가 기술적 품질을 분석했습니다.

- **Performance Highlights**: 생성된 24개의 수업 계획에 대한 LPAP 평가 결과, 평균 품질은 75%에서 80%로 나타났으며 이는 '매우 좋은 수업 계획'에 해당합니다. 모든 수업 계획이 65% 이하의 점수를 기록하지 않았으며, 하나의 수업 계획은 주제가 누락된 것으로 평가될 수 있었습니다. 연구에 따르면, 생성된 수업 계획의 품질은 인적 자원에 의해 생성된 수업 계획의 품질과 적어도 동등하거나 더 나은 수준임을 보여 주었습니다.



### DifuzCam: Replacing Camera Lens with a Mask and a Diffusion Mod (https://arxiv.org/abs/2408.07541)
- **What's New**: 본 논문에서는 렌즈가 없는 플랫 카메라 디자인을 제안하며, 열화되는 이미지를 복원하기 위해 사전 훈련된 diffusion 모델을 사용하는 혁신적인 접근 방식을 소개합니다.

- **Technical Details**: Diffusion 모델을 활용함으로써 플랫 카메라의 복원 품질을 향상시키며, 텍스트 안내 기능을 통해 이미지 복원을 더욱 개선합니다. 이 방법은 이미지 생성을 위한 강력한 우선 정보를 제공하고, 고해상도 이미지를 보다 효율적으로 처리할 수 있도록 합니다.

- **Performance Highlights**: 제안된 DifuzCam은 기존의 영상 측정 방법과 비교하여 품질 및 인지적 측면에서 최첨단 결과를 보여줍니다. 이 방법은 다른 이미징 시스템에도 적용 가능하여 다양한 환경에서의 복원 결과를 개선할 수 있습니다.



### Cross-aware Early Fusion with Stage-divided Vision and Language Transformer Encoders for Referring Image Segmentation (https://arxiv.org/abs/2408.07539)
Comments:
          Published in IEEE Transactions on Multimedia (TMM)

- **What's New**: 본 논문에서는 Cross-aware early fusion with stage-divided Vision and Language Transformer encoders (CrossVLT)라는 혁신적인 아키텍처를 제안하여 참조 이미지 분할(referring image segmentation) 문제에 접근합니다. 이 모델은 언어 및 비전 인코더가 서로 정보를 참조할 수 있도록 하여 두 인코더의 강건함을 상호 증진시킵니다.

- **Technical Details**: CrossVLT는 각 단계에서 언어와 비전 인코더 간의 교차 정보를 양방향으로 교환할 수 있도록 설계되었습니다. 저수준(low-level)에서 고수준(high-level) 특성까지 활용하여 교차 모달 정렬(cross-modal alignment)을 수행하는 기능 기반 정렬 방식을 도입합니다. 이를 통해 각 단계에서 보다 효과적인 교차 모달 융합(cross-modal fusion)이 이루어집니다.

- **Performance Highlights**: CrossVLT는 세 가지 공개 데이터셋에서 이전의 최첨단 방법들을 능가하는 성능을 달성하였으며, 참조 이미지 분할에 대한 단순하지만 효과적인 접근 방식을 제공합니다.



### Evidential Graph Contrastive Alignment for Source-Free Blending-Target Domain Adaptation (https://arxiv.org/abs/2408.07527)
- **What's New**: SF-BTDA(소스 없는 혼합 대상 도메인 적응) 설정을 도입하고, 노이즈가 포함된 대상 의사 레이블의 영향을 완화하기 위한 새로운 방법인 ECA(증거 대조 정렬)를 제안합니다.

- **Technical Details**: ECA는 두 가지 주요 도전 과제를 해결하기 위해 설계되었습니다: (1) 고품질 의사 대상 레이블을 생성하기 위한 보정된 증거 학습 모듈과 (2) 혼합된 대상 도메인 간의 분포 차이를 최소화하기 위한 그래프 대조 학습을 통해 클래스 샘플의 분포를 조정합니다.

- **Performance Highlights**: ECA는 새로운 벤치마크를 기반으로 실험을 수행하고, 다른 방법들보다 상당한 성능 향상을 기록하며 도메인 레이블이나 소스 데이터에 접근하는 경우와 비슷한 결과를 달성합니다.



### Training Overhead Ratio: A Practical Reliability Metric for Large Language Model Training Systems (https://arxiv.org/abs/2408.07482)
Comments:
          preprint, under review

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)의 훈련 시스템의 신뢰성을 평가하기 위한 새로운 지표인 Training Overhead Ratio (TOR)를 소개합니다. 이 지표는 이상적인 시스템에서의 최적 훈련 시간 대비 실제 관측된 훈련 시간을 비율로 나타내어 사용자가 LLM 훈련 시 소요되는 시간을 추정할 수 있도록 돕습니다.

- **Technical Details**: TOR는 0에서 1 사이의 값을 가지며, 값이 높을수록 시스템의 신뢰성이 높음을 의미합니다. LLM 훈련 중 발생하는 다양한 실패 유형을 감안한 TOR 방정식이 제시되며, 성능 보존 비율(performance preservation ratio)이 시스템의 신뢰성을 개선하는 데 중요한 요소로 강조됩니다. 이 비율은 최적 작업 속도가 실질적으로 달성된 비율을 나타냅니다.

- **Performance Highlights**: 본 연구에서는 첫째, LLM 훈련 시스템을 위한 최초의 신뢰성 지표를 제안하고 그 계산 방정식을 제공하였습니다. 둘째, 시스템 신뢰성을 높이는 주요 요소를 규명하였습니다. TOR 지표의 도입은 LLM 훈련 효율성을 높이고 실제 훈련 시간을 더 잘 추정하도록 할 것입니다.



### A Study on Bias Detection and Classification in Natural Language Processing (https://arxiv.org/abs/2408.07479)
Comments:
          31 pages, 15 Tables, 4 Figures

- **What's New**: 이 논문은 공공 데이터셋을 수집하고 이를 효과적으로 결합하여 증오 발언(hate speech) 탐지 및 분류 모델을 훈련하는 방법을 제시합니다. 또한 데이터셋의 부족, 자원 편향, 비지속적인 데이터 의존성과 같은 주요 문제를 분석합니다.

- **Technical Details**: 저자들은 여러 종류의 분류기를 훈련시켜 모델의 성능을 분석하고, 데이터셋 간의 조합이 모델 성능에 미치는 영향을 보여줍니다. 또한 성별, 인종, 직업, 종교, 장애, 성적 지향, 성 정체성, 국적, 나이와 같은 사회적 특성을 기반으로 한 '타겟 카테고리'를 정의합니다.

- **Performance Highlights**: 다양한 데이터셋의 조합이 모델 성능에 미치는 직접적 영향에 대한 결과가 도출되었으며, 특정 특성과 관련된 비편향적 데이터셋의 필요성이 강조되었습니다.



### Large Language Models Prompting With Episodic Memory (https://arxiv.org/abs/2408.07465)
- **What's New**: POEM(PrOmpting with Episodic Memory)은 간단하고 효율적인 프롬프트 최적화 기법으로, 에피소드 기반 메모리를 활용하여 강화학습(RL) 문제로 프롬프트 최적화를 접근합니다. 이 방법은 기존의 자원 집약적이거나 성능이 부족한 프롬프트 최적화 방법들의 한계를 극복합니다.

- **Technical Details**: POEM은 에피소드 메모리를 사용하여 입력 데이터의 조합, 몇 개의 샷 예제의 순열 및 훈련 과정에서 관찰된 보상을 기록합니다. 검증 단계에서, 상위 k개의 가장 유사한 훈련 예제에서 발생한 총 보상을 기준으로 테스트 쿼리에 대한 예제 순서를 최적화합니다.

- **Performance Highlights**: 여러 텍스트 분류 작업에서 POEM은 TEMPERA와 RLPrompt보다 각각 5.3% 및 13.4% 더 뛰어난 성능을 보였습니다. POEM은 일반적인 휴리스틱 방법보다 모든 테스트에 걸쳐 일관된 우수한 성능을 보여주며, 다양한 언어 이해 작업에 잘 적응합니다.



### Fact or Fiction? Improving Fact Verification with Knowledge Graphs through Simplified Subgraph Retrievals (https://arxiv.org/abs/2408.07453)
Comments:
          10 pages, 3 figures, appendix

- **What's New**: 이번 연구에서는 증가하는 허위 정보에 대응하기 위해, FactKG 데이터셋을 활용한 검증 모델의 성능을 개선하는 다양한 효율적인 방법을 제안합니다. 특히, 구조화된 지식 그래프를 기반으로 한 증거 검색을 단순화하여 모델의 정확도를 높였습니다.

- **Technical Details**: 연구에서는 다양한 모델 아키텍처를 사용하여 과제를 다룹니다. 여기에는 BERT 모델을 사용하는 Textual Fine-tuning, 구조적 처리를 위해 QA-GNN(Graph Neural Network)을 활용하는 Hybrid Graph-Language Model, 추가적인 fine-tuning 없이 최신 LLM(Language Model)을 활용하는 LLM Prompting이 포함됩니다. 이 과정을 통해, 수학적 효율성을 높이고, 모델의 훈련 시간을 대폭 단축시켰습니다.

- **Performance Highlights**: 모델은 테스트 세트의 정확도를 77.65%에서 93.49%로 크게 향상시켰으며, 훈련 시간은 1.5~10시간으로 단축되었습니다. 이는 이전의 벤치마크 모델(2-3일 소요) 대비 상당한 개선을 나타냅니다.



### CMU's IWSLT 2024 Simultaneous Speech Translation System (https://arxiv.org/abs/2408.07452)
- **What's New**: CMU가 IWSLT 2024 Simultaneous Speech Translation (SST) 과제에 제출한 새로운 시스템을 소개합니다. 이 시스템은 영어 음성을 독일어 텍스트로 실시간으로 번역하는 데 중점을 두고 있으며, WavLM 스피치 인코더와 Llama2-7B-Base 모델을 사용하는 종단 간 (end-to-end) 음성-텍스트 시스템을 개발했습니다.

- **Technical Details**: 시스템은 세 가지 주요 구성 요소로 구성됩니다: 스피치 인코더, 모달리티 어댑터(modality adapter), LLM 디코더. 스피치 인코더로는 WavLM을 사용하고, 모달리티 어댑터는 길이 어댑터와 선형 변환 레이어로 구성됩니다. 모델 훈련은 두 단계로 나뉘며, 먼저 스피치와 텍스트 표현을 정렬한 후, 전체 모델을 미세 조정합니다.

- **Performance Highlights**: 우리 모델은 MuST-C-v2 tst-COMMON 데이터셋에서 오프라인 BLEU 점수 31.1을, 2초 지연 조건에서의 BLEU 점수 29.5를 달성했습니다.



### LiveFC: A System for Live Fact-Checking of Audio Streams (https://arxiv.org/abs/2408.07448)
Comments:
          Under Review, 11 pages

- **What's New**: 디지털 시대의 정보 확산이 빠르게 증가하면서 허위 정보와 잘못된 정보의 전파도 심각해지고 있습니다. 본 연구에서는 실시간으로 라이브 오디오 스트림을 사실 확인할 수 있는 도구인 LiveFC를 개발하였습니다.

- **Technical Details**: LiveFC는 6개의 주요 구성 요소로 이루어져 있습니다: 1) 실시간 데이터에 작동하는 전사 모듈, 2) 오디오 세그먼트의 화자 식별을 위한 다이어리제이션 모듈, 3) 실시간으로 확인이 필요한 주장을 식별하는 주장 탐지 모듈, 4) 주장 분해 및 주제 할당 모듈, 5) 웹 검색 및 이전 사실 확인에서 최신 증거를 검색하는 증거 검색 모듈, 6) 최신의 미세 조정된 자연어 추론(NLI) 모델을 사용하는 주장 검증 모듈입니다.

- **Performance Highlights**: LiveFC는 디지털 환경에서 사용자에게 실시간으로 잘못된 정보에 대한 검증된 사실을 제공하여 사회적 불안을 예방하고, 선거 토론 및 캠페인과 같은 중요한 실시간 이벤트에서의 사실 확인을 지원합니다. 실제 검증에서는 덴마크의 사실 확인 팀과 미국의 대통령 후보 토론에서 효과를 입증하였습니다.



### Achieving Data Efficient Neural Networks with Hybrid Concept-based Models (https://arxiv.org/abs/2408.07438)
Comments:
          11 pages, 8 figures, appendix

- **What's New**: 본 논문에서는 기존의 클래스 라벨과 추가 정보인 개념(concept)을 함께 사용하는 하이브리드 개념 기반 모델(Hybrid Concept-Based Models)을 소개합니다. 이 모델은 데이터셋에 있는 더 많은 정보를 활용하여 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 하이브리드 모델 아키텍처는 기본적으로 기존의 개념 병목 모델(CBM)을 기반으로 하며, 개념 병목 레이어를 거치지 않는 추가 스킵 연결(skip connection)을 통하여 예측을 수행합니다. 또한, Sequential Concept Model(SCM)이라는 아키텍처를 제안하여 모든 개념 예측을 신경망의 여러 층에서 순차적으로 수행합니다. 이를 통해 데이터가 부족한 상황에서도 성능을 유지할 수 있도록 하였습니다.

- **Performance Highlights**: 하이브리드 개념 기반 모델은 기존의 컴퓨터 비전 모델 및 이전에 제안된 개념 기반 모델들과 비교했을 때, 특히 데이터가 희소한 환경에서 정확도(accuracy) 측면에서 우수한 성능을 보였습니다. 또한, 적대적 개념 공격(adversarial concept attacks) 알고리즘을 통해 개념 기반 모델의 해석 가능성에 대한 새로운 질문을 제기하였습니다.



### Real-world validation of safe reinforcement learning, model predictive control and decision tree-based home energy management systems (https://arxiv.org/abs/2408.07435)
- **What's New**: 최근 머신러닝을 기반으로 한 에너지 관리 접근법인 reinforcement learning(강화 학습)과 metaheuristic algorithm(메타 휴리스틱 알고리즘)을 활용한 연구가 진행되었습니다. 이 연구에서는 OptLayerPolicy(안전 계층)와 TreeC(결정 트리 제어 정책)을 통해 실제 적용 가능성을 검증했습니다.

- **Technical Details**: 본 연구에서는 4개 주택의 전기 설치를 이용하여 실험을 수행하였으며, 각 주택은 배터리, 태양광 발전기, 비제어 전기 부하, 제어 가능한 전기차 충전기 시스템을 갖추고 있습니다. 실험 결과, TreeC와 모델 예측 제어(model predictive control) 기반 방법은 유사한 비용을 기록했으며, 강화 학습 기반 방법은 다른 방법들보다 25.5% 높은 비용을 보였습니다.

- **Performance Highlights**: 실험 결과, TreeC 방법이 가장 안전한 운영 성능을 보였으며, 강화 학습 방법은 그리드 한계를 593.9 Wh 초과한 반면, TreeC는 27.1 Wh 초과하였습니다. 오차를 줄이기 위해 보다 대표적인 훈련 데이터셋을 사용하고, 모델 예측 제어 구현에서 데이터 의존성을 해결하면 비용을 더욱 줄일 수 있습니다.



### MagicFace: Training-free Universal-Style Human Image Customized Synthesis (https://arxiv.org/abs/2408.07433)
Comments:
          project page: this https URL

- **What's New**: MagicFace는 고유의 스타일을 기반으로 한 인간 이미지를 개인화하여 생성하는 새로운 접근 방식으로, 훈련 없이 단일 및 다중 개념을 커스터마이징할 수 있는 첫 번째 방법입니다.

- **Technical Details**: MagicFace는 세 가지 단계로 구성된 생성 파이프라인을 통해 작동합니다: 첫 번째 단계에서는 Reference-aware Self-Attention (RSA)을 사용하여 참조 개념으로부터 특징을 추출하고, 두 번째 단계에서는 Region-grouped Blend Attention (RBA)을 통해 생성된 이미지의 각 개념이 갖는 세부 특징을 정확히 주입합니다. 이 모든 과정에서 가중치 마스크 전략을 활용하여 모델이 참조 개념에 더 집중하도록 합니다.

- **Performance Highlights**: 실험 결과, MagicFace는 인간의 이미지를 생성하는 작업뿐만 아니라 다중 개념 커스터마이징에서도 매우 우수한 성능을 보였습니다. 사용자 연구를 통해, 대부분의 참가자들이 MagicFace의 결과에 긍정적인 피드백을 주었으며, 훈련이 필요 없는 방식으로도 높은 품질의 결과를 생성할 수 있음을 입증했습니다.



### Exploring Retrieval Augmented Generation in Arabic (https://arxiv.org/abs/2408.07425)
- **What's New**: 최근 자연어 처리(NLP) 분야에서 Retrieval Augmented Generation(RAG) 기법이 주목받고 있으며, 본 논문은 아랍어에서의 RAG 적용 사례를 종합적으로 연구합니다.

- **Technical Details**: 이 연구는 아랍어 텍스트에 대한 RAG의 구현 및 평가를 중점적으로 다루며, 검색 단계에서 다양한 semantic embedding 모델과 생성 단계에서 여러 LLM(대형 언어 모델)을 탐구합니다. 단어와 구문 간의 방언 차이로 인해 발생할 수 있는 문제 또한 고려합니다.

- **Performance Highlights**: 결과적으로, 기존의 semantic embedding 모델과 LLM들이 아랍어 RAG 파이프라인 구축에 효과적으로 활용될 수 있음을 보여줍니다.



### LLMI3D: Empowering LLM with 3D Perception from a Single 2D Imag (https://arxiv.org/abs/2408.07422)
- **What's New**: 최근 자율주행, 증강현실, 로보틱스, 및 구현된 지능(embodied intelligence) 분야의 발전은 3D 인식 알고리즘(algorithms) 필요성을 증가시켰습니다. 그러나 기존의 3D 인식 방법들은 논리적 추론, 질문-답변 처리 및 개방 시나리오 범주를 다루는 데 어려움을 겪고 있습니다. 본 연구는 LLMI3D라는 강력한 3D 인식 MLLM을 제안하며, IG3D 데이터셋을 구축하여 세부 기술 및 질문-답변 주석을 제공합니다.

- **Technical Details**: LLMI3D는 사전 훈련된 MLLM에 파라미터 효율적인 미세 조정을 적용하고, 공간 강화 로컬 기능 추출(Spatial-Enhanced Local Feature Mining), 3D 쿼리 토큰 기반 정보 디코딩(3D Query Token-Derived Info Decoding), 기하학적 투영 기반 3D 추론(Geometry Projection-Based 3D Reasoning) 방법을 도입하여 카메라 초점 길이 변화를 처리하고 공간적 특징을 효과적으로 추출하는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, LLMI3D는 기존의 방법들에 비해 월등한 성능을 보이며, 최신 최첨단 성능을 달성했습니다. 이 연구는 3D 인식 분야의 새로운 가능성을 열고 있으며, 다양한 현실 세계 문제 해결에 기여할 것으로 기대됩니다.



### Efficient Edge AI: Deploying Convolutional Neural Networks on FPGA with the Gemmini Accelerator (https://arxiv.org/abs/2408.07404)
Comments:
          8 pages, 9 figures, accepted at the 27th Euromicro Conference Series on Digital System Design (DSD) 2024

- **What's New**: 이 논문에서는 에지(Edge) 장치에서 Convolutional Neural Networks (CNNs)를 효율적으로 배포하기 위한 새로운 end-to-end workflow를 제시합니다. 특히, Gemmini accelerator를 수정하여 Field Programmable Gate Arrays (FPGAs)를 위한 최적화된 구현 방법을 다룹니다.

- **Technical Details**: 연구팀은 배포 프로세스의 각각의 최적화 단계에서 오픈 소스 소프트웨어를 활용하고, 이들 소프트웨어에 대한 커스터마이징(customizations)을 통해 최종 시스템 성능에 미치는 영향을 분석하였습니다. YOLOv7 모델을 Xilinx ZCU102 FPGA에 배포하여 36.5 GOP/s/W의 에너지 효율성과 실시간 성능을 달성했습니다.

- **Performance Highlights**: FPGA 기반 솔루션은 다른 임베디드 하드웨어 장치들에 비해 우수한 전력 효율성을 입증하였고, 다른 FPGA 참고 구현보다도 더 나은 성능을 보여주었습니다. 마지막으로, 제안된 플랫폼을 교통 모니터링 시나리오에서 테스트하여 넓은 시스템으로의 통합 가능성을 제시합니다.



### A Quantum-Inspired Analysis of Human Disambiguation Processes (https://arxiv.org/abs/2408.07402)
Comments:
          PhD thesis

- **What's New**: 본 논문에서는 자연어 처리(NLP)에서 발생하는 모호성을 해결하기 위해 양자 역학(quantum mechanics)의 개념을 활용했습니다. 기존의 NLP 접근 방식과 비교하여 실질적인 양자 이점을 탐구하는 연구가 이루어졌습니다.

- **Technical Details**: 논문에서는 양자 메커니즘에서 유래한 개념인 맥락성(contextuality)과 인과성(causality)을 적용하여 언어학에서의 모호성을 연구했습니다. 이러한 접근법은 심리언어학(psycholinguistics)에서 인간의 모호성 해소 과정과 관련된 결과들을 재현하는 데 기여했습니다.

- **Performance Highlights**: 이 연구의 결과는 인간 행동을 예측하는 데 사용되었으며, 기존의 NLP 메소드보다 더 우수한 성과를 보였습니다.



### DataVisT5: A Pre-trained Language Model for Jointly Understanding Text and Data Visualization (https://arxiv.org/abs/2408.07401)
- **What's New**: 본 논문에서는 DataVisT5라는 새로운 Pre-trained Language Model (PLM)을 소개합니다. 이는 자연어와 데이터 시각화(DV)의 상호 이해를 가능하게 하고, T5 아키텍처를 개선하기 위해 하이브리드 목적의 사전 훈련 및 다중 작업 미세 조정 전략을 채택했습니다.

- **Technical Details**: DataVisT5는 DL(Domain Language) 데이터셋과 텍스트 데이터셋을 통합하여 서로 다른 모달리티 간의 의미를 효과적으로 해석할 수 있도록 설계되었습니다. 주요 과제는 텍스트에서 DV로 변환(text-to-vis), DV에서 텍스트로 변환(vis-to-text), 데이터 시각화 관련 질문 응답(FeVisQA), 테이블에서 텍스트로 설명(table-to-text) 등을 포함합니다.

- **Performance Highlights**: Public datasets에 대한 광범위한 평가를 통해 DataVisT5가 다양한 DV 관련 작업에서 기존의 최첨단 모델(SOTA)을 지속적으로 초월하는 성능을 보여주었습니다.



### Improving Global Parameter-sharing in Physically Heterogeneous Multi-agent Reinforcement Learning with Unified Action Spac (https://arxiv.org/abs/2408.07395)
- **What's New**: 이번 연구에서는 물리적으로 이질적인 다중 에이전트 시스템(MAS)에서 에이전트 행동의 의미(Action semantics)에 따라 에이전트를 그룹으로 나눌 수 있는 새로운 접근법인 통합 행동 공간(United Action Space, UAS)을 소개합니다. 이는 다양한 행동 의미를 처리하면서도 글로벌 파라미터 공유 구조를 유지하기 위한 것입니다.

- **Technical Details**: UAS는 다양한 행동 의미를 가진 모든 에이전트 행동의 합집합입니다. 여러 에이전트는 먼저 UAS에서 자신의 통합 표현을 계산한 후, 사용 가능한 행동 마스크(available-action-masks)를 사용하여 이질적인 행동 정책을 생성합니다. 또한 Cross-Group Inverse (CGI) 손실 함수를 도입하여 그룹의 궤적 정보를 활용하여 다른 그룹의 에이전트 정책을 예측합니다.

- **Performance Highlights**: SMAC 환경에서 실험 결과 U-QMIX와 U-MAPPO 알고리즘이 여러 최첨단 MARL 방법들과 비교하여 뛰어난 성능을 보임을 입증하였습니다. 이 두 알고리즘은 파라미터 공유의 장점을 활용하면서 높은 승률을 기록했습니다.



### Sum-Product-Set Networks (https://arxiv.org/abs/2408.07394)
- **What's New**: 이번 논문은 비구조적 텐서 데이터에서 트리 구조 그래프 데이터로 확장된 확률 회로(Probabilistic Circuits)를 제안하는 sum-product-set networks를 소개합니다.

- **Technical Details**: 이 모델은 무작위 유한 집합(Random Finite Sets)을 사용하여 그래프에서 노드와 엣지의 가변 개수를 반영하며, 정확하고 효율적인 추론(Inference)을 가능하게 합니다.

- **Performance Highlights**: 우리의 효율적인 모델은 여러 신경망 기반 비효율적 모델과 유사한 성능을 보여줍니다.



### Do GPT Language Models Suffer From Split Personality Disorder? The Advent Of Substrate-Free Psychometrics (https://arxiv.org/abs/2408.07377)
Comments:
          37 pages, 7 figures, 3 tables, date v1: Mar 26 2023

- **What's New**: 이번 연구에서는 다양한 언어로 된 동일한 성격 질문지를 사용하여 언어 모델의 심리적 특성을 분석함으로써 이들 모델의 일관되지 않은 성격 개발을 확인하였습니다. 이 결과는 인공지능 시스템의 안전성에 대한 우려를 제기합니다.

- **Technical Details**: 연구진은 Bayesian 분석을 바탕으로 Gaussian Mixture Model을 적용하여 다국어로 측정된 성격 특성의 불안정성을 밝혀냈습니다. 특히, 현재의 대형 언어 모델(LLMs)은 일관성 있는 핵심 성격을 발전시키지 못한다는 점이 강조되었습니다.

- **Performance Highlights**: 대형 언어 모델인 GPT-3와 InstructGPT는 Dark Triad(나르시시즘, 정신병적 성향, 마키아벨리즘) 점수가 높았으며, Big5 성격 요인에서는 평균 이상의 점수를 보였습니다. 그러나 이들 모델에서 발생하는 어두운 성격 특성은 일반적인 긍정적인 성격과 함께 존재하여 그 안전성 문제를 더욱 부각시켰습니다.



### The Complexity of Manipulation of k-Coalitional Games on Graphs (https://arxiv.org/abs/2408.07368)
- **What's New**: 본 논문은 $k$-coalitional 게임 및 사회적 조작(socially-aware manipulation)의 복잡성을 분석하고, 이러한 조작이 사회적 복지(social welfare)를 해치지 않으면서 자신의 효용(utility)을 증가시킬 수 있는 경우를 살펴봅니다.

- **Technical Details**: 연구는 비가중 그래프(undirected graph)에서의 우정 관계(friendship connections)를 기반으로 하여, 각 에이전트의 효용은 코얼리션(coalition) 내에서의 친구 수에 따라 결정됨을 밝힙니다. 또한, 최대 공리 사회 복지(max utilitarian social welfare)와 상대적으로 적은 요구 사항을 가진 에이전트에 대한 사회 복지 최대화(max egalitarian social welfare) 문제를 다룹니다. 새로운 유형의 조작인 사회적 조작(SAM)에 대한 접근도 소개합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면 사회적 조작은 빈번하게 발생하며 제시된 XP 알고리즘의 효과성을 나타냅니다.



### RTAT: A Robust Two-stage Association Tracker for Multi-Object Tracking (https://arxiv.org/abs/2408.07344)
Comments:
          ICPR2024

- **What's New**: 이번 논문에서는 Multi-Object Tracking (MOT)에서의 데이터 연관(data association)의 중요성을 강조하며, 복잡한 장면에서의 일반화 능력을 향상시키기 위해 Robust Two-stage Association Tracker (RTAT)라는 새로운 방법을 제안합니다.

- **Technical Details**: RTAT는 두 단계의 연관 과정을 통해 동작합니다. 첫 번째 단계에서는 낮은 매칭 비용(threshold)을 설정하여 순수성이 높은 tracklets를 생성하고, 두 번째 단계에서는 메시지 전달(message-passing) GNN 프레임워크를 기반으로 tracklet 간의 연관을 수행합니다. 이를 통해 단기 tracklet들을 장기 tracklet로 재귀적으로 병합할 수 있습니다.

- **Performance Highlights**: RTAT는 MOT17 및 MOT20 벤치마크의 테스트 세트에서 HOTA, IDF1, AssA와 같은 주요 MOT 메트릭에서 1위를 기록했습니다. MOT17에서 67.2 HOTA, 84.7 IDF1, 69.7 AssA를 달성하였고, MOT20에서 66.2 HOTA, 82.5 IDF1, 68.1 AssA를 기록하였습니다.



### Towards Few-shot Self-explaining Graph Neural Networks (https://arxiv.org/abs/2408.07340)
- **What's New**: 본 연구에서는 몇 샷(few-shot) 상황에서 예측을 지원하기 위해 설명을 생성하는 새로운 프레임워크인 Meta-learned Self-Explaining GNN (MSE-GNN)을 제안합니다. 이 모델은 두 단계의 자가 설명 구조로 구성되어 있으며, 인과 추론에 필요한 고품질 설명을 생성합니다.

- **Technical Details**: MSE-GNN은 인과 추론을 위해 두 단계로 구성된 설명자(explainer)와 예측자(predictor)를 활용합니다. 설명자는 인간의 주의(attention) 메커니즘을 모방하여 중요한 특성을 포함한 서브그래프를 선택합니다. 예측자는 생성된 설명을 기반으로 예측을 수행합니다. 또한, 메타 학습(meta-learning) 과정과 작업(task) 정보 활용 메커니즘을 설계하여 새로운 상황에서도 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: MSE-GNN은 네 개의 데이터셋에 대한 실험에서 기존 방법들에 비해 예측 작업에서 우수한 성능을 보였습니다. 특히, 제한적인 데이터 상황에서도 높은 품질의 설명을 생성할 수 있어 큰 주목을 받고 있습니다.



### An Offline Meta Black-box Optimization Framework for Adaptive Design of Urban Traffic Light Management Systems (https://arxiv.org/abs/2408.07327)
Comments:
          12 pages, 7 figures, 10 tables

- **What's New**: 이 논문은 복잡한 도시 도로 네트워크에서의 교통 신호 관리 체계를 위한 새로운 최적화 프레임워크를 소개합니다. 특히 기존의 정적인 교통 패턴이 아닌 다양한 교통 패턴에 적응할 수 있도록 하는 두 가지 핵심 구성요소인 phase combination과 phase time allocation을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 오프라인 메타 블랙박스 최적화(offline meta black-box optimization) 기법을 사용합니다. 이 시스템은 Attentive Neural Process (ANP)를 통해 다양한 교통 패턴에서의 설계가 교통 혼잡에 미치는 영향을 예측하고, Bayesian optimization(Bayesian 최적화)을 사용하여 보지 못한 교통 패턴에 대한 최적 설계를 찾습니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 복잡한 도로 네트워크에서 대기 차량 숫자 측면에서 기존의 최신 기술(state-of-the-art)보다 우수한 성능을 보였습니다. 실제 도로 시스템에 배포한 결과, 원래 전략에 비해 교통 처리량이 4.80% 향상되었습니다.



### Kolmogorov-Arnold Networks (KAN) for Time Series Classification and Robust Analysis (https://arxiv.org/abs/2408.07314)
Comments:
          14 pages, 8 figs

- **What's New**: Kolmogorov-Arnold Networks (KAN)이 전통적인 Multi-Layer Perceptrons (MLP)의 유망한 대안으로 주목받고 있습니다. KAN의 성능을 128개의 시계열 데이터셋에서 검증한 결과, MLP보다 유사하거나 약간 더 나은 성능을 기록했습니다.

- **Technical Details**: KAN은 Kolmogorov-Arnold 이론을 기반으로 한 신경망으로, MLP와 유사한 구조를 가지고 있으며, 함수 근사 시 필요한 모델 크기를 명시적으로 정의합니다. KAN은 learnable activation function을 갖고 있어 MLP의 유력한 경쟁자가 될 수 있습니다.

- **Performance Highlights**: KAN과 혼합 구조인 MLP_KAN은 Adversarial robustness가 우수함을 보여주었고, KAN의 하위구조는 시스템의 성능에 주된 영향을 미쳤습니다. 연구 결과 KAN은 adversarial 공격에 대해 낮은 Lipschitz 상수를 가지고 있어 더 뛰어난 견고함을 나타냈습니다.



### Learning Multi-Modal Whole-Body Control for Real-World Humanoid Robots (https://arxiv.org/abs/2408.07295)
Comments:
          Website: this https URL

- **What's New**: 이번 논문에서는 Masked Humanoid Controller (MHC)를 도입하여 다양한 수의 휴머노이드 상태 변수를 통해 목표 궤적(trajectories)을 전체적으로 추적할 수 있도록 하였습니다. 이 시스템은 비디오, 모션 캡처(motion capture), VR 등의 다양한 출처에서 얻은 정보를 바탕으로 전체 몸 동작을 구현할 수 있습니다.

- **Technical Details**: MHC는 부분적으로 마스킹된 동작을 모방하는 실험적 커리큘럼을 통해 훈련됩니다. 이 커리큘럼은 사전 훈련된 정책 롤아웃(policy rollouts), 최적화된 참조 궤적(reference trajectories), 재타겟팅된 비디오 클립, 인간 모션 캡처 데이터로 구성된 행동 라이브러리를 활용하여 구성됩니다. MHC는 또한 강화 학습(reinforcement learning)을 통해 다양한 동작 명령 및 방해 요소를 점진적으로 훈련하여 균형 및 강건성을 유지할 수 있도록 합니다.

- **Performance Highlights**: MHC는 여러 가지 목표 동작을 실행할 수 있는 능력을 보여주며, Digit 휴머노이드 로봇을 통해 실세계에서의 시뮬레이션 전환 능력도 입증됩니다. 이 시스템은 걷기, 복싱, 상자 집기와 같은 다양한 목표 행동에서 강건성을 보여주어 현재의 시뮬레이션 대 현실 전환의 한계를 강조합니다.



### SumRecom: A Personalized Summarization Approach by Learning from Users' Feedback (https://arxiv.org/abs/2408.07294)
- **What's New**: 다수 문서 요약(Multi-document summarization) 기법의 한계를 극복하고 개인의 관심사를 반영한 사용자 맞춤형 요약을 제공하는 새로운 접근 방식인 SumRecom을 제안합니다.

- **Technical Details**: SumRecom은 사용자 피드백을 기반으로 하여 사용자의 선호도를 추출하고 개인화된 요약 결과를 생성하는 상호작용 기반 요약 기법입니다. 두 단계로 나눌 수 있으며: 1) 사용자 선호도 추출기(User Preference Extractor), 2) 요약기(Summarizer)입니다. SumRecom은 Active Learning과 Preference Learning을 활용하여 사용자의 콘텐츠 선택에서의 선호도를 추출하고, Integer Linear Programming (ILP)과 Inverse Reinforcement Learning (IRL)을 통해 요약의 품질을 평가합니다.

- **Performance Highlights**: 다양한 자동화된 평가 및 인간 평가에서 SumRecom이 사용자 맞춤형 요약을 생성하는 데 있어 탁월한 성능을 나타내었으며, 사용자의 피드백을 반영하여 최적의 요약을 생성하는 능력이 입증되었습니다.



### LiPCoT: Linear Predictive Coding based Tokenizer for Self-supervised Learning of Time Series Data via Language Models (https://arxiv.org/abs/2408.07292)
Comments:
          17 pages, 5 figures

- **What's New**: 이번 연구에서는 LiPCoT (Linear Predictive Coding 기반의 Tokenizer)를 제안하여 시계열 데이터를 토큰의 시퀀스로 인코딩하여, BERT와 같은 기존 언어 모델 아키텍처를 통해 자기지도 학습을 가능하게 합니다.

- **Technical Details**: LiPCoT는 전통적인 CNN 인코더에 의존하는 대신, 선형 예측 코딩(Linear Predictive Coding)을 통해 시계열 데이터의 잠재 공간을 생성하며, 이는 데이터의 내재적 확률적 성질을 간결하면서도 풍부하게 표현합니다. 또한, LiPCoT는 계산 효율성이 뛰어나고 다양한 샘플링 주기와 길이를 가진 시계열 데이터를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: LiPCoT를 이용하여 46명의 참가자로부터 얻은 EEG 데이터로 파킨슨병 분류를 수행한 결과, BERT 모델이 기존 방법보다 각각 7.1%의 정밀도, 2.3%의 재현율, 5.5%의 정확도, 4%의 AUC, 5%의 F1 점수 향상된 성능을 보였습니다.



### Abductive Reasoning in a Paraconsistent Framework (https://arxiv.org/abs/2408.07287)
Comments:
          This is an extended version of a paper with the same title appearing at the 21st International Conference on Principles of Knowledge Representation and Reasoning (KR 2024)

- **What's New**: 이 논문은 기존의 비일관적 이론을 설명하는 데 있어 paraconsistent 프레임워크를 채택하여 새로운 접근 방식을 제안합니다. 특히, Belnap--Dunn paraconsistent 네 가지 값 논리인 $	extsf{BD}$의 두 가지 확장을 고려하며, $	extsf{BD}_ullet$와 $	extsf{BD}_	riangle$에서의 설명 문제를 정의하고 동기를 부여합니다.

- **Technical Details**: 이 논문은 $	extsf{BD}_ullet$와 $	extsf{BD}_	riangle$를 기반으로 한 abductive reasoning의 복잡성을 분석합니다. 논리적 일관성이 보장된 원리와 설명을 정의하며, 고전적인 명제 논리에 대한 abductive reasoning 절차를 재사용할 수 있는 방식을 제시합니다. 또한, 논리적 추론 작업의 복잡성(해결 인식, 해결 존재, 가설의 관련성/필요성)에 대한 다각적인 검토가 이루어집니다.

- **Performance Highlights**: 제안된 접근 방식은 고전 논리에서 발생할 수 있는 모순 문제를 해결하는 데 효과적이며, 고전 명제 논리에서의 기존 절차와 기술들을 재사용할 수 있다는 점에서 유용함을 보입니다. 이는 AI 응용 분야에서의 진단 및 상식적 추론 등에 기여할 수 있는 가능성을 지니고 있습니다.



### Queries With Exact Truth Values in Paraconsistent Description Logics (https://arxiv.org/abs/2408.07283)
Comments:
          This is an extended version of a paper with the same title appearing at the 21st International Conference on Principles of Knowledge Representation and Reasoning (KR 2024)

- **What's New**: 이 논문에서는 네 가지 Belnapian 값을 사용하는 새로운 대칭을 가진 비일관적 설명논리(Description Logic, DL) 지식 베이스 쿼리 방법론을 제시합니다. 특별히 쿼리 언어에서 참 값 연산자를 허용하여 모순적인 증거와 긍정적인 증거가 있는 응답을 구분할 수 있는 방법을 강조합니다.

- **Technical Details**: 비일관적 $	ext{ALCHI}$ 및 그 하위 논리에서 쿼리를 답변할 때의 결합 및 데이터 복잡성을 정확히 파악할 수 있도록 고전 DL 쿼리 응답으로의 축소를 보여줍니다. Horn DL에 대해 처리 가능한 데이터 복잡성은 유지됨을 보여줍니다.

- **Performance Highlights**: Repair 기반 비일관성 수용 의미론과의 비교를 통해 두 접근 방식이 비교할 수 없음을 보여줍니다.



### Consistency Based Weakly Self-Supervised Learning for Human Activity Recognition with Wearables (https://arxiv.org/abs/2408.07282)
- **What's New**: 이 논문에서는 인간 활동 인식(HAR) 분야에서 수집된 데이터의 라벨 부족 문제를 해결하기 위한 약한 자기 지도 학습 접근 방식을 제안합니다. 모델은 두 단계로 구성되며, 첫 번째 단계에서는 데이터의 본질을 학습하여 유사한 활동을 동일한 임베딩 공간에 그룹화하고, 두 번째 단계에서는 몇 가지 샘플의 유사성 정보를 활용하여 모델을 미세 조정합니다.

- **Technical Details**: 제안된 접근 방식은 사람이 사회에 대한 도메인 지식을 활용하고 대량의 비라벨 데이터와 적은 양의 유사성 정보를 이용하여 모델의 학습 작업을 단순화하는 방법입니다. 이 과정에서 오토인코더(Autoencoder)와 잔차 신경망(ResNet), 시암 네트워크(Siamese Networks)를 활용하여 데이터 간의 패턴과 의미적 유사성을 포착합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋에서 실험을 통해 제안된 프레임워크의 효과성이 입증되었으며, 제안된 접근 방식은 완전한 라벨이 적용된 데이터셋에 직접 적용된 순수 감독 기법과 유사한 성능을 클러스터링 알고리즘이 달성하는 데 도움을 줄 수 있음을 보여주었습니다.



### Scene-wise Adaptive Network for Dynamic Cold-start Scenes Optimization in CTR Prediction (https://arxiv.org/abs/2408.07278)
Comments:
          10 pages, 6 figures, accepted by Recsys 2024

- **What's New**: 최신 모바일 E-commerce에서 사용자에게 근처 상업 서비스 추천을 제공하는 것이 점점 더 중요해지고 있습니다. 기존의 추천 시스템들은 새로운 장면에서의 cold-start 문제를 해결하는 데 어려움을 겪고 있습니다. 본 연구에서는 Scene-wise Adaptive Network (SwAN)을 제안하여 이러한 문제를 해결하였습니다.

- **Technical Details**: SwAN은 장면 유사성 학습, 사용자별 장면 전환 인지, 새로운 장면을 위한 장면별 정보 구축, 장면 간의 논리 정보 차이 강화 등 여러 가지 기능을 포함하고 있습니다. 추천 시스템은 Embedding&MLP(다층 퍼셉트론) 패러다임을 따르며, Scene Relation Graph(SRG) 및 Similarity Attention Network(SAN)를 통해 장면 간의 유사성을 파악하고, Adaptive Ensemble-experts Module(AEM)을 이용해 공유 및 특정 정보를 추출합니다.

- **Performance Highlights**: SwAN은 Meituan의 온라인 추천 서비스에 성공적으로 배포되어 기존 모델 대비 5.64%의 클릭률(CTR) 개선 효과를 보였습니다. 또한, 하루 주문량 비율이 5.19% 증가하는 성과를 달성하였습니다.



### Ensemble architecture in polyp segmentation (https://arxiv.org/abs/2408.07262)
- **What's New**: 본 연구에서는 폴립 세분화 (polyp segmentation)에 뛰어난 모델들을 평가하고, 다양한 모델의 장점을 활용하는 통합 프레임워크를 제안합니다. CNN(convolutional neural networks)과 transformer 모델에서 학습된 특징을 융합하여 최적 예측 결과를 달성하는 방식을 앙상블 기법으로 간주합니다.

- **Technical Details**: 제안된 아키텍처는 CNN과 transformer의 학습 특징을 결합하여 복합적인 세분화 성능을 구현합니다. CNN은 지역 구조를 잘 포착하고, transformer는 전역 정보를 잘 모델링합니다. 연구에서는 Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB 등 5개의 데이터셋에 대해 실험을 진행했습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 다른 기존 모델들에 비해 더 높은 학습 능력과 강인성을 보여주었습니다. 특히, CNN과 transformer의 장점을 모두 활용하여 폴립 세분화 성능을 크게 향상시켰습니다.



### GRIF-DM: Generation of Rich Impression Fonts using Diffusion Models (https://arxiv.org/abs/2408.07259)
Comments:
          Accepted to ECAI2024

- **What's New**: 이번 연구에서는 레터와 감정을 정의하는 키워드 집합을 입력으로 활용하여 특정 인상을 생생하게 표현하는 폰트를 생성하는 새로운 확산 기반 방법인 GRIF-DM을 제안합니다. 이는 기존 GAN 기반 방법에서 관찰되는 추가적인 보조 손실의 필요성을 없애면서 인상 폰트 생성을 위한 혁신적인 접근법입니다.

- **Technical Details**: GRIF-DM은 U-Net 아키텍처를 설계하고, Dual Cross-Attention 모듈을 도입하여 문자와 인상 키워드 정보를 효과적으로 통합합니다. 입력으로 한 글자와 여러 개의 인상 키워드를 받아들이며, 이를 통해 사용자 맞춤형 폰트를 생성할 수 있는 능력을 갖추고 있습니다. 또한, 사전 훈련된 BERT 모델을 사용하여 문자열 형태로 결합한 인상 키워드를 임베딩하여 사용합니다.

- **Performance Highlights**: 실험 결과, GRIF-DM은 사용자 특정 요구 사항에 맞춰 사실적이고 생동감 있으며 고충실도의 폰트를 생성할 수 있음을 입증하였습니다. 현재 사용자는 270,000개 이상의 다양한 폰트 디자인에 접근할 수 있지만, GRIF-DM은 더 높은 유연성과 저항력을 제공하여 꿈의 폰트 생성을 가능하게 합니다.



### Enhancing Autonomous Vehicle Perception in Adverse Weather through Image Augmentation during Semantic Segmentation Training (https://arxiv.org/abs/2408.07239)
- **What's New**: 이 논문은 자율주행 차량의 내비게이션 및 위치 확인에서 강력한 인식(robust perception)의 중요성을 강조하며, 다양한 날씨 조건을 고려한 세분화 모델의 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구진은 CARLA라는 3D 자율주행 차량 시뮬레이터를 사용하여 10개 마을에서 맑은 날씨 조건 아래 29개 클래스로 구성된 1200장의 이미지를 수집하고, 추가로 다양한 날씨 효과가 적용된 1200장의 이미지를 모았습니다. 이 데이터를 통해 세그멘테이션(semantic segmentation)을 수행하기 위해 encoder-decoder UNet 모델을 훈련시켰습니다.

- **Performance Highlights**: 훈련에 이미지 증강(image augmentation)을 적용함으로써 악천후 야간 조건에서의 세분화 성능이 크게 향상되었으며(p < 0.001), 그러나 날씨 데이터로 훈련된 모델은 맑은 날을 제외한 모든 조건에서 증강 데이터로 훈련된 모델보다 손실(loss)이 현저히 낮았습니다. 이는 도메인 적응(domain adaptation) 방식에 개선의 여지가 있음을 보여줍니다.



### Using Advanced LLMs to Enhance Smaller LLMs: An Interpretable Knowledge Distillation Approach (https://arxiv.org/abs/2408.07238)
- **What's New**: 이 논문은 더 작은 대형 언어 모델(LLM)의 성능을 향상시키기 위한 새로운 해석 가능한 지식 증류(knowledge distillation) 접근 방식을 제안합니다. 이 방법은 기업이 자가 호스팅할 수 있는 경제적인 모델의 성능을 높이도록 설계되었습니다.

- **Technical Details**: 전통적인 지식 증류 방식에서 '학생(student)' 모델이 '교사(teacher)' 모델의 응답을 통해 학습하는 대신, 우리의 방법은 '전략(strategy) 교육' 접근 방식을 활용합니다. 이 과정은 시나리오 생성(scenario generation) 단계와 개선 전략(strategies for improvement) 단계로 나뉘며, 학생 모델이 특정 상황에 적합한 응답을 생성하도록 돕습니다.

- **Performance Highlights**: 이 고객 서비스 응용 프로그램에서 이 방법은 성능을 향상시켰으며, 학습된 전략은 다른 LLM 및 훈련 집합을 넘어 다양한 시나리오에 적용할 수 있습니다. 실험 결과, 전략 교육이 다중 단계 생성에서 더 효과적임을 나타냈습니다.



### Direction of Arrival Correction through Speech Quality Feedback (https://arxiv.org/abs/2408.07234)
Comments:
          Submitted to Digital Signal Processing

- **What's New**: 이 논문에서는 Demucs Denoiser 모델의 방향 도착 추정(DOA) 정확도를 높이기 위한 새로운 DOA 교정 메커니즘을 제안합니다. 이 시스템은 실시간 음성 품질 측정을 활용하여 최적화 피드백 루프에서 DOA를 수정합니다.

- **Technical Details**: 제안하는 시스템은 실시간으로 음성을 개선하고, 최적화 메커니즘을 통해 음성 품질(Q) 측정값을 기반으로 복원된 DOA(θ_corr)를 사용하여 음성 개선 모듈에 피드백합니다. 이로 인해 최대 15도까지의 DOA 오류를 실시간으로 수정할 수 있습니다. 이러한 시스템은 성능을 향상시키기 위한 여러 통찰력을 제공합니다.

- **Performance Highlights**: Demucs Denoiser 모델은 20 dB 이상의 신호 대 간섭 비율(SIR)을 달성하며, 짧은 윈도우 세그먼트(0.064초)에서도 우수한 성능을 보여줍니다. 제안된 시스템은 향후 버전에서 수렴 속도를 높이고 음성 품질 추정의 변동성을 더욱 줄일 수 있는 방법을 논의합니다.



### Longitudinal Evaluation of Child Face Recognition and the Impact of Underlying Ag (https://arxiv.org/abs/2408.07225)
- **What's New**: 이번 연구는 신뢰할 수 있는 아동 얼굴 인식 기술의 필요성을 다루며, 8년 간의 데이터 수집을 통해 아동 얼굴 인식 시스템에 대한 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 Clarkson University의 CITeR 연구 그룹에 의해 수집된 YFA 데이터베이스를 활용하여 아동의 등록(enrollment) 및 검증(verification) 정확도를 장기간에 걸쳐 분석합니다. 데이터는 6개월 간격으로 수집되었습니다.

- **Performance Highlights**: 아동 얼굴 인식 기술은 다양한 신흥 응용 프로그램에서의 신뢰성을 높이고, 장기간에 걸친 신뢰할 수 있는 인식을 가능하게 합니다.



### Play Me Something Icy: Practical Challenges, Explainability and the Semantic Gap in Generative AI Music (https://arxiv.org/abs/2408.07224)
Comments:
          In Proceedings of Explainable AI for the Arts Workshop 2024 (XAIxArts 2024) arXiv:2406.14485

- **What's New**: 이 논문은 텍스트를 기반으로 한 오디오 및 음악 생성 도구의 특성을 비판적으로 분석하고 있으며, 설명 가능한 AI (explainable AI)의 문맥에서 이러한 도구의 잠재력을 탐구하고 있습니다.

- **Technical Details**: 연구자들과 실험적인 음악가들로 구성된 그룹이 이 도구들의 프롬프트 생성 (prompt creation), 제어 (control), 사용성 (usability), 이해 가능성 (understandability), 설명 가능성 (explainability), 그리고 전체 미적 효과성을 평가하기 위한 다양한 관점을 채택했습니다.

- **Performance Highlights**: 이 도구들이 다루지 않는 주요 도전 과제는 음악과 같은 추상적인 것을 설명하기 위해 텍스트 기반 도구를 사용할 때 발생하는 본질적인 의미 간극 (semantic gap)입니다. 또한, 이 도구의 사용성 vs. 설명 가능성, 사용자 제어 및 입력 vs. 인간의 창의적인 과정 간의 간극도 중요한 문제로 제기되었습니다.



### Handwritten Code Recognition for Pen-and-Paper CS Education (https://arxiv.org/abs/2408.07220)
- **What's New**: 본 논문에서는 손글씨로 작성된 코드 recognition을 위한 새로운 방법론을 제안합니다. 특히, Optical Character Recognition (OCR)과 Indentation Recognition 모듈을 결합하여 이전의 인식률 30%에서 5%로 개선하였습니다. 또한, 다중 모달 언어 모델을 활용한 end-to-end 방식의 손글씨 코드 인식 방법도 포함되어 있습니다.

- **Technical Details**: 기존에 존재하지 않았던 두 개의 공개 벤치마크 데이터세트를 제공하며, 각 접근법의 인식 정확도를 측정할 수 있는 방법론을 제안합니다. 손글씨 Python 코드의 들여쓰기 인식을 위한 새로운 방법론도 포함되어 있습니다. 이 연구는 OCR과 대형 언어 모델(LLM)의 통합을 통해 학생의 손글씨 코드에 대한 새로운 접근법을 제시합니다.

- **Performance Highlights**: 이 연구에서 제안된 방법은 구문 오류를 최소화하면서도 학생 코드의 정확한 디지털화에 기여하며, 평가 결과, 평균적인 Levenshtein 거리 감소를 통해 인식 알고리즘의 성능을 입증하였습니다. 학생들이 작성한 코드의 정확한 처리가 가능해져, CS 교육의 접근성을 높이는 데 기여할 것으로 기대됩니다.



### Hierarchical Multi-Armed Bandits for the Concurrent Intelligent Tutoring of Concepts and Problems of Varying Difficulty Levels (https://arxiv.org/abs/2408.07208)
Comments:
          Deployable RL: From Research to Practice @ Reinforcement Learning Conference 2024, 2024

- **What's New**: 본 논문에서는 오픈소스 기반의 계층적 MAB(다중무장군단) 알고리즘을 통해 원격 교육에서 문제 추천의 탐색-활용(trade-off) 문제를 해결하려는 연구 결과를 제시합니다. 이러한 지능형 튜터는 학생의 문제 추천에서 적절한 난이도를 결정하고, 동시에 학생들이 개념과 문제를 동시에 학습할 수 있도록 지원합니다.

- **Technical Details**: 이 알고리즘은 베이지안 지식 추적(Bayesian Knowledge Tracing, BKT)을 이용하여 500명의 학생을 대상으로 시뮬레이션하며, 각 학생의 학습 마스터리(content mastery)를 추정합니다. 학습 플랫폼은 비영리 조직인 Aiphabet에 구축되어 있으며, 이 알고리즘은 개념 선택을 위한 1차 MAB과 선택된 개념에 대한 문제 선택을 위한 2차 MAB으로 구성되어 있습니다.

- **Performance Highlights**: 연구 결과, 난이도에 무관한 상태에서 알고리즘의 성능이 학생들의 성공률을 획기적으로 향상시키며, 문제 난이도 적응을 추가했을 때 이러한 지표가 더욱 개선된 것으로 나타났습니다.



### Massive Dimensions Reduction and Hybridization with Meta-heuristics in Deep Learning (https://arxiv.org/abs/2408.07194)
Comments:
          8 pages, 5 figures, 3 tables, accepted at IEEE CCECE 2024 (updated Fig. 1 and conclusion remarks)

- **What's New**: 이 연구는 Histogram-based Blocking Differential Evolution (HBDE)라는 새로운 접근 방식을 소개하며, 이는 gradient-based 및 gradient-free 알고리즘을 혼합하여 매개변수를 최적화하는 데 중점을 둡니다.

- **Technical Details**: HBDE는 메타-휴리스틱 메타 분야에서 계층적 블록을 통해 높은 차원의 검색 공간에서 문제를 해결하는 접근 방식을 채택합니다. 실험 결과, 이번 연구에서 제안한 HBDE는 ResNet-18 모델의 매개변수를 11M에서 3K로 줄이고 CIFAR-10 및 CIFAR-100 데이터 세트에서 기존 알고리즘보다 우수한 성능을 보였습니다.

- **Performance Highlights**: HBDE는 gradient-free의 진화 최적화 기법을 활용하여 훈련 효과성을 높이고 메모리 소모를 줄이는데 성공했습니다. 전체 실험에서 관찰된 성능 향상은 높은 차원의 최적화 문제에 대한 대안 솔루션을 제시합니다.



### Solving Truly Massive Budgeted Monotonic POMDPs with Oracle-Guided Meta-Reinforcement Learning (https://arxiv.org/abs/2408.07192)
- **What's New**: 이 논문에서는 예산 제약이 있는 다중 구성 요소 단조 POMDPs를 효과적으로 해결하는 새로운 두 단계 접근 방식을 제안합니다. 각 구성 요소는 공유 예산에 의해 연결되기 때문에, 랜덤 포레스트 모델을 통해 예산 분배를 근사하고 후속적으로 메타 학습된 Proximal Policy Optimization (PPO) 알고리즘을 사용하여 독립적인 단일 구성 요소 POMDP를 해결합니다.

- **Technical Details**: 이 연구는 단조 POMDP의 최적 가치 함수를 랜덤 포레스트 모델을 통해 근사하고, 각 구성 요소의 예산 분배를 추정합니다. 그런 다음 오라클 정책을 기반으로 메타 학습된 PPO 알고리즘을 적용하여 각 구성 요소-예산 쌍의 정책을 계산합니다. 이 방법은 성능과 확장성을 제공하는 동시에 계산 복잡성을 줄입니다.

- **Performance Highlights**: 제안한 접근 방식은 1000개 인프라 구성 요소가 포함된 실제 유지 보수 시나리오에서 성능을 평가하며, 오라클 정책 및 현재 사용되는 기준 정책과 비교하여 구성 요소의 생존 시간을 기준으로 우수한 성능을 보였습니다. 알고리즘은 구성 요소 수에 대해 선형적인 복잡성을 나타내며, 대규모 문제에서도 유효성을 입증합니다.



### A New Dataset, Notation Software, and Representation for Computational Schenkerian Analysis (https://arxiv.org/abs/2408.07184)
- **What's New**: 이번 논문에서는 Schenkerian Analysis(SchA)를 활용한 음악 분석의 필요성을 강조하며, 이를 위한 새로운 데이터셋과 소프트웨어, 그리고 그래프 데이터 구조로서의 SchA 표현 방법을 제안합니다.

- **Technical Details**: 이 연구는 140개 이상의 SchA 발췌로 이루어진 새로운 데이터셋을 소개하며, SchA 데이터의 수집 및 시각화를 위한 소프트웨어도 제공합니다. 또한, SchA를 이질적 엣지 그래프(data structure)로 표현하는 새로운 방식을 탐구합니다. 이 데이터셋은 머신러닝 모델이 음악 구조에 대한 깊은 이해를 가질 수 있도록 도와줍니다.

- **Performance Highlights**: Schenkerian 분석이 음악의 멜로디-하모니 구조를 이해하는 데 광범위하게 사용될 수 있음을 보여줍니다. 연구에 따르면, SchA를 포함한 음악 정보 검색(MIR) 및 생성 작업에서 더 나은 결과를 도출할 수 있는 가능성이 큽니다.



### VulCatch: Enhancing Binary Vulnerability Detection through CodeT5 Decompilation and KAN Advanced Feature Extraction (https://arxiv.org/abs/2408.07181)
- **What's New**: VulCatch라는 새로운 바이너리 프로그램 취약점 탐지 프레임워크가 제안되었습니다.

- **Technical Details**: VulCatch는 Synergy Decompilation Module (SDM)과 Kolmogorov-Arnold Networks (KAN)을 도입하여 바이너리 코드를 pseudocode로 변환합니다. 이 과정에서 CodeT5를 사용하여 고급 의미를 유지하면서 Ghidra 및 IDA와 같은 도구로 딥 분석(DL analysis)을 수행합니다. KAN은 특징 변환(feature transformation)을 향상시키며, 복잡한 취약점 탐지를 가능하게 합니다.

- **Performance Highlights**: VulCatch는 98.88%의 높은 탐지 정확도와 97.92%의 정밀도를 달성하며, 1.56%의 잘못된 긍정(false positives)과 2.71%의 잘못된 부정(false negatives) 비율을 최소화합니다. 이를 통해 7개의 CVE 데이터셋에서 안정적인 성능을 보여줍니다.



### Vision Language Model for Interpretable and Fine-grained Detection of Safety Compliance in Diverse Workplaces (https://arxiv.org/abs/2408.07146)
Comments:
          20 pages, 7 figures

- **What's New**: 이번 연구에서는 안전 준수를 위한 Clip2Safety라는 새로운 감지 프레임워크를 제안합니다. 이 프레임워크는 네 가지 주요 모듈, 즉 장면 인식, 비주얼 프롬프트, 안전 물품 감지 및 세부 속성 검증으로 구성되어 있어 다양한 작업 환경에서 PPE를 효과적으로 인식하고 평가할 수 있도록 합니다.

- **Technical Details**: Clip2Safety의 프레임워크는 다음의 핵심 모듈로 구성됩니다: 1. 장면 인식 모듈은 현재 상황에 맞는 안전 장비를 식별합니다. 2. 비주얼 프롬프트 모듈은 감지 프로세스를 위한 특정 비주얼 프롬프트를 작성합니다. 3. 안전 물품 감지 모듈은 지정된 시나리오에 따라 안전 장비가 착용되고 있는지를 확인합니다. 4. 세부 속성 검증 모듈은 착용된 안전 장비가 필요한 속성 기준을 충족하는지 평가합니다.

- **Performance Highlights**: Clip2Safety는 최첨단 질문-응답 기반 VLM 모델보다 정확도 향상이 도모되었으며, 인퍼런스 시간은 기존 모델보다 200배 빨라졌습니다. 이 연구는 다양한 작업장의 안전 준수 데이터를 활용하여 성능을 평가하였습니다.



### A Theory-Based Explainable Deep Learning Architecture for Music Emotion (https://arxiv.org/abs/2408.07113)
- **What's New**: 본 연구에서는 음악에 대한 시간 변화하는 감정 반응을 예측하기 위한 이론 기반의 설명 가능한 심층 학습(Deep Learning) 합성곱 신경망(CNN) 분류기를 개발했습니다. 이 모델은 음악적 특징의 인식에 영향을 미치는 주파수 조화(harmonics)의 구조를 활용하는 새로운 CNN 필터를 설계했습니다.

- **Technical Details**: 우리의 CNN 모델은 두 가지 주요 특징으로 구분됩니다. 첫째, 음악 이론에 기반한 조화 필터를 포함하여 감정과 연관된 음악적 특징을 포착하도록 설계되었습니다. 감정은 러셀(Russell)의 발란스-각성(valence-arousal) 프레임워크를 사용하여 측정됩니다. 둘째, 이론 기반 필터는 감정에 영향을 미치는 잘 알려진 특징을 모델링하기 때문에 더 높은 설명 가능성을 제공합니다. 최종적으로, 우리는 이 모델을 YouTube 광고에 적용하여 감정 유사성에 따라 광고 삽입이 브랜드 회상률과 광고 참여를 증가시키는지를 실험했습니다.

- **Performance Highlights**: 우리의 모델은説明 가능성을 높은 상태에서 이론 기반의 감정 예측을 제공하며, 감정 유사성에 기반한 광고 삽입이 더 낮은 스킵 비율과 높은 브랜드 회상률을 가져온다는 결과를 보여주었습니다. 일반적인 비이론적 모델과 비교할 때, 본 모델은 비슷한 또는 더 나은 광고 참여도를 기록함으로써 마케팅에서의 활용 가능성을 시사합니다.



### Pattern-Matching Dynamic Memory Network for Dual-Mode Traffic Prediction (https://arxiv.org/abs/2408.07100)
- **What's New**: 새로운 Pattern-Matching Dynamic Memory Network (PM-DMNet) 모델을 제안하며, 이는 O(N) 복잡도를 사용하여 교통 예측을 가능하게 하고 계산 효율성을 크게 향상 시킨다.

- **Technical Details**: PM-DMNet는 Dynamic Memory Network (DMN)를 활용하여 교통 패턴 특징을 추출하며, Recursive Multi-step Prediction (RMP) 및 Parallel Multi-step Prediction (PMP) 방법을 통해 예측 과정을 지원한다. PMP에서는 Transition Attention Mechanism (TAM)을 적용하여 역사적 데이터와 예측 목표 간의 불일치를 줄인다.

- **Performance Highlights**: 실험 결과, PM-DMNet는 10개의 실제 데이터 세트에서 기존 최고 수준의 방법론보다 모든 데이터에 대해 현저히 우수한 성능을 보였다.



### Bearing Fault Diagnosis using Graph Sampling and Aggregation Network (https://arxiv.org/abs/2408.07099)
- **What's New**: 본 논문에서는 Bearing fault 진단 기술에 새로운 GraphSAGE 기반 알고리즘(GSABFD)을 제안하여 신호 간의 복잡한 상관관계를 고려합니다.

- **Technical Details**: 본 연구는 원본 진동 신호를 고정 크기 비중첩 슬라이딩 윈도우를 통해 슬라이스(slice)한 후, 신호 분석 방법을 활용하여 변환합니다. 그런 다음, 변환된 진동 신호의 상관관계를 구성하여 그래프의 정점(vertex)으로 변환하고, GraphSAGE 네트워크를 사용하여 훈련합니다. 마지막으로, 네트워크의 출력 레이어에서 객체의 고장 수준을 계산합니다.

- **Performance Highlights**: GSABFD 알고리즘은 실제 공개 데이터셋에서 다섯 개의 최신 알고리즘과 비교하여 AUC(Area Under Curve) 값을 5% 향상시켰습니다.



### QTypeMix: Enhancing Multi-Agent Cooperative Strategies through Heterogeneous and Homogeneous Value Decomposition (https://arxiv.org/abs/2408.07098)
Comments:
          16 pages, 8 figures

- **What's New**: 이번 논문은 다중 에이전트 협업 과제에서 이질적인 에이전트의 존재를 탐구하며, QTypeMix라는 새로운 방법을 제안합니다. QTypeMix는 가치 분해 과정(value decomposition process)을 동질적(homogeneous) 및 이질적(heterogeneous) 단계로 나누어 적용합니다.

- **Technical Details**: QTypeMix는 TE 손실(TE loss)을 통해 지역적 역사 관찰(local historical observations)로부터 유형 특징(type features)을 추출하는 학습 방법을 사용합니다. 또한, 주의 메커니즘(attention mechanisms)과 하이퍼넷(hypernets)을 포함한 고급 네트워크 구조를 도입하여 표현 능력(representation capability)을 향상시키고 가치 분해 과정을 달성합니다.

- **Performance Highlights**: SMAC 및 SMACv2의 14개 맵에서 제안된 방법을 테스트한 결과, QTypeMix는 다양한 난이도의 작업에서 최신 성능(state-of-the-art performance)을 달성했습니다.



### Attention Please: What Transformer Models Really Learn for Process Prediction (https://arxiv.org/abs/2408.07097)
- **What's New**: 본 논문은 트랜스포머(Transformer) 아키텍처 기반의 다음 활동 예측(next-activity prediction) 모델의 주의(attention) 점수가 의사결정 과정을 설명하는 데 얼마나 유용한지를 탐구합니다.

- **Technical Details**: 예측 비즈니스 프로세스 모니터링(PBPM)을 위한 다양한 예측 모델이 발전하였으며, 특히 트랜스포머 아키텍처가 주목받고 있습니다. 이 아키텍처는 주의 메커니즘을 활용하여 입력의 각 부분에 대해 주의 점수를 할당하여, 가장 관련성이 높은 정보를 우선 순위로 처리하게 됩니다. 논문은 두 가지 그래프 기반 설명 접근 방식(graph-based explanation approaches)을 제안하여 이 주의 점수의 설명 가능성을 탐구합니다.

- **Performance Highlights**: 여덟 개의 데이터 세트와 다섯 가지 보완 지표를 활용하여 설명 접근 방식의 품질을 평가하였으며, 이는 예측 비즈니스 프로세스 모델의 지속 가능한 비교 기준을 제공합니다.



### Post-Training Sparse Attention with Double Sparsity (https://arxiv.org/abs/2408.07092)
- **What's New**: 이번 논문에서는 "Double Sparsity"라는 새롭고 혁신적인 포스트 트레이닝( post-training ) 희소 주목 기법을 제안하여 Key-Value (KV) 캐시 접근을 줄이고 추론 시간을 단축시킵니다. 이 기법은 중요 토큰만을 사용하는 토큰 희소성(token sparsity)과 중요한 특징 채널을 활용하여 중요 토큰을 식별하는 채널 희소성(channel sparsity)을 결합하여 효과적입니다.

- **Technical Details**: Double Sparsity는 오프라인 보정을 통해 상대적으로 정적인 채널 희소성을 활용하여 런타임에서 중요 토큰을 동적으로 식별합니다. 이 기법은 Llama-2-7B, Llama-2-70B 및 Mixtral-8x7B 모델을 포함한 다양한 작업에서 높은 정확도를 유지하면서 1/16의 토큰 및 채널 희소성을 달성할 수 있습니다. 그리고 이를 통해 GPU의 메모리 사용량을 크게 줄이고 주목 연산을 최대 14.1배 가속할 수 있습니다.

- **Performance Highlights**: 실험 결과, Double Sparsity는 다양한 작업에서 정확성에 미치는 영향이 미미하면서 1.9배의 최종 추론 속도 향상과 함께 256K의 시퀀스 길이에서 기존 솔루션에 비해 16.3배 더 빠른 디코딩 속도를 달성했습니다. 이 기법은 메모리 접근을 줄이고 런타임 속도를 가속화하여 효율적인 추론을 가능하게 합니다.



### Node Level Graph Autoencoder: Unified Pretraining for Textual Graph Learning (https://arxiv.org/abs/2408.07091)
- **What's New**: 최근 연구에서 제안된 Node Level Graph AutoEncoder (NodeGAE)는 텍스트 그래프에 대한 새로운 비지도 학습 프레임워크로, 특성 매핑과 텍스트 복원을 통해 보다 고급의 특징 추출을 지향합니다.

- **Technical Details**: NodeGAE는 언어 모델을 기반으로 한 인코더-디코더 구조를 갖추고 있으며, 인코더는 노드의 숨겨진 임베딩을 독립적인 텍스트로 매핑합니다. 또한, InfoNCE 손실을 도입하여 이웃 간의 유사도를 극대화하고 구조 정보를 캡처합니다.

- **Performance Highlights**: NodeGAE는 ogbn-arxiv 데이터셋에서 테스트 정확도 77.10%를 달성하며, 기존의 SOTA 방법들과 유사한 성능을 보이며, GNN과의 앙상블을 통해 78.34%의 새로운 SOTA 정확도를 기록했습니다.



### InfinityMATH: A Scalable Instruction Tuning Dataset in Programmatic Mathematical Reasoning (https://arxiv.org/abs/2408.07089)
Comments:
          Accepted by CIKM 2024

- **What's New**: 최근의 Chain-of-Thoughts (CoT) 및 Program-of-Thoughts (PoT) 기법의 발전으로 언어 모델들의 수학적 추론 능력이 크게 향상되었습니다. 그러나 대규모 데이터셋 생성에 필요한 고유한 데이터와 높은 계산 비용은 스케일링에 많은 도전을 제공합니다. 이를 해결하기 위해 InfinityMATH라는 확장 가능한 지침 튜닝 데이터셋이 제안되었습니다.

- **Technical Details**: InfinityMATH는 수학적 문제에서 숫자를 분리하여 숫자에 독립적인 프로그램을 생성하는 파이프라인을 강조합니다. 이 접근 방식은 특정 숫자 값에 대한 의존성을 최소화하면서 효율적이고 유연한 스케일링을 가능하게 합니다. 이 데이터셋은 101,380개의 확장 가능한 데이터 포인트로 구성되어 있으며, Llama2 및 CodeLlama와 같은 오픈소스 언어 및 코드 모델로 조정되어 성능이 개선되었습니다.

- **Performance Highlights**: InfinityMATH로 조정된 모델은 in-domain 및 out-of-domain 벤치마크 모두에서 평균 184.7%에서 514.3%까지의 상대적인 개선을 보여주었습니다. 또한 GSM8K+ 및 MATH+ 벤치마크에서 높은 견고성을 발휘하여 다양한 수학적 문제에 대해 더 다재다능하고 효과적인 모델을 보장합니다.



### Learning Rule-Induced Subgraph Representations for Inductive Relation Prediction (https://arxiv.org/abs/2408.07088)
- **What's New**: 이번 논문은 지식 그래프에서 변동하는 링크의 예측을 위한 새로운 접근 방식을 제안합니다. 특히, 'single-source edge-wise GNN 모델'을 통해 관련 규칙을 인코딩하고 불필요한 규칙을 제거하는 방안을 모색하였습니다.

- **Technical Details**: 저자는 'Rule-inducEd Subgraph represenTations (REST)'이라는 모델을 제안하며, 단일 링크로부터 시작하는 초기화 방법을 통해 edge의 특징을 학습하도록 설계했습니다. RNN 기반 함수를 사용한 edge-wise 메시지 패싱이 시퀀스 속성을 모델링하는 데 도움을 줍니다. 이 모델은 복잡한 노드 라벨링을 필요로 하지 않아 서브그래프 전처리 시간을 획기적으로 단축시킵니다.

- **Performance Highlights**: 실험 결과는 REST 모델이 inductive relation prediction 벤치마크에서 우수한 성능을 보여주며, 서브그래프 전처리 시간을 11.66배까지 단축할 수 있음을 입증하였습니다.



### A Novel Spatiotemporal Coupling Graph Convolutional Network (https://arxiv.org/abs/2408.07087)
- **What's New**: 이 논문은 동적 Quality-of-Service (QoS) 추정에 대한 새로운 접근법을 제시합니다. Spatiotemporal Coupling GCN (SCG) 모델은 공간적(spatial) 및 시간적(temporal) 패턴을 통합적으로 모델링할 수 있는 방법을 모색합니다.

- **Technical Details**: SCG 모델은 일반화된 텐서 곱(tensor product) 프레임워크를 활용하여 동적 그래프 컨볼루션 규칙을 구축하며, 이질적인 GCN 레이어를 텐서 분해와 결합하여 사용자-서비스 그래프에서 효과적인 표현 학습을 수행합니다. 또한, 훈련의 난이도를 줄이기 위해 동적 GCN 구조를 단순화합니다.

- **Performance Highlights**: 대규모 QoS 데이터셋을 활용한 실험 결과, SCG 모델은 기존의 최신 기법들(state-of-the-arts)에 비해 높은 QoS 추정 정확도를 실현했습니다. 이는 사용자와 클라우드 서비스에 대한 강력한 표현력을 학습할 수 있음을 보여줍니다.



### Dynamic Hypergraph-Enhanced Prediction of Sequential Medical Visits (https://arxiv.org/abs/2408.07084)
- **What's New**: 이번 연구는 전자 건강 기록에서 미래의 의학적 진단을 더욱 정확하게 예측하기 위해 설계된 혁신적인 동적 하이퍼그래프 네트워크(Dynamic Hypergraph Networks, DHCE) 모델을 소개합니다.

- **Technical Details**: DHCE 모델은 환자의 방문 기록 내에서 급성(acute) 및 만성(chronic) 질병을 식별하고 구분하여, 질병 간의 복잡한 고차 상호작용을 포착하는 동적 하이퍼그래프(dynamic hypergraphs)를 구성합니다. 이 모델은 의료 언어 모델(Medical Language Model)을 통한 인코딩을 활용하여 임상 이벤트 데이터(clinical event data)를 효과적으로 통합함으로써 견고한 환자 표현을 생성합니다.

- **Performance Highlights**: MIMIC-III 및 MIMIC-IV라는 두 가지 벤치마크 데이터셋에서의 광범위한 실험을 통해 DHCE 모델은 기존의 기준 모델들과 비교하여 순차적 진단 예측의 정확도에서 상당히 우수한 성능을 보여줍니다.



### Masked EEG Modeling for Driving Intention Prediction (https://arxiv.org/abs/2408.07083)
- **What's New**: 새로운 EEG 기반 시스템이 드라이버의 의도를 예측할 수 있는 능력을 갖추고 있으며, 갑작스러운 움직임에 의해 유도된 아티팩트에 대한 견고성을 보여 줍니다. 이 연구는 BCI(Brain-Computer Interface) 지원 운전의 새로운 방향을 제시하고 있습니다.

- **Technical Details**: Masked EEG Modeling (MEM) 접근 방식이 제안되어, EEG 신호에서 운전 의도를 성공적으로 예측하기 위한 새로운 딥러닝 아키텍처를 사용합니다. EEG 신호의 독립 성분 분석과 전력 스펙트럼 밀도 분석을 통해 운전 조작 의도와 뇌의 신경 활동 간의 관계를 밝힙니다.

- **Performance Highlights**: 제안된 MEM 모델은 졸린 피험자의 운전 의도 예측에서 85.19%의 정확도를 달성하며, 채널이 손실되거나 손상되었을 경우에도 75% 이상의 정확도를 유지하여 실제 운전 상황에서의 적용 가능성을 보여줍니다.



### MathBridge: A Large-Scale Dataset for Translating Mathematical Expressions into Formula Images (https://arxiv.org/abs/2408.07081)
Comments:
          9page, 6 figures

- **What's New**: MathBridge라는 새로운 데이터셋이 소개되었으며, 이는 수학적 영어 발음을 LaTeX으로 변환하는 최초의 대규모 데이터셋입니다. 해당 데이터셋은 약 2300만 개의 LaTeX 공식과 해당 영어 표현 쌍으로 구성되어 있습니다.

- **Technical Details**: MathBridge는 영어 텍스트를 LaTeX으로 변환하여 수학 공식을 이미지로 나타내는 과정의 첫 번째 단계에서 중요한 역할을 합니다. 이 시스템은 텍스트에서 LaTeX으로, 그리고 LaTeX에서 이미지로의 변환을 포함합니다. 연구는 사전훈련된 언어 모델을 활용하여 MathBridge에서 추출한 데이터로 모델을 미세 조정했습니다.

- **Performance Highlights**: T5-large 모델을 기준으로 sacreBLEU 점수가 4.77에서 46.8로 상승하여 텍스트를 LaTeX으로 변환하는 능력이 크게 향상되었습니다. 이러한 결과는 수학 공식을 이해하는 데 있어 접근성을 개선하고, 교육 기술 향상에 기여할 수 있습니다.



### DisCoM-KD: Cross-Modal Knowledge Distillation via Disentanglement Representation and Adversarial Learning (https://arxiv.org/abs/2408.07080)
- **What's New**: 이번 연구에서는 전통적인 teacher/student 패러다임을 넘어 Cross-Modal Knowledge Distillation (CMKD)을 위한 새로운 프레임워크인 DisCoM-KD(Disentanglement-learning based Cross-Modal Knowledge Distillation)를 소개합니다. 이 프레임워크는 멀티모달 데이터에서 단일 모달 분류기로의 지식 전이를 위해 각 모달리티마다 다른 정보를 모델링합니다.

- **Technical Details**: DisCoM-KD는 분리(disentanglement) 표현 학습과 적대적 도메인 적응(adversarial domain adaptation)을 결합하여 각 모달리티에 대해 도메인 불변(domain-invariant), 도메인 정보(domain-informative), 도메인 무관(domain-irrelevant) 특성을 동시에 추출합니다. 이 방식을 통해 기존 teacher/student 모델을 개별적으로 학습시킬 필요가 없습니다.

- **Performance Highlights**: DisCoM-KD는 세 가지 표준 멀티모달 벤치마크에서 평가되었으며, 기존의 최첨단(Knowledge Distillation) 방법들과 비교했을 때, 서로 다른 모달리티가 겹치거나 겹치지 않는 상황에서 강력한 성능을 보여주었습니다. 이 결과는 멀티모달 데이터에서 단일 모달 신경망으로 정보를 추출하는 전통적인 패러다임을 재고할 수 있는 통찰력을 제공합니다.



### Anatomical Foundation Models for Brain MRIs (https://arxiv.org/abs/2408.07079)
Comments:
          12 pages

- **What's New**: 이 논문에서는 신경영상(neuroimaging)에서 신경계 질환과 신경퇴행성 질환을 탐지하기 위해 뇌 나이(brain age)를 주요 바이오마커로 사용하는 새로운 방법론인 AnatCL을 제안하고 있습니다. AnatCL은 해부학적 정보(anatomical information)와 약한 대조 학습(weakly contrastive learning) 접근 방식을 활용하여 다양한 임상 과제를 수행할 수 있는 강력한 기반 모델을 구축합니다.

- **Technical Details**: AnatCL 프레임워크는 환자의 나이와 동시에 세 가지 해부학적 특징(Mean Cortical Thickness, Gray Matter Volume, Surface Area)을 통합하여 대조 학습 손실 함수(loss function)를 수정했습니다. 이는 특정 지역(ROIs)에서 해부학적 정보를 계산하여 모델이 보다 유용한 표현 공간을 학습할 수 있도록 지원합니다. 또한, 로컬(local) 및 글로벌(global) 두 가지 버전의 프레임워크를 제안하여 특정 지역의 해부학적 변이가 모델링되도록 하였습니다.

- **Performance Highlights**: 자체 실험 결과, AnatCL은 여러 가지 다운스트림 임상 과제에서 기존의 기준 방법론보다 유사하거나 더 나은 성능을 기록했습니다. 특히, 신경영상 모델의 성능을 향상시키기 위해 타겟 해부학적 특징을 통합하는 것이 효과적이라는 것을 보여주었으며, 12개의 다양한 임상 과제와 10개의 임상 평가 점수 예측 작업에 걸쳐 광범위한 검증을 수행하였습니다.



### Decentralized Health Intelligence Network (DHIN) (https://arxiv.org/abs/2408.06240)
Comments:
          17 pages, 7 figures. arXiv admin note: substantial text overlap with arXiv:2407.02461

- **What's New**: 이 논문에서는 분산된 건강 정보 네트워크(Decentralized Health Intelligence Network, DHIN)라는 이론적 framework를 소개하고 있습니다. 이 framework는 의료 제공자 및 기관 간의 데이터 단편성(data fragmentation)으로 인해 발생하는 건강 데이터 주권(sovereignty) 및 AI 활용의 주요 도전 과제를 해결합니다.

- **Technical Details**: DHIN은 다음 세 가지 요소를 포함합니다: 1) 건강 데이터 주권을 위한 개인 건강 기록(Personal Health Record, PHR)과 결합된 자주적 신원(self-sovereign identity) 아키텍처; 2) 건강 데이터가 참가자와 함께 남아 있고 모델 파라미터 업데이트만 공유되는 분산형 AI 훈련을 위한 공공 블록체인(public blockchain)에서 구현된 확장 가능한 페더레이티드 러닝(federated learning, FL) 프로토콜; 3) 참여를 유도하고 공정한 보상 분배를 보장하는 신뢰할 필요 없는(rewardless) 보상 메커니즘.

- **Performance Highlights**: 이 framework는 참가자가 제공하는 건강 데이터에 대한 훈련 접근을 방해하거나 통제할 수 있는 주체가 없으며, 모든 과정은 변경 불가능한(public blockchain) 기록을 통해 이루어집니다. 환자는 그들의 디지털 지갑에 보상을 받아 FL 프로토콜에 참여하도록 유도되며, 궁극적으로 분산 보험 솔루션(funded decentralized insurance solutions)을 위한 길잡이 역할을 합니다.




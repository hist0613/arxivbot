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




New uploads on arXiv(cs.CL)

### PEDAL: Enhancing Greedy Decoding with Large Language Models using Diverse Exemplars (https://arxiv.org/abs/2408.08869)
- **What's New**: 본 논문에서는 PEDAL(Prompts based on Exemplar Diversity Aggregated using LLMs)이라는 하이브리드 자기 앙상블 기법을 소개하며, 다양한 예시 기반 프롬프트와 LLM 기반 집계를 결합하여 텍스트 생성에서의 성능 향상을 이끌어냅니다. 이를 통해 Greedy Decoding보다 더 나은 정확도와 낮은 추론 비용을 달성할 수 있음을 입증하였습니다.

- **Technical Details**: PEDAL 기법은 LLM을 사용하여 다양한 예시를 기반으로 하는 프롬프트를 통해 여러 후보 응답을 Greedy Decoding으로 생성한 후, 다시 LLM을 이용하여 이 응답들을 집계함으로써 최종 응답을 생성합니다. PEDAL은 데이터셋으로 SVAMP와 ARC를 사용하여 실험을 진행하였으며, 더 나은 성능을 보여주는 것을 확인했습니다. 특히, 기존 Self-Consistency(SC) 방법과 비교해 더 적은 출력 토큰을 필요로 하고 더 낮은 추론 비용을 발생시킵니다.

- **Performance Highlights**: PEDAL은 공개 데이터셋(SVAMP와 ARC)에서 Greedy Decoding 기반 전략보다 더 나은 정확도를 달성하였고, SC 기반 접근 방식에 비해 낮은 추론 비용을 제공함으로써 효과적인 성과를 보여주었습니다.



### PsychoLex: Unveiling the Psychological Mind of Large Language Models (https://arxiv.org/abs/2408.08848)
- **What's New**: 이 논문은 심리학(psychology)과 인공지능(artificial intelligence)의 교차점을 탐구합니다. 심리학적 작업을 위한 전문화된 대형 언어 모델(Large Language Models, LLMs)을 개발하고 평가하기 위해 PsychoLex라는 자원을 소개합니다.

- **Technical Details**: PsychoLex는 페르시아어(Persian)와 영어(English)에서 LLM의 심리학 작업 수행 능력을 향상시키기 위한 자료 모음으로 구성되어 있습니다. 주요 기여 사항으로는 PsychoLexQA 데이터셋(instructional content)과 PsychoLexEval 데이터셋(복잡한 심리 시나리오에서 LLM의 평가) 등이 있습니다. 또한, 심리학적 응용을 위해 특별히 최적화된 PsychoLexLLaMA 모델을 제시하고, 일반 목적 모델에 비해 우수한 성능을 입증하였습니다.

- **Performance Highlights**: 특수화된 LLMs의 잠재력이 심리학 연구 및 응용을 발전시키기 위한 가능성을 강조하며, 향후 인공지능 기반 심리학적 실습 발전을 위한 기초를 마련함을 보여주고 있습니다.



### FLEXTAF: Enhancing Table Reasoning with Flexible Tabular Formats (https://arxiv.org/abs/2408.08841)
- **What's New**: 이 논문은 테이블 추론(task)에서 고정된 형식의 테이블 대신 유연한(tabular) 형식을 사용하는 것이 성능을 개선할 수 있음을 보여줍니다. FLEXTAF-Single 및 FLEXTAF-Vote의 두 가지 방법을 제안하며, 각 방법은 특정 인스턴스(instance)와 LLM(대형 언어 모델)에 가장 적합한 테이블 형식을 선택하거나 결과를 통합하는 방식으로 성능을 향상시킵니다.

- **Technical Details**: FLEXTAF-Single은 인스턴스와 LLM에 기반하여 가장 적합한 테이블 형식을 예측하는 분류기(classifier)를 훈련합니다. FLEXTAF-Vote는 서로 다른 형식에서 얻은 결과를 통합하여 최종 답안을 결정합니다. 이 연구는 WikiTableQuestions 및 TabFact와 같은 테이블 추론 데이터셋에서 실험을 통해 두 방법의 효율성을 검증합니다.

- **Performance Highlights**: FLEXTAF-Single은 고정된 형식을 사용할 때보다 2.3%의 성능 개선을, FLEXTAF-Vote는 4.8%의 성능 개선을 보였습니다. 실험 결과, 특정 인스턴스는 특정 형식에서만 올바르게 해결될 수 있음을 나타내며, 이는 다양한 인스턴스와 모델에 따라 가장 적합한 테이블 형식이 다르다는 주장을 뒷받침합니다.



### CIKMar: A Dual-Encoder Approach to Prompt-Based Reranking in Educational Dialogue Systems (https://arxiv.org/abs/2408.08805)
Comments:
          This paper is the result of the final project of the Natural Language Processing course, Master of Artificial Intelligence, Universitas Gadjah Mada

- **What's New**: 이번 연구에서는 Gemma 언어 모델을 기반으로 한 CIKMar라는 교육 대화 시스템의 효율적인 접근 방식을 소개합니다. CIKMar는 BERT 및 SBERT 모델을 포함한 Dual-Encoder 순위 시스템을 활용하여 제공된 답변의 관련성과 정확성을 높였습니다.

- **Technical Details**: CIKMar는 BERTScore 메트릭을 사용하여 0.70의 강력한 recall과 F1-score를 달성하였으며, Gemma 1.1 2B 모델을 활용하여 12GB의 RAM과 단일 GPU T4에서 효율적으로 실행될 수 있도록 설계되었습니다. 이 시스템은 손으로 작성한 프롬프트를 사용하여 후보 출력을 재정렬하는 Dual-Encoder 전략을 채택했습니다.

- **Performance Highlights**: CIKMar는 교육 대화에서 뛰어난 응답의 관련성과 효과성을 보여주었으나, Dual-Encoder가 이론적인 응답을 실용적인 응답보다 우선시하는 경향이 있다는 중요한 도전 과제를 발견했습니다. 이는 Gemma와 같은 컴팩트하고 효율적인 모델의 가능성을 강조합니다.



### Leveraging FourierKAN Classification Head for Pre-Trained Transformer-based Text Classification (https://arxiv.org/abs/2408.08803)
- **What's New**: 이 연구에서는 전통적인 Multi-Layer Perceptron (MLP) 헤드 대신 FourierKAN (FR-KAN)이라는 새로운 분류 헤드를 도입했습니다. FR-KAN은 Kolmogorov-Arnold Networks (KAN)의 변형으로, Transformer 기반 인코더에서 사용됩니다.

- **Technical Details**: FR-KAN은 고차원의 맥락화된 임베딩을 처리하기 위해 푸리에 계수를 활용하여 고전적인 B-스플라인 대신 더 효율적이고 간단한 비선형 함수로 훈련됩니다. 연구 결과, 여러 텍스트 분류 작업에서 FR-KAN 헤드를 사용하면 MLP 헤드에 비해 훈련 파라미터가 감소하고 더 빠르게 훈련됨을 보여줍니다.

- **Performance Highlights**: 여러 텍스트 분류 데이터셋에서 FR-KAN 헤드를 사용했을 때 평균적으로 정확도가 10%, F1-score는 11% 향상되었습니다. 또한 FR-KAN 헤드는 MLP 헤드와 유사한 성능을 유지하면서도 더 적은 훈련 에포크 수를 요구합니다.



### EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics (https://arxiv.org/abs/2408.08782)
- **What's New**: 이번 논문에서는 감정적으로 지능적인 대화 시스템을 설계하기 위한 새로운 접근 방식인 EmoDynamiX를 제안합니다. 이 시스템은 언어 생성과 전략 예측을 분리하여 사용자 감정과 시스템 전략 간의 대화 역학을 모델링합니다.

- **Technical Details**: EmoDynamiX는 혼합 감정 모듈과 이질 그래프(homogeneous graph)를 사용하여 사용자 감정 상태를 세분화하고 대화 다이나믹스를 포착합니다. Emotion Recognition in Conversations (ERC) 작업을 활용하여 감정 분포 지식을 전이합니다.

- **Performance Highlights**: EmoDynamiX는 두 개의 ESC 데이터셋에서 기존 최첨단 방법에 비해 10% 이상의 개선을 달성하였고, 특히 자원이 부족한 AnnoMI 데이터셋에서는 30% 이상의 성능 향상을 보여 주목받고 있습니다.



### Large Language Models Might Not Care What You Are Saying: Prompt Format Beats Descriptions (https://arxiv.org/abs/2408.08780)
Comments:
          10 pages, 6 figures, 3 tables

- **What's New**: 이번 연구에서는 in-context learning (ICL) 중 설명적인 지시어의 기능에 대해 탐구하며, 이를 위해 여러 in-context 예시의 선택 기준을 설명하는 앙상블 프롬프트 프레임워크를 제안합니다. 초기 실험 결과, 이 프레임워크가 ICL 성능을 높이는데 기여하고 있음을 확인했습니다.

- **Technical Details**: 우리는 기계 번역(MT)에서 lexical(어휘적) 및 syntactic(구문적) 유사성을 기반으로 in-context 예시를 선택하고, 이를 조합하여 전체 예시 집합을 구성하는 방법을 고안했습니다. 추가적으로, 예시 수준의 설명을 포함하여 LLM에 유사한 단어나 유사한 구문을 설명했습니다. 실험 결과, 이러한 앙상블 프롬프트 프레임워크가 전통적인 프롬프트보다 LLM의 성능을 향상시킬 수 있음을 보였습니다. 또한, 예시 수준의 설명이 예시의 실제 출처와 일치하지 않거나 무의미할 경우에도 LLM의 성능이 향상되었습니다.

- **Performance Highlights**: 이 새로운 앙상블 프롬프트는 commonsense, 수학, 논리적 추론 및 환각 작업을 포함한 여러 과제에서 세 개의 소규모 LLM(Alpaca, Llama2, Mistral)과 하나의 대규모 LLM(GPT-3.5)에서 우수한 결과를 보였습니다. 특히, 설명적 명사가 무작위로 변경되어도 효과를 유지하는 것으로 나타났습니다.



### DAC: Decomposed Automation Correction for Text-to-SQL (https://arxiv.org/abs/2408.08779)
- **What's New**: 본 논문에서는 Text-to-SQL의 성능을 향상시키기 위해, 기존의 직접 수정 방식 대신 분해된 수정(decomposed correction) 방법을 제안합니다. 새로운 방법인 Decomposed Automation Correction (DAC)을 통해 SQL 쿼리의 수정 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: DAC는 텍스트-투-SQL 작업을 엔티티 링크(entity linking)와 스켈레톤 파싱(skeleton parsing)이라는 두 개의 하위 작업으로 분해하여 수행됩니다. 이 과정에서 DAC는 사용자의 질문에 해당하는 엔티티와 스켈레톤을 생성하고, 생성된 SQL과 비교하여 피드백을 제공하여 수정합니다. 이는 잘못된 SQL의 문제를 해결하는 데 있어 더 효과적이라고 입증됩니다.

- **Performance Highlights**: 실험 결과, DAC 방법은 Spider, Bird, KaggleDBQA의 기준선(baseline) 방법에 비해 평균 3.7%의 성능 향상을 보여주며, 이전의 텍스트-투-SQL 자동 수정 방법에 비해 1.3%의 향상을 달성했습니다. 이는 분해된 수정 방법이 효과적이라는 것을 증명합니다.



### Lower Layer Matters: Alleviating Hallucination via Multi-Layer Fusion Contrastive Decoding with Truthfulness Refocused (https://arxiv.org/abs/2408.08769)
Comments:
          9 pages, 4 figures, 5 tables

- **What's New**: 이 논문에서는 LLMs(대형 언어 모델)의 판타지 출력(즉, 사실과 다르거나 사용자 기대와 맞지 않는 결과) 문제를 해결하기 위한 새로운 접근법인 LOL(LOwer Layer Matters)라는 대비(decode) 프레임워크를 제안합니다. 기존의 방법들은 단일 층의 대비에 의존했으나, LOL은 최종 층과 저층의 대비를 결합하여 다중 계층 융합을 달성합니다. 또한, 사실 중심(factuality) 모듈을 추가하여 문맥적 지침을 사용해 사실성을 높이기 위한 개선을 이루고자 합니다.

- **Technical Details**: LOL 프레임워크는 최종 층의 대비 디코딩과 조기 종료(early exit) 저층의 디코딩을 병합하여 할로시네이션(hallucination)을 완화하려고 합니다. 또한, truthfulness refocused 모듈을 설계하여 문장 디코딩 과정에서 사실성(factuality)을 더욱 강조합니다. 실험에서는 두 개의 공개 데이터셋을 사용하였으며, LOL이 기존의 최선 기준선(model)보다 평균 4.5점 개선된 성능을 보였습니다.

- **Performance Highlights**: LOL 프레임워크는 대부분의 경우 기존 기준선보다 우수한 성능을 보였으며, TruthfulQA의 모든 지표에서 평균 4.5점 개선되었습니다. 이는 저층과 최종 층 간의 최적화를 통해 할로시네이션 문제를 효과적으로 감소시키는 것으로 나타났습니다.



### ChatZero:Zero-shot Cross-Lingual Dialogue Generation via Pseudo-Target Languag (https://arxiv.org/abs/2408.08724)
Comments:
          ECAI2024

- **What's New**: 이 논문에서는 ChatZero라는 새로운 엔드투엔드 제로샷(dialogue generation) 대화 생성 모델을 제안합니다. 이 모델은 크로스링구얼(code-switching) 방식을 기반으로 하여 다양한 언어에서 대화 생성을 가능하게 합니다.

- **Technical Details**: ChatZero는 소스 언어와 가상의 대상 언어를 코드 스위칭 언어로 결합한 후, 비지도식 대비 학습(unsupervised contrastive learning)을 활용하여 서로 다른 언어 간의 표현 갭을 최소화합니다. 가상의 대상 언어는 [MASK]와 같은 자리 표시자를 포함하여 소스 언어 단어를 대체하는 방식으로 구성됩니다.

- **Performance Highlights**: DAilyDialog 및 DSTC7-AVSD 데이터셋에서의 실험 결과, ChatZero는 제로샷 상황에서도 기존의 감독 학습(supervised learning)보다 90% 이상의 성능을 달성하며, 다른 기초 모델들에 비해 최첨단 성능을 기록하는 것으로 나타났습니다.



### Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling (https://arxiv.org/abs/2408.08696)
Comments:
          under review

- **What's New**: 본 논문에서는 Token Recycling이라는 새로운 기법을 제안합니다. 이 기법은 디코딩 과정에서 생성된 후보 토큰을 재사용하여 인퍼런스 속도를 크게 향상시키며, 추가적인 학습 없이도 적용할 수 있습니다.

- **Technical Details**: Token Recycling은 인접 행렬(adjacency matrix)에 후보 토큰을 저장하고, BFS(너비 우선 탐색)와 유사한 알고리즘을 사용해 드래프트 트리(draft tree)를 구성합니다. 이 트리는 트리 어텐션(tree attention)을 통해 검증됩니다. 이 방법은 2MB 미만의 추가 저장공간을 요구하고, 모든 LLM의 크기에서 약 2배의 속도 향상을 달성합니다.

- **Performance Highlights**: Token Recycling은 기존의 교육이 필요 없는(train-free) 방법들보다 30% 이상 성능이 우수하며, 교육 방법인 Medusa보다도 25% 이상 개선된 결과를 보여주었습니다. 이 기술은 어떤 기존 LLM과 작업에도 직접 적용할 수 있으며, SOTA(State Of The Art)를 달성했습니다.



### Quantifying the Effectiveness of Student Organization Activities using Natural Language Processing (https://arxiv.org/abs/2408.08694)
Comments:
          11 pages, 4 figures, presented in International Conference on Generative Al and its Applications (ICGAIA-24) last 22nd - 23rd, July, 2024 at Jakarta, Indonesia

- **What's New**: 학생의 과외 활동(Extracurricular Activities)이 학생의 교육 경험을 풍부하게 하는 데 중요한 역할을 한다는 점을 강조하고, 기계 학습(Machine Learning)과 자연어 처리(Natural Language Processing)를 활용하여 과외 활동의 효과를 정량화하는 연구를 제시합니다. 또한, BERT 모델을 사용하여 학생의 정서적 반응을 분석하는 새로운 워크플로우를 개발합니다.

- **Technical Details**: 이 연구에서는 Composed한 BERT(Large Language Model)를 활용하여, pysentimiento 툴킷을 통해 감정 분석(Sentiment Analysis)을 수행하는 기계 학습 파이프라인(ML Workflow)을 구축하였습니다. 데이터 전처리(Data Preprocessing), 주요 특징 선택(Key Feature Selection), LLM 기능 처리(Feature Processing), 점수 집계(Score Aggregation)의 단계로 이루어진 이 워크플로우는 각 데이터 세트에 대한 이벤트 점수(Event Score)를 산출합니다.

- **Performance Highlights**: BERT LLM은 제품 리뷰(Product Reviews) 및 댓글(Post Comments)을 넘어서 감정 분석에 효과적으로 활용될 수 있음을 보여줍니다. 이 연구는 교육 기관 학생 사무소(Student Affairs Offices)에 NLP를 실제 사례에 적용할 수 있는 실용적인 예시를 제공하며, 데이터 기반의 의사 결정(Data-Driven Decision Making)이 미칠 수 있는 잠재적 영향을 강조합니다.



### Med-PMC: Medical Personalized Multi-modal Consultation with a Proactive Ask-First-Observe-Next Paradigm (https://arxiv.org/abs/2408.08693)
Comments:
          26 pages, 5 figures

- **What's New**: 이 논문에서는 Multi-modal Large Language Models (MLLMs)의 임상 능력을 평가하기 위한 새로운 Medical Personalized Multi-modal Consultation (Med-PMC) 패러다임을 제안합니다.

- **Technical Details**: Med-PMC는 환자 시뮬레이터와 상호작용하여 다중 모달 정보 수집 및 의사 결정 작업을 수행해야 하는 최신 평가 프레임워크입니다. 이 프레임워크는 환자-행위자(agent) 시스템을 포함하여 보다 신뢰할 수 있는 임상 환경을 모델링합니다. MLLMs는 초기 환자의 정보를 바탕으로 다중 턴 의사 결정을 수행해야 합니다.

- **Performance Highlights**: MLLMs는 다중 모달 정보를 효과적으로 수집하지 못하고, 개인화된 환자 시뮬레이터와의 상담 시 의사 결정 작업에서 잠재적인 편향을 나타내었으며, Med-PMC의 효과를 통해 임상 MLLMs의 발전 방향을 제시하였습니다.



### The Fellowship of the LLMs: Multi-Agent Workflows for Synthetic Preference Optimization Dataset Generation (https://arxiv.org/abs/2408.08688)
- **What's New**: 이 논문은 다중 에이전트 워크플로우를 활용하여 합성 선호 최적화(Preference Optimization, PO) 데이터셋 생성을 평가합니다. PO 데이터셋 생성을 위해 두 개의 모듈(응답 평가 및 응답 생성)을 사용하며, LLM(대형 언어 모델)을 활용하여 인간의 주석 작업을 자동화합니다.

- **Technical Details**: 응답 평가 모듈에서는 LLM이 평가자로 사용되어 서로 다른 프롬프트 전략을 통해 우수한 응답 선택을 위해 경쟁합니다. 최종적으로 LLM를 사용한 Judge, Jury 및 Debate 방식 사이의 성능을 비교하고 Cohen의 카파(Kappa)를 통해 일치도 평가합니다. 응답 생성 모듈에서는 LLM 피드백 루프를 통한 다양한 구성의 성능을 비교합니다.

- **Performance Highlights**: GPT-4o-as-a-Judge는 GPT 계열의 응답이 포함되지 않은 데이터셋에서 더 높은 일관성을 보였으며, Llama를 생성기로, Gemma를 리뷰어로 사용하는 LLM 피드백 루프는 단일 에이전트인 Llama 및 Gemma에 비해 각각 71.8%와 73.8%의 높은 승률을 기록했습니다.



### MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector (https://arxiv.org/abs/2408.08661)
Comments:
          code and dataset: this https URL

- **What's New**: 본 논문에서는 MIA-Tuner라는 새로운 명령 기반의 구성 방안을 제안합니다. 이 방법은 LLM 스스로를 사전 학습 데이터 탐지기로 작동하게 하여 외부 MIA 점수 함수를 설계하는 대신 내부적으로 텍스트를 인식하도록 유도합니다.

- **Technical Details**: MIA-Tuner는 사용자로부터 제공된 텍스트가 LLM의 사전 학습 데이터셋에 포함되어 있는지 직접 응답하도록 유도하여 정확한 탐지를 가능하게 합니다. 또한, 기존 방법의 개인정보 보호 위험을 줄이기 위해 두 가지 instruction-based safeguard를 설계했습니다. 그리고 최신 MIA 벤치마크 dataset인 WIKIMIA-24를 구축하여 여러 가지 정렬된 및 비정렬된 LLM 전반에 걸쳐 실험을 수행했습니다.

- **Performance Highlights**: MIA-Tuner는 AUC(Area Under the Curve)를 0.7에서 0.9로 증가시켰으며, 기존 탐지 방법에 비해 53.4%와 26.5%의 성능 향상을 보였습니다. 이는 정렬 및 비정렬 LLM에 대해 검증되었습니다.



### LLMs Are Biased Towards Output Formats! Systematically Evaluating and Mitigating Output Format Bias of LLMs (https://arxiv.org/abs/2408.08656)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 성능 평가에 있어 형식 편향(format bias)을 체계적으로 분석한 첫 번째 연구입니다. 형식 제약사항을 따르면서도 성능을 측정하는 방법과 제약 조건에 관계없이 성능을 평가하는 방법을 구분하고, LLM의 형식 편향을 측정하기 위한 새로운 메트릭을 정의합니다.

- **Technical Details**: 연구에서는 15개의 다양한 형식(format)을 포함하는 4개의 범주 - 다중 선택 질문-답변(multiple-choice question-answer), 래핑(wrapping), 리스트(list), 매핑(mapping) -에 대한 형식 편향 평가를 실시했습니다. 감정 쿼리 탭을 통해 LLM의 형식 따르기(screen-follow) 능력을 높이는 방식으로 형식 편향을 완화할 수 있음을 발견하였습니다.

- **Performance Highlights**: ChatGPT의 래핑 형식에 대한 성능 편차를 235.33에서 0.71로 줄이는 데 성공하였으며, 이는 기존 형식 관련 문제의 개선된 해결책으로 보여집니다.



### Reasoning Beyond Bias: A Study on Counterfactual Prompting and Chain of Thought Reasoning (https://arxiv.org/abs/2408.08651)
- **What's New**: 이번 연구는 Massive Multi-Task Language Understanding (MMLU) 작업에서 언어 모델이 훈련 데이터에서 흡수한 편향을 조사하고, 이를 해결하기 위해 두 가지 새로운 방법인 Counterfactual Prompting with Chain of Thought (CoT) 및 Agnostically Primed CoT (APriCoT)를 제안합니다.

- **Technical Details**: 연구에서는 MMLU에서 언어 모델의 답변 선택 선호도가 학습된 통계적 규칙의 차이에 따라 어떻게 결정되는지를 규명했습니다. CoT와 CF(Counterfactual) 프롬프트를 결합하여 모델의 행동이 기준비율 확률(Base-rate Probability, BRP) 효과와 분리되어 동작하도록 하였으나, 알고 보니 BRP 효과가 증폭되는 현상이 관찰되었습니다. 이를 해결하기 위해 APriCoT를 제안하였습니다.

- **Performance Highlights**: APriCoT 방법이 언어 모델의 답변 품질과 공정성을 개선하는 데 기여함을 보였으며, 제안된 방법이 기존 방법들보다 더 나은 정확도를 나타냈습니다. 특히, 모델의 선호도가 근거 답변 분포와 거의 동일하게 분포되어 있으며, CF나 CoT보다 더 나은 정확성을 보여주었습니다.



### An End-to-End Model for Photo-Sharing Multi-modal Dialogue Generation (https://arxiv.org/abs/2408.08650)
Comments:
          Work in progress

- **What's New**: 첫 번째 end-to-end photo-sharing multi-modal dialogue 생성 모델을 제안한다. 이 모델은 이미지 인식 및 생성, 대화 생성 기능을 통합한다.

- **Technical Details**: 모델은 대형 언어 모델(LLM)과 이미지 인식을 위한 Q-Former, 이미지 생성을 위한 stable diffusion을 포함한다. 동적 어휘 변환 행렬을 설계하여 LLM와 diffusion 모델 간의 경량 변환을 가능하게 한다.

- **Performance Highlights**: End-to-end 모델은 기존 파이프라인 모델(Divter) 및 LLM 기반 파이프라인 모델과 비교하여 다양한 텍스트 및 이미지 생성 메트릭에서 최고의 성능을 보인다.



### Math-PUMA: Progressive Upward Multimodal Alignment to Enhance Mathematical Reasoning (https://arxiv.org/abs/2408.08640)
- **What's New**: 이 논문에서는 Multimodal Large Language Models (MLLMs)의 수학적 문제 해결 능력을 향상시키기 위해 Math-PUMA라는 새로운 방법론을 제안합니다. Math-PUMA는 Progressive Upward Multimodal Alignment를 기반으로 하여, 텍스트와 비주얼 정보를 동시에 처리하고 정렬하는 방식으로 진행됩니다.

- **Technical Details**: Math-PUMA 방법론은 세 가지 단계로 진행됩니다: 1단계에서는 LLM을 텍스트 기반 수학 문제로 훈련시키고, 2단계에서는 다양한 형태의 멀티모달 데이터 세트를 구성하여 비주얼과 텍스트 모달리티를 정렬합니다. 마지막 3단계에서는 고품질 멀티모달 문제 해결 데이터로 모델을 미세 조정합니다. 이 과정에서 Kullback-Leibler (KL) 발산을 활용하여 다음 토큰 예측 분포를 정렬합니다.

- **Performance Highlights**: 실험 결과, Math-PUMA로 훈련된 MLLMs는 대부분의 오픈 소스 MLLMs보다 우수한 성능을 보이며, 특히 다양한 모달리티에서 제공되는 문제들에 대한 성능 격차를 줄이는 데 효과적입니다.



### A Survey on Benchmarks of Multimodal Large Language Models (https://arxiv.org/abs/2408.08632)
- **What's New**: 이번 논문은 다중 모달 대규모 언어 모델(Multimodal Large Language Models, MLLMs)의 180개의 벤치마크 및 평가를 포괄적으로 검토하며, 이러한 모델들이 다양한 응용 분야에서 가진 성능을 다각도로 평가합니다. 특히, 인지 및 추론, 특정 도메인, 주요 기능과 다양한 모달리티에서의 능력에 중점을 두고 있습니다.

- **Technical Details**: MLLM은 비쥬얼 질문 응답(Visual Question Answering, VQA) 및 특정 도메인 과제를 포함한 다양한 응용 분야에서 뛰어난 성능을 보여줍니다. MLLM의 성능을 객관적으로 비교하고 탐색하기 위해, 영향을 미치는 다섯 가지 주요 분야를 통해 최근 논문을 분석합니다. 이러한 분야는 지각 및 이해, 인지 및 추론, 특정 도메인, 주요 기능 및 기타 모달리티로 구성됩니다. 또한, MLLMs의 동작을 지원하는 세 가지 주요 모듈인 비쥬얼 인코더, 언어 모델, 비쥬얼-언어 프로젝터에 대한 내용을 다룹니다.

- **Performance Highlights**: 최신 벤치마크에 따르면 OpenAI의 GPT-4와 Google의 Gemini는 83개의 평가 기준에서 베스트 쿼터를 기록했으며, 이러한 성과는 다중 모달 기능에서의 효율성을 높이는 데 기여하고 있습니다. 본 논문은 현재 MLLM 평가 방법의 한계를 지적하고, 향후 발전 방향에 대한 논의를 포함합니다.



### Persona is a Double-edged Sword: Enhancing the Zero-shot Reasoning by Ensembling the Role-playing and Neutral Prompts (https://arxiv.org/abs/2408.08631)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 논문에서는 LLM(대규모 언어 모델)의 사고 능력을 향상시키기 위한 새로운 프레임워크인 Jekyll & Hyde를 제안합니다. 이 프레임워크는 역할 놀이(role-playing) 그리고 중립적인(prompts) 프롬프트의 결과를 조합하여 성능 저하를 방지하고, LLM의 사고 능력을 강화합니다.

- **Technical Details**: Jekyll & Hyde는 두 가지 잠재적 솔루션을 수집하고 LLM 평가기를 통해 교차 검증을 실시하여 더 나은 솔루션을 선택합니다. 이러한 과정에서, 자가 생성된(persona) 페르소나를 활용하여 효율성 및 효용성이 증대됩니다. 또한, 달리점을 줄이는(robust position bias mitigation) 방법을 제안하여 순서 편향(position bias)을 완화합니다.

- **Performance Highlights**: 실험 결과, Jekyll & Hyde는 12개의 데이터셋에서 평균 9.98%의 정확성 향상을 보여주며, 역할 놀이 프롬프트가 LLM의 사고 능력을 방해하고 성능을 저하시킬 수 있다는 점을 규명했습니다. 또한, LLM 생성된 페르소나가 수작업으로 만든 페르소나보다 성능 안정성 측면에서 우수한 결과를 나타냈습니다.



### RealMedQA: A pilot biomedical question answering dataset containing realistic clinical questions (https://arxiv.org/abs/2408.08624)
Comments:
          Accepted at AMIA Annual Symposium 2024

- **What's New**: 이번 연구에서는 의료 전문가들의 실제 요구를 반영한 질문-답변 데이터셋인 RealMedQA를 소개합니다. 이는 인간과 LLM(대규모 언어 모델) 이 협력하여 생성한 질문으로 구성되어 있습니다.

- **Technical Details**: RealMedQA 데이터셋은 질문-답변 쌍 생성 및 검증 과정에서 LLM을 활용하며, BioASQ와 RealMedQA를 사용하여 여러 QA(질문-답변) 모델을 평가했습니다. 이 과정에서 질문과 답변의 어휘 유사성이 BioASQ에 비해 낮다는 특징이 있습니다.

- **Performance Highlights**: 우리는 LLM이 '이상적인' 질문-답변 쌍을 생성하는 데 있어 더 비용 효율적이라는 것을 보여주었으며, 상위 두 QA 모델에게 추가적인 도전을 제공합니다. 또한, 연구 결과는 코드와 데이터셋을 공개하여 향후 연구를 촉진할 것으로 기대됩니다.



### A Mechanistic Interpretation of Syllogistic Reasoning in Auto-Regressive Language Models (https://arxiv.org/abs/2408.08590)
- **What's New**: 최근 연구들은 auto-regressive Language Models (LMs)에서의 논리적 추론 능력이 체계적 추론 원칙을 학습하는 것이 아니라 훈련 데이터의 표면적인 패턴을 활용하는지를 다루고 있습니다. 이 논문은 LMs 내부의 역학을 이해하기 위해 삼단논법(syllogistic reasoning)에 대한 기계적 해석을 제시합니다.

- **Technical Details**: 이 논문에서는 회로 발견(circuit discovery) 방법론을 통해 내용 독립적(content-independent) 추론 메커니즘을 훈련 중 습득한 세계 지식(world knowledge)과 분리하려고 합니다. 두 가지介입 방법(intervention methods)을 통해 중간 항(middle-term) 억제를 포함한 회로를 발견하였으며, 이는 LMs가 전제에서 타당한 결론을 도출하는 정보를 어떻게 전이하는지를 밝힙니다.

- **Performance Highlights**: 이 연구의 결과는 LMs가 내용 독립적인 추론 메커니즘을 배운다는 것을 시사하지만, 동시에 이러한 메커니즘이 일반화 가능하고 추상적인 논리 원시(logical primitives)를 포함하지 않으며, 훈련 중 습득한 세계 지식의 오염에 취약하다는 것을 발견했습니다. 발견된 회로는 모든 삼단논법(syllogistic schemes)에 대해 충분하고 필요하며, 모델이 높은 하향 정확도(≥ 60%)를 달성할 수 있는 조건을 제공합니다.



### Overview of the BioLaySumm 2024 Shared Task on the Lay Summarization of Biomedical Research Articles (https://arxiv.org/abs/2408.08566)
Comments:
          Published in: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing

- **What's New**: 두 번째 BioLaySumm 공유 작업의 결과를 다루고 있으며, 참가 팀 수가 증가하고 자동화된 lay summarization 기법에 대한 연구가 더 활발하게 이루어지고 있습니다.

- **Technical Details**: 작업 참가자는 제공된 생물 의학 연구 기사의 텍스트를 기반으로 lay summary를 생성하는 시스템을 개발해야 합니다. CodaBench 플랫폼을 이용하여 두 개의 데이터셋(PLOS 및 eLife)을 사용하면서 경쟁을 진행하였고, 각 참가자는 각 벤치마크에 대한 예측된 lay summary를 제출하고 자동으로 평가받습니다.

- **Performance Highlights**: 이번 BioLaySumm 2024에는 총 53개 팀이 참가하여 200회 이상의 제출이 있었습니다. 이는 이전 대회인 BioLaySumm 2023에 비해 165%의 증가를 나타내며, 대회 참가 팀의 18개 팀이 Large Language Models (LLMs)를 활용하는 경향을 보였습니다.



### Integrating Multi-view Analysis: Multi-view Mixture-of-Expert for Textual Personality Detection (https://arxiv.org/abs/2408.08551)
Comments:
          Accepted by NLPCC 2024

- **What's New**: 본 논문에서는 사용자 게시물의 성격을 효과적으로 분석하기 위해 다양한 관점에서 다각적으로 접근한 Multi-view Mixture-of-Experts Model for Textual Personality Detection (MvP) 모델을 제안합니다. MvP는 다수의 전문가 네트워크를 통해 사용자 게시물을 자동으로 분석하고 사용자 간의 일관성을 유지하기 위한 정규화 기법을 도입하여 성격 특성 탐지의 정확도를 높이고자 합니다.

- **Technical Details**: MvP 모델은 Multi-view Mixture-of-Experts (MoE) 네트워크를 활용하여 사용자 게시물이 아닌 다양한 관점에서 통합된 사용자 표현을 형성합니다. 이 모델은 다중 과제 공동 학습 전략을 통해 감독된 성격 탐지와 자가 감독된 사용자 일관성 제약을 균형 있게 최적화하여 훈련됩니다.

- **Performance Highlights**: 실험 결과, MvP 모델은 두 개의 널리 사용되는 성격 탐지 데이터셋에서 효과성을 입증했습니다. 특히, 다양한 관점에서 사용자 게시물을 분석하는 것의 이점을 강조하며, MvP의 각 주요 모듈이 탐지 성능을 향상시키는 역할을 명확히 분석하였습니다.



### SelectLLM: Query-Aware Efficient Selection Algorithm for Large Language Models (https://arxiv.org/abs/2408.08545)
- **What's New**: 본 논문은 SelectLLM이라는 새로운 LLM 선택 알고리즘을 소개하며, 이 알고리즘은 다양한 LLM의 장점을 활용하여 복잡한 작업에서 발생하는 한계를 극복하는 데 중점을 둡니다. SelectLLM은 입력 쿼리를 가장 적합한 LLM 서브셋으로 유도하여 효율적으로 올바른 응답을 제공합니다.

- **Technical Details**: SelectLLM 알고리즘은 다중 레이블 분류기(multi-label classifier)를 사용하여 각 LLM의 예측 및 신뢰도 점수를 기반으로 최적의 LLM 서브셋을 선택하는 정책을 설계합니다. 이 접근 방식은 각 쿼리에 대해 최적의 작은 LLM 서브셋을 선택하여 응답 품질을 향상시키고 계산 비용을 줄입니다.

- **Performance Highlights**: 제안된 SelectLLM 모델은 개별 LLM보다 향상된 정확성을 보이며, 유사한 크기의 최상위 LLM 서브셋과 비교할 때 13% (GSM8K) 및 70% (MMLU) 낮은 레이턴시(latency)를 기록했습니다. 또한 두 개의 표준 추론 벤치마크에서 유의미한 성능 개선을 보여줍니다.



### Where is the signal in tokenization space? (https://arxiv.org/abs/2408.08541)
- **What's New**: 대형 언어 모델(LLMs)의 전통적인 텍스트 인코딩 방법과는 달리, 이 논문은 비정준(non-canonical) 토크나이징(tokenization) 방식의 가능성을 탐구합니다.

- **Technical Details**: 비정준 토크나이징은 문자열(token) 보다도 더 많은 조합이 가능하며, 이는 LLM의 확률 예측에서 중요한 요소가 됩니다. 저자들은 자동 회귀형 LLM을 위한 가장 가능성이 높은 토크나이징을 찾는 것이 계산적으로 어렵다고 주장하며, 모든 가능한 토크나이징에 대한 주변 확률(marginal probability)을 계산하는 것도 동일하게 어렵다고 설명합니다.

- **Performance Highlights**: 비정준 토크나이징의 확률을 단순히 집계함으로써, 여러 아키텍처(architecture)의 LLM 평가 벤치마크에서 성능 향상을 달성하였음을 입증했습니다.



### CommunityKG-RAG: Leveraging Community Structures in Knowledge Graphs for Advanced Retrieval-Augmented Generation in Fact-Checking (https://arxiv.org/abs/2408.08535)
- **What's New**: 본 논문은 Knowledge Graph (KG)와 Retrieval-Augmented Generation (RAG) 시스템을 통합하여 새로운 제로샷(zero-shot) 프레임워크인 CommunityKG-RAG를 제안합니다. 이를 통해 사실 검증(fact-checking) 과정의 정확성과 관련성을 향상시킬 수 있습니다.

- **Technical Details**: CommunityKG-RAG는 KG의 커뮤니티 구조를 활용하여 정보를 검색하는 방식으로 작동합니다. 이 프레임워크는 LLM과 RAG 시스템을 결합하여, KG에서 추출한 의사결정의 연관성을 강화합니다. Louvain 알고리즘을 통해 커뮤니티를 탐지하고, 각 노드에 대한 임베딩을 할당하여 사실 검증에 적합한 구조적 관련성을 보장합니다.

- **Performance Highlights**: 커뮤니티 KG와 RAG 시스템을 활용한 이 접근 방식은 기존의 방법들보다 더 높은 정확성과 효율성을 보여줍니다. CommunityKG-RAG는 추가적인 학습 없이도 다양한 도메인과 쿼리에 적응할 수 있는 강력하고 확장 가능한 솔루션입니다.



### Ex3: Automatic Novel Writing by Extracting, Excelsior and Expanding (https://arxiv.org/abs/2408.08506)
- **What's New**: 이번 논문에서는 인공지능을 이용한 장편 소설 생성의 어려움을 극복하기 위한 방법인 'Extracting Excelsior and Expanding (Ex3)'을 제안하였다. Ex3는 원시 소설 데이터로부터 구조 정보를 추출하고, 이를 바탕으로 LLM을 미세 조정하며, 최종적으로는 트리 구조의 확장 방법을 통해 자연스러운 소설 생성을 이끈다.

- **Technical Details**: Ex3는 원시 소설 데이터에서 구조 정보를 추출하는 'Extract', 이를 기반으로 LLM을 미세 조정하는 'Excelsior', 그리고 생성한 기초를 토대로 장편 소설을 생성하는 'Expand'의 세 단계로 구성되어 있다. 자아 발화(self-instructing) 방법을 활용하여 관련성을 기반으로 텍스트를 그룹화하고 요약하는 방식으로 단계적인 구조 정보 추출을 진행한다.

- **Performance Highlights**: Ex3를 통해 생성된 소설은 기존의 방법들에 비해 논리적 일관성과 매력도에서 우수한 성능을 보였으며, 장편 소설 생성의 질이 높아졌다.



### JPEG-LM: LLMs as Image Generators with Canonical Codec Representations (https://arxiv.org/abs/2408.08459)
- **What's New**: 최근 이미지 및 비디오 생성 분야에서 오토 리그레시브(autoregressive) LLM 아키텍처를 채택하여 다중 모달 시스템에 쉽게 통합할 수 있는 방법을 제시했습니다. 본 연구에서는 조작이 용이한 JPEG 및 AVC/H.264와 같은 공인된 코덱을 활용하여 이미지와 비디오를 생성하는 방법을 제안하고, 이를 통해 기존의 벡터 양자화(vector quantization) 방법보다 31% 더 나은 FID 결과를 얻었습니다.

- **Technical Details**: 본 연구는 Llama-2 아키텍처를 기반으로 하여 Jpeg-LM 및 Avc-LM이라는 두 개의 7B 모델을 프리트레인(pretrained) 합니다. 이들은 각각 256x256 크기의 이미지와 256x144 크기의 15프레임 비디오를 생성하며, 평균 컨텍스트 길이는 5K 및 15K입니다. 이미지를 JPEG 형식으로, 비디오를 AVC 형식으로 직접 압축 파일 바이트로 출력합니다.

- **Performance Highlights**: Jpeg-LM은 강력한 VQ 기반 모델보다 생성 품질이 우수하며 FID에서 평균 31%의 감소를 보였습니다. Avc-LM 또한 현실적인 움직임을 가진 비디오를 생성할 수 있음을 보여주었고, 비전 분야에서의 장기 요소에 대한 인식 능력이 특히 뛰어난 것으로 분석되었습니다.



### W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering (https://arxiv.org/abs/2408.08444)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)가 OpenQA(오픈 도메인 질문 응답) 작업에서 제한된 사실 정보를 생성하는 문제를 해결하기 위해 W-RAG라는 새로운 접근 방식을 제안합니다. W-RAG는 LLM의 랭킹 기능을 활용하여 약한 레이블 데이터를 생성하고, 이를 통해 dense retrievers(조밀 검색기)를 훈련합니다.

- **Technical Details**: W-RAG는 BM25를 통해 상위 K개의 구문을 검색하고, 각 구문이 정확한 답변을 생성할 확률에 따라 재랭크합니다. 그 후 가장 높은 순위를 가진 구문만을 긍정적인 훈련 샘플로 선택하고, 이를 바탕으로 dense retrievers를 훈련합니다. 이 과정은 question-passage(질문-구문) 쌍의 관련성을 평가하며, OpenQA를 위한 적합한 방법론으로 입증되었습니다.

- **Performance Highlights**: 저자들은 4개의 공개 OpenQA 데이터 세트에서 W-RAG의 성능을 평가하였으며, 결과적으로 이 방법이 기존 모델보다 retrieval(검색) 및 OpenQA 성능을 모두 향상시키는 것을 보여주었습니다.



### Rater Cohesion and Quality from a Vicarious Perspectiv (https://arxiv.org/abs/2408.08411)
- **What's New**: 인간의 피드백을 기반으로 한 AI 시스템에서의 불일치 해소를 위한 새로운 접근 방법으로서, 가상 주석(vicarious annotation)을 활용한 연구가 소개되었습니다. 이 방법은 주관적인 의견 제공 방식을 넘어서, 특정 집단의 입장을 대변하는 주석의 중요성을 강조합니다.

- **Technical Details**: 이 연구는 정치적 성향과 인구 통계적 배경이 주관적 피드백에 미치는 영향을 평가하기 위해 rater coherence metrics를 사용합니다. CrowdTruth의 rater quality metrics를 활용하여 주석의 질을 평가하고, 이들이 개인적 및 가상 주석 수준에서 rater 간의 응집력에 미치는 영향을 분석합니다.

- **Performance Highlights**: 이 방법은 정치적 담론과 같은 어려운 영역에서의 평가 신뢰성을 강화할 수 있는 잠재력을 지니고 있으며, 기존의 비약량적 주석 방식에서 벗어나 다양한 인구 통계적 배경을 고려하여 보다 신뢰할 수 있는 데이터를 생성할 수 있습니다.



### Zero-Shot Learning and Key Points Are All You Need for Automated Fact-Checking (https://arxiv.org/abs/2408.08400)
- **What's New**: 이 연구에서는 Zero-Shot Learning (ZSL)과 Key Points (KeP)를 기반으로 한 간단한 프레임워크를 소개하여 자동 사실 확인을 수행하였으며, AVeriTeC 공유 작업 데이터셋에서 10위로 높은 성과를 달성하였다.

- **Technical Details**: ZSL-KeP 프레임워크는 주어진 주장에 대해 키 포인트를 생성하고, BM25 알고리즘을 사용하여 관련 정보를 검색한 후, 증거를 생성하고 최종 판단을 내리는 일련의 단계를 포함한다. 이 과정에서 LLM을 활용하며, Zero-Shot Learning과 효과적인 프롬프트 전략을 통해 장기적인 맥락 이해가 가능하다.

- **Performance Highlights**: ZSL-KeP 모델은 AVeriTeC 데이터셋에서 기준 모델에 비해 Robustly한 성과 향상을 보여주며, 10위 달성이라는 결과를 기록하였다.



### Towards Realistic Synthetic User-Generated Content: A Scaffolding Approach to Generating Online Discussions (https://arxiv.org/abs/2408.08379)
- **What's New**: 본 논문에서는 사용자 생성 콘텐츠의 실용적인 대규모 합성 데이터셋 생성을 위한 프레임워크를 제안합니다. 특히, 소셜 미디어 토론 스레드를 실감나게 생성하기 위한 다단계 데이터 생성 프로세스를 도입하였습니다.

- **Technical Details**: 향후 연구에서는 Large Language Models (LLMs)를 활용하여 사용자 간 상대적인 토론 스레드를 생성합니다. 이 프로세스는 'scaffolds'라 불리는 압축된 표현을 통해 이루어지며, 다양한 플랫폼의 특성에 맞게 조정될 수 있습니다.

- **Performance Highlights**: Reddit와 Wikipedia Talk Pages를 포함한 두 개의 온라인 토론 플랫폼 데이터로 우리의 접근 방식을 검증하였으며, 합성 데이터와 실제 데이터의 비교를 위한 새로운 평가 측정 기준을 제안했습니다.



### Evaluating Text Classification Robustness to Part-of-Speech Adversarial Examples (https://arxiv.org/abs/2408.08374)
- **What's New**: 본 논문은 머신 러닝 시스템의 안전성을 높이기 위해 텍스트 기반 클래스 분류기에서 주요 단어 품사(part of speech)의 영향을 연구하고, CNN(class Convolutional Neural Network)의 약점을 드러내는 적대적 사례(adversarial examples)를 생성하는 새로운 방법론을 제시합니다.

- **Technical Details**: 저자들은 CNN 모델을 사용하여 리뷰 데이터셋에서 다양한 품사가 분류 결정에 미치는 영향을 실험적으로 조사했습니다. 이 논문에서는 적대적 신경망(adversarial neural network)을 제안하며, 이는 특정 품사를 조작하여 적대적 예제를 생성하는 데 활용됩니다. 연구는 Amazon, Yelp, IMDB의 세 가지 데이터셋을 사용하여 1%, 5%, 10% 및 15%의 단어를 제거하는 방법으로 CNN의 예측 신뢰도를 평가했습니다.

- **Performance Highlights**: 실험 결과, CNN 분류기는 동사(verbs), 명사(nouns) 및 형용사(adjectives)에 편향되어 있으며, 이러한 품사에서 소규모 패턴을 조작함으로써 효과적인 적대적 예제를 지속적으로 생성할 수 있음을 보여주었습니다.



### xGen-MM (BLIP-3): A Family of Open Large Multimodal Models (https://arxiv.org/abs/2408.08872)
- **What's New**: 이 보고서는 xGen-MM (BLIP-3으로도 알려짐)이라는 대규모 멀티모달 모델 개발 프레임워크를 소개합니다. 이 프레임워크는 정교하게 curated 된 데이터셋, 훈련 레시피, 모델 아키텍처 및 결과적으로 생성된 LMMs를 포함합니다. xGen-MM은 Salesforce의 xGen 이니셔티브를 확장하여 기초 AI 모델을 개발합니다.

- **Technical Details**: xGen-MM은 비전 토큰 샘플러(perceiver resampler)를 사용하여 Q-Former 아키텍처를 대체하고, 멀티모달 환경에서 텍스트 토큰의 오토 레그레시브 손실에 집중하여 훈련 목표를 단순화합니다. 프레임워크는 다양한 멀티모달 데이터 소스에서 오는 자율형 멀티모달 인터리브 텍스트와 비전 토큰을 입력으로 받아들입니다.

- **Performance Highlights**: 우리의 사전 훈련된 기본 모델은 강력한 인컨텍스트 학습 능력을 발휘하며, 지침 조정된 모델은 유사한 모델 크기를 가진 오픈 소스 LMMs와 비교하여 경쟁력 있는 성능을 보여줍니다. 우리는 DPO(Dynamic Policy Optimization)로 안전 조정된 모델을 간단히 소개하며, 이는 허위 정보를 완화하고 안전성을 높이고자 합니다.



### Evaluating the Evaluator: Measuring LLMs' Adherence to Task Evaluation Instructions (https://arxiv.org/abs/2408.08781)
- **What's New**: 최근에 LLMs-as-a-judge가 자동 평가에 활용되는 새로운 접근 방식으로 주목받고 있습니다. 이 방법은 사람의 평가를 대체하여 LLM을 통해 품질 판단을 수행하며, RLHF로 훈련된 GPT4와 Llama3와 같은 최신 LLM이 인식하는 품질 기준에 대해 탐구합니다.

- **Technical Details**: 본 논문에서는 질적 평가 기준의 새로운 분류 체계를 제안합니다. 이 체계는 4개의 평가 카테고리(내용, 관련성, 완전성, 참여도)로 구성되어 있으며, 이를 통해 8개의 최첨단 벤치마크 데이터셋에서 테스트한 34개의 메트릭을 포함합니다. 또한, 다양한 LLM 계열을 사용하여 LLMs-as-a-judge의 효과를 체계적으로 평가하였습니다.

- **Performance Highlights**: 퍼플렉시티(perplexity)가 주어진 지침 없이 인공지능의 인간 수준의 판단과 더 나은 상관관계를 가지는 경우가 많았으며, 평가의 중요 영역에서는 LLM을 통해 간단한 평가가 성공적으로 이루어질 수 있음을 보여줍니다. 결과적으로, 간단한 모델 퍼플렉시티가 LLMs-as-a-judge보다 우수한 평가 대안일 수 있음을 시사합니다.



### ConcateNet: Dialogue Separation Using Local And Global Feature Concatenation (https://arxiv.org/abs/2408.08729)
- **What's New**: 이 논문에서는 새로운 대화 분리 시스템 ConcateNet을 제안합니다. 이 시스템은 지역적(local) 및 전역적(global) 특징을 처리하는 획기적인 접근 방식을 기반으로 하여, 다양한 방송 신호에 대해 더 나은 일반화 성능을 목표로 합니다.

- **Technical Details**: ConcateNet은 DNN(Deep Neural Network) 구조에 기반하며, 혼합 신호에서 대화 신호를 효과적으로 추출하기 위해 지역적 및 전역적 특징을 분리합니다. 이 네트워크는 복잡 값 마스크를 사용하여 대화 신호를 추정하고, NLR(Nonlinear Refinement) 기능을 추가하여 더욱 정교한 신호 분리를 실현합니다.

- **Performance Highlights**: ConcateNet은 세 가지 데이터셋에서 평가했으며, 특히 방송 중심의 데이터셋에서 기존의 최첨단 노이즈 감소 방법들보다 우수한 성능을 보였습니다. 이로써 방송 관련 응용 프로그램을 위한 대화 신호의 분리에 대한 유망한 가능성을 제시하고 있습니다.



### LLM-PCGC: Large Language Model-based Point Cloud Geometry Compression (https://arxiv.org/abs/2408.08682)
- **What's New**: 이 연구는 대형 언어 모델(LLM)을 기반으로 한 점 구름 기하학 압축(LLM-PCGC) 방법을 제안하며, 이는 텍스트 설명 없이 점 구름 구조를 이해하고 압축하는 첫 번째 구조입니다.

- **Technical Details**: LLM-PCGC 방법은 클러스터링, K-tree 구조, 토큰 매핑 불변성(token mapping invariance), 그리고 저랭크 적응(LoRA) 기법을 활용하여 LLM을 점 구름의 압축기/생성기로 변환합니다. 입력된 3D 점 구름을 클러스터링한 후, 각 클러스터는 병렬 처리되며, 좌표 정규화, K-tree 조직, 트리 구조의 평탄화 및 청크화를 포함합니다.

- **Performance Highlights**: 실험 결과, LLM-PCGC는 MPEG 기초 점 구름 압축(G-PCC) 소프트웨어에 비해 -40.213% 비트 전송률 감소를 달성하며, 최첨단 학습 기반 방법에 비해 -2.267% 비트 전송률 감소를 기록하여 기존 방법을 월등히 초월하는 성능을 보였습니다.



### Understanding Enthymemes in Argument Maps: Bridging Argument Mining and Logic-based Argumentation (https://arxiv.org/abs/2408.08648)
Comments:
          Research note

- **What's New**: 이 논문은 논쟁 맵(argument map)과 그 안의 명제들 및 주장(claim) 사이의 관계를 추적하기 위한 새로운 방법론을 제안합니다. 기존의 논쟁 맵 분석 기술의 한계점을 극복하기 위해 고전 논리(Classical Logic)와 기본 논리(Default Logic)를 활용하여 텍스트의 명시적 정보와 암시적 정보를 형식적으로 표현하고자 합니다.

- **Technical Details**: 논문은 두 가지 주요 기술적 질문을 다룹니다. 첫 번째 질문은 논쟁 맵에서 명확한 정보(예: 전제(premise)와 주장)를 어떻게 고전 논리로 표현할 수 있는가입니다. 두 번째 질문은 암시적 정보(예: 결론을 도출하기 위해 필요한 배경 지식)를 어떻게 논리적으로 표현할 수 있는가입니다. 기본 논리를 사용하여 주장 사이의 지원(support) 및 공격(attack) 관계를 정의하고, 이를 통해 논쟁 맵의 각 노드를 논리적 주장으로 인스턴스화(instantiation)할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 접근 방식을 통해 사용자는 자동으로 생성된 논쟁 맵을 효과적으로 분석하고 비판할 수 있으며, 서로 다른 논쟁 맵을 비교할 수 있는 기반을 마련하게 됩니다. 이는 논쟁을 보다 논리적이고 체계적으로 접근함으로써, 인간의 인지 과정에서의 논리적 비약(nonevident jumps)을 줄이는 데 기여할 것입니다.



### Collaborative Cross-modal Fusion with Large Language Model for Recommendation (https://arxiv.org/abs/2408.08564)
Comments:
          10 pages, 4 figures, accepted by CIKM 2024

- **What's New**: 새로운 프레임워크인 CCF-LLM은 대규모 언어 모델(LLM)과 협업 신호를 결합하여 추천 품질을 향상시킵니다. 이는 사용자의 상호작용을 하이브리드 프롬프트로 변환하고, 두 가지 모달리티의 잠재 임베딩을 효과적으로 융합하는 방법을 제안합니다.

- **Technical Details**: CCF-LLM은 사용자의 아이템 상호작용을 하이브리드 프롬프트로 변환한 뒤 주의 깊은 교차 모달 융합 전략을 통해 두 모달리티 간의 정보 융합을 수행합니다. 이는 GATEGATE 네트워크를 사용하여 더욱 세밀한 차원별 융합 최적화를 가능하게 하며, 협업 신호와 의미적 지식을 통합적히 활용합니다.

- **Performance Highlights**: CCF-LLM은 기존의 추천 시스템들보다 더 효과적으로 의미적 신호와 협업 신호를 결합하여 우수한 추천 성능을 보여줍니다. 실험 결과는 CCF-LLM의 효율성을 입증하며, 협업 신호의 통합이 데이터셋에서 추천 결과를 극대화함을 확인하였습니다.



### MuRAR: A Simple and Effective Multimodal Retrieval and Answer Refinement Framework for Multimodal Question Answering (https://arxiv.org/abs/2408.08521)
Comments:
          Preprint

- **What's New**: 이 논문에서는 MuRAR(Multimodal Retrieval and Answer Refinement)이라는 간단하고 효과적인 프레임워크를 소개합니다. MuRAR는 텍스트 기반 답변을 향상시키고 관련된 멀티모달 데이터(영상, 이미지, 표 등)를 검색하여 일관된 멀티모달 답변을 생성하는 데 중점을 둡니다.

- **Technical Details**: MuRAR는 크게 세 가지 주요 구성 요소로 이루어져 있습니다: 1) Text Answer Generation: 사용자의 쿼리에 따라 관련 텍스트 문서 스니펫을 검색하고 LLM을 사용하여 초기 텍스트 답변을 생성합니다. 2) Source-based Multimodal Retrieval: 초기 텍스트 답변에 대한 멀티모달 데이터를 검색합니다. 3) Multimodal Answer Refinement: LLM에 프롬프트를 제공하여 검색된 멀티모달 데이터와 텍스트 답변 스니펫을 통합하여 최종 답변을 생성합니다.

- **Performance Highlights**: 인간 평가 결과, MuRAR에 의해 생성된 멀티모달 답변은 일반 텍스트 답변보다 유용하고 판독성이 더 높다는 것을 보여주었습니다. 이 프레임워크는 최소한의 수정으로 기업 수준의 AI 어시스턴트에 통합될 수 있으며, 멀티모달 데이터의 품질이 일반 텍스트 답변의 품질을 초월하는 것을 입증했습니다.



### Level Up Your Tutorials: VLMs for Game Tutorials Quality Assessmen (https://arxiv.org/abs/2408.08396)
Comments:
          Accepted at ECCV 2024 CV2 Workshop

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)을 활용하여 게임 튜토리얼의 품질을 자동으로 평가하는 혁신적인 솔루션을 제안합니다. 이 솔루션은 게임 튜토리얼의 프레임을 분석하고 인간의 인식을 시뮬레이션하기 위해 관련 질문에 자동으로 답변합니다. 이를 통해 개발자에게 즉각적인 피드백을 제공하고, 사용자 경험을 향상시킬 수 있습니다.

- **Technical Details**: 본 연구에서는 자동화된 게임 테스트 솔루션을 제안하며, VLMs를 활용하여 게임 튜토리얼의 프레임 분석 및 그에 대한 질문 대답을 통해 인간의 인식을 모방합니다. 이 과정은 개발자가 정의한 기대 결과와 비교하여 튜토리얼의 명확성과 이해도를 평가하는 방식으로 진행됩니다. 또한 다양한 오픈소스 및 폐쇄형 최신 모델들을 벤치마킹 하였습니다.

- **Performance Highlights**: 이 솔루션은 전통적인 테스트 방법보다 훨씬 더 빠르고 효율적인 평가 과정을 가능하게 하며, 문서화된 비디오와 주석이 달린 프레임을 개발자와 공유하여, 사용자 경험을 개선하는 데 기여하는 방법으로 개발 초기 단계의 시간을 단축시키는 데 효과적입니다.



### Plan with Code: Comparing approaches for robust NL to DSL generation (https://arxiv.org/abs/2408.08335)
Comments:
          9 pages, 1 figure, 5 tables. arXiv admin note: substantial text overlap with arXiv:2407.02742

- **What's New**: 이 논문은 RPA (Robotic Process Automation) 도메인에서 DSL (Domain Specific Languages) 생성을 위해 Retrieval Augmented Generation (RAG) 방법론을 최적화하여 LLM (Large Language Model)을 사용하는 방안을 제시합니다.

- **Technical Details**: NL2DSL (Natural Language to Domain Specific Language) 생성을 위한 새로운 시스템 아키텍처를 개발하고, 기존 코드 생성 방법의 환각 및 구문 오류 문제를 해결하기 위한 효율성을 평가합니다. Codex 모델을 기반으로 LoRA 기반의 미세 조정 방식을 통해 67,000개의 NL-DSL 샘플로 훈련했습니다. RAG 기술을 적용하여 DSL 생성을 위한 grounding 이슈를 해결했습니다.

- **Performance Highlights**: 최적화된 RAG 접근 방식이 특정 도메인 API 이름의 품질을 일치시키면서, 미세 조정된 모델보다 7포인트 향상된 유사도 지표를 기록했습니다. 이는 비도메인 또는 보지 않은 API 이름의 경우에도 더욱 두드러집니다.



### CodeMirage: Hallucinations in Code Generated by Large Language Models (https://arxiv.org/abs/2408.08333)
Comments:
          Accepted at AutoMates @ IJCAI 2024

- **What's New**: 이 논문은 LLM(대규모 언어 모델)에서 생성된 코드의 환각(hallucination) 현상을 최초로 연구한 것이며, 코드 환각의 정의와 포괄적인 세분류(taxonomy)를 제공한다.

- **Technical Details**: 코드 환각은 LLM이 생성하는 코드에서 구문적(syntactical) 및 논리적(logical) 오류, 보안 취약점(security vulnerabilities), 메모리 누수(memory leaks)와 같은 고급 문제를 포함한다. 코드 환각을 탐지하기 위해, 우리는 CodeMirage라는 벤치마크 데이터셋을 제안하고, OpenAI의 GPT-3.5 및 GPT-4와 오픈 소스 LLM인 CodeLLaMA를 통해 실험하였다. 실험을 통해 GPT-4가 HumanEval 데이터셋에서 가장 우수한 성과를 보였다.

- **Performance Highlights**: LLM의 코드 환각 탐지 작업의 성능에 대한 여러 기초선(baseline)을 도입했으며, LLMs가 다양한 환각 유형을 탐지하는 데 있어 합리적인 성과를 보여주었다. 특히 GPT-4는 HumanEval 데이터셋에서 최고의 성능을 보였고, MBPP 데이터셋에서도 Fine-tuned CodeBERT와 비슷한 결과를 도출하였다.



### Covert Bias: The Severity of Social Views' Unalignment in Language Models Towards Implicit and Explicit Opinion (https://arxiv.org/abs/2408.08212)
Comments:
          This work is under-review

- **What's New**: 본 연구는 대규모 언어 모델에서 암시적 편향(implied bias)과 명시적 편향(explicit bias) 간의 차이가 모델의 행동에 미치는 영향을 분석하였습니다. 특히, 극단적인 편향 시나리오에서 모델의 성능을 평가하기 위한 스트레스 테스트 평가 방식을 제시하였습니다.

- **Technical Details**: 이 연구에서는 미소지니(misogyny)와 종교적 편견(religious bigotry)과 관련된 두 가지 사회 집단에 대한 편향을 분석하기 위해 hate speech 및 stance detection의 두 가지 다운스트림 작업을 사용하였습니다. 적합한 데이터 세트를 통해 두 모델, Llama2-7b 및 Mistral-7b를 fine-tuning 하여 실험을 진행하였습니다.

- **Performance Highlights**: 편향이 있는 모델은 상대적으로 보편적인 명시적 의견에 대한 편향 경향을 보였고, 상충되는 견해와 관련된 응답을 생성할 때 더 신중한 응답을 나타냈습니다. 이는 주관적 주제에 대해 불확실성을 반영하는 표현을 통해 모델의 신뢰성을 향상시킬 필요성을 시사합니다.



New uploads on arXiv(cs.IR)

### EasyRec: Simple yet Effective Language Models for Recommendation (https://arxiv.org/abs/2408.08821)
- **What's New**: 이번 연구에서는 텍스트 기반 의미 이해를 협업 신호와 통합하는 효율적인 추천 시스템 EasyRec을 제안합니다. 이를 통해 추천 시스템의 일반화 능력을 향상시킵니다.

- **Technical Details**: EasyRec은 텍스트-행동 정렬 프레임워크(titled text-behavior alignment framework)를 기반으로 하며, 대조 학습(contrastive learning)과 협업 언어 모델 튜닝(collaborative language model tuning)을 결합하여 텍스트 보강된 의미 공간과 협업 행동 정보 간의 강력한 정렬을 보장합니다.

- **Performance Highlights**: EasyRec은 기존의 최첨단 모델들에 비해 뛰어난 성능을 보이며, 특히 텍스트 기반 제로샷(zero-shot) 추천 시나리오에서 그 우수성을 입증했습니다. 또한, 이 모델은 사용자 프로필을 동적으로 생성하여 시간의 변화에 따라 사용자 선호도에 잘 적응합니다.



### Multimodal Relational Triple Extraction with Query-based Entity Object Transformer (https://arxiv.org/abs/2408.08709)
Comments:
          15 pages, 7 figures, preprint

- **What's New**: 본 논문에서는 Multimodal Entity-Object Relational Triple Extraction이라는 새로운 작업(task)을 제안하여 이미지-텍스트 쌍에서 모든 트리플(entity span, relation, object region)을 추출하는 것을 목표로 합니다. 이를 통해 기존의 관계 추출 방법의 한계를 극복하고, 미리 정의된 엔티티와 객체에 의존하지 않는 데이터의 실용성을 높이고자 합니다.

- **Technical Details**: 제안된 QEOT (Query-based Entity-Object Transformer) 모델은 쿼리 기반의 선택적 주의 메커니즘을 적용하여 텍스트와 이미지 정보의 상호 작용 및 융합을 동적으로 탐색합니다. 이 모델은 이중 타워 아키텍처를 사용하여 텍스트와 이미지를 각각 인코딩하며, 멀티 태스크 학습 접근 방식을 통해 엔티티 추출, 관계 분류 및 객체 탐지를 동시에 수행합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 제안된 방법은 기존의 기준선 대비 8.06% 향상된 성능을 보여주었으며, 최신 성과를 달성했습니다. 이를 통해 제안된 작업이 유망한 연구 방향임을 입증하였습니다.



### SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for~Sequential Recommendation (https://arxiv.org/abs/2408.08686)
- **What's New**: 본 논문에서는 SC-Rec이라는 새로운 추천 시스템을 제안하며, 이는 다양한 아이템 인덱스와 프롬프트 템플릿으로부터 학습한 선호 지식을 결합하여 추천 결과의 일관성을 높입니다.

- **Technical Details**: SC-Rec은 세 가지 단계로 구성됩니다: (1) 아이템 다중 인덱스 생성, (2) 다중 인덱스 추천 모델 훈련, (3) 일관성 점수 기반 재정렬. 이를 통해 아이템 지식을 계층적 양자화 기법으로 생성하고, 다양한 프롬프트 템플릿을 활용하여 고유한 사용자 상호작용을 반영합니다.

- **Performance Highlights**: 실험 결과 SC-Rec은 세 가지 실제 데이터 세트에서 최신 기술들에 비해 상당한 성능 향상을 보였으며, 이질적인 인덱스와 다양한 프롬프트 템플릿에서 얻은 보완 지식을 효과적으로 통합하였습니다.



### OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction (https://arxiv.org/abs/2408.08585)
Comments:
          CIKM 2024

- **What's New**: 본 논문에서는 고객 생애 가치(Customer Lifetime Value, CLTV) 예측을 위한 새로운 최적 분포 선택 모델인 OptDist를 제안합니다. OptDist는 적응형 최적 서브 분포 선택 메커니즘을 이용하여 복잡한 CLTV 분포 모델링의 정확도를 향상시키기 위한 것입니다.

- **Technical Details**: OptDist는 분포 학습 모듈(Distribution Learning Module, DLM) 내에서 여러 후보 서브 분포 네트워크를 훈련합니다. 그런 다음 분포 선택 모듈(Distribution Selection Module, DSM)을 통해 각 샘플에 대해 적절한 서브 분포를 자동으로 선택합니다. Gumbel-Softmax 연산을 사용하여 훈련 시 샘플마다 하나의 최적 서브 분포를 선택합니다. 이 과정에서 DLM과 DSM 간의 정렬 메커니즘을 도입하여 최적화를 유도합니다.

- **Performance Highlights**: OptDist는 두 개의 공개 데이터셋과 하나의 개인 산업 데이터셋에서 기존 선진 모델들보다 우수한 성능을 발휘하며, 대규모 금융 플랫폼에서 고객 확보 마케팅 캠페인에 적용되어 효과성을 입증했습니다.



### Collaborative Cross-modal Fusion with Large Language Model for Recommendation (https://arxiv.org/abs/2408.08564)
Comments:
          10 pages, 4 figures, accepted by CIKM 2024

- **What's New**: 새로운 프레임워크인 CCF-LLM은 대규모 언어 모델(LLM)과 협업 신호를 결합하여 추천 품질을 향상시킵니다. 이는 사용자의 상호작용을 하이브리드 프롬프트로 변환하고, 두 가지 모달리티의 잠재 임베딩을 효과적으로 융합하는 방법을 제안합니다.

- **Technical Details**: CCF-LLM은 사용자의 아이템 상호작용을 하이브리드 프롬프트로 변환한 뒤 주의 깊은 교차 모달 융합 전략을 통해 두 모달리티 간의 정보 융합을 수행합니다. 이는 GATEGATE 네트워크를 사용하여 더욱 세밀한 차원별 융합 최적화를 가능하게 하며, 협업 신호와 의미적 지식을 통합적히 활용합니다.

- **Performance Highlights**: CCF-LLM은 기존의 추천 시스템들보다 더 효과적으로 의미적 신호와 협업 신호를 결합하여 우수한 추천 성능을 보여줍니다. 실험 결과는 CCF-LLM의 효율성을 입증하며, 협업 신호의 통합이 데이터셋에서 추천 결과를 극대화함을 확인하였습니다.



### Don't Click the Bait: Title Debiasing News Recommendation via Cross-Field Contrastive Learning (https://arxiv.org/abs/2408.08538)
- **What's New**: 이번 연구에서는 뉴스 추천 시스템을 위한 새로운 접근 방법인 Title Debiasing News Recommendation with Cross-field Contrastive learning (TDNR-C2)를 제안합니다. 특히, 뉴스 제목의 클릭 유도 현상을 극복하기 위해 뉴스 초록을 활용하는 방식으로, 사용자가 더욱 정확한 콘텐츠에 접근할 수 있도록 합니다.

- **Technical Details**: TDNR-C2 방법은 다중 분야 지식 추출 모듈을 통해 다양한 분야에서 뉴스에 대한 다중 관점을 추출하고, 교차 분야 대조 학습 모듈을 통해 제목과 초록 간의 학습된 지식을 대조하여 편향을 제거합니다. 이를 통해 뉴스의 진정성과 일치하는 내용을 추천하도록 설계되었습니다.

- **Performance Highlights**: 실제 데이터셋에서 실험 결과, TDNR-C2는 기존의 최첨단 방법들에 비해 우수한 성능을 나타내며, 특히 뉴스 초록의 중요성이 제목 편향 제거에서 뚜렷하게 나타났습니다.



### MuRAR: A Simple and Effective Multimodal Retrieval and Answer Refinement Framework for Multimodal Question Answering (https://arxiv.org/abs/2408.08521)
Comments:
          Preprint

- **What's New**: 이 논문에서는 MuRAR(Multimodal Retrieval and Answer Refinement)이라는 간단하고 효과적인 프레임워크를 소개합니다. MuRAR는 텍스트 기반 답변을 향상시키고 관련된 멀티모달 데이터(영상, 이미지, 표 등)를 검색하여 일관된 멀티모달 답변을 생성하는 데 중점을 둡니다.

- **Technical Details**: MuRAR는 크게 세 가지 주요 구성 요소로 이루어져 있습니다: 1) Text Answer Generation: 사용자의 쿼리에 따라 관련 텍스트 문서 스니펫을 검색하고 LLM을 사용하여 초기 텍스트 답변을 생성합니다. 2) Source-based Multimodal Retrieval: 초기 텍스트 답변에 대한 멀티모달 데이터를 검색합니다. 3) Multimodal Answer Refinement: LLM에 프롬프트를 제공하여 검색된 멀티모달 데이터와 텍스트 답변 스니펫을 통합하여 최종 답변을 생성합니다.

- **Performance Highlights**: 인간 평가 결과, MuRAR에 의해 생성된 멀티모달 답변은 일반 텍스트 답변보다 유용하고 판독성이 더 높다는 것을 보여주었습니다. 이 프레임워크는 최소한의 수정으로 기업 수준의 AI 어시스턴트에 통합될 수 있으며, 멀티모달 데이터의 품질이 일반 텍스트 답변의 품질을 초월하는 것을 입증했습니다.



### Beyond KAN: Introducing KarSein for Adaptive High-Order Feature Interaction Modeling in CTR Prediction (https://arxiv.org/abs/2408.08713)
Comments:
          KarSein for CTR

- **What's New**: 이번 논문에서는 CTR(Click-Through Rate) 예측을 위해 새로운 모델인 Kolmogorov-Arnold Represented Sparse Efficient Interaction Network(KarSein)를 소개합니다. KarSein은 고차원 feature interaction을 효율적으로 모델링하는 데 중점을 두고, 기존의 Kolmogorov-Arnold Networks(KAN)의 한계를 극복합니다.

- **Technical Details**: KarSein은 KAN의 여러 활성화 기능을 단일 feature에 하나의 활성화 기능만 할당하여 계산을 최적화합니다. 또한, 2D embedding vectors를 지원하여 feature 상호작용을 벡터 방식으로 모델링할 수 있게 합니다. KarSein은 각 레이어에서 feature의 쌍을 곱하는 추가 단계를 포함해 멀티플리케이티브(multiplicative) 관계를 학습할 수 있도록 설계되었습니다. 또한, 이 모델은 KAN의 단순화 기술을 유지하여 global explainability를 지원합니다.

- **Performance Highlights**: KarSein은 세 가지 데이터 세트에서 state-of-the-art 성능을 보여주며, 최소한의 계산 비용으로 높은 예측 정확성을 달성하였습니다. 이를 통해 KarSein은 sparse network 구조의 장점을 유지하면서 불필요한 feature를 제거하여 효율적인 추론을 가능하게 합니다.



### W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering (https://arxiv.org/abs/2408.08444)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)가 OpenQA(오픈 도메인 질문 응답) 작업에서 제한된 사실 정보를 생성하는 문제를 해결하기 위해 W-RAG라는 새로운 접근 방식을 제안합니다. W-RAG는 LLM의 랭킹 기능을 활용하여 약한 레이블 데이터를 생성하고, 이를 통해 dense retrievers(조밀 검색기)를 훈련합니다.

- **Technical Details**: W-RAG는 BM25를 통해 상위 K개의 구문을 검색하고, 각 구문이 정확한 답변을 생성할 확률에 따라 재랭크합니다. 그 후 가장 높은 순위를 가진 구문만을 긍정적인 훈련 샘플로 선택하고, 이를 바탕으로 dense retrievers를 훈련합니다. 이 과정은 question-passage(질문-구문) 쌍의 관련성을 평가하며, OpenQA를 위한 적합한 방법론으로 입증되었습니다.

- **Performance Highlights**: 저자들은 4개의 공개 OpenQA 데이터 세트에서 W-RAG의 성능을 평가하였으며, 결과적으로 이 방법이 기존 모델보다 retrieval(검색) 및 OpenQA 성능을 모두 향상시키는 것을 보여주었습니다.



### Towards Realistic Synthetic User-Generated Content: A Scaffolding Approach to Generating Online Discussions (https://arxiv.org/abs/2408.08379)
- **What's New**: 본 논문에서는 사용자 생성 콘텐츠의 실용적인 대규모 합성 데이터셋 생성을 위한 프레임워크를 제안합니다. 특히, 소셜 미디어 토론 스레드를 실감나게 생성하기 위한 다단계 데이터 생성 프로세스를 도입하였습니다.

- **Technical Details**: 향후 연구에서는 Large Language Models (LLMs)를 활용하여 사용자 간 상대적인 토론 스레드를 생성합니다. 이 프로세스는 'scaffolds'라 불리는 압축된 표현을 통해 이루어지며, 다양한 플랫폼의 특성에 맞게 조정될 수 있습니다.

- **Performance Highlights**: Reddit와 Wikipedia Talk Pages를 포함한 두 개의 온라인 토론 플랫폼 데이터로 우리의 접근 방식을 검증하였으며, 합성 데이터와 실제 데이터의 비교를 위한 새로운 평가 측정 기준을 제안했습니다.



New uploads on arXiv(cs.CV)

### xGen-MM (BLIP-3): A Family of Open Large Multimodal Models (https://arxiv.org/abs/2408.08872)
- **What's New**: 이 보고서는 xGen-MM (BLIP-3으로도 알려짐)이라는 대규모 멀티모달 모델 개발 프레임워크를 소개합니다. 이 프레임워크는 정교하게 curated 된 데이터셋, 훈련 레시피, 모델 아키텍처 및 결과적으로 생성된 LMMs를 포함합니다. xGen-MM은 Salesforce의 xGen 이니셔티브를 확장하여 기초 AI 모델을 개발합니다.

- **Technical Details**: xGen-MM은 비전 토큰 샘플러(perceiver resampler)를 사용하여 Q-Former 아키텍처를 대체하고, 멀티모달 환경에서 텍스트 토큰의 오토 레그레시브 손실에 집중하여 훈련 목표를 단순화합니다. 프레임워크는 다양한 멀티모달 데이터 소스에서 오는 자율형 멀티모달 인터리브 텍스트와 비전 토큰을 입력으로 받아들입니다.

- **Performance Highlights**: 우리의 사전 훈련된 기본 모델은 강력한 인컨텍스트 학습 능력을 발휘하며, 지침 조정된 모델은 유사한 모델 크기를 가진 오픈 소스 LMMs와 비교하여 경쟁력 있는 성능을 보여줍니다. 우리는 DPO(Dynamic Policy Optimization)로 안전 조정된 모델을 간단히 소개하며, 이는 허위 정보를 완화하고 안전성을 높이고자 합니다.



### SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and Medical Image Segmentation (https://arxiv.org/abs/2408.08870)
Comments:
          Technical Report

- **What's New**: 최근 비전 기초 모델(Vision Foundation Models)의 발전에 힘입어, Segment Anything Model 2 (SAM2)를 활용한 효과적인 이미지 세분화 프레임워크인 SAM2-UNet을 제안합니다. 이 모델은 전통적인 U-Net 구조에 SAM2의 Hiera 백본을 결합하여 뛰어난 성능을 발휘합니다.

- **Technical Details**: SAM2-UNet은 Hiera 백본을 사용하는 인코더와 전통적인 U-shaped 디자인의 디코더로 구성됩니다. 이 구조는 multi-scale feature capturing을 가능하게 하여 세밀한 세분화 작업을 지원합니다. 아울러, Adapters를 삽입하여 파라미터 효율적인 미세 조정(parametr-efficient fine-tuning)을 가능하게 합니다.

- **Performance Highlights**: SAM2-UNet은 여덟 개의 공개 데이터셋에서 실험을 통해 다섯 가지 어려운 벤치마크에서 기존의 특화된 최첨단 방법들을 간단히 초월하는 성능을 보여주었습니다. 각종 다운스트림 작업에서도 뛰어난 성능을 기록하며, 다양한 시나리오에서의 응용 가능성을 강조합니다.



### DPA: Dual Prototypes Alignment for Unsupervised Adaptation of Vision-Language Models (https://arxiv.org/abs/2408.08855)
- **What's New**: DPA는 VLM(vision-language models)의 비지도 도메인 적응(new unsupervised domain adaptation) 방법을 제안하며, CLIP과 같은 모델을 새로운 도메인에 적응시킬 수 있게 해줍니다.

- **Technical Details**: DPA는 이미지 및 텍스트 프로토타입을 사용하여 정확한 pseudo-labels를 생성하는 두 가지 프로토타입 시스템을 도입했습니다. 이 시스템은 각 프로토타입의 출력을 convex combination으로 융합하고 초기 학습 중 pseudo-labels를 정리하여 노이즈를 줄이는 방식을 활용합니다. 또한, DPA는 이미지 및 텍스트 프로토타입 간의 정렬을 통해 성능을 개선하고 있습니다.

- **Performance Highlights**: DPA는 13개의 다운스트림 비전 작업에서 zero-shot CLIP 및 최신 비지도 도메인 적응 방법론들과 비교할 때 일관되게 상당한 성능 향상을 보여줍니다.



### RGBT Tracking via All-layer Multimodal Interactions with Progressive Fusion Mamba (https://arxiv.org/abs/2408.08827)
- **What's New**: 이 논문은 All-layer 멀티모달 Interaction Network(AINet)를 제안하여 RGBT 트래킹에서 모든 레이어 간의 효율적이고 효과적인 피처 상호작용을 수행합니다. AINet는 다양한 상호작용 모델을 통합하여 모든 모달리티와 레이어의 향상된 특성을 활용하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: AINet는 Difference-based Fusion Mamba(DFM) 및 Order-dynamic Fusion Mamba(OFM)라는 두 가지 모듈을 설계합니다. DFM은 모달리티 간의 차이를 모델링하여 각각의 피처를 증강하고, OFM은 동적으로 레이어의 스캔 순서를 조정하여 모든 레이어에서의 효율적 상호작용을 지원합니다. 이 네트워크는 3840개의 토큰 시퀀스를 처리하여 대량의 데이터를 효과적으로 관리할 수 있습니다.

- **Performance Highlights**: AINet는 네 개의 공개 RGBT 트래킹 데이터셋에 대한 광범위한 실험에서 기존 최첨단 방법에 비해 뛰어난 성능과 효율성을 입증하였습니다. AINet는 저렴한 수의 파라미터와 낮은 계산 비용으로 새로운 최첨단 결과를 달성하였습니다.



### PFDiff: Training-free Acceleration of Diffusion Models through the Gradient Guidance of Past and Futur (https://arxiv.org/abs/2408.08822)
- **What's New**: 이번 연구에서는 PFDiff를 제안합니다. PFDiff는 기존의 빠른 ODE 솔버가 적은 NFE(수치 함수 평가)로 작동하도록 하는 새로운 트레이닝이 필요 없는(Training-Free) timestep-skipping 전략입니다. 이 접근법은 최근 ODE 솔버의 출력에서 큰 시간 단계 크기에서의 유사성을 기반으로 합니다.

- **Technical Details**: PFDiff는 과거 시간 단계에서의 그래디언트를 사용하여 현재의 그래디언트를 근사하는 방식을 채택하여 불필요한 노이즈 네트워크 평가를 줄입니다. 또한, Nesterov momentum에서 영감을 받아 미래 그래디언트를 통해 현재 상태를 빠르게 업데이트하여 1차 ODE 솔버의 불연속화 오류를 보정합니다.

- **Performance Highlights**: PFDiff는 다양한 조건부 DPMs에서 특히 우수한 성능을 발휘하며, DDIM을 기준으로 할 때 ImageNet 64x64에서 4 NFE에 대해 16.46 FID를 달성한 반면, 이전 방법은 138.81 FID를 기록했습니다.



### Retrieval-augmented Few-shot Medical Image Segmentation with Foundation Models (https://arxiv.org/abs/2408.08813)
- **What's New**: 이 논문에서는 DINOv2와 Segment Anything Model 2 (SAM 2)를 활용하여 새로운 retrieval-augmented few-shot 의료 이미지 세분화 방법을 제안합니다. 이 방법은 제한된 주석 데이터를 사용하여 유사한 샘플을 검색하고 이 정보를 메모리에 저장하여 세분화 정확성을 향상시키는 기법입니다.

- **Technical Details**: 제안하는 프레임워크는 DINOv2의 특징을 쿼리로 사용하여 유사한 샘플을 검색하고, 이들 이미지를 SAM 2의 메모리 주의 메커니즘을 통해 세분화 과제에 활용합니다. 특별히, 이 방법은 교육이나 파인튜닝 없이도 다른 도메인에서 쉽게 적용할 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: ACDC, CMR T1-Map, Fluoroscopy 이미지 데이터셋을 포함한 세 가지 의료 이미지 세분화 작업에서 수행된 평가에서, 제안한 프레임워크는 뛰어난 성능을 보였으며, 다양한 모달리티에서 일반화 능력을 발휘했습니다. 또한 retraining이나 fine-tuning이 필요하지 않다는 점에서 현실적인 임상 도구로서의 가능성을 보여줍니다.



### PriorMapNet: Enhancing Online Vectorized HD Map Construction with Priors (https://arxiv.org/abs/2408.08802)
- **What's New**: 최근 자율주행 기술 발전에 따라 온라인 벡터화된 고화질(HD) 지도 구축이 중요해졌습니다. 본 논문에서는 PriorMapNet이라는 새로운 접근 방식을 통해 지도 구축의 정확성을 높이고 매칭의 안정성을 개선했습니다.

- **Technical Details**: PriorMapNet은 PPS-Decoder와 PF-Encoder를 도입하여 지도 요소의 위치 및 구조적 정보를 반영한 참조점을 제공하고, DMD 교차 주의(attention)를 통해 다중 스케일 및 샘플에 걸쳐 효율성을 높입니다. 이 접근 방식은 지도 구축의 학습 난이도를 낮추고 안정적인 매칭을 가능하게 합니다.

- **Performance Highlights**: PriorMapNet은 nuScenes와 Argoverse2 데이터셋에서 온라인 벡터화된 HD 지도 구축 작업에서 최신 기술 수준의 성능(SOTA)을 달성했으며, 다양한 설정에서의 실험을 통해 견고성과 일반화 능력을 입증했습니다.



### Backward-Compatible Aligned Representations via an Orthogonal Transformation Layer (https://arxiv.org/abs/2408.08793)
Comments:
          Accepted at BEW2024 Workshop at ECCV2024

- **What's New**: 이 논문은 Visual retrieval (시각 검색) 시스템에서 구식 모델과 새로운 모델 간의 호환성을 유지하면서 성능을 향상시키는 방법을 제안합니다. 기존의 모델 업데이트 과정에서 발생하는 비효율성을 줄이기 위해, Feature space (특징 공간)을 확장하고 Orthogonal transformation (직교 변환)을 학습하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 제안된 Orthogonal Compatible Aligned (OCA) 접근 방식은 기존의 모델과 호환성을 유지하면서도 새로운 정보를 통합할 수 있게 해줍니다. 이 방법은 기존의 Feature space의 기하학을 보존하고, 여러 모델 업데이트 간에 특징을 직접 비교할 수 있도록 합니다. 실험 결과는 CIFAR-100과 ImageNet-1k에서 실시되었으며, 이 방법의 유효성을 확인하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 모델과의 호환성을 유지하면서도 여러 기존 방법보다 우수한 성능을 보여주며, state-of-the-art accuracy (최첨단 정확도)를 달성하였습니다.



### VF-NeRF: Learning Neural Vector Fields for Indoor Scene Reconstruction (https://arxiv.org/abs/2408.08766)
Comments:
          15 pages

- **What's New**: 이 논문은 Neural Radiance Fields (NeRF)를 기반으로 한 실내 밀집 표면 복원 기법을 제안합니다. 특히, 기존 NeRF 방식은 질감이 약한 평면 영역에서 어려움을 겪고 있는데, 이를 해결하기 위해 Vector Field (VF) 방식을 통해 표면을 표현합니다.

- **Technical Details**: VF는 가장 가까운 표면 점을 향하는 단위 벡터로 정의되며, 평면 표면에서는 방향이 일정하게 유지되어 강력한 귀납적 편향을 제공합니다. 이 방식은 고품질 표면 밀도를 계산하기 위해 VF의 예측을 기반으로 하는 새로운 VF-밀도 관계를 포함합니다. 또한, 다중 시점 이미지에서 VF를 학습하기 위한 훈련 절차를 제공합니다.

- **Performance Highlights**: VF-NeRF는 Replica 및 ScanNet과 같은 실내 데이터셋에서 기존 최첨단 방법들과 비교하여 뛰어난 성능을 보여주었습니다. 특히, 새로운 관점에서의 렌더링에서 탁월한 결과를 달성하여 실내 장면 복원에서의 성능을 크게 향상시켰습니다.



### PCP-MAE: Learning to Predict Centers for Point Masked Autoencoders (https://arxiv.org/abs/2408.08753)
- **What's New**: 본 논문에서는 Point Masked AutoEncoders (PCP-MAE)라는 새로운 방법론을 제안하여, 마스킹된 패치의 센터를 직접 제공하는 대신 모델이 이 센터를 예측하도록 학습시키는 효율적인 접근법을 소개합니다.

- **Technical Details**: PCP-MAE는 Predicting Center Module (PCM)을 통해 마스킹된 패치의 중심 좌표(centers)를 예측합니다. 이는 기존 인코더와 파라미터를 공유하며 크로스 어텐션(cross-attention) 메커니즘을 이용해 이러한 예측을 행합니다. 이를 통해 인코더는 가시 패치(visible patches)와 마스킹된 패치(masked patches) 간의 상호 관계를 학습하게 됩니다.

- **Performance Highlights**: PCP-MAE는 Point-MAE에 비해 각각 5.50%, 6.03%, 5.17% 향상된 성능을 보여주며, 다른 MAE 기반 방법들에 비해 높은 프리트레이닝(pre-training) 효율성을 달성합니다.



### Comparative Analysis of Generative Models: Enhancing Image Synthesis with VAEs, GANs, and Stable Diffusion (https://arxiv.org/abs/2408.08751)
- **What's New**: 이 논문은 Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), 및 Stable Diffusion 모델과 같은 세 가지 주요 생성적 모델링 프레임워크를 비교하고, Grounding DINO 및 Grounded SAM을 Stable Diffusion과 결합하여 이미지 정확도를 향상시키는 방법을 탐구합니다.

- **Technical Details**: VAEs는 잠재 표현(latent representations)을 학습하는 데 효과적하지만, 흐릿한 결과를 생성하는 경향이 있습니다. GANs는 현실적인 이미지를 생성할 수 있으나, 모드 붕괴(mode collapse) 문제에 직면하고 있습니다. Stable Diffusion 모델은 고품질 이미지를 생성하지만, 계산 리소스를 많이 소모합니다. Grounding DINO 및 Grounded SAM을 통해 Stable Diffusion과 결합하면 정밀한 분할(segmentation) 및 인페인팅(inpainting) 기법을 활용하여 이미지의 정확성을 높일 수 있습니다.

- **Performance Highlights**: Stable Diffusion 모델은 고해상도 및 의미론적 일관성을 갖춘 이미지를 생성하는 데 있어 매우 높은 성능을 보여줍니다. 이러한 모델은 복잡한 데이터 분포를 다루는 데 필요한 다양한 태스크에 효과적으로 적용할 수 있어, 향후 연구 및 응용 가능성도 넓힙니다.



### Task-Aware Dynamic Transformer for Efficient Arbitrary-Scale Image Super-Resolution (https://arxiv.org/abs/2408.08736)
Comments:
          ECAI 2024

- **What's New**: 이번 연구에서는 Task-Aware Dynamic Transformer (TADT)를 제안하여 이미지 재구성에서 가변적인 입력 이미지와 스케일에 대한 적응형 특성 추출기를 구현합니다. 이 모델은 입력 이미지와 업샘플링 스케일에 따라 동적인 경량화 구축을 사용하여 슈퍼 해상도를 수행합니다.

- **Technical Details**: TADT는 다중 스케일 특징 추출 백본과 Task-Aware Routing Controller (TARC)로 구성됩니다. TARC는 입력 이미지 및 SR 스케일에 따라 다이나믹한 추론 경로를 예측하고, 이를 통해 필요한 경우 가벼운 계산을 가능하게 하며, 최종 루팅 벡터를 Bernoulli 샘플링과 Straight-Through Estimator(스터리트 쓰루 추정기)를 결합하여 생성합니다.

- **Performance Highlights**: 실험 결과, TADT는 세 가지 인기 있는 가변 스케일 업샘플러인 MetaSR, LIIF, LTE에 대해 기존의 특성 추출기보다 더 나은 성능을 보이며, 적은 파라미터 수와 계산 비용으로 더욱 효율적인 이미지를 생성합니다.



### Correspondence-Guided SfM-Free 3D Gaussian Splatting for NVS (https://arxiv.org/abs/2408.08723)
Comments:
          arXiv admin note: text overlap with arXiv:2312.07504 by other authors

- **What's New**: 본 연구에서는 3D Gaussian Splatting을 활용한 SfM-free Novel View Synthesis (NVS) 방법을 제안합니다. 이는 상대적 카메라 포즈 최적화를 보다 효과적으로 수행할 수 있는 방식으로, 피사체와 렌더링 결과 간의 대응 관계를 이용하여 픽셀 정렬을 개선합니다.

- **Technical Details**: 제안된 방법은 2D 대응 검출을 통해 렌더링 이미지와 타겟 이미지 간의 픽셀 매칭을 찾아내고, 픽셀 매칭을 기반으로 하는 새로운 손실 함수를 설계하여 최적화를 수행합니다. 2D 스크린 공간에서의 3D Gaussian을 위한 근사 표면 렌더링 파이프라인을 개발하여, 3D Gaussian의 매개변수에 대한 그래디언트 역전파를 용이하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 방법들에 비해 우수한 성능과 시간 효율성을 보여주었습니다. 특히, 픽셀 부정렬 문제를 최소화하여 최적화의 안정성을 높였습니다.



### Decoupling Feature Representations of Ego and Other Modalities for Incomplete Multi-modal Brain Tumor Segmentation (https://arxiv.org/abs/2408.08708)
Comments:
          8 pages, 4 figures

- **What's New**: 본 논문에서는 DeMoSeg라는 새로운 방법론을 제안하여 다중 모드 뇌 종양 분할의 성능을 향상시킨다고 설명하고 있습니다. 이 방법은 자가 표현(Self-expression)과 상호 표현(Mutual-expression)의 과제를 분리하여 모드 적응(modality adaptation)을 더욱 쉽게 만듭니다.

- **Technical Details**: DeMoSeg는 각 모드를 네 개의 기능 서브 공간의 자기(feature)와 상호(Mutual-features) 표현으로 나누는 방법을 채택합니다. 첫 번째 서브 공간은 자기 기능을 표현하며, 나머지 서브 공간들은 다른 모드를 대신합니다. 또한, Channel-wised Sparse Self-Attention (CSSA)라는 새로운 주의 레이어로 상호 가이드를 허용하며, Radiologist-mimic Cross-modality expression Relationships (RCR)를 통해 결측된 모드를 보완하는 방법을 제안합니다.

- **Performance Highlights**: DeMoSeg는 BraTS2020, BraTS2018 및 BraTS2015 데이터셋에서 여러 성능 지표를 통해 기존 첨단 기술보다 Dice가 각각 0.92%, 2.95% 및 4.95% 향상된 결과를 도출했습니다. 이는 모드 적응 과정에서의 어려움을 완화하여 더 나은 성능을 발휘함을 보여줍니다.



### Beyond the Hype: A dispassionate look at vision-language models in medical scenario (https://arxiv.org/abs/2408.08704)
Comments:
          10 pages

- **What's New**: 최근 대규모 비전-언어 모델(Large Vision-Language Models, LVLMs)의 발전은 다양한 작업에서 두드러진 능력을 보여주며 AI 커뮤니티에서 큰 주목을 받고 있습니다. 그러나 의료와 같은 전문 분야에서의 성능과 신뢰성은 충분히 평가되지 않았습니다. 이 연구에서는 기존 LVLMs를 포괄적으로 평가하기 위해 RadVUQA라는 새로운 방사선 시각 이해 및 질문 응답 벤치마크를 도입했습니다.

- **Technical Details**: RadVUQA는 LVLMs의 성능을 1) 해부학적 이해(anatomical understanding), 2) 다중모달 이해(multimodal comprehension), 3) 정량적 및 공간적 추론(quantitative and spatial reasoning), 4) 생리학적 지식(physiological knowledge), 5) 강건성(robustness) 등 5가지 차원에서 평가합니다. 이 연구에서는 117개의 장기/구조와 56개의 장기의 CT 및 MR 스캔을 포함한 새로운 데이터셋을 구축하였으며, 다양한 시험 환경에서 LVLM의 성능을 테스트합니다.

- **Performance Highlights**: 평가 결과, 일반 LVLM 및 의료 특화 LVLM 모두가 다중 모달 이해와 정량적 추론 능력에서 중대한 결함을 보였습니다. 이 연구는 LVLM과 임상의 간의 큰 격차를 밝혀내며, 더 강력하고 지능적인 LVLM의 필요성을 강조합니다.



### TsCA: On the Semantic Consistency Alignment via Conditional Transport for Compositional Zero-Shot Learning (https://arxiv.org/abs/2408.08703)
- **What's New**: 본 논문에서는 Compositional Zero-Shot Learning (CZSL)에서 발생하는 문제들을 해결하기 위해 'Trisets Consistency Alignment (TsCA)' 프레임워크를 제안합니다. 이 프레임워크는 이미지, 구성, 그리고 원형(patch, composition, primitive) 간의 정밀한 정렬을 달성하기 위해 설계되었습니다.

- **Technical Details**: TsCA는 세 개의 서로 다른 분포(𝐏1, 𝐏2, 𝐏3)를 활용하여 조건부 전송(Conditional Transport, CT)을 최소화하기 위한 쌍별 비용을 구성합니다. 또한, 사이클 일관성 제약 조건을 구현하여 모달리티에 관계없이 특징의 일관성을 보장하는 학습을 수행합니다. 제안된 CCT(Consistency-aware Conditional Transport) 이론은 세 가지 분포 간의 복잡한 관계를 모델링하고, 각 분포가 동일한 의미를 표현함을 고려하여 정렬 강건성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, TsCA는 기존 방법들보다 더 높은 예측 성능을 기록하며, 비현실적인 쌍을 효과적으로 필터링하여 추론 속도를 증가시키고 정확도를 높이는 것을 입증하였습니다.



### HyCoT: Hyperspectral Compression Transformer with an Efficient Training Strategy (https://arxiv.org/abs/2408.08700)
- **What's New**: 새 논문에서는 하이퍼스펙트럴 이미지(HSI) 압축을 위한 변환기 기반 오토인코더 모델인 HyCoT(Hyperspectral Compression Transformer)를 제안합니다. 기존의 컨볼루션 필터를 이용한 방법의 한계를 극복하고, 효율적인 학습 전략을 도입하여 훈련 속도를 개선하였습니다.

- **Technical Details**: HyCoT는 스펙트럴 종속성을 활용하여 잠재 공간을 인코딩하는 데 Transformer 블록을 사용합니다. 또한, 경량 디코더를 통해 빠른 재구성을 가능하게 하고, 훈련 세트의 크기를 줄임으로써 훈련 비용을 절감합니다. 이를 통해 HyCoT는 기존의 압축 방법에 비해 높은 재구성 품질을 보여주었습니다.

- **Performance Highlights**: HyCoT는 HySpecNet-11k 데이터셋에서 테스트된 결과, 다양한 압축 비율에서 기존의 최첨단 기술보다 1 dB 이상 우수한 성능을 보이며, 계산 복잡성을 상당히 줄였습니다.



### Adaptive Layer Selection for Efficient Vision Transformer Fine-Tuning (https://arxiv.org/abs/2408.08670)
- **What's New**: 본 논문에서는 Vision Transformers (ViTs)의 효율적인 fine-tuning 방법인 ALaST(Adaptive Layer Selection Fine-Tuning for Vision Transformers)를 소개합니다. 이 방법은 fine-tuning 과정에서 필요한 자원 소모를 줄이면서도 속도를 향상시킵니다.

- **Technical Details**: ALaST는 각 layer의 중요도를 adaptively 추정하고, 이에 따라 'compute budgets'를 할당하여 자원 소모를 조정합니다. 낮은 budget이 할당된 layer는 input tokens의 수를 줄여서 학습하거나 동결(freeze)하여 계산 비용을 절감합니다. 연산 중 token을 discard하면 처리 속도를 높이고 메모리 요구량을 줄입니다.

- **Performance Highlights**: ALaST는 기존의 full fine-tuning 접근 방식에 비해 훈련 시간을 최대 1.5배, FLOPs를 최대 2배, 메모리 사용량을 최대 2배 감소시킵니다. 또한 LoRA와 같은 다른 파라미터 효율적인 fine-tuning 방법과 성공적으로 결합하여 사용할 수 있습니다.



### QMambaBSR: Burst Image Super-Resolution with Query State Space Mod (https://arxiv.org/abs/2408.08665)
- **What's New**: 본 논문에서는 새로운 Query Mamba Burst Super-Resolution (QMambaBSR) 네트워크를 도입하여, 고해상도 이미지를 복원하는 데 있어 burst low-resolution 이미지의 서브픽셀 정보를 최대한 활용하는 방법을 제안합니다. 이 네트워크는 Query State Space Model (QSSM)과 Adaptive Up-sampling (AdaUp) 모듈을 통합하여, 기존 방법에서의 단점인 노이즈 간섭과 정확한 서브픽셀 추출 문제를 해결합니다.

- **Technical Details**: QSSM은 inter-frame 쿼리와 intra-frame 스캔을 통해 서브픽셀을 효율적으로 추출하고, AdaUp 모듈은 다양한 burst 저해상도 시나리오에서 서브픽셀 정보의 공간 분포에 따라 업샘플링 커널을 동적으로 조정합니다. 이러한 과정을 통해 고해상도 세부 정보를 효과적으로 재구성할 수 있습니다. 또한, Multi-scale Fusion 모듈을 통해 다양한 스케일의 서브픽셀 정보를 융합합니다.

- **Performance Highlights**: 본 방법은 네 가지 인기 있는 합성 및 실제 벤치마크에서 새로운 최신 성과를 달성하였으며, 전반적으로 뛰어난 시각적 결과를 보여줍니다. QMambaBSR은 기존의 방법들과 비교해 더 높은 품질의 이미지 복원을 가능하게 합니다.



### Extracting polygonal footprints in off-nadir images with Segment Anything Mod (https://arxiv.org/abs/2408.08645)
- **What's New**: 이 논문에서는 OBMv2라는 새로운 모델이 제안되었습니다. OBMv2는 건물의 폴리곤 형태를 직접 예측할 수 있는 최초의 모델로, 응답 대기 없이 엔드 투 엔드 방식을 지원합니다.

- **Technical Details**: OBMv2는 Self Offset Attention (SOFA) 레이어를 사용하여 오프셋 예측 성능을 개선하고, Dynamic Scope Binary Cross Entropy Loss (DS-BCE Loss)를 포함하여 출력 결과의 품질을 향상시킵니다. 여러 개의 태스크를 결합하여 다중 작업 학습을 지원합니다.

- **Performance Highlights**: OBMv2는 공개 데이터셋인 BONAI와 OmniCity-view3에서 실험을 통해 효과를 검증했으며, Huizhou 테스트 세트에서의 일반화 테스트 결과도 제공합니다. 모델은 기존 방법들보다 더 높은 정확도와 일관된 결과를 보여줍니다.



### Historical Printed Ornaments: Dataset and Tasks (https://arxiv.org/abs/2408.08633)
- **What's New**: 이 논문은 역사적인 인쇄 장식의 연구를 현대의 비지도 컴퓨터 비전으로 발전시키는 것을 목표로 합니다.

- **Technical Details**: 주요 관심사인 세 가지 복잡한 작업(클러스터링 (clustering), 요소 발견 (element discovery), 비지도 변화 위치추적 (unsupervised change localization))을 강조하고 각 작업에 대한 평가 벤치마크를 소개합니다. 또한, 최신 모델을 조정하고 평가합니다.

- **Performance Highlights**: 실제 데이터에 직면했을 때 최신 모델의 한계를 보여주며, k-means와 같은 간단한 기초 방법이 데이터 기반에서 더 정교한 접근 방식을 능가할 수 있음을 보여줍니다.



### SketchRef: A Benchmark Dataset and Evaluation Metrics for Automated Sketch Synthesis (https://arxiv.org/abs/2408.08623)
- **What's New**: 본 논문에서는 스케치 합성의 품질 평가를 위한 새로운 벤치마크 데이터셋인 SketchRef를 소개하고, 그것을 통해 스케치 구조적 일관성을 평가하기 위한 mOKS(mean Object Keypoint Similarity)라는 평가 메트릭을 제안합니다.

- **Technical Details**: SketchRef 데이터셋은 동물, 인물 얼굴, 인체, 사물의 4개 카테고리로 구성되며, 여러 수준의 단순화가 적용된 스케치와 참조 사진 간의 시각적 및 의미적 연결을 제공합니다. 또한, 구조 수준 인식 가능성을 평가하는 데 필요한 새로운 방식으로 단순화를 제한하는 인식 가능성 계산 방법을 소개합니다.

- **Performance Highlights**: 198명의 아트 애호가로부터 수집된 8K 응답을 통해 제안된 평가 방법의 유용성이 검증되었으며, SketchRef를 통해 8개의 스케치 합성 방법에 대한 포괄적 평가가 수행되었습니다.



### Generative Dataset Distillation Based on Diffusion Mod (https://arxiv.org/abs/2408.08610)
Comments:
          The Third Place Winner in Generative Track of the ECCV 2024 DD Challenge

- **What's New**: 이번 논문은 ECCV 2024에서의 첫 번째 데이터셋 디스틸레이션 챌린지의 생성 트랙을 위한 새로운 방법을 제시합니다. 이 방법은 고품질 생성 효과로 인해 확산 모형(Diffusion model)에 기반한 데이터셋 디스틸레이션을 중점적으로 다룹니다.

- **Technical Details**: 본 연구에서는 고속 및 고품질 이미지를 생성할 수 있는 SDXL-Turbo 모델을 활용한 새로운 데이터셋 디스틸레이션 방법을 개발했습니다. 우리 방법은 CIFAR-100과 Tiny-ImageNet 데이터셋을 대상으로 하여, Tiny-ImageNet에서 IPC(Images Per Class)가 10, CIFAR-100에서 IPC가 20에 달하는 성과를 보였습니다. 또한, 클래스 정보를 텍스트 프롬프트로 사용하고, 후처리 데이터 증강(Post data augmentation) 기법을 적용하여 품질이 높은 디스틸레이션 데이터셋을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 획기적인 효과를 보였으며, ECCV 2024 DD 챌린지 생성 트랙에서 3위를 차지했습니다.



### Bi-Directional Deep Contextual Video Compression (https://arxiv.org/abs/2408.08604)
- **What's New**: 최근까지 비디오 압축 기술에서 P-frame 코딩의 발전이 두드러졌으나, B-frame 코딩의 성능이 전통적인 양방향 비디오 코덱에 비해 떨어진다는 문제가 있었습니다. 본 논문에서는 B-frame 코딩 성능을 개선하기 위해 새로운 양방향 심층 맥락 비디오 압축 기법인 DCVC-B를 제안합니다.

- **Technical Details**: DCVC-B는 비효율적인 비트 비용을 줄이기 위해 양방향 모션 차이 맥락 전파 방법을 개발하고, 다중 스케일의 시계열 맥락을 보다 잘 활용할 수 있는 양방향 맥락 압축 모델 및 해당 양방향 시간 엔트로피 모델을 제안합니다. 또한, 대규모 GOP 간에 효과적인 비트 할당을 가능하게 하는 계층적 품질 구조 기반의 훈련 전략을 개발하였습니다.

- **Performance Highlights**: DCVC-B는 랜덤 엑세스 환경에서 H.265/HEVC 기준 소프트웨어에 비해 평균 26.6%의 BD-Rate 감소를 달성하였으며, 특정 테스트 데이터셋에서는 H.266/VVC 기준 소프트웨어의 성능을 초월하는 결과를 보였습니다.



### Learning A Low-Level Vision Generalist via Visual Task Promp (https://arxiv.org/abs/2408.08601)
Comments:
          Accepted to ACMMM24

- **What's New**: 이 논문에서는 다양한 저수준 비전(low-level vision) 작업을 처리하기 위한 일반화된 모델인 GenLV를 제안합니다. 이는 Visual task Prompt-based Image Processing (VPIP) 프레임워크를 기반으로 하여, 여러 입력-목표 도메인에 걸쳐 작업을 처리하는 데 탁월한 성능을 보입니다.

- **Technical Details**: VPIP 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 엔드투엔드(End-to-End) 이미지 처리 네트워크, 프롬프트 인코더(sub-network) 및 작업별 처리에 필요한 정보 상호작용 메커니즘. 주 네트워크로는 X-Restormer를 사용하여 일반 이미지 복원 작업을 수행합니다. 프롬프트 인코더는 시각적 프롬프트를 잠재 표현(latent representation)으로 변환하고, 새로운 프롬프트 크로스-어텐션(prompt cross-attention)을 도입하였습니다.

- **Performance Highlights**: GenLV 모델은 30개의 다양한 저수준 작업을 처리하는 데 있어 기존 방법에 비해 양적 및 질적으로 뛰어난 성능을 보였습니다. 특히 다양한 입력 및 목표 도메인에서의 효과적인 처리 능력이 입증되었습니다.



### MM-UNet: A Mixed MLP Architecture for Improved Ophthalmic Image Segmentation (https://arxiv.org/abs/2408.08600)
Comments:
          OMIA2024

- **What's New**: 이번 연구에서는 안과 이미지 분할을 위한 새로운 Mixed MLP 아키텍처인 MM-UNet을 제안합니다. 이 모델은 다양한 깊이에서 특징의 상호작용을 촉진하는 multi-scale MLP (MMLP) 모듈을 포함하고 있습니다.

- **Technical Details**: MM-UNet은 UNet 구조를 기반으로 하며, MMLP 모듈을 통해 지역 정보와 전역 정보를 동시에 캡처합니다. 이 아키텍처는 채널 혼합 MLP를 생략하고 지역 토큰 혼합만을 사용하여 계산 비용을 줄여줍니다.

- **Performance Highlights**: MM-UNet은 사전 세그먼트 Optical Coherence Tomography (AS-OCT) 이미지 데이터셋과 공공 안저 이미지 데이터셋에서 실험을 수행했으며, 최신 딥 세그멘테이션 네트워크와 비교하여 우수한 성능을 보였습니다.



### Zero-Shot Dual-Path Integration Framework for Open-Vocabulary 3D Instance Segmentation (https://arxiv.org/abs/2408.08591)
Comments:
          OpenSUN 3D: 2nd Workshop on Open-Vocabulary 3D Scene Understanding (CVPR 2024)

- **What's New**: 본 논문에서는 Open-vocabulary 3D instance segmentation 기술을 개발하여, 기존의 Closed-vocabulary 방법을 넘어서는 성과를 발표했습니다. 새로운 시스템은 3D 포인트 클라우드와 2D 멀티 뷰 이미지를 모두 활용하여 클래스에 구애받지 않는 객체 마스크 제안을 생성합니다.

- **Technical Details**: 제안된 프레임워크는 3D 경로(3D pathway), 2D 경로(2D pathway), 그리고 Dual-Path Integration으로 구성됩니다. 3D 경로는 사전 훈련된 3D 모델을 사용하여 일반적인 실내 객체에 대한 공간적으로 정확한 클래스 비구속 마스크 제안을 생성하고, 2D 경로는 사전 훈련된 오픈-어휘 인스턴스 세그멘테이션 모델을 활용하여 다양한 객체 제안을 식별합니다. Dual-Path Integration 부분에서는 제안된 조건부 통합(Conditional Integration) 프로세스를 통해 두 경로에서 나온 제안을 적응적으로 필터링하고 병합합니다.

- **Performance Highlights**: 제안된 프레임워크는 사전 훈련된 모델을 제로샷(zero-shot) 방식으로 활용하였으며, 모델 의존성이 없어 ScanNet200 데이터셋과 ARKitScenes 데이터셋에 대한 평가에서 기존 기술보다 우수한 성능을 보여주었습니다.



### TAMER: Tree-Aware Transformer for Handwritten Mathematical Expression Recognition (https://arxiv.org/abs/2408.08578)
- **What's New**: 새로운 모델 TAMER(Tree-Aware Transformer)를 제안하여 수식 인식의 정확성을 향상시켰습니다. 이 모델은 수식의 나무 구조를 인식할 수 있는 혁신적인 Tree-aware Module을 도입하여 시퀀스 예측과 트리 구조 예측 작업을 결합하여 일반화 능력을 증대시켰습니다.

- **Technical Details**: TAMER는 Transformer의 유연성과 효율성을 유지하면서 트리 구조를 인식하는 과제를 해결합니다. 이 모델은 시퀀스 기반 예측과 트리 구조 예측을 공동 최적화하며, 인퍼런스 단계에서는 Tree Structure Prediction Scoring Mechanism을 사용하여 LaTeX 시퀀스의 문법적 정확성을 높입니다.

- **Performance Highlights**: CROHME 데이터셋에서 TAMER는 전통적인 시퀀스 및 트리 기반 디코딩 모델보다 뛰어난 성능을 보였으며, 복잡한 수학적 구조를 처리하는 데 특히 강력하여 SOTA 성능을 달성했습니다. 예를 들어, CROHME 2014/2016/2019 테스트 세트에서 각각 61.23%, 60.26%, 61.97%의 표현 인식률을 기록했습니다.



### Tuning a SAM-Based Model with Multi-Cognitive Visual Adapter to Remote Sensing Instance Segmentation (https://arxiv.org/abs/2408.08576)
- **What's New**: 이 논문에서는 Remote Sensing (원거리 감지) 분야에 적용하기 위해 Segment Anything Model (SAM)을 기반으로 한 다중 인지 SAM 기반 instance segmentation 모델(MC-SAM SEG)을 제안합니다. MC-SAM SEG는 Remote Sensing 이미지에 대한 자동 마스크 생성 및 인스턴스 분류의 성능을 향상시키기 위해 Multi-cognitive Visual Adapter (Mona)를 사용하고, 파라미터 효율적 미세 조정(PEFT) 기술을 도입하였습니다.

- **Technical Details**: MC-SAM SEG는 SAM과 Mona를 통합하여 Remote Sensing 이미지에 대한 고품질 feature를 추출합니다. SAM-Mona 인코더는 Remote Sensing의 복잡한 장면 분할 및 SAR 산란 특성 학습을 용이하게 하며, 인스턴스에 대한 마스크와 카테고리를 자동으로 생성하여 prompt 입력 없이 작동합니다. 또한, 다중 해상도에서 pixel decoder와 transformer decoder를 사용하여 최종 마스크를 생성합니다.

- **Performance Highlights**: MC-SAM SEG는 Optical 이미지 WHU 데이터셋에서 71.2%의 APmask를 달성하고, SAR 선박 데이터셋 HRSID에서 66.4%의 APmask를 기록하며 최신 기술 대비 우수한 성능을 입증하였습니다.



### Tell Codec What Worth Compressing: Semantically Disentangled Image Coding for Machine with LMMs (https://arxiv.org/abs/2408.08575)
- **What's New**: 이번 논문에서는 대규모 멀티모달 모델(Large Multimodal Models, LMM)을 활용하여 기계 작업에 최적화된 이미지 압축 프레임워크인 "이미지 코딩을 위한 기계(Imagen Coding for Machines, ICM)"를 제안합니다. 이는 전통적인 이미지 압축 방식과는 달리, 머신과의 하위 작업을 지원하기 위해 압축 비트스트림을 설계합니다.

- **Technical Details**: 제안된 방법인 "SDComp"(Semantically Disentangled Compression)는 LMM을 이용하여 이미지의 객체 입지를 파악하고 중요도를 순위 지정하여 압축합니다. 이는 구조화된 비트스트림을 통해 다양한 비전 태스크(예: 이미지 분류, 객체 탐지 등)를 효과적으로 지원합니다. 본 연구에서는 객체의 중요도를 평가하기 위해 LMM을 활용하는 프롬프트 설계를 포함하고, 세멘틱 구조화 이미지 압축(Semantically Structured Image Compression, SSIC) 방법을 개선하여 실현하였습니다.

- **Performance Highlights**: SDComp는 VTM(영상 전송 모듈) 대비 평균 32.3%의 성능 향상이 검증되었습니다. 또한, 시각적 질문-답변 태스크에 대한 실험에서도 높은 일반화 성능을 보여주었으며, 프롬프트 기반 및 시각화 기법을 통한 해석 가능성도 탐구하였습니다.



### EraW-Net: Enhance-Refine-Align W-Net for Scene-Associated Driver Attention Estimation (https://arxiv.org/abs/2408.08570)
Comments:
          13pages, 9 figures,

- **What's New**: 이 논문에서는 운전자의 주의력을 주행 장면과 연관짓는 새로운 방법인 EraW-Net을 제안합니다. 이 방법은 복잡한 관계를 모델링하고 두 시각의 정보를 체계적으로 통합합니다.

- **Technical Details**: EraW-Net은 W-Net 구조를 기반으로 하여 Dynamic Adaptive Filter Module (DAF-Module)과 Global Context Sharing Module (GCS-Module)을 포함하여, 각기 다른 특징을 보완하고 합치는 방식으로 이루어져 있습니다. DAF-Module은 동적 환경에서의 정보 추출과 관련하여 혁신적인 주파수-공간 분석을 통해 중요한 동적 영역을 강조하는데 초점을 맞추고, GCS-Module은 다양한 비율의 얼굴 포즈에 적응할 수 있도록 계층적 특징을 캡처하여 정제된 특징 표현을 구축합니다.

- **Performance Highlights**: 제안된 EraW-Net은 대규모 공개 데이터셋에서 운전자의 픽셀 수준 주의 맵핑을 정확하게 추정하였으며, 기존 방법들보다 우수한 성능을 보여주었습니다.



### Unsupervised Non-Rigid Point Cloud Matching through Large Vision Models (https://arxiv.org/abs/2408.08568)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 비강체(Non-rigid) 포인트 클라우드 매칭을 위한 새로운 학습 기반의 프레임워크를 제안합니다. 이 프레임워크는 어떤 대응 주석 없이 순전히 포인트 클라우드로 교육될 수 있으며 부분에서 전체 매칭으로 자연스럽게 확장될 수 있습니다.

- **Technical Details**: 본 프레임워크는 대규모 비전 모델(Large Vision Models, LVMs)에서 유도된 의미론적 특성을 기하학 기반의 형태 특징 학습에 통합합니다. 이를 통해 자기 유사성으로 인한 모호성을 해결하며, 부분 관측에 대해 강한 일반화 및 견고성을 갖습니다. 이 과정에서 픽셀-투-포인트(feature aggregation module), 지역 및 전역 주의 네트워크(local and global attention network), 그리고 기하학적 유사성 손실 함수(geometrical similarity loss function)를 제안합니다.

- **Performance Highlights**: 실험 결과에 따르면, 이 방법은 비강체 포인트 클라우드 매칭에서 최신 성과를 달성하며, 특히 근사 등각(near-isometric) 및 이질적(heterogeneous) 형태 수집에서도 뛰어난 성능을 보였습니다. 또한, 부분 포인트 클라우드와 노이즈가 있는 현실적 데이터에서도 경쟁적인 방법들보다 우수한 결과를 보여줍니다.



### A New Chinese Landscape Paintings Generation Model based on Stable Diffusion using DreamBooth (https://arxiv.org/abs/2408.08561)
Comments:
          accepted by AHPCAI

- **What's New**: 이 연구는 Stable Diffusion Model (SDM)과 Parameter-Efficient Fine-Tuning 방식을 결합하여 중국 산수화(Chinese Landscape Paintings)를 생성하는 새로운 방법을 소개합니다.

- **Technical Details**: 이 훈련 과정은 LoRA를 사전 훈련된 SDM과 결합하고, DreamBooth를 사전 훈련된 SDM과 각각 결합하여 가속화됩니다. 연구에 사용된 중국 산수화 인터넷 데이터셋에서 SDM과 DreamBooth의 결합은 우수한 성능을 보여주며, 일반적인 사전 훈련된 SDM 및 LoRA 기반 세부 조정 SDM을 능가합니다.

- **Performance Highlights**: SDM과 DreamBooth 결합 모델은 데이터셋에서 FID (Fréchet Inception Distance) 12.75를 달성하였고, 전문가 평가에서도 모든 모델을 초월하여 중국 산수화 분야에서의 모델의 다재다능성을 강조합니다. 이 연구는 전문화된 세부 조정 방법이 특정 분야 작업에서 SDM의 성능을 개선할 수 있는 잠재력을 보여줍니다.



### A training regime to learn unified representations from complementary breast imaging modalities (https://arxiv.org/abs/2408.08560)
- **What's New**: 이 연구는 Digital Breast Tomosynthesis (DBT)와 Full Field Digital Mammography (FFDM)간의 상호작용을 통해 유방 병변 감지의 정확성을 향상시키는 새로운 머신 러닝 모델을 제안합니다.

- **Technical Details**: 제안된 구조는 DBT와 FFDM 간의 관계를 학습하여 SM(Synthetic Mammogram) 이미지를 통해 단일 모달리티로 예측을 수행할 수 있게 합니다. 이 모델은 EfficientDet을 기반으로 하여 SM 이미지를 처리하며, FFDM의 정보를 효과적으로 통합합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 DBT 또는 FFDM 단독으로 학습한 모델들보다 더 높은 정확도로 유방 병변을 감지함을 보여주었습니다.



### Scaling up Multimodal Pre-training for Sign Language Understanding (https://arxiv.org/abs/2408.08544)
Comments:
          Sign language recognition; Sign language translation; Sign language retrieval

- **What's New**: 이번 연구에서는 수화 이해(Sign Language Understanding, SLU)를 위한 새로운 다중 모달(framework) 구조를 제안합니다. 이 구조는 시각적 맥락 정보와 비주얼-텍스트(vision-text) 의미 상관관계를 활용하여 수화의 표현력을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 연구진은 약 150만 개의 텍스트 라벨이 붙은 수화 포즈 데이터셋(SL-1.5M)을 수집하여 데이터 부족 문제를 해결하고자 합니다. 이 데이터셋은 여러 개의 수화 데이터셋과 미가공 데이터셋(BOBSL)에서 수집된 데이터를 포함합니다. 또한, 다중 태스크(pre-training) 전략을 통해 수화 비디오의 미세한 표현을 효과적으로 학습합니다.

- **Performance Highlights**: 제안된 모델은 다양한 SLU 작업에서 기존의 RGB 기반 방법들과 비교하여 우수한 성능을 보여주었으며, 다수의 벤치마크에서 새로운 최첨단 결과를 달성했습니다. 특히, 모델은 포즈 기반 접근법에서 새로운 기록을 세우며 효과적으로 성능 병목 문제를 완화했습니다.



### Language-Driven Interactive Shadow Detection (https://arxiv.org/abs/2408.08543)
Comments:
          ACM MM 2024

- **What's New**: 이 연구에서는 Referring Video Shadow Detection (RVSD)이라는 새로운 과제를 소개합니다. 이는 설명적인 자연어 프롬프트를 바탕으로 비디오 내 특정 그림자를 분할하는 혁신적인 접근 방식입니다.

- **Technical Details**: RVSD를 위해 86개의 비디오와 15,011개의 텍스트 설명이 쌍을 이루는 데이터셋을 수집했습니다. 제안하는 Referring Shadow-Track Memory Network (RSM-Net)는 Twin-Track Synergistic Memory (TSM)와 Mixed-Prior Shadow Attention (MSA) 모듈을 포함하고 있습니다. TSM은 클립 간 및 클립 내 메모리 기능을 저장하고, MSA는 물리적 지식을 바탕으로 대략적인 그림자 맵을 생성합니다.

- **Performance Highlights**: RSM-Net은 RVSD 과제에서 기존 기술들보다 4.4% 향상된 Overall IOU를 기록하며, 최첨단 성능을 달성했습니다.



### Privacy-Preserving Vision Transformer Using Images Encrypted with Restricted Random Permutation Matrices (https://arxiv.org/abs/2408.08529)
Comments:
          4 pages, 9 figures

- **What's New**: 새로운 방법론을 제안하여 암호화된 이미지를 사용한 비전 트랜스포머(ViTs)의 개인정보 보호형 파인튜닝을 개선하였다. 기존의 방식에 비해 성능 저하를 최소화하며, 암호화된 이미지를 사용하면서도 성능이 개선되는 것을 확인하였다.

- **Technical Details**: 제안된 방법은 비전 트랜스포머(ViT)에 사용되는 암호화된 이미지를 통해 모델 성능의 저하를 줄이는 것을 목표로 한다. 이 과정에서 제한된 랜덤 순열 행렬을 이용하여 훈련 이미지를 암호화하고, 같은 비밀 키를 이용해 쿼리 이미지를 암호화하여 모델을 개선한다. 암호화 과정은 블록 단위의 스크램블링(block scrambling)과 픽셀 순열로 구성된다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 방안들보다 높은 성능을 보임을 검증하였다. 암호화된 이미지 환경에서도 효율적인 분류 결과를 달성함으로써, 클라우드 환경에서의 개인 정보 보호와 깊은 신경망 학습 사이의 균형을 이루었다.



### Focus on Focus: Focus-oriented Representation Learning and Multi-view Cross-modal Alignment for Glioma Grading (https://arxiv.org/abs/2408.08527)
- **What's New**: 이 논문에서는 Focus on Focus (FoF) 프레임워크를 도입하여 병리학적 데이터와 유전자 정보를 결합하고, 단일 병리 데이터로도 활용할 수 있는 방법을 제시합니다. 이 프레임워크는 병리학적 표현을 효과적으로 개선하여 글리오마( glioma )의 등급을 더욱 정확하게 구분할 수 있도록 합니다.

- **Technical Details**: FoF 프레임워크는 Focus-oriented Representation Learning (FRL)과 Multi-view Cross-modal Alignment (MCA) 모듈로 구성됩니다. FRL 모듈은 모델이 글리오마 등급과 양성 또는 음성으로 관련된 영역을 식별하도록 유도합니다. MCA 모듈은 병리학적 표현을 유전자 기반 서브스페이스로 투영하여, 텍스처 특징과 유전자 생체표지자 상태를 일치시키는 방식입니다.

- **Performance Highlights**: TCGA GBM-LGG 데이터셋에서 FoF 프레임워크가 기존의 다중 모드 방법보다 우수한 성능을 보이며, 오직 병리학 슬라이드를 사용하여도 뛰어난 결과를 달성합니다. 또한, 병리학적 슬라이드 만으로도 글리오마의 등급을 정확히 구분하는 데에 있어 임상적 의의가 큽니다.



### GS-ID: Illumination Decomposition on Gaussian Splatting via Diffusion Prior and Parametric Light Source Optimization (https://arxiv.org/abs/2408.08524)
Comments:
          15 pages, 13 figures

- **What's New**: GS-ID라는 새로운 프레임워크를 제안하며, 이는 Gaussian Splatting을 기반으로 조명 분해(illumination decomposition)를 수행하여 사실적인 새로운 시점 합성을 실현하고 직관적인 조명 편집을 가능하게 합니다.

- **Technical Details**: GS-ID는 두 가지 주요 구성 요소인 잠재적 확산 사전(intrinsic diffusion priors)과 Spherical Gaussians (SGs)를 활용하여 물리 기반 렌더링을 위한 속성을 추정하고, 환경 조명과 직접 조명을 분해하여 최적화를 진행합니다. 이 프레임워크는 배치 렌더링(deferred rendering) 기법을 채택하여 계산 부하를 줄입니다.

- **Performance Highlights**: 우리의 실험 결과 GS-ID는 현대적인 조명 분해 방법에 비해 우수한 성능을 보이며, 더욱 나은 형태 재구성과 렌더링 성능을 달성하여 길어지는 조명 효과를 효과적으로 제어할 수 있습니다.



### Visual-Friendly Concept Protection via Selective Adversarial Perturbations (https://arxiv.org/abs/2408.08518)
Comments:
          Under Review

- **What's New**: 본 연구에서는 Visual-Friendly Concept Protection (VCPro) 프레임워크를 제안하여, 이미지 소유자가 선택한 주요 개념을 보다 낮은 인지도로 보호하는 적대적 변형을 활용합니다. 기존 방법들이 보호 효과에 집중한 반면, VCPro는 진정한 개념 보호의 시각적 외관도 고려합니다.

- **Technical Details**: VCPro 프레임워크는 사용자가 제공한 마스크를 사용하여 중요한 영역에 선택적인 적대적 변형을 적용합니다. 이를 통해 시각적 품질을 높이고, 더욱 미세한 적대적 변형을 생성하기 위해 Lagrangian multiplier 방법을 활용하여 최적화 목표를 완화합니다.

- **Performance Highlights**: VCPro는 Mist 및 Anti-DreamBooth와 비교하여 FID 점수를 96.24에서 27.03으로 감소시키며, 보호 효과와 변형 가시성 간의 더 나은 균형을 달성했습니다.



### Efficient Image-to-Image Diffusion Classifier for Adversarial Robustness (https://arxiv.org/abs/2408.08502)
- **What's New**: 이 논문에서는 기존의 Diffusion Models (DMs)을 기반으로 하는 적대적 방어 기법의 한계를 극복하기 위해 새로운 접근법을 제안합니다. 구체적으로, DMs의 높은 이미지 생성 품질을 유지하면서, 적은 계산 비용으로 이미지 분류 작업을 수행할 수 있는 효율적인 Image-to-Image Diffusion Classifier (IDC)를 소개합니다.

- **Technical Details**: 제안된 IDC는 다수의 입력 샘플을 정의된 직교 이미지 레이블로 변환하기 위해 이미지 변환 프레임워크를 활용합니다. 이를 통해, 입력 이미지와 이미지 레이블 간의 다대일 매핑(many-to-one mapping)을 학습하며, 계산 복잡도를 줄이는 동시에 이전 DMs의 성능을 해치지 않도록 설계되었습니다. 네트워크는 pruning된 U-Net 구조를 채택하고, 확산 시간(steps)을 줄였습니다.

- **Performance Highlights**: 제안된 방법은 다양한 공격에 대해 광범위한 평가에서 기존 DM 기반 및 CNN 기반 방법보다 더 나은 적대적 강인성을 보여주었으며, 모델 효율성과 적대적 강인성 간의 트레이드오프에서 우수한 성능을 보여줍니다. 실험 결과, 제안된 IDC는 CNN 기반 적대적 훈련 방법보다 더 적은 모델 매개변수로 더 경쟁력 있는 강인성을 달성함을 보여줍니다.



### CoSEC: A Coaxial Stereo Event Camera Dataset for Autonomous Driving (https://arxiv.org/abs/2408.08500)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 연구에서는 자율주행을 위한 새로운 신호 체계로 하이브리드 동축 이벤트-프레임 장치를 도입하였습니다. 이 장치는 이벤트 카메라의 데이터와 프레임 카메라의 데이터를 결합할 수 있는 멀티모달 시스템을 구축합니다.

- **Technical Details**: 제안된 시스템은 마이크로컨트롤러를 이용하여 시간 동기화(time synchronization)를 달성하고, 다양한 센서 간의 공간 보정을 진행합니다. 주요 기술로는 하이브리드 동축 이벤트 카메라(Coaxial Stereo Event Camera, CoSEC) 데이터 세트와 LiDAR 포인트 클라우드를 활용한 깊이(depth) 및 광속(optical flow) 레이블 생성 방법이 포함됩니다.

- **Performance Highlights**: 임상 실험 결과, 제안된 데이터 세트가 픽셀 수준의 멀티모달 융합(performance of multimodal fusion) 성능과 일반화 능력(genaralization of multimodal fusion)을 향상시킬 수 있음을 입증하였습니다.



### Achieving Complex Image Edits via Function Aggregation with Diffusion Models (https://arxiv.org/abs/2408.08495)
- **What's New**: 이번 연구는 FunEditor라는 새로운 효율적인 diffusion model을 소개합니다. 이 모델은 atomic editing functions를 학습하고, 복잡한 편집 작업을 간단한 함수들의 집합으로 수행할 수 있도록 설계되었습니다. FunEditor는 복잡한 작업을 동시에 적용할 수 있어, 기존의 모델들이 겪는 비효율성을 극복합니다.

- **Technical Details**: FunEditor는 학습 중에 간단한 atomic 편집 작업만 필요로 하며, 여러 개의 task tokens를 정의하여 이를 inference 시 조합하여 응용합니다. 각 간단한 작업은 이진 마스크를 입력으로 받아 cross-attention을 통해 수정합니다. 이 방법은 단순 작업에서도 높은 데이터 효율성을 제공합니다.

- **Performance Highlights**: FunEditor는 복잡한 편집 작업에서 기존 모델들보다 5배에서 24배 빠른 추론 속도를 보입니다. 특히, COCOEE 및 ReS 데이터셋을 활용한 평가에서 이미지 품질 평가(IQA) 및 오브젝트-배경 일관성 등의 다양한 메트릭에서 현저한 우수성을 입증하였습니다.



### TEXTOC: Text-driven Object-Centric Style Transfer (https://arxiv.org/abs/2408.08461)
- **What's New**: 우리의 연구는 텍스트 기반의 오브젝트 중심 스타일 전송(Text-driven Object-Centric Style Transfer, TEXTOC)이라는 새로운 접근 방식을 제안합니다. 이 방법은 텍스트 입력을 기반으로 오브젝트의 스타일 전송을 구동하는 혁신적인 방식입니다.

- **Technical Details**: TEXTOC의 핵심은 Patch-wise Co-Directional (PCD) 손실 함수로, 이는 텍스트에 맞춘 정밀한 오브젝트 중심 변환을 위해 설계되었습니다. 이 손실 함수는 텍스트 기반 스타일 방향을 위한 패치 방향 손실(patch directional loss)과 객체 영역 간 CLIP(Contrastive Language-Image Pretraining) 임베딩 분포의 일관성을 유지하기 위한 패치 분포 일관성 손실(patch distribution consistency loss)을 결합합니다. 또한, 텍스트에 따라 객체 위치를 식별하는 Text-Matched Patch Selection (TMPS) 및 Pre-fixed Region Selection (PRS) 모듈을 도입합니다. 이와 함께 Adaptive Background Preservation (ABP) 손실 함수를 도입하여 이미지의 배경의 원래 스타일과 구조를 유지합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TEXTOC의 효과성을 강조하였습니다. 이 방식은 시각적으로 일관되고 텍스트에 맞춰 정렬된 스타일 전송을 생성하는 데 효과적입니다. 이를 통해 광고, 영화, 비디오 게임 개발과 같은 창의적인 디지털 산업의 이미지 조작 수요를 충족시키고 있습니다.



### Beyond Uniform Query Distribution: Key-Driven Grouped Query Attention (https://arxiv.org/abs/2408.08454)
Comments:
          11 pages, 9 figures

- **What's New**: 본 연구에서는 Grouped Query Attention (GQA) 방식에 대한 두 가지 혁신적인 접근법인 Key-Distributed GQA (KDGQA)와 Dynamic Key-Distributed GQA (DGQA)를 소개하며, 이는 쿼리 할당을 정보 기반으로 수행합니다.

- **Technical Details**: KDGQA와 DGQA는 키 헤드의 L2-norms를 활용하여 동적으로 쿼리를 그룹화합니다. KDGQA는 각 forward pass에서 키 헤드의 샘플링 비율을 참조하는 반면, DGQA는 훈련 과정에서 변화하는 비율을 활용합니다. 또한 Perturbed GQA (PGQA)는 attention map에 Gaussian noise를 실험하여 그룹 형성에 변화를 주는 사례로 제안됩니다.

- **Performance Highlights**: ViT-L 모델을 CIFAR-10, CIFAR-100, Food101, Tiny ImageNet 데이터셋을 이용한 이미지 분류 실험에서 DGQA를 사용한 경우 GQA 및 기타 변형들과 비교하여 최대 8%의 정확도 향상을 보여주었습니다.



### SpectralEarth: Training Hyperspectral Foundation Models at Sca (https://arxiv.org/abs/2408.08447)
- **What's New**: 본 논문에서는 SpectralEarth라는 대규모 다중 시계열 hyperspectral 데이터셋을 소개합니다. 이 데이터셋은 지구 관측 임무에서 수집된 데이터를 기반으로 하여 구성되었으며, 538,974개의 이미지를 포함하고 있습니다. 이를 통해 hyperspectral foundation 모델의 사전 학습을 가능하게 하였습니다.

- **Technical Details**: SpectralEarth는 415,153개의 고유한 위치를 커버하고 있으며, 11,636개의 EnMAP 장면에서 수집된 데이터를 포함합니다. 약 17.5%의 위치는 여러 개의 타임스탬프를 포함하여 다중 시계열 HSI 분석을 지원합니다. 또한, Self-Supervised Learning (SSL) 알고리즘을 통해 클래식 비전 백본에 스펙트럼 어댑터를 통합하여 HSI의 고유한 특성을 반영하였습니다.

- **Performance Highlights**: 우리의 실험 결과는 모델의 다양한 태스크 및 센서에 대한 일반화 가능성을 보여줍니다. 더욱이, 해당 모델은 fine-tuning 시 빠른 수렴 속도를 보여주며, 데이터셋과 모델, 소스 코드는 공개될 예정입니다.



### PQV-Mobile: A Combined Pruning and Quantization Toolkit to Optimize Vision Transformers for Mobile Applications (https://arxiv.org/abs/2408.08437)
- **What's New**: 본 논문은 PQV-Mobile이라는 새로운 도구를 소개하며, 이는 비전 트랜스포머를 모바일 환경에 최적화하기 위해 가지치기(pruning)와 양자화(quantization)를 결합한 것입니다.

- **Technical Details**: PQV-Mobile은 중요도에 따라 다양한 구조적 가지치기를 지원하며(magnitude importance, Taylor importance, Hessian importance), FP32에서 FP16 및 int8으로의 양자화를 지원합니다. 이 도구는 x86, FBGEMM, QNNPACK, ONEDNN과 같은 여러 하드웨어 백엔드를 대상으로 최적화된 모델을 제작할 수 있습니다.

- **Performance Highlights**: DeiT 모델을 9.375% 가지치기하고 FP32에서 int8로 양자화할 경우 7.18배의 지연(latency) 감소와 2.24%의 미세한 정확도 손실을 보였습니다. 또한 PQV-Mobile을 통해 구조적 가지치기를 통해 최적화된 모델의 메모리 및 성능 성능을 검증했습니다.



### Penny-Wise and Pound-Foolish in Deepfake Detection (https://arxiv.org/abs/2408.08412)
- **What's New**: 이 논문은 새로운 딥페이크 감지 방법인 PoundNet을 제안합니다. 이 방법은 기존의 감지 모델들이 짧은 기간의 성과를 중시하는 "Penny-Wise and Pound-Foolish" 접근 방식을 비판하고, 일반화(generalization)와 지식 보존(knowledge retention)을 동시에 고려하는 방식을 채택합니다.

- **Technical Details**: PoundNet은 사전 훈련된 비전-언어 모델(pre-trained vision-language model)을 기반으로 하며, 학습 가능한 프롬프트 디자인(learnable prompt design)과 균형 잡힌 목표(balanced objective)를 포함하고 있습니다. 이 방식은 객체 분류(object classification)에서의 광범위한 지식을 유지하면서 딥페이크 감지를 위한 일반화를 향상시키기 위해 설계되었습니다.

- **Performance Highlights**: PoundNet은 10개의 공개 대규모 딥페이크 데이터셋에서 평가되었으며, 기존의 최첨단 방법들에 비해 약 19%의 성능 향상을 달성했습니다. 추가적으로, 객체 분류 작업에서 63%의 높은 성능을 유지하였고, 이는 다른 딥페이크 감지 모델들이 효과를 보이지 않는 영역입니다.



### Level Up Your Tutorials: VLMs for Game Tutorials Quality Assessmen (https://arxiv.org/abs/2408.08396)
Comments:
          Accepted at ECCV 2024 CV2 Workshop

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)을 활용하여 게임 튜토리얼의 품질을 자동으로 평가하는 혁신적인 솔루션을 제안합니다. 이 솔루션은 게임 튜토리얼의 프레임을 분석하고 인간의 인식을 시뮬레이션하기 위해 관련 질문에 자동으로 답변합니다. 이를 통해 개발자에게 즉각적인 피드백을 제공하고, 사용자 경험을 향상시킬 수 있습니다.

- **Technical Details**: 본 연구에서는 자동화된 게임 테스트 솔루션을 제안하며, VLMs를 활용하여 게임 튜토리얼의 프레임 분석 및 그에 대한 질문 대답을 통해 인간의 인식을 모방합니다. 이 과정은 개발자가 정의한 기대 결과와 비교하여 튜토리얼의 명확성과 이해도를 평가하는 방식으로 진행됩니다. 또한 다양한 오픈소스 및 폐쇄형 최신 모델들을 벤치마킹 하였습니다.

- **Performance Highlights**: 이 솔루션은 전통적인 테스트 방법보다 훨씬 더 빠르고 효율적인 평가 과정을 가능하게 하며, 문서화된 비디오와 주석이 달린 프레임을 개발자와 공유하여, 사용자 경험을 개선하는 데 기여하는 방법으로 개발 초기 단계의 시간을 단축시키는 데 효과적입니다.



### Pre-processing and Compression: Understanding Hidden Representation Refinement Across Imaging Domains via Intrinsic Dimension (https://arxiv.org/abs/2408.08381)
- **What's New**: 본 연구는 신경망의 학습된 표현의 내재 차원(intrinsic dimension, ID)이 여러 레이어를 거치며 어떻게 변하는지를 분석하고, 자연 이미지와 의료 이미지 도메인 간의 차이를 강조합니다.

- **Technical Details**: 연구는 11개의 자연 및 의료 이미지 데이터셋과 6개의 네트워크 아키텍처를 기반으로 합니다. 특히, 의료 이미지 모델은 네트워크에서 표현 ID가 더 빨리 정점을 찍는 경향이 있습니다. 이는 의료 이미지를 분석하기 위해 사용되는 이미지 특징들이 더 낮은 차원으로 압축되어야 함을 의미합니다.

- **Performance Highlights**: 연구 결과에 따르면, 학습된 표현의 ID 정점과 입력 데이터의 내재 차원 사이에는 강한 상관관계가 있으며, 이는 모델이 학습한 표현의 내재 정보 내용이 훈련 데이터에 의해 영향을 받는다는 것을 나타냅니다.



### 5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks (https://arxiv.org/abs/2408.08345)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2311.15010

- **What's New**: 이번 논문에서는 Multi-cognitive Visual Adapter(Mona) tuning이라는 새로운 어댑터 기반 튜닝 방법을 제안합니다. 이 방법은 기존의 delta-tuning 방식보다 더 효율적인 시각적 작업 전이 학습을 가능하게 합니다.

- **Technical Details**: Mona tuning은 여러 가지 시각 친화적 필터를 도입하여 시각 신호를 처리하는 능력을 향상시키며, 스케일 정규화 레이어(scaled normalization layer)를 추가하여 입력 특성의 분포를 조절합니다. 기존의 어댑터 방식과 달리, Mona는 2D convolutional operations를 활용하여 시각적 지식을 효과적으로 전이할 수 있도록 설계되었습니다.

- **Performance Highlights**: Mona tuning은 COCO, ADE20K, Pascal VOC, DOTA/STAR 및 여러 일반 데이터셋에서 인스턴스 분할, 객체 탐지 및 시맨틱 분할 작업을 포함한 다수의 전형적인 시각 작업에서 전통적인 full fine-tuning을 초월함을 입증했습니다. COCO 데이터셋에서 1% 성능 향상을 기록하는 등, Mona tuning은 다양한 작업에서 유일하게 full fine-tuning을 초과하는 delta-tuning 방법입니다.



### TurboEdit: Instant text-based image editing (https://arxiv.org/abs/2408.08332)
Comments:
          Accepted to European Conference on Computer Vision (ECCV), 2024. Project page: this https URL

- **What's New**: 본 연구에서는 소수 단계로 이루어진 diffusion 모델을 바탕으로 이미지의 정확한 복원과 분리된 이미지 편집의 문제를 해결합니다. 입력 이미지와 이전 단계의 복원 이미지를 조건으로 하는 인코더 기반의 반복적인 복원 기법을 도입하였습니다.

- **Technical Details**: TurboEdit는 입력 이미지를 복원하기 위해 노이즈를 예측하는 인버전 네트워크를 사용하며, 이는 이전 단계의 복원 이미지를 기반으로 반복적으로 수정하여 정확한 복원을 달성합니다. 또한, 텍스트 프롬프트의 속성을 변화시키는 방식으로 이미지 속성을 수정합니다. 이 과정에서 8개의 기능 평가(NFE)만으로 이미지를 복원하고, 편집 시에는 4개의 NFE만을 필요로 합니다.

- **Performance Highlights**: TurboEdit는 기존의 다단계 diffusion 모델 기반의 기술들에 비해 속도가 현저히 빠르며(< 0.5초), 텍스트와 이미지의 정렬 및 배경 보존 성능 또한 뛰어납니다. 이는 주로 수정 요청과 관련된 텍스트 프롬프트를 바탕으로 분리된 편집이 가능하다는 점에서 새로운 기준을 제시합니다.



### Segment Anything for Videos: A Systematic Survey (https://arxiv.org/abs/2408.08315)
Comments:
this https URL

- **What's New**: 최근 Foundation 모델이 이미지 처리 및 비디오 도메인에서 기존 Paradigm에 도전하고 있으며, Segment Anything 모델(SAM)의 발전이 특히 주목받고 있습니다. SAM 2의 출시로 인해 Promptable Visual Segmentation에 대한 연구 열풍이 다시 불고 있습니다.

- **Technical Details**: 본 논문에서는 SAM의 비디오 응용에 대한 체계적인 리뷰를 제공하며, 비디오 이해(video understanding), 비디오 생성(video generation), 비디오 편집(video editing)의 세 가지 주요 분야로 기존 연구를 분류하여 장단점을 분석합니다. SAM은 11백만 이미지에 대해 10억 개 이상의 마스크로 훈련되어 다양한 프롬프트 기반으로 고품질 세그멘테이션을 가능하게 하며, 특히 SAM 2는 Streaming memory가 통합된 Transformer 프레임워크로 비디오 세그멘테이션 성능을 크게 향상시킨 사례입니다.

- **Performance Highlights**: SAM 기반 방법이 현재 상태의 최첨단 방법(SOTA)과 비교하여 대표적인 벤치마크에서 놀라운 성능을 보여준다는 점을 강조하며, 기존 방법의 장점과 한계에 대한 깊이 있는 분석을 제공합니다.



### HistoGym: A Reinforcement Learning Environment for Histopathological Image Analysis (https://arxiv.org/abs/2408.08847)
- **What's New**: HistoGym은 병리학적 이미지 분석을 위한 개방형 강화 학습 환경으로, 의사의 진단 프로세스를 모델링하여 전체 슬라이드 이미지(Whole Slide Images, WSI) 진단을 촉진하는 것을 목표로 합니다. 이 환경은 의사들이 조직 샘플을 진단하는 실제 방법을 모방하며, 다양한 임상 작업을 위한 통합된 프레임워크를 제공합니다.

- **Technical Details**: HistoGym은 OpenAI Gym API를 따르는 환경으로, WSIs의 피라미드 기능을 활용하여 암의 탐지와 분류 같은 다양한 임상 작업을 지원합니다. 이 환경은 패치 기반 기법의 한계를 극복하고, 다양한 장기와 암에 대한 다양한 시나리오를 제공합니다. 에이전트는 자신이 보고 있는 필드를 제어하고 진단을 배우며, 이 과정에서 행동, 관찰, 보상 사양이 맞춤화되어 있습니다.

- **Performance Highlights**: HistoGym에서는 최신 알고리즘을 공개된 다기관 데이터셋에서 평가하여, 향후 연구를 위한 포괄적인 기준 결과를 제공합니다. 에이전트는 조직에서 세포 수준까지 다양한 스케일에서 종양 영역을 식별하도록 학습하며, 실질적인 결과와 함께 시나리오를 다룹니다.



### Assessing Generalization Capabilities of Malaria Diagnostic Models from Thin Blood Smears (https://arxiv.org/abs/2408.08792)
Comments:
          MICCAI 2024 AMAI Workshop, Accepted for presentation, Submitted Manuscript Version, 10 pages

- **What's New**: 이 연구는 말라리아 진단을 위한 CAD (Computer-Aided Diagnosis) 모델의 일반화 능력을 평가하며, 다양한 임상 환경에서의 효용성을 더욱 개선하기 위해 사이트별 데이터의 중요성을 강조합니다.

- **Technical Details**: 본 연구에서는 NIH의 공개 데이터셋과 세 개의 내부 데이터셋을 사용하여 Yolov5 모델의 일반화 능력을 평가합니다. 목적은 Plasmodium 기생충을 감지하는 것이며, 두 단계의 자동 진단 프로세스를 수행합니다: 첫 번째로 모든 적혈구를 탐지하고, 두 번째로 감염된 적혈구를 식별합니다.

- **Performance Highlights**: 사이트 특화 데이터를 포함했을 때 모델 성능이 향상되는 결과를 보여주었으며, 이는 다양한 임상 환경에서 널리 활용될 가능성을 제시합니다. 이 연구는 일반화 능력 평가와 모델의 적응성을 높이는 여러 전략을 제안합니다.



### A Disease-Specific Foundation Model Using Over 100K Fundus Images: Release and Validation for Abnormality and Multi-Disease Classification on Downstream Tasks (https://arxiv.org/abs/2408.08790)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구에서는 Fundus-Specific Pretrained Model(이미지+Fundus)이라는 새로운 감독 학습 인공지능 모델을 개발하였습니다. 이 모델은 fundus 이미지에서 이상을 탐지하도록 훈련되었으며, 57,803 개의 이미지를 활용하여 다양한 다운스트림 작업에서 우수한 성능을 보였습니다.

- **Technical Details**: 연구에 사용된 데이터는 서울대학교병원에서 수집된 113,645개의 fundus 이미지입니다. 이 모델은 두 가지 유형의 질병 특정 기초 모델을 개발했으며, 첫 번째는 fundus 이미지 데이터로 정의된 'Fundus' 모델이고, 두 번째는 대규모 비의료 데이터셋(ImageNet-1k)에서 훈련된 후 fundus 이미지 데이터셋으로 재훈련된 'ImageNet + Fundus' 모델입니다. 모델은 4개의 NVIDIA A100 GPU를 사용하여 PyTorch 기반으로 훈련되었으며, Adam 옵티마이저와 다양한 데이터 증강 기법을 포함하여 모델을 최적화했습니다.

- **Performance Highlights**: 모델은 다양한 다운스트림 작업에서 5-fold 교차 검증을 통해 평가되었으며, 모든 작업에서 AUC가 주요 성능 지표로 사용되었습니다. 개발된 이미지+Fundus 모델은 기존의 일반 모델보다 질병을 보다 정밀하게 탐지하며, 데이터 수를 줄이면서도 모델 성능을 개선할 수 있는 뚜렷한 장점을 보여주었습니다.



### Multi-task Learning Approach for Intracranial Hemorrhage Prognosis (https://arxiv.org/abs/2408.08784)
Comments:
          16 pages

- **What's New**: 이 연구는 뇌내출혈(intracranial hemorrhage, ICH)의 예후를 개선하기 위한 새로운 접근 방식을 제안합니다. 이미지 기반 예후 예측의 강력한 특징 표현(feature representation)을 학습하여 임상(linical) 및 인구통계학적(demographic) 변수들과의 상관관계를 강화합니다.

- **Technical Details**: 3D 멀티태스크 이미지 모델을 개발하여 Glasgow Coma Scale (GCS) 및 나이(age)를 예측합니다. 본 연구는 특정 임상 변수와 CT 뇌 스캔 이미지를 직접 연결하고, 이미지 모델의 학습 과정에서 GCS와 나이를 이진 및 서열형 변수로 인코딩하여 예후 정확도를 높였습니다.

- **Performance Highlights**: 본 연구의 방법은 기존의 최첨단 이미지 모델보다 우수한 성능을 보였으며, 4명의 인증된 신경영상의사(neuroradiologist)와 비교해도 높은 예후 성능을 입증했습니다. 또한, 해석 가능성(saliency maps)을 통해 모델의 예측 결과를 시각적으로 평가하였습니다.



### MicroSSIM: Improved Structural Similarity for Comparing Microscopy Data (https://arxiv.org/abs/2408.08747)
Comments:
          Accepted at BIC workshop, ECCV 24

- **What's New**: 이 논문에서는 기존의 Structural Similarity Index (SSIM) 측정 방식이 마이크로스코피 데이터에 적합하지 않음을 보여주고, 이를 개선하기 위한 𝕄⁢icroSSIM을 제안합니다. 𝕄⁢icroSSIM은 고유한 마이크로스코피 데이터의 특성을 반영하여 denoising 성능을 평가하는 새로운 척도입니다.

- **Technical Details**: 논문에서는 SSIM이 마이크로스코프에서 획득한 저신호대잡음비(Low-SNR) 이미지는 높은 신호대잡음비(High-SNR) 이미지와 비교할 때 의도치 않은 행동을 보인다고 언급합니다. 이 문제를 해결하기 위해 𝕄⁢icroSSIM을 도입하였으며, 이는 SSIM의 변형으로, 픽셀 강도의 차이를 처리할 수 있는 기능을 가지고 있습니다.

- **Performance Highlights**: 𝕄⁢icroSSIM은 unsupervised denoising 및 image splitting 작업에서 효과적으로 성능을 평가할 수 있으며, 기존의 SSIM 및 MS-SSIM 기반 척도의 한계를 극복하는 방식으로 차별성을 지닙니다.



### A lifted Bregman strategy for training unfolded proximal neural network Gaussian denoisers (https://arxiv.org/abs/2408.08742)
Comments:
          2024 IEEE International Workshop on Machine Learning for Signal Processing, Sept. 22--25, 2024, London, UK

- **What's New**: 이 논문은 unfolded proximal neural networks (PNNs)의 새로운 훈련 방식을 제안합니다. 이 방식은 Bregman 거리(Bregman distances)를 기반으로 하여 PNN의 학습 문제를 효율적으로 해결할 수 있는 특별한 계산 전략을 제공합니다.

- **Technical Details**: 연구자들은 결정론적 미니 배치 블록 좌표 전방-후방 방법을 활용하였습니다. 이 방법은 전통적인 back-propagation 방법을 넘어, 각 레이어에서 주변 최적화(proximal optimization) 알고리즘을 고정된 수의 반복으로 펼쳐서 설계된 신경망 구조를 사용합니다.

- **Performance Highlights**: PNN은 이미지 노이즈 제거(image denoising) 작업에서 효율적인 성능을 보여주며, 전통적인 딥러닝 기법에 비해 더 강력한 안정성을 나타냅니다.



### LLM-PCGC: Large Language Model-based Point Cloud Geometry Compression (https://arxiv.org/abs/2408.08682)
- **What's New**: 이 연구는 대형 언어 모델(LLM)을 기반으로 한 점 구름 기하학 압축(LLM-PCGC) 방법을 제안하며, 이는 텍스트 설명 없이 점 구름 구조를 이해하고 압축하는 첫 번째 구조입니다.

- **Technical Details**: LLM-PCGC 방법은 클러스터링, K-tree 구조, 토큰 매핑 불변성(token mapping invariance), 그리고 저랭크 적응(LoRA) 기법을 활용하여 LLM을 점 구름의 압축기/생성기로 변환합니다. 입력된 3D 점 구름을 클러스터링한 후, 각 클러스터는 병렬 처리되며, 좌표 정규화, K-tree 조직, 트리 구조의 평탄화 및 청크화를 포함합니다.

- **Performance Highlights**: 실험 결과, LLM-PCGC는 MPEG 기초 점 구름 압축(G-PCC) 소프트웨어에 비해 -40.213% 비트 전송률 감소를 달성하며, 최첨단 학습 기반 방법에 비해 -2.267% 비트 전송률 감소를 기록하여 기존 방법을 월등히 초월하는 성능을 보였습니다.



### Towards Physical World Backdoor Attacks against Skeleton Action Recognition (https://arxiv.org/abs/2408.08671)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이번 논문은 Skeleton Action Recognition (SAR) 시스템에 대한 물리적 백도어 공격(Backdoor Attacks)을 처음으로 탐구합니다. 이는 최근의 SAR 모델이 가해지는 공격의 보안 문제를 해결하고자 하는 새로운 시도를 포함하고 있습니다.

- **Technical Details**: 물리적 백도어 공격(Physical Skeleton Backdoor Attacks, PSBA)을 소개하며, 이는 기존의 스켈레톤 데이터에 감지되지 않도록 조작된 드문 행동을 트리거(Trigger)로 삽입하는 혁신적인 방법을 활용합니다. 이러한 조작된 데이터를 최소한으로 훈련 세트에 포함시켜, 트리거 행동이 존재할 경우 모든 스켈레톤 시퀀스가 목표 클래스(target class)로 잘못 분류되도록 합니다. 또한, PSBA의 내구성을 검증하기 위해 여러 데이터셋, 오염 비율(poisoning ratios), 모델 아키텍처(model architectures)에서 효율성을 입증하였습니다.

- **Performance Highlights**: PSBA의 성능은 두 가지 정량적 메트릭을 사용해 평가되었으며, 세 가지 다른 백도어 방어(backdoor defenses)에 대해 저항력을 테스트했습니다. Kinect V2 카메라를 사용하여 실제 세계에서 인간 행동의 데이터셋을 수집하였고, 이를 통해 물리적 공격 시나리오를 모방하여 제안된 공격의 효과성을 입증하였습니다.



### Modeling the Neonatal Brain Development Using Implicit Neural Representations (https://arxiv.org/abs/2408.08647)
Comments:
          Preprint, Accepted for PRIME MICCAI 2024

- **What's New**: 이 연구에서는 임신의 3분기에 발생하는 신생아 뇌의 발달을 모델링하기 위해 MR 이미지를 기반으로 하는 새로운 접근법을 제안합니다. 특히, Subject Specific Latent Vectors (SSL) 및 Stochastic Global Latent Augmentation (SGLA) 두 가지 방법으로 개인의 나이와 정체성을 분리하여 예측 모델의 성능을 향상시키고자 합니다.

- **Technical Details**: Implicit Neural Representation (INR)을 사용하여 2D 및 3D 이미지를 다양한 시점에서 예측합니다. 연구에서는 개발된 human connectome project (dHCP)의 미숙아 및 정기 출생 신생아의 MR 이미지를 사용합니다. SSL과 SGLA 방법을 통해 latent space에서 나이를 정체성과 분리하여 주제별 개발 프로세스를 모델링합니다.

- **Performance Highlights**: 제안된 모델은 나이에 조건화된 denoising diffusion model과 비교하여 개선된 성능을 보이며, 메모리 효율적인 방식으로 3D 데이터를 처리할 수 있습니다.



### A Survey on Benchmarks of Multimodal Large Language Models (https://arxiv.org/abs/2408.08632)
- **What's New**: 이번 논문은 다중 모달 대규모 언어 모델(Multimodal Large Language Models, MLLMs)의 180개의 벤치마크 및 평가를 포괄적으로 검토하며, 이러한 모델들이 다양한 응용 분야에서 가진 성능을 다각도로 평가합니다. 특히, 인지 및 추론, 특정 도메인, 주요 기능과 다양한 모달리티에서의 능력에 중점을 두고 있습니다.

- **Technical Details**: MLLM은 비쥬얼 질문 응답(Visual Question Answering, VQA) 및 특정 도메인 과제를 포함한 다양한 응용 분야에서 뛰어난 성능을 보여줍니다. MLLM의 성능을 객관적으로 비교하고 탐색하기 위해, 영향을 미치는 다섯 가지 주요 분야를 통해 최근 논문을 분석합니다. 이러한 분야는 지각 및 이해, 인지 및 추론, 특정 도메인, 주요 기능 및 기타 모달리티로 구성됩니다. 또한, MLLMs의 동작을 지원하는 세 가지 주요 모듈인 비쥬얼 인코더, 언어 모델, 비쥬얼-언어 프로젝터에 대한 내용을 다룹니다.

- **Performance Highlights**: 최신 벤치마크에 따르면 OpenAI의 GPT-4와 Google의 Gemini는 83개의 평가 기준에서 베스트 쿼터를 기록했으며, 이러한 성과는 다중 모달 기능에서의 효율성을 높이는 데 기여하고 있습니다. 본 논문은 현재 MLLM 평가 방법의 한계를 지적하고, 향후 발전 방향에 대한 논의를 포함합니다.



### Reference-free Axial Super-resolution of 3D Microscopy Images using Implicit Neural Representation with a 2D Diffusion Prior (https://arxiv.org/abs/2408.08616)
Comments:
          MICCAI2024 accepted

- **What's New**: 이 논문은 3D 마이크로스코피 (microscopy) 이미지의 재구성을 위한 새로운 방법론을 제안합니다. 기존의 3D 슈퍼 레졸루션 (super-resolution) 모델들이 갖는 데이터 부족 문제를 해결하기 위해, 독립적인 축 슬라이스(axial slices)를 최적화하여 3D 일관성을 유지하는 새로운 프레임워크를 사용합니다.

- **Technical Details**: 제안된 방법은 Implicit Neural Representation (INR) 기반으로, 고해상도 횡 단면(lateral slices)을 사용하여 저해상도 축 슬라이스에서 연속적인 볼륨 표현을 최적화합니다. 이는 주로 score distillation sampling을 활용하여 2D diffusion prior을 통합함으로써 이루어집니다.

- **Performance Highlights**: 실제 및 합성 아나수트로픽 마이크로스코피 이미지에서 다양한 실험을 통해, 제안된 접근법이 기존 최첨단 재구성 방법들을 초월하는 성능을 보이는 것으로 나타났습니다.



### S-RAF: A Simulation-Based Robustness Assessment Framework for Responsible Autonomous Driving (https://arxiv.org/abs/2408.08584)
- **What's New**: AI 기술이 발전함에 따라 자율주행(AI agents) 시스템의 강건성 및 안전성을 보장하는 것이 중요해졌습니다. 이 연구에서는 다양한 조건에서 자율주행 시스템을 엄격히 평가할 수 있는 Simulation-Based Robustness Assessment Framework (S-RAF)를 도입하여 안전 인증 프로세스를 간소화할 수 있는 데 기여하고자 합니다.

- **Technical Details**: S-RAF는 CARLA Driving Simulator를 활용하여 센서 오류, 환경 변화 및 복잡한 교통 상황 등 다양한 조건에서 자율주행 에이전트의 강건성을 평가합니다. 이 프레임워크는 강건성(robustness)과 탄소 배출(carbon emissions) 등 다른 안전 관련 요소들과의 관계를 정량화합니다.

- **Performance Highlights**: S-RAF를 통해 개발자와 이해관계자들이 안전하고 책임 있는 자율주행 에이전트를 구축할 수 있도록 돕습니다. 또한, 현실 세계에서 안전하게 테스트하기 어려운 엣지 케이스들을 탐색할 수 있으며, 테스트 비용 감소와 같은 значительные 장점을 제공합니다.



### S$^3$Attention: Improving Long Sequence Attention with Smoothed Skeleton Sketching (https://arxiv.org/abs/2408.08567)
- **What's New**: S$^3$Attention이라는 새로운 주의 구조를 제안하며, 이것은 정보를 유지하면서 계산 비용을 줄이는 균형을 효과적으로 관리합니다.

- **Technical Details**: S$^3$Attention은 두 가지 메커니즘을 활용하여 노이즈의 영향을 최소화하며, 시퀀스의 길이에 대한 선형 복잡성을 유지합니다: (1) 긴 시퀀스에서 정보를 혼합하는 스무딩 블록 (2) 입력 행렬에서 열과 행을 동시 선택하는 행렬 스케칭 방법.

- **Performance Highlights**: Long Range Arena와 6개의 시계열 예측에 대한 광범위한 연구를 통해 S$^3$Attention이 기존의 주의 모델보다 현저히 뛰어난 성능을 보임을 확인했습니다.



### Detection and tracking of MAVs using a LiDAR with rosette scanning pattern (https://arxiv.org/abs/2408.08555)
- **What's New**: 이 연구에서는 저비용 로제타 스캐닝 LiDAR를 활용하여 상업적인 마이크로 항공기(MAV)의 탐지 및 추적을 위한 새로운 방법을 제안합니다. 이 방법은 정적 배경을 캡처한 후 입자 필터를 사용하여 목표를 탐지하고 이동 위치를 추적합니다.

- **Technical Details**: 로제타 스캐닝 LiDAR는 리즐리 프리즘을 사용하여 비주기적인 로제타 형태로 레이저 빔이 움직여 전체 시야를 커버합니다. 이 센서는 목표가 검출 되는 중심에서 더 높은 밀도의 3D 포인트를 측정할 수 있도록 하며, 제안된 방법은 indoor 및 outdoor 테스트에서 검증되었습니다.

- **Performance Highlights**: 제안된 방법은 3D 점 밀도 증가, 높은 추적 정확성 및 안정적인 재탐지 시간을 제공하며, 기존의 원형 LiDAR를 사용하는 방법과 비교하여 최대 탐지 거리가 약 80% 증가했습니다. 또한 반사점으로부터의 반환 포인트 수가 여러 배로 증가했습니다.



### DFT-Based Adversarial Attack Detection in MRI Brain Imaging: Enhancing Diagnostic Accuracy in Alzheimer's Case Studies (https://arxiv.org/abs/2408.08489)
Comments:
          10 pages, 4 figures, conference

- **What's New**: 본 연구는 알츠하이머병 이미지에 대한 적대적 공격(adversarial attack)을 분석하고, 이러한 공격에 대응하기 위한 방어 방법(defensive method)을 제안합니다. 특히, 주파수 영역(transformations in frequency domain)을 활용한 공격 방식을 검토하였으며, 기존의 잘 알려진 공격 방식과의 비교를 통해 새로운 통찰력을 제공합니다.

- **Technical Details**: 우리는 CNN(convolutional neural network) 기반의 오토인코더(autoencoder) 아키텍처를 사용하여, 2차원 푸리에 변환(two-dimensional Fourier transform)을 이용한 이미지 탐지를 수행합니다. 이러한 접근 방식은 주파수 도메인에서 적대적 공격을 효과적으로 탐지하고 방어하는 데 도움을 줍니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 탐지 및 방어 메커니즘이 여러 적대적 공격에 대해 효과적으로 대응하며, 딥 뉴럴 네트워크(deep neural networks)의 강건성(robustness)을 향상시키는 것으로 나타났습니다.



### JPEG-LM: LLMs as Image Generators with Canonical Codec Representations (https://arxiv.org/abs/2408.08459)
- **What's New**: 최근 이미지 및 비디오 생성 분야에서 오토 리그레시브(autoregressive) LLM 아키텍처를 채택하여 다중 모달 시스템에 쉽게 통합할 수 있는 방법을 제시했습니다. 본 연구에서는 조작이 용이한 JPEG 및 AVC/H.264와 같은 공인된 코덱을 활용하여 이미지와 비디오를 생성하는 방법을 제안하고, 이를 통해 기존의 벡터 양자화(vector quantization) 방법보다 31% 더 나은 FID 결과를 얻었습니다.

- **Technical Details**: 본 연구는 Llama-2 아키텍처를 기반으로 하여 Jpeg-LM 및 Avc-LM이라는 두 개의 7B 모델을 프리트레인(pretrained) 합니다. 이들은 각각 256x256 크기의 이미지와 256x144 크기의 15프레임 비디오를 생성하며, 평균 컨텍스트 길이는 5K 및 15K입니다. 이미지를 JPEG 형식으로, 비디오를 AVC 형식으로 직접 압축 파일 바이트로 출력합니다.

- **Performance Highlights**: Jpeg-LM은 강력한 VQ 기반 모델보다 생성 품질이 우수하며 FID에서 평균 31%의 감소를 보였습니다. Avc-LM 또한 현실적인 움직임을 가진 비디오를 생성할 수 있음을 보여주었고, 비전 분야에서의 장기 요소에 대한 인식 능력이 특히 뛰어난 것으로 분석되었습니다.



### Efficient Data-Sketches and Fine-Tuning for Early Detection of Distributional Drift in Medical Imaging (https://arxiv.org/abs/2408.08456)
- **What's New**: 이번 논문은 CT 스캔 의료 이미지에서의 distributional drift(배포 드리프트) 감지를 위한 정교하고 민감한 접근 방식을 제시합니다. 데이터 스케치(data-sketching)와 파인튜닝(fine-tuning) 기술을 활용하여, 모델의 정확도와 효율성을 극대화하는 방법을 소개합니다.

- **Technical Details**: 연구에서는 Vision Transformer(ViT) 모델을 파인튜닝하여 유방암 이미지로부터 유의미한 특징을 추출하였으며, 사전 훈련된 모델을 기반으로 커스터마이즈한 아키텍처를 사용하여 특징 추출을 수행했습니다. 데이터 스케칭 기법으로는 MinHash를 활용하여 이미지 데이터를 처리하고, Kolmogorov-Smirnov 및 cosine similarity 점수를 통해 드리프트를 감지합니다. 또한 1%의 Salt-and-Pepper와 Speckle 노이즈에도 높은 민감성을 보였습니다.

- **Performance Highlights**: 모델의 정확도가 99.11%로 향상되었으며, 데이터 스케치 기법을 통한 새로운 요소들의 심사에서 50%에서 100%로 비슷한 데이터셋 간의 유사성 점수가 크게 개선되었습니다. 결과적으로, 제안된 방법은 동적인 임상 환경에서 진단 모델의 정확성을 유지하기 위한 확장 가능하고 신뢰성 높은 솔루션을 제공합니다.



### Predictive uncertainty estimation in deep learning for lung carcinoma classification in digital pathology under real dataset shifts (https://arxiv.org/abs/2408.08432)
Comments:
          17 pages, 2 figures, 5 tables

- **What's New**: 본 논문은 딥 러닝 기반의 진단 의사결정 시스템에서 예측 불확실성(uncertainty) 추정이 강건성과 성능에 미치는 영향을 평가합니다. 특히, 폐선암(lung carcinoma) 분류 문제에 대해 다양한 데이터 분포 이동(distribution shift) 시나리오를 살펴보았습니다.

- **Technical Details**: 주요 방법으로 Monte Carlo dropout, deep ensemble, few-shot learning을 사용하여 전국적인 곤충 관찰을 통해 예측 불확실성을 개선하고자 했습니다. 연구는 내부 및 외부의 데이터 분포 이동을 포함한 다양한 시나리오에서 이루어졌습니다.

- **Performance Highlights**: 본 연구는 폐 선암의 다양한 하위 유형과 특성 분석 데이터를 포함한 임상적으로 관련된 분포 이동 아래에서 제안된 방법들의 효과를 비교 분석하였습니다. 비교 결과, deep ensemble 방법이 가장 우수한 성능을 발휘하였고, 두 번째로 MC-dropout이 좋은 결과를 보였음을 확인했습니다.



### CT4D: Consistent Text-to-4D Generation with Animatable Meshes (https://arxiv.org/abs/2408.08342)
- **What's New**: 최근 텍스트-투-4D 생성 분야에서, 기존 영상 모델과 이미지 확산 모델의 통합을 통해 4D 콘텐츠 생성을 가능하게 하는 새로운 프레임워크 CT4D가 제안되었습니다. 이 프레임워크는 사용자 제공 프롬프트에 따라 일관된 4D 콘텐츠를 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: CT4D 프레임워크는 Generate-Refine-Animate (GRA) 알고리즘을 통해 텍스트 정렬된 메쉬를 안정적으로 생성합니다. 프레임워크는 메쉬를 여러 작은 영역으로 나누고 각 영역 내에서 일관성 있는 구동 함수를 적용하여 표면 연속성을 유지합니다. 이를 위해 프레임워크는 기하학 보존을 위해 경직 규제를 적용합니다.

- **Performance Highlights**: 실험 결과 CT4D 프레임워크는 기존의 텍스트-투-4D 기법들에 비해 프레임 간 일관성과 기하학 보존에서 우수한 성능을 보여 주며, 텍스처 수정 및 다중 객체 조합 능력을 자연스럽게 구현하는 것 또한 입증되었습니다.



### METR: Image Watermarking with Large Number of Unique Messages (https://arxiv.org/abs/2408.08340)
Comments:
          14 pages, 9 figures, code is available at this https URL

- **What's New**: 이번 논문에서는 이미지 생성의 품질을 향상시키는 Diffusion Models의 발전에 따라 새로운 수작업 알고리즘인 METR: Message Enhanced Tree-Ring을 제안합니다. 이 알고리즘은 생성 예술의 창작자를 명확히 식별할 수 있는 가능성을 제공합니다.

- **Technical Details**: METR은 Tree-Ring watermarking 알고리즘에 기반하여 설계되었으며, 공격 저항성과 이미지 품질을 저해하지 않으면서 여러 개별 메시지를 인코딩할 수 있게 합니다. 또한, METR++는 Latent Diffusion Model 아키텍처에 한정되지만 사실상 무한한 수의 고유 메시지를 삽입할 수 있도록 설계되었습니다.

- **Performance Highlights**: METR과 METR++는 공격에 대한 강건성과 많은 고유 메시지를 암호화할 수 있는 능력을 입증하며, 이미지 품질을 유지합니다. 이들은 실제 환경에서의 응용 가능성이 높습니다.



### Graph representations of 3D data for machine learning (https://arxiv.org/abs/2408.08336)
Comments:
          14 pages, 11 figures

- **What's New**: 이 연구는 3D 데이터를 분석에서 사용 가능한 조합적(combinatorial) 방법들에 대한 개요를 제공하며, 다양한 표현 방식의 장단점 및 표현 간 전환 방법에 대해 논의합니다. 또한 생명과학 및 산업 분야의 두 가지 구체적인 응용 사례를 제시합니다.

- **Technical Details**: 3D 데이터는 과학 및 산업에서 자연스럽게 발생하며, 생물영상(bioimaging), 분자 화학(molecular chemistry), 3D 모델링과 설계(plan) 등 다양한 분야를 포함합니다. 3D 데이터를 분석하는 가장 일반적인 방법은 voxel(3D 픽셀)의 표현을 사용하는 것이며, 이로 인해 계산 복잡도가 증가합니다. 본 논문은 mesh, point cloud, graph와 같은 경량 표현을 활용해 이러한 문제를 완화할 수 있는 가능성을 탐구합니다.

- **Performance Highlights**: 조합적 표현을 사용할 경우 모델의 예측에 대한 설명 가능성을 높일 수 있으며, 그래프 신경망(Graph Neural Networks, GNN)을 통해 3D 데이터를 효과적으로 처리할 수 있는 가능성이 있습니다. GNN은 스파스한 3D 데이터에 대해 뛰어난 확장성을 보이며, 2D 데이터와 거의 동일한 속도로 처리가 가능합니다.



### Can ChatGPT assist visually impaired people with micro-navigation? (https://arxiv.org/abs/2408.08321)
- **What's New**: 이 연구에서는 시각 장애인을 위한 마이크로 네비게이션(마이크로 네비게이션) 지원의 가능성을 ChatGPT를 활용하여 탐구했습니다.

- **Technical Details**: 연구팀은 113개의 장면 이미지와 그에 대한 텍스트 설명으로 구성된 실내외 마이크로 네비게이션 시나리오의 테스트 세트를 만들었습니다. 총 412개의 길찾기 질의(way-finding queries)와 그에 대한 예상 응답이 컴파일되었습니다. 다양한 조건에서 ChatGPT 4o의 민감도(Sensitivity, SEN)와 특이도(Specificity, SPE)를 평가했습니다.

- **Performance Highlights**: 기본 상태의 ChatGPT 4o는 장면 이미지를 입력할 경우 SEN 64.8%, SPE 75.9%의 성능을 보였습니다. 장면에 대한 사람의 설명을 입력으로 제공했을 때 SEN과 SPE는 각각 평균 17%와 16% 상승하였습니다. 그러나 ChatGPT 4o는 여전히 일부 경우에 마이크로 네비게이션 안내를 올바르게 제공하지 못하며, 이는 장면 이해가 네비게이션 목적에 맞게 최적화되지 않았기 때문으로 보입니다.



### ChemVLM: Exploring the Power of Multimodal Large Language Models in Chemistry Area (https://arxiv.org/abs/2408.07246)
Comments:
          11 pages, updated version

- **What's New**: 이 논문에서는 화학 분야를 위한 새로운 다중 모달 대규모 언어 모델인 ChemVLM을 소개합니다. ChemVLM은 화학 응용에 특화된 오픈 소스 모델로, 텍스트와 비주얼 화학 정보를 통합하여 더 나은 이해와 분석 능력을 제공합니다.

- **Technical Details**: ChemVLM은 Vision Transformer (ViT)와 대규모 화학 데이터로 학습된 ChemLLM을 결합한 구조입니다. 두 단계의 감독하에 미세 조정(Staged Supervised Fine-tuning) 방식을 통해 화학 작업을 최적화하는데 중점을 둡니다. 또한 ChemOCR, MMCR-Bench, MMChemBench와 같은 3개의 전문 데이터 세트를 구축해 다양한 화학 작업을 평가합니다.

- **Performance Highlights**: ChemVLM은 여러 정량적 평가에서 기존 모델 대비 향상된 성능을 보였으며, 특히 화학 이미지 인식 및 다중 모달 화학 추론(MMCR) 작업에서 뛰어난 결과를 보여 SOTA(State-of-the-Art) 성능을 달성했습니다.



New uploads on arXiv(cs.AI)

### GeoTransformer: Enhancing Urban Forecasting with Geospatial Attention Mechanisms (https://arxiv.org/abs/2408.08852)
- **What's New**: GeoTransformer는 Transformer 아키텍처와 지리 통계적 사전(geospatial statistics prior)을 결합하여 도시 예측 모델의 성능을 향상시키는 새로운 구조입니다.

- **Technical Details**: 이 모델은 혁신적인 지리적 주의(attention) 메커니즘을 사용하여 광범위한 도시 정보와 공간적 의존성을 통합하여 예측 모델로 통합합니다. 특정 지역과 주변 지역 간의 공간 가중 주의 점수를 계산하여 예측에 활용합니다.

- **Performance Highlights**: GDP 및 승차 공유 수요 예측 작업에 대한 광범위한 실험 결과, GeoTransformer는 기존 기준 모델을 현저히 초월하여 도시 예측 작업의 가능성을 입증합니다.



### Evaluating the Evaluator: Measuring LLMs' Adherence to Task Evaluation Instructions (https://arxiv.org/abs/2408.08781)
- **What's New**: 최근에 LLMs-as-a-judge가 자동 평가에 활용되는 새로운 접근 방식으로 주목받고 있습니다. 이 방법은 사람의 평가를 대체하여 LLM을 통해 품질 판단을 수행하며, RLHF로 훈련된 GPT4와 Llama3와 같은 최신 LLM이 인식하는 품질 기준에 대해 탐구합니다.

- **Technical Details**: 본 논문에서는 질적 평가 기준의 새로운 분류 체계를 제안합니다. 이 체계는 4개의 평가 카테고리(내용, 관련성, 완전성, 참여도)로 구성되어 있으며, 이를 통해 8개의 최첨단 벤치마크 데이터셋에서 테스트한 34개의 메트릭을 포함합니다. 또한, 다양한 LLM 계열을 사용하여 LLMs-as-a-judge의 효과를 체계적으로 평가하였습니다.

- **Performance Highlights**: 퍼플렉시티(perplexity)가 주어진 지침 없이 인공지능의 인간 수준의 판단과 더 나은 상관관계를 가지는 경우가 많았으며, 평가의 중요 영역에서는 LLM을 통해 간단한 평가가 성공적으로 이루어질 수 있음을 보여줍니다. 결과적으로, 간단한 모델 퍼플렉시티가 LLMs-as-a-judge보다 우수한 평가 대안일 수 있음을 시사합니다.



### Pessimistic Iterative Planning for Robust POMDPs (https://arxiv.org/abs/2408.08770)
- **What's New**: 본 논문에서는 robust POMDPs(강인 부분 관찰 마르코프 의사결정 과정)의 정책을 찾기 위해 PIP(비관적 반복 계획) 프레임워크를 제안합니다.

- **Technical Details**: PIP는 두 가지 주요 단계로 구성됩니다: (1) 불확실성 세트에서 worst-case 인스턴스를 선택하여 적대적인(non-robust) POMDP를 생성하고, (2) 해당 적대적인 POMDP를 위한 finite-state controller(FSC)를 계산합니다. rFSCNet 알고리즘에서는 RNN을 사용하여 FSC를 찾고, 이를 통해 robust 정책을 최적화합니다.

- **Performance Highlights**: 4개의 벤치마크 환경에서의 실험 평가 결과, 제안한 방법이 기존 방법보다 강인성을 향상시키고, 최신 robust POMDP 솔버와 경쟁력 있는 성능을 보여주었습니다.



### Symbolic Parameter Learning in Probabilistic Answer Set Programming (https://arxiv.org/abs/2408.08732)
Comments:
          The paper has been accepted at the ICLP2024 conference and is under consideration in Theory and Practice of Logic Programming (TPLP)

- **What's New**: 이번 논문에서는 Probabilistic Answer Set Programming 내에서의 매개변수 학습을 위한 두 가지 알고리즘을 제안합니다. 이들은 관측 결과의 확률을 최대화하기 위해 상징적 수식을 추출하는 방식을 기반으로 합니다.

- **Technical Details**: 첫 번째 알고리즘은 비선형 제약 최적화(nonlinear constrained optimization) 문제로 매개변수 학습을 설정하고 기존의 최적화 솔버를 사용하여 해결합니다. 두 번째는 Expectation Maximization (EM) 알고리즘을 구현한 방식입니다.

- **Performance Highlights**: 실험 결과, 제안된 제약 최적화 방법은 기존의 projected answer set enumeration 기반 접근법보다 해결 품질과 실행 시간을 포함하여 종종 더 빠르고 더 정확하게 나타납니다.



### NFDI4DSO: Towards a BFO Compliant Ontology for Data Scienc (https://arxiv.org/abs/2408.08698)
- **What's New**: 이 논문은 NFDI4DataScience (NFDI4DS) 프로젝트의 일환으로, Data Science (DS)와 Artificial Intelligence (AI)에서의 연구 데이터 접근성과 상호운용성을 향상시키기 위한 NFDI4DS Ontology를 소개합니다. 이 온톨로지는 FAIR (Findable, Accessible, Interoperable, and Reusable) 원칙에 따라 디지털 아티팩트를 연결하고, NFDI4DS 컨소시엄의 구조를 모델링합니다.

- **Technical Details**: NFDI4DSO는 NFDICore 온톨로지 위에 구축되었으며, 사용자 중심의 반복적 접근을 통해 개발되었습니다. NFDICore는 51개의 클래스, 55개의 객체 속성, 8개의 데이터 속성, 18개의 주석 속성을 포함하며, NFDI4DSO는 여기에 42개의 추가 클래스와 38개의 객체 속성을 포함합니다. 이 온톨로지는 BFO 및 기타 온톨로지 (예: schema.org)와 매핑되어 있습니다.

- **Performance Highlights**: NFDI4DSO는 NFDI4DS 지식 그래프 (NFDI4DS-KG)의 기초로 설계되었으며, 연구 정보 그래프 (RIG)와 연구 데이터 그래프 (RDG)를 포함합니다. RIG는 다양한 자원, 인물 및 기관에 대한 메타데이터를 포함하며, 현재 첫 번째 버전이 공개되어 접근 가능하고 검색할 수 있습니다.



### LLM-PCGC: Large Language Model-based Point Cloud Geometry Compression (https://arxiv.org/abs/2408.08682)
- **What's New**: 이 연구는 대형 언어 모델(LLM)을 기반으로 한 점 구름 기하학 압축(LLM-PCGC) 방법을 제안하며, 이는 텍스트 설명 없이 점 구름 구조를 이해하고 압축하는 첫 번째 구조입니다.

- **Technical Details**: LLM-PCGC 방법은 클러스터링, K-tree 구조, 토큰 매핑 불변성(token mapping invariance), 그리고 저랭크 적응(LoRA) 기법을 활용하여 LLM을 점 구름의 압축기/생성기로 변환합니다. 입력된 3D 점 구름을 클러스터링한 후, 각 클러스터는 병렬 처리되며, 좌표 정규화, K-tree 조직, 트리 구조의 평탄화 및 청크화를 포함합니다.

- **Performance Highlights**: 실험 결과, LLM-PCGC는 MPEG 기초 점 구름 압축(G-PCC) 소프트웨어에 비해 -40.213% 비트 전송률 감소를 달성하며, 최첨단 학습 기반 방법에 비해 -2.267% 비트 전송률 감소를 기록하여 기존 방법을 월등히 초월하는 성능을 보였습니다.



### Fine-tuning LLMs for Autonomous Spacecraft Control: A Case Study Using Kerbal Space Program (https://arxiv.org/abs/2408.08676)
Comments:
          ESA SPAICE Conference 2024. arXiv admin note: text overlap with arXiv:2404.00413

- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)을 자율 우주선 제어를 위한 방식으로 활용하고 있으며, Kerbal Space Program Differential Games(KSPDG)를 테스트 환경으로 사용하고 있습니다. 기존의 Reinforcement Learning(RL) 접근 방식의 한계를 극복하기 위해 LLM을 활용하여 우주선 제어의 가능성을 탐구하고 있습니다.

- **Technical Details**: LLM 기반 에이전트를 통해 실시간 텔레메트리를 텍스트 입력으로 변환하고 이를 처리하여 우주선의 제어 액션을 생성하는 방식으로 RL 루프를 설계하였습니다. 또한, LLaMA와 같은 개방형 모델을 채택하여 우리의 요구에 맞게 세밀한 조정을 가능하게 하였습니다.

- **Performance Highlights**: 이 연구의 에이전트는 KSPDG 챌린지에서 2위를 기록하였으며, 2024년 AIAA SciTech에서 라이브 데모로 발표되었습니다. LLM 기반 캡처가 우주 작업의 새로운 가능성을 열어주고 있습니다.



### Robust Stochastic Shortest-Path Planning via Risk-Sensitive Incremental Sampling (https://arxiv.org/abs/2408.08668)
Comments:
          Accepted for presentation at the 2024 IEEE Conference on Decision and Control (CDC)

- **What's New**: 이 논문에서는 리스크 인지 경로 계획(Risk-Aware Path Planning)의 중요성을 설명하며, 조건부 가치 손실(Conditional Value-at-Risk, CVaR)을 기반으로 한 RRT* 알고리즘의 변형인 RA-RRT* 알고리즘을 제안합니다. 이 알고리즘은 고위험 산업에서 발생하는 Stochastic Shortest-Path (SSP) 문제를 효과적으로 해결할 수 있도록 설계되었습니다.

- **Technical Details**: RA-RRT* 알고리즘은 샘플링 반복에서 각 경로 세그먼트의 CVaR를 최소화하는 노드를 선택하여 검색 트리를 확장합니다. 본 연구는 이론적 근거를 바탕으로 경로 계획의 모든 샘플링 단계에서 CVaR를 최적화함으로써, 무한히 샘플 사이즈에 수렴하는 최적 경로를 제공하는 방식으로 진행됩니다.

- **Performance Highlights**: 시뮬레이션 결과, RA-RRT* 알고리즘은 전통적인 RRT*보다 통계적 노이즈 변화에 훨씬 덜 민감하며, 경로의 견고성(robustness)을 입증하였습니다. 또한, 계산 시간과 메모리 요구사항은 비슷한 수준을 유지하며 처리 시간의 약간의 증가가 있긴 하지만, 이는 낮은 노이즈 민감성 및 실패율 감소로 상쇄되는 효과가 있음을 보였습니다.



### Understanding Enthymemes in Argument Maps: Bridging Argument Mining and Logic-based Argumentation (https://arxiv.org/abs/2408.08648)
Comments:
          Research note

- **What's New**: 이 논문은 논쟁 맵(argument map)과 그 안의 명제들 및 주장(claim) 사이의 관계를 추적하기 위한 새로운 방법론을 제안합니다. 기존의 논쟁 맵 분석 기술의 한계점을 극복하기 위해 고전 논리(Classical Logic)와 기본 논리(Default Logic)를 활용하여 텍스트의 명시적 정보와 암시적 정보를 형식적으로 표현하고자 합니다.

- **Technical Details**: 논문은 두 가지 주요 기술적 질문을 다룹니다. 첫 번째 질문은 논쟁 맵에서 명확한 정보(예: 전제(premise)와 주장)를 어떻게 고전 논리로 표현할 수 있는가입니다. 두 번째 질문은 암시적 정보(예: 결론을 도출하기 위해 필요한 배경 지식)를 어떻게 논리적으로 표현할 수 있는가입니다. 기본 논리를 사용하여 주장 사이의 지원(support) 및 공격(attack) 관계를 정의하고, 이를 통해 논쟁 맵의 각 노드를 논리적 주장으로 인스턴스화(instantiation)할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 접근 방식을 통해 사용자는 자동으로 생성된 논쟁 맵을 효과적으로 분석하고 비판할 수 있으며, 서로 다른 논쟁 맵을 비교할 수 있는 기반을 마련하게 됩니다. 이는 논쟁을 보다 논리적이고 체계적으로 접근함으로써, 인간의 인지 과정에서의 논리적 비약(nonevident jumps)을 줄이는 데 기여할 것입니다.



### Magazine Supply Optimization: a Case-study (https://arxiv.org/abs/2408.08637)
- **What's New**: 이번 논문에서는 프랑스의 20,000개 이상의 판매 지점(Point Of Sale, POS)에 대한 잡지 공급 최적화 솔루션인 AthenIA를 소개합니다. 이 솔루션은 재고 관리 문제와 수요 예측, 공급 최적화 및 비즈니스 규칙을 포함한 네 단계의 파이프라인으로 구성되어 있습니다.

- **Technical Details**: AthenIA는 군집 conformalized quantile regression 방법을 기반으로 하며, 이를 통해 도메인 전문가의 통찰력을 통합하여 다양한 제품 특성을 모델링합니다. 또한, 복잡한 비즈니스 비용 구조를 고려한 비즈니스 제약 조건 하에 수익을 최적화하는 독창적인 최적화 기술을 사용합니다.

- **Performance Highlights**: AthenIA는 잡지 출판사에게 유용한 도구로 입증되었으며, 출판사는 지속 가능한 실천을 통해 경제적 및 생태적 도전에 대응하고 있습니다. 이 최적화 솔루션은 비즈니스 네트워크의 공급과 수요를 효율적으로 조정함으로써 잉여 재고와 품절에 따른 손실을 최소화합니다.



### String Diagram of Optimal Transports (https://arxiv.org/abs/2408.08550)
Comments:
          Preprint, under review, 14 pages, 2 fugures, 1 table

- **What's New**: 최적 운송(optimal transport, OT)의 계층적 프레임워크인 문자열 다이어그램(string diagram) OTs를 제안합니다. 이 연구에서는 주어진 문자열 다이어그램에서 최소 운송 비용이 특정 임계값(threshold) 이상인지 여부를 입증하는 안전 문제(safety problem)를 다룹니다.

- **Technical Details**: 비용 행렬(cost matrix)을 구성하여 문자열 다이어그램 OTs에 대한 안전 문제를 단일 OT(monolithic OT) 문제로 축소(reduce)합니다. 이 새로운 축소는 두 가지 조합(composition), 즉 순차적 조합(sequential composition)과 병렬 조합(parallel composition)을 갖춘 비용 행렬의 대수적 구조(algebraic structure)를 활용합니다. 이를 통해 문자열 다이어그램 OTs에 대한 안전 문제를 해결하기 위한 새로운 알고리즘을 제공합니다.

- **Performance Highlights**: 실험을 통해 효율성(efficiency)과 성능(performance) 우위를 입증하였습니다.



### An Unsupervised Learning Framework Combined with Heuristics for the Maximum Minimal Cut Problem (https://arxiv.org/abs/2408.08484)
- **What's New**: 이 논문은 Maximum Minimal Cut Problem (MMCP)을 해결하기 위한 최초의 비지도 학습(unsupervised learning) 프레임워크와 휴리스틱(heuristics)을 결합한 연구로, 고품질 솔루션을 제공합니다.

- **Technical Details**: 제안하는 방법은 relaxation-plus-rounding 접근 방식에서 영감을 얻었으며, 그래프 신경망(graph neural networks)을 통해 완화된 솔루션을 매개변수화합니다. 각 솔루션은 적어도 하나의 스패닝 트리(spanning tree)와 관련이 있으며, 이 성질을 이용해 트리 변환(tree transformations)을 통해 솔루션 품질을 개선합니다.

- **Performance Highlights**: 우리는 제안한 프레임워크를 평가하기 위해 광범위한 실험을 수행하였으며, 두 가지 기존 기술과 비교하여 우리 방법의 우수성이 입증되었습니다.



### A theory of understanding for artificial intelligence: composability, catalysts, and learning (https://arxiv.org/abs/2408.08463)
Comments:
          13 pages, 3 figures

- **What's New**: 이번 연구는 인공지능(AI)에서 이해(understanding)의 개념을 분석하기 위한 새로운 프레임워크를 제안하고 있습니다. 이는 구성 가능성(composability)의 관점에서 이해를 정의하며, 주체가 객체를 이해하는 능력을 검증자(verifier)의 관점에서 만족스러운 출력으로 적절히 처리하는 능력으로 규명합니다. 이 프레임워크는 인공지능뿐만 아니라 비인간 동물, 제도 등 다양한 주체에 적용 가능합니다.

- **Technical Details**: 연구에서는 '촉매(catalysts)'라는 개념을 도입하여 입력을 최적화하여 출력 품질을 개선하는 방법을 제안합니다. 주체의 구조를 연구하고, 주체의 학습 능력을 내부 촉매를 구성하는 능력으로 간주하여 재조명합니다. 따라서 LLMs(대형 언어 모델)과 같은 시스템의 이해 능력을 분석하고, 이러한 시스템들이 일반 지능(general intelligence) 획득에 중요한 역할을 할 가능성을 조사합니다.

- **Performance Highlights**: LLMs는 자연어 이해(natural language understanding) 과제에서 인간보다 잘 수행하고 있으며, 이들 시스템들이 생성하는 출력이 스스로의 촉매 역할을 할 수 있는 모델들이 이론적으로 AI의 이해 능력 향상에 도움을 줄 수 있음을 보입니다. 이는 AI가 일반적인 문제 해결(universal problem solving)을 위한 개념을 생성하고 사용하는 경향을 향상시킬 수 있다는 것을 시사합니다.



### Automated Design of Agentic Systems (https://arxiv.org/abs/2408.08435)
Comments:
          Website: this https URL

- **What's New**: 기존의 일반-purpose 에이전트 개발에 있어 기초 모델(Foundation Models) 사용이 보편화되고 있지만, 저자들은 자동 에이전트 시스템 설계(Automated Design of Agentic Systems, ADAS)라는 새로운 연구 분야를 제시합니다.

- **Technical Details**: ADAS는 새로운 빌딩 블록을 발명하거나 이를 새로운 방식으로 결합하는 것을 목표로 합니다. 특히, 메타 에이전트( Meta Agent)가 코드로 새로운 에이전트를 자동으로 발견하는 방식을 탐구합니다. 이 과정에서 메타 에이전트는 이전의 발견 내용을 바탕으로 점진적으로 흥미로운 새로운 에이전트를 프로그래밍합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘인 Meta Agent Search는 여러 도메인에서 기존 수작업 설계 에이전트보다 월등한 성능을 발휘하는 에이전트를 점진적으로 발명할 수 있음을 보여주었습니다. 특히, 생성된 에이전트는 서로 다른 도메인과 모델에서도 뛰어난 성능을 유지하는 경향을 보였습니다.



### Multi-Modal Dialogue State Tracking for Playing GuessWhich Gam (https://arxiv.org/abs/2408.08431)
Comments:
          Published at CICAI 2023 (CAAI-A), codes at this https URL

- **What's New**: 본 논문에서는 시각적 대화 게임인 GuessWhich에서 QBot의 의사결정 과정에 시각적으로 관련된 추론을 효과적으로 모델링하는 새로운 방식을 제안합니다. 이 방식은 숨겨진 이미지에 대한 정신 모델(mental model)을 중심으로 하며, 대화 상태를 추적하여 강력한 시각적 추론을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 다이얼로그 상태 추적(DST)에 기반한 QBot 모델로, 시각적 관련 추론을 지원하기 위한 정신 이미지를 형성하는 방법을 학습합니다. 이 모델은 언어 상태와 이미지 상태를 모두 포함하는 대화 상태를 업데이트하고 추적함으로써, 질문을 생성하고 내부 표현을 업데이트합니다.

- **Performance Highlights**: VisDial 데이터셋(v0.5, 0.9, 1.0)에서 실험한 결과, 제안된 모델은 모든 메트릭에서 이전의 최첨단 모델들을 초월하여 새로운 최첨단 성능을 달성했습니다.



### Unleash The Power of Pre-Trained Language Models for Irregularly Sampled Time Series (https://arxiv.org/abs/2408.08328)
- **What's New**: 이 연구는 불규칙 샘플링 시계열(ISTS) 분석을 위한 프리 트레인 언어 모델(PLMs)의 가능성을 탐구하는 첫 번째 작업입니다. 기존의 연구들은 정기적으로 샘플링된 시계열(RSTS) 중심으로 진행되었으며, ISTS의 독특한 도전 과제를 간과했습니다.

- **Technical Details**: 연구에서는 ISTS의 효율성을 극대화하기 위한 다양한 표현 방법의 효과를 조사하였고, PLMs의 내재된 능력을 활용하기 위해 시리즈 기반 표현 방법을 제안했습니다. 또한, 시간 인식 및 변수 인식을 통해 ISTS의 내외부 시계열 모델링을 다룰 수 있는 ISTS-PLM이라는 통합 PLM 기반 프레임워크를 제안하며, 여기에 학습 가능한 입력 임베딩 레이어와 작업 특정 출력 레이어를 통합했습니다.

- **Performance Highlights**: ISTS-PLM은 분류, 보간, 외삽 등 다양한 분석 작업에서 최첨단 성능을 지속적으로 달성하였으며, 특히 헬스케어 및 생체역학과 같은 과학적 분야에서의 몇 가지 샷 및 제로 샷 학습 시나리오에서도 뛰어난 성과를 보였습니다.



### xGen-MM (BLIP-3): A Family of Open Large Multimodal Models (https://arxiv.org/abs/2408.08872)
- **What's New**: 이 보고서는 xGen-MM (BLIP-3으로도 알려짐)이라는 대규모 멀티모달 모델 개발 프레임워크를 소개합니다. 이 프레임워크는 정교하게 curated 된 데이터셋, 훈련 레시피, 모델 아키텍처 및 결과적으로 생성된 LMMs를 포함합니다. xGen-MM은 Salesforce의 xGen 이니셔티브를 확장하여 기초 AI 모델을 개발합니다.

- **Technical Details**: xGen-MM은 비전 토큰 샘플러(perceiver resampler)를 사용하여 Q-Former 아키텍처를 대체하고, 멀티모달 환경에서 텍스트 토큰의 오토 레그레시브 손실에 집중하여 훈련 목표를 단순화합니다. 프레임워크는 다양한 멀티모달 데이터 소스에서 오는 자율형 멀티모달 인터리브 텍스트와 비전 토큰을 입력으로 받아들입니다.

- **Performance Highlights**: 우리의 사전 훈련된 기본 모델은 강력한 인컨텍스트 학습 능력을 발휘하며, 지침 조정된 모델은 유사한 모델 크기를 가진 오픈 소스 LMMs와 비교하여 경쟁력 있는 성능을 보여줍니다. 우리는 DPO(Dynamic Policy Optimization)로 안전 조정된 모델을 간단히 소개하며, 이는 허위 정보를 완화하고 안전성을 높이고자 합니다.



### Optimal Symmetries in Binary Classification (https://arxiv.org/abs/2408.08823)
Comments:
          13 pages, 1 figure, 2 tables

- **What's New**: 본 연구에서는 이진 분류 작업에서 그룹 대칭(group symmetries)의 중요성을 탐구하며 Neyman-Pearson 최적성 원리를 활용한 새로운 프레임워크를 소개합니다. 저자들은 큰 대칭 그룹이 항상 분류 성능 향상과 관련이 있다는 일반적인 직관과 달리, 적절한 그룹 대칭을 선택하는 것이 일반화(generalisation) 및 샘플 효율(sample efficiency)을 최적화하는 데 결정적이라는 사실을 보여줍니다.

- **Technical Details**: 이 연구는 그룹 불변성(group invariances)과 시간 내의 대칭을 바탕으로 이질적(homogeneous) 데이터 분포를 형성하며, 이러한 대칭을 갖춘 신경망 아키텍처 설계에 대한 이론적 기초를 제공합니다. 특히, 각 층에서 그룹의 작용을 정의하고, 이러한 대칭을 데이터의 기본 확률 분포와 정렬할 수 있는 새로운 방법론을 개발했습니다. 일반적으로, 대칭을 조정하여 문제의 특정 특성에 맞춰 분류 정확도를 향상시킬 수 있습니다.

- **Performance Highlights**: 이론적 분석과 실험 결과는 최적의 분류 성능이 항상 가장 큰 동등 그룹(equivariant groups)과 관련이 있지 않음을 보여주며, 대칭 그룹의 적절한 선택이 효과적이라는 점을 강조합니다. 실험적으로, 이 접근법은 다양한 기계 학습 컨텍스트에서 그룹 불변성이 강조된 아키텍처의 효과성을 실증적으로 입증합니다.



### EasyRec: Simple yet Effective Language Models for Recommendation (https://arxiv.org/abs/2408.08821)
- **What's New**: 이번 연구에서는 텍스트 기반 의미 이해를 협업 신호와 통합하는 효율적인 추천 시스템 EasyRec을 제안합니다. 이를 통해 추천 시스템의 일반화 능력을 향상시킵니다.

- **Technical Details**: EasyRec은 텍스트-행동 정렬 프레임워크(titled text-behavior alignment framework)를 기반으로 하며, 대조 학습(contrastive learning)과 협업 언어 모델 튜닝(collaborative language model tuning)을 결합하여 텍스트 보강된 의미 공간과 협업 행동 정보 간의 강력한 정렬을 보장합니다.

- **Performance Highlights**: EasyRec은 기존의 최첨단 모델들에 비해 뛰어난 성능을 보이며, 특히 텍스트 기반 제로샷(zero-shot) 추천 시나리오에서 그 우수성을 입증했습니다. 또한, 이 모델은 사용자 프로필을 동적으로 생성하여 시간의 변화에 따라 사용자 선호도에 잘 적응합니다.



### Constructing Domain-Specific Evaluation Sets for LLM-as-a-judg (https://arxiv.org/abs/2408.08808)
Comments:
          14 pages, 8 figures

- **What's New**: 이 논문에서는 다양한 도메인에 맞춘 대규모 언어 모델(LLM) 평가를 위한 새로운 데이터 파이프라인을 소개합니다. 기존의 평가 방법에 비해 신뢰성과 정확성을 크게 향상시켜 모델 성능을 보다 잘 분석할 수 있는 접근 방식을 제안합니다.

- **Technical Details**: 새롭게 소개한 데이터 파이프라인은 수동 큐레이션(manual curation), 반감독 학습(semi-supervised learning) 및 층화 샘플링(stratified sampling)을 활용하여 도메인 및 언어에 걸쳐 균형 잡힌 표현을 보장합니다. 평가 세트는 14개 카테고리에서 1573개의 샘플로 구성되며, 10개 모델 간의 분리도(separability)는 84%에 달합니다.

- **Performance Highlights**: 본 논문에서 제안한 평가 세트는 Chatbot Arena와의 일치율이 84%, Spearman 상관계수(Spearman correlation coefficient)는 0.915로, 이전의 Arena Hard와 Alpaca-Eval 2.0 LC 대비 각각 9%, 20% 향상된 결과를 보여주고 있습니다. 또한, 오픈 소스 평가 도구를 제공하여 사용자가 정의한 카테고리별로 모델 성능을 세밀하게 분석할 수 있도록 합니다.



### CIKMar: A Dual-Encoder Approach to Prompt-Based Reranking in Educational Dialogue Systems (https://arxiv.org/abs/2408.08805)
Comments:
          This paper is the result of the final project of the Natural Language Processing course, Master of Artificial Intelligence, Universitas Gadjah Mada

- **What's New**: 이번 연구에서는 Gemma 언어 모델을 기반으로 한 CIKMar라는 교육 대화 시스템의 효율적인 접근 방식을 소개합니다. CIKMar는 BERT 및 SBERT 모델을 포함한 Dual-Encoder 순위 시스템을 활용하여 제공된 답변의 관련성과 정확성을 높였습니다.

- **Technical Details**: CIKMar는 BERTScore 메트릭을 사용하여 0.70의 강력한 recall과 F1-score를 달성하였으며, Gemma 1.1 2B 모델을 활용하여 12GB의 RAM과 단일 GPU T4에서 효율적으로 실행될 수 있도록 설계되었습니다. 이 시스템은 손으로 작성한 프롬프트를 사용하여 후보 출력을 재정렬하는 Dual-Encoder 전략을 채택했습니다.

- **Performance Highlights**: CIKMar는 교육 대화에서 뛰어난 응답의 관련성과 효과성을 보여주었으나, Dual-Encoder가 이론적인 응답을 실용적인 응답보다 우선시하는 경향이 있다는 중요한 도전 과제를 발견했습니다. 이는 Gemma와 같은 컴팩트하고 효율적인 모델의 가능성을 강조합니다.



### A Disease-Specific Foundation Model Using Over 100K Fundus Images: Release and Validation for Abnormality and Multi-Disease Classification on Downstream Tasks (https://arxiv.org/abs/2408.08790)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구에서는 Fundus-Specific Pretrained Model(이미지+Fundus)이라는 새로운 감독 학습 인공지능 모델을 개발하였습니다. 이 모델은 fundus 이미지에서 이상을 탐지하도록 훈련되었으며, 57,803 개의 이미지를 활용하여 다양한 다운스트림 작업에서 우수한 성능을 보였습니다.

- **Technical Details**: 연구에 사용된 데이터는 서울대학교병원에서 수집된 113,645개의 fundus 이미지입니다. 이 모델은 두 가지 유형의 질병 특정 기초 모델을 개발했으며, 첫 번째는 fundus 이미지 데이터로 정의된 'Fundus' 모델이고, 두 번째는 대규모 비의료 데이터셋(ImageNet-1k)에서 훈련된 후 fundus 이미지 데이터셋으로 재훈련된 'ImageNet + Fundus' 모델입니다. 모델은 4개의 NVIDIA A100 GPU를 사용하여 PyTorch 기반으로 훈련되었으며, Adam 옵티마이저와 다양한 데이터 증강 기법을 포함하여 모델을 최적화했습니다.

- **Performance Highlights**: 모델은 다양한 다운스트림 작업에서 5-fold 교차 검증을 통해 평가되었으며, 모든 작업에서 AUC가 주요 성능 지표로 사용되었습니다. 개발된 이미지+Fundus 모델은 기존의 일반 모델보다 질병을 보다 정밀하게 탐지하며, 데이터 수를 줄이면서도 모델 성능을 개선할 수 있는 뚜렷한 장점을 보여주었습니다.



### A Transparency Paradox? Investigating the Impact of Explanation Specificity and Autonomous Vehicle Perceptual Inaccuracies on Passengers (https://arxiv.org/abs/2408.08785)
Comments:
          Submitted to Transportation Research Part F: Traffic Psychology and Behaviour. arXiv admin note: text overlap with arXiv:2307.00633

- **What's New**: 본 논문은 자율주행 차량에서의 투명성(Transparency)과 설명의 구체성(Specificity)이 탑승자의 안전 인식과 불안 수준에 미치는 영향을 조사하였다. 특히, 구체적인 설명이 제공될 때 상대적으로 탑승자의 안전감이 증가하지만, 인식 시스템 오류가 드러날 경우 불안감이 증가한다는 점에서 투명성과 안전성 간의 상충 관계가 제기된다.

- **Technical Details**: 논문에서는 고급 자율주행(Autonomous Driving, AD) 맥락에서 39명의 참가자를 대상으로 한 실험을 진행하였으며, 두 가지 유형의 설명(구체적(Specific)과 추상적(Abstract))이 탑승자에게 미치는 영향을 분석하였다. 추상적 설명은 운전 상황의 세부 정보를 숨기고, 구체적 설명은 세부 정보와 인식 시스템 오류를 드러낸다. 실험 결과, 인식 오류가 드러날 경우 안전감이 감소하고 불안 수준이 증가하는 경향이 나타났다.

- **Performance Highlights**: 탑승자들은 최적의 인식 정확성을 갖춘 자율주행차가 제공하는 구체적 설명을 선호하는 것으로 나타났다. 특히, 인식 시스템의 오류가 적을 때는 안전감이 대폭 향상되는 반면, 구체적인 설명이 많을수록 불안감이 증가하는 경향이 있음을 발견하였다.



### ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Sca (https://arxiv.org/abs/2408.08739)
Comments:
          8 pages, ASVspoof 5 Workshop (Interspeech2024 Satellite)

- **What's New**: ASVspoof 5는 음성 스푸핑(specifically speech spoofing) 및 딥페이크(deepfake) 공격 연구를 촉진하기 위한 다섯 번째 챌린지로, 크라우드소싱된 데이터로 구성된 새로운 데이터베이스를 채택하였습니다. 이번 챌린지에서는 공격이 처음으로 적대적 공격(adversarial attacks)을 포함하며, 새로운 메트릭(metrics)을 통해 스푸핑-강건 자동 화자 인증(spoofing-robust automatic speaker verification, SASV) 및 스탠드얼론 탐지 솔루션을 평가합니다.

- **Technical Details**: ASVspoof 5 챌린지는 두 개의 트랙으로 나뉘며, 각 트랙은 스탠드얼론 스푸핑 및 딥페이크 탐지, 스푸핑-강건 자동 화자 인증(SASV)입니다. 새로운 데이터베이스는 4,000명이 넘는 화자(source data)에서 수집된 데이터로 구성되어 있으며, 다양한 음향 조건(acoustic conditions)에서 수집되었습니다. 또한, 새로운 공격 방식은 최신 TTS(text-to-speech) 합성 및 음성 변환(voice conversion) 알고리즘을 활용하고 있으며, 적대적 공격이 통해 CM(counters measures)과 SASV 시스템에 대한 위협을 평가합니다.

- **Performance Highlights**: 기본 시스템(baseline systems)은 공격으로 인해 상당한 손상을 입었으며, 챌린지 참가자들의 제출물(submissions)은 성과를 크게 향상시키는 결과를 보였습니다. 새로운 평가 메트릭(minimum detection cost function, minDCF)과 기타 보조 메트릭을 통해 전반적인 시스템 성능을 종합적으로 검토하였습니다.



### Correspondence-Guided SfM-Free 3D Gaussian Splatting for NVS (https://arxiv.org/abs/2408.08723)
Comments:
          arXiv admin note: text overlap with arXiv:2312.07504 by other authors

- **What's New**: 본 연구에서는 3D Gaussian Splatting을 활용한 SfM-free Novel View Synthesis (NVS) 방법을 제안합니다. 이는 상대적 카메라 포즈 최적화를 보다 효과적으로 수행할 수 있는 방식으로, 피사체와 렌더링 결과 간의 대응 관계를 이용하여 픽셀 정렬을 개선합니다.

- **Technical Details**: 제안된 방법은 2D 대응 검출을 통해 렌더링 이미지와 타겟 이미지 간의 픽셀 매칭을 찾아내고, 픽셀 매칭을 기반으로 하는 새로운 손실 함수를 설계하여 최적화를 수행합니다. 2D 스크린 공간에서의 3D Gaussian을 위한 근사 표면 렌더링 파이프라인을 개발하여, 3D Gaussian의 매개변수에 대한 그래디언트 역전파를 용이하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 최첨단 방법들에 비해 우수한 성능과 시간 효율성을 보여주었습니다. 특히, 픽셀 부정렬 문제를 최소화하여 최적화의 안정성을 높였습니다.



### Beyond KAN: Introducing KarSein for Adaptive High-Order Feature Interaction Modeling in CTR Prediction (https://arxiv.org/abs/2408.08713)
Comments:
          KarSein for CTR

- **What's New**: 이번 논문에서는 CTR(Click-Through Rate) 예측을 위해 새로운 모델인 Kolmogorov-Arnold Represented Sparse Efficient Interaction Network(KarSein)를 소개합니다. KarSein은 고차원 feature interaction을 효율적으로 모델링하는 데 중점을 두고, 기존의 Kolmogorov-Arnold Networks(KAN)의 한계를 극복합니다.

- **Technical Details**: KarSein은 KAN의 여러 활성화 기능을 단일 feature에 하나의 활성화 기능만 할당하여 계산을 최적화합니다. 또한, 2D embedding vectors를 지원하여 feature 상호작용을 벡터 방식으로 모델링할 수 있게 합니다. KarSein은 각 레이어에서 feature의 쌍을 곱하는 추가 단계를 포함해 멀티플리케이티브(multiplicative) 관계를 학습할 수 있도록 설계되었습니다. 또한, 이 모델은 KAN의 단순화 기술을 유지하여 global explainability를 지원합니다.

- **Performance Highlights**: KarSein은 세 가지 데이터 세트에서 state-of-the-art 성능을 보여주며, 최소한의 계산 비용으로 높은 예측 정확성을 달성하였습니다. 이를 통해 KarSein은 sparse network 구조의 장점을 유지하면서 불필요한 feature를 제거하여 효율적인 추론을 가능하게 합니다.



### Beam Prediction based on Large Language Models (https://arxiv.org/abs/2408.08707)
- **What's New**: 밀리미터파(mmWave) 통신의 견고성을 개선하기 위해 대규모 언어 모델(LLMs)을 활용한 새로운 방법을 제안합니다. 기존의 LSTM 모델은 정확도는 높지만 견고성과 일반화 능력이 부족한데 반해, LLMs는 이러한 문제를 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 시간 시퀀스 데이터를 텍스트 기반 표현으로 변환하고, Prompt-as-Prefix(PaP) 기술을 사용하여 문맥을 보강함으로써 LLM의 강점을 최대한 활용하는 접근 방식을 개발했습니다. 이 모델은 기존 아키텍처를 변경하지 않고도 LLM을 활용하여 최적의 빔을 예측합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 LLM 기반 방법이 기존 LSTM 기반 모델에 비해 뛰어난 견고성과 일반화 성능을 보여주었습니다. 이는 무선 통신 분야에서 LLM의 응용 가능성을 확립하는 중요한 연구 결과입니다.



### Beyond the Hype: A dispassionate look at vision-language models in medical scenario (https://arxiv.org/abs/2408.08704)
Comments:
          10 pages

- **What's New**: 최근 대규모 비전-언어 모델(Large Vision-Language Models, LVLMs)의 발전은 다양한 작업에서 두드러진 능력을 보여주며 AI 커뮤니티에서 큰 주목을 받고 있습니다. 그러나 의료와 같은 전문 분야에서의 성능과 신뢰성은 충분히 평가되지 않았습니다. 이 연구에서는 기존 LVLMs를 포괄적으로 평가하기 위해 RadVUQA라는 새로운 방사선 시각 이해 및 질문 응답 벤치마크를 도입했습니다.

- **Technical Details**: RadVUQA는 LVLMs의 성능을 1) 해부학적 이해(anatomical understanding), 2) 다중모달 이해(multimodal comprehension), 3) 정량적 및 공간적 추론(quantitative and spatial reasoning), 4) 생리학적 지식(physiological knowledge), 5) 강건성(robustness) 등 5가지 차원에서 평가합니다. 이 연구에서는 117개의 장기/구조와 56개의 장기의 CT 및 MR 스캔을 포함한 새로운 데이터셋을 구축하였으며, 다양한 시험 환경에서 LVLM의 성능을 테스트합니다.

- **Performance Highlights**: 평가 결과, 일반 LVLM 및 의료 특화 LVLM 모두가 다중 모달 이해와 정량적 추론 능력에서 중대한 결함을 보였습니다. 이 연구는 LVLM과 임상의 간의 큰 격차를 밝혀내며, 더 강력하고 지능적인 LVLM의 필요성을 강조합니다.



### Quantifying the Effectiveness of Student Organization Activities using Natural Language Processing (https://arxiv.org/abs/2408.08694)
Comments:
          11 pages, 4 figures, presented in International Conference on Generative Al and its Applications (ICGAIA-24) last 22nd - 23rd, July, 2024 at Jakarta, Indonesia

- **What's New**: 학생의 과외 활동(Extracurricular Activities)이 학생의 교육 경험을 풍부하게 하는 데 중요한 역할을 한다는 점을 강조하고, 기계 학습(Machine Learning)과 자연어 처리(Natural Language Processing)를 활용하여 과외 활동의 효과를 정량화하는 연구를 제시합니다. 또한, BERT 모델을 사용하여 학생의 정서적 반응을 분석하는 새로운 워크플로우를 개발합니다.

- **Technical Details**: 이 연구에서는 Composed한 BERT(Large Language Model)를 활용하여, pysentimiento 툴킷을 통해 감정 분석(Sentiment Analysis)을 수행하는 기계 학습 파이프라인(ML Workflow)을 구축하였습니다. 데이터 전처리(Data Preprocessing), 주요 특징 선택(Key Feature Selection), LLM 기능 처리(Feature Processing), 점수 집계(Score Aggregation)의 단계로 이루어진 이 워크플로우는 각 데이터 세트에 대한 이벤트 점수(Event Score)를 산출합니다.

- **Performance Highlights**: BERT LLM은 제품 리뷰(Product Reviews) 및 댓글(Post Comments)을 넘어서 감정 분석에 효과적으로 활용될 수 있음을 보여줍니다. 이 연구는 교육 기관 학생 사무소(Student Affairs Offices)에 NLP를 실제 사례에 적용할 수 있는 실용적인 예시를 제공하며, 데이터 기반의 의사 결정(Data-Driven Decision Making)이 미칠 수 있는 잠재적 영향을 강조합니다.



### The Fellowship of the LLMs: Multi-Agent Workflows for Synthetic Preference Optimization Dataset Generation (https://arxiv.org/abs/2408.08688)
- **What's New**: 이 논문은 다중 에이전트 워크플로우를 활용하여 합성 선호 최적화(Preference Optimization, PO) 데이터셋 생성을 평가합니다. PO 데이터셋 생성을 위해 두 개의 모듈(응답 평가 및 응답 생성)을 사용하며, LLM(대형 언어 모델)을 활용하여 인간의 주석 작업을 자동화합니다.

- **Technical Details**: 응답 평가 모듈에서는 LLM이 평가자로 사용되어 서로 다른 프롬프트 전략을 통해 우수한 응답 선택을 위해 경쟁합니다. 최종적으로 LLM를 사용한 Judge, Jury 및 Debate 방식 사이의 성능을 비교하고 Cohen의 카파(Kappa)를 통해 일치도 평가합니다. 응답 생성 모듈에서는 LLM 피드백 루프를 통한 다양한 구성의 성능을 비교합니다.

- **Performance Highlights**: GPT-4o-as-a-Judge는 GPT 계열의 응답이 포함되지 않은 데이터셋에서 더 높은 일관성을 보였으며, Llama를 생성기로, Gemma를 리뷰어로 사용하는 LLM 피드백 루프는 단일 에이전트인 Llama 및 Gemma에 비해 각각 71.8%와 73.8%의 높은 승률을 기록했습니다.



### SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for~Sequential Recommendation (https://arxiv.org/abs/2408.08686)
- **What's New**: 본 논문에서는 SC-Rec이라는 새로운 추천 시스템을 제안하며, 이는 다양한 아이템 인덱스와 프롬프트 템플릿으로부터 학습한 선호 지식을 결합하여 추천 결과의 일관성을 높입니다.

- **Technical Details**: SC-Rec은 세 가지 단계로 구성됩니다: (1) 아이템 다중 인덱스 생성, (2) 다중 인덱스 추천 모델 훈련, (3) 일관성 점수 기반 재정렬. 이를 통해 아이템 지식을 계층적 양자화 기법으로 생성하고, 다양한 프롬프트 템플릿을 활용하여 고유한 사용자 상호작용을 반영합니다.

- **Performance Highlights**: 실험 결과 SC-Rec은 세 가지 실제 데이터 세트에서 최신 기술들에 비해 상당한 성능 향상을 보였으며, 이질적인 인덱스와 다양한 프롬프트 템플릿에서 얻은 보완 지식을 효과적으로 통합하였습니다.



### Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks? (https://arxiv.org/abs/2408.08685)
- **What's New**: 이번 논문에서는 GNN(Graph Neural Networks)의 강건성을 개선하기 위해 LLMs(Large Language Models)의 가능성을 탐구했습니다. 특히, topology 공격에 대한 GNN의 취약성을 밝히고, 이를 보완하기 위한 LLM 기반의 구조 추론 프레임워크인 LLM4RGNN을 제안하였습니다.

- **Technical Details**: 이 프레임워크는 GPT-4의 추론 능력을 활용하여 악의적인 엣지를 식별하고, 중요한 엣지를 복구하는 방식으로 GNN의 강건성을 개선하는 데 초점을 맞추고 있습니다. LLM4RGNN은 지역 LLM을 사용하여 공격된 그래프 구조를 정화합니다.

- **Performance Highlights**: 실험 결과, LLM4RGNN은 여러 GNN에 걸쳐 강건성을 일관되게 향상시키며, 일부 경우에서 perturbation 비율이 40%로 증가하더라도 GNN의 정확도가 여전히 클린 그래프보다 높았습니다.



### Neural Reward Machines (https://arxiv.org/abs/2408.08677)
- **What's New**: 이번 논문에서는 Neural Reward Machines (NRM)이라는 새로운 네오 심볼릭(framework) 방식을 정의하여 Reinforcement Learning (RL)에서 비표상적인(non-symbolic) 비마르코프적(non-Markovian) 문제를 해결할 수 있는 방법을 제시합니다. 이는 기존의 Symbol Grounding (SG) 함수에 대한 의존도를 줄이면서도 RL 알고리즘의 성능을 높일 수 있게 해줍니다.

- **Technical Details**: NRM은 Moore Machines의 확률적( probabilistic ) 완화를 기반으로 한 자동자(automata) 기반의 네오 심볼릭(neurosymbolic) 프레임워크입니다. 본 시스템은 반지도 학습을 통해 기존의 기호 지식을(raw data) 결합함으로써 RL 알고리즘의 효과를 극대화할 수 있습니다.

- **Performance Highlights**: 실험 결과, NRM은 기존의 Deep RL 방법들과 비교하여 SG 함수에 대한 사전 지식이 없더라도 비표상적 환경에서 더 나은 성능을 보여주었으며, 새로운 SSSG 알고리즘을 통해 일반적인 기준 기법보다 10^3배 더 효율적으로 시간 사양(temporal specifications)의 가능한 해를 분석할 수 있음을 증명했습니다.



### MAT-SED: AMasked Audio Transformer with Masked-Reconstruction Based Pre-training for Sound Event Detection (https://arxiv.org/abs/2408.08673)
Comments:
          Received by interspeech 2024

- **What's New**: 본 연구는 순수 Transformer 기반의 Sound Event Detection (SED) 모델인 MAT-SED를 제안합니다. MAT-SED는 마스킹 복원(Masked Reconstruction) 기반의 사전 훈련 방식으로 설계되어 있습니다.

- **Technical Details**: MAT-SED는 상대적 위치 인코딩을 가진 Transformer를 контекст 네트워크로 사용하며, 이는 latent feature의 긴-range context 의존성을 효과적으로 포착합니다. 이 모델은 사전 훈련 단계에서 Masked Reconstruction 작업을 통해 context 네트워크를 self-supervised 방식으로 학습합니다.

- **Performance Highlights**: DCASE2023 Task 4에서 MAT-SED는 0.587/0.896 PSDS1/PSDS2를 달성하며, 기존의 최첨단 SED 시스템을 초월하는 성능을 보였습니다.



### Adaptive Layer Selection for Efficient Vision Transformer Fine-Tuning (https://arxiv.org/abs/2408.08670)
- **What's New**: 본 논문에서는 Vision Transformers (ViTs)의 효율적인 fine-tuning 방법인 ALaST(Adaptive Layer Selection Fine-Tuning for Vision Transformers)를 소개합니다. 이 방법은 fine-tuning 과정에서 필요한 자원 소모를 줄이면서도 속도를 향상시킵니다.

- **Technical Details**: ALaST는 각 layer의 중요도를 adaptively 추정하고, 이에 따라 'compute budgets'를 할당하여 자원 소모를 조정합니다. 낮은 budget이 할당된 layer는 input tokens의 수를 줄여서 학습하거나 동결(freeze)하여 계산 비용을 절감합니다. 연산 중 token을 discard하면 처리 속도를 높이고 메모리 요구량을 줄입니다.

- **Performance Highlights**: ALaST는 기존의 full fine-tuning 접근 방식에 비해 훈련 시간을 최대 1.5배, FLOPs를 최대 2배, 메모리 사용량을 최대 2배 감소시킵니다. 또한 LoRA와 같은 다른 파라미터 효율적인 fine-tuning 방법과 성공적으로 결합하여 사용할 수 있습니다.



### A Multivocal Literature Review on Privacy and Fairness in Federated Learning (https://arxiv.org/abs/2408.08666)
Comments:
          Accepted for publication at the Internationale Tagung Wirtschaftsinformatik 2024

- **What's New**: 본 논문은 Federated Learning(연합 학습)이 AI 애플리케이션에서 데이터 공유를 필요로 하지 않도록 혁신적인 방법을 제시하지만, 여전히 훈련 중 정보를 추출할 수 있음을 보여줍니다. 이로 인해 differential privacy(차등 개인 정보 보호)와 같은 추가적인 개인 정보 보호 조치가 필수적이라는 점을 강조합니다.

- **Technical Details**: 본 연구는 고위험 애플리케이션, 특히 healthcare(헬스케어) 분야에서 불공정한 오류의 반복을 피하는 것이 얼마나 중요한지를 논의합니다. 최근 연구에서는 privacy(개인 정보 보호)와 fairness(공정성) 간의 고유한 긴장이 있음을 밝혔으며, 이 두 개념을 Federated Learning에 통합하는 현재의 방법들에 대해 다각적으로 문헌 검토를 수행하였습니다.

- **Performance Highlights**: 현재의 개인 정보 보호와 공정성 간의 관계가 소홀히 여겨지고 있으며, 이는 실제 응용 프로그램에서 중요한 위험 요소로 작용할 수 있음을 지적합니다. 따라서 privacy, fairness, performance(성능) 간의 관계를 탐구할 필요성을 강조하고, 통합된 Federated Learning 프레임워크의 개발을 촉구합니다.



### Mitigating Backdoor Attacks in Federated Learning via Flipping Weight Updates of Low-Activation Input Neurons (https://arxiv.org/abs/2408.08655)
- **What's New**: 서버가 로컬 학습 프로세스를 직접 감독할 수 없어 악의적인 클라이언트가 백도어를 추가할 기회를 제공합니다. 본 논문에서는 저활성 입력 뉴런의 가중치 업데이트를 뒤집는(FlaI) 방법을 제안하여 federated learning에서의 백도어 공격을 방어합니다.

- **Technical Details**: FLAIN은 보조 데이터셋을 이용해 저활성 입력 뉴런을 식별하고, 이들과 관련된 가중치 업데이트를 뒤집습니다. 저활성 입력의 임계값을 점진적으로 증가시키고 업데이트를 반복해서 뒤집어 성능 저하가 수용 불가능한 수준에 도달할 때까지 진행합니다.

- **Performance Highlights**: FLAIN 방법은 다양한 공격 시나리오에서 백도어 공격의 성공률을 효과적으로 낮추고, 클린 데이터에서의 성능 저하는 최소한으로 유지됨을 보여줍니다.



### TextCAVs: Debugging vision models using tex (https://arxiv.org/abs/2408.08652)
Comments:
          11 pages, 2 figures. Accepted at iMIMIC Workshop at MICCAI 2024

- **What's New**: 이 논문에서는 TextCAVs라는 새로운 개념 기반 해석 가능성 방법을 소개하고 있습니다. 이 방법은 CLIP와 같은 비전-언어 모델을 사용하여 개념의 텍스트 설명만으로 CAVs(Concept Activation Vectors)를 생성하여 이미지 예시 없이도 해석 가능성을 제공합니다.

- **Technical Details**: TextCAVs는 두 개의 선형 계층을 훈련하여 비전 모델과 비전-언어 모델 사이의 특성을 변환합니다. 이 방법은 재구성 손실(mean squared error, MSE)과 사이클 손실(cycle loss)을 사용하여 훈련되며, 이를 통해 원래 형태로 일관성을 유지합니다.

- **Performance Highlights**: 초기 실험 결과 TextCAVs는 자연 이미지 데이터셋(ImageNet)과 가슴 X선 데이터셋(MIMIC-CXR)에서 합리적인 설명을 생산하며, 딥러닝 모델의 디버깅에도 유용함을 보여주었습니다.



### Reasoning Beyond Bias: A Study on Counterfactual Prompting and Chain of Thought Reasoning (https://arxiv.org/abs/2408.08651)
- **What's New**: 이번 연구는 Massive Multi-Task Language Understanding (MMLU) 작업에서 언어 모델이 훈련 데이터에서 흡수한 편향을 조사하고, 이를 해결하기 위해 두 가지 새로운 방법인 Counterfactual Prompting with Chain of Thought (CoT) 및 Agnostically Primed CoT (APriCoT)를 제안합니다.

- **Technical Details**: 연구에서는 MMLU에서 언어 모델의 답변 선택 선호도가 학습된 통계적 규칙의 차이에 따라 어떻게 결정되는지를 규명했습니다. CoT와 CF(Counterfactual) 프롬프트를 결합하여 모델의 행동이 기준비율 확률(Base-rate Probability, BRP) 효과와 분리되어 동작하도록 하였으나, 알고 보니 BRP 효과가 증폭되는 현상이 관찰되었습니다. 이를 해결하기 위해 APriCoT를 제안하였습니다.

- **Performance Highlights**: APriCoT 방법이 언어 모델의 답변 품질과 공정성을 개선하는 데 기여함을 보였으며, 제안된 방법이 기존 방법들보다 더 나은 정확도를 나타냈습니다. 특히, 모델의 선호도가 근거 답변 분포와 거의 동일하게 분포되어 있으며, CF나 CoT보다 더 나은 정확성을 보여주었습니다.



### A Survey on Benchmarks of Multimodal Large Language Models (https://arxiv.org/abs/2408.08632)
- **What's New**: 이번 논문은 다중 모달 대규모 언어 모델(Multimodal Large Language Models, MLLMs)의 180개의 벤치마크 및 평가를 포괄적으로 검토하며, 이러한 모델들이 다양한 응용 분야에서 가진 성능을 다각도로 평가합니다. 특히, 인지 및 추론, 특정 도메인, 주요 기능과 다양한 모달리티에서의 능력에 중점을 두고 있습니다.

- **Technical Details**: MLLM은 비쥬얼 질문 응답(Visual Question Answering, VQA) 및 특정 도메인 과제를 포함한 다양한 응용 분야에서 뛰어난 성능을 보여줍니다. MLLM의 성능을 객관적으로 비교하고 탐색하기 위해, 영향을 미치는 다섯 가지 주요 분야를 통해 최근 논문을 분석합니다. 이러한 분야는 지각 및 이해, 인지 및 추론, 특정 도메인, 주요 기능 및 기타 모달리티로 구성됩니다. 또한, MLLMs의 동작을 지원하는 세 가지 주요 모듈인 비쥬얼 인코더, 언어 모델, 비쥬얼-언어 프로젝터에 대한 내용을 다룹니다.

- **Performance Highlights**: 최신 벤치마크에 따르면 OpenAI의 GPT-4와 Google의 Gemini는 83개의 평가 기준에서 베스트 쿼터를 기록했으며, 이러한 성과는 다중 모달 기능에서의 효율성을 높이는 데 기여하고 있습니다. 본 논문은 현재 MLLM 평가 방법의 한계를 지적하고, 향후 발전 방향에 대한 논의를 포함합니다.



### RealMedQA: A pilot biomedical question answering dataset containing realistic clinical questions (https://arxiv.org/abs/2408.08624)
Comments:
          Accepted at AMIA Annual Symposium 2024

- **What's New**: 이번 연구에서는 의료 전문가들의 실제 요구를 반영한 질문-답변 데이터셋인 RealMedQA를 소개합니다. 이는 인간과 LLM(대규모 언어 모델) 이 협력하여 생성한 질문으로 구성되어 있습니다.

- **Technical Details**: RealMedQA 데이터셋은 질문-답변 쌍 생성 및 검증 과정에서 LLM을 활용하며, BioASQ와 RealMedQA를 사용하여 여러 QA(질문-답변) 모델을 평가했습니다. 이 과정에서 질문과 답변의 어휘 유사성이 BioASQ에 비해 낮다는 특징이 있습니다.

- **Performance Highlights**: 우리는 LLM이 '이상적인' 질문-답변 쌍을 생성하는 데 있어 더 비용 효율적이라는 것을 보여주었으며, 상위 두 QA 모델에게 추가적인 도전을 제공합니다. 또한, 연구 결과는 코드와 데이터셋을 공개하여 향후 연구를 촉진할 것으로 기대됩니다.



### SketchRef: A Benchmark Dataset and Evaluation Metrics for Automated Sketch Synthesis (https://arxiv.org/abs/2408.08623)
- **What's New**: 본 논문에서는 스케치 합성의 품질 평가를 위한 새로운 벤치마크 데이터셋인 SketchRef를 소개하고, 그것을 통해 스케치 구조적 일관성을 평가하기 위한 mOKS(mean Object Keypoint Similarity)라는 평가 메트릭을 제안합니다.

- **Technical Details**: SketchRef 데이터셋은 동물, 인물 얼굴, 인체, 사물의 4개 카테고리로 구성되며, 여러 수준의 단순화가 적용된 스케치와 참조 사진 간의 시각적 및 의미적 연결을 제공합니다. 또한, 구조 수준 인식 가능성을 평가하는 데 필요한 새로운 방식으로 단순화를 제한하는 인식 가능성 계산 방법을 소개합니다.

- **Performance Highlights**: 198명의 아트 애호가로부터 수집된 8K 응답을 통해 제안된 평가 방법의 유용성이 검증되었으며, SketchRef를 통해 8개의 스케치 합성 방법에 대한 포괄적 평가가 수행되었습니다.



### DeepDFA: Automata Learning through Neural Probabilistic Relaxations (https://arxiv.org/abs/2408.08622)
- **What's New**: 이 연구에서는 흔적(traces)으로부터 결정론적 유한 오토마타(DFAs)를 식별하기 위한 새로운 접근 방식인 DeepDFA를 소개합니다. 이 모델은 확률적 완화와 순환 신경망(RNNs)에서 영감을 받아 훈련 후 해석 가능성을 제공하며, 전통적인 RNN과 비교하여 복잡성이 감소하고 훈련 효율성이 향상되었습니다.

- **Technical Details**: DeepDFA는 기울기 기반(graidient-based) 최적화 기법을 활용하여 확장성(scalability)과 노이즈 저항(noise resilience)에서 조합(combinatorial) 접근 방식을 초월합니다. 이 모델은 RNN과 유사하게 작동하지만, 훈련 후에는 DFA와 같이 완전히 투명하게 변합니다. 또한, LSTMs 및 GRUs와 같은 일반적인 순환 아키텍처보다 적은 가중치(weights)를 사용하며 단 하나의 하이퍼파라미터(hyperparameter)만 필요합니다.

- **Performance Highlights**: 이번 연구의 유효성 검증 실험은 다양한 크기와 복잡성을 가진 정규 언어를 대상으로 진행되었으며, DeepDFA는 신속하고 정확한 성능을 보여주었습니다. 20개 이상의 상태를 가진 목표 DFA에 대해 정확한 SAT 기반 방법을 초과달성하였고, 사전 훈련된 RNN에서 DFA 추출과 비교할 때 더 나은 정확도를 달성하며, 목표 DFA 크기에 더 가까운 DFA를 예측하는 능력을 보여주었습니다.



### PatUntrack: Automated Generating Patch Examples for Issue Reports without Tracked Insecure Cod (https://arxiv.org/abs/2408.08619)
Comments:
          Accepted by ASE'24

- **What's New**: 본 논문은 PatUntrack이라는 자동화된 접근 방식을 제안하여 추적되지 않은 취약한 코드 없이 IR(이슈 리포트)로부터 패치 예제를 생성하는 방법을 다룹니다.

- **Technical Details**: PatUntrack은 자동 프롬프트(auto-prompting) 기술을 활용하여 LLM(대형 언어 모델)에서 취약점의 종류와 유발 논리를 분석하고 적절한 패치를 생성하도록 최적화합니다. 주요 3단계로는 취약점 유발 경로(VTP) 설명 생성, 외부 지식을 활용한 VTP 설명 수정, 수정된 VTP 설명을 기반으로 Top-K 쌍의 취약한 코드 및 패치 예제 생성이 있습니다.

- **Performance Highlights**: PatUntrack은 5,465개의 취약 IR에 대한 실험에서 최고 성능을 달성하며 전통적인 LLM 기준을 평균 +14.6% (Fix@10) 개선했습니다. 76개의 새롭게 공개된 취약 IR에 대해서도 패치 예제를 생성하였으며, 이들 IR의 저자 중 37명이 응답하여 27명이 PatUntrack이 생성한 패치 예제가 유용하다고 확인했습니다.



### Generative Dataset Distillation Based on Diffusion Mod (https://arxiv.org/abs/2408.08610)
Comments:
          The Third Place Winner in Generative Track of the ECCV 2024 DD Challenge

- **What's New**: 이번 논문은 ECCV 2024에서의 첫 번째 데이터셋 디스틸레이션 챌린지의 생성 트랙을 위한 새로운 방법을 제시합니다. 이 방법은 고품질 생성 효과로 인해 확산 모형(Diffusion model)에 기반한 데이터셋 디스틸레이션을 중점적으로 다룹니다.

- **Technical Details**: 본 연구에서는 고속 및 고품질 이미지를 생성할 수 있는 SDXL-Turbo 모델을 활용한 새로운 데이터셋 디스틸레이션 방법을 개발했습니다. 우리 방법은 CIFAR-100과 Tiny-ImageNet 데이터셋을 대상으로 하여, Tiny-ImageNet에서 IPC(Images Per Class)가 10, CIFAR-100에서 IPC가 20에 달하는 성과를 보였습니다. 또한, 클래스 정보를 텍스트 프롬프트로 사용하고, 후처리 데이터 증강(Post data augmentation) 기법을 적용하여 품질이 높은 디스틸레이션 데이터셋을 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 획기적인 효과를 보였으며, ECCV 2024 DD 챌린지 생성 트랙에서 3위를 차지했습니다.



### MM-UNet: A Mixed MLP Architecture for Improved Ophthalmic Image Segmentation (https://arxiv.org/abs/2408.08600)
Comments:
          OMIA2024

- **What's New**: 이번 연구에서는 안과 이미지 분할을 위한 새로운 Mixed MLP 아키텍처인 MM-UNet을 제안합니다. 이 모델은 다양한 깊이에서 특징의 상호작용을 촉진하는 multi-scale MLP (MMLP) 모듈을 포함하고 있습니다.

- **Technical Details**: MM-UNet은 UNet 구조를 기반으로 하며, MMLP 모듈을 통해 지역 정보와 전역 정보를 동시에 캡처합니다. 이 아키텍처는 채널 혼합 MLP를 생략하고 지역 토큰 혼합만을 사용하여 계산 비용을 줄여줍니다.

- **Performance Highlights**: MM-UNet은 사전 세그먼트 Optical Coherence Tomography (AS-OCT) 이미지 데이터셋과 공공 안저 이미지 데이터셋에서 실험을 수행했으며, 최신 딥 세그멘테이션 네트워크와 비교하여 우수한 성능을 보였습니다.



### A Mechanistic Interpretation of Syllogistic Reasoning in Auto-Regressive Language Models (https://arxiv.org/abs/2408.08590)
- **What's New**: 최근 연구들은 auto-regressive Language Models (LMs)에서의 논리적 추론 능력이 체계적 추론 원칙을 학습하는 것이 아니라 훈련 데이터의 표면적인 패턴을 활용하는지를 다루고 있습니다. 이 논문은 LMs 내부의 역학을 이해하기 위해 삼단논법(syllogistic reasoning)에 대한 기계적 해석을 제시합니다.

- **Technical Details**: 이 논문에서는 회로 발견(circuit discovery) 방법론을 통해 내용 독립적(content-independent) 추론 메커니즘을 훈련 중 습득한 세계 지식(world knowledge)과 분리하려고 합니다. 두 가지介입 방법(intervention methods)을 통해 중간 항(middle-term) 억제를 포함한 회로를 발견하였으며, 이는 LMs가 전제에서 타당한 결론을 도출하는 정보를 어떻게 전이하는지를 밝힙니다.

- **Performance Highlights**: 이 연구의 결과는 LMs가 내용 독립적인 추론 메커니즘을 배운다는 것을 시사하지만, 동시에 이러한 메커니즘이 일반화 가능하고 추상적인 논리 원시(logical primitives)를 포함하지 않으며, 훈련 중 습득한 세계 지식의 오염에 취약하다는 것을 발견했습니다. 발견된 회로는 모든 삼단논법(syllogistic schemes)에 대해 충분하고 필요하며, 모델이 높은 하향 정확도(≥ 60%)를 달성할 수 있는 조건을 제공합니다.



### S-RAF: A Simulation-Based Robustness Assessment Framework for Responsible Autonomous Driving (https://arxiv.org/abs/2408.08584)
- **What's New**: AI 기술이 발전함에 따라 자율주행(AI agents) 시스템의 강건성 및 안전성을 보장하는 것이 중요해졌습니다. 이 연구에서는 다양한 조건에서 자율주행 시스템을 엄격히 평가할 수 있는 Simulation-Based Robustness Assessment Framework (S-RAF)를 도입하여 안전 인증 프로세스를 간소화할 수 있는 데 기여하고자 합니다.

- **Technical Details**: S-RAF는 CARLA Driving Simulator를 활용하여 센서 오류, 환경 변화 및 복잡한 교통 상황 등 다양한 조건에서 자율주행 에이전트의 강건성을 평가합니다. 이 프레임워크는 강건성(robustness)과 탄소 배출(carbon emissions) 등 다른 안전 관련 요소들과의 관계를 정량화합니다.

- **Performance Highlights**: S-RAF를 통해 개발자와 이해관계자들이 안전하고 책임 있는 자율주행 에이전트를 구축할 수 있도록 돕습니다. 또한, 현실 세계에서 안전하게 테스트하기 어려운 엣지 케이스들을 탐색할 수 있으며, 테스트 비용 감소와 같은 значительные 장점을 제공합니다.



### AgentSimulator: An Agent-based Approach for Data-driven Business Process Simulation (https://arxiv.org/abs/2408.08571)
- **What's New**: 본 논문은 AgentSimulator라는 리소스 우선의 비즈니스 프로세스 시뮬레이션(AgentSimulator)을 도입하여, 이벤트 로그에서 다중 에이전트 시스템(Multi-Agent System, MAS)을 발견하고, 독특한 리소스 행동 및 상호작용 패턴을 모델링하여 과정을 시뮬레이션하는 방법을 제안합니다.

- **Technical Details**: AgentSimulator는 이벤트 로그에 기반하여 에이전트를 발견하고, 이를 사용하여 프로세스를 시뮬레이션하는 데이터 기반의 에이전트 기반 비즈니스 프로세스 시뮬레이션 접근 방식입니다. 이 접근 방식은 개별 리소스의 행동을 유연하게 조정할 수 있으며, 이는 시뮬레이션의 정확성과 해석 가능성을 높이는 데 기여합니다.

- **Performance Highlights**: AgentSimulator는 기존 접근 방식보다 훨씬 낮은 계산 시간으로 최첨단 시뮬레이션 정확도를 달성하고, 다양한 프로세스 실행 시나리오에 대한 높은 해석 가능성과 적응성을 제공합니다.



### Detecting Unsuccessful Students in Cybersecurity Exercises in Two Different Learning Environments (https://arxiv.org/abs/2408.08531)
Comments:
          To appear for publication in the FIE 2024 conference proceedings

- **What's New**: 이 논문은 사이버 보안 연습에서 기록된 데이터를 활용하여 학습 성적이 저조할 가능성이 있는 학생들을 예측하는 자동화 도구를 개발합니다. 313명의 학생 데이터를 통해 다양한 머신 러닝 알고리즘을 활용하여 학생의 성공 여부를 예측하는 평가를 실시했습니다.

- **Technical Details**: 연구는 두 개의 학습 환경(KYPO CRP 및 EDURange)에서의 학생 행동 데이터를 기반으로 하였고, 자동으로 두 가지 특징 집합을 추출했습니다. 8개의 이진 분류 모델을 훈련하고 평가하여, 각 모델에 대해 특징 선택 및 하이퍼파라미터 튜닝을 수행했습니다.

- **Performance Highlights**: 결과적으로 결정 트리 분류기가 두 개의 학습 환경 모두에서 가장 높은 균형 정확도와 민감도를 기록했습니다. 복잡한 사이버 보안 연습에서 학생의 성공을 예측할 수 있는 데이터의 사용 가치가 입증되었습니다.



### Focus on Focus: Focus-oriented Representation Learning and Multi-view Cross-modal Alignment for Glioma Grading (https://arxiv.org/abs/2408.08527)
- **What's New**: 이 논문에서는 Focus on Focus (FoF) 프레임워크를 도입하여 병리학적 데이터와 유전자 정보를 결합하고, 단일 병리 데이터로도 활용할 수 있는 방법을 제시합니다. 이 프레임워크는 병리학적 표현을 효과적으로 개선하여 글리오마( glioma )의 등급을 더욱 정확하게 구분할 수 있도록 합니다.

- **Technical Details**: FoF 프레임워크는 Focus-oriented Representation Learning (FRL)과 Multi-view Cross-modal Alignment (MCA) 모듈로 구성됩니다. FRL 모듈은 모델이 글리오마 등급과 양성 또는 음성으로 관련된 영역을 식별하도록 유도합니다. MCA 모듈은 병리학적 표현을 유전자 기반 서브스페이스로 투영하여, 텍스처 특징과 유전자 생체표지자 상태를 일치시키는 방식입니다.

- **Performance Highlights**: TCGA GBM-LGG 데이터셋에서 FoF 프레임워크가 기존의 다중 모드 방법보다 우수한 성능을 보이며, 오직 병리학 슬라이드를 사용하여도 뛰어난 결과를 달성합니다. 또한, 병리학적 슬라이드 만으로도 글리오마의 등급을 정확히 구분하는 데에 있어 임상적 의의가 큽니다.



### GS-ID: Illumination Decomposition on Gaussian Splatting via Diffusion Prior and Parametric Light Source Optimization (https://arxiv.org/abs/2408.08524)
Comments:
          15 pages, 13 figures

- **What's New**: GS-ID라는 새로운 프레임워크를 제안하며, 이는 Gaussian Splatting을 기반으로 조명 분해(illumination decomposition)를 수행하여 사실적인 새로운 시점 합성을 실현하고 직관적인 조명 편집을 가능하게 합니다.

- **Technical Details**: GS-ID는 두 가지 주요 구성 요소인 잠재적 확산 사전(intrinsic diffusion priors)과 Spherical Gaussians (SGs)를 활용하여 물리 기반 렌더링을 위한 속성을 추정하고, 환경 조명과 직접 조명을 분해하여 최적화를 진행합니다. 이 프레임워크는 배치 렌더링(deferred rendering) 기법을 채택하여 계산 부하를 줄입니다.

- **Performance Highlights**: 우리의 실험 결과 GS-ID는 현대적인 조명 분해 방법에 비해 우수한 성능을 보이며, 더욱 나은 형태 재구성과 렌더링 성능을 달성하여 길어지는 조명 효과를 효과적으로 제어할 수 있습니다.



### Ex3: Automatic Novel Writing by Extracting, Excelsior and Expanding (https://arxiv.org/abs/2408.08506)
- **What's New**: 이번 논문에서는 인공지능을 이용한 장편 소설 생성의 어려움을 극복하기 위한 방법인 'Extracting Excelsior and Expanding (Ex3)'을 제안하였다. Ex3는 원시 소설 데이터로부터 구조 정보를 추출하고, 이를 바탕으로 LLM을 미세 조정하며, 최종적으로는 트리 구조의 확장 방법을 통해 자연스러운 소설 생성을 이끈다.

- **Technical Details**: Ex3는 원시 소설 데이터에서 구조 정보를 추출하는 'Extract', 이를 기반으로 LLM을 미세 조정하는 'Excelsior', 그리고 생성한 기초를 토대로 장편 소설을 생성하는 'Expand'의 세 단계로 구성되어 있다. 자아 발화(self-instructing) 방법을 활용하여 관련성을 기반으로 텍스트를 그룹화하고 요약하는 방식으로 단계적인 구조 정보 추출을 진행한다.

- **Performance Highlights**: Ex3를 통해 생성된 소설은 기존의 방법들에 비해 논리적 일관성과 매력도에서 우수한 성능을 보였으며, 장편 소설 생성의 질이 높아졌다.



### Adversarial Contrastive Learning Based Physics-Informed Temporal Networks for Cuffless Blood Pressure Estimation (https://arxiv.org/abs/2408.08488)
Comments:
          14 pages, 8 figures

- **What's New**: 이번 연구에서는 물리적 정보가 반영된 시계열 데이터에 대한 새로운 모델인 Physics-Informed Temporal Network (PITN)를 소개합니다. 이 모델은 적은 양의 데이터로 정확한 커프리스 혈압(BP) 추정이 가능하도록 설계되었습니다.

- **Technical Details**: PITN은 기존의 물리적 정보가 반영된 신경망(Physics-Informed Neural Network, PINN)을 시간적 블록과 결합하여 개인화된 심혈관 주기를 모델링합니다. 또한, 적대적 훈련(adversarial training) 및 대조 학습(contrastive learning)을 활용해 혈압 동역학의 변화를 보다 효과적으로 학습합니다.

- **Performance Highlights**: 세 가지 널리 사용되는 데이터 세트(생체 임피던스, PPG, 밀리미터파)를 통해 실험한 결과, 제안된 방법이 이전의 최첨단 접근 방식보다 우수하고 효과적임을 보여주었습니다.



### Fairness Issues and Mitigations in (Differentially Private) Socio-demographic Data Processes (https://arxiv.org/abs/2408.08471)
- **What's New**: 본 논문은 정책 결정 및 자원 배분에 중요한 사회-인구통계 데이터를 수집하는 과정에서 샘플링 기법이 미치는 영향을 분석합니다. 특히, 샘플링 오류가 계층적 집단 수준 추정에 불균형적으로 영향을 미치며, 공정성을 저해하는 문제를 다루고 있습니다.

- **Technical Details**: 이 논문은 실제 조사 설계 과정에 기반한 최적화 접근법을 도입하여, 샘플링 비용을 최적화하고 오류 경계가 규정된 허용오차 내에 있도록 보장합니다. 또한, differential privacy (차등 개인정보보호) 기술의 영향을 분석하여, 샘플링 과정에 알림으로써 그룹 수준의 통계 신뢰성을 저해하지 않는다는 점을 밝힙니다.

- **Performance Highlights**: 분석 결과, 차등 개인정보보호의 긍정적인 효과로 인해 소수 그룹에 대한 대표성이 향상되며 불공정성이 감소할 수 있다는 놀라운 발견을 하였습니다. 이를 통해, 정책 결정에서의 공정성을 증진할 수 있는 새로운 방법론을 제시합니다.



### Context-Aware Assistant Selection for Improved Inference Acceleration with Large Language Models (https://arxiv.org/abs/2408.08470)
Comments:
          14 pages (9 pages main content + references + appendix)

- **What's New**: 본 논문에서는 여러 개의 드래프트 모델을 활용하여 대형 언어 모델(LLM)의 자동 회귀적 생성(auto-regressive generation) 과정에서의 지연(latency) 문제를 개선하는 방법을 제안합니다. 특히, 각 드래프트 모델이 LLM의 다양한 전문 영역을 커버할 수 있도록 동적으로 선택하는 정책을 학습하는 방법을 다룬 점이 중요합니다.

- **Technical Details**: 연구진은 이 문제를 컨텍스트 밴딧(contextual bandit) 문제로 정의하였으며, 각 드래프트 모델에 대한 선택이 쿼리(q)에 따라 최적의 모델을 선택하게 됩니다. 이를 통해 드래프트 모델 사이의 출력 유사성을 기반으로 하여 오프라인에서 정책을 교육하고, 이를 활용해 하드웨어 요구사항과 런타임 성능을 개선하게 됩니다. 또한, 오프라인 훈련을 통해 각 드래프트와 타겟 모델 간의 정렬(alignment) 기반으로 정책을 학습합니다.

- **Performance Highlights**: 본 연구는 다양한 설정에서 드래프트 모델이 다수인 경우에도 유연성을 제공하며, 성능 개선을 위한 기존 방법들 대비 추가 비용 없이 더 나은 성능을 발휘할 수 있음을 보여줍니다. 이는 실제 환경에서 드래프트 모델의 성능 저하 없이 효율적인 추론(inference) 속도 개선을 가능하게 합니다.



### Efficient Data-Sketches and Fine-Tuning for Early Detection of Distributional Drift in Medical Imaging (https://arxiv.org/abs/2408.08456)
- **What's New**: 이번 논문은 CT 스캔 의료 이미지에서의 distributional drift(배포 드리프트) 감지를 위한 정교하고 민감한 접근 방식을 제시합니다. 데이터 스케치(data-sketching)와 파인튜닝(fine-tuning) 기술을 활용하여, 모델의 정확도와 효율성을 극대화하는 방법을 소개합니다.

- **Technical Details**: 연구에서는 Vision Transformer(ViT) 모델을 파인튜닝하여 유방암 이미지로부터 유의미한 특징을 추출하였으며, 사전 훈련된 모델을 기반으로 커스터마이즈한 아키텍처를 사용하여 특징 추출을 수행했습니다. 데이터 스케칭 기법으로는 MinHash를 활용하여 이미지 데이터를 처리하고, Kolmogorov-Smirnov 및 cosine similarity 점수를 통해 드리프트를 감지합니다. 또한 1%의 Salt-and-Pepper와 Speckle 노이즈에도 높은 민감성을 보였습니다.

- **Performance Highlights**: 모델의 정확도가 99.11%로 향상되었으며, 데이터 스케치 기법을 통한 새로운 요소들의 심사에서 50%에서 100%로 비슷한 데이터셋 간의 유사성 점수가 크게 개선되었습니다. 결과적으로, 제안된 방법은 동적인 임상 환경에서 진단 모델의 정확성을 유지하기 위한 확장 가능하고 신뢰성 높은 솔루션을 제공합니다.



### SpectralEarth: Training Hyperspectral Foundation Models at Sca (https://arxiv.org/abs/2408.08447)
- **What's New**: 본 논문에서는 SpectralEarth라는 대규모 다중 시계열 hyperspectral 데이터셋을 소개합니다. 이 데이터셋은 지구 관측 임무에서 수집된 데이터를 기반으로 하여 구성되었으며, 538,974개의 이미지를 포함하고 있습니다. 이를 통해 hyperspectral foundation 모델의 사전 학습을 가능하게 하였습니다.

- **Technical Details**: SpectralEarth는 415,153개의 고유한 위치를 커버하고 있으며, 11,636개의 EnMAP 장면에서 수집된 데이터를 포함합니다. 약 17.5%의 위치는 여러 개의 타임스탬프를 포함하여 다중 시계열 HSI 분석을 지원합니다. 또한, Self-Supervised Learning (SSL) 알고리즘을 통해 클래식 비전 백본에 스펙트럼 어댑터를 통합하여 HSI의 고유한 특성을 반영하였습니다.

- **Performance Highlights**: 우리의 실험 결과는 모델의 다양한 태스크 및 센서에 대한 일반화 가능성을 보여줍니다. 더욱이, 해당 모델은 fine-tuning 시 빠른 수렴 속도를 보여주며, 데이터셋과 모델, 소스 코드는 공개될 예정입니다.



### W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering (https://arxiv.org/abs/2408.08444)
- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)가 OpenQA(오픈 도메인 질문 응답) 작업에서 제한된 사실 정보를 생성하는 문제를 해결하기 위해 W-RAG라는 새로운 접근 방식을 제안합니다. W-RAG는 LLM의 랭킹 기능을 활용하여 약한 레이블 데이터를 생성하고, 이를 통해 dense retrievers(조밀 검색기)를 훈련합니다.

- **Technical Details**: W-RAG는 BM25를 통해 상위 K개의 구문을 검색하고, 각 구문이 정확한 답변을 생성할 확률에 따라 재랭크합니다. 그 후 가장 높은 순위를 가진 구문만을 긍정적인 훈련 샘플로 선택하고, 이를 바탕으로 dense retrievers를 훈련합니다. 이 과정은 question-passage(질문-구문) 쌍의 관련성을 평가하며, OpenQA를 위한 적합한 방법론으로 입증되었습니다.

- **Performance Highlights**: 저자들은 4개의 공개 OpenQA 데이터 세트에서 W-RAG의 성능을 평가하였으며, 결과적으로 이 방법이 기존 모델보다 retrieval(검색) 및 OpenQA 성능을 모두 향상시키는 것을 보여주었습니다.



### PQV-Mobile: A Combined Pruning and Quantization Toolkit to Optimize Vision Transformers for Mobile Applications (https://arxiv.org/abs/2408.08437)
- **What's New**: 본 논문은 PQV-Mobile이라는 새로운 도구를 소개하며, 이는 비전 트랜스포머를 모바일 환경에 최적화하기 위해 가지치기(pruning)와 양자화(quantization)를 결합한 것입니다.

- **Technical Details**: PQV-Mobile은 중요도에 따라 다양한 구조적 가지치기를 지원하며(magnitude importance, Taylor importance, Hessian importance), FP32에서 FP16 및 int8으로의 양자화를 지원합니다. 이 도구는 x86, FBGEMM, QNNPACK, ONEDNN과 같은 여러 하드웨어 백엔드를 대상으로 최적화된 모델을 제작할 수 있습니다.

- **Performance Highlights**: DeiT 모델을 9.375% 가지치기하고 FP32에서 int8로 양자화할 경우 7.18배의 지연(latency) 감소와 2.24%의 미세한 정확도 손실을 보였습니다. 또한 PQV-Mobile을 통해 구조적 가지치기를 통해 최적화된 모델의 메모리 및 성능 성능을 검증했습니다.



### Predictive uncertainty estimation in deep learning for lung carcinoma classification in digital pathology under real dataset shifts (https://arxiv.org/abs/2408.08432)
Comments:
          17 pages, 2 figures, 5 tables

- **What's New**: 본 논문은 딥 러닝 기반의 진단 의사결정 시스템에서 예측 불확실성(uncertainty) 추정이 강건성과 성능에 미치는 영향을 평가합니다. 특히, 폐선암(lung carcinoma) 분류 문제에 대해 다양한 데이터 분포 이동(distribution shift) 시나리오를 살펴보았습니다.

- **Technical Details**: 주요 방법으로 Monte Carlo dropout, deep ensemble, few-shot learning을 사용하여 전국적인 곤충 관찰을 통해 예측 불확실성을 개선하고자 했습니다. 연구는 내부 및 외부의 데이터 분포 이동을 포함한 다양한 시나리오에서 이루어졌습니다.

- **Performance Highlights**: 본 연구는 폐 선암의 다양한 하위 유형과 특성 분석 데이터를 포함한 임상적으로 관련된 분포 이동 아래에서 제안된 방법들의 효과를 비교 분석하였습니다. 비교 결과, deep ensemble 방법이 가장 우수한 성능을 발휘하였고, 두 번째로 MC-dropout이 좋은 결과를 보였음을 확인했습니다.



### Assessing and Enhancing Large Language Models in Rare Disease Question-answering (https://arxiv.org/abs/2408.08422)
- **What's New**: 이 논문은 희귀 질병 진단에서 대형 언어 모델(LLMs)의 성능을 평가하고, 이러한 모델들이 효과적으로 진단할 수 있도록 개선할 방법을 탐구합니다. 특히, ReDis-QA라는 새로운 데이터셋을 도입하여 LLMs의 진단 성능을 평가합니다.

- **Technical Details**: ReDis-QA 데이터셋은 205개의 희귀 질병에 대한 1360개의 고품질 질문-답변 쌍을 포함하고 있으며, 각 질문에 대한 메타 데이터를 주석 처리하여 특정 질병 및 그 속성에 대한 하위 집합을 추출할 수 있게 합니다. ReCOP라는 첫 번째 희귀 질병 데이터베이스를 사용하여 LLM의 성능을 향상시키기 위해 기존 문헌에 기반한 신뢰할 수 있는 답변과 설명 생성을 유도합니다.

- **Performance Highlights**: 실험 결과, ReCOP는 ReDis-QA 데이터셋에서 LLM의 정확성을 평균 8% 향상시킵니다. 또한 LLM이 기존 문헌에 기반한 신뢰도 높은 답변과 설명을 생성하도록 성공적으로 안내합니다.



### Understanding Help-Seeking Behavior of Students Using LLMs vs. Web Search for Writing SQL Queries (https://arxiv.org/abs/2408.08401)
- **What's New**: 대규모 언어 모델(LLMs)의 사용이 SQL 쿼리 작성에 미치는 영향을 비교한 연구 결과가 발표되었습니다. 전통적인 웹 검색과 LLM의 사용 차이를 조사한 무작위 인터뷰 연구를 통해 학습 지원 도구 설계에 대한 통찰을 제공합니다.

- **Technical Details**: 39명의 학생을 대상으로 진행된 이 연구에서는 Google, Bing과 같은 전통적 웹 검색 방식을 표준 ChatGPT(3.5 모델) 및 강사가 조정한 LLM과 비교하였습니다. 연구 질문은 전통적 웹 검색과 LLM 기반 챗봇이 학습 결과에 미치는 영향과 강사가 시스템 프롬프트 추가와 같은 저렴한 조정 방법을 사용하여 LLM의 효과를 향상시킬 수 있는지를 다루었습니다.

- **Performance Highlights**: 연구 결과, 강사가 조정한 LLM은 ChatGPT 및 웹 검색에 비해 두 배 이상의 상호작용이 필요했지만, 최종 SQL 쿼리의 품질은 비슷한 것으로 나타났습니다. 동시에, 강사가 조정한 LLM을 사용하는 학생들은 더 낮은 정신적 부담을 보고했습니다.



### Decoding the human brain tissue response to radiofrequency excitation using a biophysical-model-free deep MRI on a chip framework (https://arxiv.org/abs/2408.08376)
Comments:
          This project was funded by the European Union (ERC, BabyMagnet, project no. 101115639). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them

- **What's New**: 이번 연구에서는 MRI(자기공명영상)를 위한 새로운 심층 학습 프레임워크인 DeepMonC를 개발하여, RF(라디오 주파수) 자극에 대한 뇌 조직의 반응을 신속하게 해석하고 다양한 영상 대비를 생성하는 방법을 제시했습니다. 이 프레임워크는 개인 맞춤형 캘리브레이션 스캔(28.2초) 후, 자동으로 몇 가지 생리학적 매개 변수를 포함한 다양한 이미지 대비를 생성하는 기능을 갖추고 있습니다.

- **Technical Details**: DeepMonC는 비전 트랜스포머(vision transformer) 기반으로 설계되었으며, RF 자극 정보와 실제 조직 반응 이미지로 구성된 이중 도메인 입력을 포함합니다. 이 시스템은 추가 입력 없이 3D 뇌를 통한 6개의 생리학 매개 변수를 정량화하는 확장 모듈도 포함하고 있습니다. 훈련 데이터는 9명의 건강한 자원자들로부터 수집된 3,118,692개의 이미지 및 캘리브레이션 매개 변수 쌍으로 구성되었으며, DeepMonC는 서로 다른 4명의 피험자를 통해 검증되었습니다.

- **Performance Highlights**: DeepMonC는 기존 프로토콜에 비해 94% 더 빠른 속도로 영상 생성이 가능하며, 다양한 이미지의 구조적 유사성을 평가한 결과 SSIM(구조적 유사도 측정 지수) > 0.96, PSNR(최대 신호 대 잡음비) > 36, NRMSE(정규화 평균 제곱 오차) < 3%를 기록했습니다. 전체 뇌의 6개 또는 24개 미지의 이미지 대비를 재구성하는 데 소요된 시간은 GPU(Nvidia RTX 3060) 기준으로 각각 7.674초와 10.896초로 나타났습니다.



### API-guided Dataset Synthesis to Finetune Large Code Models (https://arxiv.org/abs/2408.08343)
- **What's New**: DataScope라는 새로운 API 기반 데이터셋 합성 프레임워크가 제안되었으며, 이는 Large Code Models(LCMs)의 세밀한 조정을 위한 고품질 SFT(정밀 지도학습) 데이터셋을 효과적으로 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DataScope는 두 가지 주요 구성 요소인 Dsel과 Dgen으로 구성됩니다. Dsel은 API 커버리지를 핵심 지표로 사용하여 고품질의 데이터셋을 효율적으로 선택합니다. Dgen은 도메인 데이터셋 합성을 고급 API 기능과 코드 스켈레톤을 사용하여 구체적인 코드를 합성하는 과정으로 재구성합니다.

- **Performance Highlights**: 실험 결과, DataScope의 합성 데이터셋으로 세밀 조정된 모델은 비최적화 데이터셋으로 조정된 모델보다 성능이 최대 5배 뛰어나다는 것을 보여주었으며, 비용적인 측면에서도 도메인 특화된 SFT 데이터셋을 매우 저렴한 비용으로 생성할 수 있다는 점이 강조되었습니다.



### Exploring Latent Space for Generating Peptide Analogs Using Protein Language Models (https://arxiv.org/abs/2408.08341)
- **What's New**: 이 연구는 전통적인 데이터셋의 필요성을 줄이고, 단일 서열을 이용하여 새로운 펩타이드 유사체(peptide analogs)를 생성하는 혁신적인 방법을 제안합니다. 이는 autoencoder 형태의 모델을 사용하여 단백질 임베딩 공간(protein embedding space)을 탐색함으로써 이루어집니다.

- **Technical Details**: 제안된 방법은 펩타이드 서열을 연속적인 잠재(space)로 변환하고, 그 공간에서 노이즈를 추가하여 유사한 펩타이드 구조를 생성합니다. 또한, 이 과정에서 ProtT5 및 ESM 모델을 활용하여 펩타이드 서열 임베딩을 진행하고, 디코딩 과정을 통해 다시 펩타이드 서열로 변환합니다.

- **Performance Highlights**: 제안된 방법은 TIGIT 억제제에 대한 분자 동역학(Molecular Dynamics) 시뮬레이션을 통해 검증되었으며, 기존 모델에 비해 펩타이드 구조 유사성 및 생물활성(bioactivity) 지표에서 유의미한 개선 사항을 보여 주었습니다.



### Graph representations of 3D data for machine learning (https://arxiv.org/abs/2408.08336)
Comments:
          14 pages, 11 figures

- **What's New**: 이 연구는 3D 데이터를 분석에서 사용 가능한 조합적(combinatorial) 방법들에 대한 개요를 제공하며, 다양한 표현 방식의 장단점 및 표현 간 전환 방법에 대해 논의합니다. 또한 생명과학 및 산업 분야의 두 가지 구체적인 응용 사례를 제시합니다.

- **Technical Details**: 3D 데이터는 과학 및 산업에서 자연스럽게 발생하며, 생물영상(bioimaging), 분자 화학(molecular chemistry), 3D 모델링과 설계(plan) 등 다양한 분야를 포함합니다. 3D 데이터를 분석하는 가장 일반적인 방법은 voxel(3D 픽셀)의 표현을 사용하는 것이며, 이로 인해 계산 복잡도가 증가합니다. 본 논문은 mesh, point cloud, graph와 같은 경량 표현을 활용해 이러한 문제를 완화할 수 있는 가능성을 탐구합니다.

- **Performance Highlights**: 조합적 표현을 사용할 경우 모델의 예측에 대한 설명 가능성을 높일 수 있으며, 그래프 신경망(Graph Neural Networks, GNN)을 통해 3D 데이터를 효과적으로 처리할 수 있는 가능성이 있습니다. GNN은 스파스한 3D 데이터에 대해 뛰어난 확장성을 보이며, 2D 데이터와 거의 동일한 속도로 처리가 가능합니다.



### Plan with Code: Comparing approaches for robust NL to DSL generation (https://arxiv.org/abs/2408.08335)
Comments:
          9 pages, 1 figure, 5 tables. arXiv admin note: substantial text overlap with arXiv:2407.02742

- **What's New**: 이 논문은 RPA (Robotic Process Automation) 도메인에서 DSL (Domain Specific Languages) 생성을 위해 Retrieval Augmented Generation (RAG) 방법론을 최적화하여 LLM (Large Language Model)을 사용하는 방안을 제시합니다.

- **Technical Details**: NL2DSL (Natural Language to Domain Specific Language) 생성을 위한 새로운 시스템 아키텍처를 개발하고, 기존 코드 생성 방법의 환각 및 구문 오류 문제를 해결하기 위한 효율성을 평가합니다. Codex 모델을 기반으로 LoRA 기반의 미세 조정 방식을 통해 67,000개의 NL-DSL 샘플로 훈련했습니다. RAG 기술을 적용하여 DSL 생성을 위한 grounding 이슈를 해결했습니다.

- **Performance Highlights**: 최적화된 RAG 접근 방식이 특정 도메인 API 이름의 품질을 일치시키면서, 미세 조정된 모델보다 7포인트 향상된 유사도 지표를 기록했습니다. 이는 비도메인 또는 보지 않은 API 이름의 경우에도 더욱 두드러집니다.



### CodeMirage: Hallucinations in Code Generated by Large Language Models (https://arxiv.org/abs/2408.08333)
Comments:
          Accepted at AutoMates @ IJCAI 2024

- **What's New**: 이 논문은 LLM(대규모 언어 모델)에서 생성된 코드의 환각(hallucination) 현상을 최초로 연구한 것이며, 코드 환각의 정의와 포괄적인 세분류(taxonomy)를 제공한다.

- **Technical Details**: 코드 환각은 LLM이 생성하는 코드에서 구문적(syntactical) 및 논리적(logical) 오류, 보안 취약점(security vulnerabilities), 메모리 누수(memory leaks)와 같은 고급 문제를 포함한다. 코드 환각을 탐지하기 위해, 우리는 CodeMirage라는 벤치마크 데이터셋을 제안하고, OpenAI의 GPT-3.5 및 GPT-4와 오픈 소스 LLM인 CodeLLaMA를 통해 실험하였다. 실험을 통해 GPT-4가 HumanEval 데이터셋에서 가장 우수한 성과를 보였다.

- **Performance Highlights**: LLM의 코드 환각 탐지 작업의 성능에 대한 여러 기초선(baseline)을 도입했으며, LLMs가 다양한 환각 유형을 탐지하는 데 있어 합리적인 성과를 보여주었다. 특히 GPT-4는 HumanEval 데이터셋에서 최고의 성능을 보였고, MBPP 데이터셋에서도 Fine-tuned CodeBERT와 비슷한 결과를 도출하였다.



### First Analysis of the EU Artifical Intelligence Act: Towards a Global Standard for Trustworthy AI? (https://arxiv.org/abs/2408.08318)
Comments:
          in French language

- **What's New**: 2024년 8월 1일, EU 인공지능법(AI Act)이 유럽연합(EU)에서 시행됩니다. 이 법안은 인공지능 기술의 중심에 있는 시민과 내부 시장에서 활동 중인 산업 모두에게 중요한 법적 기반을 제공합니다.

- **Technical Details**: AI Act는 EU에서 마케팅되고 사용되는 인공지능 시스템 및 모델의 글로벌 가치 사슬에 관여하는 민간 및 공공 조직에 점진적인 규정 준수를 요구합니다. 이는 국제적으로 수직적으로 모든 관련 주체에 적용되는 독특한 규제 범위를 제공하며, 신뢰할 수 있는 AI를 지원하기 위해 글로벌한 매력을 지니고 있습니다.

- **Performance Highlights**: 이 법안의 국제적인 도입은 신뢰할 수 있는 AI 지원의 주요 과제가 될 것으로 예상되며, 이러한 법적 제도가 AI 기술의 안전성과 신뢰성을 증가시킬 것으로 기대됩니다.



### Segment Anything for Videos: A Systematic Survey (https://arxiv.org/abs/2408.08315)
Comments:
this https URL

- **What's New**: 최근 Foundation 모델이 이미지 처리 및 비디오 도메인에서 기존 Paradigm에 도전하고 있으며, Segment Anything 모델(SAM)의 발전이 특히 주목받고 있습니다. SAM 2의 출시로 인해 Promptable Visual Segmentation에 대한 연구 열풍이 다시 불고 있습니다.

- **Technical Details**: 본 논문에서는 SAM의 비디오 응용에 대한 체계적인 리뷰를 제공하며, 비디오 이해(video understanding), 비디오 생성(video generation), 비디오 편집(video editing)의 세 가지 주요 분야로 기존 연구를 분류하여 장단점을 분석합니다. SAM은 11백만 이미지에 대해 10억 개 이상의 마스크로 훈련되어 다양한 프롬프트 기반으로 고품질 세그멘테이션을 가능하게 하며, 특히 SAM 2는 Streaming memory가 통합된 Transformer 프레임워크로 비디오 세그멘테이션 성능을 크게 향상시킨 사례입니다.

- **Performance Highlights**: SAM 기반 방법이 현재 상태의 최첨단 방법(SOTA)과 비교하여 대표적인 벤치마크에서 놀라운 성능을 보여준다는 점을 강조하며, 기존 방법의 장점과 한계에 대한 깊이 있는 분석을 제공합니다.




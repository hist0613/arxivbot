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




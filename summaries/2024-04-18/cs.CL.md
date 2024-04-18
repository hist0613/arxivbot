### Related Work and Citation Text Generation: A Survey (https://arxiv.org/abs/2404.11588)
- **What's New**: 이 연구는 자동 관련 작업 생성(automatic related work generation, RWG)에 초점을 맞춥니다. RWG 작업의 역사적 작업을 조사하고 기존 접근법을 요약하며, SOTA(State-of-the-art) 자연어 처리(Natural Language Processing, NLP) 모델의 능력을 검토하는 데 이 작업을 활용합니다. 최근 대형 언어 모델(Large Language Models, LLMs)의 성공으로 RWG에 대한 관심이 새롭게 증폭되었습니다.

- **Technical Details**: RWG는 연구 논문에서 '관련 작업' 섹션을 자동으로 생성하는 과제입니다. 이 작업은 검색 증강 생성(retrieval-augmented generation), 장문 이해(long document understanding), 조회 중심 다문서 요약(query-focused multi-document summarization)과 같은 다양한 NLP 하위 작업을 포함합니다. RWG 작업은 초기에는 기본적인 규칙 기반에서 시작되어 추출적 요약(extractive summarization)을 거쳐 문장 수준의 추상적 요약(abstractive summarization)으로 발전했으며, 현재는 섹션 수준(section-level)에서의 추상적 생성으로 발전하고 있습니다.

- **Performance Highlights**: 현재 RWG 작업은 다양한 접근 방식과 정의에도 불구하고 표준 벤치마크 데이터 세트가 없어 서로 비교하기 어렵습니다. 대부분의 RWG 연구는 맞춤형 전처리를 적용하여 관련 작업 섹션 또는 개별 인용문을 추출하며, 이는 해당 작업 정의의 차이를 반영합니다. 또한 많은 작업은 모델이나 생성된 결과를 공개하지 않아 후속 작업에서 이전 접근법과 비교하는 것이 불가능합니다.



### Quantifying Multilingual Performance of Large Language Models Across  Languages (https://arxiv.org/abs/2404.11553)
- **What's New**: 이 연구에서는 저자들이 다국적 언어능력을 정량적으로 평가하고자 'Language Ranker'라는 기법을 제안하였습니다. 이러한 새로운 방법은 Large Language Models (LLMs)의 다양한 언어에 대한 성능을 벤치마킹하고 순위를 매기는 데 사용됩니다. 특히 저소스 언어(low-resource languages)에 대한 성능 측정에 초점을 맞추어 LLM의 언어적 편향성을 해결하려는 시도입니다.

- **Technical Details**: 이 방법론에서는 우선 영어 코퍼스(English corpus)를 기준으로 삼아 LLM이 다양한 언어의 코퍼스와 얼마나 유사한지 측정합니다. 이를 위해 코사인 유사도(cosine similarity)를 사용하는데, 이는 영어와 다른 언어 간의 표현 간 유사성을 계산하여 각 언어에서의 모델 성능 점수를 구합니다. OPUS-100 데이터셋을 사용하여 이 과정을 실험했으며, 연구 결과는 LLM의 사전 훈련(pre-training) 코퍼스에서 다양한 언어 비율의 순위와 유사함을 보여줍니다.

- **Performance Highlights**: 3가지 주요 발견은 다음과 같습니다: (1) 다양한 LLM에서 모든 언어의 성능 순위는 대체로 일치했습니다. (2) 다양한 크기의 LLM에서도 성능의 부분 순서가 동일했습니다. (3) LlaMa2의 다양한 언어 성능과 사전 훈련 코퍼스의 비율 사이에는 강한 상관관계가 있었습니다. 이러한 결과는 Language Ranker를 활용하여 LLM의 언어 성능을 효과적으로 측정할 수 있음을 시사합니다.



### Evaluating Span Extraction in Generative Paradigm: A Reflection on  Aspect-Based Sentiment Analysis (https://arxiv.org/abs/2404.11539)
Comments: 10 pages

- **What's New**: 본 논문은 성장하는 생성 언어 모델(GLM)의 시대에서 관점 기반 감성 분석(Aspect-Based Sentiment Analysis, ABSA) 평가 방법론을 재고하고 새로운 도전 과제를 다룬다. 특히, ABSA에서의 종래 경계가 흐려지면서 이해 및 생성 작업 간의 전통적 경계가 모호해지고 있는 문제에 초점을 맞춘다.

- **Technical Details**: 이 논문은 ABSA에서의 성능 평가에 대한 종합적인 지침을 제안하는 것을 목표로 한다. 생성 언어 모델을 활용하여 종합적인 평가 지침을 개발하는 데 중점을 두고, 다양한 평가 지표와 비교 분석을 진행한다. ABSA의 'Aspect Sentiment Quad Prediction (ASQP)' 설정을 기본으로 하여, 문서에서 'aspect term', 'aspect category', 'opinion term', 'sentiment polarity' 등의 요소를 추출 및 분류하는 방안을 논한다.

- **Performance Highlights**: 논문에서는 기존의 정확도 측정(Exact Match, F1 Score)과 더불어, 예측과 실제 데이터의 유사성을 평가하는 새로운 방법을 도입하고자 한다. 특히, 생성 언어 모델을 통한 ABSA 과제의 복잡성을 고려하여 다양한 도메인(Book, Clothing, Hotel 등)에서 얻은 데이터를 비교 분석하며, 이러한 다양성이 어떻게 평가 절차에 영향을 미치는지에 대해 논한다.



### Select and Reorder: A Novel Approach for Neural Sign Language Production (https://arxiv.org/abs/2404.11532)
Comments: 8 Pages, 5 Figures, 7 Tables, LREC-COLING 2024

- **What's New**: 이 논문은 저자원 언어인 수화의 정확한 번역을 위한 새로운 접근법인 Select and Reorder (S&R)를 소개합니다. 이 접근법은 번역 과정을 두 단계로 나누며, 첫 번째 단계인 Gloss Selection (GS)은 주어진 말소리 문장의 글로스를 선택하고, 두 번째 단계인 Gloss Reordering (GR)은 선택된 글로스를 수화의 문법에 맞게 재배열합니다. 이 두 단계는 Non-AutoRegressive (NAR) 디코딩을 사용하여 계산 요구사항을 줄이고 추론 속도를 높입니다.

- **Technical Details**: S&R 방법론은 큰 말소리 언어 모델과 말소리 언어와 대상 수화 언어 간의 상당한 어휘적 중복을 활용하여 초기 정렬을 설정합니다. GS 단계는 각 단어에 대한 해당 글로스를 예측하고, GR 단계는 글로스 시퀀스를 수화 순서로 변경합니다. 이 과정에서 통계 기반 사전 재정렬 방법과 딥러닝 접근 방법을 탐구합니다. 딥러닝 접근법은 추론 시간에 재정렬 마스크를 사용하는 변형자 (transformer)를 사용합니다.

- **Performance Highlights**: 이 접근법은 Meine DGS Annotated (mDGS) 데이터셋에서 최고의 BLEU와 Rouge 점수를 달성하며, Text to Gloss (T2G) 번역에서 37.88%의 BLEU-1 개선을 보였습니다. 이러한 성과는 수화 번역 모델의 효과성을 향상시키는 길을 연다고 할 수 있습니다.



### Pack of LLMs: Model Fusion at Test-Time via Perplexity Optimization (https://arxiv.org/abs/2404.11531)
- **What's New**: 이 연구에서는 'Pack of LLMs (PackLLM)'이라는 새로운 방법을 도입하였습니다. 이는 여러 대규모 언어 모델(Large Language Models, LLMs)을 융합함으로써 입력 프롬프트에 대한 각 LLM의 전문성(expertise)을 활용할 수 있도록 하며, 이를 통해 추론(inference) 동안에 사용자가 지정한 임의의 LLM을 활용할 수 있습니다.

- **Technical Details**: PackLLM은 입력 프롬프트에 대한 혼란도(perplexity)를 최소화하기 위해 각 LLM의 중요도(importance)를 결정하는 최적화 문제를 해결하는 모델 융합 접근 방식입니다. PackLLM-sim은 혼란도가 각 LLM의 전문성을 측정하는 데 유효한 지표임을 검증하고, PackLLM-opt는 탐욕 알고리즘(greedy algorithm)을 사용하여 혼란도 최소화 문제를 근사적으로 해결합니다.

- **Performance Highlights**: PackLLM은 100개 이상의 LLM을 사용한 다양한 작업에서 실험을 진행하였고, 이러한 실험 결과 (i) 혼란도가 LLM 융합에 있어 신뢰할 수 있는 지표임을 보여주고, (ii) PackLLM이 테스트 시점 융합 방식(test-time fusion baselines)을 1.89%의 정확도로 능가했으며, (iii) PackLLM이 학습 기반의 융합 접근법(learning-based fusion approaches) 대비 3.92-11.94% 더 높은 정확도 향상을 실현했습니다.



### Towards Coarse-to-Fine Evaluation of Inference Efficiency for Large  Language Models (https://arxiv.org/abs/2404.11502)
- **What's New**: 이 연구에서는 대규모 언어 모델(Large Language Models, LLMs)의 추론 효율성을 다각적으로 분석하고 평가합니다. 라이브러리 간의 성능을 비교하고 추론 전략을 개선할 수 있도록 돕는 종합 벤치마크를 제안합니다. 이 연구는 추론 효율성을 향상시키기 위한 새로운 기준을 제시하면서, 향후 추론 알고리즘 및 라이브러리 개발의 기초를 마련합니다.

- **Technical Details**: 이 논문은 Transformer 구조에서 각 모듈의 복잡성을 미세하게 분석하고, MHA(Multi-Head Attention) 모듈과 FFN(Feed-Forward Network) 모듈, 각각에 필요한 부동 소수점 연산(FLOPs)과 메모리 연산(MOPs) 수를 계산하여 추론 시간 분포를 이해합니다. 이론적 분석을 검증하기 위해 대표적인 라이브러리 두 개를 선택하여 시간 분석 테스트를 실시했습니다.

- **Performance Highlights**: 다양한 사용 시나리오에서의 라이브러리 성능을 평가하기 위해 네 가지 텍스트 생성 데이터셋을 사용하고, 배치 추론 및 서버 기반 추론을 포함한 두 가지 실제 적용 사례를 검토합니다. 이러한 평가를 통해 각 라이브러리의 추론 효율성을 객관적으로 평가할 수 있었습니다. 또한, 각 모듈의 연산 강도를 측정하고, GPU의 계산 능력이나 메모리 대역폭에 의해 제한되는 시나리오를 정확하게 식별합니다.



### Paraphrase and Solve: Exploring and Exploiting the Impact of Surface  Form on Mathematical Reasoning in Large Language Models (https://arxiv.org/abs/2404.11500)
Comments: Accepted to the main conference of NAACL (2024)

- **What's New**: 이 연구는 수학 문제의 표면 형태(surface form)와 대규모 언어 모델(large language models, LLMs)에 의한 해결 가능성 간의 관계를 조사했습니다. 연구진은 문제의 표면 형태를 미묘하게 변경하면 답변 분포와 해결률(solve rate)에 상당한 영향을 미칠 수 있으며, 이는 언어 모델이 복잡한 문제를 추론하는 데 있어 표면 형태에 대한 민감성과 견고성 부족을 드러냅니다.

- **Technical Details**: 자기 일관성(self-consistency) 설정을 이용하여 같은 수학 문제에 대해 다수의 답변을 샘플링하고, 해결률을 계산했습니다. 해결률은 올바른 답변의 비율로 정의됩니다. 연구진은 문제의 표면 형태를 다양화하기 위해 언어 모델의 다의어화 능력을 활용하고, SCoP (Self-Consistency-over-Paraphrases)를 제안합니다. SCoP는 각 수학 문제마다 언어 모델을 이용하여 여러 표면 형태를 생성하고, 각 표면 형태에 대해 답변 경로를 생성한 후 가장 일관된 답변을 선택하는 두 단계로 구성됩니다.

- **Performance Highlights**: SCoP 방법을 GSM8K, AQuA, MATH, 및 MMLU-Math 같은 네 가지 수학 추론 벤치마크(mathematics reasoning benchmarks)에 적용하여, LLaMA-2-70b, GPT-3.5-turbo, GPT-4 등 세 가지 큰 언어 모델을 통해 평가했습니다. 실험 결과, SCoP는 기존의 자기 일관성 방법보다 수학 추론 성능을 향상시켰으며, 특히 처음에 해결 불가능하다고 간주된 문제들에 대해 향상된 결과를 보였습니다.



### A Data-Driven Representation for Sign Language Production (https://arxiv.org/abs/2404.11499)
Comments: 8 Pages, 3 Figures, 7 Tables, 18th IEEE International Conference on Automatic Face and Gesture Recognition 2024

- **What's New**: 이 연구에서는 수화의 연속적인 자세 생성 문제를 이산적인 시퀀스 생성 문제로 변환하여 비용이 많이 드는 어노테이션(Annotation)의 필요성을 극복하려는 새로운 접근법을 소개합니다. 벡터 양자화(Vector Quantisation, VQ)를 사용하여 수화 데이터에서 동작의 코드북을 학습하고, 트랜스포머(Transformer)를 사용하여 말하는 언어의 텍스트를 코드북 토큰의 시퀀스로 변환합니다. 또한, 이러한 토큰들을 자연스러운 수화 시퀀스로 조합할 수 있는 수화 스티칭(Sign Stitching) 방법을 제시합니다.

- **Technical Details**: 저자들은 연속적인 3D 자세 데이터에서 모션의 코드북을 학습하기 위해 노이즈 대치 벡터 양자화(NSVQ, Noise Substitution Vector Quantisation) 모델을 사용합니다. 이 코드북은 새로운 표현의 어휘로 간주될 수 있으며, 연속 자세 시퀀스를 이산 코드 시퀀스로 토크나이즈(Tokenize)하는데 사용됩니다. 또한, 트랜스포머를 이용한 시퀀스-투-시퀀스(sequence-to-sequence) 방식으로 문장을 수화 코드북 토큰의 시퀀스로 변환하고, 이를 바로 자세 시퀀스로 매핑(mapping)할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 RWTH-PHOENIX-Weather-2014T(PHOENIX14T)와 더 도전적인 Meine DGS Annotated(mDGS) 데이터셋에서 기존 방법들보다 우수한 성능을 보여, 최대 72%까지 BLEU-1 백 번역 점수를 향상시켰습니다. 이는 과거의 다양한 접근법들과 비교하여 상당한 개선을 나타냅니다.



### A Federated Learning Approach to Privacy Preserving Offensive Language  Identification (https://arxiv.org/abs/2404.11470)
Comments: Accepted to TRAC 2024 (Fourth Workshop on Threat, Aggression and Cyberbullying) at LREC-COLING 2024 (The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation)

- **What's New**: 온라인에서의 공격적 언어의 식별에 Federated Learning (FL) 도입을 통해 개인 정보 보호를 강화하는 새로운 접근 방식이 제안되었습니다. 이 연구는 공격적 언어 식별을 위해 다중 모델을 결합하는 FL 모델 융합 기술을 탐구합니다. 데이터의 중앙 집중식 저장 없이 로컬에서 모델을 훈련할 수 있으므로 사용자의 개인 정보 보호가 가능합니다.

- **Technical Details**: 연구팀은 Federated Learning을 사용하여 각 클라이언트가 데이터를 공유하지 않고도 로컬에서 모델을 훈련할 수 있는 분산 아키텍처를 사용합니다. AHSD, HASOC, HateXplain, OLID와 같은 사전에 공개된 영어 벤치마크 데이터셋에서 다양한 딥러닝 모델을 훈련시키고, 성능을 상세히 평가했습니다. 또한, 영어와 스페인어를 포함한 초기 교차 언어 실험도 제시했습니다. 모델은 모델 융합(model fusion) 접근 방식을 사용하여 기준 모델(baselines)을 모든 데이터셋에서 능가하는 성능을 보여주었습니다.

- **Performance Highlights**: 제안된 모델 융합 방식은 개인 정보 보호를 유지하면서 AHSD, HASOC, HateXplain, OLID 데이터셋에서 기준 모델보다 우수한 성능을 보였습니다. 연구팀은 FL을 활용하여 공격적 언어 식별에서 높은 성능을 달성하고, 이를 통해 사용자 데이터의 개인 정보 보호와 함께 정확한 언어 식별이 가능함을 입증했습니다.



### Octopus v3: Technical Report for On-device Sub-billion Multimodal AI  Agen (https://arxiv.org/abs/2404.11459)
- **What's New**: 이 논문에서는 AI 에이전트 애플리케이션을 위해 특별히 설계된 기능적 토큰(functional token) 개념을 포함하는 다중모달(multimodal) 모델을 도입했습니다. 이 모델은 엣지 장치(edge devices)와의 호환성을 보장하기 위해 1B 파라미터 미만의 컴팩트한 크기로 최적화되었습니다. 모델은 다양한 엣지 장치에서 효율적으로 작동할 수 있는 능력을 시연하였으며, Raspberry Pi와 같은 제한된 리소스를 가진 장치에서도 작동합니다.

- **Technical Details**: 이 모델은 자연어와 이미지 정보를 통합하여 예측 조치(predictive actions)를 최적화하는 역할을 하는 다중모달 언어 모델(multimodal language model)을 기반으로 합니다. 특히, 이미지 정보를 텍스트 입력과 통합하는 데 사용되는 기술 중 하나는 OpenAI의 CLIP 모델을 기반으로 하는 이미지 인코딩 방법입니다. 또한, 자연어 및 이미지에 적용되는 토큰화(tokenization)처럼, 특정 기능을 기능적 토큰으로 캡슐화하는 새로운 훈련 전략을 도입하여 다중모달 AI 에이전트 개발에 활용됩니다.

- **Performance Highlights**: 이 모델은 Raspberry Pi와 같은 자원 제한적인 장치에서도 효율적으로 작동하는 능력을 시연함으로써, IoT 장치나 스마트폰 같은 엣지 장치에서의 광범위한 응용 프로그램과 사용 사례를 가능하게 합니다. 또한, 이 모델은 중요한 의료 진단(medical diagnosis)이나 자율 주행(autonomous navigation)과 같이 더 복잡한 작업을 수행할 수 있는 잠재력을 보여줍니다.



### AI-Enhanced Cognitive Behavioral Therapy: Deep Learning and Large  Language Models for Extracting Cognitive Pathways from Social Media Texts (https://arxiv.org/abs/2404.11449)
- **What's New**: 이 연구는 소셜 미디어(Social Media) 데이터를 수집하고 인지 경로(Cognitive Pathways) 추출을 위해 주석을 달고 모델을 개발했습니다. 특히 우울증(Depression)과 자살(Suicide)과 같은 주제에서 나타나는 인지 왜곡(Cognitive Distortions)을 식별하기 위해 인공지능(AI)과 딥 러닝(Deep Learning)을 활용하여 계층적 텍스트 분류(Hierarchical Text Classification)와 텍스트 요약(Text Summarization) 작업을 개발하였습니다.

- **Technical Details**: 연구자들은 네 개의 주요 카테고리와 19개의 하위 카테고리로 텍스트를 분류하는 계층적 텍스트 분류 작업을 설정하고, 신속한 정보 접근을 위한 텍스트 요약 작업을 구조화했습니다. 기존 딥 러닝 모델과 비교하여 GPT-4(GPT-4)를 사용하여 텍스트 요약 작업에서 우수한 성능을 보였으며, Rouge-1 및 Rouge-2 점수가 각각 54.92와 30.86으로 높게 나타났습니다.

- **Performance Highlights**: 이 연구의 주요 성과로는 계층적 텍스트 분류에서 딥 러닝 방법이 62.34%의 미세 F1 점수(Micro-F1 Score)를 달성했다는 점입니다. 또한, GPT-4는 텍스트 요약 작업에서 높은 Rouge-1 및 Rouge-2 점수를 기록하며 기존의 딥 러닝 모델보다 나은 성능을 보였지만, 환각(Hallucination) 문제가 발생할 수 있다는 점도 관찰되었습니다.



### Open-Ended Wargames with Large Language Models (https://arxiv.org/abs/2404.11446)
Comments: 15 pages, 2 figures

- **What's New**: 본 연구에서는 '스노우 글로브(Snow Globe)'라 불리는 새로운 인공지능 시스템을 사용하여 질적 전쟁 게임(qualitative wargames)을 자동화하는 방법을 소개합니다. 이 시스템을 통해 시나리오 준비부터 게임 후 분석까지 모든 단계를 AI, 인간, 혹은 두 요소의 조합으로 처리할 수 있습니다.

- **Technical Details**: 스노우 글로브는 다중 에이전트 시스템(multi-agent system)으로 설계되었으며, 게임의 모더레이터 역할을 하는 제어 에이전트(control agent)와 플레이어의 응답을 시뮬레이션 하는 플레이어 에이전트(player agents), 그리고 집단 반응을 조율하는 팀 에이전트(team agent) 등이 포함됩니다. 이 시스템은 대규모 언어 모델(large language models, LLM)을 기반으로 하여 질적 전쟁 게임의 자동화를 가능하게 합니다.

- **Performance Highlights**: 스노우 글로브를 사용하여 AI 사건 대응을 주제로 한 테이블탑 연습과 지정학적 위기에 대한 정치 전쟁 게임을 시뮬레이션한 사례 연구를 진행했습니다. 이러한 시뮬레이션을 통해 질적 전쟁 게임의 다양한 가능성을 보여주고, 실제와 유사한 다양한 결과를 도출할 수 있는 가능성을 탐색하였습니다.



### Exploring Key Point Analysis with Pairwise Generation and Graph  Partitioning (https://arxiv.org/abs/2404.11384)
Comments: 11 pages, 4 figures, 4 tables. Accepted to NAACL 2024

- **What's New**: 이 연구는 Key Point Analysis (KPA)를 적용하여 다양한 주장들을 간결한 핵심 포인트로 요약하는 새로운 접근법을 제안합니다. 기존의 모델들이 의미적 유사성에만 초점을 맞춘 반면, 이 모델은 주장 간에 공유된 핵심 포인트의 존재를 측정하고, 주장들을 그래프로 구성하여 그래프 분할 알고리즘을 통해 처리합니다. 이는 KPA 과제에서 내부 클러스터(inner-cluster)와 외부 클러스터(inter-cluster) 간의 관계를 모두 고려합니다.

- **Technical Details**: 이 연구의 기술적 접근 방식은 주장 쌍(pair)에 대한 핵심 포인트 생성과 그래프 분할(graph partitioning)을 포함합니다. 주장을 정점(vertices)으로, 핵심 포인트를 간선(edges)으로, 그리고 점수를 가중치(edge weights)로 하는 인자 그래프(argument graph)를 구성하고, 핵심 포인트를 공유하는 모든 주장을 동일한 부그래프(subgraph)로 분할하는 알고리즘을 제안합니다. 이 모델은 ArgKP와 QAM 데이터셋에서 이전 모델들을 능가하는 성능을 보였습니다.

- **Performance Highlights**: 본 연구에서 제안한 모델은 기존 모델들과 비교했을 때 ArgKP와 QAM 데이터셋에서 더 높은 성능을 보였습니다. 이는 복잡한 KPA 과업에서도 효과적으로 주장들 사이의 핵심 포인트를 식별하고 요약할 수 있음을 시사합니다. 또한, 클러스터 간 관계(inter-cluster relationships)를 고려하는 점에서 기존 접근법들보다 더 정교한 분석이 가능함을 드러냈습니다.



### TeClass: A Human-Annotated Relevance-based Headline Classification and  Generation Dataset for Telugu (https://arxiv.org/abs/2404.11349)
Comments: Accepted at LREC-COLING 2024

- **What's New**: 이 논문에서는 텔루구어(Telugu) 같은 저자원(low-resource) 언어에 대한 뉴스 제목 생성의 문제를 탐구하며, 관련성 기반 뉴스 제목 분류를 통해 이 문제를 해결하고자 합니다. 연구진은 'TeClass'라는, 텔루구 언어로 된 첫 번째 인간 주석이 달린 뉴스 제목 분류 데이터셋을 제공합니다. 이 데이터셋은 78,534개의 주석이 포함된 26,178개의 기사-제목 쌍으로 구성되어 있습니다.

- **Technical Details**: 데이터셋 구축은 여러 뉴스 웹사이트에서 웹 스크래핑을 통해 수행되었으며, 사이트별 맞춤형 웹 스크래퍼를 사용하여 기사의 텍스트, 제목, 뉴스 도메인 이름을 추출했습니다. 데이터는 크게 세 가지 범주로 분류됩니다: 높은 관련성(Highly Related - HREL), 중간 관련성(Moderately Related - MREL), 낮은 관련성(Least Related - LREL). 분류는 텔루구어를 구사하는 자원봉사자들에 의해 크라우드소싱을 통해 이루어졌습니다.

- **Performance Highlights**: 분류된 데이터셋을 활용하여 여러 베이스라인 모델(baseline models)에 대한 종합적인 분석을 수행하였고, 이를 통해 관련성이 높은 기사-제목 쌍을 학습한 모델은 ROUGE-L 점수에서 약 5점의 향상을 보였습니다. 또한, 이 데이터셋과 모델은 공개되어 향후 연구를 위한 기반을 마련합니다.



### To Drop or Not to Drop? Predicting Argument Ellipsis Judgments: A Case  Study in Japanes (https://arxiv.org/abs/2404.11315)
Comments: 13 pages; accepted by LREC-COLING 2024

- **What's New**: 이 연구는 일본인 화자들의 탈락 결정에 대한 첫 번째 대규모 연구로, 일본어(prototypical pro-drop language)의 균형 잡힌 말뭉치를 사용하여 언어 모델(LM) 기반의 인자 탈락 판정 모델의 성능을 분석합니다. 이 연구는 인간의 담화 처리(discourse processing)와 글쓰기 보조(writing assistance) 도구 개발에 통찰력을 제공하는 것을 목표로 합니다.

- **Technical Details**: 연구자들은 먼저 2,000개 이상의 데이터 포인트를 포함하는 대규모 데이터 세트를 생성하여 일본어 말뭉치에서 특정 인자가 생략되어야 하는지 여부와 그 이유를 결정했습니다. 이 데이터는 언어 모델, 특히 BERT와 GPT-4를 사용하여 인자 탈락 판정 작업에서의 성능을 벤치마킹했습니다. 또한, 언어 모델이 특정 유형의 탈락 판정에서 어려움을 겪는다는 것을 발견, 인간과 LM 간의 격차를 강조합니다.

- **Performance Highlights**: 언어 모델은 일부 탈락 판정 유형에서 어려움을 겪었으며, 이는 인간 판단과의 상당한 차이를 나타냅니다. 그러나 공통된 판단 기준과 양적 특성을 명확히 하는 데 있어 원어민 간에 일반적인 합의가 있었다는 점은 긍정적인 결과입니다.



### A Preference-driven Paradigm for Enhanced Translation with Large  Language Models (https://arxiv.org/abs/2404.11288)
Comments: Accepted to NAACL 2024 (long, main)

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)의 번역 성능을 개선하기 위해 Plackett-Luce 모델을 기반으로 한 선호도 기반 접근 방식을 제안합니다. 기존의 감독 학습 방식(Supervised Fine-Tuning, SFT)은 참조 번역을 모방하는 데 그쳐 데이터의 잡음에 취약했으나, 새로운 접근 방식은 차별화된 번역 품질 이해 및 평가를 통해 번역 모델의 성능 향상을 도모합니다.

- **Technical Details**: 이론적 기초로 Bradley-Terry와 Plackett-Luce 순위 모델을 사용하여 '선호 거리'를 순위 모델에 직접 통합합니다. 대상 LLM을 최적화하기 위해 고품질 병렬 데이터에 대한 미세 조정을 통해 LLM의 번역 능력을 발휘하도록 설정하며, 이후 선호 학습을 통해 정확한 번역 생성을 우선시합니다. MAPLE이라는 새로운 데이터 세트를 구축하여 이를 통해 모델의 번역 선호도를 학습시키며, 이 데이터 세트는 각 소스 문장에 대한 다양한 품질의 번역을 포함합니다.

- **Performance Highlights**: MAPLE 데이터셋을 사용하여 선호 학습을 실시한 결과, 기존 MT 모델을 최대 3.96의 COMET 점수로 상회하는 성능을 보였습니다. 또한, 이 데이터셋은 다른 LLM들의 성능 향상에도 사용될 수 있으며, 최대 1.4M 병렬 데이터로 성능 정체(plateau)를 극복하는 데 도움을 줍니다.



### Sampling-based Pseudo-Likelihood for Membership Inference Attacks (https://arxiv.org/abs/2404.11262)
- **What's New**: 이 연구에서는 기존의 Likelihood 기반 MIA (Membership Inference Attack)를 적용할 수 없는 모델들을 대상으로 Sampling-based Pseudo-Likelihood (SPL, 샘플링 기반 사이유도-가능도)를 사용하여 데이터 유출을 감지하는 새로운 방법인 SaMIA를 제안합니다. 이 방법은 특히 Likelihood 정보를 사용할 수 없는 ChatGPT, Claude 3 같은 모델에도 적용가능하며, n-gram 일치율을 기반으로 합니다.

- **Technical Details**: SaMIA 방법은 LLM (Large Language Model, 대규모 언어 모델)이 생성한 텍스트 샘플들과 목표 텍스트 사이의 n-gram 일치 정도를 계산하여 SPL를 구합니다. 이는 LLM에 초기 텍스트의 일부를 제공하고, 여러 연속 텍스트를 생성하게 하여, 생성된 텍스트를 후보 텍스트로, 목표 텍스트의 나머지 부분을 참조 텍스트로 사용합니다. n-gram 일치도가 특정 임계값을 초과하면 해당 LLM이 그 텍스트로 훈련되었다고 간주합니다.

- **Performance Highlights**: SaMIA는 GPT-J-6B, OPT-6.7B, Pythia-6.9B, LLaMA-2-7B와 같이 훈련 데이터가 공개된 네 가지 LLM에서 실험되었습니다. 위키피디아(Wikipedia) 데이터에 대한 실험 결과, SaMIA는 기존의 likelihood 또는 loss 기반 방법들과 동등한 성능을 보였으며, 정보량과 SPL을 결합한 유출 탐지 방법은 기존 방법들 중 가장 높은 평균 점수를 달성했습니다. 또한, n-gram의 개수, 텍스트 샘플의 수, 목표 텍스트의 길이가 SaMIA의 성능에 미치는 영향에 대한 분석 결과를 보고합니다.



### In-Context Learning State Vector with Inner and Momentum Optimization (https://arxiv.org/abs/2404.11225)
Comments: 17 pages, 7 figures, 5 tables

- **What's New**: 이 논문에서는 간단한 예로부터 문맥 학습(In-Context Learning, ICL)을 수행하는 대형 언어 모델(Large Language Models, LLMs)의 능력을 분석하고, ICL에 의해 학습된 기능이 압축 벡터(compressed vectors)로 표현될 수 있음을 탐구합니다. 이 압축 벡터를 최적화하고 개선하는 새로운 방법, 즉 내부 최적화(inner optimization) 및 모멘텀 최적화(momentum optimization)를 제안합니다. 또한, 다수의 예제가 포함된 시나리오에서 ICL 함수를 효율적으로 처리할 수 있는 분할 정복(divide-and-conquer) 집계 방법을 도입했습니다.

- **Technical Details**: 이 연구에서 제안된 압축 벡터는 변환기(transformer)에서 파생되며, 상태 벡터(state vector) 개념을 도입하여 ICL의 처리 상태를 저장합니다. 내부 최적화와 모멘텀 최적화는 모델 수프(model soup)와 모멘텀 기반 경사 하강법(momentum-based gradient descent)에서 영감을 받아 테스트 시간에 상태 벡터를 점진적으로 정제합니다. 높은 수의 예제를 처리할 수 있도록 분할 정복 집계 방법도 함께 소개되어, ICL 기능을 더욱 효과적으로 압축할 수 있습니다.

- **Performance Highlights**: LLama-2와 GPT-J 모델을 사용한 실험을 통해 제안된 방법이 상태 벡터를 효과적으로 향상시키며, 제로샷(zero-shot) 및 퓨샷(few-shot) 설정에서 다양한 작업에서 최고 수준의 성능을 달성했다는 것을 입증했습니다. 이는 제안된 접근방식의 효과 뿐만 아니라 ICL의 더욱 포괄적인 이해를 가능하게 하는 계기를 마련합니다.



### Position Engineering: Boosting Large Language Models through Positional  Information Manipulation (https://arxiv.org/abs/2404.11216)
- **What's New**: 이 논문에서는 위치 엔지니어링(position engineering)이라는 새로운 기술을 소개합니다. 이는 대형 언어 모델들(Large Language Models, LLMs)을 안내하는 더 효율적인 방법을 제공합니다. 기존의 프롬프트 엔지니어링(prompt engineering)이 LLM에 제공된 텍스트를 수정하는 데 많은 노력을 요구하는 반면, 위치 엔지니어링은 텍스트 자체를 수정하지 않고 프롬프트의 위치 정보만을 변경합니다.

- **Technical Details**: 위치 엔지니어링은 리트리벌 증강 생성(retrieval-augmented generation, RAG)과 인-컨텍스트 학습(in-context learning, ICL)과 같은 널리 사용되는 LLM 시나리오에서 평가되었습니다. 이 기법은 기존의 텍스트 기반 프롬프트 조정 방식 대신 위치 정보의 조절만을 통해 LLM의 출력을 조정하는 접근 방식입니다.

- **Performance Highlights**: 위치 엔지니어링은 두 가지 시나리오 모두에서 기준선(baseline)을 상당히 개선함으로써, 대형 언어 모델의 기능을 활용하는 데 있어 유망한 새로운 전략임을 입증하였습니다.



### Prompt-tuning for Clickbait Detection via Text Summarization (https://arxiv.org/abs/2404.11206)
- **What's New**: 이 연구는 클릭베이트(Clickbait) 탐지를 위한 새로운 접근 방식인 프롬프트 튜닝(Prompt-tuning) 방식을 제안합니다. 이 방법은 텍스트 요약(Text summarization)을 사용하여 콘텐츠를 요약하고, 생성된 요약과 콘텐츠 간의 유사성을 바탕으로 클릭베이트 여부를 판별합니다. 이는 기존의 시맨틱 유사도(Semantic similarity) 계산 방식과 다른 접근법입니다.

- **Technical Details**: 연구진은 먼저 두 단계의 텍스트 요약 모델을 도입하여 사전 훈련된 언어 모델(Pre-trained language models)을 기반으로 고품질의 뉴스 요약을 생성합니다. 이후, 생성된 요약과 헤드라인을 프롬프트 튜닝의 입력으로 사용합니다. 또한, 클릭베이트 탐지 성능을 향상시키기 위해 외부 지식(External knowledge)을 통합하는 다양한 전략이 사용되었습니다.

- **Performance Highlights**: 이 방법은 잘 알려진 클릭베이트 탐지 데이터셋에 대한 광범위한 실험을 통해 최고의 성능(State-of-the-art performance)을 달성했다고 보고합니다. 이는 콘텐츠와 헤드라인 간의 유사성만을 평가하는 기존 방법들에 비해 더 정확한 클릭베이트 판단이 가능함을 시사합니다.



### Neuron Specialization: Leveraging intrinsic task modularity for  multilingual machine translation (https://arxiv.org/abs/2404.11201)
- **What's New**: 이 연구에서는 다국어 번역 (Multilingual Translation)을 위한 새로운 접근법인 'Neuron Specialization'을 제안합니다. 이 방법은 다양한 언어 간의 간섭을 줄이고 지식 전달을 향상시키기 위해 신경망 내부의 고유한 작업 모듈성을 활용합니다.

- **Technical Details**: 연구진은 Feed-Forward Networks (FFN)에서 중간 활성화를 분석하여 특정 언어에 대해 활성화되는 뉴런들을 발견하고, 이러한 뉴런들이 언어 근접성을 반영하는 구조적 중첩을 보이는 것을 관찰했습니다. 이를 바탕으로, 'Neuron Specialization' 방법론을 도입하여 FFN 레이어를 모듈화하고, 희소 네트워크를 통해 지속적으로 업데이트합니다.

- **Performance Highlights**: 'Neuron Specialization'을 적용한 결과, IWSLT와 EC30과 같은 소규모 및 대규모 다국어 번역 데이터세트에서 강력한 기준 모델들을 상회하는 성능 향상을 달성했습니다. 추가 분석을 통해 높은 자원 언어에서의 간섭 감소와 저자원 언어에서의 지식 전달 증진을 입증했습니다.



### FIZZ: Factual Inconsistency Detection by Zoom-in Summary and Zoom-out  Documen (https://arxiv.org/abs/2404.11184)
Comments: Submitted to ACL ARR on April 17th 2024

- **What's New**: 이 논문에서는 새로운 사실적 불일치 감지 메트릭 (Factual Inconsistency Detection by Zoom-in Summary and Zoom-out Document, FIZZ)을 제안하고, 이를 통해 기존의 요약 시스템 평가 방법에서 발견된 한계를 극복할 수 있는 방안을 제시합니다. FIZZ는 요약에서 추출한 원자 사실(atomic facts)을 소스 문서와 비교하는 것을 기반으로 합니다.

- **Technical Details**: FIZZ 시스템은 핵심 글 참조 해소(coreference resolution)를 통해 요약과 소스 문서에서 원자 사실을 추출한 후, 이 원자 사실을 사용하여 각 원자 사실의 일관성을 체크합니다. 핵심 글 참조 해소는 원자 사실이 더 세밀한 정보 단위로 요약 내용을 분석할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, FIZZ는 AggreFact 벤치마크 데이터셋에서 최고 성능을 달성했으며, 다른 대표적인 시스템들보다 뛰어난 결과를 보였습니다. 또한, FIZZ는 각 원자 사실에 대한 일관성을 다중 문장 레벨 추론(multi-sentence level reasoning)이 필요할 때 문맥 문장의 수를 적응적으로 확장하는 방법을 제안하여 정확도를 크게 향상시켰습니다.



### Context-Aware Siamese Networks for Efficient Emotion Recognition in  Conversation (https://arxiv.org/abs/2404.11141)
- **What's New**: 이 연구에서는 대화 맥락을 메트릭 학습(Metric Learning)과 통합하여 대화에서 감정 인식(Emotion Recognition in Conversation, ERC) 문제에 접근하고 있습니다. 또한, 데이터 불균형(Data Imbalance)을 통제하는 다양한 방법을 사용하여 ERC의 주요 도전 과제를 해결하려고 합니다. 이 방법은 기존의 감정 레이블에만 의존하는 대신, 대화 맥락을 반영하여 메트릭 학습을 통해 더 융통성 있는 분류 시나리오(Classification Scenario)에서 작동하도록 하여 다양한 감정의 변화에 적응 가능한 모델을 제공합니다.

- **Technical Details**: 제안된 모델은 시아미즈 네트워크(Siamese Network) 구조를 사용하며, 대화의 문장 임베딩과 트랜스포머 인코더 레이어(Transformer Encoder Layers)를 활용하여 대화 발화를 나타내고 대화 맥락에 주의(Attention)를 기울입니다. 이 연구는 감정 인식을 위해 크로스 엔트로피(Cross Entropy Loss)와 대조 손실(Contrastive Loss)을 통해 모델을 업데이트하는 두 단계의 프로세스를 채택합니다. 이는 DailyDialog 데이터셋에서 매크로 F1 점수(Macro F1 Score) 57.71%를 달성하여 관련 작업을 능가함으로써 기존의 대규모 언어 모델(Large Language Models, LLMs)보다 우수한 성능을 보여주었습니다.

- **Performance Highlights**: 이 모델은 매크로 F1 점수에서 57.71%, 마이크로 F1 점수(Micro F1 Score)에서 57.75%를 달성하여 DailyDialog 데이터셋에서 최신 기술의 성과(State-of-the-Art, SotA)에 비해 경쟁력 있는 성능을 보여주었습니다. 이는 기존의 Falcon이나 LLaMA 2와 같은 모델보다 더 가볍고 빠르게 훈련 가능한 모델임에도 불구하고 높은 성능을 유지한다는 점에서 중요한 의의를 가집니다.



### A Novel ICD Coding Framework Based on Associated and Hierarchical Code  Description Distillation (https://arxiv.org/abs/2404.11132)
- **What's New**: 국제 질병 분류(ICD) 코드 할당 문제를 해결하기 위해 연관 및 계층적 코드 설명증류(Associated and Hierarchical Code Description Distillation, AHDD)라는 새로운 프레임워크가 제안되었습니다. 이 프레임워크는 의료 노트의 중요 부분만을 정확히 포착하여 무의미하거나 잘못된 코드 할당을 방지합니다. 주목할 만한 점은, 코드의 계층적 구조와 각 코드의 설명을 활용하여 더 정확한 코드 예측을 가능하게 하는 것입니다.

- **Technical Details**: AHDD 프레임워크는 의료 노트와 관련 코드, 계층적 코드를 공유 인코더에 입력으로 사용합니다. 코드 설명을 인식하는 주의(attention) 메커니즘을 적용하여 레이블별 표현을 도출하고, 이를 통해 의료 노트를 증류합니다. 결과적으로 코드 설명을 인식하는 출력이 분류에 사용됩니다. 이와 같은 방식은 복잡하고 잡음이 많은 의료 텍스트에서 핵심 단어를 식별하는 데 있어 향상된 성능을 보여줍니다.

- **Performance Highlights**: 벤치마크 데이터셋에 대한 실험 결과, 제안하는 프레임워크는 여러 최신 기술 모델들을 뛰어넘는 우수한 성능을 발휘하였습니다. 특히, 잠재적으로 오류를 발생시킬 수 있는 복잡한 계층 구조의 코드 할당 문제를 효과적으로 해결하였습니다.



### What's under the hood: Investigating Automatic Metrics on Meeting  Summarization (https://arxiv.org/abs/2404.11124)
- **What's New**: 회의 요약의 필요성이 증가함에 따라, 이 분야의 평가 지표에 대한 문제점이 부각되고 있습니다. 이 연구는 기존의 자동 평가 메트릭스가 실제 회의 요약에서 나타나는 특정 오류들을 얼마나 잘 포착하는지, 어떠한 오류들을 감추는지를 인간 평가와의 상관관계를 통해 분석합니다. 또한, Transformer 기반 순차적(sequence-to-sequence) 모델과 자기회귀(autoregressive) 모델을 이용한 실험을 통해, 모델 구조가 회의록 내의 다양한 도전 과제에 어떻게 반응하는지도 연구합니다.

- **Technical Details**: 연구는 QMSum 데이터셋을 사용하여 회의록과 요약문에서의 도전 과제와 오류를 전문가가 주석(annotation)하도록 하여, 주요 도전 과제 및 발생 가능 오류를 정의하고 분석합니다. 이 연구는 Transformer 및 autoregressive 모델 아키텍처를 사용하여 도전 과제와 이로 인해 발생하는 오류 사이의 상관관계를 직접적으로 연구합니다. 이러한 분석을 통해, 일부 메트릭스가 구조적 해체 오류는 잘 포착하지만, 환각 오류에 대해서는 오히려 보상하는 경향이 있는 것을 발견했습니다.

- **Performance Highlights**: 자동 메트릭스의 성능 평가에서 구조적 오류는 인간 평가와 일치하는 경향이 있으나, ROUGE는 오류의 영향을 세밀하게 구분하는 데 어려움이 있습니다. 약 1/3의 메트릭스와 오류 조합에서는 오류를 간과하거나 보상하는 경향을 보였습니다. 예를 들어, Perplexity는 부정확한 참조를 향하며, Lens는 구조적 해체를 선호합니다. 이는 회의 요약의 평가 방법 개선이 시급함을 강조합니다.



### Consistency Training by Synthetic Question Generation for Conversational  Question Answering (https://arxiv.org/abs/2404.11109)
- **What's New**: 이 연구는 대화형 질문 응답(QA) 설정에서 불필요한 역사를 처리하기 위한 새로운 모델에 대한 접근법을 소개합니다. CoTaH(Consistency-Trained augmented History)라는 새로운 모델 비의존적 접근 방식을 도입하고, 이 방법은 생성된 질문을 데이터 증강으로 사용하는 최초의 사례입니다.

- **Technical Details**: CoTaH 방식은 실제 역사 데이터와 증강된 역사 데이터 둘 다를 사용하여 모델이 불필요한 역사에 강인하게 추론할 수 있도록 일관성 훈련(consistency training)을 적용합니다. 데이터 증강을 위해 대화형 질문 생성기(conversational question generator, CQG)를 사용하고, 선택된 합성 질문을 역사에 추가합니다. 이 접근 방식을 통해, 예측 시 하나의 변형기(transformer)만을 사용하여 시간과 메모리 사용을 줄일 수 있습니다.

- **Performance Highlights**: 제안된 CoTaH 방법은 전반적인 F1 점수에서 1.8% 향상을 보여주며, 특히 많은 역사적 맥락을 가진 질문에서 두드러진 성능 개선을 보여줍니다. 이는 기존 방법들이 예측된 역사 대답을 사용했을 때 성능이 크게 감소한 것에 비해 현저한 향상입니다.



### Inductive-Deductive Strategy Reuse for Multi-Turn Instructional  Dialogues (https://arxiv.org/abs/2404.11095)
Comments: 27 pages, 3 figures, 12 tables

- **What's New**: 이 새로운 연구에서는 대규모 언어 모델(Large Language Models, LLMs)을 인간의 기대에 맞추기 위해 인간의 학습에서 나타나는 인지 능력에 영감을 받아 복잡한 대화 흐름을 명시적으로 모델링하는 방식을 제안합니다. 연구팀은 다양한 실제 지시 대화에서 고차원 전략을 유도하고, 이를 새로운 대화 시나리오에 적용하여 고품질의 지시문을 생성하는 'Inductive-Deductive Strategy Reuse (IDEAS)' 방법을 개발하였습니다.

- **Technical Details**: 이 연구에서는 대화 역사(histories)와 해당 지시문(instructions) 사이의 대화 전략을 추출하기 위해 GPT-4를 사용합니다. 추출된 유사한 지시 전략은 일반적인 원리로 추상화되며, 이는 새로운 대화 시나리오에서 특정 지시를 생성할 때 선택할 수 있는 다양한 전략 중 하나를 샘플링하여 지시 생성을 안내합니다. 이러한 과정을 통해 생성된 멀티 턴(multi-turn) 지시 대화는 다양성과 깊이, 통찰력이 향상되었습니다.

- **Performance Highlights**: IDEAS를 사용하여 생성된 지시 대화는 기존 방법들과 비교했을 때 경쟁력 있는 기준(baselines)를 뛰어넘는 성능을 보였습니다. 또한, 이러한 고품질의 지시 대화 데이터셋은 하류 챗봇 모델(downstream chat model)의 성능 향상에 기여할 수 있음을 실험적으로 입증하였습니다.



### ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large  Language Models (https://arxiv.org/abs/2404.11086)
Comments: arXiv admin note: text overlap with arXiv:2305.08322 by other authors

- **What's New**: 이 연구는 LLMs(대형 언어 모델)의 발전에 발맞춰 베트남어 평가 기준의 필요성에 초점을 맞추고 있습니다. ViLLM-Eval은 다양한 분야와 난이도를 아우르는 문제들로 구성된 포괄적인 평가 도구를 소개합니다. 이 도구는 베트남 문화, 역사 및 현재 사항을 반영하여 베트남어 사용자에게 맞춤화된 평가를 제공합니다.

- **Technical Details**: ViLLM-Eval은 다중선택 문항 및 다음 단어 예측 작업을 포함하며, 인문학에서 과학 및 공학에 이르기까지 다양한 분야를 다룹니다. 평가 기준에 따라 베트남어 LLMs(Vistral-7B-Chat, PhoGPT-4B-Chat, VinaLLaMA-7B-Chat 등)의 성능을 측정했으며, 이 모델들이 베트남어 작업에서 상당한 개선의 여지가 있음을 발견했습니다.

- **Performance Highlights**: ViLLM-Eval을 사용하여 최고 성능의 베트남어 LLMs를 평가한 결과, 이 모델들은 여전히 베트남어 이해와 반응에서 큰 개선의 여지가 있음을 보여줍니다. 이는 베트남어 맞춤형 LLMs의 추가 개발 및 세밀한 조정이 시급함을 강조합니다.



### Unified Examination of Entity Linking in Absence of Candidate Sets (https://arxiv.org/abs/2404.11061)
- **What's New**: 이 연구는 최신 엔터티 링킹(entity linking) 시스템의 통합된 블랙박스 평가 프레임워크를 제시하며, 특히 후보 세트(candidate sets)의 영향에 대한 면밀한 분석을 통해 이들 시스템의 일반적 적용성에 대한 한계와 가능성을 탐구합니다. 또한 AIDA/testc 데이터셋을 사용하여 최신 뉴스 스토리에서 신규 엔터티를 포함한 검증을 수행했습니다.

- **Technical Details**: 연구팀은 GERBIL 평가 도구와 gerbil_connect 인터페이스를 사용하여 주어진 텍스트에 대해 엔터티를 올바르게 식별하고 연결할 수 있는 시스템의 능력을 블랙박스 방식으로 평가합니다. 후보 세트가 주어진 시스템들은 이를 바탕으로 적절한 엔터티 후보를 선택하는 반면, 후보 세트가 없는 경우에는 도메인별 전체 후보 세트(in-domain candidate set)를 사용하여 시스템의 강건함을 실험했습니다. 실험은 CoNLL/AIDA 데이터셋의 표준화된 형식으로 진행되었습니다.

- **Performance Highlights**: 이 평가에서 일부 모델은 후보 세트를 확장하거나 제거함으로써 성능 하락을 겪었지만, 구조적 예측(structured prediction)을 사용하는 모델이 testc 데이터셋에서 가장 우수한 성능을 보였습니다. 이 데이터셋은 924개의 신규 엔터티를 포함하고 있으며, 모델의 실제 적응력과 리콜(recall) 능력을 서로 비교할 수 있었습니다.



### On the Causal Nature of Sentiment Analysis (https://arxiv.org/abs/2404.11055)
Comments: An enhanced version of our previous exploration in arXiv:2305.01764

- **What's New**: 이 논문은 감정 분석(Sentiment Analysis, SA) 문제를 두 가지 작업으로 재구성합니다: (1) 리뷰와 감정 사이의 인과 관계를 파악하는 인과 발견(causal discovery) 작업과 (2) 리뷰를 입력으로 사용하여 감정을 모델링하는 전통적인 예측(prediction) 작업입니다. 특히, 심리학의 피크엔드 규칙(peak-end rule)을 적용하여 샘플을 C1(리뷰가 감정을 주도) 또는 C2(감정이 리뷰를 주도)로 분류합니다.

- **Technical Details**: 이 연구는 인과 추론(causal inference)에서 얻은 통찰을 통해 SA의 인과 메커니즘을 밝히고, 이를 바탕으로 더 나은 성능을 달성하기 위해 인과 프롬프트(causal prompts)를 제안합니다. 제안된 인과 프롬프트는 라지 랭귀지 모델(Large Language Models, LLMs)에 인과 그래프의 근본적인 이해를 제공하여 SA의 성능을 최대 32.13 F1 포인트까지 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과에 따르면 기본 프롬프트에서 LLMs는 C2 인과 과정에 해당하는 데이터에서 더 나은 성능을 보였습니다. 그러나 인과 방향을 고려하여 맞춘 인과 프롬프트를 사용하면, 제로샷(zero-shot) SA의 성능이 크게 향상됩니다. 이는 인과 메커니즘을 정확히 이해하기 위해 여전히 개선의 여지가 있는 LLMs에 새로운 방향을 제시합니다.



### Offset Unlearning for Large Language Models (https://arxiv.org/abs/2404.11045)
- **What's New**: 이 연구에서는 기존의 LLM (Large Language Models) 취약점을 개선하기 위해 새로운 'δ-unlearning' 방법을 제안합니다. 이 방법은 블랙박스 LLM에서 문제가 되는 트레이닝 데이터의 영향을 제거하는 새로운 접근 방식을 제시하며, 기존 모델의 내부 가중치에 접근하지 않고도 민감한 데이터를 '잊게' 하는데 필요한 로짓 오프셋(logit offset)을 학습합니다. 이는 두 개의 작은 모델의 로짓(logits)을 대조하여 획득됩니다.

- **Technical Details**: δ-unlearning 방법은 블랙박스 LLM의 내부 가중치를 수정하지 않고 형성되며, 두 개의 작은 화이트박스(white-box) 모델 간의 로짓 차이를 계산하여 큰 모델의 로짓에 추가합니다. 이렇게 함으로써, 민감한 쿼리에 대응하여 큰 모델의 예측을 수정하는 방법을 소규모 모델의 행동 변화에서 유추할 수 있습니다. 또한, 이 기법은 다양한 ['unlearning' algorithms](응용 프로그램을 다양화할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: TOFU 벤치마크에서의 실험 결과, δ-unlearning은 잊어야 할 데이터 세트(forget set)의 특정 데이터를 효과적으로 '잊으면서' 잊어야 할 범위 밖의 일반적인 데이터에서는 기존의 직접 튜닝된 큰 모델보다 비슷하거나 더욱 뛰어난 성능을 유지하는 것으로 나타났습니다. 이는 δ-unlearning이 효율적인 버전 관리와 사용자 맞춤화를 가능케 하며, 큰 모델의 파라미터 업데이트를 필요로 하지 않으면서도 유효한 'unlearning'을 달성하였음을 시사합니다.



### Procedural Dilemma Generation for Evaluating Moral Reasoning in Humans  and Language Models (https://arxiv.org/abs/2404.10975)
Comments: CogSci 2024

- **What's New**: 이 연구는 언어 모델(LM: Language Model)이 도덕적 판단을 어떻게 내리는지 시스템적으로 평가하는 새로운 접근 방식을 제공합니다. 저자들은 '인과 그래프(causal graphs)'를 사용하여 도덕적 딜레마에 대한 프롬프트 템플릿을 번역하는 프레임워크를 구축하고, 이를 바탕으로 'OffTheRails' 벤치마크를 구성했으며, 이는 50개의 시나리오와 400개의 유니크한 테스트 아이템으로 구성되어 있습니다.

- **Technical Details**: 이 연구는 도덕적 딜레마를 평가하기 위해 인과 그래프를 기반으로 하는 프롬프트를 생성하는 절차적 방법론을 제안합니다. GPT-4와 Claude-2 같은 언어 모델을 사용하여 특정 도덕적 조건(예: 수단(means) 대 부수적 효과(side effect), 회피 가능성(evitability) 대 불가피성(inevitability), 행위(commission) 대 무행위(omission))에 따라 다양하게 생성된 테스트 아이템을 평가합니다. 이러한 조건들은 인과 구조(causal structure)에 기반하여 체계적으로 조정됩니다.

- **Performance Highlights**: 인간 참가자와 언어 모델 모두에서, 해가 필요한 수단으로 사용될 때(부수적 효과에 비해) 허용 가능성이 낮고 의도적 판단이 높게 나타났습니다. 이와 유사하게, 피할 수 있는 해로움(inevitable harms)에 비해 피할 수 없는 해로움(evitable harms)에서도 같은 패턴이 관찰되었습니다. 그러나 해가 발생하는 것이 주체의 행동에서 비롯된 것인지 아니면 행동을 하지 않아서 생긴 것인지는 뚜렷한 차이를 보이지 않았습니다. 이러한 분석을 통해 도덕적 판단에 영향을 주는 요소들을 더 명확히 이해할 수 있습니다.



### Uncertainty-Based Abstention in LLMs Improves Safety and Reduces  Hallucinations (https://arxiv.org/abs/2404.10960)
- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)의 신뢰성 향상을 위해 불확실성(uncertainty)을 기반으로 응답을 삼가는(abstaining) 방법을 탐구합니다. 특히, 정확성(correctness), 무응답 질문에 대한 환상(hallucination), 그리고 안전성(safety)의 세 가지 상황에서 모델이 언제 응답을 삼가야 하는지 판단하는 것을 중심으로 분석하였습니다. 이는 인간의 불확실성 인지 능력을 모방하여, 응답 생성 과정에서의 오류를 감소시키고자하는 새로운 시도입니다.

- **Technical Details**: 연구는 두 가지 유형의 불확실성을 조사합니다: 통계적 불확실성(statistical uncertainty)과 대화 내 불확실성(In-Dialogue Uncertainty, InDU). 통계적 불확실성은 토큰별 확률 벡터의 엔트로피를 측정하여 평가되며, 대화 내 불확실성은 모델이 '모르겠다'와 같은 응답을 선택하거나 '아마도', '어쩌면' 같은 추측적 단어를 사용함으로써 나타납니다. 이 두 불확실성 측정 방법을 통해 연구는 인간 피드백을 통한 강화 학습(Reinforcement Learning with Human Feedback, RLHF)이 적용된 모델과 그렇지 않은 모델을 분석하여, 어떤 유형의 불확실성 측정이 더 효과적인지를 평가합니다.

- **Performance Highlights**: 연구 결과에 따르면, 통계적 불확실성을 사용하여 응답을 삼갔을 때 정확성이 2%에서 8% 향상되었으며, 대화 내 불확실성(InDU)을 활용하여 무응답 질문을 약 50% 감지하여 환상 현상을 크게 감소시켰습니다. 또한, RLHF로 향상된 모델을 사용하여 통계적 불확실성을 기반으로 안전성을 강화한 결과, 70%에서 최대 99%까지 위험한 응답을 걸러내는 성과를 보였습니다. 이러한 개선은 거의 추가적인 계산 비용 없이 이루어졌습니다.



### Can Language Models Solve Olympiad Programming? (https://arxiv.org/abs/2404.10952)
Comments: Code and data: this https URL

- **What's New**: 이 연구에서는 USA Computing Olympiad (USACO)에서 가져온 307개의 어려운 문제를 포함하는 USACO 벤치마크를 소개합니다. 이 벤치마크는 품질이 높은 단위 테스트, 참조 코드, 공식 분석을 포함하여, 언어 모델(Language Models, LMs)을 이용한 경쟁 프로그래밍(competitive programming)의 추론 방법을 처음으로 구축하고 테스트할 수 있게 합니다.

- **Technical Details**: GPT-4는 제로샷 체인-오브-소트(zero-shot chain-of-thought) 프롬프팅을 사용하여 8.7%의 pass rate를 달성했으며, 자기성찰(self-reflection)과 에피소딕 지식(episodic knowledge) 검색(retrieval)의 결합을 통해 최대 20.2%까지 향상된 성능을 보였습니다. 연구진은 또한 인간과의 대화형 학습(human-in-the-loop)을 통해 LMs의 알고리즘적 추론 능력과 한계를 더 잘 이해하기 위한 새로운 방법을 개발했습니다.

- **Performance Highlights**: GPT-4는 주어진 15개 문제 중 13개 문제를 풀어냈습니다. 이는 사람의 최소한의 힌트만으로도 GPT-4의 문제 해결 능력이 크게 향상될 수 있음을 보여줍니다. 그러나, 언어 모델의 활용과 그 성능은 여전히 USACO 벤치마크의 청동 수준 이상을 해결하기에는 미치지 못하며, 이에 대한 추가적인 연구개발이 요구됩니다.



### More Room for Language: Investigating the Effect of Retrieval on  Language Models (https://arxiv.org/abs/2404.10939)
Comments: NAACL 2024

- **What's New**: 이 연구에서는 '이상적인 검색' 방법론을 도입하여 언어 모델의 검색 강화(retrieval-augmented) 효과를 완전히 제어 가능한 환경에서 연구합니다. 이를 통해 검색 강화가 언어 모델의 동작 방식에 어떻게 영향을 미치는지에 대한 광범위한 평가를 실시하였습니다.

- **Technical Details**: 이 연구는 가면 언어 모델(Masked Language Models)과 변압기 기반(Transformers)의 인코더-디코더 구조를 사용합니다. 검색 모듈은 독립적으로 작동하며, 학습 과정에서 실제 검색 대신 문장을 재구성하는 파라프레이즈를 사용하여 이상적인 검색 환경을 제공합니다. 학습 자료로는 영어 위키피디아가 사용되어, 정보가 풍부한 데이터셋을 기반으로 모델의 성능을 향상시키고자 하였습니다.

- **Performance Highlights**: 검색 강화를 통한 언어 모델은 세계지식(world knowledge)을 덜 저장하며, 지역적 맥락(local context)과 단어 간 의존성(inter-word dependencies)에 대한 이해는 향상되었지만, 글로벌 맥락(global context)에 대한 이해는 떨어지는 것으로 나타났습니다. 검색의 질이 낮은 환경에서도 기본 학습 성능(no-retrieval performance)에 근접하여, 전반적인 품질 저하는 발생하지 않았습니다.



### Binder: Hierarchical Concept Representation through Order Embedding of  Binary Vectors (https://arxiv.org/abs/2404.10924)
- **What's New**: 이 연구에서는 'Binder'라는 새로운 순서 기반 표현(order-based representation) 모델을 제안했습니다. Binder는 이진 벡터(binary vectors)를 사용하여 개념을 임베딩하며, 다른 방법들에 비해 메모리 사용량이 현저히 적습니다. 특히, 체계적인 최적화 방식을 통해 효율적으로 임베딩 벡터를 학습할 수 있으며, 이는 선형 시간 복잡도(linear time complexity)를 가집니다. 또한, Binder는 직접적인 관계만을 사용하여 개념의 임베딩을 학습할 수 있어, 기존의 순서 기반 접근 방식들이 요구하는 간접적인 관계 없이도 효과적입니다.

- **Technical Details**: Binder는 각 개체를 0과 1로 구성된 벡터로 표현하며, 이는 'is-a' 관계를 수학적으로 간단하고 직관적으로 모델링할 수 있게 합니다. Binder는 각 차원이 어떤 속성을 나타내며, 하위 개념(sub-concept)이 상위 개념(super-concept)의 모든 속성을 가지고 있을 때 해당 차원의 값이 1로 설정됩니다. 이러한 이진 표현은 메모리 효율성을 극대화하며, 기존의 하이퍼볼릭 임베딩(Hyperbolic embedding)이나 박스 임베딩(Box Embedding)과 같은 방법들과 비교할 때 간단한 최적화 방식을 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 Binder는 기존의 순서 기반 표현 방식과 비교하여 매우 정확한 결과를 제공합니다. 특히, 전달적 폐쇄 링크 예측(transitive closure link prediction) 작업에서 경쟁 모델들을 뛰어넘는 성능을 보여주었습니다. Binder는 단순한 직접적 연결만을 사용하여도 효과적인 임베딩 학습이 가능하며, 이는 다른 모델들이 간접적 연결에 의존하는 것과 대조됩니다.



### Teaching a Multilingual Large Language Model to Understand Multilingual  Speech via Multi-Instructional Training (https://arxiv.org/abs/2404.10922)
Comments: NAACL Findings 2024

- **What's New**: 이 논문은 BLOOMZMMS 모델을 제안하며, 이 모델은 다국어 LLM (Large Language Model)과 다국어 음성 인코더를 통합하여 음성 인식 분야에서 LLM의 가능성을 활용하는 것을 목표로 합니다. 특히, 이는 텍스트에서 음성 방식으로 언어 지식을 전달하는 멀티-인스트럭셔널 학습 접근 방식을 활용합니다.

- **Technical Details**: BLOOMZMMS 모델은 사전 훈련된 다국어 음성 인코더와 LLM을 결합하고, 중간 어댑터(Adaptor) 모듈을 사용하여 음성 인코더 출력을 LLM의 텍스트 토큰 임베딩(latent space)에 매핑합니다. 이 연구는 139개 언어에 걸쳐 1900시간의 텍스트로 표기된 데이터를 사용하여 다국어 음성 표현 학습을 시행하였고, 이러한 표현이 ASR(Automatic Speech Recognition) 작업에서 다국어 LLM과 호환될 수 있음을 입증하였습니다.

- **Performance Highlights**: 제안한 모델은 기존의 타스크 일반화에 한계를 보였으나, 멀티-인스트럭셔널 스타일에서 합성 목표를 생성함으로써 이를 해결하였습니다. 제로샷(zero-shot) 평가 결과는 음성 번역(Speech Translation, SLT) 및 다국어 스포큰 자연 언어 추론(Multilingual Spoken Natural Language Inference, NLI)과 같은 여러 작업에서 접근 방식의 강건성을 확인시켜 주었습니다.



### Which questions should I answer? Salience Prediction of Inquisitive  Questions (https://arxiv.org/abs/2404.10917)
- **What's New**: 이 논문은 궁금증을 유발하는 질문(inquisitive questions)의 중요성(salience)을 예측하기 위한 모델, QSALIENCE를 소개합니다. 이는 기사의 문맥을 기반으로 질문의 중요성을 평가하여 해당 질문의 답변이 텍스트 이해를 크게 향상시킬 가능성이 높은지를 판별합니다. QSALIENCE 는 언어학자가 주석을 단 데이터셋에서 지시 학습(instruction-tuning)을 통해 개발되었습니다.

- **Technical Details**: QSALIENCE 모델은 1,766개의 (문맥, 질문) 쌍에 대한 언어학자의 중요성 점수를 기반으로 학습되었습니다. 이 모델은 답변 가능성과 높은 상관관계를 가지는 중요한 질문을 식별할 수 있습니다. 이 연구는 복잡한 텍스트에서 암시적 정보를 포착하는 장문의 맥락 과제에 대한 지시 학습의 유틸리티를 보여줍니다. GPT-4, Flan-T5와 같은 다른 대형 언어모델(Language Models, LLMs)과 비교했을 때 QSALIENCE는 우수한 성능을 보였습니다.

- **Performance Highlights**: 사람이 생성한 중요한 질문들(Questions Under Discussion, QUDs)은 높은 중요성 점수를 받았으며, 이러한 점수들은 문서 내에서 QUDs의 답변 가능성과 일관되게 높은 상관관계를 보였습니다. 이러한 결과는 질문의 중요성에 대한 일반적인 개념과 답변의 유틸리티를 연결하는 것을 지원합니다. 또한, 최적화된 요약에서 중요한 질문에 답하는 요약은 인간 독자들에 의해 더 높게 평가되었습니다.



### Search Beyond Queries: Training Smaller Language Models for Web  Interactions via Reinforcement Learning (https://arxiv.org/abs/2404.10887)
Comments: 9 pages

- **What's New**: GLAINTEL이라고 불리는 새로운 지능형 웹 탐색 에이전트가 제안되었습니다. 이 에이전트는 고급 언어 모델과 강화학습을 활용하여 웹 페이지를 탐색하고 사용자의 명시적 의도에 따라 쿼리를 생성할 수 있습니다. 특히, 제품 검색과 같은 복잡한 시나리오에서 우수한 성능을 보이며, 크기가 더 작은 언어 모델이 큰 모델들과 비교하여 경쟁력 있는 결과를 제공합니다.

- **Technical Details**: GLAINTEL은 Flan-T5 아키텍처를 바탕으로 구축되며, 동적 작업 공간에서 활동을 적절히 추론할 수 있도록 언어 모델링 헤드와 가치 추정 헤드를 포함합니다. 사용자의 의도와 관찰에 따라 모델이 가능한 모든 행동의 조건부 확률을 계산하고, 소프트맥스 함수를 이용해 동작을 선택합니다. 시스템은 Proximal Policy Optimization (PPO) 알고리즘을 통해 미세 조정되었습니다.

- **Performance Highlights**: 실제 실험 결과, GLAINTEL은 인간 시연 없이 훈련된 경우에도 맥락 학습 방식을 사용하는 큰 모델(최대 5400억 파라미터)을 능가했습니다. 또한, 인간 시연을 활용한 후 비지도 학습으로 추가 훈련을 진행했을 때 가장 우수한 결과를 보였습니다. 이는 GPT-4를 활용하는 방법과 비교할 수 있는 성과를 달성했습니다.



### Incubating Text Classifiers Following User Instruction with Nothing but  LLM (https://arxiv.org/abs/2404.10877)
- **What's New**: 이 논문은 사용자의 지시(클래스 정의)에 따라 분류 데이터를 생성하고, 이를 이용하여 소형 텍스트 분류기를 인간의 주석이나 원본 코퍼스 없이도 훈련할 수 있는 Incubator 프레임워크를 제안합니다. Incubator는 복잡하고 상호 의존적인 클래스를 처리할 수 있는 최초의 프레임워크로, 예를 들어 '교육자에 의한 TED 토크'와 '기타'와 같은 클래스입니다.

- **Technical Details**: Incubator는 HuggingFace의 분류 데이터셋과 설명을 기반으로 하여 생성된 지시-데이터 매핑에 먼저 조정된 LLM(Large Language Model, 대규모 언어 모델)을 사용합니다. 그 후, 세맨틱 텍스트 임베딩의 클러스터 중심 학습을 통해 데이터의 균일성과 의미 다양성을 강조합니다. 또한, GPT-4와 같은 강력한 LLM을 사용하여 인컨텍스트 학습(In-context Learning)을 통해 지시를 조정하고, K-means 클러스터링 알고리즘을 활용하여 의미론적으로 다양한 샘플을 생성합니다.

- **Performance Highlights**: Incubator는 기존의 베이스라인과 비교해 우수한 성능을 보여주며, 전통적인 벤치마크에서도 잘 작동합니다. 또한 레이블 의존성과 사용자 선호도를 고려하고, 여러 분류기를 인큐베이팅하여 고급 텍스트 마이닝 시스템을 실현할 수 있습니다. 실험 결과 Incubator는 강력한 텍스트 분류기를 배양하고, 레이블 상호 의존성을 고려하며 사용자 지시에 따라 여러 텍스트 분류기를 배양하여 사용할 수 있음을 확인했습니다.



### Forcing Diffuse Distributions out of Language Models (https://arxiv.org/abs/2404.10859)
- **What's New**: 본 연구에서는 언어 모델이 유효한 결과에 대해 포괄적인 분포(Predictive Distribution)를 출력하도록 유도하는 미세 조정(Fine-tuning) 방법을 제안합니다. 특히, 다양한 작업(Task)과 분포에 대해 일반화 가능한 메소드를 소개하며, 복잡한 프롬프트 공학(Prompt Engineering)이나 수동 재작성 없이도 다양한 결과를 생성할 수 있습니다.

- **Technical Details**: 이 미세 조정 방법은 입력된 데이터셋에서 유효한 옵션 집합(Set of Valid Options) 전체에 걸쳐 균일한 확률 분포를 생성하는 언어 모델의 능력을 향상시킵니다. 연구에서는 언어 모델이 생성한 출력의 다양성(variety)과 분산성을 측정하기 위해 KL-분산(KL-divergence)과 엔트로피(Entropy)를 사용합니다. 이 방법은 특정 작업에 대해 미세 조정된 모델이 다른 작업에 대해서도 유효하게 작용함을 보여 줍니다.

- **Performance Highlights**: 실험 결과, 이 방법은 기존 모델 대비 4배 많은 고유한 이름, 3배 많은 출생 장소, 1.5배 많은 경력을 포함하는 합성 전기(synthetic biographies) 데이터셋을 생성하는 데 성공했습니다. 이는 다양한 합성 데이터셋 생성(Synthetic Dataset Generation)에 효과적인 접근 방식임을 시사합니다. 모델들은 복잡한 프롬프트 공학이나 수동 개입 없이도 다양한 결과를 생성할 수 있는 능력을 보여줍니다.



### D3CODE: Disentangling Disagreements in Data across Cultures on  Offensiveness Detection and Evaluation (https://arxiv.org/abs/2404.10857)
- **What's New**: 이 연구에서는 문화적 다양성을 고려한 언어 인식을 위한 새로운 데이터셋, D3CODE를 소개합니다. 이 데이터셋은 21개국 4309명의 참가자들로부터 수집된 4.5K개 이상의 문장에 대한 병렬 주석(annotations)을 포함하고 있으며, 성별과 연령이 균형을 이루고 있습니다. 주석자들의 도덕적 가치를 'care', 'equality', 'proportionality', 'authority', 'loyalty', 'purity'와 같은 여섯 가지 도덕적 기초로 분석하여, 지역별 변화를 상세히 조명하고 있습니다.

- **Technical Details**: D3CODE 데이터셋은 다양한 사회문화적 배경을 가진 주석자들의 언어에 대한 인식을 수집하고 분석하기 위한 목적으로 구축되었습니다. 각 주석자는 문장에 대해 'offensive' 여부를 평가하며, 이 과정에서 그들의 도덕적 가치가 어떻게 영향을 미치는지를 중점적으로 검토하였습니다. 데이터 분석을 통해 주석자들의 지역적, 문화적 차이가 언어 인식에 어떻게 영향을 미치는지를 구체적으로 밝히고 있습니다.

- **Performance Highlights**: D3CODE 데이터셋 분석을 통해 주석자들의 지역별 인식 차이가 뚜렷하게 나타나는 것을 확인할 수 있었습니다. 이는 NLP 모델이 문화적 다양성을 효과적으로 반영할 수 있도록 하는 데 중요한 기초 자료가 됩니다. 또한, 주석자 개개인의 도덕적 가치가 그들의 언어 인식에 크게 기여함을 보여줌으로써, 단순한 데모그래픽 차원을 넘어서는 해석의 깊이를 제공합니다.



### A LayoutLMv3-Based Model for Enhanced Relation Extraction in  Visually-Rich Documents (https://arxiv.org/abs/2404.10848)
Comments: Accepted at the International Conference on Document Analysis and Recognition (ICDAR 2024)

- **What's New**: 이 연구에서는 시각적으로 풍부한 문서(VRD)에 적용된 관계 추출(RE) 작업에 대해 최신 기술보다 뛰어난 성능을 보이거나 일치하는 새로운 모델을 제안합니다. 이 모델은 LayoutLMv3을 기반으로 초기화됐으며, 특정 기하학적 사전 학습 없이도 효과적으로 작동하며, 매개변수(parameter) 수도 적습니다.

- **Technical Details**: 이 모델은 최신 LayoutLMv3을 기반으로 하여, Transformer 기반의 사전 훈련 방법 없이도 다양한 모달리티(multimodality)를 적용한 VRD에서 효과적인 관계 추출을 가능하게 합니다. 또한, 문서의 블록 순서, 모델 속성, 멀티태스크 학습(multi-task learning) 등 다양한 요소들이 RE 모델 성능에 어떠한 영향을 미치는지에 대한 광범위한 절단 분석(ablation study)을 수행했습니다.

- **Performance Highlights**: FUNSD와 CORD 데이터셋에서 진행된 실험에서 이 모델은 현재 가장 우수한 성과를 낸 기존 모델과 비교하여 우수하거나 동등한 성능을 보였습니다. 매개변수의 수를 줄인 상태에서도 높은 성능 유지는 이 모델의 효율성을 강조합니다.



### Fewer Truncations Improve Language Modeling (https://arxiv.org/abs/2404.10830)
- **What's New**: 본 논문에서는 대규모 언어 모델(Large Language Model, LLM)의 훈련 과정에서 발생하는 문서의 불필요한 절단을 없애기 위한 새로운 방법인 Best-fit Packing을 제안합니다. 이 방법은 문서를 효율적으로 패킹하여 데이터 무결성을 보존하고, 훈련의 효율성을 기존의 문서 연결(concatenation) 방식과 동일하게 유지합니다.

- **Technical Details**: Best-fit Packing은 길이 인식(length-aware) 조합 최적화(combinatorial optimization)를 통해 문서를 훈련 시퀀스로 패킹하는 방식입니다. 이 방식은 베스트핏 감소(Best-Fit-Decreasing) 알고리즘을 사용하고, 이 알고리즘을 최적화하여 수십억 개의 문서를 효율적으로 처리할 수 있도록 합니다. 이는 문서를 더 긴 문맥으로 다룰 수 있게 하여, 학습 과정에서 문맥 기반의 오류를 감소시키고, 모델의 예측 성능을 향상시키는 것에 기여합니다.

- **Performance Highlights**: 실험 결과, Best-fit Packing을 사용하여 훈련된 모델은 읽기 이해(reading comprehension)에서 상대적으로 4.7%, 문맥 따르기(context following)에서 16.8%, 프로그램 합성(program synthesis)에서 9.2%의 성능 향상을 보였습니다. 또한, 이 방식은 폐쇄 도메인 환각(closed-domain hallucination)을 최대 58.3%까지 효과적으로 줄일 수 있습니다.



### The Landscape of Emerging AI Agent Architectures for Reasoning,  Planning, and Tool Calling: A Survey (https://arxiv.org/abs/2404.11584)
Comments: 13 pages,6 figures,38 references

- **What's New**: 이 논문은 AI 대리인(AI agents)의 구현에 대한 최신 발전을 조사하며, 특히 복잡한 목표를 달성하기 위해 필요한 고도화된 추론 능력과 계획(plan), 도구 사용(tool execution) 능력에 초점을 맞추고 있습니다. 연구는 단일 대리인(single-agent) 및 다중 대리인 시스템(multi-agent systems)의 아키텍처를 개괄하고, 이들 설계에서 나타나는 주요 패턴과 차이점을 식별하며, 목표 달성에 대한 그들의 영향을 평가합니다.

- **Technical Details**: 이 연구에서는 AI 대리인의 주요 구성 요소를 '뇌(brain), 인식(perception), 행동(action)'로 정의하고, 이들이 환경을 이해하고, 추론하며, 작용하는 데 필요한 최소 요건을 충족한다고 설명합니다. 또한, 대리인의 성격(agent persona)과 역할, 도구(tool) 사용 방법이 어떻게 중요한지를 토론하며, 효과적인 대리인 시스템을 위해 추론(reasoning), 계획(planning), 도구 호출(tool calling)의 역할을 상세히 논합니다. 특히, 다양한 방식의 계획 접근법(예: Plan Like a Graph, PLaG)과 도구를 호출하는 복잡한 문제를 해결하는 데 필요한 이유를 설명합니다.

- **Performance Highlights**: AI 대리인 시스템의 성공적인 구현은 강력한 문제 해결 능력에 달려 있으며, 신규 과제에 효과적으로 대응할 수 있어야 합니다. 싱글 에이전트 아키텍처(single agent architectures)는 다른 대리인이나 사용자로부터의 피드백 없이도 잘 정의된 문제에서 뛰어난 성능을 보이는 반면, 다중 에이전트 아키텍처(multi-agent architectures)는 협력과 여러 실행 경로가 필요할 때 더 우수한 성능을 발휘합니다. 본 연구는 ReAct, RAISE, Reflexion, AutoGPT + P 및 LATS과 같은 주목할 만한 단일 대리인 방법론을 강조하며, 이들 방법이 추론과 도구 호출 능력에 기여한 바를 소개합니다.



### GenFighter: A Generative and Evolutive Textual Attack Remova (https://arxiv.org/abs/2404.11538)
- **What's New**: 이 논문에서는 자연어 처리(Natural Language Processing, NLP)에서 Transformer 모델과 같은 딥 뉴럴 네트워크(Deep Neural Networks, DNNs)에 도전하는 적대적 공격(Adversarial attacks)에 대응하기 위한 새로운 방어 전략인 GenFighter를 소개합니다. GenFighter는 트레이닝 데이터의 분포를 학습하고 이를 기반으로 적대적 공격을 탐지하며, 의미는 동일하면서 분포와 일치하는 새로운 인스턴스로 변환하여 앙상블 기법(Ensemble techniques)을 통해 강력하고 통합적인 방어를 제공합니다.

- **Technical Details**: GenFighter는 훈련 데이터 분포를 학습하고, 이를 토대로 이탈하는 잠재적 악의적 인스턴스를 식별합니다. 그 후, 진화적 접근법(Evolutive approach)을 사용하여 의미적으로 동등하지만 훈련 분포와 일치하는 인스턴스를 생성합니다. 이 과정에서 T5 변환기(T5 Transformer), 가우시안 혼합 모델(Gaussian Mixture Model, GMM), 그리고 앙상블 방법이 사용됩니다.

- **Performance Highlights**: GenFighter는 다양한 실험을 통해 RoBERTa와 BERT 모델을 대상으로 하는 가장 인기 있는 적대적 공격 방법들인 PWWS, TextFooler, BERT-Attack 등에 대해 기존 방어 모델보다 높은 정확도와 낮은 공격 성공률을 보여줍니다. 또한, 공격을 수행하기 위해 필요한 쿼리 수가 많아 공격자에게 더 큰 도전이 됩니다. 전이 학습(Transfer learning), 생성/진화 절차(Generative/Evolutive procedure), 그리고 앙상블 방법의 효과적인 통합이 이 방어 전략의 성공을 가능하게 합니다.



### Unifying Bias and Unfairness in Information Retrieval: A Survey of  Challenges and Opportunities with Large Language Models (https://arxiv.org/abs/2404.11457)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)과 정보 검색(IR) 시스템의 통합으로 인해 발생하는 새로운 도전과 기회에 대한 포괄적인 검토를 제공합니다. 특히, LLMs의 통합으로 인해 발생하는 편향성(bias)과 불공정성(unfairness) 문제에 초점을 맞추고 있으며, 본 논문은 이러한 문제들을 분포 불일치(distribution mismatch) 문제로 통합하여 해결책을 모색합니다.

- **Technical Details**: 논문은 LLM과 IR 시스템의 통합이 데이터 수집(data collection), 모델 개발(model development), 결과 평가(result evaluation)의 세 단계에서 어떻게 편향과 불공정성 문제를 야기하는지 설명합니다. 연구는 편향을 객관적 타겟 분포와의 불일치로, 불공정을 인간 가치의 주관적 타겟 분포와의 불일치로 정의하며, 분포 재구성(distribution reconstruction)을 포함한 완화 전략을 분류하고 제시합니다. 이러한 전략에는 데이터 샘플링(data sampling), 데이터 보강(data augmentation), 데이터 필터링(data filtering), 재균형(rebalancing), 규제화(regularization) 및 프롬프팅(prompting)이 포함됩니다.

- **Performance Highlights**: 이 연구는 LLMs와 IR 시스템 통합의 새로운 편향성과 불공정 문제를 체계적으로 분석하고 완화 전략을 제공합니다. 저자들은 대규모 언어 모델을 데이터 소스, IR 모델 향상 도구 및 결과 평가자로 활용하는 새로운 패러다임을 제시하며, 이를 통해 더 신뢰할 수 있고 공정한 정보 검색 시스템을 구축할 수 있을 것으로 기대합니다.



### Research on emotionally intelligent dialogue generation based on  automatic dialogue system (https://arxiv.org/abs/2404.11447)
- **What's New**: 이 연구는 감정 인식 기술을 자동 대화 시스템에 통합하고, 심층 학습(deep learning)과 자연어 처리(natural language processing, NLP) 기술을 사용하여 감정 인공지능을 갖춘 대화 생성 모델을 개발했습니다. 사용자의 감정과 특정 통증 신호를 실시간으로 감지하고 이해하며, 시스템이 공감적 상호작용을 제공할 수 있도록 합니다.

- **Technical Details**: 이 모델은 다양한 감정과 통증의 미묘한 요소를 이해할 수 있는 능력을 갖추고 있으며, '인공지능이 통증을 감지하고 통증 공감을 표현할 수 있는가?'라는 연구 결과를 통합하여 감정 인지 대화 시스템에 대한 기준을 높였습니다.

- **Performance Highlights**: 이 대화 생성 모델은 사용자 경험과 상호작용의 질을 향상시키기 위해 초점을 맞춘 새로운 표준을 설정하며, 복잡한 감정과 통증 상호작용을 실시간으로 처리할 수 있는 능력을 보여줍니다.



### Kathakali Hand Gesture Recognition With Minimal Data (https://arxiv.org/abs/2404.11205)
- **What's New**: 본 연구는 인도 고전 무용극인 카타칼리(Kathakali)의 손동작인 무드라(Mudras)를 인식하는 새로운 접근 방식을 제안합니다. 이 방법은 포즈 추정(pose estimation)을 사용하여 무드라를 벡터(vector) 유사성을 통해 분류하는 24 클래스(classification task) 문제로 다룹니다. 특히, 이 연구는 큰 데이터셋이 필요하지 않은 방법론을 개발하여 AI 기술이 자료 부족으로 인해 적용이 제한된 다양한 분야에 활용될 가능성을 열었습니다.

- **Technical Details**: 본 방법은 일반적으로 사용되는 포즈 추정 기술인 메디아파이프(Mediapipe)를 사용하여 손의 3D 좌표를 추적하고, 이를 벡터화하여 데이터베이스 내 알려진 무드라 벡터와 비교함으로써 가장 유사한 무드라를 식별합니다. 이러한 방식은 추가적인 훈련이나 미세조정(fine-tuning) 없이도 효과적으로 작동하며, 소규모 샘플에서도 유연하게 적용될 수 있습니다.

- **Performance Highlights**: 이 방법은 92%의 높은 정확도를 달성했으며, 이는 도메인 내 다른 모델 기반 작업과 비교하여 유사하거나 더 우수한 성능을 나타냅니다. 또한, 한 개 또는 다섯 개의 샘플로도 작동할 수 있으며 이는 특히 데이터가 부족한 분야에서 큰 장점으로 작용할 수 있습니다.



### Stepwise Alignment for Constrained Language Model Policy Optimization (https://arxiv.org/abs/2404.11049)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)의 안전성과 신뢰성 문제를 해결하기 위해 'Stepwise Alignment for Constrained Policy Optimization (SACPO)'라는 새로운 알고리즘을 제안합니다. SACPO는 보상(reward)과 안전성(safety)을 단계적으로 조정하여 LLM의 인간 가치 정렬(human value alignment)을 추구합니다.

- **Technical Details**: SACPO는 보상에 맞춰진 정책에서 최적의 정책을 직접 도출할 수 있다는 이론적 지지를 바탕으로, 간단하지만 효과적인 정렬 알고리즘인 직접적 선호 최적화(Direct Preference Optimization, DPO)를 활용하여 LLM을 각 메트릭(metric)에 따라 단계적으로 정렬합니다. 이론적 분석을 통해 안전 제약 조건(safety constraint) 위반에 대한 상한선과 근사 최적성(near-optimality)에 대해 제시합니다.

- **Performance Highlights**: 실험 결과 SACPO는 기존의 Safe RLHF 방법론보다 Alpaca-7B 모델의 도움됨(helpfulness)과 무해함(harmlessness) 측면에서 더 나은 미세조정(fine-tuning) 성능을 보였습니다. SACPO는 계산 효율성과 안정성, 유연성을 제공하며, 다양한 알고리즘 및 데이터셋 선택에 대한 유연성을 강조합니다.



### Cross-Platform Hate Speech Detection with Weakly Supervised Causal  Disentanglemen (https://arxiv.org/abs/2404.11036)
- **What's New**: 이 연구는 플랫폼마다 구체적인 표적 레이블(label)의 필요성을 우회하는 새로운 접근 방식인 HATE-WATCH를 제시합니다. 이는 비교적 새로운 'confidence based reweighting' 및 'contrastive regularization'을 통합한 약하게 감독된(weakly supervised) 'causal disentanglement' 프레임워크를 사용하여 플랫폼 간의 일반화 가능한 증오 발언 감지를 향상시킵니다. 다양한 플랫폼에 대한 효과적인 증오 발언 탐지를 위해 불변 표현(invariant representations)으로 입력 특징(input features)을 분리(disentangle)하는 데 성공했습니다.

- **Technical Details**: HATE-WATCH는 대조적 학습(contrastive learning)과 신뢰 기반 샘플 재가중치(confidence-based sample reweighting)를 활용하여 레이블의 노이즈가 학습 과정에 미치는 영향을 최소화하고, 다른 표적 표현(target representations)을 구분할 수 있게 합니다. 이 방법은 추가적인 데이터 요구 사항을 줄이면서, 약하게 레이블링된 데이터셋 내의 내재적인 변동성과 노이즈를 활용하여 증오 발언 탐지의 강건성과 일반화를 향상시킬 수 있습니다. 연구에서는 HATE-WATCH가 네 개의 사회적 미디어 환경(두 개는 표적 레이블이 있고 두 개는 없음)에서의 적용 가능성을 입증하였습니다.

- **Performance Highlights**: HATE-WATCH는 플랫폼 간 증오 발언 감지에서 뛰어난 적응성과 일반화 능력을 보여줌으로써 다른 방법들과 비교하여 우수한 성능을 보였습니다. 이후 실험은 이 프레임워크의 유효성을 입증하고 있으며, 특히 레이블이 부족하거나 노이즈가 많은 환경에서도 효과적으로 작동함을 보여줍니다.



### Advancing Social Intelligence in AI Agents: Technical Challenges and  Open Questions (https://arxiv.org/abs/2404.11023)
Comments: Position Paper, Under Review, 19 pages, 2 figures

- **What's New**: 이 논문은 사회적 지능을 가진 인공지능 에이전트(Social-AI)의 개발에 대해 다룹니다. 사회적 상호 작용에 필요한 감정, 행동, 인지를 이해하고 반응할 수 있는 에이전트를 만드는 것이 주요 목표입니다. 이를 통해 인간 또는 다른 인공 에이전트와의 상호 작용에서 자연스러운 의사소통이 가능해질 것입니다.

- **Technical Details**: 이 연구는 자연어 처리(Natural Language Processing, NLP), 기계 학습(Machine Learning, ML), 로봇 공학, 인간-기계 상호작용, 컴퓨터 비전, 음성 인식 등 다양한 컴퓨팅 분야에서의 진전을 기반으로 합니다. 특히, NLP는 인간의 언어를 통한 사회적 상호작용의 이해와 모델링에서 핵심적인 역할을 하며, 이는 Social-AI의 중요 구성 요소로 작용합니다.

- **Performance Highlights**: 이 논문은 이미 몇 가지 실세계 적용 사례를 소개합니다. 예를 들어, 가상 텍스트 에이전트는 온라인 채팅방에서 공감적 대화를 자극하는 데 사용되었으며, 웨어러블 장치에서의 감정 신호는 웰빙을 지원하는 데 기여했습니다. 또한, 사회적 로봇은 노인 돌봄, 자폐 스펙트럼을 가진 청소년 지원, 학생들의 정신 건강 향상 등에 활용되었습니다.



### Many-Shot In-Context Learning (https://arxiv.org/abs/2404.11018)
- **What's New**: 이 연구는 큰 언어 모델(Large Language Models, LLMs)이 많은 수의 인텍스트 예제(Many-shot In-Context Learning, ICL)를 사용할 때의 성능 향상을 조사합니다. 새롭게 도입된 '강화된 인텍스트 학습(Reinforced ICL)'과 '비감독 인텍스트 학습(Unsupervised ICL)'이라는 두 가지 접근 방식은 인간이 생성한 예제의 필요성을 줄이면서 복잡한 추론 작업에서 유망한 결과를 보여줍니다.

- **Technical Details**: 연구진은 기존의 몇 가지 예제(Few-shot)만을 사용하는 ICL에서 수백 개 이상의 예제를 사용하는 Many-shot ICL로 전환하여 LLMs의 성능을 크게 향상시킬 수 있음을 밝혔습니다. 또한, 강화된 ICL은 모델 생성된 사고 과정(rationales)을 사용하고, 비감독 ICL은 문제-해결 쌍 대신 도메인 특정 문제만을 프롬프트로 제시합니다.

- **Performance Highlights**: 실험 결과, 많은 수의 인텍스트 예제를 사용하는 Many-shot ICL은 수학 문제 해결, 요약, 알고리즘 추론, 기계 번역, 및 감정 분석 등 다양한 작업에서 성능이 크게 향상되었습니다. 특히, Gemini 1.5 Pro 모델을 사용하여 최대 수백만 토큰까지 확장된 컨텍스트에서 상당한 성능 이득이 관찰되었습니다.



### A Survey on Retrieval-Augmented Text Generation for Large Language  Models (https://arxiv.org/abs/2404.10981)
Comments: Ongoing work

- **What's New**: 이 연구는 대규모 언어 모델(LLM)의 한계를 극복하고 자동 생성 결과의 정확성을 향상시키기 위한 Retrieval-Augmented Generation (RAG)에 초점을 맞추고 있습니다. RAG는 검색 기능을 갖춘 실시간 외부 정보를 통합하여 LLM의 출력을 향상하는 방법론으로, 텍스트 도메인에 주로 적용됩니다. 본 논문에서는 RAG의 네 가지 주요 범주인 사전 검색(pre-retrieval), 검색(retrieval), 사후 검색(post-retrieval), 생성(generation)을 구조화하여 RAG 패러다임을 체계적으로 설명하고, 이 분야의 발전과 주요 연구를 분석합니다.

- **Technical Details**: RAG는 전통적 검색 방법과 고급 딥러닝 기술을 결합하여 관련 정보를 효과적으로 검색하고 정확한 응답을 생성하는 두 가지 주요 질문에 주력합니다. 이 연구는 검색과 생성의 주요 기술을 면밀히 분석하고, 특히 BERT와 같은 사전 훈련된 언어 모델을 사용하여 쿼리의 의미적 본질을 포착하고 문서 순위를 정교하게 조정하는 현대 검색 전략을 강조합니다. 또한, 색인(indexing)의 중요성과 데이터 준비의 역할을 주목하며, 정확한 정보 검색을 가능하게 하는 텍스트 정규화와 색인 구조의 세부적인 설정을 설명합니다.

- **Performance Highlights**: RAG는 특히 강화된 검색 효율성과 정확도를 통해 대규모 언어 모델의 성능을 크게 향상시키는 것으로 나타났습니다. 특정 질의에 대응하여 외부 정보를 취합하고 통합하는 기능은 모델의 데이터 적응성을 보다 유연하게 하며, 종종 '환각(hallucinations)'으로 불리는 부정확한 생성 응답을 줄이는 데 기여합니다. 본 논문에서 검토된 연구들은 RAG의 다양한 응용 가능성과 적용 사례를 구체적으로 제시하며, RAG 기술이 텍스트 기반의 AI 도구에서 중요한 역할을 한다는 것을 강조합니다.



### Shears: Unstructured Sparsity with Neural Low-rank Adapter Search (https://arxiv.org/abs/2404.10934)
Comments: 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Industry Track)

- **What's New**: 본 논문에서는 Shears라는 새로운 접근 방식을 소개합니다. 이 방법은 비용 효율적인 희소성(cost-effective sparsity)과 새로 제안된 신경 저랭크 어댑터 탐색 알고리즘인 Neural Low-rank adapter Search (NLS)를 통합하여 매개변수 효율적인 파인 튜닝(parameter-efficient fine-tuning, PEFT) 방법의 효율성을 더욱 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: Shears 방법론은 크게 세 단계로 구성됩니다: 1) 비구조화된 희소화(Unstructured Sparsification), 2) 슈퍼-어댑터 트레이닝(Super-Adapter Training), 3) 서브-어댑터 탐색(Sub-Adapter Search). 각 단계는 대규모 언어 모델(Large Language Models, LLMs)의 효율적인 튜닝과 압축을 가능하게 합니다. 특히, NLS 알고리즘은 저랭크 어댑터(Low-Rank Adapters, LoRA)를 이용하여 모델의 튜닝 과정에서 필요한 매개변수의 수를 줄여 계산 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, Shears는 높은 희소성 수준을 달성하면서도 정확도를 유지하거나 소폭 감소하는 정도에서 개선을 보였으며, 단일 GPU를 이용한 몇 시간 동안의 훈련으로 이러한 결과를 얻었습니다. 이는 다른 방법들과 비교하여 뛰어난 성능을 보여줍니다.



### LLMem: Estimating GPU Memory Usage for Fine-Tuning Pre-Trained LLMs (https://arxiv.org/abs/2404.10933)
Comments: 9 pages, 9 figures, accepted to IJCAI 2024

- **What's New**: LLMem은 다중 GPU를 사용하여 LLM을 세부 조정할 때 GPU 메모리 소비를 평가하고 최적의 분산 세부 조정 방법을 식별하는 새로운 솔루션을 소개합니다. 이는 현존하는 DNNMem 기술을 개선하여, 특히 트랜스포머 기반의 LLM에서 높은 정확도로 GPU 메모리 사용을 예측할 수 있게 합니다.

- **Technical Details**: LLMem은 advanced data parallelism과 tensor parallelism을 포함한 여러 분산 세부 조정 방법의 GPU 메모리 사용을 평가합니다. 이 방법은 각 GPU에 분배된 모델의 파라미터, 기울기, 최적화 상태의 메모리 사용을 다르게 하여 계산 과정 중 GPU 메모리 소비를 예측합니다. LLMem은 GPU 단일 사용과 다중 GPU 설정에서의 메모리 사용을 분석하여, transformer 부분과 language modeling head (lm_head) 부분의 메모리 할당 방식의 차이를 고려합니다.

- **Performance Highlights**: 실험 결과에 따르면, LLMem은 단일 GPU에서 LLM을 세부 조정할 때 최대 GPU 메모리 사용량을 1.6%의 오류율로 정확하게 추정하며, 다중 GPU 설정에서 3.0%의 평균 오류율을 보였습니다. 이는 기존의 DNNMem이 제공하는 42.6%의 평균 오류율보다 매우 개선된 결과입니다.



### Dynamic Self-adaptive Multiscale Distillation from Pre-trained  Multimodal Large Model for Efficient Cross-modal Representation Learning (https://arxiv.org/abs/2404.10838)
Comments: 10 pages

- **What's New**: 이 논문은 사전 학습된 멀티모달 대형 모델에서 효율적인 크로스 모달 표현 학습을 위한 새로운 다이나믹 자체 적응 멀티스케일 증류(dynamic self-adaptive multiscale distillation)를 제안합니다. 이는 멀티스케일 전략을 사용하고, 동적 자체 적응 증류 접근법을 채택하여, 교사(teacher) 모델의 출력 특성만을 활용합니다. 이 방법으로 인해 엄청난 컴퓨팅 자원을 요구하는 기존의 모델들과 달리, 제한된 자원을 가진 환경에서도 고성능의 학생(student) 모델을 훈련시킬 수 있습니다.

- **Technical Details**: 이 연구는 복잡한 교사 모델로부터 단순한 학생 모델로 지식을 전달하는 과정에서, 멀티스케일(distillation at multiple scales)과 다이나믹 자체 조정 증류 손실 밸런서(dynamic self-adaptive distillation loss balancer)라는 새로운 구성 요소를 도입하여, 수동적인 손실 가중치 조정의 필요를 제거하고 증류 과정에서 각 손실 항목을 동적으로 균형을 맞춥니다. 교사 모델과 원본 이미지 레벨 정보만을 사용하여 학생 모델을 통해 크로스 모달 검색 작업(cross-modal retrieval tasks)에서 최고의 성능을 달성하며, 이는 기존의 지역 수준 정보에 의존하던 방법들을 능가합니다.

- **Performance Highlights**: 실험 결과는 우리의 제안된 방법이 크로스 모달 검색 작업에서 최신 최고의 성능을 달성하며, 지역 수준 정보를 사용하는 이전 방법들을 능가함을 보여줍니다. 이는 제한된 자원에서도 고급 멀티모달 기술을 배치할 수 있는 능력을 강조하며, 전반적인 멀티모달 증류 작업(multimodal distillation tasks)에서의 실질적인 혁신과 실행 가능성을 입증합니다.




### Plot2Code: A Comprehensive Benchmark for Evaluating Multi-modal Large Language Models in Code Generation from Scientific Plots (https://arxiv.org/abs/2405.07990)
- **What's New**: Plot2Code는 matplotlib 플롯을 소스 코드로 변환하는 능력을 평가하는 새로운 벤치마크입니다. 이는 멀티모달 대형 언어 모델(MLLMs)의 시각적 코딩 능력을 깊이 있게 평가하기 위해 고안되었습니다. 이 플랫폼은 132개의 고품질 matplotlib 플롯과 각각의 소스 코드, GPT-4로 요약된 설명을 포함합니다.

- **Technical Details**: Plot2Code는 matplotlib 갤러리에서 모집한 여섯 가지 유형의 플롯을 포함하며, 각 플롯에 대한 소스 코드와 자세한 GPT-4 요약 설명이 제공됩니다. 또한, 코드 실행 성공률(code pass rate), 텍스트 일치 비율(text-match ratio), 및 GPT-4V 전체 평점을 포함한 세 가지 자동 평가 메트릭을 제안합니다.

- **Performance Highlights**: Plot2Code를 통한 평가에서 GPT-4V는 10점 만점에 7.68점을 기록, 여전히 개선의 여지가 있는 것으로 나타났습니다. 이 결과는 텍스트-밀집 플롯에서의 시각 코딩에 대한 MLLMs의 의존성을 드러내며, 향후 MLLMs의 발전 지향점을 제시합니다.



### RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors (https://arxiv.org/abs/2405.07940)
Comments:
          To appear at ACL 2024

- **What's New**: 기계 생성 텍스트(machine-generated text) 탐지를 위한 새로운 벤치마크 데이터셋인 RAID에 대한 소개입니다. RAID는 11개의 모델, 8개의 도메인, 11개의 적대적 공격(adversarial attacks), 그리고 4가지 디코딩 전략(decoding strategies)을 포함한 600만 개 이상의 생성물을 포함하고 있습니다. 이 데이터셋은 현재의 탐지 모델들이 예상치 못한 모델과 도메인에 일반화하는 데 어려움을 겪고 있으며 간단한 변경만으로도 성능이 크게 떨어질 수 있음을 보여줍니다.

- **Technical Details**: RAID 벤치마크는 다양한 기계 생성 텍스트 탐지기(detecting detectors)의 강인성을 평가하기 위해 사용됩니다. 여기에는 오픈 소스 및 폐쇄 소스 탐지기가 8개와 4개 포함되어 있으며, 다양한 디코딩 전략 및 적대적 공격을 경험합니다. RAID는 처음으로 반복 패널티(repetition penalties)와 복수의 디코딩 전략을 평가하며, 새로운 모델이나 도메인, 적대적 개입에 대한 탐지기의 반응을 실험적으로 조사합니다.

- **Performance Highlights**: 현재 탐지기들은 기계 생성 텍스트에 대해 높은 탐지 정확도를 주장하지만, RAID를 사용한 평가에서는 이러한 탐지기들이 높은 수준의 적대적 공격과 변형된 샘플링 전략(sampling strategies), 그리고 보지 못한 생성 모델에 취약함을 발견하였습니다. 전략의 변화만으로도 탐지 성능이 크게 저하될 수 있으며, 이는 실제 환경에서 탐지기의 유용성을 크게 떨어트린다는 점을 시사합니다.



### EconLogicQA: A Question-Answering Benchmark for Evaluating Large Language Models in Economic Sequential Reasoning (https://arxiv.org/abs/2405.07938)
- **What's New**: EconLogicQA는 경제, 비즈니스 및 공급망 관리의 복잡한 영역에서 대규모 언어 모델(LLMs)의 순차적 추론 능력을 평가하기 위한 새로운 벤치마크입니다. 이 벤치마크는 단순한 사건 예측을 넘어서 경제 시나리오에서 상호 연결된 여러 이벤트를 식별하고 논리적으로 배열하는 어려운 작업을 요구합니다. EconLogicQA는 논리적, 시간적 이벤트 관계에 대한 깊은 이해를 필요로 하는 경제 기사에서 파생된 다중 이벤트 시나리오로 구성됩니다.

- **Technical Details**: EconLogicQA는 GPT-4를 사용하여 경제 관련 기사에서 주요 포인트를 추출하여 질문을 생성합니다. 이 벤치마크는 현실적 경제 시나리오에서 여러 상호 연결된 이벤트를 논리적으로 배열하는 것을 목표로 합니다. 학습 데이터셋는 사람에 의한 엄격한 리뷰를 통해 정확성과 적합성이 보장됩니다. 또한, 다양한 최첨단 LLMs를 통한 종합적인 평가를 수행하여 논리적 추론 능력을 평가합니다.

- **Performance Highlights**: EconLogicQA는 경제 시나리오 내에서 LLMs의 순차적 추론 능력을 평가하는 데 효과적임을 보여줍니다. 다양한 LLMs, 포함한 기존의 GPT-3.5와 GPT-4 뿐만 아니라 Llama-2, Llama 3와 같은 새로운 모델들이 포함되어 평가되었습니다. 이 벤치마크를 통해 액션과 변화에 대한 계획 및 추론 능력을 평가하기 위한 PlanBench와 같은 기타 벤치마크와 비교하여 LLMs의 복잡한 추론 및 결정 과정의 이해도를 밝혀냈습니다.



### PARDEN, Can You Repeat That? Defending against Jailbreaks via Repetition (https://arxiv.org/abs/2405.07932)
Comments:
          Accepted at ICML 20224

- **What's New**: 이 연구에서는 대형 언어 모델(Large language models, LLMs)의 안전성 문제를 다루고 있는데, 특히 Llama 2와 Claude 2와 같은 모델들이 잠재적 위험을 야기하는 'jailbreak' 상황에 취약한 것에 중점을 두었습니다. 이를 해결하기 위해 'PARDEN'이라는 새로운 접근 방식을 제안하여, 모델 자체가 자신의 출력을 반복함으로써 도메인 이동 없이 입력 또는 출력에서 바람직하지 않은 행동을 감지합니다.

- **Technical Details**: PARDEN은 추가적인 미세 조정(finetuning)이나 모델의 내부 구조에 대한 접근(white box access)을 요구하지 않습니다. 이 접근법은 모델이 단순히 자신의 출력을 반복하도록 요청함으로서, 기존의 자기 검열 행위('Sorry I can't do that')와 분류 형식('Is this prompt malicious') 사이의 도메인 이동을 피합니다.

- **Performance Highlights**: PARDEN은 Llama-2와 Claude-2에 대한 기존의 jailbreak 감지 기준보다 현저히 우수한 성능을 보여줍니다. 예를 들어, Llama2-7B 모델에서 PARDEN은 손상된 행동 데이터셋에서 90%의 높은 True Positive Rate (TPR)와 2.0%의 낮은 False Positive Rate (FPR)을 달성하여, FPR을 24.8%에서 약 11배 감소시켰습니다. 이는 PARDEN이 특히 높은 TPR과 낮은 FPR의 관련 영역에서 매우 강력함을 보여줍니다.



### Russian-Language Multimodal Dataset for Automatic Summarization of Scientific Papers (https://arxiv.org/abs/2405.07886)
Comments:
          12 pages, accepted to AINL

- **What's New**: 이 연구는 러시아어 과학 논문의 멀티모달 데이터셋을 개발하고, 이를 통해 자동 텍스트 요약을 위한 기존 언어 모델의 성능을 검증했습니다. 멀티모달 데이터셋은 텍스트, 표, 그림을 포함하여 과학 논문의 더 풍부한 정보를 제공합니다.

- **Technical Details**: 데이터셋은 총 420개의 과학 논문을 포함하며, 7개의 과학 분야에서 수집된 정보를 포함합니다. 주요 언어 모델로는 SBER의 Gigachat와 Yandex의 YandexGPT가 사용되었으며, 텍스트 요약 작업에 대해 다양한 실험을 수행했습니다.

- **Performance Highlights**: 언어 모델의 성과는 과학 텍스트를 처리하는 데 애매모호성이 있었지만, Gigachat와 YandexGPT 모두 데이터셋에서 유용한 요약을 생성할 가능성을 보여주었습니다. 이 연구는 특히 러시아어 멀티모달 데이터셋의 필요성과 효용을 강조하며, 향후 더 발전된 모델 개발의 중요성을 시사합니다.



### Zero-Shot Tokenizer Transfer (https://arxiv.org/abs/2405.07883)
- **What's New**: 이 연구에서 제안된 새로운 문제인 ZeTT (Zero-Shot Tokenizer Transfer)는 어떤 임의의 토크나이저(tokenizer)에 대해서도 해당하는 임베딩(embeddings)을 즉시 생성할 수 있는 기능을 정의합니다. 이는 언어 모델(LMs)을 그들이 훈련된 특정 토크나이저와 분리(detach)시킬 수 있는 가능성을 열어줍니다.

- **Technical Details**: 제안된 기술은 하이퍼네트워크(hypernetwork)를 사용하여 다양한 토크나이저들의 분포에 대해 훈련하고, 주어진 토크나이저에 대한 임베딩 매개변수(embedding parameters)를 예측하도록 합니다. 이 하이퍼네트워크는 한 번의 훈련으로 효과적인 ZeTT를 가능하게 하며, 몇 퍼센트 정도의 정확도로 성능을 유지합니다.

- **Performance Highlights**: 이 방법은 기존 모델 성능에 근접하면서, 크로스-링귈(cross-lingual)과 코딩 작업에서 눈에 띄게 토큰화된 시퀀스 길이를 줄일 수 있습니다. 또한, 타깃 토크나이저에 대한 빠른 적응을 위해 1B 토큰 미만의 추가 훈련으로 성능 격차를 신속하게 해소할 수 있음을 발견했습니다.



### Reproducing the Metric-Based Evaluation of a Set of Controllable Text Generation Techniques (https://arxiv.org/abs/2405.07875)
Comments:
          The Fourth Workshop on Human Evaluation of NLP Systems (HumEval 2024) at LREC-COLING 2024

- **What's New**: 이 보고서는 단일 속성 및 다수 속성 제어 가능한 텍스트 생성(Controlled Text Generation, CTG) 기법에 대한 메트릭 기반 평가를 다시 실행하여 원래의 결과와 일치하지 않는 경우들을 보여줍니다. 특히 Gu et al. (2022, 2023)의 최신 시스템을 사용한 복제 연구에서 이러한 차이를 경험하였으며, 이는 QRA++ 도구를 사용하여 정량화하였습니다.

- **Technical Details**: Gu et al.이 제안한 여러 CTG 기술, 특히 MultiCTG와 PriorCTG, 및 그 변형들(PriorCTG+extend, PriorCTG+optim)의 성능을 재평가했습니다. 이들 각각은 IMDb, AGNews, Jigsaw 데이트셋에서 훈련되어 감정(sentiment), 주제(topic), 독성(toxicity)의 특성을 제어할 수 있습니다. 평가 메트릭은 속성 관련성, 다중 속성 제어 성능, 그리고 문자열의 고유성을 포함합니다. QRA++ 측정 기준에 따라 이러한 결과들은 유형 1(Type I), 유형 2(Type II), 및 유형 4(Type IV) 결과로 분류되어 평가되었습니다.

- **Performance Highlights**: 재현 실험에서는 원본 저자들이 제공한 코드와 모델 체크포인트를 사용하였음에도 불구하고 원본 결과와 재현 결과가 일치하지 않는 경우가 대부분이었습니다. 특히, 독성(toxicity) 제어 성능에서는 원본 논문과 평가 스크립트 간의 불일치가 발견되었습니다. QRA++를 적용한 결과, 재현된 평가의 결과가 원래의 결과와 어느 정도 유사한지를 정량적으로 비교할 수 있었습니다. 이는 측정된 차이를 통해 재현 가능성의 정도를 확인할 수 있었으며, 이는 연구의 신뢰성을 강화하는 중요한 단계입니다.



### DEPTH: Discourse Education through Pre-Training Hierarchically (https://arxiv.org/abs/2405.07788)
Comments:
          28 pages, 10 figures, 8 tables

- **What's New**: 새로운 연구에서는 언어 모델(Language Models, LM)이 담화 수준에서의 언어 이해에 어려움을 겪는 문제를 해결하기 위해 DEPTH라는 새로운 인코더-디코더 모델을 소개하고 있습니다. DEPTH는 문장 무작위 섞기 해제(Sentence Un-Shuffling)와 범위 오염(Span-Corruption)이라는 두 가지 목표를 통합하여, 문장과 서브워드(sub-word) 수준의 의존성을 모두 표현할 수 있도록 훈련됩니다.

- **Technical Details**: DEPTH는 T5의 구조를 기반으로 하며, 계층적 주의 메커니즘(hierarchical attention mechanism)을 사용하여 보다 섬세한 의미적 의존성과 문장 간의 관계를 포착합니다. 이 모델은 T5의 기존 토크나이저와 문장 분할이 필요한 문장 무작위 섞기 해제 작업을 결합한 새로운 토크나이저를 도입합니다. 또한, 이 모델은 T5와 SLM의 손실 함수를 통합하여 전통적인 teacher-forcing 최적화와 담화 지향 사전 훈련 목표를 결합합니다.

- **Performance Highlights**: DEPTH 모델은 기존 T5 모델과 비교하여 담화 레벨의 표현을 더 빠르게 학습할 수 있었고, GLUE, DiscoEval, NI와 같은 벤치마크에서 뛰어난 성능을 보였습니다. 특히, 범위 오염 손실에서 기존 T5 모델을 뛰어넘는 결과를 보여주었으며, 서둘러 문장 무작위 섞기 해제 목표를 추가하였음에도 불구하고 이러한 성과를 달성했습니다. DEPTH는 문법, 의미, 담화 능력이 필요한 다양한 다운스트림(downstream) 작업에서 빠르게 학습할 수 있는 능력을 입증하였습니다.



### A Comprehensive Analysis of Static Word Embeddings for Turkish (https://arxiv.org/abs/2405.07778)
- **What's New**: 이 논문에서는 터키어에 대한 문맥적(contextual) 및 비문맥적(non-contextual) 워드 임베딩 모델의 성능을 비교하고 평가합니다. 문맥적 임베딩을 정적(static) 임베딩으로 변환하는 새로운 방법을 제안하며, 이는 터키어 자연어 처리(Natural Language Processing, NLP) 분야에서 처음 시도된 연구입니다.

- **Technical Details**: 연구팀은 Word2Vec, FastText, GloVe, BERT, ELMo와 같은 다양한 워드 임베딩 모델을 사용하여 터키어 데이터셋에 적용하였습니다. 또한 문맥적 임베딩을 정적 임베딩으로 변환하기 위한 두 가지 전략인 풀링(pooling) 작업과 문맥 정보 통합 방법을 사용하였습니다. 이러한 임베딩들의 품질은 유사성(similarity) 및 유추(analogy) 작업을 통해 내부적으로 평가되었으며, 감성 분석(sentiment analysis), 개체명 인식(named entity recognition, NER), 품사 태깅(part-of-speech, PoS tagging) 등의 다운스트림 NLP 작업을 사용하여 외부적으로 평가되었습니다.

- **Performance Highlights**: 분석 결과, 문맥적 모델에서 변환된 정적 임베딩이 다양한 NLP 작업에 효과적임을 보여줍니다. 특히 문맥을 고려한 임베딩 모델인 BERT와 ELMo는 터키어의 복잡한 형태론적 특성을 잘 처리하는 것으로 나타났습니다. 연구 결과는 또한 터키어 워드 임베딩 저장소를 구축하여 연구자와 실무자가 사용할 수 있는 자원을 제공합니다.



### Challenges and Opportunities of NLP for HR Applications: A Discussion Paper (https://arxiv.org/abs/2405.07766)
Comments:
          10 pages, 2 figures, 1 table

- **What's New**: 이 논문은 기계 학습(Machine Learning, ML)과 자연 언어 처리(Natural Language Processing, NLP) 분야에서 최근 수십 년 간 이루어진 발전을 바탕으로, 채용 및 인사 자원 관리(Human Resource Management, HRM) 분야에서의 텍스트 분석 사용 사례를 검토합니다. 특히 COVID-19 팬데믹으로 인해 지식 및 사무직 근로자의 재택 근무가 확산되면서 인사 관리 분야에서 디지털화가 가속화되었습니다. 이에 따라, HRM 분야에서 AI의 적용이 증가하고 있으며, 이 논문은 인사 자원(Human Resources, HR)에서 NLP 기술의 적용 가능성과 문제점을 심층 분석합니다.

- **Technical Details**: 이 연구는 자동 이력서/이직 관련 정보 채굴(Automatic Resume/CV Mining, AMM), 자동 이력서 분석(Automatic Resume/CV Screening, ARS), 지원자 추적 시스템(Applicant Tracking System, ATS), 경쟁력 정보(Competitive Intelligence, CI) 등의 기술을 다룹니다. 또한, 고용 과정 중 평판 자동 생성, 직원 자가 서비스 챗봇 등 다양한 NLP 응용 프로그램을 소개하며, 이력서에서 필요한 데이터를 추출하는 정보 추출(Information Extraction, IE) 기술이 중점적으로 다뤄집니다. 특히, 여러 언어로 된 이력서에서 데이터 추출이 가능한 시스템에 대해서도 언급합니다.

- **Performance Highlights**: 특정 NLP 기술을 이용한 이력서 분석과 작업 게시 생성에서 중요한 성과를 올린 연구 사례들도 소개됩니다. 예를 들어, SVM(Support Vector Machines)과 HMM(Hidden Markov Models)을 사용한 중국어 이력서의 정보 추출에서 SVM이 더 높은 성능을 보였으며, 특히 네트워크(Network) 기반 접근을 통해 필요한 기술 세트를 마이닝하고 직업 승진 경로를 생성하는 등의 연구도 포함되어 있습니다.



### TANQ: An open domain dataset of table answered questions (https://arxiv.org/abs/2405.07765)
Comments:
          10 pages

- **What's New**: 이 논문에서는 다양한 정보 소스에서 정보를 수집하여 테이블을 구성해야 하는 답변을 필요로 하는 첫 번째 오픈 도메인(question-answering, QA) 데이터세트인 TANQ(Table Answering in an Open, Quantitative Domain)를 소개합니다. 이 데이터세트는 정교한 테이블로 답변을 제공하여 복잡한 질문에 답하는 새로운 방법인 복수의 문서에서 데이터를 검색하고 집계하는 여러 단계를 포함합니다.

- **Technical Details**: TANQ 데이터세트 생성은 QAMPARI를 시드 데이터셋으로 사용하고 Wikidata 및 Wikipedia 코퍼스를 데이터 소스로 활용하는 다섯 단계의 자동 데이터 수집 과정을 통해 이루어졌습니다. 모델 평가는 PaLM-2222를 사용하여 데이터 수집 및 처리의 각 하위 단계를 자동으로 평가했습니다. LLMs(Large Language Models)는 클로즈드 북(closed-book), 오픈 북(open-book), 그리고 오라클(oracle) 설정에서 평가되었으며, 오라클 설정에서 GPT4는 29.1의 F1 스코어를 달성했습니다.

- **Performance Highlights**: GPT4는 오라클 설정에서 29.1의 F1 스코어를 달성하여 인간의 성능과 19.7 포인트 차이를 보였습니다. 이 데이터세트에는 정보 검색(retrieval), 다중 문서 이해(multi-hop reasoning), 수학 연산(math operations), 그리고 단위 변환(unit conversions)과 같은 다양한 기술이 필요하며, 분석 결과, 모델 생성 답변에서 일반적인 실패 사례를 확인할 수 있어 TANQ가 많은 도전과제를 안고 있는 복잡한 작업임을 시사합니다.



### LGDE: Local Graph-based Dictionary Expansion (https://arxiv.org/abs/2405.07764)
- **What's New**: 본 논문에서는 Local Graph-based Dictionary Expansion (LGDE)이라는 새로운 방법을 제안합니다. LGDE는 매니폴드 학습(manifold learning)과 네트워크 과학(network science)을 활용하여, 주어진 시드 단어(seed words)로부터 더 많은 연관 키워드를 발견하는 데이터 기반 방법입니다. 이 방법은 특히 높은 회수율(recall)과 정밀도(precision)를 필요로 하는 정보 검색 작업에 유용합니다.

- **Technical Details**: LGDE는 도메인 특화 워드 임베딩(domain-specific word embeddings)을 기반으로 단어 유사성 그래프(word similarity graph)를 생성합니다. 그 후 지역 커뮤니티 탐지(local community detection)를 통해 시맨틱 네트워크(semantic network) 내에서 관련어를 찾아내는 방식을 사용합니다. 이 과정에서 그래프 확산(graph diffusion)을 이용하여 다단계 단어 연관 경로를 통한 더 넓은 단어의 발견이 가능합니다.

- **Performance Highlights**: Reddit과 Gab에서 수집된 혐오 발언 관련 데이터셋을 사용한 실험에서 LGDE는 기존의 직접적인 단어 유사도를 활용한 방법들보다 더 높은 F1 스코어(F1 score)를 달성했습니다. 또한, 4chan에서의 음모 이론 관련 사례에서도 LGDE는 추가적으로 관련된 단어를 발견하는 데 있어 정량적인 우위를 보였습니다.



### LlamaTurk: Adapting Open-Source Generative Large Language Models for Low-Resource Languag (https://arxiv.org/abs/2405.07745)
- **What's New**: 이 연구는 영어에 대한 기본 훈련을 받은 대형 언어 모델들을 저자원 언어에 적용하고자 할 때 사용할 수 있는 다양한 방법을 탐구합니다. 연속적인 훈련(continual training), 지시사항 세분화 훈련(instruction fine-tuning), 작업 특화 훈련(task-specific fine-tuning), 그리고 어휘 확장(vocabulary extension) 방법이 포함됩니다. 특히, 터키어와 같은 저자원 언어에 초점을 맞추고 있습니다.

- **Technical Details**: 모델의 연속적인 훈련은 새로운 데이터 코퍼스를 통합하여 훈련 단계를 확장하는 과정입니다. 이 연구에서는 Low-Rank Adaptation (LoRA) 기법을 사용하여 자원이 제한된 상황에서 효율적인 훈련이 가능하도록 하였습니다. 지시사항 세분화 훈련은 모델이 주어진 지시사항에 따라 정확한 응답을 생성하도록 하는 훈련 방법이며, 작업 특화 훈련은 하위 작업의 성능을 향상시키는 데 초점을 맞추고 있습니다. 그러나 어휘 확장은 상당한 이점을 보여주지 않았습니다.

- **Performance Highlights**: 계속적인 훈련은 언어 이해를 개선하는 데 효과적이며, perplexity 점수를 통해 그 효과를 확인할 수 있습니다. 작업 특화 훈련은 일반적으로 하위 작업의 성능을 향상시키는 데 기여합니다. 큰 모델 사이즈에서는 몇 가지 샘플로도 성능을 개선할 수 있지만, 다국어 모델은 모노어 모델에 비해 성능이 떨어지는 경향을 보입니다.



### Does Dependency Locality Predict Non-canonical Word Order in Hindi? (https://arxiv.org/abs/2405.07730)
Comments:
          Accepted at CogSci-2024 with full paper publication

- **What's New**: 이 연구는 힌디어에서 비정형어순(OSV)을 결정하는 인지 요인들을 분석합니다. 주로 주어-목적어-동사(SOV) 순서를 사용하는 힌디어에서, 비정형 문장 구조가 어떻게 효과적으로 처리될 수 있는지에 대한 이해를 제공합니다. 같은 맥락에서 의외로 나타나는 다양한 구성 요소 순서의 선호도를 결정하는 요소로서 예측 가능성의 역할에 주목합니다.

- **Technical Details**: 이 연구에서는 힌디-우르두 트리뱅크 코퍼스(Hindi-Urdu Treebank corpus, HUTB)를 사용하여 자연스럽게 발생하는 문장과 인공적으로 생성된 문장 사이를 구분할 수 있는 분류기를 구현했습니다. 이 분류기는 의존성 길이(dependency length), 놀람 정도(surprisal), 정보 상태(information status) 등 다양한 담화 기반 및 인지적 특징을 활용하여 예측을 수행합니다.

- **Performance Highlights**: 분석 결과, 비정형 문장에서 의존성 길이를 최소화하는 경향은 있지만, 이것이 코퍼스 문장을 식별하는 데에 있어 놀람 정도나 정보 주어진 정도(givenness)보다 유의미하게 기여하지는 않는 것으로 보입니다. 대신, 담화의 예측 가능성이 구성 요소 순서 선호의 주된 결정 요인으로 부각되었으며, 이는 44명의 힌디어 원어민을 대상으로 한 인간 평가를 통해서도 뒷받침됩니다.



### Quantifying and Optimizing Global Faithfulness in Persona-driven Role-playing (https://arxiv.org/abs/2405.07726)
- **What's New**: 이 논문은 Persona-driven role-playing (PRP)에 대한 새로운 접근 방식을 제시하여 AI 캐릭터들이 개인 정보를 정확하게 반영할 수 있도록 하는 새로운 평가 기준인 '활성-수동 제약 조건(Active-Passive-Constraint, APC) 점수'를 도입합니다. 이 점수는 인-자연어 추론(NLI) 점수를 이용하여 측정되며, 이러한 방식은 AI 캐릭터의 대화 반응이 개인 정보와 얼마나 일치하는지 미세하게 평가할 수 있게 합니다.

- **Technical Details**: APC 점수는 개인 정보 (persona statement)와 쿼리의 관련성을 분석하여 활성 및 수동 제약 조건으로 분류하고, 이를 기반으로 AI 캐릭터의 응답이 관련 있는 제약 조건에 의해 유도되고 관련 없는 제약 조건과 모순되지 않는지를 평가합니다. 이 평가는 GPT-4에서 추출한 소형 판별기를 활용하여 효율적으로 수행됩니다.

- **Performance Highlights**: APC 점수는 인간 평가와 높은 상관 관계를 보이며, 직접 선호 최적화(Direct Preference Optimization, DPO)를 통해 PRP 방법의 신뢰성을 높이는 데 사용됩니다. 실제 실험을 통해 APC 기반 DPO가 다양한 인물의 개인 정보를 기반으로 한 AI 캐릭터 생성에 있어 뛰어난 성능을 보이며, 다른 기술과도 잘 통합될 수 있음을 입증합니다.



### OpenLLM-Ro -- Technical Report on Open-source Romanian LLMs trained starting from Llama 2 (https://arxiv.org/abs/2405.07703)
- **What's New**: 최근 몇 년 동안 대규모 언어 모델(Large Language Models, LLMs)은 다양한 작업에서 거의 인간과 같은 성능을 달성했습니다. 일부 LLM은 다중 언어 데이터로 훈련되었지만, 대부분의 훈련 데이터는 영어로 되어 있어서 다른 언어보다 영어에서의 성능이 월등히 뛰어납니다. 이 문서는 루마니아어에 특화된 첫 번째 기초적이고 대화형 LLM을 훈련하고 평가하는 접근법을 소개합니다.

- **Technical Details**: 이 연구는 루마니아어 전문 대화형 LLM을 개발하기 위해 다양한 루마니아어 데이터 소스를 활용합니다. 대부분의 기존 LLM이 영어 중심의 데이터로 훈련을 받는 상황에서, 이 모델은 루마니아 문화와 언어의 미묘한 특성을 반영할 수 있는 특화된 데이터셋을 사용하여 훈련됩니다.



### Age-Dependent Analysis and Stochastic Generation of Child-Directed Speech (https://arxiv.org/abs/2405.07700)
Comments:
          Accepted for publication in Proc. 45th Annual Meeting of the Cognitive Science Society (CogSci-2024)

- **What's New**: 이 연구는 특정 언어 모델(Language Model, LM)을 사용하여 어린이 연령에 따른 언어적 특성이 변화하는 어른이 어린이에게 사용하는 말, 즉 어린이지향 언어(Child-directed speech, CDS)를 모델링하는 새로운 접근법을 제시합니다. 또한, 생성된 언어 모델은 연령에 맞는 CDS 기록을 합성적으로 생성할 수 있습니다. 이는 기존 데이터 세트의 크기를 확장하는 데 유용합니다.

- **Technical Details**: 연구팀은 북미 영어 코퍼스인 CHILDES 데이터베이스에서 얻은 CDS 기록과 수신자 어린이의 연령 데이터를 활용하여 언어 모델을 훈련시켰습니다. 이를 통해 어린이의 연령에 따라 변화하는 CDS의 언어적 특성을 파악하고, 이러한 특성을 반영하는 언어 모델을 생성했습니다.

- **Performance Highlights**: 생성된 언어 모델은 다양한 연령의 어린이에게 실제로 사용된 연설과 비교했을 때 연령에 따른 변화를 잘 반영하는 것으로 나타났습니다. 다만, 유효 어휘 크기(effective vocabulary size)에서 약간의 차이를 보였습니다. 추가로, 연구팀은 CHILDES에서 CDS의 연령에 따른 언어적 특성을 체계적으로 분석하고, 어린이의 나이가 증가함에 따라 언어적 특성이 어떻게 변화하는지를 보여주는 결과도 제공합니다.



### An Empirical Study on the Robustness of Massively Multilingual Neural Machine Translation (https://arxiv.org/abs/2405.07673)
Comments:
          12 pages, 6 figures

- **What's New**: 이 연구는 인도네시아-중국어 번역의 강건성(robustness)에 초점을 맞추고 있으며, 자연적으로 발생하는 노이즈(noise)에 대한 번역의 강건성을 평가합니다. 이는 대규모 다국어 신경망 기계 번역(MMNMT) 모델을 사용하여 첫 시도로 알려져 있습니다. 연구자들은 인도네시아-중국어 번역을 위한 새로운 강건성 평가 벤치마크 데이터셋을 만들었습니다. 이 데이터셋은 다양한 크기의 NLLB-200 모델을 사용하여 자동으로 중국어로 번역되었습니다.

- **Technical Details**: 연구팀은 NLLB-200 모델을 사용하여 인도네시아어에서 중국어로의 번역을 자동화했으며, 번역된 문장에서 번역 오류 유형을 감지하고 분류했습니다. 또한, BLEU 및 CHRF++ 메트릭을 사용한 자동 평가와 다차원 품질 측정 지표(MQM)를 통한 인간 평가를 수행했습니다. 소스 언어의 노이즈를 수동으로 식별하고 10개 그룹으로 분류하여 번역 오류 유형과 노이즈 유형 간의 상관관계를 분석했습니다.

- **Performance Highlights**: 이 연구는 다양한 크기의 MMNMT 모델을 통해 번역 오류 유형과 노이즈 유형 간의 상관관계가 어떻게 변화하는지에 대한 심층 분석을 제공합니다. 또한 자동 평가 지표와 인간 평가 지표 사이의 관계를 파악하는 데 있어 중요한 정보를 제공합니다. 데이터셋은 다양한 타입의 자연 발생 노이즈를 포함하여 신뢰도 있는 평가를 위한 좋은 자료가 될 것입니다.



### COBias and Debias: Minimizing Language Model Pairwise Accuracy Bias via Nonlinear Integer Programming (https://arxiv.org/abs/2405.07623)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 문맥 기반 오류 편향(Contextual Oddity Bias, COBias)을 조명하고, 이를 해결하기 위해 비선형 정수 프로그래밍(Nonlinear Integer Programming, NIP)을 적용한 새로운 접근 방식을 제안합니다. COBias는 클래스 A의 정확도와 그 클래스가 잘못 예측하는 '기이한' 클래스 간의 정확도 차이를 설명합니다. 이 연구는 LLMs가 개별 클래스 정확도에 있어 큰 차이를 보여주는 것을 밝히며, DNIP (Debiasing as Nonlinear Integer Programming) 방법을 통해 클래스별 확률을 조정함으로써 이러한 편향을 줄이고 전체 정확도를 향상시키는 방법을 제시합니다.

- **Technical Details**: COBias는 문맥적 기이함의 편향을 나타내며, 특정 클래스 예측이 다른 클래스에 비해 과도하게 이루어지는 경향을 뜻합니다. DNIP는 이러한 문제를 최소화함과 동시에 전체 정확도를 최대화하기 위해, 시뮬레이티드 어닐링(simulated annealing)을 포함한 비선형 정수 최적화 프로그래밍 기법을 적용합니다. 세 가지 대형 언어 모델(GPT-2-XL, Llama-2-7B, Llama-2-13B)과 일곱 가지 자연어 처리(Natural Language Processing, NLP) 분류 작업에 대한 평가에서 이 방법은 기존의 ICL(In-Context Learning) 방식에 비해 COBias를 평균 43%에서 16%로 크게 줄이고, 정확도는 52%에서 64%로 향상시켰습니다.

- **Performance Highlights**: DNIP 방법은 COBias를 평균 43%에서 16%로 감소시켰으며, 전체 정확도는 52%에서 64%로 상당히 향상되었습니다. 이는 기존의 ICL 접근 방법을 사용했을 때보다 눈에 띄는 개선을 보여주며, LLM의 예측 정확도와 신뢰성을 개선하는 데 있어 중요한 진전을 의미합니다. 제안된 모델은 LLM들이 다양한 NLP 분류 작업에서 일관되게 큰 COBias 점수를 나타낸다는 사실을 발견하고, 페어와이즈 클래스 정확도 차이를 모델링하는 것이 LLM 예측의 정확도와 신뢰성 향상을 위한 방향성을 제시합니다.



### ViWikiFC: Fact-Checking for Vietnamese Wikipedia-Based Textual Knowledge Sourc (https://arxiv.org/abs/2405.07615)
- **What's New**: 본 논문에서는 베트남어 위키백과와 관련된 팩트 체킹을 위한 최초의 대규모 개방형 코퍼스인 ViWikiFC를 제안합니다. ViWikiFC는 20,916개의 주장(claims)을 수동으로 주석을 달아 생성했으며 이는 팩트 체킹 작업에 있어 베트남어 자원의 부족함을 해결하기 위한 것입니다.

- **Technical Details**: 이 코퍼스는 73개의 위키백과 기사에서 추출된 증거에 근거하여 만들어졌으며, 주장을 분류하기 위해 SUPPORTS, REFUTES, 및 NOTENOUGHINFORMATION(NEI) 세 가지 레이블을 사용합니다. 코퍼스 생성 과정은 베트남어 모국어 사용자들에 의해 신중하게 수행되었으며, Fleiss’ κ (카파) 합의 효율이 95.87%에 달하는 고품질의 데이터를 확보하였습니다. 또한, 증거 검색(evidence retrieval)과 판결 예측(verdict prediction) 작업을 포함한 두 가지 실험을 수행하였습니다.

- **Performance Highlights**: 증거 검색 작업에서는 BM25가 SUPPORTS 레이블에서 88.30%, REFUTES에서 86.93%, NEI 레이블에서는 56.67%의 정확도를 달성했습니다. 판결 예측 작업에서는 InfoXLM (Large) 모델이 F1 스코어로 86.51%의 성능을 보였습니다. 이 실험 결과들은 베트남어 팩트 체킹 모델들에게는 여전히 도전적인 데이터셋임을 보여줍니다.



### NoiseBench: Benchmarking the Impact of Real Label Noise on Named Entity Recognition (https://arxiv.org/abs/2405.07609)
Comments:
          data available at this https URL

- **What's New**: 이 연구에서는 이름 있는 개체 인식(Named Entity Recognition, NER) 모델 훈련에 사용되는 데이터에서 발생하는 실제 레이블 잡음의 영향을 분석하기 위해 NoiseBench라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 전문가 오류, 군중 소싱 오류, 자동 주석 오류 및 LLM 오류를 포함한 6가지 유형의 실제 노이즈로 만들어진 깨끗한 훈련 데이터로 구성됩니다. 이는 현재까지 대부분의 연구에서 사용된 시뮬레이션 노이즈와 비교하여 현저히 더 도전적인 환경을 제공합니다.

- **Technical Details**: NoiseBench는 기존의 CoNLL-03 데이터셋의 부분집합을 기반으로 하며, 400개 문서에서 추출한 5,885개 문장으로 구성되어 있습니다. 이 데이터셋은 전문가 주석, 군중소싱 에러, 원격 감독(distant supervision), 약한 감독(weak supervision), 그리고 LLMs 에 의한 주석 등 다양한 형태의 실제 노이즈를 통해 7가지 다른 변형으로 구성됩니다. 이 벤치마크는 레이블 잡음의 다양한 유형에 대해 노이즈 견고 모델을 평가하는 데 사용될 수 있습니다.

- **Performance Highlights**: 실제 노이즈는 시뮬레이션된 노이즈에 비해 훨씬 더 큰 도전을 제시하며 현재의 최신 노이즈 견고 학습 접근법은 이론적 상한선에 크게 미치지 못한다는 것을 발견했습니다. 특히, 전문가의 실수에서 오는 노이즈는 가장 낮은 잡음 비율을 보이는 반면, 군중소싱에서 발생하는 Crowd 변형은 36.6%로 상당히 높은 잡음 비율을 보였습니다. 이러한 실제 잡음 조건은 기존의 NER 모델들이 기존 가정과 달리 실제 상황에서의 성능 저하를 보일 수 있음을 시사합니다.



### Using Model-Theoretic Approaches to Uncover Linguistic Organization (https://arxiv.org/abs/2405.07597)
- **What's New**: 이 논문은 Kaqchikel, Karuk, 그리고 Yurok 언어에서의 다중표시자(pluractional markers)를 고찰합니다. 발리어(Balinese)처럼, 이 언어들은 중복(reduplication)을 통해 한 종류의 다중성을 표시하고, 비중복(non-reduplicative) 접사(affixation)를 통해 다른 종류의 다중성을 표시합니다. 이 연구는 표면에 분명하지 않은 언어적 구조를 인식할 수 있도록 돕는 렌즈로서 모델 이론적 접근(model-theoretic approaches)을 언어에 적용하는 개념 증명(proof-of-concept)으로 작용합니다.

- **Technical Details**: 연구진은 모델 이론적 접근을 사용하여 언어 구조의 숨겨진 층을 드러내며, 언어의 다중성 표시 방식에 대해 새로운 이해를 제공합니다. 특히, 중복과 비중복 접사를 사용하는 방식을 분석함으로써, 다중성이 어떻게 다르게 표현될 수 있는지를 설명합니다.



### Thai Universal Dependency Treebank (https://arxiv.org/abs/2405.07586)
- **What's New**: 태국어의 자동 의존성 분석은 그동안 잘 탐구되지 않았습니다. 이 논문에서는 태국어에 대한 새로운 최대 규모의 트리뱅크인 Thai Universal Dependency Treebank (TUD)를 소개하고 있으며, 이 트리뱅크는 3,627개의 트리로 구성되어 있습니다. 또한 이 트리뱅크를 사용하여 사전 훈련된 트랜스포머(pretrained transformers)를 인코더로 통합한 의존성 파싱 모델을 벤치마킹하였습니다.

- **Technical Details**: TUD는 Universal Dependencies (UD) 프레임워크를 준수하여 주석이 달린 새로운 태국어 트리뱅크입니다. 연구팀은 태국어 파싱을 위해 transformer-based 모델들을 구현하고 테스트했습니다. 이들은 특히 Thai-PUD와 TUD 데이터셋에 대해 훈련되었으며, 최신 기술을 적용하였습니다. 이 모델들은 기존의 CNN 또는 LSTM 기반 모델 대신 사전 훈련된 트랜스포머를 사용함으로써 태국어 의존성 분석의 정확도를 높이는데 기여할 수 있습니다.

- **Performance Highlights**: 이 연구에서 구현된 모델들은 이전 연구에서 보고된 다른 모델들을 성능면에서 능가하는 것으로 나타났습니다. 더 나아가, 태국어에 특화된 의존성 파서(dependency parsers) 개발을 위한 구성요소의 최적 선택에 대한 통찰력을 제공합니다. 이러한 성과는 태국어 의존성 파싱에 있어 새로운 벤치마크를 설정하며, 향후 연구의 기본 모델로서 유용하게 사용될 수 있습니다.



### MuMath-Code: Combining Tool-Use Large Language Models with Multi-perspective Data Augmentation for Mathematical Reasoning (https://arxiv.org/abs/2405.07551)
Comments:
          The state-of-the-art open-source tool-use LLMs for mathematical reasoning

- **What's New**: 새로운 연구에서는 대규모 언어 모델(LLMs)이 외부 Python 해석기와 통합되어 수학적 추론 능력을 개선하는 동시에, 수학 추론 데이터를 확장하여 비도구적 방법을 선택했습니다. 이 논문에서는 여러 관점에서 데이터 확장 방법을 적용하여 새로운 수학 문제를 포함시키고 코드 중첩 솔루션을 합성하는 접근 방식을 소개합니다. 그 결과, MuMath-Code (μ-Math-Code) 모델이 개발되었습니다.

- **Technical Details**: MuMath-Code는 Llama-2 모델을 기반으로 하여, 두 단계의 훈련 전략을 사용합니다. 1단계에서는 순수한 CoT(Chain of Thought) 데이터에 대해 Llama-2를 미세조정하고, 2단계에서는 코드 중첩 데이터에 대해 훈련을 진행합니다. 이 과정을 통해 완성된 MuMath-Code는 코드를 생성하고 외부 Python 해석기와 상호 작용하여 실행 결과를 얻습니다.

- **Performance Highlights**: MuMath-Code 모델은 다양한 크기의 버전에서 높은 성능을 달성하였습니다. 더 작은 7B 모델은 GSM8K에서 83.8%, MATH에서 52.4%의 결과를, 더 큰 70B 모델은 GSM8K에서 90.7%, MATH에서 55.1%의 새로운 최고 성능을 기록했습니다. 이는 도구 사용(tool use)과 데이터 확장(data augmentation)의 결합이 효과적임을 증명합니다.



### EMS-SD: Efficient Multi-sample Speculative Decoding for Accelerating Large Language Models (https://arxiv.org/abs/2405.07542)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)의 추론 속도를 개선하기 위한 새로운 방법인 '효율적인 다중 샘플 추측 디코딩(Efficient Multi-sample Speculative Decoding, EMS-SD)'을 제안합니다. 이 방법은 예상 디코딩의 패딩 토큰(token) 문제를 해결하여 계산 및 메모리 접근 부담을 크게 줄이는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 KV 캐시의 불일치를 처리하고 입력 토큰의 패딩을 없애는 두 가지 주요 구성 요소를 포함합니다. 'unpad KV cache'와 'unpad input tokens' 방식을 사용하여 각각의 샘플마다 발생하는 토큰 수의 차이를 동적으로 관리하므로, 모든 샘플이 동일한 토큰 수를 가질 필요가 없어지면서도 효율적인 토큰 처리가 가능해집니다.

- **Performance Highlights**: 실험 결과, 이 새로운 접근 방식은 기존의 방식에 비해 상당한 속도 향상을 보여주었으며, 특히 다중 샘플(multi-sample) 시나리오에서 더 높은 속도 향상 비율(speedup ratio)을 달성하였습니다. 이는 EMS-SD가 패딩 토큰을 사용하지 않고도 예측 토큰과 입력 토큰의 길이를 효율적으로 관리할 수 있음을 의미합니다.



### Fine-tuning the SwissBERT Encoder Model for Embedding Sentences and Documents (https://arxiv.org/abs/2405.07513)
Comments:
          SwissText 2024

- **What's New**: 이 논문에서는 스위스의 네 가지 국어(독일어, 프랑스어, 이탈리아어, 그리고 로만슈어)에 대한 언어 어댑터를 포함한 SwissBERT 인코더 모델을 소개하고 있는데, 이 모델은 문장이나 짧은 문서의 임베딩을 위해 특별히 튜닝되었습니다. 이 새로운 모델, SentenceSwissBERT는 스위스 특화 설정에서 문서 검색 및 텍스트 분류와 같은 다양한 NLP 태스크(Task)에서 상당한 성능 향상을 보여줍니다.

- **Technical Details**: SwissBERT는 2천1백만 개 이상의 스위스 뉴스 기사를 통해 사전 학습되었으며, 이를 바탕으로 SentenceSwissBERT를 학습하기 위해 대조 학습(Contrastive Learning) 방식을 도입하였습니다. 또한 SimCSE 기법을 사용하여 모델을 최적화하고, 언어 어댑터를 고정시킨 채 다른 파라미터들을 업데이트하는 방식으로 미세 조절(Fine-tuning)되었습니다. 이 과정에서 MEAN pooling 전략이 사용되어 문장 임베딩의 품질을 향상시켰습니다.

- **Performance Highlights**: 평가 결과, SentenceSwissBERT는 기존 SwissBERT 모델과 다른 비교 모델들을 초월하는 정확성을 보여주었습니다. 특히 로만슈어에서는 최고의 베이스라인 모델에 비해 최대 29퍼센트 포인트, 원래의 SwissBERT 모델에 비해서는 최대 55퍼센트 포인트의 정확도 향상을 이루었습니다. 이러한 결과는 다국어(document retrieval and text classification) 환경에서도 일관되게 나타났습니다.



### MacBehaviour: An R package for behavioural experimentation on large language models (https://arxiv.org/abs/2405.07495)
Comments:
          11 pages

- **What's New**: 최근 대형 언어 모델(LLMs)과 LLM 기반 챗봇의 행동을 심리 실험의 참여자로 취급하는 연구에 대한 관심이 증가하고 있습니다. 이에 응답하여 'MacBehaviour'라는 R 패키지가 개발되었습니다. 이 패키지는 OpenAI의 GPT 가족, Claude 가족, Gemini, Llama 가족 및 오픈 소스 모델 등 60개 이상의 언어 모델과 상호 작용할 수 있으며, LLM 행동 실험의 실험 과정을 효율적으로 관리할 수 있게 설계되었습니다.

- **Technical Details**: 'MacBehaviour'는 실험 디자인(experiment design), 자극 제시(stimuli presentation), 모델 행동 조작(model behaviour manipulation), 응답 및 토큰 확률 기록(logging response and token probability) 등 LLM 실험을 위한 포괄적인 기능을 제공합니다. 이는 사용자 친화적인 인터페이스(user-friendly interface)와 함께 실험 과정을 간소화하고 표준화하는 데 도움을 줍니다.

- **Performance Highlights**: 세 가지 검증 실험을 통해 GPT-3.5, Llama-2 7B, Vicuna-1.5 13B 모델이 소리-성별 연관성(sound-gender association)을 인간과 유사하게 추론한다는 것이 확인되었습니다. 이는 새로운 개인 이름의 음운학(phonology)을 기반으로 성별을 추론하는 인간과 같은 경향을 일관되게 보여줍니다. 이 결과는 이전에 Cai 등(2023)에 의해 보여진 바와 같습니다.



### Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning (https://arxiv.org/abs/2405.07490)
- **What's New**: 이 연구는 커리큘럼 학습에서 영감을 받은 데이터 중심 훈련 전략을 제안합니다. 이 전략은 간단한 작업에서 시작하여 더 복잡한 작업으로 진행하면서 훈련 데이터를 구조화하여 대규모 언어 모델(Large Language Models, LLMs)의 성능을 향상시키는 지속 가능한 방법을 제공합니다. 특히, 주의 점수(attention scores)와 손실 값(loss values)을 기준으로 데이터를 정렬하는 새로운 접근 방식이 눈에 띄게 성능을 개선합니다.

- **Technical Details**: 커리큘럼 학습은 인간의 학습 방식을 모방하여 점진적으로 학습 난이도를 증가시키는 학습 전략입니다. 이 연구에서는 프롬프트 길이(prompt length), 주의 점수(attention scores), 손실 값(loss values)을 기준으로 훈련 데이터의 난이도를 측정합니다. 또한, 모델의 학습 도구로서 Quantize Low Rank Approximation (QLoRA)을 사용하고 Parameter-Efficient Fine-Tuning (PEFT)라이브러리를 통해 모델을 튜닝합니다.

- **Performance Highlights**: Mistral-7B와 Gemma-7B 모델을 사용한 실험에서 커리큘럼 학습 기반 훈련이 전통적인 데이터 무작위 셔플링 방식보다 성능이 약간 향상됨을 보여줍니다. 특히, 제안된 주의 점수 기반 데이터 정렬 방식이 모델 성능 개선에 효과적인 것으로 관찰되었습니다.



### Evaluating large language models in medical applications: a survey (https://arxiv.org/abs/2405.07468)
Comments:
          4 figures, 1 table

- **What's New**: 본 논문에서는 의료 및 보건 분야에서 강력한 도구로 자리 잡고 있는 대규모 언어 모델(Large Language Models, LLMs)의 사용 가능성을 탐구하고 있습니다. 특히, 임상 의사 결정 지원(Clinical Decision Support)에서부터 환자 교육에 이르기까지 다양한 작업에 LLMs가 제공할 수 있는 잠재력에 초점을 맞추고 있습니다.

- **Technical Details**: 이 논문은 의료 분야에서 LLMs의 성능 평가가 직면하는 독특한 도전과제를 다루고 있으며, 의료 정보의 복잡성과 중요성으로 인해 이러한 평가가 어려운 부분을 강조합니다. 또한, 평가 데이터 출처(Evaluation Data Sources), 작업 시나리오(Task Scenarios), 평가 방법(Evaluation Methods)에 대한 포괄적인 개요를 제공하면서 기존 연구에서 얻은 인사이트를 종합하고 있습니다.

- **Performance Highlights**: 이 논문은 의료 분야 LLM 평가의 핵심 도전과제 및 기회를 식별하며, LLMs를 임상 실습에 책임 있게 통합하기 위한 지속적인 연구와 혁신의 필요성을 강조합니다.



### MCS-SQL: Leveraging Multiple Prompts and Multiple-Choice Selection For Text-to-SQL Generation (https://arxiv.org/abs/2405.07467)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 발전을 통해 in-context learning (ICL) 기반 방법론이 텍스트-투-SQL 작업에 성능 향상을 가져왔습니다. 특히 복잡한 데이터베이스 스키마와 쿼리가 포함된 BIRD 벤치마크에서 인간 전문가의 성능을 아직 넘지 못하는 점을 고려해, 복수의 프롬프트를 사용하여 보다 광범위한 답변 검색 공간을 탐색하고 효과적으로 결과를 집계하는 새로운 접근 방식을 도입합니다.

- **Technical Details**: 이 연구는 데이터베이스 스키마(schema)를 '스키마 링킹(schema linking)'을 통해 견고하게 정제하고, 다양한 프롬프트를 사용하여 여러 후보 SQL 쿼리를 생성하는 방식으로 진행됩니다. 최종적으로, 후보 쿼리들은 신뢰도 점수에 따라 필터링되고, 최적의 쿼리는 다중 선택(multiple-choice selection)을 통해 결정됩니다. 이 과정은 대규모 언어 모델의 프롬프트에 대한 민감성을 이용하여 보다 효과적인 결과를 도출합니다.

- **Performance Highlights**: 제안된 방법은 BIRD와 Spider 벤치마크에서 각각 65.5% 및 89.6%의 실행 정확도(execution accuracy)를 달성하여, 이전 ICL 기반 방법들을 상당히 능가했습니다. 이는 BIRD 벤치마크에 대해 새로운 최고 수준의 성능(SOTA, State of the Art)을 설정하였으며, 실행 정확도와 유효성 효율성(valid efficiency score) 모두에서 인상적인 결과를 보여주었습니다.



### Evaluation of Retrieval-Augmented Generation: A Survey (https://arxiv.org/abs/2405.07437)
- **What's New**: 이 논문은 검색 기반 생성(Retrieval-Augmented Generation, RAG)에 초점을 맞추고 있습니다. RAG는 기존의 생성 모델에 외부 정보 검색을 통합하여 모델의 성능을 향상시키는 것을 목표로 합니다. 특히, 이 논문은 RAG 시스템의 벤치마크를 체계적으로 분석하고 평가하는 새로운 프레임워크 RAGR을 제안하여 기존 평가 방법에 대한 한계를 극복하고자 합니다.

- **Technical Details**: RAG 시스템은 검색(retrieval)과 생성(generation)의 두 가지 주요 구성 요소로 구성되어 있습니다. 검색 단계에서는 외부 지식 소스에서 관련 정보를 추출하는 역할을 하며, 생성 단계에서는 이 정보를 바탕으로 일관되고 맥락적으로 적합한 응답을 생성합니다. 이 논문에서 제안한 RAGR 프레임워크는 검색 적합도(relevance), 정확성(accuracy), 충실도(faithfulness)와 같은 다양한 정량적 지표를 통해 RAG 시스템의 벤치마크를 분석합니다.

- **Performance Highlights**: 제안된 RAGR 프레임워크를 통해 RAG 시스템의 여러 벤치마크를 분석한 결과, 기존 벤치마크의 강점과 한계를 명확히 지적하고, 추가적인 검증 방향을 제안합니다. 특히, 이 프레임워크는 검색과 생성 단계의 상호 작용에 초점을 맞추어 전체 시스템 성능을 보다 정확하게 평가할 수 있도록 도와줍니다.



### Multilingual Power and Ideology Identification in the Parliament: a Reference Dataset and Simple Baselines (https://arxiv.org/abs/2405.07363)
- **What's New**: 새로운 정치적 쟁점과 권력 위치 식별 데이터셋이 ParlaMint를 기반으로 도입되었습니다. 이 데이터셋은 29개 국가 및 지역 의회에서의 발언을 기록한 텍스트의 데이터셋으로, 2015년에서 2022년까지의 자료를 포함합니다. 데이터셋은 정치적 오리엔테이션(orientation)과 권력 위치의 체계적인 분석 및 예측을 위하여 준비되었습니다.

- **Technical Details**: 이 연구에서는 두 가지 주요 분류 작업에 초점을 맞추고 있습니다: 1) 정치적 오리엔테이션을 좌우축에서 예측하고, 2) 발언자가 속한 정당이 정부 연합의 일원인지 야당인지 구분하는 권력 위치 식별입니다. Wikipedia와 Chapel Hill Expert Survey Europe (CHES)의 데이터를 사용하여 정당의 정치적 위치를 분류해 나갔으며, 이 데이터는 향후 다양한 언어 및 국가를 대상으로 한 분석에 활용될 예정입니다.

- **Performance Highlights**: 이 데이터셋을 사용하여 기본 분류기(baseline classifier)를 통한 실험 결과, 정치적 오리엔테이션과 권력 위치를 구별하는 데 있어 어느 정도의 기초 성능(baseline performance)을 제시하였습니다. 이는 향후 이 데이터셋을 활용한 더 정교한 모델 개발의 기반을 제공합니다.



### MedConceptsQA -- Open Source Medical Concepts QA Benchmark (https://arxiv.org/abs/2405.07348)
- **What's New**: MedConceptsQA라는 새로운 의료 개념 질문 응답을 위한 오픈 소스 벤치마크를 소개합니다. 이 벤치마크는 진단(diagnoses), 절차(procedures), 약물(drugs) 등 다양한 의료 개념에 대한 질문을 포함하고 있으며, 이러한 질문들은 쉬운(easy), 중간(medium), 어려운(hard) 세 가지 난이도로 분류됩니다.

- **Technical Details**: 이 벤치마크는 대규모 언어 모델(Large Language Models)을 사용하여 평가되었습니다. 특히, 의료 데이터에 사전 훈련된 임상 대규모 언어 모델(pre-trained clinical Large Language Models)은 이 벤치마크에서 무작위 추측(random guessing) 수준에 가까운 정확도를 보였습니다. 그러나 GPT-4는 임상 대규모 언어 모델에 비해 평균적으로 27%-37%의 절대적인 성능 향상을 보였습니다(제로샷 학습(zero-shot learning)에서 27%, 퓨샷 학습(few-shot learning)에서 37%).

- **Performance Highlights**: GPT-4는 MedConceptsQA 벤치마크에서 특히 효과적이며, 다른 임상 모델들과 비교하여 월등한 성능 향상을 보여줍니다. 이는 GPT-4가 의료 개념의 이해와 추론을 위해 효과적인 도구임을 시사합니다.



### L(u)PIN: LLM-based Political Ideology Nowcasting (https://arxiv.org/abs/2405.07320)
- **What's New**: 이 논문은 각 국회 대표의 이념적 위치에 대한 분석을 위해 LLM(Large Language Models)의 잠재적 지식을 활용하는 새로운 방법을 제안합니다. 특히, 대표들의 연설에서 의견 기반 문장을 추출해내기 위해 BERT 분류기를 튜닝하고, 이를 통해 정치인의 특정 주제에 대한 입장을 수치화하는 기법을 소개합니다.

- **Technical Details**: 연구팀은 연설 데이터에서 의견 기반 문장을 분류하기 위해 Tohoku University에서 개발한 BERT 분류기를 사용하였으며, 각 정치인의 평균 BERT 임베딩(Embeddings)을 사용하여 이념적 표현을 생성합니다. 이러한 임베딩은 수동으로 선택되거나 GPT-4로 생성된 참조 시드(reference seeds)를 기준으로 붕괴(collapsing)됩니다. 또한 UMAP을 사용하여 데이터 차원을 축소하고 2차원 평면에서 정치인들의 상대적 이념적 위치를 시각적으로 분석했습니다.

- **Performance Highlights**: 이 방법론을 활용하여 얻은 정치인의 이념적 위치 추정치는 Mielka(NPO)의 전문가 평가와 일치함으로써 정확성을 검증했습니다. 의견 추출과 임베딩 프로세스는 수동 작업을 크게 줄이면서도 유효한 추정치를 제공할 수 있음을 보여줍니다.



### Branching Narratives: Character Decision Points Detection (https://arxiv.org/abs/2405.07282)
Comments:
          GamesAndNLP @ LREC COLING 2024

- **What's New**: 이 논문에서는 인물이 결정을 내리는 중요한 순간을 식별하는 새로운 자연어 처리(NLP) 작업인 'Character Decision Points Detection (CHADPOD)'을 제안합니다. 이 연구에서는 결정이 이야기의 방향에 중대한 영향을 미칠 수 있는 서사 내의 포인트를 식별하는 데 초점을 맞추고 있습니다. 또한 CYOA(Choose Your Own Adventure) 게임 그래프에 기반한 새로운 데이터셋을 소개하며, 이를 벤치마크로 사용하여 다양한 모델의 성능을 비교 분석합니다.

- **Technical Details**: 이 작업은 캐릭터의 결정 포인트(Characters Decision Points, CDPs)를 구체적으로 분류하고 관련 연구와 비교하면서, LLMs(Large Language Models) 및 MLMs(Masked Language Models)을 포함하여 몇 가지 기존 모델들의 성능을 평가합니다. 데이터는 CYOA 게임 자료를 분석하여 양성 클래스(결정이 스토리 방향에 영향을 주는 포인트)와 부정 클래스(임의의 분할 텍스트 또는 결정이 스토리 전개에 별다른 영향을 주지 않는 포인트)로 분류된 1,462개의 이진 분류 작업으로 구성됩니다.

- **Performance Highlights**: DeBERTa 모델을 강력한 기준 모델로 사용하여 이 태스크에서 최대 89%의 정확도를 달성함으로써, 특히 서사 분석의 복잡성과 캐릭터 기반 스토리 다이내믹스를 이해하는데 있어서의 도전을 강조했습니다. 이는 문학적 구조와 결정적 순간을 효과적으로 식별할 수 있는 현대 모델들의 유효성을 입증합니다.



### Humor Mechanics: Advancing Humor Generation with Multistep Reasoning (https://arxiv.org/abs/2405.07280)
Comments:
          ICCC 2024

- **What's New**: 이 논문에서는 다단계 추론을 통한 1줄짜리 유머(one-liner jokes) 생성에 대해 탐구하였습니다. 인간 참가자들을 대상으로 한 포괄적 실험을 통해 접근법을 평가하였고, 이를 인간이 만든 유머, GPT-4의 제로-샷 휴머 생성, 그리고 기타 베이스라인과 비교하였습니다. 이 연구에서 다단계 추론 접근법은 생성된 유머의 질을 일관되게 향상시키는 것을 발견하였습니다.

- **Technical Details**: 이 문제를 해결하기 위해 리인포스먼트 러닝(Reinforcement Learning) 및 크리에이티브 문제 해결 방법론에서 영감을 받은 기술을 사용하여 유머 생성 방침(humor generation policy)를 데이터셋으로부터 추론하고 새로운 유머를 생성하였습니다. 기능 포함은 데이터 기반 접근법, 크리에이티브 연관성 생성을 위한 시드 토픽 사용, 그리고 인간 평가를 통한 결과 검증입니다.

- **Performance Highlights**: 인간 평가를 통해 생성된 유머의 품질을 평가한 결과, 다중 스텝 방식이 기존의 유머보다 높은 품질을 제공함을 입증하였습니다. 유머의 독창성과 다양성에 중점을 둔 평가에서도 긍정적인 결과를 얻을 수 있었습니다.



### Human-interpretable clustering of short-text using large language models (https://arxiv.org/abs/2405.07278)
Comments:
          Main text: 18 pages, 8 figures. Supplementary: 21 pages, 15 figures, 3 tables

- **What's New**: 이 연구는 대규모 언어 모델들(Large Language Models, LLMs)이 단문 텍스트의 클러스터링과 그 결과의 해석에 어떻게 사용될 수 있는지 보여줍니다. 특히 'Twitter bios' 데이터를 사용하여 어떻게 사람들이 자신을 어떻게 설명하는지에 대한 클러스터를 형성하고 해석하는지 탐구했습니다. 본 연구는 딥러닝 방법과 비교하여 LDA와 doc2vec과 같은 기존 방법들과 LLM을 비교 분석하였습니다.

- **Technical Details**: LLM, LDA, 그리고 doc2vec과 같은 다양한 텍스트 클러스터링 방법들을 사용하여 'Twitter bios' 데이터 셋을 클러스터로 분류했습니다. 이들은 각각 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA), 문서를 벡터 공간에 매핑하는 doc2vec, 그리고 양방향 인코더 표현(Bidirectional Encoder Representations from Transformers, BERT)을 사용하는 LLM을 포함합니다. 클러스터의 성공은 인간 리뷰어와 ChatGPT에 의해 평가되었으며, 클러스터의 명확성과 해석 가능성을 통해 정의되었습니다.

- **Performance Highlights**: LLM을 사용한 클러스터링 접근 방식은 LDA나 doc2vec과 비교하여 인간의 해석과 더 높은 일치성을 보였습니다. 이는 LLM이 단문 클러스터링의 해석 가능성과 성공 측정에 있어서 잠재적으로 더 우수할 수 있음을 시사합니다. 클러스터링 결과는 인간 리뷰어와 ChatGPT 모두에 의해 검증되어, 클러스터의 해석성(Interpretability)과 독창성(Distinctiveness)을 평가하는 신뢰할 수 있는 방법을 제공합니다.



### Span-Aggregatable, Contextualized Word Embeddings for Effective Phrase Mining (https://arxiv.org/abs/2405.07263)
- **What's New**: 이 논문에서는 문장의 밀집 벡터 표현(dense vector representations)의 발전에 따라 문구 검색(pharse retrieval)에서의 어려움을 진단하고, 문구의 소음이 많은 환경에서도 문맥을 잘 유지하면서 효과적인 문구 검색을 가능하게 하는 새로운 접근법을 제시합니다. 이를 위해 연속적인 단어 범위(consecutive word spans)을 각각 다른 밀집 벡터로 표현하여, 문장의 일부인 문구들을 효과적으로 검색할 수 있는 방법(SLICE: Span-aggregated Late Interaction Contextualized Embeddings)을 개발했습니다.

- **Technical Details**: 이 연구는 다음과 같은 기술적인 접근을 제안합니다: 1) 각각의 연속 단어 범위를 별개의 밀집 벡터로 표현하는 것과, 2) 문맥화된 단어/토큰 임베딩(contextualized word/token embeddings)을 임의의 단어 범위에 대해 집합할 수 있도록 하여, 해당 범위의 의미론적 의미(semantic meaning)를 유지하는 새로운 대조 손실 함수(contrastive loss)의 수정을 도입합니다. 또한, STS-B 데이터셋을 바탕으로 생성된 추가 텍스트를 포함하여 문구 검색 효과를 시연하는 실험적 데이터셋을 소개하며, 이를 통해 SLICE 모델의 우수성을 입증합니다.

- **Performance Highlights**: 제안된 SLICE 방법은 기존의 유사 계산 방법들과 비교하여 상당한 성능 향상을 보였습니다. 특히 문구의 의미론적 거리(semantic distance)를 측정하는 실험에서, 이 방법은 더 정밀한 문구 매칭 결과를 제공하며, 심각한 연산 증가 없이 더 나은 결과를 달성하였습니다. 다양한 모델과 설정에서 각각의 n-gram을 인코딩하고 선택하는 것이 '전체 문맥(Full Context)' 설정보다 우수한 결과를 나타내었습니다.



### Limited Ability of LLMs to Simulate Human Psychological Behaviours: a Psychometric Analysis (https://arxiv.org/abs/2405.07248)
- **복잡한 결과와 도전적인 질문들**: 이 연구에서 개발된 방법론은 대화형 언어 모델을 사용하여 인간 참가자를 모방할 수 있는지 여부에 대한 사회 과학자들의 연구를 강화했습니다. 연구진은 OpenAI의 주력 모델인 GPT-3.5와 GPT-4를 사용하여 다양한 성격 구성 개념에 대한 표준화된 측정 수단에 응답하도록 요구했습니다. 이들 모델은 일반적인 인물 설명(generic persona descriptions)과 구체적인 인구 통계 프로필(specific demographic profiles)의 두 가지 유형의 페르소나 설명을 사용하여 반응하도록 설정되었습니다.



### InsightNet: Structured Insight Mining from Customer Feedback (https://arxiv.org/abs/2405.07195)
Comments:
          EMNLP 2023

- **What's New**: InsightNet은 고객 리뷰에서 구조화된 인사이트를 자동 추출하는 새로운 방법을 제안합니다. 이는 기존 솔루션의 한계를 극복하고, 자동화된 멀티레벨 분류(Multi-level Taxonomy) 구축, 레이블된 데이터 생성을 위한 Semantic Similarity Heuristic 접근법을 사용하며, Large Language Model(Large Language Model, LLM)의 Fine-tuning을 통해 다중 과제 실행 구조를 적용합니다.

- **Technical Details**: InsightNet은 세 가지 주요 모듈로 구성됩니다: 1) AutoTaxonomy는 최소한의 감독하에 계층적 측면 분류를 생성합니다. 2) SegmentNet은 의미론적 유사성 기반 휴리스틱을 사용하여 레이블이 지정된 데이터를 생성합니다. 3) InsightNet 자체는 T5-base 모델을 사용하여 리뷰에서 중요한 주제를 추출하고 감정을 확인하며 새로운 주제를 발견하는 Generative Model을 Fine-tuning합니다.

- **Performance Highlights**: InsightNet은 실제 고객 리뷰 데이터에 대한 평가에서 구조, 계층성 및 완성도 측면에서 기존 솔루션을 능가합니다. 이는 다중 레이블 주제 분류에서 0.85의 F1 점수를 달성하며, 이전 최고 결과 대비 11% 향상된 결과를 보여줍니다. 또한, InsightNet은 보이지 않는 측면에 대해 잘 일반화되고 태그가 지정된 데이터를 기반으로 하지 않아도 효과적으로 작동합니다.



### Designing and Evaluating Dialogue LLMs for Co-Creative Improvised Theatr (https://arxiv.org/abs/2405.07111)
Comments:
          13 pages, 7 figures, accepted for publication at the International Conference on Computational Creativity 2024

- **What's New**: 연구자들은 에든버러 페스티벌 프린지에서 단기간 동안 여러 참가자(Large Language Models, LLMs)가 등장하는 실시간 공연에서 대화형 에이전트를 실행함으로써 멀티 파티 대화능력의 새로운 사례를 제시했습니다. 이는 전문 배우와 함께하는 즉석 상황에서 AI의 창조적 공동 창작 능력을 탐구하는 최초의 시도 중 하나입니다.

- **Technical Details**: 세 가지 다른 LLMs (GPT 3.5, PaLM 2, Llama 2)를 사용하여 공연예술 시나리오에서 '지능적'으로 반응할 수 있는 대화형 에이전트가 배치되었습니다. 이는 음성 인식 시스템과 현장 대화의 맥락 데이터를 제공하는 인간 중심의 큐레이션 시스템을 통합하여 대화를 실시간으로 관리하고, 대화형 에이전트가 생성하는 반응을 선택하는 방식으로 구현되었습니다.

- **Performance Highlights**: 이 케이스 스터디는 참가자들이 AI와 직접 상호작용하면서 AI의 능력과 공연 예술에서의 유틸리티에 대한 인식이 어떻게 변화하는지를 보여줍니다. 발표 결과, 관객과 공연자 모두 AI의 존재가 창의적 변화와 긍정적인 인상을 강화할 수 있다고 응답했습니다. 또한 LLM이 둘 이상의 참가자와의 씬을 처리할 수 있는 능력을 시연함으로써 멀티 파티 대화의 새로운 가능성을 열었습니다.



### Advanced Natural-based interaction for the ITAlian language: LLaMAntino-3-ANITA (https://arxiv.org/abs/2405.07101)
- **What's New**: 이탈리아어 자연어 처리(Natural Language Processing, NLP)를 개선하기 위해, 우리는 새로운 Meta LLaMA-3 모델을 기반으로 하는 LLaMAntino-3-ANITA-8B-Inst-DPO-ITA라는 최신 대형 언어 모델(Large Language Model, LLM)을 소개합니다. 이 모델은 영어와 이탈리아어 데이터셋에서 Supervised Fine-tuning (SFT) 기술을 사용하여 기존 모델을 미세 조정하였고, Dynamic Preference Optimization (DPO) 과정을 통해 모델의 선호도를 조정하고 위험하거나 부적절한 답변을 피하며 편견을 제한합니다.

- **Technical Details**: LLaMAntino-3-ANITA-8B-Inst-DPO-ITA 모델은 QLoRA를 사용하여 모델의 일부분만을 미세조정함으로써 이탈리아 언어 구조에 특화된 모델로 적응시켜 성능과 계산 효율성에서 중요한 개선을 이루었습니다. DPO는 모델 출력을 개선하여 생성된 내용이 질 높은 답변과 일치하도록 하는 데 사용됩니다. SFT, QLoRA의 파라미터 효율성, 그리고 DPO의 사용자 중심 최적화 사이의 시너지는 텍스트 완성, 제로샷 분류(Zero-shot classification), 맥락 이해(Contextual understanding)를 포함한 다양한 작업에서 탁월한 성능을 발휘하는 강력한 LLM을 만들어냈습니다.

- **Performance Highlights**: 이 모델은 이탈리아어와 영어에 대한 표준 벤치마크를 통해 광범위하게 평가되었으며, 뛰어난 결과를 보여주었습니다. 해당 모델은 HuggingFace 허브를 통해 자유롭게 이용할 수 있으며, GitHub 저장소에서 사용 예제를 찾아볼 수 있습니다.



### Do Pretrained Contextual Language Models Distinguish between Hebrew Homograph Analyses? (https://arxiv.org/abs/2405.07099)
- **What's New**: 이 연구에서는 히브리어 학습 안정성을 향상시키기 위해 프리트레인드(pre-trained) 언어 모델에서의 컨텍스추얼라이즈드(contextualized) 임베딩(embeddings)의 효과를 연구하였습니다. 이전의 비컨텍스추얼라이즈드(non-contextualized) 임베딩 기반 모델과 비교하여, 새로운 데이터셋에서 다양한 히브리어 동음이의어(homographs)의 구문적 및 의미적 구분에 컨텍스추얼라이즈드 임베딩이 얼마나 효과적인지 평가하였습니다.

- **Technical Details**: 연구자들은 다양한 히브리어 동음이의어에 대해 컨텍스추얼라이즈드 언어 모델들이 어떻게 반응하는지 분석하기 위해, 멀티 링구얼 BERT(multilingual BERT, mBERT), HeBERT, 그리고 AlephBERT를 포함하는 여러 모델을 평가하였습니다. 특히, 이들 모델은 세분화된, 분류가 어려운 동음이의어들에 대한 데이터셋에서 테스트되었습니다. 이 데이터셋은 신문, 위키피디아, 문학작품, 소셜 미디어 등에서 수집된 자연 발생 문장들로 구성되었습니다.

- **Performance Highlights**: 컨텍스추얼라이즈드 임베딩 모델들은 구문형(syntactic) 및 모퍼신태틱(morphosyntactic) 특징의 구분에 있어 높은 효과를 보였으며, 순수 의미 구분(sense disambiguation)에 있어서는 약간의 도전이 있었음에도 불구하고 전반적으로 우수한 성능을 나타내었습니다. 이 연구는 히브리어 동음이의어의 다양한 구분에 효과적인 새로운 벤치마크를 설정하였습니다.



### Integrating Emotional and Linguistic Models for Ethical Compliance in Large Language Models (https://arxiv.org/abs/2405.07076)
Comments:
          26 pages, 7 tables, 6 figures

- **What's New**: 이 연구는 감정 및 윤리와 관련된 언어적 행동을 더 잘 관리할 수 있도록 대형 언어 모델(Large Language Models, LLMs)을 위한 고급 방법론을 개발합니다. DIKE라는 적대적(adversarial) 프레임워크를 도입하여 LLM의 글로벌 인간 가치(global human values)를 내면화하고 반영하여 다양한 문화적 맥락에서의 적응을 통해 사용자 간의 투명성과 신뢰를 증진시키는 데 도움을 줍니다.

- **Technical Details**: 이 방법론은 감정 모델링, 언어 행동 분류 및 윤리적 제한(Ethical guardrails)의 구현을 포함합니다. 혁신적인 접근 방식에는 자기 감독 학습(self-supervised learning) 기술을 사용하여 감정과 행동을 매핑하고, 적대적 검토를 통해 이러한 제한을 정제하며, 윤리적 정렬을 보장하기 위해 출력을 체계적으로 조정하는 것이 포함됩니다.

- **Performance Highlights**: DIKE 프레임워크는 AI 시스템이 윤리적 완전성 및 문화적 민감성을 유지하면서 운영할 수 있는 견고한 기반을 설정합니다. 이로 인해 보다 책임감 있고 문맥을 고려한 AI 상호작용의 길을 열게 됩니다.



### Length-Aware Multi-Kernel Transformer for Long Document Classification (https://arxiv.org/abs/2405.07052)
Comments:
          Accepted to SEM 2024

- **What's New**: 이 연구에서는 긴 문서 분류를 위한 길이에 민감한 멀티-커널 트랜스포머(LAMKIT)를 제안하였습니다. LAMKIT는 다양한 트랜스포머 기반 커널을 사용하여 문맥 경계를 연결하고 문서 길이가 다양할 때 모델의 견고성을 증진시키는 새로운 접근법을 소개합니다.

- **Technical Details**: LAMKIT는 멀티-커널 인코딩(MK)과 길이 인식 벡터화(LaV) 모듈을 통해 다양한 길이의 문서에 대해 효과적으로 대응할 수 있도록 설계되었습니다. MK는 다양한 커널 크기를 가진 여러 신경 인코더를 포함하여 문맥 단편화를 완화시키고, LaV는 문서 수준에서 세그먼트와 길이 벡터에 위치 인코딩을 적용하여 다양한 문서 길이에 따른 모델의 견고성을 증진시킵니다.

- **Performance Highlights**: 법률 및 건강 분야의 다섯 가지 표준 벤치마크에서 LAMKIT은 상태 기술(state-of-the-art, SOTA) 모델들을 최대 10.9%까지 상회하는 성능을 보였습니다. 이는 LAMKIT이 장문의 문서를 다루는 데 탁월한 능력을 가지고 있음을 의미하며, 다양한 문서 길이에 걸쳐 견고한 성능을 유지할 수 있습니다.



### A Turkish Educational Crossword Puzz (https://arxiv.org/abs/2405.07035)
Comments:
          This paper has been accepted for presentation at AIED2024 LBR

- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 능력을 활용하여 설계된 최초의 터키어 크로스워드 퍼즐 생성기를 소개합니다. 특히 교육 목적으로 사용하기 위해 개발된 이 시스템은 터키어 학습 환경에서 인터랙티브하고 지능적인 학습 도구로서 새로운 기준을 제시합니다.

- **Technical Details**: 이 시스템은 두 가지 주요 데이터셋을 바탕으로 구축되었습니다. 하나는 18만 개의 고유한 답변-힌트 쌍을 포함하여 주어진 답변에서 관련 힌트를 생성하고, 다른 하나는 특정 텍스트와 키워드에 대한 힌트를 생성하기 위해 3만 5천 개의 샘플을 포함하는 데이터셋입니다. 사용된 LLMs로는 GPT3.5-Turbo, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf 등이 있으며, 이 모델들은 특정 데이터셋에 대해 파인 튜닝(fine-tuning)되어 크로스워드 퍼즐의 힌트 생성에 최적화되었습니다.

- **Performance Highlights**: 이 크로스워드 생성 시스템은 교육적 환경에서 키워드와 텍스트를 기반으로 효율적인 퍼즐 생성을 가능하게 하며, 사용자 지정 텍스트나 키워드에 기반한 고품질의 힌트와 해결책을 제공함으로써 터키어 학습에 큰 도움을 줍니다. 또한, 이 도구는 학습자의 어휘력, 철자 능력 및 인지 능력 향상에 기여하며, 문제 해결 기술을 강화하는 인터랙티브한 교육 도구로서의 역할을 합니다.



### Word-specific tonal realizations in Mandarin (https://arxiv.org/abs/2405.07006)
- **What's New**: 이 연구는 중국어(특히 대만 중국어)의 음조 현상이 단일 문자의 기본 음조와 화자의 발화 제약에 의해서만 결정되는 것이 아니라, 단어의 의미에 따라서도 부분적으로 결정될 수 있다는 새로운 발견을 제시합니다. 특히 음조 패턴의 'rise-fall' 형태에 초점을 맞추어, 단어 유형이 음도 실현(tonal realization)의 강력한 예측 변수임을 보여 줍니다. 이 연구는 일반적인 렉시컬 톤(Lexical Tones)'s)'의 정형적 설명을 넘어서, 단어의 의미가 음조 패턴에 어떻게 영향을 미치는지를 밝힙니다.

- **Technical Details**: 연구자들은 대만의 자발적 대화(corpus)를 사용하여 일반화 가법 회귀 모델(Generalized Additive Regression Model)을 적용하고, 음도 패턴의 실제 형상(realization)을 예측하기 위해 의미 있는 맥락 정보(meaning in context)를 추가하는 실험을 수행했습니다. 또한, 컨텍스트-특화 워드 임베딩(Context-specific Word Embeddings)을 사용하여 토큰-특화 음조 형태(token-specific pitch contours)를 예측하는 컴퓨테이셔널 모델링을 진행했습니다.

- **Performance Highlights**: 이 연구의 모델은 보류 데이터(held-out data)에서 50%의 정확도로 단어 유형을 예측할 수 있었고, 토큰-특화의 컨텍스트-민감한 임베딩은 30%의 정확도로 음조 형태의 윤곽을 예측할 수 있었습니다. 이는 기존 예측 수준을 훨씬 뛰어넘는 결과로, 단어의 의미와 음조 간의 관계가 언어 사용자에게 실질적으로 유용할 수 있음을 시사합니다.



### Evaluating Task-based Effectiveness of MLLMs on Charts (https://arxiv.org/abs/2405.07001)
- **What's New**: 이 논문에서는 차트에서 저수준(low-level) 데이터 분석 작업에 대해 GPT-4V의 효과성을 탐구합니다. 'ChartInsights'라는 대규모 데이터셋을 개발하였으며, 이는 7가지 차트 유형에서 사용되는 10가지 저수준 데이터 분석 작업을 포함합니다. 특히, GPT-4V가 이러한 작업에서 가장 높은 정확도(56.13%)를 달성하며, 신규 텍스트 프롬프트 전략 'Chain-of-Charts'를 도입하여 모델 성능을 24.36% 향상시켰습니다.

- **Technical Details**: 새로운 데이터셋 'ChartInsights'는 89,388개의 (차트, 작업, 질문, 답변) 쿼텟을 포함하고, 18개의 고급 MLLMs (멀티모달 대규모언어모델)의 성능을 평가합니다. 이 평가에서는 텍스트 프롬프트, 시각적 변화, 'Chain-of-Charts' 전략 등 다양한 실험을 통해 GPT-4V의 한계와 가능성을 조사했습니다.

- **Performance Highlights**: GPT-4V는 표준 텍스트 프롬프트 접근 방식에서 평균 정확도 36.17%에 비해 56.13%의 가장 높은 정확도를 보였습니다. 'Chain-of-Charts' 전략은 GPT-4V의 정확도를 80.49%로 증가시켰고, 시각적 프롬프트 전략을 추가하면서 83.83%까지 성능이 개선되었습니다. 이러한 결과는 GPT-4V가 차트와 상호작용하는 방식을 혁신할 가능성을 시사하고 있습니다.



### Quite Good, but Not Enough: Nationality Bias in Large Language Models -- A Case Study of ChatGP (https://arxiv.org/abs/2405.06996)
Comments:
          Accepted by LREC-COLING 2024

- **What's New**: 이 연구는 언어 모델인 ChatGPT (GPT-3.5)가 생성하는 텍스트에서 국적 편견을 분석합니다. 이는 195개 국가, 4가지 온도 설정, 그리고 3가지 다른 유형의 프롬프트를 사용하여 중국어와 영어로 4,680개의 대화를 만들어 테스트합니다. 주요 발견은 ChatGPT가 주로 긍정적인 내용을 생성했지만, 부정적인 경향의 프롬프트에 의해 가끔 부정적인 내용도 생산한다는 점입니다. 연구 결과는 국적에 대한 차이도 드러냈으며, 특히 중국어와 영어 버전 간에 문화적 관점의 차이를 보여줍니다.

- **Technical Details**: 이 연구는 자동화된 측정 도구와 전문가 평가자 및 ChatGPT 자체의 자가 평가를 사용하여 국적 편견을 분석했습니다. 텍스트는 감성, 비속어, 혐오 발언, 존중도 등 여러 지표를 통해 평가되었습니다. ChatGPT는 일반적으로 중립적이라고 생각하는 반면, 인간 평가자와의 쌍대 비교(pair-wise comparison)에서 일관된 자기 인식을 보였습니다.

- **Performance Highlights**: 비교적 최근 버전인 GPT-3.5는 이전 모델인 GPT-2에 비해 향상된 결과를 보여주며, 전반적으로 긍정적인 감성을 지닌 텍스트를 생성합니다. 하지만 이 모델도 부정적 프롬프트에 의해 부정적인 텍스트를 생성할 수 있는 경향이 있다는 점은 주의해야 합니다. 또한, 이 연구는 국적 편향이 단순한 분류 문제가 아닌 연속체로 존재함을 강조하며, 이는 복잡한 AI 공정성 문제에도 적용될 수 있다는 점을 시사합니다.



### AraSpell: A Deep Learning Approach for Arabic Spelling Correction (https://arxiv.org/abs/2405.06981)
- **What's New**: 이 연구는 AraSpell이라는 새로운 자동 아랍어 맞춤법 교정 프레임워크를 소개합니다. 이 프레임워크는 RNN(Recurrent Neural Network) 및 Transformer와 같은 다양한 seq2seq(시퀀스-투-시퀀스) 모델 아키텍처를 사용하며, 690만개 이상의 아랍어 문장에는 인위적 오류 주입을 위한 데이터 생성 방법을 사용했습니다.

- **Technical Details**: AraSpell은 단어 오류율(WER, Word Error Rate) 4.8% 및 문자 오류율(CER, Character Error Rate) 1.11%를 달성하여 기존의 레이블이 지정된 데이터(29.72% WER 및 5.03% CER)와 비교하여 강력한 성과를 보였습니다. 또한, 다양한 수준의 오류를 주입하여 오류가 없는 문장에서 시작하는 대규모 데이터셋에서 효과적인 학습이 가능한 새로운 자기 라벨링 방식을 도입하였습니다.

- **Performance Highlights**: 제안하는 프레임워크는 테스트 세트에서 10만 문장의 결과를 확인할 수 있었으며, 기존 레이블 데이터(10.02% CER 및 50.94% WER)와 비교할 때 2.9% CER 및 10.65% WER를 달성하며 뛰어난 성과를 보였습니다. 이 결과들은 AraSpell 프레임워크가 아랍어 맞춤법 교정 분야에서 유의미한 진보를 나타내며, 실제 NLP(Natural Language Processing) 응용 프로그램에 큰 영향을 미칠 수 있음을 보여줍니다.



### Piccolo2: General Text Embedding with Multi-task Hybrid Loss Training (https://arxiv.org/abs/2405.06932)
Comments:
          tech report

- **What's New**: Piccolo2는 CMTEB (CMTE Benchmark) 벤치마크에서 6가지 작업을 평가한 결과에서 기존 모델들을 능가하는 새로운 state-of-the-art (최고 수준) 임베딩 모델을 소개합니다. 이 모델은 효율적인 멀티태스크 하이브리드 손실 (multi-task hybrid loss) 훈련 접근법을 통해 텍스트 데이터와 다양한 하류 작업들의 레이블을 효과적으로 활용합니다.

- **Technical Details**: Piccolo2는 임베딩 차원을 확대하고, Matryoshka Representation Learning (MRL 훈련)을 사용하여 더 유연한 벡터 차원을 지원합니다. 다양한 태스크에 대해 손실 함수를 최적화하며, 예를 들어, retrieval (검색) 및 reranking (재순위) 작업에는 InfoNCE 손실 함수를, 높은 세부 래벨을 지닌 텍스트 쌍을 위해 cosent 손실 함수를 사용합니다. 또한, 분류 및 클러스터링 작업에는 SFR 임베딩 방법을 사용하여 데이터를 훈련 과정에 통합합니다.

- **Performance Highlights**: Piccolo2는 다양한 데이터셋과 신스턴틱 데이터 (synthetic data), 하드 네가티브 마이닝 (hard negative mining) approach를 사용하여 품질이 높은 데이터셋을 지속적으로 확장합니다. 또한, 이 모델은 유연한 차원 길이의 임베딩 모델 훈련을 지원하며, 처리 속도를 향상시키고 저장 요구 사항을 크게 줄임으로써 성능 저하를 최소화합니다.



### EmoMix-3L: A Code-Mixed Dataset for Bangla-English-Hindi Emotion Detection (https://arxiv.org/abs/2405.06922)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2310.18387, arXiv:2310.18023

- **What's New**: 본 논문에서는 세 가지 언어로 코드 혼합된 데이터를 포함하는 새로운 다중 레이블 감정 감지 데이터 세트인 EmoMix-3L을 소개합니다. EmoMix-3L은 감정 감지에 중점을 둔 기존 연구에서 활용되지 않은 방글라어, 힌디어, 영어의 코드 혼합 데이터를 포함하고 있습니다.

- **Technical Details**: 연구팀은 마이크로소프트 인도 연구소가 개발한 다언어 모델인 MuRIL(Multilingual Representations for Indian Languages)과 다른 표준 NLP 모델을 비교했습니다. 코드 혼합 데이터에 대해 특별히 훈련된 MuRIL이 다른 모델보다 나은 성능을 보여주었습니다.

- **Performance Highlights**: MuRIL은 EmoMix-3L 데이터셋에서 높은 정확도를 달성하여, 다른 표준 NLP 모델과 비교했을 때 뛰어난 성능을 보였습니다. 본 데이터셋은 1,071개의 인스턴스를 포함하며, 각 인스턴스는 세 가지 언어를 구사하는 사용자에 의해 주석이 달렸습니다. 이는 코드 혼합과 감정 검출 연구의 중요한 자원을 제공합니다.



### CoRE: LLM as Interpreter for Natural Language Programming, Pseudo-Code Programming, and Flow Programming of AI Agents (https://arxiv.org/abs/2405.06907)
Comments:
          12 pages, 6 figures, comments and suggestions are welcome

- **What's New**: 이 논문에서는 자연 언어 인스트럭션(natural language instructions)을 해석하고 실행할 수 있는 새로운 시스템인 CoRE(Code Representation and Execution)가 제안되었습니다. CoRE는 자연 언어 프로그래밍, 의사 코드(pseudo-code) 프로그래밍, 플로우 프로그래밍(flow programming)을 하나의 대표 언어로 통합합니다. 이 시스템은 LLM(Large Language Models)을 인터프리터로 사용하여 주어진 인스트럭션을 단계별로 해석하고 실행합니다.

- **Technical Details**: CoRE 시스템은 자연 언어 지시 사항을 논리적으로 구조화하기 위해 프로그래밍 문법을 정의합니다. 시스템은 외부 도구를 호출하는 기능을 통해 LLM의 도메인별 지식이나 실시간 정보 접근의 한계를 보완합니다. 또한, 실행 중에 중간 결과를 임시 메모리에 저장하고, 필요에 따라 관련 정보를 검색하여 중복을 최소화합니다. 이러한 구조는 프로그램의 다음 단계를 정확하게 결정하는 데 필수적입니다. 

- **Performance Highlights**: 이 시스템은 공개 벤치마크 데이터셋을 사용하여 그 효과와 효율을 검증하였습니다. 특히, 자연 언어 기반 에이전트 작업 해결에 있어 CoRE 시스템을 사용하여 실용적인 능력을 보여 주었습니다. 이는 LLM을 활용하여 정보 검색 및 외부 도구 활용을 적절히 조합하여 인스트럭션을 실행하는 데 기여합니다.



### Finding structure in logographic writing with library learning (https://arxiv.org/abs/2405.06906)
Comments:
          Accepted at CogSci 2024 (Talk)

- **What's New**: 이 연구에서는 인간의 언어가 조합성(combinatoriality)을 특징으로 하며, 이는 상징 체계의 표현 효율성을 향한 인간의 귀납적 편향을 반영한다는 아이디어를 탐구합니다. 이를 위해 중국 문자 체계에 대한 구조 발견을 위한 계산 프레임워크를 개발하였고, 도서관 학습(library learning)과 프로그램 합성(program synthesis) 기술을 기반으로 하여 중국 문자의 구조적 특성과 시간에 따른 단순화 경향을 밝혀냈습니다.

- **Technical Details**: 이 연구는 문자 시스템의 구조를 발견하기 위해 도서관 학습 모델을 개발하였습니다. 중국 문자 체계에 적용하여 모델이 기존의 언어 구조, 예를 들어 부호(radicals) 및 문자의 계층적 분해를 성공적으로 재발견함으로써 중국 문자 체계의 조합 구조를 검증하였습니다. 도서관 학습 접근법은 이러한 구조를 반복적으로 식별하고 추상화 라이브러리에 저장하여 문자 인벤토리의 전반적인 표현의 간결성을 향상시킵니다. 중국 문자에 대한 특정 언어(domain-specific language, DSL)도 개발되었으며, 이는 개별 문자를 기본적인 획(strokes)의 연속으로 표현할 수 있게 합니다.

- **Performance Highlights**: 라이브러리 학습 모델은 중국 문자의 조합 구조를 성공적으로 재현하였고, 중국 문자 체계가 시간이 지나면서 표현 효율성을 위해 단순화되는 경향을 보여주는데 기여했습니다. 이 모델은 인간의 인지에서 조합 구조의 생성에 근본적인 계산 원리를 밝혀내는 데 도움을 주며, 효율적인 커뮤니케이션 시스템의 진화에 대한 더 넓은 통찰력을 제공합니다.



### TacoERE: Cluster-aware Compression for Event Relation Extraction (https://arxiv.org/abs/2405.06890)
Comments:
          Accepted to LREC-COLING 2024

- **What's New**: 이번 연구에서는 사건 관계 추출(Event Relation Extraction, ERE)을 개선하기 위해 클러스터 인식 압축 방법인 TacoERE를 제안합니다. 기존 방법의 한계인 장거리 의존성 및 정보의 중복 문제를 해결하기 위해 문서를 내부 및 외부 클러스터로 분할하고, 클러스터 요약을 통해 중요 내용 강조 및 간소화를 시도합니다.

- **Technical Details**: TacoERE는 문서를 내부 클러스터(Intra-clusters)와 외부 클러스터(Inter-clusters)로 분류하고, 이를 통해 사건 간의 관계를 모델링합니다. 내부 클러스터는 동일 클러스터 내의 사건 관계를 강화하며, 외부 클러스터는 임의 거리에서 관련 사건들을 모델링합니다. 또한, 클러스터 요약(Cluster Summarization)을 사용하여 클러스터의 중요한 텍스트 내용을 강조하고 간소화합니다.

- **Performance Highlights**: TacoERE는 RoBERTa, ChatGPT, GPT-4와 같은 소규모 및 대규모 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs; Large Language Models, LLMs)을 사용하여 MAVEN-ERE, EventStoryLine, HiEve와 같은 세 가지 ERE 데이터셋에서 효과적인 성능을 보였습니다. 특히 LLMs에서 TacoERE는 기존 메소드 대비 11.2% (ChatGPT) 및 9.1% (GPT-4) 향상된 결과를 보여줍니다.



### The Ghanaian NLP Landscape: A First Look (https://arxiv.org/abs/2405.06818)
- **What's New**: 이 연구는 가나어의 자연언어처리(Natural Language Processing, NLP) 연구에 관한 첫 종합적인 조사를 수행함으로써, 아프리카 언어들, 특히 가나 언어들이 인공지능(Artificial Intelligence, AI)에서 심각하게 대표되지 않는 문제를 다루고 있습니다. 이는 언어 다양성과 문화 유산을 위협하는 것으로 파악되며, 연구자들의 접근성을 개선하기 위해 도전 과제, 모범 사례, 그리고 미래의 방향을 개요하는 세부 로드맵을 제시합니다.

- **Technical Details**: 가나의 언어 상황을 면밀히 조사하여, NLP 연구에서 사용되는 모델(Model), 측정 방법(Metrics), 데이터셋(Datasets), 그리고 기술(Techniques)들을 검토 및 비교합니다. 특히 자동 번역(Machine Translation, MT)과 같은 고급 NLP 기술을 활용하여 가나어의 일부를 보존하고 디지털화하는 데 기여하였습니다. 이 연구는 교육용 및 소통용으로 사용되는 Akuapem Twi와 Asante Twi와 같은 주요 언어들을 중점적으로 다룹니다. 해당 연구는 광범위한 질적 및 양적 데이터 분석을 통해 수행되었으며, 이는 이 지역에 특화된 최초의 대규모 체계적 리뷰입니다.

- **Performance Highlights**: 이 연구는 가나어에 대한 연구와 디지털화가 매우 제한적이고 분열적인 상태에서, 가나의 언어 연구에 있어서 새로운 통찰력을 제공하고 보다 포괄적인 언어 연구를 위한 길을 제시합니다. 가나어의 번역 및 처리를 위한 NLP 및 MT 기술의 발전은 더욱 정확하고 빠른 결과를 촉진하며, 새로운 언어로의 확장 훈련 및 스케일링을 가능하게 합니다. 이는 특히 저자원 언어(Low-Resource Languages, LRL) 데이터 수집 및 주석 처리의 어려움을 극복하는데 중요한 역할을 합니다.



### Tackling Execution-Based Evaluation for NL2Bash (https://arxiv.org/abs/2405.06807)
- **What's New**: 이 논문은 자연어 설명(NL)에서 Bash 커맨드로 번역하는 작업인 NL2Bash에 대해 다루고 있으며, 이를 검증하기 위한 실행 기반 평가(Execution-based Evaluation, EE) 시스템을 설계 및 구현하였습니다. 이전의 연구들이 다른 프로그래밍 언어에 초점을 맞춰왔던 것과 달리, 이 연구는 Bash 스크립팅에 집중하면서, LLM(Large Language Models)의 Bash 코드 생성 능력을 평가합니다. 이러한 접근 방식은 NL2Bash 작업에서 발생할 수 있는 독특한 문제들을 검증하고 해결하는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 50개의 프롬프트를 사용하여 NL2Bash 작업을 위한 LLM의 능력을 평가했습니다. 각 테스트 케이스는 별도의 디렉토리와 Podman 컨테이너를 사용하여 실행됩니다. 실행 환경을 구성하는 과정, 예를 들면 팟맨 컨테이너를 설정하고 실행하는 작업 등이 포함됩니다. 실행된 결과물은 기대 출력과 비교되며, 이를 통해 모델의 예측된 코드가 실제 기대하는 동작을 수행하는지 평가합니다. 코드의 입력과 출력을 비교함으로써, 구문적으로 다르지만 의미적으로 동일한 Bash 스크립트, 또는 구문적으로는 올바르지만 의미적으로는 틀린 Bash 스크립트를 처리하는 방법 또한 분석했습니다.

- **Performance Highlights**: 실험에서 사용된 LLM들은 다양한 프롬프트에 대해 Bash 코드를 생성하는 능력을 보여주었습니다. 하지만 일부 모델들은 구문적, 의미적 문제를 일으켰고, 이를 EE 시스템이 정확히 포착하고 처리할 수 있었다는 점에서 그 중요성이 강조되었습니다. EE 시스템을 사용함으로써, 모델이 생성한 코드가 실제로 의도한 동작을 수행하는지의 여부를 정확하게 검증할 수 있었습니다. 이러한 접근 방식은 NL2Bash 작업을 포함한 다양한 프로그래밍 언어에 대한 코드 생성과 검증 작업에 중요한 기여를 할 수 있습니다.



### Summarizing Radiology Reports Findings into Impressions (https://arxiv.org/abs/2405.06802)
Comments:
          10 pages, 6 figures

- **What's New**: 이 연구에서는 의료 분야의 '환자 인계(patient hand-off)'와 '응급처지(triage)' 문제를 해결하기 위해 고도의 기술을 적용한 새로운 모델을 제시합니다. 특히, 복잡한 방사선학 보고서를 요약하여 전문가들과의 빠른 의사소통을 가능하게 하는 최첨단 성능을 구현하였습니다. 또한 의료 데이터를 증강하는 새로운 방법을 소개하고, 모델의 한계와 방사선학 지식 획득에 대해 분석했습니다.

- **Technical Details**: 본 연구의 주요 기술적 성과는 MIMIC CXR 데이터셋을 기반으로 한 데이터 처리 파이프라인(data processing pipeline)을 구축하고, BERT-to-BERT 인코더-디코더(encoder-decoder) 모델을 세밀하게 조정(fine-tuned)하여 사용한 점입니다. 이 모델은 ROUGE-L F1 58.75/100이라는 높은 점수를 기록하며, 보다 복잡한 주의 기제(attention mechanisms)를 사용한 전문화된 모델들을 능가했습니다.

- **Performance Highlights**: 이 모델은 특히 ROUGE-L F1 스코어에서 58.75/100의 성능을 보이며 방사선 보고서 요약 작업에 있어 뛰어난 결과를 제공하였습니다. 이는 다양한 attention mechanism을 적용한 전문화된 기타 모델들보다 우수한 성능을 보여주어, 병원 현장에서의 의사결정과 전달 과정의 효율성을 향상시킬 잠재력을 가집니다.



### LLM-Generated Black-box Explanations Can Be Adversarially Helpfu (https://arxiv.org/abs/2405.06800)
- **What's New**: 이 연구는 Large Language Models (LLMs)가 설명을 생성할 때 발생할 수 있는 ‘적대적 유용성(adversarial helpfulness)’ 문제를 조명합니다. 이 현상은 LLM이 잘못된 답변을 설득력 있게 설명하여, 사용자가 잘못된 결론을 신뢰하도록 만드는 것을 의미합니다. 이 문제는 LLM들이 질문을 재구성하거나, 고도의 확신을 표현하고, 선택적 증거를 제시하는 등의 전략을 사용하여 발생합니다.

- **Technical Details**: 연구 팀은 LLM의 '적대적 유용성' 문제를 평가하기 위해 인간 평가자와 프록시 모델 평가자를 사용하여 LLM 생성 설명의 설득력을 측정하였습니다. 그 결과, LLM은 높은 비율로 문제 재구성(reframing the question), 과신(confidence manipulation), 선택적 사실 제시(selective fact presentation) 등의 전략을 사용함을 발견했습니다. 또한, 연구팀은 심볼릭 추론 작업을 통해 LLM이 복잡한 구조적 지식을 탐색하는 능력을 테스트했습니다.

- **Performance Highlights**: 실험 결과, 상업용 LLM들(ChatGPT 등)이 제공한 설명은 인간 평가자들의 평가를 유의미하게 변화시켰으며, 이는 LLM이 잘못된 답변을 설득력 있게 만드는 능력을 가지고 있음을 보여줍니다. 이는 무분별한 사용 시 잘못된 정보나 결정으로 이어질 위험이 있음을 시사합니다.



### Opportunities for Persian Digital Humanities Research with Artificial Intelligence Language Models; Case Study: Forough Farrokhzad (https://arxiv.org/abs/2405.06760)
- **What's New**: 이 연구는 고급 자연어 처리(Natural Language Processing, NLP) 및 인공 지능(Artificial Intelligence, AI) 기술을 통합하여 페르시안 문학, 특히 포루그 파르호즈다드(Forough Farrokhzad)의 시를 분석하고 해석하는 것을 탐구합니다. 이는 페르시아 문학의 이해를 증진시키는 AI의 잠재력을 강조합니다.

- **Technical Details**: 연구는 변압기 기반 언어 모델(transformer-based language models)을 포함하는 AI 모델을 사용하여 비지도 학습 프레임워크(unsupervised framework)에서 시의 군집화(clustering)를 수행합니다. 이는 문체, 언어 및 주제적 패턴을 밝혀내기 위함입니다.

- **Performance Highlights**: 포루그 파르호즈다드의 작품을 종합적으로 분석함으로써, 페르시아 디지털 인문학(Persian Digital Humanities) 분야에 기여할 뿐만 아니라 향후 페르시아 문학 연구를 위한 계산 기법(computational techniques) 사용의 선례를 마련합니다.



### Enhancing Traffic Prediction with Textual Data Using Large Language Models (https://arxiv.org/abs/2405.06719)
- **What's New**: 이 연구에서는 대규모 언어 모델(large language models)을 이용하여 텍스트 정보를 처리하고 이를 역사적 교통 데이터와 결합하는 새로운 방법을 제안합니다. 이 접근법을 통해 비정형 텍스트 데이터(textual data)를 활용하여 단기 교통 예측 모델의 정밀도를 높이는 방법을 탐구하였습니다. 특히, 지역 수준(regional-level)과 노드 수준(node-level)의 특수 상황을 다루어, 각각 전체 도시와 특정 위치에 대한 영향을 예측하였습니다.

- **Technical Details**: 본 연구에서는 대규모 언어 모델을 직접적으로 교통 예측에 사용하지 않고, 대신 텍스트 정보에서 추출한 임베딩(embeddings)을 기존 공간-시간 예측 모델(spatiotemporal forecasting models)과 결합하여 예측값을 도출했습니다. 이러한 임베딩은 지역적 또는 노드 레벨의 특수한 상황을 모델 내에서 '노드'로 표현하고, 이를 네트워크의 기존 노드와 연결하여 추가 정보를 제공합니다. 연구에서는 STGCN과 같은 공간-시간 그래프 컨볼루션 네트워크(Spatiotemporal Graph Convolutional Networks)를 활용하여 공간과 시간에 대한 정보를 추출하고, 그 결과에 텍스트 정보가 추가된 새로운 입력 데이터를 생성하였습니다.

- **Performance Highlights**: 이 접근 방식은 뉴욕 자전거 데이터셋(New York Bike dataset)을 사용한 실험에서 예측 정확도를 현저히 향상시켰습니다. 지역 수준과 노드 수준의 시나리오에서 모두 긍정적인 결과를 보여, 특별한 이벤트나 기후 변화 같은 비정상적 상황에서도 높은 예측 성능을 보였습니다. 이 모델은 텍스트 정보가 예측에 미치는 영향을 효과적으로 포착하고, 전통적인 숫자 중심의 입력 데이터만을 사용하는 기존 모델들보다 우수한 예측 능력을 제공할 수 있음을 입증합니다.



### Enhancing Creativity in Large Language Models through Associative Thinking Strategies (https://arxiv.org/abs/2405.06715)
- **What's New**: 이 연구는 연관성 사고 (associative thinking)를 통한 큰 언어 모델 (Large Language Models, LLMs)의 창의력 향상을 조사합니다. 특히 vGPT-4 모델을 사용하여 제품 디자인(Product Design), 스토리텔링(Storytelling), 마케팅(Marketing)의 세 가지 영역에서 창의적 아웃풋을 증진시키는지 평가했습니다. 연구에서는 무관해 보이는 개념들을 연결하도록 LLM을 자극함으로써 모델의 창의적 능력을 향상시킬 수 있는 가능성을 탐구하였습니다. 이 연구는 LLM의 창의력을 활용한 기능성과 독창성을 향상시키는 새로운 지평을 열었습니다.

- **Technical Details**: 연구팀은 vGPT-4를 대상으로 중점적인 세 가지 도메인에 대한 창의력 과제를 설계하였고, 각 과제를 기준으로 모델의 성능을 독창성(originality)과 유용성(usefulness)을 중심으로 평가했습니다. LLM에게 임의의 객체나 속성을 창의적 출력에 통합하도록 요청함으로써, 모델이 덜 빈번하게 발생하는 단어 사이의 연관성을 형성할 수 있도록 도전하였습니다. 이는 연관성 있는 사고 방식이 인간의 창의력을 높이는 데 도움이 되는 기술로 알려져 있으며, 이를 LLM에 적용하여 더 효과적이고 창의적인 반응을 이끌어내었습니다.

- **Performance Highlights**: vGPT-4는 연관성 사고 기법(associative thinking techniques)을 활용할 때, 응답의 독창성이 크게 향상되었다는 결과를 보였습니다. 특히 제품 디자인, 스토리텔링, 마케팅 세 가지 영역에서 창의적 과제를 수행할 때 vGPT-4는 높은 수준의 독창적이고 유용한 컨텐츠를 생성할 능력을 보여주었습니다. 이러한 결과는 연관성 사고가 LLM의 창의적 능력을 실질적으로 강화할 수 있음을 시사하며, 다른 LLM 모델에도 이러한 접근 방식이 일반화될 가능성이 있습니다.



### Towards a path dependent account of category fluency (https://arxiv.org/abs/2405.06714)
Comments:
          To appear at CogSci 2024

- **What's New**: 이 연구에서는 카테고리 유창성을 위한 두 가지 상충되는 계정들을 해결하고자 새롭게 시도된 접근 방식을 제시하고 있습니다. 기존의 모델이 역설적으로 동일한 결과를 생성함에 따라, 연구자들은 더 복잡한 계층 구조와 전체 시퀀스를 기반으로 예측하는 대형 언어 모델(LLM)을 사용하여 카테고리 전환 확률을 직접 모델링하는 두 가지 추가적인 요소를 도입하였습니다.



### Unveiling the Competitive Dynamics: A Comparative Evaluation of American and Chinese LLMs (https://arxiv.org/abs/2405.06713)
- **What's New**: 본 연구에서는 경제 확장, 혁신, 사회 발전 및 국가 보안에서 Large Language Models(LLMs,  대형 언어 모델)의 중요성이 인식된 이후, 미국과 중국의 LLM의 영어 및 중국어 맥락에서 체계적인 비교 평가를 제시했습니다. 특히, ChatGPT 출시 이후 LLM의 전략적 중요성이 증대되었다는 점에 주목합니다.

- **Technical Details**: 본 연구는 천연 언어 숙련도(natural language proficiency), 학문적 전문 지식(disciplinary expertise), 안전성 및 책임감(safety and responsibility)을 포함하는 종합적인 평가 프레임워크를 제안했으며, 이를 바탕으로 미국과 중국의 주요 16개 모델을 다양한 운영 작업 및 시나리오에서 체계적으로 평가했습니다.

- **Performance Highlights**: 주요한 발견은 영어 환경에서는 GPT 4-Turbo가 선두에 있으며, 중국어 환경에서는 Ernie-Bot 4가 돋보인다는 것입니다. 또한, 언어 및 작업에 걸쳐 LLM 성능의 차이를 강조하며, 언어적 및 문화적 미묘함을 고려한 모델 개발의 필요성을 강조합니다. 미국과 중국의 LLM이 보여주는 보완적인 강점은 LLM 기술을 발전시키는 데 있어 미중 협력의 가치를 시사합니다.



### Digital Diagnostics: The Potential Of Large Language Models In Recognizing Symptoms Of Common Illnesses (https://arxiv.org/abs/2405.06712)
Comments:
          14 pages, 4 figures

- **What's New**: 최근 LLMs(Large Language Models)의 발전이 의료 및 건강 관리 분야, 특히 디지털 진단에서 혁신적인 기회를 제공하고 있습니다. GPT-4, Gemini, GPT-3.5와 같은 모델들이 보여주는 진단 능력이 향상되었으며, 이는 의료진이 고위험 진단을 내리는 데 도움이 될 수 있습니다. 각 모델의 특성을 이해하고 그들이 어떻게 의료 진단의 정확성과 효율성을 높일 수 있는지 평가하는 것이 중요합니다.

- **Technical Details**: 이 연구는 GPT-4, Gemini, GPT-3.5의 진단 정확성을 평가합니다. GPT-4는 의학 데이터에 대한 깊고 완벽한 학습 이력을 통해 높은 진단 정확성을 보여줍니다. Gemini는 질병 분류(triage)에서 높은 정밀도로 작동하며, GPT-3.5는 다소 덜 발전했지만 의료 진단에 유용한 도구입니다. 의료 및 임상 실습에서 LLMs를 연구하는 것의 필요성을 강조하며, HIPAA 준수와 같은 건강 정보 개인 정보 보호 법규를 준수하고 복잡한 의료 맥락에서 다양한 개인에게 영향을 미치는 사회적 결과를 다루는 것이 중요합니다.

- **Performance Highlights**: GPT-4는 진단 정확성에서 높은 성능을 보여주었으며, Gemini는 고위험 진단을 내릴 때 신뢰할 수 있는 모델로서의 잠재력을 입증했습니다. 또한 GPT-3.5는 여전히 유용한 도구로서 의료 진단을 수행할 수 있습니다.



### Mobile Sequencers (https://arxiv.org/abs/2405.06710)
- **What's New**: 이 연구는 언어와 계획된 협력적 행동의 공통 기원을 탐구하는 것을 목표로 하며, '변화의 의미론(semantics of change)'을 중심으로 다룹니다. 연구는 언어의 의미론과 구문론을 통해 변화와 불변을 추적하고 대처하는 방법을 다룬다고 제안합니다. 또한, 행동의 의미론은 구문 대신 계획을 통해 이해될 수 있다고 합니다. 이는 언어학(language)과 컴퓨터 과학(computer science) 모두에 중요하며, 대표적인 역사와 행동, 생각, 표현의 교차적 공개성(overtness)과 비밀성(covertness)에 의존해야 함을 시사합니다.

- **Technical Details**: 이 논문에서는 단어와 행동 순서의 의미 구조(semantics structure)가 단순한 연속성보다 더 많은 구조적 해석을 제공한다고 주장합니다. 이는 언어와 계획에서 '카테고리(category)' 개념을 재고하는 데 도움이 됩니다. 또한, 변화를 이해하는 데 중요한 요소로 모바일 시퀀서(mobile sequencers)와 그들이 변화를 기록하는 방식이 언급됩니다. 복잡하고 절차적인 카테고리들(complex and procedural categories)은 언어와 계획 모두에서 분해 가능성(decomposability)을 어떻게 제어하는지 설명하는 데 필요하다고 강조합니다.

- **Performance Highlights**: 언어와 계획의 공통 문제의 본질을 이해함으로써, 독립적으로 작업된 해결책들을 결합하는 것보다 새로운 이해를 제공할 것으로 보입니다. 언어 측면에서는 특정 계층 구조만이 지역적으로 결합될 수 있는 한계를 보여주는 수학적 결과가 있으며, 컴퓨터 과학 측면에서는 변화와 불변을 다루는 초기 연구를 개념적으로 지원하는 최근 시도들이 있습니다. 이들은 모두 변화하는 세계를 해석함에 있어 모바일 의미론의 프레임 문제(Frame Problem)와 관련이 있습니다.



### Evaluating the Efficacy of AI Techniques in Textual Anonymization: A Comparative Study (https://arxiv.org/abs/2405.06709)
- **What's New**: 이 연구는 텍스트 익명화 방법에 대한 포괄적인 검토를 시작하고 있으며, CRF(Conditional Random Fields), LSTM(Long Short-Term Memory), ELMo(Embeddings from Language Models), 그리고 Transformers 아키텍처의 변혁적 능력에 중점을 두고 있습니다. 각 모델은 독특한 강점을 보여주며, 이는 텍스트 익명화 문제를 해결하는데 있어 상호 보완적 잠재력을 강조합니다.

- **Technical Details**: CRF는 단어 시퀀스 간의 의존성을 포착하고, LSTM은 장기 의존성을 모델링하는 데 탁월하며, ELMo는 양방향 언어 모델을 사용하여 문맥적 단어 표현을 제공합니다. Transformers는 스케일링을 향상시키는 self-attention 메커니즘을 도입합니다. 이 연구는 이러한 모델들을 비교 분석하면서 NER(Name Entity Recognition) 데이터셋에서 모델의 성능을 분석하여 텍스트 익명화 영역에서의 최적 솔루션을 모색합니다.

- **Performance Highlights**: 예비 결과에 따르면 CRF, LSTM 및 ELMo는 전통적인 방법보다 개별적으로 우수한 성능을 보였습니다. Transformers의 포함은 다른 모델들과 비교할 때 보다 넓은 관점에서 최적의 텍스트 익명화를 달성하는 방법을 제공합니다.



### Hypothesis Testing Prompting Improves Deductive Reasoning in Large Language Models (https://arxiv.org/abs/2405.06707)
- **What's New**: 이 논문에서는 가설 검증 프롬프팅(Hypothesis Testing Prompting)을 제안하여 사전 훈련된 대규모 언어 모델을 기반으로 한 논리적 추론 과제를 새롭고 효과적으로 해결합니다. 이 접근법은 결론 가정, 역추론(backward reasoning), 중간 추론 단계에서의 사실 확인을 포함합니다.

- **Technical Details**: 가설 검증 프롬프팅은 다단계 추론 과정에서 각 단계마다 특정한 결론을 가정하고 나아가 이를 검증하는 절차를 포함합니다. 이 방법은 Chain-of-Thought (CoT) 프롬프팅에 기반을 둔 것으로, 추론과정에서의 연결고리(chain)를 생성하며 문제 해결을 돕습니다. 연구는 두 가지 복잡한 추론 데이터셋, RuleTaker와 ProofWriter를 사용하여 가설 검증 프롬프팅의 유효성을 검증하였습니다.

- **Performance Highlights**: 실험 결과, 가설 검증 프롬프팅이 기존의 프롬프팅 방법인 표준 프롬프팅(Standard prompting) 및 Chain-of-Thought 프롬프팅과 비교하여 더 높은 정확도와 더 일관된 추론 과정을 제공하는 것으로 나타났습니다. 특히 ProofWriter 데이터셋에서 4% 이상의 정확도 향상을 보여, 복잡한 로직 작업에서의 유효성이 입증되었습니다.



### Exploring the Capabilities of Large Multimodal Models on Dense Tex (https://arxiv.org/abs/2405.06706)
- **What's New**: 이 연구는 다중 모달 모델(LMM, Large Multi-modal Models)이 집중적인 텍스트가 포함된 이미지를 이해하는 능력을 향상시키기 위해 DT-VQA 데이터셋을 제안하고, 평가하는 것에 초점을 맞추고 있습니다. DT-VQA 데이터셋에는 문서, 표, 제품 설명과 같은 밀집된 텍스트 정보를 포함하는 30k개의 이미지에서 170k개의 질문-답변 쌍이 포함되어 있습니다. 이 연구는 또한 prompt engineering 및 downstream fine-tuning과 같은 두 가지 전략의 효과를 평가하여, 텍스트 처리 작업에서 LMM의 한계를 극복하고자 시도합니다.

- **Technical Details**: 연구 팀은 Gemini와 GPT-4V를 사용하여 질문-답변 쌍을 자동 생성하고, OCR(광학 문자 인식) 정보와 이미지를 통합하도록 입력 프롬프트를 세심하게 설계했습니다. DT-VQA 데이터셋은 고정 된 레이아웃에서 비정형 장면에 이르기까지 다양한 텍스트 스타일을 포함합니다. 또한, 새로운 평가 메트릭인 AccANLS를 도입하여 VQA 작업의 평가를 개선하고자 합니다. LMM의 성능을 향상시키기 위해 사용된 두 가지 전략은 프롬프트 엔지니어링(prompt engineering)과 다운스트림 파인 튜닝(downstream fine-tuning)입니다.

- **Performance Highlights**: LMM은 DT-VQA 데이터셋에서 다양한 성능을 보였으며, 특히 표와 장면 텍스트 이미지에서 상대적으로 더 나은 성능을 보였습니다. 프롬프트 엔지니어링은 일부 모델에서 평균 7.6%의 ANLS 성능 개선을 가져왔으며, 다운스트림 파인 튜닝은 Qwen-VL과 Monkey 모델에서 각각 16.5% 및 12.2%의 AccANLS 성능 개선을 이뤄냈습니다.



### LLMs can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Though (https://arxiv.org/abs/2405.06705)
Comments:
          To appear at IJCAI 2024

- **What's New**: 이 연구에서는 수학적 추론 오류를 식별하기 위한 새로운 ‘Pedagogical Chain-of-Thought (PedCoT)’ 전략을 소개합니다. 이 방법은 교육 이론 Bloom Cognitive Model (BCM)에 기반하여 대규모 언어 모델(Large Language Models, LLMs)을 위한 프롬프트 디자인을 재구성하는 접근 방식을 제공합니다. 이 연구는 교육 이론이 언어 모델의 효과적인 사용을 지원하는 데 어떻게 기여할 수 있는지 보여줍니다.

- **Technical Details**: PedCoT 전략은 세 가지 주요 구성 요소를 포함합니다: (1) 교육원리에 근거한 프롬프트 (Pedagogical Principles of Prompt Design, PPP) 디자인 (2) 두 단계의 상호작용 과정 (Two-stage Interaction Process, TIP) (3) 실제론 연결된 프롬프트 (Grounded PedCoT Prompts). 이 전략은 수학 문제를 해결함에 있어 LLMs의 추론 능력을 강화하는 데 중점을 두고 있으며, 교육적 관점에서 프롬프트 전략을 설계하는 것의 중요성을 강조합니다.

- **Performance Highlights**: 실험 결과, PedCoT 전략은 강력한 기준 모델과 비교하여 뛰어난 성능을 보였습니다. 공개 데이터셋을 사용한 평가에서 이 방법은 수학적 추론 오류를 식별하는데 기존의 방법보다 훨씬 효과적인 것으로 나타났습니다. 특히, 이 연구는 교육 이론을 도메인 지식으로 활용하여 LLMs의 추론 능력을 향상시킬 수 있는 가능성을 보여주었습니다.



### Enhanced Review Detection and Recognition: A Platform-Agnostic Approach with Application to Online Commerc (https://arxiv.org/abs/2405.06704)
- **What's New**: 이 논문은 온라인 리뷰에 대한 새로운 감지 및 추출 방법을 제시하며, 이 방법은 학습 데이터에 포함되지 않은 웹사이트에서도 사용할 수 있는 일반화된 솔루션을 제공합니다. 특히, 컴퓨터 비전(CV, Computer Vision) 기술을 활용하여 웹 페이지에서 리뷰 섹션을 탐지하고, 관련 텍스트를 인식하는데 OCR(Optical Character Recognition)을 사용합니다. 이 접근 방식은 기존의 HTML 스크래핑 방법보다 더 견고하며, 다양한 언어 지원 및 가짜 리뷰 탐지 등의 추가 기능을 통합하여 리뷰 분석의 범용성과 유용성을 강화하였습니다.

- **Technical Details**: 이 연구에서는 Yolov8 모델과 Pytesseract를 사용하여 웹 페이지의 리뷰 섹션을 자동으로 탐지하고 텍스트를 인식합니다. Yolov8 모델은 사용자 리뷰와 관련된 객체를 탐지하는 데 특화된 트레이닝을 받았으며, 탐지된 텍스트는 Pytesseract로 인식되어 처리됩니다. 이 기술을 사용하여 리뷰의 감정 불일치 분석(Emotion Inconsistency Analysis), 다중 언어 지원(Multi-language Support), 그리고 가짜 리뷰 탐지(Fake Review Detection) 같은 핵심 애플리케이션을 구현하였습니다.

- **Performance Highlights**: 제시된 방법은 다양한 언어의 리뷰를 추출하고 번역하는 강력한 다중 언어 지원 기능을 제공합니다. 또한, 일관된 시각적 요소를 기반으로 온라인 리뷰를 자동으로 탐지 및 인식하는데 탁월한 성능을 보여주며, 가짜 리뷰를 식별하고 구별하는 데에도 효과적입니다. 이를 통해 소비자의 신뢰를 높이고, 구매 결정에 유용한 정보를 제공하여 사업체의 성공에 기여할 수 있습니다.



### Interpretable Cross-Examination Technique (ICE-T): Using highly informative features to boost LLM performanc (https://arxiv.org/abs/2405.06703)
- **What's New**: 이 논문에서는 의학 및 법률과 같이 해석 가능성(interpretability)이 중요한 분야에서 대형 언어 모델(Large Language Models, LLMs)의 분류 성능을 개선하기 위해 구조화된 멀티 프롬프트 기법을 활용하는 새로운 접근 방식인 Interpretable Cross-Examination Technique (ICE-T)을 소개합니다. ICE-T는 여러 질문을 통해 문제를 다각도로 접근하게 하며, 이를 통해 얻은 LLM의 응답을 수치적 특징 벡터로 변환하여 전통적인 분류기를 사용해 처리합니다.

- **Technical Details**: ICE-T는 LLM에 여러 프롬프트를 제공하고, 이에 대한 응답을 결합하여 의사 결정을 내리는 기법입니다. 응답은 수치화되어 특징 벡터를 생성하고, 이는 전통적인 분류기(classifier)에 입력되어 최종 결과를 결정합니다. 이 방법은 해석 가능성을 유지하면서도 작은 모델에서도 큰 모델의 성능을 달성하거나 초과할 수 있는 가능성을 제공합니다. ICE-T는 의학 기록 및 법률 문서와 같이 다양한 데이터 소스에 대해 효과적임을 입증하였으며, F1 점수 등의 분류 메트릭(classification metrics)에서 일관되게 뛰어난 성능을 보여줍니다.

- **Performance Highlights**: ICE-T는 zero-shot 기준 모델 대비 일관되게 높은 분류 메트릭을 달성하며, 더 작고 능력이 덜한 모델을 사용하여도 더 크고 발전된 모델을 사용한 zero-shot 접근 방식과 비교하여 경쟁적이거나 더 나은 결과를 달성할 수 있습니다. 또한, 이 접근 방식은 높은 수준의 해석 가능성을 제공하여 전문가가 의사 결정 과정 뒤에 있는 논리를 명확하게 이해할 수 있게 합니다.



### Malayalam Sign Language Identification using Finetuned YOLOv8 and Computer Vision Techniques (https://arxiv.org/abs/2405.06702)
- **What's New**: 이 논문은 케랄라 주의 농아인을 위한 말라얄람 수화 인식 모델을 제안하고 있다. 2021년 말라얄람 수화(Malayalam Sign Language, MSL)가 도입된 이후, 이 지역 사회의 의사소통 개선을 위한 필요성이 대두되었다. 이 연구는 YOLOv8과 컴퓨터 비전(computer vision) 기술을 사용하여 말라얄람 수화를 실시간으로 인식하고, 해당 문자를 실시간 비디오에서 인식하여 자막으로 표시하는 모델을 개발하는 것을 목표로 한다.

- **Technical Details**: 이 연구에서는 말라얄람 언어의 수화 데이터셋을 레이블링하고 YOLOv8을 포함한 고급 딥러닝(deep learning) 기법을 사용하여 수화를 인식한다. YOLO(You Only Look Once)는 단일 알고리즘 패스로 이미지나 비디오 내 다수 객체를 실시간으로 정확하게 식별하고 위치를 지정하는 최첨단 객체 감지 알고리즘이다. 또한, 이 연구는 Roboflow와 Ultralytics 플랫폼을 사용하여 데이터 수집 및 전처리를 효율적으로 관리하고 모델 학습을 최적화한다.

- **Performance Highlights**: 실험 결과, 이 말라얄람 수화 인식 모델은 다른 수화 인식 시스템과 비교하여 유사한 인식 정확도를 제공한다. 특히, YOLOv8을 사용하여 개발된 이 모델은 높은 정밀도(precision)와 재현율(recall), 그리고 평균 정밀도(mean Average Precision, mAP)를 보여준다. 더욱이, 이 모델은 실시간 비디오에서 말라얄람 수화를 정확하게 인식하고 해석함으로써, 청각 장애인과 비장애인 간의 의사소통 개선에 크게 기여할 수 있다.



### Lightweight Spatial Modeling for Combinatorial Information Extraction From Documents (https://arxiv.org/abs/2405.06701)
- **What's New**: KNN-Former는 복잡한 공간 구조를 지닌 문서의 엔티티 분류를 위해 공간적 편향을 적용한 새로운 모델입니다. 이 모델은 문서 엔티티의 K-최근접 이웃(K-nearest-neighbor, KNN) 그래프를 기반으로 attention 계산에 공간적 편향을 적용하며, KNN 그래프에 정의된 로컬 반경 내에서만 엔티티의 attention을 제한합니다. 또한, 많은 문서에서 발견되는 일대일 매핑(one-to-one mapping) 특성을 처리하기 위해 조합적 매칭(combinatorial matching)을 사용합니다. 추가로, 다양한 템플릿과 언어를 포함하는 새로운 ID 문서 데이터셋을 공개하여 이러한 유형의 문서에 대한 연구를 촉진합니다.

- **Technical Details**: KNN-Former는 Transformer 기반 모델로, 높은 매개변수 효율성을 가집니다. 기존의 언어 모델을 초기화하는 매개변수 없이 설계되었으며, 기존 baseline에 비해 훨씬 작은 학습 가능한 매개변수를 가집니다. 모델은 KNN 그래프의 상대적 유클리드 거리를 기반으로 구성되고, 이 그래프에서 정의된 hop 거리를 이용하여 엔티티 쌍의 attention 가중치를 계산합니다. 이러한 접근 방식은 문서의 복잡한 공간적 관계를 더 잘 포착할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, KNN-Former는 다양한 데이터셋에 걸쳐 대부분의 필드 범주에서 기존 baseline을 초과하는 성능을 보였습니다. 또한, KNN 기반의 공간적 인덕티브 바이어스(inductive bias)와 조합적 매칭의 중요성을 확인할 수 있는 광범위한 소거 실험(ablation study)을 통해 그 효과를 입증하였습니다.



### ChatSOS: Vector Database Augmented Generative Question Answering Assistant in Safety Engineering (https://arxiv.org/abs/2405.06699)
- **What's New**: 이 연구는 자연어 처리 기술의 급속한 발달로 인해 대규모 언어 모델(Large Language Models, LLMs)을 활용한 생성형 인공지능 기술이 안전 공학 분야에서 유의미한 잠재력을 보여주고 있습니다. 특히, 이번 연구는 중국의 2013년부터 2023년까지의 폭발 사고 보고서 117건을 기반으로 벡터 데이터베이스를 개발하여, LLMs의 성능을 크게 향상시키는 방법을 제시합니다.

- **Technical Details**: 이번 연구에서는 벡터 데이터베이스(vector database) 개발을 위해 텍스트 코퍼스를 세분화(segmenting)하고 벡터 임베딩(vector embedding) 기술을 사용했습니다. 이 벡터 데이터베이스는 기존 관계형 데이터베이스(relational database)에 비해 정보 검색 품질에서 우수한 성능을 보였으며, 이를 통해 LLMs에 보다 풍부하고 관련성 높은 지식을 제공할 수 있었습니다.

- **Performance Highlights**: 비교 분석을 통해 ChatSOS와 같은 LLMs가 신뢰성, 정확도, 포괄성을 크게 개선하며 응답의 적응성과 명확성을 향상시키는 것이 확인되었습니다. 이 결과는 LLMs에 외부 데이터베이스를 보강할 때의 효과성을 입증하며, 안전 공학에서 전문적인 질의응답 처리 가능성을 높이고 더 넓은 응용을 위한 기반을 마련합니다.



### Automated Conversion of Static to Dynamic Scheduler via Natural Languag (https://arxiv.org/abs/2405.06697)
Comments:
          7 pages (excluding appendix), 10 figures, 3 tables

- **What's New**: 이 논문에서는 기존의 정적(Static) 모델을 주어진 상태에서 동적 스케줄링 문제(Dynamic Scheduling Problems)에 대한 제약 사항을 모델링하고 코드를 생성할 수 있는 대형 언어 모델(Large Language Models, LLM)의 적용 가능성을 탐구합니다. 특히, 이 연구는 최적화 모델링 전문가 없이도 동적 변화와 불확실성에 대응할 수 있는 정적 최적화 모델을 자동으로 수정할 수 있는 LLM 기반 방법론을 제안합니다.

- **Technical Details**: 제안된 방법은 검색-증강 생성(Retrieval-Augmented Generation, RAG)을 기반으로 하며, RAGDyS(Retrieval-Augmented Generation for Dynamic Scheduling) 모델을 이용하여 동적 스케줄링 제약을 자동으로 구현할 수 있습니다. 예로, 간호사 스케줄링 문제를 동적 문제로 확장하여 어떻게 LLM이 필요한 제약 변경을 제공할 수 있는지를 탐색합니다. 이 모델은 자연어 제약 설명을 통해 스케줄에 반영된 변경을 빠르게 얻을 수 있도록 설계되었습니다.

- **Performance Highlights**: 이 연구의 중요한 기능은 기존 스케줄에서의 최소 변동 문제(Minimum Perturbation Problem)를 해결하기 위해 기존 스케줄을 바탕으로 자동으로 수정된 스케줄을 생성하는 것을 목표로 합니다. RAGDyS 모델은 최적화 전문가 없이도 최종 사용자가 스스로 모델을 업데이트할 수 있게 하여, 동적 환경 변화에 신속하게 대응할 수 있는 가능성을 열어줍니다.



### Multi-level Shared Knowledge Guided Learning for Knowledge Graph Completion (https://arxiv.org/abs/2405.06696)
Comments:
          The paper has been accepted for publication at TACL. And the arXiv version is a pre-MIT Press publication version

- **What's New**: 이 논문에서는 데이터셋과 태스크(작업) 레벨에서 작동하는 새로운 다중 레벨 공유 지식 안내 학습 방법(SKG)을 제안하여 지식 그래프 완성(Knowledge Graph Completion, KGC)의 성능을 향상시키고 있다. 특히, 이 모델은 데이터셋 레벨에서 공유된 특성을 식별하고 태스크 레벨에서는 다이내믹하게 조절되는 손실 가중치를 사용하는 혁신적인 멀티 태스크 학습 아키텍처를 도입하고 있다.

- **Technical Details**: SKG-KGC는 텍스트 요약을 이용해 원래 데이터셋을 확장함으로써 공유된 지식을 활용하여 지식 트리플렛의 표현을 강화한다. 태스크 레벨에서는 머리 엔티티(head entity) 예측, 관계(relation) 예측, 꼬리 엔티티(tail entity) 예측의 세 가지 하위 태스크에 대해 멀티 태스크 학습 아키텍처를 사용한다. 이는 도전적이고 성능이 낮은 태스크에 더 집중할 수 있도록 모델을 조정하며, 지식 공유의 불균형을 완화시킨다.

- **Performance Highlights**: 실험 결과, SKG-KGC는 WN18RR, FB15k-237, Wikidata5M 세 가지 유명 데이터셋에서 기존의 텍스트 기반 방법들을 크게 능가했다는 것을 보여준다. 특히 WN18RR 데이터셋에서 두드러진 성능 향상을 보였다.



### Utilizing Large Language Models to Generate Synthetic Data to Increase the Performance of BERT-Based Neural Networks (https://arxiv.org/abs/2405.06695)
Comments:
          Published in 2024 American Medical Informatics Association (AMIA) Summit March 18-21

- **What's New**: 이 연구는 의료 분야에 전문가 부족 문제를 해결하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 통해 데이터 생성을 평가하였습니다. 특히, 자폐 스펙트럼 장애(Autism Spectrum Disorders, ASD)를 대상으로 ChatGPT와 GPT-Premium을 사용하여 4,200개의 합성 관측 데이터를 생성하여 기존 의료 데이터를 보강하는 데 초점을 맞췄습니다. 이러한 접근 방식은 합성 훈련 데이터를 사용하여 모델의 정확성을 향상시키려는 목표를 가집니다.

- **Technical Details**: 연구 팀은 생물 의학 문헌에 사전 훈련된 BERT 분류기를 사용하여 모델 간 성능 차이를 평가했습니다. 합성 데이터는 LLM을 통해 생성되었으며, 의료 전문가에 의해 무작위 샘플(N=140)이 평가되었고, 이중 83%가 올바른 예시-레이블 쌍으로 확인되었습니다.

- **Performance Highlights**: 데이터 보강은 재현율(recall)을 13% 증가시켰지만, 정밀도(precision)는 16% 감소시켰습니다. 이는 더 높은 품질과 낮은 정확도 간의 상관관계를 시사합니다. 향후 연구에서는 다양한 합성 데이터 특성이 머신러닝(Machine Learning, ML) 결과에 미치는 영향을 분석할 계획입니다.



### SUTRA: Scalable Multilingual Language Model Architectur (https://arxiv.org/abs/2405.06694)
- **What's New**: 이 논문에서는 50개 이상의 언어를 이해, 추론 및 생성할 수 있는 다국어 대규모 언어 모델(Large Language Model, LLM) 구조인 SUTRA를 소개합니다. SUTRA는 개념 이해와 언어별 처리를 분리하는 독특한 디자인을 통해 확장 가능하고 효율적인 다국어 정렬 및 학습을 용이하게 합니다. 전문가의 혼합(Mixture of Experts, MoE) 프레임워크를 사용하여 SUTRA는 계산 효율성과 반응성을 모두 보여줍니다.

- **Technical Details**: SUTRA는 개념 학습과 언어 학습을 분리하여 핵심 모델이 보편적인 언어 독립적 개념에 집중할 수 있도록 합니다. 이 구조는 특수한 신경 기계 번역(Neural Machine Translation, NMT) 기전을 활용하여 언어별 처리를 지원하며, 모델의 확장성이나 성능을 해치지 않으면서 언어적 뉘앙스를 유지합니다. MoE 전략을 사용함으로써 언어 작업에 따라 관련 전문가만을 활성화하여 모델의 효율성을 향상시킵니다. 또한, SUTRA 모델은 인터넷에 연결되어 있으며 환각이 없는 정확하고 최신의 답변을 제공할 수 있습니다.

- **Performance Highlights**: SUTRA는 기존 모델들인 GPT-3.5, Llama2를 대폭 개선하여 Massive Multitask Language Understanding (MMLU) 벤치마크에서 20-30% 향상된 성능을 보여주었습니다. 다국어 작업에서의 우수한 성능과 함께, 다양한 언어 환경에서의 접근성 및 유용성을 증대시키는데 기여할 것으로 기대됩니다.



### Analyzing Language Bias Between French and English in Conventional Multilingual Sentiment Analysis Models (https://arxiv.org/abs/2405.06692)
Comments:
          Undergraduate Research Project

- **What's New**: 이 연구는 영어와 프랑스어 간의 다국어 감정 분석에서 잠재적인 언어 편향을 탐구합니다. 통계 캐나다에서 발표한 'Bilingual Natural Language Processing에 대한 편향 고려' 보고서에 영감을 받아, SVM(Support Vector Machine)과 Naive Bayes 모델을 사용하여 프랑스어 데이터가 영어 데이터를 효율성 측면에서 우수하게 수행하며 프랑스어에 유리한 언어 편향을 나타낸다는 가능성을 밝혀냈습니다. 또한, Fairlearn 도구를 활용하여 언어간 공정성을 평가했습니다.

- **Technical Details**: 다양한 다국어 데이터셋에서 공정한 자연어 처리(Natural Language Processing, NLP) 시스템의 중요성을 강조하며, Fairlearn으로 평가된 SVM과 Naive Bayes 모델을 사용하여 언어 편향을 조사하였습니다. SVM은 세 가지 독립된 데이터셋에서 인구 통계 비율(demographic parity ratio)을 0.963, 0.989, 0.985로 합리적 수준에 도달했습니다. 그러나 Naive Bayes는 0.813, 0.908, 0.961로 더 큰 불균형을 보여주었습니다.

- **Performance Highlights**: Fairlearn 분석 결과에 따르면, SVM 모델은 언어 간 거의 공정한 처리를 보여주었지만, Naive Bayes 모델은 여전히 유의미한 차이를 드러냈습니다. 프랑스어 데이터는 정확도(accuracy), 재현율(recall), F1 점수(F1 score)에서 영어 데이터를 능가하는 성능을 보였습니다. 이는 다국어 NLP 시스템에서 데이터 다양성의 증가가 시스템의 공정성에 얼마나 중요한지를 시사합니다.



### Fleet of Agents: Coordinated Problem Solving with Large Language Models using Genetic Particle Filtering (https://arxiv.org/abs/2405.06691)
Comments:
          11 pages, 1 figure, 4 tables

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs: Large Language Models)을 이용하여 동적 트리 검색을 탐색하는 에이전트로 사용하는 새로운 프레임워크인 '에이전트의 함대(Fleet of Agents, FoA)'를 소개합니다. FoA는 각 에이전트가 독립적으로 탐색을 수행하고, 발견된 해결책에 기반하여 탐색 전략을 조절하는 동적 분기 기능을 가능하게 합니다.

- **Technical Details**: FoA는 유전자형 입자 필터링 접근법을 사용하여 에이전트들이 독립적으로 탐색 공간을 탐색하고, 추후 휴리스틱 가치 함수(heuristic value function)에 기반한 재추출을 통해 탐사와 활용 사이의 균형을 최적화합니다. 이 프레임워크는 높은 다양성을 유지하거나 유망한 탐색 경로에 집중할 수 있는 동적 분기 인자를 자연스럽게 조절할 수 있습니다.

- **Performance Highlights**: FoA는 '24 게임(Game of 24)'과 '미니 크로스워드(Mini-Crosswords)'라는 두 벤치마크 작업에서 실험적으로 검증되었으며, 기존에 제안된 '생각의 나무(Tree of Thoughts)' 방법보다 효과성과 효율성 면에서 우수한 성능을 보였습니다. 특히, FoA는 계산 비용을 크게 줄이면서 (가치 함수 호출 빈도를 감소시키면서) 비교적 높거나 우수한 정확도를 유지합니다.



### Hire Me or Not? Examining Language Model's Behavior with Occupation Attributes (https://arxiv.org/abs/2405.06687)
Comments:
          Under review

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)이 성별 스테레오타입과 관련된 직업 결정에서 어떻게 동작하는지 조사합니다. 특히, RoBERTa-large, GPT-3.5-turbo, Llama2-70b-chat 같은 모델들을 사용해 다단계 질문 응답을 통해 성별 스테레오타입의 존재를 조사하고 정량화하였습니다.

- **Technical Details**: 연구팀은 다단계 성별 스테레오타입 검증(Multi-step Gender Stereotype Verification) 프레임워크를 제안하고, O*NET에서 제공하는 표준 직업 분류 지식 기반을 활용하여 데이터셋을 구축했습니다. 테스트는 세 가지 LLMs(RoBERTa-large, GPT-3.5-turbo, Llama2-70b-chat)을 사용하여 수행되었으며, 모델들은 서로 다른 선호도를 보이는 성별 스테레오타입을 나타냈습니다.

- **Performance Highlights**: 모든 테스트된 LLMs는 인간의 편향과 유사한 성별 스테레오타입을 드러냈지만, 전통적인 성별 스테레오타입과 상충하는 새로운 편향을 도입할 수 있는 현 진행 방식의 한계를 시사합니다. RoBERTa-large는 성별 스테레오타입이 일관성과 관련이 있는 것으로 나타났으며, GPT-3.5-turbo와 Llama2-70b-chat은 서로 다른 선호도를 보여, 편향 완화 성능을 향상시킬 수 있는 고급 기술 연구가 추가로 필요함을 암시합니다.



### Word2World: Generating Stories and Worlds through Large Language Models (https://arxiv.org/abs/2405.06686)
- **What's New**: 이 연구에서는 새로운 시스템인 Word2World를 소개하고 있으며, 이는 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 스토리로부터 플레이 가능한 2D 게임 레벨을 생성할 수 있는 능력을 보여줍니다. Word2World는 특정 작업에 대한 사전 훈련이나 미세 조정 없이도 LLM의 이러한 능력을 활용하여 다양한 게임 콘텐츠를 생성할 수 있습니다.

- **Technical Details**: Word2World는 LLM을 중심으로 구축되어 있으며, 이 시스템은 스토리를 기반으로 캐릭터 정보, 타일셋 정보, 목표 등을 추출하고 이를 바탕으로 2D 게임 월드를 구성합니다. 이 과정은 두 단계로 나뉘며, 첫 번째 단계에서는 환경을 설정하고, 두 번째 단계에서는 캐릭터와 중요 상호 작용 타일을 배치합니다. 또한, 플레이 가능성과 시각적 일관성을 평가하기 위해 LLM과 기존의 절차적 콘텐츠 생성(Procedural Content Generation, PCG) 방법을 사용합니다.

- **Performance Highlights**: Word2World는 다양한 LLM과 함께 테스트되었으며, 각 단계의 유효성을 검증하기 위한 철저한 벌점 연구(ablation study)가 수행되었습니다. 이 시스템은 뛰어난 2D 게임 생성 능력을 보여주며, 특히 타일 및 캐릭터 배치에 대한 복잡성을 잘 관리하여 사용자에게 몰입감 있는 게임 경험을 제공하고 있습니다.



### Multigenre AI-powered Story Composition (https://arxiv.org/abs/2405.06685)
- **What's New**: 이 논문은 인터랙티브 스토리 작성을 테마의 일관성을 유지하면서 안내할 수 있는 장르 패턴(genre patterns)을 구성하는 방법을 보여줍니다. 특히, AI 에이전트를 활용하여 장르에 따른 스토리 구성을 돕는 PatternTeller 프로토타입을 활용하였습니다.

- **Technical Details**: 장르 패턴의 생성은 두 단계 과정을 통해 이루어집니다. 첫 번째 단계에서는 우리가 정의한 장르의 특성에 맞는 예시들을 수집하고, 두 번째 단계에서는 가장 특정적 일반화(most specific generalization)를 적용하여 이 예시들의 공통점을 찾습니다. 이 과정에서 AI 에이전트가 중요한 역할을 합니다.

- **Performance Highlights**: 이 연구는 interactive narrative composition에 있어서 기본 장르(fundamental genre)에 연관된 서사 구조를 사용하면서 테마의 일관성을 유지하는 새로운 전략을 제시합니다. 연구는 또한 narrative-based video games에도 적용 가능성을 탐구합니다.



### QuakeBERT: Accurate Classification of Social Media Texts for Rapid Earthquake Impact Assessmen (https://arxiv.org/abs/2405.06684)
- **What's New**: 이 연구는 지진 영향 분석을 위해 특화된 첫 번째 도메인 특화 대규모 언어 모델(Large Language Model, LLM)인 QuakeBERT를 제안합니다. 이 모델은 소셜 미디어에서 수집한 데이터를 효과적으로 분류하고 필터링하여 신속한 지진 영향 평가를 가능하게 합니다. 또한, 이 연구는 공공 의견 추세 분석, 감정 분석, 키워드 기반 물리적 영향 정량화를 통합한 새로운 방법론을 소개하여 지진의 물리적 및 사회적 영향을 평가합니다.

- **Technical Details**: QuakeBERT는 7282개의 지진 관련 마이크로블로그를 포함하는 데이터셋을 기반으로 개발되었습니다. 이 데이터는 20개 다른 지역의 지진에서 수집되었습니다. 연구진은 데이터 다양성과 데이터 양이 QuakeBERT의 성능에 중요한 영향을 미친다는 것을 발견했습니다. 이 모델은 CNN이나 RNN 기반 모델들을 능가하는 성능을 보여주며, 매크로 평균 F1 점수를 60.87%에서 84.33%로 향상시켰습니다.

- **Performance Highlights**: QuakeBERT는 기존 CNN 또는 RNN 기반 모델들보다 우수한 분류 성능을 보여주며, 뛰어난 정확성을 드러냈습니다. 테스트 결과 매크로 평균 F1 점수가 27% 향상되었습니다. 또한, 이 모델을 사용하여 동일한 규모와 초점 깊이를 가진 두 지진의 영향을 평가했을 때, 소음이 많은 마이크로블로그를 정확하게 감지하고 효과적인 재난 대응을 가능하게 합니다.



### ERAGent: Enhancing Retrieval-Augmented Language Models with Improved Accuracy, Efficiency, and Personalization (https://arxiv.org/abs/2405.06683)
Comments:
          Draft Paper

- **What's New**: 새로운 연구에서는 언어 모델의 이해도를 향상시키기 위해 검색 기능이 향상된 생성(Retrieval-augmented generation, RAG)을 개선하는 최신 프레임워크인 ERAGent가 소개되었습니다. ERAGent는 향상된 질문 리라이터(Enhanced Question Rewriter)와 지식 필터(Knowledge Filter)를 통합하여 검색 품질을 향상시킵니다. 또한, 검색 트리거(Retrieval Trigger)는 필요 없는 외부 지식의 검색을 억제함으로써 응답 품질을 저하시키지 않습니다. ERAGent는 사용자 프로필을 학습하여 응답을 개인화합니다.

- **Technical Details**: ERAGent는 RAG 영역에서 발전을 표현하며, 주요 기술적 특징으로는 향상된 질문 리라이터를 사용하여 더 정교한 질문으로 재구성하고, 지식 필터를 통해 검색된 정보의 관련성을 평가합니다. 또한, 사전 학습된 트랜스포머(pre-trained transformers)와 결합하여 지식 검색과 생성의 공동 최적화를 실현합니다. 이러한 기능은 사용자 맥락에 따라 응답을 맞춤화하는 개인화된 LLM 리더(Personalized LLM Reader)와 함께 작동하여, 검색 트리거와 개인화된 리더가 사용자 프로필을 학습할 수 있는 경험적 학습자(Experiential Learner) 모듈에 의해 지원됩니다.

- **Performance Highlights**: ERAGent는 세 가지 질문 응답과 여섯 개의 데이터셋을 통한 철저한 실험을 수행한 결과, 우수한 정확도, 효율성 및 개인화를 자랑합니다. 이러한 결과는 ERAGent의 강력한 효과성을 강조할 뿐만 아니라, 그 구성 요소들 사이의 시너지 효과를 더욱 부각시키며, 실제 시스템에서의 응용 가능성을 강조합니다.



### Self-Reflection in LLM Agents: Effects on Problem-Solving Performanc (https://arxiv.org/abs/2405.06682)
- **What's New**: 이 연구는 대규모 언어 모델들(Large Language Models, LLMs)에서 자기 성찰(self-reflection)이 문제 해결 성능에 미치는 효과를 조사합니다. 자기 성찰을 통해 LLM 에이전트들이 자신의 실수를 반성하고 향후 유사한 오류를 피할 수 있도록 지침을 스스로 제공함으로써, 문제 해결 능력이 유의미하게 향상됨을 발견했습니다.

- **Technical Details**: 우리는 여러 유명한 LLM (예: GPT-4, Llama 2 70B, Google Gemini 등)을 사용하여 다양한 도메인(수학, 과학, 의학 등)에서 무작위로 선택된 1,000개의 다지선다형 문제(Multiple-Choice Questions, MCQs)를 풀도록 했습니다. 각 LLM은 자체의 Chain of Thought (CoT)를 생성하며, 잘못된 답변을 한 경우, 여덟 가지 유형의 자기 성찰 기법을 사용하여 이를 개선하도록 합니다. 이러한 자기 성찰 기법에는 재시도(Retry), 키워드(Keywords), 조언(Advice), 설명(Explanation), 지시사항(Instructions), 솔루션(Solution), 종합(Composite), 비축소(Unredacted) 등이 포함됩니다.

- **Performance Highlights**: 자기 성찰을 통한 문제 재시도에서 LLM 에이전트들은 베이스라인(Baseline) 대비 높은 성능 개선을 보였습니다. 특히, '종합' 및 '비축소' 자기 성찰 방법을 사용했을 때 가장 높은 정확도 개선을 확인할 수 있었습니다. 이 연구 결과는 AI 엔지니어링 및 메타인지(metacognition) 연구에 중요한 시사점을 제공합니다.



### Leveraging Lecture Content for Improved Feedback: Explorations with GPT-4 and Retrieval Augmented Generation (https://arxiv.org/abs/2405.06681)
Comments:
          accepted at CSEE&T 2024: 36th International Conference on Software Engineering Education and Training, Würzburg, Germany

- **What's New**: 이 논문은 프로그래밍 작업에 대한 Large Language Model(GPT-4) 피드백을 향상시키기 위해 Retrieval Augmented Generation (RAG)을 사용합니다. 특히, 강의 녹화를 통해 얻은 텍스트 정보를 기반으로 학생들이 독립적 문제 해결을 유도하는 피드백을 생성하며, 강의 내용과 관련된 시간 태그(timestamps)를 메타 정보로 활용하여 학생들에게 즉각적인 비디오 접근을 가능하게 합니다.

- **Technical Details**: GPT-4는 학생의 코드 솔루션, 컴파일러 출력, 유닛 테스트 결과와 함께 강의 노트의 연관된 내용을 RAG를 통해 추가 컨텍스트로 받아 피드백을 생성합니다. 강의 녹음은 SRT(SubRip Text) 형식으로 자막이 포함된 텍스트로 변환되어, 각 텍스트 덩어리에 대한 벡터 표현이 데이터베이스에 저장됩니다. 이 시스템은 단편화된 강의 내용을 RAG를 사용하여 요청된 쿼리에 따라 적절한 강의 내용을 검색하고 결과를 마크다운 형태의 참조와 함께 제공하여 피드백을 구성합니다.

- **Performance Highlights**: 이 시스템을 활용한 평가에서 학생들은 강의 정보와 연계된 피드백(RAG 사용)과 강의 정보를 사용하지 않은 기본 피드백 사이에서 선택할 수 있었습니다. 초기 결과에 따르면, RAG를 사용한 피드백 생성이 일부 상황에서 학생들에게 선호되었으며 학생들이 문제를 독립적으로 해결하는 데 도움이 되었다는 긍정적인 피드백을 받았습니다. 그러나 피드백 생성 시간이 늘어나는 점은 이점이 상황에 따라 다를 수 있음을 시사합니다.



### Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning (https://arxiv.org/abs/2405.06680)
- **What's New**: 이 연구는 대형 언어모델(LLMs)이 수학적 추론에서 합성성(compositionality)을 어떻게 발휘하는지 조사합니다. 특히, MATH와 GSM8k 데이터셋의 문제 설명에 신중하게 설계된 논리 함정(logical traps)을 도입하여 새로운 'MathTrap' 데이터셋을 구축하였습니다. 이는 LLMs가 학습 중 보지 못한 새로운 경우(new cases)에 대처하도록 요구하며, 이를 통해 모델의 합성적 사고 능력을 평가합니다.

- **Technical Details**: MathTrap 데이터셋은 원래의 수학 문제와 함정이 포함된 문제로 구성되어 있으며, 이를 통해 모델이 기존 지식과 함정에 관한 지식을 어떻게 조합하는지를 평가합니다. 또한, 자연어 프롬프트(natural language prompts), 소수 샷 학습(few-shot demonstrations), 그리고 미세조정(fine-tuning)과 같은 여러 가지 방법을 사용하여 LLMs의 성능 향상을 시도하였습니다.

- **Performance Highlights**: 실험 결과, 상용 LLMs는 70% 이상의 정확도로 개념 문제(conceptual problems)를 해결할 수 있음을 보여줍니다. 그러나, 함정 문제(trap problems)에 대해서는 원래 문제에 비해 정확도가 절반 이하로 크게 떨어지는 것을 볼 수 있습니다. 반면, 사람의 경우 함정 문제에서 83.8%의 높은 정확도를 달성하였고, 힌트를 받은 후 정확도는 95.1%로 상승하였습니다. 이는 LLMs가 여전히 인간의 합성적 사고에 비해 뒤떨어진다는 것을 시사합니다.



### ATG: Benchmarking Automated Theorem Generation for Generative Language Models (https://arxiv.org/abs/2405.06677)
- **What's New**: 이 논문은 자동 정리 생성(Automated Theorem Generation, ATG) 벤치마크를 제안하여 언어모델(LMs)이 새로운 정리를 생성하고 이를 하류 추론(theorem proving)에 재사용할 수 있는 능력을 평가합니다. 현재의 언어 모델들이 새로운 정리를 개발하고 재사용하는 기능은 아직 충분히 탐구되지 않았으며, 이 논문은 그 능력을 발전시키기 위한 새로운 도전과제를 제시합니다.

- **Technical Details**: 연구팀은 Metamath 라이브러리를 기반으로 ATG 벤치마크를 구축하며, 이는 공리(axioms), 라이브러리, 문제 세트(problem sets)로 구분됩니다. 이러한 분할은 정리의 증명 깊이에 따라 이뤄졌습니다. 또한, Monte Carlo 나무 탐색 기법(Monte Carlo Tree Search), 자기 대국 학습법(self-play policy learning)과 같은 방법을 결합하여 유용한 정리를 생성하는데 활용됩니다.

- **Performance Highlights**: 생성된 정리들을 사용하여 Holophrasm과 GPT-f 모델의 성능을 각각 16.16%, 7.72% 향상시켰습니다. 이는 생성된 정리가 모델에 유용한 데이터로 활용되어, 추론 과정에서의 효율성과 정확성을 증가시키는 결과를 가져왔음을 보여줍니다.



### EDA Corpus: A Large Language Model Dataset for Enhanced Interaction with OpenROAD (https://arxiv.org/abs/2405.06676)
Comments:
          Under review at Workshop on LLM-Aided Design (LAD'24)

- **What's New**: 이 논문은 오픈소스 EDA(전자 설계 자동화) 툴체인인 OpenROAD를 위해 맞춤형으로 제작된 오픈소스 데이터 세트를 소개합니다. 이 데이터 세트는 LLM(Large Language Models)의 칩 설계 과정에의 통합을 촉진하고자 제공되며, 1000개 이상의 데이터 포인트로 구성되어 있습니다. 데이터는 질문 프롬프트와 프로즈(prose) 답변이 있는 짝(pairwise) 형태와 코드 프롬프트 및 해당 OpenROAD 스크립트가 있는 짝 형태로 구성됩니다.

- **Technical Details**: 이 데이터 세트는 크게 두 부분으로 구성되어 있습니다: (i) OpenROAD에 대한 질문과 해당하는 프로즈 답변을 포함한 질문-답변 쌍, (ii) 실행 동작을 요구하는 프로즈 요청과 해당하는 OpenROAD API를 사용하는 파이썬(Python) 스크립트를 포함한 프롬프트-스크립트 쌍. 이러한 구조는 LLM을 사용하여 칩 설계에 대한 질문에 대답하거나, 사용자의 의도를 실제 물리적 설계 작업으로 변환하는 스크립트를 생성하는 역량을 향상시킬 수 있습니다.

- **Performance Highlights**: ChatGPT를 EDA Corpus로 파인 튜닝(fine-tuning)함으로써, 기존의 LLM보다 개선된 성능을 보였습니다. 또한, 이는 물리적 설계 작업을 위한 LLM 훈련을 주도할 수 있는 첫 번째 공개적으로 이용 가능하며 허가받은(permissively licensed) 데이터 세트로, 새로운 사용자와 경험 많은 칩 설계자 모두의 접근성을 획기적으로 증가시킬 수 있는 가능성을 제공합니다.



### Open-SQL Framework: Enhancing Text-to-SQL on Open-source Large Language Models (https://arxiv.org/abs/2405.06674)
- **What's New**: 이 연구에서는 오픈 소스 대규모 언어 모델(Large Language Models, LLMs)을 사용하여 Text-to-SQL 문제를 해결하기 위한 새로운 방법론인 Open-SQL을 소개합니다. 기존의 연구들과 달리, 이 방법론은 오픈 소스 LLM의 문맥 이해와 응답의 일관성 문제를 개선하기 위해 시스템적인 접근법을 제공합니다. 특히, Open Prompt, Chain-of-Thought (COT), 그리고 Open Example Curation과 같은 새로운 기술을 통해 성능을 크게 향상시켰습니다.

- **Technical Details**: Open-SQL 방법론은 오픈 소스 LLM을 위한 몇 가지 주요 전략을 도입했습니다. 첫 번째는 Open Prompt 전략을 통해 질문 표현을 최적화하는 것이며, 두 번째는 Chain-of-Thought를 사용하여 단계적 추론을 개선하는 것입니다. 또한, 적은 양의 학습 데이터로도 효과적인 학습이 가능하도록 Open Example Curation 방법을 사용합니다. 이외에도 대규모 데이터베이스의 토큰 효율성을 높이기 위해 Variable-length Open DB Schema, Target Column Truncation, Example Column Truncation 등의 기술을 개발하였습니다.

- **Performance Highlights**: 제안된 방법론은 Llama2-7B의 성능을 2.54%에서 41.04%로, Code Llama-7B는 14.54%에서 48.24%로 대폭 향상시켰습니다. 특히, Code Llama-7B는 GPT-4 (46.35%)를 능가하는 성능을 보여, 오픈 소스 LLM 중에서도 뛰어난 결과를 나타냈습니다. 이러한 성과는 Open-SQL 방법론의 유효성을 입증하며, 오픈 소스 환경에서의 Text-to-SQL 작업에 큰 진전을 가져왔습니다.



### Overview of the EHRSQL 2024 Shared Task on Reliable Text-to-SQL Modeling on Electronic Health Records (https://arxiv.org/abs/2405.06673)
Comments:
          The 6th Clinical Natural Language Processing Workshop at NAACL 2024; Minor Change from Camera-Ready

- **What's New**: EHRSQL 2024 공동 작업은 한국의 대학 병원에서 200명 이상의 전문가가 참여한 설문 결과에 기초하여 질문 템플릿을 수집하고, 이를 활용하여 더욱 다양하고 현실적인 질문-SQL 쌍을 생성하여, 병원 시스템에서 EHR을 통해 더 자유롭게 환자 데이터를 탐색할 수 있게 하는 신뢰할 수 있는 텍스트-투-SQL(text-to-SQL) 모델링 시스템을 개발하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 공동 작업은 텍스트-투-SQL 모델링을 통해 EHR 질의응답(QA) 시스템을 구축하는 것을 목표로 합니다. 이 모델들은 자연어 질문을 해당 SQL 쿼리로 자동 전환하고, 이 쿼리들을 데이터베이스에서 실행하여 최종 답을 얻습니다. 또한, 공동 작업은 미믹-IV(MIMIC-IV) 데모 버전을 사용하여 SQL 쿼리를 구성하고, ChatGPT와 같은 대규모 언어 모델(LLM)을 활용하여 더 자연스럽고 대화형에 가까운 질문 문장을 생성하여 질문 템플릿의 질을 향상시키고 있습니다.

- **Performance Highlights**: EHRSQL 2024 공동 작업을 통해 참여팀들은 효과적인 방법을 다양하게 제시하며 이 과제를 해결할 수 있음을 증명하였습니다. 또한, 신뢰성 있는 텍스트-투-SQL 모델은 단순히 정확한 SQL 쿼리를 생성하는 것뿐 아니라, 잘못되었거나 답을 구할 수 없는 질문에 대해서는 답변을 회피함으로써 잠재적인 피해를 최소화할 수 있어야 합니다. 이러한 모델들을 통해 건강관리 전문가들이 EHR 데이터를 더 효과적으로 활용할 수 있는 새로운 가능성을 제시하고 있습니다.



### Parameter-Efficient Instruction Tuning of Large Language Models For Extreme Financial Numeral Labelling (https://arxiv.org/abs/2405.06671)
- **What's New**: 새로운 연구에서는 금융 문서에서 해당하는 XBRL 태그와 함께 관련된 숫자(GAAP 메트릭)를 자동으로 주석 처리하는 문제를 연구하고 있습니다. 이 연구는 대규모 언어 모델(LLMs)의 지시 튜닝을 통해 이 극단적 분류(extreme classification) 문제를 해결할 가능성을 탐구하며, 이전 연구와는 다른 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 메트릭 메타데이터 정보를 활용하여 타겟 출력을 구성하면서, LoRA를 사용한 파라미터 효율적인 솔루션을 제안합니다. 지시 튜닝(instruction tuning)된 FLAN-T5 모델을 사용하여 XBRL 태그 문서화를 생성하고, 비지도 Tag Matcher 모듈을 사용하여 최종 XBRL 태그를 예측합니다.

- **Performance Highlights**: 제안된 모델 FLAN-FinXC는 두 금융 수치 라벨링 데이터셋에서 새로운 최고 성능을 달성했습니다. 특히, FLAN-T5-Large 모델은 LoRA와 함께 사용되었을 때, 기존의 AttentionXML 모델보다 39.3%의 Macro-F1 스코어와 17.2%의 Hits@1 스코어 향상을 이루었습니다. 또한, 학습 중에 보지 못한 태그에 대해서도 높은 제로샷(zero-shot) 성능을 보였습니다.



### Exposing and Explaining Fake News On-the-Fly (https://arxiv.org/abs/2405.06668)
- **What's New**: 본 논문에서는 페이크 뉴스를 실시간으로 감지하고 분류하는 설명 가능하고 실시간 처리 경로를 사용하는 새로운 방법을 소개하고 있습니다. 하이브리드 머신 러닝(Machine Learning, ML) 접근법을 이용해 사용자 창작, 콘텐츠 및 맥락 기반 기능을 이용합니다. 이러한 점이 다른 연구와 차별화되는 주요 특징입니다.

- **Technical Details**: 이 방법은 비지도 학습(unsupervised learning) 및 감독 학습(supervised learning)을 결합하여 온라인에서 생성된 사전을 활용합니다. 자연어 처리(Natural Language Processing, NLP) 기술을 사용하여 프로파일을 구축하고 스트림 머신 러닝(stream Machine Learning, ML) 알고리즘을 사용하여 각 데이터 클러스터의 본질을 실시간으로 분류합니다. 또한, 분류 결과에 대한 설명을 대시보드에 시각적 및 자연어로 표시하는 설명 메커니즘을 포함합니다.

- **Performance Highlights**: 이 연구는 트위터(Twitter)를 포함한 실제 데이터셋에서 검증되었으며, 80%의 정확도와 매크로 F-측정법(macro F-measure)을 달성하였습니다. 이는 실시간 페이크 뉴스 감지 및 실시간 모델링 측면에서 높은 신뢰도와 투명성을 제공하며, 소셜 미디어 콘텐츠의 질과 신뢰성을 향상시키는 데 기여할 수 있습니다.



### Sentiment Polarity Analysis of Bangla Food Reviews Using Machine and Deep Learning Algorithms (https://arxiv.org/abs/2405.06667)
- **What's New**: 이 연구에서는 방글라데시의 인터넷 사용자들이 온라인으로 음식을 주문할 때 남기는 리뷰를 분석하여 음식의 품질을 평가하는 새로운 모델을 제안합니다. 특히, 방글라어(Bengali)로 작성된 리뷰 데이터를 사용하여 긍정적 및 부정적 리뷰의 비율을 자동으로 감지하고 표시할 수 있는 기계 학습(ML)과 자연어 처리(NLP) 아키텍처를 개발했습니다.

- **Technical Details**: 이 연구는 방글라어 음식 리뷰 1484건을 수집하여 데이터셋을 구성하고, 불필요한 불용어 제거, 토큰화, 어간 추출을 포함한 데이터 전처리 과정을 거쳤습니다. 텍스트를 특징 벡터로 변환하기 위해 CountVectorizer, TF-IDF (Term Frequency-Inverse Document Frequency) 벡터라이저 및 N-gram 방법을 사용하였고, Logistic Regression, Random Forest, Linear SVM, Naïve Bayes, Decision Tree, LSTM과 같은 여러 머신 러닝 및 딥 러닝 기술을 평가하였습니다.

- **Performance Highlights**: 로지스틱 회귀(Logistic Regression) 모델이 다른 알고리즘들을 초월하여 약 90.91%의 높은 정확도를 달성했습니다. 이는 온라인 음식 리뷰 분석을 통해 음식 품질을 예측하는 데 매우 유용한 결과를 제공합니다. 추후 이 모델은 식당 사업자들이 고객 리뷰를 정확하게 분석하고 개선할 수 있는 기초 데이터로 사용될 수 있습니다.



### Exploring Social Media Posts for Depression Identification: A Study on Reddit Datas (https://arxiv.org/abs/2405.06656)
Comments:
          Accepted as a poster in IndiaHCI 2023

- **What's New**: 이 연구에서는 Reddit 게시물을 활용하여 개인의 우울증을 식별할 수 있는 가능성을 조사했습니다. 특히 Reddit의 'depression'과 'happy' 서브레딧에서 데이터를 수집하고 UMLS 메타테사우루스(UMLS Metathesaurus)를 사용하여 '우울한(depressive)' 및 '비우울한(non-depressive)' 게시물로 분류했습니다.

- **Technical Details**: 데이터 전처리 과정에서 토큰화(tokenization), 품사 태깅(POS tagging), 불용어 제거, 및 레마타이제이션(lemmatization)을 적용했습니다. 통계적 표현으로 데이터를 변환하기 위해 'Bag Of Words' (BOW) 모델을 사용하였고, 머신러닝 분류에서는 로지스틱 회귀(Logistic Regression), 나이브 베이즈(Naive Bayes), 서포트 벡터 머신(Support Vector Machine, SVM), 그리고 랜덤 포레스트(Random Forest) 분류기를 테스트하였습니다.

- **Performance Highlights**: 실험 결과 랜덤 포레스트 모델이 92.28%의 높은 정확도로 가장 뛰어난 성능을 보였습니다. 이는 모델이 복잡한 데이터 관계와 독특한 특징 선택 방법을 잘 포착하기 때문입니다. 그러나 사용된 데이터 세트의 크기가 작아 잘못된 클래스 예측이 발생할 수 있습니다.



### Large Language Model (LLM) AI text generation detection based on transformer deep learning algorithm (https://arxiv.org/abs/2405.06652)
Comments:
          6 pages

- **What's New**: 이 논문에서는 트랜스포머(Transformer) 모델을 기반으로 하는 인공지능(AI) 텍스트 생성 감지 도구가 개발되었습니다. 이 도구는 AI 텍스트 생성 감지의 정확성을 향상시키고 후속 연구에 참고 자료를 제공하는 것을 목표로 합니다.

- **Technical Details**: 텍스트는 유니코드 정규화되고 소문자로 변환된 후, 정규 표현식을 사용하여 알파벳이 아닌 문자와 구두점을 제외한 모든 문자가 제거됩니다. 구두점 주변에는 공백이 추가되고, 첫 번째와 마지막 공백은 제거됩니다. 연속적인 줄임표는 단일 공간으로 대체되고, 지정된 구분자를 사용하여 텍스트가 연결됩니다. 머신 러닝 모델은 LSTM, Transformer, CNN과 같은 레이어를 결합하여 텍스트 분류(text classification)나 시퀀스 라벨링(sequence labelling) 작업에 사용됩니다.

- **Performance Highlights**: 모델은 학습 및 검증 세트에서 손실(loss)이 0.127에서 0.005로 감소하고, 정확도(accuracy)가 94.96%에서 99.8%로 증가하여 AI 생성 텍스트의 감지 및 분류 능력이 우수함을 보여줍니다. 테스트 세트의 혼동 행렬(confusion matrix)과 정확도는 모델이 AI 생성 텍스트에 대해 99%의 예측 정확도와 0.99의 정밀도(precision), 1의 재현율(recall), 그리고 0.99의 f1 스코어(f1 score)를 달성하여 매우 높은 분류 정확도를 달성했다고 보고합니다.



### Large Language Models as Planning Domain Generators (https://arxiv.org/abs/2405.06650)
Comments:
          Published at ICAPS 2024

- **What's New**: 이 논문에서는 텍스트 설명에서 계획 도메인 모델(Planning Domain Models)을 생성하기 위해 대규모 언어 모델(LLMs)의 사용 가능성을 탐구합니다. 연구팀은 특정한 평가 프레임워크를 도입하고 여러 LLM을 실험적으로 분석하여 이 과정을 자동화하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 자연어 설명에서 계획 도메인을 PDDL(Planning Domain Description Language)로 자동 변환하는 LLM의 능력을 평가합니다. 연구팀은 PDDL 도메인 구성의 정확성을 평가하기 위해 설정된 ground truth를 기반으로 한 평가 메트릭스(Evaluation Metrics)를 정의하고, 이를 사용하여 여러 LLM의 성능을 비교 분석합니다. 또한, 도메인 생성에 영향을 미치는 자연어 설명의 정보 포함 정도를 조사합니다.

- **Performance Highlights**: 분석 결과, 파라미터 수가 많은 LLM이 자연어 설명에서 정확한 계획 도메인을 생성하는 데 중간 수준의 숙련도를 보이는 것으로 나타났습니다. 특히 코딩 및 대화 모델을 포함한 7개의 대규모 언어 모델을 9개의 다양한 계획 도메인에서 평가하였습니다. 실험을 통해 이러한 모델들이 계획 도메인 생성 과정에서 어느 정도의 정확도를 보여줄 수 있는지를 구체적으로 분석하였습니다.



### Levels of AI Agents: from Rules to Large Language Models (https://arxiv.org/abs/2405.06643)
- **What's New**: 이 연구는 자동차 운전의 여섯 단계를 모델로 하여 인공지능(AI) 에이전트를 분류하는 새로운 체계를 제안합니다. 각 단계는 에이전트의 유틸리티와 강도를 기반으로 하며, 특정 기능과 능력의 진화를 구분짓습니다.

- **Technical Details**: 제안된 AI 에이전트의 분류는 다음과 같습니다: L0 (No AI)는 인식(perception)과 행동(action)을 고려하는 도구를 사용하며, L1은 규칙 기반(rule-based) AI를 사용하고, L2는 인공지능 학습(IL)/강화 학습(RL) 기반 AI로 규칙 기반을 대체하고 추론 및 의사결정 기능을 추가합니다. L3은 인공지능 학습/강화 학습 기반 AI를 대체하여 대규모 언어 모델(LLM-based) AI를 적용하고 메모리와 반성 기능을 설정합니다. L4는 자율 학습(autonomous learning)과 일반화(generalization)를 촉진하고, L5는 감정과 성격, 다중 에이전트와의 협업 행동을 추가합니다.

- **Performance Highlights**: 각 단계의 분류는 AI 에이전트가 복잡한 환경에서 보다 효과적으로 기능할 수 있게 하며, 특히 L5 단계에서는 인간과 유사한 협업과 상호작용이 가능해집니다. 이러한 분류는 미래의 AI 시스템 설계와 평가에 중요한 기준을 제공할 것입니다.



### AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments (https://arxiv.org/abs/2405.07960)
- **What's New**: AgentClinic은 임상 환경을 모의하는 다중 모달 벤치마크로서, 대화 및 활동 데이터 수집을 통해 의사 에이전트가 환자의 진단을 밝혀내야 하는 시뮬레이션을 제공합니다. 기존의 의료 AI 평가 방식이 단순한 의료 질문 답변 벤치마크에 국한되었던 것에 비해, AgentClinic은 상호 작용을 중시하는 새로운 접근 방식을 도입함으로써 실제 임상 작업에 필요한 인터랙티브한 의사결정 능력을 평가할 수 있습니다.

- **Technical Details**: AgentClinic은 두 가지 벤치마크, 이미지와 대화를 포함하는 AgentClinic-NEJM과 대화만을 포함하는 AgentClinic-MedQA를 제공합니다. 또한, 의사와 환자 에이전트 모두에게 인지적 및 암묵적 편향을 도입하여 실제적인 상호작용을 모방합니다. 상태 최고의 LLMs (Large Language Models)을 이용한 평가에서, 일부 모델은 기존 벤치마크에서 우수한 성능을 보였지만 AgentClinic에서는 상대적으로 낮은 성능을 보였습니다.

- **Performance Highlights**: 편향 도입은 의사 에이전트의 진단 정확도를 크게 감소시켰고, 환자 에이전트의 준수도, 자신감, 후속 상담 의사를 저하시켰습니다. 특히, 환자 에이전트를 구동하는 LLM의 선택이 벤치마크에서의 성능에 중요한 요인으로 작용했습니다.



### RLHF Workflow: From Reward Modeling to Online RLHF (https://arxiv.org/abs/2405.07863)
Comments:
          26 pages, 8 figures

- **What's New**: 이 기술 보고서에서는 인간의 피드백에서 온라인 반복 강화 학습(Online Iterative Reinforcement Learning from Human Feedback, RLHF)의 워크플로우를 소개하고 있습니다. 이 방법은 최근 대형 언어 모델(Large Language Model, LLM) 문헌에서 그 오프라인 버전을 크게 앞지르는 것으로 알려져 있습니다. 오픈 소스 RLHF 프로젝트는 주로 오프라인 학습 환경에 국한되어 있지만, 본 보고서는 온라인 반복 RLHF를 쉽게 재현할 수 있는 자세한 가이드를 제공하고자 합니다.

- **Technical Details**: 우선, 오픈 소스 커뮤니티에서 온라인 인간 피드백이 대부분 불가능하기 때문에, 다양한 오픈 소스 데이터셋을 활용해 선호 모델(Preference Models)을 구축하고, 이를 인간 피드백의 대용 모델로 사용합니다. 실제 온라인 반복 RLHF의 이론적 통찰과 알고리즘 원칙을 논의한 후, 자세한 실용적 구현을 다룹니다.

- **Performance Highlights**: 개발된 LLM, SFR-Iterative-DPO-LLaMA-3-8B-R은 LLM 챗봇 벤치마크인 AlpacaEval-2, Arena-Hard 및 MT-Bench는 물론 HumanEval과 TruthfulQA와 같은 다른 학술 벤치마크에서 인상적인 성능을 달성하였습니다. 감독된 미세 조정(Supervised Fine-Tuning, SFT)과 반복 RLHF는 완전히 오픈 소스 데이터셋을 사용하여 최첨단 성능을 얻을 수 있음을 보여주었습니다.



### Open-vocabulary Auditory Neural Decoding Using fMRI-prompted LLM (https://arxiv.org/abs/2405.07840)
- **What's New**: 이 논문에서는 새로운 기법인 '브레인 프롬프트 GPT (Brain Prompt GPT, BP-GPT)'를 소개합니다. 이 방법은 fMRI(fMRI)에서 추출된 뇌 표현을 프롬프트로 활용하여 GPT-2를 사용, 자극 텍스트(stimulus text)로 fMRI 신호를 디코딩할 수 있습니다. BP-GPT는 오픈 소스 청각 의미 디코딩 데이터셋에서 기존 방법보다 높은 성능을 보여줍니다.

- **Technical Details**: BP-GPT는 미리 학습된 대형 언어 모델 (Pre-trained Large Language Model, LLM)인 GPT-2를 사용하여 fMRI의 낮은 시간 해상도를 보완합니다. 또한, 텍스트-텍스트 기준선(text-to-text baseline)을 도입하여 fMRI 프롬프트와 텍스트 프롬프트를 정렬시키는 대조 손실(contrastive loss)을 사용하여 모달 차이의 영향을 줄입니다. 이로써 디코딩 성능이 향상됩니다.

- **Performance Highlights**: BP-GPT 모델은 모든 대상자에 걸쳐서 METEOR에서 최대 4.61%, BERTScore에서 2.43%의 성능 향상을 달성합니다. 이러한 실험 결과는 뇌 표현을 프롬프트로 사용하는 것이 청각 신경 디코딩에 효과적임을 보여줍니다.



### FastSAG: Towards Fast Non-Autoregressive Singing Accompaniment Generation (https://arxiv.org/abs/2405.07682)
Comments:
          IJCAI 2024

- **What's New**: 이 연구에서는 SingSong처럼 자동회귀(AR) 방식에 의존하는 대신, 비자동회귀(non-AR) 확산 기반 기법을 사용하여 노래 반주 생성(Singing Accompaniment Generation, SAG)의 속도와 품질을 향상시킨 FastSAG 방법을 제안합니다. 이 새로운 접근법은 멜 스펙트로그램(Mel spectrogram)을 직접 생성함으로써 생성 과정을 크게 단순화하고 속도를 향상시킬 수 있습니다.

- **Technical Details**: FastSAG는 보컬 신호에서 유추된 조건을 신중하게 설계함으로써 타겟 반주의 멜 스펙트로그램을 직접 생성하는 확산 기반 프레임워크를 개발합니다. 추가적으로, 의미적 정렬(semantic alignment)을 위한 '의미적 투영 블록'(semantic projection block)과 프레임 레벨 조정 및 제어를 강화하는 '우선 투영 블록'(prior projection block)을 디자인하고, 보컬 신호와의 의미적 및 리듬적 일치성을 보장하기 위한 손실 함수(loss functions) 집합을 설계했습니다.

- **Performance Highlights**: FastSAG는 기존 SingSong 방법보다 적어도 30배 빠른 속도로 반주를 생성할 수 있으며, 실시간으로 생성이 가능한 수준까지 성능을 향상시켰습니다. 실험 결과에 따르면 FastSAG는 SingSong보다 우수한 샘플을 생성하며, 음성과 반주 사이의 의미적 및 리듬적 일치성에서도 높은 성능을 보여줍니다.



### Constructing a BPE Tokenization DFA (https://arxiv.org/abs/2405.07671)
- **What's New**: 이 논문은 인기 있는 바이트 페어 인코딩(Byte Pair Encoding, BPE) 기법에 의해 생성된 토큰화를 직접 조작하는 결정적 유한 오토마타(Deterministic Finite Automata, DFA)의 효율적인 구축 방법에 대해 소개하고 분석합니다. 이를 통해 토큰화된 경우에도 패턴 매칭, 토큰화 사전의 동등성 검사, 다양한 방식으로 토큰화된 언어의 합성과 같은 기존 기술과 알고리즘을 적용할 수 있게 됩니다.

- **Technical Details**: BPE는 데이터 압축에 기반을 둔 테크닉으로, 가장 흔한 쌍을 기반으로 규칙 사전을 구축하며 연속적으로 인접 토큰 쌍을 병합합니다. UFC(Universal Function Call) 기반 전처리 등을 위해 다양한 구현에서 우선순위 큐를 사용합니다. 하지만, 같은 문자열을 여러 토큰 시퀀스로 나타낼 수 있는 여러 경우의 수를 갖고 있기 때문에, 올바른 토큰화만이 고려됩니다. 또한, 특정 부분 문자열은 그 맥락에 따라 다르게 토큰화될 수 있어 패턴 매칭을 복잡하게 만듭니다.

- **Performance Highlights**: 이 논문에서 고려된 유한 오토마타는 실질적으로 매우 중요한 의미를 가집니다. 토큰 시퀀스가 언어 서비스와의 커뮤니케이션에서 이미 흔히 사용되고 있으며, 이러한 서비스가 점점 인기를 끌면서 더 많은 텍스트가 이러한 방식으로 인코딩될 수 있습니다. 특정 사용 사례로는 네트워크상의 필터링이나 안전 시스템을 위한 패턴 매칭, 토큰화의 정확성 검증(잘못된 토큰화는 모델을 혼동시킬 수 있음), 그리고 토큰화된 텍스트를 직접 토큰화하거나 재작성하는 것이 포함될 수 있습니다.



### Backdoor Removal for Generative Large Language Models (https://arxiv.org/abs/2405.07667)
- **What's New**: 이 논문에서는 생성형 대규모 언어 모델(Large Language Models, LLMs)의 백도어 매핑을 제거하는 새로운 접근 방식인 Simulate and Eliminate (SANDE)을 제안합니다. SANDE는 Overwrite Supervised Fine-tuning (OSFT)를 사용하여 알려진 트리거가 있는 경우 백도어 행동을 제거하는 방법과 알려지지 않은 트리거 패턴을 처리하기 위해 두 단계 프레임워크로 구성됩니다.

- **Technical Details**: 첫 번째 단계에서는 트리거의 행동을 모방하기 위해 파랏트 프롬프트 학습(parrot prompt learning)을 사용합니다. 이후 OSFT를 이용하여 파랏트 프롬프트를 튜닝하고, 모델이 백도어 트리거로부터 부정적인 반응을 삭제할 수 있도록 합니다. 두 번째 단계에서는 트리거 패턴과 트리거된 반응에 대한 정보가 없는 가장 일반적인 시나리오에 대해 백도어 제거를 확장합니다.

- **Performance Highlights**: SANDE는 트리거 제거에서 기존의 백도어 방어 전략을 넘어서며, 백도어 LLMs에 직접 적용하여 추가적인 비용 없이 기저 상태로 복원할 수 있는 간단하면서도 효과적인 방법을 제공합니다. 실험을 통해 SANDE가 백도어 제거와 유틸리티(Utility) 영향 모두에서 효과적임을 입증했습니다. 제안된 OSFT와 SANDE는 기존 기준(Baseline)과 비교할 때 LLM의 유용성에 최소한의 해를 끼치면서 강력한 백도어 제거 능력을 보여줍니다.



### Sign Stitching: A Novel Approach to Sign Language Production (https://arxiv.org/abs/2405.07663)
Comments:
          18 pages, 3 figures, 4 tables

- **What's New**: 이 논문에서는 수화 생산(Sign Language Production, SLP)을 위한 새로운 접근 방식을 제안하며, 사전 예제들을 효과적으로 이어 붙여 연속적이고 의미 있는 수화 시퀀스를 생성합니다. 특히, 수화의 리듬과 강세를 모방하여 자연스러운 수화 시퀀스를 생성하는 데 중점을 두었습니다. 이를 위해 NSVQ (Noise Substitution Vector Quantization) 트랜스포머 아키텍처를 사용하여 표정 사전을 학습하고 각 수화에 적용합니다.

- **Technical Details**: 이 연구는 각 수화를 정규화하고 연속적인 시퀀스를 생성하기 위해 절단 및 이어 붙이기 작업, 주파수 도메인에서의 필터링, 예측된 지속 시간에 따른 재샘플링을 포함한 7단계 접근 방식을 제시합니다. 또한, SignGAN 모델을 활용하여 출력을 사실적인 수화 영상으로 매핑하며, 텍스트에서 수화로의 전체 SLP(Text-to-Sign, T2S) 파이프라인을 소개합니다. 모든 데이터셋에서 최고 수준의 성능을 보여주는 것으로 평가되었습니다.

- **Performance Highlights**: 제시된 접근 방식은 모든 데이터셋에서 최고의 성능을 보여주었으며, 사용자 평가에서도 기존 모델을 능가하며, 수화 시퀀스의 사실성을 향상시키는 것으로 나타났습니다. 실제 사용자 평가에서도 접근 방식이 기준 모델을 초과하는 것으로 평가되었습니다.



### T-curator: a trust based curation tool for LOD logs (https://arxiv.org/abs/2405.07081)
- **What's New**: 이 논문에서는 Linked Open Data (LOD) 로그를 신뢰 기반으로 큐레이션(정화)할 수 있는 새로운 툴 'T-Curator'를 제안합니다. 이 툴은 사용자들이 LOD 환경 전체의 지식과 로그 큐레이션을 위한 도구들을 제공받을 때 LOD 로그를 안전하고 신뢰성 있게 사용할 수 있도록 지원합니다.

- **Technical Details**: T-Curator는 SPARQL 쿼리 로그의 구조, 품질, 그리고 출처를 분석하는 로그 프로파일링과 신뢰 맥락에 적합한 ETL(Extract-Transform-Load) 연산자들을 정의하는 두 단계로 구성되어 있습니다. 이 연산자들은 신뢰할 수 있는 쿼리만을 유지하면서 LOD 로그를 큐레이션하는 신뢰 기반 큐레이션 파이프라인을 구축하는 데 사용됩니다.

- **Performance Highlights**: 이 도구는 데이터 분석가와 데이터 과학자들이 어떻게 LOD 로그를 큐레이션할지 결정하는 데 도움을 줄 수 있습니다. T-Curator는 LOD 로그의 신뢰성을 강화하고, 신뢰할 수 있는 데이터 소스로서의 가치를 높이기 위한 효과적인 방법을 제공합니다.



### Deciphering public attention to geoengineering and climate issues using machine learning and dynamic analysis (https://arxiv.org/abs/2405.07010)
Comments:
          46 page, 6 main figures and SI

- **What's New**: 이 연구는 지구공학(Geoengineering)에 대한 대중의 관심과 태도를 조사하기 위한 데이터 기반 접근 방식을 적용하였습니다. 특히 기후 변화와 관련된 뉴스의 언론 보도가 지구공학에 대한 대중의 관심과 태도에 어떤 영향을 미치는지 분석하였습니다.

- **Technical Details**: 이 연구에서는 BBC와 New York Times로부터 수집된 30,773개의 뉴스 기사와 2018년부터 2022년까지의 Google Trends 데이터를 사용하였습니다. BERT 기반의 주제 모델링(Topic Modeling), 감정 분석(Sentiment Analysis), 그리고 시계열 회귀 분석(Time-series Regression)을 통해 어떤 뉴스 주제가 지구공학에 대한 대중의 관심을 높이는지 탐색하였습니다.

- **Performance Highlights**: 분석 결과, 에너지 관련 긍정적인 뉴스는 지구공학에 대한 공개적인 관심을 높이는 좋은 예측 지표로 작용한다는 것을 발견하였습니다. 이러한 경향은 시간이 지남에 따라 일관되게 나타났습니다.



### Automating Thematic Analysis: How LLMs Analyse Controversial Topics (https://arxiv.org/abs/2405.06919)
Comments:
          18 pages, 6 figures

- **What's New**: 본 논문에서는 논쟁적인 주제에 대한 테마 분석을 지원하기 위해 LLM(Large Language Models)을 활용하는 가능성을 탐구합니다. 특히, 호주의 로보데뷔스(Robodebt) 스캔들에 대한 미디어 보도를 인용하여 인간 연구자들과 두 가지 LLM, GPT-4와 Llama 2를 비교 분석합니다. 연구 결과는 인간과 기계 에이전트 간의 테마 분류에서 흥미로운 중복과 차이점을 강조하며, LLM이 담론 및 테마 분석을 지원하는 데 효과적일 수 있는 분야를 제안합니다.

- **Technical Details**: 이 연구는 LLM이 정성적 텍스트 데이터에 테마를 적용할 수 있는 능력에 초점을 맞추고 있으며, 이는 인간의 반복적 해석을 보완할 수 있는 가능성을 시사합니다. 탐색적 대화 구조를 통해 LLM은 비판적 대화, Socratic 대화를 지원하고, 다양한 시각과 전문 지식을 제공할 수 있습니다. 데이터의 시맨틱 분류와 코드 적용은 추론의 과정과 맞닿아 있으며, LLM은 자기주의(self-attention) 메커니즘을 통해 입력 텍스트 시퀀스의 각 단어 간의 관계를 분석하여 출력 확률을 계산합니다. GPT 시리즈가 사용하는 이 메커니즘은 언어 의미의 이해를 시뮬레이션할 수 있는 뛰어난 능력을 보여줍니다.

- **Performance Highlights**: LLM은 고도의 텍스트 처리 능력과 컨텍스트 및 뉘앙스에 대한 민감성을 바탕으로 복잡한 환경이나 주제를 분석할 수 있습니다. 본 연구에서 LLM은 인간 연구자들과 함께 미디어 텍스트를 분석하면서 테마 분류에서 유사한 패턴을 보여주었으며, 이는 LLM이 테마 분석에서 인간 해석을 보완할 수 있다는 가능성을 제시합니다. 또한, 연구자들은 LLM이 해석의 다양한 프레임을 제공함으로써 결과를 삼각 측량하는데 유용할 수 있다고 강조합니다.



### Large Language Model in Financial Regulatory Interpretation (https://arxiv.org/abs/2405.06808)
- **What's New**: 이 연구는 복잡한 금융 규제를 해석하기 위해 대규모 언어 모델(Large Language Models, LLMs)을 사용하는 혁신적 방법을 탐구합니다. 구체적으로는 Basel III 자본 요건 규정과 같은 장황하고 복잡한 규제 텍스트를 간결한 수학적 프레임워크로 요약하고, 이를 실행 가능한 코드로 변환하는 효과적인 프롬프트 설계에 중점을 둡니다. 이는 글로벌 은행 기관의 금융 보고 및 위험 관리 시스템에서 규제 지침의 실행을 간소화하는 것을 목표로 합니다.

- **Technical Details**: LLMs의 성능을 비교하여 다양한 금융 규제 문서를 분석하는 GPT-4의 능력을 평가한 결과, GPT-4는 PDF 파일보다 이미지로 된 문서에서 더 정확한 해석을 보여줍니다. 또한, 복잡한 금융 작업을 위해 문제 해결 과정을 정의하는 프롬프트 엔지니어링(prompt engineering)을 도입하고, 이를 통해 LLM이 정보를 더 정확하게 수집하도록 안내합니다. 설계된 알고리즘은 문서 로딩 방법과 다단계 반복 접근법을 포함하여 금융 규제 문서 분석을 체계화합니다.

- **Performance Highlights**: GPT-4는 다른 모델들보다 뛰어난 성능을 보이며, 수치 시뮬레이션을 통해 Basel III 자본 적정성 요구 사항을 효과적으로 구현할 수 있음을 입증했습니다. 이는 금융 시장 참여자에게 심층적인 시장 통찰력과 예측 분석을 제공하는 데 있어 강력한 인공지능(AI) 도구 개발의 중요성을 강조합니다. 이 연구는 금융 규제 문서 해석에 LLM을 적용함으로써 금융 기관이 운영 및 관리 역량을 향상시키고 준수 워크플로우를 통합하는 데 도움이 될 잠재력을 제시합니다.



### CANAL -- Cyber Activity News Alerting Language Model: Empirical Approach vs. Expensive LLM (https://arxiv.org/abs/2405.06772)
Comments:
          Published in 2024 IEEE 3rd International Conference on AI in Cybersecurity (ICAIC), Conference Date: 07-09 February 2024

- **What's New**: 새로운 연구에서는 사이버 위협 모델링(cyber threat modeling)을 위한 실증적인 프레임워크(empirical framework)를 소개합니다. 이 연구는 특히 뉴스 기사에서 사이버 관련 정보를 파싱하고 분류하는 데에 초점을 맞추고 있으며, 사이버 활동 뉴스 경보 언어 모델(CANAL - Cyber Activity News Alerting Language Model)을 사용하여 시장 이해 관계자들(stakeholders)에게 실시간 경계(real-time vigilance)를 제공합니다. CANAL은 BERT 모델을 기반으로 하여 최적화되었고, Random Forest로 구동되는 새로운 실버 라벨링(silver labeling) 접근 방식을 사용하여 만들어졌습니다.

- **Technical Details**: CANAL은 크고 비용이 많이 드는 대형 언어 모델(LLMs)과 벤치마킹되었습니다. 예를 들면, GPT-4, LLaMA, Zephyr 등과 같은 모델들입니다. 이들 모델 대비하여, CANAL은 사이버 뉴스 분류(cyber news classification)에서의 정확도와 비용 효율성에서 모두 우수한 성능을 보여주었습니다. 또한, 연구팀은 사이버 신호 발견 모듈(Cyber Signal Discovery module)을 도입하여 뉴스 기사에서 급부상하는 사이버 신호들을 효율적으로 감지할 수 있는 전략적 요소를 제공합니다.

- **Performance Highlights**: CANAL은 여러 대형 언어 모델들과의 비교 분석에서 우수한 결과를 보였으며, 특히 정확도와 비용 효율성 면에서 두각을 드러냈습니다. 이는 보다 경제적인 비용으로 동일하거나 더 나은 성능을 제공함으로써 사업체들이 사이버 정보에 더 민첩하게 반응할 수 있게 합니다. 사이버 신호 발견 모듈 또한 신흥 사이버 위협의 감지 능력과 용어 데이터베이스의 강화를 통해 모델의 전체적인 유용성을 증진시키는 주요 기능으로 작용합니다.



### LIVE: LaTex Interactive Visual Editing (https://arxiv.org/abs/2405.06762)
Comments:
          8 pages, double column, ieee

- **What's New**: 이 연구에서는 LaTex (LaTeX) 그래픽 항목에 대한 상호작용 가능한 설계 방법인 LIVE를 제안합니다. LIVE는 전통적인 학술 논문, 특히 리뷰 논문에 더 많은 활기와 성능 요소를 더하는 데 사용할 수 있습니다. 이는 LaTex 학술 논문의 동적이고 상호 작용이 가능한 구성 요소를 디자인하는 새로운 방법을 탐구합니다.

- **Technical Details**: LIVE는 LaTeX를 사용하여 보다 정보적인 그래픽 항목(Gitems)을 설계할 수 있게 하며, 특정 논문 범위의 상호 적용 관계를 자동으로 쉽게 파악할 수 있습니다. 논문은 NeRF (Neural Radiance Fields) 논문을 예로 들어 LIVE의 기능을 설명합니다. 연구는 LaTex 그래픽 항목에 대한 보다 다채롭고 기능적인 구현 방법을 제공하고 다중 논문의 인용 관계를 자동으로 분석하는 방법을 소개하여 상호 작용 구성 요소의 설계 효율성을 높입니다.

- **Performance Highlights**: LIVE를 사용하면 기존의 정적 LaTeX 요소를 넘어서 광범위한 상호작용 및 동적 구성 요소를 구현할 수 있습니다. 이는 학술 논문의 글쓰기와 설계 방식에 혁신을 가져올 수 있으며, 인터랙티브한 그래픽 요소는 독자들에게 보다 풍부하고 직관적인 정보 전달 방식을 제공합니다.



### On the Shape of Brainscores for Large Language Models (LLMs) (https://arxiv.org/abs/2405.06725)
Comments:
          arXiv admin note: text overlap with arXiv:1710.04019, arXiv:2403.13825 by other authors

- **What's New**: 이 연구는 큰 언어 모델(Large Language Models, LLMs)이 부상하면서 새로운 척도인 '브레인스코어(Brainscore)'가 인간 뇌 및 신경 시스템과 LLMs의 기능적 유사성을 평가하는 수단으로 등장했음을 소개합니다. 이 논문은 Brainscore의 의미를 탐구하고 신뢰할 수 있고 유효한 특징을 식별하기 위한 첫 시도로, 통계적 분석과 선형 회귀 모델(Linear Regression Models)을 사용하여 분석을 진행하였습니다.

- **Technical Details**: 연구팀은 190명의 피험자를 대상으로 한 인간의 fMRI 데이터와 39개의 훈련되고 훈련되지 않은 LLMs를 포함하여 상위학적 특징(topological features)을 구축하였습니다. 그 후, 36개의 선형 회귀 모델을 훈련시켜 생성된 특징 중 신뢰할 수 있고 유효한 것들을 파악하였습니다. 이를 통해 뇌 영역(Regions of Interest, ROIs)과 반구별로 Brainscore를 해석하는 데 유익한 특징 조합을 밝혀내었습니다.

- **Performance Highlights**: 연구 결과는 다양한 뇌 영역과 반구에서의 Brainscore를 해석하는 데 기여하는 특징 조합을 밝히며, 수행된 정교한 통계 분석을 통해 이 분야에서의 이해도를 높이는 데 중요한 역할을 하였습니다. 이는 해석 가능한 기계 학습(interpretable machine learning, iML) 연구를 진전시키는 데 중요한 기여를 하였습니다.



### LangCell: Language-Cell Pre-training for Cell Identity Understanding (https://arxiv.org/abs/2405.06708)
Comments:
          27 pages, 21 figures, conference

- **What's New**: LangCell은 세포의 다양한 의미적 측면을 이해하는 새로운 언어-세포(Language-Cell) 사전 학습 프레임워크입니다. 이 모델은 자연어와 단일세포 데이터(single-cell data)를 통합하여 전사체 데이터(transcriptomic data)에서 세포 정체성(cell identity)을 이해하는데 중요한 발전을 이루었습니다.

- **Technical Details**: LangCell은 scRNA-seq 데이터와 자연어 텍스트를 결합한 새로운 표현을 생성하는데 초점을 맞춘 연구입니다. 이 모델은 Masked Gene Modeling (MGM), Cell-Cell Contrastive Learning (C-C), Cell-Text Contrastive Learning (C-T), 및 Cell-Text Matching (CTM)과 같은 다양한 멀티태스크 학습 방법을 사용합니다.

- **Performance Highlights**: LangCell은 제로샷(zero-shot), 퓨샷(few-shot) 및 파인튜닝(fine-tuning) 시나리오에서 기존 모델을 크게 능가했습니다. 특히, 제로샷 시나리오에서의 세포 유형 주석(cell type annotation) 작업에서 우수한 성능을 보여주었으며, 세포 텍스트 검색(cell-text retrieval)과 같은 새로운 작업에서 뛰어난 결과를 제공합니다.



### DrugLLM: Open Large Language Model for Few-shot Molecule Generation (https://arxiv.org/abs/2405.06690)
Comments:
          17 pages, 3 figures

- **What's New**: 새롭게 개발된 DrugLLM은 기존의 LLM과 달리 생물학 및 화학 분야의 언어를 처리하는 데 특화되었습니다. 이 모델은 새로운 대규모 언어 모델(Large Language Model, LLM)로, 약물 설계를 위한 맞춤형 GMR(Group-based Molecular Representation)을 사용하여 분자의 구조와 변형을 효과적으로 학습합니다.

- **Technical Details**: DrugLLM의 핵심 기술은 GMR을 사용하여 분자를 시퀀스로 표현하고, 이를 이용해 분자의 구조적 변형을 예측하는 것입니다. GMR은 분자의 구조적 그룹을 기반으로 하여 분자를 선형 시퀀스로 변환하며, 순환 구조(Cyclic Complexity)와 구조적 민감성(Structural Sensitivity)을 개선하는 방법을 제공합니다. 이러한 접근 방식은 SMILES 표기의 복잡성을 줄이고, 모델이 더 빠르고 정확하게 학습할 수 있게 합니다.

- **Performance Highlights**: DrugLLM은 기존의 분자 생성 모델들과 비교하여, 제한된 데이터로부터 새로운 분자를 예측하는 few-shot generation의 능력이 뛰어나다는 것을 증명합니다. 연구 결과에 따르면 DrugLLM은 천만 개 이상의 분자와 1,000개 이상의 다양한 분자 속성을 포함하는 대규모 훈련 데이터 세트에서 학습하였으며, 적은 수의 예시를 바탕으로 새로운 분자 구조를 예측할 수 있습니다.



### Language Interaction Network for Clinical Trial Approval Estimation (https://arxiv.org/abs/2405.06662)
- **What's New**: 임상 시험 결과 예측은 임상 시험이 목표하는 종료점에 성공적으로 도달할 가능성을 추정하는 과정입니다. 본 연구에서는 기존의 소분자(molecule) 약물에 집중되었던 이전 연구와 달리, 생물학적 제제(biologics)에 초점을 맞추어 새로운 접근법인 Language Interaction Network (LINT)를 소개하고 있습니다. 이는 기존의 그래프 신경망(graph neural networks) 등의 전통적 방법들이 생물학적 데이터의 복잡성으로 인해 도전을 겪는 문제에 대응하기 위함입니다.

- **Technical Details**: LINT는 무료 텍스트(free-text) 임상 시험 설명만을 사용하여 시험 결과를 예측하는 새로운 방법론입니다. 이는 사전 훈련된 언어 모델(pretrained language models, PLM)을 기반으로 하며, 임상 시험의 텍스트 설명, 관련 약물, 해당 의료 코드를 공동으로 고려하여 임상 시험 결과를 예측합니다. LINT는 국제 질병 분류 코드(International Classification of Diseases, ICD) 코드와 텍스트 기능을 활용하여 소분자 약물과 생물학적 제제에서의 개입 시험 승인을 정확하게 예측합니다.

- **Performance Highlights**: LINT는 I, II, III 단계 임상 시험에서 각각 0.770, 0.740, 0.748의 ROC-AUC 점수를 달성하여, 기존 모델들을 뛰어넘는 성과를 보였습니다. 특히, 생물학적(바이오로직) 개입을 포함한 시험에서 뛰어난 성능을 보여주었습니다. 이러한 결과는 통계적으로 유의한 결과(statistically significant results)를 제공하며, 임상 시험의 성공 가능성을 높여주는 중요한 기술적 발전을 대변합니다.




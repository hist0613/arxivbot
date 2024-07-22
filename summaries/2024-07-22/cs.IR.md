New uploads on arXiv(cs.CL)

### Internal Consistency and Self-Feedback in Large Language Models: A Survey (https://arxiv.org/abs/2407.14507)
Comments:
          27 pages, 9 figures, 10 tables, 14 equations

- **What's New**: 최근 자연어 처리(NLP) 분야에서 대형 언어 모델(LLMs)이 상당한 발전을 이루었지만, 여전히 일관성 없는 응답이나 비논리적 추론, 허위 내용 등을 생성하는 문제를 가지고 있습니다. 이 논문에서는 이러한 문제를 해결하기 위해 여러 'Self-' 연구들이 수행되고 있으며, 이 연구들은 LLMs가 자체적으로 평가하고 업데이트하는 과정에 초점을 맞추고 있다고 설명합니다. 논문에서는 내부 일관성(Internal Consistency)이라는 이론적 프레임워크를 제시하며, 이를 통해 모델의 비논리적 응답이나 허위 내용을 설명합니다.

- **Technical Details**: 내부 일관성 프레임워크는 LLMs의 잠재 레이어(latent layer), 디코딩 레이어(decoding layer), 응답 레이어(response layer) 간의 일관성을 평가하는 방법입니다. 이 이론을 확장하여 Self-Feedback이라는 새로운 프레임워크를 제안하며, 이는 Self-Evaluation과 Self-Update의 두 가지 모듈로 구성되어 있습니다. Self-Evaluation은 모델이 자체적으로 응답을 평가하고, Self-Update는 평가 결과를 바탕으로 모델을 업데이트하는 과정입니다.

- **Performance Highlights**: Self-Feedback 프레임워크를 통해 여러 연구에서 모델의 추론 능력을 확장하고 허위 내용 생성을 줄일 수 있음이 확인되었습니다. 논문에서는 '일관성은 거의 정확성이다', '암묵적 추론과 명시적 추론의 역설' 등의 가설을 제시하며, 향후 연구의 방향도 제시하고 있습니다. 실험 코드와 참고 문서들은 오픈소스로 공개되어 있어, 연구자들이 쉽게 접근할 수 있도록 했습니다.



### Evaluating the Reliability of Self-Explanations in Large Language Models (https://arxiv.org/abs/2407.14487)
Comments:
          Not peer-reviewed. Under review at Discovery Science 2024

- **What's New**: 이 논문에서는 대형 언어 모델(large language models, LLMs)이 자신의 출력을 설명할 때 생성하는 설명의 신뢰성을 조사합니다. 2B에서 8B 파라미터를 가진 최신 LLM 세 가지를 사용하여 두 가지 다른 분류 작업(객관적 및 주관적)에 대해 추출적(extractive) 및 반사례적(counterfactual) 설명을 평가했습니다. 주요 발견은 LLM의 자체 설명이 인간의 판단과 일부 일치할 수 있지만, 모델의 실제 의사 결정 과정을 완전히 정확하게 따르지 않는다는 점입니다. 반사례적 설명을 요청함으로써 이 간극을 극복할 수 있다는 것을 보여줍니다.

- **Technical Details**: 논문에서는 트랜스포머(Transformer) 아키텍처에 기반한 사전 학습된 텍스트-텍스트 처리 시스템인 LLM을 정의합니다. LLM은 입력 텍스트를 예측하여 텍스트를 완성하는 방식으로 작동합니다. 이 논문에서는 지역적 설명(local explainability)에 중점을 두어 트랜스포머의 특정 예측을 설명합니다. 방법론적으로는 피처 점수 할당(feature attribution) 접근 방법, 서브 아키텍처 설명 방법, 대리 모델 기반 방법 등 다양한 접근을 통해 설명을 생성합니다.

- **Performance Highlights**: LLM의 추출적 설명은 인간의 평가와 높은 상관관계를 보이지만 모델의 의사 결정 과정을 완전하고 정확하게 설명하지 않습니다. 반면, 반사례적 설명을 요청하면 모델의 실제 추론 과정과 일치하며, 더 신뢰할 수 있는 설명을 생성할 수 있습니다.



### ChatQA 2: Bridging the Gap to Proprietary LLMs in Long Context and RAG Capabilities (https://arxiv.org/abs/2407.14482)
- **What's New**: 이 논문에서는 ChatQA 2 모델을 소개합니다. 이 모델은 Llama3 기반으로 설계되어, 오픈액세스 LLM(Open Access Large Language Model)과 선도적인 독점 모델들 (예: GPT-4-Turbo) 사이의 격차를 해소하기 위해 제작되었습니다. 특히 긴 문맥 이해와 검색 증강 생성(RAG)에 초점을 맞추고 있습니다.

- **Technical Details**: Llama3-70B-base 모델의 문맥 창을 8K에서 128K 토큰으로 확장하기 위한 상세한 추가 훈련 레시피를 제시합니다. 또한, 명령어 준수, RAG 성능, 긴 문맥 이해 능력을 향상시키기 위한 3단계 명령어 튜닝 프로세스를 도입하였습니다.

- **Performance Highlights**: Llama3-ChatQA-2-70B 모델은 여러 긴 문맥 이해 작업에서 GPT-4-Turbo-2024-0409과 비교할만한 정확도를 달성했으며, RAG 벤치마크에서 더 나은 성능을 보여줍니다. 최신 긴 문맥 검색기(Retriever)를 활용해 RAG 기반 결과를 개선했습니다.



### Check-Eval: A Checklist-based Approach for Evaluating Text Quality (https://arxiv.org/abs/2407.14467)
- **What's New**: 이번 논문에서는 Check-Eval이라는 새로운 평가 프레임워크를 제안하였습니다. 이 프레임워크는 큰 언어 모델(LLMs)을 활용하여 생성된 텍스트의 품질을 체크리스트 기반으로 평가합니다. Check-Eval은 참조 없이도, 또는 참조 텍스트를 기반으로 평가할 수 있어 구조적이며 해석 가능한 텍스트 품질 평가를 제공합니다.

- **Technical Details**: Check-Eval은 주요 단계로 체크리스트 생성(Generation)과 체크리스트 평가(Evaluation)로 구성됩니다. 이 방법론은 Reference-Guided, Candidate-Guided, Criterion-Guided의 세 가지 방식으로 적용될 수 있습니다. Reference-Guided는 참조 텍스트를 기반으로 체크리스트를 생성하여 후보 텍스트를 평가하고, Candidate-Guided는 후보 텍스트를 기반으로 체크리스트를 생성하여 참조 텍스트를 평가합니다. Criterion-Guided는 특정 평가 기준을 사용해 체크리스트를 생성하고 이를 후보 텍스트와 참조 텍스트를 비교하여 평가합니다.

- **Performance Highlights**: Check-Eval은 포르투갈어 법률의 의미적 텍스트 유사성(Portuguese Legal Semantic Textual Similarity)과 SummEval 데이터셋에서 평가되었으며, G-Eval과 GPTScore 등의 기존 메트릭보다 인간 판단과 더 높은 상관관계를 나타냈습니다. 이는 Check-Eval이 자연언어 생성(NLG) 작업에서 더 신뢰할 수 있고 효과적인 평가 프레임워크임을 입증합니다.



### Open Artificial Knowledg (https://arxiv.org/abs/2407.14371)
- **What's New**: OAK 데이터셋(OAK dataset)은 5억 개 이상의 토큰으로 구성된 대규모 데이터세트로, 다양한 도메인에서 고품질 텍스트 데이터를 생성하도록 설계되었습니다. 이 데이터셋은 GPT4o, LLaMa3-70B, LLaMa3-8B, Mixtral-8x7B, Gemma-7B, Gemma-2-9B와 같은 최첨단 모델들을 활용하여 생성되었습니다. OAK 데이터셋은 데이터 부족 및 프라이버시 문제를 해결하면서 더 유능하고 정렬된 언어 모델(Large Language Models, LLMs)의 개발을 촉진하는 것을 목표로 합니다. 누구나 무료로 이용할 수 있습니다.

- **Technical Details**: OAK 데이터 생성 과정은 Wikipedia의 주요 카테고리를 기반으로 한 주제 추출, 서브토픽 확장, 프롬프트 생성 등의 단계를 포함합니다. 주제 추출 단계에서는 Wikipedia를 이용해 높은 수준의 주제를 추출하고, 이를 서브토픽으로 확장합니다. 프롬프트 생성을 위해 프로그래밍 프롬프트 엔지니어링 및 메타 프롬프트(Meta-prompt) 기법을 활용하여 프롬프트의 품질, 길이, 스타일을 최적화합니다. 이 과정에서는 다양성과 일반화(Diversity and Generalization), 품질(Quality), 편향(Bias)을 특별히 고려하여 데이터를 생성합니다.

- **Performance Highlights**: OAK 데이터셋은 데이터 다양성 문제를 해결하기 위해 493,237개의 고유 서브카테고리를 생성하여 21,311개의 카테고리를 포괄합니다. 또한, 프라이버시를 보장하면서 현실 세계를 반영하는 다양한 시나리오를 포함하도록 설계되었습니다. 데이터 품질 유지와 비용 효율성을 동시에 달성하며, 사실에 기반한 정보를 제공하도록 고안되었습니다.



### LLMs left, right, and center: Assessing GPT's capabilities to label political bias from web domains (https://arxiv.org/abs/2407.14344)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구는 OpenAI의 GPT-4가 뉴스 소스의 URL만을 기반으로 정치적 편향을 얼마나 정확히 분류할 수 있는지를 조사합니다. 이는 Ad Fontes Media, AllSides, Media Bias/Fact Check(MBFC)와 같은 제3기관의 편향 평가를 기준으로 합니다.

- **Technical Details**: GPT-4는 최신의 대형 언어 모델(LLM)로서, 뉴스 소스의 웹 도메인만을 기반으로 하는 7단계 평가 체계('극좌', '좌', '중도좌', '편향 없음', '중도우', '우', '극우')에 따라 분류를 시도했습니다. 연구에서는 GPT-4의 분류 결과와 MBFC의 평가를 비교하고, 뉴스 소스의 인기도를 Open PageRank 점수를 이용해 조정하였습니다.

- **Performance Highlights**: GPT-4는 MBFC 평가와 높은 상관관계(Spearman's ρ = .89, n=5,877, p<0.001)를 보여, 모델의 신뢰성을 나타냈습니다. 그러나 GPT-4는 데이터셋의 약 2/3에 대해 분류를 유보했으며, 이는 주로 덜 유명하고 덜 편향된 소스였습니다. 또한, GPT-4의 분류는 MBFC와 비교하여 약간의 좌편향이 있음을 발견했습니다. 연구는 AI와 인간 판단의 혼합 접근법을 권장하며, 다양한 환경, 언어 및 추가 데이터셋에서 모델의 성능을 더 연구할 필요가 있음을 제안합니다.



### Multimodal Misinformation Detection using Large Vision-Language Models (https://arxiv.org/abs/2407.14321)
Comments:
          Accepted for publication in: Conference on Information and Knowledge Management (CIKM) 2024

- **What's New**: 최근 대형 언어 모델(LLMs)이 다양한 작업에서 뛰어난 성능을 보이고 있지만, 여전히 미정보(misinformation) 탐지에 있어서는 충분히 탐구되지 않았습니다. 본 논문에서는 LLMs가 미정보 탐지에 어떻게 도움이 될 수 있는지에 대해 제로샷 학습(zero-shot setting) 환경에서 탐구하였습니다. 새로운 재랭킹 방법을 통해 멀티모달(multimodal) 증거 탐색을 수행하고, LLMs와 대형 비전-언어 모델(LVLM)을 활용해 증거를 재랭킹하고 사실 확인(fact verification) 작업을 개선하는 접근법을 제안했습니다.

- **Technical Details**: 이 연구에서는 멀티모달 미정보 탐지를 위한 파이프라인을 소개하며, 특히 증거 탐색(evidence retrieval)에 대한 신규 재랭킹 접근방식을 제안합니다. 이 접근방식에서는 LLMs와 LVLMs를 활용하여 텍스트 및 이미지 증거를 재랭킹합니다. 이 후, 멀티모달 사실 확인을 위해 LVLM4FV라는 접근법을 적용하며, 프로밍 전략을 통해 텍스트와 이미지 증거를 분류기로 사용합니다. 또한, 기존 증거 탐색 데이터셋에서 불완전한 증거 샘플을 보완하여 공정한 평가를 위해 새로운 주석 작업을 수행하였습니다.

- **Performance Highlights**: 실험 결과는 두 개의 데이터셋에서 제안된 접근법이 증거 탐색과 사실 확인 작업에서 우수한 성능을 보임을 입증하였습니다. 이를 통해 제안된 접근법이 감독 학습 기반의 기존 방법들보다 데이터셋 간의 일반화 능력에서도 뛰어나다는 것을 보여주었습니다.



### How to Engage Your Readers? Generating Guiding Questions to Promote Active Reading (https://arxiv.org/abs/2407.14309)
Comments:
          arXiv admin note: text overlap with arXiv:1504.00704 by other authors

- **What's New**: 본 연구에서는 교재와 과학 기사의 텍스트 내에 포함된 질문들(in-text questions)을 분석한 새로운 데이터셋인 'GuidingQ'를 소개합니다. 이 데이터셋은 약 10,000개의 질문으로 구성되어 있으며, 텍스트 내 질문의 사용, 분포, 언어적 특성을 종합적으로 이해하는 것이 목표입니다.

- **Technical Details**: GuidingQ 데이터셋을 분석하여 질문 사이의 관계(inter-question relationships)와 질문 위치 식별(question position identification)의 중요성을 강조했습니다. 다양한 언어 모델(language models)을 사용해 질문을 생성하는 접근법들을 탐구하였고, 질문 생성 시 발생하는 문제점을 찾았습니다.

- **Performance Highlights**: 실시한 사용자 연구에서 생성된 질문들이 독자의 기억력과 이해력 향상에 있어서 사람 작성 질문들 만큼이나 효과적임을 확인했습니다. 이는 생성된 질문들이 높은 품질을 갖고 있음을 시사합니다.



### CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units (https://arxiv.org/abs/2407.14295)
Comments:
          Accepted to ACL 2024 Student Research Workshop (ACL-SRW 2024)

- **What's New**: 이 논문에서는 다국어 코드 스위칭(Code-Switching) 연구의 한계를 극복하기 위해 PSST 모델을 사용하여 CoVoST 2 데이터셋에서 추출한 억양 단위(intonation unit)를 대체하여 새로운 코드 스위칭 데이터셋 CoVoSwitch를 생성했습니다. 이 데이터셋은 13개 언어로 구성되어 있으며, 이는 기존의 데이터셋보다 더 광범위한 언어 표현을 가능하게 합니다.

- **Technical Details**: PSST(Prosodic Speech Segmentation Tool) 모델을 OpenAI의 Whisper의 사전 훈련된 모델을 사용하여 미세 조정하였으며, 이를 통해 영어 억양 단위를 감지하고 대체했습니다. 생성된 데이터셋은 M2M-100 418M과 NLLB-200 600M이라는 두 개의 다국어 번역 모델을 통해 번역 성능을 평가했습니다. 특히, 저자원 언어(low-resource language)에 대한 번역 성능도 관찰했습니다.

- **Performance Highlights**: 코드 스위칭 유닛을 포함하면 단일 언어 설정보다 높은 번역 성능을 보였습니다. 영어로의 코드 스위칭 번역이 다른 언어로의 번역보다 더 나은 성능을 보였으며, 저자원 언어에서는 특히 영어로 번역할 때 성능 향상이 두드러졌습니다. 그러나 저자원 언어로의 번역 성능은 여전히 미미한 것으로 나타났습니다. 시스템은 영어 토큰을 복사하는 데에는 강하지만 비영어 토큰에서는 어려움을 겪었으며, 단일 언어 및 코드 스위칭 설정 모두에서 목표 외(off-target) 문제와 환각(hallucination) 문제가 관찰되었습니다.



### Predictive Simultaneous Interpretation: Harnessing Large Language Models for Democratizing Real-Time Multilingual Communication (https://arxiv.org/abs/2407.14269)
Comments:
          7 pages

- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 예측 기능을 직접 활용하는 동시통역에 대한 혁신적 접근법을 도입합니다. 새로운 알고리즘을 통해 실시간 번역을 생성하며, 화자의 발언을 예측하고 나무 구조(tree-like structure)로 여러 가능성을 확장합니다. 이 방법은 기존 시스템보다 언어 간 구조적 차이를 효과적으로 극복할 수 있는 유연성과 적응성을 보여줍니다.

- **Technical Details**: 이 논문에서는 이 혁신적 개념을 학계와 공유하고, 이 기술의 이론적 기초, 잠재적 장점 및 구현 도전 과제에 대해 논의합니다. 이 새로운 알고리즘은 자연스럽고 유창한 번역을 매우 짧은 지연 시간으로 제공할 수 있다는 점에서 주목할 만합니다. 예측 기반 번역은 예제와 이론적 분석을 통해 설명됩니다.

- **Performance Highlights**: 이 접근법은 기존 시스템에 비해 더 자연스럽고 유창한 번역 결과를 제공하며, 동시통역에서 중요한 최소 지연(lag)으로 번역을 가능하게 합니다. 이 연구는 다국어 소통의 민주화를 향한 중요한 진전으로 평가될 수 있습니다.



### Voices in a Crowd: Searching for Clusters of Unique Perspectives (https://arxiv.org/abs/2407.14259)
- **What's New**: 언어 모델의 훈련 데이터에 존재하는 편향을 반영하는 문제를 해결하기 위해 새로운 프레임워크를 제안했습니다. 이 프레임워크는 주석자 메타데이터를 인코딩하지 않고 주석자 행동에 의해 정보가 제공되는 잠재 임베딩(latent embeddings)을 추출하고, 유사한 의견을 가진 클러스터를 생성합니다. 이러한 클러스터를 '목소리(voices)'라 부르며, 결과 클러스터는 내외부 정량적, 정성적 분석을 통해 검증됩니다. 이를 통해 다양한 데이터셋에서 소수자 관점을 잘 포착할 수 있음을 보여줍니다.

- **Technical Details**: 이 프레임워크는 감독 학습(supervised)과 비감독 학습(unsupervised)의 두 가지 구성 요소로 구성됩니다. 감독 학습 구성 요소는 각 텍스트 입력에 대해 주어진 주석자의 개별 주석을 예측하도록 모델을 학습시키며, 비감독 학습 구성 요소는 행동 임베딩을 대상으로 차원 축소(dimensionality reduction)를 수행하고 이후 여러 알고리즘을 사용하여 클러스터링을 수행합니다. 이 방법은 주석자 메타데이터를 사전에 수집하지 않고도 주석자 행동의 유사성을 기반으로 여러 목소리를 식별합니다.

- **Performance Highlights**: 두 가지 정치 편향과 관련된 데이터셋에서 실험한 결과, 제안된 프레임워크는 내부 및 외부 지표 모두에서 높은 일반화 성능을 보여주었습니다. 클러스터링 결과는 다수 대표 레이블(예: 좌파 성향)과 소수 대표 레이블(예: 좌파 성향 데이터셋 내의 우파 성향)을 모두 효과적으로 포착했습니다. 특히, 서로 다른 소수 그룹 간의 교차 소수자 레이블도 인식할 수 있었습니다. 예를 들어, 좌파 성향의 다수 데이터셋에서 고학력자가 아닌 다수와는 반대로 고학력 우파 성향을 가진 소수 그룹을 동적으로 식별할 수 있었습니다.



### Conditioning Chat-GPT for information retrieval: the Unipa-GPT case study (https://arxiv.org/abs/2407.14246)
- **What's New**: Unipa-GPT는 팔레르모 대학의 학부/석사 학위 과정을 선택하는 학생들을 돕기 위해 개발된 gpt-3.5-turbo 기반 챗봇입니다. 이 챗봇은 유럽 연구자들의 밤(SHARPER night) 행사에서 소개되었습니다. Unipa-GPT는 Retrieval Augmented Generation (RAG) 방식과 미세조정(fine-tuning)을 사용하여 개발되었습니다.

- **Technical Details**: Unipa-GPT는 RAG 시스템과 미세조정된 시스템을 비교하여 전체 아키텍처를 제시합니다. gpt-3.5-turbo를 기반으로 하여 unipa-corpus라는 대학 웹사이트에서 스크래핑한 문서들을 사용합니다. RAG 시스템은 LangChain과 FAISS 라이브러리를 사용하여 벡터 데이터베이스를 구성하고 있으며, 각 문서는 텍스트 임베딩(embedding)을 위해 text-embedding-ada-002 모델을 사용하여 1000 토큰 길이로 분할되었습니다. 미세조정된 시스템은 ChatGPT API를 통해 모델 추론 및 fine-tuning을 수행했습니다.

- **Performance Highlights**: 모델은 SHARPER night 행사에서 실제 사용자들이 사용해본 결과를 토대로 성능이 평가되었습니다. 미세조정된 버전과 순수 RAG 시스템을 비교한 결과, 각각의 장단점이 명확히 드러났습니다. 최적의 모델은 미세조정과 RAG 방식을 혼합하여 교육 목표와 같은 세부 정보에 대해 학습 단계를 거치지 않도록 하여 컴퓨팅 자원을 절약했습니다.



### LeKUBE: A Legal Knowledge Update BEnchmark (https://arxiv.org/abs/2407.14192)
- **What's New**: 최근의 대형 언어 모델(LLM) 연구는 법률 지능 분야에서 큰 진전을 보였습니다. LLM을 사용하는 법률 애플리케이션에서 중요한 문제 중 하나는 '법률 지식 업데이트'입니다. 이를 해결하기 위해, 새로운 Legal Knowledge Update Benchmark, LeKUBE를 도입했습니다. LeKUBE는 법률 도메인의 고유한 업데이트 필요성을 평가하기 위해 설계되었습니다. 특히, 중국의 형법과 민법에 대한 합성 업데이트를 통한 평가를 제공합니다.

- **Technical Details**: LeKUBE 벤치마크는 법률 도메인의 지식 업데이트 필요성을 다섯 가지 차원으로 분류합니다. 법률 전문가의 도움을 받아 법률 텍스트에서 합성 업데이트를 생성하고, 이 업데이트 후 질문 및 과제를 포함합니다. 평가 방법은 비매개(non-parametric) 전략과 매개(parametric) 전략으로 나뉘어집니다. 비매개 전략은 모델의 파라미터를 변경하지 않고 새로운 지식을 주입하는 방식이며, 매개 전략은 파라미터를 변경하여 지식을 주입하는 방식입니다.

- **Performance Highlights**: LeKUBE를 통해 다양한 최신 지식 업데이트 방법을 실험했으며, 이 과정에서 기존 방법과 법률 도메인의 고유한 필요성 사이의 격차를 확인했습니다. 실험 결과, 다양한 과제에서 지식 업데이트 방법의 성능 차이를 드러냈으며, 법률 지식 업데이트의 어려움과 차이를 보여주었습니다.



### Automatic Classification of News Subjects in Broadcast News: Application to a Gender Bias Representation Analysis (https://arxiv.org/abs/2407.14180)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 이 논문은 프랑스 TV 및 라디오 뉴스에서 다루는 주제에 대한 성별 분포 편향을 분석하는 계산 프레임워크를 소개합니다. 2023년에 21개 프랑스 채널에서 방송된 11.7천 시간의 데이터를 전사해 주제 분류를 수행하였으며, 이 과정에서 LLM(Large Language Model)을 사용하고, 이를 통해 작은 특화된 분류 모델을 미세 조정하여 계산 비용을 줄였습니다. 생성된 LLM 주석을 사용해 여성들은 스포츠, 정치, 분쟁 등의 주제에서 현저히 과소 대표됨을 발견했습니다.

- **Technical Details**: 공개된 데이터셋은 ARCOM 보고서를 기반으로 프랑스의 방송 뉴스를 대상으로 분석됩니다. Whisper 모델을 이용해 방송 내용을 전사하고, WhisperX 구현을 통해 11.8배 속도 향상을 얻었습니다. 804개의 대화 데이터셋이 수동으로 주석 처리되었으며, BERT 모델, few-shot LLM, Teacher/Student 모델 등을 평가했습니다. 이러한 분석을 통해 18개의 주제로 분류된 데이터를 생성했습니다.

- **Performance Highlights**: 대화 데이터셋의 주석 작업은 두 명의 주석자가 수행하여 Krippendorff’s 알파를 통해 일치도를 계산했습니다. 음성 인식 모델 적용 후 평균 Word Error Rate(WER)는 10.66%였으며, 대화 당 평균 인식된 주제 수는 2.55에서 2.62 사이였습니다. 주요 발견 사항으로는, 여성의 발언 시간이 스포츠, 정치, 분쟁에서 현저히 적고, 날씨, 광고, 건강 주제에서는 평균보다 많은 시간을 가지는 경향이 있다는 것입니다.



### I Know About "Up"! Enhancing Spatial Reasoning in Visual Language Models Through 3D Reconstruction (https://arxiv.org/abs/2407.14133)
- **What's New**: 새로운 모델인 ZeroVLM은 비주얼 랭귀지 모델(Visual Language Models, VLMs)의 시각적 공간 추론 능력을 향상시키기 위해 제안되었습니다. 기존의 VLMs는 기초적인 공간 추론 작업에서 어려움을 겪었으나, ZeroVLM은 3D 재구성 및 특정 프롬프팅 메커니즘을 사용하여 이러한 문제를 해결합니다.

- **Technical Details**: ZeroVLM은 3D 재구성 모델인 Zero-1-to-3를 사용하여 입력 이미지의 여러 시각을 생성합니다. 이를 통해 VLMs는 풍부한 시각적 정보를 획득하며, 프롬프팅 메커니즘을 추가하여 시각적 공간 추론 능력을 더욱 향상시킵니다. ZeroVLM은 이미지 인코더, 이미지와 텍스트 표현을 정렬하는 임베딩 프로젝터, 텍스트 디코더로 구성되어 다양한 시각적 언어 작업을 수행할 수 있습니다.

- **Performance Highlights**: ZeroVLM은 네 가지 시각적 공간 추론 데이터셋에서 최대 19.48%의 정확도 향상을 달성했습니다. 이는 3D 재구성 및 프롬프팅 메커니즘이 VLM의 시각적 공간 추론 능력을 효과적으로 향상시킨다는 것을 입증합니다.



### Impact of Model Size on Fine-tuned LLM Performance in Data-to-Text Generation: A State-of-the-Art Investigation (https://arxiv.org/abs/2407.14088)
Comments:
          30 pages

- **What's New**: 이번 연구는 데이터-텍스트(D2T) 생성 작업에서 LLM(대형 언어 모델)의 모델 크기가 성능에 미치는 영향을 분석합니다. 특히, 최소 5가지 주요 D2T 데이터셋(E2E, ViGGo, WikiTableText, DART, WebNLG)과 5개의 서로 다른 LLM 계열(T5, BART, OPT, BLOOM, Llama 2)에 속하는 12개의 최신 LLM을 통해 신중한 비교 분석을 수행했습니다.

- **Technical Details**: 연구는 D2T 모델 성능 평가에 세 가지 핵심 품질 - 가독성(readability), 정보 전달력(informativeness), 신뢰성(faithfulness) - 을 사용합니다. 이를 위해 	extsc{BLEU}, 	extsc{METEOR}, 	extsc{BERTScore}, 	extsc{MoverScore}, 	extsc{Parent}, 	extsc{BARTScore}와 같은 여섯 가지 자동 평가 지표를 적용했습니다. 특히 모델 크기가 D2T 작업의 성능에 미치는 영향을 살펴보는 데 중점을 두었습니다.

- **Performance Highlights**: 결과적으로, 모델 크기를 증가시키는 것은 D2T 작업에서 가독성과 정보 전달력을 향상시키는 것으로 나타났으나, 큰 LLM들은 신뢰성을 희생할 가능성이 있음을 발견했습니다. 또한, 소형 LLM들이 소스와 참조 데이터 간의 다이버전스(divergence)가 존재할 때 더 강인한 성능을 보였습니다.



### An Improved Method for Class-specific Keyword Extraction: A Case Study in the German Business Registry (https://arxiv.org/abs/2407.14085)
Comments:
          7 pages, 1 figure, 1 table. Accepted to KONVENS 2024

- **What's New**: 이 논문은 KeyBERT 라이브러리를 기반으로, 사전 정의된 클래스와 관련된 키워드만을 추출하는 방법을 제안합니다. 독일의 사업자 등록 데이터를 사용하여 경제 섹터별로 비즈니스를 분류하는 데 이 방법을 테스트하였으며, 결과는 기존 방법들보다 크게 향상된 성능을 보였습니다.

- **Technical Details**: 제안된 키워드 추출 파이프라인은 세 가지 사전 필수 요건—문서 본문, 사전 정의된 클래스, 클래스 별 시드 키워드—을 충족하는 문서 군을 배치 단위로 처리합니다. KeyBERT의 기능을 수정하여 시드 키워드 임베딩(embeddings)에 100% 집중하게 만들었으며, 이후 평균 점수와 최대 점수를 계산하여 최종 후보 키워드를 재정렬합니다. 최종 상위 키워드는 다음 반복(iteration)에서 시드 키워드 세트에 추가됩니다.

- **Performance Highlights**: 독일 사업자 등록 데이터셋을 평가 대상으로 사용하여 제안된 방법을 테스트한 결과, 제안된 class-specific 키워드 추출 방법은 기존의 키워드 추출 방법보다 뛰어난 성능을 보여 주었습니다. 이는 클래스별 키워드를 추출하는 새로운 표준 세우기에 기여하였습니다.



### LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inferenc (https://arxiv.org/abs/2407.14057)
- **What's New**: 최근 발표된 연구 LazyLLM에서는 트랜스포머 기반 대형 언어 모델(LLM)의 프리필링(prefilling) 단계를 최적화하여 첫 번째 토큰을 생성하는 시간을 줄이는 새로운 방법을 제안합니다. LazyLLM은 프리필링 및 디코딩(decoding) 단계에서 중요 토큰만을 선택적으로 계산하여 전체 성능을 가속화합니다.

- **Technical Details**: LazyLLM은 각 생성 단계에서 토큰의 중요성을 측정하고, 이전에 프룬(prune)된 토큰이라도 필요할 때 동적으로 활성화합니다. 이를 위해 모델의 주의(attention) 점수를 활용하여 토큰의 중요도를 평가하며, 반복 계산을 피하기 위해 보조 캐시(Aux Cache)를 사용합니다.

- **Performance Highlights**: LazyLLM은 다양한 표준 데이터셋을 사용한 실험에서, LLama 2 7B 모델의 프리필링 단계를 2.34배 가속화하면서도 정확성을 유지하는 성과를 보였습니다. 또한, LazyLLM은 기존의 트랜스포머 기반 모델과 무리 없이 통합될 수 있으며, 추가 학습이나 파라미터 수정 없이 바로 적용이 가능합니다.



### Rasa: Building Expressive Speech Synthesis Systems for Indian Languages in Low-resource Settings (https://arxiv.org/abs/2407.14056)
Comments:
          Accepted at INTERSPEECH 2024. First two authors listed contributed equally

- **What's New**: Rasa는 아삼어, 벵골어, 타밀어로 구성된 첫 번째 다언어 감정 TTS(TTS 데이터셋로, 10시간의 일반 음성 데이터와 각 6개의 Ekman 감정을 포함한 1-3시간의 감정 표현 음성을 제공합니다. Rasa는 자원이 제한된 언어에 대해 실용적인 데이터를 제공합니다.

- **Technical Details**: Rasa 데이터셋은 FastPitch, HiFiGAN-V1와 같은 최신 TTS 기술을 사용해 구축되었습니다. 초대형 언어 모델(LLMs)과 인간 주석자를 활용해 감정이 풍부한 스크립트를 제작하였으며, 음성 녹음은 스튜디오 환경에서 높은 품질로 이루어졌습니다. 또한 TTS 시스템을 구축하기 위해 음절적 균형 데이터를 사용했습니다.

- **Performance Highlights**: 아삼어, 벵골어, 타밀어로 된 TTS 시스템은 MUSHRA 점수에서 높은 평가를 받았습니다. 특히 1시간의 일반 음성과 30분의 감정 데이터만으로도 '공정' 수준의 TTS 시스템을 구축할 수 있음을 확인했습니다. 중립 데이터 10시간과 최소한의 감정 데이터를 추가할 경우 표현력이 크게 향상되었습니다.



### Prompted Aspect Key Point Analysis for Quantitative Review Summarization (https://arxiv.org/abs/2407.14049)
Comments:
          Accepted by ACL 2024 Main Conference

- **What's New**: 이번 논문에서는 Prompted Aspect Key Point Analysis (PAKPA)라는 새로운 프레임워크를 제안하여 리뷰 요약을 더 정확하고 효율적으로 수행합니다. 기존의 방식과 달리 PAKPA는 대규모 주석 데이터가 필요하지 않으며, 중요한 의견을 추출하고 이를 정량화하는 방법을 제시합니다. 이 기술은 Yelp와 SPACE 데이터셋에서 최첨단 성능을 보여줍니다.

- **Technical Details**: PAKPA는 다음 세 가지 주요 구성 요소로 구성됩니다: 1) 의견 분석을 통해 리뷰 코멘트에서 측면(term) 및 감정(sentiment)을 추출, 2) 유사한 측면과 감정을 공유하는 코멘트를 클러스터링하여 그룹화, 3) 클러스터에서 KP(Key Points)를 생성 및 정량화합니다. 대규모 언어 모델(LLMs)을 사용하여 주석 데이터 없이도 이러한 작업을 수행할 수 있게끔 합니다.

- **Performance Highlights**: 실험 결과, PAKPA는 Yelp 및 SPACE 데이터셋에서 현존 최상의 성능을 기록하였습니다. 특별히, 리뷰의 측면 기반 중요 포인트를 더욱 충실하고 정확하게 생성하고 정량화하여, 이전 방식들의 한계를 극복하였습니다.



### ECCO: Can We Improve Model-Generated Code Efficiency Without Sacrificing Functional Correctness? (https://arxiv.org/abs/2407.14044)
Comments:
          Code: this https URL, 14 pages, 11 figures, Pre-print

- **What's New**: 최근 발표된 ECCO는 코드 효율성을 평가하기 위한 새로운 벤치마크로, 두 가지 패러다임을 통해 프로그램의 효율성을 평가합니다: 자연어(NL) 기반 코드 생성과 히스토리 기반 코드 편집. ECCO는 실행 정정성, 런타임 효율성 및 메모리 효율성을 중심으로 프로그램을 평가하며, 50k 이상의 Python 솔루션 쌍을 포함하고 있습니다.

- **Technical Details**: ECCO는 클라우드 호스팅 코드 실행 엔진 Judge0를 사용하여 하드웨어 사양에 구애받지 않는 안정적인 실행 출력을 제공합니다. 이 엔진은 66개 이상의 프로그래밍 언어를 지원합니다. 벤치마크는 세 가지 주요 접근 방식을 평가합니다: 컨텍스트 학습, 반복적 세분화, 그리고 실행 및 편집 히스토리를 기반으로 한 파인튜닝.

- **Performance Highlights**: 실행 정보와 파인튜닝이 함수적 정정성을 유지하는 데 도움이 되며, 자연어(NL) 피드백이 효율성을 더욱 향상시키는 것으로 나타났습니다. 그러나 현재의 방법들로는 시간/공간 효율성을 개선하면서 함수적 정정성을 유지하는 데 한계가 있음이 확인되었습니다.



### BERTer: The Efficient On (https://arxiv.org/abs/2407.14039)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구에서는 BERT 모델의 성능을 향상시키기 위한 고급 미세 조정(advanced fine-tuning) 기법을 탐구했습니다. 특히 감정 분석(sentiment analysis), 패러프레이즈 검출(paraphrase detection), 그리고 의미적 텍스트 유사성(semantic textual similarity) 작업에 초점을 맞췄습니다. SMART 정규화(SMART regularization), 크로스 임베딩 시아미즈 아키텍처(cross-embedding Siamese architecture), 조기 종료 방법(early exiting methods) 등을 도입해 모델의 효율성과 효과성을 크게 향상시켰습니다.

- **Technical Details**: {'Approach': 'minBERT Transformer 모델을 개발하였으며, 이는 12개의 트랜스포머 레이어로 구성되어 있습니다. 각 레이어는 자체 주의 메커니즘(self-attention mechanism)을 포함하고 있습니다. 베이스라인 멀티태스크 BERT 모델은 미리 훈련된 minBERT의 가중치를 사용하여 모든 작업에 대한 특징 추출을 수행합니다. 아키텍처는 각 작업에 맞게 정형화된 세 개의 작업별 헤드를 확장하여 각각 감정 분석, 패러프레이즈 검출, 의미적 텍스트 유사성 작업을 수행하도록 설계되었습니다.', 'Regularization': '오버피팅(overfitting)을 방지하기 위해 SMART 정규화(SMART regularization) 기법을 사용했습니다. 이는 부드러운 적대적 정규화(smoothness-inducing adversarial regularization)와 브레그만 근방점 최적화(Bregman Proximal Point Optimization)의 두 가지 주요 구성 요소를 포함합니다.', 'Optimization': '최적화 알고리즘으로는 주로 AdamW를 사용했으며, RMSprop 및 SGD와 같은 대체 알고리즘도 실험했습니다.'}

- **Performance Highlights**: 우리의 실험 결과, SMART 정규화는 패러프레이즈 검출(paraphrase detection) 작업에서 매우 큰 초기 향상을 보였으며, 감정 분류(sentiment classification), 텍스트 유사성 분석(semantic similarity analysis)에서는 상대적으로 덜 효과적이었습니다. 하지만 전반적으로 멀티태스크 학습을 통해 모델의 총체적인 성능과 효율성이 크게 향상되었으며, 테스트 셋에서의 최고 성능 점수를 기록하여 기존 벤치마크를 초과하는 결과를 보였습니다.



### HeCiX: Integrating Knowledge Graphs and Large Language Models for Biomedical Research (https://arxiv.org/abs/2407.14030)
Comments:
          8 pages, 3 figures, under review

- **What's New**: 이번 연구에서는 HeCiX-KG (Hetionet-Clinicaltrials neXus Knowledge Graph)을 도입하여 임상 시험 및 생물학적 데이터를 종합적으로 통합한 새로운 지식 그래프를 소개했습니다. HeCiX-KG는 ClinicalTrials.gov와 Hetionet의 데이터를 하나로 합친 것으로, 이를 통해 임상 시험의 실패율을 감소시키기 위한 보다 철저한 자원을 제공합니다. 또한, HeCiX라는 시스템을 개발하여 LangChain을 사용해 HeCiX-KG와 GPT-4(일반적인 언어 모델)를 통합하였습니다.

- **Technical Details**: HeCiX-KG는 ClinicalTrials.gov와 Hetionet의 데이터를 통합하여 만들어졌으며, 6가지 특정 질병(백반증, 아토피 피부염, 원형 탈모증, 흑색종, 간질, 갑상선 기능 저하증)에 대한 데이터를 포함합니다. 총 6,509개의 노드와 14,377개의 엣지를 가진 지식 그래프로 구성됩니다. HeCiX 시스템은 LangChain의 GraphCypherQAChain를 사용하여 자연어 쿼리를 Cypher Query Language(CQL)로 변환하고 이를 HeCiX-KG에 실행한 후, 결과를 인간이 이해할 수 있는 형식으로 제시합니다.

- **Performance Highlights**: HeCiX 시스템은 임상 연구의 다양한 질문에 대해 높은 성능을 보였습니다. 질문 답변 작업에 대한 평가 결과, 모델은 신뢰성, 답변의 관련성, 정확한 컨텍스트와 회수율 등의 측면에서 상당한 향상을 보여주었습니다. 이는 임상 연구의 효율성을 높이고 약물 재창출 및 개발 성공률을 향상시키는 데 유용한 도구가 될 수 있습니다.



### NeLLCom-X: A Comprehensive Neural-Agent Framework to Simulate Language Learning and Group Communication (https://arxiv.org/abs/2407.13999)
- **What's New**: 최근 AI 연구에서 인간처럼 소통하는 언어를 생성할 수 있는 프레임워크 NeLLCom의 확장 버전인 NeLLCom-X가 소개되었습니다. 이 신기술은 더욱 현실적인 역할 교환 에이전트와 그룹 커뮤니케이션을 도입하여 언어 학습 가능성, 소통 압력 및 그룹 크기 효과를 연구할 수 있게 합니다.

- **Technical Details**: NeLLCom-X는 초기에는 강화 학습을 통해 인공 언어를 학습한 에이전트들이 이후 상호 작용을 통해 그 언어를 활용해 소통하도록 설계되었습니다. 이번 프레임워크는 에이전트가 역할을 교환할 수 있도록 하는 기능과 다양한 크기의 그룹 내 상호 작용을 허용하며, 초기 언어 노출이 동일하거나 다른 에이전트들이 함께 학습하도록 설계되었습니다.

- **Performance Highlights**: NeLLCom-X는 초기 다른 언어를 학습한 에이전트들이 빠르게 상호 이해 가능한 언어로 적응하며, 큰 그룹 안의 에이전트들이 사용하는 언어가 더 최적화되고 중복이 줄어듦을 확인했습니다. 또한 개별 사용자뿐만 아니라 그룹 수준에서도 단어 순서 및 격표지상의 상쇄(trade-off) 현상이 나타나는 것을 보여주었습니다.



### RAG-QA Arena: Evaluating Domain Robustness for Long-form Retrieval Augmented Question Answering (https://arxiv.org/abs/2407.13998)
- **What's New**: 길고 일관된 내러티브를 제공하는 인간이 작성한 장문 답변을 포함한 새로운 데이터셋인 Long-form RobustQA(LFRQA)가 소개되었습니다. 이 데이터셋은 7개의 서로 다른 도메인에 걸쳐 26,000 개의 쿼리를 다루고 있습니다. 또한 RAG-QA Arena라는 새로운 평가 프레임워크도 제안되었으며, 이는 모델 생성 답변과 LFRQA의 답변을 직접 비교합니다.

- **Technical Details**: LFRQA 데이터셋은 다수의 문서에서 추출된 짧은 답변을 하나의 일관된 장문 답변으로 통합한 것입니다. 이는 단일 출처의 짧은 정답으로 구성된 기존 데이터셋의 한계를 극복합니다. RAG-QA Arena는 모델 기반 평가자를 활용하여 LFRQA 데이터의 답변과 LLM(대형 언어 모델)의 답변을 직접 비교합니다. 이는 전체적인 품질 평가에서 인간 판정과 높은 상관관계를 보입니다.

- **Performance Highlights**: 다양한 실험을 통해 RAG-QA Arena와 인간 판단 간의 높은 상관관계가 입증되었습니다. 가장 경쟁력 있는 LLM의 답변 중 단 41.3%만이 LFRQA의 답변보다 선호된다는 결과가 나와, RAG-QA Arena가 미래 연구를 위한 도전적인 평가 플랫폼임을 보여줍니다.



### Reexamining Racial Disparities in Automatic Speech Recognition Performance: The Role of Confounding by Provenanc (https://arxiv.org/abs/2407.13982)
- **What's New**: 이번 연구는 최신 신경망 기반 자동 음성 인식(AI) 시스템인 OpenAI의 Whisper가 CORAAL 데이터셋에서 어떤 성능 차이를 보이는지 조사하고자 합니다. 중요한 결과 두 가지를 발견했습니다. 첫 번째는 AAE(African American English)에 대해 기존보다 낮은 ASR 성능이 확인되었지만, 모델의 미세 조정을 통해 어느 정도 성능을 개선할 수 있다는 점입니다. 두 번째는 음성 녹음 방식의 차이가 ASR 정확도에 중요한 영향을 미친다는 새로운 발견입니다.

- **Technical Details**: Whisper 모델은 시퀀스 투 시퀀스(seq2seq) 방식의 신경망 모델입니다. 연구에서는 CORAAL 데이터셋 2021 릴리스를 사용하여 ATL(애틀랜타, 조지아), DCA/DCB(워싱턴 디스트릭트 콜롬비아), LES(맨해튼, 뉴욕), PRV(프린스빌, 노스캐롤라이나), ROC(로체스터, 뉴욕), VLD(발대스타, 조지아) 등 6개 지역에서 수집된 인터뷰를 분석했습니다. 또한, 분석은 'This Side of The River - The Story of Princeville'이라는 다큐멘터리 비디오와 비교하여 녹음 품질의 영향을 검토했습니다.

- **Performance Highlights**: 고품질의 전문 디지털 장비로 녹음된 다큐멘터리에서는 Whisper가 월등히 높은 정확도를 보였습니다. 반면, 아날로그 장비로 녹음된 PRV 서브셋에서는 정확도가 낮았습니다. 이로 인해 데이터 수집의 방식과 장비 품질이 ASR 성능에 중요한 혼동 변수로 작용함을 확인할 수 있었습니다.



### FANTAstic SEquences and Where to Find Them: Faithful and Efficient API Call Generation through State-tracked Constrained Decoding and Reranking (https://arxiv.org/abs/2407.13945)
- **What's New**: 이번 연구에서는 API 호출 생성을 위한 신기술인 FANTASE를 소개합니다. FANTASE는 State-Tracked Constrained Decoding (SCD)와 Reranking 컴포넌트를 도입하여, 기존 방법들이 겪고 있는 높은 비용, 낮은 데이터 효율성, 그리고 API 문서와 사용자 요청에 불충실한 API 호출 생성 문제를 해결합니다.

- **Technical Details**: FANTASE는 두 가지 주요 부분으로 구성됩니다: 1) SCD는 API 문서의 제약 조건을 동적으로 통합하여 Token Search Trie 형태로 반영합니다. 이를 통해, 효율적이고 신뢰성 있는 API 호출 생성을 보장합니다. 2) Reranking 컴포넌트는 작은 모델을 활용하여 LLM의 후보 생성물을 재순위 매기는 방식으로, 효율적인 감독 신호를 제공합니다.

- **Performance Highlights**: FANTASE는 DSTC8 및 API Bank 데이터셋에서 더 높은 API 호출 생성 정확도, 추론 효율성, 문맥 효율성을 보여주었습니다. 기존의 감독 학습 및 인컨텍스트 학습 방법과 비교했을 때, 데이터 레이블링과 모델 학습 비용을 크게 절감하면서도 높은 성능을 유지합니다.



### Werewolf Arena: A Case Study in LLM Evaluation via Social Deduction (https://arxiv.org/abs/2407.13943)
Comments:
          13 pages, 10 figures

- **What's New**: 새로운 연구 프레임워크, Werewolf Arena가 소개되었습니다. 이 프레임워크는 사회적 추론 게임인 Werewolf를 통해 대형 언어 모델(LLMs)의 평가를 목적으로 합니다. Werewolf Arena에서는 LLM들이 경쟁하여 속임수, 논리적 추론, 설득력을 통해 게임의 동적 시스템을 탐색합니다. 이 연구는 Gemini와 GPT 모델 간의 토너먼트를 통해 모델들의 전략적 추론과 의사소통 능력을 분석하였습니다.

- **Technical Details**: 여기서 중요한 두 가지 기여는 다음과 같습니다. 첫째, 전략적 의사소통의 중요성을 인식하고, 플레이어들이 발언 순서를 입찰하는 동적 턴제 시스템을 도입했습니다. 이 시스템은 실제 그룹 토론에서처럼 참여자들이 발언 타이밍을 전략적으로 선택할 수 있게 합니다. 둘째, Werewolf를 검증 도구로 사용하여 LLM 간의 상대적 능력을 평가합니다. 단일 모델이 Villager와 Werewolf 역할을 모두 맡아 플레이하며, 승률이 균형잡혀 공정한 비교가 가능하도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, 모델들이 전략적 추론과 의사소통에 있어 각각의 강점과 약점을 보여주었습니다. Villager 역할의 모델들이 전략적 의사소통을 강화하지 못할 경우, 10만 번의 시뮬레이션 게임에서 단 1.2%의 승률을 나타냈습니다. 또한, Werewolf Arena를 통해 개방된 평가환경을 제공하여 향후 연구 가능성을 확대했습니다. 현재 Werewolf Arena는 https://github.com/google/werewolf_arena 에서 공개되었습니다.



### BiasDPO: Mitigating Bias in Language Models through Direct Preference Optimization (https://arxiv.org/abs/2407.13928)
- **What's New**: 이 논문은 대형 언어 모델(LLM)에서 성별, 인종, 종교 편향을 줄이기 위한 새로운 프레임워크를 소개합니다. Direct Preference Optimization(DPO)를 사용해 LLM들이 더 적은 편향을 가진 텍스트를 생성하도록 하는 방법을 제안합니다. 또한, 훈련용 데이터셋을 수작업으로 제작하여 공개했습니다.

- **Technical Details**: 제안된 방법은 DPO를 활용하여 더 적은 편향을 가진 토큰의 로그 확률을 극대화하고, 더 많을 편향을 가진 토큰의 로그 확률을 최소화하도록 모델을 훈련합니다. 이를 위해 다양한 프롬프트와 대응되는 편향적 및 비편향적 완료 텍스트를 포함하는 데이터셋을 개발했습니다. Microsoft Phi-2 모델에 이 방법을 적용하여 편향을 실질적으로 줄이는 것을 시연했습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 거의 모든 편향 기준에서 기준 모델보다 우수한 성능을 보였습니다. 또, 다른 오픈소스 모델들과 비교해도 대부분의 기준에서 더 나은 성능을 보였습니다. 결과적으로, 제안된 방법을 통해 생성된 언어는 더욱 중립적이고 존중하는 내용으로 나타났습니다.



### Crafting Efficient Fine-Tuning Strategies for Large Language Models (https://arxiv.org/abs/2407.13906)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 효율적인 미세조정을 위해 데이터 효율성 및 하이퍼파라미터 최적화를 탐구합니다. 최소 데이터 양을 규명하고 초기 단계 모델 성능을 활용한 새로운 하이퍼파라미터 최적화 방법을 제안합니다. 실험 결과, 200개의 샘플만으로 제품 속성 추출 작업에서 모델 정확도가 70%에서 88%로 향상될 수 있음을 보여줍니다. 또한, 6500개의 샘플을 넘어서면 추가 데이터가 한계 효용 감소(diminishing returns)를 보입니다.

- **Technical Details**: 논문에서 제안된 베이지안 최적화 방법(bayesian optimization method)은 총 훈련 시간의 20%에서 모델 성능을 평가하며 최종 모델 성능과 강한 상관관계를 갖습니다. 총 5개의 초기 단계 모델 중 4개가 최종 단계에서도 상위 5위 안에 드는 것을 보여줍니다. 데이터 효율성 실험에서는 데이터 양이 모델 성능에 미치는 영향을 체계적으로 조사하고, 특정 임계치를 넘어서는 추가 데이터의 효용 감소를 확인합니다.

- **Performance Highlights**: 제안된 베이지안 최적화 방법을 통해 독립적인 테스트 세트에서 기준 모델 대비 2%의 정확도 향상을 달성했습니다. 이는 대규모 데이터 집합에 대한 의존도를 줄이면서도 총 모델 성능을 개선하는 데 기여합니다. 특히, 200개의 샘플만으로도 상당한 성능 향상을 이끌어낼 수 있음을 실험으로 입증했습니다.



### Uncovering Political Bias in Emotion Inference Models: Implications for sentiment analysis in social science research (https://arxiv.org/abs/2407.13891)
- **What's New**: 이 논문은 사회과학 연구에서 사용되는 감정 추론 모델(sentiment analysis, SA)에서 발생하는 정치적 편향에 대해 조사합니다. 기존 연구에서는 주로 성별 및 인종 편향이 강조되었지만, 이 연구는 정치적 편향에 초점을 맞추고 있어 새로운 시사점을 제공합니다. 이를 통해 학계와 실제 응용 분야에서 머신 러닝 모델의 신뢰성과 공정성을 높이는 데 중요한 교훈을 줍니다.

- **Technical Details**: 이 연구는 정치적 편향을 감지하기 위해 폴란드에서 개발된 감정 분석 모델을 대상으로 편향 감사(bias audit)를 수행했습니다. 폴란드 정치인들과 관련된 이름 및 문장에 대한 valence(감정값) 예측을 분석하여 정치적 소속에 따라 시스템적 차이가 발생하는 것을 확인했습니다. 이러한 편향은 인간 평가자의 주석(annotation)에서 모델의 예측으로 전달됨을 발견했습니다. 이를 완화하기 위해 정치인들을 언급한 텍스트를 훈련 데이터셋에서 제거했으며, 편향이 줄어드는 것을 관찰했습니다. 하지만 완전히 제거되지는 않았습니다.

- **Performance Highlights**: 정치적 편향의 영향을 받은 데이터셋을 제거한 후, 편향이 줄어드는 것을 목격했지만 완전히 제거되지 않았습니다. 이러한 결과는 감정 분석 결과를 사용할 때의 주의를 강조하며, 사전 정의된 사전(lexicon-based systems)을 사용하여 보다 이념적으로 중립적인 대안을 제안하고 있습니다. 이 연구는 사회과학 연구에서 감정 분석 모델을 사용할 때 지속적인 검토와 방법론적 수정의 필요성을 강조합니다.



### Learning Goal-Conditioned Representations for Language Reward Models (https://arxiv.org/abs/2407.13887)
- **What's New**: 이 연구는 인간 피드백을 통한 강화학습(Reinforcement Learning from Human Feedback, RLHF)에서 보상 모델(reward models, RMs)을 대조적인(goal-conditioned) 방식으로 훈련하여 언어 모델(LMs)의 성능을 향상시키는 방법을 제시합니다. 이 방식은 선호하는 경로의 미래 상태 간의 표현 유사성을 높이고 비선호하는 경로의 유사성을 낮추는 대조 목표를 사용합니다. 이를 통해 보상 모델의 AUROC 성능이 최대 0.09 향상되었으며, Helpful-Harmless 데이터셋에서 2.3% 정확도 증가를 확인하였습니다.

- **Technical Details**: 연구진은 보상 모델의 감춰진 표현(hidden representations)에서 바람직한 시퀀스와 바람직하지 않은 시퀀스를 대조 학습(contrastive learning) 목표로 사용합니다. 이 방법은 보상 모델이 문제의 올바른 해결책이나 도움이 되는 응답 등 특정 목표 상태로의 경로를 평가할 수 있게 해서, 오류를 로컬라이즈하거나 부분적으로 완료된 시퀀스를 평가하는 데 도움을 줍니다. 이렇게 훈련된 보상 모델의 표현은 목표 상태를 달성할 가능성을 평가할 수 있어, 다수결 투표에서 부정확한 경로를 필터링하고, 원하는 속성의 응답으로 유도할 수 있습니다.

- **Performance Highlights**: MATH와 GSM8k 같은 도전적인 벤치마크에서 최대 0.09 AUROC 성능 향상이 확인되었습니다. Helpful-Harmless 데이터셋에서는 2.3% 정확도 증가가 관찰되었습니다. 또한, 용이한 세밀한 제어가 가능해 Llama 3 모델의 유용성을 9.6% 향상시키고 복잡성을 21.6%까지 증가시켰습니다. 이러한 방법으로 수학적 추론의 경우 생성된 토큰의 최대 55%를 필터링하여 성능 향상을 이루었습니다.



### Phi-3 Safety Post-Training: Aligning Language Models with a "Break-Fix" Cyc (https://arxiv.org/abs/2407.13833)
- **What's New**: 최근 언어 모델 훈련의 혁신은 스마트폰에서 실행할 수 있을 정도로 작은 고성능 모델을 만드는 것이 가능하다는 것을 보여주었습니다. 이러한 모델들이 점점 더 많은 도메인에 배포됨에 따라, 인간의 선호도와 안전 고려사항에 맞추는 것이 중요해졌습니다. 이 보고서에서는 Phi-3 시리즈 언어 모델의 안전 정렬 방법론을 제시합니다.

- **Technical Details**: Microsoft는 Phi-3 시리즈의 언어 모델을 안전하게 정렬하기 위해 'break-fix' 주기를 사용했습니다. 이것은 데이터셋 큐레이션, 안전 후훈련, 벤치마킹, 레드 팀 활동 및 취약점 식별의 여러 라운드를 수행하여 다양한 피해 영역을 처리하는 과정을 포함합니다. Phi-3-mini (3.8B), Phi-3-small (7B) 및 Phi-3-medium (14B)과 같은 다양한 크기의 모델들이 포함되어 있으며, 특히 Phi-3-mini는 스마트폰에서도 실행이 가능하면서도 높은 성능을 자랑합니다.

- **Performance Highlights**: Phi-3 모델은 다양한 책임있는 AI 벤치마크에서 좋은 성능을 보였습니다. 레드 팀은 'low-skilled adversary'와 'intermediate adversary'의 두 가지 페르소나를 사용하여 모델을 테스트했습니다. 이를 통해 단일 또는 다중 턴 시나리오에서 모델이 생성하는 유해한 콘텐츠를 찾아냈습니다. 반복적인 'break-fix' 접근법은 일반적으로 단일 후훈련 작업으로 달성할 수 있는 것보다 더 많은 위험을 완화하는 데 성공했습니다.



### RDBE: Reasoning Distillation-Based Evaluation Enhances Automatic Essay Scoring (https://arxiv.org/abs/2407.13781)
- **What's New**: 최근 BERT 및 T5와 같은 소형 언어 모델(small language models)을 자동 에세이 평가(AES)에 적용하는 연구가 활발히 이루어졌지만, 대부분의 기존 연구는 단순히 점수를 출력하는 데 집중했습니다. 이에 반해, 이번 연구에서 우리는 Reasoning Distillation-Based Evaluation (RDBE)를 도입하여 모델 점수의 해석 가능성을 높이고 초기 추론을 통해 성능을 향상시키는 접근법을 제안합니다. RDBE는 대형 언어 모델(Large Language Model, LLM)로부터 생성된 추론을 사용하여 소형 언어 모델(Small Language Model, SLM)에 추론 능력을 제공하면서 학습됩니다.

- **Technical Details**: RDBE는 DREsSNew 데이터셋을 기반으로, 각 데이터 포인트에 대한 다양한 평가 기준으로 라마-3-70B(Llama-3-70B) 모델을 이용하여 점수를 생성하고 이에 대한 추론과 해석을 제공합니다. LLM로부터 원시 추론을 생성한 후, 이를 소형 언어 모델에 증류(distill)하여 학습하는 방식입니다. 시스템 메시지는 모델의 특정 역할과 지침을 포함하며, 사용자 메시지는 해당 주제와 에세이, 평가 기준, 점수를 순차적으로 포함합니다. Fine-tuning은 LongT5-Base 모델을 사용하여 이루어지며, AdamW 옵티마이저(Optimizer)와 교차 엔트로피 손실 함수(Cross-Entropy Loss Function)를 사용하여 학습합니다.

- **Performance Highlights**: 실험 결과, RDBE는 DREsSNew 데이터셋의 모든 평가 기준에서 경쟁 모델을 능가하며 state-of-the-art 성능을 달성했습니다. 특히 zero-shot LLM 생성 및 기본 finetuned 모델 생성 모두를 초과하는 결과를 보여주어 실용적인 해석 출력과 향상된 성능을 입증했습니다.



### On Pre-training of Multimodal Language Models Customized for Chart Understanding (https://arxiv.org/abs/2407.14506)
- **What's New**: 최근 연구들은 도메인 특화 작업을 위해 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 맞춤화하며 유망한 결과를 내놓았습니다. 특히 과학 차트 이해 분야에서 두각을 나타내고 있습니다. 본 연구에서는 이러한 MLLMs의 차트 이해를 향상시키기 위한 교육 과정을 탐구하였으며, 기존 연구들이 간과한 자연 이미지-캡션 사전 학습 데이터와 디지털 차트 이미지-QA 데이터 간의 근본적 차이점을 해결하고자 했습니다. 그 결과, 우리가 제안하는 CHOPINLLM은 다양한 유형의 차트를 효과적으로 해석하며, 새로운 벤치마크를 통해 MLLMs의 다양한 차트 유형에 대한 이해도를 평가했습니다.

- **Technical Details**: 본 연구는 차트 데이터를 이해하기 위해 필요한 세 가지 주요 발견을 제시합니다. (1) 원시 데이터 값을 정렬된 사전 학습에 포함시키는 것이 차트 데이터 이해를 현저히 향상시킵니다. (2) 엔드 투 엔드 미세 조정 중에 이미지 대신 텍스트 표현을 임의로 대체하여 언어 추론 능력을 차트 해석 기술로 전이시킵니다. (3) 미세 조정 과정에서 모델이 먼저 기본 차트 데이터를 추출하고 나서 질문에 답하도록 요구하면 정확성을 더욱 향상시킬 수 있습니다.

- **Performance Highlights**: CHOPINLLM은 주석이 달린 차트와 주석이 없는 다양한 유형의 차트를 이해하는 데 강력한 성능을 보였습니다. 실험 결과, CHOPINLLM은 폭넓은 차트 유형을 이해하는 능력에서 매우 뛰어난 성과를 나타냈습니다.



### AudioInsight: Detecting Social Contexts Relevant to Social Anxiety from Speech (https://arxiv.org/abs/2407.14458)
Comments:
          8 pages, 4 figures, 3 tables. Accepted by ACII 2024, Glasgow, UK. To appear in the Proceedings of ACII 2024

- **What's New**: 이 연구는 다양한 사회적 컨텍스트에서 발생하는 사회적 위협을 감지하기 위해 환경 오디오(ambient audio) 세그먼트를 활용하는 새로운 접근 방식을 제안합니다. 특히, 대화 참여자 수(dyadic vs. group)와 평가 위협의 정도(explicitly evaluative vs. not explicitly evaluative)를 중점적으로 분석합니다. 본 연구는 코로나19 팬데믹 중 Zoom 기반 가상 상호작용 데이터를 바탕으로 수행되었습니다.

- **Technical Details**: 연구에서는 deep learning 방법을 사용해 환경 오디오 데이터를 분석하여 사회적 컨텍스트를 감지합니다. Convolutional Neural Networks (CNNs)을 활용하여 스펙트럼 및 오디오 특징이 포함된 이미지를 처리합니다. 핵심 오디오 특징(pitch, jitter, shimmer 등) 26개를 분석하는데, 여기에는 MFCC, Spectral Centroid, Spectral Bandwidth 등도 포함됩니다. 데이터셋은 52명의 대학생으로 구성된 Zoom 대화 녹음에서 수집되었습니다. 추가로 모델 검증을 위해 5-fold Cross Validation과 leave-one-group-out Cross Validation 방법을 사용합니다.

- **Performance Highlights**: 연구 결과, dyadic과 group 상호작용을 구분하는 모델의 정확도는 90%였으며, 평가 위협을 감지하는 모델은 83%의 정확도를 보였습니다. leave-one-group-out Cross Validation에서는 각각 82%, 77%의 정확도를 기록했습니다. 이 연구는 AI와 passive sensing 기술이 사회적 컨텍스트를 효과적으로 감지할 수 있는 가능성을 보여주며, 디지털 인터벤션을 통해 맞춤형 정신 건강 지원으로 이어질 수 있음을 시사합니다.



### System-1.x: Learning to Balance Fast and Slow Planning with Language Models (https://arxiv.org/abs/2407.14414)
Comments:
          29 pages (10 tables)

- **What's New**: System-1.x Planner는 긴 계획 문제를 해결하기 위해 빠른 'System-1' 모드와 느린 'System-2' 모드를 혼합하여 사용할 수 있는 LLM 기반 프레임워크입니다. 이 프레임워크는 사용자의 목표에 따라 모델의 행동을 조절할 수 있는 컨트롤러, System-1 Planner, System-2 Planner로 구성됩니다.

- **Technical Details**: System-1.x Planner는 문제를 하위 목표로 분해하고, 이를 각각 쉬운 문제와 어려운 문제로 분류하여 System-1 또는 System-2 방식으로 해결합니다. 사용자 지정 하이브리드 정도 (hybridization factor, x)에 따라 계획을 조절하며, 이는 세 가지 구성 요소로 구성됩니다: (i) 컨트롤러, (ii) System-1 Planner, (iii) System-2 Planner. 모든 구성 요소는 단일 LLM을 기반으로 미세 조정되어 있으면, 외부 검색 도구나 검증자에 의존하지 않습니다.

- **Performance Highlights**: System-1.x Planner는 Maze Navigation과 Blocksworld라는 두 가지 다양한 계획 작업에서 System-1 Planner와 A* 검색을 모방한 System-2 Planner, 그리고 상징적 플래너(A*)보다 우수한 성능을 보였습니다. 주요 장점은 (1) 제어 가능성: 하이브리드 정도를 높이면 성능이 향상됨, (2) 유연성: 신경망 기반 System-1과 상징적 System-2를 조합한 네오-심볼릭 (neuro-symbolic) 변형을 통해 기존 상징적 방법을 통합할 수 있음, (3) 일반화 가능성: 다양한 검색 알고리즘에서 학습할 수 있어 방법의 강인성이 향상됨.



### The Vision of Autonomic Computing: Can LLMs Make It a Reality? (https://arxiv.org/abs/2407.14402)
- **What's New**: 비전 컴퓨팅 (Autonomic Computing Vision, ACV)의 비전은 20년 전 제안되어 생물학적 유기체처럼 환경 변화에 적응하는 자율 관리 컴퓨팅 시스템을 목표로 하고 있습니다. 이 논문은 대형 언어 모델 (Large Language Models, LLMs)을 활용하여 미세 서비스 관리 (microservice management)의 자율성을 달성할 수 있는 가능성을 탐구합니다. 특히, Sock Shop 마이크로서비스 데모 프로젝트를 활용한 온라인 평가 벤치마크를 통해 LLM 기반 다중 에이전트 프레임워크의 성능을 평가합니다.

- **Technical Details**: LLM 기반의 다중 에이전트 프레임워크를 제안하며, 이를 통해 마이크로서비스의 자율 관리를 시도합니다. 프레임워크는 선언적 작업을 수행하는 고수준 그룹 관리와 개별 서비스 구성 요소에 집중하는 저수준 자율 에이전트로 구성됩니다. 자율 서비스 유지 관리를 위한 5단계 분류법을 도입하고, 혼란 공학 기법을 사용하여 Sock Shop 마이크로서비스에서 오류를 일부러 도입한 후 시스템의 문제 해결 능력을 평가합니다.

- **Performance Highlights**: LLM 기반 다중 에이전트 프레임워크는 자율 유지 관리의 5단계 분류에서 수준 3 자율성을 달성했습니다. 문제 감지 및 특정 작업 수행에서 효과적이었으며, 특히 마이크로서비스 아키텍처에서 LLM의 잠재력을 강조했습니다. 그러나 원인 분석 및 문제 완화 능력에서 추가 개선의 여지가 있습니다.



### Improving Retrieval in Sponsored Search by Leveraging Query Context Signals (https://arxiv.org/abs/2407.14346)
Comments:
          8 pages, 8 tables, 1 figure

- **What's New**: 사용자 쿼리(query)에 대한 관련 키워드(keyword)를 정확하게 찾는 것은 스폰서드 검색(Sponsored Search)에서 매우 중요하지만, 특히 짧고 애매한 쿼리에서는 여전히 도전적인 과제입니다. 이를 해결하기 위해 웹 검색 결과와 대형 언어 모델(GPT-4)에서 파생된 풍부한 문맥 신호를 사용하여 쿼리 이해력을 향상시키는 접근 방식을 제안합니다. 이를 통해 쿼리를 실세계 정보에 기반을 두고, 사용자의 의도를 명확히 하는 쿼리 재작성 및 설명을 생성하여 온라인 캐시에 저장합니다.

- **Technical Details**: 우리의 접근 방식은 웹 검색 결과와 GPT-4를 이용하여 문맥 신호(context signals)를 생성하고 이를 온라인 캐시에 저장합니다. 구체적으로, 웹 검색의 제목(title)과 스니펫(snippet)을 사용하여 쿼리를 실세계 정보에 기반을 두고, GPT-4를 사용하여 쿼리 재작성(query rewrites) 및 설명을 생성합니다. 이 신호들을 효율적으로 통합하기 위해 Fusion-in-Decoder 기반 유니티( Unity) 아키텍처를 사용하여, 전통적인 모델과 유사한 비용으로 활용할 수 있습니다. 또한 캐시에 문맥 정보가 없는 경우를 대비하여, 'context glancing'이라는 커리큘럼 학습 전략을 도입해 모델의 강건성과 성능을 높였습니다.

- **Performance Highlights**: 오프라인 실험 결과, 문맥 인식 접근 방식은 문맥 비인식 모델 보다 성능이 19.9% 향상되었습니다. 또한 온라인 A/B 테스트에서는 160개 이상의 국가에서 사용자 참여도와 광고 수익이 크게 향상되었습니다. 특히 영어 쿼리에서 1%, 비영어 쿼리에서 1.4%의 수익 증가를 달성했습니다.



### Foundation Models for Autonomous Robots in Unstructured Environments (https://arxiv.org/abs/2407.14296)
Comments:
          arXiv admin note: text overlap with arXiv:2312.07843, arXiv:2402.05741 by other authors

- **What's New**: 이 연구에서는 사전학습된 기초 모델(pretrained foundation models)이 로봇을 비구조적 환경(unstructured environments)에 도입할 수 있는 잠재적인 솔루션으로 작용할 수 있는지에 대한 기회를 탐색하였습니다. 특히, 건설 현장과 같은 예측 불가능한 환경에서도 자동화를 구현하고자 하는 목표를 가지고 있습니다.

- **Technical Details**: 이 연구는 기초 모델의 로봇공학과 비구조적 환경에서의 응용을 체계적으로 검토하였고, 이를 심사숙고 행동 이론(deliberative acting theory)과 종합했습니다. 특히, 대규모 언어 모델(LLMs)의 언어적 능력이 인간-로봇 상호작용에서 인식을 개선하는 데 주로 활용되었습니다.

- **Performance Highlights**: 연구 결과, LLMs는 건설 프로젝트 관리와 안전, 재난 관리에서의 자연 재해 탐지 등에서 보다 많은 응용 사례를 보여주었습니다. 현재 기술의 최첨단 상태는 조건부 자동화 수준에 있으며, 미래 시나리오, 도전 과제 및 솔루션을 예측해보았습니다.



### Hierarchical Windowed Graph Attention Network and a Large Scale Dataset for Isolated Indian Sign Language Recognition (https://arxiv.org/abs/2407.14224)
- **What's New**: 이번 연구는 새로운 대규모 인도 수어 데이터셋과 이를 기반으로 한 수어 인식 모델을 제안합니다. 제안된 데이터셋은 20명의 청각 장애인이 사용한 40033개의 비디오로 구성된 2002개의 일상 단어를 포함하고 있습니다. 연구는 사람의 상반신 골격 그래프 구조를 활용한 'Hierarchical Windowed Graph Attention Network'(HWGAT)라는 새로운 수어 인식 모델을 도입했습니다.

- **Technical Details**: 제안된 모델, HWGAT는 사람 골격의 상반신을 기반으로 한 그래프 구조와 주의 메커니즘을 통해 각 신체 부위의 움직임을 효과적으로 캡처합니다. 동시에 입력 데이터를 윈도우 파티셔닝 함으로써 불필요한 신체 부분의 영향을 최소화하고 표현력 있는 손 동작을 처리합니다. 또한, 데이터셋 수집 및 주석 달기 과정을 자동화하는 파이프라인도 함께 공개되었습니다.

- **Performance Highlights**: HWGAT 모델은 다른 수어 인식 데이터셋에서 기존 최신 모델들보다 우수한 성능을 보여주었습니다. INCLUDE, LSA64, AUTSL, WLASL 데이터셋에서 각각 1.10, 0.46, 0.78, 6.84 퍼센티지 포인트 향상을 이루어냈습니다.



### Braille-to-Speech Generator: Audio Generation Based on Joint Fine-Tuning of CLIP and Fastspeech2 (https://arxiv.org/abs/2407.14212)
- **What's New**: 새로운 연구는 시각 장애인의 독서 효율성을 향상시키기 위한 노력의 일환으로, 중국어 환경에 맞춰 설계된 이미지-스피치 변환 프레임워크인 CLIP-KNN-Fastspeech2를 제안했습니다.

- **Technical Details**: 이 프레임워크는 두 단계로 이루어져 있습니다. 첫 번째 단계에서는 이미지-텍스트 변환을 위해 Contrastive Language-Image Pre-Training (CLIP) 모델과 K-Nearest Neighbor (KNN) 모델을 사용하고, 두 번째 단계에서는 Fastspeech2 모델을 사용하여 텍스트를 스피치로 변환합니다. CLIP과 Fastspeech2는 각각 MUGE와 Baker 공개 데이터셋으로 사전 훈련을 거친 후, 자체 구축한 점자 이미지 데이터셋을 사용해 결합 미세 조정을 하였습니다.

- **Performance Highlights**: 복수의 공용 데이터셋(VGGSound, Flickr8k, ImageHear)과 자체 구축한 데이터셋(BIT-DP) 실험 결과, 이 모델은 BLEU4, Fréchet Audio Distance (FAD), Word Error Ratio (WER)와 같은 객관적인 지표에서 개선을 보여주었으며, 추론 속도도 향상되었습니다. 이는 제한된 데이터 하에서도 고품질의 음성을 합성할 수 있음을 입증하며, 여러 기본 모델을 통합한 결합 훈련 전략이 효과적임을 증명합니다.



### PassTSL: Modeling Human-Created Passwords through Two-Stage Learning (https://arxiv.org/abs/2407.14145)
- **What's New**: 새로운 논문 PassTSL(모델링 인간이 만든 비밀번호를 위한 두 단계 학습)을 제안합니다. PassTSL은 최근 NLP와 딥러닝 모델에서 인기를 끌고 있는 사전훈련-미세조정(pretraining-finetuning) 프레임워크에서 영감을 받았습니다.

- **Technical Details**: PassTSL은 트랜스포머(self-attention mechanism)를 활용한 딥러닝 기반 비밀번호 모델로, 큰 규모의 일반적 데이터베이스에서 사전훈련을 수행하고, 대상 비밀번호 데이터베이스에 맞는 더 작은 특수 데이터베이스를 기반으로 미세조정을 합니다. 또한, JS Divergence를 기반으로 한 미세조정 비밀번호 선택 방법도 제안되었습니다.

- **Performance Highlights**: ['PassTSL은 여섯 개의 큰 유출 비밀번호 데이터베이스에 대해 실험한 결과, 5개의 최첨단(state-of-the-art, SOTA) 비밀번호 추측 방법들보다 최대 4.11%에서 64.69%까지 더 나은 성능을 보였습니다.', '단 0.1%의 추가 훈련 데이터로도 비밀번호 추측에서 평균적으로 3% 이상의 성능 향상을 가져왔습니다.', 'PassTSL 기반 비밀번호 강도 측정기(PSM)는 기존의 FLA 기반 및 zxcvbn PSM 보다 더 정확했습니다.']



### Domain-Specific Pretraining of Language Models: A Comparative Study in the Medical Field (https://arxiv.org/abs/2407.14076)
- **What's New**: 이 논문은 특정 분야에서 더 효율적인 방법으로 전이 학습(pretraining)을 사용한 전문화된 언어 모델을 제안합니다. 특히 의료 분야에서의 도메인 전이 학습(domain-specific pretraining)을 중점적으로 다루며, 일반 목적의 언어 모델과 비교하여 성능을 분석합니다.

- **Technical Details**: 논문은 다양한 학습 방법과 도메인 특화 데이터셋(domain-specific datasets)을 활용하여 전문화된 모델을 구축하는 방법을 설명합니다. 일반 전이 학습(general pretraining) 대신, 도메인 특화 데이터셋을 사용하여 모델이 관련 정보를 더 효과적으로 학습할 수 있도록 합니다. 특히, 부족한 데이터의 경우 일반 데이터셋(general-purpose datasets)으로 초기에 학습하고, 이후에 도메인 특화 데이터로 추가 학습을 진행하는 혼합 도메인 전이 학습(mixed-domain pretraining)이 제안됩니다.

- **Performance Highlights**: 도메인 전이 학습을 진행한 전문화된 언어 모델은 일반 목적의 언어 모델에 비해 특정 도메인에서 더욱 뛰어난 성능을 보입니다. 특히, 의료 분야에서 전문화된 모델이 일반 모델보다 더 정확한 결과를 도출할 수 있음을 실험적으로 확인했습니다. 그러나, 충분한 도메인 특화 데이터가 존재할 경우, 혼합 도메인 전이 학습보다 직접적인 도메인 전이 학습이 더 나은 성능을 보일 수 있습니다.



### Clinical Reading Comprehension with Encoder-Decoder Models Enhanced by Direct Preference Optimization (https://arxiv.org/abs/2407.14000)
- **What's New**: 최근 의료 텍스트에서의 추출적 질문 응답(Extractive Question Answering)을 효과적으로 수행하기 위해 인코더-디코더(encoder-decoder) 모델과 직접 선호 최적화(Direct Preference Optimization, DPO) 기법을 결합하여 RadQA 방사선 질문 응답 과제에서 이전의 최첨단 성능을 12-15 F1 포인트 향상시켰습니다. 특히 이 연구는 인간 입력 없이 선호 데이터를 생성하는 새로운 휴리스틱을 통해 DPO 방법이 독해 문제에서도 효과적임을 처음으로 입증했습니다.

- **Technical Details**: 의료 시스템에서 생성되는 방대한 양의 임상 텍스트를 효율적으로 처리하기 위해 우수한 도구가 필요하다는 점에서 이 연구는 방사선 질문 응답 과제(RadQA)에서 인코더-디코더 모델과 함께 최근 떠오른 DPO 기법을 사용했습니다. DPO는 보상을 제공하는 모델을 별도로 훈련하지 않고도 인간의 선호도를 학습하여 언어 모델을 최적화할 수 있는 방법으로, 주로 디코더 전용 LLMs에 사용되어 왔으나 이번 연구에서는 MRC 작업에 효과적임을 보였습니다. 이 접근법을 통해 기존 인코더 전용 모델 대비 10% 이상의 F-score 향상을 보였으며, 자동으로 생성된 정답-오답 쌍 데이터를 이용해 추가적으로 1-3% F1 포인트 향상을 달성했습니다.

- **Performance Highlights**: RadQA 데이터셋에서 실행된 실험에서, 인코더-디코더 모델과 DPO를 결합한 방식은 기존의 최첨단 모델에 비해 12-15 F1 포인트 개선된 성능을 보였습니다. 이는 최신의 언어 모델과 최적화 방법론을 활용해 임상 텍스트 독해 문제에서 중요한 성과를 이룬 것입니다. 실험 코드와 생성된 선호 데이터는 논문 채택 시 GitHub에 공유될 예정입니다.



### Harmful Suicide Content Detection (https://arxiv.org/abs/2407.13942)
Comments:
          30 pages, 7 figures

- **What's New**: 인터넷상의 자살 유해 콘텐츠는 취약 계층에게 자살 생각과 행동을 유발하는 중요한 위험 요소입니다. 이러한 문제에 대해 글로벌 차원에서 많은 노력이 있지만, 특히 대한민국과 같은 고위험 지역에서는 자원이 부족합니다. 현재 연구는 주로 이러한 콘텐츠의 부정적인 영향이나 개인의 자살 위험 이해에 중점을 두고 있으며, 콘텐츠의 유해성을 자동으로 감지하는 것에 초점을 두지 않았습니다. 이를 해결하기 위해 온라인 자살 콘텐츠를 다섯 가지 유해성 수준으로 분류하는 '유해 자살 콘텐츠 검출' 과제를 소개합니다.

- **Technical Details**: 우리는 다중 모달(멀티모달, multi-modal) 벤치마크와 의학 전문가와 협력하여 작성된 과제 설명 문서를 개발했습니다. 또한 대형 언어 모델(LLMs, Large Language Models)을 활용하여 이러한 콘텐츠를 효율적으로 모니터링하는 방법을 탐구했습니다. 우리의 기여사항에는 새로운 검출 과제 제안뿐만 아니라 전문가의 주석이 포함된 한국어 다중 모달 벤치마크 개발 및 LLM를 사용하여 불법 및 유해 콘텐츠를 감지하는 전략 제안이 포함됩니다.

- **Performance Highlights**: 잠재적인 피해를 고려하여, 우리는 구현 및 벤치마크를 공개하며 윤리적 검증 과정을 포함했습니다.



### Less is More: Sparse Watermarking in LLMs with Enhanced Text Quality (https://arxiv.org/abs/2407.13803)
- **What's New**: 최근 논문에서는 Sparse Watermark라는 새로운 타입의 LLM 워터마크가 소개되었습니다. 이는 기존 방법들이 직면한 텍스트 품질과 워터마크 효과성 간의 트레이드-오프를 해결하고자 고안되었습니다. Sparse Watermark는 생성된 텍스트의 일부 토큰에만 워터마크를 적용하여 텍스트 품질을 유지하면서도 높은 검출 가능성을 갖추도록 설계되었습니다.

- **Technical Details**: Sparse Watermark의 핵심 아이디어는 특정 품사 태그(Part-of-Speech tags)를 가진 단어에 워터마크 토큰을 삽입하는 것입니다. 이를 통해 텍스트의 자연스러움을 최대한 유지하면서도 소규모, 고정밀 워터마크 삽입이 가능합니다. 실험 결과, Sparse Watermark는 다양한 생성 작업에서 텍스트 품질과 워터마크 검출 가능성 두 가지를 모두 높은 수준으로 달성했습니다.

- **Performance Highlights**: 제안된 Sparse Watermark 기법은 다양한 과업에서 이전 워터마크 방법보다 높은 품질의 텍스트를 생성하면서도 검출 가능성이 뛰어납니다. 고정된 품사 태그를 기반으로 워터마크를 삽입할 때 품질 저하 없이 워터마크를 효과적으로 검출할 수 있음을 보여주었습니다.



### Continuous Embedding Attacks via Clipped Inputs in Jailbreaking Large Language Models (https://arxiv.org/abs/2407.13796)
- **What's New**: 이 연구는 연속적인 embedding을 통해 LLM(대형 언어 모델)에서 탈옥(jailbreak) 가능성을 탐구합니다. 기존의 방법은 이산적(discrete) 또는 연속적(continuous) 접미사(suffixes)를 입력에 추가하는 방식을 사용했지만, 이 연구는 접미사나 특정 질문 없이도 직접적인 공격이 가능함을 보여줍니다. 주목할 만한 점은 반복적인 출력(overfitting)을 피하는 효과적인 전략으로 CLIP을 제안한 것입니다.

- **Technical Details**: 연구에는 두 가지 주요 도전 과제가 있습니다: 첫째, 무작위 패턴을 피하여 입력을 샘플링하는 방법, 둘째, 높은 반복 횟수에서 발생하는 과적합(overfitting) 문제를 해결하는 것입니다. CLIP 전략은 모델의 어휘 평균 값을 기준으로 입력을 투영하는 간단한 방법으로, 높은 차원의 공간에서 모델의 과적합과 무작위 변동성을 줄여줍니다. 백박스(white-box) 설정에서 사용자는 모델에 대한 모든 접근 권한을 가지며, 특정 악성 출력을 생성하기 위해 입력을 최적화합니다.

- **Performance Highlights**: CLIP을 사용한 실험 결과, 입력 길이가 40이고 1000번의 반복에서 ASR(Attack Success Rate)가 62%에서 83%로 향상됨을 확인했습니다. 이 방법은 LLama와 Vicuna 모델에서 효과적으로 검증되었습니다.



### The Honorific Effect: Exploring the Impact of Japanese Linguistic Formalities on AI-Generated Physics Explanations (https://arxiv.org/abs/2407.13787)
- **What's New**: 이 연구는 일본어 경어(honorifics)가 대형 언어 모델(LLM)의 운동량 보존 법칙에 대한 설명에 미치는 영향을 조사합니다. ChatGPT, Coral, Gemini와 같은 최신 AI 모델 6개를 분석한 결과, 경어가 AI의 응답의 품질, 일관성 및 형식성에 중요한 영향을 미치는 것을 발견했습니다. 이는 교육적 맥락에서 경어를 사용하여 AI 생성 설명의 깊이와 복잡성을 조정할 수 있음을 시사합니다.

- **Technical Details**: 연구에서는 ChatGPT 3.5 터보, ChatGPT 4.0 터보, Coral (Command R+), Gemini 1.0 Pro, Gemini 1.5 Flash, Gemini 1.5 Pro와 같은 6개 AI 모델을 사용해 14개의 서로 다른 일본어 경어 형식을 통해 데이터를 수집했습니다. 각각의 AI 모델에 대해 '운동량 보존 법칙을 설명해주세요'라는 질문을 한 후, 14종의 경어 스타일 및 주소를 50번씩 반복하여 응답을 분석했습니다. 다양한 경어 형식은 사회적 계층과 존중의 수준에 따라 선정되었습니다.

- **Performance Highlights**: 분석 결과, 경어의 사용은 AI 응답의 구조와 형식성에 큰 영향을 미쳤습니다. 예를 들어, 긴 문장과 짧은 문장의 사용 비율, 정중한 표현과 덜 정중한 표현 등의 비율을 통해 경어의 영향력을 측정했습니다. 이는 AI 모델이 사회적 맥락 단서를 해석하고 적응하는 능력을 가지고 있음을 나타내며, 교육 도구로서의 AI 응용에 문화적 요소를 고려하는 것이 중요함을 강조합니다.



### Generative Model for Small Molecules with Latent Space RL Fine-Tuning to Protein Targets (https://arxiv.org/abs/2407.13780)
Comments:
          12 pages, 6 figures, Proceedings of the ICML 2024 Workshop on Accessible and Effi- cient Foundation Models for Biological Discovery, Vienna, Austria. 2024

- **What's New**: 새로운 분자 생성 모델을 제안하는 이 논문은 SAFE라는 최근 제안된 분자 문자열 표현을 활용하여 분자의 문법적으로 유효하고 화학적으로 타당한 표현을 생성하는 문제를 해결합니다. SAFE의 개선된 버전인 SAFER를 사용하여 학습 중 생성되는 유효하지 않은 단편화 분자의 수를 줄였습니다.

- **Technical Details**: 새 모델은 변환기 아키텍처 내에서 변분 오토인코더(VAE)를 통해 구성된 잠재 변수 생성 모델입니다. SAFER tokenization을 사용하여 입력 SMILES를 정규화하고 단편을 분해할 때 분자량으로 정렬하지 않고 원래 시퀀스의 순서를 유지합니다. 또한, 개방 및 폐쇄 고리 원자를 나타내기 위해 새로운 토큰을 도입했습니다. 이러한 방법론은 특히 분자의 일관된 순서와 숫자 토큰의 필요성을 줄이는 데 기여합니다.

- **Performance Highlights**: 제안된 모델은 잠재 공간에서 샘플링함으로써 유효성 비율이 90% 이상, 단편화 비율이 1% 미만인 새로운 분자를 생성할 수 있으며, 강화 학습(RL) 세부 조정을 통해 분자 도킹 성능을 개선하여 다섯 개의 특정 단백질 타겟에 대해 히트 후보의 수를 크게 증가시켰습니다. 또한, 상위 5%의 평균 도킹 점수는 현재의 최첨단 (SOTA)와 비교할 수 있으며, 다섯 개의 타겟 중 세 개에서는 SOTA를 조금 능가했습니다.



New uploads on arXiv(cs.IR)

### Improving Retrieval in Sponsored Search by Leveraging Query Context Signals (https://arxiv.org/abs/2407.14346)
Comments:
          8 pages, 8 tables, 1 figure

- **What's New**: 사용자 쿼리(query)에 대한 관련 키워드(keyword)를 정확하게 찾는 것은 스폰서드 검색(Sponsored Search)에서 매우 중요하지만, 특히 짧고 애매한 쿼리에서는 여전히 도전적인 과제입니다. 이를 해결하기 위해 웹 검색 결과와 대형 언어 모델(GPT-4)에서 파생된 풍부한 문맥 신호를 사용하여 쿼리 이해력을 향상시키는 접근 방식을 제안합니다. 이를 통해 쿼리를 실세계 정보에 기반을 두고, 사용자의 의도를 명확히 하는 쿼리 재작성 및 설명을 생성하여 온라인 캐시에 저장합니다.

- **Technical Details**: 우리의 접근 방식은 웹 검색 결과와 GPT-4를 이용하여 문맥 신호(context signals)를 생성하고 이를 온라인 캐시에 저장합니다. 구체적으로, 웹 검색의 제목(title)과 스니펫(snippet)을 사용하여 쿼리를 실세계 정보에 기반을 두고, GPT-4를 사용하여 쿼리 재작성(query rewrites) 및 설명을 생성합니다. 이 신호들을 효율적으로 통합하기 위해 Fusion-in-Decoder 기반 유니티( Unity) 아키텍처를 사용하여, 전통적인 모델과 유사한 비용으로 활용할 수 있습니다. 또한 캐시에 문맥 정보가 없는 경우를 대비하여, 'context glancing'이라는 커리큘럼 학습 전략을 도입해 모델의 강건성과 성능을 높였습니다.

- **Performance Highlights**: 오프라인 실험 결과, 문맥 인식 접근 방식은 문맥 비인식 모델 보다 성능이 19.9% 향상되었습니다. 또한 온라인 A/B 테스트에서는 160개 이상의 국가에서 사용자 참여도와 광고 수익이 크게 향상되었습니다. 특히 영어 쿼리에서 1%, 비영어 쿼리에서 1.4%의 수익 증가를 달성했습니다.



### L^2CL: Embarrassingly Simple Layer-to-Layer Contrastive Learning for Graph Collaborative Filtering (https://arxiv.org/abs/2407.14266)
- **What's New**: 새롭게 발표된 논문은 레이어 간 대조 학습(Layer-to-Layer Contrastive Learning) 기법인 L2CL을 제안하고 있습니다. 이를 통해 그래프 신경망(GNN, Graph Neural Network) 기반의 추천 시스템에서 발생하는 노이즈 문제와 과도한 계산 복잡성을 해결합니다. 기존의 그래프 대조 학습(GCL, Graph Contrastive Learning) 접근법은 무작위 데이터 증강으로 인해 중요한 정보를 손상시키거나 불필요한 노이즈를 도입하는 문제를 겪었는데, L2CL은 이를 해결하고 보다 효과적인 사용자 및 아이템 표현을 학습할 수 있도록 합니다.

- **Technical Details**: L2CL은 그래프 네트워크의 각 레이어 간 표현을 대조하여 학습합니다. 즉, 노드와 이웃 노드 간의 구조적 유사성을 맞추는 목표를 가지고 있습니다. 데이터 증강 없이 각각의 레이어에서 의미 있는 구조적 관계를 캡처하고, 한 레이어의 출력 값을 다른 레이어의 출력 값과 대조하여 전체 구조 및 의미 유사성을 학습합니다. 이를 통해 무작위 증강이 도입한 잠재적 노이즈를 제거하고, 1-hop 대조 학습 패러다임을 사용하여 노드 표현의 질을 향상시킵니다.

- **Performance Highlights**: L2CL은 다섯 가지 실제 데이터셋에서 광범위한 실험을 통해 기존의 다양한 협업 필터링(CF, Collaborative Filtering) 방법들을 능가하는 성능을 입증했습니다. 단일 레이어 GNN과 1-hop 대조 학습만으로도 높은 질의 노드 표현을 유지하면서 효율성 또한 확보할 수 있었습니다.



### User-Creator Feature Dynamics in Recommender Systems with Dual Influenc (https://arxiv.org/abs/2407.14094)
- **What's New**: 최근 연구는 추천 시스템이 사용자와 창작자 모두에게 미치는 '이중 영향'을 조사하는 새로운 모델인 사용자-창작자 특성 동역학(user-creator feature dynamics)을 소개합니다. 이 모델은 추천 시스템이 사용자 선호도와 창작자 콘텐츠 스타일 모두에 영향을 미칠 수 있음을 정의합니다.

- **Technical Details**: 사용자-창작자 특성 동역학은 사용자의 선호도와 창작자의 콘텐츠 스타일을 나타내는 임베딩 벡터(embedding vectors)를 사용합니다. 코사인 유사도(cosine similarity)를 통해 사용자와 창작자의 관련성을 평가합니다. 이를 통해, 추천 시스템의 디자인 선택이 다양성에 어떤 영향을 미치는지 분석할 수 있습니다.

- **Performance Highlights**: 이중 영향이 있는 추천 시스템은 극단적으로 치우칠(polarize) 가능성이 높아져, 시스템 내의 다양성이 거의 제로에 가까워질 수 있음을 이론적으로 증명했습니다. 흥미롭게도, 추천 시스템의 효율성을 높이는 일반적인 방법인 top-k 추천 방식이 오히려 다양성을 유지하고 극단화를 방지하는 데 도움이 될 수 있음을 실험적으로 확인했습니다.



### Clinical Reading Comprehension with Encoder-Decoder Models Enhanced by Direct Preference Optimization (https://arxiv.org/abs/2407.14000)
- **What's New**: 최근 의료 텍스트에서의 추출적 질문 응답(Extractive Question Answering)을 효과적으로 수행하기 위해 인코더-디코더(encoder-decoder) 모델과 직접 선호 최적화(Direct Preference Optimization, DPO) 기법을 결합하여 RadQA 방사선 질문 응답 과제에서 이전의 최첨단 성능을 12-15 F1 포인트 향상시켰습니다. 특히 이 연구는 인간 입력 없이 선호 데이터를 생성하는 새로운 휴리스틱을 통해 DPO 방법이 독해 문제에서도 효과적임을 처음으로 입증했습니다.

- **Technical Details**: 의료 시스템에서 생성되는 방대한 양의 임상 텍스트를 효율적으로 처리하기 위해 우수한 도구가 필요하다는 점에서 이 연구는 방사선 질문 응답 과제(RadQA)에서 인코더-디코더 모델과 함께 최근 떠오른 DPO 기법을 사용했습니다. DPO는 보상을 제공하는 모델을 별도로 훈련하지 않고도 인간의 선호도를 학습하여 언어 모델을 최적화할 수 있는 방법으로, 주로 디코더 전용 LLMs에 사용되어 왔으나 이번 연구에서는 MRC 작업에 효과적임을 보였습니다. 이 접근법을 통해 기존 인코더 전용 모델 대비 10% 이상의 F-score 향상을 보였으며, 자동으로 생성된 정답-오답 쌍 데이터를 이용해 추가적으로 1-3% F1 포인트 향상을 달성했습니다.

- **Performance Highlights**: RadQA 데이터셋에서 실행된 실험에서, 인코더-디코더 모델과 DPO를 결합한 방식은 기존의 최첨단 모델에 비해 12-15 F1 포인트 개선된 성능을 보였습니다. 이는 최신의 언어 모델과 최적화 방법론을 활용해 임상 텍스트 독해 문제에서 중요한 성과를 이룬 것입니다. 실험 코드와 생성된 선호 데이터는 논문 채택 시 GitHub에 공유될 예정입니다.



### Knowledge Distillation Approaches for Accurate and Efficient Recommender System (https://arxiv.org/abs/2407.13952)
Comments:
          Doctoral Dissertation (2022)

- **What's New**: 본 논문은 지식 증류(KD, Knowledge Distillation) 방법을 추천 시스템에 적용하여 더 나은 성능을 발휘하는 컴팩트 모델을 개발하는 데 중점을 두고 있습니다. 기존의 분류 문제에서 지식 증류의 돌파구가 있었으나, 추천 모델과 랭킹 문제에 대한 연구는 부족했습니다. 이 논문은 추천 시스템을 위한 새로운 지식 증류 방법을 제안하고, 이를 통해 성능을 극대화하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 지식 소스에 따라 분류됩니다: (1) 잠재 지식(Latent Knowledge): 사용자/아이템 표현의 잠재 지식을 전이하는 두 가지 방법을 제안합니다. 이는 균형 잡힌 증류 전략을 통해 소수의 대형 선호 그룹에 치우치지 않도록 하여, 특수한 취향의 지식을 효과적으로 전이합니다. 또한, 사용자/아이템 관계를 표현 공간에서 전이하는 새로운 방법도 제안합니다. 이 방법은 컴팩트 모델의 제한된 용량을 고려하여 필수적인 관계를 선택적으로 전이합니다. (2) 랭킹 지식(Ranking Knowledge): 추천 결과에서 랭킹 지식을 전이하는 세 가지 방법을 제안합니다. 이는 리스트 방식의 학습 전략을 통해 KD 과정을 랭킹 매칭 문제로 공식화합니다. 추가로, 이종 추천 모델의 랭킹 지식을 압축하는 새로운 학습 프레임워크를 제시합니다. 이 프레임워크는 대부분의 추천 응용 프로그램에 대한 모델 앙상블의 계산 부담을 줄이기 위해 개발되었습니다.

- **Performance Highlights**: 제안된 방법과 프레임워크의 장점을 다양한 실험을 통해 검증했습니다. 요약하자면, 이 논문은 추천 모델의 정확성과 효율성 간의 적절한 균형을 확보하기 위해 지식 증류 접근법의 중요성을 강조하고 있습니다.



### PRAGyan -- Connecting the Dots in Tweets (https://arxiv.org/abs/2407.13909)
Comments:
          9 pages, ASONAM

- **What's New**: 이번 연구는 사회 미디어 플랫폼 특히, X(전 트위터)에서 발생하는 사건과 발언에 대한 인과 분석을 수행하기 위해 Knowledge Graphs(KGs)와 Large Language Models(LLMs)를 통합하는 방법을 탐구합니다. 연구는 PRAGyan이라는 KG 기반 Neo4j 데이터를 포함한 Retrieval-Augmented Generation(RAG) 모델을 이용하여 맥락적 인과 추론을 수행합니다. 이를 통해 GPT-3.5 Turbo 모델보다 더욱 향상된 결과를 도출하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 KGs와 LLMs를 결합하여 트윗 데이터셋의 인과 분석을 수행합니다. KGs는 도메인 특정 지식의 구조적 표현을 저장하며, 노드와 링크로 데이터를 나타내어 풍부한 의미적 관계를 캡처합니다. Neo4j를 사용하여 시간적 정보를 포함한 복잡한 관계를 관리하며, Node2Vec 및 Sentence-BERT를 사용하여 그래프와 텍스트의 인코딩을 최적화합니다. PRAGyan 모델은 RAG 방식을 사용하여 관련 문서나 지식 스니펫을 검색하고 이에 기반해 응답을 생성합니다.

- **Performance Highlights**: 연구 결과, PRAGyan 모델은 블루(BLEU) 및 코사인 유사도(cosine similarity)와 같은 정량적 지표에서 GPT-3.5 Turbo 모델을 10% 초과하여 성능을 향상시켰습니다. 질적 분석에서도 KGs와 LLMs의 결합이 해석 가능성 및 실질적 인사이트를 향상시키며, 기업과 정책 결정자들이 더 나은 결정을 내리는 데 도움을 주는 것으로 나타났습니다.



### ChatQA 2: Bridging the Gap to Proprietary LLMs in Long Context and RAG Capabilities (https://arxiv.org/abs/2407.14482)
- **What's New**: 이 논문에서는 ChatQA 2 모델을 소개합니다. 이 모델은 Llama3 기반으로 설계되어, 오픈액세스 LLM(Open Access Large Language Model)과 선도적인 독점 모델들 (예: GPT-4-Turbo) 사이의 격차를 해소하기 위해 제작되었습니다. 특히 긴 문맥 이해와 검색 증강 생성(RAG)에 초점을 맞추고 있습니다.

- **Technical Details**: Llama3-70B-base 모델의 문맥 창을 8K에서 128K 토큰으로 확장하기 위한 상세한 추가 훈련 레시피를 제시합니다. 또한, 명령어 준수, RAG 성능, 긴 문맥 이해 능력을 향상시키기 위한 3단계 명령어 튜닝 프로세스를 도입하였습니다.

- **Performance Highlights**: Llama3-ChatQA-2-70B 모델은 여러 긴 문맥 이해 작업에서 GPT-4-Turbo-2024-0409과 비교할만한 정확도를 달성했으며, RAG 벤치마크에서 더 나은 성능을 보여줍니다. 최신 긴 문맥 검색기(Retriever)를 활용해 RAG 기반 결과를 개선했습니다.



### PolySinger: Singing-Voice to Singing-Voice Translation from English to Japanes (https://arxiv.org/abs/2407.14399)
Comments:
          This paper was accepted at ISMIR 2024

- **What's New**: 이번 연구는 노래 음성에서 노래 음성으로의 번역(SV2SVT)의 필요성을 강조하고, 이를 위해 PolySinger라는 시스템을 제안합니다. PolySinger는 세계 최초로 영어 가사를 일본어로 번역하는 노래 음성 번역 시스템입니다. 이를 통해 노래 가사 번역의 복잡한 문제를 해결하려는 시도를 합니다.

- **Technical Details**: PolySinger는 여러 Music Information Retrieval(MIR) 기술을 연속적으로 사용하여 가사 번역, 음소 레벨의 가사 정렬, 프레임 수준의 보컬 멜로디 추출, 자동 가사 번역, 그리고 노래 음성 합성을 수행합니다. 이 연속적인 접근 방식은 모듈형 프레임워크를 통해 연구를 용이하게 합니다. Whisper 모델을 사용하여 자동 가사 전사를 수행하고, 고해상도의 프레임 레벨 VME 시스템을 도입해 음소 레벨의 가사 정렬을 합니다. 또한, nllb-200 모델을 활용해 영어에서 일본어로 가사 번역을 수행합니다.

- **Performance Highlights**: PolySinger의 성능은 일본 원어민을 대상으로 Mean Opinion Score(MOS) 테스트를 통해 평가되었습니다. 결과는 SV2SVT의 기본 구조가 유망하다는 것을 보여주지만, 번역된 일본어 가사가 이상적인 자연스러움에 도달하지 못했다는 것도 밝혀졌습니다. 이는 향후 연구와 개선이 필요한 부분입니다.



### Multimodal Misinformation Detection using Large Vision-Language Models (https://arxiv.org/abs/2407.14321)
Comments:
          Accepted for publication in: Conference on Information and Knowledge Management (CIKM) 2024

- **What's New**: 최근 대형 언어 모델(LLMs)이 다양한 작업에서 뛰어난 성능을 보이고 있지만, 여전히 미정보(misinformation) 탐지에 있어서는 충분히 탐구되지 않았습니다. 본 논문에서는 LLMs가 미정보 탐지에 어떻게 도움이 될 수 있는지에 대해 제로샷 학습(zero-shot setting) 환경에서 탐구하였습니다. 새로운 재랭킹 방법을 통해 멀티모달(multimodal) 증거 탐색을 수행하고, LLMs와 대형 비전-언어 모델(LVLM)을 활용해 증거를 재랭킹하고 사실 확인(fact verification) 작업을 개선하는 접근법을 제안했습니다.

- **Technical Details**: 이 연구에서는 멀티모달 미정보 탐지를 위한 파이프라인을 소개하며, 특히 증거 탐색(evidence retrieval)에 대한 신규 재랭킹 접근방식을 제안합니다. 이 접근방식에서는 LLMs와 LVLMs를 활용하여 텍스트 및 이미지 증거를 재랭킹합니다. 이 후, 멀티모달 사실 확인을 위해 LVLM4FV라는 접근법을 적용하며, 프로밍 전략을 통해 텍스트와 이미지 증거를 분류기로 사용합니다. 또한, 기존 증거 탐색 데이터셋에서 불완전한 증거 샘플을 보완하여 공정한 평가를 위해 새로운 주석 작업을 수행하였습니다.

- **Performance Highlights**: 실험 결과는 두 개의 데이터셋에서 제안된 접근법이 증거 탐색과 사실 확인 작업에서 우수한 성능을 보임을 입증하였습니다. 이를 통해 제안된 접근법이 감독 학습 기반의 기존 방법들보다 데이터셋 간의 일반화 능력에서도 뛰어나다는 것을 보여주었습니다.



### Guitar Chord Diagram Suggestion for Western Popular Music (https://arxiv.org/abs/2407.14260)
- **What's New**: 기타 연주자들이 사용하는 코드 다이어그램은 초보자나 노래의 손 위치를 공유하는 데 유용합니다. 그러나 기존의 기타 학습 도구에서 제공되는 다이어그램은 실제 연주자가 사용하는 위치를 잘 반영하지 못합니다. 이 논문에서는 코드 다이어그램을 제안하는 도구를 소개하며, 이전 코드의 다이어그램을 고려한 새로운 방법을 제안합니다.

- **Technical Details**: DadaGP 및 mySongBook 데이터셋의 통계 분석을 기반으로, 일부 코드 다이어그램이 서양 대중 음악에서 과대 대표되고 있음을 보여줍니다. 또한 한 코드가 20가지 이상의 다양한 방법으로 연주될 수 있음을 확인했습니다. 맥락을 고려해 코드 다이어그램 제안을 개선할 수 있음을 주장하며, 현재 코드만을 고려하는 모델과 비교합니다. 이전 맥락을 추가하면 이 작업의 F1-score가 최대 27% 향상되고 표준 개방 코드 제안 확률이 줄어듭니다. 코드 다이어그램의 맥락에서 '텍스처(texture)'의 개념을 정의하고 다양한 메트릭스를 통해 우리의 모델이 이전 다이어그램과의 텍스처 일관성을 개선함을 보여줍니다.

- **Performance Highlights**: 우리 모델은 F1-score를 최대 27%까지 향상시키며, 표준 개방 코드 제안 확률을 줄입니다. 또한, 텍스처 일관성을 통해 코드 다이어그램 제안의 품질을 크게 향상시킵니다.



### DisenSemi: Semi-supervised Graph Classification via Disentangled Representation Learning (https://arxiv.org/abs/2407.14081)
Comments:
          Accepted by IEEE Transactions on Neural Networks and Learning Systems (TNNLS 2024)

- **What's New**: 그래프 분류(graph classification)는 다양한 멀티미디어 응용에서 중요한 작업입니다. 하지만 실제 환경에서는 레이블이 있는 그래프 데이터가 제한적일 수 있습니다. 이에 따라 논문에서는 레이블이 있는 데이터와 없는 데이터를 동시에 학습하는 반-지도 학습(semi-supervised learning) 문제를 다룹니다. 최근 접근 방식이 전체 지식을 전이하는 것과 달리, 우리는 지도 학습과 잘 맞는 관련 의미만을 전이해야 한다고 주장합니다. 이를 위해 새로운 프레임워크 DisenSemi를 제안합니다. DisenSemi는 반-지도 그래프 분류를 위해 비틀림된(disentangled) 표현을 학습합니다.

- **Technical Details**: DisenSemi는 비틀림된 그래프 인코더(disentangled graph encoder)를 사용하여 요인별 그래프 표현을 생성합니다. 이 인코더는 감독(지도) 학습 목표 및 상호 정보(MI) 기반 제약을 통해 두 모델을 학습시킵니다. MI 기반 비틀림 일관성 규제(disentangled consistency regularization)를 도입하여 비지도 인코더에서 지도 인코더로 의미 있는 지식 이전을 보장합니다. 또한 DisenSemi 프레임워크는 최대 가능도 추정을 통해 해결할 수 있는 문제로 공식화할 수 있습니다.

- **Performance Highlights**: 공공 데이터셋을 사용한 실험 결과, DisenSemi가 높은 효율성과 뛰어난 해석 가능성을 제공함을 보여줍니다. 이는 기존 반-지도 그래프 분류 방법보다 우수한 성능을 입증합니다.



### HeCiX: Integrating Knowledge Graphs and Large Language Models for Biomedical Research (https://arxiv.org/abs/2407.14030)
Comments:
          8 pages, 3 figures, under review

- **What's New**: 이번 연구에서는 HeCiX-KG (Hetionet-Clinicaltrials neXus Knowledge Graph)을 도입하여 임상 시험 및 생물학적 데이터를 종합적으로 통합한 새로운 지식 그래프를 소개했습니다. HeCiX-KG는 ClinicalTrials.gov와 Hetionet의 데이터를 하나로 합친 것으로, 이를 통해 임상 시험의 실패율을 감소시키기 위한 보다 철저한 자원을 제공합니다. 또한, HeCiX라는 시스템을 개발하여 LangChain을 사용해 HeCiX-KG와 GPT-4(일반적인 언어 모델)를 통합하였습니다.

- **Technical Details**: HeCiX-KG는 ClinicalTrials.gov와 Hetionet의 데이터를 통합하여 만들어졌으며, 6가지 특정 질병(백반증, 아토피 피부염, 원형 탈모증, 흑색종, 간질, 갑상선 기능 저하증)에 대한 데이터를 포함합니다. 총 6,509개의 노드와 14,377개의 엣지를 가진 지식 그래프로 구성됩니다. HeCiX 시스템은 LangChain의 GraphCypherQAChain를 사용하여 자연어 쿼리를 Cypher Query Language(CQL)로 변환하고 이를 HeCiX-KG에 실행한 후, 결과를 인간이 이해할 수 있는 형식으로 제시합니다.

- **Performance Highlights**: HeCiX 시스템은 임상 연구의 다양한 질문에 대해 높은 성능을 보였습니다. 질문 답변 작업에 대한 평가 결과, 모델은 신뢰성, 답변의 관련성, 정확한 컨텍스트와 회수율 등의 측면에서 상당한 향상을 보여주었습니다. 이는 임상 연구의 효율성을 높이고 약물 재창출 및 개발 성공률을 향상시키는 데 유용한 도구가 될 수 있습니다.




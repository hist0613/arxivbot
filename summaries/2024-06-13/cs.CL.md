### Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing (https://arxiv.org/abs/2406.08464)
Comments:
          Link: this https URL

- **What's New**: 대규모 언어 모델(LLMs)의 정렬(Alignment)을 개선하기 위한 고품질 지시 데이터(instruction data)가 필요하지만, 기존 모델들의 정렬 데이터는 비공개로 유지되고 있어 AI의 민주화에 장애가 되고 있습니다. Magpie라는 새로운 자가 합성 방법(self-synthesis method)이 이러한 문제를 해결하기 위해 등장했습니다. 이 방법은 LLM에서 직접 고품질의 지시 데이터를 대규모로 생성할 수 있는 방법을 제안합니다.

- **Technical Details**: Magpie는 주로 두 단계로 구성됩니다: 1) 지시 생성(instruction generation)과 2) 응답 생성(response generation). 이 방법은 사람의 개입 없이 전적으로 자동화할 수 있습니다. Magpie는 사전 정의된 지시 템플릿(predefined instruction template)의 형식에 맞춰 입력 쿼리를 작성하고, LLM에 그 쿼리를 여러번 보내 다양한 지시를 생성합니다. 이후, 생성된 지시에 대해 LLM이 적절한 응답을 생성합니다.

- **Performance Highlights**: Magpie는 두 개의 대규모 데이터셋 Magpie-Air와 Magpie-Pro를 제작했으며, 이 데이터셋을 통해 미세 조정된 LLM 모델은 높은 성능을 보였습니다. 특히, Magpie 데이터셋으로 미세 조정된 모델은 기존의 공식 Llama-3-8B-Instruct 모델과 유사한 성능을 보였으며, 일부 작업에서는 이를 능가하기도 했습니다. 다양한 정렬 벤치마크(AlpacaEval, ArenaHard, WildBench)에서도 우수한 성능을 입증했습니다.



### OLMES: A Standard for Language Model Evaluations (https://arxiv.org/abs/2406.08446)
- **What's New**: AI 모델의 평가 방법에 있어 더 투명하고 재현 가능한 표준을 제안하는 OLMES(Open Language Model Evaluation Standard)가 등장했습니다. OLMES는 평가 과정의 모든 세부 사항을 문서화하고, 실질적인 결정들을 제공하여 AI 모델의 성능 비교를 더 공정하고 일관되게 만듭니다.

- **Technical Details**: OLMES는 프롬프트 포맷팅(prompt formatting), 컨텍스트 예제(choice of in-context examples), 확률 정규화(probability normalization), 그리고 작업 공식(task formulation) 등 다양한 평가 요소들을 세부적으로 다룹니다. 이를 통해 평가 세트업(setup)이 일관되게 유지되도록 합니다. 또한, 모델 개발 과정과 공개 리더보드, 논문 등에서 쉽게 적용할 수 있는 실질적 결정을 내립니다.

- **Performance Highlights**: OLMES는 15개의 다양한 사전 학습된 LLM들에 대해 실험을 진행했으며, 10개의 인기 있는 MCQA(Multiple-choice question answering) 벤치마크 작업을 표준화하였습니다. 이러한 벤치마크 작업에는 과학, 상식, 사실 지식 등 다양한 주제를 포함하며, 예를 들어 MMLU(Massively Multitask Language Understanding)는 57개의 과목을 다룹니다.



### TasTe: Teaching Large Language Models to Translate through Self-Reflection (https://arxiv.org/abs/2406.08434)
Comments:
          This paper has been accepted to the ACL 2024 main conference

- **What's New**: 최근 대형 언어 모델(LLMs)은 다양한 자연어 처리 작업에서 놀라운 성능을 보여주고 있습니다. LLM의 다운스트림 작업인 기계 번역에서의 역량을 향상시키기 위해 TasTe(Translating through Self-reflection) 프레임워크가 제안되었습니다. 이 방법은 LLM이 자체 평가를 통해 번역을 생성하고 이를 자기 평가에 따라 개선하게 합니다.

- **Technical Details**: TasTe 프레임워크는 두 단계의 추론 과정을 포함합니다. 첫 번째 단계에서는 LLM이 초기 번역을 생성하고 이러한 번역의 자체 평가를 수행합니다. 두 번째 단계에서는 LLM이 평가 결과에 따라 이 초기 번역을 수정합니다. 이 과정은 인간의 작업 방식과 유사한 자기 반성(self-reflection) 메커니즘을 도입합니다.

- **Performance Highlights**: TasTe 프레임워크는 WMT22 벤치마크의 4개 언어 방향에서 기존 방법보다 우수한 성능을 보여주었습니다. TasTe에서는 LLM이 초기 번역 후보를 보다 능숙하게 수정하여 최종 결과물을 개선함으로써 번역 능력을 크게 향상시킵니다.



### Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL (https://arxiv.org/abs/2406.08426)
- **What's New**: 텍스트-투-SQL (text-to-SQL)은 자연어 질문을 데이터베이스 실행 가능한 SQL 쿼리로 변환하는 오랜 문제로, 사용자 질문 이해, 데이터베이스 스키마 이해, SQL 생성 등에서 도전적인 과제를 안고 있습니다. 최근 대형 언어 모델(LLM)의 활용이 이 분야에 새로운 가능성과 도전 과제를 제시하고 있습니다.

- **Technical Details**: 이번 서베이에서는 LLM 기반의 text-to-SQL에 대한 종합적인 리뷰를 제공하며, 텍스트-투-SQL의 현재 문제점과 진화 과정에 대한 개괄적인 개요를 제시합니다. 특히, in-context learning (ICL) 및 supervised fine-tuning (SFT) 패러다임을 통해 LLM 기반 접근법이 최첨단 정확도를 달성한 방법에 대해 체계적으로 분석합니다. 텍스트-투-SQL 시스템의 전반적인 구현은 질문 이해, 스키마 이해, SQL 생성의 세 가지 측면으로 나뉩니다.

- **Performance Highlights**: LLM 기반 접근법은 PLM에 비해 더 강력한 이해 능력과 포괄적 훈련 코퍼스의 이점을 살려 성능을 강화했습니다. 특히, 질문 이해 및 스키마 이해에서의 향상된 성능을 보여주며, 다양한 데이터 세트와 메트릭을 통해 평가되었습니다.



### Tailoring Generative AI Chatbots for Multiethnic Communities in Disaster Preparedness Communication: Extending the CASA Paradigm (https://arxiv.org/abs/2406.08411)
Comments:
          21 pages

- **What's New**: 이 연구는 최초로 GPT-4 기반의 생성형 인공지능(GenAI) 챗봇을 활용해 다양한 주민에게 허리케인 대비 정보를 전달하는 다양한 프로토타입을 개발한 사례 중 하나입니다. 연구는 Black, Hispanic, Caucasian 계층의 플로리다 주민 441명을 대상으로 진행되었습니다.

- **Technical Details**: 이 연구는 '기계는 사회적 행위자' (Computers Are Social Actors, CASA) 패러다임과 재난 취약성 및 문화적 맞춤화에 관한 문헌을 바탕으로 설계되었습니다. 연구는 실험 참가자들이 톤의 형식성과 문화적 맞춤화가 다른 챗봇들과 상호작용한 후에 설문지를 작성하는 비트윈 서브젝트 실험을 통해 이루어졌습니다.

- **Performance Highlights**: SEM (구조방정식 모델링) 분석 결과, 챗봇의 톤 형식성과 문화적 맞춤화가 사용자들의 챗봇에 대한 인식과 허리케인 대비 결과에 중요한 영향을 미쳤습니다. 컴퓨터 로그 분석을 통해 챗봇-사용자 상호작용에서 의인화와 개인화가 주요 통신 주제임을 확인했습니다. 이를 통해 GenAI 챗봇이 다양한 커뮤니티의 재난 대비를 개선할 잠재력을 가지고 있음을 강조합니다.



### cPAPERS: A Dataset of Situated and Multimodal Interactive Conversations in Scientific Papers (https://arxiv.org/abs/2406.08398)
Comments:
          14 pages, 1 figure

- **What's New**: 최근 본 논문은 학술 논문에서 사용되는 텍스트, 수식, 그림, 표와 같은 다양한 구성 요소를 다루는 대화형 보조 시스템을 개발하기 위한 연구입니다. 이를 위해, 'Conversational Papers(cPAPERS)'라는 데이터를 새롭게 소개하였습니다. 이 데이터셋은 arXiv에서 제공되는 과학 문서의 리뷰와 관련 참조를 기반으로 한 대화형 질문-답변 쌍으로 구성되어 있습니다.

- **Technical Details**: 이 논문은 'OpenReview'에서 질문-답변 쌍을 수집하고, 이를 LaTeX 소스 파일과 연결하는 데이터를 수집하는 전략을 제안합니다. 또한, 제로샷(zero-shot)과 미세 조정(fine-tuning) 구성에서 대형 언어 모델(LLM)을 활용한 기본 접근법을 소개합니다. 이 데이터셋은 크게 수식(cPAPERS-EQNS), 그림(cPAPERS-FIGS), 표(cPAPERS-TBLS) 세 가지 부분으로 나뉘어 있습니다.

- **Performance Highlights**: 논문에서 소개된 데이터셋은 총 5030개의 질문-답변 쌍으로 이루어져 있으며, 이는 2020년부터 2023년까지의 NeurIPS와 ICLR 학술회의에서 열린 리뷰와 반박 자료에서 수집되었습니다. 각 질문-답변 쌍은 학술 논문에 등장하는 구체적인 시각적 정보와 문맥적 연관성이 있어, 기존의 데이터셋과 차별화됩니다.



### Towards Unsupervised Speech Recognition Without Pronunciation Models (https://arxiv.org/abs/2406.08380)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 병렬 음성-텍스트 데이터에 의존하지 않고 자동 음성 인식(ASR) 시스템을 개발하는 새로운 접근법을 제안합니다. 특히 음소(phoneme) 기반 사전을 제거하고 단어 수준의 비지도 음성 인식을 탐구합니다. 이를 통해 고빈도 영어 단어만을 포함하는 음성 코퍼스를 사용하여 거의 20%의 단어 오류율(word error rate)을 달성했습니다.

- **Technical Details**: 비지도 음성 인식 문제를 해결하기 위해 음성과 텍스트의 마스크된 토큰 보완(masked token-infilling)을 통한 공동 학습을 사용했습니다. 웨이브레벨 정보가 포함된 HubERT 및 VG-HubERT과 같은 사전 학습된 모델에서 추출한 음성 표현(representations)을 기반으로 단어 레벨의 특징을 추출하여 모델을 훈련했습니다. 또한, 변형된 트랜스포머(Transformer) 아키텍처와 JSTTI(Joint Speech-Text Token-Infilling) 기준을 도입하여 단어 레벨에서 비지도 음성 인식을 수행했습니다.

- **Performance Highlights**: 새롭게 제안된 JSTTI 시스템은 강력한 기준 모델을 능가하는 성능을 보였습니다. 각 단계에서 단어 오류율이 감소하였으며, 최종 시스템은 이전의 비지도 ASR 모델보다 현저히 개선된 성능을 나타냈습니다.



### Is Programming by Example solved by LLMs? (https://arxiv.org/abs/2406.08316)
- **What's New**: 새로운 연구는 대형 언어 모델(Large Language Models, LLMs)의 코드 생성 작업에서의 성과를 바탕으로, 이러한 모델들이 프로그래밍 예제(Programming-by-Examples, PBE)를 어느 정도 해결하였는지 조사합니다. 연구는 리스트와 문자열 같은 고전적인 도메인뿐만 아니라, 일반적인 사전 학습 데이터에 잘 나타나지 않는 그래픽 프로그래밍 도메인에서도 실험을 수행했습니다.

- **Technical Details**: 사전 학습된 모델들은 PBE에서 효과적이지 않다는 사실을 발견했지만, 테스트 문제들이 데이터 분포 내에 있을 경우 모델들을 미세 조정(fine-tuning)하여 성능을 크게 향상시킬 수 있었습니다. 연구는 이러한 모델들이 성공하고 실패하는 원인을 실험적으로 분석하고, 분포 외 일반화(out-of-distribution generalization)를 어떻게 달성할 수 있을지에 대한 이해를 증진시키기 위한 단계를 취했습니다.

- **Performance Highlights**: 프로그래밍 예제 작업의 전형적인 작업군을 해결하는 데 있어 LLMs가 강한 발전을 이루었음을 시사하며, 이는 PBE 시스템의 유연성과 적용 가능성을 잠재적으로 증대시킬 수 있습니다. 하지만 연구는 여전히 LLMs가 부족한 부분을 식별하고 개선점을 제시합니다.



### M3T: A New Benchmark Dataset for Multi-Modal Document-Level Machine Translation (https://arxiv.org/abs/2406.08255)
Comments:
          NAACL 2024, dataset at this https URL

- **What's New**: M3T라는 새로운 벤치마크 데이터셋이 소개되었습니다. 이 데이터셋은 반정형 문서의 번역을 평가하기 위해 설계되었으며, 기존 문헌 수준에서 허점을 메꾸고 실세계 응용에서의 복잡한 텍스트 레이아웃 문제를 해결하는 데 목표를 두고 있습니다.

- **Technical Details**: 기존 문서 번역 시스템은 텍스트 콘텐츠와 시각적 레이아웃 구조를 무시하는 경향이 있습니다. M3T 데이터셋은 PDF 문서를 대상으로 하며, 레이아웃 정보(레이아웃 정보가 포함된 박스 좌표 및 레이블)를 제공하여 텍스트 추출과 번역의 두 단계를 거칩니다. OCR 기반의 텍스트 추출 및 기존의 대형 언어 모델(LLM) 분석의 한계도 논의되었습니다.

- **Performance Highlights**: 초기 실험에서 최근 멀티모달 모델인 LLaVa-v1.5를 사용하여 시각적 피처가 번역 품질을 향상시킨다는 것을 발견했습니다. 하지만 개선의 여지가 많이 남아 있음을 확인했습니다.



### Leveraging Large Language Models for Web Scraping (https://arxiv.org/abs/2406.08246)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)의 지식 표현 능력과 RAG 모델의 정보 접근 능력을 결합하여, LLMs가 데이터 추출에 있어서의 제한점을 극복하는 일반적인 데이터 스크래핑(recipe)을 제시하였습니다. 특히, HTML 요소의 의미 분류(semantic classification), HTML 텍스트 조각(chunking) 및 다양한 언어 모델과 랭킹 알고리즘 비교 등의 작업을 통해 정확한 데이터 추출 방법을 분석하였습니다.

- **Technical Details**: 연구에서는 사전 훈련된 언어 모델과 잠재적 지식 검색기(latent knowledge retriever)를 사용하여, 대규모 코퍼스(corpus)에서 문서를 검색하고 참조할 수 있는 기능을 포함시켰습니다. RAG 모델 아키텍처를 활용하여 HTML 요소의 의미 분류, 효과적인 이해를 위한 HTML 텍스트 조각화, 그리고 다른 LLM들과의 비교 및 랭킹 알고리즘을 분석하였습니다.

- **Performance Highlights**: 이전 연구들은 HTML 이해 및 추출을 위한 전용 아키텍처와 훈련 절차를 개발했지만, 본 연구는 표준 자연어로 사전 훈련된 LLMs와 효과적인 조각화(chunking), 검색 및 랭킹 알고리즘의 추가를 통해 복잡한 데이터 추출을 더 효율적으로 수행할 수 있음을 보여주었습니다. 또한, 미래 연구 방향으로, 데이터 추출 프레임워크 내에서 출처 추적(provenance tracking)과 동적 지식 갱신(dynamic knowledge updates)의 문제를 해결하고자 합니다.



### Figuratively Speaking: Authorship Attribution via Multi-Task Figurative Language Modeling (https://arxiv.org/abs/2406.08218)
- **What's New**: 이 연구는 최초로 여러 비유적 언어(Figurative Language, FL) 특징을 분석하여 이를 저자 식별(AA) 작업에 적용했습니다. 연구팀은 다중 작업 비유적 언어 모델(MFLM)을 제안하여 텍스트 내 다수의 FL 특징을 동시에 감지하도록 했습니다. 이를 통해 다중 작업 모델이 개별 특화 모델들보다 더 우수한 성능을 보일 수 있음을 입증했습니다. 또한, FL 특징이 저자 식별 작업에서 성능 향상을 가져오는지 평가하였습니다.

- **Technical Details**: 이 연구에서는 RoBERTa를 기반으로 한 모델을 사용하여 메타포(Metaphor), 유사(Simile), 관용구(Idiom), 비꼬기(Sarcasm), 과장(Hyperbole) 및 아이러니(Irony)의 6가지 FL 특징을 학습했습니다. 초기 데이터는 13개의 공개 데이터셋을 이용해 구축하였고, 이 중 FL 특징을 라벨링하여 다중 작업 모델을 훈련했습니다. 저자 식별 작업을 위해 MFLM 임베딩을 다층 퍼셉트론(MLP) 분류기에 입력 피처로 사용했고, 전통적인 스타일리틱(Stylometric) 특징 및 TF-IDF 벡터 등을 비교 대상으로 활용했습니다.

- **Performance Highlights**: MFLM은 13개의 테스트 세트 중 5개에서는 개별 이진 분류 모델과 동일하거나 더 나은 성능을 보였고, 3개 테스트 세트에서는 더 높은 작업별 성능을 달성했습니다. 또한, 세 개의 공개 AA 데이터셋을 사용한 평가에서 MFLM 임베딩을 사용하여 AA 성능이 향상됨을 확인했습니다. 특히, FL 특징을 기존의 스타일리틱 특징과 결합할 때 성능 향상이 뚜렷하게 나타났습니다. FL 사용이 저자의 스타일을 잘 반영하며 저자 식별에 탁월한 예측 변수가 될 수 있음을 보여주었습니다.



### SumHiS: Extractive Summarization Exploiting Hidden Structur (https://arxiv.org/abs/2406.08215)
- **What's New**: 새로운 방식의 추출적 요약(Extractive Summarization) 접근법이 도입되었습니다. 이 접근법은 텍스트의 숨겨진 클러스터링 구조(Hidden Clustering Structure)를 활용하여 더 정확한 요약을 생성합니다. CNN/DailyMail 데이터셋에 대한 실험 결과, ROUGE-2 측정 기준에서 이전 접근법보다 10% 이상 성능이 뛰어난 것으로 나타났습니다.

- **Technical Details**: SumHiS (Summarization with Hidden Structure) 모델은 BERT 모델의 표현을 활용하여 문서의 문장을 중요도에 따라 순위 매깁니다. 그리고 랭크된 문장들을 필터링하여 문서 내 발견된 주요 주제에 초점을 맞춘 요약을 생성합니다. 이 모델은 초기 언어 모델의 컨텍스트 기반 표현(Contetualized Representations)을 사용하여 문서의 문장을 평가하고 주제적 구조(Topical Structure)를 통합합니다.

- **Performance Highlights**: SumHiS 모델은 CNN/DailyMail 데이터셋에서 ROUGE-2와 ROUGE-L 지표 기준으로 이전의 최고 성능 모델들을 능가하는 결과를 보였습니다. 특히, ROUGE-2 메트릭에서 10%의 성능 향상을 달성했으며, 추출적 요약 모델뿐만 아니라 생성적 요약(Abstractive Summarization) 모델도 능가했습니다.



### A Dialogue Game for Eliciting Balanced Collaboration (https://arxiv.org/abs/2406.08202)
- **What's New**: 두 명의 플레이어가 자율적으로 목표 상태를 협상해야 하는 2D 오브젝트 배치 게임을 소개합니다. 이 게임은 더 균형 잡힌 협업을 유도하며, 인간 플레이어들이 여러 가지 역할 분담 전략을 보여준다는 점을 입증했습니다. 또한, 이 게임을 자동 플레이하는 챌린지로서 LLM 기반 에이전트(baseline agent)를 도입했습니다.

- **Technical Details**: 개발된 게임에서는 플레이어들이 각각 다른 초기 위치에서 동일한 물체를 배치해야 하며, 서로의 화면을 볼 수 없으므로 채팅을 통해서만 소통할 수 있습니다. 슬러크(Slurk) 플랫폼을 활용해 온라인으로 게임을 제공하며, 각각의 물체 위치를 맨해튼 거리(Manhattan distance)로 측정해 점수를 매깁니다. 또한, 이 게임에서 플레이어들은 ‘리더’ 전략, ‘번갈아 하기’ 전략 등 다양한 협업 전략을 사용합니다.

- **Performance Highlights**: 실험 결과, 플레이어들은 다수의 협업 전략을 사용하며 다양한 역할 분배를 보였습니다. 리더 전략을 사용한 플레이어 그룹은 소수였으며, 주로 사용한 그룹들은 더 낮은 성과를 보였습니다. 기반 에이전트는 인간 플레이어에 비해 낮은 점수를 기록하여, 이 게임에서의 자연스러운 협업이 AI 연구의 흥미로운 도전 과제가 될 수 있음을 시사합니다.



### Underneath the Numbers: Quantitative and Qualitative Gender Fairness in LLMs for Depression Prediction (https://arxiv.org/abs/2406.08183)
- **What's New**: 이 연구는 기존의 대형 언어 모델(LLMs)에서 성 편향(gender bias)을 조사한 첫 번째 시도로서, ChatGPT, LLaMA 2, 그리고 Bard를 통해 우울증 예측에서의 성 편향을 정량적 및 정성적으로 평가합니다. 특히 ChatGPT가 다양한 성능 지표에서 가장 우수하며, LLaMA 2는 그룹 공정성(group fairness) 측면에서 다른 모델들을 능가한다는 것을 발견했습니다.

- **Technical Details**: 우울증 예측을 위해 ChatGPT, LLaMA 2, Bard를 비교했다는 점에서 이 연구는 독특합니다. 정량적 평가에서는 모델의 성능 지표와 그룹 공정성을 분석하였고, 정성적 평가는 단어 수 및 주제 분석(취학 분석)을 통해 수행되었습니다. 특히, 인간 중심 접근 방식으로 LLM이 내리는 결정에 대한 설명 능력을 평가하여 알고리즘 설명 가능성과 투명성을 높이는 방향으로 연구가 진행되었습니다.

- **Performance Highlights**: ChatGPT는 정량적 평가에서 다양한 성능 지표에서 가장 뛰어난 성능을 보였으며, LLaMA 2는 그룹 공정성 측면에서 다른 모델보다 우수했습니다. 정성적 평가에서는 ChatGPT가 더 포괄적이고 논리적인 예측 설명을 제공하는 경향이 있다는 점이 발견되었습니다. 이 연구는 LLM의 공정성 평가에서 정량적 접근 방식뿐만 아니라 정성적 접근 방식의 중요성을 강조합니다.



### Semi-Supervised Spoken Language Glossification (https://arxiv.org/abs/2406.08173)
Comments:
          Accepted to ACL2024 main

- **What's New**: Spoken language glossification(SLG)은 음성 언어 텍스트를 수어(gloss)로 번역하기 위한 연구입니다. 이번 논문에서는 대규모 단일 언어 음성 언어 텍스트를 이용한 세미-감독(semisupervised) 학습 프레임워크 $S^3LG 을 제안합니다. 본 연구는 자체 훈련 구조를 채택하여 반복적으로 가상 라벨(pseudo-label)을 생성 및 학습합니다.

- **Technical Details**: 본 연구의 $S^3LG 프레임워크는 규칙 기반(heuristic) 접근법과 모델 기반 접근법을 결합하여 자동 주석(annotations)을 생성합니다. 주석 데이터의 품질을 보완하기 위해 학습 중에 무작위로 혼합된 합성 데이터를 사용하며, 특수 토큰(special token)으로 각 데이터의 차이를 표시합니다. 또한, 합성 데이터의 노이즈 영향을 줄이기 위해 일관성 규제(consistency regularization)를 적용합니다.

- **Performance Highlights**: $S^3LG 프레임워크는 CSL-Daily와 PHOENIX14T 벤치마크에서 실험을 수행한 결과, 기존 모델에 비해 성능이 크게 향상되었습니다. 특히, 드문 빈도를 가진 글로스(low-frequency glosses)에 대한 번역 정확도가 현저히 개선되었습니다.



### Legend: Leveraging Representation Engineering to Annotate Safety Margin for Preference Datasets (https://arxiv.org/abs/2406.08124)
Comments:
          Our code is available at this https URL

- **What's New**: 새로운 연구는 고품질의 선호 데이터셋(preference dataset) 개발을 위해 효과적이고 비용 효율적인 프레임워크 'Legend'를 제안합니다. Legend는 LLM의 임베딩 공간에서 안전을 나타내는 특정 방향을 구축하고, 이를 활용해 응답 쌍의 의미적 거리(semantic distances)를 자동으로 주석 처리하여 margin을 구축합니다.

- **Technical Details**: Legend는 임베딩 벡터의 특정 방향을 안전(safety)으로 분리하고, 이 방향에 따른 의미적 거리 측정을 통해 자동으로 margin을 주석 처리합니다. 이 과정은 두 단계로 구성됩니다: 안전 벡터 발견(safety vector discovery) 및 margin 주석 처리. Legend는 단순한 추론 시간만을 요구하며, 추가적인 모델 학습이 필요 없기 때문에 비용 효율적입니다.

- **Performance Highlights**: Legend를 사용하여 주석 처리된 데이터셋은 보상 모델의 무해한 응답 선택 정확도를 2% 향상시키고, 하위 정렬에서 무해한 응답 생성의 승률을 10% 향상시켰습니다. 또한, Legend는 다른 자동 margin 주석 처리 방법과 비교해 성능이 비슷하거나 더 우수하며, 시간 비용을 크게 절감합니다.



### Supportiveness-based Knowledge Rewriting for Retrieval-augmented Language Modeling (https://arxiv.org/abs/2406.08116)
- **What's New**: 새로운 논문에서는 대형 언어 모델(LLM)의 한계점을 보완할 수 있는 'Supportiveness-based Knowledge Rewriting (SKR)'을 소개합니다. SKR은 외부 지식 베이스와 정보를 재작성하여 LLM이 더 효과적으로 지식을 활용할 수 있도록 돕습니다. 새로운 개념인 '지원성(supportiveness)'을 도입하여 하위 작업을 얼마나 효과적으로 지원하는지를 평가하며, 이를 바탕으로 최적의 지식을 LLM 생성에 맞게 재작성합니다.

- **Technical Details**: SKR은 자동으로 생성된 학습 데이터를 평가하고 필터링하여 데이터 효율성을 향상시키기 위해 지원성 점수를 사용합니다. 그리고 '직접 선호 최적화(Direct Preference Optimization, DPO)' 알고리즘을 사용하여 재작성된 텍스트를 최적의 지원성에 맞출 수 있도록 합니다. 이를 위해 백박스 LLM을 사용하여 쿼리에 대한 두 개의 응답(증강 지식이 포함된 응답과 그렇지 않은 응답)을 생성하고, 이 두 응답의 혼란도(perplexity) 비율을 계산하여 지원성을 정의합니다.

- **Performance Highlights**: SKR은 7B 파라미터만으로도 현재의 최첨단 일반 목적 LLM인 GPT-4보다 우수한 지식 재작성 능력을 보여줍니다. 여섯 가지 인기 있는 지식 집중 작업과 네 가지 LLM에 대한 종합 평가에서 그 효과와 우수성이 입증되었습니다. SKR은 노이즈와 오해의 소지가 있는 정보를 효과적으로 제거하면서 7배 이상의 압축률을 달성했습니다.



### CoXQL: A Dataset for Parsing Explanation Requests in Conversational XAI Systems (https://arxiv.org/abs/2406.08101)
Comments:
          4 pages, short paper

- **What's New**: 본 논문에서는 사용자 의도 인식(intent recognition) 기반의 대화형 설명 가능한 인공지능 시스템(ConvXAI)에 관한 새로운 데이터셋 CoXQL을 소개합니다. 이는 현재 대화형 모델의 사용자 의도 인식 문제 해결에 중점을 두고 있으며, 31개의 의도를 다루고 있습니다. 그 중 7개의 의도는 추가 슬롯(slot) 채우기가 필요합니다.

- **Technical Details**: CoXQL 데이터셋은 다양한 XAI(설명 가능한 인공지능) 방법을 요청에 매핑하는 것을 목표로 합니다. 그 과정에서 슬롯 채우기와 같은 요소를 복잡하게 조정해야 합니다. 기존의 신탁 방식을 템플릿 검증(template validations)을 통해 개선한 MP+ 방식을 적용하여 여러 대형 언어 모델(LLM)을 평가했습니다.

- **Performance Highlights**: 개선된 파싱(parsing) 접근법인 MP+는 이전의 접근법들보다 뛰어난 성능을 보여주었습니다. 하지만 여전히 여러 슬롯을 요구하는 의도(intent)는 LLM에게 큰 도전 과제로 남아 있습니다.



### Multimodal Table Understanding (https://arxiv.org/abs/2406.08100)
Comments:
          23 pages, 16 figures, ACL 2024 main conference, camera-ready version

- **What's New**: 이번 연구에서는 기존의 텍스트 기반 테이블 이해 모델이 아닌, 테이블 이미지에 직접적으로 대응할 수 있는 멀티모달 테이블 이해 문제를 제안합니다. 새로운 문제를 해결하기 위해 MMTab라는 대규모 데이터셋을 구축하고, Table-LLaVA라는 멀티모달 대규모 언어 모델(MLLM)을 개발했습니다. 이 모델은 다양한 테이블 관련 요청에 대해 정확한 응답을 생성합니다.

- **Technical Details**: MMTab 데이터셋은 14개 공개 테이블 데이터셋을 활용하여 8개 도메인의 테이블 이미지를 포함하며, 다양한 테이블 구조와 스타일을 커버합니다. 이 데이터셋은 (1) 97K 테이블 이미지에서 150K 테이블 인식 샘플을 담은 MMTab-pre, (2) 82K 테이블 이미지에서 232K 테이블 기반 작업 샘플을 담은 MMTab-instruct, (3) 23K 테이블 이미지에서 49K 테스트 샘플을 담은 MMTab-eval로 구성됩니다. Table-LLaVA 모델은 두 단계의 학습 프로세스를 거치며 개발되었습니다. 첫 번째 단계에서는 MMTab-pre를 사용하여 테이블 인식 작업을 수행하고, 두 번째 단계에서는 MMTab-instruct를 사용하여 다양한 테이블 기반 작업을 수행하도록 모델을 튜닝합니다.

- **Performance Highlights**: Table-LLaVA 모델은 17개 held-in 벤치마크와 6개 held-out 벤치마크에서 강력한 MLLM 베이스라인을 능가하는 성능을 보였으며, 일부 테스트 샘플에서는 GPT-4V와 경쟁할 만한 성능을 보였습니다. 다양한 데이터셋을 통해 학습된 이 모델은 테이블 구조 및 내용 이해 능력이 크게 향상되었습니다.



### Languages Transferred Within the Encoder: On Representation Transfer in Zero-Shot Multilingual Translation (https://arxiv.org/abs/2406.08092)
- **What's New**: 이 논문은 다국어 신경 기계 번역(MNMT)에서 나타나는 제로샷 번역 성능 저하의 원인을 밝히기 위해 '아이덴티티 페어(identity pair)'를 소개합니다. 아이덴티티 페어는 자기 자신을 번역한 문장으로, 언어 전이 및 표현 상태의 최적 상태를 나타내는 기준점 역할을 합니다.

- **Technical Details**: 본 연구에서는 다국어 번역 모델의 인코더가 소스 언어를 타겟 언어의 표현적 부분공간으로 전이시키며, 이로 인해 제로샷 번역 성능이 저하된다고 합니다. 이를 해결하기 위해, 인코더에 저차원 언어별 임베딩(Low-Rank Language-specific Embedding)과 디코더에 언어별 대조 학습(Language-specific Contrastive Learning of Representations)을 제안합니다.

- **Performance Highlights**: 제안된 방법은 Europarl-15, TED-19, OPUS-100 데이터셋에서 실험한 결과, SacreBLEU와 BERTScore 두 가지 자동 측정을 통해 기존의 강력한 베이스라인보다 우수한 성능을 보였습니다. 이 방법은 특히 제로샷 번역의 언어 전이 능력을 크게 향상시키는 것으로 나타났습니다.



### AustroTox: A Dataset for Target-Based Austrian German Offensive Language Detection (https://arxiv.org/abs/2406.08080)
Comments:
          Accepted to Findings of the Association for Computational Linguistics: ACL 2024

- **What's New**: 독성 감지(Toxicity detection)에서 모델의 해석 가능성은 토큰(Token) 수준의 주석(annotations)으로 크게 향상됩니다. 그러나 현재 이러한 주석은 영어로만 제공됩니다. 오스트리아 독일 방언을 포함한 4,562개의 사용자 댓글로 구성된 뉴스 포럼에서 공격적인 언어 감지 데이터셋을 소개합니다. 각 댓글에서 음란 언어 또는 공격성 대상이 되는 부분을 식별하여 이진 공격성 분류를 수행합니다. 우리는 fine-tuned 언어 모델과 대형 언어 모델을 zero-shot 및 few-shot 방식으로 평가했습니다. 결과는 fine-tuned 모델이 음란한 방언과 같은 언어적 특이성 감지에서 뛰어남을 나타내지만, 대형 언어 모델이 AustroTox에서 공격적인 언어 감지에 더 우수함을 보여줍니다. 데이터를 공개하며 코드도 함께 배포합니다.

- **Technical Details**: AustroTox 데이터셋은 오스트리아 신문 DerStandard의 댓글에서 수집되었으며, 각 댓글은 다섯 명의 주석자에 의해 이진 공격성 분류와 구체적인 공격성 목표와 음란 언어를 포함한 범위(주석)가 달려 있습니다. 대조군이나 공격적이지 않은 댓글에서도 음란 언어가 존재할 수 있으며, 공격적인 포스트에서는 추가로 공격적인 발언의 대상과 유형(사람, 그룹 등)을 주석합니다. Perspective API를 사용해 독성 점수를 계산하고, 이를 기반으로 층화 샘플링을 적용해 댓글을 수집했습니다.

- **Performance Highlights**: 평가 결과, fine-tuned 작은 언어 모델은 방언과 같은 언어적 특이성을 감지하는 데 뛰어났지만, 대형 언어 모델이 공격성 감지에서 더 우수한 성능을 보였습니다. 대규모 언어 모델이 예를 들어 공격적인 언어를 더 잘 감지할 수 있음을 나타냅니다. 또한, 데이터를 수집하고 주석을 다는 과정에서 공정성과 다양성을 유지하려고 노력했습니다.



### Large Language Models Meet Text-Centric Multimodal Sentiment Analysis: A Survey (https://arxiv.org/abs/2406.08068)
- **What's New**: 이 논문은 텍스트 중심의 다중 모드 감정 분석(multi-modal sentiment analysis)을 위해 대규모 언어 모델(LLMs)을 어떻게 더 잘 적용할 수 있는지에 대한 종합적인 리뷰를 제공합니다. 최근 ChatGPT와 같은 LLMs의 등장으로 텍스트 중심의 다중 모드 작업에 대한 잠재력이 커졌지만, 기존의 LLMs가 어떻게 더 잘 적응할 수 있는지는 불명확합니다. 이 논문은 LLMs와 대규모 다중 모드 모델(LMMs)의 장점과 한계를 제시하고, 다중 모드 감정 분석의 향후 연구 방향과 도전 과제를 탐구합니다.

- **Technical Details**: 다중 모드 감정 분석은 자연 언어, 이미지, 비디오, 오디오, 생리 신호 등 다양한 소스를 처리해야 합니다. 이러한 소스들에서 감정 신호를 통합하여 분석하는 것이 중요합니다. LLMs는 드문 훈련이나 무감독 학습(zero-shot)을 통해 태스크를 수행할 수 있는 강력한 능력을 보유하고 있으며, 학습된 자연 언어 지침을 기반으로 작업을 수행할 수 있는 능력을 가지고 있습니다. 이 논문은 텍스트-이미지 감정 분류, 오디오-이미지-텍스트 감정 분류 등 다양한 텍스트 중심의 부문에서 LLMs와 LMMs의 적용 방법과 퍼포먼스를 분석합니다.

- **Performance Highlights**: 기존 모델들과 비교했을 때, LLMs는 다중 모드 감정 분석 태스크에서 뛰어난 성능을 보여주었으며, 특히 시작된 학습(in-context learning)과 단계별 추론 능력(chain-of-thought)을 통해 복잡한 작업을 해결할 수 있었습니다. 최근 연구에서는 LLMs가 감정 극성 변화 및 개방형 도메인 시나리오 등의 문제를 효과적으로 처리할 수 있음을 입증했습니다. LMMs는 다양한 데이터 타입을 통합하여 더 정확한 정보를 생성하는 능력을 향상시키고 있으며, 여러 소스에서 정보를 분석하고 상관관계를 도출하는 능력을 강화하고 있습니다.



### Learning Job Title Representation from Job Description Aggregation Network (https://arxiv.org/abs/2406.08055)
Comments:
          to be published in Findings of the Association for Computational Linguistics: ACL 2024

- **What's New**: 기존의 스킬 기반 접근 방식을 넘어, 직무 기술(Job Description, JD)을 통한 직무 제목(Job Title) 표현 학습을 제안합니다. 이 프레임워크는 JD 내의 중요한 세그먼트를 가중치로 처리하며, 직무 제목과 JD 간의 양방향 관계를 고려한 대비 학습(contrastive learning)을 활용합니다.

- **Technical Details**: 제안된 프레임워크는 직무 제목과 분할된 JD를 각각 센텐스 인코더(sentence encoder)에 입력하여 표현을 얻습니다. 그런 다음, JD 애그리게이터(Aggregator)를 통해 통합된 표현을 획득합니다. 트레이닝 목표는 직무 제목과 그 JD 표현 간의 유사성을 극대화하고 다른 표현과의 유사성을 최소화하는 양방향 컨트라스티브 손실(bidirectional contrastive loss)을 사용합니다.

- **Performance Highlights**: 제안된 JD 기반 방법은 인-도메인(in-domain) 및 아웃-오브-도메인(out-of-domain) 설정 모두에서 기존의 스킬 기반 접근 방식을 능가하며, 최대 1.8%와 1.0%의 절대적인 성능 향상을 달성했습니다. 또한, 모델의 주요 세그먼트 가중치 부여 기능이 정확도에 중요한 역할을 함을 보여주었습니다.



### Adversarial Evasion Attack Efficiency against Large Language Models (https://arxiv.org/abs/2406.08050)
Comments:
          9 pages, 1 table, 2 figures, DCAI 2024 conference

- **What's New**: 최근 연구는 감정 분류 작업(Sentiment Classification Task)에서 다섯 가지 대형 언어 모델(LLMs)에 대한 세 가지 유형의 적대적 공격(adversarial attacks)의 효과성, 효율성 및 실용성을 분석합니다. 특히 단어 수준(word-level)과 문자 수준(character-level) 공격이 모델의 분류 결과에 미치는 영향이 다르다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 BERT, RoBERTa, DistilBERT, ALBERT, XLNet의 다섯 가지 모델을 사용하여 RottenTomatoes 데이터셋(영화 리뷰 데이터며 감정 분석에 주로 사용)을 대상으로 분석을 수행했습니다. 주요 공격 방식에는 BERTAttack(단어 수준), ChecklistAttack(체크리스트 기반 단어 교체), TypoAttack(문자 수준)이 포함되며, 각 공격의 효과는 Misclassification Rate(MR), Average Perturbed Words(APW), Average Required Queries(ARQ) 등의 메트릭으로 평가되었습니다.

- **Performance Highlights**: 단어 수준 공격은 더 효과적이었지만, 문자 수준 공격과 더 제한된 공격은 실용성이 더 높고 적은 수의 페르투베이션(perturbations)과 쿼리(query)만 필요로 했습니다. 이는 적대적 방어 전략을 개발할 때 중요한 요소로 고려되어야 합니다.



### It Takes Two: On the Seamlessness between Reward and Policy Model in RLHF (https://arxiv.org/abs/2406.07971)
- **What's New**: 이번 연구에서는 인간 피드백 강화 학습(RLHF)에서 정책 모델(PM)과 보상 모델(RM)의 상호작용을 효과적으로 분석하고자 합니다. 해당 연구는 PM과 RM의 질적 향상이 RLHF의 성능 향상으로 직결되지 않는 '포화 현상'을 관찰하면서 시작되었습니다. 이 현상을 해결하기 위해 PM과 RM 간의 일치도를 측정하고 개선하는 자동화 지표인 SEAM을 도입하였습니다.

- **Technical Details**: 이 연구는 PM과 RM이 각각 독립적으로 최적화될 때 RLHF 데이터에서 35%의 불일치를 보이는 것을 발견했습니다. 이 불일치는 고도로 최적화된 모델에서도 해결되지 않았습니다. SEAM 지표는 데이터 샘플이 RLHF 과정에서 발생시키는 리스크를 평가하며, SEAM을 활용한 데이터 선택(Data Selection) 및 모델 증강(Model Augmentation) 두 가지 시나리오를 통해 최대 4.5%의 성능 향상을 보여주었습니다. SEAM은 SEAMAdv, SEAMContrast, SEAMGPT 세 가지 버전으로 제공됩니다.

- **Performance Highlights**: SEAM 필터링을 통한 데이터 선택은 RLHF 성능을 4.5% 향상시켰으며, SEAM을 활용한 모델 증강은 기존 증강 방법에 비해 4%의 성능 향상을 가져왔습니다. 이로써 SEAM은 RLHF 과정의 진단 지표로 효과적으로 작용할 수 있음을 입증했습니다.



### Guiding In-Context Learning of LLMs through Quality Estimation for Machine Translation (https://arxiv.org/abs/2406.07970)
- **What's New**: 최신 연구는 대규모 언어 모델(LLMs)의 기계 번역(MT) 성능을 최적화하기 위해 새롭고 효율적인 맥락 내 학습 방법론(ICL)을 제안합니다. 이 방식은 도메인 특화 품질 추정(QE)을 통해 번역 품질을 평가하고 가장 영향력 있는 예시를 선택하는데 중점을 둡니다. 이를 통해 기존 ICL 방법론과 비교하여 번역 성능을 크게 향상시키고, 미리 학습된 mBART-50 모델을 미세 조정한 것보다 더 높은 성능을 보입니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소를 포함합니다: 예시를 선택하는 비지도 기반 탐색기(BM25)와 QE를 사용하는 검색 알고리즘입니다. 검색 알고리즘은 예시를 선택, 번역, 그리고 품질을 추정하는 단계를 통해 높은 번역 품질을 제공할 수 있는 예시 조합을 식별합니다. QE는 문장 수준에서 수행되며, 지정된 인내 임계값 내에서 번역 품질이 더 이상 향상되지 않을 때까지 반복됩니다.

- **Performance Highlights**: 독일어-영어 번역 실험에서, 이 새로운 접근 방식은 현재 최첨단 ICL 방법론과 mBART-50을 넘어서 현저히 높은 번역 품질을 보여줍니다. 특히, BM25 및 n-gram 겹침 기반의 예시 정렬 방식과 QE의 결합이 제안된 방법론의 성능을 크게 끌어올렸습니다.



### Better than Random: Reliable NLG Human Evaluation with Constrained Active Sampling (https://arxiv.org/abs/2406.07967)
Comments:
          With Appendix

- **What's New**: 이번 논문에서는 비용이 많이 들고 시간이 소요되는 인간 평가의 정확성을 높이기 위해 새로운 제약적 능동 샘플링 프레임워크(CASF)를 제안했습니다. CASF는 효율적이고 신뢰성 있는 시스템 랭킹을 구하기 위해 샘플을 선택하는 체계적인 방법을 사용합니다.

- **Technical Details**: CASF는 Learner, Systematic Sampler, Constrained Controller로 구성되어 있습니다. Learner는 샘플의 품질 점수를 예측하며, Systematic Sampler와 Constrained Controller는 낮은 중복도의 대표 샘플을 선택합니다. 각 샘플링 단계에서 선택된 샘플은 이전 단계에서 선택된 샘플과 중복되지 않으며, 인간 평가에 직접 사용됩니다.

- **Performance Highlights**: CASF는 16개의 데이터셋과 5개의 NLG 작업에서 44개의 인간 평가 지표를 기반으로 137개의 실제 NLG 평가 설정에서 테스트되었습니다. 그 결과, CASF는 93.18%의 최고 랭킹 시스템 인식 정확도를 확보했으며, 90.91%의 인간 평가 지표에서 1위 또는 2위를 차지했습니다.



### Defining and Detecting Vulnerability in Human Evaluation Guidelines: A Preliminary Study Towards Reliable NLG Evaluation (https://arxiv.org/abs/2406.07935)
- **What's New**: 새로운 인간 평가 가이드라인 데이터세트를 제공하고, 평가 가이드라인의 취약점을 탐지하기 위한 방법을 제안했습니다. 현재의 연구는 인간 평가에서 신뢰성 문제를 해결하고자 합니다.

- **Technical Details**: 3,233개의 논문을 분석한 결과, 인간 평가를 포함한 논문 중 29.84%만이 평가 가이드라인을 공개했습니다. 이 중 77.09%는 취약점을 가지고 있었습니다. 연구는 수집된 논문과 대형 언어 모델(LLM)로 생성된 가이드라인에서 취약점을 주석한 최초의 인간 평가 가이드라인 데이터세트를 구축했습니다. 취약점의 8가지 카테고리를 정의하고 평가 가이드라인 작성 원칙을 제시했습니다. 또한, LLM을 사용하여 취약점을 탐지하는 방법을 탐구했습니다.

- **Performance Highlights**: 연구에서는 평가 가이드라인의 신뢰성을 높이기 위한 8가지 취약점 카테고리(윤리적 문제, 무의식적 편향, 모호한 정의, 불명확한 평가 기준, 엣지 케이스, 사전 지식, 유연하지 않은 지침, 기타)를 정의했습니다. 또한, 체인 오브 생각(Chain of Thought, CoT) 전략을 사용하는 LLM 기반의 취약점 탐지 방법도 제안했습니다.



### Large Language Model Unlearning via Embedding-Corrupted Prompts (https://arxiv.org/abs/2406.07933)
Comments:
          55 pages, 4 figures, 66 tables

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)에서 지식을 효과적으로 '잊어버리기' 위한 간편한 프레임워크를 제안합니다. 'Embedding-COrrupted (ECO) Prompts'라는 방법은 모델의 기본 구조나 학습 없이도 원하는 데이터를 잊어버리게 할 수 있습니다.

- **Technical Details**: ECO Prompts는 두 가지 핵심 단계로 구성됩니다. 첫 번째로, 프롬프트 분류기(prompt classifier)를 이용해 잊어야 할 대상(contain content within the unlearning target)을 식별합니다. 두 번째로, 분류기가 식별한 프롬프트를 LLM에 전송해, 프롬프트가 손상된 상태(corrupted form)로 전달하여 잊혀짐 상태를 유도합니다. 손상된 프롬프트는 제로차 최적화(zeroth order optimization)를 통해 효과적으로 학습됩니다.

- **Performance Highlights**: 다양한 실험을 통해 ECO Prompts는 데이터를 잊어버려야 하는 목표를 달성하면서도 다른 일반 도메인과 관련된 도메인을 거의 영향 없이 유지하는 우수한 성능을 보였습니다. 또한, 이 방법은 최대 236B 파라미터를 가진 100개의 대형 언어 모델에 대해 추가 비용 없이 효과적임을 입증했습니다.



### Automated Information Extraction from Thyroid Operation Narrative: A Comparative Study of GPT-4 and Fine-tuned KoELECTRA (https://arxiv.org/abs/2406.07922)
Comments:
          9 pages, 2 figures, 3 tables

- **What's New**: 이번 연구는 의료 분야에서 인공지능(AI)의 통합을 통해 임상 워크플로의 자동화를 촉진하는 KoELECTRA 모델과 GPT-4 모델의 비교를 중심으로 합니다. 특히 갑상선 수술 기록에서 자동으로 정보를 추출하는 작업에 초점을 맞추고 있습니다. 기존에는 정규 표현식(Regular Expressions)에 의존하는 전통적인 방법이 있었으나, 이 연구는 이를 능가하는 자연어 처리(NLP) 기술을 활용합니다.

- **Technical Details**: 현재 의료 기록, 특히 병리 보고서에서는 자유 양식의 텍스트를 많이 사용합니다. 이런 텍스트를 처리하는 기존 방법은 정규 표현식에 크게 의존하고 있어, 다소 제한적입니다. 반면 이번 연구는 KoELECTRA와 GPT-4와 같은 고급 자연어 처리 도구를 사용해 이러한 텍스트를 효과적으로 처리하는 방법을 탐구합니다. KoELECTRA는 특히 한국어에 최적화된 모델로, 의료 데이터 처리에 더 적합할 가능성이 있습니다.

- **Performance Highlights**: 연구의 결과는 KoELECTRA 모델이 정보를 보다 정확하고 효율적으로 추출하는 데 유리하다는 점을 보여주고 있습니다. 이 모델은 특히 의료 분야의 복잡한 데이터 처리 과정에서 GPT-4보다 우수한 성능을 보입니다. 이는 곧 의료 데이터의 관리와 분석 방식을 혁신할 잠재력을 지니고 있습니다.



### DeTriever: Decoder-representation-based Retriever for Improving NL2SQL In-Context Learning (https://arxiv.org/abs/2406.07913)
- **What's New**: 최근 발표된 DeTriever라는 새로운 접근 방식을 통해 대형 언어 모델(LLM)이 포함된 문맥 학습(in-context learning, ICL)을 효과적으로 수행할 수 있도록 하는 시연 예시(demonstration examples) 선택 문제를 해결하려고 합니다. 주로 NL2SQL(자연어 질문을 구조화된 질의 언어로 번역) 작업에서 뛰어난 성능을 발휘합니다.

- **Technical Details**: DeTriever는 LLM 은닉 상태(hidden states)에서 풍부한 의미 정보를 인코딩하여 가중 조합을 학습하는 시연 예제 검색 프레임워크입니다. 모델 학습을 위해 출력 질의 간 유사성을 기반으로 예시의 상대적 이점을 추정하는 프록시 스코어(proxy score)를 제안했습니다. 이 프록시 스코어는 대조 손실(contrastive loss)을 위한 타겟으로 사용됩니다.

- **Performance Highlights**: DeTriever는 Spider와 BIRD라는 두 가지 인기 있는 NL2SQL 벤치마크 테스트에서 최첨단(SoTA) 성능을 능가했습니다. 특히, 한 번의 시연 예시만 사용한 상태(one-shot)에서 NL2SQL 작업에서 두드러진 성능 향상을 보여주었습니다.



### Exploring Self-Supervised Multi-view Contrastive Learning for Speech Emotion Recognition with Limited Annotations (https://arxiv.org/abs/2406.07900)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 최신 딥러닝과 자가 지도 학습(Self-Supervised Learning, SSL) 기법은 음성 감정 인식(Speech Emotion Recognition, SER) 성능을 상당히 개선했습니다. 그러나 정확하게 라벨링된 데이터를 충분히 얻는 것이 여전히 어렵고 비용이 많이 드는 문제입니다. 본 논문에서는 제한된 주석 데이터를 가진 상황에서 SER 성능을 향상시키기 위해 다양한 음성 표현에 적용할 수 있는 다중 뷰 SSL 사전 학습 기법을 제안합니다.

- **Technical Details**: 본 연구에서는 wav2vec 2.0, 스펙트럴 및 패럴링구이스틱(paralinguistic) 특징을 활용하여 다중 뷰 SSL 사전 학습을 수행합니다. Pairwise-CL로 명명된 프레임워크는 여러 음성 뷰별 인코더를 사전 학습하고, 이에 따라 희소한 주석 데이터로 미세 조정(fine-tuning)을 할 수 있습니다. 사전 학습은 음성 뷰의 표현 간의 대조적 SSL 손실(contrastive SSL loss)을 통해 진행됩니다. 이 프레임워크는 임베딩된 잠재 공간에서 각 발화를 정렬하도록 설계되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 제한된 주석 데이터를 가진 상황에서 무가중 평균 리콜(Unweighted Average Recall) 기준으로 최대 10%까지 SER 성능을 향상시켰습니다. 여러 실험을 통해 이 방법이 뛰어난 성능을 보임을 확인하였습니다.



### Label-aware Hard Negative Sampling Strategies with Momentum Contrastive Learning for Implicit Hate Speech Detection (https://arxiv.org/abs/2406.07886)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 이번 연구는 기존의 임플리시트(implicit) 증오 발언 감지 모델의 한계를 극복하기 위해 새로운 접근법인 '라벨 인지 하드 네거티브 샘플링 전략(Label-aware Hard Negative sampling strategies, LAHN)'을 제안합니다. LAHN은 모멘텀 통합 대조학습(momentum-integrated contrastive learning)을 사용하여 모델이 하드 네거티브 샘플로부터 세부적인 특징을 학습하도록 합니다.

- **Technical Details**: 기존의 무작위 샘플링 방식과 달리, LAHN은 앵커(anchor)와 하드 네거티브 샘플 간의 구별을 중점적으로 학습하도록 설계되었습니다. MoCo(He et al., 2020)를 참고하여 모멘텀 큐(momentum queue)를 사용, 후보 네거티브 샘플을 확장하고 상위 하드 네거티브 샘플을 추출하여 대조학습을 수행합니다. 또한, LAHN은 드롭아웃 노이즈(dropout noise) 증강을 사용함으로써 추가적인 외부 지식이나 비용 없이도 성능 향상을 이끌어냅니다.

- **Performance Highlights**: LAHN은 기존 모델에 비해 임플리시트 증오 발언 감지 성능을 크게 향상시켰습니다. 특히, 내부(in-dataset) 및 크로스 데이터셋(cross-dataset) 평가에서 뛰어난 성능을 보여주었으며, 4개의 대표적인 공공 벤치마크 데이터셋에서 최고 성능을 기록했습니다.



### Designing a Dashboard for Transparency and Control of Conversational AI (https://arxiv.org/abs/2406.07882)
Comments:
          Project page: this https URL 38 pages, 23 figures

- **What's New**: 이 논문에서는 대화형 인공지능 모델(Conversational LLMs)의 불투명성을 해결하기 위한 'TalkTuner' 시스템을 소개합니다. 이 시스템은 사용자 모델을 시각화하고 제어할 수 있는 대시보드를 제공합니다. 이를 통해 사용자는 시스템의 내부 상태를 실시간으로 확인하고, 편향된 행동을 노출하거나 제어할 수 있습니다.

- **Technical Details**: 연구팀은 개방형 대화형 언어 모델(Large Language Model, LLM)인 LLaMa2Chat-13B를 사용하여 사용자 모델의 내부 표현을 추출했습니다. 추출된 데이터는 사용자 나이, 성별, 학력 수준, 사회경제적 지위와 관련이 있으며, 이를 사용자 대시보드에 실시간으로 표시합니다. 이를 위해 'linear probes'라는 해석 가능성 기법을 사용했습니다.

- **Performance Highlights**: 사용자 연구 결과, 대시보드는 사용자가 대화형 인공지능의 응답에 대한 통찰을 제공하고, 편향된 행동을 인식하게 하며, 편향을 탐색하고 줄이는 데 도움을 주었습니다. 사용자는 시스템의 내부 상태를 볼 수 있는 것에 대해 긍정적으로 반응했으며, 이는 사용자 통제감을 높였습니다.



### BookSQL: A Large Scale Text-to-SQL Dataset for Accounting Domain (https://arxiv.org/abs/2406.07860)
Comments:
          Accepted at NAACL 2024; 20 Pages (main + appendix)

- **What's New**: 최근 텍스트 투 SQL(Text-to-SQL) 시스템을 개발하기 위한 대형 데이터셋들이 제안되었지만, 금융 및 회계 분야와 같은 중요한 도메인은 충분히 다루지 못하고 있습니다. 이를 해결하기 위해 회계 및 재무 도메인을 위한 신규 대형 Text-to-SQL 데이터셋 'BookSQL'을 제안합니다. 이 데이터셋은 100k 개의 자연어 쿼리와 SQL 쌍 및 1백만 개의 회계 데이터베이스 레코드로 구성되어 있습니다.

- **Technical Details**: BookSQL 데이터셋은 재무 전문가들과 협력하여 실제 회계 데이터베이스를 반영하게 설계했습니다. 총 27개의 서로 다른 비즈니스 데이터베이스에서 각기 35k-40k 개의 트랜잭션으로 구성되며, 전체 데이터셋은 1백만 개의 레코드를 포함합니다. 데이터베이스 스키마는 Master Transactions, Customer, Employees, Product Service, Vendor, Chart of Account, Payment Method 테이블로 구성됩니다.

- **Performance Highlights**: 기존의 최첨단 Text-to-SQL 모델(예: GPT-4)을 BookSQL 데이터셋에 적용해본 결과, 기존 대형 데이터셋(예: Spider)에서 훈련된 모델들이 BookSQL에서 상당히 낮은 성능을 보였습니다. 이는 도메인 특화 모델이 더 개발될 필요가 있다는 것을 시사합니다. BookSQL은 WikiSQL 대비 약 1.25배 많은 100k개의 Query-SQL 쌍을 가지고 있으며, 보다 복잡한 쿼리를 포함하고 있습니다.



### VALL-E R: Robust and Efficient Zero-Shot Text-to-Speech Synthesis via Monotonic Alignmen (https://arxiv.org/abs/2406.07855)
Comments:
          15 pages, 5 figures

- **What's New**: 이번 연구에서는 VALL-E R이라는 새로운 TTS (Text-to-Speech) 시스템을 제안합니다. 이는 기존 VALL-E의 단점을 보완하기 위해 개발되었으며, 특히 강력한 내구성과 효율성을 자랑합니다. 주요 개선 사항으로는 음소 일관 정렬(phoneme monotonic alignment) 방법 도입과 코덱 병합(codec-merging) 접근법이 있습니다. 이로 인해 더 정확하고 빠른 음성 합성이 가능합니다.

- **Technical Details**: VALL-E R은 음소와 음향 시퀀스 간의 연결을 강화하는 음소 일관 정렬(phoneme monotonic alignment) 전략을 채택했습니다. 이는 음향 토큰을 관련 음소와 맞출 수 있도록 제한하여 더 정밀한 정렬을 보장합니다. 또한, 코덱 병합(codec-merging) 접근법을 사용해 얕은 양자화(quantization) 층에서 불연속 코드(discrete codes)를 다운샘플링하여 디코딩 속도를 높이면서도 높은 품질의 음성을 유지합니다.

- **Performance Highlights**: VALL-E R은 음소에 대한 통제력을 향상시켜 강한 내구성을 보여줍니다. 실험 결과, 원래 음성의 WER(Word Error Rate) 수준에 가까운 결과를 도출했습니다. 또한, 자가회귀 단계(autoregressive steps)를 줄여 추론 시간을 60% 이상 단축시켰습니다.



### Dynamic Stochastic Decoding Strategy for Open-Domain Dialogue Generation (https://arxiv.org/abs/2406.07850)
Comments:
          ACL 2024 Findings

- **What's New**: 이 논문에서는 대화 생성 작업에 사용되는 기존 확률적 샘플링 방법의 한계를 극복하려는 새로운 동적 디코딩 전략(DDS)을 제안합니다. DDS는 문맥에 따라 디코딩 공간을 조절할 수 있는 기법으로, 챗봇이 다양한 시나리오에서 적응적으로 동작할 수 있도록 합니다. 이는 기존의 고정 확률적 디코딩 방식에서 발생하는 문제를 해결하기 위한 것입니다.

- **Technical Details**: DDS는 대화 생성 모델에 추가적인 다양성 예측 헤드를 도입하여, 문장 수준과 토큰 수준 모두에서 적응적인 샘플링을 가능하게 합니다. 이 예측 헤드는 디코딩 다양성을 기반으로 샘플링 과정을 안내하며, 이것은 몇 가지 매핑 함수 중 하나를 사용하여 다양성 점수를 샘플링 분포를 형성하는 온도로 변환합니다. 이 방법은 모델 추론뿐 아니라 모델 교육 단계에서도 적용되어 예측 신뢰도를 균형 있게 합니다.

- **Performance Highlights**: 사전에 훈련된 두 개의 중국어 대화 모델을 사용하여 다양한 데이터셋에서 광범위한 실험을 수행한 결과, DDS가 기존의 네 가지 확률적 디코딩 알고리즘의 성능을 크게 향상 시킬 수 있음을 확인했습니다. 인간 평가에서도 DDS가 생성한 응답의 관련성과 유창성을 유지하면서도 다양성을 크게 개선하는 것으로 나타났습니다.



### SciRIFF: A Resource to Enhance Language Model Instruction-Following over Scientific Literatur (https://arxiv.org/abs/2406.07835)
Comments:
          Submitted to NeurIPS Datasets and Benchmarks 2024

- **What's New**: 새로운 데이터셋인 SciRIFF (Scientific Resource for Instruction-Following and Finetuning)이 소개되었습니다. 이 데이터셋은 정보 추출, 요약, 질문 응답, 주장 검증, 분류 등 5가지 주요 과학 문헌 이해 능력을 포함하는 54개의 작업에 대한 137K의 지시 따르기 예시를 포함합니다. 이는 다양한 과학 분야에서 연구 문헌으로부터 정보를 추출하고 종합하는 최초의 데이터셋입니다.

- **Technical Details**: SciRIFF는 인공지능, 임상 의학 등 5개의 과학 분야에 걸쳐 있습니다. 데이터셋은 인간 주석 입력 및 출력을 통해 기존 과학 문헌 이해 데이터셋에서 파생되었습니다. 이들은 템플릿을 통해 공통된 지시 형식으로 변환되었습니다. 모델은 SciRIFF-Eval이라는 9개의 대표적인 작업을 평가 벤치마크로 사용하여 감독된 미세 조정을 수행합니다.

- **Performance Highlights**: 모델인 SciTulu는 7B 규모에서 28.1%, 70B 규모에서 6.5% 더 나은 성능을 발휘하며 일반 지시 따르기 성능에서는 기준 모델과 2% 이내의 차이를 보였습니다. 또한, 성능 향상에도 불구하고 일반적인 지시 따르기 능력은 유지했습니다. 우리는 7B와 70B 모델 및 데이터셋, 평가 코드 등을 공개할 예정입니다.



### PRoDeliberation: Parallel Robust Deliberation for End-to-End Spoken Language Understanding (https://arxiv.org/abs/2406.07823)
- **What's New**: 이번 연구에서는 PRoDeliberation이라는 새로운 방법을 소개했습니다. 이는 Connectionist Temporal Classification(CTC) 기반 디코딩 전략과 denoising objective를 활용하여 비-자가회귀(non-autoregressive) 딜리버레이션 모델을 훈련시킵니다. PRoDeliberation은 기존 자가회귀 모델보다 2-10배 낮은 지연(latency)을 달성하면서, 자동 음성 인식(ASR) 시스템의 오역을 수정할 수 있습니다.

- **Technical Details**: CTC 디코더는 음성 인식에 흔히 사용되며, 비-자가회귀 방식으로 병렬 디코딩을 통해 지연을 최적화합니다. 또한, 오염된 전사를 모델을 통해 수정하도록 요구하는 denoising 훈련 방식을 도입하여, 모델의 견고성을 높였습니다. 이 denoising 훈련은 ASR 전사를 사용하는 모든 다운스트림 작업에 적용될 수 있습니다.

- **Performance Highlights**: PRoDeliberation은 다양한 ASR 모델 크기에서 2-10배의 지연 감소를 달성했으며, Mask Predict 기반 접근 방식보다 높은 품질을 제공합니다. 또한, denoising objective를 통해 ASR 견고성을 약 0.3% 향상시켰습니다. 이는 기존 자가회귀 모델의 품질을 초과하는 결과입니다.



### Are Large Language Models Good Statisticians? (https://arxiv.org/abs/2406.07815)
Comments:
          31 pages, 10 figures,19 tables. Work in progress

- **What's New**: 대형 언어 모델(LLMs)은 수학, 물리학, 화학 등 다양한 과학 분야에서 인상적인 성과를 보였지만, 복잡한 통계 작업을 처리하는 데 있어서의 효과성은 아직 체계적으로 탐구되지 않았습니다. 이를 해결하기 위해, 통계 분석 작업을 평가하기 위한 새로운 벤치마크인 StatQA를 소개합니다. StatQA는 LLM의 전문적인 통계 작업 능력과 가설 검정 방법의 적용 가능성 평가 능력을 테스트하기 위해 11,623개의 예제를 포함합니다.

- **Technical Details**: StatQA 벤치마크는 통계 분석 작업의 적용 가능성 평가와 통계적 방법 선택 및 데이터 열 식별을 포함합니다. 또한 학습 기반 방법(GPT-4o)과 오픈소스 LLMs(LLaMA-3)의 성능을 비교하며, 다양한 프롬프트 전략 및 미세 조정 기법을 사용하여 그들의 성능을 평가했습니다.

- **Performance Highlights**: 최신 모델인 GPT-4o는 최고 64.83%의 성능을 달성했으며, 이는 상당한 개선 여지가 있음을 시사합니다. 오픈소스 LLMs는 제한된 능력을 보였지만, 미세 조정된 모델은 모든 인컨텍스트 학습 기반 방법보다 뛰어난 성능을 보였습니다. 비교 인간 실험에서는 LLM이 주로 적용성 오류를 범하는 반면, 인간은 통계 작업 혼동 오류를 주로 범하는 등 오류 유형의 현저한 차이를 강조했습니다.



### To be Continuous, or to be Discrete, Those are Bits of Questions (https://arxiv.org/abs/2406.07812)
Comments:
          ACL-2024

- **What's New**: 최근에 연속적(continuous)과 이산적(discrete) 표현 사이의 새로운 형태로 바이너리(binary) 표현이 제안되었습니다. 이 논문은 모델이 바이너리 레이블을 출력할 수 있도록 하는 접근법을 조사하고, 기존의 대비적 해싱(contrastive hashing) 방법을 확장하여 구조적 대비적 해싱(structured contrastive hashing)을 도입했습니다.

- **Technical Details**: 기존의 CKY 알고리즘을 레이블 수준(label-level)에서 비트 수준(bit-level)으로 업그레이드하고, 새로운 유사도 함수(similarity function)를 스팬 한계 확률(span marginal probabilities)을 통해 정의하였습니다. 또한, 신중하게 설계된 인스턴스 선택 전략(instance selection strategy)을 사용하는 새로운 대비 손실 함수(contrastive loss function)를 도입하였습니다.

- **Performance Highlights**: 모델은 다양한 구조적 예측 과제(structured prediction tasks)에서 경쟁력 있는 성과를 달성하였으며, 바이너리 표현이 딥 러닝의 연속적인 특성과 자연 언어의 이산적인 본질 사이의 간극을 더욱 좁히는 새로운 표현으로 고려될 수 있음을 보여주었습니다.



### PolySpeech: Exploring Unified Multitask Speech Models for Competitiveness with Single-task Models (https://arxiv.org/abs/2406.07801)
Comments:
          5 pages, 2 figures

- **What's New**: PolySpeech는 음성 인식(ASR), 음성 생성(TTS), 음성 분류(언어 식별 및 성별 식별) 작업을 지원하는 다중 작업 음성 모델을 소개하였습니다. 이 모델은 다중 모달 언어 모델을 사용하며, 음성 입력으로 의미 표현을 사용합니다. 이로 인해 다양한 작업을 단일 모델에서 효율적으로 처리할 수 있습니다.

- **Technical Details**: PolySpeech의 핵심 구조는 디코더 전용 Transformer 기반 다중 모달 언어 모델입니다. 이 모델은 음성이나 텍스트 토큰을 자기 회귀적으로 예측합니다. 음성 입력은 HuBERT 등의 자율 지도 학습 모델로부터 추출한 의미 기반의 음성 토큰을 사용합니다. 음성 재구성 방법은 고충실도의 음성을 생성할 수 있도록 설계되었습니다.

- **Performance Highlights**: PolySpeech는 다양한 작업에서 단일 작업 모델과 경쟁력 있는 성능을 보여줍니다. 다중 작업 최적화는 특정 작업에서 단일 작업 최적화보다 더 유리한 결과를 제공합니다. 이를 통해 공동 최적화가 개별 작업 성능에도 긍정적인 영향을 미친다는 것을 입증하였습니다.



### IndirectRequests: Making Task-Oriented Dialogue Datasets More Natural by Synthetically Generating Indirect User Requests (https://arxiv.org/abs/2406.07794)
- **What's New**: 새로운 연구는 자연스러운 인간 대화를 모방한 간접 사용자 요청(Indirect User Requests, IURs)을 자동으로 생성하기 위해 LLM(대형 언어 모델, Large Language Model) 기반의 파이프라인을 소개합니다. 이 연구는 자연어 이해(NLU)와 대화 상태 추적(DST) 모델의 '실제 환경' 성능을 테스트하기 위한 IndirectRequests 데이터셋을 공개했습니다.

- **Technical Details**: 연구팀은 대화 인텐트 및 슬롯 슬롯을 체계적으로 정의한 'Schema-Guided Dialog(SGD)' 접근 방식을 채택했습니다. IURs의 품질을 평가하기 위해 적절성(Appropriateness), 명확성(Unambiguity), 세계 이해(World Understanding) 세 가지 언어적 기준을 제안합니다. 연구는 GPT-3.5 및 GPT-4 모델을 사용하여 초기 IURs를 생성하고, 크라우드소싱을 통해 필터링 및 수정하여 고품질 데이터셋을 완성했습니다.

- **Performance Highlights**: 실험 결과, 최신 DST 모델의 성능이 IndirectRequests 데이터셋에서 상당히 저하됨을 보여주었습니다. 이는 IndirectRequests가 실제 환경에서의 모델 성능을 평가하는 데 도전적인 테스트베드 역할을 한다는 것을 입증합니다.



### Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs (https://arxiv.org/abs/2406.07791)
Comments:
          70 pages, around 200 figures and subfigures

- **What's New**: 새로운 연구에서는 LLM(as-a-Judge) 모델의 위치 편향(position bias)을 체계적으로 분석하고 정량화하는 프레임워크를 개발했습니다. 이 연구는 MTBench와 DevBench 벤치마크를 기반으로 22가지 작업에 대해 9개의 평가 모델과 약 40개의 답변 생성 모델을 실험하여 약 80,000개의 평가 인스턴스를 생성하였습니다. 이 포괄적 평가를 통해 평가자와 작업마다 편향의 차이가 상당함을 발견하였습니다.

- **Technical Details**: 위치 편향(position bias)은 평가 목록에서 답변의 위치에 따라 편향된 판단이 내려지는 경향을 의미합니다. 이 연구는 반복적 일관성(repetitional consistency), 위치 일관성(positional consistency), 위치 공정성(positional fairness) 등의 지표를 사용하여 위치 편향을 체계적으로 연구합니다. GPT-4 모델은 위치 일관성과 공정성에서 우수한 성과를 보였으나, 비용 효율적인 모델들이 특정 작업에서 비슷하거나 더 나은 성과를 보이는 경우도 있었습니다.

- **Performance Highlights**: 연구 결과는 GPT 시리즈가 위치 일관성과 공정성이 뛰어나며, Claude-3 모델은 일관적이지만 최근 응답을 더 선호하는 경향을 보였습니다. 또한 평가의 반복성에서 높은 일관성을 보임으로써 위치 편향이 랜덤한 변동이 아니라는 것을 확인했습니다. 반복적 일관성은 높지만 일관성이 높은 평가자가 항상 공정한 평가를 하지는 않는다는 점도 밝혀졌습니다. 예를 들어, GPT-4-0613 모델은 뛰어난 일관성을 보였지만 다른 모델에 비해 더 강한 위치 선호를 나타냈습니다.

- **Implications**: ['체계적인 프레임워크: LLM 평가자에서 위치 일관성과 선호도를 해석하는 체계적 프레임워크로 평가의 신뢰성과 확장성을 높입니다.', '평가자 모델 권장 사항: 일관성, 공정성, 비용 효율성을 균형 있게 조절할 수 있는 평가자 모델을 선택할 수 있는 상세한 권장 사항을 제공합니다.', '벤치마크 평가 개선: 이 연구에서 얻은 통찰력은 미래 벤치마크 설계와 방법론을 개선하는 데 기여합니다.', '기본 연구: 다양한 모델, 작업, 평가 유형에서 위치 편향을 명확히 함으로써 효과적인 디바이어싱(debiasing) 전략을 위한 기초를 마련합니다.']



### LT4SG@SMM4H24: Tweets Classification for Digital Epidemiology of Childhood Health Outcomes Using Pre-Trained Language Models (https://arxiv.org/abs/2406.07759)
Comments:
          Submitted for the 9th Social Media Mining for Health Research and Applications Workshop and Shared Tasks- Large Language Models (LLMs) and Generalizability for Social Media NLP

- **What's New**: 이번 논문에서는 SMM4H24 공유 작업 5에 관한 접근 방식을 제시합니다. 이 작업은 어린이의 의료 장애를 보고하는 영어 트윗의 이진 분류를 목표로 합니다. 첫 번째 접근 방식은 RoBERTa-large 모델(single model)을 미세 조정하는 것이며, 두 번째 접근 방식은 세 개의 미세 조정된 BERTweet-large 모델을 앙상블(ensemble)하는 것입니다. 두 방식 모두 검증 데이터에서는 동일한 성능을 보였으나, 테스트 데이터에서 BERTweet-large 앙상블이 더 우수한 성능을 보였습니다. 최상위 시스템은 테스트 데이터에서 F1-score 0.938을 달성하여 벤치마크(classifier)를 1.18% 초과합니다.

- **Technical Details**: 이번 작업에 사용된 주요 모델은 BioLinkBERT-large, RoBERTa-large, BERTweet-large입니다. 각 모델은 훈련 데이터셋으로 미세 조정되고 검증 데이터셋을 통해 성능이 평가되었습니다. Hyperparameter 최적화는 HuggingFace의 Trainer API와 Ray Tune 백엔드를 사용하여 수행되었고, Google Colab Pro+에서 NVIDIA A100 GPU로 실험이 진행되었습니다. 앙상블 모델은 서로 다른 초기 랜덤 시드를 사용하는 세 가지 반복 및 동일한 하이퍼파라미터로 미세 조정된 모델의 예측을 결합하여 구축되었습니다.

- **Performance Highlights**: RoBERTa-large와 BERTweet-large는 검증 데이터셋에서 유사한 성능을 보였으나, BERTweet-large 앙상블이 테스트 데이터에서 더 나은 성능을 보였습니다. 최종 모델은 SMM4H’24 Task 5에서 F1-score 0.938을 달성하여 벤치마크를 1.18% 초과했습니다. 이는 BERTweet-large의 여러 반복 실행이 데이터의 다른 측면을 포착하거나 다른 패턴을 학습하는 데 강점이 있을 수 있다는 가설을 뒷받침합니다.



### UICoder: Finetuning Large Language Models to Generate User Interface Code through Automated Feedback (https://arxiv.org/abs/2406.07739)
Comments:
          Accepted to NAACL 2024

- **What's New**: 대형 언어 모델(LLM)이 일관성 있게 UI 코드를 생성하고 시각적으로 관련된 디자인을 만드는 데 어려움을 겪는 문제를 해결하기 위해, 이 논문에서는 자동 피드백(컴파일러와 다중 모드 모델)을 사용하여 LLM이 고품질의 UI 코드를 생성하도록 유도하는 방법을 탐구합니다. 이 방식은 기존 LLM을 시작으로 자체적으로 생성한 대형 합성 데이터셋을 사용하는 모델을 반복적으로 개선합니다. 개선된 모델은 정제된 고품질 데이터셋에 대해 미세 조정되어 성능을 향상시킵니다.

- **Technical Details**: 먼저, 기존 LLM에 UI 설명 목록을 주어 대형 합성 데이터셋을 생성합니다. 그런 다음 컴파일러와 비전-언어 모델을 사용하여 이러한 샘플을 채점, 필터링, 중복 제거하여 정제된 데이터셋을 만듭니다. 이 데이터셋에서 미세 조정된 모델은 UI 코드 생성 능력을 더욱 향상시킵니다. 이 논문에서 사용된 모델은 StarCoder라는 오픈 소스 LLM에서 시작하여, StarChat-Beta 모델을 기반으로 다섯 번의 반복을 거쳐 거의 백만 개의 SwiftUI 프로그램을 생성했습니다.

- **Performance Highlights**: 평가 결과, 생성된 모델은 다운로드 가능한 다른 모든 기준 모델들을 능가하였으며, 더 큰 독점 모델들의 성능에 근접했습니다. 특히 중요한 점은 StarCoder를 기반으로 한 모델임에도 불구하고 Swift 코드 저장소가 이 모델의 훈련에서 누락되었음에도 불구하고 탁월한 성과를 냈다는 것입니다. UICoder 모델은 자연어 설명에서 SwiftUI 구현을 생성하며, 이는 텍스트-UI 코드 생성의 효과적인 해결책임을 보여줍니다.



### MultiPragEval: Multilingual Pragmatic Evaluation of Large Language Models (https://arxiv.org/abs/2406.07736)
Comments:
          8 pages, under review

- **What's New**: 최근 LLM(대규모 언어 모델)의 기능이 확장됨에 따라 단순한 지식 평가를 넘어서는 고급 언어 이해력을 평가하는 것이 중요해지고 있습니다. 이번 연구는 영어, 독일어, 한국어, 중국어를 포함한 다언어적 실용 평가를 위한 강력한 테스트 스위트인 MultiPragEval을 소개합니다. MultiPragEval은 Grice의 협력 원칙과 네 가지 대화 규칙에 따라 1200개의 질문 단위를 포함하며, LLM의 맥락 인식 및 암시적 의미 추론 능력을 심층 평가합니다.

- **Technical Details**: MultiPragEval은 영어, 독일어, 한국어, 중국어에 대한 300개의 질문 단위를 포함하여 총 1200개로 구성되었습니다. 이러한 질문은 Grice의 협력 원칙과 관련된 네 가지 대화 규칙(양, 질, 관계, 방식) 및 문자 그대로의 의미를 평가하기 위한 추가 카테고리로 구분됩니다. 또한 15개의 첨단 LLM 모델을 평가하여 맥락 인식과 실용적 이해 능력을 평가합니다.

- **Performance Highlights**: 연구 결과, Claude3-Opus가 모든 시험 언어에서 다른 모델을 크게 능가하며 분야에서 최신 상태를 확립했습니다. 오픈 소스 모델 중에서는 Solar-10.7B와 Qwen1.5-14B가 강력한 경쟁자로 나타났습니다. 이 연구는 실용적 추론에서 다언어적 평가를 선도할 뿐만 아니라, AI 시스템의 고급 언어 이해에 필요한 세부 능력에 대한 귀중한 통찰을 제공합니다.



### REAL Sampling: Boosting Factuality and Diversity of Open-Ended Generation via Asymptotic Entropy (https://arxiv.org/abs/2406.07735)
- **What's New**: 이번 연구에서는 큰 언어 모델(LLMs)에서 사실성과 다양성 사이의 균형을 잡기 위한 새로운 디코딩 방법인 REAL(Residual Entropy from Asymptotic Line) 샘플링을 제안합니다. 이 방법은 p의 적응형 기준값을 예측해, 모델이 환각(hallucination)을 일으킬 가능성이 높을 때는 p 기준값을 낮추고, 그렇지 않을 때는 p 기준값을 높여 다양성을 증진합니다.

- **Technical Details**: REAL 샘플링은 감독 없이 단계별 환각 가능성을 예측하기 위해 Token-level Hallucination Forecasting (THF) 모델을 사용합니다. THF 모델은 다양한 크기의 LLM에서 다음 토큰의 엔트로피를 외삽해 다음 토큰의 불확실성을 예측합니다. LLM의 엔트로피가 비정상적으로 높으면 환각 위험이 높은 것으로 예측되어 p 기준값을 낮추게 됩니다.

- **Performance Highlights**: FactualityPrompts 벤치마크에서 REAL 샘플링을 사용한 70M 크기의 THF 모델이 7B LLM에서 사실성과 다양성을 동시에 크게 향상시켰습니다. REAL 샘플링은 9개의 샘플링 방법보다 더 나은 성능을 보였으며, 탐욕 샘플링(greedy sampling)보다 더 사실적이고, nucleus sampling(p=0.5)보다 더 다양한 텍스트를 생성했습니다. 또한, 예측된 비대칭 엔트로피(asymptotic entropy)는 환각 탐지 작업에서도 유용한 신호로 작용할 수 있습니다.



### Sustainable self-supervised learning for speech representations (https://arxiv.org/abs/2406.07696)
- **What's New**: 이 논문은 지속 가능한 self-supervised 모델을 제안하여, 음성 표현 학습에서 데이터, 하드웨어, 알고리즘의 최적화를 통해 컴퓨팅 비용을 줄이고 환경적으로 더 책임 있는 AI를 구현하는 방안을 다룹니다. 제안된 모델은 단일 GPU를 사용하여 하루 이내에 사전 훈련을 완료할 수 있으며, downstream task에서 오류율 성능을 향상시켰습니다.

- **Technical Details**: 제안된 모델은 neural layer와 학습 최적화를 결합하여 메모리 사용량과 컴퓨팅 비용을 줄였습니다. self-supervised 학습 방법 중에서도 consistency와 self-training 접근법을 사용하였으며, 사전 훈련 단계에서 기존의 비효율적인 방법을 대신하여 효율성을 극대화하는 방식을 도입하였습니다.

- **Performance Highlights**: 자원 효율적인 baseline 대비 메모리 사용량은 한 자릿수, 그리고 컴퓨팅 비용은 거의 세 자릿수에 달하는 개선을 이루었으며, 단일 GPU에서 하루 이내에 사전 훈련을 완료할 수 있었습니다. 이는 큰 speech representation 접근법들에 비해 획기적인 효율성 개선입니다.



### Transformer Models in Education: Summarizing Science Textbooks with AraBART, MT5, AraT5, and mBAR (https://arxiv.org/abs/2406.07692)
- **What's New**: 최근 기술 발전과 인터넷 상의 텍스트 양 증가로 인해 텍스트를 효과적으로 처리하고 이해하는 도구의 개발이 시급해졌습니다. 이러한 도전에 대응하기 위해, 우리는 고급 텍스트 요약 시스템을 개발했습니다. 특히, 본 시스템은 팔레스타인 교과 과정의 11학년과 12학년 생물학 교과서를 대상으로 하고 있습니다.

- **Technical Details**: 본 시스템은 MT5, AraBART, AraT5, mBART50와 같은 최신 자연어 처리 (Natural Language Processing) 모델들을 활용하여 중요한 문장을 추출합니다. 성능 평가에는 Rouge 지표를 사용했으며, 교육 전문가와 교과서 집필자가 모델의 출력물을 평가했습니다. 이를 통해 최선의 해결책을 찾고 개선이 필요한 영역을 명확히 하려는 목표가 있습니다.

- **Performance Highlights**: 본 연구는 아랍어 텍스트 요약에 대한 솔루션을 제시하고 있으며, 아랍어 이해 및 생성 기술에 대한 연구와 개발에 새로운 지평을 열어줄 수 있는 결과를 제공합니다. 또한, 학교 교과서 텍스트를 생성 및 컴파일하고 데이터셋을 구축함으로써 아랍어 텍스트 분야에 기여하고 있습니다.



### Out-Of-Context Prompting Boosts Fairness and Robustness in Large Language Model Predictions (https://arxiv.org/abs/2406.07685)
- **What's New**: 최신 대형 언어 모델(LLMs)은 고위험 결정 과정에 점점 더 많이 사용되고 있는 반면, 여전히 사용자나 사회의 기대와 상충하는 예측을 자주 합니다. 이러한 모델의 신뢰성을 개선하기 위해 인과 추론을 도구로 활용하는 테스트 시점 전략을 제안합니다. 이 논문은 명시적으로 모델에 공정성과 강건성을 요구하는 대신, 기저의 인과 추론 알고리즘을 인코딩하는 프롬프트 설계를 통해 더 신뢰할 수 있는 예측을 이끌어냅니다. 그 구체적인 방법으로 'Out-Of-Context (OOC) prompting'을 제안합니다.

- **Technical Details**: OOC 프롬프트는 사용자의 과제 인과 모델에 대한 사전 지식을 활용하여 (임의의) 반사실적 변환을 적용하여 모델의 신뢰성을 개선합니다. 이는 추가 데이터나 재학습 없이, 공정성과 강건성을 높이는 접근법입니다. OOC 프롬프트는 사용자가 제공한 인과 가정을 바탕으로 인과 추론 알고리즘을 모사하며, 이를 통해 LLMs의 예측이 공정성과 강건성 측면에서 향상되도록 합니다.

- **Performance Highlights**: 실험적으로, 6가지 서로 다른 보호/허위 속성을 포함한 5개의 벤치마크 데이터셋을 사용하여 OOC 프롬프트가 다양한 모델 패밀리와 크기에서 공정성과 강건성에 대해 최첨단 성능을 달성함을 보여주었습니다. OOC 프롬프트는 많은 성능 저하 없이 신뢰성 있는 예측을 일관되게 생성할 수 있음을 입증하였습니다. 기존의 명시적 안전 프롬프트와 비교해 다양한 시나리오에서 더 높은 성능을 보였습니다.



### Tag and correct: high precision post-editing approach to correction of speech recognition errors (https://arxiv.org/abs/2406.07589)
Comments:
          5 pages, 3 figures, Published in Proceedings of the 17th Conference on Computer Science and Intelligence Systems (FedCSIS 2022)

- **What's New**: 이 논문은 음성 인식 오류를 교정하기 위한 새로운 후편집(post-editing) 접근 방식을 제안합니다. 이 접근 방식은 신경망 기반의 시퀀스 태거(neural sequence tagger)를 사용하여 단어별로 ASR(Automatic Speech Recognition) 가설의 오류를 교정하는 방법을 학습하고, 태거가 반환한 교정을 적용하는 교정 모듈로 구성되어 있습니다. 이 솔루션은 ASR 시스템의 아키텍처와 관계없이 적용 가능하며, 교정되는 오류에 대해 높은 정밀도 제어를 제공합니다.

- **Technical Details**: 제안된 솔루션은 신경 네트워크 기반의 시퀀스 태거를 사용하여 각 단어의 교정 여부를 학습합니다. 태거는 단어별로 오류를 탐지하고 교정 방안을 제시합니다. 이후 교정 모듈은 태거가 반환한 교정을 실제로 적용하게 됩니다. 이러한 접근 방식은 특히 제품 환경에서 중요한데, 새로운 실수를 도입하지 않고 기존 오류를 교정하는 것이 전체 결과 향상보다 더 중요한 경우가 많습니다.

- **Performance Highlights**: 결과에 따르면, 제안된 오류 교정 모델의 성능은 이전의 접근 방식과 비교하여 유사한 수준을 유지하면서도, 훈련에 필요한 자원이 훨씬 적습니다. 이는 추론 대기 시간(inference latency) 및 훈련 시간(training time)이 중요한 산업 응용 분야에서 특히 유리합니다.



### Words Worth a Thousand Pictures: Measuring and Understanding Perceptual Variability in Text-to-Image Generation (https://arxiv.org/abs/2406.08482)
Comments:
          13 pages, 11 figures

- **What's New**: 이 논문은 텍스트-이미지 변환에서 확산 모델(diffusion models)이 언어적 명령어(prompt)에 따라 이미지의 변이성을 어떻게 나타내는지 연구하고 있습니다. W1KP라는 인간 교정(calibrated) 측정 도구를 제안하여 이미지 세트 내의 변이성을 평가합니다. 이는 기존 데이터셋으로 구성한 세 가지 테스트 세트를 활용해 평가되었습니다.

- **Technical Details**: 연구진은 W1KP라는 새로운 측정 방법을 제안하였으며, 이는 기존의 이미지 쌍간 지각적 거리(metrics)를 이용해 인간이 이해하기 쉬운 형태로 교정하였습니다. 특히, DreamSim이라는 최근의 지각 거리 알고리즘을 사용했습니다. 연구 결과, W1KP는 9개의 기존 기준선을 최대 18포인트까지 능가했으며, 78%의 정확도로 인간의 평가와 일치했습니다.

- **Performance Highlights**: 연구에 따르면, 'Stable Diffusion XL', 'DALL-E 3' 및 'Imagen'과 같은 최신 확산 모델의 주요 성능 지표에 대해 평가되었습니다. 예를 들어, 'Stable Diffusion XL' 및 'DALL-E 3'의 경우 하나의 명령어가 50-200회까지 재사용 가능하지만, 'Imagen'의 경우 10-50회 재사용이 최적입니다. 또한, 텍스트 명령어의 길이, CLIP 임베딩(norm), 구체성(concreteness)에 따라 이미지의 변이성이 달라짐을 확인했습니다.



### What If We Recaption Billions of Web Images with LLaMA-3? (https://arxiv.org/abs/2406.08478)
Comments:
          * denotes equal contributions

- **What's New**: 이번 연구에서는 웹 크롤링으로 수집된 이미지-텍스트 쌍의 품질을 향상시키기 위해 LLaMA-3 모델을 사용하여 1.3억개의 이미지를 다시 캡션하는 방안을 제안합니다. 이를 통해 Recap-DataComp-1B라는 새로운 고품질 데이터셋을 구축하여, CLIP와 같은 판별 모델에서의 제로샷 성능과 텍스트-이미지 생성 모델의 사용자 텍스트 지시사항에 대한 이미지 정렬 능력을 대폭 향상시켰습니다.

- **Technical Details**: LLaMA-3-8B 모델을 미세 조정하여 LLaVA-1.5 모델을 생성하고, 이를 이용해 DataComp-1B 데이터셋의 1.3억개의 이미지를 리캡션했습니다. 이 과정에서 LLaMA-3의 언어 디코더 역할을 하며, CLIP ViT-L/14가 비전 인코더로 사용되었습니다. 모델의 성능을 검증하기 위해 MMMU와 MM-Vet 같은 멀티모달 평가 벤치마크를 활용했습니다.

- **Performance Highlights**: 우리의 LLaVA-1.5-LLaMA3-8B 모델은 벤치마크 테스트에서 이전 모델들을 크게 능가했으며, 특히 CLIP 모델들의 제로샷 성능과 텍스트-이미지 생성 모델에서 사용자 텍스트 지시사항을 따르는 이미지 생성 품질에서 큰 향상을 보였습니다.



### The Impact of Initialization on LoRA Finetuning Dynamics (https://arxiv.org/abs/2406.08447)
Comments:
          TDLR: Different Initializations lead to completely different finetuning dynamics. One initialization (set A random and B zero) is generally better than the natural opposite initialization. arXiv admin note: text overlap with arXiv:2402.12354

- **What's New**: 이 논문에서는 Hu et al. (2021)에서 도입된 Low Rank Adaptation (LoRA)에서 초기화의 역할을 연구합니다. 저자들은 두 가지 초기화 방법(하나는 B를 0으로, 다른 하나는 A를 임의로 초기화)을 비교하여 첫 번째 방법이 더 나은 성능을 나타낸다고 주장합니다.

- **Technical Details**: LoRA에서 초기화는 B를 0으로, A를 임의로 설정하거나 그 반대로 설정할 수 있습니다. 이 두 방법 모두 초기화 시점에서 BA의 곱이 0이 되어 사전 학습된 모델에서 시작하게 됩니다. 이 두 초기화 방식이 비슷해 보이지만, 첫 번째 방식은 더 큰 학습률을 사용할 수 있게 해주어 더 효율적인 학습이 가능합니다. 이는 수학적 분석과 광범위한 실험을 통해 확인되었습니다.

- **Performance Highlights**: 논문에서는 첫 번째 초기화 방법(B를 0으로, A를 임의로 설정)이 두 번째 방식보다 평균적으로 성능이 더 우수하다고 밝혔습니다. 이는 첫 번째 방식이 더 큰 학습률을 허용해 출력 불안정을 초래하지 않으면서도 학습을 더 효율적으로 진행시킬 수 있기 때문입니다. 대규모 언어 모델(LLM)에 대한 다양한 실험이 이를 검증했습니다.



### MMWorld: Towards Multi-discipline Multi-faceted World Model Evaluation in Videos (https://arxiv.org/abs/2406.08407)
- **What's New**: MMWorld는 새로운 비디오 이해 벤치마크로, Multimodal Language Language Models (MLLMs)의 다양한 실제 세계 동역학 해석 및 추론 능력을 평가하기 위해 개발되었습니다. 기존의 비디오 이해 벤치마크와는 달리, MMWorld는 다양한 학문 분야를 포괄하고, 설명, 반사상적 사고 (counterfactual thinking), 미래 예측 등 다방면의 추론을 포함합니다. 이와 같은 방대한 데이터셋을 통해 MLLMs의 '세계 모델링' 능력을 종합적으로 평가할 수 있습니다.

- **Technical Details**: MMWorld는 7개 주요 분야와 69개 세부 분야에 걸쳐 총 1910개의 비디오와 6627개의 질문-답변 쌍 및 관련 캡션으로 구성됩니다. 두 가지 데이터셋으로 나뉘어 있으며, 하나는 인간 주석(datasets)되고 다른 하나는 단일 모달리티(perception) 내에서 MLLMs를 분석하기 위한 합성 데이터셋입니다. 평가된 모델에는 2개의 사유 모델(proprietary models)과 10개의 오픈 소스 모델(open-source models)이 포함됩니다.

- **Performance Highlights**: MMWorld에서 평가된 MLLMs는 여전히 많은 도전에 직면해 있습니다. 예를 들어, GPT-4V는 52.3%의 정확도로 가장 우수한 성능을 보였지만, 이는 여전히 개선의 여지가 많음을 보여줍니다. 비디오에 특화된 네 가지 MLLMs는 무작위 추출보다도 나쁜 성능을 보였습니다. 또한, 오픈 소스 모델과 사유 모델 간에는 여전히 명확한 성능 차이가 있으며, best open-source model인 Video-LLaVA-7B는 특정 작업에서 GPT-4V와 Gemini 모델을 상당히 앞섰습니다.

- **Interesting Findings**: 사람들(비전문가)과 MLLMs을 비교한 연구에서, 문제의 난이도에 대한 사람들과 MLLMs 간의 상관관계를 발견하였습니다. MLLMs은 사람(비전문가)이 전혀 대처하지 못한 어려운 질문에 대해 합리적인 답변을 제공하면서, 동시에 사람들이 쉽게 푸는 질문에서는 어려움을 겪는 등 서로 다른 인지 및 추론 능력을 보여주었습니다. 이는 MLLMs와 인간이 서로 다른 인지 및 추론 방식을 갖고 있음을 시사합니다.



### Understanding Sounds, Missing the Questions: The Challenge of Object Hallucination in Large Audio-Language Models (https://arxiv.org/abs/2406.08402)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 이 연구에서는 대형 오디오 언어 모델(LALMs)이 오디오 관련 작업 수행 능력은 좋지만, 특정 객체의 소리 여부 식별과 같은 차별적 질문에서 약점을 보인다는 점을 지적합니다. 특히, LALMs가 객체 환각(object hallucination) 문제를 겪고 있으며, 이를 개선하기 위한 프롬프트 엔지니어링(promise engineering) 전략을 제안합니다.

- **Technical Details**: LALMs는 기존 대형 언어 모델에 오디오 인식 기능을 추가한 모델입니다. 연구에서는 AudioCaps와 CHIME-6 데이터셋을 사용하여 평가를 진행했으며, 객체 환각에 대한 평가를 위해 이항 분류(binary classification)를 수행했습니다. 모델 성능은 정확도(accuracy), 정밀도(precision), 재현율(recall), F1 스코어를 통해 측정했습니다.

- **Performance Highlights**: 연구 결과, LALMs는 오디오 캡션 작성(audio captioning) 작업에서는 전문 모델과 비슷한 수준을 보였지만, 차별적 질문에서 성능이 떨어졌습니다. 또한, LALMs의 성능은 프롬프트 디자인에 매우 민감하게 반응했습니다. 특히, 객체 환각 문제가 확인되었으며, 이 모델들은 주어진 오디오에서 정확한 정보를 추출하는 데 어려움을 겪었습니다.



### Large Language Models Must Be Taught to Know What They Don't Know (https://arxiv.org/abs/2406.08391)
Comments:
          Code available at: this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 고위험 응용 분야에 사용할 때 예측의 신뢰성을 평가하는 방법을 연구하였습니다. 저자들은 단순히 LLM을 프롬프트로 활용하는 것이 좋은 불확실성 추정을 위해 충분하지 않다고 주장하고, 대신 소량의 정답과 오답 데이터셋으로 미세 조정을 통해 일반화 성능이 좋은 불확실성 추정 모델을 구축할 수 있음을 보여주었습니다.

- **Technical Details**: 프롬프트를 통해서는 좋은 불확실성 예측을 달성하기 어렵다는 것을 입증한 후, 약 천 개의 등급 매긴 예시를 사용하여 LLM을 미세 조정하는 것으로 베이스라인보다 우수한 성능을 보여줍니다. 이 논문에서는 모델의 특징을 통해 학습하는 것이 필요하며, LoRA(저자세 연속 주입) 기법을 사용하여 대형 오픈 소스 모델에서도 가능하다고 주장합니다. 또한 강력한 보조 언어 모델(GPT 3.5 Turbo)을 이용해 정답 여부를 평가하고, 이는 인간 평가와 높은 일치를 보였습니다.

- **Performance Highlights**: 실험 결과, 천 개의 graded example로 미세 조정한 모델이 기존 베이스라인 방법을 능가했으며, 이를 통해 인간-AI 협업 환경에서 LLM의 불확실성 추정이 인간의 사용에도 큰 도움이 될 수 있음을 확인했습니다. 특히 GPT 3.5 Turbo와 인간 평가의 일치율이 높아 저비용으로도 높은 정확성을 가지는 평가 방법임을 입증했습니다.



### Speech Emotion Recognition with ASR Transcripts: A Comprehensive Study on Word Error Rate and Fusion Techniques (https://arxiv.org/abs/2406.08353)
- **What's New**: 새로운 연구는 자동 음성 인식(Auto Speech Recognition, ASR)으로 생성된 텍스트를 사용한 음성 감정 인식(Speech Emotion Recognition, SER)의 성능을 다양한 단어 오류율(WER)을 가진 텍스트로 벤치마킹했습니다. 연구에서는 IEMOCAP, CMU-MOSI, MSP-Podcast 같은 유명한 코퍼스에서 텍스트 전용 및 바이모달(bimodal) SER을 통해 다양한 융합 기술을 평가했습니다.

- **Technical Details**: 연구는 11개의 ASR 모델(Wav2Vec2, HuBERT, WavLM, Whisper 등)을 사용하여 다양한 WER을 생성하고, IEMOCAP, CMU-MOSI, MSP-Podcast 세 코퍼스를 활용하여 텍스트 전용 및 오디오와 텍스트를 결합한 SER을 수행했습니다. 또한 ASR 오류에 강한 프레임워크를 제안하여, ASR 오류 수정을 통합하고 동적 모달리티-게이티드 융합을 통해 WER을 낮추고 SER 성능을 향상시켰습니다.

- **Performance Highlights**: 제안된 프레임워크는 기존 최고의 ASR 텍스트보다 낮은 WER과 더 높은 SER 결과를 달성했습니다. 특히, 제안된 이중 단계 ASR 오류 수정과 동적 모달리티-게이티드 융합 접근 방식은 높은 WER의 부정적 영향을 줄이는 데 효과적이었습니다.



### Research Trends for the Interplay between Large Language Models and Knowledge Graphs (https://arxiv.org/abs/2406.08223)
- **What's New**: 이번 조사 논문은 대형 언어 모델(LLMs)과 지식 그래프(KGs) 사이의 상호작용을 탐구하여, AI의 이해, 추론 및 언어 처리 능력을 향상시키는 데 중점을 둡니다. 본 연구는 KG 질문 답변, 온톨로지 생성, KG 검증 및 정확성 개선을 위한 LLM의 활용 방안을 새롭게 조명합니다.

- **Technical Details**: KG-to-Text Generation(KG에서 텍스트 생성) 및 Ontology Generation(온톨로지 생성)의 다양한 방법론을 조사하며, KG Question Answering와 multi-hop question answering 등의 측면도 살펴봅니다. Pre-trained language model(사전 학습된 언어 모델)을 기반으로 한 여러 접근 방식을 포함합니다.

- **Performance Highlights**: Chen et al.이 제안한 KGTEXT 코퍼스를 활용한 방법이 KG-to-Text Generation의 성능을 크게 향상시켰고, LLMs를 통해 온톨로지를 생성 및 개선하는 다양한 시도가 성공적으로 이루어졌습니다.



### Transformer-based Model for ASR N-Best Rescoring and Rewriting (https://arxiv.org/abs/2406.08207)
Comments:
          Interspeech '24

- **What's New**: 이번 연구에서는 Transformer 기반 모델을 사용하여 N-best 가설(LIST)의 전체 컨텍스트(context)를 탐구하는 새로운 방식의 Rescore+Rewrite 모델을 제안합니다. 이 모델은 새로운 차별적 시퀀스 훈련 목적(discriminative sequence training objective)인 MQSD(Matching Query Similarity Distribution)를 도입하여 다양한 작업에서 성능을 향상시킵니다.

- **Technical Details**: ASR 시스템은 사용자가 말한 오디오를 N 개의 가설 집합으로 변환합니다. 기존의 N-best 랭킹 방법들은 개별 가설에 기반하여 순위를 재조정하지만, 새로운 모델은 N-best 가설 컨텍스트를 병렬로 처리할 수 있습니다. 본 모델은 Transformer Rescore Attention (TRA) 구조로 이루어져 있고, 별도의 음향 표현(acoustic representations)을 요구하지 않습니다. 이 모델은 cross-entropy와 MWER 손실함수(loss function)를 함께 사용하며, 학습 시 normalized probability를 생성합니다.

- **Performance Highlights**: 제안된 Rescore+Rewrite 모델은 기존의 Rescore-only 베이스라인 모델보다 성능이 뛰어나며, ASR 시스템 자체에 비해 평균적으로 8.6%의 상대적인 단어 오류율(WER) 감소를 달성했습니다. 또한, 4-gram 언어 모델 대비 더 우수한 성능을 보였습니다.



### Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark (https://arxiv.org/abs/2406.08155)
Comments:
          Our code for reproducing all our experiments is provided at this https URL

- **What's New**: 최근 논문에서는 자연어 처리(NLP)에서 중요한 역할을 하는 대형 언어 모델(LLM)과 Mixture-of-Experts (MoE) 아키텍처의 효율적인 확장 방법을 조사합니다. 특히, MoE 모델의 희소성(sparsity)을 고려한 양자화 방식이 제안되었습니다. 기존의 포스트 트레이닝 양자화(post-training quantization)방식이 MoE 모델에 직접 적용될 경우 효과가 떨어진다는 문제를 지적하고, 이를 해결하기 위한 새로운 구조 인지적 양자화(quantization) 휴리스틱을 제안합니다.

- **Technical Details**: 본 논문에서는 MoE 구조 인지적 양자화 방법(quantization heuristics)을 제안합니다. 제안된 방법은 MoE 블록, 전문가(experts), 개별 선형 가중치에 이르기까지 다양한 범위의 양자화 방식을 적용합니다. 특히, 모형의 각 부분이 필요한 가중치 비트 수에 따라 최적화되었습니다. 이를 통해 MoE 모델의 주요 가중치와 활성화를 보다 정확하게 식별하고, 이 데이터에 더 많은 비트를 할당하는 방식을 제안합니다. 또한 선형 가중치 이상점수(linear weight outlier scorer) 및 MoE 블록 점수기를 도입하여 효율성을 향상시켰습니다.

- **Performance Highlights**: 제안된 양자화 방식은 두 개의 대표적인 MoE 모델과 여섯 개의 평가 과제를 통해 광범위한 벤치마킹이 수행되었습니다. 실험 결과, 다른 MoE 구조(블록, 전문가, 선형 계층)에 따라 가중치 비트 수가 달라야 한다는 원칙이 밝혀졌습니다. 또한, 새로운 양자화 개선 방식은 기존 방법보다 더 나은 성능을 보였습니다. 특히, 가중치와 활성화 양자화(weight and activation quantization)를 결합한 실험에서는 제안된 방식이 기존의 양자화 방법들에 비해 뛰어난 효율성을 보였습니다.



### A Concept-Based Explainability Framework for Large Multimodal Models (https://arxiv.org/abs/2406.08074)
- **What's New**: 이번 연구에서는 대규모 다중 모달 모델(LMMs)의 내부 표현을 이해하기 위한 새로운 프레임워크를 제안합니다. 우리는 사전 학습된 LMM에 대해 토큰의 표현을 사전 학습 기반 접근법(dictionary learning based approach)을 통해 분석하여 다중 모달 개념(multimodal concepts)을 추출합니다. 이 개념들은 시각적 및 텍스트적으로 잘 의미가 연결되어 있습니다. 이를 통해 테스트 샘플의 표현을 해석하는 데 유용한 다중 모달 개념들을 추출할 수 있음을 보였습니다.

- **Technical Details**: 우리의 접근 방식은 입력된 특정 토큰에 대한 LMM의 내부 표현을 사전(dictionary) 학습을 통해 분해하는 것입니다. 이 과정에서 사전 내의 각 요소는 시각적 및 텍스트적 도메인 모두에서 의미 있게 연결된 개념을 나타냅니다. 이를 위해 Semi-NMF(semi-negative matrix factorization) 기반의 최적화 알고리즘을 활용하여 Multi-modal concept dictionary를 학습했습니다.

- **Performance Highlights**: 학습된 개념들은 시각적 및 텍스트적으로 의미 있게 연결되어 있으며, Qualitative 및 Quantitative 평가를 통해 다중 모달 개념(multimodal concepts)의 타당성을 검증했습니다. 실험 결과, 이 개념들은 LMM의 테스트 샘플을 해석하는 데 유용하며, 다양한 개념을 포괄하는 의미 있는 다중 모달 기초를 가지고 있음을 확인했습니다.



### Blowfish: Topological and statistical signatures for quantifying ambiguity in semantic search (https://arxiv.org/abs/2406.07990)
- **What's New**: 이 연구는 문장 임베딩(sentence embeddings)에서 모호성의 위상적 차별화가 벡터 검색 및 RAG 시스템에서 랭킹 및 설명 목적으로 활용될 수 있음을 보여줍니다. 연구팀은 모호성에 대한 작업 정의를 제안하고, 고유 데이터셋을 3, 5, 10 라인의 다양한 크기의 청크로 나누어 모호성의 시그니처를 제거할 수 있는 실험을 설계했습니다.

- **Technical Details**: 문장 임베딩의 의미 매칭은 종종 유클리드 거리(Euclidean distance), 점곱(inner product), 혹은 코사인 유사도(cosine similarity)를 사용합니다. 하지만 이러한 측정치들은 임베딩 매니폴드(manifold)가 전역적으로나 지역적으로 매끄럽지 않을 가능성 때문에 비효율적일 수 있습니다. 연구팀은 단어의 다의성(polysemy)에 대한 TDA(Topological Data Analysis)를 사용하여 단어 임베딩 매니폴드의 지역 불연속성을 해석하는 최근 연구에 기반하여 모호성의 작업 및 계산 정의를 제안합니다.

- **Performance Highlights**: 프로키 모호성(query size 10 against document size 3)과 명확한 쿼리(query size 5 against document size 10)의 비교에서 프로키 모호성은 0 및 1 기반의 호몰로지(homology) 분포에서 다른 분포를 보여줬습니다. 이를 통해 증가한 매니폴드 복잡성 또는 대략적인 불연속 임베딩 서브매니폴드(submanifolds)에 대해 논의했습니다. 이러한 결과를 새로운 유사성 점수화 전략에서 활용할 수 있는 방안을 제안합니다.



### LibriTTS-P: A Corpus with Speaking Style and Speaker Identity Prompts for Text-to-Speech and Style Captioning (https://arxiv.org/abs/2406.07969)
Comments:
          Accepted to INTERSPEECH 2024

- **What's New**: LibriTTS-P는 LibriTTS-R을 기반으로 하는 새로운 코퍼스로, 화자의 특성과 말하기 스타일을 설명하는 문장형 프롬프트(utterance-level descriptions, prompts)를 포함하고 있습니다. 이는 기존의 영어 프롬프트 데이터셋보다 더 다양한 주석(prompts)을 제공합니다.

- **Technical Details**: LibriTTS-P는 두 가지 종류의 프롬프트를 포함합니다: 화자 프롬프트와 스타일 프롬프트입니다. 화자 프롬프트는 화자의 특성을 묘사하는 반면, 스타일 프롬프트는 각 발화마다 말하기 스타일을 설명합니다. 주석은 인간이 직접 작성한 것과 합성된 것으로 구분됩니다. 스타일 프롬프트는 주파수(F0), 음절당 속도, 음량 등의 통계 데이터를 기반으로 자동 주석을 달았으며, 발화 스타일의 다섯 가지 단계 (매우 낮음(very-low), 낮음(low), 보통(normal), 높음(high), 매우 높음(very-high))로 분류됩니다. 또한 대형 언어 모델(LLM)을 활용한 데이터 증강을 수행했습니다.

- **Performance Highlights**: LibriTTS-P로 학습된 TTS 모델은 기존 데이터셋을 사용한 모델보다 더 높은 자연스러움을 달성했습니다. 또한 스타일 캡션 작업에서는 2.5배 더 정확한 단어를 생성하는 성능을 보여줬습니다.



### Political Leaning Inference through Plurinational Scenarios (https://arxiv.org/abs/2406.07964)
- **What's New**: 새로운 연구는 스페인의 바스크 지방, 카탈루냐, 갈리시아 세 지역을 대상으로 다당제 정치 분류 방식을 탐구하고 이를 좌우 이분법적 접근 방식과 비교합니다. 이 연구는 레이블이 지정된 사용자와 이들의 상호작용을 포함하는 새로운 데이터셋을 구축하여 정치 성향 감지를 위한 사용자 표현 생성 방법의 유효성을 검증합니다.

- **Technical Details**: 이 연구는 두 단계 방법론을 사용합니다. 첫 번째 단계에서는 리트윗 기반으로 비지도 학습 사용자 표현을 생성하고, 두 번째 단계에서는 이를 활용해 정치 성향 감지를 수행합니다. 또한, Relational Embeddings, ForceAtlas2, DeepWalk, Node2vec와 같은 다양한 비지도 기법을 평가하여 이들 기법의 정당 기반 정치 성향 감지에서의 성능을 비교합니다. 이 연구는 특히 극히 적은 훈련 데이터로도 효과적인 성능을 보이는 Relational Embeddings 방법의 우수성을 입증합니다.

- **Performance Highlights**: 실험 결과, Relational Embeddings를 통해 생성된 사용자 표현은 좌우 이분법적 및 다당제 정치 성향 모두에서 매우 효과적으로 작동하는 것으로 나타났습니다. 특히, 훈련 데이터가 제한적인 경우에도 뛰어난 성능을 보여줍니다. 데이터 시각화는 Relational Embeddings가 그룹 내의 복잡한 정치적 친밀도와 그룹 간의 정치적 관계를 잘 포착하는 능력을 가짐을 보여줍니다. 마지막으로, 생성된 데이터와 코드는 공개될 예정입니다.



### Toward a Method to Generate Capability Ontologies from Natural Language Descriptions (https://arxiv.org/abs/2406.07962)
- **What's New**: 이 논문은 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 능력 온톨로지(capability ontology) 모델링을 자동화하는 혁신적인 방법을 제안합니다. 전통적으로는 전문가의 수작업에 의존해야 했던 이 작업을 자연어 설명만으로 자동생성이 가능해집니다. 이 방법은 자연어 설명을 사전 정의된 프롬프트에 삽입한 후, 여러 단계에 걸쳐 자동 검증 과정을 거치게 됩니다.

- **Technical Details**: 제안된 방법은 몇 가지 중요한 단계를 포함합니다. 우선, 사용자가 제공한 자연어 설명을 LLM 프롬프트에 삽입하는 few-shot prompting 기법을 사용합니다. 생성된 온톨로지는 LLM을 이용한 반복적인 검증 과정에서 문법 검사, 모순 여부 검사, 허위 정보 및 누락된 요소 검사를 통해 자동 검증됩니다. 이러한 절차는 수작업의 노력을 크게 줄이고, 최종 인간 검토 및 수정만 필요하게 합니다.

- **Performance Highlights**: 이 방법은 기존의 수작업 방식과 비교해 시간과 노력을 크게 절감할 수 있으며, 온톨로지 모델링의 정확성과 효율성을 높입니다. 특히, LLM을 통해 고도의 자연어 처리(task)를 수행하며, prompting 기술을 통해 정확하고 관련성 높은 응답을 유도합니다.



### Guiding Frame-Level CTC Alignments Using Self-knowledge Distillation (https://arxiv.org/abs/2406.07909)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 본 논문은 자동 음성 인식(ASR) 모델의 프레임 레벨 정렬 문제를 해결하기 위해 새로운 자가 지식 증류(Self-Knowledge Distillation, SKD) 방법을 소개합니다. 기존의 교사-학생 모델을 사용하는 지식 증류와 달리, 동일한 인코더 레이어를 공유하고 서브 모델을 학생 모델로 사용하는 간단하고 효과적인 방법을 제안하였습니다.

- **Technical Details**: 제안된 SKD 방법은 Connectionist Temporal Classification(CTC) 프레임워크를 기반으로 프레임 레벨 정렬을 훈련 중에 안내합니다. 이는 중간 CTC(intermediate CTC) 방법에 기반한 새로운 지식 증류 전략을 탐구하며, 교사-학생 정렬 불일치 문제를 근본적으로 완화합니다. 또한, 블랭크(Blank) 프레임 마스킹 없이도 유용한 프레임을 효과적으로 증류할 수 있음을 검증하였습니다.

- **Performance Highlights**: 제안된 방법은 자원 효율성과 성능을 동시에 개선하는 데 효과적입니다. 실험 결과, 교사-학생 모델 간의 정렬 불일치가 SKD 환경에서 거의 문제가 되지 않음을 확인하였으며, 블랭크 프레임 마스킹 없이도 기존 방법보다 뛰어난 성능을 보였습니다.



### Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions (https://arxiv.org/abs/2406.07890)
Comments:
          Interspeech 2024

- **What's New**: 연구진은 방대한 데이터셋으로 훈련된 '기초 음성 모델(speech foundation models)'이 저자원 음성 이해 문제, 특히 아동 음성에 대해 탁월한 가능성을 갖고 있다는 것을 강조하고 있습니다. 이번 연구에서는 아동-성인 화자 구분(diarization)에 이러한 기초 음성 모델을 활용하여, 기존 화자 구분 방법에 비해 Diarization Error Rate를 39.5%, Speaker Confusion Rate를 62.3% 상대적으로 감소시킨 성과를 보여주고 있습니다.

- **Technical Details**: 연구는 화자 구분을 프레임 단위의 분류 문제로 제안하며, 이를 위해 Wav2vec 2.0, WavLM, Whisper 등의 기초 음성 모델을 활용합니다. 연구진은 다양한 음성 입력 윈도우 크기, 화자 인구 통계 및 학습 데이터 비율에 따라 모델의 성능을 평가하였습니다. 주요 기법은 음성 기본 모델을 사용하여 각 오디오 프레임에 대해 'child', 'adult', 'overlapped speech', 'silence/background noise' 라벨을 예측하는 것입니다. 이 과정에서 weight average와 1D convolutional layer들이 사용됩니다.

- **Performance Highlights**: 연구의 주요 결과는 아동-성인 화자 구분에서 기존 최신(SOTA) 방법을 뛰어넘는 성과를 보여주는 것입니다. 구체적으로, Diarization Error Rate (DER)를 39.5% 감소시키고, 다양한 인구 통계 및 적은 양의 학습 데이터에서도 높은 성능을 유지합니다. Wav2vec 2.0, WavLM, Whisper 등 여러 기초 음성 모델을 실험하여 타 모델 대비 우수한 성능을 입증하였습니다.



### An Empirical Study of Mamba-based Language Models (https://arxiv.org/abs/2406.07887)
- **What's New**: 이번 연구에서는 Mamba, Mamba-2 및 Transformer 모델들 간의 직접적인 비교를 통해 각각의 장단점을 대규모 데이터셋에서 평가합니다. 특히, 8B-parameter Mamba, Mamba-2, Transformer 모델들을 동일한 3.5T 토큰 데이터셋으로 학습시켜 결과를 분석하였습니다. 또한, Mamba-2, Attention, MLP 레이어들로 구성된 하이브리드 모델(Mamba-2-Hybrid)도 함께 평가하였습니다.

- **Technical Details**: 이번 연구는 NVIDIA의 Megatron-LM 프로젝트의 일환으로 진행되었습니다. Mamba 및 Mamba-2 모델은 Transformer 모델에 비해 훈련 및 추론 효율성이 높으며, Mamba-2-Hybrid 모델은 Mamba-2, self-attention, MLP 레이어를 혼합하여 구성되었습니다. 이 하이브리드 모델은 24개의 Mamba-2 레이어와 4개의 self-attention, 28개의 MLP 레이어로 구성되며, 다양한 자연어 처리 작업에서 평가되었습니다. 모든 모델은 동일한 데이터셋과 하이퍼파라미터로 훈련되어 공정한 비교를 가능하게 하였습니다.

- **Performance Highlights**: 순수 Mamba 모델들은 여러 작업에서 Transformer 모델을 능가하였으나, in-context learning 및 주어진 문맥에서 정보를 복사하는 능력에서는 Transformer보다 열등한 결과를 보였습니다. 반면, 8B-parameter Mamba-2-Hybrid 모델은 모든 12개의 표준 작업에서 Transformer 모델보다 평균 2.65 포인트 우수한 성능을 보였으며, 추론 시 최대 8배 빠른 성능을 예측할 수 있었습니다. 또한, 16K, 32K, 128K 시퀀스를 지원하는 추가 실험에서도 하이브리드 모델은 Transformer 모델과 유사하거나 더 나은 성능을 유지하였습니다.



### Dual-Pipeline with Low-Rank Adaptation for New Language Integration in Multilingual ASR (https://arxiv.org/abs/2406.07842)
Comments:
          5 pages, 2 figures, 4 tables

- **What's New**: 다양한 언어로 사전 학습된 다국어 자동 음성 인식(mASR) 시스템에 새로운 언어들을 통합하는 데 있어서 데이터를 적게 사용하면서도 효과적으로 통합할 수 있는 새로운 방법이 제안되었습니다. 이 논문에서 제안된 방법은 low-rank adaptation (LoRA)를 사용하는 듀얼 파이프라인(dal-pipeline) 접근법을 채택했습니다. 이를 통해 기존 언어의 성능 저하를 최소화하고, 새로운 언어를 추가하기 위한 별도의 파이프라인을 구현합니다.

- **Technical Details**: 이 논문에서는 mASR 시스템에 새로운 언어를 추가하기 위해 두 개의 데이터 흐름 파이프라인을 유지합니다. 첫 번째 파이프라인은 기존 언어를 위해 사전 학습된 매개변수들을 그대로 사용하며, 두 번째 파이프라인은 새로운 언어를 위한 언어-specific (특정 언어에 특화된) 파라미터와 별도의 디코더 모듈을 포함합니다. LoRA 기법을 적용하여 다중 헤드 어텐션(MHA)과 피드 포워드(FF) 서브 레이어에 트레인 가능한 저랭크 매트릭스를 추가합니다. 최종적으로 디코더 선택 전략을 통해 언어에 구애받지 않는 작동 모드를 제공합니다.

- **Performance Highlights**: 제안된 방법은 Whisper 모델을 19가지 새로운 언어로 확장하여 FLEURS 데이터셋에서 테스트되었습니다. 실험 결과 제안된 방법이 기존의 제로샷(Zeroshot) 및 강력한 베이스라인들과 비교하여 현저한 성능 향상을 보여주었습니다. 특히 언어 ID가 주어지지 않은 상태에서도 간단한 디코더 선택 전략을 통해 우수한 성능을 발휘했습니다.



### Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Mod (https://arxiv.org/abs/2406.07841)
- **What's New**: 온라인 미디어의 문제성 있는 콘텐츠, 특히 만화적 장난(comic mischief)을 탐지하는 도전에 대해 다룹니다. 만화적 장난은 폭력, 성인 콘텐츠 또는 풍자를 유머와 결합한 것으로, 탐지가 어렵습니다. 이를 해결하기 위해 다중모달(multi-modal) 접근 방식이 중요하다고 강조하며, 새로운 다중모달 시스템을 제안했습니다. 또한 이를 위한 새로운 데이터셋도 공개했습니다.

- **Technical Details**: 제안된 시스템은 비디오, 텍스트(자막 및 캡션), 오디오의 세 가지 모달리티를 포함한 데이터셋을 이용합니다. HIerarchical Cross-attention model with CAPtions (HICCAP)을 설계하여 이 모달리티들 간의 복잡한 관계를 포착하고자 했습니다. 다양한 도메인의 비디오 클립과 오디오 클립, 설명을 통해 모델을 사전학습(pretrain)하고, Kinetics-400, HowTo100M, Violent Scenes 데이터셋을 사용했습니다. 실험은 A100 GPU에서 PyTorch를 이용하여 수행되었으며, 최적의 모델을 찾기 위해 30 에포크를 진행했습니다.

- **Performance Highlights**: 제안된 접근 방식은 robust baselines와 state-of-the-art 모델에 비해 만화적 장난 탐지 및 유형 분류에서 상당한 개선을 보여주었습니다. UCF101, HMDB51, XD-Violence 데이터셋에서 우리 모델은 다른 최신 접근 방식들에 비해 뛰어난 성능을 입증했습니다.



### Tell Me What's Next: Textual Foresight for Generic UI Representations (https://arxiv.org/abs/2406.07822)
Comments:
          Accepted to ACL 2024 Findings. Data and code to be released at this https URL

- **What's New**: 새로운 모바일 앱 UI 프리트레이닝 방법인 Textual Foresight가 제안되었습니다. Textual Foresight는 현재 UI 화면과 지역적인 액션을 기반으로 미래의 UI 상태에 대한 전반적인 텍스트 설명을 생성하는 방식입니다. 이를 통해 현존하는 최고 성능 모델인 Spotlight를 뛰어넘는 성능을 발휘하면서도 학습 데이터는 28배 적게 사용합니다.

- **Technical Details**: Textual Foresight는 UI화면과 요소 간의 상호작용을 이해하고 이를 기반으로 미래의 UI 상태를 설명하는 목표로 설계된 프리트레이닝 목표입니다. 이는 (state, action) 예제를 통해 요소의 가능성을 암묵적으로 학습하며, 지역적 의미와 전반적인 UI 의미를 함께 이해해 캡션을 디코딩하도록 요구합니다. BLIP-2를 기반으로 프레임워크를 구축했으며, OpenApp이라는 새로운 데이터셋을 사용합니다.

- **Performance Highlights**: Textual Foresight는 screen summarization(화면 요약) 및 element captioning(요소 캡셔닝) 작업에서 최고의 평균 성능을 달성했으며, 이는 전반적인 UI 특징과 지역적 UI 특징 모두 학습해야 합니다. 기존의 Spotlight보다 28배 적은 데이터를 사용하면서 5.7% 더 나은 평균 성능을 기록했습니다.



### Spoof Diarization: "What Spoofed When" in Partially Spoofed Audio (https://arxiv.org/abs/2406.07816)
Comments:
          Accepted to Interspeech 2024

- **What's New**: 이 논문은 Partial Spoof (PS) 시나리오에서 'Spoof Diarization'이라는 새로운 작업을 정의합니다. 이 작업은 스푸핑된 부분이 언제 발생했는지를 결정하는 것으로, 스푸핑 영역을 찾아내고 이를 다른 스푸핑 방법에 따라 클러스터링하는 것을 포함합니다. Countermeasure-Condition Clustering (3C) 모델을 제안하여 이 작업을 수행하는 방법을 탐구했습니다.

- **Technical Details**: Spoof Diarization 작업은 PS 시나리오에서 기존의 바이너리 탐지 (binary detection)와 로컬라이제이션 (localization)을 확장하여, 스푸핑된 세그먼트를 다양한 스푸핑 방법에 따라 구별하고 분류하는 것을 목표로 합니다. 이를 위해 세 가지 라벨링 스킴을 사용해 효과적으로 카운터메저 (countermeasure)를 훈련시키는 방법을 탐구했으며, 스푸프 로컬라이제이션 예측을 사용하여 다이어리제이션 성능을 향상시켰습니다.

- **Performance Highlights**: 이번 연구는 단일 오디오 파일당 하나의 화자와 오라클 (oracle)의 스푸핑 방법만 있는 제한된 시나리오에서도 작업의 높은 복잡성을 나타냅니다. 실험 결과, 스푸핑 메소드에 대한 구체적인 식별이 가능한 시스템은 훨씬 더 현실적인 포렌식 상황에서 유용할 것으로 보입니다. 



### Collective Constitutional AI: Aligning a Language Model with Public Inpu (https://arxiv.org/abs/2406.07814)
- **What's New**: 이번 연구에서는 언어 모델(문구 모델, LM)의 행동을 결정하는 데 있어 더 넓은 대중의 의견을 반영하는 Collective Constitutional AI(CCAI) 방법을 제안합니다. 이것은 LM 개발자가 단독으로 모델의 행동을 결정해서는 안 된다는 인식 확산에 따른 것입니다. CCAI는 대중의 의견을 수집하고 이를 통합하여 LM을 미세 조정하는 다단계 프로세스를 마련합니다.

- **Technical Details**: CCAI는 Polis 플랫폼을 사용해 온라인 토론을 통해 대중의 선호를 수집하고, 헌법과 같은 자연어 원칙으로 이것을 언어 모델에 통합합니다. 이는 기존에 제시된 Constitutional AI를 발전시킨 것입니다. 연구팀은 이를 통해 미국 성인을 대표하는 견본을 대상으로 데이터를 수집해 'Public' 헌법을 만들고, 이를 반영한 모델과 표준 헌법을 사용한 모델을 비교했습니다.

- **Performance Highlights**: CCAI로 훈련된 모델은 9개의 사회적 차원에서 편견이 더 적었으며, 언어, 수학, 임무 성능 평가에서는 기존 모델과 동일한 성능을 유지했습니다. 특히 논란이 되는 주제에 대해 모델의 반응이 긍정적으로 재구성되는 경향을 보여줍니다. 이는 대중의 의견을 반영해 한층 공정하고 편견이 줄어든 LM 개발이 가능함을 시사합니다.



### A Critical Look At Tokenwise Reward-Guided Text Generation (https://arxiv.org/abs/2406.07780)
- **What's New**: 최근 연구는 인간 피드백을 통한 강화 학습(RLHF)을 사용하여 대형 언어 모델(LLMs)을 개선하는 방법을 탐구하고 있습니다. 새로운 연구에서 제안된 접근 방식은 부분 시퀀스에서 훈련된 Bradley-Terry 보상 모델을 사용하여 부분 시퀀스에 대한 토큰별 정책을 유도하는 것입니다.

- **Technical Details**: 본 연구는 전체 시퀀스에서 훈련된 보상 모델이 부분 시퀀스를 평가하는데 적합하지 않다는 점을 발견하였습니다. 이를 해결하기 위해 Bradley-Terry 보상 모델을 부분 시퀀스에서 Explicitly 훈련하고, 디코딩 시간 동안 유도된 토큰별 정책을 샘플링합니다. 이 모델은 두 개의 서로 다른 RLHF 정책의 비율에 비례하는 텍스트 생성 정책을 제안합니다.

- **Performance Highlights**: 제안된 접근 방식은 기존의 토큰별 보상 기반 텍스트 생성(RGTG) 방법보다 우수한 성능을 보여주며, 대형 언어 모델의 대규모 파인튜닝 없이도 강력한 오프라인 베이스라인과 유사한 성능을 달성합니다. 최신 LLM(예: Llama-2-7b)에서 실행된 실험 결과, 이 방법이 이론적 통찰과 일치하는 성능 향상을 보여줍니다.



### On Trojans in Refined Language Models (https://arxiv.org/abs/2406.07778)
- **What's New**: 최근에 발표된 논문에서 자연어처리 모델, 특히 대형 언어 모델(LLM)의 데이터 중독 공격(data-poisoning)과 이에 대한 방어를 다루고 있습니다. LLM의 보안성과 신뢰성 문제가 대두되는 상황에서, 제품 리뷰의 감정 분석 등의 특정 응용을 위해 모델을 정제할 때 트로이 목마(Trojan)를 삽입할 수 있다는 점을 강조합니다.

- **Technical Details**: 논문은 트랜스포머 기반 LLM을 대상으로 하는 백도어 위협(backdoor threat)과 이들의 변형 형태를 실험적으로 분석합니다. 예를 들어, 백도어 트리거가 명령 프롬프트의 시작, 끝, 고정 위치 또는 무작위 위치에 삽입되는 경우의 공격 성공률을 비교합니다. 또한, 영화 리뷰 도메인에서 다른 제품 리뷰로의 공격 전이(transference)에 대해서도 탐구합니다. 백도어 공격은 '클린 레이블'(clean label) 및 '더티 레이블'(dirty label) 방식으로 나뉘며, 각각의 공격 방식에 따른 효과를 분석합니다.

- **Performance Highlights**: 두 가지 방어 시나리오를 위한 간단한 방어 방법을 실험적으로 평가한 결과, 단어 빈도 기반의 방어(word-frequency based defense)가 효과적임을 확인했습니다. 이 방어 방법은 백도어를 탐지하고 트리거 토큰을 식별하는 데 유용하다고 합니다. 기존 연구들이 백도어 공격의 효율성에 대해 실험을 충분히 하지 않은 점을 지적하며, 본 논문은 이를 보완하기 위해 다양한 공격 구성(hyperparameter choices) 및 운영 시나리오에 따른 공격 성공률을 조사했습니다.



### The MuSe 2024 Multimodal Sentiment Analysis Challenge: Social Perception and Humor Recognition (https://arxiv.org/abs/2406.07753)
- **What's New**: MuSe 2024에서는 새로운 멀티모달 감정 및 감성 분석 문제 두 가지를 제시합니다. 첫 번째는 Social Perception Sub-Challenge (MuSe-Perception)로, 참가자들은 제공된 오디오-비주얼 데이터 기반으로 개개인의 16가지 사회적 속성(주장력, 지배력, 호감도, 진실성 등)을 예측해야 합니다. 두 번째는 Cross-Cultural Humor Detection Sub-Challenge (MuSe-Humor)로, 이는 Passau Spontaneous Football Coach Humor (Passau-SFCH) 데이터셋을 확장하여 다국적 및 다문화적 맥락에서 자발적인 유머 감지 문제를 다룹니다.

- **Technical Details**: MuSe 2024의 주요 목표는 멀티모달 감정 분석, 오디오-비주얼 감정 컴퓨팅, 연속 신호 처리, 자연어 처리 등 여러 연구 분야의 전문가들이 협업할 수 있는 플랫폼을 제공하는 것입니다. 이 베이스라인 논문에서는 각 서브 챌린지 및 해당 데이터셋, 각 데이터 모달리티에서 추출된 특징, 챌린지 베이스라인을 자세히 설명합니다. 베이스라인 시스템으로는 여러 Transformers와 전문가가 설계한 특징을 사용하여 Gated Recurrent Unit (GRU)-Recurrent Neural Network (RNN) 모델을 훈련시켰습니다.

- **Performance Highlights**: 베이스라인 시스템은 MuSe-Perception에서 평균 Pearson의 상관 계수($\rho$) 0.3573을, MuSe-Humor에서는 Area Under the Curve (AUC) 값 0.8682를 달성하였습니다.



### A Labelled Dataset for Sentiment Analysis of Videos on YouTube, TikTok, and Other Sources about the 2024 Outbreak of Measles (https://arxiv.org/abs/2406.07693)
Comments:
          19 pages

- **What's New**: 이 논문은 2024년 1월 1일부터 2024년 5월 31일까지 인터넷에 게시된 홍역(measles) 발병 관련 4011개의 비디오 데이터를 포함한 데이터셋을 소개합니다. 이 데이터셋은 YouTube와 TikTok에서 주로 수집되었으며(각각 48.6%, 15.2%), Instagram, Facebook, 다양한 글로벌 및 지역 뉴스 사이트도 포함됩니다. 각 비디오에 대해 URL, 포스트 제목, 포스트 설명, 비디오 게시 날짜 등의 속성이 포함되어 있습니다.

- **Technical Details**: 데이터셋을 개발한 후, VADER를 사용한 감정 분석(sentiment analysis), TextBlob을 사용한 주관성 분석(subjectivity analysis), DistilRoBERTa-base를 사용한 세분화된 감정 분석(fine-grain sentiment analysis)이 수행되었습니다. 비디오 제목과 설명을 긍정적, 부정적, 중립적 감정 클래스와, 매우 주관적, 중립적 주관적, 거의 주관적이지 않은 클래스, 그리고 공포(fear), 놀라움(surprise), 기쁨(joy), 슬픔(sadness), 분노(anger), 혐오(disgust), 중립 등의 세분화된 감정 클래스로 분류했습니다. 이러한 결과는 머신 러닝 알고리즘의 훈련 및 테스트에 사용할 수 있는 속성으로 제공됩니다.

- **Performance Highlights**: 이 논문은 제시된 데이터셋을 통해 감정 및 주관성 분석, 그리고 다른 응용 분야에서 사용할 수 있는 열린 연구 질문 목록을 제공합니다. 이는 앞으로 연구자들이 홍역 발병 관련 데이터 분석에 큰 기여를 할 수 있도록 돕습니다.



### OPTune: Efficient Online Preference Tuning (https://arxiv.org/abs/2406.07657)
Comments:
          16 pages, 7 figures

- **What's New**: 이번 연구에서는 Human Feedback(인간 피드백)을 활용한 강화 학습(RLHF)을 통해 대형 언어 모델(LLM)을 더욱 효율적으로 인간의 선호에 맞출 수 있는 새로운 방법을 제안합니다. 특히, 동적으로 정보가 풍부한 응답을 샘플링하는 온라인 환경에 적합한 데이터 탐색 전략(OPTune)을 소개하여 비용 및 훈련 속도의 문제를 해결하고자 합니다.

- **Technical Details**: OPTune은 사전에 준비된 인간 피드백 없이, 동적으로 각 생성된 응답의 유용성에 따라 데이터를 재샘플링하고 재학습하는 방식을 채택합니다. 이 방식은 최신 LLM 정책에 따라 낮은 보상을 받은 응답들을 선별하고, 이를 재생성하여 보다 높은 품질의 학습 신호를 제공합니다. 또한, OPTune은 응답 쌍의 유틸리티에 가중치를 부여하여 학습 목표를 최적화합니다. 이를 통해 데이터 생성 비용을 절감하면서도 온라인 RLHF의 학습 효율성을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, OPTune-d LLM은 표준 선호 튜닝보다 1.27-1.56배 빠른 훈련 속도를 보이며, 여전히 높은 품질의 응답을 생성함으로써 성능 향상을 달성했습니다. 또한, MMLU, GSM8k, TruthfulQA 등의 벤치마크 테스트와 GPT-4를 활용한 인간 평가에서도 높은 평가를 받았습니다.



### AIM: Let Any Multi-modal Large Language Models Embrace Efficient In-Context Learning (https://arxiv.org/abs/2406.07588)
- **What's New**: 최근 발표된 논문에서 새로운 프레임워크 AIM(이미지 정보 집합을 통한 다중모달 데모스트레이션)을 소개했습니다. 이 프레임워크는 다중모달 대형 언어 모델(MLLMs)이 다양한 모달리티의 데모스트레이션을 읽지 못하는 문제를 해결하고, 하드웨어에 부담을 주지 않으면서 ICL(In-context Learning)을 가능하게 합니다.

- **Technical Details**: 전통적인 MLLMs는 단일 이미지 데이터셋을 대상으로 훈련되었으며, 다중모달 데모스트레이션을 읽고 처리하는 데 어려움이 있었습니다. AIM 프레임워크는 동결된 백본 MLLM을 사용하여 각 이미지-텍스트 데모스트레이션을 읽고, 텍스트 상단의 벡터 표현을 추출합니다. 이 벡터는 이미지-텍스트 정보가 자연스럽게 융합된 형태로, AIM은 이를 LLM이 수용할 수 있는 가상 토큰으로 변환합니다. 이 가상 토큰은 각각의 다중모달 데모스트레이션의 변형판으로 작동하며, 현재 쿼리에 응답하도록 MLLM에 입력됩니다.

- **Performance Highlights**: AIM 프레임워크는 이미지를 포함한 다중모달 데모스트레이션을 사실상 텍스트 데모스트레이션으로 감소시켜 어떤 MLLM에도 적용할 수 있게 합니다. 또한, 동결된 MLLM을 사용하므로 파라미터 효율적이며, 공개된 다중모달 웹 코퍼스에서 훈련하여 테스트 작업과 연관이 없습니다.



### BrainChat: Decoding Semantic Information from fMRI using Vision-language Pretrained Models (https://arxiv.org/abs/2406.07584)
- **What's New**: 이 논문은 BrainChat이라는 새로운 생성적 프레임워크를 제시하여 뇌 활동으로부터 의미 정보를 디코딩하는 작업을 수행합니다. 특히 fMRI 데이터를 이용한 질문 응답(fMRI question answering, fMRI QA)과 캡션 생성(fMRI captioning)에 집중합니다. BrainChat은 현재까지의 최고 수준 방법들보다 뛰어난 성능을 보이며, 제한된 데이터 상황에서도 fMRI-텍스트 쌍만으로도 고성능을 발휘할 수 있습니다.

- **Technical Details**: BrainChat은 CoCa라는 사전 훈련된 비전-언어 모델을 활용하여 설계되었습니다. Masked Brain Modeling이라는 자가 지도 학습 방법을 통해 fMRI 데이터를 잠재 공간에서 더 압축된 표현으로 인코딩합니다. 이후, contrastive loss를 적용하여 fMRI, 이미지, 텍스트 임베딩 간의 표현을 정렬합니다. fMRI 임베딩은 cross-attention layers를 통해 생성적 Brain Decoder에 매핑되며, 캡션 손실을 최소화하는 방식으로 텍스트 콘텐츠를 생성합니다.

- **Performance Highlights**: BrainChat은 fMRI 캡션 생성 작업에서 최근의 최고 수준 방법들을 능가합니다. 또한, 처음으로 fMRI 질문 응답(fMRI QA) 작업을 도입하여 fMRI 데이터에 기반한 관련 답변을 생성하는 데 성공했습니다. 이는 상호작용적인 의미 정보 디코딩을 가능하게 하여 임상적 응용 가능성을 크게 높입니다.



### Inference Acceleration for Large Language Models on CPUs (https://arxiv.org/abs/2406.07553)
- **What's New**: 최근 몇 년 동안, 대형 언어 모델(large language models)은 다양한 자연어 처리 작업에서 놀라운 성능을 보여주고 있습니다. 그러나 실제 응용 프로그램에 이러한 모델을 배포하려면 효율적인 추론 솔루션이 필요합니다. 이 논문에서는 CPU를 사용하여 대형 언어 모델의 추론을 가속화하는 방법을 탐구합니다. 특히, 병렬 처리 접근 방식을 통해 처리량을 향상시키는 방법을 소개합니다.

- **Technical Details**: 논문에서 제안한 방법은 두 가지 주요 요소로 구성됩니다: 1) 최신 CPU 아키텍처의 병렬 처리 기능을 활용, 2) 추론 요청을 배치(batch) 처리. 이로 인해 긴 시퀀스와 더 큰 모델에서 더 큰 성능 개선이 확인되었습니다. 또한, NUMA 노드 격리를 통해 동일한 기기에서 다중 작업자를 실행할 수 있어 토큰/초 단위를 더욱 개선할 수 있습니다. 표 2에서는 4명의 작업자로 4배의 추가 개선을 확인할 수 있었습니다.

- **Performance Highlights**: 가속화된 추론 엔진은 초당 생성된 토큰(token per second)에서 18-22배의 개선을 보여주었으며, LLM의 추론을 위한 CPU 사용은 전력 소비를 48.9% 줄일 수 있다는 계산 결과를 제시했습니다.



### 3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination (https://arxiv.org/abs/2406.05132)
Comments:
          Project website: this https URL

- **What's New**: 이 논문에서는 3D-GRAND라는 대규모 데이터셋을 소개합니다. 이 데이터셋은 40,087개의 가정 장면과 620만개의 장면-언어 지시(instructions)로 이루어져 있으며, 밀도 있게 그라운딩된(grounded) 데이터가 포함되어 있습니다. 또한, 3D-POPE 벤치마크를 제안하여 3D-LLM의 헛소리(hallucination)를 평가하고, 공정한 비교를 가능하게 만듭니다.

- **Technical Details**: 3D-GRAND는 가정 장면과 장면-언어 지시가 밀도 있게 연결된 데이터셋입니다. 이를 통해 3D-LLM의 그라운딩 능력을 크게 향상시키고 헛소리를 줄일 수 있습니다. 3D-POPE는 3D-LLM의 헛소리를 평가하는 시스템적인 프로토콜로, 객체 존재 여부를 평가 질문 형태로 모델에게 묻고 응답을 평가합니다.

- **Performance Highlights**: 3D-GRAND와 3D-POPE를 활용한 실험 결과, 데이터셋 크기와 3D-LLM 성능 간의 스케일링 효과를 확인했습니다. 이 데이터셋을 활용한 튜닝은 3D-LLM의 헛소리를 줄이고 그라운딩 정확도를 높입니다. 또한, 대규모 합성 데이터로 훈련된 모델이 실제 세계의 3D 스캔에서도 잘 작동한다는 초기 신호를 발견했습니다.




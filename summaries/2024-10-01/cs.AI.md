New uploads on arXiv(cs.CL)

### LML: Language Model Learning a Dataset for Data-Augmented Prediction (https://arxiv.org/abs/2409.18957)
Comments:
          First version

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 분류 작업에 활용하기 위한 새로운 접근 방식을 소개합니다. 전통적인 머신 러닝(ML) 모델들과는 달리, LLMs를 사용하여 데이터 정리(data cleaning)와 특징 공학(feature engineering)을 간소화합니다.

- **Technical Details**: 이 논문은 'Language Model Learning (LML)'이라는 새로운 개념과 'Data-Augmented Prediction (DAP)'이라는 새로운 방법을 제안합니다. LLM이 데이터를 탐색하고 이해하며 분류를 결정하는 방식으로 분류를 수행합니다. DAP 과정에서는 데이터 요약을 사용하여 자동으로 쿼리를 생성하고, 이를 통해 관련 데이터 행을 검색한 후, 최종적인 분류 결과를 생성합니다.

- **Performance Highlights**: 테스트 사례에서 시스템의 정확도가 90%를 초과하여 기존 ML 모델을 다양한 시나리오에서 초월할 가능성을 입증하였습니다. 사용자가 예측의 논리를 검토할 수 있도록 'Explainable Machine Learning Model'로 행위하는 문구를 프롬프트에 포함시킴으로써 예측의 해석 가능성을 향상시켰습니다.



### Ruler: A Model-Agnostic Method to Control Generated Length for Large Language Models (https://arxiv.org/abs/2409.18943)
- **What's New**: 이 논문에서는 대형 언어 모델의 응답 길이를 제어하는 능력을 살펴보기 위해 Target Length Generation Task (TLG)를 제안합니다. 이를 통해 모델이 사용자의 요구에 맞게 특정 길이의 응답을 생성하는 데 어려움을 겪는 문제를 해결하고자 합니다.

- **Technical Details**: 연구진은 정확한 응답 길이를 평가하기 위해 두 가지 메트릭을 설계했습니다: Precise Match (PM)와 Flexible Match (FM). 또한, Ruler라는 새로운 모델 비종속적 접근 방식을 도입하여 Meta Length Tokens (MLTs)를 사용해 대형 언어 모델이 길이 제한이 있는 지침을 따를 수 있도록 개선합니다.

- **Performance Highlights**: Ruler는 다양한 대형 언어 모델에서 Target Length Generation Task를 수행할 때 평균 27.97의 PM 개선과 29.57의 FM 개선을 보여주며, Ruler의 효율성과 일반화 능력을 입증하기 위한 광범위한 실험 결과를 포함합니다.



### AIPatient: Simulating Patients with EHRs and LLM Powered Agentic Workflow (https://arxiv.org/abs/2409.18924)
Comments:
          42 pages, 6 figures, 7 tables

- **What's New**: AIPatient, an advanced simulated patient system, leverages Large Language Models (LLM) and integrates a Knowledge Graph (KG) from Electronic Health Records (EHRs) to enhance clinical decision-making simulations in medical education.

- **Technical Details**: AIPatient utilizes the AIPatient Knowledge Graph (AIPatient KG) sourced from the MIMIC-III database, creating a diverse cohort of 1,495 clinically relevant patients. It employs the Reasoning Retrieval-Augmented Generation (Reasoning RAG) framework, which involves six LLM-powered agents for tasks such as retrieval, KG query generation, and summarization.

- **Performance Highlights**: The AIPatient system achieved an accuracy of 94.15% in EHR-based medical Question Answering (QA) and demonstrated high readability and robustness, making it suitable for diverse applications in medical education and system integration.



### Soft Measures for Extracting Causal Collective Intelligenc (https://arxiv.org/abs/2409.18911)
Comments:
          Camera-ready version accepted for publication in the EMNLP 2024 Workshop NLP4Science

- **What's New**: 이 연구는 복잡한 사회 시스템을 설명하기 위해 집합적 지능(collective intelligence)을 이해하고 모델링하는 중요성을 강조하며, 대규모 언어 모델(large language models, LLMs)을 이용하여 퍼지 인지 맵(fuzzy cognitive maps, FCMs) 추출을 자동화하는 방법을 제안합니다.

- **Technical Details**: 연구에서는 새로운 그래프 기반 유사도 측정(graph-based similarity measures)을 도입하고, 이를 인간 평가와 비교하기 위해 Elo 등급 시스템(Elo rating system)을 적용합니다. FCM의 미세한 뉘앙스(capture nuances)를 포착하는 데 한계가 있다는 것이 강조되며, LLM을 미세 조정(fine-tuning)함으로써 성능을 향상시킬 수 있지만, 기존 측정 방식은 여전히 부족합니다.

- **Performance Highlights**: 결과는 인간 평가와 긍정적인 상관관계를 보이며, 하지만 가장 성능이 좋은 측정 방법조차 FCM의 복잡성을 완전히 포착하지 못하는 한계를 보입니다. 이 연구는 FCM 추출을 위한 부드러운 유사도 측정(soft similarity measures)의 필요성을 강조하며, 자연어 처리(NLP)와 함께 집합적 지능 모델링을 발전시킵니다.



### IDGen: Item Discrimination Induced Prompt Generation for LLM Evaluation (https://arxiv.org/abs/2409.18892)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 평가를 위한 Item Discrimination (ID) 이론에서 영감을 받은 새로운 프롬프트 합성 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 모델의 능력에 따라 평가 세트를 지속적으로 업데이트하고 정제할 수 있도록 하는 것을 목표로 합니다. 이 프레임워크는 폭과 세부성을 모두 우선시하여, LLMs의 능력을 포괄적으로 평가할 수 있는 프롬프트를 생성합니다. 또한, 프롬프트의 분별력(prompt discrimination)과 난이도 점수를 예측하기 위한 두 가지 모델을 개발하여 데이터 합성을 지원합니다.

- **Performance Highlights**: 생성된 데이터는 다섯 개의 최신 모델(SOTA models)에 대한 평가에 적용되었으며, 평균 점수는 51.92로, 분산은 10.06을 기록하였습니다. 이전 작업(SELF-INSTRUCT 및 WizardLM)이 67점을 초과하며 3.2 이하의 분산을 기록한 것과 비교할 때, 본 프레임워크에서 생성한 데이터는 더 도전적이며 분별력이 뛰어난 것으로 나타났습니다. 3,000개 이상의 정교하게 제작된 프롬프트 데이터를 공개할 예정입니다.



### Suicide Phenotyping from Clinical Notes in Safety-Net Psychiatric Hospital Using Multi-Label Classification with Pre-Trained Language Models (https://arxiv.org/abs/2409.18878)
Comments:
          submitted to AMIA Informatics Summit 2025 as a conference paper

- **What's New**: 이번 연구는 정신과 고위험 환경에서 자살 사건을 정확하게 식별하고 분류하여, 자살 예방 조치를 개선하고 운영 부담을 감소시키며 치료 품질을 향상시키는 방법을 제시합니다.

- **Technical Details**: 해당 연구에서는 두 가지 미세 조정 전략(단일 라벨 다수 및 단일 다중 라벨)을 사용하여 500개의 주석이 달린 정신과 평가 노트를 기반으로 네 가지 BERT(Bidirectional Encoder Representations from Transformers) 모델의 성능을 평가하였습니다. 노트는 자살적 사고(SI), 자살 시도(SA), 자살 노출(ES), 비자살적 자기 상해(NSSI)로 라벨링되었습니다.

- **Performance Highlights**: RoBERTa 모델이 binary relevance(이진 관련성) 방법을 사용하여 다른 모델보다 뛰어난 성능을 발휘하여 accuracy(정확도)가 0.86, F1 score가 0.78로 나타났습니다. MentalBERT는 F1 score가 0.74로 BioClinicalBERT의 0.72를 초과하였으며, 단일 다중 라벨 분류기로 미세 조정된 RoBERTa는 0.88의 정확도와 0.81의 F1 score로 성능이 더욱 향상되었습니다.



### Individuation in Neural Models with and without Visual Grounding (https://arxiv.org/abs/2409.18868)
- **What's New**: 이 논문에서는 CLIP 모델과 FastText, SBERT와 같은 텍스트 전용 모델 간의 individuating (개별화) 정보 인코딩의 차이를 보여줍니다.

- **Technical Details**: CLIP 모델이 제공하는 latent representations (잠재 표현)을 연구하며, 기초(substrates), 미세한 집합(granular aggregates), 다양한 수의 객체에 대한 정보를 분석합니다.

- **Performance Highlights**: CLIP 임베딩은 텍스트 전용 데이터로 훈련된 모델들보다 individuating (개별화)에서 정량적 차이를 더 잘 포착하며, 이로부터 도출한 individuating hierarchy (개별화 계층)는 언어학 및 인지 과학에서 제안된 계층과 일치합니다.



### Local Transcription Models in Home Care Nursing in Switzerland: an Interdisciplinary Case Study (https://arxiv.org/abs/2409.18819)
- **What's New**: 최근 자연어 처리(NLP) 분야에서의 발전은 의료 분야를 포함한 다양한 도메인에서 새로운 사용 사례를 가능하게 하고 있습니다. 특히, 전사는 간호 문서화 과정의 자동화를 지원하여 간호사들이 환자와의 상호작용에 더 많은 시간을 할애할 수 있도록 돕습니다.

- **Technical Details**: 이 case study에서는 스위스의 홈 케어 간호 문서화 사례를 조사하였습니다. 우리는 다양한 전사 도구와 모델을 평가하고, OpenAI Whisper를 사용하여 여러 가지 독일어 변형(예: 방언, 외국어 억양) 및 홈 케어 간호 분야 전문가가 수동으로 정리한 예제 텍스트로 실험을 진행했습니다.

- **Performance Highlights**: 우리가 사용한 기본 모델조차도 향후 연구를 위한 좋은 출발점이 될 만큼 충분한 성능을 발휘했습니다.



### LLMs4Synthesis: Leveraging Large Language Models for Scientific Synthesis (https://arxiv.org/abs/2409.18812)
Comments:
          12 pages, 3 figures, Accepted to JCDL 2024 Research Track

- **What's New**: 이 논문은 과학 문헌의 복잡성과 양의 증가에 대응하기 위해 LLMs4Synthesis 프레임워크를 소개합니다. 이 프레임워크는 Large Language Models (LLMs)의 능력을 향상시키기 위해 설계되었습니다.

- **Technical Details**: LLMs4Synthesis 프레임워크는 개방형(open-source) 및 독점적(propriety) LLM을 활용하여 과학적 통합을 신속하고 일관되며 맥락이 풍부하게 수행하는 데 초점을 맞춥니다. 새로운 문서 처리 방법론을 개발하고, 새로운 통합 종류를 정의하며, 통합 평가를 위한 아홉 가지 품질 기준을 설정합니다.

- **Performance Highlights**: LLMs의 통합 강화를 위한 강화 학습(reinforcement learning) 및 AI 피드백을 제안하여 통합 품질을 최적화하고, 정립된 기준에 부합하도록 보장합니다. LLMs4Synthesis 프레임워크와 그 구성 요소는 과학 연구의 통합 생성 및 평가 프로세스를 향상시킬 것으로 기대됩니다.



### A Survey on the Honesty of Large Language Models (https://arxiv.org/abs/2409.18786)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 정직성(Honesty) 문제를 다루고 있습니다. 현재의 LLMs가 보이는 정직하지 않은 행동을 분석하고, 이와 관련된 다양한 정의와 평가 방법을 제시합니다.

- **Technical Details**: 정직성에 대한 정의가 다양하고, LLMs의 알고 있는 것과 모르는 것을 구분하는 데 어려움이 있으며, 이에 대한 종합적인 이해가 부족한 상황에서, 이 논문은 LLMs의 정직성을 평가하기 위해 여러 접근 방식을 탐색합니다.

- **Performance Highlights**: 논문은 LLMs의 정직성을 향상시키는 전략과 미래 연구 방향에 대한 통찰을 제공하여, 이 중요한 분야에서의 추가 탐색을 촉진하고자 합니다.



### Read Over the Lines: Attacking LLMs and Toxicity Detection Systems with ASCII Art to Mask Profanity (https://arxiv.org/abs/2409.18708)
- **What's New**: 본 논문에서는 언어 모델이 ASCII 아트를 해석하지 못하는 것을 이용한 새로운 형태의 adversarial attacks를 소개합니다. 이러한 공격을 평가하기 위한 ToxASCII benchmark를 제안합니다.

- **Technical Details**: 연구에서는 두 개의 맞춤형 ASCII 아트 폰트를 개발하였으며, 하나는 special tokens를 활용하고 다른 하나는 텍스트로 채워진 문자 형태를 사용합니다. 이 공격들은 OpenAI의 o1-preview 및 LLaMA 3.1 포함 총 10개의 모델에서 1.0의 공격 성공률(Attack Success Rate)을 기록하였습니다.

- **Performance Highlights**: 이 논문은 연구 목적으로 사용된 유해한 언어의 예를 포함하고 있기 때문에 주의가 필요합니다.



### "Why" Has the Least Side Effect on Model Editing (https://arxiv.org/abs/2409.18679)
- **What's New**: 대형 언어 모델(LLM) 훈련 시의 비용 문제와 최신 연구 조명. 모델 편집(model editing)의 중요성 부각.

- **Technical Details**: 이 논문에서는 모델 편집 질문을 유형별로 분류하여 성능 저하의 정도가 질문 유형에 따라 어떻게 달라지는지를 조사하였습니다. 또한, 작은 모델의 통찰력이 큰 모델에 적용될 수 있는지를 평가하였습니다.

- **Performance Highlights**: 성능 저하의 차이와 배치 크기(batch size)가 성능 유지에 미치는 영향 발견. 배치 크기를 늘리는 것이 성능 저하를 완화시킬 수 있다는 결과 도출.



### Rehearsing Answers to Probable Questions with Perspective-Taking (https://arxiv.org/abs/2409.18678)
- **What's New**: 이 논문은 전문적인 구술 발표 중에 발생할 수 있는 질문에 대한 준비에 초점을 맞춘 새로운 QA(Question Answering) 연구를 제시합니다. 기존의 NLP(자연어 처리)가 독해력 및 일반 상식 QA에 중점을 두었던 것과는 다른 방향입니다.

- **Technical Details**: 연구에서는 기업 관리자와 전문 분석가 간의 실제 QA 대화 기록을 사용하여 세 가지 인과 지식 그래프(causal knowledge graphs, KGs)와 세 가지 대형 언어 모델(large language models, LLMs)을 활용하여 제안된 작업을 탐구합니다. 이는 LLMs의 전문적인 QA 시나리오에서의 응용 가능성에 대한 기초적인 통찰을 제공합니다.

- **Performance Highlights**: 이 연구는 응답을 효과적으로 생성하는 데 있어 인과 KGs와 관점 차용(perspective-taking)의 중요성을 강조하며, 전문적인 QA 환경에서 LLMs의 활용 가능성을 탐색합니다.



### Co-Trained Retriever-Generator Framework for Question Generation in Earnings Calls (https://arxiv.org/abs/2409.18677)
- **What's New**: 이 논문은 earnings call (실적 발표) 컨텍스트를 위한 multi-question generation (MQG) 작업을 최조로 제안하고 있으며, 이를 통해 전통적인 방법의 한계를 극복하려고 합니다.

- **Technical Details**: 이 연구에서는 earnings call 전사( transcripts) 수집과 잠재적인 질문을 분류하기 위한 혁신적인 주석(annotation) 기법을 사용합니다. 추가로, 관련 정보를 추출하기 위한 retriever-enhanced 전략을 도입하였습니다.

- **Performance Highlights**: 경험적 평가 결과, 제안된 방법이 생성한 질문의 정확성(accuracy), 일관성(consistency), 및 perplexity에서 두드러진 우수성을 보였습니다.



### HiCuLR: Hierarchical Curriculum Learning for Rhetorical Role Labeling of Legal Documents (https://arxiv.org/abs/2409.18647)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 이번 연구에서는 법률 문서의 수사적 역할 레이블링(Rhetorical Role Labeling, RRL)을 위한 계층적 커리큘럼 학습 프레임워크인 HiCuLR을 제안합니다. 이는 RRL을 수행하는 데 있어 문서의 난이도에 따라 모델을 점차적으로 학습시킬 수 있도록 설계되었습니다.

- **Technical Details**: HiCuLR은 외부에서 수사적 역할 수준 커리큘럼(Rhetorical Role-level Curriculum, RC)과 내부에서 문서 수준 커리큘럼(Document-level Curriculum, DC)으로 구성된 두 개의 커리큘럼을 포함합니다. DC는 문서를 난이도에 따라 분류하고, RC는 모델이 수사적 역할의 구분을 점진적으로 강화하도록 합니다.

- **Performance Highlights**: 네 가지 RRL 데이터셋에 대한 실험 결과, HiCuLR의 효과성을 입증하였으며, DC와 RC의 보완적인 특성이 두드러졌습니다.



### The Craft of Selective Prediction: Towards Reliable Case Outcome Classification -- An Empirical Study on European Court of Human Rights Cases (https://arxiv.org/abs/2409.18645)
Comments:
          Accepted to EMNLP Findings

- **What's New**: 이 논문은 법률 자연어 처리(NLP) 분야에서 케이스 결과 분류(Case Outcome Classification, COC) 모델의 신뢰성을 높이기 위한 선택적 예측(selective prediction) 프레임워크 내에서 다양한 디자인 선택이 미치는 영향을 실증적으로 조사합니다.

- **Technical Details**: 연구에서는 사전 학습 데이터(corpus), 신뢰도 추정기(confidence estimator), 미세 조정 손실(fine-tuning loss)과 같은 설계 선택이 COC 모델의 신뢰성에 미치는 영향을 분석하였습니다. 특히, 다양한 도메인 전문 사전 학습 데이터의 중요성을 강조하였으며, 더 큰 모델이 과도한 신뢰(overconfidence)를 보이는 경향이 있다는 점도 지적합니다.

- **Performance Highlights**: 실험 결과, 몬테 카를로 드롭아웃(Monte Carlo dropout) 방법이 신뢰도 추정에서 효과적이며, 신뢰도 기반의 오류 정규화(confident error regularization)가 과신(overconfidence)을 완화하는 데 기여함을 보여주었습니다. 이 연구는 법률 NLP에서 선택적 예측에 대한 최초의 체계적인 탐색으로, 법률 분야에서 모델의 신뢰성을 높이기 위한 연구의 필요성을 강조합니다.



### Incorporating Precedents for Legal Judgement Prediction on European Court of Human Rights Cases (https://arxiv.org/abs/2409.18644)
Comments:
          Accepted to EMNLP Findings

- **What's New**: 본 논문에서는 stare decisis 법 이론에 착안하여 법원 판례를 효과적으로 통합하는 방법을 제안합니다.

- **Technical Details**: 법률 문서 처리(LJP) 모델과 함께 판례 검색(retriever) 모델을 결합하기 위해 정밀한 관련성 신호를 바탕으로 훈련합니다. 판례 통합 전략으로는 케이스 간 유사성에 기반한 label interpolation을 통한 직접 통합과 stacked-cross attention 모델을 이용한 판례 융합 모듈을 통한 훈련 중 통합 방법을 제안합니다. 이 과정에서 retriever와 LJP 모델의 공동 훈련(joint training)을 통해 잠재 공간(divergence) 문제를 해결합니다.

- **Performance Highlights**: ECHR 사법 관할권의 LJP 작업에서 실험한 결과, 훈련 중 판례 통합과 retriever 및 LJP 모델의 공동 훈련이 없는 모델이나 판례를 단지 추론(inference) 단계에서 통합한 모델보다 성능이 뛰어난 것으로 나타났습니다. 특히, 희소한 기사(sparser articles)에 보다 큰 이점을 보였습니다.



### Model-based Preference Optimization in Abstractive Summarization without Human Feedback (https://arxiv.org/abs/2409.18618)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 본 연구에서는 인간 피드백 없이 모델의 요약 능력을 향상시키기 위한 새로운 접근 방식인 Model-based Preference Optimization (MPO)을 소개합니다.

- **Technical Details**: MPO는 다양한 decoding strategies를 활용하여 모델이 생성한 preference dataset을 기반으로 LLMs를 미세 조정합니다. 기존의 Direct Preference Optimization (DPO) 방식과는 달리, MPO는 비싼 인간 피드백에 의존하지 않습니다.

- **Performance Highlights**: MPO를 적용한 결과, 표준 요약 데이터셋에서 생성된 요약의 품질이 크게 향상되었습니다. 다양한 평가 지표에서 성능 개선이 나타났습니다.



### Do LLMs suffer from Multi-Party Hangover? A Diagnostic Approach to Addressee Recognition and Response Selection in Conversations (https://arxiv.org/abs/2409.18602)
Comments:
          Accepted to EMNLP 2024 main conference

- **What's New**: 본 연구는 Multi-Party Conversations (MPC)의 분류 성능을 평가하기 위한 새로운 방법론을 제안합니다. 기계 학습 모델의 성능을 특정 구조적 특성에 맞추어 조사함으로써, 전통적인 평가 방법이 간과해 온 구조적 복잡성에 따른 모델 행동의 변화를 검토합니다.

- **Technical Details**: 연구는 Response Selection과 Addressee Recognition 작업을 중심으로 진행하며, 온라인 MPC의 대규모 오픈 코퍼스에서 고정된 사용자 수와 다양한 구조적 특성을 가진 진단 서브데이터셋을 추출합니다. 또한, 개인 정보 보호를 고려하여 원본 사용자명을 사용하지 않고, 원본 텍스트 메시지 대신 대안을 제시합니다.

- **Performance Highlights**: 결과에 따르면 Response Selection은 대화 내용의 텍스트에 더 의존하는 반면, Addressee Recognition은 대화의 구조적 차원을 포착해야 함을 보여줍니다. 대규모 언어 모델(LLM)을 zero-shot 설정에서 사용하여 프롬프트 변형에 대한 민감도가 작업 의존적이라는 점을 강조합니다.



### Hit the Sweet Spot! Span-Level Ensemble for Large Language Models (https://arxiv.org/abs/2409.18583)
- **What's New**: 이 논문에서는 다양한 대규모 언어 모델(LLMs)을 조합하여 서로의 장점을 극대화하는 새로운 접근법인 SweetSpan을 제안합니다. SweetSpan은 기존의 샘플 레벨(sample-level) 및 토큰 레벨(token-level) 앙상블 방법의 한계를 극복합니다.

- **Technical Details**: SweetSpan은 후보 모델들이 공유된 접두사(prefix)를 기반으로 후보 span을 독립적으로 생성합니다. 이후에는 퍼플렉시티(perplexity) 점수를 계산하여 후보 모델 간의 상호 평가를 통해 신뢰할 수 없는 점수를 필터링하고 강력한 span 선택을 이끌어냅니다. 이를 통해 실시간 조정과 정확한 앙상블 결정을 위한 정보 간의 균형을 잘 맞춥니다.

- **Performance Highlights**: 기존의 앙상블 방법들과 비교할 때, 표준 설정 및 성능 차이가 큰 모델을 포함한 도전적 설정에서 실험 결과 SweetSpan의 효과성과 강력함, 다재다능함이 입증되었습니다.



### Research on Predicting Public Opinion Event Heat Levels Based on Large Language Models (https://arxiv.org/abs/2409.18548)
Comments:
          conference

- **What's New**: 최근 몇 년간 큰 언어 모델(large language models)의 급속한 발전으로 인해, GPT-4o와 같은 여러 모델이 언어 작업에서 인간의 성능을 초월하는 탁월한 능력을 보여주었습니다. 이 연구는 공공 여론 분석 분야에서의 잠재적 응용을 탐구합니다.

- **Technical Details**: 이 연구에서는 2022년 7월부터 2023년 12월 사이에 수집된 62,836개의 중국 핫 이벤트 데이터를 전처리하고 분류했습니다. 각 이벤트의 온라인 확산 열 지수를 기반으로 MiniBatchKMeans 알고리즘을 사용하여 이벤트를 자동으로 클러스터링하고 네 가지 열 수준으로 분류했습니다. 이후 각 열 수준에서 250개의 이벤트를 랜덤으로 선택하여 총 1,000개의 평가 데이터셋을 구축했습니다.

- **Performance Highlights**: 평가 과정에서 다양한 큰 언어 모델을 사용하여 두 가지 시나리오(참조 사례 없는 경우와 유사한 사례 참조가 있는 경우)에서 이벤트 열 수준 예측의 정확성을 평가했습니다. 결과적으로 GPT-4o와 DeepseekV2가 후자의 경우 최고의 성능을 보이며 각각 41.4%와 41.5%의 예측 정확도를 기록했습니다. 특히 저온 이벤트(Level 1)의 경우 두 모델의 예측 정확도는 각각 73.6%와 70.4%에 달했습니다. 전체적인 예측 정확도는 열 수준이 높아질수록 감소하는 경향을 보였습니다.



### A Survey on Complex Tasks for Goal-Directed Interactive Agents (https://arxiv.org/abs/2409.18538)
- **What's New**: 최근 대규모 언어 모델(large language models, LLMs)의 발전으로 목표 지향적 상호 작용 에이전트(goal-directed interactive agents)가 다양하고 도전적인 작업을 통해 인간의 일상 생활을 지원할 수 있는 가능성이 열렸습니다. 본 서베이는 이러한 에이전트 평가에 필요한 관련 작업과 환경을 정리하였습니다.

- **Technical Details**: 연구에서는 목표 지향적 상호 작용 에이전트를 평가하기 위해 다양한 과제(task)와 환경(environment)을 수집하고, 현재 에이전트가 직면하는 장애물(obstacles)을 이해하는 데 도움이 되는 차원으로 구조화하였습니다. 이를 통해 에이전트 성능(performance)을 잘 맥락화(contextualize)할 수 있습니다.

- **Performance Highlights**: 본 연구는 Agent의 평가를 위한 최신 자원(resource)을 집약하여, 향후 연구에서 참조할 수 있는 유용한 자료를 제공합니다.



### Do We Need Domain-Specific Embedding Models? An Empirical Investigation (https://arxiv.org/abs/2409.18511)
Comments:
this https URL

- **What's New**: 이 논문에서는 금융 분야에 특화된 Embedding 모델의 필요성을 살펴보며, 새로운 금융 Massive Text Embedding Benchmark (FinMTEB)를 도입하였습니다.

- **Technical Details**: FinMTEB는 금융 분야에 특화된 데이터셋으로 구성되어 있으며, 최신 Embedding 모델 7개의 성능을 평가하였습니다. 데이터셋의 복잡도를 측정하고 분석하여 FinMTEB에서의 성능 저하가 모델의 한계를 나타내는지 검증했습니다.

- **Performance Highlights**: 일반 목적의 MTEB와 비교했을 때, FinMTEB에서 최신 모델들의 성능이 현저히 저하된 것을 관찰하였으며, 이는 금융 분야에 특화된 언어적 및 의미적 패턴을 포착하는 데 어려움을 겪고 있음을 보여줍니다.



### Evaluation of OpenAI o1: Opportunities and Challenges of AGI (https://arxiv.org/abs/2409.18486)
- **What's New**: OpenAI의 o1-preview 대형 언어 모델은 다양한 복잡한 Reasoning 작업에서 인간 수준의 성과를 자주 보여주며, 고급 프로그래밍 문제, 생물학적 추론, 언어 처리 등의 여러 분야에서 두각을 나타냈습니다.

- **Technical Details**: o1-preview는 컴퓨터 과학, 수학, 자연 과학, 의학, 언어학 및 사회 과학 등을 포함한  다수의 분야에서 83.3%의 성공률을 기록하며 복잡한 프로그래밍 문제를 해결했습니다. 특히, 고등학교 수준의 수학 문제에서 100%의 정확도로 단계별 솔루션을 제공하는 등 우수한 성능을 보였습니다. 의료 분야에서는 자연어 추론(Natural Language Inference) 능력이 뛰어나고, EDA 스크립트를 생성하는 등 반도체 설계 과제에서 전문 모델을 초월하는 성과를 얻었습니다.

- **Performance Highlights**: 이 모델은 인류학 및 지질학에서 깊이 있는 이해와 Reasoning 능력을 보여주었으며, 정량적 투자에도 강력한 능력을 가지고 있습니다. 소셜 미디어 분석에서도 효과적인 성과를 거두었고, 정교한 Reasoning과 지식 통합이 필요한 작업에서 특히 뛰어난 결과를 기록했습니다. 그러나 일부 간단한 문제에서 오류가 발생할 수 있고 특정 전문 개념에 대한 도전이 관찰되는 등 일부 제한 사항도 있었습니다.



### URIEL+: Enhancing Linguistic Inclusion and Usability in a Typological and Multilingual Knowledge Bas (https://arxiv.org/abs/2409.18472)
- **What's New**: URIEL+는 URIEL의 한계를 극복하여 2898개의 언어에 대한 typological (유형론적) 특성 범위를 확장하고 사용자 경험을 개선합니다.

- **Technical Details**: URIEL은 7970개 언어에 대한 지리적, 계통적 (phylogenetic), 유형론적 벡터 표현을 제공하며, 4005개 언어의 벡터 간 거리 측정을 포함합니다. URIEL+는 강력하고 사용자 맞춤형 거리 계산을 제공하여 사용자의 요구에 더 잘 맞추도록 개선되었습니다.

- **Performance Highlights**: 업그레이드된 URIEL+는 하위 작업 (downstream tasks)에서 경쟁력 있는 성능을 제공하며, 언어적 거리 연구에 더 잘 맞는 거리를 제공합니다.



### Leveraging Long-Context Large Language Models for Multi-Document Understanding and Summarization in Enterprise Applications (https://arxiv.org/abs/2409.18454)
- **What's New**: 이 논문은 다양한 분야에서 비구조화 데이터의 급증에 따른 다중 문서 이해 및 요약의 중요성을 강조합니다. 전통적인 접근 방식이 정보의 맥락을 잘 잡지 못하고 논리적 일관성을 유지하지 못하는 문제를 다루며, Long-context Large Language Models (LLMs)의 사용을 탐구합니다.

- **Technical Details**: 본 연구는 다중 문서 요약을 효과적으로 수행하기 위한 Long-context LLM의 워크플로우를 설명하며, 법률, 인사(HR), 재무, 소싱과 같은 기업 기능, 의료 및 뉴스 도메인에서의 사례 연구를 다룹니다. 이러한 사례 연구는 효율성과 정확성 모두에서 향상을 보여줍니다.

- **Performance Highlights**: 논문은 데이터셋의 다양성, 모델 확장성 및 편향 완화, 사실 정확성과 같은 윤리적 고려사항과 같은 기술적 장애물에 대한 철저한 분석을 포함하고 있으며, LLM의 기능과 응용 프로그램을 증강하기 위한 미래 연구 방향을 제시합니다.



### Exploring Language Model Generalization in Low-Resource Extractive QA (https://arxiv.org/abs/2409.18446)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)이 폐쇄 도메인(closed-domain)에서 특정 지식이 필요한 질문-응답(extractive question answering, EQA) 작업에 대해 제로샷(zero-shot) 방식으로 잘 일반화할 수 있는지를 조사합니다.

- **Technical Details**: 우리는 여러 실험을 통해 LLM의 성능 차이를 설명합니다. 주요 발견 사항으로는: a) LLM은 폐쇄 도메인에서 긴 응답 범위를 검색하는 데 어려움을 겪습니다; b) 특정 LLM은 전반적으로 강력한 성능을 보이지만, 도메인 특정 단어의 의미 구분 같은 기본 요구 사항을 충족하는 데 약점을 보입니다; c) 모델 파라미터를 확장하는 것이 항상 교차 도메인 일반화(cross-domain generalization)에 효과적이지는 않습니다; d) 폐쇄 도메인 데이터셋은 개방형(open-domain) EQA 데이터셋과 quantitatively (정량적으로) 크게 다릅니다.

- **Performance Highlights**: LLM은 폐쇄 도메인에서 발생하는 데이터셋의 요구에 맞추는 데 어려움을 겪고 있으며, 이는 LLM의 개선 방향을 제시하는 중요한 요소입니다.



### Improving Multilingual ASR in the Wild Using Simple N-best Re-ranking (https://arxiv.org/abs/2409.18428)
- **What's New**: 이 논문에서는 다국어 자동 음성 인식(Multilingual Automatic Speech Recognition, ASR) 모델의 평가 방법을 개선하기 위한 간단하면서도 효과적인 N-best 재정렬(N-best re-ranking) 접근법을 소개합니다.

- **Technical Details**: 본 연구는 언어 모델(language models)과 텍스트 기반 언어 식별 모델(text-based language identification models)과 같은 외부 특성을 활용하여 여러 저명한 음향 모델(acoustic models)의 다국어 ASR 정확도를 향상시키는 방법을 제시합니다.

- **Performance Highlights**: FLEURS 벤치마크에서 MMS 및 Whisper 모델을 사용한 결과, 음성 언어 식별(spoken language identification) 정확도가 각각 8.7% 및 6.1% 향상되었으며, 단어 오류율(word error rates)은 각각 3.3% 및 2.0% 낮아졌습니다.



### SciDFM: A Large Language Model with Mixture-of-Experts for Scienc (https://arxiv.org/abs/2409.18412)
Comments:
          12 pages, 1 figure, 9 tables. Technical Report, Under Review

- **What's New**: 최근 대형 언어 모델(LLMs)을 활용하여 과학적 발견을 도우려는 관심이 급증하고 있습니다. 그러나 대부분의 LLM은 일반 과학에만 초점을 맞추고 있을 뿐, 화학 분자와 아미노산 서열과 같은 분야별 지식이 부족합니다. 이를 해결하기 위해 SciDFM이라는 전문가 혼합 모델을 도입하였으며, 이는 처음부터 훈련되어 대학 수준의 과학적 추론을 수행하고 분자 및 아미노산 서열을 이해할 수 있습니다.

- **Technical Details**: SciDFM은 대규모 훈련 데이터 집합을 수집하여 여러 과학 분야의 논문과 서적, 그리고 분야별 데이터베이스에서 수집한 데이터를 포함합니다. 또한, 사전 학습된 모델을 많은 지시 데이터로 추가 세부 조정하여 하위 기초 평가에서의 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과 SciDFM은 SciEval 및 SciQ와 같은 일반 과학 기초 평가에서 강력한 성능을 보이며, 동등한 크기 모델 중에서 분야별 평가에서도 SOTA(State Of The Art) 성능을 달성하였습니다. 우리는 전문가 선택 결과가 다른 분야의 데이터에 따라 달라진다는 점을 분석하였습니다. 또한 더 넓은 연구 커뮤니티를 위해 SciDFM을 오픈소스했습니다.



### MultiClimate: Multimodal Stance Detection on Climate Change Videos (https://arxiv.org/abs/2409.18346)
Comments:
          5 pages, 1 figure

- **What's New**: 이번 연구에서는 기후 변화(Climate Change, CC)와 관련된 주장을 탐지하기 위한 첫 번째 공개 소스 수동 주석 데이터셋인 MultiClimate를 소개합니다. 이 데이터셋은 100개의 CC 관련 YouTube 비디오와 4,209개의 프레임-전사 쌍으로 구성되어 있습니다.

- **Technical Details**: MultiClimate는 다양한 비전(Vision) 및 언어(Language) 모델, 그리고 멀티모달(Multimodal) 모델을 사용하여 주장을 탐지합니다. 연구 결과, 텍스트 전용 BERT 모델이 이미지 전용 ResNet50 및 ViT 모델보다 현저히 우수한 성능을 보였습니다. 두 가지 모달리티를 결합할 경우, 정확도(Accuracy) 및 F1 점수에서 각각 0.747 및 0.749를 기록하며 최신 기술 수준(State-of-the-art)을 달성했습니다.

- **Performance Highlights**: 100M 크기의 융합(fusion) 모델이 CLIP, BLIP, 그리고 훨씬 더 큰 9B 크기의 멀티모달 IDEFICS 및 텍스트 전용 Llama3와 Gemma2보다 뛰어난 성능을 보였습니다. 이는 대형 언어 모델에 대해 멀티모달 주장 탐지가 여전히 도전적임을 나타냅니다.



### A Generalized LLM-Augmented BIM Framework: Application to a Speech-to-BIM system (https://arxiv.org/abs/2409.18345)
- **What's New**: 이 논문은 건물 정보 모델링(BIM) 작업을 가속화하기 위해 LLM(대형 언어 모델)을 활용한 새로운 프레임워크를 제안합니다. 이는 전통적인 그래픽 사용자 인터페이스를 대체할 가능성을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 6단계로 구성됩니다: 해석(interpret) - 채우기(fill) - 일치(match) - 구조화(structure) - 실행(execute) - 검토(check). 이 과정은 텍스트에서 BIM 또는 음성을 BIM으로 변환하는 방식을 포함합니다.

- **Performance Highlights**: NADIA-S라는 음성 기반 BIM 응용 프로그램을 구현하여 제안된 프레임워크의 적합성을 입증하였고, 외부 벽 세부 사항을 예로 들었습니다.



### AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models (https://arxiv.org/abs/2409.18339)
Comments:
          5 pages, 4 figures

- **What's New**: 본 연구는 LLMs(대규모 언어 모델)의 감정 인식 능력에 대한 새로운 접근을 제공합니다. 특히, 단일 감정 레이블에 국한되지 않고 모호한 감정을 인식하는 능력을 탐구하여 감정 지능(emotional intelligence)의 중요한 측면을 다루고 있습니다.

- **Technical Details**: 연구에서는 LLMs의 제로샷(zero-shot) 및 몇샷(few-shot) 프롬팅(prompts) 기술을 사용하여 문맥 정보(context information)를 과거 대화와 함께 통합함으로써 모호한 감정을 인식하는 과정이 설계되었습니다. 이를 통해 LLMs의 강력한 일반화 능력과 인-컨텍스트 학습(in-context learning)을 활용합니다.

- **Performance Highlights**: 실험 결과, LLMs는 모호한 감정을 인식하는 데 있어 상당한 잠재력을 보여주었으며, 문맥 정보를 포함할 때 그 효과성이 크게 증가함을 발견했습니다. 또한, 덜 모호한 감정을 인식하는 데 높은 효율성을 보이며, 더 모호한 감정들을 인식할 가능성도 제시하였습니다.



### DisGeM: Distractor Generation for Multiple Choice Questions with Span Masking (https://arxiv.org/abs/2409.18263)
- **What's New**: 최근 자연어 처리(Natural Language Processing, NLP) 분야의 발전이 여러 하위 분야에 영향을 미쳤습니다. 이 논문은 다중 선택 질문(Multiple Choice Questions, MCQ)을 위한 방해 요소(distractor) 생성에서의 새로운 접근 방식을 제시합니다.

- **Technical Details**: 본 연구는 사전 훈련된 언어 모델(Pre-trained Language Models, PLMs)을 활용한 간단하고 일반적인 방해 요소 생성 프레임워크를 제공합니다. 기존 방법과 달리, 제안된 프레임워크는 특정 데이터셋에 대한 추가 훈련이 필요하지 않습니다. 두 단계의 프레임워크, 즉 후보 생성(candidate generation)과 후보 선택(candidate selection)으로 구성됩니다.

- **Performance Highlights**: 제안한 방해 요소 생성 프레임워크는 훈련이나 파인튜닝(fine-tuning) 없이도 이전 방법들을 초월하는 성과를 보였으며, 인간 평가에서 더 효과적이고 흥미로운 방해 요소를 생성함을 확인했습니다.



### LangSAMP: Language-Script Aware Multilingual Pretraining (https://arxiv.org/abs/2409.18199)
Comments:
          preprint

- **What's New**: 최근 다국어 사전학습 언어 모델(mPLMs)에서 언어 임베딩(embeddings)의 사용을 피하고 있습니다. 이 논문은 새로운 접근법인 LangSAMP(Language-Script Aware Multilingual Pretraining)를 제안하며, 이를 통해 언어 및 스크립트 정보를 포함하는 임베딩을 활용하여 모델의 표현 학습을 개선합니다.

- **Technical Details**: LangSAMP는 transformer 블록의 출력에 언어 및 스크립트 임베딩을 통합하며, 최종 표현을 언어 모델링 헤드로 전달하기 전에 이러한 정보를 합칩니다. 이는 다국어 코퍼스(500개 이상의 언어 포함)에 대한 XLM-R의 지속적(pretraining) 학습에 적용됩니다.

- **Performance Highlights**: LangSAMP를 적용한 모델은 기준 모델(baseline)을 일관되게 초월하는 성과를 보여줍니다. 추가 분석에서 언어 및 스크립트 임베딩이 언어/스크립트 특화 정보(encoded)는 크로스링구얼 전송(crosslingual transfer)을 위한 소스 언어 선택을 개선함을 보여줍니다.



### LowREm: A Repository of Word Embeddings for 87 Low-Resource Languages Enhanced with Multilingual Graph Knowledg (https://arxiv.org/abs/2409.18193)
Comments:
          Short paper, preview

- **What's New**: 이 논문에서는 87개의 저자원 언어에 대한 정적(Static) 임베딩(Embeddings)의 중앙화된 저장소인 LowREm을 발표합니다.

- **Technical Details**: LowREm은 저자원 언어들을 위한 GloVe 기반 임베딩을 다국어 그래프 지식(Multilingual Graph Knowledge)을 통합하여 개선하는 신규 방법을 제안합니다.

- **Performance Highlights**: 향상된 임베딩은 XLM-R에서 추출된 컨텍스트화된 임베딩과 비교하여 감정 분석(Sentiment Analysis)에서 우수한 성능을 보였습니다.



### Evaluation of Large Language Models for Summarization Tasks in the Medical Domain: A Narrative Review (https://arxiv.org/abs/2409.18170)
- **What's New**: 대형 언어 모델(Large Language Models)의 발전은 임상 자연어 생성(Clinical Natural Language Generation)을 촉진하고, 의료 텍스트의 양을 관리할 수 있는 기회를 창출했습니다. 그러나 의료의 높은 위험 수준은 신뢰할 수 있는 평가를 요구하며, 이는 여전히 도전 과제로 남아 있습니다.

- **Technical Details**: 이 논문은 임상 요약 작업(Clinical Summarization Tasks)의 현재 평가 상태를 분석하고, 전문가의 인간 평가(Human Evaluation) 자원 제약(Resource Constraints)을 해결하기 위한 미래 방향을 제안합니다. 이를 통해 임상 분야에서의 신뢰성을 높이고자 합니다.

- **Performance Highlights**: 이 리뷰는 임상 자연어 생성의 효율성을 높이는 방법을 제안하며, 향후 연구 방향을 명확히 하고, 평가 프로세스의 개선 필요성을 강조합니다.



### Charting the Future: Using Chart Question-Answering for Scalable Evaluation of LLM-Driven Data Visualizations (https://arxiv.org/abs/2409.18764)
- **What's New**: 본 연구는 Visual Question Answering (VQA) 모델을 활용하여 LLM(대형 언어 모델) 생성 데이터 시각화의 평가를 자동화하는 새로운 프레임워크를 제안합니다. 전통적인 평가 방법은 비용이 많이 들고 확장성이 부족한 인간 판단에 의존하거나 오직 데이터 정확도에만 집중하는 경향이 있습니다.

- **Technical Details**: VQA 모델을 사용하여 차트의 데이터 표현 품질 및 일반적인 의사소통 명확성을 평가합니다. 실험은 ChartQA와 PlotQA라는 두 가지 주요 VQA 벤치마크 데이터 세트를 사용하여 OpenAI의 GPT-3.5 Turbo 및 Meta의 Llama 3.1 70B-Instruct 모델로 생성된 시각화를 통해 실시되었습니다.

- **Performance Highlights**: LLM이 생성한 차트는 VQA 성과 기반으로 원래의 비-LLM 생성 차트의 정확도에 미치지 못하며, few-shot prompting이 차트 생성의 정확도를 크게 향상시킬 수 있음을 보여줍니다. 그러나 LLM이 인간이 생성한 그래프의 정확성과 완전히 일치하려면 아직 많은 발전이 필요하다는 점이 강조됩니다.



### Cross-Domain Keyword Extraction with Keyness Patterns (https://arxiv.org/abs/2409.18724)
Comments:
          26 pages, 14 figures

- **What's New**: 이 논문은 주제의존성(domain dependence)과 주석 주관성(annotation subjectivity) 문제를 해결하기 위한 감독된(keyword extraction) 키워드 추출 방식을 제안합니다. 저자는 커뮤니티 수준에서 존재하는 이차적(keyness patterns) 키니스 패턴을 학습하여 키워드를 순위화하는 접근 방식을 소개합니다.

- **Technical Details**: 제안된 접근 방식에서는 독립(feature) 특성(서브랭귀지 도메인 및 용어 길이 포함)과 세 가지 범주의 종속(dependent) 특성(휴리스틱, 특이성, 대표성)을 가진 키니스 패턴을 기반으로 키워드를 평가합니다. 두 개의 합성곱 신경망(convolutional neural network) 모델을 사용하여 키워드 데이터셋에서 키니스 패턴을 학습하며, 부트스트랩 샘플링(bootstrap sampling) 전략으로 주석 주관성을 극복합니다.

- **Performance Highlights**: 이 접근 방식은 일반 감독된 키워드 추출에서 평균 top-10-F-measure 0.316을 기록하며, 총 10개의 키워드 데이터셋에서 최첨단 성능을 달성했습니다. 또한, 훈련 과정에서 제외된 4개의 데이터셋에서 평균 top-10-F-measure 0.346을 기록하며 강력한 크로스 도메인(cross-domain) 성능을 보였습니다.



### KALE-LM: Unleash The Power Of AI For Science Via Knowledge And Logic Enhanced Large Mod (https://arxiv.org/abs/2409.18695)
- **What's New**: AI의 잠재력이 커지고 있는 가운데, 과학 연구를 진전시키기 위해 AI를 활용하는 방안에 대해 논의한 비전 논문입니다.

- **Technical Details**: KALE-LM 모델 시리즈의 일환으로 Llama3-KALE-LM-Chem-8B라는 대형 모델을 제안했으며, 화학 분야와 관련된 작업에서 우수한 성능을 나타냈습니다. 이 모델은 오픈 소스로 공개되었습니다.

- **Performance Highlights**: Llama3-KALE-LM-Chem-8B 모델은 화학 관련 작업에서 뛰어난 성능을 보여주며, 더 지능적인 AI 실현을 위한 강력한 출발점이 될 것으로 기대됩니다.



### Beyond Single-Audio: Advancing Multi-Audio Processing in Audio Large Language Models (https://arxiv.org/abs/2409.18680)
Comments:
          EMNLP24 Findings

- **What's New**: 본 논문에서는 여러 오디오 작업을 동시에 처리할 수 있는 첫 번째 multi-audio evaluation (MAE) 벤치마크를 제안합니다. 이 벤치마크는 11개 멀티 오디오 작업에서 수집한 20개의 데이터셋으로 구성되어 있습니다.

- **Technical Details**: 기존의 audio-LLMs (ALLMs)는 단일 오디오 작업의 평가에 주로 초점을 맞추었으나, 실제 애플리케이션에서는 여러 오디오 스트림을 동시에 처리하는 경우가 많습니다. 우리는 synthetic data를 활용하여 multi-audio-LLM (MALLM)을 제안하며, 이를 통해 여러 유사한 오디오 간의 음향 맥락을 포착합니다.

- **Performance Highlights**: MALLM은 기존의 모든 기준선 모델 대비 뛰어난 성능을 보여주며, 인간 주석 없이도 높은 데이터 효율성을 달성했습니다. 이는 ALLMs가 멀티 오디오 처리 시대에 접어드는 데 중요한 이정표가 될 것입니다.



### ASAG2024: A Combined Benchmark for Short Answer Grading (https://arxiv.org/abs/2409.18596)
Comments:
          Accepted at SIGCSE-Virtual 2024

- **What's New**: 이 연구는 여러 학과, 채점 기준 및 분포에 걸쳐 통합된 단답형 채점 벤치마크인 ASAG2024를 소개합니다. 이는 자동 채점 시스템의 비교를 용이하게 계속 할 수 있게 합니다.

- **Technical Details**: ASAG2024 벤치마크는 일곱 개의 일반적으로 사용되는 단답형 채점 데이터셋을 통합하여 공통 구조 및 채점 기준을 제공합니다. 연구에서는 최근의 단답형 채점 방법들의 성능을 평가하였으며, LLM 기반 접근 방식이 새로운 높은 점수를 기록하지만 여전히 인간의 성능에 미치지 못한다는 것을 보여주었습니다.

- **Performance Highlights**: 최근 SAG 방법들은 이전보다 높은 점수를 기록했지만, 인간 채점자의 성능과 비교할 때 여전히 큰 간극이 존재합니다. 이는 향후 연구에서 인간-기계 SAG 시스템의 가능성을 열어줍니다.



### "Oh LLM, I'm Asking Thee, Please Give Me a Decision Tree": Zero-Shot Decision Tree Induction and Embedding with Large Language Models (https://arxiv.org/abs/2409.18594)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)이 데이터가 제한적인 상황에서 예측 모델링을 위해 사전 지식을 활용하는 강력한 방법을 제공함을 보여줍니다. 특히, LLMs이 훈련 데이터 없이도 본질적으로 해석 가능한 머신 러닝 모델인 의사결정나무(decision trees)를 생성할 수 있음을 설명합니다.

- **Technical Details**: 이 연구에서는 LLMs가 압축된 세계 지식을 활용하여 제로샷(Zero-shot) 의사결정나무를 생성하는 방법을 보여줍니다. 이 의사결정나무는 작고 간단한 테이블 데이터셋(tabular datasets)에 대해 데이터 기반 의사결정나무보다 뛰어난 성능을 보일 수 있습니다. 또한, 이 나무에서 파생된 임베딩(embeddings)은 평균적으로 데이터 기반 의사결정나무에서 파생된 것들과 동등한 성능을 나타냅니다.

- **Performance Highlights**: 지식 기반 의사결정나무 유도와 임베딩 접근 방식은 데이터가 제한된 상황에서 데이터 기반 머신 러닝 방법의 강력한 새로운 기준선을 제공합니다.



### EmoPro: A Prompt Selection Strategy for Emotional Expression in LM-based Speech Synthesis (https://arxiv.org/abs/2409.18512)
- **What's New**: 최근 음성 합성 모델의 발전으로, 방대한 데이터셋을 활용한 제로샷(zero-shot) 능력의 향상이 두드러진다. 이런 발전에도 불구하고, 프롬프트(prompt)의 선택이 음성 생성의 품질에 중대한 영향을 미친다는 점에 주목하였다.

- **Technical Details**: 이 논문에서는 감정 조절이 가능한 음성 합성을 위한 두 단계의 프롬프트 선택 전략인 EmoPro를 제안한다. 이 전략은情緒 표현 강도(emotional expression strength), 음성 품질(speech quality), 텍스트-감정 일관성(text-emotion consistency), 모델 생성 성능(model generation performance)이라는 네 가지 관점에서 프롬프트를 평가하여 선택한다.

- **Performance Highlights**: 실험 결과, 제안된 방법으로 선택된 프롬프트는 기본 참조 모델(baseline)으로 얻은 것보다 더 감정적으로 표현력이 풍부하고 매력적인 합성 음성을 생성하는 것으로 나타났다. 오디오 샘플과 코드가 제공될 예정이다.



### Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization (https://arxiv.org/abs/2409.18433)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: Easy2Hard-Bench라는 새로운 벤치마크 데이터셋의 개발은 어려운 문제에서 쉽게 문제를 풀어내는 일반화 능력을 평가하기 위해 제안되었습니다. 각 문제는 숫자 형태의 난이도 점수로 주석이 달려 있습니다.

- **Technical Details**: 이 데이터셋은 수학, 프로그래밍 문제, 체스 퍼즐, 추리 문제 등 다양한 도메인에서 총 6개의 벤치마크 데이터셋으로 구성되어 있습니다. Item Response Theory (IRT)와 Glicko-2 모델과 같은 난이도 평가 시스템을 활용하여 문제에 대한 숫자 난이도 점수를 일관되게 부여합니다.

- **Performance Highlights**: 여섯 개의 최첨단 LLMs에 대한 광범위한 실험을 통해 다양한 난이도 수준에서의 성능과 일반화 능력을 총체적으로 분석하였으며, 이는 LLM 일반화에 대한 미래 연구에 영감을 줄 것입니다.



### VickreyFeedback: Cost-efficient Data Construction for Reinforcement Learning from Human Feedback (https://arxiv.org/abs/2409.18417)
Comments:
          16 pages, 5 figures

- **What's New**: 이 논문은 Reinforcement Learning from Human Feedback (RLHF)의 비용 효율성에 초점을 맞추고 있습니다. 기존의 preference dataset(선호도 데이터셋)의 경제적 유용성에 대한 고려가 부족했음을 지적하며, 이를 해결하기 위해 새로운 경매 메커니즘을 도입합니다.

- **Technical Details**: RLHF는 대규모 언어 모델(LLM)의 결과에 대한 인간의 선호도를 반영하기 위해 인간의 피드백을 활용합니다. 논문에서는 기존 알고리즘이 복잡한 비전이성(preference)이거나 순환적 관계를 처리하지 못하는 문제를 다룹니다. 경매 메커니즘을 사용하여 선호도 데이터 수집의 효율성을 높이는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 경매 기반 프로토콜은 고품질 피드백에 집중함으로써 LLM의 fine-tuning에서 비용 효율성을 개선하면서도 만족스러운 모델 성능을 유지하는 데 기여한다는 것이 입증되었습니다.



### Defect Prediction with Content-based Features (https://arxiv.org/abs/2409.18365)
- **What's New**: 이 논문에서는 전통적인 결함 예측(Defect Prediction) 접근법과는 다른 소스 코드(content of source code)를 기반으로 한 새로운 접근법을 탐색합니다.

- **Technical Details**: 이 연구에서는 소스 코드의 내용에서 추출한 단어(words), 주제(topics), 데이터 유형(data types), 패키지 이름(package names) 같은 콘텐츠 기반 기능(features)을 사용하여 결함 예측을 수행합니다. 이러한 기능들은 코드 복잡도(metrics that measure complexity)보다 높은 예측력을 가지고 있음을 보였습니다.

- **Performance Highlights**: 연구 결과, 콘텐츠 기반 기능이 코딩 복잡도를 측정하는 기존 방법들보다 예측 성능이 좋으며, 기능 선택(feature selection), 축소(reduction), 조합(combination)을 통해 예측 성능이 더욱 개선된다는 사실이 밝혀졌습니다.



### A Fairness-Driven Method for Learning Human-Compatible Negotiation Strategies (https://arxiv.org/abs/2409.18335)
Comments:
          EMNLP Findings 2024

- **What's New**: 최근 인공지능(AI) 및 자연어 처리(NLP)의 발전에도 불구하고, 협상(negotiation)은 여전히 AI 에이전트에게 어려운 분야입니다. 최근에 제안된 FDHC 프레임워크는 공정성(fairness)을 고려하여 인간과 호환되는 협상 전략을 학습할 수 있게 도와줍니다.

- **Technical Details**: FDHC 프레임워크는 보상 설계(reward design)와 탐색(search) 두 가지 측면에서 공정성을 통합합니다. 또한, LGM-Zero라는 새로운 RL(강화 학습) + search 기법을 도입하여 사전 훈련된 언어 모델(pre-trained language model)을 활용하여 대규모 행동 공간에서 인간과 호환되는 제안을 검색합니다.

- **Performance Highlights**: 실험 결과, FDHC는 협상 결과를 더 평등하게 만들고 협상 품질을 향상시키는 데 성공하였습니다.



### Cross-Institutional Structured Radiology Reporting for Lung Cancer Screening Using a Dynamic Template-Constrained Large Language Mod (https://arxiv.org/abs/2409.18319)
- **What's New**: 본 논문은 구조적 방사선 보고서를 생성하기 위한 향상된 오픈 소스 LLM(대형 언어 모델)을 개발하는 새로운 접근법을 제안합니다. 기존 모델들이 직면한 형식 오류, 콘텐츠 착각(content hallucinations), 개인정보 유출 문제를 해결하고자 합니다.

- **Technical Details**: 연구팀은 두 기관으로부터 수집된 5,442개의 비식별화된 LCS 보고서를 분석하고, 이 중 500개의 보고서를 무작위로 선택하여 수작업으로 라벨링하였습니다. 29개의 특징을 포함한 표준화된 보고서 템플릿을 개발했으며, 이를 기반으로 템플릿 제약 디코딩(template-constrained decoding)을 이용해 LLAMA, Qwen, Mistral 등의 오픈 소스 LLM을 향상시켰습니다. 성능 평가는 F1 점수, 신뢰 구간(confidence interval), McNemar test, z-test를 통해 수행되었습니다.

- **Performance Highlights**: 제안한 방법은 여러 기관의 데이터셋에서 LLM 성능을 일관되게 향상시켰으며, 형식 오류나 콘텐츠 착각이 없었습니다. 오픈 소스 LLAMA-3.1 405B 모델의 성능을 최대 10.42% 개선하였고, GPT-4o보다 17.19% 더 뛰어난 성과를 보였습니다. 또한, 대규모 다중 모달(multimodal) 데이터베이스에서 신규 결절 검색 시스템을 성공적으로 프로토타입하고 자동으로 통계 분석을 수행하여, 이전 결과와의 일관성을 보여주었습니다.



### Realistic Evaluation of Model Merging for Compositional Generalization (https://arxiv.org/abs/2409.18314)
- **What's New**: 본 논문에서는 다양한 merging 방법론의 상대적인 장점을 평가하고, 이를 통해 각 방법의 실제 요구 사항을 명확히 하였습니다. 특히, 이미지 분류(image classification), 이미지 생성(image generation), 자연어 처리(natural language processing) 분야에서의 compositional generalization을 위한 merging에 초점을 맞추었습니다.

- **Technical Details**: 연구는 다양한 merging 방법을 일관된 실험 환경에서 비교하였으며, 모델 아키텍처(model architecture), 데이터 가용성(data availability), 계산 예산(computational budget)에 대한 가정이 서로 다를 수 있음을 강조합니다. 또한, 각 merging 방법이 필요로 하는 계산 비용(computational costs)과 병합되는 모델 수가 증가할 때의 성능을 측정하였습니다.

- **Performance Highlights**: 연구 결과는 모델 병합(model merging) 분야의 현재 상태를 명확히 하고, 새로운 방법을 시험할 수 있는 포괄적이고 엄격한 실험 Setup을 제공합니다.



### Advancing Object Detection in Transportation with Multimodal Large Language Models (MLLMs): A Comprehensive Review and Empirical Testing (https://arxiv.org/abs/2409.18286)
- **What's New**: 이번 연구는 교통 시스템에서 객체 탐지(object detection)에 대한 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)과 대형 비전 모델(Large Vision Models, VLMs)의 응용을 포괄적으로 검토하고 실증적으로 평가하는 것을 목표로 합니다.

- **Technical Details**: 연구의 첫 번째 부분에서는 MLLMs의 교통 응용 분야에서의 잠재적인 이점에 대한 배경을 제공하고, 기존 연구에서 현재 MLLM 기술에 대한 포괄적인 리뷰를 실시했습니다. 두 번째 부분에서는 교통 응용 프로그램에서의 엔드 투 엔드 객체 탐지(taxonomy of end-to-end object detection) 개요와 향후 방향을 제시했습니다.

- **Performance Highlights**: MLLM 성능에 대한 상세한 평가 결과를 제공하며, 도로 안전 특성 추출, 안전 비상 이벤트 탐지, 열화상 이미지의 시각적 추론을 포함한 세 가지 실제 교통 문제에 대한 실증 분석을 수행했습니다. 이 연구는 MLLM의 강점과 개선이 필요한 영역을 밝혀냈으며, 교통에서의 객체 탐지를 향상시키기 위한 MLLM의 실용적인 한계와 도전에 대해 논의합니다.



### MMMT-IF: A Challenging Multimodal Multi-Turn Instruction Following Benchmark (https://arxiv.org/abs/2409.18216)
Comments:
          24 pages, 16 figures

- **What's New**: MMMT-IF 데이터셋과 Programmatic Instruction Following (PIF) 메트릭을 도입하여 멀티모달, 멀티턴 대화에서 지시사항을 따르는 능력을 평가하는 새로운 방법을 제안합니다.

- **Technical Details**: MMMT-IF는 질문들 사이에 추가된 전역 지시사항을 포함하여, 모델이 긴 대화 속에서 분산된 지시사항을 검색하고 지시사항 제약 하에서 추론할 수 있도록 도전합니다. 모든 지시사항은 코드 실행을 통해 객관적으로 검증 가능합니다. PIF 메트릭은 추론 작업을 수행하는 동안 정확하게 따르는 지시사항의 비율을 측정합니다. PIF-N-K 메트릭은 모델 응답 중 K개가 PIF 점수가 1인 비율을 측정하여 강건성을 평가합니다.

- **Performance Highlights**: Gemini 1.5 Pro, GPT-4o, Claude 3.5 Sonnet의 평균 PIF 점수가 턴 1에서 0.81에서 턴 20에서 0.64로 감소하며, 모든 응답을 4회 반복했을 때 GPT-4o와 Gemini가 모든 지시사항을 성공적으로 따르는 경우는 단 11%입니다. 지시사항이 모델 입력 문맥의 끝에 추가되면 평균 PIF 점수가 22.3 포인트 향상됩니다.



### AI Policy Projector: Grounding LLM Policy Design in Iterative Mapmaking (https://arxiv.org/abs/2409.18203)
- **What's New**: 이 논문에서는 정책 설계를 지도 제작(mapmaking)에서 영감을 얻어 새로운 AI 정책 설계 프로세스를 소개합니다.

- **Technical Details**: 이 연구에서 제안하는 Policy Projector는 모델 입력 및 출력 쌍의 지형을 탐색하고 사용자 정의 영역을 정의하며, LLM 출력에 적용할 수 있는 규칙을 탐색할 수 있도록 해줍니다. 예를 들어, '폭력(violence)'과 '그래픽 세부사항(graphic details)'이 포함된 출력이 있는 경우 그래픽 세부사항 없이 다시 쓰는 규칙을 설정할 수 있습니다.

- **Performance Highlights**: 12명의 AI 안전(AI safety) 전문가와의 평가에서, Policy Projector는 정책 설계자가 기존의 포괄적인 해악 분류(harm taxonomy)를 넘어서는 문제적 모델 행동을 해결하는 데 도움을 주었습니다.



### Data-Prep-Kit: getting your data ready for LLM application developmen (https://arxiv.org/abs/2409.18164)
Comments:
          10 pages, 7 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM) 개발을 위한 데이터 준비의 중요성을 강조하며, 사용자들이 손쉽게 확장하고 조정할 수 있는 오픈 소스 데이터 준비 툴킷인 Data Prep Kit (DPK)를 소개합니다.

- **Technical Details**: DPK는 데이터 준비를 사용자의 필요에 맞게 조정할 수 있도록 설계된 아키텍처를 가지고 있으며, 로컬 머신에서 데이터를 준비할 수 있을 뿐만 아니라 수천 개의 CPU 코어가 있는 클러스터에서도 손쉽게 확장할 수 있습니다. DPK는 자연어 및 코드 데이터를 변환하는 확장 가능한 모듈 세트를 제공합니다. 사용자가 추가적인 변환이 필요할 경우, DPK의 지원을 통해 손쉽게 개발할 수 있습니다. 이러한 모듈들은 독립적으로 또는 파이프라인 방식으로 연속적인 작업을 수행하는 데 사용될 수 있습니다.

- **Performance Highlights**: DPK의 성능은 작은 규모에서 시작하여 매우 큰 수의 CPU까지 확장 가능함을 보여줍니다. DPK의 모듈은 Granite 모델 데이터 준비에도 사용되었습니다. DPK는 AI 커뮤니티가 LLM 모델 성능을 향상하거나 Retrieval-Augmented Generation (RAG) 기능으로 모델을 미세 조정할 수 있도록 돕는 귀중한 기여라고 믿습니다.



### Modulated Intervention Preference Optimization (MIPO): Keep the Easy, Refine the Difficu (https://arxiv.org/abs/2409.17545)
Comments:
          8pages, submitted to AAAI 2025

- **What's New**: 이번 연구에서는 Modulated Intervention Preference Optimization (MIPO)라는 새로운 방법론을 제안합니다. MIPO는 주어진 데이터와 참조 모델의 정렬 상태에 따라 개입(intervention) 정도를 조절하여 모델 일치를 최적화합니다.

- **Technical Details**: MIPO는 참조 모델이 잘 정렬된 경우, 정책 모델이 참조 모델과 크게 이탈하지 않도록 개입을 증가시키고, 반대로 정렬이 좋지 않은 경우 개입을 줄여 보다 광범위한 학습을 가능하게 합니다. 이 연구에서는 Mistral-7B와 Llama3-8B 모델을 활용하여 MIPO와 DPO의 성능을 비교합니다.

- **Performance Highlights**: 실험 결과, MIPO는 다양한 평가 시나리오에서 DPO에 비해 일관되게 우수한 성능을 보였습니다.



### M$^2$PT: Multimodal Prompt Tuning for Zero-shot Instruction Learning (https://arxiv.org/abs/2409.15657)
Comments:
          EMNLP 2024

- **What's New**: 이번 연구에서는 Multimodal Prompt Tuning (M$^2$PT)이라는 새로운 접근 방식을 도입하여 대형 다중 모달 언어 모델 (MLLMs)의 효율적인 지침 조정 (instruction tuning)을 지원합니다.

- **Technical Details**: M$^2$PT는 시각 (visual) 및 텍스트 (textual) 프롬프트를 각각 비전 인코더 (vision encoder)와 언어 프로세서 (language processor)에 통합하여 파인튜닝 (finetuning) 동안 다양한 모달리티 간의 특징을 추출하고 일치시킵니다.

- **Performance Highlights**: 다양한 다중 모달 평가 데이터셋에서 우리의 접근 방식은 여러 최신 기법 (state-of-the-art baselines) 대비 우수한 성능을 보여주었으며, 포괄적인 제거 연구 (ablation studies)를 통해 프롬프트 설계와 접근 방식의 효율성을 확인하였습니다.



New uploads on arXiv(cs.IR)

### Cross-Domain Keyword Extraction with Keyness Patterns (https://arxiv.org/abs/2409.18724)
Comments:
          26 pages, 14 figures

- **What's New**: 이 논문은 주제의존성(domain dependence)과 주석 주관성(annotation subjectivity) 문제를 해결하기 위한 감독된(keyword extraction) 키워드 추출 방식을 제안합니다. 저자는 커뮤니티 수준에서 존재하는 이차적(keyness patterns) 키니스 패턴을 학습하여 키워드를 순위화하는 접근 방식을 소개합니다.

- **Technical Details**: 제안된 접근 방식에서는 독립(feature) 특성(서브랭귀지 도메인 및 용어 길이 포함)과 세 가지 범주의 종속(dependent) 특성(휴리스틱, 특이성, 대표성)을 가진 키니스 패턴을 기반으로 키워드를 평가합니다. 두 개의 합성곱 신경망(convolutional neural network) 모델을 사용하여 키워드 데이터셋에서 키니스 패턴을 학습하며, 부트스트랩 샘플링(bootstrap sampling) 전략으로 주석 주관성을 극복합니다.

- **Performance Highlights**: 이 접근 방식은 일반 감독된 키워드 추출에서 평균 top-10-F-measure 0.316을 기록하며, 총 10개의 키워드 데이터셋에서 최첨단 성능을 달성했습니다. 또한, 훈련 과정에서 제외된 4개의 데이터셋에서 평균 top-10-F-measure 0.346을 기록하며 강력한 크로스 도메인(cross-domain) 성능을 보였습니다.



### Scalable Cross-Entropy Loss for Sequential Recommendations with Large Item Catalogs (https://arxiv.org/abs/2409.18721)
Comments:
          11 pages, accepted for RecSys'24

- **What's New**: 본 논문은 기존의 Cross-Entropy (CE) 손실 함수를 대체하는 새로운 Scalable Cross-Entropy (SCE) 손실 함수를 소개합니다. 이 SCE 손실 함수는 큰 항목 목록을 다루는 데이터 세트에서 CE 손실을 근사하여, 추천 품질을 유지하면서도 시간 효율성과 메모리 사용량을 개선합니다.

- **Technical Details**: SCE 손실 함수는 선택적인 GPU 효율적 계산 전략을 활용하여, 카탈ログ에서 가장 유익한 요소에 집중하고 특히 false positives 가능성이 높은 요소에 중점을 둡니다. 이를 위해 최대 내부 곱 검색(maximum inner product search)을 통해 모델 출력의 하위 집합에 대한 softmax 분포를 근사합니다.

- **Performance Highlights**: 실험 결과, SCE는 다른 대안들과 비교하여 최대 메모리 사용량을 100배까지 줄이면서도 성능 지표를 유지하거나 초과하는 효과를 입증했습니다. 이 접근법은 대규모 언어 모델과 같은 다양한 도메인에서의 대규모 개발 가능성도 열어줍니다.



### Less is More: Towards Sustainability-Aware Persuasive Explanations in Recommender Systems (https://arxiv.org/abs/2409.18690)
Comments:
          The paper was accepted for publication and will be presented in the LBR track of RecSys 2024, 14.- 18. October 2024, Bari, Italy

- **What's New**: 이 논문은 추천 시스템에서 '지속 가능성 인식 설득적 설명'(sustainability-aware persuasive explanations)의 개념을 제안하며, 이는 유엔의 지속 가능한 발전 목표(SDGs)를 달성하는 데 기여할 수 있음을 강조합니다.

- **Technical Details**: 지속 가능성 인식 설득적 설명은 사용자 신뢰를 높이고 특정 제품 구매를 유도하며, 추천의 이유를 이해하도록 도와줍니다. 기존의 전자 상거래 플랫폼에서 흔히 보지 못했던 '적을수록 더 많은' 원칙에 중점을 두고 있습니다. 연구는 세 가지 항목 도메인에서 사용자 연구를 기반으로 하여 진행되었습니다.

- **Performance Highlights**: 사용자 수용성 및 지속 가능성 인식 설득적 설명의 잠재적 효과에 대한 연구 결과가 유망하게 나타났습니다.



### Corpus-informed Retrieval Augmented Generation of Clarifying Questions (https://arxiv.org/abs/2409.18575)
- **What's New**: 이번 연구는 웹 검색을 위한 정보에 기반한 clarifying questions를 생성하는 모델을 개발하는 것을 목표로 합니다. 이는 검색된 코퍼스와 일치하는 질문을 생성하는 방식을 보장합니다.

- **Technical Details**: Retrieval Augmented Language Models (RAG)의 효과를 입증하며, 사용자 쿼리와 검색 코퍼스를 공동으로 모델링하여 불확실성을 특정하고 명확한 질문을 만드는 과정을 중심으로 합니다. 또한, 질문의 폭을 넓히기 위해 더 많은 증거 문서를 모델링하는 방법을 제시합니다. 

- **Performance Highlights**: 현재의 데이터셋은 검색 의도가 코퍼스에 대부분 지원되지 않아 질문 생성 모델이 잘못된 의도를 제안하는 'hallucinate' 현상이 발생합니다. 이를 해결하기 위해 ground truth clarifications와 검색 코퍼스를 정렬하는 데이터셋 증강 방법을 제안하며, 증거 풀의 관련성을 높이기 위한 기법을 연구합니다. 그러나 corpus 내에서 ground truth intents를 식별하는 것은 여전히 어려운 과제임을 확인합니다.



### Decomposing the Jaccard Distance and the Jaccard Index in ABCDE (https://arxiv.org/abs/2409.18522)
- **What's New**: 이 논문은 매우 큰 클러스터링(clusterings) 간의 차이를 평가하는 데 사용되는 ABCDE 기법을 소개합니다. JaccardDistance와 JaccardIndex의 새로운 분해(decomposition)를 수행하여 Impact 및 Quality 메트릭(metric)을 도출했습니다.

- **Technical Details**: JaccardDistance는 두 클러스터링 간의 차이의 크기를 측정하는 진정한 거리 메트릭이며, JaccardIndex는 클러스터링 간의 유사성을 나타내는 보완적인 메트릭입니다. 이 두 메트릭의 관계는 JaccardDistance + JaccardIndex = 1로 표현됩니다. 이는 클러스터링 변화의 특성을 이해하는 데 도움을 줍니다.

- **Performance Highlights**: 논문에서 제안한 새로운 메트릭은 수학적으로 잘 정의되어 있으며 간단한 방정식(equations)을 통해 서로 연결되어 있습니다. 이를 통해 클러스터링 변화의 양 및 품질에 대한 통찰력을 제공합니다.



### Efficient Top-k s-Biplexes Search over Large Bipartite Graphs (https://arxiv.org/abs/2409.18473)
- **What's New**: 이 논문에서는 bipartite graph의 새로운 개념인 $s$-biplex를 소개하고, 이들 중 최대 $k$ 개의 가장 큰 $s$-biplex를 찾는 top-$k$ $s$-biplex search (TBS) 문제를 정의합니다. TBS 문제는 NP-hard임을 증명하며, 이를 해결하기 위해 새로운 branching algorithm인 MVBP를 제안합니다.

- **Technical Details**: $s$-biplex의 정의는 각 서브그래프의 정점이 반대 집합의 최대 $s$개의 정점을 제외한 모든 정점과 인접해야 한다는 것입니다. 제안된 MVBP 알고리즘은 단순한 $2^n$ enumeration 알고리즘을 개선하며, FastMVBP의 경우 $O^*(\gamma_s^{d_2})$의 시간 복잡도를 가집니다. 여기서 $\gamma_s<2$이고, $d_2$는 드문 real-world 그래프의 정점 수보다 훨씬 작습니다.

- **Performance Highlights**: FastMVBP 알고리즘은 8개의 실제 및 합성 데이터셋에 대한 광범위한 실험을 통해 기존의 벤치마크 알고리즘보다 최대 세 배 더 빠른 성능을 발휘했습니다.



### Generative Retrieval Meets Multi-Graded Relevanc (https://arxiv.org/abs/2409.18409)
Comments:
          Accepted by the NeurIPS 2024 (Spotlight)

- **What's New**: 이 논문에서는 정보 검색을 위한 새로운 접근법인 Generative retrieval에 대해 소개합니다. 기존의 방법은 이진 관련성 데이터에 한정되어 있었으나, 본 연구는 다중 등급 관련성을 위한 새로운 프레임워크인 GRaded Generative Retrieval (GR$^2$)을 제안합니다.

- **Technical Details**: GR$^2$는 두 가지 주요 구성 요소에 중점을 둡니다. 첫째, 관련성이 높고 구별 가능한 식별자(docids)를 만드는 것입니다. 이를 위해 docid 생성과 autoencoder 모델의 조합을 통해 관련성과 구별성을 동시에 최적화합니다. 둘째, 관련 등급(relevance grades) 간의 관계에 대한 정보를 훈련 과정에 통합하여 다중 등급 제약 대조 훈련(multi-graded constrained contrastive training)을 구현합니다. 이를 통해 쿼리 및 해당 관련 문서의 식별자 표현을 더 가깝게 만듭니다.

- **Performance Highlights**: GR$^2$는 다중 등급과 이진 관련성이 혼합된 데이터셋에서 광범위한 실험을 수행하였으며, 그 결과 GR$^2$의 효과를 입증하였습니다. 이러한 성과는 개별 문서를 효과적으로 표현할 수 있는 식별자 생성을 가능하게 했음을 나타냅니다.



### Evaluation of Cluster Id Assignment Schemes with ABCDE (https://arxiv.org/abs/2409.18254)
- **What's New**: 이 논문은 클러스터링에서 각 클러스터에 고유한 클러스터 ID를 할당하는 새로운 방법을 제안합니다. 특히, 클러스터가 동일한 개념을 나타낼 경우, 역사적인 클러스터와 동일한 ID를 부여하여 의미론적 ID 안정성을 유지하는 것이 목적입니다.

- **Technical Details**: 이 논문에서는 id 할당의 상대적 장점을 평가하기 위해, 역사적인 클러스터링과 이를 기반으로 한 새로운 클러스터링을 비교합니다. 이 과정에서 기본(Baseline) 방식과 실험 방식으로 할당된 ID 간의 차이를 평가하며, 이를 위해 클러스터 ID 할당 문제를 클러스터 멤버십 문제로 변환하여 ABCDE 기법으로 평가합니다. ABCDE는 수십억 개의 항목이 수백만 개의 클러스터에 그룹화되는 현실 세계의 애플리케이션에서 클러스터 멤버십의 차이를 평가하는 정교한 기술입니다.

- **Performance Highlights**: 이 연구 결과, ID 할당의 질과 규모를 특성화하는 메트릭(metrics)을 생성하여, 클러스터 멤버십과 ID의 변화를 동시에 평가할 수 있는 접근 방식을 제시합니다. 다양한 예를 활용하여 이 아이디어를 설명하고 있습니다.



### LML: Language Model Learning a Dataset for Data-Augmented Prediction (https://arxiv.org/abs/2409.18957)
Comments:
          First version

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 분류 작업에 활용하기 위한 새로운 접근 방식을 소개합니다. 전통적인 머신 러닝(ML) 모델들과는 달리, LLMs를 사용하여 데이터 정리(data cleaning)와 특징 공학(feature engineering)을 간소화합니다.

- **Technical Details**: 이 논문은 'Language Model Learning (LML)'이라는 새로운 개념과 'Data-Augmented Prediction (DAP)'이라는 새로운 방법을 제안합니다. LLM이 데이터를 탐색하고 이해하며 분류를 결정하는 방식으로 분류를 수행합니다. DAP 과정에서는 데이터 요약을 사용하여 자동으로 쿼리를 생성하고, 이를 통해 관련 데이터 행을 검색한 후, 최종적인 분류 결과를 생성합니다.

- **Performance Highlights**: 테스트 사례에서 시스템의 정확도가 90%를 초과하여 기존 ML 모델을 다양한 시나리오에서 초월할 가능성을 입증하였습니다. 사용자가 예측의 논리를 검토할 수 있도록 'Explainable Machine Learning Model'로 행위하는 문구를 프롬프트에 포함시킴으로써 예측의 해석 가능성을 향상시켰습니다.



### Suicide Phenotyping from Clinical Notes in Safety-Net Psychiatric Hospital Using Multi-Label Classification with Pre-Trained Language Models (https://arxiv.org/abs/2409.18878)
Comments:
          submitted to AMIA Informatics Summit 2025 as a conference paper

- **What's New**: 이번 연구는 정신과 고위험 환경에서 자살 사건을 정확하게 식별하고 분류하여, 자살 예방 조치를 개선하고 운영 부담을 감소시키며 치료 품질을 향상시키는 방법을 제시합니다.

- **Technical Details**: 해당 연구에서는 두 가지 미세 조정 전략(단일 라벨 다수 및 단일 다중 라벨)을 사용하여 500개의 주석이 달린 정신과 평가 노트를 기반으로 네 가지 BERT(Bidirectional Encoder Representations from Transformers) 모델의 성능을 평가하였습니다. 노트는 자살적 사고(SI), 자살 시도(SA), 자살 노출(ES), 비자살적 자기 상해(NSSI)로 라벨링되었습니다.

- **Performance Highlights**: RoBERTa 모델이 binary relevance(이진 관련성) 방법을 사용하여 다른 모델보다 뛰어난 성능을 발휘하여 accuracy(정확도)가 0.86, F1 score가 0.78로 나타났습니다. MentalBERT는 F1 score가 0.74로 BioClinicalBERT의 0.72를 초과하였으며, 단일 다중 라벨 분류기로 미세 조정된 RoBERTa는 0.88의 정확도와 0.81의 F1 score로 성능이 더욱 향상되었습니다.



### Explainable Enrichment-Driven GrAph Reasoner (EDGAR) for Large Knowledge Graphs with Applications in Drug Repurposing (https://arxiv.org/abs/2409.18659)
Comments:
          10 pages, 5 figures, 4 tables

- **What's New**: 이 논문에서는 실세계 개체 간의 연결 및 관계를 나타내는 지식 그래프(Knowledge Graphs, KGs)를 위한 새로운 링크 예측 프레임워크인 Enrichment-Driven GrAph Reasoner(EDGAR)를 제안합니다. 이 프레임워크는 엔티티 지역 규칙(entity-local rules)을 발굴하여 새로운 간선을 추론합니다.

- **Technical Details**: EDGAR는 차별적으로 발현된 유전자 세트에 공통적인 메커니즘을 식별하는 데 사용되는 통계적 방법인 enrichment 분석을 활용합니다. EDGAR의 추론 결과는 본질적으로 설명 가능하고 순위 매길 수 있으며, 각 enrichment 기반 규칙의 통계적 유의성을 나타내는 p-값을 포함합니다. 이 시스템은 ROBOKOP이라는 대규모 생물의학 KG에서 알츠하이머병(AD) 치료를 위한 약물 재창출(drug repurposing) 사례 연구를 통해 그 효과를 입증합니다.

- **Performance Highlights**: KG에서 14개의 알려진 약물을 추출한 후, enrichment 분석을 통해 20개의 맥락적 바이오마커(contextual biomarkers)를 식별하였습니다. 이를 통해 AD에 대한 공유 약물 효능과 관련된 기능적 경로(functional pathways)를 드러냈고, 상위 1000개의 enrichment 결과를 사용하여 1246개의 추가 약물 후보를 발견했습니다. 상위 10개 후보는 의료 문헌의 증거를 통해 검증되었습니다. EDGAR는 ROBOKOP 내에서 배포되어 웹 사용자 인터페이스를 갖추고 있습니다.



### Do We Need Domain-Specific Embedding Models? An Empirical Investigation (https://arxiv.org/abs/2409.18511)
Comments:
this https URL

- **What's New**: 이 논문에서는 금융 분야에 특화된 Embedding 모델의 필요성을 살펴보며, 새로운 금융 Massive Text Embedding Benchmark (FinMTEB)를 도입하였습니다.

- **Technical Details**: FinMTEB는 금융 분야에 특화된 데이터셋으로 구성되어 있으며, 최신 Embedding 모델 7개의 성능을 평가하였습니다. 데이터셋의 복잡도를 측정하고 분석하여 FinMTEB에서의 성능 저하가 모델의 한계를 나타내는지 검증했습니다.

- **Performance Highlights**: 일반 목적의 MTEB와 비교했을 때, FinMTEB에서 최신 모델들의 성능이 현저히 저하된 것을 관찰하였으며, 이는 금융 분야에 특화된 언어적 및 의미적 패턴을 포착하는 데 어려움을 겪고 있음을 보여줍니다.



### Neural Collaborative Filtering to Detect Anomalies in Human Semantic Trajectories (https://arxiv.org/abs/2409.18427)
Comments:
          Accepted for publication in the 1st ACM SIGSPATIAL International Workshop on Geospatial Anomaly Detection (GeoAnomalies'24)

- **What's New**: 이번 연구에서는 인간의 궤적(anomaly detection in human trajectories) 이상 탐지를 위해 경량화된 모델을 개발하였습니다. 이는 기존의 차량 중심 애플리케이션에 비해 부족했던 인간 수준의 궤적 이상 탐지 연구에 기여하고자 합니다.

- **Technical Details**: 우리는 Neural Collaborative Filtering 접근법을 통해 정상적인 이동 패턴(normal mobility)을 모델링하고 예측하는 방법을 제안합니다. 이 방법은 선험적 지식(prior knowledge) 없이 인간 사용자들의 일상 패턴을 모델링함으로써, 데이터가 희소하거나 불완전한 상황(예: cold start)에서 성능을 향상시킵니다. 알고리즘은 두 개의 주요 모듈로 구성됩니다: 첫 번째는 협업 필터링 모듈(collaborative filtering module)로, 개별 사용자의 정상적인 이동 패턴을 모델링합니다. 두 번째는 신경 모듈(neural module)로, 인간 궤적 데이터에 내재된 복잡한 시공간(spatio-temporal) 관계를 해석합니다.

- **Performance Highlights**: 우리는 시뮬레이션 데이터 및 실제 데이터셋을 사용하여 기존의 최첨단 궤적 이상 탐지 방법들과 비교하고, 우리의 방법이 더 우수한 성능을 보임을 보여주었습니다.



### Tracking Software Security Topics (https://arxiv.org/abs/2409.18351)
- **What's New**: 이번 논문에서는 소프트웨어 보안(Security) 관련 주제를 실시간으로 추적할 수 있도록 돕는 새로운 도구, SOSK를 제안합니다.

- **Technical Details**: SOSK는 사용자가 보안 보고서(Report) 집합을 가져와 이를 전처리(pre-process)하고, 텍스트 설명에서 가장 중요한 키워드(Keywords)를 추출합니다. 키워드의 임베딩 벡터(Embedding vector) 유사성을 기반으로 하여, SOSK는 사용자 제공 키워드 세트에서 키워드 세트를 확장하거나 정제할 수 있습니다.

- **Performance Highlights**: 초기 평가 결과, SOSK는 키워드를 효과적으로 확장하고, 사용자 요청에 맞는 보안 보고서를 성공적으로 검색할 수 있음을 보여주었습니다.



New uploads on arXiv(cs.CV)

### PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation (https://arxiv.org/abs/2409.18964)
Comments:
          Accepted to ECCV 2024. Project page: this https URL

- **What's New**: PhysGen은 단일 이미지를 입력으로 사용하여 현실적이고 물리적으로 그럴듯한 비디오를 생성하는 혁신적인 이미지-비디오 생성 방법입니다. 주목할 점은 이미지에 적용된 힘과 토크와 같은 입력 조건을 활용하여 동적 영상을 생성하는 것입니다.

- **Technical Details**: PhysGen의 핵심 구성 요소는 세 가지로 나눌 수 있습니다: (i) 이미지의 기하학, 재료 및 물리적 매개변수를 효과적으로 캡처하는 이미지 이해 모듈; (ii) 강체 물리학(rigid-body physics)과 유추된 매개변수를 사용하는 이미지 공간 동적 시뮬레이션 모델; (iii) 생성적 영상 확산(generative video diffusion) 기법을 활용해 시뮬레이션된 동작을 포함하는 현실적인 비디오 영상을 생성하는 이미지 기반 렌더링 및 정제 모듈입니다.

- **Performance Highlights**: PhysGen을 통해 생성된 비디오는 물리적 사실성과 외관 측면에서 현실적이며 정교한 제어가 가능합니다. 기존의 데이터 기반 이미지-비디오 생성 연구들과의 정량적 비교 및 포괄적인 사용자 연구를 통해 뛰어난 결과를 보여줍니다.



### Exploring Token Pruning in Vision State Space Models (https://arxiv.org/abs/2409.18962)
Comments:
          NeurIPS'24

- **What's New**: 본 논문에서는 State Space Models (SSMs)을 기반으로 한 비전 모델의 효율성을 향상시키기 위해 토큰 기반의 프루닝(token pruning) 기법을 제안합니다. 기존의 Vision Transformers (ViTs) 기술을 활용한 토큰 프루닝 기법의 제한점을 분석하고, SSM의 고유한 계산 특성을 고려하여 새로운 방법을 개발했습니다.

- **Technical Details**: 제안된 방법은 pruning-aware hidden state alignment 기법을 도입하여 남아있는 토큰의 이웃을 안정화하고, SSM 모델에 적합한 토큰 중요성 평가(token importance evaluation) 방법을 통해 토큰 프루닝을 수행합니다. 이로 인해 효율적인 구현 및 실질적인 가속화가 이루어집니다.

- **Performance Highlights**: 제안된 방법은 ImageNet에서 41.6\%의 FLOPs 감소와 함께 81.7\%의 정확도를 달성하며, 다양한 작업에서 성능에 미치는 영향을 최소화하면서도 계산량을 대폭 줄일 수 있음을 입증했습니다. 또한, 이 연구는 SSM 기반 비전 모델의 동작을 이해하는 데 더 깊은 통찰을 제공합니다.



### ProMerge: Prompt and Merge for Unsupervised Instance Segmentation (https://arxiv.org/abs/2409.18961)
Comments:
          ECCV2024 camera-ready

- **What's New**: 이 논문에서는 Unsupervised instance segmentation(비지도 인스턴스 분할)에서 새로운 방법인 Prompt and Merge(ProMerge)를 제안합니다. ProMerge는 self-supervised visual features(자기지도 시각적 특징)를 활용해 패치를 초기 그룹화하고 전략적으로 병합하는 접근법입니다.

- **Technical Details**: ProMerge는 DINO와 같은 self-supervised 모델에서 제공하는 강력한 시각적 특징을 활용하고, sophisticated background-based mask pruning technique(정교한 배경 기반 마스크 가지치기 기법)을 적용하여 초기 세그먼트를 병합합니다. 또한, 기존의 normalized-cut(정규화 컷)을 사용하는 방법에 비해 계산 요구 사항을 줄이며, inference speed(추론 속도)를 크게 개선합니다.

- **Performance Highlights**: ProMerge는 경쟁력 있는 결과를 제공하며, 기존 최첨단 normalized-cut 기반 접근법에 비해 추론 시간을 크게 단축시킵니다. 우리의 마스크 예측을 pseudo-labels(의사 레이블)로 사용하는 객체 탐지기를 훈련할 경우, 다양한 challenging instance segmentation benchmarks(도전적인 인스턴스 분할 벤치마크)에서 현재 최고 수준의 비지도 모델을 능가합니다.



### UniCal: Unified Neural Sensor Calibration (https://arxiv.org/abs/2409.18953)
Comments:
          ECCV 2024. Project page: this https URL

- **What's New**: 이 논문에서는 여러 LiDAR와 카메라를 장착한 자율주행 차량(SDV)에서 손쉽게 보정할 수 있는 통합된 프레임워크인 UniCal을 제공합니다. 이 접근법은 특정 보정 피듈(Fiducial) 없이도 연속적인 장면 표현을 통해 보정 과정을 진행할 수 있게 합니다.

- **Technical Details**: UniCal은 차별 가능한 장면 표현(differentiable scene representation)을 기반으로 하여, 다중 보기를 지오메트릭(geometric) 및 포토메트릭(photometric) 일관되게 렌더링하는 기능을 갖추고 있습니다. 또한, 드라이브-앤-칼리브레이트(drive-and-calibrate) 방식을 통해 야외 센서 데이터를 활용하여 보정 작업을 수행합니다. 이를 통해 센서 보정과 기본 장면 표현을 공동 학습합니다.

- **Performance Highlights**: UniCal은 기존 보정 시스템에 비해 비용 절감 및 운영 오버헤드 감소를 자랑하며, 여러 데이터셋에 대한 포괄적인 평가 결과 UniCal이 기존 보정 접근법에 비해 더 정확하거나 동등한 정밀도를 보임을 demonstrates합니다.



### Spectral Wavelet Dropout: Regularization in the Wavelet Domain (https://arxiv.org/abs/2409.18951)
Comments:
          Accepted by The International Conference on Machine Learning and Applications (ICMLA) 2024

- **What's New**: 이번 연구에서는 Spectral Wavelet Dropout (SWD)이라는 새로운 정규화 (regularization) 방법을 소개합니다. 이 방법은 1D-SWD와 2D-SWD 두 가지 변형을 포함하며, CNN의 일반화 (generalization) 능력을 향상시킵니다.

- **Technical Details**: SWD는 특징 맵의 이산 웨이브렛 분해 (discrete wavelet decomposition)에서 세부 주파수 대역을 무작위로 제거하여 작동합니다. SWD는 기존의 Spectral "Fourier" Dropout (2D-SFD)와 다르게, 주파수 영역(Fourier domain)에서 계수를 제거하는 대신 웨이브렛을 사용합니다. SWD는 단일 하이퍼파라미터만 요구하며, 1D-SFD의 1차원 버전도 구현하여 포괄적인 비교 연구를 수행합니다.

- **Performance Highlights**: CIFAR-10/100 벤치마크에서 1D 및 2D SWD 변형은 1D-SFD 및 2D-SFD에 비해 뛰어난 성능을 보였습니다. 특히, 1D-SWD는 1D/2D-SFD에 비해 계산 복잡도가 현저히 낮습니다. Pascal VOC 객체 탐지 벤치마크에서도 SWD 변형이 1D-SFD 및 2D-SFD를 초월하는 성능과 낮은 계산 복잡도를 보여주었습니다.



### From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding (https://arxiv.org/abs/2409.18938)
Comments:
          11 pages

- **What's New**: 본 논문에서는 MultiModal Large Language Models (MM-LLMs)과 시각적인 인코더(visual encoders)의 통합을 통해 긴 비디오 이해(long video understanding)에서의 고유한 도전 과제를 조명합니다. 기존의 정적 이미지(static image)와 짧은 비디오(short video) 이해와의 차별점을 분명히 합니다.

- **Technical Details**: MM-LLMs의 설계(model design) 및 훈련(training methodologies) 방식에서 긴 비디오 이해를 위한 발전을 다룹니다. 짧은 비디오는 연속적인 프레임을 포함하여 공간(spatial) 및 사건 내 템포럴(temporal) 정보를 가지며, 긴 비디오는 여러 사건이 여유 있는 시간 간격으로 발생합니다.

- **Performance Highlights**: 기존 MM-LLMs의 비디오 이해(video understanding) 벤치마크 성능을 비교하고, 긴 비디오 이해를 위한 향후 발전 방향을 논의합니다.



### ReviveDiff: A Universal Diffusion Model for Restoring Images in Adverse Weather Conditions (https://arxiv.org/abs/2409.18932)
- **What's New**: 본 논문에서는 다양한 열악한 환경에서 촬영된 이미지의 품질을 향상시키고 복원할 수 있는 범용 네트워크 아키텍처 'ReviveDiff'를 제안합니다. 이 접근은 특정 작업을 위한 기존의 방법들이 다른 종류의 열화 상황에 적용될 수 없다는 한계를 극복합니다.

- **Technical Details**: ReviveDiff 모델은 움직임이나 전자적 문제에 의한 열화와는 달리, 안개, 물, 저조도와 같은 자연 매체에 의해 발생하는 품질 저하 현상을 이해하고 이를 복원하기 위해 최신 diffusion models를 활용했습니다. 이 모델은 이미지 품질에 영향을 미치는 주요 요인인 선명도(sharpness), 왜곡(distortion), 노이즈 수준(noise level), 동적 범위(dynamic range), 색상 정확성(color accuracy) 등을 고려합니다.

- **Performance Highlights**: ReviveDiff는 Rainy, Underwater, Low-light, Smoke, Nighttime Hazy의 다섯 가지 열화 조건을 아우르는 일곱 개의 벤치마크 데이터셋에서 평가되었으며, 실험 결과 기존의 최첨단 방법들을 정량적 및 시각적으로 능가한다는 것을 입증했습니다.



### SurfaceAI: Automated creation of cohesive road surface quality datasets based on open street-level imagery (https://arxiv.org/abs/2409.18922)
Comments:
          4 pages, 2 figures; accepted at 2nd ACM SIGSPATIAL International Workshop on Advances in Urban-AI

- **What's New**: 본 논문은 도로 표면 유형과 품질에 대한 포괄적인 지리 참조 (georeferenced) 데이터셋을 생성하기 위해 설계된 SurfaceAI라는 파이프라인을 소개합니다. 이는 거리 수준 이미지에서 공개적으로 사용 가능한 자료를 활용하여 개발되었습니다.

- **Technical Details**: SurfaceAI는 crowdsourced (크라우드소싱) 데이터인 Mapillary 데이터를 활용하여 거리 수준 이미지에서 보이는 도로 표면의 유형과 품질을 예측하는 모델을 학습시키고, 이를 통해 전체 도로 구간 조건에 대한 통합 정보를 제공합니다.

- **Performance Highlights**: 도로의 불균형성이 교통 참가자의 안전과 편안함에 미치는 중대한 영향을 강조하며, 인프라 모델링 및 분석에 있어 세부 도로 표면 데이터의 필요성을 충족하는 데 기여합니다.



### Improving Visual Object Tracking through Visual Prompting (https://arxiv.org/abs/2409.18901)
Comments:
          Accepted and to appear in IEEE Transactions on Multimedia

- **What's New**: 본 연구에서는 PiVOT(Visual Prompting mechanism for generic Visual Object Tracking)를 통해 기존의 목표와 주변 방해물(distractor)을 구별하는 문제를 해결하기 위한 새로운 시각적 프롬프트(prompt) 생성 네트워크를 제안합니다.

- **Technical Details**: PiVOT는 CLIP이라는 사전 훈련된 모델을 사용하여 자동으로 시각적 프롬프트를 생성하고 수정합니다. CLIP은 범주 수준의 광범위한 지식을 제공하며, 트래커는 객체 인스턴스(instance-specific data)에 대한 훈련을 통해 고유한 객체 인스턴스를 인식하는 데 강점을 가집니다. PiVOT는 시각적 프롬프트를 잠재적 목표 위치를 강조하는 형태로 컴파일합니다.

- **Performance Highlights**: 여러 벤치마크에서 수행된 실험을 통해 PiVOT은 방해 물체(distractors)를 억제하고 트래커의 성능을 향상시키는 데 효과적임을 입증하였습니다.



### Unsupervised Low-light Image Enhancement with Lookup Tables and Diffusion Priors (https://arxiv.org/abs/2409.18899)
Comments:
          13 pages, 10 figures

- **What's New**: 이번 연구에서는 Poor illumination 환경에서 degraded된 이미지를 효율적으로 복구하기 위한 새로운 Unsupervised LIE framework, DPLUT(Diffusion Prior and Lookup Table)를 제안합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성요소로 이루어집니다: LLUT(Light Adjustment Lookup Table)와 NLUT(Noise Suppression Lookup Table). LLUT는 unsupervised loss를 사용하여 최적화되며, 특정 이미지의 dynamic range를 조정하기 위한 pixel-wise curve parameters를 예측합니다. NLUT는 빛의 밝기 증가 후에 발생하는 noise를 제거하도록 설계되었습니다. 또한, diffusion 모델이 noise에 민감하기 때문에 diffusion priors를 도입하여 높은 성능의 noise 억제를 달성합니다.

- **Performance Highlights**: 광학적 품질과 효율성 측면에서 우리 방법이 기존의 최신 기법(State-of-the-art methods)보다 우수한 성능을 보인다는 실험 결과가 있습니다.



### Detecting Dataset Abuse in Fine-Tuning Stable Diffusion Models for Text-to-Image Synthesis (https://arxiv.org/abs/2409.18897)
- **What's New**: 본 논문은 텍스트-이미지 합성을 위한 Stable Diffusion 모델의 파인 튜닝 과정에서 발생할 수 있는 데이터셋 남용 문제를 다룹니다. 이를 위해 데이터셋의 무단 사용을 감지하고 데이터 유출을 추적할 수 있는 데이터셋 워터마킹 프레임워크를 제시합니다.

- **Technical Details**: 제안된 프레임워크는 여러 워터마킹 방식에서 두 가지 주요 전략을 사용하여 대규모 데이터셋의 권한 부여를 효과적으로 수행합니다. 이 프레임워크는 높은 탐지 정확도를 위해 데이터의 2%만 수정하면 되는 최소한의 영향으로 데이터를 보호하는 장점을 가지고 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 프레임워크의 효과성, 데이터 유출 추적 능력 및 강인성과 전이 가능성을 입증하였습니다. 이로써 데이터셋 남용을 탐지하는 데 실용적인 응용이 가능함을 보여줍니다.



### S2O: Static to Openable Enhancement for Articulated 3D Objects (https://arxiv.org/abs/2409.18896)
- **What's New**: 이번 논문에서는 정적인 3D 객체로부터 상호작용이 가능한 3D 객체를 생성하는 새로운 S2O(Static to Openable) 작업을 소개합니다. 이는 열 수 있는 부분 탐지, 운동 예측 및 내부 기하학 완성을 통해 이루어집니다.

- **Technical Details**: S2O 작업을 수행하기 위해 통합된 프레임워크를 설계하였으며, 상호작용이 가능한 3D 객체들로 구성된 도전적인 데이터셋을 구축하였습니다. 이 데이터셋은 체계적인 평가를 위한 테스트 베드 역할을 합니다.

- **Performance Highlights**: 실험을 통해 이전 연구에서 제시된 방법들과 S2O 작업을 위한 간단하면서도 효과적인 휴리스틱 방법들을 비교하였습니다. 정적인 3D 객체를 상호작용이 가능한 형태로 변환하는 것이 가능하지만, 실제 환경에 대한 일반화에는 모두 어려움을 겪고 있음을 확인했습니다. 이를 통해 향후 연구 방향에 대한 가능성을 제시합니다.



### Explainable Artifacts for Synthetic Western Blot Source Attribution (https://arxiv.org/abs/2409.18881)
Comments:
          Accepted in IEEE International Workshop on Information Forensics and Security - WIFS 2024, Rome, Italy

- **What's New**: 이 논문은 최신 인공지능 기술이 생성한 합성 과학 이미지가 실제 이미지와 구분되지 않는 상황을 다룹니다. 특히, 'paper mills'라는 조직이 이러한 기술을 악용하여 허위 기사를 생성하는 문제를 강조합니다.

- **Technical Details**: 연구팀은 Generative Adversarial Networks (GANs)와 Diffusion Models 같은 최신 generative 모델이 생성하는 설명 가능한 아티팩트(artifacts)를 식별하려고 합니다. 이를 통해 open-set identification과 source attribution을 가능하게 하여, 이미지가 생성된 모델을 지목할 수 있도록 합니다.

- **Performance Highlights**: 이 연구는 이전의 블랙박스 솔루션에서 벗어나 합성 이미지 안의 아티팩트를 통해 검출 프로세스에 대한 인사이트를 제공하며, 다양한 모델 간의 일반화 문제를 해결하는데 기여할 것으로 기대됩니다.



### CemiFace: Center-based Semi-hard Synthetic Face Generation for Face Recognition (https://arxiv.org/abs/2409.18876)
Comments:
          accepted to NeurIPS 2024. We are preparing the camera-ready version according to the reviews

- **What's New**: 이 논문에서는 얼굴 인식(Face Recognition, FR) 기술 개발에 있어 개인정보 보호 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 생성 기법으로 합성된 얼굴 이미지는 성능 저하 문제를 겪는 경우가 많지만, 본 연구는 얼굴 이미지의 유사성이 모델 성능에 미치는 영향을 체계적으로 조사합니다.

- **Technical Details**: 우리는 Center-based Semi-hard Synthetic Face Generation (CemiFace)이라는 새로운 확산 기반 접근 방식을 제안합니다. 이 방법은 각기 다른 유사성 수준을 가진 얼굴 샘플을 생성하여 효과적인 훈련 데이터셋을 형성합니다. 논문에서는 특정 유사성을 가진 얼굴 이미지가 FR 모델 훈련에 어떻게 기여하는지를 분석합니다.

- **Performance Highlights**: 실험 결과, 적절한 유사성을 가진 얼굴 데이터셋으로 훈련 시, 이전 생성 방법에 비해 경쟁력 있는 성능을 낼 수 있다는 것을 보였습니다.



### Emu3: Next-Token Prediction is All You Need (https://arxiv.org/abs/2409.18869)
Comments:
          Project Page: this https URL

- **What's New**: Emu3라는 새로운 모달 모델이 소개되었습니다. 이 모델은 오직 next-token prediction을 사용하여 훈련되었습니다.

- **Technical Details**: Emu3는 이미지를 token화하여 텍스트 및 비디오와 함께 결합한 혼합 멀티모달 시퀀스에서 훈련된 단일 Transformer 모델을 기반으로 합니다. 모델은 고급 멀티모달 작업을 처리하도록 설계되었습니다.

- **Performance Highlights**: Emu3는 SDXL 및 LLaVA-1.6와 같은 기존의 주요 모델을 초월하며, 생성 및 인식 작업 모두에서 잘 작동합니다. 또한 Emu3는 비디오 시퀀스에서 다음 토큰을 예측하여 고충실도 비디오를 생성할 수 있습니다.



### MCUBench: A Benchmark of Tiny Object Detectors on MCUs (https://arxiv.org/abs/2409.18866)
Comments:
          Code and data are available at this https URL

- **What's New**: MCUBench라는 새로운 벤치마크가 소개되었습니다. 이 벤치마크는 100개 이상의 YOLO 기반 객체 탐지 모델을 다루고 있으며, VOC 데이터셋을 바탕으로 7가지 다양한 MCU에서 평가되었습니다.

- **Technical Details**: MCUBench는 다양한 입력 해상도와 YOLO 기반 원스테이지 탐지기에 대한 평균 정확도(average precision), 대기 시간(latency), RAM, 플래시(Flash) 사용량에 대한 상세 데이터를 제공합니다. 고정된 훈련 파이프라인을 가진 통제된 비교를 통해 포괄적인 성능 메트릭을 수집합니다.

- **Performance Highlights**: Pareto 최적 분석을 통해 현대 탐지 헤드와 훈련 기법을 통합함으로써 YOLO 아키텍처(legacy 모델인 YOLOv3 포함)가 평균 평균 정확도(mAP)와 대기 시간 간의 효율적인 균형을 이룰 수 있음을 보여주었습니다. MCUBench는 현대 객체 탐지기의 MCU 성능을 벤치마킹하는 유용한 도구이며, 특정 제약을 기반으로 모델 선택에 도움을 줍니다.



### LW2G: Learning Whether to Grow for Prompt-based Continual Learning (https://arxiv.org/abs/2409.18860)
Comments:
          submit to neurips2024

- **What's New**: 이번 논문에서는 Continual Learning (CL) 분야에서 Prompt-based Continual Learning (PCL)의 성능 향상을 위한 새로운 접근을 제안합니다. 새로운 모듈을 통해 이전 작업 간의 차이를 기반으로 프롬프트 세트를 성장할지 여부를 학습합니다.

- **Technical Details**: 제안된 모듈인 Learn Whether to Grow (LW2G)는 여러 작업이 공유하는 공통점이 있을 때는 공유 프롬프트 세트를 사용하고, 이전 작업과의 큰 차이가 있을 경우 새로운 세트를 추가하는 방식을 채택합니다. 또한, Hinder Forward Capability (HFC) 메트릭을 활용하여 새로운 작업 학습에 대한 방해 요인을 측정합니다. 이를 통해 동적 임계값을 사용해 성장 여부를 자동으로 학습하는 Dynamic Growing Approach를 설계했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 방법들에 비해 우수한 성능을 보여주었으며, 코드도 공유되어 있어 연구자들이 쉽게 활용할 수 있도록 하였습니다.



### Space-time 2D Gaussian Splatting for Accurate Surface Reconstruction under Complex Dynamic Scenes (https://arxiv.org/abs/2409.18852)
Comments:
          Project page: this https URL

- **What's New**: 새로운 Surface Reconstruction (표면 재구성) 방법을 제시하며, 복잡한 동적 장면에서의 다중 인물 활동과 인간-객체 상호작용에 대한 문제를 해결하기 위해 Space-Time 2D Gaussian Splatting (공간-시간 2D 가우시안 스플래팅) 접근법을 도입했습니다.

- **Technical Details**: 이 방법은 동적 장면에서의 기하학적 품질을 향상시키기 위해 2D Gaussian splats를 학습하고, 깊이(Depth)와 법선(Normal) 정규화기를 도입해 물체 표면에 위치한 가우시안 디스크(Disks)를 변형합니다. 또한 복잡한 장면에서의 가려짐(Occlusion) 문제를 해결하기 위해 조합 불투명도 변형 전략(Opacity Deformation Strategy)을 소개합니다.

- **Performance Highlights**: 실제 세계의 희소 시점 비디오 데이터셋과 단안 동적 데이터셋에서 실험한 결과, 우리의 재구성이 최신 기술(State-of-the-art) 방법들보다 더 우수한 성능을 보이며, 특히 세부 표면 복원에서 뛰어난 결과를 나타냈습니다.



### MinerU: An Open-Source Solution for Precise Document Content Extraction (https://arxiv.org/abs/2409.18839)
Comments:
          MinerU Technical Report

- **What's New**: MinerU는 고정밀 문서 내용 추출을 위한 새로운 오픈소스 솔루션입니다. 기존의 OCR, 레이아웃 감지 및 수식 인식 방법의 한계를 극복하기 위해 다양한 문서 유형에서 효과적으로 콘텐츠를 추출합니다.

- **Technical Details**: MinerU는 정교한 PDF-Extract-Kit 모델을 활용하고 세밀하게 조정된 전처리(preprocessing) 및 후처리(postprocessing) 규칙을 적용하여 결과의 정확성을 높입니다.

- **Performance Highlights**: 실험 결과, MinerU는 다양한 문서 유형에서 일관되게 높은 성능을 달성하여 콘텐츠 추출의 품질과 일관성을 크게 향상시킵니다.



### Classification and regression of trajectories rendered as images via 2D Convolutional Neural Networks (https://arxiv.org/abs/2409.18832)
Comments:
          13 pages, 5 figures

- **What's New**: 이번 연구는 CNN (Convolutional Neural Networks)을 사용하여 다양한 형식으로 렌더링된 합성 궤적(synthetic trajectories)을 이미지로 변환하여 분류(classification) 및 회귀(regression) 문제를 해결하는 효과를 조사합니다.

- **Technical Details**: 본 연구에서는 궤적을 이미지로 변환할 때의 파라미터로 선 두께(line thickness), 이미지 해상도(image resolution), 동작 이력(motion history, 시간 요소의 색상 코딩), 앨리어싱 방지(anti-aliasing) 등을 고려하였습니다. CNN은 이미지에서 특징(feature)의 공간적 계층(spatial hierarchies)을 학습하는 능력을 활용하여 복잡한 형태를 인식합니다.

- **Performance Highlights**: 실험 결과에 따르면, 운동 방향이 중요한 응용 프로그램에서는 모델의 깊이에 따라 적절한 이미지 해상도를 선택하는 것이 중요하다. 또한, 궤적을 이미지로 렌더링함으로써 발생할 수 있는 정보 손실과 스펙트럼 변화 등의 아티팩트(artifacts)를 고려하는 것이 필수적입니다.



### YOLOv8-ResCBAM: YOLOv8 Based on An Effective Attention Module for Pediatric Wrist Fracture Detection (https://arxiv.org/abs/2409.18826)
Comments:
          Accepted by ICONIP 2024. arXiv admin note: substantial text overlap with arXiv:2402.09329

- **What's New**: 본 연구는 YOLOv8 네트워크 아키텍처에 ResCBAM(Convolutional Block Attention Module)을 통합한 YOLOv8-ResCBAM 모델을 제안합니다.

- **Technical Details**: YOLOv8-ResCBAM은 원래 YOLOv8의 구조에 attention 모듈을 적용하여 모델 성능을 개선합니다. ResCBAM은 residual block을 활용하여 네트워크의 중요한 정보를 강화합니다.

- **Performance Highlights**: 제안된 모델은 GRAZPEDWRI-DX 데이터셋에서 mAP 50(Intersection over Union 기준) 성능이 63.6%에서 65.8%로 증가하여 최고의 성능을 달성했습니다.



### EyeTrAES: Fine-grained, Low-Latency Eye Tracking via Adaptive Event Slicing (https://arxiv.org/abs/2409.18813)
Comments:
          32 pages,15 figures,

- **What's New**: Eye-tracking 기술에서 새롭게 제안된 EyeTrAES는 신경모방(event-based) 카메라를 사용하여 높은 정확도로 자연스러운 동공(pupil) 움직임을 추적하는 접근 방식을 소개합니다.

- **Technical Details**: EyeTrAES는 적응형 윈도우/슬라이싱 알고리즘을 활용하여 다양한 눈 움직임 패턴에 대한 비동기식 이벤트 데이터의 적절한 집합을 보장합니다. 단일 눈에서의 누적된 이벤트 프레임에 대해 경량 이미지 처리 기능을 적용하여 동공 분할 및 추적을 수행합니다.

- **Performance Highlights**: EyeTrAES는 동공 추적 충실도를 6% 이상 향상시켜 IoU가 92%에 도달하며, 경쟁 기술에 비해 최소 3배 더 낮은 대기 시간(latency)을 기록합니다. 또한, 개인별 미세한 동공 운동을 통해 생체 인식(Biometric) 지문으로 활용 가능함을 보여줍니다.



### MiniVLN: Efficient Vision-and-Language Navigation by Progressive Knowledge Distillation (https://arxiv.org/abs/2409.18800)
- **What's New**: 최근 Embodied Artificial Intelligence (Embodied AI) 분야에서 모델의 크기가 증가하는 반면, 컴퓨팅 능력은 제한적이라는 문제가 발생했습니다. 본 논문은 Vision-and-Language Navigation (VLN) 과제를 해결하기 위한 두 단계 지식 증류 (knowledge distillation) 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 두 단계에 걸쳐 지식을 캡처하며, 첫 번째 단계에서는 미세한 지식(fine-grained knowledge)을 사전 학습(pretraining) 과정에서, 두 번째 단계에서는 내비게이션 특정 지식(navigation-specific knowledge)을 파인 튜닝(fine-tuning) 과정에서 획득합니다. MiniVLN이라는 학생 모델은 이러한 지식 증류 기술의 잠재력을 시연합니다.

- **Performance Highlights**: MiniVLN은 R2R와 REVERIE 벤치마크에서 교사 모델(teacher model)과 동등한 성능을 기록하며, 모델 파라미터 수는 교사 모델의 약 12%에 불과합니다.



### Supervised Learning Model for Key Frame Identification from Cow Teat Videos (https://arxiv.org/abs/2409.18797)
- **What's New**: 이 논문에서는 신경망(neural networks)과 비디오 분석(video analysis)을 이용하여 젖소의 유선 염증(티티스, mastitis) 위험 평가의 정확성을 향상시키는 방법을 제안합니다.

- **Technical Details**: 젖소의 티티스 감염을 탐지하기 위해, 저자들은 촬영된 비디오에서 유선이 온전하게 보이는 주요 프레임을 식별하는 신경망을 사용합니다. 이러한 주요 프레임은 수의사들이 유선 건강 상태를 평가할 시간적 유연성을 제공하며, 평가의 효율성과 정확성을 증가시킵니다. 복잡한 환경, 변화하는 소의 자세와 위치, 비디오에서 유선을 식별하는 어려움 등이 주요 도전과제로 제시됩니다.

- **Performance Highlights**: 제안된 방법은 유선 비디오에서 주요 프레임을 식별하는 데 있어 단일 거리 척도 또는 모델을 사용할 때보다 성능(F-score)이 향상된 것으로 나타났습니다.



### Student-Oriented Teacher Knowledge Refinement for Knowledge Distillation (https://arxiv.org/abs/2409.18785)
- **What's New**: 이 논문에서는 전통적인 지식 증류(knowledge distillation) 방법이 아닌 학생 중심의 새로운 접근 방식을 소개합니다. 'Student-Oriented Knowledge Distillation (SoKD)'를 통해 교사의 지식을 학생의 필요에 맞게 세밀하게 조정하는 방식을 제안합니다.

- **Technical Details**: SoKD는 훈련 중에 학습 가능한 feature augmentation 전략을 통합하여, 교사의 지식을 학생에게 동적으로 조정합니다. 또한, 'Distinctive Area Detection Module (DAM)'을 도입하여 교사와 학생 간의 상호 관심 지역을 식별함으로써 지식 전송을 보다 효과적으로 집중시킵니다.

- **Performance Highlights**: 광범위한 실험 결과를 통해 제안된 방법의 효율성과 일반화 가능성이 입증되었습니다. 이는 다양한 지식 증류 방법과 통합하여 사용할 수 있는 플러그인 형태로 기능합니다.



### Relighting from a Single Image: Datasets and Deep Intrinsic-based Architectur (https://arxiv.org/abs/2409.18770)
Comments:
          Accepted for publication as a Regular paper in the IEEE Transactions on Multimedia

- **What's New**: 단일 이미지 장면 리라이트( Single image scene relighting) 연구는 새로운 조명 조건에 따라 현실감 있는 이미지를 생성하는 데 주력하고 있습니다. 본 연구는 데이터셋과 방법론적 관점 모두에서 이 문제를 해결하고 있습니다.

- **Technical Details**: 우리는 두 개의 새로운 데이터셋을 제안합니다. 하나는 본질 성분(intrinsic components)의 실제 값이 포함된 합성 데이터셋(synthetic dataset)이고, 다른 하나는 실험실 조건에서 수집된 실제 데이터셋(real dataset)입니다. 리라이트 파이프라인에서 물리적 일관성(physical consistency)을 포함하기 위해 본질 분해(intrinsic decomposition) 기반의 두 단계 네트워크를 구축하였습니다. 또한 우리가 만든 데이터셋과 기존 데이터셋에서 성능 테스트를 통해 우리의 방법이 최첨단 방법보다 뛰어나고, 사전 훈련(pretraining)된 방법의 성능을 향상시킬 수 있음을 입증하였습니다.

- **Performance Highlights**: 우리의 방법은 모든 조명 조건에 적응할 수 있어 애니메이션 결과를 생산할 수 있는 능력을 가지고 있습니다. 제공된 데이터셋, 방법 및 동영상은 공개적으로 이용 가능합니다.



### State-of-the-Art Periorbital Distance Prediction and Disease Classification Using Periorbital Features (https://arxiv.org/abs/2409.18769)
Comments:
          16 pages, 4 figures, 4 tables

- **What's New**: 이번 연구에서는 눈과 눈꺼풀 주위의 거리와 특징들을 정량화하고 질병을 모니터링하는 데 있어서 중요한 정보를 제공하기 위해 3가지 딥러닝 방법을 개발했습니다. 이를 통해 수동 측정의 주관성과 시간 소모를 극복할 수 있습니다.

- **Technical Details**: 연구팀은 세 가지 딥러닝(segmentation) 방법을 사용하여 periorbital distance를 예측하였으며, 이를 통해 예측된 거리의 MAE는 훈련된 인간 주석자 간의 오류와 비슷하거나 그보다 낮았습니다. 현대 방식(SOTA)과 비교하여 모든 데이터셋에서 평균적으로 더 나은 성능을 보였습니다.

- **Performance Highlights**: 모델은 병든 눈에 대해 강력한 segmentation을 달성했으며, 건강한 눈을 사용하여 훈련된 모델에서도 효과적이었습니다. 또한, periorbital distances는 하위 분류(classification) 모델에서 고품질 특징으로 사용될 수 있음을 입증했습니다.



### Charting the Future: Using Chart Question-Answering for Scalable Evaluation of LLM-Driven Data Visualizations (https://arxiv.org/abs/2409.18764)
- **What's New**: 본 연구는 Visual Question Answering (VQA) 모델을 활용하여 LLM(대형 언어 모델) 생성 데이터 시각화의 평가를 자동화하는 새로운 프레임워크를 제안합니다. 전통적인 평가 방법은 비용이 많이 들고 확장성이 부족한 인간 판단에 의존하거나 오직 데이터 정확도에만 집중하는 경향이 있습니다.

- **Technical Details**: VQA 모델을 사용하여 차트의 데이터 표현 품질 및 일반적인 의사소통 명확성을 평가합니다. 실험은 ChartQA와 PlotQA라는 두 가지 주요 VQA 벤치마크 데이터 세트를 사용하여 OpenAI의 GPT-3.5 Turbo 및 Meta의 Llama 3.1 70B-Instruct 모델로 생성된 시각화를 통해 실시되었습니다.

- **Performance Highlights**: LLM이 생성한 차트는 VQA 성과 기반으로 원래의 비-LLM 생성 차트의 정확도에 미치지 못하며, few-shot prompting이 차트 생성의 정확도를 크게 향상시킬 수 있음을 보여줍니다. 그러나 LLM이 인간이 생성한 그래프의 정확성과 완전히 일치하려면 아직 많은 발전이 필요하다는 점이 강조됩니다.



### Enhancing Explainability in Multimodal Large Language Models Using Ontological Contex (https://arxiv.org/abs/2409.18753)
- **What's New**: 최근 Multimodal Large Language Models (MLLMs)에 대한 관심이 증가하고 있습니다. 이러한 모델들은 이미지와 텍스트를 통합하여 다양한 작업을 수행하는 데 있어 놀라운 잠재력을 가지고 있지만, 특정 도메인 응용 분야에서의 정확한 캡션과 시각적 개념 해석에 여전히 어려움을 겪고 있습니다. 본 연구에서는 이러한 문제를 해결하기 위해 온톨로지(ontology)를 통합하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 우리의 방법은 기존 식물 질병 온톨로지에서 식물 질병에 대한 개념을 활용하여 MLLMs에 질의하고 이미지에서 관련 시각적 개념을 추출합니다. 이후, 식별된 개념에 따라 질병을 분류하기 위해 온톨로지의 추론(reasoning) 능력을 활용합니다. 이는 도메인 특화 응용 분야에서 질병을 설명하는 개념이 정확하게 모델이 사용하도록 하는 데 있어 중요합니다.

- **Performance Highlights**: 온톨로지를 사용함으로써, 우리는 모델이 MLLMs에 의한 개념 주석이 온톨로지의 내용과 일치하는지를 검증하고 오류의 논리를 추적하여 결정 과정에서 투명성(transparency), 설명 가능성(explainability), 신뢰성(trust)을 높일 수 있습니다. 본 프레임워크는 기존의 잘 알려진 MLLMs를 활용한 경험적 연구에 의해 지원되는 온톨로지와 MLLMs의 시너지(new direction)를 제공하도록 설계되었습니다.



### MemFusionMap: Working Memory Fusion for Online Vectorized HD Map Construction (https://arxiv.org/abs/2409.18737)
- **What's New**: 이 논문에서는 자율 주행 시스템을 위한 고해상도 (HD) 맵 구축에 개선된 시간적 추론 능력을 가진 MemFusionMap이라는 새로운 모델을 제안합니다.

- **Technical Details**: MemFusionMap은 역사적 프레임 간의 추론을 개선하는 작업 메모리 융합 모듈을 포함하며, 차량의 궤적 정보를 명시적으로 인지하도록 설계된 새로운 시간적 오버랩 히트맵을 도입합니다. 이러한 두 가지 설계를 통합하여 HD 맵 구축의 성능을 크게 향상시킵니다.

- **Performance Highlights**: MemFusionMap은 기존 방법들보다 최대 5.4% 높은 mAP (mean Average Precision)를 기록하여 성능이 크게 향상되었습니다.



### Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieva (https://arxiv.org/abs/2409.18733)
- **What's New**: 이 논문에서는 훈련 과정이 필요 없는 SearchDet라는 긴 꼬리(long-tail) 객체 탐지(object detection) 프레임워크를 소개합니다. 이 프레임워크는 개방형 어휘(open-vocabulary) 객체 탐지 성능을 크게 향상시킵니다.

- **Technical Details**: SearchDet는 객체의 긍정적(positive) 및 부정적(negative) 이미지 집합을 검색하여 이를 임베딩(embedding)하고, 입력 이미지에 가중치가 부여된 쿼리를 계산하여 이미지 내에서 원하는 개념을 탐지합니다. 이 방법은 간단하고 훈련 과정이 필요 없으며, GroundingDINO와 같은 최신(state-of-the-art) 모델에 비해 ODinW에서 48.7% mAP 향상, LVIS에서 59.1% mAP 향상을 달성하였습니다.

- **Performance Highlights**: 우리의 접근 방식은 웹에서 가져온(exemplars) 객체의 집합에 기반하여 안정적인 성능을 제공하며, 이는 비싼 데이터 주석(data annotation) 및 훈련 절차를 줄이는 방향으로 나아갈 수 있는 가능성을 나타냅니다.



### Learning from Pattern Completion: Self-supervised Controllable Generation (https://arxiv.org/abs/2409.18694)
- **What's New**: 이번 논문에서는 인공지능(AI) 분야에서 일반적으로 사용되는 레이블이 있는 학습 데이터 세트에 의존하지 않고 스스로 조절 가능한 생성(self-supervised controllable generation) 방법론을 제안합니다. 이러한 방법은 인간 뇌의 연상 작용을 모방한 것으로, 기능적 전문화를 이루는 모듈화된 오토인코더(modular autoencoder) 네트워크를 통해 이루어집니다.

- **Technical Details**: 제안된 프레임워크인 SCG(self-supervised controllable generation)는 모듈 내 독립성(intra-module independence)과 모듈 간 상관관계(inter-module correlation)를 촉진하기 위해 동등 변환 제약조건(equivariant constraint)을 도입합니다. 이로 인해 색상, 명도, 엣지 검출 모듈 처리에서 기능적 전문성을 확보합니다. 또한, 자기 감독형 패턴 완성(self-supervised pattern completion) 접근을 통해 학습을 진행합니다.

- **Performance Highlights**: 실험 결과, SCG는 색조, 명도, 엣지 검출의 모듈 처리에서 뛰어난 기능적 전문화를 보여주며, 인간 뇌와 유사한 방향 선택성(orientation selectivity), 색상 대립(color antagonism), 중심-주변 수용 영역(center-surround receptive fields) 특징을 나타냅니다. SCG는 페인팅(다시 그린 그림), 스케치, 고대 그래피티와 같은 다양한 작업에서 뛰어난 일반화 능력을 보여줍니다. 기존의 ControlNet과 비교할 때, SCG는 더 높은 노이즈 환경에서도 우수한 강건성(robustness)을 보이며, 자기 감독형 학습 덕분에 향후 더 나은 확장 가능성(scalability) 잠재력을 지니고 있습니다.



### A Novel Unified Architecture for Low-Shot Counting by Detection and Segmentation (https://arxiv.org/abs/2409.18686)
Comments:
          Accepted to NeurIPS2024

- **What's New**: GeCo는 정확한 객체 탐지, 분할 및 수량 추정을 통합된 아키텍처에서 수행할 수 있는 새로운 저샷(object counting) 카운터입니다.

- **Technical Details**: GeCo는 객체 외형의 다양성을 극복하기 위해 새로운 밀집 객체 쿼리(dense object query) 형식을 채택합니다. 또한, 탐지 작업을 직접 최적화하는 새로운 카운팅 손실(counting loss)을 제안하여 기존의 대리 손실(surrogate loss)에서 발생하는 문제를 피합니다.

- **Performance Highlights**: GeCo는 총 카운트 MAE에서 기존의 몇몇 샷 탐지 기반 카운터 대비 약 25% 향상된 성능을 보이며, 모든 저샷 카운팅 설정에서 새로운 최첨단(state-of-the-art) 결과를 세웠습니다.



### Image-guided topic modeling for interpretable privacy classification (https://arxiv.org/abs/2409.18674)
Comments:
          Paper accepted at the eXCV Workshop at ECCV 2024. Supplementary material included. Code available at this https URL

- **What's New**: 이 논문은 이미지에 포함된 개인정보를 예측하고 설명하기 위한 새로운 접근 방식을 제시합니다. 작가는 이미지 내용에 대한 자연어(content descriptors)를 기반으로 개인 정보 보호를 예측하였습니다.

- **Technical Details**: 우리는 이미지 안내 주제 모델링(Image-guided Topic Modeling, ITM)이라는 새로운 방법을 사용하여 개인 정보 보호 점수에 해당하는 설명자를 생성합니다. ITM은 비전 정보와 비전 언어 모델의 이미지 텍스트 설명을 멀티모달 정렬(multimodality alignment)을 통해 활용합니다.

- **Performance Highlights**: Priv×ITM 분류기는 해석 가능한(reference interpretable) 기존 방법보다 5% 더 높은 정확도로 성능을 보였으며, 현재 비해석 가능한(state-of-the-art) 모델과 유사한 성능을 보입니다.



### Exploiting Motion Prior for Accurate Pose Estimation of Dashboard Cameras (https://arxiv.org/abs/2409.18673)
- **What's New**: 본 연구는 대시캠(dashcam) 이미지를 위한 정밀한 자세 추정(pose estimation) 방법을 제안합니다. 이를 통해 카메라의 고유한 모션 프라이어(camera motion prior)를 활용하여 기존 이미지 매칭 방법의 한계를 극복하고자 합니다.

- **Technical Details**: 대시캠으로 캡처된 이미지 시퀀스는 일반적으로 전방 이동(forward movement)이나 측면 회전(lateral turns)과 같은 뚜렷한 모션 프라이어를 나타냅니다. 본 연구는 이를 기반으로 카메라 모션 프라이어를 학습하는 포즈 회귀 모듈(pose regression module)을 개발하고, 이를 관계 추정(correspondence estimation) 및 자세 추정 과정에 통합했습니다.

- **Performance Highlights**: 실제 대시캠 데이터셋에서 우리의 방법은 AUC5°에 대한 포즈 추정에서 기준선(baseline)보다 22% 향상된 성능을 보였으며, 재투영 오류(reprojection error)가 적은 이미지를 19% 더 많이 추정할 수 있었습니다.



### When SAM2 Meets Video Camouflaged Object Segmentation: A Comprehensive Evaluation and Adaptation (https://arxiv.org/abs/2409.18653)
Comments:
          Technical report

- **What's New**: 이번 연구는 Segment Anything Model 2 (SAM2)이 비디오에서 위장된 객체 세분화(video camouflaged object segmentation, VCOS) 작업에서 어떻게 활용될 수 있는지를 조사합니다. VCOS는 색상과 질감이 비슷하고 조명이 좋지 않은 환경에서 객체를 감지하는 어려운 과제입니다.

- **Technical Details**: 연구에서는 SAM2의 성능을 다양한 모델과 프롬프트(클릭, 박스, 마스크)를 사용하여 위장된 비디오 데이터셋에서 평가하였습니다. 또한, 기존의 다중 모달 대형 언어 모델(multimodal large language models, MLLMs) 및 VCOS 방법과 SAM2의 통합을 탐구하였습니다. SAM2를 비디오 위장 데이터셋에 맞추어 세밀하게 조정(fine-tuning)하여 적용하였습니다.

- **Performance Highlights**: SAM2는 비디오에서 위장된 객체를 탐지하는 매우 우수한 제로샷(zero-shot) 능력을 보여주었습니다. 또한, VCOS에 맞게 SAM2의 파라미터를 조정함으로써 이 능력을 더욱 향상시킬 수 있음을 입증하였습니다.



### Unsupervised Fingerphoto Presentation Attack Detection With Diffusion Models (https://arxiv.org/abs/2409.18636)
Comments:
          Accepted by IJCB 2024

- **What's New**: 이번 연구는 스마트폰 기반 비접촉식 지문 인증의 새로운 접근 방식으로, 최신 Denoising Diffusion Probabilistic Model (DDPM)을 활용하여 기존 방식의 단점을 해결하려는 시도를 보여줍니다.

- **Technical Details**: 제안된 방법은 오직 진짜(bona fide) 샘플로만 학습되며, 입력과 출력 쌍 간의 재구성 유사성(reconstruction similarity)을 계산하여 Presentation Attacks (PA)을 탐지합니다. 이를 통해 자율적으로 새로운 Presentation Attack Instruments (PAIs)의 탐지가 가능하게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 DDPM 기반 PAD 방법은 여러 PAI 클래스에서 다른 기초 비지도(unsupervised) 접근 방식에 비해 현저히 낮은 탐지 오류율(detection error rates)을 기록했습니다.



### From One to the Power of Many: Augmentations for Invariance to Multi-LiDAR Perception from Single-Sensor Datasets (https://arxiv.org/abs/2409.18592)
- **What's New**: 최근 자율주행 차를 위한 LiDAR 인식(methods) 기법들이 deep neural networks에 의해 더욱 향상되고 있습니다. 하지만 single-sensor(단일 센서) 환경에서 훈련된 모델을 modern multi-sensor(현대 다중 센서) 차량에 적용할 때 여전히 성능의 큰 차이가 존재합니다.

- **Technical Details**: 본 논문에서는 성능 차이의 원인으로 invariance(불변성)의 부족을 연구하고, multi-sensor LiDAR 환경으로의 더 나은 전이를 촉진하는 application-specific(응용 특정) data augmentations(데이터 증강)의 초기 솔루션을 제안합니다. 실험을 통해 제안한 증강 방법이 LiDAR 센서 설정의 일반화에 긍정적인 영향을 미친다는 것을 입증합니다.

- **Performance Highlights**: 제안된 데이터 증강 방법이 다양한 LiDAR 센서 설정에서 모델의 불변성 특성에 어떻게 영향을 미치는지에 대한 연구 결과를 포함하고 있습니다.



### Off to new Shores: A Dataset & Benchmark for (near-)coastal Flood Inundation Forecasting (https://arxiv.org/abs/2409.18591)
Comments:
          Accepted at NeurIPS 2024 Datasets & Benchmarks

- **What's New**: 이 논문에서는 홍수 예측을 위한 새로운 데이터셋과 벤치마크를 제안합니다. 이를 통해 다양한 상태-of-the-art 방법을 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: 새로 구성된 데이터셋은 홍수 범위를 예상하는 데 필요한 정보들을 포함하고 있으며, 두 개의 벤치마크 트랙으로 나누어져 있습니다: i) 일반적인 홍수 예측 및 ii) 해안 지역에 집중한 예측.

- **Performance Highlights**: 이 데이터셋과 벤치마크는 홍수 예측과 관련된 연구 및 개발에 중요한 기초 자료를 제공하며, 향후 해결책을 모색하는 데 도움을 줄 것입니다.



### Cross-video Identity Correlating for Person Re-identification Pre-training (https://arxiv.org/abs/2409.18569)
Comments:
          NeurIPS 2024 Accepted Paper

- **What's New**: 본 연구에서는 Cross-video Identity-cOrrelating pre-traiNing (CION) 프레임워크를 제안하여 서로 다른 동영상에서의 동일 인물 이미지 간의 정체성 불변성을 고려합니다. 이전 연구들은 주로 인스턴스 수준 또는 단일 비디오 트랙렛 수준에서의 사전 학습에 국한되어 있었으나, CION은 이를 극복합니다.

- **Technical Details**: CION은 intra-identity consistency와 inter-identity discrimination을 포괄적으로 고려하는 노이즈 개념을 정의하고, 이를 점진적인 다단계 디노이징 문제로 모델링하여 cross-video 이미지 간의 정체성 상관관계를 탐색합니다. 또한, 정체성 유도 자기 증류 손실(identity-guided self-distillation loss)을 제안하여 사람 이미지 내의 정체성 불변성을 활용하여 대규모 사전 학습을 개선합니다.

- **Performance Highlights**: CION은 적은 수의 훈련 샘플을 사용하여도 뛰어난 성능을 기록하고 있으며, 예를 들어 ResNet50-IBN을 사용했을 때, Market1501에서 93.3%의 mAP과 MSMT17에서 74.3%의 mAP를 달성하며 기존의 최첨단 모델에 비해 높은 성능을 보였습니다. 또한, CION은 다양한 연구 및 응용 요구를 충족하기 위해 32개의 모델을 포함하는 ReIDZoo 모델 저수(zoo)를 제공합니다.



### Harmonizing knowledge Transfer in Neural Network with Unified Distillation (https://arxiv.org/abs/2409.18565)
- **What's New**: 이번 논문은 Knowledge Distillation (KD) 방법에 새로운 관점을 제시합니다. 기존의 훈련된 네트워크에서 가벼운 네트워크로 지식을 전이하는 기법을 개선하기 위해 여러 지식 출처를 통합한 KD 프레임워크를 도입하였습니다.

- **Technical Details**: 본 연구에서는 중간 레이어의 특징을 집계하여 포괄적인 표현을 생성하고, 이러한 표현으로부터 분포 매개변수를 예측합니다. 이를 통해 네트워크의 서로 다른 단계에서 지식을 효과적으로 전이할 수 있는 분포 제약을 설정하였습니다.

- **Performance Highlights**: 수많은 실험을 통해 제안된 방법의 효과를 검증하였으며, 기존 방법들보다 더 완전하고 일관성 있는 지식 전이가 이루어졌음을 보여줍니다.



### AL-GTD: Deep Active Learning for Gaze Target Detection (https://arxiv.org/abs/2409.18561)
Comments:
          Accepted to ACM Multimedia 2024

- **What's New**: 본 논문에서는 사람의 시선이 향하는 지점을 감지하는 gaze target detection 분야에서, 라벨이 부착된 데이터셋의 크기에 대한 의존성을 줄이기 위한 AL-GTD라는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: AL-GTD(Acquisition Learning for Gaze Target Detection)는 감독 학습(supervised learning)과 자기 감독 학습(self-supervised learning)을 통합하여 샘플 획득 함수(sample acquisition function)를 통해 능동 학습(active learning)을 수행합니다. 이 방법은 훈련 단계에서 분포 변화(distribution shifts)를 완화하기 위해 의사 라벨링(pseudo-labeling)을 활용합니다.

- **Performance Highlights**: AL-GTD는 전체 훈련 데이터의 40-50%만 사용하여 모든 AUC 결과에서 최고의 성과를 달성하며, 10-20%의 훈련 데이터로도 만족스러운 성능에 빠르게 도달합니다. 이는 가장 정보가 풍부한 샘플을 획득할 수 있는 기능을 통해 가능하다는 점에서 주목할 만합니다.



### Reducing Semantic Ambiguity In Domain Adaptive Semantic Segmentation Via Probabilistic Prototypical Pixel Contras (https://arxiv.org/abs/2409.18543)
Comments:
          revise

- **What's New**: 이번 연구에서는 도메인 적응의 문제점을 해결하기 위해 확률론적 프로토타입 픽셀 대비(probabilistic proto-typical pixel contrast, PPPC)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 각 픽셀 임베딩을 다변량 가우시안 분포로 모델링하여 불확실성을 최대한 활용하고, 불확실성을 줄일 수 있는 방법을 제공합니다.

- **Technical Details**: PPPC는 각 픽셀의 임베딩을 확률적으로 모델링하고, 결정 경계(decision boundary)를 모호한 지점에서 멀리하도록 도와주는 프로토타입을 도출합니다. 이 방법은 분포 간 유사성을 계산하는 효율적인 방법을 사용하여 샘플링과 재매개변수화(reparameterization)의 필요성을 없애고 계산 오버헤드를 크게 줄입니다. 또한, 이미지 수준에서 모호한 크롭을 동적으로 선택하여 대조 학습(contrastive learning)에 개입되는 경계점의 수를 증가시킵니다.

- **Performance Highlights**: PPPC는 픽셀 수준의 모호성을 해결하고 판별적 표현(discriminative representations)을 생성하며, 합성(real-synthetic) 및 주간-야간(day-to-night) 적응 작업에서 크게 개선된 성능을 보여줍니다. 특히, 가장 도전적인 주간-야간 적응 시나리오에서 이전 최첨단(state-of-the-art, SOTA)보다 +5.2% mIoU의 성과를 달성하며, 다른 보지 못한 데이터셋에 대해 더 강한 일반화(generalization)를 보입니다.



### How Effective is Pre-training of Large Masked Autoencoders for Downstream Earth Observation Tasks? (https://arxiv.org/abs/2409.18536)
- **What's New**: 이번 연구는 Earth Observation (EO) 분야에서 ViT 기반의 Masked Autoencoders (MAE)의 사전 학습(pre-training) 효과를 조사했습니다. 사전 학습된 모델이 처음부터 학습하는 것보다 얼마나 유리한지에 대한 조건을 분석했습니다.

- **Technical Details**: 두 가지 대규모 ViT 기반 MAE 사전 학습 모델인 Prithvi와 SatMAE를 사용하여 재구성(reconstruction), 분할(segmentation), 분류(classification) 작업에서의 성능을 평가했습니다. Prithvi는 재구성 및 분할 작업에 대해 평가되었고, SatMAE는 분류 작업에서 성능을 평가 받았습니다.

- **Performance Highlights**: 사전 학습은 세부 조정(fine-tuning) 작업이 사전 학습 작업과 유사할 경우, 예를 들어 재구성 작업에서 특히 유리했습니다. 그러나 분할이나 분류 작업과 같은 경우, 특정 하이퍼파라미터 조정을 통해 처음부터 학습하는 것이 동등하게 또는 더 효과적임을 보여주었습니다.



### Prompt-Driven Temporal Domain Adaptation for Nighttime UAV Tracking (https://arxiv.org/abs/2409.18533)
Comments:
          Accepted by IROS2024

- **What's New**: 이번 연구는 야간 UAV(무인 항공기) 추적을 개선하기 위해 새로운 프롬프트 기반의 시간 도메인 적응 훈련 프레임워크(TDA)를 제안합니다. 이 프레임워크는 낮과 밤의 도메인에서 시간적 맥락의 분포를 정렬하여 더 나은 추적 성능을 제공합니다.

- **Technical Details**: 제안된 TDA 프레임워크는 시간적 특징 생성기(temporal feature generator)와 판별기(discriminator)를 훈련시켜 낮과 밤의 도메인 간 시간적 맥락을 정렬합니다. 또한, 프롬프트 기반 객체 탐지기(prompt-driven object miner)를 사용하여 주석이 없는 야간 비디오에서 객체를 정확히 찾습니다. 이와 함께, 장기 야간 UAV 추적을 위한 새로운 벤치마크가 구축되었습니다.

- **Performance Highlights**: TDA 프레임워크로 훈련된 추적기(TDA-Track)는 공개 및 자가 구축된 야간 벤치마크에서 뛰어난 성능을 보였으며, 실제 야간 테스트에서도 그 실용성을 입증했습니다.



### Neural Video Representation for Redundancy Reduction and Consistency Preservation (https://arxiv.org/abs/2409.18497)
- **What's New**: 본 논문에서는 임베디드 신경 표현(Implicit Neural Representations, INRs)을 활용하여 비디오 압축을 개선하는 새로운 방법을 제안합니다. 특히, 프레임의 고주파 성분을 기반으로 특징을 추출하여 기능 중복성을 줄이고, 인접 프레임 간의 특징 차이를 이용하여 프레임 간의 관계를 효율적으로 학습할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 각 프레임의 고주파 성분을 사용하여 특징을 추출하고, 인접 프레임 간의 특징 차이를 피드백으로 활용하여 네트워크가 프레임 관계를 학습하는 데 도움을 줍니다. 이를 통해 불필요한 중복성을 줄이고 더 나은 비디오 압축을 구현합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존의 HNeRV 방법보다 90%의 비디오에서 성능이 향상되는 것으로 나타났습니다.



### Temporal2Seq: A Unified Framework for Temporal Video Understanding Tasks (https://arxiv.org/abs/2409.18478)
- **What's New**: 이번 논문에서는 여러 비디오 이해 작업을 동시에 처리할 수 있는 통합 프레임워크인 Temporal2Seq를 제안합니다. 이는 다양한 작업에 대해 매우 유용한 방향으로 나아가는 것을 목표로 합니다.

- **Technical Details**: Temporal2Seq는 temporal video understanding 작업의 출력을 일련의 개별 토큰(Discrete Tokens)으로 수식화합니다. 이를 통해 단일 아키텍처 내에서 여러 비디오 이해 작업을 위한 generalist 모델을 훈련할 수 있습니다.

- **Performance Highlights**: Temporal2Seq 모델은 세 가지 작업에 대한 테스트 세트에서 평가되었으며, 단일 작업 훈련에 비해 다양한 작업에서 우수한 결과를 도출합니다. 또한, 새로운 데이터셋에서의 일반화 성능도 기존 특정 모델보다 뛰어난 것으로 나타났습니다.



### Underwater Image Enhancement with Physical-based Denoising Diffusion Implicit Models (https://arxiv.org/abs/2409.18476)
- **What's New**: 본 논문은 자율 수중 차량(AUV)에서의 수중 이미지 향상을 위한 UW-DiffPhys라는 새로운 물리 기반(diffusion-based) 이미지 향상 방법을 소개합니다. 기존의 UW-DDPM 프레임워크에서 계산적으로 집약적인 U-Net 변환을 대체하여 복잡성을 줄이고 성능을 유지합니다.

- **Technical Details**: UW-DiffPhys는 물리 기반(UIE, Underwater Image Enhancement) 네트워크 구성요소와 노이즈 제거 U-Net을 결합하여, 최근의 UW-DDPM 솔루션을 통해 요구되는 계산 복잡도를 줄이고 있습니다. 또한, 비마르코프(non-Markovian) 샘플링을 통해 추론 과정을 가속화하기 위해 DDIM(Denoising Diffusion Implicit Model)을 사용합니다.

- **Performance Highlights**: 실험 결과, UW-DiffPhys는 UW-DDPM에 비해 상당한 계산 복잡성과 추론 시간을 감소시켰으며, PSNR, SSIM, UCIQE 등의 주요 지표에서 경쟁력 있는 성능을 달성하였고, 전체 수중 이미지 품질(UIQM) 측면에서도 향상을 보였습니다.



### FoodMLLM-JP: Leveraging Multimodal Large Language Models for Japanese Recipe Generation (https://arxiv.org/abs/2409.18459)
Comments:
          14 pages, 5 figures

- **What's New**: 음식 이미지 이해에 관한 연구는 오랜 시간 동안 진행되어 왔고, 최근 Multimodal Large Language Models (MLLMs)의 발전이 이러한 연구에 큰 기여를 할 것으로 기대됩니다.

- **Technical Details**: 이 연구에서는 일본 레시피 데이터셋을 기반으로 LLaVA-1.5와 Phi-3 Vision이라는 개방형 MLLMs를 미세 조정하였으며, 이를 폐쇄형 모델인 GPT-4o와 비교 평가하였습니다. 레시피의 내용, 재료 및 조리 절차를 평가하기 위해 일본 음식 문화를 포괄적으로 반영한 5,000개의 평가 샘플을 사용했습니다.

- **Performance Highlights**: 우리의 모델은 재료 생성에서 F1 점수 0.531을 기록하여, 현재 최첨단 모델인 GPT-4o의 F1 점수 0.481을 초월함으로써 더 높은 정확도를 나타냈습니다. 또한, 조리 절차 텍스트 생성에서는 GPT-4o와 유사한 성능을 보였습니다.



### Enhancing Crime Scene Investigations through Virtual Reality and Deep Learning Techniques (https://arxiv.org/abs/2409.18458)
- **What's New**: 이 논문은 범죄 현장을 가상 현실(VR) 환경에서 검사하기 위한 사진 측량(photogrammetric) 재구성을 제안하며, 딥러닝(DL) 알고리즘을 통한 완전 자동(object recognition) 객체 인식에 중점을 두고 있습니다.

- **Technical Details**: 클라이언트-서버(client-server) 아키텍처를 통해 사전 훈련된 Faster-RCNN 모델을 선택하였으며, 이는 VR 환경에서 전문가가 선정한 관련 객체를 효과적으로 분류할 수 있는 최적의 방법으로 평가됩니다.

- **Performance Highlights**: 실제 범죄 현장을 시뮬레이션한 실험 결과, 제안된 방법이 잠재적인 증거 가치를 가진 객체를 효과적으로 찾고 인식할 수 있음을 보여주었으며, 특히 건강 및 안전 위험이 있는 범죄 현장(화재, 폭발, 화학물질 등)의 신속한 분석을 가능하게 하였습니다. 이를 통해 주관적 편견(subjective bias)과 현장 오염을 최소화할 수 있습니다.



### DynaWeightPnP: Toward global real-time 3D-2D solver in PnP without correspondences (https://arxiv.org/abs/2409.18457)
- **What's New**: 본 논문은 실시간으로 3D와 2D 형태를 정렬하기 위한 최적의 pose 추정을 주요 다뤘습니다. 특히, correspondences(상응)이 없는 상태에서의 PnP 문제를 다루며, 복잡한 3D와 2D 형태의 등록 문제를 견고하게 처리하는 방법론을 제공합니다.

- **Technical Details**: Reproducing Kernel Hilbert Space (RKHS)를 이용하여 'big-to-small' 문제를 해결하기 위해 iterative reweighted least squares 방법을 적용합니다. 이 연구는 correspondence-free PnP에서 발생하는 회전(rotation)과 변환(translation) 간의 숫자적 모호성 문제를 구체적으로 다루며, 동적 가중치(DynaWeight) 서브 문제를 도입하여 pose 추정과 정렬 정확도를 향상시키는 알고리즘을 제안합니다.

- **Performance Highlights**: 제안된 DynaWeightPnP 알고리즘은 Endovascular Image-Guided Interventions (EIGIs)에서 3D-2D 혈관 중심선 등록 작업에 대한 실험을 통해, 현대 단일 코어 CPU에서 비슷한 방식으로 60 Hz(후처리 없음) 및 31 Hz(후처리 있음)의 등록 처리 속도를 기록했습니다. 이 결과는 기존 방법과 비교했을 때 경쟁력 있는 정확도를 나타내며, 향후 로봇 내비게이션 작업에 적합함을 강조합니다.



### Search3D: Hierarchical Open-Vocabulary 3D Segmentation (https://arxiv.org/abs/2409.18431)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문에서는 Search3D라는 새로운 접근법을 소개합니다. 이 방법은 계층적(open-vocabulary) 3D 장면 표현을 구축하여 다양한 세분화 수준에서 엔티티 검색을 가능하게 합니다.

- **Technical Details**: Search3D는 객체의 세부 파트, 전체 객체, 재료와 같은 속성으로 설명된 영역 등 다양한 레벨의 엔티티 검색을 지원합니다. 또한, MultiScan을 기반으로 하는 장면 규모의 open-vocabulary 3D 파트 분할 벤치마크를 제공하며, ScanNet++에 대해 open-vocabulary 세분화 주석을 추가합니다.

- **Performance Highlights**: Search3D는 여러 작업에서 효과성을 입증하였으며, 기존 방식들과 비교할 때 장면 규모 open-vocabulary 3D 파트 분할에서 우수한 성능을 보임과 동시에 3D 객체와 재료의 세분화에서도 강력한 성능을 유지합니다.



### Robust Network Learning via Inverse Scale Variational Sparsification (https://arxiv.org/abs/2409.18419)
Comments:
          21 pages, 7 figures

- **What's New**: 이 논문에서는 다양한 종류의 노이즈에 대한 저항력을 키우기 위해 새로운 접근법을 제안하고 있습니다.

- **Technical Details**: 본 연구는 시간 연속적(inverse scale space formulation) 인버스 스케일 변동 희소화(framework) 프레임워크를 도입하여 픽셀 간 변동 차이를 판단함으로써 점진적으로 세부적인 대규모 특징을 학습합니다. 이를 통해 작은 스케일 특징을 부드럽게 하여 노이즈를 제거하고, 질감(textures) 및 물체 윤곽(object contours)과 같은 고대비(high-contrast) 세부 사항을 유지합니다.

- **Performance Highlights**: 제안된 방법은 다양한 노이즈 유형에 대한 강력한 저항력을 보여주며, 기존의 주파수 기반(frequency-based) 방법들과 비교 시 간단하고 효율적인 구현을 제공합니다.



### Query matching for spatio-temporal action detection with query-based object detector (https://arxiv.org/abs/2409.18408)
- **What's New**: 이 논문에서는 영상에서 시간적 일관성을 유지해야 하는 스페이쇼-템포럴(spatio-temporal) 액션 감지(action detection) 모델에 DETR(query-based object detection 모델)을 확장하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 각 프레임에 DETR을 적용하고, 특징 이동(feature shift)을 이용하여 시간 정보를 통합합니다. 하지만 DETR의 각 프레임에서의 객체 쿼리가 서로 다른 객체와 대응될 수 있어 단순한 특징 이동은 비효율적일 수 있습니다. 이를 해결하기 위해, 우리는 서로 다른 프레임 간의 쿼리 매칭(query matching)을 제안하여 동일한 객체에 대한 쿼리가 매칭되도록 하고, 특징 이동에 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 쿼리 매칭을 사용하여 쿼리 특징을 이동시킬 때 JHMDB21 데이터셋에서 성능이 상당히 향상되었음을 보여줍니다.



### GenesisTex2: Stable, Consistent and High-Quality Text-to-Texture Generation (https://arxiv.org/abs/2409.18401)
- **What's New**: 이 논문에서는 대규모 텍스트 기반 이미지 확산 모델을 활용한 새로운 텍스처 합성 방법을 제안합니다. 텍스트에서 텍스처로의 변환을 위한 프레임워크를 소개하여 3D 기하학을 위한 텍스처 생성의 도전을 극복하고자 합니다.

- **Technical Details**: 우리의 접근법은 사전 훈련된 diffusion models를 기반으로 하며, self-attention 레이어에서 지역적 주의 재조정 메커니즘(local attention reweighing mechanism)을 도입하여 서로 다른 시점(viewpoint) 간에 공간적 연관성이 있는 패치에 집중하도록 모델을 유도합니다. 또한, 새로운 잠재 공간 병합 파이프라인(latent space merge pipeline)을 제안하여 다양성을 유지하면서도 서로 다른 시점 간의 일관성을 보장합니다.

- **Performance Highlights**: 이 방법은 기존의 최첨단 기술에 비해 텍스처 일관성(texture consistency)과 시각적 품질(visual quality)에서 상당히 우수한 성능을 보였으며, 증류 기반(methods based on distillation) 방법들보다 훨씬 빠른 결과를 제공했습니다. 추가적인 훈련이나 미세 조정(fine-tuning)이 필요 없어, 공공 플랫폼에서 사용할 수 있는 다양한 모델에 쉽게 적용할 수 있습니다.



### You Only Speak Once to S (https://arxiv.org/abs/2409.18372)
Comments:
          7 pages, 4 figures, submitted to ICASSP 2025

- **What's New**: 새로운 접근 방식 YOSS(You Only Speak Once to See)를 소개하여 오디오를 활용하여 시각적 장면 내 객체를 기초하는 방법(오디오 기초화)을 제안합니다.

- **Technical Details**: 사전 훈련된 오디오 모델과 시각 모델을 대조 학습(contrastive learning) 및 다중 모달 정렬(multi-modal alignment)을 통해 통합합니다. 이 과정에서 음성 명령 또는 설명을 캡처하고 이를 이미지 내의 해당 객체에 직접 매핑합니다.

- **Performance Highlights**: 실험 결과에 따르면, 오디오 가이드를 객체 기초화에 효과적으로 적용할 수 있으며, 현재의 객체 기초화 방법의 정밀성과 견고성을 향상시킬 수 있는 가능성을 제시합니다. 이는 로봇 시스템 및 컴퓨터 비전 애플리케이션의 성능 개선에 기여할 수 있습니다.



### Multi-hypotheses Conditioned Point Cloud Diffusion for 3D Human Reconstruction from Occluded Images (https://arxiv.org/abs/2409.18364)
Comments:
          17 pages, 7 figures, accepted NeurIPS 2024

- **What's New**: 본 논문에서는 심각한 가림(occlusion) 상황에서도 3D 인간 형태 재구성을 위한 새로운 파이프라인, MHCDIFF를 제안합니다. 이 방법은 픽셀 정렬된 세부 3D 인간 재구성을 위해 점 구름(diffusion) 모델을 도입합니다.

- **Technical Details**: MHCDIFF는 단일 RGB 이미지에서 기하학적 세부 사항을 캡처하기 위해 다수의 가설화된 SMPL(-X) 메쉬로부터 로컬 특징(local features)을 추출하고 이 특징 집합을 활용하여 확산 모델(diffusion model)을 조정합니다. 점 구름 확산(point cloud diffusion) 모델은 누락된(osculaded) 영역을 생성하기 위해 전체적인 일관성(global consistent features)을 캡처하며, 잡음 제거(denoising) 과정에서 잘못 정렬된 SMPL 메쉬를 수정합니다.

- **Performance Highlights**: CAPE 및 MultiHuman 데이터셋에서의 실험 결과, 제안된 방법은 SMPL, 암묵적 함수(implicit functions), 점 구름 확산(point cloud diffusion) 및 이들의 결합 기반의 다양한 최신 기술(SOTA) 방법들과 비교하여 우수한 성능을 보였습니다.



### SinoSynth: A Physics-based Domain Randomization Approach for Generalizable CBCT Image Enhancemen (https://arxiv.org/abs/2409.18355)
Comments:
          MICCAI 2024

- **What's New**: 이번 연구에서는 SinoSynth라는 물리 기반의 열화 모델을 제시하여 다양한 CBCT 특별 아티팩트를 시뮬레이션하고, 고품질 CT 이미지로부터 합성 CBCT 이미지를 생성합니다. 이러한 접근 방식은 사전 정렬된 데이터 없이도 다양한 이미지를 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: SinoSynth는 이미지 전환 방법을 통해 CBCT 아티팩트를 해결하려는 전통적인 접근 방식을 개선합니다. 이 모델은 고정합성 데이터 없이 다양한 CBCT 전용 아티팩트를 생성할 수 있으며, 이를 통해 여러 생성 네트워크가 다양한 다기관 데이터 세트에서 상업용 데이터보다 우수한 성능을 발휘하는 것을 확인하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해, 우리의 합성 데이터에서 훈련된 여러 생성 네트워크가 이질적인 다기관 데이터 세트에서 뛰어난 결과를 달성하며, 실제 데이터에서 훈련된 같은 네트워크보다도 성능이 우수함을 보여주었습니다. 또한, 우리의 열화 모델은 조건부 생성 모델에서 해부학적 제약을 강제하는 수단을 제공하여 고품질 및 구조 보존적인 합성 CT 이미지를 생성할 수 있게 해줍니다.



### Does End-to-End Autonomous Driving Really Need Perception Tasks? (https://arxiv.org/abs/2409.18341)
Comments:
          Technical Report

- **What's New**: 이 논문에서는 16개의 내비게이션 기반 토큰(navigation-guided tokens)만을 활용하여 정밀한 장면 정보를 효율적으로 추출하는 새로운 프레임워크인 SSR(Sparse Scene Representation)을 소개합니다. 이는 기존의 감독 학습(supervised learning) 기반의 방법에서 벗어나, 자율주행 시스템의 실시간 배치를 보다 유연하게 만듭니다.

- **Technical Details**: SSR은 E2E 자율주행(End-to-End Autonomous Driving) 방법론의 한계를 극복하기 위해 설계된 프레임워크로, 감독 학습 작업을 필요로 하지 않고 내비게이션 의도(navigation intent)에 직접적으로 관련된 주요 요소에 자원을 집중할 수 있도록 합니다. 또한, Bird's-Eye View (BEV) 월드 모델을 활용한 시간적 향상 모듈을 도입하여, 예측된 미래 장면과 실제 미래 장면을 자기 감독(self-supervision) 방식을 통해 정렬합니다.

- **Performance Highlights**: SSR은 nuScenes 데이터셋에서 최첨단 계획 성능을 달성하며, L2 오류에서 27.2% 상대 감소, UniAD보다 51.6% 낮은 충돌률을 기록했습니다. 또한, SSR은 10.9배 빠른 추론 속도(inference speed)와 13배 더 빠른 훈련 시간(training time)을 제공합니다.



### DeBaRA: Denoising-Based 3D Room Arrangement Generation (https://arxiv.org/abs/2409.18336)
Comments:
          Accepted at NeurIPS 2024. Preprint version

- **What's New**: 본 논문에서는 DeBaRA라는 새로운 score-based model을 소개합니다. 이 모델은 제약이 있는 환경에서 정밀하고 유연한 배치 생성을 위한 것으로, 3D 공간 인식(3D spatial awareness)을 핵심으로 설계되었습니다.

- **Technical Details**: DeBaRA는 객체의 크기와 위치를 정확히 결정하는 것이 장면 합성 시스템(Scene synthesis system)에서 가장 중요한 요소라고 주장합니다. 이 모델은 경량의 conditional score-based model로 설계되어 있으며, 훈련된 DeBaRA 모델을 통해 다양한 다운스트림 애플리케이션(예: scene synthesis, completion, rearrangement)을 수행할 수 있습니다. 또한, Self Score Evaluation 절차를 도입하여 외부 LLM 모델과 최적의 조합으로 사용할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 기존의 최신 방법들에 비해 여러 시나리오에서 중요한 개선을 보여주었습니다. DeBaRA는 객체의 공간 속성(spatial attributes)에 집중하여 다양한 작업을 효과적으로 수행할 수 있습니다.



### Automated Segmentation and Analysis of Microscopy Images of Laser Powder Bed Fusion Melt Tracks (https://arxiv.org/abs/2409.18326)
Comments:
          21 pages, 10 figures

- **What's New**: 메탈 적층 제조 (metal additive manufacturing) 의 채택이 증가함에 따라, 연구자들과 실무자들은 데이터 기반 접근법(data-driven approaches)으로 인쇄 조건을 최적화하려 하고 있습니다. 이 논문에서는 크로스 섹션 이미지(cross-sectional images)로부터 용융 트랙(melt track)의 차원을 자동으로 식별하고 측정하는 이미지 세분화 신경망(image segmentation neural network)을 제시합니다.

- **Technical Details**: U-Net 아키텍처를 사용하여 서로 다른 연구실, 기계 및 재료에서 수집된 62개의 사전 라벨링된 이미지 데이터 세트(data set)로 훈련합니다. 이미지 증강(image augmentation)과 결합하여, 신경망의 하이퍼파라미터(hyperparameters)인 배치 크기(batch size) 및 학습률(learning rate)을 적절히 조정할 경우, 모델은 분류 정확도(classification accuracy)가 99% 이상, F1 스코어(F1 score)가 90% 이상을 보여줍니다. 신경망은 다양한 사용자에 의해 캡처된 이미지에서도 견고함(robustness)을 보이며, 서로 다른 기계 및 현미경을 사용하여 획득한 이미지에서도 검증되었습니다. 후처리 모듈(post-processing module)은 용융 풀(melt pool)의 높이(height) 및 너비(width)와 젖음 각도(wetting angles)를 추출합니다.

- **Performance Highlights**: 모델 성능 향상을 위한 기회와 다른 적층 제조 공정(directed energy deposition)으로의 전이 학습(transfer learning) 가능성에 대해 논의합니다.



### Harnessing Wavelet Transformations for Generalizable Deepfake Forgery Detection (https://arxiv.org/abs/2409.18301)
- **What's New**: 이번 논문에서는 Deepfake 탐지 방법의 약점을 해결하기 위해 Wavelet-CLIP이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Wavelet 변환(Wavelet Transforms)과 ViT-L/14 아키텍처에서 얻은 특징을 통합하여, 복잡한 Deepfake를 효과적으로 탐지할 수 있도록 설계되었습니다.

- **Technical Details**: Wavelet-CLIP는 Wavelet Transform을 이용하여 이미지의 공간적(spatial) 및 주파수(frequency) 특징을 깊이 분석합니다. 사전 훈련된 CLIP 방법으로 ViT-L/14 아키텍처의 특징들을 활용하여 높은 효율성을 발휘합니다.

- **Performance Highlights**: 테스트 결과, Wavelet-CLIP는 교차 데이터에 대한 일반화(cross-dataset generalization)에서 평균 AUC 0.749, 보이지 않는 Deepfake에 대한 강인성(robustness)에서 0.893을 기록하여 기존의 최첨단 방법들을 능가하는 뛰어난 성능을 보여주었습니다.



### SOAR: Self-supervision Optimized UAV Action Recognition with Efficient Object-Aware Pretraining (https://arxiv.org/abs/2409.18300)
- **What's New**: 이번 연구에서는 UAV(무인 항공기)로 촬영된 공중 영상에 대한 새로운 Self-supervised pretraining 알고리즘인 SOAR를 소개합니다. 이전 연구와는 달리, SOAR는 프리트레이닝(pretraining) 과정에서 인간 객체 지식을 통합하여 효율성을 높였습니다.

- **Technical Details**: SOAR는 두 가지 주요 접근 방식을 사용합니다. 첫째, 객체 인식과 관련된 특정 패치를 유지하기 위한 객체 인식 마스킹 전략(object-aware masking strategy)을 제안하였습니다. 둘째, 객체 정보를 활용하여 재구성 손실(reconstruction loss)을 조정하는 객체 인식 손실 함수(object-aware loss function)를 도입했습니다. 이를 통해 불필요한 배경 패치에 대한 편향을 방지할 수 있습니다.

- **Performance Highlights**: SOAR는 vanilla ViT 백본(backbone)을 사용하였으며, NEC-Drone과 UAV-Human 데이터셋에서 각각 9.7% 및 21.4%의 top-1 정확も(accuracy) 증가를 기록하며 기존 UAV 행동 인식(action recognition) 모델을 능가하였습니다. 특히, SOAR는 18.7ms의 비디오 당 추론 속도(inference speed)를 제공하며, 이는 2배에서 5배 더 빠릅니다. 추가로, SOAR는 이전의 Self-supervised learning 방법들과 유사한 정확도를 보여주면서도 87.5% 적은 프리트레이닝 시간과 25% 적은 메모리 사용을 요구합니다.



### Efficient Microscopic Image Instance Segmentation for Food Crystal Quality Contro (https://arxiv.org/abs/2409.18291)
- **What's New**: 본 논문은 제조 과정에서 식품 결정(식품 크리스탈)의 품질 관리를 위한 효율적인 예측 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 기존의 수동 카운팅 방법 대신 오브젝트 디텍션(object detection)에 기반한 효율적인 instance segmentation 방법을 사용하여 식품 결정의 수와 크기 분포를 예측합니다. 식품 결정 분할(segmentation)은 결정의 다양한 형태와 주변의 경질 모사체(hard mimics)로 인해 도전적인 문제입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 분할 방법들과 비교할 때 결정 카운팅 정확도가 유사하면서도 다섯 배 빠른 속도를 보여줍니다. 또한, 실험을 통해 경질 모사체와 식품 결정을 분리하기 위한 객관적인 기준을 정의하여 유사 데이터셋에서 수동 주석(annotation) 작업에 도움이 될 수 있습니다.



### Advancing Object Detection in Transportation with Multimodal Large Language Models (MLLMs): A Comprehensive Review and Empirical Testing (https://arxiv.org/abs/2409.18286)
- **What's New**: 이번 연구는 교통 시스템에서 객체 탐지(object detection)에 대한 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)과 대형 비전 모델(Large Vision Models, VLMs)의 응용을 포괄적으로 검토하고 실증적으로 평가하는 것을 목표로 합니다.

- **Technical Details**: 연구의 첫 번째 부분에서는 MLLMs의 교통 응용 분야에서의 잠재적인 이점에 대한 배경을 제공하고, 기존 연구에서 현재 MLLM 기술에 대한 포괄적인 리뷰를 실시했습니다. 두 번째 부분에서는 교통 응용 프로그램에서의 엔드 투 엔드 객체 탐지(taxonomy of end-to-end object detection) 개요와 향후 방향을 제시했습니다.

- **Performance Highlights**: MLLM 성능에 대한 상세한 평가 결과를 제공하며, 도로 안전 특성 추출, 안전 비상 이벤트 탐지, 열화상 이미지의 시각적 추론을 포함한 세 가지 실제 교통 문제에 대한 실증 분석을 수행했습니다. 이 연구는 MLLM의 강점과 개선이 필요한 영역을 밝혀냈으며, 교통에서의 객체 탐지를 향상시키기 위한 MLLM의 실용적인 한계와 도전에 대해 논의합니다.



### Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation (https://arxiv.org/abs/2409.18261)
Comments:
          ECCV 2024 (poster). Github page: this https URL

- **What's New**: Omni6D라는 새로운 RGBD 데이터셋을 소개하며, 다양한 범주와 배경을 포함하여 6D 객체 자세 추정의 현실적인 맥락을 제공한다.

- **Technical Details**: Omni6D 데이터셋은 166개의 범주, 4688개의 인스턴스, 80만 개 이상의 캡처로 구성되어 있어 기존 데이터셋보다 훨씬 넓은 평가 범위를 제공한다. 우리는 대칭성 인식(symmetry-aware) 메트릭을 도입하고 기존 알고리즘에 대한 체계적인 벤치마크를 수행하여 새로운 도전과 통찰력을 탐구한다. 또한, 기존 데이터셋에서 모델을 조정(adapt)할 수 있는 효과적인 파인 튜닝(fine-tuning) 접근법도 제안한다.

- **Performance Highlights**: 이 연구는 산업 및 학계 모두에서 6D 자세 추정의 경계를 확장하고 새로운 통찰력과 상당한 진행을 이룰 수 있는 기반을 마련할 것으로 기대된다.



### PCEvE: Part Contribution Evaluation Based Model Explanation for Human Figure Drawing Assessment and Beyond (https://arxiv.org/abs/2409.18260)
- **What's New**: 본 연구에서는 인간 형상 드로잉(Human Figure Drawing, HFD) 평가 작업에서 모델 결정의 명확성과 설명 가능성을 높이기 위해 부분 기여 평가 기반 모델 설명(Part Contribution Evaluation based Model Explanation, PCEvE) 프레임워크를 제안합니다.

- **Technical Details**: PCEvE는 각 개별 부분의 Shapley Value를 측정하여 모델 결정에 대한 기여도를 평가합니다. 기존의 pixel-level attribution 기반 설명 가능한 AI 설명 방식과 달리, PCEvE는 부분 기여 히스토그램(part contribution histogram)이라는 직관적인 설명을 제공합니다. 또한, PCEvE는 설명의 범위를 샘플 수준(sample-level)에서 클래스 수준(class-level) 및 작업 수준(task-level)으로 확장합니다.

- **Performance Highlights**: PCEvE는 여러 HFD 평가 데이터셋에 대한 광범위한 실험을 통해 엄격하게 검증되었으며, 제안된 방법의 타당성을 확인하기 위한 제어된 실험을 수행했습니다. 또한, PCEvE는 스탠포드 자동차와 같은 포토리얼리스틱(photo-realistic) 데이터셋에도 적용하여 다양성과 응용 가능성을 입증했습니다.



### Amodal Instance Segmentation with Diffusion Shape Prior Estimation (https://arxiv.org/abs/2409.18256)
Comments:
          Accepted at ACCV2024

- **What's New**: 새로운 연구에서는 Amodal Instance Segmentation (AIS)를 위한 AISDiff 모델을 제안합니다. 이 모델은 기존의 방법들이 갖고 있는 한계인 과적합(overfitting) 문제를 해결하고, 객체의 카테고리 세부정보를 고려합니다.

- **Technical Details**: AISDiff는 Diffusion Shape Prior Estimation (DiffSP) 모듈을 통해 가시적인 세분화 마스크와 객체의 카테고리를 예측합니다. 이 과정에서는 occluding masks를 예측하여 occlusion-aware 프로세싱을 수행합니다. DiffSP 모듈은 방대한 데이터셋에서 사전 학습된 conditioned diffusion models을 활용해 시각적 특징을 추출하여 shape prior를 추정합니다.

- **Performance Highlights**: 다양한 AIS 벤치마크에서의 실험을 통해 AISDiff의 효과가 입증되었습니다. 이는 기존 방법들에 비해 성능이 향상되었음을 나타냅니다.



### Spatial Visibility and Temporal Dynamics: Revolutionizing Field of View Prediction in Adaptive Point Cloud Video Streaming (https://arxiv.org/abs/2409.18236)
- **What's New**: 이번 논문에서는 점 구름 비디오(PCV)의 Field-of-View (FoV) 적응 스트리밍을 통해 대역폭 요구 사항을 크게 줄이는 방법을 제안합니다. 기존의 방법들은 일반적으로 6 자유도(6DoF) FoV 예측에 초점을 맞추지만, 본 연구는 셀 가시성(cell visibility) 관점에서 FoV 예측 문제를 재구성하였습니다.

- **Technical Details**: 우리는 역사적 3D 가시성 데이터를 활용하고, 공간 인식(spatial perception), 이웃 셀 상관관계(neighboring cell correlation), 가리기 정보(occlusion information)를 통합하는 새로운 공간 가시성(spatial visibility) 및 객체 인식(object-aware) 그래프 모델을 개발했습니다. 이 모델은 예측된 가시성 분포를 기반으로 3D 데이터 전송 결정을 정확하게 합니다.

- **Performance Highlights**: 우리 모델은 장기 셀 가시성 예측(long-term cell visibility prediction)을 크게 향상시켰으며, 최첨단 모델들에 비해 예측 MSE 손실을 최대 50% 줄였습니다. 동시에 100만 개 이상의 포인트를 가진 점 구름 비디오에 대해 30fps 이상의 실시간 성능을 유지합니다.



### Visual Concept Networks: A Graph-Based Approach to Detecting Anomalous Data in Deep Neural Networks (https://arxiv.org/abs/2409.18235)
- **What's New**: 이 논문에서는 깊은 신경망(Deep Neural Networks, DNN)이 비정상적(anomalous)이고 분포 밖(out-of-distribution, OOD) 데이터에 대해 강화되는 새로운 방법을 소개합니다. 기존 OOD 벤치마크는 단일 객체 작업에 초점을 맞추어 복잡한 현실 세계의 비정상성을 충분히 반영하지 못했습니다.

- **Technical Details**: 제안된 방법은 그래프(graph) 구조와 위상적(topological) 특징을 활용하여 원거리 OOD와 근거리 OOD 데이터를 효과적으로 탐지합니다. 이미지를 상호 연결된 이해 가능한 특징 또는 시각 개념의 네트워크로 변환하여 처리합니다.

- **Performance Highlights**: 대규모 어휘와 다양한 작업을 포함한 두 가지 새로운 작업에서의 광범위한 테스트를 통해 이 방법의 효과를 입증했습니다. 이 접근법은 DNN의 OOD 데이터에 대한 회복력(resilience)을 향상시키며 다양한 애플리케이션에서 성능 개선을 약속합니다.



### Analysis of Spatial augmentation in Self-supervised models in the purview of training and test distributions (https://arxiv.org/abs/2409.18228)
Comments:
          Accepted in ECCV 2024 Workshop on Out-of-distribution generalization in computer vision (OOD-CV)

- **What's New**: 이 논문에서는 self-supervised representation learning 방법에 사용되는 공간적 데이터 증강 기법인 random crop과 cutout에 대한 실증 연구를 제시합니다.

- **Technical Details**: 우리는 random cropping을 overlap과 patch로 구분하여, 두 augmentations의 overlapping 면적과 patch 크기가 downstream task의 정확도에 미치는 영향을 세밀하게 분석했습니다. 또한, cutout augmentation이 좋은 representation을 학습하지 못하는 이유에 대한 통찰도 제공합니다. 마지막으로, scene-centric 이미지에서 두 공간 뷰 사이의 픽셀 거리와 비례하는 margin을 적용하여 객체 중심 분포의 downstream task를 위한 scene-centric representation 학습에 대한 distance-based margin을 제안합니다.

- **Performance Highlights**: 단순한 margin 설정이 learned representation을 향상시킬 수 있음을 보여주며, 이러한 연구는 training augmentations와 test distribution 간의 domain-gap에 대한 이해를 발전시킵니다.



### Evaluation of Security of ML-based Watermarking: Copy and Removal Attacks (https://arxiv.org/abs/2409.18211)
- **What's New**: 이번 논문은 AI 기반 미디어의 저작권 보호 및 데이터 출처 검증을 위한 디지털 워터마킹의 안전성을 평가합니다. 특히 파운데이션 모델 기반의 워터마킹 시스템에 초점을 맞추고, 적대적 공격(adversarial attacks)에 대한 보안을 탐구합니다.

- **Technical Details**: 이 논문은 세 가지 세대의 디지털 워터마킹 시스템을 다룹니다: 수공예 방법(handcrafted methods), 오토인코더 기반(autoencoder-based) 방식, 파운데이션 모델 기반(foundational model) 방법. 저자들은 적대적 임베딩(adversarial embedding) 기술을 활용하여 잠재 공간(latent space)에서 워터마킹의 보안을 평가하는 실험을 수행하였습니다.

- **Performance Highlights**: 실험 결과, 복사(copy) 및 제거(removal) 공격 하에서의 다양한 보안 차원을 조사하였으며, 이 시스템들이 직면한 취약점(vulnerabilities)에 대한 경험적 통찰을 제공합니다. 모든 실험 코드와 결과는 제공된 URL에서 확인할 수 있습니다.



### SSP-RACL: Classification of Noisy Fundus Images with Self-Supervised Pretraining and Robust Adaptive Credal Loss (https://arxiv.org/abs/2409.18147)
Comments:
          IEEE BioCAS 2024

- **What's New**: 새롭게 제안된 방법론, Self-Supervised Pre-training with Robust Adaptive Credal Loss (SSP-RACL)을 통해 fundus 이미지 데이터셋에서 label noise 문제를 해결하는 내용이다.

- **Technical Details**: Masked Autoencoders (MAE)을 활용한 사전 학습(pre-training)과 RACL을 이용해 신뢰도 기준 및 적응형 레이블 이완(parameter) 설정으로 가능성 분포(possibility distributions)를 구축하는 방식이다. 이러한 접근법은 메모리 효과(memoration effect)를 억제하여 더욱 신뢰할 수 있는 ground-truth 추정치를 제공한다.

- **Performance Highlights**: 실험 결과, 제안한 SSP-RACL이 label noise를 처리하는 기존 방법들보다 뛰어난 성능을 보이는 것으로 나타났다.



### UniEmoX: Cross-modal Semantic-Guided Large-Scale Pretraining for Universal Scene Emotion Perception (https://arxiv.org/abs/2409.18877)
Comments:
          Submitted to TIP

- **What's New**: 이번 연구에서는 감정 인식을 위한 새로운 대규모 사전 학습 프레임워크인 UniEmoX를 소개합니다. 이는 심리학 연구에서 개인과 환경 간의 상호작용이 감정 탐색 과정과 분리될 수 없다는 점에 영감을 받았습니다.

- **Technical Details**: UniEmoX는 심장 중심(scene-centric) 및 개인 중심(person-centric) 저수준 이미지 공간 구조 정보를 통합하여 더 미세하고 구분 가능한 감정 표현을 도출하는 것을 목표로 합니다. 또한, CLIP 모델을 활용하여 이미지-텍스트 샘플 간의 유사성을 통해 감정 임베딩 표현을 효과적으로 향상시키는 방식으로 동작합니다.

- **Performance Highlights**: 총 6개의 벤치마크 데이터 세트와 2개의 하위 작업에서 실시한 실험을 통해 UniEmoX의 효과성을 검증하였으며, 다양한 시나리오에서 감정 분석을 위한 기존 심리학 이론과 현대적인 대비 학습(contrastive learning) 및 마스크 이미지 모델링 기법을 통합한 최초의 대규모 사전 학습 프레임워크입니다.



### Simulating Dynamic Tumor Contrast Enhancement in Breast MRI using Conditional Generative Adversarial Networks (https://arxiv.org/abs/2409.18872)
- **What's New**: 이번 논문은 유방 MRI에서 가상 대비 증강(virtual contrast enhancement) 방법을 제시하며, 기존의 대비 물질 기반 DCE-MRI(Dynamic Contrast-Enhanced MRI) 방식에 대한 비침습적(alternative) 대안을 제공합니다.

- **Technical Details**: 우리는 조건부 생성적 적대 신경망(conditional generative adversarial network, GAN)을 사용하여 대비가 없는 MRI에서 DCE-MRI 이미지를 예측하고, 여러 대응 DCE-MRI 시간 점의 순차적으로 생성된 이미지를 포함합니다. 이를 통해 종양의 위치(Localization) 파악과 특성 분석(Characterization)이 가능해지며, 관련된 건강 위험을 피할 수 있습니다.

- **Performance Highlights**: 우리는 합성된 DCE-MRI 이미지의 질적(qualitatively) 및 정량적(quantitatively) 평가를 수행하고, 종양 분할(tumor segmentation) 하위 작업에서의 유용성을 평가하기 위해 다중 메트릭(SAMe)을 제안합니다. 이 접근법은 현실적(realistic)이고 유용한 DCE-MRI 시퀀스를 생성하는 데 있어 유망한 결과를 보여주며, 특히 대비 물질 투여가 금기인 환자의 유방암 진단 및 치료 향상에 대한 가능성을 강조합니다.



### Positional Encoder Graph Quantile Neural Networks for Geographic Data (https://arxiv.org/abs/2409.18865)
Comments:
          17 main text pages, 4 figures

- **What's New**: 이번 논문에서는 Positional Encoder Graph Neural Networks (PE-GNNs)의 한계를 극복하기 위해 Positional Encoder Graph Quantile Neural Network (PE-GQNN)를 소개합니다. 이 방법은 PE-GNNs와 Quantile Neural Networks, 다시 조정 기술을 결합하여 예측 분포에 대한 최소한의 가정을 가지고 완전히 비모수적인 프레임워크를 제공합니다.

- **Technical Details**: PE-GQNN의 새로운 네트워크 아키텍처를 제안하며, quantile 기반 손실 함수를 결합하여 계산 복잡성을 증가시키지 않고도 정확하고 신뢰할 수 있는 확률적 모델을 생성합니다. 또한 KNN 예측기를 모델에 통합할 수 있는 구조화된 방법을 소개하며, GNN 레이어 연산을 통해 데이터 누수를 방지합니다.

- **Performance Highlights**: 벤치마크 데이터셋에 대한 실험 결과, PE-GQNN은 예측 정확도와 불확실성 정량화 모두에서 기존의 최고 수준의 방법들보다 상당한 개선을 나타냅니다.



### Early diagnosis of Alzheimer's disease from MRI images with deep learning mod (https://arxiv.org/abs/2409.18814)
Comments:
          7 pages, 3 figures, Presented at the 20-th CSI International Symposium on Artificial Intelligence and Signal Processing (AISP) 21-22 February, 2024, Mazandaran University of Science and Technology, Babol, Iran

- **What's New**: 이 연구에서는 알츠하이머병(Alzheimer's disease, AD) 진단을 위한 새로운 방법으로 합성 소수 샘플 오버샘플링 기술(SMOTE)과 사전 훈련된 CNN(Convolutional Neural Network)을 활용한 뇌 MRI(Magnetic Resonance Imaging) 분석 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 환자의 의료 기록, 신경심리학적 검사, MRI를 포함한 다양한 접근 방식을 통해 알츠하이머병의 특징을 식별합니다. 특히, Kaggle에서 얻은 이미지 데이터셋의 클래스 불균형 문제를 SMOTE를 통해 해결하였고, DEMNET라는 알츠하이머 네트워크에 사전 훈련된 CNN을 적용하여 중요한 특징들을 추출했습니다.

- **Performance Highlights**: 제안된 모델은 98.67%의 높은 정확도로 알츠하이머병 이미지를 분류하는 성능을 달성하였습니다.



### Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs (https://arxiv.org/abs/2409.18794)
- **What's New**: 이번 연구에서는 Open-Nav라는 새로운 접근 방식을 제안하여, 오픈 소스 LLMs(large language models)의 활용을 통해 VLN(task) 문제를 제로샷(zero-shot)으로 해결하고자 하였다.

- **Technical Details**: Open-Nav는 공간-시간적 사고 연쇄(chain-of-thought) 방식으로 VLN 작업을 수행하며, 이를 통해 작업을 이해하고, 진행 상황을 추정하며, 결정을 내리는 단계로 나눈다. 또한, 세밀한 객체(object) 및 공간(spatial) 지식을 활용하여 장면 인식을 개선하고 LLM의 내비게이션(navigation) 추론 능력을 향상시킨다.

- **Performance Highlights**: 모의 환경과 실제 환경에서의 광범위한 실험을 통해 Open-Nav는 폐쇄형 LLM을 사용한 경우와 비교하여 경쟁력 있는 성능을 달성하였다.



### Excavating in the Wild: The GOOSE-Ex Dataset for Semantic Segmentation (https://arxiv.org/abs/2409.18788)
Comments:
          Submitted to IEEE for review

- **What's New**: 이번 연구에서는 독일의 야외 및 오프로드 데이터셋(GOOSE) 프레임워크의 일반화 가능성을 다룹니다. 특히 새로운 GOOSE-Ex 데이터셋을 오픈소스하여 5000개의 추가 멀티모달 프레임을 제공합니다.

- **Technical Details**: GOOSE-Ex 데이터셋은 다양한 완전히 다른 환경에서 기록된 라벨이 있는 멀티모달 프레임을 포함하며, 로봇 굴착기와 4족 동물을 사용합니다. 이 연구에서는 미지의 환경에서 다양한 플랫폼과 센서 모달리티에 대한 의미적 분할(semantic segmentation) 성능을 종합적으로 분석합니다.

- **Performance Highlights**: 결합된 데이터셋은 오프로드 내비게이션(offroad navigation), 객체 조작(object manipulation), 장면 완성(scene completion)과 같은 다양한 다운스트림 애플리케이션이나 대회에 활용될 수 있음을 보여줍니다.



### DualDn: Dual-domain Denoising via Differentiable ISP (https://arxiv.org/abs/2409.18783)
Comments:
          Accepted at ECCV 2024, Project page: this https URL

- **What's New**: 이 논문에서는 이미지 노이즈 제거를 위한 새로운 방법론인 DualDn을 제안합니다. 기존의 단일 도메인(domaian) 노이즈 제거 방식과 달리 DualDn은 원시(raw) 도메인과 sRGB 도메인 각각에 대해 독립적인 두 개의 노이즈 제거 네트워크를 구성합니다.

- **Technical Details**: DualDn은 원시 도메인에서 센서 특이 노이즈 및 공간적으로 변하는 노이즈 수준에 적응하고, sRGB 도메인에서는 ISP(이미지 신호 처리)의 변동성에 적응하여 ISP에 의해 증폭된 잔여 노이즈를 제거합니다. 두 네트워크는 미분 가능(differentiable)한 ISP와 연결되어 end-to-end 방식으로 훈련됩니다.

- **Performance Highlights**: DualDn은 다양한 새로운 노이즈, ISP 파라미터, 그리고 신규 ISP 파이프라인에 적응할 수 있어 일반화(generalizability)에서 뛰어난 성능을 기록합니다. 실험 결과, DualDn은 최신 성능(state-of-the-art performance)을 달성했으며, 실제 카메라에서 재훈련 없이도 플러그 앤 플레이(plug-and-play) 노이즈 제거 모듈로 사용할 수 있습니다.



### A Generalized Tensor Formulation for Hyperspectral Image Super-Resolution Under General Spatial Blurring (https://arxiv.org/abs/2409.18731)
- **What's New**: 본 논문에서는 하이퍼스펙트럴 이미지의 슈퍼 해상도를 달성하기 위한 새로운 텐서 기반 접근법을 제안합니다. 이 방법은 일반화된 텐서 형식을 사용하여 임의의 공간 저하 매트릭스를 처리할 수 있도록 합니다.

- **Technical Details**: 제안된 방법은 Kronecker 분해(Kronecker decomposition)를 기반으로 한 일반화된 텐서 공식화(generalized tensor formulation)를 사용합니다. 또한, 독립적인 수평 및 수직 블러링(horizontal and vertical blurring)으로 가정되지 않은 비분리(anistropic) 블러링을 모델링할 수 있는 조건을 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 일반화된 텐서 접근법이 전통적인 매트릭스 기반 기법뿐만 아니라 최신 텐서 기반 방법들과 비교하여 뛰어난 성능을 보임을 확인했습니다. 특히 비분리 공간 블러링(anisotropic spatial blurring) 경우에 큰 성능 향상이 있었습니다.



### Effectiveness of learning-based image codecs on fingerprint storag (https://arxiv.org/abs/2409.18730)
Comments:
          Accepted ad Wifs 2024

- **What's New**: 이 연구는 학습 기반 이미지 코덱이 지문 이미지 저장에 어떤 영향을 미치는지를 최초로 조사한 것으로, 학습 기반 압축 기술이 생체 데이터 저장 분야에 적용 가능성을 강조합니다.

- **Technical Details**: 연구에서는 JPEG-AI와 같은 학습 기반 이미지 코딩 표준의 효과를 살펴보며, 압축 아티팩트가 지문 특징과 랜드마크(landmarks) 추출에 미치는 영향을 분석합니다. 지문 이미지의 특성과 일반적인 컬러 이미지의 특성이 다르기 때문에 기존 모델이 이러한 이미지에서 어떻게 작동하는지에 대한 분석이 필수적입니다.

- **Performance Highlights**: 실험 결과, 고정 비율 포인트에서 학습된 솔루션은 JPEG2000과 같은 기존 지문 코딩 표준보다 왜곡(distortion)과 minutiae 보존 측면에서 뛰어난 성능을 보였습니다. 특히, 지문 인식 자동화에는 영향을 미치지 않으며, 인간의 시각적 검사를 위한 이미지 품질을 향상시킵니다.



### Multi-modal Medical Image Fusion For Non-Small Cell Lung Cancer Classification (https://arxiv.org/abs/2409.18715)
- **What's New**: 본 논문에서는 비소세포 폐암(Non-small cell lung cancer, NSCLC)의 조기 발견과 세분화된 하위 유형 분류를 위한 혁신적인 다중 모드 데이터 통합 방법을 제안합니다. 이는 CT 및 PET 스캔과 임상 건강 기록 및 유전체 데이터를 융합하는 새로운 접근 방식을 포함합니다.

- **Technical Details**: 우리는 MedClip 및 BEiT와 같은 고급 머신러닝(advanced machine learning) 모델을 활용하여, 이미지 특징 추출을 위한 정교한 방법을 개발했습니다. 이러한 다중 모드 분류기 모델은 94.04%의 정확도를 기록하며, 기존의 접근 방식보다 상당한 성능 향상을 보여줍니다.

- **Performance Highlights**: 우리의 연구는 NSCLC 검출 및 분류의 정확성, 정밀도, 재현율 및 F1 점수 등 주요 성능 지표에서 두드러진 개선을 나타냅니다. 이는 NSCLC 진단을 변화시킬 수 있는 잠재력을 가지고 있으며, 더욱 효과적인 치료 계획 및 결과 개선에 기여할 것입니다.



### 3DPX: Single Panoramic X-ray Analysis Guided by 3D Oral Structure Reconstruction (https://arxiv.org/abs/2409.18701)
- **What's New**: 이번 연구에서는 Panoramic X-ray (PX) 영상 분석을 위한 새로운 방법인 3DPX를 제안합니다. 3DPX는 2D-to-3D reconstruction 기법을 활용하여 PX에서 결손된 3D 해부학 정보를 보완합니다.

- **Technical Details**: 3DPX는 (i) 새로운 점진적 재구성 네트워크(progressive reconstruction network)를 통해 2D-to-3D 재구성을 개선하고, (ii) 3D 유도 2D PX 분류 및 분할 작업을 위한 대비 유도 양방향 다중 모드 정렬 모듈을 포함합니다. 이 네트워크는 여러 피라미드 수준에서 중간 재구성에 대한 지식을 적용하여 3D 이미지를 점진적으로 재구성합니다.

- **Performance Highlights**: 464개의 연구를 포함한 두 개의 구강 데이터 세트에 대한 광범위한 실험 결과, 3DPX는 2D-to-3D 재구성, PX 분류 및 병변 분할을 포함한 다양한 작업에서 최첨단 방법을 능가했습니다.



### Enhanced Convolution Neural Network with Optimized Pooling and Hyperparameter Tuning for Network Intrusion Detection (https://arxiv.org/abs/2409.18642)
Comments:
          7 Pages , 2 figures , 4 Tables , Conference paper

- **What's New**: 이번 연구에서는 네트워크 침입 탐지 시스템(Network Intrusion Detection Systems, NIDS)을 위한 향상된 합성곱 신경망(Enhanced Convolutional Neural Network, EnCNN)을 제안합니다. 이 방법론은 KDDCUP'99 데이터셋을 사용하여 성능을 평가하고, 기존의 방법에 비해 10% 이상의 정확도 향상을 보여주었습니다.

- **Technical Details**: 연구에서는 데이터 전처리(data preprocessing), 탐색적 데이터 분석(exploratory data analysis, EDA), 기능 공학(feature engineering)을 포함하는 포괄적인 방법론을 사용하였습니다. EnCNN의 성능을 로지스틱 회귀(Logistic Regression), 결정 트리(Decision Trees), 서포트 벡터 머신(Support Vector Machines, SVM), 랜덤 포레스트(Random Forest), 아다부스트(AdaBoost), 투표 앙상블(Voting Ensemble) 등 다양한 기계 학습 알고리즘과 비교했습니다.

- **Performance Highlights**: EnCNN은 기존 최첨단 접근 방식에 비해 10%의 정확도 향상을 이루었습니다. 이는 실시간 네트워크 침입 탐지에서 EnCNN의 효과성을 보여주며, 보안 위협을 식별 및 완화하고 전체 네트워크의 복원력을 강화하는 강력한 솔루션을 제공함을 의미합니다.



### Towards Integrating Epistemic Uncertainty Estimation into the Radiotherapy Workflow (https://arxiv.org/abs/2409.18628)
Comments:
          Keywords: Epistemic Uncertainty - Out-of-Distribution Detection - CT Segmentation - OAR contouring - Radiotherapy

- **What's New**: 본 연구는 방사선 치료 계획에서의 목표 구조물 및 위험 장기(Organ-at-Risk, OAR)의 윤곽을 정확히 지정하는 것의 중요성을 강조하며, 최근 딥러닝(deep learning)의 발전을 통해 OAR 윤곽화 성능을 개선했으나, OOD(out-of-distribution) 시나리오의 신뢰성 문제를 다루고 있다. 특히, epistemic uncertainty(계량적 불확실성) 추정 통합을 통해 OOD 감지를 위한 새로운 방법론을 제시한다.

- **Technical Details**: 연구에서는 OAR 윤곽화 워크플로우에 epistemic uncertainty estimation을 통합하여 임상적으로 관련 있는 시나리오에서 OOD를 감지하는 방법을 제안한다. 구체적으로 설계된 데이터 세트를 사용하여 OOD 감지를 위한 고급 통계적 방법을 도입하고, 예측의 신뢰성이 떨어지는 사례를 식별하는 데 효과적임을 입증한다.

- **Performance Highlights**: 이 연구에서 제안한 접근법은 OOD 감지를 위한 AUC-ROC(Area Under the Curve - Receiver Operating Characteristic)를 0.95로 달성하였으며, 임플란트 사례에 대한 특이도(specificity) 0.95 및 민감도(sensitivity) 0.92를 기록하였다. 이는 모델 예측의 신뢰성을 판단하는 데 있어 전문가의 검토가 필요한 경우를 효과적으로 표시하는 데 기여한다.



### Metasurface-generated large and arbitrary analog convolution kernels for accelerated machine vision (https://arxiv.org/abs/2409.18614)
- **What's New**: 인공지능(AI) 분야의 발전 속에서, 기계 비전(machine vision) 및 의료 진단에서 중요한 역할을 하는 합성곱 신경망(convolutional neural networks)의 디지털 합성곱 층을 광학 메타표면(optical metasurface)으로 대체하는 연구가 이뤄졌습니다.

- **Technical Details**: 광학 메타표면을 이용하여 공간 주파수(domain) 훈련 방법(spatial frequency domain training method)을 개발하였으며, 이를 통해 아날로그(analog) 합성곱 커널(convolution kernels)을 임의의 형태로 생성할 수 있게 되었습니다. 이 방법은 비정형 조명(incoherent illumination) 조건에서 다수의 병렬 합성곱 커널을 생성합니다.

- **Performance Highlights**: MNIST 데이터셋에서 98.59%의 분류 정확도를 달성하였으며, Fashion-MNIST 및 CIFAR-10 데이터셋에서는 각각 92.63%와 68.67%의 정확도를 보였습니다. 이 연구는 아날로그 광학 합성곱의 독특한 이점을 강조하며, 특히 엣지 디바이스(edge devices)에서 기계 비전 작업을 가속화할 수 있는 유망한 경로입니다.



### CodeSCAN: ScreenCast ANalysis for Video Programming Tutorials (https://arxiv.org/abs/2409.18556)
- **What's New**: 이번 논문에서는 코딩 스크린캐스트(video tutorial) 분석을 위한 대규모 데이터셋인 CodeSCAN을 소개합니다. 이는 프로그래밍 교육에서의 비디오 튜토리얼의 검색 문제를 해결하기 위해 개발되었습니다.

- **Technical Details**: CodeSCAN 데이터셋은 Visual Studio Code 환경에서 캡처된 12,000개의 스크린샷으로 구성되어 있으며, 24개의 프로그래밍 언어, 25개의 폰트(font), 90개 이상의 테마(theme), 다양한 레이아웃(layout) 변화 및 현실적인 사용자 상호작용을 포함합니다. 또한, 통합 개발 환경(IDE) 요소 탐지, 흑백 변환(color-to-black-and-white conversion), 광학 문자 인식(OCR)에 대한 세부적인 정량적(quantitative) 및 정성적(qualitative) 평가를 실시합니다.

- **Performance Highlights**: 이 연구의 결과는 코딩 스크린캐스트 분석에 대한 연구 촉진을 기대하며, 데이터셋 및 벤치마크의 소스 코드를 공개하여 연구원들이 이용할 수 있도록 하였습니다.



### Efficient Noise Mitigation for Enhancing Inference Accuracy in DNNs on Mixed-Signal Accelerators (https://arxiv.org/abs/2409.18553)
- **What's New**: 본 논문에서는 아날로그 신경망의 정확도에 미치는 프로세스 유도 및 노화 관련 변동의 영향을 완화하여 신경 모델의 강인성을 향상시키는 프레임워크를 제안합니다.

- **Technical Details**: 변동은 활성화의 정밀도에 영향을 미치는 노이즈로 모델링되며, 사전 훈련된 모델의 선택된 레이어 사이에 삽입된 디노이징 블록(denoising block)을 소개합니다. 디노이징 블록을 훈련하여 다양한 노이즈 수준에 대한 모델의 강인성을 크게 증가 시킬 수 있음을 입증하였습니다. 디노이징 블록 추가로 인한 오버헤드를 최소화하기 위해 최적의 삽입 지점을 식별하는 탐색 알고리즘을 제시하고, 혼합 신호 가속기에 통합할 수 있는 효율적인 디노이징 블록 아키텍처를 제안합니다.

- **Performance Highlights**: DNN 모델을 ImageNet 및 CIFAR-10 데이터셋에서 훈련하여 접근 방식을 평가한 결과, 평균적으로 2.03%의 파라미터 카운트 오버헤드를 수용함으로써 변동으로 인한 정확도 감소가 31.7%에서 1.15%로 줄어드는 것을 보여주었습니다.



### Token Caching for Diffusion Transformer Acceleration (https://arxiv.org/abs/2409.18523)
- **What's New**: Diffusion transformers의 성능은 뛰어나지만, 높은 계산 비용이 동반됩니다. 이에 따라 TokenCache라는 새로운 사후 훈련 가속화 방법을 제안합니다.

- **Technical Details**: TokenCache는 토큰 기반 멀티 블록 아키텍처를 활용하여 추론 단계 간 토큰 간 중복 계산을 줄입니다. 세 가지 주요 질문에 답합니다: (1) 중복성을 없애기 위해 어떤 토큰을 가지칠지, (2) 어떤 블록을 효율적으로 가지칠지, (3) 속도와 품질을 균형 있게 하기 위해 언제 캐싱을 적용할지.

- **Performance Highlights**: 실험 결과, TokenCache는 여러 모델에서 diffusion transformers의 생성 품질과 추론 속도 간의 효과적인 균형을 달성하였습니다.



### Med-IC: Fusing a Single Layer Involution with Convolutions for Enhanced Medical Image Classification and Segmentation (https://arxiv.org/abs/2409.18506)
Comments:
          13 pages, 5 figures, 4 tables, preprint submitted to an Elsevier journal

- **What's New**: 이번 연구에서는 의학 이미지를 처리하는 데 있어, Convolutional Neural Network (CNN) 모델에 Involution 레이어를 단일 추가함으로써 분류 및 세분화 성능이 향상된다는 새로운 접근법을 소개합니다.

- **Technical Details**: Convolution(합성곱) 연산은 의료 이미지를 포함한 다양한 영상에서 시각적 패턴을 추출하는 데 제한적인 성능을 보입니다. Involution(역합성곱) 과정은 이러한 제한을 보완하며, CNN 아키텍처 전에 단일 Involution 레이어를 적용함으로써 성능을 개선할 수 있습니다. 연구는 Involution 레이어가 과도하게 사용될 경우 의료 이미지에 대한 예측 정확도가 저하될 수 있음을 보여줍니다.

- **Performance Highlights**: 단일 Involution 레이어 추가 전략은 대부분의 기존 연구 성과를 초과하는 결과를 낳아, 상당히 적은 수의 가중치 매개변수로도 개선된 성능을 유지할 수 있습니다.



### Towards Diverse Device Heterogeneous Federated Learning via Task Arithmetic Knowledge Integration (https://arxiv.org/abs/2409.18461)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 Fedrated Learning(Federated Learning, FL)의 한계를 극복하기 위해 TAKFL이라는 새로운 KD 기반 프레임워크를 제안합니다. 이 프레임워크는 다양한 이질적인 장치 모델의 지식 전이를 독립적으로 수행하여, 각 장치의 고유한 기여를 보존합니다.

- **Technical Details**: TAKFL은 각 장치 프로토타입의 앙상블에서 지식 전이를 별도의 작업으로 처리하며, 각 장치의 정보를 효과적으로 증류할 수 있도록 설계되었습니다. 또한, KD 기반의 self-regularization 기법을 도입하여 noise와 비지도 앙상블 증류 과정에서 발생하는 문제를 완화시킵니다.

- **Performance Highlights**: TAKFL은 컴퓨터 비전(CV) 및 자연어 처리(NLP) 작업에서 종합적인 평가를 수행하였으며, 다양한 데이터셋과 설정에서 SOTA(State Of The Art) 결과를 달성하여 기존 KD 기반 방법들보다 월등한 성능을 보였습니다.



### Gradient-free Decoder Inversion in Latent Diffusion Models (https://arxiv.org/abs/2409.18442)
Comments:
          19 pages, Accepted to NeurIPS 2024

- **What's New**: 본 논문에서는 Latent Diffusion Models (LDMs)에서 효율적인 gradient-free decoder inversion 방법을 제안합니다. 이는 기존의 gradient 기반 방법이 요구하는 메모리와 시간을 대폭 줄이는 혁신적인 접근 방식입니다.

- **Technical Details**: 이 접근 방식은 다양한 latent 모델에 적용 가능하며, 이론적 수렴 특성을 연구하였습니다. 특히, forward step method와 inertial Krasnoselskii-Mann (KM) iteration을 분석하였으며, 최근 LDMs에 만족되는 완만한 코코어시비티(cocoercivity) 조건 하에 적용됩니다.

- **Performance Highlights**: Adam optimizer와 learning rate scheduling을 활용한 본 방법은 기존의 gradient 기반 방법보다 계산 시간과 메모리 소모를 획기적으로 줄였으며, noise-space watermarking과 같은 다양한 응용에서 효율적인 계산을 가능하게 하였습니다. 이로써 유사한 오류 수준을 유지하면서도 성능이 개선되었습니다.



### A3: Active Adversarial Alignment for Source-Free Domain Adaptation (https://arxiv.org/abs/2409.18418)
Comments:
          Accepted at ICMLA 2024

- **What's New**: 본 논문은 레이블이 없는 대상 도메인으로부터 지식을 전이하기 위해 레이블이 있는 출처 도메인에서 지식을 전이하는 비지도 도메인 적응(Unsupvised Domain Adaptation, UDA) 분야의 최신 동향인 Source-free UDA에 대한 새로운 접근법인 Active Adversarial Alignment (A3)를 제안합니다.

- **Technical Details**: A3는 self-supervised learning, adversarial training, 및 active learning을 결합하여 견고한 Source-free UDA를 가능하게 합니다. 이 프레임워크는 acquisition function을 사용하여 유익하고 다양한 데이터를 능동적으로 샘플링하고 모델을 adversarial losses와 consistency regularization을 통해 적응시킵니다. 이는 출처 데이터에 접근하지 않고도 분포를 정렬합니다.

- **Performance Highlights**: A3는 효과적인 도메인 정렬 및 노이즈 감소를 위해 능동적 및 적대적 학습의 시너지를 활용하여 Source-free UDA를 발전시킵니다.



### MultiClimate: Multimodal Stance Detection on Climate Change Videos (https://arxiv.org/abs/2409.18346)
Comments:
          5 pages, 1 figure

- **What's New**: 이번 연구에서는 기후 변화(Climate Change, CC)와 관련된 주장을 탐지하기 위한 첫 번째 공개 소스 수동 주석 데이터셋인 MultiClimate를 소개합니다. 이 데이터셋은 100개의 CC 관련 YouTube 비디오와 4,209개의 프레임-전사 쌍으로 구성되어 있습니다.

- **Technical Details**: MultiClimate는 다양한 비전(Vision) 및 언어(Language) 모델, 그리고 멀티모달(Multimodal) 모델을 사용하여 주장을 탐지합니다. 연구 결과, 텍스트 전용 BERT 모델이 이미지 전용 ResNet50 및 ViT 모델보다 현저히 우수한 성능을 보였습니다. 두 가지 모달리티를 결합할 경우, 정확도(Accuracy) 및 F1 점수에서 각각 0.747 및 0.749를 기록하며 최신 기술 수준(State-of-the-art)을 달성했습니다.

- **Performance Highlights**: 100M 크기의 융합(fusion) 모델이 CLIP, BLIP, 그리고 훨씬 더 큰 9B 크기의 멀티모달 IDEFICS 및 텍스트 전용 Llama3와 Gemma2보다 뛰어난 성능을 보였습니다. 이는 대형 언어 모델에 대해 멀티모달 주장 탐지가 여전히 도전적임을 나타냅니다.



### DRL-STNet: Unsupervised Domain Adaptation for Cross-modality Medical Image Segmentation via Disentangled Representation Learning (https://arxiv.org/abs/2409.18340)
Comments:
          MICCAI 2024 Challenge, FLARE Challenge, Unsupervised domain adaptation, Organ segmentation, Feature disentanglement, Self-training

- **What's New**: 본 논문에서는 cross-modality(크로스 모달리티) 의료 이미지 분할을 위한 새로운 프레임워크인 DRL-STNet을 제안합니다. 이 방법은 Generative Adversarial Networks(GANs), Disentangled Representation Learning(DRL), Self-Training(ST)를 활용합니다.

- **Technical Details**: DRL-STNet은 GAN 내에서 DRL을 이용하여 이미지를 소스 모달리티에서 타겟 모달리티로 변환합니다. 초기 단계에서는 변환된 이미지와 소스 레이블로 분할 모델을 학습한 후, 합성 이미지와 실제 이미지의 결합을 통해 pseudo-labels(의사 레이블) 및 실제 레이블로 반복적으로 미세 조정합니다.

- **Performance Highlights**: 제안된 DRL-STNet은 FLARE 도전 과제 데이터셋에서 복부 장기 분할에서 11.4% Dice similarity coefficient(다이스 유사도 계수) 및 13.1% Normalized Surface Dice metric(정규화된 표면 다이스 측정치) 향상을 보여 주며, 각각 74.21% 및 80.69%의 점수를 기록했습니다. 평균 실행 시간은 41초이며, GPU 메모리-시간 곡선 아래 면적은 11,292 MB입니다.



### Photon Inhibition for Energy-Efficient Single-Photon Imaging (https://arxiv.org/abs/2409.18337)
Comments:
          Accepted for ECCV 2024. Supplementary material and code available at this https URL

- **What's New**: 최근 등장한 단일광자 카메라(Single-Photon Cameras, SPCs)는 저조도, 높은 동적 범위, 빠른 움직임의 복잡한 이미지 분석에 유용하게 사용되고 있습니다. 하지만, 단일광자 눈금 다이오드(Single-Photon Avalanche Diode, SPAD)를 기반으로 한 SPC의 경우, 각 광자 감지 과정에서 소비되는 에너지가 복잡하다는 문제를 가지고 있습니다. 이 논문에서는 이 문제를 해결하기 위해 "photon inhibition"이라는 새로운 계산적 이미징(computational imaging) 접근 방식을 제안하고 있습니다.

- **Technical Details**: 이 연구에서는 photon inhibition을 통해 실제 SPAD 픽셀을 실시간으로 비활성화하여 가장 유용한 차기 광자를 선택하는 경량(on-sensor) 계산적 억제 정책을 개발했습니다. 이 정책은 다운스트림 추론 작업 목표 및 자원 제약에 따라 공간 및 시간 내에서 감지를 전략적으로 할당합니다.

- **Performance Highlights**: 시뮬레이션과 실제 SPC로 캡처한 데이터를 통해, 이미지 재구성 및 경계 감지 작업에 대한 맞춤형 정책을 설계하여 90
% 이상의 광자 감지 감소를 보였으나, 작업 성능 지표를 유지하는 성능을 시연했습니다. 이러한 결과는 에너지 효율적인 단일광자 이미징의 미래 가능성을 열어줍니다.



### Realistic Evaluation of Model Merging for Compositional Generalization (https://arxiv.org/abs/2409.18314)
- **What's New**: 본 논문에서는 다양한 merging 방법론의 상대적인 장점을 평가하고, 이를 통해 각 방법의 실제 요구 사항을 명확히 하였습니다. 특히, 이미지 분류(image classification), 이미지 생성(image generation), 자연어 처리(natural language processing) 분야에서의 compositional generalization을 위한 merging에 초점을 맞추었습니다.

- **Technical Details**: 연구는 다양한 merging 방법을 일관된 실험 환경에서 비교하였으며, 모델 아키텍처(model architecture), 데이터 가용성(data availability), 계산 예산(computational budget)에 대한 가정이 서로 다를 수 있음을 강조합니다. 또한, 각 merging 방법이 필요로 하는 계산 비용(computational costs)과 병합되는 모델 수가 증가할 때의 성능을 측정하였습니다.

- **Performance Highlights**: 연구 결과는 모델 병합(model merging) 분야의 현재 상태를 명확히 하고, 새로운 방법을 시험할 수 있는 포괄적이고 엄격한 실험 Setup을 제공합니다.



### Flat'n'Fold: A Diverse Multi-Modal Dataset for Garment Perception and Manipulation (https://arxiv.org/abs/2409.18297)
- **What's New**: Flat'n'Fold는 의류 조작을 위한 새로운 대규모 데이터셋으로, 기존 데이터셋의 중요한 격차를 해결합니다. 44개의 고유 의류 항목을 8개 카테고리에 걸쳐 총 1,212개의 인간과 887개의 로봇 시연으로 구성되어 있습니다.

- **Technical Details**: 이 데이터셋은 구겨진 상태에서 접힌 상태까지의 전체 조작 과정을 포착하며, 동기화된 다중 관점 RGB-D 이미지, 포인트 클라우드(point clouds), 그리고 손 또는 그리퍼(gripper) 위치 및 회전을 포함한 행동 데이터를 제공합니다. 이 데이터에 대한 다양성과 복잡성을 기존 기준과 비교하여 정량화하였습니다.

- **Performance Highlights**: Flat'n'Fold의 유용성을 보여주기 위해, 그리핑 포인트(prediction) 예측 및 하위 작업(subtask) 분해에 대한 새로운 기준을 설정했습니다. 이 작업에 대해 최첨단 모델을 평가한 결과, 개선이 필요한 부분이 많음을 확인했습니다. 이는 변형 가능한 객체의 로봇 인식과 조작 분야에서의 발전 가능성을 강조합니다.



### Synthesizing beta-amyloid PET images from T1-weighted Structural MRI: A Preliminary Study (https://arxiv.org/abs/2409.18282)
- **What's New**: 이번 연구에서는 T1-weighted MRI 스캔을 이용하여 3D diffusion models를 활용한 Aβ-PET 이미지를 합성하는 방안을 제시합니다. 이는 기존의 Aβ-PET 이미징의 제한된 사용을 극복하기 위한 시도로, 구조적 MRI를 대안으로 활용합니다.

- **Technical Details**: Aβ-PET 이미지는 알츠하이머병(AD)의 병리학적 특징인 아밀로이드 플라크의 축적을 나타내며, 본 연구에서는 고품질의 Aβ-PET 이미지를 생성하기 위해 3D diffusion models를 사용했습니다. Aβ 침착 패턴의 변동성 때문에 경도 인지 장애(MCI) 환자에 대해서는 효과가 덜하다는 한계를 지녔습니다.

- **Performance Highlights**: 인지가 정상인 경우에는 고품질의 Aβ-PET 이미지가 생성되었으나, MCI 환자에 대해서는 추가적인 데이터(예: MCI 사례의 샘플 수 증대 및 임상적, 인구통계적 정보, 인지 및 기능 평가, 종단적 데이터 포함)가 필요할 것으로 보입니다.



### Task-recency bias strikes back: Adapting covariances in Exemplar-Free Class Incremental Learning (https://arxiv.org/abs/2409.18265)
Comments:
          Accepted for NeurIPS 2024

- **What's New**: 본 논문에서는 과거 데이터에 접근할 수 없는 상황에서 모델을 훈련하는 Exemplar-Free Class Incremental Learning (EFCIL) 문제를 다룹니다. 기존의 최첨단 방법은 클래스들을 나타내기 위해 Gaussian 분포를 사용하지만, 두 가지 주요 문제로 인해 그 효과성이 제한됩니다.

- **Technical Details**: 첫 번째 문제는 각 클래스의 공분산 행렬(covariance matrices)이 변화하고 매 작업(task)마다 적응해야 한다는 점입니다. 두 번째 문제는 훈련 중 발생하는 차원 축소(dimensionality collapse)로 인한 작업 최신성(task-recency) 편향입니다. 본 연구에서는 AdaGauss라는 새로운 방법을 제안하여, 작업 간 공분산 행렬을 적응시키고 추가적인 anti-collapse 손실 함수(loss function)를 통해 작업 최신성 편향을 완화합니다.

- **Performance Highlights**: AdaGauss는 기존의 EFCIL 벤치마크(benchmarks) 및 데이터셋(datasets)에서 최상급 결과를 나타내며, 스크래치부터 훈련할 때나 사전 훈련(pre-trained)된 모델에서 시작할 때 모두 우수한 성과를 보여줍니다.



### Developing a Dual-Stage Vision Transformer Model for Lung Disease Classification (https://arxiv.org/abs/2409.18257)
Comments:
          3 pages, 3 figures, Applied to the IEEE MetroCon 2024 Conference

- **What's New**: 이번 연구에서는 폐 질환 진단을 위해 Vision Transformer (ViT)와 Swin Transformer를 통합한 이중 단계(dual-stage) 비전 트랜스포머 모델을 개발했습니다.

- **Technical Details**: 이 모델은 X-ray 스캔을 기반으로 14종의 폐 질환을 분류하며, 데이터 전처리와 신경망(neural network) 훈련 과정을 거친 후 계산된 정확도는 92.06%에 이릅니다.

- **Performance Highlights**: 제안된 모델은 폐 질환 분류 및 진단에 있어 높은 정확도를 보여주어 유망한 결과를 나타냈습니다.



### PNR: Physics-informed Neural Representation for high-resolution LFM reconstruction (https://arxiv.org/abs/2409.18223)
- **What's New**: 본 논문에서는 고해상도 Light Field Microscopy (LFM) 재구성을 위한 새로운 방법인 PNR (Physics-informed Neural Representation)을 소개합니다. PNR은 기존의 방법들보다 성능을 크게 향상시킵니다.

- **Technical Details**: PNR은 비지도학습(unsupervised) 및 명시적(feature representation) 접근 방식을 활용하여 RLD보다 PSNR에서 6.1 dB의 개선을 달성합니다. 또한, 고주파 정보 회복을 위한 주파수 기반(frequency-based) 훈련 손실(train loss)을 사용하며, 이로 인해 SOTA 방법들에 비해 LPIPS를 최소 절반으로 줄였습니다. PNR은 물리 기반의 변형 보정(aberration correction) 전략을 통합하여 최적화 과정에서 Zernike 다항식 파라미터를 최적화합니다.

- **Performance Highlights**: PNR은 복잡한 생물학적 시나리오에서도 높은 성능을 제공하며, 향후 고해상도 생물학적 이미징 응용에서 유망한 솔루션이 될 것입니다. 코드는 공개될 예정입니다.



### Learning to Drive via Asymmetric Self-Play (https://arxiv.org/abs/2409.18218)
Comments:
          ECCV 2024

- **What's New**: 본 연구에서는 실제 데이터에 의존하지 않고도 운전 정책을 학습할 수 있는 비대칭 자기 플레이(asymmetric self-play) 방법을 제안합니다. 이 방법은 도전적이고 현실적인 합성 시나리오(synthetic scenarios)를 활용하여 데이터 세트를 확장합니다.

- **Technical Details**: 비대칭 자기 플레이는 두 개의 에이전트로 구성되어 있으며, 하나는 시나리오를 생성하는 '교사(teacher)' 역할을 하고, 다른 하나는 그 시나리오를 해결하는 '학생(student)' 역할을 합니다. 이를 통해 운전 시뮬레이션에서 더 적은 충돌로 현실적인 정책을 학습합니다.

- **Performance Highlights**: 우리의 정책은 기존의 적대적 접근(adversarial approaches)이나 실제 데이터를 사용하는 방법에 비해 종합적으로 성능이 크게 향상되며, 특히 드문(long-tail) 시나리오에서도 우수한 결과를 보입니다.



### Toward Efficient Deep Blind RAW Image Restoration (https://arxiv.org/abs/2409.18204)
Comments:
          IEEE International Conference on Image Processing (ICIP) 2024. arXiv admin note: text overlap with arXiv:2312.15487

- **What's New**: 이 논문에서는 sRGB 도메인에서의 처리의 복잡성에도 불구하고, RAW 이미지에서 직접적으로 이미지 복원(image restoration)을 수행하는 새로운 방법을 제시하고 있습니다.

- **Technical Details**: 새로운 사실적 저하(degradation) 파이프라인을 설계하여, 심층 심각한 RAW 복원 모델(deep blind RAW restoration models)을 훈련합니다. 이 파이프라인은 사실적인 센서 노이즈(sensor noise), 모션 블러(motion blur), 카메라 흔들림(camera shake) 및 기타 일반적인 저하를 고려합니다.

- **Performance Highlights**: 본 연구에서 훈련된 모델은 여러 센서의 데이터를 사용하여 noise와 blur을 성공적으로 줄이고, 다양한 카메라로 촬영된 RAW 이미지의 세부사항을 복원할 수 있습니다.



New uploads on arXiv(cs.AI)

### UniEmoX: Cross-modal Semantic-Guided Large-Scale Pretraining for Universal Scene Emotion Perception (https://arxiv.org/abs/2409.18877)
Comments:
          Submitted to TIP

- **What's New**: 이번 연구에서는 감정 인식을 위한 새로운 대규모 사전 학습 프레임워크인 UniEmoX를 소개합니다. 이는 심리학 연구에서 개인과 환경 간의 상호작용이 감정 탐색 과정과 분리될 수 없다는 점에 영감을 받았습니다.

- **Technical Details**: UniEmoX는 심장 중심(scene-centric) 및 개인 중심(person-centric) 저수준 이미지 공간 구조 정보를 통합하여 더 미세하고 구분 가능한 감정 표현을 도출하는 것을 목표로 합니다. 또한, CLIP 모델을 활용하여 이미지-텍스트 샘플 간의 유사성을 통해 감정 임베딩 표현을 효과적으로 향상시키는 방식으로 동작합니다.

- **Performance Highlights**: 총 6개의 벤치마크 데이터 세트와 2개의 하위 작업에서 실시한 실험을 통해 UniEmoX의 효과성을 검증하였으며, 다양한 시나리오에서 감정 분석을 위한 기존 심리학 이론과 현대적인 대비 학습(contrastive learning) 및 마스크 이미지 모델링 기법을 통합한 최초의 대규모 사전 학습 프레임워크입니다.



### Mitigating Selection Bias with Node Pruning and Auxiliary Options (https://arxiv.org/abs/2409.18857)
- **What's New**: 본 연구는 대규모 언어 모델(LLMs)의 선택 편향(selection bias)을 모델의 내부 표현에서 조사하고, 이를 해결하기 위한 새로운 방법인 Bias Node Pruning (BNP)과 Auxiliary Option Injection (AOI)을 제안합니다.

- **Technical Details**: Bias Node Pruning (BNP)은 편향에 기여하는 선형 계층 파라미터를 제거하는 방법으로, 선택 편향을 줄이는 데 효과적입니다. Auxiliary Option Injection (AOI)은 블랙박스 LLMs에서도 호환되는 입력 수정 기술로, 단순하면서도 효과적인 디바이싱(debiasing) 방법입니다. 또한 Choice Kullback-Leibler Divergence (CKLD)라는 새로운 메트릭을 소개하여, 기존 메트릭의 라벨 불균형(label imbalance)에 대한 민감도 부족 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들이 다양한 데이터셋에서 강력하고 적응력이 뛰어난 성능을 보였으며, 세 가지 LLM에 적용되었습니다.



### LLM With Tools: A Survey (https://arxiv.org/abs/2409.18807)
Comments:
          10 pages

- **What's New**: 본 논문은 대형 언어 모델(LLM)의 효율성과 정확성을 향상시키기 위해 외부 도구를 통합하는 새로운 접근방식을 탐구합니다. LLM이 외부 도구를 사용하는 방법을 교육하는 과정에서의 방법론, 도전 과제 및 발전을 깊이 있게 설명합니다.

- **Technical Details**: 우리는 사용자 지침을 실행 가능한 계획 및 실행으로 매핑하는 일련의 함수에 의해 안내되는 도구 통합을 위한 표준화된 패러다임을 소개합니다. 이는 사용자 의도 이해, 도구 선택, 동적 계획 조정의 중요성을 강조합니다. 이 과정에서 도구 호출 시기, 선택의 정확성, 견고한 추론 과정의 필요성과 같은 여러 도전 과제를 식별했습니다.

- **Performance Highlights**: 최종적으로, 우리는 Chameleon의 결과를 ScienceQA에서 재현하고 코드 구조를 분석했습니다. 이 연구는 LLM이 단순한 도구 사용자에서 도구 생성자로서의 역할로 재정의될 수 있는 가능성을 탐구합니다.



### Learning from Demonstration with Implicit Nonlinear Dynamics Models (https://arxiv.org/abs/2409.18768)
Comments:
          21 pages, 9 figures

- **What's New**: 이 논문에서는 Learning from Demonstration (LfD) 프로세스의 오류 누적 문제를 해결하는 새로운 접근 방식을 제안합니다. 고정 비선형 동적 시스템을 포함하는 독창적인 신경망 레이어를 개발하여, 동적 특성을 조정할 수 있도록 하였습니다.

- **Technical Details**: 이 새로운 신경망 레이어는 reservoir computing에 영감을 받아 설계되었으며, 기존의 신경망 아키텍처에 통합하여 인간의 필기 동작을 재현하는 LASA Human Handwriting Dataset을 사용하여 검증했습니다. 이를 통해 정책 실행 시 오류 누적 문제를 효과적으로 해결하였습니다.

- **Performance Highlights**: 경험적 실험 결과, 제안한 레이어를 통합한 모델이 기존의 정책 예측의 시간적 앙상블 및 Echo State Networks (ESNs)와 비교할 때 필기 작업에서 더 높은 정확도와 강건성을 보여주었으며, 다양한 동적 조건에서도 일반화할 수 있는 성능을 보였습니다.



### Autoregressive Policy Optimization for Constrained Allocation Tasks (https://arxiv.org/abs/2409.18735)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 이 논문에서는 한정된 자원을 각 개체(entity)에 배분해야 하는 제약 할당 작업(constrained allocation tasks)을 다루고 있습니다. 특히 자율 회귀 과정(autoregressive process)을 기반으로 하는 새로운 방법을 제안하고 초기에 발생하는 편향을 상쇄하기 위한 새로운 디바이싱 메커니즘(de-biasing mechanism)을 도입하였습니다.

- **Technical Details**: 이 방법은 각 개체에 대한 할당을 순차적으로 샘플링하는 방식으로 설계되었습니다. 제약 조건(linear constraints)에 따라 투자자들은 특정 산업 부문에 자금의 30% 이상을 할당할 수 없다는 등의 규칙이 있습니다. 이는 허용되는 할당(action space)을 복잡하게 제한하므로 제약을 위반하지 않는 정책을 학습하기 어렵습니다.

- **Performance Highlights**: 제안한 방법은 포트폴리오 최적화(portfolio optimization), 컴퓨팅 작업 분배(computational workload distribution), 그리고 합성 할당 벤치마크(synthetic allocation benchmark)의 세 가지 제약 할당 작업에서 다양한 제약 강화 학습(Constrained Reinforcement Learning, CRL) 방법들과 비교하여 뛰어난 성능을 보였습니다.



### Semantic Model Component Implementation for Model-driven Semantic Communications (https://arxiv.org/abs/2409.18704)
- **What's New**: 본 논문에서는 모델 기반의 의미(semantic) 통신에서 모델의 전파(propagation)가 핵심 기능임을 설명하고, 교차 소스 도메인(cross-source domain) 및 교차 작업(cross-task) 의미 컴포넌트 모델을 설계하였습니다.

- **Technical Details**: 의미 모델 컴포넌트(SMC)는 물리적 채널을 통해 지능이 흐를 수 있도록 설계되었으며, 엣지 노드(edge node)에서 기본 모델을 배포하고 대형 서버 노드가 이를 업데이트하는 구조를 가지고 있습니다. 이 시스템은 다양한 소스와 작업(tasks)을 처리할 수 있습니다. 또한, 채널 노이즈(channel noise)가 모델 성능에 미치는 영향을 논의하고, 노이즈 저항성을 높이기 위한 방법으로 노이즈 주입(injection noise) 및 정규화(regularization)를 제안합니다.

- **Performance Highlights**: SMC는 더 작은 모델 파라미터를 사용하면서도 성능을 유지하고 노이즈에 대한 저항성을 개선하여 교차 소스 및 교차 작업 기능을 달성합니다. 모델 컴포넌트를 실제 응용 프로그램에 적용하기 위해 부품 전이 기반의 무인 차량 추적 프로토타입이 구현되었습니다.



### KALE-LM: Unleash The Power Of AI For Science Via Knowledge And Logic Enhanced Large Mod (https://arxiv.org/abs/2409.18695)
- **What's New**: AI의 잠재력이 커지고 있는 가운데, 과학 연구를 진전시키기 위해 AI를 활용하는 방안에 대해 논의한 비전 논문입니다.

- **Technical Details**: KALE-LM 모델 시리즈의 일환으로 Llama3-KALE-LM-Chem-8B라는 대형 모델을 제안했으며, 화학 분야와 관련된 작업에서 우수한 성능을 나타냈습니다. 이 모델은 오픈 소스로 공개되었습니다.

- **Performance Highlights**: Llama3-KALE-LM-Chem-8B 모델은 화학 관련 작업에서 뛰어난 성능을 보여주며, 더 지능적인 AI 실현을 위한 강력한 출발점이 될 것으로 기대됩니다.



### Toward Universal and Interpretable World Models for Open-ended Learning Agents (https://arxiv.org/abs/2409.18676)
Comments:
          4 pages including appendix, 6 including appendix and references; 2 figures

- **What's New**: 이번 논문에서는 오픈 엔디드(Open-ended) 학습 에이전트를 지원하는 일반적이고 조합적이며 해석 가능한 생성적 세계 모델(generative world models)의 새로운 클래스를 소개합니다.

- **Technical Details**: 이 모델은 광범위한 확률 과정(stochastic processes)을 근사화할 수 있는 희소한 베이즈 네트워크(Bayesian networks)로 구성되어 있으며, 에이전트가 해석 가능하고 계산적으로 확장 가능한 방식으로 세계 모델을 학습할 수 있도록 지원합니다. 이 접근 방식은 베이즈 구조 학습(Bayesian structure learning)과 본질적으로 동기 부여된 계획(intrinsically motivated planning)을 통합하여 에이전트가 능동적으로 세계 모델을 개발하고 수정할 수 있게 합니다.

- **Performance Highlights**: 이러한 방법론은 오픈 엔디드 학습을 촉진하고 더욱 견고하고 적응적인 행동을 이끌어낼 수 있는 가능성을 보여줍니다.



### Not the Silver Bullet: LLM-enhanced Programming Error Messages are Ineffective in Practic (https://arxiv.org/abs/2409.18661)
Comments:
          To appear in the proceedings of the 2024 UK and Ireland Computing Education Research conference (UKICER '24)

- **What's New**: 이 연구는 새로운 언어 모델인 GPT-4가 프로그래밍 오류 메시지를 이해하고 해결하는 데 있어 초보 프로그래머에게 미치는 영향을 실질적인 상황에서 평가했습니다.

- **Technical Details**: 106명의 참가자를 대상으로 한 연구로, 참가자들은 6개의 버그가 있는 C 프로그램을 수정하는 과제를 수행했습니다. 각각의 프로그램에 대해 참가자들은 일반 컴파일러 오류 메시지, 전문가가 작성한 오류 메시지, 또는 GPT-4가 생성한 오류 메시지 중 하나를 무작위로 부여받았습니다.

- **Performance Highlights**: GPT-4가 생성한 오류 메시지는 6개 과제 중 단 1개의 과제에서만 전통적인 컴파일러 오류 메시지를 능가하는 것으로 나타났으며, 전문가의 손으로 작성된 설명이 여전히 LLM과 전통적인 오류 메시지보다 모든 측면에서 더 뛰어난 결과를 보였습니다.



### Reducing Diversity to Generate Hierarchical Archetypes (https://arxiv.org/abs/2409.18633)
- **What's New**: 이 논문에서는 자동으로 추상화 계층을 생성하는 프레임워크(framework)를 제안합니다. 이는 지능적인 행동을 구축하기 위해 필수적인 요구사항으로, 최근의 신경과학 연구에 의해 명확히 드러났습니다.

- **Technical Details**: 작업의 기반이 되는 원시(primitives)의 특성을 명시하고, 이를 바탕으로 구성 아키타입(constructive archetypes)의 계층을 자동으로 생성할 수 있는 방법론(methodology)을 개발하였습니다. 또한, 수리적 정의와 증명을 통해 이 프레임워크의 효과성을 입증하였습니다.

- **Performance Highlights**: 이 프레임워크는 기존의 인공지능 시스템에서 계층적 추상화를 구축하는 데 사용할 수 있으며, 복잡한 문제 해결 및 지능형 시스템 개발에 있어서 새로운 통찰력을 제공할 것으로 기대됩니다.



### Refutation of Spectral Graph Theory Conjectures with Search Algorithms) (https://arxiv.org/abs/2409.18626)
- **What's New**: 이 논문에서는 스펙트럴 그래프 이론(spectral graph theory) 추측의 자동 반박(automatic refutation) 방법을 제안합니다. 기존 연구들은 제한된 크기의 그래프를 생성하거나 딥 강화 학습(deep reinforcement learning)을 통해 반박하는 방식이었습니다.

- **Technical Details**: 우리는 이러한 한계를 해결하기 위해 검색 알고리즘(search algorithms)을 사용합니다. 이 방법을 통해 잠재적으로 큰 반례(counter-example)를 몇 초 만에 찾을 수 있습니다. 다양한 검색 알고리즘을 Graffiti의 여러 추측에 적용했습니다.

- **Performance Highlights**: 우리가 연구한 13개의 Graffiti에서 이미 반박된 추측 중 12개를 몇 초 만에 반박하는 데 성공했습니다. 또한, 지금까지 미해결이었던 Graffiti의 추측 197도 반박하였습니다.



### Unsupervised Cognition (https://arxiv.org/abs/2409.18624)
- **What's New**: 본 논문에서는 최신 인지 모델에 영감을 받아서 의사결정을 위한 기존의 획기적인 primitive 기반 비지도학습(unconstrained learning) 접근 방식을 제안합니다.

- **Technical Details**: 이 방법론은 입력 공간을 독립적으로 구성된 분산 계층 구조(distributed hierarchical structure)로 모델링합니다. 이는 기존의 비지도 학습(classification) 기법들과 비교되며, 특히 암 유형 분류(cancer type classification)에 집중하여 성능을 평가합니다.

- **Performance Highlights**: 제안된 접근 방식은 기존의 최첨단 비지도 학습 알고리즘뿐만 아니라 감독 학습(supervised learning) 알고리즘을 초월하여 더 나은 성능과 인지 모델과 유사한 행동 패턴을 보여줍니다.



### ASAG2024: A Combined Benchmark for Short Answer Grading (https://arxiv.org/abs/2409.18596)
Comments:
          Accepted at SIGCSE-Virtual 2024

- **What's New**: 이 연구는 여러 학과, 채점 기준 및 분포에 걸쳐 통합된 단답형 채점 벤치마크인 ASAG2024를 소개합니다. 이는 자동 채점 시스템의 비교를 용이하게 계속 할 수 있게 합니다.

- **Technical Details**: ASAG2024 벤치마크는 일곱 개의 일반적으로 사용되는 단답형 채점 데이터셋을 통합하여 공통 구조 및 채점 기준을 제공합니다. 연구에서는 최근의 단답형 채점 방법들의 성능을 평가하였으며, LLM 기반 접근 방식이 새로운 높은 점수를 기록하지만 여전히 인간의 성능에 미치지 못한다는 것을 보여주었습니다.

- **Performance Highlights**: 최근 SAG 방법들은 이전보다 높은 점수를 기록했지만, 인간 채점자의 성능과 비교할 때 여전히 큰 간극이 존재합니다. 이는 향후 연구에서 인간-기계 SAG 시스템의 가능성을 열어줍니다.



### "Oh LLM, I'm Asking Thee, Please Give Me a Decision Tree": Zero-Shot Decision Tree Induction and Embedding with Large Language Models (https://arxiv.org/abs/2409.18594)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)이 데이터가 제한적인 상황에서 예측 모델링을 위해 사전 지식을 활용하는 강력한 방법을 제공함을 보여줍니다. 특히, LLMs이 훈련 데이터 없이도 본질적으로 해석 가능한 머신 러닝 모델인 의사결정나무(decision trees)를 생성할 수 있음을 설명합니다.

- **Technical Details**: 이 연구에서는 LLMs가 압축된 세계 지식을 활용하여 제로샷(Zero-shot) 의사결정나무를 생성하는 방법을 보여줍니다. 이 의사결정나무는 작고 간단한 테이블 데이터셋(tabular datasets)에 대해 데이터 기반 의사결정나무보다 뛰어난 성능을 보일 수 있습니다. 또한, 이 나무에서 파생된 임베딩(embeddings)은 평균적으로 데이터 기반 의사결정나무에서 파생된 것들과 동등한 성능을 나타냅니다.

- **Performance Highlights**: 지식 기반 의사결정나무 유도와 임베딩 접근 방식은 데이터가 제한된 상황에서 데이터 기반 머신 러닝 방법의 강력한 새로운 기준선을 제공합니다.



### Experimental Evaluation of Machine Learning Models for Goal-oriented Customer Service Chatbot with Pipeline Architectur (https://arxiv.org/abs/2409.18568)
- **What's New**: 이 논문은 고객 서비스 챗봇을 위한 맞춤형 실험 평가 접근법을 제시하여, 자연어 이해(NLU), 대화 관리(DM), 자연어 생성(NLG)라는 세 가지 주요 구성 요소에 중점을 둡니다.

- **Technical Details**: 연구에서는 NLU(BERT 및 LSTM 사용), DM(DQN 및 DDQN 활용), NLG(GPT-2 및 DialoGPT 활용)의 개별 평가를 강조하며, 하이퍼파라미터 최적화와 후보 모델 평가를 수행합니다.

- **Performance Highlights**: NLU에서는 BERT가 의도 탐지에서 우수한 성과를 보였고, LSTM은 슬롯 채우기에서 더 나은 결과를 나타냈습니다. DM에서는 DDQN 모델이 DQN을 능가하며 더 적은 턴 수, 더 높은 보상 및 성공률을 달성했습니다. NLG에서는 대형 언어 모델인 GPT-2가 DialoGPT보다 BLEU, METEOR 및 ROUGE 지표에서 더 뛰어난 성능을 보여주었습니다.



### Align$^2$LLaVA: Cascaded Human and Large Language Model Preference Alignment for Multi-modal Instruction Curation (https://arxiv.org/abs/2409.18541)
- **What's New**: 최근 Multi-modal Large Language Models (MLLMs) 분야에서 LLaVA 시리즈 모델과 같은 최신 기술은 방대한 기계 생성 지침 데이터 조정에 의해 발전하고 있습니다. 그러나 이러한 자동 지침 수집 파이프라인은 데이터 품질에 중대한 변동성을 초래합니다.

- **Technical Details**: 이 논문에서는 두 가지 독특한 관점, 즉 인간과 LLM 선호 정렬을 통해 기계 생성 멀티모달 지침 데이터의 대량 집합을 압축하고 고품질 형태로 변환하는 새로운 지침 큐레이션 알고리즘을 소개합니다. (i) 인간 선호 정렬: 전문가는 주관적 및 객관적 기준을 통해 데이터 품질을 평가하고, 주석이 달린 데이터셋에 대해 보상 모델을 훈련시켜 인간의 지침 정렬에 대한 미세한 이해를 내재화합니다. (ii) LLM 선호 정렬: 보상 모델에 의해 선택된 지침을 바탕으로 MLLM에서 사용되는 내부 LLM의 작성 스타일에 시각적 지침을 맞추는 방법을 제안합니다.

- **Performance Highlights**: 신 synthetic 멀티모달 지침을 최대 90% 압축하여 모델 성능을 유지하거나 개선할 수 있음을 실험을 통해 입증했습니다. 특히, 전체 훈련 샘플 수를 158k에서 14k로 감소시킴으로써(9배 작아짐) 다양한 MLLM 벤치마크에서 모델이 전체 데이터셋에 비해 일관되게 우수한 성능을 발휘했습니다.



### Data Analysis in the Era of Generative AI (https://arxiv.org/abs/2409.18475)
- **What's New**: 본 논문은 AI 기반 도구가 데이터 분석을 재형성할 수 있는 잠재력을 탐구합니다. 특히, 대형 언어 모델과 다중 모달 모델의 등장으로 데이터 분석 작업 흐름의 다양한 단계를 향상시키기 위한 새로운 기회들이 제공된다는 점에 주목합니다.

- **Technical Details**: AI 도구는 사용자의 고급 의도를 실행 가능한 코드, 차트 및 인사이트로 변환하는 기능을 갖추고 있으며, 인간 중심 디자인 원칙을 통해 직관적인 상호작용을 촉진하고 사용자 신뢰를 구축하는 방법을 논의합니다. 이러한 시스템이 직면한 연구 도전 과제로는 모델 능력 향상, 평가 및 벤치마크, 최종 사용자 요구 사항 이해 등이 있습니다.

- **Performance Highlights**: AI-assisted 분석(workflow)이 여러 앱에 걸쳐 신속하게 진행될 수 있도록 하여 데이터 분석의 효율성을 높이는 방안을 제시하고 있습니다.



### Cost-Aware Dynamic Cloud Workflow Scheduling using Self-Attention and Evolutionary Reinforcement Learning (https://arxiv.org/abs/2409.18444)
Comments:
          This paper has been accepted by ICSOC (International Conference on Service-Oriented Computing) 2024

- **What's New**: 이번 논문에서는 클라우드 환경에서 Cost-aware Dynamic Multi-Workflow Scheduling (CDMWS) 문제를 다룹니다. 이는 가상 머신 (VM) 인스턴스를 효율적으로 할당하여 서비스 수준 계약 (SLA) 위반에 따른 벌금과 VM 임대료를 최소화하는 데 목표를 두고 있습니다.

- **Technical Details**: 전통적인 Reinforcement Learning (RL) 정책 네트워크는 기본적인 feedforward 아키텍처를 사용하여 각 VM 인스턴스의 적합성을 개별적으로 결정합니다. 그러나 본 논문에서는 Self-Attention Policy Network (SPN-CWS)를 제안하여 모든 VM으로부터 전역 정보를 포착하는 방법을 제시합니다. 또한 Evolution Strategy 기반의 RL 시스템(ERL)을 개발하여 SPN-CWS를 신뢰성 있게 훈련합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 여러 벤치마크 CDMWS 문제에 대해 기존의 최첨단 알고리즘들보다 눈에 띄게 우수한 성능을 보였음을 확인했습니다.



### Physics Augmented Tuple Transformer for Autism Severity Level Detection (https://arxiv.org/abs/2409.18438)
Comments:
          12 pages

- **What's New**: 이 논문은 자폐 스펙트럼 장애(ASD)의 중증도 인식을 위해 물리 법칙을 활용하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 스켈레톤 기반의 운동 궤적을 통해 피험자의 행동을 고차원 잠재 공간에서 인코딩하는 물리 기반 신경망 아키텍처를 포함하고 있습니다.

- **Technical Details**: 제안된 네트워크는 두 가지 디코더를 사용합니다: 물리 기반 디코더(physics-based decoder)와 비물리 기반 디코더(non-physics-based decoder). 물리 기반 디코더는 스켈레톤 시퀀스에 적용되는 물리 법칙을 예측 과정에 적용하며, 비물리 기반 디코더는 예측된 행동과 실제 행동 간의 차이를 최소화하도록 최적화됩니다. 또한, 분류기도 동일한 잠재 공간 임베딩을 활용하여 ASD의 중증도를 인식합니다.

- **Performance Highlights**: 제안된 방법은 여러 ASD 진단 벤치마크에서 최첨단 성능을 달성했으며, ASD 진단 작업 외에도 낙상 예측(publicly available benchmark for fall prediction) 실험을 통해 모델의 우수성을 입증했습니다.



### Multimodal Trajectory Prediction for Autonomous Driving on Unstructured Roads using Deep Convolutional Network (https://arxiv.org/abs/2409.18399)
Comments:
          11 pages,6 figures

- **What's New**: 최근 자율 주행 기술이 열린 광산(오픈핏 마이닝)에서 안전하고 효율적인 광물 수송을 이루기 위해 주목받고 있습니다. 이 논문에서는 비구조적 도로에서의 경로 예측 문제를 해결하기 위한 새로운 방법이 제안되었습니다.

- **Technical Details**: 제안된 방법은 목표 차량의 여러 가능한 경로와 그 확률을 예측합니다. 주변 환경과 목표 차량의 과거 경로는 래스터화한 이미지로 인코딩되며, 이는 심층 합성곱 네트워크를 사용하여 입력으로 넣어집니다. 이 기술은 열린 광산 자율 주행 시나리오에 특화된 데이터셋을 기반으로 오프라인 테스트를 수행하였습니다.

- **Performance Highlights**: 제안된 방법은 물리 기반(physics-based) 방법과 비교하여 평가되었으며, 실제 적용 가능성을 높이는 중요한 결과를 도출하였습니다. 오픈 소스 코드 및 데이터는 제공된 링크에서 확인할 수 있습니다.



### Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving (https://arxiv.org/abs/2409.18343)
- **What's New**: 이번 연구는 자율주행 차량의 에이전트 행동 모델링에서 에이전트를 보다 신뢰성 있게 만들기 위한 방법을 제시합니다. 특히, 행동 모델을 강화 학습( reinforcement learning)을 통해 폐쇄 루프( closed-loop)로 미세 조정하여 개선된 성능을 보여줍니다.

- **Technical Details**: 연구에서는 Waymo Open Sim Agents 챌린지에서 개선된 전반적인 성능은 물론 충돌률(collision rate)과 같은 목표 지표(targeted metrics)의 향상도 확인되었습니다. 또한, 시뮬레이션된 에이전트의 능력을 직접 평가하기 위한 새로운 정책 평가 벤치마크(policy evaluation benchmark)를 도입하여 자율주행 차량 플래너의 품질을 측정하는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 에이전트 행동의 신뢰성을 높이며, 자율주행 차량 계획의 성과를 향상시키는 우수성을 입증하였습니다.



### A Fairness-Driven Method for Learning Human-Compatible Negotiation Strategies (https://arxiv.org/abs/2409.18335)
Comments:
          EMNLP Findings 2024

- **What's New**: 최근 인공지능(AI) 및 자연어 처리(NLP)의 발전에도 불구하고, 협상(negotiation)은 여전히 AI 에이전트에게 어려운 분야입니다. 최근에 제안된 FDHC 프레임워크는 공정성(fairness)을 고려하여 인간과 호환되는 협상 전략을 학습할 수 있게 도와줍니다.

- **Technical Details**: FDHC 프레임워크는 보상 설계(reward design)와 탐색(search) 두 가지 측면에서 공정성을 통합합니다. 또한, LGM-Zero라는 새로운 RL(강화 학습) + search 기법을 도입하여 사전 훈련된 언어 모델(pre-trained language model)을 활용하여 대규모 행동 공간에서 인간과 호환되는 제안을 검색합니다.

- **Performance Highlights**: 실험 결과, FDHC는 협상 결과를 더 평등하게 만들고 협상 품질을 향상시키는 데 성공하였습니다.



### Input-Dependent Power Usage in GPUs (https://arxiv.org/abs/2409.18324)
- **What's New**: 이번 논문에서는 GPU의 입력 데이터를 조정하여 고전력 소모 문제를 해결할 수 있는 방법을 제안합니다.

- **Technical Details**: 연구에서는 행렬-행렬 곱셈(GEMM)에 대한 입력 값 분포(value distribution), 비트 유사성(bit similarity), 위치 배치(placement), 희소성(sparsity) 등 네 가지 가지 입력 변화를 실험하였습니다. 이 입력 변화를 통해 GPU의 전력 사용량이 최대 40%까지 변화할 수 있음을 확인하였습니다.

- **Performance Highlights**: 입력에 따라  비트 플립(bit flip)의 수가 변화하게 되며, 이를 통해 컴파일러와 스케줄러 최적화를 가능하게 하여 GPU 전력 관리 및 에너지 소비 절감을 할 수 있음을 제안합니다.



### Cross-Institutional Structured Radiology Reporting for Lung Cancer Screening Using a Dynamic Template-Constrained Large Language Mod (https://arxiv.org/abs/2409.18319)
- **What's New**: 본 논문은 구조적 방사선 보고서를 생성하기 위한 향상된 오픈 소스 LLM(대형 언어 모델)을 개발하는 새로운 접근법을 제안합니다. 기존 모델들이 직면한 형식 오류, 콘텐츠 착각(content hallucinations), 개인정보 유출 문제를 해결하고자 합니다.

- **Technical Details**: 연구팀은 두 기관으로부터 수집된 5,442개의 비식별화된 LCS 보고서를 분석하고, 이 중 500개의 보고서를 무작위로 선택하여 수작업으로 라벨링하였습니다. 29개의 특징을 포함한 표준화된 보고서 템플릿을 개발했으며, 이를 기반으로 템플릿 제약 디코딩(template-constrained decoding)을 이용해 LLAMA, Qwen, Mistral 등의 오픈 소스 LLM을 향상시켰습니다. 성능 평가는 F1 점수, 신뢰 구간(confidence interval), McNemar test, z-test를 통해 수행되었습니다.

- **Performance Highlights**: 제안한 방법은 여러 기관의 데이터셋에서 LLM 성능을 일관되게 향상시켰으며, 형식 오류나 콘텐츠 착각이 없었습니다. 오픈 소스 LLAMA-3.1 405B 모델의 성능을 최대 10.42% 개선하였고, GPT-4o보다 17.19% 더 뛰어난 성과를 보였습니다. 또한, 대규모 다중 모달(multimodal) 데이터베이스에서 신규 결절 검색 시스템을 성공적으로 프로토타입하고 자동으로 통계 분석을 수행하여, 이전 결과와의 일관성을 보여주었습니다.



### Retrospective Comparative Analysis of Prostate Cancer In-Basket Messages: Responses from Closed-Domain LLM vs. Clinical Teams (https://arxiv.org/abs/2409.18290)
- **What's New**: 이번 논문에서는 RadOnc-GPT라는 특화된 대형 언어 모델(Large Language Model, LLM)을 소개합니다. 이 모델은 전립선암의 방사선 치료를 중점적으로 다루며, 환자와의 소통을 개선하기 위해 설계되었습니다.

- **Technical Details**: RadOnc-GPT는 병원 전체 전자 건강 기록(Electronic Health Record, EHR) 데이터베이스와 방사선 종양학에 특화된 내부 데이터베이스를 통합하여 운영됩니다. 이 모델은 158개의 이전 메시지 상호작용을 기반으로 평가되었으며, 정량적 자연어 처리(Natural Language Processing, NLP) 분석과 임상팀의 두 차례에 걸친 평가로 응답의 질을 분석하였습니다.

- **Performance Highlights**: RadOnc-GPT는 응답의 'Clarity'(명확성)와 'Empathy'(공감) 부분에서 임상 팀보다 약간 우수한 성과를 보였으며, 'Completeness'(완전성)와 'Correctness'(정확성)에서 비슷한 점수를 기록했습니다. 이 모델은 간호사에게 메시지 처리 시 5.2분, 임상 의사에게는 2.4분의 시간을 절약할 것으로 추정되며, 임상 팀의 업무 부담을 줄이고 의료비 절감에도 기여할 수 있습니다.



### Trustworthy AI: Securing Sensitive Data in Large Language Models (https://arxiv.org/abs/2409.18222)
Comments:
          40 pages, 1 figure

- **What's New**: 이 논문은 Large Language Models (LLMs)의 신뢰 메커니즘을 통합하여 민감한 정보의 공개를 동적으로 제어할 수 있는 포괄적인 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 핵심 요소로 구성됩니다: 사용자 신뢰 프로파일링(User Trust Profiling), 정보 민감도 감지(Information Sensitivity Detection), 및 적응형 출력 제어(Adaptive Output Control). Role-Based Access Control (RBAC), Attribute-Based Access Control (ABAC), Named Entity Recognition (NER)와 같은 기술을 활용하고, 맥락 분석(contextual analysis) 및 차등 프라이버시(differential privacy)와 같은 개인 정보 보호 방법을 통합합니다.

- **Performance Highlights**: 이 시스템은 사용자의 신뢰 수준에 따라 민감한 정보를 적절히 공개하여 데이터 유용성과 개인 정보 보호의 균형을 맞추고, LLMs의 안전한 배치를 위한 새로운 접근 방식을 제공합니다.



### MMMT-IF: A Challenging Multimodal Multi-Turn Instruction Following Benchmark (https://arxiv.org/abs/2409.18216)
Comments:
          24 pages, 16 figures

- **What's New**: MMMT-IF 데이터셋과 Programmatic Instruction Following (PIF) 메트릭을 도입하여 멀티모달, 멀티턴 대화에서 지시사항을 따르는 능력을 평가하는 새로운 방법을 제안합니다.

- **Technical Details**: MMMT-IF는 질문들 사이에 추가된 전역 지시사항을 포함하여, 모델이 긴 대화 속에서 분산된 지시사항을 검색하고 지시사항 제약 하에서 추론할 수 있도록 도전합니다. 모든 지시사항은 코드 실행을 통해 객관적으로 검증 가능합니다. PIF 메트릭은 추론 작업을 수행하는 동안 정확하게 따르는 지시사항의 비율을 측정합니다. PIF-N-K 메트릭은 모델 응답 중 K개가 PIF 점수가 1인 비율을 측정하여 강건성을 평가합니다.

- **Performance Highlights**: Gemini 1.5 Pro, GPT-4o, Claude 3.5 Sonnet의 평균 PIF 점수가 턴 1에서 0.81에서 턴 20에서 0.64로 감소하며, 모든 응답을 4회 반복했을 때 GPT-4o와 Gemini가 모든 지시사항을 성공적으로 따르는 경우는 단 11%입니다. 지시사항이 모델 입력 문맥의 끝에 추가되면 평균 PIF 점수가 22.3 포인트 향상됩니다.



### Autonomous Network Defence using Reinforcement Learning (https://arxiv.org/abs/2409.18197)
- **What's New**: 이 논문에서는 네트워크 보안의 방어자가 공격자에 비해 불리한 상황을 반전시키기 위해 자율 에이전트(autonomous agents)의 효과를 조사합니다.

- **Technical Details**: 강화 학습(reinforcement learning)에 대한 배경을 제공하고, 3개의 서브넷(subnets)으로 구성된 13개의 호스트(hosts)의 네트워크 환경 시뮬레이션에서 이벤트를 테스트합니다. 논문에서는 신뢰할 수 있는 방어를 위한 새로운 강화 학습 에이전트를 설계 및 훈련합니다.

- **Performance Highlights**: 이 에이전트는 공격자가 두 가지 유형 – 하나는 네트워크 레이아웃에 대한 완전한 지식을 가진 고급 지속 위협(advanced persistent threat, APT) 에이전트와 다른 하나는 탐색을 통해 자원을 발견해야 하지만 보다 일반적인 에이전트로서의 공격을 계속 방어할 수 있음을 보여줍니다.



### Data-Prep-Kit: getting your data ready for LLM application developmen (https://arxiv.org/abs/2409.18164)
Comments:
          10 pages, 7 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM) 개발을 위한 데이터 준비의 중요성을 강조하며, 사용자들이 손쉽게 확장하고 조정할 수 있는 오픈 소스 데이터 준비 툴킷인 Data Prep Kit (DPK)를 소개합니다.

- **Technical Details**: DPK는 데이터 준비를 사용자의 필요에 맞게 조정할 수 있도록 설계된 아키텍처를 가지고 있으며, 로컬 머신에서 데이터를 준비할 수 있을 뿐만 아니라 수천 개의 CPU 코어가 있는 클러스터에서도 손쉽게 확장할 수 있습니다. DPK는 자연어 및 코드 데이터를 변환하는 확장 가능한 모듈 세트를 제공합니다. 사용자가 추가적인 변환이 필요할 경우, DPK의 지원을 통해 손쉽게 개발할 수 있습니다. 이러한 모듈들은 독립적으로 또는 파이프라인 방식으로 연속적인 작업을 수행하는 데 사용될 수 있습니다.

- **Performance Highlights**: DPK의 성능은 작은 규모에서 시작하여 매우 큰 수의 CPU까지 확장 가능함을 보여줍니다. DPK의 모듈은 Granite 모델 데이터 준비에도 사용되었습니다. DPK는 AI 커뮤니티가 LLM 모델 성능을 향상하거나 Retrieval-Augmented Generation (RAG) 기능으로 모델을 미세 조정할 수 있도록 돕는 귀중한 기여라고 믿습니다.



### A Survey on Multimodal Benchmarks: In the Era of Large AI Models (https://arxiv.org/abs/2409.18142)
Comments:
          Ongoing project

- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)에 대한 벤치마크(benchmark) 분석이 부족한 상황에서, 211개의 벤치마크를 체계적으로 검토하였습니다.

- **Technical Details**: MLLMs의 평가를 위해 이해, 추론, 생성 및 응용의 네 가지 핵심 영역에서 벤치마크를 분석하였으며, 작업 설계(task design), 평가 지표(evaluation metrics) 및 데이터셋 구성(dataset construction)에 대한 세부적인 분석을 제공합니다.

- **Performance Highlights**: 이 논문은 MLLM 연구의 발전에 기여할 수 있는 포괄적인 벤치마킹 관행(overview of benchmarking practices)을 제공하며, 향후 연구의 유망한 방향성을 제시하고자 합니다.



### PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation (https://arxiv.org/abs/2409.18964)
Comments:
          Accepted to ECCV 2024. Project page: this https URL

- **What's New**: PhysGen은 단일 이미지를 입력으로 사용하여 현실적이고 물리적으로 그럴듯한 비디오를 생성하는 혁신적인 이미지-비디오 생성 방법입니다. 주목할 점은 이미지에 적용된 힘과 토크와 같은 입력 조건을 활용하여 동적 영상을 생성하는 것입니다.

- **Technical Details**: PhysGen의 핵심 구성 요소는 세 가지로 나눌 수 있습니다: (i) 이미지의 기하학, 재료 및 물리적 매개변수를 효과적으로 캡처하는 이미지 이해 모듈; (ii) 강체 물리학(rigid-body physics)과 유추된 매개변수를 사용하는 이미지 공간 동적 시뮬레이션 모델; (iii) 생성적 영상 확산(generative video diffusion) 기법을 활용해 시뮬레이션된 동작을 포함하는 현실적인 비디오 영상을 생성하는 이미지 기반 렌더링 및 정제 모듈입니다.

- **Performance Highlights**: PhysGen을 통해 생성된 비디오는 물리적 사실성과 외관 측면에서 현실적이며 정교한 제어가 가능합니다. 기존의 데이터 기반 이미지-비디오 생성 연구들과의 정량적 비교 및 포괄적인 사용자 연구를 통해 뛰어난 결과를 보여줍니다.



### Exploring Token Pruning in Vision State Space Models (https://arxiv.org/abs/2409.18962)
Comments:
          NeurIPS'24

- **What's New**: 본 논문에서는 State Space Models (SSMs)을 기반으로 한 비전 모델의 효율성을 향상시키기 위해 토큰 기반의 프루닝(token pruning) 기법을 제안합니다. 기존의 Vision Transformers (ViTs) 기술을 활용한 토큰 프루닝 기법의 제한점을 분석하고, SSM의 고유한 계산 특성을 고려하여 새로운 방법을 개발했습니다.

- **Technical Details**: 제안된 방법은 pruning-aware hidden state alignment 기법을 도입하여 남아있는 토큰의 이웃을 안정화하고, SSM 모델에 적합한 토큰 중요성 평가(token importance evaluation) 방법을 통해 토큰 프루닝을 수행합니다. 이로 인해 효율적인 구현 및 실질적인 가속화가 이루어집니다.

- **Performance Highlights**: 제안된 방법은 ImageNet에서 41.6\%의 FLOPs 감소와 함께 81.7\%의 정확도를 달성하며, 다양한 작업에서 성능에 미치는 영향을 최소화하면서도 계산량을 대폭 줄일 수 있음을 입증했습니다. 또한, 이 연구는 SSM 기반 비전 모델의 동작을 이해하는 데 더 깊은 통찰을 제공합니다.



### ProMerge: Prompt and Merge for Unsupervised Instance Segmentation (https://arxiv.org/abs/2409.18961)
Comments:
          ECCV2024 camera-ready

- **What's New**: 이 논문에서는 Unsupervised instance segmentation(비지도 인스턴스 분할)에서 새로운 방법인 Prompt and Merge(ProMerge)를 제안합니다. ProMerge는 self-supervised visual features(자기지도 시각적 특징)를 활용해 패치를 초기 그룹화하고 전략적으로 병합하는 접근법입니다.

- **Technical Details**: ProMerge는 DINO와 같은 self-supervised 모델에서 제공하는 강력한 시각적 특징을 활용하고, sophisticated background-based mask pruning technique(정교한 배경 기반 마스크 가지치기 기법)을 적용하여 초기 세그먼트를 병합합니다. 또한, 기존의 normalized-cut(정규화 컷)을 사용하는 방법에 비해 계산 요구 사항을 줄이며, inference speed(추론 속도)를 크게 개선합니다.

- **Performance Highlights**: ProMerge는 경쟁력 있는 결과를 제공하며, 기존 최첨단 normalized-cut 기반 접근법에 비해 추론 시간을 크게 단축시킵니다. 우리의 마스크 예측을 pseudo-labels(의사 레이블)로 사용하는 객체 탐지기를 훈련할 경우, 다양한 challenging instance segmentation benchmarks(도전적인 인스턴스 분할 벤치마크)에서 현재 최고 수준의 비지도 모델을 능가합니다.



### $O(d/T)$ Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions (https://arxiv.org/abs/2409.18959)
- **What's New**: 본 논문은 점수 기반 확산 모델(score-based diffusion models)의 이론적 수렴성을 개선하여 빠른 수렴 이론을 확립합니다. 기존의 이론적 보장이 엄격한 가정이나 최적이 아닌 수렴 속도에 의해 제한되는 문제를 해결합니다.

- **Technical Details**: 우리는 최소한의 가정 하에 인기 있는 SDE(Stochastic Differential Equation) 기반 샘플러의 빠른 수렴 이론을 수립합니다. 분석 결과, $	ext{l}_2$-정확한 점수 함수(score function) 추정치를 제공하면, 목표 분포(target distribution)와 생성된 분포들 간의 총 변이 거리(total variation distance)가 O(d/T)로 상한이 설정됩니다. 여기서 d는 데이터 차원, T는 단계의 수를 나타냅니다.

- **Performance Highlights**: 이 결과는 유한한 1차 모멘트를 가지는 모든 목표 분포에 대해 적용되며, 기존의 SDE 기반 샘플러와 ODE(Ordinary Differential Equation) 기반 샘플러의 수렴 이론을 개선합니다. 이 연구는 각 단계에서 오류가 어떻게 전파되는지를 세밀하게 나타내는 새로운 분석 도구를 통해 이루어졌습니다.



### LML: Language Model Learning a Dataset for Data-Augmented Prediction (https://arxiv.org/abs/2409.18957)
Comments:
          First version

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 분류 작업에 활용하기 위한 새로운 접근 방식을 소개합니다. 전통적인 머신 러닝(ML) 모델들과는 달리, LLMs를 사용하여 데이터 정리(data cleaning)와 특징 공학(feature engineering)을 간소화합니다.

- **Technical Details**: 이 논문은 'Language Model Learning (LML)'이라는 새로운 개념과 'Data-Augmented Prediction (DAP)'이라는 새로운 방법을 제안합니다. LLM이 데이터를 탐색하고 이해하며 분류를 결정하는 방식으로 분류를 수행합니다. DAP 과정에서는 데이터 요약을 사용하여 자동으로 쿼리를 생성하고, 이를 통해 관련 데이터 행을 검색한 후, 최종적인 분류 결과를 생성합니다.

- **Performance Highlights**: 테스트 사례에서 시스템의 정확도가 90%를 초과하여 기존 ML 모델을 다양한 시나리오에서 초월할 가능성을 입증하였습니다. 사용자가 예측의 논리를 검토할 수 있도록 'Explainable Machine Learning Model'로 행위하는 문구를 프롬프트에 포함시킴으로써 예측의 해석 가능성을 향상시켰습니다.



### Unconditional stability of a recurrent neural circuit implementing divisive normalization (https://arxiv.org/abs/2409.18946)
- **What's New**: 본 연구에서는 생물학적으로 그럴듯한 신경역학 모델의 안정성을 높이기 위해 동적 분배 정규화(dynamic divisive normalization, DN)와 ORGaNICs 회로 모델의 안정성을 연결했습니다. 이는 다양한 신경생리학적 현상을 모사할 수 있는 회로 모델입니다.

- **Technical Details**: 우리는 Lyapunov의 간접법을 사용하여 재귀 가중치 행렬이 단위 행렬일 때 임의 차원의 ORGaNICs 회로에 대한 무조건적 지역 안정성을 증명했습니다. 또한 이를 커플링된 감쇠 조화 진동기 시스템에 연결하여 회로의 에너지 함수 도출과 함께 회로와 개별 뉴런이 달성하고자 하는 목표를 규명합니다. 일반적인 재귀 가중치 행렬의 경우 2D 모델의 안정성을 증명하고, 더 높은 차원에서도 안정성이 유지됨을 실험적으로 입증했습니다.

- **Performance Highlights**: ORGaNICs는 자체적인 안정성 특성과 적응형 시간 상수 덕분에 폭발, 소멸 및 진동하는 그래디언트 문제를 해결하며, 시간에 따른 역전파(backpropagation through time) 방법으로 훈련될 수 있습니다. RNN 벤치마크에서 ORGaNICs는 정적 이미지 분류 작업에서 대안 신경역학 모델보다 우수한 성능을 보였으며, 순차 작업에서는 LSTM과 유사한 성능을 나타냈습니다.



### Building Trust Through Voice: How Vocal Tone Impacts User Perception of Attractiveness of Voice Assistants (https://arxiv.org/abs/2409.18941)
Comments:
          Extended Abstract

- **What's New**: 본 논문은 음성 비서(Voice Assistants, VAs)가 복잡한 작업에 적합하다고 인식될 수 있도록 음성의 톤이 사용자의 신뢰성에 미치는 영향을 탐구하였습니다.

- **Technical Details**: 연구에서는 다양한 음성 톤을 가진 VA의 음성을 실험 참가자들에게 제공하여, 각 음성의 매력도(attractiveness)와 신뢰성(trustworthiness)을 평가하였습니다. 긍정적이거나 중립적인 톤을 가진 VA가 더욱 매력적이고 신뢰성 있게 인식되었습니다.

- **Performance Highlights**: VA의 음성 톤 설계를 통해 사용자가 느끼는 신뢰성을 향상시킬 수 있다는 결론에 도달하였습니다. 연구 결과, 매력적으로 느껴지는 VA가 사용자의 신뢰를 높이는 데 긍정적인 영향을 미친다는 것을 보였습니다.



### From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding (https://arxiv.org/abs/2409.18938)
Comments:
          11 pages

- **What's New**: 본 논문에서는 MultiModal Large Language Models (MM-LLMs)과 시각적인 인코더(visual encoders)의 통합을 통해 긴 비디오 이해(long video understanding)에서의 고유한 도전 과제를 조명합니다. 기존의 정적 이미지(static image)와 짧은 비디오(short video) 이해와의 차별점을 분명히 합니다.

- **Technical Details**: MM-LLMs의 설계(model design) 및 훈련(training methodologies) 방식에서 긴 비디오 이해를 위한 발전을 다룹니다. 짧은 비디오는 연속적인 프레임을 포함하여 공간(spatial) 및 사건 내 템포럴(temporal) 정보를 가지며, 긴 비디오는 여러 사건이 여유 있는 시간 간격으로 발생합니다.

- **Performance Highlights**: 기존 MM-LLMs의 비디오 이해(video understanding) 벤치마크 성능을 비교하고, 긴 비디오 이해를 위한 향후 발전 방향을 논의합니다.



### AIPatient: Simulating Patients with EHRs and LLM Powered Agentic Workflow (https://arxiv.org/abs/2409.18924)
Comments:
          42 pages, 6 figures, 7 tables

- **What's New**: AIPatient, an advanced simulated patient system, leverages Large Language Models (LLM) and integrates a Knowledge Graph (KG) from Electronic Health Records (EHRs) to enhance clinical decision-making simulations in medical education.

- **Technical Details**: AIPatient utilizes the AIPatient Knowledge Graph (AIPatient KG) sourced from the MIMIC-III database, creating a diverse cohort of 1,495 clinically relevant patients. It employs the Reasoning Retrieval-Augmented Generation (Reasoning RAG) framework, which involves six LLM-powered agents for tasks such as retrieval, KG query generation, and summarization.

- **Performance Highlights**: The AIPatient system achieved an accuracy of 94.15% in EHR-based medical Question Answering (QA) and demonstrated high readability and robustness, making it suitable for diverse applications in medical education and system integration.



### Soft Measures for Extracting Causal Collective Intelligenc (https://arxiv.org/abs/2409.18911)
Comments:
          Camera-ready version accepted for publication in the EMNLP 2024 Workshop NLP4Science

- **What's New**: 이 연구는 복잡한 사회 시스템을 설명하기 위해 집합적 지능(collective intelligence)을 이해하고 모델링하는 중요성을 강조하며, 대규모 언어 모델(large language models, LLMs)을 이용하여 퍼지 인지 맵(fuzzy cognitive maps, FCMs) 추출을 자동화하는 방법을 제안합니다.

- **Technical Details**: 연구에서는 새로운 그래프 기반 유사도 측정(graph-based similarity measures)을 도입하고, 이를 인간 평가와 비교하기 위해 Elo 등급 시스템(Elo rating system)을 적용합니다. FCM의 미세한 뉘앙스(capture nuances)를 포착하는 데 한계가 있다는 것이 강조되며, LLM을 미세 조정(fine-tuning)함으로써 성능을 향상시킬 수 있지만, 기존 측정 방식은 여전히 부족합니다.

- **Performance Highlights**: 결과는 인간 평가와 긍정적인 상관관계를 보이며, 하지만 가장 성능이 좋은 측정 방법조차 FCM의 복잡성을 완전히 포착하지 못하는 한계를 보입니다. 이 연구는 FCM 추출을 위한 부드러운 유사도 측정(soft similarity measures)의 필요성을 강조하며, 자연어 처리(NLP)와 함께 집합적 지능 모델링을 발전시킵니다.



### Improving Visual Object Tracking through Visual Prompting (https://arxiv.org/abs/2409.18901)
Comments:
          Accepted and to appear in IEEE Transactions on Multimedia

- **What's New**: 본 연구에서는 PiVOT(Visual Prompting mechanism for generic Visual Object Tracking)를 통해 기존의 목표와 주변 방해물(distractor)을 구별하는 문제를 해결하기 위한 새로운 시각적 프롬프트(prompt) 생성 네트워크를 제안합니다.

- **Technical Details**: PiVOT는 CLIP이라는 사전 훈련된 모델을 사용하여 자동으로 시각적 프롬프트를 생성하고 수정합니다. CLIP은 범주 수준의 광범위한 지식을 제공하며, 트래커는 객체 인스턴스(instance-specific data)에 대한 훈련을 통해 고유한 객체 인스턴스를 인식하는 데 강점을 가집니다. PiVOT는 시각적 프롬프트를 잠재적 목표 위치를 강조하는 형태로 컴파일합니다.

- **Performance Highlights**: 여러 벤치마크에서 수행된 실험을 통해 PiVOT은 방해 물체(distractors)를 억제하고 트래커의 성능을 향상시키는 데 효과적임을 입증하였습니다.



### Multi-Source Hard and Soft Information Fusion Approach for Accurate Cryptocurrency Price Movement Prediction (https://arxiv.org/abs/2409.18895)
- **What's New**: 본 연구에서는 암호화폐 가격 예측의 정확성을 향상시키기 위한 새로운 접근 방식인 Hard and Soft Information Fusion (HSIF)를 소개합니다.

- **Technical Details**: HSIF 방식은 역사적 가격 기록과 기술적 지표를 포함하는 하드 정보와, X(구 트위터)에서 수집한 뉴스 제목 및 트윗 등의 소프트 정보를 결합합니다. 이 데이터는 기계 학습 모델인 Bidirectional Encoder Representations from Transformers (BERT) 기반의 감성 분석 방법인 Financial BERT (FinBERT)를 통해 처리됩니다. 마지막으로 처리된 하드 및 소프트 데이터를 기반으로 bidirectional long short-term memory (BiLSTM) 모델을 사용하여 장기 종속성을 포착합니다.

- **Performance Highlights**: 모델은 비트코인 관련 데이터를 테스트한 결과, 약 96.8%의 가격 변동 예측 정확도를 기록했습니다. 이 접근 방식은 소셜 감정의 영향을 확장하여 기술적 분석 예측을 보완함으로써, 단일 출처 데이터에 의존하는 기존 모델보다 우수하다는 것을 강조했습니다.



### Suicide Phenotyping from Clinical Notes in Safety-Net Psychiatric Hospital Using Multi-Label Classification with Pre-Trained Language Models (https://arxiv.org/abs/2409.18878)
Comments:
          submitted to AMIA Informatics Summit 2025 as a conference paper

- **What's New**: 이번 연구는 정신과 고위험 환경에서 자살 사건을 정확하게 식별하고 분류하여, 자살 예방 조치를 개선하고 운영 부담을 감소시키며 치료 품질을 향상시키는 방법을 제시합니다.

- **Technical Details**: 해당 연구에서는 두 가지 미세 조정 전략(단일 라벨 다수 및 단일 다중 라벨)을 사용하여 500개의 주석이 달린 정신과 평가 노트를 기반으로 네 가지 BERT(Bidirectional Encoder Representations from Transformers) 모델의 성능을 평가하였습니다. 노트는 자살적 사고(SI), 자살 시도(SA), 자살 노출(ES), 비자살적 자기 상해(NSSI)로 라벨링되었습니다.

- **Performance Highlights**: RoBERTa 모델이 binary relevance(이진 관련성) 방법을 사용하여 다른 모델보다 뛰어난 성능을 발휘하여 accuracy(정확도)가 0.86, F1 score가 0.78로 나타났습니다. MentalBERT는 F1 score가 0.74로 BioClinicalBERT의 0.72를 초과하였으며, 단일 다중 라벨 분류기로 미세 조정된 RoBERTa는 0.88의 정확도와 0.81의 F1 score로 성능이 더욱 향상되었습니다.



### CESNET-TimeSeries24: Time Series Dataset for Network Traffic Anomaly Detection and Forecasting (https://arxiv.org/abs/2409.18874)
- **What's New**: 이 논문은 네트워크 트래픽에서의 이상 탐지(anomaly detection)가 컴퓨터 네트워크의 보안을 유지하고 악의적인 활동을 식별하는 데 중요하다는 점을 강조합니다. 기존의 이상 탐지 기법에서 부족했던 실제 네트워크 데이터셋을 제공하여 이론과 실제의 간극을 메우고자 합니다.

- **Technical Details**: 저자는 CESNET3 네트워크에서 수집한 네트워크 엔티티의 행동에 대한 시계열(time series) 데이터로 구성된 데이터셋을 소개합니다. 이 데이터셋은 275,000개의 활성 IP 주소에서 40주 동안의 네트워크 트래픽 데이터를 포함하고 있습니다. ISP 출처는 네트워크 엔티티 간에 높은 수준의 변동성을 보장합니다.

- **Performance Highlights**: 이 데이터셋은 예측 기반(forecasting) 이상 탐지 접근법의 실제 적용에 대한 유용한 통찰력을 제공합니다. 다양한 네트워크 환경에서의 성능 평가에 기여할 수 있습니다.



### Individuation in Neural Models with and without Visual Grounding (https://arxiv.org/abs/2409.18868)
- **What's New**: 이 논문에서는 CLIP 모델과 FastText, SBERT와 같은 텍스트 전용 모델 간의 individuating (개별화) 정보 인코딩의 차이를 보여줍니다.

- **Technical Details**: CLIP 모델이 제공하는 latent representations (잠재 표현)을 연구하며, 기초(substrates), 미세한 집합(granular aggregates), 다양한 수의 객체에 대한 정보를 분석합니다.

- **Performance Highlights**: CLIP 임베딩은 텍스트 전용 데이터로 훈련된 모델들보다 individuating (개별화)에서 정량적 차이를 더 잘 포착하며, 이로부터 도출한 individuating hierarchy (개별화 계층)는 언어학 및 인지 과학에서 제안된 계층과 일치합니다.



### Positional Encoder Graph Quantile Neural Networks for Geographic Data (https://arxiv.org/abs/2409.18865)
Comments:
          17 main text pages, 4 figures

- **What's New**: 이번 논문에서는 Positional Encoder Graph Neural Networks (PE-GNNs)의 한계를 극복하기 위해 Positional Encoder Graph Quantile Neural Network (PE-GQNN)를 소개합니다. 이 방법은 PE-GNNs와 Quantile Neural Networks, 다시 조정 기술을 결합하여 예측 분포에 대한 최소한의 가정을 가지고 완전히 비모수적인 프레임워크를 제공합니다.

- **Technical Details**: PE-GQNN의 새로운 네트워크 아키텍처를 제안하며, quantile 기반 손실 함수를 결합하여 계산 복잡성을 증가시키지 않고도 정확하고 신뢰할 수 있는 확률적 모델을 생성합니다. 또한 KNN 예측기를 모델에 통합할 수 있는 구조화된 방법을 소개하며, GNN 레이어 연산을 통해 데이터 누수를 방지합니다.

- **Performance Highlights**: 벤치마크 데이터셋에 대한 실험 결과, PE-GQNN은 예측 정확도와 불확실성 정량화 모두에서 기존의 최고 수준의 방법들보다 상당한 개선을 나타냅니다.



### MECG-E: Mamba-based ECG Enhancer for Baseline Wander Remova (https://arxiv.org/abs/2409.18828)
Comments:
          7 pages, 5 figures

- **What's New**: 본 논문에서는 노이즈 조건이 심한 상황에서도 효과적으로 작동하는 새로운 ECG (Electrocardiogram) denoising 모델, Mamba 기반 ECG Enhancer (MECG-E)를 제안합니다.

- **Technical Details**: MECG-E 모델은 빠른 추론 속도와 뛰어난 비선형 매핑 능력으로 잘 알려진 Mamba 아키텍처를 활용합니다. 다양한 ECG denoising 방법들이 존재하지만, 기존 기법들은 매우 노이즈가 많은 조건에서 성능이 미흡하고, 추론 중 여러 단계를 요구하여 온라인 처리 시 지연 시간이 발생합니다.

- **Performance Highlights**: 실험 결과, MECG-E는 여러 지표에서 기존의 유명한 모델들을 초월하며, 노이즈 조건에 따른 다양한 상황에서도 우수한 성능을 보였습니다. 또한, MECG-E는 최첨단 diffusion 기반 ECG denoiser보다 추론 시간이 짧아 모델의 실용성과 효율성을 입증합니다.



### Early diagnosis of Alzheimer's disease from MRI images with deep learning mod (https://arxiv.org/abs/2409.18814)
Comments:
          7 pages, 3 figures, Presented at the 20-th CSI International Symposium on Artificial Intelligence and Signal Processing (AISP) 21-22 February, 2024, Mazandaran University of Science and Technology, Babol, Iran

- **What's New**: 이 연구에서는 알츠하이머병(Alzheimer's disease, AD) 진단을 위한 새로운 방법으로 합성 소수 샘플 오버샘플링 기술(SMOTE)과 사전 훈련된 CNN(Convolutional Neural Network)을 활용한 뇌 MRI(Magnetic Resonance Imaging) 분석 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 환자의 의료 기록, 신경심리학적 검사, MRI를 포함한 다양한 접근 방식을 통해 알츠하이머병의 특징을 식별합니다. 특히, Kaggle에서 얻은 이미지 데이터셋의 클래스 불균형 문제를 SMOTE를 통해 해결하였고, DEMNET라는 알츠하이머 네트워크에 사전 훈련된 CNN을 적용하여 중요한 특징들을 추출했습니다.

- **Performance Highlights**: 제안된 모델은 98.67%의 높은 정확도로 알츠하이머병 이미지를 분류하는 성능을 달성하였습니다.



### LLMs4Synthesis: Leveraging Large Language Models for Scientific Synthesis (https://arxiv.org/abs/2409.18812)
Comments:
          12 pages, 3 figures, Accepted to JCDL 2024 Research Track

- **What's New**: 이 논문은 과학 문헌의 복잡성과 양의 증가에 대응하기 위해 LLMs4Synthesis 프레임워크를 소개합니다. 이 프레임워크는 Large Language Models (LLMs)의 능력을 향상시키기 위해 설계되었습니다.

- **Technical Details**: LLMs4Synthesis 프레임워크는 개방형(open-source) 및 독점적(propriety) LLM을 활용하여 과학적 통합을 신속하고 일관되며 맥락이 풍부하게 수행하는 데 초점을 맞춥니다. 새로운 문서 처리 방법론을 개발하고, 새로운 통합 종류를 정의하며, 통합 평가를 위한 아홉 가지 품질 기준을 설정합니다.

- **Performance Highlights**: LLMs의 통합 강화를 위한 강화 학습(reinforcement learning) 및 AI 피드백을 제안하여 통합 품질을 최적화하고, 정립된 기준에 부합하도록 보장합니다. LLMs4Synthesis 프레임워크와 그 구성 요소는 과학 연구의 통합 생성 및 평가 프로세스를 향상시킬 것으로 기대됩니다.



### Esports Debut as a Medal Event at 2023 Asian Games: Exploring Public Perceptions with BERTopic and GPT-4 Topic Fine-Tuning (https://arxiv.org/abs/2409.18798)
- **What's New**: 이번 연구는 2023 아시안 게임에서의 e스포츠에 대한 대중의 의견과 가치 공동 창출(co-creation) 과정을 LLM-enhanced BERTopic 모델링 분석을 통해 조사하였습니다.

- **Technical Details**: 다섯 가지 주요 테마를 통해 대중의 인식과 주요 이해관계자들이 e스포츠 생태계 내외에서 가치를 공동 창출하는 방식을 식별했습니다. 소셜 미디어 마케팅의 전략적 사용이 대중의 의견에 영향을 미치고 e스포츠 이벤트 및 브랜드를 홍보하는 데 중요한 역할을 했습니다.

- **Performance Highlights**: e스포츠가 메달 이벤트로 포함되면서 보다 넓은 수용을 보여주었고, 부정적인 대중의 인식을 완화하는 데 기여했습니다. 전통적인 e스포츠 생태계 외부의 이해관계자들이 기여한 가치가 국가 대표 및 성과 증진에 중요한 역할을 했음을 발견했습니다. 이는 e스포츠를 스포츠로 정당화하는 지속적인 노력에 대한 지지를 나타냅니다.



### Supervised Learning Model for Key Frame Identification from Cow Teat Videos (https://arxiv.org/abs/2409.18797)
- **What's New**: 이 논문에서는 신경망(neural networks)과 비디오 분석(video analysis)을 이용하여 젖소의 유선 염증(티티스, mastitis) 위험 평가의 정확성을 향상시키는 방법을 제안합니다.

- **Technical Details**: 젖소의 티티스 감염을 탐지하기 위해, 저자들은 촬영된 비디오에서 유선이 온전하게 보이는 주요 프레임을 식별하는 신경망을 사용합니다. 이러한 주요 프레임은 수의사들이 유선 건강 상태를 평가할 시간적 유연성을 제공하며, 평가의 효율성과 정확성을 증가시킵니다. 복잡한 환경, 변화하는 소의 자세와 위치, 비디오에서 유선을 식별하는 어려움 등이 주요 도전과제로 제시됩니다.

- **Performance Highlights**: 제안된 방법은 유선 비디오에서 주요 프레임을 식별하는 데 있어 단일 거리 척도 또는 모델을 사용할 때보다 성능(F-score)이 향상된 것으로 나타났습니다.



### Hierarchical Federated ADMM (https://arxiv.org/abs/2409.18796)
- **What's New**: 본 논문에서는 널리 사용되는 gradient descent (경량 경사 하강법) 기반의 계층적 연합 학습 (hierarchical federated learning, FL) 알고리즘에서 벗어나, alternating direction method of multipliers (ADMM) 기반의 새로운 계층적 FL 프레임워크를 개발했습니다.

- **Technical Details**: 이 프레임워크 내에서 두 가지 새로운 FL 알고리즘을 제안합니다. 첫 번째 알고리즘은 상위 계층에 ADMM을 사용하고, 하위 계층에서도 ADMM을 사용하는 알고리즘과, 두 번째 알고리즘은 하위 계층에 전통적인 gradient descent 방법을 사용하는 알고리즘입니다.

- **Performance Highlights**: 제안된 알고리즘은 기존 알고리즘보다 학습 수렴성 (learning convergence) 및 정확도 (accuracy) 면에서 우수함을 실험을 통해 입증했습니다. 또한, 하위 계층에서의 gradient descent는 지역 단계 (local steps)가 매우 제한적일 경우에도 잘 작동하며, 두 계층에서 모두 ADMM을 사용할 경우 더 나은 성능을 보입니다.



### A Survey on the Honesty of Large Language Models (https://arxiv.org/abs/2409.18786)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문은 대형 언어 모델(LLMs)의 정직성(Honesty) 문제를 다루고 있습니다. 현재의 LLMs가 보이는 정직하지 않은 행동을 분석하고, 이와 관련된 다양한 정의와 평가 방법을 제시합니다.

- **Technical Details**: 정직성에 대한 정의가 다양하고, LLMs의 알고 있는 것과 모르는 것을 구분하는 데 어려움이 있으며, 이에 대한 종합적인 이해가 부족한 상황에서, 이 논문은 LLMs의 정직성을 평가하기 위해 여러 접근 방식을 탐색합니다.

- **Performance Highlights**: 논문은 LLMs의 정직성을 향상시키는 전략과 미래 연구 방향에 대한 통찰을 제공하여, 이 중요한 분야에서의 추가 탐색을 촉진하고자 합니다.



### HardCore Generation: Generating Hard UNSAT Problems for Data Augmentation (https://arxiv.org/abs/2409.18778)
- **What's New**: 이 논문에서는 SAT 문제의 핵심 기여자들을 식별하고 조작하는 새로운 접근법을 제안하고 있습니다. 기존의 데이터셋의 한계를 극복하기 위해 그래프 뉴럴 네트워크를 활용한 빠른 core detection 기법을 소개하였으며, 이를 통해 도전적인 SAT 문제를 보다 효율적으로 생성할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 'core'라고 알려진 문제의 'hardness'의 핵심 요소를 다루며, 기존의 전통적 휴리스틱(core detection techniques) 방법의 시간 비용을 절감하는 데 초점을 맞추고 있습니다. 이를 위해 그래프 뉴럴 네트워크(graph neural network)를 활용하여 빠른 core detection 절차를 개발하였습니다.

- **Performance Highlights**: 생성된 합성 SAT 문제들은 해결하기 어려운 문제를 유지하며, 원래 예제 문제의 주요 속성을 보존합니다. 실험을 통해 이러한 합성 SAT 문제들이 solver 런타임 예측의 개선에 기여할 수 있음을 입증하였습니다.



### State-of-the-Art Periorbital Distance Prediction and Disease Classification Using Periorbital Features (https://arxiv.org/abs/2409.18769)
Comments:
          16 pages, 4 figures, 4 tables

- **What's New**: 이번 연구에서는 눈과 눈꺼풀 주위의 거리와 특징들을 정량화하고 질병을 모니터링하는 데 있어서 중요한 정보를 제공하기 위해 3가지 딥러닝 방법을 개발했습니다. 이를 통해 수동 측정의 주관성과 시간 소모를 극복할 수 있습니다.

- **Technical Details**: 연구팀은 세 가지 딥러닝(segmentation) 방법을 사용하여 periorbital distance를 예측하였으며, 이를 통해 예측된 거리의 MAE는 훈련된 인간 주석자 간의 오류와 비슷하거나 그보다 낮았습니다. 현대 방식(SOTA)과 비교하여 모든 데이터셋에서 평균적으로 더 나은 성능을 보였습니다.

- **Performance Highlights**: 모델은 병든 눈에 대해 강력한 segmentation을 달성했으며, 건강한 눈을 사용하여 훈련된 모델에서도 효과적이었습니다. 또한, periorbital distances는 하위 분류(classification) 모델에서 고품질 특징으로 사용될 수 있음을 입증했습니다.



### OpenObject-NAV: Open-Vocabulary Object-Oriented Navigation Based on Dynamic Carrier-Relationship Scene Graph (https://arxiv.org/abs/2409.18743)
Comments:
          Project website: this https URL

- **What's New**: 이 논문은 로봇이 자주 사용되는 물체와 그 위치의 관계를 동적으로 업데이트할 수 있는 새로운 방법을 제시합니다. 이를 위해 Carrier-Relationship Scene Graph (CRSG)를 구축하고, 로봇 내비게이션 중 실시간으로 Carrying status를 반영합니다.

- **Technical Details**: CRSG는 여러 개의 객체와 그 객체를 가지고 있는 정적 캐리어 간의 관계를 캡처합니다. 내비게이션 과정은 Markov Decision Process로 모델링되며, 각 단계에서 Large Language Model의 상식 지식과 시각-언어 기능 유사성을 바탕으로 의사 결정을 진행합니다.

- **Performance Highlights**: Habitat 시뮬레이터에서 진행된 일련의 장기 시퀀스 내비게이션 테스트에서 로봇이 이동한 목표물로 효율적으로 내비게이션 할 수 있음을 입증했습니다. 이 알고리즘은 실제 로봇에 배포되어 실용적인 효과성을 검증했습니다.



### MemFusionMap: Working Memory Fusion for Online Vectorized HD Map Construction (https://arxiv.org/abs/2409.18737)
- **What's New**: 이 논문에서는 자율 주행 시스템을 위한 고해상도 (HD) 맵 구축에 개선된 시간적 추론 능력을 가진 MemFusionMap이라는 새로운 모델을 제안합니다.

- **Technical Details**: MemFusionMap은 역사적 프레임 간의 추론을 개선하는 작업 메모리 융합 모듈을 포함하며, 차량의 궤적 정보를 명시적으로 인지하도록 설계된 새로운 시간적 오버랩 히트맵을 도입합니다. 이러한 두 가지 설계를 통합하여 HD 맵 구축의 성능을 크게 향상시킵니다.

- **Performance Highlights**: MemFusionMap은 기존 방법들보다 최대 5.4% 높은 mAP (mean Average Precision)를 기록하여 성능이 크게 향상되었습니다.



### Multi-modal Medical Image Fusion For Non-Small Cell Lung Cancer Classification (https://arxiv.org/abs/2409.18715)
- **What's New**: 본 논문에서는 비소세포 폐암(Non-small cell lung cancer, NSCLC)의 조기 발견과 세분화된 하위 유형 분류를 위한 혁신적인 다중 모드 데이터 통합 방법을 제안합니다. 이는 CT 및 PET 스캔과 임상 건강 기록 및 유전체 데이터를 융합하는 새로운 접근 방식을 포함합니다.

- **Technical Details**: 우리는 MedClip 및 BEiT와 같은 고급 머신러닝(advanced machine learning) 모델을 활용하여, 이미지 특징 추출을 위한 정교한 방법을 개발했습니다. 이러한 다중 모드 분류기 모델은 94.04%의 정확도를 기록하며, 기존의 접근 방식보다 상당한 성능 향상을 보여줍니다.

- **Performance Highlights**: 우리의 연구는 NSCLC 검출 및 분류의 정확성, 정밀도, 재현율 및 F1 점수 등 주요 성능 지표에서 두드러진 개선을 나타냅니다. 이는 NSCLC 진단을 변화시킬 수 있는 잠재력을 가지고 있으며, 더욱 효과적인 치료 계획 및 결과 개선에 기여할 것입니다.



### Read Over the Lines: Attacking LLMs and Toxicity Detection Systems with ASCII Art to Mask Profanity (https://arxiv.org/abs/2409.18708)
- **What's New**: 본 논문에서는 언어 모델이 ASCII 아트를 해석하지 못하는 것을 이용한 새로운 형태의 adversarial attacks를 소개합니다. 이러한 공격을 평가하기 위한 ToxASCII benchmark를 제안합니다.

- **Technical Details**: 연구에서는 두 개의 맞춤형 ASCII 아트 폰트를 개발하였으며, 하나는 special tokens를 활용하고 다른 하나는 텍스트로 채워진 문자 형태를 사용합니다. 이 공격들은 OpenAI의 o1-preview 및 LLaMA 3.1 포함 총 10개의 모델에서 1.0의 공격 성공률(Attack Success Rate)을 기록하였습니다.

- **Performance Highlights**: 이 논문은 연구 목적으로 사용된 유해한 언어의 예를 포함하고 있기 때문에 주의가 필요합니다.



### Speech Boosting: Low-Latency Live Speech Enhancement for TWS Earbuds (https://arxiv.org/abs/2409.18705)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 이 논문은 진정한 무선 스테레오(TWS) 이어버드에서 사용하기 위한 음성 향상 솔루션을 소개합니다. 이 솔루션은 소음이 많은 환경에서 대화를 지원하도록 특별히 설계되었습니다.

- **Technical Details**: 주요한 설계 요소로는 네트워크 아키텍처(network architecture)와 도메인(domain), 손실 함수(loss functions)의 설계, 프루닝(pruning) 방법, 하드웨어 특화 최적화(hardware-specific optimization)가 포함됩니다. 음성 향상 모델의 계산 복잡성을 감소시키고, 3ms 이하의 지연(lag)을 유지하여 실시간 대화를 보장하는 데 중점을 두었습니다.

- **Performance Highlights**: 기존의 기준(baseline) 모델들과 비교했을 때, 음성 향상 품질에서 상당한 개선을 보이며 계산 복잡성과 알고리즘 지연을 동시에 줄였습니다.



### Learning from Pattern Completion: Self-supervised Controllable Generation (https://arxiv.org/abs/2409.18694)
- **What's New**: 이번 논문에서는 인공지능(AI) 분야에서 일반적으로 사용되는 레이블이 있는 학습 데이터 세트에 의존하지 않고 스스로 조절 가능한 생성(self-supervised controllable generation) 방법론을 제안합니다. 이러한 방법은 인간 뇌의 연상 작용을 모방한 것으로, 기능적 전문화를 이루는 모듈화된 오토인코더(modular autoencoder) 네트워크를 통해 이루어집니다.

- **Technical Details**: 제안된 프레임워크인 SCG(self-supervised controllable generation)는 모듈 내 독립성(intra-module independence)과 모듈 간 상관관계(inter-module correlation)를 촉진하기 위해 동등 변환 제약조건(equivariant constraint)을 도입합니다. 이로 인해 색상, 명도, 엣지 검출 모듈 처리에서 기능적 전문성을 확보합니다. 또한, 자기 감독형 패턴 완성(self-supervised pattern completion) 접근을 통해 학습을 진행합니다.

- **Performance Highlights**: 실험 결과, SCG는 색조, 명도, 엣지 검출의 모듈 처리에서 뛰어난 기능적 전문화를 보여주며, 인간 뇌와 유사한 방향 선택성(orientation selectivity), 색상 대립(color antagonism), 중심-주변 수용 영역(center-surround receptive fields) 특징을 나타냅니다. SCG는 페인팅(다시 그린 그림), 스케치, 고대 그래피티와 같은 다양한 작업에서 뛰어난 일반화 능력을 보여줍니다. 기존의 ControlNet과 비교할 때, SCG는 더 높은 노이즈 환경에서도 우수한 강건성(robustness)을 보이며, 자기 감독형 학습 덕분에 향후 더 나은 확장 가능성(scalability) 잠재력을 지니고 있습니다.



### MG-Net: Learn to Customize QAOA with Circuit Depth Awareness (https://arxiv.org/abs/2409.18692)
Comments:
          29 pages, 16 figures

- **What's New**: 이번 논문에서는 Quantum Approximate Optimization Algorithm (QAOA)와 그 변형들이 조합 최적화(combinatorial optimization) 문제를 해결하는 데 가진 잠재력을 분석합니다. 특히, QAOA의 수렴(convergence) 행동을 연구하고, 문제 특유의 회로 깊이(circuit depth)와 관련된 딜레마를 다룹니다.

- **Technical Details**: 우리는 Mixer Generator Network (MG-Net)라는 새로운 심층 학습(deep learning) 프레임워크를 제안합니다. MG-Net은 특정 작업(task)과 회로 깊이에 따라 최적의 mixer Hamiltonians를 동적으로 생성하는 기능을 가지고 있습니다. 이 네트워크는 Ising 모델과 최대 컷(maximum cut) 문제와 같은 여러 시뮬레이션을 통해 성능을 입증했습니다.

- **Performance Highlights**: MG-Net은 64 큐비트(qubits)까지의 위상별(Max-Cut instances)와 Ising 모델에서 높은 근사 비율(approximation ratio)과 효율성(efficiency)을 보여줍니다.



### Beyond Single-Audio: Advancing Multi-Audio Processing in Audio Large Language Models (https://arxiv.org/abs/2409.18680)
Comments:
          EMNLP24 Findings

- **What's New**: 본 논문에서는 여러 오디오 작업을 동시에 처리할 수 있는 첫 번째 multi-audio evaluation (MAE) 벤치마크를 제안합니다. 이 벤치마크는 11개 멀티 오디오 작업에서 수집한 20개의 데이터셋으로 구성되어 있습니다.

- **Technical Details**: 기존의 audio-LLMs (ALLMs)는 단일 오디오 작업의 평가에 주로 초점을 맞추었으나, 실제 애플리케이션에서는 여러 오디오 스트림을 동시에 처리하는 경우가 많습니다. 우리는 synthetic data를 활용하여 multi-audio-LLM (MALLM)을 제안하며, 이를 통해 여러 유사한 오디오 간의 음향 맥락을 포착합니다.

- **Performance Highlights**: MALLM은 기존의 모든 기준선 모델 대비 뛰어난 성능을 보여주며, 인간 주석 없이도 높은 데이터 효율성을 달성했습니다. 이는 ALLMs가 멀티 오디오 처리 시대에 접어드는 데 중요한 이정표가 될 것입니다.



### Exploiting Motion Prior for Accurate Pose Estimation of Dashboard Cameras (https://arxiv.org/abs/2409.18673)
- **What's New**: 본 연구는 대시캠(dashcam) 이미지를 위한 정밀한 자세 추정(pose estimation) 방법을 제안합니다. 이를 통해 카메라의 고유한 모션 프라이어(camera motion prior)를 활용하여 기존 이미지 매칭 방법의 한계를 극복하고자 합니다.

- **Technical Details**: 대시캠으로 캡처된 이미지 시퀀스는 일반적으로 전방 이동(forward movement)이나 측면 회전(lateral turns)과 같은 뚜렷한 모션 프라이어를 나타냅니다. 본 연구는 이를 기반으로 카메라 모션 프라이어를 학습하는 포즈 회귀 모듈(pose regression module)을 개발하고, 이를 관계 추정(correspondence estimation) 및 자세 추정 과정에 통합했습니다.

- **Performance Highlights**: 실제 대시캠 데이터셋에서 우리의 방법은 AUC5°에 대한 포즈 추정에서 기준선(baseline)보다 22% 향상된 성능을 보였으며, 재투영 오류(reprojection error)가 적은 이미지를 19% 더 많이 추정할 수 있었습니다.



### Effects of AI Feedback on Learning, the Skill Gap, and Intellectual Diversity (https://arxiv.org/abs/2409.18660)
- **What's New**: 이번 연구에서는 대규모 온라인 체스 플랫폼에서 52,000명의 의사결정자의 데이터를 사용하여 AI 피드백이 학습, 기술 격차(skill gap), 그리고 의사결정 전략의 다양성에 미치는 영향을 탐구합니다.

- **Technical Details**: 연구에 따르면, 개인은 실패보다 성공 경험이 있는 상황에서 AI 피드백을 찾을 가능성이 더 높습니다. 그러나 성공에 대한 피드백은 미래 성과를 감소시키고, 실패에 대한 피드백은 성과를 증가시키는 것으로 나타났습니다. 또한, 고숙련 의사결정자는 AI 피드백을 더 자주 구하며, 실패 후에 피드백을 찾는 경향이 더 강하고, 저숙련자들보다 피드백에서 더 큰 혜택을 봅니다.

- **Performance Highlights**: AI 피드백 접근성이 오히려 고숙련자와 저숙련자 간의 기술 격차를 증가시키고, 42개의 주요 플랫폼 업데이트를 통해 AI 피드백 접근성 증가가 지적 다양성의 감소로 이어질 수 있음을 보여줍니다. 이는 AI 피드백으로부터 학습하는 것이 자동적이지 않으며, AI를 올바르게 사용하는 것이 자체적으로 하나의 기술임을 나타냅니다.



### When SAM2 Meets Video Camouflaged Object Segmentation: A Comprehensive Evaluation and Adaptation (https://arxiv.org/abs/2409.18653)
Comments:
          Technical report

- **What's New**: 이번 연구는 Segment Anything Model 2 (SAM2)이 비디오에서 위장된 객체 세분화(video camouflaged object segmentation, VCOS) 작업에서 어떻게 활용될 수 있는지를 조사합니다. VCOS는 색상과 질감이 비슷하고 조명이 좋지 않은 환경에서 객체를 감지하는 어려운 과제입니다.

- **Technical Details**: 연구에서는 SAM2의 성능을 다양한 모델과 프롬프트(클릭, 박스, 마스크)를 사용하여 위장된 비디오 데이터셋에서 평가하였습니다. 또한, 기존의 다중 모달 대형 언어 모델(multimodal large language models, MLLMs) 및 VCOS 방법과 SAM2의 통합을 탐구하였습니다. SAM2를 비디오 위장 데이터셋에 맞추어 세밀하게 조정(fine-tuning)하여 적용하였습니다.

- **Performance Highlights**: SAM2는 비디오에서 위장된 객체를 탐지하는 매우 우수한 제로샷(zero-shot) 능력을 보여주었습니다. 또한, VCOS에 맞게 SAM2의 파라미터를 조정함으로써 이 능력을 더욱 향상시킬 수 있음을 입증하였습니다.



### Enhanced Convolution Neural Network with Optimized Pooling and Hyperparameter Tuning for Network Intrusion Detection (https://arxiv.org/abs/2409.18642)
Comments:
          7 Pages , 2 figures , 4 Tables , Conference paper

- **What's New**: 이번 연구에서는 네트워크 침입 탐지 시스템(Network Intrusion Detection Systems, NIDS)을 위한 향상된 합성곱 신경망(Enhanced Convolutional Neural Network, EnCNN)을 제안합니다. 이 방법론은 KDDCUP'99 데이터셋을 사용하여 성능을 평가하고, 기존의 방법에 비해 10% 이상의 정확도 향상을 보여주었습니다.

- **Technical Details**: 연구에서는 데이터 전처리(data preprocessing), 탐색적 데이터 분석(exploratory data analysis, EDA), 기능 공학(feature engineering)을 포함하는 포괄적인 방법론을 사용하였습니다. EnCNN의 성능을 로지스틱 회귀(Logistic Regression), 결정 트리(Decision Trees), 서포트 벡터 머신(Support Vector Machines, SVM), 랜덤 포레스트(Random Forest), 아다부스트(AdaBoost), 투표 앙상블(Voting Ensemble) 등 다양한 기계 학습 알고리즘과 비교했습니다.

- **Performance Highlights**: EnCNN은 기존 최첨단 접근 방식에 비해 10%의 정확도 향상을 이루었습니다. 이는 실시간 네트워크 침입 탐지에서 EnCNN의 효과성을 보여주며, 보안 위협을 식별 및 완화하고 전체 네트워크의 복원력을 강화하는 강력한 솔루션을 제공함을 의미합니다.



### Quantum Algorithms for Drone Mission Planning (https://arxiv.org/abs/2409.18631)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 논문은 ISR (Intelligence, Surveillance and Reconnaissance) 자산을 활용하여 드론(UAV)이 여러 목표를 방문하는 임무 계획을 최적화하는 방법을 제안합니다. 특히 새로운 제한사항과 목표가 발생할 때 신속하게 해결책을 찾는 데 중점을 두었습니다.

- **Technical Details**: 이 논문에서는 다수의 문제를 Mixed Integer Linear Program (MILP)으로 공식화한 후, 이를 Quadratic Unconstrained Binary Optimisation (QUBO)로 변환하는 과정을 다룹니다. 제안된 공식화는 다양한 제한사항에 맞게 조정 가능하며, qubit의 확장이 명확하게 제공됩니다.

- **Performance Highlights**: 상업용 양자 어닐러(quantum annealer)를 사용하여 QUBO 공식화를 해결한 결과를 현재의 edge classical solvers와 비교했습니다. 또한 QAOA (Quantum Approximate Optimisation Algorithms)를 통해 문제를 해결한 결과도 분석하였고, 그러한 결과 또한 논의하였습니다. 마지막으로, Variational Quantum Eigensolver (VQE) 형식으로 문제를 효율적으로 인코딩하는 방법과 qubit을 효과적으로 활용하기 위한 ansatz를 조정한 방법을 제시하였습니다.



### Entropy, concentration, and learning: a statistical mechanics primer (https://arxiv.org/abs/2409.18630)
- **What's New**: 이 논문은 손실 최소화(loss minimization)를 통해 훈련된 인공지능 모델이 정보 이론(information theory)과 통계 물리학(statistical physics)에서 유래된 원칙에 기반하여 성공적으로 작동함을 보여줍니다. 특히, 통계 역학(statistical mechanics)의 관점에서 이러한 연결을 탐구합니다.

- **Technical Details**: AI 및 머신러닝(ML) 기반의 샘플 농도(sample concentration) 행동을 설명하기 위해 기본 원리(first-principles)에서 시작하여, 통계 역학의 발전이 exponential families의 중요성을 강조하고, 통계(statistics), 물리학(physics), 정보 이론의 양(quantity)과의 관계를 정립하고 있음을 설명합니다.

- **Performance Highlights**: 이 연구는 통계 역학을 활용해 AI 모델링 및 손실 최소화 방법에 대한 새로운 통찰을 제공하며, 기존의 원리를 통합하여 성능 향상을 도모할 수 있는 가능성을 제시합니다.



### Towards Integrating Epistemic Uncertainty Estimation into the Radiotherapy Workflow (https://arxiv.org/abs/2409.18628)
Comments:
          Keywords: Epistemic Uncertainty - Out-of-Distribution Detection - CT Segmentation - OAR contouring - Radiotherapy

- **What's New**: 본 연구는 방사선 치료 계획에서의 목표 구조물 및 위험 장기(Organ-at-Risk, OAR)의 윤곽을 정확히 지정하는 것의 중요성을 강조하며, 최근 딥러닝(deep learning)의 발전을 통해 OAR 윤곽화 성능을 개선했으나, OOD(out-of-distribution) 시나리오의 신뢰성 문제를 다루고 있다. 특히, epistemic uncertainty(계량적 불확실성) 추정 통합을 통해 OOD 감지를 위한 새로운 방법론을 제시한다.

- **Technical Details**: 연구에서는 OAR 윤곽화 워크플로우에 epistemic uncertainty estimation을 통합하여 임상적으로 관련 있는 시나리오에서 OOD를 감지하는 방법을 제안한다. 구체적으로 설계된 데이터 세트를 사용하여 OOD 감지를 위한 고급 통계적 방법을 도입하고, 예측의 신뢰성이 떨어지는 사례를 식별하는 데 효과적임을 입증한다.

- **Performance Highlights**: 이 연구에서 제안한 접근법은 OOD 감지를 위한 AUC-ROC(Area Under the Curve - Receiver Operating Characteristic)를 0.95로 달성하였으며, 임플란트 사례에 대한 특이도(specificity) 0.95 및 민감도(sensitivity) 0.92를 기록하였다. 이는 모델 예측의 신뢰성을 판단하는 데 있어 전문가의 검토가 필요한 경우를 효과적으로 표시하는 데 기여한다.



### Model-based Preference Optimization in Abstractive Summarization without Human Feedback (https://arxiv.org/abs/2409.18618)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 본 연구에서는 인간 피드백 없이 모델의 요약 능력을 향상시키기 위한 새로운 접근 방식인 Model-based Preference Optimization (MPO)을 소개합니다.

- **Technical Details**: MPO는 다양한 decoding strategies를 활용하여 모델이 생성한 preference dataset을 기반으로 LLMs를 미세 조정합니다. 기존의 Direct Preference Optimization (DPO) 방식과는 달리, MPO는 비싼 인간 피드백에 의존하지 않습니다.

- **Performance Highlights**: MPO를 적용한 결과, 표준 요약 데이터셋에서 생성된 요약의 품질이 크게 향상되었습니다. 다양한 평가 지표에서 성능 개선이 나타났습니다.



### TemporalPaD: a reinforcement-learning framework for temporal feature representation and dimension reduction (https://arxiv.org/abs/2409.18597)
- **What's New**: TemporalPaD는 시계열 데이터셋을 위한 새로운 end-to-end 딥러닝 프레임워크로, 강화 학습(reinforcement learning)과 신경망(neural networks)을 통합하여 특성 표현(feature representation) 및 특성 감소(feature reduction)를 동시에 진행할 수 있도록 설계되었습니다.

- **Technical Details**: TemporalPaD는 Actor-Critic(AC) 프레임워크를 기반으로 하는 세 가지 협력 모듈인 Policy Module(정책 모듈), Representation Module(표현 모듈), Classification Module(분류 모듈)로 구성됩니다. 정책 모듈은 RL을 통해 차원 축소를 담당하고, 표현 모듈은 특성을 추출하며, 분류 모듈은 비평가 역할을 합니다.

- **Performance Highlights**: TemporalPaD는 29개의 UCI 데이터셋을 사용하여 10회 독립 테스트 및 10겹 교차 검증을 통해 종합적으로 평가되었습니다. 또한, 실제 DNA 분류 문제에 적용하여 우수한 성능을 입증하였습니다. 이 프레임워크는 구조화된 데이터와 시퀀스 데이터셋 모두에 적용 가능합니다.



### Analysis of Truncated Singular Value Decomposition for Koopman Operator-Based Lane Change Mod (https://arxiv.org/abs/2409.18586)
Comments:
          Submitted to the 21st International Conference on Informatics in Control, Automation and Robotics (ICINCO 2024)

- **What's New**: 자동 운전(context of autonomous driving)에서 차량 성능 및 안전성을 향상시키기 위한 복합 동적 시스템 이해 및 모델링의 중요성에 대한 연구가 진행되었습니다. 특히, 최근에 강조된 방법인 Koopman operators와 그 근사 방식인 Extended Dynamic Mode Decomposition (EDMD)에 대한 연구가 포함되어 있습니다.

- **Technical Details**: 이 연구에서는 Koopman operators로부터 대규모 데이터 세트를 효율적으로 근사하기 위해 특히 truncated SVD(특이값 분해)를 사용합니다. EDMD에서 사용되는 다양한 basis functions의 평가가 진행되며, lane change behavior 모델을 나타내기 위한 truncated SVD의 순위도 매겨집니다. 이 과정의 목적은 계산 효율성(computational efficiency)과 정보 손실(information loss) 간의 균형을 맞추는 것입니다.

- **Performance Highlights**: 연구 결과에 따르면, truncated SVD 기법이 계산 교육 시간(computational training time)을 크게 줄이지 못하고, 상당한 정보 손실을 초래한다는 점이 발견되었습니다.



### An Enhanced Federated Prototype Learning Method under Domain Shif (https://arxiv.org/abs/2409.18578)
Comments:
          8 pages, 6 figures

- **What's New**: 이 논문은 Federated Learning (FL)에서 서로 다른 도메인에서 샘플링된 데이터의 이질성(heterogeneity)이 모델 성능에 미치는 영향을 해결하기 위해 새로운 방식을 제안합니다. 특히, variance-aware dual-level prototype clustering을 도입하고 $
abla$-sparsity prototype loss를 사용하여 intra-class similarity와 inter-class similarity를 조절합니다.

- **Technical Details**: 제안된 알고리즘은 Federated Prototype Learning with Convergent Clusters (FedPLCC)로 명명되며, 클러스터 내에서 특징이 수렴하도록 개선되었습니다. 클러스터의 크기에 따라 각 프로토타입에 가중치를 부여하여 inter-class 거리를 증가시키고, 서로 다른 도메인에서 오는 프로토타입의 거리를 줄이기 위해 손실 함수 계산을 위해 일정 비율의 프로토타입만 선택합니다.

- **Performance Highlights**: Digit-5, Office-10 및 DomainNet 데이터셋에서 평가한 결과, 제안된 방법은 기존 접근 방식에 비해 더 나은 성능을 보였습니다.



### Efficient Noise Mitigation for Enhancing Inference Accuracy in DNNs on Mixed-Signal Accelerators (https://arxiv.org/abs/2409.18553)
- **What's New**: 본 논문에서는 아날로그 신경망의 정확도에 미치는 프로세스 유도 및 노화 관련 변동의 영향을 완화하여 신경 모델의 강인성을 향상시키는 프레임워크를 제안합니다.

- **Technical Details**: 변동은 활성화의 정밀도에 영향을 미치는 노이즈로 모델링되며, 사전 훈련된 모델의 선택된 레이어 사이에 삽입된 디노이징 블록(denoising block)을 소개합니다. 디노이징 블록을 훈련하여 다양한 노이즈 수준에 대한 모델의 강인성을 크게 증가 시킬 수 있음을 입증하였습니다. 디노이징 블록 추가로 인한 오버헤드를 최소화하기 위해 최적의 삽입 지점을 식별하는 탐색 알고리즘을 제시하고, 혼합 신호 가속기에 통합할 수 있는 효율적인 디노이징 블록 아키텍처를 제안합니다.

- **Performance Highlights**: DNN 모델을 ImageNet 및 CIFAR-10 데이터셋에서 훈련하여 접근 방식을 평가한 결과, 평균적으로 2.03%의 파라미터 카운트 오버헤드를 수용함으로써 변동으로 인한 정확도 감소가 31.7%에서 1.15%로 줄어드는 것을 보여주었습니다.



### Research on Predicting Public Opinion Event Heat Levels Based on Large Language Models (https://arxiv.org/abs/2409.18548)
Comments:
          conference

- **What's New**: 최근 몇 년간 큰 언어 모델(large language models)의 급속한 발전으로 인해, GPT-4o와 같은 여러 모델이 언어 작업에서 인간의 성능을 초월하는 탁월한 능력을 보여주었습니다. 이 연구는 공공 여론 분석 분야에서의 잠재적 응용을 탐구합니다.

- **Technical Details**: 이 연구에서는 2022년 7월부터 2023년 12월 사이에 수집된 62,836개의 중국 핫 이벤트 데이터를 전처리하고 분류했습니다. 각 이벤트의 온라인 확산 열 지수를 기반으로 MiniBatchKMeans 알고리즘을 사용하여 이벤트를 자동으로 클러스터링하고 네 가지 열 수준으로 분류했습니다. 이후 각 열 수준에서 250개의 이벤트를 랜덤으로 선택하여 총 1,000개의 평가 데이터셋을 구축했습니다.

- **Performance Highlights**: 평가 과정에서 다양한 큰 언어 모델을 사용하여 두 가지 시나리오(참조 사례 없는 경우와 유사한 사례 참조가 있는 경우)에서 이벤트 열 수준 예측의 정확성을 평가했습니다. 결과적으로 GPT-4o와 DeepseekV2가 후자의 경우 최고의 성능을 보이며 각각 41.4%와 41.5%의 예측 정확도를 기록했습니다. 특히 저온 이벤트(Level 1)의 경우 두 모델의 예측 정확도는 각각 73.6%와 70.4%에 달했습니다. 전체적인 예측 정확도는 열 수준이 높아질수록 감소하는 경향을 보였습니다.



### An Epistemic Human-Aware Task Planner which Anticipates Human Beliefs and Decisions (https://arxiv.org/abs/2409.18545)
Comments:
          15 pages, 4 figures, 1 table

- **What's New**: 이번 연구에서는 인간과 로봇 간의 신뢰도 차이가 큰 상황에서 단체 행동(shared execution experience)이 간헐적으로 이루어지는 경우를 위한 Human-Aware Task Planning 프레임워크를 확장하였습니다. 특히, 통제할 수 없는 인간 행동을 고려하여 로봇 정책을 구축하는 것이 목표입니다.

- **Technical Details**: 제안된 새로운 계획 프레임워크는 AND-OR 검색 기반의 솔버(solver)를 기반으로 하며, 상황 평가(situation assessment) 및 관점 취득(perspective taking)을 포함한 지식 추론(knowledge reasoning)을 통합합니다. 이 시스템은 잠재적 진전을 동적으로 모델링하고 관리하며, 에이전트가 작업 실행 경험을 공유할 때와 공유하지 않을 때를 정확히 추적합니다.

- **Performance Highlights**: 초기 실험은 새로운 도메인과 적응된 도메인에서 수행되었으며, 프레임워크의 효과를 입증하였습니다. 새로운 솔버는 로봇과 인간의 다르게 믿는 맥락을 추정할 수 있어, 로봇이 적절한 시점에 소통(communication)하거나 작업 수행을 연기할 수 있도록 계획을 합성(synthesis)할 수 있게 합니다.



### MIMII-Gen: Generative Modeling Approach for Simulated Evaluation of Anomalous Sound Detection System (https://arxiv.org/abs/2409.18542)
- **What's New**: 본 논문에서는 기계 소음에서의 이상 탐지(anomaly detection) 시스템 개발을 위한 새로운 접근 방식을 제안합니다. 특히, 레이턴트 확산(latent diffusion) 기반 모델을 사용하여 다양한 이상(anomalies)을 생성함으로써 기존의 데이터 부족 문제를 해결하고자 하였습니다.

- **Technical Details**: 제안된 방법에서는 Flan-T5 모델을 활용하여 오디오 파일 메타데이터에서 파생된 캡션을 인코딩하고, U-Net 아키텍처를 사용하여 조건부 생성(conditional generation)을 수행합니다. 이는 EnCodec 레이턴트 공간(latent space) 내에서 오디오 신호를 생성하여 높은 맥락적 관련성(contextual relevance)과 품질을 보장합니다.

- **Performance Highlights**: 제작된 소리의 품질은 Fréchet Audio Distance (FAD) 점수와 기타 메트릭스를 통해 객관적으로 평가되었으며, 기존 모델들을 초월하는 성능을 보였습니다. 또한, 생성된 데이터를 활용한 이상 탐지 시스템의 평가 결과 AUC(Area Under Curve) 점수가 원본 데이터와 4.8% 차이를 보이며, 생성된 데이터의 효과성을 입증하였습니다.



### EmoPro: A Prompt Selection Strategy for Emotional Expression in LM-based Speech Synthesis (https://arxiv.org/abs/2409.18512)
- **What's New**: 최근 음성 합성 모델의 발전으로, 방대한 데이터셋을 활용한 제로샷(zero-shot) 능력의 향상이 두드러진다. 이런 발전에도 불구하고, 프롬프트(prompt)의 선택이 음성 생성의 품질에 중대한 영향을 미친다는 점에 주목하였다.

- **Technical Details**: 이 논문에서는 감정 조절이 가능한 음성 합성을 위한 두 단계의 프롬프트 선택 전략인 EmoPro를 제안한다. 이 전략은情緒 표현 강도(emotional expression strength), 음성 품질(speech quality), 텍스트-감정 일관성(text-emotion consistency), 모델 생성 성능(model generation performance)이라는 네 가지 관점에서 프롬프트를 평가하여 선택한다.

- **Performance Highlights**: 실험 결과, 제안된 방법으로 선택된 프롬프트는 기본 참조 모델(baseline)으로 얻은 것보다 더 감정적으로 표현력이 풍부하고 매력적인 합성 음성을 생성하는 것으로 나타났다. 오디오 샘플과 코드가 제공될 예정이다.



### Fairness-aware Multiobjective Evolutionary Learning (https://arxiv.org/abs/2409.18499)
Comments:
          14 pages

- **What's New**: 이 논문은 Multiobjective evolutionary learning (MOEL)에서 모델 훈련 중 동적으로 공정성 측정치의 대표 세트를 결정하는 방법을 제안합니다. 이는 이전에 정해진 정적 세트 대신 훈련 중에 적응적으로 변경될 수 있습니다.

- **Technical Details**: 기존의 MOEL 접근법은 데이터셋과 사전 지식에 의존하고 상당한 계산 비용이 요구되며, 공정성 측정 지표가 모델 훈련 과정에 따라 다를 수 있습니다. 본 연구에서는 12개의 잘 알려진 벤치마크 데이터셋에서 실험을 수행하여, 동적으로 결정된 공정성 측정 세트를 최적화 목표로 사용하며, 이는 훈련 과정 중에 시간에 따라 변화할 수 있습니다.

- **Performance Highlights**: 제안된 MOEL 프레임워크는 정확도와 25개의 공정성 측정을 포함하여 불공정성을 완화하기 위한 최신 방법들과 비교하여 뛰어난 성능을 보였습니다. 이러한 결과는 훈련 중 최적화 목표를 동적으로 설정하는 것이 중요함을 강조합니다.



### Towards Diverse Device Heterogeneous Federated Learning via Task Arithmetic Knowledge Integration (https://arxiv.org/abs/2409.18461)
Comments:
          NeurIPS 2024

- **What's New**: 이번 논문에서는 Fedrated Learning(Federated Learning, FL)의 한계를 극복하기 위해 TAKFL이라는 새로운 KD 기반 프레임워크를 제안합니다. 이 프레임워크는 다양한 이질적인 장치 모델의 지식 전이를 독립적으로 수행하여, 각 장치의 고유한 기여를 보존합니다.

- **Technical Details**: TAKFL은 각 장치 프로토타입의 앙상블에서 지식 전이를 별도의 작업으로 처리하며, 각 장치의 정보를 효과적으로 증류할 수 있도록 설계되었습니다. 또한, KD 기반의 self-regularization 기법을 도입하여 noise와 비지도 앙상블 증류 과정에서 발생하는 문제를 완화시킵니다.

- **Performance Highlights**: TAKFL은 컴퓨터 비전(CV) 및 자연어 처리(NLP) 작업에서 종합적인 평가를 수행하였으며, 다양한 데이터셋과 설정에서 SOTA(State Of The Art) 결과를 달성하여 기존 KD 기반 방법들보다 월등한 성능을 보였습니다.



### Review of Digital Asset Development with Graph Neural Network Unlearning (https://arxiv.org/abs/2409.18455)
- **What's New**: 이 논문은 디지털 자산 분야에서 Graph Neural Networks (GNNs)의 역할과 이에 맞춘 혁신적인 unlearning 기술을 소개합니다. 특히, 데이터 개인 정보 보호와 규제 준수의 필요성이 증가하는 배경에서 이러한 기술이 중요하다는 점을 강조합니다.

- **Technical Details**: 이 연구에서는 unlearning 전략을 데이터 기반 근사(data-driven approximation)와 모델 기반 근사(model-driven approximation)로 두 가지 주요 클래스로 분류합니다. 각각의 접근 방법은 특정 노드의 영향을 제거하기 위한 그래프 구조 변경과 GNN 내부 파라미터 및 아키텍처 수정 등을 포함합니다.

- **Performance Highlights**: 이 논문에서 제안된 방법은 사기 탐지, 위험 평가, 토큰 관계 예측 및 분산 거버넌스와 같은 다양한 사용 사례에서 효율적입니다. 또한, 실시간 금융 애플리케이션에서 모델 성능과 데이터 unlearning 요구사항 간의 균형을 맞추는 데 있어 직면하는 도전 과제를 논의하며, 두 가지 unlearning 전략의 장점을 결합한 하이브리드 접근 방식을 제안하여 GNN의 효율성과 효과성을 향상시키고자 합니다.



### Leveraging Long-Context Large Language Models for Multi-Document Understanding and Summarization in Enterprise Applications (https://arxiv.org/abs/2409.18454)
- **What's New**: 이 논문은 다양한 분야에서 비구조화 데이터의 급증에 따른 다중 문서 이해 및 요약의 중요성을 강조합니다. 전통적인 접근 방식이 정보의 맥락을 잘 잡지 못하고 논리적 일관성을 유지하지 못하는 문제를 다루며, Long-context Large Language Models (LLMs)의 사용을 탐구합니다.

- **Technical Details**: 본 연구는 다중 문서 요약을 효과적으로 수행하기 위한 Long-context LLM의 워크플로우를 설명하며, 법률, 인사(HR), 재무, 소싱과 같은 기업 기능, 의료 및 뉴스 도메인에서의 사례 연구를 다룹니다. 이러한 사례 연구는 효율성과 정확성 모두에서 향상을 보여줍니다.

- **Performance Highlights**: 논문은 데이터셋의 다양성, 모델 확장성 및 편향 완화, 사실 정확성과 같은 윤리적 고려사항과 같은 기술적 장애물에 대한 철저한 분석을 포함하고 있으며, LLM의 기능과 응용 프로그램을 증강하기 위한 미래 연구 방향을 제시합니다.



### State-free Reinforcement Learning (https://arxiv.org/abs/2409.18439)
- **What's New**: 이 연구에서는 	extit{상태 없는 RL} (state-free RL) 문제를 다루며, 알고리즘이 환경과 상호작용하기 전에 상태 정보가 없음을 제시합니다.

- **Technical Details**: 우리는 도달할 수 있는 상태 집합 ${S}^	ext{Π} := igl	m{s | 	ext{max}_{	ext{π} 	ext{in Π}} q^{P, 	ext{π}}(s) > 0} igr	m$를 정의하고, 상태 공간 $S$에 대한 정보 없이도 작동하는 알고리즘을 설계했습니다. 이 알고리즘의 후회(regret)는 ${S}$와 완전히 독립적이며 오로지 ${S}^	ext{Π}$에만 의존합니다.

- **Performance Highlights**: 이 연구는 하이퍼 파라미터 조정이 필요 없는 	extit{파라미터 없는 RL} (parameter-free RL) 알고리즘 설계의 구체적인 첫 걸음으로 볼 수 있습니다.



### Multi-agent Reinforcement Learning for Dynamic Dispatching in Material Handling Systems (https://arxiv.org/abs/2409.18435)
- **What's New**: 본 논문은 다중 에이전트 강화 학습(MARL) 접근 방식을 통해 동적 디스패칭(dynamic dispatching) 전략을 학습하는 방법을 제안합니다. 이 연구는 다양한 산업에서 자재 처리 시스템의 처리량 최적화에 중요한 역할을 합니다.

- **Technical Details**: 본 연구에서는 실제 시스템의 복잡성을 반영한 자재 처리 환경을 개발하였으며, 다양한 위치에서의 활동, 물리적 제약 및 고유의 불확실성과 같은 요소를 포함합니다. 학습 중 탐색(employ exploration)을 개선하기 위해 기존의 동적 디스패칭 휴리스틱(heuristics)을 통합하는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 중앙 처리량(median throughput) 측면에서 기존 휴리스틱보다 최대 7.4% 향상된 성능을 보여주었습니다. 또한 서로 다른 기능을 가진 여러 에이전트를 훈련할 때 다양한 아키텍처가 MARL 성능에 미치는 영향을 분석하였습니다. 첫 번째 MARL 에이전트를 휴리스틱으로 사용하여 두 번째 MARL 에이전트를 훈련함으로써 성과를 더욱 개선할 수 있음을 보여주고 있습니다.



### Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization (https://arxiv.org/abs/2409.18433)
Comments:
          NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: Easy2Hard-Bench라는 새로운 벤치마크 데이터셋의 개발은 어려운 문제에서 쉽게 문제를 풀어내는 일반화 능력을 평가하기 위해 제안되었습니다. 각 문제는 숫자 형태의 난이도 점수로 주석이 달려 있습니다.

- **Technical Details**: 이 데이터셋은 수학, 프로그래밍 문제, 체스 퍼즐, 추리 문제 등 다양한 도메인에서 총 6개의 벤치마크 데이터셋으로 구성되어 있습니다. Item Response Theory (IRT)와 Glicko-2 모델과 같은 난이도 평가 시스템을 활용하여 문제에 대한 숫자 난이도 점수를 일관되게 부여합니다.

- **Performance Highlights**: 여섯 개의 최첨단 LLMs에 대한 광범위한 실험을 통해 다양한 난이도 수준에서의 성능과 일반화 능력을 총체적으로 분석하였으며, 이는 LLM 일반화에 대한 미래 연구에 영감을 줄 것입니다.



### A3: Active Adversarial Alignment for Source-Free Domain Adaptation (https://arxiv.org/abs/2409.18418)
Comments:
          Accepted at ICMLA 2024

- **What's New**: 본 논문은 레이블이 없는 대상 도메인으로부터 지식을 전이하기 위해 레이블이 있는 출처 도메인에서 지식을 전이하는 비지도 도메인 적응(Unsupvised Domain Adaptation, UDA) 분야의 최신 동향인 Source-free UDA에 대한 새로운 접근법인 Active Adversarial Alignment (A3)를 제안합니다.

- **Technical Details**: A3는 self-supervised learning, adversarial training, 및 active learning을 결합하여 견고한 Source-free UDA를 가능하게 합니다. 이 프레임워크는 acquisition function을 사용하여 유익하고 다양한 데이터를 능동적으로 샘플링하고 모델을 adversarial losses와 consistency regularization을 통해 적응시킵니다. 이는 출처 데이터에 접근하지 않고도 분포를 정렬합니다.

- **Performance Highlights**: A3는 효과적인 도메인 정렬 및 노이즈 감소를 위해 능동적 및 적대적 학습의 시너지를 활용하여 Source-free UDA를 발전시킵니다.



### VickreyFeedback: Cost-efficient Data Construction for Reinforcement Learning from Human Feedback (https://arxiv.org/abs/2409.18417)
Comments:
          16 pages, 5 figures

- **What's New**: 이 논문은 Reinforcement Learning from Human Feedback (RLHF)의 비용 효율성에 초점을 맞추고 있습니다. 기존의 preference dataset(선호도 데이터셋)의 경제적 유용성에 대한 고려가 부족했음을 지적하며, 이를 해결하기 위해 새로운 경매 메커니즘을 도입합니다.

- **Technical Details**: RLHF는 대규모 언어 모델(LLM)의 결과에 대한 인간의 선호도를 반영하기 위해 인간의 피드백을 활용합니다. 논문에서는 기존 알고리즘이 복잡한 비전이성(preference)이거나 순환적 관계를 처리하지 못하는 문제를 다룹니다. 경매 메커니즘을 사용하여 선호도 데이터 수집의 효율성을 높이는 방법을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 경매 기반 프로토콜은 고품질 피드백에 집중함으로써 LLM의 fine-tuning에서 비용 효율성을 개선하면서도 만족스러운 모델 성능을 유지하는 데 기여한다는 것이 입증되었습니다.



### SciDFM: A Large Language Model with Mixture-of-Experts for Scienc (https://arxiv.org/abs/2409.18412)
Comments:
          12 pages, 1 figure, 9 tables. Technical Report, Under Review

- **What's New**: 최근 대형 언어 모델(LLMs)을 활용하여 과학적 발견을 도우려는 관심이 급증하고 있습니다. 그러나 대부분의 LLM은 일반 과학에만 초점을 맞추고 있을 뿐, 화학 분자와 아미노산 서열과 같은 분야별 지식이 부족합니다. 이를 해결하기 위해 SciDFM이라는 전문가 혼합 모델을 도입하였으며, 이는 처음부터 훈련되어 대학 수준의 과학적 추론을 수행하고 분자 및 아미노산 서열을 이해할 수 있습니다.

- **Technical Details**: SciDFM은 대규모 훈련 데이터 집합을 수집하여 여러 과학 분야의 논문과 서적, 그리고 분야별 데이터베이스에서 수집한 데이터를 포함합니다. 또한, 사전 학습된 모델을 많은 지시 데이터로 추가 세부 조정하여 하위 기초 평가에서의 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과 SciDFM은 SciEval 및 SciQ와 같은 일반 과학 기초 평가에서 강력한 성능을 보이며, 동등한 크기 모델 중에서 분야별 평가에서도 SOTA(State Of The Art) 성능을 달성하였습니다. 우리는 전문가 선택 결과가 다른 분야의 데이터에 따라 달라진다는 점을 분석하였습니다. 또한 더 넓은 연구 커뮤니티를 위해 SciDFM을 오픈소스했습니다.



### BoT-Drive: Hierarchical Behavior and Trajectory Planning for Autonomous Driving using POMDPs (https://arxiv.org/abs/2409.18411)
- **What's New**: 이 논문은 자율 주행에서의 동적 도로 환경의 불확실성을 해결하기 위한 BoT-Drive 계획 알고리즘을 소개합니다.

- **Technical Details**: BoT-Drive는 Partially Observable Markov Decision Process (POMDP) 프레임워크 내에서 행동 및 경로 수준의 불확실성을 다룹니다. 이 알고리즘은 운전 모델을 사용하여 알 수 없는 행동 의도를 특징짓고, 이러한 모델 파라미터를 활용하여 숨겨진 운전 스타일을 추론합니다. 또한 BoT-Drive는 운전 모델을 자율 차량의 의사결정 행동으로 취급하여 POMDP 전의 복잡성을 효과적으로 해소합니다. 안전성과 견고성을 높이기 위해, 계획자는 계획된 고수준 행동에 조건화된 주행 경로를 개선하기 위해 중요 샘플링(importance sampling)을 적용합니다.

- **Performance Highlights**: 실제 데이터에 대한 평가 결과, BoT-Drive는 기존의 계획 방법과 학습 기반 방법 모두를 초월하여 정규 및 복잡한 도시 주행 장면에서 일관되게 높은 성능을 나타내었으며, 주행 안전성과 신뢰성에서 중요한 개선을 보였습니다.



### GenesisTex2: Stable, Consistent and High-Quality Text-to-Texture Generation (https://arxiv.org/abs/2409.18401)
- **What's New**: 이 논문에서는 대규모 텍스트 기반 이미지 확산 모델을 활용한 새로운 텍스처 합성 방법을 제안합니다. 텍스트에서 텍스처로의 변환을 위한 프레임워크를 소개하여 3D 기하학을 위한 텍스처 생성의 도전을 극복하고자 합니다.

- **Technical Details**: 우리의 접근법은 사전 훈련된 diffusion models를 기반으로 하며, self-attention 레이어에서 지역적 주의 재조정 메커니즘(local attention reweighing mechanism)을 도입하여 서로 다른 시점(viewpoint) 간에 공간적 연관성이 있는 패치에 집중하도록 모델을 유도합니다. 또한, 새로운 잠재 공간 병합 파이프라인(latent space merge pipeline)을 제안하여 다양성을 유지하면서도 서로 다른 시점 간의 일관성을 보장합니다.

- **Performance Highlights**: 이 방법은 기존의 최첨단 기술에 비해 텍스처 일관성(texture consistency)과 시각적 품질(visual quality)에서 상당히 우수한 성능을 보였으며, 증류 기반(methods based on distillation) 방법들보다 훨씬 빠른 결과를 제공했습니다. 추가적인 훈련이나 미세 조정(fine-tuning)이 필요 없어, 공공 플랫폼에서 사용할 수 있는 다양한 모델에 쉽게 적용할 수 있습니다.



### Code Vulnerability Repair with Large Language Model using Context-Aware Prompt Tuning (https://arxiv.org/abs/2409.18395)
- **What's New**: 대규모 언어 모델(Large Language Models, LLMs)의 취약한 코드를 탐지 및 수정하는 데에 있어 중요한 문제를 다룬 본 연구에서는, GitHub Copilot을 사용하여 버퍼 오버플로우(vulnerabilities involving buffer overflow) 취약점을 중심으로 연구를 진행했습니다.

- **Technical Details**: 실험 결과, Copilot의 버퍼 오버플로우 취약점 탐지율은 76%로 나타났으나, 취약점 수리율은 단 15%에 그쳤습니다. 이를 개선하기 위해 컨텍스트 인식 프롬프트 튜닝(context-aware prompt tuning) 기법을 제안하여, 다양한 보안 및 코드 맥락에 대한 도메인 지식을 주입하게 됩니다.

- **Performance Highlights**: 이러한 접근 방식을 통해 Copilot의 취약점 수리율은 63%로 증가하며, 도메인 지식 없이 수리한 경우에 비해 4배 이상의 개선을 나타냈습니다.



### Speech to Reality: On-Demand Production using Natural Language, 3D Generative AI, and Discrete Robotic Assembly (https://arxiv.org/abs/2409.18390)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible. An updated version will replace this version

- **What's New**: 본 논문에서는 자연어(Natural Language)를 사용하여 음성을 물리적인 객체로 변환하는 시스템을 소개합니다. 이 시스템은 3D generative Artificial Intelligence와 로봇 조립 기술을 결합하여, 3D 모델링이나 로봇 프로그래밍에 대한 전문 지식이 없는 사용자도 물리적인 객체를 만들 수 있도록 합니다.

- **Technical Details**: 시스템은 음성을 해석하여 3D 객체를 생성합니다. 생성된 객체는 voxel 컴포넌트로 분리되고, 그 후 최적의 조립 순서가 계산됩니다. 마지막으로 로봇 툴패스(Toolpath)가 생성되어, lattice 기반의 voxel 구성요소를 사용한 이산(discrete) 로봇 조립을 통해 물리 생산에서 발생할 수 있는 문제들을 해결합니다. 이 시스템은 여러 가지 객체(예: 의자, 선반 등)를 조립하는 데 이용될 수 있으며, 조립 과정은 6축 로봇 팔을 사용하여 5분 이내에 완료됩니다.

- **Performance Highlights**: 본 시스템은 다양한 객체를 신속하게 조립할 수 있는 능력을 보여줍니다. 음성 명령을 통해 신속하게 3D 객체를 생성하고 조립할 수 있는 기능은 제조와 디자인 프로세스를 혁신적으로 변화시킬 잠재력이 있습니다.



### Robo-CSK-Organizer: Commonsense Knowledge to Organize Detected Objects for Multipurpose Robots (https://arxiv.org/abs/2409.18385)
- **What's New**: 이번 논문에서는 로봇의 맥락 인식 능력을 향상시키기 위해 고전 지식 기반에서 상식(commonsense knowledge)을 통합한 Robo-CSK-Organizer 시스템을 소개합니다. 이 시스템은 탐지된 객체를 작업에 적합한 방식으로 분류하여 조직할 수 있도록 돕습니다.

- **Technical Details**: Robo-CSK-Organizer는 ChatGPT와 같은 딥러닝 도구에만 의존하는 시스템과는 달리 다음과 같은 여러 가지 장점을 제공합니다. 명확성이 뛰어나며, 객체 배치에서 일관성을 유지합니다. 또한, 다양한 작업 기반 분류(task-based classifications)에 적응할 수 있습니다. 이 시스템은 설명 가능한 AI(explainable AI)에 기여하여 신뢰(trust)와 인간-로봇 협력(human-robot collaboration)을 향상시키는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 가정용 로봇 환경을 시뮬레이션하여 Robo-CSK-Organizer가 맥락적으로 관련된 위치에 객체를 배치하는 데 뛰어난 성능을 보였습니다. 이 연구는 AI 기반 시스템이 인간의 인지(cognition) 수준에 더 가까운 상식에 기반한 의사결정을 할 수 있는 능력을 강조합니다.



### Adaptive Learning of the Latent Space of Wasserstein Generative Adversarial Networks (https://arxiv.org/abs/2409.18374)
- **What's New**: 본 연구에서는 새로운 프레임워크인 latent Wasserstein GAN (LWGAN)을 제안하여 Wasserstein auto-encoder와 Wasserstein GAN을 융합했습니다. 이를 통해 데이터 매니폴드(manifold)의 내재 차원(intrinsic dimension)을 적응적으로 학습할 수 있습니다.

- **Technical Details**: LWGAN은 인코더 네트워크와 제너레이터 네트워크를 통해 학습된 인코딩 분포의 내재 차원이 데이터 매니폴드의 차원과 같음을 입증합니다. 이론적으로, 우리의 추정된 내재 차원이 데이터 매니폴드의 진정한 차원의 일관된 추정임을 수립했습니다. 또한 LWGAN의 일반화 오류(generalization error)에 대한 상한을 제공합니다.

- **Performance Highlights**: 종합적인 실험 결과, LWGAN은 여러 시나리오에서 올바른 내재 차원을 식별할 수 있으며, 학습된 잠재 분포(latent distribution)로부터 고품질의 합성 데이터(synthetic data)를 생성할 수 있음을 보여주었습니다.



### Multi-hypotheses Conditioned Point Cloud Diffusion for 3D Human Reconstruction from Occluded Images (https://arxiv.org/abs/2409.18364)
Comments:
          17 pages, 7 figures, accepted NeurIPS 2024

- **What's New**: 본 논문에서는 심각한 가림(occlusion) 상황에서도 3D 인간 형태 재구성을 위한 새로운 파이프라인, MHCDIFF를 제안합니다. 이 방법은 픽셀 정렬된 세부 3D 인간 재구성을 위해 점 구름(diffusion) 모델을 도입합니다.

- **Technical Details**: MHCDIFF는 단일 RGB 이미지에서 기하학적 세부 사항을 캡처하기 위해 다수의 가설화된 SMPL(-X) 메쉬로부터 로컬 특징(local features)을 추출하고 이 특징 집합을 활용하여 확산 모델(diffusion model)을 조정합니다. 점 구름 확산(point cloud diffusion) 모델은 누락된(osculaded) 영역을 생성하기 위해 전체적인 일관성(global consistent features)을 캡처하며, 잡음 제거(denoising) 과정에서 잘못 정렬된 SMPL 메쉬를 수정합니다.

- **Performance Highlights**: CAPE 및 MultiHuman 데이터셋에서의 실험 결과, 제안된 방법은 SMPL, 암묵적 함수(implicit functions), 점 구름 확산(point cloud diffusion) 및 이들의 결합 기반의 다양한 최신 기술(SOTA) 방법들과 비교하여 우수한 성능을 보였습니다.



### Tracking Software Security Topics (https://arxiv.org/abs/2409.18351)
- **What's New**: 이번 논문에서는 소프트웨어 보안(Security) 관련 주제를 실시간으로 추적할 수 있도록 돕는 새로운 도구, SOSK를 제안합니다.

- **Technical Details**: SOSK는 사용자가 보안 보고서(Report) 집합을 가져와 이를 전처리(pre-process)하고, 텍스트 설명에서 가장 중요한 키워드(Keywords)를 추출합니다. 키워드의 임베딩 벡터(Embedding vector) 유사성을 기반으로 하여, SOSK는 사용자 제공 키워드 세트에서 키워드 세트를 확장하거나 정제할 수 있습니다.

- **Performance Highlights**: 초기 평가 결과, SOSK는 키워드를 효과적으로 확장하고, 사용자 요청에 맞는 보안 보고서를 성공적으로 검색할 수 있음을 보여주었습니다.



### A Generalized LLM-Augmented BIM Framework: Application to a Speech-to-BIM system (https://arxiv.org/abs/2409.18345)
- **What's New**: 이 논문은 건물 정보 모델링(BIM) 작업을 가속화하기 위해 LLM(대형 언어 모델)을 활용한 새로운 프레임워크를 제안합니다. 이는 전통적인 그래픽 사용자 인터페이스를 대체할 가능성을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 6단계로 구성됩니다: 해석(interpret) - 채우기(fill) - 일치(match) - 구조화(structure) - 실행(execute) - 검토(check). 이 과정은 텍스트에서 BIM 또는 음성을 BIM으로 변환하는 방식을 포함합니다.

- **Performance Highlights**: NADIA-S라는 음성 기반 BIM 응용 프로그램을 구현하여 제안된 프레임워크의 적합성을 입증하였고, 외부 벽 세부 사항을 예로 들었습니다.



### DRL-STNet: Unsupervised Domain Adaptation for Cross-modality Medical Image Segmentation via Disentangled Representation Learning (https://arxiv.org/abs/2409.18340)
Comments:
          MICCAI 2024 Challenge, FLARE Challenge, Unsupervised domain adaptation, Organ segmentation, Feature disentanglement, Self-training

- **What's New**: 본 논문에서는 cross-modality(크로스 모달리티) 의료 이미지 분할을 위한 새로운 프레임워크인 DRL-STNet을 제안합니다. 이 방법은 Generative Adversarial Networks(GANs), Disentangled Representation Learning(DRL), Self-Training(ST)를 활용합니다.

- **Technical Details**: DRL-STNet은 GAN 내에서 DRL을 이용하여 이미지를 소스 모달리티에서 타겟 모달리티로 변환합니다. 초기 단계에서는 변환된 이미지와 소스 레이블로 분할 모델을 학습한 후, 합성 이미지와 실제 이미지의 결합을 통해 pseudo-labels(의사 레이블) 및 실제 레이블로 반복적으로 미세 조정합니다.

- **Performance Highlights**: 제안된 DRL-STNet은 FLARE 도전 과제 데이터셋에서 복부 장기 분할에서 11.4% Dice similarity coefficient(다이스 유사도 계수) 및 13.1% Normalized Surface Dice metric(정규화된 표면 다이스 측정치) 향상을 보여 주며, 각각 74.21% 및 80.69%의 점수를 기록했습니다. 평균 실행 시간은 41초이며, GPU 메모리-시간 곡선 아래 면적은 11,292 MB입니다.



### AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models (https://arxiv.org/abs/2409.18339)
Comments:
          5 pages, 4 figures

- **What's New**: 본 연구는 LLMs(대규모 언어 모델)의 감정 인식 능력에 대한 새로운 접근을 제공합니다. 특히, 단일 감정 레이블에 국한되지 않고 모호한 감정을 인식하는 능력을 탐구하여 감정 지능(emotional intelligence)의 중요한 측면을 다루고 있습니다.

- **Technical Details**: 연구에서는 LLMs의 제로샷(zero-shot) 및 몇샷(few-shot) 프롬팅(prompts) 기술을 사용하여 문맥 정보(context information)를 과거 대화와 함께 통합함으로써 모호한 감정을 인식하는 과정이 설계되었습니다. 이를 통해 LLMs의 강력한 일반화 능력과 인-컨텍스트 학습(in-context learning)을 활용합니다.

- **Performance Highlights**: 실험 결과, LLMs는 모호한 감정을 인식하는 데 있어 상당한 잠재력을 보여주었으며, 문맥 정보를 포함할 때 그 효과성이 크게 증가함을 발견했습니다. 또한, 덜 모호한 감정을 인식하는 데 높은 효율성을 보이며, 더 모호한 감정들을 인식할 가능성도 제시하였습니다.



### Embodied-RAG: General non-parametric Embodied Memory for Retrieval and Generation (https://arxiv.org/abs/2409.18313)
Comments:
          Web: this https URL

- **What's New**: 새로운 연구에서는 Embodied-RAG라는 프레임워크를 소개하여 로봇의 탐색 및 학습 능력을 향상시키는 동시에, 비모수 메모리 시스템을 활용하여 계층적인 지식을 구성할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: Embodied-RAG는 다양한 환경 및 쿼리 유형에서 공간적(spatial) 및 의미적(semantic) 해상도를 처리할 수 있는 구조를 가지고 있습니다. 이 시스템의 핵심은 세밀한 언어 설명을 저장하는 의미적 숲(semantic forest) 형태의 메모리에 있으며, 이를 통해 다양한 로봇 플랫폼에서 맥락에 맞는 출력을 효율적으로 생성할 수 있습니다.

- **Performance Highlights**: Embodied-RAG는 19개 환경에서 200개 이상의 설명 및 탐색 쿼리를 성공적으로 처리하여 RAG를 로봇공학(domain of robotics) 분야와 연결하는 데 효과적임을 입증하였습니다.



### Harnessing Wavelet Transformations for Generalizable Deepfake Forgery Detection (https://arxiv.org/abs/2409.18301)
- **What's New**: 이번 논문에서는 Deepfake 탐지 방법의 약점을 해결하기 위해 Wavelet-CLIP이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Wavelet 변환(Wavelet Transforms)과 ViT-L/14 아키텍처에서 얻은 특징을 통합하여, 복잡한 Deepfake를 효과적으로 탐지할 수 있도록 설계되었습니다.

- **Technical Details**: Wavelet-CLIP는 Wavelet Transform을 이용하여 이미지의 공간적(spatial) 및 주파수(frequency) 특징을 깊이 분석합니다. 사전 훈련된 CLIP 방법으로 ViT-L/14 아키텍처의 특징들을 활용하여 높은 효율성을 발휘합니다.

- **Performance Highlights**: 테스트 결과, Wavelet-CLIP는 교차 데이터에 대한 일반화(cross-dataset generalization)에서 평균 AUC 0.749, 보이지 않는 Deepfake에 대한 강인성(robustness)에서 0.893을 기록하여 기존의 최첨단 방법들을 능가하는 뛰어난 성능을 보여주었습니다.



### SOAR: Self-supervision Optimized UAV Action Recognition with Efficient Object-Aware Pretraining (https://arxiv.org/abs/2409.18300)
- **What's New**: 이번 연구에서는 UAV(무인 항공기)로 촬영된 공중 영상에 대한 새로운 Self-supervised pretraining 알고리즘인 SOAR를 소개합니다. 이전 연구와는 달리, SOAR는 프리트레이닝(pretraining) 과정에서 인간 객체 지식을 통합하여 효율성을 높였습니다.

- **Technical Details**: SOAR는 두 가지 주요 접근 방식을 사용합니다. 첫째, 객체 인식과 관련된 특정 패치를 유지하기 위한 객체 인식 마스킹 전략(object-aware masking strategy)을 제안하였습니다. 둘째, 객체 정보를 활용하여 재구성 손실(reconstruction loss)을 조정하는 객체 인식 손실 함수(object-aware loss function)를 도입했습니다. 이를 통해 불필요한 배경 패치에 대한 편향을 방지할 수 있습니다.

- **Performance Highlights**: SOAR는 vanilla ViT 백본(backbone)을 사용하였으며, NEC-Drone과 UAV-Human 데이터셋에서 각각 9.7% 및 21.4%의 top-1 정확も(accuracy) 증가를 기록하며 기존 UAV 행동 인식(action recognition) 모델을 능가하였습니다. 특히, SOAR는 18.7ms의 비디오 당 추론 속도(inference speed)를 제공하며, 이는 2배에서 5배 더 빠릅니다. 추가로, SOAR는 이전의 Self-supervised learning 방법들과 유사한 정확도를 보여주면서도 87.5% 적은 프리트레이닝 시간과 25% 적은 메모리 사용을 요구합니다.



### Flat'n'Fold: A Diverse Multi-Modal Dataset for Garment Perception and Manipulation (https://arxiv.org/abs/2409.18297)
- **What's New**: Flat'n'Fold는 의류 조작을 위한 새로운 대규모 데이터셋으로, 기존 데이터셋의 중요한 격차를 해결합니다. 44개의 고유 의류 항목을 8개 카테고리에 걸쳐 총 1,212개의 인간과 887개의 로봇 시연으로 구성되어 있습니다.

- **Technical Details**: 이 데이터셋은 구겨진 상태에서 접힌 상태까지의 전체 조작 과정을 포착하며, 동기화된 다중 관점 RGB-D 이미지, 포인트 클라우드(point clouds), 그리고 손 또는 그리퍼(gripper) 위치 및 회전을 포함한 행동 데이터를 제공합니다. 이 데이터에 대한 다양성과 복잡성을 기존 기준과 비교하여 정량화하였습니다.

- **Performance Highlights**: Flat'n'Fold의 유용성을 보여주기 위해, 그리핑 포인트(prediction) 예측 및 하위 작업(subtask) 분해에 대한 새로운 기준을 설정했습니다. 이 작업에 대해 최첨단 모델을 평가한 결과, 개선이 필요한 부분이 많음을 확인했습니다. 이는 변형 가능한 객체의 로봇 인식과 조작 분야에서의 발전 가능성을 강조합니다.



### Enhancing Lossy Compression Through Cross-Field Information for Scientific Applications (https://arxiv.org/abs/2409.18295)
Comments:
          9 pages, 9 figures, accepted by DRBSD-10

- **What's New**: 본 논문은 과학 데이터 세트 내에서의 중요한 교차 필드 상관 관계를 식별하고, CNN(Convolutional Neural Network)을 이용하여 이러한 정보를 추출하는 새로운 하이브리드 예측 모델을 제안합니다.

- **Technical Details**: 제안된 하이브리드 모델은 기존의 로컬 필드 정보와 결합하여 교차 필드 정보를 활용합니다. 이를 통해 예측 정확성을 높이고, 손실 압축(lossy compression) 성능을 개선합니다.

- **Performance Highlights**: 이 모델은 세 가지 과학 데이터 세트에서 검증되었으며, 특정 오류 한계 내에서 압축 비율을 최대 25% 개선하고, 데이터 세부정보를 더 잘 보존하며, 아티팩트(artifacts)를 줄이는 성능을 보였습니다.



### Criticality and Safety Margins for Reinforcement Learning (https://arxiv.org/abs/2409.18289)
Comments:
          17 pages, 10 figures. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문에서는 최근 강화 학습(Reinforcement Learning) 방법들이 안전하지 않은 상황을 만날 수 있다는 점을 다루고 있습니다. 이에 따라, 이러한 상황을 식별하는 것이 분석 및 실행 과정에서 인간의 감시를 요청할 수 있는 이점을 제공할 수 있는 중요성을 강조하고 있습니다.

- **Technical Details**: 논문에서는 true criticality와 proxy criticality라는 두 가지 개념을 도입하여, 강화 학습 에이전트가 정책(policy)에서 벗어나 무작위 행동을 n번 연속으로 따를 때 예상되는 보상의 감소를 기반으로 합니다. true criticality는 구체적인 지표로 작용하며, proxy criticality는 통계적으로 true criticality와 단조롭게(monotonic) 관계를 구성하는 저오버헤드(low-overhead) 메트릭입니다. 이 메트릭들은 신뢰 구간 내에서 tolerable 성능 손실을 초과하지 않는 무작위 행동의 수로 정의된 안전 여유(safety margins)에 의해 더욱 설명 가능합니다.

- **Performance Highlights**: Atari Beamrider 환경에서 A3C 에이전트를 사용하여 한 실험을 진행한 결과, 최소 5%의 안전 여유가 47%의 에이전트 손실을 포함하는 것으로 나타났습니다. 즉, 5%의 결정을 감독함으로써 에이전트의 오류를 약 50% 예방할 수 있다는 것을 제시합니다. 이러한 점에서 이 프레임워크는 나쁜 결정이 이루어지기 전에 그 잠재적 영향을 측정할 수 있어, 자율 에이전트의 더 효과적인 디버깅(debugging)과 감독을 가능하게 합니다.



### Advancing Object Detection in Transportation with Multimodal Large Language Models (MLLMs): A Comprehensive Review and Empirical Testing (https://arxiv.org/abs/2409.18286)
- **What's New**: 이번 연구는 교통 시스템에서 객체 탐지(object detection)에 대한 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)과 대형 비전 모델(Large Vision Models, VLMs)의 응용을 포괄적으로 검토하고 실증적으로 평가하는 것을 목표로 합니다.

- **Technical Details**: 연구의 첫 번째 부분에서는 MLLMs의 교통 응용 분야에서의 잠재적인 이점에 대한 배경을 제공하고, 기존 연구에서 현재 MLLM 기술에 대한 포괄적인 리뷰를 실시했습니다. 두 번째 부분에서는 교통 응용 프로그램에서의 엔드 투 엔드 객체 탐지(taxonomy of end-to-end object detection) 개요와 향후 방향을 제시했습니다.

- **Performance Highlights**: MLLM 성능에 대한 상세한 평가 결과를 제공하며, 도로 안전 특성 추출, 안전 비상 이벤트 탐지, 열화상 이미지의 시각적 추론을 포함한 세 가지 실제 교통 문제에 대한 실증 분석을 수행했습니다. 이 연구는 MLLM의 강점과 개선이 필요한 영역을 밝혀냈으며, 교통에서의 객체 탐지를 향상시키기 위한 MLLM의 실용적인 한계와 도전에 대해 논의합니다.



### Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation (https://arxiv.org/abs/2409.18261)
Comments:
          ECCV 2024 (poster). Github page: this https URL

- **What's New**: Omni6D라는 새로운 RGBD 데이터셋을 소개하며, 다양한 범주와 배경을 포함하여 6D 객체 자세 추정의 현실적인 맥락을 제공한다.

- **Technical Details**: Omni6D 데이터셋은 166개의 범주, 4688개의 인스턴스, 80만 개 이상의 캡처로 구성되어 있어 기존 데이터셋보다 훨씬 넓은 평가 범위를 제공한다. 우리는 대칭성 인식(symmetry-aware) 메트릭을 도입하고 기존 알고리즘에 대한 체계적인 벤치마크를 수행하여 새로운 도전과 통찰력을 탐구한다. 또한, 기존 데이터셋에서 모델을 조정(adapt)할 수 있는 효과적인 파인 튜닝(fine-tuning) 접근법도 제안한다.

- **Performance Highlights**: 이 연구는 산업 및 학계 모두에서 6D 자세 추정의 경계를 확장하고 새로운 통찰력과 상당한 진행을 이룰 수 있는 기반을 마련할 것으로 기대된다.



### PCEvE: Part Contribution Evaluation Based Model Explanation for Human Figure Drawing Assessment and Beyond (https://arxiv.org/abs/2409.18260)
- **What's New**: 본 연구에서는 인간 형상 드로잉(Human Figure Drawing, HFD) 평가 작업에서 모델 결정의 명확성과 설명 가능성을 높이기 위해 부분 기여 평가 기반 모델 설명(Part Contribution Evaluation based Model Explanation, PCEvE) 프레임워크를 제안합니다.

- **Technical Details**: PCEvE는 각 개별 부분의 Shapley Value를 측정하여 모델 결정에 대한 기여도를 평가합니다. 기존의 pixel-level attribution 기반 설명 가능한 AI 설명 방식과 달리, PCEvE는 부분 기여 히스토그램(part contribution histogram)이라는 직관적인 설명을 제공합니다. 또한, PCEvE는 설명의 범위를 샘플 수준(sample-level)에서 클래스 수준(class-level) 및 작업 수준(task-level)으로 확장합니다.

- **Performance Highlights**: PCEvE는 여러 HFD 평가 데이터셋에 대한 광범위한 실험을 통해 엄격하게 검증되었으며, 제안된 방법의 타당성을 확인하기 위한 제어된 실험을 수행했습니다. 또한, PCEvE는 스탠포드 자동차와 같은 포토리얼리스틱(photo-realistic) 데이터셋에도 적용하여 다양성과 응용 가능성을 입증했습니다.



### AI Policy Projector: Grounding LLM Policy Design in Iterative Mapmaking (https://arxiv.org/abs/2409.18203)
- **What's New**: 이 논문에서는 정책 설계를 지도 제작(mapmaking)에서 영감을 얻어 새로운 AI 정책 설계 프로세스를 소개합니다.

- **Technical Details**: 이 연구에서 제안하는 Policy Projector는 모델 입력 및 출력 쌍의 지형을 탐색하고 사용자 정의 영역을 정의하며, LLM 출력에 적용할 수 있는 규칙을 탐색할 수 있도록 해줍니다. 예를 들어, '폭력(violence)'과 '그래픽 세부사항(graphic details)'이 포함된 출력이 있는 경우 그래픽 세부사항 없이 다시 쓰는 규칙을 설정할 수 있습니다.

- **Performance Highlights**: 12명의 AI 안전(AI safety) 전문가와의 평가에서, Policy Projector는 정책 설계자가 기존의 포괄적인 해악 분류(harm taxonomy)를 넘어서는 문제적 모델 행동을 해결하는 데 도움을 주었습니다.



### Evaluation of Large Language Models for Summarization Tasks in the Medical Domain: A Narrative Review (https://arxiv.org/abs/2409.18170)
- **What's New**: 대형 언어 모델(Large Language Models)의 발전은 임상 자연어 생성(Clinical Natural Language Generation)을 촉진하고, 의료 텍스트의 양을 관리할 수 있는 기회를 창출했습니다. 그러나 의료의 높은 위험 수준은 신뢰할 수 있는 평가를 요구하며, 이는 여전히 도전 과제로 남아 있습니다.

- **Technical Details**: 이 논문은 임상 요약 작업(Clinical Summarization Tasks)의 현재 평가 상태를 분석하고, 전문가의 인간 평가(Human Evaluation) 자원 제약(Resource Constraints)을 해결하기 위한 미래 방향을 제안합니다. 이를 통해 임상 분야에서의 신뢰성을 높이고자 합니다.

- **Performance Highlights**: 이 리뷰는 임상 자연어 생성의 효율성을 높이는 방법을 제안하며, 향후 연구 방향을 명확히 하고, 평가 프로세스의 개선 필요성을 강조합니다.



### Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey (https://arxiv.org/abs/2409.18169)
- **What's New**: 최근 연구에 따르면, 새로운 fine-tuning-as-a-service 비즈니스 모델이 심각한 안전 문제를 드러내고 있습니다. 사용자가 업로드한 몇 가지 해로운 데이터로 인한 fine-tuning이 모델의 안전 정렬(safety alignment)을 위협할 수 있습니다. 이러한 공격은 harmful fine-tuning으로 알려져 있으며, 연구 커뮤니티 내에서 큰 관심을 받고 있습니다.

- **Technical Details**: 논문은 먼저 문제의 위협 모델(threat model)을 제시하고, harmful fine-tuning 공격과 그 변종을 소개합니다. 이후 기존 문헌에 대한 체계적인 조사(survey)를 진행하여 공격 및 방어(defenses), 기계적 분석(mechanical analysis)에 대해 논의합니다.

- **Performance Highlights**: 논문은 연구 문제를 공식적으로 정립하고, 향후 연구 방향(future research directions)을 제안합니다. 또한, 동료 평가(peer review) 과정에서 실험/공격/방어 설정의 현실성에 대한 질문을 제기할 때 유용할 수 있는 흥미로운 질문 목록을 포함하고 있습니다.



### A Survey on Neural Architecture Search Based on Reinforcement Learning (https://arxiv.org/abs/2409.18163)
- **What's New**: 딥러닝의 폭발적인 발전 덕분에 기계 학습의 feature extraction 자동화가 성공적으로 이루어졌습니다. 이제 연구자들은 최적의 네트워크 구조와 하이퍼파라미터를 자동으로 찾을 수 있는 방법을 모색하고 있습니다.

- **Technical Details**: 논문에서는 Neural Architecture Search (NAS)의 전반적인 발전 상황을 소개하고, 특히 reinforcement learning과 관련된 NAS 작업의 개요를 제공합니다. 또한, 복잡한 구조와 자원이 부족한 환경에서의 개선 및 변형에 대해 논의합니다.

- **Performance Highlights**: Neural Architecture Search를 통해 다양한 작업에 최적화된 네트워크 구조를 자동적으로 발견할 수 있는 가능성이 열리며, 이는 성능 향상에 큰 기여를 할 것으로 기대됩니다.



### The Nexus of AR/VR, Large Language Models, UI/UX, and Robotics Technologies in Enhancing Learning and Social Interaction for Children: A Systematic Review (https://arxiv.org/abs/2409.18162)
Comments:
          none

- **What's New**: 이 리뷰 연구에서는 대형 언어 모델(LLMs), 증강 현실(AR) 및 사용자 인터페이스/사용자 경험(UI/UX) 디자인의 결합이 자폐 스펙트럼 장애(ASD)를 가진 아동 치료에서 어떻게 활용될 수 있는지를 살펴보았습니다.

- **Technical Details**: 연구에서는 PubMed, ACM, IEEE Xplore, Elsevier 및 Google Scholar에서 150개의 관련 출판물을 찾아냈고, 이 중 42개를 방법론적 엄격성과 관련성을 기준으로 선정하여 심층 분석하였습니다. 세 가지 주요 영역이 다뤄졌습니다: AR이 사회적 및 학습 결과를 어떻게 개선할 수 있는지, LLMs가 의사소통에 어떻게 기여할 수 있는지, UI/UX 디자인이 이러한 기술의 효과성에 어떤 영향을 미치는지입니다.

- **Performance Highlights**: 결과에 따르면 LLMs는 개인화된 학습 및 의사소통 지원을 제공할 수 있지만, AR은 사회적 기술, 동기 및 집중력을 향상시키는 데 긍정적인 영향을 미치는 것으로 나타났습니다. ASD 아동을 위한 접근 가능하고 흥미로운 개입은 효과적인 UI/UX 디자인에 크게 의존하며, 이 연구는 개인화, 접근성 및 통합과 관련된 난제들을 해결하기 위한 추가 연구의 필요성을 강조하고 있습니다.



### Decomposition of one-layer neural networks via the infinite sum of reproducing kernel Banach spaces (https://arxiv.org/abs/2409.18132)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 RKBS(Reproducing Kernel Banach Spaces)의 합을 정의하고, RKBS의 특성 정리를 이용하여 이를 직접 합(Direct Sum) 기능공간과의 호환성을 증명합니다.

- **Technical Details**: RKBS의 합은 $p$-norm RKBS의 합으로 분해할 수 있으며, 이는 RKBS 클래스의 구조적 이해에 기여하는 응용 프로그램을 제공합니다.

- **Performance Highlights**: 이 연구는 RKBS의 수학적 구조를 더 깊이 이해하는 데 중요한 기초를 마련하며, 기능적 관점에서 RKBS의 합이 어떻게 작용하는지를 설명합니다.



### Modulated Intervention Preference Optimization (MIPO): Keep the Easy, Refine the Difficu (https://arxiv.org/abs/2409.17545)
Comments:
          8pages, submitted to AAAI 2025

- **What's New**: 이번 연구에서는 Modulated Intervention Preference Optimization (MIPO)라는 새로운 방법론을 제안합니다. MIPO는 주어진 데이터와 참조 모델의 정렬 상태에 따라 개입(intervention) 정도를 조절하여 모델 일치를 최적화합니다.

- **Technical Details**: MIPO는 참조 모델이 잘 정렬된 경우, 정책 모델이 참조 모델과 크게 이탈하지 않도록 개입을 증가시키고, 반대로 정렬이 좋지 않은 경우 개입을 줄여 보다 광범위한 학습을 가능하게 합니다. 이 연구에서는 Mistral-7B와 Llama3-8B 모델을 활용하여 MIPO와 DPO의 성능을 비교합니다.

- **Performance Highlights**: 실험 결과, MIPO는 다양한 평가 시나리오에서 DPO에 비해 일관되게 우수한 성능을 보였습니다.



### M$^2$PT: Multimodal Prompt Tuning for Zero-shot Instruction Learning (https://arxiv.org/abs/2409.15657)
Comments:
          EMNLP 2024

- **What's New**: 이번 연구에서는 Multimodal Prompt Tuning (M$^2$PT)이라는 새로운 접근 방식을 도입하여 대형 다중 모달 언어 모델 (MLLMs)의 효율적인 지침 조정 (instruction tuning)을 지원합니다.

- **Technical Details**: M$^2$PT는 시각 (visual) 및 텍스트 (textual) 프롬프트를 각각 비전 인코더 (vision encoder)와 언어 프로세서 (language processor)에 통합하여 파인튜닝 (finetuning) 동안 다양한 모달리티 간의 특징을 추출하고 일치시킵니다.

- **Performance Highlights**: 다양한 다중 모달 평가 데이터셋에서 우리의 접근 방식은 여러 최신 기법 (state-of-the-art baselines) 대비 우수한 성능을 보여주었으며, 포괄적인 제거 연구 (ablation studies)를 통해 프롬프트 설계와 접근 방식의 효율성을 확인하였습니다.




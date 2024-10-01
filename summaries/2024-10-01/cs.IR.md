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




New uploads on arXiv(cs.CL)

### Granularity is crucial when applying differential privacy to text: An investigation for neural machine translation (https://arxiv.org/abs/2407.18789)
- **What's New**: 교육자들이 학생 평가에 소요하는 시간을 줄이기 위한 자동 MCQ 생성 기법을 개선하기 위해 지식 종속 대답 가능성(KDA)이라는 새로운 자동 평가 메트릭이 제안되었습니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 측정하며, 자동화된 평가 메트릭인 KDA_disc와 KDA_cont를 통해 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이는 기존의 n-gram 기반 유사도 메트릭과 결합될 때 MCQ 품질 측정의 예측력을 강화합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 강의실 환경에서의 사용성과 KDA와 강한 상관관계를 나타내며, 전문가가 레이블을 붙인 다양한 MCQ 품질 측정에서도 강력한 예측력을 보였습니다.



### The power of Prompts: Evaluating and Mitigating Gender Bias in MT with LLMs (https://arxiv.org/abs/2407.18786)
- **What's New**: 본 논문에서는 머신 번역에서 성 편향(gender bias)을 대규모 언어 모델(LLMs)의 관점에서 연구하였다. 다양한 LLM와 기존 최신 신경 머신 번역(NMT) 모델을 비교하여 번역 품질과 성 편향을 평가하였다.

- **Technical Details**: En → Ca(영어-카탈루냐어) 및 En → Es(영어-스페인어) 번역 방향으로 네 가지 테스트 세트를 사용하여 모델들의 성 편향을 측정하였으며, 기본 LLM이 NMT 모델에 비해 더 높은 성 편향을 나타낸다는 결과를 도출하였다. 또한, 지시 튜닝된 LLM에 적용된 프롬프트 엔지니어링(prompt engineering) 기법을 탐구하였다.

- **Performance Highlights**: 프롬프트 구조를 조정하여 WinoMT 평가 데이터셋에서 성 편향을 최대 12%까지 감소시킬 수 있었으며, 이는 LLM과 전통적인 NMT 시스템 간의 성 편향 정확성 격차를 크게 줄이는 성과를 보였다.



### Knowledge Graph Structure as Prompt: Improving Small Language Models Capabilities for Knowledge-based Causal Discovery (https://arxiv.org/abs/2407.18752)
Comments:
          accepted at ISWC'24

- **What's New**: 이번 연구에서는 MCQ(다중 선택 질문) 생성의 평가 메트릭으로 '지식 종속 가능성(KDA)'을 제안하여, MCQ의 대답 가능성을 요약 평가하는 방법을 소개합니다. 기존의 평가 메트릭들은 교육적 가치와 학생의 지식 평가 능력을 고려하지 않았던 문제를 다룹니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 하여 MCQ의 대답 가능성을 측정합니다. KDA를 근사하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하며, 이들은 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다. 또한, 교육에서의 사용 가능성과 KDA 간에 강한 상관관계를 보여주었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 라벨링을 받은 다양한 MCQ 품질 측정값에 대해 강력한 예측력을 보여주었으며, 기존 n-그램 기반 유사도 메트릭과 결합했을 때 더욱 우수한 성능을 발휘했습니다.



### Towards Effective and Efficient Continual Pre-training of Large Language Models (https://arxiv.org/abs/2407.18743)
Comments:
          16 pages, 10 figures, 16 tables

- **What's New**: 본 논문에서는 자동 MCQ 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하며, 이를 통해 MCQ의 대답 가능성을 측정함.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정되며, 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안하여 pre-trained language models를 활용하여 학생의 문제해결 행동을 모방함. 이 방법은 실제 교육 환경에서의 사용성을 고려함.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 레이블을 붙인 MCQ의 품질 측정에서 n-그램 기반 유사성 메트릭과 결합했을 때 강력한 예측력을 보임. 실험 결과, 이 메트릭들이 KDA 및 실제 수업 설정의 사용성과 강한 상관관계를 가진 것으로 나타남.



### Towards Generalized Offensive Language Identification (https://arxiv.org/abs/2407.18738)
Comments:
          Accepted to ASONAM 2024

- **What's New**: 본 논문에서는 기존의 MCQ (Multiple Choice Questions) 생성기법의 한계를 극복하기 위해, 새로운 자동 평가 메트릭인 'Knowledge Dependent Answerability (KDA)'를 제안합니다. 이는 학생의 목표 사실(target fact)에 대한 지식을 고려하여 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 pre-trained language models를 활용하여 학생의 문제 해결 행동을 모사합니다. 이 접근법은 일반적인 n-gram 기반 유사도 메트릭과 결합하여 MCQ의 품질을 더욱 정교하게 평가할 수 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 교실 환경에서의 전문가에 의해 라벨링된 사용성과 KDA와 강한 상관관계를 보였으며, 다양한 전문가 라벨링 MCQ 품질 측정에 대해 예측력이 뛰어난 것으로 나타났습니다.



### Creating an Aligned Corpus of Sound and Text: The Multimodal Corpus of Shakespeare and Milton (https://arxiv.org/abs/2407.18730)
- **What's New**: 이번 연구에서는 윌리엄 셰익스피어(William Shakespeare)와 존 밀턴(John Milton)의 시를 공공 도메인 자료로 풍부하게 만든 코퍼스를 제시합니다. 모든 줄을 각각의 오디오 세그먼트와 정렬하고, 기본 시각화 플랫폼을 제공합니다.

- **Technical Details**: 우리는 각 줄, 단어, 음절 및 음소 수준에서 오디오 세그먼트와의 정렬을 수행하며, 이 시의 운율(sansion)도 포함되어 있습니다.

- **Performance Highlights**: 우리는 향후 방향에 대한 가능성을 추측함으로써 이 연구의 잠재적인 가치를 논의합니다.



### ChatSchema: A pipeline of extracting structured information with Large Multimodal Models based on schema (https://arxiv.org/abs/2407.18716)
- **What's New**: 이 논문은 자동 MCQ 생성 과정에서 교육적 가치 평가를 위한 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하고, 이를 토대로 두 가지 자동 평가 메트릭 KDA_disc 및 KDA_cont를 소개합니다.

- **Technical Details**: KDA는 학생들의 응답을 기반으로 MCQ의 대답 가능성을 측정하며, 이 과정에서 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. 연구는 기존의 n-그램 기반 유사성 메트릭과 결합하여 MCQ 품질 예측의 강력한 예측력을 입증하였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가에 의해 레이블링 된 실제 교실 환경에서의 사용성과 강한 상관관계를 보였고, n-그램 기반 유사성 메트릭과 결합 시 다양한 MCQ 품질 평가 지표에 대한 예측력이 강화되었습니다.



### Adaptive Contrastive Search: Uncertainty-Guided Decoding for Open-Ended Text Generation (https://arxiv.org/abs/2407.18698)
- **What's New**: 자동 MCQ 생성의 효율성을 평가하기 위한 새로운 메트릭인 지식 종속 가능성(KDA)을 제안하고, 이를 통해 MCQ의 교육적 가치를 평가할 수 있습니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 두 메트릭은 사전 훈련된 language models를 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 강의실 환경에서 실제 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가 레이블의 MCQ 품질 효율성을 예측하는 데 강력한 성능을 보입니다.



### The BIAS Detection Framework: Bias Detection in Word Embeddings and Language Models for European Languages (https://arxiv.org/abs/2407.18689)
- **What's New**: 자동 MCQ 생성에 대한 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하여 기존의 평가 메트릭의 한계를 극복하고 교육적 가치를 평가할 수 있도록 한다.

- **Technical Details**: KDA는 주관적인 인적 조사에서 학생들의 반응을 기반으로 측정되며, KDA_disc와 KDA_cont의 두 가지 자동 평가 메트릭을 제안 한다. 이는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 접근 방식이다. 이 방법은 n-gram 기반 유사도 메트릭과 결합되어 전문가가 라벨링한 다양한 MCQ 품질 지표에 대해 강한 예측력을 보인다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 KDA 및 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, 사용성과 교육적 효과를 동시에 고려한 평가 방식으로 MCQ 품질 향상에 기여할 수 있다.



### Every Part Matters: Integrity Verification of Scientific Figures Based on Multimodal Large Language Models (https://arxiv.org/abs/2407.18626)
Comments:
          28 pages, 11 figures, under review

- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성을 위한 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'가 제안되었습니다. KDA는 학생의 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont가 제안되어, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한, 이 메트릭들은 n-gram 유사도 평가 메트릭과 결합되었을 때 강력한 예측력을 발휘합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 전문가가 라벨링한 KDA 및 사용성과 강한 상관관계를 가진다는 것을 보여주며, MCQ 품질 측면에서 우수한 예측력과 정확성을 달성했습니다.



### Dynamic Language Group-Based MoE: Enhancing Efficiency and Flexibility for Code-Switching Speech Recognition (https://arxiv.org/abs/2407.18581)
- **What's New**: 이 논문은 다국어 및 코드 스위치(coding-switching) 챌린지를 해결하기 위해 설계된 DLG-MoE(Dynamic Language Group-based Mixture of Experts) 모델을 소개합니다.

- **Technical Details**: DLG-MoE는 다중 전문가 아키텍처를 사용하여, 명시적 언어 모델링을 위한 공유 가중치(language router)와 언어 외의 속성을 처리하는 독립적인 비지도 라우터들로 구성됩니다. 이 구조는 전문가 확장 능력을 향상시키고, 다양한 top-k 값에서 유연한 추론을 가능하게 하는 동적 top-k 훈련을 지원합니다.

- **Performance Highlights**: 이 모델은 사전 훈련이 필요 없으며, 스트리밍 인식을 지원하여 다른 방법들에 비해 유사한 유연성을 유지하면서도 최고의 성능(state-of-the-art, SOTA)을 달성합니다.



### Learning Robust Named Entity Recognizers From Noisy Data With Retrieval Augmentation (https://arxiv.org/abs/2407.18562)
- **What's New**: 자동 MCQ 생성 및 노이즈가 있는 텍스트에서의 Named Entity Recognition (NER) 모델의 향상을 위한 새로운 방법을 제안함.

- **Technical Details**: 자동 MCQ에 대한 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 소개하며, 이 메트릭은 학생이 제출한 답변을 기반으로 MCQ의 대답 가능성을 측정한다. 또한, NLP 태스크에서의 robustness를 향상시키기 위해 '여러 개의' counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 사용하여 인과관계를 파악하는 방법을 제안한다. NER은 노이즈가 있는 텍스트에서 지식 코퍼스에서 관련 텍스트를 검색하여 원본 노이즈 입력의 표현을 향상시킨다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 평가한 클래스룸 환경에서의 MCQ 사용성과 강한 상관관계를 가지고 있으며, 이는 기존 n-gram 기반 메트릭과 결합하여 MCQ 품질 예측에 강력한 예측력을 보여준다. 또한, 검색 중심의 NER 모델은 다양한 노이즈가 있는 NER 환경에서도 뛰어난 성능 향상을 달성하였다.



### A Universal Prompting Strategy for Extracting Process Model Information from Natural Language Text using Large Language Models (https://arxiv.org/abs/2407.18540)
- **What's New**: 자동 MCQ 생성의 평가에 대한 새로운 접근법인 Knowledge Dependent Answerability (KDA)를 제안하고, 이를 통해 MCQ의 대답 가능성을 평가하는 방법을 기술하였습니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭과는 다르게 KDA는 학생의 지식을 직접적으로 평가할 수 있는 능력을 가지고 있습니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA 및 실제 강의실 환경에서의 사용성과 강한 상관관계를 가진다는 것을 보여주었으며, n-gram 기반 메트릭과 결합할 경우, 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.



### Towards a Multidimensional Evaluation Framework for Empathetic Conversational Systems (https://arxiv.org/abs/2407.18538)
Comments:
          13 pages, 4 figures

- **What's New**: 자동 생성된 Multiple Choice Questions (MCQ)의 대답 가능성을 평가하기 위한 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 기존의 BLEU, ROUGE, METEOR와 같은 평가 방법만으로는 교육적 가치를 평가하기에 한계가 있었습니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 주관적으로 라벨링한 실제 교실 환경에서의 사용성과 KDA와 강한 상관관계를 보여줍니다. n-gram 기반의 유사성 메트릭과 결합했을 때, 다양한 전문가 라벨된 MCQ 품질 측정을 예측하는 데 강력한 성능을 발휘합니다.



### Is larger always better? Evaluating and prompting large language models for non-generative medical tasks (https://arxiv.org/abs/2407.18525)
Comments:
          arXiv admin note: text overlap with arXiv:2402.01713

- **What's New**: 최근 대형 언어 모델(LLMs)의 의료 분야 활용이 증가하고 있지만, 구조화된 전자 건강 기록(EHR) 데이터와 비구조화 임상 노트를 처리하는 능력에 대한 연구는 미비하다. 본 연구는 여러 모델을 비교하여 비생성적 의료 작업에 대한 성능을 평가한다.

- **Technical Details**: 구조화된 EHR 데이터와 비구조화된 의료 텍스트를 다루기 위해, 14개의 언어 모델(9개 GPT 기반, 5개 BERT 기반)과 7개의 전통적인 예측 모델을 MIMIC 데이터셋(ICU 환자 기록)과 TJH 데이터셋(조기 COVID-19 EHR 데이터)을 활용하여 평가했다.

- **Performance Highlights**: 결과적으로 LLM은 구조화된 EHR 데이터에 대해서는 robust한 zero-shot 예측 능력을 보였으나, 비구조화된 의료 텍스트에 대해서는 finetuned BERT 모델이 더 우수한 성능을 나타냈다. 따라서 구조화된 데이터에서는 LLM의 zero-shot 학습이 효과적이지만, 비구조화 텍스트에 대해서는 finetuned BERT 모델이 더 적합하다는 점이 강조된다.



### The formation of perceptual space in early phonetic acquisition: a cross-linguistic modeling approach (https://arxiv.org/abs/2407.18501)
Comments:
          51 pages

- **What's New**: 본 연구는 학습자가 초기 음성 습득 과정에서 지각 공간(perceptual space)을 어떻게 조직하는지를 조사하며, 기존 연구를 두 가지 주요 측면에서 발전시켰습니다. 첫째, 학습된 숨겨진 표현(hidden representation)의 모양(shape)과 음성 카테고리(categorize phonetic categories)를 범주화하는 능력을 검토합니다. 둘째, 맥락적 단서(contextual cues)를 포함하지 않고 모델을 훈련시키는 것이 음성 습득에 미치는 영향을 탐구합니다.

- **Technical Details**: 영어(English)와 만다린(Mandarin)에서 autoencoder 모델을 훈련시키고, 영아 언어 인식 연구에서 사용된 실험 조건을 따르며 본 연구에서는 원어(native)와 비원어(non-native) 조건에서 평가했습니다. 이 연구는 맥락 없는(acoustic information) 정보에 대한 비지도 하의(bottom-up) 훈련이 원어와 비원어 조건 간의 지각 공간에 대한 유사한 학습 표현을 이룰 수 있음을 보여줍니다.

- **Performance Highlights**: 이 결과는 초기 유아의 보편적 듣기(universal listening)와 유사한 방식으로, 영어와 만다린 모두에서 원어 및 비원어 조건 간에 지각 공간을 조직하는 방식에 대한 통찰력을 제공합니다. 이는 음성 카테고리의 형성과 표현에 대한 우리의 이해를 더욱 풍부하게 합니다.



### A Reliable Common-Sense Reasoning Socialbot Built Using LLMs and Goal-Directed ASP (https://arxiv.org/abs/2407.18498)
- **What's New**: 이 논문에서는 AutoCompanion이라는 소셜봇을 제안하며, 이는 대형 언어 모델(LLM)을 사용하여 자연어를 술어로 변환하고 (반대로도) 대화 중 일반 상식을 적용하여 대화를 유도할 수 있도록 설계되었습니다.

- **Technical Details**: AutoCompanion은 s(CASP)라는 목표 지향적 Answer Set Programming (ASP) 구현을 백엔드로 사용하여 인간과의 소셜 대화를 진행합니다. 이 프레임워크는 LLM을 사용하여 사용자 메시지를 분석하고 s(CASP) 엔진의 출력을 바탕으로 응답을 생성하는 과정을 설명합니다.

- **Performance Highlights**: 논문에서는 챗봇이 영화와 책에 대해 이야기하며 사용자를 재미있게 유지하는 목표를 가지고 진행한 실제 대화를 기술함으로써 (i) 답변의 정확성, (ii) 대화의 일관성 및 정밀성, (iii) 주요 주제에서의 이탈 없음 등을 보장하는 방법을 제시합니다.



### Towards More Accurate Prediction of Human Empathy and Emotion in Text and Multi-turn Conversations by Combining Advanced NLP, Transformers-based Networks, and Linguistic Methodologies (https://arxiv.org/abs/2407.18496)
- **What's New**: 본 논문에서는 학생 평가를 위한 Multiple Choice Questions (MCQ) 자동 생성의 효율성을 높이기 위해 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 기존의 BLEU, ROUGE 및 METEOR와 같은 메트릭은 MCQ의 교육적 가치를 점검하지 못했습니다. KDA는 MCQ의 답변 가능성을 측정하여 학생의 지식을 평가할 수 있는 능력을 제공합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, 자동 평가 메트릭 KDA_disc와 KDA_cont를 통해 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행위를 모방합니다. 이를 통해 KDA는 실제 교실 환경에서의 사용성과 강한 상관관계를 보여줍니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 다양한 MCQ 품질 측정 지표에 대해 예측력이 강한 것으로 나타났습니다. 또한, 이러한 새로운 메트릭은 n-gram 기반 유사성 메트릭과 결합할 때 더욱 우수한 성능을 보였습니다.



### A Role-specific Guided Large Language Model for Ophthalmic Consultation Based on Stylistic Differentiation (https://arxiv.org/abs/2407.18483)
- **What's New**: 전통적인 평가 메트릭들이 교육적 가치를 간과하는 문제를 해결하기 위해, '지식 종속 가능성(KDA, Knowledge Dependent Answerability)'라는 새로운 자동 평가 메트릭을 제안했습니다.

- **Technical Details**: KDA는 교실 환경에서의 사용성과 KDA 간의 강한 상관관계를 보여주는 인적 평가를 기반으로 하며, 기존 n-gram 기반 메트릭과 결합했을 때 MCQ 품질 예측력을 강화하는 두 개의 자동 평가 메트릭 KDA_disc와 KDA_cont를 도입합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 레이블이 붙은 MCQ 품질 측정치에서 강력한 예측력을 보였으며, 지식 기반 평가에서 더 나은 결과를 제공합니다.



### Multi-turn Response Selection with Commonsense-enhanced Language Models (https://arxiv.org/abs/2407.18479)
- **What's New**: 자동 MCQ 생성에 대한 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안함으로써, MCQ의 학생 지식 평가 능력을 측정하려고 한다.

- **Technical Details**: 기존 메트릭은 n-gram 기반 유사성에만 초점을 맞춰 교육적 가치를 고려하지 않지만, KDA는 학생의 응답을 바탕으로 MCQ의 대답 가능성을 수치화한다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가의 레이블이 붙은 실제 강의실 환경에서의 사용성과 강한 상관관계를 보이며, n-gram 기반 유사성 메트릭과 결합하면 다양한 전문가 레이블링 MCQ 품질 측정의 예측력을 강하게 개선한다.



### Constructing the CORD-19 Vaccine Datas (https://arxiv.org/abs/2407.18471)
- **What's New**: 본 논문에서는 COVID-19 백신 관련 연구를 위한 새로운 데이터셋인 CORD-19-Vaccination을 소개합니다.

- **Technical Details**: 이 데이터셋은 CORD-19 데이터셋에서 추출되었으며, 언어 세부사항, 저자 인구 통계, 키워드 및 각 논문의 주제에 대한 새로운 열로 보강되었습니다. Facebook의 fastText 모델을 사용하여 언어를 식별하고, Google의 검색 API를 통해 저자 인구 통계를 정립하였습니다. 'Yake'를 사용하여 키워드를 추출하고, LDA(잠재 디리클레 할당) 알고리즘을 통해 주제 정보를 추가했습니다.

- **Performance Highlights**: 이 데이터셋은 30,000개의 연구 논문을 포함하고 있으며, NLP 연구, 특히 COVID-19 백신 연구에 관한 텍스트 마이닝, 정보 추출, 질문 응답의 중요한 자원이 될 수 있습니다.



### Fairness Definitions in Language Models Explained (https://arxiv.org/abs/2407.18454)
- **What's New**: 자동 생성된 객관식 질문(MCQ)의 평가 방식에 대한 새로운 접근법과, 언어 모델의 공정성 정의에 대한 체계적인 설문 조사가 제안되었습니다.

- **Technical Details**: MCQ의 대답 가능성(answerability)을 측정하는 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이 제안되었으며, KDA_disc 및 KDA_cont와 같은 두 가지 자동 평가 메트릭이 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한 ML 모델이 사회적 편견을 어떻게 확대할 수 있는지를 설명하고, 이를 해결하기 위한 여러 공정성 개념을 분류하는 새로운 분류법이 제시되었습니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 강의실 환경에서의 실용성과 KDA와 강한 상관관계를 가지며, n-그램 기반 유사성 메트릭과 결합 시 전문가가 제시한 다양한 MCQ 품질 측정에 대한 예측력이 뛰어남을 입증했습니다. 또한, 제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화를 향상시킨 결과를 보여주었습니다.



### Guidance-Based Prompt Data Augmentation in Specialized Domains for Named Entity Recognition (https://arxiv.org/abs/2407.18442)
- **What's New**: 자동 생성된 객관식 질문(MCQ)의 교육적 가치를 평가할 수 있는 새로운 자동 평가 메트릭, 지식 종속 가능성(KDA)을 제안했다. 이 메트릭은 MCQ의 대답 가능성을 측정하며, 학생의 지식 수준에 맞춰 질문의 품질을 평가한다.

- **Technical Details**: 기존의 MCQ 평가 메트릭들은 n-gram 기반의 유사성에 집중했으나, 우리는 KDA를 측정하기 위해 학생의 응답을 기준으로 하여 KDA_disc와 KDA_cont라는 두 가지 새로운 자동 평가 메트릭을 제안하였다. 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가들에 의해 라벨링된 MCQ 품질 지표와 강한 상관관계를 보였다. 나아가, n-gram 기반의 유사성 메트릭과 결합할 때 이들 메트릭은 다양한 MCQ 품질 측정에 대한 예측력을 높였다.



### Self-Directed Synthetic Dialogues and Revisions Technical Repor (https://arxiv.org/abs/2407.18421)
Comments:
          25 pages, 3 figures, 4 tables

- **What's New**: 자동 생성된 Multiple Choice Questions (MCQ)은 교육자들이 평가에 소요되는 시간을 대폭 줄일 수 있는 잠재력을 지니고 있습니다. 그러나 기존의 MCQ 생성 평가 메트릭들은 교육적 가치를 고려하지 않은 채, 단순히 n-그램 (n-gram) 기반의 유사성만을 중점적으로 평가하고 있습니다.

- **Technical Details**: 새로운 자동 평가 메트릭, 지식 종속 가능성 (KDA)을 제안하며, 이는 MCQ의 대답 가능성을 측정하고 학생이 목표 사실에 대한 지식을 평가하는 능력을 평가합니다. KDA는 학생의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안되었습니다. 이는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 KDA 및 실제 교실에서의 사용성과 강한 상관관계를 가지고 있으며, n-그램 기반 메트릭과 결합될 경우 다양한 전문가 레이블의 MCQ 품질 측정에 대한 강력한 예측력을 보여줍니다.



### The Art of Refusal: A Survey of Abstention in Large Language Models (https://arxiv.org/abs/2407.18418)
Comments:
          preprint

- **What's New**: 자동 MCQ 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하며, 이는 MCQ가 특정 사실에 대해 학생의 지식을 어떻게 평가하는지를 측정한다. 또한, 기존 기법과는 달리 LLM(대형 언어 모델)의 중지(abstention) 행동을 세 가지 관점에서 분석한다.

- **Technical Details**: KDA라는 새로운 메트릭을 통해 학생의 응답을 바탕으로 MCQ의 대답 가능성을 평가하고, KDA_disc 및 KDA_cont 두 가지 자동 평가 메트릭을 제안한다. 또한, counterfactual augmentation을 통한 robust한 모델 개발을 목표로 하여, 데이터셋의 spurious correlation 문제를 해결하기 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 이용한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 강의실에서의 사용성과 KDA 메트릭과 강한 상관관계를 가지며, 여러 전문가가 라벨링한 MCQ 품질 측정에서도 좋은 예측력을 나타낸다.



### PersonaGym: Evaluating Persona Agents and LLMs (https://arxiv.org/abs/2407.18416)
Comments:
          21 pages, 5 figures

- **What's New**: 이번 논문은 교육, 헬스케어 및 엔터테인먼트와 같은 다양한 분야에서 사용할 수 있는 'Persona agents'를 평가하기 위한 새로운 평가 프레임워크인 PersonaGym과 자동화된 메트릭인 PersonaScore를 소개합니다.

- **Technical Details**: Persona agents는 주어진 페르소나에 따라 행동하는 LLM agents로서, 해당 페르소나에 맞는 응답을 정렬할 수 있는 가능성을 제공합니다. 그러나 다양한 환경에서 페르소나 준수를 평가하는 것은 매우 복잡합니다. PersonaScore는 이러한 평가를 위한 결정 이론을 기반으로 한 자동화된 메트릭입니다.

- **Performance Highlights**: 우리는 6개의 오픈 및 클로즈드 소스 LLM을 평가했으며, 200개의 페르소나와 10,000개의 질문을 포함한 벤치마크를 사용했습니다. 그 결과, 모델 크기와 복잡성이 증가한다고 해서 성능이 반드시 향상되지 않음을 발견했습니다.



### Robust Claim Verification Through Fact Detection (https://arxiv.org/abs/2407.18367)
- **What's New**: 자동 MCQ 생성, 컨트라스티브 학습, 그리고 주장 검증을 위한 새로운 방법론을 제안하며 이들 방법들이 교육적 가치 및 데이터 로버스트성을 향상시킨다는 점에서 기존 연구와 차별화된다.

- **Technical Details**: 1. 자동 MCQ 생성에서 지식 종속 가능성(KDA)이라는 평가 메트릭을 제안. 2. 컨트라스티브 학습을 통해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 각 용어의 인과관계를 이해. 3. 새로운 FactDetect 접근 방식을 활용하여 짧은 사실을 증거에서 추출하고, 이를 LLMs와 결합하여 주장 검증에서의 설명 가능성과 성능을 개선.

- **Performance Highlights**: 1. KDA_disc와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 강한 상관관계를 보임. 2. 논문의 접근 방식이 zero-shot 주장 검증에서 15% F1 점수 향상을 보이며, AugFactDetect가 가장 잘 수행되는 기준과 비교하여 평균 17.3% 성능 향상을 기록.



### Unveiling Scoring Processes: Dissecting the Differences between LLMs and Human Graders in Automatic Scoring (https://arxiv.org/abs/2407.18328)
Comments:
          Non-archival Presenting at EDM 2024 Workshop on Large Language Models

- **What's New**: 이번 논문에서는 자동 MCQ 생성과 관련된 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)와, counterfactual augmentation 기법, 그리고 대형 언어 모델(LLM)의 채점 메트릭을 개선하기 위한 연구를 제안합니다.

- **Technical Details**: MCQ의 적절성을 평가하기 위해 제안된 KDA는 학생 응답을 기반으로 측정되며, KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭을 사용하여 LLM은 더 깊이 있는 채점 로직을 요구하는 과정에서 인간 채점 기준과의 정렬을 개선할 수 있습니다. 또한, 우리는 '집합적 의사 결정(collective decisions)'을 통해 여러 개의 counterfactuals를 생성하여 인과관계를 이해하는 방법을 모색했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ 품질 측정과의 강한 상관관계를 보였으며, 개선된 채점 정확도를 바탕으로 LLM의 높은 성과를 문서화하였습니다. 고품질의 분석적 채점 기준을 통해 인간 채점 로직을 반영하는 것이 LLM의 채점 정확도를 향상시켰습니다.



### The Need for Guardrails with Large Language Models in Medical Safety-Critical Settings: An Artificial Intelligence Application in the Pharmacovigilance Ecosystem (https://arxiv.org/abs/2407.18322)
Comments:
          27 pages, 6 figures, 4 tables and supplementary material provided

- **What's New**: 이번 연구에서는 자동 MCQ (Multiple Choice Questions) 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 기존의 n-gram 기반 메트릭의 한계를 극복하고자 합니다.

- **Technical Details**: KDA는 학생의 목표 사실에 대한 지식을 기반으로 MCQ의 대답 가능성을 측정하며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭들은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 (1) KDA와 (2) 전문가가 점검한 실제 수업 환경에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 때 다양한 MCQ 품질 척도에 대한 예측력이 뛰어난 성능을 보입니다.



### Wolf: Captioning Everything with a World Summarization Framework (https://arxiv.org/abs/2407.18908)
- **What's New**: 이번 연구에서는 MCQ의 자동 생성 및 평가 방법을 제안합니다. 기존 평가 메트릭이 교육적 가치를 고려하지 않아 지식 종속 가능성을 (Knowledge Dependent Answerability, KDA) 측정하는 새로운 메트릭을 개발했습니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정되며, 자동 평가 메트릭인 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이 메트릭은 n-gram 기반 유사성 측정과 결합하여 MCQ의 품질을 보다 정확하게 예측합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 설정에서의 사용성과 높은 상관관계를 보이며, 전문가로부터 라벨링된 다양한 MCQ 품질 기준에 대해 강력한 예측 능력을 보여주었습니다.



### AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents (https://arxiv.org/abs/2407.18901)
Comments:
          ACL'24 Camera Ready

- **What's New**: 본 연구에서는 학생 평가를 위한 MCQ 자동 생성 방식과 관련하여 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 이 메트릭은 학생의 대상 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: 전통적인 MCQ 생성 평가 기법인 BLEU, ROUGE, METEOR와는 달리, KDA는 학생의 실제 반응을 기반으로 하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이들 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보여주었으며, n-gram 기반의 유사도 메트릭과 결합했을 때 다양한 전문가 평가 MCQ 품질 측정에 대해 강력한 예측력을 나타냈습니다.



### Embedding And Clustering Your Data Can Improve Contrastive Pretraining (https://arxiv.org/abs/2407.18887)
Comments:
          16 pages, 3 figures, 2 tables

- **What's New**: 이 논문에서는 대규모 contrastive pretraining에서 단일 소스 미니배치를 사용하는 것이 전반적인 모델 정확성을 크게 향상시킬 수 있음을 보여준다.

- **Technical Details**: 더 나아가, pretrained text embedding 모델과 k-means clustering 알고리즘을 활용하여 각 소스 내의 의미 클러스터에 따라 훈련 데이터를 세분화하는 방법을 탐구했다.

- **Performance Highlights**: MSMARCO passage retrieval 데이터셋의 query-passage 쌍으로 BERT 기반 텍스트 임베딩 모델을 pretrained 했을 때 NDCG@10의 눈에 띄는 향상을 관찰했다.



### Cluster-norm for Unsupervised Probing of Knowledg (https://arxiv.org/abs/2407.18712)
Comments:
          34 pages, 35 figures

- **What's New**: 자동 생성된 MCQ의 평가 메트릭에 지식 종속 가능성(KDA)을 도입하여 MCQ의 대답 가능성을 측정하는 새로운 접근 방식을 제안하였다.

- **Technical Details**: KDA는 학생의 응답을 바탕으로 평가되며, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 통해 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모방하여 근사화하였다. 또한 counterfactual augmentation을 통해 robust한 의사 결정을 하는 방법을 제안하였고, unsupervised probing 기술을 활용하여 인코딩된 지식을 추출하는 방안을 소개하였다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가에 의해 라벨링된 다양한 MCQ 품질 측정에 대해 강한 예측력을 보였으며, human studies를 통해 실제 교육 환경에서의 활용성과 KDA와 강한 상관관계를 입증하였다. 또한, 제안된 접근 방식은 counterfactual robustness, cross-domain generalization 및 scarce data에서의 generalization을 보였다.



### Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention (https://arxiv.org/abs/2407.18552)
Comments:
          38 Pages, 9 Tables, 12 Figures

- **What's New**: 최근 자동 Multiple Choice Questions (MCQ) 생성의 기술이 교육 평가 시간을 줄일 수 있는 가능성을 지니고 있으나, 기존의 평가 메트릭은 교육적 가치를 고려하지 않는 문제를 가지고 있다. 이를 해결하기 위해, '지식 종속 가능성'(KDA)이라는 새로운 자동 평가 메트릭을 제안하였다.

- **Technical Details**: KDA는 특정 사실에 대한 학생의 지식에 기반하여 MCQ의 대답 가능성(answerability)을 측정한다. 기존의 BLEU, ROUGE, METEOR 같은 n-gram 기반 메트릭은 이러한 교육적 요소를 평가하지 않았다. 우리는 KDA의 측정 방식과 함께 KDA_disc, KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모사하였다.

- **Performance Highlights**: KDA_disc와 KDA_soft는 강의실 환경에서의 사용성과 KDA와의 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합할 경우 MCQ 품질 측정에 대한 강력한 예측력을 보여주었다.



### Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation (https://arxiv.org/abs/2407.18461)
Comments:
          accepted by Interspeech 2024

- **What's New**: 이 논문에서는 새로운 dysarthric speech recognition (DSR) 방법론을 제안하여 보지 못한 dysarthric 화자에 대해 성능을 크게 향상시키며, 전통적인 화자 적응 방법과 달리 추가 조정 (fine-tuning) 없이도 가능하다는 점이 특징이다.

- **Technical Details**: 이 방법은 HuBERT로 훈련된 feature extractor를 사용하여 이전에 보지 못한 화자의 특성을 캡슐화하는 per-word prototypes를 생성한다. 이러한 prototypes는 분류의 기초로 사용되며, supervised contrastive learning을 통해 feature extraction을 개선하여 DSR 성능을 향상시킨다.

- **Performance Highlights**: 우리의 접근법은 representation의 질을 향상시킴으로써 DSR 성능을 개선하여 효과적인 개인화된 DSR을 가능하게 하며, 코드 또한 공개되었다.



### Exploring Bengali Religious Dialect Biases in Large Language Models with Evaluation Perspectives (https://arxiv.org/abs/2407.18376)
Comments:
          10 Pages, 4 Figures. Accepted to the 1st Human-centered Evaluation and Auditing of Language Models Workshop at CHI 2024 (Workshop website: this https URL)

- **What's New**: 본 연구에서는 신뢰할 수 없는 패턴에 의존하는 최근의 deep model들이 NLP 태스크에서 보이는 한계를 극복하기 위해 contrastive learning과 counterfactual augmentation을 활용하는 방법을 제안한다.

- **Technical Details**: 기존의 방법들은 counterfactual을 추가하는 데 인간의 개입이 필요하거나, 기계가 비슷한 counterfactual을 자동으로 찾아야 했으나, 이 연구에서는 '여러 개의' counterfactual을 생성하여 집합적 의사 결정을 통해 더 robust한 인과관계 파악을 시도한다.

- **Performance Highlights**: 우리의 방법은 attribution-based synthesis의 task model bias에 덜 민감하며, counterfactual robustness, cross-domain generalization, 그리고 scarce data에서 일반화 측면에서 상당한 개선을 보인다.



### Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreemen (https://arxiv.org/abs/2407.18370)
- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성의 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하고, 이를 통해 MCQ의 대답 가능성을 평가할 수 있는 방안을 제공한다.

- **Technical Details**: 기존의 평가 메트릭(BLEU, ROUGE, METEOR)은 생성된 MCQ와 데이터셋 내 골드 샘플 간의 n-gram 유사성에 초점을 맞추지만, KDA는 학생의 지식에 따라 MCQ의 대답 가능성을 측정한다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이들은 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가에 의해 라벨링된 MCQ 품질 측정값과 강한 상관관계를 보이며, n-gram 기반 유사성 메트릭과 결합할 경우 강력한 예측력을 발휘한다. 또한, Cascaded Selective Evaluation 프레임워크를 통해 LLM 모델의 평가가 인간의 판단과 강하게 일치함을 보였다.



### AI Safety in Generative AI Large Language Models: A Survey (https://arxiv.org/abs/2407.18369)
- **What's New**: 최근 Generative AI (GAI) 모델의 채택이 증가하면서, 이러한 모델들이 지닌 위험과 안전성에 대한 우려가 커지고 있다. 본 논문에서는 GAI-Large Language Models (LLMs)의 AI 안전성 연구에 대한 최신 동향을 기술적 관점에서 조사한다.

- **Technical Details**: 우리는 LLM의 작동 방식에 대한 간결한 도입과 관련 자료를 통해, 생성 모델의 근본적 제약과 LLM의 매개변수 수 증가에 따른 성능 및 안전성의 무역 관계에 대한 이전 연구를 논의한다. 또한, LLM을 인간의 선호에 맞추는 접근 방식과 그에 따른 도전과제를 파헤친다.

- **Performance Highlights**: 이 연구는 LLM의 안전성을 보장하고 정렬된 모델 개발을 촉진하기 위한 통찰을 제공하고, 향후 LLM이 AI 안전성 분야에서 나아가야 할 방향에 대한 통찰을 제시한다.



### Analyzing Speech Unit Selection for Textless Speech-to-Speech Translation (https://arxiv.org/abs/2407.18332)
- **What's New**: 자동 MCQ 생성의 최신 연구에서는 기존의 평가 메트릭이 교육적 가치를 고려하지 않는 문제를 해결하기 위해 새로운 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)를 제안합니다.

- **Technical Details**: KDA는 학생들의 반응을 기반으로 MCQ의 대답 가능성을 측정합니다. KDA_disc 및 KDA_cont 같은 두 개의 자동 평가 메트릭을 제안하고, 이를 통해 학생들의 문제 해결 행동을 모방하여 KDA를 근사합니다. 이 연구는 또한 여러 번의 Human evaluation을 통해 KDA 메트릭들이 실제 강의실의 사용성과 강한 상관관계를 갖고 있다는 것을 입증합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-그램(n-gram) 기반 유사성 메트릭과 결합했을 때 다양한 전문가가 라벨링한 MCQ 품질 측정에 대해 강력한 예측력을 보여주었습니다. 또한, counterfactual augmentation을 통해 robustness를 높이고, 다양한 차원에서 성과를 개선하는 방법도 제시되었습니다.



### AMA-LSTM: Pioneering Robust and Fair Financial Audio Analysis for Stock Volatility Prediction (https://arxiv.org/abs/2407.18324)
- **What's New**: 자동 다지선다형 질문(MCQ) 생성의 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여, 기존의 평가 방식의 한계를 극복합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반의 유사성 평가 메트릭 대신, 학생의 반응을 바탕으로 KDA를 측정하고, 두 가지 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안합니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 KDA 및 실제 수업 환경에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가 평가 MCQ 품질 측정에 대해 강력한 예측력을 보여주었습니다.



New uploads on arXiv(cs.IR)

### A Flexible and Scalable Approach for Collecting Wildlife Advertisements on the Web (https://arxiv.org/abs/2407.18898)
- **What's New**: 본 논문에서는 MCQ 자동 생성의 새로운 평가 메트릭인 지식 종속 가능성(knowledge dependent answerability, KDA)을 제안하였습니다. 이는 학생이 특정 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: KDA는 인간 설문조사에서 얻은 학생의 응답을 바탕으로 측정됩니다. 또한, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하여 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실 환경에서의 유용성과 강한 상관관계를 가지고 있음을 발견했습니다. 또한, n-gram 기반 유사도 메트릭과 결합했을 때, 다양한 전문가 레이블 MCQ 품질 측정에 대한 예측력이 뛰어난 것으로 나타났습니다.



### Human-artificial intelligence teaming for scientific information extraction from data-driven additive manufacturing research using large language models (https://arxiv.org/abs/2407.18827)
Comments:
          11 pages, 5 Figures, 3 Tables. This paper has been accepted to be published in the proceedings of IDETC-CIE 2024

- **What's New**: 최근 자동 MCQ 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 MCQ가 학생의 지식을 어떻게 평가하는지를 측정할 수 있게 되었다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ 품질 측정과 강한 예측력을 가지며, 강의실 설정에서의 사용성과도 강한 상관관계를 나타냈다.



### REAPER: Reasoning based Retrieval Planning for Complex RAG Systems (https://arxiv.org/abs/2407.18553)
- **What's New**: 자동 MCQ 생성의 교육적 가치에 주목한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 또한, 기존 NLP 모델의 robustness 향상을 위한 contrastive learning과 counterfactual augmentation 방식을 소개합니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 반면, 기존의 counterfactual 생성 방법은 직접 데이터를 수정하는 방식을 사용하며, spurious correlation의 영향을 받습니다. 우리가 제안하는 방법은 다수의 counterfactual을 생성하고 집합적 의사 결정을 통해 인과관계를 보다 강력하게 파악합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 강의실에서의 사용성과 강한 상관관계를 보이며, n-gram 기반 유사도 메트릭과 결합하여 다양한 전문가 라벨링 MCQ 품질 지표에 대해 강력한 예측력을 나타냅니다. 특히, REAPER 모델은 Agent 기반 시스템보다 지연 시간을 대폭 단축시키며 새로운 사용 사례에 쉽게 확장 가능합니다.



### FedUD: Exploiting Unaligned Data for Cross-Platform Federated Click-Through Rate Prediction (https://arxiv.org/abs/2407.18472)
- **What's New**: 자동 MCQ 생성의 기존 평가 메트릭이 교육적 가치를 무시한다는 점을 해결하기 위한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 검증하며, KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 통해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 실제 강의실 세팅에서 전문가에 의해 라벨링된 사용성과 강한 상관관계를 보여주며, n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가 평가 메트릭에 대한 예측력이 뛰어납니다.



### Synergizing Knowledge Graphs with Large Language Models: A Comprehensive Review and Future Prospects (https://arxiv.org/abs/2407.18470)
- **What's New**: 최근 대형 언어 모델(LLMs)과 지식 그래프(KGs)의 통합 연구가 활성화되고 있으며, 이를 통해 LLMs의 지식 부족과 사실 불일치를 보완할 수 있는 새로운 방향성이 제시되었습니다.

- **Technical Details**: 이 논문에서는 KGs와 LLMs의 결합점을 분석하고, 이 두 가지의 통합을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 기존 지식을 통합하고, 새로운 실제 적용 경로를 제시하여 학술 연구의 변환적 영향을 강화하는 것을 목표로 합니다.

- **Performance Highlights**: 새로운 자동 평가 메트릭, KDA(knowledge dependent answerability),를 도입하여 MCQ의 대답 가능성을 측정하며, 이는 교육적 맥락에서의 사용성과 강한 상관관계를 갖는 것으로 나타났습니다. 또한, 대조학습과 반사실적 데이터 강화를 통해 다양한 면에서 성능이 향상되었습니다.



### Supporting Evidence-Based Medicine by Finding Both Relevant and Significant Works (https://arxiv.org/abs/2407.18383)
- **What's New**: 이 논문에서는 의학 정보 검색의 관련성과 신뢰성을 개선하기 위한 새로운 접근법인 Level of Evidence (LoE) 개념을 활용하고 있다.

- **Technical Details**: LoE 프레임워크는 의학 출판물을 기초로 한 경험적 증거에 따라 7개의 별개 레벨로 분류하며, 우리는 이 프레임워크를 통해 MEDLINE 데이터베이스의 2600만 개 이상의 문서에 대해 자동으로 LoE를 할당하는 분류 모델을 개발하였다.

- **Performance Highlights**: TREC PM 데이터셋에서의 검색 실험 결과, LoE를 검색 필터로 사용했을 때 검색 관련성에서 상당한 향상을 보여주었다.



### Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation (https://arxiv.org/abs/2407.18910)
Comments:
          Accepted to CIKM 2024

- **What's New**: 이 논문은 교육자들이 학생 평가에 소요하는 시간을 줄이기 위해 자동 MCQ 생성의 중요성을 강조하고, 기존의 평가 메트릭이 교육적 가치를 간과하고 있음을 지적합니다. 새로운 자동 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 도입하여 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 평가되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하며, n-그램 기반 유사성 메트릭과 결합하여 MCQ 품질 측정의 예측력을 높입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보이며, 전문가가 라벨링한 다양한 MCQ 품질 측정치에 대해 강력한 예측력을 입증합니다. 이 연구는 교육용 문제 생성의 새로운 방향을 제시하며, 실용적인 응용 가능성을 높입니다.



### AutoRDF2GML: Facilitating RDF Integration in Graph Machine Learning (https://arxiv.org/abs/2407.18735)
Comments:
          accepted at ISWC'24

- **What's New**: 이 논문에서는 RDF 데이터를 그래프 머신 학습 작업에 맞춘 데이터 표현으로 변환하기 위한 프레임워크인 AutoRDF2GML을 소개합니다.

- **Technical Details**: AutoRDF2GML은 RDF 데이터 타입 속성에 기반한 콘텐츠 기반 특징과 RDF 객체 속성에 기반한 토폴로지 기반 특징을 모두 생성할 수 있습니다. 이 프레임워크는 자동화된 특징 추출을 통해 RDF 및 SPARQL에 익숙하지 않은 사용자도 그래프 머신 학습 작업을 위해 준비된 데이터 표현을 생성할 수 있도록 합니다.

- **Performance Highlights**: 우리는 AutoRDF2GML을 사용하여 대규모 RDF 지식 그래프에서 생성된 네 가지 새로운 벤치마크 데이터셋을 제시합니다. 이러한 데이터셋은 그래프 신경망과 같은 그래프 머신 학습 방법을 평가하는 데 유용한 자원으로 활용될 수 있습니다. 전반적으로, 이 프레임워크는 그래프 머신 학습과 시맨틱 웹 커뮤니티 간의 간극을 효과적으로 연결합니다.



### Decoding Knowledge Claims: The Evaluation of Scientific Publication Contributions through Semantic Analysis (https://arxiv.org/abs/2407.18646)
Comments:
          This paper was submitted to STI 2024 - 28th International Conference on Science, Technology and Innovation Indicators STI 2024

- **What's New**: 최근의 과학 출판물 증가로 인해 출판 수를 과학 발전의 척도로 사용하기 어려워졌으며, 이로 인해 질과 혁신을 강조하는 대안 지표가 필요해졌다. 이 논문에서는 Relaxed Word Mover's Distance (RWMD)를 제안하여 과학 논문의 혁신성을 평가한다.

- **Technical Details**: RWMD는 의미적 텍스트 유사도를 측정하는 방법으로, 과학 지식의 성장 정도를 보다 효과적으로 평가할 수 있다고 가정한다. 이 연구에서는 RWMD를 사용하여 기초가 되는 논문들을 평가하며, Hirsch의 H-Index 논문을 주요 사례로 삼는다.

- **Performance Highlights**: RWMD 결과를 H-Index 관련 논문, 과학계량학 연구, 무관한 논문 세 그룹 간에 비교하여 중복된 문헌과 진정한 혁신을 구분하려고 하였으며, 지식 주장을 강조하는 것이 과학 기여에 대한 깊이 있는 통찰을 제공한다는 결과를 도출하였다.



### Constructing the CORD-19 Vaccine Datas (https://arxiv.org/abs/2407.18471)
- **What's New**: 본 논문에서는 COVID-19 백신 관련 연구를 위한 새로운 데이터셋인 CORD-19-Vaccination을 소개합니다.

- **Technical Details**: 이 데이터셋은 CORD-19 데이터셋에서 추출되었으며, 언어 세부사항, 저자 인구 통계, 키워드 및 각 논문의 주제에 대한 새로운 열로 보강되었습니다. Facebook의 fastText 모델을 사용하여 언어를 식별하고, Google의 검색 API를 통해 저자 인구 통계를 정립하였습니다. 'Yake'를 사용하여 키워드를 추출하고, LDA(잠재 디리클레 할당) 알고리즘을 통해 주제 정보를 추가했습니다.

- **Performance Highlights**: 이 데이터셋은 30,000개의 연구 논문을 포함하고 있으며, NLP 연구, 특히 COVID-19 백신 연구에 관한 텍스트 마이닝, 정보 추출, 질문 응답의 중요한 자원이 될 수 있습니다.



### Using Bibliometrics to Detect Unconventional Authorship Practices and Examine Their Impact on Global Research Metrics, 2019-2023 (https://arxiv.org/abs/2407.18331)
Comments:
          17 pages, 6 tables, 5 figures

- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성을 위한 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 이 메트릭은 MCQ의 대답 가능성을 측정하여 학생의 특정 지식 평가 능력을 평가합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정하며, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계를 보여주며, n-gram 기반 유사성 메트릭과 결합할 경우, 다양한 전문가 평가 MCQ 품질 측정에 대한 강력한 예측력을 입증합니다.



### The Structure of Financial Equity Research Reports -- Identification of the Most Frequently Asked Questions in Financial Analyst Reports to Automate Equity Research Using Llama 3 and GPT-4 (https://arxiv.org/abs/2407.18327)
Comments:
          JEL classes: C45; G11; G12; G14

- **What's New**: 이번 연구에서는 금융 주식 연구 보고서(ERRs)의 내용을 분류하여 분석하고, 해당 보고서에서 얼마나 자주 특정 정보가 나타나는지를 조사하였습니다. 이를 통해 자동화 가능성과 필요한 인간 판단 요소를 평가했습니다.

- **Technical Details**: 연구팀은 72개의 ERR을 문장 단위로 분석하며 총 4940개의 문장을 169개의 고유 질문 유형으로 분류하였습니다. 질문은 사전에 정의되지 않고 ERR의 진술에서 유도되었습니다. 공공 기업 보고서를 활용하여 질문의 자동화 잠재력을 분류했습니다. 답변이 기업 보고서에서 접근 가능한 경우 '텍스트 추출 가능'으로 레이블이 붙었습니다.

- **Performance Highlights**: 연구 결과, ERR의 78.7% 질문이 자동화 가능하며, 이 중 48.2%는 대형 언어 모델(LLM) 처리에 적합한 텍스트 추출 가능 질문입니다. 전체 질문 중 21.3%만이 인간의 판단이 필요합니다. Llama-3-70B와 GPT-4-turbo-2024-04-09 모델을 통해 연구팀은 ERR의 약 80% 내용을 자동화할 수 있음을 입증하였습니다.



New uploads on arXiv(cs.CV)

### Floating No More: Object-Ground Reconstruction from a Single Imag (https://arxiv.org/abs/2407.18914)
Comments:
          Project Page: this https URL

- **What's New**: 최근 3D 오브젝트 재구성 기술은 주로 오브젝트 형태의 정확성을 개선하는 데 집중해왔으나, 지면(ground)과 카메라 사이의 관계를 정확하게 캡처하지 못하는 한계가 있다. 이를 해결하기 위해 ORG(Object Reconstruction with Ground)라는 새로운 작업을 제안한다.

- **Technical Details**: ORG 방법은 카메라, 오브젝트, 지면 간의 관계를 묘사하기 위해 두 가지의 컴팩트 픽셀 레벨 표현을 사용한다. 이 방법은 기존 단일 이미지 3D 재구성 기술에 비해 오브젝트-지면 기하학을 효과적으로 재구성할 수 있다.

- **Performance Highlights**: 실험 결과, 제안된 ORG 모델은 보이지 않는 데이터에서 오브젝트-지면 기하학을 효과적으로 재구성하며, 그림자 생성(shadow generation)과 오브젝트 자세 조작(pose manipulation)의 품질을 크게 향상시킨다.



### SHIC: Shape-Image Correspondences with no Keypoint Supervision (https://arxiv.org/abs/2407.18907)
Comments:
          ECCV 2024. Project website this https URL

- **What's New**: 자동 MCQ 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)이 제안되었으며, 이는 MCQ의 대답 가능성을 평가하는 방식으로, 학생의 지식에 대한 실제 평가를 포함합니다.

- **Technical Details**: KDA를 측정하기 위해 학생 응답을 기반으로 하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안됩니다. 이 메트릭은 사전 훈련된 언어 모델을 활용하여 학생 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_soft는 실제 학습 환경에서 사용성과 강한 상관관계를 보여주며, n-gram 기반 유사성 메트릭과 결합했을 때 다양하고 전문가에 의해 라벨이 붙은 MCQ 품질 측정의 예측력이 강하다는 것을 입증했습니다.



### A Scalable Quantum Non-local Neural Network for Image Classification (https://arxiv.org/abs/2407.18906)
Comments:
          draft, 13 pages (including references and appendix), 5 figures

- **What's New**: 이 논문은 기존의 평가 메트릭이 아닌, MCQ의 대답 가능성 (KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 학생의 지식을 평가하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MCQ 생성 품질을 평가하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA 및 실제 교실 환경에서의 사용성과 강한 상관관계를 보여주며, n-gram 기반의 유사도 메트릭과 결합될 때 다양한 전문가 레이블 MCQ 품질 측정의 예측력을 강하게 발휘합니다.



### Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistenc (https://arxiv.org/abs/2407.18899)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 Source data-Free Active Domain Adaptation (SFADA)라는 새로운 DA 패러다임을 제안하여, 소스 데이터가 접근 불가능한 상태에서 타겟 도메인에 적응하는 방법을 탐구합니다.

- **Technical Details**: 기존의 접근방식에서는 소스 데이터 없이 정보를 효율적으로 라벨링할 타겟 샘플을 식별하는 것이 어려웠으나, 이 논문에서는 learn from the learnt (LFTL) 방식과 Contrastive Active Sampling을 도입하여 이전 모델의 가설에서 학습하고, 더욱 효율적으로 정보를 제공합니다.

- **Performance Highlights**: 세 가지 널리 사용되는 벤치마크에서 LFTL은 최신 성능을 달성하였고, 컴퓨팅 효율성과 주석 예산 증대에 따른 지속적인 성능 개선을 보여주었습니다.



### Unifying Visual and Semantic Feature Spaces with Diffusion Models for Enhanced Cross-Modal Alignmen (https://arxiv.org/abs/2407.18854)
- **What's New**: 이 논문에서는 MCQ(Multiple Choice Questions) 생성의 자동화를 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. KDA는 학생의 목표 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 미리 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 또한, spurious pattern에 의존하지 않고 여러 개의 counterfactual을 생성해 집합적 의사 결정을 통해 인과관계를 파악합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 라벨링된 MCQ 품질 측정에서 강한 예측력을 보이며, 실제 강의실 설정에서의 사용성과도 강한 상관관계를 나타냅니다. MARNet 모델은 Vireo-Food172와 Ingredient-101 두 개의 벤치마크 데이터셋에서 이미지 정보를 효과적으로 개선하는 성능을 보여주었습니다.



### Scalable Group Choreography via Variational Phase Manifold Learning (https://arxiv.org/abs/2407.18839)
Comments:
          Accepted at ECCV 2024

- **What's New**: 자동 MCQ 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. 또한, contrastive learning과 counterfactual augmentation을 통한 robustness 향상에 대한 연구와 그룹 댄스 모션 생성을 위한 변형 생성 모델을 소개합니다.

- **Technical Details**: KDA 메트릭은 학생의 응답을 바탕으로 대답 가능성을 측정하며, KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭을 이용합니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한, 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 인과 관계를 조사하는 방법을 제안합니다. 그룹 댄스 생성에서는 위상 기반의 변형 생성 모델을 사용하여 자연스러움과 동기화를 유지하면서 무제한의 댄서 수를 지원합니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 Human 평가 결과, KDA와 실제 강의실에서의 사용성과 강한 상관관계를 보여주었으며, n-gram 기반 메트릭과 결합하여 다양한 전문가 라벨링 MCQ 품질 지표에 대한 예측력이 뛰어남을 입증했습니다. 또한, proposed group dance 생성 방법은 높은 충실도와 메모리 최소화를 실현하며 대규모 댄서 생성을 가능하게 합니다.



### Deep Companion Learning: Enhancing Generalization Through Historical Consistency (https://arxiv.org/abs/2407.18821)
Comments:
          ECCV 2024

- **What's New**: 이번 연구에서는 자동 MCQ(다중 선택 질문) 생성 평가를 위한 새로운 메트릭인 지식 종속 가능성(KDA)을 제안하여, 기존의 BLEU, ROUGE, METEOR와 같은 메트릭의 한계를 극복하고자 합니다.

- **Technical Details**: KDA는 학생의 목표 사실(target fact)에 대한 지식에 기반하여 MCQ의 대답 가능성(answerability)을 측정합니다. 특히 실험을 통해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하였으며, 이를 통해 사전 훈련된 언어 모델(pre-trained language models)로 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 KDA 및 실제 강의실 설정에서의 사용성과 강한 상관관계를 가진다는 것을 입증하였고, n-gram 기반 메트릭과 결합할 경우 다양한 전문가가 라벨링한 MCQ 품질 측정의 예측력을 높이는 것으로 나타났습니다.



### Predicting 3D Rigid Body Dynamics with Deep Residual Network (https://arxiv.org/abs/2407.18798)
- **What's New**: 이번 연구는 상호 작용하는 3차원 강체의 동역학을 예측하기 위해 deep residual networks를 적용한 것이다. C++로 구현된 3D 물리 시뮬레이터와 PyTorch로 구성된 딥러닝 모델을 결합한 프레임워크를 제시한다.

- **Technical Details**: 시뮬레이터는 선형 및 각 운동, 탄성 충돌, 유체 마찰, 중력 효과 및 감쇠를 포함한 훈련 데이터를 생성한다. 우리의 deep residual network는 입력 층, 여러 개의 residual block 및 출력 층으로 구성되어 있으며, 3D 동역학의 복잡성을 다룰 수 있도록 설계되었다.

- **Performance Highlights**: 모델은 10,000개의 시뮬레이션 시나리오를 사용하여 성능을 평가하였고, 위치 예측에서 평균 제곱 오차(MSE) 0.015, 방향 예측에서 0.022를 기록하며 기존 방법에 비해 25% 향상된 성과를 보여주었다.



### Benchmarking Dependence Measures to Prevent Shortcut Learning in Medical Imaging (https://arxiv.org/abs/2407.18792)
Comments:
          Accepted to the 15th International Workshop on Machine Learning in Medical Imaging (MLMI 2024)

- **What's New**: 자동 생성된 선택형 질문(MCQ)의 평가에서 교육적 가치를 고려하는 새로운 메트릭, 지식 종속 가능성(KDA)을 제안합니다.

- **Technical Details**: 기존의 평가 메트릭들은 n-gram 기반의 유사성만을 측정하는 반면, KDA는 학생의 목표 사실에 대한 지식에 기반하여 MCQ의 대답 가능성을 평가합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이는 데이터세트의 전문가에 의해 라벨링된 MCQ 품질 지표들과 강한 상관관계를 보입니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 실제 강의실에서는 사용성과 강한 상관관계를 보여주며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 MCQ 품질 지표들에 대해 높은 예측력을 보여줍니다.



### BCTR: Bidirectional Conditioning Transformer for Scene Graph Generation (https://arxiv.org/abs/2407.18715)
Comments:
          9 pages, 3 figures

- **What's New**: 본 연구에서는 MCQ 생성에 대한 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안하며, 이 메트릭은 학생의 대상 사실에 대한 지식을 기반으로 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생의 응답을 바탕으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다. 이 메트릭은 n-gram 기반 유사성 메트릭과 결합 시, 다양한 전문가 레이블의 MCQ 품질 측정치를 강력하게 예측할 수 있습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_soft가 KDA 및 실제 강의실 환경에서의 사용성과 강한 상관관계를 갖는다는 것을 입증하였으며, BCTR 모델은 Visual Genome 및 Open Image V6 데이터셋에서 최첨단 성능을 달성하였습니다.



### PIV3CAMS: a multi-camera dataset for multiple computer vision problems and its application to novel view-point synthesis (https://arxiv.org/abs/2407.18695)
- **What's New**: 본 논문에서는 자동 다지선다형 질문(MCQ) 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. 기존 메트릭은 교육적 가치를 무시하고 n-gram 기반 유사성에만 초점을 맞추고 있습니다.

- **Technical Details**: KDA는 특정 사실에 대한 학생의 지식을 평가하는 MCQ의 대답 가능성을 측정합니다. 우리는 KDA를 평가하기 위해 학생 응답을 기반으로 측정 방법을 제시하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이들은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont는 강의실 환경에서 실제 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합 시 다양한 전문가 평가 MCQ 품질 측정의 예측력을 향상시켰습니다.



### Rapid Object Annotation (https://arxiv.org/abs/2407.18682)
- **What's New**: 이번 논문에서는 자동 MCQ 생성에 대한 새로운 평가 방식인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하며, 기존의 n-gram 기반 메트릭을 보완합니다.

- **Technical Details**: KDA는 학생의 답변을 바탕으로 MCQ의 대답 가능성을 평가하며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계를 보이며, n-gram 기반 유사성 메트릭과 결합할 경우 전문가가 라벨링한 MCQ 품질 측정에 대한 예측 능력이 뛰어납니다.



### A Survey on Cell Nuclei Instance Segmentation and Classification: Leveraging Context and Attention (https://arxiv.org/abs/2407.18673)
- **What's New**: 이 논문에서는 MCQ 생성에 대한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 소개하고, 이를 통해 자동 평가 방법을 개선하는 방법을 제안합니다.

- **Technical Details**: KDA는 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 포함합니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 레이블링한 다양한 MCQ 품질 측정에서 높은 예측력을 보여주며, 실제 교실 환경에서의 실용성과도 강력한 상관관계를 갖고 있습니다.



### A Labeled Ophthalmic Ultrasound Dataset with Medical Report Generation Based on Cross-modal Deep Learning (https://arxiv.org/abs/2407.18667)
- **What's New**: 자동 MCQ 생성의 효율성을 높이기 위해 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하였으며, 이는 MCQ의 대답 가능성을 평가하는 데 초점을 맞추고 있다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 등의 메트릭은 생성된 MCQ와 금본 예제(gold sample) 사이의 n-그램 기반 유사성에만 집중하며, 교육적 가치를 무시하였다. KDA는 인간 응답을 기반으로 하여 평가하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 소개한다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 전문가들이 라벨링한 MCQ 품질 지표와 강한 상관관계를 보이며, n-그램 기반 유사성 메트릭과 결합했을 때 예측력이 뛰어나고, 실제 교실 환경에서도 높은 사용성을 입증하였다.



### Local Binary Pattern(LBP) Optimization for Feature Extraction (https://arxiv.org/abs/2407.18665)
- **What's New**: 자동 MCQ 생성 과정에서 교육적 가치를 평가할 수 있는 새로운 메트릭 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다.

- **Technical Details**: KDA는 학생들의 응답을 바탕으로 MCQ의 대답 가능성을 측정합니다. 이를 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이들은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가에 의해 평가된 실제 강의실 설정에서의 사용성과 높은 상관관계를 보이며, n-gram 기반 유사도 메트릭과 결합했을 때 MCQ 품질에 대한 강한 예측력을 나타냅니다.



### Adversarial Robustification via Text-to-Image Diffusion Models (https://arxiv.org/abs/2407.18658)
Comments:
          Code is available at this https URL

- **What's New**: 본 논문에서는 MCQ(다중 선택 문제) 자동 생성의 교육적 가치 평가를 위한 새로운 메트릭인 지식 종속 가능성(KDA)과, 이를 통한 새로운 평가 방법론을 제안합니다. 이에 더하여, 다양한 태스크에서의 저항성 향상을 위한 여러 방안도 제시하고 있습니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 n-gram 기반 평가 메트릭은 MCQ의 교육적 가치를 간과하고 있습니다. 이에 반해, KDA는 학생의 목표 사실에 대한 지식 기반에서 MCQ의 답변 가능성을 측정합니다. 또한, 이번 연구에서는 대조 학습(contrastive learning)과 반사적 증강(counterfactual augmentation)을 통해 텍스트-이미지 모델의 효율적인 활용을 보여줍니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 교실에서의 사용성과 KDA와 강한 상관관계를 가진다고 나타났습니다. 데이터가 없는 상황에서도 CLIP 모델을 활용하여 저항성을 증대시킬 수 있는 방법을 제시하며, 이전 방식들에 비해 뛰어난 성능 향상을 보여주었습니다.



### Auto DragGAN: Editing the Generative Image Manifold in an Autoregressive Manner (https://arxiv.org/abs/2407.18656)
Comments:
          This paper has been accepted as a poster paper for ACM Multimedia 2024

- **What's New**: 최근 자동 MCQ 생성 분야에서 교육적 가치를 평가할 수 있는 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안하였다. KDA는 MCQ의 대답 가능성을 평가하여 학생의 지식 수준을 측정하는 데 중점을 둔다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 메트릭들은 n-gram 기반으로 생성된 MCQ가 골드 샘플과 얼마나 일치하는지만 평가하였다. 그러나 KDA는 학생의 응답을 바탕으로 MCQ의 대답 가능성을 측정하며, 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안하여 사전 학습된 언어 모델을 활용해 이를 대체한다.

- **Performance Highlights**: 인간 평가와 연구를 통해 KDA_disc와 KDA_cont가 교육 전문가들에 의해 레이블링된 실제 강의실에서의 사용성과 강한 상관관계를 보이며, 기존 n-gram 기반 메트릭과 결합했을 때 MCQ 품질 측정의 예측력이 향상됨을 확인하였다.



### DynamicTrack: Advancing Gigapixel Tracking in Crowded Scenes (https://arxiv.org/abs/2407.18637)
- **summary**: [{"What's New": '자동 MCQ 생성이 교사의 평가 시간을 줄일 수 있는 잠재력이 있지만, 기존의 평가 메트릭들이 교육적 가치를 고려하지 않고 있음. 새로운 자동 평가 메트릭, 지식 종속 가능성(KDA) 제안.', 'Technical Details': 'KDA는 학생의 지식 기반 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 신경망 언어 모델을 활용하여 학생의 문제 해결 행동을 모방.', 'Performance Highlights': 'KDA_disc 및 KDA_soft는 KDA와 실전 강의실에서의 사용성과 강한 상관관계를 보이며, n-그램 기반 유사성 메트릭과 결합했을 때 다양한 전문가 레이블이 있는 MCQ 품질 측정에 대한 강력한 예측력을 보여줌.'}, {"What's New": '딥 모델의 NLP 태스크에서의 높은 정확도에도 불구하고, spurious patterns에 의존하면서 robustness가 제한됨. 이는 counterfactual augmentation과 대조적 학습을 활용하여 개선하고자 함.', 'Technical Details': '기존 방법들이 spurious correlation에 영향을 받는 문제점이 있으며, 본 연구에서는 여러 개의 counterfactual을 생성하여 집합적 의사 결정을 통해 인과관계를 robust하게 파악하는 방법을 제안.', 'Performance Highlights': '상대적으로 과제 모델의 편향에 덜 민감하며, counterfactual robustness, cross-domain generalization, 그리고 데이터가 부족한 상황에서의 일반화에서 상당한 향상을 이루었음.'}, {"What's New": 'Gigapixel 시나리오에서의 객체 추적의 필요성과 기존 알고리즘의 성능 한계를 논의하고, DynamicTrack이라는 동적 추적 프레임워크를 제안.', 'Technical Details': '대조적 학습을 활용하여 보행자의 머리와 몸을 동시에 감지하는 동적 감지기를 제안. 이를 바탕으로 효과적으로 머리와 몸 정보를 사용하는 동적 연관 알고리즘 설계.', 'Performance Highlights': 'DynamicTrack은 gigapixel 혼잡 장면에 특별히 설계된 추적 벤치마크에서 최첨단 성능을 달성했음.'}]



### MOoSE: Multi-Orientation Sharing Experts for Open-set Scene Text Recognition (https://arxiv.org/abs/2407.18616)
Comments:
          Accepted in ICDAR2024

- **What's New**: 이 논문에서는 자동 Multiple Choice Questions (MCQ) 생성의 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하고, 이를 통해 MCQ의 교육적 가치 평가 문제를 해결하려고 하였다.

- **Technical Details**: 기존의 평가 메트릭인 BLEU, ROUGE, METEOR는 생성된 MCQ와 데이터셋에 있는 골드 샘플의 유사성만 비교하는 반면, KDA는 학생의 지식에 따라 MCQ의 대답 가능성을 실제 인간 응답을 바탕으로 측정한다. 이 논문에서는 KDA를 근사하기 위해 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하며, 이는 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: 인간 연구 결과에 따르면 KDA_disc와 KDA_cont는 KDA 및 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가 라벨이 부여된 MCQ 품질 지표에 대해 강력한 예측력을 보여준다.



### LookupForensics: A Large-Scale Multi-Task Dataset for Multi-Phase Image-Based Fact Verification (https://arxiv.org/abs/2407.18614)
Comments:
          Pages 1-13 are the main body of the paper, and pages 14-16 are the supplementary material

- **What's New**: 자동 MCQ 생성에 대한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하고, 교실 설정에서의 사용성을 고려하여 KDA를 평가하는 새로운 방법론을 개발하였다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 메트릭들이 n-gram 기반 유사성에만 중점을 두고 교육적 가치를 평가하지 못하는 문제를 해결하기 위해, KDA는 학생의 응답을 바탕으로 MCQ의 대답 가능성을 측정한다. 또한, KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 환경에서의 사용성과 강한 상관관계를 보이며, n-gram 기반 유사성 메트릭과 결합하였을 때 다양한 전문가 라벨링 MCQ 품질 측정의 예측력이 높아지는 것을 보여주었다.



### Dilated Strip Attention Network for Image Restoration (https://arxiv.org/abs/2407.18613)
- **What's New**: 자동 MCQ 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA) 및 이미지 복원에 대한 새로운 dilated strip attention network(DSAN)을 포함한 최신 연구 결과들이 발표되었습니다.

- **Technical Details**: 먼저 MCQ 생성에서 평가 메트릭으로 KDA를 제안하여, 학생의 지식 기반으로 MCQ의 대답 가능성을 측정합니다. 이미지 복원 기법에서는 DSA 메커니즘을 통해 각 픽셀의 이웃 픽셀에서의 문맥 정보를 효과적으로 수집하며, 다중 스케일 수용 필드를 활용하여 표현 학습을 개선합니다.

- **Performance Highlights**: 우리의 KDA_disc와 KDA_cont는 전문가 라벨링된 MCQ 품질 평가에서 높은 예측력을 보여주며, DSAN은 여러 이미지 복원 태스크에서 최첨단 알고리즘을 초월하는 결과를 도출하였습니다.



### IOVS4NeRF:Incremental Optimal View Selection for Large-Scale NeRFs (https://arxiv.org/abs/2407.18611)
- **What's New**: 이 논문에서는 고화질의 3D 재구성을 위해 Neural Radiance Fields (NeRF)에서 발생하는 다양한 관점에서의 아티팩트를 해결하기 위해 새로운 NeRF 프레임워크를 제안한다.

- **Technical Details**: 제안된 방법은 이미지 내용 및 포즈 데이터를 사용하여 다음 최적의 관점을 반복적으로 계획하는 프로세스를 포함한다. 이 과정에서, 후보 집합에서 최대 정보 이득을 가진 관점을 선택하는 불확실성 추정을 중요한 요소로 한다.

- **Performance Highlights**: 이 iterative process는 시간이 지남에 따라 렌더링 품질을 향상시키며, Vonoroi diagram 및 threshold sampling을 도입하여 효율성을 높이며, 기존의 NeRF 네트워크를 그대로 유지할 수 있다. 이 방법은 baseline 및 유사한 이전 작업들보다 뛰어난 성능을 보여준다.



### LinguaLinker: Audio-Driven Portraits Animation with Implicit Facial Control Enhancemen (https://arxiv.org/abs/2407.18595)
- **What's New**: 이 연구에서는 다국어 오디오 입력과 얼굴 동역학을 동기화하는 방법의 복잡성을 탐구하며, 확산 기반 기법을 통해 시각적으로 매력적이고 시간 동기화된 애니메이션을 생성하는 데 중점을 둡니다. 기존의 매개변수 모델과는 달리, LinguaLinker라 불리는 새로운 접근 방식을 제안하여 오디오 기반 시각적 합성을 통합하고 있습니다.

- **Technical Details**: 우리 방법은 오디오 특징을 별도로 처리하고, 눈, 입, 머리의 움직임을 제어하는 게이트를 유도하여, 초상화의 출처와는 무관하게 움직임을 조절합니다. 이는 오디오 입력과 출력 비디오의 호환성을 유지하면서, 다양한 언어 간에 뚜렷한 인물을 효과적으로 표현할 수 있는 방법입니다.

- **Performance Highlights**: 우리 방법은 애니메이션 초상화의 충실도와 입 모양 싱크의 정확성에서 현저한 개선을 가져오며, 다양한 모션 변화를 통해 각기 다른 언어로 어떤 초상화도 애니메이션할 수 있는 다재다능한 도구로 자리잡고 있습니다.



### Content-driven Magnitude-Derivative Spectrum Complementary Learning for Hyperspectral Image Classification (https://arxiv.org/abs/2407.18593)
Comments:
          accepted by TGRS

- **What's New**: 이 논문에서는 Hyperspectral Image (HSI) 분류를 위한 복잡한 스펙트럼 세부정보에서 구별 가능한 정보를 추출하는 방법을 제안합니다. 이는 현재의 방법들이 스펙트럼 크기 특성에 의존하여 혼란을 일으키고 정확성을 저하시킬 수 있는 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 스펙트럼 크기와 도함수 스펙트럼의 상호 보완성을 활용한 Content-driven Spectrum Complementary Network을 기반으로 하며, Magnitude-Derivative Dual Encoder를 사용하여 두 가지 특성을 결합된 입력으로 사용합니다. Content-adaptive Point-wise Fusion Module을 통해 이중 인코더 특성을 포인트 별로 선택적으로 융합합니다.

- **Performance Highlights**: 이 방법은 WHU-OHS 데이터셋 및 8개의 다른 벤치마크 데이터셋에서 최첨단 결과를 보여줍니다.



### From 2D to 3D: AISG-SLA Visual Localization Challeng (https://arxiv.org/abs/2407.18590)
- **What's New**: 자동 MCQ 생성의 교육적 가치를 평가하기 위해 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하며, 학생의 지식을 평가하는 능력을 강조합니다.

- **Technical Details**: KDA는 학생의 설문 반응을 기반으로 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입하여 학생의 문제 해결 행동을 모방하려고 합니다. 기존 n-gram 기반 유사성 메트릭과 결합할 경우, MCQ 품질 평가에 강력한 예측력을 보여줍니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 KDA와 강한 상관관계를 보여주며, 전문가가 라벨링한 다양한 MCQ 품질 측정에 대한 강력한 예측력을 입증했습니다.



### HICEScore: A Hierarchical Metric for Image Captioning Evaluation (https://arxiv.org/abs/2407.18589)
Comments:
          Accepted by ACM MM2024

- **What's New**: 자동 MCQ 생성, 지식 종속 가능성(KDA), 강력한 이미지 캡셔닝 평가 메트릭인 HICE-S가 제안됨.

- **Technical Details**: 기존의 MCQ 생성 평가 메트릭이 BLEU, ROUGE, METEOR 등 n-gram 기반 유사성에만 의존하던 반면, KDA는 학생의 지식에 기반하여 MCQ의 대답 가능성을 평가하는 새로운 메트릭이다. 또한, SOTA(Sate of the Art) 성능을 달성한 HICE-S는 텍스트 및 지역 기반 평가 방식을 결합해 변화된 평가 메커니즘을 구축했다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA 및 실제 강의실에서의 사용성과 강한 상관관계를 보여주며, 다양한 전문가 라벨링 MCQ 품질 측정에서 강력한 예측력을 보여준다. HICE-S는 기존 평가 메트릭보다 우수하며, 상세한 캡션에 대한 평가 과정이 인간 판단과 유사하다는 연구 결과를 보였다.



### Learning to Enhance Aperture Phasor Field for Non-Line-of-Sight Imaging (https://arxiv.org/abs/2407.18574)
- **What's New**: 깊은 모델들이 NLP 태스크에서 사람보다 나은 정확성을 보였음에도 불구하고, spurious patterns에 의존하는 문제로 인해 robustness가 제한된다는 문제를 해결하기 위해, 새로운 counterfactual augmentation 방식을 제안하였다.

- **Technical Details**: 기존 방법이 spurious correlation에 영향을 받는 데 비해, 본 연구에서는 '여러 개의' counterfactual을 생성하고, 집합적 의사 결정을 통해 각 용어의 인과관계를 보다 robust하게 식별할 수 있는 방법을 제안한다.

- **Performance Highlights**: 우리의 방법은 다양한 차원에서 개선을 보여주었으며, counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화 성능에서 유의미한 향상을 이루었다.



### Learning Spectral-Decomposed Tokens for Domain Generalized Semantic Segmentation (https://arxiv.org/abs/2407.18568)
Comments:
          accecpted by ACMM MM2024

- **What's New**: 자동으로 Multiple Choice Questions (MCQ)를 생성하는 프로세스를 개선하기 위한 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하고, 이 메트릭이 실제 교육 환경에서 학생의 지식을 평가하는 방식에 중점을 둡니다.

- **Technical Details**: 기존의 평가 메트릭들은 n-그램 기반 유사성 (similarity)만을 측정하는 반면, KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 평가합니다. 이 연구에서는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 개발하여 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문 평가자에 의해 라벨링된 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였으며, n-그램 기반 유사성 메트릭과 결합 시 다양한 MCQ 품질 측정치에 대한 예측력을 갖추었습니다.



### VSSD: Vision Mamba with Non-Casual State Space Duality (https://arxiv.org/abs/2407.18559)
Comments:
          16 pages, 5 figures, 7 tables

- **What's New**: 이 논문에서는 자동 MCQ(객관식 질문) 생성 과정에서 교육적 가치를 평가하기 위한 새로운 자동 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. 기존 메트릭들이 교육적 가치를 놓치고 있는 문제를 해결하고자 합니다.

- **Technical Details**: KDA는 특정 사실에 대한 학생의 지식을 기반으로 MCQ의 대답 가능성을 평가합니다. KDA를 측정하기 위해 학생 응답에 대한 Human survey를 활용한 후, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 사용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가에 의해 라벨링된 MCQ 품질 측정 기준에 대해 강한 예측력을 보이며, 기존 n-gram 기반 유사도 메트릭과 결합했을 때 더욱 효율적으로 작용합니다. 연구 결과, 이 두 메트릭은 실제 강의실 세트에서의 사용성과 강한 상관관계를 나타냅니다.



### Skin Cancer Detection utilizing Deep Learning: Classification of Skin Lesion Images using a Vision Transformer (https://arxiv.org/abs/2407.18554)
- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성을 위한 새로운 평가 지표인 지식 종속 가능성(KDA)을 제안하여 학생의 지식을 더욱 효과적으로 평가할 수 있는 방안을 제공합니다.

- **Technical Details**: 기존의 평가 메트릭은 단순히 n-gram 기반의 유사성에 의존하고 있으나, KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 강사에 의해 라벨링된 MCQ 품질 측정의 예측력을 갖추고 있으며, 실제 학습 환경에서의 사용성과 강한 상관관계를 보여줍니다.



### Boosting Cross-Domain Point Classification via Distilling Relational Priors from 2D Transformers (https://arxiv.org/abs/2407.18534)
- **summary**: [{"What's New": '자동 MCQ 생성에서 교육적 가치를 고려한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안했습니다.'}, {'Technical Details': 'KDA는 학생의 응답을 바탕으로 측정되며, 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)을 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 기존의 BLEU, ROUGE, METEOR 같은 n-gram 기반 메트릭과 결합하여 성능을 향상시킵니다.'}, {'Performance Highlights': 'KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 KDA와 강한 상관관계를 보여주었으며, 다양한 전문가 평가 MCQ 품질 측정치를 예측하는 강력한 능력을 보였습니다.'}]



### Text-Region Matching for Multi-Label Image Recognition with Missing Labels (https://arxiv.org/abs/2407.18520)
Comments:
          Accepted to ACM International Conference on Multimedia (ACM MM) 2024

- **What's New**: 이 논문에서는 MCQ 생성의 자동화를 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하고 연구한다. KDA는 학습자가 특정 사실에 대한 지식을 바탕으로 MCQ의 답변 가능성을 평가한다.

- **Technical Details**: 기존의 BLEU, ROUGE 및 METEOR와 같은 메트릭은 MCQ의 교육적 가치를 고려하지 않으며 단순히 n-gram 기반 유사성에 집중한다. KDA는 학생의 응답을 통해 측정되며, 두 개의 새로운 자동 평가 메트릭 KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 설정에서의 사용성과 강한 상관관계를 나타내며, n-gram 기반 유사성 메트릭과 결합할 때 다양한 MCQ 품질 측정에서 강력한 예측 능력을 보여준다.



### Revisit Event Generation Model: Self-Supervised Learning of Event-to-Video Reconstruction with Implicit Neural Representations (https://arxiv.org/abs/2407.18500)
- **What's New**: MCQ(객관식 질문)의 자동 생성에 대한 새로운 평가 메트릭인 지식 종속 가능성(KDA) 제안. KDA는 학생의 지식을 기반으로 MCQ의 대답 가능성을 측정하며, n-gram 기반 메트릭과 결합 시 강력한 예측력을 보임.

- **Technical Details**: KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 두 가지 자동 평가 메트릭이다. 기존 부족한 평가 방법들이 교육적 가치를 경시한 것과 달리, 이들 메트릭은 강사 세트에서의 사용성과도 높은 상관관계를 지닌다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 평가한 MCQ 품질 측정에서 높은 예측력을 뒷받침하며, 이들은 기존 n-gram 기반 메트릭과 결합할 때 성능이 향상됐다. KDA는 실제 강의실 환경에서의 사용성과 교육적 기능을 중요하게 고려하고 있다.



### Answerability Fields: Answerable Location Estimation via Diffusion Models (https://arxiv.org/abs/2407.18497)
Comments:
          IROS2024

- **What's New**: 이 논문에서는 복잡한 실내 환경에서의 대답 가능성(predicting answerability)을 예측하는 새로운 접근 방식인 Answerability Fields를 제안합니다.

- **Technical Details**: 3D 질문 응답 데이터셋을 활용하여 다양한 장면(scene)과 ScanNet에서의 질문들을 포함하는 포괄적인 Answerability Fields 데이터셋을 구축합니다. 확산 모델(diffusion model)을 사용하여 이러한 Answerability Fields를 추론하고 평가합니다.

- **Performance Highlights**: 객체와 그 위치의 중요성을 강조하며, Answerability Fields가 장면 이해(scene-understanding) 작업을 안내하는 데 효과적임을 보여줍니다. 이는 지능형 에이전트와 그들의 환경 간의 상호 작용 향상을 위한 기초를 마련합니다.



### Neural Modulation Alteration to Positive and Negative Emotions in Depressed Patients: Insights from fMRI Using Positive/Negative Emotion Atlas (https://arxiv.org/abs/2407.18492)
- **What's New**: 자동 생성된 MCQ의 평가 메트릭을 단순한 n-그램 유사성에서 교육적 가치에 기반한 '지식 종속 가능성(KDA)'으로 전환하는 혁신적인 접근을 제안하였습니다.

- **Technical Details**: KDA 메트릭은 학생의 응답을 기반으로 측정되며, 자동 평가 메트릭 KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 가지며, n-그램 기반 유사성 메트릭과 결합할 때 전문가가 라벨링한 MCQ 품질 측정에서 강력한 예측력을 보였습니다.



### SMPISD-MTPNet: Scene Semantic Prior-Assisted Infrared Ship Detection Using Multi-Task Perception Networks (https://arxiv.org/abs/2407.18487)
- **What's New**: 자동 MCQ 생성의 평가 메트릭으로 지식 종속 가능성(KDA)을 제안하여 교육적 가치를 반영하고, 강화된 robustness를 위한 contrastive learning과 counterfactual augmentation을 활용하는 방법을 소개합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 평가하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 pre-trained language models를 활용하여 학생의 문제 해결 행동을 모방합니다. 또한, Scene Semantic Prior-Assisted Multi-Task Perception Network (SMPISD-MTPNet) 모델을 이용하여 복잡한 장면에서의 선박 탐지 능력을 향상시키는 방식도 설명됩니다.

- **Performance Highlights**: KDA의 human studies 결과 KDA_disc와 KDA_soft는 실제 강의실 환경에서 높은 사용성과 강한 상관관계를 증명하였으며, SMPISD-MTPNet은 SOTA 방법들과 비교하여 우수한 성능을 보였음을 강조합니다.



### A Progressive Single-Modality to Multi-Modality Classification Framework for Alzheimer's Disease Sub-type Diagnosis (https://arxiv.org/abs/2407.18466)
- **What's New**: 새로운 MCQ 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하여 MCQ의 교육적 가치와 학생의 지식을 평가합니다. 이번 연구에서는 다중 모달리티를 활용한 Alzheimer's disease (AD) 진단 프레임워크도 소개합니다.

- **Technical Details**: KDA는 대상 사실에 대한 학생의 대답 가능성을 측정하고, 자동 메트릭인 KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이와 동시에, AD 진단에서는 텍스트 분리 네트워크와 다중 모달리티 특성 융합 모듈을 설계하여 초기 단계에서 수집된 데이터를 처리합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서 사용성과 강한 상관관계를 가지며, 다양한 전문가 레이블된 MCQ 품질 측정에 대한 예측력이 뛰어납니다. 또한, 제안한 AD 진단 프레임워크는 8280개의 대규모 공개 및 가정 데이터세트에서 기존 방법보다 우수한 성능을 보입니다.



### Textile Anomaly Detection: Evaluation of the State-of-the-Art for Automated Quality Inspection of Carp (https://arxiv.org/abs/2407.18450)
Comments:
          Accepted at the 2023 Australasian Conference on Robotics and Automation (ACRA 2023) Publication url this https URL ISSN: 14482053 Contains 10 pages and three figures

- **What's New**: 자동 생성된 객관식 질문(MCQ)의 평가에서 교육적 가치를 고려한 새로운 메트릭, 지식 종속 가능성(KDA)을 제안하고, 이를 통해 교실 환경에서 학생의 지식을 평가할 수 있는 방법을 모색한다.

- **Technical Details**: 기존의 MCQ 생성 메트릭은 BLEU, ROUGE, METEOR을 사용하였으나, 이는 n-그램(n-gram) 유사성에 초점을 맞춰 교육적 가치와 학생의 지식 평가를 간과했다. 연구에서는 KDA를 기반으로 한 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 도입하여, 사전 훈련된 언어 모델(pre-trained language models)을 사용해 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 교실 환경에서도 강한 상관관계를 보이며, n-그램 기반 유사성 지표와 결합했을 때 다양한 전문가 라벨의 MCQ 품질 지표에 대해 높은 예측력을 가진다.



### HybridDepth: Robust Depth Fusion for Mobile AR by Leveraging Depth from Focus and Single-Image Priors (https://arxiv.org/abs/2407.18443)
- **What's New**: 이번 논문에서는 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안하여 MCQ의 답변 가능성을 측정하고 대상 사실에 대한 학생의 지식을 평가하는 능력을 평가합니다. 또한, counterfactual augmentation을 활용하여 모델의 강건성을 향상시키기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 자동 평가 메트릭 KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 방식으로 KDA를 근사합니다. HYBRIDDEPTH는 Depth from Focus(DFF) 방법의 스케일 정확성과 단일 이미지 깊이 우선 순위의 일반화 능력을 결합하여 모바일 AR의 깊이 추정에 적합한 파이프라인을 제공합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, HYBRIDDEPTH는 DDFF12, NYU Depth v2와 같은 공통 데이터셋에서 최첨단 모델들(SOTA)을 초월하는 성능을 보였습니다. 또한, ARKitScenes 데이터셋에서는 제로샷(zero-shot) 일반화에서 뛰어난 성능을 입증했습니다.



### Mixed Non-linear Quantization for Vision Transformers (https://arxiv.org/abs/2407.18437)
Comments:
          16 pages, 4 figures, under review

- **What's New**: 최근의 자동 MCQ 생성 기술은 교사들의 평가 시간을 절감할 수 있는 잠재력을 가지고 있지만, 기존의 평가 메트릭들은 교육적 가치를 고려하지 않는다. 이 논문에서는 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하여 MCQ의 대답 가능성을 평가하는 방법을 소개한다.

- **Technical Details**: 논문에서는 KDA를 측정하기 위해 데이터에서 학생들의 응답을 기반으로 한 human evaluation을 사용하고, 사전 훈련된 언어 모델(pre-trained language models)을 이용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 다양한 전문가 라벨링된 MCQ 품질 측정 기준에 대해 강력한 예측력을 보여주며, 실제 강의실 환경에서의 사용성과의 상관관계도 높다.



### A Reference-Based 3D Semantic-Aware Framework for Accurate Local Facial Attribute Editing (https://arxiv.org/abs/2407.18392)
- **What's New**: 자동 MCQ 생성 및 평가에 관한 새로운 메트릭인 Knowledge Dependent Answerability (KDA)를 제안하여 학생의 지식 평가 능력을 측정합니다.

- **Technical Details**: KDA는 학생 응답을 바탕으로 측정되며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제공하여 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한, 우리가 제안하는 방법은 다양한 expert-labeled MCQ 품질 측정에서 강력한 예측 능력을 보여줍니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 KDA와 강한 상관관계를 가지며, n-gram 기반의 유사성 메트릭과 결합될 경우 다양한 MCQ 품질 측정에서 예측력이 뛰어난 것으로 나타났습니다.



### UOUO: Uncontextualized Uncommon Objects for Measuring Knowledge Horizons of Vision Language Models (https://arxiv.org/abs/2407.18391)
Comments:
          10 pages

- **What's New**: 본 연구는 MCQ(다중 선택 질문) 자동 생성의 평가를 위한 새로운 메트릭, 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. 이 메트릭은 생성된 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가하는 능력을 반영합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR 등 평가 메트릭은 단순히 데이터셋의 골드 샘플과 n-gram 유사도를 비교하는 데 집중하여, 교육적 가치를 간과했습니다. KDA는 학생의 응답을 기반으로 하여 계산되며, KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 학생의 문제 해결 행동을 모방하도록 설계되었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 강의실에서의 사용성 간의 강한 상관관계를 보여주며, n-gram 기반 유사도 메트릭과 결합했을 때 전문가가 라벨링한 MCQ 품질 측정에서 강력한 예측력을 나타냅니다.



### Neural Surface Detection for Unsigned Distance Fields (https://arxiv.org/abs/2407.18381)
Comments:
          Accepted to ECCV 2024

- **What's New**: 자동 MCQ 생성, 지식 종속 가능성(KDA) 메트릭 제안, 강력한 인과관계 감독 기법, UDF에서 SDF로의 변환 딥러닝 접근법 소개.

- **Technical Details**: MCQ 생성의 기존 평가 메트릭(BLEU, ROUGE, METEOR)은 교육적 가치를 고려하지 않음. KDA 메트릭은 학생의 지식에 기반한 대답 가능성을 평가하며, KDA_disc와 KDA_cont는 미리 훈련된 언어 모델을 활용하여 이를 자동으로 측정. contrastive learning과 counterfactual augmentation 기법을 통해 robuste한 모델을 설계, UDF를 SDF로 변환 가능.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 높은 상관관계. 새로운 접근법은 다양한 면에서 significant improvements를 보여주며, unseen shape에 대해서도 잘 일반화되는 능력을 갖춤. DualMeshUDF와의 결합 시 추가적인 개선 효과.



### SMiCRM: A Benchmark Dataset of Mechanistic Molecular Images (https://arxiv.org/abs/2407.18338)
Comments:
          Under Submission

- **What's New**: 자동화된 MCQ(Multiple Choice Questions) 생성의 시간 효율성을 높이기 위한 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안했습니다. 기존의 평가 메트릭이 MCQ의 교육적 가치를 고려하지 않았던 문제를 해결하였습니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 미리 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서 전문가에 의해 레이블이 지정된 MCQ 품질 측정치와의 강한 상관관계를 보여주며, n-gram 기반 유사도 메트릭과 결합 시 다양한 평가에서 강력한 예측력을 입증하였습니다.



### Several questions of visual generation in 2024 (https://arxiv.org/abs/2407.18290)
Comments:
          12 pages, 3 figures

- **What's New**: 이번 논문에서는 기존 자동 MCQ 생성 메트릭의 한계를 극복하기 위해 지식 종속 가능성(KDA)이라는 새로운 평가 메트릭을 제안한다. KDA는 MCQ의 대답 가능성을 측정하고 학생의 지식을 평가하는 능력을 강조한다.

- **Technical Details**: 기존의 평가 메트릭(예: BLEU, ROUGE, METEOR)은 n-gram의 유사성에 집중하지만, KDA는 인간 조사에서의 학생 반응으로부터 MCQ의 효과성을 측정한다. 두 개의 자동 평가 메트릭, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 KDA를 근사한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 현장 강의에서의 사용성과 KDA와 강한 상관관계를 가지며, n-그램 기반 유사성 메트릭과 결합할 경우, 다양한 전문가가 라벨링한 MCQ 품질 메트릭의 예측력이 크게 향상된다.



### MARINE: A Computer Vision Model for Detecting Rare Predator-Prey Interactions in Animal Videos (https://arxiv.org/abs/2407.18289)
Comments:
          This is an MSc thesis by Zsofia Katona, supervised by the two other authors

- **What's New**: 이번 논문에서는 MCQ (Multiple Choice Questions)의 자동 생성 시 교육적 가치를 평가할 수 있는 새로운 평가 메트릭인 KDA (Knowledge Dependent Answerability)를 제안한다.

- **Technical Details**: 기존의 평가 메트릭은 n-gram 기반 유사성만 고려하지만, KDA는 학생의 지식에 따른 MCQ의 대답 가능성을 측정한다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제시하였으며, 이들은 사전 학습된 언어 모델을 활용하여 학생 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 강의실 설정에서의 사용성과 강한 상관관계를 나타내며, n-gram 기반 유사성 메트릭과 결합 시 전문가가 평가한 MCQ 품질 측정의 예측력이 뛰어난 것으로 나타났다.



### Leveraging Foundation Models via Knowledge Distillation in Multi-Object Tracking: Distilling DINOv2 Features to FairMO (https://arxiv.org/abs/2407.18288)
Comments:
          This is an MSc thesis by Niels Faber, supervised by the two other authors

- **What's New**: 자동 MCQ 생성 및 평가를 위해 새로운 지식 종속 가능성(KDA) 메트릭을 제안하며, 이를 통해 MCQ가 학생의 지식을 얼마나 잘 평가할 수 있는지를 정량화한다. 또한, 시각적 추적(MOT) 분야에서 DINOv2라는 기반 모델을 활용하여 머신 러닝 방법의 가능성을 탐구한다.

- **Technical Details**: KDA 메트릭은 학생의 반응을 기반으로 하고, 자동 평가 메트릭 KDA_disc와 KDA_cont를 제안한다. 딥러닝 모델의 robustness를 높이기 위해 counterfactual augmentation 및 contrastive learning 기법을 사용한다. MOT 분야에서는 teacher-student 아키텍처를 통해 DINOv2와 FairMOT의 결합된 접근 방식을 사용한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가의 라벨링된 MCQ 품질 측정과의 상관관계가 강하다는 것을 보여주며, 시각적 추적에서 제안된 방법론은 특정 시나리오에서 개선을 보이지만 일관되게 FairMOT 모델을 능가하진 못한다.



### HRP: Human Affordances for Robotic Pre-Training (https://arxiv.org/abs/2407.18911)
Comments:
          Accepted to Robotics Science and Systems 2024

- **What's New**: 이 논문에서는 기존의 자동 MCQ 생성 평가 메트릭들이 교육적 가치와 학생의 지식을 평가하는 능력을 고려하지 않는 한계를 극복하기 위하여 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하며, 이를 바탕으로 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 개발하였습니다.

- **Technical Details**: KDA 논문에서는 Human survey를 통해 KDA를 측정한 방법과 함께 KDA를 근사화하기 위해 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 방법을 소개합니다. KDA_disc와 KDA_cont는 기존의 n-gram 기반 유사도 메트릭과 함께 사용할 때 다양한 전문가 라벨링된 MCQ 품질 척도의 예측 성능을 향상시킵니다.

- **Performance Highlights**: 실험적으로, KDA_disc와 KDA_cont는 KDA 및 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, MCQ의 품질을 평가하는 데 있어 뛰어난 예측력을 지녔습니다. 또한, 3000건 이상의 로봇 시험을 통해 제안된 접근 방식이 적어도 15%의 성능 향상을 이루었다고 보고하였습니다.



### Wolf: Captioning Everything with a World Summarization Framework (https://arxiv.org/abs/2407.18908)
- **What's New**: 이번 연구에서는 MCQ의 자동 생성 및 평가 방법을 제안합니다. 기존 평가 메트릭이 교육적 가치를 고려하지 않아 지식 종속 가능성을 (Knowledge Dependent Answerability, KDA) 측정하는 새로운 메트릭을 개발했습니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정되며, 자동 평가 메트릭인 KDA_disc와 KDA_cont는 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이 메트릭은 n-gram 기반 유사성 측정과 결합하여 MCQ의 품질을 보다 정확하게 예측합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 설정에서의 사용성과 높은 상관관계를 보이며, 전문가로부터 라벨링된 다양한 MCQ 품질 기준에 대해 강력한 예측 능력을 보여주었습니다.



### Every Part Matters: Integrity Verification of Scientific Figures Based on Multimodal Large Language Models (https://arxiv.org/abs/2407.18626)
Comments:
          28 pages, 11 figures, under review

- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성을 위한 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'가 제안되었습니다. KDA는 학생의 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont가 제안되어, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한, 이 메트릭들은 n-gram 유사도 평가 메트릭과 결합되었을 때 강력한 예측력을 발휘합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 전문가가 라벨링한 KDA 및 사용성과 강한 상관관계를 가진다는 것을 보여주며, MCQ 품질 측면에서 우수한 예측력과 정확성을 달성했습니다.



### How To Segment in 3D Using 2D Models: Automated 3D Segmentation of Prostate Cancer Metastatic Lesions on PET Volumes Using Multi-Angle Maximum Intensity Projections and Diffusion Models (https://arxiv.org/abs/2407.18555)
Comments:
          11 pages, 2 figures, accepted in the DGM4MICCAI workshop, MICCAI, 2024

- **What's New**: 이 논문에서는 다중 선택 질문(MCQ) 자동 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA, Knowledge Dependent Answerability)을 제안하여 학생의 대상 사실 지식 평가 능력을 측정한다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다. 이러한 메트릭은 n-gram 기반 유사성 메트릭과 결합하여 다양한 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 보여준다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 KDA 간에 강한 상관관계를 보이며, 자동으로 생성된 MCQ의 교육적 효과성을 더욱 향상시킨다.



### Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention (https://arxiv.org/abs/2407.18552)
Comments:
          38 Pages, 9 Tables, 12 Figures

- **What's New**: 최근 자동 Multiple Choice Questions (MCQ) 생성의 기술이 교육 평가 시간을 줄일 수 있는 가능성을 지니고 있으나, 기존의 평가 메트릭은 교육적 가치를 고려하지 않는 문제를 가지고 있다. 이를 해결하기 위해, '지식 종속 가능성'(KDA)이라는 새로운 자동 평가 메트릭을 제안하였다.

- **Technical Details**: KDA는 특정 사실에 대한 학생의 지식에 기반하여 MCQ의 대답 가능성(answerability)을 측정한다. 기존의 BLEU, ROUGE, METEOR 같은 n-gram 기반 메트릭은 이러한 교육적 요소를 평가하지 않았다. 우리는 KDA의 측정 방식과 함께 KDA_disc, KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모사하였다.

- **Performance Highlights**: KDA_disc와 KDA_soft는 강의실 환경에서의 사용성과 KDA와의 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합할 경우 MCQ 품질 측정에 대한 강력한 예측력을 보여주었다.



### She Works, He Works: A Curious Exploration of Gender Bias in AI-Generated Imagery (https://arxiv.org/abs/2407.18524)
Comments:
          11 pages, 8 figures

- **What's New**: 이 논문은 AI가 생성한 건설 노동자 이미지에서 성 불균형을 조사하며, 남성과 여성 인물의 묘사에서 나타나는 차이를 강조하고 있다.

- **Technical Details**: 그리젤다 폴록(Griselda Pollock)의 시각 문화 및 젠더 이론에 기반하여 AI 모델이 여성 인물을 성적화하고 남성 인물은 더 권위 있고 능숙하게 묘사하는 경향이 있음을 분석하였다.

- **Performance Highlights**: 이 연구 결과는 AI가 사회적 편향을 반영하고 지속시킬 수 있는 잠재력을 강조하며, AI 생성 콘텐츠와 관련된 비판적 접근의 필요성을 부각시킨다.



### Lensless fiber endomicroscopic phase imaging with speckle-conditioned diffusion mod (https://arxiv.org/abs/2407.18456)
- **What's New**: 이번 연구에서는 자동 생성된 객관식 질문(MCQ)의 평가 메트릭으로 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하였습니다. 기존의 BLEU, ROUGE, METEOR와 같은 메트릭이 교육적 가치를 평가하지 못하는 문제를 해결하기 위한 것입니다.

- **Technical Details**: KDA는 학생의 응답을 바탕으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 미리 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이 방법은 MCQ의 대답 가능성을 평가하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 레이블의 MCQ 품질 측정치에 대해 강력한 예측력을 가지는 것으로 나타났습니다.



### Towards A Generalizable Pathology Foundation Model via Unified Knowledge Distillation (https://arxiv.org/abs/2407.18449)
- **What's New**: 자동 MCQ 생성의 효율성을 높이기 위해 교육적 가치를 고려한 새로운 평가 메트릭 KDA를 제안하였으며, 이는 학생의 지식을 평가하는 데 중점을 둔다.

- **Technical Details**: KDA는 학생 응답을 기반으로 하여 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 프리트레인된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함으로써 KDA를 근사화한다. 또한, 기존의 spurious correlation 문제를 해결하기 위해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 인과관계를 파악하는 방법을 제안하였다.

- **Performance Highlights**: MCQ 생성 관련 연구에서 KDA_disc와 KDA_cont는 실제로 강의실에서의 사용성과 강한 상관관계를 보였으며, GPFM(Generalizable Pathology Foundation Model)은 39개의 과제에서 새로운 기준을 세우며 뛰어난 모델링 능력을 입증하였다.



### Weighted Risk Invariance: Domain Generalization under Invariant Feature Shif (https://arxiv.org/abs/2407.18428)
- **What's New**: 이 논문에서는 자동 MCQ 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. 기존의 평가 메트릭들이 교육적 가치를 간과하던 문제를 해결합니다.

- **Technical Details**: KDA는 대상 사실에 대한 학생의 지식을 바탕으로 MCQ의 대답 가능성을 측정합니다. 또한 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하여, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 레이블링한 실제 강의실 세트에서의 사용성과 KDA와 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가 레이블링 MCQ 품질 기준의 예측력이 뛰어남을 입증했습니다.



### Adapting Mouse Pathological Model to Human Glomerular Lesion Segmentation (https://arxiv.org/abs/2407.18390)
- **What's New**: 이 논문은 MCQ 생성의 자동화를 개선하기 위해 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이라는 새로운 자동 평가 메트릭을 제안한다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR과 같은 메트릭은 MCQ의 교육적 가치를 평가하지 못하고, KDA는 학생의 지식 기반에서 MCQ의 대답 가능성을 측정한다. KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모방하여 KDA를 근사화하는 자동 평가 메트릭으로 제안되었다.

- **Performance Highlights**: Human evaluation을 통해 KDA_disc와 KDA_cont가 전문가에 의해 레이블링된 실제 강의실 환경에서의 사용성과 높은 상관관계를 보였으며, n-gram 기반 유사도 메트릭과 결합할 경우 MCQ 품질 측정에서 강력한 예측력을 나타냈다.



### KI-Bilder und die Widerst\"andigkeit der Medienkonvergenz: Von prim\"arer zu sekund\"arer Intermedialit\"at? (https://arxiv.org/abs/2407.18363)
Comments:
          in German language

- **What's New**: 이번 연구에서는 자동 Multiple Choice Questions (MCQ) 생성 시 기존 평가 메트릭이 교육적 가치를 고려하지 못하는 문제를 다루며, 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다.

- **Technical Details**: KDA는 학생들의 응답을 기반으로 MCQ의 대답 가능성을 평가하며, pretrained language models를 활용한 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안합니다. 이들은 실험적 연구를 통해 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합하여 다양한 전문가 라벨이 있는 MCQ 품질 측정에서 강한 예측력을 보여주었습니다.



### Retinal IPA: Iterative KeyPoints Alignment for Multimodal Retinal Imaging (https://arxiv.org/abs/2407.18362)
- **What's New**: 이 논문에서는 다중 모달망 망막 이미지 간의 매칭 및 등록을 향상시키기 위한 새로운 프레임워크를 제안합니다.

- **Technical Details**: 제안된 모델은 이전의 학습 기반 특징 탐지 및 서술 방법을 기반으로 하며, 레이블이 없는 데이터를 더 잘 활용하고 관련 키포인트를 재현하도록 제한하기 위해 키포인트 기반 분할(task)을 통합합니다. 이 과정은 자가 감독(self-supervised) 방식으로 진행되며, 동일 이미지의 서로 다른 증강에 대한 분할 일관성을 강제합니다.

- **Performance Highlights**: 두 개의 공개 데이터 세트와 하나의 내부 데이터 세트에 대한 광범위한 평가를 통해 모달리티 불가지론적인 망막 특징 정렬 성능이 크게 향상되었음을 보여줍니다.



### When, Where, and What? A Novel Benchmark for Accident Anticipation and Localization with Large Language Models (https://arxiv.org/abs/2407.16277)
- **What's New**: 이 연구에서는 자율 주행 시스템의 교통 사고 예측 능력을 크게 향상시키는 새로운 프레임워크를 제안합니다. 전통적인 사고 예측 모델은 사건을 예측하는 데 탁월하지만, 발생 장소 및 관련된 대상을 파악하는 데 한계가 있습니다.

- **Technical Details**: 우리는 사건 발생의 '무엇', '언제', '어디서'를 예측할 수 있도록 대형 언어 모델(Large Language Models, LLMs)을 통합하였습니다. 복잡한 주행 장면에서 높은 위험 요소를 우선적으로 고려하는 체인 기반 주의(Attention) 메커니즘을 개발하고, 이를 통해 LLMs에 대한 세부적인 다중 모달(inputs) 처리를 구현했습니다.

- **Performance Highlights**: DAD, CCD, A3D 데이터셋에 대한 실증적 검증 결과, 평균 정밀도(Average Precision, AP) 및 평균 사고까지 소요 시간(Mean Time-To-Accident, mTTA)에서 우수한 성능을 발휘하며 사고 예측 기술의 새로운 기준을 수립했습니다.



New uploads on arXiv(cs.AI)

### Repairing Networks of $\mathcal{EL_\perp}$ Ontologies using Weakening and Completing -- Extended version (https://arxiv.org/abs/2407.18848)
Comments:
          This is a slightly revised and extended version of a paper published at ISWC 2024. arXiv admin note: text overlap with arXiv:2208.00486

- **What's New**: 최근 논문들은 자동 MCQ 생성, 강건성 (robustness)을 위한 대조 학습 (contrastive learning) 및 반사실적 증강 (counterfactual augmentation)을 활용하는 방법을 제안하고 있습니다. 또한, 온톨로지 네트워크의 품질을 개선하기 위한 새로운 프레임워크를 소개합니다.

- **Technical Details**: 첫 번째 논문에서는 KDA(Knowledge Dependent Answerability)라는 새로운 자동 평가 메트릭을 통해 MCQ의 교육적 가치를 높이고 있습니다. 두 번째 논문에서는 집합적 의사 결정(collective decision-making) 방식으로 counterfactual을 생성하여 모델의 편향을 감소시키는 방법을 제안합니다. 마지막으로, 세 번째 논문은 온톨로지 네트워크의 디버깅, 약화 및 완성 작업을 포함한 프레임워크를 통해 품질을 개선하는 방법을 논의합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서 사용성과 강한 상관관계를 보이며, 기존 n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 레이블의 MCQ 품질 측정 예측에서 강력한 성능을 보여줍니다. 두 번째 연구에서는 다양한 차원에서 상당한 개선을 이루어내며, 마지막 연구는 이전 작업의 확장을 위한 청사진을 제공합니다.



### Understanding XAI Through the Philosopher's Lens: A Historical Perspectiv (https://arxiv.org/abs/2407.18782)
Comments:
          10 pages

- **What's New**: 자동 생성된 객관식 질문(MCQ)의 평가를 위한 새로운 메트릭인 지식 종속 대답 가능성(KDA)을 제안하여 교육적 가치를 평가하는 접근법을 제공.

- **Technical Details**: 기존의 평가 메트릭은 n-gram 기반 유사성에 초점을 맞추나, 우리는 KDA를 통해 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정. KDA_disc 및 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 사용한 학생의 문제 해결 행동을 모방.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 교실에서의 유용성과 강한 상관관계를 보이며, n-gram 기반 메트릭과 결합했을 때 다양한 전문가 레이블의 MCQ 품질 측정에 대한 예측력이 강함을 입증.



### Any four real numbers are on all fours with analogy (https://arxiv.org/abs/2407.18770)
- **What's New**: 이번 연구에서는 숫자에 대한 유사성(analogy)을 일반화된 평균(generalized means)에 의존하는 정형화를 제시합니다. 이는 인공지능 및 머신러닝의 최근 발전에 힘입어 성립된 개념입니다.

- **Technical Details**: 이 연구는 'power parameter'라는 개념을 통해 정의된 일반화된 평균에 기초해 유사성의 통일적 관점을 제안합니다. 특히, 증가하는 네 개의 양수는 적절한 거듭 제곱에서 하나의 유사성으로 간주될 수 있습니다.

- **Performance Highlights**: 모든 유사적 방정식은 증가하는 숫자를 위해 해를 가지며, 이는 복소수(complex numbers)에서도 제한 없이 일반화될 수 있다는 기초적인 결과를 도출합니다. 이러한 결과는 숫자가 표현되는 영역에서 유사성에 대한 이해를 증진시킵니다.



### Multi-Robot System Architecture design in SysML and BPMN (https://arxiv.org/abs/2407.18749)
- **What's New**: 이 논문에서는 Multi-Robot System (MRS)의 설계 복잡성을 줄이기 위한 모듈 형 모델링 및 시뮬레이션 기술을 소개합니다.

- **Technical Details**: MRS 모델링은 Systems Modeling Language (SysML) 및 Business Process Model and Notation (BPMN)이라는 두 가지 공식 Architecture Description Languages (ADLs)를 사용하여 달성되며, 이를 통해 기술에 구애받지 않는 구현이 가능합니다. 시뮬레이션 단계에서는 다중 에이전트 환경이 사용되어 Java Agent Development (JADE) 미들웨어 내에서 MRS 청사진을 시뮬레이션합니다.

- **Performance Highlights**: 이 접근 방식을 통해 제안된 MRS 모델은 성능 평가 행렬 형태로 분석 및 검증이 가능하다는 장점이 있습니다.



### Neurosymbolic AI for Enhancing Instructability in Generative AI (https://arxiv.org/abs/2407.18722)
- **What's New**: 최근 Generative AI, 특히 Large Language Models (LLMs)을 통해 콘텐츠 생성 방식이 변화하고 있으며, Instruction tuning이 모델의 지시 이행 능력을 향상시키는 데 기여하고 있다.

- **Technical Details**: Instruction tuning은 특정 작업과 해당 지시에 맞춰 형식화된 데이터셋을 기반으로 LLM을 학습시키는 감독된 미세 조정 기법이다. 본 논문은 neurosymbolic AI를 활용하여 LLM의 지시 가능성을 향상시키기 위한 방법을 탐구한다.

- **Performance Highlights**: Neurosymbolic 접근 방식은 고차원 지시를 구조화된 작업으로 분해하고, 이 작업을 실행 가능한 행동으로 연결하며, 동적으로 상태를 명시적으로 표현하여 LLM이 더 높은 정확성과 유연성으로 다양한 지시 맥락에 반응할 수 있도록 돕는다.



### Cluster-norm for Unsupervised Probing of Knowledg (https://arxiv.org/abs/2407.18712)
Comments:
          34 pages, 35 figures

- **What's New**: 자동 생성된 MCQ의 평가 메트릭에 지식 종속 가능성(KDA)을 도입하여 MCQ의 대답 가능성을 측정하는 새로운 접근 방식을 제안하였다.

- **Technical Details**: KDA는 학생의 응답을 바탕으로 평가되며, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 통해 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모방하여 근사화하였다. 또한 counterfactual augmentation을 통해 robust한 의사 결정을 하는 방법을 제안하였고, unsupervised probing 기술을 활용하여 인코딩된 지식을 추출하는 방안을 소개하였다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가에 의해 라벨링된 다양한 MCQ 품질 측정에 대해 강한 예측력을 보였으며, human studies를 통해 실제 교육 환경에서의 활용성과 KDA와 강한 상관관계를 입증하였다. 또한, 제안된 접근 방식은 counterfactual robustness, cross-domain generalization 및 scarce data에서의 generalization을 보였다.



### Collaborative Evolving Strategy for Automatic Data-Centric Developmen (https://arxiv.org/abs/2407.18690)
Comments:
          23 pages, 7 figures

- **What's New**: 이 논문에서는 자동 데이터 중심 개발(AD^2)이라는 새로운 작업을 소개하고, 이 작업에서 필요한 도메인 전문가 수준의 작업 스케줄링 및 구현 능력의 핵심 과제를 제시합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)의 강력한 복잡한 문제 해결 능력을 활용하여, Collaborative Knowledge-STudying-Enhanced Evolution by Retrieval (Co-STEER)라는 전략을 갖춘 LLM 기반의 자율 에이전트를 제안합니다. Co-STEER 에이전트는 도메인 지식을 확장하고, 도메인별 실천 경험을 축적하여 스케줄링 및 구현 기술을 발전시킵니다.

- **Performance Highlights**: Co-STEER 에이전트는 AD^2 연구에 새로운 지평을 열고, 진화 가능한 스케줄 및 구현 능력을 보유하며, 그 구성 요소의 효과성을 입증하는 많은 실험 결과를 보여줍니다.



### Using GPT-4 to guide causal machine learning (https://arxiv.org/abs/2407.18607)
- **What's New**: 학습 평가를 위한 자동 MCQ 생성에 대한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하였으며, 이는 학생의 지식에 대한 MCQ의 대답 가능성을 평가한다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모사한다. 이 두 메트릭은 n-gram 기반 유사성과 결합했을 때 MCQ 품질 평가의 다양한 전문가 레이블과 강한 예측력을 보인다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 강한 상관관계를 가지며, 전반적으로 MCQ 품질 측정에서 유용하게 활용될 수 있음을 보여준다. 또한, 우리의 방법은 기존의 평가 메트릭보다 더 튼튼한 평가를 제공한다.



### A Black Swan Hypothesis in Markov Decision Process via Irrationality (https://arxiv.org/abs/2407.18422)
- **What's New**: 이 논문에서는 블랙 스완 사건(Black swan events)에 대한 기존 정의의 불완전성을 지적하며, 변화가 없는 환경에서도 위험도가 높은 사건이 발생할 수 있다는 새로운 관점을 제안한다.

- **Technical Details**: 저자들은 공간적 블랙 스완 사건(spatial black swan events)이라는 용어를 도입하여 인간의 가치 및 가능성에 대한 오해로 인해 발생하는 블랙 스완 사건을 구분하고 정의를 수학적으로 공식화한다.

- **Performance Highlights**: 이 정의는 인간의 인식을 합리적으로 수정함으로써 블랙 스완 사건을 예방하기 위한 알고리즘 개발에 기여할 수 있기를 바란다.



### Combining Cognitive and Generative AI for Self-explanation in Interactive AI Agents (https://arxiv.org/abs/2407.18335)
Comments:
          10 pages, 2 figures, 2 tables, 1 appendix, HEXED Workshop @EDM July 2024

- **What's New**: 최근 연구에서 VERA(가상 실험 연구 보조 도구)는 학습자가 복잡한 생태계 시스템의 개념 모델을 구축하고 이를 기반으로 에이전트 기반 시뮬레이션을 실험할 수 있는 탐구 기반 학습 환경으로 소개되고 있다.

- **Technical Details**: 이 연구는 인지 AI와 생성 AI의 융합을 탐구하며, VERA에 기능 모델, 지식 및 추론을 Task--Method--Knowledge (TMK) 언어로 표현하여 부여한다. 또한, ChatGPT, LangChain, Chain-of-Thought를 사용하여 VERA TMK 모델에 기반해 사용자 질문에 대한 답변을 생성한다.

- **Performance Highlights**: VERA에서 생성된 설명의 초기 평가는 66개의 질문에 기초하여 수집되었으며, 유망한 결과로 나타났다.



### Affectively Framework: Towards Human-like Affect-Based Agents (https://arxiv.org/abs/2407.18316)
Comments:
          5 pages, 2 figures, 2 tables

- **What's New**: 이번 연구에서는 게임 환경을 이용한 가상 에이전트 훈련의 새로운 기회를 제안하며, 인공지능 모델이 인간의 감정(Affect)을 관찰 공간의 일부로 포함하는 강화 학습 프레임워크를 제시합니다.

- **Technical Details**: 제안된 	extit{Affectively Framework}는 Open-AI Gym 환경을 기반으로 하며, 감정을 관찰 공간에 통합하여 가상 에이전트의 성능을 개선합니다. 이 프레임워크는 세 가지 게임 환경을 포함하고 있습니다.

- **Performance Highlights**: 이 연구는 제안된 감정 기반 프레임워크의 효과성과 가능성을 검토하기 위한 기초 실험(baseline experiments)을 수행하여, 가상 에이전트의 상호작용에서 보다 향상된 성능을 보입니다.



### SOAP-RL: Sequential Option Advantage Propagation for Reinforcement Learning in POMDP Environments (https://arxiv.org/abs/2407.18913)
- **What's New**: 본 논문은 MCQ(다중 선택 질문) 자동 생성을 위한 새로운 평가 메트릭, 즉 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. 이는 학생이 특정 사실에 대한 지식을 바탕으로 MCQ에 대답할 수 있는 능력을 측정하는 데 중점을 둡니다.

- **Technical Details**: 우리는 KDA를 측정하기 위해 Human survey를 바탕으로 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안하며, 이는 미리 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 이러한 메트릭들은 n-gram 기반 유사도 지표와 결합될 때 MCQ 품질 측정에서 강력한 예측력을 보여줍니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성 및 KDA와 강한 상관관계를 보였습니다. 이는 자동 생성된 MCQ의 품질 평가에 있어 보다 효과적인 접근 방식임을 시사합니다.



### A Scalable Quantum Non-local Neural Network for Image Classification (https://arxiv.org/abs/2407.18906)
Comments:
          draft, 13 pages (including references and appendix), 5 figures

- **What's New**: 이 논문은 기존의 평가 메트릭이 아닌, MCQ의 대답 가능성 (KDA)이라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 학생의 지식을 평가하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MCQ 생성 품질을 평가하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 메트릭들은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA 및 실제 교실 환경에서의 사용성과 강한 상관관계를 보여주며, n-gram 기반의 유사도 메트릭과 결합될 때 다양한 전문가 레이블 MCQ 품질 측정의 예측력을 강하게 발휘합니다.



### Lessons from Learning to Spin "Pens" (https://arxiv.org/abs/2407.18902)
Comments:
          Website: this https URL

- **What's New**: 이번 논문에서는 Pen과 같은 물체를 조작하는 기술의 경계를 확장하여, 고품질 시뮬레이션 데이터를 생성하고 이를 통해 현실 세계에서의 조작 능력을 향상시킴을 보여줍니다.

- **Technical Details**: 강화 학습(reinforcement learning)을 사용하여 고급 정보를 가진 오라클 정책(oracle policy)을 훈련시키고, 시뮬레이션에서 고충실도 궤적 데이터셋을 생성합니다. 이 데이터셋은 센서 모터 정책(sensorimotor policy) 사전 훈련 및 실세계에서의 개방형 루프 궤적 재생(open-loop trajectory replay)에 사용됩니다.

- **Performance Highlights**: 50개 미만의 궤적을 이용하여, 다양한 물리적 특성을 가진 10개 이상의 Pen과 유사한 물체를 회전시키는 정책을 학습합니다.



### AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents (https://arxiv.org/abs/2407.18901)
Comments:
          ACL'24 Camera Ready

- **What's New**: 본 연구에서는 학생 평가를 위한 MCQ 자동 생성 방식과 관련하여 새로운 자동 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 이 메트릭은 학생의 대상 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: 전통적인 MCQ 생성 평가 기법인 BLEU, ROUGE, METEOR와는 달리, KDA는 학생의 실제 반응을 기반으로 하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이들 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보여주었으며, n-gram 기반의 유사도 메트릭과 결합했을 때 다양한 전문가 평가 MCQ 품질 측정에 대해 강력한 예측력을 나타냈습니다.



### Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistenc (https://arxiv.org/abs/2407.18899)
Comments:
          ECCV 2024

- **What's New**: 이 논문에서는 Source data-Free Active Domain Adaptation (SFADA)라는 새로운 DA 패러다임을 제안하여, 소스 데이터가 접근 불가능한 상태에서 타겟 도메인에 적응하는 방법을 탐구합니다.

- **Technical Details**: 기존의 접근방식에서는 소스 데이터 없이 정보를 효율적으로 라벨링할 타겟 샘플을 식별하는 것이 어려웠으나, 이 논문에서는 learn from the learnt (LFTL) 방식과 Contrastive Active Sampling을 도입하여 이전 모델의 가설에서 학습하고, 더욱 효율적으로 정보를 제공합니다.

- **Performance Highlights**: 세 가지 널리 사용되는 벤치마크에서 LFTL은 최신 성능을 달성하였고, 컴퓨팅 효율성과 주석 예산 증대에 따른 지속적인 성능 개선을 보여주었습니다.



### SHANGUS: Deep Reinforcement Learning Meets Heuristic Optimization for Speedy Frontier-Based Exploration of Autonomous Vehicles in Unknown Spaces (https://arxiv.org/abs/2407.18892)
- **What's New**: 이 논문은 SHANGUS라는 깊은 강화 학습 (Deep Reinforcement Learning, DRL)과 휴리스틱 최적화를 결합한 새로운 프레임워크를 소개하여 알려지지 않은 환경에서의 탐색 효율성을 개선하는 방법을 제안합니다.

- **Technical Details**: SHANGUS는 DRL의 적응성과 휴리스틱 우선화를 활용하여 탐색 효율성을 크게 향상시키고, 완료 시간과 이동 거리를 최소화합니다. 프론티어 선택 노드와 Twin Delayed Deep Deterministic Policy Gradient (TD3) 알고리즘을 사용하는 DRL 내비게이션 노드를 포함하여 robust한 경로 계획과 동적인 장애물 회피를 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, SHANGUS는 기존의 Nearest Frontier (NF), Novel Frontier-Based Exploration Algorithm (CFE), Goal-Driven Autonomous Exploration (GDAE) 알고리즘과 비교하여 복잡한 시나리오에서 특히 우수한 완료 시간, 이동 거리 및 탐색 비율을 보여주었습니다.



### Generative Adversarial Networks for Imputing Sparse Learning Performanc (https://arxiv.org/abs/2407.18875)
- **What's New**: 이 논문은 교육 관련 자동 문제 생성 및 평가 메트릭에 관한 여러 연구 결과를 다루고 있습니다. 특히, 기존의 평가 메트릭이 교육적 가치를 고려하지 못하는 문제를 해결하기 위해 지식 종속 가능성(KDA)과 같은 새로운 자동 평가 메트릭을 제안합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정합니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입하여 사전 훈련된 언어 모델(pre-trained language models)을 활용해 학생의 문제 해결 행동을 모방합니다. 또한, Generative Adversarial Imputation Networks(GAIN)를 활용하여 sparse learning performance data를 3D 텐서 공간에서 보완합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 라벨이 있는 MCQ 품질 측정에서 강력한 예측 능력을 나타내며, GAIN 접근 방식은 기존 방법보다 높은 데이터 보완 정확성을 보여줍니다. 이러한 연구는 AI 기반 교육에서의 학습 데이터 모델링의 향상에 기여합니다.



### Engaging with Children's Artwork in Mixed Visual-Ability Families (https://arxiv.org/abs/2407.18874)
- **What's New**: 자동 MCQ 생성과 관련된 새로운 평가 메트릭, 지식 종속 가능성(KDA)을 제안하여 학생의 지식 평가 능력을 보다 효과적으로 측정합니다. 또한, AI를 활용하여 시각 장애 가족 구성원과 자녀의 예술 작품과의 상호작용 방식을 탐구합니다.

- **Technical Details**: KDA 메트릭은 인간 설문 응답을 바탕으로 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동화된 평가 메트릭을 사용합니다. 이러한 메트릭은 사전 훈련된 언어 모델을 활용하여 학생 문제 해결 행동을 모방합니다. AI는 또한 자녀의 예술 작품에 대한 설명을 생성하여 시각 장애 가족 구성원이 대화에 참여하도록 도와줍니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 레이블을 붙인 MCQ 품질 척도를 예측하는 데 강한 예측력을 보였으며, 교육적 사용성의 강한 상관관계를 입증했습니다. AI 생성된 설명은 가족 간의 대화 촉진과 자녀의 자발적인 예술 탐색에도 기여할 수 있음을 보여줍니다.



### Unifying Visual and Semantic Feature Spaces with Diffusion Models for Enhanced Cross-Modal Alignmen (https://arxiv.org/abs/2407.18854)
- **What's New**: 이 논문에서는 MCQ(Multiple Choice Questions) 생성의 자동화를 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. KDA는 학생의 목표 사실에 대한 지식을 바탕으로 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여, 미리 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 또한, spurious pattern에 의존하지 않고 여러 개의 counterfactual을 생성해 집합적 의사 결정을 통해 인과관계를 파악합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 라벨링된 MCQ 품질 측정에서 강한 예측력을 보이며, 실제 강의실 설정에서의 사용성과도 강한 상관관계를 나타냅니다. MARNet 모델은 Vireo-Food172와 Ingredient-101 두 개의 벤치마크 데이터셋에서 이미지 정보를 효과적으로 개선하는 성능을 보여주었습니다.



### Enhancing material property prediction with ensemble deep graph convolutional networks (https://arxiv.org/abs/2407.18847)
Comments:
          9 pages, 6 figures, 2 tables

- **What's New**: 이번 연구는 여러 ML 모델이 새로운 재료 발견 및 디자인 과정을 가속화하는 방법을 강조하며, 특히 ensemble 전략을 사용하여 깊은 그래프 신경망에서 재료 성질 예측의 정확도를 높이는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구에서는 Crystal Graph Convolutional Neural Network (CGCNN)와 그 다중 작업 버전인 MT-CGCNN을 실험하여, ensemble 기법이 formation energy per atom ($\Delta E^{f}$), band gap ($E_{g}$) 및 density ($\rho$)과 같은 주요 성질들에서 전통적인 메트릭을 넘는 정확도를 향상시키는 것을 보여주었습니다.

- **Performance Highlights**: 33,990개의 안정한 무기 재료를 대상으로 한 연구 결과, ensemble 기법, 특히 prediction averaging이 높은 정확도를 달성하여 ML 및 DL 분야에서의 예측 정확도를 향상시킬 수 있음을 입증하였습니다.



### Human-artificial intelligence teaming for scientific information extraction from data-driven additive manufacturing research using large language models (https://arxiv.org/abs/2407.18827)
Comments:
          11 pages, 5 Figures, 3 Tables. This paper has been accepted to be published in the proceedings of IDETC-CIE 2024

- **What's New**: 최근 자동 MCQ 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 MCQ가 학생의 지식을 어떻게 평가하는지를 측정할 수 있게 되었다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정하고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 전문가가 라벨링한 MCQ 품질 측정과 강한 예측력을 가지며, 강의실 설정에서의 사용성과도 강한 상관관계를 나타냈다.



### Online Planning in POMDPs with State-Requests (https://arxiv.org/abs/2407.18812)
- **What's New**: 이 논문에서는 자동 MCQ 생성의 평가에서 교육적 가치를 반영할 수 있는 새로운 메트릭인 지식 종속 가능성(KDA)를 제안한다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 메트릭은 n-gram 기반의 유사성에 중점을 두고 있으며, 학생의 지식 평가 능력을 간과한다. KDA는 학생의 응답을 기반으로 MCQ의 대답 가능성을 측정하고, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다.

- **Performance Highlights**: 실험 결과, KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실에서의 사용성과 높은 상관관계를 보이며, n-gram 기반의 유사성 메트릭과 결합했을 때 MCQ 품질 측정의 강력한 예측력을 나타낸다.



### Learning Chaotic Systems and Long-Term Predictions with Neural Jump ODEs (https://arxiv.org/abs/2407.18808)
- **What's New**: 이번 연구에서는 여러 개의 상반된 사실(counterfactual)을 생성하고, 이를 통해 MCQ의 교육적 가치를 평가하는 새로운 자동 평가 메트릭(Knowledge Dependent Answerability, KDA)을 제안합니다. 또한, KDA를 근사화하기 위해 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 도입합니다.

- **Technical Details**: 제안한 KDA 메트릭은 학생의 지식 수준에 따라 MCQ의 대답 가능성을 측정하며, 선행 학습된 언어 모델을 통해 학생의 문제 해결 행동을 모사하여 실험을 진행합니다. 특히, KDA는 기존의 n-gram 기반 유사성 메트릭과 결합되어 다양한 전문가가 라벨링한 MCQ 품질 측정에서 강력한 예측력을 보여주었습니다.

- **Performance Highlights**: 인간 평가를 통해 KDA_disc와 KDA_cont가 실제 강의실 환경에서 KDA와 사용성(expert usability) 모두와 강한 상관관계를 가지며, 이는 MCQ의 질을 평가하는 데 매우 유용한 지표로 확인되었습니다.



### Robust Learning in Bayesian Parallel Branching Graph Neural Networks: The Narrow Width Lim (https://arxiv.org/abs/2407.18807)
- **What's New**: 이번 논문에서는 Bayesian Parallel Branching Graph Neural Network (BPB-GNN) 의 좁은 폭 한계에서의 학습 개선을 제안하며, 기존의 넓은 폭 네트워크가 일반화에 기여한다는 통념에 도전합니다.

- **Technical Details**: BPB-GNN 아키텍처는 residual networks를 닮았으며, 일반적으로 훈련 예제의 수에 비해 폭이 significantly 적을 때 각 브랜치가 커널 재정규화의 대칭 깨짐(symmetry breaking) 덕분에 더 견고한 학습을 수행합니다.

- **Performance Highlights**: 좁은 폭 한계의 BPB-GNN 성능은 일반적으로 넓은 폭 한계에서의 성능과 유사하거나 더 우수하게 나타나며, 각 브랜치의 읽기 규범(readout norms)은 아키텍처의 하이퍼파라미터와는 독립적입니다.



### TAGIFY: LLM-powered Tagging Interface for Improved Data Findability on OGD portals (https://arxiv.org/abs/2407.18764)
- **What's New**: 이 논문에서는 열린 정부 데이터(OGD)에서 데이터셋의 태깅을 자동화하여 데이터 찾기를 개선하기 위한 솔루션을 제안합니다.

- **Technical Details**: 제안된 솔루션 Tagify는 GPT-3.5-turbo 및 GPT-4와 같은 대형 언어 모델(LLM)을 활용하여 데이터셋의 태그를 자동으로 생성합니다. 이 시스템은 영어와 에스토니아어로 태그를 생성하여 데이터 출처의 메타데이터 준비 과정에 기여합니다.

- **Performance Highlights**: 에스토니아의 열린 데이터 포털에서 약 11%의 데이터셋에는 태그가 없고 26%는 단 하나의 태그만 지정되어 있다는 분석 결과가 있었습니다. Tagify는 이러한 문제를 해결함으로써 OGD 포털의 데이터 접근성과 찾기 용이성을 개선하는 데 기여할 것으로 기대됩니다.



### Evaluating Human Trajectory Prediction with Metamorphic Testing (https://arxiv.org/abs/2407.18756)
Comments:
          MET'24: 9th ACM International Workshop on Metamorphic Testing

- **What's New**: 이번 논문에서는 무작위 human trajectory 예측의 정확성을 높이기 위해 idiosyncratic한 metamorhpic testing 방법을 탐구하였다.

- **Technical Details**: Metamorphic testing은 테스트 기준이 명확하지 않거나 없는 경우에 적합하며, input의 변형을 통해 기대하는 human behavior의 많은 대칭성을 활용한다.

- **Performance Highlights**: 이 방법을 통해 stochastic human trajectory 예측의 정확성을 높일 수 있으며, Wasserstein Violation Criterion을 도입하여 label-preserving metamorphic relation이 위배되는지를 통계적으로 평가할 수 있다.



### Score matching through the roof: linear, nonlinear, and latent variables causal discovery (https://arxiv.org/abs/2407.18755)
- **What's New**: 자동 생성된 객관식 질문(MCQ)의 평가에서 기존 방식이 교육적 가치를 고려하지 않고 있다는 문제를 해결하기 위해, 지식 종속 가능성(KDA)이라는 새로운 자동 평가 메트릭을 제안합니다.

- **Technical Details**: KDA는 대상 사실에 대한 학생의 지식에 기반하여 MCQ의 대답 가능성을 평가합니다. 이 연구는 KDA를 학생 응답을 통해 측정하고, KDA_disc 및 KDA_cont라는 두 개의 자동 평가 메트릭을 제안하여, 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사도 메트릭과 결합했을 때 다양한 전문가 라벨링 MCQ 품질 측정에 대해 강력한 예측력을 나타냅니다.



### Knowledge Graph Structure as Prompt: Improving Small Language Models Capabilities for Knowledge-based Causal Discovery (https://arxiv.org/abs/2407.18752)
Comments:
          accepted at ISWC'24

- **What's New**: 이번 연구에서는 MCQ(다중 선택 질문) 생성의 평가 메트릭으로 '지식 종속 가능성(KDA)'을 제안하여, MCQ의 대답 가능성을 요약 평가하는 방법을 소개합니다. 기존의 평가 메트릭들은 교육적 가치와 학생의 지식 평가 능력을 고려하지 않았던 문제를 다룹니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 하여 MCQ의 대답 가능성을 측정합니다. KDA를 근사하는 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하며, 이들은 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다. 또한, 교육에서의 사용 가능성과 KDA 간에 강한 상관관계를 보여주었습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 라벨링을 받은 다양한 MCQ 품질 측정값에 대해 강력한 예측력을 보여주었으며, 기존 n-그램 기반 유사도 메트릭과 결합했을 때 더욱 우수한 성능을 발휘했습니다.



### Towards Generalized Offensive Language Identification (https://arxiv.org/abs/2407.18738)
Comments:
          Accepted to ASONAM 2024

- **What's New**: 본 논문에서는 기존의 MCQ (Multiple Choice Questions) 생성기법의 한계를 극복하기 위해, 새로운 자동 평가 메트릭인 'Knowledge Dependent Answerability (KDA)'를 제안합니다. 이는 학생의 목표 사실(target fact)에 대한 지식을 고려하여 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 pre-trained language models를 활용하여 학생의 문제 해결 행동을 모사합니다. 이 접근법은 일반적인 n-gram 기반 유사도 메트릭과 결합하여 MCQ의 품질을 더욱 정교하게 평가할 수 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 교실 환경에서의 전문가에 의해 라벨링된 사용성과 KDA와 강한 상관관계를 보였으며, 다양한 전문가 라벨링 MCQ 품질 측정에 대해 예측력이 뛰어난 것으로 나타났습니다.



### AutoRDF2GML: Facilitating RDF Integration in Graph Machine Learning (https://arxiv.org/abs/2407.18735)
Comments:
          accepted at ISWC'24

- **What's New**: 이 논문에서는 RDF 데이터를 그래프 머신 학습 작업에 맞춘 데이터 표현으로 변환하기 위한 프레임워크인 AutoRDF2GML을 소개합니다.

- **Technical Details**: AutoRDF2GML은 RDF 데이터 타입 속성에 기반한 콘텐츠 기반 특징과 RDF 객체 속성에 기반한 토폴로지 기반 특징을 모두 생성할 수 있습니다. 이 프레임워크는 자동화된 특징 추출을 통해 RDF 및 SPARQL에 익숙하지 않은 사용자도 그래프 머신 학습 작업을 위해 준비된 데이터 표현을 생성할 수 있도록 합니다.

- **Performance Highlights**: 우리는 AutoRDF2GML을 사용하여 대규모 RDF 지식 그래프에서 생성된 네 가지 새로운 벤치마크 데이터셋을 제시합니다. 이러한 데이터셋은 그래프 신경망과 같은 그래프 머신 학습 방법을 평가하는 데 유용한 자원으로 활용될 수 있습니다. 전반적으로, 이 프레임워크는 그래프 머신 학습과 시맨틱 웹 커뮤니티 간의 간극을 효과적으로 연결합니다.



### Graph Neural Networks for Virtual Sensing in Complex Systems: Addressing Heterogeneous Temporal Dynamics (https://arxiv.org/abs/2407.18691)
Comments:
          This paper extends our previous conference paper (Best Paper at European Conference of the PHM Society 2024, this https URL)

- **What's New**: 이번 연구에서는 MCQ(다중 선택 질문)의 자동 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. KDA는 MCQ가 특정 대상 사실에 대한 학생의 지식을 얼마나 잘 평가하는지를 측정합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 평가 방법들은 MCQ의 단어 유사성에만 초점을 맞추었으나, KDA는 학생의 응답을 기반으로 하여 MCQ의 대답 가능성을 평가합니다. 또한, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 통해 KDA를 근사하는 방법을 제안합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 교실 설정에서의 사용성과 강한 상관관계를 나타내며, 전문가가 레이블링한 다양한 MCQ 품질 측정과도 높은 예측력을 보여줍니다. 이 연구는 MCQ 생성의 자동화에 있어 교육적 가치의 평가를 가능하게 합니다.



### Every Part Matters: Integrity Verification of Scientific Figures Based on Multimodal Large Language Models (https://arxiv.org/abs/2407.18626)
Comments:
          28 pages, 11 figures, under review

- **What's New**: 자동 Multiple Choice Questions (MCQ) 생성을 위한 새로운 평가 메트릭 'Knowledge Dependent Answerability (KDA)'가 제안되었습니다. KDA는 학생의 지식을 바탕으로 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont가 제안되어, 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한, 이 메트릭들은 n-gram 유사도 평가 메트릭과 결합되었을 때 강력한 예측력을 발휘합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 실제 강의실 설정에서 전문가가 라벨링한 KDA 및 사용성과 강한 상관관계를 가진다는 것을 보여주며, MCQ 품질 측면에서 우수한 예측력과 정확성을 달성했습니다.



### Topology Optimization of Random Memristors for Input-Aware Dynamic SNN (https://arxiv.org/abs/2407.18625)
Comments:
          15 pages, 5 figures

- **What's New**: 이 논문에서는 MCQ(다중 선택 질문)의 자동 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)와 기존의 spurious pattern 문제를 해결하기 위한 새로운 방법론을 제안합니다.

- **Technical Details**: MCQ 생성의 KDA는 특정 사실에 대한 학생의 응답을 기반으로 하여 평가하였고, KDA_disc와 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 다른 연구에서는 spurious correlation 문제를 해결하기 위해 여러 개의 counterfactual을 생성하고, 집합적으로 결정하는 방식으로 robust성을 강화합니다. 또한, PRIME이라는 입력 인식 동적 메모리스티브 스파이킹 신경망 최적화를 통한 에너지 효율적인 처리를 제안했습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서 전문가에 의해 평가된 사용성과 KDA와 강한 상관관계를 보였으며, 새로운 모델은 이미지 분류 및 인페인팅에서 62.50배의 에너지 효율성 향상과 최대 77.0%의 계산 부하 절감을 달성했습니다.



### Climbing the Complexity Ladder with Expressive Attention (https://arxiv.org/abs/2407.18601)
- **What's New**: 자동 MCQ 생성(Automatic MCQ Generation)은 교육자의 평가 시간을 크게 줄이는 가능성을 가지고 있으며, 새로운 평가 메트릭인 지식 종속 가능성(KDA, Knowledge Dependent Answerability)을 제안하여 MCQ의 교육적 가치를 평가한다.

- **Technical Details**: 기존 평가 메트릭은 BLEU, ROUGE, METEOR과 같은 n-gram 기반 유사성에 초점을 맞추었지만, KDA는 학생의 지식에 기반한 대답 가능성을 측정한다. KDA_disc와 KDA_cont는 프리트레인된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사화한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 강의실 내에서의 사용성에 강한 상관관계를 보이며, 여러 전문가의 레이블이 있는 MCQ 품질 측정 측면에서도 뛰어난 예측력을 가진다.



### Reinforcement Learning for Sustainable Energy: A Survey (https://arxiv.org/abs/2407.18597)
Comments:
          22 pages excluding references, 40 pages including references, 7 images

- **What's New**: 이 논문에서는 지속 가능한 에너지로의 전환을 위한 강화 학습(reinforcement learning) 기술의 최신 동향을 조사합니다. 에너지 생산, 저장, 전송 및 소비 단계에서의 새로운 의사 결정 과제를 모델링하는 방법을 제공합니다.

- **Technical Details**: 논문은 지속 가능한 에너지와 기계 학습의 연구 영역 간의 격차를 메우기 위한 체계적인 문헌 조사를 수행합니다. 특히, 강화 학습 문제로 모델링할 수 있는 지속 가능성 문제 및 현재 문헌에서 존재하는 해결 방안들을 리스트화합니다.

- **Performance Highlights**: 여러 에이전트(multi-agent), 오프라인(offline), 안전한 강화 학습(safe reinforcement learning)과 같은 지속 가능성 전반에서 나타나는 주요 강화 학습 주제들을 식별하고, 두 연구 분야를 연결하기 위한 환경 표준화의 필요성을 강조합니다.



### Speech Bandwidth Expansion Via High Fidelity Generative Adversarial Networks (https://arxiv.org/abs/2407.18571)
- **What's New**: 자동 생성된 Multiple Choice Questions (MCQ)의 평가 메트릭에 대한 혁신적인 접근법이 소개되었습니다. 기존의 BLEU, ROUGE, METEOR 등의 메트릭이 교육적 가치를 무시하는 반면, 이 연구에서는 지식 종속 가능성(KDA)을 제안하여 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 평가되고, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭이 제안됩니다. 이 두 메트릭은 사전 학습된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하고, KDA와 강한 상관관계를 보입니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가들이 평가한 다양한 MCQ 품질 지표에 대한 강력한 예측력을 보여주며, 실제 강의실 환경에서도 우수한 사용성을 입증했습니다.



### PP-TIL: Personalized Planning for Autonomous Driving with Instance-based Transfer Imitation Learning (https://arxiv.org/abs/2407.18569)
- **What's New**: 이번 연구에서는 MCQ(Multiple Choice Questions) 생성을 자동화하는 새로운 평가 메트릭인 Knowledge Dependent Answerability (KDA)를 제안합니다. 기존의 평가 메트릭들은 교육적 가치를 평가하지 못했으나, KDA는 학생의 지식에 대한 MCQ의 대답 가능성을 측정합니다.

- **Technical Details**: KDA는 학생의 응답을 바탕으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이 두 메트릭은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한, 이후 연구에서 이 두 메트릭은 n-gram 기반 유사성 메트릭과 결합될 때, 전문가가 레이블된 MCQ 품질 측정의 예측력이 강한 것으로 나타났습니다.

- **Performance Highlights**: 우리는 KDA_disc와 KDA_cont가 KDA 및 실제 강의실에서의 사용성과 강한 상관관계를 가지며, 전문가 평가에 의해 레이블이 붙은 데이터에서 다양한 품질 측정에 대해 우수한 성능을 보임을 입증했습니다.



### Learning Robust Named Entity Recognizers From Noisy Data With Retrieval Augmentation (https://arxiv.org/abs/2407.18562)
- **What's New**: 자동 MCQ 생성 및 노이즈가 있는 텍스트에서의 Named Entity Recognition (NER) 모델의 향상을 위한 새로운 방법을 제안함.

- **Technical Details**: 자동 MCQ에 대한 새로운 평가 메트릭인 KDA(Knowledge Dependent Answerability)를 소개하며, 이 메트릭은 학생이 제출한 답변을 기반으로 MCQ의 대답 가능성을 측정한다. 또한, NLP 태스크에서의 robustness를 향상시키기 위해 '여러 개의' counterfactual을 생성하고 집합적 의사 결정(collective decisions)을 사용하여 인과관계를 파악하는 방법을 제안한다. NER은 노이즈가 있는 텍스트에서 지식 코퍼스에서 관련 텍스트를 검색하여 원본 노이즈 입력의 표현을 향상시킨다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 평가한 클래스룸 환경에서의 MCQ 사용성과 강한 상관관계를 가지고 있으며, 이는 기존 n-gram 기반 메트릭과 결합하여 MCQ 품질 예측에 강력한 예측력을 보여준다. 또한, 검색 중심의 NER 모델은 다양한 노이즈가 있는 NER 환경에서도 뛰어난 성능 향상을 달성하였다.



### Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs (https://arxiv.org/abs/2407.18556)
Comments:
          Accepted to CIKM 2024

- **What's New**: 이 논문에서는 sparsely populated Knowledge Graphs (KGs)에서의 지식 완성을 위한 새로운 접근 방식인 LoGRe(Globally Look and Reason) 모델을 제안합니다. 기존의 path-based 모델들이 외부 모델에 의존하는 대신, 내부적인 관점에서 문제를 해결하는 방법을 설명합니다.

- **Technical Details**: LoGRe 모델은 두 단계의 path reasoning을 통해 sparse KGs의 데이터를 전반적으로 분석하여 관계-경로 reasoning 스키마를 구성합니다. 이 스키마를 기반으로 LoGRe는 경로를 집계하여 답변을 유도하는 방식으로 작동합니다.

- **Performance Highlights**: 다섯 가지 benchmark sparse KG 데이터셋에서 실시한 실험 결과, 제안된 LoGRe 모델은 기존의 접근 방식보다 효과적인 지식 완성 성과를 보여주었습니다.



### How To Segment in 3D Using 2D Models: Automated 3D Segmentation of Prostate Cancer Metastatic Lesions on PET Volumes Using Multi-Angle Maximum Intensity Projections and Diffusion Models (https://arxiv.org/abs/2407.18555)
Comments:
          11 pages, 2 figures, accepted in the DGM4MICCAI workshop, MICCAI, 2024

- **What's New**: 이 논문에서는 다중 선택 질문(MCQ) 자동 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA, Knowledge Dependent Answerability)을 제안하여 학생의 대상 사실 지식 평가 능력을 측정한다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방한다. 이러한 메트릭은 n-gram 기반 유사성 메트릭과 결합하여 다양한 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 보여준다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 KDA 간에 강한 상관관계를 보이며, 자동으로 생성된 MCQ의 교육적 효과성을 더욱 향상시킨다.



### Multi-Agent Trajectory Prediction with Difficulty-Guided Feature Enhancement Network (https://arxiv.org/abs/2407.18551)
- **What's New**: 자동 MCQ 생성, 지식 종속 가능성(KDA) 평가 메트릭, 스푸리어스 패턴을 넘어서는 새로운 방법론 및 자율주행을 위한 새로운 네트워크 제안.

- **Technical Details**: MCQ 평가를 위한 지식 종속 가능성(KDA) 메트릭 제안, KDA_disc 및 KDA_cont의 사용, 역대 최상의 성능을 달성한 DGFNet 아키텍처, spatio-temporal feature encoding 및 difficulty-guided decoder 사용.

- **Performance Highlights**: MCQ 생성의 교육적 가치를 개선, 다양한 전문가가 평가한 MCQ 품질 측정과 높은 상관관계, DGFNet은 Argoverse 1&2 벤치마크에서 state-of-the-art 성능을 달성하고, 정확성과 실시간 추론 속도를 균형있게 유지.



### ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments (https://arxiv.org/abs/2407.18550)
Comments:
          ECCV 2024 (Project page: this https URL)

- **What's New**: 자동화된 MCQ(다중 선택 질문) 생성은 교육자들이 학생 평가에 소요하는 시간을 크게 줄일 수 있는 가능성을 가지고 있지만, 기존의 평가 메트릭은 교육적 가치를 평가하지 못하고 있다. 이를 해결하기 위해 '지식 종속 가능성(KDA)'이라는 새로운 자동 평가 메트릭을 제안한다.

- **Technical Details**: KDA는 대상 사실에 대한 학생의 지식을 기반으로 MCQ의 대답 가능성을 평가하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다. 이들은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방하고 KDA를 근사화한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실 설정에서의 사용성과 KDA 간의 강한 상관관계를 보이며, n-그램 기반의 유사성 메트릭과 결합했을 때 전문가가 라벨링한 여러 MCQ 품질 측정의 예측력이 강한 것으로 나타났다.



### Towards Improving NAM-to-Speech Synthesis Intelligibility using Self-Supervised Speech Models (https://arxiv.org/abs/2407.18541)
Comments:
          Accepted at Interspeech 2024

- **What's New**: 자동 MCQ 생성의 효율성을 높이기 위한 새로운 평가 메트릭 'KDA (Knowledge Dependent Answerability)'를 제안하고, 이 메트릭이 실제 강의실에서도 유용하다는 것을 입증하였다.

- **Technical Details**: KDA는 학생 설문조사를 기반으로 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 개발하여 사전 훈련된 언어 모델을 활용해 학생들의 문제 해결 행동을 모방한다.

- **Performance Highlights**: KDA_disc와 KDA_soft는 전문가에 의해 라벨링된 실제 강의실에서의 사용성과 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 전문가 라벨링 MCQ 품질 지표에 대한 예측력이 높아졌다.



### A Universal Prompting Strategy for Extracting Process Model Information from Natural Language Text using Large Language Models (https://arxiv.org/abs/2407.18540)
- **What's New**: 자동 MCQ 생성의 평가에 대한 새로운 접근법인 Knowledge Dependent Answerability (KDA)를 제안하고, 이를 통해 MCQ의 대답 가능성을 평가하는 방법을 기술하였습니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 평가 메트릭과는 다르게 KDA는 학생의 지식을 직접적으로 평가할 수 있는 능력을 가지고 있습니다. KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하며, 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: 인간 연구를 통해 KDA_disc와 KDA_cont가 KDA 및 실제 강의실 환경에서의 사용성과 강한 상관관계를 가진다는 것을 보여주었으며, n-gram 기반 메트릭과 결합할 경우, 전문가가 라벨링한 MCQ 품질 지표에 대해 강력한 예측력을 보였습니다.



### Outer Approximation and Super-modular Cuts for Constrained Assortment Optimization under Mixed-Logit Mod (https://arxiv.org/abs/2407.18532)
- **What's New**: 이번 논문에서는 혼합 로짓(mixed-logit) 고객 선택 모델 아래에서의 조합 최적화 문제를 연구하고, 기존의 방법들보다 선호도 기반의 더 효율적인 해결책을 제시합니다.

- **Technical Details**: 조합 최적화는 수익 관리(revenue management)의 주요 주제로, 혼합 로짓 모델은 고객의 구매 행동을 모델링하고 예측하는 데 매우 유연한 접근법으로 알려져 있습니다. 기존의 방법들은 혼합 정수 선형 프로그래밍(mixed-integer linear programming, MILP)이나 두 번째 차원 원뿔(cone, CONIC) 개혁을 사용했지만, 약한 연속 완화(continuous relaxation) 문제로 인해 대규모 문제에 대한 해결 속도가 느린 단점이 있었습니다.

- **Performance Highlights**: 본 연구에서는 목표 함수의 완비성을 증명할 수 있는 구성 요소에 집중하여 유효한 절단(valid cuts)을 도출합니다. 이러한 절단은 Cutting Plane 또는 Branch-and-Cut 방법에 통합되어 문제를 정확하게 해결하는 데 사용됩니다. 광범위한 실험 결과, 우리의 접근 방식이 기존 방법에 비해 솔루션 품질과 계산 시간 모두에서 일관되게 우수함을 보여주었습니다.



### Is larger always better? Evaluating and prompting large language models for non-generative medical tasks (https://arxiv.org/abs/2407.18525)
Comments:
          arXiv admin note: text overlap with arXiv:2402.01713

- **What's New**: 최근 대형 언어 모델(LLMs)의 의료 분야 활용이 증가하고 있지만, 구조화된 전자 건강 기록(EHR) 데이터와 비구조화 임상 노트를 처리하는 능력에 대한 연구는 미비하다. 본 연구는 여러 모델을 비교하여 비생성적 의료 작업에 대한 성능을 평가한다.

- **Technical Details**: 구조화된 EHR 데이터와 비구조화된 의료 텍스트를 다루기 위해, 14개의 언어 모델(9개 GPT 기반, 5개 BERT 기반)과 7개의 전통적인 예측 모델을 MIMIC 데이터셋(ICU 환자 기록)과 TJH 데이터셋(조기 COVID-19 EHR 데이터)을 활용하여 평가했다.

- **Performance Highlights**: 결과적으로 LLM은 구조화된 EHR 데이터에 대해서는 robust한 zero-shot 예측 능력을 보였으나, 비구조화된 의료 텍스트에 대해서는 finetuned BERT 모델이 더 우수한 성능을 나타냈다. 따라서 구조화된 데이터에서는 LLM의 zero-shot 학습이 효과적이지만, 비구조화 텍스트에 대해서는 finetuned BERT 모델이 더 적합하다는 점이 강조된다.



### She Works, He Works: A Curious Exploration of Gender Bias in AI-Generated Imagery (https://arxiv.org/abs/2407.18524)
Comments:
          11 pages, 8 figures

- **What's New**: 이 논문은 AI가 생성한 건설 노동자 이미지에서 성 불균형을 조사하며, 남성과 여성 인물의 묘사에서 나타나는 차이를 강조하고 있다.

- **Technical Details**: 그리젤다 폴록(Griselda Pollock)의 시각 문화 및 젠더 이론에 기반하여 AI 모델이 여성 인물을 성적화하고 남성 인물은 더 권위 있고 능숙하게 묘사하는 경향이 있음을 분석하였다.

- **Performance Highlights**: 이 연구 결과는 AI가 사회적 편향을 반영하고 지속시킬 수 있는 잠재력을 강조하며, AI 생성 콘텐츠와 관련된 비판적 접근의 필요성을 부각시킨다.



### Patched MOA: optimizing inference for diverse software development tasks (https://arxiv.org/abs/2407.18521)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 성능을 최적화하는 패치된 혼합 에이전트(Patched MOA) 기술을 소개합니다. 이 기술은 다양한 소프트웨어 개발 작업에서 LLM의 성능을 향상시키는 데 기여합니다.

- **Technical Details**: 세 가지 추론 최적화 알고리즘(Best of N, Mixture of Agents, 그리고 Monte Carlo Tree Search)을 평가했습니다. Patched MOA는 더 작은 모델의 성능을 향상시켜서 더 크고 비싼 모델을 초월할 수 있음을 입증했습니다.

- **Performance Highlights**: 특히, gpt-4o-mini 모델이 Arena-Hard-Auto 벤치마크에서 15.52% 향상된 성능을 내어 gpt-4-turbo보다 우수한 성과를 보였습니다. 우리 방법은 모델에 독립적이며, 최종 사용자에게 투명하게 적용될 수 있습니다.



### TCGPN: Temporal-Correlation Graph Pre-trained Network for Stock Forecasting (https://arxiv.org/abs/2407.18519)
- **What's New**: 자동 MCQ 생성은 교육자의 평가 시간을 줄일 수 있으나, 기존의 평가 메트릭은 교육적 가치를 고려하지 않고 있는 문제를 해결하기 위한 새로운 메트릭 KDA를 제안합니다.

- **Technical Details**: KDA는 학생의 지식에 따른 MCQ의 대답 가능성을 측정하며, KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하여 사전 훈련된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보이며, n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가 평가 MCQ 품질 측정에 대해 강력한 예측력을 보여줍니다.



### SLIM: Style-Linguistics Mismatch Model for Generalized Audio Deepfake Detection (https://arxiv.org/abs/2407.18517)
- **What's New**: 자동 MCQ 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)를 도입하여 학생의 지식 평가 능력을 측정하는 방법을 제안합니다.

- **Technical Details**: 기존의 MCQ 생성 메트릭이 교육적 가치를 고려하지 않고 n-gram 유사성에만 초점을 맞추는 반면, KDA는 학생의 반응을 기반으로 MCQ의 대답 가능성을 측정합니다. KDA_disc 및 KDA_cont는 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실 환경에서의 사용성과 KDA와 강한 상관관계를 보여주며, n-gram 기반 유사성 메트릭과 결합했을 때 다양한 MCQ 품질 측정에 대한 예측력이 강하다는 결과를 도출하였습니다.



### Non-Overlapping Placement of Macro Cells based on Reinforcement Learning in Chip Design (https://arxiv.org/abs/2407.18499)
- **What's New**: 이 논문에서는 자동 MCQ 생성, robustness 향상 및 칩 설계에서의 새로운 방법론이 제안되었습니다.

- **Technical Details**: 자동 MCQ 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 MCQ의 대답 가능성을 측정합니다. 또한, 대조 학습(contrastive learning)과 반사실적 증강(counterfactual augmentation)을 이용한 robust 학습 방안을 연구합니다. 마지막으로, SRLPlacer라는 새로운 강화 학습(reinforcement learning) 기반의 배치 방법론을 통해 칩 설계의 레이아웃 오버랩 문제를 해결합니다.

- **Performance Highlights**: KDA는 실제 강의실 사용성과 강한 상관관계를 가지며, 기존 n-gram 기반 평가 메트릭과 결합 시 다양한 전문가 레이블링 된 MCQ 질 측정에서 높은 예측력을 보입니다. 또한, 대조 학습을 통한 방법은 다양한 측면에서 significant improvements를 달성했으며, SRLPlacer는 공공 벤치마크 ISPD2005에서 마크로 셀의 오버랩 문제를 효과적으로 해결했습니다.



### A Reliable Common-Sense Reasoning Socialbot Built Using LLMs and Goal-Directed ASP (https://arxiv.org/abs/2407.18498)
- **What's New**: 이 논문에서는 AutoCompanion이라는 소셜봇을 제안하며, 이는 대형 언어 모델(LLM)을 사용하여 자연어를 술어로 변환하고 (반대로도) 대화 중 일반 상식을 적용하여 대화를 유도할 수 있도록 설계되었습니다.

- **Technical Details**: AutoCompanion은 s(CASP)라는 목표 지향적 Answer Set Programming (ASP) 구현을 백엔드로 사용하여 인간과의 소셜 대화를 진행합니다. 이 프레임워크는 LLM을 사용하여 사용자 메시지를 분석하고 s(CASP) 엔진의 출력을 바탕으로 응답을 생성하는 과정을 설명합니다.

- **Performance Highlights**: 논문에서는 챗봇이 영화와 책에 대해 이야기하며 사용자를 재미있게 유지하는 목표를 가지고 진행한 실제 대화를 기술함으로써 (i) 답변의 정확성, (ii) 대화의 일관성 및 정밀성, (iii) 주요 주제에서의 이탈 없음 등을 보장하는 방법을 제시합니다.



### A Role-specific Guided Large Language Model for Ophthalmic Consultation Based on Stylistic Differentiation (https://arxiv.org/abs/2407.18483)
- **What's New**: 전통적인 평가 메트릭들이 교육적 가치를 간과하는 문제를 해결하기 위해, '지식 종속 가능성(KDA, Knowledge Dependent Answerability)'라는 새로운 자동 평가 메트릭을 제안했습니다.

- **Technical Details**: KDA는 교실 환경에서의 사용성과 KDA 간의 강한 상관관계를 보여주는 인적 평가를 기반으로 하며, 기존 n-gram 기반 메트릭과 결합했을 때 MCQ 품질 예측력을 강화하는 두 개의 자동 평가 메트릭 KDA_disc와 KDA_cont를 도입합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가 레이블이 붙은 MCQ 품질 측정치에서 강력한 예측력을 보였으며, 지식 기반 평가에서 더 나은 결과를 제공합니다.



### Diffusion-Driven Semantic Communication for Generative Models with Bandwidth Constraints (https://arxiv.org/abs/2407.18468)
Comments:
          13 pages, 7 figures, submitted to IEEE for possible publication

- **What's New**: 최근 몇 년간 AI 생성 콘텐츠(AIGC)에 확산 모델(diffusion models)이 효과적으로 활용되고 있으며, 이 논문에서는 대역폭 제약이 있는 무선 통신을 위한 새로운 확산 기반의 의미 통신(semantic communication) 프레임워크를 소개합니다.

- **Technical Details**: 이 프레임워크는 VAE(변분 오토 인코더)-기반 압축을 사용하여, 무선 채널을 통한 신호 전송 과정을 확산의 전향 과정으로 간주합니다. 또한, 수신자에서 가우시안 분포에 맞는 복원된 피쳐를 보장하기 위해 다운샘플링 모듈과 쌍으로 된 업샘플링 모듈을 포함합니다.

- **Performance Highlights**: 우리는 제안한 시스템의 손실 함수(loss function)를 도출하고, 포괄적인 실험을 통해 성능을 평가하였습니다. 실험 결과는 PSNR(신호 대 잡음비)와 LPIPS(학습된 지각 이미지 패치 유사도)와 같은 픽셀 수준 메트릭에서 유의미한 개선을 보여 주었으며, DJSCC(심층 결합 소스-채널 코딩) 대비 압축 비율과 SNR에서 더욱 두드러진 개선을 나타냈습니다.



### Fairness Definitions in Language Models Explained (https://arxiv.org/abs/2407.18454)
- **What's New**: 자동 생성된 객관식 질문(MCQ)의 평가 방식에 대한 새로운 접근법과, 언어 모델의 공정성 정의에 대한 체계적인 설문 조사가 제안되었습니다.

- **Technical Details**: MCQ의 대답 가능성(answerability)을 측정하는 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)이 제안되었으며, KDA_disc 및 KDA_cont와 같은 두 가지 자동 평가 메트릭이 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다. 또한 ML 모델이 사회적 편견을 어떻게 확대할 수 있는지를 설명하고, 이를 해결하기 위한 여러 공정성 개념을 분류하는 새로운 분류법이 제시되었습니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 강의실 환경에서의 실용성과 KDA와 강한 상관관계를 가지며, n-그램 기반 유사성 메트릭과 결합 시 전문가가 제시한 다양한 MCQ 품질 측정에 대한 예측력이 뛰어남을 입증했습니다. 또한, 제안된 방법은 counterfactual robustness, cross-domain generalization, 그리고 scarce data에서의 일반화를 향상시킨 결과를 보여주었습니다.



### Capturing the security expert knowledge in feature selection for web application attack detection (https://arxiv.org/abs/2407.18445)
- **What's New**: 이번 논문은 웹 어플리케이션 방화벽(WAF)의 효과성을 높이기 위해 보안 전문가의 전문성을 복제할 수 있는 상호 정보 값(mutual information values)을 사용하는 방법을 제안합니다.

- **Technical Details**: WAF는 HTTP 트래픽을 분석하여 알려진 공격 패턴을 식별하고 잠재적인 악성 요청을 차단하는 데 사용됩니다. 그러나, 허위 양성(false positive)의 발생은 정상적인 트래픽을 차단할 수 있어 큰 문제입니다. 본 연구는 특성 선택(feature selection)을 위한 감독 학습(supervised learning)과 One-Class SVM 모델 교육을 위한 반감독 학습(semi-supervised learning)의 결합 방법을 사용합니다.

- **Performance Highlights**: 제안된 알고리즘으로 선택된 특성을 사용한 모델은 전문가 기반 선택 접근법에 비해 성능에서 우수한 결과를 보여주었으며, 전통적인 규칙 기반 WAF인 ModSecurity에서도 개선된 성과를 보였습니다.



### Mixed Non-linear Quantization for Vision Transformers (https://arxiv.org/abs/2407.18437)
Comments:
          16 pages, 4 figures, under review

- **What's New**: 최근의 자동 MCQ 생성 기술은 교사들의 평가 시간을 절감할 수 있는 잠재력을 가지고 있지만, 기존의 평가 메트릭들은 교육적 가치를 고려하지 않는다. 이 논문에서는 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하여 MCQ의 대답 가능성을 평가하는 방법을 소개한다.

- **Technical Details**: 논문에서는 KDA를 측정하기 위해 데이터에서 학생들의 응답을 기반으로 한 human evaluation을 사용하고, 사전 훈련된 언어 모델(pre-trained language models)을 이용하여 학생의 문제 해결 행동을 모방하는 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안한다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 다양한 전문가 라벨링된 MCQ 품질 측정 기준에 대해 강력한 예측력을 보여주며, 실제 강의실 환경에서의 사용성과의 상관관계도 높다.



### Investigating the Privacy Risk of Using Robot Vacuum Cleaners in Smart Environments (https://arxiv.org/abs/2407.18433)
Comments:
          18 pages, 11 figures, 4 tables, The 26th International Conference on Information and Communications Security, 26-28 August, 2024, Mytilene, Lesvos, Greece (ICICS2024)

- **What's New**: 자동 다중 선택 질문(MCQ) 생성을 위한 새로운 평가 메트릭, 지식 종속 가능성(KDA) 제안 및 사용자 응답 기반의 KDA 측정 방법을 제시함.

- **Technical Details**: KDA는 학생의 지식을 평가하는 능력을 측정하며, 두 가지 자동 평가 메트릭(KDA_disc, KDA_cont)이 사전 훈련된 언어 모델을 통해 학생의 문제 해결 행동을 모방하여 KDA를 근사화.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 기존의 n-그램 기반 유사도 메트릭과 결합 시 다양한 전문가 라벨 MCQ 품질 측정에 대해 강한 예측력을 보여줌.



### Weighted Risk Invariance: Domain Generalization under Invariant Feature Shif (https://arxiv.org/abs/2407.18428)
- **What's New**: 이 논문에서는 자동 MCQ 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다. 기존의 평가 메트릭들이 교육적 가치를 간과하던 문제를 해결합니다.

- **Technical Details**: KDA는 대상 사실에 대한 학생의 지식을 바탕으로 MCQ의 대답 가능성을 측정합니다. 또한 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안하여, 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 레이블링한 실제 강의실 세트에서의 사용성과 KDA와 강한 상관관계를 보였으며, n-gram 기반 유사성 메트릭과 결합할 경우 다양한 전문가 레이블링 MCQ 품질 기준의 예측력이 뛰어남을 입증했습니다.



### HDL-GPT: High-Quality HDL is All You Need (https://arxiv.org/abs/2407.18423)
Comments:
          DAC 2024 Invited Paper

- **What's New**: 이 논문은 고품질의 HDL(High Definition Language) 코드를 활용하여 우수한 성능을 가진 코드 모델을 개발하는 HDL-GPT라는 새로운 접근법을 제안합니다.

- **Technical Details**: HDL-GPT는 오픈 소스 HDL 코드에서 대규모 데이터를 수집하고 증대시키는 방법을 사용하여, 변동성이 큰 데이터를 고품질 데이터로 변환합니다. 여기에는 신중한 프롬프트(prompts)와 컨텍스트 유지(context maintenance)가 포함됩니다.

- **Performance Highlights**: HDL-GPT는 SOTA(State-of-the-Art) HDL 모델에 비해 50%에서 200%의 성능 향상을 보여주며, HDL 회로 설명, 코드 생성, 포멀 및 시뮬레이션 테스트벤치 생성, 버그 분류 및 수정 등 다양한 작업에서 뛰어난 결과를 나타냅니다.



### PersonaGym: Evaluating Persona Agents and LLMs (https://arxiv.org/abs/2407.18416)
Comments:
          21 pages, 5 figures

- **What's New**: 이번 논문은 교육, 헬스케어 및 엔터테인먼트와 같은 다양한 분야에서 사용할 수 있는 'Persona agents'를 평가하기 위한 새로운 평가 프레임워크인 PersonaGym과 자동화된 메트릭인 PersonaScore를 소개합니다.

- **Technical Details**: Persona agents는 주어진 페르소나에 따라 행동하는 LLM agents로서, 해당 페르소나에 맞는 응답을 정렬할 수 있는 가능성을 제공합니다. 그러나 다양한 환경에서 페르소나 준수를 평가하는 것은 매우 복잡합니다. PersonaScore는 이러한 평가를 위한 결정 이론을 기반으로 한 자동화된 메트릭입니다.

- **Performance Highlights**: 우리는 6개의 오픈 및 클로즈드 소스 LLM을 평가했으며, 200개의 페르소나와 10,000개의 질문을 포함한 벤치마크를 사용했습니다. 그 결과, 모델 크기와 복잡성이 증가한다고 해서 성능이 반드시 향상되지 않음을 발견했습니다.



### Adversarial Robust Decision Transformer: Enhancing Robustness of RvS via Minimax Returns-to-go (https://arxiv.org/abs/2407.18414)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 교육적 가치를 고려한 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하여 MCQ의 대답 가능성을 측정합니다. KDA는 학생의 지식 수준에 따라 평가할 수 있도록 설계되었습니다.

- **Technical Details**: KDA는 학생 응답을 기반으로 측정되며, 두 가지 자동 평가 메트릭인 KDA_disc 및 KDA_cont를 제안하여 사전 학습된 언어 모델을 활용해 학생의 문제 해결 행동을 모방합니다. 이 방법은 실제 강의실 환경에서 전문가에 의해 라벨링된 MCQ 품질 측정과 강한 상관관계를 보여줍니다.

- **Performance Highlights**: KDA_disc 및 KDA_cont는 n-gram 기반 유사성 메트릭과 결합 시 각기 다른 전문가 라벨링 기준에 대한 예측력을 보여줍니다. 이러한 단일 및 집합적 결정 방식을 통해, 기존 방법들보다 더 강한 robust 성능을 보입니다.



### Simulation of Neural Responses to Classical Music Using Organoid Intelligence Methods (https://arxiv.org/abs/2407.18413)
Comments:
          10 pages, 9 figures

- **What's New**: 이번 연구에서는 MCQ(다중 선택 질문)의 자동 생성에서 평가 기준을 개선하기 위한 새로운 자동 평가 메트릭인 KDA(knowledge Dependent Answerability)를 제안하였습니다. 이는 기존의 n-gram 기반 메트릭이 아닌, MCQ가 특정 사실에 대한 학생의 지식을 어떻게 평가하는지를 탐구합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 개발하였습니다. 이 메트릭들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA와 실제 교실 설정에서의 사용성과 강한 상관관계를 보여주며, n-gram 기반 메트릭과 결합될 경우 여러 전문가가 레이블링한 MCQ 품질 측정에서 강력한 예측력을 발휘합니다.



### SCALE: Self-regulated Clustered federAted LEarning in a Homogeneous Environmen (https://arxiv.org/abs/2407.18387)
Comments:
          This research article got accepted in COMPSAC conference and going to be published to IEEE

- **What's New**: 이 논문에서는 Federated Learning (FL)의 새로운 방법론을 제안하여, 기존의 중앙 집중형 인프라에 대한 의존성을 없애고 통신 효율성을 향상시킵니다.

- **Technical Details**: 제안된 방법은 서버 보조 근접 평가(server-assisted Proximity Evaluation)를 통해 데이터 유사성, 성능 지표 및 지리적 근접성에 기반한 동적 클러스터 형성을 포함합니다. 또한, Hybrid Decentralized Aggregation Protocol을 통해 지역 모델 학습과 동료 간 가중치 교환, 동적으로 선택된 드라이버 노드에 의한 중앙집중식 최종 집계를 결합하여 전체 통신 오버헤드를 크게 줄입니다.

- **Performance Highlights**: 유방암 데이터셋으로 검증한 결과, 통신 오버헤드가 거의 10배 감소하며, 훈련 지연(latency) 및 에너지 소비를 현저하게 줄이면서도 높은 학습 성능을 유지하는 것을 보여주었습니다.



### Robust Claim Verification Through Fact Detection (https://arxiv.org/abs/2407.18367)
- **What's New**: 자동 MCQ 생성, 컨트라스티브 학습, 그리고 주장 검증을 위한 새로운 방법론을 제안하며 이들 방법들이 교육적 가치 및 데이터 로버스트성을 향상시킨다는 점에서 기존 연구와 차별화된다.

- **Technical Details**: 1. 자동 MCQ 생성에서 지식 종속 가능성(KDA)이라는 평가 메트릭을 제안. 2. 컨트라스티브 학습을 통해 여러 개의 counterfactual을 생성하고 집합적 의사 결정을 통해 각 용어의 인과관계를 이해. 3. 새로운 FactDetect 접근 방식을 활용하여 짧은 사실을 증거에서 추출하고, 이를 LLMs와 결합하여 주장 검증에서의 설명 가능성과 성능을 개선.

- **Performance Highlights**: 1. KDA_disc와 KDA_cont 메트릭이 실제 강의실 환경에서의 사용성과 강한 상관관계를 보임. 2. 논문의 접근 방식이 zero-shot 주장 검증에서 15% F1 점수 향상을 보이며, AugFactDetect가 가장 잘 수행되는 기준과 비교하여 평균 17.3% 성능 향상을 기록.



### FADAS: Towards Federated Adaptive Asynchronous Optimization (https://arxiv.org/abs/2407.18365)
Comments:
          Accepted by ICML 2024

- **What's New**: 이 논문에서는 프라이버시 보호 기계 학습을 위한 Federated Learning(FL)에서 비동기 업데이트를 통합한 Federated Adaptive Asynchronous Optimization(FADAS)라는 새로운 방법을 제안한다.

- **Technical Details**: FADAS는 적응형 연합 최적화 방법에 비동기 업데이트를 추가하여, 지연 클라이언트가 존재하는 상황에서도 실용적으로 배포할 수 있다. 또한, 대기 적응형 학습 조정 전략을 통해 비동기 지연이 큰 상황에서도 효율성과 탄력성을 향상시킨다.

- **Performance Highlights**: FADAS의 수렴 속도를 엄밀하게 확립하고, 경험적 결과에서 FADAS가 다른 비동기 FL 기본선보다 우수한 성능을 보임을 보여주었다.



### KI-Bilder und die Widerst\"andigkeit der Medienkonvergenz: Von prim\"arer zu sekund\"arer Intermedialit\"at? (https://arxiv.org/abs/2407.18363)
Comments:
          in German language

- **What's New**: 이번 연구에서는 자동 Multiple Choice Questions (MCQ) 생성 시 기존 평가 메트릭이 교육적 가치를 고려하지 못하는 문제를 다루며, 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다.

- **Technical Details**: KDA는 학생들의 응답을 기반으로 MCQ의 대답 가능성을 평가하며, pretrained language models를 활용한 두 가지 자동 평가 메트릭(KDA_disc와 KDA_cont)을 제안합니다. 이들은 실험적 연구를 통해 KDA와 실제 강의실 환경에서의 사용성과 강한 상관관계를 보였습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 n-gram 기반 유사성 메트릭과 결합하여 다양한 전문가 라벨이 있는 MCQ 품질 측정에서 강한 예측력을 보여주었습니다.



### Generative AI like ChatGPT in Blockchain Federated Learning: use cases, opportunities and futur (https://arxiv.org/abs/2407.18358)
Comments:
          We are going to submit this research article into a conference which is best fit for this topic

- **What's New**: 자동 생성된 객관식 질문(MCQ) 평가에 새로운 메트릭, 지식 종속 대답 가능성(Knowledge Dependent Answerability, KDA)이 제안되었습니다.

- **Technical Details**: KDA는 학생의 지식을 기반으로 MCQ의 대답 가능성을 평가하며, 학생 응답을 통해 KDA를 측정하는 방법이 설명됩니다. 또한, KDA_disc와 KDA_cont라는 새로운 자동 평가 메트릭이 기존의 n-gram 기반 유사성 메트릭과 결합되어 MCQ 품질을 예측하는 데에 높은 예측력을 가진다고 합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 기존 평가 메트릭과 비교하여 실제 강의실 세팅에서의 사용성과 강한 상관관계를 보였으며, 전문가가 라벨링한 다양한 MCQ 품질 측정에서 뛰어난 성능을 보였습니다.



### Introducing {\delta}-XAI: a novel sensitivity-based method for local AI explanations (https://arxiv.org/abs/2407.18343)
- **What's New**: 자동 MCQ 생성, robust한 NLP 모델 설계 및 설명 가능한 인공지능(XAI) 방법에 대한 세 가지 새로운 접근을 제안합니다.

- **Technical Details**: MCQ 생성의 KDA(지식 종속 가능한 대답 가능성) 메트릭, 대조 학습과 반사실적 증강(counterfactual augmentation) 기법, 그리고 delta-XAI 방법을 통해 ML 모델의 설명력을 높이려는 내용이 포함되어 있습니다.

- **Performance Highlights**: KDA 기법이 기존 평가 메트릭과 함께 사용될 때 강력한 예측 능력을 보여주며, delta-XAI 방법은 ML 모델의 예측에 대한 외부 설명을 제공하여 임상 실무에서의 적용 가능성을 높입니다.



### A Comprehensive Analysis of Machine Learning Models for Algorithmic Trading of Bitcoin (https://arxiv.org/abs/2407.18334)
- **What's New**: 이번 연구는 41개의 머신 러닝 모델을 Bitcoin 가격 예측을 위해 평가하는 내용을 다루고 있다. 특히 21개의 분류기(classifiers)와 20개의 회귀 모델(regressors)을 사용하여 알고리즘 거래에서의 성능을 분석하였다.

- **Technical Details**: 다양한 시장 조건에서 이러한 모델의 정확성(accuracy), 강건성(robustness), 변동성 있는 암호화폐 시장에 대한 적응성(adaptability)을 강조하였다. 모델 성능 평가를 위해 머신 러닝 메트릭(Mean Absolute Error, Root Mean Squared Error)과 거래 메트릭(Profit and Loss percentage, Sharpe Ratio)을 적용하였다.

- **Performance Highlights**: 특정 모델들, 예를 들어 Random Forest와 Stochastic Gradient Descent가 수익과 위험 관리 측면에서 다른 모델들보다 뛰어난 성과를 보였으며, 이러한 통찰력은 암호화폐 거래를 위한 실제 적용 가능성을 보장한다.



### AutoVCoder: A Systematic Framework for Automated Verilog Code Generation using LLMs (https://arxiv.org/abs/2407.18333)
- **What's New**: 자동 MCQ 생성, 지식 종속 가능성(KDA) 메트릭을 제안하며, 최근 NLP 태스크의 robustness 저하 문제와 회피해야 할 spurious patterns의 한계를 극복하고자 하는 접근법을 설명하고 있습니다.

- **Technical Details**: MCQ 생성의 경우 KDA는 대상 사실에 대한 학생의 지식을 평가하는 새로운 자동 평가 메트릭입니다. NLP에서는 contrastive learning과 counterfactual augmentation 기술을 활용하여 robustness를 개선하는 방안을 제시하고 있습니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 강의실 환경에서 전문가에 의해 확인된 사용성과 강한 상관관계를 보이며, 다양한 MCQ 품질 척도에 대해 예측력이 뛰어난 결과를 도출합니다. AutoVCoder는 Verilog 코드 생성에서 BetterV 및 RTLCoder 대비 기능적 정확도에서 0.5%와 2.2%, 문법적 정확도 및 기능적 정확도에서 각각 3.4% 개선된 성능을 나타냅니다.



### The Need for Guardrails with Large Language Models in Medical Safety-Critical Settings: An Artificial Intelligence Application in the Pharmacovigilance Ecosystem (https://arxiv.org/abs/2407.18322)
Comments:
          27 pages, 6 figures, 4 tables and supplementary material provided

- **What's New**: 이번 연구에서는 자동 MCQ (Multiple Choice Questions) 생성의 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안하여 기존의 n-gram 기반 메트릭의 한계를 극복하고자 합니다.

- **Technical Details**: KDA는 학생의 목표 사실에 대한 지식을 기반으로 MCQ의 대답 가능성을 측정하며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이 메트릭들은 사전 훈련된 언어 모델을 활용하여 학생들의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 (1) KDA와 (2) 전문가가 점검한 실제 수업 환경에서의 사용성과 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 때 다양한 MCQ 품질 척도에 대한 예측력이 뛰어난 성능을 보입니다.



### Revolutionizing Undergraduate Learning: CourseGPT and Its Generative AI Advancements (https://arxiv.org/abs/2407.18310)
Comments:
          8 pages

- **What's New**: 이 논문에서는 CourseGPT라는 Generative AI 도구를 소개하고 있으며, 이는 교육적 맥락에서 학습 경험을 향상시키기 위해 설계되었습니다. 이 도구는 개방형 대형 언어 모델(LLMs)을 기반으로 하여 교육자에게 지속적인 지원과 정기적인 강의 자료 업데이트를 제공합니다.

- **Technical Details**: CourseGPT는 수업 자료, 슬라이드, 보충 읽기 및 참고자료와 같은 수업 특정 콘텐츠를 활용하여 학생의 질문에 대해 정확하고 동적으로 생성된 응답을 제공합니다. 기존의 일반 AI 모델과 달리 CourseGPT는 교육자가 응답을 관리하고 통제할 수 있게 해줍니다.

- **Performance Highlights**: CourseGPT는 CPR E 431(정보 시스템 보안의 기초) 수업에서 파일럿 테스트를 수행하며, Mixtral-8x7b 모델이 88.0%의 정확도와 66.6%의 신뢰도를 달성해 더 작은 모델들을 능가했습니다. 학생들과 교육 조교들이 CourseGPT의 정확성과 유용성을 극찬하는 피드백도 수집되었습니다.



### Rome was Not Built in a Single Step: Hierarchical Prompting for LLM-based Chip Design (https://arxiv.org/abs/2407.18276)
Comments:
          Accepted at MLCAD '24. 10 pages, 7 figures, 5 tables

- **What's New**: 자동 생성된 선택형 질문(MCQ)을 위한 새로운 평가 메트릭, 지식 종속 가능성(KDA) 제안. KDA는 MCQ의 대답 가능성을 측정하며, 교육적 가치를 평가하는 데 초점을 맞추고 있다.

- **Technical Details**: 기존의 평가 메트릭(BLEU, ROUGE, METEOR)은 MCQ가 데이터셋의 골드 샘플과 얼마나 유사한지 확인하지만, 교육적 가치나 학생의 지식에 대한 평가를 고려하지 않음. KDA는 학생의 반응을 기반으로 측정될 수 있으며, KDA_disc와 KDA_cont 두 가지 자동 평가 메트릭은 미리 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방함.

- **Performance Highlights**: KDA_disc와 KDA_cont는 실제 강의실에서의 사용성과 강한 상관관계를 보여주었으며, 기존의 n-그램 기반 유사성 메트릭과 결합했을 때 MCQ 품질 측정 지표에 대한 예측력이 강한 것으로 나타났다.



### Adaptive Differentially Private Structural Entropy Minimization for Unsupervised Social Event Detection (https://arxiv.org/abs/2407.18274)
Comments:
          Accepted to ACM CIKM 2024

- **What's New**: 다수의 MCQ 생성에 대한 기존 평가 메트릭의 한계를 극복하기 위해, KDA라는 새로운 자동 평가 메트릭을 제안합니다. 이 메트릭은 MCQ의 대답 가능성을 평가합니다.

- **Technical Details**: KDA는 학생의 반응을 바탕으로 측정되며, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 통해 사전 훈련된 언어 모델을 이용해 학생의 문제 해결 행동을 모방합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 강한 상관관계를 가지며, n-gram 기반 유사성 메트릭과 결합할 때 전문가가 평가한 다양한 MCQ 품질 지표를 강력히 예측합니다.



### AICircuit: A Multi-Level Dataset and Benchmark for AI-Driven Analog Integrated Circuit Design (https://arxiv.org/abs/2407.18272)
- **What's New**: 이 논문은 기존의 MCQ(다중 선택 질문) 생성 및 평가 메트릭의 한계를 극복하고, 새로운 자동 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안합니다.

- **Technical Details**: 기존의 BLEU, ROUGE, METEOR와 같은 메트릭은 MCQ의 교육적 가치를 평가하지 않지만, KDA는 목표 사실(target fact)에 대한 지식을 바탕으로 MCQ의 대답 가능성(answerability)을 평가합니다. KDA를 측정하기 위해 두 가지 자동 평가 메트릭, KDA_disc와 KDA_cont를 제안하며, 이들은 사전 훈련된 언어 모델을 활용하여 학생의 문제 해결 행동을 모방하는 방식으로 KDA를 근사합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 KDA 및 실제 강의실 환경에서의 사용성(usability)과 강한 상관관계를 보여주며, n-그램 기반 유사성 메트릭과 결합했을 때 다양한 전문가 라벨링 MCQ 품질 측정치에 대해 강한 예측력을 발휘합니다.



### Large Language Model for Verilog Generation with Golden Code Feedback (https://arxiv.org/abs/2407.18271)
- **What's New**: 이번 연구는 자연어 명령어로부터 RTL(레지스터 전송 수준) 코드, 특히 Verilog,의 자동 생성을 위한 새로운 접근 방식을 제안하며, 특히 golden code feedback을 사용한 강화 학습(reinforcement learning) 기법을 도입한 것이 주목할 만하다.

- **Technical Details**: 연구팀은 오픈 소스 데이터와 기초 모델을 활용하여 기존 상용 LLM에 비해 상당한 차이로 최신 기술(state-of-the-art, SOTA) 성과를 달성하였다. 특히 그들의 6.7B 파라미터 모델은 13B 및 16B 모델에 비해 뛰어난 성능을 보였다.

- **Performance Highlights**: 강화 학습의 훈련 동역학(training dynamics) 및 직접적인 미세 조정의 한계를 종합적으로 분석한 결과, Verilog 코드의 고유한 병렬 의미(parallel semantics)에 맞춘 종합적인 감독 신호의 개발이 효과적인 코드 생성을 위해 중요하다고 주장하고 있다.



### LaMAGIC: Language-Model-based Topology Generation for Analog Integrated Circuits (https://arxiv.org/abs/2407.18269)
- **What's New**: 이 논문에서는 MCQ 생성을 위한 새로운 평가 메트릭인 지식 종속 가능성(Knowledge Dependent Answerability, KDA)을 제안하며, 이는 학생의 지식에 기반한 MCQ의 답변 가능성을 평가한다.

- **Technical Details**: 기존의 메트릭(BLEU, ROUGE, METEOR)은 MCQ의 교육적 가치를 고려하지 않았고, 우리는 KDA를 통해 학생 응답을 기반으로 한 평가를 수행한다. 또한 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안하고, 이는 학생 문제 해결 행동을 모방하는 사전 훈련된 언어 모델을 이용하여 KDA를 근사화한다.

- **Performance Highlights**: 인간 연구 결과, KDA_disc와 KDA_soft는 KDA 및 실제 강의실 설정에서의 사용성과 강한 상관관계를 지닌 것으로 나타났으며, n-그램 기반 유사성 메트릭과 결합했을 때 여러 전문가가 라벨링한 MCQ 품질 측정에 대해 높은 예측력이 있음을 보여주었다.



### MCU-MixQ: A HW/SW Co-optimized Mixed-precision Neural Network Design Framework for MCUs (https://arxiv.org/abs/2407.18267)
- **What's New**: 자동 MCQ 생성에서 교육적 가치를 고려한 새로운 평가 메트릭인 지식 종속 가능성(KDA)을 제안합니다. KDA는 MCQ의 대답 가능성을 측정하고, n-gram 기반 유사성 지표와 함께 사용할 때 다양한 MCQ 품질 측정에서 강한 예측력을 보여줍니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 평가되며, 두 가지 자동 평가 메트릭인 KDA_disc와 KDA_cont를 제안합니다. 이들 메트릭은 사전 훈련된 언어 모델을 이용하여 학생의 문제 해결 행동을 모방합니다. 또한, 최근 deep model의 robust 문제를 해결하기 위해 대조 학습(contrastive learning)과 counterfactual augmentation을 활용한 방법을 소개하며, 이를 통해 인과관계를 더 잘 파악할 수 있습니다.

- **Performance Highlights**: 실험 결과 KDA_disc와 KDA_cont는 실제 강의실 환경에서의 사용성과 KDA 사이에 강한 상관관계를 가지며, 다양한 전문가 레이블의 MCQ 품질 측정에서 예측력이 있음을 보였습니다. 또한, MCU-MixQ 프레임워크는 기존 기술들에 비해 MCUNet와 CMix-NN에 대해 각각 2.1배와 1.4배 속도 향상을 이뤘습니다.



### Latency optimized Deep Neural Networks (DNNs): An Artificial Intelligence approach at the Edge using Multiprocessor System on Chip (MPSoC) (https://arxiv.org/abs/2407.18264)
Comments:
          25. ITG Fachtagung Mobilkommunikation

- **What's New**: 새로운 자동 MCQ 생성 메트릭인 지식 종속 가능성(KDA)을 제안하여 MCQ의 교육적 가치를 평가하는 접근 방식을 포함하고 있습니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정되며, KDA_disc와 KDA_cont라는 두 개의 자동 평가 메트릭을 통해 학생의 문제 해결 행동을 모방합니다. 이 방법은 n-gram 기반 유사성 메트릭과 결합하여 고품질 MCQ 평가를 지원합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가의 평가와 높은 상관성을 보이며, 다양한 MCQ 품질 측정 기준의 예측력을 강화합니다. 또한, deep model의 robustness를 개선하고, cross-domain generalization의 성능을 향상시킵니다.



### CORN: Contact-based Object Representation for Nonprehensile Manipulation of General Unseen Objects (https://arxiv.org/abs/2403.10760)
Comments:
          ICLR 2024

- **What's New**: 자동 선택형 질문(MCQ) 생성의 새로운 접근 방식으로 지식 종속 가능성(Knowledge Dependent Answerability, KDA) 메트릭을 제안하였습니다. 이 메트릭은 질문의 대답 가능성을 측정하고 학생의 지식 평가 능력을 강조합니다.

- **Technical Details**: KDA는 학생의 응답을 기반으로 측정되며, 이후 사전 훈련된 언어 모델을 사용하여 KDA_disc와 KDA_cont라는 두 가지 자동 평가 메트릭을 제안합니다. 이들은 학생의 문제 해결 행동을 모방하여 KDA를 근사합니다.

- **Performance Highlights**: KDA_disc와 KDA_cont는 전문가가 라벨링한 실제 강의실에서의 사용성과 KDA와 강한 상관관계를 보이며, n-gram 기반 유사성 메트릭과 결합하면 MCQ 품질 측정에서 강력한 예측력을 보여줍니다.




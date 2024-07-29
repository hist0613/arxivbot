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




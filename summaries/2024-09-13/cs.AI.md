New uploads on arXiv(cs.CL)

### Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources (https://arxiv.org/abs/2409.08239)
- **What's New**: 이 논문에서는 Source2Synth라는 새로운 방법을 제안합니다. 이 방법은 비싼 인간 주석에 의존하지 않고 대규모 언어 모델(LLMs)에게 새로운 기술을 가르치기 위한 것입니다.

- **Technical Details**: Source2Synth는 세 가지 단계로 구성됩니다: Dataset Generation, Dataset Curation, 그리고 Model Fine-tuning입니다. 입력으로 사용자 정의 데이터 소스를 받아들여, 실제 세계의 출처에 기반한 중간 추론 단계를 포함한 합성 데이터 포인트를 생성합니다. 이 방법은 데이터 품질을 개선하고 불량 생성물을 제거합니다.

- **Performance Highlights**: Source2Synth는 WikiSQL에서 TQA(표 기반 질문 응답) 성능을 25.51% 향상시키고, HotPotQA에서 MHQA(다단계 질문 응답) 성능을 22.57% 향상시켰습니다.



### AudioBERT: Audio Knowledge Augmented Language Mod (https://arxiv.org/abs/2409.08199)
Comments:
          Preprint

- **What's New**: 최근 연구에서 텍스트 전용 데이터셋에 사전 훈련된 언어 모델이 기본적인 시각적 지식, 예를 들어 일상 객체의 색상이 부족하다는 것을 확인했습니다. 이를 통해 우리는 언어 모델이 청각 지식에 대해서도 동일한 결점이 있는지를 알아보았고, 이에 대한 평가를 위해 새로운 데이터셋 AuditoryBench를 구축했습니다.

- **Technical Details**: AuditoryBench는 두 가지 새로운 과제를 포함하는 데이터셋으로, 언어 모델의 청각 지식을 평가합니다. 첫 번째 과제는 동물 소리 인식이며, 두 번째 과제는 소리의 피치 비교입니다. 또한, 우리는 AudioBERT라는 새로운 방법을 제안하여 BERT의 청각 지식을 증강합니다. 이는 텍스트에서 청각 지식이 필요한 부분을 탐지하고, 관련 오디오를 효율적으로 검색하는 절차로 진행됩니다.

- **Performance Highlights**: AudioBERT는 AuditoryBench에서 40% 이상의 성능 향상을 달성하며, 언어 모델에 청각 상식 지식을 주입하는 첫 번째 알고리즘이라는 점에서 의미가 있습니다.



### Fine-tuning Large Language Models for Entity Matching (https://arxiv.org/abs/2409.08185)
Comments:
          8 pages, 4 figures. For related code and data, see this this https URL

- **What's New**: 이 연구는 엔티티 매칭(entity matching)을 위한 대규모 생성 언어 모델(LLM) 미세 조정(fine-tuning)의 가능성을 탐구합니다. 이를 통해 기존의 프리트레인(pre-trained) 언어 모델보다 높은 제로샷(zero-shot) 성능을 발휘하는 LLM의 잠재력을 분석합니다.

- **Technical Details**: 미세 조정은 두 가지 방향으로 진행됩니다: 1) 훈련 샘플의 표현에서 LLM이 생성한 다양한 설명을 추가하여 훈련 세트를 실험합니다. 2) LLM을 사용하여 훈련 샘플을 선택 및 생성합니다. 또한, 작은 모델에서 미세 조정이 성능을 크게 향상시키지만, 큰 모델에서는 결과가 혼합되는 경향이 관찰되었습니다.

- **Performance Highlights**: 각종 실험을 통해 미세 조정이 세 가지 모델에서 긍정적인 영향을 미치며, 구조화된 설명을 추가하면 성능이 향상되는 것으로 나타났습니다. 반면, GPT-4o Mini 모델의 경우 예시 선택 및 생성 방법이 성능을 낮추는 결과를 보였습니다.



### On the Role of Context in Reading Time Prediction (https://arxiv.org/abs/2409.08160)
- **What's New**: 이번 연구에서는 독자가 실시간 언어 이해 중 맥락을 통합하는 방식을 새롭게 접근합니다. 연구진은 surprisal 이론에 기반하여, 맥락 정보로부터의 예측력을 다른 방법으로도 생성할 수 있음을 제안합니다. 기존의 방법과는 달리, frequency와 독립적인 새로운 예측기법을 도입했습니다.

- **Technical Details**: 연구는 surprisal (경쟁율) 이론에 대해 비판적으로 접근하며, pointwise mutual information (PMI)을 대안으로 제안합니다. 이 두 가지 방법론은 Frequency와 직선 결과적으로 선형 모델의 예측력을 동일하게 합니다. 연구진은 Frequency의 직교 보충에 Surprisal을 투영하는 기법을 도입하여, 기존의 예측기법이 독립적이지 않음을 보였습니다.

- **Performance Highlights**: 실험 결과, 맥락이 orthogonalized 예측기를 통해 표현될 때 읽기 시간에 대한 설명 분산이 훨씬 작아지는 것을 발견했습니다. 이는 이전 연구들이 맥락의 역할을 과대 평가했음을 시사합니다.



### LLM-POTUS Score: A Framework of Analyzing Presidential Debates with Large Language Models (https://arxiv.org/abs/2409.08147)
- **What's New**: 이 논문은 대통령 토론 성과 평가를 위한 새로운 접근 방식을 제시합니다. 이는 정치적 논의 분석에 LLMs(large language models)를 적용하는 방법을 탐구하며, 과거의 주관적인 평가 방식을 넘어서려고 합니다.

- **Technical Details**: 제안된 프레임워크는 후보자의 '정책(Policies)', '개인 이미지(Persona)', '관점(Perspective)'을 분석하고 이들이 유권자, 사업체, 기부자 및 정치인이라는 네 가지 주요 청중 그룹의 '관심(Interests)', '이념(Ideologies)', '정체성(Identity)'과 어떻게 조화되는지를 평가합니다. LLM-POTUS Score는 이 측면들의 정량적 지표로서, 3P와 3I 간의 정렬을 기반으로 합니다.

- **Performance Highlights**: 연구 결과는 후보자의 다른 토론 전략의 효과성과 이들이 다양한 청중 세그먼트에 미치는 영향을 드러냅니다. 또한, 이 도구는 미디어 편향으로부터 독립적인 평가를 제공하여 민주적 참여를 높이고, 유권자들이 스스로 평가할 수 있는 방법을 제공합니다.



### WhisperNER: Unified Open Named Entity and Speech Recognition (https://arxiv.org/abs/2409.08107)
- **What's New**: WhisperNER는 자동 음성 인식(ASR)과 이름 있는 엔티티 인식(NER)을 통합하여 음성 전사 및 엔티티 인식을 동시에 수행할 수 있는 새로운 모델입니다. 이 모델은 새로운 유형의 엔티티 인식을 가능하게 하며, ASR 프로세스에 직접 통합되어 오류 전파를 줄여줍니다.

- **Technical Details**: WhisperNER는 Whisper ASR 모델을 기반으로 하며, 합성 음성과 텍스트 데이터셋을 활용하여 훈련됩니다. 훈련 데이터는 350K 샘플과 약 1.8M 개의 고유 엔티티 타입으로 구성되어 있습니다. 모델은 오픈 타입 NER을 지원하여 훈련 시 관찰되지 않은 새로운 엔티티 타입에도 일반화할 수 있습니다.

- **Performance Highlights**: WhisperNER는 NER의 자연 기준선보다 뛰어난 성능을 보였으며, 개방형 NER 및 감독 하의 미세 조정(finetuning) 모두에서 우수한 결과를 나타냈습니다. 다양한 엔티티 태그를 출력하며, 기존 ASR 데이터셋에 새로운 엔티티 태그를 추가하는 방법으로 평가되었습니다.



### The Faetar Benchmark: Speech Recognition in a Very Under-Resourced Languag (https://arxiv.org/abs/2409.08103)
- **What's New**: 본 논문에서는 Faetar Automatic Speech Recognition Benchmark라는 새로운 벤치마크 코퍼스를 소개합니다. 이 코퍼스는 자원 부족 언어의 음성 인식 기술을 발전시키기 위해 설계되었습니다. Faetar는 이탈리아에서 주로 사용되는 프랑코-프로방살어의 한 종류로, 표준 확정된 철자도 없고 거의 모든 텍스트 및 음성 자원이 부족합니다. 이 벤치마크는 필드 레코딩에서 수집된 데이터로 이루어져 있으며, 약 5시간의 주석이 달린 음성과 20시간의 주석이 없는 음성을 포함하고 있습니다.

- **Technical Details**: Faetar는 이탈리아 아풀리아 지역의 Faeto에서 사용되는 언어로, 고유의 구조와 어휘를 가지고 있습니다. 코퍼스는 노이즈가 있는 필드 레코딩에서 추출되었으며, 상태 최적화 다국어 음성 기초 모델을 사용하여 최고 전화 오류율(Phone Error Rate)은 30.4%입니다. 코퍼스는 오디오와 독립적으로 생성된 주석 데이터가 거의 없어, 연구자들은 도전 과제를 통해 접근 방식을 비교할 수 있습니다.

- **Performance Highlights**: Faetar 코퍼스를 기반으로 한 실험 결과, 최고 전화 오류율(Phone Error Rate) 30.4%를 달성했습니다. 일반적으로, 자원 부족 언어의 음성 인식 성능 향상에 대한 연구에 기여할 수 있는 중요한 데이터를 제공합니다. 이 벤치마크는 연구자들이 자원의 제한에 대한 일반화 가능한 교훈을 도출할 수 있도록 돕는 것을 목표로 합니다.



### The CLC-UKET Dataset: Benchmarking Case Outcome Prediction for the UK Employment Tribuna (https://arxiv.org/abs/2409.08098)
- **What's New**: 이번 논문은 영국 고용 재판소(UKET)의 사건 결과 예측을 위한 벤치마크를 개발하여 기술 혁신과 접근성 문제를 탐구합니다. 대량의 수동 주석 작업을 해결하기 위해, 대형 언어 모델(LLM)을 사용하여 자동 주석을 생성한 CLC-UKET 데이터셋을 만들었습니다.

- **Technical Details**: CLC-UKET 데이터셋은 약 19,000개의 UKET 사건 및 메타데이터로 구성되며, 사실, 청구, 판례 참조, 법령 참조, 사건 결과 등 포괄적인 법적 주석이 포함되어 있습니다. 향후 UKET 사건의 다양한 결과를 예측하기 위한 멀티 클래스 논문 작업을 수행하였습니다. 피험자의 예측 값과 모델 성능을 비교하기 위한 기준을 설정하기 위한 인간 성과도 수집되었습니다.

- **Performance Highlights**: 파인튜닝된 변환 모델이 제로샷(Zero-Shot) 및 몇 샷(Few-Shot) LLMs보다 성능이 우수하다는 결과가 도출되었습니다. 제로샷 LLM의 성능은 작업 관련 정보를 몇 샷 예제에 통합함으로써 향상될 수 있음을 보여주었습니다. CLC-UKET 데이터셋은 UK 고용 관련 분쟁 해결을 위한 중요한 벤치마크로 작용할 전망입니다.



### A corpus-based investigation of pitch contours of monosyllabic words in conversational Taiwan Mandarin (https://arxiv.org/abs/2409.07891)
- **What's New**: 본 연구에서는 대만 마 찬(台灣閩南語)에서의 단음절 단어의 피치 컨투어(pitch contour)가 어떻게 맥락적인 예측기(Contextual predictors) 및 단어의 의미에 의해 공동 결정되는지를 탐구합니다. 이를 위해 3824개의 단어 토큰을 분석하였습니다.

- **Technical Details**: 연구는 일반화 가법 혼합 모델(generalized additive mixed model)을 사용하여 단음절 단어의 F0 컨투어를 분석하였으며, 이로 인해 단어의 비전형적인 음조가 어떻게 수정되는지를 보여줍니다.

- **Performance Highlights**: 연구 결과 T2와 T3는 낮은 평평한 음조로 나타나는 반면 T1은 높은 음조로, T4는 고중 하강 음조로 나타났습니다. 또한, 중립 음조(T0)는 정규 설명에 따르면 이전 음조에 기반해 실현되지만, 본 연구에서 독립적인 낮은 음조로 나타났음을 보여줍니다.



### Learning Rules from KGs Guided by Language Models (https://arxiv.org/abs/2409.07869)
Comments:
          proof of concept

- **What's New**: 이번 연구는 대규모 사전 학습된 언어 모델(Pre-trained Language Models)을 활용하여 지식 그래프(Knowledge Graph, KG)에서 파생된 규칙을 평가하는 새로운 접근방식을 소개합니다. 이 접근법은 기존 알고리즘의 한계를 극복하고 규칙 학습의 품질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 시스템 프로토타입은 언어 모델 ℒ를 대규모 텍스트 코퍼스를 학습시키거나 사전 학습된 모델을 이용해 규칙을 추출하고, 혼합 품질 함수(hybrid quality function)를 사용해 규칙을 평가합니다. 혼합 품질 함수는 기존 통계 품질 측정과 언어 모델 기반 점수를 결합하여 성능을 개선합니다. 특히, 언어 모델은 예측 및 규칙 평가에 중요한 역할을 합니다.

- **Performance Highlights**: 예비 실험에서는 유명한 Bert 언어 모델을 사용하여 Wiki44K 데이터셋에서 규칙 평가를 수행했으며, 언어 모델 사용의 잠재적 이점이 강조되었습니다. 교차 검증에서, 이 방법이 기존 KG 임베딩 모델보다 더 나은 성과를 나타내는 것이 관찰되었습니다.



### Controllable Synthetic Clinical Note Generation with Privacy Guarantees (https://arxiv.org/abs/2409.07809)
- **What's New**: 이 논문에서는 개인 건강 정보(PII, PHI)를 포함하는 데이터셋을 '복제(clone)'하는 새로운 방법을 제시합니다. 이 방법은 환자 개인 정보를 침해하지 않으면서도 원본 데이터의 주요 특징과 유용성을 유지하는 것을 목표로 합니다.

- **Technical Details**: 제안하는 방법은 Differential Privacy 기법을 활용하여 무작위로 노이즈를 추가함으로써 환자 개별 데이터의 기여를 숨깁니다. 우리가 사용한 모델은 Clinical Named Entity Extraction, Relationship Mapping 및 Instruction Tuning 등의 기술을 포함합니다.

- **Performance Highlights**: 실험 결과, 복제된 데이터셋은 전통적인 익명화 데이터셋에 비해 모델 성능을 향상시키며, 개인 정보 보호 기준도 충족함을 입증했습니다. 이 연구는 민감한 의료 데이터를 윤리적이고 효과적으로 활용할 수 있는 가능성을 제공합니다.



### Full-text Error Correction for Chinese Speech Recognition with Large Language Mod (https://arxiv.org/abs/2409.07790)
- **What's New**: 이 논문은 Automatic Speech Recognition (ASR) 시스템에서 생성된 긴 스피치 녹음의 전체 텍스트(error correction) 오류 수정을 위한 대규모 언어 모델(LLMs)의 효과성을 조사합니다. 주로 팟캐스트, 뉴스 방송 및 회의에서 생성된 전사 텍스트를 다룹니다.

- **Technical Details**: 연구팀은 중국어 전체 텍스트 오류 수정 데이터셋인 ChFT를 개발하였으며, 이 데이터셋은 텍스트 수집, 텍스트-음성 변환(TTS), ASR 및 오류 수정 쌍 추출기로 구성된 파이프라인을 활용하였습니다. LLM을 개선하기 위해 다양한 프롬프트와 출력 형식으로 미세 조정하였습니다.

- **Performance Highlights**: 미세 조정된 LLM은 여러 실험 환경에서 우수한 성능을 보여주었으며, 각 프롬프트마다 고유한 강점과 약점이 확인되었습니다. 무엇보다도, 전체 텍스트 오류 수정을 위한 유망한 기준선을 설정하는 데 기여했습니다.



### Stable Language Model Pre-training by Reducing Embedding Variability (https://arxiv.org/abs/2409.07787)
- **What's New**: 본 논문에서는 언어 모델의 안정적인 사전 훈련(pre-training)을 위한 새로운 접근 방식을 제안합니다. 'Token Embedding Variability (TEV)'를 통한 간단하고 효율적인 안정성 평가 방법을 탐구하며, 'Multi-head Low-Rank Attention (MLRA)' 아키텍처를 통해 안정성을 개선하는 방안을 모색합니다.

- **Technical Details**: TEV는 임베딩 레이어의 표준 편차를 기반으로 하여 사전 훈련의 안정성을 추정합니다. MLRA는 출력 임베딩 분산의 기하급수적인 성장을 제한하여 그래디언트 폭주(gradient explosion)를 방지하는 구조적 해법입니다. 이러한 접근 방식은 GPT-2 모델에서 실험적으로 검증되었습니다.

- **Performance Highlights**: MLRA를 사용한 GPT-2의 사전 훈련 결과, 모델의 안정성이 향상되고 perplexity가 감소함을 보여주었습니다. 특히 깊은 모델에서 그 효과가 더욱 두드러졌습니다.



### Supporting Online Discussions: Integrating AI Into the adhocracy+ Participation Platform To Enhance Deliberation (https://arxiv.org/abs/2409.07780)
- **What's New**: 이번 연구에서는 대규모 오픈 소스 참여 플랫폼인 adhocracy+에 두 가지 AI 기반의 토론 모듈을 추가하여 온라인 토론의 질과 참가자 간의 상호작용을 향상시키는 방안을 제안합니다.

- **Technical Details**: 토론 모듈은 스탠스 감지 모델을 기반으로 한 댓글 추천 기능(Comment Recommendation Module)과 AQuA 점수를 활용하여 고품질 댓글을 자동으로 강조하는 심의 품질 모듈(Deliberative Quality Module)으로 구성됩니다. 이를 통해 참가자들은 서로 다른 의견에 대응할 수 있는 댓글을 추천받고, 논의 과정에서 고품질의 댓글이 강조됩니다.

- **Performance Highlights**: 심의 품질 모듈에서는 댓글의 AQuA 점수를 계산하고, 고품질 댓글을 강조하여 사용자 참여를 증가시키는 효과를 목표로 합니다. 이는 사용자 간의 상호작용을 촉진하고, 더 의미 있는 토론을 유도할 것으로 기대됩니다.



### Ruri: Japanese General Text Embeddings (https://arxiv.org/abs/2409.07737)
- **What's New**: Ruri라는 일본어 일반 텍스트 임베딩 모델 시리즈의 개발이 보고되었습니다. 일본어 텍스트 임베딩 모델의 개발이 부족한 가운데, 이를 해결하기 위해 LLMs(대규모 언어 모델)로 생성된 합성 데이터셋을 사용하여 모델을 훈련시켰습니다.

- **Technical Details**: Ruri 개발 과정에서 두 단계의 학습 접근 방식이 사용되었습니다. 첫 번째 단계는 대규모 약한 감독 데이터셋을 활용한 contrastive pre-training(대비 사전 훈련)이며, 이후에는 수작업으로 레이블이 지정된 고품질 데이터셋을 이용한 fine-tuning(미세 조정)이 이루어졌습니다. 합성 데이터셋은 QA(질문-답변) 데이터셋과 NLI(자연어 추론) 데이터셋을 포함하여 모델 훈련에 사용되었습니다.

- **Performance Highlights**: Ruri 모델은 텍스트 임베딩 벤치마크에서 기존 모델들보다 현저히 높은 성능을 기록하였으며, 일본어 검색 모델 중 가장 우수한 성능을 보여주었습니다. 이는 기존 다국어 모델을 초월하는 성과입니다.



### Experimenting with Legal AI Solutions: The Case of Question-Answering for Access to Justic (https://arxiv.org/abs/2409.07713)
Comments:
          Accepted into GenLaw '24 (ICML 2024 workshop)

- **What's New**: 본 연구는 일반인을 위한 법률 상담에 있어 Generative AI 모델을 활용하는 새로운 접근 방식인 인간 중심의 법률 NLP 파이프라인을 제안합니다.

- **Technical Details**: 우리는 실제 법률 질문과 법률 전문가들의 답변으로 구성된 LegalQA 데이터셋을 소개하고 공개하였습니다. 이 데이터셋은 고품질의 질문-답변 쌍을 포함하며, 법률 전문가의 승인을 받았습니다. 또한, 효과적인 데이터 검색 과정을 통해 모델 성능 향상을 꾀하는 방안을 제시합니다.

- **Performance Highlights**: 법률 전문가가 승인한 850개의 인용 데이터에서 검색하는 retrieval-augmented generation 접근 방식이 인터넷 전반에서 쿼리 검색하는 것과 성능이 동등하거나 우수한 결과를 보여줍니다.



### An Unsupervised Dialogue Topic Segmentation Model Based on Utterance Rewriting (https://arxiv.org/abs/2409.07672)
Comments:
          in Chinese language

- **What's New**: 이번 연구에서는 다중 대화에서의 주제 세분화(Dialog topic segmentation)를 위한 새로운 비지도 학습(unsupervised learning) 모델인 Discourse Rewriting Topic Segmentation Model (UR-DTS)을 제안합니다. 이 모델은 대화 내용에서 동시 지시어(co-referent)와 생략된 단어들을 복원하기 위해 Utterance Rewriting (UR) 기법을 결합합니다.

- **Technical Details**: UR-DTS 모델은 인접 담화 매칭(adjacent discourse matching)과 의사 세분화(pseudo segmentation)를 통해 학습할 수 있는 유용한 단서를 최대한 활용하기 위해 대화 데이터를 재작성합니다. 이로 인해, 담화의 의미적 유사성 계산이 향상됩니다.

- **Performance Highlights**: DialSeg711에서의 절대 오류 점수(absolute error score)가 약 6% 향상되어 11.42%를 기록하였고, WD도 12.97%로 개선되었습니다. Doc2Dial에서는 절대 오류 점수가 약 3%, WD가 2% 향상되어 각각 35.17% 및 38.49%를 기록하며 SOTA(State-Of-The-Art)에 도달하였습니다.



### SimulBench: Evaluating Language Models with Creative Simulation Tasks (https://arxiv.org/abs/2409.07641)
Comments:
          Website: this https URL

- **What's New**: SimulBench를 소개합니다. 이는 다양한 창의적 시뮬레이션 시나리오에서 대형 언어 모델(LLMs)을 평가하기 위해 설계된 벤치마크입니다.

- **Technical Details**: SimulBench는 LLM의 일반 지능을 측정하기 위한 다중 턴 대화 스크립트를 사용하여 다양한 시뮬레이션 작업을 평가합니다. 다양한 인터페이스에서 작업을 수행하도록 설계된 109개의 시뮬레이션 작업을 수집했습니다.

- **Performance Highlights**: GPT-4-turbo가 LLaMA-3-70B-Chat보다 18.55% 더 많은 경우에서 우수한 성능을 보여줍니다. 최신 LLM들이 고유의 스테이트풀(stateful) 작업에서 더 나은 성능을 보이지만, 복잡한 시뮬레이션 작업에 대한 성능은 여전히 부족합니다.



### Zero-Shot Machine-Generated Text Detection Using Mixture of Large Language Models (https://arxiv.org/abs/2409.07615)
Comments:
          Preprint, work in progress

- **What's New**: 이 논문은 여러 개의 검출 모델을 조합하는 새로운 앙상블 방법을 제안하여 인공지능(AI) 텍스트의 탐지를 개선하는 방법을 탐구합니다. 최근의 텍스트 생성 AI의 발전으로 인해 허위 정보 및 악의적인 콘텐츠의 생산이 용이해지는 상황에서, 이러한 탐지 기법의 필요성이 커지고 있습니다.

- **Technical Details**: 논문에서는 여러 개의 Large Language Models (LLMs)가 각기 다른 강점을 활용하여 인공지능 생성 텍스트를 보다 효과적으로 탐지하는 앙상블 방법을 제시합니다. 본 방법은 정보 이론의 원리를 바탕으로 하며, 훈련 및 테스트 데이터로는 기존 벤치마크 및 새로운 기계 생성 텍스트 데이터를 사용하였습니다. 여러 LLM을 이용한 실험을 통해 우리의 방법이 다양한 생성 모델을 견고하게 탐지할 수 있음을 확인하였습니다.

- **Performance Highlights**: 우리의 방법은 여러 최근 제안들과 비교할 때 우수한 성과를 보여줍니다. 특히, 다양한 생성 모델을 적용한 실험 결과에 따르면, 모델을 앙상블에 추가할 때 탐지 성능이 증가함을 볼 수 있으며, 다국어 데이터셋을 활용한 분석을 통해 각 모델의 기여도가 어떻게 변화하는지도 탐구하였습니다.



### The Design of Informative Take-Over Requests for Semi-Autonomous Cyber-Physical Systems: Combining Spoken Language and Visual Icons in a Drone-Controller Setting (https://arxiv.org/abs/2409.08253)
Comments:
          21 pages, 8 figures

- **What's New**: 본 연구에서는 사이버 물리 시스템(cyber-physical systems)이 인간 파트너와 상호작용하는 방식을 개선하기 위해 중요 정보의 효과적인 전달을 위한 족독(작업의 요청, ToR, Take-Over Request) 메시지의 디자인을 제안합니다.

- **Technical Details**: 이 연구에서는 반 자율 드론 조종 시나리오를 테스트베드로 사용하여, 다양한 메시지 형식(전체 문장 vs. 단편)과 시각적 강조가 언어 메시지와 동기화(동기식) 또는 비동기화(비동기식)되는 경우의 반응을 비교하였습니다.

- **Performance Highlights**: 비모달 ToR 사용 시 참가자들은 더 나은 정확도를 보였고, 언어 메시지만 단편으로 작성할 경우 정확도나 반응 속도가 향상되지 않았습니다. 또한, 시각 강조가 언어와 동기화되었을 때 정확도가 증가하지 않고 반응 시간이 더 늘어나는 경향을 보였습니다.



### LLM Honeypot: Leveraging Large Language Models as Advanced Interactive Honeypot Systems (https://arxiv.org/abs/2409.08234)
Comments:
          7 pages, 5 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 활용하여 보다 현실적이고 인터랙티브한 honeypot 시스템을 개발하는 새로운 접근 방식을 제안합니다. 기존의 honeypot 기술의 한계를 극복하고, 공격자와의 상호작용을 보다 정교하게 실행할 수 있는 가능성을 탐구합니다.

- **Technical Details**: 이 연구는 공격자 생성 명령어와 응답 데이터셋을 기반으로 사전 훈련된 오픈 소스 언어 모델을 최적화하는 Supervised Fine-Tuning(SFT) 과정을 통해 LLM honeypot 시스템을 개발했습니다. 데이터 수집, 프롬프트 엔지니어링, 모델 선택 및 성능 최적화 등의 주요 단계를 포함하며, 최종적으로 모델은 공용 IP 주소에서 실제 공격자와 실시간으로 상호작용할 수 있도록 배포되었습니다.

- **Performance Highlights**: 이론적 검증 및 실제 배포를 통해 평가한 결과, 제안된 LLM honeypot는 정확하고 유용한 응답을 생성할 수 있는 효과적인 솔루션임을 입증했습니다. 이러한 기술은 공격자의 행동을 분석하고 비즈니스 보안 인프라를 강화하는 데 중요한 도구가 될 것으로 기대됩니다.



### What Makes a Maze Look Like a Maze? (https://arxiv.org/abs/2409.08202)
- **What's New**: 본 논문에서는 Deep Schema Grounding (DSG)이라는 새로운 프레임워크를 도입하여 시각적 추상 개념을 이해하는 데 도움을 주는 방법을 제안합니다. DSG는 시각적 추상화를 위한 명시적 구조화된 표현을 활용하여 추론합니다.

- **Technical Details**: DSG의 핵심 요소는 schema로, 이는 추상 개념을 보다 원시적인 기호로 분해하는 의존성 그래프(Dependency Graph) 설명입니다. DSG는 대형 언어 모델(LLM)을 사용하여 schema를 추출하고, 이를 비전-언어 모델(Vision-Language Model, VLM)과 계층적으로 결합하여 이미지에 대해 추상 개념의 구체적인 구성 요소를 접지합니다.

- **Performance Highlights**: DSG는 비전-언어 모델의 추상 시각 추론 성능을 유의미하게 향상시켰습니다. 특히, GPT-4V의 경우 8.3%의 상대적 개선을 보였고, Counting 관련 질문에서는 11.0%의 상대적 개선을 달성했습니다.



### TravelAgent: An AI Assistant for Personalized Travel Planning (https://arxiv.org/abs/2409.08069)
- **What's New**: TravelAgent라는 새로운 여행 계획 시스템을 소개합니다. 이 시스템은 대형 언어 모델(LLMs)을 기반으로 하여, 동적인 여행 시나리오에서 합리적이고, 포괄적이며 개인화된 여행 일정을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: TravelAgent는 Tool-usage, Recommendation, Planning, Memory 모듈의 네 가지로 구성됩니다. 각 모듈은 원활하게 협력하며, 여행 제약 사항을 모델링하여 계획 단계의 합리성을 향상시키고, 실시간 도구를 활용하여 포괄성을 해결하며, 개인화를 증진시키기 위한 새로운 추천 프레임워크를 제공합니다.

- **Performance Highlights**: TravelAgent는 인간 사용자와 시뮬레이션된 사용자 평가를 통해 Rationality(합리성), Comprehensiveness(포괄성), Personalization(개인화) 기준에서 높은 성능을 발휘하였으며, 기존의 GPT-4+ 에이전트보다 우수한 성능을 보였습니다.



### Enhanced Online Grooming Detection Employing Context Determination and Message-Level Analysis (https://arxiv.org/abs/2409.07958)
- **What's New**: 본 논문은 Online Grooming (OG) 공격에 대한 실시간 탐지 방법론을 제안합니다. 기존 솔루션의 한계를 극복하기 위해 BERT와 RoBERTa와 같은 고급 모델을 활용하며, 단순한 바이너리 분석을 넘어서 실질적인 커뮤니케이션 패턴을 명확히 해석합니다.

- **Technical Details**: 제안된 방법론은 Actor Significance Threshold와 Message Significance Threshold를 포함한 상호작용 분류를 위한 Context Determination 접근 방식을 갖추고 있습니다. 이는 OG 탐지의 정확성과 강인성을 향상시키는데 중점을 두고 있습니다. 다양한 데이터셋을 통한 실험을 통해 제안된 방법의 견고성과 다용성을 평가합니다.

- **Performance Highlights**: 이 연구는 OG 탐지의 범위를 확장하고 다각적인 공격의 복잡성을 더욱 잘 포착하는 효율적인 탐지 방법을 최초로 도입했습니다. 특히 크로스 데이터셋 실험을 통해 다양한 상황에 대한 적용 가능성을 보여주며, 기존 문헌의 공백을 채울 수 있는 의미 있는 기여를 합니다.



### FPMT: Enhanced Semi-Supervised Model for Traffic Incident Detection (https://arxiv.org/abs/2409.07839)
Comments:
          14 pages, 3 figures, accepted by ICPR 2024

- **What's New**: 이 논문은 반지도 학습(semi-supervised learning)을 활용한 교통 사고 탐지(traffic incident detection)에 중점을 두며, FPMT라는 새로운 모델을 제안합니다.

- **Technical Details**: FPMT 모델은 MixText 프레임워크 내에서 작동하며, Generative Adversarial Networks(GAN)를 통해 데이터 셋을 균형잡고 확장하는 데이터 증강(data augmentation) 모듈을 도입합니다. 숨겨진 공간(hidden space)에서의 mix-up 과정 동안 확률적 의사 혼합(probabilistic pseudo-mixing) 메커니즘을 적용해 정규화(regularization)를 강화하고 모델의 정밀도를 높입니다.

- **Performance Highlights**: FPMT 모델은 4개 실제 데이터셋에 대한 경험적 검증을 통해 다양한 지표에서 뛰어난 성능을 보였으며, 특히 낮은 레이블 비율(low label rates)에서도 강력한 성능을 나타냅니다.



### Online vs Offline: A Comparative Study of First-Party and Third-Party Evaluations of Social Chatbots (https://arxiv.org/abs/2409.07823)
- **What's New**: 이 논문은 대화형 챗봇 평가에서 온라인 방식과 오프라인 방식의 효능을 비교하는 연구를 소개합니다. 특히, 첫 번째 평가 방식인 1차 직접 상호작용과 3자 관찰 평가를 비교합니다.

- **Technical Details**: 이번 연구에서는 iEval 데이터셋을 활용하여 온라인 상호작용을 통한 사용자의 피드백과 오프라인 3자 평가의 결과를 비교했습니다. 1920개의 대화 데이터가 사용되었으며, 챗봇의 반응을 평가하기 위해 다섯 가지 기준(예: politeness, empathy 등)으로 평가 점수를 매겼습니다.

- **Performance Highlights**: 연구 결과, 오프라인 인간 평가가 온라인 평가보다 대화형 챗봇의 상호작용의 미세한 차이를 포착하지 못하는 것으로 나타났습니다. 더욱이, GPT-4 모델을 사용한 자동화된 3자 평가가 첫 번째 사용자 판단의 더 나은 근사치를 제공하는 것으로 평가되었습니다.



### Top-down Activity Representation Learning for Video Question Answering (https://arxiv.org/abs/2409.07748)
Comments:
          presented at MIRU2024

- **What's New**: 이 논문에서는 비디오 질문 응답(VideoQA) 작업에서 복잡한 계층적 인간 활동을 효과적으로 캡처하기 위해 CLIP 모델의 공간 시각적 상황 표현 능력을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 긴 비디오 시퀀스를 단일 그리드 이미지로 변환하고, LLaVA 멀티모달 모델을 파인튜닝합니다. 이를 통해 비디오의 맥락적 사건을 비연속적으로 표현할 수 있으며, STAR와 NExTQA 벤치마크에서 높은 성능을 달성했습니다.

- **Performance Highlights**: 이 연구는 STAR 작업에서 78.4%의 정확도를 기록하였으며, NExTQA 작업에서는 기존의 최첨단 점수를 2.8포인트 초과하는 성과를 보여주었습니다.



### Multi-object event graph representation learning for Video Question Answering (https://arxiv.org/abs/2409.07747)
Comments:
          presented at MIRU2024

- **What's New**: 이 논문에서는 비디오 질문 응답(Video Question Answering, VideoQA) 분야에서 다수 객체를 포함한 사건(event) 표현을 포착하기 위한 새로운 접근법인 CLanG(Contrastive Language Event Graph Representation Learning)를 제안하고 있습니다. 기존의 방법들은 개별 객체의 움직임에 초점을 맞추었지만, CLanG는 복잡한 시나리오를 처리할 수 있는 방법을 제공합니다.

- **Technical Details**: CLanG는 다층 GNN-클러스터 모듈을 사용하여 비디오에서 추출된 다수 객체의 사건 표현을 효과적으로 학습합니다. 이 모듈은 경쟁적 그래프 표현 학습을 통해 질문 텍스트와 관련된 다수 객체 사건 그래프 간의 대비 학습을 가능하게 합니다. 또한, 그래프의 출력 노드 수를 조정하여 원래의 그래프 정보를 보존하며, 자기 주의(self-attention) 계층을 통해 계층적 사건 그래프의 영향을 강조합니다.

- **Performance Highlights**: CLanG는 NExT-QA 및 TGIF-QA-R이라는 두 개의 도전적인 VideoQA 데이터셋에서 강력한 기준선보다 최대 2.2% 더 높은 정확도를 달성하며, 특히 인과(causal) 및 시간적(temporal) 질문 처리에서 2.8% 향상된 성능을 보여줍니다.



### Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG (https://arxiv.org/abs/2409.07691)
Comments:
          Accepted for the 1st Workshop on GenAI and RAG Systems for Enterprise @ CIKM 2024

- **What's New**: 이 논문은 다양한 공개된 ranking 모델에 대해 벤치마킹을 수행하고, 그것이 ranking 정확도에 미치는 영향을 분석합니다. 특히 Retrieval-Augmented Generation (RAG) 시스템에서의 질문-응답(text retrieval) 작업에 중점을 두고 있습니다. 또한, NV-RerankQA-Mistral-4B-v3라는 최신 ranking 모델을 소개하며 이전의 reranker와 비교하여 약 14%의 정확도 향상을 달성했습니다.

- **Technical Details**: 논문에서는 Transformer 아키텍처를 기반으로 한 sparsity 및 density embedding 모델이 상위 K 후보 passage를 검색한 후, ranking 모델이 해당 passage의 최종 순위를 재정렬하는 다단계 text retrieval 파이프라인을 제안합니다. 특히, 다양한 크기와 손실(attributive loss) 및 self-attention 메커니즘을 가진 ranking 모델의 fine-tuning 방법에 대한 ablation study를 포함합니다. 또한, 이들 모델이 상용 사용에 적합한지 여부를 평가합니다.

- **Performance Highlights**: NV-RerankQA-Mistral-4B-v3 모델은 기존 reranker 기반의 파이프라인에 비해 약 14%의 정확도 증가를 보이며, 텍스트 검색 시스템의 indexing 및 serving latency와 throughput 간의 trade-off를 논의합니다. 이는 상용 텍스트 검색 파이프라인을 위한 유용한 벤치마크로 기업에서 적용할 수 있는 모델을 제공합니다.



### Leveraging User-Generated Reviews for Recommender Systems with Dynamic Headers (https://arxiv.org/abs/2409.07627)
Comments:
          7 pages, 3 figures, PAIS 2024 (ECAI)

- **What's New**: 본 연구는 기존의 정적인 추천 캐러셀 헤더 텍스트 대신, 사용자 생성 리뷰를 활용하여 역동적인 헤더 텍스트 생성을 제안합니다. 이러한 접근 방식을 통해 더 개인화된 추천 경험을 제공할 수 있습니다.

- **Technical Details**: 이 연구에서는 사용자 리뷰에서 긍정적인 속성(아스펙트)를 추출하고, 이를 기반으로 한 조건부 순위 작업 프레임워크 하에서 그래프 신경망(Graph Neural Network) 모델을 훈련합니다. 우리가 제안하는 방법론은 Dynamic Text Snippets (DTS)로, 이는 앵커 아이템과 회수 세트를 위한 여러 개의 헤더 텍스트를 생성합니다.

- **Performance Highlights**: 수행된 오프라인 및 온라인 실험을 통해 DTS 모델의 효과를 평가한 결과, 아스펙트 정보를 통합함으로써 추천의 다양성이 유의하게 향상되었음을 확인했습니다.



### OneEdit: A Neural-Symbolic Collaboratively Knowledge Editing System (https://arxiv.org/abs/2409.07497)
Comments:
          LLM+KG@VLDB2024, code is available at this https URL

- **What's New**: OneEdit는 사용자가 자연어를 통해 지식을 쉽게 관리할 수 있도록 돕는 신경-상징(Neural-Symbolic) 프로토타입 시스템으로, 지식 그래프(KG)와 대형 언어 모델(LLM)을 통합했습니다.

- **Technical Details**: OneEdit는 세 가지 주요 모듈로 구성되어 있습니다: 1) Interpreter는 자연어를 통한 사용자 상호작용을 담당합니다; 2) Controller는 다양한 사용자로부터 편집 요청을 관리하며 갈등 해결과 독성 지식 공격 방지를 위한 롤백(rollback) 기능을 사용합니다; 3) Editor는 Controller의 지식을 활용하여 KG와 LLM을 편집합니다.

- **Performance Highlights**: OneEdit는 두 개의 새로운 데이터셋을 활용하여 실험을 실시했으며, KG를 사용한 지식 편집에서 기존 모델을 초월하는 성능을 보여주었습니다.



### Responsible AI for Test Equity and Quality: The Duolingo English Test as a Case Study (https://arxiv.org/abs/2409.07476)
- **What's New**: 이번 장(chapter)에서는 인공지능(AI)의 평가(evaluation) 활용에 따른 기회와 위험에 대해 다루며, 책임 있는 AI(Responsible AI, RAI) 관행의 중요성을 강조합니다.

- **Technical Details**: 특히, 이 장에서는 Duolingo English Test (DET)라는 AI 기반의 고위험 영어 언어 평가 사례를 통해 RAI 기준과의 관계를 탐구하며, RAI의 도메인 불가지론(domain-agnostic) 원칙들과의 연관성을 설명합니다.

- **Performance Highlights**: RAI 관행의 구체적인 예시를 통해 검사의 공정성(fairness) 및 품질(quality)을 보장하기 위한 윤리적 원칙인 유효성(validity), 신뢰성(reliability), 프라이버시(priacy) 및 보안(security), 투명성(transparency) 및 책임(accountability) 기준을 어떻게 의미 있게 충족하는지를 보여줍니다.



New uploads on arXiv(cs.IR)

### On the challenges of studying bias in Recommender Systems: A UserKNN case study (https://arxiv.org/abs/2409.08046)
Comments:
          Accepted at FAccTRec@RecSys 2024, 11 pages

- **What's New**: 이 논문은 추천 시스템에서 발생하는 인기 편향(popularity bias)을 측정하고 보고하는 데 있어 직면하는 여러 도전 과제를 탐구합니다. 특히, 데이터 특성과 알고리즘 구성의 영향을 조합하여 실험을 통해 발견한 내용을 제시합니다.

- **Technical Details**: 인기 편향을 측정하기 위해 UserKNN을 구현한 다양한 추천 시스템 프레임워크와 합성 데이터(synthetic data)를 결합하여 평가합니다. 총 5개의 합성 데이터셋과 5개의 UserKNN 구성을 사용하여 인기 편향을 평가하고 이들의 상호 작용이 갖는 영향을 파악합니다.

- **Performance Highlights**: 결과에 따르면, 데이터 특성에 따라 다양한 UserKNN 구성이 인기 편향의 전파에 미치는 영향이 달라질 수 있습니다. 또한 UserKNN 구성의 설정(예: 최소 유사도 및 최소 이웃)도 인기 편향의 강도에 큰 영향을 미침을 확인하였습니다.



### An Evaluation Framework for Attributed Information Retrieval using Large Language Models (https://arxiv.org/abs/2409.08014)
- **What's New**: 본 논문에서는 정보 검색을 위한 새로운 프레임워크를 제안하며, 기존 연구들이 질문-답변(attributed question answering)에 집중하는 반면, 더 도전적인 정보 탐색(information-seeking) 질문에 중점을 두었습니다.

- **Technical Details**: 제안된 프레임워크는 (1) Generate, (2) Retrieve then Generate, (3) Generate then Retrieve 세 가지 아키텍처에서 사용 가능하며, HAGRID라는 정보 탐색 데이터셋을 활용하여 실험을 수행합니다.

- **Performance Highlights**: HAGRID 데이터셋을 활용한 실험 결과, 다양한 시나리오가 답변의 정확성과 attributed 품질에 미치는 영향을 보여줍니다.



### Collaborative Automatic Modulation Classification via Deep Edge Inference for Hierarchical Cognitive Radio Networks (https://arxiv.org/abs/2409.07946)
Comments:
          arXiv admin note: text overlap with arXiv:2407.20772

- **What's New**: 최근의 연구에서는 계층적 인지 무선 네트워크에서 엣지 장치와 엣지 서버 간의 협업 자동 변조 분류(C-AMC)를 위한 엣지 학습(EL) 기반 프레임워크를 제안했습니다. 이는 데이터 전송 비용과 프라이버시 문제를 해결하기 위해 설계되었습니다.

- **Technical Details**: 제안된 C-AMC 프레임워크는 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 스펙트럼 의미 압축 신경망(SSCNet) - 엣지 장치에서 원시 데이터를 압축하여 엣지 서버로 전송합니다. (2) 변조 분류 신경망(MCNet) - 엣지 서버에서 신호의 변조 유형을 결정하기 위해 양방향 장기 단기 메모리(Bi-LSTM)와 다중 헤드 주의 레이어를 결합하여 사용합니다.

- **Performance Highlights**: 시뮬레이션 결과, C-AMC 프레임워크는 기존 방법에 비해 모델 크기와 계산 복잡성을 크게 줄이고, 전송 오버헤드를 효과적으로 감소시켰습니다. 또한, 압축 비율, 신호 대 잡음 비율(SNR) 등이 분류 정확도에 미치는 영향을 분석하여 실제 응용을 위한 유용한 통찰력을 제공했습니다.



### Enhancing Cross-Market Recommendation System with Graph Isomorphism Networks: A Novel Approach to Personalized User Experienc (https://arxiv.org/abs/2409.07850)
Comments:
          7 pages, 1 figure, 3 tables, 5 equations

- **What's New**: 본 논문에서는 Graph Isomorphism Networks (GINs)를 활용한 CrossGR 모델을 제안하여 cross-market recommendation 시스템(CMRs)의 성능을 향상시키고 있습니다. 이 모델은 NDCG@10 및 HR@10 메트릭에서 기존 벤치마크를 초월하며, 다양한 시장 세그먼트를 처리하는 데 있어 적응성과 정확성을 나타냅니다.

- **Technical Details**: CrossGR 모델은 GNNs (Graph Neural Networks)를 사용하여 시장별 특성과 데이터 희소성을 극복하고 있으며, 고차 연결성 (high-order connectivity)을 통해 더 복잡한 패턴과 관계를 감지할 수 있도록 설계되었습니다. 이로 인해 CMR 시스템에서 사용자 맞춤형 추천을 제공할 수 있습니다.

- **Performance Highlights**: CrossGR 모델은 평가 시간에 관계없이 일관된 성능을 보여줘, 새로운 시장 동향 및 사용자 선호도를 반영할 수 있는 잠재력을 가지고 있습니다. 이는 글로벌 전자상거래의 동적 환경에서 사용자 친화적인 추천 시스템으로 발전할 수 있는 기회를 제공합니다.



### PDC-FRS: Privacy-preserving Data Contribution for Federated Recommender System (https://arxiv.org/abs/2409.07773)
- **What's New**: 이번 논문에서는 사용자의 데이터 기여를 차별적 프라이버시 보장을 통해 가능하게 하는 최초의 연합 추천 프레임워크인 PDC-FRS를 제안합니다. 이로 인해 사용자들은 안전하게 데이터를 공유할 수 있게 되었습니다.

- **Technical Details**: PDC-FRS는 사용자가 공유한 데이터를 기반으로 보조 모델을 훈련시킵니다. 이 보조 모델은 각 사용자의 로컬 데이터셋을 증강시키고 전역 협력 정보를 통합하여 연합 추천 시스템(FedRec)의 성능을 향상시킵니다. PDC-FRS는 또한 지수 메커니즘(exponential mechanism)을 사용하여 사용자가 공유하는 데이터에 대해 지역 차별적 프라이버시(local differential privacy)를 제공합니다.

- **Performance Highlights**: PDC-FRS는 두 개의 인기 있는 추천 데이터셋에 대한 광범위한 실험을 통해 기존의 연합 추천 기법들과 비교하여 우수한 성능을 보여주었습니다.



### Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG (https://arxiv.org/abs/2409.07691)
Comments:
          Accepted for the 1st Workshop on GenAI and RAG Systems for Enterprise @ CIKM 2024

- **What's New**: 이 논문은 다양한 공개된 ranking 모델에 대해 벤치마킹을 수행하고, 그것이 ranking 정확도에 미치는 영향을 분석합니다. 특히 Retrieval-Augmented Generation (RAG) 시스템에서의 질문-응답(text retrieval) 작업에 중점을 두고 있습니다. 또한, NV-RerankQA-Mistral-4B-v3라는 최신 ranking 모델을 소개하며 이전의 reranker와 비교하여 약 14%의 정확도 향상을 달성했습니다.

- **Technical Details**: 논문에서는 Transformer 아키텍처를 기반으로 한 sparsity 및 density embedding 모델이 상위 K 후보 passage를 검색한 후, ranking 모델이 해당 passage의 최종 순위를 재정렬하는 다단계 text retrieval 파이프라인을 제안합니다. 특히, 다양한 크기와 손실(attributive loss) 및 self-attention 메커니즘을 가진 ranking 모델의 fine-tuning 방법에 대한 ablation study를 포함합니다. 또한, 이들 모델이 상용 사용에 적합한지 여부를 평가합니다.

- **Performance Highlights**: NV-RerankQA-Mistral-4B-v3 모델은 기존 reranker 기반의 파이프라인에 비해 약 14%의 정확도 증가를 보이며, 텍스트 검색 시스템의 indexing 및 serving latency와 throughput 간의 trade-off를 논의합니다. 이는 상용 텍스트 검색 파이프라인을 위한 유용한 벤치마크로 기업에서 적용할 수 있는 모델을 제공합니다.



### Leveraging User-Generated Reviews for Recommender Systems with Dynamic Headers (https://arxiv.org/abs/2409.07627)
Comments:
          7 pages, 3 figures, PAIS 2024 (ECAI)

- **What's New**: 본 연구는 기존의 정적인 추천 캐러셀 헤더 텍스트 대신, 사용자 생성 리뷰를 활용하여 역동적인 헤더 텍스트 생성을 제안합니다. 이러한 접근 방식을 통해 더 개인화된 추천 경험을 제공할 수 있습니다.

- **Technical Details**: 이 연구에서는 사용자 리뷰에서 긍정적인 속성(아스펙트)를 추출하고, 이를 기반으로 한 조건부 순위 작업 프레임워크 하에서 그래프 신경망(Graph Neural Network) 모델을 훈련합니다. 우리가 제안하는 방법론은 Dynamic Text Snippets (DTS)로, 이는 앵커 아이템과 회수 세트를 위한 여러 개의 헤더 텍스트를 생성합니다.

- **Performance Highlights**: 수행된 오프라인 및 온라인 실험을 통해 DTS 모델의 효과를 평가한 결과, 아스펙트 정보를 통합함으로써 추천의 다양성이 유의하게 향상되었음을 확인했습니다.



### Multilingual Prompts in LLM-Based Recommenders: Performance Across Languages (https://arxiv.org/abs/2409.07604)
- **What's New**: 이번 연구는 비영어 프롬프트(non-English prompts)가 추천 성능에 미치는 영향을 조사합니다. 또한 OpenP5 플랫폼을 이용하여 영어, 스페인어, 터키어 프롬프트를 사용한 LLM 기반 추천 시스템의 성능을 평가하였습니다.

- **Technical Details**: 연구자들은 OpenP5 플랫폼을 통해 LLM 기반 추천 모델을 개발하고 평가하였습니다. 이 모델의 프롬프트 템플릿을 영어에서 스페인어와 터키어로 확장하였고, 이 다양한 언어 프롬프트의 효과를 세 가지 실제 데이터셋(ML1M, LastFM, Amazon-Beauty)에서 평가했습니다. 구체적으로, 비영어 프롬프트를 사용할 경우 성능이 낮아지고, 특히 자원이 부족한 터키어에서는 더욱 감소되는 경향을 보였으며, 이 모델을 다국어 프롬프트로 재훈련할 경우 성능이 균형 있게 개선되는 결과를 발견했습니다.

- **Performance Highlights**: 비영어 프롬프트를 사용하는 경우 성능 저하가 발생했으며, 특히 터키어와 같이 자원이 부족한 언어에서 두드러졌습니다. 그러나 다국어 프롬프트로 재훈련된 모델은 언어 간 성능이 보다 균형 잡혀졌으나, 영어 프롬프트 성능은 약간 감소했습니다.



### Music auto-tagging in the long tail: A few-shot approach (https://arxiv.org/abs/2409.07730)
Comments:
          Published in Audio Engineering Society NY Show 2024 as a Peer Reviewed (Category 1) paper

- **What's New**: 본 연구는 음악 자동 태깅(music auto-tagging) 작업에 대해 few-shot learning 기법을 도입하였으며, 사전 훈련된 오디오 임베딩(pre-trained audio embeddings)을 입력으로 사용하는 간단한 다중 레이블 모델을 제안합니다.

- **Technical Details**: 이 연구에서 제안한 모델은 lightweight linear classifier(가벼운 선형 분류기)를 사용하며, Binary Cross-Entropy loss 및 Sigmoid 활성화를 통해 동시에 여러 태그를 처리할 수 있도록 설계되었습니다. 다양한 사전 훈련된 특성과 클래스 수, 클래스 별 샘플 수의 변화를 실험하여 모델의 성능을 평가했습니다.

- **Performance Highlights**: 제안된 모델은 단 20개의 샘플로도 최신 모델에 근접한 성능을 달성하였으며, 전체 훈련 데이터세트에서 훈련될 경우 유명 모델들과 경쟁력을 갖출 수 있음을 보였습니다. 이 연구는 transfer learning 기반의 few-shot 접근 방식이 제한된 라벨 데이터로도 긴 꼬리 태그를 자동 할당하는 문제를 효과적으로 해결할 수 있음을 보여주었습니다.



### Harnessing TI Feeds for Exploitation Detection (https://arxiv.org/abs/2409.07709)
Comments:
          This paper appears at IEEE International Conference on Cyber Security and Resilience (IEEE CSR 2024)

- **What's New**: 이 논문은 다양한 Threat Intelligence (TI) 피드에서 취약점 악용을 자동으로 탐지하는 머신 러닝 파이프라인을 제시합니다. 특히, 최신 임베딩 기법인 Doc2Vec과 BERT를 사용하여 느슨하게 구조화된 TI 피드에서 위협 어휘를 모델링하고 이를 통해 감독 학습 분류기를 훈련시킵니다.

- **Technical Details**: 이 연구는 TI 피드의 정보 흐름을 분석하고, 보안 취약점의 악용 탐지에 특화된 TI2Vec과 TIBERT라는 특화된 임베딩 기법을 제안합니다. Static Embeddings와 Dynamic Embeddings를 결합하여 TI 피드의 비정형 정보를 효과적으로 캡처하여, 분류기를 훈련합니다.

- **Performance Highlights**: 본 연구의 longitudinal evaluation 결과, 제안한 방식이 과거 데이터를 사용하여 학습하고도 withheld된 TI 피드에서 취약점 악용 사건을 정확하게 식별할 수 있음을 보여주었습니다. 이는 데이터 기반 취약점 위험 평가와 같은 다양한 하위 작업에 유용할 것으로 기대됩니다.



### DV-FSR: A Dual-View Target Attack Framework for Federated Sequential Recommendation (https://arxiv.org/abs/2409.07500)
- **What's New**: 이 논문은 연쇄 추천 시스템(Federated Sequential Recommendation, FSR)에서의 타겟 공격(targeted attack) 및 방어 메커니즘에 대한 새롭고 체계적인 접근을 제안한다. 특히, 기존의 공격 방법들이 FSR에 대한 제한된 유효성을 가지고 있다는 점을 강조하며, 새로운 이중 관점 공격 프레임워크(DV-FSR)를 소개한다.

- **Technical Details**: DV-FSR는 샘플링 기반의 명시적(exlicit) 전략과 대조 학습 기반의 암시적(implicit) 기울기 전략을 결합하여 조직적인 공격을 수행한다. 이 연구는 연쇄 추천 시스템을 위한 강화된 방어 메커니즘을 제안하며, 이는 공격 효과를 완화하는 것을 목표로 한다. 또한, 제안된 방법은 기하 중앙값(geometric median) 개념을 바탕으로 하여 하이브리드 강건 가중 집계 알고리즘을 설계하였다.

- **Performance Highlights**: 실험 결과, DV-FSR 공격 및 방어 전략은 두 개의 공개 데이터셋을 이용하여 검증되었으며, 기존의 공격 방법들이 FSR에 적용될 때의 효과가 제한적이라는 점이 입증되었다. 제안된 방법은 연쇄 추천 모델에서의 보안 취약점을 드러내며, 향후 연구에 대한 기초를 마련한다.



### OneEdit: A Neural-Symbolic Collaboratively Knowledge Editing System (https://arxiv.org/abs/2409.07497)
Comments:
          LLM+KG@VLDB2024, code is available at this https URL

- **What's New**: OneEdit는 사용자가 자연어를 통해 지식을 쉽게 관리할 수 있도록 돕는 신경-상징(Neural-Symbolic) 프로토타입 시스템으로, 지식 그래프(KG)와 대형 언어 모델(LLM)을 통합했습니다.

- **Technical Details**: OneEdit는 세 가지 주요 모듈로 구성되어 있습니다: 1) Interpreter는 자연어를 통한 사용자 상호작용을 담당합니다; 2) Controller는 다양한 사용자로부터 편집 요청을 관리하며 갈등 해결과 독성 지식 공격 방지를 위한 롤백(rollback) 기능을 사용합니다; 3) Editor는 Controller의 지식을 활용하여 KG와 LLM을 편집합니다.

- **Performance Highlights**: OneEdit는 두 개의 새로운 데이터셋을 활용하여 실험을 실시했으며, KG를 사용한 지식 편집에서 기존 모델을 초월하는 성능을 보여주었습니다.



New uploads on arXiv(cs.CV)

### DreamHOI: Subject-Driven Generation of 3D Human-Object Interactions with Diffusion Priors (https://arxiv.org/abs/2409.08278)
Comments:
          Project page: this https URL

- **What's New**: DreamHOI라는 새로운 방법론을 제안하여, 주어진 텍스트 설명에 기반해 3D 인간 모델이 임의의 객체와 현실적으로 상호작용할 수 있는 제로샷(zero-shot) 합성을 가능하게 합니다.

- **Technical Details**: 이 방법은 3D 인간 메쉬의 관절 움직임 최적화를 위해 Score Distillation Sampling (SDS) 기법을 활용하며, 이를 통해 이미지 공간에서의 편집 예측을 수행합니다. 복잡한 관절 매개변수에 대한 직접적인 역전파(backpropagation)는 비효율적이므로, 우리는 스킨드 메쉬의 이중 암묵적-명시적(implicit-explicit) 표현을 도입하여 NeRF와 골격 기반 매쉬의 관절 움직임을 결합합니다.

- **Performance Highlights**: 우리의 접근 방식을 통해 다양한 실험을 수행한 결과, 현실적인 인간-객체 상호작용(HOIs)을 생성하는 데 효과적임을 입증하였습니다.



### Depth on Demand: Streaming Dense Depth from a Low Frame Rate Active Sensor (https://arxiv.org/abs/2409.08277)
Comments:
          Accepted for publication at the European Conference on Computer Vision (ECCV) 2024

- **What's New**: 본 논문에서는 Depth on Demand (DoD) 기술을 활용하여 로봇 공학 및 자동차 인식에 필수적인 고속 및 정확한 깊이 추정을 가능하게 하는 방법을 제안합니다.

- **Technical Details**: DoD는 고속의 RGB 센서를 사용하여 정확한 시간적(Temporal) 및 공간적(Spatial) 깊이 밀도를 구현합니다. 이는 낮은 프레임 속도와 희박한 활성 깊이 센서를 결합하여 이루어지며, 세 가지 핵심 단계인 i) 다중 모드 인코딩(multi-modal encoding), ii) 반복적 다중 모드 통합(iterative multi-modal integration), iii) 깊이 디코딩(depth decoding)을 통해 이루어집니다.

- **Performance Highlights**: 우리는 DoD 기술이 indoor 및 outdoor 비디오 데이터셋에서 실내 환경 스캔 및 자동차 인식 사용 사례를 포함한 다양한 상황에서 효과적임을 보여주는 증거를 제시합니다.



### Click2Mask: Local Editing with Dynamic Mask Generation (https://arxiv.org/abs/2409.08272)
Comments:
          Project page is available at this https URL

- **What's New**: 이번 논문은 Click2Mask라는 새로운 접근 방식을 제안하여 사용자 입력을 단순화하고, 지역적으로 새로운 콘텐츠를 추가하는 과정에서 기존의 정확한 마스크나 상세한 설명 없이 단 하나의 기준 점만으로 편집할 수 있게 합니다.

- **Technical Details**: Click2Mask는 Blended Latent Diffusion (BLD) 프로세스를 활용하여 제공된 기준 점 주위로 동적으로 마스크를 성장시키며, 이를 masked CLIP 기반의 semantic loss로 안내합니다. 기존의 세분화(segmentation)-기반 및 세밀한 조정(fine-tuning) 방법의 한계를 극복하였습니다.

- **Performance Highlights**: 실험 결과 Click2Mask는 사용자 effort를 최소화할 뿐만 아니라, 인간 판단과 자동 메트릭에 따라 SoTA(state-of-the-art) 방법들과 비교해 경쟁력 있는 성능 또는 우수한 결과를 보여주었습니다.



### DreamBeast: Distilling 3D Fantastical Animals with Part-Aware Knowledge Transfer (https://arxiv.org/abs/2409.08271)
Comments:
          Project page: this https URL, code: this https URL

- **What's New**: 이 논문에서는 DreamBeast라는 새로운 방법론을 소개합니다. 이 방법론은 score distillation sampling (SDS)을 기반으로 하여 독특한 부분으로 구성된 환상적인 3D 동물 자산을 생성하는 데 초점을 맞추고 있습니다. 기존의 SDS 방법들은 텍스트-이미지 확산 모델에서 부분 수준의 의미를 제대로 이해하지 못해 이러한 생성 작업에서 어려움을 겪고 있습니다.

- **Technical Details**: DreamBeast는 Stable Diffusion 3(SD3) 모델에서 파트 수준의 지식을 효율적으로 추출하여 3D Part-Affinity 형태로 변환하는 독창적인 파트 인식 지식 전달 메커니즘을 통해 이러한 한계를 극복합니다. 이 방법은 원래의 SDS를 통해 생성된 자산을 바탕으로 파트-어피니티 맵을 즉시 생성하여 멀티-뷰 확산 모델의 안내를 조절하는 데 사용됩니다.

- **Performance Highlights**: DreamBeast는 사용자 지정 파트 조합에 따라 생성되는 3D 생물의 품질을 획기적으로 향상시키면서 계산 비용을 줄일 수 있습니다. 실험을 통해 DreamBeast는 생성 시간을 7시간에서 78분으로 줄이고 GPU 메모리 사용량도 24GB 줄였습니다. 또한 정량적 및 정성적 평가 결과, DreamBeast는 뛰어난 3D 동물 생성 능력을 입증했습니다.



### FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally (https://arxiv.org/abs/2409.08270)
Comments:
          ECCV'2024

- **What's New**: 본 연구는 2D 마스크에서 3D Gaussian Splatting(3D-GS) 세분화(세그멘테이션)를 정확하게 수행하는 도전에 집중하고 있습니다. 기존의 방법은 반복적인 gradient descent에 의존했지만, 본 논문에서는 간단하면서도 전역적으로 최적화된 솔루션을 제안합니다.

- **Technical Details**: 제안된 접근법은 재구성된 3D-GS 장면을 활용하여 2D 마스크 렌더링이 각 Gaussian의 레이블에 대한 선형 함수로 보일 수 있음을 인식합니다. 여기서 최적의 레이블 할당은 linear programming(선형 프로그래밍)을 통해 닫힌 형태로 해결할 수 있습니다. 또한, 최적화의 목표 함수에 배경 편향을 통합하여, 3D 세분화에서 노이즈에 대한 강력함을 보여줍니다. 최적화는 약 30초 내에 완료되며, 기존 방법보다 50배 빠릅니다.

- **Performance Highlights**: 광범위한 실험을 통해 다양한 장면을 세분화하는 데 있어 제안된 방법의 우수성을 입증하였으며, 객체 제거 및 인페인팅과 같은 다운스트림(downstream) 작업에서도 개선된 성능을 보여주었습니다.



### Improving Text-guided Object Inpainting with Semantic Pre-inpainting (https://arxiv.org/abs/2409.08260)
Comments:
          ECCV 2024. Source code is available at this https URL

- **What's New**: 최근 대규모 텍스트-이미지 확산 모델의 성공은 고품질 이미지를 생성할 수 있는 놀라운 가능성으로 이어졌습니다. 본 논문에서는 텍스트 프롬프트에 의해 설명된 새로운 객체를 특정 지역에 삽입하는 'inpainting' 작업을 탐구하여 이미지의 편집 가능성을 향상시키고자 합니다.

- **Technical Details**: 본 연구에서는 전통적인 단일 단계 객체 inpainting 방식에서 벗어나 두 가지 연속적인 과정으로 분해합니다: 1) 다중 모달 특징 공간에서 원하는 객체의 의미적 특징을 추론하는 '의미적 전처리(inpainting)'와 2) 이러한 의미적 특징에 기초하여 diffusion 잠재 공간에서 객체 생성을 수행하는 단계입니다. 이를 위해 우리는 Transformer 기반의 의미적 인페인터와 객체 인페인팅 diffusion 모델을 연계한 새로운 CAscaded Transformer-Diffusion (CAT-Diffusion) 프레임워크를 제안합니다.

- **Performance Highlights**: CAT-Diffusion은 OpenImages-V6와 MSCOCO 데이터셋에 대한 광범위한 평가를 통해 최첨단 방법들에 대한 우수성을 입증했습니다. 특히, 제안된 프레임워크는 정확한 텍스트-객체 정렬 학습을 용이하게 하며, 이를 통해 높은 성능의 객체 inpainting이 가능합니다.



### Improving Virtual Try-On with Garment-focused Diffusion Models (https://arxiv.org/abs/2409.08258)
Comments:
          ECCV 2024. Source code is available at this https URL

- **What's New**: 이 논문에서는 GarDiff라는 새로운 Diffusion 모델을 제안하여 VTON(이미지 기반 가상 착용) 작업을 위한 의복 중심의 확산 과정을 구현합니다. GarDiff는 CLIP과 VAE 인코딩에서 도출된 시각적 외관과 세부 텍스처(고주파 세부 사항)의 정보를 통합하여, 타겟 의복을 착용한 사람의 고품질 이미지를 생성합니다.

- **Technical Details**: GarDiff는 사전 학습된 Latent Diffusion Model을 기반으로 하며, 의복에 대한 시각적 정보 및 고주파 텍스처 세부 사항을 극대화하여 Diffusion 과정을 안내합니다. 고유한 의복 중심 어댑터가 UNet에 통합되어 사람 이미지와 참조 의복 및 인체 포즈 간의 정밀한 정렬(local fine-grained alignment)을 추구합니다. 또한, 고주파 세부 사항의 보존을 위해 의복에 대한 외관 손실(appearance loss)을 설계합니다.

- **Performance Highlights**: VITON-HD 및 DressCode 데이터셋에 대한 광범위한 실험을 통해, GarDiff는 기존 최첨단 VTON 방법들과 비교하여 우수한 성능을 보여주었습니다.



### Dynamic Prompting of Frozen Text-to-Image Diffusion Models for Panoptic Narrative Grounding (https://arxiv.org/abs/2409.08251)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 이 논문에서는 Extractive-Injective Phrase Adapter (EIPA)라는 새로운 구조를 소개하여, 이미지 특징을 사용하여 문구 프롬프트를 동적으로 업데이트하고 다중 모달 정보를 주입하는 방식으로 Panoptic Narrative Grounding (PNG) 성능을 향상시킵니다.

- **Technical Details**: EIPA 구조는 고정된 Diffusion 모델을 PNG 작업에 적용할 때 발생하는 큰 작업 격차와 비전-언어(vision-language) 상호작용의 부족 문제를 해결합니다. 또한, Multi-Level Mutual Aggregation (MLMA) 모듈을 설계하여 다중 수준의 이미지 및 문구 특징을 상호 융합하여 세분화를 개선합니다.

- **Performance Highlights**: PNG 벤치마크에서 본 연구 방법이 새로운 최첨단(state-of-the-art) 성능을 달성하였음을 확인했습니다.



### TextBoost: Towards One-Shot Personalization of Text-to-Image Models via Fine-tuning Text Encoder (https://arxiv.org/abs/2409.08248)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 단 하나의 참조 이미지로 고품질의 개인화를 달성하는 혁신적인 접근 방식을 제안합니다. 기존의 방법들이 성능 저하를 겪는 문제를 해결하고, 특히 텍스트 인코더에 대한 세밀한 미세 조정(selective fine-tuning) 전략을 통해 개인화 성능을 향상시킵니다.

- **Technical Details**: 주요 기술로는 (1) 특징 분리를 촉진하고 과적합(overfitting)을 완화하는 augmentation token 도입, (2) 문자 드리프트(language drift)를 줄이고 다양한 프롬프트에 대한 일반화(generalizability)를 촉진하기 위한 knowledge-preservation loss, (3) 효율적인 훈련을 위한 SNR-weighted sampling 기법을 포함합니다.

- **Performance Highlights**: 제안된 방법은 단 하나의 참조 이미지만을 사용하여 다양한 텍스트 프롬프트에 대해 높은 품질의 다양한 이미지를 생성할 수 있음이 실험을 통해 입증되었습니다. 메모리와 저장 공간 요구 사항이 최소화되어 실제 적용 가능성이 높아집니다.



### Style Based Clustering of Visual Artworks (https://arxiv.org/abs/2409.08245)
Comments:
          29 pages

- **What's New**: 이 논문은 예술 작품의 스타일 기반 클러스터링(style-based clustering)이라는 다루어지지 않은 문제를 해결하기 위해 새로운 방법을 제안합니다. 이는 인공지능(AI) 기법을 활용하여 예술 작품의 스타일적 표현의 다양성을 탐구하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구진은 세 가지 스타일 기반 특징(feature representations) 유형을 조사하였습니다: (i) Neural Network 기반 스타일 특징, (ii) StyleGAN 기반 스타일 특징, (iii) 텍스트 기반 스타일 특징입니다. 클러스터링 방법론으로는 Deep Embedded Clustering (DEC) 모델을 채택하여, 각 작품의 특징 및 클러스터 할당을 동시에 학습합니다.

- **Performance Highlights**: 논문에서 제안한 방법은 다양한 데이터셋을 통해 질적 및 양적 분석을 시행하여 스타일 기반 클러스터링의 유효성을 입증했습니다. 이 연구는 예술 스타일의 진화를 이해하는 데 도움이 되는 중요한 통찰을 제공합니다.



### IFAdapter: Instance Feature Control for Grounded Text-to-Image Generation (https://arxiv.org/abs/2409.08240)
- **What's New**: 이번 연구는 Instance Feature Generation (IFG) 작업을 처음으로 도입하여 텍스트에서 이미지로의 생성 모델의 포지셔닝과 기능 생성을 동시에 개선하고자 합니다. IFG 작업에 대한 해결책으로 Instance Feature Adapter (IFAdapter)를 제안하여 이미지 생성의 세밀함을 향상시킵니다.

- **Technical Details**: IFAdapter는 새로운 appearance tokens와 Instance Semantic Map을 도입하여 인스턴스 기능의 정확성을 보장합니다. Appearance tokens는 인스턴스별 특성 정보를 추출하고, 2D semantic map은 인스턴스 기능과 공간적 위치를 정렬하는 데 사용됩니다. 이 장치는 기존의 확산 모델에 plug-and-play 방식으로 통합될 수 있습니다. 실험에서는 COCO-IFG 벤치마크와 검증 파이프라인을 통해 객관적인 비교를 수행하였습니다.

- **Performance Highlights**: 실험 결과 IFAdapter는 기존 모델에 비해 수치적 및 질적 평가 모두에서 우수한 성능을 보여줍니다. 특히 IFAdapter는 정확한 포지셔닝과 기능 생성을 동시에 달성하여 다양한 생성 모델에 적용 가능성이 높습니다.



### LT3SD: Latent Trees for 3D Scene Diffusion (https://arxiv.org/abs/2409.08215)
Comments:
          Project page: this https URL Video: this https URL

- **What's New**: LT3SD는 고품질의 대규모 3D 장면 생성을 위한 새로운 latent diffusion 모델입니다. 이 모델은 복잡하고 다양한 구조의 3D 장면 생성을 위해 새로운 latent tree representation을 도입하여 lower-frequency geometry와 higher-frequency detail을 효과적으로 인코딩합니다.

- **Technical Details**: LT3SD는 coarse-to-fine hierarchy를 사용하여 3D 장면을 트리 구조로 분해하며, 각 해상도에서 장면의 latent 구성 요소를 모델링합니다. 이를 통해, scene patches를 기반으로 diffusion 모델을 훈련시켜 임의 크기의 3D 장면을 생성합니다. 이 접근법은 복잡한 비정렬 3D 장면 대신 높은 유사성을 가진 로컬 구조에 학습 초점을 이동시킵니다.

- **Performance Highlights**: 실험 결과, LT3SD는 기존 모델보다 70% 개선된 FID 점수를 기록하며, 고품질 무조건적 3D 장면 생성을 위한 효율성과 장점을 입증했습니다.



### VI3DRM:Towards meticulous 3D Reconstruction from Sparse Views via Photo-Realistic Novel View Synthesis (https://arxiv.org/abs/2409.08207)
- **What's New**: 최근 3D 재구성 분야에서 VI3DRM(Visual Isotropy 3D Reconstruction Model)이라는 새로운 모델이 소개되었습니다. 이 모델은 아이디어 일관성과 관점이 분리된 3D 잠재 공간 내에서 작동하며, 기존의 방식보다 더 사실적인 이미지를 생성할 수 있음을 보여줍니다.

- **Technical Details**: VI3DRM은 기존의 단일 보기 기반 방법의 한계를 극복하기 위해 설계된 모델로, 색상, 소재 속성 및 조명 같은 의미 정보를 분리하는 데 중점을 두고 있습니다. 이 모델은 pseudo cross-attention 방식을 통해 서로 다른 관점 간의 정보 교환을 용이하게 합니다. GSO 데이터셋에서의 NVS(Novel View Synthesis) 작업에서 PSNR 38.61, SSIM 0.929, LPIPS 0.027이라는 뛰어난 성능을 기록했습니다.

- **Performance Highlights**: VI3DRM은 DreamComposer 모델에 비해 구조적 일관성과 텍스처 세부 사항 모두에서 우수한 성능을 보이며, 이전 결과에서 상당한 향상이 있었습니다. 60초 만에 고품질의 텍스처가 잘 입혀진 메쉬를 생성할 수 있는 능력도 강조되었습니다.



### ComAlign: Compositional Alignment in Vision-Language Models (https://arxiv.org/abs/2409.08206)
- **What's New**: 본 논문에서는 Vision-Language Models (VLMs)인 CLIP의 훈련 과정에서 발생하는 조합적 구조(compositional structure) 이해의 한계를 극복하기 위한 Compositional Alignment (ComAlign)라는 새로운 접근법을 제안합니다. 이 방법은 이미지와 텍스트의 구성 요소 간의 정확한 대응 관계를 발견하기 위해 약한 감독(weak supervision)인 이미지-텍스트 쌍만을 활용합니다.

- **Technical Details**: ComAlign는 이미지 및 텍스트 인코더 위에 가벼운 네트워크를 훈련하여 이미지와 텍스트 모달리티의 노드와 엣지를 정렬하는 방법입니다. 이 과정에서 텍스트 모달에서 추출한 개체(entity)와 관계(relations)를 이미지 모달에서도 유지함으로써 정밀한 개념 대응(correspondence)을 강화합니다. 네트워크는 기존 VLM의 기능을 최대한 활용하여 보다 세분화된 VLM을 제공합니다.

- **Performance Highlights**: 실험 결과, ComAlign를 적용한 다양한 VLM과 데이터셋에서 검색(retrieval) 및 조합적 벤치마크(compositional benchmarks)에서 유의미한 성능 향상을 보여주었습니다. 예를 들어, CLIP-ViT-B32의 MSCOCO 데이터셋에서 I2T 검색 성능이 5.60%, T2I 검색 성능이 6.27% 향상되었습니다.



### What Makes a Maze Look Like a Maze? (https://arxiv.org/abs/2409.08202)
- **What's New**: 본 논문에서는 Deep Schema Grounding (DSG)이라는 새로운 프레임워크를 도입하여 시각적 추상 개념을 이해하는 데 도움을 주는 방법을 제안합니다. DSG는 시각적 추상화를 위한 명시적 구조화된 표현을 활용하여 추론합니다.

- **Technical Details**: DSG의 핵심 요소는 schema로, 이는 추상 개념을 보다 원시적인 기호로 분해하는 의존성 그래프(Dependency Graph) 설명입니다. DSG는 대형 언어 모델(LLM)을 사용하여 schema를 추출하고, 이를 비전-언어 모델(Vision-Language Model, VLM)과 계층적으로 결합하여 이미지에 대해 추상 개념의 구체적인 구성 요소를 접지합니다.

- **Performance Highlights**: DSG는 비전-언어 모델의 추상 시각 추론 성능을 유의미하게 향상시켰습니다. 특히, GPT-4V의 경우 8.3%의 상대적 개선을 보였고, Counting 관련 질문에서는 11.0%의 상대적 개선을 달성했습니다.



### Gaussian Garments: Reconstructing Simulation-Ready Clothing with Photorealistic Appearance from Multi-View Video (https://arxiv.org/abs/2409.08189)
- **What's New**: 본 논문에서는 다중 뷰 비디오에서 사실적이고 시뮬레이션 준비가 된 의류 자산을 재구성하는 새로운 방법인 Gaussian Garments를 소개합니다. 이 방법은 3D 메시(mesh)와 색상 및 고주파(surface detail)를 인코딩한 Gaussian 텍스처를 결합하여 의류를 표현합니다. 이를 통해 다양한 비디오에 대한 의류 기하학의 정확한 등록과 조명을 고려한 알베도 텍스처의 분리가 가능해집니다.

- **Technical Details**: Gaussian Garments는 3D 메시와 Gaussian 기반의 외관 모델링을 결합하여 동작합니다. 다중 뷰 이미지에서 초기 의류 메시를 획득하고, Gaussian splatting 기반의 포토메트릭 최적화 기법을 통해 이를 다중 뷰 비디오에 등록합니다. 이후 Gaussian 텍스처를 최적화하여 의류의 세부 외관을 복구하고, 등록된 메시를 사용해 그래프 신경망(GNN)을 미세 조정하여 의류의 실제 행동을 모방합니다.

- **Performance Highlights**: Gaussian Garments는 기존의 방법들보다 더 세밀한 3D 의류 자산을 재구성할 수 있으며, 여러 의상 아이템을 조합하여 복합적인 의상으로 생성할 수 있습니다. 또한, 미세 조정된 GNN을 활용하여 동적인 행동을 사실적으로 모델링할 수 있습니다.



### Enhancing Canine Musculoskeletal Diagnoses: Leveraging Synthetic Image Data for Pre-Training AI-Models on Visual Documentations (https://arxiv.org/abs/2409.08181)
- **What's New**: 본 연구는 개의 근골격계( musculoskeletal system )를 평가하기 위한 새로운 방법을 개발하였습니다. 특히, AI 기반의 진단 지원 시스템을 위해 시각적 자료의 데이터 부족 문제를 극복하고자 하는 시도를 포함합니다.

- **Technical Details**: 본 연구에서 생성된 합성 데이터( synthetic data )는 질병의 현실적인 시각적 문서화를 모방하는 이미지 데이터를 포함합니다. 초기에는 세 가지 클래스를 갖는 기본 데이터셋이 생성되었고, 이어서 36개 클래스를 포함하는 더 복잡한 데이터셋이 만들어졌습니다. 이 데이터셋들은 AI 모델의 사전 훈련( pre-training )에 사용되었습니다.

- **Performance Highlights**: 평가 데이터셋에서 25개의 예시를 사용한 결과, 실제 시각적 문서화를 모방한 생성된 합성 이미지를 활용할 경우 진단 정확도( diagnosis accuracy )가 약 10% 향상되는 것으로 나타났습니다. 그러나 250개의 예제가 포함된 더 큰 평가 데이터셋에서는 이러한 결과가 나타나지 않았습니다. 이는 합성 데이터를 활용한 AI 모델의 사전 훈련의 이점이 주로 질병에 대한 적은 수의 시각적 문서화에 적용될 수 있다는 것을 보여줍니다.



### Low-Cost Tree Crown Dieback Estimation Using Deep Learning-Based Segmentation (https://arxiv.org/abs/2409.08171)
Comments:
          16 pages, 5 figures

- **What's New**: 이번 연구는 저비용 드론 기술과 딥러닝을 활용하여 리다르(LiDAR)와 같은 고가의 장비 없이 RGB 항공 데이터를 기반으로 수관 고사(crown dieback) 평가를 수행합니다. 이는 전통적인 모니터링 기술의 한계를 극복하고, 작물 문제를 보다 효과적으로 감지할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 딥러닝(deep learning)과 식생 지수(vegetation indices)를 기반으로 한 접근 방법을 사용하여, 지중해 생태계에서의 가뭄으로 인한 수관 고사를 평가합니다. Mask R-CNN 모델을 활용하여 예측된 수관 발자국을 현장 조사 데이터와 맞추는 반복적인 방법을 사용합니다. 이 과정에서 전체 세분화 정확도(mAP: 0.519)를 기록하며, 이는 비전문가도 활용할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 자동화된 데이터 수집 및 처리의 잠재력을 보여주며, 딥러닝을 통해 삼림 고사 모니터링의 범위, 속도 및 비용을 개선할 수 있는 가능성을 입증합니다. 또한, 색상 좌표 기반의 고사 추정치가 전문가의 현장 추정치와 잘 상관관계가 있음을 발견하였으며, Mask R-CNN 모델 예측의 대체가 고사 추정치에 미치는 영향은 미미하다는 것을 보였습니다.



### Learning to Match 2D Keypoints Across Preoperative MR and Intraoperative Ultrasound (https://arxiv.org/abs/2409.08169)
Comments:
          Accepted for publication at the International Workshop of Advances in Simplifying Medical UltraSound (ASMUS) at MICCAI 2024

- **What's New**: 이 논문에서는 수술 전 Magnetic Resonance (MR) 이미지와 수술 중 Ultrasound (US) 이미지를 매칭하기 위해 특별히 설계된 texture-invariant 2D keypoints descriptor를 제안합니다. 특히, 다양한 MR 영상 모달리티와 수술 중 US 변동성을 고려한 매칭-by-synthesis 전략을 도입하였습니다.

- **Technical Details**: 이 방법은 수술 중 US 이미지를 수술 전 MR 이미지에서 합성하는 방식으로, 환자 개별적인 MR 이미지를 사용하여 cross-modality descriptor 네트워크를 훈련합니다. 이 네트워크는 supervised contrastive 방식으로 훈련되어 US texture 변화에 무관하게 강건한 keypoints descriptor를 학습합니다. 데이터셋은 자동으로 생성되며, 주석이 달린 key points가 필요 없습니다.

- **Performance Highlights**: 실제 사례에 대한 실험에서 제안된 방법이 최신 기술 대비 우수한 성능을 보이며, 평균 80.35%의 매칭 정확도를 달성함을 보여주었습니다.



### High-Frequency Anti-DreamBooth: Robust Defense Against Image Synthesis (https://arxiv.org/abs/2409.08167)
Comments:
          ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond

- **What's New**: 이 논문에서는 텍스트-이미지 생성 모델의 악용 문제를 다룬 새로운 적대적 공격 방법을 제안합니다. 기존의 Anti-DreamBooth와 같은 방법이 악의적인 생성으로부터 이미지를 보호하지만, 적대적 정화 기법에 의해 쉽게 제거될 수 있는 점을 지적하고, 강력한 섭동을 이미지의 고주파 영역에 추가하여 더욱 견고한 방어를 실현합니다.

- **Technical Details**: 제안된 방법은 3×3 Laplacian 필터를 사용하여 이미지의 엣지를 추출하고, 해당 엣지 마스크를 생성하여 고주파 영역에 강력한 적대적 노이즈를 추가합니다. 이 과정에서 ASPL (Alternating Surrogate and Perturbation Learning) 기법을 사용하여, 대체 모델을 학습시키고 적대적 섭동을 최적화하는 방식으로 구성됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방식으로 생성된 적대적 이미지가 적대적 정화 이후에도 여전히 노이즈를 유지하여, 악의적인 이미지 생성을 방해하는 성과를 보여주었습니다.



### Cross-Attention Based Influence Model for Manual and Nonmanual Sign Language Analysis (https://arxiv.org/abs/2409.08162)
- **What's New**: 이 연구는 American Sign Language (ASL)의 Sign Language Translation (SLT)에서 수동 마커(manual markers)뿐만 아니라 비수동 마커(non-manual markers)인 얼굴 표정의 기여를 분석하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 신경망 구조는 두 개의 스트림 인코더로 구성되어 있으며, 하나는 얼굴을 처리하고 다른 하나는 손이 포함된 상체를 처리합니다. 우리는 새로운 병렬 교차 주의 해독 매커니즘(parallel cross-attention decoding mechanism)을 제안하여 각 입력 모달리티가 출력에 미치는 영향을 정량화합니다.

- **Performance Highlights**: RWTH-PHOENIX-Weather2014T 데이터셋과 실제 ASLing 데이터셋을 사용하여 제안된 방법을 평가하였으며, 이 연구의 결과는 Sign Language Translation에서 수동 및 비수동 요소의 기여 정도를 관한 중요한 통찰력을 제공합니다.



### SDformer: Efficient End-to-End Transformer for Depth Completion (https://arxiv.org/abs/2409.08159)
Comments:
          Presented at the International Conference on Industrial Automation, Robotics and Control Engineering (IARCE) 2022

- **What's New**: 이번 논문에서는 기존의 CNN 기반 깊이 보완(dense depth completion) 모델의 한계를 극복하기 위한 새로운 접근법으로 Sparse-to-Dense Transformer(SDformer)를 제안합니다. 이 모델은 depth 센서로부터 얻은 희소한 깊이 측정값을 가지고 밀집 깊이 맵을 예측하는 데 중점을 두고 있으며, Transformer 구조를 활용하여 더 효과적인 특성을 추출합니다.

- **Technical Details**: SDformer는 입력 모듈(Input module)에서 RGB 이미지 및 깊이 맵 특성을 추출하고 결합하여 U자형 인코더-디코더 Transformer를 통해 깊이 특성을 처리합니다.이 구조는 Different Window-based Multi-Scale Self-Attention(DWSA) 및 Gated Feed-Forward Network(GFFN)으로 구성되어 있습니다. 각 계층에서는 특징을 샘플링하고 업샘플링하는 절차를 통해 깊이 종속성을 캡처합니다.

- **Performance Highlights**: SDformer는 NYU Depth V2 및 KITTI DC 데이터셋에서 CNN 기반 깊이 보완 모델과 비교하여 더 낮은 계산 부하와 더 적은 파라미터로 최고 수준의 성과를 달성하였습니다.



### MagicStyle: Portrait Stylization Based on Reference Imag (https://arxiv.org/abs/2409.08156)
- **What's New**: 이번 논문에서는 초상화 이미지의 스타일화를 위한 새로운 확산 모델 기반 접근법인 MagicStyle을 제안합니다.

- **Technical Details**: MagicStyle은 두 가지 주요 단계, 즉 Content and Style DDIM Inversion (CSDI)와 Feature Fusion Forward (FFF)로 구성됩니다. CSDI 단계에서는 컨텐츠 이미지와 스타일 이미지에서 각기 별도로 DDIM Inversion을 수행하며, 이 과정에서 두 이미지의 self-attention query, key, value 기능을 저장합니다. FFF 단계에서는 저장된 기능을 바탕으로 텍스처와 색상 정보를 통합하여 확산 생성 과정에서 조화롭게 결합합니다.

- **Performance Highlights**: 제안된 MagicStyle과 Well-designed Feature Fusion Attention (FFA)의 효과성을 검증하기 위해 포괄적인 비교 및 소거 실험을 수행했습니다.



### Bayesian Self-Training for Semi-Supervised 3D Segmentation (https://arxiv.org/abs/2409.08102)
Comments:
          Accepted at ECCV 2024

- **What's New**: 본 연구에서는 Bayesian 딥러닝을 기반으로 하는 반지도 학습을 위한 새로운 자가 학습 프레임워크를 제안합니다. 이를 통해 소량의 레이블된 데이터와 대량의 비레이블 데이터를 효과적으로 활용하여 3D 분할 문제를 해결합니다.

- **Technical Details**: 연구의 핵심은 모노 샘플링과 드롭아웃 기반의 몬테 카를로 적분을 통해 얻은 불확실성 추정치를 이용하여 신뢰성 있는 의사 레이블을 생성하는 것입니다. 이 과정에서 Shannon 엔트로피를 사용하여 불확실한 점들을 필터링합니다.

- **Performance Highlights**: 제안된 방법은 SemanticKITTI와 ScribbleKITTI에서 3D 의미적 분할에 대한 최신 결과를, ScanNet과 S3DIS에서 3D 인스턴스 분할의 경우에도 뛰어난 성능을 보여주었습니다. 또한 ScanRefer에서 감독 학습만 수행한 기존 방법들에 비해 실질적인 개선을 달성했습니다.



### EZIGen: Enhancing zero-shot subject-driven image generation with precise subject encoding and decoupled guidanc (https://arxiv.org/abs/2409.08091)
- **What's New**: 본 연구에서는 이미지 생성 시 객체의 정체성을 유지하면서 텍스트 지침에 맞춰 생성하는 새로운 접근 방식인 EZIGen을 소개합니다. EZIGen은 고유한 주제 이미지 인코더 설계를 통해 정체성 보존 품질을 크게 향상시킵니다.

- **Technical Details**: EZIGen은 UNet 아키텍처 기반의 주제 이미지 인코더를 활용하여 주제와 텍스트 간의 정합성을 유지하고, 초기 레이아웃 생성을 통해 정체성 보존과 텍스트 정렬 간의 균형을 이룹니다. 이 과정은 Layout Generation Process와 Appearance Transfer Process로 나누어져 진행됩니다.

- **Performance Highlights**: EZIGen은 다양한 주제 기반 벤치마크에서 최신 기술로 뛰어난 성능을 달성하며, 단 100배 적은 훈련 데이터로도 유니파이드 모델로서 우수한 결과를 보여줍니다.



### SimMAT: Exploring Transferability from Vision Foundation Models to Any Image Modality (https://arxiv.org/abs/2409.08083)
Comments:
          Github link: this https URL

- **What's New**: 본 논문에서는 SimMAT이라는 단순하면서도 효과적인 프레임워크를 제시하여 자연 RGB 이미지로 훈련된 비전 기초 모델의 다른 이미지 모달리티로의 전이 가능성을 탐구하고 있습니다. 이 프레임워크는 모달리티에 구애받지 않는 전이 레이어(MAT)와 사전 훈련된 기초 모델로 구성됩니다.

- **Technical Details**: SimMAT는 다양한 모달리티를 입력으로 허용하며, 이미지 간의 관계에 대한 도메인 지식이 필요하지 않습니다. 연구에서는 SAM(Segment Anything Model)에 SimMAT를 적용하여 다양한 이미지 모달리티에서 분할 성능을 향상시킵니다. 매개변수 효율적인 세밀 조정 전략(PEFT)을 비교하고, LoRA, MLP Adapter, prompt tuning 등의 최적 성능을 분석합니다.

- **Performance Highlights**: SimMAT는 평가된 모달리티에 대해 평균적으로 분할 성능(mIoU)을 22.15%에서 53.88%로 향상시키며 다른 기본선 모델들보다 일관되게 뛰어난 성과를 보였습니다. 이는 뚜렷한 물리적 속성을 갖는 다양한 센서의 성능을 향상시키는 잠재력을 확인시켜 줍니다.



### Diffusion-Based Image-to-Image Translation by Noise Correction via Prompt Interpolation (https://arxiv.org/abs/2409.08077)
Comments:
          16 pages, 5 figures, 6 tables

- **What's New**: 본 논문에서는 diffusion 기반의 이미지-이미지 변환을 위한 간단하면서도 효과적인 훈련 없는 접근 방식을제안합니다. 이 방법은 사전 훈련된 diffusion 모델의 원래 noise prediction network를 수정하여 noise correction term을 도입합니다.

- **Technical Details**: 제안하는 방법은 두 가지 noise 예측의 차이를 noise correction term으로 정의합니다. 첫 번째 noise 예측은 denoising network에서 소스와 타겟 프롬프트 임베딩을 점진적으로 보간하여 계산하고, 두 번째는 소스 프롬프트 임베딩에 기반한 noise 예측입니다. 최종 noise 예측 네트워크는 표준 denoising term과 noise correction term의 선형 조합으로 구성되며, 이는 유지해야 할 영역을 재구성하는 동시에 관심 있는 영역을 효과적으로 수정하는 역할을 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 낮은 대기 시간으로 뛰어난 성능을 보여주며 기존 프레임워크에 통합할 때 지속적으로 개선된 성능을 나타냅니다.



### Expansive Supervision for Neural Radiance Field (https://arxiv.org/abs/2409.08056)
Comments:
          12 pages, 7 figures

- **What's New**: 본 논문에서는 Neural Radiance Fields (NeRF) 훈련에서 컴퓨팅 부담을 효과적으로 줄이는 새로운 방법인 "expansive supervision" 메커니즘을 도입합니다. 이 방법은 시각 품질과 유연성을 균형있게 유지하며 메모리 및 시간 소모를 크게 줄이는 데 기여합니다.

- **Technical Details**: expansive supervision 메커니즘은 선택적으로 작은 픽셀 집합을 렌더링하고, 이 값을 확장하여 전체 영역의 에러를 추정합니다. 이는 전통적인 감독 방식보다 불필요한 렌더링 과정을 우회하여 훈련 속도를 높이고 69%의 메모리 절약과 42%의 시간 절약을 달성합니다. 또한, content-aware permutation 기법을 통해 이미지 컨텐츠와 훈련 에러 간의 강한 상관관계를 활용합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존 고급 가속 프레임워크에 통합되었을 때 시각 품질을 거의 손상없이 69%의 메모리 절약과 42%의 시간 절약을 기록했습니다. 이는 NeRF의 훈련 속도를 비약적으로 향상시키며, 에러 배급과 이미지 컨텐츠의 관계를 이용한 최초의 접근 방식이라는 점에서 의미가 있습니다.



### Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis (https://arxiv.org/abs/2409.08042)
Comments:
          17 pages, 4 figures, 3 tables

- **What's New**: 이 논문은 열적 적외선 영상에서 새로운 시각의 합성을 위한 물리 기반 3D Gaussian splatting 방법인 Thermal3D-GS를 소개합니다. 이 방법은 대기 전파 및 열전도와 같은 물리적 특성을 모델링하고 이를 최적화 목표에 통합하여 열적 적외선 이미지를 개선하는 데 초점을 맞추고 있습니다. 또한, TI-NSD라는 대규모 벤치마크 데이터셋을 구축하여 최초의 열적 적외선 새로운 시각 합성을 위한 기반을 마련했습니다.

- **Technical Details**: Thermal3D-GS는 심층 신경망을 사용하여 대기 전파 현상과 열전도를 시뮬레이션합니다. 이 과정에서 이미지를 합성할 때 발생하는 부유물(floater) 및 불명확한 경계 특징을 개선하기 위해 물리적 제약조건인 온도 일관성 제약을 도입했습니다. 이렇게 모델링된 특성을 통해 더 정확한 열적 적외선 이미지 복원이 가능해졌습니다.

- **Performance Highlights**: 실험 결과, Thermal3D-GS는 표준 기준 방법인 3D-GS에 비해 평균 PSNR이 3.03 dB 향상되었으며, 부유물 문제와 불명확한 경계 특징을 효과적으로 해결했습니다. 이 새로운 방법의 비디오 프레임에서 도출된 결과는 기존 방법들보다 월등한 비주얼 성능을 보여줍니다.



### LED: Light Enhanced Depth Estimation at Nigh (https://arxiv.org/abs/2409.08031)
Comments:
          Preprint. Code and dataset available at this https URL

- **What's New**: 이번 연구에서는 Light Enhanced Depth (LED)라는 새로운 심층학습 방식을 도입하여 저조도 환경에서의 깊이 추정 성능을 획기적으로 개선합니다. 이를 위해 현대 차량에 장착된 고해상도 헤드라이트가 프로젝터 역할을 하여 장면에 패턴을 투사하도록 활용하였습니다.

- **Technical Details**: LED는 HD 헤드라이트에서 투사된 체크무늬 패턴을 사용하여 깊이 추정 모델의 성능을 높입니다. 이 방식은 여러 깊이 추정 아키텍처(encoder-decoder, Adabins, DepthFormer)에 적용할 수 있으며, 아키텍처에 종속되지 않고 다양한 모델에서 개선된 성능을 보여줍니다. 깊이 추정은 카메라와 투사된 패턴 간의 차이를 이용하여 이루어집니다.

- **Performance Highlights**: LED 기법은 야간에 깊이 추정을 수행할 때 -11% RMSE 향상이라는 눈에 띄는 결과를 보여주었으며, Adabins 및 DepthFormer의 경우 각각 -24.06% 및 -8.00% RMSE 개선을 이루었습니다. 또한, 조명된 지역을 넘어 전반적인 장면 이해 능력도 증가함을 입증하였습니다.



### Scribble-Guided Diffusion for Training-free Text-to-Image Generation (https://arxiv.org/abs/2409.08026)
- **What's New**: 이번 논문에서는 Text-to-Image Diffusion (텍스트-이미지 확산) 모델의 새로운 접근 방식인 Scribble-Guided Diffusion (ScribbleDiff)를 제안합니다. 사용자 제공 스크리블을 비주얼 프롬프트로 활용하여 이미지 생성을 유도하는 방식입니다.

- **Technical Details**: Scribble-Guided Diffusion는 기존의 텍스트 입력과 바운딩 박스 또는 영역 마스크를 사용하는 방법의 한계를 극복하기 위해 개발되었습니다. 스크리블을 확산 모델에 효과적으로 통합하기 위해 Moment Alignment와 Scribble Propagation 기법을 도입하여 생성된 이미지와 스크리블 간의 정렬을 개선했습니다.

- **Performance Highlights**: PASCAL-Scribble 데이터셋을 이용한 실험 결과, 공간적 제어(Spatial Control)와 일관성(Consistency)에서 현저한 개선이 나타났습니다. 스크리블 기반 가이드를 통해 향상된 효과를 확인할 수 있었습니다.



### Depth Matters: Exploring Deep Interactions of RGB-D for Semantic Segmentation in Traffic Scenes (https://arxiv.org/abs/2409.07995)
- **What's New**: 새로운 학습 가능 Depth interaction Pyramid Transformer (DiPFormer)를 제안하여 깊이 정보의 효과성을 탐구합니다. 기존의 방법들이 깊이 정보를 보조적인 요소로 간주한 것과 달리, 이 연구는 깊이 맵의 내재적 공간 속성을 강조합니다.

- **Technical Details**: DiPFormer는 Depth Spatial-Aware Optimization (Depth SAO)과 Depth Linear Cross-Attention (Depth LCA) 모듈을 포함하고 있습니다. Depth SAO는 깊이의 위치 정보를 통해 공간 관계를 표현하고, Depth LCA는 RGB-D의 피쳐 공간에서 유사성을 학습하여 픽셀 수준에서 공간적 차이를 분명히 합니다. 마지막으로, 다중 스케일 피쳐를 효과적으로 융합하기 위해 MLP 디코더를 사용합니다.

- **Performance Highlights**: 제안된 DiPFormer는 KITTI 및 Cityscapes 데이터셋에서 기존 기술(AI state-of-the-art)보다 우수한 성능을 보여줍니다. KITTI 도로에서 97.57%의 F-score, KITTI-360 및 Cityscapes에서 각각 68.74% 및 83.4%의 mIoU를 기록했습니다. 또한, 깊이 정보를 활용함으로써 도로 탐지에서 7.5%, 의미 분할에서 4.9% 및 1.5%의 성능 향상을 달성했습니다.



### Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms (https://arxiv.org/abs/2409.07989)
- **What's New**: 본 논문은 기존의 metric-based 방법들이 지니고 있는 한계를 극복하기 위한 새로운 접근법을 제안합니다. 이 방법은 다양한 feature 공간을 통해 샘플을 매핑하는 multi-output embedding 네트워크를 활용하여, 노드의 서로 다른 단계에서 feature 벡터를 추출하고 이를 통해 전역(global) 및 추상(abstract) feature를 모두 캡처하는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 ResNet18을 feature extractor로 사용하여, 다섯 개의 서로 다른 단계에서 feature map을 추출합니다. 각 단계마다 learnable parameter weights를 도입하여 feature representation 능력을 향상시키고, self-attention 메커니즘을 통해 각 feature map의 정보를 풍부하게 합니다. 이러한 구성 요소들은 모델 성능을 크게 향상시키는데 기여합니다.

- **Performance Highlights**: MiniImageNet 및 FC100 데이터셋에서 5-way 1-shot 및 5-way 5-shot 시나리오에 대해 종합적으로 평가한 결과, 제안된 방법이 최신 기법들에 비해 우수한 성능을 보였으며, MiniImageNet 데이터셋에서 CUB 데이터셋으로의 cross-domain 작업에서도 높은 정확도를 기록했습니다.



### SPARK: Self-supervised Personalized Real-time Monocular Face Captur (https://arxiv.org/abs/2409.07984)
Comments:
          SIGGRAPH Asia 2024 Conference Paper. Project page: this https URL

- **What's New**: 이 논문에서는 비제한 비디오 컬렉션을 활용하여 고정밀 3D 얼굴 캡처 방법을 제안합니다. 기존의 방법들이 제공하는 대략적인 3D 얼굴 모델의 한계를 극복하고, 개인화된 3D 모델을 실시간으로 추정할 수 있는 새로운 접근 방식입니다.

- **Technical Details**: 제안된 SPARK 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 여러 비디오로부터 상세한 3D 얼굴 아바타를 구축하고, 두 번째 단계에서는 사전 훈련된 모노큘러(monocular) 얼굴 재구성 방법의 인코더를 활용하고 제어기에 맞춰 개인화된 모델로 대체하여 전이 학습(transfer learning)을 진행합니다. 이를 통해 다양한 조명 및 포즈를 갖는 미지의 이미지에서 실시간으로 정확한 포즈 및 표정 매개변수를 회귀(regress)할 수 있는 인코더를 제공합니다.

- **Performance Highlights**: 경량화된 실험 결과, 제안된 방법이 미지의 조명, 포즈 및 표정에서도 성공적으로 일반화된다는 것을 보여줍니다. 제안한 방법은 비슷한 추론 시간을 가진 경쟁 기법들에 비해 더 정확한 포즈 및 표정 추정과 더 풍부한 메쉬 추정을 가능하게 합니다.



### Sparse R-CNN OBB: Ship Target Detection in SAR Images Based on Oriented Sparse Proposals (https://arxiv.org/abs/2409.07973)
Comments:
          Submitted to 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)

- **What's New**: Sparse R-CNN OBB는 SAR 이미지를 위한 새로운 방향성 객체 탐지 프레임워크로, 300개의 희소 학습 가능 제안을 활용하여 높은 성능을 달성합니다.

- **Technical Details**: Sparse R-CNN OBB는 ResNet-50 백본과 Feature Pyramid Network (FPN)를 사용하여 다중 스케일 특징 융합을 수행합니다. 이 모델은 회전된 제안을 위해 두 가지 주요 조정 사항이 있습니다: RoI 풀링 단계에서 표준 RoIAlign 대신 회전 RoIAlign (R-RoIAlign)을 사용합니다.

- **Performance Highlights**: Sparse R-CNN OBB는 내부 및 외부 시나리오에서 다른 최신 모델들을 초월하는 뛰어난 성능을 보여줍니다.



### Deep Height Decoupling for Precise Vision-based 3D Occupancy Prediction (https://arxiv.org/abs/2409.07972)
- **What's New**: 본 논문에서는 처음으로 Deep Height Decoupling (DHD)이라는 새로운 프레임워크를 제안합니다. DHD는 명시적인 높이 정보를 활용하여 3D 특징을 분리하고 혼란스러운 특징을 필터링합니다.

- **Technical Details**: DHD는 LiDAR 신호를 통해 높이 맵을 생성한 후, Mask Guided Height Sampling (MGHS) 모듈을 통해 다양한 높이에서 특징을 샘플링합니다. MGHS는 여러 개의 높이 마스크를 생성하여 필터링 작업을 수행하고, Synergistic Feature Aggregation (SFA) 모듈을 통해 채널과 공간 유사성을 활용하여 3D 특징 표현을 향상시킵니다.

- **Performance Highlights**: DHD는 Occ3D-nuScenes 벤치마크에서 최첨단 성능을 달성하며, 최소한의 입력 프레임으로도 우수한 결과를 나타냅니다. 공개된 코드는 동료 연구자들이 활용할 수 있도록 제공됩니다.



### Locality-aware Cross-modal Correspondence Learning for Dense Audio-Visual Events Localization (https://arxiv.org/abs/2409.07967)
- **What's New**: 이번 연구에서는 LOCO라는 새로운 프레임워크를 제안하여 Dense-localization Audio-Visual Events (DAVE) 문제를 해결하고자 합니다. LOCO는 로컬 시간 연속성을 탐색하여 오디오-비주얼 이벤트의 처리에서 불필요한 정보를 필터링하는 데 도움을 줍니다.

- **Technical Details**: LOCO는 두 가지 주요 구성 요소로 이루어져 있습니다: 1) Locality-aware Correspondence Correction (LCC), 이는 단일 모드 특징에 대한 교차 모드 로컬 상관 속성을 활용하여 불필요한 정보를 자동으로 필터링합니다. 2) Cross-modal Dynamic Perception (CDP), 이는 모달리티 간의 일관성을 유지하며, 오디오-비주얼 이벤트의 지역적 시간 패턴을 이해하기 위해 윈도우 기반 메커니즘을 적용합니다.

- **Performance Highlights**: 시뮬레이션 결과 LOCO는 기존 DAVE 방법들을 초월하여 모든 메트릭에서 성능 향상을 보였습니다. 특히 mAP@0.9에서 3.4% 향상을 기록하였으며, 이를 통해 LOCO가 오디오-비주얼 이벤트 감지 문제에 효과적임을 입증했습니다.



### ProbTalk3D: Non-Deterministic Emotion Controllable Speech-Driven 3D Facial Animation Synthesis Using VQ-VAE (https://arxiv.org/abs/2409.07966)
Comments:
          14 pages, 9 figures, 3 tables. Includes code. Accepted at ACM SIGGRAPH MIG 2024

- **What's New**: 이 논문에서는 감정 조절이 가능한 음성 기반 3D 얼굴 애니메이션 합성을 위한 비결정적(Non-Deterministic) 신경망 접근 방식을 제안합니다. 특히, 감정 데이터를 활용하여 다양한 감정 표현을 동시에 합성할 수 있는 모델을 개발했습니다.

- **Technical Details**: ProbTalk3D라는 모델은 두 단계의 VQ-VAE(Vektor Quantized Variational Autoencoder) 구조를 기반으로 하며, 3DMEAD라는 감정적으로 풍부한 얼굴 애니메이션 데이터셋을 사용하여 설계되었습니다. 이 모델은 감정 레이블과 강도를 통한 감정 조절 기능을 제공하는 최초의 비결정적 3D 얼굴 애니메이션 합성 방법입니다.

- **Performance Highlights**: 제안된 모델은 최신 감정 조절 모델들과 비교하여 뛰어난 성능을 보여주며, 특히 주관적인 평가에서는 사용자 연구를 통해 모델의 결과가 우수하다는 것을 입증했습니다. 또한, 모델의 코드는 공개적으로 사용 가능합니다.



### Estimating atmospheric variables from Digital Typhoon Satellite Images via Conditional Denoising Diffusion Models (https://arxiv.org/abs/2409.07961)
Comments:
          8 pages, 5 figures

- **What's New**: 본 연구는 확산 모델(diffusion models)을 활용하여 백서 (typhoons) 분야에서 디지털 태풍 (Digital Typhoon) 위성 이미지를 기반으로 여러 가지 ERA5 기상 변수들을 동시에 예측하는 방법을 탐구합니다. 타이완은 태풍에 매우 취약한 지역으로, 이 연구의 주요 초점이 되고 있습니다.

- **Technical Details**: Conditional Denoising Diffusion Probability Model (CDDPM)을 통해 타이완 지역의 기상 변수(u10, v10, sp, t2m)를 예측하였으며, 이 모델은 Convolutional Neural Networks (CNN) 및 Squeeze-and-Excitation Networks (SENet)와 비교하였습니다. CDDPM은 PSNR (Peak Signal-to-Noise Ratio) 32.807을 기록하며, CNN과 SENet에 비해 각각 7.9% 및 5.5% 높은 성능을 보였습니다. 또한 RMSE (Root Mean Square Error)는 0.032로, CNN보다 11.1% 개선되었고, SENet보다 8.6% 향상되었습니다.

- **Performance Highlights**: CDDPM은 여러 가지 재분석 변수들에서 가장 높은 PSNR 및 SSIM (Structural Similarity Index) 점수를 기록하며, 가장 낮은 FID (Fréchet Inception Distance) 및 LPIPS (Learned Perceptual Image Patch Similarity) 점수를 달성했습니다. 이 결과는 태풍 예측 및 기존 데이터셋의 결측치 보완에 활용 가능성이 높습니다.



### Do Vision Foundation Models Enhance Domain Generalization in Medical Image Segmentation? (https://arxiv.org/abs/2409.07960)
- **What's New**: 이 논문은 다양한 기초 모델(Foundational Models, FMs)의 의학 이미지 분할에서의 도메인 일반화 성능을 조사하고, HQHSAM이라는 새로운 디코드 헤드 아키텍처를 도입하여 성능 향상에 기여하고 있음을 밝힙니다.

- **Technical Details**: 연구에 사용된 기초 모델에는 DinoV2, SAM, MedSAM 및 MAE가 있으며, Ladder 및 Rein(+LoRA)와 같은 여러 파라미터 효율적 미세 조정 기법을 활용하였습니다. HQHSAM 디코드 헤드 아키텍처는 HSAM 및 HQSAM에서 요소를 통합하여 분할 성능을 향상시킵니다. 실험은 건강한 뇌, 전립선, 요추 및 뇌종양에 대한 다양한 데이터 세트를 포함하여 수행되었습니다.

- **Performance Highlights**: HQHSAM 디코드 헤드를 사용할 때 FMs의 도메인 일반화 성능이 향상되며, PEFT 기법의 효과는 모델에 따라 다르게 나타납니다. 이 연구는 의학 이미지 분할에서의 FMs의 잠재력을 강조하며, 결론적으로 다양한 임상 환경에서의 신경망 성능 향상을 위한 기초를 제공합니다.



### Control+Shift: Generating Controllable Distribution Shifts (https://arxiv.org/abs/2409.07940)
Comments:
          ECCV2024, "Synthetic Data for Computer Vision" workshop

- **What's New**: 새로운 디지털 데이터셋을 생성하는 방법을 제안하며, 이는 다양한 분포 변화(distribution shift)를 갖는 현실적인 이미지 데이터셋을 생성할 수 있습니다. 이러한 데이터셋은 모델 성능 저하를 분석하는 데 유용합니다.

- **Technical Details**: 본 연구에서는 decoder 기반 생성 모델을 활용하여, 훈련 데이터와의 부분적인 교집합을 가지는 분포 변화를 모델링합니다. 또한, 최근의 score-based 모델(score-based model, SBM)을 사용하여 사실적 이미지를 생성하며, 이 난해한 과정을 통해 조절 가능한 강도의 분포 변화를 갖는 데이터셋을 생성합니다.

- **Performance Highlights**: 다양한 분류기를 테스트한 결과, 데이터 증강(data augmentation)을 통해 강건성은 증가했지만 여전히 분포 변화에서 성능 저하를 겪었습니다. 또한 훈련 데이터의 크기를 늘리는 것만으로는 분포 변화에 대한 강건성을 충분히 얻을 수 없습니다. 마지막으로, 컨볼루션 네트워크(convolutional networks)가 다른 아키텍처보다 더 강건한 경향이 있음을 발견하여, 훈련 분포와의 지리적 거리에서 성능 저하가 선형적(linear)임을 관찰했습니다.



### Task-Augmented Cross-View Imputation Network for Partial Multi-View Incomplete Multi-Label Classification (https://arxiv.org/abs/2409.07931)
- **What's New**: 본 논문에서는 부분적인 다중 시각(multi-view) 비어 있는 다중 레이블(multi-label) 분류를 처리하기 위해 태스크 증강 교차 시각 보간 네트워크(TACVI-Net)를 제안합니다. 이 네트워크는 불완전한 교육 데이터의 문제를 해결하고 높은 태스크 관련 기능을 도출하여 누락된 뷰를 복구하는데 중점을 두고 있습니다.

- **Technical Details**: TACVI-Net은 두 단계로 구성된 네트워크로, 첫 번째 단계에서는 정보 병목 이론(information bottleneck theory)을 활용하여 각 뷰에 대한 식별 가능한 표현을 얻습니다. 두 번째 단계에서는 자동 인코더(autoencoder)를 기반으로 한 다중 뷰 재구성 네트워크를 사용하여 증강 특징의 고수준 의미 표현을 추출하고 누락된 데이터를 복구합니다.

- **Performance Highlights**: 다섯 개의 데이터셋에 대한 광범위한 실험을 통해 TACVI-Net이 다른 최첨단 방법들을 초월하는 성능을 보였다는 것을 입증했습니다. 제안된 모델은 유용한 정보를 활용하여 누락된 뷰를 효과적으로 복구할 수 있습니다.



### UGAD: Universal Generative AI Detector utilizing Frequency Fingerprints (https://arxiv.org/abs/2409.07913)
- **What's New**: 본 논문에서는 AI 생성 이미지와 진짜 이미지를 구분하기 위한 새로운 다중 모달 접근법인 UGAD를 제안합니다. 인공지능 기술의 발전으로 인해 가짜 이미지의 생성이 더욱 용이해졌으며, 이에 따라 실체 이미지와 생성된 이미지를 식별하는 것이 매우 중요해졌습니다.

- **Technical Details**: UGAD는 세 가지 핵심 단계로 구성되어 있습니다: 우선, RGB 이미지를 YCbCr 채널로 변환하고, 적분 방사형 연산(Integral Radial Operation, RIO)을 적용하여 두드러진 방사형 특징을 강조합니다. 두 번째로, 공간 푸리에 추출(Spatial Fourier Extraction, SFE) 작업을 통해 공간 이동을 수행하며, 이는 사전 훈련된 깊은 학습 네트워크를 활용하여 최적의 특징을 추출합니다. 마지막으로, 깊은 신경망 분류 단계에서 소프트맥스를 사용하여 데이터를 처리하고 분류합니다.

- **Performance Highlights**: 제안된 방법은 기존 최첨단 기법에 비해 정확도에서 12.64% 증가, AUC에서 28.43% 향상된 결과를 보여줍니다. 또한, 최신 AI 생성 방법에 의해 생성된 이미지에 대해 정밀한 탐지가 가능함을 입증하였습니다.



### From COCO to COCO-FP: A Deep Dive into Background False Positives for COCO Detectors (https://arxiv.org/abs/2409.07907)
- **What's New**: 이 연구에서는 COCO-FP라는 새로운 평가 데이터셋을 제시합니다. COCO-FP는 ImageNet-1K 데이터셋에서 파생되어, 객체 탐지기가 비목표 시각 잡음(visual clutter)으로 인한 false positive를 줄이는 능력을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: COCO-FP 데이터셋은 COCO 검증 데이터셋을 확장하여 비목표 객체(annotated categories에 포함되지 않는 배경 물체)로 인해 발생하는 false positive 문제를 해결하기 위해 설계되었습니다. 이 연구는 COCO-FP에서 YOLOv9-E의 AP50이 COCO에서 72.8에서 65.7로 감소하는 것을 보여줍니다.

- **Performance Highlights**: 기존 폐쇄 세트(closed-set) 및 최신 개방 세트(open-set) 객체 탐지기들의 성능을 COCO-FP에서 평가한 결과, 많은 탐지기들이 높은 수준의 false positive를 생성하는 것으로 나타났습니다. COCO-FP 데이터셋은 현재의 탐지 방법들이 현실 세계의 도전적인 상황에서 어떻게 성능이 저하되는지를 드러냅니다.



### FACT: Feature Adaptive Continual-learning Tracker for Multiple Object Tracking (https://arxiv.org/abs/2409.07904)
- **What's New**: 본 논문에서는 Feature Adaptive Continual-learning Tracker (FACT)라는 새로운 다중 물체 추적(MOT) 프레임워크를 제안합니다. 이 프레임워크는 모든 과거 추적 정보를 활용하여 실시간 추적과 특성 학습을 가능하게 합니다.

- **Technical Details**: FACT 프레임워크는 다양한 최신 특성 기반 추적기와 통합될 수 있으며, 이는 최소한의 추가 계산으로 추적 성능을 향상시킵니다. 주요 구성 요소로는 모든 과거 추적 정보를 동시에 사용할 수 있도록 온라인에서 훈련할 수 있는 Feature Adaptive Continual-learning (FAC) 모듈과 새로운 대상의 초기화시 강인한 추적을 보장하는 두 단계의 연관 모듈이 포함됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 MOT17 및 MOT20 벤치마크에서 최신 온라인 추적 성능을 달성함을 입증하였습니다. 특히, HOTA 지표에서 탁월한 성과를 보였습니다.



### Microscopic-Mamba: Revealing the Secrets of Microscopic Images with Just 4M Parameters (https://arxiv.org/abs/2409.07896)
Comments:
          5 pages, 1 figures

- **What's New**: 이번 연구에서는 Microscopic-Mamba라는 새로운 아키텍처를 제안했습니다. 이는 Mamba의 글로벌 특징 학습 장점과 CNN의 로컬 특징 추출 능력을 결합한 경량 혼합 아키텍처입니다. 또한, Microscopic-Mamba는 MIC 작업을 위한 첫 번째 Mamba 응용 사례로, 향후 MIC 탐색을 위한 새로운 기준을 제시합니다.

- **Technical Details**: 해당 모델은 Hybrid-Conv-SSM Block을 핵심 구성 요소로 포함하고 있습니다. 이 블록은 Conv Branch와 SSM Branch의 이중 가지 구조를 가지고 있으며, 각각의 채널을 분리하여 처리함으로써 로컬 및 글로벌 특징을 추출합니다. 또한, Modulation Interaction Feature Aggregation (MIFA) 모듈을 통해 두 가지 정보를 효율적으로 융합합니다.

- **Performance Highlights**: 다섯 개의 공개 데이터셋에 대한 광범위한 실험 결과, Microscopic-Mamba는 기존 최선의 방법들보다 우수한 성능을 나타내었으며, 더 적은 파라미터와 계산 복잡도를 유지합니다.



### UNIT: Unsupervised Online Instance Segmentation through Tim (https://arxiv.org/abs/2409.07887)
- **What's New**: 이 연구에서는 Lidar 포인트 클라우드에서 클래스 무관의 비지도 학습 온라인 인스턴스 분할(instance segmentation) 및 추적(tracking) 작업을 위한 새로운 접근 방식을 제안하고 있습니다. 이 방법은 수동 주석 없이 생성된 의사 라벨(pseudo-labels)을 활용하여 실시간으로 객체를 추적합니다.

- **Technical Details**: 제안하는 방법은 4D 클러스터링(spatio-temporal clustering)을 통해 Lidar 시퀀스로부터 의사 객체 세그먼트를 생성하고, 이를 바탕으로 autoregressive 방식의 네트워크를 훈련시키는 것입니다. 각 Lidar 스캔을 독립적으로 처리하며, 반복적으로 입력하여 객체를 라벨링합니다.

- **Performance Highlights**: SemanticKITTI와 PandaSet-GT 데이터셋에서 강력한 기준선(baselines)과 비교했을 때, 제안된 UNIT 방법이 더 우수한 성능을 보였음을 확인했습니다. 특히 이 방법은 동적 객체의 수에 구애받지 않고 훈련할 수 있는 장점이 있습니다.



### Real-time Multi-view Omnidirectional Depth Estimation System for Robots and Autonomous Driving on Real Scenes (https://arxiv.org/abs/2409.07843)
- **What's New**: 본 연구에서는 로봇 내비게이션 및 자율주행을 위한 옴니디렉셔널(depth estimation) 깊이 추정 시스템인 HexaMODE를 제안합니다. 이 시스템은 여섯 개의 물고기눈 카메라를 활용하여 360도 깊이 맵을 캡쳐하는 동시에, 실시간 깊이 추정을 가능케 하는 RtHexa-OmniMVS 알고리즘을 함께 제시합니다.

- **Technical Details**: 제안된 HexaMODE 시스템은 여섯 개의 물고기눈 카메라로 구성된 패턴을 기반으로 구축되어 있으며, NVIDIA Jetson AGX Orin을 엣지 컴퓨팅 디바이스로 사용하여 깊이 추정 및 시스템 제어를 수행합니다. 새로운 복합 구형 스위핑(Combined Spherical Sweeping) 방법을 도입하여, 복잡한 매칭 데이터의 구축 과정을 단순화하여 연산 효율성을 크게 향상시켰습니다. 또한, 선생님-학생(self-training) 구조를 활용한 자기 학습 전략을 통해, 실제 환경에서 발생할 수 있는 복잡성을 해결하고, 높은 정확도를 보장합니다.

- **Performance Highlights**: 제안된 RtHexa-OmniMVS 알고리즘은 다양한 실험 환경에서 뛰어난 성능을 보이며, 특히 실내 및 실외의 복잡한 환경에서도 높은 정확도를 달성합니다. NVIDIA Orin 플랫폼에서 15 fps 이상의 추론 속도를 기록하며, 현실 세계의 시나리오에 대한 높은 견고성과 일반화 성능을 입증했습니다.



### Structured Pruning for Efficient Visual Place Recognition (https://arxiv.org/abs/2409.07834)
- **What's New**: 이 논문은 Visual Place Recognition (VPR) 분야에 Structured Pruning을 도입하여 네트워크 아키텍처의 비필수 weights를 제거하고 리소스 효율성을 높이는 새로운 방법을 제안합니다.

- **Technical Details**: Structured Pruning은 Neural Network에서 accuracy에 미치는 영향이 최소인 비중요 weights 그룹을 제거하여 네트워크 크기를 줄이는 방법입니다. 이는 전통적인 unstructured pruning과 달리 필터나 뉴런을 통째로 제거하여 메모리 접근 효율성을 유지하며 병렬 처리를 지원합니다.

- **Performance Highlights**: 이 방법을 적용한 결과 모델 메모리 사용량과 지연 시간이 각각 21%, 16% 감소하며, recall@1 정확도는 1% 미만으로 소폭 줄어들었습니다. 이러한 개선은 정확도를 거의 손실 없이 리얼타임 응용 프로그램에서 사용할 수 있도록 허용합니다.



### A Comprehensive Survey on Deep Multimodal Learning with Missing Modality (https://arxiv.org/abs/2409.07825)
Comments:
          Work in progress and welcome to discussion

- **What's New**: 이 문서는 Multimodal Learning with Missing Modality (MLMM)에 대한 최근 발전을 포괄적으로 다루고 있으며, 특히 딥러닝 기술에 초점을 맞춘 최초의 종합적인 설문조사입니다. MLMM과 전통적인 Multimodal Learning with Full Modality (MLFM)의 차이를 분명히 하고, 현재의 MLMM 방법 및 응용분야, 데이터셋에 대한 깊이 있는 분석을 제공합니다.

- **Technical Details**: MLMM의 주요 과제는 데이터 수집의 제한이나 배포 환경의 제약으로 인해 훈련 및 테스트 동안 여러 개의 모달리티를 동적으로 처리하고 통합하는 것입니다. 이 설문조사는 MLMM 방법론의 구조를 모달리티 보강, 특성 공간 설계, 아키텍처 설계 및 모델 선택의 네 가지 주요 차원에서 분류합니다.

- **Performance Highlights**: MLMM 접근법은 의료 진단, 정보 검색, 원거리 감지 및 로봇 비전 등 다양한 분야에서 응용되고 있으며, 이 설문조사는 이러한 응용 프로그램을 통해 MLMM의 현장 적용 가능성을 강조합니다.



### What is YOLOv9: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector (https://arxiv.org/abs/2409.07813)
- **What's New**: YOLOv9는 YOLOv8에 비해 49%의 매개변수 감소와 43%의 계산 복잡성 감소를 달성하면서도 0.6%의 정확도 향상을 보여주는 최신 객체 탐지 모델입니다. 이 모델은 Programmable Gradient Information (PGI)와 Generalized Efficient Layer Aggregation Network (GELAN) 등의 혁신적인 아키텍처와 훈련 방법론을 통합하여 성능이 크게 향상되었습니다.

- **Technical Details**: YOLOv9는 CNN 기반의 백본(backbone)을 유지하면서, GELAN을 통합하여 멀티 스케일(feature extraction) 특성과 계층 간 정보를 효과적으로 보존합니다. PGI는 정보 손실을 줄이는 데 기여하며, PGI의 가역적 구조는 예측과 훈련 동안 중요한 데이터를 보존하여 신뢰할 수 있는 결과를 제공합니다. YOLOv9는 또한 PyTorch 및 TensorRT와의 원활한 통합을 지원합니다.

- **Performance Highlights**: YOLOv9는 Microsoft COCO 데이터셋에서 평균 평균 정밀도(mean Average Precision, mAP)와 추론 속도에서 YOLOv8을 능가했습니다. 다양한 하드웨어 플랫폼에서의 배포가 용이하며, IoT 기기부터 고성능 GPU에 이르기까지 다양한 응용 분야에서 뛰어난 성능을 보여줍니다.



### SURGIVID: Annotation-Efficient Surgical Video Object Discovery (https://arxiv.org/abs/2409.07801)
Comments:
          9 pages, 4 figures, 2 tables

- **What's New**: 본 연구에서는 수술 장면의 의미론적 분할을 위한 주석 효율적인 프레임워크를 제안합니다. 이 프레임워크는 비지도 학습(Self-supervised)을 활용하여 도구 및 해부 구조의 위치 식별을 수행하며, 최소한의 주석만 필요한 설정을 통해 이전의 완전 감독된 모델과 유사한 성능을 보입니다.

- **Technical Details**: 우리의 방법론은 DINO(Distillation with No Labels)라는 자기 지도학습 모델을 활용합니다. 이 모델은 자기 증류(Self-distillation), 대조 학습(Contrastive Learning), 비전 트랜스포머(Vision Transformers, ViT)를 사용하여 수술 장면의 픽셀 수준에서의 의미론적 분할을 위한 정보-rich한 피처를 추출합니다. 분할 마스크 생성을 그래프 분할(Graph Partitioning) 문제로 모델링하며, 생성된 비지도 마스크는 Mask2Former 모델을 사용하여 후처리 및 세분화됩니다.

- **Performance Highlights**: 제안된 프레임워크는 36개의 주석 레이블만으로도 높은 위치 정확도를 달성하였으며, 수술 단계 레이블을 약한 레이블로 활용하여 도구 위치 식별에서 약 2% 향상을 보였습니다. CaDIS 데이터셋에서의 대규모 절제 연구를 통해, 제안된 솔루션의 효과가 검증되었습니다.



### GateAttentionPose: Enhancing Pose Estimation with Agent Attention and Improved Gated Convolutions (https://arxiv.org/abs/2409.07798)
- **What's New**: 이 논문은 pose estimation 과제를 위한 UniRepLKNet 아키텍처를 개선한 GateAttentionPose라는 혁신적인 접근법을 소개합니다. 두 가지 주요 기여로는 Agent Attention 모듈과 Gate-Enhanced Feedforward Block (GEFB) 등이 있습니다. 이 방법은 기존의 대규모 커널 합성을 대체하여 계산 효율성을 크게 개선하면서도 전역 맥락 모형을 보존합니다.

- **Technical Details**: GateAttentionPose 모델은 GLACE 모듈로부터 시작하여, 크기가 [96, 64, 48]인 feature map으로 변환된 후 개선된 backbone으로 처리됩니다. 이 아키텍처에서 Agent Attention 모듈은 대규모 커널 합성을 대체하고, GEFB는 feature extraction 효율을 증가시킵니다. 또한, 최적화된 다운샘플 블록과 CBAM을 통해 중요한 특징을 강조하며, Batch Normalization과 Squeeze-and-Excitation 블록을 통해 모델의 일반화 능력을 최적화합니다.

- **Performance Highlights**: GateAttentionPose는 COCO 및 MPII 데이터셋에 대한 광범위한 평가를 통해 기존의 최첨단 방법들, 특히 원래의 UniRepLKNet과 비교하여 개선된 효율성과 함께 우수한 성능을 보였습니다. 이 연구는 pose estimation 분야에서 이론적 이해와 실제 응용을 발전시킬 수 있는 강력한 솔루션을 제공합니다.



### Quaternion Nuclear Norm minus Frobenius Norm Minimization for color image reconstruction (https://arxiv.org/abs/2409.07797)
Comments:
          This paper was accepted by Pattern Recognition on September 5, 2024

- **What's New**: 이번 논문에서는 색상 이미지 복원을 위한 새로운 방법, 즉 Quaternion Nuclear Norm Minus Frobenius Norm Minimization (QNMF)을 제안합니다. 이 방법은 색상 채널 간의 상관관계를 고려하여 색상 왜곡과 아티팩트를 최소화합니다.

- **Technical Details**: QNMF는 쿼터니언(Quaternion) 대수를 사용하여 RGB 채널 간의 관계를 포괄적으로 포착합니다. 핵 규범(nuclear norm)과 프로베니우스 규범(Frobenius norm)의 차이를 활용한 정규화 기법을 통해 쿼터니언으로 인코딩된 색상 이미지의 저등급(low-rank) 구조를 근사합니다. 이 방법은 다양한 색상 저수준 비전(low-level vision) 작업에 적용되며, 이론적 증명이 제공되어 수학적 무결성을 보장합니다.

- **Performance Highlights**: QNMF는 색상 이미지 노이즈 제거, 블러 제거, 인페인팅(inpainting), 임펄스 노이즈 제거 등의 다양한 작업에서 최첨단 결과를 달성하였습니다. 또한, QNMF는 기존의 방법들과 비교했을 때 성능을 향상시키고 강건성을 증가시켜 주목받고 있습니다.



### In-Situ Fine-Tuning of Wildlife Models in IoT-Enabled Camera Traps for Efficient Adaptation (https://arxiv.org/abs/2409.07796)
- **What's New**: 이 논문에서는 WildFit이라는 새로운 접근 방식을 도입하여 머신 러닝 모델이 카메라 트랩 애플리케이션에서 높은 도메인 일반화 성능을 유지하면서도 효율적인 추론을 가능하게 합니다.

- **Technical Details**: WildFit은 환경 변화에 맞춰서 모델을 지속적으로 미세 조정하는 방법을 활용합니다. 이는 백그라운드 이미지와 소스 도메인에서의 동물 이미지를 혼합하여 새로운 도메인을 대표하는 훈련 이미지를 생성하는 백그라운드 인식 데이터 합성을 통해 이루어집니다. 이를 통해 복잡한 계산 리소스 없이도 새로운 환경에 대해 높은 분류 정확도를 유지할 수 있습니다.

- **Performance Highlights**: WildFit은 여러 카메라 트랩 데이터셋에 대한 광범위한 평가를 통해 전통적인 접근 방식에 비해 분류 정확도와 계산 효율성에서 현저한 개선을 달성했습니다.



### Lagrange Duality and Compound Multi-Attention Transformer for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2409.07793)
Comments:
          5 pages, 4 figures, 3 tables

- **What's New**: 이 논문에서는 CMAformer라는 새로운 모델을 제안하며, 이는 ResUNet과 Transformer의 장점을 결합한 것입니다. 또한, 장기 분포 문제를 해결하기 위해 Lagrange Duality Consistency (LDC) Loss와 Boundary-Aware Contrastive Loss를 통합한 반지도 학습 프레임워크를 소개합니다.

- **Technical Details**: CMAformer는 spatial attention과 channel attention을 효과적으로 융합하기 위해 Cross Attention 레이어를 사용하며, 이는 멀티스케일 특성 융합을 위한 중요한 요소입니다. 또한, Transformer 블록 내에서 multi-head self-attention (MSA)과 다층 퍼셉트론 (MLP)이 포함되어 각 쿼리 매트릭스에 대해 개별적으로 레이어 정규화를 수행하고 있습니다.

- **Performance Highlights**: CMAformer는 여러 공공 의료 이미지 데이터셋에서 세분화 작업에 대해 대부분의 최신 모델을 능가하는 성능을 입증하였습니다. 이를 통해 반지도 학습 앙상블에서의 강력한 상호 보완성을 보여주고 있습니다.



### ASSNet: Adaptive Semantic Segmentation Network for Microtumors and Multi-Organ Segmentation (https://arxiv.org/abs/2409.07779)
Comments:
          8 pages, 4 figures, 3 tables

- **What's New**: 이번 연구에서는 의학 영상 분할을 위한 Adaptive Semantic Segmentation Network (ASSNet)라는 새로운 Transformer 기반 아키텍처를 제안합니다. ASSNet은 로컬 및 글로벌 피쳐 통합을 통해 미세한 종양 및 소형 기관의 정확한 분할을 지원합니다.

- **Technical Details**: ASSNet은 U자형 인코더-디코더 네트워크로 구성되며, 인코더는 다섯 가지 해상도에서 이동 윈도우 셀프 어텐션(shifted window self-attention)을 활용해 다중 스케일 특성을 추출합니다. 디코더는 Adaptive Feature Fusion (AFF) 구조를 통해 장거리 의존성(long-range dependencies)과 다중 스케일 피쳐 융합을 지원합니다.

- **Performance Highlights**: ASSNet은 LiTS2017, ISICDM2019, Synapse 데이터셋에서 최신 기술 점수를 초과하며, 다양한 의학 이미지 분할 작업에서 최첨단 성능을 보여주었습니다.



### Exploring Kolmogorov-Arnold networks for realistic image sharpness assessmen (https://arxiv.org/abs/2409.07762)
- **What's New**: 이번 연구에서는 Kolmogorov-Arnold 네트워크(KAN)를 활용하여 실제 이미지 품질 평가에서의 점수 예측을 탐구했습니다. 특히 Taylor 시리즈 기반의 KAN(TaylorKAN)을 제안하며, 다양한 KAN 변형을 이미지 품질 평가에 적용하여 성능을 비교했습니다.

- **Technical Details**: TaylorKAN은 연속 다변수 함수를 일변수 함수의 합으로 표현하는 Kolmogorov-Arnold 정리에 영감을 받아 개발되었습니다. 본 연구에서는 지원 벡터 회귀(SVR)와 다층 퍼셉트론(MLP)과 함께 mid-level과 high-level 특성을 활용하여 BID2011, CID2013, CLIVE, KonIQ-10k 데이터베이스에서 점수 예측 성능을 평가했습니다.

- **Performance Highlights**: TaylorKAN은 세 가지 데이터베이스에서 mid-level 특성을 사용 시 최고의 성능을 보였으며, KAN은 전반적으로 SVR과 MLP보다 우수하거나 경쟁력을 보였습니다. 그러나 high-level 특성을 사용할 때 CLIVE 데이터베이스에서는 KAN이 SVR 및 MLP보다 열세를 보였습니다.



### From Uncertainty to Clarity: Uncertainty-Guided Class-Incremental Learning for Limited Biomedical Samples via Semantic Expansion (https://arxiv.org/abs/2409.07757)
- **What's New**: 이번 연구는 생물의학 분야에서 제한된 샘플을 활용한 클래스 점진적 학습(class incremental learning) 방법을 처음으로 제안했습니다. 이 방법은 새로운 질병의 사례가 제한적으로 발생하는 상황에서 이전 지식을 유지하면서 새로운 클래스의 지식을 학습할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 모듈로 구성됩니다. 첫째, Uncertainty Trajectory Analyzer (UTA) 모듈은 샘플의 불확실성을 측정하여 가장 중요한 샘플을 메모리 뱅크에 저장하여 후속 학습에서 재사용할 수 있도록 합니다. 둘째, 세분화된 의미 확장(Fine-Grained Semantic Expansion) 모듈은 각 클래스를 더 세부적인 하위 특징으로 분해하여 특징 공간을 풍부하게 하고, 새로운 클래스에 대한 일반화 여지를 증가시킵니다.

- **Performance Highlights**: 이 방법은 PathMNIST 및 BloodMNIST 데이터셋에서 기존의 최첨단 클래스 점진적 학습 방법들보다 평균 7.53%에서 37.12%까지 높은 정확도를 달성하여 성능 향상을 입증했습니다. 또한, 불균형 데이터와 긴 꼬리 분포를 가진 데이터셋에서도 효과적으로 작동함을 실험적으로 입증했습니다.



### DiTAS: Quantizing Diffusion Transformers via Enhanced Activation Smoothing (https://arxiv.org/abs/2409.07756)
- **What's New**: 본 논문에서는 자원 제약이 있는 장치에서 Diffusion Transformers (DiTs)의 효율적인 추론을 위한 데이터 없는 포스트 트레이닝 양자화(Post-Training Quantization) 방법인 DiTAS를 제안합니다. DiTAS는 입력 활성값 내의 채널별 이상치(channel-wise outliers)의 영향을 완화하기 위해 제안된 시간 집계 스무딩(temporal-aggregated smoothing) 기술에 의존하여 극히 낮은 비트 폭에서도 훨씬 낮은 양자화 오류를 달성합니다.

- **Technical Details**: DiTAS는 입력 활성값에 대한 이상치의 영향을 최소화하기 위해 시간 집계 스무딩(TAS) 기술을 사용합니다. TAS는 모든 시간 단계에서 이상치의 크기를 집계하는 채널별 스무딩 인자를 사용하는 기술로, 이를 통해 양자화된 DiT의 성능을 향상시킵니다. 또한, 각 입력 채널에 대한 스무딩 인자를 조정하기 위해 레이어별 그리드 서치(layer-wise grid search) 전략을 도입하였습니다. LoRA 모듈과 교대 최적화(Alternating Optimization)를 통합하여 양자화 가중치의 성능을 개선합니다.

- **Performance Highlights**: DiTAS는 ImageNet의 256x256 및 512x512 해상도에서 최신 성능을 달성합니다. 특히, W4A8 구성에서 50 샘플링 스텝에 대해 9.05의 FID-10K 점수를, 100 샘플링 스텝에서는 6.86을 기록하였으며, 512x512 해상도에서도 각각 17.92 및 13.45의 FID-10K 점수를 달성했습니다.



### GatedUniPose: A Novel Approach for Pose Estimation Combining UniRepLKNet and Gated Convolution (https://arxiv.org/abs/2409.07752)
- **What's New**: 이번 논문에서는 GatedUniPose라는 새로운 포즈 추정(pose estimation) 방법을 제안하였습니다. 이 방법은 UniRepLKNet과 Gated Convolution을 결합하고 GLACE 모듈을 도입하여 임베딩(embedding) 기능을 개선하였습니다.

- **Technical Details**: GatedUniPose는 DySample 업샘플링(upsampling)을 이용하여 헤드 레이어(head layer)의 특징 맵(feature map) 연결 방식을 향상시킵니다. 또한, 복잡한 장면(scene)과 가림(occlusion) 문제를 효과적으로 처리합니다.

- **Performance Highlights**: GatedUniPose는 COCO, MPII, CrowdPose 데이터셋에서 실험을 진행한 결과, 상대적으로 적은 매개변수 수로도 밝은 성능 향상을 보였으며, 유사하거나 더 많은 파라미터를 가진 모델과 비교했을 때 더 나은 또는 동등한 결과를 나타냈습니다.



### Top-down Activity Representation Learning for Video Question Answering (https://arxiv.org/abs/2409.07748)
Comments:
          presented at MIRU2024

- **What's New**: 이 논문에서는 비디오 질문 응답(VideoQA) 작업에서 복잡한 계층적 인간 활동을 효과적으로 캡처하기 위해 CLIP 모델의 공간 시각적 상황 표현 능력을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 긴 비디오 시퀀스를 단일 그리드 이미지로 변환하고, LLaVA 멀티모달 모델을 파인튜닝합니다. 이를 통해 비디오의 맥락적 사건을 비연속적으로 표현할 수 있으며, STAR와 NExTQA 벤치마크에서 높은 성능을 달성했습니다.

- **Performance Highlights**: 이 연구는 STAR 작업에서 78.4%의 정확도를 기록하였으며, NExTQA 작업에서는 기존의 최첨단 점수를 2.8포인트 초과하는 성과를 보여주었습니다.



### Multi-object event graph representation learning for Video Question Answering (https://arxiv.org/abs/2409.07747)
Comments:
          presented at MIRU2024

- **What's New**: 이 논문에서는 비디오 질문 응답(Video Question Answering, VideoQA) 분야에서 다수 객체를 포함한 사건(event) 표현을 포착하기 위한 새로운 접근법인 CLanG(Contrastive Language Event Graph Representation Learning)를 제안하고 있습니다. 기존의 방법들은 개별 객체의 움직임에 초점을 맞추었지만, CLanG는 복잡한 시나리오를 처리할 수 있는 방법을 제공합니다.

- **Technical Details**: CLanG는 다층 GNN-클러스터 모듈을 사용하여 비디오에서 추출된 다수 객체의 사건 표현을 효과적으로 학습합니다. 이 모듈은 경쟁적 그래프 표현 학습을 통해 질문 텍스트와 관련된 다수 객체 사건 그래프 간의 대비 학습을 가능하게 합니다. 또한, 그래프의 출력 노드 수를 조정하여 원래의 그래프 정보를 보존하며, 자기 주의(self-attention) 계층을 통해 계층적 사건 그래프의 영향을 강조합니다.

- **Performance Highlights**: CLanG는 NExT-QA 및 TGIF-QA-R이라는 두 개의 도전적인 VideoQA 데이터셋에서 강력한 기준선보다 최대 2.2% 더 높은 정확도를 달성하며, 특히 인과(causal) 및 시간적(temporal) 질문 처리에서 2.8% 향상된 성능을 보여줍니다.



### Learning Brain Tumor Representation in 3D High-Resolution MR Images via Interpretable State Space Models (https://arxiv.org/abs/2409.07746)
Comments:
          The code is available at this https URL

- **What's New**: 본 연구는 고해상도 3D 다중 대조 자기공명 (MR) 이미지를 효과적으로 처리하고 해석 가능성을 높일 수 있는 새로운 상태 공간 모델 (State-Space Model, SSM) 기반의 마스크 자동 인코더(autoencoder)를 제안합니다.

- **Technical Details**: 이 모델은 ViT와 유사한 구조를 사용하여 높은 해상도의 데이터를 처리하며, 포괄적이고 세밀한 특징을 동시에 캡처할 수 있습니다. 특히, 라테ント 특성(latent features)과 입력 볼륨 내 특정 영역 간의 관계를 시각화할 수 있는 라테ント-투-스페이셜 매핑(latent-to-spatial mapping) 기법을 도입하여 모델의 결정 과정을 보다 명확히 이해할 수 있도록 합니다.

- **Performance Highlights**: IDH 돌연변이 상태(id status) 및 1p/19q 동시 결실(co-deletion classification) 작업에서 최첨단 성능을 달성하며, SSM 기반의 자가 감독 학습(self-supervised learning) 기법이 효율성과 해석 가능성을 결합하여 방사선학(radiomics) 분석을 혁신할 잠재력이 있음을 보여줍니다.



### Transfer Learning Applied to Computer Vision Problems: Survey on Current Progress, Limitations, and Opportunities (https://arxiv.org/abs/2409.07736)
Comments:
          16 pages, 8 figures

- **What's New**: 본 연구는 Transfer Learning (TL) 기술을 활용하여 컴퓨터 비전 (CV) 문제들을 해결하는 새로운 접근 방식을 제시합니다. 기존의 컴퓨터 비전 기술이 가진 한계들을 극복하기 위해 TL의 최근 발전 상황과 이로 인한 기회를 탐구하고 있습니다.

- **Technical Details**: 본 연구는 인공지능 (AI), 머신러닝 (ML), 신경망 (NN), 딥 뉴럴 네트워크 (DNN), 컨볼루션 뉴럴 네트워크 (CNN), 순환 신경망 (RNN) 등의 개념들을 다루며, TL이 어떻게 CV 문제에 적용되는지를 설명합니다. TL은 한 도메인에서 얻은 지식을 다른 도메인에서의 학습 과정에 활용하여, 데이터가 부족할 때 효과적으로 모델을 학습시키는 기법입니다.

- **Performance Highlights**: TL을 활용하여 적은 데이터로도 높은 정확도를 달성할 수 있다는 점에 주목하십시오. 예를 들어, COVID-19 발생 초기에 데이터 부족 문제를 해결하기 위해 TL을 통해 기존 모델을 개조하여 성공적으로 질병을 감지하는 모델이 개발되었습니다. 이는 TL이 다양한 컴퓨터 비전 문제 해결에 매우 유용할 수 있음을 시사합니다.



### Advancing Depth Anything Model for Unsupervised Monocular Depth Estimation in Endoscopy (https://arxiv.org/abs/2409.07723)
Comments:
          7 pages, 6 figures

- **What's New**: 이 연구에서는 Depth Anything Model을 위한 새로운 파인튜닝 전략인 RVLoRA를 소개하며, 이는 깊이 추정의 성능을 향상시키기 위해 랜덤 벡터를 기반으로 하는 저랭크 적응(low-rank adaptation) 기법을 통합합니다.

- **Technical Details**: LVLoRA는 다양한 스케일에 대한 모델의 적응성을 향상시키며, Res-DSC 모듈은 Vision Transformer(ViT) 내에 CNN 구성을 통합하여 고주파 세부정보, 즉 에지 및 텍스처와 같은 지역적 특징을 더 잘 포착할 수 있게 합니다.

- **Performance Highlights**: SCARED 데이터셋에서 실험한 결과, 제안된 방법은 최소한의 학습 가능한 파라미터로 최신 성능(state-of-the-art performance)을 달성하였습니다. 이 방법은 최소 침습 내시경 수술의 정확성과 안전성을 크게 향상시킬 것으로 기대됩니다.



### FIReStereo: Forest InfraRed Stereo Dataset for UAS Depth Perception in Visually Degraded Environments (https://arxiv.org/abs/2409.07715)
Comments:
          Under review in RA-L. The first 2 authors contributed equally

- **What's New**: 이 논문은 자율 항공 시스템을 위한 시각적으로 저하된 환경에서의 스테레오 열 깊이 인식 데이터 세트 FIReStereo를 소개합니다. 이 데이터 세트는 다양한 조건에서 수집된 스테레오 열 이미지와 LiDAR, IMU, 그리고 실제 깊이 맵(ground truth depth maps)으로 구성되어 있습니다. 특히, 이 데이터 세트는 재난 대응 상황에서 로봇의 인식을 향상시키기 위해 설계되었습니다.

- **Technical Details**: FIReStereo 데이터 세트는 열 화상 카메라 두 대를 이용하여 수집되었으며, 구조물과 숲이 혼합된 환경에서의 깊이 추정 알고리즘 개발에 도움을 주기 위해 만들어졌습니다. 카메라는 24cm의 작은 간격으로 배치되어 있으며, 스모크 및 각종 날씨 조건에서 촬영된 데이터가 포함됩니다. 모든 센서 데이터는 동기화되어 제공되며, LiDAR 데이터를 이용한 실제 깊이 정보도 포함됩니다. 이 데이터는 로봇이 복잡한 환경을 탐색하는 데 필요한 정보를 제공합니다.

- **Performance Highlights**: 훈련된 모델들은 이전에 본 적 없는 스모키한 조건에서도 우수한 일반화 성능을 보여주었으며, 이는 스테레오 열 화상이 깊이 인식을 위한 강건함을 잘 나타냅니다. 다양한 환경 조건에서의 성능을 벤치마킹하여, 자율 비행체 플랫폼에서의 실시간 메트릭 깊이 추정의 강점과 한계를 분석하였습니다.



### CollaMamba: Efficient Collaborative Perception with Cross-Agent Spatial-Temporal State Space Mod (https://arxiv.org/abs/2409.07714)
Comments:
          Submitted to AAAI 2025

- **What's New**: 이번 연구에서는 효율적인 자원 활용이 가능한 크로스 에이전트(spatial-temporal) 협력 상태 공간 모델(SSM)인 콜라밤바(CollaMamba)를 제안합니다. 이 모델은 기존의 CNN이나 Transformer 기반 접근법을 대체하며, 긴 범위의 공간적 및 시간적 의존관계를 효과적으로 캡처할 수 있도록 설계되었습니다.

- **Technical Details**: 콜라밤바의 아키텍처는 크로스 에이전트 협력의 기본 백본 네트워크 및 두 개의 플러그 앤 플레이 모듈(단일 에이전트 역사 인식 기능 향상 모듈과 크로스 에이전트 협력 예측 모듈)로 이루어져 있습니다. 공간적 causal 의존관계를 모델링하는 Mamba 기반 인코더 및 디코더와 함께 동작하는 이 구조는 효율적으로 자원을 소모하면서도 높은 품질의 기술적 표현을 제공합니다.

- **Performance Highlights**: 실험 결과, 콜라밤바는 기존의 최첨단 기법들보다 우수한 모델 정확도를 달성하며, 계산 및 통신 오버헤드를 각각 71.9% 및 1/64까지 줄이는 성과를 보였습니다.



### TMFNet: Two-Stream Multi-Channels Fusion Networks for Color Image Operation Chain Detection (https://arxiv.org/abs/2409.07701)
Comments:
          15 pages, 12 figures

- **What's New**: 이 논문에서는 색상 이미지의 운영 체인 감지를 위한 새로운 두 흐름 다중 채널 융합 네트워크를 제안합니다. 이는 공간 잡음 스트림과 잡음 잔여 스트림을 보완적인 방식으로 탐색하여 색상 이미지 조작 연쇄 감지의 성능을 향상시킵니다.

- **Technical Details**: 제안된 네트워크는 풀링 없는 심층 잔여 아키텍처를 통해 다중 채널 상관관계의 전역 특징을 학습하며, 잡음 잔여 스트림에서 다중 채널의 상관 정보를 집계하는 필터를 설계합니다. 이를 통해 저수준 및 고수준 특징을 추출하고, 두 스트림의 특징을 융합 모듈에 입력하여 조작 체인에 대한 더욱 풍부한 구별 표현을 학습합니다.

- **Performance Highlights**: 제안된 방법은 크로스 데이터베이스, 크로스 해상도에서 뛰어난 일반화 능력을 보이며, JPEG 압축에 대한 강인성을 유지합니다. 또한, 긴 조작 체인 및 소셜 플랫폼 상의 공유 체인 감지에서도 만족스러운 성능을 발휘합니다.



### Learn from Balance: Rectifying Knowledge Transfer for Long-Tailed Scenarios (https://arxiv.org/abs/2409.07694)
- **What's New**: 본 논문에서는 Knowledge Rectification Distillation (KRDistill)이라는 새로운 프레임워크를 제안하여 imbalanced data에서의 Knowledge Distillation의 문제를 해결합니다. KRDistill은 teacher network에서 bias된 예측을 수정하고 balanced category priors를 통합하여 학습된 student network의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: KRDistill은 teacher network의 feature representations와 predictions의 불균형 문제를 해결하기 위해 representation-rectified distillation loss와 logit-rectified distillation loss를 사용합니다. 이러한 방법들은 teacher network가 제공하는 knowledge의 편향을 수정하고, tail categories에서의 오류를 감소시켜 student network의 훈련을 개선합니다.

- **Performance Highlights**: 여러 long-tailed datasets에서 수행된 실험 결과, KRDistill은 기존의 state-of-the-art Knowledge Distillation 방법과 비교했을 때, 높은 성능을 보이며, 지나치게 편향된 데이터 분포를 가진 상황에서도 신뢰할 수 있는 student networks의 훈련이 가능함을 입증했습니다.



### Open-Vocabulary Remote Sensing Image Semantic Segmentation (https://arxiv.org/abs/2409.07683)
- **What's New**: 본 논문에서는 원격 감지 이미지에 특화된 첫 번째 오픈 어휘 이미지 의미 분할(OVS) 프레임워크를 제안합니다. 기존 방법들과 달리 이 프레임워크는 빠르게 변화하는 방향과 상당한 크기 변화를 다루기 위해 개발되었습니다.

- **Technical Details**: 본 연구에서는 회전 집계 유사도 계산 모듈을 도입하여 초기 의미 맵을 생성합니다. 이 모듈은 입력 원격 감지 이미지를 여러 방향으로 회전시켜 방향 적응형 유사도 맵을 생성하고, 이 맵을 활용해 더 정교한 의미 맵을 생성합니다. 또한 다중 스케일 이미지 특징을 업샘플링 과정에 통합하여 최종적으로 크기 인식 의미 마스크를 만듭니다.

- **Performance Highlights**: 제안된 방법은 새로운 원격 감지 OVS 벤치마크에서 extensive experiments를 통해 기존의 최첨단 자연 이미지 기반 OVS 접근 방식들을 능가하는 성능을 입증하였습니다.



### Foundation Models Boost Low-Level Perceptual Similarity Metrics (https://arxiv.org/abs/2409.07650)
Comments:
          Code: this https URL

- **What's New**: 이번 연구에서는 기존의 FR-IQA(full-reference image quality assessment) 모델들이 주로 최종 레이어 또는 embedding에 의존했던 것과 달리, 대형 모델의 중간 특징(intermediate features)을 활용하여 더 효과적인 저수준 인지 유사도 메트릭을 개발하고자 한다. 이를 통해 머신 러닝 접근 방식을 통해 얻어진 인지 유사도 점수가 인간의 판단에 더 잘 맞도록 함을 증명하였다.

- **Technical Details**: 연구에서는 DINO 및 CLIP과 같은 파운데이션 모델의 중간 특징을 평가하고, 다양한 데이터셋 및 거리 측정 방법을 통해 이들 모델이 저수준 인지 유사도 메트릭 개발에 있어 더 정확하고 강인하다는 것을 입증하였다. 특히, 중간 특징을 활용한 접근 방식이 embedding을 이용한 방식보다 성능이 더 뛰어난 것으로 나타났다.

- **Performance Highlights**: DINOv1-ViT-B 모델은 모든 데이터셋과 기하학적 왜곡 유형에서 저수준 인지 유사도 작업을 위한 가장 효과적인 모델로 검증되었다. CLIP-ViT-B 모델 역시 중간 특징을 사용했을 때 최상의 성능을 보여주며, 저수준 인지 유사도 점수에서 이전의 최신 기술보다 우수한 결과를 산출하였다. 이로써 해당 메트릭은 기존의 전통적인 방법 및 최신 학습된 메트릭을 초월할 수 있음을 입증하였다.



### DiffTED: One-shot Audio-driven TED Talk Video Generation with Diffusion-based Co-speech Gestures (https://arxiv.org/abs/2409.07649)
- **What's New**: DiffTED는 단일 이미지로부터 음성 오디오에 따라 코스피치 제스처가 포함된 TED 스타일의 비디오를 생성하는 새로운 접근 방식을 제안합니다. 이 방법은 기존의 비디오-비디오 변환 기술과 대신 혁신적인 diffusion 모델을 활용하여 비디오 생성을 가능하게 하여, 단일 이미지에서 실사 비디오를 생성할 수 있도록 합니다.

- **Technical Details**: DiffTED는 Thin-Plate Spline (TPS) 모션 모델을 기반으로 하여 2D TPS 키포인트를 생성하고, 이를 통해 제스처를 자연스럽고 일관성 있게 흐르게 합니다. 이 과정에서 classifier-free guidance를 사용하여 사전 훈련된 분류기에 의존하지 않고 오디오 입력과 함께 제스처를 일치시킵니다. 모형은 다수의 비디오 프레임을 변환하는 데 필요한 구조적 복잡성을 줄이고, 상대적으로 간단한 네트워크 구조로 구성되어 있습니다.

- **Performance Highlights**: DiffTED는 실험 결과에서 시간적으로 일관된 비디오와 다양성이 풍부한 코스피치 제스처를 생성하며, 기존의 LSTM 및 CNN 기반 모델보다 우수한 성능을 보입니다. 논문 발표 후 소스 코드는 공개될 예정이며, 이는 연구자로 하여금 DiffTED의 방법론을 바탕으로 한 추가 연구를 가능하게 합니다.



### Feature Importance in Pedestrian Intention Prediction: A Context-Aware Review (https://arxiv.org/abs/2409.07645)
- **What's New**: 이번 연구에서는 Context-aware Permutation Feature Importance (CAPFI)라는 새로운 접근 방식을 도입하여 보행자의 횡단 의도를 예측하는 모델의 해석 가능성을 높였습니다. 이 방법은 시나리오 맥락을 세분화하여 입력 특성이 최종 예측에 기여하는 방식을 보다 명확하게 파악할 수 있도록 합니다.

- **Technical Details**: CAPFI는 목표 특성들에 대한 무작위 셔플링을 통해 특성 중요성을 평가하는 방법입니다. 본 연구에서는 Pedestrian Intention Estimation (PIE) 데이터셋을 16개의 비교 가능한 맥락 집합으로 나누고, 각각의 맥락에서 다섯 개의 신경망 아키텍처의 성능을 평가했습니다. CAPFI는 보행자 경계 상자와 자가차량의 속도가 보행자 의도 예측에 미치는 중요성을 강조합니다.

- **Performance Highlights**: 연구 결과, 보행자의 경계 상자와 자가 차량의 속도가 보행자 의도 예측에서 중요한 역할을 하며, 속도 특성의 타당한 예측 편향이 발생할 수 있음을 밝혔습니다. 또한, 보행자-차량 상호작용을 고려한 대안적인 특성 표현을 제안함으로써 입력 특성이 의도 예측에 미치는 기여도를 향상시켰습니다.



### Token Turing Machines are Efficient Vision Models (https://arxiv.org/abs/2409.07613)
- **What's New**: 이 논문에서는 Vision Token Turing Machines (ViTTM)을 제안하여 효율적이고 낮은 지연 시간의 메모리 증강 Vision Transformer (ViT)를 구현하였습니다. ViTTM은 기존의 Neural Turing Machines와 Token Turing Machines의 원리를 비전 분야에 적용하고 있으며, 이미지 분류 및 분할과 같은 비순차적 컴퓨터 비전 작업을 위해 설계되었습니다.

- **Technical Details**: ViTTM은 프로세스 토큰과 메모리 토큰의 두 세트를 생성하며, 프로세스 토큰은 엔코더 블록을 통과하고 각 블록에서 메모리 토큰을 읽고 기록할 수 있습니다. 메모리 토큰의 수가 프로세스 토큰보다 많아 지연 시간을 줄이면서도 정확도를 유지할 수 있도록 구성됩니다. ViTTM은 ImageNet-1K와 ADE20K 데이터셋에서 평가되었습니다.

- **Performance Highlights**: ImageNet-1K에서 기존 ViT-B 모델은 529.5ms의 지연 시간과 81.0%의 정확도를 보이는 반면, ViTTM-B는 234.1ms로 56% 빠르며, 2.4배 적은 FLOPs로 82.9%의 정확도를 기록하였습니다. ADE20K 데이터셋에서는 ViTTM-B가 45.17 mIoU와 26.8 FPS를 달성하여, 기존 ViT-B보다 94% 더 높은 FPS를 보였습니다.



### 2D bidirectional gated recurrent unit convolutional Neural networks for end-to-end violence detection In videos (https://arxiv.org/abs/2409.07588)
Comments:
          8 pages, 6 figures, 2020 International Conference on Image Analysis and Recognition (ICIAR)

- **What's New**: 이번 연구에서는 Bidirectional Gated Recurrent Unit (BiGRU)와 2D Convolutional Neural Network (CNN)를 결합한 아키텍처를 제안하여 비디오 시퀀스에서 폭력을 탐지하는 시스템을 개발했습니다. CNN이 각 프레임에서 공간적 특징을 추출하고, BiGRU가 여러 프레임에서 추출된 특징을 기반으로 시간적 및 지역적 운동 특징을 추출합니다.

- **Technical Details**: 제안된 네트워크는 VGG16 모델을 바탕으로 하여 시간에 따라 분산된 2D CNN과 BiGRU를 결합하여 동작 인식을 수행합니다. VGG16은 INRA 인물 데이터셋을 사전 훈련하여 인물 인식 성능을 끌어올립니다. GRU는 LSTM의 기울기 소실 문제를 피할 수 있도록 선택되었으며, 양방향 GRU를 사용하여 이전과 이후의 시퀀스를 고려하여 정확성을 높였습니다.

- **Performance Highlights**: 제안된 BiGRU-CNN 네트워크는 Hockey 데이터셋에서 98%의 정확도, Violent Flow 데이터셋에서 95.5%의 정확도를 달성했습니다. 이는 기존 기술보다 뛰어난 성능을 보여줍니다.



### Minimizing Embedding Distortion for Robust Out-of-Distribution Performanc (https://arxiv.org/abs/2409.07582)
Comments:
          Accepted to ECCV 2024 workshop

- **What's New**: 이 논문에서는 기존의 Foundation 모델을 특정 다운스트림 작업에 적응시키면서도 강력한 일반화 능력을 유지하기 위한 새로운 접근법인 'similarity loss'를 소개합니다. 이는 파인튜닝 과정에 통합될 수 있으며, 사전 훈련된 임베딩(embeddings)과 파인튜닝된 임베딩 간의 왜곡을 최소화하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방법은 similarity loss를 사용하여 파인튜닝 중에 사전 훈련된 임베딩의 일반화 능력을 보존하면서도 작업 특화적(adapted) 성능을 향상시키는 것입니다. 이 방법은 임베딩 공간의 왜곡을 최소화하여 OOD(Out-of-Distribution) 성능을 유지합니다. 본 실험에서는 위성 이미지(image classification on satellite imagery)와 얼굴 인식(face recognition)이라는 두 가지 도전적인 작업을 통해 접근 방식을 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 OOD 성능을 크게 개선하면서도 ID 성능의 작은 감소만을 초래했습니다. 이는 다양한 작업에서 성능을 유지하는 데 있어 강력한 일반화 능력을 보존할 수 있음을 보여줍니다.



### Violence detection in videos using deep recurrent and convolutional neural networks (https://arxiv.org/abs/2409.07581)
Comments:
          11 pages, 7 figures, 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)

- **What's New**: 본 연구에서는 폭력 감지를 위한 새로운 딥러닝 아키텍처를 제안합니다. 이 아키텍처는 순환신경망(RNN)과 2차원 합성곱 신경망(2D CNN)을 결합하여 비디오에서 폭력 장면을 효과적으로 감지합니다.

- **Technical Details**: 제안된 방법은 RGB 비디오 프레임과 함께 캡처한 영상의 광학 흐름(optical flow)을 사용하여 공간적(spatial) 및 시간적(temporal) 특성을 추출합니다. EfficientNet B0을 활용하여 CNN을 구성하고, LSTM을 이용한 RNN으로 시간적 표현을 얻습니다. 이 방식은 두 개의 개별 네트워크(하나는 RGB 프레임용, 다른 하나는 광학 흐름용)로 구성되고, 이들의 특징을 결합하여 최종 클래스를 도출합니다.

- **Performance Highlights**: 제안된 접근법은 3개의 공개 데이터베이스에서 테스트되었으며, 최신 기술들과 유사한 성능을 달성하고 때로는 이를 초월하는 결과를 보였습니다.



### Self-Masking Networks for Unsupervised Adaptation (https://arxiv.org/abs/2409.07577)
Comments:
          Oral at GCPR'24, code at this https URL

- **What's New**: 이번 연구에서는 Self-Supervised Masking Networks (SMNs)라는 새로운 접근 방식을 제안하여, 사전 훈련된 일반 모델을 효율적으로 세밀하게 조정할 수 있는 방법을 모색했습니다. 특히, 레이블이 부족한 환경에서 높은 성능을 끌어내는 것이 가능합니다.

- **Technical Details**: SMNs는 이진 마스크를 학습하여, 모델이 더 적은 양의 레이블 데이터로도 효과적으로 적응할 수 있도록 도와줍니다. 이 방법은 79배 더 저장 효율이 높으며, 8개의 데이터셋과 3개의 모델 아키텍처를 통해 검증되었습니다.

- **Performance Highlights**: SMNs를 사용한 결과, 레이블이 적은 다운스트림 태스크에서 성능이 크게 향상되는 것을 확인했습니다. 이를 통해 모델이 레이블 효율적인 상황에서도 유용하게 사용될 수 있음을 보여주었습니다.



### FaVoR: Features via Voxel Rendering for Camera Relocalization (https://arxiv.org/abs/2409.07571)
Comments:
          Submitted to the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Tucson, Arizona, US, Feb 28-Mar 4, 2025

- **What's New**: FaVoR는 2D 특징의 전역적으로 희소하지만 지역적으로 조밀한 3D 표현을 활용하여 카메라 위치 추정의 정확성을 크게 향상시키는 새로운 접근 방법을 제안합니다.

- **Technical Details**: FaVoR는 볼륨 렌더링(volumetric rendering)을 사용하여 3D 랜드마크를 추적하고 삼각측량(triangulation)하여 희소한 복셀 맵을 구성합니다. 이는 카메라에서 관찰되는 이미지 패치 설명자를 최적화하여 생성합니다.

- **Performance Highlights**: 7-Scenes와 Cambridge Landmarks 데이터셋에서 평가한 결과, FaVoR는 기존의 최첨단 특징 표현 기법보다 실내 환경에서 최대 39%의 중위 변환 오류 개선을 보여주었고 메모리와 계산 비용 또한 낮게 유지했습니다.



### EchoDFKD: Data-Free Knowledge Distillation for Cardiac Ultrasound Segmentation using Synthetic Data (https://arxiv.org/abs/2409.07566)
- **What's New**: 본 연구에서는 심장 초음파 비디오에 대한 머신러닝의 적용이 진전을 이루고 있으며, 이는 최근 대규모 공개 데이터셋의 출현 덕분입니다.

- **Technical Details**: 전통적인 supervisedered tasks, 예를 들어 ejection fraction의 회귀(regression) 대신, 데이터 분포의 잠재적 구조(latent structure)에 중점을 두고, 생성적 방법(generative methods)을 채택하고 있습니다. 모델은 teacher model이 제안하는 masks를 통해 지식 증류(knowledge distillation)로 학습되며, 실제 또는 합성(synthetic) 데이터에 적용됩니다.

- **Performance Highlights**: 우리의 방법은 end-diastolic 및 end-systolic 프레임 식별(task)에서 최첨단(SOTA) 결과를 달성하였으며, 합성 데이터에 대한 학습만으로도 실제 데이터에서 학습한 성능에 근접한 세그멘테이션(segmentation) 능력을 보여줍니다. 5개의 주요 방법과 비교했을 때, 우리의 방법은 대부분의 경우에서 다른 방법들보다 우수함을 보였습니다.



### Unsupervised Point Cloud Registration with Self-Distillation (https://arxiv.org/abs/2409.07558)
Comments:
          Oral at BMVC 2024

- **What's New**: 본 논문은 기존의 ground truth poses 수집의 한계를 극복하기 위해 self-distillation 접근 방식을 사용하여 포인트 클라우드 등록(Point Cloud Registration)을 비지도 방식으로 학습하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 teacher 네트워크와 student 네트워크를 통한 self-distillation 기법을 사용합니다. teacher는 학습 가능한 feature extractor와 robust solver(RANSAC)을 포함하여, 일관성을 유지하고 비지도 데이터의 inlier 비율을 최적화합니다. 이 과정에서 초기 수동 특성이나 연속적인 포인트 클라우드 프레임에 대한 의존성이 제거되어 훈련 절차가 간소화됩니다.

- **Performance Highlights**: 제안된 방법은 RGB-D 벤치마크 3DMatch에서 SGP 및 EYOC보다 우수한 성능을 보여주며, 자동차 레이더 데이터에 대해서도 잘 일반화되는 성능을 보입니다. 또한, 하이퍼파라미터 조정이 덜 필요하여 효율적입니다.



### ENACT: Entropy-based Clustering of Attention Input for Improving the Computational Performance of Object Detection Transformers (https://arxiv.org/abs/2409.07541)
- **What's New**: 이번 연구에서는 시각 기반 객체 탐지에서 Transformers의 효율성을 개선하기 위해, 입력을 클러스터링(clustering)하는 새로운 접근 방식을 제안합니다. 특히, 각 픽셀의 자정보다 피크 정보를 기반으로 한 ENACT 모듈을 도입하여 연산 비용을 줄이고 훈련 시간을 단축시킵니다.

- **Technical Details**: 제안된 ENACT 모듈은 각 Key 픽셀의 자정보(self-information)를 기반으로 클러스터링을 수행합니다. 이 과정은 Softmax 함수를 사용하여 확률 밀도 함수를 추정하고, 픽셀의 자정보를 계산한 후 두 번째 도함수의 곡률을 토대로 클러스터링을 진행합니다. 이 방법은 특성 벡터의 유클리드 거리 대신 Shannon 엔트로피를 사용하여 특징의 유사성을 측정합니다.

- **Performance Highlights**: COCO 객체 탐지 데이터셋을 사용하는 실험에서, ENACT 모듈을 사용한 3개의 탐지 Transformers 모델이 평균 20%에서 40%의 GPU 메모리 사용량을 절감하고, 5%에서 15%의 훈련 시간을 감소시키는 동시에 탐지 정확도는 거의 유지됨을 보여주었습니다.



### Small Object Detection for Indoor Assistance to the Blind using YOLO NAS Small and Super Gradients (https://arxiv.org/abs/2409.07469)
- **What's New**: 이번 연구는 시각 장애인을 위한 실내 보조 기술을 위한 새로운 접근법을 제시하고 있으며, 특히 작은 물체 탐지 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 방법은 YOLO NAS Small 아키텍처를 기반으로 하며, Super Gradients 트레이닝 프레임워크를 이용하여 최적화된 경량 물체 탐지 모델입니다. 이 조합은 소형 물체를 실시간으로 탐지할 수 있도록 해주며, 특히 가구, 가전 제품 및 가정용 물품 등에서 중요합니다.

- **Performance Highlights**: 이 방법은 낮은 대기 시간(low latency)과 높은 정확도(high accuracy)를 강조하여, 사용자에게 공간 인식(spatial awareness)과 주변 상호작용(interaction)에 대한 정보를 제공할 수 있는 음성 기반 가이드를 가능하게 합니다.



### Reflective Human-Machine Co-adaptation for Enhanced Text-to-Image Generation Dialogue System (https://arxiv.org/abs/2409.07464)
- **What's New**: 본 연구에서는 이미지 생성 시스템의 사용자 친화성을 개선하기 위한 'RHM-CAS'라는 반영적 인간-기계 협응 전략을 제안합니다. 이 전략은 사용자의 의도를 이해하고 이미지 결과물을 최적화하기 위해 여러 차례의 상호작용을 활용합니다.

- **Technical Details**: 제안된 방법은 모듈형 아키텍처로 구성되어 있으며, 기억(Memory), 요약기(Summarizer), 생성 모델(Generation Model), 반영 블록(Reflection Block), 평가자(Evaluator) 및 모호성 추론(Min-Inf) 구성 요소를 포함합니다. 이러한 구성 요소들은 사용자의 과거 대화를 저장하고, 생성할 이미지를 위해 적절한 프롬프트를 생성하며, 생성된 이미지를 평가하고 사용자의 의도를 파악하는 데 기여합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법의 효과성이 입증되었습니다. 일반 이미지 및 패션 이미지 생성 작업에서 우수한 성능을 보여, 사용자 요구를 충족시키는 이미지 결과물을 생성하는 데 유용합니다.



### Multi-Modal Instruction-Tuning Small-Scale Language-and-Vision Assistant for Semiconductor Electron Micrograph Analysis (https://arxiv.org/abs/2409.07463)
Comments:
          Paper published at AAAI 2024 Spring Symposium Series

- **What's New**: 이 논문에서는 반도체 제조에서 전자 현미경 이미지를 분석하고 해석하기 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 비전-언어 지침 튜닝(vision-language instruction tuning)을 사용하여 기존의 멀티모달 대형 언어 모델들을 활용하여 문서화된 데이터를 생성하고, 이를 통해 제로샷 비주얼 질문 응답(zero-shot visual question answering, VQA) 및 분류 작업을 수행하도록 맞춤형으로 설계된 소형 멀티모달 모델(smaller multimodal models, SMMs)을 개발하였습니다.

- **Technical Details**: 제안된 프레임워크는 교사-학생 방식(teacher-student approach)을 특징으로 하며, GPT-4 Turbo의 비전 기능을 이용해 질문-응답 쌍을 생성합니다. 각 레이블된 쌍은 질문 이미지와 관련된 텍스트 지침 및 가장 정확한 응답을 포함합니다. 이 프레임워크는 지침 기반 이미지 인코더와 텍스트 인코더를 활용하며, 다양한 손실 함수를 최적화하여 이미지와 언어 간의 정렬을 보장하는 복잡한 구조를 가집니다. 이 방식으로 텍스트 생성과 비쥬얼 추론의 grounded 언어 생성을 개선합니다.

- **Performance Highlights**: MVaEMa 모델은 제로샷 VQA 성능 향상에 크게 기여하며, 고해상도 전자 미세 이미지 분석에서의 정확도를 높이는 데 성공하였습니다. 이 모델은 특히 반도체 제조의 품질 보증을 위한 비주얼 질문 응답에 효과적이며, 큰 기업 데이터에 대한 세부 조정이 가능하여 사용자의 요구에 따라 효율성을 높일 수 있습니다.



### Hand-Object Interaction Pretraining from Videos (https://arxiv.org/abs/2409.08273)
- **What's New**: 본 논문은 3D 손-물체 상호작용 궤적을 통해 일반 로봇 조작 우선 순위를 학습하는 접근 방식을 제시합니다. 야생 비디오를 활용하여 센서모터 로봇 궤적을 생성하는 프레임워크를 구축하였습니다.

- **Technical Details**: 이 방법에서는 사람의 손과 조작된 물체를 공유 3D 공간에서 각각의 궤적을 맵핑하여 로봇의 동작으로 리타겟팅합니다. 사전 훈련된 정책은 센서모터 로봇 궤적에 대한 조건부 분포 일치를 목표로 하는 인과 변환기(causal transformer)의 가중치에 내재되어 있습니다.

- **Performance Highlights**: 사례 연구를 통해, 이 조작 우선 순위가 기존 방법보다 기술 습득을 상당히 가속화하며, 훈련 비디오에 없는 기술들도 잘 발휘되고 최종 정책의 일반화 및 견고성을 향상시킨다는 결과를 보여주었습니다.



### Model Ensemble for Brain Tumor Segmentation in Magnetic Resonance Imaging (https://arxiv.org/abs/2409.08232)
Comments:
          11 pages, 6 figures, 2 tables; This method ranked 1st, 3rd and 4th for BraTS2023 PED, MEN, and MET, respectively. This paper was accepted at MICCAI 2023's BrainLes Workshop

- **What's New**: 본 논문은 2023년 Brain Tumor Segmentation (BraTS) 챌린지에 새롭게 추가된 세 가지 작업, 즉 소아 뇌종양(PED), 두개 내 수막종(MEN), 뇌 전이종양(MET)을 다루며 최신 딥러닝 모델을 통한 앙상블(ensemble) 전략을 소개합니다. 특히 nnU-Net과 Swin UNETR 모델의 출력을 조합하여 세분화된 성과를 보여주었습니다.

- **Technical Details**: 논문에서는 두 가지 최첨단 모델(nnU-Net, Swin UNETR)을 앙상블하여 사용합니다. nnU-Net은 자가 구성되는 세분화 프레임워크로, 각 데이터셋의 특성에 맞춰 조정됩니다. Swin UNETR은 비전 변환기 기반의 구조로, 다중 스케일 작업에 적합한 로컬 윈도우 자기 주의를 활용합니다. 각 작업에 대해 완전 해상도 3D nnU-Net 및 Swin UNETR 모델을 5겹 교차 검증 방식으로 훈련시켰습니다.

- **Performance Highlights**: 제안된 방법은 PED에서 0.653, 0.809, 0.826의 Dice 점수를 기록하였고, MEN에서는 0.876, 0.867, 0.849, MET에서는 0.555, 0.6, 0.58의 점수를 기록하여 각각 첫 번째, 세 번째, 네 번째로 평가되었습니다.



### AD-Lite Net: A Lightweight and Concatenated CNN Model for Alzheimer's Detection from MRI Images (https://arxiv.org/abs/2409.08170)
Comments:
          NA

- **What's New**: 이번 연구에서는 경량의 CNN 모델인 AD-Lite Net을 개발했습니다. 이 모델은 Depth Wise Separable Convolutional (DWSC) 레이어와 Global Average Pooling (GAP) 레이어를 포함하여 경량화되었습니다. 또한, 병렬 연결 블록(parallel concatenation block, pcb)을 활용하여 모델의 성능을 향상시켰습니다.

- **Technical Details**: AD-Lite Net 모델은 Tx-layer를 도입하여 MRI 이미지의 특징을 전환하는 방법을 사용하여 클래스 불균형 문제를 잘 해결하였습니다. 이 모델은 3개의 서로 다른 MRI 데이터셋에서 테스트되었고, 10-겹 교차검증(10-fold cross-validation)을 통해 일반화 성능을 검증하였습니다. 실험 결과, 다른 기존 CNN 모델과 최근의 Vision Transformer (ViT) 모델보다 더 뛰어난 성과를 보였습니다.

- **Performance Highlights**: AD-Lite Net은 MRI 데이터셋에서 Alzheimer’s Disease를 조기 탐지하는 데 있어 높은 정확도를 기록했습니다. 특히, 기존 모델들과 비교하였을 때 98-99%의 정확도로 AD 탐지 문제를 해결하여, 신경과 의사들의 진단 과정에서 시간을 크게 단축시킬 수 있을 것으로 기대됩니다.



### Open Source Infrastructure for Automatic Cell Segmentation (https://arxiv.org/abs/2409.08163)
- **What's New**: 본 연구에서는 UNet 모델을 활용하여 자동화된 세포 분할을 위한 오픈 소스 인프라를 제안합니다. 이 구현은 DeepChem 패키지에 통합되어 연구자와 실무자들이 더욱 접근하기 쉽고 사용 편리하게 만들어졌습니다.

- **Technical Details**: 이 모델은 깊은 학습(d deep learning) 아키텍처인 UNet을 기반으로 하며, 이미지 분할(image segmentation) 작업의 효과성을 높이기 위해 설계되었습니다. DeepChem의 ImageLoader, ImageDataset 및 ImageTransformer 클래스를 통해 이미지 데이터를 불러오고 전처리하는 과정을 간소화합니다.

- **Performance Highlights**: 모델 성능은 다양한 오픈 소스 데이터셋을 통해 평가되었습니다. Intersection over Union (IoU), F1 Score, Area under ROC (AuROC)와 같은 지표를 사용하여 모델의 정확성과 일반화를 입증하였습니다. BBBC003과 BBBC039 데이터셋에 대한 실험 결과, 좋은 성능을 나타냈습니다.



### Effective Segmentation of Post-Treatment Gliomas Using Simple Approaches: Artificial Sequence Generation and Ensemble Models (https://arxiv.org/abs/2409.08143)
Comments:
          Invited for an Oral Presentation at the MICCAI BraTS Challenge 2024

- **What's New**: 이 논문은 BraTS 2024 Challenge의 새로운 데이터셋을 활용하여 수술 후로 치료된 신경아교종(glioma) 분할(segmentation)의 복잡성을 해결하기 위한 자동화 도구 개발을 촉진하는 방향으로 연구하고 있습니다. 두 가지 접근 방식을 제안하며, 하나는 MRI 시퀀스의 선형 결합을 기반으로 하는 추가 입력을 통합하는 것이고, 다른 하나는 여러 모델의 조합을 통해 성능 개선을 달성하는 것입니다.

- **Technical Details**: 연구에서는 약 2,200명의 환자 MRI 스캔을 사용하였으며, T1, T1Gd, T2, T2 FLAIR와 같은 다양한 이미징 모달리티를 포함하고 있습니다. nnUNet, nnUNet ResEnc, SegResNet 등 다양한 네트워크 아키텍처를 사용하여 모델을 구축하였고, 데이터 증강(data augmentation) 기법도 활용하였습니다. 또한, T1Gd-T1 이미지를 추가 입력으로 사용하여 종양을 더욱 잘 강조하는 방식으로 접근하였습니다.

- **Performance Highlights**: 이 접근법은 기존 모델에 비해 분할 성능을 유의미하게 향상시켰습니다. 특히, 간단한 방법으로도 전문의와의 협력을 통해 수술 후 신경아교종의 효과적인 분할을 보여주며, 이러한 성과는 의료 이미지 분석의 발전 가능성을 시사합니다.



### The JPEG Pleno Learning-based Point Cloud Coding Standard: Serving Man and Machin (https://arxiv.org/abs/2409.08130)
Comments:
          28 pages, 12 figures, submitted to IEEE Access

- **What's New**: 최근 JPEG가 JPEG Pleno Learning-based Point Cloud Coding (PCC) 표준을 최종 확정하였습니다. 이 표준은 깊이있는 학습 모델을 활용하여 정적 포인트 클라우드의 효율적인 손실 압축을 목표로 하고 있습니다.

- **Technical Details**: 이 논문에서는 JPEG PCC 표준의 완전한 기술 설명과 현재의 최첨단 기술과의 성능 비교를 포함하고 있으며, 기하학(geometry)와 색상(color) 코딩을 위한 희소 합성곱 신경망(sparse convolutional neural networks)을 사용하여 3D 형태의 기하학을 직접 처리합니다. 색상 데이터는 2D 이미지를 통해 투영되고, 학습 기반의 JPEG AI 표준을 이용해 인코딩됩니다.

- **Performance Highlights**: 압축 성능 면에서 JPEG PCC는 기존의 MPEG PCC 표준을 초월하며, 특히 기하학 코딩에서 상당한 비율 감소를 달성하였습니다. 색상 압축 성능은 다소 낮지만, 기하학과 색상 모두에 대한 전면적인 학습 기반 코딩 프레임워크의 힘으로 이를 극복하고 있습니다.



### GAZEploit: Remote Keystroke Inference Attack by Gaze Estimation from Avatar Views in VR/MR Devices (https://arxiv.org/abs/2409.08122)
Comments:
          15 pages, 20 figures, Accepted at ACM CCS'24

- **What's New**: 가장 최신의 눈 추적 기반의 공격 방법인 GAZEploit이 개발되어 VR/MR 디바이스에서 입력한 민감한 키 스트로크 정보를 원격으로 추출할 수 있음을 밝혀냈습니다.

- **Technical Details**: GAZEploit 공격은 사용자의 가상 아바타를 통해 눈 추적 데이터를 유출하여 텍스트 입력 정보를 재구성하는 방식으로 작동합니다. 본 연구에서는 30명의 참가자를 대상으로 실험을 수행하였으며, 85.9%의 정밀도와 96.8%의 재현율을 달성했습니다.

- **Performance Highlights**: 메시지, 비밀번호, 패스코드 입력에 대해 각각 92.1%, 77.0%, 73.0%의 높은 성과를 기록했으며, VR/MR 디바이스의 보안 취약성을 강조했습니다.



### AutoPET Challenge: Tumour Synthesis for Data Augmentation (https://arxiv.org/abs/2409.08068)
Comments:
          9 pages

- **What's New**: 본 연구에서는 제한된 데이터셋으로 인해 발생하는 자동화된 병변 세분화 모델의 성능 저하를 극복하기 위해 생성 모델(Generative Model)에서 파생된 딥 프라이어(Deep Prior)를 활용한 데이터 증강(data augmentation) 기술을 탐구하였습니다.

- **Technical Details**: DiffTumor 방법론을 PET-CT 이미지에 맞게 조정하여 합성 PET-CT 이미지를 생성하고, 이를 통해 AutoPET 데이터셋을 확장하였습니다. 이 과정에서 DynUnet 모델을 사용하여 기존 데이터셋과 증강된 데이터셋에서 세분화 모델의 성능을 비교했습니다.

- **Performance Highlights**: 증강된 데이터셋에서 훈련된 모델은 더 높은 Dice 점수를 달성하여, 데이터 증강 접근 방식의 가능성을 입증했습니다. 주목할 만한 점은, 증강된 데이터셋을 사용한 모델과 기존 데이터셋을 사용한 모델 간의 성능 차이가 있었고, 전반적으로 데이터 증강이 성능 향상에 기여함을 나타냈습니다.



### OCTAMamba: A State-Space Model Approach for Precision OCTA Vasculature Segmentation (https://arxiv.org/abs/2409.08000)
Comments:
          5 pages, 2 figures

- **What's New**: 이 논문에서는 OCTA(Optical Coherence Tomography Angiography) 이미지에서 혈관을 정확하게 분할하기 위해 Mamba 아키텍처를 기반으로 한 새로운 U자형 네트워크인 OCTAMamba를 제안합니다.

- **Technical Details**: OCTAMamba는 Quad Stream Efficient Mining Embedding (QSEME) 모듈, Multi-Scale Dilated Asymmetric Convolution Module (MSDAM), Focused Feature Recalibration Module (FFRM)를 통합하여 OCTA 혈관을 정밀하게 분할합니다. 이 네트워크는 지역 특징을 효과적으로 추출하고 멀티스케일 혈관 구조를 캡처하며, 높은 계산 효율성을 가지고 있습니다.

- **Performance Highlights**: OCTA 3M, OCTA 6M, ROSSA 데이터셋에서의 실험 결과, OCTAMamba는 기존의 최첨단 방법들보다 우수한 성능을 보이며, OCTA 분할을 위한 효율적인 새로운 기준을 제공합니다.



### InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation (https://arxiv.org/abs/2409.07914)
Comments:
          Accepted at Conference on Robot Learning (CoRL) 2024

- **What's New**: InterACT: Hierarchical attention 기반의 이펙타치기(Action Chunking) 프레임워크를 소개합니다. 이 프레임워크는 이양 손 조인트 상태와 시각적 입력 간의 상호 의존성을 포착하여 이양 조작(bimanual manipulation)을 위한 새로운 모방 학습(imitation learning) 방식을 제공합니다.

- **Technical Details**: InterACT은 Hierarchical Attention Encoder와 Multi-arm Decoder로 구성되어 있으며, 세그먼트 단위의 인코딩과 크로스 세그먼트 인코딩을 통해 다중 모드 입력을 처리합니다. 세그먼트 인코더는 각 세그먼트 내의 정보 집합을 추구하여 상호 의존성을 캡처합니다. 크로스 세그먼트 인코더는 여러 세그먼트 간의 관계를 포착하여 각 팔이 연속적이고 효율적으로 협력할 수 있도록 합니다.

- **Performance Highlights**: 다양한 시뮬레이션 및 실제 세계의 이양 조작 작업에 대한 실험 결과, InterACT은 기존 방법보다 탁월한 성능을 보여줍니다. 중요한 구성 요소인 CLS 토큰, 크로스 세그먼트 인코더 및 동기화 블록의 기여도를 평가하는 상세한 제거 연구(ablation studies)가 수행되었습니다.



### Context-Aware Optimal Transport Learning for Retinal Fundus Image Enhancemen (https://arxiv.org/abs/2409.07862)
- **What's New**: 본 논문에서 제안하는 방법은 context-informed optimal transport (OT) 학습 프레임워크를 통해 비표준 fundus 이미지 향상을 다루는 것을 목표로 하며, 기존의 generative 이미지 향상 기법이 해결하기 어려운 문맥 정보 처리를 향상시킵니다.

- **Technical Details**: 제안된 방법은 low-quality 이미지와 high-quality 이미지 간의 1:1 매핑을 찾아내며, deep contextual features를 활용하여 earth mover’s distance를 기반으로 합니다. 이는 높은 구조적 유사성과 신뢰성을 보장합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법이 신호 대 잡음 비율(signal-to-noise ratio), 구조적 유사성 지수(structural similarity index) 및 여러 후행 작업에서 여러 최신 방법들보다 우수한 성능을 보임을 입증하였습니다.



### ReGentS: Real-World Safety-Critical Driving Scenario Generation Made Stab (https://arxiv.org/abs/2409.07830)
Comments:
          Accepted to ECCV 2024 W-CODA Workshop

- **What's New**: 이 연구는 안전-critical 운전 시나리오를 실시간 데이터 기반으로 생성하며, 복잡한 실제 시나리오를 통해 시나리오 생성을 최적화하는 새로운 접근 방식인 ReGentS를 제안합니다.

- **Technical Details**: ReGentS는 생성된 궤적을 안정화시키고 이야기의 가시성과 충돌 위험을 줄이는 휴리스틱을 도입합니다. 이 방법은 실제 시뮬레이터와의 결합으로 방향이 정해진 경량 형태의 최적화 프로세스를 간소화합니다.

- **Performance Highlights**: ReGentS는 기존 방법들보다 더 안정적인 궤적을 제공하며, 특히 안전-critical 시나리오의 생성을 통해 머신러닝 기반 플래너의 추가 학습에 유용하게 사용할 수 있습니다.



### Bridging Paintings and Music -- Exploring Emotion based Music Generation through Paintings (https://arxiv.org/abs/2409.07827)
- **What's New**: 이 연구는 시각 예술의 감정을 음악으로 변환하는 새로운 모델을 개발하였습니다. 감정 레이블링, 이미지 캡셔닝(image captioning), 그리고 언어 모델을 통합하여 시각 입력을 음악 작곡으로 변환하는 데 중점을 두었습니다.

- **Technical Details**: Emotion Painting Music Dataset을 구축하여 그림과 해당 음악을 짝지어 효과적인 훈련과 평가를 가능하게 하였습니다. 이중 단계 프레임워크는 이미지를 감정적 내용을 기술하는 텍스트 설명으로 변환한 다음, 이러한 설명을 음악으로 변환합니다. Fréchet Audio Distance (FAD), Total Harmonic Distortion (THD), Inception Score (IS), KL divergence와 같은 지표를 사용하여 성능을 평가하였습니다.

- **Performance Highlights**: 생성된 음악과 텍스트 간의 높은 일치를 확인하였으며, 시각 장애인을 위한 접근성을 향상시키고 교육 및 치료적 응용 분야에서 새로운 다감각 경험을 제공할 수 있는 가능성을 보였습니다.



### SwinGS: Sliding Window Gaussian Splatting for Volumetric Video Streaming with Arbitrary Length (https://arxiv.org/abs/2409.07759)
- **What's New**: 최근 3D Gaussian Splatting (3DGS)의 발전이 컴퓨터 비전 및 그래픽스 분야에서 주목받고 있으며, 본 논문에서는 SwinGS라는 새로운 프레임워크를 소개합니다. SwinGS는 실시간으로 볼륨 비디오를 전달하고 렌더링할 수 있도록 설계되었습니다.

- **Technical Details**: SwinGS는 spacetime Gaussian과 Markov Chain Monte Carlo (MCMC)를 통합하여 모델이 다양한 3D 장면에 맞게 조정됩니다. 또한, sliding window 기법을 통해 각 프레임에 대한 Gaussian 스냅샷을 누적적으로 캡처합니다.

- **Performance Highlights**: 실험 결과, SwinGS는 이전 연구에 비해 전송 비용을 83.6% 줄이는 동시에 PSNR 품질 저하를 무시할 수 있는 수준으로 유지합니다. 또한, SwinGS는 긴 비디오 시퀀스에도 품질 저하 없이 쉽게 확장 가능합니다.



### Object Depth and Size Estimation using Stereo-vision and Integration with SLAM (https://arxiv.org/abs/2409.07623)
Comments:
          Accepted version of the published article in IEEE Sensors Letters

- **What's New**: 이 논문에서는 자율 로봇을 위한 SLAM(Simultaneous Localization and Mapping) 시스템에 LiDAR를 보완하기 위해 고도로 정확한 스테레오 비전 방법을 제안합니다. 이 시스템은 인간의 시각적 인식을 모방하면서 유용한 물체 탐지를 통해 유무형 객체의 깊이와 크기를 추정합니다.

- **Technical Details**: 제안된 시스템은 두 대의 수평으로 배치된 카메라를 사용하여 환경의 이미지를 동시에 캡처하고, YOLOv8을 통한 심층 학습 기반의 물체 탐지를 통해 장애물을 탐지합니다. 이후 각 객체의 픽셀 오프셋을 통해 깊이와 크기를 계산하여 SLAM 프로세스에 통합합니다.

- **Performance Highlights**: 평가 결과, LiDAR와 스테레오 비전 시스템이 장착된 자율 로봇에서 객체의 깊이와 크기에 대한 추정에서 높은 정확도를 보였으며, 실제 로봇 내비게이션 시스템에서 장애물 회피와 안전한 내비게이션에 효과적입니다.



### A Cost-Aware Approach to Adversarial Robustness in Neural Networks (https://arxiv.org/abs/2409.07609)
- **What's New**: 이번 연구는 생성 AI의 대두와 함께 적대적 공격(adversarial attacks)에 대한 모델의 강인성을 평가하는 새로운 방법론을 제시합니다. 기존의 대량의 테스트 세트를 사용하는 방법에서는 적절한 성능 보장을 할 수 없기 때문에, 생존 분석(survival analysis)을 활용하여 극단적인 경우를 시뮬레이션하고 적은 양의 샘플로 모델을 검증하는 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 가속 실패 시간 모델(Accelerated Failure Time models, AFT)을 사용하여 하드웨어 선택, 배치 크기(batch size), 에폭 수(number of epochs), 테스트 세트 정확도(test-set accuracy)의 영향을 측정합니다. 또한, Tree Parzen Estimator를 활용하여 모델의 강인성을 극대화하고 모델의 실행 시간을 동시에 최소화하는 방법론을 제공합니다.

- **Performance Highlights**: 신형 하드웨어를 사용할 때 훈련 시간이 단축되는 것은 사실이나, 그에 따른 비용과 전력 소모는 정확도의 한계 이득을 초과하는 것으로 나타났습니다. 본 연구는 훈련과 평가 과정에서 다양한 하이퍼 파라미터의 효과를 추정하며, 모델의 강인성과 비용 간의 균형을 도출하는 방법론을 제시하였습니다.



### DS-ViT: Dual-Stream Vision Transformer for Cross-Task Distillation in Alzheimer's Early Diagnosis (https://arxiv.org/abs/2409.07584)
Comments:
          8 pages, 3 figures, 3 tables

- **What's New**: 이 연구에서는 알츠하이머병 진단을 위한 자원 효율을 극대화하고 분류 성능을 향상시키기 위해 세분화(segmentation)와 분류(classification) 작업 간의 지식 공유를 촉진하는 새로운 이중 스트림 파이프라인(Dual-Stream Pipeline)을 제안합니다.

- **Technical Details**: 제안된 DS-ViT(Dual-Stream Vision Transformer) 파이프라인은 FastSurfer(CNN 기반 모델)와 ADAPT(Vision Transformer 기반 모델) 간의 교차 작업 및 아키텍처 지식 공유를 허용합니다. 이중 스트림 임베딩 모듈을 통해 세분화 및 분류 모델의 특징 표현을 통합하고, 3D Bottleneck MLP 구조를 활용하여 이 정보를 이용해 분류 모델을 유도합니다. 또한, 잔여 시간 주의 블록(Residual Temporal Attention Block, RTAB)을 추가하여 시퀀스 MRI 이미지의 특성 맵 간의 차이를 분석, 조기 진단을 지원합니다.

- **Performance Highlights**: 여러 3D 데이터셋을 통해 검증한 결과, DS-ViT는 ADAPT 모델에 비해 평균 7%의 분류 정확도 향상과 훈련 시간 절반 단축을 달성했습니다. RTAB로 확장된 DS-ViT 파이프라인은 전체 70%의 예측 정확도를 보였으며, 고신뢰 샘플에서는 86%의 정확도를 기록하여 조기 유병 단계에서 개입 가능성을 제시하고 있습니다.



### TabMixer: Noninvasive Estimation of the Mean Pulmonary Artery Pressure via Imaging and Tabular Data Mixing (https://arxiv.org/abs/2409.07564)
Comments:
          Accepted for the 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024

- **What's New**: 본 논문에서는 비침습적 심장 자기 공명 영상(Cardio Magnetic Resonance Imaging, CMR)에서 평균 폐동맥압(mean Pulmonary Artery Pressure, mPAP)을 추정하는 새로운 접근법을 제안합니다. 이를 위해 인구통계적 특성과 임상 측정 결과를 추가적인 모달리티로 도입하여 딥러닝 모델의 예측 능력을 향상시키고 있습니다.

- **Technical Details**: 연구팀은 TabMixer라는 새로운 모듈을 소개합니다. 이 모듈은 MLP(Multi-Layer Perceptron) 아키텍처를 활용하여 영상진단 정보와 표 형식 데이터를 공간적(spatial), 시간적(temporal), 채널(channel) 차원에서 혼합할 수 있게 합니다. TabMixer는 CNN(Convolutional Neural Networks), 3D-MLP 및 Vision Transformer와 통합되어 mPAP 추정에서 성능을 개선하며, 기존 모듈들과의 경쟁력을 보여줍니다.

- **Performance Highlights**: TabMixer를 통해 비침습적 mPAP 추정의 정확성을 높이며, 폐고혈압(Pulmonary Hypertension) 환자들에게 삶의 질 향상에 기여할 수 있는 가능성을 제시합니다. 본 접근법은 의료 과정에서 두 가지 모달리티(영상 및 표 형식 데이터)를 결합하는 혁신적인 방법으로, 특히 CMR 비디오를 통한非침습적 mPAP 추정의 첫 사례로 평가받고 있습니다.



### Complex Emotion Recognition System using basic emotions via Facial Expression, EEG, and ECG Signals: a review (https://arxiv.org/abs/2409.07493)
Comments:
          29 pages, 11 figures

- **What's New**: 이 논문에서는 복합 감정 인식 시스템(Complex Emotion Recognition System, CERS)의 개발과 그로 인한 정서 역학에 대한 깊은 통찰을 제공하며, 머신러닝과 딥러닝을 활용한 정서 인식의 최신 동향을 조망합니다.

- **Technical Details**: CERS는 기본 감정의 조합, 상호 연결 및 동적 변화를 분석하여 복합 감정 상태를 해독합니다. 심전도(ECG) 및 뇌파(EEG)와 같은 생리적 신호를 통합함으로써 사용자 정서 상태에 대한 귀중한 통찰을 제공하고, 데이터셋의 품질을 향상시키며 시스템의 신뢰성을 강화합니다. 또한, 머신러닝, 딥러닝, 메타 학습(meta-learning) 방법의 효율성을 평가하기 위한 포괄적인 문헌 조사가 수행되었습니다.

- **Performance Highlights**: CERS의 성능을 개선하기 위한 메타 학습 접근 방법의 중요성이 강조되며, 복합 감정 인식에 대한 연구의 격차 및 도전과제를 밝히고 추가 연구를 장려하고 있습니다.



### An Artificial Neural Network for Image Classification Inspired by Aversive Olfactory Learning Circuits in Caenorhabditis Elegans (https://arxiv.org/abs/2409.07466)
- **What's New**: 이 연구는 유사한 신경 회로를 모방한 인공 신경망(ANN)을 활용하여 이미지 분류 작업에서 우수한 성능을 보이는 새로운 접근법을 제안합니다. 모델은 Caenorhabditis elegans의 가소성 후각 학습 메커니즘에서 영감을 받아 설계되었습니다.

- **Technical Details**: 제안된 ANN은 C. elegans의 모양과 기능을 본따 만들어졌으며, 121개의 기능적 신경 뉴런으로 구성되어 있습니다. 연구에서는 고속 유전자 시퀀싱과 행동 실험을 통해 후각 자극을 피하는 학습 메커니즘을 규명하였습니다.

- **Performance Highlights**: 비교 분석 결과, 제안된 ANN은 기존의 두 가지 아키텍처를 기반으로 한 ANN보다 이미지 분류 작업에서 더 높은 정확도, 일관성 및 빠른 수렴 속도를 보였습니다.



### LSST: Learned Single-Shot Trajectory and Reconstruction Network for MR Imaging (https://arxiv.org/abs/2409.07457)
- **What's New**: 본 연구는 단일 샷 자기 공명 (Magnetic Resonance, MR) 이미징에서 k-공간 (k-space) 데이터 수집의 속도를 높이고 T2-블러 (T2-blur)를 줄이는 새로운 방법을 제안합니다. 이를 위해 샘플 수를 줄이고 k-공간 투사 최적화에 초점을 맞추며, 전체 이미지의 재구성 품질을 향상시키기 위해 물리학적 제약을 준수합니다.

- **Technical Details**: 이 연구에서 제안된 방법은 (a) k-공간 측정을 위한 궤적을 최적화하고, (b) 샘플 수를 줄여 수집 속도를 증가시키며, (c) T2-블러의 영향을 줄입니다. 또한, T2-블러를 고려하여 재구성 품질을 향상시키기 위해 Density Compensation Factor (DCF)를 훈련 중 통합하고, 초기 입력에 모델 기반 솔루션을 사용합니다. k-공간에서 루틴 궤적을 생성하기 위해 Traveling Salesman Problem (TSP) 접근법을 사용하며, 이를 통해 적합한 궤적을 최적화하는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과는 경험 많은 방사선과의 평가를 바탕으로, 제안된 방법이 비교 방법보다 ACL 섬유(Anterior Cruciate Ligament fibers)에 대해 더 선명한 재구성을 달성했음을 보여줍니다. 특히 8배 및 16배 가속 요소를 사용한 fastMRI 멀티채널 데이터셋에서 검증되었습니다.



New uploads on arXiv(cs.AI)

### Windows Agent Arena: Evaluating Multi-Modal OS Agents at Sca (https://arxiv.org/abs/2409.08264)
- **What's New**: 대규모 언어 모델(LLMs)의 가능성을 활용하여 Windows 운영 체제(OS) 내에서 에이전트를 테스트하는 Windows Agent Arena를 소개합니다. 이는 멀티 모달(multi-modal) 작업에서 계획 및 추론을 통해 인간의 생산성을 높이고 소프트웨어 접근성을 향상시키기 위해 설계되었습니다.

- **Technical Details**: Windows Agent Arena는 150개 이상의 다양한 Windows 작업으로 구성되어 있으며, 에이전트의 계획, 화면 이해, 도구 사용 능력을 요구합니다. 이 벤치마크는 Azure에서 병렬 컴퓨팅이 가능하여 전체 평가가 단 20분 이내에 완료될 수 있도록 설계되었습니다. 새로운 에이전트인 Navi는 Windows 도메인에서 19.5%의 성공률을 기록했습니다.

- **Performance Highlights**: Navi는 Mind2Web와 같은 웹 기반 벤치마크에서도 우수한 성능을 보여 주며, 에이전트의 개발 및 데이터 생성에 대한 향후 연구 기회를 제공하기 위한 다양한 정량적 및 정성적 분석이 포함되어 있습니다.



### TravelAgent: An AI Assistant for Personalized Travel Planning (https://arxiv.org/abs/2409.08069)
- **What's New**: TravelAgent라는 새로운 여행 계획 시스템을 소개합니다. 이 시스템은 대형 언어 모델(LLMs)을 기반으로 하여, 동적인 여행 시나리오에서 합리적이고, 포괄적이며 개인화된 여행 일정을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: TravelAgent는 Tool-usage, Recommendation, Planning, Memory 모듈의 네 가지로 구성됩니다. 각 모듈은 원활하게 협력하며, 여행 제약 사항을 모델링하여 계획 단계의 합리성을 향상시키고, 실시간 도구를 활용하여 포괄성을 해결하며, 개인화를 증진시키기 위한 새로운 추천 프레임워크를 제공합니다.

- **Performance Highlights**: TravelAgent는 인간 사용자와 시뮬레이션된 사용자 평가를 통해 Rationality(합리성), Comprehensiveness(포괄성), Personalization(개인화) 기준에서 높은 성능을 발휘하였으며, 기존의 GPT-4+ 에이전트보다 우수한 성능을 보였습니다.



### Games for AI Control: Models of Safety Evaluations of AI Deployment Protocols (https://arxiv.org/abs/2409.07985)
Comments:
          7 pages, with appendices

- **What's New**: 이번 논문은 AI-Control Games라는 새로운 모델을 도입하여 비신뢰 AI 시스템을 안전하게 배포하기 위한 프로토콜을 평가합니다. 게임 이론을 적용하여 멀티 목적, 부분 관측 가능한 확률적 게임으로 형식화하였고, 이를 통해 비신뢰 언어 모델을 프로그래밍 보조 도구로 활용하기 위한 Trusted Monitoring 프로토콜을 제안합니다.

- **Technical Details**: 이 논문에서는 비신뢰 AI 시스템(U)을 안전하게 배포하기 위한 다양한 AI-Control 프로토콜을 연구하며, 비신뢰 AI 시스템이 작동하는 방식에 대한 최악의 시나리오를 시뮬레이션하기 위해 인간 레드팀 프로세스를 도입합니다. AI-Control Games는 부분 관측 가능한 멀티 목적 게임으로 정의되며, 이를 통해 여러 프로토콜의 안전성과 유용성을 평가하고 향상시킬 수 있는 알고리즘을 제시합니다.

- **Performance Highlights**: 본 연구에서는 Trusted Monitoring 프로토콜을 활용하여 시스템의 안전성과 유용성을 정량화하고, 기존의 경험적 연구에 비해 개선된 결과를 나타냈습니다. 또한, 안전성과 유용성 간의 트레이드오프를 분석하고, 여러 새로운 설정에서 프로토콜을 평가하여 Pareto 개선을 증명했습니다.



### Autonomous Vehicle Controllers From End-to-End Differentiable Simulation (https://arxiv.org/abs/2409.07965)
- **What's New**: 이번 연구에서는 자율주행차(AV) 컨트롤러 학습을 위한 새로운 방법을 제안합니다. 기존 Behavioral Cloning 방법이 이력 데이터에만 의존하여 일반화에 어려움을 겪는 반면, 우리는 차별화 가능한 시뮬레이터를 활용하여 Analytic Policy Gradients (APG) 방식으로 대규모 Waymo Open Motion Dataset에서 학습을 진행합니다.

- **Technical Details**: 탐색해야 할 동적 정보를 명시적으로 통합하여 기법 효율성을 높이는 차별화 가능한 시뮬레이터를 통해 학습합니다. 우리의 APG 기법은 시간 정보를 효율적으로 전파할 수 있는 순환 신경망 아키텍처(Recurrent Architecture)를 결합하여 긴 시뮬레이션 경로 상에서 작동합니다. 우리는 전문가의 경로를 사용하여 정책을 직접 최적화하며, 환경의 역학을 통해 경량화된 접근 방식을 채택합니다.

- **Performance Highlights**: 우리의 접근 방식은 Behavioral Cloning과 비교했을 때, 성능의 향상과 더불어 동적 노이즈에 대한 강력함이显현되었습니다. 결과적으로 인간처럼 직관적인 핸들링을 보여주는 더 빠르고 정확한 정책을 학습하는 데 성공했습니다.



### A Spatiotemporal Stealthy Backdoor Attack against Cooperative Multi-Agent Deep Reinforcement Learning (https://arxiv.org/abs/2409.07775)
Comments:
          6 pages, IEEE Globecom 2024

- **What's New**: 이번 연구에서는 단일 에이전트에 백도어(Backdoor)를 삽입하여 전체 다중 에이전트 팀에 대한 공격을 수행하는 새로운 기법을 제안합니다. 기존의 백도어 공격 방식에서는 시각적 패턴 또는 특정 행동 상태에 의존하는 경우가 많았으나, 본 연구에서는 시공간 행동 패턴(Spatiotemporal behavior patterns)을 사용하여 스텔스(stealthiness)와 실용성을 보장합니다.

- **Technical Details**: 이 공격 방법은 다중 에이전트 시스템(c-MADRL)의 에이전트 서로 간의 의존성을 활용하며, 특히 RNN(Recurrent Neural Network)을 사용하여 과거 정보를 기억하고 결합하여 결정합니다. 또한, 보상을 조작하여 백도어 에이전트가 팀 전체에 악영향을 미치도록 유도합니다.

- **Performance Highlights**: 두 가지 일반적인 c-MADRL 알고리즘인 VDN과 QMIX에서 실험을 진행한 결과, 91.6%의 높은 공격 성공률과 3.7%의 낮은 정상 성능 변동률을 유지함을 확인하였습니다.



### DSBench: How Far Are Data Science Agents to Becoming Data Science Experts? (https://arxiv.org/abs/2409.07703)
- **What's New**: 본 논문에서는 데이터 과학 에이전트를 평가하기 위한 포괄적인 벤치마크인 DSBench를 소개합니다. 이 벤치마크는 실제 데이터 과학 과제를 반영하여 466개의 데이터 분석 작업과 74개의 데이터 모델링 작업을 포함하고 있습니다.

- **Technical Details**: DSBench는 Eloquence와 Kaggle 대회에서 제공된 다양한 데이터 분석 및 모델링 작업을 포함합니다. 이 벤치마크는 긴 컨텍스트, 멀티모달 태스크 배경, 대용량 데이터 파일 및 멀티 테이블 구조에 대한 추론, 그리고 엔드투엔드 데이터 모델링 작업을 수행하는 현실적인 설정을 제공합니다.

- **Performance Highlights**: 최신의 LLM 및 LVLM 모델을 평가한 결과, 가장 잘 수행된 에이전트가 데이터 분석 작업의 34.12%만 해결하는 등 대부분의 작업에서 어려움을 겪고 있음을 발견했습니다. 이는 데이터 과학 에이전트의 발전이 필요하다는 것을 강조합니다.



### Passed the Turing Test: Living in Turing Futures (https://arxiv.org/abs/2409.07656)
Comments:
          Author's version. Forthcoming in Intelligent Computing, a Science Partner Journal published in affiliation with Zhejiang Lab (this https URL). First submitted 19 Feb 2024. Revised 16 Jul 2024. Accepted 15 Aug 2024

- **What's New**: 이번 논문에서는 Turing 테스트(Turing Test)의 역사와 의미, 그리고 최근의 pretrained 모델과 transformer 기반의 AI가 이 테스트를 통과할 수 있는 가능성에 대해 논의하고 있습니다.

- **Technical Details**: 논문은 Alan Turing이 1950년에 제안한 Turing 테스트와 그것의 기초가 된 '모방 게임(imitation game)'에 대해 설명합니다. Turing은 기계가 인간처럼 행동할 수 있는지를 판단하기 위해 비전문가를 활용하여 기계의 대화 능력을 평가했습니다. 이는 초기 인공지능(AI) 연구에 큰 영향을 미쳤고, 기계 인텔리전스에 대한 기준을 세웠습니다.

- **Performance Highlights**: 논문에서는 발전된 'child machines'가 사회와 자연에 미치는 영향력에 대해 언급하며, 현재 machine learning이 이루고 있는 성과가 Turing이 꿈꾸던 미래와 어떻게 연결될 수 있는지를 제시합니다. 이를 통해 AI가 실제로 Turing 테스트를 통과할 능력을 보유하고 있음을 설명하고 있습니다.



### Can We Count on LLMs? The Fixed-Effect Fallacy and Claims of GPT-4 Capabilities (https://arxiv.org/abs/2409.07638)
- **What's New**: 본 논문은 LLM(대형 언어 모델)의 성능 평가에 중점을 두며, GPT-4의 여러 결정적 작업에서의 성과 측정을 제시합니다. 각 작업은 기본 계산을 포함하며, 입력 파라미터는 잘 정의된 대규모 집합에서 추출됩니다.

- **Technical Details**: 연구팀은 500회 이상의 실험을 통해 통계적으로 유의미한 차이를 감지할 수 있도록 여러 조건을 검토했습니다. 기본적인 계산 작업에서 숫자 세기, 곱셈 등의 작업을 수행하며, 입력과 쿼리 문구의 다양한 조합을 통해 성능 변화를 측정했습니다.

- **Performance Highlights**: GPT-4는 여전히 상당한 성능을 보이지 않으며, 사소한 쿼리 문구 수정이나 입력 파라미터의 변화가 성과에 큰 영향을 미친다는 사실을 발견했습니다. 이를 통해 LLM의 능력을 정량화하는 노력에 경고를 주며, 실험적 관찰이 일반적인 주장으로 확대될 수 있는 위험성을 강조합니다.



### Understanding Foundation Models: Are We Back in 1924? (https://arxiv.org/abs/2409.07618)
Comments:
          7 pages, 4 Figures, to appear in Proceedings of the 2nd International Conference on Foundation and Large Language Models (FLLM2024) 26-29 November, 2024, Dubai, UAE

- **What's New**: Foundation Models(FMs)의 빠른 발전과 그들이 인공지능(AI) 및 추론에 미치는 영향을 다룬 연구 논문입니다. 모델 크기의 증가가 아닌 혁신적인 훈련 기술로 인한 추론 능력의 발전을 강조하고 있습니다.

- **Technical Details**: Foundation Models는 대규모 AI 모델로, 방대한 비주석 데이터셋에서 훈련되어 통계적 표현을 생성합니다. 이들은 Transformer Architecture를 사용하여 학습되고, 텍스트 데이터의 경우 의미론적 관계를 포착하는 벡터로 표현됩니다. 최근 모델들은 더욱 효과적인 reasoning을 보여주고 있으며, 이는 주로 novel training techniques 덕분입니다.

- **Performance Highlights**: Mistral-7B와 같은 최신 모델들은 크기 대비 뛰어난 성능을 발휘하며, 그룹 쿼리 주의(Grouped-query attention) 및 슬라이딩 윈도우 주의(Sliding window attention) 기법을 통해 효율성을 높이고 있습니다. GPT-4는 이러한 모델들 중 최초로 뛰어난 추론 능력을 입증하였습니다.



### A Novel Mathematical Framework for Objective Evaluation of Ideas using a Conversational AI (CAI) System (https://arxiv.org/abs/2409.07578)
Comments:
          20 pages, 12 figures, 5 tables

- **What's New**: 이 연구는 대화형 인공지능 시스템이 생성한 아이디어를 객관적으로 평가하기 위한 포괄적인 수학적 프레임워크를 도입합니다. 이 방법은 초보 디자이너들이 유망한 아이디어를 선택하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 연구에서는 아이디어를 고차원 벡터로 변환하고, UMAP, DBSCAN, PCA와 같은 도구를 사용하여 아이디어 간의 다양성을 정량적으로 측정합니다. 이를 통해 아이디어의 신뢰성과 객관적인 선택 방식을 제공합니다.

- **Performance Highlights**: 본 연구는 CAI 시스템과 인간이 생성한 아이디어를 자동으로 분석하여 효율적인 아이디어 생성 단계 향상에 기여할 수 있음을 구성적으로 검증합니다.



### Machine Learning and Constraint Programming for Efficient Healthcare Scheduling (https://arxiv.org/abs/2409.07547)
- **What's New**: 이번 논문은 간호사 스케줄링 문제(Nurse Scheduling Problem, NSP)를 해결하기 위해 새로운 암시적(implicit) 및 명시적(explicit) 접근 방식을 소개합니다. 추가적으로, 머신 러닝(Machine Learning) 기법을 활용하여 NSP의 제약 조건과 목표를 학습하는 방법을 제안합니다.

- **Technical Details**: ARP(Active Resource Planning) 문제는 일정 기간 내에 간호사들을 적절한 교대 근무에 할당하기 위한 문제로, 제약 조건과 목표를 충족해야 합니다. 이를 해결하기 위해 Constraint Satisfaction Problem (CSP) 프레임워크를 사용하고, Stochastic Local Search(SLS) 방법론 및 새로운 Branch and Bound 알고리즘을 개발하여 향상된 제약 전파 기법과 변수/값 정렬 휴리스틱을 적용합니다. 또한, Frobenius Norm을 이용하여 생성된 솔루션과 역사적 데이터 간의 평균 오류를 측정하여 암시적 접근법의 품질을 정량화합니다.

- **Performance Highlights**: 결과적으로, 제안된 암시적 및 명시적 접근법은 실질적으로 NSP를 해결하는 데 있어 높은 효율성을 보였으며, 제약 조건에 대한 불확실성을 대응하는 데 효과적이라는 것이 입증되었습니다. 여러 연구와 이전 시도들과의 비교를 통해, 제안된 접근 방식은 최적화된 솔루션을 생성하고 근무 패턴을 개선하는 성과를 보여주었습니다.



### Still More Shades of Null: A Benchmark for Responsible Missing Value Imputation (https://arxiv.org/abs/2409.07510)
- **What's New**: 본 논문은 Shades-of-NULL이라는 새로운 벤치마크를 제안하여, 다양한 결측값 대체 기법들을 포함하고 있으며, 이를 머신 러닝 개발 라이프사이클에 통합합니다. 또한, 다양한 결측 발생 시나리오를 모델링하고, 공정성과 안정성을 포함한 종합적인 평가 방안을 제공합니다.

- **Technical Details**: Shades-of-NULL은 MCAR, MAR, MNAR를 초월하는 다중 메커니즘 결측 (multi-mechanism missingness)과 결측 변화 (missingness shift) 등의 현실적인 결측 시나리오를 모델링합니다. 정확한 기법 평가를 위해 20,952개의 실험 파이프라인을 활용하였습니다.

- **Performance Highlights**: 다양한 결측 상황에 따라 특정 기법이 우수한 결과를 보이는 경향이 있으며, 예측 성능, 공정성 및 안정성 간의 트레이드오프를 식별하였습니다. 단일 우수한 대체 기법은 없지만, 간단한와 복잡한 결측 상황에서의 성능 패턴이 흥미롭게 나타났습니다.



### Traceable LLM-based validation of statements in knowledge graphs (https://arxiv.org/abs/2409.07507)
- **What's New**: 이 논문은 LLMs(대규모 언어 모델)를 사용하여 RDF triples를 검증하는 새로운 방법을 제안합니다. 그동안 LLM이 사용자의 질문에 대한 응답을 구성하는 데 사용된 정보의 출처를 신뢰할 수 없기 때문에, 이 방법은 LLM 내부의 사실적 지식을 전혀 사용하지 않는 방식을 채택하였습니다.

- **Technical Details**: 제안된 방법은 검증된 RDF 문장을 웹 검색이나 위키피디아를 통해 검색한 외부 문서의 청크(chunk)와 비교하는 방식입니다. 이 작업의 적용 가능성을 평가하기 위해 BioRED 데이터셋의 1,719개의 긍정적 진술과 같은 수의 새롭게 생성된 부정적 진술을 평가하였습니다. 이를 통해 얻어진 정밀도(precision)는 88%이며, 재현율(recall)은 44%로 나타났습니다. 이는 인간의 감독이 필요함을 시사합니다.

- **Performance Highlights**: Wikidata에서 SPARQL 쿼리를 사용하여 검증이 필요한 진술을 자동으로 검색하는 방법을 시연하였습니다. 전반적으로 이 결과는 LLM이 KGs(지식 그래프) 내의 진술을 대규모로 검증하는 데 사용될 수 있음을 보여주며, 이는 이전에는 인간 주석 비용 때문에 불가능한 작업이었습니다.



### OneEdit: A Neural-Symbolic Collaboratively Knowledge Editing System (https://arxiv.org/abs/2409.07497)
Comments:
          LLM+KG@VLDB2024, code is available at this https URL

- **What's New**: OneEdit는 사용자가 자연어를 통해 지식을 쉽게 관리할 수 있도록 돕는 신경-상징(Neural-Symbolic) 프로토타입 시스템으로, 지식 그래프(KG)와 대형 언어 모델(LLM)을 통합했습니다.

- **Technical Details**: OneEdit는 세 가지 주요 모듈로 구성되어 있습니다: 1) Interpreter는 자연어를 통한 사용자 상호작용을 담당합니다; 2) Controller는 다양한 사용자로부터 편집 요청을 관리하며 갈등 해결과 독성 지식 공격 방지를 위한 롤백(rollback) 기능을 사용합니다; 3) Editor는 Controller의 지식을 활용하여 KG와 LLM을 편집합니다.

- **Performance Highlights**: OneEdit는 두 개의 새로운 데이터셋을 활용하여 실험을 실시했으며, KG를 사용한 지식 편집에서 기존 모델을 초월하는 성능을 보여주었습니다.



### AnySkin: Plug-and-play Skin Sensing for Robotic Touch (https://arxiv.org/abs/2409.08276)
- **What's New**: 이번 논문에서는 AnySkin이라는 새로운 촉각 센서를 소개합니다. AnySkin은 저렴하며 사용이 간편하고, 다양한 센서 인스턴스 간에 일관된 응답을 보장합니다.

- **Technical Details**: AnySkin은 기존 ReSkin의 설계를 기반으로 하여, 감지 메커니즘과 상호작용 표면을 분리하여 제작 간소성을 높였습니다. 이 센서는 자석이 부착된 형태로, 여러 형상에 쉽게 적용할 수 있고, 손쉽게 교체할 수 있는 디자인을 갖추고 있습니다. 또한, AnySkin은 기존의 촉각 솔루션들보다 빠르고 효율적인 재사용이 가능합니다.

- **Performance Highlights**: AnySkin은 xArm, Franka 등 다양한 로봇에 적용될 수 있으며, USB 삽입과 같은 정밀 작업을 위한 slip detection 및 policy learning에 호환됩니다. AnySkin은 센서를 교체하는 데 평균 12초가 소요되며, 이전 센서에서 훈련된 모델이 다른 AnySkin으로 제로샷(Zero-shot) 일반화에 성공합니다. 이는 성능 저하가 13%에 불과하여, ReSkin의 43% 성능 저하와 비교됩니다.



### Hand-Object Interaction Pretraining from Videos (https://arxiv.org/abs/2409.08273)
- **What's New**: 본 논문은 3D 손-물체 상호작용 궤적을 통해 일반 로봇 조작 우선 순위를 학습하는 접근 방식을 제시합니다. 야생 비디오를 활용하여 센서모터 로봇 궤적을 생성하는 프레임워크를 구축하였습니다.

- **Technical Details**: 이 방법에서는 사람의 손과 조작된 물체를 공유 3D 공간에서 각각의 궤적을 맵핑하여 로봇의 동작으로 리타겟팅합니다. 사전 훈련된 정책은 센서모터 로봇 궤적에 대한 조건부 분포 일치를 목표로 하는 인과 변환기(causal transformer)의 가중치에 내재되어 있습니다.

- **Performance Highlights**: 사례 연구를 통해, 이 조작 우선 순위가 기존 방법보다 기술 습득을 상당히 가속화하며, 훈련 비디오에 없는 기술들도 잘 발휘되고 최종 정책의 일반화 및 견고성을 향상시킨다는 결과를 보여주었습니다.



### FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally (https://arxiv.org/abs/2409.08270)
Comments:
          ECCV'2024

- **What's New**: 본 연구는 2D 마스크에서 3D Gaussian Splatting(3D-GS) 세분화(세그멘테이션)를 정확하게 수행하는 도전에 집중하고 있습니다. 기존의 방법은 반복적인 gradient descent에 의존했지만, 본 논문에서는 간단하면서도 전역적으로 최적화된 솔루션을 제안합니다.

- **Technical Details**: 제안된 접근법은 재구성된 3D-GS 장면을 활용하여 2D 마스크 렌더링이 각 Gaussian의 레이블에 대한 선형 함수로 보일 수 있음을 인식합니다. 여기서 최적의 레이블 할당은 linear programming(선형 프로그래밍)을 통해 닫힌 형태로 해결할 수 있습니다. 또한, 최적화의 목표 함수에 배경 편향을 통합하여, 3D 세분화에서 노이즈에 대한 강력함을 보여줍니다. 최적화는 약 30초 내에 완료되며, 기존 방법보다 50배 빠릅니다.

- **Performance Highlights**: 광범위한 실험을 통해 다양한 장면을 세분화하는 데 있어 제안된 방법의 우수성을 입증하였으며, 객체 제거 및 인페인팅과 같은 다운스트림(downstream) 작업에서도 개선된 성능을 보여주었습니다.



### LoRID: Low-Rank Iterative Diffusion for Adversarial Purification (https://arxiv.org/abs/2409.08255)
Comments:
          LA-UR-24-28834

- **What's New**: 이번 연구에서는 확산 기반의 정화 방법에 대한 정보 이론적 분석을 제시하며, LoRID라는 새로운 Low-Rank Iterative Diffusion 정화 방법을 도입합니다. 해당 방법은 적대적 교란을 제거하는 데 초점을 맞추고 있습니다.

- **Technical Details**: LoRID는 다단계 정화 프로세스를 중심으로 하며, 확산 모델의 초기 단계에서 여러 번의 확산-노이즈 제거 루프를 활용합니다. 또한, 매트릭스 분해의 확장인 Tucker 분해를 통합하여 높은 노이즈 환경에서도 적대적 노이즈를 제거합니다. LoRID의 이론적 기초는 Markov 기반 정화의 내재적 오류를 다룹니다. 특히, DDPM(노이즈 제거 확산 확률 모델)을 활용하여 정화의 효과성을 높였습니다.

- **Performance Highlights**: LoRID는 CIFAR-10/100, CelebA-HQ, ImageNet 데이터셋에서 흰 상자 및 검은 상자 조건 모두에서 뛰어난 강건성 성능을 달성하였습니다. LoRID는 SOTA(SOta) 성능의 벤치마크를 초과하는 성능을 보입니다.



### OmniQuery: Contextually Augmenting Captured Multimodal Memory to Enable Personal Question Answering (https://arxiv.org/abs/2409.08250)
- **What's New**: OmniQuery는 개인의 메모리에 대한 복잡한 질문을 처리하는 새로운 시스템으로, 여러 개의 상호 연결된 메모리에서 흩어져 있는 맥락 정보를 통합하여 작동합니다.

- **Technical Details**: OmniQuery는 두 가지 주요 구성 요소로 이루어져 있습니다: (i) 상이한 메모리를 기반으로 하는 맥락 정보를 통해 캡처된 메모리를 보강하는 파이프라인과 (ii) 이러한 처리된 메모리를 검색하여 포괄적인 답변을 생성하는 자연어 질문 응답 시스템입니다. 설계된 맥락 정보 분류법은 29명의 참가자와의 일기 연구를 통해 수집된 299개의 사용자 쿼리에 기초하였습니다.

- **Performance Highlights**: OmniQuery는 71.5%의 정확도로 일반 RAG 기반 시스템을 능가하며, 비교 평가에서 74.5%의 경우에 승리하거나 동점을 기록하여 사용자 지각 정확성과 완전성에서 우수한 성과를 보였습니다.



### IFAdapter: Instance Feature Control for Grounded Text-to-Image Generation (https://arxiv.org/abs/2409.08240)
- **What's New**: 이번 연구는 Instance Feature Generation (IFG) 작업을 처음으로 도입하여 텍스트에서 이미지로의 생성 모델의 포지셔닝과 기능 생성을 동시에 개선하고자 합니다. IFG 작업에 대한 해결책으로 Instance Feature Adapter (IFAdapter)를 제안하여 이미지 생성의 세밀함을 향상시킵니다.

- **Technical Details**: IFAdapter는 새로운 appearance tokens와 Instance Semantic Map을 도입하여 인스턴스 기능의 정확성을 보장합니다. Appearance tokens는 인스턴스별 특성 정보를 추출하고, 2D semantic map은 인스턴스 기능과 공간적 위치를 정렬하는 데 사용됩니다. 이 장치는 기존의 확산 모델에 plug-and-play 방식으로 통합될 수 있습니다. 실험에서는 COCO-IFG 벤치마크와 검증 파이프라인을 통해 객관적인 비교를 수행하였습니다.

- **Performance Highlights**: 실험 결과 IFAdapter는 기존 모델에 비해 수치적 및 질적 평가 모두에서 우수한 성능을 보여줍니다. 특히 IFAdapter는 정확한 포지셔닝과 기능 생성을 동시에 달성하여 다양한 생성 모델에 적용 가능성이 높습니다.



### Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources (https://arxiv.org/abs/2409.08239)
- **What's New**: 이 논문에서는 Source2Synth라는 새로운 방법을 제안합니다. 이 방법은 비싼 인간 주석에 의존하지 않고 대규모 언어 모델(LLMs)에게 새로운 기술을 가르치기 위한 것입니다.

- **Technical Details**: Source2Synth는 세 가지 단계로 구성됩니다: Dataset Generation, Dataset Curation, 그리고 Model Fine-tuning입니다. 입력으로 사용자 정의 데이터 소스를 받아들여, 실제 세계의 출처에 기반한 중간 추론 단계를 포함한 합성 데이터 포인트를 생성합니다. 이 방법은 데이터 품질을 개선하고 불량 생성물을 제거합니다.

- **Performance Highlights**: Source2Synth는 WikiSQL에서 TQA(표 기반 질문 응답) 성능을 25.51% 향상시키고, HotPotQA에서 MHQA(다단계 질문 응답) 성능을 22.57% 향상시켰습니다.



### LLM Honeypot: Leveraging Large Language Models as Advanced Interactive Honeypot Systems (https://arxiv.org/abs/2409.08234)
Comments:
          7 pages, 5 figures

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 활용하여 보다 현실적이고 인터랙티브한 honeypot 시스템을 개발하는 새로운 접근 방식을 제안합니다. 기존의 honeypot 기술의 한계를 극복하고, 공격자와의 상호작용을 보다 정교하게 실행할 수 있는 가능성을 탐구합니다.

- **Technical Details**: 이 연구는 공격자 생성 명령어와 응답 데이터셋을 기반으로 사전 훈련된 오픈 소스 언어 모델을 최적화하는 Supervised Fine-Tuning(SFT) 과정을 통해 LLM honeypot 시스템을 개발했습니다. 데이터 수집, 프롬프트 엔지니어링, 모델 선택 및 성능 최적화 등의 주요 단계를 포함하며, 최종적으로 모델은 공용 IP 주소에서 실제 공격자와 실시간으로 상호작용할 수 있도록 배포되었습니다.

- **Performance Highlights**: 이론적 검증 및 실제 배포를 통해 평가한 결과, 제안된 LLM honeypot는 정확하고 유용한 응답을 생성할 수 있는 효과적인 솔루션임을 입증했습니다. 이러한 기술은 공격자의 행동을 분석하고 비즈니스 보안 인프라를 강화하는 데 중요한 도구가 될 것으로 기대됩니다.



### Design Optimization of Nuclear Fusion Reactor through Deep Reinforcement Learning (https://arxiv.org/abs/2409.08231)
Comments:
          16 pages

- **What's New**: 이번 연구는 Deep Reinforcement Learning (DRL)을 활용하여 원자력 융합로 디자인 최적화에 대한 새로운 접근 방식을 제안합니다. DRL을 통해 여러 물리적 및 공학적 제약을 보다 효율적으로 해결하고, 최적의 원자로 설계를 찾을 수 있습니다.

- **Technical Details**: 이 연구에서는 원자력 융합로의 설계를 최적화하기 위해 DRL 프레임워크를 적용하였으며, tokamak 설계 시 고려해야 할 다양한 운영 한계들(예: density limit, beta limit 등)을 정리했습니다. 연구에 사용된 계산 코드는 플라즈마 압력, 밀도, 비용 등의 매개변수를 포함하여 설계 파라미터를 계산하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 제안한 방법은 기존의 grid search algorithm에 비해 효율적으로 운영 제약을 충족하면서 비용을 절감하는 원자로 설계를 찾는 데 성공하였습니다. DRL을 통해 미래의 융합로 설계에 실질적인 통찰력을 제공하였습니다.



### Photonic Quantum Computers (https://arxiv.org/abs/2409.08229)
Comments:
          47 pages, 16 figures

- **What's New**: 이 논문은 포토닉(photonic) 기반 양자 컴퓨팅의 최근 발전을 종합적으로 조망하며, 산업의 주요 참여자들이 개발한 성능, 아키텍처 디자인과 대규모 내결함성(fault-tolerance) 포토닉 양자 컴퓨터 개발 전략을 탐구합니다.

- **Technical Details**: 포토닉 양자 컴퓨터는 포톤을 큐비트(qubit) 인코딩 및 조작의 매개체로 사용하며, 이는 고유한 노이즈 저항성을 제공합니다. 또한 이들은 모듈식이어서 쉽게 네트워크화 할 수 있는 장점을 지니고 있습니다. QIP(Quantum Information Processing)에서 요구되는 네 가지 조건은 대규모성, 보편성, 낮은 손실, 양자 간섭 유지입니다.

- **Performance Highlights**: 포토닉 양자 컴퓨터는 암호화, 양자 화학 및 재료 과학 분야에서 고전 시스템을 초월하는 계산 효율성을 향상시킬 잠재력을 지니고 있습니다. 주요 기업들인 iPronics, ORCA Computing, PsiQuantum 등은 각기 다른 성능의 포토닉 양자 프로세서를 개발하여 이 분야의 발전에 기여하고 있으며, 최근의 실험은 포토닉 기반 양자 컴퓨터의 고유한 계산적 이점을 활용합니다.



### CliquePH: Higher-Order Information for Graph Neural Networks through Persistent Homology on Clique Graphs (https://arxiv.org/abs/2409.08217)
- **What's New**: 이 논문에서는 CliquePH라는 새로운 토폴로지(layer)를 도입하여 그래프 신경망(GNN)에 추가할 수 있으며, 이는 더 높은 차원의 그래프 구조에 대한 정보를 제공합니다.

- **Technical Details**: CliquePH는 낮은 차원의 persistent homology 알고리즘을 여전히 사용하면서 그래프의 더 높은 차원 구조에 대한 정보를 추출하는 방법을 제안합니다. 이 기법은 그래프를 여러 연결 클리크 그래프로 '리프트(lift)'한 후, 각 '고차' 그래프에 대해 1차원 persistent homology를 적용합니다. 이 과정에서 단순히 노드의 수에 비례하는 선형 복잡도를 유지하며 효율적으로 더 높은 차원 정보를 캡처합니다.

- **Performance Highlights**: 표준 벤치마크 데이터셋에서 이 방법을 적용한 결과, 최대 31%의 테스트 정확도 향상을 달성하였습니다.



### LT3SD: Latent Trees for 3D Scene Diffusion (https://arxiv.org/abs/2409.08215)
Comments:
          Project page: this https URL Video: this https URL

- **What's New**: LT3SD는 고품질의 대규모 3D 장면 생성을 위한 새로운 latent diffusion 모델입니다. 이 모델은 복잡하고 다양한 구조의 3D 장면 생성을 위해 새로운 latent tree representation을 도입하여 lower-frequency geometry와 higher-frequency detail을 효과적으로 인코딩합니다.

- **Technical Details**: LT3SD는 coarse-to-fine hierarchy를 사용하여 3D 장면을 트리 구조로 분해하며, 각 해상도에서 장면의 latent 구성 요소를 모델링합니다. 이를 통해, scene patches를 기반으로 diffusion 모델을 훈련시켜 임의 크기의 3D 장면을 생성합니다. 이 접근법은 복잡한 비정렬 3D 장면 대신 높은 유사성을 가진 로컬 구조에 학습 초점을 이동시킵니다.

- **Performance Highlights**: 실험 결과, LT3SD는 기존 모델보다 70% 개선된 FID 점수를 기록하며, 고품질 무조건적 3D 장면 생성을 위한 효율성과 장점을 입증했습니다.



### What Makes a Maze Look Like a Maze? (https://arxiv.org/abs/2409.08202)
- **What's New**: 본 논문에서는 Deep Schema Grounding (DSG)이라는 새로운 프레임워크를 도입하여 시각적 추상 개념을 이해하는 데 도움을 주는 방법을 제안합니다. DSG는 시각적 추상화를 위한 명시적 구조화된 표현을 활용하여 추론합니다.

- **Technical Details**: DSG의 핵심 요소는 schema로, 이는 추상 개념을 보다 원시적인 기호로 분해하는 의존성 그래프(Dependency Graph) 설명입니다. DSG는 대형 언어 모델(LLM)을 사용하여 schema를 추출하고, 이를 비전-언어 모델(Vision-Language Model, VLM)과 계층적으로 결합하여 이미지에 대해 추상 개념의 구체적인 구성 요소를 접지합니다.

- **Performance Highlights**: DSG는 비전-언어 모델의 추상 시각 추론 성능을 유의미하게 향상시켰습니다. 특히, GPT-4V의 경우 8.3%의 상대적 개선을 보였고, Counting 관련 질문에서는 11.0%의 상대적 개선을 달성했습니다.



### AudioBERT: Audio Knowledge Augmented Language Mod (https://arxiv.org/abs/2409.08199)
Comments:
          Preprint

- **What's New**: 최근 연구에서 텍스트 전용 데이터셋에 사전 훈련된 언어 모델이 기본적인 시각적 지식, 예를 들어 일상 객체의 색상이 부족하다는 것을 확인했습니다. 이를 통해 우리는 언어 모델이 청각 지식에 대해서도 동일한 결점이 있는지를 알아보았고, 이에 대한 평가를 위해 새로운 데이터셋 AuditoryBench를 구축했습니다.

- **Technical Details**: AuditoryBench는 두 가지 새로운 과제를 포함하는 데이터셋으로, 언어 모델의 청각 지식을 평가합니다. 첫 번째 과제는 동물 소리 인식이며, 두 번째 과제는 소리의 피치 비교입니다. 또한, 우리는 AudioBERT라는 새로운 방법을 제안하여 BERT의 청각 지식을 증강합니다. 이는 텍스트에서 청각 지식이 필요한 부분을 탐지하고, 관련 오디오를 효율적으로 검색하는 절차로 진행됩니다.

- **Performance Highlights**: AudioBERT는 AuditoryBench에서 40% 이상의 성능 향상을 달성하며, 언어 모델에 청각 상식 지식을 주입하는 첫 번째 알고리즘이라는 점에서 의미가 있습니다.



### Fine-tuning Large Language Models for Entity Matching (https://arxiv.org/abs/2409.08185)
Comments:
          8 pages, 4 figures. For related code and data, see this this https URL

- **What's New**: 이 연구는 엔티티 매칭(entity matching)을 위한 대규모 생성 언어 모델(LLM) 미세 조정(fine-tuning)의 가능성을 탐구합니다. 이를 통해 기존의 프리트레인(pre-trained) 언어 모델보다 높은 제로샷(zero-shot) 성능을 발휘하는 LLM의 잠재력을 분석합니다.

- **Technical Details**: 미세 조정은 두 가지 방향으로 진행됩니다: 1) 훈련 샘플의 표현에서 LLM이 생성한 다양한 설명을 추가하여 훈련 세트를 실험합니다. 2) LLM을 사용하여 훈련 샘플을 선택 및 생성합니다. 또한, 작은 모델에서 미세 조정이 성능을 크게 향상시키지만, 큰 모델에서는 결과가 혼합되는 경향이 관찰되었습니다.

- **Performance Highlights**: 각종 실험을 통해 미세 조정이 세 가지 모델에서 긍정적인 영향을 미치며, 구조화된 설명을 추가하면 성능이 향상되는 것으로 나타났습니다. 반면, GPT-4o Mini 모델의 경우 예시 선택 및 생성 방법이 성능을 낮추는 결과를 보였습니다.



### Towards a graph-based foundation model for network traffic analysis (https://arxiv.org/abs/2409.08111)
Comments:
          Pre-print of Accepted Workshop paper to 3rd GNNet, co-located with CoNEXT'24

- **What's New**: 이 논문은 대규모 사전 학습(large-scale pretraining)을 이용한 네트워크 트래픽 분석을 위한 새로운 그래프 기반 접근 방식을 제안합니다. 이전 연구들은 토큰화된 패킷 데이터를 사용했지만, 이 연구는 흐름 수준에서 네트워크 트래픽을 동적인 시공간 그래프로 표현합니다.

- **Technical Details**: 제안된 접근 방식은 자기 지도 링크 예측(self-supervised link prediction) 사전 훈련(Task)를 사용하여 네트워크 그래프의 공간적 및 시간적 역학을 캡처합니다. 이 방법은 수많은 신경망 작업에 대해 효율적이며, GNN(그래프 신경망) 모델을 기반으로 합니다.

- **Performance Highlights**: 모델은 침입 탐지(intrusion detection), 트래픽 분류(traffic classification), 봇넷 분류(botnet classification)라는 세 가지 다운스트림 작업에서 몇 번의 학습(few-shot learning) 실험을 수행하였으며, 사전 훈련된 모델에서 미세 조정된 모델이 대조군에 비해 평균 6.87%의 성능 향상을 보였습니다.



### The CLC-UKET Dataset: Benchmarking Case Outcome Prediction for the UK Employment Tribuna (https://arxiv.org/abs/2409.08098)
- **What's New**: 이번 논문은 영국 고용 재판소(UKET)의 사건 결과 예측을 위한 벤치마크를 개발하여 기술 혁신과 접근성 문제를 탐구합니다. 대량의 수동 주석 작업을 해결하기 위해, 대형 언어 모델(LLM)을 사용하여 자동 주석을 생성한 CLC-UKET 데이터셋을 만들었습니다.

- **Technical Details**: CLC-UKET 데이터셋은 약 19,000개의 UKET 사건 및 메타데이터로 구성되며, 사실, 청구, 판례 참조, 법령 참조, 사건 결과 등 포괄적인 법적 주석이 포함되어 있습니다. 향후 UKET 사건의 다양한 결과를 예측하기 위한 멀티 클래스 논문 작업을 수행하였습니다. 피험자의 예측 값과 모델 성능을 비교하기 위한 기준을 설정하기 위한 인간 성과도 수집되었습니다.

- **Performance Highlights**: 파인튜닝된 변환 모델이 제로샷(Zero-Shot) 및 몇 샷(Few-Shot) LLMs보다 성능이 우수하다는 결과가 도출되었습니다. 제로샷 LLM의 성능은 작업 관련 정보를 몇 샷 예제에 통합함으로써 향상될 수 있음을 보여주었습니다. CLC-UKET 데이터셋은 UK 고용 관련 분쟁 해결을 위한 중요한 벤치마크로 작용할 전망입니다.



### AI-accelerated discovery of high critical temperature superconductors (https://arxiv.org/abs/2409.08065)
Comments:
          11 pages, 7 figures, 4 tables

- **What's New**: 이 논문에서는 AI 검색 엔진을 개발하여, 기존의 데이터베이스에서 높은 임계 온도($T_c$)를 가진 초전도체를 발견하는 새로운 방법론을 제안합니다. 이 방법은 깊은 모델(pre-training)과 세부 조정(fine-tuning) 기술, 확산 모델(diffusion models), 물리 기반 접근 방식(first-principles electronic structure calculation)을 통합합니다.

- **Technical Details**: AI 검색 엔진을 통해 총 74개의 동적 안정성(dynamic stability)을 가진 새로운 물질이 발견되었습니다. 이 물질들은 $T_c\geq 15 K$로 예측되었으며, 기존의 데이터셋에는 포함되지 않은 새로운 후보 초전도체입니다. B$_4$CN$_3$와 B$_5$CN$_2$의 임계 온도는 각각 24.08 K와 15.93 K입니다. 이렇게 발견된 새로운 고온 초전도체들에 대한 전반적인 경향 분석도 포함되어 있습니다.

- **Performance Highlights**: AI 기술을 활용해 고온 초전도체 발굴의 가능성을 보여주며, 특정 속성을 가진 물질의 발견을 가속화할 수 있는 잠재력을 강조합니다.



### Unleashing Worms and Extracting Data: Escalating the Outcome of Attacks against RAG-based Inference in Scale and Severity Using Jailbreaking (https://arxiv.org/abs/2409.08045)
Comments:
          for Github, see this https URL

- **What's New**: 본 논문에서는 GenAI 모델의 탈옥(jailbreak) 능력을 통해 공격자들이 RAG 기반 GenAI 응용 프로그램에 대한 공격의 심각성과 규모를 확대할 수 있음을 보여줍니다.

- **Technical Details**: 논문의 첫 번째 부분에서는 RAG membership inference 공격 및 RAG entity extraction 공격이 RAG documents extraction 공격으로 확대될 수 있음을 보여주며, 세 가지 추출 방법에 대한 결과를 평가합니다. 두 번째 부분에서는 RAG 데이터 중독 공격이 단일 GenAI 기반 응용 프로그램에서 전체 GenAI 생태계로 확대될 수 있는 방법을 논의합니다. 이는 컴퓨터 웜을 활용하여 체인 반응을 유도하는 적대적 자기 복제 프롬프트를 제작함으로써 이루어집니다.

- **Performance Highlights**: 연구 결과, 공격자는 Q&A 챗봇의 데이터베이스에 저장된 데이터의 80%-99.8%를 추출할 수 있으며, RAG 기반 이메일 도우미에서 기밀 데이터 추출 체인을 생성하는 성능을 평가하였습니다. 또한, 다양한 방어 수단(guardrails)의 효과도 검토하고 논의합니다.



### Edge-Wise Graph-Instructed Neural Networks (https://arxiv.org/abs/2409.08023)
- **What's New**: 본 연구에서는 그래프 기반 다중 과제 회귀(Multi-task Regression on Graph Nodes, RoGN) 문제를 다루며, 기존의 Graph-Instructed Neural Network (GINN) 아키텍처의 한계를 논의합니다. 이를 통해 새로운 Edge-Wise GI (EWGI) 레이어를 형식화하고, 이 레이어가 GINN보다 좋은 성과를 보임을 보여줍니다.

- **Technical Details**: EWGI 레이어는 그래프 노드에 연결된 추가 가중치를 사용하는 새로운 그래프 지시(GI) 레이어로, 정보의 전송 과정을 엣지 단위로 맞춤화할 수 있습니다. 이를 통해 기존 GI 레이어의 일반화 능력을 향상시키고, GINN보다 성능을 개선하였습니다.

- **Performance Highlights**: EWGINN은 Erdos-Rényi 그래프 구조에서 GINN보다 더 좋은 성능을 보여주었으며 학습 가중치의 수는 소폭 증가하였습니다. 실험은 Barabási-Albert 및 Erdos-Rényi 그래프 기반의 두 가지 라벨링 네트워크에서 수행되었습니다.



### Learning Causally Invariant Reward Functions from Diverse Demonstrations (https://arxiv.org/abs/2409.08012)
- **What's New**: 본 논문에서는 인버스 강화 학습(Inverse Reinforcement Learning, IRL) 방법론의 새로운 정규화 접근법을 제안합니다. 이 접근법은 인과 불변성 원칙(causal invariance principle)을 기반으로 하여, 보상 기능의 일반화를 개선하는 것을 목표로 합니다.

- **Technical Details**: 저자들은 IRL의 학습 작업에 이 정규화를 적용하며, 이를 통해 복구된 보상 기능으로 훈련된 정책의 성능이 우수하다는 것을 입증합니다. 보상 학습 문제에서 발생할 수 있는 스푸리어스(spurious) 상관관계를 피하고, 다양한 전문가는 동일한 작업을 수행하는 전문가의 시연을 통해 보상 기능을 학습하는 방법에 대해 설명합니다.

- **Performance Highlights**: 정확한 설정(exact setting) 및 근사 설정(approximate setting) 모두에서 이 접근법의 효율성을 입증하였으며, 특히 동적 환경 변화가 있는 MDP에서 지상 진리 성능이 개선됨을 보여주었습니다.



### Enhancing Few-Shot Image Classification through Learnable Multi-Scale Embedding and Attention Mechanisms (https://arxiv.org/abs/2409.07989)
- **What's New**: 본 논문은 기존의 metric-based 방법들이 지니고 있는 한계를 극복하기 위한 새로운 접근법을 제안합니다. 이 방법은 다양한 feature 공간을 통해 샘플을 매핑하는 multi-output embedding 네트워크를 활용하여, 노드의 서로 다른 단계에서 feature 벡터를 추출하고 이를 통해 전역(global) 및 추상(abstract) feature를 모두 캡처하는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 ResNet18을 feature extractor로 사용하여, 다섯 개의 서로 다른 단계에서 feature map을 추출합니다. 각 단계마다 learnable parameter weights를 도입하여 feature representation 능력을 향상시키고, self-attention 메커니즘을 통해 각 feature map의 정보를 풍부하게 합니다. 이러한 구성 요소들은 모델 성능을 크게 향상시키는데 기여합니다.

- **Performance Highlights**: MiniImageNet 및 FC100 데이터셋에서 5-way 1-shot 및 5-way 5-shot 시나리오에 대해 종합적으로 평가한 결과, 제안된 방법이 최신 기법들에 비해 우수한 성능을 보였으며, MiniImageNet 데이터셋에서 CUB 데이터셋으로의 cross-domain 작업에서도 높은 정확도를 기록했습니다.



### ProbTalk3D: Non-Deterministic Emotion Controllable Speech-Driven 3D Facial Animation Synthesis Using VQ-VAE (https://arxiv.org/abs/2409.07966)
Comments:
          14 pages, 9 figures, 3 tables. Includes code. Accepted at ACM SIGGRAPH MIG 2024

- **What's New**: 이 논문에서는 감정 조절이 가능한 음성 기반 3D 얼굴 애니메이션 합성을 위한 비결정적(Non-Deterministic) 신경망 접근 방식을 제안합니다. 특히, 감정 데이터를 활용하여 다양한 감정 표현을 동시에 합성할 수 있는 모델을 개발했습니다.

- **Technical Details**: ProbTalk3D라는 모델은 두 단계의 VQ-VAE(Vektor Quantized Variational Autoencoder) 구조를 기반으로 하며, 3DMEAD라는 감정적으로 풍부한 얼굴 애니메이션 데이터셋을 사용하여 설계되었습니다. 이 모델은 감정 레이블과 강도를 통한 감정 조절 기능을 제공하는 최초의 비결정적 3D 얼굴 애니메이션 합성 방법입니다.

- **Performance Highlights**: 제안된 모델은 최신 감정 조절 모델들과 비교하여 뛰어난 성능을 보여주며, 특히 주관적인 평가에서는 사용자 연구를 통해 모델의 결과가 우수하다는 것을 입증했습니다. 또한, 모델의 코드는 공개적으로 사용 가능합니다.



### WirelessAgent: Large Language Model Agents for Intelligent Wireless Networks (https://arxiv.org/abs/2409.07964)
- **What's New**: WirelessAgent라는 새로운 접근 방식을 소개합니다. 이 방법은 대형 언어 모델(LLMs)을 활용하여 복잡한 무선 네트워크 작업을 관리할 수 있는 AI 에이전트를 개발합니다. WirelessAgent는 고급 추론, 다중 모달 데이터 처리, 자율적 의사결정을 통해 네트워크 성능을 효과적으로 향상시킬 수 있으며, 네트워크 슬라이싱 관리에 대한 실제 적용 가능성과 혜택을 보여줍니다.

- **Technical Details**: WirelessAgent는 감지(perception), 기억(memory), 계획(planning), 행동(action)이라는 네 가지 핵심 모듈로 구성된 프레임워크입니다. 에이전트는 다중 모달 입력을 해석하고, 복잡한 작업을 자동화하며, 외부 지식 기반과 도구의 도움으로 솔루션을 출력합니다. WirelessAgent는 사용자의 의도를 정확하게 이해하고, 슬라이스 리소스를 효과적으로 할당하며, 지속적으로 최적의 성능을 유지할 수 있는 능력을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, WirelessAgent는 사용자 의도를 정확히 이해하고, 슬라이스 리소스를 효과적으로 할당하며, 네트워크 성능을 지속적으로 최적화하는 능력을 보여주었습니다. 이는 무선 통신의 복잡성을 관리하는 데 있어 AI 에이전트의 잠재력을 입증합니다.



### Rapid Parameter Estimation for Extreme Mass Ratio Inspirals Using Machine Learning (https://arxiv.org/abs/2409.07957)
- **What's New**: 본 연구는 EMRI 신호의 Bayesian posterior estimation을 위한 새로운 머신 러닝 기법인 flow matching 기술을 ODE 신경망을 기반으로 한 응용을 통해 제안합니다. 이는 전통적인 MCMC 방법에 비해 계산 효율성이 여러 배 향상된 것을 보여줍니다.

- **Technical Details**: EMRI(Exteme-Mass-Ratio Inspiral) 신호를 분석하기 위해 머신 러닝 기술을 활용한 Bayesian posterior 추정 방법을 제안하며, 이 방법은 여러 개의 local maxima와 flat regions로 인해 어려워지는 파라미터 식별 문제를 해결합니다. 연구에서는 Continuous Normalizing Flows (CNFs)를 사용하여 복잡한 파라미터 공간 탐색을 가능하게 합니다.

- **Performance Highlights**: 제안된 접근 방법은 MCMC 방법보다 수 배 이상의 계산 효율성을 보이며, 신뢰할 수 있는 파라미터 추정을 유지합니다. 또한 최대 17개의 다양한 파라미터를 효과적으로 처리할 수 있는 가능성을 제시합니다.



### Reinforcement Learning Discovers Efficient Decentralized Graph Path Search Strategies (https://arxiv.org/abs/2409.07932)
- **What's New**: 이 논문은 Reinforcement Learning (RL)을 사용하여 분산된 그래프 경로 탐색 문제를 다룹니다. 연구진은 다중 에이전트 시스템에서 국소적 정보만을 활용하여 경로를 찾는 새로운 접근 방식을 제안하여, 기존의 중앙 집중형 모델이 지닌 한계를 극복하고자 하였습니다.

- **Technical Details**: 저자들은 Decentralized Partially Observable Markov Decision Process (Dec-POMDP) 프레임워크를 기반으로 한 다중 에이전트 RL 모델을 사용하는 방법을 제안합니다. 에이전트들은 각자의 로컬 보기에서 그래프 구조와 이웃 속성을 활용하여 목표 노드에 대한 경로를 찾습니다. 여기서 제안된 방법은 Graph Attention Network (GAT)를 통해 노드 표현을 학습하며, 중앙 집중 훈련과 분산 실행 (Centralized Training and Decentralized Execution, CTDE) 패러다임을 따릅니다.

- **Performance Highlights**: 본 모델은 합성 데이터 및 실제 소셜 네트워크를 포함한 다양한 실험에서 기존의 학습 및 휴리스틱 기준선 모델에 비해 뛰어난 성능을 보였습니다. 또한, 다양한 차원에서 탐색을 위한 의미 있는 임베딩(embedding)을 구축할 수 있음을 보여주었습니다.



### A convolutional neural network approach to deblending seismic data (https://arxiv.org/abs/2409.07930)
- **What's New**: 이 논문에서는 기계 학습을 기반으로 한 데이터 주도 심층 학습(deep learning) 방식의 신속하고 효율적인 지진 데이터 디블렌딩(seismic deblending) 방법을 제시합니다.

- **Technical Details**: 지진 데이터를 위한 합성(source)과 공통 채널(channel) 영역으로 정렬된 혼합된 데이터를 사용하여 혼합 소음(blending noise)의 특성을 일관된 사건(coherent events)에서 비일관된(distributed) 형태로 변환합니다. 특수한 지진 데이터의 특성에 맞춰 설계된 합성곱 신경망(CNN)이 전통 산업 디블렌딩 알고리즘과 유사한 결과를 얻습니다. 네트워크는 20000개 이상의 훈련 예제를 포함한 필드 지진 데이터를 사용하여 훈련 및 검증되었습니다.

- **Performance Highlights**: 실험 결과, 초기 신호 대 잡음비(signal to noise ratio, SNR)가 최종 디블렌딩 결과의 품질을 좌우하는 주요 요소로 확인되었습니다. 또한, 학습된 모델을 사용하여 다른 지질 영역에서 새로운 데이터 세트를 디블렌딩하거나 잡음이 포함된 데이터의 상단 부분을 디블렌딩하는 데 있어 네트워크의 견고성과 적응성이 입증되었습니다.



### A framework for measuring the training efficiency of a neural architectur (https://arxiv.org/abs/2409.07925)
- **What's New**: 이 논문은 신경망 시스템 개발에서의 효율성 측정을 위한 실험적 프레임워크를 제시합니다. 이를 통해 MNIST 및 CIFAR-10 작업에서 CNN(Convolutional Neural Networks)과 BCNN(Bayesian Convolutional Neural Networks)의 훈련 효율성을 분석합니다.

- **Technical Details**: 훈련 효율성의 측정을 위한 자연스러운 비율을 신경 모델의 정확도와 이를 달성하기 위해 소모되는 에너지의 비율로 정의합니다. 연구 결과, 훈련 효율성이 시간이 지남에 따라 감소하고, 특정 신경 모델과 학습 작업에 따라 점검 기준이 다를 수 있음을 보여줍니다. 또한, 훈련 효율성, 모델 크기, 및 훈련 중단 기준 사이의 비선형적 관계를 발견했습니다.

- **Performance Highlights**: CNN은 MNIST 및 CIFAR-10 데이터 세트 모두에서 BCNN보다 훈련 효율성이 높으며, 학습 작업이 복잡해질수록 다른 아키텍처 간의 훈련 효율성의 상대적 차이가 더욱 드러납니다.



### Tidal MerzA: Combining affective modelling and autonomous code generation through Reinforcement Learning (https://arxiv.org/abs/2409.07918)
- **What's New**: Tidal-MerzA는 라이브 코딩( Live Coding ) 맥락에서 인간과 기계 에이전트 간의 협업 공연을 위한 새로운 시스템입니다. 이 시스템은 음악 패턴 생성을 중심으로 개발되었습니다.

- **Technical Details**: Tidal-MerzA는 두 가지 기본 모델인 ALCAA (Affective Live Coding Autonomous Agent)와 Tidal Fuzz라는 컴퓨팅 프레임워크를 결합하였습니다. 감정적 모델링( affective modelling )과 계산적 생성을 통합하여, 이 시스템은 TidalCycles 프레임워크 내에서 음악 작곡 파라미터를 동적으로 조정하는 강화 학습( reinforcement learning ) 기법을 활용합니다. 이를 통해 패턴의 감정 품질과 문법적 정확성을 모두 보장합니다.

- **Performance Highlights**: Tidal-MerzA의 개발로 인해 두 가지 독립적인 에이전트가 등장합니다. 하나는 음악 표현을 위한 미니 표기 문자열(mini-notation strings) 생성에 중점을 두고, 다른 하나는 강화 학습을 통해 목표 감정 상태와 음악을 정렬하는 것을 목표로 합니다. 이러한 접근 방식은 라이브 코딩 관행의 적응성과 창의적 잠재력을 향상시키며, 인간과 기계 간의 창작 상호작용을 탐구할 수 있게 합니다.



### InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation (https://arxiv.org/abs/2409.07914)
Comments:
          Accepted at Conference on Robot Learning (CoRL) 2024

- **What's New**: InterACT: Hierarchical attention 기반의 이펙타치기(Action Chunking) 프레임워크를 소개합니다. 이 프레임워크는 이양 손 조인트 상태와 시각적 입력 간의 상호 의존성을 포착하여 이양 조작(bimanual manipulation)을 위한 새로운 모방 학습(imitation learning) 방식을 제공합니다.

- **Technical Details**: InterACT은 Hierarchical Attention Encoder와 Multi-arm Decoder로 구성되어 있으며, 세그먼트 단위의 인코딩과 크로스 세그먼트 인코딩을 통해 다중 모드 입력을 처리합니다. 세그먼트 인코더는 각 세그먼트 내의 정보 집합을 추구하여 상호 의존성을 캡처합니다. 크로스 세그먼트 인코더는 여러 세그먼트 간의 관계를 포착하여 각 팔이 연속적이고 효율적으로 협력할 수 있도록 합니다.

- **Performance Highlights**: 다양한 시뮬레이션 및 실제 세계의 이양 조작 작업에 대한 실험 결과, InterACT은 기존 방법보다 탁월한 성능을 보여줍니다. 중요한 구성 요소인 CLS 토큰, 크로스 세그먼트 인코더 및 동기화 블록의 기여도를 평가하는 상세한 제거 연구(ablation studies)가 수행되었습니다.



### UGAD: Universal Generative AI Detector utilizing Frequency Fingerprints (https://arxiv.org/abs/2409.07913)
- **What's New**: 본 논문에서는 AI 생성 이미지와 진짜 이미지를 구분하기 위한 새로운 다중 모달 접근법인 UGAD를 제안합니다. 인공지능 기술의 발전으로 인해 가짜 이미지의 생성이 더욱 용이해졌으며, 이에 따라 실체 이미지와 생성된 이미지를 식별하는 것이 매우 중요해졌습니다.

- **Technical Details**: UGAD는 세 가지 핵심 단계로 구성되어 있습니다: 우선, RGB 이미지를 YCbCr 채널로 변환하고, 적분 방사형 연산(Integral Radial Operation, RIO)을 적용하여 두드러진 방사형 특징을 강조합니다. 두 번째로, 공간 푸리에 추출(Spatial Fourier Extraction, SFE) 작업을 통해 공간 이동을 수행하며, 이는 사전 훈련된 깊은 학습 네트워크를 활용하여 최적의 특징을 추출합니다. 마지막으로, 깊은 신경망 분류 단계에서 소프트맥스를 사용하여 데이터를 처리하고 분류합니다.

- **Performance Highlights**: 제안된 방법은 기존 최첨단 기법에 비해 정확도에서 12.64% 증가, AUC에서 28.43% 향상된 결과를 보여줍니다. 또한, 최신 AI 생성 방법에 의해 생성된 이미지에 대해 정밀한 탐지가 가능함을 입증하였습니다.



### Enhancing Cross-Market Recommendation System with Graph Isomorphism Networks: A Novel Approach to Personalized User Experienc (https://arxiv.org/abs/2409.07850)
Comments:
          7 pages, 1 figure, 3 tables, 5 equations

- **What's New**: 본 논문에서는 Graph Isomorphism Networks (GINs)를 활용한 CrossGR 모델을 제안하여 cross-market recommendation 시스템(CMRs)의 성능을 향상시키고 있습니다. 이 모델은 NDCG@10 및 HR@10 메트릭에서 기존 벤치마크를 초월하며, 다양한 시장 세그먼트를 처리하는 데 있어 적응성과 정확성을 나타냅니다.

- **Technical Details**: CrossGR 모델은 GNNs (Graph Neural Networks)를 사용하여 시장별 특성과 데이터 희소성을 극복하고 있으며, 고차 연결성 (high-order connectivity)을 통해 더 복잡한 패턴과 관계를 감지할 수 있도록 설계되었습니다. 이로 인해 CMR 시스템에서 사용자 맞춤형 추천을 제공할 수 있습니다.

- **Performance Highlights**: CrossGR 모델은 평가 시간에 관계없이 일관된 성능을 보여줘, 새로운 시장 동향 및 사용자 선호도를 반영할 수 있는 잠재력을 가지고 있습니다. 이는 글로벌 전자상거래의 동적 환경에서 사용자 친화적인 추천 시스템으로 발전할 수 있는 기회를 제공합니다.



### A Comprehensive Survey on Deep Multimodal Learning with Missing Modality (https://arxiv.org/abs/2409.07825)
Comments:
          Work in progress and welcome to discussion

- **What's New**: 이 문서는 Multimodal Learning with Missing Modality (MLMM)에 대한 최근 발전을 포괄적으로 다루고 있으며, 특히 딥러닝 기술에 초점을 맞춘 최초의 종합적인 설문조사입니다. MLMM과 전통적인 Multimodal Learning with Full Modality (MLFM)의 차이를 분명히 하고, 현재의 MLMM 방법 및 응용분야, 데이터셋에 대한 깊이 있는 분석을 제공합니다.

- **Technical Details**: MLMM의 주요 과제는 데이터 수집의 제한이나 배포 환경의 제약으로 인해 훈련 및 테스트 동안 여러 개의 모달리티를 동적으로 처리하고 통합하는 것입니다. 이 설문조사는 MLMM 방법론의 구조를 모달리티 보강, 특성 공간 설계, 아키텍처 설계 및 모델 선택의 네 가지 주요 차원에서 분류합니다.

- **Performance Highlights**: MLMM 접근법은 의료 진단, 정보 검색, 원거리 감지 및 로봇 비전 등 다양한 분야에서 응용되고 있으며, 이 설문조사는 이러한 응용 프로그램을 통해 MLMM의 현장 적용 가능성을 강조합니다.



### Over-the-Air Federated Learning via Weighted Aggregation (https://arxiv.org/abs/2409.07822)
- **What's New**: 이 논문은 채널 상태 정보(CSIT)를 필요로 하지 않고 적응형 가중치를 사용하여 페더레이티드 러닝(Federated Learning) 성능을 향상시키는 새로운 오버더에어 연산(Over-the-Air Computation) 방식을 제안합니다.

- **Technical Details**: 제안된 방식은 가중치 오버더에어 페더레이티드 러닝(WAFeL)으로, 계산 이질성(computational heterogeneity)과 일반적인 손실 함수에 대한 수렴 경계를 수학적으로 도출합니다. 이 방식은 장치의 다양한 계산 능력을 활용하여 최적화된 가중치를 사용하여 집계(aggregation)하는 알고리즘을 포함합니다.

- **Performance Highlights**: WAFeL 방식은 CSIT를 사용하는 기존 방법보다 15%, CSIT 없이 사용하는 방법보다 30% 향상된 정확성을 보여주었으며, 다양한 채널 조건과 장치 이질성에도 불구하고 효과적인 성능을 입증하였습니다.



### In-Situ Fine-Tuning of Wildlife Models in IoT-Enabled Camera Traps for Efficient Adaptation (https://arxiv.org/abs/2409.07796)
- **What's New**: 이 논문에서는 WildFit이라는 새로운 접근 방식을 도입하여 머신 러닝 모델이 카메라 트랩 애플리케이션에서 높은 도메인 일반화 성능을 유지하면서도 효율적인 추론을 가능하게 합니다.

- **Technical Details**: WildFit은 환경 변화에 맞춰서 모델을 지속적으로 미세 조정하는 방법을 활용합니다. 이는 백그라운드 이미지와 소스 도메인에서의 동물 이미지를 혼합하여 새로운 도메인을 대표하는 훈련 이미지를 생성하는 백그라운드 인식 데이터 합성을 통해 이루어집니다. 이를 통해 복잡한 계산 리소스 없이도 새로운 환경에 대해 높은 분류 정확도를 유지할 수 있습니다.

- **Performance Highlights**: WildFit은 여러 카메라 트랩 데이터셋에 대한 광범위한 평가를 통해 전통적인 접근 방식에 비해 분류 정확도와 계산 효율성에서 현저한 개선을 달성했습니다.



### Lagrange Duality and Compound Multi-Attention Transformer for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2409.07793)
Comments:
          5 pages, 4 figures, 3 tables

- **What's New**: 이 논문에서는 CMAformer라는 새로운 모델을 제안하며, 이는 ResUNet과 Transformer의 장점을 결합한 것입니다. 또한, 장기 분포 문제를 해결하기 위해 Lagrange Duality Consistency (LDC) Loss와 Boundary-Aware Contrastive Loss를 통합한 반지도 학습 프레임워크를 소개합니다.

- **Technical Details**: CMAformer는 spatial attention과 channel attention을 효과적으로 융합하기 위해 Cross Attention 레이어를 사용하며, 이는 멀티스케일 특성 융합을 위한 중요한 요소입니다. 또한, Transformer 블록 내에서 multi-head self-attention (MSA)과 다층 퍼셉트론 (MLP)이 포함되어 각 쿼리 매트릭스에 대해 개별적으로 레이어 정규화를 수행하고 있습니다.

- **Performance Highlights**: CMAformer는 여러 공공 의료 이미지 데이터셋에서 세분화 작업에 대해 대부분의 최신 모델을 능가하는 성능을 입증하였습니다. 이를 통해 반지도 학습 앙상블에서의 강력한 상호 보완성을 보여주고 있습니다.



### ASSNet: Adaptive Semantic Segmentation Network for Microtumors and Multi-Organ Segmentation (https://arxiv.org/abs/2409.07779)
Comments:
          8 pages, 4 figures, 3 tables

- **What's New**: 이번 연구에서는 의학 영상 분할을 위한 Adaptive Semantic Segmentation Network (ASSNet)라는 새로운 Transformer 기반 아키텍처를 제안합니다. ASSNet은 로컬 및 글로벌 피쳐 통합을 통해 미세한 종양 및 소형 기관의 정확한 분할을 지원합니다.

- **Technical Details**: ASSNet은 U자형 인코더-디코더 네트워크로 구성되며, 인코더는 다섯 가지 해상도에서 이동 윈도우 셀프 어텐션(shifted window self-attention)을 활용해 다중 스케일 특성을 추출합니다. 디코더는 Adaptive Feature Fusion (AFF) 구조를 통해 장거리 의존성(long-range dependencies)과 다중 스케일 피쳐 융합을 지원합니다.

- **Performance Highlights**: ASSNet은 LiTS2017, ISICDM2019, Synapse 데이터셋에서 최신 기술 점수를 초과하며, 다양한 의학 이미지 분할 작업에서 최첨단 성능을 보여주었습니다.



### Training Spiking Neural Networks via Augmented Direct Feedback Alignmen (https://arxiv.org/abs/2409.07776)
Comments:
          20 pages, 8 figures, 2 tables

- **What's New**: 이 논문에서는 Spiking Neural Networks (SNNs)의 훈련 방법으로 Augmented Direct Feedback Alignment (aDFA)를 제안합니다. 기존의 비분화성 가중치 조정 문제를 해결하기 위해, 무작위 투영 기반의 기울기 없는 방법을 사용하여 SNN을 훈련하는 것이 물리적으로 실제 구현 가능하면서 생물학적으로 실행 가능하다는 점에서 중요한 의미를 지닙니다.

- **Technical Details**: aDFA는 사전 정보의 일부만을 요구하며, 고정된 무작위 초기화된 시냅스 가중치를 통해 전역 오차를 각 층에 직접 주입합니다. 이 방식은 신경망의 훈련 과정에서 생물학적으로 더 타당하고, 물리적으로 구현하기 쉬운 방법입니다. 저자들은 유전 알고리즘(Genetic Algorithm, GA)을 사용하여 최대 성능을 발휘할 수 있는 적절한 후방 함수의 설정을 분석하였습니다.

- **Performance Highlights**: aDFA-SNNs는 기존의 BP(Back Propagation)와 전통적인 직접 피드백 정렬에 비해 우월한 성능과 안정성을 보여주며, 시스템에 대한 정확한 사전 지식 없이도 경쟁력 있는 성능을 달성할 수 있음을 증명하였습니다.



### Universal Pooling Method of Multi-layer Features from Pretrained Models for Speaker Verification (https://arxiv.org/abs/2409.07770)
Comments:
          Preprint

- **What's New**: 본 연구는 자동 음성 인증(ASV) 기술의 최신 진전을 다루고 있으며, 대규모로 사전 훈련된 네트워크를 활용하여 ASV의 다층적 특성을 극대화하는 새로운 접근 방식을 제시합니다. 특히, 레이어(Level)/프레임(Frame) 수준의 네트워크와 각 레이어 및 프레임 축을 위한 두 단계의 풀링 아키텍처를 포함하는 구조를 소개합니다.

- **Technical Details**: 제안된 방법론은 사전 훈련된 모델의 여러 레이어 출력을 활용하여, 합성곱 신경망(convolutional architecture)을 직접 활용하여 스피커 관련 특성을 생성합니다. 이를 통해 가장 대표적인 값으로 레이어의 중요성을 가늠하는 채널 주의(channel attention) 기반의 방식을 통해 레이어 간 처리를 수행합니다. 최종적으로 프레임 수준 표현에 대한 주의 통계(attentive statistics)를 활용하여 단일 벡터 스피커 임베딩을 생성합니다.

- **Performance Highlights**: 실험 결과는 사전 훈련된 아키텍처를 활용한 다층 출력의 안정성과 성능 향상을 확인하였으며, 전통적인 방법에 비해 비용 효율성을 보여주었습니다. 다양한 데이터 환경과 다채로운 사전 훈련 모델을 이용한 비교 실험을 통해 제안한 방법의 유니버설성과 성능 개선 효과가 입증되었습니다.



### Reimagining Linear Probing: Kolmogorov-Arnold Networks in Transfer Learning (https://arxiv.org/abs/2409.07763)
Comments:
          10 pages, 5 figure

- **What's New**: 이 논문은 전이 학습(transfer learning)에서 전통적인 linear probing 방식을 개선한 Kolmogorov-Arnold Networks (KAN)를 소개합니다. KAN은 복잡한 데이터 관계를 모델링할 수 있는 스플라인 기반의 표현을 활용하여 linear probing 레이어를 대체합니다.

- **Technical Details**: 본 연구에서는 ImageNet에서 사전 훈련된 ResNet-50 모델에 KAN을 통합하고 CIFAR-10 데이터 세트에서 성능을 평가했습니다. KAN의 유연성과 정확성을 최적화하기 위해 grid size와 spline degree (k)에 대한 체계적인 하이퍼파라미터 탐색이 진행되었습니다.

- **Performance Highlights**: KAN은 전통적인 linear probing 방식을 지속적으로 초월하여 다양한 설정에서 정확도와 일반화 성능에서 중대한 향상을 이루었습니다. 이 결과는 KAN이 전이 학습에서 더 강력하고 적응 가능한 옵션임을 시사합니다.



### Relevance for Human Robot Collaboration (https://arxiv.org/abs/2409.07753)
- **What's New**: 이 논문에서는 효과적인 인간-로봇 협력(HRC)을 위해 로봇이 인간과 유사한 지능을 갖추어야 한다는 점을 강조하며, 새로운 개념인 'relevance'(관련성)를 도입합니다.

- **Technical Details**: 이 논문은 장면을 이해하는 접근 방식을 제안하여 장면의 모든 요소를 식별하고 이를 관련 클래스(group)로 구분하여, 각 요소의 긍정적(강화), 부정적(방해), 중립적(선택적) 효과를 파악합니다. 이 과정에서 이벤트 기반(event-based) 프레임워크와 확률적(probabilistic) 방법론을 개발하여 계산 속도를 크게 향상시킵니다.

- **Performance Highlights**: 시뮬레이션 결과, proposed relevance framework는 HRC 설정에서 0.99의 정밀도(precision)와 0.94의 재현율(recall)을 기록하였으며, 시간 계획(Task Planning Time)을 79.56% 단축시키고, 객체 탐지기의 인식 지연(perception latency)을 최대 26.53% 감소시키는 등의 성과를 보였습니다.



### Top-down Activity Representation Learning for Video Question Answering (https://arxiv.org/abs/2409.07748)
Comments:
          presented at MIRU2024

- **What's New**: 이 논문에서는 비디오 질문 응답(VideoQA) 작업에서 복잡한 계층적 인간 활동을 효과적으로 캡처하기 위해 CLIP 모델의 공간 시각적 상황 표현 능력을 활용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 접근 방식은 긴 비디오 시퀀스를 단일 그리드 이미지로 변환하고, LLaVA 멀티모달 모델을 파인튜닝합니다. 이를 통해 비디오의 맥락적 사건을 비연속적으로 표현할 수 있으며, STAR와 NExTQA 벤치마크에서 높은 성능을 달성했습니다.

- **Performance Highlights**: 이 연구는 STAR 작업에서 78.4%의 정확도를 기록하였으며, NExTQA 작업에서는 기존의 최첨단 점수를 2.8포인트 초과하는 성과를 보여주었습니다.



### Multi-object event graph representation learning for Video Question Answering (https://arxiv.org/abs/2409.07747)
Comments:
          presented at MIRU2024

- **What's New**: 이 논문에서는 비디오 질문 응답(Video Question Answering, VideoQA) 분야에서 다수 객체를 포함한 사건(event) 표현을 포착하기 위한 새로운 접근법인 CLanG(Contrastive Language Event Graph Representation Learning)를 제안하고 있습니다. 기존의 방법들은 개별 객체의 움직임에 초점을 맞추었지만, CLanG는 복잡한 시나리오를 처리할 수 있는 방법을 제공합니다.

- **Technical Details**: CLanG는 다층 GNN-클러스터 모듈을 사용하여 비디오에서 추출된 다수 객체의 사건 표현을 효과적으로 학습합니다. 이 모듈은 경쟁적 그래프 표현 학습을 통해 질문 텍스트와 관련된 다수 객체 사건 그래프 간의 대비 학습을 가능하게 합니다. 또한, 그래프의 출력 노드 수를 조정하여 원래의 그래프 정보를 보존하며, 자기 주의(self-attention) 계층을 통해 계층적 사건 그래프의 영향을 강조합니다.

- **Performance Highlights**: CLanG는 NExT-QA 및 TGIF-QA-R이라는 두 개의 도전적인 VideoQA 데이터셋에서 강력한 기준선보다 최대 2.2% 더 높은 정확도를 달성하며, 특히 인과(causal) 및 시간적(temporal) 질문 처리에서 2.8% 향상된 성능을 보여줍니다.



### Transfer Learning Applied to Computer Vision Problems: Survey on Current Progress, Limitations, and Opportunities (https://arxiv.org/abs/2409.07736)
Comments:
          16 pages, 8 figures

- **What's New**: 본 연구는 Transfer Learning (TL) 기술을 활용하여 컴퓨터 비전 (CV) 문제들을 해결하는 새로운 접근 방식을 제시합니다. 기존의 컴퓨터 비전 기술이 가진 한계들을 극복하기 위해 TL의 최근 발전 상황과 이로 인한 기회를 탐구하고 있습니다.

- **Technical Details**: 본 연구는 인공지능 (AI), 머신러닝 (ML), 신경망 (NN), 딥 뉴럴 네트워크 (DNN), 컨볼루션 뉴럴 네트워크 (CNN), 순환 신경망 (RNN) 등의 개념들을 다루며, TL이 어떻게 CV 문제에 적용되는지를 설명합니다. TL은 한 도메인에서 얻은 지식을 다른 도메인에서의 학습 과정에 활용하여, 데이터가 부족할 때 효과적으로 모델을 학습시키는 기법입니다.

- **Performance Highlights**: TL을 활용하여 적은 데이터로도 높은 정확도를 달성할 수 있다는 점에 주목하십시오. 예를 들어, COVID-19 발생 초기에 데이터 부족 문제를 해결하기 위해 TL을 통해 기존 모델을 개조하여 성공적으로 질병을 감지하는 모델이 개발되었습니다. 이는 TL이 다양한 컴퓨터 비전 문제 해결에 매우 유용할 수 있음을 시사합니다.



### GRE^2-MDCL: Graph Representation Embedding Enhanced via Multidimensional Contrastive Learning (https://arxiv.org/abs/2409.07725)
- **What's New**: 이번 논문에서는 Graph Representation Embedding Enhanced via Multidimensional Contrastive Learning (GRE2-MDCL)이라는 새로운 그래프 표현 학습 모델을 제안합니다. 이 모델은 멀티헤드 어텐션 그래프 신경망(GNN)을 핵심으로 하는 독창적인 삼중 네트워크 아키텍처를 도입하며, 그래프의 글로벌 및 로컬 구조를 효과적으로 보강하는 기술을 활용하여 기존 모델들이 가지던 한계를 극복하고자 합니다.

- **Technical Details**: GRE2-MDCL 모델은 그래프 데이터를 먼저 SVD 및 LAGNN 기법을 사용하여 글로벌 및 로컬 차원에서 증대시킵니다. 이 후, 다양한 그래프 비교 방식(예: cross-network, cross-view, neighbor contrast)을 적용하여 멀티디멘션(Multidimensional) 그래프 대조 손실을 구성하고, 이를 통해 모델을 최적화합니다.

- **Performance Highlights**: Cora, Citeseer, PubMed의 벤치마크 데이터셋에서 GRE2-MDCL은 각각 82.5%, 72.5%, 81.6%의 평균 정확도를 달성하며, 시각화 결과에서도 클러스터 간 명확한 경계를 보여 기존 GCL 모델보다 우수한 성능을 입증하였습니다.



### Advancing Depth Anything Model for Unsupervised Monocular Depth Estimation in Endoscopy (https://arxiv.org/abs/2409.07723)
Comments:
          7 pages, 6 figures

- **What's New**: 이 연구에서는 Depth Anything Model을 위한 새로운 파인튜닝 전략인 RVLoRA를 소개하며, 이는 깊이 추정의 성능을 향상시키기 위해 랜덤 벡터를 기반으로 하는 저랭크 적응(low-rank adaptation) 기법을 통합합니다.

- **Technical Details**: LVLoRA는 다양한 스케일에 대한 모델의 적응성을 향상시키며, Res-DSC 모듈은 Vision Transformer(ViT) 내에 CNN 구성을 통합하여 고주파 세부정보, 즉 에지 및 텍스처와 같은 지역적 특징을 더 잘 포착할 수 있게 합니다.

- **Performance Highlights**: SCARED 데이터셋에서 실험한 결과, 제안된 방법은 최소한의 학습 가능한 파라미터로 최신 성능(state-of-the-art performance)을 달성하였습니다. 이 방법은 최소 침습 내시경 수술의 정확성과 안전성을 크게 향상시킬 것으로 기대됩니다.



### FIReStereo: Forest InfraRed Stereo Dataset for UAS Depth Perception in Visually Degraded Environments (https://arxiv.org/abs/2409.07715)
Comments:
          Under review in RA-L. The first 2 authors contributed equally

- **What's New**: 이 논문은 자율 항공 시스템을 위한 시각적으로 저하된 환경에서의 스테레오 열 깊이 인식 데이터 세트 FIReStereo를 소개합니다. 이 데이터 세트는 다양한 조건에서 수집된 스테레오 열 이미지와 LiDAR, IMU, 그리고 실제 깊이 맵(ground truth depth maps)으로 구성되어 있습니다. 특히, 이 데이터 세트는 재난 대응 상황에서 로봇의 인식을 향상시키기 위해 설계되었습니다.

- **Technical Details**: FIReStereo 데이터 세트는 열 화상 카메라 두 대를 이용하여 수집되었으며, 구조물과 숲이 혼합된 환경에서의 깊이 추정 알고리즘 개발에 도움을 주기 위해 만들어졌습니다. 카메라는 24cm의 작은 간격으로 배치되어 있으며, 스모크 및 각종 날씨 조건에서 촬영된 데이터가 포함됩니다. 모든 센서 데이터는 동기화되어 제공되며, LiDAR 데이터를 이용한 실제 깊이 정보도 포함됩니다. 이 데이터는 로봇이 복잡한 환경을 탐색하는 데 필요한 정보를 제공합니다.

- **Performance Highlights**: 훈련된 모델들은 이전에 본 적 없는 스모키한 조건에서도 우수한 일반화 성능을 보여주었으며, 이는 스테레오 열 화상이 깊이 인식을 위한 강건함을 잘 나타냅니다. 다양한 환경 조건에서의 성능을 벤치마킹하여, 자율 비행체 플랫폼에서의 실시간 메트릭 깊이 추정의 강점과 한계를 분석하였습니다.



### Attack End-to-End Autonomous Driving through Module-Wise Nois (https://arxiv.org/abs/2409.07706)
- **What's New**: 본 논문은 모듈식 엔드투엔드(End-to-End) 자율 주행 모델에 대한 적대적인 보안 연구를 최초로 수행하였습니다. 이 연구는 모델 추론 과정에서의 잠재적 취약점을 검토하고, 모듈별 노이즈 주입을 통한 범용 공격 방식을 설계합니다.

- **Technical Details**: 연구는 n개의 하위 모듈로 구성된 엔드투엔드 자율 주행 모델을 사용하여, 입력 이미지에 적대적 노이즈를 주입함으로써 공격 목표를 설정합니다. 각 모듈의 상호작용 정보를 취약점으로 고려하고, 주입된 노이즈로 인해 의사결정 결과가 변경되는 방식으로 실험을 진행하였습니다.

- **Performance Highlights**: 대규모 실험을 통해 기존 공격 방법보다 향상된 성과를 나타내었으며, 본 연구는 자율 주행 시스템의 안전성과 신뢰성을 보장하기 위한 새로운 통찰력을 제공할 것으로 기대합니다.



### Super Monotonic Alignment Search (https://arxiv.org/abs/2409.07704)
Comments:
          Technical Report

- **What's New**: 이 논문은 Glow-TTS에서 도입한 Monotonic Alignment Search (MAS) 알고리즘을 GPU에서 가속화하는 새로운 방법을 제시합니다.

- **Technical Details**: MAS 알고리즘은 텍스트와 음성 간의 정렬을 추정하는 데 사용되며, 기존 방법은 CPU에서 동적 프로그래밍을 사용하여 $O(T\times S)$의 시간 복잡도를 가지고 있습니다. 저자들은 Triton 커널과 PyTorch JIT 스크립트를 사용하여 MAS를 GPU에서 병렬화했습니다. 특히 Super-MAS Triton 커널은 극단적인 길이의 경우 최대 72배 빠른 성능을 보입니다.

- **Performance Highlights**: 기존 Cython 구현에 비해 Super-MAS Triton 커널이 최소 19배에서 최대 72배 빠른 성능을 제공합니다. 이 결과는 CPU에서 대형 텐서를 복사하는 데 필요한 비효율성을 줄여주며, GPU에서 더 효율적입니다.



### Modeling Information Narrative Detection and Evolution on Telegram during the Russia-Ukraine War (https://arxiv.org/abs/2409.07684)
Comments:
          12 pages, International AAAI Conference on Web and Social Media 2025

- **What's New**: 이 연구는 우크라이나에 대한 러시아 연방의 전면 침공 이후에 나타난 다양한 정보 서사(narratives)의 진화를 탐구하는 새로운 접근 방식을 제시합니다. 독특한 점은 서로 다른 커뮤니티에서의 서사 변화와 그 메커니즘(mechanisms)을 모델링(model)하고 분석했다는 것입니다.

- **Technical Details**: 연구는 Telegram 플랫폼 내의 커뮤니티를 대상으로 진행되었으며, 침공 이후의 첫 3개월 동안의 담론 분석(discourse analysis)을 수행하였습니다. 연구팀은 친러시아(pro-Russian) 및 친우크라이나(pro-Ukrainian) 커뮤니티의 서사 및 인식(perceptions) 간의 차이를 발견했습니다. 또한, 각 그룹의 주요 주제와 서사의 진화를 촉진하는 근본적인 메커니즘을 분석하였습니다.

- **Performance Highlights**: 연구 결과는 두 커뮤니티 간 서사의 큰 격차를 입증하며, 정보 서사의 진화가 온라인 커뮤니티의 인식과 태도에 미치는 영향을 명확히 하였습니다. 이러한 통찰력은 정보 환경(information environment)의 복잡성을 이해하고 대응하는 데 중요한 기여를 할 것으로 보입니다.



### Open-Vocabulary Remote Sensing Image Semantic Segmentation (https://arxiv.org/abs/2409.07683)
- **What's New**: 본 논문에서는 원격 감지 이미지에 특화된 첫 번째 오픈 어휘 이미지 의미 분할(OVS) 프레임워크를 제안합니다. 기존 방법들과 달리 이 프레임워크는 빠르게 변화하는 방향과 상당한 크기 변화를 다루기 위해 개발되었습니다.

- **Technical Details**: 본 연구에서는 회전 집계 유사도 계산 모듈을 도입하여 초기 의미 맵을 생성합니다. 이 모듈은 입력 원격 감지 이미지를 여러 방향으로 회전시켜 방향 적응형 유사도 맵을 생성하고, 이 맵을 활용해 더 정교한 의미 맵을 생성합니다. 또한 다중 스케일 이미지 특징을 업샘플링 과정에 통합하여 최종적으로 크기 인식 의미 마스크를 만듭니다.

- **Performance Highlights**: 제안된 방법은 새로운 원격 감지 OVS 벤치마크에서 extensive experiments를 통해 기존의 최첨단 자연 이미지 기반 OVS 접근 방식들을 능가하는 성능을 입증하였습니다.



### An Unsupervised Dialogue Topic Segmentation Model Based on Utterance Rewriting (https://arxiv.org/abs/2409.07672)
Comments:
          in Chinese language

- **What's New**: 이번 연구에서는 다중 대화에서의 주제 세분화(Dialog topic segmentation)를 위한 새로운 비지도 학습(unsupervised learning) 모델인 Discourse Rewriting Topic Segmentation Model (UR-DTS)을 제안합니다. 이 모델은 대화 내용에서 동시 지시어(co-referent)와 생략된 단어들을 복원하기 위해 Utterance Rewriting (UR) 기법을 결합합니다.

- **Technical Details**: UR-DTS 모델은 인접 담화 매칭(adjacent discourse matching)과 의사 세분화(pseudo segmentation)를 통해 학습할 수 있는 유용한 단서를 최대한 활용하기 위해 대화 데이터를 재작성합니다. 이로 인해, 담화의 의미적 유사성 계산이 향상됩니다.

- **Performance Highlights**: DialSeg711에서의 절대 오류 점수(absolute error score)가 약 6% 향상되어 11.42%를 기록하였고, WD도 12.97%로 개선되었습니다. Doc2Dial에서는 절대 오류 점수가 약 3%, WD가 2% 향상되어 각각 35.17% 및 38.49%를 기록하며 SOTA(State-Of-The-Art)에 도달하였습니다.



### Feature Importance in Pedestrian Intention Prediction: A Context-Aware Review (https://arxiv.org/abs/2409.07645)
- **What's New**: 이번 연구에서는 Context-aware Permutation Feature Importance (CAPFI)라는 새로운 접근 방식을 도입하여 보행자의 횡단 의도를 예측하는 모델의 해석 가능성을 높였습니다. 이 방법은 시나리오 맥락을 세분화하여 입력 특성이 최종 예측에 기여하는 방식을 보다 명확하게 파악할 수 있도록 합니다.

- **Technical Details**: CAPFI는 목표 특성들에 대한 무작위 셔플링을 통해 특성 중요성을 평가하는 방법입니다. 본 연구에서는 Pedestrian Intention Estimation (PIE) 데이터셋을 16개의 비교 가능한 맥락 집합으로 나누고, 각각의 맥락에서 다섯 개의 신경망 아키텍처의 성능을 평가했습니다. CAPFI는 보행자 경계 상자와 자가차량의 속도가 보행자 의도 예측에 미치는 중요성을 강조합니다.

- **Performance Highlights**: 연구 결과, 보행자의 경계 상자와 자가 차량의 속도가 보행자 의도 예측에서 중요한 역할을 하며, 속도 특성의 타당한 예측 편향이 발생할 수 있음을 밝혔습니다. 또한, 보행자-차량 상호작용을 고려한 대안적인 특성 표현을 제안함으로써 입력 특성이 의도 예측에 미치는 기여도를 향상시켰습니다.



### Weather-Informed Probabilistic Forecasting and Scenario Generation in Power Systems (https://arxiv.org/abs/2409.07637)
- **What's New**: 본 논문은 확률적 예측(probablistic forecasting)과 가우시안 코풀라(Gaussian copula)를 접목하여 고차원 환경에서의 일일 예측 및 시나리오 생성을 제안합니다. 기상 정보(weather covariates)를 통합하고 시공간 상관관계(spatio-temporal correlations)를 복원함으로써, 재생 가능 에너지 소스 (RES)의 예측 정확성을 향상시킵니다.

- **Technical Details**: 제안된 방법은 기상 정보와 함께 다변량 시계열 예측(multivariate time series forecasting)을 활용하여 고차원 예측 문제를 해결합니다. 본 연구는 미드컨티넨트 독립 시스템 운영자(MISO)로부터 얻은 실제 고차원 데이터셋을 활용하여 다양한 시계열 모델의 효과성을 비교하며, 전반적인 성능을 종합적 기준(comprehensive metrics)으로 평가합니다.

- **Performance Highlights**: 제안된 날씨 정보 기반 시계열 융합 변환기(Weather-informed Temporal Fusion Transformer, WI-TFT) 모델은 실제 시나리오를 생성하는 데 있어 우수한 성능을 보여주며, 기상 정보의 중요성을 강조합니다.



### Dividable Configuration Performance Learning (https://arxiv.org/abs/2409.07629)
Comments:
          Submitted to TSE as a regular journal paper. arXiv admin note: text overlap with arXiv:2306.06651

- **What's New**: 본 논문에서는 소프트웨어 시스템의 구성 성능을 예측하기 위한 모델 불가지론적(model-agnostic)이고 희소성 강건한(sparsity-robust) 프레임워크인 DaL을 제안합니다. 이는 'divide-and-learn'이라는 새로운 패러다임을 기반으로 하여 구성 옵션과 데이터 샘플의 희소성이 높은 환경에서 효과적으로 성능을 예측합니다.

- **Technical Details**: DaL은 샘플 희소성을 처리하기 위해 구성 데이터 샘플을 먼 구역으로 나눈 후 각 구역에 대해 희소(local) 모델을 구축합니다. 이러한 모델들은 규제된 Hierarchical Interaction Neural Network와 같은 방법을 활용하여 특징(feature) 희소성을 해결합니다. 또한, DaL은 추가적인 훈련이나 프로파일링 없이 시스템과 샘플 크기에 필요한 최적의 구역 수를 적응적으로 결정합니다.

- **Performance Highlights**: 12개의 실제 시스템과 5세트의 훈련 데이터에서 실험 결과, DaL은 기존 최첨단 접근 방식에 비해 60개 사례 중 44개에서 동등한 성능을 보이며 최대 1.61배 향상된 정확도를 기록했습니다. 또한, 동일한 혹은 더 나은 정확도를 달성하는 데 필요한 샘플 수가 적었으며, 훈련 오버헤드도 수용가능한 수준을 유지했습니다.



### Ensemble Methods for Sequence Classification with Hidden Markov Models (https://arxiv.org/abs/2409.07619)
- **What's New**: 본 연구에서는 Hidden Markov Models (HMMs)의 앙상블 방법을 사용하여 가벼운 시퀀스 분류 접근 방식을 제안합니다. 이 방법은 특히 데이터 불균형이 심한 경우에 효과적이며, 다양한 길이의 시퀀스를 비교할 수 있는 점이 특징입니다.

- **Technical Details**: HMM은 고차원 데이터 처리와 다양한 시퀀스 길이를 처리하는 데 강력한 성능을 발휘합니다. 새로운 학습 접근 방식은 다중 클래스 문제로 일반화할 수 있으며, 순차적으로 중요한 점들을 비교하여 복합 점수를 생성합니다. 이 방법은 SVM 및 신경망과 같은 다운스트림 분류기와 결합하여 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 제안된 방법은 여러 벤치마크에서 견고한 성능을 보이며, 특히 데이터가 적은 환경에서 HMM의 효율성과 강건성을 강조합니다. 평균 정밀도와 AUC가 높아, 다양하고 실용적인 애플리케이션을 위한 유연한 프레임워크를 제공합니다.



### The Role of Deep Learning Regularizations on Actors in Offline RL (https://arxiv.org/abs/2409.07606)
Comments:
this https URL

- **What's New**: 이번 연구에서는 오프라인 강화 학습(Offline Reinforcement Learning)에서 액터 네트워크(Actor Network)에 딥러닝 정규화 기법(regularization techniques)을 적용했을 때 평균 6%의 성능 향상을 보여줍니다. 이는 일반화(generalization) 문제를 해결하는 데 도움을 줄 수 있음을 시사합니다.

- **Technical Details**: 오프라인 RL에서는 고정된 데이터셋을 활용하여 최적의 정책(policy)을 학습해야 하며, 액터 네트워크의 성능이 알고리즘의 병목 현상(bottleneck)이 됩니다. 본 연구에서는 Dropout, Weight Decay 및 Layer Normalization 등의 정규화 기법을 D4RL 벤치마크(benchmark)에서 ReBRAC 및 IQL 알고리즘을 사용하여 실험하였습니다.

- **Performance Highlights**: 정규화 기법을 적용함으로써 액터 네트워크의 일반화 능력이 개선되었으며, 온라인 RL에서는 대부분 효과가 없던 정규화 기술이 오프라인 RL에서는 성능 향상에 기여할 수 있음을 확인했습니다.



### Efficient Localized Adaptation of Neural Weather Forecasting: A Case Study in the MENA Region (https://arxiv.org/abs/2409.07585)
Comments:
          Our codebase and pre-trained models can be accessed at: [this url](this https URL)

- **What's New**: 기후 모델링의 효율성과 정확성을 높이기 위해 Low-Rank Adaptation (LoRA) 및 그 변형들을 활용하여 특정 지역의 날씨 예측을 수행하는 신경망 기반 모델을 개발하였다. 본 연구는 MENA 지역을 사례로 선정하여, 해당 지역의 기후적 특수성을 반영한 맞춤형 예측 접근 방식을 제안한다.

- **Technical Details**: 우리의 모델은 Vision Transformer (ViT) 아키텍처를 활용하며, 예측을 위해 D×H×W 형태의 입력을 처리한다. LoRA는 훈련 가능한 매개 변수를 통해 대규모 모델의 효율성을 높이는 접근법으로, 기존의 완전한 미세 조정 없이도 유연하게 여러 LoRA 모듈을 통합할 수 있다. 모델은 두 가지 해상도(5.625∘ 및 1.40625∘)에서 작동한다.

- **Performance Highlights**: 본 연구는 LoRA를 통해 기후 모델링의 예측 정확도 향상과 함께 훈련 속도, 메모리 효율성에도 기여할 것으로 예상된다. 플래시 어텐션(flash attention) 메커니즘을 도입하여 주의 계산 과정을 가속화하고, 리소스 효율적인 모델 훈련을 가능하게 한다.



### DS-ViT: Dual-Stream Vision Transformer for Cross-Task Distillation in Alzheimer's Early Diagnosis (https://arxiv.org/abs/2409.07584)
Comments:
          8 pages, 3 figures, 3 tables

- **What's New**: 이 연구에서는 알츠하이머병 진단을 위한 자원 효율을 극대화하고 분류 성능을 향상시키기 위해 세분화(segmentation)와 분류(classification) 작업 간의 지식 공유를 촉진하는 새로운 이중 스트림 파이프라인(Dual-Stream Pipeline)을 제안합니다.

- **Technical Details**: 제안된 DS-ViT(Dual-Stream Vision Transformer) 파이프라인은 FastSurfer(CNN 기반 모델)와 ADAPT(Vision Transformer 기반 모델) 간의 교차 작업 및 아키텍처 지식 공유를 허용합니다. 이중 스트림 임베딩 모듈을 통해 세분화 및 분류 모델의 특징 표현을 통합하고, 3D Bottleneck MLP 구조를 활용하여 이 정보를 이용해 분류 모델을 유도합니다. 또한, 잔여 시간 주의 블록(Residual Temporal Attention Block, RTAB)을 추가하여 시퀀스 MRI 이미지의 특성 맵 간의 차이를 분석, 조기 진단을 지원합니다.

- **Performance Highlights**: 여러 3D 데이터셋을 통해 검증한 결과, DS-ViT는 ADAPT 모델에 비해 평균 7%의 분류 정확도 향상과 훈련 시간 절반 단축을 달성했습니다. RTAB로 확장된 DS-ViT 파이프라인은 전체 70%의 예측 정확도를 보였으며, 고신뢰 샘플에서는 86%의 정확도를 기록하여 조기 유병 단계에서 개입 가능성을 제시하고 있습니다.



### Violence detection in videos using deep recurrent and convolutional neural networks (https://arxiv.org/abs/2409.07581)
Comments:
          11 pages, 7 figures, 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)

- **What's New**: 본 연구에서는 폭력 감지를 위한 새로운 딥러닝 아키텍처를 제안합니다. 이 아키텍처는 순환신경망(RNN)과 2차원 합성곱 신경망(2D CNN)을 결합하여 비디오에서 폭력 장면을 효과적으로 감지합니다.

- **Technical Details**: 제안된 방법은 RGB 비디오 프레임과 함께 캡처한 영상의 광학 흐름(optical flow)을 사용하여 공간적(spatial) 및 시간적(temporal) 특성을 추출합니다. EfficientNet B0을 활용하여 CNN을 구성하고, LSTM을 이용한 RNN으로 시간적 표현을 얻습니다. 이 방식은 두 개의 개별 네트워크(하나는 RGB 프레임용, 다른 하나는 광학 흐름용)로 구성되고, 이들의 특징을 결합하여 최종 클래스를 도출합니다.

- **Performance Highlights**: 제안된 접근법은 3개의 공개 데이터베이스에서 테스트되었으며, 최신 기술들과 유사한 성능을 달성하고 때로는 이를 초월하는 결과를 보였습니다.



### A Survey of Inverse Constrained Reinforcement Learning: Definitions, Progress and Challenges (https://arxiv.org/abs/2409.07569)
Comments:
          28 pages

- **What's New**: 본 논문은 Inverse Constrained Reinforcement Learning (ICRL)의 최신 발전에 대한 포괄적인 카테고리 조사를 제시합니다. ICRL은 전문가 에이전트의 시연 데이터에서 암묵적인 제약을 추론하는 작업으로, 최근 많은 주목을 받고 있는 연구 주제입니다.

- **Technical Details**: ICRL은 Constrained Reinforcement Learning (CRL)과 Inverse Constraint Inference (ICI)의 업데이트를 번갈아 가며 진행합니다. ICRL의 알고리즘 프레임워크는 Deterministic 또는 Stochastic 환경에서 제약 추론을 가능하게 하며, 다양한 시나리오에서 작동할 수 있습니다. 논문에서는 제약을 학습하는 데이터 기반 접근 방식을 통해 전문가 행동을 설명하는 방법을 제시합니다.

- **Performance Highlights**: ICRL은 자율주행, 로봇 제어 및 스포츠 분석 분야에서의 적용 가능성을 논의하며, 기준이 되는 문제와 도전 과제를 정리하였습니다. 이러한 연구는 이론적인 이해와 실용적인 산업 응용을 연결할 수 있는 길을 열어주는 데 기여할 것입니다.



### A Survey of Anomaly Detection in In-Vehicle Networks (https://arxiv.org/abs/2409.07505)
- **What's New**: 이 논문은 차량 내부 네트워크의 이상 감지에 대한 최신 연구를 포괄적으로 리뷰합니다. 특히, Controller Area Network(CAN) 버스에 중점을 두어, 기존 기법 및 최신 딥러닝 기법을 비교 분석합니다.

- **Technical Details**: 이 연구에서는 차량 내부 네트워크에서 이상 감지 방법 및 사용된 데이터셋을 평가합니다. 시간 시계열 기반 이상 감지 연구와 CAN 버스의 이상 감지 알고리즘을 검토하며, 통계적 방법과 딥러닝 기법이 포함됩니다. 이 논문은 2015년부터 2023년까지의 관련 자료를 대상으로 하였습니다.

- **Performance Highlights**: 글로벌적으로 증가하는 이상 감지 연구에 발맞추어, 이 연구는 통계 기반 및 딥러닝 기법, 그리고 CAN 버스에서의 적용성을 통해 차량 안전성을 향상시키기 위한 관점에서 중요한 기여를 합니다. 또한, 기존 연구의 한계를 지적하고 앞으로의 연구 방향을 제시합니다.



### AdaPPA: Adaptive Position Pre-Fill Jailbreak Attack Approach Targeting LLMs (https://arxiv.org/abs/2409.07503)
- **What's New**: 본 논문에서는 Large Language Models (LLMs)에서의 새로운 jailbreak 공격 방법인 Adaptive Position Pre-Filled Jailbreak Attack (AdaPPA)를 제안합니다. 기존의 공격 방법들이 주로 의미적 수준에 초점을 맞추어 쉽게 탐지되는 한계를 극복하고, 모델의 출력 위치에 따라 보호 능력의 차이를 활용합니다.

- **Technical Details**: AdaPPA는 다단계 과정을 통해 작동합니다. 첫 번째 단계는 안전하고 유해한 데이터로 사전 훈련된 프리필(content generation) 모델에 저랭크(training) 교육을 진행합니다. 두 번째 단계에서 질문을 다시 작성하고 프리필된 콘텐츠를 생성하여 최종 모델에 공격하며, 세 번째 단계에서는 공격 결과를 평가합니다. 이 방식은 Lowa2 모델을 이용하여 안전한 답변과 유해한 답변을 각각 생성하고 조합하여 공격을 최적화합니다.

- **Performance Highlights**: AdaPPA는 기존의 공격 방법에 비해 Llama2 모델에서 공격 성공률을 47% 향상시키는 성과를 보여주었습니다. 실험 결과는 LLM을 공격하는 새로운 접근방식이 효과적임을 증명합니다.



### Complex Emotion Recognition System using basic emotions via Facial Expression, EEG, and ECG Signals: a review (https://arxiv.org/abs/2409.07493)
Comments:
          29 pages, 11 figures

- **What's New**: 이 논문에서는 복합 감정 인식 시스템(Complex Emotion Recognition System, CERS)의 개발과 그로 인한 정서 역학에 대한 깊은 통찰을 제공하며, 머신러닝과 딥러닝을 활용한 정서 인식의 최신 동향을 조망합니다.

- **Technical Details**: CERS는 기본 감정의 조합, 상호 연결 및 동적 변화를 분석하여 복합 감정 상태를 해독합니다. 심전도(ECG) 및 뇌파(EEG)와 같은 생리적 신호를 통합함으로써 사용자 정서 상태에 대한 귀중한 통찰을 제공하고, 데이터셋의 품질을 향상시키며 시스템의 신뢰성을 강화합니다. 또한, 머신러닝, 딥러닝, 메타 학습(meta-learning) 방법의 효율성을 평가하기 위한 포괄적인 문헌 조사가 수행되었습니다.

- **Performance Highlights**: CERS의 성능을 개선하기 위한 메타 학습 접근 방법의 중요성이 강조되며, 복합 감정 인식에 대한 연구의 격차 및 도전과제를 밝히고 추가 연구를 장려하고 있습니다.



### RAGent: Retrieval-based Access Control Policy Generation (https://arxiv.org/abs/2409.07489)
Comments:
          Submitted to Usenix 2025

- **What's New**: 본 논문에서는 조직의 고수준 요구사항 명세서에서 접근 제어 정책을 자동으로 생성하기 위한 새로운 프레임워크인 RAGent를 제안합니다. 이는 기존 프레임워크의 한계를 극복하여 접근 제어 정책 생성의 신뢰성을 개선하는 데 중점을 둡니다.

- **Technical Details**: RAGent는 언어 모델(LMs)을 기반으로 한 검색 기반의 접근 제어 정책 생성 프레임워크로, 고수준 요구사항 명세서에서 접근 요구사항을 식별하고 이를 정책으로 변환합니다. 평균적으로 최첨단 F1 점수 87.9%를 기록하며, 정책의 구성 요소를 보다 복잡한 조건과 목적까지 포함하여 생성할 수 있습니다. 또한, 생성된 정책을 자동으로 검증하고 반복적으로 정제하는 새로운 검증-정제 메커니즘을 통해 신뢰성을 추가로 향상시킵니다.

- **Performance Highlights**: RAGent는 NLACPs를 식별하고 정책으로 변환하는 과정에서 평균 F1 점수 77.9%를 기록합니다. 기존 프레임워크와 비교할 때, RAGent는 39.1% 더 높은 F1 점수인 80.6%를 달성하며, 복잡한 접근 요구사항을 효과적으로 처리할 수 있습니다. 또한, 이 프레임워크는 세 가지 주석 데이터셋을 공개하여 접근 제어 정책 생성 분야의 데이터 부족 문제를 해결하려 합니다.



### MarS: a Financial Market Simulation Engine Powered by Generative Foundation Mod (https://arxiv.org/abs/2409.07486)
Comments:
          19 pages, 12 figures

- **What's New**: 본 논문에서는 금융 시장 시뮬레이션을 위한 새로운 Generative 모델인 Large Market Model (LMM)과 이를 활용한 금융 Market Simulation 엔진인 MarS를 제안합니다. 이는 다양한 행동의 시장 효과를 시뮬레이션하여 안전하게 전략을 학습할 수 있는 환경을 제공합니다.

- **Technical Details**: LMM은 주문(order) 레벨의 생성 모델로, 금융 시장에서의 데이터 구조화된 정보를 활용합니다. MarS는 높은 해상도, 제어 가능성, 상호작용을 중시하여 설계되었으며, 사용자가 시뮬레이션된 시장과 직접 상호작용할 수 있도록 합니다. 또한, 주문 클립을 통해 시장 조건을 시뮬레이션하는 조건부 생성 과정을 제공합니다.

- **Performance Highlights**: MarS는 예측 도구, 위험 탐지 시스템, 분석 플랫폼, 에이전트 훈련 환경 등 네 가지 주요 사용 사례를 통해 산업 내 활용 가능성을 증명합니다. 이를 통해 LMM의 스케일링 법칙 평가, MarS의 현실성 검토, 제어된 생성과 시장 영향 간의 균형을 평가하며, 실제 금융 시장의 요구를 충족시키는 혁신적인 접근 방식을 제시합니다.



### Optimization and Deployment of Deep Neural Networks for PPG-based Blood Pressure Estimation Targeting Low-power Wearables (https://arxiv.org/abs/2409.07485)
- **What's New**: 본 논문은 저전력 웨어러블 디바이스에서 혈압( BP) 추정을 위한 새로운 방법론을 제시합니다. 특히, Neural Architecture Search (NAS)와 Quantization을 활용한 완전 자동화된 DNN 설계 파이프라인을 통해, 정확하면서도 경량화된 모델을 GAP8 같은 초저전력 SoC에 배포할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 1D Convolutional Neural Networks(CNNs)를 기반으로 한 두 가지 모델, 즉 회귀 모델과 signal-to-signal( sig2sig) 모델을 시작으로 합니다. NAS를 활용하여 각 레이어의 연산을 자동으로 선택하고, 모델의 깊이를 조정하여 BP 예측 오류와 모델 크기 사이의 균형을 찾습니다. 이후 int8 정밀도로 양자화(Quantization)하여 모델의 크기와 에너지 소비를 더욱 줄입니다.

- **Performance Highlights**: 최적화된 모델은 최고의 기존 DNN과 비교하여 최대 4.99%의 오류 감소 또는 동일한 오류 수준에서 최대 73.36%의 매개변수 감소를 달성했습니다. GAP8에 배포된 우리의 DNN은 평균 절대 오류(MAE)가 7.86에 달하며, 소모 전력은 0.36-0.45 mJ로 매우 낮습니다.



### VSLLaVA: a pipeline of large multimodal foundation model for industrial vibration signal analysis (https://arxiv.org/abs/2409.07482)
- **What's New**: 본 논문은 VSLLaVA라는 새로운 파이프라인을 제시하여 산업 진동 신호 분석에 필요한 전문가 지식을 통합한다. 이는 이미지 인식 작업에 사용되는 대형 다중모달 모델을 활용하여 신호 매개변수 식별 및 결함 진단 능력을 향상시킨다.

- **Technical Details**: VSLLaVA에는 전문가 규칙 보조 신호 생성기가 포함되어 있어, 진동 분석 전문가가 제공한 신호와 도메인 특화 질문-답변 쌍을 결합하여 신호-질문-답변 삼중항(Signal-Question-Answer triplets)을 생성한다. 저자는 Contrastive Language-Image Pretraining (CLIP) 및 대형 언어 모델의 선형 계층을 정교하게 조정하기 위해 저순위 적응(low-rank adaptation) 방법을 사용한다. 이 조정된 모델은 정확성과 관련성을 평가하기 위해 전문가 규칙과 협력하여 평가된다.

- **Performance Highlights**: VSLLaVA 모델은 산업 신호 분석 및 모니터링을 위한 기초 모델 개발의 가능성을 보여준다. 8가지 유형의 신호에 대해 전문가와 LLM 간의 협력 평가를 통해, 신호 분석 및 결함 진단에서 지식이 크게 향상되었음을 입증하였다.



### EEG-Language Modeling for Pathology Detection (https://arxiv.org/abs/2409.07480)
- **What's New**: 이 연구는 전통적인 병리 탐지 분야의 발전에 기여하기 위해 생리적 뇌 데이터와 임상 텍스트를 결합하여 EEG-언어 모델을 학습하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: EEG-언어 모델(ELMs)은 임상 보고서와 15,000개의 EEG를 기반으로 훈련되었으며, 잘 정의된 구조적 정보(환자의 병력, EEG 설명 등)를 텍스트 보고서 내에서 활용하여 멀티모달 정렬(multi-modal alignment) 방법을 확장했습니다. 연속적인 학습(contrastive learning) 기법을 사용하여 모델의 정밀성을 높였습니다.

- **Performance Highlights**: EEG-언어 모델이 EEG 전용 모델에 비해 병리 탐지 성능에서 뛰어난 결과를 보였으며, 특히 주석이 부족한 환경에서도 높은 분류 정확도를 유지했습니다. 이 연구는 뇌 활동 데이터와 임상 텍스트의 통합이 의료 응용 프로그램에 있어 중요한 발전을 나타냄을 강조합니다.



### Responsible AI for Test Equity and Quality: The Duolingo English Test as a Case Study (https://arxiv.org/abs/2409.07476)
- **What's New**: 이번 장(chapter)에서는 인공지능(AI)의 평가(evaluation) 활용에 따른 기회와 위험에 대해 다루며, 책임 있는 AI(Responsible AI, RAI) 관행의 중요성을 강조합니다.

- **Technical Details**: 특히, 이 장에서는 Duolingo English Test (DET)라는 AI 기반의 고위험 영어 언어 평가 사례를 통해 RAI 기준과의 관계를 탐구하며, RAI의 도메인 불가지론(domain-agnostic) 원칙들과의 연관성을 설명합니다.

- **Performance Highlights**: RAI 관행의 구체적인 예시를 통해 검사의 공정성(fairness) 및 품질(quality)을 보장하기 위한 윤리적 원칙인 유효성(validity), 신뢰성(reliability), 프라이버시(priacy) 및 보안(security), 투명성(transparency) 및 책임(accountability) 기준을 어떻게 의미 있게 충족하는지를 보여줍니다.



### Ethical AI Governance: Methods for Evaluating Trustworthy AI (https://arxiv.org/abs/2409.07473)
Comments:
          6 pages, 1 figure, accepted for presentation at AIEB 2024: Workshop on Implementing AI Ethics Through a Behavioural Lens - ECAI, Octoebr 2024

- **What's New**: 본 논문은 신뢰할 수 있는 인공지능(Trustworthy Artificial Intelligence, TAI) 평가 방법과 시스템을 정리하고 하위 분류를 제안하여, AI 시스템의 윤리적 기준 및 안전성을 확보하는 데 기여하고자 합니다.

- **Technical Details**: TAI의 검토 방법으로는 개념적 평가 방법, 수동 평가 방법, 자동화 평가 방법, 반자동화 평가 방법과 같은 네 가지 범주를 제안하였습니다. 각 방법은 공정성, 투명성, 리스크 및 책임, 신뢰 및 안전이라는 하위 주제로 분류되었습니다.

- **Performance Highlights**: 본 연구는 기존의 조사 방법과는 달리 TAI 평가의 결과를 점수화하는 접근 방식을 강조하며, 논의된 여러 프레임워크와 방법론이 실제 AI 시스템의 정책 설정 및 투명성 향상에 중요한 기여를 할 수 있음을 시사합니다.



### AI, Climate, and Transparency: Operationalizing and Improving the AI Ac (https://arxiv.org/abs/2409.07471)
Comments:
          5 pages, 1 table, preprint

- **What's New**: 이 논문은 AI Act의 기후 관련 투명성 조항을 비판적으로 검토하며 이행의 주요 간극과 도전 과제를 강조합니다. 특히 AI 추론 시 에너지 소비 제외, AI 응용 프로그램의 간접 온실가스 배출 미포함, 표준화된 보고 방법의 결여 등의 문제를 지적합니다.

- **Technical Details**: AI Act는 고위험 AI 시스템의 경우 개발, 훈련, 테스트 및 검증에 사용된 컴퓨팅 리소스를 문서화해야 하지만, 에너지 소비를 명시적으로 요구하지 않아 투명성이 제한됩니다. 일반 목적 AI(GPAI) 모델에 대해 제공자는 모델 개발 단계에서의 에너지 소비를 포함해야 하나, 이는 추론 단계의 에너지 소비를 간과하고 있습니다. 우선, 서버 수준의 누적 에너지 보고 방식을 권장하며, AI의 환경 영향을 해결하기 위한 지속 가능성 리스크 평가 및 재생 가능 에너지 목표와 같은 포괄적인 정책 변화도 제안합니다.

- **Performance Highlights**: AI Act는 AI 관련 기후 보고의 첫 단계이지만, 많은 허점과 모호한 표현으로 인해 효과적인 이행이 어렵습니다. 이는 AI의 에너지 사용 평가와 관련된 탄소 배출 및 재생 가능 에너지 인프라에 미치는 영향을 심각하게 저해하고 있습니다. 향후 AI 및 위임 행위에서 추론을 보고 범주에 포함시키는 것이 중요하며, 간접 배출이나 물 소비의 포함을 요구하는 사항은 아직 반영되지 않았습니다.



### Small Object Detection for Indoor Assistance to the Blind using YOLO NAS Small and Super Gradients (https://arxiv.org/abs/2409.07469)
- **What's New**: 이번 연구는 시각 장애인을 위한 실내 보조 기술을 위한 새로운 접근법을 제시하고 있으며, 특히 작은 물체 탐지 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 방법은 YOLO NAS Small 아키텍처를 기반으로 하며, Super Gradients 트레이닝 프레임워크를 이용하여 최적화된 경량 물체 탐지 모델입니다. 이 조합은 소형 물체를 실시간으로 탐지할 수 있도록 해주며, 특히 가구, 가전 제품 및 가정용 물품 등에서 중요합니다.

- **Performance Highlights**: 이 방법은 낮은 대기 시간(low latency)과 높은 정확도(high accuracy)를 강조하여, 사용자에게 공간 인식(spatial awareness)과 주변 상호작용(interaction)에 대한 정보를 제공할 수 있는 음성 기반 가이드를 가능하게 합니다.



### An Artificial Neural Network for Image Classification Inspired by Aversive Olfactory Learning Circuits in Caenorhabditis Elegans (https://arxiv.org/abs/2409.07466)
- **What's New**: 이 연구는 유사한 신경 회로를 모방한 인공 신경망(ANN)을 활용하여 이미지 분류 작업에서 우수한 성능을 보이는 새로운 접근법을 제안합니다. 모델은 Caenorhabditis elegans의 가소성 후각 학습 메커니즘에서 영감을 받아 설계되었습니다.

- **Technical Details**: 제안된 ANN은 C. elegans의 모양과 기능을 본따 만들어졌으며, 121개의 기능적 신경 뉴런으로 구성되어 있습니다. 연구에서는 고속 유전자 시퀀싱과 행동 실험을 통해 후각 자극을 피하는 학습 메커니즘을 규명하였습니다.

- **Performance Highlights**: 비교 분석 결과, 제안된 ANN은 기존의 두 가지 아키텍처를 기반으로 한 ANN보다 이미지 분류 작업에서 더 높은 정확도, 일관성 및 빠른 수렴 속도를 보였습니다.



### Reflective Human-Machine Co-adaptation for Enhanced Text-to-Image Generation Dialogue System (https://arxiv.org/abs/2409.07464)
- **What's New**: 본 연구에서는 이미지 생성 시스템의 사용자 친화성을 개선하기 위한 'RHM-CAS'라는 반영적 인간-기계 협응 전략을 제안합니다. 이 전략은 사용자의 의도를 이해하고 이미지 결과물을 최적화하기 위해 여러 차례의 상호작용을 활용합니다.

- **Technical Details**: 제안된 방법은 모듈형 아키텍처로 구성되어 있으며, 기억(Memory), 요약기(Summarizer), 생성 모델(Generation Model), 반영 블록(Reflection Block), 평가자(Evaluator) 및 모호성 추론(Min-Inf) 구성 요소를 포함합니다. 이러한 구성 요소들은 사용자의 과거 대화를 저장하고, 생성할 이미지를 위해 적절한 프롬프트를 생성하며, 생성된 이미지를 평가하고 사용자의 의도를 파악하는 데 기여합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법의 효과성이 입증되었습니다. 일반 이미지 및 패션 이미지 생성 작업에서 우수한 성능을 보여, 사용자 요구를 충족시키는 이미지 결과물을 생성하는 데 유용합니다.



### LSST: Learned Single-Shot Trajectory and Reconstruction Network for MR Imaging (https://arxiv.org/abs/2409.07457)
- **What's New**: 본 연구는 단일 샷 자기 공명 (Magnetic Resonance, MR) 이미징에서 k-공간 (k-space) 데이터 수집의 속도를 높이고 T2-블러 (T2-blur)를 줄이는 새로운 방법을 제안합니다. 이를 위해 샘플 수를 줄이고 k-공간 투사 최적화에 초점을 맞추며, 전체 이미지의 재구성 품질을 향상시키기 위해 물리학적 제약을 준수합니다.

- **Technical Details**: 이 연구에서 제안된 방법은 (a) k-공간 측정을 위한 궤적을 최적화하고, (b) 샘플 수를 줄여 수집 속도를 증가시키며, (c) T2-블러의 영향을 줄입니다. 또한, T2-블러를 고려하여 재구성 품질을 향상시키기 위해 Density Compensation Factor (DCF)를 훈련 중 통합하고, 초기 입력에 모델 기반 솔루션을 사용합니다. k-공간에서 루틴 궤적을 생성하기 위해 Traveling Salesman Problem (TSP) 접근법을 사용하며, 이를 통해 적합한 궤적을 최적화하는 과정이 포함됩니다.

- **Performance Highlights**: 실험 결과는 경험 많은 방사선과의 평가를 바탕으로, 제안된 방법이 비교 방법보다 ACL 섬유(Anterior Cruciate Ligament fibers)에 대해 더 선명한 재구성을 달성했음을 보여줍니다. 특히 8배 및 16배 가속 요소를 사용한 fastMRI 멀티채널 데이터셋에서 검증되었습니다.




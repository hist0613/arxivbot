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




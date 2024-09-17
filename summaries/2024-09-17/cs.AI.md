New uploads on arXiv(cs.CL)

### DILA: Dictionary Label Attention for Mechanistic Interpretability in High-dimensional Multi-label Medical Coding Prediction (https://arxiv.org/abs/2409.10504)
- **What's New**: 본 논문에서는 DIctionary Label Attention (DILA)이라는 새로운 메커니즘 해석 모듈을 제안하여 의료 코딩과 같은 고차원 멀티레이블 예측을 위해 기계적 해석 가능성을 향상시켰습니다.

- **Technical Details**: DILA 모듈은 밀집 임베딩(dense embeddings)을 희소 임베딩 공간(sparse embedding space)으로 분리하여 각 비영(非零) 요소가 전 세계적으로 학습된 의료 개념을 나타내도록 합니다. 이는 대규모 언어 모델(LLMs)을 활용하여 학습된 사전 기능을 자동으로 식별하고 요약합니다.

- **Performance Highlights**: DILA는 대규모 MIMIC-III 데이터셋에서 기존의 최첨단 블랙박스 기준과 경쟁적인 성능을 유지하며, 희소성과 해석 가능성을 통합하여 의료 코딩과 같은 극단적인 멀티레이블 예측 작업에서 효율적임을 입증했습니다.



### Schrodinger's Memory: Large Language Models (https://arxiv.org/abs/2409.10482)
- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 메모리 메커니즘을 UAT(보편적 근사 정리) 이론을 통해 설명하고, 다양한 모델 간의 메모리 능력을 비교하여 LLM의 성능을 평가할 새로운 접근 방식을 제안합니다.

- **Technical Details**: UAT 이론은 딥러닝의 기초 이론으로, Transformer 기반 LLM의 메모리 기능을 설명하기 위해 수학적 형태를 제공합니다. UAT는 주어진 입력에 따라 함수를 동적으로 조정할 수 있는 Transformer 모델의 능력을 강조합니다. 다층 변환기 모델의 수학적 구조가 UAT와 일반적으로 일치함을 보여줍니다.

- **Performance Highlights**: 실험을 통해 LLM이 갖춘 메모리 능력을 검증하고, 인간 두뇌와 LLM 간의 기억 용량을 포괄적으로 분석하였습니다. 연구는 LLM과 인간 인지 사이의 유사점과 차이점을 강조하며, LLM 기능의 기초가 되는 메모리 구조에 대한 이해를 심화합니다.



### A Knowledge-Enhanced Disease Diagnosis Method Based on Prompt Learning and BERT Integration (https://arxiv.org/abs/2409.10403)
Comments:
          Knowledge Enhancement,Disease Diagnosis,Prompt Learning,BERT,Knowledge Graph

- **What's New**: 이 논문은 prompt learning 프레임워크를 기반으로 한 지식 향상 질병 진단 방법을 제안합니다. 외부 지식 그래프에서 구조화된 지식을 검색하고 이를 인코딩하여 PROPMPT 템플릿에 주입함으로써 언어 모델의 이해력 및 추론 능력을 향상시킵니다.

- **Technical Details**: 이 방법은 CHIP-CTC, IMCS-V2-NER, KUAKE-QTR의 세 가지 공개 데이터셋에서 실험을 수행했으며, 지식 주입 모듈이 F1 점수에 미치는 중요한 역할을 강조합니다. 지식 주입은 언어 모델이 의료 관련 지식을 습득하는 데 효과적이며, 이는 진단 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 제안된 방법은 CHIP-CTC 데이터셋에서 2.4%, IMCS-V2-NER 데이터셋에서 3.1%, KUAKE-QTR 데이터셋에서 4.2%의 F1 점수 향상을 포함하여 기존 모델들보다 상당한 성능 향상을 보여주었습니다. 이 방법은 질병 진단의 정확성을 개선할 뿐만 아니라 예측의 해석 가능성을 높여 임상 진단에 더 신뢰할 수 있는 지원을 제공합니다.



### Detecting Sexism in German Online Newspaper Comments with Open-Source Text Embeddings (Team GDA, GermEval2024 Shared Task 1: GerMS-Detect, Subtasks 1 and 2, Closed Track) (https://arxiv.org/abs/2409.10341)
Comments:
          6 pages, 4 figures, 2 tables

- **What's New**: 이번 연구는 독일어 온라인 댓글에서 성차별과 여성비하를 효과적으로 탐지하기 위해 단일 언어 및 다국어 오픈 소스 텍스트 임베딩을 활용했습니다. 특히, 오스트리아 신문의 댓글 데이터를 기반으로 전통적인 분류기와 텍스트 임베딩 모델을 사용하여 성차별을 검출하는 방법론과 성능을 보고합니다.

- **Technical Details**: 연구에서는 'mE5-large'와 'German BERT large paraphrase cosine' 모델을 사용하여 독일어 댓글에서 성차별을 탐지했습니다. 이를 통해 훈련된 분류기는 인간 주석자의 판단을 잘 모방하며, GermEval 2024 GerMS-Detect Subtask 1에서 평균 매크로 F1 점수 0.597(4위), Subtask 2에서 평균 Jensen-Shannon 거리 0.301(2위)을 달성했습니다.

- **Performance Highlights**: 제안한 방법론은 다양한 언어와 언어적 맥락에서 확장 가능성을 보여주며, 주석자 간의 변동성을 반영한 주석 데이터를 바탕으로 높은 성능을 나타냈습니다. 이러한 접근 방식은 성차별을 포함한 더 포괄적인 온라인 공간을 만들기 위한 기반이 될 수 있습니다.



### The 20 questions game to distinguish large language models (https://arxiv.org/abs/2409.10338)
- **What's New**: 이 연구는 블랙박스(black-box) 환경에서 두 개의 대형 언어 모델(LLMs)이 동일한 모델인지 여부를 판단하는 새로운 방법을 제시합니다. 주어진 목표는 일반적으로 20개의 질문 이하로 구성된 소규모(benign) 이진 질문 세트를 사용하는 것입니다.

- **Technical Details**: 문제를 공식화하고, 알려진 벤치마크 데이터셋에서 임의의 질문 선택을 사용하여 20개 질문 내에서 거의 100% 정확도로 기준선을 설정합니다. 이후 이 문제에 대한 최적 경계를 보여준 후, 동일한 작업을 수행하기 위해 질문 수를 절반으로 줄일 수 있는 두 가지 효과적인 질문 휴리스틱(heuristics)을 소개합니다.

- **Performance Highlights**: 이 방법은 모델 샘플 유출에 대한 의구심을 가진 감사인(auditors)이나 저작권 소유자에게 중요한 장점을 제공합니다.



### MGSA: Multi-granularity Graph Structure Attention for Knowledge Graph-to-Text Generation (https://arxiv.org/abs/2409.10294)
- **What's New**: 이번 논문은 Multi-granularity Graph Structure Attention (MGSA) 모델을 소개하여 Knowledge Graph에서 Text로의 변환 작업에서 발생하는 한계를 극복하고자 하였습니다. MGSA 모델은 엔티티 수준과 단어 수준의 구조 정보를 동시에 통합하여 보다 질 높은 자연어 텍스트 생성을 목표로 합니다.

- **Technical Details**: MGSA 모델은 세 가지 모듈로 구성됩니다: (1) 엔티티 수준 구조 인코딩 모듈, (2) 단어 수준 구조 인코딩 모듈, (3) 집계 모듈. 이 모듈들은 서로 다른 수준의 구조 정보를 결합하여 KG의 구조를 보다 포괄적으로 이해하는 데 도움을 줍니다.

- **Performance Highlights**: WebNLG와 EventNarrative 데이터셋을 사용하여 MGSA 모델을 평가한 결과, 단일 수준의 구조 정보에만 의존하는 모델들에 비해 일관되게 뛰어난 성능을 보였습니다. 이로 인해 우리의 접근법의 효과성을 입증하였습니다.



### From Text to Emoji: How PEFT-Driven Personality Manipulation Unleashes the Emoji Potential in LLMs (https://arxiv.org/abs/2409.10245)
Comments:
          Submitted to NeurIPS 2024 Workshop on Behavioral Machine Learning

- **What's New**: 본 논문은 LLMs(대형 언어 모델)의 개인 특성 조작에 대한 새로운 접근 방식을 제시합니다. 특히 Big Five(빅 파이브) 성격 특성을 조작하기 위한 Opinion QA 데이터셋을 도입하고, 기존 방법론보다 일관성 있고 지속적인 결과를 보여주는 PEFT(파라미터 효율적인 미세 조정)를 사용하여 성격을 효과적으로 변형할 수 있음을 증명했습니다.

- **Technical Details**: PEFT의 일환으로 Quantized Low-Rank Adaptation(QLORA) 방법론을 활용하여 Mistral-7B-Instruct 및 Llama-2-7B-chat 모델을 조정했습니다. 이 방법은 모델이 감정 표현을 위해 이모지를 생성하도록 했으며, 이는 독창적인 소통 방식으로 나타났습니다. 또한, 기존의 IKE(In-Context Knowledge Editing) 방법과 비교하여 더 높은 일관성을 보였습니다.

- **Performance Highlights**: Llama-2-7B-chat은 99.5%의 확률로 외향성과 관련된 테스트 사례에서 이모지를 생성하였고, Mistral-8B-Instruct는 92.5%의 확률로 개방성과 관련된 테스트에서 이모지를 생성했습니다. 이 연구를 통해 LLM의 성격 조작 및 인지 가능성을 개선함으로써 사용자와의 상호작용을 더 풍부하게 만들 수 있는 기반을 마련하였습니다.



### LLMs for clinical risk prediction (https://arxiv.org/abs/2409.10191)
- **What's New**: 이 연구는 GPT-4와 clinalytix Medical AI의 섬망 위험 예측 효과를 비교하였습니다. GPT-4는 양성 사례를 식별하는 데 상당한 한계를 보였으며, 신뢰할 수 있는 확률 추정치를 제공하는 데 어려움을 겪었습니다. 반면, clinalytix Medical AI는 우수한 정확도를 보여주었습니다.

- **Technical Details**: 연구는 190건의 사례 데이터를 분석했으며, 비구조적(구조가 없는) 원시 텍스트 및 실험실 결과, 약물 기록, 활력 징후와 같은 구조적 데이터를 포함했습니다. LLM은 각 사례에 대해 섬망의 위험을 평가하고 확률 점수를 제공하도록 요청받았습니다.

- **Performance Highlights**: clinalytix Medical AI는 섬망 위험 예측에서 현저히 높은 정확성을 보여주었으나, LLM은 약 38%의 실제 양성 사례를 놓치며 낮은 재현율을 보였습니다. 또한 LLM은 충분한 정보가 없다고 응답하는 경우가 많았으며, 모델의 예측을 설명하는 데 있어서 조심해야 한다고 강조되었습니다.



### Augmenting Automatic Speech Recognition Models with Disfluency Detection (https://arxiv.org/abs/2409.10177)
Comments:
          Accepted by SLT2024

- **What's New**: 이 연구에서는 기존 오토메이션 음성 인식(ASR) 모델의 한계를 극복하기 위해 개방형(오픈셋) 발화 비유창성(dysfluency) 탐지 기능을 추가하는 인퍼런스(inference) 전용 접근법을 제시합니다.

- **Technical Details**: 발화 비유창성은 반복, 중재, 수정 등으로 나타나는 대화의 흐름 방해를 의미합니다. 제안하는 파이프라인은 ASR 모델과 특성 추출기(framework feature extractor)를 조합하여 전사(transcription) 및 정렬(alignment)을 수행하며, 수정된 연결주의 시간 분류(Connectionist Temporal Classification, CTC) 기반 강제 정렬 알고리즘을 사용하여 단어 수준의 타임스탬프를 예측합니다. 이 방법은 발화 비유창성의 위치와 지속 시간을 효과적으로 캡처합니다.

- **Performance Highlights**: 제안된 모델은 81.62%의 정확도와 80.07%의 F1 점수를 달성하였으며, 비유창한 데이터셋에서 74.13%의 단어를 초기 전사에서 놓친 것을 보완했습니다. 이 결과는 하위 작업(downstream tasks)에서 활용될 가능성을 보여줍니다.



### jina-embeddings-v3: Multilingual Embeddings With Task LoRA (https://arxiv.org/abs/2409.10173)
Comments:
          20 pages, pp11-13 references, pp14-20 appendix and experiment tables

- **What's New**: jina-embeddings-v3는 5억 7천만 개의 파라미터를 가진 최신 텍스트 임베딩 모델로, 다중 언어 데이터 및 긴 맥락 검색 작업에서 최첨단 성능을 발휘합니다. 이 모델은 최대 8192개의 토큰까지 지원하며, 쿼리-문서 검색, 클러스터링, 분류, 텍스트 매칭 등의 작업에 특화된 Low-Rank Adaptation (LoRA) 어댑터를 포함하고 있습니다.

- **Technical Details**: jina-embeddings-v3는 긴 텍스트 시퀀스를 효과적으로 인코딩하고 특정 작업에 맞춘 임베딩 생성을 가능하게 하는 구조로, XLM-RoBERTa 모델을 기반으로 합니다. FlashAttention 2와 DeepSpeed 프레임워크를 활용하여 효율적인 분산 훈련을 지원하고, Rotary Position Embeddings (RoPE)를 사용하여 절대 위치 임베딩을 대체해 긴 시퀀스를 처리합니다.

- **Performance Highlights**: MTEB 벤치마크에서 jina-embeddings-v3는 OpenAI 및 Cohere의 최신 상용 임베딩을 초월하여 영어 작업에서 뛰어난 성과를 보였고, 모든 다국어 작업에서 multilingual-e5-large-instruct을 능가했습니다. 또한 7.1억 개의 파라미터를 가진 LLM 기반 임베딩보다 비용 효율성이 높아 실제 생산 및 엣지 컴퓨팅에 더 적합합니다.



### LLMs4OL 2024 Overview: The 1st Large Language Models for Ontology Learning Challeng (https://arxiv.org/abs/2409.10146)
Comments:
          15 pages, 1 figure, Will appear in "The 1st LLMs4OL Challenge @ ISWC 2024" proceedings

- **What's New**: LLMs4OL 2024은 대규모 언어 모델(LLMs)을 온톨로지 학습(OL)에서 활용하기 위한 첫 번째 공동 기획으로, 23회 국제 시맨틱 웹 회의(ISWC 2024)와 함께 진행됩니다. 이 챌린지는 LLMs의 온톨로지 학습 과정에서의 혁신과 기여를 도모하는 것을 목적으로 합니다.

- **Technical Details**: LLMs4OL 챌린지는 세 가지 주요 작업(Task A, B, C)으로 구성됩니다. 각 작업은 텍스트에서 구조적 지식을 추출하는 것을 목표로 하며, LLM 기반 솔루션을 요구합니다. 테스트는 Few-shot 및 Zero-shot 단계로 나뉘며, 각 참가자는 특정 작업(Task A: 용어 분류, Task B: 계층 구조 발견, Task C: 비계층적 관계 추출)에 참여할 수 있습니다.

- **Performance Highlights**: 이 챌린지를 통해 LLM의 온톨로지 학습에서의 가능성을 탐구하며, 평가 단계에서 LLM의 일반화 능력과 전이 가능성을 확인할 수 있습니다. 챌린지에서 사용된 데이터세트는 GitHub에서 확인 가능하며, 표준 평가 지표를 기반으로 모든 작업에 적용됩니다.



### StruEdit: Structured Outputs Enable the Fast and Accurate Knowledge Editing for Large Language Models (https://arxiv.org/abs/2409.10132)
- **What's New**: 이 논문에서는 구조적 편집(Structural Editing, StruEdit)이라는 새로운 지식 편집 방법을 제안합니다. StruEdit는 자연어 출력의 비구조적인 특성을 해결하기 위해 LLM(대형 언어 모델)에서 구조화된 출력 구조를 생성하고, 잠재적으로 구식인 정보를 제거한 후 최신 정보를 효율적으로 다시 채우는 프로세스를 포함합니다.

- **Technical Details**: StruEdit는 LLM이 추론(triplet reasoning) 구조를 통해 정보를 처리하도록 유도하고, 파라미터 수정 및 문맥 입력 없이 구조화된 사실 체인을 업데이트합니다. 이 방법은 연결된 추론 단계 간의 결합 이슈를 제거하며, 멀티-홉(multi-hop) 편집 작업에서 높은 정확도와 낮은 지연(latency)을 달성합니다.

- **Performance Highlights**: 실험 결과, StruEdit는 기존의 모든 지식 편집 방법들과 비교했을 때 가장 높은 정확도와 최단 응답 시간을 기록하였고, 편집 작업의 수가 증가하더라도 안정적인 성능을 유지합니다. 특히 기존 방법들과 달리, StruEdit는 특정 지식의 위치를 찾지 않아도 되는 장점이 있어, 결과적으로 사실의 환각(hallucination)을 줄입니다.



### Self-Supervised Syllable Discovery Based on Speaker-Disentangled HuBER (https://arxiv.org/abs/2409.10103)
Comments:
          Accepted by IEEE SLT 2024

- **What's New**: 이 논문에서는 Self-Distillation HuBERT (SD-HuBERT) 방법을 개선하기 위해 음성만을 이용한 자기 지도 학습(self-supervised learning) 기법을 제안합니다. 이 방법은 음성 단위(syllabic units)와 화자 정보(speaker information)를 분리하여 성능을 개선합니다.

- **Technical Details**: 제안된 방법은 speaker perturbation을 통한 데이터 증강(data augmentation)을 적용하고, CLS token 대신 프레임 레벨(frame-level)의 학습 목표를 사용하여 파라링귀스틱(paralinguistic) 정보를 집계하는 것을 방지합니다. 이 방법은 고차원의 Transformer 층을 사용하여 언어적 코스(grained) 단위와 연관된 학습 목표를 설정합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 Librispeech에서 음절 세분화(syllable segmentation) 및 음절 단위 품질(syllabic unit quality)에 대한 평가 지표에서 현존하는 최첨단 방법을 초과하는 성능을 보임을 보여줍니다.



### LLM-DER:A Named Entity Recognition Method Based on Large Language Models for Chinese Coal Chemical Domain (https://arxiv.org/abs/2409.10077)
- **What's New**: 이 논문에서는 중국 석탄 화학 산업 도메인에서의 복잡한 구조를 가진 개체 인식 문제를 해결하기 위해 LLM-DER이라는 대형 언어 모델(LLMs)을 기반으로 한 개체 인식 프레임워크를 제안합니다.

- **Technical Details**: LLM-DER는 LLMs를 사용하여 개체 유형을 포함하는 관계 목록을 생성하여 개체 정보를 풍부하게 하고, 잘못 인식된 개체를 제거하기 위한 신뢰성 및 일관성 평가 방법을 설계합니다. 이는 특정 도메인에서 복잡한 구조의 개체 인식 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 실험 결과 LLM-DER는 Resume 데이터셋과 자체 구축한 석탄 화학 데이터셋에서 뛰어난 성능을 보여주며, 기존의 GPT-3.5-turbo 모델은 물론 완전 감독 기반 모델도 초월하는 성과를 올렸습니다.



### Increasing faithfulness in human-human dialog summarization with Spoken Language Understanding tasks (https://arxiv.org/abs/2409.10070)
- **What's New**: 본 연구는 대화 요약의 신뢰성을 높이기 위해 작업 관련 정보를 통합하는 새로운 방법을 제안합니다. 특히, 고객 서비스 대화에서의 통화 의도 및 도메인 관련 엔터티를 도입하여 요약의 의미적 충실도를 높이고자 합니다.

- **Technical Details**: 연구는 SLU(Spoken Language Understanding)에서 제안된 의미적 정보를 기반으로 통화 의도, 도메인 엔터티와 같은 작업 관련 요소들을 요약 과정에 통합하여 신뢰성을 높이는 방법을 다룹니다. DECODA 코퍼스를 사용하여 지표인 NEHR(주요 엔터티 부재 비율) 및 KL 발산(Kullback-Leibler divergence)을 통한 요약 선택 기준을 제안했습니다.

- **Performance Highlights**: 실험 결과, 작업 관련 정보를 통합한 요약 모델이 정확도를 향상시키는 것으로 나타났으며, 다양한 단어 오류율에도 불구하고 일관성을 유지하는 동시에 신뢰할 수 있는 요약을 생성할 수 있음을 보여주었습니다.



### MindGuard: Towards Accessible and Sitgma-free Mental Health First Aid via Edge LLM (https://arxiv.org/abs/2409.10064)
- **What's New**: 이 논문은 정신 건강 문제에 대한 접근성을 높이고 낙인을 없애며 전문적인 모바일 정신 건강 관리 시스템인 MindGuard를 제시합니다. MindGuard는 최초로 사용자 행동 데이터와 LLM 기반 대화 데이터를 통합하여 종합적인 정신 건강 관리 시스템을 구축합니다.

- **Technical Details**: MindGuard의 핵심은 사용자 행동 데이터와 LLM(대형 언어 모델) 기술을 결합하여 개인화된 대화와 개입을 제공하는 것입니다. 이 시스템은 자동 회귀적 특성을 가진 LLM의 환각(hallucination) 문제를 해결하기 위해 고품질 정신 건강 데이터셋을 구축하고 지속적인 관련 기술(PT), 감독형 미세 조정(SFT), 및 인간 검토를 포함한 후처리 방법을 사용합니다.

- **Performance Highlights**: MindGuard는 2주간의 실제 배포 결과를 통해 사용자 행동 데이터를 정확하게 분석하고 잠재적인 정신 건강 문제를 신속하게 식별할 수 있음이 입증되었습니다. 실험 결과는 MindGuard가 주요 정신 건강 관리 제공에 있어서 신뢰성과 정확성을 크게 향상시켰음을 보여줍니다.



### Householder Pseudo-Rotation: A Novel Approach to Activation Editing in LLMs with Direction-Magnitude Perspectiv (https://arxiv.org/abs/2409.10053)
- **What's New**: 이 논문에서는 큰 언어 모델의 내부 표현을 직접 수정하여 원하는 행동을 유도하는 새로운 편집 방법인 Householder Pseudo-Rotation (HPR)을 제안합니다. 기존의 방법들은 주로 활성화를 공간의 점으로 취급하고, 이를 조정하는 벡터를 추가하는 방식이었습니다. 반면, HPR은 활성화의 방향과 크기를 다루어 활성화의 정규화를 유지하면서도 성능을 향상시킬 수 있는 접근법입니다.

- **Technical Details**: HPR은 활성화 벡터를 원점 주위에서 회전시켜 모델의 행동을 조정하려고 합니다. 이 방법은 특이한 고차원 회전 행렬을 사용하는 대신 개념적으로 더 간단한 하이퍼플레인을 통해 활성화 벡터를 반사하여 원하는 방향으로 이동시킵니다. 이는 계산의 복잡성을 줄이고, 각 활성화마다 별도의 회전 행렬을 저장할 필요가 없어 메모리 효율성을 높입니다.

- **Performance Highlights**: TruthfulQA 데이터셋을 사용한 실험 결과, HPR 방법은 Steering Vector 편집 방법보다 성능이 현저히 개선되었습니다. 또한, HPR은 언어 모델의 편향, 윤리, 독성 문제와 같은 다른 행동 관련 문제에서도 성능을 향상시킬 수 있음을 보여주었습니다.



### On the Diagram of Though (https://arxiv.org/abs/2409.10038)
- **What's New**: 새롭게 소개된 DoT(도형적 사고) 프레임워크는 대형 언어 모델(LLM) 내에서 반복적인 추론을 비선형적으로 모델링하며, 기존의 직선적 사고 방식과 차별화됩니다. 이 프레임워크는 DAG(Directed Acyclic Graph)을 사용하여 여러 요소의 상호작용을 처리합니다.

- **Technical Details**: DoT 프레임워크는 세 가지 역할(제안자 <proposer>, 비평가 <critic>, 요약자 <summarizer>)을 관리하며, 자동 회귀적 다음 토큰 예측을 활용하여 각 역할 간의 원활한 전환을 가능하게 합니다. 이 과정은 제안, 비평, 수정, 검증의 순환적 구조로 이루어집니다.

- **Performance Highlights**: DoT는 추론 프로세스를 단일 모델 내에서 통합하여 복잡한 문제 해결을 촉진하고, 훈련 및 추론의 효율성을 향상시킵니다. 이 접근법은 여러 모델이나 외부 제어 메커니즘의 필요성을 제거하고, LLM의 훈련 효율성과 robust한 추론 능력을 강화합니다.



### AceParse: A Comprehensive Dataset with Diverse Structured Texts for Academic Literature Parsing (https://arxiv.org/abs/2409.10016)
Comments:
          5 pages, 3 figures, 3 tables

- **What's New**: 데이터 중심의 AI 발전에 따른 새로운 방향으로, 학술 문헌의 구조화된 텍스트 파싱을 지원하는 AceParse 데이터셋을 소개합니다. 또한, AceParser라는 멀티모달 모델을 세밀하게 조정하여 다양한 구조화된 텍스트에 대한 파싱 정확도를 향상시켰습니다.

- **Technical Details**: AceParse는 수학적 표현이 포함된 문장, 테이블, 알고리즘 등의 다양한 구조화된 텍스트를 포함하는 최초의 포괄적인 오픈소스 데이터셋입니다. 이 데이터셋은 LaTeX 마크업 언어를 사용하여 텍스트 구조를 주석 처리합니다. AceParser는 Florence2 아키텍처를 기반으로 하여 문서 이미지를 패치로 나누고, 시각적 토큰과 텍스트 토큰 임베딩을 결합한 후 멀티모달 토큰 임베딩을 적용하여 구조화된 텍스트를 파싱합니다.

- **Performance Highlights**: AceParser는 F1 점수 4.1% 및 Jaccard 유사도 5% 향상을 달성하여 기존의 최첨단 모델을 능가하며, 멀티모달 모델이 학술 문헌 파싱에서 가지는 잠재력을 보여줍니다.



### HALO: Hallucination Analysis and Learning Optimization to Empower LLMs with Retrieval-Augmented Context for Guided Clinical Decision Making (https://arxiv.org/abs/2409.10011)
Comments:
          10 pages, 4 figures

- **What's New**: 최근 발표된 HALO 프레임워크는 의료 질문-답변(QA) 시스템의 정확성과 신뢰성을 향상시키기 위해 고안되었습니다. 이 프레임워크는 hallucinations(환각)을 감지하고 완화하는 데 초점을 맞춥니다.

- **Technical Details**: HALO 프레임워크는 LLMs(대규모 언어 모델)의 여러 쿼리 변형을 생성하고, 외부 열린 지식 기반에서 관련 정보를 검색하여 문맥을 풍부하게 만듭니다. 최대 한계 관련성(maximum marginal relevance) 점수를 사용하여 검색된 문맥의 우선순위를 매기고 이를 LLM에 제공하여 정답 생성을 합니다. LangChain을 통합하여 과정을 간소화했습니다.

- **Performance Highlights**: HALO는 Llama-3.1의 정확도를 44%에서 65%로, ChatGPT는 56%에서 70%로 크게 향상시켰습니다. 이 결과는 HALO가 의료 QA 시스템에서 환각 문제를 해결하는 데 중요한 역할을 함을 보여줍니다.



### SelECT-SQL: Self-correcting ensemble Chain-of-Thought for Text-to-SQL (https://arxiv.org/abs/2409.10007)
- **What's New**: SelECT-SQL은 자연어 질문을 SQL 쿼리로 자동 변환하는 Text-to-SQL 작업을 개선하는 새로운 in-context learning 솔루션으로, 이전 결과를 초월하는 성능을 보여줍니다.

- **Technical Details**: SelECT-SQL은 Chain-of-Thought (CoT) 프롬프트, 자기 교정 기능, 앙상블 기법을 조합하여 구성되어 있으며, 이를 통해 complex한 Text-to-SQL 작업에서 정확성을 극대화합니다. 특히, 구조 합성 CoT 기법을 사용하여 SQL 쿼리를 생성하는 다양한 단계를 자동으로 유도합니다.

- **Performance Highlights**: SelECT-SQL은 GPT-3.5-Turbo를 기반으로 Spider 리더보드 개발 세트에서 84.2%의 실행 정확도를 달성하며, 이는 다른 GPT-3.5-Turbo 솔루션(81.1%) 및 GPT-4 성능(83.5%)을 초월하는 결과입니다.



### Comprehensive Study on Sentiment Analysis: From Rule-based to modern LLM based system (https://arxiv.org/abs/2409.09989)
Comments:
          2 Images

- **What's New**: 이 논문은 인공지능(AI) 및 대형 언어 모델(LLMs) 맥락에서 감정 분석(Sentiment Analysis)에 대한 포괄적인 조사를 제공합니다. 전통적인 규칙 기반 방법에서 고급 딥 러닝 기술로의 전환을 강조하며 감정 분석의 역사적 발전을 살펴봅니다.

- **Technical Details**: 감정 분석은 자연어 처리(NLP)의 중요한 측면으로, 사전 기반 및 패턴 기반 접근에서 기계 학습(Machine Learning) 및 딥 러닝(Deep Learning) 모델로의 발전을 포함합니다. 또한 이 논문에서는 이중 언어 텍스트 처리, 풍자 감지, 편향 문제 등 주요 도전 과제를 논의합니다.

- **Performance Highlights**: 최신 접근 방식을 검토하고 새로운 트렌드를 식별하며 이 분야의 발전을 위한 미래 연구 방향을 제시합니다. 현재의 방법론을 종합하고 미래의 기회를 탐색하기 위해 감정 분석을 AI 및 LLM 맥락에서 철저히 이해하는 것을 목표로 하고 있습니다.



### Gaps or Hallucinations? Gazing into Machine-Generated Legal Analysis for Fine-grained Text Evaluations (https://arxiv.org/abs/2409.09947)
- **What's New**: 본 연구는 기계 생성 법률 분석의 평가 기준을 제시하고 새로운 비중의 개념인 '갭(gaps)'을 도입하였습니다. 이는 인간 작성과 기계 생성 법률 분석 사이의 차이를 나타내며, 갭이 무조건 잘못된 생성이라고 할 수 없음을 강조합니다.

- **Technical Details**: 연구진은 CLERC 생성 작업을 바탕으로 갭의 유형 분류 체계를 구축하고, 법률 전문가와 협력하여 수작업으로 주석이 달린 데이터셋을 생성했습니다. 최상의 탐지기는 67% F1 점수와 80% 정밀도를 달성했습니다. 또한, GapScore와 GapHalu라는 자동 평가 지표를 통해 약 80%의 CLERC 생성물이 환각(hallucinations)을 포함한다고 밝혔습니다.

- **Performance Highlights**: 이 연구의 탐지기는 법률 분석 생성의 자동 평가를 수행할 수 있으며, 기존 SOTA LLMs가 생성한 법률 분석의 정확성을 개선하는 데 기여할 것으로 기대됩니다.



### Towards Data Contamination Detection for Modern Large Language Models: Limitations, Inconsistencies, and Oracle Challenges (https://arxiv.org/abs/2409.09927)
Comments:
          12 pages, 1 figure

- **What's New**: 이 연구는 최신의 대형 언어 모델(LLM)에서의 데이터 오염(datat contamination) 감지 방법의 효과성과 신뢰성을 평가하기 위한 새로운 접근을 제시합니다. 연구진은 SOTA LLM의 오염 상태와 감지 방법의 견고함을 이중 분석함으로써 오염 감지의 복잡성과 필요성을 강조합니다.

- **Technical Details**: 연구에서는 다섯 가지 데이터 오염 감지 접근법을 평가하였으며, 이를 네 개의 최신 LLM 모델(GPT-4, Claude 3 Sonnet, LLaMA-3-Chat, LLaMA-2-Chat)과 여덟 개의 도전적인 데이터셋을 사용하여 분석하였습니다. 또한, LLaMA-2 모델을 사용하여 의도적으로 오염된 상태를 생성하고, 감지 방법의 효과성을 평가하기 위한 기준을 설정했습니다.

- **Performance Highlights**: 실험 결과, 현재 사용되고 있는 모든 데이터 오염 감지 방법들은 각각의 기본 가정이나 실제 응용에 한계가 있으며, 특히 최신의 도전적인 기준을 통해서는 일관된 결과를 도출하지 못했습니다. 더불어, 기존의 방법들이 일반적으로 고전적인 기준에서는 어느 정도 효과를 보이지만, 복잡한 기준에서는 불충분한 성능을 보임을 확인했습니다.



### SFR-RAG: Towards Contextually Faithful LLMs (https://arxiv.org/abs/2409.09916)
Comments:
          Technical report

- **What's New**: 이번 연구에서 우리는 SFR-RAG 모델을 소개합니다. 이 모델은 외부 맥락 정보를 적극적으로 활용하여 정보의 정확성과 관련성을 향상시키기 위해 설계된 작은 LLM입니다. 또한, ContextualBench라는 새로운 평가 프레임워크를 도입하여 다양한 RAG 벤치마크에서 모델의 성능을 일관되게 평가합니다.

- **Technical Details**: SFR-RAG 모델은 지식 검색기와 연계하여 작동하며, 복잡한 멀티홉 추론(multi-hop reasoning)과 신뢰할 수 있는 인용을 생성하는 기능이 포함되어 있습니다. 모델은 90억 파라미터로 구성되어 있으며, RAG 및 관련 에이전틱 작업에 특화된 교육을 받았습니다. 이 모델은 문서 데이터베이스에서 관련 패시지를 검색하는 과정을 단순화하며, ContextualBench를 통해 일관된 평가를 제공합니다.

- **Performance Highlights**: SFR-RAG-9B 모델은 ContextualBench의 7개 벤치마크 중 3개에서 최고 성능을 기록하며, Command-R+ (104B)와 GPT-4o를 상회하는 성능을 보여주었습니다. 이 모델은 10배 적은 파라미터로도 강력한 성능을 유지하며, 맥락 정보의 변화에도 강한 저항력을 보입니다. 또한 일반적인 지시 준수 작업에서도 경쟁력 있는 성능을 나타냅니다.



### Rediscovering the Latent Dimensions of Personality with Large Language Models as Trait Descriptors (https://arxiv.org/abs/2409.09905)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)이 성격 특성을 암묵적으로 인코딩하고 있는지를 조사합니다. LLMs의 다음 단어 예측에서 얻은 로그 확률을 분석하여 잠재적인 성격 차원을 발견하는 새로운 접근법을 소개합니다.

- **Technical Details**: 본 연구에서는 singular value decomposition (SVD)를 사용하여 LLM의 trait-descriptive adjectives(특성 설명 형용사)의 로그 확률을 분석하여 Core Personality Traits(핵심 성격 특성)를 Rediscover(재발견)합니다. 연구는 100개의 성격 형용사를 바탕으로 작성된 개인 이야기를 사용하여 이루어집니다.

- **Performance Highlights**: 연구 결과, LLMs는 Big Five(오대성격) 모델과 유사한 잠재 구조를 인코딩하고 있으며, 평가 정확성을 평균 5% 향상시키고 이전 기법에 비해 최대 21% 더 높은 성과를 보여줍니다.



### Acquiring Pronunciation Knowledge from Transcribed Speech Audio via Multi-task Learning (https://arxiv.org/abs/2409.09891)
Comments:
          5 pages

- **What's New**: 본 논문에서는 텍스트-음성 변환(TTS)을 위한 통합 시퀀스-투-시퀀스(Seq2Seq) 언어 전면모델을 부트스트래핑하는 새로운 방법을 제안합니다. 이 방법은 복잡한 자동 음성 인식(ASR) 모델을 사용하지 않고, 다중 작업 학습(MTL)을 통해 전사된 음성을 추가 학습 원천으로 활용합니다.

- **Technical Details**: 제안된 MTL 기반 방법은 기존 Seq2Seq 전면모델과 전사된 음성을 통한 추가 작업(음향 특성 회귀)을 동시에 학습하여 발음 지식을 향상시킵니다. 기존의 방법에 비해 PER(Phoneme Error Rate)을 2.5%에서 1.6%로 감소시키는 성과를 보였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 전사된 음성만을 통해 다루어진 단어들에 대해 유사한 성능을 달성하면서도, 훨씬 단순한 구현 프로세스를 가지고 있습니다.



### Constructing a Singing Style Caption Datas (https://arxiv.org/abs/2409.09866)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 S2Cap라는 새로운 오디오-텍스트 쌍 데이터셋을 소개합니다. 이 데이터셋은 다양한 보컬 및 음악적 특성을 포함하고 있으며, 특히 노래 스타일 캡셔닝(Singing Style Captioning) 작업을 수행하기 위한 최초의 전용 작업으로 제안되었습니다.

- **Technical Details**: S2Cap 데이터셋은 피치(pitch), 볼륨(volume), 템포(tempo), 기분(mood), 가수의 성별(gender) 및 나이(age), 음악 장르(musical genre) 및 감정 표현(emotional expression)과 같은 9가지 속성(attribute)을 고려합니다. 이를 기반으로 한 새로운 baseline 알고리즘은 CRESCENDO라는 메커니즘을 사용하여 오디오 인코더와 텍스트 디코더의 정렬 문제를 해결하며, 보컬 데믹싱(vocal demixing)을 통해 모델이 보컬 특성을 보다 정확하게 캡처할 수 있도록 합니다.

- **Performance Highlights**: S2Cap 데이터셋은 12,105개의 음악 트랙에서 71,215개의 캡션을 추출하였으며, 이를 통해 노래 스타일 캡셔닝의 성능을 크게 향상시킬 수 있음을 보여줍니다. 이 연구는 노래의 보컬 및 음악적 특성을 보다 잘 반영하는 캡션의 생성 가능성을 높입니다.



### A Benchmark Dataset with Larger Context for Non-Factoid Question Answering over Islamic Tex (https://arxiv.org/abs/2409.09844)
- **What's New**: 이 논문에서는 이슬람의 경전인 꾸란(Quran)과 하디스(Ahadith)에 대한 질문-답변(QA) 시스템의 필요성을 강조하고, 이들을 위한 맞춤형 데이터세트를 소개합니다. 이 데이터세트는 73,000개 이상의 질문-답변 쌍으로 구성되어 있으며, 이 분야에서 가장 큰 데이터세트로 보고됩니다.

- **Technical Details**: 제안된 데이터세트는 질문 및 답변에 대한 맥락 정보를 포함하고 있으며, QA 시스템의 훈련 및 평가를 위한 중요한 자원으로 사용될 수 있습니다. 또한 논문은 QA 성능을 평가하기 위한 벤치마크를 도입하여 연구자들이 모델의 성능을 평가하고 기존 접근 방식과 비교할 수 있도록 합니다.

- **Performance Highlights**: 자동 평가 기술의 한계가 드러났습니다. 전문가와 모델 판단 간의 일치율이 11%에서 20% 사이로 조사되었으며, 맥락 이해는 50%에서 90% 사이로 나타났습니다. 이는 종교 텍스트의 이해에서 섬세함과 복잡성을 포착하는 평가 기술의 필요성을 강조합니다.



### Generating Synthetic Free-text Medical Records with Low Re-identification Risk using Masked Language Modeling (https://arxiv.org/abs/2409.09831)
- **What's New**: 본 논문에서는 Masked Language Modeling (MLM)을 이용하여 퇴원 요약서, 입원 기록 및 의사 간의 서신 등 합성 자유 형식 의무기록을 생성하는 시스템을 소개합니다. 이 시스템은 기록의 중요한 정보를 보존하면서도 다양성을 높이고 재식별 위험을 최소화하도록 설계되었습니다.

- **Technical Details**: 시스템은 Protected Health Information (PHI)을 마스킹하기 위해 Philter라는 탈식별(de-identification) 구성 요소를 포함하고, 주요 의료 정보를 보존하기 위해 Medical Entity Recognition (NER) 모델을 사용합니다. 다양한 마스킹 비율(masking ratios)과 마스킹 채우기 기법(mask-filling techniques)을 탐구하여 합성 출력의 다양성과 충실도 간의 균형을 맞추었습니다.

- **Performance Highlights**: 시스템은 높은 품질의 합성 데이터를 생성할 수 있으며 0.96의 HIPAA 준수 PHI 회수율과 0.035의 낮은 재식별 위험을 달성했습니다. 또한 NER 작업을 통한 후속 평가에서는 합성 데이터가 실제 데이터로 훈련된 모델과 동등한 성능을 발휘할 수 있음을 보여주었습니다.



### GP-GPT: Large Language Model for Gene-Phenotype Mapping (https://arxiv.org/abs/2409.09825)
- **What's New**: GP-GPT는 유전적 표현형 지식 표현 및 유전체 관계 분석을 위한 첫 번째 전문화된 대형 언어 모델로, 3,000,000개 이상의 유전체, 단백질체 및 의학 유전 관련 용어를 사용하는 두 단계의 미세 조정을 통해 개발되었습니다.

- **Technical Details**: GP-GPT는 OMIM, DisGeNET 및 dbGaP와 같은 여러 주요 데이터 출처의 구조화된 데이터와 비구조화된 데이터를 활용하여 유전자 및 표현형의 관계를 통합적으로 모델링합니다. 이 모델은 질병 유전학 쿼리에서 정확한 답변을 제공하는 능력, 유전 정보 검색, 그리고 유전자와 표현형 간의 관계 결정 능력을 평가하기 위해 세 가지 주요 NLP 작업을 수행합니다.

- **Performance Highlights**: GP-GPT는 Llama2, Llama3 및 GPT-4와 같은 최신 LLM보다 우수한 성능을 보였습니다. 이 모델은 유전 질환의 관계 연구를 향상시키고 유전체 및 의료 유전학 분야에서 정확하고 효율적인 분석을 촉진할 수 있는 잠재력을 가지고 있습니다.



### Causal Inference with Large Language Model: A Survey (https://arxiv.org/abs/2409.09822)
Comments:
          15 pages, 2 figures, 3 tables

- **What's New**: 본 논문은 causal inference (인과 추론) 분야에서의 large language models (LLMs)의 최근 발전을 정리하고, 각각의 인과 문제와 접근 방식을 비교하여 LLM이 제시하는 독특한 기회를 탐구합니다.

- **Technical Details**: 인과 추론은 관찰 뒤에 있는 인과 관계를 밝혀내는 중요한 분야로, 인간의 지식, 수학적 추론, 데이터 마이닝을 복합적으로 통합해야 합니다. 전통적인 인과 추론 프레임워크에서는 numerical 값보다 domain knowledge (도메인 지식)를 강조하는 반면, LLMs는 대량의 텍스트 정보에서 도메인 지식을 추출하여 인과 관계를 더 잘 이해하는 데 기여할 수 있습니다.

- **Performance Highlights**: LLMs는 인과 관계를 이해하고 해석하는데 있어 명확한 이점을 제시하며, 텍스트 데이터 내에서의 인과 관계 파악 및 설명 가능한 인과 추론을 가능하게 합니다. 이 논문에서는 LLM 기반 인과 추론에서앞으로의 연구 방향과 제약 사항에 대한 논의도 포함되어 있습니다.



### Large Language Model Based Generative Error Correction: A Challenge and Baselines forSpeech Recognition, Speaker Tagging, and Emotion Recognition (https://arxiv.org/abs/2409.09785)
Comments:
          IEEE SLT 2024. The initial draft version has been done in December 2023. Post-ASR Text Processing and Understanding Community: this https URL

- **What's New**: 이번 논문에서는 고정된 자동 음성 인식(ASR) 모델의 텍스트 디코딩 결과를 활용하여 음향 모델링 과제를 향상시키는 방법을 모색합니다. 특히, 새로운 '생성 음성 전사 오류 수정(GenSEC)' 챌린지를 소개하며, 이는 ASR 후의 언어 모델링 작업으로 3가지 주요 과제를 포함합니다.

- **Technical Details**: 제안된 챌린지는 ASR-LLM(대형 언어 모델) 연계 모델을 통해 성능 향상을 꾀하고 있으며, 시행될 세 가지 작업은 (1) ASR 후 전사 수정, (2) 화자 태깅, (3) 감정 인식을 포함합니다. 이 챌린지는 텍스트 기반 추론을 통해 음향적 정보가 포함된 오류 수정 방법을 탐구하여, 음성 인식 시스템을 사용할 때의 새로운 접근법을 제시합니다.

- **Performance Highlights**: 기존 ASR 처리 방식에서 LLM은 외부 지식을 활용하여 정확성을 높이고 있으며, 다양한 성능 측정 지표를 통해 그 효용성 및 가능성을 보여주고 있습니다. 특히, 대형 언어 모델을 활용하여 비언어적 정보와 발화의 맥락을 활용하면서 음성 인지의 틀을 확장하고자 합니다.



### Benchmarking LLMs in Political Content Text-Annotation: Proof-of-Concept with Toxicity and Incivility Data (https://arxiv.org/abs/2409.09741)
Comments:
          Paper prepared for delivery at the 8th Monash-Warwick-Zurich Text-as-Data Workshop, September 16-17, 2024: 11 pages, 3 tables, 3 figures

- **What's New**: 이번 연구는 OpenAI의 GPT 모델들과 여러 오픈 소스 LLM이 정치적 콘텐츠에 대한 주석 작업을 수행하는 능력을 벤치마킹했습니다. 이 연구는 300만 개 이상의 디지털 상호작용을 포함하는 새로운 시위 사건 데이터 세트를 사용했으며, 소셜 미디어에서의 독성 및 무례함에 대해 인간 코더가 주석을 단 진리 기준(gold standard) 레이블을 포함했습니다.

- **Technical Details**: 벤치마킹에 포함된 모델은 Google의 Perspective 알고리즘과 OpenAI의 GPT 모델들, 그리고 로컬에서 배포된 오픈 소스 LLM들입니다. 결과적으로, 느슨한 임계 값을 사용하는 Perspective API, GPT-4o 및 Nous Hermes 2 Mixtral이 다른 LLM의 제로 샷 분류 주석보다 우수한 성능을 보였습니다.

- **Performance Highlights**: Nous Hermes 2와 Mistral OpenOrca는 매개변수가 적음에도 불구하고 높은 성능을 발휘하여 성능, 구현 비용 및 컴퓨팅 시간 간의 좋은 균형을 제공할 수 있는 매력적인 선택지로 평가되었습니다. GPT 모델은 전반적으로 좋은 신뢰성과 컴퓨팅 시간을 보여주었지만, 오픈 소스 LLM만이 주석 작업의 완전한 재현성을 보장할 수 있었습니다.



### AlpaPICO: Extraction of PICO Frames from Clinical Trial Documents Using LLMs (https://arxiv.org/abs/2409.09704)
Comments:
          Accepted at Methods

- **What's New**: 최근 증가하는 임상 시험 보고서로 인해 체계적인 리뷰 수행이 어려워지고 있으며, 본 연구에서는 자연어 처리(NLP) 기반의 자동 PICO 관련 용어 추출 방법을 제안합니다. 이는 기존의 수동 데이터 주석 작업을 대체하기 위한 시도로, 대규모 주석 데이터를 필요로 하지 않는 비지도 학습 설정에서 진행됩니다.

- **Technical Details**: PICO(Population, Intervention, Comparator, Outcome) 프레임 추출을 위해 본 연구는 In-Context Learning (ICL) 전략을 채택했으며, 기존 대형 언어 모델(LLM)에서 수집된 사전 훈련된 지식을 활용합니다. 또한, Low Rank Adaptation (LORA) 기법을 적용하여 자원이 제한된 환경에서도 대형 모델 훈련이 가능하도록 하였습니다.

- **Performance Highlights**: 실험 결과, ICL 기반 프레임워크는 EBM-NLP 데이터셋의 모든 버전에서 비교 가능한 결과를 보여주었으며, 지침 조정된(instruction-tuned) 버전은 EBM-NLP 데이터셋에서 최첨단(SOTA) 결과를 기록하였습니다.



### Leveraging Open-Source Large Language Models for Native Language Identification (https://arxiv.org/abs/2409.09659)
- **What's New**: 이 논문은 오픈소스 LLMs를 사용한 Native Language Identification (NLI) 성능을 탐구한 최초의 연구입니다. 연구 결과에 따르면, 오픈소스 LLMs는 궁극적으로 라벨이 붙은 훈련 데이터에 대해 파인튜닝(fine-tuned) 될 경우 상용 LLMs와 유사한 성능을 달성할 수 있습니다.

- **Technical Details**: NLI는 기계 학습(Machine Learning) 관점에서 다중 클래스 분류(multi-class classification) 문제로 접근되며, L1의 언어적 특징이 L2의 글쓰기에서 어떻게 나타나는지를 분석합니다. 연구에서는 5개의 오픈소스 LLMs(LLaMA-2, LLaMA-3, Gemma, Mistral, Phi-3)와 2개의 폐쇄형 LLMs(GPT-3.5, GPT-4)를 비교하였으며, TOEFL11과 ICLE-NLI 데이터를 사용하여 성능을 평가했습니다.

- **Performance Highlights**: 오픈소스 LLMs는 초기 상태에서는 상용 LLMs의 정확도에 도달하지 못하지만, 라벨 데이터로 파인튜닝하면 상용 LLMs와 대등한 성능을 나타냅니다. 특히, GPT-4는 TOEFL11 벤치마크 데이터셋에서 91.7%의 정확도를 기록하며 NLI 작업에서 뛰어난 성과를 보여주었습니다.



### Unveiling Gender Bias in Large Language Models: Using Teacher's Evaluation in Higher Education As an Examp (https://arxiv.org/abs/2409.09652)
- **What's New**: 이 연구는 대학 환경에서 GPT-4가 생성한 교사 평가에서 성별 편향(gender bias)을 조사합니다. LLM(대형 언어 모델) 이 생성한 평가에서 특정 단어들이 여성이 아닌 남성 강사에게 더 자주 사용되는 경향을 보여주며, 이는 사회적 고정관념을 반영합니다.

- **Technical Details**: 연구는 Odds Ratio (OR) 분석, Word Embedding Association Test (WEAT), 감정 분석(sentiment analysis), 맥락 분석(contextual analysis) 등 포괄적인 해석 프레임워크를 적용하여 성별 관련 언어 패턴을 식별했습니다. 연구 결과, 여성 강사에게는 접근성과 지원을 나타내는 단어가 더 자주 사용되는 반면, 남성 강사에게는 오락과 관련된 단어가 주로 사용되었습니다. 또한 남성 형용사와 남성 이름 간의 강한 연관성이 발견되었습니다.

- **Performance Highlights**: 이 연구의 발견은 LLM이 생성하는 텍스트가 기존의 성별 편향을 반영하고 있다는 선행 연구와 일치합니다. 연구는 AI가 만들어내는 언어가 교사 평가와 같은 중요한 전문 평가에서 성별 고정관념을 지속할 가능성을 가지고 있음을 강조합니다.



### A Simple HMM with Self-Supervised Representations for Phone Segmentation (https://arxiv.org/abs/2409.09646)
Comments:
          Accepted to SLT 2024

- **What's New**: 최근의 자기 지도 학습(self-supervised learning) 접근방식과는 달리, Mel 스펙트로그램(Mel spectrograms)에서의 피크 감지를 통해 음성 구간(segmentation)을 수행하는 강력한 기준선을 제시합니다. 이 연구에서는 간단한 Hidden Markov Model(HMM)을 제안하고, 이를 통해 음성 구간을 효과적으로 분리합니다.

- **Technical Details**: 입력된 음성 신호는 Mel 스펙트로그램 처리 후, SVF(spectral variation function)와 결합된 피크 감지 알고리즘을 사용하여 음의 경계를 탐지합니다. HMM은 자기 지도 표현(self-supervised representations)과 경계 특징을 결합하여, 음을 더 효과적으로 구분할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 HMM은 TIMIT와 Buckeye 데이터셋에서 검증된 결과, 기존의 피크 감지 및 DPDP(Duration-Penalized Dynamic Programming) 기법들을 지속적으로 초월하는 성능을 보였으며, 신경망을 사용하는 접근 방법들과도 유사하거나 더 나은 성능을 기록했습니다.



### Towards understanding evolution of science through language model series (https://arxiv.org/abs/2409.09636)
- **What's New**: AnnualBERT는 과학 텍스트의 시간적 진화를 포착하기 위해 특별히 설계된 언어 모델 시리즈입니다. 이 모델은 전체 단어를 토큰으로 사용하고, 2008년까지 발표된 1.7백만 개의 arXiv 논문의 전체 텍스트로부터 초기학습(Pretraining)을 한 기본 RoBERTa 모델로 구성되어 있습니다.

- **Technical Details**: AnnualBERT는 과학 텍스트를 연도별로 정리하여 지속적으로 훈련하는 두 단계의 훈련 전략을 채택합니다. 이 모델은 표준 NLP 작업에서 유사한 성능을 보일 뿐만 아니라, 도메인 특화 NLP 작업 및 arXiv 인용 네트워크의 링크 예측 작업에서 최첨단 성능을 달성합니다.

- **Performance Highlights**: AnnualBERT 모델은 arXiv 인용 네트워크의 링크 예측 작업에서 다른 모델들에 비해 뛰어난 성능을 나타내며, 과학 텍스트 처리 작업에서도 성능 향상을 보여줍니다.



### Confidence Estimation for LLM-Based Dialogue State Tracking (https://arxiv.org/abs/2409.09629)
- **What's New**: 이 논문은 대화형 AI 시스템의 신뢰성을 개선하기 위한 방법을 제시하며, 주로 대화 상태 추적(Dialogue State Tracking, DST)에서 모델 불확실성을 활용하는 새로운 접근 방식을 탐색하고 있습니다.

- **Technical Details**: 연구에서는 open-weight 및 closed-weight LLMs에 대해 신뢰도 점수(confidence scores)를 추정하는 네 가지 방법을 평가하고, softmax, 원시 토큰 점수(raw token scores), 언어로 표현된 신뢰도(verbalized confidences) 및 이들의 조합을 바탕으로 한 방법을 사용했습니다. AUC(Area Under the Curve) 메트릭을 이용하여 신뢰도 점수를 평가하며, self-probing 메커니즘을 통해 closed models의 성능을 향상시켰습니다.

- **Performance Highlights**: fine-tuning된 open-weight 모델을 활용한 DST에서 우수한 joint goal accuracy(JGA)를 달성했으며, 모델 신뢰도 점수의 보정 수준이 향상되었다는 점이 주요 성과로 나타났습니다.



### Enhancing Text Annotation through Rationale-Driven Collaborative Few-Shot Prompting (https://arxiv.org/abs/2409.09615)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)을 활용하여 데이터 주석 작업의 효율성과 일관성을 개선하는 새로운 방법론인 근거 기반 협업 텍스트 주석 방법(Rationale-Driven Collaborative annotation method)을 제안합니다.

- **Technical Details**: 연구진은 LLM의 출력을 개선하기 위해 여러 LLM의 주석 결과를 통합하는 근거 기반 협업 접근 방식을 사용합니다. 이 방법은 이전 주석을 참조하여 순차적으로 주석을 수행하며, 유사한 예제를 비교하여 정밀도를 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 몇 개의 샷(few-shot) 기법 및 기타 기준 방법들보다 향상된 주석 정확도를 보여주었으며, 특히 복잡한 주석 작업에서 두드러진 성과를 나타냈습니다.



### Rethinking KenLM: Good and Bad Model Ensembles for Efficient Text Quality Filtering in Large Web Corpora (https://arxiv.org/abs/2409.09613)
- **What's New**: 이 논문에서는 대량의 웹 데이터에서 고품질 데이터를 효과적으로 필터링하기 위해 Good KenLM(우수한 n-그램 모델)과 Bad KenLM(저품질 데이터 모델)을 앙상블하는 방법을 제안합니다. 기존의 KenLM 교육 방식의 한계를 극복할 수 있는 접근법입니다.

- **Technical Details**: Good KenLM은 고품질 데이터로 훈련된 n-그램 기반 언어 모델이며, Bad KenLM은 노이즈가 포함된 저품질 데이터로 훈련되었습니다. 두 모델의 PPL(perplexity) 점수를 결합하여 불필요한 내용을 효과적으로 제거하며, Z-score normalization을 통해 두 모델의 점수를 통합합니다.

- **Performance Highlights**: 제안된 앙상블 방법은 전통적인 KenLM 훈련 방법에 비해 노이즈 콘텐츠를 효과적으로 줄이며, 고품질 콘텐츠를 보존하면서도 계산 리소스 사용을 최소화한 실험 결과를 보였습니다.



### Improving Statistical Significance in Human Evaluation of Automatic Metrics via Soft Pairwise Accuracy (https://arxiv.org/abs/2409.09598)
- **What's New**: 이 논문에서는 Soft Pairwise Accuracy (SPA)라는 새로운 메타 메트릭을 제안합니다. 이는 기존 Pairwise Accuracy (PA)의 단점을 보완하며, 인간의 평가와 메트릭 평가의 통계적 중요성을 모두 고려합니다.

- **Technical Details**: SPA는 메트릭의 평가와 인간의 평가의 p-값을 사용하여 메트릭이 인간의 판단과 얼마나 일치하는지를 평가합니다. 이 메트릭은 시스템 수와 평가 세그먼트 수에 대한 안정성이 높으며, 메트릭 간의 동점 문제(quantization)를 완화합니다.

- **Performance Highlights**: SPA는 2024 WMT 메트릭 공유 과제의 공식 시스템 레벨 메트릭으로 선정되었습니다. PA에 비해 보다 통계적으로 유의미한 결과를 제공하며, 두 시스템 간의 성능 비교를 보다 세밀하게 수행할 수 있습니다.



### Thesis proposal: Are We Losing Textual Diversity to Natural Language Processing? (https://arxiv.org/abs/2409.09568)
- **What's New**: 이번 논문에서는 현재 널리 사용되고 있는 Natural Language Processing (NLP) 알고리즘의 한계에 대해 논의하며, Neural Machine Translation (NMT) 작업을 테스트베드로 설정하여 다양한 텍스트의 특성과 처리에서의 문제를 탐구합니다. 특히, 텍스트 다원성의 통계적 특성을 바탕으로 측정 기준을 정의하고 이를 통해 NMT 시스템이 비정상적인 텍스트 처리에 어려움을 겪는지 분석합니다.

- **Technical Details**: 본 연구는 텍스트의 정보 분포에 대한 유니폼성과 전형성을 다양한 스케일(문장, 담화, 언어)에서 평가하기 위한 기준을 수립합니다. 정보 내용 측정을 위한 surprisal 개념을 사용하고, kaynak 모델에 의해 예측되는 읽기 시간을 기준으로 문장 및 문단의 surprisal 값의 유니폼성을 평가합니다. 또한, 담화 수준에서의 정서적 패턴을 관찰하여 청중의 참여도와 관련된 예측을 수행합니다.

- **Performance Highlights**: NMT 시스템은 특히 시적인 텍스트와 같은 높은 놀라움을 주는 텍스트의 surprisal 분포의 비유니폼성을 유지하는 데 실패하는 경향을 보였습니다. 이런 문제를 해결하기 위해, 새로운 디코딩 알고리즘과 훈련 목표를 제안하며, 이전 세대 모델의 출력을 활용하여 MT 품질 평가 메트릭스에 기반한 샘플링 기반 알고리즘을 도입할 계획입니다.



### ASR Error Correction using Large Language Models (https://arxiv.org/abs/2409.09554)
Comments:
          Submitted to IEEE Transactions on Audio, Speech and Language Processing

- **What's New**: 이 논문은 대형 언어 모델(LLM)을 활용하여 자동 음성 인식(ASR) 전사에서 발생하는 오류를 수정하는 혁신적인 방법을 제안합니다. 특히, N-best 리스트를 활용하여 보다 풍부한 맥락 정보를 통해 오류 수정 성능을 향상시키는 방법을 탐구합니다.

- **Technical Details**: 연구에서는 ASR의 N-best 리스트를 입력으로 사용하는 오류 수정(EC) 모델을 제안합니다. 이 모델은 다양한 인코더-디코더 아키텍처를 사용할 수 있으며, 접근 제한이 있는 ASR 시스템에서도 적용 가능하도록 설계되었습니다. N-best 리스트를 기반으로 한 제한된 디코딩(constrained decoding) 기법을 도입하여 오류 수정 과정에서 더 높은 성능을 달성합니다.

- **Performance Highlights**: 세 가지 표준 데이터셋에서 실험을 통해, ASR N-best 리스트를 활용한 EC 모델이 기존의 1-best 가설에 비해 뛰어난 성능을 보였음을 입증했습니다. 이 연구는 모델 앙상블(model ensembling) 방법으로도 적용 가능성을 보여주며, ASR 오류 수정 분야에서 성과를 한 단계 끌어올리는 데 기여하고 있습니다.



### Comparing Retrieval-Augmentation and Parameter-Efficient Fine-Tuning for Privacy-Preserving Personalization of Large Language Models (https://arxiv.org/abs/2409.09510)
- **What's New**: 이 논문은 개인화된 대형 언어 모델(LLM)에 대한 프라이버시 보존 방법을 체계적으로 비교한 첫 논문이다. RAG 기반 방법과 PEFT 기반 방법 간의 비교를 통해 다양한 개인화 작업에서의 성능을 평가한다.

- **Technical Details**: 이 연구는 LLM을 개인화하기 위한 두 가지 접근 방식, 즉 Retrieval-Augmented Generation (RAG)과 Parameter-Efficient Fine-Tuning (PEFT)을 비교한다. RAG는 사용자 프로필에서 개인화된 정보를 검색하여 입력을 수정하는 반면, PEFT는 사용자 맞춤 데이터에 따라 LLM의 매개변수를 세분화하여 조정한다. 실험은 LaMP 벤치마크에서 수집된 7개의 다양한 데이터셋을 사용하여 수행되었다.

- **Performance Highlights**: RAG 기반 개인화는 비개인화 LLM에 비해 평균적으로 14.92%의 성능 향상을 가져오며, PEFT 기반 방법은 1.07%의 향상을 보인다. 두 방법을 결합할 경우, 성능 향상은 15.98%에 달한다. 또한, 사용자 데이터 양과 PEFT의 효과성 간의 긍정적인 상관관계를 발견했으며, 이는 RAG가 데이터가 제한된 신규 사용자에게 더 적합하다는 것을 나타낸다.



### Uddessho: An Extensive Benchmark Dataset for Multimodal Author Intent Classification in Low-Resource Bangla Languag (https://arxiv.org/abs/2409.09504)
Comments:
          Accepted for publication in "18th International Conference on Information Technology and Applications (ICITA 2024)"

- **What's New**: 본 논문은 소셜 미디어 포스트에서의 Bangla 언어의 의도 분류를 위한 새로운 접근법인 Multimodal-based Author Bangla Intent Classification (MABIC) 프레임워크를 제안합니다. 기존의 텍스트 기반 방법의 한계를 극복하고, 텍스트 및 이미지를 활용하여 저자의 잠재적 목적을 분석하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MABIC 프레임워크는 Early Fusion 및 Late Fusion 기술을 통해 텍스트와 이미지를 결합하여 저자 의도를 분류합니다. 본 연구에서 사용된 데이터셋은 'Uddessho'라는 이름으로, 3048개의 Bangla 소셜 미디어 게시물을 포함하며, 6개의 카테고리(Informative, Advocative, Promotive, Exhibitionist, Expressive, Controversial)로 분류됩니다.

- **Performance Highlights**: 단일 모달 접근 방식은 Bangla 텍스트 의도 해석에서 64.53%의 정확도를 기록했지만, 다중 모달 접근 방식은 76.19%의 정확도로 전통적인 단일 모달 방법보다 11.66% 향상된 성능을 보여주었습니다. 이는 저자 의도 분류를 위한 새로운 연구의 중요한 첫 발걸음이 됩니다.



### Synthetic4Health: Generating Annotated Synthetic Clinical Letters (https://arxiv.org/abs/2409.09501)
Comments:
          ongoing work, 48 pages

- **What's New**: 이 연구는 민감한 정보를 포함하는 임상 서신(clinical letters)의 생성을 위한 다양한 방법론을 탐색하며, 이러한 서신들을 쉽게 사용할 수 있는 신뢰할 수 있는 비식별화된 합성 임상 서신(synthetic clinical letters)을 생성하는 것을 목표로 합니다.

- **Technical Details**: 다양한 사전 훈련된 언어 모델(pre-trained language models, PLMs)을 활용하여 텍스트 마스킹(masking) 및 생성을 시도했습니다. Bio_ClinicalBERT 모델을 중점적으로 사용하며, 단어(예: 명사, 동사) 마스킹 전략(masking strategies)의 효과를 실험하고, 성능 평가는 정성적(methods) 및 정량적(methods) 측면에서 진행되었습니다. 또한, 합성 서신의 유용성을 평가하기 위해 명명된 개체 인식(Named Entity Recognition, NER) 작업도 수행되었습니다.

- **Performance Highlights**: 1) 인코더 전용 모델(encoder-only models)이 인코더-디코더 모델(encoder-decoder models)보다 성능이 우수하다. 2) 일반적인 코퍼스에서 훈련된 인코더 전용 모델이 임상 데이터에서 훈련된 모델과 비슷한 성능을 보인다. 3) 임상 개체 및 문서 구조의 보존이 모델의 미세 조정보다 우선시되어야 한다. 4) 다양한 마스킹 전략은 합성 임상 서신의 품질에 영향을 미친다. 5) 평가 지표로 BERTScore가 기본 지표로 사용되어야 하며, 나머지 지표들은 보조적인 참고 지표로 활용된다. 6) 문맥 정보는 모델의 이해도에 큰 영향을 미치지 않는다.



### Keeping Humans in the Loop: Human-Centered Automated Annotation with Generative AI (https://arxiv.org/abs/2409.09467)
- **What's New**: 이 연구에서는 최근 발표된 컴퓨터 사회과학 논문에 포함된 11개의 비공개 데이터세트에서 27개의 주석 작업을 복제하여 GPT-4의 성능을 점검하였습니다. 이는 기존의 연구 결과가 데이터 유출 및 오염의 영향을 받을 가능성이 높다는 점을 해결하기 위해 수행되었습니다.

- **Technical Details**: 연구에서는 GPT-4를 사용하여 주석을 생성하고, 인간 주석자에 의해 생성된 기준 레이블과 비교하였습니다. 모든 주석 작업은 고품질의 인간 주석 데이터를 기반으로 하며, 점검된 성과는 중간 정확도 0.850 및 F1 점수 0.707을 기록하였습니다. 하지만 27개의 작업 중 9개는 정밀도나 재현율이 0.5 이하로 떨어진 경우도 있었습니다.

- **Performance Highlights**: GPT-4의 주석 성능은 작업과 데이터셋에 따라 불일치가 보였으며, 27개 작업 중 20개 작업에서 재현율이 정밀도보다 높았습니다. 또한, 각 작업의 성능은 특히 잘된 경우와 맞지 않았으며, 충분한 훈련 데이터가 제공되면, 감독 학습 기반의 텍스트 분류기가 GPT-4보다 우수한 성능을 보였습니다.



### Enhancing LLM Problem Solving with REAP: Reflection, Explicit Problem Deconstruction, and Advanced Prompting (https://arxiv.org/abs/2409.09415)
Comments:
          524 pages, 3 figures

- **What's New**: 이 논문은 LLMs (Large Language Models)의 문제 해결 능력을 향상시키기 위한 REAP (Reflection, Explicit Problem Deconstruction, and Advanced Prompting) 방법을 소개합니다. 이는 동적인 Context Generation Framework에 기반한 혁신적인 접근 방식입니다.

- **Technical Details**: REAP은 쿼리에 대한 반성을 유도하고, 이를 관리 가능한 구성 요소로 분해하며, 솔루션 과정을 향상시키기 위해 적절한 Context를 생성합니다. 논문에서는 REAP을 사용하는 평가를 통해 OpenAI의 여러 모델과 Google's Gemini 1.5 Pro, Claude 3.5 Sonnet 등 총 6개 최첨단 모델과 함께 제로샷(Zero-shot) 프롬프트와 REAP 향상 프롬프트를 비교하였습니다.

- **Performance Highlights**: REAP을 통해 o1-mini는 40.97%, GPT-4o는 66.26%, GPT-4o-mini는 112.93% 향상되었습니다. o1-preview의 강력한 기준 성능에 비해 최소한의 향상이 관찰되었으며, REAP은 GPT-4o-mini와 같은 저비용 모델로도 경쟁력 있는 결과를 도출했습니다. REAP은 모델의 출력 명확성을 향상시킴으로써 인간이 모델 결과의 추론을 이해하기 쉽게 하고 문제 식별 및 해결 과정을 단순화하는 데 기여합니다.



### Constructive Approach to Bidirectional Causation between Qualia Structure and Language Emergenc (https://arxiv.org/abs/2409.09413)
Comments:
          20 pages, 4 Figures

- **What's New**: 이 논문은 언어의 출현과 주관적 경험의 관계 구조, 즉 qualia 구조 간의 상호작용에 대한 새로운 관점을 제시합니다. 이는 내부 표현의 정렬 과정을 통해 발생하는 언어의 체계적인 구조화와 관련이 있습니다.

- **Technical Details**: 이 연구에서는 neural network 기반의 언어 모델이 구조적 내부 표현을 형성하며, 여러 양식의 언어 모델이 언어와 지각 정보를 공유할 수 있음을 보여줍니다. 또한, cognitive systems의 상징 출현을 다루며, collective predictive coding (CPC) 모델을 통해 언어의 출현과 qualia 구조간의 상호 의존성을 수학적으로 모델링합니다.

- **Performance Highlights**: 연구 결과, 언어는 단순한 의사소통의 도구가 아니라, 주관적 경험의 정렬을 가능하게 하는 메커니즘으로 작용함을 발견하였습니다. 이는 언어와 인간 인식 간의 복잡한 관계를 이해하는 데 기여할 것입니다.



### Towards Diverse and Efficient Audio Captioning via Diffusion Models (https://arxiv.org/abs/2409.09401)
Comments:
this https URL

- **What's New**: 이번 논문에서는 다양한 오디오 캡셔닝을 위해 특별히 설계된 비 자가 회귀 방식의 디퓨전 모델(Diffusion-based Audio Captioning, DAC)을 소개합니다. 기존 모델들이 속도와 다양성 면에서 한계를 가지고 있었던 반면, DAC는 본래의 확률적 속성과 전체적인 문맥 모델링을 바탕으로 이러한 문제를 극복합니다.

- **Technical Details**: DAC는 Denoising Diffusion Probabilistic Model (DDPM)을 기반으로 하며, 텍스트 설명을 연속적인 텍스트 잠재 공간으로 변환하는 과정을 포함합니다. 오디오 내용은 Mel Spectrogram으로 변환되고, 경량화된 프로젝션 모듈을 통해 특징 공간으로 투영됩니다. DAC는 Mean Squared Error (MSE) 손실, Cross Entropy (CE) 손실, 보조 유효 토큰 손실 같은 세 가지 유형의 손실을 결합하여 최적화합니다.

- **Performance Highlights**: DAC는 기존 기준 모델들과 비교하여 생성 품질(SOTA 성능)에서도 경쟁력을 유지하며, 생성 속도와 다양성 면에서도 상존하는 전통적 자가 회귀 모델들보다 월등한 성능을 보여주었습니다. 추가적으로 CLAP 및 GPT4-eval과 같은 의미 기반 측정을 통해 DAC의 장점을 강조합니다.



### Generating Event-oriented Attribution for Movies via Two-Stage Prefix-Enhanced Multimodal LLM (https://arxiv.org/abs/2409.09362)
- **What's New**: 이 논문은 영화 비디오에서 사건의 원인을 이해하는 새로운 접근 방식을 제안합니다. 여기서는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 활용한 'Two-Stage Prefix-Enhanced MLLM (TSPE)' 방법론을 소개하며, 단일 클립 내의 사건 요약과 영화 전체를 아우르는 사건 분석을 두 단계로 진행합니다.

- **Technical Details**: 두 단계에서는 먼저 각 클립 내의 사건을 간략히 요약하는 지역(stage) 이해를 수행한 후, 전체 영화의 관점에서 사건 간의 관계를 분석하는 글로벌(stage) 분석을 실시합니다. 지역 단계에서는 상호작용 인식 프리픽스(interaction-aware prefix)를 활용하여 MLLM이 관련 멀티모달 정보를 효과적으로 집중할 수 있도록 합니다. 대조적으로, 글로벌 단계에서는 외부 지식 그래프 ATOMIC을 활용하여 전 사건 간의 의미적 연결을 강화하고, 사건 인식 프리픽스를 설계하여 관련 사건에 집중하도록 하였습니다.

- **Performance Highlights**: MovieGraph 데이터세트와 CHAR 데이터셋에서 실험을 통해 TSPE 방법이 최신 기술 수준(State-of-the-Art, SOTA) 방법들을 초월하는 성능을 보여줍니다. 구체적으로, TSPE는 사건 전달 정확도를 개선하고 멀티모달 정보를 효과적으로 요약하는 능력을 발휘하고 있습니다.



### Efficient Fine-Tuning of Large Language Models for Automated Medical Documentation (https://arxiv.org/abs/2409.09324)
Comments:
          4 pages, 3 Figures, 3 Tables, This is a preprint version of the article. The final version will be published in the proceedings of the IEEE conference

- **What's New**: MediGen은 의료 대화에서 의료 보고서를 자동 생성하기 위해 설계된 정교한 대형 언어 모델(LLM)입니다. 이 연구는 의료의 효율성을 높이고 의사의 번아웃을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: MediGen은 최신 방법론을 활용하여 공개 소스의 사전 훈련된 모델을 세밀하게 조정합니다. LLaMA3-8B 모델의 정교한 조정을 통해 임상 상호작용을 정확히 전사하고 요약하는 높은 정확도를 달성합니다. 모델의 성능은 ROUGE 스코어 58% 및 BERTScore-F1 72%로 나타났습니다.

- **Performance Highlights**: MediGen의 도입으로 의사들의 행정 업무 부담이 크게 줄어들 것으로 기대되며, 이는 의료 효율성 및 의사의 웰빙을 향상시킬 것으로 보입니다.



### A Compressive Memory-based Retrieval Approach for Event Argument Extraction (https://arxiv.org/abs/2409.09322)
Comments:
          15 pages

- **What's New**: 본 연구에서는 Event Argument Extraction (EAE) 작업을 위한 Compressive Memory 기반의 Retrieval (CMR) 메커니즘을 제안합니다. 기존의 Retrieval 기반 EAE 방법의 두 가지 주요 한계를 극복하여 다양한 정보 검색이 가능하게 합니다.

- **Technical Details**: CMR 메커니즘은 검색된 정보의 캐싱을 위한 동적 메모리 매트릭스를 사용합니다. 이를 통해 긴 입력의 제한을 극복하고, 입력 쿼리 기반으로 메모리에서 관련 정보를 검색하여 정보와 추론 모델 간의 간극을 메울 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 CMR 메커니즘은 RAMS, WikiEvents, ACE05의 세 가지 공개 데이터셋에서 기존의 Retrieval 기반 EAE 방법보다 유의미하게 성능이 향상되었습니다.



### ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models (https://arxiv.org/abs/2409.09318)
- **What's New**: 본 논문에서는 Multimodal Large Language Models (MLLMs)의 'hallucination'(환각) 문제를 다루며, 기존의 정적인 평가 방식에서 벗어나 동적인 프로토콜인 ODE(Open-Set Dynamic Evaluation)를 제안합니다. 이 프로토콜은 모델이 실제 개체를 얼마나 잘 인식하는지를 평가하기 위해 그래프 구조를 사용하여 실세계 개념 간의 연관성을 모델링하고, 새로운 샘플을 생성합니다.

- **Technical Details**: ODE는 Open-set(열린 집합) 접근 방식을 기반으로 하며, 기존의 정적인 벤치마크에서 발생할 수 있는 데이터 오염(data contamination)을 피하는 데 중점을 두고 있습니다. 이는 다양한 개념 조합을 동적으로 생성하여 평가하는 방식으로, 모델이 특정 작업의 핵심 개념을 진정으로 이해하고 있는지를 검증합니다. ODE는 일반적 개념과 구체적 도메인에 대한 다양한 테스트 데이터를 생성하기 위해 네 단계의 프로세스를 포함합니다: 1) 그래프 구조로 실세계 시나리오 모델링, 2) 시각적 정보의 개념적 설계, 3) 이미지 생성 및 필터링, 4) 텍스트 템플릿 설계.

- **Performance Highlights**: 실험 결과 ODE로 생성된 샘플을 사용할 경우 MLLMs의 환각 발생률이 높아지며, 기존의 정적인 벤치마크보다 더 향상된 성능 평가를 보여줍니다. 각 모델의 환각 경향성을 분석한 결과, 다양한 분포 상황에서 모델의 한계와 능력 경계를 식별할 수 있었습니다. 이를 통해 ODE의 효과성과 포괄성을 검증하였습니다.



### Language Models "Grok" to Copy (https://arxiv.org/abs/2409.09281)
Comments:
          5 pages, 7 figures

- **What's New**: 이 연구는 언어 모델의 사전 훈련 동력을 조사하며, 모델의 텍스트 복사 능력이 기초적인 기술이라는 관점에서 실험을 통해 이 능력이 어떻게 발전하는지를 밝힙니다. 특히 Transformer 기반 언어 모델이 'grokking' 이라는 현상에 따라 복사 능력을 발전시킨다고 주장합니다.

- **Technical Details**: 연구팀은 12층 Llama 모델을 사용하여 400억 토큰(40 billion tokens)의 데이터를 가지고 훈련하였으며, 모델의 콘텍스트 복사 능력을 평가하기 위해 다양한 랜덤 토큰 하위 시퀀스를 입력으로 제공합니다. 이 과정에서 induction heads(복사를 담당하는 attention heads)가 어떻게 형성되는지, 그리고 이들이 얕은 층에서 깊은 층으로 이동하면서 발생하는지를 분석합니다.

- **Performance Highlights**: 연구 결과, 모델의 콘텍스트 복사 정확도는 훈련 손실이 안정화된 이후에도 갑자기 증가하며, 이를 통해 'grokking' 현상과의 관련성을 확인하였습니다. 또한, 레귤화 기법을 도입하여 콘텍스트 복사 능력을 향상시킬 수 있음을 보여주었습니다.



### An empirical evaluation of using ChatGPT to summarize disputes for recommending similar labor and employment cases in Chines (https://arxiv.org/abs/2409.09280)
Comments:
          14 pages, 5 figures, 2 tables, the 18th Int'l Workshop on Juris-Informatics (JURISIN 2024), associated with the 16th JSAI International Symposium on AI (JSAI-isAI 2024)

- **What's New**: 이번 연구에서는 노동 및 고용 소송의 유사 사례를 추천하기 위한 하이브리드 메커니즘을 제안합니다. 이 분류기는 법원에서 준비한 사례 간의 항목화된 분쟁을 기반으로 유사성을 결정합니다.

- **Technical Details**: 연구팀은 분쟁을 클러스터링하고, 분쟁 간의 코사인 유사성(cosine similarity)을 계산하여 이 결과를 분류 작업의 특성으로 사용합니다. 이전 시스템에서는 분쟁의 클러스터 정보만을 고려했으나, 이번 연구에서는 법원에서 준비한 정보를 대신하여 GPT-3.5와 GPT-4로 생성된 항목화된 분쟁을 사용했습니다.

- **Performance Highlights**: 실험 결과, GPT-4로 생성된 분쟁을 사용할 경우 더 나은 성과를 보였으며, ChatGPT가 생성한 분쟁을 사용할 때는 성능이 이전보다 떨어졌지만 결과는 만족할 만한 수준이었습니다. 향후 대형 언어 모델들이 실제로 유용해지기를 기대합니다.



### Analyzing Correlations Between Intrinsic and Extrinsic Bias Metrics of Static Word Embeddings With Their Measuring Biases Aligned (https://arxiv.org/abs/2409.09260)
- **What's New**: 이 논문에서는 정적 단어 임베딩(static word embeddings)의 내재적(bias) 및 외재적(bias) 편향 측정 지표들이 자연어 처리(NLP) 시스템의 편향된 행동을 예측할 수 있는 능력을 조사합니다. 이들은 이전 연구에서 발견된 상관관계의 부재가 실제인지 의문을 제기합니다.

- **Technical Details**: 연구진은 외재적 편향 지표의 데이터셋에서 특징 단어들(characteristic words)을 추출하고, 그 단어들과의 내재적 편향 지표 간의 상관관계를 분석하였습니다. 내재적 편향 지표는 WEAT와 RNSB를 활용하여 측정되었고, 외재적 편향 측정은 공통적인 NLP 시스템을 통해 이루어졌습니다.

- **Performance Highlights**: 일부 외재적 편향 지표와는 중간 내지 높은 상관관계가 관찰되었으나, 다른 지표와는 거의 상관관계가 발견되지 않았습니다. 이 결과는 특정 설정에서 내재적 편향 지표가 편향된 행동 예측에 신뢰할 수 있지만, 다른 설정에서는 신뢰할 수 없음을 시사합니다.



### NovAScore: A New Automated Metric for Evaluating Document Level Novelty (https://arxiv.org/abs/2409.09249)
- **What's New**: 이번 연구에서는 NovAScore(Novelty Evaluation in Atomicity Score)를 소개하며, 문서 수준에서의 새로움을 평가하기 위한 자동화된 지표를 제공하고 있습니다. NovAScore는 원자적(atomic) 정보의 새로움과 중요도를 집계하여 해석 가능성과 문서 새로움에 대한 세부 분석을 제공합니다.

- **Technical Details**: NovAScore는 문서의 원자적 콘텐츠 단위(Atomic Content Units, ACUs)를 분해한 후, 이를 기존 문서의 ACUBank와 비교하여 평가합니다. 이 과정은 문서의 세부 정보까지 정확하게 식별하여 해석 가능성을 높입니다. ACUBank는 역사적 문서에서 ACUs를 저장하고 유사도 검색을 통해 새로움을 평가합니다.

- **Performance Highlights**: 실험 결과, NovAScore는 사람의 새로움 판단과 강한 상관관계를 보이며, TAP-DLND 1.0 데이터셋에서 0.626 포인트-이진 상관관계와 내부 인간 주석 데이터셋에서 0.920 피어슨 상관관계를 달성했습니다.



### Autoregressive + Chain of Thought (CoT) $\simeq$ Recurrent: Recurrence's Role in Language Models and a Revist of Recurrent Transformer (https://arxiv.org/abs/2409.09239)
- **What's New**: 이 논문은 Transformer 아키텍처의 한계를 극복할 수 있는 'Chain of Thought'(CoT) 프롬프트의 효용성을 연구하고, CoT가 재발적 구조의 장점을 모방하여 모델의 추론 능력을 향상시키는 방법을 탐구합니다.

- **Technical Details**: Transformer 구조는 반복적 연결을 제거하여 병렬 학습을 가능하게 하지만, 이는 Chomsky의 계산 계층에서 하위 성능으로 평가될 수 있습니다. 이 연구에서는 CoT가 그러한 한계를 극복할 수 있는 방법을 제시하며, 실험 결과를 통해 CoT가 autoregression과 recurrence의 다리를 잇는 역할을 한다고 주장합니다.

- **Performance Highlights**: CoT 접근 방식을 활용하면 Transformer 기반 언어 모델의 성능이 유의미하게 향상되며 기본적인 계산 문제(예: 곱셈이나 개수 세기)를 해결하는 데 있어 이전에는 불가능했던 작업들을 수행할 수 있도록 돕습니다.



### Contextual Evaluation of Large Language Models for Classifying Tropical and Infectious Diseases (https://arxiv.org/abs/2409.09201)
- **What's New**: 이 논문에서는 열대 및 감염병에 대한 질문 응답에서 대형 언어 모델(LLMs)의 성능을 평가하기 위해 TRINDs 데이터셋을 확장하고 다양한 컨텍스트 요소(예: 인구통계학적 정보, 위치, 위험 요소)의 영향을 심층적으로 분석합니다. 또한 LLM의 응답을 개선하기 위해 In-context prompt tuning을 적용하며, TRINDs-LM이라는 연구 도구를 개발하여 LLM의 성능에 미치는 컨텍스트의 영향을 탐색합니다.

- **Technical Details**: 연구진은 원래 TRINDs 데이터셋을 11,719개의 질문으로 확장하였으며, 이 데이터셋은 여러 인구통계학적 요인, 장소, 성별, 인종 등의 다양한 변형이 포함되어 있습니다. Gemini Ultra와 MedLM Medium라는 두 가지 LLM을 기반 모델로 사용하여 성능을 비교하였으며, 5회의 반복 실험을 통해 정확성을 평가하였습니다. 또한, LLM-Consumer 쿼리를 생성하여 임상적 톤 대신 일반 소비자의 관점에서 질문을 재구성하였습니다.

- **Performance Highlights**: 실험 결과, 컨텍스트 정보(예: 인구통계학적 요인, 위치 등)는 LLM의 응답 정확도를 높이는 데 중요한 역할을 하며, LLMs는 보다 구체적인 정보와 위험 요소를 포함하는 쿼리에 대해 더 나은 성과를 보였습니다. 연구진은 LLM이 제공하는 결과와 인적 전문가의 기준 점수를 비교하여 LLM의 응답이 여전히 불완전할 수 있음을 보여주었습니다.



### Towards Precision Characterization of Communication Disorders using Models of Perceived Pragmatic Similarity (https://arxiv.org/abs/2409.09170)
Comments:
          submitted to IEEE ICASSP 2025

- **What's New**: 이 논문은 의사소통 장애를 가진 개인에 대한 진단과 치료에서 음성 기술의 적용 기회를 탐구하고 있습니다. 특히, 기존 연구에서 논의되지 않았던 조건의 다양성, 실용적 결핍의 역할, 그리고 한정된 데이터의 도전 과제를 어떻게 극복할 수 있는지 설명합니다.

- **Technical Details**: 저자들은 일반적인 모델을 제안하여 두 개의 발화를 입력으로 받아 실용적 유사성을 추정합니다. 이 모델은 사전 훈련된 HuBert 모델의 24번째 레이어에서 특징을 추출하고, 선택된 특징 벡터의 코사인 유사도를 사용하여 실용적 유사성을 예측합니다. 이 접근 방식은 기존의 의미적 유사성 모델과는 차별화된 실용적 유사성 모델을 제안합니다.

- **Performance Highlights**: 특정 발화의 실용적 유사성을 측정하기 위한 초기 결과에 따르면, 모델의 예측은 미국 영어 발화에 대한 인간의 평가와 0.74의 상관관계를 보이며, 이는 인간 평가자 간의 동의 수준에 가깝습니다. 다양한 활용 사례(예: 비정상 화자 감지, 비슷한 화자 찾기 등)를 통해 임상 치료와 데이터 활용의 개선 가능성을 제시합니다.



### RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieva (https://arxiv.org/abs/2409.10516)
Comments:
          16 pages

- **What's New**: 본 논문에서는 RetrievalAttention이라는 새로운 방법을 제안합니다. 이 방법은 긴 컨텍스트에서의 주의(just attention) 계산을 가속화하기 위해 훈련이 필요 없는 접근 방식을 사용합니다.

- **Technical Details**: RetrievalAttention은 동적 희소성(dymanic sparsity)을 활용하여 KV 벡터에 대한 근사 최근접 이웃 검색(Approximate Nearest Neighbor Search, ANNS) 인덱스를 구축하고, 토큰 생성 시 벡터 검색(vector search)을 통해 가장 관련 있는 키를 검색합니다. 또한, 이 기법은 쿼리 벡터와 키 벡터 간의 분포가 상이하다는 문제를 해결하기 위해 주의에 민감한 벡터 검색 알고리즘을 도입합니다.

- **Performance Highlights**: RetrievalAttention은 8B 매개변수를 가진 LLM에서 단일 NVIDIA RTX4090(24GB) GPU를 사용하여 128K 토큰을 서비스하는 데 16GB의 GPU 메모리만 필요하며, 0.188초에 하나의 토큰을 생성할 수 있습니다. 또한, 4090 GPU에서 128K 컨텍스트의 경우, 정확도를 유지하면서도 기존의 ANNS 인덱스와 전통적인 KNN 기반 방법 대비 각각 4.9배와 1.98배의 디코딩 지연(latency)을 감소시킵니다.



### An Efficient Self-Learning Framework For Interactive Spoken Dialog Systems (https://arxiv.org/abs/2409.10515)
Comments:
          Presented at ICML 2024

- **What's New**: 본 연구는 대화 시스템에서 사용할 수 있는 새로운 전반적인 프레임워크를 제안하여, 단일 발화(turn)에서 학습하는 것을 넘어, 다회 대화(contextual dialog)에서 사용자 피드백을 반영하여 적응할 수 있는 자동 음성 인식(ASR) 시스템을 개발합니다.

- **Technical Details**: 제안된 프레임워크는 학생-교사(student-teacher) 학습 기법과 컨텍스트 인식 대화 처리(context-aware dialog processing)에서의 발전을 기반으로 하며, 새로운 온라인 하드 네거티브 마이닝(online hard-negative mining) 알고리즘을 통해 대화에서의 피드백을 처리합니다. 이 과정에서 명시적 컨텍스트 신호와 암묵적 피드백 신호를 활용하여 다중 단계 교사 모델을 도입합니다.

- **Performance Highlights**: 실제 대화 시스템에서 기존 시스템에 비해 약 10%의 상대적 WER(word error rate) 개선을 보였으며, 공개된 합성 데이터에서는 최대 26%까지 개선되었습니다. 저자원(domain) 환경에서도 약 22.8%의 WERR(word error rate reduction)을 보여주는 성과를 달성했습니다.



### Causal Language Modeling Can Elicit Search and Reasoning Capabilities on Logic Puzzles (https://arxiv.org/abs/2409.10502)
Comments:
          26 pages

- **What's New**: 이 연구는 Transformer 모델이 복잡한 문제인 Sudoku 퍼즐을 해결할 수 있는지에 대한 새로운 통찰을 제공합니다. 특히, 이전의 논의에 이어 LLM(대형 언어 모델)이 내재적으로 강력한 추론 능력을 갖추고 있음을 보여줍니다.

- **Technical Details**: 연구에서는 Transformer 모델이 논리적 단계의 순서를 바탕으로 훈련되어야만 Sudoku 퍼즐을 효과적으로 해결할 수 있음을 발견했습니다. 연구는 가상의 Sudoku 퍼즐 190만 개를 사용해 훈련하고, 최종 모델은 94.21%의 퍼즐을 정확히 푸는 성과를 보였습니다. 추가적으로, Zebra 퍼즐에서도 92.04%의 정확도를 기록했습니다.

- **Performance Highlights**: Transformer 모델은 논리적 단계의 순서에 따라 훈련되었을 때 높은 성능을 보였습니다. 이 실험을 통해, LLM이 실제로 문제를 이해하고 해결하는 능력을 조명하며, 내재된 강력한 추론 엔진이 Transformer 가중치 내에 존재함을 시사합니다.



### Incorporating Classifier-Free Guidance in Diffusion Model-Based Recommendation (https://arxiv.org/abs/2409.10494)
Comments:
          8 pages

- **What's New**: 이 논문은 분산 기반 추천 시스템을 제안하며, 여기에는 분류기 없는 가이드(classifier-free guidance)가 통합되어 있습니다. 기존 추천 시스템들은 협업 필터링(collaborative filtering)이나 콘텐츠 기반 필터링(content-based filtering)과 같은 전통적인 방법에 의존하고 있습니다. 본 논문은 사용자가 상품을 탐색하고 평가할 때의 순서를 반영하여 추천 시스템에 분산(difussion)을 통합합니다.

- **Technical Details**: 분산 모델(Diffusion Models, DMs)은 자연과학의 확산( diffusion) 개념에 기반하여 Gaussian 노이즈를 이용해 이미지를 생성하는 새로운 종류의 생성 모델입니다. 본 연구에서는 DMs를 추천 시스템에 통합하고, 분류기 없는 가이드를 활용하여 생성 능력을 향상시킵니다. 이러한 접근 방식은 영화 추천, 상품 추천과 같은 다양한 추천 작업에서의 성능을 개선하는 데 기여했습니다.

- **Performance Highlights**: 본 연구의 결과는 여러 데이터셋에서 최신 추천 시스템(state-of-the-art) 대비 다양한 지표에서 성능 향상을 보여주었습니다. 특히, 데이터가 희소한 경우에 대해서도 더 나은 추천을 제공할 수 있는 잠재력을 나타냅니다.



### Meta-Whisper: Speech-Based Meta-ICL for ASR on Low-Resource Languages (https://arxiv.org/abs/2409.10429)
- **What's New**: 이번 연구에서는 Meta-Whisper라는 혁신적인 접근법을 제안하여 저자원 언어의 자동 음성 인식(ASR) 성능을 개선합니다. 이 방법은 Whisper 모델과 Meta In-Context Learning (Meta-ICL)을 활용하며, k-Nearest Neighbors (KNN) 알고리즘을 통해 샘플 선택을 수행합니다.

- **Technical Details**: Meta-Whisper는 저자원 언어의 ASR 작업을 수행하기 위해 최소한의 짝지어진 음성과 텍스트 예제를 사용하도록 Whisper를 조정합니다. Whisper는 8개의 공통 언어로 사전 훈련을 수행하여 Meta-ICL 기능을 갖추게 되며, KNN 샘플링 방법을 통해 더 효과적인 샘플링을 지원합니다. 이 프레임워크는 예제 기반 음성 분류 및 텍스트 예측을 제공합니다.

- **Performance Highlights**: ML-SUPERB 데이터세트에서의 실험 결과, Meta-Whisper는 원래의 Whisper 모델에 비해 저자원 언어의 Character Error Rate (CER)를 크게 감소시킨 것으로 나타났습니다. 이 접근법은 제한된 자원을 가진 언어의 ASR 시스템을 더욱 적응 가능하게 발전시키기 위한 유망한 솔루션을 제공합니다.



### Instigating Cooperation among LLM Agents Using Adaptive Information Modulation (https://arxiv.org/abs/2409.10372)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 에이전트를 인간 전략적 행동의 프록시로 결합하고, 강화 학습(Reinforcement Learning, RL)을 통해 이들 에이전트를 진화하는 전략적 상호작용에 참여시키는 새로운 프레임워크를 제안합니다. 결과적으로, 전통적인 에이전트 기반 시뮬레이션을 확장하여 LLM 에이전트의 전략적 의사결정 행동을 모델링하고, 정보 접근성을 조절하는 pro-social promoting RL agent(PPA)를 도입하여 사회 복지와 공동체 행동을 증진시킵니다.

- **Technical Details**: 이 프레임워크는 LLM 기반 에이전트를 사용하여 인간 상호작용을 위한 동적인 거버넌스 메커니즘을 생성하고, LLM과 인간 에이전트가 결합된 하이브리드 환경에서 공동체 행동을 향상시킵니다. RL 기반 거버넌스 에이전트는 전략적 행동에 대한 정보 접근 수준을 동적으로 조정함으로써 협력 수준을 높이며, 이를 통해 시스템의 자율성을 유지하고 게임 보상의 변화 없이 개입합니다. 이 방법론은 반복 게임, 특히 죄수의 딜레마(Prisoner’s Dilemma)에서 LLM 에이전트의 전략적 적응을 검증합니다.

- **Performance Highlights**: 이 논문에서는 강화 학습 에이전트가 정보를 동적으로 조절함으로써 협력률을 증가시키는 데 성공한다는 것을 보여줍니다. 또한, LLM 에이전트가 정보 접근의 변화에 적절하게 반응하며 행동을 조정할 수 있음을 입증하였고, 이는 기존의 정적 개입 방법과 비교하여 인상적인 결과를 낳았습니다. 그 결과, AI 매개 사회 역학 분야에 대한 중요한 통찰을 제공하고 있습니다.



### 2D or not 2D: How Does the Dimensionality of Gesture Representation Affect 3D Co-Speech Gesture Generation? (https://arxiv.org/abs/2409.10357)
- **What's New**: 본 연구는 2D 또는 3D의 관절 좌표를 훈련 데이터로 사용하여 발화-제스처 딥 생성 모델의 성능에 미치는 영향을 조사합니다. 특히, 생성된 2D 제스처를 3D로 변환하는 방법과 3D에서 직접 생성된 제스처 간의 품질을 비교합니다.

- **Technical Details**: 연구에서는 Convolutional Neural Network(CNN)를 활용하여 생성된 2D 제스처를 3D로 변환하는 과정을 거칩니다. 두 가지 모델, 즉 Denoising Diffusion Probabilistic Model(DDPM)과 Recurrent Neural Generative Model을 사용하여 발화에 따른 다양한 제스처를 생성합니다.

- **Performance Highlights**: 객관적 평가와 사용자 연구를 통해 2D에서 3D로 전환된 제스처와 직접 3D에서 생성된 제스처의 품질을 비교한 결과, 후자의 모델이 더 인간적인 움직임과 자연스러운 제스처를 생성하는 것으로 나타났습니다.



### ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework (https://arxiv.org/abs/2409.10289)
- **What's New**: ReflectDiffu라는 경량화된 포괄적 프레임워크를 소개하며, 감정 전염(emotion contagion)과 의도 예측(intent prediction)을 통합하여 의미 있는 공감적 반응 생성을 달성합니다.

- **Technical Details**: ReflectDiffu는 Emotion-Contagion Encoder와 Intent Exploring-Sampling-Correcting 메커니즘을 활용하여 감정적 의도를 정밀하게 조정하고, 대화의 맥락을 공감적으로 이해할 수 있게 돕습니다.

- **Performance Highlights**: ReflectDiffu는 자동 및 인간 평가에서 기존 모델들보다 향상된 유의미성(relevance), 제어 가능성(controllability), 정보성(informativeness)을 보이며, 최신 기술 수준에 도달한 성능을 보여줍니다.



### Fit and Prune: Fast and Training-free Visual Token Pruning for Multi-modal Large Language Models (https://arxiv.org/abs/2409.10197)
- **What's New**: 본 논문에서는 MLLMs(Multimodal Large Language Models)의 비주얼 토큰 프루닝(Token Pruning)을 위한 새로운 접근 방식인 FitPrune을 제안하였습니다. 이는 훈련 없이도 빠르게 프루닝 레시피를 생성할 수 있도록 합니다.

- **Technical Details**: FitPrune은 토큰 프루닝을 통계적 문제로 간주하고, 주의(Attention) 분포의 발산을 최소화하는 최적의 프루닝 계획을 찾는 것을 목표로 합니다. 이 방법은 소량의 추론 데이터의 주의 통계를 기반으로 신속하게 수행될 수 있습니다.

- **Performance Highlights**: FitPrune은 LLaVA-NEXT에 대해 54.9%의 FLOPs 감소를 달성하며, 정확도는 0.5%만 감소시켰고, 모든 VL 태스크에 대해 프루닝 레시피를 약 5분 만에 생성할 수 있었습니다.



### Quantile Regression for Distributional Reward Models in RLHF (https://arxiv.org/abs/2409.10164)
- **What's New**: 이 논문에서는 기존의 보상 모델이 단일 스칼라 값을 생성하는 방식의 한계를 극복하기 위한 방법으로 Quantile Reward Models (QRMs)을 제안합니다. QRM은 단일 보상 값 대신 보상 분포를 학습하며, 이를 통해 인간의 다양한 가치와 선호를 보다 정확하게 반영할 수 있습니다.

- **Technical Details**: QRM은 quantile regression을 활용하여 보상의 전체 형태에 대한 다중 모드 분포를 학습합니다. 이 방법은 레이블 노이즈와 상충되는 선호를 처리할 수 있는 유연성을 제공하며, 인간 피드백에서의 복잡성을 보다 잘 모델링합니다. 이 접근 방식을 통해 알 수 있는 것은 Llama-3 모델을 사용하여 8억 개의 매개변수를 가진 강화 학습 정책을 생성할 수 있다는 점입니다.

- **Performance Highlights**: 실험 결과, QRM은 RewardBench에서 전통적인 점 추정 모델보다 성능이 뛰어났습니다. 또한, 분포 기반의 보상 모델은 리스크에 주의하는 강화 학습에서 활용될 수 있으며, 극단적인 부정적 반응을 생성하는 것을 줄이는 정책을 개발하는 데 기여합니다.



### Trustworthiness in Retrieval-Augmented Generation Systems: A Survey (https://arxiv.org/abs/2409.10102)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 신뢰성을 향상시키기 위한 통합 프레임워크를 제안합니다. 이는 여섯 가지 주요 차원에서 RAG 시스템의 신뢰성을 평가하며, 이러한 차원에는 사실성 (factuality), 강건성 (robustness), 공정성 (fairness), 투명성 (transparency), 책임성 (accountability), 개인 정보 보호 (privacy)가 포함됩니다.

- **Technical Details**: 이 연구에서는 RAG 시스템의 신뢰성을 평가하기 위해 문헌 조사를 통해 기존 연구를 분석하고, 신뢰성을 기준으로 다양한 LLM을 평가하는 벤치마크를 구축합니다. 이를 통해 LLM의 다양한 모델이 실제 애플리케이션에서 신뢰성을 어떻게 발휘하는지를 평가합니다.

- **Performance Highlights**: 여섯 가지 신뢰성 차원에 대한 종합적인 평가를 통해, 다양한 독점적 (proprietary) 및 오픈소스 모델의 신뢰성 성능을 비교하고 향후 RAG 시스템 개발에 대한 실질적인 통찰을 제공합니다.



### Benchmarking Large Language Model Uncertainty for Prompt Optimization (https://arxiv.org/abs/2409.10044)
- **What's New**: 본 논문은 대형 언어 모델(LLM)의 prompt 최적화를 위한 불확실성 추정(uncertainty estimation) 측정의 필요성을 강조합니다. 새로운 벤치마크 데이터셋을 도입하여 Answer, Correctness, Aleatoric, Epistemic Uncertainty를 평가하며, 기존의 지표들이 Answer Uncertainty와 더 밀접하게 연관되어 있음을 보여줍니다.

- **Technical Details**: 논문에서는 multi-step reasoning에 대해 불확실성을 추정하기 위한 벤치마크 데이터셋을 제안합니다. 이 데이터셋은 LLM의 출력에서 큰 나무 구조의 추론 흔적(tree-structured reasoning traces)을 구성하여 다양한 불확실성 유형을 계산합니다. 불확실성 추정 프로세스는 1) Random Perturb, 2) Random Sampling, 3) Calculate Uncertainty의 세 단계로 구성됩니다.

- **Performance Highlights**: 우리는 벤치마크 데이터셋을 통해 각 불확실성 유형이 prompt 최적화 문제에 얼마나 잘 대응하는지를 평가하고, 기존 메트릭의 한계를 극복하여 더 나은 프롬프트 최적화를 통해 성능 향상을 달성할 가능성을 제시합니다.



### Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models (https://arxiv.org/abs/2409.09788)
Comments:
          20 pages, 13 figures

- **What's New**: 최근 비전-언어 모델(VLMs)의 성능 향상을 보여주는 연구에도 불구하고, 객체 크기와 거리의 정량적 추론 능력은 아직 충분히 탐구되지 않았다. 이번 연구에서는 정량적 공간 추론을 위해 설계된 271개의 질문이 포함된 수동 주석 기준인 Q-Spatial Bench를 소개한다.

- **Technical Details**: Q-Spatial Bench는 다섯 가지 범주로 구성된 질문들이 포함되어 있으며, 이는 정량적 공간 추론을 평가하기 위해 고안되었다. 분석 결과, 객체 간 거리 추론이 특히 도전적임을 밝혀냈으며, 일부 VLMs는 다른 모델에 비해 뛰어난 성능을 보였다. 저자들은 SpatialPrompt라는 제로샷 프롬프팅 기법을 개발하여 VLMs가 참조 객체를 시각적 단서로 사용하여 정량적 공간 질문에 응답하도록 유도하였다.

- **Performance Highlights**: Gemini 1.5 Pro, Gemini 1.5 Flash, 및 GPT-4V와 같은 모델은 SpatialPrompt를 사용함으로써 각각 40점, 20점, 30점 이상의 성공률 향상을 달성했다. 이는 데이터 추가, 모델 구조 수정, 또는 파인튜닝 없이 이루어진 개선이다.



### ELMI: Interactive and Intelligent Sign Language Translation of Lyrics for Song Signing (https://arxiv.org/abs/2409.09760)
Comments:
          18 pages excluding reference and appendix

- **What's New**: 이 논문은 ELMI라는 새로운 접근법을 소개하고 있습니다. ELMI는 사용자가 노래 가사를 수화로 번역하는 데 도움이 되는 웹 기반 도구로, 노래-signing의 접근성을 크게 향상시키고 있습니다.

- **Technical Details**: ELMI는 노래 가사를 라인별로 편집할 수 있게 해주며, 실시간으로 가사 하이라이트와 음악 비디오 클립을 함께 제공합니다. 또한 대규모 언어 모델(LLM) 기반의 인공지능과의 대화 기능을 통해 의미, 표기, 감정, 타이밍에 대해 논의할 수 있습니다.

- **Performance Highlights**: 연구 참가자들은 ELMI를 통해 번역 품질이 크게 향상되었다고 보고했으며, 대화 기능이 자신감과 독립성을 증가시켰다고 합니다. ELMI는 사용자에게 필요한 모든 자원을 한 곳에서 제공하여 의사 결정 과정을 간소화하며, 사용자는 ELMI를 자신의 작업 흐름에 통합하고자 하는 의사를 보였습니다.



### PersonaMark: Personalized LLM watermarking for model protection and user attribution (https://arxiv.org/abs/2409.09739)
Comments:
          Under review

- **What's New**: 이 논문은 LLM(대규모 언어 모델) 저작권 보호를 위한 개인화된 텍스트 워터마킹 방식인 PersonaMark를 제안합니다. 기존의 방식은 사용자별로 서로 다른 워터마크를 주입하는 필요성을 간과했지만, PersonaMark는 사용자의 ID에 기반하여 고유한 워터마크를 생성하고 주입합니다.

- **Technical Details**: PersonaMark는 문장을 구성하는 구조를 이용하여 텍스트에 개인화된 워터마크를 삽입하는 방식을채택합니다. 문장 구조는 의존 구문 분석기를 이용해 추출되어 해싱 함수로 처리되며, 이를 통해 각 사용자에 대해 고유한 watermark 주입을 가능하게 합니다. 이 과정은 처리속도가 빠르며, 여러 사용자를 효과적으로 관리할 수 있습니다.

- **Performance Highlights**: Gemma 2B 모델 시리즈에 대한 평가는 perplexity 지표에서 2.74를 기록하며 기존 방법인 KGW의 12.32와 비교하여 월등한 성과를 보였습니다. 또, 감정 값, 핵심 정보 유사성, 문장 가독성 점수에서도 워터마킹이 없는 모델과 유사한 결과를 달성하였습니다.



### Automatic Control With Human-Like Reasoning: Exploring Language Model Embodied Air Traffic Agents (https://arxiv.org/abs/2409.09717)
- **What's New**: 최근 언어 모델의 발전이 항공 교통 관제 분야에서 새로운 기회를 창출하고 있습니다. 본 논문은 언어 모델 기반 에이전트를 사용하여 인적 개입 없이 항공 교통 갈등을 해결하는 방법을 구체적으로 탐구합니다.

- **Technical Details**: 본 연구는 대규모 언어 모델, 에이전트가 시뮬레이터와 상호작용할 수 있는 도구 및 경험 라이브러리라는 새로운 개념을 구성 요소로 포함하고 있습니다. 경험 라이브러리는 벡터 데이터베이스로, 에이전트가 시뮬레이션 및 언어 모델과의 상호작용을 통해 학습한 지식을 저장합니다.

- **Performance Highlights**: 최고 성능을 보이는 구성은 총 120개의 긴급 갈등 시나리오 중 단 하나를 제외하고 모두 해결할 수 있었습니다. 에이전트는 교통 상황 및 갈등 해결 전략에 대해 인간 수준의 설명을 제공할 수 있습니다.



### ExploreSelf: Fostering User-driven Exploration and Reflection on Personal Challenges with Adaptive Guidance by Large Language Models (https://arxiv.org/abs/2409.09662)
Comments:
          17 pages excluding reference and appendix

- **What's New**: 이번 연구에서는 LLM(큰 언어 모델)을 활용한 새로운 애플리케이션 ExploreSelf를 제안하여 사용자가 자신의 감정을 돌아보고 정리하는 과정을 지원합니다. ExploreSelf는 사용자가 동적으로 생성된 질문을 통해 반성할 수 있도록 돕는 시스템입니다.

- **Technical Details**: ExploreSelf는 사용자가 자신의 초기 서사에서 관련 주제를 자유롭게 확장하고, 깊이 있는 반성을 위해 주제의 세부 사항을 탐구할 수 있도록 설계되었습니다. 시스템은 사용자의 글쓰기에 맞춤형으로 키워드와 의견을 제공하는 온디맨드(guidance) 지침을 제공합니다.

- **Performance Highlights**: 19명의 참가자들과 진행된 탐색 연구를 통해, 참가자들은 ExploreSelf의 가이드와 자유로운 탐색의 균형을 높이 평가하며, 이는 더 깊은 몰입과 통찰로 이어졌습니다. 참가자들은 개인적인 도전 과제를 탐색하면서 시스템의 적응형 기능이 그들의 통제감과 자율성을 향상시켰다고 보고했습니다.



### Towards Data-Centric RLHF: Simple Metrics for Preference Dataset Comparison (https://arxiv.org/abs/2409.09603)
Comments:
          Working Paper

- **What's New**: 언어 모델의 인간 선호를 맞추기 위한 목표는 이러한 선호를 드러내는 데이터의 수집을 요구합니다. 현재까지 데이터셋 품질을 측정하고 비교하는 기존 노력이 없던 가운데, 본 연구는 선호 데이터셋을 체계적으로 연구하고 다양한 비교축을 제시하여 데이터 중심의 정렬(data-centric alignment) 접근 방식을 제안합니다.

- **Technical Details**: 본 연구에서는 선호 데이터셋을 평가하기 위해 세 가지 관점을 제안합니다: 1) effective sample size, 2) noise invariance, 3) information content. 이러한 측정 방법을 통해 RLHF에서의 보상 모델 성능을 효율적으로 훈련하기 위한 데이터 수집을 지원합니다.

- **Performance Highlights**: 세 가지 직관적 관점을 통해 선호 데이터셋을 이해하는 데 기여함으로써, 새로운 데이터셋의 개발과 관련된 광범위한 응용에 기여합니다. 다양한 모델 크기에서의 ablation(소거법)을 통한 검증을 통해 이러한 측정과 보상 모델 성능 간의 연결 고리를 입증합니다.



### ValueCompass: A Framework of Fundamental Values for Human-AI Alignmen (https://arxiv.org/abs/2409.09586)
- **What's New**: 최신 연구에서는 AI 시스템의 인간 가치와의 정렬의 필요성을 강조하며, ValueCompass라는 프레임워크를 제안합니다. 이는 심리학 이론과 체계적 검토에 기반하여 인간-AI 정렬을 평가하는 도구입니다.

- **Technical Details**: ValueCompass는 12가지 동기적 가치 유형을 포함하며, 49개의 기본 가치를 체계적으로 코드화합니다. 이 프레임워크는 생성적 AI 모델과 인간 간의 가치의 정렬 정도를 측정하는 측정 도구인 'Value Form'을 제공합니다. 72명의 참가자와 5개의 최신 언어 모델(LMs)을 활용하여 4가지 시나리오에서 인간과 LMs의 가치 평가를 비교합니다.

- **Performance Highlights**: 연구의 결과, LMs는 인간과 상당한 가치 불일치를 보였으며, 'Choose Own Goals'와 같은 가치를 LMs는 지지했지만, 인간들은 대부분 이의에 동의하지 않았습니다. 이는 LMs가 윤리적 원칙보다 운영 효율성을 우선시할 수 있음을 나타냅니다. 이러한 불일치는 AI 개발의 윤리적 접근 방식을 재고할 필요성을 제기합니다.



### RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation (https://arxiv.org/abs/2409.09584)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 연구에서는 코드 생성을 위한 새로운 접근 방식인 RethinkMCTS를 소개합니다. RethinkMCTS는 Monte Carlo Tree Search(MCTS) 알고리즘을 활용하여 코드 생성 전에 사고 수준에서 검색을 수행하며, 코드 실행의 세세한 피드백을 통해 잘못된 생각을 정제하는 과정을 도입합니다.

- **Technical Details**: RethinkMCTS는 사고와 코드 간의 관계를 명시적으로 모델링하여 사고를 통한 코드 생성 전략을 탐색합니다. 특히, 코드 실행 피드백을 통합하여 검증 과정을 개선하고, 잘못된 생각을 정정하는 'rethink' 작업을 도입합니다. 또한, 두 가지 평가 방법을 적용하여 실행 중의 행동을 보다 효과적으로 평가합니다.

- **Performance Highlights**: 광범위한 실험을 통해 RethinkMCTS가 기존의 검색 기반 및 피드백 기반 코드 생성 기준을 초월함을 입증했습니다. HumanEval 데이터셋에서 GPT-3.5-turbo의 pass@1 성능을 70.12에서 89.02로, GPT-4o-mini는 87.20에서 94.51로 향상시켰습니다.



### Planning Transformer: Long-Horizon Offline Reinforcement Learning with Planning Tokens (https://arxiv.org/abs/2409.09513)
Comments:
          11 pages, 5 figures, Submitted to AAAI

- **What's New**: 본 논문에서는 Decision Transformer를 기반으로 한 새로운 Planning Tokens 개념을 도입하여, 오프라인 강화 학습에서 긴 시간 측면 문제를 효과적으로 해결합니다. 이는 기존의 다음 토큰 예측을 넘어, 에이전트의 장기적 미래 정보를 포함하고 있습니다.

- **Technical Details**: 우리는 Planning Transformer 모델을 제안하며, 이는 Dual-Timescale Token Prediction을 통해 고급 Planning Tokens을 사용하여 에이전트의 저수준 정책을 안내함으로써 성능을 향상시킵니다. 이 접근 방식은 오토회귀 모델의 누적 오류 문제를 해결하고, HRL의 신용 할당 문제를 극복합니다.

- **Performance Highlights**: 이 모델은 복잡한 D4RL 환경에서 기존의 오프라인 RL 방법들과 비교하여 우수한 성능을 보이며, 긴 시간 지향 작업에서도 새로운 최첨단 성능을 설정합니다. 또한, Planning Tokens는 에이전트의 정책 해석 가능성을 높입니다.



### Rethinking the Influence of Source Code on Test Case Generation (https://arxiv.org/abs/2409.09464)
Comments:
          23 pages

- **What's New**: 이 논문은 잘못된 소스 코드가 주어졌을 때 대형 언어 모델(LLMs)이 테스트 생성을 어떻게 잘못 인도하는지를 탐구합니다.

- **Technical Details**: 연구에서는 5개의 오픈 소스 및 6개의 클로즈드 소스 LLM을 사용하여 4개의 데이터 세트에서 성능을 평가하였으며, 테스트 케이스의 정확도(accuracy), 커버리지(coverage), 버그 감지 효과를 통해 그 효과성을 측정하였습니다.

- **Performance Highlights**: HumanEval 데이터 세트에서 LLMs는 올바른 코드가 주어졌을 때 80.45%의 테스트 정확도를 달성했으나, 잘못된 코드가 주어졌을 때는 57.12%로 감소하였고, APPS 데이터 세트에서는 올바른 코드로 39.85%의 버그를 감지한 반면, 잘못된 코드에서는 단 19.61%의 버그만 감지하였습니다.



### LLM-Powered Ensemble Learning for Paper Source Tracing: A GPU-Free Approach (https://arxiv.org/abs/2409.09383)
- **What's New**: 이번 논문에서는 KDD CUP 2024의 논문 출처 추적 대회에서 3위에 오른 성과를 보고합니다. 대부분의 팀들이 BERT 및 ChatGLM과 같은 프리 트레인(pre-trained)된 신경망 언어 모델을 조정하여 문제를 해결한 반면, 우리는 클로즈드 소스(closed-source) 대형 언어 모델(LLMs)을 사용하여 문제를 해결합니다.

- **Technical Details**: 우리의 접근법은 문서의 텍스트와 관련 정보를 활용하여 직접적인 추론(reasoning)을 수행하는 프리 트레인된 LLM을 사용합니다. 우리는 논문 메타데이터, 인용 통계, 레퍼런스 메타데이터 및 맥락 키워드를 포함하는 특성(feature)을 추출하여 성능을 향상시킵니다. 이를 통해 GPU가 없는 환경에서도 효과적으로 문제를 해결할 수 있습니다.

- **Performance Highlights**: 우리는 KDD CUP 2024에서 3위를 차지하였으며, 이 과정에서 GPU를 사용하지 않고도 경쟁력을 갖춘 결과를 달성한 유일한 팀이었습니다. 이는 제한된 자원 속에서도 높은 성능을 발휘할 수 있는 우리의 방법의 유효성을 보여줍니다.



### Overcoming linguistic barriers in code assistants: creating a QLoRA adapter to improve support for Russian-language code writing instructions (https://arxiv.org/abs/2409.09353)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문에서는 인기 있는 언어 모델 'zephyr-7b-beta'에 대한 어댑터 모델을 훈련하고 평가하는 방법이 소개되었습니다. 이 어댑터는 프로그래밍 및 러시아어 이해와 관련된 작업에서 기초 모델의 성능을 개선하기 위해 개발되었습니다.

- **Technical Details**: 어댑터는 프로그래밍과 관련된 질문-답변 쌍 및 러시아어로 된 코드 관련 텍스트를 포함한 크고 다양한 데이터 세트를 사용하여 훈련되었습니다. 적용된 훈련 방법론은 러시아어 지침에 따라 Python 코드를 이해하고 생성하는 데 있어 모델의 답변 품질을 향상시키는 것을 보장합니다.

- **Performance Highlights**: 어댑터 설치된 기초 모델의 성능을 다양한 메트릭을 사용하여 평가하였으며, 이를 통해 기초 모델 및 이 분야의 최신 모델들과 비교하였습니다. 얻어진 결과는 Python 코드 작성 및 러시아어 처리와 관련된 작업에서 상당한 개선을 보여주었으며, 제안된 어댑터의 효과성을 확인하였습니다.



### Guiding Vision-Language Model Selection for Visual Question-Answering Across Tasks, Domains, and Knowledge Types (https://arxiv.org/abs/2409.09269)
Comments:
          8 pages + references + 6 pages of Appendix

- **What's New**: 이 논문은 Visual Question-Answering (VQA) 작업을 위한 Vision-Language Models (VLMs)을 평가하는 포괄적인 프레임워크를 제시합니다. 또한, 다양한 작업 유형 및 응용 도메인과 함께 지식 유형으로 주석이 달린 새로운 데이터셋을 소개하며, 이를 통해 특정 작업 요구사항에 따라 VLMs를 선택할 수 있는 방법을 제공합니다.

- **Technical Details**: 논문에서는 VQA 작업의 평가를 위한 표준화된 프레임워크를 개발하며, task type, application domain, knowledge type의 세 가지 측면에서 VLMs를 비교합니다. 이를 위해 GoEval이라는 다중 양식(multi-modal) 평가 지표를 도입하고, 이는 GPT-4o를 기반으로 하여 사람의 평가와 높은 상관관계를 보입니다. 실험에서는 8개의 다양한 VLM 변종을 평가하여 각각의 강점과 약점을 분석합니다.

- **Performance Highlights**: 엑실리에서의 실험 결과는 Gemini-1.5-Pro 및 GPT-4o-mini와 같은 독점 모델이 일반적으로 다른 모델보다 우수한 성능을 보이지만, InternVL-2-8B 및 CogVLM-2-Llama-3-19B와 같은 오픈소스 모델도 특정 맥락에서 경쟁력을 갖추고 있다는 것을 보여줍니다. VLM들의 성능은 서로 높은 상관관계가 없으며, 각 모델은 특정 카테고리에서는 잘 수행하지만 다른 카테고리에서는 어려움을 겪는 경향이 있습니다.



### What Is Wrong with My Model? Identifying Systematic Problems with Semantic Data Slicing (https://arxiv.org/abs/2409.09261)
- **What's New**: 새로운 프레임워크 SemSlicer는 기계학습 모델의 오류를 분석하고 체계적인 문제를 파악하는 데 도움을 줍니다. 이는 기존 특징 없이도 의미론적 데이터 슬라이싱을 가능하게 하며, 사용자 정의 슬라이싱 기준에 따라 적합한 데이터 조각을 생성합니다.

- **Technical Details**: SemSlicer는 Large Language Models (LLMs)를 사용하여 데이터셋을 주석화하고 사용자 정의 슬라이싱 기준으로부터 데이터 슬라이스를 생성하는 기능을 갖추고 있습니다. 사용자는 슬라이스 기준, 데이터셋, 슬라이싱 구성 옵션을 제공하기만 하면 됩니다. SemSlicer는 이를 기반으로 슬라이싱 프롬프트를 자동 생성하고 최적화합니다.

- **Performance Highlights**: SemSlicer는 다양한 슬라이싱 기준에 대해 정확한 슬라이싱 기능을 생성할 수 있으며, 비용과 정확성 간의 유연한 타협이 가능합니다. 이를 통해 모델 평가, 버그 수정, 데이터 수집 등의 다양한 활동에 유용합니다.



### Unleash LLMs Potential for Recommendation by Coordinating Twin-Tower Dynamic Semantic Token Generator (https://arxiv.org/abs/2409.09253)
- **What's New**: 이번 연구에서는 Twin-Tower Dynamic Semantic Recommender (TTDS)를 제안하여 기존의 정적 인덱스 기반 추천 시스템의 한계를 극복하고자 합니다. TTDS는 동적 의미 인덱스 체계를 채택하여 시맨틱(semantic) 및 협업(collaborative) 지식 간의 정렬 문제를 해결하고, 사용자-아이템의 고차 상호작용 패턴을 활용합니다.

- **Technical Details**: TTDS는 항시 최적화되는 쌍둥이 타워 기반의 의미 토큰 생성기를 포함하며, 사용자를 위한 의미 인덱스와 아이템을 위한 의미 인덱스를 계층적으로 할당합니다. 이 시스템은 사전 학습된 대규모 언어 모델(LLM)을 기반으로 하여 사용자 행동의 역사적 데이터를 반영합니다. 또한, 다중 그레인 정렬(multi-grained alignment)을 용이하게 하도록 설계된 이중 변량 오토인코더(dual-modality variational auto-encoder)를 제안합니다.

- **Performance Highlights**: TTDS는 세 가지 공개 데이터 세트에서 실험을 수행한 결과, 추천 성능에서 Hit-Rate는 평균 19.41%, NDCG는 20.84% 향상되었습니다. 이는 기존의 가장 최신 벤치마크 방법들에 비해 큰 개선 효과를 보여줍니다.



### Robust Training of Neural Networks at Arbitrary Precision and Sparsity (https://arxiv.org/abs/2409.09245)
- **What's New**: 본 연구에서는 정량화(quantization)와 희소성(sparsification)으로 인한 비연속적인 작동이 역전파(backpropagation)에 미치는 영향을 극복하기 위한 새로운 솔루션을 제안합니다. 이를 위해, 훈련 과정에서 정량화와 희소성을 섭동(perturbation)으로 설정하고, 접근 방법으로는 여 Ridge 회귀(ridge regression)를 기반으로 합니다.

- **Technical Details**: 제안하는 방법은 세 가지 기본 단계로 구성됩니다: 1) 정량화를 위한 Affine Transform: 입력 신호를 크기 조절하는 초기 변환; 2) 섭동 주입: 정량화의 효과를 모델링하기 위한 임의의 섭동 δ를 신호에 주입; 3) 복원을 위한 Denoising Affine Transform: 양자화 노이즈를 완화하며 원래 신호를 효과적으로 복원합니다. 이 방법은 기존의 모델 아키텍처 및 훈련 레시피와 호환되어 많은 변화 없이 적용 가능하며, 다양한 효율적인 신경망의 개발을 가능하게 합니다.

- **Performance Highlights**: 제안하는 방법은 초저정밀 모델에 대한 최신 결과를 달성하며, 기존 아키텍처와 표준 훈련 레시피를 사용하여도 우수한 성능을 발휘합니다. 또한 희소 신경망(sparse neural networks)과 시간 이진 신경망(temporal binary neural networks)의 성공적인 훈련을 통해 인공 신경망과 생물학적 신경망 간의 간극을 좁힐 수 있는 가능성을 보여줍니다.



### Multi-modal Speech Transformer Decoders: When Do Multiple Modalities Improve Accuracy? (https://arxiv.org/abs/2409.09221)
- **What's New**: 이 논문은 디코더 전용 이산 토큰 언어 모델이 자동 음성 인식(ASR)에서 다양한 모달리티가 성능에 미치는 영향을 체계적으로 분석하는 첫 번째 연구 중 하나입니다. 특히, 음성, 이미지, 입술 정보를 결합하여 ASR 정확도를 향상시키는 방법을 제시합니다.

- **Technical Details**: 논문에서 제안하는 모델은 디스크리트 멀티모달 언어 모델(DMLM)이며, OPT를 백본으로 삼고 다양한 입력 모달리티(음성, 이미지, 입술 움직임, OCR 텍스트)를 처리할 수 있습니다. 실험을 위해서는 3-Equations라는 합성 다중 모달 데이터셋이 사용되었습니다. 이 데이터셋은 기계적으로 생성된 수학 방정식과 함께 음성 샘플 및 입술 동영상 샘플로 구성되어 있습니다.

- **Performance Highlights**: 모달리티를 추가함으로써 ASR 정확도가 평균적으로 향상되는 것으로 나타났으며, 특히 보통의 잡음 수준에서 이미지가 보조 모달리티로 가장 큰 이점을 제공합니다. 또한, 가장 관련성이 높은 시각 정보를 사전 처리 단계에서 필터링했을 때 성능이 개선되었습니다.



### ReCLAP: Improving Zero Shot Audio Classification by Describing Sounds (https://arxiv.org/abs/2409.09213)
Comments:
          Code and Checkpoints: this https URL

- **What's New**: 본 논문에서 제안하는 ReCLAP 모델은 기존의 음성-텍스트 분류에 있어 단순하면서도 효과적인 방식을 통해 성능을 개선합니다. 기존의 추상적인 카테고리 레이블 대신, 독창적인 서술적 특성이 포함된 프롬프트를 사용하여 제로샷 오디오 분류의 정확도를 향상시킵니다.

- **Technical Details**: ReCLAP은 Liange Language Model (LLM)을 사용하여 오디오 캡션을 재작성하여 배경 소리에 대한 이해를 향상시키도록 훈련됩니다. 각 사운드 이벤트는 고유한 구별 특성을 기반으로 설명된 캡션으로 재구성되며, 이는 모델이 다양한 장면 내에서 사운드를 보다 잘 인식하게 만듭니다.

- **Performance Highlights**: ReCLAP은 멀티모달 오디오-텍스트 검색 및 제로샷 오디오 분류 모두에서 모든 기준선을 초과합니다. 제안된 방법은 ReCLAP의 성능을 1%-55% 향상시키며, 오디오 분류 및 검색 벤치마크에서 최고 성능을 보입니다.



### Transformer with Controlled Attention for Synchronous Motion Captioning (https://arxiv.org/abs/2409.09177)
- **What's New**: 이 논문에서는 사람의 움직임 시퀀스와 동기화된 언어 설명을 생성하는 어려운 작업인 동기화 모션 캡셔닝(synchronous motion captioning)에 대해 다룹니다. 이 방식은 수신어 수화 전사(aligned sign language transcription), 무감독 동작 세분화(unsupervised action segmentation), 시간 기반 텍스트 정렬(temporal grounding) 등 여러 응용 분야와 관련이 있습니다. 논문은 Transformer의 자기(Self-) 및 교차(Cross-) 주의(attention) 분포를 제어하기 위한 메커니즘을 도입하여, 해석 가능성(interpretability)과 시간에 맞춘 텍스트 생성을 수행합니다.

- **Technical Details**: 제안된 방법에서는 마스킹 전략과 구조화된 손실(structuring losses)을 활용하여 모델이 모션 단어 생성에 기여하는 가장 중요한 프레임에만 최대 주의(attention)를 두도록 유도합니다. 이러한 제약은 attention 맵에서 정보의 원치 않는 혼합(mixing)을 방지하고, 토큰 간의 단조(monotonic) attention 분포를 제공합니다. 따라서, 토큰의 교차 주의는 사람의 움직임 시퀀스와 동기화된 점진적(text generation) 텍스트 생성을 위해 사용됩니다.

- **Performance Highlights**: KIT-ML 및 HumanML3D의 두 개의 벤치마크 데이터셋에서 평가하여 제안된 방법의 우수한 성능을 입증합니다. 이 작업에 시각적 평가가 필수적이므로 코드 저장소에서 애니메이션 시각화의 포괄적인 세트를 제공합니다.



### DomURLs_BERT: Pre-trained BERT-based Model for Malicious Domains and URLs Detection and Classification (https://arxiv.org/abs/2409.09143)
- **What's New**: 이 논문에서는 의심스러운 도메인 및 URL을 감지하고 분류하기 위해 특화된 BERT 기반 인코더인 DomURLs_BERT를 소개합니다. 이 모델은 다국어 URL, 도메인 이름 및 DGA 데이터 셋을 사용하여 마스크드 언어 모델링(Masked Language Modeling, MLM) 목표로 사전 훈련되었습니다.

- **Technical Details**: DomURLs_BERT는 도메인 및 URL의 감지를 위한 독창적인 전처리 방법과 함께, SentencePiece 토크나이징을 사용하여 새롭게 훈련된 도메인 특화 토크나이저를 제공합니다. 이 모델은 DGA, DNS 터널링 기술, 악성 소프트웨어 분류 및 피싱/악성 URL 분류를 포함한 다양한 데이터셋에 대해 포괄적인 평가를 수행하였습니다.

- **Performance Highlights**: DomURLs_BERT는 기존의 6개 문자 기반 심층 학습 모델 및 4개 BERT 기반 모델에 비해 우수한 성능을 발휘하였으며, 다양한 분류 작업에서 성능이 입증되었습니다.



### Multimodal Fusion with LLMs for Engagement Prediction in Natural Conversation (https://arxiv.org/abs/2409.09135)
Comments:
          22 pages, first three authors equal contribution

- **What's New**: 이번 연구에서는 '스마트 안경'을 활용하여 쌍방향 대화에서 참여도를 예측하는 새로운 접근법을 제시합니다. 이 장치에는 비디오 카메라, 눈 카메라, 마이크로폰 등의 센서가 장착되어 있어 비언어적 행동을 자연적인 환경에서 분석할 수 있는 기회를 제공합니다.

- **Technical Details**: 34명의 참가자가 포함된 데이터셋을 수집하여 각 대화 후 자기 보고식 참여도 평가를 진행했습니다. 여러 행동 모달리티를 통합한 새로운 'multimodal transcript' 프레임워크를 통해 데이터 처리에 효과적이며, 최초 구현에서도 기존 융합 기술과 유사한 성능을 보여주었습니다. 이 프레임워크는 대화 참여도의 정황 추론을 가능하게 합니다.

- **Performance Highlights**: 제안한 방법론은 기존의 융합 기법들과 동등한 성능을 달성하는 것으로 나타났으며, 이는 향후 인간 행동 모델링 및 사회적으로 지능화된 기술 개발에 큰 잠재력을 지닙니다.



### AccentBox: Towards High-Fidelity Zero-Shot Accent Generation (https://arxiv.org/abs/2409.09098)
- **What's New**: 이번 연구에서는 새로운 Zero-Shot Accent Generation (ZAG) 기법을 제안하여 외국어 억양 변환(FAC), 억양 TTS, 그리고 Zero-Shot Text-to-Speech (ZS-TTS)의 기능을 통합했습니다. 전반적으로 보다 높은 억양 충실도(accent fidelity)를 달성하고 사전에 학습되지 않은 억양을 생성할 수 있는 가능성을 제공합니다.

- **Technical Details**: 본 연구에서는 두 단계의 파이프라인을 통해 작업을 수행합니다. 첫 번째 단계에서는 GenAID라는 AID 모델을 통해 0.56 f1 스코어를 달성하여 무시된 화자에 대한 억양 식별 성능을 극대화합니다. 두 번째 단계에서는 이렇게 추출한 억양 임베딩을 기반으로 YourTTS 기반의 ZS-TTS 시스템을 조건화하는 AccentBox를 도입합니다. 이 시스템은 연속적이고 speaker-agnostic(화자 비구속) 억양 임베딩을 활용하여 높은 품질의 제로샷 억양 생성을 수행합니다.

- **Performance Highlights**: 제안된 시스템은 고유 및 교차 억양 생성에 있어 57.4%에서 70.0%의 억양 유사도 선호도를 보여주며, 이는 기존의 강력한 기준 모델과 비교했을 때 두드러진 성과입니다. GenAID의 AID에서는 SOTA 결과를 보여주며, 13가지 억양 분류에서 무시된 화자에 대해 0.56의 f1 점수를 기록합니다.



### An Evaluation of GPT-4V for Transcribing the Urban Renewal Hand-Written Collection (https://arxiv.org/abs/2409.09090)
Comments:
          Published in Digital Humanities (DH 2024). Aug 6-9. Arlington, VA

- **What's New**: 이번 연구에서는 1960년대에서 1980년대에 걸쳐 도시 재개발과 관련된 수많은 기록을 디지털화하기 위해 OpenAI의 GPT-4V를 활용했습니다. 특히, 손으로 작성된 문서를 대규모로 효율적으로 전사하고 분석할 수 있는 가능성을 제시합니다.

- **Technical Details**: 이 연구는 Asheville의 주택당국이 작성한 도시 재개발 문서 5050개의 커버 페이지를 대상으로 하여 GPT-4V를 사용하여 텍스트 전사 과정을 실행했습니다. 사용자 친화적인 프롬프트를 통해 프로그램을 제어하고 정보 필드를 특정 JSON 형식으로 정리했습니다.

- **Performance Highlights**: GPT-4V는 문서 읽기에 약 19.11분이 소요되었고, 'ward' 필드는 0.80의 가장 높은 정확도를 기록했습니다. 반면, 'owner' 필드는 0.18로 가장 낮은 정확도를 보였습니다. 이 연구는 LLM의 성능 차이를 분석하고, 향후 다양한 필드에서의 정확도 향상을 위한 방향성도 제시합니다.



### United in Diversity? Contextual Biases in LLM-Based Predictions of the 2024 European Parliament Elections (https://arxiv.org/abs/2409.09045)
- **What's New**: 본 연구는 대형 언어 모델(LLM)을 활용하여 2024년 유럽 의회 선거의 투표 행동을 예측하며, LLM의 사회적 편향과 제한된 효용성을 강조합니다.

- **Technical Details**: GPT-4-Turbo를 활용하여 익명화된 개인 수준의 배경 정보를 바탕으로 다양한 프롬프트 내용과 언어를 사용하여 LLM이 각 개인의 투표 행동을 예측하게 합니다. 예측한 결과를 실제 선거 결과와 비교 분석합니다.

- **Performance Highlights**: 1. LLM 기반의 미래 투표 행동 예측이 주로 실패함. 2. 예측 정확도가 국가 및 언어적 맥락에 따라 고르게 분포되지 않음. 3. LLM 예측을 개선하려면 개인에 대한 자세한 태도 정보 제공이 필요함.



### Acceptable Use Policies for Foundation Models (https://arxiv.org/abs/2409.09041)
Comments:
          10 pages, 2 figures, 2 tables

- **What's New**: 기초 모델(foundation model) 개발자들이 사용자에게 해로운 사용을 방지하기 위한 Acceptable Use Policies (AUPs)를 채택하고 있다는 점을 밝힌 논문입니다. 이 정책들은 특정 용도에 대한 사용 금지를 법적으로 규제하는 방안으로써, AI 생태계에서의 규제를 이해하는 데 중요한 시각을 제공합니다.

- **Technical Details**: 이 논문은 30명의 기초 모델 개발자들이 채택한 AUPs를 식별하고 분석하여, 그들이 포함한 사용 제한의 유형과 내용을 검토합니다. 각 개발자의 AUPs는 127개의 상이한 사용 제한을 포함하고 있으며, 이는 AI 공급망 전반에 걸쳐 이질성을 초래할 수 있습니다. 개발자들은 경쟁업체나 특정 산업이 그들의 모델을 사용하는 것을 방지하기 위한 정책을 적용합니다.

- **Performance Highlights**: 일반적으로 AUPs는 사용자가 기초 모델을 통해 생성할 수 있는 콘텐츠의 유형을 제한하여 해를 끼칠 수 있는 행위를 방지하는 목적을 가지고 있습니다. 그러나 AUPs의 실시와 집행은 쉽지 않으며, 엄격한 집행은 연구자들의 접근을 제한하고 이로운 사용을 제한할 수 있습니다. 그럼에도 불구하고 AUPs는 기초 모델 시장과 AI 생태계에 상당한 영향을 미치는 초기 자율 규제(self-regulation)의 예로 평가됩니다.



### ChatSUMO: Large Language Model for Automating Traffic Scenario Generation in Simulation of Urban MObility (https://arxiv.org/abs/2409.09040)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 새로운 에이전트인 ChatSUMO를 소개합니다. ChatSUMO는 사용자의 언어 입력을 처리하여 도시 교통 시뮬레이터인 SUMO에서 시뮬레이션 시나리오를 생성할 수 있도록 돕습니다. 사용자는 전문 지식 없이도 단순한 텍스트 입력만으로 맞춤형 시나리오를 생성할 수 있습니다.

- **Technical Details**: ChatSUMO는 Llama 3.1 모델을 활용하여 구성된 다중 모듈 아키텍처를 가지고 있습니다. 입력 모듈은 사용자 입력을 처리하여 관련 키워드를 생성하고, 시뮬레이션 생성 모듈은 이 키워드를 사용하여 SUMO에서 교통 시나리오를 생성합니다. 사용자 맞춤화 모듈을 통해 교통 신호 최적화와 차량 경로 조정 등의 기능을 지원하며, 분석 모듈은 시뮬레이션 결과를 해석하여 상세한 보고서를 제공합니다.

- **Performance Highlights**: Albany 도시에 대한 실제 교통 시뮬레이션을 96%의 정확도로 생성할 수 있었으며, ChatSUMO를 통해 사용자들은 복잡한 설정 없이도 손쉽게 맞춤형 시나리오를 생성하고 지속적인 상호작용을 통해 다양한 시나리오를 만들어 낼 수 있습니다.



New uploads on arXiv(cs.IR)

### Incorporating Classifier-Free Guidance in Diffusion Model-Based Recommendation (https://arxiv.org/abs/2409.10494)
Comments:
          8 pages

- **What's New**: 이 논문은 분산 기반 추천 시스템을 제안하며, 여기에는 분류기 없는 가이드(classifier-free guidance)가 통합되어 있습니다. 기존 추천 시스템들은 협업 필터링(collaborative filtering)이나 콘텐츠 기반 필터링(content-based filtering)과 같은 전통적인 방법에 의존하고 있습니다. 본 논문은 사용자가 상품을 탐색하고 평가할 때의 순서를 반영하여 추천 시스템에 분산(difussion)을 통합합니다.

- **Technical Details**: 분산 모델(Diffusion Models, DMs)은 자연과학의 확산( diffusion) 개념에 기반하여 Gaussian 노이즈를 이용해 이미지를 생성하는 새로운 종류의 생성 모델입니다. 본 연구에서는 DMs를 추천 시스템에 통합하고, 분류기 없는 가이드를 활용하여 생성 능력을 향상시킵니다. 이러한 접근 방식은 영화 추천, 상품 추천과 같은 다양한 추천 작업에서의 성능을 개선하는 데 기여했습니다.

- **Performance Highlights**: 본 연구의 결과는 여러 데이터셋에서 최신 추천 시스템(state-of-the-art) 대비 다양한 지표에서 성능 향상을 보여주었습니다. 특히, 데이터가 희소한 경우에 대해서도 더 나은 추천을 제공할 수 있는 잠재력을 나타냅니다.



### Large Language Model Enhanced Hard Sample Identification for Denoising Recommendation (https://arxiv.org/abs/2409.10343)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)을 활용하여 추천 시스템의 하드 샘플을 구별하고 노이즈 샘플을 정화하는 새로운 프레임워크인 LLMHD를 제안합니다. 기존 방법들이 하드 샘플과 노이즈 샘플의 구별에 어려움을 겪고 있는 문제를 해결하고자 합니다.

- **Technical Details**: LLMHD 프레임워크는 세 가지 주요 모듈로 구성됩니다: Variance-based Sample Pruning(분산 기반 샘플 가지치기), LLM-based Sample Scoring(LLM 기반 샘플 점수화), Iterative Preference Updating(반복적 선호 업데이트). 처음에는 분산 기반 가지치기를 통해 하드 샘플 후보를 선택한 뒤, LLM을 통해 샘플의 하드니스(hardness)를 평가하여 최종적으로 하드 샘플을 식별합니다. 또한, 사용자 선호도를 개선하기 위한 반복적 업데이트 모듈을 포함하여, 잘못된 레이블을 수정합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋과 네 가지 백본 추천 시스템에서 폭넓은 실험을 진행한 결과, LLMHD는 뛰어난 성능과 강력한 노이즈 내성을 보였습니다. 실험 결과, LLMHD는 추천의 질을 크게 향상시키며, 노이즈로 인한 영향을 효과적으로 감소시킬 수 있음을 입증하였습니다.



### beeFormer: Bridging the Gap Between Semantic and Interaction Similarity in Recommender Systems (https://arxiv.org/abs/2409.10309)
Comments:
          Accepted to RecSys 2024

- **What's New**: 최근 제안된 beeFormer는 문장 Transformer 모델을 상호작용 데이터로 훈련시킬 수 있는 새로운 프레임워크입니다. 이 모델은 내용 기반 필터링과 협업 필터링의 장점이 통합되어 있습니다.

- **Technical Details**: beeFormer는 세 가지 기술(gradient checkpointing, gradient accumulation, 그리고 negative sampling)을 통해 높은 효과의 배치 사이즈를 유지하면서 Transformer 훈련 시 상호작용 데이터를 직접 사용하는 방식을 도입합니다. 이로 인해 문장 Transformer 모델이 고세밀한 상호작용 유사성을 포착할 수 있도록 합니다.

- **Performance Highlights**: beeFormer로 훈련된 모델은 콜드 스타트(cold-start), 제로샷(zero-shot), 그리고 시간 분할(time-split) 추천 시나리오에서 모든 기준 선을 초과하여 성능을 발휘했습니다. 또한, 다양한 도메인에서 훈련된 모델은 도메인 간 지식 전이 능력을 입증했습니다.



### Causal Discovery in Recommender Systems: Example and Discussion (https://arxiv.org/abs/2409.10271)
Comments:
          Accepted at the CONSEQUENCES '24 workshop, co-located with ACM RecSys '24

- **What's New**: 인공지능 및 머신러닝 커뮤니티에서 인과성(causality)에 대한 관심이 증가하고 있으며, 이 논문은 인과 그래프(causal graphs)를 이용한 추천 시스템(recommender system) 문제 모델링의 사례를 제시합니다.

- **Technical Details**: 이 논문에서는 KuaiRand라는 오픈 소스 데이터셋을 활용하여 관찰 데이터와 사전 지식을 결합하여 인과 그래프를 학습하는 과정을 설명합니다. 인과 추론(causal discovery) 과정은 총 다섯 단계로 구성되어 있으며, 관련 없는 특성의 제거, 특성 이산화(discretization), 사전 지식의 포함 방식 정의 등을 포함합니다.

- **Performance Highlights**: 학습된 인과 그래프는 분석된 피드백 신호에 효과적으로 영향을 미치는 변수가 적다는 것을 보여줍니다. 기존 머신러닝 커뮤니티의 대규모 모델 포함 경향과 대조적으로, 이 연구는 대부분의 변수는 결정에Noise를 추가하고 무관하다는 결과를 제시합니다.



### Enhancing Personalized Recipe Recommendation Through Multi-Class Classification (https://arxiv.org/abs/2409.10267)
- **What's New**: 이 논문은 다양한 요리 선호도에 맞춘 개인화된 레시피 추천의 어려움을 해결하고자 합니다. 이전의 추천 시스템에서 벗어나, 사용자의 다양한 요소를 감안한 혁신적인 접근법을 제시하고 있습니다.

- **Technical Details**: 레시피 추천에 있어서, 본 논문은 연관 분석(association analysis)과 분류(classification) 기법을 사용합니다. 연관 분석은 다양한 재료 간의 관계와 연결을 탐구하여 사용자 경험을 개선하고, 분류는 사용자가 정의한 재료와 선호도에 따라 레시피를 범주화합니다. 독특하게도, 여러 클래스를 포함하는 레시피와 재료를 고려하여 요리 조합의 복잡성을 인식합니다.

- **Performance Highlights**: 이 시스템은 요리 레시피의 분류 및 추천에 있어 정교한 접근 방식을 요구하며, 개인화된 추천을 정확하게 제공하기 위한 과정도 탐구합니다.



### Trustworthiness in Retrieval-Augmented Generation Systems: A Survey (https://arxiv.org/abs/2409.10102)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 신뢰성을 향상시키기 위한 통합 프레임워크를 제안합니다. 이는 여섯 가지 주요 차원에서 RAG 시스템의 신뢰성을 평가하며, 이러한 차원에는 사실성 (factuality), 강건성 (robustness), 공정성 (fairness), 투명성 (transparency), 책임성 (accountability), 개인 정보 보호 (privacy)가 포함됩니다.

- **Technical Details**: 이 연구에서는 RAG 시스템의 신뢰성을 평가하기 위해 문헌 조사를 통해 기존 연구를 분석하고, 신뢰성을 기준으로 다양한 LLM을 평가하는 벤치마크를 구축합니다. 이를 통해 LLM의 다양한 모델이 실제 애플리케이션에서 신뢰성을 어떻게 발휘하는지를 평가합니다.

- **Performance Highlights**: 여섯 가지 신뢰성 차원에 대한 종합적인 평가를 통해, 다양한 독점적 (proprietary) 및 오픈소스 모델의 신뢰성 성능을 비교하고 향후 RAG 시스템 개발에 대한 실질적인 통찰을 제공합니다.



### CROSS-JEM: Accurate and Efficient Cross-encoders for Short-text Ranking Tasks (https://arxiv.org/abs/2409.09795)
- **What's New**: 이 논문에서는 CROSS-Encoders with Joint Efficient Modeling (CROSS-JEM)이라는 새로운 랭킹 접근 방식을 제안합니다. 이는 transformer 기반 모델이 여러 아이템을 공동으로 점수 매길 수 있도록 하여 효율성을 극대화합니다.

- **Technical Details**: CROSS-JEM은 쿼리와 아이템 목록에 대한 listwise 랭킹을 모델링하며, 쿼리-아이템 및 아이템-아이템 상호작용을 동시에 캡처합니다. 이것은 token redundancy와 model logits를 사용하여 ranking probabilities를 추정하는 새로운 training objective인 Ranking Probability Loss (RPL)를 포함합니다.

- **Performance Highlights**: CROSS-JEM은 두 개의 공개 벤치마크 데이터셋에서 최소 3% 이상의 MRR 성능 향상을 보였고, 대규모 검색 기반 추천 시스템에서 기존 모델보다 13% 더 높은 정확도를 달성하면서도 표준 cross-encoders보다 6배 이상 빠른 성능을 나타냈습니다.



### Measuring Recency Bias In Sequential Recommendation Systems (https://arxiv.org/abs/2409.09722)
Comments:
          Accepted at the CONSEQUENCES '24 workshop, co-located with ACM RecSys '24

- **What's New**: 본 논문에서는 sequential recommendation system(순차 추천 시스템)에서 recency bias(최근성 편향)을 정량화할 수 있는 새로운 지표 HRLI(Hit Rate of the Last Item)를 제안합니다. 이 지표는 기존 모델들이 가지는 과도한 최근성 편향을 측정하고, 이로 인해 추천 성능이 저하되는 현상을 밝혀냅니다.

- **Technical Details**: 제안한 HRLI 지표는 사용자 세션에서 마지막 아이템의 추천 빈도를 측정합니다. HRLI는 모든 추천 모델에 적용할 수 있는 범용적인 지표로, 높은 HRLI 값은 추천 시스템이 사용자의 넓은 선호도를 효과적으로 반영하지 못하고 있다는 것을 나타냅니다. 실험에서 6개의 주요 sequential recommendation 모델에 대해 HRLI@10을 평가하였고, 결과는 HRLI가 Hit@10보다 항상 높다는 것을 보여주었습니다.

- **Performance Highlights**: HRLI가 높은 모델(SASRec, LightSANs, FEARec)은 성능 향상이 최대 43.02%에 달했으며, 반대로 HRLI가 낮은 모델(GRU4Rec, STAMP)은 성능 증대가 미비하였습니다. 이는 HRLI라는 지표가 추천 성능에 미치는 영향을 명확히 보여줍니다.



### Unleash LLMs Potential for Recommendation by Coordinating Twin-Tower Dynamic Semantic Token Generator (https://arxiv.org/abs/2409.09253)
- **What's New**: 이번 연구에서는 Twin-Tower Dynamic Semantic Recommender (TTDS)를 제안하여 기존의 정적 인덱스 기반 추천 시스템의 한계를 극복하고자 합니다. TTDS는 동적 의미 인덱스 체계를 채택하여 시맨틱(semantic) 및 협업(collaborative) 지식 간의 정렬 문제를 해결하고, 사용자-아이템의 고차 상호작용 패턴을 활용합니다.

- **Technical Details**: TTDS는 항시 최적화되는 쌍둥이 타워 기반의 의미 토큰 생성기를 포함하며, 사용자를 위한 의미 인덱스와 아이템을 위한 의미 인덱스를 계층적으로 할당합니다. 이 시스템은 사전 학습된 대규모 언어 모델(LLM)을 기반으로 하여 사용자 행동의 역사적 데이터를 반영합니다. 또한, 다중 그레인 정렬(multi-grained alignment)을 용이하게 하도록 설계된 이중 변량 오토인코더(dual-modality variational auto-encoder)를 제안합니다.

- **Performance Highlights**: TTDS는 세 가지 공개 데이터 세트에서 실험을 수행한 결과, 추천 성능에서 Hit-Rate는 평균 19.41%, NDCG는 20.84% 향상되었습니다. 이는 기존의 가장 최신 벤치마크 방법들에 비해 큰 개선 효과를 보여줍니다.



### HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented Generation System for AI Legal and Policy Applications (https://arxiv.org/abs/2409.09046)
Comments:
          Under review for the EMNLP 2024 Workshop on Customizable NLP: Progress and Challenges in Customizing NLP for a Domain, Application, Group, or Individual

- **What's New**: 본 논문은 AI 법률 및 정책 분야에 최적화된 Hybrid Parameter-Adaptive RAG (HyPA-RAG) 시스템을 제안합니다. 이 시스템은 질의 복잡성 분류기를 통해 동적으로 매개변수를 조정하고, 밀집, 희소, 지식 그래프 탐색을 결합한 하이브리드 검색 전략을 사용하며, 특정 질문 유형과 메트릭을 포함한 평가 프레임워크를 갖추고 있습니다.

- **Technical Details**: HyPA-RAG는 질의 복잡성 분류기를 활용하여 매개변수를 최적화하고, 밀집 및 희소 검색 방법과 지식 그래프 검색 방법을 결합하여 검증된 검색 정확도를 높입니다. 시스템 디자인은 초기 k 결과를 찾은 후, 정해진 매개변수 매핑에 따라 검색 결과를 정제합니다.

- **Performance Highlights**: NYC Local Law 144 (LL144)를 기반으로 한 테스트에서 HyPA-RAG는 정확성과 충실도를 크게 향상시켰으며, 복잡하고 높은 위험의 AI 법적 및 정책 응용 분야에 적합한 적응형 NLP 시스템의 필요성을 충족하였습니다.



### jina-embeddings-v3: Multilingual Embeddings With Task LoRA (https://arxiv.org/abs/2409.10173)
Comments:
          20 pages, pp11-13 references, pp14-20 appendix and experiment tables

- **What's New**: jina-embeddings-v3는 5억 7천만 개의 파라미터를 가진 최신 텍스트 임베딩 모델로, 다중 언어 데이터 및 긴 맥락 검색 작업에서 최첨단 성능을 발휘합니다. 이 모델은 최대 8192개의 토큰까지 지원하며, 쿼리-문서 검색, 클러스터링, 분류, 텍스트 매칭 등의 작업에 특화된 Low-Rank Adaptation (LoRA) 어댑터를 포함하고 있습니다.

- **Technical Details**: jina-embeddings-v3는 긴 텍스트 시퀀스를 효과적으로 인코딩하고 특정 작업에 맞춘 임베딩 생성을 가능하게 하는 구조로, XLM-RoBERTa 모델을 기반으로 합니다. FlashAttention 2와 DeepSpeed 프레임워크를 활용하여 효율적인 분산 훈련을 지원하고, Rotary Position Embeddings (RoPE)를 사용하여 절대 위치 임베딩을 대체해 긴 시퀀스를 처리합니다.

- **Performance Highlights**: MTEB 벤치마크에서 jina-embeddings-v3는 OpenAI 및 Cohere의 최신 상용 임베딩을 초월하여 영어 작업에서 뛰어난 성과를 보였고, 모든 다국어 작업에서 multilingual-e5-large-instruct을 능가했습니다. 또한 7.1억 개의 파라미터를 가진 LLM 기반 임베딩보다 비용 효율성이 높아 실제 생산 및 엣지 컴퓨팅에 더 적합합니다.



### Global Lightning-Ignited Wildfires Prediction and Climate Change Projections based on Explainable Machine Learning Models (https://arxiv.org/abs/2409.10046)
- **What's New**: 이 연구에서는 전세계의 번개에 의해 점화된 산불을 예측하고 특성화하기 위한 머신러닝 (Machine Learning, ML) 모델을 개발하였습니다. 기존의 모델들은 특정 지역에 맞춰져 있어 글로벌 적용에 제한적이었습니다. 연구는 번개 점화 산불과 인위적 산불을 분류하고, 기상 조건 및 식생과 같은 다양한 요인을 기반으로 점화 확률을 높은 정확도로 추정합니다.

- **Technical Details**: 모델은 eXplainable Artificial Intelligence (XAI) 프레임워크를 사용하여 다양한 특성이 모델에 미치는 영향을 분석합니다. 또한, 데이터셋은 번개 및 산불 활동의 글로벌 데이터 기반으로 설계되었으며, ML 모델은 90% 이상의 정확도를 달성했습니다. 연구는 데이터의 시계열 동안 평균적인 번개 점화 위험 증가를 추정하고, 미래 기후 조건에서의 모델 예측을 평가합니다.

- **Performance Highlights**: 모델은 2021년을 홀드 아웃 연도로 설정하여 일반화 능력을 테스트했으며, 인위적 산불과 번개 점화 산불의 차이를 강조합니다. 연구 결과, 기후 변화가 번개 점화 산불의 위험을 증가시키고 있으며, 특정 모델이 특정 유형의 산불에 대해 잘 작동하지 않다는 점을 발견했습니다.



### DiffATR: Diffusion-based Generative Modeling for Audio-Text Retrieva (https://arxiv.org/abs/2409.10025)
Comments:
          Accepted by Interspeech2024

- **What's New**: 이 논문은 오디오-텍스트 검색(Audio-Text Retrieval, ATR)에서 기존의 차별 모델(discriminative model)이 가진 한계를 극복하기 위해 생성적인 관점(generative perspective)에서 접근합니다. 이를 통해 오디오와 텍스트 간의 관계를 모델링하는 새로운 접근법인 DiffATR(디퓨전 기반 ATR 프레임워크)를 제안합니다.

- **Technical Details**: DiffATR는 노이즈로부터 점진적으로 결합 분포(joint distribution)를 생성하는 반복(iterative) 프로시저로 ATR 작업을 모델링합니다. 이 프레임워크는 생성 손실(generation loss) 및 대조 손실(contrastive loss)을 통해 훈련되며, Kullback-Leibler (KL) 발산을 활용하여 생성기를 최적화하고, NT-Xent 손실을 통해 특징 추출기(feature extractor)를 최적화합니다.

- **Performance Highlights**: AudioCaps 및 Clotho 데이터셋에서 실험을 통해 DiffATR는 우수한 성능을 보였으며, 특히 출처가 다른 데이터에 대한 검색(out-of-domain retrieval)에서 조정 없이도 강력한 성능을 지속적으로 나타냈습니다.



### Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search (https://arxiv.org/abs/2409.09913)
Comments:
          Preprint

- **What's New**: 이 연구에서는 RaBitQ 방법을 확장한 새로운 양자화 방법을 소개하며, 이 방법은 압축률을 낮추면서도 높은 정확도를 보장하는 데 중점을 둡니다.

- **Technical Details**: 새로운 방법은 RaBitQ의 이론적 보장을 상속받아 공간과 오류 경계 간의 상관 관계를 최적화합니다. 또한, 기존의 RaBitQ와 비교하여 압축률을 8x 및 4x로 조정할 수 있으며, 효율적인 거리 추정 계산이 가능하다.

- **Performance Highlights**: 실제 데이터셋에서의 광범위한 실험 결과, 제안된 방법은 기존의 최신 방법들에 비해 더 높은 정확도와 효율성을 보여 주었으며, 동일한 메모리 소모량에서 우수한 성능을 기록했습니다.



### Proximal Ranking Policy Optimization for Practical Safety in Counterfactual Learning to Rank (https://arxiv.org/abs/2409.09881)
Comments:
          Accepted at the CONSEQUENCES 2024 workshop, co-located with ACM RecSys 2024

- **What's New**: 본 논문은 기존의 안전한 카운터팩추얼 학습(maintain counterfactual learning to rank, CLTR) 접근 방식의 한계를 극복하는 새로운 방법인 근접 랭킹 정책 최적화(Proximal Ranking Policy Optimization, PRPO)를 제안합니다. PRPO는 사용자 행동에 대한 특정한 가정 없이도 배포 안전성을 보장합니다.

- **Technical Details**: PRPO는 기존 CLTR 방법론에서 나타나는 트러스트 바이어스(trust bias)를 처리할 수 없는 기존의 안전 기준을 대체하며, 안전한 랭킹 모델과 너무 비슷하지 않은 랭킹 행동을 학습하도록 유도하는 인센티브를 제거합니다. PRPO는 특정 사용자 가정에 의존하지 않으면서 배포 중 모델의 성능 메트릭 저하를 제한합니다.

- **Performance Highlights**: 실험 결과, PRPO는 기존의 안전한 인버스 포텐시 스코어링(inverse propensity scoring) 접근 방식보다 더 높은 성능을 제공하며, 극단적인 적대적 상황에서도 항상 안전성을 유지합니다. 이는 PRPO가 현실 세계에서의 안전성을 보장하는 첫 번째 방법이라는 점에서 중요합니다.



### AlpaPICO: Extraction of PICO Frames from Clinical Trial Documents Using LLMs (https://arxiv.org/abs/2409.09704)
Comments:
          Accepted at Methods

- **What's New**: 최근 증가하는 임상 시험 보고서로 인해 체계적인 리뷰 수행이 어려워지고 있으며, 본 연구에서는 자연어 처리(NLP) 기반의 자동 PICO 관련 용어 추출 방법을 제안합니다. 이는 기존의 수동 데이터 주석 작업을 대체하기 위한 시도로, 대규모 주석 데이터를 필요로 하지 않는 비지도 학습 설정에서 진행됩니다.

- **Technical Details**: PICO(Population, Intervention, Comparator, Outcome) 프레임 추출을 위해 본 연구는 In-Context Learning (ICL) 전략을 채택했으며, 기존 대형 언어 모델(LLM)에서 수집된 사전 훈련된 지식을 활용합니다. 또한, Low Rank Adaptation (LORA) 기법을 적용하여 자원이 제한된 환경에서도 대형 모델 훈련이 가능하도록 하였습니다.

- **Performance Highlights**: 실험 결과, ICL 기반 프레임워크는 EBM-NLP 데이터셋의 모든 버전에서 비교 가능한 결과를 보여주었으며, 지침 조정된(instruction-tuned) 버전은 EBM-NLP 데이터셋에서 최첨단(SOTA) 결과를 기록하였습니다.



### A Comparative Study on Enhancing Prediction in Social Network Advertisement through Data Augmentation (https://arxiv.org/abs/2404.13812)
Comments:
          Accepted by 2024 4th International Conference on Machine Learning and Intelligent Systems Engineering (MLISE)

- **What's New**: 이 연구에서는 사회망 광고 데이터를 위한 생성적 증강 프레임워크를 제시하고, 데이터 증강을 위해 Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), Gaussian Mixture Models (GMMs) 세 가지 모델을 탐구합니다. 이 연구의 핵심은 제한된 데이터셋으로 인해 발생하는 예측 모델의 성능 저하를 해결하는 것입니다.

- **Technical Details**: 이 연구는 GAN, VAE, GMM을 활용하여 새로운 데이터를 생성하고, 원본 데이터와 결합하여 모델을 훈련합니다. 총 400명의 사용자를 포함한 데이터셋을 준비하고, 각 모델에서 추가적인 200명을 생성하여 성능 향상을 평가합니다. 모델 성능 평가는 분류 정밀도, F1 점수 및 AUC 점수를 기준으로 합니다.

- **Performance Highlights**: GAN을 사용한 결정 트리 모델은 AUC 점수가 0.84에서 0.94로 향상되었으며, KNN 모델은 GMM 및 VAE의 개선 효과가 나타났습니다. 로지스틱 회귀는 정확도는 향상되지 않았지만 F1 점수가 상승했습니다. 전반적으로 모든 모델에서 데이터 증강은 예측 성능을 향상시키는 것으로 나타났습니다.



New uploads on arXiv(cs.CV)

### Do Pre-trained Vision-Language Models Encode Object States? (https://arxiv.org/abs/2409.10488)
- **What's New**: 이 연구에서는 웹 규모 데이터로 사전 훈련된 비전-언어 모델(VLM)들이 물체 상태 인코딩의 효과를 학습하는지를 조사하였습니다. 이를 통해 'ChangeIt-Frames'라는 객체 상태 인식 데이터셋을 통해 다양한 VLM을 평가하였으며, 신뢰성 있는 객체 인식은 가능하지만 물체의 물리적 상태를 정확히 식별하는 데에는 실패하고 있음을 발견하였습니다.

- **Technical Details**: ChangeIt-Frames 데이터셋은 다양한 물체 상태를 가진 이미지를 포함하고 있으며, VLM의 객체 중심 표현을 사용하여 색상과 형태에 대한 개념 결합 작업의 개선을 테스트합니다. VLM인 CLIP과 OpenCLIP은 이미지-텍스트 대비 학습을 적용하여 성능을 평가하며, 객체 정확도(Object Accuracy)와 상태 정확도(State Accuracy)를 별도로 산출하여 성능을 측정합니다.

- **Performance Highlights**: 객체 인식 정확도는 일반적으로 높으나, 상태 인식 정확도에서 약 30%의 일관된 감소가 관찰되었습니다. 물체 중심 VLM을 사용할 경우 성능 향상이 가능하나 여전히 인식 한계가 존재하며, FLAVA는 다른 VLM보다 저항력이 높아 도전적 상황에서도 상대적으로 잘 수행되었습니다.



### Exploring 3D Face Reconstruction and Fusion Methods for Face Verification: A Case-Study in Video Surveillanc (https://arxiv.org/abs/2409.10481)
Comments:
          Accepted at T-CAP - Towards a Complete Analysis of People: Fine-grained Understanding for Real-World Applications, workshop in conjunction with the 18th European Conference on Computer Vision ECCV 2024

- **What's New**: 이 연구에서는 서로 다른 3D 얼굴 재구성(3DFR) 알고리즘(EOS, 3DDFA v2, NextFace)을 활용하여 얼굴 인식 성능을 개선하고자 한다. 각 알고리즘이 개별적으로 분류기를 훈련시키는데 도움을 주고, 최종 결정은 점수 수준 융합(score-level fusion)을 통해 이루어진다.

- **Technical Details**: 3DFR 알고리즘의 성능은 카메라의 거리 및 인식하는 얼굴의 각도와 같은 다양한 환경 요인에 따라 달라질 수 있다. 본 연구에서는 여러 3DFR 알고리즘을 조합하여 비디오 감시 시나리오에서의 얼굴 인식 성능을 향상시키고, 다양한 점수 수준 융합 방법의 효과를 분석한다.

- **Performance Highlights**: 본 연구의 결과, 서로 다른 3DFR 알고리즘으로부터 얻은 보완 정보는 한 번도 본 적 없는 거리 및 카메라 특성에서 성능을 크게 향상시켰다는 것을 보여주었다. 이는 3DFR 기반 접근법에 대한 추가 연구를 촉진할 것으로 기대된다.



### SimInversion: A Simple Framework for Inversion-Based Text-to-Image Editing (https://arxiv.org/abs/2409.10476)
- **What's New**: 이 논문은 기존의 DDIM(Inversion) 반전 방법을 개선하기 위해 비대칭(guidance scale) 안내 범위를 수정하고, 효율성을 희생하지 않고 성능을 극대화하기 위한 제안을 하였습니다.

- **Technical Details**: 기존 DDIM 반전은 텍스트 조건에 최적화되지 않았으며, 누적된 오류가 성능에 부정적인 영향을 미치는 문제가 있었습니다. 이 연구에서는 소스 및 타겟 브랜치에서 안내 범위를 분리하여 오류를 줄이고, 이론적으로 더 나은 안내 범위를 도출했습니다.

- **Performance Highlights**: PIE-Bench 데이터셋에 대한 실험 결과, 제안한 방법(SimInversion)은 DDIM 반전의 성능을 크게 향상시켰으며, 모든 실험에서 일관된 개선을 보였습니다.



### MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion (https://arxiv.org/abs/2409.10473)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 연구에서는 Masked Conditional Diffusion (MacDiff)라는 새로운 프레임워크를 제안하여 인간의 스켈레톤 모델링을 수행합니다. 이는 첫 번째로 확산 모델(difussion models)을 효과적인 스켈레톤 표현 학습기(regresentation learners)로 활용하는 접근법을 제시합니다.

- **Technical Details**: MacDiff는 의미론적 인코더(semantic encoder)와 확산 디코더(diffusion decoder)로 구성되어 있으며, 인코더 입력에 랜덤 마스킹(random masking)을 적용하여 정보 병목(information bottleneck)을 도입하고 스켈레톤의 중복성을 제거합니다. 이 방법은 추상 대표성이 높아지고 상향식 과제에 대한 일반화 성능을 향상시키는 효과를 입증합니다.

- **Performance Highlights**: MacDiff는 NTU RGB+D 및 PKUMMD 데이터셋에서 최첨단 성능을 달성하며, 특히 제한된 라벨 데이터가 있는 상황에서 데이터 증강(data augmentation)을 통해 미세 조정(fine-tuning) 성능을 크게 향상시킵니다.



### Deep-Wide Learning Assistance for Insect Pest Classification (https://arxiv.org/abs/2409.10445)
- **What's New**: 본 연구에서는 농업에서 정확한 해충 인식의 중요성을 강조하며, 특히 인공지능을 활용한 새로운 해충 분류 시스템인 DeWi를 제안합니다. 이는 새로운 단계의 학습 전략을 포함하여, 여러 CNN에서 효과적으로 작동할 수 있습니다.

- **Technical Details**: DeWi는 triplet margin loss를 최적화하여 discriminative features(구별 가능 특성)를 학습하고, data augmentation(데이터 증강) 방법인 Mixup을 사용해 다양한 해충 카테고리에 대한 일반화 능력을 향상시킵니다. DeWi의 구조는 간단하면서도 효과적이며, 다양한 CNN 백본 아키텍처에 호환됩니다.

- **Performance Highlights**: 실험 결과, DeWi는 IP102 데이터셋에서 76.44%의 정확도와 D0 데이터셋에서 99.79%의 정확도로 두 개의 해충 분류 벤치마크에서 최고의 성능을 기록했습니다. 또한, 포괄적인 실험과 ablation study(소거 연구)를 통해 DeWi의 우수성을 입증했습니다.



### Learning Semi-Supervised Medical Image Segmentation from Spatial Registration (https://arxiv.org/abs/2409.10422)
- **What's New**: CCT-R은 의료 영상 세분화의 반지도 학습을 개선하기 위해 등록 정보(registration information)를 통합한 새로운 대조적 교차 교육(framework) 방법입니다.

- **Technical Details**: CCT-R는 두 가지 새로운 모듈인 registration supervision loss (RSL)과 registration-enhanced positive sampling (REPS)을 포함합니다. RSL은 레이블이 있는 볼륨과 레이블이 없는 볼륨 쌍 간의 변환에서 도출된 세분화 지식을 활용하여 추가적인 의사 레이블(pseudo-labels)을 제공합니다. REPS는 등록 변환을 사용하여 해부학적으로 대응하는 양성 샘플을 식별하여 대조 학습의 효과를 향상시킵니다.

- **Performance Highlights**: CCT-R은 ACDC 심장 MRI 분할에서 단일 레이블 사례만으로 Dice 계수(DSC)를 33.6% 향상시키고 Hausdorff 거리(HD)를 32.8mm 줄이며, Synapse 복부 CT에서 DSC를 21.3% 개선하고 HD를 58.1mm 감소시키는 성과를 보였습니다.



### Prompt-and-Transfer: Dynamic Class-aware Enhancement for Few-shot Segmentation (https://arxiv.org/abs/2409.10389)
- **What's New**: 이 논문은 사람의 시각적 인식 패턴을 모사하여 동적인 클래스 인식 프롬프트 패러다임인 'Prompt and Transfer' (PAT)를 제안합니다. 이 방법은 인코더가 현재 작업에서 관심 있는 객체(타겟 클래스)에 집중하도록 조정합니다.

- **Technical Details**: PAT는 세 가지 주요 요소를 통해 동적 클래스 인식 프롬프트를 개선합니다: 1) Cross-modal linguistic information을 활용하여 각 작업에 대한 프롬프트를 초기화합니다. 2) Semantic Prompt Transfer (SPT)를 사용하여 이미지 내 클래스 특정 의미를 프롬프트로 정밀하게 전달합니다. 3) Part Mask Generator (PMG)가 SPT와 함께 작동하여 개별에 대해 서로 다른 보완적인 부분 프롬프트를 생성합니다.

- **Performance Highlights**: PAT는 표준 FSS, Cross-domain FSS, Weak-label FSS, Zero-shot Segmentation을 포함한 4가지 다른 작업에서 경쟁력 있는 성능을 달성하며, 11개의 벤치마크에서 새로운 최첨단 성능을 기록합니다.



### Mamba-ST: State Space Model for Efficient Style Transfer (https://arxiv.org/abs/2409.10385)
- **What's New**: 본 연구에서는 스타일 전송 (style transfer)을 수행하기 위해 새로운 Mamba-ST 모델을 제안합니다. 이 모델은 기존의 transformer와 diffusion 기반 모델에 비해 메모리 사용량과 시간 복잡성을 효과적으로 낮추는 방식으로 설계되었습니다.

- **Technical Details**: Mamba-ST는 Mamba의 선형 방정식을 수정하여 크로스 어텐션(cross-attention) 레이어의 행동을 모방합니다. 이를 통해 두 개의 데이터 스트림을 입력으로 받아 결합할 수 있게 됩니다. Mamba-ST 디코더 (MSTD)는 스타일 정보와 내용 정보를 융합하여 스타일 전송을 수행합니다.

- **Performance Highlights**: 실험 결과 Mamba-ST는 ArtFID 및 FID 메트릭에서 기존 transformer 및 diffusion 모델에 비해 우수한 결과를 보였습니다. 이 방법은 스타일 전송에서 더 나은 품질과 효율성을 제공하며, 특히 낮은 메모리 요구량과 빠른 추론 시간 측면에서 향상을 이루었습니다.



### Robust image representations with counterfactual contrastive learning (https://arxiv.org/abs/2409.10365)
Comments:
          Code available at this https URL

- **What's New**: 이번 연구에서는 'counterfactual contrastive learning'이라는 새로운 방법론을 소개합니다. 이는 기존의 contrastive learning에서의 데이터 증강(data augmentation) 문제를 해결하기 위해 고안된 방식으로, 진정한 도메인 변화를 반영한 긍정적인 쌍을 생성합니다. 이 방법은 의료 이미징에서의 데이터 수집과 관련된 변형을 배제하면서도 의미론적 의미를 유지하는데 초점을 맞춥니다.

- **Technical Details**: 이 논문은 메디컬 이미징에서의 contrastive learning을 위한 새로운 데이터 쌍 생성 프레임워크를 제안합니다. 특히, Hierarchical Variational Autoencoder (HVAE) 모델을 사용하여 'counterfactual' 이미지 생성을 통해 실제 스캐너 간의 차이를 모사한 긍정적인 데이터 쌍을 생성합니다. 이에 따라 강력한 이미지 특성을 학습하도록 지원하여 도메인 변화에 대한 강건성을 증가시키는 것이 목표입니다.

- **Performance Highlights**: Counterfactual contrastive learning 방법론은 SimCLR과 DINO-v2와 같은 널리 사용되는 contrastive learning 프레임워크에서 5개의 데이터세트로 평가되었으며, 긍정적인 데이터 쌍 생성을 통해 도메인 변화에 대한 강건성을 높였음을 입증했습니다. 특히, 한정된 레이블을 가진 경우 및 학습 데이터에 과소 표현된 도메인에서 뛰어난 성능 향상을 보였습니다.



### Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning (https://arxiv.org/abs/2409.10362)
- **What's New**: 본 연구에서는 FOurier 변환 압축(Fourier transform compression)과 자기 지식 증류(self-knowledge distillation)를 통합한 새로운 Self-Supervised Learning(SSL) 접근 방식을 제안합니다. 이는 기존의 주파수 필터링 방식의 두 가지 제한점을 해결하며, 이미지의 주파수 반응에 따라 동적으로 마스킹된 주파수를 선택합니다.

- **Technical Details**: FOLK 프레임워크는 두 가지 주요 아이디어를 결합합니다. 첫째, 이미지 압축에서 영감을 받아, 각 이미지의 주파수 반응에 맞춰 마스킹된 주파수를 적응적으로 선택하여 pre-training을 위한 보다 적절한 SSL 작업을 만듭니다. 둘째, 두 개의 브랜치로 구성된 프레임워크를 사용하여 필터링된 이미지와 원본 이미지를 모두 입력으로 사용함으로써 다운스트림 작업의 부담을 크게 줄입니다.

- **Performance Highlights**: FOLK는 다양한 다운스트림 작업(예: 이미지 분류, 몇 샷 학습, 의미론적 분할)에서 최신 SSL 방법들과 경쟁력 있는 성능을 보임을 실험적으로 입증하였습니다.



### 2D or not 2D: How Does the Dimensionality of Gesture Representation Affect 3D Co-Speech Gesture Generation? (https://arxiv.org/abs/2409.10357)
- **What's New**: 본 연구는 2D 또는 3D의 관절 좌표를 훈련 데이터로 사용하여 발화-제스처 딥 생성 모델의 성능에 미치는 영향을 조사합니다. 특히, 생성된 2D 제스처를 3D로 변환하는 방법과 3D에서 직접 생성된 제스처 간의 품질을 비교합니다.

- **Technical Details**: 연구에서는 Convolutional Neural Network(CNN)를 활용하여 생성된 2D 제스처를 3D로 변환하는 과정을 거칩니다. 두 가지 모델, 즉 Denoising Diffusion Probabilistic Model(DDPM)과 Recurrent Neural Generative Model을 사용하여 발화에 따른 다양한 제스처를 생성합니다.

- **Performance Highlights**: 객관적 평가와 사용자 연구를 통해 2D에서 3D로 전환된 제스처와 직접 3D에서 생성된 제스처의 품질을 비교한 결과, 후자의 모델이 더 인간적인 움직임과 자연스러운 제스처를 생성하는 것으로 나타났습니다.



### Taming Diffusion Models for Image Restoration: A Review (https://arxiv.org/abs/2409.10353)
Comments:
          Review paper; any comments and suggestions are most welcome!

- **What's New**: 이 논문에서는 이미지 복원(image restoration) 작업을 위해서 최근 적용되고 있는 생성적 확산 모델(generative diffusion models)의 최신 기법들을 소개합니다. 특히, 이러한 모델들이 안정적인 훈련 과정과 현실적인 이미지 생성에서의 성공적인 성능 향상을 어떻게 이루는지를 다룹니다.

- **Technical Details**: 본 리뷰 논문은 생성적 확산 모델의 원리와 그에 따른 Image Restoration 적용 기법들을 설명합니다. 확산 모델의 기본 개념인 Denoising Diffusion Probabilistic Models (DDPM)와 Score-based Stochastic Differential Equations (Score-SDEs)에 대해 논의하며, 조건부 확산 모델(Conditional Diffusion Models, CDMs)을 사용한 이미지 생성 방법도 자세히 설명합니다. 이를 통해 이미지 복원 작업에 대한 다양한 관점을 제시합니다.

- **Performance Highlights**: Diffusion 모델은 현재 이미지 복원 작업에서 뛰어난 성능을 보여주고 있으며, 특정 조건 하에 훈련 없는 접근법을 제공하여 비블라인드(non-blind) IR 과제를 해결할 수 있는 가능성을 가지고 있습니다. 이 논문에서는 Diffusion 기반 IR 기법을 통해 높은 인식 및 사진 현실적인 결과를 얻는 방법을 종합적으로 정리하고 있으며, 관련된 도전과제와 미래의 연구 방향도 제시합니다.



### InfoDisent: Explainability of Image Classification Models by Information Disentanglemen (https://arxiv.org/abs/2409.10329)
- **What's New**: InfoDisent이라는 하이브리드 모델을 소개하며, 이미지 분류 네트워크의 결정 과정을 더 잘 이해할 수 있도록 돕는다.

- **Technical Details**: InfoDisent은 미리 훈련된 딥 네트워크의 최종 레이어에서 정보를 분리(disentangle)하여, 분류 결정 과정을 기본적이고 이해하기 쉬운 원자적 구성 요소로 나눈다. 이 모델은 post-hoc 방법과 intrinsic 방법의 장점을 결합한다.

- **Performance Highlights**: InfoDisent는 ImageNet, CUB-200-2011, Stanford Cars, Stanford Dogs와 같은 벤치마크 데이터셋에서 성능을 검증하였다.



### Fuse4Seg: Image-Level Fusion Based Multi-Modality Medical Image Segmentation (https://arxiv.org/abs/2409.10328)
- **What's New**: 이번 논문에서는 Multi-Modality Medical Image Segmentation (MM-MIS)을 위한 Fuse4Seg라는 새로운 이미지 수준 융합(Image-level fusion) 기반의 세분화(segmentation) 방법론을 제안합니다. 기존의 feature-level fusion 접근 방식이 가지는 의미적 불일치(semantic inconsistencies) 문제를 해결하기 위해 이미지 수준에서의 융합을 채택합니다.

- **Technical Details**: Fuse4Seg는 두 개의 모듈로 구성된 bi-level 학습(framework) 프레임워크로, 융합(fusion) 네트워크와 세분화 네트워크(segmentation network)로 이루어져 있습니다. 이 방법은 중간 계층에서의 feature을 결합하기 보다는 바로 이미지 등급으로 융합하여 세분화 결과를 향상시키며, 이중 최적화(bi-level optimization)을 통해 두 작업의 상호작용을 극대화합니다.

- **Performance Highlights**: Extensive 실험을 통해 Fuse4Seg는 기존의 최첨단(State-of-the-art, SOTA) 방법론에 비해 세분화 및 융합 성능을 모두 향상시켰으며, BraTS-Fuse 데이터셋에서 2,040 쌍의 원본 및 융합 이미지, 그리고 ground truth를 포함하여 의료 이미지 융합의 가장 큰 데이터셋으로 자리 잡았습니다.



### Baking Relightable NeRF for Real-time Direct/Indirect Illumination Rendering (https://arxiv.org/abs/2409.10327)
Comments:
          Under review

- **What's New**: 본 연구에서는 진정한 실시간 relighting을 위해 CNN 기반 렌더러와 해시 그리드 기반 렌더러를 결합한 새로운 방법을 제안합니다. 이 방법은 이전에 존재하지 않았던 간접 조명(indirect illumination)을 처리하는 baking 방법을 추가로 포함하고 있습니다.

- **Technical Details**: 제안된 접근법은 CNN을 사용하여 직접 조명에 필요한 표면 포인트와 렌더링 파라미터를 예측하고, 해시 그리드 기반의 경량 렌더러를 사용하여 간접 조명을 처리합니다. 이 두 렌더러는 사전 훈련된 모델에서 distillation을 통해 학습하여 실시간 물리 기반 렌더링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 직접 조명에서 최대 84.62배 빠른 렌더링 속도를 달성하였고, 평균 PSNR 손실이 0.62로 나타났습니다. 간접 조명을 포함한 경우에는 최대 85.4배 빠른 렌더링 속도를 기록하며 평균 PSNR 손실은 0.39였습니다. 전체적으로 SSIM 및 LPIPS 성능 지표에서도 교사 모델을 초과하거나 근접한 결과를 보였습니다.



### On Synthetic Texture Datasets: Challenges, Creation, and Curation (https://arxiv.org/abs/2409.10297)
- **What's New**: 이 논문은 고품질의 다양한 텍스처 이미지를 생성하기 위한 새로운 방법론을 소개하며, 362,880개의 텍스처 이미지를 포함하는 Prompted Textures Dataset (PTD)을 개발했습니다. 이 데이터셋은 56개의 텍스처 클래스를 포함하고 있으며, 기존의 텍스처 데이터셋들이 가진 한계를 극복합니다.

- **Technical Details**: 제시된 방법론에는 (1) 텍스트-투-이미지 모델에 입력할 설명자로부터 프롬프트를 개발하고, (2) Stable Diffusion 파이프라인을 채택 및 조정하여 해당 이미지를 생성 및 필터링하며, (3) CLIP 점수를 통해 최고 품질의 이미지로 추가 필터링하는 단계가 포함됩니다.

- **Performance Highlights**: 생성된 데이터셋은 Inception 및 FID 점수를 통해 품질과 다양성을 평가하였으며, 독립적인 인간 평가에서도 높은 품질을 보였습니다. 그러나 이미지 생성 과정에서 NSFW 안전 필터가 텍스처에 매우 민감하여 약 60%의 텍스처 이미지가 플래그를 받는 문제를 발견하였습니다.



### Anatomical Positional Embeddings (https://arxiv.org/abs/2409.10291)
- **What's New**: 이 논문에서는 3D 해부학적 위치 임베딩(Anatomical Positional Embeddings, APE)을 생성하는 자가 지도(self-supervised) 학습 모델을 제안합니다. APE는 의료 이미지를 구성하는 각 voxel의 해부학적 근접성을 인코딩하여, 같은 장기나 인접한 장기의 voxel은 더 가까운 위치 임베딩을 갖도록 합니다.

- **Technical Details**: APE 모델은 UNet-like 아키텍처를 활용하여 3D 위치 임베딩을 효율적으로 생성하며, 단일 volumetric 입력 이미지를 위해 voxel-wise 임베딩 맵을 생성할 수 있습니다. APE의 훈련 목표는 예측된 위치 임베딩 간의 거리와 해당 voxel 간의 물리적 거리가 밀접하게 연관되도록 만드는 것입니다.

- **Performance Highlights**: 모델은 기존의 해부학적 랜드마크 검색 및 약한 지도(few-shot) 장기 위치화 과제에서 뛰어난 성능을 발휘했으며, CT 이미지를 다양한 해부학적 관심 영역으로_crop할_수 있는 방법을 제공하며, 재현율(recall)은 0.99에 달하고 이미지 볼륨을 10-100배 감소시킬 수 있었습니다.



### Enhancing Image Classification in Small and Unbalanced Datasets through Synthetic Data Augmentation (https://arxiv.org/abs/2409.10286)
- **What's New**: 이 논문에서는 의료 이미지를 확인하기 위해 클래스 특화 변분 오토인코더(Variational Autoencoders, VAE)와 잠재 공간 보간(interpolation)을 활용하여 새로운 합성 데이터 증강 전략을 소개합니다. 이 방법은 데이터 부족과 클래스 불균형 문제를 해결하여 의료 이미지 분류의 정확성과 강건성을 향상시킵니다.

- **Technical Details**: 제안된 방법론은 클래스 내 잠재 표현의 보간(interpolation)에 의존하여 훈련 세트를 풍부하게 만들어 모델의 일반화 능력과 진단 정확성을 개선합니다. VAEs는 각 클래스에 대해 특화된 오토인코더를 통한 합성 이미지 생성을 가능하게 하여 데이터의 다양성과 표현력을 향상시킵니다.

- **Performance Highlights**: 작은 데이터 세트에서 실제 및 합성 데이터를 결합함으로써 가장 도전적인 저조한 클래스의 정확도가 18% 이상 증가했습니다. 전반적인 정확도와 정밀도에서도 각각 6% 향상을 보였습니다.



### Performance of Human Annotators in Object Detection and Segmentation of Remotely Sensed Data (https://arxiv.org/abs/2409.10272)
Comments:
          14 pages, 10 figures, 2 tables

- **What's New**: 본 연구는 인적 주석자(annotation)가 항공 이미지의 단면을 탐지 및 분할하는 작업에서 효과적으로 성능을 발휘하는 영향을 평가하는 실험을 도입했습니다. 인공태양광 패널을 케이스 스터디로 선정하고 다양한 주석 전략과 데이터 불균형이 성능에 미치는 영향을 검토하였습니다.

- **Technical Details**: 이 실험은 0.15m의 픽셀 크기를 가진 이미지를 사용하여, 전문가와 비전문가 모두가 참여한 가운데 진행되었습니다. 실험은 개별 주석자 설정과 그룹 주석자 설정, 타겟-배경 비율에 맞춘 다양한 전략을 비교하여 수행되었습니다. 연구 결과, 주석자들은 분할(segmentation) 작업보다 객체 탐지(object detection) 작업에서 일반적으로 더 효과적으로 수행하였고, Type II 오류(False Negatives)에 대한 경향이 두드러졌으며, 이는 일관된 해결 편향을 암시합니다.

- **Performance Highlights**: 객체의 부족한 분포로 인해 타겟-배경 비율이 낮은 영역에서는 성능이 감소했습니다. 전체적인 성능은 타겟-배경 비율이 높은 작업에서 우수하였으며, 과거 경험은 성과에 큰 영향을 미치지 않았습니다. 연구 결과는 원거리 감시 연구(annotation strategies for remote sensing research)의 향상을 도모하고 인적 주석자의 선택 및 안내에 기여할 수 있는 증거를 제공합니다.



### BAFNet: Bilateral Attention Fusion Network for Lightweight Semantic Segmentation of Urban Remote Sensing Images (https://arxiv.org/abs/2409.10269)
- **What's New**: 본 논문은 경량화된 양방향 세분화 네트워크인 BAFNet을 제안하여 고해상도 도시 원격 감지 이미지를 효율적으로 세분화할 수 있도록 발전시킵니다.

- **Technical Details**: BAFNet은 종속 경로(dependency path)와 원격-로컬 경로(remote-local path)로 구성되며, 대형 커널 주의 기법(large kernel attention)을 활용하여 이미지에서 장거리 의존성(long-range dependencies)을 취득합니다. 또한, 효율적인 원격 주의 모듈(efficient remote attention module)과 다중 스케일 로컬 주의 모듈(multi-scale local attention module)을 설계하여 원격-로컬 경로를 구성합니다.

- **Performance Highlights**: Vaihingen과 Potsdam 데이터셋에서 각각 83.20%와 86.53%의 mIoU 점수를 기록하며, 기존의 경량화 모델들과 비교하여 세분화 성능이 향상되었습니다. BAFNet은 대규모 Transformer를 활용한 고성능 세분화 네트워크와 비교했을 때, 정확도를 유지하면서도 부동 소수점 연산량을 10배 줄였습니다.



### Hydra-SGG: Hybrid Relation Assignment for One-stage Scene Graph Generation (https://arxiv.org/abs/2409.10262)
- **What's New**: 본 논문에서는 Hydra-SGG라는 혁신적인 Scene Graph Generation (SGG) 방법론을 제안합니다. 기존 DETR 기반 SGG 모델들이 직면한 두 가지 문제점인 희소한 감독(sparse supervision) 및 잘못된 음성 샘플(false negative samples)을 완화하기 위해 Hybrid Relation Assignment를 도입하였습니다. 이 방법은 하나의 진짜 관계가 여러 쿼리에 할당되도록 하여 훈련 샘플을 증가시킵니다.

- **Technical Details**: Hydra-SGG는 One-to-One Relation Assignment와 IoU 기반의 One-to-Many Relation Assignment를 결합한 Hybrid Relation Assignment를 사용합니다. 이를 통해 각 진짜 객체 관계는 높은 IoU를 가지는 여러 쿼리에 할당됩니다. 또한 self-attention 없이 작동하는 Hydra Branch를 도입하여 서로 다른 쿼리가 동일한 관계를 예측하도록 하여 중복된 관계 예측을 감소시킵니다.

- **Performance Highlights**: Hydra-SGG는 VG150 데이터셋에서 10.6 mR@20 및 16.0 mR@50의 결과를 달성하며, 12개의 훈련 에포크만으로 기존 성능을 초과하며 Open Images V6와 GQA에서도 새로운 최고 성능을 기록했습니다.



### SOLVR: Submap Oriented LiDAR-Visual Re-Localisation (https://arxiv.org/abs/2409.10247)
Comments:
          Submitted to ICRA2025

- **What's New**: 이 논문은 LiDAR-Visual 재위치를 위한 학습 기반의 통합 파이프라인 SOLVR을 제안합니다. SOLVR은 서로 다른 센서 모달리티 간에 장소 인식과 6 자유도(6-DoF) 변환을 수행합니다.

- **Technical Details**: SOLVR은 스테레오 이미지 스트림을 활용하여 메트릭 깊이 예측을 생성한 후, Probabilistic Occupancy 프레임워크를 사용하여 여러 장면 뷰를 융합합니다. 또한, 데이터셋 KITTI와 KITTI360에서 실험을 진행하여 기존 방법론과 비교해 뛰어난 성능을 입증했습니다.

- **Performance Highlights**: SOLVR은 LiDAR-Visual 장소 인식 및 등록에서 최첨단 성능을 달성하며, 쿼리와 검색된 장소 간의 큰 거리에서도 등록 정확도를 크게 향상시킵니다.



### Robust Bird's Eye View Segmentation by Adapting DINOv2 (https://arxiv.org/abs/2409.10228)
Comments:
          ECCV 2024 - 2nd Workshop on Vision-Centric Autonomous Driving (VCAD)

- **What's New**: DINOv2를 Low Rank Adaptation(LoRA) 방법으로 조정하여 Bird’s Eye View(BEV) 추정을 개선함과 동시에 기존 BEV 방법들이 다양한 왜곡에 대해 성능이 저하되는 문제를 해결하고자 하는 연구입니다.

- **Technical Details**: 본 연구에서는 최신 BEV 세분화 모델인 SimpleBEV에 DINOv2를 통합하며, LoRA를 사용하여 효율적으로 업데이트합니다. DINOv2의 강력한 표현 공간을 활용하여 모델의 정확도와 파라미터 효율성, 수렴 동작을 분석합니다.

- **Performance Highlights**: 실험 결과, 적대적인 조건 하에서도 BEV 인식의 강인성이 향상되었으며, 학습 가능한 파라미터 수는 감소하고 훈련 시간은 단축되었습니다.



### Neuromorphic Facial Analysis with Cross-Modal Supervision (https://arxiv.org/abs/2409.10213)
Comments:
          Accepted for publication at the ECCV 2024 workshop on Neuromorphic Vision: Advantages and Applications of Event Cameras (NEVI)

- **What's New**: FACEMORPHIC 데이터셋은 RGB 비디오와 이벤트 스트림의 시간 동기화된 멀티모달(face dataset)로, 얼굴의 Action Units를 분류하는 데 효과적으로 활용될 수 있습니다. 이 데이터셋은 64명의 사용자를 대상으로 4시간 이상의 녹화를 포함하며, 기존의 단순한 데이터에 비해 귀중한 정보를 제공합니다.

- **Technical Details**: FACEMORPHIC 데이터셋은 얼굴의 미세한 움직임을 캡처하기 위해 고안된 이벤트 카메라(이벤트 처리 카메라)를 사용합니다. 이 데이터셋은 시간 동기화가 가능하여 RGB 스트림으로부터 더 나은 감독 신호(supervision signal)를 발전시킬 수 있도록 합니다. 연구자들은 비디오 레벨 감독과 도메인 간 감독(cross-domain supervision)을 결합하여 Action Units을 분류하는 데 도움을 줄 수 있는 모델을 훈련합니다.

- **Performance Highlights**: FACEMORPHIC 데이터셋을 통해 기존의 RGB 비전 모델이 이벤트 기반 작업에 효과적으로 적용될 수 있음을 보여줍니다. 연구 결과, 3D Morphable Model의 계수를 통해 Action Unit 분류 아키텍처의 성능이 크게 향상된 것으로 나타났습니다.



### Garment Attribute Manipulation with Multi-level Attention (https://arxiv.org/abs/2409.10206)
Comments:
          Accepted for publication at the ECCV 2024 workshop FashionAI

- **What's New**: 본 논문에서는 다양한 의류 특성을 정밀하게 조작할 수 있는 GAMMA (Garment Attribute Manipulation with Multi-level Attention) 프레임워크를 제안합니다. 이는 다단계 주의 기반 아키텍처를 통합하여 사용자들이 높은 정확도로 패션 이미지 속성을 조정할 수 있게 합니다.

- **Technical Details**: GAMMA 프레임워크는 dual-encoder Transformer와 메모리 블록을 활용하여, disentangled representation을 기반으로 의류 속성을 조작합니다. 이를 통해 사용자가 원하는 특성에 맞춰 패션 아이템을 검색할 수 있도록 지원하며, 주의 기반 아키텍처를 사용할 수 있습니다.

- **Performance Highlights**: Shopping100k와 DeepFashion 데이터셋에서 최첨단 성능을 달성하였으며, 인터랙티브 이미지 검색 기능을 통해 사용자 만족도를 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Fit and Prune: Fast and Training-free Visual Token Pruning for Multi-modal Large Language Models (https://arxiv.org/abs/2409.10197)
- **What's New**: 본 논문에서는 MLLMs(Multimodal Large Language Models)의 비주얼 토큰 프루닝(Token Pruning)을 위한 새로운 접근 방식인 FitPrune을 제안하였습니다. 이는 훈련 없이도 빠르게 프루닝 레시피를 생성할 수 있도록 합니다.

- **Technical Details**: FitPrune은 토큰 프루닝을 통계적 문제로 간주하고, 주의(Attention) 분포의 발산을 최소화하는 최적의 프루닝 계획을 찾는 것을 목표로 합니다. 이 방법은 소량의 추론 데이터의 주의 통계를 기반으로 신속하게 수행될 수 있습니다.

- **Performance Highlights**: FitPrune은 LLaVA-NEXT에 대해 54.9%의 FLOPs 감소를 달성하며, 정확도는 0.5%만 감소시켰고, 모든 VL 태스크에 대해 프루닝 레시피를 약 5분 만에 생성할 수 있었습니다.



### RealDiff: Real-world 3D Shape Completion using Self-Supervised Diffusion Models (https://arxiv.org/abs/2409.10180)
- **What's New**: 본 논문에서는 RealDiff라는 새로운 자기 지도 학습 프레임워크를 제안합니다. 이 프레임워크는 3D 센서로 획득한 실제 측정을 기반으로 점 구름( point cloud) 완성을 조건부 생성 문제로 설정합니다.

- **Technical Details**: RealDiff는 다소 혼란스러운 관측을 처리하기 위해 추가적인 기하학적 단서를 활용합니다. 특히, 누락된 객체 부분에서 확산 프로세스를 시뮬레이션하며, 부분 입력을 조건으로 생성 과정을 제어합니다. 또한, 훈련 과정에서 객체 실루엣과 깊이 맵을 외부에서 추정한 것과 일치시키는 정규화를 적용합니다.

- **Performance Highlights**: 실험 결과, RealDiff는 실제 점 구름 완성 작업에서 최신 기법들을 지속적으로 능가함을 보여주었습니다. 기존 연구들과 비교했을 때, 제안된 방법은 종합적으로 우수한 성능을 입증하였습니다.



### ExelMap: Explainable Element-based HD-Map Change Detection and Upda (https://arxiv.org/abs/2409.10178)
Comments:
          17 pages, 3 figures

- **What's New**: 이 논문에서는 설명 가능한 요소 기반 HD 맵 변화 탐지 및 업데이트의 새로운 작업을 제안하고 있습니다. 이는 기존의 문제점을 해결하는 새로운 접근 방식으로, HD 맵 업데이트를 위해 최근의 온라인 매핑 기술을 구형 맵 정보를 활용하여 확장하고 있습니다.

- **Technical Details**: ExelMap이라는 새로운 설명 가능한 요소 기반 맵 업데이트 전략을 제시하며, 변화된 맵 요소를 명확히 식별합니다. 또한, 현재 사용되는 평가 지표들이 변화 탐지 성능을 충분히 포착하지 못하고 있다는 점을 강조합니다.

- **Performance Highlights**: 실제 보행자 횡단과 관련된 Argoverse 2 맵 변화 데이터셋을 통한 실험 연구를 수행하였으며, 이 연구가 실제 세계의 요소 기반 HD 맵 변화 탐지 및 업데이트의 포괄적인 문제 조사가 처음임을 강조합니다.



### VideoRun2D: Cost-Effective Markerless Motion Capture for Sprint Biomechanics (https://arxiv.org/abs/2409.10175)
Comments:
          Preprint of the paper presented to the Workshop on IAPR International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이번 연구에서는 MoveNet과 CoTracker라는 두 가지 일반적인 픽셀 및 신체 추적기를 현실적인 생체 역학 분석에 적합하도록 조정한 후, 수동 추적기(Kinovea)를 기준으로 이들을 평가하였습니다. 결과적으로, Sprint 생체 역학을 위해 특별히 조정된 최고의 마커리스 신체 추적기인 VideoRun2D를 제안합니다.

- **Technical Details**: VideoRun2D 시스템은 비디오 전처리, 관절 포인트 추적, 후처리, 생체 역학적 특징 생성을 포함하는 다섯 개의 처리 모듈로 구성되어 있습니다. MoveNet은 열 지도를 사용하여 인간 신체의 주요 포인트를 정확하게 로컬라이즈하는 반면, CoTracker는 트랜스포머 기반의 포인트 추적 모델로, 비디오 시퀀스에서 포인트 밀집 그리드를 추적합니다.

- **Performance Highlights**: MoveNet 기준으로 각 관절 각도가 3.2°에서 5.5° 사이의 오류로 정확하게 추정된 결과를 보인 반면, CoTracker는 수동 레이블링 접근법과 비교하여 큰 차이를 보였습니다. VideoRun2D는 스프린트 생체 역학 분석을 위한 유용한 도구로 보이나, 특히 고도의 정확도가 필요한 응용에는 여전히 부족함을 보입니다.



### Contrastive Learning for Character Detection in Ancient Greek Papyr (https://arxiv.org/abs/2409.10156)
- **What's New**: 이번 논문에서는 Greek letter recognition에 대한 SimCLR(대조 학습 기법)의 효과를 조사하였습니다. 다양한 augmentation 기법의 영향을 중점적으로 연구하였으며, SimCLR 백본을 Alpub 데이터셋(프리트레인 데이터셋)으로 사전 학습한 후, 작은 규모의 ICDAR 데이터셋(파인튜닝 데이터셋)에서 성능을 비교하였습니다.

- **Technical Details**: 연구에서는 세 가지 주요 접근 방식을 검토했습니다: (1) cross-entropy 손실을 사용하는 베이스라인 모델, (2) 분류 레이어를 갖춘 triplet embedding 모델, (3) 분류 레이어와 함께 사전 학습된 SimCLR 모델. ICDAR 데이터셋에서 ResNet-18 및 ResNet-50 네트워크를 사용하여 93개의 augmentation으로 학습했으며, 통계적 t-검정을 통해 상위 네 가지 augmentation을 선택했습니다. SimCLR의 사전 학습은 Alpub 데이터셋에서 진행되었고, 이후 ICDAR 데이터셋에서 파인튜닝이 이루어졌습니다.

- **Performance Highlights**: 실험 결과, SimCLR은 문자 인식 작업에서 베이스라인 모델을 초과하지 못했습니다. cross-entropy 손실을 사용하는 베이스라인 모델이 SimCLR 및 triplet 손실 모델보다 더 나은 성능을 보여주었습니다. 이 연구는 문자 인식을 위한 대조 학습의 한계를 강조하고, 이 작업에서 전통적인 감독 학습 모델의 강점을 부각시킵니다. 특히 SimCLR의 cropping 전략이 입력 이미지에서 의미적 변화를 초래하여 대규모 사전 학습 데이터셋에도 불구하고 훈련 효과를 감소시킬 수 있음을 제안합니다.



### AutoPET Challenge III: Testing the Robustness of Generalized Dice Focal Loss trained 3D Residual UNet for FDG and PSMA Lesion Segmentation from Whole-Body PET/CT Images (https://arxiv.org/abs/2409.10151)
Comments:
          11 pages, 5 figures, 2 tables

- **What's New**: 이번 연구에서 저자들은 PET/CT 스캔에서 암 병변의 자동 세분화(automated segmentation)를 위한 3D Residual UNet 모델을 사용하고 Generalized Dice Focal Loss 함수를 적용했습니다. 이를 통해 AutoPET Challenge 2024 데이터셋으로 모델을 훈련시켰습니다.

- **Technical Details**: 연구에 사용된 데이터는 900명의 환자로부터 수집된 1014개의 FDG 사례와 378명의 환자로부터 수집된 597개의 PSMA 사례로 구성되었습니다. 모델은 5겹 교차 검증(5-fold cross-validation)을 통해 훈련되었으며, 평균 앙상블 기법이 사용되었습니다. 네트워크 구조는 5개의 인코더와 디코더 레이어로 구성되며, Generalized Dice Loss와 Focal Loss를 결합한 Generalized Dice Focal Loss를 손실 함수로 활용했습니다.

- **Performance Highlights**: 초기 테스트 단계에서 평균 Dice Similarity Coefficient (DSC) 0.6687, 평균 거짓 음성 부피(FNV) 10.9522 ml, 평균 거짓 긍정 부피(FPV) 2.9684 ml를 달성했습니다.



### PSHuman: Photorealistic Single-view Human Reconstruction using Cross-Scale Diffusion (https://arxiv.org/abs/2409.10141)
- **What's New**: PSHuman은 새로운 멀티뷰 확산 모델을 활용하여 단일 RGB 이미지에서 고해상도 3D 인간 메시를 재구성하는 프레임워크입니다. 이 모델은 복잡한 의류 및 체형에 대한 도전 과제를 효과적으로 해결합니다.

- **Technical Details**: PSHuman은 다중 뷰 확산 모델의 사전 지식을 활용하여 인간 메시를 명시적으로 재구성하는 방법을 제안합니다. 본 연구에서는 SMPL-X 모델을 사용하여 바디 포즈를 조건화하고, 바디-페이스 크로스 스케일 확산 프레임워크를 도입하여 자세와 얼굴 디테일을 보존합니다.

- **Performance Highlights**: PSHuman은 CAPE와 THuman2.1 데이터 세트에서 지오메트리 세부사항, 텍스처 진실성 및 일반화 능력에서 우수한 성능을 보여주었습니다. 전체 재구성 과정은 빠르게 진행되며, 고품질의 텍스처가 있는 3D 인간 메시는 단 몇 분 안에 생성됩니다.



### A Comparative Study of Open Source Computer Vision Models for Application on Small Data: The Case of CFRP Tape Laying (https://arxiv.org/abs/2409.10104)
- **What's New**: 이 논문은 소규모 데이터 환경에서 인공지능(AI) 모델을 훈련시키기 위한 Transfer Learning의 활용 가능성을 탐구합니다. 특히, 항공우주 제조의 Carbon Fiber Reinforced Polymer (CFRP) 테이프 레이핑의 품질 관리 사례를 통해 AI 모델 개발에 필요한 최소 데이터 양을 조사합니다.

- **Technical Details**: 연구진은 컴퓨터 비전 모델들의 성능을 평가하기 위해 오픈 소스 사전 훈련 모델을 사용하고, 실험적으로 데이터 양을 지속적으로 줄여 가며 AI 모델 학습의 효율성을 분석합니다. Transfer Learning 방식을 통해 이미 학습된 모델을 소규모 데이터 세트에서 미세 조정하여 더 나은 성능을 얻을 수 있다는 점을 강조합니다.

- **Performance Highlights**: 결과에 따르면, AI 모델을 성공적으로 훈련시키는 데 필요한 데이터 양을 크게 줄일 수 있으며, 작은 모델을 사용하는 것이 반드시 성능 저하로 이어지지 않는다는 사실이 밝혀졌습니다. 이 방법은 여러 응용에 적용 가능하며, 특히 industrial manufacturing에 적합하다는 것을 보여줍니다.



### Adaptive Segmentation-Based Initialization for Steered Mixture of Experts Image Regression (https://arxiv.org/abs/2409.10101)
- **What's New**: 본 논문에서는 Steered-Mixture-of-Experts (SMoE)와 Radial-Basis-Function (RBF) 네트워크의 최적화를 위한 새로운 적응형 세분화 기반 초기화 방법을 제안합니다. 이 방법은 미리 계산된 이미지 세그먼트에 커널을 할당하며, 각 세그먼트에 대해 최적의 커널 수, 위치 및 파라미터를 유도합니다.

- **Technical Details**: 적응형 초기화 방법은 이미지 세그먼트를 기반으로 하여, 적응적인 커널 수를 할당하고, 이는 고주파 세부 묘사를 더 잘 포착하게 합니다. 초기화된 커널은 최적화 결과에 더 가깝게 위치하여 최적화 시간을 단축시키며 SMoE, RBF 및 관련 커널 이미지 회귀 방법에 적용 가능합니다.

- **Performance Highlights**: 제안된 초기화 방법은 기존의 정규 그리드 초기화 방법 및 K-Means 초기화 방법과 비교하여 객관적 및 주관적 품질에서 현저한 개선을 보여주며, 커널 수를 약 50% 감소시키고, 최종적으로 최대 50%의 실행 시간 절약을 달성합니다.



### Human Insights Driven Latent Space for Different Driving Perspectives: A Unified Encoder for Efficient Multi-Task Inferenc (https://arxiv.org/abs/2409.10095)
- **What's New**: 본 논문에서는 다중 컴퓨터 비전 작업을 통해 도시 내비게이션에 필요한 스티어링 앵글을 정확히 추정하기 위해 공유 인코더를 제안합니다. 이 인코더는 깊이(depth), 포즈(pose), 3D 장면 흐름(scene flow) 추정, 세그멘테이션 등의 다양한 시각 정보를 통합해 스티어링 각도를 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 다중 작업 학습을 위해 새로운 다중 스케일(feature network) 아키텍처를 도입하고, 사전 훈련된 멀티 백본(multi-backbone) 모델에서 지식을 증류(knowledge distillation)하며, 이로써 안정적인 학습과 성능 향상을 달성합니다. 또한, 공유 인코더는 다양한 시각 작업에 대한 전반적 인식 능력을 제공하여 스티어링 각도 추정에 도움이 됩니다.

- **Performance Highlights**: 본 논문의 결과는 기존 스타트(state-of-the-art) 방법과 비교했을 때 경쟁력 있는 성능을 보여주며, 다중 작업 학습을 통해 인간과 유사한 시각 인식을 통합함으로써 자율 주행 시스템의 발전 가능성을 제시합니다.



### DDoS: Diffusion Distribution Similarity for Out-of-Distribution Detection (https://arxiv.org/abs/2409.10094)
- **What's New**: 이번 논문은 Out-of-Distribution (OoD) 데이터 감지를 위한 새로운 확산 기반 탐지 프레임워크를 제안합니다. 기존의 방법이 저수준 이미지 패턴에 의존한 한계를 극복하기 위해, 정보 특징 공간과 확률 공간에서 배운 분포 유사성을 평가하는 새로운 유사성 지표를 도입합니다.

- **Technical Details**: 제안된 프레임워크는 Diffusion Distribution Similarity (DDoS)로 명명되며, 분류기(classifier-under-protection)가 두 이미지 간의 분포 유사성을 평가할 수 있도록 합니다. 이 방법은 비정상적 OoD 정보를 제거하는 전략을 포함하고 있어, 감지를 용이하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 DDoS 프레임워크가 기존의 신뢰하는 퍼셉션 기반 지표들보다 뛰어난 분포 유사성 평가 성능을 보이며, 최신 최첨단 탐지 성능을 달성함을 보여줍니다.



### MotionCom: Automatic and Motion-Aware Image Composition with LLM and Video Diffusion Prior (https://arxiv.org/abs/2409.10090)
- **What's New**: MotionCom은 훈련이 필요 없는 motion-aware (모션 인지) diffusion 기반 이미지 합성을 제안합니다. 이는 사용자 정의 개념을 바탕으로 하여 목표 객체를 새로운 장면에 자동으로 통합할 수 있게 합니다.

- **Technical Details**: 이 시스템은 Large Vision Language Model (LVLM)과 multi-modal Chain-of-Thought (CoT) prompting을 결합하여 자동 계획을 수행합니다. 또한 MotionPaint라는 새로운 방법을 통해 사전 훈련된 video diffusion 모델에서 모션 인식 정보를 증류하여 동적인 특징을 부여합니다.

- **Performance Highlights**: MotionCom은 제로샷(zero-shot) motion-aware 합성을 달성하며, 새로운 장면에서 목표 객체를 쉽게 통합할 수 있어, 진정한 모션과 상호작용을 묘사하는 구성물을 만들어냅니다.



### DAE-Fuse: An Adaptive Discriminative Autoencoder for Multi-Modality Image Fusion (https://arxiv.org/abs/2409.10080)
- **What's New**: 이번 연구에서는 DAE-Fuse라는 새로운 두 단계의 차별화된 오토인코더(framework) 구조를 제안하여 높은 선명도와 자연스러운 이미지 융합(fusion)을 생성합니다.

- **Technical Details**: DAE-Fuse는 어드버셜(feature extraction) 학습을 통해 입력 이미지의 세부 정보를 보존하며 구조적 차이를 구별하는 두 개의 차별화된 블록(discriminative blocks)을 포함합니다. 이 프레임워크는 깊은 고주파 인코더(Deep High-frequency Encoder)와 깊은 저주파 인코더(Deep Low-frequency Encoder)를 병렬로 사용하여 다양한 주파수 특성을 추출합니다. 또한, Cross-Attention Fusion을 통해 두 개의 서로 다른 모달리티에서 중요한 상호작용(interaction)을 포착합니다.

- **Performance Highlights**: 공공 IR-VISIBLE 데이터셋 및 의료 영상 융합(MIF) 데이터셋에서 DAE-Fuse는 정량적인 평가와 정성적인 평가 모두에서 최첨단의 성능을 입증했습니다. 또한, 다운스트림 MMOD 작업에 있어서도 성능 향상을 기록하며, 추가적인 미세 조정(fine-tuning) 없이도 뛰어난 일반화 능력을 보여줍니다.



### Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation (https://arxiv.org/abs/2409.10071)
Comments:
          8 pages, 6 figures, submitted to the 2025 IEEE International Conference on Robotics & Automation (ICRA)

- **What's New**: 이 연구는 안전-critical 환경에서의 embodied navigation 에이전트에 대한 실용적인 적대적 공격 방법을 제안합니다. 특히, 학습 가능한 텍스처와 불투명도(opacity)를 가진 적대적 패치를 객체에 부착함으로써 다양한 시점에서의 효과를 보장하고, 인간 관찰자에게 눈에 띄지 않도록 하는 두 가지 주요 기법을 도입하였습니다.

- **Technical Details**: 제안된 방법은 객체 인식에 기반한 샘플링을 활용하여 다중 시점 최적화(multi-view optimization) 전략으로 적대적 패치를 개선하는 것입니다. 이를 통해 에이전트의 탐색 모델으로부터의 피드백을 사용하여 패치의 텍스처(texture)를 최적화합니다. 또한, 불투명도 최적화(opacity optimization) 메커니즘을 도입하여 패치의 불투명도를 조정하고, 이로 인해 패치를 인간에게 덜 인식 가능하도록 만듭니다.

- **Performance Highlights**: 실험 결과, 제안된 적대적 패치는 탐색 성공률을 약 40% 감소시키는 것으로 나타났으며, 이전 방법들에 비해 실용성, 효과성, 자연스러움에서 우수한 성능을 보였습니다. 또한, 객체 인식을 기반으로 하는 다중 시점 최적화가 패치의 성공적인 작동에 필수적이라는 것을 확인하였습니다.



### GlobalMapNet: An Online Framework for Vectorized Global HD Map Construction (https://arxiv.org/abs/2409.10063)
- **What's New**: 본 논문은 고충전식 HD 맵 구축을 위한 최초의 온라인 프레임워크인 GlobalMapNet을 소개합니다. 이 프레임워크는 자동차가 수집한 로컬 맵 정보를 지속적으로 업데이트하고 활용하여 일관된 인식 결과를 생성합니다.

- **Technical Details**: GlobalMapBuilder를 통해 로컬 맵을 계속해서 일치시키고 병합하여 글로벌 맵을 생성합니다. 중복 맵 요소를 제거하는 새로운 알고리즘인 Map NMS를 설계하였으며, 역사적 맵 정보를 집계하여 예측의 일관성을 개선하는 GlobalMapFusion을 제안합니다.

- **Performance Highlights**: nuScenes와 Argoverse2 데이터셋에서 GlobalMapNet의 성능을 평가하였으며, 이 프레임워크가 글로벌 일관된 결과를 생성할 수 있음을 보여주었습니다.



### DENSER: 3D Gaussians Splatting for Scene Reconstruction of Dynamic Urban Environments (https://arxiv.org/abs/2409.10041)
- **What's New**: DENSER는 3D Gaussian splatting (3DGS)을 활용하여 동적인 도시 환경의 재구성을 위한 효율적이고 효과적인 접근 방식을 제시합니다. 기존의 메서드들이 동적 객체의 모델링에서 한계를 보였던 부분을 개선하였습니다.

- **Technical Details**: DENSER는 동적인 객체의 모양 표현을 강화하기 위해 포인트 클라우드를 씬의 여러 프레임에 걸쳐 밀집화하는 방법을 적용합니다. 또한, 동적인 객체의 Appearance를 모델링하기 위해 Spherical Harmonics (SH)를 직접 사용하지 않고 웨이브렛(wavelets)을 활용하여 SH 기초를 동적으로 추정하는 새로운 방법을 도입하였습니다.

- **Performance Highlights**: KITTI 데이터세트에 대한 광범위한 평가 결과, DENSER는 최신 방법들보다 현저히 높은 성능을 보여주었습니다.



### AttnMod: Attention-Based New Art Styles (https://arxiv.org/abs/2409.10028)
- **What's New**: 이번 논문에서는 AttnMod을 제안하여 기존의 diffusion 모델에서 새로운 예술 스타일을 생성하기 위한 주의(attention)를 수정하는 메커니즘을 다룹니다. 이를 통해 예술가는 이미지 생성 과정에서 특정 요소에 집중하거나 희석할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 UNet의 attention layer를 열어 주의 수정을 통해 생성된 이미지의 예술적 차이를 탐구합니다. 각우선, 텍스트 프롬프트와 결합된 cross attention 블록을 사용하여 denoising diffusion 중 주의 수정을 진행합니다. 이 과정에서 attention multiplier를 도입하여 조건화에 사용된 주의의 양을 정량화합니다.

- **Performance Highlights**: AttnMod는 다양한 예술 스타일에서 실험되었으며, 각 스타일의 텍스트 프롬프트에 따라 생성되는 이미지의 예술적 변화가 관찰되었습니다. 또한, 고정된 또는 가변 속도로 주의를 변화시키는 방식이 생성 결과에 어떤 영향을 미치는지도 연구하였습니다.



### LithoHoD: A Litho Simulator-Powered Framework for IC Layout Hotspot Detection (https://arxiv.org/abs/2409.10021)
Comments:
          14 pages to appear in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems

- **What's New**: 본 논문에서는 리소그래피 시뮬레이터를 이용한 새로운 핫스팟 탐지 프레임워크를 제안합니다. 기존의 학습 기반 핫스팟 탐지기가 단순히 훈련 데이터의 문제 있는 레이아웃 패턴만 인식하는 데 한계가 있는 점을 극복하고자 하였습니다. 새로운 프레임워크는 리소그래피 시뮬레이터와 객체 탐지 네트워크를 통합하여 핫스팟을 보다 정확히 검출할 수 있습니다.

- **Technical Details**: 제안된 핫스팟 탐지 프레임워크는 두 가지 주요 요소로 구성됩니다. 첫째, 리소그래피 시뮬레이터(LithoNet)로부터 도출된 변형 맵을 활용하여 레이아웃 패턴과 핫스팟 간의 삼중 관계를 학습하며, 둘째, RetinaNet과 같은 객체 탐지 네트워크를 사용하여 핫스팟을 탐지합니다. 이 과정에서 크로스-어텐션 블록을 이용해 피쳐 텐서를 융합하여 탐지 정확도를 향상시킵니다.

- **Performance Highlights**: 대규모 실험 결과, 제안된 시뮬레이터 기반 핫스팟 탐지 프레임워크가 기존의 최첨단 방법보다 높은 성능을 보임을 입증하였습니다. 이는 더욱 복잡한 실제 세계의 레이아웃에서의 핫스팟 탐지에 대한 일반화 능력을 향상시켰습니다.



### 2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction (https://arxiv.org/abs/2409.09969)
Comments:
          ECCV2024 this https URL

- **What's New**: 이 논문은 새로운 2S-ODIS (Two-Stage Omni-Directional Image Synthesis) 방법을 제안하여, 고품질의 omni-directional 이미지를 생성하면서 훈련 시간을 대폭 단축시켰습니다. 기존 GAN을 기반으로 한 방법들과 달리, VQGAN (Vector Quantized GAN)을 사용하여 훈련 없이 직접 omni-directional 이미지를 합성할 수 있게 되었습니다.

- **Technical Details**: 제안된 방법은 두 단계 구조를 갖추고 있으며, 첫 번째 단계에서는 글로벌 코스 이미지가 생성되고, 두 번째 단계에서는 다수의 NFoV (Normal Field of View) 이미지를 통합하여 지오메트릭 왜곡(geometric distortion)을 보정한 후 고해상도(1024×2048 pixels)로 정제된 이미지를 생성합니다. 이 모든 과정에서 사전 훈련된 VQGAN 모델이 사용됩니다.

- **Performance Highlights**: 2S-ODIS 방법은 기존 OmniDreamer 방법에 비해 훈련 시간을 14일에서 4일로 대폭 줄이면서도 더 높은 품질의 omni-directional 이미지를 생성하는 데 성공했습니다.



### Artificial Intelligence-Based Opportunistic Coronary Calcium Screening in the Veterans Affairs National Healthcare System (https://arxiv.org/abs/2409.09968)
- **What's New**: 이번 연구에서는 비심장 목적으로 촬영된 비조영(non-contrast) CT 스캔에서 관상동맥 칼슘(CAC)을 자동으로 정량화하는 심층 학습 알고리즘(AI-CAC)을 개발했습니다. 이는 98개의 의무 의료센터로부터 다양한 이미징 프로토콜과 스캐너를 포함한 데이터를 활용하여 수행되었습니다.

- **Technical Details**: AI-CAC은 446개의 전문가 세분화를 사용하여 개발되었으며, 비심장 비게이트(non-gated) CT 스캔에서 정확한 CAC 측정을 제공합니다. 이 알고리즘은 환자 795명의 데이터를 대상으로 ECG-게이트(gated) CAC 스코어와 비교하였으며, 0 vs. 비 0, 100 미만 vs. 100 이상의 Agatston 점수에서 각각 89.4% 및 87.3%의 정확성을 기록했습니다.

- **Performance Highlights**: AI-CAC는 10년 전반적인 사망률 예측 및 최초의 뇌졸중(stroke), 심근경색(MI), 사망에 대한 결합 예측에서 CAC 0 그룹과 >400 그룹 간에 각각 25.4% vs. 60.2%, 33.5% vs. 63.8%의 결과를 보였으며, 8,052명의 저선량 폐암 스크리닝 CT 데이터셋에서 38.4%가 AI-CAC >400으로 나타났습니다. 4명의 심장 전문의가 무작위로 샘플링한 400 이상 AI-CAC 환자들에 대한 이미지를 검토한 결과, 99.2%가 지질 저하 요법(lipid-lowering therapy)을 통해 혜택을 받을 것으로 확인되었습니다.



### Uncertainty-Guided Appearance-Motion Association Network for Out-of-Distribution Action Detection (https://arxiv.org/abs/2409.09953)
Comments:
          Accepted by MIPR 2024

- **What's New**: 이 논문에서는 동적 멀티미디어 시나리오에서 Out-of-Distribution (OOD) Action Detection (ODAD) 문제를 해결하기 위해 Uncertainty-Guided Appearance-Motion Association Network (UAAN)을 제안합니다. UAAN은 외관 특징과 모션 맥락을 탐색하여 OOD 행동을 탐지합니다.

- **Technical Details**: UAAN은 외관 및 모션 지향 객체 표현을 추출하기 위해 별도의 브랜치를 설계하며, 각 브랜치에서는 공간-시간 그래프를 구성하여 객체 간 상호작용을 추론합니다. 마지막으로, appearance-motion attention 모듈을 통해 두 브랜치에서의 특징을 융합하여 최종 행동 탐지를 수행합니다.

- **Performance Highlights**: 두 개의 도전적인 데이터세트에서 실험한 결과, UAAN은 기존의 최첨단 방법들보다 현저하게 우수한 성능을 보여 OOD 행동 탐지의 효과iveness를 입증하였습니다.



### Forearm Ultrasound based Gesture Recognition on Edg (https://arxiv.org/abs/2409.09915)
Comments:
          Please contact the authors for code and any additional questions pertaining to the project. You can reach Keshav Bimbraw at bimbrawkeshav at gmail dot com

- **What's New**: 이 논문에서는 팔꿈치 초음파를 이용한 손 제스처 인식의 실시간 시스템을 구현한 연구 결과를 소개합니다. 특히, 모바일 엣지 장치에서 Deep Neural Networks를 활용하여 독립 실행형 시스템을 만드는 데 중점을 두었습니다.

- **Technical Details**: 이 연구에서는 Convolutional Neural Network (CNN) 모델을 통해 초음파 기반 손 제스처 인식 시스템을 구현하였습니다. Float16 양자화(quantization) 기법을 적용하여 Raspberry Pi에서 92%의 테스트 정확도(test accuracy)와 0.31초의 추론 시간(inference time)을 달성했습니다. 모델 크기는 1.5MB에서 0.24MB로 크게 줄어들었습니다.

- **Performance Highlights**: Raspberry Pi에서의 성능 결과는 우수합니다. 모델은 Float16을 통한 양자화 후에도 92%의 정확도를 유지하였으며, 이는 리소스 제한 환경에서도 실시간 손 제스처 인식이 가능함을 보여줍니다. 이 기술은 향후 웨어러블 초음파 시스템을 위한 토대를 마련할 것으로 기대됩니다.



### Rapid Adaptation of Earth Observation Foundation Models for Segmentation (https://arxiv.org/abs/2409.09907)
Comments:
          9 pages 2 figures

- **What's New**: Low-Rank Adaptation (LoRA)의 사용이 지구 관측(Earth Observation) 기초 모델을 홍수 세분화(flood segmentation) 작업에 적합하게 조정하는 방식으로, 기존 방식보다 훨씬 더 효율적인 성능 개선을 보여주었다.

- **Technical Details**: LoRA 기법은 사전 훈련된 변환기(Transformer) 기반 모델의 주의(attention) 레이어에 저차원(low-rank) 업데이트 행렬을 도입합니다. 이로 인해 적은 양의 추가 매개변수로 특정 작업에 맞게 조정할 수 있게 되고, 모델의 연산 효율성을 높이면서도 성능을 유지하거나 개선합니다.

- **Performance Highlights**: 실험 결과, LoRA 기반의 세분화에 의해 F1 점수가 6.66 포인트, IoU가 0.11 향상되었으며, 연산 비용은 대폭 줄어들어 기존의 전체 미세 조정(full fine-tuning) 방식과 비교해 더욱 효과적임을 입증하였다.



### GRIN: Zero-Shot Metric Depth with Pixel-Level Diffusion (https://arxiv.org/abs/2409.09896)
- **What's New**: 이번 논문에서는 GRIN(Geometric RIN)라는 새로운 확산(difussion) 모델을 제안하여 희소한 비구조적(training data) 훈련 데이터를 효과적으로 처리하면서 3D 깊이 예측을 수행할 수 있는 방법을 보여줍니다. 이 모델은 3D 기하학적 위치 인코딩을 활용하여 깊이 예측의 정확도를 높입니다.

- **Technical Details**: GRIN은 기존 U-Net 아키텍처에 의존하지 않고 효율적으로 실시간 처리할 수 있는 픽셀 수준의 확산 아키텍처를 기반으로 합니다. 모델은 이미지 특징과 3D 기하학적 위치 인코딩을 결합하여 깊이 추정의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: GRIN은 8888개의 다양한 실내 및 실외 데이터셋에 대한 포괄적인 실험을 통해 제로샷(Zero-Shot) 메트릭 단안 깊이 예측에서 새로운 최고 성능을 달성하였습니다.



### Resolving Inconsistent Semantics in Multi-Dataset Image Segmentation (https://arxiv.org/abs/2409.09893)
- **What's New**: 새로운 논문에서는 여러 개의 데이터셋을 활용한 이미지 분할 모델의 훈련 방법을 제안합니다. 이 방법은 언어 기반 임베딩(language-based embeddings)과 레이블 공간별 쿼리 임베딩(label space-specific query embeddings)을 통합하여 레이블 불일치 문제를 해결합니다.

- **Technical Details**: 제안된 RESI 프레임워크는 Mask2Former 아키텍처를 기반으로 하며, (1) 클래스를 단일 일관된 공간으로 매핑하는 CLIP의 시각-언어 임베딩(vision & language embeddings)을 사용하고, (2) 쿼리 임베딩을 마스크 예측(masks predictions)의 디코더에 추가하여 레이블 공간에 조건을 부여합니다. 이를 통해 다양한 데이터셋 조합에 유연하게 대응할 수 있습니다.

- **Performance Highlights**: RESI는 여러 벤치마크 데이터셋에서 기존 방법보다 1.6% mIoU(Mean Intersection over Union), 12.1% AP(Average Precision), 9.1% PQ(Panoptic Quality), 그리고 3.0% PIQ(Panoptic Instance Quality)에서 성능이 향상됨을 입증하였습니다.



### REG: Refined Generalized Focal Loss for Road Asset Detection on Thai Highways Using Vision-Based Detection and Segmentation Models (https://arxiv.org/abs/2409.09877)
Comments:
          14 pages

- **What's New**: 이 논문은 태국 고속도로에서 중요한 도로 자산을 탐지하고 분할하는 새로운 프레임워크를 소개합니다. 이를 통해 Refined Generalized Focal Loss (REG)라는 고급 손실 함수를 활용하여, 작은 도로 요소들이나 클래스 imbalance 문제를 해결합니다.

- **Technical Details**: 제안된 REG 손실 함수는 복잡한 환경에서 예측 불확실성을 고려하는 확률적 정제와 도로 자산의 공간적 분포를 반영하는 공간-문맥 조정 항을 포함합니다. REG는 다중 작업 학습(multi-task learning) 전략을 적용하여 동시에 탐지와 분할을 최적화합니다.

- **Performance Highlights**: 실험 결과, REG를 적용한 모델은 mAP50 80.34와 F1-score 77.87를 달성하며 전통적인 방법보다 성능이 크게 향상되었음을 보여주었습니다. 이는 도로 자산 탐지와 분할의 정확성과 견고성을 높이는데 기여합니다.



### Towards Kinetic Manipulation of the Latent Spac (https://arxiv.org/abs/2409.09867)
- **What's New**: 이번 연구에서는 기존의 Graphical User Interfaces (GUIs)에서 벗어나, 실시간 RGB 카메라 피드를 활용하여 사전 훈련된 Convolutional Neural Networks (CNNs)를 이용한 간단한 feature extraction을 통해 생성 모델의 latent space를 효과적으로 조작하는 방법을 제시합니다. 이러한 새로운 패러다임은 Visual-reactive Interpolation으로 명명되었습니다.

- **Technical Details**: 이 시스템은 Generative Adversarial Networks (GANs) 및 diffusion 기반 text-to-image 모델을 활용하여, 사용자가 자신의 몸이나 얼굴 동작을 통해 생성 모델의 latent space를 실시간으로 조작할 수 있도록 합니다. StyleGAN의 latent space를 조작하는 데 초점을 맞추며, 최종적으로는 다양한 생성 모델과의 호환성을 통해 폭넓은 적용성을 목표로 합니다. 고급 기능 추출기(pre-trained feature extractors)를 사용하여, GAN의 latent space로의 인코딩을 가능하게 합니다.

- **Performance Highlights**: 이 연구는 사용자가 전체 몸, 얼굴, 장면 내 물체를 통해 이미지를 생성할 수 있는 상호작용을 가능하게 하여, 예술적 표현의 새로운 가능성을 열어줍니다. 카메라 또한 이미지 생성 파이프라인의 '두 번째 주체'로 작용하여, 생성된 출력에 협력적으로 영향을 미칠 수 있게 합니다.



### Tracking Virtual Meetings in the Wild: Re-identification in Multi-Participant Virtual Meetings (https://arxiv.org/abs/2409.09841)
Comments:
          Accepted to ECCV 2024 workshop

- **What's New**: 이 논문에서는 비디오 미팅에서 참가자들을 효과적으로 추적하고 재식별하는 새로운 방법을 제안합니다. 기존의 YOLO 기반 접근 방식을 초월하여 지원되는 스페이셜-템포럴(Spatio-Temporal) 우선순위를 활용하여 더욱 정확한 추적이 가능합니다.

- **Technical Details**: 제안된 방법론은 'Gallery View' 형식으로 비디오 미팅의 동적인 참가자들을 다루며, 참가자 위치에 대한 메타데이터 없이 단일 비디오 소스에서 참가자들을 추적할 수 있습니다. 전통적인 객체 추적 방식은 주로 CCTV 영상의 선형 동작 경로를 가정하지만, 본 접근법은 대화형 비디오 미팅의 비선형적이고 예측 불가능한 참가자 동작을 고려합니다.

- **Performance Highlights**: 본 연구는 YOLO 기반 추적 방법과 비교하여 평균 95%의 오류율 감소를 달성하였으며, 이는 원활한 참가자 식별을 보장합니다. 이러한 결과는 복잡한 동적 환경에서도 높은 정확도로 비디오 미팅 중 참가자들을 신뢰성 있게 추적할 수 있는 가능성을 보여줍니다.



### Template-based Multi-Domain Face Recognition (https://arxiv.org/abs/2409.09832)
Comments:
          IJCB 2024 - Special Session on Recognition at Long Range and from High Altitude

- **What's New**: 본 논문에서는 표준 average pooling보다 우수한 성능을 보이는 Norm Pooling이라는 템플릿 생성 알고리즘을 소개합니다. 이 알고리즘은 다양한 도메인에서의 얼굴 인식 과제를 향상시키는 데 기여합니다.

- **Technical Details**: 본 연구는 단일 출처 (visible)와 다중 목표 (SWIR, 긴거리/원거리, 감시, 착용형) 얼굴 인식 문제에 초점을 맞추고 있습니다. Norm Pooling 알고리즘이 제안되었으며, 이는 기존의 average pooling 방식과는 다른 접근 방식을 통해 템플릿을 생성합니다.

- **Performance Highlights**: IARPA JANUS Benchmark Multi-domain Face (IJB-MDF) 데이터셋에서의 실험을 통해, Norm Pooling 알고리즘이 다양한 도메인과 네트워크 성능을 초과함을 입증했습니다.



### Famba-V: Fast Vision Mamba with Cross-Layer Token Fusion (https://arxiv.org/abs/2409.09808)
Comments:
          Camera ready version of ECCV 2024 The Fourth Workshop on Computational Aspects of Deep Learning

- **What's New**: Famba-V는 Vision Mamba(Vim) 모델의 훈련 효율성을 높이기 위해 cross-layer token fusion 기술을 도입한 새로운 방법론입니다. 기존의 token fusion 방법은 모든 레이어에 일관적으로 적용되었으나, Famba-V는 다양한 cross-layer 전략을 사용하여 유사한 토큰들만을 선택하여 융합함으로써 효율성을 극대화합니다.

- **Technical Details**: Famba-V는 2D 비주얼 데이터를 처리하기 위해 Vision Mamba(Vim) 모델 내에서 토큰의 유사성을 측정하여 유사 정보를 포함한 토큰들을 융합합니다. 토큰 융합 전략은 레이어 선택에서 최적의 정확도 및 효율성 트레이드오프를 제공하는 데 기여합니다. 실험은 CIFAR-100 데이터셋에서 수행되었습니다.

- **Performance Highlights**: Famba-V는 기존의 Vim 모델과 비교할 때 훈련 시간을 줄이고 최대 메모리 사용량을 감소시킴으로써 훈련 효율성을 significantly 향상시킵니다. cross-layer token fusion 전략을 적용한 결과 기존의 전 레이어 token fusion 방식보다 더 나은 정확도-효율성 트레이드오프를 제공합니다.



### Abnormal Event Detection In Videos Using Deep Embedding (https://arxiv.org/abs/2409.09804)
- **What's New**: 이 논문은 비감독 학습(unsupervised learning) 방식을 활용하여 비디오에서 이상 사건(anomalous event)을 탐지하기 위한 새로운 접근 방식을 제안합니다. 기존의 이상 탐지 방법들이 정상 이벤트를 학습한 후 이상 값을 감별하는 방식이었다면, 본 논문에서는 심층 신경망(deep neural network)의 목적함수와 이상 탐지 과제를 동시에 최적화하는 하이브리드 아키텍처(hybrid architecture)를 통해 효율성을 극대화합니다.

- **Technical Details**: 제안된 비정상 사건 탐지 방법은 세 가지 주요 단계로 구성됩니다: 1) 레이턴트 피처 추출(Latent Feature Extraction), 2) 피처 융합(Feature Fusion), 3) 일급 분류(One-class classification). 이 시스템은 깊이(depth) 맵, 광학 흐름(optical flow), 그리고 외형 특징(appearance features)을 융합하여 비디오 신호에서 비정상적인 사건을 식별합니다. 사전 학습된 모델을 사용하여 각 입력 모달리티의 레이턴트 표현(latent representation)을 합칩니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 비정상 데이터의 임베딩이 하이퍼센터(hypercenter)에서 멀리 떨어지도록 조정함으로써, 정상 데이터는 하이퍼센터 근처에 군집화되도록 설계되었습니다. 이러한 방식은 영상 내에서 비정상 사건을 효과적으로 식별할 수 있도록 하여, 더욱 높은 성능을 자랑하는 이상 탐지 시스템을 구현합니다.



### Multiple Rotation Averaging with Constrained Reweighting Deep Matrix Factorization (https://arxiv.org/abs/2409.09790)
- **What's New**: 본 논문에서는 기존의 최적화 기반 및 학습 기반 방법의 장점을 결합하여 여전히 레이블이 필요하지 않은 비지도 학습(Unsupervised Learning) 방식으로 다회전 평균(Multiple Rotation Averaging, MRA) 문제를 해결하는 새로운 방법을 제안합니다.

- **Technical Details**: 딥 매트릭스 팩토리제이션(Deep Matrix Factorization) 기법을 통해 비제한 선형 공간에서 다회전 평균 문제를 직접 해결합니다. 명시적으로 저랭크(Low-rank) 및 대칭(Symmetric) 구조를 가진 신경망 모델을 설계하여 관찰된 상대 측정값을 네트워크 최적화에 대한 제약으로 활용합니다. 스패닝 트리 기반의 엣지 필터링(Spanning Tree-based Edge Filtering) 및 재가중치 기법(Reweighting Scheme)을 통해 회전 이상치(Outlier)의 영향을 억제합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과를 통해 제안된 방법의 효과가 입증되었으며, 성능 개선을 위한 동적 깊이 선택 전략(Dynamic Depth Selection Strategy)도 도입하였습니다.



### Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models (https://arxiv.org/abs/2409.09788)
Comments:
          20 pages, 13 figures

- **What's New**: 최근 비전-언어 모델(VLMs)의 성능 향상을 보여주는 연구에도 불구하고, 객체 크기와 거리의 정량적 추론 능력은 아직 충분히 탐구되지 않았다. 이번 연구에서는 정량적 공간 추론을 위해 설계된 271개의 질문이 포함된 수동 주석 기준인 Q-Spatial Bench를 소개한다.

- **Technical Details**: Q-Spatial Bench는 다섯 가지 범주로 구성된 질문들이 포함되어 있으며, 이는 정량적 공간 추론을 평가하기 위해 고안되었다. 분석 결과, 객체 간 거리 추론이 특히 도전적임을 밝혀냈으며, 일부 VLMs는 다른 모델에 비해 뛰어난 성능을 보였다. 저자들은 SpatialPrompt라는 제로샷 프롬프팅 기법을 개발하여 VLMs가 참조 객체를 시각적 단서로 사용하여 정량적 공간 질문에 응답하도록 유도하였다.

- **Performance Highlights**: Gemini 1.5 Pro, Gemini 1.5 Flash, 및 GPT-4V와 같은 모델은 SpatialPrompt를 사용함으로써 각각 40점, 20점, 30점 이상의 성공률 향상을 달성했다. 이는 데이터 추가, 모델 구조 수정, 또는 파인튜닝 없이 이루어진 개선이다.



### Enhancing Lesion Segmentation in PET/CT Imaging with Deep Learning and Advanced Data Preprocessing Techniques (https://arxiv.org/abs/2409.09784)
- **What's New**: 이번 연구는 PET/CT 이미징에서 병변(segmentation)의 정확성을 높이기 위해 딥러닝(deep learning)을 적용하였으며, AutoPET challenge III에서 제공된 900개의 전체 신체 FDG-PET/CT 데이터와 600개의 PSMA-PET/CT 데이터를 활용하였습니다.

- **Technical Details**: 연구에서는 데이터 전처리(preprocessing) 및 데이터 증강(augmentation) 기법을 활용하여 모델의 견고성과 일반화 가능성을 확보했습니다. 비영점(normalization) 정규화과 RandGaussianSharpen 추가 및 Gamma 변환 매개변수의 조정을 통해 성능 향상을 검토하였습니다.

- **Performance Highlights**: FDG 추적자의 경우 기본 설정에서 Dice 계수가 63.19%로 중간 정도의 분할 정확도를 나타냈으며, PSMA 추적자의 경우 32.07%로 나타났습니다. Gaussian sharpening 기법을 활용한 데이터 증강 후 FDG는 64.11%, PSMA는 44.55%로 향상되었습니다. 특히 최대 클립 값 280을 적용했을 때 FDG의 Dice 점수는 53.69%로 개선되었습니다.



### Underwater Image Enhancement via Dehazing and Color Restoration (https://arxiv.org/abs/2409.09779)
- **What's New**: 최근 해양 공학 프로젝트에서 수중 영상 처리 기술의 필요성이 큰 증가를 하고 있습니다. 본 논문에서는 수중 이미지 품질을 향상시키기 위한 새로운 네트워크인 WaterFormer를 제안합니다. WaterFormer는 수중에서 빛의 비선형 감쇠로 인한 문제를 해결하기 위해 다양한 블록(DehazeFormer Block, Color Restoration Block, Channel Fusion Block)을 통합한 구조로 설계되었습니다.

- **Technical Details**: WaterFormer는 다음과 같은 주요 구성 요소로 이루어져 있습니다: DehazeFormer Block을 통해 자가 상관된 헤이즈 특성을 포착하고, Color Restoration Block(CRB)을 통해 색상 왜곡을 복원하며, Channel Fusion Block(CFB)을 통해 다양한 특성의 융합을 수행합니다. 각 블록은 네트워크 내에서 기능을 통합하고, 마지막에 수중 이미징 물리 모델에 기반한 부드러운 재구성 층을 통해 결과의 진위성을 보장합니다.

- **Performance Highlights**: 실험 결과, WaterFormer는 기존의 최첨단 방법들보다 우수한 성능을 보여주었습니다. 교육 과정에서 Chromatic Consistency Loss와 Sobel Color Loss를 도입하여 색상의 일관성을 유지하고, 이미지 품질 및 다양한 데이터셋에 대한 일반화 능력을 향상시켰습니다.



### DiFSD: Ego-Centric Fully Sparse Paradigm with Uncertainty Denoising and Iterative Refinement for Efficient End-to-End Autonomous Driving (https://arxiv.org/abs/2409.09777)
- **What's New**: 이 논문에서는 기존의 모듈형 자율주행 시스템의 한계를 극복하기 위해, 인간의 운전 행동을 재조명하고 에고 중심의 완전 희소(paradigm) 모형인 DiFSD를 제안합니다. 이 시스템은 감지, 추적 및 온라인 매핑을 포함하는 희소 인식 모듈과, 대칭적 상호 작용을 목표로 하는 계층적 상호 작용 모듈, 그리고 다중 모달 자기 궤적을 최적화하는 반복적인 동작 계획자를 포함합니다.

- **Technical Details**: DiFSD는 감지와 추적을 동시에 수행하는 희소 인식(sparse perception) 모듈, 가장 적합한 차량(CIPV/CIPS)을 선택하는 계층적 상호 작용(hierarchical interaction) 모듈, 그리고 상호 작용하는 주체들과의 공동 동작 예측(joint motion prediction)을 고려하는 반복적인 동작 계획(iterative motion planner)까지 이루어져 있습니다. 이 시스템은 동작 분산(position-level motion diffusion) 및 경로 계획의 노이즈 감소(trajectory-level planning denoising) 기법을 통해 학습 안정성과 수렴성을 높입니다.

- **Performance Highlights**: DiFSD는 nuScenes 데이터셋에서 실험을 통해 평균 L2 오류를 66% 줄이고 충돌율을 77% 감소시켰으며, UniAD에 비해 8.2배 더 빠른 실행 효율성을 달성했습니다.



### Generalizing Alignment Paradigm of Text-to-Image Generation with Preferences through $f$-divergence Minimization (https://arxiv.org/abs/2409.09774)
Comments:
          32 pages

- **What's New**: 이번 연구에서는 텍스트-이미지 모델의 정렬(Alignment) 과정에서 역 쿨벡-라이블러 발산(Reverse Kullback-Leibler divergence)만을 사용하는 기존 접근 방식의 한계를 넘어 $f$-발산($f$-divergence)으로 확대하였습니다. 이는 텍스트-이미지 모델의 정렬 성능 및 생성 다양성을 동시에 개선하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 연구는 텍스트-이미지 모델의 정렬 기준을 역 쿨벡-라이블러 발산에서 $f$-발산으로 일반화하는 방안을 제안하고 있습니다. 다양한 발산 제약(constrains)이 정렬 과정에 미치는 영향을 기울기(field)의 관점에서 분석하였습니다. 또한, 기준 방법으로 Step-aware Preference Optimization (SPO)을 사용하고, Stable Diffusion V1.5를 모델로 삼아 다양한 발산 제약 하에 성능 평가를 실시하였습니다.

- **Performance Highlights**: Jensen-Shannon 발산(Jensen-Shannon divergence)을 기준으로 할 때, 이미지-텍스트 정렬 성능, 인간 가치 정렬(human value alignment) 성능 및 생성 다양성(generation diversity) 성능 간에 최적의 균형을 달성한 것으로 나타났습니다. 따라서 텍스트-이미지 모델 정렬에서 적절한 발산 제약 선택의 중요성이 강조됩니다.



### Automated Lesion Segmentation in Whole-Body PET/CT in a multitracer setting (https://arxiv.org/abs/2409.09766)
- **What's New**: 본 연구는 FDG 및 PSMA PET/CT 이미지를 자동으로 분할하는 워크플로우를 탐구하며, 이 과정에서 YOLOv8을 활용하여 데이터를 분류하고 각각의 이미지를 별도로 전처리하여 종양 분할 정확도를 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: 자동 종양 분할 과정은 두 단계로 구성됩니다. 첫째, FDG-PET 및 PSMA-PET 의학 이미지를 구별하기 위한 분류 모델을 훈련합니다. 둘째, 독립적으로 훈련된 두 개의 3D U-Net을 사용해 각각 FDG와 PSMA 데이터를 위한 장기 및 종양 분할을 시행합니다. 데이터는 AutoPET challenge III에서 제공된 전체 신체 FDG PET/CT와 PSMA PET/CT를 기반으로 하고, nnU-Net 프레임워크에 통합된 전처리 절차를 거칩니다.

- **Performance Highlights**: PET 모델은 FDG와 PSMA PET 이미지를 구별하는 데 있어 99.85%의 분류 정확도를 달성했습니다. FDG의 경우 다이스 계수(Dice coefficient)는 0.8408, PSMA는 0.7385였으며, 거짓 양성(FPvol)과 거짓 음성(FNvol) 볼륨은 각각 FDG는 1.7979 및 2.3625, PSMA는 9.3574 및 5.0745로 나타났습니다. 이는 두 가지 이미징 모달리티 간의 분할 성능 차이를 보여줍니다.



### MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation (https://arxiv.org/abs/2409.09756)
Comments:
          18 pages, 8 figures, ECCV 2024

- **What's New**: 본 논문에서는 3D Gaussian Splatting을 활용한 새로운 방법인 MesonGS를 제안합니다. MesonGS는 사후 교육(post-training) 압축 기술을 사용하여 3D Gaussian의 파일 크기를 줄이고 교육 시간을 최소화하며 품질을 유지하는 효율적인 방법입니다.

- **Technical Details**: MesonGS는 두 가지 주요 변환을 통해 속성의 중복성과 엔트로피를 감소시키며, 각 Gaussian의 중요성을 평가하기 위해 보편적인 Gaussian 가지치기(universal Gaussian pruning), 속성 변환(attribute transformation), 블록 양자화(block quantization) 기술을 적용합니다. 특히, 회전 쿼터니언(rotation quaternions)을 오일러 각도(Euler angles)로 변환하고, 지역 적응형 계층 변환(region adaptive hierarchical transform)를 사용하여 주요 속성의 엔트로피를 낮추는 접근 방식을 채택합니다.

- **Performance Highlights**: MesonGS는 3D Gaussian의 크기를 상당히 줄이면서도 경쟁력 있는 품질을 유지하는 성능을 보여주며, 기존 모델들과 비교했을 때 13배의 압축률 증가를 달성하여 품질 저하가 거의 없는 결과를 보입니다.



### Towards Single-Lens Controllable Depth-of-Field Imaging via All-in-Focus Aberration Correction and Monocular Depth Estimation (https://arxiv.org/abs/2409.09754)
Comments:
          The source code and the established dataset will be publicly available at this https URL

- **What's New**: 이 연구는 Minimalist Optical Systems (MOS)을 사용하여 단일 렌즈 조절 가능한 Depth-of-Field (DoF) 이미징을 위한 새로운 방법을 제안합니다. 특히, Depth-aware Controllable DoF Imaging (DCDI) 프레임워크는 All-in-Focus (AiF) 왜곡 보정 및 단안 깊이 추정(Depth Estimation)을 활용하여 다양한 DoF를 지원하는 이미징 결과를 생성합니다.

- **Technical Details**: DCDI 프레임워크는 세 가지 주요 작업, 즉 왜곡 이미지의 깊이 추정, Computation Aberration Correction (CAC), 그리고 Point Spread Function (PSF) 특성화를 포함합니다. Depth-aware Degradation-adaptive Training (DA2T) 스킴을 도입해 다양한 물체 거리에서 PSF를 시뮬레이션하여 Depth-aware Aberration MOS (DAMOS) 데이터셋을 구축했습니다. 두 가지 플러그 앤 플레이 깊이 인식 메커니즘인 Residual Depth-Image Cross-Attention Block (RDICAB) 및 Depth-aware Deformable Convolution Block (D2CB)을 설계하여 깊이 정보를 이미지 복원에 효과적으로 결합했습니다.

- **Performance Highlights**: 실험 결과, 제안된 DA2T 스킴과 두 개의 깊이 인식 메커니즘이 적용된 모델이 비깊이 인식 모델보다 더 우수한 복원 성능을 보여주었습니다. 특히, 다양한 깊이에서 장면을 성공적으로 복원할 수 있었고, 높은 품질의 AiF 왜곡 보정이 이루어져 조절 가능한 DoF 이미징을 가능하게 했습니다. 이 연구는 MOS를 통한 새로운 조절 가능한 DoF 이미징의 기초를 마련했습니다.



### DARDA: Domain-Aware Real-Time Dynamic Neural Network Adaptation (https://arxiv.org/abs/2409.09753)
- **What's New**: 본 연구에서는 기존 TTA 방법의 성능 저하 문제를 해결하기 위해 Domain-Aware Real-Time Dynamic Adaptation (DARDA)를 제안했습니다. DARDA는 입력에 영향을 미치는 오염 유형의 잠재 표현을 선제적으로 학습합니다.

- **Technical Details**: DARDA는 (i) 현재 오염의 잠재 표현을 추정하고; (ii) 해당 오염과 가장 가까운 서브네트워크를 선택한 후; (iii) DNN 상태를 조정하여 오염에 맞는 표현으로 발전시킵니다. 이를 통해 DNN은 다양한 오염에 빠르게 적응할 수 있습니다.

- **Performance Highlights**: DARDA는 Raspberry Pi와 NVIDIA Jetson Nano에서 시험된 결과, 에너지 소비와 평균 캐시 메모리 사용량을 각각 1.74배 및 2.64배 절감하면서도 CIFAR-10, CIFAR-100 및 TinyImagenet에서 성능을 각각 10.4%, 5.7%, 4.4% 향상시켰습니다.



### Explore the Hallucination on Low-level Perception for MLLMs (https://arxiv.org/abs/2409.09748)
- **What's New**: 본 논문에서는 Multi-modality Large Language Models(MLLMs)의 낮은 수준의 시각 인식 및 이해 작업에서의 자가 인식(self-awareness)을 정의하고 평가하기 위해 QL-Bench라는 벤치마크를 제안합니다.

- **Technical Details**: QL-Bench는 2,990장의 단일 이미지와 1,999장의 이미지 쌍으로 구성된 LLSAVisionQA 데이터셋을 바탕으로 하며, 각 이미지에 대한 저수준(low-level) 특성에 관한 개방형 질문이 포함되어 있습니다. 모델의 자가 인식 능력은 질문 유형에 따라 달라지며, 특히 복잡한 질문에 대해 더 나은 자가 인식 성능을 보이는 경향이 있습니다.

- **Performance Highlights**: 15개의 MLLMs을 평가한 결과, 일부 모델은 낮은 수준의 시각 능력을 잘 수행하나 자가 인식 능력은 상대적으로 부족함을 나타냈습니다. 간단한 질문에 대한 정답률은 높지만 복잡한 질문에서는 자가 인식이 향상되는 경향이 보였습니다.



### VGG-Tex: A Vivid Geometry-Guided Facial Texture Estimation Model for High Fidelity Monocular 3D Face Reconstruction (https://arxiv.org/abs/2409.09740)
- **What's New**: 이번 연구에서는 고품질의 기하학적 형상과 텍스처를 동시에 재구성하기 위해 VGG-Tex라는 새로운 모델을 제안합니다. 기존의 방법이 기하학적 재구성에 중점을 두었던 반면, 본 연구는 텍스처 예측의 중요성을 강조합니다. 이 모델은 3D 파라메트릭 프라이어(3D parametric priors)를 활용하여 2D UV 텍스처 예측 결과를 향상시키는 접근법을 사용합니다.

- **Technical Details**: VGG-Tex는 주로 세 가지 주요 모듈로 구성됩니다: Facial Attributes Encoding Module, Geometry-Guided Texture Generator, 그리고 Visibility-Enhanced Texture Completion Module입니다. Facial Attributes Encoding Module은 FLAME 모델의 파라미터를 예측하고, Geometry-Guided Texture Generator는 비전 트랜스포머(vision Transformer) 인코더와 텍스처 디코더를 사용하여 UV 텍스처를 추정합니다. 마지막으로, Visibility-Enhanced Texture Completion Module에서는 입력 이미지에 랜덤 마스크를 적용하여 가려진 부분을 효과적으로 채워넣을 수 있는 기능을 제공합니다.

- **Performance Highlights**: 실제 실험을 통해 VGG-Tex는 여러 벤치마크 데이터셋(FHQ-UV, VGGFace2, NoW)에서 기존의 최첨단 방법들보다 현저히 향상된 텍스처 재구성 성능을 보여주었습니다. VGG-Tex는 기하학적 텍스처 추정을 위한 지침을 통해 전체 3D 얼굴 재구성의 충실도를 높이는 데 중점을 두고 있습니다.



### MFCLIP: Multi-modal Fine-grained CLIP for Generalizable Diffusion Face Forgery Detection (https://arxiv.org/abs/2409.09724)
- **What's New**: 이 논문에서는 최첨단의 기본 모델인 contrastive language-image pre-training (CLIP)을 활용하여 일반화 가능한 확산 기반 얼굴 변조 탐지(DFFD)를 달성하는 새로운 접근법을 제안합니다.

- **Technical Details**: 제안된 모델 MFCLIP(multi-modal fine-grained CLIP)은 이미지-노이즈 모달리티 전반에서 포괄적이고 세밀한 변조 흔적을 발굴하는 언어 기반 얼굴 변조 표현 학습을 통해 발전을 도모합니다. 특히, Fine-grained Language Encoder (FLE)와 Multi-modal Vision Encoder (MVE)를 도입하여 세밀한 글로벌 언어 특징 및 이미지 변조 임베딩을 추출합니다. 또한, 혁신적인 샘플 쌍 주의(sample pair attention) 방법을 개발하여 관련 없는 쌍을 억제하는 방식으로 다양성을 증가시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 MFCLIP 모델은 교차 생성자, 교차 변조 및 교차 데이터셋 평가와 같은 다양한 설정에서 기존의 최신 기술들보다 우수한 성능을 보여 주었습니다.



### Disentangling Visual Priors: Unsupervised Learning of Scene Interpretations with Compositional Autoencoder (https://arxiv.org/abs/2409.09716)
- **What's New**: 이 연구에서는 객체, 모양 및 기하학적 변환과 같은 기본적인 시각 개념을 다루기 위한 새로운 신경 기호 언어 기반 아키텍처인 DVP(Disentangling Visual Priors)를 제안합니다.

- **Technical Details**: DVP는 세 가지 주요 구성 요소로 이루어져 있습니다: 1) Perception 모듈 - CNN(Convolutional Neural Network)을 사용하여 2D 입력 이미지를 잠재 벡터 z로 변환합니다. 2) DSL(도메인 특정 언어) 프로그램 - z로 매개변수화된 시각적 원시 요소를 생성하고 변환하는 역할을 합니다. 3) Renderer - 시각적 원시 요소를 시각화합니다.

- **Performance Highlights**: DVP는 이미지 형성 과정에서 개체 색상, 모양, 기하학적 변환 및 개체 카테고리를 분리하여 학습할 수 있으며, 이는 작은 데이터에서 학습하고 외삽(Out-of-sample) 일반화를 가능하게 합니다.



### Pre-Training for 3D Hand Pose Estimation with Contrastive Learning on Large-Scale Hand Images in the Wild (https://arxiv.org/abs/2409.09714)
Comments:
          HANDS@ECCV24 (Extended Abstracts)

- **What's New**: 이 논문에서는 야외에서 수집한 손 이미지 기반의 대조 학습 프레임워크인 HandCLR을 제안합니다. 이는 3D 손 자세 추정을 위한 사전 훈련(pre-training) 모델로, 기존의 방법들이 활용하지 못했던 다양한 야외 손 이미지를 최대한 활용할 수 있도록 설계되었습니다.

- **Technical Details**: HandCLR은 200만 개 이상의 손 이미지를 수집하고, 서로 다른 샘플에서의 유사한 손 자세 쌍을 중심으로 대조 학습을 수행합니다. 기존의 SimCLR 및 PECLR과는 달리, 서로 다른 이미지에서 유사한 손을 학습하여 더욱 유의미한 정보 획득을 가능하게 합니다. 이러한 방법은 2D 손 자세 추정기를 통해 유사한 손을 찾아 정의된 양성 쌍을 생성합니다.

- **Performance Highlights**: HandCLR은 FreiHand, DexYCB 및 AssemblyHands와 같은 다양한 데이터셋에서 기존의 최첨단(pre-state-of-the-art) 방법보다 각각 15%, 10%, 4%의 성능 향상을 보였습니다.



### ELSA: Exploiting Layer-wise N:M Sparsity for Vision Transformer Acceleration (https://arxiv.org/abs/2409.09708)
- **What's New**: 이 논문에서는 비전 트랜스포머(ViT) 모델에서 레이어별 맞춤형 $N{:}M$ 희소성(N:M sparsity) 구성을 얻기 위한 새로운 방법인 ELSA(Exploiting Layer-wise N:M Sparsity for ViTs)를 제안합니다. 이는 기존의 모든 레이어에 대해 균일 설정 또는 매개변수 수에 의해 히우리스틱(heuristic) 방법으로 결정하는 방식을 넘어서는 접근입니다.

- **Technical Details**: ELSA는 주어진 가속기에서 지원하는 모든 $N{:}M$ 희소성 수준과 기대되는 처리량 개선을 고려합니다. 이 방법론은 메모리 사용량과 추론 시간(inference time) 감소와 함께 미미한 정확도 손실(accuracy loss)을 교환하여 Mixed Sparsity를 지원하는 가속기의 이점을 활용합니다.

- **Performance Highlights**: 본 접근법은 Swin-B와 DeiT-B 모델 모두에서 ImageNet 데이터셋에 대해 2.9배의 FLOPs(부동 소수점 연산 수) 감소를 달성하였으며, 이는 정확도의 경미한 저하로 이어졌습니다. 이 연구 결과는 앞으로 가속기 지원을 통한 모델 최적화에 중요한 영향을 미칠 것으로 기대됩니다.



### Synergistic Spotting and Recognition of Micro-Expression via Temporal State Transition (https://arxiv.org/abs/2409.09707)
- **What's New**: 본 논문에서는 고전적인 윈도우 기반의 분류 방식을 비디오 수준의 회귀로 대체한 새로운 시간 상태 전이 아키텍처를 제안합니다. 이는 마이크로 표정(Micro-expressions, MEs) 분석에서 상태 공간 모델(state space model)을 활용한 최초의 연구로, 탐지(spotting)와 인식(recognition) 간의 상호작용을 강화하여 전체 분석 성능을 향상시킵니다.

- **Technical Details**: 제안된 방법론은 비디오 입력에서 관심 영역(ROIs)으로부터 옵티컬 플로우(optical flow)를 추출하여 특징을 초기화합니다. 이후 두 개의 경로로 나뉘어 각각 탐지와 인식을 수행하며, 최종적으로 이 두 결과를 상호작용하여 MEs의 분석 결과를 도출합니다. 이 과정에서 고정된 윈도우 크기로 인한 한계를 극복하기 위해 회귀 접근 방식을 사용하며, 비디오 길이에 대해 선형 복잡도를 유지합니다.

- **Performance Highlights**: 본 논문에서 제안하는 방법은 기존의 방법들에 비해 최첨단의 성능을 보여주며, 파라미터 수는 18K로 적고, 곱셈-누산 연산(Multiply-Accumulate Operations, MACs)은 1M에 불과합니다.



### E-Commerce Inpainting with Mask Guidance in Controlnet for Reducing Overcompletion (https://arxiv.org/abs/2409.09681)
- **What's New**: 이번 논문에서는 e-커머스 (e-commerce) 분야의 이미지 생성에서 핵심적인 문제인 과도한 재현(overcompletion)을 다룹니다.

- **Technical Details**: 우리는 두 가지 해결책을 제안합니다: 1. 인스턴스 마스크(instance mask)를 활용한 세밀한 인페인팅(inpainting) 모델을 사용하는 방법과, 2. ControlNet과 UNet을 결합할 때 개선된 제품 마스크를 제약조건으로 사용하는 Train-free (기차 없는) 마스크 가이던스(mask guidance) 접근 방식입니다.

- **Performance Highlights**: 우리의 방법은 실제 응용에서 유망한 결과를 달성하였으며, 이 분야에서 영감을 주는 기술 보고서가 되기를 바랍니다.



### A Comprehensive Methodological Survey of Human Activity Recognition Across Divers Data Modalities (https://arxiv.org/abs/2409.09678)
- **What's New**: 본 논문은 2014년부터 2024년까지의 Human Activity Recognition (HAR) 시스템에 대한 포괄적인 조사로, 머신 러닝(ML) 및 딥 러닝(DL) 접근 방식을 입력 데이터 모달리티에 따라 분류합니다. 다양한 데이터 모달리티(예: RGB 이미지, 비디오, 스켈레톤 등)를 활용한 최신 연구 동향을 분석하였습니다.

- **Technical Details**: 연구는 단일 모달리티 및 다중 모달리티 기술을 포함하며, 특히 융합 기반(fusion-based) 및 공동 학습(co-learning) 프레임워크에 대한 진전을 다룹니다. 또한 인간-객체 상호작용 인식 및 활동 탐지에 대한 방법도 포함됩니다. HAR에 사용되는 다양한 데이터셋에 대한 세부 설명도 제공됩니다.

- **Performance Highlights**: 최신 HAR 시스템의 요약 및 벤치마크 데이터셋에서의 비교 결과를 제시하였으며, HAR 분야에서의 효과적인 연구 방향을 제안하고 있습니다. 이 연구는 HAR의 발전과 동향을 이해하는 데 중요한 기초 자료를 제공합니다.



### SITSMamba for Crop Classification based on Satellite Image Time Series (https://arxiv.org/abs/2409.09673)
- **What's New**: 이번 연구에서는 원거리 탐사(Satellite Image Time Series, SITS) 데이터를 활용하여 작물 분류를 위한 새로운 방법인 Satellite Image Time Series Mamba (SITSMamba)를 제안했습니다. 기존의 SITS 분류 방법은 주로 작물 레이블에 의존하여 시간적 정보를 충분히 활용하지 못했습니다. SITSMamba는 CNN 기반의 공간 인코더와 Mamba 기반의 시간 인코더를 결합해 보다 풍부한 시간 정보를 활용합니다.

- **Technical Details**: SITSMamba는 두 개의 디코더 분기로 구성됩니다. 첫 번째 분기는 작물 분류 분기(CBranch)로, 여기서 ConvBlock을 사용하여 특징을 작물 맵으로 디코딩합니다. 두 번째 분기는 SITS 재구성 분기(RBranch)로, 선형 계층을 사용하여 인코딩된 특징을 원래의 입력값을 예측하는 데 사용됩니다. 또한, RBranch에 적용되는 위치 가중치(Positional Weight, PW)를 설계하였습니다.

- **Performance Highlights**: SITSMamba는 작물 분류에서 이전 방법들을 능가하는 성능을 보였습니다. 다중 작업 학습 프레임워크를 통해 CBranch와 RBranch 간의 학습 균형을 유지하며, RBranch의 보조 학습이 공간 및 시간 인코딩의 학습을 향상시키고, 결과적으로 작물 분류의 정확도를 개선합니다.



### Unsupervised Hyperspectral and Multispectral Image Blind Fusion Based on Deep Tucker Decomposition Network with Spatial-Spectral Manifold Learning (https://arxiv.org/abs/2409.09670)
Comments:
          Accepted by TNNLS 2024

- **What's New**: 이번 논문에서는 Tucker decomposition과 공간-스펙트럼 매니폴드 학습(spatial-spectral manifold learning)을 기반으로 하는 비지도 블라인드 퓨전 방법(DTDNML)을 제안합니다. 이를 통해 고해상도 다중 스펙트럼 이미지(HR-MSI)와 저해상도 하이퍼스펙트럼 이미지(LR-HSI)를 효과적으로 결합하여 고해상도 하이퍼스펙트럼 이미지(HR-HSI)를 생성하는 것을 목표로 합니다.

- **Technical Details**: 논문에서 제안한 방법은 깊은 Tucker 분해 네트워크(deep Tucker decomposition network)를 사용하여 LR-HSI와 HR-MSI를 일관된 특징 공간으로 매핑하고, 공유 매개변수를 가진 디코더를 통해 재구성을 수행합니다. 또한 공간-스펙트럼 주의 메커니즘(spatial spectral attention mechanism)을 도입하여 서로 다른 스케일에서의 특징 정렬 및 융합을 지원합니다. Laplacian 기반의 공간-스펙트럼 매니폴드 제약조건은 공유 디코더에서 전역 정보를 캡처하는 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 논문에서 제안한 DTDNML 방법은 네 가지 온도 측정 데이터셋에 대한 광범위한 실험을 통해 기존의 최신 방법에 비해 우수한 성능과 효율성을 보여주었습니다. 이 방법은 높은 정확도로 하이퍼스펙트럼 및 다중 스펙트럼 퓨전을 성공적으로 수행하였습니다.



### EditBoard: Towards A Comprehensive Evaluation Benchmark for Text-based Video Editing Models (https://arxiv.org/abs/2409.09668)
- **What's New**: 이 논문에서는 텍스트 기반 비디오 편집 모델에 대한 첫 번째 종합 평가 기준인 EditBoard를 제안합니다. 이는 9가지 자동 평가 지표를 포함하며, 비디오 편집 모델의 성능을 종합적으로 평가할 수 있는 방향을 제시합니다.

- **Technical Details**: EditBoard는 4개의 주요 차원에서 평가되며, 이에는 편집된 프레임과 원본 프레임 간의 충실도(fidelity), 타겟 프롬프트의 실행(success execution), 프레임 일관성(consistency), 그리고 스타일 평가(aesthetic quality)가 포함됩니다. 새로운 평가 지표로는 FF-α, FF-β, 및 Semantic Score가 있습니다.

- **Performance Highlights**: 이 평가 기준을 통해 5개의 최첨단 비디오 편집 모델을 평가하고, 각 모델의 강점과 약점을 명확히 하였습니다. 이 연구는 현재 모델의 이해도를 높이고 향후 연구 방향을 제시하는 중요한 기초 자료로 활용될 수 있습니다.



### SparX: A Sparse Cross-Layer Connection Mechanism for Hierarchical Vision Mamba and Transformer Networks (https://arxiv.org/abs/2409.09649)
Comments:
          Code will be publicly available at: this https URL

- **What's New**: 본 논문에서는 Mamba 기반 비전 모델을 위한 효율적인 교차 레이어 피처 집합 메커니즘인 SparX를 소개합니다. SparX는 인간 시각 시스템의 망막 신경절 세포(Retinal Ganglion Cells, RGCs)에서 영감을 받아 개발되었으며, 이로 인해 교차 레이어의 피처 상호작용과 재사용이 효과적으로 개선됩니다.

- **Technical Details**: SparX는 서로 다른 두 종류의 네트워크 레이어, 즉 신경절 레이어(ganglion layers)와 일반 레이어(normal layers)를 구축하여 설계되었습니다. 신경절 레이어는 높은 연결성과 복잡성을 가지고 여러 레이어의 피처 집합 및 상호작용을 가능하게 하며, 일반 레이어는 비교적 낮은 연결성과 복잡성을 지닙니다. 이러한 두 레이어를 겹쳐서 배열함으로써 SparX-Mamba라는 새로운 비전 네트워크 아키텍처가 설계되어서 모델 크기, 계산 비용, 메모리 비용, 정확도 간의 우수한 균형을 달성했습니다.

- **Performance Highlights**: SparX-Mamba는 VMamba-T에 비해 top-1 정확도를 82.5%에서 83.5%로 개선했으며, SparX-Swin-T는 Swin-T에 비해 1.3%의 top-1 정확도 향상을 보여줍니다. 실험 결과에 따르면, SparX-Mamba와 SparX-Swin은 기존 모델들에 비해 우수한 성능과 일반화 능력을 가지고 있습니다.



### A Novel Framework For Text Detection From Natural Scene Images With Complex Background (https://arxiv.org/abs/2409.09635)
- **What's New**: 이번 논문에서는 복잡한 배경에서 텍스트 영역을 탐지하기 위해 Wavelet Transform을 활용한 새로운 효율적인 방법을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 원본 이미지를 그레이스케일 형태로 변환한 후, Sub-band filtering을 통해 Wavelet 변환을 수행합니다. 이후에는 중심점을 기준으로 Region clustering 기법을 적용하여 각 영역에 Bounding box를 적합시켜 텍스트 영역을 식별합니다.

- **Performance Highlights**: 이 방법은 이전 방법들에 비해 더욱 정교하고 효율적이며, 특정 텍스트 글꼴 크기에 제한되지 않아 일반화된 접근이 가능합니다. 실험에 사용된 샘플 세트는 다양한 배경을 가진 50개의 이미지로 구성되어 있습니다.



### Can Large Language Models Grasp Event Signals? Exploring Pure Zero-Shot Event-based Recognition (https://arxiv.org/abs/2409.09628)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 이벤트 기반(advanced event-based) 비주얼 콘텐츠를 이해하는 능력을 탐구한 최초의 연구로, CLIP과 추가적인 교육 없이도 이벤트 기반 객체 인식을 수행할 수 있음을 보여줍니다. 특히 GPT-4o 및 4turbo와 같은 모델이 이벤트 기반 비주얼 콘텐츠를 직접 인식할 수 있는 능력을 평가했습니다.

- **Technical Details**: 이 논문에서는 이벤트 스트림을 이벤트 프레임(event frame) 또는 복원된 프레임(reconstructed frame) 형식으로 변환하여 LLMs에 입력합니다. 이벤트 스트림은 픽셀 위치와 폴라리티로 특징지어진 타임스탬프를 가진 이벤트로 구성되어 있으며, 이 데이터는 E2VID 또는 E2HQV와 같은 기술을 통해 시각적으로 이해 가능한 이미지로 재구성됩니다. 마지막으로 정확도(accuracy)를 평가 지표로 사용하여 다양한 모델의 인식 능력을 검증합니다.

- **Performance Highlights**: 실험 결과, GPT-4o는 N-ImageNet 데이터셋에서 가장 최신 방법들보다 다섯 배 더 높은 인식 정확도를 기록하며, 잘 설계된 프롬프트가 인식 성능을 크게 향상시킬 수 있음을 입증했습니다. LLMs의 성능은 각기 다른 이벤트 데이터 입력 표현의 영향을 체계적으로 평가하여 개선 가능성을 보여줍니다.



### Enhancing Weakly-Supervised Object Detection on Static Images through (Hallucinated) Motion (https://arxiv.org/abs/2409.09616)
- **What's New**: 본 연구는 정적인 이미지에서의 약한 지도 객체 탐지(Weakly-Supervised Object Detection, WSOD)를 위한 새로운 접근 방식을 제안합니다. 정적 이미지에서 환각된 동작 정보를 활용하여 WSOD 방법을 개선하는 방법론을 소개합니다.

- **Technical Details**: 이 방법은 Siamese 네트워크를 사용하여 동작을 통합하고, 동작 정규화를 통해 카메라의 움직임을 해결하고, 객체의 동작에 기반하여 이미지를 선택적으로 학습합니다. 또한 FlowNet 2.0을 활용하여 비디오의 연속 프레임 간의 광학적 흐름(optical flow)을 계산합니다.

- **Performance Highlights**: COCO 및 YouTube-BB 데이터셋에서 실험을 진행한 결과, 최첨단 방법 대비 성능이 향상되었음을 보여주었습니다. 정적인 이미지에서의 동작 정보가 객체 탐지 성능을 증대시키는 것으로 나타났습니다.



### Integrating Audio Narrations to Strengthen Domain Generalization in Multimodal First-Person Action Recognition (https://arxiv.org/abs/2409.09611)
- **What's New**: 본 논문에서는 첫 번째 시점 활동 인식을 위한 멀티모달 프레임워크를 제안하여 도메인 일반화를 개선하고자 합니다. 이 프레임워크는 모션(motion), 오디오(audio), 외관(appearance) 기능을 통합하여 다양한 환경에서의 도메인 이동에 대한 회복력을 분석합니다.

- **Technical Details**: 제안된 방법은 비디오 클립과 관련 오디오, 시각적 내레이션(visual narration), 오디오 내레이션(audio narration)을 쌍으로 구성된 훈련 샘플을 사용합니다. 서로 다른 모달리티에 대해 별도의 인코더를 훈련하여 각 모달리티의 기능을 추출하고, 오디오와 비주얼 간의 일관성(consistency)을 평가하여 훈련 중 예측에 미치는 영향을 최적화합니다.

- **Performance Highlights**: ARGO1M 데이터셋에서 최첨단 성과를 달성하여 보지 못한 시나리오와 위치에서도 효과적인 일반화를 보여주었습니다. 특히, 모션과 오디오 기능은 각각 25.8%, 32.7%의 성능 저하를 보인 반면, 외관 기능은 54.8%의 성능 저하를 기록하여, 도메인 이동에 견고한 특징을 강조합니다.



### TextureDiffusion: Target Prompt Disentangled Editing for Various Texture Transfer (https://arxiv.org/abs/2409.09610)
- **What's New**: 최근 텍스트 기반 이미지 편집 기술이 크게 발전했습니다. 그러나 기존 방법들은 나무나 금과 같은 단순한 질감만을 편집하는 데 제한적이었습니다. 복잡한 질감, 예를 들면 구름이나 불의 질감을 변경하는 데는 어려움이 있었습니다. 본 논문에서는 TextureDiffusion이라는 새로운 방법을 제안하여 텍스처 전송을 한층 향상시킵니다. 이 방법은 대상 프롬프트를 '<texture>'로 설정하여 질감을 입력 이미지의 내용과 분리합니다.

- **Technical Details**: TextureDiffusion 방법은 튜닝이 필요 없는 이미지 편집 방법으로, 질감 전송에 적용됩니다. 먼저, 목표 프롬프트를 '<texture>'로 수정하여 질감 표현을 자유롭게 하고, 그 다음 스스로 주의(self-attention)에서 질감 특성을 추출하여 잔여 블록(residual block) 특성과 결합하여 입력 이미지의 구조를 보존합니다. 마지막으로, 배경을 유지하기 위해 편집 위치화 기술(edit localization technique)을 도입합니다.

- **Performance Highlights**: 종합적인 실험 결과, TextureDiffusion은 다양한 질감을 조화롭게 전송하며, 구조와 배경을 우수하게 보존할 수 있음을 입증했습니다.



### DreamMover: Leveraging the Prior of Diffusion Models for Image Interpolation with Large Motion (https://arxiv.org/abs/2409.09605)
Comments:
          ECCV 2024

- **What's New**: 본 연구에서는 대규모 움직임이 있는 이미지 쌍으로부터 중간 이미지를 생성하는 문제를 다룹니다. 이를 위해 우리는 'DreamMover'라는 새로운 이미지 보간(framework)을 제안하며, 이는 세 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: 1) 자연 흐름 추정기(natural flow estimator): 두 이미지 간의 의미적 일치를 암묵적으로 추론하기 위해 확산 모델(diffusion model)을 기반으로 구축되었습니다. 2) 이미지 융합 시 세부정보 손실을 방지하기 위해 고급(high-level) 공간과 저급(low-level) 공간으로 나누어 정보를 융합합니다. 3) 생성된 이미지와 입력 간의 일관성을 높이기 위해 자기 주의 집합(self-attention concatenation and replacement) 접근 방식을 제안합니다.

- **Performance Highlights**: 우리의 방법은 최첨단 비디오 프레임 보간(video frame interpolation) 및 이미지 변형(image morphing) 방법들보다 뛰어난 성능을 보였습니다. 또한, 사용자 연구를 통해 인간의 시각에서도 우리의 방법이 우수함을 입증했습니다.



### One-Shot Learning for Pose-Guided Person Image Synthesis in the Wild (https://arxiv.org/abs/2409.09593)
- **What's New**: 본 논문에서는 Test-time fine-tuning(테스트 시 미세 조정) 패러다임을 채택하여 사전 훈련된 Text2Image (T2I) 모델을 맞춤화하는 새로운 방법을 제안합니다. 이 방법은 단일 소스 이미지만으로 높은 품질의 포즈 전송 결과를 생성할 수 있는 OnePoseTrans라는 이름의 접근 방식을 소개합니다.

- **Technical Details**: OnePoseTrans는 Visual Consistency Module (VCM)을 도입하여 얼굴 정체성 및 외관 속성의 일관성을 보장합니다. 우리는 ControlNet을 사용하여 포즈 이미지를 조건으로 T2I 모델을 조정하고, 이를 통해 다양한 시각적 속성을 결합하여 훈련된 모델의 일반화 가능성을 높입니다.

- **Performance Highlights**: OnePoseTrans는 NVIDIA V100 GPU를 사용하여 약 48초 만에 각 테스트 케이스에 대해 모델을 맞춤화하며, 전통적인 데이터 기반 방법들보다 더 높은 안정성을 제공합니다. 우리의 접근 방식은 특히 다양한 실제 이미지 도메인에서의 일반화 성능을 크게 향상시킵니다.



### GLCONet: Learning Multi-source Perception Representation for Camouflaged Object Detection (https://arxiv.org/abs/2409.09588)
Comments:
          Accepted at TNNLS 2024

- **What's New**: 본 논문에서는 GLCONet이라는 새로운 Global-Local Collaborative Optimization Network을 제안합니다. 이 네트워크는 지역적 세부정보와 전역적 장기 관계를 동시에 모델링하여 camouflaged object detection (COD) 작업의 정확도를 높이는 기능을 제공합니다.

- **Technical Details**: GLCONet은 multi-source perception 관점에서 협업 최적화 전략을 설계하여 캠플라주된 객체를 더 잘 감지합니다. 이 네트워크는 multi-scale transformer block (MTB)을 포함하는 global perception module (GPM)과 지역 세부정보를 추출하는 local refinement module (LRM)을 포함합니다. 또한, group-wise hybrid interaction module (GHIM)을 통해 다양한 정보를 통합하여 기능 표현 능력을 강화합니다.

- **Performance Highlights**: GLCONet은 세 가지 공공 COD 데이터세트에서 20개 이상의 최신 방법들보다 우수한 성능을 발휘하며, 다양한 backbone 네트워크를 통해 실험적으로 검증되었습니다. 이는 camouflaged objects 감지의 정확도를 크게 향상시킵니다.



### NEVLP: Noise-Robust Framework for Efficient Vision-Language Pre-training (https://arxiv.org/abs/2409.09582)
- **What's New**: 본 논문에서는 노이즈에 강한 비전-언어 전처리 프레임워크인 NEVLP를 제안합니다. 이 프레임워크는 기존의 대규모 웹 데이터 세트를 사용할 필요 없이 효과적으로 모델을 학습할 수 있도록 설계되었습니다.

- **Technical Details**: NEVLP는 두 가지 혁신적인 학습 전략인 노이즈 적응 학습(noise-adaptive learning)과 개념 강화 학습(concept-enhanced learning)을 도입합니다. 노이즈 적응 학습에서는 이미지-텍스트 쌍의 노이즈 확률을 추정하고, 이를 바탕으로 이미지-텍스트 대조 학습(image-text contrastive learning)에 적용하여 크로스 모달 정렬(cross-modal alignment)을 조정합니다. 개념 강화 학습은 불완전한 텍스트를 시각적 개념으로 보강하여 이미지-텍스트 매칭(image-text matching)과 이미지 주체 텍스트 생성(image-grounded text generation)의 효율성을 높입니다.

- **Performance Highlights**: NEVLP는 노이즈가 포함된 웹 데이터를 효과적으로 활용하여 매우 적은 전처리 데이터로도 뛰어난 성능을 발휘합니다. 또한, 이미지-텍스트 검색(image-text retrieval), 이미지 캡셔닝(image captioning), 시각적 질문 답변(visual question answering)과 같은 다양한 비전-언어 작업에서 최첨단 성능을 달성했습니다.



### Learning Transferable Features for Implicit Neural Representations (https://arxiv.org/abs/2409.09566)
- **What's New**: 새로운 연구에서는 Implicit Neural Representations(INRs)의 전이 가능성에 대한 탐구가 진행되었습니다. STRAINER라는 새로운 프레임워크가 소개되어, 학습된 특징을 활용하여 새로운 신호에 대한 적합성을 빠르고 높은 품질로 개선합니다.

- **Technical Details**: 이 연구에서 STRAINER는 입력 좌표를 특징으로 매핑하는 '인코더'와, 그러한 특징을 출력값으로 매핑하는 '디코더'로 INRs를 구분합니다. 인코더는 여러 훈련 신호를 통해 학습되며 각 신호마다 독립적인 디코더를 사용합니다. 테스트 시에는 훈련된 인코더가 새 INR의 초기화로 사용됩니다.

- **Performance Highlights**: STRAINER는 같은 도메인 내 이미지를 적합할 때 매우 강력한 초기화를 제공하며, 훈련되지 않은 INR에 비해 약 +10dB의 신호 품질 개선을 보여줍니다. 또한, 다양한 문제에서 STRAINER의 특징이 전이 가능함을 입증하며, 그 성능을 여러 신호 적합 작업과 역문제에서 평가하였습니다.



### TG-LLaVA: Text Guided LLaVA via Learnable Latent Embeddings (https://arxiv.org/abs/2409.09564)
- **What's New**: 논문에서 제안된 Text Guided LLaVA (TG-LLaVA)는 텍스트로 시각 인코더를 유도하여 비전-언어 모델(VLM)을 최적화하는 새로운 접근 방식을 제시합니다. 이는 기존의 방법들이 커넥터나 언어 모델 컴포넌트 개선에 집중했던 것과는 정반대의 접근입니다.

- **Technical Details**: TG-LLaVA 아키텍처는 두 개의 주요 모듈인 text-guided feature optimization mask (TG-FOM) 모듈과 text-guided detail perceiver (TG-DP) 모듈로 구성됩니다. TG-FOM 모듈은 입력 텍스트를 전역적으로 분석하여 언어 정보를 이미지 특성에 추가하는 역할을 합니다. TG-DP 모듈은 세밀한 분석을 통해 텍스트를 파싱하고, 이미지 관점에서 정보를 융합하는 데 사용됩니다.

- **Performance Highlights**: TG-LLaVA는 여러 데이터셋에서 기존 기준 모델인 LLaVA-1.5에 비해 성능 향상을 보여주었으며, 추가적인 데이터 없이도 경쟁력 있는 결과를 구출하였습니다. 실험 결과는 TG-LLaVA의 효과성을 확고히 하였으며 다양한 설정에서도 일관된 향상이 있음을 입증했습니다.



### Evaluating authenticity and quality of image captions via sentiment and semantic analyses (https://arxiv.org/abs/2409.09560)
- **What's New**: 이 연구는 COCO-MS 데이터셋을 사용하여 인간 생성 캡션의 감정 분석과 의미 변동성을 평가하는 새로운 방법을 제안합니다. 또한, 모델이 생성한 캡션의 감정 점수와 비교하여 훈련 데이터의 품질을 평가하는 접근 방법을 보여줍니다.

- **Technical Details**: 약 150K 이미지로 구성된 COCO-MS 데이터셋에서 각각의 이미지에 대해 4-6개의 캡션이 수집되었습니다. 감정 분석은 사전 훈련된 모델인 Twitter-RoBERTa-base를 사용하였고, BERT 임베딩을 사용하여 의미의 변동성을 분석했습니다. 회귀 분석(multiple linear regression)을 통해 감정 점수와 객체 카테고리의 관계를 조사했습니다.

- **Performance Highlights**: 대부분의 캡션은 중립적인 감정을 지닌 것으로 나타났으며, 강한 감정이 있는 캡션은 약 6%로 특정 객체 카테고리에 의해 영향을 받았습니다. 모델이 생성한 캡션의 강한 감정 비율은 1.5%에 불과하여 인간 생성 캡션과의 상관관계가 없었습니다.



### An Augmentation-based Model Re-adaptation Framework for Robust Image Segmentation (https://arxiv.org/abs/2409.09530)
Comments:
          Accepted in the European Conference on Computer Vision (ECCV) 2024 workshop

- **What's New**: 본 논문은 산업 검사에 특히 상업용 위조 방지 코드를 분할하는 데 있어 Segment Anything Model (SAM)의 적용이 어려움을 겪고 있음을 지적하고, 이를 해결하기 위한 Augmentation-based Model Re-adaptation Framework (AMRF)를 제안합니다.

- **Technical Details**: AMRF는 훈련 중 데이터 증강(data augmentation) 기술을 활용하여 분할 모델의 일반화를 향상시키고, 최근 출시된 데이터셋에 적응할 수 있도록 합니다. FCN과 U-Net으로부터 관찰한 분할 마스크를 바탕으로 효율성과 모델 성능을 최적화하는 최소한의 증강 세트를 결정합니다.

- **Performance Highlights**: Fine-tuned FCN 모델은 두 개의 시간 계속 데이터셋에서 각각 3.29%와 3.02%의 크롭 정확도 향상 및 5.27%와 4.04%의 분류 정확도 향상을 보였고, fine-tuned U-Net는 각각 7.34%와 4.94%의 크롭 향상, 8.02%와 5.52%의 분류 향상을 나타냈습니다. 두 모델 모두 SAM 모델(ViT-Large 및 ViT-Base)보다 평균 각각 11.75% 및 9.01%의 크롭 정확도와 2.93% 및 4.83%의 분류 정확도를 초과했습니다.



### Enhancing Skin Disease Diagnosis: Interpretable Visual Concept Discovery with SAM Empowermen (https://arxiv.org/abs/2409.09520)
- **What's New**: 본 연구에서는 피부 병변 진단을 위한 새로운 Cross-Attentive Fusion 프레임워크를 제안합니다. 이는 Segment Anything Model (SAM)을 활용하여 피부 질병에 대한 비주얼 개념을 생성하고, 지역적 시각 개념과 글로벌 이미지 특징을 통합하여 진단 성능을 향상시킵니다.

- **Technical Details**: SAM(Segment Anything Model)은 11백만 개의 라이센스 이미지로부터 1억 개 이상의 마스크를 추출하여 학습된 프롬프트 기반(segmentation) 모델입니다. 이 모델은 단일 포인트, 포인트 집합, 바운딩 박스 또는 텍스트와 같은 여러 가지 프롬프트를 지원합니다. SAM의 구조는 이미지 인코더, 프롬프트 인코더, 마스크 디코더로 구성되어 있으며, 특히 Masked Autoencoder(MAE)와 Vision Transformer(ViT)를 이용하여 높은 해상도의 입력 처리와 세부 정보 캡처를 가능하게 합니다.

- **Performance Highlights**: 두 개의 피부 질환 데이터셋에 대한 광범위한 평가 결과, 제안된 방법이 병변 진단과 해석 가능성에서 효과적임을 입증했습니다. 이 연구는 실제 사례에서의 적용 가능성과 신뢰도를 높이는 데 기여합니다.



### One missing piece in Vision and Language: A Survey on Comics Understanding (https://arxiv.org/abs/2409.09502)
Comments:
          under review. project website: this https URL

- **What's New**: 본 논문은 Comics Understanding을 위한 새로운 프레임워크인 Layer of Comics Understanding (LoCU)을 소개하며, 이 프레임워크를 통해 비주얼-언어 모델을 사용한 기초적인 작업 정의의 필요성을 강조하고 있습니다.

- **Technical Details**: 이 논문에서는 Comics의 독특한 구조와 다양성을 분석하고, 이미지 분류, 객체 탐지, 시맨틱 분할, 고유한 내러티브 이해 등 다양한 비주얼-언어 작업을 위한 데이터를 체계적으로 조사합니다. LoCU 프레임워크는 비주얼-언어 과제를 계층적으로 정리하고, 기존 모델과의 비교를 통해 그 한계를 극복하고자 합니다.

- **Performance Highlights**: 이 연구는 Comics Understanding을 위한 선도적인 작업 목록을 제공하며 데이터 가용성 및 과제 정의에서 발생하는 주요 격차를 발견했습니다. 또한, 비주얼-언어 모델의 발전과 함께 Comics 분야의 성장 가능성을 제시하고 있으며, 향후 연구 방향에 대해서도 논의합니다.



### Multi-Scale Grouped Prototypes for Interpretable Semantic Segmentation (https://arxiv.org/abs/2409.09497)
Comments:
          8 pages, 5 figures, 4 tables

- **What's New**: 본 논문은 multi-scale 프로토타입(part) 학습을 활용한 해석 가능한 의미론적 분할 방법인 ScaleProtoSeg를 제안합니다. 이 방법은 서로 다른 스케일에서 다양한 프로토타입 부분을 명시적으로 학습하고, 이들 사이의 상호작용에 대한 정보를 제공하는 희소 집합(sparse grouping) 메커니즘을 응용합니다.

- **Technical Details**: 제안된 ScaleProtoSeg는 (i) 여러 스케일에서 프로토타입을 명시적으로 학습하고 (ii) 스케일 특정 프로토타입들의 희소 조합(sparse combination)을 학습하는 집합 절차(grouping procedure)를 정의합니다. 이 방식은 해석 가능성을 증가시키며, 결정 과정에서 소수의 프로토타입으로 제약을 두어 모델의 성능을 유지합니다.

- **Performance Highlights**: Pascal VOC, Cityscapes, ADE20K 세 가지 벤치마크 데이터셋에서 제안된 ScaleProtoSeg 방법이 기존의 프로토타입 기반 방법보다 우수한 성과를 보였으며, 해석 가능성 측면에서도 안정성(stability), 일관성(consistency), 희소성(sparsity)으로 측정된 개선된 결과를 보였습니다.



### Learning Keypoints for Multi-Agent Behavior Analysis using Self-Supervision (https://arxiv.org/abs/2409.09455)
- **What's New**: 본 연구에서는 개체 식별이 어려운 다수의 상호작용하는(agent) 동물 비디오에서 키포인트(keypoint)를 자율적으로 발견하는 새로운 방법 B-KinD-multi를 소개합니다. 이 방법은 사전 훈련된 비디오 분할 모델을 활용하여 키포인트 발견 프로세스를 가이드함으로써 수작업 주석의 필요성을 제거합니다.

- **Technical Details**: B-KinD-multi는 동물의 이미지 분할을 위한 DEVA(Decoupled Video Segmentation) 프레임워크를 도입하여, 시각적으로 유사한 여러 개체를 효과적으로 구분하고 각 개체에 대한 키포인트를 독립적으로 발견합니다. 이 과정에서는 스페이셜(spatial) 정렬과 표현(representation), 정수 프로그래밍(integer programming) 등의 기법을 적용하여 최적의 개체 제안 집합을 선택하고, 개체의 움직임과 위치를 추론합니다.

- **Performance Highlights**: B-KinD-multi는 초파리, 쥐, 쥐의 비디오에서 키포인트 회귀(keypoint regression)와 행동 분류(behavioral classification) 성능을 개선하였고, 개미, 벌, 인간과 같은 다른 종에도 잘 일반화되는 특성을 보여줍니다. 이는 자동화된 다중 에이전트 행동 분석에 대한 잠재력을 강조합니다.



### On the Generalizability of Foundation Models for Crop Type Mapping (https://arxiv.org/abs/2409.09451)
- **What's New**: 이 논문은 다양한 농업 환경에서 지리적 위치에 대한 일반화 능력을 연구하며, 전 세계에서 이용 가능한 농작물 분류 데이터 세트를 통합하여 모델의 전이 학습(transfer learning) 가능성을 높이려는 노력을 보여줍니다.

- **Technical Details**: 연구진은 전 세계 5개 대륙에 걸쳐 6개의 농작물 분류 데이터 세트를 조화시켰습니다. 크기 및 클래스를 통합하여 옥수수(maize), 대두(soybean), 쌀(rice), 그리고 밀(wheat)의 4대 주요 곡물에 초점을 맞췄습니다. SSL4EO-S12, SatlasPretrain, ImageNet을 기반으로 한 3가지 기초 모델의 성능을 ID 및 OOD 평가를 통해 비교하여, Sentinel-2에 맞추어진 사전 훈련 가중치가 더 일반적인 사전 훈련 가중치보다 우수함을 보여주었습니다.

- **Performance Highlights**: 이 연구에서는 OOD 데이터로 사전 훈련(pre-training) 시, ID 훈련 샘플이 10~100개일 때 가장 큰 성능 향상을 보였으며, 개발 도상국의 농작물 레이블 부족 문제 해결에 대한 가능성을 제시합니다. 모든 데이터 세트와 코드가 오픈 소스로 제공되어 다른 연구자들이 재현할 수 있도록 하였습니다.



### MulCPred: Learning Multi-modal Concepts for Explainable Pedestrian Action Prediction (https://arxiv.org/abs/2409.09446)
- **What's New**: 본 논문에서는 pedestrian action prediction을 향상시키기 위한 새로운 프레임워크인 MulCPred를 제안합니다. MulCPred는 다중 모달 (multi-modal) 개념을 기반으로 하는 예측 설명을 제공합니다. 이전 방법들이 가진 한계를 극복하며, pedestrian 행동 예측의 신뢰성을 높이는 데 기여합니다.

- **Technical Details**: MulCPred는 입력을 여러 개념에 투영하고, 각 개념은 입력과의 유사성을 나타내는 활성화 점수를 가집니다. 이 시스템은 작동의 한 부분으로서 channel-wise recalibration module을 사용하여 지역적 spatiotemporal 영역에 주목합니다. 또한, 다양한 패턴 학습을 장려하는 feature regularization loss를 포함하고 있습니다.

- **Performance Highlights**: MulCPred는 다양한 데이터셋과 작업에서 평가되어, qualitative 및 quantitative 결과 모두에서 pedestrian action prediction의 설명 가능성을 개선했으며, 성능 저하 없이도 이러한 효과를 달성했습니다. 더 나아가, 인식할 수 없는 개념을 제거함으로써.cross-dataset prediction 성능이 향상되었습니다.



### KAN-HyperpointNet for Point Cloud Sequence-Based 3D Human Action Recognition (https://arxiv.org/abs/2409.09444)
- **What's New**: 본 논문에서는 3D 행동 인식을 위한 새로운 방법론인 KAN-HyperpointNet을 제안합니다. 이 방법은 D-Hyperpoint라는 새로운 데이터 유형을 도입하여 미세한 팔다리 움직임과 전반적인 자세 구조를 모두 고려하는 점에서 혁신적입니다.

- **Technical Details**: KAN-HyperpointNet은 D-Hyperpoint Embedding 모듈과 D-Hyperpoint KANsMixer 모듈로 구성되어 있습니다. D-Hyperpoint는 지역적인 순간 동작과 전반적인 정적 자세를 통합하여 각 순간의 인간 행동을 요약합니다. 또한 KANsMixer 모듈은 Kolmogorov-Arnold Networks(KAN)를 활용하여 D-Hyperpoint 간의 시공간 상호작용을 강화합니다.

- **Performance Highlights**: MSR Action3D 및 NTU-RGB+D 60 데이터셋에 대한 실험 결과, KAN-HyperpointNet은 기존 방법들과 비교하여 최첨단 성능을 발휘하며, 메모리와 계산 효율성 또한 눈에 띄게 향상되었습니다.



### Detecting Looted Archaeological Sites from Satellite Image Time Series (https://arxiv.org/abs/2409.09432)
- **What's New**: 이번 연구에서는 아프가니스탄의 고고학적 유적지 보호를 위한 첫 번째 공개 다중시계열 원거리 감지 데이터셋인 DAFA Looted Sites dataset (DAFA-LS)를 소개합니다. 이 데이터셋은 675개의 아프가니스탄 고고학 유적지에서 매월 수집된 총 55,480개의 이미지를 포함하며, 이 중 135개는 약탈된 사이트입니다.

- **Technical Details**: DAFA-LS는 위성 이미지 시계열(Satellite Image Time Series, SITS)로 구성되어 있으며, 시간에 따른 이미지를 통해 '약탈된' 사이트와 '보존된' 사이트를 구분합니다. 이 데이터셋은 약탈 감지 작업을 위해 SITS 분류 방법을 평가할 수 있는 최초의 공개 데이터셋으로, 클래스 불균형과 제한된 학습 샘플 수, 미세한 변화 감지의 어려움이 도전 과제로 제시됩니다.

- **Performance Highlights**: 실험 결과, 기초 모델(Foudation Models)을 활용할 경우 성능을 크게 향상시킬 수 있으며, 단일 이미지 대신 전체 시간 시퀀스를 사용함으로써 추가적인 성능 향상이 가능함을 보여줍니다.



### Evaluating Pre-trained Convolutional Neural Networks and Foundation Models as Feature Extractors for Content-based Medical Image Retrieva (https://arxiv.org/abs/2409.09430)
Comments:
          29 pages

- **What's New**: 본 연구에서는 미리 훈련된 합성곱 신경망(CNN)과 기초 모델들을 활용하여 다양한 유형의 의료 영상을 기반으로 한 콘텐츠 기반 의료 이미지 검색(CBMIR) 성능을 분석하였다. 또한, 이미지 크기가 CBMIR 성능에 미치는 영향을 조사하였다.

- **Technical Details**: 연구는 MedMNIST V2 데이터셋의 부분집합을 사용하여 2D 및 3D 의료 이미지에서 CBMIR 성능을 평가하였다. 여러 종류의 미리 훈련된 CNN(VGG19, ResNet-50 등)과 기초 모델(MedCLIP, BioMedCLIP, OpenCLIP 등)을 적용하여 영상을 자동적으로 특징 추출하였다. 또한, 이미지 크기(28x28, 224x224 등)에 따른 성능 변화를 조사하였다.

- **Performance Highlights**: 결과에 따르면 2D 데이터셋의 경우, 기초 모델들이 CNN에 비해 월등한 성능을 보였다. UNI 모델이 전체 데이터셋과 이미지 크기에서 가장 우수한 성능을 나타냈으며, 3D 데이터셋에서는 CONCH 모델이 최상의 성능을 달성하였다. 큰 이미지 크기를 사용했을 때 성능이 약간 개선되었지만, 작은 이미지 크기로도 경쟁력 있는 CBMIR 성능을 얻을 수 있었다.



### NBBOX: Noisy Bounding Box Improves Remote Sensing Object Detection (https://arxiv.org/abs/2409.09424)
- **What's New**: 이번 논문은 원거리 감지(remote sensing)에서 오브젝트 탐지(object detection) 성능 향상을 위해 bounding box의 변형(transformation) 방법을 제안합니다. 특히, 기존의 이미지 조정(image adjustments) 방식이 아닌 bounding box의 변화에 초점을 맞춘 NBBOX(Noise Injection into Bounding Box)라는 데이터 증강(data augmentation) 기법을 소개합니다.

- **Technical Details**: NBBOX 방법은 스케일링(scaling), 회전(rotation), 변환(translation)과 같은 세 가지 기하학적 변환을 사용하여 라벨 노이즈(label noise)가 있는 오리엔테이션(bounding box)에 노이즈를 추가하는 방식입니다. 이 방법은 기존의 감지 프레임워크에 쉽게 통합될 수 있으며, DOTA와 DIOR-R와 같은 데이터셋에서 실험이 진행되었습니다.

- **Performance Highlights**: 실험 결과, NBBOX 방법이 원거리 감지의 성능을 크게 향상시키는 것으로 나타났으며, 다른 최첨단(data augmentation strategies) 기법들과 비교하여 시간이 더 효율적임을 보여주었습니다.



### Label Convergence: Defining an Upper Performance Bound in Object Recognition through Contradictory Annotations (https://arxiv.org/abs/2409.09412)
- **What's New**: 본 연구에서는 'label convergence'라는 개념을 도입하여, 상충하는 테스트 주석으로 인한 성능 한계를 설명합니다. 이는 모델 정확도의 상한선을 정의하는 것으로, 95% 신뢰 구간에서 LVIS 데이터셋의 label convergence가 62.63-67.52 mAP@[0.5:0.95:0.05]로 추정됩니다.

- **Technical Details**: 연구에서는 다섯 개의 실제 데이터셋, 특히 LVIS 데이터셋을 분석하여 label convergence 현상을 조사합니다. 라벨이 금본위(gold standard)로 간주되었던 복잡한 데이터의 경우, 완벽한 기준 진실(ground truth)은 종종 달성할 수 없는 목표가 됩니다. 각 주석가의 변동성, 주석 관행의 차이 등이 label 불일치의 원인일 수 있습니다.

- **Performance Highlights**: 현재 최첨단(SOTA) 모델들은 LVIS 데이터셋의 label convergence 구간 상단에 있으며, 이는 현재 물체 탐지 문제를 해결하기 위한 모델의 용량이 충분함을 의미합니다. 향후 연구는 (1) 주석 잡음(label noise)을 고려한 평가 관행 수정, (2) 더 깨끗한 데이터 생성, (3) 주석 변동성을 조사하기 위한 다중 주석 데이터 포함에 집중해야 합니다.



### Real-world Adversarial Defense against Patch Attacks based on Diffusion Mod (https://arxiv.org/abs/2409.09406)
- **What's New**: DIFFender는 텍스트-유도 확산 모델을 활용하여 적대적 패치 공격에 대응하는 혁신적인 방어 프레임워크입니다. 또한 AAP(Adversarial Anomaly Perception) 현상을 발견하여 패치의 정확한 감지 및 위치 지정을 가능하게 합니다.

- **Technical Details**: DIFFender는 적대적 패치를 정밀하게 감지하기 위해 여러 دenoised(디노이즈된) 이미지 사이의 분포적 차이를 분석합니다. 이 프레임워크는 패치 로컬라이제이션(patch localization)과 복원(restoration) 작업을 통합된 확산 모델( Diffusion Model) 내에서 수행하며, 효율적인 few-shot prompt-tuning 알고리즘을 사용하여 모델 적응을 촉진합니다.

- **Performance Highlights**: DIFFender는 이미지 분류 및 얼굴 인식 작업에서 우수한 방어 성능을 보여주었으며, IR(Infrared, 적외선) 도메인으로의 확장 가능성을 통해 다양한 공격 방법론에 대한 일반화 성능이 뛰어납니다.



### AI-Driven Virtual Teacher for Enhanced Educational Efficiency: Leveraging Large Pretrain Models for Autonomous Error Analysis and Correction (https://arxiv.org/abs/2409.09403)
- **What's New**: 본 논문에서는 VATE(Virtual AI Teacher)라는 혁신적인 시스템을 제안하여, 학생들이 수학 문제를 해결할 때 발생하는 오류를 자율적으로 분석하고 수정하는 방안을 제공합니다. 이 시스템은 학생의 초안을 주요 분석 자료로 활용하여 학습 과정을 깊이 이해하고, 실시간 대화 기능을 통해 학생과의 상호작용을 원활하게 진행합니다.

- **Technical Details**: VATE 시스템은 고급 자연어 처리(NLP) 모델인 대형 언어 모델(LLMs)을 활용합니다. 이 시스템은 오류 분석을 위해 복잡한 프롬프트 엔지니어링을 통합하고, 오류 풀(error pool)을 유지하여 계산적 부담을 줄입니다. 시스템은 학생 초안의 이미지와 다중 모드 데이터를 사용하여 오류를 정확하게 로컬라이징 및 분석할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: 학생들의 응답에 대한 오류 분석 정확도가 78.3%에 달하며, 학생들의 학습 효율성이 눈에 띄게 향상되었습니다. 또한, 판매 직원 대상으로 실시한 만족도 조사에서 10점 만점에 8점 이상의 긍정적인 피드백을 얻어, 교육 방식의 혁신 가능성을 입증하였습니다.



### Tran-GCN: A Transformer-Enhanced Graph Convolutional Network for Person Re-Identification in Monitoring Videos (https://arxiv.org/abs/2409.09391)
- **What's New**: 이 논문에서는 다양한 카메라에서 일어나는 보행자 인식을 개선하기 위해 Transformer-enhanced Graph Convolutional Network (Tran-GCN) 모델을 제안합니다. 대부분의 기존 방법이 지역적 특성 간의 관계를 간과하면서 보행자의 자세 변화와 국소 신체 부분의 occlusion을 제대로 처리하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: Tran-GCN 모델은 네 가지 주요 구성 요소로 이루어져 있습니다: (1) 자세 추정 학습 브랜치는 보행자의 자세 정보를 추정하여 주요 키 포인트 정보를 추출합니다; (2) Transformer 학습 브랜치는 세부적이고 의미 있는 지역 특성 간의 글로벌 의존성을 학습합니다; (3) 합성곱 학습 브랜치는 기본 ResNet 아키텍처를 사용하여 세부 지역 특성을 추출합니다; (4) 그래프 합성곱 모듈(GCM)은 지역 특성, 글로벌 특성 및 신체 정보 통합합니다.

- **Performance Highlights**: Market-1501, DukeMTMC-ReID, MSMT17와 같은 세 가지 데이터셋에서의 정량적 및 정성적 분석 실험 결과, Tran-GCN 모델이 감시 영상에서 판별 가능한 사람 특성을 더 정확하게 포착하며 인식 정확도를 크게 향상시킨 것으로 나타났습니다.



### AMBER -- Advanced SegFormer for Multi-Band Image Segmentation: an application to Hyperspectral Imaging (https://arxiv.org/abs/2409.09386)
Comments:
          submitted to Neural Computing & Applications (Springer). Currently under review

- **What's New**: 이번 논문에서는 다중밴드 이미지 분할을 위한 고급 SegFormer 모델인 AMBER를 새롭게 소개합니다. AMBER는 삼차원 합성곱(three-dimensional convolutions)을 통합하여 하이퍼스펙트럴 데이터(hyperspectral data)를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: AMBER는 하이히어라키 Transformer 인코더와 가벼운 All-MLP 디코더로 구성됩니다. 이 인코더는 높은 해상도의 조잡한 특징과 낮은 해상도의 세부 특징을 생성합니다. 입력 이미지를 3x3x3 패치로 나누고, 이 패치를 인코더에 입력하여 다중 수준 특징을 얻습니다. 마지막 디코더 단계에서는 이 다중 레벨 특징을 융합하여 이차원 의미 분할 마스크를 생성합니다.

- **Performance Highlights**: AMBER는 Indian Pines, Pavia University, PRISMA 데이터셋에서 전통적인 CNN 기반 방법들과 비교하여 전체 정확도(Overall Accuracy), 카파 계수(Kappa coefficient), 평균 정확도(Average Accuracy)에서 월등한 성능을 보였으며, PRISMA 데이터셋에서는 최첨단 성과를 기록했습니다.



### Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology (https://arxiv.org/abs/2409.09369)
Comments:
          24 pages, 11 tables, 6 figures

- **What's New**: 이번 연구에서는 비전-언어 생존 분석 (Vision-Language Survival Analysis, VLSA)라는 새로운 패러다임을 제안합니다. 이는 기존의 생존 분석법이 직면한 데이터 부족 및 약한 감독 학습의 한계를 극복하도록 설계되었습니다.

- **Technical Details**: VLSA는 병리학적 비전-언어(VL) 기초 모델을 기반으로 하며, 고성능 네트워크에 의존하지 않고 데이터 효율성이 높은 점이 특징입니다. 비전 측면에서는 프로그노스틱(예후) 언어 우선순위를 인코딩하고, 이를 보조 신호로 사용하여 다중 인스턴스 수준에서 프로그노스틱 비주얼 특징을 집계하는 방법을 사용합니다. 또한 연속적인 생존 레이블을 텍스트 프롬프트로 변환하는 방법과 예측 타겟으로서의 순서화된 발생 함수도 제안합니다.

- **Performance Highlights**: 다섯 가지 데이터셋에 대한 실험 결과, VLSA는 기존 기술보다 더 뛰어난 성능을 기록했으며, 적은 계산 비용으로 최신 기술 수준의 성능을 달성했습니다. VLSA는 약한 감독의 다중 인스턴스 학습을 위한 효과적인 방법으로서 CPATH에서의 생존 분석을 위한 새로운 길을 열 가능성을 보입니다.



### MHAD: Multimodal Home Activity Dataset with Multi-Angle Videos and Synchronized Physiological Signals (https://arxiv.org/abs/2409.09366)
- **What's New**: 이번 논문에서는 가정에서 비접촉으로 생리 신호를 모니터링할 수 있는 새로운 데이터셋, MHAD를 소개합니다. 이 데이터셋은 40명의 피험자로부터 촬영된 1,440개의 비디오를 갖추고 있으며, 다양한 각도에서 촬영된 6가지 일반적인 가정 활동을 포함하고 있습니다.

- **Technical Details**: MHAD 데이터셋은 6가지 생리 신호(호흡, PPG, ECG, SpO2, 혈압)를 기록하며, 이를 통해 다양하고 포괄적인 생리 신호 데이터를 제공합니다. 비디오 촬영은 2미터 거리에서 3개의 USB 카메라를 사용하여 동기화되어 진행되었습니다.

- **Performance Highlights**: 기존 최첨단 방법을 평가한 결과, 비디오의 각도와 활동에 따라 성능이 달라졌습니다. POS 방법은 TV 시청 중에서 상대적으로 정확도 높은 결과를 보였으나, 다른 활동에서는 평균 절대 오차(MAE)가 10 이상으로 나타났습니다.



### LACOSTE: Exploiting stereo and temporal contexts for surgical instrument segmentation (https://arxiv.org/abs/2409.09360)
Comments:
          Preprint submitted to Medical Image Analysis

- **What's New**: 이 논문에서는 Location-Agnostic COntexts in Stereo and TEmporal images를 활용한 새로운 LACOSTE 모델을 제안하며, 수술 영상에서의 기구 분할 성능을 향상시킵니다. 기존의 단일 프레임 기반 접근법에서 벗어나, 시계열적 특성과 스테레오 특성을 고려한 쿼리 기반 세분화 접근 방식을 사용합니다.

- **Technical Details**: LACOSTE 모델은 다음 세 가지 성능 향상 모듈을 포함합니다: 1) disparity-guided feature propagation (DFP) 모듈로 깊이 인식 기능을 향상시키고, 2) stereo-temporal set classifier (STSCls)를 통해 시계열 및 스테레오 컨텍스트를 통합하여 예측의 정확성을 높이며, 3) location-agnostic classifier (LACls)를 사용하여 위치 편향을 완화합니다. 이러한 구성 요소들은 다양한 수술 비디오 데이터셋에서의 일반화 능력을 높이도록 설계되었습니다.

- **Performance Highlights**: LACOSTE 모델은 EndoVis Challenge의 두 가지 벤치마크와 실제의 radical prostatectomy 수술 데이터셋 GraSP를 포함한 세 개의 공개 수술 비디오 데이터셋에서 성능을 평가받았으며, 기존의 최신 기술(SOTA) 접근 방식들과 비교해 동등하거나 우수한 결과를 달성했습니다.



### OPUS: Occupancy Prediction Using a Sparse S (https://arxiv.org/abs/2409.09350)
- **What's New**: 이번 연구에서는 기존의 밀집한 데이터 표현 대신, 점유 예측(occupancy prediction)을 직접 집합 예측(set prediction) 문제로 설정하는 새로운 접근 방식을 소개합니다. 이를 통해 복잡한 3D 공간 모델링이나 희소화(sparsification) 절차 없이, 점유된 위치와 클래스 예측을 동시에 수행할 수 있는 OPUS라는 프레임워크를 제안합니다.

- **Technical Details**: OPUS는 Transformer encoder-decoder 아키텍처를 활용하여, 다중 뷰 이미지로부터 2D 특징을 추출하고, 학습 가능한 쿼리를 통해 점유된 위치와 의미적 클래스를 동시에 예측합니다. 세미틱 클래스는 학습된 위치에 따라 최근접 이웃 탐색을 통해 적응적으로 할당됩니다. 이 과정에서 Chamfer distance loss를 사용하여 모델 훈련을 엔드 투 엔드(End-to-End)로 수행합니다.

- **Performance Highlights**: 저자들은 OPUS가 Occ3D-nuScenes 데이터셋에서 기존의 최신 방법들보다 뛰어난 RayIoU 성능을 보이며, 가장 경량화된 모델이 2배 이상의 FPS로 운영된다고 보고합니다. 가장 무거운 모델은 이전의 최상의 결과를 6.1 RayIoU 점수로 초과 달성하여 새로운 기준을 세웠습니다.



### QTG-VQA: Question-Type-Guided Architectural for VideoQA Systems (https://arxiv.org/abs/2409.09348)
- **What's New**: 본 연구는 비디오 질문 응답(VQA) 시스템에서 질문 유형의 중요성을 다루며, 질문 유형의 다양성이 모델의 학습 능력에 미치는 영향을 탐구하는 최초의 시도를 합니다.

- **Technical Details**: QTG-VQA이라는 새로운 아키텍처를 제안하며, 질문 유형을 반영한 Attention 메커니즘과 Masking Frame Modeling 기법을 사용해 모델의 시간 의존성(temporal dependency)을 향상시킵니다. 이를 통해 질문 유형별로 학습 과정의 가중치와 학습 속도를 조정하는 Attention-Weighted Multi-Task Adaptive Learning 방법(AWMTL)을 도입합니다.

- **Performance Highlights**: 실험 결과, QTG-VQA 모델이 SUTD-TrafficQA 데이터셋에서 효과적인 성능 향상을 보였으며, 제안된 IFWAA와 EWAA라는 새로운 평가 지표를 통해 모델의 성능이 제대로 평가됨을 확인했습니다.



### LawDNet: Enhanced Audio-Driven Lip Synthesis via Local Affine Warping Deformation (https://arxiv.org/abs/2409.09326)
- **What's New**: 이 논문에서는 Audio 기반의 Lip motion(입술 움직임) 합성을 위한 새로운 딥러닝 구조인 LawDNet을 소개합니다. LawDNet은 Local Affine Warping Deformation(국소 아핀 왜곡) 메커니즘을 통해 리얼리즘을 높은 수준으로 향상시키며, 이는 오디오 입력에 대해 입술 움직임을 더욱 정교하게 모델링할 수 있게 합니다.

- **Technical Details**: 법칙적인 능률을 제공하는 LawDNet은 각 Feature map에 대해 Self-learned keypoints(자가 학습된 키포인트)와 Adaptive radii(적응성 반경)를 사용하여 지역 아핀 변환을 정의합니다. 이 메커니즘은 matrix-exponential operations을 사용하여 강화된다. 이와 함께 Spatial Discriminator(공간적 판별기)와 Temporal Discriminator(시간적 판별기)를 포함하여 연속성을 강화합니다.

- **Performance Highlights**: LawDNet은 기존 방식에 비해 더욱 향상된 Robustness(강건성) 및 Dynamic performance(동적 성능)를 보여줍니다. Extensive evaluations(광범위한 평가)를 통해 이전 방법들에 비해 우수한 결과를 도출함을 입증하였으며, 이는 연구 공동체에 공개될 예정입니다.



### Implicit Neural Representations with Fourier Kolmogorov-Arnold Networks (https://arxiv.org/abs/2409.09323)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이 논문에서는 임의의 주파수 구성 요소를 효과적으로 학습하기 위해 Fourier Kolmogorov Arnold Network (FKAN)를 제안합니다. FKAN은 초기 계층에서 Fourier 급수를 모델링한 학습 가능한 활성화 함수를 사용하여 고유 주파수 구성 요소를 캡처하는 데 중점을 둡니다.

- **Technical Details**: FKAN은 적응형 Fourier 계수를 사용하여 스펙트럼 편향을 조정합니다. 이는 학습 가능한 활성화 함수가 고정 밀도(fixed density)를 설정하여 입력 신호의 저주파 및 고주파 성분을 효율적으로 표현할 수 있게 해줍니다. 이 네트워크는 다층 퍼셉트론(MLP) 구조에 기반하여 다양한 주파수 정보를 유연하게 처리합니다.

- **Performance Highlights**: 실험 결과 FKAN 모델은 기존의 SIREN, WIRE, INCODE 및 FFN과 같은 세 가지 최신 기법보다 향상된 성능을 보여주었습니다. 이미지 표현 작업에서 PSNR(피크 신호 대 잡음 비율)을 최대 8.91% 향상시켰으며, SSIM(구조적 유사도 지수)를 최대 5.62% 개선했습니다. 3D 점유 볼륨 표현 작업에서는 IoU(교차 비율)를 최대 0.96 개선하였습니다. 또한, FKAN은 두 작업 모두에서 기본 모델보다 더 빠른 수렴 속도를 보였습니다.



### ChildPlay-Hand: A Dataset of Hand Manipulations in the Wild (https://arxiv.org/abs/2409.09319)
- **What's New**: 본 논문에서는 AR/VR 애플리케이션에서 발생한 여러 개의 egocentric dataset을 기반으로 손-객체 상호작용(Hand-Object Interaction, HOI)에 대한 관심이 증가하고 있음을 강조합니다. 특히, 제3자 관점에서의 HOI에 대한 연구가 부족하고, ChildPlay-Hand라는 새로운 데이터셋을 제안합니다.

- **Technical Details**: ChildPlay-Hand 데이터셋은 개인과 객체의 bounding box 및 조작 행동을 포함하고 있으며, 다음의 특징을 갖습니다: (1) 손별 주석 제공; (2) 자연스러운 상호작용을 보이는 비통제 설정에서의 비디오 포함; (3) ChildPlay-Gaze 데이터셋의 시선(label) 정보를 포함하여 조작 및 시선의 공동 모델링. 조작 행동은 잡기(grasping), 잡고 있기(holding), 운영하기(operating) 및 다양한 방출(releasing) 단계를 포괄합니다.

- **Performance Highlights**: 본 논문에서는 객체 손에 대한 감지(object in hand detection, OiH) 및 조작 단계(manipulation stages, ManiS)에 대한 두 가지 작업을 연구하였으며, 이를 통해 ChildPlay-Hand가 야외에서의 HOI 모델링을 위한 도전적인 새로운 벤치마크임을 발견했습니다.



### Tensor-Based Synchronization and the Low-Rankness of the Block Trifocal Tensor (https://arxiv.org/abs/2409.09313)
Comments:
          31 pages, 4 figures

- **What's New**: 이 논문은 trifocal tensor의 block tensor를 활용하여 3D 장면의 삼중보기 기하학을 연구합니다. 특히, 카메라의 위치와 방향을 회복하는 동기화(synchronization) 문제를 다루며, 블록 trifocal tensor의 Tucker factorization을 제시하여 (6,4,4)의 낮은 다중선형 순위(multi-linear rank)를 도출합니다.

- **Technical Details**: 주요 기술적 기여로, block trifocal tensor의 Tucker factorization이 있습니다. 이 tensor는 적절한 크기 조정 조건 하에서 (6,4,4)의 낮은 순위를 가지며, 이는 노이즈가 없는 경우에 카메라 포즈 복원이 가능함을 증명합니다. 이를 바탕으로 higher-order singular value decomposition을 이용한 동기화 알고리즘이 개발되었습니다.

- **Performance Highlights**: 실험 결과, 기존의 글로벌 동기화(global synchronization) 방법에 비해 위치 추정 정확도가 상당히 개선되었습니다. 이 연구는 동기화 문제에서의 higher-order 상호작용을 활용하여, 전통적인 쌍(pairwise) 기반 접근 방식보다 성능을 높일 수 있음을 시사합니다.



### Registration between Point Cloud Streams and Sequential Bounding Boxes via Gradient Descen (https://arxiv.org/abs/2409.09312)
- **What's New**: 이 논문에서는 포인트 클라우드(stream)와 순차 바운딩 박스(bounding box) 간의 정합(registration)을 위한 새로운 알고리즘을 제안합니다. 이전의 기법들과 달리, 이 알고리즘은 바운딩 박스의 속성(크기, 형상, 시간 정보 등)에 기반하여 정합을 수행합니다.

- **Technical Details**: 우리는 전체 목표 함수(objective function)를 모델링(modeling)하여 정합 과정(registration process)을 정의하고, 경량화된 제약조건(constraints)을 포함하여 경량화된 모델을 최적화하는 방법을 제안합니다. 이 과정에서 경량화된 제약조건은 주어진 바운딩 박스의 속성과 일치하도록 설계됩니다. 우리는 경량화된 목표 함수를 뉴턴 방법(Newton’s method)을 사용하여 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 IoU(Intersection over Union)에서 40%의 성능 향상을 보여줍니다. 이는 포인트 클라우드 스트림과 순차 바운딩 박스 간의 더 강력한 정합을 입증합니다.



### Keypoints-Integrated Instruction-Following Data Generation for Enhanced Human Pose Understanding in Multimodal Models (https://arxiv.org/abs/2409.09306)
- **What's New**: 이 논문은 복잡한 인간의 자세 및 행동을 이해하는 데 필요한 전문적인 instruction-following 데이터의 부족 문제를 해결하기 위한 새로운 데이터 생성 방법을 제안합니다. 이를 통해 humain keypoints와 전통적인 시각적 특성(예: captions, bounding boxes)을 결합하여, 인간 중심의 활동에 대한 모델의 성능을 향상시킬 수 있습니다.

- **Technical Details**: 제안된 방법은 LLaVA-7B 모델에 대한 미세 조정(fine-tuning)을 통해 세 가지 구체적인 유형의 instructional data(대화, 세부 설명, 복잡한 추론) 생성에 중점을 둡니다. 기존 LLaVA 모델은 캡션과 경계 상자만을 사용한 반면, 본 연구에서는 인간 keypoints를 추가하여 데이터 생성의 정확성과 풍부함을 높였습니다.

- **Performance Highlights**: LLaVA-7B 모델을 제안된 데이터 세트로 미세 조정한 결과, 인간 자세 관련 작업에서 원래 모델보다 21.18% 향상된 성능을 보였습니다. 이는 keypoints가 포함된 데이터가 멀티모달 모델의 성능 향상에 효과적임을 보여줍니다.



### ManiDext: Hand-Object Manipulation Synthesis via Continuous Correspondence Embeddings and Residual-Guided Diffusion (https://arxiv.org/abs/2409.09300)
- **What's New**: 이 논문에서는 손 조작 및 그리프 포즈를 3D 객체 궤적에 기반하여 생성하기 위한 통합된 계층적 diffusion 기반 프레임워크인 ManiDext를 소개합니다. 이는 객체와 손의 상호작용 동안의 접촉 대응 모델링이 중요하다는 인사이트에 기반합니다.

- **Technical Details**: ManiDext 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계에서는 객체 표면에서 접촉 맵과 연속 대응 임베딩을 생성합니다. 두 번째 단계에서는 손 포즈 생성을 위한 반복적인 정제 과정을 diffusion 프로세스에 통합하여 손 포즈 잔차를 조건으로 하여 네트워크를 유도합니다. 이 접근법은 기존의 최적화 프로세스와 자연스럽게 정렬되어 있습니다.

- **Performance Highlights**: 실험 결과, ManiDext는 다양한 작업에서 신체적으로 그럴듯하고 사실적인 동작을 생성할 수 있음을 보여줍니다. 단일 및 양손 그리프 및 강체 및 관절체 객체 조작을 포함한 여러 시나리오에서 높은 품질의 동작 생성이 가능합니다.



### Associate Everything Detected: Facilitating Tracking-by-Detection to the Unknown (https://arxiv.org/abs/2409.09293)
- **What's New**: 본 논문은 CV-MOT과 OV-MOT을 통합할 수 있는 새로운 프레임워크, Associate Everything Detected (AED)를 제안합니다. 이 프레임워크는 사전 지식에 의존하지 않고도 고도로 강인한 feature learning을 통해 복잡한 경로를 처리할 수 있습니다.

- **Technical Details**: AED는 먼저 객체 탐지를 수행하고, 그 후에 이를 연관시키는 'tracking-by-detection' 접근 방식에 기반하여, 객체와 추적 경로의 유사성을 디코딩하는 방법으로 연결됩니다. 여기서는 시뮬러리티 디코더(sim-decoder)와 연관 중심 학습 메커니즘을 적용하여 공간적, 시간적, 크로스-클립 유사성을 활용합니다.

- **Performance Highlights**: AED는 TAO, SportsMOT, DanceTrack 데이터셋에서 기존의 강력한 OV-MOT 및 CV-MOT 방법들보다 우수한 성능을 달성하며, 어떤 사전 지식이 필요하지 않습니다.



### StyleTalk++: A Unified Framework for Controlling the Speaking Styles of Talking Heads (https://arxiv.org/abs/2409.09292)
Comments:
          TPAMI 2024

- **What's New**: 이번 논문에서는 개인화된 발화 스타일을 반영하는 스타일 제어가 가능한 Talking head 생성 방법인 StyleTalk++를 제안합니다. 이 방법은 참조 비디오에서 발화 스타일을 학습하여 단일 초상 이미지를 기반으로 다양한 스타일의 비디오를 생성할 수 있습니다.

- **Technical Details**: 제안하는 메서드는 3D Morphable Model(3DMM)의 계수를 동기화하고, 스타일 인코더를 통해 스타일 코드로 변환하여 자연스러운 Facial expressions 및 Head poses를 생성합니다. 그리고 스타일 인식을 통해 오디오와 결합하여보다 자연스러운 비디오를 생성합니다.

- **Performance Highlights**: 광범위한 실험을 통해, StyleTalk++는 하나의 초상 사진과 오디오 클립만으로도 시각적으로 사실적이고 다양한 발화 스타일의 talking head 비디오를 생성할 수 있음을 입증하였습니다.



### Infrared and Visible Image Fusion with Hierarchical Human Perception (https://arxiv.org/abs/2409.09291)
- **What's New**: 이 연구에서는 이미지 융합(image fusion) 기술을 개선하기 위한 새로운 방법인 Hierarchical Perception Fusion (HPFusion)을 제안합니다. 이 방법은 대규모 비전-언어 모델(Large Vision-Language Model, LVM)을 활용하여 인간의 계층적 의미 선행 정보(hierarchical human semantic priors)를 통합함으로써, 기존의 이미지 융합 방법들이 간과했던 인간의 인지적 시각을 개선합니다.

- **Technical Details**: HPFusion은 여러 질문 세트를 정의하여 이미지 쌍을 관찰할 때 인간이 중점을 두는 것에 초점을 맞춥니다. 이 질문들은 텍스트로 생성되며, CLIP(Contrastive Language-Image Pre-training) 모델에 의해 융합 네트워크에 인코딩 되어 과정에 반영됩니다. 여기서는 전문가 조언을 받아 질문이 이미지 쌍의 내용을 묘사하고 주요 객체를 식별하며, 고대비 및 정보가 풍부한 특정 지역에 주목하도록 설계되었습니다.

- **Performance Highlights**: HPFusion은 정보 보존과 인간 시각 개선 모두에서 높은 품질의 융합 결과를 도출함을 실험을 통해 입증하였습니다. 실험 결과는 HPFusion이 기존의 방법들에 비해 더욱 적합한 시각적 정보의 통합을 가능하게 함을 보여줍니다.



### SAM-OCTA2: Layer Sequence OCTA Segmentation with Fine-tuned Segment Anything Model 2 (https://arxiv.org/abs/2409.09286)
- **What's New**: 이번 연구에서는 Optical Coherence Tomography Angiography (OCTA)의 비침습적 이미징 기술을 사용하여, 저순위 적응(low-rank adaptation) 기법을 활용하여 Segment Anything Model (SAM) 버전 2를 미세조정(fine-tuning)한 SAM-OCTA2를 제안합니다. 이는 OCTA 스캐닝 레이어 시퀀스에서 특정 객체를 효과적으로 추적 및 분할할 수 있게 설계되었습니다.

- **Technical Details**: SAM-OCTA2 모델은 3D OCTA 데이터 샘플의 지역 혈관(RV)과 중심 구역(FAZ) 분할에 중점을 두고 있으며, 이미지 인코더, 프롬프트 인코더, 마스크 디코더를 포함한 SAM 구조를 기반으로 합니다. LoRA(저순위 적응)을 통해 이미지 인코더에 대한 포괄적인 미세조정을 수행하며, 프롬프트 포인트 생성 및 희소 주석 방법(sparse annotation)을 통해 적절한 분할 처리를 가능하게 합니다.

- **Performance Highlights**: SAM-OCTA2는 OCTA-500 데이터셋에서 시행된 실험에서, 2D en-face 이미지에서 FAZ 분할을 위한 최첨단 성능을 달성하였고, 스캐닝 레이어 시퀀스 전반에 걸쳐 지역 혈관을 효과적으로 추적하였습니다. 이 모델은 해당 분야의 기존 방법들보다 우수한 성능을 보여주었습니다.



### LabellessFace: Fair Metric Learning for Face Recognition without Attribute Labels (https://arxiv.org/abs/2409.09274)
- **What's New**: 고유한 인구 통계 그룹 레이블 없이 얼굴 인식의 인구 통계적 편향을 개선하는 새로운 프레임워크인 'LabellessFace'를 제안합니다.

- **Technical Details**: LabellessFace는 데이터셋 전반에서 특정 클래스에 대한 편애 정도를 측정하는 'class favoritism level'을 도입하며, 이를 바탕으로 'fair class margin penalty'를 적용하여 학습 매개변수를 동적으로 조정합니다. 이 방법은 주어진 데이터셋의 개개인에 대한 인증 정확도의 편향을 최소화하도록 학습을 촉진합니다.

- **Performance Highlights**: 종합 실험을 통해 LabellessFace가 기존 접근 방식과 비교하여 인증 정확도를 유지하면서 공정성을 효과적으로 향상시키는 것으로 나타났습니다.



### Guiding Vision-Language Model Selection for Visual Question-Answering Across Tasks, Domains, and Knowledge Types (https://arxiv.org/abs/2409.09269)
Comments:
          8 pages + references + 6 pages of Appendix

- **What's New**: 이 논문은 Visual Question-Answering (VQA) 작업을 위한 Vision-Language Models (VLMs)을 평가하는 포괄적인 프레임워크를 제시합니다. 또한, 다양한 작업 유형 및 응용 도메인과 함께 지식 유형으로 주석이 달린 새로운 데이터셋을 소개하며, 이를 통해 특정 작업 요구사항에 따라 VLMs를 선택할 수 있는 방법을 제공합니다.

- **Technical Details**: 논문에서는 VQA 작업의 평가를 위한 표준화된 프레임워크를 개발하며, task type, application domain, knowledge type의 세 가지 측면에서 VLMs를 비교합니다. 이를 위해 GoEval이라는 다중 양식(multi-modal) 평가 지표를 도입하고, 이는 GPT-4o를 기반으로 하여 사람의 평가와 높은 상관관계를 보입니다. 실험에서는 8개의 다양한 VLM 변종을 평가하여 각각의 강점과 약점을 분석합니다.

- **Performance Highlights**: 엑실리에서의 실험 결과는 Gemini-1.5-Pro 및 GPT-4o-mini와 같은 독점 모델이 일반적으로 다른 모델보다 우수한 성능을 보이지만, InternVL-2-8B 및 CogVLM-2-Llama-3-19B와 같은 오픈소스 모델도 특정 맥락에서 경쟁력을 갖추고 있다는 것을 보여줍니다. VLM들의 성능은 서로 높은 상관관계가 없으며, 각 모델은 특정 카테고리에서는 잘 수행하지만 다른 카테고리에서는 어려움을 겪는 경향이 있습니다.



### VSFormer: Mining Correlations in Flexible View Set for Multi-view 3D Shape Understanding (https://arxiv.org/abs/2409.09254)
Comments:
          accepted by TVCG 2024

- **What's New**: 본 논문은 다중 뷰(views)에서의 유연한 조직(organization) 및 명시적 상관관계 학습(correlation learning)을 소개하며, 이를 통해 3D 형태 이해(3D shape understanding)에서의 한계를 극복하고자 한다.

- **Technical Details**: 이 논문에서 제안한 	extit{View Set}은 서로 다른 3D 형태의 뷰를 순열 불변(permutation-invariant) 셋으로 통합하여 경직된 관계 설정을 제거하고 뷰 간의 정보 교환과 융합을 촉진한다. 	extit{VSFormer}라는 Transformer 모델은 이 셋의 모든 요소의 쌍(pairwise) 및 고차원(higher-order) 상관관계를 명시적으로 포착하도록 설계되었다.

- **Performance Highlights**: VSFormer는 RGBD, ScanObjectNN, ModelNet40에서 각각 98.4%(+4.1%), 95.9%(+1.9%), 98.8%(+1.1%)의 전체 정확도로 최신 성능 기준을 초과하였으며, SHREC'17 검색 벤치마크에서도 새로운 기록을 수립하였다.



### Investigation of Hierarchical Spectral Vision Transformer Architecture for Classification of Hyperspectral Imagery (https://arxiv.org/abs/2409.09244)
Comments:
          \c{opyright} 2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

- **What's New**: 이 논문에서는 HSI(고해상도 스펙트럼 이미지) 분류를 위한統 하나의 통합 계층 스펙트럼 비전 트랜스포머 아키텍처가 제안되었습니다. 이는 CNN(합성곱 신경망) 및 다양한 믹서 모듈(예: CNN-mixer, SSA-mixer, CSA-mixer)과 통합되어 HSI 분류의 성능을 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 CNN과 비전 트랜스포머 기반 모델 간 교육 과정 및 왜곡 강인성(disturbance robustness)과 해시안의 최대 고유값(eigenvalue of the Hessian) 분포를 비교하고, MSA(multi-head self-attention) 구성 요소에 의존하지 않고도 비전 트랜스포머의 우월한 성능이 전체 아키텍처에 기인한다는 점을 강조합니다.

- **Performance Highlights**: 여러 믹서 모델을 평가한 결과, 통합 아키텍처가 CNN에 비해 뛰어난 성능을 보였으며, HSI 분류 작업에서 새로운 비전 트랜스포머 모델들이 효과적으로 작동할 수 있음을 보여줍니다.



### Multi-modal Speech Transformer Decoders: When Do Multiple Modalities Improve Accuracy? (https://arxiv.org/abs/2409.09221)
- **What's New**: 이 논문은 디코더 전용 이산 토큰 언어 모델이 자동 음성 인식(ASR)에서 다양한 모달리티가 성능에 미치는 영향을 체계적으로 분석하는 첫 번째 연구 중 하나입니다. 특히, 음성, 이미지, 입술 정보를 결합하여 ASR 정확도를 향상시키는 방법을 제시합니다.

- **Technical Details**: 논문에서 제안하는 모델은 디스크리트 멀티모달 언어 모델(DMLM)이며, OPT를 백본으로 삼고 다양한 입력 모달리티(음성, 이미지, 입술 움직임, OCR 텍스트)를 처리할 수 있습니다. 실험을 위해서는 3-Equations라는 합성 다중 모달 데이터셋이 사용되었습니다. 이 데이터셋은 기계적으로 생성된 수학 방정식과 함께 음성 샘플 및 입술 동영상 샘플로 구성되어 있습니다.

- **Performance Highlights**: 모달리티를 추가함으로써 ASR 정확도가 평균적으로 향상되는 것으로 나타났으며, 특히 보통의 잡음 수준에서 이미지가 보조 모달리티로 가장 큰 이점을 제공합니다. 또한, 가장 관련성이 높은 시각 정보를 사전 처리 단계에서 필터링했을 때 성능이 개선되었습니다.



### Are Sparse Neural Networks Better Hard Sample Learners? (https://arxiv.org/abs/2409.09196)
Comments:
          Accepted at British Machine Vision Conference (BMVC 2024)

- **What's New**: 이 논문은 Sparse Neural Networks (SNNs)의 학습 효과를 복잡하고 어려운 샘플을 사용할 때 평가하며, 특히 이러한 조건에서 SNN들이 밀집 모델보다 성능이 뛰어날 수 있음을 제시합니다. 연구자는 레이어별 밀도 비율(layer-wise density ratio)이 SNN 성능에 미치는 영향을 강조합니다.

- **Technical Details**: SNNs는 밀집 모델에 비해 연산 비용을 줄이고 과적합(overfitting) 문제를 완화하는 데 유용합니다. 이 연구에서는 학습의 어려움을 평가하기 위해 EL2N(Error L2 Norm)을 활용하며, 도전 샘플에 대한 훈련 방법을 연관된 다양한 방법론을 통해 탐구합니다. 이는 흥미로운 샵들에서 레이어 밀도를 조정할 때 효율성을 최대화하는 전략을 담고 있습니다.

- **Performance Highlights**: SNN은 제한된 훈련 데이터에서도 밀집 모델보다 성능이 개선되며, 특정 수준의 희소성(sparsity)에서 정확성을 달성하거나 초과하는 경향을 보입니다. 또한 얕은 레이어에서 높은 밀도를 유지하는 것이 성능 향상에 기여할 수 있습니다.



### Transformer with Controlled Attention for Synchronous Motion Captioning (https://arxiv.org/abs/2409.09177)
- **What's New**: 이 논문에서는 사람의 움직임 시퀀스와 동기화된 언어 설명을 생성하는 어려운 작업인 동기화 모션 캡셔닝(synchronous motion captioning)에 대해 다룹니다. 이 방식은 수신어 수화 전사(aligned sign language transcription), 무감독 동작 세분화(unsupervised action segmentation), 시간 기반 텍스트 정렬(temporal grounding) 등 여러 응용 분야와 관련이 있습니다. 논문은 Transformer의 자기(Self-) 및 교차(Cross-) 주의(attention) 분포를 제어하기 위한 메커니즘을 도입하여, 해석 가능성(interpretability)과 시간에 맞춘 텍스트 생성을 수행합니다.

- **Technical Details**: 제안된 방법에서는 마스킹 전략과 구조화된 손실(structuring losses)을 활용하여 모델이 모션 단어 생성에 기여하는 가장 중요한 프레임에만 최대 주의(attention)를 두도록 유도합니다. 이러한 제약은 attention 맵에서 정보의 원치 않는 혼합(mixing)을 방지하고, 토큰 간의 단조(monotonic) attention 분포를 제공합니다. 따라서, 토큰의 교차 주의는 사람의 움직임 시퀀스와 동기화된 점진적(text generation) 텍스트 생성을 위해 사용됩니다.

- **Performance Highlights**: KIT-ML 및 HumanML3D의 두 개의 벤치마크 데이터셋에서 평가하여 제안된 방법의 우수한 성능을 입증합니다. 이 작업에 시각적 평가가 필수적이므로 코드 저장소에서 애니메이션 시각화의 포괄적인 세트를 제공합니다.



### Adaptive Multi-Modal Control of Digital Human Hand Synthesis Using a Region-Aware Cycle Loss (https://arxiv.org/abs/2409.09149)
Comments:
          This paper has been accepted by the ECCV 2024 HANDS workshop

- **What's New**: 이번 논문에서는 손 포즈 생성의 세부적인 제어를 위한 Diffusion 모델의 한계를 극복하기 위해, How2Sign 데이터셋을 개선하고 새로운 Region-Aware Cycle Loss (RACL)을 도입하여 손 영역의 품질을 향상시키는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 다중 모드 제어 기능 융합 네트워크와 RACL을 활용하여 손 포즈 생성을 향상시킵니다. RACL은 생성된 이미지와 실제 이미지 간의 주요 포인트 거리의 가중치를 계산하여 손 영역의 품질을 개선하며, 손-PSNR 및 손-거리(hand-Distance)를 통해 손 포즈 생성 품질을 평가합니다.

- **Performance Highlights**: 실험 결과, 다양한 모드를 사용한 디지털 인간 생성에서 제안된 접근 방식이 손 영역 생성의 품질을 현저하게 개선한 것으로 나타났습니다. 특히 surface normal을 사용한 경우, 단일 모드 제어의 경우에 비해 우수한 결과를 도출했습니다.



### PrimeDepth: Efficient Monocular Depth Estimation with a Stable Diffusion Preimag (https://arxiv.org/abs/2409.09144)
- **What's New**: 이 논문에서는 제로샷(Zero-shot) 모노큘러(그림 한 장) 깊이 추정을 다룹니다. 최근에는 Stable Diffusion과 같은 텍스트-이미지(Text-to-Image) 기반의 파운데이션 모델을 활용하여 이 과제를 해결하는 방법이 주목받고 있습니다. 제안하는 PrimeDepth는 테스트 시 매우 효율적이며, 기존의 확산 모델의 장점을 유지하거나 강화하는 데에 중점을 두고 있습니다.

- **Technical Details**: PrimeDepth는 Stable Diffusion에서 단일 디노이징(denoising) 단계만을 통과시켜 추출한 정지된 이미지 표현(preimage)을 사용합니다. 이 표현은 리파이너(refiner) 네트워크에 입력되며, 이후 하위 작업(downstream task)으로 전달됩니다. PrimeDepth는 기존의 확산(model) 기반 방법인 Marigold보다 평균적으로 100배 빠르며, 특히 조명 조건이 극단적인 상황에서도 더욱 강력한 성능을 발휘합니다.

- **Performance Highlights**: PrimeDepth는 관련 평가에서 Depth Anything보다 수량적으로는 뒤쳐지지만, 20배 더 많은 레이블 데이터(labelled data)를 요구하는 Depth Anything보다 더 상세한 깊이 맵을 예측합니다. 평균적으로 100배 더 빠르며, 어려운 장면에서도 이전의 방법들보다 더 견고한 성능을 보여줍니다.



### CtRNet-X: Camera-to-Robot Pose Estimation in Real-world Conditions Using a Single Camera (https://arxiv.org/abs/2409.10441)
Comments:
          7 pages, 5 figures, project website: this https URL

- **What's New**: 본 논문에서는 카메라에서 로봇을 향한 포즈 추정을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 부분적으로 보이는 로봇 조작기를 다룰 수 있는 능력을 가지고 있으며, Vision-Language Model을 활용하여 로봇 구성 요소를 탐지하고 키포인트를 선택하여 포즈 추정을 수행합니다.

- **Technical Details**: 제안하는 접근 방식은 기존의 CtRNet 구조를 기반으로 하며, CLIP과 같은 비전-언어 모델을 사용하여 이미지 프레임에서 보이는 로봇 부품을 식별합니다. 또, 키포인트 탐지 퍼포먼스를 높이기 위해 DARK(Distribution-Aware coordinate Representation of Keypoint) 방법을 통합합니다.

- **Performance Highlights**: 본 연구의 프레임워크는 공개된 로봇 데이터셋과 자가 수집한 부분 보기 데이터셋에서 평가되었으며, 기존 방법들에 비해 우수한 성능을 입증했습니다. 또한, 커다란 규모의 로봇 조작 데이터셋에 대한 정확한 카메라 외부 정보를 추정할 수 있는 능력을 보여주었습니다.



### Point2Graph: An End-to-end Point Cloud-based 3D Open-Vocabulary Scene Graph for Robot Navigation (https://arxiv.org/abs/2409.10350)
Comments:
          8 pages, 9 figures

- **What's New**: 이번 논문에서는 기존의 RGB-D 이미지와 카메라 포즈에 의존하던 open-vocabulary scene graph generation 알고리즘의 한계를 극복하기 위해 Point2Graph라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: Point2Graph는 point cloud 데이터를 기반으로 한 end-to-end 방법론으로, 방(room)과 객체(object) 탐지/세분화 및 open-vocabulary 분류 기능을 포함하고 있습니다. 특히 방 탐지에서는 기하학적 경계 검출(geometry-based border detection) 알고리즘과 학습 기반 지역 탐지(learning-based region detection)를 결합하여 방을 세분화하고, 'Snap-Lookup' 프레임워크를 통한 open-vocabulary 방 분류를 가능하게 합니다. 또한, 객체 탐지 레이어에서는 오직 3D point cloud 데이터만을 사용하여 3D 객체를 탐지하고 분류하는 end-to-end 파이프라인을 구현합니다.

- **Performance Highlights**: 실험 결과, Point2Graph는 널리 사용되는 실제 장면 데이터셋에서 현재 최첨단(SOTA) open-vocabulary 객체 및 방 세분화와 분류 알고리즘을 초월하는 성능을 보임을 확인했습니다.



### VAE-QWGAN: Improving Quantum GANs for High Resolution Image Generation (https://arxiv.org/abs/2409.10339)
Comments:
          5 pages, 8 figures

- **What's New**: 본 논문에서는 VAE(변분 오토 인코더)와 QWGAN(하이브리드 양자 워셔스타인 생성적 적대 신경망)을 결합한 새로운 하이브리드 양자 생성 모델 VAE-QWGAN을 제안합니다. 이 모델은 VAE의 디코더와 QGAN 생성기를 단일 양자 모델로 통합하여 훈련 중에는 잠재 벡터 샘플링 과정에서 VAE의 인코더를 활용합니다.

- **Technical Details**: VAE-QWGAN은 VAE 인코더를 사용해 잠재 랜덤 벡터를 샘플링하여 QGAN 생성기를 훈련합니다. 또한, 훈련된 모델의 성능을 극대화하기 위해, 가우시안 혼합 모델(GMM)을 사용하여 입력 잠재 벡터를 샘플링합니다. 이러한 접근 방식은 생성된 이미지의 다양성과 품질을 향상시키는 데 기여합니다.

- **Performance Highlights**: MNIST 및 Fashion-MNIST 데이터셋에서 VAE-QWGAN의 성능을 평가한 결과, 기존 방법들에 비해 생성된 이미지의 품질과 다양성이 크게 개선된 것으로 나타났습니다.



### Phys3DGS: Physically-based 3D Gaussian Splatting for Inverse Rendering (https://arxiv.org/abs/2409.10335)
Comments:
          Under review

- **What's New**: 본 논문에서는 3D Gaussian splatting (3DGS) 기반의 역 렌더링(다시 렌더링) 품질을 향상시키기 위해 두 가지 새로운 아이디어를 제안합니다: deferred rendering(지연 렌더링)의 채택과 mesh-based representation(메시 기반 표현)입니다. 특히, 기존의 방법에서 발생하는 hidden Gaussians(숨겨진 가우시안) 문제를 보고하고, 이 문제를 해결하기 위한 novel two-step training approach(새로운 2단계 훈련 접근법)를 제안합니다.

- **Technical Details**: 제안된 방법은 (1) hybrid mesh-3DGS representation(하이브리드 메시-3DGS 표현)을 활용하여 메시 추출을 통해 지오메트리를 이용하고, (2) 메시를 더욱 효과적으로 활용하기 위한 novel regularization methods(새로운 정규화 방법)를 적용합니다. 이를 통해 드로우에서의 품질을 개선하고, 자연 방출 조건에서 더 나은 렌더링 결과를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 Phys3DGS 방법이 기존의 relightable 3DGS 방법에 비해 더 뛰어난 렌더링 품질을 제공하며, 전통적인 voxel-grid 기반의 역 렌더링 방법에 비해 더 빠르고 높은 품질의 렌더링을 제공합니다.



### DRIVE: Dependable Robust Interpretable Visionary Ensemble Framework in Autonomous Driving (https://arxiv.org/abs/2409.10330)
- **What's New**: 최근 자율 주행 분야에서 end-to-end learning 패러다임이 발전하고 있으며, 이는 센서 입력을 주행 동작으로 직접 매핑하여 자율 차량의 견고성과 적응성을 향상시킵니다. 그러나 이러한 모델은 해석 가능성을 희생하는 경우가 많아 신뢰성, 안전성, 규제 준수에 significant challenges를 발생시킵니다. 이에 대한 해결책으로 우리는 'DRIVE'라는 포괄적인 프레임워크를 도입하여 end-to-end 비지도 자율 주행 모델의 설명의 의존성과 안정성을 개선합니다.

- **Technical Details**: DRIVE는 Consistent Interpretability (일관된 해석 가능성), Stable Interpretability (안정적인 해석 가능성), Consistent Output (일관된 출력), Stable Output (안정적인 출력)의 네 가지 핵심 속성을 정의합니다. 이 속성들은 서로 다른 시나리오 및 변동성에서 설명이 신뢰할 수 있고 견고하게 유지되도록 보장합니다. DRIVE의 분석을 통해 DCG 모델의 의존성 문제를 해결하고, 이로 인해 현재 모델의 한계를 극복하는 데 기여할 수 있습니다.

- **Performance Highlights**: 우리의 실증 평가를 통해 DRIVE 프레임워크가 설명의 안정성과 의존성을 향상시키는 데 효과적임을 입증했습니다. 이는 자율 주행 시스템의 신뢰도를 향상시키고, 실제 응용에서의 고도화를 위한 기초를 마련합니다. 이러한 발전은 자율 주행 시스템의 더 넓은 수용 및 배포를 촉진할 것이라고 기대합니다.



### SPAC: Sampling-based Progressive Attribute Compression for Dense Point Clouds (https://arxiv.org/abs/2409.10293)
Comments:
          136pages, 13 figures

- **What's New**: 본 연구는 **Dense Point Clouds**에 대한 완전한 끝-투-끝 속성 압축 방법을 제안하였다. 특히, **주파수 샘플링 모듈**, **적응형 스케일 특징 추출 모듈**, **지오메트리 지원 속성 특징 정제 모듈**, 그리고 **글로벌 하이퍼프라이어 엔트로피 모델**을 통합하여 새로운 방식을 개발하였다.

- **Technical Details**: 제안한 방법은 **Hamming window**와 **Fast Fourier Transform (FFT)**를 활용하여 포인트 클라우드의 고주파 성분을 추출하며, 샘플링된 포인트 클라우드와 원본 클라우드 간의 차이를 여러 개의 하위 포인트 클라우드로 나눈다. 그런 다음, **옥트리 (octree)**를 통해 이 하위 클라우드를 파티션하여 특징 추출을 위한 구조화된 입력을 제공한다. 적응형 합성곱 레이어를 포함한 특징 추출 모듈은 지역 및 글로벌 특징을 포착하며, 지오메트리 지원 특성 정제 모듈을 통해 추출된 특징을 정제하고, 글로벌 하이퍼프라이어 모델을 통해 효율적인 엔트로피 인코딩을 수행한다.

- **Performance Highlights**: MPEG 공통 테스트 조건 (CTCs)하에서 제안된 방법은 최신 G-PCC 테스트 모델(TMC13v23)을 초월하는 성과를 보여주었다. Y 성분에 대해 평균 Bjønte가르 delta bitrate 감소는 24.58%로, 전체 YUV 결합 시 21.23%로 나타났고, 느슨한 데이터셋에서는 22.48%와 17.19%를 기록하였다.



### Self-Updating Vehicle Monitoring Framework Employing Distributed Acoustic Sensing towards Real-World Settings (https://arxiv.org/abs/2409.10259)
- **What's New**: 최근 도입된 Distributed Acoustic Sensing (DAS) 기술을 활용하여 교통으로 인한 지진 데이터를 효과적으로 수집하는 실시간 반자동화 차량 모니터링 프레임워크를 소개합니다. 이 프레임워크는 초기 학습에 소량의 수동 레이블만 필요하고 비레이블 데이터를 활용하여 모델을 개선합니다.

- **Technical Details**: 본 연구는 DAS 데이터를 1차원 신호 전처리 후 객체 탐지에 사용하며, 새로운 차량 추적 방법인 shape prior loss를 도입하여 다양한 속도의 차량을 추적합니다. 실험은 Stanford 2 DAS Array에서 이루어졌으며, 1D 및 2D 데이터 처리 방법을 결합한 전처리 워크플로우를 구축하였습니다.

- **Performance Highlights**: 35개의 레이블된 이미지만으로 YOLO의 mAP 0.5:0.95 기준을 18% 초과 달성하고, Efficient Teacher보다 7% 향상된 성능을 보였습니다. 자가 업데이트를 위한 최적 방법을 식별하였으며, 이는 단일 패스에서 전체 데이터로 수행된 비오버피팅 훈련을 초월합니다.



### FGR-Net:Interpretable fundus imagegradeability classification based on deepreconstruction learning (https://arxiv.org/abs/2409.10246)
- **What's New**: 이 논문에서는 FGR-Net이라고 불리는 새로운 프레임워크를 제안하여 망막 이미지 품질을 자동으로 평가하고 해석하는 방법을 소개합니다. 이 시스템은 오토인코더(autoencoder) 네트워크와 분류기(classifier) 네트워크를 결합하여 작동합니다.

- **Technical Details**: FGR-Net 모델은 입력 이미지를 재구성(reconstruct)하기 위해 딥 오토인코더(deep autoencoder)를 사용하며, 자가 지도 학습(self-supervised learning)을 기반으로 입력 망막 이미지의 시각적 특성을 추출합니다. 추출된 특성은 딥 분류기 네트워크로 전달되어 gradable과 ungradable 망막 이미지를 구분합니다. 모델은 다양한 해석 가능성(interpretability) 방법에 대해 평가되었습니다.

- **Performance Highlights**: FGR-Net의 실험 결과는 기존 최첨단 품질 평가 방법들을 초과하는 성능을 보였으며, 정확도(accuracy) 89%와 F1-score 87%를 기록했습니다.



### SteeredMarigold: Steering Diffusion Towards Depth Completion of Largely Incomplete Depth Maps (https://arxiv.org/abs/2409.10202)
- **What's New**: 본 연구에서는 SteeredMarigold라는 새로운 'zero-shot' 깊이 완성 방법을 제안합니다. 이 방법은 훈련이 필요 없으며, 대규모로 불완전한 깊이 맵을 처리할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: SteeredMarigold는 주어진 희소 깊이 점들을 이용하여 변별력 있는 'denoising diffusion probabilistic model(DDPM)'을 조작합니다. 기존의 Marigold 모델을 활용하여 상황에 맞는 조건부 깊이 생성을 수행합니다. 이 방법은 전체 깊이 측정이 결여된 상황에서도 성능을 발휘할 수 있습니다.

- **Performance Highlights**: SteeredMarigold는 NYUv2 데이터셋의 테스트에서 기존 최고 성능 모델들을 초월하는 결과를 보여주었습니다. 특히, 깊이 맵이 자료의 대규모 결여를 가진 경우에도 놀라운 강인성을 입증하며, 최신 기술에 의한 깊이 맵 완성에서 최첨단 성능을 달성했습니다.



### NEUSIS: A Compositional Neuro-Symbolic Framework for Autonomous Perception, Reasoning, and Planning in Complex UAV Search Missions (https://arxiv.org/abs/2409.10196)
- **What's New**: 이 논문은 자율 UAV(무인 항공기)의 검색 임무 문제를 다루고 있으며, UAV가 제한된 시간 내에 특정 관심 개체(Entities of Interest, EOIs)를 위치 찾는 것을 목표로 합니다. 이를 위해 NEUSIS라는 새로운 조합적 신경-상징 시스템을 제안합니다.

- **Technical Details**: NEUSIS는 세 가지 주요 구성 요소로 이루어져 있습니다: 1) GRiD(Perception, Grounding, Reasoning in 3D)는 UAV 시각 센서를 사용하여 3D 공간에서 관심 개체를 인식하고 이유를 찾습니다. 2) Probabilistic World Model은 GRiD의 출력값을 개선하고, 세계 지식 기반으로 신뢰도를 업데이트합니다. 3) SNaC는 고수준 계획, 중수준 계획 및 저수준 계획을 통해 효율적인 경로 계획을 지원합니다.

- **Performance Highlights**: NEUSIS는 AirSim 및 Unreal Engine을 활용한 도시 검색 임무 실험에서 최신 비전-언어 모델 및 검색 계획 모델보다 성공률, 검색 효율성, 3D 위치 지정에서 우수한 성능을 보여, 자율 UAV 시스템의 검색 임무에 대한 해결책으로서의 가능성을 입증하였습니다.



### SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting (https://arxiv.org/abs/2409.10161)
- **What's New**: 이 논문은 Sim2Real 문제를 해결하기 위해 Gaussian Splatting 기법을 활용한 새로운 프레임워크인 SplatSim을 제안합니다. 이 프레임워크는 전통적인 메쉬 표현을 대체하여 RGB 기반 조작 정책의 Sim2Real 격차를 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: SplatSim은 기존 시뮬레이터의 물리적 백엔드를 활용하여 Gaussian Splatting을 주된 렌더링 요소로 사용합니다. 이는 단일 비디오만으로 로봇과 객체 간의 물리적 상호작용을 효과적으로 시뮬레이션할 수 있게 해줍니다. 이 프레임워크는 상태 최적화된 행동 복제(bahavior cloning) 기법을 통합하여, 시뮬레이션 데이터로 훈련된 조작 정책을 실세계에서 제로샷(zero-shot) 방식으로 전이할 수 있습니다.

- **Performance Highlights**: SplatSim을 사용한 RGB 정책의 제로샷 전이를 통해 86.25%의 평균 성공률을 달성하였으며, 이는 실세계 데이터로 훈련된 정책의 97.5%에 비해 낮은 수치입니다. 하지만, 시뮬레이션 기반 접근 방식으로도 비교적 우수한 성능을 보여주고 있습니다.



### P2U-SLAM: A Monocular Wide-FoV SLAM System Based on Point Uncertainty and Pose Uncertainty (https://arxiv.org/abs/2409.10143)
Comments:
          The source code will be made publicly available at this https URL

- **What's New**: 본 논문에서는 폭넓은 시야( wide Field of View, FoV) 카메라를 이용한 P2U-SLAM이라는 시각적 동시 위치 측정 및 맵핑(SLAM) 시스템을 제안합니다. 이 시스템은 포즈 불확실성(pose uncertainty) 및 점 불확실성(point uncertainty)을 활용하여 SLAM 성능을 향상시키고자 합니다.

- **Technical Details**: P2U-SLAM은 조건부 확률 모델을 기반으로 하여 데이터 속성 변화가 최적화 과정에 미치는 영향을 규명하고, 이를 점 불확실성 및 포즈 불확실성으로 구체화합니다. 이 시스템은 추적 모듈 및 지역 맵핑에 이 두 가지 불확실성을 각각 삽입하고, 지역 맵핑, 맵 병합, 루프 닫기 등 각 최적화 작업 후에 이 불확실성을 업데이트합니다.

- **Performance Highlights**: P2U-SLAM은 PALVIO 및 TUM-VI의 두 개의 공공 데이터셋에서 27개의 시퀀스를 통해 exhaustively 평가되었습니다. PALVIO 데이터셋에서는 평균 경로 오류(Absolute Trajectory Error, ATE)가 59.7% 감소하였으며, TUM-VI 데이터셋에서는 장기 시퀀스와 단기 실내 시퀀스에서 비교 가능한 누적 드리프트를 유지하여, 기존 최고 성능 방법들을 초월하는 결과를 보여주었습니다.



### Data-Centric Strategies for Overcoming PET/CT Heterogeneity: Insights from the AutoPET III Lesion Segmentation Challeng (https://arxiv.org/abs/2409.10120)
Comments:
          Contribution to the data-centric task of the autoPET III Challenge 2024

- **What's New**: 이번 AutoPET 챌린지에서는 데이터 중심(data-centric) 작업으로 초점을 이동하여 PET/CT 이미징에서 전이성 병변(segmentation) 분할 개선을 위한 데이터 품질 및 처리 전략을 강조했습니다. 이전 버전에서 모델 개발에 집중했던 대신, 데이터 품질 향상 및 처리 방법에 대한 새로운 접근 방식을 선보였습니다.

- **Technical Details**: 본 연구에서는 PET과 CT 이미지 간의 정렬 오류(misalignment) 및 미세한 전이성 병변을 다루기 위해 수정된 데이터 증강(data augmentation) 기법을 적용했습니다. misalignment augmentation을 포함한 두 가지 주요 요소를 통해 세그멘테이션의 정확도를 높이는 데 기여합니다. 또한, 이미지 크기 변동에 따른 예측 시간을 최적화하기 위해 동적 앙상블(dynamic ensembling)과 테스트 시간 증강(test-time augmentation, TTA) 전략을 도입했습니다. 이 접근 방식은 모든 이미지에서 일반화(generalization) 가능성을 극대화합니다.

- **Performance Highlights**: 우리의 솔루션은 다양한 트레이서와 기관 설정에 견고하게 작동하도록 설계되었습니다. 5분의 예측 시간 제한 속에서 최대 3개의 모델을 앙상블하며 TTA의 수를 동적으로 조정하여 최대 50%의 Dice 점수(Dice score)를 달성했습니다. 본 연구는 다양한 이미징 조건에도 일관된 성능을 유지하며, 데이터 중심 AI의 중요성을 강조하고 있습니다.



### Cross-modality image synthesis from TOF-MRA to CTA using diffusion-based models (https://arxiv.org/abs/2409.10089)
- **What's New**: 본 연구는 TOF-MRA 이미지를 입력으로 하여 합성된 CTA 이미지를 생성하기 위한 확산 기반 (diffusion-based) 이미지 간 변환 모델을 탐구합니다. 특히, 이 연구는 TOF-MRA에서 CTA로의 변환이 가능함을 보여주고, 기존의 U-Net 기반 접근법보다 확산 모델이 더 우수한 성능을 보임을 밝혀냈습니다.

- **Technical Details**: 연구에서는 쌍 이미지 간 변환 과제를 위해 여러 최신 확산 아키텍처(diffusion architectures) 및 샘플러(samplers)를 비교하고, 최적의 모델 성능을 위한 권장 사항을 제공합니다. 이 과정에서, 데이터 부족(data scarcity)가 CTA 모델의 개발을 얼마나 저해하는지를 설명하며, 딥 러닝(deep learning)을 활용하여 2D 슬라이스에서의 쌍 변환이 가능하다는 것을 입증합니다.

- **Performance Highlights**: 연구 결과, 확산 모델(difusion models)이 기존의 밀집 예측(U-Net-based) 접근법보다 높은 품질의 CTA 이미지를 생성할 수 있음을 보여주었습니다. 이는 CTA와 TOF-MRA 간의 이미지 간 변환(task)에서 기존의 한계를 극복하는 중요한 성과로 평가됩니다.



### Towards Real-Time Generation of Delay-Compensated Video Feeds for Outdoor Mobile Robot Teleoperation (https://arxiv.org/abs/2409.09921)
Comments:
          8 pages, 4 figures, 3 tables

- **What's New**: 이 연구에서는 농업 로봇의 원격 조작을 원활하게 하기 위해 지연 보상을 위한 모듈형 학습 기반 비전 파이프라인을 제안합니다. 다양한 성장 단계에서의 오프라인 평가를 통해 제안된 접근 방식이 현재의 최신 기술들에 비해 더 정확한 이미지를 생성함을 입증하였습니다.

- **Technical Details**: 제안한 파이프라인은 단안 메트릭 깊이 추정(Monocular Metric Depth Estimation) 모델, 로봇 운동학 모델, 효율적인 구형 렌더러(Sphere-based Renderer), 학습 기반 인페인팅 모델을 포함합니다. 이 구조는 real-world 환경에서의 데이터에 통합되어 실제 시간 연산을 가능하게 합니다.

- **Performance Highlights**: 연구 결과는 제안된 파이프라인이 기존의 전통적 이미지 처리 방법 및 최첨단 학습 기반 이미지 생성 접근 방식과 비교하여 우수한 성능을 보이며, 실제 환경에서의 로봇 데이터 통합을 통해 지연 보상 효과를 성공적으로 보여주었습니다.



### Enhancing Visual Inertial SLAM with Magnetic Measurements (https://arxiv.org/abs/2409.09904)
- **What's New**: 이 연구는 자기계측기(magnetometer) 측정값을 긴밀하게 결합하여 비주얼 관성 오도메트리(Visual Inertial Odometry, VIO)를 확장한 내용을 보여줍니다. 핵심 프레임의 슬라이딩 윈도우를 최적화하고, IMU의 방향 전파 결과를 기반으로 자기계측기의 측정값을 효율적으로 변환합니다.

- **Technical Details**: 슬라이딩 윈도우 최적화(graph optimization) 기법을 사용하여 재투영 오류와 상대 관성이론적 오류 및 자기계측기의 방향 오류를 최소화합니다. IMU 전집합(preintegration) 알고리즘을 활용하여 모든 측정값에 대한 자기계측기의 잔여값을 효율적으로 계산하며, 자기계측값은 타원체 맞춤 알고리즘(ellipsoid fitting algorithm)을 통해 보정됩니다.

- **Performance Highlights**: 제안된 VIO 확장은 방향 오류(orientation error)를 크게 줄이고, 자기 북쪽(magnetic north)에 대한 진정한 요우(yaw) 방향을 복원하는데 기여합니다. 특히, 플로리다와 멕시코의 수중 동굴에서 수행된 실험 결과는 기존 동굴 지도와의 정렬 및 방향 오류가 현저히 줄어든 결과를 보여줍니다.



### Revisiting Physical-World Adversarial Attack on Traffic Sign Recognition: A Commercial Systems Perspectiv (https://arxiv.org/abs/2409.09860)
Comments:
          Accepted by NDSS 2025

- **What's New**: 본 논문은 기존의 학술 Traffic Sign Recognition (TSR) 모델에 대한 공격을 상업적 TSR 시스템에 적용한 첫 번째 대규모 측정을 수행하였습니다. 이 연구는 상업적 TSR 시스템에서 물리적 적대적 공격이 미치는 영향을 규명하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 연구는 여러 상업적 차량 모델을 대상으로 하여 숨기기 공격(hiding attacks)과 나타내기 공격(appearing attacks)의 효과를 평가하였습니다. 특히, 상업적 TSR 시스템 내의 공간 기억 디자인(spatial memorization design)이 공격의 성공률에 미치는 영향을 분석하며, 이는 TSR 시스템 수준의 공격 성공률을 평가하는 새로운 메트릭(metric) 설계를 기반으로 하고 있습니다.

- **Performance Highlights**: 시험 결과, 특정 상업적 TSR 시스템의 기능에 대해 기존 공격 연구에서 100%의 높은 성공률이 관찰되었으나, 전반적인 공격 성공률은 6.67%로 예상보다 훨씬 낮은 수치를 기록했습니다. 이는 상업적 TSR 시스템에서 공간 기억 디자인이 공격 효과에 미치는 영향을 고려하지 않았기 때문임을 보여줍니다.



### NARF24: Estimating Articulated Object Structure for Implicit Rendering (https://arxiv.org/abs/2409.09829)
Comments:
          extended abstract as submitted to ICRA@40 anniversary conference

- **What's New**: 이번 연구에서는 로봇이 복잡하게 연결된 아티큘레이트(Articulated) 객체를 인식하고 렌더링하기 위한 새로운 방법인 NARF24를 소개합니다. 이 방법은 소수의 수집된 장면에서 공통적인 Neural Radiance Field (NeRF) 표현을 학습하여 이미지 분할을 기반으로 각각의 연결 및 관절 파라미터를 추정할 수 있게 합니다.

- **Technical Details**: NARF24는 색상(RGB) 이미지와 이미지 공간에서의 파트 분할을 함께 처리하여, 각 장면 내의 파트 포인트 클라우드를 생성하고 Teaser++를 사용하는 방법으로 클라우드들을 등록합니다. 그렇게 생성된 연결 정보를 활용해 관절 유형과 파라미터를 추정하며, Chamfer distance를 기반으로 관절 연결성과 분류를 예측합니다.

- **Performance Highlights**: 초기 실험 결과, NARF24는 실제 데이터 세트에서 clamp 객체의 관절 추정치를 성공적으로 추출할 수 있음을 보여주었습니다. 제한된 이미지 분할 마스크 사항에서도 적절한 관절 추정치를 얻을 수 있었으며, NARF24는 소량의 분할 레이블로도 효과적인 렌더러 생성을 가능하게 함을 증명하였습니다.



### Domain and Content Adaptive Convolutions for Cross-Domain Adenocarcinoma Segmentation (https://arxiv.org/abs/2409.09797)
Comments:
          5 pages, 1 figure, 1 table

- **What's New**: 이번 연구는 Cross-Organ and Cross-Scanner Adenocarcinoma Segmentation (COSAS) 챌린지에서 U-Net 기반의 segmentation framework를 도입하여, 다양한 조직학적 샘플의 cross-domain adenocarcinoma segmentation 문제를 해결하는 방법론을 제시합니다.

- **Technical Details**: 이 논문에서는 nnU-Net 및 Domain and Content Adaptive Convolution (DCAC) 모듈을 이용하여 adenocarcinoma segmentation 작업을 수행하였습니다. DCAC 모듈은 domain-adaptive convolution (DAC) 및 content-adaptive convolution (CAC) 모듈로 구성되어 있으며, 이는 encoder-decoder 구조에 통합되어 도메인 변화에 대한 저항력을 향상시킵니다.

- **Performance Highlights**: 최종 테스트 세트에서 cross-organ track에 대해 0.8020, cross-scanner track에 대해 0.8527의 segmentation 점수를 달성하며, 두 트랙에서 모두 최상위 성적을 기록하였습니다.



### Universal Topology Refinement for Medical Image Segmentation with Polynomial Feature Synthesis (https://arxiv.org/abs/2409.09796)
Comments:
          Accepted by the 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2024)

- **What's New**: 이 논문에서는 의료 이미징에서의 분할(segmentation) 정확성 외에도 topology(위상) 정확성을 고려하는 새로운 접근법을 제시합니다. 제안된 방법은 기존의 모델에 비해, 여러 종류의 topological errors(위상 오류)를 해결할 수 있는 플러그 앤 플레이(post-processing) 구조를 가지고 있습니다.

- **Technical Details**: 제안된 topology refinement 네트워크는 기존의 segmentation 네트워크와 연결될 수 있으며, 가상의 segmentation 데이터를 사용하여 다양한 위상 오류를 학습할 수 있습니다. 이에 사용된 기법은 Stone-Weierstrass 정리에 영감을 받아 제작된 topology-perturbation masks를 포함하고 있으며, 이는 orthogonal polynomial bases의 무작위 샘플을 기반으로 합니다. 이러한 다항식 기법은 오류의 종류와 정도를 조정할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 본 연구에서 개발된 모델은 기존에서 사용되는 topology-aware loss 기능이나 post-processing 방식보다 더 나은 성능을 보여주며, 3D CoW 데이터셋에서도 우수한 결과를 기록했습니다. 게다가, 이 방법론은 기존의 학습 기반 모델과 결합하여 성능 향상이 가능하다는 점에서 뛰어난 활용성을 보여줍니다.



### Learning Two-factor Representation for Magnetic Resonance Image Super-resolution (https://arxiv.org/abs/2409.09731)
- **What's New**: 본 연구에서는 저해상도 MR 이미지로부터 고해상도 이미지를 생성하기 위한 새로운 두 요소 모델(two-factor model)을 제안합니다. 이 방법은 학습 가능 기반(basis)과 계수(coefficient) 요소의 선형 조합으로 강력한 연속 볼류메트릭(representational volumetric) 표현을 가능하게 합니다.

- **Technical Details**: 제안한 방법은 MR 이미지의 강도 신호(intensity signals)를 학습 가능 기반과 계수 요소로 분해하여, 고해상도 이미지를 효율적으로 재구성하는 것입니다. 또한 좌표 기반 인코딩(coordinate-based encoding)을 도입하여 sparse voxel 간의 구조적 관계를 잘 캡처하고, 미관찰 영역의 부드러운 보완을 용이하게 합니다.

- **Performance Highlights**: BraTS 2019와 MSSEG 2016 데이터셋에서 실험한 결과, 본 방법은 현재 최고의 성능(state-of-the-art performance)을 달성하였으며, 특히 대규모 업샘플링(scale MR image super-resolution)에서 뛰어난 시각적 충실도와 견고성을 제공합니다.



### Precise Pick-and-Place using Score-Based Diffusion Networks (https://arxiv.org/abs/2409.09725)
Comments:
          8 pages, 7 figures. Project webpage: this https URL

- **What's New**: 본 논문에서는 로봇 조작 작업 내 픽 앤 플레이스 작업의 정확성을 높이기 위한 새로운 coarse-to-fine continuous pose diffusion 방법을 제안합니다. 이를 통해 물체의 자세를 보다 정확하게 인식할 수 있으며, 이는 픽 앤 플레이스 성공률과 전반적인 조작 정밀도를 향상시킵니다.

- **Technical Details**: 이 방법론은 RGB-D 카메라에서 투사된 top-down RGB 이미지를 활용하며, coarse-to-fine 아키텍처를 채택하여 coarse 및 fine 모델의 효율적인 학습을 가능하게 합니다. 특히, continuous pose estimation에 초점을 맞추어 회전 각도에 대한 더 정밀한 물체 조작이 가능해집니다. 또한, 제한된 데이터로도 효과적인 훈련이 가능하도록 pose 및 color augmentation 기법을 적용합니다.

- **Performance Highlights**: 광범위한 시뮬레이션 및 실제 환경에서의 실험을 통해 제안된 방법의 효과를 검증하였습니다. 실험 결과, 적은 양의 훈련 데이터로도 높은 정밀도와 성공률을 달성하였으며, 기존의 기준선 성능을 초월하는 결과를 보였습니다.



### Finetuning CLIP to Reason about Pairwise Differences (https://arxiv.org/abs/2409.09721)
Comments:
          10 pages

- **What's New**: 이번 논문에서는 CLIP의 이미지 임베딩 스페이스에서의 차이점을 텍스트 설명과 정렬시키기 위한 새로운 방법론인 PC-CLIP (Pairwise Comparison CLIP)을 제안합니다. 이를 통해 이미지 간의 비교를 통해 더 나은 추론 능력을 달성하고, 비교 프롬프팅(comparative prompting) 기법을 도입하여 분류 성능을 높였습니다.

- **Technical Details**: PC-CLIP은 대형 언어 모델(LLM)을 이용하여 이미지-캡션 데이터셋에서 이미지 간의 차이를 설명하는 텍스트 생성하여 이를 CLIP의 이미지 임베딩과 정렬하는 과정입니다. 이 과정을 통해 CLIP의 임베딩 스페이스에서 기하학적 속성을 자연스럽게 만족시킬 수 있도록 하였으며, 최근의 실험 결과를 통해 향상된 기하학적 특성을 입증했습니다.

- **Performance Highlights**: PC-CLIP은 차별적인 분류 작업에서 큰 성능 향상을 보였으며, 14 포인트까지 정확도 향상을 달성했습니다. 또한 기존 CLIP의 성능과 비교했을 때, 다양한 이미지 분류 작업에서 개선된 제로샷(zero-shot) 분류 성능을 나타냈습니다. 마지막으로, 기하학적 속성이 잘 반영된 임베딩을 통해 이미지 생성 작업에서도 우수한 결과를 보여주었습니다.



### Reliable Multi-View Learning with Conformal Prediction for Aortic Stenosis Classification in Echocardiography (https://arxiv.org/abs/2409.09680)
Comments:
          This preprint has not undergone any post-submission improvements or corrections. The Version of Record of this contribution is published in: International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer (2024) under the same title

- **What's New**: 이 논문에서는 초음파 진단의 한계인 2D 이미지를 3D 해부학적 정보와 결합하여 불확실성을 시간에 따라 재학습하는 방법인 Re-Training for Uncertainty (RT4U)를 소개합니다. RT4U는 불확실성을 약한 정보를 가진 입력 데이터에 도입하여 분류 정확도를 높이는 방법입니다.

- **Technical Details**: RT4U는 데이터 중심의 접근 방식으로, 약한 정보를 가진 입력의 불확실성을 증가시키는 모델-불가지론적(training agnostic) 기법입니다. 기존의 심장 판막 질환 분류 기법에 통합하여 사용할 수 있으며, conformal prediction 기술과 결합할 경우, 예측 세트를 동적으로 크기 조정하여 높은 정확도로 실제 클래스가 포함되어 있음을 보장합니다.

- **Performance Highlights**: 다양한 3개의 데이터셋(TMED-2, 비공식 AS 데이터셋, CIFAR-10 유사 장난감 데이터셋)에서 RT4U는 모든 데이터셋에서 향상된 결과를 보였으며, 이는 자동화된 심장 판막 질환 검출의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Bias Begets Bias: The Impact of Biased Embeddings on Diffusion Models (https://arxiv.org/abs/2409.09569)
Comments:
          19 pages, 4 figures

- **What's New**: 본 연구는 Text-to-Image (TTI) 시스템의 확산 모델에 내재된 편향을 조사하고 이를 해결하기 위한 새로운 공정성 기준과 방법을 제안합니다. 특히 임베딩 공간의 편향이 생성된 이미지와 프롬프트 간의 정렬 평가에 미치는 영향을 분석합니다.

- **Technical Details**: 정량적 공정성 기준을 제시하여 모델의 내부 표현에 기반하여 통계적 그룹 공정성을 정의합니다. 부정확한 임베딩은 비공정한 생성물을 초래하고 이러한 부정확성이 TTI 모델의 평가에 영향을 미치는 메커니즘을 설명합니다. 특정한 임베딩 공간의 관점에서 새로운 공정성 조건을 확립합니다.

- **Performance Highlights**: BIAS (편향) 임베딩을 사용하는 TTI 모델은 공정한 평가를 방해하며, CLIP와 같은 다중 모드 임베딩의 영향을 통해 공정성 평가의 한계를 나타냅니다. 연구 결과에 따르면, 임베딩의 공정성이 전체 모델의 공정성에 중대한 영향을 미친다는 사실을 강조합니다.



### Enhancing Printed Circuit Board Defect Detection through Ensemble Learning (https://arxiv.org/abs/2409.09555)
- **What's New**: 이 논문은 서로 다른 PCB 결함 탐지 모델들의 시너지를 활용하지 않은 기존 연구의 한계를 극복하기 위해 앙상블 학습 전략을 적용한 종합 검사 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 EfficientDet, MobileNet SSDv2, Faster RCNN, YOLOv5의 네 가지 최첨단 PCB 결함 감지 모델을 포함하여, 이러한 모델들을 하이브리드 투표 전략으로 앙상블하여 결함 탐지 정확도를 향상시키고자 합니다.

- **Performance Highlights**: 이 앙상블 학습 프레임워크는 다양한 PCB 결함을 감지하는 데 있어 95%의 정확도를 달성하며, 이는 개별 모델보다 상당히 향상된 성능임을 나타냅니다.



### MANGO: Disentangled Image Transformation Manifolds with Grouped Operators (https://arxiv.org/abs/2409.09542)
Comments:
          Submitted to IEEE ICASSP 2025. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문에서는 MANGO라는 새로운 접근 방식을 제안하여, 다양한 이미지 변환을 설명하는 disentangled operators를 효율적으로 학습합니다. 이 방법은 변환을 원하는 방향으로 정의할 수 있어, 학습된 operator의 의미를 개선합니다.

- **Technical Details**: MANGO는 이미지 변환을 그룹화된 operator로 나누어 학습할 수 있도록 하며, 각각의 operator가 독립적인 잠재 서브스페이스에 작용하도록 제약함으로써 disentangled 특성을 강화합니다. 이 방식을 통해, 실제적인 이미지 생성이 가능하며, 추정된 변환 속도는 이전 방법에 비해 100배 향상됩니다.

- **Performance Highlights**: MANGO는 이전의 manifold autoencoder (MAE) 기법보다 100배 더 빠른 한 단계 훈련 루틴을 제공하며, 이미지 변환 조합 가능성을 실험을 통해 입증했습니다.



### Self-Prompting Polyp Segmentation in Colonoscopy using Hybrid Yolo-SAM 2 Mod (https://arxiv.org/abs/2409.09484)
- **What's New**: 이 논문은 Segment Anything Model (SAM 2)와 YOLOv8 모델을 통합하여 대장 내시경 이미지에서 폴립(segmentation) 검출을 위한 새로운 방법을 제안합니다. 이 접근 방식은 YOLOv8의 경계 상자 예측을 활용하여 SAM 2의 입력 프롬프트를 자동으로 생성함으로써 수작업 주석의 필요성을 줄입니다.

- **Technical Details**: 제안한 모델은 YOLO와 SAM 2 두 가지 최첨단 알고리즘을 통합합니다. YOLO 모델은 폴립을 감지하고 경계 상자를 생성하며, 이 경계 상자는 SAM 2 모델의 입력 프롬프트로 사용됩니다. SAM 2 모델은 경량화된 구조와 높은 정확성을 가지고 있으며, 경계 상자의 좌표를 바탕으로 폴립의 세부 segmentation을 수행합니다.

- **Performance Highlights**: 모델은 5개의 벤치마크 대장 내시경 이미지 데이터셋과 2개의 비디오 데이터셋에서 성능을 검증하였으며, 기존 최첨단 모델들을 초과하는 성능을 입증했습니다. 특히, 경계 상자 주석만 사용하여 높은 segmentation 정확도를 달성함으로써 주석 시간과 노력을 크게 줄였습니다.



### MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry (https://arxiv.org/abs/2409.09479)
- **What's New**: 이번 논문에서는 MAC-VO라는 새로운 학습 기반의 스테레오 VO(Visual Odometry) 방법을 제안합니다. 이 방법은 학습된 메트릭스 인식(metrice-aware) 매칭 불확실성을 활용하여 두 가지 목적을 달성합니다: 키포인트 선택 및 위치 그래프 최적화에서의 잔차 가중치. 전통적인 기하학적 방법과는 달리, MAC-VO는 저품질 특징을 필터링하기 위해 학습된 불확실성을 사용합니다.

- **Technical Details**: MAC-VO는 메트릭스 인식 공분산(covariance) 모델을 통해 3D 키포인트의 공간 오류와 서로 다른 축 간의 상관관계를 포착합니다. 또한, 2D 불확실성에 기반한 3D 공분산 모델을 제시하여 기존의 스케일에 관계없는 대각선 공분산 행렬보다 더 정확한 특징을 제공합니다. 최적화 과정에서 3D 공분산을 통해 등록된 키포인트 간의 거리를 최소화하여 상대적인 모션을 최적화합니다.

- **Performance Highlights**: MAC-VO는 공개 벤치마크 데이터셋에서 기존의 VO 알고리즘 및 일부 SLAM(Simultaneous Localization and Mapping) 알고리즘보다 뛰어난 성능을 보여주었습니다. 어려운 환경에서도 정확한 위치 추정이 가능하며, 이는 자율 시스템의 의사결정에 도움이 될 수 있습니다.



### From FDG to PSMA: A Hitchhiker's Guide to Multitracer, Multicenter Lesion Segmentation in PET/CT Imaging (https://arxiv.org/abs/2409.09478)
- **What's New**: 이번 논문은 autoPET III 챌린지에 대한 우리의 솔루션을 소개하며, nnU-Net 프레임워크를 활용한 멀티 트레이서(multi-tracer) 및 멀티 센터(multi-center) 일반화를 목표로 하고 있습니다. 이를 위해 CT, MR, PET 데이터셋에서 다중 알고리즘 사전 훈련(multi-modal pretraining) 및 부정합 데이터 증대(misalignment data augmentation) 기법을 사용했습니다.

- **Technical Details**: nnU-Net 프레임워크에 기반한 우리의 방법론은 ResEncL 아키텍처를 채택하였습니다. 우리는 3D 의료 영상에서 4000 에폭(epoch) 동안 훈련을 진행하며 각 데이터셋에 대해 별도의 세그멘테이션 헤드를 적용했습니다. 또한, organ supervision을 도입하여 다양한 기관을 식별하고, 레션 세그멘테이션의 정확성을 향상시켰습니다. 이 과정에서 배치 사이즈(batch size)는 24, 패치 사이즈(patch size)는 [192,192,192]로 설정하였습니다.

- **Performance Highlights**: 우리가 제안한 모델은 Dice 점수(Dice score)가 68.40에 달했으며, 이는 기본 nnU-Net(57.61) 및 ResEncL(65.31)에 비해 상당한 성능 향상을 보여줍니다. 또한, 잘못 분류된 양성(volume of false positives: FPvol: 7.82) 및 음성(volume of false negatives: FNvol: 10.35) 부피도 감소하여 효과성을 입증했습니다.



### Estimating Neural Orientation Distribution Fields on High Resolution Diffusion MRI Scans (https://arxiv.org/abs/2409.09387)
Comments:
          16 pages, 8 figures, conference: Medical Image Computing and Computer-Assisted Intervention (MICCAI)

- **What's New**: 이번 연구에서는 Orientation Distribution Function (ODF)를 추정하기 위한 새로운 방법인 HashEnc를 제안합니다. HashEnc는 grid-hash-encoding 기반의 접근 방식을 통해 ODF 필드의 공간적으로 연속적인 추정을 구현하며, 기존의 SIREN 방식에 비해 구조적 및 텍스처적 특성을 유지하는 데 효과적입니다.

- **Technical Details**: HashEnc는 grid-like 지역 임베딩을 사용하여 ODF를 추정합니다. 이 방법은 작은 MLP (Multi-Layer Perceptron) 가중치만 업데이트하며, 전체 네트워크 가중치의 반복적 평가 없이도 훈련을 진행할 수 있습니다. 이를 통해 HashEnc는 이미지 품질이 10% 향상되며, 기존 방법보다 최대 3배 빠른 훈련 속도를 달성할 수 있습니다.

- **Performance Highlights**: HashEnc는 cerebellum과 같은 고해상도 영역에서 우수한 성능을 보이며, 세밀한 구조와 텍스처의 세부 사항을 학습하는 능력이 뛰어납니다. 또한 ODF 추정에서 지나치게 매끄럽게 결과를 제공하는 SIREN 방식에 비해 HashEnc는 더 나은 세부 표현을 보여줍니다.



### MotionTTT: 2D Test-Time-Training Motion Estimation for 3D Motion Corrected MRI (https://arxiv.org/abs/2409.09370)
- **What's New**: 이번 논문에서는 MRI(자기 공명 영상)에서 발생하는 환자의 움직임으로 인한 아티팩트 문제를 해결하기 위한 딥러닝 기반의 테스트 타임 트레이닝(test-time training) 방법을 제안합니다. 이는 3D 비틀림 모션(3D rigid motion) 추정을 위한 첫 번째 딥러닝 기반 방법으로, 기존의 접근 방식을 한 단계 발전시켰습니다.

- **Technical Details**: 제안하는 방법은 2D 재구성 네트워크를 사용하여 3D에서 비틀림 모션을 추정합니다. 이 과정에서 모션이 없을 경우 손실(loss)이 작아지도록 최적화하며, 이를 통해 정확한 모션 추정(motion estimation)을 가능하게 합니다. 네트워크를 통해 전달된 모션 매개변수를 최적화하여 실제 움직임을 보정하고, 보다 정확한 모션 보정 이미지(motion-corrected images)를 재구성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 단순한 신호 모델 및 신경망 모델에 대해 모션 매개변수를 확실하게 재구성할 수 있음을 증명하였습니다. 또한, 회고적 시뮬레이션(retrospectively simulated motion)과 실제 움직임으로 오염된 데이터(prospectively collected real motion-corrupted data)에서 모두 효과적임을 보여주었습니다.



### Beta-Sigma VAE: Separating beta and decoder variance in Gaussian variational autoencoder (https://arxiv.org/abs/2409.09361)
Comments:
          Accepted for ICPR 2024

- **What's New**: 이 연구에서는 Variational Autoencoder (VAE)의 흐릿한 출력 문제를 다루며, Gaussian decoder의 분산 및 beta-VAE의 beta 값을 활용하여 이를 해결하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법인 Beta-Sigma VAE (BS-VAE)는 beta와 decoder variance (σ²_x)를 명확하게 분리하여 모형의 성능을 향상시킵니다. 이전 방식에서는 두 매개변수가 혼동되어 모델 분석과 성능 향상을 방해했습니다. BS-VAE는 rate-distortion curve를 기반으로 적절한 decoder variance를 얻어내며, 매개변수 조정 가능성을 제공합니다.

- **Performance Highlights**: BS-VAE는 자연 이미지 합성 (natural image synthesis)에서 뛰어난 성능을 발휘하며, 기존 VAE보다 더 예측 가능한 분석을 가능하게 합니다. 실험 평가에서는 컴퓨터 비전 데이터셋에서의 성능 향상을 확인하였으며, 다양한 VAE 변형에 적용 가능하다는 장점을 가지고 있습니다.



### VOMTC: Vision Objects for Millimeter and Terahertz Communications (https://arxiv.org/abs/2409.09330)
- **What's New**: 본 논문은 밀리미터파 및 테라헤르츠 통신을 위한 새로운 비전 데이터셋인 VOMTC(Vision Objects for Millimeter and Terahertz Communications)를 제안합니다. 이 데이터셋은 20,232 쌍의 RGB 및 깊이 이미지로 구성되어 있으며, 각 이미지는 사람, 휴대폰, 노트북의 세 가지 객체 카테고리로 라벨링되어 있습니다.

- **Technical Details**: VOMTC 데이터셋은 다양한 환경(교실, 카페테리아, 엘리베이터 등)에서 수집된 이미지로 이루어져 있으며, 이 데이터셋은 DL(Deep Learning) 기반 CV(Computer Vision) 기술을 활용하여 무선 통신 작업에 필요한 훈련 데이터를 제공합니다. 또한 VBM(VOMTC-aided Beam Management) 기법을 통해 특정 객체(휴대폰)의 위치를 식별하고, 이를 바탕으로 방향성 빔을 생성하는 방법을 제시합니다.

- **Performance Highlights**: VOMTC 훈련된 객체 탐지기를 활용하여 기존의 5G 빔 관리 기술에 비해 36% 이상의 데이터 전송 속도 향상을 달성하였으며, 그 효과는 DL 기반 CV 기술을 무선 통신에 적용하는 데 있어 실질적인 이점을 제공합니다.



### ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models (https://arxiv.org/abs/2409.09318)
- **What's New**: 본 논문에서는 Multimodal Large Language Models (MLLMs)의 'hallucination'(환각) 문제를 다루며, 기존의 정적인 평가 방식에서 벗어나 동적인 프로토콜인 ODE(Open-Set Dynamic Evaluation)를 제안합니다. 이 프로토콜은 모델이 실제 개체를 얼마나 잘 인식하는지를 평가하기 위해 그래프 구조를 사용하여 실세계 개념 간의 연관성을 모델링하고, 새로운 샘플을 생성합니다.

- **Technical Details**: ODE는 Open-set(열린 집합) 접근 방식을 기반으로 하며, 기존의 정적인 벤치마크에서 발생할 수 있는 데이터 오염(data contamination)을 피하는 데 중점을 두고 있습니다. 이는 다양한 개념 조합을 동적으로 생성하여 평가하는 방식으로, 모델이 특정 작업의 핵심 개념을 진정으로 이해하고 있는지를 검증합니다. ODE는 일반적 개념과 구체적 도메인에 대한 다양한 테스트 데이터를 생성하기 위해 네 단계의 프로세스를 포함합니다: 1) 그래프 구조로 실세계 시나리오 모델링, 2) 시각적 정보의 개념적 설계, 3) 이미지 생성 및 필터링, 4) 텍스트 템플릿 설계.

- **Performance Highlights**: 실험 결과 ODE로 생성된 샘플을 사용할 경우 MLLMs의 환각 발생률이 높아지며, 기존의 정적인 벤치마크보다 더 향상된 성능 평가를 보여줍니다. 각 모델의 환각 경향성을 분석한 결과, 다양한 분포 상황에서 모델의 한계와 능력 경계를 식별할 수 있었습니다. 이를 통해 ODE의 효과성과 포괄성을 검증하였습니다.



### Real-Time Stochastic Terrain Mapping and Processing for Autonomous Safe Landing (https://arxiv.org/abs/2409.09309)
- **What's New**: 본 논문은 최적의 착륙 안전성을 평가하기 위한 실시간 확률적 지형 매핑 알고리즘을 소개합니다. 이 알고리즘은 샘플링된 지점 간의 지형적 불확실성을 고려하여 드로니 삼각분할(Delaunay triangulation) 및 국소 가우시안 프로세스 회귀(local Gaussian process regression)를 활용하여 가우시안 디지털 고도 지도(Gaussian DEM)를 효율적으로 구축합니다.

- **Technical Details**: 지형의 불확실성을 효과적으로 처리하는 실시간 확률적 위험 탐지(hazard detection) 알고리즘이 개발되었습니다. 기하학적 분석을 통해 지역 경사(local slope)와 거칠기(roughness) 평가가 수행되며, 이 과정에서 로컬 평면(local plane)의 비싼 계산을 피하는 방법이 채택됩니다.

- **Performance Highlights**: 이 방법론은 실제 및 합성 지형 데이터를 조합하여 효과성을 입증하며, 제한된 데이터를 기반으로 지형 분석 및 착륙 안전성 평가를 지원하여 새로운 확률적 유도 알고리즘의 발달을 이끌 가능성을 제시합니다.



### Robust Training of Neural Networks at Arbitrary Precision and Sparsity (https://arxiv.org/abs/2409.09245)
- **What's New**: 본 연구에서는 정량화(quantization)와 희소성(sparsification)으로 인한 비연속적인 작동이 역전파(backpropagation)에 미치는 영향을 극복하기 위한 새로운 솔루션을 제안합니다. 이를 위해, 훈련 과정에서 정량화와 희소성을 섭동(perturbation)으로 설정하고, 접근 방법으로는 여 Ridge 회귀(ridge regression)를 기반으로 합니다.

- **Technical Details**: 제안하는 방법은 세 가지 기본 단계로 구성됩니다: 1) 정량화를 위한 Affine Transform: 입력 신호를 크기 조절하는 초기 변환; 2) 섭동 주입: 정량화의 효과를 모델링하기 위한 임의의 섭동 δ를 신호에 주입; 3) 복원을 위한 Denoising Affine Transform: 양자화 노이즈를 완화하며 원래 신호를 효과적으로 복원합니다. 이 방법은 기존의 모델 아키텍처 및 훈련 레시피와 호환되어 많은 변화 없이 적용 가능하며, 다양한 효율적인 신경망의 개발을 가능하게 합니다.

- **Performance Highlights**: 제안하는 방법은 초저정밀 모델에 대한 최신 결과를 달성하며, 기존 아키텍처와 표준 훈련 레시피를 사용하여도 우수한 성능을 발휘합니다. 또한 희소 신경망(sparse neural networks)과 시간 이진 신경망(temporal binary neural networks)의 성공적인 훈련을 통해 인공 신경망과 생물학적 신경망 간의 간극을 좁힐 수 있는 가능성을 보여줍니다.



### Spectral U-Net: Enhancing Medical Image Segmentation via Spectral Decomposition (https://arxiv.org/abs/2409.09216)
- **What's New**: Spectral U-Net은 의학 이미지 분할을 위한 새로운 딥 러닝 네트워크로, DTCWT(Dual Tree Complex Wavelet Transform)를 사용하여 정보 손실을 줄이고 해상도 재구축을 향상시키는 구조를 제안합니다.

- **Technical Details**: Spectral U-Net는 Wave-Block과 iWave-Block으로 구성되어 있으며, DTCWT를 통해 특징 맵을 고주파와 저주파 성분으로 분해하고, 다운샘플링과 업샘플링 과정에서 정보 손실을 최소화합니다. 인코딩 단계에서 DTCWT를 사용하여 특징 맵을 처리하고, 디코딩 단계에서는 iDTCWT를 통해 더 높은 해상도의 특징 맵을 복원합니다.

- **Performance Highlights**: Retina Fluid, Brain Tumor, Liver Tumor 세그멘테이션 데이터셋을 nnU-Net 프레임워크에 대해 평가하고, 기존 방법들에 비해 Spectral U-Net이 우수한 성능을 보임을 입증했습니다.



### Hierarchical Hypercomplex Network for Multimodal Emotion Recognition (https://arxiv.org/abs/2409.09194)
Comments:
          The paper has been accepted at MLSP 2024

- **What's New**: 이번 연구에서는 다채로운 생리적 신호를 기반으로 하는 감정 인식 시스템을 위한 새로운 하이퍼복합 네트워크 모델을 제안합니다. 특히, 인코더와 융합 모듈 모두 하이퍼복합 영역에서 작동하도록 하여, 각 신호의 채널 간 관계와 모달리티 간의 상관 관계를 효과적으로 학습합니다.

- **Technical Details**: 제안된 Hierarchical Hypercomplex (H2) 모델은 계층 구조로 구성되어 있으며, 여기서 각 인코더는 입력 신호의 차원 수에 따라 하이퍼파라미터를 설정하여 작동합니다. 인코더는 parameterized hypercomplex convolutions (PHCs)을 통해 각 모달리티의 채널 간 관계를 학습하고, 융합 모듈은 parameterized hypercomplex multiplications (PHMs)을 통해 모달리티 간의 관계를 학습합니다.

- **Performance Highlights**: MAHNOB-HCI 데이터셋에서 아우로살(activation) 및 발란스(valence) 분류 성능이 40.20% 및 57.11% 향상된 F1-score을 각각 0.557 및 0.685로 달성하였습니다.



### FiAt-Net: Detecting Fibroatheroma Plaque Cap in 3D Intravascular OCT Images (https://arxiv.org/abs/2409.09188)
- **What's New**: 이 논문에서는 심혈관 사건의 위험을 평가하기 위해 fibroatheroma(FA) 플라크를 탐지하고 세분화하는 새로운 딥러닝 기반 접근법인 FiAt-Net을 제안합니다. 이 방법은 3D intravascular optical coherence tomography(IVOCT) 이미지를 사용하여 FA와 플라크 캡을 식별합니다.

- **Technical Details**: FiAt-Net은 IVOCT 이미지의 2D 프레임을 클러스터링하고, 데이터 불균형 문제를 해결하기 위해 binary partitioning 방법을 적용합니다. 또한 auxiliary images를 생성하여 IVOCT의 강도 변화를 캡처하고, multi-head self-attention 메커니즘을 통해 다양한 스케일의 정보를 융합합니다.

- **Performance Highlights**: FiAt-Net은 3D IVOCT 관상동맥 이미지 데이터셋에서 높은 성능을 기록하였으며, IVOCT 이미지 내 FA 캡을 정확하게 탐지하는 데 효과적임을 입증했습니다.



### Phikon-v2, A large and public feature extractor for biomarker prediction (https://arxiv.org/abs/2409.09173)
- **What's New**: 이 논문에서는 100개 이상의 공개 코호트에서 수집한 4억 6천만 개의 병리 타일로 구성된 다양한 데이터 세트를 이용하여, DINOv2를 사용한 대규모 자가 지도 학습( Self-supervised learning ) 기반의 비전 트랜스포머 모델인 Phikon-v2를 훈련하고 이 모델을 공개했습니다. Phikon-v2는 이전 모델인 Phikon을 능가하며, 소유 데이터로 훈련된 다른 병리학 기초 모델들과 비슷한 성능을 보입니다.

- **Technical Details**: Phikon-v2는 460 million (억 6천만) 개의 병리 타일을 기반으로 트레이닝된 ViT-L (Vision Transformer Large) 모델로, DINOv2에 의해 학습되었습니다. 이 모델은 WSI (Whole Slide Images) 레벨의 바이오마커 예측과 질병 분류 작업에서 경쟁력 있는 성능을 보여줍니다. 또한 14개의 다양한 병리학 feature extractor와 비교하여 그 평가가 가장 포괄적입니다.

- **Performance Highlights**: 모델의 카운터파트인 Phikon (ViT-B)와 Phikon-v2 (ViT-L)는도 8가지 슬라이드 레벨 작업에서 통계적으로 유의미하게 (+1.75 AUC 증가) 향상된 성능을 보였습니다. 최신 기초 모델들이 임상 배치 대비 한계를 보일 수 있지만, 이들은 더 전문화되고 비용 효율적인 병리학 인코더 개발의 기초를 제공합니다.



### Trimming the Risk: Towards Reliable Continuous Training for Deep Learning Inspection Systems (https://arxiv.org/abs/2409.09108)
- **What's New**: 이 논문은 딥 러닝 (Deep Learning) 기반의 결함 감지 시스템을 유지 관리하기 위한 강력한 지속적 학습 (Continuous Training) 접근법을 제안합니다. 이 접근법은 두 단계의 필터링 과정을 통해 신뢰할 수 있는 데이터 선택을 사용하여 모델을 업데이트 합니다.

- **Technical Details**: 제안된 방법은 첫 번째 단계에서 신뢰도가 낮은 예측 결과를 필터링하고, 두 번째 단계에서는 변분 오토인코더 (Variational Auto-Encoder)와 픽셀 히스토그램을 사용하여 이미지 임베딩을 생성합니다. 이를 통해 ID (In-Distribution) 데이터로부터 크게 변동된 입력을 불량 데이터로 판단하고 제거합니다.

- **Performance Highlights**: 산업 검사의 실제 사례에서 적용한 결과, 잘못 레이블된 데이터의 8%만이 필터링 후에도 유지되었으며, 모델 성능이 생산 데이터에서 최대 14% 향상되었고, 원래 검증 데이터에 대한 결과 또한 손상되지 않았습니다.



### Inf-MLLM: Efficient Streaming Inference of Multimodal Large Language Models on a Single GPU (https://arxiv.org/abs/2409.09086)
- **What's New**: 이 논문에서는 MLLMs(다중모드 대형 언어 모델)의 효율적인 추론 프레임워크인 Inf-MLLM을 소개합니다. Inf-MLLM은 무한한 컨텍스트에서 단일 GPU에서 MLLMs의 스트리밍 추론을 가능하게 합니다.

- **Technical Details**: Inf-MLLM은 'attention saddles'라는 새로운 어텐션 패턴을 기반으로 하여 동적으로 최근 토큰과 관련 토큰을 캐시합니다. 또한, attention bias를 제안하여 MLLMs이 장기 종속성을 잡을 수 있도록 합니다. 이 프레임워크는 계산 복잡성을 줄이고 메모리 사용량을 감소시킵니다.

- **Performance Highlights**: Inf-MLLM은 4M 토큰의 긴 텍스트와 1시간 분량의 비디오를 가진 다중 라운드 대화에서 안정적인 성능을 달성하며, StreamingLLM보다 우수한 스트리밍 추론 품질과 H2O에 비해 2배 빠른 속도를 자랑합니다.



### HESSO: Towards Automatic Efficient and User Friendly Any Neural Network Training and Pruning (https://arxiv.org/abs/2409.09085)
Comments:
          preprint

- **What's New**: HESSO(Hybrid Efficient Structured Sparse Optimizer)는 DNN(Deep Neural Networks)의 자동화된 구조적 가지치기와 교육을 지원하는 새로운 최적화 기법으로, 튜닝 없이 높은 성능의 서브 네트워크를 생성하는데 중점을 두고 있습니다. 이와 함께, CRIC(Corrective Redundant Identification Cycle)를 통해 가지치기 과정에서 필수 구조의 손실을 방지할 수 있습니다.

- **Technical Details**: HESSO는 사전 훈련된 DNN에서 redundant structures를 효율적으로 제거하고, 튜닝이 거의 필요 없는 progressive pruning 전략을 채택하여 사용자 친화적인 인터페이스를 제공합니다. CRIC 메커니즘은 다중 샘플링 접근 방식을 통해 정확한 redundant identification을 도와줍니다.

- **Performance Highlights**: HESSO는 컴퓨터 비전, 자연어 처리 등 다양한 분야에서 최신 기술보다 경쟁력 있는 성능을 보여주었으며, 특히 DNN 아키텍처에 대한 지원을 확대했습니다. CRIC은 가지치기 후에도 성능 저하를 방지하는 효과적인 방법으로 기능합니다.



### Deep learning-based classification of breast cancer molecular subtypes from H&E whole-slide images (https://arxiv.org/abs/2409.09053)
Comments:
          16 pages, 5 figures (+4 supplementary figures), 4 tables

- **What's New**: 이번 연구에서는 H&E 염색된 전체 슬라이드 이미지(Whole Slide Images, WSI)를 사용하여 유방암의 분자 아형을 예측할 수 있는 가능성을 조사했습니다. 이는 기존의 면역조직화학(immunohistochemistry, IHC) 및 유전자 발현 프로파일링(gene expression profiling)과 같은 전통적인 방법들에 대한 대안으로 제시됩니다.

- **Technical Details**: 연구팀은 1,433개의 유방암 WSI를 사용하여 두 단계의 파이프라인을 구축했습니다. 첫 번째 단계에서는 종양(tile)과 비종양(tile)을 분류하여 종양 영역만 사용하였고, 두 번째 단계에서는 One-vs-Rest(OvR) 전략을 employed하여 네 개의 바이너리 OvR 분류기를 훈련시키고, 그 결과를 eXtreme Gradient Boosting(XGBoost) 모델을 사용해 집계했습니다.

- **Performance Highlights**: 이 파이프라인은 221개의 검증용 WSI에서 테스트되었으며, 종양 탐지에 대해 전체 macro F1 점수 0.95를, 분자 아형 분류에서는 0.73을 기록했습니다. 이러한 결과들은 감독형 심층 학습(supervised deep learning) 모델이 유방암의 분자 아형 분류를 위한 보조 도구로 활용될 가능성을 나타냅니다.



### OrthoDoc: Multimodal Large Language Model for Assisting Diagnosis in Computed Tomography (https://arxiv.org/abs/2409.09052)
Comments:
          8 pages, 1 figure

- **What's New**: OrthoDoc는 120,000개의 CT 이미지와 진단 보고서를 기반으로 훈련된 다중 모달 대형 모델로, CT 진단을 위해 특별히 설계되었습니다. 기존 모델들과 비교하여 우수한 진단 능력과 정확성을 보여줍니다.

- **Technical Details**: OrthoDoc는 CT 이미지와 진단 보고서의 다중 모달 학습을 통해 의료 이미징 특징을 습득하고, RAG(검색 보완 생성) 모듈을 이용해 모델의 환각(hallucination) 문제를 완화합니다. 이 모델은 복잡한 CT 이미지를 처리하고 자연어로 상세한 진단 보고서를 생성할 수 있습니다.

- **Performance Highlights**: OrthoDoc는 기존 상용 모델과 비교하여 골절, 관절염 및 종양과 같은 일반적인 정형외과 질환의 진단에서 91% 이상의 정확도를 달성했습니다. 또한, 희귀 및 복잡한 사례를 처리할 때 탁월한 일반화 능력을 보이며 임상 응용에서 실용적인 유용성을 입증하고 있습니다.



### AutoGeo: Automating Geometric Image Dataset Creation for Enhanced Geometry Understanding (https://arxiv.org/abs/2409.09039)
- **What's New**: AutoGeo라는 새로운 접근 방식을 통해 대규모 고품질 기하학 이미지 데이터셋을 자동으로 생성하는 방법을 제안하였습니다. AutoGeo-100k라는 10만 쌍의 이미지-텍스트를 포함하는 데이터셋을 구축하여 기하학적 문제 해결의 정확도를 향상시켰습니다.

- **Technical Details**: AutoGeo는 Augmented Geometry Clause System (AGCS), Rule-based Clause Selector (RCS), Sample Generator (SG)로 구성되어 있습니다. AGCS는 난이도에 따라 숫자 조항을 포함하며, RCS는 원하는 복잡도에 맞는 기하학 조항을 선택합니다. SG는 이 조항들을 통해 Python을 사용한 이미지 생성을 수행하고, ChatGPT를 통해 텍스트 설명을 생성합니다.

- **Performance Highlights**: AutoGeo-100k를 사용하여 여러 Multimodal Large Language Models (MLLMs)을 미세 조정한 결과, 기하학 이미지 이해 능력이 크게 향상되었으며, 기하학 캡셔닝과 수학적 추론 과제에서의 정확도가 개선되었습니다.



New uploads on arXiv(cs.AI)

### Instigating Cooperation among LLM Agents Using Adaptive Information Modulation (https://arxiv.org/abs/2409.10372)
- **What's New**: 이 논문은 LLM(대형 언어 모델) 에이전트를 인간 전략적 행동의 프록시로 결합하고, 강화 학습(Reinforcement Learning, RL)을 통해 이들 에이전트를 진화하는 전략적 상호작용에 참여시키는 새로운 프레임워크를 제안합니다. 결과적으로, 전통적인 에이전트 기반 시뮬레이션을 확장하여 LLM 에이전트의 전략적 의사결정 행동을 모델링하고, 정보 접근성을 조절하는 pro-social promoting RL agent(PPA)를 도입하여 사회 복지와 공동체 행동을 증진시킵니다.

- **Technical Details**: 이 프레임워크는 LLM 기반 에이전트를 사용하여 인간 상호작용을 위한 동적인 거버넌스 메커니즘을 생성하고, LLM과 인간 에이전트가 결합된 하이브리드 환경에서 공동체 행동을 향상시킵니다. RL 기반 거버넌스 에이전트는 전략적 행동에 대한 정보 접근 수준을 동적으로 조정함으로써 협력 수준을 높이며, 이를 통해 시스템의 자율성을 유지하고 게임 보상의 변화 없이 개입합니다. 이 방법론은 반복 게임, 특히 죄수의 딜레마(Prisoner’s Dilemma)에서 LLM 에이전트의 전략적 적응을 검증합니다.

- **Performance Highlights**: 이 논문에서는 강화 학습 에이전트가 정보를 동적으로 조절함으로써 협력률을 증가시키는 데 성공한다는 것을 보여줍니다. 또한, LLM 에이전트가 정보 접근의 변화에 적절하게 반응하며 행동을 조정할 수 있음을 입증하였고, 이는 기존의 정적 개입 방법과 비교하여 인상적인 결과를 낳았습니다. 그 결과, AI 매개 사회 역학 분야에 대한 중요한 통찰을 제공하고 있습니다.



### ReflectDiffu: Reflect between Emotion-intent Contagion and Mimicry for Empathetic Response Generation via a RL-Diffusion Framework (https://arxiv.org/abs/2409.10289)
- **What's New**: ReflectDiffu라는 경량화된 포괄적 프레임워크를 소개하며, 감정 전염(emotion contagion)과 의도 예측(intent prediction)을 통합하여 의미 있는 공감적 반응 생성을 달성합니다.

- **Technical Details**: ReflectDiffu는 Emotion-Contagion Encoder와 Intent Exploring-Sampling-Correcting 메커니즘을 활용하여 감정적 의도를 정밀하게 조정하고, 대화의 맥락을 공감적으로 이해할 수 있게 돕습니다.

- **Performance Highlights**: ReflectDiffu는 자동 및 인간 평가에서 기존 모델들보다 향상된 유의미성(relevance), 제어 가능성(controllability), 정보성(informativeness)을 보이며, 최신 기술 수준에 도달한 성능을 보여줍니다.



### Cognitive Kernel: An Open-source Agent System towards Generalist Autopilots (https://arxiv.org/abs/2409.10277)
- **What's New**: Cognitive Kernel은 일반적인任务을 독립적으로 수행할 수 있는 오픈 소스 에이전트 시스템이다. 이 시스템은 사용자로부터의 입력이 필요 없는 자율적인 자동 조종 장치(autopilot) 시스템 개발을 목표로 하고 있다.

- **Technical Details**: Cognitive Kernel은 정책 모델(policy model) 기반의 설계를採用하고 있으며, 이는 LLM (대형 언어 모델)의 세부 조정된 버전이다. 이 시스템은 인지, 행동 결정 및 상태 저장 기능을 담당하는 세 가지 주요 구성 요소인 reasoning kernel, perception kernel, memory kernel로 이루어져 있다. Atomic actions를 사용하여 환경과 상호작용하며, Python을 사용하여 복합 작업을 수행한다.

- **Performance Highlights**: Cognitive Kernel은 실시간 정보 관리, 개인 정보 관리, 장기 메모리 관리의 세 가지 주요 역량에서 기존의 폐쇄형 시스템에 비해 우수한 성능을 보여 주었다. 모델과 시스템 설계를 깊이 통합함으로써 최적의 성능을 달성하는 것으로 나타났다.



### Automatic Control With Human-Like Reasoning: Exploring Language Model Embodied Air Traffic Agents (https://arxiv.org/abs/2409.09717)
- **What's New**: 최근 언어 모델의 발전이 항공 교통 관제 분야에서 새로운 기회를 창출하고 있습니다. 본 논문은 언어 모델 기반 에이전트를 사용하여 인적 개입 없이 항공 교통 갈등을 해결하는 방법을 구체적으로 탐구합니다.

- **Technical Details**: 본 연구는 대규모 언어 모델, 에이전트가 시뮬레이터와 상호작용할 수 있는 도구 및 경험 라이브러리라는 새로운 개념을 구성 요소로 포함하고 있습니다. 경험 라이브러리는 벡터 데이터베이스로, 에이전트가 시뮬레이션 및 언어 모델과의 상호작용을 통해 학습한 지식을 저장합니다.

- **Performance Highlights**: 최고 성능을 보이는 구성은 총 120개의 긴급 갈등 시나리오 중 단 하나를 제외하고 모두 해결할 수 있었습니다. 에이전트는 교통 상황 및 갈등 해결 전략에 대해 인간 수준의 설명을 제공할 수 있습니다.



### Towards Data-Centric RLHF: Simple Metrics for Preference Dataset Comparison (https://arxiv.org/abs/2409.09603)
Comments:
          Working Paper

- **What's New**: 언어 모델의 인간 선호를 맞추기 위한 목표는 이러한 선호를 드러내는 데이터의 수집을 요구합니다. 현재까지 데이터셋 품질을 측정하고 비교하는 기존 노력이 없던 가운데, 본 연구는 선호 데이터셋을 체계적으로 연구하고 다양한 비교축을 제시하여 데이터 중심의 정렬(data-centric alignment) 접근 방식을 제안합니다.

- **Technical Details**: 본 연구에서는 선호 데이터셋을 평가하기 위해 세 가지 관점을 제안합니다: 1) effective sample size, 2) noise invariance, 3) information content. 이러한 측정 방법을 통해 RLHF에서의 보상 모델 성능을 효율적으로 훈련하기 위한 데이터 수집을 지원합니다.

- **Performance Highlights**: 세 가지 직관적 관점을 통해 선호 데이터셋을 이해하는 데 기여함으로써, 새로운 데이터셋의 개발과 관련된 광범위한 응용에 기여합니다. 다양한 모델 크기에서의 ablation(소거법)을 통한 검증을 통해 이러한 측정과 보상 모델 성능 간의 연결 고리를 입증합니다.



### Autonomous Goal Detection and Cessation in Reinforcement Learning: A Case Study on Source Term Estimation (https://arxiv.org/abs/2409.09541)
- **What's New**: 본 논문에서는 AGDC(Autonomous Goal Detection and Cessation) 모듈을 소개하여 강화학습(RL) 알고리즘의 목표 자율 감지 및 작업 완료 시 시스템의 효율성을 개선합니다.

- **Technical Details**: AGDC 모듈은 Bayesian 추정 기법을 활용하여 환경 변화를 추정하고, 작업 진행 상황을 평가합니다. 지정된 임계값에 도달하면 자동으로 목표 종료 신호를 제공합니다.

- **Performance Highlights**: AGDC를 통합한 RL 알고리즘은 성공률, 이동 거리, 탐색 시간 측면에서 전통적인 통계적 방법을 크게 초해 복잡한 현실 시나리오에서 특히 뛰어난 성과를 보였습니다.



### Enumerating Minimal Unsatisfiable Cores of LTLf formulas (https://arxiv.org/abs/2409.09485)
- **What's New**: 이 논문에서는 $	ext{LTL}_f$ (Linear Temporal Logic over finite traces) 사양의 최소 불만족 코어(minimal unsatisfiable cores, MUC)를 열거하는 새로운 기술을 소개합니다. 기존의 불만족 코어를 찾는 연구를 확장하여, $	ext{LTL}_f$ 포뮬라를 Answer Set Programming (ASP) 사양으로 인코딩하여 MUC를 직접적으로 얻을 수 있도록 합니다.

- **Technical Details**: MUC 열거기를 설계하기 위해 $	ext{LTL}_f$ 포뮬라 집합을 주고, 이를 통해 생성된 ASP 프로그램의 최소 불만족 부분 집합(MUS)과 원래 $	ext{LTL}_f$ 사양의 MUC 간의 일대일 대응을 활용합니다. 이 방법은 기존의 $	ext{LTL}_f$ 제한적 만족 가능성 검증을 위한 인코딩을 바탕으로 하여, ASP 시스템의 효율성을 높임으로써 불만족성을 탐색하는 반복적인 검증을 진행합니다.

- **Performance Highlights**: 실험 결과, 제안된 MUC 열거기는 기존 시스템과 비교하여 우수한 성능을 보여주었으며, 구체적으로 정의된 벤치마크에서 기존에 단 하나의 MUC만을 산출하던 시스템과 경쟁력을 갖췄습니다. 이는 MUC 열거가 매우 효율적이라는 것을 의미합니다.



### Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models (https://arxiv.org/abs/2409.09345)
- **What's New**: 이 논문은 결정적인 단계가 필요한 작업 처리에서 LLM 에이전트의 성능 향상 방법을 제안합니다. 특히, Q-value 모델을 활용하여 행동 선택을 안내하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 본 연구에서는 Monte Carlo Tree Search (MCTS)를 통해 수집한 결정적 경로와 단계별 Q 값으로 주석이 달린 데이터를 기반으로, 또 다른 LLM을 이용해 해당 선호도를 학습합니다. 이 과정에서 단계별 Direct Policy Optimization (DPO)을 사용하여 Q-value 모델을 형성하였으며, 각 결정 단계에서 LLM 에이전트가 가장 높은 Q 값을 선택하도록 설계하였습니다.

- **Performance Highlights**: Q-value 모델을 적용한 결과, Phi-3-mini-4k-instruct로 구축된 에이전트는 WebShop에서 103%, HotPotQA에서 75% 향상된 성능을 보였으며, 이는 GPT-4o-mini를 초월하는 성능입니다. 또한 Q-value 모델은 다양한 LLM 에이전트에 대한 일반화 가능성과 기존 프롬프트 전략과의 원활한 통합 등의 장점을 제공합니다.



### Multimodal Fusion with LLMs for Engagement Prediction in Natural Conversation (https://arxiv.org/abs/2409.09135)
Comments:
          22 pages, first three authors equal contribution

- **What's New**: 이번 연구에서는 '스마트 안경'을 활용하여 쌍방향 대화에서 참여도를 예측하는 새로운 접근법을 제시합니다. 이 장치에는 비디오 카메라, 눈 카메라, 마이크로폰 등의 센서가 장착되어 있어 비언어적 행동을 자연적인 환경에서 분석할 수 있는 기회를 제공합니다.

- **Technical Details**: 34명의 참가자가 포함된 데이터셋을 수집하여 각 대화 후 자기 보고식 참여도 평가를 진행했습니다. 여러 행동 모달리티를 통합한 새로운 'multimodal transcript' 프레임워크를 통해 데이터 처리에 효과적이며, 최초 구현에서도 기존 융합 기술과 유사한 성능을 보여주었습니다. 이 프레임워크는 대화 참여도의 정황 추론을 가능하게 합니다.

- **Performance Highlights**: 제안한 방법론은 기존의 융합 기법들과 동등한 성능을 달성하는 것으로 나타났으며, 이는 향후 인간 행동 모델링 및 사회적으로 지능화된 기술 개발에 큰 잠재력을 지닙니다.



### Proactive and Reactive Constraint Programming for Stochastic Project Scheduling with Maximal Time-Lags (https://arxiv.org/abs/2409.09107)
- **What's New**: 본 연구에서는 최대 시간 지연을 가진 확률 자원 제한 프로젝트 일정 문제(SRCPSP/max)에 대한 새로운 일정 전략을 제안합니다. 새로운 제약 프로그래밍(CP) 기반의 완전한 사전 대응 방법과 온라인 재일정을 활용한 반응적 접근 방식, 그리고 불확실성을 기반으로 한 단순 시간 네트워크(STNUs)를 이용한 방법을 포함합니다.

- **Technical Details**: 이 논문은 두 가지 주요 접근 방식인 사전 대응 일정(proactive scheduling)과 반응적 일정(reactive scheduling)에 대해 논의합니다. 특히, CP를 이용한 새로운 사전 대응 및 하이브리드 접근 방식을 소개하고, STNU 기반의 알고리즘이 솔루션 품질에서 우수하다는 통계적 분석 결과를 제시합니다. 또한 재일정 절차를 통해 반응적 접근 방식을 지원하는 방법들이 다루어집니다.

- **Performance Highlights**: STNU 기반의 알고리즘은 솔루션의 품질과 오프라인 및 온라인 계산 시간 모두에서 뛰어난 성능을 보였습니다. 다양한 방법에 대한 성공적인 비교를 통해 연구자가 실제 일정 문제에 적합한 방법을 선택할 수 있도록 하는 실용적인 가이드를 제공합니다.



### Shadowed AHP for multi-criteria supplier selection (https://arxiv.org/abs/2409.09082)
Comments:
          14 pages

- **What's New**: 이번 논문은 비즈니스 도메인에서 사용되는 다기준 의사결정 기법(MCDM) 중 하나인 Analytical Hierarchical Process (AHP)에 대해 새로운 접근 방식을 제안합니다. 특히, 다양한 불확실한 숫자들을 사용하여 AHP의 선호 값을 표현하는 문제를 해결하기 위한 새로운 방법을 소개합니다.

- **Technical Details**: 이번 연구에서는 Shadowed Fuzzy Numbers (SFNs)를 사용하여 다중 유형의 불확실한 숫자로 표현된 선호 값을 처리하는 Shadowed AHP 방법을 제안합니다. SFNs는 다양한 유형의 퍼지 숫자를 근사화하며 그들의 불확실성 특성을 보존합니다. 새로운 접근 방식은 다중 그라뉴라 선호 값을 통합된 모델인 SFNs로 변환하고 이러한 특성을 활용합니다.

- **Performance Highlights**: 이 새로운 접근 방식은 다수의 그라뉴라 정보를 사용하는 공급자 선택 문제에 적용되어 의사 결정 애플리케이션에서 중요성을 가집니다. 또한, 집합 선호의 결과를 정렬하는 새로운 순위 접근 방식을 도입하여, MCDM 문제 해결에 기여합니다.



### An Efficient Self-Learning Framework For Interactive Spoken Dialog Systems (https://arxiv.org/abs/2409.10515)
Comments:
          Presented at ICML 2024

- **What's New**: 본 연구는 대화 시스템에서 사용할 수 있는 새로운 전반적인 프레임워크를 제안하여, 단일 발화(turn)에서 학습하는 것을 넘어, 다회 대화(contextual dialog)에서 사용자 피드백을 반영하여 적응할 수 있는 자동 음성 인식(ASR) 시스템을 개발합니다.

- **Technical Details**: 제안된 프레임워크는 학생-교사(student-teacher) 학습 기법과 컨텍스트 인식 대화 처리(context-aware dialog processing)에서의 발전을 기반으로 하며, 새로운 온라인 하드 네거티브 마이닝(online hard-negative mining) 알고리즘을 통해 대화에서의 피드백을 처리합니다. 이 과정에서 명시적 컨텍스트 신호와 암묵적 피드백 신호를 활용하여 다중 단계 교사 모델을 도입합니다.

- **Performance Highlights**: 실제 대화 시스템에서 기존 시스템에 비해 약 10%의 상대적 WER(word error rate) 개선을 보였으며, 공개된 합성 데이터에서는 최대 26%까지 개선되었습니다. 저자원(domain) 환경에서도 약 22.8%의 WERR(word error rate reduction)을 보여주는 성과를 달성했습니다.



### MusicLIME: Explainable Multimodal Music Understanding (https://arxiv.org/abs/2409.10496)
Comments:
          GitHub repository: this https URL

- **What's New**: 이 연구에서는 다중 모달 음악 이해 시스템을 위한 MusicLIME이라는 모델 불문(feature importance explanation) 방법을 도입합니다. 기존의 단일 모달 방법들과 달리 Audio(오디오)와 Lyrical(가사) 특성 간의 상호작용을 설명하여 결정 과정을 보다 포괄적으로 이해할 수 있습니다.

- **Technical Details**: MusicLIME은 다중 모달 데이터(텍스트 및 오디오)를 처리하기 위해 RoBERTa와 Audio Spectrogram Transformer(AST) 모델을 결합하여 설계되었습니다. 이 방법은 오디오를 10개의 시간 구간으로 분할하고, 각 구간을 구성하는 소스(보컬, 드럼 등)로 세분화하여 오디오와 텍스트의 특성을 동시에 분석합니다.

- **Performance Highlights**: MusicLIME은 다중 모달 음악 모델의 해석 가능성을 향상시키며, 사용자들이 모델의 행동을 더 잘 이해하고 정보를 기반으로 한 선택을 할 수 있도록 돕습니다. 이를 통해 공정하고 투명한 음악 이해 시스템을 구축하는 데 기여합니다.



### Flash STU: Fast Spectral Transform Units (https://arxiv.org/abs/2409.10489)
- **What's New**: 본 논문은 Spectral Transform Unit (STU)의 효율적인 오픈 소스 PyTorch 구현을 소개하고, 언어, 로봇 공학, 시뮬레이션 동적 시스템 등 다양한 분야의 시퀀스 예측 작업을 탐구합니다. STU와 그 변형들이 동일한 파라미터 수에서 Transformer 및 기타 최신 상태 공간 모델들보다 우수한 성능을 낸다는 결과를 도출했습니다.

- **Technical Details**: STU는 고정된 컨볼루션 필터를 사용하여 학습을 필요로 하지 않는 스펙트럼 필터링 기술에 기반한 신경망 아키텍처입니다. STU의 아키텍처에서는 입력 시퀀스를 변환하여 출력을 생성하며, 이 과정에서 고정 필터와 선택적 비선형 함수들을 사용합니다. 연구에서는 STU의 텐서 곱 근사(STU-T) 기법을 사용하여 계산 복잡도를 줄이고, 입력과 출력 차원 사이의 텐서를 두 개의 행렬로 근사합니다.

- **Performance Highlights**: 실험 결과에 따르면, STU는 긴 메모리와 비선형성이 있는 환경에서 잘 동작되었습니다. 기존의 S4 및 기본 Transformer 레이어와 비교했을 때, 평균 제곱 오차(mean squared error) 면에서 STU의 예측 정확도가 뛰어난 것으로 나타났습니다.



### Do Pre-trained Vision-Language Models Encode Object States? (https://arxiv.org/abs/2409.10488)
- **What's New**: 이 연구에서는 웹 규모 데이터로 사전 훈련된 비전-언어 모델(VLM)들이 물체 상태 인코딩의 효과를 학습하는지를 조사하였습니다. 이를 통해 'ChangeIt-Frames'라는 객체 상태 인식 데이터셋을 통해 다양한 VLM을 평가하였으며, 신뢰성 있는 객체 인식은 가능하지만 물체의 물리적 상태를 정확히 식별하는 데에는 실패하고 있음을 발견하였습니다.

- **Technical Details**: ChangeIt-Frames 데이터셋은 다양한 물체 상태를 가진 이미지를 포함하고 있으며, VLM의 객체 중심 표현을 사용하여 색상과 형태에 대한 개념 결합 작업의 개선을 테스트합니다. VLM인 CLIP과 OpenCLIP은 이미지-텍스트 대비 학습을 적용하여 성능을 평가하며, 객체 정확도(Object Accuracy)와 상태 정확도(State Accuracy)를 별도로 산출하여 성능을 측정합니다.

- **Performance Highlights**: 객체 인식 정확도는 일반적으로 높으나, 상태 인식 정확도에서 약 30%의 일관된 감소가 관찰되었습니다. 물체 중심 VLM을 사용할 경우 성능 향상이 가능하나 여전히 인식 한계가 존재하며, FLAVA는 다른 VLM보다 저항력이 높아 도전적 상황에서도 상대적으로 잘 수행되었습니다.



### Exploring 3D Face Reconstruction and Fusion Methods for Face Verification: A Case-Study in Video Surveillanc (https://arxiv.org/abs/2409.10481)
Comments:
          Accepted at T-CAP - Towards a Complete Analysis of People: Fine-grained Understanding for Real-World Applications, workshop in conjunction with the 18th European Conference on Computer Vision ECCV 2024

- **What's New**: 이 연구에서는 서로 다른 3D 얼굴 재구성(3DFR) 알고리즘(EOS, 3DDFA v2, NextFace)을 활용하여 얼굴 인식 성능을 개선하고자 한다. 각 알고리즘이 개별적으로 분류기를 훈련시키는데 도움을 주고, 최종 결정은 점수 수준 융합(score-level fusion)을 통해 이루어진다.

- **Technical Details**: 3DFR 알고리즘의 성능은 카메라의 거리 및 인식하는 얼굴의 각도와 같은 다양한 환경 요인에 따라 달라질 수 있다. 본 연구에서는 여러 3DFR 알고리즘을 조합하여 비디오 감시 시나리오에서의 얼굴 인식 성능을 향상시키고, 다양한 점수 수준 융합 방법의 효과를 분석한다.

- **Performance Highlights**: 본 연구의 결과, 서로 다른 3DFR 알고리즘으로부터 얻은 보완 정보는 한 번도 본 적 없는 거리 및 카메라 특성에서 성능을 크게 향상시켰다는 것을 보여주었다. 이는 3DFR 기반 접근법에 대한 추가 연구를 촉진할 것으로 기대된다.



### MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion (https://arxiv.org/abs/2409.10473)
Comments:
          Accepted by ECCV 2024

- **What's New**: 본 연구에서는 Masked Conditional Diffusion (MacDiff)라는 새로운 프레임워크를 제안하여 인간의 스켈레톤 모델링을 수행합니다. 이는 첫 번째로 확산 모델(difussion models)을 효과적인 스켈레톤 표현 학습기(regresentation learners)로 활용하는 접근법을 제시합니다.

- **Technical Details**: MacDiff는 의미론적 인코더(semantic encoder)와 확산 디코더(diffusion decoder)로 구성되어 있으며, 인코더 입력에 랜덤 마스킹(random masking)을 적용하여 정보 병목(information bottleneck)을 도입하고 스켈레톤의 중복성을 제거합니다. 이 방법은 추상 대표성이 높아지고 상향식 과제에 대한 일반화 성능을 향상시키는 효과를 입증합니다.

- **Performance Highlights**: MacDiff는 NTU RGB+D 및 PKUMMD 데이터셋에서 최첨단 성능을 달성하며, 특히 제한된 라벨 데이터가 있는 상황에서 데이터 증강(data augmentation)을 통해 미세 조정(fine-tuning) 성능을 크게 향상시킵니다.



### HiFi-CS: Towards Open Vocabulary Visual Grounding For Robotic Grasping Using Vision-Language Models (https://arxiv.org/abs/2409.10419)
- **What's New**: 이 연구에서는 HiFi-CS라는 새로운 Referring Grasp Synthesis (RGS) 모델을 소개합니다. 이 모델은 Featurewise Linear Modulation (FiLM)을 활용하여 이미지와 텍스트 임베딩을 융합하고, 복잡한 속성을 가진 텍스트 쿼리의 시각적 그라운딩을 개선합니다.

- **Technical Details**: HiFi-CS는 두 단계로 구성된 RGS 프로세스를 채택하며, 첫 번째 단계인 Visual Grounding (VG)에서는 자연어 쿼리를 기반으로 작업 공간 내의 객체를 식별하고, 두 번째 단계인 Grasp Pose Estimation (GPE)에서는 해당 객체의 그랩 파라미터를 결정합니다. 이 연구는 경량화된 디코더를 사용해 생성 모델의 성능을 최적화하고, 2D 시각적 그라운딩에서 정밀한 예측을 가능하게 합니다.

- **Performance Highlights**: HiFi-CS는 두 개의 로봇 비주얼 그라운딩 데이터셋에서 기존 방법보다 평균 87%의 Intersection over Union 정확도를 달성하며, 15개의 실세계 복잡한 장면에서 90.33%의 시각적 그라운딩 정확도를 기록했습니다.



### Geometric Clustering for Hardware-Efficient Implementation of Chromatic Dispersion Compensation (https://arxiv.org/abs/2409.10416)
- **What's New**: 이 논문은 광섬유 통신 시스템에서 chromatic dispersion compensation (CDC) 알고리즘의 계산 복잡성을 줄이기 위한 새로운 방법을 제안합니다. 효과적인 하드웨어 실행을 위한 이론적 분석과 FPGA(Field-Programmable Gate Array)를 통한 구현을 통해 TDCE(Time-Domain Clustered Equalizer) 기술을 도입하였습니다.

- **Technical Details**: 본 연구는 CDC 필터의 tap overlapping 효과를 이론적으로 분석하였으며, 이 현상을 활용하여 복잡성을 줄이는 새로운 접근법인 TDCE를 제안합니다. TDCE의 하드웨어 구현은 최대 640km 길이의 섬유에 대해 수행되었으며, 메모리 관리와 병렬화 기술에 중점을 두었습니다. 제안된 알고리즘은 machine learning을 통해 최적화될 수 있습니다.

- **Performance Highlights**: TDCE 하드웨어 구현은 기존의 frequency domain equalizer (FDE)와 비교하여 최대 70.7%의 에너지 절약 및 71.4%의 multiplier 사용량 절약을 달성하였습니다. 계산 복잡성은 높지만, 하드웨어 구현 전략에 따라 에너지 소비와 자원 효율성에 중대한 영향을 미칠 수 있음을 보여주었습니다.



### A Knowledge-Enhanced Disease Diagnosis Method Based on Prompt Learning and BERT Integration (https://arxiv.org/abs/2409.10403)
Comments:
          Knowledge Enhancement,Disease Diagnosis,Prompt Learning,BERT,Knowledge Graph

- **What's New**: 이 논문은 prompt learning 프레임워크를 기반으로 한 지식 향상 질병 진단 방법을 제안합니다. 외부 지식 그래프에서 구조화된 지식을 검색하고 이를 인코딩하여 PROPMPT 템플릿에 주입함으로써 언어 모델의 이해력 및 추론 능력을 향상시킵니다.

- **Technical Details**: 이 방법은 CHIP-CTC, IMCS-V2-NER, KUAKE-QTR의 세 가지 공개 데이터셋에서 실험을 수행했으며, 지식 주입 모듈이 F1 점수에 미치는 중요한 역할을 강조합니다. 지식 주입은 언어 모델이 의료 관련 지식을 습득하는 데 효과적이며, 이는 진단 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 제안된 방법은 CHIP-CTC 데이터셋에서 2.4%, IMCS-V2-NER 데이터셋에서 3.1%, KUAKE-QTR 데이터셋에서 4.2%의 F1 점수 향상을 포함하여 기존 모델들보다 상당한 성능 향상을 보여주었습니다. 이 방법은 질병 진단의 정확성을 개선할 뿐만 아니라 예측의 해석 가능성을 높여 임상 진단에 더 신뢰할 수 있는 지원을 제공합니다.



### MOST: MR reconstruction Optimization for multiple downStream Tasks via continual learning (https://arxiv.org/abs/2409.10394)
- **What's New**: 본 연구에서는 여러 하위 작업(downstream task)에 최적화된 Magnetic Resonance (MR) 재구성 네트워크를 제안합니다. 특히, 연속 학습(continual learning) 기법인 MOST를 통해 다수의 하위 작업을 처리할 수 있도록 확장하였습니다.

- **Technical Details**: 이 연구는 반복 기반(replay-based) 연속 학습 기술과 이미지 유도 손실(image-guided loss)을 통합하여 재구성 네트워크가 지속적인 학습을 통해 재구성 과정에서의 오류 전파(error propagation)와 도메인 간 격차(domain gaps)를 극복하도록 하였습니다. 적응형 학습을 통한 다중 하위 작업 최적화를 실현했습니다.

- **Performance Highlights**: 기업 실험 결과, MOST는 미세 조정(finetuning) 없이도 성능이 우수했으며, 단순 미세 조정 및 기존의 연속 학습 기법보다 좋은 성능을 보였습니다. 이는 MR 재구성 네트워크가 여러 하위 작업을 위해 최적화될 수 있음을 보여줍니다.



### Robust image representations with counterfactual contrastive learning (https://arxiv.org/abs/2409.10365)
Comments:
          Code available at this https URL

- **What's New**: 이번 연구에서는 'counterfactual contrastive learning'이라는 새로운 방법론을 소개합니다. 이는 기존의 contrastive learning에서의 데이터 증강(data augmentation) 문제를 해결하기 위해 고안된 방식으로, 진정한 도메인 변화를 반영한 긍정적인 쌍을 생성합니다. 이 방법은 의료 이미징에서의 데이터 수집과 관련된 변형을 배제하면서도 의미론적 의미를 유지하는데 초점을 맞춥니다.

- **Technical Details**: 이 논문은 메디컬 이미징에서의 contrastive learning을 위한 새로운 데이터 쌍 생성 프레임워크를 제안합니다. 특히, Hierarchical Variational Autoencoder (HVAE) 모델을 사용하여 'counterfactual' 이미지 생성을 통해 실제 스캐너 간의 차이를 모사한 긍정적인 데이터 쌍을 생성합니다. 이에 따라 강력한 이미지 특성을 학습하도록 지원하여 도메인 변화에 대한 강건성을 증가시키는 것이 목표입니다.

- **Performance Highlights**: Counterfactual contrastive learning 방법론은 SimCLR과 DINO-v2와 같은 널리 사용되는 contrastive learning 프레임워크에서 5개의 데이터세트로 평가되었으며, 긍정적인 데이터 쌍 생성을 통해 도메인 변화에 대한 강건성을 높였음을 입증했습니다. 특히, 한정된 레이블을 가진 경우 및 학습 데이터에 과소 표현된 도메인에서 뛰어난 성능 향상을 보였습니다.



### Point2Graph: An End-to-end Point Cloud-based 3D Open-Vocabulary Scene Graph for Robot Navigation (https://arxiv.org/abs/2409.10350)
Comments:
          8 pages, 9 figures

- **What's New**: 이번 논문에서는 기존의 RGB-D 이미지와 카메라 포즈에 의존하던 open-vocabulary scene graph generation 알고리즘의 한계를 극복하기 위해 Point2Graph라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: Point2Graph는 point cloud 데이터를 기반으로 한 end-to-end 방법론으로, 방(room)과 객체(object) 탐지/세분화 및 open-vocabulary 분류 기능을 포함하고 있습니다. 특히 방 탐지에서는 기하학적 경계 검출(geometry-based border detection) 알고리즘과 학습 기반 지역 탐지(learning-based region detection)를 결합하여 방을 세분화하고, 'Snap-Lookup' 프레임워크를 통한 open-vocabulary 방 분류를 가능하게 합니다. 또한, 객체 탐지 레이어에서는 오직 3D point cloud 데이터만을 사용하여 3D 객체를 탐지하고 분류하는 end-to-end 파이프라인을 구현합니다.

- **Performance Highlights**: 실험 결과, Point2Graph는 널리 사용되는 실제 장면 데이터셋에서 현재 최첨단(SOTA) open-vocabulary 객체 및 방 세분화와 분류 알고리즘을 초월하는 성능을 보임을 확인했습니다.



### Large Language Model Enhanced Hard Sample Identification for Denoising Recommendation (https://arxiv.org/abs/2409.10343)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)을 활용하여 추천 시스템의 하드 샘플을 구별하고 노이즈 샘플을 정화하는 새로운 프레임워크인 LLMHD를 제안합니다. 기존 방법들이 하드 샘플과 노이즈 샘플의 구별에 어려움을 겪고 있는 문제를 해결하고자 합니다.

- **Technical Details**: LLMHD 프레임워크는 세 가지 주요 모듈로 구성됩니다: Variance-based Sample Pruning(분산 기반 샘플 가지치기), LLM-based Sample Scoring(LLM 기반 샘플 점수화), Iterative Preference Updating(반복적 선호 업데이트). 처음에는 분산 기반 가지치기를 통해 하드 샘플 후보를 선택한 뒤, LLM을 통해 샘플의 하드니스(hardness)를 평가하여 최종적으로 하드 샘플을 식별합니다. 또한, 사용자 선호도를 개선하기 위한 반복적 업데이트 모듈을 포함하여, 잘못된 레이블을 수정합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋과 네 가지 백본 추천 시스템에서 폭넓은 실험을 진행한 결과, LLMHD는 뛰어난 성능과 강력한 노이즈 내성을 보였습니다. 실험 결과, LLMHD는 추천의 질을 크게 향상시키며, 노이즈로 인한 영향을 효과적으로 감소시킬 수 있음을 입증하였습니다.



### Hyperedge Modeling in Hypergraph Neural Networks by using Densest Overlapping Subgraphs (https://arxiv.org/abs/2409.10340)
- **What's New**: 본 논문에서는 기존의 Graph Neural Networks (GNNs)의 한계를 극복하기 위해 Hypergraph Neural Networks (HGNNs)에서 densest overlapping subgraphs를 활용하는 새로운 접근 방식을 제안합니다. 이를 통해 더 복잡한 구조 정보를 활용하여 노드 분류 작업에서 높은 성능을 발휘합니다.

- **Technical Details**: Densest Overlapping Subgraphs를 찾기 위해 소개된 새로운 알고리즘인 Agglomerative Greedy Enumeration (DOSAGE)을 사용하여, 제약 조건을 가진 top-K densest overlapping subgraphs 문제를 해결합니다. 이 방법은 서브그래프의 밀도와 거리 정보를 동시에 고려하여 전체 그래프의 완전한 커버리지를 보장합니다.

- **Performance Highlights**: 표준 벤치마크에서 수행된 실험 결과, DOSAGE 알고리즘은 HGNNs와 다른 여섯 가지 방법에 비해 노드 분류 작업에서 현저한 성능 향상을 보여주었습니다.



### The 20 questions game to distinguish large language models (https://arxiv.org/abs/2409.10338)
- **What's New**: 이 연구는 블랙박스(black-box) 환경에서 두 개의 대형 언어 모델(LLMs)이 동일한 모델인지 여부를 판단하는 새로운 방법을 제시합니다. 주어진 목표는 일반적으로 20개의 질문 이하로 구성된 소규모(benign) 이진 질문 세트를 사용하는 것입니다.

- **Technical Details**: 문제를 공식화하고, 알려진 벤치마크 데이터셋에서 임의의 질문 선택을 사용하여 20개 질문 내에서 거의 100% 정확도로 기준선을 설정합니다. 이후 이 문제에 대한 최적 경계를 보여준 후, 동일한 작업을 수행하기 위해 질문 수를 절반으로 줄일 수 있는 두 가지 효과적인 질문 휴리스틱(heuristics)을 소개합니다.

- **Performance Highlights**: 이 방법은 모델 샘플 유출에 대한 의구심을 가진 감사인(auditors)이나 저작권 소유자에게 중요한 장점을 제공합니다.



### InfoDisent: Explainability of Image Classification Models by Information Disentanglemen (https://arxiv.org/abs/2409.10329)
- **What's New**: InfoDisent이라는 하이브리드 모델을 소개하며, 이미지 분류 네트워크의 결정 과정을 더 잘 이해할 수 있도록 돕는다.

- **Technical Details**: InfoDisent은 미리 훈련된 딥 네트워크의 최종 레이어에서 정보를 분리(disentangle)하여, 분류 결정 과정을 기본적이고 이해하기 쉬운 원자적 구성 요소로 나눈다. 이 모델은 post-hoc 방법과 intrinsic 방법의 장점을 결합한다.

- **Performance Highlights**: InfoDisent는 ImageNet, CUB-200-2011, Stanford Cars, Stanford Dogs와 같은 벤치마크 데이터셋에서 성능을 검증하였다.



### SEAL: Towards Safe Autonomous Driving via Skill-Enabled Adversary Learning for Closed-Loop Scenario Generation (https://arxiv.org/abs/2409.10320)
Comments:
          8 pages, 4 figures, 2 tables

- **What's New**: 이 논문에서는 자율 주행(AD) 시스템의 안전성을 보장하기 위한 접합적이고 인간과 유사한 기술을 사용하는 새로운 시나리오 생성 방법인 SEAL(Skill-Enabled Adversary Learning)을 제안합니다. 기존의 간단한 목표에 기반한 접근 방식의 한계를 극복하기 위해 더 다양하고 현실적인 적대적 시나리오 생성을 목표로 합니다.

- **Technical Details**: SEAL 방법은 두 가지 혁신적인 구성 요소를 포함하고 있습니다. 첫째, 학습된 스코어링 함수를 도입하여 반응적인 에고(ego) 에이전트가 적대적 행동에 어떻게 반응할지 예측합니다. 둘째, 인간의 인지를 모방한 계층적 프레임워크를 통해 인간과 비슷한 기술 원리를 선택하여 적대적 행동의 현실성을 높이고 안전성을 유지합니다.

- **Performance Highlights**: SEAL로 수정된 시나리오는 SOTA(SOTA) 기준보다 더 현실적이며, 다양한 실제 안전 관련 시나리오에서 에고 작업 성공률을 20% 이상 향상시킵니다. 이 결과는 우리의 방법론이 기존 방법보다 현저히 개선된 성능을 굽혔음을 보여줍니다.



### Know your limits! Optimize the robot's behavior through self-awareness (https://arxiv.org/abs/2409.10308)
Comments:
          Accepted to Humanoids 2024 and HFR 2024. Project Page: this https URL

- **What's New**: 우리는 로봇이 제공된 인간의 움직임을 모방할 때 로봇의 성능을 예상하고, 최적의 참조 동작을 생성하여 목표 명령을 효과적으로 수행할 수 있도록 하는 Self-Aware 모델(SAW)을 도입했습니다.

- **Technical Details**: SAW는 다양한 기준에 따라 로봇의 행동을 평가하는 시스템으로, 예를 들어우리는 이 시스템을 통해 낙상의 가능성(fall likelihood), 참조 동작에 대한 일관성(adherence to the reference motion), 그리고 움직임의 매끄러움(smoothness)을 평가합니다. 이를 통해 SAW는 로봇이 따라야 할 최적의 참조를 선택하도록 합니다.

- **Performance Highlights**: SAW는 로봇이 주어진 참조 동작을 모방할 때 99.29%의 정확도로 낙상을 예측할 수 있으며, 최적의 로봇 행동을 보장하기 위해 심화된 움직임 생성과 로봇 제어 기술을 통합하고 있습니다.



### How to do impactful research in artificial intelligence for chemistry and materials scienc (https://arxiv.org/abs/2409.10304)
- **What's New**: 이 논문은 화학 분야에서의 머신러닝(Machine Learning, ML) 적용 현황과 과제를 총체적으로 제시합니다. 화학 분야에서 머신러닝이 폭넓게 사용되고 있지만, 그 잠재력을 충분히 발휘하고 있지 않다는 점을 강조합니다.

- **Technical Details**: 이 노트에서는 ML이 예측(prediction), 생성(generation), 합성(synthesis), 힘의 장(force fields), 스펙트로스코피(spectroscopy), 반응 최적화(reaction optimization) 및 기초 모델(foundational models)과 같은 다양한 화학 문제에 적용되는 방식을 탐구합니다. ML의 문제 해결 능력을 증대시키기 위한 협업 및 관점 교류의 중요성도 논의됩니다.

- **Performance Highlights**: ML 모델들은 복잡한 데이터 내의 패턴을 학습하고, 예측 정확도를 향상시키기 위해 진화해왔습니다. 특히 구조-특성 모델이 널리 사용되어 실험적으로 검증된 예측을 제공했습니다. 이 과정에서 화학 공간(chemical space) 탐색과 같은 새로운 접근 방법도 논의됩니다.



### On Synthetic Texture Datasets: Challenges, Creation, and Curation (https://arxiv.org/abs/2409.10297)
- **What's New**: 이 논문은 고품질의 다양한 텍스처 이미지를 생성하기 위한 새로운 방법론을 소개하며, 362,880개의 텍스처 이미지를 포함하는 Prompted Textures Dataset (PTD)을 개발했습니다. 이 데이터셋은 56개의 텍스처 클래스를 포함하고 있으며, 기존의 텍스처 데이터셋들이 가진 한계를 극복합니다.

- **Technical Details**: 제시된 방법론에는 (1) 텍스트-투-이미지 모델에 입력할 설명자로부터 프롬프트를 개발하고, (2) Stable Diffusion 파이프라인을 채택 및 조정하여 해당 이미지를 생성 및 필터링하며, (3) CLIP 점수를 통해 최고 품질의 이미지로 추가 필터링하는 단계가 포함됩니다.

- **Performance Highlights**: 생성된 데이터셋은 Inception 및 FID 점수를 통해 품질과 다양성을 평가하였으며, 독립적인 인간 평가에서도 높은 품질을 보였습니다. 그러나 이미지 생성 과정에서 NSFW 안전 필터가 텍스처에 매우 민감하여 약 60%의 텍스처 이미지가 플래그를 받는 문제를 발견하였습니다.



### MGSA: Multi-granularity Graph Structure Attention for Knowledge Graph-to-Text Generation (https://arxiv.org/abs/2409.10294)
- **What's New**: 이번 논문은 Multi-granularity Graph Structure Attention (MGSA) 모델을 소개하여 Knowledge Graph에서 Text로의 변환 작업에서 발생하는 한계를 극복하고자 하였습니다. MGSA 모델은 엔티티 수준과 단어 수준의 구조 정보를 동시에 통합하여 보다 질 높은 자연어 텍스트 생성을 목표로 합니다.

- **Technical Details**: MGSA 모델은 세 가지 모듈로 구성됩니다: (1) 엔티티 수준 구조 인코딩 모듈, (2) 단어 수준 구조 인코딩 모듈, (3) 집계 모듈. 이 모듈들은 서로 다른 수준의 구조 정보를 결합하여 KG의 구조를 보다 포괄적으로 이해하는 데 도움을 줍니다.

- **Performance Highlights**: WebNLG와 EventNarrative 데이터셋을 사용하여 MGSA 모델을 평가한 결과, 단일 수준의 구조 정보에만 의존하는 모델들에 비해 일관되게 뛰어난 성능을 보였습니다. 이로 인해 우리의 접근법의 효과성을 입증하였습니다.



### Neuromorphic Spintronics (https://arxiv.org/abs/2409.10290)
Comments:
          Neuromorphic Spintronics is a chapter of a book titled "Artificial Intelligence and Intelligent Matter". This is not the final version of the chapter. For the final version, please go to the book published by Springer (the DOI and other details will be put here once the book has been published.)

- **What's New**: 이 논문은 신경형 스핀트로닉스(neuromorphic spintronics)라는 혁신적인 분야를 소개하며, 뇌의 구조와 기능을 모방하여 전통적인 전자공학의 한계를 극복할 수 있는 가능성을 제시합니다.

- **Technical Details**: 신경형 스핀트로닉스는 전자의 스핀(spin)을 활용하여 정보 처리 및 저장을 개선하는 방법론입니다. 이는 전통적인 전자공학에서 사용하는 전하(charge)에 의존하지 않으며, 메모리 및 적응성을 통해 에너지 소모를 줄일 수 있습니다. 스핀 기반 시스템은 대칭적 및 비대칭적 자기 배열을 통해 비휘발성 저장소(non-volatile storage)를 제공합니다.

- **Performance Highlights**: 이 기술은 빠른 처리 속도와 높은 에너지 효율성을 제공하며, 특히 스핀전이 토크(spin-transfer torque) 및 스핀-오르빗 토크(spin-orbit torque)와 같은 다양한 물리적 효과를 활용하여 차세대 스토리지 및 컴퓨팅 장치를 개발할 수 있는 기회를 제공합니다.



### DreamHead: Learning Spatial-Temporal Correspondence via Hierarchical Diffusion for Audio-driven Talking Head Synthesis (https://arxiv.org/abs/2409.10281)
- **What's New**: 새로운 연구에서 DreamHead라는 계층적 확산(framework hierarchy diffusion) 모델을 제안하여 오디오 기반의 토킹 헤드(talking head) 합성을 위한 공간-시간(spatial-temporal) 일치를 효과적으로 학습합니다. 이 모델은 음성과 관련된 얼굴 동작을 보다 정밀하게 예측할 수 있도록 설계되었습니다.

- **Technical Details**: DreamHead는 두 단계의 확산 구조를 가지고 있습니다: (1) 오디오-랜드마크 확산(audio-to-landmark diffusion) 단계는 주어진 오디오 신호에 대한 얼굴 랜드마크의 연속을 예측합니다. (2) 랜드마크-이미지 확산(landmark-to-image diffusion) 단계는 예측된 랜드마크를 기반으로 실제와 유사한 얼굴 비디오 포트레이트를 생성합니다. 두 단계 모두가 협력하여 공간적 및 시간적 일치를 효과적으로 모델링할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, DreamHead는 HDTF와 MEAD 데이터셋에서 매우 사실적이며 공간-시간적으로 일관된 얼굴 비디오를 생성하는 데 성공하였습니다. 기존의 최신 기술들과 비교했을 때, 주관적 및 객관적 측면에서 우수한 생성 결과를 보였습니다.



### Causal Discovery in Recommender Systems: Example and Discussion (https://arxiv.org/abs/2409.10271)
Comments:
          Accepted at the CONSEQUENCES '24 workshop, co-located with ACM RecSys '24

- **What's New**: 인공지능 및 머신러닝 커뮤니티에서 인과성(causality)에 대한 관심이 증가하고 있으며, 이 논문은 인과 그래프(causal graphs)를 이용한 추천 시스템(recommender system) 문제 모델링의 사례를 제시합니다.

- **Technical Details**: 이 논문에서는 KuaiRand라는 오픈 소스 데이터셋을 활용하여 관찰 데이터와 사전 지식을 결합하여 인과 그래프를 학습하는 과정을 설명합니다. 인과 추론(causal discovery) 과정은 총 다섯 단계로 구성되어 있으며, 관련 없는 특성의 제거, 특성 이산화(discretization), 사전 지식의 포함 방식 정의 등을 포함합니다.

- **Performance Highlights**: 학습된 인과 그래프는 분석된 피드백 신호에 효과적으로 영향을 미치는 변수가 적다는 것을 보여줍니다. 기존 머신러닝 커뮤니티의 대규모 모델 포함 경향과 대조적으로, 이 연구는 대부분의 변수는 결정에Noise를 추가하고 무관하다는 결과를 제시합니다.



### Enhancing Personalized Recipe Recommendation Through Multi-Class Classification (https://arxiv.org/abs/2409.10267)
- **What's New**: 이 논문은 다양한 요리 선호도에 맞춘 개인화된 레시피 추천의 어려움을 해결하고자 합니다. 이전의 추천 시스템에서 벗어나, 사용자의 다양한 요소를 감안한 혁신적인 접근법을 제시하고 있습니다.

- **Technical Details**: 레시피 추천에 있어서, 본 논문은 연관 분석(association analysis)과 분류(classification) 기법을 사용합니다. 연관 분석은 다양한 재료 간의 관계와 연결을 탐구하여 사용자 경험을 개선하고, 분류는 사용자가 정의한 재료와 선호도에 따라 레시피를 범주화합니다. 독특하게도, 여러 클래스를 포함하는 레시피와 재료를 고려하여 요리 조합의 복잡성을 인식합니다.

- **Performance Highlights**: 이 시스템은 요리 레시피의 분류 및 추천에 있어 정교한 접근 방식을 요구하며, 개인화된 추천을 정확하게 제공하기 위한 과정도 탐구합니다.



### FGR-Net:Interpretable fundus imagegradeability classification based on deepreconstruction learning (https://arxiv.org/abs/2409.10246)
- **What's New**: 이 논문에서는 FGR-Net이라고 불리는 새로운 프레임워크를 제안하여 망막 이미지 품질을 자동으로 평가하고 해석하는 방법을 소개합니다. 이 시스템은 오토인코더(autoencoder) 네트워크와 분류기(classifier) 네트워크를 결합하여 작동합니다.

- **Technical Details**: FGR-Net 모델은 입력 이미지를 재구성(reconstruct)하기 위해 딥 오토인코더(deep autoencoder)를 사용하며, 자가 지도 학습(self-supervised learning)을 기반으로 입력 망막 이미지의 시각적 특성을 추출합니다. 추출된 특성은 딥 분류기 네트워크로 전달되어 gradable과 ungradable 망막 이미지를 구분합니다. 모델은 다양한 해석 가능성(interpretability) 방법에 대해 평가되었습니다.

- **Performance Highlights**: FGR-Net의 실험 결과는 기존 최첨단 품질 평가 방법들을 초과하는 성능을 보였으며, 정확도(accuracy) 89%와 F1-score 87%를 기록했습니다.



### Hedging Is Not All You Need: A Simple Baseline for Online Learning Under Haphazard Inputs (https://arxiv.org/abs/2409.10242)
- **What's New**: 이 논문은 그동안 다루지 않았던 Edge 장치에서의 비정형 스트리밍 데이터 문제를 해결하기 위해 HapNet이라는 새로운 모델을 제안합니다.

- **Technical Details**: HapNet은 self-attention 기반으로 설계되어 기존의 hedging 방법에 비해 구현이 쉽고 일반화 및 스케일링이 더 뛰어난 장점이 있습니다. 또한, positionally uncorrelated (위치적으로 비상관) 문제를 해결하기 위한 HapNetPU 모델도 제안되었습니다.

- **Performance Highlights**: HapNet과 HapNetPU 모델은 5개의 벤치마크에서 평가되었으며, 경쟁력 있는 성능을 보여 기존의 최첨단 모델들과 비교되었습니다.



### NEUSIS: A Compositional Neuro-Symbolic Framework for Autonomous Perception, Reasoning, and Planning in Complex UAV Search Missions (https://arxiv.org/abs/2409.10196)
- **What's New**: 이 논문은 자율 UAV(무인 항공기)의 검색 임무 문제를 다루고 있으며, UAV가 제한된 시간 내에 특정 관심 개체(Entities of Interest, EOIs)를 위치 찾는 것을 목표로 합니다. 이를 위해 NEUSIS라는 새로운 조합적 신경-상징 시스템을 제안합니다.

- **Technical Details**: NEUSIS는 세 가지 주요 구성 요소로 이루어져 있습니다: 1) GRiD(Perception, Grounding, Reasoning in 3D)는 UAV 시각 센서를 사용하여 3D 공간에서 관심 개체를 인식하고 이유를 찾습니다. 2) Probabilistic World Model은 GRiD의 출력값을 개선하고, 세계 지식 기반으로 신뢰도를 업데이트합니다. 3) SNaC는 고수준 계획, 중수준 계획 및 저수준 계획을 통해 효율적인 경로 계획을 지원합니다.

- **Performance Highlights**: NEUSIS는 AirSim 및 Unreal Engine을 활용한 도시 검색 임무 실험에서 최신 비전-언어 모델 및 검색 계획 모델보다 성공률, 검색 효율성, 3D 위치 지정에서 우수한 성능을 보여, 자율 UAV 시스템의 검색 임무에 대한 해결책으로서의 가능성을 입증하였습니다.



### Relative Positioning for Aerial Robot Path Planning in GPS Denied Environmen (https://arxiv.org/abs/2409.10193)
Comments:
          12 pages, 4 images

- **What's New**: 호주에서 자율 비행 로봇, 즉 Unmanned Aerial Vehicles (UAV)가 산불 모니터링 및 예측 작업에 유용하게 사용되고 있다는 점과, 전통적인 GPS 신호에 의존하지 않고도 임무를 수행할 수 있는 방법을 제안합니다.

- **Technical Details**: 이 논문은 자율 UAV의 내비게이션에서 가장 중요한 요소 중 하나인 초기 위치 결정(Localisation)에 초점을 맞추고 있습니다. 제안된 솔루션은 자율 UAV 팀이 원거리 및 혹독한 날씨 조건에서도 기지(base of operation)와의 상대 위치를 설정할 수 있도록 하여, 산불에 영향을 받은 지역에서 팀 탐색(search and reconnaissance)을 시작하고 GPS 신호 없이도 기지로 돌아올 수 있게 합니다.

- **Performance Highlights**: 제안된 솔루션은 비상 상황에서의 UAV의 위치 결정과 작동 능력을 크게 향상시킬 수 있으며, 이는 산불 대응 전략에 중요한 기여를 할 것입니다.



### Augmenting Automatic Speech Recognition Models with Disfluency Detection (https://arxiv.org/abs/2409.10177)
Comments:
          Accepted by SLT2024

- **What's New**: 이 연구에서는 기존 오토메이션 음성 인식(ASR) 모델의 한계를 극복하기 위해 개방형(오픈셋) 발화 비유창성(dysfluency) 탐지 기능을 추가하는 인퍼런스(inference) 전용 접근법을 제시합니다.

- **Technical Details**: 발화 비유창성은 반복, 중재, 수정 등으로 나타나는 대화의 흐름 방해를 의미합니다. 제안하는 파이프라인은 ASR 모델과 특성 추출기(framework feature extractor)를 조합하여 전사(transcription) 및 정렬(alignment)을 수행하며, 수정된 연결주의 시간 분류(Connectionist Temporal Classification, CTC) 기반 강제 정렬 알고리즘을 사용하여 단어 수준의 타임스탬프를 예측합니다. 이 방법은 발화 비유창성의 위치와 지속 시간을 효과적으로 캡처합니다.

- **Performance Highlights**: 제안된 모델은 81.62%의 정확도와 80.07%의 F1 점수를 달성하였으며, 비유창한 데이터셋에서 74.13%의 단어를 초기 전사에서 놓친 것을 보완했습니다. 이 결과는 하위 작업(downstream tasks)에서 활용될 가능성을 보여줍니다.



### jina-embeddings-v3: Multilingual Embeddings With Task LoRA (https://arxiv.org/abs/2409.10173)
Comments:
          20 pages, pp11-13 references, pp14-20 appendix and experiment tables

- **What's New**: jina-embeddings-v3는 5억 7천만 개의 파라미터를 가진 최신 텍스트 임베딩 모델로, 다중 언어 데이터 및 긴 맥락 검색 작업에서 최첨단 성능을 발휘합니다. 이 모델은 최대 8192개의 토큰까지 지원하며, 쿼리-문서 검색, 클러스터링, 분류, 텍스트 매칭 등의 작업에 특화된 Low-Rank Adaptation (LoRA) 어댑터를 포함하고 있습니다.

- **Technical Details**: jina-embeddings-v3는 긴 텍스트 시퀀스를 효과적으로 인코딩하고 특정 작업에 맞춘 임베딩 생성을 가능하게 하는 구조로, XLM-RoBERTa 모델을 기반으로 합니다. FlashAttention 2와 DeepSpeed 프레임워크를 활용하여 효율적인 분산 훈련을 지원하고, Rotary Position Embeddings (RoPE)를 사용하여 절대 위치 임베딩을 대체해 긴 시퀀스를 처리합니다.

- **Performance Highlights**: MTEB 벤치마크에서 jina-embeddings-v3는 OpenAI 및 Cohere의 최신 상용 임베딩을 초월하여 영어 작업에서 뛰어난 성과를 보였고, 모든 다국어 작업에서 multilingual-e5-large-instruct을 능가했습니다. 또한 7.1억 개의 파라미터를 가진 LLM 기반 임베딩보다 비용 효율성이 높아 실제 생산 및 엣지 컴퓨팅에 더 적합합니다.



### Algorithmic Behaviors Across Regions: A Geolocation Audit of YouTube Search for COVID-19 Misinformation between the United States and South Africa (https://arxiv.org/abs/2409.10168)
Comments:
          28 pages. Under submission

- **What's New**: 이번 연구는 COVID-19 관련 잘못된 정보의 유포를 조사하기 위해 글로벌 남반부(Global South)와 북반부(Global North) 간의 YouTube 검색 결과를 비교한 최초의 대규모 지리적 기반(comparative geolocation-based) 감사(audit) 연구입니다. 연구팀은 미국과 남아프리카 공화국의 YouTube 검색에서 잘못된 정보의 유병률을 분석했습니다.

- **Technical Details**: 연구는 2023년 1월 30일부터 2월 9일까지 10일 동안 진행되었습니다. 각국의 3개의 지리적 위치에서 '실제' 사용자를 모방하는 봇(sock-puppets)을 사용하여 48개의 검색 쿼리에 대해 YouTube 검색 결과(Search Engine Result Pages, SERPs)를 수집했습니다. 수집된 데이터는 915,000건의 검색 결과로 구성되었습니다.

- **Performance Highlights**: 연구 결과, 남아프리카의 상위 10개 검색 결과 중 31.55%가 COVID-19 잘못된 정보를 포함하고 있었으며, South Africa의 봇은 미국 봇보다 더 많은 잘못된 정보에 직면했습니다(p<<<0.001). YouTube의 '관련성(Relevance)' 필터를 기준으로 한 검색 결과에서 남아프리카 사용자는 미국 사용자보다 통계적으로 유의미하게 더 많은 잘못된 정보를 발견했습니다.



### Quantile Regression for Distributional Reward Models in RLHF (https://arxiv.org/abs/2409.10164)
- **What's New**: 이 논문에서는 기존의 보상 모델이 단일 스칼라 값을 생성하는 방식의 한계를 극복하기 위한 방법으로 Quantile Reward Models (QRMs)을 제안합니다. QRM은 단일 보상 값 대신 보상 분포를 학습하며, 이를 통해 인간의 다양한 가치와 선호를 보다 정확하게 반영할 수 있습니다.

- **Technical Details**: QRM은 quantile regression을 활용하여 보상의 전체 형태에 대한 다중 모드 분포를 학습합니다. 이 방법은 레이블 노이즈와 상충되는 선호를 처리할 수 있는 유연성을 제공하며, 인간 피드백에서의 복잡성을 보다 잘 모델링합니다. 이 접근 방식을 통해 알 수 있는 것은 Llama-3 모델을 사용하여 8억 개의 매개변수를 가진 강화 학습 정책을 생성할 수 있다는 점입니다.

- **Performance Highlights**: 실험 결과, QRM은 RewardBench에서 전통적인 점 추정 모델보다 성능이 뛰어났습니다. 또한, 분포 기반의 보상 모델은 리스크에 주의하는 강화 학습에서 활용될 수 있으며, 극단적인 부정적 반응을 생성하는 것을 줄이는 정책을 개발하는 데 기여합니다.



### SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting (https://arxiv.org/abs/2409.10161)
- **What's New**: 이 논문은 Sim2Real 문제를 해결하기 위해 Gaussian Splatting 기법을 활용한 새로운 프레임워크인 SplatSim을 제안합니다. 이 프레임워크는 전통적인 메쉬 표현을 대체하여 RGB 기반 조작 정책의 Sim2Real 격차를 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: SplatSim은 기존 시뮬레이터의 물리적 백엔드를 활용하여 Gaussian Splatting을 주된 렌더링 요소로 사용합니다. 이는 단일 비디오만으로 로봇과 객체 간의 물리적 상호작용을 효과적으로 시뮬레이션할 수 있게 해줍니다. 이 프레임워크는 상태 최적화된 행동 복제(bahavior cloning) 기법을 통합하여, 시뮬레이션 데이터로 훈련된 조작 정책을 실세계에서 제로샷(zero-shot) 방식으로 전이할 수 있습니다.

- **Performance Highlights**: SplatSim을 사용한 RGB 정책의 제로샷 전이를 통해 86.25%의 평균 성공률을 달성하였으며, 이는 실세계 데이터로 훈련된 정책의 97.5%에 비해 낮은 수치입니다. 하지만, 시뮬레이션 기반 접근 방식으로도 비교적 우수한 성능을 보여주고 있습니다.



### AutoPET Challenge III: Testing the Robustness of Generalized Dice Focal Loss trained 3D Residual UNet for FDG and PSMA Lesion Segmentation from Whole-Body PET/CT Images (https://arxiv.org/abs/2409.10151)
Comments:
          11 pages, 5 figures, 2 tables

- **What's New**: 이번 연구에서 저자들은 PET/CT 스캔에서 암 병변의 자동 세분화(automated segmentation)를 위한 3D Residual UNet 모델을 사용하고 Generalized Dice Focal Loss 함수를 적용했습니다. 이를 통해 AutoPET Challenge 2024 데이터셋으로 모델을 훈련시켰습니다.

- **Technical Details**: 연구에 사용된 데이터는 900명의 환자로부터 수집된 1014개의 FDG 사례와 378명의 환자로부터 수집된 597개의 PSMA 사례로 구성되었습니다. 모델은 5겹 교차 검증(5-fold cross-validation)을 통해 훈련되었으며, 평균 앙상블 기법이 사용되었습니다. 네트워크 구조는 5개의 인코더와 디코더 레이어로 구성되며, Generalized Dice Loss와 Focal Loss를 결합한 Generalized Dice Focal Loss를 손실 함수로 활용했습니다.

- **Performance Highlights**: 초기 테스트 단계에서 평균 Dice Similarity Coefficient (DSC) 0.6687, 평균 거짓 음성 부피(FNV) 10.9522 ml, 평균 거짓 긍정 부피(FPV) 2.9684 ml를 달성했습니다.



### LLMs4OL 2024 Overview: The 1st Large Language Models for Ontology Learning Challeng (https://arxiv.org/abs/2409.10146)
Comments:
          15 pages, 1 figure, Will appear in "The 1st LLMs4OL Challenge @ ISWC 2024" proceedings

- **What's New**: LLMs4OL 2024은 대규모 언어 모델(LLMs)을 온톨로지 학습(OL)에서 활용하기 위한 첫 번째 공동 기획으로, 23회 국제 시맨틱 웹 회의(ISWC 2024)와 함께 진행됩니다. 이 챌린지는 LLMs의 온톨로지 학습 과정에서의 혁신과 기여를 도모하는 것을 목적으로 합니다.

- **Technical Details**: LLMs4OL 챌린지는 세 가지 주요 작업(Task A, B, C)으로 구성됩니다. 각 작업은 텍스트에서 구조적 지식을 추출하는 것을 목표로 하며, LLM 기반 솔루션을 요구합니다. 테스트는 Few-shot 및 Zero-shot 단계로 나뉘며, 각 참가자는 특정 작업(Task A: 용어 분류, Task B: 계층 구조 발견, Task C: 비계층적 관계 추출)에 참여할 수 있습니다.

- **Performance Highlights**: 이 챌린지를 통해 LLM의 온톨로지 학습에서의 가능성을 탐구하며, 평가 단계에서 LLM의 일반화 능력과 전이 가능성을 확인할 수 있습니다. 챌린지에서 사용된 데이터세트는 GitHub에서 확인 가능하며, 표준 평가 지표를 기반으로 모든 작업에 적용됩니다.



### Towards Explainable Automated Data Quality Enhancement without Domain Knowledg (https://arxiv.org/abs/2409.10139)
- **What's New**: 이번 연구에서는 다양한 분야에서 데이터셋의 품질을 자동 평가하고 수정할 수 있는 종합적인 프레임워크를 제안합니다. 이 프레임워크는 텍스트 및 수치 데이터를 아우르며, 결측치(absence), 중복치(redundancy), 비일관성(incoherence)이라는 세 가지 핵심 결함을 다룰 수 있습니다. 특히, 데이터 이상을 확인하고 수정하는 과정의 설명 가능성과 해석 가능성에 주목하고 있습니다.

- **Technical Details**: 제안된 프레임워크는 통계적 방법과 머신러닝(Machine Learning) 알고리즘을 통합하여 데이터 품질을 평가합니다. 데이터 품질 평가 과정에서 발생할 수 있는 시간 효율성과 정확성의 문제를 해결하기 위해, 필요할 때만 자원 집약적인 알고리즘을 사용하며 가능한 한 간단하고 효율적인 솔루션을 우선 적용합니다. 이 프레임워크는 결측치와 중복치, 오타 오류를 감지하고 수정하는 데 효과적이며, 통계적 이상치 및 논리 오류에 대한 정확성을 향상시키는 도전 과제를 남겨두고 있습니다.

- **Performance Highlights**: 실제 분석을 통해 제안된 방법이 결측치와 중복치, 오타를 감지하고 수정하는 데 효과적임이 입증되었습니다. 기존의 머신러닝 및 통계적 방법들과의 비교를 통해, 우리 프레임워크는 결과의 설명 가능성 및 해석 가능성을 확보하면서도 높은 정확성을 유지할 수 있음을 보여주었습니다.



### Advancing Towards a Marine Digital Twin Platform: Modeling the Mar Menor Coastal Lagoon Ecosystem in the South Western Mediterranean (https://arxiv.org/abs/2409.10134)
- **What's New**: 이번 연구는 스페인 무르시아 지역의 마르 메노르 해안 습지 생태계를 모델링하기 위한 Marine Digital Twin Platform의 개발을 선도합니다. 이 플랫폼은 Artificial Intelligence(AI)를 사용하여 복잡한 수문학 및 생태학 모델을 모사하고, 다양한 스트레스 요인에 대한 생태계 반응을 예측하는 시뮬레이션을 가능하게 합니다.

- **Technical Details**: 이 연구는 Open Data Sources(공개 데이터 소스)를 식별, 융합 및 통합하여 마르 메노르 분지의 생화학적 매개변수와 위성 이미지를 포함한 종합적인 디지털 표현을 구축합니다. 데이터 수집 및 융합, AI 모델 구현 및 출력 생성(back-end)뿐만 아니라 사용자 친화적인 인터페이스(front-end)를 포함한 DTO의 모든 단계에서 기술들을 개발합니다.

- **Performance Highlights**: 이 플랫폼의 모듈형 설계는 실시간 이해관계자 참여와 해양 관리에서의 정보에 기반한 의사 결정을 가능하게 하며, 이는 해양 과학과 디지털 트윈 기술의 혁신적인 발전에 기여합니다. 또한, 해파리 번식, 저산소 위기 및 수질 추정과 같은 잠재적 시나리오를 다루는 문제를 강조합니다.



### StruEdit: Structured Outputs Enable the Fast and Accurate Knowledge Editing for Large Language Models (https://arxiv.org/abs/2409.10132)
- **What's New**: 이 논문에서는 구조적 편집(Structural Editing, StruEdit)이라는 새로운 지식 편집 방법을 제안합니다. StruEdit는 자연어 출력의 비구조적인 특성을 해결하기 위해 LLM(대형 언어 모델)에서 구조화된 출력 구조를 생성하고, 잠재적으로 구식인 정보를 제거한 후 최신 정보를 효율적으로 다시 채우는 프로세스를 포함합니다.

- **Technical Details**: StruEdit는 LLM이 추론(triplet reasoning) 구조를 통해 정보를 처리하도록 유도하고, 파라미터 수정 및 문맥 입력 없이 구조화된 사실 체인을 업데이트합니다. 이 방법은 연결된 추론 단계 간의 결합 이슈를 제거하며, 멀티-홉(multi-hop) 편집 작업에서 높은 정확도와 낮은 지연(latency)을 달성합니다.

- **Performance Highlights**: 실험 결과, StruEdit는 기존의 모든 지식 편집 방법들과 비교했을 때 가장 높은 정확도와 최단 응답 시간을 기록하였고, 편집 작업의 수가 증가하더라도 안정적인 성능을 유지합니다. 특히 기존 방법들과 달리, StruEdit는 특정 지식의 위치를 찾지 않아도 되는 장점이 있어, 결과적으로 사실의 환각(hallucination)을 줄입니다.



### Industry 6.0: New Generation of Industry driven by Generative AI and Swarm of Heterogeneous Robots (https://arxiv.org/abs/2409.10106)
Comments:
          submitted to IEEE conf

- **What's New**: 본 논문은 Industry 6.0의 개념을 제시하며, 사용자 제공 자연어 설명을 기반으로 제품 디자인 및 제조 과정을 자율적으로 처리하는 세계 최초의 완전 자동화 생산 시스템을 소개합니다. Generative AI를 활용하여 제품 청사진 디자인, 부품 제조, 물류 및 조립 등 생산의 중요한 측면을 자동화합니다.

- **Technical Details**: 이 시스템은 Large Language Models (LLMs)와 통합된 다양한 AI를 갖춘 이질적인 로봇 군집을 통해 생산 과정을 조율합니다. 시스템은 조작기 팔, 배달 드론, 조립 청사진을 생성할 수 있는 3D 프린터를 포함합니다. 사용자의 자연어 명령을 기반으로 2D 서명 거리 함수(SDF)를 구축한 후 이를 3D STL 조립으로 변환합니다.

- **Performance Highlights**: 시스템은 평균 생산 시간을 119.10분으로 단축시켰으며, 전문가 팀은 평균 528.64분이 걸렸습니다(개선 비율 4.4배). 제품 청사진 단계에서 시스템은 인간 CAD 운영자를 47배 성능 초과하며, 작업을 0.5분 만에 완료했습니다. 이는 완전 자동화 제조를 향한 큰 도약을 의미합니다.



### Trustworthiness in Retrieval-Augmented Generation Systems: A Survey (https://arxiv.org/abs/2409.10102)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템의 신뢰성을 향상시키기 위한 통합 프레임워크를 제안합니다. 이는 여섯 가지 주요 차원에서 RAG 시스템의 신뢰성을 평가하며, 이러한 차원에는 사실성 (factuality), 강건성 (robustness), 공정성 (fairness), 투명성 (transparency), 책임성 (accountability), 개인 정보 보호 (privacy)가 포함됩니다.

- **Technical Details**: 이 연구에서는 RAG 시스템의 신뢰성을 평가하기 위해 문헌 조사를 통해 기존 연구를 분석하고, 신뢰성을 기준으로 다양한 LLM을 평가하는 벤치마크를 구축합니다. 이를 통해 LLM의 다양한 모델이 실제 애플리케이션에서 신뢰성을 어떻게 발휘하는지를 평가합니다.

- **Performance Highlights**: 여섯 가지 신뢰성 차원에 대한 종합적인 평가를 통해, 다양한 독점적 (proprietary) 및 오픈소스 모델의 신뢰성 성능을 비교하고 향후 RAG 시스템 개발에 대한 실질적인 통찰을 제공합니다.



### A Riemannian Approach to Ground Metric Learning for Optimal Transpor (https://arxiv.org/abs/2409.10085)
- **What's New**: 이 연구는 기계 학습과 신호 처리 응용에서 많은 주목을 받고 있는 최적 수송(Optimal Transport, OT) 이론의 새로운 발전을 제안합니다. 특히, 이 연구에서는 대칭 양의 정의 매트릭스(symmetric positive definite matrix)로 파라미터화된 적절한 잠재 기초 메트릭(ground metric)을 학습하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법론은 대칭 양의 정의 매트릭스( SPD )의 풍부한 리만 기하학(Riemannian geometry)을 활용하여 OT 거리와 기초 메트릭을 공동으로 학습합니다. 교차 최적화 방식에서 운송 계획(transport plan)과 SPD 매트릭스를 각각 독립적으로 최적화할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 접근 방식은 출처(source)와 목표(target) 데이터 세트가 다른 클래스 및 특성 분포를 가질 때 도메인 적응(domain adaptation) 설정에서 평가되었습니다. 그 결과, 일반화 성능(generalization performance) 및 강건성(robustness) 면에서 기존 기준선(baselines) 대비 우수한 성능을 나타냈습니다.



### DAE-Fuse: An Adaptive Discriminative Autoencoder for Multi-Modality Image Fusion (https://arxiv.org/abs/2409.10080)
- **What's New**: 이번 연구에서는 DAE-Fuse라는 새로운 두 단계의 차별화된 오토인코더(framework) 구조를 제안하여 높은 선명도와 자연스러운 이미지 융합(fusion)을 생성합니다.

- **Technical Details**: DAE-Fuse는 어드버셜(feature extraction) 학습을 통해 입력 이미지의 세부 정보를 보존하며 구조적 차이를 구별하는 두 개의 차별화된 블록(discriminative blocks)을 포함합니다. 이 프레임워크는 깊은 고주파 인코더(Deep High-frequency Encoder)와 깊은 저주파 인코더(Deep Low-frequency Encoder)를 병렬로 사용하여 다양한 주파수 특성을 추출합니다. 또한, Cross-Attention Fusion을 통해 두 개의 서로 다른 모달리티에서 중요한 상호작용(interaction)을 포착합니다.

- **Performance Highlights**: 공공 IR-VISIBLE 데이터셋 및 의료 영상 융합(MIF) 데이터셋에서 DAE-Fuse는 정량적인 평가와 정성적인 평가 모두에서 최첨단의 성능을 입증했습니다. 또한, 다운스트림 MMOD 작업에 있어서도 성능 향상을 기록하며, 추가적인 미세 조정(fine-tuning) 없이도 뛰어난 일반화 능력을 보여줍니다.



### LLM-DER:A Named Entity Recognition Method Based on Large Language Models for Chinese Coal Chemical Domain (https://arxiv.org/abs/2409.10077)
- **What's New**: 이 논문에서는 중국 석탄 화학 산업 도메인에서의 복잡한 구조를 가진 개체 인식 문제를 해결하기 위해 LLM-DER이라는 대형 언어 모델(LLMs)을 기반으로 한 개체 인식 프레임워크를 제안합니다.

- **Technical Details**: LLM-DER는 LLMs를 사용하여 개체 유형을 포함하는 관계 목록을 생성하여 개체 정보를 풍부하게 하고, 잘못 인식된 개체를 제거하기 위한 신뢰성 및 일관성 평가 방법을 설계합니다. 이는 특정 도메인에서 복잡한 구조의 개체 인식 문제를 효과적으로 해결합니다.

- **Performance Highlights**: 실험 결과 LLM-DER는 Resume 데이터셋과 자체 구축한 석탄 화학 데이터셋에서 뛰어난 성능을 보여주며, 기존의 GPT-3.5-turbo 모델은 물론 완전 감독 기반 모델도 초월하는 성과를 올렸습니다.



### Increasing faithfulness in human-human dialog summarization with Spoken Language Understanding tasks (https://arxiv.org/abs/2409.10070)
- **What's New**: 본 연구는 대화 요약의 신뢰성을 높이기 위해 작업 관련 정보를 통합하는 새로운 방법을 제안합니다. 특히, 고객 서비스 대화에서의 통화 의도 및 도메인 관련 엔터티를 도입하여 요약의 의미적 충실도를 높이고자 합니다.

- **Technical Details**: 연구는 SLU(Spoken Language Understanding)에서 제안된 의미적 정보를 기반으로 통화 의도, 도메인 엔터티와 같은 작업 관련 요소들을 요약 과정에 통합하여 신뢰성을 높이는 방법을 다룹니다. DECODA 코퍼스를 사용하여 지표인 NEHR(주요 엔터티 부재 비율) 및 KL 발산(Kullback-Leibler divergence)을 통한 요약 선택 기준을 제안했습니다.

- **Performance Highlights**: 실험 결과, 작업 관련 정보를 통합한 요약 모델이 정확도를 향상시키는 것으로 나타났으며, 다양한 단어 오류율에도 불구하고 일관성을 유지하는 동시에 신뢰할 수 있는 요약을 생성할 수 있음을 보여주었습니다.



### Enhancing Anomaly Detection via Generating Diversified and Hard-to-distinguish Synthetic Anomalies (https://arxiv.org/abs/2409.10069)
Comments:
          Accepted at CIKM 2024

- **What's New**: 이 논문은 도메인 비특이적 방법을 소개하여 비지도(anomaly detection) 변별자(discriminator)와 조건부 변동기(perturbator)를 사용해 분산된 방식으로 새로운 합성(anomalies) 이상치 발견을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 조건부 변동기를 통해 입력 의존적(input-dependent) 변동을 생성하고, 이를 통해 합성 이상치를 구성합니다. 변별자는 정상 샘플과 합성 이상치를 구분할 수 있도록 훈련됩니다. 두 가지 주요 전략을 통해 합성 이상치의 다양성을 보장하고, 정상 샘플과의 구분을 어렵게 합니다: (i) 변동을 서로 직각으로 만드는 것, (ii) 변동이 정상 샘플에 근접하도록 제한하는 것.

- **Performance Highlights**: 실제 데이터셋에서 실시한 실험 결과, 제안된 방법은 최신 벤치마크들에 비해 뛰어난 성능을 보였으며, 이미지 데이터뿐만 아니라 표 형식(tabular data)에서도 효과적임을 확인했습니다. 또한, 반지도 학습(semi-supervised settings) 환경에서도 전반적인 성능 향상을 보였습니다.



### MindGuard: Towards Accessible and Sitgma-free Mental Health First Aid via Edge LLM (https://arxiv.org/abs/2409.10064)
- **What's New**: 이 논문은 정신 건강 문제에 대한 접근성을 높이고 낙인을 없애며 전문적인 모바일 정신 건강 관리 시스템인 MindGuard를 제시합니다. MindGuard는 최초로 사용자 행동 데이터와 LLM 기반 대화 데이터를 통합하여 종합적인 정신 건강 관리 시스템을 구축합니다.

- **Technical Details**: MindGuard의 핵심은 사용자 행동 데이터와 LLM(대형 언어 모델) 기술을 결합하여 개인화된 대화와 개입을 제공하는 것입니다. 이 시스템은 자동 회귀적 특성을 가진 LLM의 환각(hallucination) 문제를 해결하기 위해 고품질 정신 건강 데이터셋을 구축하고 지속적인 관련 기술(PT), 감독형 미세 조정(SFT), 및 인간 검토를 포함한 후처리 방법을 사용합니다.

- **Performance Highlights**: MindGuard는 2주간의 실제 배포 결과를 통해 사용자 행동 데이터를 정확하게 분석하고 잠재적인 정신 건강 문제를 신속하게 식별할 수 있음이 입증되었습니다. 실험 결과는 MindGuard가 주요 정신 건강 관리 제공에 있어서 신뢰성과 정확성을 크게 향상시켰음을 보여줍니다.



### GlobalMapNet: An Online Framework for Vectorized Global HD Map Construction (https://arxiv.org/abs/2409.10063)
- **What's New**: 본 논문은 고충전식 HD 맵 구축을 위한 최초의 온라인 프레임워크인 GlobalMapNet을 소개합니다. 이 프레임워크는 자동차가 수집한 로컬 맵 정보를 지속적으로 업데이트하고 활용하여 일관된 인식 결과를 생성합니다.

- **Technical Details**: GlobalMapBuilder를 통해 로컬 맵을 계속해서 일치시키고 병합하여 글로벌 맵을 생성합니다. 중복 맵 요소를 제거하는 새로운 알고리즘인 Map NMS를 설계하였으며, 역사적 맵 정보를 집계하여 예측의 일관성을 개선하는 GlobalMapFusion을 제안합니다.

- **Performance Highlights**: nuScenes와 Argoverse2 데이터셋에서 GlobalMapNet의 성능을 평가하였으며, 이 프레임워크가 글로벌 일관된 결과를 생성할 수 있음을 보여주었습니다.



### Audio-Driven Reinforcement Learning for Head-Orientation in Naturalistic Environments (https://arxiv.org/abs/2409.10048)
Comments:
          submitted to ICASSP 2025

- **What's New**: 이번 연구에서는 인간-로봇 상호작용에서 음성 기반의 DRL(Deep Reinforcement Learning) 프레임워크를 제안합니다. 이 프레임워크는 스테레오 음성 녹음을 기반으로 하여 로봇이 화자의 방향으로 머리를 회전시키는 방법을 학습할 수 있도록 합니다. 이전까지 주목받지 못했던 분야에서 significant 발전을 보여줍니다.

- **Technical Details**: 연구에서 제안한 방법은 Deep Q-Learning 알고리즘을 사용하여, RNN(Recurrent Neural Network) 아키텍처를 기반으로 한 에이전트를 개발합니다. 이를 통해 에이전트는 다양한 반향(reverberation) 환경에서 화자인 녹음에 대한 방향성을 정하고, 훈련된 정책(Policy)이 다양한 자연 환경에서의 일반화 능력을 평가합니다.

- **Performance Highlights**: 에이전트는 반향이 없는 환경에서 거의 완벽하게 작업을 수행할 수 있게 학습하였으며, 자연 환경에서도 기본 랜덤 에이전트보다 우수한 성과를 보였습니다. 특히, 중간 및 높은 반향환경에서 학습한 정책은 낮은 반향환경에 대해서도 일반화되는 경향을 보였습니다. 이러한 결과는 실제 환경에 적합한 견고한 일반화 훈련 전략의 필요성을 강조합니다.



### On the Diagram of Though (https://arxiv.org/abs/2409.10038)
- **What's New**: 새롭게 소개된 DoT(도형적 사고) 프레임워크는 대형 언어 모델(LLM) 내에서 반복적인 추론을 비선형적으로 모델링하며, 기존의 직선적 사고 방식과 차별화됩니다. 이 프레임워크는 DAG(Directed Acyclic Graph)을 사용하여 여러 요소의 상호작용을 처리합니다.

- **Technical Details**: DoT 프레임워크는 세 가지 역할(제안자 <proposer>, 비평가 <critic>, 요약자 <summarizer>)을 관리하며, 자동 회귀적 다음 토큰 예측을 활용하여 각 역할 간의 원활한 전환을 가능하게 합니다. 이 과정은 제안, 비평, 수정, 검증의 순환적 구조로 이루어집니다.

- **Performance Highlights**: DoT는 추론 프로세스를 단일 모델 내에서 통합하여 복잡한 문제 해결을 촉진하고, 훈련 및 추론의 효율성을 향상시킵니다. 이 접근법은 여러 모델이나 외부 제어 메커니즘의 필요성을 제거하고, LLM의 훈련 효율성과 robust한 추론 능력을 강화합니다.



### Can GPT-O1 Kill All Bugs? (https://arxiv.org/abs/2409.10033)
- **What's New**: 이번 연구에서는 ChatGPT의 최신 버전인 O1 모델의 성능을 기존 버전 및 이전 연구와 비교하여 자동 프로그램 수리(Auto Program Repair, APR) 분야에서의 효과성을 평가했습니다. O1 모델은 새로운 강화 학습(Reinforcement Learning, RL) 및 사고의 연쇄(Chain of Thought, COT) 기법을 활용하여 기존 ChatGPT보다 향상된 수리 기능을 보입니다.

- **Technical Details**: 연구에서는 QuixBugs 벤치마크를 사용하여 O1 모델의 APR 성능을 다양한 측면(수리 성공율, 수리 소요 시간, 모델 행동 패턴)에서 평가했습니다. 두 단계로 이루어진 수리 과정에서 먼저 기본 프롬프트 템플릿을 제공하여 ChatGPT에게 버그 수정을 요청하고, 테스트 케이스에서 실패한 경우 오류 정보를 제시하여 추가 수리를 유도합니다.

- **Performance Highlights**: O1 모델은 기존 ChatGPT-4o(38/40)보다 더욱 우수한 성능을 보여주며, QuixBugs 벤치마크의 40개의 모든 버그를 성공적으로 수정했습니다. O1의 사고의 연쇄 방식은 복잡한 논리를 이해하고 올바른 수리 아이디어를 제시하는 데 효과적임을 입증했습니다.



### AttnMod: Attention-Based New Art Styles (https://arxiv.org/abs/2409.10028)
- **What's New**: 이번 논문에서는 AttnMod을 제안하여 기존의 diffusion 모델에서 새로운 예술 스타일을 생성하기 위한 주의(attention)를 수정하는 메커니즘을 다룹니다. 이를 통해 예술가는 이미지 생성 과정에서 특정 요소에 집중하거나 희석할 수 있는 방법을 제시합니다.

- **Technical Details**: 본 연구에서는 UNet의 attention layer를 열어 주의 수정을 통해 생성된 이미지의 예술적 차이를 탐구합니다. 각우선, 텍스트 프롬프트와 결합된 cross attention 블록을 사용하여 denoising diffusion 중 주의 수정을 진행합니다. 이 과정에서 attention multiplier를 도입하여 조건화에 사용된 주의의 양을 정량화합니다.

- **Performance Highlights**: AttnMod는 다양한 예술 스타일에서 실험되었으며, 각 스타일의 텍스트 프롬프트에 따라 생성되는 이미지의 예술적 변화가 관찰되었습니다. 또한, 고정된 또는 가변 속도로 주의를 변화시키는 방식이 생성 결과에 어떤 영향을 미치는지도 연구하였습니다.



### E2Map: Experience-and-Emotion Map for Self-Reflective Robot Navigation with Language Models (https://arxiv.org/abs/2409.10027)
Comments:
          19 pages, 28 figures. Project page: this https URL

- **What's New**: 이 연구는 Experience-and-Emotion Map (E2Map)을 도입하여 언어 모델의 일반 지식과 실제 환경에서의 에이전트 경험을 결합하여 계획을 수정하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: E2Map은 에이전트의 감정 반응과 시각-언어 특성을 포함하는 공간적 지도입니다. 이 지도는 가우시안 분포의 가중 합계로 모델링되며, 플래닝 및 제어 모듈은 이를 비용 함수로 사용하여 행동을 안내합니다.

- **Performance Highlights**: 제안된 방법론은 기존의 LLM 기반 접근 방식보다 확률적 환경에서의 성능을 크게 향상시키며, 동적인 객체가 존재하는 환경에서도 안정적으로 행동을 조정할 수 있음을 입증하였습니다.



### AceParse: A Comprehensive Dataset with Diverse Structured Texts for Academic Literature Parsing (https://arxiv.org/abs/2409.10016)
Comments:
          5 pages, 3 figures, 3 tables

- **What's New**: 데이터 중심의 AI 발전에 따른 새로운 방향으로, 학술 문헌의 구조화된 텍스트 파싱을 지원하는 AceParse 데이터셋을 소개합니다. 또한, AceParser라는 멀티모달 모델을 세밀하게 조정하여 다양한 구조화된 텍스트에 대한 파싱 정확도를 향상시켰습니다.

- **Technical Details**: AceParse는 수학적 표현이 포함된 문장, 테이블, 알고리즘 등의 다양한 구조화된 텍스트를 포함하는 최초의 포괄적인 오픈소스 데이터셋입니다. 이 데이터셋은 LaTeX 마크업 언어를 사용하여 텍스트 구조를 주석 처리합니다. AceParser는 Florence2 아키텍처를 기반으로 하여 문서 이미지를 패치로 나누고, 시각적 토큰과 텍스트 토큰 임베딩을 결합한 후 멀티모달 토큰 임베딩을 적용하여 구조화된 텍스트를 파싱합니다.

- **Performance Highlights**: AceParser는 F1 점수 4.1% 및 Jaccard 유사도 5% 향상을 달성하여 기존의 최첨단 모델을 능가하며, 멀티모달 모델이 학술 문헌 파싱에서 가지는 잠재력을 보여줍니다.



### HALO: Hallucination Analysis and Learning Optimization to Empower LLMs with Retrieval-Augmented Context for Guided Clinical Decision Making (https://arxiv.org/abs/2409.10011)
Comments:
          10 pages, 4 figures

- **What's New**: 최근 발표된 HALO 프레임워크는 의료 질문-답변(QA) 시스템의 정확성과 신뢰성을 향상시키기 위해 고안되었습니다. 이 프레임워크는 hallucinations(환각)을 감지하고 완화하는 데 초점을 맞춥니다.

- **Technical Details**: HALO 프레임워크는 LLMs(대규모 언어 모델)의 여러 쿼리 변형을 생성하고, 외부 열린 지식 기반에서 관련 정보를 검색하여 문맥을 풍부하게 만듭니다. 최대 한계 관련성(maximum marginal relevance) 점수를 사용하여 검색된 문맥의 우선순위를 매기고 이를 LLM에 제공하여 정답 생성을 합니다. LangChain을 통합하여 과정을 간소화했습니다.

- **Performance Highlights**: HALO는 Llama-3.1의 정확도를 44%에서 65%로, ChatGPT는 56%에서 70%로 크게 향상시켰습니다. 이 결과는 HALO가 의료 QA 시스템에서 환각 문제를 해결하는 데 중요한 역할을 함을 보여줍니다.



### SelECT-SQL: Self-correcting ensemble Chain-of-Thought for Text-to-SQL (https://arxiv.org/abs/2409.10007)
- **What's New**: SelECT-SQL은 자연어 질문을 SQL 쿼리로 자동 변환하는 Text-to-SQL 작업을 개선하는 새로운 in-context learning 솔루션으로, 이전 결과를 초월하는 성능을 보여줍니다.

- **Technical Details**: SelECT-SQL은 Chain-of-Thought (CoT) 프롬프트, 자기 교정 기능, 앙상블 기법을 조합하여 구성되어 있으며, 이를 통해 complex한 Text-to-SQL 작업에서 정확성을 극대화합니다. 특히, 구조 합성 CoT 기법을 사용하여 SQL 쿼리를 생성하는 다양한 단계를 자동으로 유도합니다.

- **Performance Highlights**: SelECT-SQL은 GPT-3.5-Turbo를 기반으로 Spider 리더보드 개발 세트에서 84.2%의 실행 정확도를 달성하며, 이는 다른 GPT-3.5-Turbo 솔루션(81.1%) 및 GPT-4 성능(83.5%)을 초월하는 결과입니다.



### FreeMark: A Non-Invasive White-Box Watermarking for Deep Neural Networks (https://arxiv.org/abs/2409.09996)
- **What's New**: 이 논문에서는 DNN (Deep Neural Network) 모델의 지적 재산 보호를 위한 새로운 워터마크 시스템인 FreeMark를 소개합니다. 기존 방법들은 모델 구조를 수정해야 하지만, FreeMark는 원래의 모델을 변경하지 않고도 워터마크를 삽입할 수 있어 성능 저하 없이 DNN을 보호할 수 있습니다.

- **Technical Details**: FreeMark는 기계 학습의 원리에 기반하여, 사전에 생성된 워터마크 벡터와 호스트 모델을 이용해 그래디언트 하강법 (gradient descent)으로 비밀 키를 생성합니다. 이 비밀 키는 모델의 활성화 값으로부터 워터마크를 추출하는 데 사용되며, 신뢰할 수 있는 제 3자에게 안전하게 저장됩니다. 이를 통해 의심되는 모델로부터 신뢰성 있는 워터마크 추출이 가능합니다.

- **Performance Highlights**: FreeMark는 다양한 워터마크 제거 공격을 효과적으로 저항할 수 있으며, 높은 워터마크 용량을 유지합니다. 실험 결과에 따르면 성능 저하는 없으면서도 강력한 방어력을 제공함을 입증하였습니다.



### Comprehensive Study on Sentiment Analysis: From Rule-based to modern LLM based system (https://arxiv.org/abs/2409.09989)
Comments:
          2 Images

- **What's New**: 이 논문은 인공지능(AI) 및 대형 언어 모델(LLMs) 맥락에서 감정 분석(Sentiment Analysis)에 대한 포괄적인 조사를 제공합니다. 전통적인 규칙 기반 방법에서 고급 딥 러닝 기술로의 전환을 강조하며 감정 분석의 역사적 발전을 살펴봅니다.

- **Technical Details**: 감정 분석은 자연어 처리(NLP)의 중요한 측면으로, 사전 기반 및 패턴 기반 접근에서 기계 학습(Machine Learning) 및 딥 러닝(Deep Learning) 모델로의 발전을 포함합니다. 또한 이 논문에서는 이중 언어 텍스트 처리, 풍자 감지, 편향 문제 등 주요 도전 과제를 논의합니다.

- **Performance Highlights**: 최신 접근 방식을 검토하고 새로운 트렌드를 식별하며 이 분야의 발전을 위한 미래 연구 방향을 제시합니다. 현재의 방법론을 종합하고 미래의 기회를 탐색하기 위해 감정 분석을 AI 및 LLM 맥락에서 철저히 이해하는 것을 목표로 하고 있습니다.



### Artificial Intelligence-Based Opportunistic Coronary Calcium Screening in the Veterans Affairs National Healthcare System (https://arxiv.org/abs/2409.09968)
- **What's New**: 이번 연구에서는 비심장 목적으로 촬영된 비조영(non-contrast) CT 스캔에서 관상동맥 칼슘(CAC)을 자동으로 정량화하는 심층 학습 알고리즘(AI-CAC)을 개발했습니다. 이는 98개의 의무 의료센터로부터 다양한 이미징 프로토콜과 스캐너를 포함한 데이터를 활용하여 수행되었습니다.

- **Technical Details**: AI-CAC은 446개의 전문가 세분화를 사용하여 개발되었으며, 비심장 비게이트(non-gated) CT 스캔에서 정확한 CAC 측정을 제공합니다. 이 알고리즘은 환자 795명의 데이터를 대상으로 ECG-게이트(gated) CAC 스코어와 비교하였으며, 0 vs. 비 0, 100 미만 vs. 100 이상의 Agatston 점수에서 각각 89.4% 및 87.3%의 정확성을 기록했습니다.

- **Performance Highlights**: AI-CAC는 10년 전반적인 사망률 예측 및 최초의 뇌졸중(stroke), 심근경색(MI), 사망에 대한 결합 예측에서 CAC 0 그룹과 >400 그룹 간에 각각 25.4% vs. 60.2%, 33.5% vs. 63.8%의 결과를 보였으며, 8,052명의 저선량 폐암 스크리닝 CT 데이터셋에서 38.4%가 AI-CAC >400으로 나타났습니다. 4명의 심장 전문의가 무작위로 샘플링한 400 이상 AI-CAC 환자들에 대한 이미지를 검토한 결과, 99.2%가 지질 저하 요법(lipid-lowering therapy)을 통해 혜택을 받을 것으로 확인되었습니다.



### An Offline Adaptation Framework for Constrained Multi-Objective Reinforcement Learning (https://arxiv.org/abs/2409.09958)
- **What's New**: 이번 연구에서는 여러 목적을 동시에 고려하는 multi-objective reinforcement learning (MORL) 문제를 해결하기 위해, 수작업으로 설계된 목표 선호도 없이도 몇 가지 시연(demonstration)만으로 기대되는 정책(preferred policy)의 선호를 암시적으로 파악할 수 있는 오프라인 적응 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크인 Preference Distribution Offline Adaptation (PDOA)는 두 가지 주요 기능을 포함합니다: 1) 다양한 선호에 반응하는 정책 세트를 학습하고, 2) 제공된 몇 가지 시연을 기반으로 목표 선호의 분포를 조정합니다. 이러한 접근법은 안전 임계값(safety threshold)이 명확하지 않은 경우에도 적용할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 제시된 시연에 의해 암시된 제약 조건을 충족하며, 실제 선호와 일치하는 정책을 추론하여 안전한 다목적 RL 문제를 해결하는 데 성공했습니다.



### Deep Graph Anomaly Detection: A Survey and New Perspectives (https://arxiv.org/abs/2409.09957)
Comments:
          24 pages, 6 figures, and 7 tables

- **What's New**: 이 논문은 Graph Anomaly Detection (GAD) 분야의 현재 방법론과 이론적 통찰을 종합적으로 검토하고, 이를 통해 GAD의 다양한 문제를 다룰 수 있는 효과적인 모델 디자인을 제안합니다.

- **Technical Details**: 이 연구에서는 GAD를 위한 딥러닝 접근 방식의 포괄적 리뷰를 제공합니다. GNN (Graph Neural Networks) 기반의 GAD 방법론을 GNN 백본 설계, GAD를 위한 프록시 태스크 설계, 그래프 이상 탐지 측정이라는 세 가지 관점에서 체계적으로 검토합니다. 또한, 13개의 세분화된 방법 카테고리를 제안하여 모델 디자인에 대한 보다 깊이 있는 통찰을 제공합니다.

- **Performance Highlights**: 본 연구는 GAD의 복잡한 문제와 도전 과제들을 논의하고, 현재까지 가장 많이 사용된 GAD 데이터셋 및 실험적 비교를 정리했습니다. 또한, GAD에서의 미해결 문제들을 제시하여 향후 연구 방향을 제안합니다.



### Fault Analysis And Predictive Maintenance Of Induction Motor Using Machine Learning (https://arxiv.org/abs/2409.09944)
Comments:
          Presented at ICEECCOT-2018, Published in IEEE Xplore, 6 pages, 3 figures

- **What's New**: 이 논문은 산업에서 널리 사용되는 유도 전동기(Induction motor)의 결함 감지 및 분류를 위한 기계 학습 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 3상 전압(three phase voltages) 및 전류(currents)를 입력으로 사용하며, 전압 및 전류 값을 기준으로 결함을 감지합니다. 모델은 오버전압(overvoltage), 저전압(under voltage), 단일 상(single phasing), 불균형 전압(unbalanced voltage), 과부하(overload), 접지 결함(ground fault)과 같은 일반적인 전기 결함을 빠르게 감지하는 심층 인공신경망(deep artificial neural network) 기반입니다. 또한, 유도 전동기 자체가 센서 역할을 하도록 하는 별도의 모델-프리(monitoring system) 모니터링 시스템을 제시합니다.

- **Performance Highlights**: 이 모델은 0.33 HP 유도 전동기의 실시간 데이터로 학습 및 테스트되었으며, 특정 순간에 제공되는 전압 및 전류 값을 분석하고 결함이 없는 상태 또는 특정 결함으로 분류합니다. 모델은 실제 전동기와 인터페이스되어 결함을 정확히 감지하고 분류하여 필요한 추가 작업을 취할 수 있도록 합니다.



### Towards Data Contamination Detection for Modern Large Language Models: Limitations, Inconsistencies, and Oracle Challenges (https://arxiv.org/abs/2409.09927)
Comments:
          12 pages, 1 figure

- **What's New**: 이 연구는 최신의 대형 언어 모델(LLM)에서의 데이터 오염(datat contamination) 감지 방법의 효과성과 신뢰성을 평가하기 위한 새로운 접근을 제시합니다. 연구진은 SOTA LLM의 오염 상태와 감지 방법의 견고함을 이중 분석함으로써 오염 감지의 복잡성과 필요성을 강조합니다.

- **Technical Details**: 연구에서는 다섯 가지 데이터 오염 감지 접근법을 평가하였으며, 이를 네 개의 최신 LLM 모델(GPT-4, Claude 3 Sonnet, LLaMA-3-Chat, LLaMA-2-Chat)과 여덟 개의 도전적인 데이터셋을 사용하여 분석하였습니다. 또한, LLaMA-2 모델을 사용하여 의도적으로 오염된 상태를 생성하고, 감지 방법의 효과성을 평가하기 위한 기준을 설정했습니다.

- **Performance Highlights**: 실험 결과, 현재 사용되고 있는 모든 데이터 오염 감지 방법들은 각각의 기본 가정이나 실제 응용에 한계가 있으며, 특히 최신의 도전적인 기준을 통해서는 일관된 결과를 도출하지 못했습니다. 더불어, 기존의 방법들이 일반적으로 고전적인 기준에서는 어느 정도 효과를 보이지만, 복잡한 기준에서는 불충분한 성능을 보임을 확인했습니다.



### SFR-RAG: Towards Contextually Faithful LLMs (https://arxiv.org/abs/2409.09916)
Comments:
          Technical report

- **What's New**: 이번 연구에서 우리는 SFR-RAG 모델을 소개합니다. 이 모델은 외부 맥락 정보를 적극적으로 활용하여 정보의 정확성과 관련성을 향상시키기 위해 설계된 작은 LLM입니다. 또한, ContextualBench라는 새로운 평가 프레임워크를 도입하여 다양한 RAG 벤치마크에서 모델의 성능을 일관되게 평가합니다.

- **Technical Details**: SFR-RAG 모델은 지식 검색기와 연계하여 작동하며, 복잡한 멀티홉 추론(multi-hop reasoning)과 신뢰할 수 있는 인용을 생성하는 기능이 포함되어 있습니다. 모델은 90억 파라미터로 구성되어 있으며, RAG 및 관련 에이전틱 작업에 특화된 교육을 받았습니다. 이 모델은 문서 데이터베이스에서 관련 패시지를 검색하는 과정을 단순화하며, ContextualBench를 통해 일관된 평가를 제공합니다.

- **Performance Highlights**: SFR-RAG-9B 모델은 ContextualBench의 7개 벤치마크 중 3개에서 최고 성능을 기록하며, Command-R+ (104B)와 GPT-4o를 상회하는 성능을 보여주었습니다. 이 모델은 10배 적은 파라미터로도 강력한 성능을 유지하며, 맥락 정보의 변화에도 강한 저항력을 보입니다. 또한 일반적인 지시 준수 작업에서도 경쟁력 있는 성능을 나타냅니다.



### REG: Refined Generalized Focal Loss for Road Asset Detection on Thai Highways Using Vision-Based Detection and Segmentation Models (https://arxiv.org/abs/2409.09877)
Comments:
          14 pages

- **What's New**: 이 논문은 태국 고속도로에서 중요한 도로 자산을 탐지하고 분할하는 새로운 프레임워크를 소개합니다. 이를 통해 Refined Generalized Focal Loss (REG)라는 고급 손실 함수를 활용하여, 작은 도로 요소들이나 클래스 imbalance 문제를 해결합니다.

- **Technical Details**: 제안된 REG 손실 함수는 복잡한 환경에서 예측 불확실성을 고려하는 확률적 정제와 도로 자산의 공간적 분포를 반영하는 공간-문맥 조정 항을 포함합니다. REG는 다중 작업 학습(multi-task learning) 전략을 적용하여 동시에 탐지와 분할을 최적화합니다.

- **Performance Highlights**: 실험 결과, REG를 적용한 모델은 mAP50 80.34와 F1-score 77.87를 달성하며 전통적인 방법보다 성능이 크게 향상되었음을 보여주었습니다. 이는 도로 자산 탐지와 분할의 정확성과 견고성을 높이는데 기여합니다.



### Critic as Lyapunov function (CALF): a model-free, stability-ensuring agen (https://arxiv.org/abs/2409.09869)
Comments:
          IEEE Conference on Decision and Control. Accepted for publication in proceedings of the conference

- **What's New**: 본 연구에서는 Critic As Lyapunov Function (CALF)이라는 새로운 reinforcement learning (강화 학습) 에이전트를 소개하고, 이를 통해 동적 시스템의 안정화를 보장함으로써 온라인 환경에서의 학습 성능을 크게 향상시키는 경우를 보여줍니다. CALF는 actor-critic (액터-비평가) 구조를 기반으로 하며, 기존의 SARSA 방법보다 나은 성과를 냈습니다.

- **Technical Details**: CALF는 model-free (모델 무관) 접근 방식을 채택하며, 매 학습 에피소드 동안 환경의 안정화를 보장합니다. 연구에 따르면 CALF는 모바일 로봇 시뮬레이터에서의 사례 연구에서 좋은 성과를 내었고, 향상된 nominal stabilizer (명목 안정기)를 개선할 수 있음을 보였습니다. CALF는 기존의 SARSA-m 알고리즘보다 훨씬 높은 성능을 발휘했습니다.

- **Performance Highlights**: CALF는 동적 시스템의 안정화에 있어 매우 효과적이며, 기존의 제어 이론과 강화 학습을 융합할 수 있는 가능성을 보여줍니다. 대부분의 경쟁 접근법은 offline (오프라인)이거나 model-based (모델 기반)이기 때문에 CALF의 장점이 더욱 두드러집니다.



### Towards Kinetic Manipulation of the Latent Spac (https://arxiv.org/abs/2409.09867)
- **What's New**: 이번 연구에서는 기존의 Graphical User Interfaces (GUIs)에서 벗어나, 실시간 RGB 카메라 피드를 활용하여 사전 훈련된 Convolutional Neural Networks (CNNs)를 이용한 간단한 feature extraction을 통해 생성 모델의 latent space를 효과적으로 조작하는 방법을 제시합니다. 이러한 새로운 패러다임은 Visual-reactive Interpolation으로 명명되었습니다.

- **Technical Details**: 이 시스템은 Generative Adversarial Networks (GANs) 및 diffusion 기반 text-to-image 모델을 활용하여, 사용자가 자신의 몸이나 얼굴 동작을 통해 생성 모델의 latent space를 실시간으로 조작할 수 있도록 합니다. StyleGAN의 latent space를 조작하는 데 초점을 맞추며, 최종적으로는 다양한 생성 모델과의 호환성을 통해 폭넓은 적용성을 목표로 합니다. 고급 기능 추출기(pre-trained feature extractors)를 사용하여, GAN의 latent space로의 인코딩을 가능하게 합니다.

- **Performance Highlights**: 이 연구는 사용자가 전체 몸, 얼굴, 장면 내 물체를 통해 이미지를 생성할 수 있는 상호작용을 가능하게 하여, 예술적 표현의 새로운 가능성을 열어줍니다. 카메라 또한 이미지 생성 파이프라인의 '두 번째 주체'로 작용하여, 생성된 출력에 협력적으로 영향을 미칠 수 있게 합니다.



### Constructing a Singing Style Caption Datas (https://arxiv.org/abs/2409.09866)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 S2Cap라는 새로운 오디오-텍스트 쌍 데이터셋을 소개합니다. 이 데이터셋은 다양한 보컬 및 음악적 특성을 포함하고 있으며, 특히 노래 스타일 캡셔닝(Singing Style Captioning) 작업을 수행하기 위한 최초의 전용 작업으로 제안되었습니다.

- **Technical Details**: S2Cap 데이터셋은 피치(pitch), 볼륨(volume), 템포(tempo), 기분(mood), 가수의 성별(gender) 및 나이(age), 음악 장르(musical genre) 및 감정 표현(emotional expression)과 같은 9가지 속성(attribute)을 고려합니다. 이를 기반으로 한 새로운 baseline 알고리즘은 CRESCENDO라는 메커니즘을 사용하여 오디오 인코더와 텍스트 디코더의 정렬 문제를 해결하며, 보컬 데믹싱(vocal demixing)을 통해 모델이 보컬 특성을 보다 정확하게 캡처할 수 있도록 합니다.

- **Performance Highlights**: S2Cap 데이터셋은 12,105개의 음악 트랙에서 71,215개의 캡션을 추출하였으며, 이를 통해 노래 스타일 캡셔닝의 성능을 크게 향상시킬 수 있음을 보여줍니다. 이 연구는 노래의 보컬 및 음악적 특성을 보다 잘 반영하는 캡션의 생성 가능성을 높입니다.



### A Survey of Out-of-distribution Generalization for Graph Machine Learning from a Causal View (https://arxiv.org/abs/2409.09858)
Comments:
          15 pages, 2 figures, 1 table

- **What's New**: 이 논문은 그래프 기계 학습(Graph Machine Learning, GML)에서의 인과관계 기반 접근 방식의 중요성을 강조하며, 특히 OOD(Out-of-Distribution) 데이터에 대한 일반화 문제를 해결하는 데 어떻게 도움이 되는지를 조명합니다.

- **Technical Details**: GML은 그래프 신경망(Graph Neural Network, GNN)과 같은 기술을 활용하여 다양한 분야에서 데이터 분석을 수행합니다. 최근의 연구들은 인과관계(causality)에 집중하여 데이터 생성 메커니즘을 이해하고 모델 예측을 향상시키는 데 기여하고 있습니다. 기존의 통계적 의존성에 의존하는 전통적인 GML 방식과는 달리, 인과론적 접근은 가짜 상관관계를 제거하여 진정한 인과 메커니즘을 파악합니다.

- **Performance Highlights**: 이 논문은 최근 인과관계가 포함된 GML 일반화의 발전을 체계적으로 검토하며, 인과관계를 활용하여 그래프 모델의 일반화를 향상시키는 주요 개념과 방법론을 제시합니다. 또한, 설명 가능성(explanation), 공정성(fairness), 강건성(robustness) 등 신뢰할 수 있는 GML의 다른 중요 영역에서 인과관계의 응용을 탐구합니다.



### Latent Diffusion Models for Controllable RNA Sequence Generation (https://arxiv.org/abs/2409.09828)
- **What's New**: 이 논문은 RNAdiffusion이라는 새로운 잠재 확산 모델(latent diffusion model)을 소개하며, 이를 사용하여 이산 RNA 서열을 생성하고 최적화하는 방법을 제안합니다. RNA는 생물학적 과정에서 매우 유동적이고 다양한 분자로, 그 서열은 길이와 구조적 특성이 다양합니다.

- **Technical Details**: RNAdiffusion은 사전 훈련된 BERT(Bidirectional Encoder Representations from Transformers) 타입 모델을 사용하여 원시 RNA를 생물학적으로 의미 있는 토큰 수준의 표현으로 인코딩합니다. Q-Former를 사용하여 이러한 표현을 고정 길이의 잠재 벡터로 압축하며, 자기 회귀 디코더가 잠재 변수에서 RNA 서열을 재구성하도록 훈련됩니다. 우리는 이 잠재 공간 내에서 연속 확산 모델을 개발하고 보상 네트워크(reward networks)를 훈련시켜 RNA의 기능적 특성을 추정합니다.

- **Performance Highlights**: RNAdiffusion의 실험 결과, 생성된 비암호화 RNA가 다양한 생물학적 지표에서 자연 분포와 잘 일치함을 확인했습니다. 특히 미 번역 영역(UTR)에서 높은 평균 리보솜 적재량(Mean Ribosome Loading, MRL)과 번역 효율성(Translation Efficiency, TE)을 기록했으며, TE는 166.7%, MRL은 52.6% 개선됨을 보여주었습니다.



### On the Effect of Robot Errors on Human Teaching Dynamics (https://arxiv.org/abs/2409.09827)
Comments:
          Accepted to 2024 International Conference on Human-Agent Interaction (HAI)

- **What's New**: 이 연구는 로봇 오류가 인간의 교수 동태에 미치는 영향을 조사하여, 효율적인 인터페이스 설계를 위한 중요한 통찰을 제공합니다.

- **Technical Details**: 본 논문은 RLHF(Reinforcement Learning from Human Feedback) 설정에서 진행된 사용자 연구를 통해 로봇 오류의 존재와 심각성이 인간 교수 동태의 세 가지 차원(피드백 세분화, 피드백 풍부함, 교수 시간)에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: 연구 결과, 오류가 있는 로봇을 가르칠 때 사람들이 더 많은 시간을 할애하고, 더 상세한 피드백을 제공하는 경향이 있음을 확인했습니다. 이는 사용자 경험과 로봇 학습 알고리즘 최적화에 중요한 시사점을 제공합니다.



### GP-GPT: Large Language Model for Gene-Phenotype Mapping (https://arxiv.org/abs/2409.09825)
- **What's New**: GP-GPT는 유전적 표현형 지식 표현 및 유전체 관계 분석을 위한 첫 번째 전문화된 대형 언어 모델로, 3,000,000개 이상의 유전체, 단백질체 및 의학 유전 관련 용어를 사용하는 두 단계의 미세 조정을 통해 개발되었습니다.

- **Technical Details**: GP-GPT는 OMIM, DisGeNET 및 dbGaP와 같은 여러 주요 데이터 출처의 구조화된 데이터와 비구조화된 데이터를 활용하여 유전자 및 표현형의 관계를 통합적으로 모델링합니다. 이 모델은 질병 유전학 쿼리에서 정확한 답변을 제공하는 능력, 유전 정보 검색, 그리고 유전자와 표현형 간의 관계 결정 능력을 평가하기 위해 세 가지 주요 NLP 작업을 수행합니다.

- **Performance Highlights**: GP-GPT는 Llama2, Llama3 및 GPT-4와 같은 최신 LLM보다 우수한 성능을 보였습니다. 이 모델은 유전 질환의 관계 연구를 향상시키고 유전체 및 의료 유전학 분야에서 정확하고 효율적인 분석을 촉진할 수 있는 잠재력을 가지고 있습니다.



### Causal Inference with Large Language Model: A Survey (https://arxiv.org/abs/2409.09822)
Comments:
          15 pages, 2 figures, 3 tables

- **What's New**: 본 논문은 causal inference (인과 추론) 분야에서의 large language models (LLMs)의 최근 발전을 정리하고, 각각의 인과 문제와 접근 방식을 비교하여 LLM이 제시하는 독특한 기회를 탐구합니다.

- **Technical Details**: 인과 추론은 관찰 뒤에 있는 인과 관계를 밝혀내는 중요한 분야로, 인간의 지식, 수학적 추론, 데이터 마이닝을 복합적으로 통합해야 합니다. 전통적인 인과 추론 프레임워크에서는 numerical 값보다 domain knowledge (도메인 지식)를 강조하는 반면, LLMs는 대량의 텍스트 정보에서 도메인 지식을 추출하여 인과 관계를 더 잘 이해하는 데 기여할 수 있습니다.

- **Performance Highlights**: LLMs는 인과 관계를 이해하고 해석하는데 있어 명확한 이점을 제시하며, 텍스트 데이터 내에서의 인과 관계 파악 및 설명 가능한 인과 추론을 가능하게 합니다. 이 논문에서는 LLM 기반 인과 추론에서앞으로의 연구 방향과 제약 사항에 대한 논의도 포함되어 있습니다.



### Famba-V: Fast Vision Mamba with Cross-Layer Token Fusion (https://arxiv.org/abs/2409.09808)
Comments:
          Camera ready version of ECCV 2024 The Fourth Workshop on Computational Aspects of Deep Learning

- **What's New**: Famba-V는 Vision Mamba(Vim) 모델의 훈련 효율성을 높이기 위해 cross-layer token fusion 기술을 도입한 새로운 방법론입니다. 기존의 token fusion 방법은 모든 레이어에 일관적으로 적용되었으나, Famba-V는 다양한 cross-layer 전략을 사용하여 유사한 토큰들만을 선택하여 융합함으로써 효율성을 극대화합니다.

- **Technical Details**: Famba-V는 2D 비주얼 데이터를 처리하기 위해 Vision Mamba(Vim) 모델 내에서 토큰의 유사성을 측정하여 유사 정보를 포함한 토큰들을 융합합니다. 토큰 융합 전략은 레이어 선택에서 최적의 정확도 및 효율성 트레이드오프를 제공하는 데 기여합니다. 실험은 CIFAR-100 데이터셋에서 수행되었습니다.

- **Performance Highlights**: Famba-V는 기존의 Vim 모델과 비교할 때 훈련 시간을 줄이고 최대 메모리 사용량을 감소시킴으로써 훈련 효율성을 significantly 향상시킵니다. cross-layer token fusion 전략을 적용한 결과 기존의 전 레이어 token fusion 방식보다 더 나은 정확도-효율성 트레이드오프를 제공합니다.



### Abnormal Event Detection In Videos Using Deep Embedding (https://arxiv.org/abs/2409.09804)
- **What's New**: 이 논문은 비감독 학습(unsupervised learning) 방식을 활용하여 비디오에서 이상 사건(anomalous event)을 탐지하기 위한 새로운 접근 방식을 제안합니다. 기존의 이상 탐지 방법들이 정상 이벤트를 학습한 후 이상 값을 감별하는 방식이었다면, 본 논문에서는 심층 신경망(deep neural network)의 목적함수와 이상 탐지 과제를 동시에 최적화하는 하이브리드 아키텍처(hybrid architecture)를 통해 효율성을 극대화합니다.

- **Technical Details**: 제안된 비정상 사건 탐지 방법은 세 가지 주요 단계로 구성됩니다: 1) 레이턴트 피처 추출(Latent Feature Extraction), 2) 피처 융합(Feature Fusion), 3) 일급 분류(One-class classification). 이 시스템은 깊이(depth) 맵, 광학 흐름(optical flow), 그리고 외형 특징(appearance features)을 융합하여 비디오 신호에서 비정상적인 사건을 식별합니다. 사전 학습된 모델을 사용하여 각 입력 모달리티의 레이턴트 표현(latent representation)을 합칩니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 비정상 데이터의 임베딩이 하이퍼센터(hypercenter)에서 멀리 떨어지도록 조정함으로써, 정상 데이터는 하이퍼센터 근처에 군집화되도록 설계되었습니다. 이러한 방식은 영상 내에서 비정상 사건을 효과적으로 식별할 수 있도록 하여, 더욱 높은 성능을 자랑하는 이상 탐지 시스템을 구현합니다.



### Multiple Rotation Averaging with Constrained Reweighting Deep Matrix Factorization (https://arxiv.org/abs/2409.09790)
- **What's New**: 본 논문에서는 기존의 최적화 기반 및 학습 기반 방법의 장점을 결합하여 여전히 레이블이 필요하지 않은 비지도 학습(Unsupervised Learning) 방식으로 다회전 평균(Multiple Rotation Averaging, MRA) 문제를 해결하는 새로운 방법을 제안합니다.

- **Technical Details**: 딥 매트릭스 팩토리제이션(Deep Matrix Factorization) 기법을 통해 비제한 선형 공간에서 다회전 평균 문제를 직접 해결합니다. 명시적으로 저랭크(Low-rank) 및 대칭(Symmetric) 구조를 가진 신경망 모델을 설계하여 관찰된 상대 측정값을 네트워크 최적화에 대한 제약으로 활용합니다. 스패닝 트리 기반의 엣지 필터링(Spanning Tree-based Edge Filtering) 및 재가중치 기법(Reweighting Scheme)을 통해 회전 이상치(Outlier)의 영향을 억제합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과를 통해 제안된 방법의 효과가 입증되었으며, 성능 개선을 위한 동적 깊이 선택 전략(Dynamic Depth Selection Strategy)도 도입하였습니다.



### BEnDEM:A Boltzmann Sampler Based on Bootstrapped Denoising Energy Matching (https://arxiv.org/abs/2409.09787)
Comments:
          20 pages, 7 figures, 2 tables

- **What's New**: 이 연구에서는 Boltzmann 분포에서 독립적이고 동일하게 분포된(IID, Independent and Identically Distributed) 샘플을 생성할 수 있는 효율적인 샘플러를 개발하는 데 중점을 두었습니다. 에너지 함수를 기반으로 하는 신경 샘플링 기술을 학습하여 에너지 기반의 Denoising Energy Matching(EnDEM) 샘플러를 제안합니다.

- **Technical Details**: EnDEM은 Monte Carlo(MC)에서 추정된 에너지를 목표로 하며, 분산 폭증(Variance Exploding, VE) 노이즈 프로세스에 따릅니다. 이를 통해 적은 잡음 대상으로 샘플링을 수행할 수 있으며, Bootstrap ENDEM(BEnDEM)은 에너지 추정의 편향과 분산 균형을 맞추기 위해 부트스트래핑 기법을 사용하여 성능을 개선합니다.

- **Performance Highlights**: BEnDEM은 40 차원 Gaussian Mixture Model(GMM) 및 4 입자 double-welling potential(DW-4) 실험에서 state-of-the-art 성능을 달성했습니다. 이 모델들은 기존 방법들보다 더 높은 견고성을 보여줍니다.



### Large Language Model Based Generative Error Correction: A Challenge and Baselines forSpeech Recognition, Speaker Tagging, and Emotion Recognition (https://arxiv.org/abs/2409.09785)
Comments:
          IEEE SLT 2024. The initial draft version has been done in December 2023. Post-ASR Text Processing and Understanding Community: this https URL

- **What's New**: 이번 논문에서는 고정된 자동 음성 인식(ASR) 모델의 텍스트 디코딩 결과를 활용하여 음향 모델링 과제를 향상시키는 방법을 모색합니다. 특히, 새로운 '생성 음성 전사 오류 수정(GenSEC)' 챌린지를 소개하며, 이는 ASR 후의 언어 모델링 작업으로 3가지 주요 과제를 포함합니다.

- **Technical Details**: 제안된 챌린지는 ASR-LLM(대형 언어 모델) 연계 모델을 통해 성능 향상을 꾀하고 있으며, 시행될 세 가지 작업은 (1) ASR 후 전사 수정, (2) 화자 태깅, (3) 감정 인식을 포함합니다. 이 챌린지는 텍스트 기반 추론을 통해 음향적 정보가 포함된 오류 수정 방법을 탐구하여, 음성 인식 시스템을 사용할 때의 새로운 접근법을 제시합니다.

- **Performance Highlights**: 기존 ASR 처리 방식에서 LLM은 외부 지식을 활용하여 정확성을 높이고 있으며, 다양한 성능 측정 지표를 통해 그 효용성 및 가능성을 보여주고 있습니다. 특히, 대형 언어 모델을 활용하여 비언어적 정보와 발화의 맥락을 활용하면서 음성 인지의 틀을 확장하고자 합니다.



### Enhancing Lesion Segmentation in PET/CT Imaging with Deep Learning and Advanced Data Preprocessing Techniques (https://arxiv.org/abs/2409.09784)
- **What's New**: 이번 연구는 PET/CT 이미징에서 병변(segmentation)의 정확성을 높이기 위해 딥러닝(deep learning)을 적용하였으며, AutoPET challenge III에서 제공된 900개의 전체 신체 FDG-PET/CT 데이터와 600개의 PSMA-PET/CT 데이터를 활용하였습니다.

- **Technical Details**: 연구에서는 데이터 전처리(preprocessing) 및 데이터 증강(augmentation) 기법을 활용하여 모델의 견고성과 일반화 가능성을 확보했습니다. 비영점(normalization) 정규화과 RandGaussianSharpen 추가 및 Gamma 변환 매개변수의 조정을 통해 성능 향상을 검토하였습니다.

- **Performance Highlights**: FDG 추적자의 경우 기본 설정에서 Dice 계수가 63.19%로 중간 정도의 분할 정확도를 나타냈으며, PSMA 추적자의 경우 32.07%로 나타났습니다. Gaussian sharpening 기법을 활용한 데이터 증강 후 FDG는 64.11%, PSMA는 44.55%로 향상되었습니다. 특히 최대 클립 값 280을 적용했을 때 FDG의 Dice 점수는 53.69%로 개선되었습니다.



### Automated Lesion Segmentation in Whole-Body PET/CT in a multitracer setting (https://arxiv.org/abs/2409.09766)
- **What's New**: 본 연구는 FDG 및 PSMA PET/CT 이미지를 자동으로 분할하는 워크플로우를 탐구하며, 이 과정에서 YOLOv8을 활용하여 데이터를 분류하고 각각의 이미지를 별도로 전처리하여 종양 분할 정확도를 향상시키는 것을 목표로 하고 있습니다.

- **Technical Details**: 자동 종양 분할 과정은 두 단계로 구성됩니다. 첫째, FDG-PET 및 PSMA-PET 의학 이미지를 구별하기 위한 분류 모델을 훈련합니다. 둘째, 독립적으로 훈련된 두 개의 3D U-Net을 사용해 각각 FDG와 PSMA 데이터를 위한 장기 및 종양 분할을 시행합니다. 데이터는 AutoPET challenge III에서 제공된 전체 신체 FDG PET/CT와 PSMA PET/CT를 기반으로 하고, nnU-Net 프레임워크에 통합된 전처리 절차를 거칩니다.

- **Performance Highlights**: PET 모델은 FDG와 PSMA PET 이미지를 구별하는 데 있어 99.85%의 분류 정확도를 달성했습니다. FDG의 경우 다이스 계수(Dice coefficient)는 0.8408, PSMA는 0.7385였으며, 거짓 양성(FPvol)과 거짓 음성(FNvol) 볼륨은 각각 FDG는 1.7979 및 2.3625, PSMA는 9.3574 및 5.0745로 나타났습니다. 이는 두 가지 이미징 모달리티 간의 분할 성능 차이를 보여줍니다.



### ELMI: Interactive and Intelligent Sign Language Translation of Lyrics for Song Signing (https://arxiv.org/abs/2409.09760)
Comments:
          18 pages excluding reference and appendix

- **What's New**: 이 논문은 ELMI라는 새로운 접근법을 소개하고 있습니다. ELMI는 사용자가 노래 가사를 수화로 번역하는 데 도움이 되는 웹 기반 도구로, 노래-signing의 접근성을 크게 향상시키고 있습니다.

- **Technical Details**: ELMI는 노래 가사를 라인별로 편집할 수 있게 해주며, 실시간으로 가사 하이라이트와 음악 비디오 클립을 함께 제공합니다. 또한 대규모 언어 모델(LLM) 기반의 인공지능과의 대화 기능을 통해 의미, 표기, 감정, 타이밍에 대해 논의할 수 있습니다.

- **Performance Highlights**: 연구 참가자들은 ELMI를 통해 번역 품질이 크게 향상되었다고 보고했으며, 대화 기능이 자신감과 독립성을 증가시켰다고 합니다. ELMI는 사용자에게 필요한 모든 자원을 한 곳에서 제공하여 의사 결정 과정을 간소화하며, 사용자는 ELMI를 자신의 작업 흐름에 통합하고자 하는 의사를 보였습니다.



### Explore the Hallucination on Low-level Perception for MLLMs (https://arxiv.org/abs/2409.09748)
- **What's New**: 본 논문에서는 Multi-modality Large Language Models(MLLMs)의 낮은 수준의 시각 인식 및 이해 작업에서의 자가 인식(self-awareness)을 정의하고 평가하기 위해 QL-Bench라는 벤치마크를 제안합니다.

- **Technical Details**: QL-Bench는 2,990장의 단일 이미지와 1,999장의 이미지 쌍으로 구성된 LLSAVisionQA 데이터셋을 바탕으로 하며, 각 이미지에 대한 저수준(low-level) 특성에 관한 개방형 질문이 포함되어 있습니다. 모델의 자가 인식 능력은 질문 유형에 따라 달라지며, 특히 복잡한 질문에 대해 더 나은 자가 인식 성능을 보이는 경향이 있습니다.

- **Performance Highlights**: 15개의 MLLMs을 평가한 결과, 일부 모델은 낮은 수준의 시각 능력을 잘 수행하나 자가 인식 능력은 상대적으로 부족함을 나타냈습니다. 간단한 질문에 대한 정답률은 높지만 복잡한 질문에서는 자가 인식이 향상되는 경향이 보였습니다.



### Benchmarking LLMs in Political Content Text-Annotation: Proof-of-Concept with Toxicity and Incivility Data (https://arxiv.org/abs/2409.09741)
Comments:
          Paper prepared for delivery at the 8th Monash-Warwick-Zurich Text-as-Data Workshop, September 16-17, 2024: 11 pages, 3 tables, 3 figures

- **What's New**: 이번 연구는 OpenAI의 GPT 모델들과 여러 오픈 소스 LLM이 정치적 콘텐츠에 대한 주석 작업을 수행하는 능력을 벤치마킹했습니다. 이 연구는 300만 개 이상의 디지털 상호작용을 포함하는 새로운 시위 사건 데이터 세트를 사용했으며, 소셜 미디어에서의 독성 및 무례함에 대해 인간 코더가 주석을 단 진리 기준(gold standard) 레이블을 포함했습니다.

- **Technical Details**: 벤치마킹에 포함된 모델은 Google의 Perspective 알고리즘과 OpenAI의 GPT 모델들, 그리고 로컬에서 배포된 오픈 소스 LLM들입니다. 결과적으로, 느슨한 임계 값을 사용하는 Perspective API, GPT-4o 및 Nous Hermes 2 Mixtral이 다른 LLM의 제로 샷 분류 주석보다 우수한 성능을 보였습니다.

- **Performance Highlights**: Nous Hermes 2와 Mistral OpenOrca는 매개변수가 적음에도 불구하고 높은 성능을 발휘하여 성능, 구현 비용 및 컴퓨팅 시간 간의 좋은 균형을 제공할 수 있는 매력적인 선택지로 평가되었습니다. GPT 모델은 전반적으로 좋은 신뢰성과 컴퓨팅 시간을 보여주었지만, 오픈 소스 LLM만이 주석 작업의 완전한 재현성을 보장할 수 있었습니다.



### From Challenges and Pitfalls to Recommendations and Opportunities: Implementing Federated Learning in Healthcar (https://arxiv.org/abs/2409.09727)
- **What's New**: 이 논문은 Federated Learning(FL)을 사용한 최신 건강관리 연구를 종합적으로 검토하고, 임상적 유용성이 낮은 이유를 분석합니다. 데이터 프라이버시와 보안을 유지하면서도 다양한 센터에서 대규모 연구 협업이 가능하다는 FL의 잠재력을 강조하고, 이에 대한 실용적인 적용 방안을 제시합니다.

- **Technical Details**: FL은 각 클라이언트(병원 및 기관)가 데이터를 로컬에 두고 파라미터 업데이트만 중앙 서버에 보내는 방식으로, 중앙 집중형 데이터 없이 모델을 훈련시키는 패러다임입니다. 본 리뷰는 FL의 최근 연구를 기반으로 메소드의 방법론적 결함과 편향, 커뮤니케이션 비용, 개인정보 보호 문제 등을 다룹니다.

- **Performance Highlights**: FL 방식으로 훈련된 모델들은 중앙집중형 데이터셋에서 훈련된 모델들과 비슷한 성능을 가지며, 단일 기관 데이터셋에서 훈련된 모델보다 우수한 결과를 보입니다. 특히, 여러 기관의 데이터에 접근할 수 있는 FL 모델이 더 높은 일반화 성능을 보이는 가능성을 보여줍니다.



### Exploring Utility in a Real-World Warehouse Optimization Problem: Formulation Based on Quantun Annealers and Preliminary Results (https://arxiv.org/abs/2409.09706)
Comments:
          2 pages, 2 figures. Paper presented at the 5th IEEE International Conference on Quantum Computing and Engineering (IEEE QCE 2024)

- **What's New**: 본 논문은 D-Wave의 Quantum Annealer를 활용하여 Warehouse Optimization Problem (WOP)을 효과적으로 해결하기 위한 'Quantum Initialization for Warehouse Optimization Problem (QI4WOP)'라는 메커니즘을 제안합니다. 기존의 고전 소프트웨어에 통합하여 산업 문제를 최적화하기 위해 설계되었습니다.

- **Technical Details**: QI4WOP 모듈은 두 가지 단계 (First level placement phase와 Post-processing phase)로 나뉩니다. 첫 번째 단계는 하위 문제인 sub-WOP을 해결하여 가능한 많은 항목을 저장하는 데 집중하고, Leap Constrained Quadratic Model (CQM) Hybrid Solver (LeapCQMHybrid)를 사용하여 최적화합니다. 두 번째 단계에서는 partial solution을 완성하고, 다양성을 추가하기 위해 mutant solution을 생성합니다.

- **Performance Highlights**: 예비 실험에서 QI4WOP은 고전 초기화 모듈에 비해 훨씬 더 많은 feasible solutions을 더 짧은 시간 내에 생성할 수 있음을 보여주었습니다. 또한 인스턴스의 크기가 증가함에 따라 기존의 방법이 비효율적이었던 반면, QI4WOP은 문제의 복잡성이 증가할수록 더 뛰어난 성능을 발휘했습니다.



### GFlowNet Pretraining with Inexpensive Rewards (https://arxiv.org/abs/2409.09702)
- **What's New**: 이 논문에서는 분자 구조 생성에서 화학적 공간을 더 포괄적으로 탐색하기 위해 개별 원자를 빌딩 블록으로 활용하는 Atomic GFlowNets (A-GFNs)라는 새로운 생성 모델을 소개합니다.

- **Technical Details**: A-GFNs는 비지도(pre-training) 사전 훈련 방식을 사용하여 오프라인 약물 유사 분자 데이터 세트로부터 학습하며, 약물 유사성(drug-likeliness), 위상 극성 표면적(topological polar surface area), 합성 용이성(synthetic accessibility score)과 같은 저렴하지만 유용한 분자 서술자(molecular descriptors)를 통해 모델에 조건을 줍니다. 이 과정에서 A-GFNs는 약리학적 특성이 바람직한 화학적 공간의 영역으로 유도됩니다.

- **Performance Highlights**: 논문에서는 ZINC15 오프라인 데이터 세트에서 A-GFN을 사전 훈련한 후, 특정 목표 속성을 최적화하기 위한 목표 조건(goal-conditioned) 미세 조정(fine-tuning) 과정을 통해 다른 기존의 약물 디자인 방법들에 비해 접근 방식의 효과성을 입증했습니다.



### Training Safe Neural Networks with Global SDP Bounds (https://arxiv.org/abs/2409.09687)
- **What's New**: 이 논문은 신경망의 안전성을 보장하면서 훈련하는 새로운 접근 방식을 제시합니다. 우리는 반정형 프로그래밍(Semidefinite Programming, SDP)을 사용하여 안전성을 검증하고, 고차원 입력 영역에서의 한계를 극복합니다. 특히 ADMM(Alternating Direction Method of Multipliers) 기반 훈련 기법을 도입하여 Adversarial Spheres 데이터셋에서 신경망 분류기를 훈련시켜 100%의 정확성을 달성하였습니다.

- **Technical Details**: 저자는 안전 개념과 원하는 행동을 정의하기 위해 안전 서브셋과 타겟 서브셋을 소개합니다. 안전한 행동을 보장하기 위해 SDP를 사용하여 신경망의 출력에 대해 정밀한 경계를 생성하는 방법을 발표하였으며, ADMM 방법을 활용하여 입력 차원이 40인 신경망 분류기를 성공적으로 훈련했습니다. 이를 통해 우리는 고차원 시스템에 대한 신뢰할 수 있는 신경망 검증 방법 개발을 진전시키고자 합니다.

- **Performance Highlights**: 개발된 신경망 분류기는 입력 차원 d=40에서 안전하게 작동하며, 높은 정확도와 안전성을 입증하였습니다. SDP를 사용한 연구에서는 기존의 선형 경계보다 훨씬 더 정밀한 경계를 제공하여 고차원 입력 영역에서의 검증을 가능하게 하였습니다.



### Anatomy of Machines for Markowitz: Decision-Focused Learning for Mean-Variance Portfolio Optimization (https://arxiv.org/abs/2409.09684)
Comments:
          7 pages, 3 figures, 3 tables

- **What's New**: 이번 연구에서는 Decision-Focused Learning (DFL) 방법을 통해 평균-분산 최적화(mean-variance optimization, MVO)에서 주식 수익률 예측 모델을 어떻게 조정할 수 있는지를 조사하고, asset(자산)별 예측 오차를 달리 감소시키는 방법을 설명합니다.

- **Technical Details**: DFL은 예측(prediction)과 최적화(optimization) 단계를 통합하여 의사결정을 향상시키는 새로운 학습 프레임워크입니다. MSE(Mean Squared Error)로 예측 오차를 최소화하는 전통적인 예측 방식의 한계를 극복하고, DFL은 결정 오류를 줄이는 손실 함수(loss function)를 최소화하여 모델을 훈련합니다. 이는 MVO의 입력 매개변수인 예상 수익률, 분산, 공분산의 추정을 보다 효과적으로 할 수 있도록 돕습니다.

- **Performance Highlights**: DFL을 적용함으로써 우리는 주식 포트폴리오의 효율성을 높일 수 있으며, 이는 투자 성과 향상으로 이어질 수 있습니다. 기존의 연구들이 DFL의 투자성과에 대한 가능성을 제시한 바 있지만, 주식 수익률 예측 모델의 구체적인 변화 메커니즘에 대한 분석은 없었습니다.



### Reliable Multi-View Learning with Conformal Prediction for Aortic Stenosis Classification in Echocardiography (https://arxiv.org/abs/2409.09680)
Comments:
          This preprint has not undergone any post-submission improvements or corrections. The Version of Record of this contribution is published in: International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer (2024) under the same title

- **What's New**: 이 논문에서는 초음파 진단의 한계인 2D 이미지를 3D 해부학적 정보와 결합하여 불확실성을 시간에 따라 재학습하는 방법인 Re-Training for Uncertainty (RT4U)를 소개합니다. RT4U는 불확실성을 약한 정보를 가진 입력 데이터에 도입하여 분류 정확도를 높이는 방법입니다.

- **Technical Details**: RT4U는 데이터 중심의 접근 방식으로, 약한 정보를 가진 입력의 불확실성을 증가시키는 모델-불가지론적(training agnostic) 기법입니다. 기존의 심장 판막 질환 분류 기법에 통합하여 사용할 수 있으며, conformal prediction 기술과 결합할 경우, 예측 세트를 동적으로 크기 조정하여 높은 정확도로 실제 클래스가 포함되어 있음을 보장합니다.

- **Performance Highlights**: 다양한 3개의 데이터셋(TMED-2, 비공식 AS 데이터셋, CIFAR-10 유사 장난감 데이터셋)에서 RT4U는 모든 데이터셋에서 향상된 결과를 보였으며, 이는 자동화된 심장 판막 질환 검출의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### ExploreSelf: Fostering User-driven Exploration and Reflection on Personal Challenges with Adaptive Guidance by Large Language Models (https://arxiv.org/abs/2409.09662)
Comments:
          17 pages excluding reference and appendix

- **What's New**: 이번 연구에서는 LLM(큰 언어 모델)을 활용한 새로운 애플리케이션 ExploreSelf를 제안하여 사용자가 자신의 감정을 돌아보고 정리하는 과정을 지원합니다. ExploreSelf는 사용자가 동적으로 생성된 질문을 통해 반성할 수 있도록 돕는 시스템입니다.

- **Technical Details**: ExploreSelf는 사용자가 자신의 초기 서사에서 관련 주제를 자유롭게 확장하고, 깊이 있는 반성을 위해 주제의 세부 사항을 탐구할 수 있도록 설계되었습니다. 시스템은 사용자의 글쓰기에 맞춤형으로 키워드와 의견을 제공하는 온디맨드(guidance) 지침을 제공합니다.

- **Performance Highlights**: 19명의 참가자들과 진행된 탐색 연구를 통해, 참가자들은 ExploreSelf의 가이드와 자유로운 탐색의 균형을 높이 평가하며, 이는 더 깊은 몰입과 통찰로 이어졌습니다. 참가자들은 개인적인 도전 과제를 탐색하면서 시스템의 적응형 기능이 그들의 통제감과 자율성을 향상시켰다고 보고했습니다.



### KAN v.s. MLP for Offline Reinforcement Learning (https://arxiv.org/abs/2409.09653)
Comments:
          5 pages,2 figures

- **What's New**: 이 논문은 Kolmogorov-Arnold Networks (KAN)를 오프라인 강화 학습(offline Reinforcement Learning, RL) 알고리즘의 기본 블록으로 활용한 최초의 연구입니다. KAN을 이용해 Multi-Layer Perceptions (MLP)과의 성능, 파라미터 규모, 훈련 효율성을 비교합니다.

- **Technical Details**: KAN은 학습과정에서 MLP와는 다른 이론적 접근법과 네트워크 아키텍처를 사용하여 더 적은 파라미터로도 동등하거나 더 우수한 성능을 발휘합니다. 본 연구에서는 KAN과 MLP 기반 반복적 Q-학습(Conservative Q-learning, CQL)을 결합하여 D4RL 벤치마크에서 실험을 진행했습니다.

- **Performance Highlights**: KAN 기반 CQL을 통해 성능은 MLP와 비슷하거나 더 나은 결과를 나타내며, 파라미터 수는 크게 줄어들어 훈련 효율성이 향상되었습니다. 이는 오프라인 RL 작업에 따라 기본 네트워크를 선택할 수 있는 옵션을 제공합니다.



### Self-supervised Learning for Acoustic Few-Shot Classification (https://arxiv.org/abs/2409.09647)
- **What's New**: 본 논문에서는 바이오아쿠스틱(bioacoustics)에서의 레이블(label) 부족 문제를 해결하기 위해 self-supervised learning을 이용한 새로운 아키텍처를 제안합니다. 이 아키텍처는 CNN(Convolutional Neural Network)과 SSM(Structured State Space Model)을 결합하여 acoustic signal을 효과적으로 분류하는 방법을 탐구합니다.

- **Technical Details**: 제안된 아키텍처는 CNN 기반의 전처리 블록과 S4(S4 Structured State Space Sequence Model) 구조를 포함합니다. CNN은 주파수 도메인(frequency domain)에서의 의존성을 캡처하는데 사용되며, SSM은 시간 도메인(temporal domain)에서의 장기 의존성을 학습합니다. 이러한 구조는 raw audio waveform을 stacked spectrogram으로 변환한 후, CNN과 SSM 블록을 통해 처리하여 contrastive learning과 classification을 위한 잠재 공간(latent space)을 생성합니다.

- **Performance Highlights**: 5-class 5-shot classification 문제에서 제안된 아키텍처는 기존의 state-of-the-art 모델들을 초월하는 성능을 보여주었습니다. 특히, 이 모델은 낮은 레이블 샘플 수에서도 우수한 정확도를 달성하여 적은 라벨링 예산을 가진 실세계 데이터에도 효과적으로 적용될 수 있음을 입증하였습니다.



### COSCO: A Sharpness-Aware Training Framework for Few-shot Multivariate Time Series Classification (https://arxiv.org/abs/2409.09645)
Comments:
          5 pages, 5 figures, CIKM '24 Short Paper Track

- **What's New**: 본 논문에서는 다변량 시계열 분류를 위한 새로운 학습 프레임워크인 COSCO(Centroid Oriented Sharpness-Controlled Optimization)를 제안합니다. COSCO는 불충분한 학습 데이터 상황에서도 DNN(Deep Neural Networks)의 일반화 능력을 향상시키기 위해 sharpness-aware minimization (SAM) 최적화 및 prototypical loss 함수를 결합한 것입니다.

- **Technical Details**: COSCO는 모델 훈련 중에 sharp local minima와 같은 문제가 발생하지 않도록 loss 값을 기반으로 perturb-then-update 작업을 수행합니다. 기존의 cross-entropy loss를 대체하기 위해 prototypical loss를 사용하여 데이터가 부족한 환경에서의 outlier에 대한 강인성을 높이고 있습니다. 이는 DNN의 일반화 성능을 크게 개선하는 것으로 나타났습니다.

- **Performance Highlights**: COSCO는 기존의 기준 모델들보다 강력한 성능을 보이며, 몇 차례의 실험을 통해 그 효과성을 입증하였습니다. 특히, ResNet 모델이 SpokenArabicDigits 데이터셋에서 30-shot 설정 시 87.0%에서 10-shot에서는 70.0%, 그리고 1-shot 설정에서는 36.2%로 급락하는 문제를 해결하는 데 기여하고 있습니다.



### AACessTalk: Fostering Communication between Minimally Verbal Autistic Children and Parents with Contextual Guidance and Card Recommendation (https://arxiv.org/abs/2409.09641)
Comments:
          19 pages excluding reference and appendix

- **What's New**: 이번 연구에서는 AACessTalk라는 새로운 태블릿 기반 AI 매개 커뮤니케이션 시스템을 소개합니다. 이 시스템은 최소 언어적 자폐 아동(MVA)과 부모 간의 의미 있는 대화를 촉진합니다.

- **Technical Details**: AACessTalk는 대화 중 부모가 아동과 소통하도록 돕기 위해 실시간 안내를 제공하며, 아동에게는 맥락에 맞는 어휘 카드를 추천합니다. 이 시스템은 부모의 음성 입력을 텍스트로 전사하여 대화 상황에 맞는 카드 큐레이션도 수행합니다. 또한, 이 시스템은 하루 11개의 MVA 아동-부모 쌍과 함께 진행된 2주간의 배포 연구를 포함합니다.

- **Performance Highlights**: 연구 결과, AACessTalk를 통해 모든 쌍의 높은 참여도가 나타났으며 대화 빈도와 턴 테이킹(turn-taking)이 증가했습니다. 부모는 아동이 완전한 문장을 만드는 압박감을 덜 느끼고, 아동은 자신의 의도를 표현할 기회를 얻어 소통의 즐거움을 느꼈습니다.



### A Novel Framework For Text Detection From Natural Scene Images With Complex Background (https://arxiv.org/abs/2409.09635)
- **What's New**: 이번 논문에서는 복잡한 배경에서 텍스트 영역을 탐지하기 위해 Wavelet Transform을 활용한 새로운 효율적인 방법을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 원본 이미지를 그레이스케일 형태로 변환한 후, Sub-band filtering을 통해 Wavelet 변환을 수행합니다. 이후에는 중심점을 기준으로 Region clustering 기법을 적용하여 각 영역에 Bounding box를 적합시켜 텍스트 영역을 식별합니다.

- **Performance Highlights**: 이 방법은 이전 방법들에 비해 더욱 정교하고 효율적이며, 특정 텍스트 글꼴 크기에 제한되지 않아 일반화된 접근이 가능합니다. 실험에 사용된 샘플 세트는 다양한 배경을 가진 50개의 이미지로 구성되어 있습니다.



### Confidence Estimation for LLM-Based Dialogue State Tracking (https://arxiv.org/abs/2409.09629)
- **What's New**: 이 논문은 대화형 AI 시스템의 신뢰성을 개선하기 위한 방법을 제시하며, 주로 대화 상태 추적(Dialogue State Tracking, DST)에서 모델 불확실성을 활용하는 새로운 접근 방식을 탐색하고 있습니다.

- **Technical Details**: 연구에서는 open-weight 및 closed-weight LLMs에 대해 신뢰도 점수(confidence scores)를 추정하는 네 가지 방법을 평가하고, softmax, 원시 토큰 점수(raw token scores), 언어로 표현된 신뢰도(verbalized confidences) 및 이들의 조합을 바탕으로 한 방법을 사용했습니다. AUC(Area Under the Curve) 메트릭을 이용하여 신뢰도 점수를 평가하며, self-probing 메커니즘을 통해 closed models의 성능을 향상시켰습니다.

- **Performance Highlights**: fine-tuning된 open-weight 모델을 활용한 DST에서 우수한 joint goal accuracy(JGA)를 달성했으며, 모델 신뢰도 점수의 보정 수준이 향상되었다는 점이 주요 성과로 나타났습니다.



### Can Large Language Models Grasp Event Signals? Exploring Pure Zero-Shot Event-based Recognition (https://arxiv.org/abs/2409.09628)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 이벤트 기반(advanced event-based) 비주얼 콘텐츠를 이해하는 능력을 탐구한 최초의 연구로, CLIP과 추가적인 교육 없이도 이벤트 기반 객체 인식을 수행할 수 있음을 보여줍니다. 특히 GPT-4o 및 4turbo와 같은 모델이 이벤트 기반 비주얼 콘텐츠를 직접 인식할 수 있는 능력을 평가했습니다.

- **Technical Details**: 이 논문에서는 이벤트 스트림을 이벤트 프레임(event frame) 또는 복원된 프레임(reconstructed frame) 형식으로 변환하여 LLMs에 입력합니다. 이벤트 스트림은 픽셀 위치와 폴라리티로 특징지어진 타임스탬프를 가진 이벤트로 구성되어 있으며, 이 데이터는 E2VID 또는 E2HQV와 같은 기술을 통해 시각적으로 이해 가능한 이미지로 재구성됩니다. 마지막으로 정확도(accuracy)를 평가 지표로 사용하여 다양한 모델의 인식 능력을 검증합니다.

- **Performance Highlights**: 실험 결과, GPT-4o는 N-ImageNet 데이터셋에서 가장 최신 방법들보다 다섯 배 더 높은 인식 정확도를 기록하며, 잘 설계된 프롬프트가 인식 성능을 크게 향상시킬 수 있음을 입증했습니다. LLMs의 성능은 각기 다른 이벤트 데이터 입력 표현의 영향을 체계적으로 평가하여 개선 가능성을 보여줍니다.



### Understanding Simplicity Bias towards Compositional Mappings via Learning Dynamics (https://arxiv.org/abs/2409.09626)
Comments:
          4 pages

- **What's New**: 이 논문은 compositional mapping을 학습할 때 모델의 일반화 성능을 높이는 방법에 대한 새로운 관점을 제시합니다.

- **Technical Details**: 저자들은 compositional mappings를 코딩 길이(coding length)의 관점에서 가장 간단한 일대일 대응(bijection)으로 보고, 이는 모델의 Kolmogorov 복잡성(Kolmogorov complexity)의 상한을 나타냅니다. 또한, 간단함의 편향(simplicity bias)이 gradient descent를 통한 신경망(neural network) 훈련의 본질적인 속성임을 보여줍니다.

- **Performance Highlights**: 이러한 발견들은 모델이 적절히 훈련될 때 자발적으로 잘 일반화(generalize) 되는 이유를 부분적으로 설명합니다.



### Stutter-Solver: End-to-end Multi-lingual Dysfluency Detection (https://arxiv.org/abs/2409.09621)
Comments:
          IEEE Spoken Language Technology Workshop 2024

- **What's New**: 현재의 비정상 발화 모델링 방법은 템플릿 매칭 알고리즘을 사용하여 언어 간 범위가 일반화되지 않으며, 훈련 데이터가 증가함에 따라 확장 불가능합니다. 이러한 문제를 해결하기 위해, YOLO 객체 탐지 알고리즘에서 영감을 받은 Stutter-Solver라는 엔드 투 엔드 프레임워크를 제안합니다.

- **Technical Details**: Stutter-Solver는 비정상 발화가 포함된 음성 및 참조 텍스트를 입력받아 비정상 발화 유형과 시간을 정확히 예측합니다. 이를 위해 VCTK-Pro, VCTK-Art 및 AISHELL3-Pro라는 세 가지 새로운 비정상 발화 코퍼스를 도입하여 반복, 차단, 누락, 대체 및 연장과 같은 상호 비정상 발화 자연어 구사를 시뮬레이션합니다. 또한, 명시적 템플릿 없이 다국어 비정상 발화를 탐지할 수 있습니다.

- **Performance Highlights**: Stutter-Solver는 새로운 기준(VCTK-Art, VCTK-Pro, AISHELL3-Pro), 공개 코퍼스 및 nfvPPA 발화에서 최고 성능을 달성했습니다. 이 연구는 비정상 발화 탐지의 최첨단 성능을 기록했습니다.



### Enhancing Text Annotation through Rationale-Driven Collaborative Few-Shot Prompting (https://arxiv.org/abs/2409.09615)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)을 활용하여 데이터 주석 작업의 효율성과 일관성을 개선하는 새로운 방법론인 근거 기반 협업 텍스트 주석 방법(Rationale-Driven Collaborative annotation method)을 제안합니다.

- **Technical Details**: 연구진은 LLM의 출력을 개선하기 위해 여러 LLM의 주석 결과를 통합하는 근거 기반 협업 접근 방식을 사용합니다. 이 방법은 이전 주석을 참조하여 순차적으로 주석을 수행하며, 유사한 예제를 비교하여 정밀도를 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 몇 개의 샷(few-shot) 기법 및 기타 기준 방법들보다 향상된 주석 정확도를 보여주었으며, 특히 복잡한 주석 작업에서 두드러진 성과를 나타냈습니다.



### Rethinking KenLM: Good and Bad Model Ensembles for Efficient Text Quality Filtering in Large Web Corpora (https://arxiv.org/abs/2409.09613)
- **What's New**: 이 논문에서는 대량의 웹 데이터에서 고품질 데이터를 효과적으로 필터링하기 위해 Good KenLM(우수한 n-그램 모델)과 Bad KenLM(저품질 데이터 모델)을 앙상블하는 방법을 제안합니다. 기존의 KenLM 교육 방식의 한계를 극복할 수 있는 접근법입니다.

- **Technical Details**: Good KenLM은 고품질 데이터로 훈련된 n-그램 기반 언어 모델이며, Bad KenLM은 노이즈가 포함된 저품질 데이터로 훈련되었습니다. 두 모델의 PPL(perplexity) 점수를 결합하여 불필요한 내용을 효과적으로 제거하며, Z-score normalization을 통해 두 모델의 점수를 통합합니다.

- **Performance Highlights**: 제안된 앙상블 방법은 전통적인 KenLM 훈련 방법에 비해 노이즈 콘텐츠를 효과적으로 줄이며, 고품질 콘텐츠를 보존하면서도 계산 리소스 사용을 최소화한 실험 결과를 보였습니다.



### Integrating Audio Narrations to Strengthen Domain Generalization in Multimodal First-Person Action Recognition (https://arxiv.org/abs/2409.09611)
- **What's New**: 본 논문에서는 첫 번째 시점 활동 인식을 위한 멀티모달 프레임워크를 제안하여 도메인 일반화를 개선하고자 합니다. 이 프레임워크는 모션(motion), 오디오(audio), 외관(appearance) 기능을 통합하여 다양한 환경에서의 도메인 이동에 대한 회복력을 분석합니다.

- **Technical Details**: 제안된 방법은 비디오 클립과 관련 오디오, 시각적 내레이션(visual narration), 오디오 내레이션(audio narration)을 쌍으로 구성된 훈련 샘플을 사용합니다. 서로 다른 모달리티에 대해 별도의 인코더를 훈련하여 각 모달리티의 기능을 추출하고, 오디오와 비주얼 간의 일관성(consistency)을 평가하여 훈련 중 예측에 미치는 영향을 최적화합니다.

- **Performance Highlights**: ARGO1M 데이터셋에서 최첨단 성과를 달성하여 보지 못한 시나리오와 위치에서도 효과적인 일반화를 보여주었습니다. 특히, 모션과 오디오 기능은 각각 25.8%, 32.7%의 성능 저하를 보인 반면, 외관 기능은 54.8%의 성능 저하를 기록하여, 도메인 이동에 견고한 특징을 강조합니다.



### A Survey of Foundation Models for Music Understanding (https://arxiv.org/abs/2409.09601)
Comments:
          20 pages, 2 figures

- **What's New**: 이번 논문은 인공지능(AI) 기술과 음악 이해의 교차점을 다룬 초기 리뷰 중 하나로, 특히 대규모 음악 기초 모델의 음악 이해 능력을 조사하고 분석하였습니다. 최근 발전한 대규모 언어 모델(LLMs)과 기초 모델(FMs)들이 음악과 언어의 통합, 그리고 복잡한 음악적 특징과 개념을 포착할 가능성을 제시합니다.

- **Technical Details**: 대규모 기초 모델은 수많은 파라미터를 통해 복잡한 음악적 특징과 패턴을 포착하고, 언어와 음악 기능을 크로스 모달(feature alignment)로 통합할 수 있습니다. 이러한 모델들은 기존 모델들이 가지던 제한점을 극복하고 문맥적 이해 능력을 통해 복합적인 음악 이해 작업을 처리할 수 있는 가능성을 가지고 있습니다. 이를 통해 감정 분석과 심리적 지식이 풍부해져 음악 감정 이해 능력이 향상됩니다.

- **Performance Highlights**: 기존 음악 모델들은 단순한 작업에서만 유용했으나, 대규모 음악 모델들은 다중 레이블 음악 분류나 음악 시나리오 reasoning 등의 작업에서 두드러진 성과를 보여주고 있습니다. 그러나 여전히 섬세한 감정 인식과 음악의 내부 논리 구조 연관성에는 제한이 있어 향후 연구 방향에 대한 제언이 필요합니다.



### Improving Statistical Significance in Human Evaluation of Automatic Metrics via Soft Pairwise Accuracy (https://arxiv.org/abs/2409.09598)
- **What's New**: 이 논문에서는 Soft Pairwise Accuracy (SPA)라는 새로운 메타 메트릭을 제안합니다. 이는 기존 Pairwise Accuracy (PA)의 단점을 보완하며, 인간의 평가와 메트릭 평가의 통계적 중요성을 모두 고려합니다.

- **Technical Details**: SPA는 메트릭의 평가와 인간의 평가의 p-값을 사용하여 메트릭이 인간의 판단과 얼마나 일치하는지를 평가합니다. 이 메트릭은 시스템 수와 평가 세그먼트 수에 대한 안정성이 높으며, 메트릭 간의 동점 문제(quantization)를 완화합니다.

- **Performance Highlights**: SPA는 2024 WMT 메트릭 공유 과제의 공식 시스템 레벨 메트릭으로 선정되었습니다. PA에 비해 보다 통계적으로 유의미한 결과를 제공하며, 두 시스템 간의 성능 비교를 보다 세밀하게 수행할 수 있습니다.



### Open-World Test-Time Training: Self-Training with Contrast Learning (https://arxiv.org/abs/2409.09591)
Comments:
          10page

- **What's New**: 본 논문에서는 Open-World Test-Time Training (OWTTT) 접근 방식을 소개하며, 이는 강한 Out-of-Distribution (OOD) 데이터의 존재에 직면했을 때 딥러닝 모델을 일반화하는 데 초점을 두고 있습니다. 동시에, Open World Dynamic Contrastive Learning (OWDCL)이라는 새로운 방법론을 통해 초기 피처 추출 문제를 해결하고 모델의 강건성을 촉진합니다.

- **Technical Details**: OWDCL은 상반된 샘플 쌍을 활용하여 초기 TTT 단계에서의 대비(constrast)를 강화합니다. NT-XENT contrastive learning loss function을 사용하여 이 과정을 지원하고 있으며, 이러한 접근법은 weak OOD 데이터를 강한 OOD로 잘못 분류하는 문제를 해결하는 데 기여하고 있습니다.

- **Performance Highlights**: OWDCL 모델은 기존의 최첨단 모델에 비해 다양한 데이터셋에서 우수한 성능을 보여줍니다. 이는 실질적인 응용을 위한 딥러닝 모델의 적응 및 일반화를 보다 효과적으로 수행할 수 있음을 나타냅니다.



### ValueCompass: A Framework of Fundamental Values for Human-AI Alignmen (https://arxiv.org/abs/2409.09586)
- **What's New**: 최신 연구에서는 AI 시스템의 인간 가치와의 정렬의 필요성을 강조하며, ValueCompass라는 프레임워크를 제안합니다. 이는 심리학 이론과 체계적 검토에 기반하여 인간-AI 정렬을 평가하는 도구입니다.

- **Technical Details**: ValueCompass는 12가지 동기적 가치 유형을 포함하며, 49개의 기본 가치를 체계적으로 코드화합니다. 이 프레임워크는 생성적 AI 모델과 인간 간의 가치의 정렬 정도를 측정하는 측정 도구인 'Value Form'을 제공합니다. 72명의 참가자와 5개의 최신 언어 모델(LMs)을 활용하여 4가지 시나리오에서 인간과 LMs의 가치 평가를 비교합니다.

- **Performance Highlights**: 연구의 결과, LMs는 인간과 상당한 가치 불일치를 보였으며, 'Choose Own Goals'와 같은 가치를 LMs는 지지했지만, 인간들은 대부분 이의에 동의하지 않았습니다. 이는 LMs가 윤리적 원칙보다 운영 효율성을 우선시할 수 있음을 나타냅니다. 이러한 불일치는 AI 개발의 윤리적 접근 방식을 재고할 필요성을 제기합니다.



### MindScape Study: Integrating LLM and Behavioral Sensing for Personalized AI-Driven Journaling Experiences (https://arxiv.org/abs/2409.09570)
Comments:
          arXiv admin note: text overlap with arXiv:2404.00487

- **What's New**: MindScape는 AI 기반 저널링에서 새로운 접근 방식을 선보이며, 학생들의 정신 건강 문제를 해결하기 위한 효과적인 중재 방안을 제공합니다.

- **Technical Details**: MindScape는 대화 참여(conversational engagement), 수면(sleep), 위치(location)와 같은 수동적으로 수집된 행동 패턴을 대형 언어 모델(Large Language Models, LLMs)과 통합하여 개인화된 저널링 경험을 제공합니다. 이 연구는 20명의 대학생을 대상으로 8주간 진행되었습니다.

- **Performance Highlights**: MindScape 앱은 긍정 정서(positive affect)를 7% 향상시키고, 부정 정서(negative affect), 외로움(loneliness), 불안(anxiety) 및 우울증(depression)을 각각 11%, 6% 감소시켰습니다. PHQ-4 점수도 주별로 감소하였으며, 마음 챙김(mindfulness)은 7%, 자기 반성(self-reflection)은 6% 개선되었습니다.



### TG-LLaVA: Text Guided LLaVA via Learnable Latent Embeddings (https://arxiv.org/abs/2409.09564)
- **What's New**: 논문에서 제안된 Text Guided LLaVA (TG-LLaVA)는 텍스트로 시각 인코더를 유도하여 비전-언어 모델(VLM)을 최적화하는 새로운 접근 방식을 제시합니다. 이는 기존의 방법들이 커넥터나 언어 모델 컴포넌트 개선에 집중했던 것과는 정반대의 접근입니다.

- **Technical Details**: TG-LLaVA 아키텍처는 두 개의 주요 모듈인 text-guided feature optimization mask (TG-FOM) 모듈과 text-guided detail perceiver (TG-DP) 모듈로 구성됩니다. TG-FOM 모듈은 입력 텍스트를 전역적으로 분석하여 언어 정보를 이미지 특성에 추가하는 역할을 합니다. TG-DP 모듈은 세밀한 분석을 통해 텍스트를 파싱하고, 이미지 관점에서 정보를 융합하는 데 사용됩니다.

- **Performance Highlights**: TG-LLaVA는 여러 데이터셋에서 기존 기준 모델인 LLaVA-1.5에 비해 성능 향상을 보여주었으며, 추가적인 데이터 없이도 경쟁력 있는 결과를 구출하였습니다. 실험 결과는 TG-LLaVA의 효과성을 확고히 하였으며 다양한 설정에서도 일관된 향상이 있음을 입증했습니다.



### Evaluating authenticity and quality of image captions via sentiment and semantic analyses (https://arxiv.org/abs/2409.09560)
- **What's New**: 이 연구는 COCO-MS 데이터셋을 사용하여 인간 생성 캡션의 감정 분석과 의미 변동성을 평가하는 새로운 방법을 제안합니다. 또한, 모델이 생성한 캡션의 감정 점수와 비교하여 훈련 데이터의 품질을 평가하는 접근 방법을 보여줍니다.

- **Technical Details**: 약 150K 이미지로 구성된 COCO-MS 데이터셋에서 각각의 이미지에 대해 4-6개의 캡션이 수집되었습니다. 감정 분석은 사전 훈련된 모델인 Twitter-RoBERTa-base를 사용하였고, BERT 임베딩을 사용하여 의미의 변동성을 분석했습니다. 회귀 분석(multiple linear regression)을 통해 감정 점수와 객체 카테고리의 관계를 조사했습니다.

- **Performance Highlights**: 대부분의 캡션은 중립적인 감정을 지닌 것으로 나타났으며, 강한 감정이 있는 캡션은 약 6%로 특정 객체 카테고리에 의해 영향을 받았습니다. 모델이 생성한 캡션의 강한 감정 비율은 1.5%에 불과하여 인간 생성 캡션과의 상관관계가 없었습니다.



### Enhancing Printed Circuit Board Defect Detection through Ensemble Learning (https://arxiv.org/abs/2409.09555)
- **What's New**: 이 논문은 서로 다른 PCB 결함 탐지 모델들의 시너지를 활용하지 않은 기존 연구의 한계를 극복하기 위해 앙상블 학습 전략을 적용한 종합 검사 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 EfficientDet, MobileNet SSDv2, Faster RCNN, YOLOv5의 네 가지 최첨단 PCB 결함 감지 모델을 포함하여, 이러한 모델들을 하이브리드 투표 전략으로 앙상블하여 결함 탐지 정확도를 향상시키고자 합니다.

- **Performance Highlights**: 이 앙상블 학습 프레임워크는 다양한 PCB 결함을 감지하는 데 있어 95%의 정확도를 달성하며, 이는 개별 모델보다 상당히 향상된 성능임을 나타냅니다.



### COMFORT: A Continual Fine-Tuning Framework for Foundation Models Targeted at Consumer Healthcar (https://arxiv.org/abs/2409.09549)
Comments:
          25 pages, 10 figures. This work has been submitted to the ACM for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: COMFORT라는 새로운 프레임워크를 제안하여 웨어러블 의료 센서(WMS) 데이터 기반의 질병 탐지에서 Transformer 기반 모델을 지속적으로 미세 조정하는 방법을 제공합니다. 이를 통해 사용자 데이터를 활용하여 다양한 질병을 효과적으로 탐지할 수 있는 경량화된 솔루션을 개발했습니다.

- **Technical Details**: COMFORT는 건강한 개인으로부터 수집된 대규모 생리 신호 데이터셋을 사용하여 Transformer 기반 모델의 사전 학습을 수행하며, masked data modeling (MDM) 접근 방식을 채택합니다. 저차수 적응(low-rank adaptation, LoRA)와 같은 파라미터 효율적 미세 조정(PEFT) 방법을 사용하여 다양한 질병 탐지 작업에 적응하도록 모델을 미세 조정합니다.

- **Performance Highlights**: COMFORT는 기존 방법에 비해 최대 52%의 메모리 오버헤드를 줄이면서 매우 경쟁력 있는 성능을 달성하였으며, 개인화된 조기 질병 탐지 솔루션을 제공하는 데 중요한 기반을 다졌습니다.



### VernaCopter: Disambiguated Natural-Language-Driven Robot via Formal Specifications (https://arxiv.org/abs/2409.09536)
- **What's New**: 이 논문은 자연 언어(NL) 명령을 Robot에 전달하기 위해 신호 시간 논리(STL) 스펙을 사용하는 새로운 LLM 기반 로봇 모션 플래너인 VernaCopter를 제안합니다. 이 시스템은 NL 명령의 모호성을 줄이고 로봇의 모션 제어를 위한 질 높은 경로 생성을 가능하게 합니다.

- **Technical Details**: VernaCopter는 자연어 명령과 구체적인 작업 목표 간의 다리 역할을 하는 신호 시간 논리(STL) 사양을 통한 로봇 모션 플래닝 시스템입니다. STL을 활용함으로써 NL의 불확실성과 LLM의 모호성을 줄이고 시스템의 효율성과 신뢰성을 높입니다. 각 기능 모드는 효율성과 복잡성 수준에 맞게 조정될 수 있으며, 수정된 프롬프트 기술이 성공률을 높이는 데 기여합니다.

- **Performance Highlights**: VernaCopter 플래너는 전통적인 NL 프롬프트 기반 플래너에 비해 움직임 계획 작업의 성공률을 크게 향상시키는 것으로 실험을 통해 검증되었습니다. 두 가지 작은 실험적 시나리오에서 이 시스템의 효과를 입증하며, NL 주도의 로봇 설계에서의 잠재력을 강조합니다.



### Enhancing Skin Disease Diagnosis: Interpretable Visual Concept Discovery with SAM Empowermen (https://arxiv.org/abs/2409.09520)
- **What's New**: 본 연구에서는 피부 병변 진단을 위한 새로운 Cross-Attentive Fusion 프레임워크를 제안합니다. 이는 Segment Anything Model (SAM)을 활용하여 피부 질병에 대한 비주얼 개념을 생성하고, 지역적 시각 개념과 글로벌 이미지 특징을 통합하여 진단 성능을 향상시킵니다.

- **Technical Details**: SAM(Segment Anything Model)은 11백만 개의 라이센스 이미지로부터 1억 개 이상의 마스크를 추출하여 학습된 프롬프트 기반(segmentation) 모델입니다. 이 모델은 단일 포인트, 포인트 집합, 바운딩 박스 또는 텍스트와 같은 여러 가지 프롬프트를 지원합니다. SAM의 구조는 이미지 인코더, 프롬프트 인코더, 마스크 디코더로 구성되어 있으며, 특히 Masked Autoencoder(MAE)와 Vision Transformer(ViT)를 이용하여 높은 해상도의 입력 처리와 세부 정보 캡처를 가능하게 합니다.

- **Performance Highlights**: 두 개의 피부 질환 데이터셋에 대한 광범위한 평가 결과, 제안된 방법이 병변 진단과 해석 가능성에서 효과적임을 입증했습니다. 이 연구는 실제 사례에서의 적용 가능성과 신뢰도를 높이는 데 기여합니다.



### Deep Learning Under Siege: Identifying Security Vulnerabilities and Risk Mitigation Strategies (https://arxiv.org/abs/2409.09517)
Comments:
          10 pages, 1 table, 6 equations/metrics

- **What's New**: 본 연구에서는 현재 배포되고 있는 Deep Learning (DL) 모델의 보안 도전 과제를 다루며, 향후 발전할 DL 및 AI 기술에 대한 도전 과제도 예측합니다. 이를 통해, 리스크 완화 기술을 제안하고 이의 효과를 측정하기 위한 메트릭을 제시합니다.

- **Technical Details**: DL 모델은 인간의 뇌의 학습 능력을 모방하며, 적절한 입력 집합이 있다면, 충분한 복잡성을 가진 DL 모델은 함수 f:𝑥→𝑦를 학습할 수 있습니다. 이는 많은 뉴런, 연결 및 비선형 활성화 덕분에 가능합니다. 본 연구에서는 DL 보안의 최전선 도전 과제 목록을 제공하고, 공격 분류를 위한 새로운 명명 규칙을 도입하며, DL 보안 도전 과제별 리스크 완화 전략을 종합적으로 다룹니다.

- **Performance Highlights**: 이 연구는 DL 아키텍처의 보안 도전 과제를 체계적으로 정리하고, 각 공격 벡터를 수치적으로 사용할 수 있는 메트릭을 제안하여 효과성을 정량화할 수 있는 방법을 시연합니다. 이를 통해 상징적인 시나리오를 통해 메트릭 사용 방법을 설명합니다.



### Planning Transformer: Long-Horizon Offline Reinforcement Learning with Planning Tokens (https://arxiv.org/abs/2409.09513)
Comments:
          11 pages, 5 figures, Submitted to AAAI

- **What's New**: 본 논문에서는 Decision Transformer를 기반으로 한 새로운 Planning Tokens 개념을 도입하여, 오프라인 강화 학습에서 긴 시간 측면 문제를 효과적으로 해결합니다. 이는 기존의 다음 토큰 예측을 넘어, 에이전트의 장기적 미래 정보를 포함하고 있습니다.

- **Technical Details**: 우리는 Planning Transformer 모델을 제안하며, 이는 Dual-Timescale Token Prediction을 통해 고급 Planning Tokens을 사용하여 에이전트의 저수준 정책을 안내함으로써 성능을 향상시킵니다. 이 접근 방식은 오토회귀 모델의 누적 오류 문제를 해결하고, HRL의 신용 할당 문제를 극복합니다.

- **Performance Highlights**: 이 모델은 복잡한 D4RL 환경에서 기존의 오프라인 RL 방법들과 비교하여 우수한 성능을 보이며, 긴 시간 지향 작업에서도 새로운 최첨단 성능을 설정합니다. 또한, Planning Tokens는 에이전트의 정책 해석 가능성을 높입니다.



### Explaining Deep Learning Embeddings for Speech Emotion Recognition by Predicting Interpretable Acoustic Features (https://arxiv.org/abs/2409.09511)
- **What's New**: 이 논문은 음성 감정 인식(SER) 분야에서 딥러닝의 임베딩(embedding) 기술을 설명하기 위한 수정된 프로빙(probing) 접근법을 제안합니다. 저자들은 감정에 기여하는 특정 해석 가능한 음향(acoustic) 특성을 식별하고, 이를 통해 임베딩의 신뢰성을 높이고 의료 및 보안 응용 프로그램에서의 사용을 촉진하려고 합니다.

- **Technical Details**: 이 연구에서는 WavLM(DL) 임베딩과 eGeMAPS 음향 특성을 사용하여 SER을 평가합니다. 두 가지 주요 방법론이 포함됩니다: (i) 모든 임베딩 집합에서 예측하고, (ii) 각 감정 예측에 가장 중요한 임베딩 차원들의 부분 집합을 활용하여 음향 특성을 예측합니다. 저자들은 SHAP를 사용해 SER 분류에 대한 eGeMAPS 특성의 중요도를 순위별로 매기고, WavLM DL 모델의 정보를 해석합니다.

- **Performance Highlights**: 실험 결과, 에너지, 주파수, 스펙트럴 및 시간적 요소의 음향 특성이 SER에서 점차적으로 정보량이 감소함을 보여줍니다. 이 연구는 감정을 구분하는 데 있어 특정 음향 특성이 임베딩 모델에 얼마나 중요한지를 밝히고, 프로빙 분류기 방법이 해석 가능한 음향 특성과 임베딩 간의 관계를 성립시키는 데 유용하다는 것을 입증합니다.



### ESPnet-EZ: Python-only ESPnet for Easy Fine-tuning and Integration (https://arxiv.org/abs/2409.09506)
Comments:
          Accepted to SLT 2024

- **What's New**: ESPnet-EZ는 기존의 ESPnet 모델을 다양한 작업에 신속하게 쉽게 미세 조정하고 추론할 수 있도록 설계된 확장 툴킷입니다. Python 전용 인터페이스를 통해 연구원들이 모델을 구축하고 디버깅하는 데 필요한 노력을 대폭 줄이고 있습니다.

- **Technical Details**: ESPnet-EZ는 Kaldi 스타일의 종속성을 제거하고, Python 인터페이스를 통해 ESPnet의 핵심 로직을 노출합니다. Trainer와 ESPNetEZDataset의 두 가지 주요 모듈로 구성되며, 사용자가 특정 작업에 맞게 Trainer를 쉽게 커스터마이징할 수 있습니다.

- **Performance Highlights**: ESPnet-EZ는 ESPnet에 비해 새로 작성해야 할 코드가 2.7배 줄어들고 의존 코드의 양이 6.7배 감소했습니다. 또한 기존 머신러닝 Python 프레임워크와의 통합이 용이하며, 다양한 음성 처리 작업을 지원합니다.



### Synthetic4Health: Generating Annotated Synthetic Clinical Letters (https://arxiv.org/abs/2409.09501)
Comments:
          ongoing work, 48 pages

- **What's New**: 이 연구는 민감한 정보를 포함하는 임상 서신(clinical letters)의 생성을 위한 다양한 방법론을 탐색하며, 이러한 서신들을 쉽게 사용할 수 있는 신뢰할 수 있는 비식별화된 합성 임상 서신(synthetic clinical letters)을 생성하는 것을 목표로 합니다.

- **Technical Details**: 다양한 사전 훈련된 언어 모델(pre-trained language models, PLMs)을 활용하여 텍스트 마스킹(masking) 및 생성을 시도했습니다. Bio_ClinicalBERT 모델을 중점적으로 사용하며, 단어(예: 명사, 동사) 마스킹 전략(masking strategies)의 효과를 실험하고, 성능 평가는 정성적(methods) 및 정량적(methods) 측면에서 진행되었습니다. 또한, 합성 서신의 유용성을 평가하기 위해 명명된 개체 인식(Named Entity Recognition, NER) 작업도 수행되었습니다.

- **Performance Highlights**: 1) 인코더 전용 모델(encoder-only models)이 인코더-디코더 모델(encoder-decoder models)보다 성능이 우수하다. 2) 일반적인 코퍼스에서 훈련된 인코더 전용 모델이 임상 데이터에서 훈련된 모델과 비슷한 성능을 보인다. 3) 임상 개체 및 문서 구조의 보존이 모델의 미세 조정보다 우선시되어야 한다. 4) 다양한 마스킹 전략은 합성 임상 서신의 품질에 영향을 미친다. 5) 평가 지표로 BERTScore가 기본 지표로 사용되어야 하며, 나머지 지표들은 보조적인 참고 지표로 활용된다. 6) 문맥 정보는 모델의 이해도에 큰 영향을 미치지 않는다.



### Multi-Scale Grouped Prototypes for Interpretable Semantic Segmentation (https://arxiv.org/abs/2409.09497)
Comments:
          8 pages, 5 figures, 4 tables

- **What's New**: 본 논문은 multi-scale 프로토타입(part) 학습을 활용한 해석 가능한 의미론적 분할 방법인 ScaleProtoSeg를 제안합니다. 이 방법은 서로 다른 스케일에서 다양한 프로토타입 부분을 명시적으로 학습하고, 이들 사이의 상호작용에 대한 정보를 제공하는 희소 집합(sparse grouping) 메커니즘을 응용합니다.

- **Technical Details**: 제안된 ScaleProtoSeg는 (i) 여러 스케일에서 프로토타입을 명시적으로 학습하고 (ii) 스케일 특정 프로토타입들의 희소 조합(sparse combination)을 학습하는 집합 절차(grouping procedure)를 정의합니다. 이 방식은 해석 가능성을 증가시키며, 결정 과정에서 소수의 프로토타입으로 제약을 두어 모델의 성능을 유지합니다.

- **Performance Highlights**: Pascal VOC, Cityscapes, ADE20K 세 가지 벤치마크 데이터셋에서 제안된 ScaleProtoSeg 방법이 기존의 프로토타입 기반 방법보다 우수한 성과를 보였으며, 해석 가능성 측면에서도 안정성(stability), 일관성(consistency), 희소성(sparsity)으로 측정된 개선된 결과를 보였습니다.



### Hacking, The Lazy Way: LLM Augmented Pentesting (https://arxiv.org/abs/2409.09493)
Comments:
          9 pages, 7 figures

- **What's New**: 본 논문에서는 'LLM Augmented Pentesting'이라는 새로운 접근 방식을 소개하며, 이는 'Pentest Copilot'이라는 도구를 통해 입증되었습니다. 이 도구는 대형 언어 모델(LLM)을 침투 테스트(penetration testing) 워크플로우에 통합하여 안전성 향상을 목표로 합니다.

- **Technical Details**: 이 연구는 'chain of thought' 메커니즘을 포함하여 토큰 사용을 간소화하고 성능을 향상시키며, 고유한 Retrieval Augmented Generation 구현을 통해 환각(hallucination)을 최소화하고 최신 기술에 모델을 정렬할 수 있도록 합니다. 또한, LLM이 파일을 이해할 수 있도록 하는 새로운 파일 분석 접근 방식을 제안합니다.

- **Performance Highlights**: 이 시스템은 사용자 지원 침투 테스트(in-browser assisted penetration testing)를 가능하게 하는 고유한 인프라 시스템을 강조하며, 이는 현대 사이버 보안 팀이 직면한 문제에 대한 강력한 솔루션을 제공하는 중요한 발전을 나타냅니다.



### From FDG to PSMA: A Hitchhiker's Guide to Multitracer, Multicenter Lesion Segmentation in PET/CT Imaging (https://arxiv.org/abs/2409.09478)
- **What's New**: 이번 논문은 autoPET III 챌린지에 대한 우리의 솔루션을 소개하며, nnU-Net 프레임워크를 활용한 멀티 트레이서(multi-tracer) 및 멀티 센터(multi-center) 일반화를 목표로 하고 있습니다. 이를 위해 CT, MR, PET 데이터셋에서 다중 알고리즘 사전 훈련(multi-modal pretraining) 및 부정합 데이터 증대(misalignment data augmentation) 기법을 사용했습니다.

- **Technical Details**: nnU-Net 프레임워크에 기반한 우리의 방법론은 ResEncL 아키텍처를 채택하였습니다. 우리는 3D 의료 영상에서 4000 에폭(epoch) 동안 훈련을 진행하며 각 데이터셋에 대해 별도의 세그멘테이션 헤드를 적용했습니다. 또한, organ supervision을 도입하여 다양한 기관을 식별하고, 레션 세그멘테이션의 정확성을 향상시켰습니다. 이 과정에서 배치 사이즈(batch size)는 24, 패치 사이즈(patch size)는 [192,192,192]로 설정하였습니다.

- **Performance Highlights**: 우리가 제안한 모델은 Dice 점수(Dice score)가 68.40에 달했으며, 이는 기본 nnU-Net(57.61) 및 ResEncL(65.31)에 비해 상당한 성능 향상을 보여줍니다. 또한, 잘못 분류된 양성(volume of false positives: FPvol: 7.82) 및 음성(volume of false negatives: FNvol: 10.35) 부피도 감소하여 효과성을 입증했습니다.



### TX-Gen: Multi-Objective Optimization for Sparse Counterfactual Explanations for Time-Series Classification (https://arxiv.org/abs/2409.09461)
Comments:
          Preprint, under review

- **What's New**: 이 논문은 TX-Gen이라는 새로운 알고리즘을 소개합니다. 이 알고리즘은 NSGA-II(Non-dominated Sorting Genetic Algorithm II)를 기반으로 하여 시간 시계열 분류를 위한 counterfactual(대안적 설명) 생성에 혁신적인 접근을 제안합니다.

- **Technical Details**: TX-Gen은 다목적 최적화(evolutionary multi-objective optimization)를 활용하여 sparse(희소성)와 valid(유효성)이면서도 원래 시간 시계열 데이터와의 유사성을 최소화하는 다양한 counterfactual을 찾습니다. 이 방법은 미리 정의된 가정에 의존하지 않고 flexible reference-guided mechanism(유연한 참고기반 메커니즘)을 통합합니다. 또한, 이 연구는 proximity(근접성), sparsity, validity 같은 주요 목표를 모두 균형 있게 고려합니다.

- **Performance Highlights**: TX-Gen은 벤치마크 데이터셋에서의 실험을 통해 기존의 방법보다 더 높은 품질의 counterfactual을 생성하며, 시간 시계열 모델의 투명성과 해석 가능성을 증가시킵니다. 특히, TX-Gen은 기존의 방법들과 비교하여 counterfactual의 sparsity(희소성), validity(유효성) 및 proximity(근접성) 면에서 우수성을 입증하였습니다.



### MulCPred: Learning Multi-modal Concepts for Explainable Pedestrian Action Prediction (https://arxiv.org/abs/2409.09446)
- **What's New**: 본 논문에서는 pedestrian action prediction을 향상시키기 위한 새로운 프레임워크인 MulCPred를 제안합니다. MulCPred는 다중 모달 (multi-modal) 개념을 기반으로 하는 예측 설명을 제공합니다. 이전 방법들이 가진 한계를 극복하며, pedestrian 행동 예측의 신뢰성을 높이는 데 기여합니다.

- **Technical Details**: MulCPred는 입력을 여러 개념에 투영하고, 각 개념은 입력과의 유사성을 나타내는 활성화 점수를 가집니다. 이 시스템은 작동의 한 부분으로서 channel-wise recalibration module을 사용하여 지역적 spatiotemporal 영역에 주목합니다. 또한, 다양한 패턴 학습을 장려하는 feature regularization loss를 포함하고 있습니다.

- **Performance Highlights**: MulCPred는 다양한 데이터셋과 작업에서 평가되어, qualitative 및 quantitative 결과 모두에서 pedestrian action prediction의 설명 가능성을 개선했으며, 성능 저하 없이도 이러한 효과를 달성했습니다. 더 나아가, 인식할 수 없는 개념을 제거함으로써.cross-dataset prediction 성능이 향상되었습니다.



### NBBOX: Noisy Bounding Box Improves Remote Sensing Object Detection (https://arxiv.org/abs/2409.09424)
- **What's New**: 이번 논문은 원거리 감지(remote sensing)에서 오브젝트 탐지(object detection) 성능 향상을 위해 bounding box의 변형(transformation) 방법을 제안합니다. 특히, 기존의 이미지 조정(image adjustments) 방식이 아닌 bounding box의 변화에 초점을 맞춘 NBBOX(Noise Injection into Bounding Box)라는 데이터 증강(data augmentation) 기법을 소개합니다.

- **Technical Details**: NBBOX 방법은 스케일링(scaling), 회전(rotation), 변환(translation)과 같은 세 가지 기하학적 변환을 사용하여 라벨 노이즈(label noise)가 있는 오리엔테이션(bounding box)에 노이즈를 추가하는 방식입니다. 이 방법은 기존의 감지 프레임워크에 쉽게 통합될 수 있으며, DOTA와 DIOR-R와 같은 데이터셋에서 실험이 진행되었습니다.

- **Performance Highlights**: 실험 결과, NBBOX 방법이 원거리 감지의 성능을 크게 향상시키는 것으로 나타났으며, 다른 최첨단(data augmentation strategies) 기법들과 비교하여 시간이 더 효율적임을 보여주었습니다.



### Distributed Clustering based on Distributional Kern (https://arxiv.org/abs/2409.09418)
- **What's New**: 본 논문은 분산 네트워크에서의 클러스터링을 위한 새로운 프레임워크인 KDC(Distributional Kernel 기반 분산 클러스터링)를 소개합니다. KDC는 초기 클러스터의 분포에 대한 유사성을 기준으로 최종 클러스터를 생성합니다. KDC는 중앙 집중형 클러스터링 결과와 동일한 결과를 보장하며, 클러스터 형태가 자유로운 분산 클러스터링 프레임워크입니다.

- **Technical Details**: KDC는 세 가지 주요 특성을 가지고 있습니다: 1) 모든 사이트의 통합 클러스터링 결과가 중앙 집중형 클러스터링 결과와 일치하도록 보장합니다. 2) 분산 모드에서의 최대 실행 시간 비용이 중앙 집중 모드보다 작습니다. 3) 임의의 형태와 밀도를 가진 클러스터를 발견하도록 설계되었습니다. 또한, 새로 개발된 Kernel Bounded Cluster Cores 알고리즘을 통해 KDC에서 기존 클러스터링 알고리즘 중 가장 우수한 성능을 자랑합니다.

- **Performance Highlights**: KDC의 분포 기반 클러스터링은 기존의 분산 클러스터링 방법보다 훨씬 나은 클러스터링 결과를 제공합니다. KDC는 대규모 데이터셋에 적용할 수 있는 일반적인 프레임워크로, 시간 복잡도를 최소화하면서 클러스터의 형태, 크기 및 밀도를 수용할 수 있는 가능성을 보여줍니다.



### Enhancing LLM Problem Solving with REAP: Reflection, Explicit Problem Deconstruction, and Advanced Prompting (https://arxiv.org/abs/2409.09415)
Comments:
          524 pages, 3 figures

- **What's New**: 이 논문은 LLMs (Large Language Models)의 문제 해결 능력을 향상시키기 위한 REAP (Reflection, Explicit Problem Deconstruction, and Advanced Prompting) 방법을 소개합니다. 이는 동적인 Context Generation Framework에 기반한 혁신적인 접근 방식입니다.

- **Technical Details**: REAP은 쿼리에 대한 반성을 유도하고, 이를 관리 가능한 구성 요소로 분해하며, 솔루션 과정을 향상시키기 위해 적절한 Context를 생성합니다. 논문에서는 REAP을 사용하는 평가를 통해 OpenAI의 여러 모델과 Google's Gemini 1.5 Pro, Claude 3.5 Sonnet 등 총 6개 최첨단 모델과 함께 제로샷(Zero-shot) 프롬프트와 REAP 향상 프롬프트를 비교하였습니다.

- **Performance Highlights**: REAP을 통해 o1-mini는 40.97%, GPT-4o는 66.26%, GPT-4o-mini는 112.93% 향상되었습니다. o1-preview의 강력한 기준 성능에 비해 최소한의 향상이 관찰되었으며, REAP은 GPT-4o-mini와 같은 저비용 모델로도 경쟁력 있는 결과를 도출했습니다. REAP은 모델의 출력 명확성을 향상시킴으로써 인간이 모델 결과의 추론을 이해하기 쉽게 하고 문제 식별 및 해결 과정을 단순화하는 데 기여합니다.



### Constructive Approach to Bidirectional Causation between Qualia Structure and Language Emergenc (https://arxiv.org/abs/2409.09413)
Comments:
          20 pages, 4 Figures

- **What's New**: 이 논문은 언어의 출현과 주관적 경험의 관계 구조, 즉 qualia 구조 간의 상호작용에 대한 새로운 관점을 제시합니다. 이는 내부 표현의 정렬 과정을 통해 발생하는 언어의 체계적인 구조화와 관련이 있습니다.

- **Technical Details**: 이 연구에서는 neural network 기반의 언어 모델이 구조적 내부 표현을 형성하며, 여러 양식의 언어 모델이 언어와 지각 정보를 공유할 수 있음을 보여줍니다. 또한, cognitive systems의 상징 출현을 다루며, collective predictive coding (CPC) 모델을 통해 언어의 출현과 qualia 구조간의 상호 의존성을 수학적으로 모델링합니다.

- **Performance Highlights**: 연구 결과, 언어는 단순한 의사소통의 도구가 아니라, 주관적 경험의 정렬을 가능하게 하는 메커니즘으로 작용함을 발견하였습니다. 이는 언어와 인간 인식 간의 복잡한 관계를 이해하는 데 기여할 것입니다.



### Label Convergence: Defining an Upper Performance Bound in Object Recognition through Contradictory Annotations (https://arxiv.org/abs/2409.09412)
- **What's New**: 본 연구에서는 'label convergence'라는 개념을 도입하여, 상충하는 테스트 주석으로 인한 성능 한계를 설명합니다. 이는 모델 정확도의 상한선을 정의하는 것으로, 95% 신뢰 구간에서 LVIS 데이터셋의 label convergence가 62.63-67.52 mAP@[0.5:0.95:0.05]로 추정됩니다.

- **Technical Details**: 연구에서는 다섯 개의 실제 데이터셋, 특히 LVIS 데이터셋을 분석하여 label convergence 현상을 조사합니다. 라벨이 금본위(gold standard)로 간주되었던 복잡한 데이터의 경우, 완벽한 기준 진실(ground truth)은 종종 달성할 수 없는 목표가 됩니다. 각 주석가의 변동성, 주석 관행의 차이 등이 label 불일치의 원인일 수 있습니다.

- **Performance Highlights**: 현재 최첨단(SOTA) 모델들은 LVIS 데이터셋의 label convergence 구간 상단에 있으며, 이는 현재 물체 탐지 문제를 해결하기 위한 모델의 용량이 충분함을 의미합니다. 향후 연구는 (1) 주석 잡음(label noise)을 고려한 평가 관행 수정, (2) 더 깨끗한 데이터 생성, (3) 주석 변동성을 조사하기 위한 다중 주석 데이터 포함에 집중해야 합니다.



### Real-world Adversarial Defense against Patch Attacks based on Diffusion Mod (https://arxiv.org/abs/2409.09406)
- **What's New**: DIFFender는 텍스트-유도 확산 모델을 활용하여 적대적 패치 공격에 대응하는 혁신적인 방어 프레임워크입니다. 또한 AAP(Adversarial Anomaly Perception) 현상을 발견하여 패치의 정확한 감지 및 위치 지정을 가능하게 합니다.

- **Technical Details**: DIFFender는 적대적 패치를 정밀하게 감지하기 위해 여러 دenoised(디노이즈된) 이미지 사이의 분포적 차이를 분석합니다. 이 프레임워크는 패치 로컬라이제이션(patch localization)과 복원(restoration) 작업을 통합된 확산 모델( Diffusion Model) 내에서 수행하며, 효율적인 few-shot prompt-tuning 알고리즘을 사용하여 모델 적응을 촉진합니다.

- **Performance Highlights**: DIFFender는 이미지 분류 및 얼굴 인식 작업에서 우수한 방어 성능을 보여주었으며, IR(Infrared, 적외선) 도메인으로의 확장 가능성을 통해 다양한 공격 방법론에 대한 일반화 성능이 뛰어납니다.



### AI-Driven Virtual Teacher for Enhanced Educational Efficiency: Leveraging Large Pretrain Models for Autonomous Error Analysis and Correction (https://arxiv.org/abs/2409.09403)
- **What's New**: 본 논문에서는 VATE(Virtual AI Teacher)라는 혁신적인 시스템을 제안하여, 학생들이 수학 문제를 해결할 때 발생하는 오류를 자율적으로 분석하고 수정하는 방안을 제공합니다. 이 시스템은 학생의 초안을 주요 분석 자료로 활용하여 학습 과정을 깊이 이해하고, 실시간 대화 기능을 통해 학생과의 상호작용을 원활하게 진행합니다.

- **Technical Details**: VATE 시스템은 고급 자연어 처리(NLP) 모델인 대형 언어 모델(LLMs)을 활용합니다. 이 시스템은 오류 분석을 위해 복잡한 프롬프트 엔지니어링을 통합하고, 오류 풀(error pool)을 유지하여 계산적 부담을 줄입니다. 시스템은 학생 초안의 이미지와 다중 모드 데이터를 사용하여 오류를 정확하게 로컬라이징 및 분석할 수 있는 기능을 갖추고 있습니다.

- **Performance Highlights**: 학생들의 응답에 대한 오류 분석 정확도가 78.3%에 달하며, 학생들의 학습 효율성이 눈에 띄게 향상되었습니다. 또한, 판매 직원 대상으로 실시한 만족도 조사에서 10점 만점에 8점 이상의 긍정적인 피드백을 얻어, 교육 방식의 혁신 가능성을 입증하였습니다.



### AMBER -- Advanced SegFormer for Multi-Band Image Segmentation: an application to Hyperspectral Imaging (https://arxiv.org/abs/2409.09386)
Comments:
          submitted to Neural Computing & Applications (Springer). Currently under review

- **What's New**: 이번 논문에서는 다중밴드 이미지 분할을 위한 고급 SegFormer 모델인 AMBER를 새롭게 소개합니다. AMBER는 삼차원 합성곱(three-dimensional convolutions)을 통합하여 하이퍼스펙트럴 데이터(hyperspectral data)를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: AMBER는 하이히어라키 Transformer 인코더와 가벼운 All-MLP 디코더로 구성됩니다. 이 인코더는 높은 해상도의 조잡한 특징과 낮은 해상도의 세부 특징을 생성합니다. 입력 이미지를 3x3x3 패치로 나누고, 이 패치를 인코더에 입력하여 다중 수준 특징을 얻습니다. 마지막 디코더 단계에서는 이 다중 레벨 특징을 융합하여 이차원 의미 분할 마스크를 생성합니다.

- **Performance Highlights**: AMBER는 Indian Pines, Pavia University, PRISMA 데이터셋에서 전통적인 CNN 기반 방법들과 비교하여 전체 정확도(Overall Accuracy), 카파 계수(Kappa coefficient), 평균 정확도(Average Accuracy)에서 월등한 성능을 보였으며, PRISMA 데이터셋에서는 최첨단 성과를 기록했습니다.



### LLM-Powered Ensemble Learning for Paper Source Tracing: A GPU-Free Approach (https://arxiv.org/abs/2409.09383)
- **What's New**: 이번 논문에서는 KDD CUP 2024의 논문 출처 추적 대회에서 3위에 오른 성과를 보고합니다. 대부분의 팀들이 BERT 및 ChatGLM과 같은 프리 트레인(pre-trained)된 신경망 언어 모델을 조정하여 문제를 해결한 반면, 우리는 클로즈드 소스(closed-source) 대형 언어 모델(LLMs)을 사용하여 문제를 해결합니다.

- **Technical Details**: 우리의 접근법은 문서의 텍스트와 관련 정보를 활용하여 직접적인 추론(reasoning)을 수행하는 프리 트레인된 LLM을 사용합니다. 우리는 논문 메타데이터, 인용 통계, 레퍼런스 메타데이터 및 맥락 키워드를 포함하는 특성(feature)을 추출하여 성능을 향상시킵니다. 이를 통해 GPU가 없는 환경에서도 효과적으로 문제를 해결할 수 있습니다.

- **Performance Highlights**: 우리는 KDD CUP 2024에서 3위를 차지하였으며, 이 과정에서 GPU를 사용하지 않고도 경쟁력을 갖춘 결과를 달성한 유일한 팀이었습니다. 이는 제한된 자원 속에서도 높은 성능을 발휘할 수 있는 우리의 방법의 유효성을 보여줍니다.



### Text Prompt is Not Enough: Sound Event Enhanced Prompt Adapter for Target Style Audio Generation (https://arxiv.org/abs/2409.09381)
Comments:
          5 pages, 2 figures, submitted to ICASSP 2025

- **What's New**: 이 논문에서는 기존의 텍스트 프롬프트 기반 오디오 생성 방법의 한계를 극복하기 위해 Sound Event Enhanced Prompt Adapter를 제안합니다. 이전 방법들이 세부적인 스타일을 잡아내지 못한 점을 개선하고, 데이터 간의 교차 주의를 통해 스타일 임베딩을 추출하여 적응형 스타일 제어를 가능하게 합니다.

- **Technical Details**: 본 논문은 두 가지 주요 방법으로 추가적인 prior knowledge를 오디오 생성에 통합합니다. 첫째, 컨트롤 조건을 이용해 생성된 오디오의 피치, 에너지 및 시간적 관계를 조작하는 방법. 둘째, 이미지와 비디오와 같은 다른 모달리티 인퍼를 사용하는 multi-modal 프롬프트를 통한 방법이 있습니다.

- **Performance Highlights**: 제안된 방법은 Fréchet Distance 26.94, KL Divergence 1.82라는 최신 성능을 기록하며, Tango, AudioLDM, AudioGen을 초월하는 결과를 보여줍니다. 생성된 오디오는 해당 오디오 참조와 높은 유사성을 보입니다.



### Prevailing Research Areas for Music AI in the Era of Foundation Models (https://arxiv.org/abs/2409.09378)
- **What's New**: 최근 음악 생성 AI의 발전으로 인해, generative music AI 응용 프로그램의 필요성이 증가하고 있습니다. 연구자들은 이러한 생성 모델의 기초적 표현, 음악 데이터셋의 현재 상태와 한계, 다양한 generative 모델의 평가 방법 등을 탐구하며, 사용 가능한 연구 방향과 Copyright(저작권)의 잠재적 문제에 대해 논의하고 있습니다.

- **Technical Details**: 음악 생성 AI는 멜로디, 화음, 리듬 및 다양한 스타일적 및 문화적 음악 특성을 포착하는 다차원적 이해를 제공합니다. 최근에는 text-conditioned audio-based generative music models가 주목받고 있으며, 이 모델들은 저작권 데이터에서 훈련되는 등 여러 가지 법적 쟁점이 있습니다. 현재 음악 태깅 모델과 XAI(Explainable AI)의 적용은 아직 초기 단계입니다.

- **Performance Highlights**: 지난 1년 동안, 일부 text-to-music 모델이 길이가 긴 음악을 생성할 수 있는 것으로 발전하였으나, 많은 모델은 여전히 짧은 조각만을 생성합니다. 또한, 효율적인 GPU 자원이 필요한 이 시스템들을 실제 음악 작곡에 활용하기 위해서는 디지털 오디오 워크스테이션과의 통합이 필수적입니다.



### LACOSTE: Exploiting stereo and temporal contexts for surgical instrument segmentation (https://arxiv.org/abs/2409.09360)
Comments:
          Preprint submitted to Medical Image Analysis

- **What's New**: 이 논문에서는 Location-Agnostic COntexts in Stereo and TEmporal images를 활용한 새로운 LACOSTE 모델을 제안하며, 수술 영상에서의 기구 분할 성능을 향상시킵니다. 기존의 단일 프레임 기반 접근법에서 벗어나, 시계열적 특성과 스테레오 특성을 고려한 쿼리 기반 세분화 접근 방식을 사용합니다.

- **Technical Details**: LACOSTE 모델은 다음 세 가지 성능 향상 모듈을 포함합니다: 1) disparity-guided feature propagation (DFP) 모듈로 깊이 인식 기능을 향상시키고, 2) stereo-temporal set classifier (STSCls)를 통해 시계열 및 스테레오 컨텍스트를 통합하여 예측의 정확성을 높이며, 3) location-agnostic classifier (LACls)를 사용하여 위치 편향을 완화합니다. 이러한 구성 요소들은 다양한 수술 비디오 데이터셋에서의 일반화 능력을 높이도록 설계되었습니다.

- **Performance Highlights**: LACOSTE 모델은 EndoVis Challenge의 두 가지 벤치마크와 실제의 radical prostatectomy 수술 데이터셋 GraSP를 포함한 세 개의 공개 수술 비디오 데이터셋에서 성능을 평가받았으며, 기존의 최신 기술(SOTA) 접근 방식들과 비교해 동등하거나 우수한 결과를 달성했습니다.



### Symbolic Regression with a Learned Concept Library (https://arxiv.org/abs/2409.09359)
Comments:
          preprint version; 10 pages

- **What's New**: 새로운 기법인 LaSR을 통해 심볼릭 회귀(SR) 문제를 해결하는 방식을 제시합니다. 이 방법은 대규모 언어 모델(LLM)을 이용하여 가설을 발견하고 발전시키는 과정에서 추상적인 개념의 라이브러리를 유도하는 데 중점을 둡니다.

- **Technical Details**: LaSR 알고리즘은 세 가지 단계로 이루어져 있습니다: (i) 개념 지향의 가설 진화, (ii) 높은 성능을 가진 기존 가설로부터의 패턴 추상화, (iii) 개념의 진화를 통한 더 간결하고 일반적인 형태로의 발전. 이러한 과정은 진화 탐색과 LLM의 불투명한 지식을 결합한 형태로 이루어집니다.

- **Performance Highlights**: LaSR은 피인먼 방정식과 여러 합성 벤치마크에서 기존의 최신 SR 접근 방식보다 우수한 성능을 발휘했습니다. 피인먼 방정식에서 LaSR은 66개의 목표 방정식을 발견했고, 가장 좋은 기존 접근 방식은 59개 방정식만 발견했습니다.



### Joint Semantic Knowledge Distillation and Masked Acoustic Modeling for Full-band Speech Restoration with Improved Intelligibility (https://arxiv.org/abs/2409.09357)
Comments:
          Demo link this https URL

- **What's New**: MaskSR2는 되었으며, 사전 훈련된 자가 감독 교사 모델의 의미 표현을 사용하여 MaskSR의 음성 인코더 구성 요소를 개선하여 이해 가능성을 크게 향상시켰습니다.

- **Technical Details**: MaskSR2는 음성 인코더에 의미 지식 증류(semantic knowledge distillation, KD)를 도입하여, 손상된 음성 신호의 STFT를 기반으로 목표 음성의 의미 표현을 예측합니다. 이 데이터는 전이 학습된 HuBERT 모델을 통해 추출됩니다. 이 모델은 비지도 학습(self-supervised learning)을 통해 배운 의미 패턴을 인코딩하며, MaskSR2는 이러한 의미 기능을 기반으로 음향 토큰을 예측합니다.

- **Performance Highlights**: MaskSR2는 MaskSR 대비 단어 오류율(Word Error Rate, WER)을 19%에서 38%까지 감소시켰으며, 음질 측면에서도 다양한 모델에 비해 우수한 성능을 보입니다.



### PeriGuru: A Peripheral Robotic Mobile App Operation Assistant based on GUI Image Understanding and Prompting with LLM (https://arxiv.org/abs/2409.09354)
- **What's New**: 이 논문에서는 PeriGuru라는 새로운 모바일 앱 운영 보조기구를 제안합니다. 이 장치와 시스템은 GUI 이미지 인식 및 Large Language Model (LLM)에 기반하여 모바일 앱의 작동을 지원합니다.

- **Technical Details**: PeriGuru는 컴퓨터 비전(Computer Vision, CV) 기술을 활용하여 GUI 스크린샷 이미지를 분석하고, 이를 바탕으로 의사결정을 위한 프롬프트를 생성합니다. 이 과정에서 LLM이 사용되며, 최종적으로 로봇 팔이 실행 작업을 수행합니다. 전체 프로세스는 인식 단계, 의사결정 단계, 행동 단계로 나뉘며, 각 단계에서 데이터와 정보를 가공하여 최적의 출력 행동을 도출합니다.

- **Performance Highlights**: PeriGuru는 실험에서 총 계획 성공률 89.71%와 실행 성공률 81.94%를 달성했습니다. 이러한 결과는 GUI 이미지 해석 및 프롬프트 설계를 통해 기존 방식에 비해 두 배 이상의 성능 상승을 보여주며, 시니어와 장애인을 포함한 사용자들이 모바일 앱을 더 쉽게 사용할 수 있도록 돕습니다.



### Egocentric Speaker Classification in Child-Adult Dyadic Interactions: From Sensing to Computational Modeling (https://arxiv.org/abs/2409.09340)
Comments:
          pre-print under review

- **What's New**: 이번 연구는 아동-성인 상호작용에서 wearable sensors를 사용하여 egocentric(1인칭) 음성 샘플을 수집하고, Ego4D 음성 샘플을 사전 학습하여 아동-성인 화자 분류 정확도를 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구에서는 TILES Audio Recorder (TAR)를 사용하여 아동-성인 dyadic interaction 중에 발생하는 음성을 수집하고, Ego4D 데이터셋에 대해 wav2vec 2.0 모델을 사용하여 pre-training을 수행합니다. 이를 통해 학습된 모델은 아동과 성인의 음성이 포함된 샘플을 분류하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, egocentric 음성 수집 및 사전 학습이 아동-성인 화자 분류 정확도를 향상시키는 잠재력을 보여줍니다. 연구는 2-7세 아동 10명을 대상으로 진행되었으며, 특히 ASD 아동 3명이 포함되었습니다.



### Wave-U-Mamba: An End-To-End Framework For High-Quality And Efficient Speech Super Resolution (https://arxiv.org/abs/2409.09337)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 논문에서는 Speech Super-Resolution (SSR) 기술을 개선하기 위해 새로운 접근 방식인 Wave-U-Mamba를 제안합니다. 기존의 접근 방식은 주로 log-mel 특징을 재구성한 후 vocoder를 통해 고해상도 음성을 생성하는 방식을 사용하였으나, 이로 인해 성능 저하가 발생할 수 있습니다. Wave-U-Mamba는 시간 영역에서 직접 SSR을 수행하며, 높은 성능과 빠른 생성 속도를 자랑합니다.

- **Technical Details**: Wave-U-Mamba는 GAN 기반의 프레임워크로, UR-net 구조를 활용하여 저해상도(Low-Resolution, LR) 음성을 입력받아 고해상도(High-Resolution, HR) 음성을 생성합니다. MambaBlock이라는 구조를 도입하여 긴 의존성을 효과적으로 학습하며, Multi-Period Discriminator (MPD)와 Multi-Scale Discriminator (MSD)를 이용한 적대적 훈련이 특징입니다. 모델은 LR 음성을 HR 음성으로 변환하는 과정에서 저해상도 신호의 고주파 성분을 집중적으로 추정하게 해 주는 잔차 연결(residual connection)을 특징으로 합니다.

- **Performance Highlights**: Wave-U-Mamba는 다양한 저해상도 샘플링 속도(8 kHz부터 24 kHz까지)에 걸쳐 가장 낮은 Log-Spectral Distance (LSD)를 기록하였고, Mean Opinion Score (MOS) 평가에서 자연스럽고 인간과 유사한 품질의 음성을 생성하였습니다. 또한, 단일 A100 GPU에서 베이스라인 모델보다 아홉 배 이상 빠른 속도로 고해상도 음성을 생성할 수 있었습니다.



### Efficient Fine-Tuning of Large Language Models for Automated Medical Documentation (https://arxiv.org/abs/2409.09324)
Comments:
          4 pages, 3 Figures, 3 Tables, This is a preprint version of the article. The final version will be published in the proceedings of the IEEE conference

- **What's New**: MediGen은 의료 대화에서 의료 보고서를 자동 생성하기 위해 설계된 정교한 대형 언어 모델(LLM)입니다. 이 연구는 의료의 효율성을 높이고 의사의 번아웃을 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: MediGen은 최신 방법론을 활용하여 공개 소스의 사전 훈련된 모델을 세밀하게 조정합니다. LLaMA3-8B 모델의 정교한 조정을 통해 임상 상호작용을 정확히 전사하고 요약하는 높은 정확도를 달성합니다. 모델의 성능은 ROUGE 스코어 58% 및 BERTScore-F1 72%로 나타났습니다.

- **Performance Highlights**: MediGen의 도입으로 의사들의 행정 업무 부담이 크게 줄어들 것으로 기대되며, 이는 의료 효율성 및 의사의 웰빙을 향상시킬 것으로 보입니다.



### The T05 System for The VoiceMOS Challenge 2024: Transfer Learning from Deep Image Classifier to Naturalness MOS Prediction of High-Quality Synthetic Speech (https://arxiv.org/abs/2409.09305)
Comments:
          Accepted by IEEE SLT 2024. Our MOS prediction system (UTMOSv2) is available in this https URL

- **What's New**: 2024년 VoiceMOS Challenge (VMC) Track 1을 위한 T05 시스템을 소개합니다. 이 시스템은 고품질 합성 음성의 자연성 평균 의견 점수(MOS)를 정확하게 예측하기 위해 설계되었습니다.

- **Technical Details**: T05 시스템은 사전 훈련된 자기 지도 학습(self-supervised learning, SSL) 기반의 음성 피처 추출기와 이미지 피처 추출기를 결합하여 합성 음성을 정확히 평가합니다. 두 개의 MOS 예측기를 각각 SSL 및 스펙트로그램 기반 피처로 훈련시키고, 두 피처를 융합하여 성능을 향상시킵니다. EfficientNetV2를 사용하여 스펙트로그램의 차이를 포착합니다.

- **Performance Highlights**: T05 시스템은 VMC 2024 Track 1에서 16개의 평가 메트릭 중 7개에서 1위를, 나머지 9개에서는 2위를 차지했습니다. 평가 결과, 두 피처를 융합함으로써 상관 기반 평가 메트릭의 개선이 있었고, 대규모 MOS 데이터셋을 사용하는 것이 MOS 예측 성능을 강화한다고 보고되었습니다.



### Matrix Profile for Anomaly Detection on Multidimensional Time Series (https://arxiv.org/abs/2409.09298)
- **What's New**: 이 논문은 다차원 시계열 데이터에서의 이상 감지(anomaly detection) 문제에 대해 탐구합니다. 특히 Matrix Profile(MP)의 확장성을 다루며, 다양한 학습 설정에 따라 MP를 활용하는 방법을 설명합니다.

- **Technical Details**: 다차원 시계열의 경우, pairwise distance matrix가 n x n x d 텐서로 표현되며, 이를 profile vector로 압축하는 다양한 전략을 분석합니다. 또한, k-nearest neighbors를 효율적으로 찾기 위한 방법을 제안합니다.

- **Performance Highlights**: 119개의 다차원 TSAD 데이터셋에 대한 실험에서 MP는 다른 19가지 방법들에 비해 모든 학습 설정에서 일관되게 높은 성능을 기록한 유일한 방법으로 확인되었습니다.



### Language Models "Grok" to Copy (https://arxiv.org/abs/2409.09281)
Comments:
          5 pages, 7 figures

- **What's New**: 이 연구는 언어 모델의 사전 훈련 동력을 조사하며, 모델의 텍스트 복사 능력이 기초적인 기술이라는 관점에서 실험을 통해 이 능력이 어떻게 발전하는지를 밝힙니다. 특히 Transformer 기반 언어 모델이 'grokking' 이라는 현상에 따라 복사 능력을 발전시킨다고 주장합니다.

- **Technical Details**: 연구팀은 12층 Llama 모델을 사용하여 400억 토큰(40 billion tokens)의 데이터를 가지고 훈련하였으며, 모델의 콘텍스트 복사 능력을 평가하기 위해 다양한 랜덤 토큰 하위 시퀀스를 입력으로 제공합니다. 이 과정에서 induction heads(복사를 담당하는 attention heads)가 어떻게 형성되는지, 그리고 이들이 얕은 층에서 깊은 층으로 이동하면서 발생하는지를 분석합니다.

- **Performance Highlights**: 연구 결과, 모델의 콘텍스트 복사 정확도는 훈련 손실이 안정화된 이후에도 갑자기 증가하며, 이를 통해 'grokking' 현상과의 관련성을 확인하였습니다. 또한, 레귤화 기법을 도입하여 콘텍스트 복사 능력을 향상시킬 수 있음을 보여주었습니다.



### An empirical evaluation of using ChatGPT to summarize disputes for recommending similar labor and employment cases in Chines (https://arxiv.org/abs/2409.09280)
Comments:
          14 pages, 5 figures, 2 tables, the 18th Int'l Workshop on Juris-Informatics (JURISIN 2024), associated with the 16th JSAI International Symposium on AI (JSAI-isAI 2024)

- **What's New**: 이번 연구에서는 노동 및 고용 소송의 유사 사례를 추천하기 위한 하이브리드 메커니즘을 제안합니다. 이 분류기는 법원에서 준비한 사례 간의 항목화된 분쟁을 기반으로 유사성을 결정합니다.

- **Technical Details**: 연구팀은 분쟁을 클러스터링하고, 분쟁 간의 코사인 유사성(cosine similarity)을 계산하여 이 결과를 분류 작업의 특성으로 사용합니다. 이전 시스템에서는 분쟁의 클러스터 정보만을 고려했으나, 이번 연구에서는 법원에서 준비한 정보를 대신하여 GPT-3.5와 GPT-4로 생성된 항목화된 분쟁을 사용했습니다.

- **Performance Highlights**: 실험 결과, GPT-4로 생성된 분쟁을 사용할 경우 더 나은 성과를 보였으며, ChatGPT가 생성한 분쟁을 사용할 때는 성능이 이전보다 떨어졌지만 결과는 만족할 만한 수준이었습니다. 향후 대형 언어 모델들이 실제로 유용해지기를 기대합니다.



### LabellessFace: Fair Metric Learning for Face Recognition without Attribute Labels (https://arxiv.org/abs/2409.09274)
- **What's New**: 고유한 인구 통계 그룹 레이블 없이 얼굴 인식의 인구 통계적 편향을 개선하는 새로운 프레임워크인 'LabellessFace'를 제안합니다.

- **Technical Details**: LabellessFace는 데이터셋 전반에서 특정 클래스에 대한 편애 정도를 측정하는 'class favoritism level'을 도입하며, 이를 바탕으로 'fair class margin penalty'를 적용하여 학습 매개변수를 동적으로 조정합니다. 이 방법은 주어진 데이터셋의 개개인에 대한 인증 정확도의 편향을 최소화하도록 학습을 촉진합니다.

- **Performance Highlights**: 종합 실험을 통해 LabellessFace가 기존 접근 방식과 비교하여 인증 정확도를 유지하면서 공정성을 효과적으로 향상시키는 것으로 나타났습니다.



### SafeEar: Content Privacy-Preserving Audio Deepfake Detection (https://arxiv.org/abs/2409.09272)
Comments:
          Accepted by ACM CCS 2024. Please cite this paper as "Xinfeng Li, Kai Li, Yifan Zheng, Chen Yan, Xiaoyu Ji, Wenyuan Xu. SafeEar: Content Privacy-Preserving Audio Deepfake Detection. In Proceedings of ACM Conference on Computer and Communications Security (CCS), 2024."

- **What's New**: 본 논문은 SafeEar라는 새로운 프레임워크를 제안하여, 오디오 딥페이크를 감지하면서도 음성 콘텐츠의 프라이버시를 보호할 수 있는 방법을 모색합니다. 이 접근법은 의미적 정보와 음향 정보를 분리하여, 감지 과정에서 의미적 내용을 노출하지 않도록 설계되었습니다.

- **Technical Details**: SafeEar는 신경 오디오 코덱(neural audio codec)을 사용하여 오디오 샘플에서 음향 정보(예: 프로소디(prosody)와 음색(timbre))만을 활용해 딥페이크 감지를 수행하는 새로운 분리 모델을 구현합니다. 이 시스템은 Transformer 기반의 탐지기를 사용하여, 다채로운 음향 패턴을 효과적으로 캡처할 수 있도록 최적의 멀티 헤드 자기 주의(MHSA)를 식별합니다. 또한, 다양한 통신 플랫폼에서 발생할 수 있는 압축 효과를 극복하기 위해 여러 대표적인 코덱을 통합하여 안전성과 신뢰성을 보장합니다.

- **Performance Highlights**: SafeEar는 다양한 딥페이크 기법 감지에서 2.02%의 동등 오류율(Equal Error Rate, EER)을 달성하며, 또한 음성 콘텐츠 보호에서도 93.93% 이상의 단어 오류율(Word Error Rate, WER)을 기록하여, 기존 시스템들과 비교하여 높은 성능을 보였습니다. 이 프레임워크는 영어 및 다른 네 가지 언어에서의 음성 콘텐츠를 성공적으로 보호할 수 있는 가능성을 보여줍니다.



### Guiding Vision-Language Model Selection for Visual Question-Answering Across Tasks, Domains, and Knowledge Types (https://arxiv.org/abs/2409.09269)
Comments:
          8 pages + references + 6 pages of Appendix

- **What's New**: 이 논문은 Visual Question-Answering (VQA) 작업을 위한 Vision-Language Models (VLMs)을 평가하는 포괄적인 프레임워크를 제시합니다. 또한, 다양한 작업 유형 및 응용 도메인과 함께 지식 유형으로 주석이 달린 새로운 데이터셋을 소개하며, 이를 통해 특정 작업 요구사항에 따라 VLMs를 선택할 수 있는 방법을 제공합니다.

- **Technical Details**: 논문에서는 VQA 작업의 평가를 위한 표준화된 프레임워크를 개발하며, task type, application domain, knowledge type의 세 가지 측면에서 VLMs를 비교합니다. 이를 위해 GoEval이라는 다중 양식(multi-modal) 평가 지표를 도입하고, 이는 GPT-4o를 기반으로 하여 사람의 평가와 높은 상관관계를 보입니다. 실험에서는 8개의 다양한 VLM 변종을 평가하여 각각의 강점과 약점을 분석합니다.

- **Performance Highlights**: 엑실리에서의 실험 결과는 Gemini-1.5-Pro 및 GPT-4o-mini와 같은 독점 모델이 일반적으로 다른 모델보다 우수한 성능을 보이지만, InternVL-2-8B 및 CogVLM-2-Llama-3-19B와 같은 오픈소스 모델도 특정 맥락에서 경쟁력을 갖추고 있다는 것을 보여줍니다. VLM들의 성능은 서로 높은 상관관계가 없으며, 각 모델은 특정 카테고리에서는 잘 수행하지만 다른 카테고리에서는 어려움을 겪는 경향이 있습니다.



### Operational Wind Speed Forecasts for Chile's Electric Power Sector Using a Hybrid ML Mod (https://arxiv.org/abs/2409.09263)
- **What's New**: 이번 연구는 칠레의 재생 가능 에너지 통합의 향상을 목표로 하여, 두 가지 맞춤형 기계 학습 모델을 결합한 하이브리드 바람 속도 예측 방법론을 소개합니다. 이 접근법은 단기(최대 48시간)와 중기(최대 10일) 예측을 위한 차별화된 기법을 적용하여, 전통적인 예측 방법을 개선합니다.

- **Technical Details**: 이 연구에서는 TiDE라는 다층 퍼셉트론(MLP) 기반 기계 학습 모델을 사용하여 단기 풍속 예측을 수행하고, GraphCast라는 그래프 신경망 모델을 통해 중기 예측을 진행합니다. TiDE 모델은 복잡한 비선형 종속성을 처리하는 데 최적화되어 있으며, GraphCast는 지구의 날씨 상태를 예측하기 위해 설계된 구조입니다. 이들은 비 재생에너지 출처와의 변동성을 고려한 예측을 가능하게 합니다.

- **Performance Highlights**: 하이브리드 방법론은 단기 예측에서 4-21%, 중기 예측에서 5-23%의 정확도 향상을 보여주고 있으며, 이는 칠레의 열 발전 시스템에 미치는 바람 발전의 영향을 직접적으로 감소시킬 수 있습니다. 이 결과는 재생 가능 에너지의 통합이 증가하는 가운데, 칠레 전력 시스템의 운영을 효율적으로 관리하는 데 기여할 것입니다.



### What Is Wrong with My Model? Identifying Systematic Problems with Semantic Data Slicing (https://arxiv.org/abs/2409.09261)
- **What's New**: 새로운 프레임워크 SemSlicer는 기계학습 모델의 오류를 분석하고 체계적인 문제를 파악하는 데 도움을 줍니다. 이는 기존 특징 없이도 의미론적 데이터 슬라이싱을 가능하게 하며, 사용자 정의 슬라이싱 기준에 따라 적합한 데이터 조각을 생성합니다.

- **Technical Details**: SemSlicer는 Large Language Models (LLMs)를 사용하여 데이터셋을 주석화하고 사용자 정의 슬라이싱 기준으로부터 데이터 슬라이스를 생성하는 기능을 갖추고 있습니다. 사용자는 슬라이스 기준, 데이터셋, 슬라이싱 구성 옵션을 제공하기만 하면 됩니다. SemSlicer는 이를 기반으로 슬라이싱 프롬프트를 자동 생성하고 최적화합니다.

- **Performance Highlights**: SemSlicer는 다양한 슬라이싱 기준에 대해 정확한 슬라이싱 기능을 생성할 수 있으며, 비용과 정확성 간의 유연한 타협이 가능합니다. 이를 통해 모델 평가, 버그 수정, 데이터 수집 등의 다양한 활동에 유용합니다.



### Unleash LLMs Potential for Recommendation by Coordinating Twin-Tower Dynamic Semantic Token Generator (https://arxiv.org/abs/2409.09253)
- **What's New**: 이번 연구에서는 Twin-Tower Dynamic Semantic Recommender (TTDS)를 제안하여 기존의 정적 인덱스 기반 추천 시스템의 한계를 극복하고자 합니다. TTDS는 동적 의미 인덱스 체계를 채택하여 시맨틱(semantic) 및 협업(collaborative) 지식 간의 정렬 문제를 해결하고, 사용자-아이템의 고차 상호작용 패턴을 활용합니다.

- **Technical Details**: TTDS는 항시 최적화되는 쌍둥이 타워 기반의 의미 토큰 생성기를 포함하며, 사용자를 위한 의미 인덱스와 아이템을 위한 의미 인덱스를 계층적으로 할당합니다. 이 시스템은 사전 학습된 대규모 언어 모델(LLM)을 기반으로 하여 사용자 행동의 역사적 데이터를 반영합니다. 또한, 다중 그레인 정렬(multi-grained alignment)을 용이하게 하도록 설계된 이중 변량 오토인코더(dual-modality variational auto-encoder)를 제안합니다.

- **Performance Highlights**: TTDS는 세 가지 공개 데이터 세트에서 실험을 수행한 결과, 추천 성능에서 Hit-Rate는 평균 19.41%, NDCG는 20.84% 향상되었습니다. 이는 기존의 가장 최신 벤치마크 방법들에 비해 큰 개선 효과를 보여줍니다.



### ETAGE: Enhanced Test Time Adaptation with Integrated Entropy and Gradient Norms for Robust Model Performanc (https://arxiv.org/abs/2409.09251)
- **What's New**: 본 논문에서는 Test Time Adaptation (TTA)에서의 새로운 접근법인 ETAGE를 제안합니다. ETAGE는 고전적인 방법들을 발전시켜 엔트로피 최소화와 기울기 정규화를 통합, 적응 과정에서 발생하는 불안정성을 줄이는 데 중점을 둡니다.

- **Technical Details**: ETAGE는 엔트로피 최소화와 기울기 정규화(Gradient Norms), Pseudo Label Probability Difference (PLPD)를 결합하여 적응 샘플을 선택하는 방법을 개선합니다. 이를 통해 노이즈에 대한 과적합을 피하고 모델의 안정성을 높입니다. ETAGE는 CIFAR-10-C와 CIFAR-100-C 데이터셋에서 검증되었습니다.

- **Performance Highlights**: ETAGE는 기존 TTA 기법들보다 더 높은 일반화성과 일관된 성능을 보여주며, 특히 어려운 분포 변동 환경에서 우수한 성능을 발휘했습니다. 경험적 결과를 통해 ETAGE의 성능 향상이 입증되었습니다.



### Robust Training of Neural Networks at Arbitrary Precision and Sparsity (https://arxiv.org/abs/2409.09245)
- **What's New**: 본 연구에서는 정량화(quantization)와 희소성(sparsification)으로 인한 비연속적인 작동이 역전파(backpropagation)에 미치는 영향을 극복하기 위한 새로운 솔루션을 제안합니다. 이를 위해, 훈련 과정에서 정량화와 희소성을 섭동(perturbation)으로 설정하고, 접근 방법으로는 여 Ridge 회귀(ridge regression)를 기반으로 합니다.

- **Technical Details**: 제안하는 방법은 세 가지 기본 단계로 구성됩니다: 1) 정량화를 위한 Affine Transform: 입력 신호를 크기 조절하는 초기 변환; 2) 섭동 주입: 정량화의 효과를 모델링하기 위한 임의의 섭동 δ를 신호에 주입; 3) 복원을 위한 Denoising Affine Transform: 양자화 노이즈를 완화하며 원래 신호를 효과적으로 복원합니다. 이 방법은 기존의 모델 아키텍처 및 훈련 레시피와 호환되어 많은 변화 없이 적용 가능하며, 다양한 효율적인 신경망의 개발을 가능하게 합니다.

- **Performance Highlights**: 제안하는 방법은 초저정밀 모델에 대한 최신 결과를 달성하며, 기존 아키텍처와 표준 훈련 레시피를 사용하여도 우수한 성능을 발휘합니다. 또한 희소 신경망(sparse neural networks)과 시간 이진 신경망(temporal binary neural networks)의 성공적인 훈련을 통해 인공 신경망과 생물학적 신경망 간의 간극을 좁힐 수 있는 가능성을 보여줍니다.



### Cross-Entropy Optimization for Hyperparameter Optimization in Stochastic Gradient-based Approaches to Train Deep Neural Networks (https://arxiv.org/abs/2409.09240)
Comments:
          6 pages, 2 figures

- **What's New**: 본 논문에서는 딥 뉴럴 네트워크(Deep Neural Networks) 훈련을 위한 확률적 경량 기반 접근법에서 하이퍼파라미터 최적화를 위한 교차 엔트로피 최적화 기법을 제시합니다.

- **Technical Details**: 하이퍼파라미터의 값은 모델의 성능, 특히 수렴 속도(convergence speed) 및 일반화 성능 메트릭(generalization performance metrics)에 큰 영향을 미칩니다. 본 논문에서는 기대 최대화(expectation maximization, EM) 프레임워크에서 이 방법에 대한 심도 있는 분석을 제공합니다. 제안된 교차 엔트로피 최적화 하이퍼파라미터 최적화 알고리즘(CEHPO)은 딥 러닝의 다른 최적화 문제에도 동일하게 적용 가능하다고 밝힙니다.

- **Performance Highlights**: 제안된 방법이 머신러닝 분야의 다양한 최적화 문제에 대한 새로운 시각을 제공하고 통찰력을 제시할 수 있을 것으로 기대합니다.



### Autoregressive + Chain of Thought (CoT) $\simeq$ Recurrent: Recurrence's Role in Language Models and a Revist of Recurrent Transformer (https://arxiv.org/abs/2409.09239)
- **What's New**: 이 논문은 Transformer 아키텍처의 한계를 극복할 수 있는 'Chain of Thought'(CoT) 프롬프트의 효용성을 연구하고, CoT가 재발적 구조의 장점을 모방하여 모델의 추론 능력을 향상시키는 방법을 탐구합니다.

- **Technical Details**: Transformer 구조는 반복적 연결을 제거하여 병렬 학습을 가능하게 하지만, 이는 Chomsky의 계산 계층에서 하위 성능으로 평가될 수 있습니다. 이 연구에서는 CoT가 그러한 한계를 극복할 수 있는 방법을 제시하며, 실험 결과를 통해 CoT가 autoregression과 recurrence의 다리를 잇는 역할을 한다고 주장합니다.

- **Performance Highlights**: CoT 접근 방식을 활용하면 Transformer 기반 언어 모델의 성능이 유의미하게 향상되며 기본적인 계산 문제(예: 곱셈이나 개수 세기)를 해결하는 데 있어 이전에는 불가능했던 작업들을 수행할 수 있도록 돕습니다.



### Contextual Evaluation of Large Language Models for Classifying Tropical and Infectious Diseases (https://arxiv.org/abs/2409.09201)
- **What's New**: 이 논문에서는 열대 및 감염병에 대한 질문 응답에서 대형 언어 모델(LLMs)의 성능을 평가하기 위해 TRINDs 데이터셋을 확장하고 다양한 컨텍스트 요소(예: 인구통계학적 정보, 위치, 위험 요소)의 영향을 심층적으로 분석합니다. 또한 LLM의 응답을 개선하기 위해 In-context prompt tuning을 적용하며, TRINDs-LM이라는 연구 도구를 개발하여 LLM의 성능에 미치는 컨텍스트의 영향을 탐색합니다.

- **Technical Details**: 연구진은 원래 TRINDs 데이터셋을 11,719개의 질문으로 확장하였으며, 이 데이터셋은 여러 인구통계학적 요인, 장소, 성별, 인종 등의 다양한 변형이 포함되어 있습니다. Gemini Ultra와 MedLM Medium라는 두 가지 LLM을 기반 모델로 사용하여 성능을 비교하였으며, 5회의 반복 실험을 통해 정확성을 평가하였습니다. 또한, LLM-Consumer 쿼리를 생성하여 임상적 톤 대신 일반 소비자의 관점에서 질문을 재구성하였습니다.

- **Performance Highlights**: 실험 결과, 컨텍스트 정보(예: 인구통계학적 요인, 위치 등)는 LLM의 응답 정확도를 높이는 데 중요한 역할을 하며, LLMs는 보다 구체적인 정보와 위험 요소를 포함하는 쿼리에 대해 더 나은 성과를 보였습니다. 연구진은 LLM이 제공하는 결과와 인적 전문가의 기준 점수를 비교하여 LLM의 응답이 여전히 불완전할 수 있음을 보여주었습니다.



### Hierarchical Hypercomplex Network for Multimodal Emotion Recognition (https://arxiv.org/abs/2409.09194)
Comments:
          The paper has been accepted at MLSP 2024

- **What's New**: 이번 연구에서는 다채로운 생리적 신호를 기반으로 하는 감정 인식 시스템을 위한 새로운 하이퍼복합 네트워크 모델을 제안합니다. 특히, 인코더와 융합 모듈 모두 하이퍼복합 영역에서 작동하도록 하여, 각 신호의 채널 간 관계와 모달리티 간의 상관 관계를 효과적으로 학습합니다.

- **Technical Details**: 제안된 Hierarchical Hypercomplex (H2) 모델은 계층 구조로 구성되어 있으며, 여기서 각 인코더는 입력 신호의 차원 수에 따라 하이퍼파라미터를 설정하여 작동합니다. 인코더는 parameterized hypercomplex convolutions (PHCs)을 통해 각 모달리티의 채널 간 관계를 학습하고, 융합 모듈은 parameterized hypercomplex multiplications (PHMs)을 통해 모달리티 간의 관계를 학습합니다.

- **Performance Highlights**: MAHNOB-HCI 데이터셋에서 아우로살(activation) 및 발란스(valence) 분류 성능이 40.20% 및 57.11% 향상된 F1-score을 각각 0.557 및 0.685로 달성하였습니다.



### ProcessTBench: An LLM Plan Generation Dataset for Process Mining (https://arxiv.org/abs/2409.09191)
Comments:
          6 pages, 4 figures, dataset available at this https URL

- **What's New**: ProcessTBench 데이터셋은 기존의 TaskBench 데이터셋을 기반으로 하여 LLMs(대형 언어 모델)의 계획 생성 능력을 평가하기 위한 새로운 도전적 환경을 제공합니다. 특히, 병렬 작업 수행과 다른 조건 및 형식에서 동일한 프로세스 실행을 요구하는 상황에서의 LLMs의 행동을 이해하는 데 초점을 맞추고 있습니다.

- **Technical Details**: ProcessTBench 데이터셋은 TaskBench의 가장 도전적인 하위 세트를 기반으로 하며, 532개의 기본 쿼리가 포함되어 있습니다. 각 쿼리는 5~6개로 패러프레이즈되었으며, 각 쿼리는 평균 4.08개의 솔루션 계획을 제공합니다. 이 데이터셋은 다국어 지원을 포함하여 LLM의 계획 생성 프레임워크와 포괄적인 계획 데이터셋을 제공합니다.

- **Performance Highlights**: ProcessTBench 데이터셋은 LLM이 복잡한 작업을 수행할 때 생성하는 계획의 다양성과 신뢰성을 평가할 수 있는 기회를 제공합니다. 이를 통해 LLMs의 쿼리 해석, 도구 사용, 행동 시퀀스 생성의 효율성과 정확성을 평가할 수 있으며, LLMs가 다국어 쿼리와 패러프레이즈된 쿼리를 처리하는 능력도 시험할 수 있습니다.



### Incorporation of Verifier Functionality in the Software for Operations and Network Attack Results Review and the Autonomous Penetration Testing System (https://arxiv.org/abs/2409.09174)
Comments:
          The U.S. federal sponsor has requested that we not include funding acknowledgement for this publication

- **What's New**: 이번 논문에서는 SONARR(Operations and Network Attack Results Review) 시스템에 검증자(verifiers)를 추가하여 현실 세계의 조건을 확인하고 네트워크 사실(facts)을 업데이트하는 방법을 제안했습니다. 이는 SONARR의 운영이 실제 시스템과 일치하도록 보장하는 일관된 방법을 제공합니다.

- **Technical Details**: SONARR와 APTS(Autonomous Penetration Testing System)는 디지털 트윈 네트워크를 사용하여 실제 엔티티를 표현합니다. 그러나 사실 값이 자주 변경되는 경우도 있어 SONARR와 APTS의 객체들이 현실 세계의 카운터파트를 일관되게 표현하는 데 어려움이 있었습니다. 검증자는 임의의 스크립트와 동적 인수를 SONARR의 일반 작업에 추가할 수 있도록 하여 유연성과 일관성을 제공합니다.

- **Performance Highlights**: 검증자의 도입으로 SONARR는 실행 환경에서 사실 값을 검색하고 네트워크를 업데이트할 수 있어 소프트웨어의 출력이 더욱 신뢰성 있게 됩니다. 이를 통해 운영 결과가 평가되는 실제 시스템과 더 잘 일치하게 됩니다.



### Phikon-v2, A large and public feature extractor for biomarker prediction (https://arxiv.org/abs/2409.09173)
- **What's New**: 이 논문에서는 100개 이상의 공개 코호트에서 수집한 4억 6천만 개의 병리 타일로 구성된 다양한 데이터 세트를 이용하여, DINOv2를 사용한 대규모 자가 지도 학습( Self-supervised learning ) 기반의 비전 트랜스포머 모델인 Phikon-v2를 훈련하고 이 모델을 공개했습니다. Phikon-v2는 이전 모델인 Phikon을 능가하며, 소유 데이터로 훈련된 다른 병리학 기초 모델들과 비슷한 성능을 보입니다.

- **Technical Details**: Phikon-v2는 460 million (억 6천만) 개의 병리 타일을 기반으로 트레이닝된 ViT-L (Vision Transformer Large) 모델로, DINOv2에 의해 학습되었습니다. 이 모델은 WSI (Whole Slide Images) 레벨의 바이오마커 예측과 질병 분류 작업에서 경쟁력 있는 성능을 보여줍니다. 또한 14개의 다양한 병리학 feature extractor와 비교하여 그 평가가 가장 포괄적입니다.

- **Performance Highlights**: 모델의 카운터파트인 Phikon (ViT-B)와 Phikon-v2 (ViT-L)는도 8가지 슬라이드 레벨 작업에서 통계적으로 유의미하게 (+1.75 AUC 증가) 향상된 성능을 보였습니다. 최신 기초 모델들이 임상 배치 대비 한계를 보일 수 있지만, 이들은 더 전문화되고 비용 효율적인 병리학 인코더 개발의 기초를 제공합니다.



### The Challenges of Effective AGM Belief Contraction (https://arxiv.org/abs/2409.09171)
Comments:
          20 pages, 4 figures

- **What's New**: AGM (Alchourrón, Gärdenfors, Makinson) 패러다임의 신뢰 변화(computability)를 비국한 논리(non-finitary logics)에서 탐구함. 이 논문은 비국한 논리에서의 AGM 축소(AGM contraction) 함수들이 무계산 가능(uncomputable)하다는 흥미로운 결과를 보여줌. 또한, 선형 시간 논리(Linear Temporal Logic, LTL)에서 계산 가능한 무한 클래스의 AGM 축소 함수를 확인.

- **Technical Details**: AGM 패러다임은 신뢰의 일관성을 유지하기 위해 정보 수집 시 최소한의 변경을 요구하는 원칙(principle of minimal change)을 따름. 저자들은 Büchi automata를 사용하여 LTL의 에피스테믹 상태를 표현하고, 이 공간 내에서 계산 가능한 축소 함수(families of contraction functions)를 구축함을 증명함. 무한한 클래스의 실용적인 computable AGM contraction 함수를 식별했으며, Büchi-Mealy automata로 표현된 에피스테믹 선호 관계에 기반하여 computability를 유도.

- **Performance Highlights**: 이 연구는 비국한 논리에서의 신뢰 변화의 계산적 측면을 탐구하며, 기존의 고전적 논리와는 다른 분석적 접근을 제시함. LTL 기반의 함수를 통해 광범위한 적용을 목표로 하며, 소프트웨어 및 하드웨어 시스템의 명세와 검증을 포함한 다양한 분야에서 효용성을 증대시킬 가능성이 있음.



### Curricula for Learning Robust Policies over Factored State Representations in Changing Environments (https://arxiv.org/abs/2409.09169)
Comments:
          17th European Workshop on Reinforcement Learning (EWRL 2024)

- **What's New**: 이번 연구에서는 강화 학습(Reinforcement Learning) 에이전트가 비예측적이고 동적인 환경에서 견고성을 유지하고 적응할 수 있는 정책을 개발하기 위한 새로운 접근 방식을 제안합니다. 특히, 상태 공간과 행동 공간을 분해하는 Factored representations를 통해 정책 학습에서 일반화(generalization) 및 샘플 효율(sample efficiency)을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 사전학습 과정에서 일반적으로 Markov Decision Process (MDP)를 활용하여 에이전트의 행동을 모델링합니다. MDP는 상태 집합(𝒮), 행동 집합(𝒜), 전이 확률 함수(P), 보상 함수(R), 할인 계수(γ)로 정의됩니다. 연구에서 사용된 Curriculum Learning 전략은 다양한 서브 태스크를 순차적으로 배치하여 학습 효율성을 높이고, 복잡한 환경에서 에이전트가 더욱 견고한 정책을 학습하도록 돕는 방향으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, Factored representations를 활용한 커리큘럼 학습이 없을 경우, 간단한 커리큘럼만으로는 견고한 정책을 학습하기에 불충분함을 발견했습니다. 그러나, 랜덤한 변화를 주는 커리큘럼이나 다양한 사례를 섞는 커리큘럼을 통해 정책의 견고성을 크게 향상시킬 수 있음을 확인했습니다. 이는 강화 학습의 실제 환경에서 정책의 일반화 및 적응 능력을 개선하기 위한 실용적인 통찰을 제공합니다.



### Neural Message Passing Induced by Energy-Constrained Diffusion (https://arxiv.org/abs/2409.09111)
Comments:
          Extended version from DIFFormer paper in ICLR2023

- **What's New**: 본 논문에서는 메시지 패싱 신경망(MPNNs)의 메커니즘을 이해하고 새로운 아키텍처 설계를 탐색하기 위한 원리적이며 해석 가능한 프레임워크로 에너지 제약 확산 모델을 제안합니다. 이 모델은 물리 시스템에 영감을 받아 매니폴드에서의 확산의 귀납적 편향과 에너지 최소화를 위한 레이어별 제약 조건을 결합합니다.

- **Technical Details**: 에너지 제약 확산 모델은 관측 데이터와 잠재 구조가 결합된 리만 매니폴드에서의 위치로 구성된 확산 PDE 방정식을 기반으로 정의됩니다. 이 모델은 레이어 종속의 쌍 연결 가중치를 통해 노드 쌍 간의 특성 전파를 가능하게 하며, 에너지 함수는 진화 방향에 대한 정규화를 강제합니다. 우리의 분석은 확산 프로세스의 유한 차분 반복과 연관된 에너지 최소화 다이나믹스 간의 본질적인 동등성을 보여줍니다.

- **Performance Highlights**: 다양한 데이터셋을 대상으로 한 실험에서 확산 기반 변환기(DIFFormer) 모델이 그래프 기반 예측 태스크(동질적인 그래프, 이질적인 그래프, 대규모 그래프, 불완전한 관측 그래프 등) 및 일반적인 예측 태스크(이미지, 텍스트, 물리 입자 등)에서 뛰어난 성능을 기록했습니다.



### Recent Trends in Modelling the Continuous Time Series using Deep Learning: A Survey (https://arxiv.org/abs/2409.09106)
- **What's New**: 이번 논문은 지속적인 시간 시리즈(Continuous-time series)의 최신 동향과 이 모델이 다양한 실제 애플리케이션에서 어떻게 사용되는지를 중점적으로 다뤘습니다.

- **Technical Details**: 이 논문은 수많은 현대 애플리케이션 분야에서 시간 시리즈의 역할을 강조하며, 특히 데이터의 동적 시스템을 미분 방정식(Differential equation)으로 모델링하는 것이 얼마나 어려운지를 설명합니다. 여러 신경망(Neural network) 모델을 통해 지속적인 시간 시리즈를 모델링하기 위한 여러 접근 방식이 논의됩니다.

- **Performance Highlights**: 최근의 심층 학습(Deep learning) 모델들이 지속적인 시간 시리즈 모델링의 다양한 문제들을 해결하는 데 기여하고 있지만, 여전히 다양한 속성, 행동, 단계의 지속 시간, 에너지 및 데이터 샘플링 속도(Sampling rate) 등의 제한이 존재합니다.



### Inf-MLLM: Efficient Streaming Inference of Multimodal Large Language Models on a Single GPU (https://arxiv.org/abs/2409.09086)
- **What's New**: 이 논문에서는 MLLMs(다중모드 대형 언어 모델)의 효율적인 추론 프레임워크인 Inf-MLLM을 소개합니다. Inf-MLLM은 무한한 컨텍스트에서 단일 GPU에서 MLLMs의 스트리밍 추론을 가능하게 합니다.

- **Technical Details**: Inf-MLLM은 'attention saddles'라는 새로운 어텐션 패턴을 기반으로 하여 동적으로 최근 토큰과 관련 토큰을 캐시합니다. 또한, attention bias를 제안하여 MLLMs이 장기 종속성을 잡을 수 있도록 합니다. 이 프레임워크는 계산 복잡성을 줄이고 메모리 사용량을 감소시킵니다.

- **Performance Highlights**: Inf-MLLM은 4M 토큰의 긴 텍스트와 1시간 분량의 비디오를 가진 다중 라운드 대화에서 안정적인 성능을 달성하며, StreamingLLM보다 우수한 스트리밍 추론 품질과 H2O에 비해 2배 빠른 속도를 자랑합니다.



### D3-GNN: Dynamic Distributed Dataflow for Streaming Graph Neural Networks (https://arxiv.org/abs/2409.09079)
Comments:
          14 pages, 7 figures, published at VLDB'24

- **What's New**: D3-GNN은 온라인 쿼리 설정에서 실시간 그래프 업데이트를 처리하기 위해 설계된 첫 번째 분산 하이브리드 병렬 스트리밍 GNN 시스템입니다. 이는 동적 그래프 상태를 지속적으로 캡처하고 노드 표현을 업데이트하는 데 있어서 효율성을 높이고 저지연을 유지합니다.

- **Technical Details**: D3-GNN은 스트리밍 GNN 집계기와 함께 전개된 분산 계산 그래프 아키텍처를 활용하여 연속적인 그래프 업데이트를 처리합니다. 또한 이 시스템은 데이터 관리와 알고리즘, 시스템 문제를 해결하여 노드 표현의 최신 상태 유지 및 최적의 메모리와 처리량, 불연속성 허용을 달성합니다.

- **Performance Highlights**: D3-GNN은 대규모 그래프 스트림에서 높은 효율성과 확장성을 달성하였으며, DGL에 비해 스트리밍 작업에 대해 약 76배의 처리량 향상을 기록했습니다. 윈도우 개선을 통해 실행 시간을 약 10배 단축시키고 메시지 볼륨을 최대 15배 줄이는 성과를 보였습니다.



### Fair Reinforcement Learning Algorithm for PV Active Control in LV Distribution Networks (https://arxiv.org/abs/2409.09074)
- **What's New**: 이 논문에서는 PV 패널로부터 발생하는 전압 문제를 해결하기 위해 리인포스먼트 학습 기법을 제안합니다. 특히 고객 간의 활성 전력 제한에 대한 공정성을 고려하여 전압 문제를 효과적으로 조절할 수 있는 방법을 다룹니다.

- **Technical Details**: 연구는 분산 전력 네트워크에서 PV 패널을 활용하여 발생하는 전압 문제를 해결합니다. 제안된 방법은 활성 전력 및 반응 전력의 출력을 관리하는 심층 리인포스먼트 학습(Deep Reinforcement Learning) 기술을 사용합니다. 이 접근 방식은 네트워크 데이터 요구 사항이 제한적이면서도 공정하고 최적의 솔루션을 제공할 수 있습니다.

- **Performance Highlights**: 실험을 통해 제안된 방식의 효율성을 검증하였으며, 고객 간 공정성을 유지하면서도 전압을 효과적으로 관리할 수 있는 능력을 입증하였습니다. 이는 향후 PV 패널 설치 장려에 긍정적인 영향을 미칠 것으로 기대됩니다.



### Joint Model Assignment and Resource Allocation for Cost-Effective Mobile Generative Services (https://arxiv.org/abs/2409.09072)
- **What's New**: 이 논문은 사용자 요청에 따라 생성 모델의 계산 작업을 에지 서버에 적절히 할당하는 에지 기반 인공지능 생성 콘텐츠(AIGC) 서비스 시스템을 설계하여 전체 사용자 경험을 향상시키고 콘텐츠 생성 지연을 줄이는 방법을 제안합니다.

- **Technical Details**: 제안된 시스템은 사용자 생성 요청을 처리하기 위해 동적으로 적절한 모델을 할당하고 각 카테고리의 프롬프트에서 특징에 기반하여 컴퓨팅 자원을 할당합니다. 시스템의 핵심은 카테고리 레이블에 따라 생성된 콘텐츠의 품질 점수를 추정하는 확률론적 모델 할당 접근 방식입니다. 또한, 생성 단계와 자원 할당을 적응적으로 구성할 수 있는 휴리스틱 알고리즘을 도입하였습니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 이 시스템은 벤치마크 대비 생성된 콘텐츠의 품질을 최대 4.7% 향상시키고 응답 지연을 최대 39.1% 감소시킬 수 있습니다.



### ELMS: Elasticized Large Language Models On Mobile Devices (https://arxiv.org/abs/2409.09071)
Comments:
          Technical Report

- **What's New**: 이번 연구에서 우리는 ELMS라는 새로운 온디바이스 LLM(대형 언어 모델) 서비스를 제안합니다. ELMS는 LLMaaS(LLM-as-a-Service)에서의 유연성을 모델과 프롬프트의 차원 모두에 제공하는 시스템입니다. 기존의 LLM들이 다양한 서비스 수준 목표(SLO)에 맞추기 어려운 문제를 해결하려는 노력의 일환으로, 이 연구는 기존 LLM의 한계에 대처하고 사용자 경험을 향상시키고자 합니다.

- **Technical Details**: ELMS는 다음과 같은 혁신적인 기술을 사용합니다: 
1. 한 번의 뉴론 재배치(one-time neuron reordering) 기술로 Transformer 모델 내의 자연스러운 순열 일관성을 활용하여 고품질의 탄력적인 서브 모델을 생성합니다. 
2. 듀얼 헤드 컴팩트 언어 모델(dual-head compact language model)을 통해 프롬프트를 효율적으로 개선하고 모델과 프롬프트 간의 탄력적 적응을 조정합니다.

- **Performance Highlights**: ELMS는 여러 상용 스마트폰에서 실행되었으며, 독립적인 NLP 및 모바일 에이전트 데이터셋과 합성된 엔드 투 엔드 추적을 사용하여 평가되었습니다. 다양한 SLO에서 ELMS는 절대 정확성에서 4개의 강력한 기준 모델을 평균 16.83% 및 11.04% 초과 달성하였으며, 전환 오버헤드가 1% 미만으로 발생하였습니다. ELMS는 비슷한 메모리 사용량을 유지하면서 100시간 이하의 오프라인 GPU 시간을 사용했습니다.



### Temporal Many-valued Conditional Logics: a Preliminary Repor (https://arxiv.org/abs/2409.09069)
Comments:
          8 pages

- **What's New**: 이 논문에서는 다가치 (many-valued) 시간 조건 논리를 제안합니다. 이 논리는 전형성 (typicality)을 가진 다가치 논리를 출발점으로 하여 선형 시간 템포럴 논리 (Linear Time Temporal Logic, LTL)의 시간 연산자를 확장하여 시스템의 동적 특성을 포착하도록 발전되었습니다.

- **Technical Details**: 다양한 진리 정도 집합 𝗗 (D) 위에서 정의된 다가치 전제 논리로서, 논리 연결사 (logical connectives)인 ∧, ∨, ¬ 및 →를 사용하여 제안됩니다. 조건문은 물질적 함의를 기반으로 형식화되며, 전형성 연산자를 포함하여 점진적인 주장을 정의합니다. 이러한 관점에서, 조건문은 점진적 주장을 다루기 위한 시간 조건 의미론을 고려합니다.

- **Performance Highlights**: 이 새로운 다가치 논리는 시스템의 점진적 주장을 분석하는 데 효과적이며, 차별화된 시간 속성에 대한 사고를 촉진하여 설명 능력을 강화합니다. 논문은 이론적 담론을 기반으로 한 응용 가능성을 모색하며, 고급 지식 기반의 동적 구조를 명확히 밝힐 수 있는 가능성을 제시합니다.



### TS-EoH: An Edge Server Task Scheduling Algorithm Based on Evolution of Heuristic (https://arxiv.org/abs/2409.09063)
- **What's New**: 이 논문은 진화 컴퓨팅(EC) 이론과 휴리스틱 알고리즘을 기반으로 한 새로운 태스크 스케줄링 방법을 제안합니다. 기존의 방법들이 여러 최적화 목표를 효과적으로 균형 잡지 못하는 문제를 해결하고자 하며, 대형 언어 모델(LLMs) 서비스 활용을 통해 진화 과정에서 다양한 스케줄링 방식 평가를 수행합니다.

- **Technical Details**: 제안된 태스크 스케줄링 방법(TE-SoE)은 진화 알고리즘과 휴리스틱 알고리즘의 조합으로 이루어져 있으며, 태스크 요청을 태스크 시퀀스로 모델링합니다. EoH 프레임워크를 사용하여 태스크 스케줄링 문제를 조합 최적화 문제로 변환하고, LLMs에서 생성된 스코어링 방법을 통해 각 태스크 시퀀스를 평가합니다. 이 과정은 초기화, 돌연변이 및 진화 전략으로 구성됩니다. 

- **Performance Highlights**: 실험 결과, 제안된 알고리즘이 기존의 휴리스틱 및 전통적인 강화 학습 방법을 초월하는 성능을 보여주었습니다. 또한 다양한 휴리스틱 전략의 효과를 조사하고, 여러 LLM 서비스에서의 진화 결과를 비교하였습니다.



### Redefining Data-Centric Design: A New Approach with a Domain Model and Core Data Ontology for Computational Systems (https://arxiv.org/abs/2409.09058)
- **What's New**: 이 논문은 데이터 중심(data-centric) 패러다임을 통해 컴퓨팅 시스템을 설계하는 혁신적인 접근 방식을 제시합니다. 기존의 노드 중심(node-centric) 프레임워크에서 벗어나 객체(objects), 사건(events), 개념(concepts), 행동(actions)을 포함하는 다중 모드(multimodal) 접근 방식을 채택합니다.

- **Technical Details**: 제안된 모델은 핵심 요소에 기반하여 포괄적인 온톨로지(ontology)를 구축하고, 이를 통해 분산 생태계에 걸쳐 데이터의 의미적 일관성(semantic consistency) 및 안전한 데이터 처리를 촉진합니다. OWL 2 온톨로지로서의 이 모델의 구현을 탐구하며, 관련 응용 프로그램(application) 및 확장성(scalability) 및 연구의 향후 방향에 대해서도 논의합니다.

- **Performance Highlights**: 이 연구는 시스템 설계자(system designers)와 데이터 아키텍트(data architects)가 보다 안전하고, 상호 운용 가능(interoperable), 확장 가능한 데이터 시스템을 개발하기 위한 기초 자료로 활용될 수 있도록 합니다.



### Evaluating the Performance of Large Language Models in Competitive Programming: A Multi-Year, Multi-Grade Analysis (https://arxiv.org/abs/2409.09054)
Comments:
          7 pages, Inista 2024

- **What's New**: 이 연구는 루마니아 정보 올림피아드에서의 경쟁적 프로그래밍 문제를 해결하는 대규모 언어 모델(LLMs)의 성능을 조사합니다. 연구는 2002년부터 2023년까지의 304개의 과제를 분석하여, LLM들이 문제를 해결하는 데 있어 성능 차이를 이해하는 데 중점을 두었습니다.

- **Technical Details**: 데이터셋은 C++와 Python 언어로 작성된 문제들로 구성되며, LLM 성능은 여러 모델(GPT-4, CodeLlama, RoMistral 등)을 기반으로 평가되었습니다. 모델은 주어진 문제에 대해 여러 번의 시도 및 피드백 과정을 통해 성능을 테스트했으며, 특히 GPT-4가 더 낮은 등급에서 강한 성능을 보였습니다.

- **Performance Highlights**: 연구 결과, LLM의 성능이 등급과 문제 유형에 따라 큰 차이를 보임을 확인했습니다. GPT-4는 중학생용 교육 도구로서의 가능성을 보여주었고, 코드 품질 및 스타일에서도 모델 간 차이가 관찰되었습니다.



### Deep learning-based classification of breast cancer molecular subtypes from H&E whole-slide images (https://arxiv.org/abs/2409.09053)
Comments:
          16 pages, 5 figures (+4 supplementary figures), 4 tables

- **What's New**: 이번 연구에서는 H&E 염색된 전체 슬라이드 이미지(Whole Slide Images, WSI)를 사용하여 유방암의 분자 아형을 예측할 수 있는 가능성을 조사했습니다. 이는 기존의 면역조직화학(immunohistochemistry, IHC) 및 유전자 발현 프로파일링(gene expression profiling)과 같은 전통적인 방법들에 대한 대안으로 제시됩니다.

- **Technical Details**: 연구팀은 1,433개의 유방암 WSI를 사용하여 두 단계의 파이프라인을 구축했습니다. 첫 번째 단계에서는 종양(tile)과 비종양(tile)을 분류하여 종양 영역만 사용하였고, 두 번째 단계에서는 One-vs-Rest(OvR) 전략을 employed하여 네 개의 바이너리 OvR 분류기를 훈련시키고, 그 결과를 eXtreme Gradient Boosting(XGBoost) 모델을 사용해 집계했습니다.

- **Performance Highlights**: 이 파이프라인은 221개의 검증용 WSI에서 테스트되었으며, 종양 탐지에 대해 전체 macro F1 점수 0.95를, 분자 아형 분류에서는 0.73을 기록했습니다. 이러한 결과들은 감독형 심층 학습(supervised deep learning) 모델이 유방암의 분자 아형 분류를 위한 보조 도구로 활용될 가능성을 나타냅니다.



### OrthoDoc: Multimodal Large Language Model for Assisting Diagnosis in Computed Tomography (https://arxiv.org/abs/2409.09052)
Comments:
          8 pages, 1 figure

- **What's New**: OrthoDoc는 120,000개의 CT 이미지와 진단 보고서를 기반으로 훈련된 다중 모달 대형 모델로, CT 진단을 위해 특별히 설계되었습니다. 기존 모델들과 비교하여 우수한 진단 능력과 정확성을 보여줍니다.

- **Technical Details**: OrthoDoc는 CT 이미지와 진단 보고서의 다중 모달 학습을 통해 의료 이미징 특징을 습득하고, RAG(검색 보완 생성) 모듈을 이용해 모델의 환각(hallucination) 문제를 완화합니다. 이 모델은 복잡한 CT 이미지를 처리하고 자연어로 상세한 진단 보고서를 생성할 수 있습니다.

- **Performance Highlights**: OrthoDoc는 기존 상용 모델과 비교하여 골절, 관절염 및 종양과 같은 일반적인 정형외과 질환의 진단에서 91% 이상의 정확도를 달성했습니다. 또한, 희귀 및 복잡한 사례를 처리할 때 탁월한 일반화 능력을 보이며 임상 응용에서 실용적인 유용성을 입증하고 있습니다.



### AI Meets the Classroom: When Does ChatGPT Harm Learning? (https://arxiv.org/abs/2409.09047)
- **What's New**: 본 논문은 생성 AI, 특히 대형 언어 모델(LLMs)이 코딩 수업에서 학습에 미치는 영향을 조사하는 연구입니다. 세 가지 연구를 통해 LLM 사용이 학습 결과에 긍정적이기도 하고 부정적이기도 한 효과를 미칠 수 있음을 보여주었습니다.

- **Technical Details**: 대학 수준의 프로그래밍 수업에서 수집된 관찰 데이터를 사용하여 LLM 사용의 효과를 입증하고, 일반적인 학습 시나리오를 반영한 실험 연구를 통해 인과 관계를 분석하였습니다. 학생들이 LLM을 개인 튜터처럼 활용할 경우, 즉 주제에 대해 질문하고 설명을 요청할 때는 이득이 있지만, 연습 문제의 해결을 위해 LLM에 과도하게 의존할 경우 학습이 저해된다는 것을 발견했습니다.

- **Performance Highlights**: LLMs 사용 시 학생들의 자가 인식 향상이 실제 학습 효과를 초과하여 자신의 능력을 과대평가할 위험이 있음이 확인되었습니다. 또한, LLMs는 경험이 적은 학생들에게 더 큰 학습 혜택을 제공합니다. 전반적으로 LLM을 학습 지원 도구로 활용할 가능성이 크지만, 학생들은 이러한 도구의 사용에 있어 주의해야 할 점이 많다는 것을 강조하고 있습니다.



### HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented Generation System for AI Legal and Policy Applications (https://arxiv.org/abs/2409.09046)
Comments:
          Under review for the EMNLP 2024 Workshop on Customizable NLP: Progress and Challenges in Customizing NLP for a Domain, Application, Group, or Individual

- **What's New**: 본 논문은 AI 법률 및 정책 분야에 최적화된 Hybrid Parameter-Adaptive RAG (HyPA-RAG) 시스템을 제안합니다. 이 시스템은 질의 복잡성 분류기를 통해 동적으로 매개변수를 조정하고, 밀집, 희소, 지식 그래프 탐색을 결합한 하이브리드 검색 전략을 사용하며, 특정 질문 유형과 메트릭을 포함한 평가 프레임워크를 갖추고 있습니다.

- **Technical Details**: HyPA-RAG는 질의 복잡성 분류기를 활용하여 매개변수를 최적화하고, 밀집 및 희소 검색 방법과 지식 그래프 검색 방법을 결합하여 검증된 검색 정확도를 높입니다. 시스템 디자인은 초기 k 결과를 찾은 후, 정해진 매개변수 매핑에 따라 검색 결과를 정제합니다.

- **Performance Highlights**: NYC Local Law 144 (LL144)를 기반으로 한 테스트에서 HyPA-RAG는 정확성과 충실도를 크게 향상시켰으며, 복잡하고 높은 위험의 AI 법적 및 정책 응용 분야에 적합한 적응형 NLP 시스템의 필요성을 충족하였습니다.



### United in Diversity? Contextual Biases in LLM-Based Predictions of the 2024 European Parliament Elections (https://arxiv.org/abs/2409.09045)
- **What's New**: 본 연구는 대형 언어 모델(LLM)을 활용하여 2024년 유럽 의회 선거의 투표 행동을 예측하며, LLM의 사회적 편향과 제한된 효용성을 강조합니다.

- **Technical Details**: GPT-4-Turbo를 활용하여 익명화된 개인 수준의 배경 정보를 바탕으로 다양한 프롬프트 내용과 언어를 사용하여 LLM이 각 개인의 투표 행동을 예측하게 합니다. 예측한 결과를 실제 선거 결과와 비교 분석합니다.

- **Performance Highlights**: 1. LLM 기반의 미래 투표 행동 예측이 주로 실패함. 2. 예측 정확도가 국가 및 언어적 맥락에 따라 고르게 분포되지 않음. 3. LLM 예측을 개선하려면 개인에 대한 자세한 태도 정보 제공이 필요함.



### ElasticAI: Creating and Deploying Energy-Efficient Deep Learning Accelerator for Pervasive Computing (https://arxiv.org/abs/2409.09044)
Comments:
          The paper is accepted by 2023 IEEE International Conference on Pervasive Computing and Communications (Best Demo Award)

- **What's New**: 본 논문은 에너지 효율적인 Deep Learning (DL) 가속기를 임베디드 Field Programmable Gate Arrays (FPGA)에서 생성하고 배포하는 ElasticAI-Workflow를 제안합니다. 이 워크플로우는 DL 개발자가 하드웨어 가속기를 보다 쉽게 만들 수 있도록 지원합니다.

- **Technical Details**: ElasticAI-Workflow는 ElasticAI-Creator와 Elastic Node 두 가지 핵심 구성 요소로 이루어져 있습니다. ElasticAI-Creator는 FPGA에서 DL 가속기를 자동으로 생성하는 도구 체인(toolchain)이고, Elastic Node는 생성된 가속기의 성능을 검증하는 하드웨어 플랫폼입니다. 이를 통해 고유한 DL 가속기의 성능을 충분히 보장할 수 있습니다.

- **Performance Highlights**: ElasticAI-Workflow는 세 단계로 구성되어 있으며, 각 단계는 DL 모델 설계, RTL 변환, 하드웨어 실행 및 성능 검증을 포함합니다. 이 워크플로우는 DL 모델의 최적화 및 에너지 효율성을 보장하며, 실제 하드웨어에서의 전력 소비를 모니터링할 수 있는 기능이 있습니다.



### Semantic Communication for Cooperative Perception using HARQ (https://arxiv.org/abs/2409.09042)
- **What's New**: 이 논문에서는 자율주행 차량에서의 협력 인식(cooperative perception) 향상을 위해 중간 융합(intermediate fusion)을 사용하는 새로운 의미적 소통 프레임워크를 제안합니다. 또한, 신호 대 잡음비(SNR)가 낮은 환경에서도 전송 신뢰성을 보장하기 위한 새로운 의미적 오류 탐지 방법을 도입했습니다.

- **Technical Details**: 제안하는 프레임워크는 중요성 맵(importance map)을 활용하여 주요 의미 정보를 증류하고, 직교 주파수 분할 다중화(OFDM)를 적용하여 시간 변동 다중 경로 페이딩 문제를 해결합니다. 이 시스템은 하이브리드 자동 반복 요청(HARQ)과 통합되어 데이터 전송의 신뢰성을 높입니다. SimCRC라는 새로운 의미적 오류 감지 방법이 성능을 향상시킵니다.

- **Performance Highlights**: 모의 실험 결과, 제안하는 모델은 기존의 개별 원천-채널 코딩 방식보다 인식 성능과 소비량에서 뛰어난 성능을 보여주었습니다. 특히, SimHARQ-I 및 SimHARQ-II를 통한 HARQ 기법은 기존 방법에 비해 우수한 효율성을 입증했습니다.



### Acceptable Use Policies for Foundation Models (https://arxiv.org/abs/2409.09041)
Comments:
          10 pages, 2 figures, 2 tables

- **What's New**: 기초 모델(foundation model) 개발자들이 사용자에게 해로운 사용을 방지하기 위한 Acceptable Use Policies (AUPs)를 채택하고 있다는 점을 밝힌 논문입니다. 이 정책들은 특정 용도에 대한 사용 금지를 법적으로 규제하는 방안으로써, AI 생태계에서의 규제를 이해하는 데 중요한 시각을 제공합니다.

- **Technical Details**: 이 논문은 30명의 기초 모델 개발자들이 채택한 AUPs를 식별하고 분석하여, 그들이 포함한 사용 제한의 유형과 내용을 검토합니다. 각 개발자의 AUPs는 127개의 상이한 사용 제한을 포함하고 있으며, 이는 AI 공급망 전반에 걸쳐 이질성을 초래할 수 있습니다. 개발자들은 경쟁업체나 특정 산업이 그들의 모델을 사용하는 것을 방지하기 위한 정책을 적용합니다.

- **Performance Highlights**: 일반적으로 AUPs는 사용자가 기초 모델을 통해 생성할 수 있는 콘텐츠의 유형을 제한하여 해를 끼칠 수 있는 행위를 방지하는 목적을 가지고 있습니다. 그러나 AUPs의 실시와 집행은 쉽지 않으며, 엄격한 집행은 연구자들의 접근을 제한하고 이로운 사용을 제한할 수 있습니다. 그럼에도 불구하고 AUPs는 기초 모델 시장과 AI 생태계에 상당한 영향을 미치는 초기 자율 규제(self-regulation)의 예로 평가됩니다.



### ChatSUMO: Large Language Model for Automating Traffic Scenario Generation in Simulation of Urban MObility (https://arxiv.org/abs/2409.09040)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 새로운 에이전트인 ChatSUMO를 소개합니다. ChatSUMO는 사용자의 언어 입력을 처리하여 도시 교통 시뮬레이터인 SUMO에서 시뮬레이션 시나리오를 생성할 수 있도록 돕습니다. 사용자는 전문 지식 없이도 단순한 텍스트 입력만으로 맞춤형 시나리오를 생성할 수 있습니다.

- **Technical Details**: ChatSUMO는 Llama 3.1 모델을 활용하여 구성된 다중 모듈 아키텍처를 가지고 있습니다. 입력 모듈은 사용자 입력을 처리하여 관련 키워드를 생성하고, 시뮬레이션 생성 모듈은 이 키워드를 사용하여 SUMO에서 교통 시나리오를 생성합니다. 사용자 맞춤화 모듈을 통해 교통 신호 최적화와 차량 경로 조정 등의 기능을 지원하며, 분석 모듈은 시뮬레이션 결과를 해석하여 상세한 보고서를 제공합니다.

- **Performance Highlights**: Albany 도시에 대한 실제 교통 시뮬레이션을 96%의 정확도로 생성할 수 있었으며, ChatSUMO를 통해 사용자들은 복잡한 설정 없이도 손쉽게 맞춤형 시나리오를 생성하고 지속적인 상호작용을 통해 다양한 시나리오를 만들어 낼 수 있습니다.



### AutoGeo: Automating Geometric Image Dataset Creation for Enhanced Geometry Understanding (https://arxiv.org/abs/2409.09039)
- **What's New**: AutoGeo라는 새로운 접근 방식을 통해 대규모 고품질 기하학 이미지 데이터셋을 자동으로 생성하는 방법을 제안하였습니다. AutoGeo-100k라는 10만 쌍의 이미지-텍스트를 포함하는 데이터셋을 구축하여 기하학적 문제 해결의 정확도를 향상시켰습니다.

- **Technical Details**: AutoGeo는 Augmented Geometry Clause System (AGCS), Rule-based Clause Selector (RCS), Sample Generator (SG)로 구성되어 있습니다. AGCS는 난이도에 따라 숫자 조항을 포함하며, RCS는 원하는 복잡도에 맞는 기하학 조항을 선택합니다. SG는 이 조항들을 통해 Python을 사용한 이미지 생성을 수행하고, ChatGPT를 통해 텍스트 설명을 생성합니다.

- **Performance Highlights**: AutoGeo-100k를 사용하여 여러 Multimodal Large Language Models (MLLMs)을 미세 조정한 결과, 기하학 이미지 이해 능력이 크게 향상되었으며, 기하학 캡셔닝과 수학적 추론 과제에서의 정확도가 개선되었습니다.



### Regional Style and Color Transfer (https://arxiv.org/abs/2404.13880)
Comments:
          Accepted by 2024 5th International Conference on Computer Vision, Image and Deep Learning

- **What's New**: 이번 논문은 지역적 스타일 전이(regional style transfer) 분야에 대한 혁신적인 기여를 제안합니다. 기존 방법이 전체 이미지에 균일하지만 비자연적인 스타일을 적용하는 문제를 해결하기 위해, 세분화 네트워크(segmentation network)를 활용하여 입력 이미지 내의 전경 객체를 정확히 격리한 후 배경 영역에 스타일 전이를 적용하는 새로운 접근 방식을 제시하였습니다.

- **Technical Details**: 제안된 방법에서는 이미지 세분화를 위해 DeepLabv3+ 모델을 사용하여 전경 객체와 배경을 효과적으로 구분하고, 스타일 및 색상 정보의 지역적 전이를 수행합니다. 이렇게 얻어진 배경과 전경 이미지는 블렌딩(blending) 기법을 사용해 매끄럽게 통합되어 최종 이미지를 생성합니다.

- **Performance Highlights**: 평가 결과, 제안된 접근 방식이 기존의 스타일 전이 방법들과 비교하여 훨씬 자연스러운 스타일 변환을 제공함을 입증하였습니다. 또한, 이를 통해 창의적인 표현의 새로운 가능성을 제시하며, 예술적 렌더링의 중요한 기초를 마련하였습니다.



### Confidence Trigger Detection: Accelerating Real-time Tracking-by-detection Systems (https://arxiv.org/abs/1902.00615)
Comments:
          Accepted by 2024 5th International Conference on Electronic Communication and Artificial Intelligence

- **What's New**: 본 연구에서는 Confidence-Triggered Detection (CTD)이라는 혁신적인 접근 방식을 제안하여 객체 추적의 속도와 정확성을 동시에 향상시키는 방법을 모색합니다. CTD는 다음 프레임이 이전 상태와 유사한 경우 객체 감지를 우회하도록 전략적으로 설계되었습니다.

- **Technical Details**: CTD는 객체의 신뢰도 점수를 활용하여 객체 감지를 트리거하는 시점을 결정합니다. 이는 객체의 최소 이동을 보일 때 불필요한 계산을 줄이고, 실질적으로 카메라로부터 수집된 다양한 프레임에 대한 관측 결과를 기반으로 새로운 감지를 발생시키는 방식입니다.

- **Performance Highlights**: 다양한 신뢰도 임계값에서의 평가를 통해 CTD는 추적 속도와 정확성 간의 최적의 트레이드 오프를 식별하였으며, 여러 탐지 모델에서의 실험을 통해 CTD 프레임워크의 견고성과 다재다능성을 입증하였습니다.



### Contextual Hourglass Network for Semantic Segmentation of High Resolution Aerial Imagery (https://arxiv.org/abs/1810.12813)
Comments:
          Accepted by 2024 5th International Conference on Electronic Communication and Artificial Intelligence

- **What's New**: 본 논문에서는 고해상도 항공 이미지의 의미 분할을 위한 새로운 방법인 Contextual Hourglass Network (CxtHGNet)을 제안합니다. 이 네트워크는 새로운 컨텍스트 시간 유리 모듈을 설계하여 낮은 해상도 피쳐 맵에서 컨텍스트 의미를 활용하고, 다양한 크기의 객체를 효과적으로 분별할 수 있습니다.

- **Technical Details**: CxtHGNet은 스택된 인코더-디코더 구조를 기반으로 하며, 여러 개의 컨텍스트 시간 유리 모듈을 연결하여 다중 스케일 특징을 효과적으로 추출합니다. 이 방법은 주목 메커니즘을 포함하여 라벨링의 견고성을 개선하고, 중간 감독을 통한 의미 학습을 더 잘 수행할 수 있도록 피드백 루프를 추가합니다.

- **Performance Highlights**: Potsdam 및 Vaihingen 데이터셋에서 테스트한 결과, 제안한 CxtHGNet이 이전 벤치마크 방법들보다 전체 성능 면에서 가장 우수한 결과를 나타냈습니다. 이 방법은 작은 크기의 객체를 포착하고, 보다 일관된 결과를 생성하는 데 기여했습니다.




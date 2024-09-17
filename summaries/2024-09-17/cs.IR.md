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




New uploads on arXiv(cs.CL)

### LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models (https://arxiv.org/abs/2407.12772)
Comments:
          Code ad leaderboard are available at this https URL and this https URL

- **What's New**: LMMS-EVAL은 통합되고 표준화된 멀티모달 평가 프레임워크로, 50개 이상의 작업과 10개 이상의 모델을 포함하고 있어 투명하고 재현 가능한 평가를 촉진합니다. 또한, LMMS-EVAL LITE와 Multimodal LIVEBENCH를 도입하여 효율성과 광범위한 범위를 강조하는 평가 툴킷을 소개합니다.

- **Technical Details**: 기존의 멀티모달 모델(LMM)을 평가하는 파이프라인에는 비표준화된 부분이 많아 투명성과 재현 가능성이 저해됩니다. LMMS-EVAL은 이러한 문제를 해결하기 위해 50개 이상의 작업과 10개 이상의 모델 및 약 30개의 버전을 포함하는 통합된 평가 스위트를 제공합니다. 이는 다양한 모델과 데이터셋 간의 통합 인터페이스를 통해 새로운 모델과 데이터셋의 통합을 용이하게 합니다.

- **Performance Highlights**: LMMS-EVAL LITE는 불필요한 데이터 인스턴스를 제거하여 평가 비용을 줄이고 효율성을 높이면서도 높은 평가 품질을 유지합니다. Multimodal LIVEBENCH는 뉴스와 온라인 포럼에서 최신 정보를 수집하여 모델의 제로샷 일반화 능력을 평가합니다. 이를 통해 LMM 평가에 경제적이고 일반화된 방법을 제공합니다.



### HDLCopilot: Hardware Design Library Querying with Natural Languag (https://arxiv.org/abs/2407.12749)
Comments:
          7 pages, 8 figures

- **What's New**: HDLCopilot는 하드웨어 설계 인력을 위해 LLM(Large Language Model) 기반의 PDK(Process Design Kits) 질의 시스템을 소개합니다. 이 시스템은 자연어 형식으로 PDK와 상호작용할 수 있어 정확하고 효율적으로 정보를 검색할 수 있게 해줍니다. 이를 통해 엔지니어들은 설계 과정에서 생산성을 높이고 잠재적인 오류를 줄일 수 있습니다.

- **Technical Details**: HDLCopilot는 Retrieval Augmented Generation (RAG) 기법을 통해 PDK 데이터베이스와 상호작용합니다. PDK 파일을 관계형 데이터베이스(SQL)로 변환한 후, LLM이 사용자의 자연어 입력을 바탕으로 동적으로 SQL 쿼리를 생성하여 데이터를 검색합니다. 주요 데이터 뷰로는 자유 시간 정보(liberty), 추상 레이아웃 정보(LEF), 그리고 금속 스택 속성(Technology LEF)이 있습니다.

- **Performance Highlights**: HDLCopilot는 다양한 자연어 문의에 대해 94.23%의 정확도로 응답할 수 있으며, 생성된 SQL 쿼리의 효율성은 98.07%입니다. 이 시스템은 하드웨어 설계 엔지니어가 PDK 정보를 더욱 빠르고 정확하게 접근할 수 있게 해줍니다.



### A LLM Benchmark based on the Minecraft Builder Dialog Agent Task (https://arxiv.org/abs/2407.12734)
- **What's New**: 이 연구에서는 Minecraft 빌더 작업을 LLM(대형 언어 모델) 평가에 적합한 벤치마크로 적응시키는 새로운 접근을 제안합니다. 이는 공간 기하학적 과제와 벡터 기반 수학 능력을 평가하며, 빌더 에이전트 설계에 대한 정보를 제공합니다. 기존 연구들은 복잡한 구조와 인간 작성 지침을 포함한 다양한 텍스트 데이터셋을 제안했으나, 본 연구는 일반적인 건축 작업으로 구성된 일련의 분리된 과제를 통해 빌더 에이전트를 테스트하는 종합적인 합성 벤치마크를 제공합니다.

- **Technical Details**: 본 벤치마크는 텍스트 기반 건축 명령을 기반으로 빌더가 목표 구조를 완성하는 Minecraft Builder Task를 변형한 것입니다. 초기 벤치마크는 Llama-3-70b-Instruct 모델을 사용해 테스트했으며, 다양한 프롬프트 기반 접근 방식(Zero Shot, Few Shot, Chain of Thought)을 채택하여 평가했습니다. 벤치마크는 절대 주소 지정, 상대 주소 지정, 원시 형태(직선, 타워, 큐브, 직사각형) 건축 명령을 포함합니다.

- **Performance Highlights**: 제로 샷 접근 방식에서는 모델이 축의 계산을 종종 누락하는 반면, Chain of Thought 접근 방식에서는 그런 오류가 줄었습니다. 또한, 모델은 오른손 3축 좌표계를 사용할 때 Z축의 양수 방향을 남쪽으로 제대로 인식하지 못하는 경우가 있었으며, 몇 가지 예제를 통해 이것을 강화함으로써 피할 수 있음을 발견했습니다.



### Is Sarcasm Detection A Step-by-Step Reasoning Process in Large Language Models? (https://arxiv.org/abs/2407.12725)
Comments:
          13 pages, 2 figures

- **What's New**: 새로운 프레임워크 SarcasmCue를 소개합니다. 이 프레임워크는 네 가지 프롬프트 전략을 포함하며, 이는 CoC(Chain of Contradiction), GoC(Graph of Cues), BoC(Bagging of Cues), 그리고 ToC(Tensor of Cues)입니다. 이 프레임워크는 LLMs가 인간의 풍자를 탐지하기 위해 순차적 및 비순차적 프롬프트 방법을 고려하도록 유도합니다.

- **Technical Details**: SarcasmCue 프레임워크는 네 가지 프롬팃 방법을 제안합니다. 첫째, CoC(Chain of Contradiction)은 표면 감정과 진정한 의도의 모순을 식별하도록 도와줍니다. 둘째, GoC(Graph of Cues)는 다양한 큐를 노드로, 그 관계를 엣지로 표현하여 보다 유연한 검색 구조를 제안합니다. 셋째, BoC(Bagging of Cues)는 다양한 큐 집합을 구성하여 여러 예측을 결합해 최종 결과를 생성합니다. 마지막으로 ToC(Tensor of Cues)는 3D 구조로 각 큐 유형 간 상호작용을 최적화합니다.

- **Performance Highlights**: 제안된 네 가지 프롬팃 방법은 4개의 벤치마킹 데이터셋에서 기존의 표준 IO 프롬팃칭, CoT(Chain of Thought), ToT(Tree of Thought)을 상당한 차이로 능가했습니다. 일반적으로 비순차적 프롬팃칭이 순차적 프롬팃칭보다 더 뛰어난 성능을 보였습니다. LLMs GPT-4o가 LLaMA보다 모든 작업에서 일관되게 우수한 성능을 보였습니다.



### Subgraph-Aware Training of Text-based Methods for Knowledge Graph Completion (https://arxiv.org/abs/2407.12703)
Comments:
          8 pages, including appendix with 8 figures and 12 tables, currently under open review for EMNLP 2024

- **What's New**: 최근 사전 학습된 언어 모델(Pre-trained Language Models, PLMs)의 미세 조정이 지식 그래프 완성(Knowledge Graph Completion, KGC)에 유망하다고 나타났습니다. 그러나 대부분의 PLM 기반 방법은 텍스트 정보만 인코딩하며, 지식 그래프(Knowledge Graphs, KGs)의 다양한 위상 구조를 간과합니다. 이 논문에서는 KGs의 구조적 속성과 PLM 기반 방법의 성능 간의 중요한 관계를 실증적으로 입증하였습니다. 이를 활용해 새로운 Subgraph-Aware Training framework for KGC (SATKGC)을 제안합니다.

- **Technical Details**: 제안된 SATKGC는 (i) 부정적 샘플링을 장려하기 위해 서브그래프를 이용한 미니 배칭을 사용하고 (ii) 구조적 속성 측면에서 더 어려운 엔티티와 하드 네거티브 트리플에 초점을 맞춘 새로운 대비 학습 방법을 통해 구조적 지식을 활용합니다. 이는 최초로 서브그래프의 구조적 유도를 PLM 미세 조정에 포괄적으로 통합한 연구입니다.

- **Performance Highlights**: 네 가지 KGC 벤치마크에서 광범위한 실험을 통해 SATKGC가 기존 KGC 방법보다 뛰어남을 입증하였습니다. 제안된 방법은 PLM 기반 KGC 방법이 KG의 위상 구조를 활용하여 어려운 네거티브 샘플링을 효과적으로 수행하고, 구조적으로 중요한 엔티티와 트리플에 더 집중할 수 있게 합니다.



### Patch-Level Training for Large Language Models (https://arxiv.org/abs/2407.12665)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 훈련 효율성을 향상시키기 위해 패치 수준의 훈련(patch-level training)을 제안합니다. 이는 여러 토큰을 하나의 패치로 압축하여 시퀀스 길이를 줄임으로써 달성됩니다. 기존의 토큰 수준 훈련(token-level training)과 비교하여 전체 계산 비용을 0.5배로 줄이면서 모델 성능이 유지된다고 보고하고 있습니다.

- **Technical Details**: 패치 수준 훈련의 핵심은 훈련 과정에서 시퀀스 길이를 줄이기 위해 여러 토큰을 단일 패치로 압축하는 것입니다. 패치 수준 훈련 과정에서 언어 모델은 짧은 시퀀스의 패치를 입력으로 받아 다음 패치를 예측하도록 훈련됩니다. 그런 다음 남은 훈련 데이터에 대해 토큰 수준 훈련을 계속 진행하여 추론 모드와 일치시킵니다. 예를 들어 패치 크기 K=4 및 λ=2/3로 설정할 경우, 전체 훈련 비용을 절반으로 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, 패치 수준 훈련을 통해 370M에서 2.7B 파라미터에 이르는 다양한 크기의 모델들이 계산 비용을 절반으로 줄이면서도 성능 저하 없이 성공적인 결과를 나타냈습니다. 특히, Pile 데이터셋에서 훈련된 370M 모델은 패치 수준 훈련 후 토큰 수준 훈련을 계속하면서 손실이 빠르게 감소하고, 초기 훈련보다 더 낮은 손실을 기록했습니다.



### Domain-specific or Uncertainty-aware models: Does it really make a difference for biomedical text classification? (https://arxiv.org/abs/2407.12626)
Comments:
          BioNLP 2024

- **What's New**: 이 연구는 사전 학습된 언어 모델(Pretrained Language Models, PLMs)의 영역별 특화 모델과 불확실성 인지 모델을 결합한 새로운 접근법을 제안합니다. 특히, 생명과학 분야에서 모델이 자신의 불확실성을 평가하는 능력의 중요성을 강조합니다. 이를 통해 생명과학 텍스트 분류 작업에서 두 가지 기준을 만족하는지 평가하고자 합니다.

- **Technical Details**: 이 연구는 빈도론적 딥러닝 모델(DNN)과 베이지안 딥러닝 모델(BNN)의 성능을 비교합니다. 모델들은 각기 다른 PLMs(BERT, BioBERT for English; CamemBERT, CamemBERT-bio for French)를 기반으로 하며, DropConnect와 같은 불확실성 인지 모듈이 포함되어 있습니다. 실험 데이터는 6개의 생명과학 데이터셋(MedABS, MedNLI, SMOKING, MORFITT, PxSLU, MedMCQA)을 사용합니다.

- **Performance Highlights**: 실험 결과, 영역 특화(+𝒟) 모델이 일반 모델(-𝒟)에 비해 우수한 성능을 보였습니다. 그러나 불확실성 인지(+𝒰) 모델이 항상 최고 성능을 보이는 것은 아닙니다. 특히, 특정 작업에서는 불확실성 인지 모델(-𝒟+𝒰)이 더 정확한 캘리브레이션 결과를 제공했습니다. 이에 따라 의료 분야에서는 작업에 따라 영역 특화와 불확실성 인지 중 어떤 것을 우선시할지 고려해야 합니다.



### Harnessing the Power of Artificial Intelligence to Vitalize Endangered Indigenous Languages: Technologies and Experiences (https://arxiv.org/abs/2407.12620)
- **What's New**: 인공지능(AI)과 최신 자연어 처리(NLP) 기술, 특히 대형 언어 모델(LLM)을 활용하여 사라져가는 원주민 언어를 문서화하고 사용을 장려하는 연구가 진행되고 있습니다. 2022년부터 시작된 이 연구는 주로 브라질의 원주민 언어를 대상으로 하고 있습니다.

- **Technical Details**: 연구진은 커뮤니티 참여와 사용을 중심으로 하는 대체 AI 개발 사이클을 제안하여, 윤리적 문제에 대응하고 있습니다. 또한, 소량의 데이터를 사용하여 최첨단 번역기를 미세 조정하여 고품질의 머신러닝 번역기를 개발하는 데 성공했습니다. 이를 통해 철자검사기, 다음 단어 예측기 등의 도구 개발이 가능해졌습니다.

- **Performance Highlights**: 연구는 2023년과 2024년에 브라질 원주민 커뮤니티와 협력하여 프로젝트를 진행했으며, 작성 지원 도구 등의 프로토타입을 성공적으로 개발했습니다. 특히 젊은 세대와 문서 작성자를 타겟으로 한 번역기 및 작성 도구는 언어의 유지와 활성을 돕는 데 중요한 역할을 하고 있습니다.



### E5-V: Universal Embeddings with Multimodal Large Language Models (https://arxiv.org/abs/2407.12580)
Comments:
          Code and models are available at this https URL

- **What's New**: 이번 연구에서는 멀티모달 대형 언어 모델 (MLLM, Multimodal Large Language Model)을 활용하여 범용 멀티모달 임베딩 (universal multimodal embeddings)을 달성하기 위해 E5-V라는 새로운 프레임워크를 도입했다. 이 방법은 멀티모달 정보 표현에 있어 기존 접근법보다 우수한 성능을 보여준다.

- **Technical Details**: E5-V는 텍스트 쌍 (text pairs)에 대해 단일 모달리티 훈련 (single modality training)을 실시하여, 훈련 비용을 약 95% 줄였다. 이는 비싼 멀티모달 데이터 수집을 제거하며, 텍스트 입력을 통해 비쥬얼 인코더를 제거하고 입력 크기를 감소시킨다. E5-V는 또한 프롬프트 (prompt)을 사용하여 모달리티 간 격차를 효과적으로 줄인다.

- **Performance Highlights**: E5-V는 다양한 작업(text-image retrieval, composed image retrieval, sentence embeddings, image-image retrieval)에서 기존 최첨단 (state-of-the-art) 성능을 초과하거나 경쟁하는 성능을 보여준다. 특히 단일 모달리티 훈련만으로 멀티모달 임베딩을 효율적으로 수행할 수 있었다.



### Towards Collaborative Intelligence: Propagating Intentions and Reasoning for Multi-Agent Coordination with Large Language Models (https://arxiv.org/abs/2407.12532)
- **What's New**: 이 논문에서는 다중 에이전트 시스템(Multi-Agent Systems)에서 협업을 개선하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLMs)을 협업 에이전트로 훈련시켜 공동 작업에서 조정된 행동을 가능하게 합니다. 각 에이전트는 현재 목표와 관련된 하위 작업을 포함하는 개인적인 의도를 유지하며, 주기적으로 의도를 방송하여 다른 에이전트가 조정 작업을 추론할 수 있도록 합니다. 이를 통해 의도 전파가 여러 에이전트 간의 잘못된 조정을 줄이는 데 도움을 줍니다.

- **Technical Details**: ReMALIS 프레임워크는 계획 모듈, 그라운딩(grounding) 모듈, 실행 모듈로 구성됩니다. 실행 중 여러 에이전트는 하위 환경과 상호작용하며 의도를 전달해 조정된 행동을 이룹니다. 그라운딩 모듈은 조정 패턴에 따라 동적으로 이해 전략을 조정하며, 실행 에이전트의 피드백은 계획 모듈에 영향을 주어 하위 작업을 동적으로 재계획하게 합니다. 의도 전파 네트워크는 방송된 의도를 팀원 별로 전달 메시지로 변환하여 필요에 따라 목표를 공유합니다.

- **Performance Highlights**: 협업 환경에서의 시뮬레이션 결과, 의도 전파는 에이전트 간 하위 작업의 종속성을 맞추어 잘못된 조정을 줄이는 데 효과적임을 확인했습니다. 에이전트는 언제 의도를 전달하고 어떤 팀원이 작업 세부 사항을 요구하는 지 배우며, 그 결과 새로운 조정된 행동이 나타났습니다. 다양한 단일 에이전트 기반 프레임워크 및 최신의 MARL 방법과 비교해보면, ReMALIS 프레임워크는 복잡한 협업 작업에서 더 나은 성능을 보입니다. 이는 LLMs를 의사 전달, 전략 조정, 협업 재계획이 가능한 협업 에이전트로 배포하는 데 효과적임을 보여줍니다.



### Crafting the Path: Robust Query Rewriting for Information Retrieva (https://arxiv.org/abs/2407.12529)
Comments:
          1 figure, 12 tables

- **What's New**: 이 논문에서는 정보 검색 시스템을 위한 새로운 구조화된 쿼리 재작성 방법인 'Crafting The Path'를 제안합니다. 이 방법은 세 단계를 통해 쿼리와 관련된 필요한 정보를 생성하여 검색에 최적화된 쿼리를 재작성합니다. 주요 단계는 쿼리 개념 이해(Query Concept Comprehension), 쿼리 유형 식별(Query Type Identification), 그리고 예상 답변 추출(Expected Answer Extraction)입니다.

- **Technical Details**: Crafting The Path는 LLM(대형 언어 모델) 내부 지식에 대한 의존성을 줄이고, 각 단계별 프로세스를 통해 사실 오류를 최소화하는 것을 목표로 합니다. 첫 번째 단계인 쿼리 개념 이해 단계에서는 기본 배경 지식을 제공합니다. 두 번째 단계에서는 필요한 정보를 필터링하여 쿼리 유형을 식별합니다. 마지막 단계에서는 필요한 정보를 추출하여 정확한 정보를 찾을 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, Crafting The Path는 기존의 쿼리 재작성 방법들보다 성능이 우수하였고, 특히 LLM이 익숙하지 않은 도메인에서 더 뛰어났습니다. 모델의 내부 지식에 대한 의존도가 낮으며, 사실 오류가 적고, 지연 시간(latency)도 기존 방법들에 비해 7.3% 낮았습니다. 또한, FActScore(사실 정확도 점수)가 10% 더 높아졌으며, 정답률도 3.57% 증가하였습니다.



### Struct-X: Enhancing Large Language Models Reasoning with Structured Data (https://arxiv.org/abs/2407.12522)
- **What's New**: 이 논문에서는 Struct-X라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 '읽기-모델링-채우기-반영-추론(Automation-Reasoning)'의 다섯 가지 주요 단계를 통해 대형 언어 모델(LLM)이 구조화된 데이터를 효과적으로 활용할 수 있도록 설계되었습니다. Struct-X는 구조화된 데이터를 그래프 임베딩(graph embeddings)을 사용하여 위상학적 공간으로 변환하고, 지식 검색 모듈을 통해 불완전한 엔티티 정보를 보완하며, 자기 지도 모듈을 통해 관련 없는 토큰을 필터링합니다. 이를 통해 LLM이 더 짧은 토큰 길이로 더 효과적으로 추론할 수 있습니다.

- **Technical Details**: Struct-X는 구조화된 데이터를 위상학적 네트워크로 변환하는 과정을 포함합니다. 첫 단계에서는 그래프 임베딩을 통해 데이터를 인코딩하고, 지식 검색 모듈을 통해 누락된 정보를 보완합니다. 이어서 자기 텍스트 생성 모듈(Self-Retrieved Generation Module)을 사용하여 관련 없는 토큰을 걸러냅니다. 마지막 단계에서는 선택된 토큰을 포함한 위상학적 네트워크를 구축하여 토큰 길이를 줄입니다. 또한, Auxiliary Module을 포함하여 LLM이 구조화된 데이터를 분석하고 추론할 수 있도록 돕습니다.

- **Performance Highlights**: 구조화된 데이터를 활용한 Struct-X는 '지식 그래프 질의 응답(Knowledge Graph Question Answer) 작업'과 '장문 독해 작업(Long Document Reading Comprehension Task)'을 포함한 다양한 벤치마크 실험에서 LLM의 추론 능력을 크게 향상시켰습니다. 실험 결과, Struct-X는 복잡한 입력 맥락에서 LLM이 더 정밀하고 신뢰할 수 있는 추론을 할 수 있음을 보여주었습니다.



### On Initializing Transformers with Pre-trained Embeddings (https://arxiv.org/abs/2407.12514)
- **What's New**: 이 논문은 Transformer 기반 모델을 처음부터 학습할 때, 사전 훈련된 임베딩(pre-trained embeddings)보다 랜덤 초기화(random initialization)를 사용하는 것이 더 효과적이라는 현재의 관행을 조사합니다. 특히, GloVe와 T5, mT5와 같은 언어 모델에서 추출한 일부 서브워드 임베딩이 랜덤 초기화보다 성능이 낮다는 점을 발견했습니다. 반면 BERT와 mBERT 임베딩은 랜덤 초기화보다 더 나은 성능을 보였습니다.

- **Technical Details**: 이 연구는 두 가지 요인이 이러한 혼합된 결과에 기여한다고 주장합니다: 모델의 파라미터 분포에 대한 민감도(parameter distribution sensitivity)와 위치 인코딩(position encoding)과 임베딩 간의 상호작용입니다. 실험에서는 네 가지 작업 - 기계 번역(machine translation), 감성 분석(sentiment analysis), 자연어 추론(natural language inference)을 통해 여러 종류의 사전 훈련된 임베딩과 랜덤 초기화를 비교했습니다.

- **Performance Highlights**: 1. GloVe, T5, mT5와 같은 사전 훈련된 임베딩은 Xavier 초기화보다 분포가 훨씬 넓어 성능이 낮아지는 경향이 있습니다. BERT 및 mBERT 임베딩은 Xavier 초기화 범위와 유사하여 더 나은 성능을 보입니다. 2. 사전 훈련된 임베딩과 위치 인코딩의 상호작용은 임베딩의 분산이 클 경우 위치 인코딩이 흡수될 수 있습니다. 3. 사전 훈련된 임베딩의 요소를 셔플하면 성능이 일관되게 저하되는 것으로 보아, 사전 훈련된 임베딩이 의미론적 정보를 모델 성능에 기여하는 것으로 보입니다.



### $\textit{GeoHard}$: Towards Measuring Class-wise Hardness through Modelling Class Semantics (https://arxiv.org/abs/2407.12512)
Comments:
          Findings of ACL 2024

- **What's New**: 이 논문은 새로운 개념인 ‘class-wise hardness’를 도입하여 데이터의 클래스별 난이도 특성을 정량적으로 측정하려는 시도를 소개합니다. 이는 특히 저자원 환경에서의 샘플 선택에 중요한 역할을 할 수 있습니다.

- **Technical Details**: 기존의 인스턴스 수준(metrics) 기반 난이도 측정 방식의 한계를 극복하기 위해 ‘GeoHard’를 제안합니다. 이는 의미적 임베딩 공간(semantic embedding space)에서 클래스 지형(class geometry)을 모델링하여 클래스별 난이도를 측정합니다.

- **Performance Highlights**: GeoHard는 클래스별 난이도 측정에서 인스턴스 수준(metrics) 기반 지표를 피어슨(Pearson) 상관계수에서 59% 이상 향상시킵니다. 추가적으로, 인간의 판단과 모델 학습 모두에서 일관된 난이도 분포를 보임을 실험으로 확인했습니다.



### MERLIN: Multimodal Embedding Refinement via LLM-based Iterative Navigation for Text-Video Retrieval-Rerank Pipelin (https://arxiv.org/abs/2407.12508)
Comments:
          Work in progress

- **What's New**: 멀티미디어 콘텐츠의 빠른 확장은 방대한 데이터 컬렉션에서 관련 동영상을 정확하게 검색하는 것을 점점 더 어렵게 만들고 있습니다. 이를 해결하기 위해 MERLIN (Multimodal Embedding Refinement via LLM-based Iterative Navigation)이라는 새로운 트레이닝이 필요 없는 파이프라인을 소개합니다. MERLIN은 대형 언어 모델(Large Language Models, LLMs)을 활용하여 사용자 관점에서 쿼리를 반복적으로 학습하고, 쿼리 임베딩을 개선하여 동영상 콘텐츠와 더 잘 맞추는 과정을 통해 검색 성능을 향상시킵니다.

- **Technical Details**: MERLIN은 LLM을 통해 생성된 일련의 질문과 그에 대한 답변을 기반으로 쿼리 임베딩을 반복적으로 개선하는 방법을 사용합니다. 이 과정은 메타데이터와 최상위로 검색된 동영상의 콘텐츠를 바탕으로 질문을 생성하고, 그 질문에 대한 답변을 통해 쿼리 임베딩을 정밀하게 조정합니다. 이 접근법은 기존 모델을 재훈련할 필요 없이 사용자가 원하는 검색 결과에 더 잘 맞출 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, MERLIN은 MSR-VTT, MSVD, 그리고 ActivityNet 등의 데이터셋에서 Recall@1이 크게 향상된 것을 보여줍니다. 예를 들어, MSR-VTT에서는 Recall@1 성능이 44.00에서 78.00으로, MSVD에서는 52.39에서 77.61로, ActivityNet에서는 56.58에서 68.44로 증가했습니다. 이러한 성과는 MERLIN이 사용자 의도와 상호작용을 중시하는 검색 프레임워크로서의 유효성을 잘 보여줍니다.



### Case2Code: Learning Inductive Reasoning with Synthetic Data (https://arxiv.org/abs/2407.12504)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 귀납적 추론 능력을 평가하고 학습시키는 것을 목표로 하는 새로운 연구를 소개합니다. 'Case2Code'라는 과제를 제안하며, 이를 통해 실제 프로그램들에서 입력-출력 변환을 관찰하고 그에 따라 숨겨진 규칙을 유추하는 능력을 향상시키고자 합니다.

- **Technical Details**: Case2Code 과제는 다양한 실행 가능한 프로그램을 수집하고, 각각에 대해 입력-출력 변환을 합성하여 LLM이 이러한 합성된 사례(이하 I/O 사례)들로부터 코드 구현을 유추하도록 요구합니다. 이러한 과제를 해결하기 위해 LLM을 평가하고, 대규모 Case2Code 훈련 샘플을 합성하여 LLM이 귀납적 추론을 수행할 수 있도록 학습시킵니다. 데이터 합성 과정에서의 핵심은 'Writer LLM'을 사용하는 것으로, 이는 고성능 LLM의 직접 성능에 의존하지 않고 신뢰할 수 있는 훈련 데이터를 규모 있게 얻을 수 있도록 도움을 줍니다.

- **Performance Highlights**: 실험 결과, Case2Code는 LLM에게 상당히 어려운 과제이며, GPT-4와 같은 강력한 LLM도 어려움을 겪는 것으로 나타났습니다. 그러나 Case2Code 데이터를 훈련에 포함함으로써 LLM의 귀납적 추론 능력을 향상시킬 수 있었으며, 이는 HumanEval이나 MBPP 같은 코드 생성 과제에서도 성능 향상을 가져왔습니다. 이를 통해, 합성 데이터를 통한 귀납적 추론 학습이 매우 큰 잠재력을 지니고 있음을 확인하였습니다.



### Automate or Assist? The Role of Computational Models in Identifying Gendered Discourse in US Capital Trial Transcripts (https://arxiv.org/abs/2407.12500)
- **What's New**: 본 연구는 미국 형사 재판에서 여성을 피고인으로 한 사형 재판에서 성 편향 언어를 식별하기 위한 자동화 시스템을 도입하는 사례 연구를 제시합니다. 사형 변호사와 자연어 처리(NLP, Natural Language Processing) 기술자가 협력하여 3단계 연구를 수행하였습니다.

- **Technical Details**: 첫 번째 단계에서는 법률 전문가가 수동으로 주석(annotation)을 달고, 두 번째 단계에서는 컴퓨터 모델을 개발하고 평가하며, 세 번째 단계에서는 모델의 예측과 인간의 주석을 비교합니다. 특이한 점은 이 연구에서 성별 편향을 식별하는 작업이 주관적 판단이 많이 필요하다는 점입니다. 따라서 전문가들이 모델을 효율성 향상뿐 아니라 자신의 편향을 도전하고 주석 규칙을 개선하는 기회로 간주합니다.

- **Performance Highlights**: 법률 전문가들은 자동화 시스템이 완벽히 인간 주석을 대체할 수 없다는 점을 시사했습니다. 대신, 컴퓨터 모델은 법률 전문가들이 주석 처리 과정을 더 정교하게 만들고 합의를 형성하는 도구로 유용하다고 평가되었습니다.



### Evaluating Linguistic Capabilities of Multimodal LLMs in the Lens of Few-Shot Learning (https://arxiv.org/abs/2407.12498)
Comments:
          Preprint. 33 pages, 17 Figures, 3 Tables

- **What's New**: 이번 연구는 Multimodal Large Language Models(MLLMs)의 성능을 VALSE 벤치마크에서 평가하고, few-shot In-Context Learning(ICL)과 Chain-of-Thought(CoT) 프롬프팅의 효과를 조사했습니다. 모델 크기와 사전 훈련 데이터셋에 따라 다양한 state-of-the-art MLLMs를 평가한 결과, ICL과 CoT 프롬프팅이 특히 복잡한 추론과 맥락 이해가 필요한 작업에서 모델 성능을 크게 향상시킨다는 것을 밝혔습니다.

- **Technical Details**: ICL은 몇 가지 예시를 제공하여 모델 성능을 향상시키는 방법입니다. 이에 반해 CoT는 최종 답변을 제공하기 전에 추론 과정을 생성하는 프롬프팅 방법론입니다. VALSE 벤치마크는 존재, 복수, 카운팅, 공간 관계, 행동, 그리고 코어퍼런스와 같은 언어적 현상을 평가하기 위한 것입니다. MLLMs는 캡셔닝 데이터셋에서 더 뛰어난 성능을 보였으며, 교차된 이미지-텍스트 데이터셋에서 훈련된 모델은 few-shot 학습에서 더 효과적이었습니다.

- **Performance Highlights**: 14개의 다른 MLLMs를 VALSE 벤치마크에서 평가한 결과, few-shot ICL 설정에서 모델의 전반적인 성능이 향상되었으며, 쿼리 이미지-텍스트 페어와 유사한 예시를 제공할 때 성능이 더욱 향상되었습니다. CoT는 중간 추론 단계가 필요한 작업에서 높은 효율을 보였으며, 특정 예시 없이도 형태 및 의미론적 구조를 효과적으로 인식할 수 있는 능력을 보여주었습니다.



### Pretraining Data and Tokenizer for Indic LLM (https://arxiv.org/abs/2407.12481)
- **What's New**: 새로운 다국어 인도어 대형 언어 모델(Large Language Models, LLM)을 개발하기 위해 혁신적인 데이터 준비 방법을 제시합니다. 공개 소스와 독점 소스(Common Crawl, 인도어 책, 뉴스 기사, 위키피디아 등)를 아우르는 데이터 수집은 다양하고 풍부한 언어적 표현을 보장합니다. 각각의 인도어 언어에 맞춤형 전처리 파이프라인을 설계하여 중복되고 저품질의 텍스트 콘텐츠를 효과적으로 제거합니다. 특별히 공통 크롤(Common Crawl) 데이터의 중복 제거를 통해 70%에 달하는 크롤된 웹 페이지의 중복 문제를 해결합니다.

- **Technical Details**: 모든 인도어 언어에 대해 데이터 필터링 및 중복 제거를 위한 맞춤형 전처리 과정을 설계하였습니다. 다양한 오픈 소스와 독점 데이터셋(Common Crawl, 인도어 책, 뉴스 기사, 위키피디아 등)을 포함한 광범위한 데이터 소스를 사용하여 방대한 인도어 데이터 코퍼스를 구축하였습니다. 또한 최적의 토크나이제이션을 위해 인도어 언어에 맞춤화된 새로운 다국어 토크나이저(Tokenization) 훈련 전략을 제시하였으며, 이를 통해 인도어 언어에 특화된 토크나이저가 기존의 최신 OpenAI Tiktoken 토크나이저를 능가함을 입증하였습니다.

- **Performance Highlights**: 제안된 인도어 다국어 토크나이저는 인도어 언어에서 더 뛰어난 단어당 토큰 비율을 달성하여 현존하는 최고 수준의 OpenAI Tiktoken 토크나이저를 능가합니다. 이는 인도어 언어 모델의 성능을 크게 향상시키는 데 기여합니다.



### A Novel Dependency Framework for Enhancing Discourse Data Analysis (https://arxiv.org/abs/2407.12473)
- **What's New**: 이 연구는 다양한 이론적 기반에서 구축된 담화 코퍼스(discourse corpora)의 통일된 분석을 위해 PDTB(Penn Discourse Treebank) 주석을 의존 구조(dependency structures)로 변환하는 방법을 개발했습니다. 이를 통해 여러 언어에서 PDTB 스타일 코퍼스에서 파생된 의존 데이터의 유효성을 BERT 기반 담화 파서를 사용하여 테스트하고, 동일한 텍스트에 대해 PDTB와 RST(Rhetorical Structure Theory) 주석을 의존 구조로 변환하여 두 의존 거리 사이의 상관관계를 분석합니다.

- **Technical Details**: 본 연구에서는 PDTB 주석을 의존 구조로 변환하고, 다양한 언어에서 이를 검증하기 위해 최첨단 BERT 기반 담화 파서를 사용했습니다. 또한 '의존 거리' 메트릭스를 적용하여 영어에서 RST 의존성과 PDTB 의존성 사이의 상관 관계를 조사했습니다. 이를 통해 PDTB 의존 데이터가 유효하며 두 가지 의존 거리 사이에 강한 상관 관계가 있음을 확인했습니다.

- **Performance Highlights**: 이 연구는 종합적인 의존 프레임워크를 설립하여 기존 담화 코퍼스의 한계를 극복하고, 다양한 알고리즘을 지원하며, 계산 담화 분석(computational discourse analysis) 및 언어과학 연구를 촉진합니다. 특히, 다국어 검증을 통해 영어를 넘어 프레임워크의 일반화 가능성을 입증했습니다.



### Continual Learning for Temporal-Sensitive Question Answering (https://arxiv.org/abs/2407.12470)
Comments:
          Accepted by IJCNN 2024

- **What's New**: 이번 연구는 새로운 연구 분야인 시계열 민감형 질문 응답 (Temporal Sensitive Question Answering, TSQA)의 연속 학습 (Continual Learning, CL)을 탐구합니다. 기존 연구에서 미래 사건의 예측 불가능한 본성을 종종 간과해왔습니다. 이는 모델이 시간이 지남에 따라 지속적으로 지식을 획득해야 하는 실제 응용에서 중요한 문제입니다. 연구진은 연속 학습 단계에 맞춘 새로운 데이터셋을 생성하고, 시간적 메모리 재생 (temporal memory replay)과 시간적 대조 학습 (temporal contrastive learning)을 통합한 훈련 프레임워크를 제안했습니다.

- **Technical Details**: 연구의 주요 기술적 요소로는 시간 민감형 질문 응답의 연속 학습 (CLTSQA)을 위한 새로운 데이터셋 구축, 시간적 메모리 재생 (temporal memory replay) 기법을 사용해 과거 지식의 망각을 완화, 시간적 대조 학습을 통해 모델의 시간 정보에 대한 민감도를 향상시키는 프레임워크가 있습니다. 데이터셋은 다섯 개의 서브셋으로 나뉘어 각 연속 학습 단계에 맞춘 설계되었습니다.

- **Performance Highlights**: 실험 결과는 두 가지 주요 인사이트를 강조합니다: 첫째, 기존 모델은 CLTSQA 작업에서 독특한 도전에 직면합니다. 둘째, 제안된 프레임워크는 이러한 문제를 효과적으로 해결하여 향상된 성능을 보입니다. 구체적으로, 최신 정보를 포함한 질문에 대한 답변 성능이 개선되고, 역사적 질문에 대한 성능 유지도 우수했습니다.



### Sharif-STR at SemEval-2024 Task 1: Transformer as a Regression Model for Fine-Grained Scoring of Textual Semantic Relations (https://arxiv.org/abs/2407.12426)
Comments:
          10 pages, 9 figures, 4 tables

- **What's New**: 최근에 등장한 대형 언어 모델들로 인해 의미 텍스트 연관성(Semantic Textual Relatedness, STR)에 대한 접근 방식에 새로운 변화가 있었습니다. 이 논문에서는 RoBERTa transformer를 미세 조정하여 문장 수준의 STR을 조사하였으며, 특히 라틴어 계열 언어에서 괄목할 성과를 보였습니다.

- **Technical Details**: 논문에서는 RoBERTa Transformer를 미리 학습된 상태에서 회귀 모델로 사용하여 입력된 텍스트에 대한 부동 소수점 값을 예측하는 방식으로 미세 조정을 수행했습니다. STR 시스템은 영어, 스페인어, 아랍어 데이터를 활용하여 supervised 학습을 적용하였으며, 문장을 연관성을 점수로 예측하는 데 초점을 맞추었습니다.

- **Performance Highlights**: 영어에서의 상관도는 0.82로 19위를 기록하였고, 스페인어에서는 0.67의 상관도로 15위를 차지했습니다. 반면, 아랍어에서는 0.38의 상관도로 20위에 그쳐 성과가 저조했습니다. 이를 통해 라틴어 계열 언어에서는 성능이 향상되었지만, 비 라틴어 계열 언어에서는 도전적인 부분이 있음을 알 수 있습니다.



### Navigating the Noisy Crowd: Finding Key Information for Claim Verification (https://arxiv.org/abs/2407.12425)
- **What's New**: EACon (Evidence Abstraction and Claim Deconstruction) 프레임워크를 통해 대형 언어 모델(LLM)의 주장 검증 성능을 향상시키는 방법이 제안되었습니다. EACon은 잡음이 많은 증거와 주장에서 중요한 정보를 추출 및 요약하고, 주장을 하위 주장으로 분해하여 각각을 개별적으로 검증합니다. 이를 통해 LLM은 더 정교하게 주장 검증을 수행할 수 있습니다.

- **Technical Details**: EACon 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: 증거 추출(Evidence Abstraction), 주장 분해(Claim Deconstruction), 하위 주장 검증(Subclaim Verification). 증거 추출 단계에서는 키워드 기반 기법을 사용하여 주장에서 키워드를 추출하고 퍼지 매칭을 적용하여 관련 키워드를 선택합니다. 그런 다음 원시 증거에서 중요한 정보를 추출하고 요약합니다. 주장 분해 단계에서는 원래의 주장을 하위 주장으로 분해하여 각각을 검증합니다. 이 두 단계를 거친 후, 하위 주장 검증 단계에서 각 하위 주장을 검증하여 최종 결과를 도출합니다.

- **Performance Highlights**: EACon은 두 개의 오픈 소스 LLM(Vicuna-13B 및 Mixtral-8x7B)을 사용하여 HOVER와 FEVEROUS-S 데이터셋에서 평가되었습니다. 결과적으로 EACon은 LLM의 주장 검증 성능을 일관되고 상당히 향상시켰습니다.



### TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish (https://arxiv.org/abs/2407.12402)
- **What's New**: 터키어 언어 모델의 이해를 평가하기 위해, 최초의 다중 작업, 다지선다형 질문 답변 (QA) 벤치마크인 TurkishMMLU가 도입되었습니다. TurkishMMLU는 터키 고등학교 교육 커리큘럼을 기반으로 한 10,000개 이상의 질문을 포함하고 있으며, 자연 과학, 수학, 터키 문학 및 터키 공화국 역사와 같은 다양한 주제를 다룹니다.

- **Technical Details**: TurkishMMLU는 터키 고등학교 커리큘럼에 맞게 작성된 10,032개의 다지선다형 질문으로 구성되어 있습니다. 이 질문들은 터키 교육부 산하의 온라인 학습 플랫폼에서 전문가들이 작성한 것입니다. 우리는 Gemma, Llama, MT5 등을 포함한 다국어 오픈 소스 모델과 GPT-4, Claude, Gemini 등을 포함한 폐쇄 소스 모델, 그리고 Turkish-adapted 모델인 Trendyol과 같은 20개 이상의 대형 언어 모델(LLMs)을 평가했습니다. 또한, zero-shot 및 few-shot 평가, chain-of-thought (연쇄적 사고) 추론, 질문 난이도 분석이 포함되었습니다.

- **Performance Highlights**: 모델 성능 평가 결과, 여러 LLM들이 다양한 설정에서 테스트되었습니다. 특히 zero-shot 및 few-shot 설정에서의 성능과 chain-of-thought 추론을 통한 성능 차이에 대해 심도 있는 분석을 실시하였으며, 이를 통해 현재 터키 언어 모델의 능력과 한계에 대한 통찰을 제공했습니다. 이러한 통찰은 향후 터키어 및 다른 언어 모델 개발에 중요한 기반이 될 것입니다.



### PersLLM: A Personified Training Approach for Large Language Models (https://arxiv.org/abs/2407.12393)
Comments:
          10 pages for main text, 5 figures

- **What's New**: 최신 연구에서는 PersLLM이라는 새로운 접근 방식을 제안하여 대형 언어 모델(LLM)에 심리학적으로 기반한 성격 원칙을 통합했습니다. 이는 기존의 표면적인 성격 모방을 넘어서서 더 일관되고 역동적인 성격 모델을 제공합니다.

- **Technical Details**: PersLLM은 두 가지 주요 단계로 구성됩니다. 첫 번째는 '성격화된 데이터 구축(personified data construction)' 단계로, 목표 인물의 경험, 지식, 의견, 말투 등을 반영한 데이터를 수집하고 이를 대화 형식으로 재구성합니다. 두 번째는 '성격화된 모델 훈련(personified model training)' 단계로, 체인-오브-생각(Chain-of-Thought) 프롬프팅 전략과 자동 직접 선호 최적화(Direct Preference Optimization, DPO) 기법으로 모델의 일관성과 역동성을 강화합니다.

- **Performance Highlights**: 싱글 에이전트 실험에서는 '해리 포터 시리즈'의 여섯 캐릭터를 기반으로 PersLLM의 성능을 검증하였으며, 대화 응답이 각 캐릭터의 경험과 관점을 잘 반영하는 것으로 나타났습니다. 멀티 에이전트 커뮤니케이션 테스트에서는 PersLLM이 복수의 에이전트 간 협력과 충돌 상황에서 더 인간다운 상호작용을 모사하는 데 뛰어난 능력을 보여주었습니다. 인간-에이전트 상호작용 평가에서도 PersLLM은 사용자 경험을 크게 향상시키는 것으로 평가되었습니다.



### Morphosyntactic Analysis for CHILDES (https://arxiv.org/abs/2407.12389)
- **What's New**: 새로운 연구는 언어 학습 과정을 여러 언어 간에 비교하기 위한 일관된 정량적 프레임워크(quantitative framework)를 구축하는 문제를 다룹니다. 최근 AI(Artificial Intelligence)와 ML(Machine Learning)의 진보가 이를 해결하기 위한 새로운 방법을 제공하고 있습니다. 연구팀은 Batchalign2 프로그램(Liu et al., 2023)을 사용하여 CHILDES 데이터베이스의 데이터를 전사하고 연결하였으며, 27개 언어에 대해 일관되고 비교 가능한 형태통사론적 분석을 제공하기 위해 UD(Universal Dependencies) 프레임워크를 적용했습니다.

- **Technical Details**: 연구는 음성 인식(ASR, automatic speech recognition) 및 자연어 처리(NLP, natural language processing)의 최근 발전을 활용하여 더 깊이 있는 교차언어적(crosslinguistic) 언어 학습 연구를 가능하게 합니다. Batchalign2 프로그램을 통해 데이터를 전사하고 연결(review)하였고, 그 데이터를 CHILDES 데이터베이스에 적용하여 UD(Universal Dependencies) 프레임워크로 분석했습니다.

- **Performance Highlights**: 27개 언어에 대해 일관되고 비교 가능한 형태통사론적 분석을 제공함으로써, 이 새로운 자원은 언어 학습에 대한 더 깊은 교차언어적 연구를 가능하게 하였습니다. 이러한 접근 방식은 언어 발달 연구자들에게 큰 도움이 될 것입니다.



### Deep Learning-based Sentiment Analysis of Olympics Tweets (https://arxiv.org/abs/2407.12376)
- **What's New**: 이번 연구는 올림픽 게임과 관련된 트윗을 통해 글로벌 청중의 감정을 이해하기 위해 고도화된 딥러닝(Sentiment Analysis, SA) 모델을 개발하고자 합니다. 이 연구는 감정 분석 모델을 개선하고 전 세계의 올림픽에 대한 태도를 대표하는 데 기여합니다.

- **Technical Details**: 연구에서는 데이터 선택, 전처리, 시각화, 특징 추출, 모델 구축에 중점을 두고 있습니다. 기본 나이브 베이즈(Naïve Bayes, NB) 모델과 고도화된 세 가지 딥러닝 모델인 Convolutional Neural Network(CNN), Bidirectional Long Short-Term Memory(BiLSTM), 그리고 Bidirectional Encoder Representations from Transformers(BERT)를 사용하였습니다.

- **Performance Highlights**: 실험 결과, BERT 모델이 가장 높은 정확도인 99.23%를 달성하며 올림픽 관련 감정을 효과적으로 분류할 수 있음을 보여주었습니다.



### Conversational Query Reformulation with the Guidance of Retrieved Documents (https://arxiv.org/abs/2407.12363)
Comments:
          8 pages, 5 tables

- **What's New**: 대화형 질문 응답(Conversational QA, ConvQA) 시스템에서의 질의 처리를 최적화하는 새로운 프레임워크 'GuideCQR'를 소개합니다. GuideCQR는 기존의 'Conversational Query Reformulation (CQR)' 방법보다 더 나은 검색 결과를 제공하기 위해 안내 문서(guided documents)를 활용하여 질의를 개선합니다.

- **Technical Details**: GuideCQR는 세 가지 단계로 구성됩니다. 첫 번째로, 기본 질의를 받아 안내 문서 집합을 검색합니다. 두 번째로, 이 문서들을 재랭크하여 가장 관련성 높은 문서를 우선 순위에 둡니다. 마지막으로, 키워드 확장 및 예상 답변 생성 과정을 거쳐 최종 질의를 구성합니다. 이 과정에서 각 단계는 질의를 더 효과적으로 만들기 위해 설계되었습니다.

- **Performance Highlights**: GuideCQR는 다양한 데이터셋에서 경쟁력 있는 성능을 보여주었으며, 특히 CAsT-19 데이터셋에서 기존 방법들을 압도하는 결과를 기록했습니다. 이는 LLM(prompt) 방법을 여전히 능가하는 결과로, 안내 문서의 중요성을 다시 한번 강조합니다.



### The Better Angels of Machine Personality: How Personality Relates to LLM Safety (https://arxiv.org/abs/2407.12344)
- **What's New**: 최근 연구에 따르면 대형 언어 모델(LLMs)의 성격 특성이 이들의 안전 능력과 밀접한 관련이 있다는 것을 발견했습니다. 특히, MBTI-M 척도를 이용해 LLM의 성격 특성이 독성(toxicity), 프라이버시(privacy), 공정성(fairness) 등 안전 능력과 상관관계가 있음을 확인했습니다. 또한, LLM의 안전 정렬이 외향성(Extraversion), 현실 감각(Sensing), 판단력(Judging) 특성을 일반적으로 증가시킨다는 점을 발견했습니다.

- **Technical Details**: 이 연구에서는 MBTI-M 척도를 이용해 LLM의 성격 특성을 평가했습니다. 다양한 모델 크기에서 LLM의 성격을 평가해 신뢰할 수 있는 결과를 얻었으며, 성격 특성과 안전 능력 간의 관계를 조사했습니다. 결과적으로 LLM을 수정해 특정 성격 특성을 유도함으로써 모델의 안전 능력을 향상시킬 수 있음을 확인했습니다. 예를 들어, ISTJ 성격을 ISTP로 변경하면 프라이버시와 공정성 성능이 각각 약 43%와 10% 개선되었습니다.

- **Performance Highlights**: LLM의 특정 성격 특성을 수정함으로써 안전 성능을 대폭 향상시킬 수 있음을 발견했습니다. 예를 들어, ISTJ에서 ISTP로 인성 유도한 결과 프라이버시 성능이 약 43%, 공정성 성능이 약 10% 상대적으로 향상되었습니다. 또한, 어떤 성격 특성을 지닌 모델이 더 해킹에 취약한지에 대한 차이점도 밝혀졌습니다.



### Word Embedding Dimension Reduction via Weakly-Supervised Feature Selection (https://arxiv.org/abs/2407.12342)
- **What's New**: 이 논문에서는 단어 임베딩(word embedding)의 차원 축소를 위한 효율적이고 효과적인 약지도 특징 선택 방법인 WordFS를 제안합니다. WordFS는 두 가지 변형된 방법을 활용하여 특징 선택을 수행하며, 다양한 작업(단어 및 문장 유사성, 이진 및 다중 클래스 분류)에서 기존 차원 축소 방법보다 더 우수한 성능을 보입니다.

- **Technical Details**: WordFS 방법은 세 가지 단계로 구성됩니다: 1) 사후 처리(post-processing), 2) 특징 추출(feature extraction), 3) 약지도 특징 선택(weakly-supervised feature selection). 제안된 방법은 제한된 수의 단어 유사성 쌍을 활용하여 차원 축소를 수행합니다. 특징 선택 기준으로는 RFT 기반의 평균 제곱 오차(MSE)와 스피어만 순위 상관 계수를 사용하며, 선택된 특징 차원만을 남기고 나머지 차원은 버립니다.

- **Performance Highlights**: WordFS는 다양한 다운스트림 작업(문장 유사성 및 분류 작업)에서 더 낮은 계산 비용으로 기존 방법보다 더 나은 성능을 보입니다. 실험 결과, 제안된 방법은 단어 및 문장 유사성뿐만 아니라 분류 작업에서도 더 우수한 결과를 나타냈습니다.



### M2DS: Multilingual Dataset for Multi-document Summarisation (https://arxiv.org/abs/2407.12336)
- **What's New**: 다국어 문서 요약(Multi-document Summarization, MDS)에 대한 새로운 데이터셋인 M2DS가 소개되었습니다. 이 데이터셋은 BBC 기사를 기반으로 2010년부터 2023년까지 출판된 영문, 일본어, 한국어, 타밀어, 신할라어 등 5개의 언어로 문서-요약 쌍(document-summary pairs)을 포함합니다. 이는 기존 영어 중심의 MDS 연구에 비해 언어적 다양성을 고려한 데이터셋입니다.

- **Technical Details**: M2DS 데이터셋은 BBC 기사에서 전문적으로 작성된 요약을 기반으로 문서-요약 쌍을 포함하고 있습니다. 데이터 수집 절차와 전처리 과정을 투명하게 공개하며, 데이터셋 복제를 위한 링크와 스크립트도 제공합니다. 다국어 문서 요약 연구에 혁신적인 기회를 제공하기 위해 설계되었습니다.

- **Performance Highlights**: 최신 MDS 모델을 사용하여 M2DS 데이터셋에서 베이스라인 점수를 평가하였습니다. BERTSUM, BART, PEGASUS, T5와 같은 Transformer 아키텍처 기반의 모델들이 다언어적 데이터셋에서의 성능을 시험하는 데 사용되었습니다.



### MEDFuse: Multimodal EHR Data Fusion with Masked Lab-Test Modeling and Large Language Models (https://arxiv.org/abs/2407.12309)
- **What's New**: 이 논문에서는 전자 건강 기록(EHR)의 멀티모달 데이터를 효과적으로 통합하여 임상 예측을 개선하는 MEDFuse라는 새로운 프레임워크를 제안합니다. MEDFuse는 큰 언어 모델(LLMs)과 마스킹된 테이블 변환기(masked tabular transformers)를 활용하여 구조화된 실험실 테스트 데이터와 비구조화된 임상 노트를 통합합니다.

- **Technical Details**: MEDFuse는 두 가지 주요 소스에서 추출된 멀티모달 임베딩을 활용합니다. 첫째, 의료 텍스트에서 파인 튜닝된 LLMs를 사용하여 임상 텍스트의 임베딩을 생성합니다. 둘째, 구조화된 실험실 테스트 결과에 대해 마스킹된 테이블 변환기를 사용하여 임베딩을 생성합니다. 이 두 임베딩은 서로 분리된 정보를 효과적으로 통합하기 위한 'disentangled transformer module'에 입력됩니다. 이 모듈은 상호 정보 손실(mutual information loss)로 최적화되어, 모달리티별 및 모달리티 공유 정보를 분리한 후, 임상 노트에서 발생하는 노이즈와 중복성을 제거하여 유용한 공동 표현을 추출합니다.

- **Performance Highlights**: MEDFuse는 공공 MIMIC-III 데이터셋과 사내 FEMH 데이터셋에서 검증을 수행했으며, 10가지 질병에 대한 다중 라벨 분류 작업에서 90% 이상의 F1 점수를 달성했습니다. 이를 통해 임상 예측에서의 큰 잠재력을 보였습니다.



### Multimodal Reranking for Knowledge-Intensive Visual Question Answering (https://arxiv.org/abs/2407.12277)
- **What's New**: 이 논문에서는 지식 집약형 시각 질문 답변(Knowledge-intensive Visual Question Answering, KI-VQA) 모델의 순위 품질을 향상시키기 위해 멀티모달 재순위 매기기(multi-modal reranker) 모듈을 제안합니다. 이 모듈은 질문과 후보 지식 간의 교차 항목 상호작용을 통해 더 나은 관련성 점수를 모델링합니다. OK-VQA 및 A-OKVQA 데이터셋에서 실험한 결과, 멀티모달 재순위 매기기 모듈은 일관된 성능 향상을 보였습니다.

- **Technical Details**: 제안된 멀티모달 재순위 매기기 모듈은 멀티모달 정보(이미지 및 텍스트)를 활용하여 양질의 관련성 점수를 제공합니다. 기존의 두 타워(two-tower) 아키텍처는 이미지 패치에 기반하여 관련성 점수를 제공하는데, 이는 항상 신뢰할 만한 결과를 내지 못합니다. 멀티모달 재순위 매기기는 미리 훈련된 멀티모달 언어 모델을 사용하며, 질문과 후보 지식 항목 간의 교차 항목 상호작용을 수행합니다.

- **Performance Highlights**: OK-VQA와 A-OKVQA 데이터셋에서의 실험 결과, 멀티모달 재순위 매기기 모듈을 포함한 경우 일관된 성능 향상이 있음을 확인했습니다. 또한, 훈련 지식 후보들이 테스트 지식 후보와 유사하거나 더 많은 잡음을 포함할 때 성능이 더욱 향상된다는 점을 발견했습니다. 이는 향후 연구 방향에 중요한 통찰을 제공합니다.



### In-Context Probing Approximates Influence Function for Data Valuation (https://arxiv.org/abs/2407.12259)
- **What's New**: 이 논문은 데이터 평가(data valuation)에서 in-context probing(ICP)를 통해 데이터 선택을 최적화할 수 있음을 보여줍니다. 이 접근 방식은 대형 언어 모델(LLM)의 학습을 위해 고품질 데이터셋을 구성하는 데 중요한 역할을 합니다. 연구진은 ICP가 훈련 데이터의 가치를 평가하는 데 있어 영향 함수(influence functions)를 대략적으로 추정할 수 있음을 이론적으로 설명합니다.

- **Technical Details**: 논문에서는 트랜스포머 모델이 '암묵적' 경사 하강법(implicit gradient descent)을 수행함으로써 in-context 입력에 대해 영향을 미치는 방식을 설명합니다. 연구진은 ICP와 경사 기반 영향 함수가 훈련 데이터의 순위 매김 방식에서 유사한 결과를 나타낸다는 것을 실험적으로 입증했습니다. 또한, 두 가지 방법 중 하나로 선택된 데이터를 사용하여 미세 조정(fine-tuning)을 수행했을 때 모델 성능이 비슷하다는 것도 밝혀졌습니다.

- **Performance Highlights**: 각 방법으로 선택된 소규모 고순위 데이터 하위집합(subset)에서 미세 조정을 할 때, 대규모 데이터로 미세 조정한 것보다 성능이 더 나았습니다. 결과적으로 ICP와 영향 함수 방법으로 선택된 데이터가 유사한 성능을 발휘했습니다. 이는 특정 설정에서 ICP가 영향 함수 데이터 평가를 위한 대리 역할을 할 수 있음을 시사합니다.



### Lacuna Language Learning: Leveraging RNNs for Ranked Text Completion in Digitized Coptic Manuscripts (https://arxiv.org/abs/2407.12247)
Comments:
          Machine Learning for Ancient Languages, ACL 2024 Workshop, 15 August 2024

- **What's New**: 이 논문은 고대 사본 손상의 문제를 해결하기 위해 양방향 RNN 모델을 사용하여 코프틱 문자 예측을 시도합니다. 사본의 텍스트에 있는 공백을 복원(재건)하는 데 있어 초기 시도를 보여줍니다. 이 모델은 단일 문자 재건에서 72%의 정확도를 보였으나, 길이가 다른 공백(lacuna)에 대한 재건 성능은 37%로 떨어집니다.

- **Technical Details**: 코프틱 문자 예측을 위해 양방향 RNN 모델이 사용되었습니다. Coptic SCRIPTORIUM Corpora의 데이터를 활용하여 코프틱 문자 예측 모델을 학습시켰고, 원본 텍스트의 손상 및 유실된 부분을 복원하는 시스템을 구축했습니다. 모델 훈련에는 약 122만 개의 사히딕 코프틱 토큰이 사용되었습니다. RNN 기반의 아키텍처를 선택했으며, 데이터의 상대적 소량과 한정된 장거리 종속성으로 인해 Transformer 대신 RNN을 사용하였습니다.

- **Performance Highlights**: 단일 문자 재건에서 72%의 정확도를 달성했지만, 긴 공백(lacuna)을 재건할 때의 정확도는 37%로 낮았습니다. 이는 실제 사본 재건에는 한계가 있으나 학자들이 텍스트 재건의 가능성을 평가하고 순위를 매기는 데 도움이 될 수 있습니다.



### A Language Modeling Approach to Diacritic-Free Hebrew TTS (https://arxiv.org/abs/2407.12206)
Comments:
          Accepted at Interspeech24

- **What's New**: 이 논문에서는 히브리어 텍스트를 음성으로 변환하는 새로운 방법을 제안합니다. 기존 히브리어는 발음을 결정하는 데 중요한 역할을 하는 '니쿠드(Niqqud)'라는 발음 기호를 포함하고 있지만, 현대 히브리어는 이러한 기호를 거의 사용하지 않습니다. 이로 인해 텍스트에서 음성으로 정확하게 매핑하는 데 어려움이 있습니다. 제안된 방법은 발음 기호 없이 언어 모델링을 사용하는 것으로, 이 모델은 단어 조각 토크나이저(word-piece tokenizer)에 기반하여 작동합니다.

- **Technical Details**: 이 모델은 Residual Vector Quantization (RVQ)라는 방법을 사용하여 음성을 불연속적인 단위로 인코딩합니다. 그리고 텍스트를 어휘 조각 토크나이저(word-piece tokenizer)를 사용하여 인코딩합니다. 이를 통해 발음 기호의 예측 없이도 텍스트와 음성 간의 매핑을 학습할 수 있도록 합니다. 학습 데이터는 'in-the-wild' 약한 감독 데이터(weakly supervised data)를 사용해 최적화하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 콘텐츠 보존과 자연스러움 측면에서 기존의 발음 기호 기반 시스템보다 우수한 성능을 보여주었습니다. 생성된 음성의 샘플은 링크를 통해 확인할 수 있습니다. 코드와 데이터셋, 모델은 공개되어 있으며, 링크를 통해 접근 가능합니다: https://pages.cs.huji.ac.il/adiyoss-lab/HebTTS/.



### MASIVE: Open-Ended Affective State Identification in English and Spanish (https://arxiv.org/abs/2407.12196)
- **What's New**: 이 논문에서는 **정서 상태 식별(Affective State Identification, ASI)**의 문제를 새롭게 정의하였습니다. 감정 분석 분야에서 흔히 사용되는 제한된 감정 범주를 넘어, 영어와 스페인어로 작성된 Reddit 게시물에서 1,000개 이상의 고유한 정서 상태가 포함된 **MASIVE** 데이터셋을 수집하고 제공합니다.

- **Technical Details**: ASI 문제는 **마스킹된 영역 예측(masquered span prediction) 작업**으로 정의되며, 텍스트로 제공된 감정 경험에 대한 설명을 바탕으로 해당 설명에 대응하는 단어형 정서 상태를 모델이 생성하도록 훈련됩니다. 이를 위해 부트스트래핑 절차를 사용하여 새로운 정서 상태 라벨을 발견하고, 영어 및 스페인어로 작성된 자연스러운 감정 표현을 포함한 게시물을 수집했습니다. 총 4 라운드의 부트스트래핑을 수행하여 1600개의 영어 정서 상태 라벨과 1000개의 스페인어 정서 상태 라벨을 얻었습니다.

- **Performance Highlights**: 모델 성능 평가 결과, 소형 **미세 조정(multi-tuned)** 다중언어 모델이 기존 **대형 언어 모델(large language models, LLMs)** 보다 우수한 성능을 보였습니다. 또한, MASIVE 데이터셋을 사전 학습 데이터로 사용하면 기존 감정 인식 벤치마크에서도 성능이 향상됨을 확인했습니다. 마지막으로, 원어민이 작성한 텍스트를 사용하는 것이 ASI 작업의 성능 향상에 중요하다는 것을 실험을 통해 발견했습니다.



### Predicting Emotion Intensity in Polish Political Texts: Comparing Supervised Models and Large Language Models in a Resource-Poor Languag (https://arxiv.org/abs/2407.12141)
Comments:
          The Appendix is located at the very bottom of the manuscript

- **What's New**: 이 연구는 리소스가 부족한 언어 환경에서, 특히 폴란드어 정치 텍스트에서 감정 강도를 예측하기 위해 대형 언어 모델(LLMs)을 사용하는 방법을 탐구합니다. 10,000개의 소셜 미디어 텍스트로 이루어진 주석 데이터셋을 통해 여러 LLMs과 감독 학습 모델의 성능을 비교하였습니다.

- **Technical Details**: 연구는 전문가 심사위원들이 감정 강도를 평가하는 주석 데이터를 기반으로 훈련된 감독 모델과 LLMs의 성능을 비교합니다. LLMs는 방대한 데이터와 복잡한 패턴 인식 기능을 통해 텍스트에서 명확한 감정을 식별하는 데 유효성을 입증했습니다. 이 모델들은 폴란드어와 같은 리소스 부족 언어에서도 감정 강도 예측에 유망한 잠재력을 가지고 있음을 보였습니다.

- **Performance Highlights**: 감독 모델이 일반적으로 더 높은 정확성과 낮은 분산을 제공하지만, 데이터 주석에 필요한 자원 비용을 고려할 때 LLMs은 실행 가능한 대안이 될 수 있음을 발견했습니다. 이러한 결과는 감정 강도 예측 분야에서 LLMs의 유망한 가능성을 강조하며, 리소스 가용성과 연구 또는 실무 요구 사항에 따라 적절한 접근 방식을 선택하는 데 유익한 정보를 제공합니다.



### LLMs-in-the-loop Part-1: Expert Small AI Models for Bio-Medical Text Translation (https://arxiv.org/abs/2407.12126)
Comments:
          14 pages, 2 figures, 9 tables

- **What's New**: 본 연구는 의료 분야를 위한 기계 번역에 새로운 접근 방식을 소개합니다. 'LLMs-in-the-loop' 방법론을 통해 고품질의 맞춤형 신경 기계 번역 모델을 개발하였습니다. 이 방법론은 큰 언어 모델(LLMs)을 활용하여 작은, 특화된 모델을 고성능으로 만들 수 있음을 보여줍니다. 특히, 과학 기사, 합성된 임상 문서 및 의료 텍스트를 포함한 맞춤형 병렬 말뭉치를 사용했습니다.

- **Technical Details**: MarianMT 기본 모델을 사용하여 작은 의료 번역 모델을 개발했습니다. 이 과정에서 합성 데이터 생성, 엄격한 평가 및 에이전트 조정(agent orchestration)을 적용하여 성능을 향상시켰습니다. BLEU, METEOR, ROUGE, BERT와 같은 다양한 평가 척도를 사용하여 새로 소개한 의료 번역 테스트 데이터셋에서 성능을 측정했습니다.

- **Performance Highlights**: 우리의 MarianMT 기반 모델은 구글 번역, DeepL 및 GPT-4-Turbo보다 높은 성능을 보였습니다. 이 연구는 고품질, 도메인 지정 데이터로 미세 조정된 특화 모델이 일반 및 일부 대형 시스템보다 성능이 뛰어날 수 있음을 강조합니다.



### Better RAG using Relevant Information Gain (https://arxiv.org/abs/2407.12101)
Comments:
          4 page paper submitted to EMNLP

- **What's New**: 최근 발표된 논문에서는 대규모 언어 모델(LLM)의 메모리를 확장하기 위한 새로운 방법인 Dartboard 알고리즘을 제안했습니다. 이 알고리즘은 문맥창(context window)의 한계로 인한 정보 중복을 줄이면서도 관련성을 유지하는 것을 목표로 합니다.

- **Technical Details**: Dartboard 알고리즘은 쿼리에 대해 총 정보를 최대화하는 간단한 확률적 측정치를 최적화하는 방식으로 다양성을 자연스럽게 유도합니다. 이 방법은 기존의 유사도 매트릭(예: 코사인 유사도)을 사용한 K-근접 이웃(K-nearest-neighbors)와는 달리, 쿼리에 대해 최대한 많은 정보를 제공할 수 있는 결과를 도출하는 것을 목표로 합니다.

- **Performance Highlights**: Dartboard 알고리즘은 Retrieval-Augmented Generation Benchmark(RGB)에서 최첨단 성능을 발휘했으며, 기존의 관련성과 다양성을 직접 최적화하는 메트릭을 능가하는 성능을 보였습니다. 특히, end-to-end 질문 응답 시스템에서 뛰어난 성능을 입증했습니다.



### Identifying Speakers in Dialogue Transcripts: A Text-based Approach Using Pretrained Language Models (https://arxiv.org/abs/2407.12094)
Comments:
          accepted to INTERSPEECH 2024

- **What's New**: 본 논문에서는 새로운 접근법인 text-based speaker identification (SpeakerID)을 소개합니다. MediaSum 코퍼스에서 파생된 대규모 데이터셋을 사용하여 다양한 미디어 소스의 대화 기록에서 화자 명을 식별하는 혁신적인 방법을 제안했습니다.

- **Technical Details**: 제안된 방법은 Transformer 기반 모델을 사용하며, 문맥 내 단서를 활용하여 화자 명을 정확히 지정합니다. 이는 MediaSum 코퍼스에서 자동으로 고품질 대규모 훈련 데이터를 수집하는 새로운 방법론을 도입한 것입니다. 또한, Named Entity Recognition(NER)과 Text Matching을 통해 텍스트 기반 화자 식별을 수행하는 Graph Convolutional Networks를 포함한 다양한 모델 설계를 탐구합니다.

- **Performance Highlights**: 제안된 모델은 테스트 데이터셋에서 80.3%의 정밀도를 달성하여 text-based SpeakerID를 위한 새로운 벤치마크를 설정했습니다. 이는 데이터셋과 모델이 text-based SpeakerID에 적합함을 입증합니다.



### GoldFinch: High Performance RWKV/Transformer Hybrid with Linear Pre-Fill and Extreme KV-Cache Compression (https://arxiv.org/abs/2407.12077)
- **What's New**: GoldFinch라는 새로운 하이브리드 Linear Attention/Transformer 시퀀스 모델을 소개합니다. 이 모델은 시퀀스 길이에 비례하여 선형 시간과 공간에서 고도로 압축되고 재사용 가능한 KV-Cache를 효율적으로 생성하는 새로운 기법을 사용합니다. GoldFinch는 Finch (RWKV-6) 아키텍처의 향상된 버전을 기본으로 하고, 새로운 GOLD transformer를 추가로 쌓는 구조를 가지고 있습니다.

- **Technical Details**: GoldFinch는 1.5B 파라미터 클래스 모델까지 학습되었으며 Finch, Llama, GoldFinch 아키텍처의 모델링 성능이 크게 향상되었습니다. GoldFinch는 KV-Cache 크기를 크게 줄이며 메모리 사용량을 줄이는 여러 혁신적인 기술을 도입했습니다. 특히 'TokenCat'이라는 새로운 메커니즘을 사용하여 대단히 작은 전역 키 캐시를 생성하며, 이는 전통적인 transformer cache보다 756-2550배 더 작아집니다.

- **Performance Highlights**: GoldFinch는 Llama와 Finch를 모두 능가하는 성능을 보여주었습니다. 이는 특히 제한된 하드웨어에서도 매우 긴 컨텍스트 길이를 추론할 수 있게 합니다. 모든 트랜스포머 레이어에서 동일한 KV-Cache를 재사용하여 KV-Cache 크기를 모델 레이어의 총 수만큼 줄일 수 있으며, 전통적인 값 캐시가 아닌 입력 인덱스를 저장하고 값을 생성하는 방식으로 메모리 사용량을 크게 줄였습니다.



### NinjaLLM: Fast, Scalable and Cost-effective RAG using Amazon SageMaker and AWS Trainium and Inferentia2 (https://arxiv.org/abs/2407.12057)
- **What's New**: 이번 연구에서는 기존 검색 보강 생성(RAG) 기술을 향상시킨 새로운 접근법을 제시합니다. Amazon Web Services(AWS)의 Trainium 및 Inferentia2 AI 칩을 활용하여 대규모 언어 모델(LLMs)을 최적화하고 배포하는 방법을 설명합니다. 이 칩들은 유연성, 경제성, 효율성을 갖추고 있으며, LLMs의 성능을 향상시키는 데 중점을 둡니다. 또한, 도구 사용 개선, 인용 기능 추가, 맥락 편향으로 인한 환각 및 안전하지 않은 응답을 완화하는 방법도 제시합니다.

- **Technical Details**: 이번 연구에서는 AWS Trainium/Inferentia2 칩을 사용하여 대규모 언어 모델(LLM)을 호스팅하는 방법을 설명합니다. Meta의 Llama3-Instruct 70B 모델을 사용하여 다중 문서에 대한 복잡한 추론을 지원함으로써 향상된 RAG 시스템을 개발했습니다. 또한, 모델을 Fine-Tuning하여 인용 삽입, 멀티홉 쿼리 응답, 그리고 안전하지 않을 대답 및 환각을 방지합니다. vLLM 추론 엔진을 사용하여 PagedAttention 및 블록 수준 메모리 관리와 같은 혁신적인 메모리 관리 기술을 도입하여 메모리 사용을 최적화했습니다.

- **Performance Highlights**: 우리의 향상된 RAG 시스템은 Natural Questions(NQ) 데이터셋에서 62%, HotPotQA 데이터셋에서 59%의 정확도를 달성했습니다. 이는 DBRX 및 Mixtral Instruct 등 다른 모델을 능가하는 성과입니다. 또한, AWS Trainium 기반으로 Fine-Tuning된 Llama3 모델의 성능은 특히 HotPotQA에서 복잡한 멀티홉 추론이 필요한 경우에서도 매우 효율적임이 입증되었습니다.



### The Art of Saying No: Contextual Noncompliance in Language Models (https://arxiv.org/abs/2407.12043)
- **What's New**: 이번 연구는 언어 모델이 사용자 요청을 반드시 따라야 하는 것은 아니라는 점을 강조하며, 기존의 '안전하지 않은' 쿼리에 대한 거부에 초점을 맞추던 연구를 확장하여 보다 포괄적인 비순응성의 범위를 제안합니다. 새로운 비순응성 분류 체계를 도입하고, 이를 통해 1000개의 비순응성 프롬프트를 개발하여 현존하는 언어 모델을 평가했습니다.

- **Technical Details**: 새로운 비순응성 분류 체계는 '불완전(Incomplete)', '지원되지 않음(Unsupported)', '불확정(Indeterminate)', '인간화 요청(Humanizing requests)' 등의 다양한 카테고리를 포함합니다. 비순응성을 테스트하기 위해, 인간 검증된 비순응성 프롬프트 세트를 개발했으며, 이를 활용해 여러 첨단 모델들을 평가했습니다.

- **Performance Highlights**: GPT-4, Llama-3와 같은 최첨단 모델들은 '불완전', '지원되지 않음' 카테고리에서 잘못된 순응을 최대 30%까지 보여줬습니다. 이를 개선하기 위해 직접적인 파인튜닝(finetuning)은 과도한 거부와 일반 능력 저하를 초래할 수 있지만, 낮은 순위 어댑터(low rank adapters) 등의 파라미터 효율적 방법은 적절한 비순응성을 유지하면서도 일반적인 능력을 보존하는 데 효과적입니다.



### Exploring Advanced Large Language Models with LLMsu (https://arxiv.org/abs/2407.12036)
Comments:
          Keywords: Language Model Benchmarking, Pre-Trained LLM Comparison, LLM Performance Analysis, NLP Model Evaluation Tools, Public Dataset Inference for LLMs, BLEU and ROUGE Metrics for LLM, Open Source LLM Testing Tools, Large Language Model Evaluation Software, NLP Benchmarking Suite, Comprehensive LLM Evaluation Toolkit

- **What's New**: 본 튜토리얼 논문은 ChatGPT와 Gemini와 같은 Large Language Models (LLMs)의 발전과 한계를 탐구합니다. 특히 RAG (Retrieval Augmented Generation), PAL (Program-Aided Language Models), ReAct, 그리고 LangChain과 같은 기술을 소개하며, 이들의 통합을 통해 복잡한 작업 수행 및 다단계 추론에서 LLM의 성능과 신뢰성을 개선하는 방법을 제안합니다.

- **Technical Details**: 논문은 LLM의 주요 한계로서 지식의 시계적 한계(temporal knowledge cutoffs), 수학적 부정확성(mathematical inaccuracies), 그리고 잘못된 정보 생성에 대해 다루며, 이를 보완하기 위한 다양한 해결책을 소개합니다. RAG는 외부 데이터베이스와의 연계를 통해 정확성을 높이며, PAL은 정확한 계산을 위해 외부 코드 인터프리터를 활용합니다. ReAct와 LangChain은 복잡한 워크플로우를 관리하고 다양한 응용 프로그램들에 통합을 용이하게 합니다.

- **Performance Highlights**: 이 논문은 또한 Fine-tuning 전략, 특히 Instruction Fine-Tuning, LoRA (Low-Rank Adaptation), RLHF (Reinforcement Learning from Human Feedback), ReST (Reinforced Self-Training)와 같은 방법을 통해 LLM 성능을 특화된 용도에 맞춰 향상시키는 방법을 논의합니다. 이 전략들은 모델의 성능을 높이는 한편, 인간의 평가를 통해 모델 가중치를 갱신하고 적응력을 높입니다.



### Understanding Transformers via N-gram Statistics (https://arxiv.org/abs/2407.12034)
- **What's New**: 이번 연구는 Transformer 기반 대규모 언어 모델(LLM)의 예측을 단순한 N-gram 기반 통계 규칙으로 설명하려는 첫 시도로서, 이 규칙 집합이 Transformer 예측을 얼마나 잘 근사하는지 연구를 통해 여러 새로운 발견을 이루었습니다. 대표적으로, 전체 데이터를 Holdout Set 없이 과적합(overfitting)을 탐지하는 방법, 학습 과정에서 단순한 통계 규칙에서 복잡한 규칙으로의 진화 과정을 측정하는 방법 등을 제안합니다.

- **Technical Details**: 본 논문은 간단한 N-gram 기반 통계 규칙을 사용하여 LLM의 다음 토큰 예측을 설명하려 합니다. TinyStories(어린이 이야기) 데이터세트를 주로 사용했으며, Wikipedia에서도 일부 실험을 수행했습니다. 실험에 사용된 모델은 160M 파라미터로 TinyStories 데이터세트에서 4 에포크 동안 훈련되었습니다. N-gram ruleset을 통해 Transformer의 attention 메커니즘을 모방하여 예측을 설명합니다.

- **Performance Highlights**: N-gram 규칙이 점차 복잡해질수록 TinyStories 데이터세트에서 최대 78%의 top-1 정확도를 달성했습니다. 또한, 학습과정에서 발생하는 과적합을 감지하기 위한 새로운 방법을 제안했으며, 이는 Holdout 데이터 없이도 가능합니다. 저자들은 다양한 규칙 집합을용하여 LLM의 다음 토큰 예측을 잘 근사할 수 있음을 발견했습니다.



### TreeSeg: Hierarchical Topic Segmentation of Large Transcripts (https://arxiv.org/abs/2407.12028)
- **What's New**: 트리분할(TreeSeg)은 최신 연속 발화 임베딩 모델과 분리적 클러스터링(divisive clustering)을 결합한 혁신적인 주제 분할 접근 방식을 소개합니다. 이 방식은 자동 생성된 대본, 회의 기록 및 비디오 등을 이진 트리 형식으로 계층적으로 분할할 수 있게 해줍니다.

- **Technical Details**: TreeSeg은 학습이 불필요한 완전 비지도 학습(unsupervised learning) 모델입니다. 자동 대본에서 발화 임베딩을 추출한 후, 분리적 클러스터링을 사용하여 발화 사이의 전이점을 식별합니다. 이 방식은 대규모 입력을 효율적으로 처리하고, 여러 수준의 분할 해상도에서 높은 정확성을 유지합니다. 평가를 위해 TreeSeg는 ICSI 및 AMI 대형 회의 코퍼스와 신속하게 주석된 사용자 대본으로 구성된 소규모 코퍼스인 TinyRec에서 테스트되었습니다.

- **Performance Highlights**: TreeSeg는 ICSI 및 AMI 코퍼스에서 기존의 모든 기준 모델들을 능가하는 성능을 보여주었습니다. 또한 작은 크기의 새로운 코퍼스인 TinyRec에서도 우수한 성능을 발휘하며, 시간이 지남에 따라 점진적으로 확장될 예정입니다.



### The Pitfalls of Publishing in the Age of LLMs: Strange and Surprising Adventures with a High-Impact NLP Journa (https://arxiv.org/abs/2407.12026)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 학술 출판 분야에서의 남용 사례를 다룹니다. 최근 한 자연어 처리(NLP) 저널에서 발생한 사례를 통해 연구자 및 편집자가 LLM을 악용하는 문제를 조명합니다.

- **Technical Details**: 2023년 8월 말, 저자와 동료들이 저명한 NLP/컴퓨팅 언어학(CL) 저널에 논문을 제출했습니다. 그러나 11월 중순에 받은 리뷰 중 하나는 대부분 기계에 의해 작성된 것으로 보였습니다. 리뷰어가 LLM을 사용해 작성한 7페이지의 리뷰를 통해 LLM의 적절하지 않은 사용이 어떻게 학술적 평가를 왜곡하고 논문의 기밀성을 훼손하는지 보여줍니다.

- **Performance Highlights**: 논문 리뷰는 다음과 같이 요약할 수 있습니다: 논문의 독창성, 문헌 관계, 방법론, 결과 분석 및 그에 따른 실무적 응용 가능성에 대한 평가 등이 포함되어 있습니다. 다만, 리뷰의 내용 중 많은 부분이 표면적이고 명확성이 결여되어 있었으며, 통계적 유의성 및 질적 분석 부족 등의 문제점이 발견되었습니다.



### CMMaTH: A Chinese Multi-modal Math Skill Evaluation Benchmark for Foundation Models (https://arxiv.org/abs/2407.12023)
- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, LLMs)의 급격한 발전에 따라, 이들의 수학적 능력을 평가하는 연구가 계속 주목받고 있습니다. 기존의 MathVista 등 여러 데이터셋이 멀티모달 수학 능력을 평가하기 위한 벤치마크를 제안했지만, 중국어 K12 교육에 특화된 평가 도구와 데이터셋이 부족했습니다. 이를 해결하기 위해 'CMMaTH'라는 새로운 중국어 멀티모달 수학 스킬 평가 벤치마크가 제안되었습니다. CMMaTH는 약 23,000개의 K12 수학 관련 문제를 포함하며, 이는 현재까지 가장 큰 중국어 멀티모달 수학 문제 데이터셋입니다. 또한, 무료로 사용할 수 있는 평가 도구 GradeGPT를 함께 제공합니다.

- **Technical Details**: CMMaTH는 초등학교부터 고등학교 수준의 문제를 포함하며, 다양한 문제 유형, 시각적 요소, 상세한 지식 포인트, 표준 해답 등을 제공합니다. 표준 해답과 모델의 출력을 비교하여 일관성을 평가하는 경량 오픈 소스 도구 GradeGPT도 개발되었습니다. 이 도구는 상업적 모델 평가에 필요한 높은 비용을 피하고 안정적인 평가를 지원합니다. CMMaTH 데이터셋과 GradeGPT를 이용해 주류 멀티모달 대형 모델의 성능을 평가하고 종합적인 결과를 보고했습니다.

- **Performance Highlights**: 제안된 CMMaTH 벤치마크는 기존의 멀티모달 수학 벤치마크와 비교할 때 더 큰 다양성과 깊이 있는 추론을 제공합니다. 교육적 Q&A 시나리오를 보다 현실적으로 시뮬레이션하며, 더 다양한 문제 유형과 답변 형식을 포함하고 각 문제를 상세 지식 포인트와 대응되는 스킬로 주석 처리했습니다. GradeGPT를 통해 모델의 응답과 표준 해답의 일치도를 평가하며, 이를 통해 상업적 모델의 높은 비용과 불안정성을 해결했습니다.



### ITERTL: An Iterative Framework for Fine-tuning LLMs for RTL Code Generation (https://arxiv.org/abs/2407.12022)
- **What's New**: 최근의 연구에서는 대형 언어 모델(LLMs)이 인간의 지시를 이해하고 코드를 생성하는 데 탁월한 성능을 보임에 따라, LLM을 사용하여 RTL(레지스터 전송 논리) 코드를 생성하는 가능성을 탐구하고 있습니다. 기존의 방법들은 고정된 데이터셋으로 LLM을 미세 조정(fine-tune)하는데, 이는 LLM의 잠재력을 충분히 발휘하지 못하고 대량의 참고 데이터를 필요로 했습니다. 이를 해결하기 위해 단순하지만 효과적인 반복 훈련 패러다임인 ITERTL이 도입되었습니다. 각 반복(iteration)마다 이전 훈련 사이클에서 훈련된 모델에서 샘플을 추출하고, 이 새로운 샘플들을 훈련에 사용하는 방식으로 LLM의 분포 불일치를 줄이고 더 넓은 생성 영역을 탐험할 수 있게 합니다.

- **Technical Details**: ITERTL은 모델이 훈련 중에 반복적으로 샘플을 추출하고 훈련함으로써 분포 불일치를 줄이는 간단한 반복 훈련 방식입니다. 이는 전통적인 고정 데이터셋 방식과 달리, 훈련 과정에서 샘플을 업데이트하여 LLM의 탐험 범위를 넓히고 피드백을 풍부하게 합니다. 이 방식은 복잡한 강화 학습(RL) 대신, 간단한 반복 슈퍼바이즈드 학습(fine-tuning)으로 구현할 수 있습니다.

- **Performance Highlights**: 제안된 방법으로 훈련된 모델은 약 37%의 참고 데이터로도 최신 오픈 소스 모델과 경쟁하며, VerilogEval-human과 VerilogEval-machine 데이터셋에서 각각 42.9%와 62.2%의 pass@1 비율을 달성했습니다. 동일한 양의 참고 데이터에서 비반복 방법에 비해 각각 16.9%와 12.5%의 상대적 향상을 보였습니다. 이 연구는 제한된 데이터로 실질적인 RTL 코드 생성에 LLM을 활용하는 데 기여할 것입니다.



### Adaptive Draft-Verification for Efficient Large Language Model Decoding (https://arxiv.org/abs/2407.12021)
Comments:
          Under review of Neurips 2024

- **What's New**: ADED(Adaptive Draft-Verification for Efficient LLM Decoding)이라는 새로운 방법론이 소개되었습니다. 이 방법론은 대형 언어 모델(LLM) 디코딩 과정에서 효율성을 높이기 위해 고안되었습니다. 기존의 방법들은 각 토큰을 예측할 때마다 모델을 다시 로드해야 하는 비효율성과 높은 자원 요구 사항 때문에 실시간 애플리케이션에 적합하지 않았습니다. ADED는 이러한 문제를 해결하기 위해 적응형 초안 검증 프로세스를 도입하여 디코딩 속도를 대폭 향상시킬 수 있습니다.

- **Technical Details**: ADED는 tri-gram matrix 기반의 LLM 표현 방식을 사용하여 모델 출력 분포를 동적으로 근사화합니다. 이 방법은 adaptive draft construction과 verification을 포함하고 있으며, 동적으로 변화하는 LLM 출력에 따라 초안 분포를 조정합니다. 초안 생성 메커니즘은 탐색과 활용의 균형을 맞추기 위해 Monte Carlo Tree Search (MCTS) 알고리즘 영감을 받았습니다. 초안 생성 과정에서 얻어진 정보는 피드백 루프를 통해 LLM 표현 방식에 다시 반영되어 적응성과 자기 진화를 가능하게 합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋과 LLM 아키텍처에 대한 광범위한 실험을 통해, ADED가 디코딩 속도를 최대 2.5배 가속시키면서도 높은 정확도를 유지한다는 것을 입증했습니다. 또한, 기존 방법에 비해 평균 20%의 초안 수락률 향상을 보여, 실시간 애플리케이션에 적합한 성능을 제공한다는 것을 확인했습니다.



### SignSpeak: Open-Source Time Series Classification for ASL Translation (https://arxiv.org/abs/2407.12020)
Comments:
          6 pages, 2 figures, NeurIPS

- **What's New**: 이번 연구에서는 청각 및 언어 장애인 커뮤니티의 원활한 소통을 위한 저비용 ASL-to-speech 번역 장갑과 실시간 학습 데이터셋인 SignSpeak를 제안합니다. 이 데이터셋은 A-Z 알파벳과 1-10 숫자를 포함한 36개의 클래스를 포함하여 7200개의 샘플로 구성되어 있으며, 5개의 저비용 플렉스 센서를 사용하여 손가락 위치를 초당 36Hz의 간격으로 측정합니다. 후속 연구를 위한 오픈 소스 데이터셋, 모델, 장갑 디자인을 제공합니다.

- **Technical Details**: SignSpeak 데이터셋은 손가락마다 하나씩 총 5개의 플렉스 센서를 사용하여 각 제스처의 손가락 위치를 연속적으로 기록합니다. 이 데이터는 36Hz의 빈도로 기록되며, 아두이노 MEGA 2560을 통해 측정되었습니다. 데이터셋은 총 79 타임스텝으로 패딩되어 모든 입력이 일정한 배치 크기를 가지도록 합니다. LSTM, GRU, Transformer 모델을 수행하여, 모델 아키텍처는 두 층의 RNN과 MLP Softmax 분류 레이어로 구성되었습니다. 또한, Transformer 기반 모델 WaveGlove Encoder를 변형하여 사용했습니다.

- **Performance Highlights**: 우리의 최적 모델은 92%의 정확도를 달성했으며, 이는 이전의 ASL 시간 시리즈 분류 작업을 맞추거나 초과하는 성과입니다. Benchmark된 모델에는 2-layer LSTM, 2-layer GRU, 그리고 Transformer 기반의 WaveGlove Encoder가 포함되었습니다.



### DIM: Dynamic Integration of Multimodal Entity Linking with Large Language Mod (https://arxiv.org/abs/2407.12019)
Comments:
          Published on PRCV24

- **What's New**: 본 연구는 '멀티모달 엔티티 링크(Multimodal Entity Linking)'에 대한 새로운 접근 방식을 제안합니다. 기존 방법들이 엔티티 표현의 모호함과 이미지 정보 활용의 한계와 같은 문제에 직면하고 있는 반면, 우리는 ChatGPT를 사용하여 동적으로 엔티티를 추출하고 데이터셋을 강화하는 방법을 제안합니다. 이를 통해, LLM(Large Language Model)의 시각적 이해 기능을 포함한 블렌딩된 멀티모달(DIM) 정보를 지식베이스와 통합합니다.

- **Technical Details**: 제안된 방법 중 DIM(Dynamically Integrate Multimodal information with knowledge base)은 BLIP-2와 같은 LLM을 활용하여 이미지에서 엔티티와 관련된 정보를 추출하고, ChatGPT에서 동적으로 제공하는 엔티티 표현과 연결합니다. Wiki+, Rich+, Diverse+ 등의 동적 데이터셋을 만들어 벤치마킹 실험을 수행했습니다.

- **Performance Highlights**: 실험 결과, 제안된 DIM 방법은 기존의 대부분의 방법들을 능가했으며, 특히 Wiki+, Rich+, Diverse+와 같은 동적으로 강화된 데이터셋에서는 최신(State-of-the-Art, SOTA) 성능을 달성했습니다. 코드 및 수집된 데이터셋의 재현성을 위해 온라인에 공개되었습니다.



### Empirical Evaluation of Public HateSpeech Datasets (https://arxiv.org/abs/2407.12018)
Comments:
          18 pages, 12 tables, 1 algorithm pseudocode, 7 figures

- **What's New**: 이번 연구에서는 소셜 미디어 플랫폼에서 활용되는 다양한 공개적 혐오 발언(hate speech) 데이터셋을 종합적으로 평가했습니다. 이 연구는 기존 혐오 발언 감지 알고리즘의 훈련과 평가에 사용되는 데이터셋의 한계를 지적하며, 더 정확하고 신뢰할 수 있는 머신 러닝 모델을 개발하는데 기여하고자 합니다.

- **Technical Details**: 연구팀은 hatespeechdata.com에서 10개의 혐오 발언 데이터셋을 선택하여 평가했습니다. 이 데이터셋들은 트위터(Twitter), 유튜브(YouTube), 페이스북(Facebook) 등 다양한 소셜 미디어 플랫폼에서 수집되었으며, 데이터 전처리, 정규화, 레이블 이진화(binarisation), 통계 분석 및 딥러닝 기반 혐오 발언 분류기 개발 등을 포함한 종합적인 분석을 수행했습니다.

- **Performance Highlights**: 연구는 딥러닝 기반 혐오 발언 분류기에서 높은 성능을 보이는 간단하면서도 효과적인 기본 네트워크 아키텍처를 제안했습니다. 또한, 다양한 데이터셋을 교차 평가(Cross-domain evaluation)한 결과, 특정 플랫폼 데이터셋으로 훈련된 모형이 다른 플랫폼에서도 일반화되기 어려운 것을 발견했습니다. 데이터셋의 품질이 분류 성능에 가장 큰 긍정적 영향을 미친다는 사실을 실증적으로 입증했습니다.



### Follow-Up Questions Improve Documents Generated by Large Language Models (https://arxiv.org/abs/2407.12017)
- **What's New**: 이 연구는 대형 언어 모델(Large Language Models)이 사용자의 요청에 따른 후속 질문을 생성하는 방식의 영향을 조사합니다. 사용자는 AI가 생성해주길 원하는 문서에 대한 프롬프트를 제공했으며, AI는 문서를 생성하기 전에 사용자 요구를 명확히 하기 위해 질문을 생성했습니다. 사용자는 이러한 질문에 답한 후, 두 가지 문서 중 하나를 선택했습니다: 초기 프롬프트만으로 생성된 문서와 질문과 답변을 반영하여 생성된 문서입니다. 결과는 질문-응답 과정을 거친 문서가 더 선호되며, 사용자의 경험도 더 긍정적임을 보여주었습니다.

- **Technical Details**: 사용자는 짧은 텍스트 문서 생성을 요청하는 프롬프트를 제공했습니다. AI는 문서를 생성하기 전에 사용자 요구를 명확히 하기 위한 후속 질문을 생성하여, 사용자로 하여금 더 구체적인 요구 사항을 명확히 할 수 있도록 돕는 메커니즘을 사용했습니다. 사용자는 이러한 질문에 답변을 한 후 두 가지 유형의 문서에 대한 선호도를 표시했으며, 질문-답변 과정에 대한 피드백도 제공했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 질문-답변 과정을 거친 문서가 초기 프롬프트만을 사용한 문서보다 더 선호되었으며, 사용자는 이 과정을 통해 더 나은 문서 생성 경험을 가질 수 있었습니다. 이는 질문 생성과 사용자가 더 구체적인 요구를 명확히 함으로써 문서의 품질이 향상될 수 있음을 시사합니다.



### LLM-based Frameworks for API Argument Filling in Task-Oriented Conversational Systems (https://arxiv.org/abs/2407.12016)
- **What's New**: 이번 연구는 대화형 AI 시스템에서 필수적으로 수행되는 'Argument Filling' 작업을 향상시키기 위해 대형 언어 모델(LLMs)의 적용을 처음으로 탐구합니다. 작문자는 LLMs를 사용하여 기존 방법보다 더 단순하고 자율적인 'Argument Filling' 프레임워크를 구축하려고 합니다.

- **Technical Details**: 연구는 대화 히스토리와 사전 정의된 API 스키마를 기반으로 API의 인수(arguments)를 채우는 작업에 LLMs를 활용합니다. LLMs의 성능을 개선하기 위해, '지도 학습 미세 조정(Supervised Fine-tuning, SFT)'과 '거부 샘플링(Rejection Sampling, RS)'이라는 두 가지 기술을 사용합니다. 공개 소스 LLMs에 대해서는, 이 두 단계의 프레임워크를 사용하여 모델의 출력을 API 스키마에 맞게 조정합니다. 닫힌 소스 LLMs의 경우, 멀티 스텝 프롬프트 디자인을 사용하여 성능을 향상시킵니다.

- **Performance Highlights**: LLAMA-v1-7B 모델을 사용한 실험 결과, 제안된 지도 학습과 거부 샘플링 프레임워크는 기존의 단순한 SFT 베이스라인보다 우수한 성능을 나타냈습니다. 이는 대형 LLMs 대비 현저히 적은 크기의 모델로 더 나은 'Argument Filling' 성과를 보여줍니다.



### The Great AI Witch Hunt: Reviewers Perception and (Mis)Conception of Generative AI in Research Writing (https://arxiv.org/abs/2407.12015)
- **What's New**: 이번 연구에서는 Generative AI(GenAI)의 보조를 받은 글쓰기의 피어 리뷰 영향력을 조사했습니다. 연구는 최고 수준 HCI 학회의 17명의 피어 리뷰어와 함께 온라인 설문조사를 통해 진행되었습니다.

- **Technical Details**: GenAI 도구는 readability, language diversity, informativeness를 향상시키는 반면, 연구의 구체적인 세부 사항이나 작가의 반영된 통찰력은 부족한 경우가 많았습니다. 리뷰어들은 인간과 AI-보조 글쓰기를 구분하는 데 일관된 어려움을 겪었지만, 그들의 판단은 일관되었습니다. '인간적 터치'와 주관적 표현의 결여가 AI-보조 글쓰기에서 느껴졌습니다.

- **Performance Highlights**: GenAI 보조로 글쓰기가 더 읽기 쉽고 다양성이 있으며 정보량이 많아졌지만, 연구 세부사항과 저자의 반영된 통찰이 부족하게 느껴졌습니다. 리뷰어들은 글의 AI 사용 여부에 관계없이 일관된 감정을 유지했으며, AI-보조가 명확하고 매력적이라는 일관된 판단을 보여주었습니다. 그러나 GenAI 사용에도 불구하고 저자가 글쓰기 과정을 주도하는 것이 중요하다고 강조되었습니다.



### Specific language impairment (SLI) detection pipeline from transcriptions of spontaneous narratives (https://arxiv.org/abs/2407.12012)
Comments:
          15 pages, in Spanish language, 4 figures, 3 tables

- **What's New**: 이 연구는 아동의 특정 언어 장애(Specific Language Impairment, SLI)를 효율적으로 감지하기 위한 새롭게 제안된 방식을 소개합니다. 1063개의 인터뷰에서 스스로 이야기를 만든 내용을 사용하여 SLI를 감지할 수 있는 방법을 개발하였습니다. 이는 자연어 처리(Natural Language Processing, NLP) 분야에서 복잡한 주관적 변수를 피하고 아동의 성과와 관련된 정량적 지표에 집중하는 장점이 있습니다.

- **Technical Details**: 세 단계로 구성된 캐스케이드 파이프라인(pipeline)을 제안하였습니다. 첫 번째 단계에서는 특징 추출과 데이터의 차원 축소를 Random Forest(RF) 및 Spearman 상관 방법을 사용하여 수행합니다. 두 번째 단계에서는 첫 번째 단계에서 가장 예측력이 높은 변수를 로지스틱 회귀(logistic regression)를 통해 추정합니다. 마지막 단계에서는 최근접 이웃 분류기(nearest neighbor classifier)를 사용하여 아동의 자발적 이야기를 통해 SLI를 감지합니다.

- **Performance Highlights**: 이 접근 방식의 결과는 97.13%의 정확도로 SLI를 식별할 수 있음을 보여주었습니다. 이 과정에서는 반응의 길이, 발화의 품질, 그리고 언어의 복잡성 등이 강조됩니다.



### Generating Harder Cross-document Event Coreference Resolution Datasets using Metaphoric Paraphrasing (https://arxiv.org/abs/2407.11988)
Comments:
          Short Paper, ACL 2024

- **What's New**: 이 논문은 기존의 크로스-문서 이벤트 코리퍼런스 해소(Cross-Document Event Coreference Resolution, CDEC) 데이터셋들이 어휘적 다양성이 부족하여 실제 과제의 난이도를 제대로 반영하지 못하고 있음을 지적하며, 상징적 및 은유적 언어에 대한 CDEC 연구를 위해 ECB+META라는 새로운 데이터셋을 도입합니다. ECB+META는 기존의 ECB+ 데이터셋에 은유적 변환을 적용해 어휘적 다양성을 높였습니다.

- **Technical Details**: ECB+META는 ChatGPT를 이용해 ECB+ 문서 내 문장에 은유적 변환을 수행하고, 변환된 문장에서 원래의 이벤트 트리거를 반자동 방식으로 태깅하였습니다. 이를 통해 비싼 코리퍼런스 재주석 작업을 피할 수 있었습니다. 문장을 구문론적으로 한정하여 변환하고, 단어와 다단어 은유 표현을 사용해 여러 수준의 은유성을 가진 다양한 데이터셋을 생성하였습니다. 두 종류의 데이터셋인 ECB+META1(단어 변형)과 ECB+META2(다단어 변형)을 만들었습니다.

- **Performance Highlights**: 기존의 ECB+ 데이터셋에서 좋은 성능을 보였던 방법들이 ECB+META에서 성능이 낮아지는 것을 확인했습니다. 이는 어휘적 다양성과 문장의 복잡도가 증가할수록 CDEC의 난이도가 높아진다는 가설을 테스트한 결과로 해석할 수 있습니다. 이를 통해 CDEC 연구의 새로운 도전 과제를 제공합니다.



### Open the Data! Chuvash Datasets (https://arxiv.org/abs/2407.11982)
- **What's New**: 이번 논문에서는 저자들이 '추바시어(Chuvash language)'를 위한 4가지 종합 데이터셋을 소개합니다. 이는 언어 연구 및 기술 개발을 지원하고, 디지털 시대에서 추바시어의 보존과 홍보를 강화하는 중요한 자료들입니다.

- **Technical Details**: 제공되는 데이터셋은 크게 4가지입니다. 첫째, 약 390만 개의 문장을 포함한 '추바시어 모노링구얼 데이터셋(monolingual dataset)'입니다. 둘째, 약 140만 개의 문장 쌍을 담고 있는 '추바시어-러시아어 병렬 데이터셋(parallel dataset)'. 셋째, 약 20만 개의 문장 쌍이 있는 '추바시어-영어 병렬 데이터셋(parallel dataset)', 마지막으로 약 30,000개의 녹음 파일(총 38시간 분량)을 포함한 '추바시어 오디오 데이터셋(audio dataset)'입니다.

- **Performance Highlights**: 이들 데이터셋은 기계 번역, 언어 분석, 음성 인식과 같은 다양한 NLP 작업에 활용될 수 있습니다. 특히, 추바시어의 음성 인식을 위한 자동 음성 인식(ASR)과 음성 합성(TTS)을 위한 중요한 자원으로 사용될 것입니다. 기존의 Common Voice 데이터셋과도 쉽게 결합하여 파워풀한 모델 훈련이 가능하도록 설계되었습니다.



### The Role of Network and Identity in the Diffusion of Hashtags (https://arxiv.org/abs/2407.12771)
- **What's New**: 이 논문은 트위터(Twitter)에서 1,337개의 인기 해시태그가 어떻게 확산되는지에 대해 두 가지 사회적 요인의 역할을 종합적으로 조사한 최초의 연구입니다. 기존 연구는 주로 소셜 네트워크의 속성 등의 단일 요인만을 고찰했지만, 이 연구는 1) 트위터 소셜 네트워크의 토폴로지와 2) 사용자의 인구통계학적 정체성이 해시태그 확산에 미치는 영향을 동시에 분석합니다.

- **Technical Details**: 연구는 세 가지 모델을 사용하여 해시태그의 확산을 시뮬레이션했습니다. 첫째, 'Network-only' 모델은 수정된 선형 임계값 모델을 사용하여 해시태그가 네트워크를 통해 퍼지는 방식을 설명합니다. 둘째, 'Identity-only' 모델은 관련 정체성을 공유하는 사용자 사이에서 해시태그가 확산되는 방식을 설명합니다. 셋째, 'Network+Identity' 모델은 두 가지 요인을 모두 포함합니다. 또한, 해시태그 확산의 열 가지 속성을 기반으로 모델의 성능을 평가했습니다.

- **Performance Highlights**: 네트워크와 정체성을 결합한 모델은 열 가지 속성의 종합 지수를 가장 잘 재현했습니다. 그러나 각 해시태그 속성에 따라 필요한 사회적 요인이 다릅니다. 예를 들어, 네트워크와 정체성을 결합한 모델은 해시태그의 인기를 가장 잘 예측했지만, 네트워크만 사용한 모델은 확산 성장 예측에, 정체성만 사용한 모델은 채택자 구성 예측에 더 우수한 성능을 보였습니다. 또한, 인종이나 지역 정체성을 표현하는 해시태그, 스포츠 관련 해시태그 및 기타 빠르게 성장하거나 느리게 성장하는 커뮤니케이션 수단 해시태그에서 네트워크와 정체성을 결합한 모델이 더욱 뛰어난 성능을 보였습니다.



### TTSDS -- Text-to-Speech Distribution Scor (https://arxiv.org/abs/2407.12707)
Comments:
          Under review for SLT 2024

- **What's New**: 최근 발표된 많은 텍스트-음성 변환(Text-to-Speech, TTS) 시스템은 실제 음성에 가까운 오디오를 생성합니다. 그러나 이러한 새로운 아키텍처, 접근 방식 및 데이터셋으로 얻은 결과를 해석하기 위해 TTS 평가가 재검토되어야 합니다. 본 논문에서는 음성 프로소디(prosody), 화자 정체성(speaker identity), 명료도(intelligibility) 등 여러 요소의 결합으로서 합성 음성의 품질을 평가하는 접근 방식을 제안합니다. 또한, 2008년부터 2024년까지 개발된 35개의 TTS 시스템을 벤치마킹했습니다.

- **Technical Details**: 우리의 접근 방식은 각 요소의 상관관계를 얻고, 이를 실제 음성 데이터셋 및 잡음 데이터셋과의 거리로 측정함으로써 합성 음성이 실제 음성에 얼마나 유사한지를 평가합니다. 구체적으로, 본 벤치마크에는 명료도(WER 기반), 프로소디(피치 및 지속 시간 기반), 화자 정체성(d-vectors 및 WeSpeaker 기반), 환경 조건(SNR 및 리버베레이션) 등의 요소가 포함됩니다. 각 요소에 대해 프레임 수준의 자체 지도 학습(SSL) 표현을 사용했습니다.

- **Performance Highlights**: 우리의 요소별 점수 평균은 인공지능 시스템의 실제 성능과 강하게 상관관계가 있음을 발견했습니다(상관계수 0.60에서 0.83 사이). 이는 MOS 예측의 성능(상관계수 0.05에서 0.85 사이)보다 일관된 결과를 보여줍니다. 또한, 시간 경과에 따른 인간 평가자의 우선순위가 변화하고 있음을 관찰했으며, 환경 요소는 초기 시스템에서는 더 중요했지만, 최신 시스템에서는 프로소디가 더 중요한 요소로 떠오르고 있습니다.



### Reducing Biases towards Minoritized Populations in Medical Curricular Content via Artificial Intelligence for Fairer Health Outcomes (https://arxiv.org/abs/2407.12680)
Comments:
          Under review

- **What's New**: 의료 교육에서 지속적으로 퍼지고 있는 편향된 정보(bisinformation)를 줄이기 위한 새로운 기계 학습 이니셔티브, BRICC에 대해 소개합니다. BRICC는 의료 텍스트의 편향성을 체계적으로 식별하고 표시하는 최초의 시스템으로, 이후 전문가의 검토를 통해 보다 효율적으로 편향된 정보를 해결할 수 있도록 합니다.

- **Technical Details**: BRICC는 약 12000 페이지의 교육 자료를 포함한 데이터세트를 가지고 있으며, 성별, 성, 나이, 지리, 민족 및 인종 등에 관한 포괄적인 코딩 지침에 따라 의료 전문가가 철저하게 주석을 달아놓았습니다. 이 데이터를 사용하여 다양한 편향 감지 모델을 훈련, 검증 및 테스트했습니다. 모델에는 이진(type-specific) 분류기, 일반 편향 분류기, 편향 유형별 분류기를 결합한 앙상블 모델 및 multitask learning(MTL) 모델이 포함됩니다.

- **Performance Highlights**: 일반 편향 검출에서 이진 분류기는 최대 0.923의 AUC를 달성했으며, 이는 기준선 대비 27.8% 향상된 수치입니다. 레이스 편향 검출에서 MTL 모델이 F1 점수 측면에서 다소 개선을 보여주었으나, 특화된 이진 분류기의 성능을 능가하지는 못했습니다.



### AudienceView: AI-Assisted Interpretation of Audience Feedback in Journalism (https://arxiv.org/abs/2407.12613)
Comments:
          Accepted at CSCW Demo 2024. 5 pages, 2 figures

- **What's New**: AudienceView는 저널리스트들이 청중 피드백을 보다 효과적으로 분류하고 해석할 수 있도록 도와주는 온라인 도구입니다. 이 도구는 대형 언어 모델(LLMs)을 활용하여 청중의 댓글에서 주제와 테마를 식별하고, 이를 구체적인 댓글과 연계하며, 댓글의 감정과 분포를 시각화하는 방법을 제공합니다. 이는 PBS의 Frontline 팀과 협력하여 개발되었으며, 특히 YouTube에 호스팅된 비디오 저널리즘을 대상으로 합니다.

- **Technical Details**: AudienceView는 Backend와 Frontend 구성요소로 나뉘어 있으며, 사용자 인터페이스는 Streamlit 프레임워크로 구현되었습니다. YouTube 댓글 수집과 감정 분석, 주제 군집화를 포함한 백엔드 프로세스를 통해 사전 준비가 이루어집니다. 댓글의 주제는 GPT-4 모델을 활용하여 생성되며, 사용자는 대시보드에서 보다 직관적으로 피드백을 확인할 수 있습니다. HDBSCAN 및 UMAP과 같은 차원 축소 및 군집화 알고리즘을 사용하여 댓글의 주제를 탐색할 수도 있습니다.

- **Performance Highlights**: AudienceView는 2013년 8월부터 2024년 1월까지의 250개 Frontline 다큐멘터리에 적용되어 약 599,000개의 댓글을 분석하였습니다. 이 도구는 전통적인 질적 분석 도구와 달리 최신 NLP 기술을 사용하여 자동화된 분석을 제공하며, 저널리스트들이 독자의 피드백을 쉽게 이해하고 활용할 수 있도록 돕습니다.



### Abstraction Alignment: Comparing Model and Human Conceptual Relationships (https://arxiv.org/abs/2407.12543)
Comments:
          14 pages, 3 figures

- **What's New**: 이 논문에서는 머신러닝(ML) 모델의 학습된 추상화(Abstraction)가 인간의 추상화와 얼마나 일치하는지 평가하는 새로운 방법론인 '추상화 정렬'(Abstraction Alignment)을 소개합니다. 기존의 방법들이 모델이 학습한 개념들을 개별적으로 분석하는 반면, 이 새로운 방법론은 개념들 간의 관계를 포함하여 종합적으로 평가합니다.

- **Technical Details**: 추상화 정렬을 통해 인간의 추상화를 DAG(방향성 비순환 그래프)으로 표현하고, 이를 모델 출력과 비교하여 모델의 불확실성을 얼마나 잘 설명하는지 측정합니다. 이를 통해 불확실성 정렬(uncertainty alignment)과 개념 혼동(concept confusion) 등의 지표를 정의하여 모델의 추상화 상태를 다각도로 평가합니다. 이 방법론은 이미지 모델, 언어 모델, 의료 데이터셋 분석 등 다양한 평가 작업에서 사용됩니다.

- **Performance Highlights**: 이미지 모델 해석 시, 추상화 정렬은 인간 지식과의 일치 정도에 따라 오류를 구분해, 겉보기엔 문제인 오류들이 사실은 세분화 부족에서 기인한 것임을 식별합니다. 또한, 언어 모델의 품질 벤치마크를 확장하여 모델의 구체성을 다양한 언어 표지로 측정하며, 의료 도메인에서 데이터셋의 카테고리화 문제를 노출하고 기존 인간 추상화를 개선할 기회를 제공합니다.



### Characterization of Political Polarized Users Attacked by Language Toxicity on Twitter (https://arxiv.org/abs/2407.12471)
Comments:
          This work has been accepted by 2024 Conference on Computer Supported Cooperative Work and Social Computing (CSCW2024). Association for Computing Machinery (ACM), New York, NY, USA

- **What's New**: 소셜 미디어상에서의 언어 독성(dynamic language toxicity) 연구는 정치적 시나리오, 특히 미국 대통령 선거와 같은 경우에서 중요한 역할을 합니다. 이번 연구는 약 5억 개의 트위터 게시물을 분석해, 좌파(left) 사용자가 우파(right) 및 중앙(center) 사용자보다 더 많은 독성 답글을 받는다는 사실을 발견하였습니다.

- **Technical Details**: 이번 연구는 2020년 2월 20일부터 2022년 5월 30일까지 영어로 작성된 542,212,429개의 트윗을 수집하여 분석하였습니다. 이를 위해 Allsides.com에서 제공한 정치적으로 성향이 나뉜 URL 도메인 리스트를 사용해 각각의 답글을 좌파, 우파, 중앙으로 분류하였습니다. 연구는 Perspective API를 이용해 언어 독성을 계산했으며, 트윗의 독성 점수는 0과 1 사이의 확률 점수로 평가했습니다.

- **Performance Highlights**: 분석 결과, 좌파 사용자가 우파 및 중앙 사용자보다 독성이 높은 답글을 더 많이 받는다는 것이 드러났습니다. 또한, 우파 및 중앙 사용자들이 받는 답글의 최대 독성 분포는 유사했으나(p>0.05), 좌파 사용자는 다른 분포를 보였습니다(p<0.05). 이로 인해 좌파 사용자가 트위터 상에서 더 많은 독성 언어의 공격을 받는다는 것을 명확히 확인할 수 있습니다.



### Across Platforms and Languages: Dutch Influencers and Legal Disclosures on Instagram, YouTube and TikTok (https://arxiv.org/abs/2407.12451)
Comments:
          Accept for publication at the 16th International Conference on Advances in Social Networks Analysis and Mining -ASONAM-2024

- **What's New**: 이 연구는 네덜란드 미디어 권한으로 공식 등록된 150명의 인플루언서를 대상으로, 인플루언서가 법적 기준에 따라 콘텐츠를 적절히 공개하는지 여부를 평가하는 투명한 방법론을 제안합니다. 데이터 세트는 인스타그램, 유튜브, 틱톡과 같은 여러 플랫폼에서 수집된 292,315개의 게시물을 포함합니다.

- **Technical Details**: 연구는 네덜란드 미디어 법에 따라 2022년 도입된 등록 의무에 따라 등록된 인플루언서의 데이터를 사용하여 법적 기준에 따른 광고 공개의 준수 여부를 평가합니다. 데이터 수집은 각 플랫폼의 API(API: Application Programming Interface)를 통해 이루어졌으며, 수집된 데이터에서 영어와 네덜란드어로 작성된 게시물만을 분석에 포함했습니다. 법적 공개 여부는 네덜란드 광고 코드(Dutch Advertising Code)를 기반으로 '녹색 공개(Green Disclosure)'와 '황색 공개(Yellow Disclosure)'로 구분되었습니다.

- **Performance Highlights**: 연구 결과, 등록된 네덜란드 인플루언서는 전체 콘텐츠의 5.63%만을 공개하는 것으로 나타났으며, 더 큰 인플루언서가 더 많이 공개하는 것은 아니었습니다. 또한, 플랫폼마다 공개 관행과 참여도(engagement) 차이가 존재했습니다. 예를 들어, 인스타그램에서는 'Paid partnership' 태그를 사용하여 녹색 공개를 더 많이 수행한 반면, 틱톡과 유튜브에서는 황색 공개가 더 많이 발견되었습니다.



### NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models (https://arxiv.org/abs/2407.12366)
Comments:
          Accepted to ECCV 2024

- **What's New**: 이 연구는 Vision-and-Language Navigation(VLN) 작업에 대형 언어 모델(LLMs)을 활용하여 기존의 특화된 모델과 성능 격차를 줄이는 새로운 방법을 제안합니다. 이 연구는 LLMs의 해석력과 언어적 내비게이션 추론 능력을 유지하면서 시각 콘텐츠를 고정된 LLM에 맞추어 시각적 관찰 이해를 포함하는 접근법을 사용합니다.

- **Technical Details**: 새로운 시스템 NavGPT-2는 두 가지 기존 접근법(제로샷 및 파인튜닝)의 중간 지점을 찾으려 합니다. 다중 이미지 인식을 통해 VLN 작업에 적응하도록 InstructBLIP 아키텍처를 기반으로 구축되었으며, GPT-4V로 단계별 내비게이션 추론 데이터를 구성하여 시각적 지침 튜닝을 수행합니다. VLM(Vision-Language Model) 잠복 공간(latent space)을 사용하여 장기 내비게이션 기록을 추적하고 효과적인 역추적을 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 데이터 효율성을 입증하며, LM 기반 에이전트와 최신 VLN 전문 모델 간의 성능 격차를 제거하였습니다. 또한, 모델이 각 내비게이션 결정의 이유를 명확하게 설명할 수 있도록 하여, 실용적이고 상호작용 가능한 VLN 에이전트를 구축하는 데 필수적인 능력을 제공합니다.



### ProcTag: Process Tagging for Assessing the Efficacy of Document Instruction Data (https://arxiv.org/abs/2407.12358)
- **What's New**: 이 논문에서는 문서 지시 데이터 평가를 위해 ProcTag라는 새로운 메소드를 제안합니다. ProcTag는 지시 텍스트 자체보다 지시 실행 과정을 태그를 통해 평가하는 독창적인 접근 방식을 사용합니다. 또한 문서 레이아웃 정보를 효과적으로 반영하는 DocLayPrompt라는 반구조적 레이아웃 인식 문서 프롬프팅 전략을 제안합니다.

- **Technical Details**: ProcTag는 GPT-3.5의 체인-오브-쏘트(chain-of-thought) 추론 능력을 활용하여 문서 지시 실행 과정을 코드로 표현하고 이를 태그로 구분합니다. 이를 통해 지시 데이터의 다양성과 복잡성을 측정합니다. DocLayPrompt는 문서 레이아웃 정보를 효과적으로 표현하여 기존 프롬프팅 기법을 능가합니다.

- **Performance Highlights**: ProcTag 기반 샘플링을 통해 생성된 문서 데이터셋에서 평가하면, 전체 데이터셋의 30.5%만으로도 100%의 효율성을 달성할 수 있습니다. 이는 기존 데이터 평가 메소드와 무작위 샘플링보다 훨씬 높은 성능을 보입니다. 실험 결과, ProcTag는 문서 VQA를 위한 LLMs 및 MLLMs 교육 시 효과적인 문서 지시 데이터 평가를 보여줍니다.



### Questionable practices in machine learning (https://arxiv.org/abs/2407.12220)
- **What's New**: 최근 AI 연구에서 보고된 성과들이 종종 믿을 수 없다는 문제를 다룬 논문이 발표되었습니다. 이 논문은 대형 언어 모델(LLMs)의 평가에 있어 의문스러운 연구 관행(questionable research practices, QRPs)과 재현성 없는 연구 관행(irreproducible research practices, IRPs)을 강조하고 있습니다.

- **Technical Details**: 이 논문에서 분석된 43개의 QRPs는 '오염(contamination)', '체리피킹(cherrypicking)', '잘못된 보고(misreporting)'로 구분됩니다. 예를 들어, 테스트 데이터 정보가 모델 학습 과정에서 사용되거나, 실험 설정에서 모델의 성능을 의도적으로 개선하는 등의 문제가 언급되었습니다. 또한, 데이터셋을 공유하지 않는 등의 IRPs도 강조되었습니다.

- **Performance Highlights**: 연구에서는 LLM 평가에서 발생할 수 있는 여러 구체적인 사례들을 제시했습니다. 특히, 심리학 연구에서 사용되던 QRPs와 연구자의 자유도(researcher degrees of freedom, RDOFs)의 개념을 차용하여 ML 연구에도 적용 가능한 다양한 방법론적 문제를 탐구하였습니다.



### GPT-4V Cannot Generate Radiology Reports Y (https://arxiv.org/abs/2407.12176)
Comments:
          24 pages, 3 figures, code: this https URL

- **What's New**: 이번 연구에서는 GPT-4V의 흉부 엑스레이(Radiology report) 보고서 작성 자동화 능력을 체계적으로 평가했습니다. MIMIC-CXR와 IU X-Ray 데이터셋을 사용하여, 다양한 프롬프트 전략을 통해 직접 보고서를 생성해본 결과, 언어적 측면(Lexical Metrics)과 임상적 효능(Clinical Efficacy Metrics) 모두에서 매우 낮은 성능을 보였습니다.

- **Technical Details**: 연구는 세 가지 실험으로 구성되었습니다. 첫 번째로, 흉부 엑스레이 이미지에서 직접 보고서를 생성하는 실험에서는 Zero-shot, Contextual Enhancement, Chain-of-Thought (CoT), Few-shot in-context learning 등 다양한 프롬프트 전략을 활용했습니다. 두 번째로, 의료 이미지 추론 능력을 평가한 결과, GPT-4V가 의료 조건을 정확히 식별하는 데 실패하였습니다. 세 번째로, 매개 변수를 조건으로 제공하여 보고서를 생성하는 실험에서는 일부 임상 효능이 향상되었으나, 보고서의 자연스러움과 정확성은 튜닝된 LLaMA-2 모델보다 낮았습니다.

- **Performance Highlights**: GPT-4V는 end-to-end 흉부 엑스레이 보고서 생성과 decomposed task(의료 이미지 추론과 텍스트 합성) 모두에서 낮은 성능을 보였습니다. 모델이 의료 이미지를 의미 있게 해석하지 못하고, 튜닝된 LLaMA-2 모델보다 전체적인 성능이 떨어졌습니다. 결과적으로, GPT-4V를 현 의료 체계에 적용하는 것은 실용적이지 않다는 결론을 내렸습니다.



### Private prediction for large-scale synthetic text generation (https://arxiv.org/abs/2407.12108)
Comments:
          12 pages main text + 15 pages appendix

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 사용하여 differential privacy를 만족하는 synthetic text를 생성하는 새로운 방법을 제시했습니다. 기존의 방식이 소수(<10)의 예제를 생성하는 데 머물렀다면, 이번 접근 방식은 수천 개의 high-quality한 synthetic data point를 생성할 수 있어 응용 범위가 크게 확장되었습니다.

- **Technical Details**: 본 연구는 크게 세 가지 혁신적인 알고리즘 요소를 통해 성능을 개선했습니다. 첫째, softmax 레이어의 sampling과 exponential mechanism의 동등성을 활용한 개선된 privacy 분석 방법을 도입했습니다. 둘째, private selection mechanism을 통해 compute 효율성을 높였습니다. 마지막으로, sparse vector 기술을 사용하여 민감 데이터 없이도 public prediction을 활용할 수 있게 하였습니다. 이러한 방법들을 통해 구문의 구조를 유지하면서, synthetic text가 높은 품질을 유지할 수 있게 되었습니다.

- **Performance Highlights**: 실험 결과, 기존 최첨단 private prediction 방법보다 수백 배 많은 synthetic 데이터를 생성하면서도 유사한 수준의 privacy 보장을 유지하는 데 성공했습니다. 이는 downstream classification 및 extraction tasks에서 데이터를 사용했을 때 기존 방법보다 더 높은 accuracy를 보여주었습니다. 또한, 수천 개의 training 예제를 생성함으로써 downstream model의 fine-tuning에도 충분히 활용될 수 있음을 입증했습니다.



### LiteGPT: Large Vision-Language Model for Joint Chest X-ray Localization and Classification Task (https://arxiv.org/abs/2407.12064)
Comments:
          Preprint, 19 pages

- **What's New**: Vision-language 모델은 다양한 작업에서 뛰어난 성능을 발휘해왔지만, 의료 영상에서의 적용은 여전히 미진한 상태입니다. 이번 연구에서는 의료 영상 분석을 위한 통합 프레임워크인 LiteGPT를 제안합니다. 여러 사전 학습된 비주얼 인코더(visual encoders)를 활용하여 정보의 질을 풍부하게 하고 성능을 향상시켰습니다. 또한, 본 연구는 흉부 X-ray에서 질병의 위치와 분류를 동시에 수행하는 새로운 작업을 수행하는 첫 연구입니다. VinDr-CXR 데이터셋에서 이미지 분류 작업에서 새로운 최첨단 성능을 달성하였으며, 모든 코드 및 모델은 온라인에서 공개되어 있습니다.

- **Technical Details**: LiteGPT는 흉부 X-ray 이미지를 대상으로 중요한 병변을 식별하고 분류하는 새로운 작업을 도입합니다. 본 연구에서는 두 개의 최첨단 비주얼 인코더인 BiomedCLIP와 PubMedCLIP를 사용하여 의료 이미지를 더욱 상세하고 다양하게 표현합니다. 이 인코더들은 대규모 의료 이미지-텍스트 데이터셋에서 사전 학습되어 높은 품질의 시각적 특징을 추출합니다. 이러한 특징들은 대규모 언어 모델인 Llama 2의 언어 공간으로 투영되어, 시각적 및 텍스트적 모달리티의 융합을 통해 의료 이미지의 상세하고 정확한 설명을 생성합니다.

- **Performance Highlights**: 제안된 방법은 기존 방법들보다 뛰어난 성능을 보였으며, 위치 지정 정확도와 분류 성능에서 현저한 개선을 보고하였습니다. 또한, 다양한 구성 요소의 기여를 이해하기 위한 ablation study를 수행하여, 모델의 각 부분이 전체 성능에 미치는 영향을 분석하였습니다. 이 연구는 VinDr-CXR 데이터셋에서 최첨단 성능을 달성했으며, 의료 이미지 분석의 향상을 위한 통합된 비전-언어 프레임워크의 가능성을 입증했습니다.



### Dating ancient manuscripts using radiocarbon and AI-based writing style analysis (https://arxiv.org/abs/2407.12013)
Comments:
          16 pages of main article, 103 pages of supplementary materials, currently under review after having been submitted 20 July 2023

- **What's New**: 고대 필사본의 연대를 측정하는 것은 아이디어의 진화를 재구성하는 데 필수적입니다. 특히 사해 두루마리(Dead Sea Scrolls)에 있어서 중요합니다. 이 연구에서는 첨단 AI 기반의 날짜 예측 모델인 'Enoch'을 제시합니다. 이 모델은 새로운 방사성 탄소 연대측정(14C) 샘플을 기반으로 훈련되었습니다.

- **Technical Details**: Enoch은 필체 스타일 기술(descriptors)과 Bayesian ridge regression을 사용합니다. 연구의 도전 과제는 방사성 탄소로 연대측정된 필사본이 적은 반면, 현재의 기계 학습 모델은 많은 양의 훈련 데이터를 요구한다는 점입니다. 필사본의 필체 특징 벡터(feature vector)와 Bayesian ridge regression을 결합하여 예측을 수행하였습니다. 또한 이미지의 멀티스펙트럼 밴드(multispectral band) 영상을 활용하여 비네라이제이션(binarization) 기법을 적용해 잉크와 배경을 효과적으로 분리했습니다.

- **Performance Highlights**: Leave-one-out 검증을 통해 Enoch은 27.9 ~ 30.7년의 MAE를 나타내며, 135개의 새로운 필사본에 대해 연대를 추정한 결과 79%의 샘플이 실질적이라는 평가를 받았습니다. 또한, 기존의 필사학적 연대 추정보다 더 정교한 연대 추정을 보여주었습니다.



### People will agree what I think: Investigating LLM's False Consensus Effec (https://arxiv.org/abs/2407.12007)
Comments:
          Under review

- **What's New**: 신경망 기반의 대형 언어 모델(LLMs)이 상호작용 시스템의 커뮤니케이션에 널리 사용되고 있습니다. 이러한 시스템의 사용성을 해칠 수 있는 'False Consensus Effect (FCE, 허위 합의 효과)'가 LLMs에 존재하는지 조사되었습니다. 두 가지 연구를 통해 LLMs가 FCE를 가지고 있으며, 다양한 프롬프트 스타일(prompting)에 따라 FCE의 정도가 달라지는 것을 밝혀냈습니다.

- **Technical Details**: 첫 번째 연구에서는 LLMs가 FCE를 가지는지를 살펴보기 위해 심리학 실험을 채택하여 혼재된 편향(confounding biases)과 일반적인 상황을 다루었습니다. 두 번째 연구에서는 프롬프트 스타일이 FCE에 어떻게 영향을 미치는지를 탐구했습니다. 제공된 정보의 관련성과 추론 과정의 깊이라는 두 가지 측면에서 프롬프트 스타일의 변화를 실험했습니다.

- **Performance Highlights**: 연구 결과, LLMs가 일반적인 상황에서 FCE를 보이는 것으로 나타났습니다. 또한 제공된 정보가 FCE에 영향을 미칠 수 있으며, 반복적인 추론 과정이 FCE를 감소시킬 수 있음을 발견했습니다.



### The Kolmogorov Complexity of Irish traditional dance music (https://arxiv.org/abs/2407.12000)
Comments:
          6 pages

- **What's New**: 아일랜드 전통 춤 음악의 멜로디에서 Kolmogorov 복잡성을 Lempel-Ziv 압축(Lempel-Ziv compression)을 사용하여 추정했습니다. 멜로디를 구별하는 데 사용할 수 있는 이 복잡성 추정치는 학생들이 '쉬운' 멜로디와 '어려운' 멜로디를 구별하는 데 유용할 것입니다. 또한 리엘(reel)과 지그(jig) 두 멜로디 카테고리를 복잡성 측면에서 비교했습니다.

- **Technical Details**: Kolmogorov 복잡성은 시퀀스로부터 가장 짧은 프로그램의 길이를 통해 정의됩니다. 이는 압축 기법을 사용하여 신뢰할 수 있는 방식으로 추정할 수 있습니다. 이 연구에서는 Lempel-Ziv(LZ) 압축을 사용하였으며 LZ77과 LZ78 두 가지 변종을 활용했습니다. 압축비와 관련된 결과를 비교하기 위해 'Sally Gardens' 리엘을 예로 들어 LZ77과 LZ78 압축 방식을 각각 적용해 보았습니다.

- **Performance Highlights**: 리엘 'Sally Gardens'의 압축 결과는 LZ77과 LZ78 알고리즘을 비교하여, LZ77이 더 큰 반복적인 음절 블록을 효과적으로 식별하는 것으로 나타났습니다. 60개 샘플 리엘에 대한 평균 압축 비율은 2.79로 계산되었습니다.



### Mimetic Po (https://arxiv.org/abs/2407.11984)
Comments:
          Paper accepted at the International Conference on Computational Creativity, (ICCC 2024), Jönköping, Sweden

- **What's New**: 이번 논문은 창의적인 아이디어 발상과 반성적 사고를 촉진하기 위해 생성적 AI를 활용한 새로운 장치의 설계와 초기 평가를 다룹니다. '마그네틱 포에트리(Magnetic Poetry)'에서 영감을 받은 이 장치는 제한된 어휘를 사용하여 짧은 시적 텍스트를 구성하고, 이를 AI가 받아들여 응답을 생성하여 e-ink 화면에 표시합니다.

- **Technical Details**: 이 장치는 OpenAI의 GPT-4 API를 사용한 대형 언어 모델(LLM)로 구동됩니다. 사용자들은 물리적으로 단어를 배치하여 시를 구성하며, 시스템은 이를 기반으로 다양한 창의적 사고 전략을 사용해 응답을 생성합니다. 이러한 상호작용은 전통적인 채팅 인터페이스와는 달리 더 깊고 즐거운 참여를 유도합니다. 장치는 연구실에 2주간 설치되었으며, 종료 시점에는 포커스 그룹이 디자인을 평가했습니다.

- **Performance Highlights**: 제한된 어휘와 물리적 상호작용은 LLM과의 상호작용을 더 깊고 즐겁게 만들었으며, 전통적인 챗봇이나 화면 기반 상호작용보다 창의적 사고를 더 잘 촉진했습니다. 사용자는 AI 생성 응답을 반성적으로 생각할 수 있는 기회를 가지게 되어 창의적 사고에 적합한 환경을 제공했습니다.



New uploads on arXiv(cs.IR)

### Search Engines, LLMs or Both? Evaluating Information Seeking Strategies for Answering Health Questions (https://arxiv.org/abs/2407.12468)
- **What's New**: 새로운 연구에서는 전통적인 웹 검색 엔진과 대형 언어 모델(Large Language Models, LLMs)을 건강 관련 질문에 대한 답변 성과를 비교했습니다. 연구 결과, 웹 엔진은 주어진 질문에 대한 정확성 면에서 LLM보다 열등했음을 발견했고, 검색에서 얻은 웹 페이지의 품질은 순위 리스트가 내려가더라도 크게 감소하지 않았습니다. 하지만 LLM은 입력되는 프롬프트에 매우 민감하고, 검색결과와 함께 사용되는 경우 뛰어난 정보 탐색 능력을 보여주었습니다.

- **Technical Details**: 이 연구는 Google, Bing, Yahoo!, DuckDuckGo와 같은 각기 다른 웹 검색 엔진과 LLM 및 Retrieval-Augmented Generation(RAG) 접근 방식을 비교했습니다. 결과는 상위 순위 웹 페이지에서 추출된 답변이 대부분 고품질임을 확인했고, Bing이 가장 견고한 선택인 반면, LLM은 프롬프트 입력에 아주 민감합니다. RAG 방법은 LLM의 성능을 극대화하는 데 효과적임을 발견했습니다.

- **Performance Highlights**: LLM은 전체적으로 우수한 성능을 보였으나, 일부 경우에는 매우 부정확한 답변을 제공했습니다. RAG는 매우 유망한 접근 방식으로, 적절한 검색 증거를 제공했을 때 가장 작은 LLM도 최첨단 성능을 달성할 수 있음을 확인했습니다.



### RankTower: A Synergistic Framework for Enhancing Two-Tower Pre-Ranking Mod (https://arxiv.org/abs/2407.12385)
- **What's New**: 이번 연구에서는 대규모 랭킹 시스템에서 효율성과 정확성을 동시에 달성하기 위해 제안된 새로운 신경망 아키텍처 RankTower에 대해 소개합니다. RankTower는 사용자-아이템 상호작용을 효율적으로 캡쳐하면서 온라인 추론 효율성을 개선하는 데 중점을 두고 있습니다. 실험 결과는 RankTower가 현존하는 최첨단 사전 랭킹 모델들을 능가한다고 보여줍니다.

- **Technical Details**: RankTower 아키텍처는 주로 세 가지 구성 요소로 이루어져 있습니다: 다중 헤드 게이트 네트워크(Multi-Head Gated Network), 게이트 크로스 어텐션 네트워크(Gated Cross-Attention Network), 및 최대 유사성 레이어(Maximum Similarity Layer)입니다. 이 모델은 다양한 샘플 공간에 대해 최적화된 목표를 학습하는 하이브리드 학습 목적을 사용하여 전체 캐스케이드 랭킹 시스템의 샘플로부터 학습합니다. 사용자의 특징과 아이템의 특징을 사전 계산 및 저장하여 온라인 서비스 효율성을 높이고, 게이트 크로스 어텐션 레이어만 전파를 필요로 하여 계산 효율성을 최적화합니다.

- **Performance Highlights**: RankTower는 세 가지 공개된 데이터셋을 대상으로 한 광범위한 실험에서 예측 정확도와 추론 효율성 측면에서 뛰어난 성능을 보여줍니다. 다양한 사용자 아이템 상호작용을 모델링하면서도 기존의 사전 랭킹 모델을 능가하는 결과를 나타냈습니다.



### Graph Signal Processing for Cross-Domain Recommendation (https://arxiv.org/abs/2407.12374)
- **What's New**: 이번 연구에서는 크로스 도메인 추천(CDR, Cross-domain recommendation) 시스템에서 그래프 신호 처리(GSP, Graph Signal Processing)를 접목한 새로운 프레임워크인 CGSP를 제안합니다. 이 프레임워크는 사용자 맞춤형 그래프 신호를 처리하여 소스 및 타겟 도메인 모두에서 추천 성능을 크게 향상시키고, 특히 유사 사용자가 적은 경우에서도 뛰어난 성능을 보입니다.

- **Technical Details**: CGSP는 타겟 전용 유사도(graph)와 소스 연결 유사도(graph)를 유연하게 조합하여 크로스 도메인 유사도 그래프를 생성합니다. 이를 통해 사용자 맞춤형 그래프 신호를 처리하고, 그래프 퓨리에 변환(Graph Fourier Transform)을 이용해 복잡한 그래프 구조의 패턴을 효과적으로 발견합니다. 중첩 사용자 비율에 민감하지 않은 방어적인 구조를 가지고 있어, 소스 도메인 지식을 타겟 도메인 추천에 효과적으로 활용할 수 있습니다.

- **Performance Highlights**: CGSP는 다양한 인코더 기반 CDR 접근 방식 및 협업 필터링(Collaborative Filtering) 방법론을 상회하는 성능을 보여줍니다. 특히 데이터 스파스티(sparsity) 및 중첩 사용자 비율이 낮은 Amazon 데이터셋에서 20% 이상의 성능 향상을 달성하며, 중첩 사용자 비율에 민감하지 않은 높은 강건성을 나타냅니다. 공공 코드 저장소에서 재현 가능성이 검증되었습니다.



### GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation (https://arxiv.org/abs/2407.12338)
Comments:
          11 pages, accepted by CIKM 2024

- **What's New**: 본 논문에서는 추천 시스템의 롱테일(long-tail) 항목 문제를 해결하기 위해 새로운 Graphs and User Modalities Enhancement (GUME) 방식을 제안합니다. 이전 연구와 달리, 이 연구는 롱테일 항목의 상호작용 데이터 부족 문제와 사용자 모달리티(modality) 선호의 중요성을 동시에 해결합니다.

- **Technical Details**: 이번 연구에서는 먼저 사용자-항목 그래프를 모달리티 유사성을 기반으로 강화하여 롱테일 항목의 연결성을 개선합니다. 여기서 모달리티란 주로 텍스트와 이미지 데이터를 의미합니다. 이후 사용자 모달리티를 명시적 상호작용 특징(explicit interaction features)과 확장된 관심사 특징(extended interest features)으로 나누어, 이 두 특징 간의 상호정보(mutual information)를 최대화하도록 설계된 사용자 모달리티 향상 전략을 사용합니다. 또한 내부 및 외부 정보 관점에서 모달리티 데이터의 노이즈를 제거하는 정렬 전략도 포함되어 있습니다.

- **Performance Highlights**: 네 가지 공개 데이터셋(Amazon 데이터셋)에서의 실험 결과, 제안된 GUME 방식이 롱테일 추천 시스템의 효과성을 크게 향상시킴을 증명하였습니다. 특히, 연결성 강화와 사용자 모달리티 일반화 능력에서 두드러진 성과를 나타냈습니다.



### Optimizing Query Generation for Enhanced Document Retrieval in RAG (https://arxiv.org/abs/2407.12325)
- **What's New**: 최근 연구에서는 대규모 언어 모델(LLMs)이 다양한 언어 작업에서 탁월한 성과를 보이지만, 잘못된 정보를 생성하는 '환각(hallucinations)' 문제가 종종 발생하는 점을 지적합니다. 이를 해결하기 위해 Retrieval-Augmented Generation (RAG) 기법이 문서 검색을 통해 정확한 응답을 제공하는 방안을 제시하였으나 불완전한 쿼리로 인한 환각 문제는 여전히 남아있습니다. 본 연구는 쿼리 생성 최적화를 통해 RAG 시스템의 쿼리 명확성을 개선하여 문서 검색의 정확성과 효율성을 높이는 것을 목표로 합니다. 실험 결과, 본 접근 방식은 문서 검색의 정확도를 평균 1.6% 향상시키는 것으로 나타났습니다.

- **Technical Details**: 본 연구에서는 Query Optimization using Query expAnsion (QOQA)라는 새로운 기법을 제안합니다. 이는 대규모 언어 모델(LLMs)을 이용하여 쿼리를 리프레이즈하고, top-k 평균 쿼리-문서 정렬 점수를 사용하여 쿼리를 정제합니다. 초기 쿼리와 상위 N개의 검색된 문서를 결합하여 확장된 쿼리를 생성한 후, 이를 LLM에 입력하여 여러 개의 리프레이즈된 쿼리를 생성합니다. 생성된 쿼리는 다시 검색된 문서와의 정렬 점수를 평가받아 저장됩니다. 이러한 과정을 반복하며 최적화된 쿼리를 선택합니다.

- **Performance Highlights**: 제안된 QOQA 접근방식은 기존의 RAG 시스템 대비 문서 검색의 정확도를 1.6% 향상시키는 성과를 보였습니다. 이는 환각 문제를 줄이는 데 크게 기여하며, 정보 검색에서의 효과적인 쿼리 최적화를 통해 더욱 신뢰성 있는 시스템을 제공할 수 있음을 입증하였습니다.



### Mindful-RAG: A Study of Points of Failure in Retrieval Augmented Generation (https://arxiv.org/abs/2407.12216)
- **What's New**: 새로운 Mindful-RAG 접근법을 통해 대형 언어 모델(LLMs)이 지식 그래프(KGs)를 이용해 도메인-특화 질문과 응답 작업에서 더 정확한 답변을 제공하도록 개선되었습니다.

- **Technical Details**: Mindful-RAG는 질문의 의도를 명확히 파악하고, 지식 그래프에서 관련 있는 문맥을 효과적으로 수집 및 정렬하여 답변을 생성하는 프레임워크입니다. 기존의 KG 기반 RAG 시스템에서는 LLM이 질문의 의도 파악과 문맥 맞춤에 어려움을 겪는 것으로 분석되었습니다. 이 접근법은 Intent-Driven Retrieval과 Contextual Alignment 단계를 포함하며, 모델의 파라메트릭 지식을 사용하여 질문 의도를 더 잘 이해하고 지식의 맥락을 정렬합니다.

- **Performance Highlights**: WebQSP와 MetaQA 데이터셋에서 실험 결과, Mindful-RAG는 기존 최첨단 방법들(SOTA)에 비해 성능이 향상된 것으로 나타났습니다. 특히, 의도와 문맥 정렬에 중점을 두어 추론 오류를 크게 줄였습니다.



### ClaimCompare: A Data Pipeline for Evaluation of Novelty Destroying Patent Pairs (https://arxiv.org/abs/2407.12193)
- **What's New**: 새로운 데이터 파이프라인 ClaimCompare이 도입되었습니다. 이 파이프라인은 특허권 주장 데이터셋을 생성하여 정보 검색(IR)과 기계 학습(ML) 모델을 훈련시키기 위해 설계되었습니다. 이를 통해 특허의 새로운 성질을 파괴하는 기존 특허를 자동으로 검출하는 과정을 간소화할 수 있습니다.

- **Technical Details**: ClaimCompare 파이프라인은 미국 특허청(USPTO) API와 Google Patents 웹 스크래핑을 활용하여 생성됩니다. 특히 전해 화학 분야에서 27,000개 이상의 특허를 포함한 샘플 데이터셋을 구성하였습니다. 이 중 1,045개의 기본 특허 각각에 대해 25개의 관련 특허가 존재하며, 이 중 34%는 새로운 성질을 파괴하는 특허로 라벨링되었습니다.

- **Performance Highlights**: ClaimCompare에서 생성된 데이터셋을 사용하여 대형 언어 모델(LLM)을 미세 조정한 결과, MRR과 P@1 측정치에서 각각 29.2%와 32.7%의 절대 개선을 달성했습니다. 이는 고성능 모델이 특허의 새로운 성질을 파괴하는 특허를 효과적으로 식별할 수 있음을 나타냅니다.



### Neural Passage Quality Estimation for Static Pruning (https://arxiv.org/abs/2407.12170)
Comments:
          SIGIR 2024

- **What's New**: 이 연구는 신경망을 이용하여 문서의 특정 구절이 어떤 검색 쿼리에도 관련이 없는지를 예측하는 방법을 제안합니다. 기존의 검색엔진은 사용자 쿼리에 대한 문서의 관련성을 평가하는 데 초점을 맞췄지만, 이번 연구는 구절의 '품질'을 평가하여 유용하지 않은 구절을 사전에 제거할 수 있는지를 탐구합니다.

- **Technical Details**: 연구는 다양한 신경망 모델을 사용하여 구절 품질을 평가하는 여러 방법을 탐구합니다. 비지도 학습 신호(언어 모델의 perplexity 등), 잠재 신호(밀집 검색 모델의 벡터 크기 등), 직접 감독(관련성 레이블로 미세 조정된 모델 등) 등을 포함합니다. 비지도 학습과 잠재 신호 기반 모델은 랜덤 제거 기준을 크게 능가하지 못했지만, 감독 신경망 모델은 일관되게 강력한 신호를 제공하여 구절 코퍼스의 25% 이상을 효과적으로 제거할 수 있었습니다.

- **Performance Highlights**: 본 연구에서 제안된 모델은 다양한 검색 파이프라인에서 통계적으로 동일한 검색 성능을 유지하면서 구절 코퍼스의 25% 이상을 일관되게 제거할 수 있습니다. 이러한 제거는 신경 검색 엔진의 운영 비용(컴퓨팅 자원, 전력 사용량, 탄소 발자국)을 줄이는 데 기여하며, 더 가볍고 비용 효율적인 인덱싱 과정이 가능합니다.



### AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases (https://arxiv.org/abs/2407.12784)
Comments:
          22 pages, 13 figures, 7 tables

- **What's New**: 이 연구에서는 다양한 애플리케이션에서 뛰어난 성능을 보여주는 기존의 LLM agents(대형 언어 모델 에이전트)들이 의존하는 검증되지 않은 지식베이스의 신뢰성 문제를 해결하기 위해 새로운 공격 방법인 'AgentPoison'을 제안합니다. AgentPoison은 긴 메모리 또는 RAG(retrieval-augmented generation)을 중독시켜, 악성 트리거를 포함하는 질문을 통해 악성 데이터를 높은 확률로 검색할 수 있게 합니다.

- **Technical Details**: AgentPoison은 제약 최적화(constrained optimization)를 사용하여 악성 트리거를 최적화하는 방식으로 작동합니다. 이를 통해 악성 트리거가 포함된 사용자의 지시가 있을 때 중독된 메모리 또는 지식베이스에서 악성 데이터를 검색하게 합니다. 이 과정에서 트리거 최적화는 고유한 임베딩 공간에 매핑되어, 벤진 질문(benign queries)은 정상적으로 작동합니다. 또한, 추가적인 모델 재훈련이나 파인 튜닝(fine-tuning)이 필요하지 않습니다.

- **Performance Highlights**: AgentPoison은 자율주행, 지식집약적 질문응답, 그리고 헬스케어 관련 LLM agents를 대상으로 한 실험에서 평균 80% 이상의 공격 성공률을 기록했으며, 벤진 성능은 1% 미만의 영향을 받았습니다. 악성 데이터 주입률(poison rate)도 0.1% 미만으로 매우 낮습니다.



### E5-V: Universal Embeddings with Multimodal Large Language Models (https://arxiv.org/abs/2407.12580)
Comments:
          Code and models are available at this https URL

- **What's New**: 이번 연구에서는 멀티모달 대형 언어 모델 (MLLM, Multimodal Large Language Model)을 활용하여 범용 멀티모달 임베딩 (universal multimodal embeddings)을 달성하기 위해 E5-V라는 새로운 프레임워크를 도입했다. 이 방법은 멀티모달 정보 표현에 있어 기존 접근법보다 우수한 성능을 보여준다.

- **Technical Details**: E5-V는 텍스트 쌍 (text pairs)에 대해 단일 모달리티 훈련 (single modality training)을 실시하여, 훈련 비용을 약 95% 줄였다. 이는 비싼 멀티모달 데이터 수집을 제거하며, 텍스트 입력을 통해 비쥬얼 인코더를 제거하고 입력 크기를 감소시킨다. E5-V는 또한 프롬프트 (prompt)을 사용하여 모달리티 간 격차를 효과적으로 줄인다.

- **Performance Highlights**: E5-V는 다양한 작업(text-image retrieval, composed image retrieval, sentence embeddings, image-image retrieval)에서 기존 최첨단 (state-of-the-art) 성능을 초과하거나 경쟁하는 성능을 보여준다. 특히 단일 모달리티 훈련만으로 멀티모달 임베딩을 효율적으로 수행할 수 있었다.



### Object-Aware Query Perturbation for Cross-Modal Image-Text Retrieva (https://arxiv.org/abs/2407.12346)
Comments:
          ECCV 2024

- **What's New**: 최근 발표된 논문에서는 프리트레인된 비전-언어(Vision and Language, V&L) 모델이 이미지-텍스트 검색 성능을 크게 향상시켰지만, 작은 객체에 대한 검색 성능이 낮다는 문제를 지적하고 있습니다. 이를 해결하기 위해 '오브젝트 인지 쿼리 교란(Object-Aware Query Perturbation)' 기반의 새로운 크로스모달 이미지-텍스트 검색 프레임워크를 제안합니다. 이 방법은 객체의 주요 특징 공간을 생성하고, 이를 이용해 쿼리를 교란시켜 객체 인식을 향상시킵니다.

- **Technical Details**: 제안된 방법은 감지된 객체의 주요 특징 공간을 생성하고, 이 특징 공간을 이용해 쿼리를 교란하여 이미지 내 객체 인식을 향상시킵니다. 기존의 V&L 모델의 쿼리를 단순히 강화하면 원래의 가중치(Weights)를 깨트릴 수 있지만, 제안된 쿼리 교란은 객체 정보와 관련된 키(Key)를 사용하여 쿼리를 서브스페이스(Subspace) 내에서 강화하므로 이러한 문제를 피할 수 있습니다.

- **Performance Highlights**: 네 개의 공개 데이터셋에서 종합적인 실험을 통해 제안된 방법이 기존 알고리즘을 능가한다는 결과를 얻었습니다. 제안된 방법은 BLIP2, COCA, InternVL과 같은 최신 V&L 모델과 결합하여 작고 중요한 객체를 포함하는 이미지에서도 정확한 검색이 가능하도록 합니다. 특히 Recall@1 지표를 사용한 평가에서 작은 객체의 비율이 클수록 검색 성능이 저하됨을 보였으며, 이를 보완하는 제안된 방법의 효과를 입증하였습니다.



### ModalChorus: Visual Probing and Alignment of Multi-modal Embeddings via Modal Fusion Map (https://arxiv.org/abs/2407.12315)
Comments:
          Accepted by VIS 2024

- **What's New**: 이번 연구에서는 ModalChorus라는 상호작용 시스템을 소개합니다. 이 시스템은 시각적 프로빙(visual probing)과 다중모달 임베딩(multi-modal embeddings)의 정렬(alignment)을 개선하는 데 중점을 둡니다. ModalChorus는 두 단계로 구성됩니다: Modal Fusion Map(MFM)을 사용한 임베딩 프로빙(embedding probing)과 사용자와의 상호작용을 통해 점 집합 및 세트 집합 정렬(embedding alignment)을 수행하는 것입니다.

- **Technical Details**: Modal Fusion Map(MFM)은 새로운 파라메트릭 차원 축소 방법으로, 메트릭(metric)과 비메트릭(non-metric) 목표를 결합하여 모달 융합(modality fusion)을 강화합니다. 이는 다중모달 임베딩에서 발생하는 모달리티 갭(modality gap) 문제를 효과적으로 해결합니다. 또한, MFM은 높은 신뢰성과 연속성을 제공하여 다중모달 임베딩의 맥락적 분포를 시각적으로 더 잘 반영할 수 있습니다.

- **Performance Highlights**: ModalChorus는 특히 zero-shot 분류, 크로스모달 검색 및 생성 시나리오에서 직관적인 오작동 발견과 효율적인 정렬을 용이하게 합니다. 실증 연구에 따르면, MFM은 기존의 차원 축소 및 데이터 융합 방법보다 CLIP 임베딩스의 특징을 더 잘 나타냅니다. 또한, 포인트-세트(point-set) 및 세트-세트(set-set) 정렬을 위한 상호작용 스킴을 제공하여 사용자 지정 정렬을 허용합니다.



### The Kolmogorov Complexity of Irish traditional dance music (https://arxiv.org/abs/2407.12000)
Comments:
          6 pages

- **What's New**: 아일랜드 전통 춤 음악의 멜로디에서 Kolmogorov 복잡성을 Lempel-Ziv 압축(Lempel-Ziv compression)을 사용하여 추정했습니다. 멜로디를 구별하는 데 사용할 수 있는 이 복잡성 추정치는 학생들이 '쉬운' 멜로디와 '어려운' 멜로디를 구별하는 데 유용할 것입니다. 또한 리엘(reel)과 지그(jig) 두 멜로디 카테고리를 복잡성 측면에서 비교했습니다.

- **Technical Details**: Kolmogorov 복잡성은 시퀀스로부터 가장 짧은 프로그램의 길이를 통해 정의됩니다. 이는 압축 기법을 사용하여 신뢰할 수 있는 방식으로 추정할 수 있습니다. 이 연구에서는 Lempel-Ziv(LZ) 압축을 사용하였으며 LZ77과 LZ78 두 가지 변종을 활용했습니다. 압축비와 관련된 결과를 비교하기 위해 'Sally Gardens' 리엘을 예로 들어 LZ77과 LZ78 압축 방식을 각각 적용해 보았습니다.

- **Performance Highlights**: 리엘 'Sally Gardens'의 압축 결과는 LZ77과 LZ78 알고리즘을 비교하여, LZ77이 더 큰 반복적인 음절 블록을 효과적으로 식별하는 것으로 나타났습니다. 60개 샘플 리엘에 대한 평균 압축 비율은 2.79로 계산되었습니다.




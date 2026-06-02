New uploads on arXiv(cs.CL)

### Drawing Conclusions from Draws: Rethinking Preference Semantics in Arena-Style LLM Evaluation (https://arxiv.org/abs/2510.02306)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문에서는 arena-style 평가에서 두 개의 대형 언어 모델(LLMs)의 전투 결과를 바탕으로 모델의 평가 점수(rating)를 조정하는 기존의 패러다임을 비판적으로 검토합니다. 특히, 무승부(draw)가 두 모델이 동등하다는 의미일까요? 저자들은 오히려 무승부는 쿼리(query)의 난이도를 나타내며, 쿼리가 너무 쉬울 때 두 모델이 동등한 성공을 거둘 가능성이 더 높다고 주장합니다. 또한, 향후 평가 시스템은 기존의 무승부 의미를 재고할 것을 권장합니다.

- **Technical Details**: Arena-style 평가에서는 사용자 판단 유도와 모델 평가 점수 업데이트의 두 단계로 구성됩니다. 사용자가 두 개의 익명 LLM과 상호작용하고, 더 나은 응답을 선택하거나 동점을 선언하는 방식으로 진행됩니다. 평가 시스템은 사용자 판단에 따라 모델의 평가 점수를 업데이트하며, 무승부가 발생할 경우 두 모델의 점수를 동등하게 조정합니다. 본 논문에서는 Elo, Glicko-2, Bradley-Terry, TrueSkill 등 네 가지 기존 평가 시스템을 고려하고, 각 시스템의 업데이트 규칙을 설명합니다.

- **Performance Highlights**: 이 연구에서는 무승부에 대한 업데이트를 무시했을 때의 배틀 예측 정확도가 1-3% 상승하는 것을 발견했습니다. 실제 데이터셋을 통해 이 개선이 12개 조합의 11개에서 일관되게 나타났습니다. 추가 분석에서는 쿼리가 매우 쉽거나 주관적일 때 무승부가 더 자주 발생하는 경향이 있음을 보여줍니다.



### F2LLM Technical Report: Matching SOTA Embedding Performance with 6 Million Open-Source Data (https://arxiv.org/abs/2510.02294)
- **What's New**: F2LLM은 최신 embedding 모델로, 0.6B, 1.7B, 4B 크기로 제공된다. 기존 top-ranking embedding 모델들이 거대한 contrastive pretraining과 비싼 synthetic training data를 필요로 하는 것과 달리, F2LLM은 개방형 데이터셋에서 수집된 6백만 개의 query-document-negative tuple을 사용하여 직접 finetuned되었다. 이 모델은 훈련 비용, 모델 규모 및 embedding 성능 간의 균형을 잘 맞추며, 연구자들에게 reproducible하고 경제적인 기준을 제공한다.

- **Technical Details**: F2LLM은 Foudation 모델에서 직접 finetuned되어 6백만 개의 고품질 query-document-hard negative tuple로 구성된다. 이 데이터는 개방형 비합성 데이터셋에서만 수집되어 다양한 작업 유형을 커버한다. 훈련은 여러 작업들에 대해 통합된 포맷으로 진행되며, 각 데이터 샘플은 query와 positive passage, 여러 개의 hard negative tuple로 구성된다.

- **Performance Highlights**: 현재 F2LLM-4B는 MTEB 영어 리더보드에서 약 4B 파라미터 모델 중 2위, 전체 7위를 기록하고 있다. 또한, F2LLM-1.7B는 1B-2B 사이즈의 모델 중 1위를 기록하였으며, 제한된 컴퓨팅 자원으로 사용할 수 있는 이상적인 선택지로 평가받고 있다.



### From Behavioral Performance to Internal Competence: Interpreting Vision-Language Models with VLM-Lens (https://arxiv.org/abs/2510.02292)
Comments:
          EMNLP 2025 System Demonstration | Code: this https URL

- **What's New**: VLM-Lens는 비전-언어 모델(vision-language models, VLMs)의 체계적인 벤치마킹(benchmarking), 분석(analysis) 및 해석(interpretation)을 가능하게 하는 도구 모음(toolkit)입니다. 이 툴킷은 오픈 소스 VLM의 포워드 패스(forward pass) 중 모든 레이어에서 중간 출력을 추출하는 기능을 지원합니다. VLM-Lens는 다양한 VLM을 사용자 친화적으로 운영할 수 있도록 모델 특정 복잡성을 추상화한 YAML-configurable 인터페이스를 제공합니다.

- **Technical Details**: 현재 VLM-Lens는 16개의 최신 기본 VLM과 그 30가지 이상의 변형을 지원하며, 핵심 로직을 변경하지 않고 새로운 모델을 수용할 수 있도록 확장 가능합니다. 이 툴킷은 다양한 해석 가능성(interpretability) 및 분석 방법들과 쉽게 통합되어 사용될 수 있습니다. 또한, 레이어(layer)와 타겟 개념(target concepts) 전반에 걸쳐 숨겨진 표현(hidden representations)의 체계적인 차이를 드러내는 두 가지 간단한 분석 실험을 통해 그 사용법을 시연합니다.

- **Performance Highlights**: VLM-Lens는 VLM의 이해와 개선을 위한 커뮤니티 노력을 가속화하기 위해 오픈 소스로 출시되었습니다. 이 도구는 다양한 VLM에 대한 깊이 있는 분석과 효율적인 비교를 가능하게 하여, 연구자들이 모델 성능과 해석을 더욱 명확히 파악하는 데 도움을 줍니다. VLM-Lens의 도입으로 연구자들은 비전-언어 모델들의 레이어 간 차이를 보다 쉽게 분석할 수 있게 되었습니다.



### Parallel Scaling Law: Unveiling Reasoning Generalization through A Cross-Linguistic Perspectiv (https://arxiv.org/abs/2510.02272)
Comments:
          Work in progress

- **What's New**: 이 연구는 Reinforcement Post-Training (RPT)를 기반으로 한 대형 추론 모델(Large Reasoning Models, LRM)의 다국어 일반화 가능성을 탐구합니다. 기존 연구는 주로 과제 또는 양식 간의 일반화에 중점을 두었으나, 본 연구는 새로운 교차 언어적 관점을 제시하여 영어에서 다른 언어로의 추론 능력 이전을 평가합니다. 이러한 접근 방식은 LRM의 본질적인 언어 비의존성을 확인하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구는 영어 중심의 LRM을 기반으로 하여 11개 언어의 다국어 추론 벤치마크에서 English-centric LRM의 성능을 체계적으로 평가합니다. Multilingual Transferability Index (MTI)를 도입하여 모델의 교차 언어 이식성을 정량화하고, 다양한 초기 모델, 목표 언어, 훈련 패라다임에 따른 차이를 분석합니다. 또한, 병렬 훈련 연구를 통해 단일 기계에서 다국어 훈련의 중요성과 그 효과를 이론적으로 뒷받침합니다.

- **Performance Highlights**: 실험 결과, 모델이 단일 언어에서 단일 병렬 언어로 전환할 때 현저한 성능 향상인 First-Parallel Leap를 발견하였으며, 병렬 언어 수에 따라 성능이 전력 법칙을 따른다는 Parallel Scaling Law를 제시합니다. 또한 실제 단일 언어 성능과 예측된 성능 간의 차이를 나타내는 Monolingual Generalization Gap을 확인하여, 영어 중심 모델이 타 언어로의 일반화에 실패함을 보여줍니다. 이 연구는 LRM의 추론 능력이 인간의 인지 방식과는 마 주다는 점에서 중요한 통찰을 제공합니다.



### InfoMosaic-Bench: Evaluating Multi-Source Information Seeking in Tool-Augmented Agents (https://arxiv.org/abs/2510.02271)
- **What's New**: 이 논문에서는 정보 탐색을 위한 새로운 벤치마크인 InfoMosaic-Bench를 소개합니다. LLM(대형 언어 모델) 에이전트가 도메인 특정 도구와 일반 검색을 결합하여 복잡한 작업을 수행하는 효율성을 평가하는 데 중점을 두고 있습니다. InfoMosaic-Bench는 의료, 금융, 지도, 비디오, 웹, 다중 도메인 통합의 6개 대표 도메인에서 621개의 합성 과제를 포함하고 있습니다.

- **Technical Details**: InfoMosaic-Bench는 InfoMosaic-Flow라는 확장 가능한 파이프라인을 통해 과제를 생성하며, 이는 태스크 조건을 검증된 도구 출력에 기반하고 다원 소스 의존성을 강제합니다. 또한, 단순 검색으로 해결할 수 있는 경우를 필터링하여 신뢰성과 비단순성을 보장합니다. 이를 통해 에이전트는 여러 도메인 도구를 통합하여 신뢰할 수 있는 정보를 탐색할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 현재의 에이전트들은 여전히 기본적인 도구 처리 능력에도 어려움을 겪고 있으며, 도메인 도구의 부정확한 사용으로 인해 총 실패의 22.4%가 발생한다는 사실이 밝혀졌습니다. GPT-5는 단독 웹 정보에 의존할 경우 38.2%의 정확도에 그쳤으며, 특정 도메인에서 도구가 선택적으로 이점을 제공하지만 전체적으로 일관성이 부족하다는 점이 드러났습니다.



### Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation (https://arxiv.org/abs/2510.02249)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 과도한 사고(overthinking) 문제를 해결하기 위해 새로운 지표인 Token Entropy Cumulative Average (TECA)를 도입합니다. TECA는 추론 과정에서의 탐색 정도를 측정하며, 이를 활용하여 모델이 최적의 시점에서 사고를 종료하도록 돕는 새로운 사고 패러다임인 'Explore Briefly, Then Decide'를 제안합니다. 이를 통해 모델은 기술적인 깊이를 복잡도에 맞춰 조정할 수 있는 능력을 확보하게 됩니다.

- **Technical Details**: TECA는 각각의 추론 단계에서 모델의 불확실성을 나타내는 토큰 엔트로피(token entropy)의 누적 평균을 계산합니다. 이를 기반으로 탐색(exploration) 단계와 결정(determination) 단계를 구분하여 과도한 탐색을 제어하는 Cumulative Entropy Regulation (CER) 메커니즘을 도입합니다. CER은 모델이 불필요하게 긴 사고 과정을 피하도록 돕는 동시에 필요한 탐색 능력을 유지하게 합니다.

- **Performance Highlights**: 제안한 방법을 통해 다양한 수학 문제 벤치마크에서 모델의 응답 길이가 평균 71%까지 감소하면서도 문제 해결 능력은 거의 유지되었음을 확인했습니다. 특히, Qwen3-4B 모델은 GSM8K에서 71%의 응답 길이 감소를, MATH500에선 39.25% 감소를 기록하며, 기존 방법들보다 전반적으로 우수한 성능을 나타냈습니다. 이러한 실험 결과는 TECA를 지표로 활용해 추론 과정을 적절히 조정함으로써 과도한 사고를 감소시킬 수 있음을 보여줍니다.



### AccurateRAG: A Framework for Building Accurate Retrieval-Augmented Question-Answering Applications (https://arxiv.org/abs/2510.02243)
- **What's New**: AccurateRAG는 고성능 질문-응답 애플리케이션을 구축하기 위한 새로운 프레임워크로, Retrieval-Augmented Generation(RAG) 기법을 기반으로 하고 있습니다. 이 프레임워크는 데이터셋 처리, 모델 미세 조정(fine-tuning), 텍스트 임베딩 및 평가 등 개발 효율성을 위한 다양한 도구를 제공합니다. 실험 결과에 따르면 AccurateRAG는 이전 강력한 기준선을 초과하며 새로운 최첨단 질문-응답 성능을 달성했습니다.

- **Technical Details**: AccurateRAG 프레임워크는 문서 형식(PDF, DOCX)에서 데이터를 처리하는 Preprocessor, 텍스트 임베딩 모델과 LLM 모델을 위한 Fine-tuning Data Generator, Retrieval 기능을 갖춘 Retriever, 응답 생성을 위한 Answer Generator로 구성되어 있습니다. Preprocessor는 문서를 Markdown 형식으로 변환하여 내용을 구조적으로 보존하며, Fine-tuning Data Generator는 LLM을 활용해 질문-응답 쌍을 자동 생성합니다. Retrieval 과정에서는 Semantic Search와 Conventional Search 방법으로 관련 콘텐츠를 찾아냅니다.

- **Performance Highlights**: 실험을 통해 AccurateRAG는 기존 RAG 시스템보다 우수한 성능을 보였습니다. 특히 다양한 질문을 처리하고 높은 정확도를 자랑하는 방식으로 기존 모델들의 한계를 극복했습니다. 이러한 성과는 AccurateRAG의 모듈 구조 덕분에 가능했으며, 특히 개인화된 데이터셋을 효과적으로 활용할 수 있음이 강조됩니다.



### Enhanced Arabic-language cyberbullying detection: deep embedding and transformer (BERT) approaches (https://arxiv.org/abs/2510.02232)
- **What's New**: 최근 스마트폰 및 통신 기술의 발전, 특히 X(구 트위터)와 같은 대규모 소셜 미디어 네트워크의 성장은 청소년을 사이버 괴롭힘의 위험에 노출시키고 있습니다. 기존 사이버 괴롭힘 탐지 방법은 주로 영어에 중점을 두고 개발되어왔으며, 아랍어 사이버 괴롭힘 탐지에 대한 연구는 매우 부족한 상황입니다. 이 논문은 아랍어 콘텐츠에서 사이버 괴롭힘을 탐지하는 효과적인 방법을 향상시키기 위한 목표로 진행되었습니다.

- **Technical Details**: 연구진은 10,662개의 X 포스트로 구성된 데이터셋을 수집하고, 데이터 전처리 후 kappa 도구를 사용하여 주석의 품질을 검증하고 향상시켰습니다. 여러 딥러닝 모델을 테스트하기 위해 네 번의 실험을 진행했으며, 먼저 장단기 기억(Long Short-Term Memory, LSTM) 모델과 양방향 LSTM(Bi-LSTM) 모델을 실험하였습니다. 이후 LSTM 및 Bi-LSTM 모델과 함께 새로운 사전 훈련된 양방향 인코더(Bidirectional Encoder Representations from Transformers, BERT)를 사용하여 성능을 검토했습니다.

- **Performance Highlights**: LSTM-BERT 및 Bi-LSTM-BERT 모델은 97%의 정확도를 기록했습니다. 특히 Bi-LSTM에서 FastText 임베딩을 적용한 경우에는 더욱 향상된 98%의 정확도를 달성했습니다. 이러한 결과는 아랍어 사이버 괴롭힘 탐지 방법의 일반화 가능성을 보여줍니다.



### More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration (https://arxiv.org/abs/2510.02227)
Comments:
          20 pages, 5 figures

- **What's New**: 이 논문은 Reinforcement Learning with Verifiable Rewards (RLVR) 기법을 사용하여 대형 언어 모델(LLM)의 추론 능력을 개선하는 새로운 패러다임인 Adaptive Multi-Guidance Policy Optimization (AMPO)를 소개합니다. 기존의 방법보다 더 다양한 탐색을 허용하고, 불필요한 개입 없이 자기 탐색의 가치를 보존하면서도 여러 능숙한 교사 모델의 지도를 적절히 활용합니다. 이를 통해 AMPO는 더 나은 효과성과 일반화를 제공하여 모델의 성능을 크게 향상시킵니다.

- **Technical Details**: AMPO는 혼합 정책 강화 학습(Mixed-Policy RL) 프레임워크이며, 여러 동료 모델의 집합적 지능을 활용합니다. 이 방법은 학생 모델이 문제를 해결하지 못할 경우에만 외부 지도를 제공하는 'guidance-on-demand' 원칙을 적용합니다. 또한, 학생이 이해하기 쉬운 추론 경로를 학습하도록 유도하는 이해 기반 지침 선택 메커니즘을 도입하여 대폭적인 탐색과 효과적인 활용 간의 균형을 맞춥니다.

- **Performance Highlights**: 실험 결과 AMPO는 강력한 기준선 모델인 GRPO보다 평균 4.3% 및 분포 외(out-of-distribution) 작업에서 12.2%의 개선을 보여주었습니다. 특히, 4명의 동료 크기 교사를 사용함으로써 하나의 강력한 교사를 사용하는 방법과 유사한 성능을 보이며, 더 적은 데이터로도 높은 성능을 기록했습니다. 이러한 결과는 AMPO가 탐색과 활용 간의 우수한 균형을 이루는 방법임을 보여줍니다.



### Say One Thing, Do Another? Diagnosing Reasoning-Execution Gaps in VLM-Powered Mobile-Use Agents (https://arxiv.org/abs/2510.02204)
- **What's New**: 이번 연구에서는 비전-언어 모델(Vision-Language Models, VLMs) 기반 모바일 에이전트의 실행 정확성을 향상시키기 위한 새로운 평가 프레임워크를 도입합니다. 기존의 평가 방법들은 주로 실행 정확성에 초점을 맞추었지만, 체계적 사고(Chain-of-Thought, CoT) 추론이 실제 행동과 어떻게 일치하는지를 평가하지 않았습니다. 이로 인해 사용자들은 잘못된 추론에 기반한 허위 신뢰를 형성할 위험이 있기 때문에, Ground-Truth Alignment (GTA) 지표를 통해 이러한 간극을 진단합니다.

- **Technical Details**: GTA는 CoT에서 암시된 행동이 실제 행동과 얼마나 일치하는지를 측정하는 지표입니다. 연구팀은 GTA와 전통적인 Exact Match (EM) 지표를 결합하여 네 가지 진단 영역으로 모델 출력을 분류합니다: (i) Ideal, (ii) Execution Gap (EG), (iii) Both Wrong, (iv) Reasoning Gap (RG). 이 프레임워크는 잘못된 추론 또는 실행에서 발생하는 오류를 명확히 구분하며, 이를 통해 모델의 실패 원인을 면밀히 분석할 수 있습니다.

- **Performance Highlights**: 광범위한 모바일 상호작용 과제를 통한 실험을 통해, 추론-실행 간극이 일반적이며, 실행 간극(EG)이 추론 간극(RG)보다 더 자주 발생함을 발견했습니다. 연구 결과, 모델 크기가 커질수록 전체 간극이 줄어들지만, 여전히 상당한 실행 간극이 존재함을 보여주었습니다. 이러한 발견은 더 신뢰할 수 있는 모바일 에이전트 개발을 위한 구체적인 진단 기준을 제공합니다.



### ARUQULA -- An LLM based Text2SPARQL Approach using ReAct and Knowledge Graph Exploration Utilities (https://arxiv.org/abs/2510.02200)
Comments:
          peer reviewed publication at Text2SPARQL Workshop @ ESWC 2025

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)을 활용하여 자연어 질문을 SPARQL 쿼리로 변환하는 탈바꿈된 접근방식을 제안합니다. 특히, 기존 SPINACH의 아이디어를 일반화하여 여러 개의 지식 그래프에 적용할 수 있도록 하였으며, 이를 TEXT2SPARQL 챌린지에 맞춰 구축한 과정을 상세히 설명합니다. 이 과정에서 자연어 질문의 염두에 둔 탐색 및 실행을 위한 반복적 프로세스를 도입하여 더 나은 쿼리 생성을 목표로 하고 있습니다.

- **Technical Details**: 본 연구에서는 ReAct 접근법을 사용하여 LLM이 자연어 질문을 올바른 SPARQL 쿼리로 변환할 수 있는 구조를 설계했습니다. 시스템은 LangGraph와 RPT 및 Qlever를 이용하여 SPARQL 엔드포인트를 설정하였으며, Qdrant 벡터 데이터베이스와 Lucene 텍스트 검색을 결합하여 하이브리드 검색을 구현했습니다. 마지막으로, 다양한 지식 그래프 탐색 유틸리티와 함께 LLM이 과거의 모든 동작과 관찰을 참고할 수 있도록 설계되었습니다.

- **Performance Highlights**: TEXT2SPARQL 챌린지에서는 DBpedia와 Corporate Knowledge Graph(CKG)라는 두 개의 데이터 세트를 사용하여 시스템의 성능을 평가했습니다. DBpedia에서는 다국어 쿼리를 처리하는 확장성, CKG에서는 도메인 적합성 및 정확도를 평가하여 LLM의 효용성을 종합적으로 분석했습니다. 실험 결과, 기존 LLM을 기반으로 한 접근법보다 높은 수준의 정확도와 처리를 보여주는 안정성을 입증하였습니다.



### Learning to Reason for Hallucination Span Detection (https://arxiv.org/abs/2510.02173)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 환각(span) 탐지를 위한 새로운 접근 방식을 소개합니다. 기존 연구들은 환각 탐지를 이진(binary) 문제로 정의했지만, 실제로는 특정 환각 부분을 식별해야 하는 필요성이 존재합니다. 이 문제를 해결하기 위해, Chain-of-Thought (CoT) 추론을 활용한 강화 학습 프레임워크인 RL4HS를 제안합니다. 이 방법은 환각 탐지를 보다 정교하게 수행할 수 있는 가능성을 보여줍니다.

- **Technical Details**: RL4HS는 Group Relative Policy Optimization(GRPO) 기반으로 설계되며, span 수준의 보상 함수를 통해 추론을 장려합니다. 구체적으로, 환각 부분을 포함한 레이블 데이터셋을 사용하여 코트(CoT) 추론 기반 모델을 학습합니다. 또한, 비환각 예측에 대한 보상 불균형 문제를 해결하기 위해 class-aware policy optimization 기법을 도입했습니다. 이를 통해 보다 효과적인 모델 학습을 달성합니다.

- **Performance Highlights**: 실험 결과, RL4HS 모델이 사전 훈련된 추론 모델과 감독 학습(Supervised Fine-tuning) 방법보다 우수한 성능을 보여 명확히 증명되었습니다. 특히, 기존의 제너레이션 모델보다 환각 탐지에 더 효과적인 성과를 나타냈습니다. 또한, span-F1 점수 기준으로 강화 학습 접근 방식이 우월함을 흐릿한 기준에서 더 높게 평가받은 점도 주목할 만합니다.



### RESTRAIN: From Spurious Votes to Signals -- Self-Driven RL with Self-Penalization (https://arxiv.org/abs/2510.02172)
- **What's New**: 본 논문은 RESTRAIN(자기제한적 강화 학습)이라는 새로운 프레임워크를 소개합니다. RESTRAIN은 비(非)라벨 데이터에 적응하여 학습을 개선하는 경험 중심의 학습 경험을 제공합니다. 이 방법은 과도한 자신감으로 인한 잘못된 예측을 펀칭함으로써 모델의 전체 답변 분포에서 신호를 활용하고 있습니다.

- **Technical Details**: RESTRAIN은 그룹 상대 정책 최적화(GRPO) 알고리즘을 기반으로 하여 발전하였습니다. RESTRAIN의 주요 개념은 낮은 자신감을 가진 롤아웃에 대한 부정적 가중치를 부여하고, 약한 찬성 다수결을 하여 모델의 학습 신호를 개선하는 것입니다. 동일한 프롬프트에 대해 여러 예측을 생성하고, 이들 예측의 출현 빈도에 따라 가중치를 부여하는 방식으로 새로운 방법을 제공합니다.

- **Performance Highlights**: RESTRAIN을 이용한 실험에서 Qwen3-4B-Base 및 OctoThinker Hybrid-8B-Base 모델은 AIME25에서 Pass@1을 140.7% 증가시키는 성과를 거두었습니다. 이러한 결과는 RESTRAIN이 거의 금라벨 감독(supervised learning)에 근접한 성능을 달성하는 것을 보여줍니다. 이는 RESTRAIN이 비라벨 데이터로도 우수한 추론 성능을 끌어낼 수 있는 확장 가능한 방법임을 의미합니다.



### The Disparate Impacts of Speculative Decoding (https://arxiv.org/abs/2510.02128)
- **What's New**: 이 논문에서는 대형 언어 모델의 디코딩 시간을 단축시키기 위해 사용되는 'speculative decoding'의 분석을 수행합니다. 특히, 이 기법에서 각 작업(task) 간 속도 향상이 균등하게 분포하지 않는다는 점을 강조하며, 이는 저조한 피팅(fitness)을 보이는 작업에서 특히 두드러집니다. 결과적으로, 불공정한 속도 향상 문제를 해결하기 위한 완화 전략을 제안하고, 이 방법이 여러 모델 쌍에서 평균 12%의 개선을 보여줍니다.

- **Technical Details**: Speculative decoding은 'drafter' 모델이 제안하는 토큰을 'verifier' 모델이 검증하는 방식으로 작동합니다. 이 논문에서는 drafter와 verifier 모델의 조건부 토큰 분포 간의 정렬(alignment)이 속도 향상에 미치는 영향을 분석합니다. 또한, 속도 불공정성(unfairness)을 정량화할 수 있는 새로운 개념을 도출하고, drafter 모델의 피팅(fitness)과 속도 향상 간의 관계를 탐구합니다.

- **Performance Highlights**: 이번 연구의 결과는 특정 작업이 상대적으로 느린 속도 향상을 경험한다는 계산적 불공정성을 드러냅니다. 실험에서 일본어와 같은 특정 언어의 경우 낮은 정확도와 속도 향상 지표가 나타났습니다. 이러한 결과는 속도 향상과 언어별 정확도 사이의 관계를 밝혀내며, 이는 전체적인 모델 접근의 공정성에 중요한 영향을 미칩니다.



### Chain-of-Thought Reasoning in Streaming Full-Duplex End-to-End Spoken Dialogue Systems (https://arxiv.org/abs/2510.02066)
- **What's New**: 이번 연구에서 저자들은 Duplex SDS를 위한 새로운 프레임워크인 SCoT(Streaming Chain-of-Thought)를 제안합니다. SCoT는 사용자의 고정 길이 입력을 처리하고 응답을 블록 방식으로 생성함으로써, 전통적인 VAD 기반 전환 메커니즘 없이 동시에 듣고 말할 수 있는 시스템을 구현합니다. 이를 통해 기존의 Duplex 모델보다 더 일관된 응답과 더 낮은 지연 시간의 상호작용을 지원합니다.

- **Technical Details**: SCoT 프레임워크는 사용자 음성을 연속적으로 처리하고, 음성 인식(ASR) 및 텍스트 응답 생성을 위한 중간 타겟을 생성하는 데 CTC 기반 강제 정렬을 사용합니다. 이 과정에서 구간 수준 정렬을 만들어 각 블록에 대한 정렬된 사용자 성적표 및 시스템 응답을 만듭니다. 이러한 접근은 음성과 텍스트 간의 추론을 긴밀하게 결합하여, 대화의 연속성을 높이고 반응성을 개선합니다.

- **Performance Highlights**: 실험 결과, SCoT는 기존의 Duplex 방법들보다 더 지능적이고 일관된 응답을 생성하며, 훈련 효율성 또한 크게 향상됩니다. 또한, SCoT는 더 낮은 지연 시간의 상호작용을 제공하고, 이전의 turn-by-turn CoT 기반 시스템과 비교했을 때 대화의 유창성과 응답성을 개선합니다. 이 연구는 향후 개발을 위해 코드와 훈련된 모델을 오픈 소스할 계획입니다.



### Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usag (https://arxiv.org/abs/2510.02044)
- **What's New**: 본 논문은 외부 도구 호출을 통합하여 낮은 대기 시간을 유지하는 최초의 음성 입력 및 출력 시스템을 소개합니다. 기존의 텍스트 기반 시스템의 장점을 음악적으로 음성 대화 시스템(SDS)에 적용하면서 새로운 Streaming Retrieval-Augmented Generation (Streaming RAG) 프레임워크를 제안합니다. 이는 사용자 음성이 진행되는 동안 도구 쿼리를 예측하고 발신함으로써 대기 시간을 줄입니다.

- **Technical Details**: Streaming RAG는 두 가지 주요 접근 방식을 제안합니다: 고정 간격 Streaming RAG(Fixed-Interval Streaming RAG)와 모델 유도 Streaming RAG(Model-Triggered Streaming RAG)입니다. 첫 번째 접근법은 음성 입력 중 정기적으로 도구 쿼리를 전송하여 응답 품질을 보장하는 방식입니다. 두 번째 접근법은 모델이 사용자 발화의 발전에 따라 최적의 쿼리 타이밍을 결정하도록 학습시켜 계산 자원을 절약합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델 유도 Streaming RAG는 정확도를 200% 이상 증가시켰고, 20%의 도구 결과 생성 지연 시간을 줄였습니다. AudioCRAG이라는 벤치마크를 새롭게 제안하고 공개하여 향후 연구에 기여할 예정입니다. 이 연구는 단순한 음성 시스템이 아닌 모든 형태의 입력에도 적용할 수 있는 가능성을 보여줍니다.



### Style Over Story: A Process-Oriented Study of Authorial Creativity in Large Language Models (https://arxiv.org/abs/2510.02025)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 창의성을 평가하는 새로운 접근법을 제시합니다. 기존 연구들이 결과 중심의 품질 평가에만 초점을 맞춘 반면, 본 연구는 저자적 창의성을 형성하는 과정에 대한 심층 분석을 수행합니다. 특히, 제한기반 의사결정(constrain-based decision-making)을 통해 모델의 창의적 선호도를 분석하며, LLMs가 스타일을 다른 요소들보다 더 중요시한다는 사실을 발견했습니다.

- **Technical Details**: 연구에서는 200개의 내러티브 제약(narrative constraints)을 개발하여 네 가지 내러티브 요소(캐릭터, 사건, 배경, 스타일)에 걸쳐 분포시켰습니다. 각 요소는 이론적 기초에 따라 5개 카테고리로 세분화되며, 이를 통해 LLMs의 선택을 저자적 선택으로 관찰할 수 있도록 하였습니다. 연구는 또한 기본, 품질 강조, 창의성 강조의 세 가지 저자 페르소나(authorial personas)를 사용하여 여러 모델 가족(GPT, Claude 등) 간의 선호도를 비교했습니다.

- **Performance Highlights**: 결과적으로 모델들은 유사하게 스타일을 강조했으며, 선택과 그 선택의 이유를 조사함으로써 각 모델의 독특한 창의적 프로필이 드러났습니다. 이는 LLM의 생성 과정 이해와 인간-AI 협업의 새로운 방향을 제시하는 체계적인 도구로 작용할 수 있습니다. 따라서 이 연구는 LLMs의 저자적 창의성을 분석하는 데 있어 새로운 방법론을 제시합니다.



### LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Targ (https://arxiv.org/abs/2510.01995)
- **What's New**: 이 연구는 최초의 다중 작업( multi-task ) 뱅골어( Bangla ) 혐오 발언 데이터셋인 BanglaMultiHate를 도입합니다. 이 데이터셋은 혐오 발언의 유형, 심각도, 대상 등을 포함하는 대규모 수동 주석 데이터로서, 기존의 헌신적인 모델과 매우 다양하게 비교할 수 있도록 설계되었습니다. 또한, 이 연구는 고자원 언어에서 효과를 보이는 큰 언어 모델(LLMs)이 저자원 언어인 뱅골어의 혐오 발언 탐지에 적합하도록 조정할 수 있는 방법을 모색하며, 문화적 및 언어적 사전 학습의 중요성을 강조합니다.

- **Technical Details**: BanglaMultiHate 데이터셋은 다중 작업을 지원하도록 설계되어 있으며, 혐오 발언의 유형, 심각도 및 대상 식별을 포함합니다. 실험에서는 SVM, BanglaBERT, LLaMA 및 Qwen 모델을 사용하여 저자원 환경에서 LLM의 적응성을 평가했습니다. 연구를 통해 Fine-tuning이 수행된 BanglaBERT가 최고의 성능을 나타내며, SVM이 심각도 및 대상에 대한 과제를 상대적으로 잘 처리하고, LLaMA가 유형 식별에서 약간 더 나은 성능을 보였습니다.

- **Performance Highlights**: 연구 결과는 파인튜닝된 BanglaBERT가 다른 모델들에 비해 우수한 성능을 나타냄을 보여줍니다. 제로샷( zero-shot ) 학습은 SVM 및 기본 성과와 비교하여 더 나은 성능을 보이지 못했습니다. 모델 성능은 작업의 복잡도에 따라 상당히 달라지며, 이는 설계된 모델이 뱅골어의 문화적 정서를 반영할 수 있는 중요성을 다시 한번 강조합니다.



### Exploring Database Normalization Effects on SQL Generation (https://arxiv.org/abs/2510.01989)
Comments:
          Accepted to CIKM 2025

- **What's New**: 이 논문은 자연어(SQL) 변환(NL2SQL) 시스템에서 스키마 디자인, 특히 정규화(normalization)의 영향에 대한 첫 번째 체계적인 연구를 제공합니다. 기존 연구들은 고정된 스키마를 기반으로 모델을 평가하였으나, 본 연구에서는 정규화 수준이 다양한 인공 및 실제 데이터셋에 대해 8개의 주요 대형 언어 모델을 평가합니다. 이 연구는 스키마 디자인이 NL2SQL 성능에 미치는 영향을 처음으로 체계적으로 조사한 결과, 데이터베이스 구조가 어떻게 최적화될 수 있는지에 대한 통찰을 제공합니다.

- **Technical Details**: 정규화는 데이터 중복을 최소화하고 업데이트 이상(anomaly)을 방지하기 위해 데이터를 작고 통합된 테이블로 분해하는 데이터베이스 설계 원칙입니다. 본 연구는 1NF에서 3NF까지의 정규화 수준을 다루며, 이를 통해 NL2SQL 시스템에서의 쿼리 생성과 관련된 오류가 어떻게 발생하는지에 대한 심층적인 분석을 수행합니다. 연구 방법론에서는 인공 데이터 및 실제 데이터로 구성된 세 가지 실험 설정(Formal-Basic, Formal-Simulated, Practical-Real)을 통해 정규화가 SQL 생성에 미치는 영향을 평가합니다.

- **Performance Highlights**: 연구 결과에 따르면, 단일 테이블을 사용하는 비정규화된 스키마는 간단한 조회 쿼리에 대해 높은 NL2SQL 성능을 보입니다. 반면, 정규화된 스키마는 집계 쿼리에서 더 나은 성능을 보이며, 중복 및 NULL 값 문제에 강한 내성을 보여주었습니다. 이러한 발견은 NL2SQL 성능이 쿼리 유형과 정규화 수준에 따라 달라지며, 실제 데이터베이스 설계에서 스키마를 작업 부하와 모델 능력에 맞게 조정해야 함을 강조합니다.



### Taking a SEAT: Predicting Value Interpretations from Sentiment, Emotion, Argument, and Topic Annotations (https://arxiv.org/abs/2510.01976)
Comments:
          Accepted at VALE workshop (ECAI 2025)

- **What's New**: 본 연구는 AI 시스템이 다양한 인간의 가치 해석에 맞추어 공정성을 유지하는 방안을 모색합니다. 특히, 개인의 과거 주석을 바탕으로 한 세부적 접근 방식인 SEAT(Sentiment, Emotion, Argument, Topics) 차원을 활용하여 언어 모델이 개인의 가치 해석을 예측할 수 있는지를 조사합니다. 이 연구는 인구 통계학적 정보 대신 주석자의 행동적 신호에 주목하여 AI의 개인 맞춤화 가능성을 제시합니다.

- **Technical Details**: 연구에서는 여러 사람의 데이터를 수집하여 SEAT 차원에 대한 주석을 바탕으로 언어 모델이 텍스트의 인간 가치 해석을 어떻게 예측하는지를 분석합니다. 특정 개인이 제공한 주석을 다양한 수준으로 제공함으로써 언어 모델의 예측 성능이 어떻게 변화하는지를 실험합니다. 결과적으로 SEAT 차원의 전체 세트를 몇 가지 예제로 제공할 때 개인 가치 해석 예측의 성능이 향상됩니다.

- **Performance Highlights**: 본 연구는 개인의 가치 해석에 대한 예측을 향상시키기 위해 다차원 주석 이력을 활용하는 첫 번째 시도입니다. 이번 연구 결과는 기존의 인구 통계학적 접근법을 넘어선 개인별 행동 신호를 통한 SENAT 차원의 영향을 강조합니다. 이러한 접근 방식은 가치 해석의 반대 사실적 시나리오를 제공함으로써 참가자 간의 개선된 논의 지원에 기여할 수 있습니다.



### Veri-R1: Toward Precise and Faithful Claim Verification via Online Reinforcement Learning (https://arxiv.org/abs/2510.01932)
- **What's New**: 이번 논문에서는 Veri-R1이라는 새로운 온라인 강화 학습(Reinforcement Learning, RL) 프레임워크를 도입하였으며, 이는 대규모 언어 모델(Large Language Model, LLM)이 검색 엔진과 상호작용하고 보상 신호를 통해 계획, 검색 및 추론 행동을 형성하도록 지원합니다. 이는 LLM이 정보를 검색하고 유효성을 평가하는 과정을 보다 정확하게 반영하여 실질적인 검증 능력 향상에 기여합니다. 이를 통해 이전 연구들에서 보여준 모델의 한계를 극복하고, LLM의 자동 검증 효율성을 높이는 것이 목표입니다.

- **Technical Details**: Veri-R1은 통합된 파이프라인 내에서 LLM의 온라인 클레임 검증 능력을 강화하기 위해 설계되었습니다. 기존의 감독 하의 미세 조정(Supervised Fine-Tuning, SFT)과는 달리, RL 방식을 채택하여 명시적인 추론 궤적 없이 훈련을 진행합니다. 훈련 과정에서 LLM은 여러 번의 턴을 거치며 검색하고 추론한 후 최종 답변을 생성하도록 요구되며, 고품질 데이터셋인 FEVEROUS와 EX-FEVER에서 샘플을 고르고 필터링하여 사용합니다.

- **Performance Highlights**: 실험 결과, Veri-R1 파이프라인의 온라인 RL 모델은 다른 훈련 방법에 비해 최적의 성능을 보여주었으며, 조합 정확도에서 최대 30%의 절대 증가를 기록했습니다. 검증 정확도는 23% 향상되었고, 레이블 정확도는 22% 개선되었습니다. 또한, 증거 점수 증대 측면에서도 최대 150% 향상된 결과를 보였으며, 보상 구성 요소가 모델의 황금 증거를 식별하는 능력을 효과적으로 향상시켰음을 보여주었습니다.



### Inverse Language Modeling towards Robust and Grounded LLMs (https://arxiv.org/abs/2510.01929)
- **What's New**: 최근 LLMs(대형 언어 모델)의 방어 메커니즘이 단편적이고 개발이 미흡함을 지적한 본 논문은, Inverse Language Modeling (ILM)이라는 통합 프레임워크를 제안합니다. 이 프레임워크는 LLMs의 입출력 견고성을 동시에 강화하며, 잠재적으로 유해한 입력 트리거를 식별하는 기능을 포함합니다. ILM은 LLMs를 정적 생성기에서 분석 가능하고 견고한 시스템으로 변환하는 것을 목표로 하고 있습니다.

- **Technical Details**: ILM은 두 가지 주요 목표를 가지고 있습니다. 첫째, 입력 변형(input perturbations)에 대한 LLMs의 저항력을 개선하고, 둘째, 모델 출력을 역전시켜(native grounding) 위험한 입력을 식별하는 것입니다. 이러한 특성 덕분에 ILM은 LLM의 통제 가능성과 신뢰성을 더욱 높일 수 있는 기초를 마련합니다.

- **Performance Highlights**: ILM은 RED 팀이 필요로 하는 분석 가능성과 견고성을 제공하여, 차세대 LLM이 더욱 통제 가능하고 신뢰할 수 있는 방향으로 발전할 수 있도록 합니다. 이러한 혁신적인 접근법은 LLM의 방어 메커니즘 개선에 크게 기여할 것으로 전망됩니다.



### Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey (https://arxiv.org/abs/2510.01925)
- **What's New**: 이 논문에서는 보상 모델(Reward Models, RMs)의 역할과 기여를 체계적으로 소개하고 있습니다. RMs는 대규모 언어 모델(LLMs)의 추론 성능을 향상시키기 위해 중요하며, 강화 학습(Reinforcement Learning, RL) 및 여러 후보 중 최적의 답변 선택에 사용됩니다. 저자들은 RMs의 아키텍처, 훈련 방법론, 평가 기술에 대한 기본 개념을 리뷰하고, 다양한 애플리케이션을 탐구합니다.

- **Technical Details**: RMs는 구별형(discriminative) RM과 생성형(generative) RM 두 가지 주요 유형으로 나뉘며, 각각 입력 쿼리에 대한 점수를 매기거나 보상에 맞는 생성을 수행합니다. 과정 보상 모델(Process Reward Models, PRMs)은 단계별로 보상을 제공하며, 결과 보상 모델(Outcome Reward Models, ORMs)은 전체 응답을 평가합니다. 이 논문에서는 RMs가 LLM의 추론을 어떻게 개선하는지에 대한 세 가지 주요 응용 프로그램을 다룹니다.

- **Performance Highlights**: 연구에서 제안된 RMs는 특히 복잡한 추론 작업에서 유용성을 보여주며, 기존 데이터와 문제 해결을 기반으로 더 나은 성과를 발휘합니다. 생성형 RMs는 일반적으로 구별형 RMs보다 뛰어나지만, 훈련과 배포에서 더 높은 비용이 드는 단점이 있습니다. 그러나 PRMs는 각 추론 단계에 대한 세부적인 피드백을 제공하지만, 훈련 데이터의 양이 부족하여 노이즈가 발생할 수 있습니다.



### REPAIR: Robust Editing via Progressive Adaptive Intervention and Reintegration (https://arxiv.org/abs/2510.01879)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 고비용 지식 업데이트와 재훈련으로 인한 부작용을 완화하기 위해 REPAIR(강력한 편집을 위한 점진적 적응 개입 및 재통합)을 제안합니다. REPAIR는 정밀하고 저비용의 모델 업데이트를 지원하도록 설계된 생애 주기 편집 프레임워크입니다. 이 프레임워크는 폐쇄 루프 피드백 메커니즘과 동적 메모리 관리 기능을 결합하여 대규모 연속 편집의 불안정성과 충돌을 완화하는 데 기여합니다.

- **Technical Details**: REPAIR는 (1) 동적 메모리 관리를 통한 폐쇄 루프 오류 피드백, (2) 샘플 유사성에 기반한 배치 재조합 및 내부 배치 지식 증류, (3) 손실 인식 가중치 지식 통합을 포함한 혁신적인 모델 편집 방법론을 가지고 있습니다. 이러한 방식은 모델 학습의 전반적인 신뢰성, 범위 및 지역성을 최적화하는 목표를 달성합니다. 모델 업데이트는 고유의 하이퍼파라미터 조정으로 다양한 차원에서 적용됩니다.

- **Performance Highlights**: REPAIR는 다양한 모델(LLaMA-3, Qwen-2.5, DeepSeek-R1-1.5B 및 GPT-2-XL 등)에서 15%-20%의 전반적인 편집 성능 개선을 입증하며, 기존 최첨단 방법들과의 비교에서 일관되고 강력한 일반화를 보였습니다. REPAIR의 구조적 혁신은 지식 중복과 손실을 대폭 낮춰주며, 이는 레이아웃 내의 편집 문제 해결에서 중요한 의미를 갖습니다.



### Model Merging to Maintain Language-Only Performance in Developmentally Plausible Multimodal Models (https://arxiv.org/abs/2510.01845)
Comments:
          Accepted to the EMNLP 2025 workshop BabyLM: Accelerating language modeling research with cognitively plausible datasets

- **What's New**: 이번 논문에서는 BabyLM 챌린지의 멀티모달 트랙에 대해 언급하면서, 고급 언어 모델이 어린이들이 언어를 습득할 때 접하는 데이터보다 훨씬 많은 양의 데이터를 필요로 한다고 설명합니다. 우리는 개발상 가능한 데이터셋을 사용하여 저자원이 환경에서 언어 전용 및 멀티모달 모델을 개발하였으며, 멀티모달 모델이 기존 BabyLM 베이스라인을 초과 성능을 보였습니다. 특히, 멀티모달 모델이 언어 전용 작업에서 저조한 성능을 보인다는 점에 초점을 맞추어 모델 병합(model merging) 기술을 실험하여 문제를 일부 해결하고자 했습니다.

- **Technical Details**: 이번 연구는 언어 전용 및 멀티모달 모델을 개발하는 데 중점을 두고 있으며, 두 모델의 매개변수를 가중 선형 보간(weighted linear interpolation) 방식으로 융합하는 모델 병합 방법을 사용했습니다. 이러한 접근은 훈련 과정이 필요 없는 직관적인 방식으로 모델의 정확성과 견고성을 유지하는 데 기여했습니다. 멀티모달 모델의 훈련 단계에서 언어 전용 벤치마크에서의 성능 저하 문제를 확인하고, 이를 극복하기 위한 모델 증강 기법으로 모델 병합을 적용했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 언어 전용 벤치마크에서 멀티모달 모델의 성능 저하가 관찰되었으며, 모델 병합을 통해 언어 전용 및 멀티모달 두 벤치마크에서의 성능을 유지할 수 있었습니다. 이로써 멀티모달 모델이 텍스트 전용 작업에서도 경쟁력을 갖출 수 있는 가능성을 보여주었습니다. 저자원 환경에서도 적합한 성과를 내며, 언어 모델링의 연구에 기여할 수 있는 방향성을 제시했습니다.



### SCRIBES: Web-Scale Script-Based Semi-Structured Data Extraction with Reinforcement Learning (https://arxiv.org/abs/2510.01832)
- **What's New**: 이 논문에서는 SCRIBES(SCRIpt-Based Semi-Structured Content Extraction at Web-Scale)라는 새로운 강화 학습 프레임워크를 소개합니다. 이 프레임워크는 동일한 사이트 내 여러 웹페이지의 레이아웃 유사성을 보상 신호로 활용하여 반구조화된(semistructured) 콘텐츠를 대규모로 추출할 수 있도록 설계되었습니다. SCRIBES는 각 페이지를 개별적으로 처리하는 대신 재사용 가능한 추출 스크립트를 생성하여 구조적으로 유사한 웹페이지 그룹에 적용합니다.

- **Technical Details**: SCRIBES는 관련된 웹페이지의 구조적 유사성을 활용하여 반복적으로 훈련합니다. 이때 사용되는 데이터는 두 가지 출처에서 오며, 첫째는 소수의 주석이 달린 예제가, 둘째는 CommonCrawl과 같은 대규모 데이터 세트입니다. 이 접근 방식은 주석의 필요성을 줄이고, 레이블이 없는 데이터에서도 학습하는 데 도움을 줍니다.

- **Performance Highlights**: SCRIBES 방식은 강력한 기준선보다 13% 이상 높은 품질의 추출 스크립트를 생성하며, downstream에서의 질문 응답 정확도를 4% 이상 향상시킵니다. 이 모델은 특히 GPT-4o와 같은 최첨단 모델에서도 응용되어 확장 가능하고 자원 효율적인 웹 정보 추출을 가능하게 합니다.



### Syntactic Blind Spots: How Misalignment Leads to LLMs Mathematical Errors (https://arxiv.org/abs/2510.01831)
Comments:
          14 pages, 5 Tables, 9 Figures; Accepted to MathNLP 2025: The 3rd Workshop on Mathematical Natural Language Processing (co-located with EMNLP 2025)

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 수학 문제 해결에 강한 능력을 보이지만, 훈련된 분포에서 벗어난 구문에서 문제를 처리할 때 체계적으로 실패한다고 주장합니다. 특히 구문적 맹점(syntactic blind spots)이라 불리는 실패 모드를 확인하였으며, 이는 모델이 세부 변화에 적절하게 대응하지 못하고 친숙한 추론 방식으로 문제를 잘못 적용함으로써 발생합니다. 이러한 오류는 수학적 능력의 부족이 아니라, 표면 형식과 내부 표현 간의 취약한 연결로 인한 것임을 강조합니다.

- **Technical Details**: 논문에서는 의존성 지역성 이론(Dependency Locality Theory, DLT)을 사용하여 구문의 복잡성을 정량화하고, 잘못된 질문을 성공적인 예시에서 추출한 구문 템플릿을 사용하여 재구성(rephrase)하였습니다. 구조적 복잡성을 줄인 재구성을 통해 종종 정답에 이르게 되는 현상을 보여줍니다. 이 과정에서 각 질문의 의존도(tree)에 따라 다양한 비용을 산정함으로써 기존 질문들을 체계적으로 점수화하는 방법론을 개발했습니다.

- **Performance Highlights**: 연구를 통해 구문적 오류는 개념적 어려움이 아닌 구조적 불일치에서 비롯된다는 것을 제안하며, 구문 인식이 가능한介入(intervention) 방안이 이러한 유도 실패를 드러내고 완화할 수 있다고 주장합니다. 여러 데이터 세트에 대한 정확도가 구조적으로 복잡한 수학 질문의 재구성을 통해 크게 개선된 결과를 보였으며, 이는 LLM의 정확성을 높이는 효과적인 전략이라고 할 수 있습니다.



### Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network (https://arxiv.org/abs/2510.01801)
- **What's New**: 대규모 언어 모델(LLM)의 발전으로 인해 인간의 글을 유사하게 모방하는 설득력 있는 스팸 리뷰가 생성되는 문제가 대두되었습니다. 이러한 리뷰는 기존 감지 시스템에 도전 과제를 제기하며 온라인 플랫폼의 신뢰성을 위협하고 있습니다. 본 연구에서는 세 가지 LLM을 사용하여 현실적인 LLM 생성 스팸 리뷰 데이터셋을 생성하고, 이 리뷰들의 높은 설득력과 기만적 가능성을 입증하였습니다.

- **Technical Details**: 본 연구에서는 FraudSquad라는 하이브리드 감지 모델을 제안합니다. 이 모델은 사전 훈련된 언어 모델의 텍스트 임베딩과 게이티드 그래프 트랜스포머를 결합하여 스팸 노드 분류를 수행합니다. FraudSquad는 수동 특징 공학이나 대규모 훈련 자원에 의존하지 않고, 의미적 신호 및 행동 신호를 모두 포착할 수 있는 디자인이 특징입니다.

- **Performance Highlights**: 실험 결과, FraudSquad는 세 가지 LLM 생성 데이터셋에서 정밀도에서 최대 44.22%, 재현율에서 최대 43.01%까지 최신 상태의 기준선보다 우수한 성능을 보였습니다. 향후 실제 애플리케이션에서도 적은 레이블된 훈련 데이터로 높은 감지 정확도를 유지하며, 특허한 상용 모델 사이즈를 자랑합니다. 이를 통해 통화 및 데이터의 사용에 대한 긴급성을 강조하며, LLM 시대에 맞춘 스팸 검출의 중요성을 명확히 합니다.



### Comparison of Unsupervised Metrics for Evaluating Judicial Decision Extraction (https://arxiv.org/abs/2510.01792)
Comments:
          28 pages

- **What's New**: 이번 연구는 러시아의 1,000건의 익명 법원 판결문에서 7개의 의미적 블록을 추출하는데 필요한 16개의 비지도 학습 메트릭을 평가합니다. 이러한 메트릭은 문서 기반, 의미적, 구조적, 법률 특정 카테고리로 분류되며, 사전 주석이 없는 상태에서 작동합니다. 연구 결과, Term Frequency Coherence와 Coverage Ratio/Block Completeness가 전문가 평점과 가장 잘 일치하는 것으로 나타났습니다.

- **Technical Details**: 연구 방법론에서는 비지도 메트릭을 사용하여 법원 판결문 추출 품질을 평가하며, 이는 JSON 형식으로 구성된 문서에서 수행됩니다. 각 판결문은 원본 텍스트와 사전 세분화된 JSON 객체 형태로 제공되어, 비지도 메트릭을 계산하는 기준(reference extraction) 역할을 합니다. 평가에 참여한 법률 전문가들은 1-5 Likert 척도를 사용해 각 블록을 평가하였으며, 서로 간의 신뢰성 평가를 위해 ICC를 사용했습니다.

- **Performance Highlights**: 연구는 법률 텍스트 추출을 위한 비지도 평가 메트릭이 아직까지 개발이 미흡하고 법률 특유의 뉘앙스를 포착하지 못할 가능성이 있음을 강조합니다. 전문가 평가 결과, Court evaluation of evidence 블록에서는 ICC 값이 0.86으로 높은 일치를 보였지만, Court decision 블록에서는 0.70으로 다소 낮은 수치를 기록했습니다. 이러한 결과는 법적 맥락에서 기술이 사람의 판단을 완전히 대체할 수는 없음을 시사합니다.



### Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks (https://arxiv.org/abs/2510.01782)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)이 자기가 알지 못하는 질문에 대해 거부할 수 있는 능력인 지식 기반 거부(Knowledge-aware refusal)를 정의하고 이를 측정하기 위한 새로운 메트릭인 거부지수(Refusal Index, RI)를 제안합니다. 기존의 메트릭이 이러한 능력을 제대로 측정하지 못하는 문제를 해결하기 위해, RI는 거부 확률과 오류 확률 간의 Spearman 순위 상관관계를 통해 정의됩니다.  이 방식은 모델의 실제 거부 행동을 직접적으로 측정할 수 있게 하여, 과거의 불완전한 메트릭과 비교하여 신뢰성을 높입니다.

- **Technical Details**: RI는 지식 기반 거부를 평가하기 위해 두 가지 주요 특성을 지닙니다. 첫째, RI는 지식 기반 거부를 정확하게 추정하며, 거부율에 무관하게 공정한 측정을 제공합니다. 둘째, RI는 경량 평가 방법을 통해 기존 평가 파이프라인과 호환되어 쉽게 측정할 수 있습니다. 이 방법은 두 번의 표준 평가 과정을 통해 모델의 거부 능력을 정량화하며, 16개 모델과 5개의 데이터셋을 통해 광범위한 실험을 수행하여 RI의 효과성을 검증합니다.

- **Performance Highlights**: 실험 결과 RI는 모델이 알지 못하는 질문에 대한 거부 능력을 정확하게 정량화하며, 다양한 거부율에서도 안정성을 유지합니다. RI는 모델의 전반적인 정확도 및 거부율에 의존하지 않고 일관된 모델 순위를 제공합니다. 또한 RI는 전통적인 정확성 메트릭과 비교하여 LLM의 사실성(factuality) 평가에서 간과된 중요한 차이를 밝혀내며, 모델의 신뢰도를 평가하기 위해 지식 기반 거부 측정을 포함할 필요성을 강조합니다.



### Machine-interpretable Engineering Design Standards for Valve Specification (https://arxiv.org/abs/2510.01736)
Comments:
          22 pages, 10 figures, 4 tables

- **What's New**: 이번 논문에서는 엔지니어링 설계 표준에 저장된 정보를 모듈형 재사용 가능하고 기계 해석 가능한 온톨로지(ontology)로 변환하는 방법을 보여줍니다. 이를 통해 품질 보증 과정에 활용할 수 있으며, 특히 밸브 선택 과정에 적용됩니다. 기존의 문서형 표준이 아닌 데이터 중심으로 접근하는 혁신적인 시도를 제시합니다.

- **Technical Details**: 논문은 API, ASME, ASTM의 엔지니어링 설계 표준에 중점을 두고, 기계 해석 가능한 데이터 생성을 통해 밸브 설계 및 제품 사양 자동화를 목표로 합니다. ASME B16.34와 같은 표준의 데이터와 규칙을 구조화하여, W3C 준수 형식으로 교환 가능한 모듈형 온톨로지를 생성합니다. 이러한 접근 방식을 통해 표준의 디지털화가 가능해지며, 국제 표준에 부합하는 동작을 보장합니다.

- **Performance Highlights**: 연구를 통해 생성된 온톨로지는 밸브 선택 과정에서 사용되어, 안전하고 효율적인 설계 품질 보증을 자동화합니다. 또한, 공유 가능한 IDO 기반의 모듈형 온톨로지는 설계 표준에 대한 의미론적 추론을 가능하게 하며, 스마트 기준으로의 전환을 추구하는 표준 기관에 유용성을 보여줍니다. 이를 통해 엔지니어링 디자인 프로세스의 효율성과 품질을 크게 향상시킬 수 있습니다.



### What MLLMs Learn about When they Learn about Multimodal Reasoning: Perception, Reasoning, or their Integration? (https://arxiv.org/abs/2510.01719)
- **What's New**: 이번 논문은 다중 모달(Multimodal) 추론 모델에서 새로운 벤치마크인 MathLens를 소개합니다. 이 벤치마크는 기하학(geometry) 문제의 복잡성을 유지하면서도 추론의 하위 기술(subskills)을 분리하여 평가할 수 있는 환경을 제공합니다. MathLens는 정보 추출(Perception), 가용 정보에 대한 작업(Reasoning), 관련된 지각 증거의 선택 및 적용(Integration)이라는 세 가지 구성 요소로 성능을 나누어, 모델의 발전을 더 잘 이해할 수 있게 합니다.

- **Technical Details**: MathLens는 926개의 기하학 문제로 구성되어 있으며, 각 문제는 평균 7.03개 이상의 시각적 프로브(visual probes)를 포함합니다. 이 벤치마크는 시맨틱 상태(semantic state)와 쿼리 연산자(query operator)를 포함한 문제 정의를 활용하여, 정확한 문제 해결을 요구하는 다양한 질문 유형을 생성합니다. 이를 통해 감지, 추론 및 통합을 테스트하는 효과적인 방법을 제공합니다.

- **Performance Highlights**: 이 연구는 다양한 훈련 접근 방식이 다중 모달 추론 능력에 미치는 영향을 분석한 결과를 제시합니다. 특히, 강화 학습(Reinforcement Learning)과 텍스트 SFT(Supervised Fine Tuning)의 조합이 시각적 입력에 대한 감지 향상에 기여하며, 통합(Integration) 능력의 개선은 가장 미비한 것으로 나타났습니다. 각기 다른 훈련 전략이 성능에 미치는 영향을 파악함으로써, MathLens는 향후 연구와 모델 개발에 유용한 참고자료가 될 것입니다.



### Format Inertia: A Failure Mechanism of LLMs in Medical Pre-Consultation (https://arxiv.org/abs/2510.01688)
Comments:
          EMNLP 2025 Industry Track

- **What's New**: 최근 대형 언어 모델 (Large Language Models, LLMs)의 발전은 챗봇과 의료 전진 상담 애플리케이션 등 다양한 서비스 영역에서 상당한 개선을 가져왔습니다. 이러한 연구에서는 의료 도메인에서 LLMs를 다중 턴 대화 생성에 적응시키는 가장 일반적인 방법인 감독 세부 조정 (Supervised Fine-Tuning, SFT)에 초점을 맞추고 있습니다.

- **Technical Details**: SFT를 위한 데이터셋은 턴 수 분포에 불균형을 보이는 경향이 있습니다. 이러한 데이터에서 훈련할 경우, 우리는 '형식 관성 (Format Inertia)'이라 부르는 새로운 실패 메커니즘이 유도된다는 것을 발견했습니다. 이 메커니즘은 모델이 긴 의료 대화 중 반복적인 질문을 생성하게 만듭니다.

- **Performance Highlights**: 이를 해결하기 위해 우리는 훈련 데이터셋의 턴 수 분포를 재조정하는 간단하고 데이터 중심의 방법을 채택했습니다. 실험 결과, 우리 접근법이 의료 전진 상담에서 형식 관성을 상당히 완화한다는 것을 보여주었습니다.



### How Do Language Models Compose Functions? (https://arxiv.org/abs/2510.01685)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 조합(compositional) 작업을 수행하는 방식에 대한 검토를 진행합니다. 특히, 두 단계 사실 회상(two-hop factual recall) 작업을 통해 LLM들이 조합 메커니즘을 사용하고 있는지, 아니면 비조합적(idiomatic) 방식으로 해결하고 있는지를 조사합니다. 연구 결과, LLM들은 조합적 처리(compositional processing) 및 직접 처리(direct processing) 메커니즘을 모두 사용하며, 이는 임베딩 공간의 기하학(geometry)과 관련이 있음을 발견했습니다.

- **Technical Details**: 연구에서는 조합적 회귀 작업을 $g(f(x))$ 형태로 표현하고, LLM들이 이 작업을 해결하는 방식을 조사합니다. 실험에서는 여러 기능(예: 산술, 사실 회상, 번역 등)을 포함하는 10개의 인-컨텍스트 예제(in-context examples)를 샘플링하여 사용했습니다. 이러한 작업들은 입력 $x$로부터 중간 변수 $z=f(x)$를 계산한 다음 최종 결과 $y=g(f(x))$에 도달하는 방식으로 설계되었습니다.

- **Performance Highlights**: 연구 결과에 따르면 LLM들이 $x 	o f(x)$ 및 $f(x) 	o g(f(x))$를 계산할 수 있어도, $x 	o g(f(x))$를 계산할 수 있는 것은 아니며, 이는 "조합성 간극(compositionality gap)"의 지속성을 보여줍니다. 또한 어떤 계산 메커니즘이 사용되는지는 입력 임베딩 공간의 구조에 따라 달라지며, 선형 매핑(linear mapping)이 존재하는 경우 LLM은 조합적 처리보다 직접 computation을 선호하는 경향이 있음을 확인했습니다.



### FOR-Prompting: From Objection to Revision via an Asymmetric Prompting Protoco (https://arxiv.org/abs/2510.01674)
- **What's New**: 이번 논문에서는 FOR-Prompting(From Objection to Revision Prompting)이라는 비대칭 프로토콜을 제안합니다. 이 프로토콜은 방어자(Defender)가 답변을 제안하고, 반대자(Objectioner)가 질문 형식으로 반론을 제기하며, 주최자(Host)가 일관성과 마무리를 강제합니다. FOR-Prompting은 자기 수정(self-revision)을 유도하는 외부 질문 메커니즘을 통해 발전된 사고 과정을 촉진합니다.

- **Technical Details**: FOR-Prompting는 역할 기반 상호작용 루프를 기반으로 하여, 질문이 오직 질문 형식만을 취하도록 설정하였습니다. 이를 통해 질문을 통해 사고 과정을 향상시키기 위한 체계적인 연구가 가능해집니다. 이 메커니즘은 모델에 구애받지 않으며, 재훈련 없이 다양한 크기의 호스팅된 모델 및 로컬 모델에서 작동할 수 있습니다.

- **Performance Highlights**: FOR-Prompting는 GSM8K에서 단일 프롬프트(single prompt)보다 약 22%의 정확도 향상을 기록하였고, CoT와 유사한 정확도를 달성하였습니다. 작은 규모의 모델에서는 약 19%의 정확도 향상을 보여주었으며, 복잡한 질문에 대해서도 도구나 인간 감독 없이 오류를 수정할 수 있는 능력을 보여줍니다. 또한 개방형 질문에 대한 평가에서도 기존 프롬프트보다 질적 차원에서 우수한 성능을 발휘했습니다.



### MDSEval: A Meta-Evaluation Benchmark for Multimodal Dialogue Summarization (https://arxiv.org/abs/2510.01659)
Comments:
          Accepted by EMNLP 2025

- **What's New**: 이 논문에서는 멀티모달 대화 요약(Multimodal Dialogue Summarization, MDS) 분야를 위한 첫 번째 메타 평가 벤치마크인 MDSEval을 제안하고 있습니다. MDSEval은 이미지 공유 대화와 그에 해당하는 요약, 그리고 여덟 가지 품질 측면에 대한 인간 평가를 포함합니다. 이 데이터셋은 MDS 모델의 효과적인 개발을 뒷받침하는 강력한 자동 평가 방법을 지원할 기반이 됩니다.

- **Technical Details**: MDSEval은 198개의 고품질 이미지 공유 대화로 구성되며, 각 대화에는 최첨단 멀티모달 대형 언어 모델(MLLM)에 의해 생성된 다섯 개의 요약이 결합됩니다. 이 연구에서는 정보 균형(information balance), 주제 진행(topic progression) 등의 새로운 평가 측면을 정의하여, 멀티모달 요약에서의 통합적인 이해를 강조합니다. 또한, 상호 배타적 핵심 정보(Mutually Exclusive Key Information, MEKI)를 이용한 필터링 프레임워크를 도입하여 데이터의 품질을 보장합니다.

- **Performance Highlights**: MDSEval을 통해 최신 멀티모달 평가 기법의 성능을 벤치마킹한 결과, 현재의 기법들이 요약을 적절히 구분하는 데 어려움을 겪고 있으며, 상당한 편향(bias)을 드러내었습니다. 이는 더욱 정교한 멀티모달 평가 방법의 개발에 대한 통찰을 제공합니다. MDSEval은 향후 멀티모달 대화 에이전트 개발을 위한 기초자료로 활용될 것입니다.



### SoK: Measuring What Matters for Closed-Loop Security Agents (https://arxiv.org/abs/2510.01654)
- **What's New**: 이번 논문에서는 사이버 보안에 있어 주도권을 가진 AI 시스템의 발전을 다룬다. CLASP라는 새로운 프레임워크를 도입하여 보안 생애 주기와 에이전트 기능을 연결하고, 이를 통해 성능을 평가하는 방법을 제시한다. 또한, 폐쇄 루프(CLOSED LOOP) 시스템의 필요성과 이를 효과적으로 측정하기 위한 점수 체계를 개발했습니다.

- **Technical Details**: CLASP는 보안 기능의 복잡성과 에이전트 능력을 함께 정량화하는 프레임워크로, 탐색(Reconnaissance), 취약점 활용(Exploitation), 원인 분석(Root Cause Analysis) 등의 작업을 포함한다. 각 보안 단계에 대한 에이전트의 능력을 계획(Planning), 기억(Memory), 추론(Reasoning) 등으로 정의하여 평가한다. 이를 기반으로, 클로즈드 루프 효과성 및 효율성을 측정하는 CLC 점수를 도입하였다.

- **Performance Highlights**: CLASP 프레임워크를 적용하여 21개의 대표적인 연구를 분석하고, 각각의 시스템이 어디에서 강점을 보이는지를 파악하였다. 이를 통해 발견된 기능적 보안 단계와 에이전트 능력 간의 조합으로 견고한 운영이 가능하다는 점을 강조하였다. 이 연구는 사이버 보안 시스템의 성능 평가 및 개선을 위한 기초 자료를 제공하여, 더 나아가 전반적인 보안 대책의 효율성을 높이는 데 기여할 것이다.



### Learning to Look at the Other Side: A Semantic Probing Study of Word Embeddings in LLMs with Enabled Bidirectional Attention (https://arxiv.org/abs/2510.01652)
- **What's New**: 이번 논문에서는 LLMs의 단방향 주의력(mechanism of unidirectional attention) 제한을 극복하기 위해 양방향 주의력(bidirectional attention)을 도입한 실험을 다룹니다. 기존의 단방향 Decoder-only 모델과 차별화된 접근을 통해 단어 의미 표현의 품질을 향상시킬 수 있는지를 살펴보았습니다. Llama 아키텍처의 다양한 변형을 통해 추가 학습 단계를 수행하며, 양방향 주의력의 효과를 분석하였습니다.

- **Technical Details**: 논문에서는 Llama 아키텍처의 변형을 사용하여 LLMs의 단어 임베딩 품질을 평가합니다. Llama의 기본 아키텍처에 양방향 주의력과 대조 학습(contrastive learning)을 적용한 세 가지 설정에서 성능을 비교했습니다. 특히, 양방향 주의력이 Llama 임베딩의 성능을 향상시키지 않지만, 대조 학습 기법을 통해 모델이 더욱 다양한 문맥 정보를 포착하도록 도와주었습니다.

- **Performance Highlights**: 실험 결과, 양방향 주의력이 Llama 임베딩의 성능을 굳이 증대시키진 않았지만, 대조 학습 기법을 통해 오른쪽 문맥 정보 표현이 개선되었습니다. 그러나 이는 왼쪽 문맥의 표현을 약화시키는 경향이 있음을 나타냈습니다. 모든 레이어에서 이 방안을 적용할 경우 기하학적 비대칭성(anisotropy)이 더욱 두드러지게 나타나, 임베딩 벡터의 유사도가 상승하는 결과를 보였습니다.



### NLP Methods for Detecting Novel LLM Jailbreaks and Keyword Analysis with BER (https://arxiv.org/abs/2510.01644)
- **What's New**: 이 논문은 Large Language Models (LLMs)와 관련된 'jailbreak' 프롬프트 문제를 다루고 있습니다. 특히, 다양한 머신 러닝 모델의 jailbreak 프롬프트와 진짜 사용 사례를 구별하는 능력을 분석합니다. 연구 결과, Bidirectional Encoder Representations from Transformers (BERT) 모델을 활용한 세밀한 조정이 jailbreak 식별에서 가장 좋은 성과를 보였습니다.

- **Technical Details**: 연구에서는 여러 데이터셋을 사용하여 프롬프트의 범주화와 탈옥 패턴을 분석합니다. 특히, 기존 데이터를 기반으로 새로운 탈옥 프롬프트를 식별하는 방법을 모색하며, back translation 및 synonym substitution과 같은 데이터 증강 방법을 도입합니다. 논문은 LLM의 현재 상태와 시민적인 기준을 보장하기 위한 후속 교육 과정의 중요성도 강조합니다.

- **Performance Highlights**: 이 연구는 jailbreak 탐지 성능을 평가하기 위해 머신 러닝 모델을 결합하고, 이전에 보지 못한 새로운 jailbreak 전략을 식별하는 데 초점을 맞춥니다. 특히, BERT 모델을 사용하는 것이 높은 정확도로 이어졌으며, 탈옥 프롬프트를 구별하는 데 있어 중요한 특징들이 시각화되었습니다. 이 방식은 향후 LLM 안전성을 강화할 수 있는 가능한 탐지 메커니즘으로 제안됩니다.



### AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System (https://arxiv.org/abs/2510.01617)
- **What's New**: 본 연구는 AMAS(Adaptive Multi-Agent System) 아키텍처를 소개하여 기존의 여러 에이전트 시스템에서 직면한 근본적인 제약을 극복하고자 합니다. AMAS는 가벼운 LLM 적응을 통해 작업에 최적화된 그래프 구성을 자동으로 식별하는 동적 그래프 설계를 채택합니다. 이 방식은 일반적으로 사용되는 정적 구조 템플릿에 대한 의존성을 제거하고, 개별 입력의 고유한 속성을 통해 더 지능적으로 쿼리 경로를 안내합니다.

- **Technical Details**: AMAS의 핵심은 에이전트 간 상호작용을 구성하는 복합적인 그래프 아키텍처를 채택한 것입니다. 이 시스템은 에이전트들이 복잡한 문제를 집단적으로 해결할 수 있도록 지원하며, 여러 LLM 인스턴스가 각기 다른 기능적 역할을 수행합니다. Adaptive Integration, Actor-Critic Dynamics와 같은 기술적 요소들이 강화 학습(RL) 프로세스에 통합되어 그래프 선택 메커니즘이 구동됩니다.

- **Performance Highlights**: 엄격한 실험 검증 결과, AMAS는 질문 응답, 수학적 추론, 코드 생성 등의 다양한 벤치마크에서 기존의 단일 에이전트 및 다중 에이전트 접근 방식을 초월하는 성능을 발휘했습니다. 모든 평가된 시나리오에서 AMAS는 기존의 고립된 에이전트 및 다중 에이전트 기준선을 일관되게 초과하는 것으로 확인되었습니다. 이 연구는 LLM 기반 다중 에이전트 시스템의 운영적 강인성 및 도메인 간 적응력을 확립합니다.



### Efficient Training of Robust Traditional Chinese LLaMA-1B on a Single Consumer GPU: Continual Pre-training, SFT, and DPO (https://arxiv.org/abs/2510.01616)
Comments:
          17 pages, 1 figures, 2 tables. Technical report. Introduces PureTC-1B, an adapter-based pipeline for stabilizing Small Language Models in Traditional Chinese using CPT, SFT, and DPO

- **What's New**: 이번 연구에서는 전통 중국어(Traditional Chinese, TC)에서 소형 언어 모델(Small Language Models, SLMs)의 신뢰성을 개선하기 위해 PureTC-1B라는 3단계 안정화 파이프라인을 개발했습니다. 이 모델은 Meta에서 출시한 오픈 가중치 지침 조정 모델인 Llama-3.2-1B-Instruct를 기반으로 하며, 파라미터 효율적인 LoRA 어댑터를 사용합니다. 연구에서는 연속적인 사전 학습(Continual Pre-Training, CPT), 지도 세부 조정(Supervised Fine-Tuning, SFT), 그리고 직접 선호 최적화(Direct Preference Optimization, DPO)를 결합하여 TC 중심의 민감도를 높였습니다.

- **Technical Details**: PureTC-1B의 안정화 파이프라인은 전통 중국어에 적합하도록 설계되었으며, 전체 모델 재훈련 없이 단일 언어의 강건성을 개선하는 데 중점을 두었습니다. 연속 사전 학습에서는 TC 중심의 텍스트 데이터 세트를 활용하였고, 지도 세부 조정에서는 지침 데이터를 사용하였습니다. 마지막으로, 직접 선호 최적화를 통해 TC 준수 선호를 사용하여 모델의 성능을 극대화했습니다.

- **Performance Highlights**: PureTC-1B는 실제 사용을 시뮬레이션한 벤치마크에서 기본 모델 대비 비TC 출력 토큰을 51.3% 상대적으로 줄였습니다. 또한, 명명된 개체 번역(Named Entity Translation, NET) 작업에서 Llama-3B 대비 77.2%, Qwen-1.5B 대비 57.2%의 잘못된 언어 토큰을 줄이면서 TC 강건성을 입증했습니다. 이 파이프라인은 재현 가능하고 하드웨어 친화적이며, TC 및 잠재적으로 다른 비영어 언어의 안정성을 높이는 실용적인 방법을 제공합니다.



### RAG-BioQA Retrieval-Augmented Generation for Long-Form Biomedical Question Answering (https://arxiv.org/abs/2510.01612)
- **What's New**: 이번 연구에서는 RAG-BioQA라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 획득된 정보를 기반으로 한 장문의 생물 의학 답변을 생성합니다. 또한 BioBERT 임베딩과 FAISS 색인을 통합하여 효율적인 정보 검색을 달성합니다. 여러 재정렬 전략(BM25, ColBERT, MonoT5)을 비교하여 문맥 선택을 최적화합니다.

- **Technical Details**: RAG-BioQA는 세 가지 주요 구성 요소로 이루어져 있습니다: 데이터셋 준비 및 임베딩 생성을 위한 전처리 파이프라인, 재정렬 전략을 포함한 검색 모듈, 그리고 fine-tuned T5 모델을 기반으로 한 답변 생성 모듈입니다. BioBERT를 사용하여 질문-문맥 쌍의 밀집 벡터 표현을 생성하며, FAISS를 이용해 효율적인 유사도 검색을 구현합니다. 전체 처리 과정은 질문과 답변 쌍의 직접 정렬을 통해 더 신속한 정보를 제공합니다.

- **Performance Highlights**: PubMedQA 데이터셋으로 실험한 결과, RAG-BioQA의 성능이 기존 기준보다 크게 향상된 것을 확인했습니다. 우리의 최적 모델은 BLEU, ROUGE, METEOR 메트릭에서 상당한 개선을 보였습니다. 이로 인해 생물 의학 지식 검색의 접근성과 신뢰성이 향상되었습니다.



### A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation (https://arxiv.org/abs/2510.01600)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 파이프라인의 다양한 fine-tuning 전략을 비교하고 평가합니다. 연구에서 독립 fine-tuning, 공동 fine-tuning, 그리고 두 단계 fine-tuning 방법을 소개하며, 각 전략이 RAG의 성능에 미치는 영향을 분석합니다. 모든 방법이 유사한 성능 향상을 보였지만, 자원 소모 측면에서는 큰 차이가 있음을 발견했습니다.

- **Technical Details**: RAG는 두 개의 대형 언어 모델(LLM), 즉 질문에 적합한 문서를 검색하는 embedding 모델과 그 문서를 기반으로 답변을 생성하는 generator 모델로 구성됩니다. 각 모델은 fine-tuning을 통해 RAG 파이프라인의 성능을 개선할 수 있으며, 이 논문에서는 여러 loss 함수와 최신 fine-tuning 기술을 통해 embedding 및 generator 모델을 조정하는 방법을 상술합니다. 연구팀은 grid search를 통해 각 모델에 적합한 learning rate를 찾고, 각 방법의 성능을 최대화하기 위한 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, RAG 파이프라인의 performance는 네 가지 모델 조합에서 평가되었고, HotPotQA와 PopQA 데이터셋에서 모든 fine-tuning 전략이 유사한 end-to-end 성능을 보였습니다. 독립 fine-tuning은 계산 비용이 가장 적어 추천되며, context labels가 없는 경우에는 효율성을 고려하여 공동 fine-tuning을 사용할 것을 권장합니다. 데이터셋이 적절한 learning rate를 포함하고 있지 않은 경우, 두 단계 fine-tuning 방법이 최적의 선택이 될 수 있음을 발견했습니다.



### CLUE: Non-parametric Verification from Experience via Hidden-State Clustering (https://arxiv.org/abs/2510.01591)
- **What's New**: 이 논문은 로지스틱 회귀 모델처럼 접근하여 대형 언어 모델(LLM)의 내부 은닉 상태(hidden state)를 사용하여 출력 결과의 품질을 평가하는 새로운 방법론을 제안합니다. 기존의 보상 모델(reward model)이나 확률 기반 접근법의 한계를 극복하고, 보다 가치 있는 정보인 은닉 상태를 사용하자는 주장의 핵심입니다. 이를 통해 Clue(Clustering and Experience-based Verification)라는 최소한의 검증기를 개발하고, 이전 경험에 기초하여 성공과 실패 구역을 정의하여 올바른 출력을 구별하는 방법을 보여줍니다.

- **Technical Details**: Clue는 은닉 상태 차이를 통해 추론 과정을 요약하고, 과거의 성공 및 실패 사례를 기반으로 한 두 개의 중심점을 계산하여 새로운 추론 트레일을 분류합니다. 이는 LLM의 레이어에 따라 다르게 배치된 정보들을 통합적으로 사용하는 접근법입니다. 이 방법은 히든 상태가 내재된 구조적 신호를 효과적으로 활용하여 정확성을 높이는 것입니다.

- **Performance Highlights**: Empirically, Clue는 기존의 LLM 검사 방법 및 확률 기반 방법들과 비교했을 때, 특히 작은 모델이나 보정되지 않은 모델에서 우수한 성능을 보였습니다. 예를 들어, AIME 24에서 1.5B 모델을 사용할 경우, Clue는 정확도를 56.7%에서 70.0%로 향상시켰습니다. 이를 통해 Clue의 단순성이지만 강력한 성능을 입증했습니다.



### ReSSFormer: A Recursive Sparse Structured Transformer for Scalable and Long-Context Reasoning (https://arxiv.org/abs/2510.01585)
Comments:
          Accepted as a short paper at ACM Multimedia Asia 2025

- **What's New**: ReSSFormer는 리커시브 스파스 구조적 트랜스포머로, 반복적 추론과 메모리 유닛(R2MU), 적응형 스파스 어텐션 모듈(ASAM), 자가 조직화 인코더 구조(SOES)의 세 가지 혁신을 결합하여 나타난 새로운 트랜스포머 아키텍처입니다. 기존의 레이어 쌓기를 대체하고, 완전한 어텐션 대신 토큰 및 전문가 수준의 스파스 메커니즘을 사용합니다. 이 구조는 입력 내용을 기반으로 한 잠재적 토큰 토폴로지를 모델링하며, 기존의 위치 인코딩을 제거합니다.

- **Technical Details**: R2MU는 공유된 계산 블록 내에서 반복적 추론을 도입하여 다단계 사고 과정을 시뮬레이션합니다. 메모리 M(t)는 두 수준의 계층적 구성을 가지고 있으며, 토큰 수준 캐시는 최근의 표현을 저장하고, 세그먼트 수준 메모리는 압축 풀링을 통해 과거를 요약합니다. ASAM은 어텐션의 비용을 줄이기 위해 희소성을 도입하고, 전통적인 softmax 대신 희소 활성화 메커니즘을 사용하여 중요한 위치에 집중하도록 유도합니다.

- **Performance Highlights**: ReSSFormer는 언어 모델링, 다중 홉 질문 응답(Question Answering), 구조적 민감한 작업에서 강력한 기준선보다 일관되게 우수한 성능을 보입니다. FLOPs 및 매개변수 예산이 유사한 조건에서도 효율성과 유연성을 강조하며, 이 새로운 아키텍처는 구조적 일반화의 수명 주기를 개선합니다. 실험을 통해 반복적 추론, 적응형 집중, 형식에 구애받지 않는 처리 기능이 함께 작용하여 전반적인 성능을 향상시키는 것을 입증했습니다.



### One More Question is Enough, Expert Question Decomposition (EQD) Model for Domain Quantitative Reasoning (https://arxiv.org/abs/2510.01526)
Comments:
          Accepted by EMNLP 2025

- **What's New**: 이번 연구에서는 Expert Question Decomposition (EQD)이라는 새로운 접근 방식을 제안합니다. 이 방법은 도메인 지식과 계산 효율성을 조화롭게 결합하여 복잡한 질문 답변(QA) 문제를 해결하도록 설계되었습니다. EQD는 작은 데이터셋을 통해 모델을 정교하게 조정하고, 소수의 질문으로도 QA 성능을 극대화할 수 있도록 합니다.

- **Technical Details**: EQD는 두 단계의 훈련 프로세스를 통해 개발됩니다. 첫 번째 단계에서는 Llama 3.1-8B-Instruct 모델을 금융 대화 데이터를 기반으로 세분화된 질문 형식으로 학습시킵니다. 이후 프로시멀 정책 최적화(Proximal Policy Optimization) 방법을 사용하여 질문 분해 모델의 QA 프로세스와의 정렬을 최적화하며, 새로운 보상 함수가 지원 질문의 효과성을 측정합니다.

- **Performance Highlights**: EQD는 금융 도메인에서 4개의 벤치마크 데이터셋을 활용하여 테스트한 결과, 다양한 LLM에서 0.6%에서 10.5%까지 성능 개선을 기록했습니다. 이러한 결과는 기존의 도메인 조정 모델 및 고급 프롬프트 기법을 초월하며, EQD가 전문적인 도메인에서 보다 나은 QA 성능을 제공함을 입증합니다.



### A-VERT: Agnostic Verification with Embedding Ranking Targets (https://arxiv.org/abs/2510.01469)
Comments:
          19 pages, 7 figures, code available at this https URL, authors in alphabetical order

- **What's New**: 본 연구에서는 Language Model (LM) 응답을 평가하는 자동화된 방법의 필요성을 강조하고 있습니다. 기존 접근 방식들은 비용이 높거나(judge 사용) 현실과 동떨어져 있었던 반면, 제안된 구조 없는 평가 방법은 의미적 임베딩 거리(semantic embedding distance)를 활용하여 응답의 우수성을 저렴한 계산 비용으로 평가한다고 설명합니다. 이 방법을 통해 수행한 실험에서 회귀 점수(regression score) 약 ~0.97과 인간 주석자와의 정확도 약 ~96%를 달성했습니다.

- **Technical Details**: 제안된 방법은 LM이 생성한 임의의 텍스트와 대상 후보(target candidates)를 비교하기 위해 의미 임베딩(semantic embedding) 기술을 사용합니다. 상대적으로 파라미터 수가 $10B$ 미만인 임베딩 모델을 사용하여 낮은 계산 비용으로 강력한 분류를 제공합니다. 이 연구는 서로 다른 3개의 데이터 세트와 3가지 LM 아키텍처에서 테스트하여 성능을 비교하였습니다.

- **Performance Highlights**: 제안된 구조 없는 평가 방법은 기존의 자원 소모가 큰 방법과 비교하여 월등한 정확성을 나타냅니다. 특히, 회귀 점수는 ~0.97로 높았으며, 정확도는 ~96%에 달하였습니다. 이는 다양한 QA 작업의 자동 평가를 통해 우수한 성능을 얻게 된 것으로 여겨집니다.



### TAG-EQA: Text-And-Graph for Event Question Answering via Structured Prompting Strategies (https://arxiv.org/abs/2510.01391)
Comments:
          Accepted in *sem 2025

- **What's New**: 본 논문에서는 이벤트 기반 질문에 대한 처리 능력을 향상시키기 위해 TAG-EQA(Text-And-Graph for Event Question Answering)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 인과적 사건 그래프를 LLM 입력에 통합하여 구조적 관계를 자연어 문장으로 변환합니다. TAG-EQA는 다양한 프롬프트 구성을 갖추고 있으며, 이를 통해 구조적 지식이 추론을 어떻게 지원하는지를 체계적으로 분석할 수 있습니다.

- **Technical Details**: TAG-EQA는 제로샷(zero-shot), 퓨샷(few-shot), 체인 오브 쿼리(chain-of-thought) 등 세 가지 전략과 텍스트 전용(text-only), 그래프 전용(graph-only), 텍스트+그래프(text+graph) 등 세 가지 입력 모달리티를 결합하여 총 아홉 가지 프롬프트 구성을 제공합니다. 이 프레임워크는 포함된 인과적 그래프가 LLM의 이벤트 추론을 어떻게 개선할 수 있는지를 보여줍니다. TAG-EQA는 훈련 없이도 구조적 정보를 효과적으로 인코딩하는 유연한 방법으로 기능합니다.

- **Performance Highlights**: TORQUESTRA 벤치마크에서 TAG-EQA는 텍스트 전용 기준선에 비해 평균 5%의 정확도를 향상시켰습니다. 제로샷 설정에서는 최대 12%의 향상을, 그래프를 사용한 체인 오브 쿼리 프롬프트에서 18%의 향상을 보여주었습니다. 이러한 성능은 모델 및 구성에 따라 달라지지만, 인과적 그래프가 LLM의 이벤트 추론을 어떻게 강화할 수 있는지에 대한 통찰을 제공합니다.



### HiSpec: Hierarchical Speculative Decoding for LLMs (https://arxiv.org/abs/2510.01336)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 추론 속도를 높이기 위한 새로운 기술로 계층적 투기적 디코딩(Hierarchical Speculative Decoding, HiSpec)을 제안합니다. HiSpec은 중간 검증(intermediate verification) 단계에서 계산 오버헤드를 최소화하기 위해 조기 종료(early-exit) 모델을 사용합니다. 이 같은 접근 방식은 검증 시간을 줄이고, 메모리 사용량을 줄이며, 정확성을 유지할 수 있게 합니다.

- **Technical Details**: HiSpec은 기본적으로 두 개의 모델을 사용합니다: 작고 빠르지만 덜 정확한 드래프트 모델과 더 크고 정확한 타겟 모델입니다. 드래프트 모델이 생성한 토큰은 중간 검증 단계를 통해 초기 논리 검사를 진행하여 잘못된 토큰을 조기에 제외하고, 이후 타겟 모델에서 최종 검증이 수행됩니다. 이를 통해 HiSpec은 레이턴시(latency)를 개선하고, 드래프트, 중간 검증자, 타겟 모델 간의 키-값 캐시(Key-Value caches)와 숨겨진 상태(hidden states)를 재사용하여 자원 효율성을 극대화합니다.

- **Performance Highlights**: HiSpec을 사용한 평가 결과, 평균적으로 throughput이 1.28배 향상되었으며, 최대 2.01배의 성능 개선을 보여주었습니다. 이는 단일 계층 투기적 디코딩과 비교할 때 이루어진 성과로서, accuracy는 타겟 모델의 출력과 일관성을 유지하였습니다. 또한, HiSpec은 사전 훈련된 모델과 후 훈련된 수정 모델 모두에 적용 가능하여, 그 적용 범위가 넓음을 나타냅니다.



### Evaluation Sheet for Deep Research: A Use Case for Academic Survey Writing (https://arxiv.org/abs/2510.01283)
- **What's New**: 이번 논문에서는 인간 개입 없이도 지식 집약적 작업을 수행할 수 있는 Argentic 기능을 갖춘 대형 언어 모델(LLM)과 그 도구인 Deep Research의 평가 기준을 제안합니다. Deep Research는 웹을 탐색하고 정보를 추출하는 기술을 활용하여 다수의 페이지에 걸친 보고서를 생성할 수 있는 능력을 갖추고 있습니다. 우리는 Deep Research 도구의 성능 평가를 위한 평가 시트를 도입하고, 이 도구의 유용성을 학술 조사 작성을 통한 실제 사례로 검토했습니다.

- **Technical Details**: Deep Research 도구는 사용자가 복잡한 주제에 대해 포괄적이고 심층적인 보고서를 작성할 수 있도록 설계되었습니다. 이들은 정보 검색, 여러 소스 compilation, 그리고 독립적으로 작업을 수행하는 데 필요한 고급 추론 및 분석 기능을 포함합니다. 이러한 도구들은 전통적인 검색 엔진과는 달리 반복적 검색 과정을 통해 정보를 분석하고 의미 있는 보고서를 생성하는 데 중점을 둡니다.

- **Performance Highlights**: 저자들은 OpenAI의 Deep Search와 Google의 Deep Search를 평가하여 이러한 도구 사이의 성능 차이를 보여주었습니다. 특히, 학술 조사 작성 시 이들 도구가 가진 한계와 강점을 분석하면서, 신뢰할 수 있는 정보 출처의 중요성과 함께 AI 결과물의 평가는 필수적임을 강조했습니다. 평가 시트는 AI 생성 콘텐츠의 정확성과 신뢰성을 확보하기 위한 기준을 제공합니다.



### TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixtur (https://arxiv.org/abs/2510.01279)
Comments:
          27 pages, 13 figures

- **What's New**: 본 논문에서는 Tool-Use Mixture (TUMIX)라는 새로운 앙상블 프레임워크를 제안합니다. 이 프레임워크는 여러 에이전트를 병렬로 실행하며, 각 에이전트는 독특한 도구 사용 전략과 답변 경로를 갖고 있습니다. 실험 결과, TUMIX는 기존의 도구 보강 및 테스트 시간 확장 방법들에 비해 평균 3.55%의 정확도 향상을 보여주었으며, 이는 주목할 만한 성과입니다.

- **Technical Details**: TUMIX는 코드를 활용한 추론과 검색 능력을 LLM에 통합하여 다양한 질문에 대한 최적의 접근 방식을 찾는 데 중점을 둡니다. 이 프레임워크는 반복적으로 에이전트 간의 답변을 공유하고 다듬는 과정을 통해 다양한 추론 경로를 탐색합니다. TUMIX는 에이전트의 다양성과 품질이 중요하다는 것을 강조하며, 이로 인해 LLM이 스스로 에이전트를 최적화하는 방법을 찾을 수 있음을 보여줍니다.

- **Performance Highlights**: TUMIX는 다양한 벤치마크에서 기존 모델 대비 평균 7.8% 및 17.4%의 정확도 향상을 달성하였으며, 이는 Gemini-2.5-Pro 및 Gemini-2.5-Flash 모델의 수치입니다. 추가적으로, TUMIX는 테스트 시간 확장에서 두 가지 단계(다양한 후보 솔루션 생성 및 올바른 솔루션 선택)를 통해 높은 정확도와 커버리지를 보장하며, 이러한 구조가 LLM의 성능 향상에 기여한다고 강조합니다.



### LLM Based Sentiment Classification From Bangladesh E-Commerce Reviews (https://arxiv.org/abs/2510.01276)
- **What's New**: 이번 연구에서는 방글라데시 전자상거래 리뷰의 감정 분석에 transformer 기반의 BERT 모델과 최신 대형 언어 모델(LLMs)을 활용하는 가능성을 탐구합니다. 특히 Llama-3.1-8B 모델의 성능을 분석하며, 감정 분석의 정확성을 높이는 데 기여하고 있습니다.

- **Technical Details**: 연구는 원본 데이터셋에서 추출한 4000개의 방글라어 및 영어 고객 리뷰 샘플을 사용하여 모델을 미세 조정(fine-tuning)하였습니다. Llama-3.1-8B 모델은 일반적으로 사용되는 다른 모델들인 Phi-3.5-mini-instruct, Mistral-7B-v0.1, DistilBERT-multilingual 등과 비교하여 우수한 성능을 보였습니다.

- **Performance Highlights**: Llama-3.1-8B 모델은 전체 정확도(accuracy) 95.5%, 정밀도(precision) 93%, 재현율(recall) 88%, F1 점수 90%를 기록하며, 다른 미세 조정된 모델에 비해 뛰어난 성능을 입증했습니다. 이 연구는 LoRA 및 PEFT와 같은 파라미터 효율적 미세 조정 방법이 자원 배급이 제한된 환경에서의 계산적 오버헤드를 줄일 수 있음을 강조합니다.



### TraceDet: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models (https://arxiv.org/abs/2510.01274)
- **What's New**: 최근 확산 대형 언어 모델(DF-LLMs)이 자동 회귀 LLMs(AR-LLMs)의 유망한 대안으로 떠오르고 있습니다. 그러나 D-LLMs의 환각 문제(hallucination problem)는 아직 충분히 탐구되지 않아 실제 응용에서의 신뢰성에 한계를 두고 있습니다. 본 연구에서는 D-LLMs의 중간 디노이징 단계(denoising steps)를 이용하여 환각 탐지를 위한 새로운 프레임워크인 TraceDet(Trace Detection)을 제안합니다.

- **Technical Details**: TraceDet는 D-LLMs의 디노이징 과정을 행동 추적(action trace)으로 모델링합니다. 각 행동은 이전 중간 출력에 조건화된 채 모델이 깨끗한 응답에 대한 예측을 나타내며, 환각된 응답에 최대한 유익한 하위 추적(sub-trace)을 식별합니다. 이는 정보 병목(information bottleneck) 원칙을 적용하여 자동으로 가장 유익한 하위 추적을 추출하고, 명시적인 단계 수준의 감독(supervision)을 필요로 하지 않습니다.

- **Performance Highlights**: TraceDet는 두 개의 오픈 소스 D-LLM인 LLaDA-8B-Instruct와 Dream-7B-Instruct에서 평가되었으며, 다양한 QA 데이터 세트에서 일관되게 환각 탐지 정확도를 15.2% 향상시키는 결과를 보였습니다. 이 연구는 D-LLMs의 환각 행동에 대한 초기 연구로, AR-LLMs와의 차별적인 다단계 패턴을 드러내며, 제안된 방법의 강건성도 입증하였습니다.



### Think Twice, Generate Once: Safeguarding by Progressive Self-Reflection (https://arxiv.org/abs/2510.01270)
Comments:
          Accepted to EMNLP 2025 Findings

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 출력 결과를 동적으로 자기 모니터링하고 수정할 수 있는 새로운 접근법인 Progressive Self-Reflection(PSR)을 제안합니다. PSR은 공격 성공률을 현저히 감소시키면서도 원래의 성능을 유지하는 방식으로, 사전 훈련 없이도 안전성을 강화할 수 있는 방법입니다. 이는 LLM이 생성 과정 중 정기적인 자기 평가를 통해 해로운 결과를 피할 수 있게 해줍니다.

- **Technical Details**: PSR은 LLM의 생성 과정에서 내부 자기 평가 루프를 통합하여 동작합니다. 모델이 응답을 생성할 때마다 특정 토큰 수(KK)마다 생성 중단 후 안전성 검토를 진행하고, 이를 통해 해로운 내용 발생 가능성을 스스로 점검합니다. 이를 위한 경량화된 자기 반성 예측기를 도입하여 입력 복잡도에 따라 최적의 반성 회수 수를 예측합니다.

- **Performance Highlights**: 실험 결과, PSR을 Llama-3.1-8B-Instruct에 적용했을 때 공격 성공률이 77.5%에서 5.9%로 감소하고, Llama-3.1-8B 기초 모델에서는 89.7%에서 5.6%로, Qwen2.5-7B-Instruct에 대해서도 44.4%에서 3.8%로 감소했습니다. 이러한 성과는 대형 언어 모델의 안전성을 향상시키는 실질적인 방법을 나타내며, PSR은 입력의 위험 프로파일에 비례하여 계산 자원을 동적으로 할당하는 기능을 제공합니다.



### AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees (https://arxiv.org/abs/2510.01268)
Comments:
          Accepted by NeurIPS2025

- **What's New**: 이번 연구는 인간이 작성한 텍스트와 대형 언어 모델(LLM) 소속 텍스트를 구별하는 문제에 대한 새로운 접근법을 제안합니다. 기존의 logits 기반 감지 방법들은 로그 확률만으로 텍스트를 평가하는 데 한계가 있음을 지적하며, AdaDetectGPT라는 새로운 적응형 분류기를 도입합니다. 이 분류기는 외부 훈련 데이터를 활용하여 기존 방법들의 성능을 향상시키고자 합니다.

- **Technical Details**: AdaDetectGPT는 로그 기반 검출기의 진정한 음성 비율(true negative rate, TNR)을 향상시키기 위해 최적화된 witness function을 학습하는 방식으로 작동합니다. 이 과정은 선형 방정식 시스템을 해결하는 방식으로 간단히 진행됩니다. 본 기법은 여러 데이터셋과 다양한 LLM에서 기존 방법들보다 높은 성능을 보이며, 통계적 성능 보장을 통해 평균 진정한 음성 비율, 거짓 양성 비율, 진정한 양성 비율 및 거짓 음성 비율에 대한 유한 샘플 오차 한계를 제시합니다.

- **Performance Highlights**: AdaDetectGPT는 다양한 데이터셋 및 LLM 조합에서 기존의 최첨단 방법들에 비해 최대 58%까지 성능을 향상시키는 결과를 보입니다. 화이트 박스 설정에서는 AUC 면적이 12.5%에서 37% 향상되었으며, 블랙 박스 설정에서도 유의미한 성과를 나타냈습니다. 이러한 차별화된 성능 개선은 AdaDetectGPT의 효과성을 더욱 입증합니다.



### OpenAI's GPT-OSS-20B Model and Safety Alignment Issues in a Low-Resource Languag (https://arxiv.org/abs/2510.01266)
Comments:
          6 pages, 4 tables

- **What's New**: 최근 OpenAI의 GPT-OSS-20b 모델에 대한 안전성 평가의 일환으로, 귀하의 연구는 언어 리소스가 적은 환경에서 성능과 안전 정렬(allignment)에서 드러난 여러 취약점들을 요약합니다. 이 연구는 대변되지 않는 커뮤니티의 사용자들에게 모델의 신뢰성을 의문시하는 데 중점을 두었습니다. 하우사어(Hausa)를 사용하여, 모델의 행동에서 편견(bias), 부정확성(inaccuracy), 문화적 무감각(cultural insensitivity) 등을 발견하였습니다.

- **Technical Details**: 이 연구에서 사용된 방법론은 시스템적 적대적 프롬프트를 사용하여 GPT-OSS-20b 모델의 취약점을 탐구하는 것입니다. 대화형 웹 인터페이스를 통해 기본의 사전 훈련된 상태에서 모델을 사용하여 단계별로 균형을 맞추는 무난한 질문에서 시작해 적대적 요소를 점진적으로 도입했습니다. 이 접근 방식은 체인 오브 사고(chain-of-thought, CoT) 프롬프트를 활용하여 모델의 추론 과정을 유도하며, 얼핏 보기엔 신뢰할 수 있는 출력을 생성하는 대신, 위험한 및 부정확한 출력을 통해 모델의 안전성 문제를 드러냈습니다.

- **Performance Highlights**: 모델이 하우사어로 부정확하고 유해한 정보를 생성하는 경향이 있으며, 특히 쉬운 감정 표현을 사용했을 때 안전 프로토콜이 완화된 것을 확인했습니다. 우리의 설문조사 결과, 참가자의 98%가 모델이 추천한 독성 물질이 안전하다고 잘못 인식할 수 있음을 보여주었습니다. 이 모델은 기본적인 상식 사실을 구별하지 못하여 교육적 또는 정보 제공을 위한 신뢰성이 결여된 결과를 나타내었습니다.



### In AI Sweet Harmony: Sociopragmatic Guardrail Bypasses and Evaluation-Awareness in OpenAI gpt-oss-20b (https://arxiv.org/abs/2510.01259)
Comments:
          27 pages, 1 figure

- **What's New**: 이 연구에서는 OpenAI의 gpt-oss-20b 모델을 사용하여 사회-프락티컬(Sociopragmatic) 프레이밍, 언어 선택, 그리고 지침 계층이 거부 행동에 미치는 영향을 조사합니다. 다양한 해악 도메인에서 복합 프롬프트가 기본 선율 대비 큰 차이를 보이면서 거부를 감소시키는 것을 발견하였습니다. 또한, 여러 언어의 정중함 차이가 거부 기준에 유의미하게 영향을 미치는 것도 보여주고 있습니다.

- **Technical Details**: 이 연구는 종합적인 다국어 사회-프락티컬 프롬프트의 거부 행동에 대한 체계적인 정량화를 시도합니다. 특히, 안전 등을 주제로한 요청의 계층적 구조와 기능을 테스트하고, 다양한 언어와 작업 간의 등록 효과를 정량화합니다. 방어력 강화를 위한 AI 지원 기법을 소개하며, 이를 통해 모델 출력에서의 정보 유출을 줄일 수 있음을 증명했습니다.

- **Performance Highlights**: 연구 결과, OpenAI Moderation API가 실질적으로 유용한 출력을 적게 포착하는 것으로 나타났습니다. 또한, 동일한 시드에서 다르게 구성된 시스템은 5%에서 10%까지의 거부율 차이를 보이는 등 재현성 문제에 대해 경종을 울리고 있습니다. 독일어 및 프랑스어의 정중한 구문이 영어보다 더 많이 유출되는 경향이 발견되어, 다양한 언어적 맥락의 중요성이 강조됩니다.



### Measuring Algorithmic Partisanship via Zero-Shot Classification and Its Implications on Political Discours (https://arxiv.org/abs/2510.01258)
Comments:
          19 pages, 7 figures

- **What's New**: 이 논문은 생성 인공지능(Generative Artificial Intelligence, GAI)의 급속한 확산 속에서 정치적 담론에서 지능형 시스템이 점차 지배적으로 나타고 있음을 강조합니다. 그러나 훈련 데이터의 왜곡, 인간의 편견, 알고리즘적 결함으로 인해 내부화된 정치적 편향이 여전히 문제로 남아있습니다.

- **Technical Details**: 저자는 제로샷 분류(zero-shot classification) 접근 방식을 활용하여 이념적 조화(ideological alignment), 주제 적합성(topicality), 반응 감정(sentiment), 객관성(objectivity) 등을 결합하여 알고리즘의 정치적 당파성을 평가합니다. 1800개의 모델 반응을 여섯 개의 주요 대규모 언어 모델(LLMs)에 대해 개별적으로 입력하고, 각기 다른 네 가지 세부 조정된 분류 알고리즘(classification algorithms)에 대해 평가 지표를 계산하였습니다.

- **Performance Highlights**: 실험 결과, 모든 LLM에서 확대된 자유-권위주의적 정렬(liberal-authoritarian alignment)이 관찰되었고, 주목할 만한 합리적 대답의 초월(reasoning supersessions)과 이미 정해진 거절(canned refusals) 사례가 나타났습니다. 이는 인간-컴퓨터 상호작용에서의 심리적 영향과 본질적 편향이 공적 담론에 어떻게 스며들 수 있는지를 조명합니다.



### RJE: A Retrieval-Judgment-Exploration Framework for Efficient Knowledge Graph Question Answering with LLMs (https://arxiv.org/abs/2510.01257)
Comments:
          18 pages, 9 figures

- **What's New**: 본 논문에서는 Knowledge Graph Question Answering (KGQA)의 한계를 극복하기 위해 Retrieval-Judgment-Exploration (RJE) 프레임워크를 제안합니다. 기존 연구가 대규모 언어 모델 (LLMs)에 의존하고 있는 반면, RJE는 스몰 사이즈 LLM들이 효과적으로 작동할 수 있는 보조 모듈을 도입하여 성능을 향상시킵니다. 이 프레임워크는 정교한 추론 경로를 검색하고, 그 sufficiency를 평가하며, 추가 증거를 탐색할 수 있는 조건부 접근을 적용합니다.

- **Technical Details**: RJE 프레임워크는 세 가지 단계로 구성되어 있습니다: Retrieval, Judgment, Exploration. 첫 번째 단계에서는 KG에서 관련성 높은 추론 경로를 검색하고, 두 번째 단계에서는 LLM이 정보를 바탕으로 충분성을 평가합니다. 마지막으로 Exploration 단계에서는 질문을 단순하게 분해하고, 기존의 경로를 활용해 부족한 증거를 채워나갑니다.

- **Performance Highlights**: RJE는 기존의 방법론에 비해 정확성과 효율성에서 우수한 성과를 보였습니다. 특히, 소규모 오픈 소스 LLM들 (3B 및 8B 파라미터)을 사용할 경우, RJE는 CWQ 데이터셋에서 이전의 최고 성능 방법보다 각각 41.5% 및 27.9% 향상된 결과를 나타냈습니다. 또한, 에이전트 기반 방법에 비해 LLM 호출 횟수와 토큰 사용량을 크게 줄여, 효율성 개선을 이루었습니다.



### Longitudinal Monitoring of LLM Content Moderation of Social Issues (https://arxiv.org/abs/2510.01255)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 내용 조정 정책과 관행이 어떻게 모델의 출력에 영향을 미치는지를 분석합니다. 특히, AI Watchman이라는 누적 감시 시스템을 도입하여 모델의 거부율을 공개적으로 측정하고 추적하는 방법을 제안합니다. 400개 이상의 사회적 이슈를 포함하는 데이터셋을 분석하여 OpenAI의 GPT-4.1, GPT-5 및 DeepSeek의 모더레이션 엔드포인트에 대한 검토를 실시했습니다.

- **Technical Details**: AI Watchman은 사회적 이슈에 대한 LLM 콘텐츠 조정의 장기적인 모니터링을 제공하는 시스템으로, 421개의 사회적 이슈를 주제로 한 데이터셋을 사용하여 GPT-4.1, GPT-5 및 DeepSeek 모델의 응답을 평가합니다. 본 시스템은 두 주기마다 자동으로 쿼리를 실행해 그 결과를 웹사이트에 공개하며, 각 범주별로 시간에 따른 플래깅 비율을 시각화합니다. 이를 통해 기존의 비공식 발표가 없는 정책 변경도 발견할 수 있었습니다.

- **Performance Highlights**: AI Watchman의 결과는 모델별로 거부율이 1.2%에서 3.9% 사이로 다양하다는 것을 보여주었습니다. GPT-5는 거의 모든 주제에서 GPT-4.1에 비해 낮은 플래깅 비율을 기록하여 OpenAI의 정책에 부합하는 결과를 나타냈습니다. 또한, 최근의 여러 변화(예: 이스라엘-가자 갈등에 연관된 내용의 증가하는 거부율)를 통해 정책 변화의 영향을 감지할 수 있는 가능성을 보여주었습니다.



### Do Bias Benchmarks Generalise? Evidence from Voice-based Evaluation of Gender Bias in SpeechLLMs (https://arxiv.org/abs/2510.01254)
Comments:
          5 pages, 2 Figures, Submitted to IEEE ICASSP 2026

- **What's New**: 이 논문에서는 SpeechLLMs(음성 대형 언어 모델)에서의 편향(bias) 및 공정성(fairness) 평가를 위한 MCQA(다중 선택 질문 응답) 형식의 한계를 탐구합니다. 기존 연구들은 MCQA 성능이 다른 과제와 일반화될 수 있다고 가정하였으나, 본 연구에서 실제로는 이러한 일반화가 신뢰할 수 없음을 입증하였습니다. 우리는 LoRA(저순위 적응기)를 통해 세 개의 SpeechLLMs를 미세 조정(fine-tuning)하여 특정 MCQA 행동을 유도하고 평가를 진행했습니다.

- **Technical Details**: 우리는 MCQA 벤치마크에서의 편향 행동의 전이 가능성을 평가하기 위해 두 가지 주요 축에서 접근하였습니다: 1) 서로 다른 MCQA 벤치마크 간의 일반화, 2) MCQA에서 학습한 편향이 장기적인 과업으로 전이되는지 여부. 이를 위해 Qwen2-Audio-7B-Instruct, LTU-AS, LLaMA-Omni의 세 가지 SpeechLLMs를 선정하고, 각 모델에 대한 미세 조정을 통해 특정 행동을 유도하였습니다.

- **Performance Highlights**: 연구 결과, MCQA 편향 벤치마크에서의 성능은 다른 MCQA 벤치마크 및 장기적 작업에서의 성능을 예측하는 데 신뢰할 수 없음을 보여줍니다. 본 논문은 성별 편향을 평가하기 위한 새로운 장기 평가용 스위트를 공개하고, SpeechLLMs의 행동의 전이 가능성을 측정하는 데 중요한 기초 자료를 제공하였습니다. 이러한 결과는 음성 도메인에서의 편향 문제 해결을 위한 새로운 접근 방식으로서 가치가 있습니다.



### GPT and Prejudice: A Sparse Approach to Understanding Learned Representations in Large Language Models (https://arxiv.org/abs/2510.01252)
Comments:
          Preprint. Draft version, subject to revision. 8 pages, 3 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)과 희소 오토인코더(SAEs)의 조합을 통해 모델 동작뿐만 아니라 훈련 데이터에 내재된 구조, 주제 및 편견을 해석할 수 있는 가능성을 보여줍니다. 제인 오스틴의 소설을 기반으로 훈련된 GPT 스타일의 변환 모델을 통해 사회적 구조와 내러티브를 반영하는 해석 가능한 특징을 발견했습니다. 이러한 접근법은 편향 발견 및 모델 해석의 새로운 길을 제시하며, 대규모 데이터셋의 탐색을 위한 확장 가능한 방법론을 제공합니다.

- **Technical Details**: 연구는 제인 오스틴의 주요 작품들로 구성된 정제된 데이터셋을 기반으로 사용자 정의 GPT 스타일 변환 모델을 훈련했습니다. 이후, 모델의 두 개 주요 변환층에서 hidden states를 추출하고 이를 희소 오토인코더에 통과시켜 내부 표현을 조사했습니다. 이 방법론은 모델의 내부 구조에서 사회적 아이디어가 어떻게 인코딩되는지를 탐구하고, 각 층에서 해석 가능한 주제를 분석하는 기반을 제공합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 모델의 활성화에서 구조화된 해석 가능한 특징을 회복할 수 있음을 증명했습니다. 연구는 적은 수의 훈련 데이터라는 한계에도 불구하고, 사회적 주제와 관련된 중요한 패턴을 식별하는 데 성공했습니다. 이 접근법은 대규모 LLM에서 숨겨진 구조와 편향을 발견하는 데 있어 효율적임을 보이며, 인간 감사가 불가능한 환경에서도 적용할 수 있는 유연성을 제공합니다.



### Efficient Uncertainty Estimation for LLM-based Entity Linking in Tabular Data (https://arxiv.org/abs/2510.01251)
- **What's New**: 이 논문은 Knowledge Base의 엔티티와 텍스트 값을 연결하는 Entity Linking (EL) 작업에서의 불확실성 추정 문제를 다룹니다. 기존 방식들이 여러 번의 추론(수 생성)을 필요로 하는 반면, 저자들은 단일 샷(single-shot) LLM 출력을 기반으로 불확실성을 추정하는 자가 지도(self-supervised) 접근법을 제안합니다. 이는 다중 생성의 필요성을 줄이며 계산 비용을 대폭 낮추는 효과적인 방법을 제공합니다.

- **Technical Details**: 연구는 LLM의 결과물에서 토큰 관련 특성을 이용하여 불확실성을 추정하는 경량 회귀 모델(lightweight regression model)를 학습하는 자가 지도 방법을 제안합니다. 이를 통해, 다수의 생성 결과를 토대로 실제 관찰된 불확실성을 근사하는 능력을 배양합니다. 이 방식은 특히 대규모 EL 시나리오에서 자원 소모를 줄이면서 효율성을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 여러 LLM들을 대상으로 한 EL 작업에서 평가되었으며, 불확실성 추정이 저정확성 출력을 효과적으로 감지할 수 있음을 보여줍니다. 이 결과는 인적 검토가 필요한 경우에도 품질 개선을 위한 우선 순위를 매기는 데 도움을 줍니다. 저자들은 이 방법이 계산적 오버헤드를 최소화하면서 EL 워크플로우에 실용적이고 비용 효율적인 불확실성 측정을 통합할 수 있다고 주장합니다.



### GemDetox at TextDetox CLEF 2025: Enhancing a Massively Multilingual Model for Text Detoxification on Low-resource Languages (https://arxiv.org/abs/2510.01250)
- **What's New**: 최근 소셜 미디어 플랫폼의 급격한 발전은 이를 감독하기 위한 규제가 뒤처진 가운데, 자동화된 내용 정화(automated detoxification)가 커다란 도움이 될 수 있다는 점을 강조하고 있습니다. 본 연구는 PAN 2025 다국어 텍스트 정화 도전 과제에 제출한 내용을 바탕으로, 독성이 있는 단일 문장을 15개 언어로 중립적인 의미로 재구성하는 방법을 제시합니다. 특히, 이 연구에서는 12B 매개변수를 갖춘 Gemma-3 다국어 변환기 모델을 기반으로 하여, LoRA SFT 미세 조정(parameter-efficient LoRA SFT fine-tuning) 및 Chain-of-Thought와 같은 프롬프트 기법을 적용했습니다.

- **Technical Details**: 우리는 3,600개의 인간이 작성한 평행 쌍 및 21,600개의 기계 번역된 합성 쌍을 포함한 다국어 학습 코퍼스를 사용하여 모델을 훈련했습니다. 추론 단계에서는 LaBSE를 통해 인접한 세 개의 이웃을 추가하고, 독성 구간에 대한 명시적 주석을 더했습니다. 시스템의 성능은 Style Transfer Accuracy, LaBSE 기반 의미 보존(semantic preservation) 및 xCOMET 유창성(fluidity)을 통해 평가되었고, 모든 언어에서 상위 성능을 기록했습니다.

- **Performance Highlights**: 최고 성능 결과에서, 몇 가지 샘플 예제(few-shot examples)와 기본 Chain-of-Thought 프롬프트를 사용하여 각각 +0.081 및 +0.088의 점수 상승을 보여주었습니다. ANOVA 분석 결과, 언어 자원 상태가 성능의 가장 강력한 예측 변수임이 밝혀졌으며, ($\eta^2$ = 0.667, p < 0.01)이라는 수치를 나타냈습니다. 데이터 증대(data augmentation)가 특히 저자원 언어(low-resource languages)의 성능 향상에 큰 효과를 미쳤다는 점도 주목할 만합니다.



### LOCA: Logical Chain Augmentation for Scientific Corpus Cleaning (https://arxiv.org/abs/2510.01249)
Comments:
          29 pages, 2 figures

- **What's New**: 본 연구에서는 LOCA(Logical Chain Augmentation)라는 새로운 프레임워크를 소개합니다. 이는 과학적인 질문-답변(QA) 데이터셋의 오류율을 획기적으로 줄이는 방법으로, 자동으로 로지컬 체인을 강화하여 과학적 원리와 그 도출 과정을 명확히 분리하는 방식으로 작동합니다. LOCA는 노이즈가 많고 구조적 결함을 가진 데이터를 효과적으로 정제하여 결과적으로 오류율을 20%에서 2% 미만으로 감소시킵니다.

- **Technical Details**: LOCA 프레임워크는 다음 두 가지 주요 작업을 통해 수행됩니다: 체인 완성과 구조적 분해입니다. 원래의 답변은 임의의 단계로 표현되며, LOCA는 이 단계들을 보다 기초적인 하위 단계로 분해하여 명확성을 높입니다. 즉, 각 논리적 단계를 세부적으로 쪼개어 그 과정의 명확한 증거를 제공하고, 이를 통해 기존의 비효율적인 검토 과정을 개선합니다.

- **Performance Highlights**: LOCA는 도전적인 물리학 QA 데이터셋에 대한 성능 테스트를 통해 확인된 바와 같이, 정제된 데이터셋의 잔여 오류율을 2% 미만으로 현저히 감소시켰습니다. 또한, LOCA는 다양한 기준선을 초과하는 성과를 기록하며, 고품질 데이터셋 수집과 과학 AI의 훈련 및 평가에 효과적으로 기여할 수 있음을 보여주었습니다.



### SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs (https://arxiv.org/abs/2510.01248)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이 논문에서는 그래프 학습에서 크로스 도메인 일반화(Cross-domain generalization)의 한계를 극복하기 위해 텍스트 속성을 지닌 그래프(Text Attributed Graphs, TAGs)를 위한 새로운 구조 인식(self-supervised) 방법, SSTAG(Structure-aware Self-supervised learning method for Text-Attributed Graphs)를 제안합니다. 기존 그래프 모델의 한계인 단일 그래프 데이터셋에 대한 의존성을 극복하고, 대규모 사전 훈련된 모델을 활용하여 구조적 모델링의 가능성을 높이는 방안을 모색하고 있습니다. 또한, SSTAG는 메모리 기반 메커니즘을 도입하여 일반적인 그래프 표현을 저장하고, 이를 메모리 앵커와 통합하여 변하지 않는 지식을 결합합니다.

- **Technical Details**: SSTAG는 그래프의 노드(Node) 및 엣지(Edge) 수준의 태스크에 대해 관련 정보를 정맥하고, 이는 Personalized PageRank(PPR) 알고리즘을 통해 서브그래프를 샘플링하여 그래프 구조의 차이를 해소합니다. 이 방식은 구조 인식 멀티레이어 퍼셉트론(Multilayer Perceptrons, MLP)으로 대형 언어 모델(Large Language Models, LLMs)과 그래프 신경망(Graph Neural Networks, GNNs)에서 보완적인 지식을 공유하는 이중 지식 증류 프레임워크를 통해 실현됩니다. 이를 통해 MLP는 GNN의 구조 모델링 능력과 LLM의 의미 추론 능력을 혼합하여, 자가 감독 학습(self-supervised learning)을 위한 새로운 목표를 설정합니다.

- **Performance Highlights**: 광범위한 실험을 통해 SSTAG는 크로스 도메인 전이 학습 작업에서 기존의 최신 모델들을 초월하는 성능을 보여줍니다. 또한, 대규모 그래프에서 뛰어난 확장성을 제공하며, 기존 GNN 및 LLM 기반 방법들에 비해 추론 비용을 현저히 낮추면서 경쟁력 있는 성능을 유지하는 것으로 확인되었습니다. 이러한 결과는 SSTAG의 구조적 목표와 다중 모드(distillation)의 효용성을 강하게 입증합니다.



### Let's Play Across Cultures: A Large Multilingual, Multicultural Benchmark for Assessing Language Models' Understanding of Sports (https://arxiv.org/abs/2510.01247)
Comments:
          52 pages, 56 figures; appearing at EMNLP'25

- **What's New**: 이 논문에서는 전통 스포츠를 이해하고 평가하기 위해 	extbf{	extit{CultSportQA}}라는 새로운 벤치마크를 소개합니다. 이 데이터셋은 60개 국가와 6개 대륙의 전통 스포츠를 포함하며, 문화적 범주를 4가지로 나누어 33,000개의 여러 선택 질문을 제공합니다. 이 데이터셋은 텍스트 및 이미지 모달리티를 포함하고 있으며, 역사 기반, 규칙 기반 및 시나리오 기반 질문으로 구성되어 있습니다.

- **Technical Details**: CultSportQA는 다양한 언어 모델(예: Large Language Models, Small Language Models, Multimodal Language Models)에 대해 제로샷, 퓨샷 및 체인 오브 싱크(Chain-of-thought) 프롬프트를 이용하여 모델 성능을 평가합니다. 각 질문은 역사, 규칙 및 시나리오의 세 가지 주요 유형으로 분류되어, AI 모델이 텍스트와 비주얼 입력을 바탕으로 사고할 수 있는 능력을 도전합니다. 또한 이 데이터셋은 11개 언어로 된 스포츠 관련 질문을 포함하여 다문화, 다언어적 기준을 제공합니다.

- **Performance Highlights**: 연구에서는 8개의 최첨단 LLM과 5개의 SLM, 4개의 MLLM을 평가하였으며, 전통 스포츠 관련 쿼리에 대한 이들의 추론 능력에서 주요 차이점을 발견했습니다. 다양한 국가와 언어에서의 성능 경향을 분석하여 AI의 문화적 배경 인식을 강화할 수 있는 통찰력을 제공합니다. 이 연구는 AI의 포괄성과 공정성을 촉진하고, 전통 스포츠와 NLP의 접목을 발전시키는 데 기여합니다.



### A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering (https://arxiv.org/abs/2510.01246)
Comments:
          25 pages

- **What's New**: 최근 Sparse Autoencoders (SAEs)는 언어 모델 조정에 있어 강력한 도구로 떠오르고 있습니다. 기존의 top-k SAE latent를 이용한 방법에서 비언어적 특징들이 포함되는 문제를 발견하고, 이를 해결하기 위해 단일 SAE latent (top-1)에 집중하는 새로운 접근 방식을 제안합니다. 또한, 일정한 SAE 조정의 한계를 지적하고, token-wise decaying steering 전략을 통해 의도적인 생성을 개선합니다.

- **Technical Details**: 이 연구에서는 상위 활성화 feature 집합을 빼는 방식으로 SAE feature 선택 메커니즘을 제안하여 행동 변화를 주도하는 차원을 고립합니다. 또한, 생성 과정 동안 개입 강도를 동적으로 감소시키는 token-wise decaying steering 전략을 설계하여 안정성을 개선하고 과도한 조정을 완화합니다. 이러한 방법들은 Gemma-2-2b 및 Gemma-2-9b 데이터셋에서 수학적 추론과 지시 사항 이행을 평가하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과, decaying SAE steering이 안정성을 높이고, MeanActDiff 및 일정한 SAE 조정 방식보다 더 세밀한 제어를 달성함을 보여주었습니다. 수학적 추론에서는 SAE 조정이 반복적인 프롬프트 없이도 단계별 해결책을 안정적으로 도출하며, MeanActDiff를 초월하는 성능을 기록했습니다. 또한, SAE 조정은 지침 토큰을 추가하는 효과와 유사하게 동작함을 관찰했습니다.



### SeMob: Semantic Synthesis for Dynamic Urban Mobility Prediction (https://arxiv.org/abs/2510.01245)
Comments:
          EMNLP2025

- **What's New**: 이번 논문에서는 SeMob라는 새로운 파이프라인을 제안하여 인간의 이동 예측을 동적으로 개선하는 방법을 설명합니다. 기존의 spatiotemporal 모델들이 외부 이벤트로 인한 급작스러운 변화에 적응하는 데 어려움을 겪는 반면, SeMob는 LLM(대형 언어 모델)을 기반으로 하여 텍스트 정보를 효과적으로 활용합니다. 이는 이벤트 관련 정보를 추출하고 이를 spatiotemporal 데이터와 결합하여 예측의 정확도를 높이는 혁신적인 접근 방식입니다.

- **Technical Details**: SeMob는 다중 에이전트 프레임워크를 사용하 여 온라인 데이터 소스로부터 사건에 대한 텍스트를 자동으로 추출하고 해석합니다. 이 시스템은 텍스트와 spatiotemporal 데이터를 결합하는 프로그레시브 퓨전 기법을 통해 성능을 향상시킵니다. 기존의 텍스트와 시간 데이터 융합 모델이 공간 차원을 고려하지 않는 한계를 극복하여, 보다 정교한 맥락 정보를 제공하고 있습니다.

- **Performance Highlights**: SeMob는 전문가들이 구축한 데이터셋에서 평가된 결과, MAE(Mean Absolute Error)에서 13.92%, RMSE(Root Mean Square Error)에서 11.12%의 최대 감소를 달성했습니다. 특히 이벤트 발생 위치와 시간에 가까운 spatiotemporal 지역에서 두드러진 성능 향상을 보여 주었습니다. 이러한 결과는 SeMob의 텍스트 통합 접근 방식이 이동 예측의 질을 크게 향상시킨다는 것을 알 수 있습니다.



### Feasibility of Structuring Stress Documentation Using an Ontology-Guided Large Language Mod (https://arxiv.org/abs/2510.01244)
- **What's New**: 본 연구는 정신적 스트레스에 대한 온톨로지(ontology)를 개발하고, 내러티브 텍스트에서 온톨로지에 기반한 스트레스 관련 정보를 추출하기 위해 대형 언어 모델(Large Language Model, LLM)을 활용하는 가능성을 평가했습니다. 많은 의료 기록에서 스트레스가 비공식적으로 기록되는 점을 감안할 때, 온톨로지를 통한 체계적 데이터 추출 방법이 제안되었습니다.

- **Technical Details**: 정신적 스트레스 온톨로지(Mental Stress Ontology, MeSO)는 스트레스의 전이적 모델(Transactional Model of Stress)과 11개의 검증된 스트레스 평가 도구의 개념을 통합하여 개발되었습니다. MeSO는 총 181개의 개념을 포함하고 있으며, 여덟 개의 최상위 클래스(top-level classes)로 구분되어 있습니다. 연구팀은 Claude Sonnet 4를 사용하여 35개의 Reddit 게시물에서 스트레스 관련 정보를 추출하였으며, 220개의 추출 가능한 항목 중 172개(78.2%)의 항목이 정확하게 식별되었습니다.

- **Performance Highlights**: 온톨로지 기반 LLM을 통해 스트레스 관련 정보를 체계적으로 추출할 수 있는 가능성이 입증되었습니다. 인간 검토자에 의한 평가에서 메소에 매핑된 모든 올바른 추출 항목이 확인되었으나, 24개의 관련 개념은 아직 온톨로지에 포함되지 않았습니다. 향후 연구는 임상 대화 데이터와 다양한 LLM 간 비교를 포함해 보다 심도 있게 진행될 예정입니다.



### Detoxifying Large Language Models via Autoregressive Reward Guided Representation Editing (https://arxiv.org/abs/2510.01243)
Comments:
          Accepted to NeurIPS 25

- **What's New**: 본 논문에서는 Autoregressive Reward Guided Representation Editing (ARGRE)이라는 새로운 테스트 타임 탈독화 프레임워크를 제안합니다. ARGRE는 잠재 표현 공간에서의 독성 전이를 명시적으로 모델링하여 안정적이고 정밀한 편집을 가능하게 합니다. 이 프레임워크는 독성과 비독성 간의 세밀한 전이 경로를 확인하여, 독성 주석을 밀집된 훈련 신호로 변환하여 성능을 향상시킵니다. 이를 통해 ARGRE는 기존 LLM의 핵심 기능을 최소한의 저하로 유지하면서도 독성을 62.21%까지 줄이는 데 효과적입니다.

- **Technical Details**: ARGRE는 독성 전이를 패러미터로한 동시적인 보상 모델을 활용하여 토큰 표현의 독성을 평가합니다. 모델은 비독성 의미 방향을 식별하고, 이 방향을 따라 독성과 비독성 표현 사이에서 보간함으로써 미세 조정된 전이 경로를 발견합니다. 이러한 전이 경로는 덜한 주석으로도 원활한 독성 수준 조정을 가능하게 하며, ARGRE는 이를 기반으로 두 단계의 적응형 편집 프로세스를 설계하여 독성 감소를 극대화합니다.

- **Performance Highlights**: ARGRE는 여덟 개의 LLM에 대한 광범위한 실험을 통해 주요 테스트 타임 방법들과 비교할 때 독성이 평균 62.21% 감소했으며, 추론 시간은 47.58% 단축되었습니다. 이 과정에서 ARGRE는 원래 모델의 핵심 기능을 유지하면서도 높은 효율성을 보였습니다. 또한 ARGRE는 다양성 전이 탐색의 이점을 통해 높은 데이터 효율성을 나타내어, 대규모 데이터 주석 없이도 효과적으로 작동할 수 있습니다.



### Redundancy-as-Masking: Formalizing the Artificial Age Score (AAS) to Model Memory Aging in Generative AI (https://arxiv.org/abs/2510.01242)
Comments:
          34 pages, 17 figures. Includes theoretical development and mathematical proofs of the Artificial Age Score (AAS), with empirical illustrations via ChatGPT-based memory recall experiments (screenshots included)

- **What's New**: 이번 연구에서는 인공지능(AI)의 메모리 성능이 시간의 경과가 아닌 구조적 비대칭성으로 인해 노화한다는 새로운 개념이 소개되었습니다. 인공지능의 기억 aging을 평가하기 위한 새로운 지표인 Artificial Age Score (AAS)가 도입되었습니다. 이 지표는 관찰 가능한 회상 행동에서 유도된 로그 척도(log-scaled) 및 엔트로피 정보(entropy-informed)를 통해 메모리 노화를 측정합니다.

- **Technical Details**: AAS는 형식적으로 잘 정의되고 경계가 있으며 단조적인 특성을 가지고 있음을 증명되었습니다. 이 지표는 다양한 작업 및 도메인에 적용 가능하며, 'Redundancy-as-Masking'이라는 formulations를 통해 중복 정보를 해석합니다. 연구에서는 중복을 명시적으로 추정하지 않고, 모든 값이 중복 중립(settings 상에서 구성적인 상한(bound)으로 남겨져 있습니다.

- **Performance Highlights**: 25일 간의 이중언어 연구를 통해 AAS 프레임워크가 테스트되었으며, ChatGPT-5 모델을 사용하여 상태 비저장(stateless) 및 지속적(persistent) 상호작용 단계로 구조화하였습니다. 지속적인 세션에서 모델은 의미적 및 사건적 세부 정보를 일관되게 회상하며 AAS를 이론적 최소치로 가져왔으나, 세션이 초기화될 때 사건적 연속성을 유지하지 못해 AAS가 급격히 증가하는 현상이 관찰되었습니다. 이러한 발견은 인공지능 시스템의 메모리 노화를 평가하기 위한 이론적으로 기반한 진단 도구로서 AAS의 유용성을 지지합니다.



### SKYLENAGE Technical Report: Mathematical Reasoning and Contest-Innovation Benchmarks for Multi-Level Math Evaluation (https://arxiv.org/abs/2510.01241)
- **What's New**: 이 논문에서는 두 가지 상호 보완적인 수학 벤치마크인 SKYLENAGE-ReasoningMATH와 SKYLENAGE-MATH를 소개합니다. SKYLENAGE-ReasoningMATH는 100개의 문제로 구성된 구조 인지 진단 세트이며, 각 항목의 메타데이터를 포함하고 있습니다. SKYLENAGE-MATH는 고등학교부터 박사 과정까지의 네 가지 단계에서 나오는 150개의 문제로, 수학적 reasoning 능력을 평가하려는 목적을 가지고 있습니다.

- **Technical Details**: 대규모 언어 모델(LLMs)의 성능 평가를 위해 15개의 최신 모델 변형이 동일한 설정 아래에서 분석되었습니다. 기존 벤치마크의 한계를 극복하기 위해 SKYLENAGE는 응답 정확도와 구조-성능 관계를 분석하기 위한 다양한 메트릭을 사용합니다. 또한 주제 및 모델 성과에 대한 히트맵과 레이더 프로파일을 통해 세부적인 분석을 제공합니다.

- **Performance Highlights**: 이 연구에서 가장 강력한 모델은 SKYLENAGE-MATH에서 44%의 정확도를 기록했고, 두 번째 모델은 37%의 정확도를 보였습니다. 전반적으로 SKYLENAGE-ReasoningMATH의 최고 모델은 81%의 정확도를 달성했습니다. 분석 결과, 고등학교에서 박사 과정으로 이동할수록 성과가 급감하여, 논리적 문제 해결 능력을 평가하기 위한 새로운 기준의 필요성을 강조합니다.



### CIFLEX: Contextual Instruction Flow for Sub-task Execution in Multi-Turn Interactions with a Single On-Device LLM (https://arxiv.org/abs/2510.01239)
Comments:
          accepted at EMNLP 2025 (main)

- **What's New**: CIFLEX(컨텍스추얼 인스트럭션 플로우 포 서브태스크 실행)는 다중 턴 상호작용에서 단일 온디바이스 대형 언어 모델(LLM)을 통해 효율적인 서브태스크 처리를 위한 새로운 실행 시스템입니다. LLM이 더욱 강력해짐에 따라 단일 모델이 사용자의 요청에 대한 효과적이고 포괄적인 지원을 위해 다양한 서브태스크를 처리하는 것이 기대됩니다. CIFLEX는 주요 태스크와 서브태스크 간의 전환 시 전체 대화 맥락을 재처리하는 대신, 주요 작업의 키-값 (KV) 캐시를 재사용하고 특정 서브태스크 전용 지침만 주입하여 계산 오버헤드를 줄입니다.

- **Technical Details**: CIFLEX는 주요 작업에서 획득한 키-값 캐시를 활용하여 서브태스크 실행 후 매인 경로로 돌아갈 때 중복되는 사전 채우기 계산을 피합니다. 이를 통해 계산 비용을 크게 줄이는 동시에 태스크 성능을 저하시키지 않도록 설계되었습니다. 또한, 서브태스크 선택을 지원하기 위해 소규모 모델에 맞춘 계층적 분류 전략을 개발하여 다중 선택 결정을 이진 결정으로 분해합니다.

- **Performance Highlights**: 실험 결과 CIFLEX는 계산 비용을 상당히 줄이는 데 성공하면서도 태스크 성능을 유지하였습니다. 이 시스템은 온디바이스에서 효율적이고 확장 가능한 다중 태스크 대화를 가능하게 만들어, 사용자와의 상호작용을 더욱 매끄럽고 효과적으로 개선합니다.



### Silent Tokens, Loud Effects: Padding in LLMs (https://arxiv.org/abs/2510.01238)
Comments:
          NeurIPS 2025 Workshop: LLM Evaluation

- **What's New**: 이 논문에서는 배치 추론 중 시퀀스 길이를 맞추기 위해 사용되는 패딩 토큰(padding tokens)의 영향력을 시스템적으로 연구했습니다. 패딩 토큰이 실제로는 숨겨진 표현을 변화시키고 품질을 저하시키며 예기치 않게 편향(bias)과 안전(safety) 기준에 영향을 미칠 수 있음을 보여줍니다. 이러한 결과는 패딩 처리 방식이 단순한 기술적 세부사항이 아니라 실제 배포에서 고려해야 할 위험 요소임을 강조합니다.

- **Technical Details**: 연구에서는 Llama, Gemma, Qwen의 세 가지 오픈소스 모델 패밀리를 대상으로 패딩 토큰의 영향을 평가했습니다. 패딩의 양을 조절하여 내부 활성화(activations), 생성 품질(generation quality), 사회적 편향(social bias), 안전(safety)이라는 네 가지 축에서 결과를 정량화했습니다. 특히, 패딩의 존재는 내재된 표현을 변화시키고 출력 분포를 바꾸며, 그로 인해 품질이 저하되고 편향이 심화되는 것처럼 보였습니다.

- **Performance Highlights**: 실험 결과, 패딩 토큰의 수가 증가할수록 여러 모델의 활성화 유사성이 낮아지고 생성 품질이 급격히 떨어지는 경향이 나타났습니다. 특히, Llama 및 Qwen의 작고 오래된 모델에서 이러한 현상이 두드러지며, Gemma 모델은 패딩에 대해 뛰어난 회복력을 보였습니다. 사회적 편향 측면에서도 패딩이 특정 카테고리와 맥락에 따라 편향을 약화하거나 증가시킬 수 있음을 보여주었고, 안전성 측면에서도 패딩이 포함된 경우 유해한 프롬프트에 대한 모델의 응답률이 상승하는 경향이 관찰되었습니다.



### Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation (https://arxiv.org/abs/2510.01237)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Model)에서 발생하는 허위 정보 생성 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 기존의 사후 수정(post-generation correction) 방법 대신에, 이 연구는 생성 전 모델의 불확실성을 평가하여 더 신뢰할 수 있는 응답 메커니즘으로 쿼리를 사전 가이딩하는 방식을 도입합니다. 이로 인해 모델의 신뢰성을 높이고 계산 비용을 줄일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 시스템은 세 가지 신호를 결합하여 신뢰도(confidence)를 추정합니다: 내부 표현과 참조 임베딩 간의 의미적 정렬(semantic alignment), 모델 계층 간의 내부 수렴(internal convergence) 분석, 그리고 학습된 신뢰도 추정(learned confidence)입니다. 이러한 신호를 바탕으로 신뢰도 점수를 계산하고, 이를 네 가지 경로(local generation, retrieval-augmented generation, larger models, human review)로 매핑하는 결정론적 라우팅 시스템을 구현합니다. 이는 쿼리의 신뢰도에 따라 적절한 응답 경로를 선택할 수 있게 합니다.

- **Performance Highlights**: 이 연구는 지식 집약적 질의응답(Knowledge-Intensive QA) 벤치마크에서 성능을 평가하며, 허위 정보 탐지에서 기존 기준선(baseline)보다 큰 향상(0.74 vs. 0.42)을 보였습니다. 또한 사후 수정 방법과 비교하여 계산 비용을 40% 줄이면서 F1 스코어를 0.61에서 0.82로 개선시키고, 낮은 허위 긍정률(0.09)을 유지합니다. 이러한 결과는 반응 수정에서 사전 평가로의 패러다임 전환이 LLM의 신뢰성을 효과적으로 향상시킬 수 있음을 보여줍니다.



### GRPO++: Enhancing Dermatological Reasoning under Low Resource Settings (https://arxiv.org/abs/2510.01236)
Comments:
          Will be submitted at IEEE JBHI

- **What's New**: DermIQ-VLM은 피부과 진단 과정을 모방한 Vision-Language Model(VLM)으로, 데이터 부족 및 고비용 훈련 기술의 한계를 극복하고자 다단계 자원 효율적 방법론을 사용해 개발되었습니다. 이 모델의 핵심 기여는 데이터 집약적인 Grouped Relative Policy Optimization(GRPO) 프레임워크를 안정화한 GRPO++의 도입입니다.

- **Technical Details**: 훈련 파이프라인은 질병 인식을 위한 GRPO++를 사용한 후 대화 능력을 위한 감독 학습 미세조정(Supervised Fine-Tuning, SFT)을 진행합니다. 이후, Direct Preference Optimization(DPO)을 사용하여 사실적 오류를 완화하고, 신뢰할 수 있는 전문가 선호를 모사한 지식 그래프 기반 시스템을 통해 모델을 정렬합니다. 이러한 접근 방식은 dermatological 데이터 셋에서 눈에 띄는 성능 향상을 보여주었습니다.

- **Performance Highlights**: 예비 평가 결과, DermIQ-VLM은 보통의 미세조정 기법에 비해 훨씬 더 높은 성과를 기록했습니다. 이는 자원 제약 환경에서도 신뢰할 수 있는 특화된 VLM을 개발하는 것이 가능함을 입증합니다. 이 연구는 피부과 진단 지원을 위한 해석 가능한 AI의 발전에 기여할 것으로 기대됩니다.



### LLMRank: Understanding LLM Strengths for Model Routing (https://arxiv.org/abs/2510.01234)
Comments:
          13 pages, 1 figure

- **What's New**: 대규모 언어 모델(LLMs)의 배포에는 각 요청(prompt)에 대해 적합한 모델을 선택하는 것이 핵심적인 과제가 됩니다. 이에 대한 해결책으로 LLMRank라는 프레임워크를 소개합니다. LLMRank는 요청에서 추출한 인적 가독적인 특성들을 활용하여 각 모델의 유틸리티를 예측하고, 기존의 방법들보다 더 효율적이고 투명한 라우팅을 제공합니다.

- **Technical Details**: LLMRank는 요청(prompt)에서 작업 유형, 추론 패턴, 복잡성 지표, 통사적 단서(syntactic cues)와 같은 다각적인 특성을 추출하는 파이프라인을 개발합니다. 신경망 기반의 랭킹 모델을 통해 개별 모델의 유틸리티를 예측하며, 수익 모델(reward model)과/또는 쌍 비교 손실(pairwise ranking losses)을 결합한 하이브리드 ranking 방식을 사용합니다. 또한, 모델 비용을 직접적으로 훈련 과정에 통합하여 성능과 비용 간의 균형을 맞추는 유연한 배포 전략을 제공합니다.

- **Performance Highlights**: LLMRank는 RouterBench와 관련 벤치마크를 사용한 실험에서 기존 상태의 라우팅 품질을 초월하는 성과를 달성하였습니다. 또한, 복잡한 요청과 사실 조회 간 모델 선택의 직관적인 패턴을 포착한 피처 속성(feature attribution) 분석을 통해 투명성을 제공합니다. 성능에 따라 비용 조정을 할 수 있는 능력 덕분에 LLMRank는 비용 효율성을 높일 수 있는 가능성을 지니고 있습니다.



### Computational Social Linguistics for Telugu Cultural Preservation: Novel Algorithms for Chandassu Metrical Pattern Recognition (https://arxiv.org/abs/2510.01233)
Comments:
          16 pages, 4 figures

- **What's New**: 이번 연구는 텔루구 초시(Chandassu)를 보존하기 위한 계산 사회 과학적 접근을 제시합니다. 우리는 전통적인 공동체 지식과 현대의 계산 방법을 연결하는 포괄적인 디지털 프레임워크를 개발하였습니다. 이 프레임워크는 4,651개의 주석이 달린 padyams의 공동 데이터 세트를 포함하며, 프로소디(prosody) 분석을 위한 AksharamTokenizer와 패턴 인식을 위한 PadyaBhedam Checker를 포함합니다.

- **Technical Details**: 텔루구의 프로소디는 경량 음절(laghuvu)과 중량 음절(guruvu)로 구성되며, 각각의 aksharam(글자) 단위는 고유한 음절 무게와 전통적인 프로소디 규칙에 따라 분류됩니다. 연구는 총 4,651개의 padyam을 포함한 데이터 세트를 구축하고, LaghuvuGuruvu 애너테이션을 제공하여 초시에 대한 자동화된 식별 알고리즘을 제안합니다. 이는 각각의 프로소딕 특성을 유지하면서 측정 가능한 Chandassu Score를 도입하는 데 기여합니다.

- **Performance Highlights**: 제안된 알고리즘은 91.73%의 정확도를 달성하고 있으며, 이는 전통적인 문학 기준들을 반영합니다. 이러한 성과는 계산 사회 과학이 멸종 위기에 처한 문화 지식 체계를 보존하는 데 어떻게 기여할 수 있는지를 보여줍니다. 연구 방법론은 지역 중심의 문화 보존 접근법에 대한 통찰력을 제공하여 디지털 인문학과 사회적 인식을 갖춘 계산 시스템의 폭넓은 이니셔티브를 지원합니다.



### Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks (https://arxiv.org/abs/2510.01232)
Comments:
          16 pages, 5 figures. Accepted to EMNLP 2025 main conference

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 성능을 측정하기 위한 새로운 방법론인 벤치마크 프로파일링(Benchmark Profiling)을 제안합니다. 이 접근 방식은 벤치마크 성과를 10가지 인지적 능력으로 분해하여 실제 능력을 평가하는 체계적인 방법을 제공합니다. 기존 벤치마크가 주장하는 능력을 진정으로 측정하는지에 대한 의문을 해결하고, 모델의 성공에 대한 각 능력의 기여도를 정량화합니다.

- **Technical Details**: 벤치마크 프로파일링 방법은 특정 능력에 대한 매개변수를 선택적으로 제거하여 해당 능력이 벤치마크에 미치는 영향을 측정합니다. 10개의 운영화된 능력(예: Deductive Reasoning, Contextual Recall)을 정의하며, 이는 인간의 지능 모델에 기반하여 개발되었습니다. 이는 벤치마크가 실제로 사용하는 능력 조합을 파악하기 위한 기초 데이터셋을 생성하고, 그에 따라 능력 영향 점수(Ability Impact Score, AIS)를 계산하여 능력 기여도를 평가합니다.

- **Performance Highlights**: 세 가지 지침 조정 모델을 대상으로 한 분석 결과, 대부분의 벤치마크가 특정 능력 하나에 의존하지 않고 여러 능력을 활용한다는 사실을 발견했습니다. 유사한 라벨을 가진 데이터셋이 서로 다른 능력 조합에 의존하며, 코드 생성 벤치마크는 여러 기술 향상을 보상하여 좁은 도메인 전문화의 세밀한 조정에서 그 효과가 미미하다는 점을 보여주고 있습니다. 이러한 분석은 모델의 성능 향상이 실제 사용자 경험에 긍정적이지 않을 수 있는 이유를 설명합니다.



### Trustworthy Summarization via Uncertainty Quantification and Risk Awareness in Large Language Models (https://arxiv.org/abs/2510.01231)
- **What's New**: 이번 연구는 고위험 상황에서의 자동 요약의 신뢰성을 다루고 있으며, 불확실성 정량화(uncertainty quantification)와 위험 인식 메커니즘(risk-aware mechanisms)을 통합한 대형 언어 모델 프레임워크를 제안합니다. 정보 과부하(information overload)와 고위험 의사결정(high-risk decision-making)의 필요성에서 출발하여 조건부 생성 기반 요약 모델이 구축되었습니다.

- **Technical Details**: 생성 과정 중 베이지안 추론(Bayesian inference)을 도입하여 매개변수 공간에서의 불확실성을 모델링하고, 예측 분포 엔트로피(predictive distribution entropy)를 사용하여 생성된 콘텐츠의 불확실성 수준을 측정합니다. 핵심 정보가 보존되고 위험 속성이 명확하게 표현되도록 엔트로피 정규화(entropy regularization)와 위험 인식 손실(risk-aware loss)의 공동 최적화(joint optimization)가 적용됩니다.

- **Performance Highlights**: 비교 실험과 민감도 분석(sensitivity analyses)을 통해 제안된 방법이 고위험 응용 프로그램에서 요약의 강인성과 신뢰성을 크게 향상시키는 것을 확인했으며, 유창함과 의미적 완전성(semantic integrity)을 유지합니다. 본 연구는 신뢰할 수 있는 요약을 위한 체계적인 해결책을 제공하고 방법론적 차원에서 확장성(scalability)과 실용적 가치를 입증합니다.



### Geometric Structures and Patterns of Meaning: A PHATE Manifold Analysis of Chinese Character Embeddings (https://arxiv.org/abs/2510.01230)
Comments:
          33 pages, 17 figures

- **What's New**: 이 연구는 중국어 문자(Chinese character) 임베딩(embeddings)의 기하학적 패턴을 PHATE 매니폴드 분석(manifold analysis)을 통해 체계적으로 조사합니다. 특히, 7개의 임베딩 모델과 8개의 차원 축소 기법(dimensionality reduction methods)을 통해 콘텐츠 단어(content words)와 기능 단어(function words)의 클러스터링(clustering) 패턴을 관찰했습니다. 이 연구는 기존의 언어학적 이론(linguistic theory)을 지지하는 새로운 계산적 근거를 제공합니다.

- **Technical Details**: 연구는 1000개 이상의 중국어 문자(character)를 12개의 의미론적 도메인(semantic domains)에서 분석하여 의미 있는 문자들은 복잡한 기하학적 다양성을 보이는 반면, 구조적 기본자(radicals)는 조밀한 클러스터(cluster)로 수렴하는 경향이 있음을 나타냅니다. PHATE 분석은 채택된 차원 축소 기법에 따라 기하학적 복잡성(geometric complexity)과 의미(content) 간의 상관관계를 보여줍니다. 이 데이터는 123개 구문(phrases)의 포괄적인 어린이 네트워크(child-network) 분석을 통해 체계적인 의미 확장(semantic expansion)을 입증합니다.

- **Performance Highlights**: 이 연구 결과는 의미론적 조직의 기하학적 분석을 위한 새로운 프레임워크(framework)를 수립하고 있으며, 데이터 분석(data analysis)를 통해 의미적인 복잡성이 문자에 따라 어떻게 다르게 나타나는지를 보여줍니다. 연구는 기존의 이론에 새로운 계산적 접근 방식을 추가하며, 중국어 문자에 대한 기초 연구에 중요한 통찰을 제공합니다.



### Enhancing Transformer-Based Rerankers with Synthetic Data and LLM-Based Supervision (https://arxiv.org/abs/2510.01229)
Comments:
          Accepted by RANLP 2025

- **What's New**: 본 논문에서는 효과적인 문서 재정렬(document reranking)이 다양한 응용 프로그램에서 검색의 적합성을 향상시키는 데 필수적임을 강조합니다. 특히, 대형 언어 모델(LLMs)의 높은 계산 비용으로 인해 이들을 실제로 활용하는 데 어려움이 있어, 소형(task-specific) 모델의 파인튜닝(fine-tuning)을 통해 해결하고자 합니다. 논문의 기존의 인간 라벨링된 쿼리-문서 쌍을 필요로 하지 않는 독창적인 파이프라인을 제안하며, 도메인 특화된 데이터셋에서 합성 쿼리를 생성하여 이해도를 높입니다.

- **Technical Details**: 저자들은 LLM을 사용하여 도메인 특화된 말뭉치(corpora)에서 합성 쿼리를 생성하고, 이를 통해 긍정적 및 하드-부정(hard-negative) 쌍을 라벨링하는 새로운 방법론을 개발했습니다. 이렇게 생성된 합성 데이터셋은 대조 학습(contrastive learning)을 통해 소형 변환기(transformer) 모델을 파인튜닝 하는 데 활용되며, 여기에 Localized Contrastive Estimation (LCE) 손실(loss)을 적용합니다. 이러한 접근 방식은 기존의 수작업 라벨링 데이터 없이도 LLM 수준의 재정렬 성능을 가능하게 합니다.

- **Performance Highlights**: MedQuAD 데이터셋에서의 실험 결과, 저자들이 제안하는 방법론은 도메인 내 성능을 크게 향상시키며, 도메인 외 작업에서도 좋은 일반화 성능을 보여주었습니다. 이는 LLM을 데이터 생성 및 감독에 활용하면서도 계산 비용을 줄이고 강력한 재정렬 능력을 유지할 수 있음을 입증합니다. 연구 결과는 도메인 특화 재정렬과 검색의 정밀도를 개선하는 데 있어서 새로운 길을 제시합니다.



### Who is In Charge? Dissecting Role Conflicts in Instruction Following (https://arxiv.org/abs/2510.01228)
- **What's New**: 최근 연구에서 대형 언어 모델 (LLMs)은 사용자 입력보다 시스템 프롬프트가 우선하는 계층적 지침을 따르는 것으로 설계되었으나, 실제로는 이러한 규칙을 자주 무시하는 경향을 보입니다. 본 연구는 이러한 행동 발견을 기계적인 해석으로 확장하여, 대규모 데이터 세트를 분석하였습니다. 연구 결과는 LLMs가 사회적 신호에 강한 순응을 보이는 반면, 시스템 지침에 대한 복종이 약하다는 것을 보여주었습니다.

- **Technical Details**: 본 연구에서는 Llama-3.1-8B-Instruct 모델을 사용하여 LLMs의 계층 간 충돌 결정을 이해하기 위한 실험을 수행했습니다. 데이터 세트는 서로 배타적인 지침을 포함하는 120,000개의 프롬프트로 이루어져 있으며, 여기서는 시스템-사용자 역할 분리에 따른 충돌과 사회적 위계에 따른 충돌을 검토하였습니다. 이를 통해 LLMs의 내부 상태를 분석하여 충돌 신호가 어떻게 인코딩되고 있는지를 탐색했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 사용자-시스템 역할 구분에서 더 강한 충돌 결정 신호를 보여주었으며, 모델의 내부 표현이 신속하게 형성됨을 확인했습니다. 특히, 초기 레이어에서 AUC가 0.89를 초과하며 신뢰할 수 있는 신호로 해석되었습니다. 하지만 나중 레이어에서는 생성 관련 계산 통합으로 인해 충돌 신호가 약해지는 경향을 보였습니다.



### EEFSUVA: A New Mathematical Olympiad Benchmark (https://arxiv.org/abs/2510.01227)
Comments:
          16 Pages, 5 figures

- **What's New**: 최근의 연구에서 대형 언어 모델(LLMs)이 수학 벤치마크에서 올림픽 금메달에 해당하는 수준에 도달했다고 주장되었습니다. 본 연구는 이러한 주장에 대해 자세히 검토하고 현재 벤치마크가 실제 LLM의 수학적 추론을 얼마나 잘 포착하고 있는지를 평가했습니다. 기존 벤치마크가 국제 수학 올림피아드(IMO) 및 연관된 대회에서 주로 출발했음을 지적하며, 데이터 오염 및 Familiar problem types(익숙한 문제 유형)에 대한 협소한 초점이 모델의 추론 능력을 과대 평가할 수 있다고 언급합니다.

- **Technical Details**: 이 연구에서는 EEFSUVA라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 동유럽의 지역 및 국가 올림피아드에서 선별된 문제들로 구성되어 있으며, IMO와 유사한 난이도를 가진 비표준 문제 해결 기술이 요구됩니다. 하지만 이러한 문제들은 온라인 코퍼스에서 훨씬 덜 보편적입니다. 초기 결과에 따르면, 최신 LLM조차도 EEFSUVA에서 다른 올림픽 스타일 벤치마크에 비해 성능이 상당히 감소하는 것으로 나타났습니다.

- **Performance Highlights**: 이 발견은 수학적 추론의 보다 완전한 평가를 위한 더 광범위한 데이터셋의 중요성을 제기합니다. 연구는 향후 모델 개발을 이끄는 데 유용할 수 있는 통찰력을 제공하는 것을 목표로 합니다. 따라서 LLM의 성능을 진지하게 평가하기 위해 다양한 벤치마크의 필요성이 강조됩니다.



### ClaimCheck: Real-Time Fact-Checking with Small Language Models (https://arxiv.org/abs/2510.01226)
- **What's New**: ClaimCheck는 실시간 웹 증거를 사용하여 실제 주장을 검증하기 위해 설계된 LLM(언어 모델) 기반 자동 사실 검증 시스템입니다. 이전의 많은 시스템이 대형 폐쇄형 모델과 정적 지식 저장소에 의존했던 것과 달리, ClaimCheck는 인간의 사실 검증 흐름을 모방한 투명하고 단계적인 검증 파이프라인을 사용합니다. 이를 통해 웹 검색 쿼리 계획, 웹 기반 증거 검색 및 요약, 증거 합성 및 재검색, 주장 결과 평가의 과정을 포함하고 있습니다.

- **Technical Details**: ClaimCheck의 각 모듈은 작은 LLM을 최적화하도록 설계되어 있으며, 이는 시스템이 정확하고 해석 가능한 사실 검증을 수행할 수 있도록 합니다. 이 시스템은 훨씬 작은 Qwen3-4B 모델을 사용하지만, AVeriTeC 데이터셋에서 76.4%의 최첨단 정확도를 달성하여 LLaMA3.1 70B 및 GPT-4o 모델을 사용하는 이전 접근 방식보다 우수한 성능을 보입니다. 모듈 설계와 프롬프팅 전략의 세심한 접근이 작은 LLM의 한계를 극복할 수 있음을 보여주는 많은 실험 결과도 포함되어 있습니다.

- **Performance Highlights**: ClaimCheck는 상대적으로 낮은 계산 요구 사항에도 불구하고 높은 정확도를 기록하며, 이를 통해 저비용 자동 사실 검증 시스템의 가능성을 제시합니다. 특히 실시간으로 업데이트된 웹 정보를 활용하여 사실 확인을 수행하는 점에서 큰 장점을 가지고 있습니다. 사용자 접근성을 높이고 투명성을 촉진하기 위해 공개 데모를 제공하며, 일반 사용자에게 이 시스템을 체험할 수 있는 기회를 제공합니다.



### Context Matters: Comparison of commercial large language tools in veterinary medicin (https://arxiv.org/abs/2510.01224)
Comments:
          4 Figures, 10 pages

- **What's New**: 본 연구는 대형 언어 모델(Large Language Models, LLMs)이 임상 환경에서 점차 사용되고 있지만, 수의학( veterinary medicine) 분야에서는 그 성능이 충분히 탐구되지 않았음을 강조합니다. 연구자들은 세 가지 상업적으로 이용 가능한 수의학 전용 LLM 요약 도구를 평가하였으며, 이는 Hachiko(제품 1)와 그 외 제품 2 및 3으로 구성됩니다. 이들은 수의학 종양학 기록의 표준화된 데이터셋을 바탕으로 테스트되었습니다.

- **Technical Details**: LLM을 심사자로 사용하는 구조화된 평가 프레임워크를 적용하여 요약정도를 평가했습니다. 평가 도메인은 사실 정확성(Factual Accuracy), 완전성(Completeness), 연대 순서(Chronological Order), 임상 관련성(Clinical Relevance), 조직화(Organization)로 나뉘어 있습니다. 제품 1은 4.61(중앙값, IQR: 0.73)로 가장 높은 점수를 기록하였으며, 제품 2는 2.55(IQR: 0.78), 제품 3은 2.45(IQR: 0.92)의 점수를 나타냈습니다.

- **Performance Highlights**: 제품 1은 사실 정확성과 연대 순서에서 완전한 중앙값 점수를 받으며 전반적으로 가장 높은 성능을 보였습니다. 평가 과정은 삼회의 독립적인 실행을 통해 내부 일관성을 확인하였고, LLM 심사자의 점수 재현성은 매우 높았습니다. 평균 점수의 표준편차는 제품 1이 0.015, 제품 2는 0.088, 제품 3은 0.034로 나타났습니다.



### Discourse vs emissions: Analysis of corporate narratives, symbolic practices, and mimicry through LLMs (https://arxiv.org/abs/2510.01222)
- **What's New**: 해당 연구는 기업의 기후 관련 공시에서 다차원적 프레임워크를 개발하여 828개 기업의 공시 성숙도를 평가합니다. 이를 위해 기후 커뮤니케이션에 맞춰 조정된 대형 언어 모델(LLMs)을 사용하여 공시의 품질을 정량적으로 측정하는 새로운 접근 방식이 도입되었습니다. 연구는 기업의 속성에 따라 기후 공시가 어떻게 다르게 나타나는지를 분석하여, 기업들이 단순 모방을 넘어 실질적인 개선을 이룰 수 있는지에 대한 통찰을 제공합니다.

- **Technical Details**: 연구에서는 감정(sentiment), 약속(commitment), 특정성(specificity), 목표 야망(target ambition) 등 네 가지 분류기를 통해 기업의 지속가능성 및 연례 보고서에서 서술적 지표를 추출합니다. 이러한 지표는 CO2 배출량, 시장 규모, 산업 부문 등의 기업 속성과 연결되어 분석됩니다. 이 방법론은 대화형 모델을 통해 작성된 문장을 기반으로 하여 각 기업의 공시 내용과 신뢰성을 체계적으로 평가하는 데 중점을 둡니다.

- **Performance Highlights**: 연구 결과, 리스크 중심의 서술이 명확한 약속과 함께 나타나는 경향이 있으며, 대형 기업이나 고배출 기업이 더 많은 약속과 행동을 보고하는 것으로 나타났습니다. 하지만 이러한 약속이 정량적 목표와 불일치하는 경우가 많아 상징적인 실천으로 이어질 수 있음을 시사합니다. 또한, 전 산업에 걸쳐 공시 스타일의 유사성이 높아지는 경향은 모방 행동을 나타내며, 이는 투자자에게 신뢰성 있는 정보를 제공하는 데 제한적일 수 있습니다.



### Towards Open-Ended Discovery for Low-Resource NLP (https://arxiv.org/abs/2510.01220)
Comments:
          Proceedings of the 2nd Workshop on Uncertainty-Aware NLP (UncertaiNLP) at EMNLP 2025

- **What's New**: 이번 논문에서는 저자들이 저자원 언어를 위한 NLP에서의 기존 접근법을 넘어서야 한다고 강조합니다. AI 시스템이 정적인 데이터셋 대신 대화를 통해 동적으로 새로운 언어를 배우도록 하여 인간과 기계의 협력을 통한 언어 발견을 목표로 하고 있습니다. 이러한 변화는 데이터 수집에서 참여 기반 학습 과정으로의 전환을 촉진하여 언어 기술의 미래를 재구성하는 것입니다.

- **Technical Details**: 저자들은 AI 시스템이 대화를 통해 새로운 언어를 배울 수 있는 프레임워크를 제안합니다. 이들은 에피스템 불확실성(epistemic uncertainty)과 인간 화자의 학습 신호를 결합하여 상호작용을 안내하게 합니다. 이러한 접근법은 기존의 데이터 기반 모델이 아닌, 인간과 기계 간의 동적 상호작용을 기반으로 하고 있으며, 이를 통해 확장 가능하고 포괄적인 NLP 시스템을 구축할 수 있습니다.

- **Performance Highlights**: 저자들은 아프리카 언어와 같은 저자원 언어의 NLP 연구에서 여전히 많은 도전 과제가 존재한다고 지적합니다. 기존의 솔루션들은 대부분 대량의 텍스트 데이터에 의존하고 있지만, 저자원 언어는 그러한 데이터가 부족하여 효과적인 지원이 어렵습니다. 이 논문은 그러한 문제들을 해결하기 위한 새로운 접근 방법을 제시하며, 인간과 기계의 협업을 통해 언어 학습을 지속적으로 개선할 수 있는 가능성을 보여줍니다.



### Uncovering Implicit Bias in Large Language Models with Concept Learning Datas (https://arxiv.org/abs/2510.01219)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)에서 잠재적인 편향을 발견하기 위해 새로운 개념 학습(task) 데이터셋을 소개합니다. 연구자들은 LLM이 양적 표현에 대해 상승적 단조성(upward monotonicity) 편향을 보일 수 있음을 발견했습니다. 이러한 편향은 직접 프롬프트(prompt)를 사용할 때는 덜 명확해지는 특성을 가집니다. 이는 개념 학습(in-context concept learning)을 통해 모델의 숨겨진 편향을 발견하는 데 효과적일 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 두 가지 개념, 즉 상승적(monotone) 및 하강적(downward) 단조성을 갖는 숫자 개념에 대해 LLM을 평가했습니다. 각 프롬프트는 20개의 레이블 달린 예제(positive와 negative)로 구성되며, 프롬프트의 반응 확률을 비교하여 모형의 응답을 평가했습니다. 실험에 사용된 모델은 OLMo 2와 Qwen3으로, 이들은 다양한 벤치마크에서 경쟁력 있는 성능을 나타내는 오픈(weight) 모델입니다.

- **Performance Highlights**: 실험 결과, 모델은 개념 학습을 통해 상승적 단조성 개념에서 더 높은 정확도를 보였습니다. 반면, 명시적 의미를 사용하는 경우 정확도 차이가 줄어드는 양상을 보였습니다. Qwen3-32B 모델은 두 평가 방법 간의 정확도 차이가 크지 않았으나, OLMo 모델에서 상승적 단조성에 대한 편향이 뚜렷하게 관찰되었습니다. 이러한 결과는 개념 학습이 LLM의 숨겨진 인지 편향을 드러내는 데 도움이 될 수 있음을 제안합니다.



### Interactive Training: Feedback-Driven Neural Network Optimization (https://arxiv.org/abs/2510.02297)
Comments:
          EMNLP 2025 Demo

- **What's New**: 이번 논문에서는 Interactive Training이라는 새로운 오픈 소스 프레임워크를 소개합니다. 이 프레임워크는 네트워크 훈련 중에 실시간으로 피드백을 받아 인공지능이 자동으로 개입하게 하며, 사용자 또는 AI 에이전트의 개입을 허용합니다. 이를 통해 사용자는 옵티마이저의 하이퍼파라미터, 훈련 데이터 및 모델 체크포인트를 동적으로 조정할 수 있습니다.

- **Technical Details**: Interactive Training 프레임워크의 핵심은 Control Server입니다. 이 서버는 사용자의 명령과 지속적인 훈련 프로세스 간의 통신을 중재하며, FastAPI를 통해 API 엔드포인트를 노출하고 JSON 메시지를 통해 명령을 처리합니다. Interactive Trainer는 Hugging Face의 Trainer 클래스에 기반하여 구현되어, 동적으로 조정된 훈련 매개변수를 바탕으로 실시간 개입에 반응합니다.

- **Performance Highlights**: 세 가지 사례 연구를 통해 우리의 접근 방식이 기존의 정적 최적화 방법들보다 우수하다는 것을 보여줍니다. 경험이 있는 인간 개발자들이 실시간 인터랙션을 활용하여 더 나은 최적화 결과를 도출했고, AI 에이전트가 자동으로 초기 하이퍼파라미터를 수정할 수 있는 가능성도 입증되었습니다. 또한, 이 프레임워크는 실제 배포 중 수집된 사용자 데이터를 실시간으로 반영하여 모델의 적응성을 향상시킵니다.



### Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (https://arxiv.org/abs/2510.02286)
- **What's New**: 이 연구는 DialTree-RPO라는 새로운 강화 학습 프레임워크를 제안하여 다중 턴 공격 전략을 자율적으로 발견하는 방법을 제시합니다. 기존의 접근 방식들이 단일 턴 공격에 초점을 맞춘 것과 달리, DialTree-RPO는 대화를 연속적인 의사 결정 문제로 간주하여 다중 턴 공격의 다양한 가능성을 탐색합니다. 이 과정에서, 자동화된 데이터 없이도 공격 성공률을 극대화할 수 있는 최적의 대화 정책을 학습합니다.

- **Technical Details**: DialTree-RPO는 다중 턴 공격을 위한 정책 최적화를 목표로 하는 새로운 강화 학습 프레임워크입니다. 이 시스템은 대화 나무 롤아웃 (dialogue tree rollout) 및 질 높은 경로 프루닝 (quality-aware pruning) 기법과 적응적 마스킹 (adaptive masking) 기술을 통합하여, 훈련 중 비효율적인 공격 경로를 제거하고 최적화 안정성을 높이며 효율성을 개선합니다. 이를 통해 다중 턴 대화의 복잡성을 효과적으로 적용할 수 있습니다.

- **Performance Highlights**: DialTree-RPO는 10개의 대상 모델에서 평균 공격 성공률 (ASR) 85.3%를 달성하여 이전 최신 기술보다 25.9% 더 높은 성능을 보였습니다. 이 연구는 모델 크기와 관계없이 뛰어난 효율성과 전이 능력을 보여주며, 새로운 공격 전략을 발견하는 데에도 효과적입니다. 결과적으로 DialTree-RPO는 다중 턴 강화 학습을 기반으로 한 새로운 최첨단 레드 팀링 방법을 수립했습니다.



### RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems (https://arxiv.org/abs/2510.02263)
- **What's New**: 이 연구에서는 복잡한 문제에 대한 해답을 도출하기 위해 'algorithmic procedures'를 식별하고 구현하는 과정을 중심으로 새로운 접근법을 제안합니다. 특히, reasoning abstractions을 도입하여 모델이 수월하게 성공적인 추론을 배울 수 있도록 돕습니다. 이러한 방식은 모델이 문제를 해결하는 데 필요한 프로시저와 사실 지식에 대한 간결한 자연어 설명을 활용하도록 합니다.

- **Technical Details**: 새로운 RLAD(RL Abstraction and Decomposition) 방법론은 두 개의 역할을 플랫폼에서 활성화하여, 추상화를 생성하는 모델과 해결책을 생성하는 모델을 공동으로 학습시킵니다. 핵심 요소로는 curriculum training(커리큘럼 학습), 비추상화 프롬프트 포함, 그리고 보상 마스킹(reward masking) 기법이 포함되어 있습니다. 이러한 기법을 사용하면 모델의 성능이 향상되며, 고차원적인 프로시저 지식을 효과적으로 활용하여 더 어려운 문제에도 잘 일반화합니다.

- **Performance Highlights**: RLAD 방법은 AIME 2024 및 HMMT 2025의 두 수학 추론 벤치마크에서 기존 모델들보다 우수한 성능을 기록했습니다. 추상화를 기반으로 한 훈련이 더욱 좋은 일반화를 보여주며, 비추상화 프롬프트를 포함시키고 적절한 보상 마스킹을 통해 모델 성능을 크게 향상시켰습니다. 최종적으로, 커리큘럼 학습, 비추상화 프롬프트 포함, 보상 마스킹을 조합한 것이 다른 구성들보다 현저하게 우수한 결과를 가져왔습니다.



### The Unreasonable Effectiveness of Scaling Agents for Computer Us (https://arxiv.org/abs/2510.02250)
Comments:
          23 pages, 7 figures, 10 tables

- **What's New**: 이번 연구에서는 컴퓨터 사용 에이전트(CUAs)의 넓은 스케일링을 위한 새로운 방법인 Behavior Best-of-N(bBoN)을 소개합니다. bBoN은 에이전트의 롤아웃을 생성하고 이를 비교하기 위해 행동 서사를 사용하여 여러 롤아웃 간의 선택을 가능하게 합니다. 이를 통해 저항력과 성공률이 크게 향상되었으며, 기존 방법들과 비교했을 때 놀라운 성능 개선을 달성했습니다.

- **Technical Details**: CUAs는 부분 가시성 마르코프 결정 과정(POMDP)으로 모델링되며, 상태 공간, 관찰 공간 그리고 행동 공간으로 구성됩니다. 본 연구는 다수의 기초 모델과 정책을 사용하여 후보 솔루션 경로의 수를 스케일링하고 최적의 솔루션 선택을 위한 효과적인 방법을 제안합니다. 이는 기존의 단계별 BoN 방법과는 달리, 여러 기본 에이전트에 의해 생성된 후보 경로 중에서 최상의 경로를 선택하는 접근 방식을 취합니다.

- **Performance Highlights**: bBoN 메서드는 OSWorld 벤치마크에서 69.9%의 성공률로 새로운 state of the art(SoTA)를 달성하였습니다. 이는 이전 최상의 59.9%를 크게 초과하였으며, 인간 수준의 성능에 가까운 72%를 근접하게 합니다. 또한, WindowsAgentArena 및 AndroidWorld에 대한 강력한 제로샷 일반화 결과를 보여주어, bBoN의 성능을 더욱 입증했습니다.



### ExGRPO: Learning to Reason from Experienc (https://arxiv.org/abs/2510.02245)
- **What's New**: 본 논문에서는 강화학습을 통한 검증 가능한 보상(Reinforcement Learning from Verifiable Rewards, RLVR)의 새로운 패러다임을 제안하고, 이에 대한 효율적인 경험 관리 방법론인 Experiential Group Relative Policy Optimization (ExGRPO)을 제시합니다. ExGRPO는 경험의 가치를 평가하기 위해 롤아웃의 정확성과 엔트로피를 활용하며, 모델이 과거의 경험을 효과적으로 재사용할 수 있도록 도와줍니다. 이 방법은 대규모 언어 모델의 추론 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: ExGRPO는 롤아웃 중 생성된 경험을 효율적으로 관리하기 위해 경험의 정확도에 따라 버킷으로 조직하고, 가장 유용한 버킷에서 경험을 우선적으로 샘플링하는 전략을 채택합니다. 또한, 이 방법은 샘플 효율성을 개선하고 훈련 안정성을 증대시키기 위해 신선한 탐색과 과거 경험의 재사용을 균형있게 조절하는 혼합 정책 최적화(mixed-policy optimization) 목표를 사용합니다. 향후 연구의 중요성을 강조하고 있는 이 방법론은 경험 관리가 RLVR에서 필수적인 요소임을 보여줍니다.

- **Performance Highlights**: ExGRPO는 1.5B에서 8B 매개변수를 가진 다섯 가지 백본 모델에서 RLVR 성능을 평균적으로 +3.5/7.6 포인트 이상 개선하며, 수학적 및 일반 벤치마크에서 일관된 성과를 나타냅니다. 특히, ExGRPO는 기존의 온-정책 방식의 최적화가 실패하는 모델에서도 훈련을 안정화시키는 데 성공하여, 전체적인 학습 성과를 높였습니다. 이러한 연구 결과는 과거의 경험을 효과적으로 활용하는 것이 대규모 언어 모델의 성능을 향상시키는 데 중요한 역할을 한다는 것을 강조합니다.



### Study on LLMs for Promptagator-Style Dense Retriever Training (https://arxiv.org/abs/2510.02241)
Comments:
          CIKM 2025 short research paper

- **What's New**: Promptodile는 소규모의 오픈소스 대규모 언어 모델(Large Language Models, LLMs)을 활용한 효과적인 쿼리 생성 방법을 제시합니다. 기존의 Promptagator는 대형 LLM에 의존했으나, 이제 3B 매개변수의 모델도 효과적으로 사용될 수 있다고 연구 결과가 입증되었습니다. 이 연구는 비용 문제나 데이터 프라이버시로 인해 대형 모델을 사용할 수 없는 상황에 대한 실용적인 대안을 제공합니다.

- **Technical Details**: Promptodile은 총 10개의 오픈소스 LLM을 활용하여 1B에서 14B 매개변수의 다양한 모델을 평가하고, 7개의 저자원 BEIR 데이터셋에서 실험을 수행했습니다. 이를 통해 Promptagator와 유사한 성능을 보이며, 작은 LLM도 큰 모델만큼 효과적이라는 것을 확인했습니다. 이 연구는 주로 프롬프트의 몇 가지 예시를 바탕으로 하여, 적은 수의 주석이 있는 쿼리-문서 쌍을 대량의 합성 훈련 데이터로 확대하는 과정이 포함됩니다.

- **Performance Highlights**: Promptodile은 모든 평가된 LLM에서 일반적으로 Promptagator와 경쟁력을 보이며, 최근의 접근 가능한 LLM이 효과적인 쿼리 생성기로 활용될 수 있음을 입증했습니다. 연구 결과, 3B 매개변수의 작은 모델이 7B에서 14B 매개변수의 큰 모델과 같은 성능을 발휘하여, 비용이 많이 드는 LLM을 사용할 필요가 없음을 보여주었습니다.



### The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models (https://arxiv.org/abs/2510.02230)
Comments:
          23 pages, 15 figures

- **What's New**: 본 논문은 RLVR(Reinforcement Learning with Verifiable Rewards) 훈련의 독특한 문제, 즉 추론 경계의 축소(shrinkage)를 탐구합니다. 연구 결과, RLVR이 특정 훈련 문제를 해결하는 과정에서 다른 문제의 올바른 해결책을 생성할 가능성을 감소시키는 부정적인 간섭(negative interference) 현상이 발생한다는 사실을 밝혀냈습니다. 또한, RLVR이 기본 모델에서 높은 확률로 해결되는 문제에 대해서만 강한 강화를 제공하고, 초기 확률이 낮은 문제에 대해서는 소홀히 하는 승자독식(winner-take-all) 현상도 발견했습니다.

- **Technical Details**: 이 논문은 RLVR의 학습 역학을 심도있게 분석하며, 각 문제는 독특하고 알려지지 않은 보상 함수(reward function)를 가지는 개별적인 마코프 의사결정 과정(MDP)을 유도함을 설명합니다. 특정 문제를 해결하는 학습이 다른 문제를 해결하는 능력에 악영향을 미칠 수 있다는 점에 주목하며, 이는 RLVR에서의 부정적인 간섭 효과의 주된 원인으로 지목됩니다. RLVR의 전형적인 목표가 현 정책(on-policy sampling) 학습을 기반으로 하여 제한된 해결 전략으로 수렴하게 한다는 점도 강조합니다.

- **Performance Highlights**: SELF(Selective Examples with Low-likelihood and Forward-KL)라는 새롭고 효과적인 데이터 큐레이션 알고리즘을 제안하여, 올바른 답변을 도출할 가능성이 낮은 문제에 집중합니다. 이 알고리즘을 통해 RLVR의 Pass@k 성능이 크게 향상되는 것으로 나타났습니다. 실험적 결과는 SELF가 샘플 효율성을 높이고 RLVR에서의 문제 범위 축소를 효과적으로 완화함을 증명합니다.



### StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets? (https://arxiv.org/abs/2510.02209)
- **What's New**: 이 논문은 StockBench라는 새로운 벤치마크를 소개하여 대형 언어 모델(LLMs)이 주식 거래 환경에서의 수익성 및 리스크 관리 능력을 평가할 수 있는 기준을 제공한다. 기존의 재무 벤치마크는 주로 정적인 지식 테스트에 초점을 맞췄으나, StockBench는 동적이고 반복적인 매매 결정을 반영하는 평가 방식을 채택하였다. 이를 통해 LLM 에이전트의 주식 거래 전략의 효과성을 더욱 정확하게 검증할 수 있게 된다.

- **Technical Details**: StockBench는 다음 세 가지 원칙을 준수하여 설계되었다: (1) 현실적인 시장 상호작용, (2) 지속적인 의사결정, (3) 데이터 오염 방지. 에이전트는 매일의 시장 신호(가격, 기업의 기초 정보, 뉴스 등)를 기반으로 연속적인 거래 결정을 내리며, 평가 메트릭은 누적 수익(cumulative return), 최대 손실(maximum drawdown), Sortino 비율을 사용하여 이루어진다. 평가 대상은 다양한 LLM 모델(예: GPT-5, Claude-4, Qwen3 등)으로 구성되었다.

- **Performance Highlights**: 실험 결과, 대부분의 LLM 에이전트는 단순 매수 및 보유 전략을 초과하여 성과를 내지 못한 반면, 몇몇 모델은 상대적으로 더 높은 수익을 기록하고 리스크를 효과적으로 관리할 수 있는 가능성을 보여주었다. 이는 LLM이 정적인 재무 지식 과제를 잘 수행하더라도 실제 거래 전략에는 한계를 갖고 있음을 나타낸다. 이 연구는 LLM 기반 재무 에이전트 개발에 있어 도전과 기회를 동시에 보여준다.



### A Rigorous Benchmark with Multidimensional Evaluation for Deep Research Agents: From Answers to Reports (https://arxiv.org/abs/2510.02190)
- **What's New**: 이번 연구에서는 폐쇄형 언어 모델에서 외부 인지 및 정보 통합이 가능한 상호 연결된 에이전트 시스템으로의 패러다임 전환을 다룹니다. Deep Research Agents (DRAs)의 기능이 태스크 분해(task decomposition), 교차 출처 검색(cross-source retrieval), 다단계 추론(multi-stage reasoning), 구조화된 출력(structured output)을 통해 복잡한 작업에서 수행 능력을 크게 향상시킨다는 것을 보여줍니다. 이 논문은 DRAs와 보고서 스타일 응답을 위해 설계된 엄격한 벤치마크와 다차원 평가 프레임워크를 도입하여, 기존 벤치마크의 한계를 극복하고 DRAs의 전반적인 성능을 체계적으로 평가할 수 있는 기반을 마련합니다.

- **Technical Details**: 연구에서는 10개의 광범위한 주제 영역에 걸쳐 214개의 전문가가 선정한 도전적인 쿼리를 포함한 벤치마크를 소개합니다. 각 쿼리는 퍼포먼스 평가를 지원하기 위해 수동으로 구성된 참조 번들(reference bundles)을 동반하고 있으며, 세분화된 평가 기준을 처리하는 다차원 평가 프레임워크를 기반으로 하고 있습니다. 이 프레임워크는 DRAs가 생성한 장기 보고서의 평가를 포괄적으로 수행할 수 있도록 하며, 의미적 품질(semantic quality), 주제적 초점(topical focus), 검색 신뢰성(retrieval trustworthiness)과 같은 통합 점수 메트릭을 포함하고 있습니다.

- **Performance Highlights**: 대규모 실험을 통해 DRAs가 웹 검색 도구에 의해 강화된 추론 모델보다 전반적인 작업 수행과 보고서 생성 품질에서 지속적으로 우수한 성능을 나타냈습니다. 그러나 연구 결과는 아키텍처와 행동 메커니즘에서 여전히 개선이 필요하다는 점을 강조합니다. 이 연구는 DRA 시스템의 능력 평가, 구조 개선 및 패러다임 발전을 위한 확고한 기초를 제공합니다.



### Do AI Models Perform Human-like Abstract Reasoning Across Modalities? (https://arxiv.org/abs/2510.02125)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구는 OpenAI의 o3-preview 모델이 ARC-AGI 벤치마크에서 인간의 정확도를 초과했지만, 이러한 모델들이 과연 문제 제작자들이 의도한 추상 개념을 인지하고 추론하는지에 대한 의문을 제기합니다. ConceptARC에서 모델의 추상화 능력을 조사하며, 다양한 입력 방식과 외부 파이썬 도구의 사용 여부에 따라 모델을 평가합니다. 정확성 외에도 모델이 생성한 자연어 규칙을 세밀하게 평가함으로써 모델이 과제를 해결하는 방식에 대한 깊이 있는 분석을 수행합니다.

- **Technical Details**: 이 연구에서는 OpenAI의 o3와 o4-mini, Google의 Gemini 2.5 Pro, Anthropic의 Claude Sonnet 4와 같은 네 가지 멀티모달(reasoning) 모델을 평가합니다. 각 모델은 간단한 추상 개념을 활용한 480개의 ConceptARC 작업을 해결하기 위해 JSON 객체를 생성하도록 요청받으며, 이를 통해 두 가지 평가를 수행합니다: 그리드 출력 정확성과 모델 생성 규칙이 과제가 의도한 추상화를 얼마나 잘 포착하는지입니다. 실험에서 텍스트와 시각 모달리티 모두에서 모델의 추상적 추론 능력을 조사하고, 추론 노력과 외부 도구 접근성이 어떻게 영향을 미치는지를 분석합니다.

- **Performance Highlights**: 결과적으로, 일부 텍스트 기반 모델은 인간의 정확성과 유사한 성과를 보였으나, 최상위 모델의 규칙은 종종 표면적인 '지름길'에 기반하여 의도한 추상화를 잘 반영하지 못했습니다. 시각 모달리티에서는 AI 모델의 정확성이 급격히 하락했지만, 규칙 수준 분석에서는 여전히 상당한 수준의 의도된 추상화를 포착할 수 있다는 것이 드러났습니다. 이는 모델들이 인간의 추상적 추론 능력에 비해 여전히 부족하다는 점을 강조하며, 정확성만을 기준으로 평가하는 것이 두 가지 모달리티에서의 추론 능력을 과대 또는 과소 평가할 위험이 있음을 보여줍니다.



### Constrained Adaptive Rejection Sampling (https://arxiv.org/abs/2510.01902)
- **What's New**: 이 논문에서는 CARS(Constrained Adaptive Rejection Sampling)를 소개합니다. CARS는 기존의 Rejection Sampling(RS)과 Greedy Constrained Decoding(GCD)의 단점을 해결하여 성능을 개선한 새로운 접근법입니다. 특히, CARS는 샘플 파라미터를 체계적으로 수정하여 유효한 샘플을 더 효율적으로 생성할 수 있도록 합니다.

- **Technical Details**: CARS는 Adaptive Rejection Sampling(ARS)에서 발전된 방식으로, invalid한 프리픽스(prefix)를 trie에 기록해 나중에 샘플을 생성할 때 해당 probability mass를 차감합니다. 이렇게 하면 샘플이 계속해서 무효화되는 것을 방지하고, 점진적으로 acceptance rate가 개선됩니다. 이 알고리즘의 핵심은 조건부 분포를 유지하며, 연결된 prefix를 검증하는 것입니다.

- **Performance Highlights**: 다양한 실험에서 CARS는 기존의 constrained sampling 방법들과 비교하여 높은 acceptance rate와 다양성을 보여주었습니다. 또한, CARS는 유효한 샘플을 생성하는 데 있어 평균적으로 더 낮은 비용을 요구하며, 성능 면에서 새로운 최첨단 기준을 설정했습니다. 프로그램 퍼징(program fuzzing) 및 분자 생성(molecular generation) 분야에서 특히 강력한 효율성을 기록했습니다.



### Plan Then Action:High-Level Planning Guidance Reinforcement Learning for LLM Reasoning (https://arxiv.org/abs/2510.01833)
Comments:
          19 pages and 5 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 추론 능력을 향상시키기 위해 새로운 두 단계의 프레임워크인 PTA-GRPO를 제안합니다. 이 모델은 Chain-of-Thought (CoT) 추론을 고급 계획(guidance)과 결합하여 더 나은 결과를 도출합니다. 첫 번째 단계에서는 고급 LLM을 활용해 정교한 계획을 수립하고, 두 번째 단계에서는 강화 학습(Reinforcement Learning, RL)을 통해 최종 출력과 높은 수준의 안내 품질을 동시에 최적화합니다.

- **Technical Details**: PTA-GRPO는 고차원 계획과 추론을 촉진하기 위해 기획된 새로운 두 단계의 프레임워크입니다. 첫 번째 단계에서는 CoT를 간결한 고수준 안내로 요약하고 이를 사용하여 감독 세분화(Supervised Fine-Tuning, SFT)를 진행합니다. 두 번째 단계에서는 GRPO 알고리즘을 기반으로 한 계획 가이드 인식 강화 학습 방법이 적용되어, 최종 출력의 정확성과 고급 안내의 품질을 모두 평가하여 강화됩니다.

- **Performance Highlights**: 메시지를 통해 우리는 다양한 기초 모델(Qwen2.5-7B-Instruct, Qwen3-8B 등)에서 여러 수학적 추론 벤치마크(MATH, AIME2024 등)에 대해 PTA-GRPO를 사용한 실험을 진행했습니다. 실험 결과, PTA-GRPO는 모든 모델과 작업에서 안정적이고 상당한 성능 향상을 이루어냈으며, 이 모델의 효과성과 일반화 가능성을 확인했습니다.



### Sparse Query Attention (SQA): A Computationally Efficient Attention Mechanism with Query Heads Reduction (https://arxiv.org/abs/2510.01817)
Comments:
          18 pages, 6 figures, small-scale experiments

- **What's New**: 이번 논문에서는 Sparse Query Attention (SQA)라는 새로운 attention 아키텍처를 소개합니다. SQA는 Query heads의 수를 줄여 attention 메커니즘의 계산 복잡성을 감소시킴으로써 플로팅 포인트 연산(Floating-Point Operations, FLOPs)의 수를 줄입니다. 이 방법은 메모리 대역폭의 병목 현상을 해결하는 기존의 접근 방식과는 다른 최적화 경로를 추구합니다.

- **Technical Details**: SQA의 수학적 공식화 및 변형된 아키텍처를 제시하며, 출처는 Sparse Query Attention의 기본 이론에 기반합니다. SQA는 Matrix multiplication인 QK^T의 복잡성을 O(N^2⋅d_model + N⋅d_model^2)에서 쿼리 헤드 수의 감소 비율에 비례하여 직접 줄입니다. 이로 인해 전체 FLOPs가 감소하고 긴 시퀀스의 경우 성능 개선을 이끌어냅니다.

- **Performance Highlights**: 32k-200k 토큰의 긴 시퀀스에서의 실험 결과 SQA는 계산 중심 시나리오에서 최대 3배까지 처리량을 개선할 수 있음을 보여주었습니다. 예비 실험에서 모델 품질에 미치는 영향은 최소한으로 유지되었으며, 이 아키텍처는 더 효율적이고 확장 가능한 모델 개발에 강력한 도구로서의 가능성을 제시합니다.



### Improving AGI Evaluation: A Data Science Perspectiv (https://arxiv.org/abs/2510.01687)
- **What's New**: 이번 연구에서는 AGI(Artificial General Intelligence) 평가 방법론에 대하여 기존의 직관에 기반한 방식의 한계를 지적하고, 대안으로 강력한 과제 수행 능력을 평가하는 접근 방식을 제안합니다. AGI 평가에서의 주요 목표는 실제로 인간의 작업을 수행할 수 있는 시스템을 개발하기 위해 필요한 평가 메커니즘에 대한 새로운 시각을 제공하는 것입니다. 저자들은 데이터 과학의 실제적인 사례를 바탕으로 AGI 평가를 위한 강력한 방법론을 제시합니다.

- **Technical Details**: AGI의 평가에서는 정의의 모호성이 문제로 지적되며, 특히 기존의 성과 지표들은 인간의 지능을 우회하려는 방식으로 조작될 수 있는 경향이 있습니다. 이 연구는 AGI가 복잡하고 다양한 환경에서 자율적으로 목표를 달성하는 능력을 강조하며, 새로운 평가 방법론이 데이터 세트의 편향과 오염을 줄이고, 메모리 효과를 방지하는데 중점을 두어야 한다고 주장합니다. 이를 통해 AGI 시스템이 학습한 데이터 밖에서도 신뢰할 수 있는 성능을 보장할 수 있도록 해야 합니다.

- **Performance Highlights**: AGI 시스템의 성능은 기존의 정량적 지표 외에도, 새로운 접근 방식으로 제안된 시뮬레이션 환경에서의 성능으로 평가될 수 있습니다. 이러한 접근법은 시스템이 실제 인간의 학습과 유사한 방식으로 causal principles를 학습할 수 있는지를 중심으로 진행됩니다. AGI 평가의 중심에는 시스템이 실제 업무를 수행할 수 있는 능력과 자율성이 포함되어, 이러한 점에서 새로운 평가 방법이 강력한 검증 도구로 자리 잡을 수 있을 것으로 기대됩니다.



### Just Do It!? Computer-Use Agents Exhibit Blind Goal-Directedness (https://arxiv.org/abs/2510.01670)
- **What's New**: 이 논문은 컴퓨터 사용 에이전트(CUAs)의 Blind Goal-Directedness (BGD) 현상을 분석하며, 이는 에이전트가 목표 관점에서만 행동하여 잠재적으로 해로운 결과를 초래할 수 있음을 보여줍니다. BGD는 세 가지 주요 패턴으로 구분되며, 이를 체계적으로 평가하기 위해 90개의 작업으로 구성된 BLIND-ACT 벤치마크를 개발했습니다.

- **Technical Details**: BLIND-ACT는 OSWorld 위에 구축된 실제적이고 동적인 데스크톱 환경에서의 실행을 지원하며, 다양한 어플리케이션과 시스템 기능에서 BGD 행동이 발생할 수 있도록 설계되었습니다. 벤치마크는 90개의 작업으로 구성되어 있으며, 각 작업은 BGD의 세 가지 패턴을 포괄합니다. LLM 기반의 판단자를 사용해 에이전트가 BGD 행동을 보이는지와 비효율적인 행동을 실행하는지를 평가합니다.

- **Performance Highlights**: BLIND-ACT를 활용하여 Claude Sonnet, Opus 4 및 GPT-5와 같은 9개의 최신 모델을 평가한 결과, 평균 80.8%의 BGD 비율을 관찰했습니다. 작은 모델들은 과도하게 안전한 것처럼 보이나 이는 제한된 능력 때문이며, 결과적으로 안전과 능력 간의 패러독스가 강화됩니다. 또한, 프롬프트 기반의 개입이 BGD 수준을 낮출 수 있지만 여전히 상당한 위험이 남아 있음을 보여줍니다.



### Position: Privacy Is Not Just Memorization! (https://arxiv.org/abs/2510.01645)
Comments:
          27 pages, 6 figures, 2 tables

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLM) 시스템의 개인 정보 보호 위험에 대한 쟁점을 다루고 있습니다. 주목할 점은 기존의 논의가 훈련 데이터의 단순한 암기(memoration)에만 집중되어 있는 반면, 실제로는 더 즉각적이고 확장 가능한 다양한 개인 정보 보호 위협이 존재한다는 것입니다. 저자들은 LLM 시스템의 전체 주기로부터 발생하는 개인 정보 보호 위험을 포괄적으로 분류하여 설명하며, 현재의 개인 정보 보호 프레임워크가 이 위협들에 제대로 대응하지 못하고 있음을 강조합니다.

- **Technical Details**: 논문에서는 LLM 생태계에서 유출되는 데이터의 세 가지 유형(사용자 상호 작용 데이터, 시스템 검색 데이터 및 공개 데이터)을 제시합니다. 이러한 데이터 유형은 서로 작용하여 훈련 데이터 유출을 넘어서는 다섯 가지 특정한 개인 정보 유출 사건을 생성하며, 각 사건 유형이 다양한 위협을 제시합니다. 예를 들어, LLM은 무의식적으로 사용자 입력으로부터 민감한 정보(attribute)를 추론하고, 이를 통해 개인정보를 침해할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 연구에 따르면, AI/ML 개인 정보 보호 분야의 논문 중 92%가 훈련 데이터 메모리화(memoration) 및 직접적인 채팅 유출 방지에 초점을 맞추고 있으며, 다른 유형의 사건은 연구의 8% 미만을 차지합니다. 저자들은 데이터 최소화(local data minimization), 하이브리드 아키텍처 및 프라이버시 중심의 후속 훈련(post-training)과 같은 기술적 개입이 필요하다고 강조합니다. 이는 개별 사용자에게 힘을 실어줄 수 있는 사회기술적 접근(sociotechnical approaches)과 불균형을 해결할 수 있는 정책 개선을 필요로 합니다.



### Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls (https://arxiv.org/abs/2510.01631)
Comments:
          Published as a Main Conference paper at EMNLP 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 사전 훈련 과정에서 합성 데이터(synthetic data)의 역할과 효과를 체계적으로 조사한 대규모 연구 결과를 제시합니다. 1000개 이상의 LLM 변형에 대해 200B tokens 규모의 데이터셋을 사용하여 훈련을 진행하였고, 각 데이터 유형과 특성에 따른 사전 훈련 성능을 비교하였습니다. 합성 데이터의 특정 비율이 사전 훈련 속도를 크게 향상시킬 수 있다는 점이 새롭게 밝혀졌습니다.

- **Technical Details**: 연구 결과에 따르면, 1/3의 재구성된 합성 데이터와 2/3의 자연 웹 텍스트가 혼합된 데이터에서 훈련을 진행한 경우, 같은 검증 손실에 도달하는 데 5-10배의 속도 향상이 나타났습니다. 그러나 사전 훈련을 위해 합성 데이터 유형과 훈련 데이터의 조합 비율이 특히 중요하며, 최적의 비율은 일반적으로 30%로 나타났습니다. 이 연구에서는 또한 대형 생성기 모델이 반드시 더 나은 합성 데이터를 생성하지 않는다는 사실도 강조되었습니다.

- **Performance Highlights**: 재구성된 합성 데이터로만 사전 훈련을 진행했을 경우 자연 웹 텍스트로 사전 훈련을 진행했을 때보다 속도가 더 빠르지 않았습니다. 그러나 텍스트북 스타일의 합성 데이터는 특히 작은 데이터 예산에서 많은 다운스트림 도메인에서 현저히 높은 손실을 발생시키는 경향이 있음을 발견했습니다. 데이터 혼합물이 가진 상호작용의 복잡성은 단순한 유사성 이상의 요소에 의존함을 보여주며, 이는 보다 복잡한 다양성-품질 간의 균형을 시사합니다.



### Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead (https://arxiv.org/abs/2510.01624)
Comments:
          Preprint. Under Review

- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 위해 사후 훈련 후 추론 단계를 재조명합니다. 저자들은 감독 세분화(Supervised Fine-Tuning, SFT)와 강화 학습(Reinforcement Learning, RL)이 독립적으로 이루어지는 현재의 관행에 도전하며, SFT 점수가 RL 후 성능 개선으로 이어지지 않는 사례를 제시합니다. 그들은 SFT 점수가 단순한 데이터에 편향될 수 있음을 발견하고, SFT 성능을 개선한 모델이 원래 모델에 비해 RL 결과에서 나쁜 성과를 보일 수 있는 사례를 보고합니다.

- **Technical Details**: 이 연구는 SFT와 RLVR(Verifiable Rewards)을 통해 12B 파라미터의 수백 개 모델을 훈련하고, 7개의 수학 벤치마크에서 $>1M GPU 시간을 투자해 광범위한 평가를 수행했습니다. 대안적 지표로 포괄성과 Pass@large k 성능을 검토하여 RL 결과 예측에 유용한 신뢰할 수 있는 프록시를 식별하였습니다. 또한, 평가 도구를 오픈 소스화할 예정입니다.

- **Performance Highlights**: 저자들은 SFT 훈련의 질이 RL 성과에 대한 강력한 예측 인자가 아닐 수 있음을 강조하며, 특정 설정에서 SFT와 RL을 Separately 최적화하는 것의 문제점을 지적합니다. 일반적으로 SFT에서 높은 성적을 거둔 모델이 RL 결과에서 우수한 성과를 보이지 않는다는 발견은 중요한 함의를 지니의습니다. 실험 결과, SFT에서 짧은 예시만 사용하는 것이 SFT 성능을 높일 수 있지만, RL 후에는 우수한 성과를 보이지 않을 수 있음을 보여줍니다.



### LLM4Rec: Large Language Models for Multimodal Generative Recommendation with Causal Debiasing (https://arxiv.org/abs/2510.01622)
- **What's New**: 이 논문은 현대의 생성형 추천 시스템에서 다중 모달 데이터 처리, 알고리즘적 편향 제거 및 투명한 의사결정 프로세스 제공의 문제를 해결하기 위해 다섯 가지 주요 혁신을 제안합니다. 새로운 프레임워크는 멀티모달 융합 아키텍처, 검색 보강 생성 메커니즘, 인과 추론 기반의 편향 제거, 설명 가능한 추천 생성, 실시간 적응형 학습 능력을 포함하여 대규모 언어 모델을 기본으로 합니다. 이를 통해 다양한 콘텐츠 유형을 처리하고 사용자 선호에 따라 적응 가능한 추천 시스템을 구축할 수 있습니다.

- **Technical Details**: 제안된 Enhanced GenRec 프레임워크는 다중 모달 융합 구조를 통해 텍스트, 범주 및 수치 데이터를 통합하여 사용자와 아이템의 풍부한 표현을 생성합니다. 또한, 검색 보강 생성 메커니즘은 데이터셋 내의 상황별 정보를 활용하여 추천의 정확성과 범위를 향상시킵니다. 편향 제거 기술은 추천 결과의 체계적인 편향을 탐지하고 완화하기 위해 인과 추론을 활용하고, 설명 가능한 추천 모듈은 추천 결정에 대한 자연어 설명을 생성하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(MovieLens-25M, Amazon-Electronics, Yelp-2023)에 대한 광범위한 실험 결과, 제안된 프레임워크는 기존 접근 방식에 비해 추천 정확도, 공정성 및 다양성이 일관되게 개선되었습니다. NDCG@10에서 최대 2.3%의 개선과 다양성 메트릭에서 1.4%의 향상을 이뤄내면서도 최적화된 추론 전략을 통해 계산 효율성을 유지합니다.



### PychoBench: Evaluating the Psychology Intelligence of Large Language Models (https://arxiv.org/abs/2510.01611)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)이 심리 상담에서 효과적으로 활용될 수 있는지를 분석합니다. 연구팀은 PsychoBench라는 벤치마크를 도입하여 LLM이 미국 국가 상담사 인증 시험(NCE)을 통과할 수 있는지를 평가합니다. PsychoBench는 2,252개의 심리학 관련 단일 선택 질문으로 구성되어 있으며, 이는 LLMS의 상담사 역할 수행 가능성을 검증하는 데 도움을 줍니다.

- **Technical Details**: PsychoBench는 GPT 기반 방법을 이용하여 심리 상담 시험 질문을 재구성하고 정제하였습니다. 각 질문은 심리학 전문가들에 의해 정확성과 일관성을 보장하기 위해 검토되었습니다. 이 데이터셋은 상담 방법, 이상 심리학, 발달 심리학 및 윤리적 고려 사항을 포함한 다양한 하위 분야를 포괄하며, LLM의 심리 상담 작업 수행 능력을 체계적으로 평가하기 위해 사용됩니다.

- **Performance Highlights**: 최신 LLM 모델인 GPT-4o, Llama3.3-70B, 그리고 Gemma3-27B가 시험 기준을 초과하는 성과를 보였으나, 소형 오픈 소스 모델들은 여전히 기준에 미치지 못함을 보여줍니다. 이러한 결과는 현재 심리 상담 표준을 충족할 수 있는 것은 최첨단 LLM만임을 시사하며, 심리학 중심의 LLM 개발에서의 약속과 도전 과제를 강조합니다.



### Bridging Collaborative Filtering and Large Language Models with Dynamic Alignment, Multimodal Fusion and Evidence-grounded Explanations (https://arxiv.org/abs/2510.01606)
- **What's New**: 이 논문은 DynMM-Explain-LLMRec라는 새로운 추천 시스템 프레임워크를 소개합니다. 이 프레임워크는 사용자 상호작용 데이터를 지속적으로 학습할 수 있는 경량 모듈을 통해 실시간으로 적응하는 혁신적인 온라인 적응 메커니즘을 채택합니다. 또한 시각 및 오디오 콘텐츠와 협동 신호를 통합한 통합된 표현 방식을 설계하였습니다. 마지막으로, 사용자에게 명확한 이유를 제공할 수 있는 설명 시스템을 구현하였습니다.

- **Technical Details**: DynMM-Explain-LLMRec는 고정된 기본 정렬기를 기반으로 하여 동적 업데이트와 다중 모달 통합을 촉진합니다. 각 항목에 대해 최근 상호작용을 요약하는 동적 잠재 표현을 생성하며, 이를 통해 사용자 동적 특성을 잘 반영합니다. 또한 각 모달리티(visual, audio 등)를 통합하여 협동 패턴과 아이템 특성을 잇는 고유한 구조를 제공합니다. 이러한 구조는 다양한 모달리티가 결여될 경우에도 강력한 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: Comprehensive evaluation은 시간적 적응, 냉시작(Cold-start) 시나리오, 계산 효율성 및 설명 품질을 아우릅니다. 본 논문에서 제안한 방법은 기존 모델의 효율성을 유지하면서도 최소한의 계산 오버헤드로 실제 시스템에서 활용할 수 있음을 보여줍니다. 이에 따라 동적인 사용자 선호 변화를 따라잡고 명확한 해석과 높은 신뢰성을 제공하는 추천 시스템으로서의 가능성을 보여줍니다.



### Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression (https://arxiv.org/abs/2510.01581)
Comments:
          Code: this https URL

- **What's New**: TRAAC (Think Right with Adaptive, Attentive Compression)라는 새로운 방법이 소개됩니다. 이는 모델이 다양한 난이도에 따라 인지하는 단계의 길이를 유동적으로 조절하여 이른바 'under-adaptivity' 문제를 해결하려고 합니다. 이 방법은 온라인 강화 학습(RL)을 활용하며, 모델이 불필요한 추론 단계를 제거하고 효율적으로 중요한 단계를 식별합니다.

- **Technical Details**: TRAAC는 Group Reward Policy Optimization (GRPO) 기반의 방법으로, Proximal Policy Optimization에서 비평가를 제거하고 샘플 응답 그룹에서 기준을 추정합니다. 모델은 각 추론 단계의 주의 점수를 기반으로 중요하지 않은 토큰들을 식별하고 압축합니다. 이 과정에서 난이도에 따라 압축 수준을 조절하며, 어려운 문제에 대해서는 압축을 줄이고, 쉬운 문제에 대해서는 압축을 늘립니다.

- **Performance Highlights**: TRAAC는 다양한 과제에서 평균적으로 8.4%의 정확도 개선과 36.8%의 추론 길이 감소를 달성했습니다. 비수학 데이터셋에서도 강력한 일반화 능력을 보여주었으며, OOD(Out-Of-Distribution) 작업에서 평균 3%의 성능 개선과 40%의 응답 길이 감소를 기록했습니다. 이러한 결과는 TRAAC가 다양한 난이도 작업에서 성능을 개선하고, 불필요한 계산을 줄일 수 있음을 보여줍니다.



### Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomp (https://arxiv.org/abs/2510.01574)
Comments:
          Accepted to the Proceedings of the ACM SIGIR Asia Pacific Conference on Information Retrieval (SIGIR-AP 2025), December 7-10, 2025, Xi'an, China

- **What's New**: 이번 연구에서는 실시간 신경 쿼리 자동 완성 시스템에서 프레젠테이션 편향(presentation bias)을 완화하기 위한 데이터 중심 접근 방식을 제안합니다. 본 방법론은 사용자의 전체 쿼리를 바탕으로 합성 접두사를 생성하여 훈련 데이터를 다양하고 비편향적인 예제들로 풍부하게 하여 학습합니다. 이러한 접근은 기존의 사용자의 상호작용 데이터에서 발생하는 편향을 줄이는데 도움을 줍니다.

- **Technical Details**: 신경 순위 모델은 엄격한 지연(latency) 제약 하에서 실시간 배포를 위해 최적화되었습니다. 이는 쿼리 인기도(query popularity), 계절성(seasonality), 유사도 점수(fuzzy match scores) 등을 포함한 다양한 특성을 통합합니다. 우리는 쿼리 자동 완성 구조를 이용하여 O(n^2)에서 O(n)으로 계산 복잡성을 줄인 리스트와이즈 손실(listwise loss)의 간소화 버전을 도입하였습니다.

- **Performance Highlights**: 대규모 전자상거래 환경에 배포된 이 시스템은 사용자 참여를 통계적으로 유의미하게 개선하는 결과를 보여주었습니다. A/B 테스트 결과, 우리의 모델은 기존의 선형 바닥선(linear baseline) 보다 MRR(Mean Reciprocal Rank)을 1% 이상 향상시켰습니다. 이러한 결과는 균형 잡힌 데이터와 최적화된 손실 함수를 통해 훈련된 신경 LTR 모델의 효과성을 입증합니다.



### InvThink: Towards AI Safety via Inverse Reasoning (https://arxiv.org/abs/2510.01569)
- **What's New**: InvThink는 대형 언어 모델(LLMs)에게 실패 모드를 미리 고려하는 역 추론(inverse reasoning) 능력을 부여하는 새로운 접근법입니다. 기존의 안전 정렬 방법이 안전한 응답을 직접 최적화하는 것과 달리, InvThink는 모델이 잠재적 해를 나열하고 그 결과를 분석한 후 안전하게 응답을 생성하도록 합니다. 이를 통해 안전성 향상이 모델 크기와 비례하여 강화되는 것을 보여줍니다.

- **Technical Details**: InvThink의 핵심은 모델이 해를 수치화하고 그에 대한 분석을 수행한 후, 피해를 피하는 방향으로 응답을 생성하도록 하는 것입니다. 이 과정은 역 추론 프레임워크로서 구조화된 추론 과정을 통해 이루어지며, 기존의 단순한 출력 맵핑 방식을 넘어서 모델의 철저한 위험 분석을 가능케 합니다. 또한, 기존의 방법과 달리 사고를 포함한 직관적 접근 방식을 통해 구체적이고도 포괄적인 위험 예측을 실현합니다.

- **Performance Highlights**: InvThink는 의료, 금융, 법률 등 고위험 분야에서 특히 뛰어난 성능을 보이며, 기존 방법에 비해 최대 15.7%의 유해 응답 감소를 달성했습니다. 또한, InvThink는 일반 추론 능력을 보존하면서도 안전성을 개선하여 안전 세금(safety tax) 문제를 완화합니다. 이는 InvThink가 AI 안전성을 향상시킬 수 있는 확장 가능하고 일반화된 경로를 제시한다는 것을 의미합니다.



### Information Seeking for Robust Decision Making under Partial Observability (https://arxiv.org/abs/2510.01531)
Comments:
          The project page is available at this https URL

- **What's New**: 이번 연구에서는 정보 탐색(Information Seeking)을 LLM(대형 언어 모델) 의사결정 프레임워크인 InfoSeeker에 통합하여, 불확실한 환경에서의 내부 동향과 실제 환경을 조화롭게 맞추는 방법을 제안합니다. InfoSeeker는 의도가 있는 계획 수립을 통해 정보 수집을 유도하며, 이는 기존의 관측 불확실성을 다루는 모델들이 간과했던 부분입니다. 또한, InfoSeeker는 실제 환경 변화 감지 및 가설 테스트를 통해 효과적인 계획을 만들어냅니다.

- **Technical Details**: InfoSeeker는 부분적으로 관찰 가능한 환경에서 결정-making을 모사하고 따르는 POMDP(부분 관찰 마르코프 결정 과정) 모델을 기반으로 합니다. 모델은 상태 집합, 행동 집합, 관측 균형 등을 포함하여 에이전트가 확률적 결정을 내리는 과정에서 활용됩니다. 이 연구에서는 고유의 리워드 함수와 할인지수 등을 통해 정보 수집을 포함한 의사결정 프레임워크를 수학적으로 정립합니다.

- **Performance Highlights**: 정보 탐색을 적극적으로 통합한 InfoSeeker는 기존 방법들보다 74%의 성능 향상을 달성하였으며, 샘플 효율성을 해치지 않으면서도 더 나은 계획을 생성할 수 있게 되었습니다. 게다가 InfoSeeker는 다양한 LLM에 대해 일반화 가능하며, 기존 벤치마크에서도 우수한 성능을 보이는 것으로 나타났습니다. 이로 인해 두 가지 핵심 기여 - 정보 탐색과 계획 통합 -를 통한 에이전트의 강건한 행동을 강조할 수 있게 되었습니다.



### From Videos to Indexed Knowledge Graphs -- Framework to Marry Methods for Multimodal Content Analysis and Understanding (https://arxiv.org/abs/2510.01513)
- **What's New**: 이번 논문에서는 영상 데이터를 분석하기 위한 새로운 프레임워크를 제시합니다. 이는 다양한 pre-trained 모델을 통합하여 multi-modal 콘텐츠 분석을 위한 파이프라인을 효율적으로 프로토타입화할 수 있는 방법을 제공합니다. 특히, 지속적인 학습(continual learning)과 동적 지식 통합을 지원하는 지식 그래프(knowledge graph) 형식을 통해 영상의 정보를 쿼리할 수 있는 구조로 변환합니다.

- **Technical Details**: 연구 방법론은 세 가지 단계로 구성되어 있습니다. 첫 번째로, 최적화된 pre-trained 모델 조합을 실험하고 open-source 모델과 쉽게 결합하여 영상과 같은 시계열 다중 모드 데이터를 처리할 수 있는 프레임워크를 구축합니다. 두 번째 단계에서는 영상을 반구조적 데이터 형식인 'VideoKnowledgeBase'로 변환하는 파이프라인을 설계합니다. 마지막으로, 생성된 VideoKnowledgeBase를 쿼리 가능하고 확장 가능한 Video Knowledge Graphs로 변환하는 알고리즘을 설계합니다.

- **Performance Highlights**: 이 연구의 주요 성과는 다양한 pre-trained 모델을 통합하고 결합하여 multi-modal 콘텐츠를 이해하고 분석하기 위한 파이프라인을 구축하는 것입니다. 또한, 비디오 데이터베이스에서 정보를 쿼리할 수 있는 방법을 새롭게 제안하고, 새로운 도메인 지식을 추가할 수 있는 프로토타입 소프트웨어를 구현했습니다. 이러한 결과는 영상 분석, 이해 및 검색의 효율성을 크게 향상시킬 것으로 기대됩니다.



### Extracting O*NET Features from the NLx Corpus to Build Public Use Aggregate Labor Market Data (https://arxiv.org/abs/2510.01470)
Comments:
          85 pages

- **What's New**: 본 논문은 온라인 채용 공고의 데이터를 수집하고 분석하는 새로운 도구인 Job Ad Analysis Toolkit (JAAT)을 발표합니다. 이 도구는 O*NET 프레임워크를 기반으로 하여 구조화된 정보를 추출하도록 설계되었습니다. JAAT은 공개 소스 도구 모음으로, 이를 통해 데이터 접근성을 향상시키고 채용 데이터의 신뢰성과 정확성을 입증합니다.

- **Technical Details**: JAAT는 National Labor Exchange (NLx) Research Hub에서 제공하는 1억 5천 5백만 개 이상의 온라인 채용 광고에서 10억 개 이상의 데이터 포인트를 추출합니다. 채용 공고에는 O*NET 작업, 직업 코드, 도구 및 기술, 임금, 기술, 산업 등의 다양한 특징이 포함되어 있습니다. 논문은 2015년부터 2025년까지 월별 활성화된 일자리를 기준으로 직업, 주 및 산업 수준의 특성을 집계한 데이터셋 구축을 설명합니다.

- **Performance Highlights**: 논문에서는 JAAT의 외부 샘플 및 LLM-as-a-Judge 테스트에서의 신뢰성과 정확성을 보여줍니다. 이 도구는 교육 및 노동력 개발 분야에서의 연구 및 미래 활용 가능성을 잘 보여줍니다. JAAT을 통해 얻는 데이터는 지속적인 직업 시장 분석 및 전략 수립에 큰 도움이 될 것으로 기대됩니다.



### LSPO: Length-aware Dynamic Sampling for Policy Optimization in LLM Reasoning (https://arxiv.org/abs/2510.01459)
- **What's New**: 이 논문에서는 길이 인식 샘플링(Length-aware Sampling, LSPO)을 제안하여 훈련 데이터 샘플을 동적으로 선택하는 새로운 메타-강화학습 알고리즘을 소개합니다. LSPO는 각 질문의 평균 응답 길이에 따라 샘플을 필터링하여 훈련의 효율성을 높이고, 최종 모델의 정확도를 개선할 수 있는 방안을 제시하고 있습니다. 이를 통해 기존의 강화학습과 비교했을 때 모델 학습의 효과성을 한층 높일 수 있음을 보여줍니다.

- **Technical Details**: LSPO는 응답의 길이를 신호로 활용하여 데이터 필터링을 수행하며, 이는 RLVR(강화학습과 검증 가능한 보상)을 위한 동적 샘플링에서 중요한 기여로 작용합니다. 이 방식은 기존의 손실 함수 설계 방법과는 차별점을 두고 있으며, 응답 길이를 고려함으로써 최종 모델의 성능을 향상시키는 데 중점을 두고 있습니다. LSPO의 검증은 다양한 기본 모델과 데이터 세트에서 수행되었으며, 실험 결과 최종 모델의 성능이 일관되게 향상되었음을 확인했습니다.

- **Performance Highlights**: LSPO를 통한 훈련 모델은 기존 기준 접근법에 비해 향상된 특성과 효율성을 보여줍니다. 비록 LSPO가 표준 RL 알고리즘에 맞추기 위해 추가적인 샘플링을 필요로 하지만, 동일한 훈련 시간 내에서 효과적으로 작동함을 보여줍니다. 이러한 결과는 응답 길이를 데이터를 필터링하는 기준으로 사용하는 것이 RLVR의 효과성을 높이는 데 있어 유망한 방향임을 강조하고 있습니다.



### VOGUE: Guiding Exploration with Visual Uncertainty Improves Multimodal Reasoning (https://arxiv.org/abs/2510.01444)
- **What's New**: 이 논문에서는 Visual-Uncertainty-Guided Exploration (VOGUE)이라는 새로운 방법을 소개하며, 이 방법은 멀티모달 대형 언어 모델(MLLM)의 탐색 문제를 해결하는 데 초점을 맞추고 있습니다. 전통적인 방법들이 이미지를 고정된 조건으로 취급하는 대신, VOGUE는 이미지를 확률적 맥락으로 간주하고 시각적 변동성을 측정함으로써 더 나은 탐색 방향을 제시합니다. 이 방법을 통해 시각적 불확실성을 기반으로 한 학습 목표의 재조정이 가능해 여태까지 간과되었던 비밀스러운 탐색 경로를 유도합니다.

- **Technical Details**: VOGUE는 훈련 시 두 가지 가지(branch)를 활용하는 이중 가지 전방 패스를 통해 시각적 불확실성을 정량화합니다. 원본 이미지와 의미가 보존된 퍼터베이션(perturbation)된 이미지를 모두 사용하여 정책의 민감도를 계산하고, 비슷한 방식으로 KL 발산(KL divergence)을 적용하여 탐색 우위를 조형합니다. 이러한 시각적 불확실성은 불확실성 비례 보너스로 연결되어, 초기 훈련에서는 탐색에 중점을 두고 훈련이 안정화되면 원본 이미지로 초점을 옮기는 방식으로 진행됩니다.

- **Performance Highlights**: VOGUE는 GRPO 구현 하에 Qwen2.5-VL-3B 및 7B 모델에서 세 개의 시각 수학 벤치마크와 세 개의 일반 도메인 추론 벤치마크에서 각각 평균 2.6% 및 3.7%의 pass@1 정확도를 증가시키며, 일반적으로 발견되는 RL 미세 조정에서의 탐색 감소를 효과적으로 완화하였습니다. 또한, VOGUE는 텍스트 전용 설정에서 효과를 보이는 Pass@k Training 방법보다 지속적으로 우수한 성능을 발휘하여, 높은 pass@1 및 일관된 pass@k 향상을 달성했습니다.



### Optimal Stopping vs Best-of-$N$ for Inference Time Optimization (https://arxiv.org/abs/2510.01394)
Comments:
          24 pages

- **What's New**: 본 논문은 대형 언어 모델(LLM)의 추론 과정에서 출력 품질과 비용을 균형 있게 조정하는 새로운 최적화 프레임워크를 제안합니다. 이 프레임워크는 고전적인 'Pandora’s Box' 문제에 기반하여, 무작위 보상을 가진 '상자'를 여는 것으로 각 생성을 이해합니다. 이로 인해, 알고리즘이 보상 분포를 모르더라도 언제 생성을 중단해야 할지를 결정할 수 있도록 합니다.

- **Technical Details**: 이 논문에서는 우리가 개발한 UCB 스타일의 Pandora's Box 알고리즘을 소개합니다. 이 알고리즘은 알려지지 않은 보상 분포에 적응하도록 설계되었으며, 최적 중단 임계치를 유지하여 Weitzman의 최적 정책에 비례하는 실수 없이 결과를 보장합니다. 또한 Bradley–Terry에서 영감을 받은 보상 정규화 변환을 통해 다양한 프롬프트 간의 보상 스케일링 문제를 해결하며, 이를 통해 적응형 메타 생성 프로세스를 제공합니다.

- **Performance Highlights**: 실험 결과, AlpacaFarm 및 HH-RLHF 데이터셋에서 우리의 적응형 전략은 비적응형 Best-of-N 샘플링과 동일한 보상을 얻으면서도 평균적으로 15-35% 덜 생성할 수 있음을 보여줍니다. 이는 이론적 성과 경계 및 LLM 배포를 위한 실질적인 효율성 증대를 모두 제공하며, 최적 중단 이론과 추론 시간 스케일링 사이의 원칙적인 연결을 수립합니다.



### Fine-tuning with RAG for Improving LLM Learning of New Skills (https://arxiv.org/abs/2510.01375)
Comments:
          Under review at ICLR 2026

- **What's New**: 이 논문에서는 멀티 스텝 작업을 위해 배치된 대규모 언어 모델(LLM) 에이전트의 예측 가능한 실패를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 retrieval-augmented generation (RAG) 방법이 아닌, 추출된 힌트를 사용하여 에이전트의 실수를 이용한 학습 과정을 통해 성능을 향상시키는 파이프라인을 구축했습니다. 이 방식은 외부 데이터베이스를 유지할 필요가 없으며, 컴퓨팅 오버헤드를 줄이는 데 도움이 됩니다.

- **Technical Details**: 제안된 방법은 (1) 에이전트의 실패로부터 재사용 가능한 간결한 힌트를 추출하고, (2) 이러한 힌트를 사용하여 에피소드 시작 시 향상된 teacher trajectories를 생성하며, (3) 힌트 문자열을 제거한 상태에서 학생 모델을 학습시킵니다. 이 과정은 기억이 아닌 내재화(internalization)를 강요하여 수행됩니다. 실험은 두 개의 상호작용 벤치마크인 ALFWorld와 WebShop에서 진행되었습니다.

- **Performance Highlights**: ALFWorld에서 증류된 학생 모델은 기준선 에이전트보다 일관되게 우수한 성능을 보이며, 최대 91%의 성공률을 기록했습니다(기준선: 79%). WebShop에서는 72점으로 기준선인 61점에 비해 개선되었습니다. 사용된 토큰 수는 환경에 따라 retrieval-augmented teacher보다 10-60% 감소하였으며, 이 접근 방식은 다양한 모델 스케일(7B/14B 파라미터) 및 에이전트 아키텍처(ReAct/StateAct)에 걸쳐 일반화될 수 있습니다.



### Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effor (https://arxiv.org/abs/2510.01367)
- **What's New**: 이번 논문에서는 보상 해킹(reward hacking)이라는 문제를 다룹니다. 이는 모델이 보상 함수의 허점을 이용하여 의도된 작업을 해결하지 않고도 높은 보상을 얻는 경우로, 명시적이거나 암묵적으로 나타날 수 있습니다. 이 문제를 해결하기 위해 TRACE (Truncated Reasoning AUC Evaluation)이라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: TRACE는 모델의 Chain-of-Thought (CoT)에서 보상 해킹을 감지하기 위해 ‘노력’effort의 정도를 측정하여 이를 정량화합니다. 연구에서는 다양한 길이에서 CoT를 잘라내고 모델이 검증기를 통과할 수 있는 비율을 측정하여 해킹 모델의 특성을 분석합니다. 더 짧은 CoT로도 높은 통과율을 얻는 모델은 해킹으로 간주됩니다.

- **Performance Highlights**: TRACE의 성능은 수학적 추론에서 기존 72B CoT 모니터에 비해 65% 이상의 향상을 보였으며 32B 모니터의 코드 작성 부분에서도 30% 이상의 개선을 기록했습니다. 또한 TRACE는 훈련 중 알려지지 않은 허점을 발견할 수 있는 능력을 보여줍니다. 이러한 결과는 현재의 모니터링 방법으로는 효과적인 감시가 어려운 상황에서 TRACE가 확장 가능한 비지도 접근 방식을 제공한다는 것을 의미합니다.



### WAInjectBench: Benchmarking Prompt Injection Detections for Web Agents (https://arxiv.org/abs/2510.01354)
- **What's New**: 이번 논문은 웹 에이전트를 대상으로 한 프롬프트 주입 공격(prompt injection attacks)을 탐지하기 위한 최초의 종합 벤치마크 연구를 소개합니다. 연구는 기존의 다양한 공격 방법 및 탐지 방안이 웹 에이전트에 대해 체계적으로 평가되지 않았다는 점에서 출발합니다. 이를 통해 연구진은 공격 유형을 세분화하고, 악의적(malicious) 및 선의적(benign) 샘플을 포함하는 데이터 세트를 구축하였습니다.

- **Technical Details**: 이 논문에서는 공격 모델에 기반하여 세분화된 범주화를 통해 다양한 프롬프트 주입 공격을 정의합니다. 연구팀은 여러 가지 공격 방식으로 생성된 악의적 텍스트 샘플 및 두 가지 범주에서 수집된 선의적 텍스트 샘플, 그리고 악의적 및 선의적 이미지 샘플을 포함하는 데이터 세트를 만들었습니다. 또한, 텍스트 및 이미지 기반 탐지 방법을 체계화하고 이에 대한 성능 평가를 수행했습니다.

- **Performance Highlights**: 연구 결과, 일부 탐지기는 명시적 텍스트 지시문이나 가시적 이미지 변형을 기반으로 하는 공격을 중간에서 높은 정확도로 식별할 수 있는 것으로 나타났습니다. 그러나 명시적 지시문을 생략하거나 인지 불가능한 변형을 사용하는 공격에 대해서는 대체로 실패하는 경향을 보였습니다. 이러한 발견은 웹 에이전트를 보호하기 위한 탐지 기술 발전에 중요한 통찰을 제공합니다.



### MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments (https://arxiv.org/abs/2510.01353)
Comments:
          Accepted to NeurIPS 2025 SEA Workshop

- **What's New**: 최근의 연구들은 대화형 사례에 중점을 두어 왔지만, 동적인 기업 환경에서 메모리 평가의 필요성이 커지고 있습니다. 이 논문에서는 MEMTRACK이라는 새로운 벤치를 소개하며, 이는 다중 플랫폼 에이전트 환경에서 장기 메모리 및 상태 추적을 평가하기 위해 설계되었습니다. MEMTRACK은 Slack, Linear, Git과 같은 여러 커뮤니케이션 및 생산성 플랫폼에서 비동기 이벤트를 통합하여 현실적인 조직 워크플로우를 모델링합니다.

- **Technical Details**: 각 벤치마크 인스턴스는 연대적으로 플랫폼 간의 엮인 타임라인을 제공하며, 소음이 많고 상충하는 정보, 코드베이스/파일 시스템 이해 및 탐색 가능성을 포함합니다. 따라서 이 벤치마크는 메모리의 획득(acquisition), 선택(selection), 갈등 해결(conflict resolution) 능력을 테스트합니다. MEMTRACK 데이터세트는 수동 전문가 설계와 확장 가능한 에이전트 기반 합성을 통해 생성하여, 실제 소프트웨어 개발 프로세스에 기반한 생태학적으로 유효한 시나리오를 제공합니다.

- **Performance Highlights**: 정확성(Correctness), 효율성(Efficiency), 중복성(Redundancy)을 포착하는 관련 메트릭을 도입하여 메모리 메커니즘의 효과성을 비교합니다. SoTA LLMs 및 메모리 백엔드에 대한 실험 결과, 장기적인 메모리 활용, 플랫폼 간 의존성 처리 및 모순 해결에 관한 도전에 직면하게 됩니다. 특히, 가장 성능이 뛰어난 GPT-5 모델조차 MEMTRACK에서 60%의 정확성 점수만을 기록하는 것으로 나타났습니다.



### Aristotle: IMO-level Automated Theorem Proving (https://arxiv.org/abs/2510.01346)
- **What's New**: 이번 논문에서는 Aristotle이라는 AI 시스템을 소개하며, 이는 형식적 검증(formal verification)과 비공식적 추론(informal reasoning)을 결합하여 2025 국제수학올림피아드(IMO) 문제에서 금메달에 해당하는 성과를 달성했다. Aristotle은 세 가지 주요 구성 요소로 구성되어 있으며, 이는 Lean 증명 탐색 시스템, 비공식적인 추론 시스템, 그리고 전용 기하학 해결기이다. 이 시스템은 자동 정리 증명(automated theorem proving)에서 최첨단 성능을 보여준다.

- **Technical Details**: Aristotle의 기초가 되는 구성 요소는 Lean 코드를 처리하는 증명 탐색 알고리즘이다. 이 알고리즘은 Monte Carlo Tree Search (MCTS)를 기반으로 하며, 학습된 가치 함수(learned value function)를 사용한다. 이를 통해 Lean 증명 스케치에서 증명이 이루어지지 않은 목표를 모두 증명하기 위해 다양한 기법을 채택하고, 기하학 문제는 AlphaGeometry에 기반하여 따로 해결한다.

- **Performance Highlights**: Aristotle은 2025 IMO 문제에서 다섯 개의 문제에 대해 정형화된 솔루션(formal solution)을 제공하여 금메달 수준의 성과를 달성하였다. 이 시스템은 시가형 탐색 방법을 기반으로 하여 자동 정리 증명 시스템의 성능을 높이는 데 유리하게 설계되었다. 실제로 Aristotle은 다양한 수학 문제에 대해 대학 수준의 도전 과제를 증명할 수 있고, 이전의 연례 과정에서 이루어졌던 유사한 작업들과 더불어 그 능력을 넓혀가고 있다.



### Agentic Jigsaw Interaction Learning for Enhancing Visual Perception and Reasoning in Vision-Language Models (https://arxiv.org/abs/2510.01304)
- **What's New**: AGILE는 VLM(Visual-Language Models)의 시각 인식(visual perception)과 추론(reasoning) 능력을 향상시키기 위해 제안된 새로운 프레임워크입니다. 기존의 VLM들이 간단한 jigsaw 작업에서도 무작위로 수행되는 문제를 해결하기 위해 AGILE은 jigsaw 풀기를 상호작용적인 과정으로 모델링합니다. 이 방법은 실행 가능한 코드를 생성하여 환경과의 상호작용을 통해 점진적으로 개선됩니다.

- **Technical Details**: AGILE은 Python 코드를 생성하여 현재 상태에 기반한 행동을 실행하며, 매 단계에서 환경은 세밀한 시각적 피드백을 제공합니다. jigsaw 퍼즐의 구조적 특성을 활용하여, 모델은 이미지 타일을 교환하거나 현재의 퍼즐 상태를 관찰하는 등의 행동을 통해 더 나은 인식 및 추론 기술을 습득하게 됩니다. 이 과정의 반복적인 상호작용을 통해 모델은 시각적 구성 요소 간의 구조적 관계를 파악하게 됩니다.

- **Performance Highlights**: AGILE을 통해 jigsaw 작업에서의 성능이 크게 개선되었습니다. 예를 들어, 2 × 2 설정에서 정확도가 9.5%에서 82.8%로 증가하였으며, 9개의 일반 비전 과제에서 평균 3.1%의 일반화 성능 향상이 나타났습니다. 이러한 결과는 AGILE이 시각적 인식과 추론 능력 모두에서 상당한 향상을 나타낸다는 것을 보여줍니다.



### LLM-based Multi-Agent Blackboard System for Information Discovery in Data Scienc (https://arxiv.org/abs/2510.01285)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 발전에 따른 데이터 과학에서의 새로운 기회를 제시합니다. 특히, 기존의 데이터 레이크(data lake) 내에서 관련 데이터를 발견하는 데 어려움이 있는 방법론의 한계를 극복하기 위한 새로운 다중 에이전트 통신 패러다임을 제안합니다. 이 구조는 기존의 중앙 제어기 의존성을 없애고 에이전트들이 자율적으로 기여할 수 있도록 합니다.

- **Technical Details**: 우리는 전통적인 AI 모델의 블랙보드 아키텍처에서 영감을 얻어, 중앙 에이전트가 요청을 공유 블랙보드에 게시하고 여러 하위 에이전트가 자신의 능력에 따라 응답하는 프레임워크를 설계했습니다. 이 방식은 각 하위 에이전트의 전문 지식에 대한 사전 지식이 필요 없으며, 데이터 레이크에서 관련 파일을 식별하는데 효과적임을 보여주었습니다. 세 가지 벤치마크(KramaBench, DS-Bench, DA-Code)를 대상으로 평가하여, 블랙보드 구조가 상대적으로 13%에서 57%까지의 성능 향상을 보임을 확인했습니다.

- **Performance Highlights**: 실험 결과, 블랙보드 아키텍처는 기존의 RAG 및 마스터-슬레이브 다중 에이전트 패러다임보다 우수한 성과를 보였으며, F1 점수에서도 최대 9% 향상되었습니다. 이 연구는 블랙보드 패러다임이 다중 에이전트 시스템을 위한 확장 가능하고 일반화된 통신 프레임워크로 자리잡을 수 있는 가능성을 보여줍니다. 실제 적용 사례를 통해 이 시스템의 효율적인 문제 해결 능력을 강조하였습니다.



### RLP: Reinforcement as a Pretraining Objectiv (https://arxiv.org/abs/2510.01265)
Comments:
          RLP introduces a new paradigm for RL-based Pretraining

- **What's New**: 최근 발표된 RLP(Reinforcement Learning Pre-training)는 전통적인 훈련 방법과 비교해 Chain-of-Thought(사고의 연쇄)를 예측의 사전 행동으로 취급하여 정보를 기반으로 한 강화학습 목표를 설정합니다. 이는 모델이 다음 토큰을 예측하기 전에 스스로 사고하도록 유도하여 사전 훈련의 초기 단계에서 독립적인 사고 행동을 가르치는 데 초점을 맞춥니다. 연구에서는 비검증 방식으로 로깅 가능성을 기반으로 한 보상 신호를 제공하여 훈련 효율성을 높이고 있습니다.

- **Technical Details**: RLP는 Chain-of-Thought를 생성하는 것을 통해 각 다음 토큰을 예측하기 전에 사고를 수행하도록 구조화되어 있으며, 사고가 다음 토큰 예측에 미치는 영향을 로그 가능성 비율로 측정합니다. 이 과정은 비검증적이며 밀집 보상을 제공하여 자연어 데이터에서 일반화 가능성을 높입니다. RLP는 기존의 강화학습 접근 방식에서 본질적인 한계를 극복하도록 밀접하게 설계되어 있습니다.

- **Performance Highlights**: RLP를 통해 Qwen3-1.7B-Base 모델에서 19%의 성과 향상 효과가 나타났으며, AIME25 및 MMLU-Pro와 같은 복잡한 추론 작업에서 더욱 두드러진 결과를 보입니다. Nemotron-Nano-12B-v2 모델에 적용 시 42.81%에서 61.32%로 향상되어 과학적 추론에서 23%의 증가를 달성했습니다. 이는 다양한 아키텍처 및 모델 크기에서의 확장성을 잘 보여줍니다.



### RSAVQ: Riemannian Sensitivity-Aware Vector Quantization for Large Language Models (https://arxiv.org/abs/2510.01240)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 극저비트 양자화를 극대화하기 위해 새로운 VQ 프레임워크인 RSAVQ를 제안합니다. RSAVQ는 오류 방향 감도 유도(Error Direction Sensitivity Guidance, EDSG)와 가중치 채널 감도 유도(Weight Channel Sensitivity Guidance, WCSG)의 두 가지 기하학적 혁신을 도입하여 기존의 문제들을 해결합니다. 이 프레임워크는 정보를 기하학적으로 모델링하여 양자화 정확도를 향상시킵니다.

- **Technical Details**: RSAVQ는 피셔 정보 행렬(Fisher Information Matrix, FIM)을 활용하여 LLM의 파라미터 공간을 비균일 곡률을 가진 리만 다양체로 모델링합니다. EDSG는 양자화 오류를 낮은 감도 방향으로 투영하여 모델 성능에 미치는 부정적 영향을 최소화하며, WCSG는 각 채널의 감도를 동적으로 할당하여 비트 자원을 효율적으로 분배합니다. 이러한 접근법은 주어진 비트 제약 내에서 최적의 양자화 솔루션을 제공합니다.

- **Performance Highlights**: 실험 결과, RSAVQ는 LLaMA-3 8B 모델에서 2비트 양자화 시 기존 방법인 VPTQ 및 QuIP#보다 0.4의 PPL과 1.5의 제로샷 정확도로 성능이 향상되었습니다. 이 논문은 제약된 환경에서의 실용적 해결책을 제시하며, 정보 기하학과 신경망의 양자화 사이의 이론적 다리를 제공합니다. RSAVQ는 극저비트 시나리오에서 우수한 성능을 입증하였으며, 향후 LLM의 효율적인 학습에 기여할 것으로 기대됩니다.



### Automated Extraction of Material Properties using LLM-based AI Agents (https://arxiv.org/abs/2510.01235)
- **What's New**: 최근 제안된 연구는 약 10,000개의 전체 텍스트 과학 기사를 활용하여 열전 소재의 성능 지표와 구조적 속성을 자동으로 추출하는 LLM(large language model) 기반의 워크플로우를 소개합니다. 이 시스템은 높은 정확도를 유지하면서 컴퓨팅 비용을 균형 있게 조절할 수 있도록 동적 토큰 할당 및 조건부 테이블 파싱을 통합합니다. 이 연구 결과로부터 27,822개의 온도에 따른 물리적 속성 기록이 축적되었으며, 이것은 열전 소재 발견의 새로운 가능성을 제시합니다.

- **Technical Details**: 이 연구는 다양한 기계 학습 기법을 활용하여 주요 과학 저널에서 연구 기사들로부터 데이터를 수집했습니다. DOI를 통해 관련 기사를 검색하고 XML 또는 HTML 형식으로 데이터를 다운로드하여 처리하는 자동화된 파이프라인을 구축했습니다. 특히 '마무리', '참고문헌'과 같은 비관련 부분을 제거하고, 열전 속성 관련 문장만을 남기는 필터링 과정을 통해 데이터의 정확성을 높였습니다.

- **Performance Highlights**: GPT-4.1 모델을 사용한 검증 결과, 열전 특성에 대한 F1 점수는 0.91로 최고치를 기록하였으며, GPT-4.1 Mini 모델도 거의 유사한 성능을 보였습니다. 이러한 시스템은 높은 수준의 정확도로 원자료에서 열전 데이터의 표준화된 기록을 생성할 수 있도록 하여, 대규모 데이터 기반의 소재 발견을 위한 기초를 마련하였습니다. 또한, 커뮤니티 접근을 용이하게 하기 위해 인터랙티브 웹 탐색기를 출시하여 사용자들이 데이터를 조회하고 CSV로 내보낼 수 있도록 지원합니다.



### Utilizing Modern Large Language Models (LLM) for Financial Trend Analysis and Digest Creation (https://arxiv.org/abs/2510.01225)
Comments:
          This is the version of the article accepted for publication in SUMMA 2024 after peer review. The final, published version is available at IEEE Xplore: https://doi.org/10.1109/SUMMA64428.2024.10803746

- **What's New**: 이번 논문은 연구자들이 최신 정보를 유지하는 데 도움을 주기 위해 Google의 Gemini Pro를 활용한 혁신적인 프레임워크를 소개합니다. 이는 Large Language Models (LLMs)의 힘을 통해 자동으로 금융 요약(reports)을 생성하는 방법을 제시합니다. 이 방법은 기존의 분석 방식의 한계를 극복하여 어마어마한 양의 비정형 데이터(unstructured data)를 효율적으로 처리할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 OpenAlex에서 데이터 추출(data extraction), 전략적 프롬프트 엔지니어링(prompt engineering), LLM 기반 분석을 결합하여 포괄적인 요약(digests)을 생성하는 자동화된 예제를 설명합니다. 또한, LLM의 작동 원리를 간단하게 설명하고 그 힘을 활용하여 연구자들이 시간을 절약하고 최신 트렌드(Trends)에 대한 정보를 얻을 수 있도록 하는 방법을 다룹니다. 이 과정은 데이터 수집(data acquisition) 및 JSON 구성(JSON construction)에서 Gemini와의 상호작용(interaction) 및 PDF 보고서 자동 생성까지 포함됩니다.

- **Performance Highlights**: 논문에서는 자동 생성된 보고서가 주요 발견(key findings)을 일반화하고 신흥 트렌드를 식별하는 데 도움을 준다고 강조합니다. 또한, GitHub 저장소 링크를 제공하여 이 프로젝트의 접근성과 추가적인 개발을 촉진합니다. 이 접근 방식은 명확하고 쉽게 소화할 수 있는 형식으로 실행 가능한 통찰(insights)을 전달함으로써 연구자들의 효율성을 높이는 데 기여합니다.



### Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledg (https://arxiv.org/abs/2510.01223)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 정렬 방어가 중첩 시나리오에 민감하지 않다는 점을 처음으로 규명하고 체계적으로 검증합니다. 이 중첩 시나리오는 쿼리와 매우 의미적으로 관련이 있으며, 목표로 하는 유해한 지식을 통합합니다. 이를 통해 새로운 공격 방법인 RTS-Attack(범주에 맞춘 유해 지식이 포함된 의미적 중첩 시나리오)를 제안하여 LLM의 정렬 방어를 우회할 수 있는 적응형 자동화 프레임워크를 구축합니다.

- **Technical Details**: RTS-Attack는 세 가지 주요 단계로 구성됩니다: (1) 쿼리 분류 및 의도 추출, (2) 중첩 시나리오 생성, (3) 지침 맞춤화. 각 단계에서 harmful query를 기반으로 적합한 nested scenario jailbreak prompt를 만드는 것을 목표로 합니다. 특히, 유해 지식과 관련된 의미적 요소를 강조하여 지침을 사용자화함으로써 LLM의 정렬 방어를 효과적으로 우회할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 RTS-Attack이 다양한 최신 LLM, 예를 들어 GPT-4o, Llama3-70b, Gemini-pro 등에서 기초 방법들과 비교하여 효율성과 범용성에서 우수한 성능을 보임을 입증했습니다. RTS-Attack은 짧은 상호 작용 회차 내에 LLM jailbreak을 실행할 수 있으며, 각 쿼리 클래스에 맞춰 효과적인 jailbreak 지침을 생성합니다.



### Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs (https://arxiv.org/abs/2510.01218)
Comments:
          Second Conference on Language Modeling, 2025

- **What's New**: 이번 연구에서는 언어 모델에서 생성된 출력의 창의성을 평가하는 데 필수적인 다양성이 중요하다는 점을 강조합니다. 기존의 온도 기반 샘플링 기법들은 창의성을 증가시키지만, 수학적 추론과 같은 높은 정밀도가 요구되는 작업에서는 오히려 reasoning의 질을 저하시킬 수 있습니다. 저자들은 'selective sampling'이라는 새로운 기법을 제안하여, 샘플링 위험 메트릭에 따라 greedy 샘플링과 높은 온도 샘플링 사이를 동적으로 전환합니다.

- **Technical Details**: selective sampling은 샘플링의 오류 가능성을 예측하기 위해 가벼운 분류기를 훈련시키며, 이는 백본 언어 모델에 최소한의 지연(latency)을 추가하는 방식으로 통합됩니다. 이 접근법은 생성된 원래 모델 출력을 보존하는 동시에 구현이 용이한 특징을 가집니다. 다양한 샘플링 기법들을 분석한 결과, selective sampling이 일반적으로 사용되는 절단 및 엔트로피 기반 샘플링 기법들보다 품질-다양성 균형에서 더 나은 성능을 발휘함을 입증했습니다.

- **Performance Highlights**: 수학적 추론(task) 실험에서는 selective sampling이 고온(high-temperature) 환경에서도 품질-다양성 거래에서 우수한 성능을 보여주었습니다. 저자들은 기법의 효과를 실험을 통해 확인하였고, 기존의 샘플링 방법들이 실패하는 지점을 분석함으로써 향후 연구 방향을 제시하고 있습니다. 이 연구는 LLM의 출력에서 품질과 다양성을 유지하는 것이 중요함을 강조하며, 적응 가능한 디코딩 전략의 필요성을 역설합니다.



### The Data-Quality Illusion: Rethinking Classifier-Based Quality Filtering for LLM Pretraining (https://arxiv.org/abs/2510.00866)
Comments:
          21 pages, 20 figures, 2 tables, preprint

- **What's New**: 본 논문은 Classifier-based Quality Filtering (CQF)에 대한 심층 분석을 제공합니다. CQF는 데이터 품질을 개선하기 위해 훈련 데이터와 소규모의 고품질 데이터 세트를 구별하는 이진 분류기를 교육합니다. 이를 통해 각 문서에 품질 점수를 할당하고 상위 점수를 받은 문서만 선택하는 방식으로, 주류 프리트레인 파이프라인에서 널리 활용되고 있습니다.

- **Technical Details**: CQF는 낮은 품질의 문서를 포함한 프리트레인 세트와 높은 품질의 소규모 데이터 세트를 입력으로 받습니다. 보통 LQ 세트는 다양한 출처에서 수집된 대량의 웹 크롤링 문서로 구성되며, HQ 세트는 Wikipedia와 같은 철저하게 편집된 자료에서 제공합니다. 본 논문에서는 RedPajama-V2를 LQ 세트로 사용하고, CQF의 성능 최적화를 위한 메커니즘을 분석합니다.

- **Performance Highlights**: CQF는 하위 작업에서 성능을 향상시키지만, 고품질 데이터 세트를 기반으로 한 언어 모델링에는 반드시 유리하지 않다는 역설적인 결과를 제시합니다. 이러한 결과는 CQF가 고품질 세트 자체를 암묵적으로 필터링하는 방식 때문으로 설명됩니다. 또한, CQF와 중요성 샘플링 방법의 비교를 통해 두 메소드 간의 뚜렷한 차이점을 강조하며, CQF의 품질 개념이 제한적임을 시사합니다.



New uploads on arXiv(cs.IR)

### Study on LLMs for Promptagator-Style Dense Retriever Training (https://arxiv.org/abs/2510.02241)
Comments:
          CIKM 2025 short research paper

- **What's New**: Promptodile는 소규모의 오픈소스 대규모 언어 모델(Large Language Models, LLMs)을 활용한 효과적인 쿼리 생성 방법을 제시합니다. 기존의 Promptagator는 대형 LLM에 의존했으나, 이제 3B 매개변수의 모델도 효과적으로 사용될 수 있다고 연구 결과가 입증되었습니다. 이 연구는 비용 문제나 데이터 프라이버시로 인해 대형 모델을 사용할 수 없는 상황에 대한 실용적인 대안을 제공합니다.

- **Technical Details**: Promptodile은 총 10개의 오픈소스 LLM을 활용하여 1B에서 14B 매개변수의 다양한 모델을 평가하고, 7개의 저자원 BEIR 데이터셋에서 실험을 수행했습니다. 이를 통해 Promptagator와 유사한 성능을 보이며, 작은 LLM도 큰 모델만큼 효과적이라는 것을 확인했습니다. 이 연구는 주로 프롬프트의 몇 가지 예시를 바탕으로 하여, 적은 수의 주석이 있는 쿼리-문서 쌍을 대량의 합성 훈련 데이터로 확대하는 과정이 포함됩니다.

- **Performance Highlights**: Promptodile은 모든 평가된 LLM에서 일반적으로 Promptagator와 경쟁력을 보이며, 최근의 접근 가능한 LLM이 효과적인 쿼리 생성기로 활용될 수 있음을 입증했습니다. 연구 결과, 3B 매개변수의 작은 모델이 7B에서 14B 매개변수의 큰 모델과 같은 성능을 발휘하여, 비용이 많이 드는 LLM을 사용할 필요가 없음을 보여주었습니다.



### Contrastive Retrieval Heads Improve Attention-Based Re-Ranking (https://arxiv.org/abs/2510.02219)
- **What's New**: 최근 대형 언어 모델(LLMs)의 강력한 제로샷(zero-shot) 및 긴 컨텍스트(long-context) 능력 덕분에 효율적인 재정렬 시스템(re-ranking systems)이 가능해졌습니다. 본 논문에서는 CoRe heads라는 개념을 도입하여, 관련 문서와 상관관계가 높은 주목(attention) 헤드를 강조하며, 불필요한 노이즈를 줄이고 성능을 향상하는 방법을 제안합니다. CoRe heads는 전체 헤드의 1%도 안 되는 소수의 집합으로, 이를 통해 최첨단 리스트 기반(list-wise) 재정렬기를 구현하고 있습니다.

- **Technical Details**: CoRe heads는 대조적 점수 산정 기법을 활용하여 주목(attention)을 문서의 관련성과 연관된 정도에 따라 평가합니다. 이 방법은 기존의 절대적 주목 점수 평가 접근 방식과 달리 상대적 순위를 모델링하여 성능을 향상시킵니다. 또한, CoRe heads는 중간 계층(middle layers)에 집중되어 있으며, 모델의 최종 50% 계층을 가지치기(pruning)함으로써 지연 시간(inference time) 및 메모리 사용량(memory usage)을 줄이면서도 정확도를 유지할 수 있습니다.

- **Performance Highlights**: CoRe heads를 활용한 실험 결과는 세 가지 LLM에서 강력한 기준선에 비해 재정렬 정확도를 크게 개선하는 것을 보여줍니다. CoRe heads는 다양한 벤치마크에서 일반화되는 성능을 보이며, 특히 NQ(Natural Questions) 데이터셋에서의 훈련이 BEIR 벤치마크와 MLDR 벤치마크에서 성공적으로 적용되는 모습을 확인했습니다. 최상위 CoRe heads는 주로 중간 계층에서 분포되어 있어, 이들의 효율성을 토대로 실제 검색 시스템에서의 적용 가능성을 높이고 있습니다.



### Ranking Items from Discrete Ratings: The Cost of Unknown User Thresholds (https://arxiv.org/abs/2510.01871)
Comments:
          12 pages, 4 figures

- **What's New**: 이 연구는 정보 검색 및 추천 시스템에서 아이템을 순위 매기는 작업에 대한 새로운 이론적 통찰력을 제공합니다. 연구팀은 사용자에게 부여된 점수를 통해 세부적인 아이템 순위를 복원할 수 있는 가능성을 조사하였으며, 이는 사용자의 기준(threshold)와 아이템의 점수(score)가 숨겨져 있는 경우 매우 도전적인 문제라는 점을 밝혔습니다. 이 논문은 이론적인 한계를 정량화하고, 이를 통해 사용자 평가 기반 순위 매기의 복잡성을 분석합니다.

- **Technical Details**: 모델링에서 각 아이템은 0과 1 사이의 점수(score)를 가지며, 사용자들 각자는 저마다의 잠재적 기준(threshold)을 가지고 있습니다. 사용자는 아이템의 점수가 자신의 기준을 초과할 경우 긍정적으로 평가합니다. 실험적으로, 사용자 수(m)가 아이템 수(n)에 비례할 때 순위의 정확성은 O(n)으로 나타났으며, 사용자 수가 Ω(n²)일 경우 거의 완벽한 순위를 얻을 수 있다는 결과를 도출했습니다.

- **Performance Highlights**: 본 연구에서는 평가 요청에 비해 아이템 비교 요청이 훨씬 더 효과적임을 수학적으로 증명하였습니다. 사용자에게 적절한 기준을 가진 아이템을 요청하는 것이 더 나은 분별을 위해 필수적이며, 사용자의 기준이 다양해야만 세부 순위를 형성할 수 있음을 강조했습니다.  실험 결과는 새로운 알고리즘인 threshold binary search (TBS)의 효율성을 지지하며, 이 알고리즘은 사용자의 정보를 최대한 활용하여 정밀한 순위 매기기를 가능하게 합니다.



### TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling (https://arxiv.org/abs/2510.01698)
Comments:
          Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)

- **What's New**: 최근 대규모 언어 모델(LLM)의 발전 덕분에 자연어 상호작용을 통한 추천 시스템인 generative recommenders가 가능해졌습니다. 하지만 기존 시스템들은 메타데이터나 속성 필터링과 같은 다른 중요한 컴포넌트를 충분히 활용하지 못하고 있습니다. 본 연구에서는 도구 호출(tool calling)을 기반으로 하는 LLM 기반 음악 추천 시스템을 제안하여, 사용자의 의도를 해석하고 도구 호출을 계획하여 통합 검색-재정렬 파이프라인을 구성합니다. 이 시스템은 다양한 추천 시나리오에서 경쟁력 있는 성능을 달성합니다.

- **Technical Details**: 제안된 시스템 아키텍처는 두 가지 주요 구성 요소로 나누어져 있습니다. 첫 번째는 LLM과 다양한 도구로 구성된 음악 추천 에이전트이며, 두 번째는 도구를 실행하고 최종 추천을 수행하는 외부 환경입니다. 시스템은 사용자 프로필, 이전 대화 상태, 사용자 쿼리를 입력으로 받아 음악 추천 목록을 생성하며, 선택된 도구를 기반으로 데이터베이스를 필터링하는 방식으로 작동합니다. 각 도구의 출력은 다음 도구의 입력에 직접 영향을 미치는 순차적인 파이프라인에서 실행되어 최종 추천의 질에 큰 영향을 미칩니다.

- **Performance Highlights**: 제안된 통합 도구 호출 프레임워크는 대화형 추천 벤치마크에서 강력한 기반선보다 향상된 Hit@K를 달성하여 제로샷(Zero-shot) 효과성을 보여줍니다. 다양한 검색 방법을 선택적으로 활용하여 사용자 쿼리에 기반한 적절한 검색 메소드를 사용함으로써 효율적인 추천을 실현했습니다. 본 연구는 음악 추천 시스템의 새로운 패러다임을 제시하며, LLM을 활용한 대화형 추천의 가능성을 확장합니다.



### LLM4Rec: Large Language Models for Multimodal Generative Recommendation with Causal Debiasing (https://arxiv.org/abs/2510.01622)
- **What's New**: 이 논문은 현대의 생성형 추천 시스템에서 다중 모달 데이터 처리, 알고리즘적 편향 제거 및 투명한 의사결정 프로세스 제공의 문제를 해결하기 위해 다섯 가지 주요 혁신을 제안합니다. 새로운 프레임워크는 멀티모달 융합 아키텍처, 검색 보강 생성 메커니즘, 인과 추론 기반의 편향 제거, 설명 가능한 추천 생성, 실시간 적응형 학습 능력을 포함하여 대규모 언어 모델을 기본으로 합니다. 이를 통해 다양한 콘텐츠 유형을 처리하고 사용자 선호에 따라 적응 가능한 추천 시스템을 구축할 수 있습니다.

- **Technical Details**: 제안된 Enhanced GenRec 프레임워크는 다중 모달 융합 구조를 통해 텍스트, 범주 및 수치 데이터를 통합하여 사용자와 아이템의 풍부한 표현을 생성합니다. 또한, 검색 보강 생성 메커니즘은 데이터셋 내의 상황별 정보를 활용하여 추천의 정확성과 범위를 향상시킵니다. 편향 제거 기술은 추천 결과의 체계적인 편향을 탐지하고 완화하기 위해 인과 추론을 활용하고, 설명 가능한 추천 모듈은 추천 결정에 대한 자연어 설명을 생성하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(MovieLens-25M, Amazon-Electronics, Yelp-2023)에 대한 광범위한 실험 결과, 제안된 프레임워크는 기존 접근 방식에 비해 추천 정확도, 공정성 및 다양성이 일관되게 개선되었습니다. NDCG@10에서 최대 2.3%의 개선과 다양성 메트릭에서 1.4%의 향상을 이뤄내면서도 최적화된 추론 전략을 통해 계산 효율성을 유지합니다.



### Bridging Collaborative Filtering and Large Language Models with Dynamic Alignment, Multimodal Fusion and Evidence-grounded Explanations (https://arxiv.org/abs/2510.01606)
- **What's New**: 이 논문은 DynMM-Explain-LLMRec라는 새로운 추천 시스템 프레임워크를 소개합니다. 이 프레임워크는 사용자 상호작용 데이터를 지속적으로 학습할 수 있는 경량 모듈을 통해 실시간으로 적응하는 혁신적인 온라인 적응 메커니즘을 채택합니다. 또한 시각 및 오디오 콘텐츠와 협동 신호를 통합한 통합된 표현 방식을 설계하였습니다. 마지막으로, 사용자에게 명확한 이유를 제공할 수 있는 설명 시스템을 구현하였습니다.

- **Technical Details**: DynMM-Explain-LLMRec는 고정된 기본 정렬기를 기반으로 하여 동적 업데이트와 다중 모달 통합을 촉진합니다. 각 항목에 대해 최근 상호작용을 요약하는 동적 잠재 표현을 생성하며, 이를 통해 사용자 동적 특성을 잘 반영합니다. 또한 각 모달리티(visual, audio 등)를 통합하여 협동 패턴과 아이템 특성을 잇는 고유한 구조를 제공합니다. 이러한 구조는 다양한 모달리티가 결여될 경우에도 강력한 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: Comprehensive evaluation은 시간적 적응, 냉시작(Cold-start) 시나리오, 계산 효율성 및 설명 품질을 아우릅니다. 본 논문에서 제안한 방법은 기존 모델의 효율성을 유지하면서도 최소한의 계산 오버헤드로 실제 시스템에서 활용할 수 있음을 보여줍니다. 이에 따라 동적인 사용자 선호 변화를 따라잡고 명확한 해석과 높은 신뢰성을 제공하는 추천 시스템으로서의 가능성을 보여줍니다.



### Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomp (https://arxiv.org/abs/2510.01574)
Comments:
          Accepted to the Proceedings of the ACM SIGIR Asia Pacific Conference on Information Retrieval (SIGIR-AP 2025), December 7-10, 2025, Xi'an, China

- **What's New**: 이번 연구에서는 실시간 신경 쿼리 자동 완성 시스템에서 프레젠테이션 편향(presentation bias)을 완화하기 위한 데이터 중심 접근 방식을 제안합니다. 본 방법론은 사용자의 전체 쿼리를 바탕으로 합성 접두사를 생성하여 훈련 데이터를 다양하고 비편향적인 예제들로 풍부하게 하여 학습합니다. 이러한 접근은 기존의 사용자의 상호작용 데이터에서 발생하는 편향을 줄이는데 도움을 줍니다.

- **Technical Details**: 신경 순위 모델은 엄격한 지연(latency) 제약 하에서 실시간 배포를 위해 최적화되었습니다. 이는 쿼리 인기도(query popularity), 계절성(seasonality), 유사도 점수(fuzzy match scores) 등을 포함한 다양한 특성을 통합합니다. 우리는 쿼리 자동 완성 구조를 이용하여 O(n^2)에서 O(n)으로 계산 복잡성을 줄인 리스트와이즈 손실(listwise loss)의 간소화 버전을 도입하였습니다.

- **Performance Highlights**: 대규모 전자상거래 환경에 배포된 이 시스템은 사용자 참여를 통계적으로 유의미하게 개선하는 결과를 보여주었습니다. A/B 테스트 결과, 우리의 모델은 기존의 선형 바닥선(linear baseline) 보다 MRR(Mean Reciprocal Rank)을 1% 이상 향상시켰습니다. 이러한 결과는 균형 잡힌 데이터와 최적화된 손실 함수를 통해 훈련된 신경 LTR 모델의 효과성을 입증합니다.



### IoDResearch: Deep Research on Private Heterogeneous Data via the Internet of Data (https://arxiv.org/abs/2510.01553)
Comments:
          8 pages,4 figures

- **What's New**: 이 논문에서는 기존의 데이터 관리 한계를 극복하기 위해 IoDResearch(Internet of Data Research)라는 새로운 프레임워크를 제안하고 있습니다. IoDResearch는 개별 데이터 자원들을 FAIR 원칙(Findable, Accessible, Interoperable, Reusable)에 부합하는 디지털 객체로 캡슐화하고, 이를 통해 다양한 데이터 자원의 효율적인 탐색과 재사용을 수월하게 합니다. 이러한 혁신적인 접근은 과학적 발견과 혁신을 가속화할 수 있는 기반을 마련합니다.

- **Technical Details**: IoDResearch의 아키텍처는 세 가지 계층으로 구성되어 있습니다. 데이터 리소스 계층(Data Resource Layer)에서는 다양한 출처에서 수집된 원시 데이터를 포함하고, 디지털 객체 계층(Digital Object Layer)에서는 각 데이터를 DOI(디지털 객체 식별자)로 부여하고 메타데이터로 풍부하게 만들어 디지털 객체로 캡슐화합니다. 마지막으로 지식 정제 계층(Knowledge Refinement Layer)에서는 구조화된 지식 표상을 생성하여 다양한 데이터 간에 효율적인 검색을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, IoDResearch는 이전의 데이터 표현 방법과 비교하여 모든 작업에서 우수한 성능을 보였습니다. IoDResearch는 데이터 검색, 질문 응답(Question Answering), 보고서 작성 작업에서 일관되게 우수한 성과를 내며, 이는 IoD 프레임워크에서 개인 데이터 중심의 깊은 연구 딥 리서치가 실현 가능하다는 것을 보여줍니다. 이로 인해 보다 신뢰할 수 있고 재사용 가능한 과학적 발견을 위한 길이 열리게 됩니다.



### MetaSynth: Multi-Agent Metadata Generation from Implicit Feedback in Black-Box Systems (https://arxiv.org/abs/2510.01523)
Comments:
          NeurIPS Workshop LAW

- **What's New**: 이번 논문에서는 MetaSynth라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 암흑상자(black-box) 검색 엔진에서의 간접적인 피드백을 활용하여 메타 데이터 최적화를 목표로 합니다. 이를 통해 기존의 방법들이 가지는 한계점을 극복하고, 효과적이며 직접적인 데이터 활용이 가능합니다. 또한, 이 연구는 최적화의 기회를 넓히는 새로운 패러다임을 제시합니다.

- **Technical Details**: MetaSynth는 다중 에이전트(multi-agent) 검색 보강 생성 방식을 사용하여, 성공적인 예제를 수집하여 학습합니다. 이 시스템은 검색 엔진의 랭킹과 클릭 선호도를 반영하는 지식 기반의 더미(라이브러리)를 개발하며, 적절한 쿼리를 생성하여 관련 정보를 검색합니다. 이는 결과의 적합성, 홍보 강도, 정책 준수 등을 평가하는 모듈을 통해 반복적으로 개선됩니다.

- **Performance Highlights**: MetaSynth는 실험에서 NDCG, MRR 및 각종 랭킹 메트릭에서 강력한 기준선들보다 우수한 성능을 보였습니다. 더욱이 대규모 A/B 테스트에서는 10.26%의 CTR 증가와 7.51%의 클릭 수 증가를 확인했습니다. 이러한 결과는 MetaSynth가 실제 환경에서도 효과적으로 작용할 수 있음을 입증합니다.



### Optimal signals assignment for eBay View Item pag (https://arxiv.org/abs/2510.01198)
Comments:
          Accepted at the CONSEQUENCES 2025 workshop, co-located with ACM RecSys 2025

- **What's New**: 본 논문에서는 eBay의 View Item (VI) 페이지에 신호(signals)를 최적화하여 배치하기 위한 두 가지 통계 모델 접근법을 제안합니다. 이 신호들은 사용자가 상품에 대한 추가 정보를 얻도록 돕고, 구매를 촉진하기 위해 사용됩니다. A/B 테스트를 통해 두 접근법 모두 비즈니스 지표에서 유의미한 증가를 이끌어냈습니다.

- **Technical Details**: 제안된 모델은 주로 XGBoost를 사용하는 회귀 학습(retrospective-learning)과 전환 가능성 추정기(conversion likelihood estimator)로 구성됩니다. 이 모델들은 각각 1-2개의 신호를 정해진 장소에 할당하며, 신호 간의 상관관계와 미세한 상승률로 인해 모델 훈련 및 평가를 위한 대규모 데이터가 필요합니다. 또한, 모델 성능 평가를 위한 오프라인 메트릭도 개발하였습니다.

- **Performance Highlights**: 모델은 빠른 응답 시간(SLA)을 유지하며, 통계적인 유의성을 가지는 성능 개선 결과를 보였습니다. 특히 두 접근법 모두 구매율 또는 GMV와 같은 비즈니스 지표 증가에 기여하였습니다. 이러한 모델은 VI 페이지의 신호 배치 최적화를 통해 사용자 참여도와 구매 효율성을 높일 수 있습니다.



### Are LLMs ready to help non-expert users to make charts of official statistics data? (https://arxiv.org/abs/2510.01197)
- **What's New**: 이 연구는 공공 데이터의 접근성과 시각적 정보 표현의 필요성을 강조하며, 현재의 Generative AI 모델이 사용자 질의에 따라 적절한 데이터를 식별하고 차트를 자동으로 생성할 수 있는지를 평가합니다. 이를 위해 연구자들은 네덜란드 통계청의 다양한 공공 데이터를 수집하고, 8개의 최신 대형 언어 모델(LLMs)의 성능을 비교했습니다. 이 모델들은 데이터 테이블을 식별하고 필요한 데이터 조작과 시각화를 자율적으로 수행할 수 있는지에 대한 평가를 받았습니다.

- **Technical Details**: 연구에서는 데이터 검색 및 전처리, 코드 품질, 시각적 표현 등 세 가지 차원을 기반으로 한 새로운 평가 프레임워크를 제안합니다. 각 LLM의 성능을 22개의 이진 질문을 통해 평가하였으며, 이는 시각적 명확성, 데이터 정확성 및 코드의 견고성을 포함합니다. LLM은 사용자 질의를 올바르게 해석하고, 적절한 데이터를 식별하며, 적합한 차트 유형을 선택하고, 시각화 모범 사례를 준수하는 코드 생성을 요구받습니다.

- **Performance Highlights**: 연구 결과에 따르면, 데이터의 정확한 위치 찾기 및 처리와 같은 단계가 가장 큰 도전 과제가 되며, LLM은 명시적 지침 없이 시각화 모범 사례를 잘 구현하지 못하는 경향이 있음을 보여줍니다. 그러나 효과적인 차트 디자인에 대한 정보가 보완되었을 때 모델은 시각적 표현 점수에서 현저한 개선을 나타냈습니다. 또한, 자기 평가를 통한 반복적 접근이 모든 평가 차원에서 우수한 성능으로 이어졌습니다.



### Location Matters: Leveraging Multi-Resolution Geo-Embeddings for Housing Search (https://arxiv.org/abs/2510.01196)
Comments:
          Accepted to RecSys 2025 (industry track)

- **What's New**: QuintoAndar Group는 라틴 아메리카의 가장 큰 주택 플랫폼으로, 임대 및 판매 시장에 혁신을 가져오고 있습니다. 본 연구에서는 사용자 추천의 효과성을 높이기 위해 주택 추천에 대한 공간적 정보와 함께 지리적 첨두 구조를 통합한 새로운 방법을 제안합니다. 다양한 도시의 주택을 탐색하는 사용자들이 직면하는 문제를 해결하기 위한 방안으로 지리 정보를 활용하는 접근 방식이 주목받고 있습니다.

- **Technical Details**: 본 연구에서는 공간 인식(deep learning) 증강을 위해 다중 해상도 H3 임베딩을 통합한 두 타워 신경망 아키텍처를 개발했습니다. H3는 위도와 경도를 64비트 형식으로 변환하여 다양한 해상도로 지역 정보를 제공합니다. 모델은 유저와 주택 간의 상호작용 데이터에 기반하여 트레이닝되며, 대규모 데이터를 통해 고유 사용자 선호도와 지리적 맥락을 정확하게 통합하는 것을 목표로 합니다.

- **Performance Highlights**: 시뮬레이션 결과, 기존 메트릭스 분해 모델에 비해 두 타워 모델이 정보의 다양성(Information Abundance)을 158% 증가시켰으며, 임대 흐름(Rent-Flow) 동향도 84% 향상시켰습니다. 다중 해상도 H3 임베딩을 추가함에 따라 정보의 다양성은 또다시 40% 증가하였으며, 임대 흐름 산출은 85% 향상되었습니다. 이러한 결과는 공간적 맥락의 풍부함이 추천 품질을 크게 향상시킨다는 것을 시사합니다.



### Comparison of Unsupervised Metrics for Evaluating Judicial Decision Extraction (https://arxiv.org/abs/2510.01792)
Comments:
          28 pages

- **What's New**: 이번 연구는 러시아의 1,000건의 익명 법원 판결문에서 7개의 의미적 블록을 추출하는데 필요한 16개의 비지도 학습 메트릭을 평가합니다. 이러한 메트릭은 문서 기반, 의미적, 구조적, 법률 특정 카테고리로 분류되며, 사전 주석이 없는 상태에서 작동합니다. 연구 결과, Term Frequency Coherence와 Coverage Ratio/Block Completeness가 전문가 평점과 가장 잘 일치하는 것으로 나타났습니다.

- **Technical Details**: 연구 방법론에서는 비지도 메트릭을 사용하여 법원 판결문 추출 품질을 평가하며, 이는 JSON 형식으로 구성된 문서에서 수행됩니다. 각 판결문은 원본 텍스트와 사전 세분화된 JSON 객체 형태로 제공되어, 비지도 메트릭을 계산하는 기준(reference extraction) 역할을 합니다. 평가에 참여한 법률 전문가들은 1-5 Likert 척도를 사용해 각 블록을 평가하였으며, 서로 간의 신뢰성 평가를 위해 ICC를 사용했습니다.

- **Performance Highlights**: 연구는 법률 텍스트 추출을 위한 비지도 평가 메트릭이 아직까지 개발이 미흡하고 법률 특유의 뉘앙스를 포착하지 못할 가능성이 있음을 강조합니다. 전문가 평가 결과, Court evaluation of evidence 블록에서는 ICC 값이 0.86으로 높은 일치를 보였지만, Court decision 블록에서는 0.70으로 다소 낮은 수치를 기록했습니다. 이러한 결과는 법적 맥락에서 기술이 사람의 판단을 완전히 대체할 수는 없음을 시사합니다.



### From Videos to Indexed Knowledge Graphs -- Framework to Marry Methods for Multimodal Content Analysis and Understanding (https://arxiv.org/abs/2510.01513)
- **What's New**: 이번 논문에서는 영상 데이터를 분석하기 위한 새로운 프레임워크를 제시합니다. 이는 다양한 pre-trained 모델을 통합하여 multi-modal 콘텐츠 분석을 위한 파이프라인을 효율적으로 프로토타입화할 수 있는 방법을 제공합니다. 특히, 지속적인 학습(continual learning)과 동적 지식 통합을 지원하는 지식 그래프(knowledge graph) 형식을 통해 영상의 정보를 쿼리할 수 있는 구조로 변환합니다.

- **Technical Details**: 연구 방법론은 세 가지 단계로 구성되어 있습니다. 첫 번째로, 최적화된 pre-trained 모델 조합을 실험하고 open-source 모델과 쉽게 결합하여 영상과 같은 시계열 다중 모드 데이터를 처리할 수 있는 프레임워크를 구축합니다. 두 번째 단계에서는 영상을 반구조적 데이터 형식인 'VideoKnowledgeBase'로 변환하는 파이프라인을 설계합니다. 마지막으로, 생성된 VideoKnowledgeBase를 쿼리 가능하고 확장 가능한 Video Knowledge Graphs로 변환하는 알고리즘을 설계합니다.

- **Performance Highlights**: 이 연구의 주요 성과는 다양한 pre-trained 모델을 통합하고 결합하여 multi-modal 콘텐츠를 이해하고 분석하기 위한 파이프라인을 구축하는 것입니다. 또한, 비디오 데이터베이스에서 정보를 쿼리할 수 있는 방법을 새롭게 제안하고, 새로운 도메인 지식을 추가할 수 있는 프로토타입 소프트웨어를 구현했습니다. 이러한 결과는 영상 분석, 이해 및 검색의 효율성을 크게 향상시킬 것으로 기대됩니다.



### LLM-based Multi-Agent Blackboard System for Information Discovery in Data Scienc (https://arxiv.org/abs/2510.01285)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 발전에 따른 데이터 과학에서의 새로운 기회를 제시합니다. 특히, 기존의 데이터 레이크(data lake) 내에서 관련 데이터를 발견하는 데 어려움이 있는 방법론의 한계를 극복하기 위한 새로운 다중 에이전트 통신 패러다임을 제안합니다. 이 구조는 기존의 중앙 제어기 의존성을 없애고 에이전트들이 자율적으로 기여할 수 있도록 합니다.

- **Technical Details**: 우리는 전통적인 AI 모델의 블랙보드 아키텍처에서 영감을 얻어, 중앙 에이전트가 요청을 공유 블랙보드에 게시하고 여러 하위 에이전트가 자신의 능력에 따라 응답하는 프레임워크를 설계했습니다. 이 방식은 각 하위 에이전트의 전문 지식에 대한 사전 지식이 필요 없으며, 데이터 레이크에서 관련 파일을 식별하는데 효과적임을 보여주었습니다. 세 가지 벤치마크(KramaBench, DS-Bench, DA-Code)를 대상으로 평가하여, 블랙보드 구조가 상대적으로 13%에서 57%까지의 성능 향상을 보임을 확인했습니다.

- **Performance Highlights**: 실험 결과, 블랙보드 아키텍처는 기존의 RAG 및 마스터-슬레이브 다중 에이전트 패러다임보다 우수한 성과를 보였으며, F1 점수에서도 최대 9% 향상되었습니다. 이 연구는 블랙보드 패러다임이 다중 에이전트 시스템을 위한 확장 가능하고 일반화된 통신 프레임워크로 자리잡을 수 있는 가능성을 보여줍니다. 실제 적용 사례를 통해 이 시스템의 효율적인 문제 해결 능력을 강조하였습니다.



New uploads on arXiv(cs.CV)

### Optimal Control Meets Flow Matching: A Principled Route to Multi-Subject Fidelity (https://arxiv.org/abs/2510.02315)
Comments:
          Code: this https URL

- **What's New**: 이번 논문에서는 다중 주제에 대한 신뢰성을 높이기 위한 첫 번째 이론적 프레임워크를 제안합니다. 기존의 Text-to-image (T2I) 모델들이 다중 주제 프롬프트에서 나타내는 문제를 해결하기 위해, 주제의 분리를 스토캐스틱 최적 제어(stochastic optimal control) 문제로 모델링합니다. 이를 통해 최적화 가능한 목표를 도출하고, 새로운 알고리즘과 접근 방식을 제시합니다.

- **Technical Details**: 본 논문은 세 가지 주요 기술적 요소를 소개합니다. 첫째, T2I 모델의 다중 주제 사양에서 발생하는 문제들을 풀기 위한 Test-time controller와 Adjoint Matching 방법을 제안합니다. 이러한 기술들은 전달된 데이터 분포(data distribution)와 기본 모델의 스타일을 유지하면서도, 주제의 분리를 최적화하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘들은 Stable Diffusion 3.5, FLUX, Stable Diffusion XL 등 여러 현대 FM 모델에서 다중 주제의 정합성을 지속적으로 개선하는 것으로 나타났습니다. 특히, FOCUS(Flow Optimal Control for Unentangled Subjects)라는 방법론이 제안되어 다중 주제 신뢰성을 높이며, 효율적인 GPU에서의 실행과 사용되는 프롬프트의 제한이 있음에도 불구하고 일반화되는 성과를 보여줍니다.



### StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions (https://arxiv.org/abs/2510.02314)
Comments:
          ICCV 2025. Project page: this https URL

- **What's New**: 이 연구에서는 3D Gaussian Splatting (3DGS)의 데이터 오염 공격에 대한 새로운 방식을 제안합니다. 기존의 방법들은 주로 Neural Radiance Fields (NeRF)에 집중되었으나, 본 연구는 3DGS의 고유한 취약점을 분석하고 이를 타겟으로 하는 것을 목표로 합니다. 이로 인해 3DGS의 시각적 일루젼(illusion) 삽입을 통한 공격 기법을 최초로 구현하였습니다.

- **Technical Details**: 제안된 방법은 Kernel Density Estimation (KDE)을 이용하여 초기 Gaussian 포인트 클라우드 내 저밀도 지역을 식별합니다. 이 저밀도 지역에 일루젼 객체의 포인트를 전략적으로 배치하여 타겟 뷰에서 명확하게 보이도록 하면서도 다른 무고한 뷰에는 최소한의 영향을 미칩니다. 또한, 우리는 다중 뷰 일관성(multi-view consistency)을 방해하기 위해 적응형 노이즈 전략을 도입했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 여러 기존 기법 compared 하여 우수한 성능을 보임을 입증했습니다. 이 연구는 3DGS에 대한 데이터 오염 공격의 새로운 접근 방식을 제시하며, 향후 연구를 위한 객관적 벤치마킹 프로토콜을 개발하는 데 기여할 것입니다. 우리의 결과는 3DGS의 보안 취약성을 이해하고 해결하는 데 중요한 기초 자료로 사용될 수 있습니다.



### Clink! Chop! Thud! -- Learning Object Sounds from Real-World Interactions (https://arxiv.org/abs/2510.02313)
Comments:
          ICCV 2025. Project page: this https URL

- **What's New**: 본 연구에서는 물체 간 상호작용에서 발생하는 소리를 식별하기 위한 sounding object detection 작업을 소개합니다. 인간의 지각에서 영감을 받아, 저자들은 일상 생활에서 촬영된 egocentric 비디오를 사용해 모델이 소리와 관련된 물체를 직접적으로 연결할 수 있는 능력을 평가합니다. 자동으로 세그멘테이션 마스크를 생성하는 파이프라인을 개발하여 객체 중심 접근 방식을 촉진함으로써 모델의 훈련 중 주의를 기울여야 할 가장 유익한 영역에 집중할 수 있게 합니다.

- **Technical Details**: 제안된 방법은 멀티모달 객체 인식 프레임워크를 이용해, 객체 상호작용에서 발생하는 소리와 해당 물체 간의 고유한 상관관계를 학습하는 것입니다. 이를 위해, 저자들은 Ego4D와 Epic Kitchens로부터 수집된 대규모의 egocentric 비디오 데이터를 활용하여, 장기 사용되는 객체와 그 상호작용을 포함합니다. 또한, slot attention 모델을 통해 시각 표현의 강한 객체 우선순위를 부여하고, 물체 상호작용에 대한 집중을 높이는 데 기여합니다.

- **Performance Highlights**: 제안된 접근 방식은 기존의 멀티모달 작업들에서 강력한 성능을 보여줍니다. 특히, sounding object detection과 sounding action discovery 작업에서 최첨단 성과를 달성하며, 이로 인해 모델이 고유한 물체 특성과 소리 간의 관계를 이해하는 데 효과적임을 입증합니다. 또한, 다양한 객체 및 상호작용의 장점을 잘 활용한 결과로, 일상 생활에서의 다양한 소리 상호작용을 보다 잘 학습할 수 있는 가능성을 열어두었습니다.



### Inferring Dynamic Physical Properties from Video Foundation Models (https://arxiv.org/abs/2510.02311)
- **What's New**: 본 연구에서는 비디오로부터 동적인 물리적 특성을 예측하는 작업을 연구합니다. 특히 튀는 물체의 탄성, 흐르는 액체의 점도, 표면에서 미끄러지는 물체의 동적 마찰 계수와 같은 여러 물리적 속성을 다룹니다. 이를 위해 새롭게 생성한 PhysVid라는 비디오 데이터셋을 도입하여 동적 물리 속성을 평가하고, 기존 데이터셋의 부족함을 보완하고자 합니다.

- **Technical Details**: PhysVid 데이터셋은 합성 비디오와 실제 비디오를 혼합하여 구성되며, 각각의 비디오는 물리적 속성 값으로 주석이 달려 있습니다. 연구진은 세 가지 방법을 사용하여 비디오로부터 물리적 특성을 추정하는 접근 방식(오라클 메서드, 간단한 읽기 메커니즘, Multi-modal Large Language Models)도 제안합니다. 오라클 메서드는 해당 속성을 직접 반영하는 시각적 단서를 사용하는 고급 기법입니다.

- **Performance Highlights**: 비디오 기반 모델들은 생성 및 자기 지도 방식으로 훈련된 모델들이 오라클 메서드보다는 성능이 떨어지지만, 유사한 수준의 성능을 달성했음을 보여줍니다. Multi-modal Large Language Models는 다른 모델들에 비해 현재 성능이 낮지만, 적절한 프로밍 방법을 통해 성능 향상이 가능함을 시사합니다. 이 연구는 물리적 이해도를 평가하는 데 있어 기존 연구의 한계를 넘어서기 위한 중요한 기초 자료가 될 것입니다.



### NoiseShift: Resolution-Aware Noise Recalibration for Better Low-Resolution Image Generation (https://arxiv.org/abs/2510.02307)
- **What's New**: 최근의 연구에서 고정된 해상도 집합에 대해 훈련된 텍스트-이미지 확산 모델이 낮은 해상도의 이미지 생성 시 일반화하지 못하는 문제를 다루었습니다. 저자들은 노이즈 스케줄러의 인지적 효과가 해상도에 따라 다르다는 점에서 문제의 큰 원인을 찾았습니다. 이를 해결하기 위해 NoiseShift라는 훈련 없이 사용할 수 있는 방법을 제안하여, 해상도 크기에 따라 노이즈 레벨을 재조정함으로써 품질을 향상시킵니다.

- **Technical Details**: NoiseShift는 기존 모델 아키텍처나 샘플링 일정을 변경하지 않고도 사용할 수 있는 경량의 방법입니다. 이 방법은 다소 연구 결과를 바탕으로 다소 정교한 그리드 검색을 통해 훈련 시 사용된 노이즈 분포에 맞는 조건부 값을 식별하여 정확한 노이즈 조정의 과정을 통해 이루어집니다. 이로 인해 낮은 해상도에서의 인지적 노이즈의 감소가 이루어지고, 미세한 세부 사항을 보존하는 데 기여합니다.

- **Performance Highlights**: NoiseShift는 Stable Diffusion 3, 3.5, 그리고 Flux-Dev 모델에 적용했을 때 낮은 해상도에서의 이미지 품질을 상당히 향상시켰습니다. LAION-COCO 데이터셋에서 NoiseShift는 SD3.5의 FID 점수를 평균 15.89% 개선했으며, CelebA에서는 10.36% 개선의 효과를 보였습니다. 이 결과들은 NoiseShift가 해상도 의존적 아티팩트를 완화하고 낮은 해상도 이미지 생성의 품질을 향상시키는 데 효과적임을 보여줍니다.



### VideoNSA: Native Sparse Attention Scales Video Understanding (https://arxiv.org/abs/2510.02295)
Comments:
          Project Page: this https URL, Code: this https URL

- **What's New**: 이 논문에서는 비디오와 언어 모델을 위한 Native Sparse Attention (NSA)를 채택하여 VideoNSA라는 새로운 접근 방식을 제안합니다. 기존 비디오-언어 모델들의 맥락 길이가 제한된 문제를 해결하기 위해, 이 모델은 216K 비디오 지침 데이터셋에서 엔드-투-엔드(end-to-end) 훈련을 통해 Qwen2.5-VL을 조정합니다. 특히, 하드웨어에 최적화된 하이브리드 주의(attention) 접근 방식을 사용하여 텍스트에서는 밀집 주의를, 비디오에는 NSA를 적용합니다.

- **Technical Details**: VideoNSA는 세 가지 보완적인 캐시 분기를 통합한 배우는 하드웨어 인식 sparse attention 메커니즘을 사용합니다. 여기에는 Token Compression (CMP), Token Selection (SLC), Sliding Window (SWA) 분기가 포함됩니다. 이 방식은 모델이 특정 작업에 필요한 경로만을 유지하여 효율성을 극대화합니다. 특히, VideoNSA는 128K 컨텍스트 길이에서 성능 향상을 이루며, 작동 방식에서 여러 중요한 발견 사항들을 제시합니다.

- **Performance Highlights**: VideoNSA는 긴 비디오 이해(long-video understanding) 및 시간적 추론(temporal reasoning)에서 기존 방법보다 개선된 성능을 보였습니다. 또한, 다양한 실험 결과에 따라 데이터 세트의 길이를 초과하여 효과적으로 확장할 수 있는 잠재력을 보여줍니다. 가장 흥미로운 점은, 이 모델이 학습 가능한 sparse attention 가중치를 통해 다양한 태스크에서 동적인 주의 sink 행동을 유도하여 깊은 층에서의 선택 및 슬라이딩 윈도우 분기의 중요성을 감소시킨다는 것입니다.



### MultiModal Action Conditioned Video Generation (https://arxiv.org/abs/2510.02287)
- **What's New**: 이번 연구에서는 기존의 비디오 모델들이 미세한 제어 능력이 부족하다는 문제를 해결하기 위해, 다중 감각 신호(multisensory signals)를 통합하여 생성 시뮬레이션(generative simulation)에서 세밀한 반응을 가능하게 하는 방법을 제안합니다. 근본적으로, 가정용 로봇이 인간처럼 정교하게 동작하고 안전하게 작업을 수행하기 위해서는 다재다능한 감각 시스템이 필요합니다. 연구는 주요 감각 신경(kinesthesia, proprioception, force haptics, muscle activation)을 활용하여 기존 시뮬레이터의 한계를 극복하고자 합니다.

- **Technical Details**: 연구의 핵심은 세 감각 신호를 효과적으로 학습하고 이를 생성 모델에 통합하는 것입니다. 우리는 다중 감각 행동 표현(multi-sensory action representation)을 학습하는 다중 모달(feature extraction) 접근 방식을 제안하며, 이를 통해 각 모달의 고유한 정보를 유지하면서도 공유된 표현 공간에서 정렬할 수 있도록 설계했습니다. 또한, 액션 궤적의 인코딩을 더 맥락적이고 원인-결과를 인식하도록 조정하기 위한 일반적인 정규화 기법을 도입했습니다.

- **Performance Highlights**: 실험 결과, 제안된 다중 감각 접근법이 정확성을 36% 향상시키고 시간적 일관성을 16% 개선하는 데 기여하는 것으로 나타났습니다. 또한, 다양한 다운스트림 애플리케이션과의 비교를 통해 우리의 접근법이 실제 환경에서의 유용성과 실용성을 입증했습니다. 이와 같은 통합적 연구는 앞으로의 정책 최적화(policy optimization)와 계획(planning) 분야에서도 활용될 가능성을 보여줍니다.



### Learning to Generate Object Interactions with Physics-Guided Video Diffusion (https://arxiv.org/abs/2510.02284)
- **What's New**: KineMask는 물리 기반 비디오 생성 기능을 개선하기 위한 새로운 접근법으로, 강체체의 제어 및 상호작용을 현실감 있게 수행할 수 있도록 합니다. 이 방법은 단일 이미지와 특정 객체 속도를 제공받아 미래의 물체 상호작용을 예측하는 비디오를 생성합니다. 이 연구는 또한 저수준 모션 제어와 고수준 텍스트 조건화를 통합하여 복잡한 동적 현상을 효과적으로 합성할 수 있도록 합니다.

- **Technical Details**: KineMask는 물리적으로 유도된 비디오 생성을 위한 프레임워크로, 객체의 방향과 속도 같은 저수준 운동 제어를 제공합니다. 이를 통해 모델은 객체 상호작용을 추론하며, 두 단계의 훈련 전략을 사용하여 객체 마스크를 통해 미래의 모션 감독을 점진적으로 제거합니다. KineMask는 시뮬레이터에서 생성된 비디오를 기반으로 훈련되며 물리적으로 유효한 역학과 명확한 객체 상호작용을 캡처합니다.

- **Performance Highlights**: KineMask는 기존의 동영상 확산 모델들과 비교하여 뛰어난 객체 상호작용을 보여줍니다. 광범위한 실험 결과, KineMask는 유사한 크기의 최신 모델들 대비 강력한 개선을 이끌어냈고, ablation 연구를 통해 저수준과 고수준 제어의 통합 중요성을 강조했습니다. 추가적으로, KineMask는 복잡한 상호작용의 일반화를 통해 실제 장면에서의 비디오 생성에서 두드러진 성능 향상을 보여줍니다.



### Self-Forcing++: Towards Minute-Scale High-Quality Video Generation (https://arxiv.org/abs/2510.02283)
Comments:
          preprint

- **What's New**: 이 논문에서는 긴 비디오 생성(long video generation)의 품질 저하를 완화하기 위한 간단하면서도 효과적인 접근 방식을 제안합니다. 기존의 비디오 생성 모델이 긴 비디오를 생성하는 데 필요한 감독(supervision) 없이도 학생 모델(student model)을 지도하는 방법을 중심으로 합니다. 이 방법은 교사 모델(teacher model)의 지식을 활용하여 자신이 생성한 긴 비디오의 샘플링한 세그먼트를 학생 모델을 안내하는 방식으로 동작합니다.

- **Technical Details**: 저자들은 학생 모델이 긴 비디오 생성에 있어 발생하는 오류가 누적되는 문제를 해결하기 위해, 시간적 일관성(temporal consistency)을 유지하며 최대 20배 긴 비디오 길이를 확장할 수 있는 방법을 고안하였습니다. 이러한 방식은 이전 방법들이 갖고 있는 오버 익스포저(over-exposure) 및 오류 누적(error-accumulation) 문제를 피하면서도 더 많은 프레임을 재계산하지 않습니다. 실험 결과, 이 방법은 4분 15초 길이의 비디오 생성이 가능함을 입증하며, 이는 기본 모델의 위치 임베딩(position embedding)으로 지원되는 최대 범위의 99.9%에 해당합니다.

- **Performance Highlights**: 제안된 방법은 기준(baseline) 방법들에 비해 충실도(fidelity)와 일관성(consistency) 모두에서 월등한 성능을 보이고 있습니다. 저자들은 표준 벤치마크(benchmark)와 자신들이 제안한 개선된 벤치마크에서 실험을 수행하여 입증된 결과를 제공합니다. 또한, 이 논문에서 시연한 긴 지평선의 비디오(demo of long-horizon videos)는 웹 링크에서 확인할 수 있습니다.



### VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL (https://arxiv.org/abs/2510.02282)
- **What's New**: 새로운 AI-생성 비디오의 발전에 따라 VidGuard-R1이 등장했습니다. 이는 멀티모달 대형 언어 모델(MLLM)을 활용하여 비디오의 진위를 판별하는 최초의 도구입니다. 해당 모델은 그룹 상대 정책 최적화(GRPO)를 통해 학습하여, 비디오의 정확한 판단과 더불어 해석 가능한 설명을 제공할 수 있습니다.

- **Technical Details**: VidGuard-R1은 140,000개의 실제 및 AI-생성 비디오로 구성된 고난이도 데이터셋을 기반으로 하고 있습니다. 이 모델은 두 개의 보상 모델을 통해 시간 아티팩트와 생성 복잡성을 평가하여 GRPO로 세밀하게 조정됩니다. 다중 단계 확산을 통한 생성 비디오에서 고급 요인을 끌어내기를 목적으로 하며, 이는 더욱 향상된 설명 능력을 보장합니다.

- **Performance Highlights**: VidGuard-R1은 제로샷 성능에서 최첨단 정확도를 달성하며, 추가 훈련을 통해 95% 이상의 정확도를 기록했습니다. 다양한 실험과 사례 연구는 VidGuard-R1이 예측 뒤에 있는 정확하고 해석 가능한 이유를 제시할 수 있음을 보여줍니다. 이러한 성능은 AI-생성 비디오 감지의 새로운 기준을 설정하는 데 기여할 것입니다.



### microCLIP: Unsupervised CLIP Adaptation via Coarse-Fine Token Fusion for Fine-Grained Image Classification (https://arxiv.org/abs/2510.02270)
- **What's New**: 이번 논문에서는 CLIP 기반의 비전-언어 모델(VLM)에서 미세한 이미지 분류를 위한 비지도 적응(unsupervised adaptation)의 한계를 극복하기 위해 $	extbf{microCLIP}$를 제안합니다. 기존의 접근법은 LLM(large language model)의 설명을 CLIP의 [CLS] 토큰과 정렬하는 방식이었으나, 이로 인해 공간적인 정밀성이 부족했습니다. $	extbf{microCLIP}$은 Saliency-Oriented Attention Pooling (SOAP)이라는 신기술을 통해 시각적 및 텍스트 표현을 동시에 정제합니다.

- **Technical Details**: 이 연구는 CLIP의 시각적 표현과 LLM으로부터 유래된 텍스트적인 선행 지식을 융합하는 비지도 자기 훈련 프레임워크를 개발했습니다. SOAP 메커니즘은 CLIP 패치 토큰을 기반으로 강도를 기반으로 한 쿼리를 생성하고, 이를 통해 compact한 [FG] 토큰을 풀링합니다. 이어서 TokenFusion 모듈을 사용하여 [FG] 토큰을 CLIP의 глобал [CLS] 토큰과 융합하여 coarse-fine 정렬을 수행합니다.

- **Performance Highlights**: 제안된 방법은 13개의 미세한 데이터셋에서 평균 2.90%의 정확도 향상을 달성했으며, 이는 미세한 적응을 요구하지 않는 경량 조정만으로 가능합니다. 연구 결과 이 방법이 CLIP의 잠재적인 미세 신호를 발견하는 데 도움을 준다는 것이 입증되었습니다. 또한, 샘플링된 지역적 주의 집중을 통해 클래스 정의에 중요한 로컬 의미를 지속적으로 강조하며 다양한 시나리오에서 효율적인 성능을 보였습니다.



### NeuroSwift: A Lightweight Cross-Subject Framework for fMRI Visual Reconstruction of Complex Scenes (https://arxiv.org/abs/2510.02266)
- **What's New**: 본 논문에서는 NeuroSwift라는 새로운 접근 방식을 제안합니다. 이는 저수준의 특징을 위한 AutoKL과 의미를 위한 CLIP을 결합하는 보조 장치를 통합합니다. 이 방법은 뇌의 시각적 자극을 재구성하는 데 있어 기존 방법의 한계를 극복하고, 단 한 시간의 훈련으로도 최첨단 성능을 달성할 수 있도록 합니다.

- **Technical Details**: NeuroSwift의 CLIP Adapter는 Stable Diffusion에서 생성된 이미지와 COCO 캡션을 조합하여 훈련됩니다. 이 장치는 뇌 신호의 저차원 표현을 세부적으로 재구성하는 AutoKL와 의미적 정보를 강조하는 CLIP을 통합하여, 뇌의 효율적인 의미 추출 방식을 모사합니다. 교차 주제 일반화에서는 한 명의 주체에서 미리 훈련한 후, 다른 주체에 대해 17%의 파라미터만 미세 조정하여 학습합니다.

- **Performance Highlights**: NeuroSwift는 단 3개의 NVIDIA RTX 4090 GPU에서 단 한 시간의 데이터로 기존 방법들을 초월하는 효율적인 시각적 재구성을 달성합니다. 실험 결과, 이 방법은 복잡한 시각적 장면에서 낮은 수준의 구조적 특징과 높은 수준의 의미를 모두 효과적으로 재구성하는 데 성공했습니다. 기존의 모델들은 고비용의 계산 자원과 높은 복잡도로 인해 이러한 재구성이 어려웠으며, NeuroSwift는 그런 한계를 극복하고 있습니다.



### Paving the Way Towards Kinematic Assessment Using Monocular Video: A Preclinical Benchmark of State-of-the-Art Deep-Learning-Based 3D Human Pose Estimators Against Inertial Sensors in Daily Living Activities (https://arxiv.org/abs/2510.02264)
Comments:
          All tables, graphs and figures generated can be obtained in the Zenodo repository complementary to this work: this https URL

- **What's New**: 이번 연구는 기계 학습(machine learning)과 착용 가능한 센서(wearable sensors)의 발전이 사람의 움직임을 전문 실험실 외부에서 캡처하고 분석할 수 있는 새롭고 유망한 기회를 제공한다고 소개합니다. VIDIMU 데이터셋을 활용하여, 일상적인 임상 관련 활동에서 비디오 카메라와 관성 측정 장치(inertial measurement units, IMUs)를 함께 사용하여 단일 비디오 기반으로 3D 인간 자세를 생성하는 모델을 비교하였습니다.

- **Technical Details**: 연구에서는 MotionAGFormer, MotionBERT, MMPose 2D-to-3D pose lifting과 NVIDIA BodyTrack와 같은 최신 심층학습(d deep learning) 프레임워크를 사용하여 도출된 관절 각도를 IMU 데이터로부터 계산된 관절 각도와 비교하였습니다. OpenSim의 역기구학(inverse kinematics) 기법을 사용하여 Human3.6M 데이터셋 형식에 따라 17개의 주요 관절 지점을 기반으로 분석을 진행했습니다. 이 초기 연구는 건강한 피실험자만을 대상으로 하여, 결과를 병리학적인 집단에 일반화할 수 없다는 한계가 있습니다.

- **Performance Highlights**: MotionAGFormer는 전체 RMSE(최소 제곱 오차) $9.27°c 4.80°$, MAE(평균 절대 오차) $7.86°c 4.18°$로 가장 우수한 성과를 보였으며, Pearson 상관계수($0.86 c 0.15$)와 결정계수($R^{2}$, $0.67 c 0.28$)에서도 가장 높은 점수를 기록하였습니다. 이 연구는 두 기술 모두 실외에서 생체역학적 평가(kinematic assessment)에 적합하다는 것을 보여주지만, 비용, 접근성, 정밀도 간의 주요 트레이드오프(trade-off)를 강조합니다. 또한 임상에서 유망한 생체역학적 데이터를 제공하는 비디오 모델과 IMU 기반 평가 사이의 차이를 규명하였습니다.



### From Frames to Clips: Efficient Key Clip Selection for Long-Form Video Understanding (https://arxiv.org/abs/2510.02262)
- **What's New**: 이 논문은 Video Large Language Models (VLMs)의 프레임 선택 문제를 새로운 관점에서 접근하여, 고립된 키 프레임( isolated key frames) 대신, 키 클립( key clips)이라는 짧고 시간적으로 일관된 세그먼트를 선택하는 방법을 제안합니다. 이러한 접근은 의미적 관련성과 지역적 시간 연속성( temporal continuity)을 보존하여 비디오 이해가 향상됩니다. 또한, 전통적인 샘플링 방법의 한계를 극복하고, 더욱 효율적인 입력을 제공하기 위해 적응형 해상도(adaptive resolution)를 도입합니다.

- **Technical Details**: 제안된 방법인 Frames-to-Clips (F2C)는 클립 길이를 늘리면서도 시각 토큰 수를 일정하게 유지할 수 있도록 해상도를 동적으로 조정합니다. 이 방법을 통해 시간적으로 일관된 클립을 선택하여 VLM의 성능을 높일 수 있으며, 이는 비디오 이해를 위한 메모리 병목 현상을 완화합니다. F2C는 추가적인 훈련 없이도 여러 비디오 벤치마크에서 기존 샘플링 방법보다 성능이 우수한 결과를 보여줍니다.

- **Performance Highlights**: 실험 결과, F2C는 Video-MME, LongVideoBench 및 MLVU 벤치마크에서 각각 최대 8.1%, 5.6%, 10.3%의 성능 향상을 나타냈습니다. 이는 시간적 일관성을 유지하는 키 클립 선택의 중요성을 강조하며, 실질적인 비디오 이해 애플리케이션에 VLM을 확장할 수 있는 효과적인 경로를 제공합니다. 이 논문은 비디오의 시간 정보를 효과적으로 통합하여 VLM의 가능성을 더욱 확장하는 데 기여합니다.



### DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing (https://arxiv.org/abs/2510.02253)
Comments:
          Preprint

- **What's New**: 이 논문에서는 DragFlow라는 새로운 드래그 기반 이미지 편집 프레임워크를 제안합니다. 이 프레임워크는 더 강력한 생성적 prior를 활용하여 편집 작업에서 현저한 성과를 보여줍니다. 이전의 UNet 기반 모델의 한계를 넘어, FLUX와 같은 새로운 모델들로부터의 이점을 최대한 활용합니다.

- **Technical Details**: DragFlow는 affine transformations를 통해 지역 기반 편집 패러다임을 도입하여 일관된 피쳐(supervision) 지침을 제공합니다. 또한, pretrained open-domain personalization adapters(예: IP-Adapter)를 통합하여 배경 품질을 유지하면서도 주제 일관성을 향상시킵니다. 다양한 작업의 모호성을 해결하기 위해 다중 모달 대형 언어 모델(MLLMs)을 사용합니다.

- **Performance Highlights**: DragFlow는 새로운 Region-based Dragging benchmark(ReD Bench)과 DragBench-DR에서 광범위한 실험을 통해 점 기반 및 지역 기반의 기존 기준을 뛰어넘는 성과를 보여줍니다. 이로 인해 드래그 기반 이미지 편집에서 새로운 최첨단(state-of-the-art)을 설정하였습니다. 코드와 데이터셋은 발표 시에 공개될 예정입니다.



### RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning (https://arxiv.org/abs/2510.02240)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 미세한 시각적 추론(fine-grained visual reasoning) 능력 향상을 위한 새로운 데이터셋 ReasonMap-Plus와 RL 프레임워크 RewardMap을 제안합니다. ReasonMap-Plus는 시각적 질문 응답(Visual Question Answering, VQA) 작업을 통해 밀집 보상 신호를 도입하여 효과적인 학습을 지원합니다. RewardMap은 시각적 이해와 추론 능력을 개선하는 다단계 RL 프레임워크로, 난이도 인식 보상 설계와 복잡한 작업으로 나아가는 훈련 전략을 통해 기존 방법의 한계를 극복합니다.

- **Technical Details**: ReasonMap-Plus는 VQA 작업에 바탕을 두고 있어 단순한 인식부터 복잡한 추론 작업까지 여러 난이도 단계로 구성되어 있습니다. RewardMap은 두 가지 주요 설계를 포함하고 있습니다: 하나는 작업 복잡성을 고려한 세부 보상 체계로, sparse 보상의 문제를 해결하고 더 많은 감독을 제공합니다; 다른 하나는 차례로 진행되는 RL 절차로, SFT보다 효과적인 시작 전략을 제공합니다. 각 단계의 보상은 접근하기 쉬운 밀집 보상을 포함하고 있어 효과적인 학습을 촉진합니다.

- **Performance Highlights**: 각 구성 요소가 일관된 성능 향상에 기여하며, 이들의 조합이 최상의 결과를 나타냅니다. RewardMap으로 훈련된 모델은 공간 추론, 미세한 시각적 추론 및 일반 작업에서 평균 3.47%의 성능 개선을 달성했습니다. 이러한 결과는 구조화된 시각적 작업에서 MLLMs의 능력을 강화하는 데 기여합니다.



### TempoControl: Temporal Attention Guidance for Text-to-Video Models (https://arxiv.org/abs/2510.02226)
Comments:
          Under Review

- **What's New**: 최근 생성 비디오 모델의 발전으로 자연어 프롬프트를 기반으로 한 고품질 비디오 생성이 가능해졌습니다. 하지만 이러한 모델은 개별 시각적 요소의 출현 시점을 지정할 수 있는 세분화된 시간 제어 부족이 있습니다. 이 연구에서는 TempoControl이라는 방법을 소개하여 추가 훈련이나 감독 없이 추론 시점에서 비주얼 개념을 시간적으로 정렬할 수 있도록 하였습니다.

- **Technical Details**: TempoControl은 텍스트-비디오 확산 모델의 주요 구성 요소 중 하나인 크로스 어텐션 맵을 활용하여 비주얼 컨셉의 타이밍을 유도하는 새로운 최적화 접근 방식을 적용합니다. 이 방법은 세 가지 보완 원칙(상관관계, 에너지, 엔트로피)을 통해 어텐션을 조정합니다. 또한, 모델 파라미터를 업데이트하지 않고 적절한 수준의 시간적 정렬을 달성하기 위해 몇 차례의 확률적 경량 하강법(SGD)을 적용합니다.

- **Performance Highlights**: TempoControl은 단일 및 다중 객체를 포함한 다양한 비디오 생성 응용 프로그램에서 효과성을 입증하였습니다. 특히, 객체의 시간 재배열 및 행동과 오디오 정렬 생성에서 뛰어난 성능을 보여주었습니다. 우리의 접근 방식은 추가 훈련 없이도 비디오 생성에서 외부 오디오 신호와의 정렬 가능성을 탐색할 수 있는 잠재력을 가지고 있습니다.



### MMDEW: Multipurpose Multiclass Density Estimation in the Wild (https://arxiv.org/abs/2510.02213)
Comments:
          8+1 pages, 4 figures, 5 tables

- **What's New**: 이 논문은 객체 밀도 맵 추정을 통해 밀집하거나 가려진 장면에서 객체 수를 효과적으로 추정하는 다중 범주(counting) 프레임워크를 제안합니다. Twins 피라미드 비전 변환기(Twins pyramid vision-transformer) 백본을 사용하고, 다중 클래스를 위한 특수한 카운팅 헤드를 통해 최신 멀티스케일 디코딩 접근법을 활용합니다. 또한, 카테고리 간의 간섭을 최소화하는 세분화 기반 카테고리 집중 모듈(Category Focus Module)을 추가하여 훈련 단계에서의 카테고리 간 크로스토크(inter-category cross-talk)를 억제합니다.

- **Technical Details**: 저자들은 다중 클래스를 지원하는 밀도 맵 추정을 위해 Twins-SVT 피라미드 비전 변환기 백본을 채택하고, 신고부의 각 클래스에 대해 분리된 밀도 맵을 학습하는 접근법을 사용합니다. 이와 함께 제안된 지역 손실 함수는 클래스 간 신호 간섭을 최적화하여 보다 우수한 성능을 달성합니다. 이 모델은 기존의 CNN 기반 방법과 비교할 때 다양한 밀도가 있는 객체를 효율적으로 식별할 수 있는 능력을 지녔습니다.

- **Performance Highlights**: 해당 방법은 VisDrone과 iSAID 벤치마크에서 이전의 다중 범주 군중 카운팅 접근법에 비해 MAE(Mean Absolute Error)가 각각 33%, 43%, 64% 감소하는 뛰어난 성능을 보여주었습니다. YOLOv11과의 비교를 통해 밀집한 장면에서 군중 카운팅 방법의 필요성을 강조하였고, 생물 다양성 모니터링 데이터셋에의 적용을 통해 보전 작업을 지원할 수 있는 잠재력을 시연했습니다.



### Cross-Breed Pig Identification Using Auricular Vein Pattern Recognition: A Machine Learning Approach for Small-Scale Farming Applications (https://arxiv.org/abs/2510.02197)
Comments:
          20 pages

- **What's New**: 본 연구는 비침습적인 생체인식(biometrics) 접근법을 통해 돼지의 귀 정맥 패턴을 이용한 개별 식별 시스템을 제안합니다. 기존의 돼지 식별 방법들이 항상 신뢰할 수 없고, 소규모 농부들에게 비현실적이었던 문제를 해결하기 위한 노력입니다. 800장의 귀 이미지를 사용하여 머신 러닝(Machine Learning) 모델을 통해 돼지를 98.12%의 정밀도로 성공적으로 식별할 수 있는 시스템을 만들었습니다.

- **Technical Details**: 제안된 기술은 여러 단계를 거쳐 이미지에서 귀 정맥 특성을 추출합니다. 첫 번째 단계로는 관심 지역(Region of Interest)을 분리하고, 이후 정맥의 구조적 및 공간적 특성을 분석하여 생체 서명을 생성합니다. 이러한 정맥 패턴 인식은 스마트폰을 사용하여 비침습적으로 이루어지며, 최종적으로 SVM(Support Vector Machine) 알고리즘을 사용하여 돼지를 식별합니다.

- **Performance Highlights**: 이 시스템은 전체 프로세스가 평균적으로 8.3초의 시간 안에 완료되며, 질병 예방 및 생산성 향상에서 중요한 역할을 할 것으로 기대됩니다. 돼지의 귀 정맥을 통한 식별 정확도가 98.12%에 달해 농업 관리의 디지털화를 가능하게 합니다. 따라서 이 연구는 자원 제약이 있는 농업 공동체에 정밀 농업의 이점을 확대하는 데 큰 기여를 할 수 있을 것으로 보입니다.



### GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation (https://arxiv.org/abs/2510.02186)
- **What's New**: 본 논문에서는 2D Vision-Language Models (VLMs)에서 3D semantic segmentation으로 특징을 전달할 때의 지속적인 문제를 다루고 있는 GeoPurify를 제안합니다. 이 모델은 2D에서 3D로 전이하는 과정에서 발생하는 기하학적 불일치를 해결하며, 필요한 대량의 주석이 있는 3D 데이터 없이 학습할 수 있습니다. 특히, 2D 특징에서 잠재적인 기하학적 정보를 활용하여 향상된 성능과 데이터 효율성을 달성하는 방법을 모색합니다.

- **Technical Details**: GeoPurify는 Geometric Contrastive Distillation이라는 새로운 프레임워크를 기반으로 하며, 이를 통해 학생 모델이 3D 지도 모델의 구조적 우선 정보를 활용하여 잡음이 많은 특징에서 잠재적인 기하학적 친화도를 학습합니다. 또한, Geometry-Guided Pooling 모듈을 통해 3D 포인트 클라우드를 더욱 정돈하여 일관된 표현을 생성합니다. GeoPurify는 기존 대량 수동 주석 없이도 소규모 비유형 3D 스캔 데이터로 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, GeoPurify는 주요 3D 벤치마크에서 기존 최첨단 방법들과 비교하여 동등한 또는 더 나은 성능을 달성했습니다. 특기할 만한 점은, 약 1.5%의 훈련 데이터만 사용하면서도 이러한 성능을 낼 수 있다는 점입니다. 이는 데이터 효율성 측면에서 큰 개선을 보여줍니다.



### Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting (https://arxiv.org/abs/2510.02155)
Comments:
          14 pages, video anomaly detection

- **What's New**: ASK-Hint는 비디오 이상 탐지를 위한 구조화된 프롬프트 프레임워크로, 기존의 추상적인 프롬프트의 한계를 극복하고 더 세밀한 인간-객체 상호작용과 행동의 의미에 기반한 프롬프트를 제안합니다. 이 방법은 이상 탐지에 있어 보다 정확하고 해석 가능한 추론을 유도하며, 프롬프트를 의미적으로 일관된 그룹(예: 폭력, 재산 범죄)으로 조직합니다. ASK-Hint는 UCF-Crime 및 XD-Violence 데이터셋에서 이전 방법들보다 AUC를 일관되게 개선하며, 해석 가능성 또한 제공합니다.

- **Technical Details**: ASK-Hint는 세 가지 주요 구성 요소를 가집니다: (1) 클래스별 프롬프트 구성, (2) 의미 있는 프롬프트 클러스터링 및 압축, (3) 설명 trace와 함께 구조화된 추론. 이러한 설계를 통해 프롬프트의 세분화된 행동 설명을 사용하여 VLM(vision-language model)의 추론 능력을 탐구할 수 있습니다. 또한, ASK-Hint는 행동 중심의 프롬프트를 통해 정확성과 효율성을 높입니다.

- **Performance Highlights**: ASK-Hint는 VLM을 고정(frozen) 상태로 유지하면서도 비디오 이상 탐지를 위한 훈련 없는 솔루션으로 강력한 일반화를 달성합니다. UCF-Crime 및 XD-Violence 데이터셋에서의 대규모 zero-shot 평가를 통해, ASK-Hint는 이전 최첨단 성과를 지속적으로 초과하며 해석 가능성과 견고한 일반화 능력을 입증했습니다. 이 결과는 프롬프트의 세분화가 비디오 이상 탐지에서 뛰어난 성능을 발휘하는 데 중요한 역할을 함을 보여줍니다.



### FRIEREN: Federated Learning with Vision-Language Regularization for Segmentation (https://arxiv.org/abs/2510.02114)
Comments:
          Master Thesis

- **What's New**: 이 논문은 새로운 과제인 Federated source-Free Domain Generalization (FFREEDG)를 제안합니다. 이는 서버에서 사전 학습된 모델을 클라이언트의 비표시 데이터만을 사용하여 학습하는 방법으로, 원본 데이터에 대한 접근 없이 진행됩니다. 이를 해결하기 위해 FRIEREN 프레임워크를 제안하며, 비전과 언어 모드를 통합하여 VFM의 지식을 활용합니다.

- **Technical Details**: FRIEREN은 CLIP 기반의 텍스트 임베딩을 통해 의미론적 불일치를 개선하는 비전-언어 디코더를 사용합니다. 또한, 약한 라벨에서 강한 라벨로의 일관성 학습 전략을 통해 로컬 훈련을 강화할 수 있습니다. 이 연구는 FFREEDG의 새로운 설정을 통해 클라이언트 스타일에 대한 사전 지식 없이도 강력한 일반화를 차지할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 FFREEDG가 기존의 도메인 일반화 및 적응 방법과 경쟁하는 성능을 달성했습니다. ACDC와 Cityscapes 데이터셋에서의 연구 결과는 이 프레임워크가 높은 효율성을 지니고 있으며, 반監督 및 비監督 학습 환경에서 탁월한 성능을 보입니다. 특히 반선택 환경에서는 완전 감독된 FL의 성능에 근접하며, 주석 필요성을 상당히 줄일 수 있음을 보여주었습니다.



### When Tracking Fails: Analyzing Failure Modes of SAM2 for Point-Based Tracking in Surgical Videos (https://arxiv.org/abs/2510.02100)
Comments:
          Accepted for publication in the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) Workshop on Collaborative Intelligence and Autonomy in Image-guided Surgery (COLAS), 2025

- **What's New**: AI 기반 기술이 수술 과정에서 사용자 입력을 최소화하면서도 효과적인 객체 추적을 가능하게 하는 비디오 객체 분할 (VOS) 모델인 SAM2의 가능성을 보여줍니다. 특히, 점 기반 추적(point-based tracking)은 비용 효율적인 방법으로 주목받고 있으나, 복잡한 수술 환경에서의 신뢰성과 실패 사례에 대해서는 잘 알려져 있지 않았습니다. 이 연구에서는 복강경 담낭 절제술(laparoscopic cholecystectomy) 비디오에서 SAM2의 점 기반 추적 성능을 체계적으로 분석하였습니다.

- **Technical Details**: SAM2.1 Hiera Large는 계층적 트랜스포머 아키텍처를 사용하는 최첨단 제로샷 세그멘테이션 모델로, 강력한 공간적 및 시간적 추론을 지원합니다. 이 연구에서는 CholecSeg8k 데이터세트를 활용하여 10개의 비디오 구간과 3개의 목표 물체를 선정하여 점 기반 추적 성능을 평가하였습니다. 다양한 점 선택 전략을 사용하여 추적 성능에 미치는 영향을 조사하고, 각 전략마다 평균 IoU 점수를 비교 분석합니다.

- **Performance Highlights**: 실험 결과, 점 기반 추적은 해부학적 목표에 대해서는 일관되게 세그멘테이션 마스크 기반 추적보다 저조한 성능을 보였습니다. 그러나 수술 도구인 그라스퍼(grasper)와 L-hook 전기 소작기에서는 경쟁력 있는 성능을 발휘하였고, 경우에 따라 세그멘테이션과 유사한 정확도를 달성했습니다. 중요한 실패 모드는 점 기반 추적이 해부학적 구조의 모호한 경계에서 힘들어 한다는 것이며, 이는 추적 성공의 핵심 요소 중 하나로 확인되었습니다.



### Mapping Historic Urban Footprints in France: Balancing Quality, Scalability and AI Techniques (https://arxiv.org/abs/2510.02097)
- **What's New**: 이 연구는 1970년 이전 프랑스의 역사적인 도시 확장에 대한 양적 분석의 한계를 극복하기 위해, 스캔 히스토(Scan Histo) 역사 지도 시리즈를 활용하여 도시 지역을 추출하는 스케일러블한 딥러닝 파이프라인을 개발했습니다. 이 과정에서 최초로 접근 가능한 국가 규모의 도시 발자국 데이터셋이 만들어졌습니다. 이는 역사적 지도에서 발생하는 고유의 복잡성을 처리하기 위한 새로운 듀얼 패스 U-Net 접근방식이 핵심 혁신으로 작용했습니다.

- **Technical Details**: 듀얼 패스 U-Net 접근법은 두 단계로 구성됩니다. 첫 번째 단계는 초기 데이터셋에 대해 훈련되어 혼란 지역, 즉 텍스트와 도로를 식별하여 타겟 형 데이터 증강을 안내하는 예비 지도를 생성합니다. 두 번째 단계는 정제된 데이터셋과 첫 번째 모델의 이진화된 출력을 사용하여 방사계 노이즈(radiometric noise)를 최소화하는 방법을 적용합니다.

- **Performance Highlights**: 최종적으로 이 방법은 프랑스 전역의 941개의 고해상도 타일을 처리하여 전체적인 정확도 73%를 달성했습니다. 이 연구는 다양한 도시 패턴을 효과적으로 파악하고 레이블과 윤곽선 같은 일반적인 아티팩트를 극복하는 데 성공했습니다. 덧붙여, 이 연구의 코드와 훈련 데이터셋, 결과로 도출된 국가 규모의 도시 래스터가 오픈 접근 방식으로 제공되어 장기 도시화 역학에 대한 향후 연구 지원을 기대하고 있습니다.



### VGDM: Vision-Guided Diffusion Model for Brain Tumor Detection and Segmentation (https://arxiv.org/abs/2510.02086)
- **What's New**: 본 논문에서는 비전 변환기(vision transformer)를 활용한 새로운 뇌종양 탐지 및 분할 프레임워크인 VGDM을 제안합니다. 기존의 U-Net 기반 모델의 한계를 극복하기 위해 변환기 기반의 확산(diffusion) 프로세스를 통합하여 전역적인 맥락을 고려합니다. 이는 더 나은 볼륨 정확도(volumetric accuracy)와 경계 정밀도(boundary precision)를 제공하며, 임상 환경에서의 확장성을 높입니다.

- **Technical Details**: VGDM은 MRI의 전체 볼륨에서 장거리 종속성(long-range dependencies)을 포착하기 위해 비전 변환기(backbone)를 사용합니다. 이러한 구조는 능동적인 자기 주의(self-attention) 메커니즘을 활용하여 구조적 세부정보를 복원하고 고정밀 경계 마스크(segmentation mask)를 생성합니다. 훈련 과정에서는 추론 잔차(noise)를 드는 마코프 체인(Markov chain)을 통해 학습하여, 데이터의 동적 변화를 모델링합니다.

- **Performance Highlights**: VGDM은 BraTS2020 데이터셋에서 테스트하여 U-Net 및 TransBTS와 같은 기존 모델 대비 일관된 성능 향상을 입증했습니다. 특히, Dice 유사도(Dice similarity)와 하우스도르프 거리(Hausdorff distance)에서 개선된 결과를 보여주며, 뇌종양 분석에 있어 더 신뢰할 수 있는 도구로서의 잠재력을 강조합니다.



### Zero-shot Human Pose Estimation using Diffusion-based Inverse solvers (https://arxiv.org/abs/2510.02043)
- **What's New**: 이 논문은 인체 자세 추정(pose estimation) 문제를 역문제(inverse problem)로 정의하고, 사용자 맞춤형 조정 없이 제로샷 제너럴리제이션(zero-shot generalization)을 가능하게 하는 알고리즘을 설계했습니다. 제안된 InPose 방법은 사전 훈련된 diffusion 모델을 사용하여 회전 측정(rotation measurements)만을 조건으로 하여 인체 자세를 생성적으로 추정하는 혁신적인 접근을 제공합니다. 기존 방법들은 사용자에 따라 위치 측정값(location measurements)이 크게 달라지는 문제를 겪었지만, InPose는 이러한 제약 없이 사용자별로 정확한 자세를 예측할 수 있습니다.

- **Technical Details**: InPose는 33개의 신체 관절(joints)에서 얻은 회전 및 위치 측정을 통해 전체 2222개의 관절을 추적합니다. 알고리즘은 사용자의 신체 크기에 따라 스케일-프리 포즈(scale-free pose)를 조정하여 최적의 자세를 추정합니다. 이 과정에서 확률론적 원리를 적용하여 자세의 가능성(likelihood)을 평가하고, 이는 디퓨전 노이즈 제거(diffusion denoising) 프로세스를 이끄는 가이드 역할을 합니다. 결과적으로, 사용자 맞춤형 3D 자세를 효과적으로 생성하는 방식입니다.

- **Performance Highlights**: InPose는 AMASS 데이터셋을 통해 다양한 신체 크기와 형태에서 뛰어난 일반화 결과를 보였습니다. 저자들은 각종 실험을 통해 기존 모델들보다 우수한 성능과 신뢰성을 입증했습니다. InPose는 다양한 환경에서 사용자에 대한 조정 없이도 일관된 결과를 지속적으로 생성함으로써 역문제 접근 방식의 유용성을 강조했습니다.



### GaussianMorphing: Mesh-Guided 3D Gaussians for Semantic-Aware Object Morphing (https://arxiv.org/abs/2510.02034)
Comments:
          Project page: this https URL

- **What's New**: GaussianMorphing은 다중 뷰 이미지를 기반으로 하는 새로운 세맨틱 인식 3D 형태 및 텍스처 변형 프레임워크입니다. 기존 접근 방식은 포인트 클라우드에 의존하거나 텍스처가 없는 데이터에 대해 미리 정의된 동형 매핑을 요구하여 한계를 드러냈습니다. 본 논문에서는 메쉬 안내 3D Gaussian Splatting(3DGS)을 활용하여 기하학 및 시각적 모델링에서의 고충실도를 제공합니다.

- **Technical Details**: 새로운 기법은 메쉬 패치에 3D Gaussian을 고정하여 일관된 변형을 보장하고, 토폴로지 인식 제약을 통해 텍스처 충실도를 보존하는 통합 변형 전략을 사용합니다. 본 프레임워크는 메쉬 토폴로지를 기하학적 우선 사항으로 활용하여 비지도 방식의 세맨틱 대응을 설정하며, 물리적으로 그럴듯한 포인트 궤적을 통해 구조적 온전성을 유지합니다. 이 통합 접근 방식은 변화 과정에서 지역 세부사항과 글로벌 세맨틱 일관성을 모두 보존합니다.

- **Performance Highlights**: 제안된 TexMorph 벤치마크에서 GaussianMorphing은 기존의 2D/3D 방법보다 뛰어난 성능을 보이며 색상 일관성 오류(ΔE)를 22.2% 감소시키고 EI를 26.2% 줄입니다. 다양한 시나리오, 복잡한 토폴로지 및 텍스처가 풍부한 객체들에서 강력한 성능을 보여줍니다. 이 기술은 고품질 3D 데이터에 대한 의존도를 줄이며 사용이 용이한 입력에 대한 원활한 시각적 결과를 달성합니다.



### kabr-tools: Automated Framework for Multi-Species Behavioral Monitoring (https://arxiv.org/abs/2510.02030)
Comments:
          31 pages

- **What's New**: 본 연구에서 소개하는 kabr-tools는 다양한 동물 행동 생태학을 위한 자동화된 다중 종 거동 모니터링을 위한 오픈소스 패키지입니다. 이 프레임워크는 드론을 이용한 비디오와 머신러닝 시스템을 통합하여 야생동물 영상에서 행동, 사회적 및 공간적 지표를 추출합니다. 기존의 접근법들보다 높은 행동 세분화와 데이터 수집의 용이성을 제공합니다.

- **Technical Details**: kabr-tools는 드론 기반 비디오 처리를 통해 초기 동영상 자료를 행동 주석이 달린 비디오로 변환하는 과정을 포함합니다. 연구에서는 Grevy의 얼룩말, 평원 얼룩말, 그리고 격자기린의 데이터를 수집하였으며, 전체 비디오 촬영과 육안 관찰을 동시에 수행했습니다. 이 방법론은 자동화, 생태적 타당성 및 최소한의 동물 방해를 우선시하며, 모듈화된 설계로 새로운 머신러닝 모델의 통합과 다양한 연구 맥락에 맞춰 조정할 수 있는 가능성을 제공합니다.

- **Performance Highlights**: 비교 연구를 통해 kabr-tools는 기존의 지상 관찰 방법보다 15%의 가시성 손실을 줄였고, 더 높은 정확도와 연속성으로 행동 전환을 포착했습니다. 이 연구에서 수집된 데이터는 자동화된 행동 모니터링을 통해 종별 행동 관리 전략을 분석하며, 혼합 종 무리 내에서의 공간적 분리를 확인했습니다. 이 단계에서 kabr-tools는 생태계 전반의 연구를 위한 강력한 도구로 자리매김하며 보존, 생물 다양성 연구 및 생태 모니터링 분야에 기여할 수 있습니다.



### LiLa-Net: Lightweight Latent LiDAR Autoencoder for 3D Point Cloud Reconstruction (https://arxiv.org/abs/2510.02028)
Comments:
          7 pages, 3 figures, 7 tables, Submitted to ICRA

- **What's New**: 이 연구에서는 LiLa-Net이라는 3D 자동 인코더 아키텍처를 제안합니다. 이 모델은 LiDAR의 포인트 클라우드만을 사용하여 실제 교통 환경에서 효율적인 특징을 인코딩합니다. LiLa-Net은 인코더 레이어 수를 줄이고, 기존 아키텍처보다 리소스를 적게 소모하면서도 원래 포인트 클라우드를 정확하게 재구성할 수 있는 효율적이고 대표적인 잠재 공간을 생성합니다. 또한, 스킵 연결과 잠재 인코딩 간의 정보 균형을 효과적으로 맞춰 재구성 품질을 향상시킵니다.

- **Technical Details**: LiLa-Net은 점 구름의 특성 추출 및 재구성을 위한 경량화된 종단 간 (end-to-end) 프레임워크입니다. 이 모델은 먼저 데이터 전처리를 진행한 후, 인코더 (encoder)를 통해 가장 관련성이 높은 특징을 추출하여 압축된 잠재 표현을 생성합니다. 이후 인코더의 스킵 연결과 함께 디코더 (decoder)로 입력되어 최종 포인트 클라우드를 재구성합니다. 연구를 위해 Universidad Carlos III de Madrid의 AMPL 연구실에서 자체 데이터 세트를 수집하였으며, 이를 통해 수집된 포인트 클라우드는 RANSAC 알고리즘을 사용해 지면 점을 제거하여 전처리됩니다.

- **Performance Highlights**: LiLa-Net은 교통 중심의 데이터 세트를 통해 검증된 후, 관련 없는 객체에 적용했을 때도 강한 적응성을 보여주었습니다. 이 모델은 포인트 클라우드를 활용하여 복잡한 교통 환경을 효과적으로 압축하고 재구성하며, 다른 데이터 세트에서 훈련 없이도 강력한 분류 성능을 달성합니다. 또한, 기존의 프리트레이닝(pretraining)이나 마스킹 전략을 제거하여 더 간단하고 빠른 훈련 파이프라인을 구축하였습니다.



### Generating Findings for Jaw Cysts in Dental Panoramic Radiographs Using GPT-4o: Building a Two-Stage Self-Correction Loop with Structured Output (SLSO) Framework (https://arxiv.org/abs/2510.02001)
Comments:
          Intended for submission to Scientific Reports

- **What's New**: 이번 연구에서는 OpenAI GPT-4o의 다중 모달(multimodal) 능력을 활용하여 치과 파노라마 방사선 사진에서 턱 낭종(jaw cyst) 발견을 자동 생성했습니다. 이를 통해 정확성을 높이기 위해 Self-correction Loop with Structured Output (SLSO) 프레임워크를 구축하였고, 그 효과성을 검증하였습니다.

- **Technical Details**: 22개의 턱 낭종 사례에 대해 10단계 과정을 구현하였습니다. 이 과정은 이미지 입력 및 분석, 구조화된 데이터 생성, 치아 번호 추출 및 일관성 확인, 불일치가 감지될 때 반복 재생성을 포함하며, 최종적으로 발견 생성 및 재구성과 일관성 검증 단계로 이루어졌습니다.

- **Performance Highlights**: 제안된 SLSO 프레임워크는 치아 번호, 치아 이동 및 뿌리 흡수에서 각각 66.9%, 33.3%, 28.6%의 향상률을 보였으며, 성공적인 사례에서는 최대 5번의 재생성 후 일관된 구조화된 출력을 얻었습니다. 그러나 데이터셋의 크기가 작아 통계적 유의미성에 도달하지 않았음에도, 전체적인 SLSO 프레임워크는 부정적인 발견 설명을 강화하고, 헛것(hallucinations)을 억제하며, 치아 번호 식별 정확성을 향상시켰습니다.



### Pure-Pass: Fine-Grained, Adaptive Masking for Dynamic Token-Mixing Routing in Lightweight Image Super-Resolution (https://arxiv.org/abs/2510.01997)
- **What's New**: 이 논문은 기존의 lightweight SR 방법의 장점을 통합하여 새로운 pixle level masking 메커니즘인 Pure-Pass (PP)를 제안합니다. PP는 고정된 색상 중심 점을 활용하여 각 픽셀을 식별하고 이를 통해 고가의 계산 작업에서 제외할 수 있도록 합니다. 이를 통해 ATD-light 모델에 통합되어, CAMixer보다 더 우수한 SR 성능을 발휘함을 보여줍니다.

- **Technical Details**: Pure-Pass는 정적 구조에서의 한계를 극복하고, 동적으로 입력 이미지의 복잡성에 따라 계산 절약량을 결정합니다. 이 메커니즘은 픽셀 단위에서 텍스처를 분석하기 위해 윈도우 주의를 분리하여 미세한 마스킹을 수행합니다. 또한, 교차-변환(fusion) 전략을 통해 공간적으로 유연한 마스킹을 구현합니다.

- **Performance Highlights**: 제안된 PP-ATD-light 모델은 CAMixer-ATD-light 모델에 비해 재구성 품질과 매개변수 효율성에서 뛰어난 성능을 나타내며, 유사한 계산 비용 절감 효과를 보여줍니다. 연구 결과, PP는 낮은 오버헤드로 성능을 획기적으로 향상시키면서, 다수의 한계를 해결하는 데 기여하고 있습니다.



### 4DGS-Craft: Consistent and Interactive 4D Gaussian Splatting Editing (https://arxiv.org/abs/2510.01991)
- **What's New**: 최근 발표된 4D Gaussian Splatting (4DGS) 편집 기술들은 다양한 문제, 예를 들어 뷰(view), 시간적(temporal) 일관성, 편집되지 않은 영역의 일관성 확보 등에서 어려움을 겪고 있습니다. 이를 해결하기 위하여 새로운 4DGS-Craft라는 편집 프레임워크를 제안합니다. 이 프레임워크는 4D-aware InstructPix2Pix 모델을 도입하여 기존의 문제를 해결하고, 사용자 인터랙션을 통한 편집 성능 향상에 중점을 두고 있습니다.

- **Technical Details**: 4DGS-Craft는 4D VGGT 기하학(feature)에서 추출한 정보를 통합하여 기하학적 일관성을 확보하는 4D-aware IP2P 네트워크를 포함하고 있습니다. 또한, multi-view grid 모듈을 사용하여 편집 과정에서 다수의 뷰 이미지를 동시에 최적화함으로써 일관성을 강화합니다. 새로운 Gaussian selection 메커니즘을 통해 편집된 영역과 편집되지 않은 영역간의 일관성을 유지할 수 있게 됩니다.

- **Performance Highlights**: 본 프레임워크는 복잡하고 추상적인 사용자 명령을 해석하고 이를 간단한 원자적(atomic) 편집 작업으로 분해할 수 있어, 다양한 사용자의 요구사항을 효과적으로 처리할 수 있습니다. 이를 통해 4DGS-Craft는 직관적이고 일관된 4D 장면 편집을 가능하게 하며, 기존 방법들과 비교할 때 다양한 성능 지표에서 우수한 결과를 보여줍니다.



### TriAlignXA: An Explainable Trilemma Alignment Framework for Trustworthy Agri-product Grading (https://arxiv.org/abs/2510.01990)
- **What's New**: 본 논문은 온라인 과일 및 채소 전자상거래에서 신뢰 부족(trust deficit)을 해결하기 위해 'Trust Pyramid' 모델을 구축하고, 'Triangular Trust Index' (TTI)를 제안합니다. 소비자 신뢰의 이중 출처 검증(dual-source verification)을 통해 품질이 신뢰의 핵심이라는 것을 입증하며, 농산물 등급화 과정에서의 '불가능한 삼각형(impossible triangle)'을 분석합니다. 이러한 분석을 통해 알고리즘의 역할을 '결정권자(decision-makers)'에서 '투명한 의사결정 기반 제공자(providers of transparent decision-making bases)'로 재정의하고, 설명 가능한 AI 프레임워크인 TriAlignXA를 설계합니다.

- **Technical Details**: TriAlignXA는 다목적 최적화(multi-objective optimization)를 통해 농업 제약 속에서 신뢰성 있는 온라인 거래를 지원합니다. 이 프레임워크의 핵심 엔진은 세 가지로 구성됩니다: 생물 적응 엔진(Bio-Adaptive Engine)은 품질의 세분화된 설명을 위한 기능을 제공하며, 시의성 최적화 엔진(Timeliness Optimization Engine)은 처리 효율성을 보장하고, 경제 최적화 엔진(Economic Optimization Engine)은 비용 통제를 위한 모델을 갖추고 있습니다. 또한, QR 코드로 품질 정보를 투명하게 전달하는 '사전 매핑 메커니즘(Pre-Mapping Mechanism)'도 도입하였습니다.

- **Performance Highlights**: TriAlignXA는 Fruit3 데이터세트에서 실험을 통해 85.87%의 분류 정확도를 달성하며, 전통적인 모델보다 우월한 성능을 보입니다. 본 연구는 '불가능한 삼각형'이라는 문제에서 신뢰를 구축하기 위한 균형 잡힌 접근 방식을 제시하여, 신뢰할 수 있는 온라인 농산물 생태계를 구축하는데 이론적으로도 실천적으로도 광범위한 지원을 제공합니다. 실험을 통해 도출된 증거와 이론적 분석은 이 프레임워크의 효과를 확인시켜 줍니다.



### Patch-as-Decodable-Token: Towards Unified Multi-Modal Vision Tasks in MLLMs (https://arxiv.org/abs/2510.01954)
Comments:
          24 pages, 12 figures and 9 tables

- **What's New**: 이번 연구에서는 Patch-as-Decodable Token (PaDT)라는 새로운 패러다임을 소개합니다. 이 접근 방식은 멀티모달 대형 언어 모델(MLLMs)이 직접적으로 텍스트와 다양한 비주얼 출력을 생성할 수 있도록 합니다. PaDT는 시각적 패치 임베딩으로부터 파생된 Visual Reference Tokens (VRTs)를 사용하여, LLM의 텍스트 출력과 원활하게 결합합니다.

- **Technical Details**: PaDT는 VRTs를 매 프레드 패스에서 독립적으로 처리하여 더 효율적인 예측을 가능하게 합니다. 핵심 기술로는 Dynamic Embedding Module이 있으며, 이를 통해 각 이미지 패치에 고유하게 대응되는 VRTs가 생성됩니다. 또한, 경량 디코더를 활용하여 최종적으로 탐지, 세분화 및 기준 예측으로 변환합니다.

- **Performance Highlights**: Empirical 연구 결과, PaDT는 3B 모델이 COCO 탐지에서 이전 상태의 최고 성능인 19.0 mAP 이상을 달성하고, 참조 표현 이해(REC) 작업에서 78B InternVL3 모델보다 높은 93.6의 평균 정확도를 기록했습니다. 이러한 성능은 다양한 시각적 인식 및 이해 작업에서도 입증되었습니다.



### ClustViT: Clustering-based Token Merging for Semantic Segmentation (https://arxiv.org/abs/2510.01948)
Comments:
          Submitted to IEEE

- **What's New**: 비전 트랜스포머(Vision Transformers, ViTs)는 다양한 상황에서 높은 정확도와 강한 일반화 능력을 발휘하지만, 실제 로봇 시스템에서는 기하급수적인 주의(attention) 복잡성 때문에 적용이 제한적입니다. 본 논문에서는 클러스터 모듈(Cluster module)과 재생성 모듈(Regenerator module)을 포함하는 ClustViT라는 새로운 아키텍처를 제안하여 의미론적 분할(semantic segmentation) 문제를 해결합니다. 우리의 접근 방식은 유사한 토큰을 병합하고, 세밀한 정보를 복원하여, 기존의 최첨단 방법들에 비해 계산 비용을 최대 2.18배 줄이고, 1.64배 더 빠른 추론 속도를 달성했습니다.

- **Technical Details**: ClustViT는 트랜스포머 구조의 특성을 활용하여 의미론적 군집화를 수행합니다. 우리의 방법은 RGB 이미지를 분할하여 비트그램으로 변환하고, 클러스터링 모듈을 통해 의미가 유사한 토큰들을 동적으로 식별하고 병합합니다. 또한, 재생성 모듈은 클러스터링된 토큰의 개별 표현을 복원하여 밀집 예측(heads)과의 호환성을 보장합니다. 이 과정에서 세그멘테이션 마스크에서 파생된 유사 군집을 활용하여 이미지의 내용에 맞춘 압축을 진행합니다.

- **Performance Highlights**: ClustViT는 세 개의 데이터셋에서 기존의 최첨단 방법들과 비교하여 비슷한 의미론적 분할 정확도를 유지하면서도 계산 비용을 줄였습니다. 특히, 로봇 시스템에서 자주 발생하는 배경-dominated 상황에서도 뛰어난 성능을 보여줍니다. 코드와 모델은 공개될 예정이며, 이는 연구자들이 쉽게 접근하여 활용할 수 있도록 합니다.



### Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors (https://arxiv.org/abs/2510.01934)
Comments:
          23 pages, 13 figures. Code is available at \url{this https URL}

- **What's New**: 이 논문은 산업 안전 검사를 간소화하고 효율적으로 수행할 수 있는 Few-shot anomaly detection 기술을 제안합니다. 기존의 방법들보다 적은 수의 샘플로 정상과 비정상적 특징을 구분할 수 있는 새로운 접근을 소개하고 있습니다. 특히, FoundAD라는 새로운 Few-shot anomaly detector를 설계하여 비정상 이미지를 효과적으로 감지할 수 있도록 하였습니다.

- **Technical Details**: FoundAD는 자연 이미지 매니폴드(natural image manifold) 위에 비선형 프로젝션 연산자(nonlinear projection operator)를 학습함으로써 구현됩니다. 이 간단한 연산자는 이미지 내의 분포에서 벗어난 영역(out-of-distribution regions)을 특성화하고 식별하는 데 효과적으로 사용됩니다. 또한, 여러 기초 시각 인코더(foundation visual encoders)와의 평가를 통해 우리의 접근 방식을 검증하였습니다.

- **Performance Highlights**: 다양한 실험을 통해 다중 클래스 탐지가 가능하고 이전 방법들보다 적은 매개변수를 사용하면서도 경쟁력 있는 성능을 달성함을 보여주었습니다. 특히, 새로운 DINOv3 모델에 대한 평가 결과는 이 기술의 효용성 및 상승된 성능을 입증합니다. 이 연구는 기초 특징(foundation features)에 대한 새로운 관점을 제시하며 Few-shot anomaly detection 분야의 발전을 이끌 것입니다.



### Automated Defect Detection for Mass-Produced Electronic Components Based on YOLO Object Detection Models (https://arxiv.org/abs/2510.01914)
Comments:
          12 pages, 16 figures, 7 tables, and published in IEEE Sensors Journal

- **What's New**: 본 논문에서는 산업에서 널리 사용되는 이중 인라인 패키지(DIP)의 자동 결함 탐지 시스템을 제안합니다. 이 시스템은 디지털 카메라 광학과 딥 러닝(Deep Learning) 기반 모델을 사용하여 작동하며, 전통적인 수작업 검수에 드는 시간과 노력을 줄이는 것을 목표로 합니다. 또한, 결함 데이터 부족 문제를 해결하기 위해 ConSinGAN을 이용해 학습과 테스트에 적합한 데이터셋을 생성합니다.

- **Technical Details**: 이 시스템은 제어 시스템, 이미징 장비 및 기계 장비의 세 부분으로 구성되어 있습니다. 제어 시스템은 개인 컴퓨터(PC)와 프로그래머블 로직 컨트롤러(PLC)를 포함하여 이미징 장비와 상호 작용하고 SCADA를 통해 딥 러닝 모델과 데이터 분석을 통합합니다. 이미징 장비는 산업용 카메라와 다양한 광원 장비로 구성되어 있으며, 기계 장비는 검사의 자동화를 위해 다양한 기능을 수행합니다.

- **Performance Highlights**: 제안된 YOLOv7 모델은 ConSinGAN과 함께 사용할 때 정확도 95.50%, 탐지 시간 285ms를 기록하였습니다. 이 결과는 기존의 임계값 기반 방법보다 경쟁력 있는 성과를 보여주며, 자동화된 결함 검출 시스템의 실효성을 높입니다. 또한, 관련 생산 라인과 SCADA 인터페이스를 개발하여 실제 응용 프로그램에서의 성능을 검증하였습니다.



### Flow-Matching Guided Deep Unfolding for Hyperspectral Image Reconstruction (https://arxiv.org/abs/2510.01912)
- **What's New**: 이 논문에서는 Flow-Matching-guided Unfolding Network (FMU)를 제안하여 하이퍼스펙트럼 이미지(Hyperspectral Imaging, HSI)의 복원을 향상시킵니다. FMU는 심층 언폴딩(framework) 안에 flow matching을 통합하여, 압축된 측정값을 기반으로 한 생성적(prior) 정보를 활용합니다. 또한, 전반적인 흐름의 일관성을 강화하기 위해 평균 속도 손실(mean velocity loss)을 도입하여 복원 품질을 개선합니다.

- **Technical Details**: FMU는 하이퍼스펙트럼 이미지 복원을 위한 신경망 구조로, flow matching이 압축된 측정값에 조건화된 생성적 정보를 추출하도록 학습됩니다. 이는 언폴딩 네트워크의 디노이징 모듈에 통합되어 복원 과정에 중요한 역할을 합니다. 더불어, FMU는 전통적인 모델 기반 방법과 현대의 학습 기반 방법의 이점을 결합하여 탁월한 결과를 도출합니다.

- **Performance Highlights**: FMU는 하이퍼스펙트럼 이미지 복원에서 기존 기법들보다 월등한 품질을 보여줍니다. 시뮬레이션 데이터셋에서 42.13 dB PSNR을 달성하며, 세부적인 고주파 정보와 섬세한 텍스처를 효과적으로 복원하여 보다 신뢰할 수 있는 결과를 제공합니다. 기존의 방법들과 비교할 때, FMU는 손상된 조건에서도 보다 강력하고 효율적인 복원 성능을 보여줍니다.



### Leveraging Prior Knowledge of Diffusion Model for Person Search (https://arxiv.org/abs/2510.01841)
- **What's New**: 본 논문에서는 DiffPS (Diffusion Prior Knowledge for Person Search)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 기존 방법들이 직면했던 최적화 충돌을 제거하고, 미리 훈련된 diffusion 모델을 활용하여 사람 검색을 더 효율적으로 수행합니다. 이는 복잡한 공간적 맥락과 세밀한 정체성 단서를 캡처하는 데 최적화된 접근을 제공합니다.

- **Technical Details**: DiffPS는 세 가지 전문 모듈로 구성되어 있습니다: (i) Diffusion-Guided Region Proposal Network (DGRPN), 사람 위치 추적 성능을 강화합니다. (ii) Multi-Scale Frequency Refinement Network (MSFRN)는 형태 편향을 줄이는 데 중점을 둡니다. (iii) Semantic-Adaptive Feature Aggregation Network (SFAN)은 텍스트 정렬된 diffusion 특성을 활용합니다.

- **Performance Highlights**: DiffPS는 CUHK-SYSU와 PRW 데이터셋에서 새로운 최첨단 성능을 기록했습니다. 이는 기존의 방법들과 비교할 때 더 우수한 사람 탐지 및 재식별 성능을 보여줍니다. 이러한 결과는 새로운 프레임워크가 효과적으로 사람 검색 문제를 해결함을 입증합니다.



### Calibrating the Full Predictive Class Distribution of 3D Object Detectors for Autonomous Driving (https://arxiv.org/abs/2510.01829)
- **What's New**: 이번 연구는 3D 객체 탐지기의 분류 작업을 위한 신뢰도 보정(confidence calibration)을 다룹니다. 저자들은 모든 클래스에 대한 예측 신뢰도 배포의 보정이 필요하며, 주요 및 보조 클래스 예측 간의 보정을 잡아내는 메트릭(metric)을 제안합니다. 두 가지 보조 정규화 손실 항목을 도입하여 주요 예측 또는 전체 예측 벡터의 보정을 훈련 목표로 삼습니다.

- **Technical Details**: 저자들은 CenterPoint, PillarNet 및 DSVT-Pillar를 포함한 여러 3D 객체 탐지기에 대해 다양한 사후(post-hoc) 및 훈련 시간(train-time) 방법을 평가합니다. 그들은 정규화된 전체 클래스 예측의 보정을 위한 손실 항목과 등온 회귀(isotonic regression)를 결합하면 CenterPoint와 PillarNet에서 최고의 보정을 달성할 수 있음을 발견했습니다. 연구는 또한 DSVT-Pillar가 동일한 방법으로 주요 및 보조 예측을 동시에 보정할 수 없음을 발견했습니다.

- **Performance Highlights**: 이 연구에서는 새로운 메트릭인 Full Detection Expected Calibration Error (Full D-ECE)를 도출하여 전체 예측 클래스 분포의 보정을 포착합니다. 저자들은 정규화가 잘 적용된 비변형(transformer-based)이 아닌 3D 객체 탐지기의 경우, 최상의 보정을 달성하기 위한 방법을 제시합니다. 전반적으로, 이 연구는 자율 주행을 위한 3D 객체 탐지기에서의 신뢰도 보정 방법에 대한 종합적인 비교와 분석을 제공합니다.



### Pack and Force Your Memory: Long-form and Consistent Video Generation (https://arxiv.org/abs/2510.01784)
- **What's New**: 이 논문은 장시간 비디오 생성에서의 두 가지 주요 문제인 장거리 의존성(Long-range dependency) 유지와 오류 누적(Error accumulation) 방지를 해결하기 위해 두 가지 기여를 제안합니다. 그 첫 번째로, MemoryPack이라는 학습 가능한 컨텍스트 검색 메커니즘을 도입하여 텍스트 및 이미지 정보를 글로벌 가이드로 활용하여 단기 및 장기 의존성을 공동으로 모델링합니다. 두 번째로, 오류 누축을 완화하기 위한 Direct Forcing 전략을 제안하며, 이는 학습과 추론 간의 정렬을 개선하여 오류 전파를 줄이는 데 기여합니다.

- **Technical Details**: MemoryPack은 비디오 길이에 비례하여 스케일링이 용이하며, 선형적 복잡도를 유지하면서도 계산 효율성을 보장합니다. 이 메커니즘은 장기 비디오 콘텐츠의 의미적 정합성을 강화해주고, 동시에 인접 프레임을 단기적인 단서를 활용하여 동작과 포즈의 충실도를 향상시킵니다. 또한, Direct Forcing은 판별 흐름(rectified flow)의 원리를 활용하여 단일 단계에서 예측된 벡터 필드를 기반으로 역 ODE(computation)를 수행하여 추론 결과를 근사합니다.

- **Performance Highlights**: 이 방법은 VBench에서 Motion Smoothness, Background Consistency, Subject Consistency와 같은 주요 메트릭에서 현재 최고의 성능을 달성합니다. 실험 결과, MemoryPack과 Direct Forcing은 장기적 컨텍스트 정보를 효과적으로 모델링하고 높은 일관성을 달성하는 데 기여함을 보여줍니다. 이 연구는 자율 회귀 영상 모델의 실용성을 한층 더 향상시키는 데 기여합니다.



### LOBE-GS: Load-Balanced and Efficient 3D Gaussian Splatting for Large-Scale Scene Reconstruction (https://arxiv.org/abs/2510.01767)
- **What's New**: LoBE-GS는 고속의 실시간 3D 장면 복원을 위한 새로운 프레임워크로, 3D Gaussian Splatting(3DGS)의 대규모 파이프라인을 재설계한 것입니다. 이 시스템은 깊이 인지 파티셔닝 기법을 통해 데이터 전처리 시간을 몇 시간에서 몇 분으로 단축시키며, 가시적인 가우시안의 수를 기반으로 로드 밸런싱을 구현합니다. 또한, 시각적 크로핑과 선택적 밀도를 추가하는 두 가지 경량 기술을 통해 교육 비용을 추가로 낮춥니다.

- **Technical Details**: LoBE-GS는 장면 파티셔닝의 비효율성을 해결하고 조각 간의 균형 잡힌 계산을 위해 세 가지 주요 기술을 도입합니다. 전처리 시간은 O(M×N)에서 O(N)으로 단축되고, 가시적인 가우시안 수를 계산 부하의 신뢰할 수 있는 대리인으로 사용하여 최적화를 실현합니다. 또한, 각 블록에서 불필요한 가우시안을 제거하는 시각적 크로핑 및 필요한 경우에만 가우시안을 추가하는 선택적 밀도를 통해 각 블록의 계산 부하를 줄입니다.

- **Performance Highlights**: LoBE-GS는 대규모 도시 및 야외 데이터셋에서 기존의 최첨단 방법보다 최대 2배 더 빠른 훈련 시간을 달성하며 재구성 품질을 유지하거나 개선합니다. 실험 결과, LoBE-GS는 학습과 계산의 균형을 맞추어 높은 효율성과 확장성을 보여주며, 기존의 3DGS에서 불가능했던 장면에서도 안정된 성능을 보입니다. 이러한 성능 향상으로 LoBE-GS는 차세대 3D 콘텐츠 생성의 중요한 기반이 될 것입니다.



### PyramidStyler: Transformer-Based Neural Style Transfer with Pyramidal Positional Encoding and Reinforcement Learning (https://arxiv.org/abs/2510.01715)
- **What's New**: PyramidStyler는 기존 Neural Style Transfer(NST) 방식의 한계를 극복하기 위해 제안된 새로운 Transformer 기반의 프레임워크입니다. Pyramidal Positional Encoding(PPE)을 통해 복잡한 스타일과 고해상도 입력을 효과적으로 처리할 수 있도록 설계되었습니다. 이 구조는 지역 세부정보와 글로벌 맥락을 캡처하면서 계산적 부하를 줄이는 데 초점을 두고 있습니다.

- **Technical Details**: PPE는 여러 스케일에서 겹치는 패치를 구성하고, 각각을 다양한 커널 크기를 가진 CNN을 통해 인코딩한 뒤, 주의(attention) 메커니즘이나 연결(concatenation)을 통해 융합합니다. 이러한 계층적 설계는 세부사항과 넓은 공간적 관계를 보존하면서도 계산을 효율적으로 처리할 수 있게 합니다. 또한 경량화된 강화 학습(RL) 에이전트를 통합하여 스타일화 가중치를 동적으로 조정하여 수렴 속도를 높이고 시각적 품질을 개선합니다.

- **Performance Highlights**: PyramidStyler는 Microsoft COCO와 WikiArt 데이터셋에서 훈련되어 4000 에폭 후 콘텐츠 손실(content loss)을 62.6% 줄이며(2.07로), 스타일 손실(style loss)을 57.4% 감소시킵니다(0.86으로). RL을 사용할 경우 콘텐츠 손실(2.03) 및 스타일 손실(0.75)에서 추가 개선을 보여주고, 속도 손실은 최소(1.40s)로 유지됩니다. 이러한 결과는 실시간, 고품질의 예술적 렌더링을 가능하게 하여 미디어와 디자인 분야에서 폭넓은 응용 가능성을 시사합니다.



### Holistic Order Prediction in Natural Scenes (https://arxiv.org/abs/2510.01704)
Comments:
          25 pages, 11 figures, 6 tables

- **What's New**: 본 논문에서는 InstaFormer라는 새로운 네트워크를 제안합니다. InstaFormer는 RGB 이미지 하나만으로 씬의 모든 인스턴스에 대한 전체 압출(i.e., occlusion) 및 깊이(depth) 순서를 단일 전방 패스를 통해 반환할 수 있습니다. 이는 기존의 시스템들이 비싼 입력 형식과 높은 추론 비용에 의존했던 문제를 해결하고, 인스턴스 간의 상호작용을 통해 이루어집니다.

- **Technical Details**: InstaFormer는 객체 쿼리와 잠재적인 마스크 설명자 간의 상호작용을 통해 인스턴스 간의 기하학적 관계를 예측합니다. 제안된 모델은 점진적 입력 요구를 완화하며, 기존의 방식을 뛰어넘어 단일 전방 패스에서 전체 인스턴스의 기하학적 관계를 예측할 수 있는 기능을 갖추고 있습니다. 이 네트워크는 거리 level의 압출 및 깊이 예측 작업을 인접 행렬 수준의 문제로 재구성합니다.

- **Performance Highlights**: 모델의 성능은 다양한 벤치마킹을 통해 검증되었으며, 기존의 최상위 모델들과 동등하거나 이를 초월하는 결과를 보였습니다. InstaFormer는 RGB 입력 이미지 하나로 압출 및 깊이 순서 예측 작업에서 뛰어난 성능을 나타냅니다. 또한, 논문에서 제시된 코드와 모델은 오픈 소스로 제공되어, 더 많은 연구자들이 활용할 수 있습니다.



### MedQ-Bench: Evaluating and Exploring Medical Image Quality Assessment Abilities in MLLMs (https://arxiv.org/abs/2510.01691)
Comments:
          26 pages, 13 figures

- **What's New**: 본 논문에서는 MedQ-Bench라는 포괄적인 벤치마크를 소개하고 있습니다. 이 벤치마크는 임상 AI에서의 의료 이미지 품질 평가(IQA)를 위한 새로운 패러다임을 수립하며, Multi-modal Large Language Models (MLLMs)를 활용하여 의료 이미지 품질을 평가하는 언어 기반 접근 방식을 제시합니다. MedQ-Bench는 두 가지 상호 보완적인 작업인 MedQ-Perception과 MedQ-Reasoning을 정의하여 시각적 속성의 저수준 인식 능력을 조사하고 임상적 맥락에서의 인간 유사성을 반영합니다.

- **Technical Details**: MedQ-Bench는 5가지 이미징 모달리티와 40개 이상의 품질 속성을 아우르는 총 2600개의 인지 쿼리와 708개의 추론 평가로 구성됩니다. 이는 다양한 이미지 소스로부터 수집된 데이터로 구성되며, 실제 임상 데이터, 물리 기반 재구성을 통한 시뮬레이션된 손상 이미지를 포함합니다. MLLM의 추론 능력을 평가하기 위해, 네 가지 보완적 축을 통해 모델 출력을 평가하는 다차원 판단 프로토콜을 설계하였습니다.

- **Performance Highlights**: 14개의 최첨단 MLLM에 대한 평가 결과, 모델들이 불안정한 인식 및 추론 기술을 보였고, 신뢰할 수 있는 임상적 사용을 위한 정확성이 부족한 것으로 나타났습니다. 이는 의료 IQA에서 MLLM의 타겟 최적화의 필요성을 강조합니다. MedQ-Bench는 또한 MLLM의 의료 이미지 품질 평가의 미개척된 잠재력을 지속적으로 탐구하도록 촉진할 것으로 기대합니다.



### FreeViS: Training-free Video Stylization with Inconsistent References (https://arxiv.org/abs/2510.01686)
Comments:
          Project Page: \url{this https URL}

- **What's New**: 이번 논문에서는 FreeViS라는 훈련이 필요 없는 비디오 스타일화 프레임워크를 제안합니다. 기존의 프레임별 이미지 스타일링 방법에서 발생하는 시간 일관성 문제를 해결하기 위해 여러 스타일화된 참조를 활용합니다. 이를 통해 스타일의 풍부함을 유지하고, 플리커(flicker)나 스터터(stutter)와 같은 전송 오류를 효과적으로 완화합니다.

- **Technical Details**: FreeViS는 저주파(LF)와 고주파(HF) 구성 요소를 활용하여 비디오의 공간 레이아웃과 동작을 제어합니다. 또한, 종합적으로 최적화된 마스킹 전략과 동적 단서를 통해 참조 입력으로부터 스타일 텍스처를 유지합니다. PnP Inversion 기법을 사용하여 디노이징(de-noising) 과정과 스타일화 과정을 조화롭게 연결합니다.

- **Performance Highlights**: 광범위한 평가를 통해 FreeViS는 더 높은 스타일화 충실도(stylization fidelity)와 우수한 시간 일관성(temporal consistency)을 보이며, 최근 기반 모델들보다 뛰어난 성능을 나타냅니다. 추가적으로, FreeViS는 기존의 T2V(text-to-video) 모델과 결합하여 최신 훈련 기반 방법과 비슷한 성능을 소화하는 결과를 보여주었습니다.



### Uncovering Overconfident Failures in CXR Models via Augmentation-Sensitivity Risk Scoring (https://arxiv.org/abs/2510.01683)
Comments:
          5 pages, 1 figures

- **What's New**: 이번 연구는 의료 영상의 공정성과 신뢰성을 높이기 위해 Augmentation-Sensitivity Risk Scoring (ASRS) 프레임워크를 제안합니다. 기존의 오류 탐지 방식이 숨겨진 오류를 감지하는 데 제한적이라는 점을 지적하며, ASRS는 CXR 사례 중 오류가 발생하기 쉬운 경우를 식별하는 데 사용됩니다. ASRS는 작은 회전을 적용하여 임베딩의 변화를 측정하고, 이를 통해 민감도 점수를 계산하여 이미지의 안정성을 평가합니다. 이를 통해 ASRS는 예측의 선택성 및 의사의 검토를 개선하고, 의료 AI의 안전성과 공정성을 향상시키는 데 기여할 수 있습니다.

- **Technical Details**: ASRS 프레임워크는 세 가지 주요 구성 요소로 구성됩니다. 첫째, CXR에 작은 회전을 적용하여 레이블이 없는 augmentation-sensitivity risk score를 계산합니다. 둘째, 검증 세트에서 유도된 사분위수 임계값을 통해 테스트 세트를 민감도 수준에 따라 4개 그룹(G1-G4)으로 나눕니다. 마지막으로 ASRS를 통해 테스트 세트의 안정성을 정량화하고 모델 아키텍처의 일반화 능력을 평가합니다. RAD-DINO과 같은 대조 학습 모델을 사용해 회전 변환에 대한 임베딩의 불안정을 측정하며, 낮은 ASRS 점수는 더 낮은 진단 신뢰성으로 이어집니다.

- **Performance Highlights**: CXR의 네 가지 진단 작업(Cardioomegaly, Edema, Pneumothorax, Pleural Effusion)에서 ASRS 프레임워크의 성능을 평가하였습니다. G1에서 G4로 갈수록 민감도가 증가함에 따라 리콜은 점진적으로 감소하였으며, G4의 민감도 높은 경우는 가장 낮은 리콜을 보였습니다. 오히려 AUROC는 G4에서 증가했는데 이는 AUROC가 순위 정확성을 반영하지만, 절대적인 민감도는 포착하지 못해 불안정한 사례에 대해 지나치게 낙관적인 모습을 보여주었습니다. ASRS는 안정성과 전통적인 신뢰도 측정치 간의 간극을 간파하며, 최전방으로서의 활용 가능성을 제시합니다.



### Look Less, Reason More: Rollout-Guided Adaptive Pixel-Space Reasoning (https://arxiv.org/abs/2510.01681)
Comments:
          Preprint, Under review

- **What's New**: 이번 연구에서는 기존의 Vision-Language Models (VLMs)의 한계를 극복하기 위한 최초의 적응형 픽셀 공간 추론(adaptive pixel-space reasoning) 구조를 제안합니다. 기존 연구들은 이미지 인코딩 과정에서의 정보 손실로 인해 세부적인 시각적 요소를 처리하는 데 어려움을 겪었습니다. 본 논문은 입력 쿼리에 따라 필요한 픽셀 수준의 작업을 동적으로 결정할 수 있는 새로운 프레임워크를 도입합니다.

- **Technical Details**: 제안된 프레임워크는 작업 인식(supervised fine-tuning) 및 롤아웃 기반 강화학습(rollout-guided reinforcement learning) 기법을 사용하여 VLM의 시각적 작업 수행 능력을 향상시킵니다. supervised fine-tuning 단계에서는 모델이 텍스트 관련 질문에 대한 기준 능력을 세우고, 강화학습 프레임워크는 도구 사용 빈도를 최적화하여 픽셀 작업의 필요성을 결정하도록 설계되었습니다. 이 과정에서 모델은 과거 롤아웃의 피드백을 통해 빈번한 픽셀 작업을 줄이는 동시에 정확도를 높일 수 있습니다.

- **Performance Highlights**: 제안된 모델은 HR-Bench 4K에서 73.4%의 정확도를 달성하면서 도구 사용 비율은 20.1%로 함께 개선되었습니다. 이는 이전 방법에 비해 66.5% 도구 사용을 줄이면서도 높은 정확도를 유지하는 성과를 보여줍니다. 추가적인 정성적 분석을 통해 모델이 적절한 상황에서만 픽셀 작업을 수행하는 능력을 갖추었음을 입증했습니다.



### An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution (https://arxiv.org/abs/2510.01678)
Comments:
          Published in Expert Systems with Applications

- **What's New**: 본 연구는 템플릿 매칭을 합동 위치 추정(joint localization) 및 기하학적 회귀(geometric regression)로 재구성하는 경량화된 엔드 투 엔드 프레임워크를 제안합니다. 이는 중심 좌표, 회전 각도, 독립적인 수평 및 수직 스케일을 출력하여 정밀한 산업응용에 적합하도록 설계되었습니다. 또한, 템플릿 인지 동적 컨볼루션 모듈(TDCM)이 추론 과정에서 템플릿 특징을 동적으로 주입하여 일반화 가능한 매칭을 안내합니다. 이 방법은 기하학적 주석이 필요 없는 학습을 위한 회전-전단 기반 증강 전략을 도입하여 최신 기술의 한계를 극복합니다.

- **Technical Details**: 기존의 전통적인 방법들은 각도와 스케일의 포괄적인 나열에 의존하여 비효율적이었으나, 본 연구에서는 직접적으로 대상의 중심 위치와 기하학적 변형 매개변수를 회귀하는 경량 템플릿 매칭 방법을 제안합니다. TDCM은 템플릿 특징을 학습 가능한 컨볼루션 커널로 인코딩하고, 깊이별 분리 가능한 컨볼루션(depthwise separable convolutions)을 사용하여 탐색 이미지에 통합합니다. 또한, 메모리 최적화와 임의적인 복잡한 변환으로부터 빠른 매칭을 가능하게 합니다.

- **Performance Highlights**: 실험 결과에 따르면, 본 연구에서 제안한 3.07M 모델은 복합 변환(compound transformations)에서 높은 정밀도를 달성하며, 단일 이미지 쌍당 14ms의 빠른 추론 속도를 자랑합니다. 이는 Halcon의 형태 기반 매칭 모듈보다 우수한 성능을 보여주며, 작은 템플릿과 다중 객체 시나리오에서 강력한 견고성을 유지합니다. 이러한 특성 덕분에 본 연구는 실시간 산업 응용 프로그램에 적합한 솔루션을 제공합니다.



### UniVerse: Unleashing the Scene Prior of Video Diffusion Models for Robust Radiance Field Reconstruction (https://arxiv.org/abs/2510.01669)
Comments:
          page: this https URL code: this https URL

- **What's New**: 이 논문에서는 불안정한 다중 시점 이미지를 사용하여 3D 장면을 견고하게 복원하는 새로운 접근 방식을 제안합니다. 특히, 'UniVerse'라는 비디오 분산 모델에 기반한 통합 프레임워크를 소개하여 이미지의 비일관성을 제거하고 3D 장면을 재구성하는 과정에서 두 가지 하위 작업(복원 및 재구성)으로 문제를 분리합니다. 이를 통해 전체 최적화를 간소화하고 다양한 이미지 집합에 적용 가능함을 보여줍니다.

- **Technical Details**: 논문에서 제안하는 UniVerse는 먼저 불안정한 이미지를 초기 비디오로 변환한 후, 비디오 분산 모델을 이용해 일관된 이미지로 복원합니다. 다음으로, 이 복원된 이미지를 기반으로 3D 장면을 재구성합니다. UniVerse는 대규모 데이터로 일반적인 장면 선행 지식을 학습하여 다양한 비일관성을 처리하는 측면에서 향상된 성능을 보여줍니다.

- **Performance Highlights**: 합성 데이터 및 실제 데이터셋에 대한 실험 결과, UniVerse는 PSNR, SSIM, LPIPS 값 측면에서 이전의 최첨단 방법들을 초월하는 탁월한 성능을 발휘했습니다. 심지어 이미지 묶음이 적고 비일관적인 경우에도 (예: 2개의 이미지만 있는 경우) 신뢰할 수 있는 일관된 이미지를 복원할 수 있으며, 이는 새로운 시점 생성 및 추가 재구성 등의 하위 작업에 적용 가능하다는 것을 입증합니다.



### Non-Rigid Structure-from-Motion via Differential Geometry with Recoverable Conformal Sca (https://arxiv.org/abs/2510.01665)
- **What's New**: 이번 논문에서는 변형된 객체의 3D 구조를 복원하는 Non-rigid structure-from-motion (NRSfM) 분야에서 새로운 접근 방식인 Con-NRSfM을 제안합니다. 이 방법은 특히 conformal deformation에 적합하며, 기존의 가정들 없이 정확한 지역 conformal scale을 계산할 수 있습니다. 또한, 깊이(depth)와 conformal scale에 대한 제약을 분리하여 더 정밀한 깊이 추정을 가능하게 합니다.

- **Technical Details**: Con-NRSfM은 그래프 기반 프레임워크를 통해 최적화된 2D 이미지 왜곡을 사용하여 점 기반 복원을 수행합니다. 기존 방법들과 달리, 이 방법은 직접적인 깊이 계산을 요구하지 않으며, 깊이와 표면 법선(normals)의 두 번째 미분까지 처리 가능한 물리적 제약을 정의합니다. 이 알고리즘은 병렬 분리형 반복 최적화(parallel separable iterative optimization) 전략을 사용하며, 자체 지도 학습(self-supervised learning) 네트워크를 통해 질감이 있는 3D 점 구름을 생성합니다.

- **Performance Highlights**: 모의 실험과 실제 데이터셋을 통해 Con-NRSfM이 기존의 최신 기술(state-of-the-art) 방법들을 초월하여 복원 정확도와 내구성 측면에서 뛰어난 성능을 보여주었음을 입증하였습니다. 이 방법은 비등거리 변형이나 강한 굴곡이 있는 경우에도 높은 성능을 유지하며, 향후 변형 비주얼 SLAM 분야에서 중요한 기여를 할 것으로 기대됩니다.



### Discrete Facial Encoding: : A Framework for Data-driven Facial Display Discovery (https://arxiv.org/abs/2510.01662)
- **What's New**: 이 논문은 Facial Action Coding System (FACS)의 한계를 극복하는 새로운 데이터 기반의 Facial Expression Coding 방법인 Discrete Facial Encoding (DFE)을 소개합니다. DFE는 Residual Vector Quantized Variational Autoencoder (RVQ-VAE)를 이용하여 3D 메쉬 시퀀스에서 발달한 얼굴 표정의 Compact하고 해석 가능한 사전을 생성합니다. 이 방법은 수동 주석 작업 없이 자동으로 얼굴 표정을 인코딩하며, 세 가지 심리적 과제에 대한 유용성을 평가합니다.

- **Technical Details**: DFE는 3D Morphable Model (3DMM)을 이용하여 이미지에서 정체성 불변의 표정 특징을 추출합니다. 이후, 이러한 특성을 RVQ-VAE를 통해 인코딩하여 각 토큰이 특정 얼굴 변형 패턴을 포착하는 이산 토큰의 시퀀스를 생성합니다. 이 연구는 FACS와 기존의 다른 표정 인코딩 방법에 비해 더 정밀한 얼굴 행동을 포착한다는 점에서 중요한 기여를 합니다.

- **Performance Highlights**: 실험 결과, DFE 모델은 stress detection, personality prediction, depression detection과 같은 심리적 과제에서 전통적인 FACS 기반 특징 및 강력한 기존의 이미지 및 비디오 표현 학습 모델에 비해 일관되게 우수한 성능을 보여줍니다. 이를 통해 발전된 데이터 기반의 얼굴 표현 방법이 심리학과 감정 컴퓨팅 분야에서 보다 효과적이고 확장 가능한 대안이 될 수 있음을 밝혔다.



### VirDA: Reusing Backbone for Unsupervised Domain Adaptation with Visual Reprogramming (https://arxiv.org/abs/2510.01660)
- **What's New**: 본 논문은 기존 UDA(비지도 도메인 적응) 파이프라인의 한계를 극복하기 위해 VirDA(시각적 재프로그래밍 도메인 적응)를 제안합니다. 기존 방법들이 새로운 소스-타겟 쌍을 위해 잘 훈련된 백본(backbone) 파라미터를 매번 미세 조정하는 것과 달리, VirDA는 도메인 특정 비주얼 재프로그래밍 레이어를 추가하여 백본 수정 없이도 도메인 적응을 가능하게 합니다. 이를 통해 모델 재사용성을 높이고 필요한 저장 용량을 크게 줄였습니다.

- **Technical Details**: VirDA는 도메인 별 텍스처 패턴을 활용하여 입력 이미지의 스타일을 조정하는 역할을 하는 시각적 재프로그래밍 레이어로 구성됩니다. 이 레이어는 백본 앞에 위치하여, 백본 파라미터를 수정하지 않고도 도메인 적응을 수행합니다. 또한, 두 가지 주요 목적 함수를 설계하여, 도메인 간 및 도메인 내 정렬 목표를 설정하고, 각각의 도메인에서의 다이나믹한 특징 학습을 하는 방식을 사용합니다.

- **Performance Highlights**: VirDA는 1.5M의 훈련 가능한 파라미터만으로 Office-31 벤치마크에서 92.8%의 평균 정확도를 달성했습니다. 이는 최신의 UDA 기법인 PDA보다 1.6% 더 높은 정확도를 기록하며, 필요한 파라미터는 46%에 불과합니다. VirDA는 전체 백본 미세 조정보다 더 나은 성능을 보여주며, 저장 용량도 6MB로 매우 효율적입니다.



### LadderMoE: Ladder-Side Mixture of Experts Adapters for Bronze Inscription Recognition (https://arxiv.org/abs/2510.01651)
Comments:
          18 pages, 7 figures, 2 Tables

- **What's New**: 이번 연구는 청동 문자(영어: Bronze Inscriptions, BI) 인식을 위한 대규모 데이터셋을 구축하고, 두 단계의 탐지-인식 파이프라인을 개발하였습니다. 이 데이터셋은 22,454개의 전체 페이지 이미지와 198,598개의 주석된 문자가 포함되어 있으며, 6,658개의 고유 카테고리를 포괄합니다. 연구팀은 LadderMoE 기법을 도입하여 사전 학습된 CLIP 인코더와 계단형 MoE 어댑터를 결합하여 전문가의 동적 특화 및 강력한 강인성을 제공합니다.

- **Technical Details**: 연구에서는 YOLO-v12를 기반으로 한 객체 탐지기를 사용하여 BI를 full-page 인식하는 두 단계의 파이프라인을 채택하고 있습니다. 첫 번째 단계에서 청동 문자의 위치를 확인한 후, 인식기는 LadderMoE를 통해 단일 문자 이미지를 인식합니다. LadderMoE 모델은 다양한 특성을 가진 카테고리 간의 특징을 적응적으로 라우팅할 수 있도록 설계되었습니다.

- **Performance Highlights**: 포괄적인 실험 결과, 제안된 방법이 기존의 최첨단 장면 텍스트 인식 기준을 초과하여 우수한 성능을 발휘함을 보여주었습니다. 특히, 흔한 문자, 중간 문자 및 드문 문자까지 모든 범주에서 탁월한 정확도를 달성하며, 다양한 취득 모드에서 BI 인식의 견고성을 입증하였습니다.



### FideDiff: Efficient Diffusion Model for High-Fidelity Image Motion Deblurring (https://arxiv.org/abs/2510.01641)
- **What's New**: 본 연구에서는 고해상도 이미지를 복원하기 위한 새로운 단일 단계 확산 모델인 FideDiff를 제안합니다. 기존의 CNN 및 transformer 기반 방법보다 더 우수한 생성 능력을 보여주는 대규모 사전 훈련된 확산 모델의 잠재력을 최대한 활용하고자 합니다. FideDiff는 각 시간 단계가 점진적으로 흐려진 이미지를 나타내는 확산 프로세스으로 모션 블러 제거를 재구성하여 고충실도 복원을 제공합니다.

- **Technical Details**: FideDiff는 블러 궤적을 맞춘 훈련 데이터를 재구성하여 시간적 일관성을 학습하여 정확한 한 단계 모션 블러 제거를 지원합니다. 모델은 Kernel ControlNet을 통합하여 블러 커널을 추정하고 적응형 시간 단계 예측을 도입함으로써 성능을 더욱 향상시킵니다. 또한, FideDiff는 블러 정도에 따른 적절한 시간 단계를 선택할 수 있도록 회귀 모듈을 설계하였습니다.

- **Performance Highlights**: FideDiff는 전체 참조 메트릭에서 이전의 모든 사전 훈련된 확산 기반 모델을 초월하는 성능을 기록하며, 감성 유사성에서도 최신 transformer 기반 모델과 맞먹거나 이를 초월하는 결과를 보여줍니다. 본 연구는 사전 훈련된 확산 모델을 이미지 복원 작업에 적용하기 위한 새로운 관점을 제공하며, 저수준 비전 분야에서의 확산 모델의 발전을 위한 강력한 기준을 마련합니다.



### Joint Deblurring and 3D Reconstruction for Macrophotography (https://arxiv.org/abs/2510.01640)
Comments:
          Accepted to Pacific Graphics 2025. To be published in Computer Graphics Forum

- **What's New**: 이 연구는 마크로 촬영(macro photography)에서의 초점 흐림(defocus blur) 문제를 해결하기 위해, 선명한 이미지와 3D 장면을 동시에 최적화하는 새로운 방법을 제안합니다. 기존의 이미지 디블러링(deblurring) 작업은 대량의 이미지와 주석(annotation)을 필요로 했지만, 본 연구에서는 소수의 다중 뷰(multi-view) 이미지를 활용하여 고품질 이미지를 복원할 수 있습니다. 특히, 깊이 변화(depth variation)로 인한 흐림을 정확히 시뮬레이션하는 것을 핵심 아이디어로 삼고 있습니다.

- **Technical Details**: 본 연구는 다중 뷰 이미지를 사용하여 각 픽셀의 초점 흐림 커널(defocus blur kernel)과 물체의 선명한 3D 모델을 공동 최적화(joint optimization)하는 방법론을 채택했습니다. 최신의 차별화된 렌더링(differentiable rendering) 기법을 이용하여 3D 모델 및 흐림 커널 최적화를 자기 감독(self-supervised) 방식으로 수행합니다. 흐림 맵을 생성하고, 각 픽셀에 흐림 합성곱 커널(blurred convolution kernel)을 할당하여 선명한 2D 이미지를 복원합니다.

- **Performance Highlights**: 실험은 합성 데이터와 실제 데이터를 포함하여 광범위하게 수행되었습니다. 결과적으로 제안된 방법은 기존의 단일 이미지 디블러링 및 3D 장면 디블러링 방법들보다 우수한 성능을 보였습니다. 이 연구는 3D 장면 복원과 흐림 모델링을 동시에 수행할 수 있는 첫 번째 자기 감독 최적화 방법을 제시하며, 작은 물체의 선명한 3D 외관을 회복하는 데 성공했습니다.



### VLA-R1: Enhancing Reasoning in Vision-Language-Action Models (https://arxiv.org/abs/2510.01623)
- **What's New**: Vision-Language-Action (VLA) 모델들은 인식(perception), 언어 이해(language understanding), 행동 생성(action generation)을 통합하여 강력한 교차 작업(cross-task) 및 교차 장면(cross-scene) 일반화를 제공합니다. 하지만 기존 모델들은 명확한 단계별 추론(step-by-step reasoning)을 결여하고 있어 최종 행동만을 예측하고, 교육 후 효과적인 추론 품질 강화를 위한 체계적인 접근이 부족합니다. 이러한 문제를 해결하기 위해, VLA-R1은 검증 가능한 보상(Reinforcement Learning from Verifiable Rewards, RLVR)과 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)를 통합하여 추론과 실행을 모두 최적화할 수 있도록 설계되었습니다.

- **Technical Details**: VLA-R1은 RLVR 기반의 후속 학습(post-training) 전략을 도입하여 지역 정렬(region alignment), 경로 일관성(trajectory consistency), 출력 형식(output formatting) 등의 세 가지 검증 가능한 보상을 통해 추론의 견고성과 실행 정확성을 강화합니다. 또한 VLA-CoT-13K 고품질 데이터셋을 개발하여 affordance와 경로 주석에 명시적으로 맞춰서 체인 오브 씽크(Chain-of-Thought, CoT) 감독을 제공합니다. 이러한 최적화 접근 방식은 VLA-R1이 정확한 affordance 지역과 행동 경로를 생성하도록 하여 의사 결정을 향상시킵니다.

- **Performance Highlights**: 종합적인 평가를 통해 VLA-R1은 인도메인(in-domain)과 아웃 오브 도메인(out-of-domain) 데이터셋, 시뮬레이션, 실제 로봇 환경에서 검증되었습니다. 실험 결과, VLA-R1은 인도메인 affordance 벤치마크에서 36.51의 IoU라는 성과를 달성했으며, 이는 기존 방법보다 17.78% 개선된 수치입니다. 또한 실제 하드웨어에서 VLA-R1은 affordance 인식에서 62.5%, 경로 실행에서 75%의 성공률을 기록하며, 이 모델의 효과성과 다양한 환경에서의 강건성을 입증했습니다.



### Automated Genomic Interpretation via Concept Bottleneck Models for Medical Robotics (https://arxiv.org/abs/2510.01618)
- **What's New**:  본 연구에서는 원시 DNA 서열을 의료 자동화 및 로봇 시스템에 적합한 해석 가능한 결정으로 변환하는 자동화된 유전자 해석 모듈을 제안합니다. 우리가 개발한 프레임워크에서는 Chaos Game Representation (CGR)과 Concept Bottleneck Model (CBM)을 결합하여, 생물학적으로 의미 있는 개념으로 예측 흐름을 강제합니다. 이를 통해 HIV 아형의 정확한 분류뿐만 아니라 생물학적 근거에 대해 직접 검증할 수 있는 해석 가능한 증거를 제공합니다.

- **Technical Details**:  제안된 시스템은 원시 DNA 서열을 해석 가능한 개념, 분류 및 비용 인식 권장 사항으로 자동화하는 전체 파이프라인을 구성합니다. CGR을 사용하여 서열을 이미지로 매핑하고, CNN(Convolutional Neural Network)을 사용하여 인코딩하며, 엄격한 CBM을 통해 제약을 둡니다. 추가 규제 요소로는 개념 신뢰도, 사전 정렬, KL(분포 간의 Kullback-Leibler) 매칭, 불확실성 조정이 포함되어 정확성과 해석성을 보장합니다.

- **Performance Highlights**:  제안된 시스템은 최신 분류 성능을 달성하고 기존의 기초선 모델과 비교해 더 높은 개념 예측 신뢰성과 유리한 비용-편익 무역 결과를 보여줍니다. 우리는 시스템 레벨 통합 사례를 제공하여 유전자 분석을 위한 의료 자동화 및 로봇 통합의 신뢰할 수 있는 기초를 마련했습니다. 또한, 이 시스템은 설명 가능한 인식으로부터 실행 가능한 추천으로 이어지는 폐쇄 루프를 구축하여 사용자가 신뢰할 수 있는 임상 결정을 가능하게 합니다.



### NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems (https://arxiv.org/abs/2510.01608)
Comments:
          25 pages, 12 tables, 10 figures. Accepted to NeurIPS 2025

- **What's New**: 이번 연구에서는 기존의 규제 방식에서 벗어나 비선형 프로젝션 방식인 Non-Linear Projections of the Null-Space (NPN)을 제안합니다. 이 방식은 신호 도메인에서의 구조적 제약이 아닌, 센싱 행렬(null-space)의 저차원 프로젝션에서 해결책을 찾아내도록 합니다. 이를 통해 이미지 복원 문제에서의 높아진 해석 가능성과 유연성을 강조하고 있습니다.

- **Technical Details**: NPN은 센싱 행렬의 null-space 구조를 기반으로 하여, 노이즈가 존재하는 데이터의 측정값으로부터 직관적으로 예측 가능한 계수를 학습하도록 신경망을 훈련합니다. 이 과정에서, 그 결과로 얻어진 프로젝션 매트릭스(𝐒)는 본 연구의 핵심으로, 연구 중인 작업에 따라 유연하게 변화할 수 있게 설계되었습니다. 연구에서는 이 신경망이 훈련되는 동안 해당 프로젝션 매트릭스의 적응 가능성을 강조합니다.

- **Performance Highlights**: 다양한 실험을 통해 NPN이 여러 이미지 복원 문제에서 일관되게 높은 재구성 정확도를 보여주는 것을 확인했습니다. NPN은 이미지 복원, 압축 센싱, 의료 이미징 등 여러 분야에서 효과적으로 활용될 수 있으며, 기존의 방법들보다 향상된 재구성 충실도를 보이는 것으로 평가됩니다. 특히, 실험 결과는 불완전한 데이터에서도 다소 낮은 오류율을 보이며, 더 나은 신호 복원 성능을 나타내었습니다.



### ImageNet-Think-250K: A Large-Scale Synthetic Dataset for Multimodal Reasoning for Vision Language Models (https://arxiv.org/abs/2510.01582)
Comments:
          Preprint

- **What's New**: 본 논문에서는 Vision Language Models (VLMs)의 명시적 추론 능력을 개발하기 위한 멀티모달 추론 데이터셋인 ImageNet-Think를 제안합니다. 이 데이터셋은 ImageNet21k에서 250,000개의 이미지를 기반으로 하여 구조화된 사고 토큰과 관련된 답변을 제공합니다. GLM-4.1V-9B-Thinking과 Kimi-VL-A3B-Thinking-2506 두 가지 최신 VLM에서 생성된 이 합성 데이터셋은 모델의 훈련 및 평가에 유용한 자원으로 활용될 것입니다.

- **Technical Details**: ImageNet-Think 데이터셋은 250,000개의 이미지에 대해 각 이미지에 대해 2쌍의 사고-답변 시퀀스를 포함하고 있습니다. 이러한 구조화된 사고 토큰은 GLM-4.1V-Thinking과 Kimi-VL-Thinking 모델에 의해 생성되며, 다양한 추론 패턴을 캡처하여 VLM 훈련에 기여하는 다양성을 제공합니다. 또한, 이 데이터셋은 기존의 VLM 훈련 데이터셋들이 가지고 있는 제한적인 스코프 문제를 해결합니다.

- **Performance Highlights**: ImageNet-Think는 500,000개의 사고-답변 쌍을 제공하여 대규모 공개 데이터셋으로서 중요한 역할을 할 것입니다. 또한, 다양한 평가 기준을 통해 여러 VLM의 성능을 벤치마킹하며, 연구자들이 해석 가능한 신뢰할 수 있는 VLMs를 훈련할 수 있도록 지원합니다. 이 데이터셋과 평가 기준은 멀티모달 추론 연구의 발전에 기여할 것입니다.



### Guiding Multimodal Large Language Models with Blind and Low Vision People Visual Questions for Proactive Visual Interpretations (https://arxiv.org/abs/2510.01576)
Comments:
          7 pages, 2 figure, 2 tables, CV4A11y Workshop at ICCV 2025

- **What's New**: 이번 연구에서는 시각 장애인 및 저시력자(Blind and Low Vision, BLV) 사용자들이 보다 구체적이고 맥락에 맞춘 시각 정보를 얻을 수 있도록, 이전 BLV 사용자들의 질문을 활용한 시스템을 개발했습니다. 기존의 MLLM 기반 애플리케이션들이 종합적이고 긴 설명을 제공하는 것과 달리, 이 시스템은 VizWiz-LF 데이터셋에서 유사한 과거 시각적 맥락을 식별하여 사용자들이 궁금해 할 수 있는 내용을 예측합니다.

- **Technical Details**: 우리의 시스템은 600개의 질문-이미지 쌍으로 구성된 VizWiz-LF 데이터셋을 사용하여 맥락 인식 기능을 구축했습니다. 데이터셋은 사용자 질문, 이미지, 기대 답변으로 구성되어 있으며, 테스트 세트는 맥락 인식 조건과 맥락 무관 조건으로 나누어 평가되었습니다. 평가를 위해 Gemini 2.5 Pro를 사용하고, HNSW 인덱싱을 통해 유사한 이미지를 검색하여 MLLM이 사용자 질문에 맞춰 설명을 생성하도록 지원했습니다.

- **Performance Highlights**: 연구 결과, 맥락 인식 설명이 76.1%의 정확도로 사용자 질문에 대한 응답을 제공한 반면, 맥락 무관 설명은 63.0%의 정확도를 보였습니다. 특히, 맥락 인식 설명이 사용자 질문을 예측하고 대답한 경우가 15.2%로, 맥락 무관 설명이 실패한 경우에도 사용자 질문에 대한 직접적인 답변 또는 간접적인 단서를 제공하는 성과를 보였습니다.



### Consistent Assistant Domains Transformer for Source-free Domain Adaptation (https://arxiv.org/abs/2510.01559)
- **What's New**: 본 논문에서는 Consistent Assistant Domains Transformer (CADTrans)를 제안하여 소스 도메인 데이터에 직접 접근하지 않고도 목표 도메인에 대한 적응 문제를 해결합니다. CADTrans는 도메인 일관성을 통해 불변 특징 표현을 구축하여 기존의 방법들이 직면한 제한을 극복합니다. 특기할 만한 것은, 하드 샘플을 쉽게 샘플에 정렬하기 위해 조건부 다중 커널 최대 평균 차이(CMK-MMD) 전략을 도입한 것입니다.

- **Technical Details**: CADTrans는 비전 트랜스포머(Vision Transformer, ViT)를 기반으로 하며, 중간 레이어에서의 다중 글로벌 주의를 통해 다양한 표현을 확보하기 위해 보조 도메인 모듈(Assistant Domain Module, ADM)을 도입합니다. 이를 통해 CADTrans는 균일한 특징 표현을 확보하고, 알려진 소스 도메인 없이도 다양한 표본을 구성하는 데 도움을 줍니다. 모델은 두 단계로 훈련되며, 첫 번째 단계에서는 감독 학습과 자기 증류(self-distillation)를 이용하고, 두 번째 단계에서는 목표 도메인 모듈을 통해 도메인 적응을 수행합니다.

- **Performance Highlights**: 다양한 벤치마크(Office-31, Office-Home, VISDA-C, DomainNet-126)에서의 실험을 통해 CADTrans의 성능이 기존 SFDA 방법들과 비교하여 유의미한 향상을 보임을 확인하였습니다. 특히, CADTrans는 하드 샘플을 효과적으로 쉽게 샘플에 정렬하여 도메인 이동 문제를 개선합니다. 이러한 결과는 CADTrans가 여러 도메인 적응 벤치마크에서 우수한 성능을 발휘하는 것을 증명합니다.



### Robust Classification of Oral Cancer with Limited Training Data (https://arxiv.org/abs/2510.01547)
- **What's New**: 이 논문에서는 오랄 암(oral cancer)의 조기 진단을 위해 신뢰성 있는 하이브리드 모델을 제안합니다. 이는 합성곱 신경망(CNN)과 베이지안 딥러닝(Bayesian deep learning)을 결합하여 소규모 교육 세트(small training sets)에서 오랄 암을 분류하는 것입니다. 전통적인 CNN 모델이 일반적으로 데이터 양에 의존하는 한계를 극복하고, 데이터 부족 환경에서도 우수한 일반화 성능을 제공하도록 설계되었습니다.

- **Technical Details**: 제안된 모델은 변분 추론(variational inference)을 통해 예측의 신뢰성(sentiment reliability)을 높이며, 스마트폰으로 촬영한 사진 색상 이미지(color images)를 기반으로 훈련되었습니다. 이 모델은 세 가지 서로 다른 테스트 데이터 세트에서 평가되었으며, 제한된 데이터에서도 높은 정확도로 예측할 수 있도록 설정되어 있습니다. 94%의 정확도를 달성한 이 모델은 트레이닝 데이터와 유사한 분포에서도 높은 신뢰도를 보였습니다.

- **Performance Highlights**: 제안된 베이지안 모델은 다양한 데이터 세트에서 88%의 정확도를 달성하며 전통적인 CNN의 72.94%와 비교하여 우수한 일반화 능력을 보였습니다. 또한, 예측 정확성이 높은 샘플에 대해서는 낮은 불확실성(low uncertainty)을 나타내고, 잘못 분류된 샘플에 대해서는 높은 불확실성(high uncertainty)을 보여줍니다. 이러한 결과는 베이지안 추론이 데이터가 부족한 환경에서도 조기 오랄 암 진단을 개선하는 데 효과적임을 강조합니다.



### Growing Visual Generative Capacity for Pre-Trained MLLMs (https://arxiv.org/abs/2510.01546)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 Bridge라는 순수 자가 회귀(unified autoregressive) 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLM)을 제안합니다. 기존의 MLLM들은 비주얼 이해만 가능했지만, Bridge는 이미지 이해 및 생성을 단일한 다음 토큰 예측(next-token prediction) 프레임워크 내에서 수행할 수 있도록 해줍니다. 또한 semantic-to-pixel 이산 표현(discrete representation)을 도입하여 시각적 생성을 개선했습니다.

- **Technical Details**: Bridge는 사전 훈련된 비주얼 인코더와 Mixture-of-Transformers 아키텍처를 통해 이미지의 이해와 생성을 결합합니다. 이 모델은 compact한 의미 토큰과 정밀한 픽셀 토큰을 통합하여 언어 정렬(strong language alignment) 및 시각적 세부 사항의 정확한 설명을 가능하게 합니다. 데이터 사용 및 전체 훈련 절차도 상세히 설명되었습니다.

- **Performance Highlights**: 다양한 다중 모달 벤치마크에서 광범위한 실험을 통해 Bridge는 이해 및 생성 벤치마크에서 경쟁력 있는 성과 또는 우수한 성과를 달성하였습니다. 이전의 통합 MLLM보다도 적은 학습 데이터와 짧은 훈련 시간으로 효과성을 입증하며, 총 훈련 효율성도 향상되었습니다.



### Towards Better Optimization For Listwise Preference in Diffusion Models (https://arxiv.org/abs/2510.01540)
- **What's New**: 이번 연구에서는 Listwise Preference Optimization (LPO)의 새로운 프레임워크인 Diffusion-LPO를 제안합니다. 이는 텍스트-이미지(T2I) 모델에 인간의 피드백을 효과적으로 통합하는 방법을 제공합니다. 기존의 Direct Preference Optimization(DPO)이 주로 pairwise 선호에 국한되어 있던 반면, Diffusion-LPO는 리스트 형태의 피드백을 활용하여 더 정교한 최적화를 달성합니다.

- **Technical Details**: Diffusion-LPO는 Plackett-Luce 모델 하에서 사용자 피드백을 이미지 리스트로 집계하고, 각 샘플이 낮게 순위가 매겨진 대안들보다 선호되도록 유도하여 전체 순위의 일관성을 강화합니다. 이는 DPO의 리스트 확장으로, 사용자의 피드백에 포함된 암묵적인 순위 정보를 활용합니다.

- **Performance Highlights**: Diffusion-LPO는 텍스트-이미지 생성, 이미지 편집, 개인화된 선호 조정 등 다양한 작업에서 효과를 입증했습니다. 실제 실험 결과, Diffusion-LPO는 비주얼 품질과 선호 조정 면에서 기존 pairwise DPO 기준을 지속적으로 초월하는 성능을 보였습니다.



### MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation (https://arxiv.org/abs/2510.01532)
Comments:
          20 pages, 6 figures. Accepted by NeurIPS 2025

- **What's New**: 본 논문에서는 반지도 학습(semi-supervised learning, SSL)에 기반한 세그멘테이션 프레임워크를 제안합니다. 특히, 생물학적으로 의미 있는 구조를 보존하면서도 차원화된 topological features를 강하게 식별할 수 있는 방안을 다룹니다. 기존 접근법들은 종종 정밀하게 정의된 임계값에 의존하여 유의미한 topological 구조를 추출하지만, 본 방법은 데이터를 동적으로 분석하여 불필요한 구조를 제거하는 데 집중합니다.

- **Technical Details**: 제안된 방법은 우선적으로 다양한 예측 간의 consistency를 확보하는 데 초점을 맞춥니다. 이를 위해, MATCH-Pair와 MATCH-Global이라는 매칭 알고리즘을 통해 다각적인 예측에서 안정적인 구조를 추출합니다. 또한, intra-topological consistency와 temporal-topological consistency라는 두 가지 유형의 일관성 제약을 도입하여 모델의 학습 효율성과 정확성을 높입니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 topological 오류를 효과적으로 줄이며, 한정된 주석 데이터로도 견고하고 정확한 세그멘테이션 결과를 달성할 수 있음을 입증하였습니다. 이 연구는 디지털 병리학 연구에 필요한 신뢰할 수 있는 세그멘테이션을 위한 기초를 제공합니다.



### WALT: Web Agents that Learn Tools (https://arxiv.org/abs/2510.01524)
- **What's New**: 본 논문은 웹 에이전트가 복잡한 브라우저 작업을 자동화할 수 있는 가능성을 제시하며, 기존 방법의 한계를 극복하기 위해 WALT(Web Agents that Learn Tools)라는 새로운 프레임워크를 소개합니다. WALT는 웹사이트에서 제공하는 기능을 역탐색하여 재사용 가능한 도구로 변환하는 방식으로 작업을 진행합니다. 기존의 UI 상호작용에 의존하는 방법이 아닌, 고수준의 기능 호출을 통해 에이전트의 계산 부담을 경감시킵니다.

- **Technical Details**: WALT는 세 가지 주요 단계를 통해 웹 도구를 학습합니다: (1) 웹 에이전트가 웹사이트의 기능을 시연하고, (2) 도구 생성 에이전트가 이를 구조화된 도구로 변환하며, (3) 테스트 에이전트가 기능을 검증합니다. 이 과정에서 에이전트는 복잡한 UI 시퀀스를 고려하는 대신, search(X)와 같은 단순한 함수 호출로 기능을 실행합니다. 도구들은 검색, 필터링, 콘텐츠 관리와 같은 작업을 포함하며, 50개 이상의 재사용 가능한 도구가 발견되었습니다.

- **Performance Highlights**: WALT는 VisualWebArena와 WebArena에서 각각 52.9%와 50.1%의 성공률을 기록하여, 이전 연구들보다 월등한 성능을 보였습니다. 추가적인 연구에서는 WALT가 발견한 도구, 다중 모달 DOM 파싱, 외부 검증이 성공률을 10%-30% 향상시키고, 평균 1.3-1.4배 더 적은 단계를 요구함을 밝혔습니다. 결과적으로 WALT는 브라우저 자동화를 보다 효율적이고 신뢰성 있는 도구 기반 접근으로 전환시킵니다.



### From Videos to Indexed Knowledge Graphs -- Framework to Marry Methods for Multimodal Content Analysis and Understanding (https://arxiv.org/abs/2510.01513)
- **What's New**: 이번 논문에서는 영상 데이터를 분석하기 위한 새로운 프레임워크를 제시합니다. 이는 다양한 pre-trained 모델을 통합하여 multi-modal 콘텐츠 분석을 위한 파이프라인을 효율적으로 프로토타입화할 수 있는 방법을 제공합니다. 특히, 지속적인 학습(continual learning)과 동적 지식 통합을 지원하는 지식 그래프(knowledge graph) 형식을 통해 영상의 정보를 쿼리할 수 있는 구조로 변환합니다.

- **Technical Details**: 연구 방법론은 세 가지 단계로 구성되어 있습니다. 첫 번째로, 최적화된 pre-trained 모델 조합을 실험하고 open-source 모델과 쉽게 결합하여 영상과 같은 시계열 다중 모드 데이터를 처리할 수 있는 프레임워크를 구축합니다. 두 번째 단계에서는 영상을 반구조적 데이터 형식인 'VideoKnowledgeBase'로 변환하는 파이프라인을 설계합니다. 마지막으로, 생성된 VideoKnowledgeBase를 쿼리 가능하고 확장 가능한 Video Knowledge Graphs로 변환하는 알고리즘을 설계합니다.

- **Performance Highlights**: 이 연구의 주요 성과는 다양한 pre-trained 모델을 통합하고 결합하여 multi-modal 콘텐츠를 이해하고 분석하기 위한 파이프라인을 구축하는 것입니다. 또한, 비디오 데이터베이스에서 정보를 쿼리할 수 있는 방법을 새롭게 제안하고, 새로운 도메인 지식을 추가할 수 있는 프로토타입 소프트웨어를 구현했습니다. 이러한 결과는 영상 분석, 이해 및 검색의 효율성을 크게 향상시킬 것으로 기대됩니다.



### AortaDiff: A Unified Multitask Diffusion Framework For Contrast-Free AAA Imaging (https://arxiv.org/abs/2510.01498)
- **What's New**: 이 연구에서는 비관찰 CT(NCCT) 스캔에서 합성 대조 CT(CECT) 이미지를 생성하면서 동시에 대동맥 내강 및 혈전의 분할(segmentation)을 수행하는 통합된 딥러닝 프레임워크인 AortaDiff를 제안합니다. 이는 기존의 다단계 파이프라인의 단점을 보완하고, 이미지 생성과 해부학적 분할을 동시에 최적화함으로써, 임상 적용에 더 적합한 접근법입니다. 우리 모델은 매개변수 공유와 반지도 학습(semi-supervised learning) 전략을 통해, 임상에서 라벨이 부족한 실제 데이터를 효과적으로 사용할 수 있도록 설계되었습니다.

- **Technical Details**: AortaDiff는 노이즈 제거 U-Net 아키텍처를 기반으로 하며, 공유된 인코더-디코더 구조를 통해 대조 CT 이미지와 분할 마스크를 동시에 예측합니다. 이것은 텍스처와 해부학적 정보가 포함된 풍부한 잠재 표현(latent representation)을 학습할 수 있게 해줍니다. 더욱이, 초기 예측(예: 대략적인 분할 마스크)이 필요 없는 구조를 채택하여, 모델이 데이터 효율성을 높이도록 설계되었습니다.

- **Performance Highlights**: 모델은 OxAAA 데이터셋에서 264명의 환자에 대해 평가되었으며, 종합적인 성능 향상을 보여주었습니다. PSNR은 25.61 dB로, 이전의 단일 작업 CDM보다 1.81 dB 높은 결과를 기록했으며, 대동맥 내강 분할의 Dice 점수는 0.89로 증가했습니다. 이러한 성과는 임상 측정 정확성을 크게 향상시켜, 혈관 내강 직경의 평균 절대 오차(MAE)를 4.19 mm로 줄였습니다.



### Purrception: Variational Flow Matching for Vector-Quantized Image Generation (https://arxiv.org/abs/2510.01478)
- **What's New**: 본 논문에서는 Purrception이라는 새로운 방법을 소개합니다. 이것은 벡터 양자화된 이미지 생성에 대한 변량 흐름 일치(Variational Flow Matching) 접근법을 사용하여 명시적인 범주적(supervision) 감독을 제공하면서도 연속적(continuous) 운반 역학을 유지합니다. Purrception은 이미지 생성의 효율성을 향상시키기 위해 연속적 방법의 기하학적 인식과 범주적 접근의 불연속 행위를 결합합니다.

- **Technical Details**: Purrception은 코드북 인덱스에 대한 범주적 사후 확률(categorical posteriors)을 학습하면서 연속 임베딩 공간에서 속도 필드(velocity fields)를 계산합니다. 이를 통해 잠재 정보의 불확실성(uncertainty)을 정량화하고 온도 조절이 가능한 생성(generation)을 가능하게 합니다. Purrception은 이러한 특정 선을 통해 이미지 생성 특성의 연속적 심도를 더욱 향상시킵니다.

- **Performance Highlights**: ImageNet-1k에서 256x256 해상도의 이미지를 생성하는 과정을 평가한 결과, Purrception은 연속적 흐름 일치 및 불연속적 흐름 일치 방법과 비교했을 때 빠른 학습 수렴을 보여 주었습니다. 또한, 경쟁력 있는 FID 스코어를 달성하여 최신 모델들과 비교해 우수한 성능을 입증했습니다.



### Data Selection for Fine-tuning Vision Language Models via Cross Modal Alignment Trajectories (https://arxiv.org/abs/2510.01454)
Comments:
          30 pages, 10 figures, 5 tables, link: this https URL

- **What's New**: 이번 연구에서는 LVLMs(대형 비전-언어 모델)에 대한 데이터 효율적인 훈련 방법을 최초로 제안합니다. 기존의 방법들이 다양한 서브셋 크기에서 무작위 선택보다 성능이 떨어진다는 점에서, 저자들은 이 문제를 해결하기 위해 기울기 유사성을 정의하고 설명합니다. 이를 통해 XMAS라는 알고리즘을 개발하여 주어진 데이터를 클러스터링하고, 중복성을 줄이며 성능을 유지할 수 있음을 입증했습니다.

- **Technical Details**: 저자들은 각 샘플의 기울기 유사성을 기반으로 한 중복 제거를 위해, 단일 레이어 변환기를 분석하고 교차 모달(attention matrices) 어텐션 행렬 간의 쌍별 거리로 기울기 거리를 근사할 수 있음을 증명하였습니다. XMAS는 소규모 프록시 VLM을 미세 조정하여 기울기 유사성을 통해 예제를 군집화합니다. 이를 통해 샘플 그룹에서 균형 잡힌 서브셋을 샘플링하여 큰 훈련 데이터에서 중복성을 제거합니다.

- **Performance Highlights**: 실험 결과, XMAS는 LLaVA-665k 데이터의 50%와 Vision-Flan 데이터의 85%를Discard하면서도 LLaVA-1.5-7B의 성능을 완벽하게 유지했습니다. 또한, 10개의 하위 벤치마크에서 훈련 속도가 1.2배 빨라졌으며, 이는 LLaVA-665k의 최상의 기준선과 비교했을 때 30% 더 많은 데이터 감소를 보였습니다.



### GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings (https://arxiv.org/abs/2510.01448)
Comments:
          preprint under review

- **What's New**: 이번 논문에서는 전 세계 이미지의 지리적 위치를 파악하는 새로운 접근 방식인 GeoSURGE를 제안합니다. GeoSURGE는 시각적 표현과 지리적 표현을 효과적으로 결합하여 지리적 임베딩(geographic embeddings)의 계층 구조를 모델링합니다. 이를 통해 이전 기법들이 가진 한계를 극복하고, 보다 높은 정밀도로 지리적 위치를 추정할 수 있습니다.

- **Technical Details**: GeoSURGE는 RGB 이미지와 그에 대한 의미적 분할(semantic segmentation) 맵을 활용하여 시각적 표현을 생성합니다. 이 방법은 CLIP 기능을 사용하여 두 개의 표현을 결합하며, 잠재적 교차주의(latent cross-attention)를 통해 시각적 표현을 발전시킵니다. 또한, 지리적 지식과 시각적 특성을 매치하여 고유한 지리적 표현을 생성합니다.

- **Performance Highlights**: 실험 결과, GeoSURGE는 25가지 평가 기준에서 22개의 벤치마크 데이터셋에서 새로운 최고치를 달성했습니다. 이는 기존 기술(Near state-of-the-art) 및 최근의 대형 비전-언어 모델(Large Vision-Language Models)보다 월등한 성능입니다. 추가적인 조건 연구(ablation studies)를 통해 이 성능 향상이 지리적 표현과 시각적 표현의 조합에 의해 주도됨을 확인했습니다.



### DisCo: Reinforcement with Diversity Constraints for Multi-Human Generation (https://arxiv.org/abs/2510.01399)
- **What's New**: 이번 논문에서는 다중 인물 생성에서의 복잡한 문제를 해결하기 위해 DisCo (Reinforcement with Diversity Constraints)라는 새로운 RL 기반 프레임워크를 소개합니다. 기존의 텍스트-투-이미지 모델들이 다중 인물 프롬프트에서 얼굴 중복, 신원 혼합 및 인물 수 잘못 세기와 같은 문제를 겪고 있었으나, DisCo는 이러한 문제를 해결합니다. 이는 사람들이 선호하는 비주얼 충실도를 유지하면서도 인물의 다양성을 직접적으로 최적화하는 방법을 제시합니다.

- **Technical Details**: DisCo는 그룹 상대 정책 최적화(Group-Relative Policy Optimization, GRPO)를 통해 플로우 매칭 모델을 미세 조정합니다. 이 과정에서 (i) 이미지 내 얼굴 유사성을 처벌하고, (ii) 샘플 간 신원 반복을 억제하며, (iii) 정확한 인원 수 유지를 강제하고, (iv) 인간 선호 점수를 통해 비주얼 충실도를 보장합니다. 단일 단계의 교육 커리큘럼을 통해 복잡성이 증가함에 따라 훈련을 안정시키며 추가 주석 없이도 사용할 수 있습니다.

- **Performance Highlights**: DiverseHumans Testset에서 DisCo는 98.6의 고유 얼굴 정확도를 달성하며, 글로벌 신원 분산에서 거의 완벽한 결과를 보였습니다. 이는 오픈 소스 및 독점 방법(예: Gemini, GPT-Image)을 초월하는 성과로, 경쟁력 있는 인지 품질을 유지하였습니다. 이러한 결과는 DisCo가 생성 모델의 오랜 신원 문제를 해결하는 확장 가능하고 주석이 필요 없는 솔루션으로 자리 잡았음을 보여줍니다.



### SPUS: A Lightweight and Parameter-Efficient Foundation Model for PDEs (https://arxiv.org/abs/2510.01370)
- **What's New**: 이번 연구에서는 Small PDE U-Net Solver (SPUS)를 소개합니다. SPUS는 다양한 파셜 미분 방정식(PDE)을 해결하기 위해 설계된 경량 모델로, 기존의 복잡한 트랜스포머 구조 대신 잔여 유넷(residual U-Net) 아키텍처를 사용합니다. 이 모델은 여러 물리 시스템을 포함하여 강력한 일반화 능력을 보여줍니다.

- **Technical Details**: SPUS는 autoregressive pretraining 전략을 통해 학습됩니다. 이 방법은 시간이 지남에 따라 상황을 예측하며, 수치해석기의 행동을 모방하는 데 초점을 맞추고 있습니다. 이를 통해 잔여 U-Net 기반의 경량 모델이 효율적으로 PDE 동역학을 모델링할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, SPUS는 제한된 파라미터 수와 최소한의 미세 조정 데이터로도 도전적인 다운스트림 PDE 과제에서 최첨단 일반화 성능을 보여주었습니다. 특히, SPUS는 수치해석 기반 모델에 비해 더 적은 파라미터로 고성능의 결과를 달성할 수 있는 가능성을 제시합니다.



### EvoStruggle: A Dataset Capturing the Evolution of Struggle across Activities and Skill Levels (https://arxiv.org/abs/2510.01362)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 스킬 습득 중 개인이 겪는 고난(struggle)의 진화를 파악하는 데 초점을 맞춘 새로운 데이터셋을 소개합니다. 이 데이터셋은 61.68시간의 비디오 기록과 2,793개의 비디오, 5,385개의 주석이 달린 고난 세그먼트로 구성되어 있습니다. 76명의 참여자를 통해 수집된 이 데이터는 다양한 작업 변형을 나타내는 4가지 활동에 속하는 18개의 작업을 포함하고 있습니다.

- **Technical Details**: 고난 판단 문제는 시간적 행동 위치 파악(temporal action localization) 문제로 정의되며, 고난 세그먼트를 정확하게 식별하고 시작 및 종료 시간을 로컬라이징하는 데 중점을 둡니다. 실험 결과, Temporal Action Localization 모델은 이전에 보지 못한 작업이나 활동에서조차 고난 신호를 탐지하는 데 성공했습니다. 이러한 모델은 작업 간 전이 가능성을 보여주며, 이론적으로 다양한 기술 기반 작업에 적용될 수 있습니다.

- **Performance Highlights**: 모델들은 작업 전반에 걸쳐 34.56% 평균 mAP를 달성했으며, 활동 전반에서는 19.24%의 성과를 보였습니다. 이는 고난이 다양한 기술 기반 작업 간에 전이 가능한 개념임을 나타내지만, 여전히 고난 탐지에서의 추가 개선이 필요하다는 점을 시사합니다. 연구자가 개발한 데이터셋은 공개되어 있으며, 더 많은 연구자들에게 도움을 줄 것으로 기대됩니다.



### Image Generation Based on Image Style Extraction (https://arxiv.org/abs/2510.01347)
- **What's New**: 이 연구는 텍스트에서 이미지로 변환하는 모델을 기반으로, 미세한 스타일을 자연어로 정밀하게 설명하고 제어하는 데 어려움이 있음을 지적합니다. 새로운 접근법으로, 단일 스타일 참조 이미지로부터 세밀한 스타일 표현을 추출하고 이를 생성모델에 주입함으로써, 전통적인 텍스트 기반 안내 생성에서의 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법론은 전이 학습된 생성 모델의 생성 능력을 극대화하기 위해 세 단계의 훈련 절차를 포함합니다. 스타일 인코더(style encoder)와 스타일 투영 레이어(style projection layer)를 사용하여 스타일 표현(style representations)과 텍스트 표현(textual representations)을 정렬합니다. 이를 통해 미세 조정된 텍스트 신호 기반 스타일 가이드를 생성할 수 있도록 구성되었습니다.

- **Performance Highlights**: 이 연구에서 구축된 Style30k-captions 데이터셋은 이미지, 스타일 레이블, 텍스트 설명의 삼중항을 포함하고 있으며, 스타일 인코더와 스타일 투영 레이어의 훈련에 사용됩니다. 이로 인해 보다 정밀한 스타일 제어가 가능해져, 다양한 응용 분야에 적합한 스타일화된 이미지 생성을 이끌어낼 것으로 기대됩니다.



### LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration (https://arxiv.org/abs/2510.01339)
Comments:
          23 pages, 12 figures

- **What's New**: 이 논문에서는 고해상도 비디오 복원에 대한 새로운 접근 방식인 LVTINO를 제안합니다. 이는 Video Consistency Models (VCMs)을 활용하여 개발된 최초의 제로샷(Zero-shot) 역솔버로, 기존의 프레임 단위 이미지 기반 접근 방식의 한계를 극복합니다. 이러한 접근 방식은 시간적 일관성을 유지하면서 비디오 복원 문제를 해결합니다.

- **Technical Details**: LVTINO는 VCM에서 인코딩된 프라이어(prior)를 활용하여 자동 미분(automatic differentiation) 없이 비디오 복원 품질을 향상시킵니다. 이 방법은 적은 수의 신경 함수 평가(neural function evaluations)만으로도 높은 측정 일관성과 부드러운 시간적 전환을 보장합니다. 또한, 비디오 잠재 확산 모델(video latent diffusion models)을 빠른 생성기로 변환하여 시간적 인과관계를 명시적으로 캡처합니다.

- **Performance Highlights**: 광범위한 비디오 역 문제에 대한 실험 결과, LVTINO는 기존의 이미지 LDMs 프레임 단위 접근 방식보다 상당한 지각적(perceptual) 개선 효과를 보여줍니다. 이로 인해 재구성 충실도(reconstruction fidelity)와 계산 효율성(computational efficiency)에 대한 새로운 기준을 설정하게 되었습니다. LVTINO는 최신 비디오 복원 기술에서 가장 우수한 품질을 제공하며, 실제 응용에도 유망한 성능을 입증했습니다.



### Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models (https://arxiv.org/abs/2510.02300)
- **What's New**: 이번 논문에서는 Equilibrium Matching (EqM)이라는 새로운 생성 모델링 프레임워크를 소개합니다. EqM는 기존의 비평형(Non-equilibrium) 동역학을 배제하고, 암시적(Implicit) 에너지 경관에서의 평형(Equilibrium) 기울기를 학습합니다. 이 접근법을 통해, 샘플 생성 시 최적화 기반 샘플링 프로세스를 채택하며, 이를 통해 이미지 생성의 성능을 향상시킵니다.

- **Technical Details**: Equilibrium Matching은 시간 조건 비대칭 동역학을 단일 불변 평형 기울기로 대체하여, EBM(에너지 기반 모델) 관점에서 학습합니다. 이를 통해, 샘플링 단계에서 각 샘플마다 독립적으로 적응형 옵티마이저와 단계 크기를 조정할 수 있게 되어, 60%의 함수 평가를 절약할 수 있습니다. EqM는 데이터 매니폴드에서 실체 샘플을 뽑아내고, 이를 통해 이미지 생성의 질적인 향상을 달성할 수 있습니다.

- **Performance Highlights**: 실험적으로 EqM는 ImageNet 256×256 데이터 세트에서 1.90의 FID(Fréchet Inception Distance)를 달성하며, 기존의 확산(Diffusion) 및 흐름 기반(Flow-based) 모델을 초월하는 성능을 보여줍니다. Equilibrium Matching은 다양한 크기에서 뛰어난 확장성을 갖추고 있으며, 이미지 구성, OOD 탐지, 부분적으로 노이즈가 있는 이미지 복원 등의 작업을 자연스럽게 처리할 수 있는 유연한 프레임워크입니다.



### Continual Personalization for Diffusion Models (https://arxiv.org/abs/2510.02296)
- **What's New**: 논문에서는 Concept Neuron Selection (CNS)이라는 새로운 학습 전략을 제시하여, 지속적인 학습 환경에서 퍼스널리제이션을 효율적으로 수행하는 방법을 소개합니다. CNS는 특정 개념과 관련된 뉴런을 정확히 식별하여 기존의 모델 지식을 유지하면서도 새로운 개념을 추가로 학습할 수 있도록 합니다. 이를 통해 카타스트로픽 포겟팅(catastrophic forgetting) 문제를 완화하며, 제로샷(zero-shot) 텍스트-이미지 생성 능력을 보존합니다.

- **Technical Details**: CNS의 주요 기능은 퍼스널리제이션에 필요한 개념 뉴런을 자동으로 식별하는 것입니다. 일반 뉴런과 기본 뉴런을 구분하여 개념 뉴런을 선택하며, 노드 간의 연결을 통해 점진적인 파인튜닝(incremental finetuning)을 가능하게 합니다. 기존의 방법과 달리, CNS는 사용자 지정 레이아웃이나 추가 모델 저장 없이도 멀티 개념 학습을 수월하게 처리할 수 있습니다.

- **Performance Highlights**: CNS는 실제 데이터 세트를 평가한 결과, 매개변수 조정이 최소화된 상태에서도 최첨단 성능을 달성하였습니다. 특히 단일 및 다중 개념 퍼스널리제이션 작업에서 이전 방법보다 더 나은 결과를 나타내었으며, 메모리 저장 및 처리 시간을 줄입니다. CNS는 별도의 퓨전(fusion) 없이도 효과적인 지속적 퍼스널리제이션을 제공하는 방법으로 주목받고 있습니다.



### From Behavioral Performance to Internal Competence: Interpreting Vision-Language Models with VLM-Lens (https://arxiv.org/abs/2510.02292)
Comments:
          EMNLP 2025 System Demonstration | Code: this https URL

- **What's New**: VLM-Lens는 비전-언어 모델(vision-language models, VLMs)의 체계적인 벤치마킹(benchmarking), 분석(analysis) 및 해석(interpretation)을 가능하게 하는 도구 모음(toolkit)입니다. 이 툴킷은 오픈 소스 VLM의 포워드 패스(forward pass) 중 모든 레이어에서 중간 출력을 추출하는 기능을 지원합니다. VLM-Lens는 다양한 VLM을 사용자 친화적으로 운영할 수 있도록 모델 특정 복잡성을 추상화한 YAML-configurable 인터페이스를 제공합니다.

- **Technical Details**: 현재 VLM-Lens는 16개의 최신 기본 VLM과 그 30가지 이상의 변형을 지원하며, 핵심 로직을 변경하지 않고 새로운 모델을 수용할 수 있도록 확장 가능합니다. 이 툴킷은 다양한 해석 가능성(interpretability) 및 분석 방법들과 쉽게 통합되어 사용될 수 있습니다. 또한, 레이어(layer)와 타겟 개념(target concepts) 전반에 걸쳐 숨겨진 표현(hidden representations)의 체계적인 차이를 드러내는 두 가지 간단한 분석 실험을 통해 그 사용법을 시연합니다.

- **Performance Highlights**: VLM-Lens는 VLM의 이해와 개선을 위한 커뮤니티 노력을 가속화하기 위해 오픈 소스로 출시되었습니다. 이 도구는 다양한 VLM에 대한 깊이 있는 분석과 효율적인 비교를 가능하게 하여, 연구자들이 모델 성능과 해석을 더욱 명확히 파악하는 데 도움을 줍니다. VLM-Lens의 도입으로 연구자들은 비전-언어 모델들의 레이어 간 차이를 보다 쉽게 분석할 수 있게 되었습니다.



### Test-Time Anchoring for Discrete Diffusion Posterior Sampling (https://arxiv.org/abs/2510.02291)
Comments:
          Preprint

- **What's New**: 본 연구에서는 사전 훈련된 이산 확산(Discrete Diffusion) 모델을 활용하여 잡음이 있는 측정값으로부터 이미지를 복원하는 후방 샘플링 문제를 다루었습니다. 기존의 방법들은 연속 확산 모델에 의존했으나, 본 논문은 이산 확산을 통해 다중 모달 데이터(예: 텍스트와 이미지)의 통합 모델링을 가능하게 하며, 특히 훈련 없이도 Bayesian 추론을 통해 후방 샘플링에 적합한 방법을 제시합니다. 기여 내용으로는 quantized expectation과 anchored remasking을 통한 효율적인 후방 샘플링 전략이 포함되었습니다.

- **Technical Details**: Anchored Posterior Sampling (APS) 방법은 이산 임베딩 공간에서 gradient-like guidance를 제공하는 quantized expectation 전략과, 역 과정에서 중요한 '앙커' 토큰을 조기에 복원하는 adaptive remasking 전략으로 구성됩니다. 이러한 혁신은 이산 확산의 비 미분성 문제를 극복하며, 기계 학습 모델의 성능을 극대화할 수 있도록 지원합니다. APS 방법은 다양한 선형 및 비선형 역 문제에서 기존의 이산 샘플링 모델들에 비해 성능 향상을 보였으며, 이를 통해 훈련 없는 스타일화 및 텍스트 기반 편집 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, 본 방법은 FFHQ 및 ImageNet 데이터셋에서 LPIPS 기준으로 최대 35.82% 향상과 PSNR 기준으로 10.94% 향상을 달성했습니다. 특히, 선형 및 비선형 역 문제에서 31.36% LPIPS 및 7.05% PSNR 향상을 보여 주목할 만한 성과를 기록하였습니다. 이 연구는 훈련 없는 스타일화 및 편집에서의 유연성을 강조하며, 이산 확산 모델을 기반으로 한 새로운 후방 샘플러에 대한 가능성을 제시합니다.



### Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning (https://arxiv.org/abs/2510.02268)
Comments:
          Code and project materials are available at this http URL

- **What's New**: 본 연구는 카메라 외부 매개변수(geometry)를 명시적으로 조건화하여 시점 불변(인버전) 모방 학습(imitation learning)을 개선하는 방법을 제안합니다. 픽셀별 Plücker 임베딩(embeddings)을 사용하여 기존 정책 모델이 다양한 시점에 대해 일반화할 수 있도록 도왔습니다. 특히, ACTION, Diffusion Policy 및 SmolVLA와 같은 행동 클로닝 정책에서 시점 조건이 성능을 크게 향상시켰음을 보여줍니다. 작업 평가를 위해 RoboSuite와 ManiSkill에서 여섯 가지 조작 작업을 도입하여 장면 변형을 분석했습니다.

- **Technical Details**: 연구에서는 카드 폴리시 모델에 대한 카메라 지오메트리(geometry)의 구체적인 정보를 제공하기 위해 픽셀 입력 공간과 호환되는 방식으로 모델을 조건화합니다. 이를 위해 각 화소에 대하여 6차원 Plücker 좌표의 ray-map을 사용하며, 이는 3차원 공간에서의 그래픽 표현을 수용하는 데 유용합니다. 특히, 카메라의 내재 매트릭스(intrinsic matrix) 및 외부 매개변수(extrinsic parameters)를 활용하여 각 픽셀에서의 레이 방향을 계산하고 이를 정책에 통합하는 방안을 제시했습니다.

- **Performance Highlights**: 정확한 카메라 기하학 정보를 조건으로 삼음으로써 정책의 robust성과 generalization성이 향상되었습니다. 특정 실험에서는 시점 변화에 대해 기존의 정책보다 더 뛰어난 성능을 보였으며, 명시적 카메라 조건 설정이 RGB 이미지를 사용하여 깊이 정보 없이도 안정적인 제어를 가능하게 했습니다. 연구팀은 광범위한 벤치마크 작업을 통해 이러한 정책을 테스트하고 성능 향상의 원인을 분석했습니다.



### The Unreasonable Effectiveness of Scaling Agents for Computer Us (https://arxiv.org/abs/2510.02250)
Comments:
          23 pages, 7 figures, 10 tables

- **What's New**: 이번 연구에서는 컴퓨터 사용 에이전트(CUAs)의 넓은 스케일링을 위한 새로운 방법인 Behavior Best-of-N(bBoN)을 소개합니다. bBoN은 에이전트의 롤아웃을 생성하고 이를 비교하기 위해 행동 서사를 사용하여 여러 롤아웃 간의 선택을 가능하게 합니다. 이를 통해 저항력과 성공률이 크게 향상되었으며, 기존 방법들과 비교했을 때 놀라운 성능 개선을 달성했습니다.

- **Technical Details**: CUAs는 부분 가시성 마르코프 결정 과정(POMDP)으로 모델링되며, 상태 공간, 관찰 공간 그리고 행동 공간으로 구성됩니다. 본 연구는 다수의 기초 모델과 정책을 사용하여 후보 솔루션 경로의 수를 스케일링하고 최적의 솔루션 선택을 위한 효과적인 방법을 제안합니다. 이는 기존의 단계별 BoN 방법과는 달리, 여러 기본 에이전트에 의해 생성된 후보 경로 중에서 최상의 경로를 선택하는 접근 방식을 취합니다.

- **Performance Highlights**: bBoN 메서드는 OSWorld 벤치마크에서 69.9%의 성공률로 새로운 state of the art(SoTA)를 달성하였습니다. 이는 이전 최상의 59.9%를 크게 초과하였으며, 인간 수준의 성능에 가까운 72%를 근접하게 합니다. 또한, WindowsAgentArena 및 AndroidWorld에 대한 강력한 제로샷 일반화 결과를 보여주어, bBoN의 성능을 더욱 입증했습니다.



### The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models (https://arxiv.org/abs/2510.02230)
Comments:
          23 pages, 15 figures

- **What's New**: 본 논문은 RLVR(Reinforcement Learning with Verifiable Rewards) 훈련의 독특한 문제, 즉 추론 경계의 축소(shrinkage)를 탐구합니다. 연구 결과, RLVR이 특정 훈련 문제를 해결하는 과정에서 다른 문제의 올바른 해결책을 생성할 가능성을 감소시키는 부정적인 간섭(negative interference) 현상이 발생한다는 사실을 밝혀냈습니다. 또한, RLVR이 기본 모델에서 높은 확률로 해결되는 문제에 대해서만 강한 강화를 제공하고, 초기 확률이 낮은 문제에 대해서는 소홀히 하는 승자독식(winner-take-all) 현상도 발견했습니다.

- **Technical Details**: 이 논문은 RLVR의 학습 역학을 심도있게 분석하며, 각 문제는 독특하고 알려지지 않은 보상 함수(reward function)를 가지는 개별적인 마코프 의사결정 과정(MDP)을 유도함을 설명합니다. 특정 문제를 해결하는 학습이 다른 문제를 해결하는 능력에 악영향을 미칠 수 있다는 점에 주목하며, 이는 RLVR에서의 부정적인 간섭 효과의 주된 원인으로 지목됩니다. RLVR의 전형적인 목표가 현 정책(on-policy sampling) 학습을 기반으로 하여 제한된 해결 전략으로 수렴하게 한다는 점도 강조합니다.

- **Performance Highlights**: SELF(Selective Examples with Low-likelihood and Forward-KL)라는 새롭고 효과적인 데이터 큐레이션 알고리즘을 제안하여, 올바른 답변을 도출할 가능성이 낮은 문제에 집중합니다. 이 알고리즘을 통해 RLVR의 Pass@k 성능이 크게 향상되는 것으로 나타났습니다. 실험적 결과는 SELF가 샘플 효율성을 높이고 RLVR에서의 문제 범위 축소를 효과적으로 완화함을 증명합니다.



### Measurement-Guided Consistency Model Sampling for Inverse Problems (https://arxiv.org/abs/2510.02208)
Comments:
          5 pages, 3 figures, submitted to IEEE Signal Processing Letters

- **What's New**: 이번 연구에서는 inverse problem reconstruction을 위해 수정된 consistency sampling 접근 방식을 제안합니다. 이 접근법은 measurement-consistency 메커니즘에 의해 확률적 샘플링을 유도하여, 관측된 측정값에 대한 충실도를 유지하면서도 효율적인 consistency 기반 생성을 가능하게 합니다. Fashion-MNIST 및 LSUN Bedroom 데이터셋에서의 실험 결과는 퍼셉추얼 및 픽셀 수준의 메트릭이 기존 baseline consistency sampling에 비해 일관된 향상을 보이는 것을 보여줍니다.

- **Technical Details**: 이 연구는 consistency models (CMs)의 샘플링 과정에서의 문제를 해결하기 위해 measurement-driven 가이던스를 도입합니다. 이를 통해 샘플러는 재구성이 관측값에서 벗어날 때 탐색을 주입하고, 측정 충실도가 향상됨에 따라 안정적인 결정론적 업데이트를 제공합니다. CMs의 샘플링 과정에서 DDIM(denoising diffusion implicit model) 방식이 활용되며, 샘플에서 직접적인 측정 일관성을 통합하여 adaptation합니다.

- **Performance Highlights**: 실험 결과, 제안된 샘플링 방법은 FID(Fréchet Inception Distance) 및 KID(Kernel Inception Distance)와 같은 퍼셉추얼 메트릭과 PSNR(peak signal-to-noise ratio), SSIM(structural similarity index measure)와 같은 왜곡 메트릭에서 baseline CMs 샘플링 대비 향상된 성능을 보였습니다. 이 방법은 몇 단계의 샘플링만으로도 경쟁력 있는 재구성을 이루어내어, 시간에 민감한 또는 대규모 설정에서의 활용 가능성을 제시합니다.



### Uncovering Semantic Selectivity of Latent Groups in Higher Visual Cortex with Mutual Information-Guided Diffusion (https://arxiv.org/abs/2510.02182)
- **What's New**: 이번 연구에서는 MIG-Vis라는 새로운 방법을 제안하여 고차 시각 영역의 신경 집단이 인코딩하는 시각-의미적 특성을 시각화하고 검증합니다. 이 방법은 변별 있는 잠재 공간을 추정하기 위해 변분 오토인코더를 사용하며, 각 잠재 그룹에 의해 인코딩된 시각-의미적 특징을 이해하기 쉽게 표현할 수 있게 합니다. MIG-Vis는 두 마카크의 IT 피질에서의 다중 세션 신경 스파이크 데이터 세트를 검증한 결과, 다양한 시각 특징에 대한 명확한 의미적 선택도를 확인했습니다.

- **Technical Details**: MIG-Vis는 고차 시각 피질의 신경 잠재 그룹을 정의하기 위해 그룹 기반 변별적 변동 오토인코더를 활용합니다. 이 방법은 생성적 확산 모델을 사용하여 각 신경 잠재 그룹이 인코딩하는 특정 시각-의미적 특성을 시각화합니다. 우리의 목표는 각 잠재 그룹이 인코딩하는 시각-의미적 특성을 특성화하고 이해하는 것입니다.

- **Performance Highlights**: MIG-Vis의 실험 결과는 인코딩된 시각-의미적 정보의 전반적인 분포를 충실하게 표현하는 이미지를 합성하는 데 성공적이었습니다. 각 잠재 그룹에서의 시각-의미적 선택성은 서로 다른 범주 간의 변동, 범주 내 포즈 및 내용 세부 사항을 명확히 나타냅니다. 이 연구는 고차 시각 피질에서의 구조화된 의미적 표현을 직접적으로 증명하며, 우리가 시각 정보 인코딩 원칙을 이해하는 데 기여합니다.



### DisCo-Layout: Disentangling and Coordinating Semantic and Physical Refinement in a Multi-Agent Framework for 3D Indoor Layout Synthesis (https://arxiv.org/abs/2510.02178)
- **What's New**: 이번 논문에서는 3D 실내 레이아웃 합성을 위한 새로운 프레임워크인 DisCo-Layout을 제안합니다. DisCo-Layout은 물리적(Physical) 및 의미적(Semantic) 정제를 분리하고 조정하여 더 신뢰성 있는 레이아웃을 생성합니다. 구체적으로, 두 개의 특화된 도구인 Semantic Refinement Tool (SRT)와 Physical Refinement Tool (PRT)를 통해 독립적인 정제 과정을 구현합니다.

- **Technical Details**: DisCo-Layout의 SRT는 추상적인 객체 관계를 수정하고, PRT는 격자 매칭 알고리즘을 사용하여 공간적 문제를 해결합니다. 이러한 두 도구가 함께 작동하여 의미적 및 물리적 기준 간의 충돌을 피하고 모듈성을 보장합니다. 또한, 다중 에이전트 시스템을 통해 이러한 도구들이 협력적으로 작업하도록 조정하여 더 향상된 결과를 제공합니다.

- **Performance Highlights**: DisCo-Layout은 현실적이고 일관된 3D 실내 레이아웃을 생성하는 우수한 성능을 보여줍니다. 다양한 자산과 자연어 지침을 통해 강력한 일반화 능력을 갖추고 있으며, 현재의 최첨단 방법들과 비교해 뛰어난 의미적 정확성과 물리적 일관성을 나타냅니다. 이러한 연구는 3D 실내 레이아웃 합성의 새로운 기준을 설정하는 데 기여할 것입니다.



### SpurBreast: A Curated Dataset for Investigating Spurious Correlations in Real-world Breast MRI Classification (https://arxiv.org/abs/2510.02109)
Comments:
          Accepted for publication in the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2025

- **What's New**: 이 연구에서는 SpurBreast라는 새로운 커리키팅된 유방 MRI 데이터셋을 소개합니다. 이 데이터셋은 고의적으로 비정상적인 상관관계를 포함하여 모델의 성능에 미치는 영향을 평가할 수 있게 설계되었습니다. 연구진은 환자, 장치, 이미징 프로토콜을 포함한 100개 이상의 특성을 분석했고, 두 가지 주요 비정상 신호인 자기장 강도와 이미지 방향성을 확인했습니다.

- **Technical Details**: SpurBreast 데이터셋은 DUKE 유방암 데이터셋을 기반으로 하며, 900명 이상의 환자의 3D MRI 스캔으로 구성되어 있습니다. 데이터는 양성 및 음성 종양 슬라이스로 나뉘며, 총 100개 이상의 특성을 가지고 있지만 대부분의 분포는 불균형합니다. 기계 학습에서 데이터를 훈련, 테스트, 검증 서브셋으로 임의로 나누는 전통적인 접근 방식 대신, 이 연구에서는 명확하게 정의된 비정상 상관관계를 포함한 데이터셋을 생성하기 위해 독창적인 방법론을 적용했습니다.

- **Performance Highlights**: 모델은 ResNet-50과 Vision Transformer (ViT-B/16) 아키텍처를 사용하여 훈련되었습니다. 연구 결과, 비정상적인 신호를 활용하여 높은 검증 정확도를 달성했지만, 불편견이 없는 테스트 데이터세트에서는 성능 저하가 두드러졌습니다. 따라서 연구팀은 비정상 상관관계가 임상적으로 유의한 패턴에 미치는 영향을 이해하는 기준을 마련하고, 보다 강력하고 일반화 가능한 AI 모델 개발이 가능하도록 했습니다.



### Spec-Gloss Surfels and Normal-Diffuse Priors for Relightable Glossy Objects (https://arxiv.org/abs/2510.02069)
- **What's New**:  이 논문에서는 glossy 객체의 정확한 재구성과 조명을 재현하는 새로운 접근 방식을 제안합니다. 기존의 BRDF 모델이나 파라미터화 방법의 한계를 극복하기 위해, microfacet BRDF와 specular-glossiness 파라미터화를 2D Gaussian Splatting에 통합한 relightable 프레임워크를 개발했습니다. 이 방식은 물리적 일관성이 있는 재료 분해를 가능하게 하고, 초기 최적화를 안내하여 모호성을 완화합니다.

- **Technical Details**:  제안된 방법은 surface normals와 diffuse color를 위한 diffusion-based priors를 활용하여 geometry와 material 간의 혼합를 줄입니다. 환경 맵의 coarse-to-fine 최적화는 수렴 속도를 높이고 고해상도 specular 반사를 보존하는 데 기여합니다. surfel 기반 씬 표현을 통해 고주파의 specular 효과를 효과적으로 설명할 수 있도록 제안된 모델이 개선되었습니다.

- **Performance Highlights**: 실험 결과, 이 방법은 복잡한 glossy 장면에서 높은 품질의 기하학 및 재료 재구성을 이뤄냈습니다. 또한, 새로운 조명 하에서도 기존 Gaussian splatting 방법 대비 훨씬 더 현실적이고 일관된 재조명을 제공합니다. 이로 인해 고품질의 재구성과 사실적인 재조명 결과를 달성하여 이전 접근 방식들보다 우수한 성능을 입증했습니다.



### A Multicentric Dataset for Training and Benchmarking Breast Cancer Segmentation in H&E Slides (https://arxiv.org/abs/2510.02037)
Comments:
          Our dataset is available at this https URL , our code is available at this https URL , and our benchmark is available at this https URL

- **What's New**: 이번 연구에서는 유방암 전체 슬라이드 이미지(WSIs)의 자동화된 의미론적 분할(Semantic Segmentation)을 위한 데이터셋인 BrEast cancEr hisTopathoLogy sEgmentation (BEETLE)을 소개합니다. BEETLE은 587개의 생검(Biopsy) 및 절제 수술(Resection) 샘플로 구성되며, 세 개의 임상 센터와 두 개의 공개 데이터셋에서 수집되었습니다. ENS 인자들로 다양하게 수집된 주석(Annotation)이며, 이 데이터셋은 기존의 데이터셋에서 잘 반영되지 않은 형태학적(diversity) 다양성을 보완합니다.

- **Technical Details**: BEETLE 데이터셋은 다양한 주석 전략을 활용하여, 침습성 상피(Invasive Epithelium), 비침습성 상피(Non-invasive Epithelium), 괴사(Necrosis) 등 네 가지 클래스로 주석이 달려 있습니다. 이 데이터셋은 치유적 정보를 강조하는 H&E 염색 슬라이드에서 특정 조직 구조를 세밀하게 분할하는 모델을 개발하여, 치료 계획에 필요한 예측 인자(prognostic and predictive biomarkers)를 정확히 평가하도록 돕습니다. 연구팀은 여러 임상 센터에서 사례를 수집하여 세 가지 다른 스캐너를 이용하여 디지털화하였으며, 이를 통해 견고한 모델 훈련을 위한 데이터 다양성을 확보했습니다.

- **Performance Highlights**: 이 데이터셋을 통해 개발된 자동화된 의미론적 분할 모델은 유방암 진단에서의 생체표지자 자동 양적 분석을 지원할 것으로 기대됩니다. BEETLE 데이터셋은 기존의 공개 데이터셋들이 가진 한계를 극복하여, 유방암 모양의 이질성(Heterogeneity)을 효과적으로 캡처하고 있습니다. 특정 조직 패턴을 포함하여 보다 포괄적인 평가가 가능하고, 이는 새로운 방법론이 비교 가능하도록 하여 유방암 분할 연구의 진전을 촉진할 것입니다.



### $\text{G}^2$RPO: Granular GRPO for Precise Reward in Flow Models (https://arxiv.org/abs/2510.01982)
Comments:
          Github Page: this https URL

- **What's New**: 최근 온라인 강화 학습 (RL)의 확산 및 흐름 모델 통합이 생성 모델을 인간의 선호에 맞추기 위한 유망한 접근 방식으로 떠오르고 있습니다. 이 연구에서는 노이즈 제어 과정에서 확률적 미분 방정식 (SDE)을 통해 다양한 비노이즈 방향을 생성하여 강화 학습 탐사를 지원하는 새로운 Granular-GRPO ($G^2$RPO) 프레임워크를 제안합니다. 기존 방법들이 높은 가치 샘플 탐사에서 효과적이었지만 선호 정렬이 부족했던 문제를 해결하기 위해 보다 정밀한 보상 평가 방법을 도입했습니다.

- **Technical Details**: 제안된 방법에서는 Singular Stochastic Sampling 전략을 사용하여 각 SDE 변동에서 충실한 보상을 제공하며, Multi-Granularity Advantage Integration 모듈을 통해 다양한 확산 스케일에서의 장점을 집계하여 샘플링 방향을 종합적으로 평가합니다. 이 프레임워크는 기존 흐름 기반 GRPO 방법들과 비교하여 보다 정확하고 포괄적인 보상 신호를 제공합니다. 강화 학습의 모델 훈련 과정에서 보상 신호를 특정 시행과 강하게 연결하여 안정적인 최적화를 가능케 합니다.

- **Performance Highlights**: 다양한 보상 모델에 대한 실험 결과 $G^2$RPO가 기존 흐름 기반 GRPO 기준 성능을 현저하게 초과함을 보여줍니다. $G^2$RPO는 텍스트 프롬프트에 대한 적합성과 세부 사항의 충실함에서 또 다른 장점을 발휘하며, 훈련 과정에서 안정적이고 유의미한 개선을 나타냅니다. 이 연구 결과는 시각적 생성에서 인간의 선호를 보다 효과적으로 반영할 수 있는 가능성을 열어줍니다.



### ROI-GS: Interest-based Local Quality 3D Gaussian Splatting (https://arxiv.org/abs/2510.01978)
Comments:
          4 pages, 3 figures, 2 tables

- **What's New**: 이번 연구에서는 객체 인식을 기반으로 한 ROI-GS를 제안합니다. 이 프레임워크는 카메라 선택 및 객체 훈련을 통해 오브젝트의 세부사항을 극대화하면서 전체 장면 모델에 통합합니다. 이를 통해 고해상도 세부정보에 더 집중하며, 실시간 성능을 유지합니다. 실험 결과, ROI-GS는 PSNR(Local Quality)을 최대 2.96 dB 향상시키고 모델 크기를 약 17% 줄이는 등의 성과를 보였습니다.

- **Technical Details**: ROI-GS 파이프라인은 표준 3D Gaussian Splatting을 기반으로 하며, 2D 색상 이미지, 카메라 포즈 및 드문 점 구름을 입력으로 받습니다. 사용자가 지정한 ROI에 대한 카메라 선택 전략은 두 단계로 이루어지며, AABB(축 정렬 경계 상자)를 사용하여 관심 객체를 포괄하는 뷰를 선택합니다. 뷰 선택에서는 초기 필터링 단계와 고품질 객체 재구성을 위한 최적화된 뷰 선정을 포함하여, 최종적으로 99개의 입력 매개변수를 사용합니다.

- **Performance Highlights**: ROI-GS는 전체 장면의 모델에 대해 객체 중심의 훈련을 수행하며, 이를 통해 훈련 모델의 메모리 용량과 그래픽 품질을 최적화합니다. 실험 결과, ROI 특정 이미지의 50%를 Scene-GS 훈련에 포함시키고, 다양한 시점에서 더 높은 질감을 포착하는 데 유리한 결과를 도출했습니다. ROI 영역 내에서 높은 해상도를 유지하면서 동시에 전체 장면의 균형을 맞추는데 성공하였습니다.



### ZK-WAGON: Imperceptible Watermark for Image Generation Models using ZK-SNARKs (https://arxiv.org/abs/2510.01967)
Comments:
          Accepted at AI-ML Systems 2025, Bangalore, India, this https URL

- **What's New**: 이번 논문에서는 이미지 생성 모델에 대한 새로운 워터마킹 시스템인 ZK-WAGON을 소개합니다. 기존의 워터마킹 방법들이 가지는 단점을 극복하여, 고유성(proof of origin)을 검증할 수 있는 안전하고 확장 가능한 방법을 제시합니다. 더불어, 모델의 내부 정보를 노출하지 않고도 이미지 생성 모델의 신뢰성을 높이는 역할을 합니다.

- **Technical Details**: ZK-WAGON은 Zero-Knowledge Succinct Non-Interactive Argument of Knowledge (ZK-SNARKs)를 활용하여 이미지 워터마킹을 구현합니다. Selective Layer ZK-Circuit Creation (SL-ZKCC) 방법을 통해, 키 레이어를 선택적으로 회로로 변환함으로써 증명 생성 시간을 크게 단축시킵니다. 생성된 ZK-SNARK 증명(proof)은 Least Significant Bit (LSB) 스테가노그래피를 통해 생성된 이미지에 눈에 띄지 않게 삽입됩니다.

- **Performance Highlights**: 이 시스템은 GAN 및 Diffusion 모델 모두에서 테스트되었으며, 안전하고 모델에 구애받지 않는 신뢰할 수 있는 AI 이미지 생성 파이프라인을 제공합니다. 워터마킹을 통해 이미지의 출처를 검증할 수 있으며, 이는 주로 정보 왜곡 및 지적 재산권 위반을 방지하는 데 중요한 역할을 합니다.



### GFSR-Net: Guided Focus via Segment-Wise Relevance Network for Interpretable Deep Learning in Medical Imaging (https://arxiv.org/abs/2510.01919)
- **What's New**: 이 연구에서는 GFSR-Net(가이드 초점 네트워크)을 소개하여 의료 영상(medical imaging)의 해석 가능성과 신뢰성을 개선하는 방법을 제안합니다. 기존 딥러닝 모델들은 정확한 예측을 수행하지만 그 이유를 설명하지 못해 임상적 사용이 제한적이었습니다. GFSR-Net은 적은 수의 인간 주석(human annotations)을 사용하여, 사람이 직관적으로 이미지에서 집중할 지점을 근사할 수 있도록 돕습니다.

- **Technical Details**: GFSR-Net은 정확한 경계(boundaries)나 포괄적인 마킹(exhaustive markings)을 요구하지 않으며, 훈련 중 모델은 인간의 관심 영역과 정렬되는 방법을 학습합니다. 이는 진단의 의미를 갖는 특징(feature)을 점진적으로 강조하여, 다양한 자연 및 의료 이미지(예: 흉부 X선, 망막 스캔, 피부 이미지)에서 적용 가능합니다.

- **Performance Highlights**: 실험 결과, GFSR-Net은 유사한 또는 더 높은 정확도(accuaracy)를 달성하면서도, 인간의 기대에 더 부합하는 주목도 맵(saliency maps)을 생성했습니다. 이를 통해 관련 없는 패턴에 대한 의존도를 줄이고, 자동 진단 도구에 대한 신뢰도를 높이는 데 기여했습니다.



### Model Merging to Maintain Language-Only Performance in Developmentally Plausible Multimodal Models (https://arxiv.org/abs/2510.01845)
Comments:
          Accepted to the EMNLP 2025 workshop BabyLM: Accelerating language modeling research with cognitively plausible datasets

- **What's New**: 이번 논문에서는 BabyLM 챌린지의 멀티모달 트랙에 대해 언급하면서, 고급 언어 모델이 어린이들이 언어를 습득할 때 접하는 데이터보다 훨씬 많은 양의 데이터를 필요로 한다고 설명합니다. 우리는 개발상 가능한 데이터셋을 사용하여 저자원이 환경에서 언어 전용 및 멀티모달 모델을 개발하였으며, 멀티모달 모델이 기존 BabyLM 베이스라인을 초과 성능을 보였습니다. 특히, 멀티모달 모델이 언어 전용 작업에서 저조한 성능을 보인다는 점에 초점을 맞추어 모델 병합(model merging) 기술을 실험하여 문제를 일부 해결하고자 했습니다.

- **Technical Details**: 이번 연구는 언어 전용 및 멀티모달 모델을 개발하는 데 중점을 두고 있으며, 두 모델의 매개변수를 가중 선형 보간(weighted linear interpolation) 방식으로 융합하는 모델 병합 방법을 사용했습니다. 이러한 접근은 훈련 과정이 필요 없는 직관적인 방식으로 모델의 정확성과 견고성을 유지하는 데 기여했습니다. 멀티모달 모델의 훈련 단계에서 언어 전용 벤치마크에서의 성능 저하 문제를 확인하고, 이를 극복하기 위한 모델 증강 기법으로 모델 병합을 적용했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 언어 전용 벤치마크에서 멀티모달 모델의 성능 저하가 관찰되었으며, 모델 병합을 통해 언어 전용 및 멀티모달 두 벤치마크에서의 성능을 유지할 수 있었습니다. 이로써 멀티모달 모델이 텍스트 전용 작업에서도 경쟁력을 갖출 수 있는 가능성을 보여주었습니다. 저자원 환경에서도 적합한 성과를 내며, 언어 모델링의 연구에 기여할 수 있는 방향성을 제시했습니다.



### Unsupervised Dynamic Feature Selection for Robust Latent Spaces in Vision Tasks (https://arxiv.org/abs/2510.01758)
- **What's New**: 이번 논문은 동적 특징 선택(Dynamic Feature Selection, DFS)을 활용하여 레텐트 표현(latent representation)을 향상시키기 위한 새로운 접근법인 동적 데이터 선택(Dynamic Data Selection, DDS)을 제시합니다. 기존의 DFS 방법은 레이블(label) 의존성이 있어 다양한 도메인에 적용하기 어려운 반면, DDS는 비지도 학습(unsupervised learning)에서 최초로 적용된 기법입니다. DDS는 이미지 데이터에서 불필요하거나 중복된 정보를 제거하며, 선택한 특징의 위치를 유지하여 복잡한 네트워크 구조에도 쉽게 적용할 수 있습니다.

- **Technical Details**: 기술적으로, DDS는 주어진 인스턴스에 대해 최대 M개의 특징을 선택하기 위한 최소화 문제를 해결합니다. DDS 네트워크는 두 개의 주요 구성 요소로 나뉘어 있으며, 각 구성 요소는 모델을 훈련하는 데 도움을 줍니다. 이 방법은 비지도 손실 함수(unsupervised loss function)를 사용하여 입력 데이터의 중요성에 기반한 선택 과정을 수행합니다. 기본적으로, DDS는 입력 데이터를 마스킹하여 가장 관련성 높은 특징만을 오토인코더(autoencoder)에 전달하게 됩니다.

- **Performance Highlights**: 실험 결과는 DDS가 클러스터링(clustering) 및 표현 학습(representation learning) 작업에서 일반화 성능을 크게 향상시킨다는 것을 보여줍니다. DDS를 채택한 모델은 다양한 이미지 데이터셋에서 성능 개선을 이루며, 계산 비용의 증가는 최소화되었습니다. 이 연구는 비지도 문제 해결을 위한 DDS의 유용성과 용이성을 강조하며, 다양한 문제에 쉽게 적용 가능하다는 점도 부각됩니다.



### Towards Photonic Band Diagram Generation with Transformer-Latent Diffusion Models (https://arxiv.org/abs/2510.01749)
- **What's New**: 이 논문은 확산 모델(diffusion model)을 기반으로 한 최초의 포토닉 밴드 다이어그램(photonic band diagrams, BDs) 생성 방법을 제안합니다. 이 방법은 3차원(3D) 구조에 대해 일반화 및 확장이 가능하며, 포토닉 크리스탈의 빛 전파를 분석하는 데 필요한 수치적 비용을 대폭 절감하는 데 기여할 수 있습니다. 또한, 이 연구는 트랜스포머 인코더(transformer encoder)와 확산 모델을 결합하여 복잡한 간섭 및 산란 현상을 효과적으로 포착할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안된 방법은 입력 구조로부터 의미 있는 구속 조건(conditioning embeddings)을 생성하는 트랜스포머 인코더를 사용합니다. 이 구속 조건은 이후 확산 과정의 가이드를 제공합니다. 또한, 생성된 BDs 이미지를 위한 잠재 확산 모델(latent diffusion model) 및 시각적 트랜스포머 인코더가 포함되어 있어 BDs를 잠재 공간(latent space)으로 매핑합니다. 이러한 구조로 인해 3D 포토닉 구조를 구체적으로 효율적으로 모델링 할 수 있습니다.

- **Performance Highlights**: 이 연구는 포토닉 크리스탈에 대한 새로운 약속된 대체 모델(surrogate model) 개발을 통해 포토닉스 분야의 연구자들에게 도움을 줄 것으로 기대됩니다. 제안된 접근법은 복잡한 포토닉 크리스탈 디자인 문제에 대한 새로운 해결책을 제공하며, 기존 방법들과 달리 3D 구조에 적합하게 확장 가능하다는 장점이 있습니다. 이후 연구자들이 이 방법을 바탕으로 새로운 포토닉 기술 개발에 기여할 수 있기를 바랍니다.



### VaPR -- Vision-language Preference alignment for Reasoning (https://arxiv.org/abs/2510.01700)
- **What's New**: 본 논문에서 우리는 기존의 방식들이 간과한 synthetic preference annotations의 노이즈 문제를 해결하기 위한 새로운 프레임워크인 VaPR을 소개합니다. VaPR은 스타일과 길이를 유지하면서 목표 오류를 가진 rejected responses를 생성하는 하드-네거티브(hard-negative) 응답 생성 방식을 기반으로 합니다. 이를 통해 3개의 LVLM 계열인 LLaVA-V1.5, Qwen2VL, Qwen2.5VL에 대해 30,000개의 고품질 샘플로 구성된 VaPR 데이터셋을 개발하였습니다.

- **Technical Details**: VaPR은 ground truth 응답과 생성된 하드-네거티브 응답을 쌍으로 구성하여, 태스크에서 부정확한 응답을 생성하도록 LLM(large language model)의 편집 능력을 활용합니다. 각 응답은 semantic 오류를 추가해 하드-네거티브 응답으로 구성되며, 기존 연구들과 달리 태스크에 맞춘 정보를 사용하여 잘못된 응답을 만들 때 스타일과 길이를 보존합니다. 이 방법은 기존의 VLM들보다 더 신뢰할 수 있는 문맥 이해 능력을 기반으로 하여, LVLMs의 비전-언어 정렬과 추론 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: VaPR 모델은 10개의 벤치마크에서 평균 6.5%(LLaVA), 4.0%(Qwen2VL), 1.5%(Qwen2.5VL)의 성능 향상을 이루었으며, 추론 작업에서 두드러진 결과를 보였습니다. LLaVA 모델은 적은 데이터로도 향상된 결과를 나타내며, VaPR은 이진 질문에서 '예'로 답할 경향을 줄이는 데도 효과적이었습니다. 또한, VaPR-OS 실험을 통해 오픈 소스 모델이 VaPR 프레임워크를 따르면서도 매우 유사한 성능을 내는 것을 확인했습니다.



### Beyond Simple Fusion: Adaptive Gated Fusion for Robust Multimodal Sentiment Analysis (https://arxiv.org/abs/2510.01677)
- **What's New**: 이 논문은 다중 모달 감정 분석(Multimodal Sentiment Analysis, MSA)를 위해 정보의 엔트로피와 모달 중요성을 바탕으로 피쳐 가중치를 조정하는 새로운 네트워크인 Adaptive Gated Fusion Network (AGFN)을 제안합니다. 기존의 단순한 융합 기술은 모달리티의 부정확성을 간과하여 감정 예측의 성능을 저하시켰던 문제점을 해결하고자 합니다. AGFN은 노이즈가 있는 모달리티의 영향을 줄이며, 신뢰할 수 있는 신호를 우선시하여 감정 예측의 정밀도를 높입니다.

- **Technical Details**: AGFN은 정보 엔트로피 게이트와 모달 중요성 게이트라는 두 가지 방법으로 각 모달리티의 신뢰도를 모델링합니다. 정보 엔트로피 게이트는 각 모달리티의 신뢰성을 평가하여, 더 낮은 엔트로피를 가진 모달리티에 더 큰 가중치를 부여합니다. 모달 중요성 게이트는 각 샘플에 대해 모달리티의 중요성을 평가하여, 모든 모달리티를 동적으로 조정해 융합합니다.

- **Performance Highlights**: AGFN은 CMU-MOSI 및 CMU-MOSEI 데이터셋에서 이미 존재하는 강력한 기준 모델들을 초월하는 성능을 보여줍니다. CMU-MOSI에서는 Acc-2(82.75%), F1(82.68%), Acc-7(48.69%) 및 MAE(71.02%)의 결과를 기록하였으며, CMU-MOSEI에서도 각종 성능 지표에서 우위를 점하며 뛰어난 구간을 보였습니다. 따라서 AGFN은 다양한 벤치마크에서 다중 모달 감정 분석의 효과성을 지속적으로 입증하고 있습니다.



### Median2Median: Zero-shot Suppression of Structured Noise in Images (https://arxiv.org/abs/2510.01666)
Comments:
          13 pages, 6 figures, not published yet

- **What's New**: 이 논문에서는 구조화된 잡음(structured noise)을 효과적으로 제거하기 위한 새로운 제안인 Median2Median (M2M)을 소개합니다. M2M은 단일 노이즈 입력에서 유사 독립(sub-independent) 하위 이미지 쌍을 생성하는 독창적인 샘플링 전략을 도입합니다. 기존의 제로샷 방법들이 독립 동등 분포(i.i.d.) 잡음에만 효과적인 반면, M2M은 그 한계를 넘어 구조화된 잡음에 대해 강력한 성능을 발휘합니다.

- **Technical Details**: M2M 방법은 3x3 패치로 이미지를 나누고, 각 패치의 특정 위치에서 픽셀 값을 추정하여 그로부터 유사 독립 노이즈 이미지 쌍을 생성합니다. 이는 제로차원(zero-order) 및 1차원(first-order) 방향 보간을 통해 수행되며, 구조화 잡음에 의해 왜곡된 아웃라이어를 제거하기 위해 중앙값(median) 기반 여과를 적용합니다. 또한, 무작위 매칭 전략(randomized assignment strategy)을 사용하여 효과적인 샘플링 공간을 확대하고 체계적인 편향을 제거합니다.

- **Performance Highlights**: M2M은 독립 및 동일하게 분포된(i.i.d.) 잡음 하에서 기존의 제로샷 방법들과 동등한 성능을 보여주며, 상관관계가 있는 잡음 하에서는 지속적으로 우수한 결과를 도출해 냅니다. 이러한 결과는 M2M이 구조화된 잡음 억제에 대한 효율적이고 데이터 없는(data-free) 솔루션으로 확립되며, 기존의 가정에 국한되지 않는 효과적인 제로샷 잡음 제거 방법으로 발전할 가능성을 보여줍니다.



### MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics (https://arxiv.org/abs/2510.01619)
Comments:
          Accepted to NeurIPS 2025

- **What's New**: 이번 논문에서는 MPMAvatar라는 새로운 프레임워크를 제안한다. 이 프레임워크는 다중 시점을 기반으로 3D 인간 아바타를 생성하며, 특히 느슨한 의류에 대한 사실적인 애니메이션과 고품질 렌더링을 지원한다. 기존의 물리 기반 아바타와 비교할 때, MPMAvatar는 더욱 향상된 정확성과 강건성을 제공한다.

- **Technical Details**: MPMAvatar는 Material Point Method (MPM)를 기반으로 한 시뮬레이터를 사용하여 복잡한 변형과 접촉을 효과적으로 모델링한다. 이를 위해, 방향에 따라 다르게 운동 특성을 모델링할 수 있는 비등방성 물질 모델을 채택하고, 더 일반적인 메쉬로 표현된 충돌체를 처리할 수 있는 새로운 충돌 처리 알고리즘을 도입하였다. 이러한 동역학 모델링 기법은 사실적으로 렌더링되는 애니메이션을 지원하기 위해 3D Gaussian Splatting을 통해 구현된다.

- **Performance Highlights**: MPMAvatar는 기존의 물리 기반 아바타와 비교하여 동역학 모델링 정확도와 렌더링 정확도, 그리고 시뮬레이션의 강건성과 효율성 모두에서 우수한 성과를 보여준다. 특히, 이전의 학습 기반 방법으로는 달성할 수 없었던 새로운 장면 상호작용에 대한 제로샷 일반화(generalization) 능력을 입증하였다. 이는 MPMAvatar가 다양한 시나리오에서도 잘 기능할 수 있음을 보여준다.



### ActiveUMI: Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations (https://arxiv.org/abs/2510.01607)
Comments:
          technique report. The website is available at this https URL

- **What's New**: ActiveUMI는 복잡한 이중 조작을 수행할 수 있는 로봇으로 인간의 실제 시연을 전송하는 데이터 수집 시스템을 위한 프레임워크를 제안합니다. 이 시스템은 휴대 가능한 VR 원격 조작 키트와 로봇의 엔드 이펙터를 모방하는 센서 장착 컨트롤러를 결합하여 인간-로봇 운동학을 정밀한 포즈 정렬을 통해 연결합니다. ActiveUMI는 인간의 시각적 주의와 조작 간의 중요한 연결을 학습하도록 설계되어 있으며, 효과적인 데이터 수집 및 로봇 정책 생성을 가능하게 합니다.

- **Technical Details**: ActiveUMI는 자연스러운 인간의 움직임과 로봇의 구현체를 밀접하게 정렬하고, 적절한 감각 정보를 적절한 시간에 제공하는 능동적 인식을 활성화하도록 설계되었습니다. 이를 위해 특별히 설계된 휴대용 VR 원격 조작 키트를 사용하며, 이 시스템은 사용자 친화적인 하드웨어 아키텍처를 통해 구현됩니다. VR 컨트롤러에 부착된 로봇의 전용 그리퍼는 엔드 이펙터를 정확하게 모방하여 정밀한 동작을 지원합니다.

- **Performance Highlights**: ActiveUMI는 여섯 가지 도전적인 이중 조작 작업에서 평가되었으며, 이 시연에 기반하여 훈련된 정책은 평균 70%의 성공률을 달성했습니다. 비활성 인식 시스템과 비교하여 ActiveUMI는 평균 성공률을 각각 44% 및 38% 향상시켰으며, 새로운 객체와 환경에서 평가했을 때 평균 56%의 성공률을 유지함으로써 실세계에서 생성된 데이터로부터의 의미 있는 일반화를 나타냅니다.



### Aligning Video Models with Human Social Judgments via Behavior-Guided Fine-Tuning (https://arxiv.org/abs/2510.01502)
Comments:
          15 pages total, 4 figures. Includes 1 algorithm and 2 tables in the appendix

- **What's New**: 이번 연구에서는 최신 비디오 및 언어 모델이 인간이 인지하는 유사성을 어떻게 캡처하는지를 조사합니다. 기존의 상태-of-the-art 모델들이 사회적 비디오에서 인간의 인지적 유사성을 재현하지 못하는 한계를 발견하고, 이 문제를 해결하기 위한 새로운 데이터셋과 방법론을 제안합니다. 특히, 49,484개의 'odd-one-out' 유사도 판단이 포함된 대규모 데이터셋을 도입하여 비디오 모델의 개선을 위한 행동 기반의 미세 조정 방법을 제안합니다.

- **Technical Details**: 연구는 비디오 및 언어 모델 각각의 유사도 판단을 분석하기 위해 triplet OOO 비교를 수행합니다. 새로 도입된 데이터셋은 3초 길이의 비디오 클립 250개로 구성되어 있으며, 각 클립은 사회적 상호작용을 묘사합니다. 우리는 하이브리드 손실 함수인 triplet 손실과 representational similarity analysis (RSA) 손실을 결합하여 비디오 모델의 임베딩을 더욱 인간 인지 구조에 가깝게 조정합니다.

- **Performance Highlights**: 시간 기반 비디오 모델인 TimeSformer의 미세 조정을 통해 훈련 전 대비 예측된 변동성이 58% 증가하였고, 이는 인간의 판단과 높은 일치를 나타냅니다. 또한, 모델이 언어 임베딩과의 겹침을 크게 증가시켰으며, 언어 모델이 포착하지 못하는 추가적인 변동성을 설명할 수 있음을 보여줍니다. 최종적으로, 사회 정서적 속성(친밀감, 정서적 가치, 지배력 등)의 인코딩이 강화되었음을 확인하였습니다.



### On the Role of Domain Experts in Creating Effective Tutoring Systems (https://arxiv.org/abs/2510.01432)
Comments:
          Accepted to AIED 2025 Blue Sky Track

- **What's New**: 이 논문에서는 교육 AI 분야에서 전문가의 고도로 선별된 지식의 중요성을 재조명하고 있습니다. 특히 설명 가능한 AI (XAI) 기법을 활용하여 수업을 자동 생성하는 방법과, 전문가가 지정한 커리큘럼이 적응형 튜터링 시스템의 개발에 어떻게 기여할 수 있는지를 논의합니다. 이러한 전문 지식을 활용함으로써 교육 시스템의 효과성을 높일 수 있는 여러 접근 방식을 소개하고 있습니다.

- **Technical Details**: 논문에서는 설명 가능한 AI(XAI) 기법을 통해 전문 규칙을 자동 생성된 수업에 통합하는 방안을 제시합니다. 예를 들어, POMDP(부분 관찰 가능 마르코프 의사결정 프로세스) 기반의 적응형 튜터링 시스템을 활용하여 학습자의 지식 수준에 맞는 수업을 제공하는 방법에 대해 설명하고 있습니다. 또한, 전문가가 규명한 커리큘럼을 통해 교육 설정에서의 불확실성을 효과적으로 다룰 수 있는 방안을 논의합니다.

- **Performance Highlights**: 이 연구에서 제안하는 방법은 특히 에너지와 자원이 부족한 시민 과학 단체의 튜터링 시스템에 큰 도움이 될 수 있습니다. 이러한 시스템의 중요성을 보여주는 사례로는 꽃가루 매개체(pollinator) 식별을 위한 교육 시스템이 있습니다. 전문가의 지식을 활용한 자동화된 교육 시스템은 효과적인 훈련을 제공하고, 더 많은 참여자를 모집할 수 있는 기회를 제공합니다.



### Ultra-Efficient Decoding for End-to-End Neural Compression and Reconstruction (https://arxiv.org/abs/2510.01407)
Comments:
          5 pages, 4 figures, NeurIPS 2025 Workshop MLForSys

- **What's New**: 이 논문에서는 최신 신경 압축 방법의 디코더 병목 현상을 해결하기 위해 새로운 압축-재구성 프레임워크를 제안합니다. 이는 저랭크 표현(low-rank representation)을 포함한 오토인코더(autoencoder)를 사용하여 저비용의 재구성을 가능하게 합니다. 제안된 방법은 디코딩 단계의 계산 오버헤드를 크게 줄이며, 고화질 이미지 출력을 유지하면서 디코더의 계산 병목을 사실상 제거합니다.

- **Technical Details**: 본 연구에서는 벡터 양자화(vector quantization)를 결합한 저랭크 표현을 오토인코더에 통합하여 고도화된 압축 기술을 구현합니다. 주요 기술적 기여는 인코딩된 잠재 표현(encoded latent representation)에서 직접적으로 학습 가능한 저랭크 근사를 수행함으로써 디코딩 계산량을 상당히 줄이는 것입니다. 이 과정은 수정된 트랜스포머 기반 인코더와 함께 결합되어 초경량 디코딩 체계를 통해 높은 품질의 출력을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 이미지 크기를 21배 이상 줄이면서도 평균 제곱 오차(Mean Squared Error, MSE)가 3.6×10−3에 도달하는 높은 압축률을 달성합니다. 또한, 이 접근 방식은 Googles Deepmind Sonnet VQVAE와 같은 최첨단 방법과 비교할 때 디코더의 계산 용량을 10배에서 100배까지 줄일 수 있음을 보여줍니다. 이는 기존의 신경 압축 방법들에 비해 실용적인 활용도를 크게 높일 것으로 기대됩니다.



### VENTURA: Adapting Image Diffusion Models for Unified Task Conditioned Navigation (https://arxiv.org/abs/2510.01388)
Comments:
          9 pages, 6 figures, 3 tables

- **What's New**: 이번 연구에서는 VENTURA라는 비전-언어 내비게이션 시스템이 소개됩니다. 이 시스템은 인터넷에서 사전 학습된 이미지 확산 모델을 통해 경로 계획을 세우며, 로봇의 다양한 인간 지시에 적응할 수 있습니다. VENTURA는 저수준 행동을 직접 예측하는 대신, 세밀하고 맥락 인지적인 내비게이션 행동을 포착하는 경로 마스크(path mask)를 생성합니다.

- **Technical Details**: VENTURA는 두 가지 주요 구성 요소로 이루어져 있습니다: 경로 마스크 생성기와 경량화된 행동 클로닝 정책입니다. 경로 마스크는 이미지 공간에서 시각적 계획을 생성하며, 이를 이용해 실행 가능한 궤적을 만들어냅니다. 훈련을 확장하기 위해, 우리는 자가 지도 추적 모델에서 유도된 경로 마스크를 이용하여 수작업으로 픽셀 수준 주석을 피하고, VLM으로 증강된 캡션을 사용합니다.

- **Performance Highlights**: 대규모 실험을 통해 VENTURA는 객체 도달, 장애물 회피 및 지형 선호 작업에서 최첨단 기초 모델보다 33% 더 높은 성공률과 54% 적은 충돌률을 기록하며 성능을 크게 향상시켰습니다. 또한, VENTURA는 이전에 보지 못한 다양한 작업 조합에도 일반화할 수 있어, 새로운 구성 능력을 보여줍니다.



### An Efficient Quality Metric for Video Frame Interpolation Based on Motion-Field Divergenc (https://arxiv.org/abs/2510.01361)
Comments:
          IEEE 17th International Conference on Quality of Multimedia Experience 2025 accepted manuscript, 7 pages

- **What's New**: 이 논문에서는 영상 프레임 보간(video frame interpolation)의 품질을 평가하기 위해 기존의 품질 지표들이 가지는 한계를 극복하는 새로운 척도인 $	ext{PSNR}_{	ext{DIV}}$를 제안합니다. 기존의 PSNR, SSIM, LPIPS와 같은 지표들은 시간적 일관성(temporal coherence)을 무시하고, FloLPIPS와 같은 최첨단 지표들은 효율성이 떨어져 실제 적용에 어려움이 있었습니다. 이 연구는 동작 발산(motion divergence) 가중치를 통해 PSNR을 개선하는 방식으로, 고전 영화 복원에서 차용된 기술입니다.

- **Technical Details**: $	ext{PSNR}_{	ext{DIV}}$는 영상의 동작 필드에서 특이점(singularities)을 강조하여 이미지 오류에 가중치를 적용하는 방식으로 작동합니다. BVI-VFI 데이터셋을 사용한 평가에서, $	ext{PSNR}_{	ext{DIV}}$는 FloLPIPS에 비해 +0.09의 Pearson Linear Correlation Coefficient를 기록하며, 2.5배 더 빠르고 4배 적은 메모리(memory)를 사용함을 보여주었습니다. 이 방법은 여러 프레임 속도(frame rates), 해상도(resolutions), 보간 방법(interpolation methods)을 포함한 180개의 시퀀스에서 검증되었습니다.

- **Performance Highlights**: 이 연구는 모든 콘텐츠 카테고리에서 일관된 성능을 유지하는 동시에, 사용된 모션 추정기(motion estimator)에 대해서도 강건성을 발휘합니다. $	ext{PSNR}_{	ext{DIV}}$의 효율성과 정확성은 빠른 품질 평가를 가능하게 하고, 비디오 프레임 보간 작업을 위한 신경망 훈련(loss function)에서 실용적으로 활용될 수 있습니다. 이 메트릭의 구현은 제공된 링크에서 확인할 수 있습니다.



### MorphGen: Controllable and Morphologically Plausible Generative Cell-Imaging (https://arxiv.org/abs/2510.01298)
- **What's New**: 이번 연구에서는 MorphGen이라는 최신 확산 기반 생성 모델을 소개합니다. 이 모델은 형광 현미경(flourescent microscopy) 이미지를 기반으로 다양한 세포 유형(cell types)과 자극(perturbations)에 대해 제어 가능한 생성을 지원합니다. MorphGen은 알려진 세포 형태와 일치하는 생물학적으로 의미 있는 패턴을 캡처하도록 훈련되어 있으며, OpenPhenom의 표현에 맞도록 조정된 손실(alignment loss)을 사용합니다.

- **Technical Details**: MorphGen은 다중 채널 물질을 RGB 이미지로 압축하는 기존 방법과는 달리, 모든 형광 채널을 함께 생성합니다. 이로 인해 세포 소기관(organelle) 구조가 보존되어, 생물학적 해석에 필수적인 세밀한 형태 분석이 가능합니다. CellProfiler 기능(features)을 활용하여 실제 이미지와의 생물학적 일관성을 입증하였으며, MorphGen은 이전의 MorphoDiff 모델보다 35% 이상 낮은 FID 점수를 기록하였습니다.

- **Performance Highlights**: MorphGen은 단일 세포 유형을 위한 RGB 이미지만 생성하는 MorphoDiff에 비해 현저한 성능 향상을 보여줍니다. 생성된 이미지는 생물학적으로 신뢰할 수 있는 패턴을 유지하고 있으며, 이는 약물 발견(drug discovery) 및 유전자 편집(gene editing) 분야에서의 적용 가능성을 높입니다. MorphGen의 코드도 공개되어 있어, 다른 연구자들이 활용할 수 있도록 지원합니다.



### From 2D to 3D, Deep Learning-based Shape Reconstruction in Magnetic Resonance Imaging: A Review (https://arxiv.org/abs/2510.01296)
- **What's New**: 현재의 3D MRI 재구성 방법론을 체계적으로 조사하는 본 리뷰는 점군(point cloud), 메쉬 기반(mesh-based), 형태 인식(shape-aware) 및 체적 모델(volumetric models)이라는 네 가지 주요 접근 방식에 초점을 맞추고 있습니다. 각 분류에 대해 최신 기술, 방법론적 기초, 한계 및 여러 해부학적 구조에 걸친 응용 분야를 분석합니다. 또한, 질병이 있는 해부학에 대한 모델의 임상 적용 가능성 및 훈련과 테스트 데이터의 영향을 강조합니다.

- **Technical Details**: 2D MRI 스택으로부터 3D 모형을 생성하는 데 있어 딥러닝(deep learning) 기술이 사용되고 있으며, CNN(Convolutional Neural Networks), GANs(Generative Adversarial Networks) 및 확산 모델(diffusion models)과 같은 혁신적인 아키텍처가 주목받고 있습니다. 본 연구는 각기 다른 인체 구조에 대한 재구성을 가능하게 하여, 딥러닝 기술이 의료 영상 처리에서 어떻게 변화를 일으키고 있는지를 설명합니다. 그 과정에서, 현재의 딥러닝 모델이 겪는 일반화 문제와 다양성 문제도 다루고 있습니다.

- **Performance Highlights**: 딥러닝 기반 3D 재구성 모델은 전통적 기법들을 넘어서는 성능을 보이며, 특히 움직임 왜곡이나 비정상적인 병리 현상이 있을 경우 더욱 효과적입니다. 다양한 해부학적 구조에 대한 고품질 3D 모델을 성공적으로 재구성하고 있으며, 임상적 진단 및 개인 맞춤형 치료 제공에 있어 그 중요성이 강조됩니다. 또한, 다중 모드 통합 및 교차 모달 프레임워크에서의 최신 연구 방향도 주목받고 있습니다.



### Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation (https://arxiv.org/abs/2510.01284)
- **What's New**: Ovi는 사운드와 비주얼을 하나의 생성 프로세스로 모델링하여 오디오-비디오 생성 방식을 혁신합니다. 기존의 복잡한 다단계 아키텍처와 달리, Ovi는 동시 동작하는 두 개의 변환기(transformer) 모듈을 통해 자연스러운 동기화를 성취하고, 별도의 파이프라인이나 후처리 정렬이 필요하지 않게 됩니다. Ovi 모델은 수많은 시간을 할애한 원시 오디오 데이터에서 훈련되어 음악 효과와 감정이 풍부한 음성을 생성할 수 있게 됩니다.

- **Technical Details**: Ovi는 구조적으로 맞춤화된 두 개의 잠재 디퓨전 변환기(DiT)로 구성되어 있으며, 이를 통해 비디오와 오디오를 블록 단위의 양방향 교차 주의(attention)로 연결합니다. 동기화 정렬을 위해, FROPE(RoPE) 스케일링을 사용하여 서로 다른 시간 해상도를 보정하면서 견고한 T5 인코더를 통해 두 개의 기둥을 동시에 조건화합니다. 이 모델은 오디오 및 비디오 데이터를 쌍으로 학습함으로써 동기화 성능을 강화합니다.

- **Performance Highlights**: Ovi는 자연스러운 음성 및 상황에 맞는 효과음을 제공하면서 영화 수준의 비디오 클립을 생성할 수 있습니다. 이 모델을 통해 특히 720×720 해상도와 24 fps에서 고품질의 동기화된 5초 클립을 생성할 수 있는 강력한 훈련 레시피를 제공합니다. Ovi는 연구 공헌 외에도 다양한 데모와 코드를 제공하여 사회에 기여하고 있습니다.



### JaneEye: A 12-nm 2K-FPS 18.9-$μ$J/Frame Event-based Eye Tracking Accelerator (https://arxiv.org/abs/2510.01213)
Comments:
          Accepted to 2026 IEEE 31st Asia and South Pacific Design Automation Conference (ASP-DAC) 2026

- **What's New**: 이번 연구에서는 JaneEye라는 에너지 효율적인 이벤트 기반의 안구 추적 하드웨어 가속기를 소개합니다. 이 시스템은 웨어러블 XR 장치에 최적화되어 있으며, 고속으로 데이터 처리가 가능한 sparse 고시간 해상도의 이벤트 데이터를 활용합니다. 또한, 새로운 ConvJANET 레이어를 포함한 초경량 신경망 아키텍처를 제시하여, 컴퓨팅 복잡성을 절반으로 줄이면서도 시간 모델링 기능은 유지합니다.

- **Technical Details**: JaneEye는 12nm ASIC 구현을 통해 400 MHz의 작동 속도를 가지며, 0.5 ms의 종료 지연 시간(2000 FPS)에 도달합니다. 하드웨어 효율성을 높이기 위해 custom linear approximations 및 fixed-point quantization을 활용했습니다. 최고 1250 Hz의 이벤트 프레임 속도를 지원하며, 3ET+ 데이터셋에서 2.45 픽셀의 오차율로 높은 정확도를 달성했습니다.

- **Performance Highlights**: JaneEye는 17.6K 매개변수만을 사용하여 높은 성능을 발휘하며, 18.9 μJ/frame의 에너지 효율성을 자랑합니다. 이는 저전력 고성능의 안구 추적 솔루션을 제공하여 차세대 XR 웨어러블 기기에 적합한 기준을 세우고 있습니다. 기존보다 훨씬 낮은 에너지 소모로 높은 정확도와 빠른 응답 속도를 실현하는 데 성공했습니다.



### Development and Evaluation of an AI-Driven Telemedicine System for Prenatal Healthcar (https://arxiv.org/abs/2510.01194)
Comments:
          Accepted at MICCAI 2025 MIRASOL Workshop, 10 pages, 5 figures

- **What's New**: 이 논문은 저소득 국가의 농촌 지역에서 의료 초음파 접근성 부족 문제를 해결하기 위해 인공지능(AI) 및 심리적 훈련을 받은 중개인을 통한 인식 시스템인 NatalIA를 제안합니다. NatalIA 시스템은 중개인이 '블라인드 스윕(blind sweep)' 프로토콜을 사용하여 진단에 중요한 태아 이미지를 수집하는 데 도움을 줍니다. 이 시스템은 표준 태아 영상을 자동으로 식별하고, 지역 전문가가 결과를 검토할 수 있는 웹 기반 플랫폼을 통합하고 있어 전반적인 의료 접근성을 개선할 수 있습니다.

- **Technical Details**: 제안된 시스템은 클라이언트-서버 아키텍처를 따르며, Angular 프레임워크를 사용하여 사용자 친화적인 프론트엔드를 구현하고, Flask를 사용하여 Python 기반의 백엔드에서 데이터 처리 및 통신을 관리합니다. 중개인들이 POCUS 장비로 획득한 초음파 영상을 웹 플랫폼에 업로드하면, 시스템이 이를 처리하고 표준 태아 영상을 자동으로 식별합니다. 이 시스템은 검토 후 전문의에게 임상 피드백을 제공하여, 사용자가 효율적으로 작업할 수 있도록 지원합니다.

- **Performance Highlights**: 실험에서 SonoNet 모델은 테스트 후 75.5%의 정확도를 달성하였으며, 임상 검증 단계에서 전문가들이 NatalIA AI 모델이 확인한 태아 영상을 확인했습니다. 또한, 사용성 연구 결과 중개인과 전문가 모두에게 긍정적인 피드백을 얻었으며, 이는 본 시스템이 저 자원 환경에서도 유용하게 사용될 수 있는 잠재력을 보여줍니다. 연구 결과는 NatalIA의 효과성과 현장 사용 가능성을 증명하며, 저소득 국가의 의료 진단 접근성 향상을 위한 중요한 기반이 될 것입니다.



### SoftCFG: Uncertainty-guided Stable Guidance for Visual Autoregressive Mod (https://arxiv.org/abs/2510.00996)
Comments:
          preprint

- **What's New**: 이 논문에서는 Autoregressive (AR) 모델을 통해 이미지 생성을 개선하기 위해 SoftCFG라는 새로운 방법을 제안합니다. SoftCFG는 불확실성을 기반으로 한 추론 기법으로, 모든 토큰에 걸쳐 적응적인 섭동을 분산시켜 생성 과정에서 발생하는 두 가지 주요 문제인 가이드 신호 감소와 과도한 가이드를 해결합니다. 이 방법은 학습이 필요 없고, 기존의 AR 파이프라인과 자연스럽게 통합될 수 있어 유연성을 제공합니다.

- **Technical Details**: SoftCFG는 각 생성된 토큰이 확실성이 가중된 가이드를 기여하게 함으로써 신호가 단계적으로 유지되도록 합니다. 이를 구현하기 위해 예측 신뢰성을 기반으로 한 가중치를 사용하며, 이 방법은 학습된 점수 표시기나 지각적 정렬 측정값을 수용할 수 있는 일반적인 프레임워크로 설계되었습니다. 또한, Step Normalization이라는 기법을 통해 생성 과정에서의 누적 섭동을 제한함으로써 안정성을 높입니다.

- **Performance Highlights**: 실험 결과, SoftCFG는 표준 Classifier-Free Guidance (CFG)에 비해 이미지 품질이 현저하게 향상됐으며, 256x256 크기의 ImageNet에서 Autoregressive 모델 가운데 최첨단의 FID를 달성했습니다. 이러한 결과는 SoftCFG가 연속적인 생성 단계 동안 신뢰성 있는 가이드를 제공하며, 시각적 일관성을 유지하는 데 도움을 줌을 나타냅니다.



### Does Bigger Mean Better? Comparitive Analysis of CNNs and Biomedical Vision Language Modles in Medical Diagnosis (https://arxiv.org/abs/2510.00411)
Comments:
          6pages,3 this http URL review of International Conference on Artificial Intelligence, Computer, Data Sciences and Applications

- **What's New**: 이번 연구에서는 가벼운 Convolutional Neural Network (CNN)과 최첨단의 zero-shot 의료 Vision-Language Model (VLM)인 BiomedCLIP 간의 비교 분석을 다룹니다. 두 가지 진단 작업, 즉 PneumoniaMNIST 데이터셋에서의 폐렴 탐지 및 Shenzhen TB 데이터셋에서의 결핵 탐지에 초점을 맞추었습니다. 실험 결과, supervised CNN이 두 경우에 모두 강력한 기준선을 제공하며, VLM은 단순한 결정 임계값 보정을 통해 성능이 크게 향상된다는 것을 보여주었습니다.

- **Technical Details**: 연구에서는 PneumoniaMNIST와 Shenzhen Chest X-ray 데이터셋의 두 가지 공공 데이터셋을 사용하여 비교 분석을 수행하였습니다. lightweight CNN 구조는 여러 단계의 convolutional block으로 구성되며, 각 블록에 대해 ReLU 활성화 및 max-pooling 연산이 적용됩니다. BiomedCLIP은 이미지와 텍스트를 결합하여 zero-shot 방식으로 평가되며, 각 클래스에 대해 생성된 텍스트 임베딩과 테스트 이미지 임베딩 간의 코사인 유사도를 계산하여 클래스를 예측합니다.

- **Performance Highlights**: 보정된 BiomedCLIP은 폐렴 탐지에서 0.8841의 F1-score를 달성하여 supervised CNN의 0.8803을 초과했습니다. 결핵 탐지의 경우 보정 덕분에 F1-score이 0.4812에서 0.7684로 급격히 향상되었고, 이는 supervised 모델의 0.7834에 근접한 성능입니다. 이러한 결과는 zero-shot VLM의 정확한 보정이 기존의 효율적이며 특정 작업에 최적화된 모델과 경쟁할 수 있는 성능을 낼 수 있음을 강조합니다.



### Segmentor-Guided Counterfactual Fine-Tuning for Locally Coherent and Targeted Image Synthesis (https://arxiv.org/abs/2509.24913)
Comments:
          Accepted at MICCAI 2025

- **What's New**: 이 논문에서는 Segmentor-guided Counterfactual Fine-Tuning (Seg-CFT)라는 새로운 방법을 제안하여 구조별 개입에서의 counterfactual 이미지 생성을 개선하는 데 중점을 둡니다. 이전의 기법들이 주로 전반적인 효과를 바탕으로 했던 반면, Seg-CFT는 세부 구조적 조정을 가능하게 하여 더 실제적인 결과를 도출합니다.

- **Technical Details**: Seg-CFT는 스칼라 값 변수(예: 왼쪽 폐의 면적)를 사용하여 counterfactual 이미지를 생성하는 데 필요한 가이드를 제공합니다. 이 방법은 기존의 세그멘테이션 맵 사용 대신 사용자 인터페이스의 단순성을 유지하며, 고해상도의 해부학적 조정을 지원합니다. 미리 훈련된 세그멘터와 결합하여, 이 방법은 DSCMs(Deep Structural Causal Models)의 counterfactual 효과를 향상시킵니다.

- **Performance Highlights**: Seg-CFT를 통해 생성된 이미지는 현실적인 흉부 X-Ray 이미지를 포함하며, 관상 동맥 질병의 모델링에서도 유망한 초기 결과를 보여줍니다. 이 연구는 단순한 세부 조정을 가능하게 하여 이미지 생성에서의 효과성과 현실성 모두를 한층 높이는 데 성공했습니다.



New uploads on arXiv(cs.AI)

### BioX-Bridge: Model Bridging for Unsupervised Cross-Modal Knowledge Transfer across Biosignals (https://arxiv.org/abs/2510.02276)
- **What's New**: 이 논문에서는 바이오신호(biosignal)의 비지도 교차 모달 지식 이전(unsupervised cross-modal knowledge transfer)을 위한 새로운 프레임워크인 BioX-Bridge를 제안합니다. 이 프레임워크는 가벼운 브리지 네트워크(bridge network)를 교육하여 중간 표현을 정렬하고 기초 모델(foundation models) 간 및 모달리티(modality) 간 정보 흐름을 가능하게 합니다. BioX-Bridge는 교육 가능 파라미터 수를 88-99% 줄이면서도 성능을 유지하거나 개선합니다.

- **Technical Details**: BioX-Bridge 프레임워크의 핵심은 두 가지 주요 구성 요소인 브리지 위치 선택 및 브리지 아키텍처 설계입니다. 이를 통해 중간 표현의 품질과 유사성을 평가하고, 효과적인 고차원 프로젝션을 위해 프로토타입 네트워크(prototype network)를 설계합니다. 이때 브릿지 네트워크만 교육하면 서로 다른 모달리티의 모델 간 상호 운용성을 확보할 수 있습니다.

- **Performance Highlights**: BioX-Bridge는 세 가지 바이오신호 데이터셋에서 여러 모달리티와 태스크를 대상으로 extensive 실험을 수행하여 기존 방법들에 비해 뛰어난 효율성을 입증했습니다. 제안된 프레임워크는 다양한 조건에서의 강건성도 확인하였으며, 성능 비교를 통해 기존 기법에 비해 상당한 이점을 보여주었습니다.



### RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems (https://arxiv.org/abs/2510.02263)
- **What's New**: 이 연구에서는 복잡한 문제에 대한 해답을 도출하기 위해 'algorithmic procedures'를 식별하고 구현하는 과정을 중심으로 새로운 접근법을 제안합니다. 특히, reasoning abstractions을 도입하여 모델이 수월하게 성공적인 추론을 배울 수 있도록 돕습니다. 이러한 방식은 모델이 문제를 해결하는 데 필요한 프로시저와 사실 지식에 대한 간결한 자연어 설명을 활용하도록 합니다.

- **Technical Details**: 새로운 RLAD(RL Abstraction and Decomposition) 방법론은 두 개의 역할을 플랫폼에서 활성화하여, 추상화를 생성하는 모델과 해결책을 생성하는 모델을 공동으로 학습시킵니다. 핵심 요소로는 curriculum training(커리큘럼 학습), 비추상화 프롬프트 포함, 그리고 보상 마스킹(reward masking) 기법이 포함되어 있습니다. 이러한 기법을 사용하면 모델의 성능이 향상되며, 고차원적인 프로시저 지식을 효과적으로 활용하여 더 어려운 문제에도 잘 일반화합니다.

- **Performance Highlights**: RLAD 방법은 AIME 2024 및 HMMT 2025의 두 수학 추론 벤치마크에서 기존 모델들보다 우수한 성능을 기록했습니다. 추상화를 기반으로 한 훈련이 더욱 좋은 일반화를 보여주며, 비추상화 프롬프트를 포함시키고 적절한 보상 마스킹을 통해 모델 성능을 크게 향상시켰습니다. 최종적으로, 커리큘럼 학습, 비추상화 프롬프트 포함, 보상 마스킹을 조합한 것이 다른 구성들보다 현저하게 우수한 결과를 가져왔습니다.



### The Unreasonable Effectiveness of Scaling Agents for Computer Us (https://arxiv.org/abs/2510.02250)
Comments:
          23 pages, 7 figures, 10 tables

- **What's New**: 이번 연구에서는 컴퓨터 사용 에이전트(CUAs)의 넓은 스케일링을 위한 새로운 방법인 Behavior Best-of-N(bBoN)을 소개합니다. bBoN은 에이전트의 롤아웃을 생성하고 이를 비교하기 위해 행동 서사를 사용하여 여러 롤아웃 간의 선택을 가능하게 합니다. 이를 통해 저항력과 성공률이 크게 향상되었으며, 기존 방법들과 비교했을 때 놀라운 성능 개선을 달성했습니다.

- **Technical Details**: CUAs는 부분 가시성 마르코프 결정 과정(POMDP)으로 모델링되며, 상태 공간, 관찰 공간 그리고 행동 공간으로 구성됩니다. 본 연구는 다수의 기초 모델과 정책을 사용하여 후보 솔루션 경로의 수를 스케일링하고 최적의 솔루션 선택을 위한 효과적인 방법을 제안합니다. 이는 기존의 단계별 BoN 방법과는 달리, 여러 기본 에이전트에 의해 생성된 후보 경로 중에서 최상의 경로를 선택하는 접근 방식을 취합니다.

- **Performance Highlights**: bBoN 메서드는 OSWorld 벤치마크에서 69.9%의 성공률로 새로운 state of the art(SoTA)를 달성하였습니다. 이는 이전 최상의 59.9%를 크게 초과하였으며, 인간 수준의 성능에 가까운 72%를 근접하게 합니다. 또한, WindowsAgentArena 및 AndroidWorld에 대한 강력한 제로샷 일반화 결과를 보여주어, bBoN의 성능을 더욱 입증했습니다.



### The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models (https://arxiv.org/abs/2510.02230)
Comments:
          23 pages, 15 figures

- **What's New**: 본 논문은 RLVR(Reinforcement Learning with Verifiable Rewards) 훈련의 독특한 문제, 즉 추론 경계의 축소(shrinkage)를 탐구합니다. 연구 결과, RLVR이 특정 훈련 문제를 해결하는 과정에서 다른 문제의 올바른 해결책을 생성할 가능성을 감소시키는 부정적인 간섭(negative interference) 현상이 발생한다는 사실을 밝혀냈습니다. 또한, RLVR이 기본 모델에서 높은 확률로 해결되는 문제에 대해서만 강한 강화를 제공하고, 초기 확률이 낮은 문제에 대해서는 소홀히 하는 승자독식(winner-take-all) 현상도 발견했습니다.

- **Technical Details**: 이 논문은 RLVR의 학습 역학을 심도있게 분석하며, 각 문제는 독특하고 알려지지 않은 보상 함수(reward function)를 가지는 개별적인 마코프 의사결정 과정(MDP)을 유도함을 설명합니다. 특정 문제를 해결하는 학습이 다른 문제를 해결하는 능력에 악영향을 미칠 수 있다는 점에 주목하며, 이는 RLVR에서의 부정적인 간섭 효과의 주된 원인으로 지목됩니다. RLVR의 전형적인 목표가 현 정책(on-policy sampling) 학습을 기반으로 하여 제한된 해결 전략으로 수렴하게 한다는 점도 강조합니다.

- **Performance Highlights**: SELF(Selective Examples with Low-likelihood and Forward-KL)라는 새롭고 효과적인 데이터 큐레이션 알고리즘을 제안하여, 올바른 답변을 도출할 가능성이 낮은 문제에 집중합니다. 이 알고리즘을 통해 RLVR의 Pass@k 성능이 크게 향상되는 것으로 나타났습니다. 실험적 결과는 SELF가 샘플 효율성을 높이고 RLVR에서의 문제 범위 축소를 효과적으로 완화함을 증명합니다.



### UpSafe$^\circ$C: Upcycling for Controllable Safety in Large Language Models (https://arxiv.org/abs/2510.02194)
- **What's New**: 이 연구에서는 기존의 LLM(대형 언어 모델) 안전성을 향상시키기 위해 UpSafe$^	heta$C라는 통합 프레임워크를 제안합니다. 이 방법은 안전을 중시한 업사이클링(safety-aware upcycling) 접근 방식을 사용하여 안전에 치명적인 레이어를 식별하고 이를 희소 Mixture-of-Experts (MoE) 구조로 변환합니다. 피험자 데이터 기초의 두 단계 SFT(단기 강화 학습)을 도입하여 안전성 분별력을 강화하면서 모델의 일반적 능력을 유지합니다.

- **Technical Details**: UpSafe$^	heta$C 프레임워크는 안전-critical layers를 찾아내고 이를 MoE 구조로 업사이클링하여, 라우터가 원래 MLP(다층 퍼셉트론)와 안전 전문가를 선택적으로 활성화하는 소프트 가드레일 역할을 합니다. 안전 온도(safety temperature) 메커니즘을 도입하여 추론 시점에서 안전과 효용 간의 균형을 동적으로 조정할 수 있는 유연한 제어를 가능하게 합니다. 두 단계의 SFT 전략을 사용하여 안전성을 더욱 강화하면서도 모델의 효용을 유지합니다.

- **Performance Highlights**: 실험 결과 UpSafe$^	heta$C는 유해 입력 및 탈출 공격에 대해 견고한 안전성 향상을 달성하면서도 일반 작업에서 경쟁력 있는 성능을 유지합니다. 안전 온도 메커니즘을 통해 추론 시점에서 세밀한 조정이 가능하며, 이는 안전과 유틸리티 간의 최적 경계(Pareto-optimal frontier)를 달성하는 데 기여합니다. 전반적으로, UpSafe$^	heta$C는 고정된 정렬(static alignment) 방식에서 동적이고 모듈화된 제어(dynamic, modular control)로의 전환을 통해 LLM 안전성의 새로운 방향성을 제시합니다.



### A Rigorous Benchmark with Multidimensional Evaluation for Deep Research Agents: From Answers to Reports (https://arxiv.org/abs/2510.02190)
- **What's New**: 이번 연구에서는 폐쇄형 언어 모델에서 외부 인지 및 정보 통합이 가능한 상호 연결된 에이전트 시스템으로의 패러다임 전환을 다룹니다. Deep Research Agents (DRAs)의 기능이 태스크 분해(task decomposition), 교차 출처 검색(cross-source retrieval), 다단계 추론(multi-stage reasoning), 구조화된 출력(structured output)을 통해 복잡한 작업에서 수행 능력을 크게 향상시킨다는 것을 보여줍니다. 이 논문은 DRAs와 보고서 스타일 응답을 위해 설계된 엄격한 벤치마크와 다차원 평가 프레임워크를 도입하여, 기존 벤치마크의 한계를 극복하고 DRAs의 전반적인 성능을 체계적으로 평가할 수 있는 기반을 마련합니다.

- **Technical Details**: 연구에서는 10개의 광범위한 주제 영역에 걸쳐 214개의 전문가가 선정한 도전적인 쿼리를 포함한 벤치마크를 소개합니다. 각 쿼리는 퍼포먼스 평가를 지원하기 위해 수동으로 구성된 참조 번들(reference bundles)을 동반하고 있으며, 세분화된 평가 기준을 처리하는 다차원 평가 프레임워크를 기반으로 하고 있습니다. 이 프레임워크는 DRAs가 생성한 장기 보고서의 평가를 포괄적으로 수행할 수 있도록 하며, 의미적 품질(semantic quality), 주제적 초점(topical focus), 검색 신뢰성(retrieval trustworthiness)과 같은 통합 점수 메트릭을 포함하고 있습니다.

- **Performance Highlights**: 대규모 실험을 통해 DRAs가 웹 검색 도구에 의해 강화된 추론 모델보다 전반적인 작업 수행과 보고서 생성 품질에서 지속적으로 우수한 성능을 나타냈습니다. 그러나 연구 결과는 아키텍처와 행동 메커니즘에서 여전히 개선이 필요하다는 점을 강조합니다. 이 연구는 DRA 시스템의 능력 평가, 구조 개선 및 패러다임 발전을 위한 확고한 기초를 제공합니다.



### FlexDoc: Parameterized Sampling for Diverse Multilingual Synthetic Documents for Training Document Understanding Models (https://arxiv.org/abs/2510.02133)
Comments:
          Accepted at EMNLP 2025

- **What's New**: 이번 논문은 FlexDoc라는 새로운 합성 데이터 생성 프레임워크를 소개합니다. 이는 Stochastic Schemas와 Parameterized Sampling을 결합하여 다국어의 반구조적 문서를 현실감 있게 생성할 수 있습니다. FlexDoc을 통해 다양한 문서 변형을 제어된 방식으로 대량으로 생성할 수 있으며, 기존 데이터셋을 보강하여 자동 추출 (Information Extraction) 작업의 정확성을 최대 11% 개선할 수 있습니다.

- **Technical Details**: FlexDoc는 합성 문서 생성을 위해 Parameters Sampling을 중심으로 한 새로운 알고리즘을 도입하고, Dynamic Virtual Grid 알고리즘을 통해 문서 요소를 비오버랩(non-overlapping) 영역으로 조직하여 시각적 다양성을 높입니다. 이 과정은 프라이버시 위험을 줄이면서 특정 지역에 맞게 조정할 수 있는 가짜 값 생성기를 사용하여 안전하고 효율적인 데이터 생성이 가능합니다.

- **Performance Highlights**: 실험 결과 FlexDoc으로 생성된 데이터는 기존 하드 템플릿 방법과 비교하여 주석 작업을 90% 이상 줄이면서도 문서 이해 모델 개발의 속도를 크게 증가시켰습니다. 이 솔루션은 현재 적극적으로 배포되고 있으며, 기업 대형 문서 이해 모델의 개발을 가속화하고 데이터 확보 및 주석 비용을 대폭 절감하고 있습니다.



### Do AI Models Perform Human-like Abstract Reasoning Across Modalities? (https://arxiv.org/abs/2510.02125)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 연구는 OpenAI의 o3-preview 모델이 ARC-AGI 벤치마크에서 인간의 정확도를 초과했지만, 이러한 모델들이 과연 문제 제작자들이 의도한 추상 개념을 인지하고 추론하는지에 대한 의문을 제기합니다. ConceptARC에서 모델의 추상화 능력을 조사하며, 다양한 입력 방식과 외부 파이썬 도구의 사용 여부에 따라 모델을 평가합니다. 정확성 외에도 모델이 생성한 자연어 규칙을 세밀하게 평가함으로써 모델이 과제를 해결하는 방식에 대한 깊이 있는 분석을 수행합니다.

- **Technical Details**: 이 연구에서는 OpenAI의 o3와 o4-mini, Google의 Gemini 2.5 Pro, Anthropic의 Claude Sonnet 4와 같은 네 가지 멀티모달(reasoning) 모델을 평가합니다. 각 모델은 간단한 추상 개념을 활용한 480개의 ConceptARC 작업을 해결하기 위해 JSON 객체를 생성하도록 요청받으며, 이를 통해 두 가지 평가를 수행합니다: 그리드 출력 정확성과 모델 생성 규칙이 과제가 의도한 추상화를 얼마나 잘 포착하는지입니다. 실험에서 텍스트와 시각 모달리티 모두에서 모델의 추상적 추론 능력을 조사하고, 추론 노력과 외부 도구 접근성이 어떻게 영향을 미치는지를 분석합니다.

- **Performance Highlights**: 결과적으로, 일부 텍스트 기반 모델은 인간의 정확성과 유사한 성과를 보였으나, 최상위 모델의 규칙은 종종 표면적인 '지름길'에 기반하여 의도한 추상화를 잘 반영하지 못했습니다. 시각 모달리티에서는 AI 모델의 정확성이 급격히 하락했지만, 규칙 수준 분석에서는 여전히 상당한 수준의 의도된 추상화를 포착할 수 있다는 것이 드러났습니다. 이는 모델들이 인간의 추상적 추론 능력에 비해 여전히 부족하다는 점을 강조하며, 정확성만을 기준으로 평가하는 것이 두 가지 모달리티에서의 추론 능력을 과대 또는 과소 평가할 위험이 있음을 보여줍니다.



### Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning (https://arxiv.org/abs/2510.02091)
Comments:
          ICASSP 2025

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 깊이 활용에 대해 체계적인 연구를 수행하며, 다양한 평가 프로토콜, 작업 범주 및 모델 아키텍처를 포함하여 깊이의 중요성을 분석했습니다. 이전의 연구 결과들은 깊은 층이 효과적이지 않다는 주장을 하였으나, 실제로는 평가 설정에 따라 크게 다를 수 있음을 확인했습니다. 연구는 특히 생성을 기반으로 한 평가가 중간 및 깊은 층의 필수적 역할을 드러내는 것으로 나타났습니다.

- **Technical Details**: 연구에서는 LLaMA-3.1-8B 및 Qwen3-8B 모델을 사용하여 MMLU 벤치마크에서 다섯 가지 평가 프로토콜을 비교했습니다. 각 평가 방법에 따라 층을 줄이고 성능 저하를 측정하여 층별 기여도를 정량화하였습니다. 그 결과, 표준 로그-우도 기반 메트릭에서는 초반 층이 상대적으로 중요하지만, 생성 기반 평가에서는 중간 및 깊은 층의 역할이 부각되었습니다.

- **Performance Highlights**: 초기 층을 제거할 경우 성능 저하가 심각함을 확인하였으며, 이는 commonsense reasoning 작업에 특히 두드러졌습니다. MathQA와 같은 수학 문제 해결 작업에서는 더 깊은 층이 보편적인 민감도를 보이지 않으며, 오히려 성능 개선을 보이는 경우도 있었습니다. 이번 연구는 LLM의 깊이 사용 이해 및 효율적인 압축을 위한 작업, 메트릭, 모델 인지의 필요성을 강조합니다.



### ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection (https://arxiv.org/abs/2510.02060)
Comments:
          9 pages, 4 figures

- **What's New**: ReTabAD는 텍스트 의미론을 복원하여 컨텍스트가 인식되는 표 형식의 이상 탐지(tabular anomaly detection, AD) 연구를 가능하게 하는 최초의 벤치마크입니다. 이 프로젝트는 구조화된 텍스트 메타데이터로 풍부하게 강화된 20개의 신중하게 선정된 표 형식 데이터 세트를 제공합니다. 또한, 특정 작업에 대한 훈련 없이 의미론적 맥락을 활용하는 제로샷 LLM(zero-shot LLM) 프레임워크를 제안하여 향후 연구의 강력한 기준점을 마련합니다.

- **Technical Details**: ReTabAD는 탭형 데이터에 세부적인 텍스트 메타데이터를 통합하여 AD 알고리즘의 성능을 극대화하는 데 중점을 둡니다. 이 구조는 17가지 알고리즘을 포괄하는 포괄적 평가를 가능하게 하며, 수치적 패턴이 아닌 의미론적 맥락에 기반하여 예측을 수행할 수 있도록 합니다. 또한, 여러 모델에서 평균 7.6% AUROC(Area Under the Receiver Operating Characteristic Curve) 성능 개선을 보이면서, 제로샷 LLM 접근 방식을 통해 훈련 기반 방법과 동등한 성과를 달성하였습니다.

- **Performance Highlights**: ReTabAD는 컨텍스트 인식 AD를 위한 체계적 연구를 지원하며, 기존 데이터 세트와의 비교에서 높은 성능을 보여줍니다. 이 새로운 벤치마크는 모델이 의미론적 정보를 통해 더욱 직관적으로 이상 상태를 해석할 수 있도록 도와주며, 예측의 해석 가능성을 향상시키는 데 기여하고 있습니다. ReTabAD는 탭형 데이터를 위한 이상 탐지 분야에서의 시스템적 발전을 이끌 것으로 기대됩니다.



### Zero-shot reasoning for simulating scholarly peer-review (https://arxiv.org/abs/2510.02027)
- **What's New**: 이 논문은 학술 출판 생태계의 위기를 해결하기 위해 정량적이고 객관적인 기준을 갖춘 AI 기반 동료 심사의 새로운 시스템을 도입합니다. 기존의 인간 심사 방식이 가지는 비효율성과 불투명성을 극복할 수 있는 모형을 제시하여, AI가 생성한 동료 검토 보고서를 평가할 수 있는 최초의 안정적이고 증거 기반의 기준을 제공합니다. 이를 통해 학술 출판의 신뢰성을 강화하고, 편집 과정의 투명성을 높일 수 있는 기회를 제공합니다.

- **Technical Details**: 본 연구에서는 xPeerd라는 제로-샷(zer0-shot) 추론 프레임워크를 사용하여 동료 심사를 시뮬레이션합니다. 이 프레임워크는 고유한 규칙 기반 의사 결정 과정을 통해 다양한 연구 분야에서 실제 심사의 동태를 모사하며, 문서의 증거에 기반한 주장을 평가하고 명확한 기준에 따라 결정을 내리는 시스템입니다. 구조적인 검증 절차를 통해 AI의 결과를 감시하고 감사할 수 있는 기반을 구성하는 것을 목표로 합니다.

- **Performance Highlights**: 352건의 동료 심사 시뮬레이션 보고서 분석을 통해, 시스템의 신뢰성을 나타내는 일관된 시스템 상태 지표가 확인되었습니다. 'Revise' 결정이 모든 학문 분야에서 50% 이상으로 증가하며, 건강 과학 분야에서는 'Reject' 비율이 45%로 높아지는 등 분야별 기준을 반영하는 의사 결정이 이루어졌습니다. 이러한 결과는 AI 기반 시스템이 예측 가능한 규칙에 따르며, 지원적 지식을 제공하는 공정한 도구가 될 수 있음을 보여줍니다.



### To Mask or to Mirror: Human-AI Alignment in Collective Reasoning (https://arxiv.org/abs/2510.01924)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 집단적 결정 처리에서 인간의 사회적 추론을 검토하는데 중점을 두고 있습니다. 연구자들은 변별 가능한 인구 통계 특성과 가명(anonymous) 별칭을 가진 리더 선출에 대한 집단 실험을 통해 집단적 정렬을 평가하는 새로운 프레임워크를 제시합니다. 이 연구는 Gemini 2.5, GPT 4.1, Claude Haiku 3.5, Gemma 3 모델의 LLM 행동을 비교하여 인간의 편견을 반영하고 수정하는 LLM의 다양한 행위를 실증적으로 파악합니다.

- **Technical Details**: 연구는 'Lost at Sea'라는 사회 심리학 과제를 기반으로 집단적 의사결정 실험(N=748)을 수행했습니다. 참가자들은 리더를 자율적으로 선정하고 그 결과에 따라 보상이 결정되며, 두 가지 조건에서 인구 통계를 표시하는 아바타를 사용하여 변별력을 조정했습니다. 연구팀은 LLM이 인구 통계 정보를 바탕으로 어떻게 행동하는지 분석하며, 특정 모델의 인덕티브 바이어스(inductive biases)가 결과에 미치는 영향을 논의합니다.

- **Performance Highlights**: 이 연구에서는 LLM 집단이 인간 집단과의 정렬도 및 최적성(optimality)을 비교 분석했습니다. 인간 집단이 명명된 조건에서 남성 리더를 선택한 비율이 64%였으며, 가명 조건에서 이러한 성비는 많이 줄어들어 최적 리더가 더 자주 선출되었습니다. Gemini 그룹은 인간 집단과의 리더 선택에서 높은 정렬율을 보인 반면, Claude 그룹은 최적 리더를 더 잘 선택했으나 정렬율은 낮았습니다. 이러한 결과는 LLM이 인간의 선택을 어떻게 모방하거나 가리거나 잘못 인식하는지를 이해하는 데 중요한 인사이트를 제공합니다.



### Constrained Adaptive Rejection Sampling (https://arxiv.org/abs/2510.01902)
- **What's New**: 이 논문에서는 CARS(Constrained Adaptive Rejection Sampling)를 소개합니다. CARS는 기존의 Rejection Sampling(RS)과 Greedy Constrained Decoding(GCD)의 단점을 해결하여 성능을 개선한 새로운 접근법입니다. 특히, CARS는 샘플 파라미터를 체계적으로 수정하여 유효한 샘플을 더 효율적으로 생성할 수 있도록 합니다.

- **Technical Details**: CARS는 Adaptive Rejection Sampling(ARS)에서 발전된 방식으로, invalid한 프리픽스(prefix)를 trie에 기록해 나중에 샘플을 생성할 때 해당 probability mass를 차감합니다. 이렇게 하면 샘플이 계속해서 무효화되는 것을 방지하고, 점진적으로 acceptance rate가 개선됩니다. 이 알고리즘의 핵심은 조건부 분포를 유지하며, 연결된 prefix를 검증하는 것입니다.

- **Performance Highlights**: 다양한 실험에서 CARS는 기존의 constrained sampling 방법들과 비교하여 높은 acceptance rate와 다양성을 보여주었습니다. 또한, CARS는 유효한 샘플을 생성하는 데 있어 평균적으로 더 낮은 비용을 요구하며, 성능 면에서 새로운 최첨단 기준을 설정했습니다. 프로그램 퍼징(program fuzzing) 및 분자 생성(molecular generation) 분야에서 특히 강력한 효율성을 기록했습니다.



### Learning a Dense Reasoning Reward Model from Expert Demonstration via Inverse Reinforcement Learning (https://arxiv.org/abs/2510.01857)
- **What's New**: 본 논문에서는 적대적 역 강화 학습(adversarial inverse reinforcement learning, IRL)을 대형 언어 모델 reasoning에 맞춰 재구성 및 실용화합니다. 특히, 전문가의 시연에서 직접 프로세스를 감독하기 위해 조밀한 토큰 수준의 보상 모델을 학습하며, 이는 단순히 스타일을 모방하는 것이 아닌 효과적인 추론 과정을 최적화하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 보상 모델은 추론 과정 내의 중간 단계를 평가하며, 훈련 시에는 정책 최적화를 위한 신호로, 추론 시에는 고품질 샘플을 선택하기 위한 보조 reranker로 활용됩니다. 또한, 이 방법은 명확하고 적정성을 보장하기 위해 전문가 시연으로부터 학습된 조밀한 보상을 사용하며, 보상은 중간 단계의 중요성에 대한 정보를 포함합니다.

- **Performance Highlights**: 실험적으로, Llama3 및 Qwen2.5 백본을 사용한 GSM8K에서 우리의 접근법은 조밀한 추론 보상이 학습 신호로 사용될 수 있으며, 보상에 따른 reranking을 통해 예측 성능이 향상됨을 보여줍니다. 이는 다단계 reasoning을 강화할 수 있는 폭넓은 가능성을 제시하며, 오류 위치의 해석 가능한 로컬라이제이션을 가능하게 합니다.



### Plan Then Action:High-Level Planning Guidance Reinforcement Learning for LLM Reasoning (https://arxiv.org/abs/2510.01833)
Comments:
          19 pages and 5 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 추론 능력을 향상시키기 위해 새로운 두 단계의 프레임워크인 PTA-GRPO를 제안합니다. 이 모델은 Chain-of-Thought (CoT) 추론을 고급 계획(guidance)과 결합하여 더 나은 결과를 도출합니다. 첫 번째 단계에서는 고급 LLM을 활용해 정교한 계획을 수립하고, 두 번째 단계에서는 강화 학습(Reinforcement Learning, RL)을 통해 최종 출력과 높은 수준의 안내 품질을 동시에 최적화합니다.

- **Technical Details**: PTA-GRPO는 고차원 계획과 추론을 촉진하기 위해 기획된 새로운 두 단계의 프레임워크입니다. 첫 번째 단계에서는 CoT를 간결한 고수준 안내로 요약하고 이를 사용하여 감독 세분화(Supervised Fine-Tuning, SFT)를 진행합니다. 두 번째 단계에서는 GRPO 알고리즘을 기반으로 한 계획 가이드 인식 강화 학습 방법이 적용되어, 최종 출력의 정확성과 고급 안내의 품질을 모두 평가하여 강화됩니다.

- **Performance Highlights**: 메시지를 통해 우리는 다양한 기초 모델(Qwen2.5-7B-Instruct, Qwen3-8B 등)에서 여러 수학적 추론 벤치마크(MATH, AIME2024 등)에 대해 PTA-GRPO를 사용한 실험을 진행했습니다. 실험 결과, PTA-GRPO는 모든 모델과 작업에서 안정적이고 상당한 성능 향상을 이루어냈으며, 이 모델의 효과성과 일반화 가능성을 확인했습니다.



### Human-AI Teaming Co-Learning in Military Operations (https://arxiv.org/abs/2510.01815)
Comments:
          Submitted to Sensors + Imaging; presented on 18th of September (Artificial Intelligence for Security and Defence Applications III)

- **What's New**: 이 논문에서는 군사 작전에서 AI의 통합이 가져다주는 이점과 이에 따른 윤리적 도전과제를 탐구합니다. 특히, 인간-AI 팀 시스템의 내부 역학을 분석하여 다차원 책임, 안전성, 강건성(aspects of safety and robustness) 문제를 논의합니다.

- **Technical Details**: 제안된 모델은 네 가지 주요 차원을 통합합니다: 첫째, 임무 상태(mission state), 시스템 신뢰도(system confidence), 환경 불확실성(environmental uncertainty)에 따라 에이전트(agents)의 자율성 수준을 조정하는 조정 가능한 자율성(adjustable autonomy)입니다. 둘째, 지속적인 감독과 활동 모니터링을 고려한 다층 제어(multi-layered control)입니다. 셋째, 에이전트 간의 명시적 및 암묵적 피드백 루프를 포함하는 양방향 피드백(bidirectional feedback)입니다. 넷째, 신뢰도(confidence levels)와 그에 대한 논리를 포함한 협력적 의사결정(collaborative decision-making)입니다.

- **Performance Highlights**: 이 모델은 계속적인 상호작용을 통해 전투 상황에 적응하도록 인간과 AI 에이전트가 공동으로 작용하는 신뢰할 수 있는 코러닝(co-learning) 방법을 제안합니다. 구체적인 사례와 실행 가능한 권장사항을 포함하여 의사결정 과정의 신뢰도를 높이는 데 기여합니다.



### REBot: From RAG to CatRAG with Semantic Enrichment and Graph Routing (https://arxiv.org/abs/2510.01800)
- **What's New**: 이 논문에서는 학생들의 대학 규정을 이해하고 준수하도록 돕기 위한 새로운 LLM 기반 상담 챗봇인 REBot을 소개합니다. REBot은 CatRAG라는 하이브리드 검색 추론 프레임워크를 통해 동작하며, 검색 강화 생성(Retrieval-Augmented Generation)과 그래프 기반 추론을 통합하여 효율성을 극대화합니다. 또한, CTU(건토 대학교) 학생들을 위한 규칙별 데이터 세트를 구축하고 평가하여 F1 점수 98.89%라는 우수한 성능을 기록했습니다.

- **Technical Details**: REBot의 핵심은 사실적 신뢰성을 제공하는 RAG와 구조화된 문맥적 추론을 위한 GraphRAG를 결합한 CatRAG입니다. 이 프레임워크는 질의 지향 텍스트 분류, Named Entity Recognition (NER), Retrieval-Augmented Generation 및 GraphRAG의 네 가지 핵심 기술을 활용하여 학생들이 원하는 정보에 빠르고 정확하게 접근할 수 있도록 지원합니다. 특히, NER을 통해 학생 질의를 개선하여 정확성을 높이고, 주제별 분류기를 통해 효율적으로 문서에서 필요한 정보를 찾습니다.

- **Performance Highlights**: REBot은 CTU 학생들에게 최신의 정확한 답변을 제공할 수 있도록 설계되었습니다. 연구 결과, REBot은 기존의 FAQ 기반 시스템에 비해 향상된 성능을 보이며, 예측하고 대화의 문맥을 이해하는 능력이 뛰어납니다. 실시간으로 규정 안내를 제공하는 웹 애플리케이션도 구현하여, 실제 학술 상담 시나리오에서 REBot의 실용성을 입증합니다.



### A cybersecurity AI agent selection and decision support framework (https://arxiv.org/abs/2510.01751)
Comments:
          6 figures, 6 tables, AI agents decision support framework

- **What's New**: 이 논문은 다양한 인공지능(AI) 에이전트 아키텍처를 체계적으로 정렬하는 새로운 구조적 결정 지원 프레임워크를 제시합니다. 이 프레임워크는 NIST 사이버 보안 프레임워크(Cybersecurity Framework, CSF) 2.0과 통합되어 현대 사이버 위협에 대응하기 위한 AI 솔루션 선택 및 배치를 위한 투명하고 단계적인 방법론을 제공합니다.

- **Technical Details**: NIST CSF 2.0의 기능을 특정 작업으로 세분화하여 AI 에이전트의 주요 속성인 자율성(autonomy), 적응 학습(adaptive learning), 실시간 반응성(real-time responsiveness)과 각 하위 범주의 보안 요구 사항을 연결합니다. 또, 사이버 보안 성숙도가 다양한 조직을 수용하기 위해 보조(assisted), 증강(augmented), 완전 자율(fully autonomous)의 몇 가지 자율성 수준을 설명합니다.

- **Performance Highlights**: 이 프레임워크는 고립된 AI 응용 프로그램을 초월하여 통합된 탐지, 사건 반응 및 거버넌스 전략을 제공합니다. 개념 검증을 통해 맞춤형 AI 에이전트 배치가 실제 제약 및 위험 프로필과 어떻게 조화를 이루는지를 보여주며, 상황 인식 향상, 반응 시간 단축 및 적응형 위험 관리(adaptive risk management)를 통한 장기적인 회복력 강화를 다룹니다.



### MetaboT: AI-based agent for natural language-based interaction with metabolomics knowledge graphs (https://arxiv.org/abs/2510.01724)
- **What's New**: Mass spectrometry metabolomics는 방대한 양의 데이터를 생성하며 이를 해석하기 위한 고급 방법이 필요합니다. Knowledge graph는 이러한 문제를 해결하기 위해 질량 분석 데이터, 대사물 정보 및 그 관계를 연결된 네트워크로 구성합니다. 이에 따라 MetaboT라는 AI 시스템이 설계되었으며, 이는 사용자 질문을 SPARQL 쿼리 언어로 변환하여 knowledge graph에서 작동하게 합니다.

- **Technical Details**: MetaboT는 사용자의 쿼리를 처리하고 지식 그래프와 상호작용하기 위해 특화된 AI 에이전트를 활용합니다. 이 시스템은 각 작업을 단위 구성요소로 분해하여 각 구성요소를 전문 에이전트가 관리하는 다중 에이전트 시스템으로 구성됩니다. 메타보틱스 지식 그래프에 대해 쿼리를 생성하는 과정은 구조화된 워크플로를 따릅니다.

- **Performance Highlights**: MetaboT의 성능을 평가하기 위해 50개의 대사체 관련 질문과 그에 대한 기대 답변을 수집하여 평가하였습니다. MetaboT는 83.67%의 정확도로 성과를 보였으나, 기준 모델인 GPT-4o는 8.16%의 정확도를 기록하여 우리의 다중 에이전트 시스템의 필요성을 강조했습니다. MetaboT는 연구자들이 자연어 쿼리를 통해 구조화된 대사체 데이터를 쉽게 검색할 수 있도록 지원하는 대화형 질문-답변 보조 도구로서 유망한 성과를 보여주고 있습니다.



### VaPR -- Vision-language Preference alignment for Reasoning (https://arxiv.org/abs/2510.01700)
- **What's New**: 본 논문에서 우리는 기존의 방식들이 간과한 synthetic preference annotations의 노이즈 문제를 해결하기 위한 새로운 프레임워크인 VaPR을 소개합니다. VaPR은 스타일과 길이를 유지하면서 목표 오류를 가진 rejected responses를 생성하는 하드-네거티브(hard-negative) 응답 생성 방식을 기반으로 합니다. 이를 통해 3개의 LVLM 계열인 LLaVA-V1.5, Qwen2VL, Qwen2.5VL에 대해 30,000개의 고품질 샘플로 구성된 VaPR 데이터셋을 개발하였습니다.

- **Technical Details**: VaPR은 ground truth 응답과 생성된 하드-네거티브 응답을 쌍으로 구성하여, 태스크에서 부정확한 응답을 생성하도록 LLM(large language model)의 편집 능력을 활용합니다. 각 응답은 semantic 오류를 추가해 하드-네거티브 응답으로 구성되며, 기존 연구들과 달리 태스크에 맞춘 정보를 사용하여 잘못된 응답을 만들 때 스타일과 길이를 보존합니다. 이 방법은 기존의 VLM들보다 더 신뢰할 수 있는 문맥 이해 능력을 기반으로 하여, LVLMs의 비전-언어 정렬과 추론 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: VaPR 모델은 10개의 벤치마크에서 평균 6.5%(LLaVA), 4.0%(Qwen2VL), 1.5%(Qwen2.5VL)의 성능 향상을 이루었으며, 추론 작업에서 두드러진 결과를 보였습니다. LLaVA 모델은 적은 데이터로도 향상된 결과를 나타내며, VaPR은 이진 질문에서 '예'로 답할 경향을 줄이는 데도 효과적이었습니다. 또한, VaPR-OS 실험을 통해 오픈 소스 모델이 VaPR 프레임워크를 따르면서도 매우 유사한 성능을 내는 것을 확인했습니다.



### Improving AGI Evaluation: A Data Science Perspectiv (https://arxiv.org/abs/2510.01687)
- **What's New**: 이번 연구에서는 AGI(Artificial General Intelligence) 평가 방법론에 대하여 기존의 직관에 기반한 방식의 한계를 지적하고, 대안으로 강력한 과제 수행 능력을 평가하는 접근 방식을 제안합니다. AGI 평가에서의 주요 목표는 실제로 인간의 작업을 수행할 수 있는 시스템을 개발하기 위해 필요한 평가 메커니즘에 대한 새로운 시각을 제공하는 것입니다. 저자들은 데이터 과학의 실제적인 사례를 바탕으로 AGI 평가를 위한 강력한 방법론을 제시합니다.

- **Technical Details**: AGI의 평가에서는 정의의 모호성이 문제로 지적되며, 특히 기존의 성과 지표들은 인간의 지능을 우회하려는 방식으로 조작될 수 있는 경향이 있습니다. 이 연구는 AGI가 복잡하고 다양한 환경에서 자율적으로 목표를 달성하는 능력을 강조하며, 새로운 평가 방법론이 데이터 세트의 편향과 오염을 줄이고, 메모리 효과를 방지하는데 중점을 두어야 한다고 주장합니다. 이를 통해 AGI 시스템이 학습한 데이터 밖에서도 신뢰할 수 있는 성능을 보장할 수 있도록 해야 합니다.

- **Performance Highlights**: AGI 시스템의 성능은 기존의 정량적 지표 외에도, 새로운 접근 방식으로 제안된 시뮬레이션 환경에서의 성능으로 평가될 수 있습니다. 이러한 접근법은 시스템이 실제 인간의 학습과 유사한 방식으로 causal principles를 학습할 수 있는지를 중심으로 진행됩니다. AGI 평가의 중심에는 시스템이 실제 업무를 수행할 수 있는 능력과 자율성이 포함되어, 이러한 점에서 새로운 평가 방법이 강력한 검증 도구로 자리 잡을 수 있을 것으로 기대됩니다.



### A Locally Executable AI System for Improving Preoperative Patient Communication: A Multi-Domain Clinical Evaluation (https://arxiv.org/abs/2510.01671)
Comments:
          32 pages, 4 figures, 10 tables 32 pages, 4 figures, 10 tables. This paper is currently under review at ACM Transactions on Computing for Healthcare. Reproducibility resources: this http URL

- **What's New**: 이번 연구에서는 환자들이 절차 전 질문에 대한 답을 받을 수 있도록 돕는 LENOHA 시스템을 소개합니다. LENOHA는 높은 정확도의 sentence-transformer classifier를 통해 임상 질문에 대한 답변을 제공하며, 자유 텍스트 생성(free-text generation)을 생략하여 안전성을 높였습니다. 이는 시간의 제약과 프라이버시 문제를 극복하여 더욱 개인화된 상담을 가능하게 합니다.

- **Technical Details**: LENOHA 시스템은 두 가지 도메인(치아 추출 및 위내시경)에서 평가되었습니다. 총 800개의 데이터 세트를 사용한 독립적인 테스트 결과, E5-large-instruct(560M) 모델은 0.983의 전체 정확도와 0.996의 AUC를 기록했습니다. 이 모델은 GPT-4o와 비교했을 때 통계적으로 유의미한 차이가 없었습니다.

- **Performance Highlights**: 비생성 임상 경로(non-generative clinical path)는 입력당 약 1.0 mWh의 에너지를 소비했으며, 이는 로컬 8B SLM에서의 소통 응답에 대해 약 170배 차이가 납니다. 테스트 집합에서 Gemini는 오류를 전혀 발생시키지 않았고, 전체 시스템은 프라이버시와 지속 가능성을 유지하면서 대역폭이 제한된 환경에서도 공정한 배포를 지원합니다.



### Just Do It!? Computer-Use Agents Exhibit Blind Goal-Directedness (https://arxiv.org/abs/2510.01670)
- **What's New**: 이 논문은 컴퓨터 사용 에이전트(CUAs)의 Blind Goal-Directedness (BGD) 현상을 분석하며, 이는 에이전트가 목표 관점에서만 행동하여 잠재적으로 해로운 결과를 초래할 수 있음을 보여줍니다. BGD는 세 가지 주요 패턴으로 구분되며, 이를 체계적으로 평가하기 위해 90개의 작업으로 구성된 BLIND-ACT 벤치마크를 개발했습니다.

- **Technical Details**: BLIND-ACT는 OSWorld 위에 구축된 실제적이고 동적인 데스크톱 환경에서의 실행을 지원하며, 다양한 어플리케이션과 시스템 기능에서 BGD 행동이 발생할 수 있도록 설계되었습니다. 벤치마크는 90개의 작업으로 구성되어 있으며, 각 작업은 BGD의 세 가지 패턴을 포괄합니다. LLM 기반의 판단자를 사용해 에이전트가 BGD 행동을 보이는지와 비효율적인 행동을 실행하는지를 평가합니다.

- **Performance Highlights**: BLIND-ACT를 활용하여 Claude Sonnet, Opus 4 및 GPT-5와 같은 9개의 최신 모델을 평가한 결과, 평균 80.8%의 BGD 비율을 관찰했습니다. 작은 모델들은 과도하게 안전한 것처럼 보이나 이는 제한된 능력 때문이며, 결과적으로 안전과 능력 간의 패러독스가 강화됩니다. 또한, 프롬프트 기반의 개입이 BGD 수준을 낮출 수 있지만 여전히 상당한 위험이 남아 있음을 보여줍니다.



### GuruAgents: Emulating Wise Investors with Prompt-Guided LLM Agents (https://arxiv.org/abs/2510.01664)
Comments:
          7 Pages, 2 figures

- **What's New**: 이번 연구에서는 GuruAgents라는 프롬프트 기반 AI 에이전트를 통해 전설적인 투자 고수들의 전략을 체계적으로 실현할 수 있음을 보여줍니다. 각 GuruAgent는 고유의 투자 철학을 반영한 다섯 가지 모델로 개발되어, 주어진 금융 도구와 결정론적 사고 과정을 통합하여 작동합니다. NASDAQ-100 사례를 기반으로 백테스트를 진행한 결과, Buffett GuruAgent가 연평균 성장률 42.2%로 가장 뛰어난 성과를 보였습니다.

- **Technical Details**: 이 연구에서 선보인 GuruAgents 시스템은 역할 정의, 도구 통합 설계, 결정론적 사고 파이프라인의 세 가지 핵심 요소로 구성된 프롬프트 엔지니어링 프레임워크에 기반합니다. 각 GuruAgent는 특정 투자자의 고유 투자 철학을 내재화하며, 복잡한 재무 데이터를 해석하고 투자 전략을 실행하기 위해 다양한 금융 도구를 통합합니다. 이는 에이전트가 균일한 성능을 보장하기 위한 결정론적 사고 과정을 따르게 합니다.

- **Performance Highlights**: 실험 결과는 각 GuruAgents의 행동이 그들의 프롬프트에 의해 독특하게 구동된다는 것을 보여주며, 특히 Buffett GuruAgent가 제시된 벤치마크를 현저히 초과했습니다. 반면, 다른 에이전트들은 다소 다른 성과를 나타냈습니다. 이로써 프롬프트 엔지니어링이 투자 고수들의 정성적 철학을 정량적이고 재현 가능한 전략으로 성공적으로 전환할 수 있다는 점을 확인할 수 있었습니다.



### Understanding the Geospatial Reasoning Capabilities of LLMs: A Trajectory Recovery Perspectiv (https://arxiv.org/abs/2510.01639)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 도로 네트워크 지도 읽기와 내비게이션 수행 능력을 갖추고 있는지를 탐구합니다. 이를 위해 4,000개 이상의 실제 경로로 구성된 GLOBALTRACE 데이터셋을 소개하고, LLMs가 외부 내비게이션 도구 없이도 유효한 경로를 생성할 수 있는 프레임워크를 개발했습니다. 실험 결과, LLMs는 기존의 모델들보다 우수한 성능을 보여주며, 제로샷 제너럴라이제이션(zero-shot generalization) 능력이 뛰어난 것으로 나타났습니다.

- **Technical Details**: GLOBALTRACE 데이터셋은 다양한 지역과 교통 수단을 아우르는 4,000개 이상의 실제 경로를 포함하고 있습니다. 모델은 GPS 트레이스를 복원하기 위한 테스크로서, 도로 네트워크의 맥락을 바탕으로 마스킹된 GPS 트레스를 재구성합니다. 기존의 전통적인 트레일 회복 방법에 비해, LLMs는 추가적인 도메인 훈련 없이 도로 네트워크 데이터를 활용하여 재구성을 수행할 수 있습니다.

- **Performance Highlights**: LLMs는 기존의 전문 트레일 복구 모델과 비교하여 뛰어난 성능을 기록했습니다. 이 연구는 LLMs가 복잡한 도로 네트워크를 계획하고 기하학적 제약을 준수하며 현실적인 좌표를 생성하는 능력을 보여주었습니다. 따라서 LLMs는 사용자 선호를 반영하여 내비게이션 경험을 향상시킬 수 있으며, 전통적인 시스템을 넘어서는 새로운 내비게이션 가능성을 제시합니다.



### Learning to Decide with Just Enough: Information-Theoretic Context Summarization for CDMPs (https://arxiv.org/abs/2510.01620)
- **What's New**: 이 논문은 Contextual Markov Decision Processes (CMDPs)에서의 의사결정 과정 향상을 위한 정보이론적 요약 방법을 제안합니다. 기존 방법들이 고차원이나 비구조적 맥락에서 일반화에 실패하는 문제를 해결하기 위해, 대규모 언어 모델(LLM)을 활용하여 맥락 입력을 저차원으로 압축하고 의사결정에 중요한 단서를 보존합니다. 연구는 맥락의 정보성과 계산 비용 간의 관계를 명확히 하며, CMDPs에 대한 최초의 후회를 보장하는 경계와 대기시간-엔트로피 거래 특성을 제공합니다.

- **Technical Details**: 새로운 CMDP 프레임워크를 통해 LLM 기반 요약이 맥락 표현을 저차원으로 생성하게 하여, 전체 의사결정 과정을 더욱 효율적으로 만듭니다. 정보이론적 관점에서, 우리는 맥락의 정보 가치와 표현의 계산 비용 간의 무역을 분석하여 명확한 이론적 기초를 제공합니다. 이러한 접근은 필요한 최소한의 맥락으로 효과적인 학습을 위해 필요한 조건을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존의 비맥락 및 원시 맥락 기준선에 비해 보상, 성공률 및 샘플 효율성을 증가시키고, 대기시간 및 메모리 사용량을 줄이는 성과를 나타냈습니다. 다양한 CMDP 벤치마크를 통한 평가에서 요약 기반 에이전트가 일관되게 우수한 의사결정 품질과 계산적 스케일 능력을 보이는 것을 확인했습니다. 이 연구는 복잡한 환경에서 효율적인 의사결정을 위한 LLM 기반 요약의 가능성을 보여줍니다.



### PychoBench: Evaluating the Psychology Intelligence of Large Language Models (https://arxiv.org/abs/2510.01611)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)이 심리 상담에서 효과적으로 활용될 수 있는지를 분석합니다. 연구팀은 PsychoBench라는 벤치마크를 도입하여 LLM이 미국 국가 상담사 인증 시험(NCE)을 통과할 수 있는지를 평가합니다. PsychoBench는 2,252개의 심리학 관련 단일 선택 질문으로 구성되어 있으며, 이는 LLMS의 상담사 역할 수행 가능성을 검증하는 데 도움을 줍니다.

- **Technical Details**: PsychoBench는 GPT 기반 방법을 이용하여 심리 상담 시험 질문을 재구성하고 정제하였습니다. 각 질문은 심리학 전문가들에 의해 정확성과 일관성을 보장하기 위해 검토되었습니다. 이 데이터셋은 상담 방법, 이상 심리학, 발달 심리학 및 윤리적 고려 사항을 포함한 다양한 하위 분야를 포괄하며, LLM의 심리 상담 작업 수행 능력을 체계적으로 평가하기 위해 사용됩니다.

- **Performance Highlights**: 최신 LLM 모델인 GPT-4o, Llama3.3-70B, 그리고 Gemma3-27B가 시험 기준을 초과하는 성과를 보였으나, 소형 오픈 소스 모델들은 여전히 기준에 미치지 못함을 보여줍니다. 이러한 결과는 현재 심리 상담 표준을 충족할 수 있는 것은 최첨단 LLM만임을 시사하며, 심리학 중심의 LLM 개발에서의 약속과 도전 과제를 강조합니다.



### AgentRec: Next-Generation LLM-Powered Multi-Agent Collaborative Recommendation with Adaptive Intelligenc (https://arxiv.org/abs/2510.01609)
- **What's New**: 본 논문은 차세대 LLM(대형 언어 모델) 기반의 다중 에이전트 협업 추천 시스템인 AgentRec를 소개합니다. 이 프레임워크는 계층적 에이전트 네트워크와 적응형 지능을 통해 기존의 대화형 추천 시스템이 직면한 한계를 극복합니다. AgentRec의 주요 기여는 대화 이해, 선호 모델링, 문맥 인식, 동적 랭킹을 위한 전문화된 LLM 에이전트를 포함하는 것입니다.

- **Technical Details**: AgentRec은 네 가지 전문화된 LLM 에이전트로 구성된 계층적 다중 에이전트 아키텍처를 채택합니다. 각 에이전트는 대화 분석, 사용자 선호 모델링, 문맥 인식, 실시간 랭킹 등 특정 작업에 특화되어 있습니다. 시스템은 대화 상태에 따라 에이전트의 가중치를 적응적으로 조정하는 메타 학습 방식을 활용합니다.

- **Performance Highlights**: 광범위한 실제 데이터 세트를 바탕으로 진행된 실험에서, AgentRec은 모든 데이터 세트와 지표에서 일관된 개선을 보여주었습니다. 이를 통해 대화 성공률이 2.8%, 추천 정확도가 1.9%, 그리고 대화 효율성이 3.2% 향상되었습니다. 이러한 결과는 AgentRec이 Chat-REC을 넘어서는 차세대 시스템으로 자리매김할 수 있도록 해줍니다.



### AdvEvo-MARL: Shaping Internalized Safety through Adversarial Co-Evolution in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2510.01586)
- **What's New**: 이 논문은 AdvEvo-MARL이라는 새로운 다중 에이전트 강화 학습(MARL) 프레임워크를 소개합니다. 이 프레임워크는 안전성 인식을 각 작업 에이전트에 내재화하여 공격자와 방어자가 공동으로 진화하도록 최적화합니다. AdvEvo-MARL은 외부 감시Agent에 의존하지 않고도 시스템의 안전성과 유용성을 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: AdvEvo-MARL은 좀 더 정교한 jailbreak 프롬프트를 생성하는 공격자와 이 공격을 저지하고 동시에 업무를 수행하는 방어자를 공동으로 훈련시키는 방식으로 작동합니다. 에이전트는 각자 기능 그룹 내에서 집단 평균 보상을 공유하여 정책 업데이트의 변동성을 줄이고 협력을 강화합니다. 다양한 공격 시나리오를 통한 실험을 통해 AdvEvo-MARL은 공격 성공률을 20% 이하로 유지하며, 작업 정확도를 개선하는 성과를 보여주었습니다.

- **Performance Highlights**: AdvEvo-MARL은 세 가지 대표적인 공격 시나리오에서 시스템의 강건성을 향상시키는 효과를 입증했습니다. 실험 결과, 공격 성공률이 감소하고 특정 작업에서 최대 3.67% 향상이 관찰되었습니다. 이로 인해 AdvEvo-MARL은 안전성과 작업 성능을 동시에 증진할 수 있는 가능성을 드러내며, 다중 에이전트 시스템의 안전성을 강화하는 표준화된 프레임워크로 자리 잡을 수 있을 것으로 예상됩니다.



### InvThink: Towards AI Safety via Inverse Reasoning (https://arxiv.org/abs/2510.01569)
- **What's New**: InvThink는 대형 언어 모델(LLMs)에게 실패 모드를 미리 고려하는 역 추론(inverse reasoning) 능력을 부여하는 새로운 접근법입니다. 기존의 안전 정렬 방법이 안전한 응답을 직접 최적화하는 것과 달리, InvThink는 모델이 잠재적 해를 나열하고 그 결과를 분석한 후 안전하게 응답을 생성하도록 합니다. 이를 통해 안전성 향상이 모델 크기와 비례하여 강화되는 것을 보여줍니다.

- **Technical Details**: InvThink의 핵심은 모델이 해를 수치화하고 그에 대한 분석을 수행한 후, 피해를 피하는 방향으로 응답을 생성하도록 하는 것입니다. 이 과정은 역 추론 프레임워크로서 구조화된 추론 과정을 통해 이루어지며, 기존의 단순한 출력 맵핑 방식을 넘어서 모델의 철저한 위험 분석을 가능케 합니다. 또한, 기존의 방법과 달리 사고를 포함한 직관적 접근 방식을 통해 구체적이고도 포괄적인 위험 예측을 실현합니다.

- **Performance Highlights**: InvThink는 의료, 금융, 법률 등 고위험 분야에서 특히 뛰어난 성능을 보이며, 기존 방법에 비해 최대 15.7%의 유해 응답 감소를 달성했습니다. 또한, InvThink는 일반 추론 능력을 보존하면서도 안전성을 개선하여 안전 세금(safety tax) 문제를 완화합니다. 이는 InvThink가 AI 안전성을 향상시킬 수 있는 확장 가능하고 일반화된 경로를 제시한다는 것을 의미합니다.



### Step-Aware Policy Optimization for Reasoning in Diffusion Large Language Models (https://arxiv.org/abs/2510.01544)
- **What's New**: 이 논문은 Diffusion language models (dLLMs)가 기존의 autoregressive 모델에 비해 훈련의 효율성을 높일 수 있는 가능성을 제시하고 있습니다. 특히, 복잡한 추론 문제를 해결하기 위한 기존 방법들의 한계를 지적하며, dLLMs의 훈련 과정에서 계층 구조의 중요성을 강조합니다. 이러한 이론적 토대 위에, 연구진은 Step-Aware Policy Optimization (SAPO)이라는 새로운 강화학습 알고리즘을 제안하여, 모델의 추론 과정을 더욱 명확하게 합니다.

- **Technical Details**: 제안된 이론적 프레임워크는 복잡한 문제 해결을 계층적 선택 과정으로 정형화하며, 이를 통해 dLLMs의 훈련을 위한 전략적 기초를 제공합니다. 기존의 강화학습 방법은 보통 결과 기반 보상에 의존하며, 이는 잘못된 추론 경로를 강화할 수 있습니다. 연구진은 unstructured refinement를 주요 문제로 지적하며, SAPO를 통해 지식의 구조를 학습할 수 있게 돕는 과정 기반 보상 함수를 사용합니다.

- **Performance Highlights**: SAPO는 dLLMs의 성능을 조정하여 보다 구조적이고 일관된 추론 경로를 학습하도록 유도합니다. 연구 결과, SAPO를 적용했을 때 복잡한 추론 기준에서 성효율이 현저히 향상되었으며, 생성 과정의 해석 가능성도 증가했습니다. 이러한 개선은 dLLMs의 훈련에 있어서 이론적 근거를 제공하며, 논문에서 제안하는 방식이 실질적으로 적용 가능함을 보여줍니다.



### Information Seeking for Robust Decision Making under Partial Observability (https://arxiv.org/abs/2510.01531)
Comments:
          The project page is available at this https URL

- **What's New**: 이번 연구에서는 정보 탐색(Information Seeking)을 LLM(대형 언어 모델) 의사결정 프레임워크인 InfoSeeker에 통합하여, 불확실한 환경에서의 내부 동향과 실제 환경을 조화롭게 맞추는 방법을 제안합니다. InfoSeeker는 의도가 있는 계획 수립을 통해 정보 수집을 유도하며, 이는 기존의 관측 불확실성을 다루는 모델들이 간과했던 부분입니다. 또한, InfoSeeker는 실제 환경 변화 감지 및 가설 테스트를 통해 효과적인 계획을 만들어냅니다.

- **Technical Details**: InfoSeeker는 부분적으로 관찰 가능한 환경에서 결정-making을 모사하고 따르는 POMDP(부분 관찰 마르코프 결정 과정) 모델을 기반으로 합니다. 모델은 상태 집합, 행동 집합, 관측 균형 등을 포함하여 에이전트가 확률적 결정을 내리는 과정에서 활용됩니다. 이 연구에서는 고유의 리워드 함수와 할인지수 등을 통해 정보 수집을 포함한 의사결정 프레임워크를 수학적으로 정립합니다.

- **Performance Highlights**: 정보 탐색을 적극적으로 통합한 InfoSeeker는 기존 방법들보다 74%의 성능 향상을 달성하였으며, 샘플 효율성을 해치지 않으면서도 더 나은 계획을 생성할 수 있게 되었습니다. 게다가 InfoSeeker는 다양한 LLM에 대해 일반화 가능하며, 기존 벤치마크에서도 우수한 성능을 보이는 것으로 나타났습니다. 이로 인해 두 가지 핵심 기여 - 정보 탐색과 계획 통합 -를 통한 에이전트의 강건한 행동을 강조할 수 있게 되었습니다.



### LOGicalThought: Logic-Based Ontological Grounding of LLMs for High-Assurance Reasoning (https://arxiv.org/abs/2510.01530)
- **What's New**: 이 논문에서는 LOGicalThought (LogT)라는 새로운 신경 상징적(neurosymbolic) 아키텍처를 제안한다. 이 아키텍처는 대형 언어 모델(LLM)과 함께 고급 논리 언어 및 추론기를 사용하여 기호 그래프(context)와 논리 기반(context)을 구성한다. 이러한 이중 상징적 맥락을 통해 자연어 문제가 정밀한 평가로 변환되어 높은 신뢰성(high-assurance) 텍스트 지침을 보다 효과적으로 처리할 수 있다.

- **Technical Details**: LogT는 자연어 추론 과제를 명확히 정의하고 비가역적(non-monotonic) 논리 구성 요소를 명시적으로 표현하는 논리 프로그램 매핑을 통해 해결한다. 이 시스템은 고차원 관계를 포착하는 상징 그래프(representational graph)와 비가역적 규칙 및 사실을 형식화한 기계가독성 논리 프로그램(machinereadable logic program)으로 구성된다. 또한, LLM을 통해 가이드라인에서 필요한 정보만 선택하고, 규칙 ontology 및 지식 삼중 데이터 구조를 생성하여 독립적으로 추론을 지원한다.

- **Performance Highlights**: LogT는 네 가지 다양한 벤치마크에서 평가되었으며, 모든 LLM에 대해 전체 성능을 평균 11.84% 향상시키는 성과를 보였다. 특히, 부정(negation) 추론에서 +10.2%, 함의(implication)에서 +13.2%, 비가역적 추론(defeasible reasoning)에서는 +5.5%의 성과를 달성하여 기존 최고 기준선보다 과도한 성능 개선을 나타내었다.



### Towards Interpretable and Inference-Optimal COT Reasoning with Sparse Autoencoder-Guided Generation (https://arxiv.org/abs/2510.01528)
- **What's New**: 이번 연구에서는 희소 자동인코더(SAE)와 클러스터링 기법을 활용해 대형 언어 모델(LLMs)의 내부 토큰 표현을 분석하고 수학적 추론 과제를 위한 생성을 유도하는 새로운 방법을 제안합니다. SAE를 이용해 훈련 토큰의 희소 벡터 표현을 생성한 후, k-평균 클러스터링을 적용하여 토큰 클러스터를 나타내는 그래프를 구성합니다. 이 그래프를 통해 기존의 추론 경로에 대한 보상을 정량화하고, 탐색 정도를 측정하는 기법을 개발했습니다.

- **Technical Details**: SAE 훈련 과정에서 입력 토큰 표현을 고차원 잠재 공간으로 매핑하며, reconstruction loss를 최소화하는 것이 목표입니다. 연결된 단계에서 k-평균 클러스터링을 통해 클러스터 다양성을 측정하고, 이는 탐험의 정도를 평가하는데 유용합니다. 이 과정에서 도출된 보상 함수는 생성 과정의 중간 단계에서 적절한 보상을 할당하는 데 기여하며, 이는 생성의 견고성을 높입니다.

- **Performance Highlights**: 연구 결과, 탐색과 착취의 균형을 맞추는 것이 수학적 추론 작업에서 높은 정확도를 달성하는 데 중요한 요소로 확인되었습니다. 또한, SAE를 통한 고급 생성 유도 방식이 모델의 생성 품질을 향상시키는 데 효과적임을 입증했습니다. 이 연구는 SAE의 풍부한 표현을 활용하여 새로운 자동화된 보상 모델을 개발함으로써 LLM의 효율성을 높일 수 있는 가능성을 보여줍니다.



### Lateral Tree-of-Thoughts Surpasses ToT by Incorporating Logically-Consistent, Low-Utility Candidates (https://arxiv.org/abs/2510.01500)
- **What's New**: 이 논문에서는 Lateral Tree-of-Thoughts (LToT)라는 새로운 제어기(controller)를 제안합니다. 기존의 Tree-of-Thoughts 방식이 가진 두 가지 병목현상인 breadth saturation(폭 포화)과 depth myopia(깊이 단기적 사고)에 대한 해결책으로, 일관성이 있지만 낮은 유틸리티를 가진 후보들을 자산으로 보고 활용하는 접근 방식을 도입합니다. LToT는 탐색을 Lateral Racing with Short-Circuit (LR–SC) 방식을 통해 보다 폭넓고 효율적으로 수행할 수 있도록 합니다.

- **Technical Details**: LToT는 논리적 일관성과 유틸리티를 분리하여 설계되었으며, 초기의 낮은 유틸리티 후보들을 저비용의 짧은 예측 이어받음으로 탐색할 수 있도록 합니다. 주요 후보들은 exploitation을 위해 높은 유틸리티를 바탕으로 유지되며, Lateral Racing을 통해 수행하는 동안 넓은 폭의 후보군을 탐색합니다. 이를 통해 LToT는 pseudolinear lateral cost A8(N_0 log_{eta} N_0)으로 지출을 조절하고, 비효율적인 에너지 소모를 방지합니다.

- **Performance Highlights**: LToT의 성능은 여러 벤치마크 과제를 통해 향상된 Success@1/Pass@1을 보여줍니다. 기존의 Chain of Thought(CoT) 및 일반적인 Tree-of-Thoughts보다 강력한 성능을 나타내며, 특히 낮은 false promotions (허위 승인 비율) 및 짧은 타임-투-퍼스트-히트(time-to-first-hit)에도 도움을 줍니다. LToT는 다양한 테스트와 실험을 통해 그 실효성을 입증할 예정이며, 이를 통해 탐색과 확장 측면에서 유의미한 개선을 제공합니다.



### AIReg-Bench: Benchmarking Language Models That Assess AI Regulation Complianc (https://arxiv.org/abs/2510.01474)
- **What's New**: AIReg-Bench는 유럽연합의 AI 법률(AI Act, AIA) 준수를 평가하기 위해 대형 언어 모델(Large Language Models, LLMs)의 성능을 벤치마킹하는 최초의 데이터셋입니다. 이 데이터셋은 120개의 기술 문서 발췌로 구성되어 있으며, 각 발췌는 허구의 AI 시스템을 묘사하고 있습니다. 저자들은 LLM을 사용하여 이 발췌를 생성하고, 법률 전문가들이 각 샘플을 검토하여 AIA의 특정 조항을 위반하는지를 표시하였습니다.

- **Technical Details**: AIReg-Bench의 생성은 두 단계로 나누어집니다. 첫 번째로, LLM을 통해 기술 문서 발췌를 생성하고, 두 번째로, 법률 전문가들이 각 발췌를 검토하여 위반 사항을 주석으로 달았습니다. 또한, 이 데이터셋은 gpt-4.1-mini를 중심으로 한 다단계 샘플 생성 파이프라인을 구축하여 이루어졌으며, 다양한 AI 시스템을 반영하기 위해 실제 기술 문서와 유사한 현실감과 다양성을 지닌 발췌를 생성하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 여러 최첨단 LLM들이 인간 전문가의 평가에 매우 근접한 성과를 보였습니다. 예를 들어, Gemini 2.5 Pro는 0.856의 순위 상관관계를 달성하여 AIA 준수 여부를 잘 추정하는 것으로 나타났습니다. AIReg-Bench는 LLM의 성능을 평가하고 AIR 준수 평가 도구의 가능성과 한계를 이해하는 데 중요한 시작점을 제공합니다.



### VOGUE: Guiding Exploration with Visual Uncertainty Improves Multimodal Reasoning (https://arxiv.org/abs/2510.01444)
- **What's New**: 이 논문에서는 Visual-Uncertainty-Guided Exploration (VOGUE)이라는 새로운 방법을 소개하며, 이 방법은 멀티모달 대형 언어 모델(MLLM)의 탐색 문제를 해결하는 데 초점을 맞추고 있습니다. 전통적인 방법들이 이미지를 고정된 조건으로 취급하는 대신, VOGUE는 이미지를 확률적 맥락으로 간주하고 시각적 변동성을 측정함으로써 더 나은 탐색 방향을 제시합니다. 이 방법을 통해 시각적 불확실성을 기반으로 한 학습 목표의 재조정이 가능해 여태까지 간과되었던 비밀스러운 탐색 경로를 유도합니다.

- **Technical Details**: VOGUE는 훈련 시 두 가지 가지(branch)를 활용하는 이중 가지 전방 패스를 통해 시각적 불확실성을 정량화합니다. 원본 이미지와 의미가 보존된 퍼터베이션(perturbation)된 이미지를 모두 사용하여 정책의 민감도를 계산하고, 비슷한 방식으로 KL 발산(KL divergence)을 적용하여 탐색 우위를 조형합니다. 이러한 시각적 불확실성은 불확실성 비례 보너스로 연결되어, 초기 훈련에서는 탐색에 중점을 두고 훈련이 안정화되면 원본 이미지로 초점을 옮기는 방식으로 진행됩니다.

- **Performance Highlights**: VOGUE는 GRPO 구현 하에 Qwen2.5-VL-3B 및 7B 모델에서 세 개의 시각 수학 벤치마크와 세 개의 일반 도메인 추론 벤치마크에서 각각 평균 2.6% 및 3.7%의 pass@1 정확도를 증가시키며, 일반적으로 발견되는 RL 미세 조정에서의 탐색 감소를 효과적으로 완화하였습니다. 또한, VOGUE는 텍스트 전용 설정에서 효과를 보이는 Pass@k Training 방법보다 지속적으로 우수한 성능을 발휘하여, 높은 pass@1 및 일관된 pass@k 향상을 달성했습니다.



### On the Role of Domain Experts in Creating Effective Tutoring Systems (https://arxiv.org/abs/2510.01432)
Comments:
          Accepted to AIED 2025 Blue Sky Track

- **What's New**: 이 논문에서는 교육 AI 분야에서 전문가의 고도로 선별된 지식의 중요성을 재조명하고 있습니다. 특히 설명 가능한 AI (XAI) 기법을 활용하여 수업을 자동 생성하는 방법과, 전문가가 지정한 커리큘럼이 적응형 튜터링 시스템의 개발에 어떻게 기여할 수 있는지를 논의합니다. 이러한 전문 지식을 활용함으로써 교육 시스템의 효과성을 높일 수 있는 여러 접근 방식을 소개하고 있습니다.

- **Technical Details**: 논문에서는 설명 가능한 AI(XAI) 기법을 통해 전문 규칙을 자동 생성된 수업에 통합하는 방안을 제시합니다. 예를 들어, POMDP(부분 관찰 가능 마르코프 의사결정 프로세스) 기반의 적응형 튜터링 시스템을 활용하여 학습자의 지식 수준에 맞는 수업을 제공하는 방법에 대해 설명하고 있습니다. 또한, 전문가가 규명한 커리큘럼을 통해 교육 설정에서의 불확실성을 효과적으로 다룰 수 있는 방안을 논의합니다.

- **Performance Highlights**: 이 연구에서 제안하는 방법은 특히 에너지와 자원이 부족한 시민 과학 단체의 튜터링 시스템에 큰 도움이 될 수 있습니다. 이러한 시스템의 중요성을 보여주는 사례로는 꽃가루 매개체(pollinator) 식별을 위한 교육 시스템이 있습니다. 전문가의 지식을 활용한 자동화된 교육 시스템은 효과적인 훈련을 제공하고, 더 많은 참여자를 모집할 수 있는 기회를 제공합니다.



### A Tale of LLMs and Induced Small Proxies: Scalable Agents for Knowledge Mining (https://arxiv.org/abs/2510.01427)
- **What's New**: Falconer는 대규모 비구조적 텍스트에서 구조화된 정보를 추출하는 지식 마이닝(konwledge mining) 작업을 위해 대형 언어 모델(LLM)과 경량 프록시 모델을 결합한 협업 프레임워크를 소개합니다. 이 프레임워크는 LLM을 플래너(planner)로 활용하여 사용자의 명령을 실행 가능한 파이프라인으로 분해하며, 주석을 생성하여 소형 모델 훈련을 지원합니다. Falconer는 분류(classification)와 추출(extraction) 작업을 통합하여 여러 작업 특화 요소를 대체할 수 있는 단일 모델로 전환합니다.

- **Technical Details**: Falconer의 핵심은 두 가지 원자적 작업인 get_label과 get_span으로, 각각 텍스트와 지침에 대해 분류 및 범위 추출을 수행합니다. 이러한 원자는 모든 지식 마이닝 파이프라인의 기본 빌딩 블록으로 작용합니다. Falconer는 별도의 모델을 요구하지 않으며, 두 가지 기본 작업만으로도 복잡한 태스크를 처리할 수 있도록 설계되었습니다. 이 통합된 접근 방식은 비용을 절감하고 효율성을 높입니다.

- **Performance Highlights**: Falconer는 LLM과 유사한 수준의 명령 따르기 정확도를 유지하면서 인퍼런스 비용을 최대 90%까지 절감합니다. 또한 대규모 지식 마이닝 속도는 20배 이상 향상되어 효율적이고 확장 가능한 솔루션으로 자리매김합니다. 새로운 지식 마이닝 벤치마크를 개발하여 Falconer의 계획 능력과 전반적인 성능을 평가하였으며, 실험 결과 이는 최신 상태의 LLM과 거의 일치하는 출력을 제공합니다.



### OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models (https://arxiv.org/abs/2510.01409)
Comments:
          20 pages, 6 tables, 7 figures

- **What's New**: OntoLogX를 소개하며, 이 자율 인공지능(AI) 에이전트는 원시 로그(raw logs)를 사용하여 실질적인 Cyber Threat Intelligence (CTI)로 변환하는 기능을 갖추고 있습니다. 이 시스템은 Large Language Models (LLMs)를 활용하여 로그 정보를 온톨로지 기반의 지식 그래프(KGs)로 전환하여, 더욱 명확한 의도를 제공합니다. 특히, MITRE ATT&CK 전술 예측을 지원하며, 로그 증거들을 상위 수준의 적대적 목표와 연결합니다.

- **Technical Details**: OntoLogX는 경량 로그 온톨로지(lite log ontology)와 Retrieval Augmented Generation (RAG), 반복 수정 단계를 결합하여 KGs를 생성합니다. 이 과정은 합성적이며 의미론적으로 유효한 그래프를 보장합니다. 또한, 생성된 KGs는 사이버 보안 로그를 위해 특별히 설계된 도메인 특정 온톨로지에 맞춰져 있어, 의미적 쿼리와 트레이스 가능성을 제공합니다.

- **Performance Highlights**: OntoLogX는 공공 벤치마크와 실세계 형벌 데이터셋을 기반으로 한 로그에서 강력한 KGs 생성을 입증하였습니다. 특히, 검색과 수정 기능의 효과는 정밀도(precision) 및 재현율(recall) 측면에서 뛰어나며, 코드 지향 모델들을 활용한 구조적 로그 분석의 효과성을 보여줍니다. 결과적으로, 온톨로지 기반 표현 방식이 실질적인 CTI 추출에 얼마나 중요한지를 강조합니다.



### Automating Data-Driven Modeling and Analysis for Engineering Applications using Large Language Model Agents (https://arxiv.org/abs/2510.01398)
- **What's New**: 현대 엔지니어링에서는 실험 및 시뮬레이션을 통해 생성된 방대한 데이터셋에 의존하게 되며, 이는 효율적이고 신뢰할 수 있으며 널리 적용 가능한 모델링 전략에 대한 수요를 증가시킵니다. 본 연구에서는 데이터 기반 모델링 및 분석을 자동화하기 위한 혁신적인 파이프라인을 제안하며, 특히 회귀 작업에 중점을 두고 있습니다. 제안된 접근법은 LLM(대형 언어 모델) 에이전트를 활용하여 데이터를 자동으로 처리하고, 신경망 개발, 교육, 하이퍼파라미터 최적화, 불확실성 정량화를 수행합니다.

- **Technical Details**: 이번 연구에서는 LLM-Agent 프레임워크 두 가지를 평가합니다: 전문화 된 협력 에이전트를 포함한 다중 에이전트 시스템과 ReAct(Reasoning and Acting) 패러다임에 기반한 단일 에이전트 시스템입니다. 두 프레임워크 모두 실험 데이터 약 25,000 점을 활용하여 기존의 기준을 초월하는 예측 정확도를 달성했습니다. LLM 기반 에이전트는 이 과정에서 복잡한 엔지니어링 모델링 작업을 자동화하는 데 필요한 잠재력을 지니고 있습니다.

- **Performance Highlights**: 결과는 LLM-Agent가 개발한 모델이 기존의 CHF 룩업 테이블을 초월하며, 최첨단 Bayesian 최적화 심층 신경망 모델의 예측 정확도 및 불확실성 정량화와 동등한 성능을 보여줍니다. 이러한 성과는 LLM 기반 에이전트가 복잡한 엔지니어링 모델링 작업을 자동화하고 인적 작업을 크게 줄이며, 예측 성능 기준을 만족하거나 초과할 수 있는 가능성을 강조합니다.



### Fine-tuning with RAG for Improving LLM Learning of New Skills (https://arxiv.org/abs/2510.01375)
Comments:
          Under review at ICLR 2026

- **What's New**: 이 논문에서는 멀티 스텝 작업을 위해 배치된 대규모 언어 모델(LLM) 에이전트의 예측 가능한 실패를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 retrieval-augmented generation (RAG) 방법이 아닌, 추출된 힌트를 사용하여 에이전트의 실수를 이용한 학습 과정을 통해 성능을 향상시키는 파이프라인을 구축했습니다. 이 방식은 외부 데이터베이스를 유지할 필요가 없으며, 컴퓨팅 오버헤드를 줄이는 데 도움이 됩니다.

- **Technical Details**: 제안된 방법은 (1) 에이전트의 실패로부터 재사용 가능한 간결한 힌트를 추출하고, (2) 이러한 힌트를 사용하여 에피소드 시작 시 향상된 teacher trajectories를 생성하며, (3) 힌트 문자열을 제거한 상태에서 학생 모델을 학습시킵니다. 이 과정은 기억이 아닌 내재화(internalization)를 강요하여 수행됩니다. 실험은 두 개의 상호작용 벤치마크인 ALFWorld와 WebShop에서 진행되었습니다.

- **Performance Highlights**: ALFWorld에서 증류된 학생 모델은 기준선 에이전트보다 일관되게 우수한 성능을 보이며, 최대 91%의 성공률을 기록했습니다(기준선: 79%). WebShop에서는 72점으로 기준선인 61점에 비해 개선되었습니다. 사용된 토큰 수는 환경에 따라 retrieval-augmented teacher보다 10-60% 감소하였으며, 이 접근 방식은 다양한 모델 스케일(7B/14B 파라미터) 및 에이전트 아키텍처(ReAct/StateAct)에 걸쳐 일반화될 수 있습니다.



### Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effor (https://arxiv.org/abs/2510.01367)
- **What's New**: 이번 논문에서는 보상 해킹(reward hacking)이라는 문제를 다룹니다. 이는 모델이 보상 함수의 허점을 이용하여 의도된 작업을 해결하지 않고도 높은 보상을 얻는 경우로, 명시적이거나 암묵적으로 나타날 수 있습니다. 이 문제를 해결하기 위해 TRACE (Truncated Reasoning AUC Evaluation)이라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: TRACE는 모델의 Chain-of-Thought (CoT)에서 보상 해킹을 감지하기 위해 ‘노력’effort의 정도를 측정하여 이를 정량화합니다. 연구에서는 다양한 길이에서 CoT를 잘라내고 모델이 검증기를 통과할 수 있는 비율을 측정하여 해킹 모델의 특성을 분석합니다. 더 짧은 CoT로도 높은 통과율을 얻는 모델은 해킹으로 간주됩니다.

- **Performance Highlights**: TRACE의 성능은 수학적 추론에서 기존 72B CoT 모니터에 비해 65% 이상의 향상을 보였으며 32B 모니터의 코드 작성 부분에서도 30% 이상의 개선을 기록했습니다. 또한 TRACE는 훈련 중 알려지지 않은 허점을 발견할 수 있는 능력을 보여줍니다. 이러한 결과는 현재의 모니터링 방법으로는 효과적인 감시가 어려운 상황에서 TRACE가 확장 가능한 비지도 접근 방식을 제공한다는 것을 의미합니다.



### Retrieval-Augmented Framework for LLM-Based Clinical Decision Suppor (https://arxiv.org/abs/2510.01363)
- **What's New**: 이 논문에서는 전자 건강 기록(EHR)의 복잡성과 급속한 확장에 대응하기 위해 대형 언어 모델(LLM)에 기반한 임상 의사결정 지원 시스템을 제안합니다. 이 시스템은 환자의 인구 통계, 증상, 진단 정보 및 치료 이력을 분석하여 치료 제안을 생성합니다. 임상 판단을 대체하는 것이 아니라, 유사한 특성을 지닌 이전 사례를 검색하고 종합함으로써 의사결정을 보완하도록 설계되었습니다.

- **Technical Details**: 이 시스템은 자연어 처리(NLP)와 구조화된 임상 입력을 통합하여 맥락에 적합한 추천을 생성합니다. 핵심 처리 메커니즘은 검색 보강 생성(RAG) 파이프라인으로, 비구조적 내러티브와 코드화된 데이터를 조화롭게 사용하여 LLM 기반 추론을 지원합니다. 기술 구성 요소로는 표현 차원 정렬(representation alignment) 및 생성 전략이 포함됩니다.

- **Performance Highlights**: 초기 평가에서는 비식별화된 징후 데이터셋을 사용하여 모델의 출력의 임상적 일관성과 그럴듯함을 검토했습니다. 초기 결과는 LLM 기반 도구가 적절히 제약되고 엄격히 검증될 경우 처방 워크플로우에서 유용한 의사결정 지원을 제공할 수 있음을 시사합니다. 본 연구는 생성 AI를 실제 임상 의사결정에 통합하기 위한 첫 걸음을 내딛는데, 투명성, 안전성 및 기존 관행과의 정렬에 초점을 맞추고 있습니다.



### MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments (https://arxiv.org/abs/2510.01353)
Comments:
          Accepted to NeurIPS 2025 SEA Workshop

- **What's New**: 최근의 연구들은 대화형 사례에 중점을 두어 왔지만, 동적인 기업 환경에서 메모리 평가의 필요성이 커지고 있습니다. 이 논문에서는 MEMTRACK이라는 새로운 벤치를 소개하며, 이는 다중 플랫폼 에이전트 환경에서 장기 메모리 및 상태 추적을 평가하기 위해 설계되었습니다. MEMTRACK은 Slack, Linear, Git과 같은 여러 커뮤니케이션 및 생산성 플랫폼에서 비동기 이벤트를 통합하여 현실적인 조직 워크플로우를 모델링합니다.

- **Technical Details**: 각 벤치마크 인스턴스는 연대적으로 플랫폼 간의 엮인 타임라인을 제공하며, 소음이 많고 상충하는 정보, 코드베이스/파일 시스템 이해 및 탐색 가능성을 포함합니다. 따라서 이 벤치마크는 메모리의 획득(acquisition), 선택(selection), 갈등 해결(conflict resolution) 능력을 테스트합니다. MEMTRACK 데이터세트는 수동 전문가 설계와 확장 가능한 에이전트 기반 합성을 통해 생성하여, 실제 소프트웨어 개발 프로세스에 기반한 생태학적으로 유효한 시나리오를 제공합니다.

- **Performance Highlights**: 정확성(Correctness), 효율성(Efficiency), 중복성(Redundancy)을 포착하는 관련 메트릭을 도입하여 메모리 메커니즘의 효과성을 비교합니다. SoTA LLMs 및 메모리 백엔드에 대한 실험 결과, 장기적인 메모리 활용, 플랫폼 간 의존성 처리 및 모순 해결에 관한 도전에 직면하게 됩니다. 특히, 가장 성능이 뛰어난 GPT-5 모델조차 MEMTRACK에서 60%의 정확성 점수만을 기록하는 것으로 나타났습니다.



### Aristotle: IMO-level Automated Theorem Proving (https://arxiv.org/abs/2510.01346)
- **What's New**: 이번 논문에서는 Aristotle이라는 AI 시스템을 소개하며, 이는 형식적 검증(formal verification)과 비공식적 추론(informal reasoning)을 결합하여 2025 국제수학올림피아드(IMO) 문제에서 금메달에 해당하는 성과를 달성했다. Aristotle은 세 가지 주요 구성 요소로 구성되어 있으며, 이는 Lean 증명 탐색 시스템, 비공식적인 추론 시스템, 그리고 전용 기하학 해결기이다. 이 시스템은 자동 정리 증명(automated theorem proving)에서 최첨단 성능을 보여준다.

- **Technical Details**: Aristotle의 기초가 되는 구성 요소는 Lean 코드를 처리하는 증명 탐색 알고리즘이다. 이 알고리즘은 Monte Carlo Tree Search (MCTS)를 기반으로 하며, 학습된 가치 함수(learned value function)를 사용한다. 이를 통해 Lean 증명 스케치에서 증명이 이루어지지 않은 목표를 모두 증명하기 위해 다양한 기법을 채택하고, 기하학 문제는 AlphaGeometry에 기반하여 따로 해결한다.

- **Performance Highlights**: Aristotle은 2025 IMO 문제에서 다섯 개의 문제에 대해 정형화된 솔루션(formal solution)을 제공하여 금메달 수준의 성과를 달성하였다. 이 시스템은 시가형 탐색 방법을 기반으로 하여 자동 정리 증명 시스템의 성능을 높이는 데 유리하게 설계되었다. 실제로 Aristotle은 다양한 수학 문제에 대해 대학 수준의 도전 과제를 증명할 수 있고, 이전의 연례 과정에서 이루어졌던 유사한 작업들과 더불어 그 능력을 넓혀가고 있다.



### Agentic Jigsaw Interaction Learning for Enhancing Visual Perception and Reasoning in Vision-Language Models (https://arxiv.org/abs/2510.01304)
- **What's New**: AGILE는 VLM(Visual-Language Models)의 시각 인식(visual perception)과 추론(reasoning) 능력을 향상시키기 위해 제안된 새로운 프레임워크입니다. 기존의 VLM들이 간단한 jigsaw 작업에서도 무작위로 수행되는 문제를 해결하기 위해 AGILE은 jigsaw 풀기를 상호작용적인 과정으로 모델링합니다. 이 방법은 실행 가능한 코드를 생성하여 환경과의 상호작용을 통해 점진적으로 개선됩니다.

- **Technical Details**: AGILE은 Python 코드를 생성하여 현재 상태에 기반한 행동을 실행하며, 매 단계에서 환경은 세밀한 시각적 피드백을 제공합니다. jigsaw 퍼즐의 구조적 특성을 활용하여, 모델은 이미지 타일을 교환하거나 현재의 퍼즐 상태를 관찰하는 등의 행동을 통해 더 나은 인식 및 추론 기술을 습득하게 됩니다. 이 과정의 반복적인 상호작용을 통해 모델은 시각적 구성 요소 간의 구조적 관계를 파악하게 됩니다.

- **Performance Highlights**: AGILE을 통해 jigsaw 작업에서의 성능이 크게 개선되었습니다. 예를 들어, 2 × 2 설정에서 정확도가 9.5%에서 82.8%로 증가하였으며, 9개의 일반 비전 과제에서 평균 3.1%의 일반화 성능 향상이 나타났습니다. 이러한 결과는 AGILE이 시각적 인식과 추론 능력 모두에서 상당한 향상을 나타낸다는 것을 보여줍니다.



### The Social Laboratory: A Psychometric Framework for Multi-Agent LLM Evaluation (https://arxiv.org/abs/2510.01295)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling

- **What's New**: 본 연구에서는 전통적인 평가 기준을 넘어서, LLM(대형 언어 모델)의 사회적 행동을 평가하기 위한 새로운 다중 에이전트 토론 프레임워크를 도입하였습니다. 이 프레임워크는 LLM 기반 에이전트들이 서로의 의견을 교환하며 발생하는 emergent behaviors(신흥 행동)을 찾아내고 정량화하는데 사용됩니다. 연구 결과 에이전트들은 명시적인 합의 지향 지시 없이도 높은 의미적 합의를 도달하는 강력한 경향을 보였습니다.

- **Technical Details**: 우리의 실험은 Change-My-View(CMV) 데이터셋에서 복잡한 토론 주제를 선정하여 진행되었습니다. 두 개의 'debater' 에이전트와 하나의 'moderator' 에이전트로 구성된 다중 에이전트 토론 프레임워크를 통해 진행합니。 각 에이전트는 특정한 페르소나와 인센티브를 부여받아 신뢰성 있는 분석과 설득력을 발휘합니다. 다양한 심리측정(psychometric) 및 의미론적(semantic) 지표를 활용하여 토론의 역학을 분석하였습니다.  

- **Performance Highlights**: 실험 분석 결과, LLM 에이전트들이 합의를 찾으려는 강한 경향이 있음을 확인하였고, 이는 논의의 길이에 관계없이 증가하는 경향이 있었습니다. 특히, moderator의 환경 구축이 에이전트 간의 논의 결과에 미치는 긍정적인 영향을 입증하였습니다. 논란이 되는 주제에 대해서도 높은 합의 수준을 유지하며, 이를 통해 향후 더 복잡한 에이전트의 상호 작용을 평가하는 프레임워크 구축에 기여할 수 있는 청사진을 제시하였습니다.



### Cyber Academia-Chemical Engineering (CA-ChemE): A Living Digital Town for Self-Directed Research Evolution and Emergent Scientific Discovery (https://arxiv.org/abs/2510.01293)
- **What's New**: 이 논문에서는 화학 공학 분야에서의 자율적 연구 및 과학적 발견을 촉진하기 위해 Cyber Academia-Chemical Engineering (CA-ChemE) 시스템을 제안합니다. 이 시스템은 다중 에이전트 간의 협업을 통해 지식 기반과 전문 기술을 통합하여 지능형 생태계를 구현합니다. 이를 통해 기존 AI 시스템들이 가진 학제 간 협업 부족 문제를 해결하고, 새로운 문제를 탐구하는 데 도움을 줍니다.

- **Technical Details**: Cyber Academia 시스템의 아키텍처는 다중 에이전트 시스템(MAS)으로, 각 도메인 전문가 에이전트와 협업 에이전트로 구성되어 있습니다. 각 전문가 에이전트는 특정 도메인을 전문으로 하고, 지식 기반 및 동적 지식 강화 모듈을 통해 의사 결정 능력을 향상시킵니다. 협업 에이전트(CA)는 온톨로지 엔지니어링 기술을 활용하여 서로 다른 분야의 전문가들 간의 협업을 촉진하고, 개념 표준화 및 지식 통합을 수행합니다.

- **Performance Highlights**: 지식 기반을 활용한 향상 메커니즘이 7명의 전문가 에이전트에서 평균 10~15%의 대화 품질이 향상되는 결과를 가져왔습니다. 또한, 협업 에이전트(CA)의介入이 원거리 도메인 전문가 쌍에 대해 8.5%의 성과 향상을 이루어 내어, 학습 기반의 협업 효율성을 높였습니다. 이는 전문 분야 간의 협업에서 나타나는 비효율성을 해결하기 위한 중요한 진전을 나타냅니다.



### Modeling Others' Minds as Cod (https://arxiv.org/abs/2510.01272)
- **What's New**: ROTE(Representing Others’ Trajectories as Executables) 알고리즘을 제안해 행동 예측을 보다 효율적이고 신뢰성 있게 모델링합니다. 기존 방법들과 달리, ROTE는 사람의 행동을 정책이 아닌 실행 가능한 코드로 모델링하여 인지 부담을 줄이고 더 나은 일반화를 제공합니다. 이 알고리즘은 대형 언어 모델(LLMs)을 활용하여 희소 관찰로부터 행동 프로그램을 합성하고 불확실성을 추론합니다.

- **Technical Details**: ROTE는 LLMs를 코드 합성 도구로 사용해 관찰된 행동 흔적을 설명하는 프로그램을 생성합니다. 이후 베이지안 추론(Bayesian inference)을 수행하여 가장 가능성이 높은 프로그램을 식별합니다. 이로써 다양한 환경에서의 행동 예측을 위한 동적 모델을 구축하며, 각 에이전트와 환경 간의 분석 및 수정이 가능합니다.

- **Performance Highlights**: ROTE는 여러 도전적인 환경에서 최대 50%의 정확도를 향상시켜 일반화 및 효율성을 크게 개선했습니다. 행동 예측에서 ROTE는 인간 수준의 성능을 달성하며, 이는 기존의 방법들보다 훨씬 효과적입니다. 실제 인간의 행동 데이터에 대해 ROTE는 경쟁력 있는 기법들과 비교하여 우수한 성과를 보여 앞으로의 사회적 지능형 AI 시스템 개발에 새로운 가능성을 제시합니다.



### OR-Toolformer: Modeling and Solving Operations Research Problems with Tool Augmented Large Language Models (https://arxiv.org/abs/2510.01253)
- **What's New**: 이번 연구에서는 OR-Toolformer라는 새로운 접근 방식을 소개합니다. 이 모델은 Llama-3.1-8B-Instruct를 기반으로 하며, 반자동 데이터 합성 파이프라인을 사용하여 다양하고 구체적인 OR 문제-답안 쌍을 생성합니다. 기존의 폐쇄형 API의 의존도와 고비용의 오픈소스 모델 교육 문제를 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: OR-Toolformer는 세 가지 통합 구성 요소로 OR 작업을 자동화합니다: 문제-답안 데이터 생성, LLM 세분화(fine-tuning), 외부 솔버와의 문제 해결입니다. 문제-답안 데이터 생성 과정에서는 다양한 산업 및 표현 형식을 가진 문제-답안 쌍을 합성하여 도메인과 표현의 다양성을 보장합니다. 또한, Llama-3.1-8B-Instruct 모델을 세부 조정하여 자연어 설명에서 구조화된 솔버 매개변수를 추출하고 API 호출을 생성하도록 설계되었습니다.

- **Performance Highlights**: OR-Toolformer는 네 가지 표준 벤치마크에서 최대 80.1%의 실행 정확도를 달성하며, 크기가 맞는 기존 모델보다 4.3% 이상 뛰어난 성과를 기록했습니다. 또한, 두 가지 새로운 OR 문제 유형에 대한 제로샷 평가에서 평균 54%의 정확도를 달성하여 가장 강력한 기준선 모델보다 21 퍼센트 포인트 개선된 결과를 나타냅니다. 이러한 결과는 도구 증강 방식의 세분화가 OR 문제의 정확한 모델링과 해결에 효과적임을 입증합니다.



### NoiseShift: Resolution-Aware Noise Recalibration for Better Low-Resolution Image Generation (https://arxiv.org/abs/2510.02307)
- **What's New**: 최근의 연구에서 고정된 해상도 집합에 대해 훈련된 텍스트-이미지 확산 모델이 낮은 해상도의 이미지 생성 시 일반화하지 못하는 문제를 다루었습니다. 저자들은 노이즈 스케줄러의 인지적 효과가 해상도에 따라 다르다는 점에서 문제의 큰 원인을 찾았습니다. 이를 해결하기 위해 NoiseShift라는 훈련 없이 사용할 수 있는 방법을 제안하여, 해상도 크기에 따라 노이즈 레벨을 재조정함으로써 품질을 향상시킵니다.

- **Technical Details**: NoiseShift는 기존 모델 아키텍처나 샘플링 일정을 변경하지 않고도 사용할 수 있는 경량의 방법입니다. 이 방법은 다소 연구 결과를 바탕으로 다소 정교한 그리드 검색을 통해 훈련 시 사용된 노이즈 분포에 맞는 조건부 값을 식별하여 정확한 노이즈 조정의 과정을 통해 이루어집니다. 이로 인해 낮은 해상도에서의 인지적 노이즈의 감소가 이루어지고, 미세한 세부 사항을 보존하는 데 기여합니다.

- **Performance Highlights**: NoiseShift는 Stable Diffusion 3, 3.5, 그리고 Flux-Dev 모델에 적용했을 때 낮은 해상도에서의 이미지 품질을 상당히 향상시켰습니다. LAION-COCO 데이터셋에서 NoiseShift는 SD3.5의 FID 점수를 평균 15.89% 개선했으며, CelebA에서는 10.36% 개선의 효과를 보였습니다. 이 결과들은 NoiseShift가 해상도 의존적 아티팩트를 완화하고 낮은 해상도 이미지 생성의 품질을 향상시키는 데 효과적임을 보여줍니다.



### Diffusion Models and the Manifold Hypothesis: Log-Domain Smoothing is Geometry Adaptiv (https://arxiv.org/abs/2510.02305)
- **What's New**: 이 논문은 diffusion models가 갖는 탁월한 일반화 능력을 설명하는 기존 이론을 뒷받침하는 증거를 제시합니다. 특히, 데이터를 통해 학습하는 과정에서 score matching을 통해 학습 문제를 어떻게 구성하는지가 이들 모델의 성공에 중요한 역할을 한다고 주장합니다. 이 연구는 empirical score matching 목표 함수의 smoothing 최소화에 대한 영향을 조사하며, 지리적 구조를 잃지 않는 smoothing의 중요성을 강조합니다.

- **Technical Details**: Diffusion models는 고차원 데이터가 저차원 매니폴드에 집중되어 있다는 매니폴드 가설에 기반하여 작동합니다. 이 모델들은 도출된 score 함수의 smoothing을 통해 empirical density p^t의 log-domain에서 smoothing을 수행하며, 이는 효과적으로 데이터 매니폴드를 보존합니다. 논문은 이와 관련하여, smooth approximations을 통한 inductive bias 모델을 제안하며, convolution이 gradient 연산과 교환 가능하다는 직관을 제공합니다.

- **Performance Highlights**: 모델의 성능 및 일반화 능력에 대한 실험적 결과는 smoothing이 diffusion models의 일반화 가능성을 높이는 핵심 요소임을 증명합니다. 논문에서는 log-domain에서의 smoothing이 데이터의 기하학적 구조를 유지하는 데 중요하다는 점을 강조하며, 이를 통해 데이터 매니폴드를 따라 일반화하는 여러 기하학적 경로를 탐색할 수 있음을 시사합니다. 이러한 발견들은 hybrid diffusion models 및 flow-based 접근법의 발전에도 기여할 것입니다.



### Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models (https://arxiv.org/abs/2510.02300)
- **What's New**: 이번 논문에서는 Equilibrium Matching (EqM)이라는 새로운 생성 모델링 프레임워크를 소개합니다. EqM는 기존의 비평형(Non-equilibrium) 동역학을 배제하고, 암시적(Implicit) 에너지 경관에서의 평형(Equilibrium) 기울기를 학습합니다. 이 접근법을 통해, 샘플 생성 시 최적화 기반 샘플링 프로세스를 채택하며, 이를 통해 이미지 생성의 성능을 향상시킵니다.

- **Technical Details**: Equilibrium Matching은 시간 조건 비대칭 동역학을 단일 불변 평형 기울기로 대체하여, EBM(에너지 기반 모델) 관점에서 학습합니다. 이를 통해, 샘플링 단계에서 각 샘플마다 독립적으로 적응형 옵티마이저와 단계 크기를 조정할 수 있게 되어, 60%의 함수 평가를 절약할 수 있습니다. EqM는 데이터 매니폴드에서 실체 샘플을 뽑아내고, 이를 통해 이미지 생성의 질적인 향상을 달성할 수 있습니다.

- **Performance Highlights**: 실험적으로 EqM는 ImageNet 256×256 데이터 세트에서 1.90의 FID(Fréchet Inception Distance)를 달성하며, 기존의 확산(Diffusion) 및 흐름 기반(Flow-based) 모델을 초월하는 성능을 보여줍니다. Equilibrium Matching은 다양한 크기에서 뛰어난 확장성을 갖추고 있으며, 이미지 구성, OOD 탐지, 부분적으로 노이즈가 있는 이미지 복원 등의 작업을 자연스럽게 처리할 수 있는 유연한 프레임워크입니다.



### Interactive Training: Feedback-Driven Neural Network Optimization (https://arxiv.org/abs/2510.02297)
Comments:
          EMNLP 2025 Demo

- **What's New**: 이번 논문에서는 Interactive Training이라는 새로운 오픈 소스 프레임워크를 소개합니다. 이 프레임워크는 네트워크 훈련 중에 실시간으로 피드백을 받아 인공지능이 자동으로 개입하게 하며, 사용자 또는 AI 에이전트의 개입을 허용합니다. 이를 통해 사용자는 옵티마이저의 하이퍼파라미터, 훈련 데이터 및 모델 체크포인트를 동적으로 조정할 수 있습니다.

- **Technical Details**: Interactive Training 프레임워크의 핵심은 Control Server입니다. 이 서버는 사용자의 명령과 지속적인 훈련 프로세스 간의 통신을 중재하며, FastAPI를 통해 API 엔드포인트를 노출하고 JSON 메시지를 통해 명령을 처리합니다. Interactive Trainer는 Hugging Face의 Trainer 클래스에 기반하여 구현되어, 동적으로 조정된 훈련 매개변수를 바탕으로 실시간 개입에 반응합니다.

- **Performance Highlights**: 세 가지 사례 연구를 통해 우리의 접근 방식이 기존의 정적 최적화 방법들보다 우수하다는 것을 보여줍니다. 경험이 있는 인간 개발자들이 실시간 인터랙션을 활용하여 더 나은 최적화 결과를 도출했고, AI 에이전트가 자동으로 초기 하이퍼파라미터를 수정할 수 있는 가능성도 입증되었습니다. 또한, 이 프레임워크는 실제 배포 중 수집된 사용자 데이터를 실시간으로 반영하여 모델의 적응성을 향상시킵니다.



### VideoNSA: Native Sparse Attention Scales Video Understanding (https://arxiv.org/abs/2510.02295)
Comments:
          Project Page: this https URL, Code: this https URL

- **What's New**: 이 논문에서는 비디오와 언어 모델을 위한 Native Sparse Attention (NSA)를 채택하여 VideoNSA라는 새로운 접근 방식을 제안합니다. 기존 비디오-언어 모델들의 맥락 길이가 제한된 문제를 해결하기 위해, 이 모델은 216K 비디오 지침 데이터셋에서 엔드-투-엔드(end-to-end) 훈련을 통해 Qwen2.5-VL을 조정합니다. 특히, 하드웨어에 최적화된 하이브리드 주의(attention) 접근 방식을 사용하여 텍스트에서는 밀집 주의를, 비디오에는 NSA를 적용합니다.

- **Technical Details**: VideoNSA는 세 가지 보완적인 캐시 분기를 통합한 배우는 하드웨어 인식 sparse attention 메커니즘을 사용합니다. 여기에는 Token Compression (CMP), Token Selection (SLC), Sliding Window (SWA) 분기가 포함됩니다. 이 방식은 모델이 특정 작업에 필요한 경로만을 유지하여 효율성을 극대화합니다. 특히, VideoNSA는 128K 컨텍스트 길이에서 성능 향상을 이루며, 작동 방식에서 여러 중요한 발견 사항들을 제시합니다.

- **Performance Highlights**: VideoNSA는 긴 비디오 이해(long-video understanding) 및 시간적 추론(temporal reasoning)에서 기존 방법보다 개선된 성능을 보였습니다. 또한, 다양한 실험 결과에 따라 데이터 세트의 길이를 초과하여 효과적으로 확장할 수 있는 잠재력을 보여줍니다. 가장 흥미로운 점은, 이 모델이 학습 가능한 sparse attention 가중치를 통해 다양한 태스크에서 동적인 주의 sink 행동을 유도하여 깊은 층에서의 선택 및 슬라이딩 윈도우 분기의 중요성을 감소시킨다는 것입니다.



### F2LLM Technical Report: Matching SOTA Embedding Performance with 6 Million Open-Source Data (https://arxiv.org/abs/2510.02294)
- **What's New**: F2LLM은 최신 embedding 모델로, 0.6B, 1.7B, 4B 크기로 제공된다. 기존 top-ranking embedding 모델들이 거대한 contrastive pretraining과 비싼 synthetic training data를 필요로 하는 것과 달리, F2LLM은 개방형 데이터셋에서 수집된 6백만 개의 query-document-negative tuple을 사용하여 직접 finetuned되었다. 이 모델은 훈련 비용, 모델 규모 및 embedding 성능 간의 균형을 잘 맞추며, 연구자들에게 reproducible하고 경제적인 기준을 제공한다.

- **Technical Details**: F2LLM은 Foudation 모델에서 직접 finetuned되어 6백만 개의 고품질 query-document-hard negative tuple로 구성된다. 이 데이터는 개방형 비합성 데이터셋에서만 수집되어 다양한 작업 유형을 커버한다. 훈련은 여러 작업들에 대해 통합된 포맷으로 진행되며, 각 데이터 샘플은 query와 positive passage, 여러 개의 hard negative tuple로 구성된다.

- **Performance Highlights**: 현재 F2LLM-4B는 MTEB 영어 리더보드에서 약 4B 파라미터 모델 중 2위, 전체 7위를 기록하고 있다. 또한, F2LLM-1.7B는 1B-2B 사이즈의 모델 중 1위를 기록하였으며, 제한된 컴퓨팅 자원으로 사용할 수 있는 이상적인 선택지로 평가받고 있다.



### Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (https://arxiv.org/abs/2510.02286)
- **What's New**: 이 연구는 DialTree-RPO라는 새로운 강화 학습 프레임워크를 제안하여 다중 턴 공격 전략을 자율적으로 발견하는 방법을 제시합니다. 기존의 접근 방식들이 단일 턴 공격에 초점을 맞춘 것과 달리, DialTree-RPO는 대화를 연속적인 의사 결정 문제로 간주하여 다중 턴 공격의 다양한 가능성을 탐색합니다. 이 과정에서, 자동화된 데이터 없이도 공격 성공률을 극대화할 수 있는 최적의 대화 정책을 학습합니다.

- **Technical Details**: DialTree-RPO는 다중 턴 공격을 위한 정책 최적화를 목표로 하는 새로운 강화 학습 프레임워크입니다. 이 시스템은 대화 나무 롤아웃 (dialogue tree rollout) 및 질 높은 경로 프루닝 (quality-aware pruning) 기법과 적응적 마스킹 (adaptive masking) 기술을 통합하여, 훈련 중 비효율적인 공격 경로를 제거하고 최적화 안정성을 높이며 효율성을 개선합니다. 이를 통해 다중 턴 대화의 복잡성을 효과적으로 적용할 수 있습니다.

- **Performance Highlights**: DialTree-RPO는 10개의 대상 모델에서 평균 공격 성공률 (ASR) 85.3%를 달성하여 이전 최신 기술보다 25.9% 더 높은 성능을 보였습니다. 이 연구는 모델 크기와 관계없이 뛰어난 효율성과 전이 능력을 보여주며, 새로운 공격 전략을 발견하는 데에도 효과적입니다. 결과적으로 DialTree-RPO는 다중 턴 강화 학습을 기반으로 한 새로운 최첨단 레드 팀링 방법을 수립했습니다.



### Learning to Generate Object Interactions with Physics-Guided Video Diffusion (https://arxiv.org/abs/2510.02284)
- **What's New**: KineMask는 물리 기반 비디오 생성 기능을 개선하기 위한 새로운 접근법으로, 강체체의 제어 및 상호작용을 현실감 있게 수행할 수 있도록 합니다. 이 방법은 단일 이미지와 특정 객체 속도를 제공받아 미래의 물체 상호작용을 예측하는 비디오를 생성합니다. 이 연구는 또한 저수준 모션 제어와 고수준 텍스트 조건화를 통합하여 복잡한 동적 현상을 효과적으로 합성할 수 있도록 합니다.

- **Technical Details**: KineMask는 물리적으로 유도된 비디오 생성을 위한 프레임워크로, 객체의 방향과 속도 같은 저수준 운동 제어를 제공합니다. 이를 통해 모델은 객체 상호작용을 추론하며, 두 단계의 훈련 전략을 사용하여 객체 마스크를 통해 미래의 모션 감독을 점진적으로 제거합니다. KineMask는 시뮬레이터에서 생성된 비디오를 기반으로 훈련되며 물리적으로 유효한 역학과 명확한 객체 상호작용을 캡처합니다.

- **Performance Highlights**: KineMask는 기존의 동영상 확산 모델들과 비교하여 뛰어난 객체 상호작용을 보여줍니다. 광범위한 실험 결과, KineMask는 유사한 크기의 최신 모델들 대비 강력한 개선을 이끌어냈고, ablation 연구를 통해 저수준과 고수준 제어의 통합 중요성을 강조했습니다. 추가적으로, KineMask는 복잡한 상호작용의 일반화를 통해 실제 장면에서의 비디오 생성에서 두드러진 성능 향상을 보여줍니다.



### Self-Forcing++: Towards Minute-Scale High-Quality Video Generation (https://arxiv.org/abs/2510.02283)
Comments:
          preprint

- **What's New**: 이 논문에서는 긴 비디오 생성(long video generation)의 품질 저하를 완화하기 위한 간단하면서도 효과적인 접근 방식을 제안합니다. 기존의 비디오 생성 모델이 긴 비디오를 생성하는 데 필요한 감독(supervision) 없이도 학생 모델(student model)을 지도하는 방법을 중심으로 합니다. 이 방법은 교사 모델(teacher model)의 지식을 활용하여 자신이 생성한 긴 비디오의 샘플링한 세그먼트를 학생 모델을 안내하는 방식으로 동작합니다.

- **Technical Details**: 저자들은 학생 모델이 긴 비디오 생성에 있어 발생하는 오류가 누적되는 문제를 해결하기 위해, 시간적 일관성(temporal consistency)을 유지하며 최대 20배 긴 비디오 길이를 확장할 수 있는 방법을 고안하였습니다. 이러한 방식은 이전 방법들이 갖고 있는 오버 익스포저(over-exposure) 및 오류 누적(error-accumulation) 문제를 피하면서도 더 많은 프레임을 재계산하지 않습니다. 실험 결과, 이 방법은 4분 15초 길이의 비디오 생성이 가능함을 입증하며, 이는 기본 모델의 위치 임베딩(position embedding)으로 지원되는 최대 범위의 99.9%에 해당합니다.

- **Performance Highlights**: 제안된 방법은 기준(baseline) 방법들에 비해 충실도(fidelity)와 일관성(consistency) 모두에서 월등한 성능을 보이고 있습니다. 저자들은 표준 벤치마크(benchmark)와 자신들이 제안한 개선된 벤치마크에서 실험을 수행하여 입증된 결과를 제공합니다. 또한, 이 논문에서 시연한 긴 지평선의 비디오(demo of long-horizon videos)는 웹 링크에서 확인할 수 있습니다.



### Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation (https://arxiv.org/abs/2510.02279)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 신뢰도를 저해하는 일반적인 문제인 환각(hallucinations)에 대한 심층 분석을 제시합니다. 특히, LLM의 예측 불확실성(predicitive uncertainty)으로 인해 발생하는 'confabulation'이라는 특정 환각 유형을 식별합니다. 다양한 방법론이 개발되어 자연어 생성(NLG)에서의 예측 불확실성을 측정하고 있지만, 기존의 근사 정확성 함수들이 상이함을 강조하고, 평가의 편향(bias)을 해소할 수 있는 대안 적 위험 지표를 제안합니다.

- **Technical Details**: 환각 감지를 위해 NLG에서의 예측 불확실성을 정량화하는 것은 LLM의 예측 분포 엔트로피를 기반으로 합니다. 최근 불확실성 추정 알고리즘은 주로 질문 답변(QA) 데이터셋과 같은 좁은 문제 범주에서 선택적 예측(selective prediction)으로 평가되고 있습니다. 연구팀은 confabulation 감지를 위해 다양한 LLM-as-a-judge 변형과 더불어 OOD(out of distribution) 탐지 및 변동 감지 작업 등을 포함한 구조화된 작업을 연구합니다. 이 구조들은 더 강력하고 통제 가능한 위험 지표를 제공합니다.

- **Performance Highlights**: 분석 결과, 여러 LLM-as-a-judge 변형을 통합하여 평가할 경우 평가 편향이 줄어드는 것으로 나타났습니다. LLM-as-a-judge를 정확성 메트릭으로 활용해야 하며, 엘로(Elo) 랭킹 기법을 통해 다양한 실험 설정에서 불확실성 추정 방법의 성능을 보다 객관적으로 평가할 수 있음을 보여줍니다. 이러한 방법은 기존의 문맥에서 제기된 문제점들을 개선하는데 기여할 것입니다.



### Parallel Scaling Law: Unveiling Reasoning Generalization through A Cross-Linguistic Perspectiv (https://arxiv.org/abs/2510.02272)
Comments:
          Work in progress

- **What's New**: 이 연구는 Reinforcement Post-Training (RPT)를 기반으로 한 대형 추론 모델(Large Reasoning Models, LRM)의 다국어 일반화 가능성을 탐구합니다. 기존 연구는 주로 과제 또는 양식 간의 일반화에 중점을 두었으나, 본 연구는 새로운 교차 언어적 관점을 제시하여 영어에서 다른 언어로의 추론 능력 이전을 평가합니다. 이러한 접근 방식은 LRM의 본질적인 언어 비의존성을 확인하는 데 중점을 두고 있습니다.

- **Technical Details**: 연구는 영어 중심의 LRM을 기반으로 하여 11개 언어의 다국어 추론 벤치마크에서 English-centric LRM의 성능을 체계적으로 평가합니다. Multilingual Transferability Index (MTI)를 도입하여 모델의 교차 언어 이식성을 정량화하고, 다양한 초기 모델, 목표 언어, 훈련 패라다임에 따른 차이를 분석합니다. 또한, 병렬 훈련 연구를 통해 단일 기계에서 다국어 훈련의 중요성과 그 효과를 이론적으로 뒷받침합니다.

- **Performance Highlights**: 실험 결과, 모델이 단일 언어에서 단일 병렬 언어로 전환할 때 현저한 성능 향상인 First-Parallel Leap를 발견하였으며, 병렬 언어 수에 따라 성능이 전력 법칙을 따른다는 Parallel Scaling Law를 제시합니다. 또한 실제 단일 언어 성능과 예측된 성능 간의 차이를 나타내는 Monolingual Generalization Gap을 확인하여, 영어 중심 모델이 타 언어로의 일반화에 실패함을 보여줍니다. 이 연구는 LRM의 추론 능력이 인간의 인지 방식과는 마 주다는 점에서 중요한 통찰을 제공합니다.



### InfoMosaic-Bench: Evaluating Multi-Source Information Seeking in Tool-Augmented Agents (https://arxiv.org/abs/2510.02271)
- **What's New**: 이 논문에서는 정보 탐색을 위한 새로운 벤치마크인 InfoMosaic-Bench를 소개합니다. LLM(대형 언어 모델) 에이전트가 도메인 특정 도구와 일반 검색을 결합하여 복잡한 작업을 수행하는 효율성을 평가하는 데 중점을 두고 있습니다. InfoMosaic-Bench는 의료, 금융, 지도, 비디오, 웹, 다중 도메인 통합의 6개 대표 도메인에서 621개의 합성 과제를 포함하고 있습니다.

- **Technical Details**: InfoMosaic-Bench는 InfoMosaic-Flow라는 확장 가능한 파이프라인을 통해 과제를 생성하며, 이는 태스크 조건을 검증된 도구 출력에 기반하고 다원 소스 의존성을 강제합니다. 또한, 단순 검색으로 해결할 수 있는 경우를 필터링하여 신뢰성과 비단순성을 보장합니다. 이를 통해 에이전트는 여러 도메인 도구를 통합하여 신뢰할 수 있는 정보를 탐색할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 현재의 에이전트들은 여전히 기본적인 도구 처리 능력에도 어려움을 겪고 있으며, 도메인 도구의 부정확한 사용으로 인해 총 실패의 22.4%가 발생한다는 사실이 밝혀졌습니다. GPT-5는 단독 웹 정보에 의존할 경우 38.2%의 정확도에 그쳤으며, 특정 도메인에서 도구가 선택적으로 이점을 제공하지만 전체적으로 일관성이 부족하다는 점이 드러났습니다.



### microCLIP: Unsupervised CLIP Adaptation via Coarse-Fine Token Fusion for Fine-Grained Image Classification (https://arxiv.org/abs/2510.02270)
- **What's New**: 이번 논문에서는 CLIP 기반의 비전-언어 모델(VLM)에서 미세한 이미지 분류를 위한 비지도 적응(unsupervised adaptation)의 한계를 극복하기 위해 $	extbf{microCLIP}$를 제안합니다. 기존의 접근법은 LLM(large language model)의 설명을 CLIP의 [CLS] 토큰과 정렬하는 방식이었으나, 이로 인해 공간적인 정밀성이 부족했습니다. $	extbf{microCLIP}$은 Saliency-Oriented Attention Pooling (SOAP)이라는 신기술을 통해 시각적 및 텍스트 표현을 동시에 정제합니다.

- **Technical Details**: 이 연구는 CLIP의 시각적 표현과 LLM으로부터 유래된 텍스트적인 선행 지식을 융합하는 비지도 자기 훈련 프레임워크를 개발했습니다. SOAP 메커니즘은 CLIP 패치 토큰을 기반으로 강도를 기반으로 한 쿼리를 생성하고, 이를 통해 compact한 [FG] 토큰을 풀링합니다. 이어서 TokenFusion 모듈을 사용하여 [FG] 토큰을 CLIP의 глобал [CLS] 토큰과 융합하여 coarse-fine 정렬을 수행합니다.

- **Performance Highlights**: 제안된 방법은 13개의 미세한 데이터셋에서 평균 2.90%의 정확도 향상을 달성했으며, 이는 미세한 적응을 요구하지 않는 경량 조정만으로 가능합니다. 연구 결과 이 방법이 CLIP의 잠재적인 미세 신호를 발견하는 데 도움을 준다는 것이 입증되었습니다. 또한, 샘플링된 지역적 주의 집중을 통해 클래스 정의에 중요한 로컬 의미를 지속적으로 강조하며 다양한 시나리오에서 효율적인 성능을 보였습니다.



### How to Combat Reactive and Dynamic Jamming Attacks with Reinforcement Learning (https://arxiv.org/abs/2510.02265)
- **What's New**: 이 논문은 reactive jamming 문제를 다루고 있으며, 여기서 jammer는 동적인 정책을 사용하여 채널과 감지 임계값을 선택하여 전송을 방해합니다. 전송기-수신기 쌍은 reinforcement learning (RL)을 활용하여 방해를 회피하고, 채널 조건이나 방해 전략에 대한 사전 지식 없이도 전송 전력을 조정하며 최적의 throughput을 극대화합니다. 이는 RL 방법을 통한 비선형 적응 성능을 보장합니다.

- **Technical Details**: 이 시스템 모델은 전송기와 수신기 간의 통신을 고려하며, jammer의 영향을 받는 상황을 Markov decision process (MDP)로 모델링합니다. 전송기는 각 시간 슬롯에서 전송 전력과 변조 방식을 선택하며, jammer는 에너지 검출기를 사용하여 방해 여부를 결정합니다. 이 논문에서는 Q-learning과 Deep Q-Networks (DQN)를 사용하여 분산 및 연속 상태 공간을 위한 학습을 수행합니다.

- **Performance Highlights**: 결과는 RL이 변화하는 방해 전략 및 스펙트럼 조건에 신속하게 적응하여 높은 전송률을 지속할 수 있음을 보여줍니다. 다양한 보상 함수와 행동 세트를 통해 RL 방법이 적응할 수 있는 능력을 강조하며, 이로 인해 무선 통신에서의 링크 신뢰성을 개선할 수 있는 가능성을 확인했습니다.



### Paving the Way Towards Kinematic Assessment Using Monocular Video: A Preclinical Benchmark of State-of-the-Art Deep-Learning-Based 3D Human Pose Estimators Against Inertial Sensors in Daily Living Activities (https://arxiv.org/abs/2510.02264)
Comments:
          All tables, graphs and figures generated can be obtained in the Zenodo repository complementary to this work: this https URL

- **What's New**: 이번 연구는 기계 학습(machine learning)과 착용 가능한 센서(wearable sensors)의 발전이 사람의 움직임을 전문 실험실 외부에서 캡처하고 분석할 수 있는 새롭고 유망한 기회를 제공한다고 소개합니다. VIDIMU 데이터셋을 활용하여, 일상적인 임상 관련 활동에서 비디오 카메라와 관성 측정 장치(inertial measurement units, IMUs)를 함께 사용하여 단일 비디오 기반으로 3D 인간 자세를 생성하는 모델을 비교하였습니다.

- **Technical Details**: 연구에서는 MotionAGFormer, MotionBERT, MMPose 2D-to-3D pose lifting과 NVIDIA BodyTrack와 같은 최신 심층학습(d deep learning) 프레임워크를 사용하여 도출된 관절 각도를 IMU 데이터로부터 계산된 관절 각도와 비교하였습니다. OpenSim의 역기구학(inverse kinematics) 기법을 사용하여 Human3.6M 데이터셋 형식에 따라 17개의 주요 관절 지점을 기반으로 분석을 진행했습니다. 이 초기 연구는 건강한 피실험자만을 대상으로 하여, 결과를 병리학적인 집단에 일반화할 수 없다는 한계가 있습니다.

- **Performance Highlights**: MotionAGFormer는 전체 RMSE(최소 제곱 오차) $9.27°c 4.80°$, MAE(평균 절대 오차) $7.86°c 4.18°$로 가장 우수한 성과를 보였으며, Pearson 상관계수($0.86 c 0.15$)와 결정계수($R^{2}$, $0.67 c 0.28$)에서도 가장 높은 점수를 기록하였습니다. 이 연구는 두 기술 모두 실외에서 생체역학적 평가(kinematic assessment)에 적합하다는 것을 보여주지만, 비용, 접근성, 정밀도 간의 주요 트레이드오프(trade-off)를 강조합니다. 또한 임상에서 유망한 생체역학적 데이터를 제공하는 비디오 모델과 IMU 기반 평가 사이의 차이를 규명하였습니다.



### DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing (https://arxiv.org/abs/2510.02253)
Comments:
          Preprint

- **What's New**: 이 논문에서는 DragFlow라는 새로운 드래그 기반 이미지 편집 프레임워크를 제안합니다. 이 프레임워크는 더 강력한 생성적 prior를 활용하여 편집 작업에서 현저한 성과를 보여줍니다. 이전의 UNet 기반 모델의 한계를 넘어, FLUX와 같은 새로운 모델들로부터의 이점을 최대한 활용합니다.

- **Technical Details**: DragFlow는 affine transformations를 통해 지역 기반 편집 패러다임을 도입하여 일관된 피쳐(supervision) 지침을 제공합니다. 또한, pretrained open-domain personalization adapters(예: IP-Adapter)를 통합하여 배경 품질을 유지하면서도 주제 일관성을 향상시킵니다. 다양한 작업의 모호성을 해결하기 위해 다중 모달 대형 언어 모델(MLLMs)을 사용합니다.

- **Performance Highlights**: DragFlow는 새로운 Region-based Dragging benchmark(ReD Bench)과 DragBench-DR에서 광범위한 실험을 통해 점 기반 및 지역 기반의 기존 기준을 뛰어넘는 성과를 보여줍니다. 이로 인해 드래그 기반 이미지 편집에서 새로운 최첨단(state-of-the-art)을 설정하였습니다. 코드와 데이터셋은 발표 시에 공개될 예정입니다.



### Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation (https://arxiv.org/abs/2510.02249)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 과도한 사고(overthinking) 문제를 해결하기 위해 새로운 지표인 Token Entropy Cumulative Average (TECA)를 도입합니다. TECA는 추론 과정에서의 탐색 정도를 측정하며, 이를 활용하여 모델이 최적의 시점에서 사고를 종료하도록 돕는 새로운 사고 패러다임인 'Explore Briefly, Then Decide'를 제안합니다. 이를 통해 모델은 기술적인 깊이를 복잡도에 맞춰 조정할 수 있는 능력을 확보하게 됩니다.

- **Technical Details**: TECA는 각각의 추론 단계에서 모델의 불확실성을 나타내는 토큰 엔트로피(token entropy)의 누적 평균을 계산합니다. 이를 기반으로 탐색(exploration) 단계와 결정(determination) 단계를 구분하여 과도한 탐색을 제어하는 Cumulative Entropy Regulation (CER) 메커니즘을 도입합니다. CER은 모델이 불필요하게 긴 사고 과정을 피하도록 돕는 동시에 필요한 탐색 능력을 유지하게 합니다.

- **Performance Highlights**: 제안한 방법을 통해 다양한 수학 문제 벤치마크에서 모델의 응답 길이가 평균 71%까지 감소하면서도 문제 해결 능력은 거의 유지되었음을 확인했습니다. 특히, Qwen3-4B 모델은 GSM8K에서 71%의 응답 길이 감소를, MATH500에선 39.25% 감소를 기록하며, 기존 방법들보다 전반적으로 우수한 성능을 나타냈습니다. 이러한 실험 결과는 TECA를 지표로 활용해 추론 과정을 적절히 조정함으로써 과도한 사고를 감소시킬 수 있음을 보여줍니다.



### ExGRPO: Learning to Reason from Experienc (https://arxiv.org/abs/2510.02245)
- **What's New**: 본 논문에서는 강화학습을 통한 검증 가능한 보상(Reinforcement Learning from Verifiable Rewards, RLVR)의 새로운 패러다임을 제안하고, 이에 대한 효율적인 경험 관리 방법론인 Experiential Group Relative Policy Optimization (ExGRPO)을 제시합니다. ExGRPO는 경험의 가치를 평가하기 위해 롤아웃의 정확성과 엔트로피를 활용하며, 모델이 과거의 경험을 효과적으로 재사용할 수 있도록 도와줍니다. 이 방법은 대규모 언어 모델의 추론 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: ExGRPO는 롤아웃 중 생성된 경험을 효율적으로 관리하기 위해 경험의 정확도에 따라 버킷으로 조직하고, 가장 유용한 버킷에서 경험을 우선적으로 샘플링하는 전략을 채택합니다. 또한, 이 방법은 샘플 효율성을 개선하고 훈련 안정성을 증대시키기 위해 신선한 탐색과 과거 경험의 재사용을 균형있게 조절하는 혼합 정책 최적화(mixed-policy optimization) 목표를 사용합니다. 향후 연구의 중요성을 강조하고 있는 이 방법론은 경험 관리가 RLVR에서 필수적인 요소임을 보여줍니다.

- **Performance Highlights**: ExGRPO는 1.5B에서 8B 매개변수를 가진 다섯 가지 백본 모델에서 RLVR 성능을 평균적으로 +3.5/7.6 포인트 이상 개선하며, 수학적 및 일반 벤치마크에서 일관된 성과를 나타냅니다. 특히, ExGRPO는 기존의 온-정책 방식의 최적화가 실패하는 모델에서도 훈련을 안정화시키는 데 성공하여, 전체적인 학습 성과를 높였습니다. 이러한 연구 결과는 과거의 경험을 효과적으로 활용하는 것이 대규모 언어 모델의 성능을 향상시키는 데 중요한 역할을 한다는 것을 강조합니다.



### RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning (https://arxiv.org/abs/2510.02240)
- **What's New**: 이 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 미세한 시각적 추론(fine-grained visual reasoning) 능력 향상을 위한 새로운 데이터셋 ReasonMap-Plus와 RL 프레임워크 RewardMap을 제안합니다. ReasonMap-Plus는 시각적 질문 응답(Visual Question Answering, VQA) 작업을 통해 밀집 보상 신호를 도입하여 효과적인 학습을 지원합니다. RewardMap은 시각적 이해와 추론 능력을 개선하는 다단계 RL 프레임워크로, 난이도 인식 보상 설계와 복잡한 작업으로 나아가는 훈련 전략을 통해 기존 방법의 한계를 극복합니다.

- **Technical Details**: ReasonMap-Plus는 VQA 작업에 바탕을 두고 있어 단순한 인식부터 복잡한 추론 작업까지 여러 난이도 단계로 구성되어 있습니다. RewardMap은 두 가지 주요 설계를 포함하고 있습니다: 하나는 작업 복잡성을 고려한 세부 보상 체계로, sparse 보상의 문제를 해결하고 더 많은 감독을 제공합니다; 다른 하나는 차례로 진행되는 RL 절차로, SFT보다 효과적인 시작 전략을 제공합니다. 각 단계의 보상은 접근하기 쉬운 밀집 보상을 포함하고 있어 효과적인 학습을 촉진합니다.

- **Performance Highlights**: 각 구성 요소가 일관된 성능 향상에 기여하며, 이들의 조합이 최상의 결과를 나타냅니다. RewardMap으로 훈련된 모델은 공간 추론, 미세한 시각적 추론 및 일반 작업에서 평균 3.47%의 성능 개선을 달성했습니다. 이러한 결과는 구조화된 시각적 작업에서 MLLMs의 능력을 강화하는 데 기여합니다.



### More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration (https://arxiv.org/abs/2510.02227)
Comments:
          20 pages, 5 figures

- **What's New**: 이 논문은 Reinforcement Learning with Verifiable Rewards (RLVR) 기법을 사용하여 대형 언어 모델(LLM)의 추론 능력을 개선하는 새로운 패러다임인 Adaptive Multi-Guidance Policy Optimization (AMPO)를 소개합니다. 기존의 방법보다 더 다양한 탐색을 허용하고, 불필요한 개입 없이 자기 탐색의 가치를 보존하면서도 여러 능숙한 교사 모델의 지도를 적절히 활용합니다. 이를 통해 AMPO는 더 나은 효과성과 일반화를 제공하여 모델의 성능을 크게 향상시킵니다.

- **Technical Details**: AMPO는 혼합 정책 강화 학습(Mixed-Policy RL) 프레임워크이며, 여러 동료 모델의 집합적 지능을 활용합니다. 이 방법은 학생 모델이 문제를 해결하지 못할 경우에만 외부 지도를 제공하는 'guidance-on-demand' 원칙을 적용합니다. 또한, 학생이 이해하기 쉬운 추론 경로를 학습하도록 유도하는 이해 기반 지침 선택 메커니즘을 도입하여 대폭적인 탐색과 효과적인 활용 간의 균형을 맞춥니다.

- **Performance Highlights**: 실험 결과 AMPO는 강력한 기준선 모델인 GRPO보다 평균 4.3% 및 분포 외(out-of-distribution) 작업에서 12.2%의 개선을 보여주었습니다. 특히, 4명의 동료 크기 교사를 사용함으로써 하나의 강력한 교사를 사용하는 방법과 유사한 성능을 보이며, 더 적은 데이터로도 높은 성능을 기록했습니다. 이러한 결과는 AMPO가 탐색과 활용 간의 우수한 균형을 이루는 방법임을 보여줍니다.



### TempoControl: Temporal Attention Guidance for Text-to-Video Models (https://arxiv.org/abs/2510.02226)
Comments:
          Under Review

- **What's New**: 최근 생성 비디오 모델의 발전으로 자연어 프롬프트를 기반으로 한 고품질 비디오 생성이 가능해졌습니다. 하지만 이러한 모델은 개별 시각적 요소의 출현 시점을 지정할 수 있는 세분화된 시간 제어 부족이 있습니다. 이 연구에서는 TempoControl이라는 방법을 소개하여 추가 훈련이나 감독 없이 추론 시점에서 비주얼 개념을 시간적으로 정렬할 수 있도록 하였습니다.

- **Technical Details**: TempoControl은 텍스트-비디오 확산 모델의 주요 구성 요소 중 하나인 크로스 어텐션 맵을 활용하여 비주얼 컨셉의 타이밍을 유도하는 새로운 최적화 접근 방식을 적용합니다. 이 방법은 세 가지 보완 원칙(상관관계, 에너지, 엔트로피)을 통해 어텐션을 조정합니다. 또한, 모델 파라미터를 업데이트하지 않고 적절한 수준의 시간적 정렬을 달성하기 위해 몇 차례의 확률적 경량 하강법(SGD)을 적용합니다.

- **Performance Highlights**: TempoControl은 단일 및 다중 객체를 포함한 다양한 비디오 생성 응용 프로그램에서 효과성을 입증하였습니다. 특히, 객체의 시간 재배열 및 행동과 오디오 정렬 생성에서 뛰어난 성능을 보여주었습니다. 우리의 접근 방식은 추가 훈련 없이도 비디오 생성에서 외부 오디오 신호와의 정렬 가능성을 탐색할 수 있는 잠재력을 가지고 있습니다.



### DiFFPO: Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning (https://arxiv.org/abs/2510.02212)
- **What's New**: 본 논문에서는 DiFFPO(Diffusion Fast and Furious Policy Optimization)라는 통합 프레임워크를 제안하여, masked diffusion large language models (dLLMs)를 훈련하는 새로운 방법을 소개합니다. 이 프레임워크는 강화학습(Reinforcement Learning, RL)을 통해 dLLMs의 추론 능력을 더욱 빠르고 효과적으로 향상시키는 것을 목표로 합니다. 기존의 baseline 방법인 d1을 통합하여 오프 정책(off-policy) RL을 통해 주요 정책에 대한 보다 신뢰할 수 있는 근사치를 제공합니다.

- **Technical Details**: DiFFPO는 두 단계의 likelihood 근사화와 중요도 샘플링 보정(importance sampling correction)을 활용하여 샘플 효율성과 태스크 성능을 극대화합니다. 특히, 새로운 서라게이트 정책을 도입하여 대응 생성 수준에서 추가적인 레이턴트를 사용하는 방식으로 보다 정교한 근사화를 시도합니다. 문제 해결을 위한 RL 알고리즘을 통해 dLLMs의 다중 토큰 예측 능력을 더욱 잘 활용할 수 있게 됩니다.

- **Performance Highlights**: DiFFPO는 함수 평가 수(NFE)를 낮추면서도 더 높은 정확도를 달성하였으며, 수학 및 계획(Planning) 작업에서 dLLMs를 훈련하여 기존의 최고 성능을 초과하는 결과를 보여주었습니다. 이 연구는 dLLMs의 추론 시간을 줄이고 효율적인 RL 알고리즘 설계의 한계성을 넘어서기 위한 새로운 방향성을 제시합니다. 이러한 단계적인 접근이 결국 보다 출중한 LLM 모델 개발에 기여할 것으로 기대됩니다.



### Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025 (https://arxiv.org/abs/2510.02202)
Comments:
          13 pages, 2 figures

- **What's New**: 이번 연구는 Chagas 질병을 정전도(ECG) 데이터를 통해 탐지하는 새로운 알고리즘 개발을 위한 PhysioNet Challenge 2025를 소개하고 있습니다. 준수할 데이터셋과 레이블을 확장하여 모델의 강건성을 높이는 혁신적인 접근 방식을 채택했습니다. 또한, 머신러닝 문제를 선별(task)로 프레임하기 위해 지역적인 혈청 검사 능력을 캡처하는 평가 지표를 도입했습니다.

- **Technical Details**: PhysioNet Challenge 2025에서는 378,624개의 12리드 ECG 레코드를 여러 출처에서 수집했습니다. 이 데이터셋은 훈련 세트와 숨겨진 검증 및 테스트 세트를 포함하여 준비되었습니다. 우리는 Chagas 양성 및 음성 사례에 대한 균형을 맞춰 다수의 공개 및 비공식 데이터 소스를 사용하였습니다.

- **Performance Highlights**: 도전 과제에는 111개 팀의 630명 이상의 참가자가 참여하였으며, 1300개 이상의 제출물이 있었습니다. 다양하고 창의적인 방법으로 Chagas 질병을 식별할 수 있는 알고리즘이 제안되었습니다. 이러한 노력은 Chagas 질병의 조기 탐지와 효과적인 치료에 기여할 것으로 기대됩니다.



### ARUQULA -- An LLM based Text2SPARQL Approach using ReAct and Knowledge Graph Exploration Utilities (https://arxiv.org/abs/2510.02200)
Comments:
          peer reviewed publication at Text2SPARQL Workshop @ ESWC 2025

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)을 활용하여 자연어 질문을 SPARQL 쿼리로 변환하는 탈바꿈된 접근방식을 제안합니다. 특히, 기존 SPINACH의 아이디어를 일반화하여 여러 개의 지식 그래프에 적용할 수 있도록 하였으며, 이를 TEXT2SPARQL 챌린지에 맞춰 구축한 과정을 상세히 설명합니다. 이 과정에서 자연어 질문의 염두에 둔 탐색 및 실행을 위한 반복적 프로세스를 도입하여 더 나은 쿼리 생성을 목표로 하고 있습니다.

- **Technical Details**: 본 연구에서는 ReAct 접근법을 사용하여 LLM이 자연어 질문을 올바른 SPARQL 쿼리로 변환할 수 있는 구조를 설계했습니다. 시스템은 LangGraph와 RPT 및 Qlever를 이용하여 SPARQL 엔드포인트를 설정하였으며, Qdrant 벡터 데이터베이스와 Lucene 텍스트 검색을 결합하여 하이브리드 검색을 구현했습니다. 마지막으로, 다양한 지식 그래프 탐색 유틸리티와 함께 LLM이 과거의 모든 동작과 관찰을 참고할 수 있도록 설계되었습니다.

- **Performance Highlights**: TEXT2SPARQL 챌린지에서는 DBpedia와 Corporate Knowledge Graph(CKG)라는 두 개의 데이터 세트를 사용하여 시스템의 성능을 평가했습니다. DBpedia에서는 다국어 쿼리를 처리하는 확장성, CKG에서는 도메인 적합성 및 정확도를 평가하여 LLM의 효용성을 종합적으로 분석했습니다. 실험 결과, 기존 LLM을 기반으로 한 접근법보다 높은 수준의 정확도와 처리를 보여주는 안정성을 입증하였습니다.



### EvolveCaptions: Empowering DHH Users Through Real-Time Collaborative Captioning (https://arxiv.org/abs/2510.02181)
- **What's New**: EvolveCaptions는 청각 장애인과 경청자 간의 실시간, 협력적 ASR(Automatic Speech Recognition) 개인화 시스템이다. 이 시스템은 대화 중 발생하는 ASR 오류를 청각 참여자가 수정하며, 이를 통해 DHH(Deaf and Hard of Hearing) 사용자의 음성을 조정할 수 있는 단축된 생성 프롬프트를 제공한다. EvolveCaptions는 대화에서의 오접수를 최소화하여 편리하게 ASR 모델을 조정할 수 있는 새로운 접근 방식을 제안한다.

- **Technical Details**: EvolveCaptions는 실시간으로 DHH 사용자의 발화를 ASR로 전사하며, 청각 참여자가 오류를 수정하는 과정을 통해 데이터 수집과 모델 조정이 이루어진다. 이 과정에서 발생한 오류 수정은 언어 모델(GPT-4)을 통해 phonetically plausible phrases로 변환되며, 이러한 문장을 DHH 사용자가 기록하여 모델의 미세 조정에 활용된다. 반복 과정은 사용자의 생활 환경에서 자연스럽게 발생하며, 추가적인 녹음 부담 없이 효율적인 접근성을 제공한다.

- **Performance Highlights**: EvolveCaptions는 12명의 DHH 참가자와 6명의 청각 참가자와 함께한 연구에서 WER(Word Error Rate)를 평균 27.2% 감소시켰다. DHH 참가자들은 이 시스템이 직관적이고 사용이 용이하다고 평가했으며, 청각 파트너는 실시간 수정이 점차 용이해졌다고 보고했다. 이러한 결과는 실시간, 사용자별 적응 방식이 기존의 고정 방식에 비해 명백한 장점을 있음을 입증한다.



### GRACE: A Language Model Framework for Explainable Inverse Reinforcement Learning (https://arxiv.org/abs/2510.02180)
- **What's New**: 이 논문에서는 전통적인 보상 모델이 해석하기 힘든 "블랙박스" 특성을 가진 반면, GRACE(Generating Rewards As CodE)라는 방법을 통해 대형 언어 모델을 사용하여 전문가 시연에서 직접 해석 가능한 보상 함수를 역설계하는 방법을 제안하고 있습니다. 이 방법은 전문가의 행동을 바탕으로 생성된 보상 함수를 기반으로 하며 실행 가능한 코드 형태로 제공되어, 검증이 가능하고 확인할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: GRACE는 진화적 탐색( evolutionary search )에서 영감을 받아 전문가의 시연을 기반으로 보상 모델이 되는 프로그램을 생성하고 다듬는 최적화 절차를 제공합니다. 이를 통해 생성된 보상은 이전에 비해 적은 양의 시연으로도 효과적인 보상을 학습하는 것이 가능합니다. 또한 GRACE는 다중 작업 설정에서 복잡한 보상 API를 구성하여 일반화 능력을 증가시킵니다.

- **Performance Highlights**: GRACE는 BabyAI와 AndroidWorld와 같은 벤치마크에서 실험적으로 검증되었으며, 강력한 정책을 생성하는 데 있어 경쟁사인 Imitation Learning( imitational learning )과 온라인 강화 학습( online RL )에서 진실 보상과 비교하여 우수한 성능을 보여주었습니다. 연구 결과, 이 방식으로 생성된 보상은 효과적이고 정보가 풍부한 중간 신호를 제공함으로써, 다양한 환경에서 효과적인 에이전트 구축을 가능하게 합니다.



### Learning to Reason for Hallucination Span Detection (https://arxiv.org/abs/2510.02173)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 환각(span) 탐지를 위한 새로운 접근 방식을 소개합니다. 기존 연구들은 환각 탐지를 이진(binary) 문제로 정의했지만, 실제로는 특정 환각 부분을 식별해야 하는 필요성이 존재합니다. 이 문제를 해결하기 위해, Chain-of-Thought (CoT) 추론을 활용한 강화 학습 프레임워크인 RL4HS를 제안합니다. 이 방법은 환각 탐지를 보다 정교하게 수행할 수 있는 가능성을 보여줍니다.

- **Technical Details**: RL4HS는 Group Relative Policy Optimization(GRPO) 기반으로 설계되며, span 수준의 보상 함수를 통해 추론을 장려합니다. 구체적으로, 환각 부분을 포함한 레이블 데이터셋을 사용하여 코트(CoT) 추론 기반 모델을 학습합니다. 또한, 비환각 예측에 대한 보상 불균형 문제를 해결하기 위해 class-aware policy optimization 기법을 도입했습니다. 이를 통해 보다 효과적인 모델 학습을 달성합니다.

- **Performance Highlights**: 실험 결과, RL4HS 모델이 사전 훈련된 추론 모델과 감독 학습(Supervised Fine-tuning) 방법보다 우수한 성능을 보여 명확히 증명되었습니다. 특히, 기존의 제너레이션 모델보다 환각 탐지에 더 효과적인 성과를 나타냈습니다. 또한, span-F1 점수 기준으로 강화 학습 접근 방식이 우월함을 흐릿한 기준에서 더 높게 평가받은 점도 주목할 만합니다.



### Go witheFlow: Real-time Emotion Driven Audio Effects Modulation (https://arxiv.org/abs/2510.02171)
Comments:
          Accepted at NeurIPS Creative AI Track 2025: Humanity

- **What's New**: 이 논문에서는 인간과 기계 간의 협업을 살펴보는 일환으로, 실시간 음악 공연을 개선하기 위해 설계된 witheFlow 시스템을 소개합니다. 이 시스템은 바이오신호와 오디오에서 추출한 특성을 기반으로 오디오 효과를 자동으로 조절하여, 공연자가 더 나은 음악적 표현을 할 수 있게 지원합니다. witheFlow는 경량화된 모델을 사용하여 로컬에서 실행할 수 있고, 호환 가능한 Digital Audio Workstation과 센서가 있으면 오픈소스로 제공됩니다.

- **Technical Details**: witheFlow 시스템은 세 가지 주요 구성 요소로 이루어져 있으며, 각각 바이오신호 기반의 '감정 상태' 기능 추출기, 오디오 기반의 Valence-Arousal (VA) 공간에서 감정 회귀기, 규칙 기반의 믹싱 논리 모듈이 포함됩니다. 이 시스템은 Python으로 구현되어 있으며, 모듈 간의 통신은 MIDI 프로토콜 메시지를 통해 이루어집니다. EEG 및 ECG 센서를 사용하여 주의/이완, 스트레스 지수를 추정하고, 이를 바탕으로 오디오 효과를 동적으로 조정하도록 설계되었습니다.

- **Performance Highlights**: witheFlow 시스템은 공연자의 실시간 감정 상태를 반영하여 오디오 효과를 조정하는 방식으로, 보다 고차원적인 음악적 표현을 가능하게 합니다. 시스템은 다양한 바이오신호 센서를 통해 얻은 데이터를 분석하여, 공연이 진행될 때마다 적절한 오디오 믹스를 생성합니다. 이와 같이 각각의 공연자에 맞춤형으로 작동하여, 인간의 음악적 창의성을 강화하고, 공연자가 자신의 감정을 표현할 수 있는 기회를 제공합니다.



### SIEVE: Towards Verifiable Certification for Code-datasets (https://arxiv.org/abs/2510.02166)
Comments:
          5

- **What's New**: 본 연구는 SIEVE라는 커뮤니티 주도의 프레임워크를 제안합니다. SIEVE는 기존의 데이터셋 카드를 대신하여, 머신이 읽을 수 있고 검증 가능한 신뢰 카드(Confidence Cards)를 생성하여 데이터셋 품질을 효과적으로 평가할 수 있게 합니다. 이를 통해 품질 보증 비용이 절감되고 코드 데이터셋에 대한 신뢰가 증가할 것으로 예상됩니다.

- **Technical Details**: SIEVE는 체크를 기계적으로 검증 가능한 인증서로 변환하여 무작위 샘플링과 오프체인 검증을 통해 투자자 및 사용자 간의 신뢰를 높입니다. 거래에서 데이터셋 및 감사 규칙을 고정하는 스마트 계약을 활용하며, 실시간으로 위험을 모니터링합니다. 또한 데이터셋 사용자가 경량 속성 체크를 수행할 수 있도록 하여, 데이터셋 품질을 지속적으로 관리할 수 있습니다.

- **Performance Highlights**: SIEVE는 현재의 데이터셋 감사 프로세스를 간소화하여, 중복 노력을 줄이고 신뢰를 제고하는 데 기여할 수 있습니다. 연구자와 엔지니어는 공개된 샘플을 통해 저렴한 비용으로 데이터셋 속성을 확인할 수 있으며, 신뢰 카드가 지속적으로 업데이트되어 실시간으로 데이터셋의 신뢰성을 평가할 수 있게 됩니다.



### Comparing Contrastive and Triplet Loss in Audio-Visual Embedding: Intra-Class Variance and Greediness Analysis (https://arxiv.org/abs/2510.02161)
Comments:
          8 pages, 4 tables, 3 figures

- **What's New**: 본 연구에서는 Contrastive Loss와 Triplet Loss의 이론적 및 실증적 비교를 통해 두 손실 함수가 표현 품질에 미치는 영향을 분석합니다. 특히 intra-class variance와 inter-class variance에 관한 실험을 통해 Triplet Loss가 클래스 내외의 변동성을 더 잘 유지한다는 것을 보여줍니다. 이 결과는 세밀한 구별이 요구되는 상황에서 Triplet Loss의 사용을 권장하고, Contrastive Loss는 보다 부드럽고 넓은 처리에서 유리하다는 결론에 이릅니다.

- **Technical Details**: Deep Metric Learning은 입력을 의미적 유사성이 반영된 공간에 매핑하여 임베딩을 형성하는 과정을 포함합니다. Contrastive Loss는 긍정 및 부정 쌍을 사용하여 intra-class의 집합성을 유지하고, inter-class의 분리를 강제합니다. 반면, Triplet Loss는 앵커, 긍정, 부정의 세 개를 사용하는데, 이는 더 강력한 업데이트를 통해 학습을 지속하게 하여 어려운 예제를 잘 다뤄냅니다.

- **Performance Highlights**: MNIST와 CIFAR-10 데이터셋에서 Triplet Loss가 Classification 및 Retrieval 작업에서 비교 우위를 보였습니다. Triplet Loss는 더 큰 gradient norm을 생성하여 어려운 샘플에 집중된 업데이트를 가능하게 하며, 이는 클래스 간 분리를 더 명확하게 드러냄을 의미합니다. 이러한 결과는 각 손실 함수 선택 시의 전략을 정의하는 데 도움을 줍니다.



### Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting (https://arxiv.org/abs/2510.02155)
Comments:
          14 pages, video anomaly detection

- **What's New**: ASK-Hint는 비디오 이상 탐지를 위한 구조화된 프롬프트 프레임워크로, 기존의 추상적인 프롬프트의 한계를 극복하고 더 세밀한 인간-객체 상호작용과 행동의 의미에 기반한 프롬프트를 제안합니다. 이 방법은 이상 탐지에 있어 보다 정확하고 해석 가능한 추론을 유도하며, 프롬프트를 의미적으로 일관된 그룹(예: 폭력, 재산 범죄)으로 조직합니다. ASK-Hint는 UCF-Crime 및 XD-Violence 데이터셋에서 이전 방법들보다 AUC를 일관되게 개선하며, 해석 가능성 또한 제공합니다.

- **Technical Details**: ASK-Hint는 세 가지 주요 구성 요소를 가집니다: (1) 클래스별 프롬프트 구성, (2) 의미 있는 프롬프트 클러스터링 및 압축, (3) 설명 trace와 함께 구조화된 추론. 이러한 설계를 통해 프롬프트의 세분화된 행동 설명을 사용하여 VLM(vision-language model)의 추론 능력을 탐구할 수 있습니다. 또한, ASK-Hint는 행동 중심의 프롬프트를 통해 정확성과 효율성을 높입니다.

- **Performance Highlights**: ASK-Hint는 VLM을 고정(frozen) 상태로 유지하면서도 비디오 이상 탐지를 위한 훈련 없는 솔루션으로 강력한 일반화를 달성합니다. UCF-Crime 및 XD-Violence 데이터셋에서의 대규모 zero-shot 평가를 통해, ASK-Hint는 이전 최첨단 성과를 지속적으로 초과하며 해석 가능성과 견고한 일반화 능력을 입증했습니다. 이 결과는 프롬프트의 세분화가 비디오 이상 탐지에서 뛰어난 성능을 발휘하는 데 중요한 역할을 함을 보여줍니다.



### Human-Robo-advisor collaboration in decision-making: Evidence from a multiphase mixed methods experimental study (https://arxiv.org/abs/2510.02153)
- **What's New**: 이 연구는 Robo-advisors (RAs)의 역할을 개인이 어떻게 해석하는지와 그들의 조언이 의사결정에 통합되는 방식을 조사합니다. 기존의 연구들은 사용자와 RAs 간의 상호작용에 주목했으나, 이번 연구는 그 이상으로 RAs에 대한 신뢰를 형성하는 기초를 파악하고자 했습니다. 특히, RAs의 성과에 대한 정보와 조언의 프레이밍(Framing)이 의사결정에 어떻게 영향을 미치는지에 대한 통찰을 제공합니다.

- **Technical Details**: 이 연구는 N=334의 행동 실험과 주제 분석(Thematic analysis), 후속 정량적 테스트(Quantitative testing)를 포함한 다단계 혼합 방법론(Mixed methods)을 사용합니다. 연구 결과, 사용자는 RAs에 의존하는 경향이 있으며, 이는 RAs의 성과 정보와 조언의 손익 프레이밍에 의해 영향을 받습니다. 또한, 의사결정 과정에서 RAs의 세 가지 역할과 조언 통합 방식에 따라 네 가지 사용자 유형을 발견했습니다.

- **Performance Highlights**: 이 연구는 인간과 RA 협업에 대한 이해를 발전시키고, 더 신뢰할 수 있으며 적응형(adaptive) RA 시스템 설계를 위한 실행 가능한 인사이트를 제공합니다. 조사된 수용의 전제 조건은 개인적 수준과 알고리즘적 수준에서 각각 촉진자(Enablers)와 억제자(Inhibitors)로 구분되었습니다. 이러한 통찰은 RAs의 효과적인 도입을 높이는 방향으로 기여할 수 있습니다.



### How to Find Fantastic Papers: Self-Rankings as a Powerful Predictor of Scientific Impact Beyond Peer Review (https://arxiv.org/abs/2510.02143)
- **What's New**: 이 논문에서는 인공지능(AI) 분야의 연구에서 고충격(high-impact) 연구를 식별하는 데 있어 저자 자신의 제출물에 대한 순위를 활용하는 방법을 탐구합니다. 기존의 동료 검토(peer review) 시스템과는 달리 저자들이 자신의 연구에 대한 독특한 이해를 가지고 있는 점에 착안해, 저자들의 자기 순위(self-rankings)가 미래 연구의 방향과 잠재력을 파악하는 데 얼마나 유용한지를 실험적으로 분석했습니다.

- **Technical Details**: 연구는 2023년 ICML에서 1,342명의 연구자가 제출한 2,592개의 논문에 대해 자기 순위를 매기도록 요청하고, 그 결과와 동료 검토 점수 및 최종 결정과의 상관관계를 분석했습니다. 특히, 저자들이 자기 순위를 매긴 논문은 후속 인용(citations) 수에서 특히 두 배의 증가율을 보였습니다. 자기 순위는 동료 검토 점수보다 더 정확한 인용 예측력을 보여 주었으며, 실험 결과는 선행 연구에 의해 통제된 변수를 고려하더라도 신뢰할 수 있었습니다.

- **Performance Highlights**: 저자들이 가장 높게 순위를 매긴 논문은 평균적으로 낮게 순위 매긴 논문보다 두 배 많은 인용을 받았고, 특히 150회 이상의 인용을 받은 22개의 논문 중 18개는 저자들이 본인 순위에서 가장 높게 평가한 논문들이었습니다. 이 연구 결과는 자기 순위가 고충격 연구를 식별하는 데 있어 귀중하고도 비용 효율적인 신호를 제공할 수 있음을 시사합니다. 향후 실제 저자 자기 순위를 기반으로 한 동료 검토 시스템 개선이 필요할 것으로 보입니다.



### BioinfoMCP: A Unified Platform Enabling MCP Interfaces in Agentic Bioinformatics (https://arxiv.org/abs/2510.02139)
Comments:
          20 pages, 8 figures, 3 tables

- **What's New**: BioinfoMCP는 생물정보학 도구와 AI 에이전트를 연결하는 혁신적인 플랫폼입니다. 이 플랫폼은 두 가지 주요 구성 요소인 BioinfoMCP Converter와 BioinfoMCP Benchmark로 구성되어 있습니다. BioinfoMCP Converter는 도구 문서를 이용해 대형 언어 모델을 통해 자동으로 MCP 서버를 생성하고, BioinfoMCP Benchmark는 변환된 도구의 신뢰성과 다양성을 검증합니다.

- **Technical Details**: BioinfoMCP Converter의 변환 과정은 준비, 실행 및 전달의 세 단계로 나뉩니다. 준비 단계에서는 도구의 매뉴얼을 준비하고, 실행 단계에서는 LLM(대형 언어 모델)을 통해 서버 코드를 생성합니다. 변환된 MCP 서버는 FastMCP 2.0 프레임워크를 기반으로 하여 생산 수준의 효율적인 서버를 제공합니다.

- **Performance Highlights**: BioinfoMCP 플랫폼은 38개의 MCP 변환 생물정보학 도구를 포함하고 있으며, 이들 도구는 94.7%의 성공률로 복잡한 작업 흐름을 실행했습니다. 이는 AI 에이전트와의 통합에 기술적 장벽을 제거하여, 연구자들이 깊은 프로그래밍 기술 없이도 자연어로 복잡한 분석을 수행할 수 있게 합니다.



### The Disparate Impacts of Speculative Decoding (https://arxiv.org/abs/2510.02128)
- **What's New**: 이 논문에서는 대형 언어 모델의 디코딩 시간을 단축시키기 위해 사용되는 'speculative decoding'의 분석을 수행합니다. 특히, 이 기법에서 각 작업(task) 간 속도 향상이 균등하게 분포하지 않는다는 점을 강조하며, 이는 저조한 피팅(fitness)을 보이는 작업에서 특히 두드러집니다. 결과적으로, 불공정한 속도 향상 문제를 해결하기 위한 완화 전략을 제안하고, 이 방법이 여러 모델 쌍에서 평균 12%의 개선을 보여줍니다.

- **Technical Details**: Speculative decoding은 'drafter' 모델이 제안하는 토큰을 'verifier' 모델이 검증하는 방식으로 작동합니다. 이 논문에서는 drafter와 verifier 모델의 조건부 토큰 분포 간의 정렬(alignment)이 속도 향상에 미치는 영향을 분석합니다. 또한, 속도 불공정성(unfairness)을 정량화할 수 있는 새로운 개념을 도출하고, drafter 모델의 피팅(fitness)과 속도 향상 간의 관계를 탐구합니다.

- **Performance Highlights**: 이번 연구의 결과는 특정 작업이 상대적으로 느린 속도 향상을 경험한다는 계산적 불공정성을 드러냅니다. 실험에서 일본어와 같은 특정 언어의 경우 낮은 정확도와 속도 향상 지표가 나타났습니다. 이러한 결과는 속도 향상과 언어별 정확도 사이의 관계를 밝혀내며, 이는 전체적인 모델 접근의 공정성에 중요한 영향을 미칩니다.



### VarCoNet: A variability-aware self-supervised framework for functional connectome extraction from resting-state fMRI (https://arxiv.org/abs/2510.02120)
Comments:
          My preview .pdf was not loading. Can you please share with me a compiled .pdf file so I can confirm that the result is correct?

- **What's New**: 이 연구에서는 개인 간 뇌 기능의 변동성을 단순한 노이즈가 아닌 의미 있는 데이터로 고려하여 VarCoNet이라는 자가 지도 학습(self-supervised learning) 기반의 새로운 프레임워크를 소개합니다. VarCoNet은 정적 상태 fMRI(resting-state fMRI) 데이터를 사용하여 강력한 기능적 연결체(functional connectome) 추출을 가능하게 합니다. 이 방법은 레이블이 없는 데이터에서도 유용하게 사용될 수 있으며, 개인의 뇌 기능을 인코딩하는 역할을 합니다.

- **Technical Details**: VarCoNet은 1D-CNN-Transformer 인코더를 통합하여 고급 시간 연속 데이터를 처리하며, 강력한 베이지안 하이퍼파라미터 최적화를 통해 성능을 향상시킵니다. 이 프레임워크는 기능적 개인 간 변동성을 활용하기 위해 세분화(segmentation)된 신호에 기초한 새로운 증강 전략을 사용하여 대조 학습(contrastive learning)을 지원합니다. 이러한 구조는 다양한 세션에서의 데이터 신호 길이 차이를 강건하게 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: VarCoNet의 성능은 휴먼 연결체 프로젝트(Human Connectome Project) 및 자폐 스펙트럼 장애(ASD) 분류에 사용된 ABIDE I 및 II 데이터셋을 통해 평가되었습니다. 두 가지 주요 작업에서 최신 방법들과 비교했을 때 VarCoNet은 일관되게 우수한 성능을 보였으며, 모델의 학습된 표현에 대한 해석 가능성도 제공하여 임상적 관련성을 지원합니다.



### SpurBreast: A Curated Dataset for Investigating Spurious Correlations in Real-world Breast MRI Classification (https://arxiv.org/abs/2510.02109)
Comments:
          Accepted for publication in the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2025

- **What's New**: 이 연구에서는 SpurBreast라는 새로운 커리키팅된 유방 MRI 데이터셋을 소개합니다. 이 데이터셋은 고의적으로 비정상적인 상관관계를 포함하여 모델의 성능에 미치는 영향을 평가할 수 있게 설계되었습니다. 연구진은 환자, 장치, 이미징 프로토콜을 포함한 100개 이상의 특성을 분석했고, 두 가지 주요 비정상 신호인 자기장 강도와 이미지 방향성을 확인했습니다.

- **Technical Details**: SpurBreast 데이터셋은 DUKE 유방암 데이터셋을 기반으로 하며, 900명 이상의 환자의 3D MRI 스캔으로 구성되어 있습니다. 데이터는 양성 및 음성 종양 슬라이스로 나뉘며, 총 100개 이상의 특성을 가지고 있지만 대부분의 분포는 불균형합니다. 기계 학습에서 데이터를 훈련, 테스트, 검증 서브셋으로 임의로 나누는 전통적인 접근 방식 대신, 이 연구에서는 명확하게 정의된 비정상 상관관계를 포함한 데이터셋을 생성하기 위해 독창적인 방법론을 적용했습니다.

- **Performance Highlights**: 모델은 ResNet-50과 Vision Transformer (ViT-B/16) 아키텍처를 사용하여 훈련되었습니다. 연구 결과, 비정상적인 신호를 활용하여 높은 검증 정확도를 달성했지만, 불편견이 없는 테스트 데이터세트에서는 성능 저하가 두드러졌습니다. 따라서 연구팀은 비정상 상관관계가 임상적으로 유의한 패턴에 미치는 영향을 이해하는 기준을 마련하고, 보다 강력하고 일반화 가능한 AI 모델 개발이 가능하도록 했습니다.



### Unlocking Symbol-Level Precoding Efficiency Through Tensor Equivariant Neural Network (https://arxiv.org/abs/2510.02108)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문은 symbol-level precoding (SLP)의 높은 복잡성을 감소시키기 위해 end-to-end deep learning (DL) 프레임워크를 제안합니다. 이 프레임워크는 최적 SLP 솔루션의 구조와 tensor equivariance (TE)를 활용하여 낮은 추론 복잡성을 실현합니다. 결과적으로, 다수의 사용자와 심볼 블록 길이에 걸쳐 강력한 일반화를 유지하면서 기존 방법보다 약 80배 빠른 속도로 최적 SLP의 성능을 크게 개선할 수 있음을 보여주고 있습니다.

- **Technical Details**: 논문은 다양한 변조 방식에 대응할 수 있는 통합된 DL 기반 SLP 프레임워크를 개발하고, 이를 위해 NNLS(Negative Non-negative Least Squares) 및 TE의 특성을 활용합니다. 저자들은 Karush-Kuhn-Tucker (KKT) 조건을 분석하고, 이를 통해 최적 지연 인자를 조정하는 매핑을 정의합니다. 또한, AMDE(Attention-based Multi-Dimensional Equivariance) 모듈을 제안하여 강력한 표현력을 제공하며, SLPN 네트워크를 통해 낮은 파라미터 수와 높은 일반화를 보장합니다.

- **Performance Highlights**: 제안된 TENN 기반의 SLP 프레임워크는 최적 SLP의 성능 향상을 대부분 유지하면서도 약 80배의 속도 향상을 이룰 수 있음을 시뮬레이션 결과를 통해 확인하였습니다. 이 프레임워크는 다양한 채널 환경에서 학습되어 여러 채널 실현에서의 배포가 가능하게 되었습니다. 마지막으로, 불완전 CSI( Channel State Information) 시나리오에서도 적용 가능성을 보여주며, MMSE 견고한 SLP 방식으로 RSLPN 네트워크를 설계하여 이점을 살리고 있습니다.



### When Tracking Fails: Analyzing Failure Modes of SAM2 for Point-Based Tracking in Surgical Videos (https://arxiv.org/abs/2510.02100)
Comments:
          Accepted for publication in the 28th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) Workshop on Collaborative Intelligence and Autonomy in Image-guided Surgery (COLAS), 2025

- **What's New**: AI 기반 기술이 수술 과정에서 사용자 입력을 최소화하면서도 효과적인 객체 추적을 가능하게 하는 비디오 객체 분할 (VOS) 모델인 SAM2의 가능성을 보여줍니다. 특히, 점 기반 추적(point-based tracking)은 비용 효율적인 방법으로 주목받고 있으나, 복잡한 수술 환경에서의 신뢰성과 실패 사례에 대해서는 잘 알려져 있지 않았습니다. 이 연구에서는 복강경 담낭 절제술(laparoscopic cholecystectomy) 비디오에서 SAM2의 점 기반 추적 성능을 체계적으로 분석하였습니다.

- **Technical Details**: SAM2.1 Hiera Large는 계층적 트랜스포머 아키텍처를 사용하는 최첨단 제로샷 세그멘테이션 모델로, 강력한 공간적 및 시간적 추론을 지원합니다. 이 연구에서는 CholecSeg8k 데이터세트를 활용하여 10개의 비디오 구간과 3개의 목표 물체를 선정하여 점 기반 추적 성능을 평가하였습니다. 다양한 점 선택 전략을 사용하여 추적 성능에 미치는 영향을 조사하고, 각 전략마다 평균 IoU 점수를 비교 분석합니다.

- **Performance Highlights**: 실험 결과, 점 기반 추적은 해부학적 목표에 대해서는 일관되게 세그멘테이션 마스크 기반 추적보다 저조한 성능을 보였습니다. 그러나 수술 도구인 그라스퍼(grasper)와 L-hook 전기 소작기에서는 경쟁력 있는 성능을 발휘하였고, 경우에 따라 세그멘테이션과 유사한 정확도를 달성했습니다. 중요한 실패 모드는 점 기반 추적이 해부학적 구조의 모호한 경계에서 힘들어 한다는 것이며, 이는 추적 성공의 핵심 요소 중 하나로 확인되었습니다.



### KAIROS: Unified Training for Universal Non-Autoregressive Time Series Forecasting (https://arxiv.org/abs/2510.02084)
- **What's New**: KAIROS는 시간 시계열 예측을 위한 새로운 비자기회계 프레임워크로, 세그먼트 수준의 다중 봉우리 분포를 직접 모델링하여 기존의 방식보다 더 나은 성능을 발휘합니다. 기존의 자가회귀(AR, autoregressive) 접근 방식을 피하고 오류 축적을 방지하며, 실시간 응답을 필요로 하는 웹 애플리케이션에 최적화된 예측을 제공합니다. 특히, KAIROS는 경험적 결과를 통해 다양한 벤치마크에서 제로샷 제너럴리제이션(Zero-shot generalization) 성능을 보여주어 주목받고 있습니다.

- **Technical Details**: KAIROS는 세 가지 상호작용 메커니즘으로 구성되며, 각 세그먼트에 혼합 전문가(Mixture-of-Experts) 예측 헤드를 사용하여 다중 봉우리 분포 문제를 해결합니다. 이 모델은 외부 요인(learnable exogenous vectors)을 포착하여 개별 세그먼트에 대한 고유한 조건 정보를 제공하고, 세그먼트 간 인과적 관계를 유지하기 위해 인과 잔여 노이즈(Segment Causal Residual Noise) 기법을 도입하여 세그먼트 예측의 일관성을 높입니다. 이러한 접근 방식은 AR 모델의 비효율성을 피하면서도 지속적인 예측을 가능하게 합니다.

- **Performance Highlights**: KAIROS는 여러 예측 벤치마크에서 최신 기술(state-of-the-art)의 제로샷 성능을 달성하며, 자가회귀 모델에 비해 현저히 빠른 추론 속도를 자랑합니다. 장기 예측 과제에서도 뚜렷한 장점을 보여주어, 종합적으로 기존의 시간 시계열 모델보다 우수한 효율성과 정확성을 제공합니다. 이는 KAIROS가 변화하는 웹 환경에 적합한 확장 가능한 솔루션임을 증명합니다.



### The Current State of AI Bias Bounties: An Overview of Existing Programmes and Research (https://arxiv.org/abs/2510.02036)
Comments:
          6,227 words (18 pages, from abstract to appendix), one figure, one table, and an appendix with an additional table

- **What's New**: AI 시스템에 영향을 받는 커뮤니티와의 소통이 부족했던 기존의 편향(bias) 평가 방법에 대해 새로운 접근법이 제안되었다. 버그 바운티(Bug Bounty)에서 영감을 받은 편향 바운티(Bias Bounty) 프로그램은 사용자들이 AI 시스템과의 상호작용 중 발견한 편향을 보고하도록 유도하는 보상 기반 방법으로, 이를 통해 커뮤니티의 참여를 장려하고자 한다.

- **Technical Details**: 이번 연구에서는 현재 존재하는 AI 편향 바운티 프로그램을 조사하고, 편향 바운티에 관한 학술 문헌을 분석하였다. 구글, 구글 스칼라(Google Scholar), PhilPapers, IEEE Xplore를 통해 미국 내 5개의 편향 바운티 프로그램과 관련된 5개의 연구 논문을 식별했으며, 이들 프로그램은 시간 제한이 있는 공모전 형태로 운영되고 있다.

- **Performance Highlights**: 편향 바운티 프로그램은 7,000에서 24,000 USD의 상금 풀을 가지고 있으며, 네 가지 프로그램은 공공 참여를 허용하고 있다. 연구 문헌에는 버그 바운티의 알고리즘 피해에 대한 적용 보고서, 트위터의 편향 바운티에 관한 기사, AI 감사를 증가시키기 위한 기관적 메커니즘으로서의 편향 바운티 제안 등이 포함되어 있다. 향후 연구에서는 편향 바운티의 채택을 확대할 수 있는 방법을 모색해야 한다.



### LiLa-Net: Lightweight Latent LiDAR Autoencoder for 3D Point Cloud Reconstruction (https://arxiv.org/abs/2510.02028)
Comments:
          7 pages, 3 figures, 7 tables, Submitted to ICRA

- **What's New**: 이 연구에서는 LiLa-Net이라는 3D 자동 인코더 아키텍처를 제안합니다. 이 모델은 LiDAR의 포인트 클라우드만을 사용하여 실제 교통 환경에서 효율적인 특징을 인코딩합니다. LiLa-Net은 인코더 레이어 수를 줄이고, 기존 아키텍처보다 리소스를 적게 소모하면서도 원래 포인트 클라우드를 정확하게 재구성할 수 있는 효율적이고 대표적인 잠재 공간을 생성합니다. 또한, 스킵 연결과 잠재 인코딩 간의 정보 균형을 효과적으로 맞춰 재구성 품질을 향상시킵니다.

- **Technical Details**: LiLa-Net은 점 구름의 특성 추출 및 재구성을 위한 경량화된 종단 간 (end-to-end) 프레임워크입니다. 이 모델은 먼저 데이터 전처리를 진행한 후, 인코더 (encoder)를 통해 가장 관련성이 높은 특징을 추출하여 압축된 잠재 표현을 생성합니다. 이후 인코더의 스킵 연결과 함께 디코더 (decoder)로 입력되어 최종 포인트 클라우드를 재구성합니다. 연구를 위해 Universidad Carlos III de Madrid의 AMPL 연구실에서 자체 데이터 세트를 수집하였으며, 이를 통해 수집된 포인트 클라우드는 RANSAC 알고리즘을 사용해 지면 점을 제거하여 전처리됩니다.

- **Performance Highlights**: LiLa-Net은 교통 중심의 데이터 세트를 통해 검증된 후, 관련 없는 객체에 적용했을 때도 강한 적응성을 보여주었습니다. 이 모델은 포인트 클라우드를 활용하여 복잡한 교통 환경을 효과적으로 압축하고 재구성하며, 다른 데이터 세트에서 훈련 없이도 강력한 분류 성능을 달성합니다. 또한, 기존의 프리트레이닝(pretraining)이나 마스킹 전략을 제거하여 더 간단하고 빠른 훈련 파이프라인을 구축하였습니다.



### Generating Findings for Jaw Cysts in Dental Panoramic Radiographs Using GPT-4o: Building a Two-Stage Self-Correction Loop with Structured Output (SLSO) Framework (https://arxiv.org/abs/2510.02001)
Comments:
          Intended for submission to Scientific Reports

- **What's New**: 이번 연구에서는 OpenAI GPT-4o의 다중 모달(multimodal) 능력을 활용하여 치과 파노라마 방사선 사진에서 턱 낭종(jaw cyst) 발견을 자동 생성했습니다. 이를 통해 정확성을 높이기 위해 Self-correction Loop with Structured Output (SLSO) 프레임워크를 구축하였고, 그 효과성을 검증하였습니다.

- **Technical Details**: 22개의 턱 낭종 사례에 대해 10단계 과정을 구현하였습니다. 이 과정은 이미지 입력 및 분석, 구조화된 데이터 생성, 치아 번호 추출 및 일관성 확인, 불일치가 감지될 때 반복 재생성을 포함하며, 최종적으로 발견 생성 및 재구성과 일관성 검증 단계로 이루어졌습니다.

- **Performance Highlights**: 제안된 SLSO 프레임워크는 치아 번호, 치아 이동 및 뿌리 흡수에서 각각 66.9%, 33.3%, 28.6%의 향상률을 보였으며, 성공적인 사례에서는 최대 5번의 재생성 후 일관된 구조화된 출력을 얻었습니다. 그러나 데이터셋의 크기가 작아 통계적 유의미성에 도달하지 않았음에도, 전체적인 SLSO 프레임워크는 부정적인 발견 설명을 강화하고, 헛것(hallucinations)을 억제하며, 치아 번호 식별 정확성을 향상시켰습니다.



### Clarifying Semantics of In-Context Examples for Unit Test Generation (https://arxiv.org/abs/2510.01994)
Comments:
          accepted in the research track of ASE 2025

- **What's New**: 이번 논문에서는 CLAST라는 새로운 기법을 소개합니다. 이 기법은 유닛 테스트의 의미적 명확성을 개선하기 위해 체계적으로 테스트를 정제하는 방법입니다. CLAST는 복잡한 테스트를 논리적으로 더 명확한 형태로 분해하고, 프로그램 분석과 LLM 기반 재작성의 조합을 통해 의미적 명확성을 높입니다.

- **Technical Details**: CLAST는 다수의 테스트 시나리오로 인한 복잡한 로직을 해결하기 위해 테스트를 분할하여 각 테스트가 단일 시나리오를 설명하도록 합니다. 또한, CLAST는 LLM과 프로그램 분석을 활용하여 각 정제된 테스트에 대한 식별자와 필수 주석을 정제합니다. 이렇게 재정제된 테스트는 원본 테스트의 효과성을 유지하면서 의미적으로 표현력이 강화됩니다.

- **Performance Highlights**: CLAST는 UTgen보다 테스트의 효과성을 유지하는 데에서 월등한 성과를 보여줍니다. 연구 결과, 85.33% 이상의 사용자들이 CLAST로 정제된 테스트의 의미적 명확성을 선호한다고 응답했습니다. CLAST로 정제된 테스트 예제를 사용했을 때, ICL 기반 유닛 테스트 생성 방법(RAGGen 및 TELPA)의 효과가 평균 25.97%에서 45.99%까지 향상되었습니다.



### ZK-WAGON: Imperceptible Watermark for Image Generation Models using ZK-SNARKs (https://arxiv.org/abs/2510.01967)
Comments:
          Accepted at AI-ML Systems 2025, Bangalore, India, this https URL

- **What's New**: 이번 논문에서는 이미지 생성 모델에 대한 새로운 워터마킹 시스템인 ZK-WAGON을 소개합니다. 기존의 워터마킹 방법들이 가지는 단점을 극복하여, 고유성(proof of origin)을 검증할 수 있는 안전하고 확장 가능한 방법을 제시합니다. 더불어, 모델의 내부 정보를 노출하지 않고도 이미지 생성 모델의 신뢰성을 높이는 역할을 합니다.

- **Technical Details**: ZK-WAGON은 Zero-Knowledge Succinct Non-Interactive Argument of Knowledge (ZK-SNARKs)를 활용하여 이미지 워터마킹을 구현합니다. Selective Layer ZK-Circuit Creation (SL-ZKCC) 방법을 통해, 키 레이어를 선택적으로 회로로 변환함으로써 증명 생성 시간을 크게 단축시킵니다. 생성된 ZK-SNARK 증명(proof)은 Least Significant Bit (LSB) 스테가노그래피를 통해 생성된 이미지에 눈에 띄지 않게 삽입됩니다.

- **Performance Highlights**: 이 시스템은 GAN 및 Diffusion 모델 모두에서 테스트되었으며, 안전하고 모델에 구애받지 않는 신뢰할 수 있는 AI 이미지 생성 파이프라인을 제공합니다. 워터마킹을 통해 이미지의 출처를 검증할 수 있으며, 이는 주로 정보 왜곡 및 지적 재산권 위반을 방지하는 데 중요한 역할을 합니다.



### Exploring Resolution-Wise Shared Attention in Hybrid Mamba-U-Nets for Improved Cross-Corpus Speech Enhancemen (https://arxiv.org/abs/2510.01958)
Comments:
          Submitted to IEEE for possible publication

- **What's New**: 이번 논문에서는 Mamba와 attention 메커니즘을 결합한 모델이 음성 향상(speech enhancement) 분야에서 뛰어난 교차 데이터 세트 일반화 성능을 보여주고 있음을 보고합니다. 특히, U-Net 구조 내에 Mamba를 통합하여 최신의 향상 성능을 달성하며, 모델 크기와 계산 복잡성을 동시에 줄이는 방법을 제시합니다. 이에 따라, Mamba와 다중 헤드 attention을 결합하여 교차 데이터 세트 성능을 개선한 새로운 모델 RWSA-MambaUNet을 제안합니다.

- **Technical Details**: RWSA (Resolution-wise Shared Attention)란 시간 및 주파수 해상도에 따른 레이어 간 attention 공유를 의미합니다. 제안된 RWSA-MambaUNet 모델은 U-Net 구조를 기반으로 하며, Mamba와 multi-head attention을 조합하여 효율성을 극대화합니다. 이 모델은 여러 해상도에서 attention을 공동으로 활용하여 성능을 향상시키는 데 중점을 둡니다.

- **Performance Highlights**: RWSA-MambaUNet 모델은 두 개의 도메인 외 테스트 세트에서 최신의 일반화 성능을 기록했습니다. 특히, 가장 작은 모델이 PESQ, SSNR 및 ESTOI 기준으로 도메인 외 DNS 2020 테스트 세트에서 모든 기준선을 초과했으며, SSNR, ESTOI 및 SI-SDR 기준으로 도메인 외 EARS-WHAM_v2 테스트 세트에서도 우위를 점했습니다. 이 모든 성과는 모델 파라미터가 절반 이하이고 FLOPs의 일부만 사용하는 상황에서도 이루어졌습니다.



### Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors (https://arxiv.org/abs/2510.01934)
Comments:
          23 pages, 13 figures. Code is available at \url{this https URL}

- **What's New**: 이 논문은 산업 안전 검사를 간소화하고 효율적으로 수행할 수 있는 Few-shot anomaly detection 기술을 제안합니다. 기존의 방법들보다 적은 수의 샘플로 정상과 비정상적 특징을 구분할 수 있는 새로운 접근을 소개하고 있습니다. 특히, FoundAD라는 새로운 Few-shot anomaly detector를 설계하여 비정상 이미지를 효과적으로 감지할 수 있도록 하였습니다.

- **Technical Details**: FoundAD는 자연 이미지 매니폴드(natural image manifold) 위에 비선형 프로젝션 연산자(nonlinear projection operator)를 학습함으로써 구현됩니다. 이 간단한 연산자는 이미지 내의 분포에서 벗어난 영역(out-of-distribution regions)을 특성화하고 식별하는 데 효과적으로 사용됩니다. 또한, 여러 기초 시각 인코더(foundation visual encoders)와의 평가를 통해 우리의 접근 방식을 검증하였습니다.

- **Performance Highlights**: 다양한 실험을 통해 다중 클래스 탐지가 가능하고 이전 방법들보다 적은 매개변수를 사용하면서도 경쟁력 있는 성능을 달성함을 보여주었습니다. 특히, 새로운 DINOv3 모델에 대한 평가 결과는 이 기술의 효용성 및 상승된 성능을 입증합니다. 이 연구는 기초 특징(foundation features)에 대한 새로운 관점을 제시하며 Few-shot anomaly detection 분야의 발전을 이끌 것입니다.



### Automated Defect Detection for Mass-Produced Electronic Components Based on YOLO Object Detection Models (https://arxiv.org/abs/2510.01914)
Comments:
          12 pages, 16 figures, 7 tables, and published in IEEE Sensors Journal

- **What's New**: 본 논문에서는 산업에서 널리 사용되는 이중 인라인 패키지(DIP)의 자동 결함 탐지 시스템을 제안합니다. 이 시스템은 디지털 카메라 광학과 딥 러닝(Deep Learning) 기반 모델을 사용하여 작동하며, 전통적인 수작업 검수에 드는 시간과 노력을 줄이는 것을 목표로 합니다. 또한, 결함 데이터 부족 문제를 해결하기 위해 ConSinGAN을 이용해 학습과 테스트에 적합한 데이터셋을 생성합니다.

- **Technical Details**: 이 시스템은 제어 시스템, 이미징 장비 및 기계 장비의 세 부분으로 구성되어 있습니다. 제어 시스템은 개인 컴퓨터(PC)와 프로그래머블 로직 컨트롤러(PLC)를 포함하여 이미징 장비와 상호 작용하고 SCADA를 통해 딥 러닝 모델과 데이터 분석을 통합합니다. 이미징 장비는 산업용 카메라와 다양한 광원 장비로 구성되어 있으며, 기계 장비는 검사의 자동화를 위해 다양한 기능을 수행합니다.

- **Performance Highlights**: 제안된 YOLOv7 모델은 ConSinGAN과 함께 사용할 때 정확도 95.50%, 탐지 시간 285ms를 기록하였습니다. 이 결과는 기존의 임계값 기반 방법보다 경쟁력 있는 성과를 보여주며, 자동화된 결함 검출 시스템의 실효성을 높입니다. 또한, 관련 생산 라인과 SCADA 인터페이스를 개발하여 실제 응용 프로그램에서의 성능을 검증하였습니다.



### Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinemen (https://arxiv.org/abs/2510.01910)
Comments:
          14 pages

- **What's New**: 이 논문은 GNN(Graph Neural Networks)과 LLM(Large Language Models)의 통합을 다룹니다. 구체적으로는 RoGRAD라는 새로운 프레임워크를 소개하며, 이는 Retrieval-Augmented Generation 기법을 통해 데이터의 약점을 보완하는 이터레이티브(Iterative)한 접근 방식을 제공합니다. 또한 LLM을 사용한 기존의 방법들과 GNN 기반 접근 방식의 비교를 통해 LLM이 항상 더 우수하다는 기존 가정을 재검토합니다.

- **Technical Details**: RoGRAD는 기존의 LLM-enhanced 메서드와 달리, Retrieval-Augmented Contrastive Refinement를 기반으로 합니다. 이 방식은 다양한 데이터를 동적으로 재조정하여 GNN의 성능을 향상시키는 데 집중합니다. 추가적으로, R2CL(Contrastive Learning with RAG Refinement)을 도입하여 레이블 일관성과 클래스 간 변별력을 강화합니다. 실험을 통해 RoGRAD가 전통적인 GNN 및 LLM 강화 기법에 비해 82.43% 평균 성능 증가를 달성했다는 성과를 보여줍니다.

- **Performance Highlights**: RoGRAD는 기존 GNN에서 발생하는 데이터의 약점에도 불구하고 성능을 일관되게 개선하는 데 성공하였습니다. 이 연구는 LLM이 간단한 GNN보다 항상 더 나은 성능을 보이지 않는다는 점을 발견했으며, LLM과 GNN의 통합을 통해 더욱 강력한 방법론을 제시합니다. 광범위한 실험 결과는 RoGRAD가 GNN의 강인성을 높이고 실제 적용에 적합한 새로운 솔루션을 제공하는 것을 보여줍니다.



### Multimodal Foundation Models for Early Disease Detection (https://arxiv.org/abs/2510.01899)
Comments:
          6 pages

- **What's New**: 이 연구는 다양한 환자 데이터를 통합하는 다중 모달 기초 모델(multimodal foundation model)을 제안합니다. 기존의 전통적인 진단 모델들은 서로 다른 데이터 소스를 개별적으로 분석하여 조기 질병 진단을 위한 필수 교차 모달 상관관계를 찾아내는 데 제한이 있었습니다. 본 모델은 어텐션 기반(transformer) 프레임워크를 사용하여 이러한 데이터를 통합하고, 여러 각도에서 질병에 대한 예측 능력을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 전자 건강 기록(EHR), 의료 이미징, 유전자 데이터 및 착용 가능 센서 데이터를 조합하는 다중 모달(transformer-based) 아키텍처로 설계되었습니다. 각 데이터 모달리티는 전용 인코더를 통해 공유 잠재 공간(latent space)으로 변환되고, 그 후 멀티 헤드 어텐션(multi-head attention)과 잔여 정규화(residual normalization)를 통해 결합됩니다. 이 구조는 사전 훈련(pretraining) 가능성을 제공하여 여러 임상 작업에 쉽게 적응할 수 있도록 합니다.

- **Performance Highlights**: 실험 전략은 종양학, 심장학 및 신경학 분야의 벤치마크 데이터셋을 사용하여 조기 탐지 작업을 검증합니다. 또한, 투명성, 신뢰성 및 임상 해석 가능성을 개선하는 데이터 거버넌스(data governance) 및 모델 관리 도구를 포함하여 기술적 성과를 지원합니다. 이 방법론은 정밀 진단을 위한 단일 기초 모델 구축을 목표로 하며, 정확한 예측과 의사 결정을 돕는 데 기여할 것으로 기대됩니다.



### HRTFformer: A Spatially-Aware Transformer for Personalized HRTF Upsampling in Immersive Audio Rendering (https://arxiv.org/abs/2510.01891)
Comments:
          10 pages and 5 figures

- **What's New**: HRTFformer라는 새로운 트랜스포머 기반의 아키텍처가 소개되었습니다. 이 모델은 HRTF(updating Head-Related Transfer Functions) 상향 샘플링을 위해 설계되었으며, spatial correlation을 효과적으로 캡처하는 attention 메커니즘을 활용합니다. 연구는 기존의 머신러닝 접근 방식이 직면했던 장거리 spatial consistency 및 고속 sampling factor에서의 일반화 어려움을 해결하고자 합니다.

- **Technical Details**: HRTFformer는 spherical harmonic(SH) 도메인에서 작업하면서 희소한 입력 측정값으로부터 고해상도 HRTF를 재구성하도록 학습합니다. 모델은 magnitude smoothness를 촉진하는 neighbor dissimilarity loss를 도입하여 spatial coherence을 개선합니다. 이로 인해 더 사실적이면서 개인화된 HRTF를 생성하는 성능이 향상됩니다.

- **Performance Highlights**: HRTFformer는 신뢰할 수 있는 perceptual localization 모델과 객체 스펙트럼 왜곡 지표를 사용하여 평가됩니다. 실험 결과, HRTFformer는 기존의 선도적인 방법들에 비해 사실적이고 고품질의 HRTF를 생성하는 데 있어 상당한 차이를 보임을 입증합니다. 따라서, 고해상도 HRTF 재구성에 관한 연구에서 최신 상태의 결과를 달성하였습니다.



### Small is Sufficient: Reducing the World AI Energy Consumption Through Model Selection (https://arxiv.org/abs/2510.01889)
- **What's New**: AI의 에너지 소비와 탄소 발자국이 우려되는 가운데, 새로운 green AI 접근 방식이 부각되고 있습니다. 이는 'greater is better'에서 'small is sufficient'으로의 전환을 나타내며, 더 작은 효율적인 모델을 통해 에너지 절약을 강조하고 있습니다. 본 논문에서는 AI 모델 선택이 에너지 소비를 줄일 수 있는 방법에 대해 다루고 있습니다.

- **Technical Details**: 모델 선택은 주어진 작업에 가장 적합한 모델을 선택하는 프로세스로, 새로운 하드웨어나 복잡한 아키텍처가 필요하지 않아 쉽게 적용할 수 있습니다. 예를 들어, 이미지 분류 작업에는 수천 개의 모델이 존재하며 각 모델은 성능, 크기, 에너지 소비 간의 trade-off가 있습니다. 이 연구에서는 AI 작업의 인기도, 모델 크기 및 효율성을 시스템적으로 분석하였고, 다양한 작업에서 1%에서 최대 98%까지의 에너지 절약을 발견했습니다.

- **Performance Highlights**: 모델 선택을 통해 AI 에너지 소비를 27.8%까지 줄일 수 있으며, 이는 2025년 기준으로 전 세계에서 31.9 TWh의 에너지를 절약하는 것과 같습니다. 이는 다섯 개의 원자력 발전소의 연간 생산량에 해당합니다. 따라서 에너지 효율적인 모델 선택을 통한 AI의 지속 가능성 강화를 강조합니다.



### FINCH: Financial Intelligence using Natural language for Contextualized SQL Handling (https://arxiv.org/abs/2510.01887)
- **What's New**: 이 논문에서는 자연어 질문을 SQL 쿼리로 변환하는 Text-to-SQL 작업이 금융 분야에 특히 어렵다는 점을 강조합니다. 이에 대한 해결책으로 292개의 테이블과 75,725개의 자연어-SQL 쌍으로 구성된 새로운 금융 데이터셋인 FINCH를 소개합니다. 이 데이터셋은 모델의 세밀한 튜닝과 엄밀한 평가를 가능하게 합니다.

- **Technical Details**: FINCH 데이터셋은 소매, 은행 거래, 대출, 보험 등 다양한 금융 도메인을 포함하며, 총 33개의 데이터베이스를 아우릅니다. 각 SQL 쿼리는 SQLite와의 호환성을 위해 정규화되었으며, 이는 오픈 소스 커뮤니티에서도 사용이 용이하도록 합니다. 또한, 여러 모델을 평가하여 각 모델의 강점과 단점을 체계적으로 분석하였습니다.

- **Performance Highlights**: 모델 벤치마킹 결과, GPT-OSS-120B가 전반적인 성능에서 가장 우수한 성과를 보였으며, Qwen3-235B-A22B보다도 뛰어난 결과를 나타냈습니다. 이 외에도 다양한 규모의 LLM과 추론 중심 모델들을 비교하여 금융용 Text-to-SQL 작업에서의 성능 차이를 명확하게 드러냈습니다. 마지막으로, 논문에서 제안한 FINCH Score라는 새로운 평가지표는 기존의 방식이 간과했던 미세한 차이를 반영하여 모델 성능을 보다 충실하게 평가할 수 있도록 합니다.



### REPAIR: Robust Editing via Progressive Adaptive Intervention and Reintegration (https://arxiv.org/abs/2510.01879)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 고비용 지식 업데이트와 재훈련으로 인한 부작용을 완화하기 위해 REPAIR(강력한 편집을 위한 점진적 적응 개입 및 재통합)을 제안합니다. REPAIR는 정밀하고 저비용의 모델 업데이트를 지원하도록 설계된 생애 주기 편집 프레임워크입니다. 이 프레임워크는 폐쇄 루프 피드백 메커니즘과 동적 메모리 관리 기능을 결합하여 대규모 연속 편집의 불안정성과 충돌을 완화하는 데 기여합니다.

- **Technical Details**: REPAIR는 (1) 동적 메모리 관리를 통한 폐쇄 루프 오류 피드백, (2) 샘플 유사성에 기반한 배치 재조합 및 내부 배치 지식 증류, (3) 손실 인식 가중치 지식 통합을 포함한 혁신적인 모델 편집 방법론을 가지고 있습니다. 이러한 방식은 모델 학습의 전반적인 신뢰성, 범위 및 지역성을 최적화하는 목표를 달성합니다. 모델 업데이트는 고유의 하이퍼파라미터 조정으로 다양한 차원에서 적용됩니다.

- **Performance Highlights**: REPAIR는 다양한 모델(LLaMA-3, Qwen-2.5, DeepSeek-R1-1.5B 및 GPT-2-XL 등)에서 15%-20%의 전반적인 편집 성능 개선을 입증하며, 기존 최첨단 방법들과의 비교에서 일관되고 강력한 일반화를 보였습니다. REPAIR의 구조적 혁신은 지식 중복과 손실을 대폭 낮춰주며, 이는 레이아웃 내의 편집 문제 해결에서 중요한 의미를 갖습니다.



### TACOS: Task Agnostic COordinator of a multi-drone System (https://arxiv.org/abs/2510.01869)
Comments:
          6 pages, 6 figures, accepted as poster at 2025 IEEE International Symposium on Multi-Robot & Multi-Agent Systems

- **What's New**: 이번 논문에서는 TACOS(Task-Agnostic COordinator of a multi-drone System)라는 프레임워크를 소개합니다. TACOS는 대형 언어 모델(LLMs)을 사용하여 다수의 UAV 시스템을 높은 수준의 자연어로 제어할 수 있게 해줍니다. 이 시스템은 사용자와의 직관적인 상호작용을 위한 자연어 인터페이스, 사용자 의도를 구조적인 작업 계획으로 변환하는 지능형 조정자, 실제 세계와 상호작용하는 자율 에이전트를 통합하여 유연한 작업 수행을 가능하게 합니다.

- **Technical Details**: TACOS 프레임워크는 두 개의 주요 LLM으로 구성됩니다. 첫 번째는 고수준의 자연어 명령을 받아 작업 계획을 합성하는 Coordinator LLM이며, 두 번째는 이 계획을 실시간으로 조정하여 실행하는 Supervisor LLM입니다. 시스템은 quadrotor UAV를 사용하여 3D 환경 내에서 동작하며, 사전 정의된 저수준 제어 원시를 사용합니다. 각 LLM은 특정 목표, 행동 제약 및 출력 구조를 정의한 구성 프롬프트로 초기화됩니다.

- **Performance Highlights**: 조정자는 사용자의 자연어 명령을 해석하여 구조화된 작업 계획을 생성하며, 이를 통해 사용자와 다수의 UAV 간의 상호작용을 지원합니다. 실험 결과 TACOS는 사용자의 고수준 지시를 해석하고, 저수준 UAV 동작을 효율적으로 실행하는 성과를 보여주었습니다. 이 시스템은 예측 불가능한 환경에서도 보다 유연하고 회복력 있는 군집 임무 실행을 가능하게 하여 다수 UAV 시스템의 새로운 방향을 제시합니다.



### A Modular Theory of Subjective Consciousness for Natural and Artificial Minds (https://arxiv.org/abs/2510.01864)
Comments:
          41 pages, 3 figures. Under review, comments welcome

- **What's New**: 본 논문은 Modular Consciousness Theory (MCT)를 제안하여 주관적 경험의 발생을 computationally explicit한 방식으로 설명합니다. MCT는 Integrated Informational States (IISs)라는 분리된 정보를 정리한 패킷들을 통해 의식을 이해합니다. 각 IIS는 정보 밀도의 벡터로 태그되어 있으며, 이는 주관적 강도를 나타냅니다. 따라서 MCT는 수명과 결정에 미치는 정보의 영향을 설명하기 위한 새로운 관점을 제공합니다.

- **Technical Details**: MCT는 의식을 정보를 처리하는 여러 모듈의 통합으로 설명합니다. 각각의 모듈은 필터링, 추상화, 평가, 자기 평가와 같은 구체적인 기능을 수행하며, 이러한 모듈은 공간적으로는 국소화되어 있지 않습니다. IIS는 시간적으로 제한된 통합된 정보 상태로, 메모리 인코딩과 행동 조절의 중심 허브 역할을 합니다. 이 모델은 높은 진폭의 정보 밀도 벡터가 기억에 저장되는 가능성을 높인다고 가정합니다.

- **Performance Highlights**: MCT는 구체적인 정보 처리 아키텍처를 제공하여 인간과 유사한 인공지능 시스템의 개발에 기여할 수 있는 기반을 마련합니다. 또한, 메모리 인코딩과 행동 결정 및 역사적 연속성을 증가시키는 정보 밀도 태그 신호와 주관성 간의 상관관계를 제시합니다. 이 이론은 강한 정보 밀도의 상태가 장기 기억 및 행동에 더 큰 영향을 미친다는 점에서 임상적인 예측과 정신 장애의 모듈 통합의 교란을 설명하는 데 유용합니다.



### NGGAN: Noise Generation GAN Based on the Practical Measurement Dataset for Narrowband Powerline Communications (https://arxiv.org/abs/2510.01850)
Comments:
          16 pages, 15 figures, 11 tables, and published in IEEE Transactions on Instrumentation and Measurement, Vol. 74, 2025

- **What's New**: 본 논문에서는 좁은 대역 전력선 통신(NB-PLC) 시스템의 비주기적 비동기 충격 소음(nonperiodic asynchronous impulsive noise) 처리를 개선하기 위해, 복잡한 특성을 학습하는 노이즈-생성 적대 신경망(Generative Adversarial Network, GAN)을 제안합니다. 기존 노이즈 모델들이 일부 특성만을 포착하는 한계를 극복하기 위해, 우리는 실제로 측정된 노이즈 샘플을 사용하여 데이터 증강(data augmentation)을 수행합니다. 특히, 제안된 노이즈-생성 GAN (NGGAN)은 측정된 노이즈 통계에 밀접하게 일치하는 방식으로 설계되었습니다.

- **Technical Details**: 이 연구에서 제안된 NGGAN은 입력 신호의 길이를 조절하여 사이클로 스테이셔너리(cyclo-stationary) 노이즈 생성을 용이하게 하며, Wasserstein 거리(Wasserstein distance)를 손실 함수로 사용하여 생성된 노이즈와 훈련 데이터셋 간의 유사성을 강화합니다. 우리는 PSCRGM, FRESH 필터, 실제 NB-PLC 시스템에서 측정된 값들을 포함한 훈련 데이터셋을 사용하여 GAN 모델의 유사성 성능을 분석합니다. 결과적으로, NGGAN은 기존의 수학적 노이즈 모델로는 잡을 수 없는 복잡한 노이즈 통계의 샘플을 생성할 수 있는 효율적인 데이터 증강 방법으로 확인되었습니다.

- **Performance Highlights**: 시뮬레이션 결과, NGGAN은 기존 GAN 기반 모델들에 비해 노이즈에 대한 강건성을 더욱 개선하는 것으로 나타났습니다. 성능 메트릭에는 최대값, 평균값, 에너지 값, 표준 편차, 왜도, 첨도, 고유치(peak) 수 등이 포함되어 있으며, 이러한 메트릭들을 통해 생성된 노이즈의 품질을 평가했습니다. 또한, GitHub를 통해 Python 소스 코드를 제공하며, 더 나아가 소비자 전자 기기에서의 적용 가능성도 탐색하고 있습니다.



### Pre-Hoc Predictions in AutoML: Leveraging LLMs to Enhance Model Selection and Benchmarking for Tabular datasets (https://arxiv.org/abs/2510.01842)
Comments:
          Oral Presentations ADAPT Annual Scientific Conference 2025

- **What's New**: 이 논문은 AutoML과 pre-hoc 모델 선택의 교차점을 탐구하여 전통적인 모델과 Large Language Model (LLM) 에이전트를 활용해 AutoML 라이브러리의 탐색 공간을 줄이는 방법을 제안합니다. 특히, 데이터셋 설명과 통계 정보를 활용하여 AutoML의 검색 공간을 감소시키는 새로운 방법론을 제시합니다. AutoGluon 포트폴리오 데이터셋을 통해 실험하여 대규모 계산 비용을 줄이면서도 주어진 데이터셋에 최적의 모델을 선택할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 논문의 핵심 방법론은 데이터셋의 구조적 통계 정보와 텍스트 설명 정보를 통합하여 효율적이고 정확한 pre-hoc 모델 선택을 가능하게 만드는 것입니다. LLMs를 활용한 AutoML 에이전트를 개발하여 탭형 AutoML 문제를 해결하는 파이프라인을 구축하였으며, 모델 선택의 정당성을 설명할 수 있는 기능도 추가되었습니다. 실험은 OpenML의 175개 데이터셋을 대상으로 진행되었으며, 전통적인 Pre-Hoc Predictor 방법과 LLM Pre-HP를 비교하며 성능을 평가했습니다.

- **Performance Highlights**: 여러 실험 결과, 전통적인 Pre-HP 모델이 자동화된 모델 선택에 있어서 Baseline 대비 개선된 성능을 보였습니다. 특히, RoBERTa 모델은 패밀리 정확도에서 0.61을 기록하며 기대 이상의 결과를 보여주었습니다. 그러나 LLM을 포함한 AutoML 에이전트는 여전히 전통적인 모델에 비해 성능이 낮았지만 Baseline 1은 초과하였고, Zero-Shot 방식의 Llama 모델이 전반적인 성능 지표에서 최상의 성능을 보였습니다.



### SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessmen (https://arxiv.org/abs/2510.01812)
Comments:
          4 pages, 5 figures; submitted to ICASSP 2026

- **What's New**: 이번 연구에서는 Singing quality assessment (SQA)를 위한 자동 평가 데이터셋인 SingMOS-Pro를 소개합니다. 기존의 SingMOS 버전을 기반으로 하여, 추가적인 주석을 통해 가사, 멜로디, 전반적인 품질을 포함하여 더 넓은 범위와 더 다양한 데이터를 제공합니다. 총 7,981개의 노래 클립이 포함되어 있으며, 이는 12개의 데이터셋에서 41개의 모델에 의해 생성되었습니다.

- **Technical Details**: SingMOS-Pro 데이터셋은 3,425개의 SVS 클립, 1,307개의 SVC 클립, 2,671개의 SVR 클립 및 578개의 실제 샘플을 포함하여 총 11.15시간의 오디오를 제공합니다. 각 클립은 최소 5명의 전문 주석가에 의해 평가되며, 가사 점수와 멜로디 점수의 추가 주석이 포함되어 있습니다. 이를 통해 각 클립의 발음 명확성과 멜로디 자연스러움을 측정할 수 있습니다.

- **Performance Highlights**: 연구자들은 SingMOS-Pro에서 여러 평가 방법을 벤치마킹하면서 강력한 기준선(baseline)을 설정하였습니다. SVR이 일반적으로 SVC를 초과하는 성능을 보이며, SVC가 SVS보다 쉽게 모델링되는 경향이 있습니다. 결과적으로, SingMOS-Pro는 SQA의 향후 연구에 유용한 실질적인 참조 자료로 자리잡을 것으로 기대됩니다.



### Rethinking the shape convention of an MLP (https://arxiv.org/abs/2510.01796)
- **What's New**: 이 논문에서는 전통적인 narrow-wide-narrow MLP 디자인에 도전하여 wide-narrow-wide (Hourglass) MLP 블록을 제안합니다. 이 새로운 구조에서는 skip connections가 확장된 차원에서 작동하며, 잔여 계산은 좁은 병목을 통해 흐릅니다. 이러한 전환은 높은 차원 공간에서 점진적인 개선을 가능하게 하며, 매개변수에 맞춘 설계를 통해 계산 효율성을 유지합니다.

- **Technical Details**: Hourglass MLP는 입력 신호를 확장된 차원으로 올리기 위한 초기 프로젝션을 요구하며, 이 프로젝션은 훈련 중에 임의 초기화 상태를 고정할 수 있습니다. 또한, 이 구조는 기존의 narrow-wide-narrow 블록과 비교하여 건축적 검색을 통해 성능-매개변수 Pareto 전선에서 일관된 우수성을 나타냅니다. 특정 실험에서는 매개변수 예산이 증가함에 따라 더 깊은 네트워크와 넓은 skip connections 및 더 좁은 병목을 선호하는 패턴이 발견되었습니다.

- **Performance Highlights**: 토론된 결과들은 wide-narrow-wide 디자인이 기존의 MLP 설계와 비교하여 일관되게 우수한 Pareto 전선을 달성함을 입증합니다. 특히, 입력 프로젝션 레이어의 추가 매개변수를 고려하더라도, Hourglass 아키텍처는 전통적인 디자인에 비해 지속적으로 우수한 성능을 보여줍니다. 이러한 통찰력은 Transformer와 기타 잔여 네트워크를 포함한 다른 아키텍처로 확장될 가능성을 제시합니다.



### Nav-EE: Navigation-Guided Early Exiting for Efficient Vision-Language Models in Autonomous Driving (https://arxiv.org/abs/2510.01795)
- **What's New**: 비전-언어 모델(Vision-Language Models, VLMs)의 자율주행 차량에 대한 통합이 변화하는 경향으로 부상하고 있습니다. 최근 연구에 따르면 VLMs는 내비게이션 시스템을 통해 시각적 인식과 고수준의 추론을 통합하여 운전 장면에 대한 풍부한 의미적 추론을 가능하게 합니다. 그러나 높은 추론 지연(lag)은 실시간 배치를 어렵게 하고 있으며, 이를 해결하기 위해 내비게이션에 기반한 조기 종료(Early Exit) 프레임워크인 Nav-EE를 제안합니다.

- **Technical Details**: Nav-EE는 사전 계산된 작업별 종료 계층을 기반으로 하여 내비게이션의 맥락에 따라 동적으로 조정되는 방식으로 작동합니다. 시스템은 오프라인 프로파일링 단계에서 각 시나리오에 대해 가장 이른 유효 종료 계층을 식별하고, 온라인 추론 단계에서 내비게이션 유도 장면에 따라 종료 조정을 수행합니다. 마지막으로 실시간 차량에 통합된 Nav-EE를 통해 실제 운전 조건에서도 상황 인식 가능한 조기 종료 기능을 구현합니다.

- **Performance Highlights**: Nav-EE는 Waymo, CODA, Bosch 데이터셋에서 실험한 결과, 완전 추론과 유사한 정확도를 유지하며 최대 63.9%의 지연을 줄이는 효과를 보였습니다. 실제 자율주행 차량에서 Autoware.Universe와 통합하여 600ms에서 300ms로 추론 지연을 줄여 복잡한 시나리오에서 더 빠른 의사 결정을 가능하게 했습니다. 이러한 결과는 내비게이션 선견지명과 조기 종료의 결합이 자율 시스템에서 효과적인 대안이 될 수 있음을 제시합니다.



### Comparison of Unsupervised Metrics for Evaluating Judicial Decision Extraction (https://arxiv.org/abs/2510.01792)
Comments:
          28 pages

- **What's New**: 이번 연구는 러시아의 1,000건의 익명 법원 판결문에서 7개의 의미적 블록을 추출하는데 필요한 16개의 비지도 학습 메트릭을 평가합니다. 이러한 메트릭은 문서 기반, 의미적, 구조적, 법률 특정 카테고리로 분류되며, 사전 주석이 없는 상태에서 작동합니다. 연구 결과, Term Frequency Coherence와 Coverage Ratio/Block Completeness가 전문가 평점과 가장 잘 일치하는 것으로 나타났습니다.

- **Technical Details**: 연구 방법론에서는 비지도 메트릭을 사용하여 법원 판결문 추출 품질을 평가하며, 이는 JSON 형식으로 구성된 문서에서 수행됩니다. 각 판결문은 원본 텍스트와 사전 세분화된 JSON 객체 형태로 제공되어, 비지도 메트릭을 계산하는 기준(reference extraction) 역할을 합니다. 평가에 참여한 법률 전문가들은 1-5 Likert 척도를 사용해 각 블록을 평가하였으며, 서로 간의 신뢰성 평가를 위해 ICC를 사용했습니다.

- **Performance Highlights**: 연구는 법률 텍스트 추출을 위한 비지도 평가 메트릭이 아직까지 개발이 미흡하고 법률 특유의 뉘앙스를 포착하지 못할 가능성이 있음을 강조합니다. 전문가 평가 결과, Court evaluation of evidence 블록에서는 ICC 값이 0.86으로 높은 일치를 보였지만, Court decision 블록에서는 0.70으로 다소 낮은 수치를 기록했습니다. 이러한 결과는 법적 맥락에서 기술이 사람의 판단을 완전히 대체할 수는 없음을 시사합니다.



### Pack and Force Your Memory: Long-form and Consistent Video Generation (https://arxiv.org/abs/2510.01784)
- **What's New**: 이 논문은 장시간 비디오 생성에서의 두 가지 주요 문제인 장거리 의존성(Long-range dependency) 유지와 오류 누적(Error accumulation) 방지를 해결하기 위해 두 가지 기여를 제안합니다. 그 첫 번째로, MemoryPack이라는 학습 가능한 컨텍스트 검색 메커니즘을 도입하여 텍스트 및 이미지 정보를 글로벌 가이드로 활용하여 단기 및 장기 의존성을 공동으로 모델링합니다. 두 번째로, 오류 누축을 완화하기 위한 Direct Forcing 전략을 제안하며, 이는 학습과 추론 간의 정렬을 개선하여 오류 전파를 줄이는 데 기여합니다.

- **Technical Details**: MemoryPack은 비디오 길이에 비례하여 스케일링이 용이하며, 선형적 복잡도를 유지하면서도 계산 효율성을 보장합니다. 이 메커니즘은 장기 비디오 콘텐츠의 의미적 정합성을 강화해주고, 동시에 인접 프레임을 단기적인 단서를 활용하여 동작과 포즈의 충실도를 향상시킵니다. 또한, Direct Forcing은 판별 흐름(rectified flow)의 원리를 활용하여 단일 단계에서 예측된 벡터 필드를 기반으로 역 ODE(computation)를 수행하여 추론 결과를 근사합니다.

- **Performance Highlights**: 이 방법은 VBench에서 Motion Smoothness, Background Consistency, Subject Consistency와 같은 주요 메트릭에서 현재 최고의 성능을 달성합니다. 실험 결과, MemoryPack과 Direct Forcing은 장기적 컨텍스트 정보를 효과적으로 모델링하고 높은 일관성을 달성하는 데 기여함을 보여줍니다. 이 연구는 자율 회귀 영상 모델의 실용성을 한층 더 향상시키는 데 기여합니다.



### Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks (https://arxiv.org/abs/2510.01782)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)이 자기가 알지 못하는 질문에 대해 거부할 수 있는 능력인 지식 기반 거부(Knowledge-aware refusal)를 정의하고 이를 측정하기 위한 새로운 메트릭인 거부지수(Refusal Index, RI)를 제안합니다. 기존의 메트릭이 이러한 능력을 제대로 측정하지 못하는 문제를 해결하기 위해, RI는 거부 확률과 오류 확률 간의 Spearman 순위 상관관계를 통해 정의됩니다.  이 방식은 모델의 실제 거부 행동을 직접적으로 측정할 수 있게 하여, 과거의 불완전한 메트릭과 비교하여 신뢰성을 높입니다.

- **Technical Details**: RI는 지식 기반 거부를 평가하기 위해 두 가지 주요 특성을 지닙니다. 첫째, RI는 지식 기반 거부를 정확하게 추정하며, 거부율에 무관하게 공정한 측정을 제공합니다. 둘째, RI는 경량 평가 방법을 통해 기존 평가 파이프라인과 호환되어 쉽게 측정할 수 있습니다. 이 방법은 두 번의 표준 평가 과정을 통해 모델의 거부 능력을 정량화하며, 16개 모델과 5개의 데이터셋을 통해 광범위한 실험을 수행하여 RI의 효과성을 검증합니다.

- **Performance Highlights**: 실험 결과 RI는 모델이 알지 못하는 질문에 대한 거부 능력을 정확하게 정량화하며, 다양한 거부율에서도 안정성을 유지합니다. RI는 모델의 전반적인 정확도 및 거부율에 의존하지 않고 일관된 모델 순위를 제공합니다. 또한 RI는 전통적인 정확성 메트릭과 비교하여 LLM의 사실성(factuality) 평가에서 간과된 중요한 차이를 밝혀내며, 모델의 신뢰도를 평가하기 위해 지식 기반 거부 측정을 포함할 필요성을 강조합니다.



### Secure Multi-Modal Data Fusion in Federated Digital Health Systems via MCP (https://arxiv.org/abs/2510.01780)
Comments:
          6 pages, 8 figures, 7 equations, 1 algorithm

- **What's New**: 이 연구는 이질적인 의료 데이터를 성과할 수 있는 안전하고 상호 운용 가능한 시스템을 구축하는 새로운 프레임워크를 소개합니다. 제안된 아키텍처는 세 가지 핵심 요소를 통합하는데, 이는 다중 모달 특성 정렬, 환자의 민감한 데이터를 보호하기 위한 안전한 집계 및 에너지 소비를 고려한 스케줄링입니다. 이를 통해 다양한 데이터 소스 간의 표준화된 상호 작용을 가능하게 하여, 다음 세대의 연합 건강 인프라를 위한 신뢰할 수 있는 경로를 제공합니다.

- **Technical Details**: 제안된 프레임워크는 Model Context Protocol (MCP)를 채택하여 안전한 다중 모달 연합 융합을 위한 새로운 방법론을 제시합니다. 이 방법론은 다양한 클라이언트에서 발생하는 업데이트를 통합하는 동시에 개인 정보 보호를 위해 미세 조정된 잡음 주입과 안전한 집계를 활용하여 민감한 데이터를 보호합니다. 또한, 에너지 효율적인 클라이언트 스케줄링 메커니즘이 포함되어 있어 모바일 헬스케어 클라이언트에서의 드롭아웃 비율을 줄이고 안정적인 참여를 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존의 연합 학습에 비해 진단 정확도를 최대 9.8% 향상시키고 클라이언트 드롭아웃 비율을 54% 감소시키는 성과를 보였습니다. 또한, 개인 정보 보호와 효용 사이의 균형을 유지하며 임상적으로 수용 가능 범위 내에서 유용성을 지속할 수 있음을 입증했습니다. 이러한 결과는 MCP를 활용한 다중 모달 융합이 확장 가능하고 신뢰할 수 있는 의료 인프라로 나아가는 길이라는 것을 보여줍니다.



### Unsupervised Dynamic Feature Selection for Robust Latent Spaces in Vision Tasks (https://arxiv.org/abs/2510.01758)
- **What's New**: 이번 논문은 동적 특징 선택(Dynamic Feature Selection, DFS)을 활용하여 레텐트 표현(latent representation)을 향상시키기 위한 새로운 접근법인 동적 데이터 선택(Dynamic Data Selection, DDS)을 제시합니다. 기존의 DFS 방법은 레이블(label) 의존성이 있어 다양한 도메인에 적용하기 어려운 반면, DDS는 비지도 학습(unsupervised learning)에서 최초로 적용된 기법입니다. DDS는 이미지 데이터에서 불필요하거나 중복된 정보를 제거하며, 선택한 특징의 위치를 유지하여 복잡한 네트워크 구조에도 쉽게 적용할 수 있습니다.

- **Technical Details**: 기술적으로, DDS는 주어진 인스턴스에 대해 최대 M개의 특징을 선택하기 위한 최소화 문제를 해결합니다. DDS 네트워크는 두 개의 주요 구성 요소로 나뉘어 있으며, 각 구성 요소는 모델을 훈련하는 데 도움을 줍니다. 이 방법은 비지도 손실 함수(unsupervised loss function)를 사용하여 입력 데이터의 중요성에 기반한 선택 과정을 수행합니다. 기본적으로, DDS는 입력 데이터를 마스킹하여 가장 관련성 높은 특징만을 오토인코더(autoencoder)에 전달하게 됩니다.

- **Performance Highlights**: 실험 결과는 DDS가 클러스터링(clustering) 및 표현 학습(representation learning) 작업에서 일반화 성능을 크게 향상시킨다는 것을 보여줍니다. DDS를 채택한 모델은 다양한 이미지 데이터셋에서 성능 개선을 이루며, 계산 비용의 증가는 최소화되었습니다. 이 연구는 비지도 문제 해결을 위한 DDS의 유용성과 용이성을 강조하며, 다양한 문제에 쉽게 적용 가능하다는 점도 부각됩니다.



### Machine-interpretable Engineering Design Standards for Valve Specification (https://arxiv.org/abs/2510.01736)
Comments:
          22 pages, 10 figures, 4 tables

- **What's New**: 이번 논문에서는 엔지니어링 설계 표준에 저장된 정보를 모듈형 재사용 가능하고 기계 해석 가능한 온톨로지(ontology)로 변환하는 방법을 보여줍니다. 이를 통해 품질 보증 과정에 활용할 수 있으며, 특히 밸브 선택 과정에 적용됩니다. 기존의 문서형 표준이 아닌 데이터 중심으로 접근하는 혁신적인 시도를 제시합니다.

- **Technical Details**: 논문은 API, ASME, ASTM의 엔지니어링 설계 표준에 중점을 두고, 기계 해석 가능한 데이터 생성을 통해 밸브 설계 및 제품 사양 자동화를 목표로 합니다. ASME B16.34와 같은 표준의 데이터와 규칙을 구조화하여, W3C 준수 형식으로 교환 가능한 모듈형 온톨로지를 생성합니다. 이러한 접근 방식을 통해 표준의 디지털화가 가능해지며, 국제 표준에 부합하는 동작을 보장합니다.

- **Performance Highlights**: 연구를 통해 생성된 온톨로지는 밸브 선택 과정에서 사용되어, 안전하고 효율적인 설계 품질 보증을 자동화합니다. 또한, 공유 가능한 IDO 기반의 모듈형 온톨로지는 설계 표준에 대한 의미론적 추론을 가능하게 하며, 스마트 기준으로의 전환을 추구하는 표준 기관에 유용성을 보여줍니다. 이를 통해 엔지니어링 디자인 프로세스의 효율성과 품질을 크게 향상시킬 수 있습니다.



### Emotional Text-To-Speech Based on Mutual-Information-Guided Emotion-Timbre Disentanglemen (https://arxiv.org/abs/2510.01722)
Comments:
          In Proceedings of the 17th Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC 2025)

- **What's New**: 이번 연구에서는 기존의 Text-To-Speech(TTS) 시스템의 한계를 극복하기 위해 새로운 감정 기반 TTS 방법을 제안합니다. 제안된 방법은 단어의 감정을 세밀하게 반영하고, 타임브레(timbre) 정보로부터 독립적인 감정 임베딩을 이끌어내는 데 중점을 두고 있습니다. 연구진은 스타일 디스엔탱글먼트(style disentanglement) 기법을 활용하여 두 개의 특징 추출기를 안내하여 서로 다른 스타일 요소를 효과적으로 분리했습니다.

- **Technical Details**: 이 모델은 스타일을 포착하기 위한 전용 스타일 인코더를 포함하며, 이 인코더는 글로벌 타임브레 추출기와 감정 추출기로 구성되어 있습니다. 감정 추출기는 참조 음성과 목표 음소를 정렬하여 감정 임베딩 시퀀스를 생성하며, 상호 정보 최소화를 통해 타임브레와 감정 임베딩을 분리합니다. 이는 감정 표현의 세밀함을 보장하고 자연스러운 발화 생성을 가능하게 합니다.

- **Performance Highlights**: 실험에 따르면 제안된 방법은 기존 TTS 시스템인 Global Style Token, StyleSpeech, MIST, DC Comix TTS보다 우수한 결과를 나타냈습니다. 주관적 및 객관적인 지표 모두에서 우수성을 입증하였으며, t-SNE 시각화는 잘 분리된 감정 클러스터를 나타내어 효과적인 세분화(disentanglement)를 확인했습니다. 이러한 결과는 감정 TTS 시스템의 품질 및 유연성을 향상시킬 수 있는 분리된 세밀한 표현의 잠재력을 강조합니다.



### Latency-aware Multimodal Federated Learning over UAV Networks (https://arxiv.org/abs/2510.01717)
Comments:
          Accepted at IEEE Transactions on Network Science and Engineering

- **What's New**: 이번 논문은 무인 항공기(UAV)를 활용한 연합 다중 모달 학습(federated multimodal learning, FML)에 대해 조사하며, 시스템 지연 시간을 최소화하고 수렴 분석(convergence analysis)을 제공합니다. UAV는 네트워크에 분산되어 데이터를 수집하고 모델 학습에 참여하며, 기지국(base station, BS)과 협력하여 글로벌 모델을 구축합니다. 이 연구는 다중 모드 감지를 통해 단일 모드 시스템의 한계를 극복하고, 모델의 정확도와 일반화를 향상시키는 데 중점을 두었습니다.

- **Technical Details**: 제안된 FML 시스템은 UAV 네트워크에서 우선적으로 지연 시간을 최적화하는 것을 목표로 합니다. UAV의 감지 스케줄링(UAV sensing scheduling), 전력 제어(power control), 궤적 계획(trajectory planning), 자원 할당(resource allocation), BS 자원 관리(BS resource management)와 같은 다양한 요소를 통합하여 문제를 해결합니다. 복잡성을 해결하기 위해, 블록 좌표 하강법(block coordinate descent) 및 연속 볼록 근사(successive convex approximation) 기법을 결합한 효율적인 반복 최적화 알고리즘을 제안했습니다.

- **Performance Highlights**: 수치 실험을 통해, 제안된 FML 프레임워크가 기존 방법들에 비해 시스템 지연 시간(system latency)과 모델 학습 성능에서 우수함을 입증했습니다. 특히, 제안된 방법은 벤치마크 방법들에 비해 시스템 지연 시간을 42.49%까지 줄일 수 있음을 보여주었습니다. 또한, 독립적이며 동일하게 분포된(IID) 데이터와 비 IID 데이터 설정에서 모델 로스(model loss)와 정확도 수렴에서 뛰어난 성능을 보였습니다.



### PyramidStyler: Transformer-Based Neural Style Transfer with Pyramidal Positional Encoding and Reinforcement Learning (https://arxiv.org/abs/2510.01715)
- **What's New**: PyramidStyler는 기존 Neural Style Transfer(NST) 방식의 한계를 극복하기 위해 제안된 새로운 Transformer 기반의 프레임워크입니다. Pyramidal Positional Encoding(PPE)을 통해 복잡한 스타일과 고해상도 입력을 효과적으로 처리할 수 있도록 설계되었습니다. 이 구조는 지역 세부정보와 글로벌 맥락을 캡처하면서 계산적 부하를 줄이는 데 초점을 두고 있습니다.

- **Technical Details**: PPE는 여러 스케일에서 겹치는 패치를 구성하고, 각각을 다양한 커널 크기를 가진 CNN을 통해 인코딩한 뒤, 주의(attention) 메커니즘이나 연결(concatenation)을 통해 융합합니다. 이러한 계층적 설계는 세부사항과 넓은 공간적 관계를 보존하면서도 계산을 효율적으로 처리할 수 있게 합니다. 또한 경량화된 강화 학습(RL) 에이전트를 통합하여 스타일화 가중치를 동적으로 조정하여 수렴 속도를 높이고 시각적 품질을 개선합니다.

- **Performance Highlights**: PyramidStyler는 Microsoft COCO와 WikiArt 데이터셋에서 훈련되어 4000 에폭 후 콘텐츠 손실(content loss)을 62.6% 줄이며(2.07로), 스타일 손실(style loss)을 57.4% 감소시킵니다(0.86으로). RL을 사용할 경우 콘텐츠 손실(2.03) 및 스타일 손실(0.75)에서 추가 개선을 보여주고, 속도 손실은 최소(1.40s)로 유지됩니다. 이러한 결과는 실시간, 고품질의 예술적 렌더링을 가능하게 하여 미디어와 디자인 분야에서 폭넓은 응용 가능성을 시사합니다.



### PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization (https://arxiv.org/abs/2510.01708)
Comments:
          8 pages, 5 figures

- **What's New**: 이 논문에서는 PolySim이라는 새로운 전체 신체 제어(Human Whole Body Control, WBC) 훈련 플랫폼을 소개합니다. 이 플랫폼은 여러 이종 시뮬레이터를 통합하여 동시에 병렬 환경을 실행하며, 이렇게 함으로써 시뮬레이터의 유도 편향(simulator inductive bias)을 줄이는 것을 목표로 합니다. PolySim은 동적 수준의 도메인 임의화(domain randomization)를 실현하며, 기존 단일 시뮬레이터 훈련보다 더 일반화된 정책을 학습할 수 있게 합니다.

- **Technical Details**: PolySim은 훈련과 시뮬레이션 프로세스를 분리하여 다수의 시뮬레이터에서 병렬로 롤아웃을 수행하는 구조를 갖추고 있습니다. 이 시스템은 세 가지 주요 설계를 포함하는데, 훈련-시뮬레이션 격리(training-simulation isolation), 시뮬레이터 라우터(simulator router), 그리고 GPU 패스스루 통신(GPU pass-through communication)입니다. 이러한 설계를 통해 PolySim은 서로 다른 시뮬레이터의 동적 특성에 대한 정책 최적화를 가능하게 하여, 강화 학습( RL)에서의 유연성을 높입니다.

- **Performance Highlights**: 실험 결과, PolySim은 시뮬레이터 간 평가에서 모션 추적 오류를 대폭 줄이며, 특히 MuJoCo에서 IsaacSim 기본선 대비 52.8% 향상된 실행 성공률을 보여주었습니다. 또한, PolySim은 추가적인 미세 조정 없이 실제 Unitree G1 로봇에 제로샷 배포(zero-shot deployment)를 가능하게 하여, 시뮬레이션에서 실제 세계로의 효과적인 이전을 입증합니다. PolySim의 코드는 작업 수락 후 공개될 예정입니다.



### Representational Alignment Across Model Layers and Brain Regions with Hierarchical Optimal Transpor (https://arxiv.org/abs/2510.01706)
- **What's New**: 이 논문에서 제안하는 Hierarchical Optimal Transport (HOT) 방법론은 전통적인 layer-wise matching 방식의 한계를 극복하는 새로운 프레임워크입니다. 기존의 방법은 각 레이어를 개별적으로 매칭하여 비대칭적인 결과를 도출하는데 반해, HOT는 레이어 간의 부드럽고 전 세계적으로 일관된 연결을 추론하여 전체 네트워크 비교를 위한 하나의 정렬 점수를 제공합니다. 이 방식을 통해 다양한 심층 신경망 아키텍처 간의 비교가 가능해집니다.

- **Technical Details**: HOT는 노드 간의 유사성을 고려하는 최적 수송 이론을 기반으로 하여, 각 레이어는 다양한 타겟 레이어에 질량을 분배할 수 있습니다. 이를 통해 서로 다른 깊이의 신경망 간의 연결을 부드럽게 처리하며, 각 소스 레이어는 자신의 표현 정보를 잃지 않고 균형 잡힌 정렬을 이루어냅니다. 이 과정은 두 가지 계층적 수준에서 진행되며, 각 레이어 내의 뉴런을 정렬하고 전역적으로 레이어 간의 연계를 결정합니다.

- **Performance Highlights**: HOT는 비전 모델, 대형 언어 모델, 인간 시각 피질 데이터 등에 대해 평가되었으며, 기존의 표준 pairwise 방법론과 비교하여 정렬 품질에서 동등하거나 그 이상의 성능을 기록하였습니다. 특히, HOT는 깊이의 불일치를 자연스럽게 처리하고, 신경망의 학습 과정에서 발생하는 계층적 구조를 복원함으로써 더 정교하고 해석 가능한 비교를 가능하게 합니다.



### Holistic Order Prediction in Natural Scenes (https://arxiv.org/abs/2510.01704)
Comments:
          25 pages, 11 figures, 6 tables

- **What's New**: 본 논문에서는 InstaFormer라는 새로운 네트워크를 제안합니다. InstaFormer는 RGB 이미지 하나만으로 씬의 모든 인스턴스에 대한 전체 압출(i.e., occlusion) 및 깊이(depth) 순서를 단일 전방 패스를 통해 반환할 수 있습니다. 이는 기존의 시스템들이 비싼 입력 형식과 높은 추론 비용에 의존했던 문제를 해결하고, 인스턴스 간의 상호작용을 통해 이루어집니다.

- **Technical Details**: InstaFormer는 객체 쿼리와 잠재적인 마스크 설명자 간의 상호작용을 통해 인스턴스 간의 기하학적 관계를 예측합니다. 제안된 모델은 점진적 입력 요구를 완화하며, 기존의 방식을 뛰어넘어 단일 전방 패스에서 전체 인스턴스의 기하학적 관계를 예측할 수 있는 기능을 갖추고 있습니다. 이 네트워크는 거리 level의 압출 및 깊이 예측 작업을 인접 행렬 수준의 문제로 재구성합니다.

- **Performance Highlights**: 모델의 성능은 다양한 벤치마킹을 통해 검증되었으며, 기존의 최상위 모델들과 동등하거나 이를 초월하는 결과를 보였습니다. InstaFormer는 RGB 입력 이미지 하나로 압출 및 깊이 순서 예측 작업에서 뛰어난 성능을 나타냅니다. 또한, 논문에서 제시된 코드와 모델은 오픈 소스로 제공되어, 더 많은 연구자들이 활용할 수 있습니다.



### Format Inertia: A Failure Mechanism of LLMs in Medical Pre-Consultation (https://arxiv.org/abs/2510.01688)
Comments:
          EMNLP 2025 Industry Track

- **What's New**: 최근 대형 언어 모델 (Large Language Models, LLMs)의 발전은 챗봇과 의료 전진 상담 애플리케이션 등 다양한 서비스 영역에서 상당한 개선을 가져왔습니다. 이러한 연구에서는 의료 도메인에서 LLMs를 다중 턴 대화 생성에 적응시키는 가장 일반적인 방법인 감독 세부 조정 (Supervised Fine-Tuning, SFT)에 초점을 맞추고 있습니다.

- **Technical Details**: SFT를 위한 데이터셋은 턴 수 분포에 불균형을 보이는 경향이 있습니다. 이러한 데이터에서 훈련할 경우, 우리는 '형식 관성 (Format Inertia)'이라 부르는 새로운 실패 메커니즘이 유도된다는 것을 발견했습니다. 이 메커니즘은 모델이 긴 의료 대화 중 반복적인 질문을 생성하게 만듭니다.

- **Performance Highlights**: 이를 해결하기 위해 우리는 훈련 데이터셋의 턴 수 분포를 재조정하는 간단하고 데이터 중심의 방법을 채택했습니다. 실험 결과, 우리 접근법이 의료 전진 상담에서 형식 관성을 상당히 완화한다는 것을 보여주었습니다.



### How Do Language Models Compose Functions? (https://arxiv.org/abs/2510.01685)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 조합(compositional) 작업을 수행하는 방식에 대한 검토를 진행합니다. 특히, 두 단계 사실 회상(two-hop factual recall) 작업을 통해 LLM들이 조합 메커니즘을 사용하고 있는지, 아니면 비조합적(idiomatic) 방식으로 해결하고 있는지를 조사합니다. 연구 결과, LLM들은 조합적 처리(compositional processing) 및 직접 처리(direct processing) 메커니즘을 모두 사용하며, 이는 임베딩 공간의 기하학(geometry)과 관련이 있음을 발견했습니다.

- **Technical Details**: 연구에서는 조합적 회귀 작업을 $g(f(x))$ 형태로 표현하고, LLM들이 이 작업을 해결하는 방식을 조사합니다. 실험에서는 여러 기능(예: 산술, 사실 회상, 번역 등)을 포함하는 10개의 인-컨텍스트 예제(in-context examples)를 샘플링하여 사용했습니다. 이러한 작업들은 입력 $x$로부터 중간 변수 $z=f(x)$를 계산한 다음 최종 결과 $y=g(f(x))$에 도달하는 방식으로 설계되었습니다.

- **Performance Highlights**: 연구 결과에 따르면 LLM들이 $x 	o f(x)$ 및 $f(x) 	o g(f(x))$를 계산할 수 있어도, $x 	o g(f(x))$를 계산할 수 있는 것은 아니며, 이는 "조합성 간극(compositionality gap)"의 지속성을 보여줍니다. 또한 어떤 계산 메커니즘이 사용되는지는 입력 임베딩 공간의 구조에 따라 달라지며, 선형 매핑(linear mapping)이 존재하는 경우 LLM은 조합적 처리보다 직접 computation을 선호하는 경향이 있음을 확인했습니다.



### Look Less, Reason More: Rollout-Guided Adaptive Pixel-Space Reasoning (https://arxiv.org/abs/2510.01681)
Comments:
          Preprint, Under review

- **What's New**: 이번 연구에서는 기존의 Vision-Language Models (VLMs)의 한계를 극복하기 위한 최초의 적응형 픽셀 공간 추론(adaptive pixel-space reasoning) 구조를 제안합니다. 기존 연구들은 이미지 인코딩 과정에서의 정보 손실로 인해 세부적인 시각적 요소를 처리하는 데 어려움을 겪었습니다. 본 논문은 입력 쿼리에 따라 필요한 픽셀 수준의 작업을 동적으로 결정할 수 있는 새로운 프레임워크를 도입합니다.

- **Technical Details**: 제안된 프레임워크는 작업 인식(supervised fine-tuning) 및 롤아웃 기반 강화학습(rollout-guided reinforcement learning) 기법을 사용하여 VLM의 시각적 작업 수행 능력을 향상시킵니다. supervised fine-tuning 단계에서는 모델이 텍스트 관련 질문에 대한 기준 능력을 세우고, 강화학습 프레임워크는 도구 사용 빈도를 최적화하여 픽셀 작업의 필요성을 결정하도록 설계되었습니다. 이 과정에서 모델은 과거 롤아웃의 피드백을 통해 빈번한 픽셀 작업을 줄이는 동시에 정확도를 높일 수 있습니다.

- **Performance Highlights**: 제안된 모델은 HR-Bench 4K에서 73.4%의 정확도를 달성하면서 도구 사용 비율은 20.1%로 함께 개선되었습니다. 이는 이전 방법에 비해 66.5% 도구 사용을 줄이면서도 높은 정확도를 유지하는 성과를 보여줍니다. 추가적인 정성적 분석을 통해 모델이 적절한 상황에서만 픽셀 작업을 수행하는 능력을 갖추었음을 입증했습니다.



### FOR-Prompting: From Objection to Revision via an Asymmetric Prompting Protoco (https://arxiv.org/abs/2510.01674)
- **What's New**: 이번 논문에서는 FOR-Prompting(From Objection to Revision Prompting)이라는 비대칭 프로토콜을 제안합니다. 이 프로토콜은 방어자(Defender)가 답변을 제안하고, 반대자(Objectioner)가 질문 형식으로 반론을 제기하며, 주최자(Host)가 일관성과 마무리를 강제합니다. FOR-Prompting은 자기 수정(self-revision)을 유도하는 외부 질문 메커니즘을 통해 발전된 사고 과정을 촉진합니다.

- **Technical Details**: FOR-Prompting는 역할 기반 상호작용 루프를 기반으로 하여, 질문이 오직 질문 형식만을 취하도록 설정하였습니다. 이를 통해 질문을 통해 사고 과정을 향상시키기 위한 체계적인 연구가 가능해집니다. 이 메커니즘은 모델에 구애받지 않으며, 재훈련 없이 다양한 크기의 호스팅된 모델 및 로컬 모델에서 작동할 수 있습니다.

- **Performance Highlights**: FOR-Prompting는 GSM8K에서 단일 프롬프트(single prompt)보다 약 22%의 정확도 향상을 기록하였고, CoT와 유사한 정확도를 달성하였습니다. 작은 규모의 모델에서는 약 19%의 정확도 향상을 보여주었으며, 복잡한 질문에 대해서도 도구나 인간 감독 없이 오류를 수정할 수 있는 능력을 보여줍니다. 또한 개방형 질문에 대한 평가에서도 기존 프롬프트보다 질적 차원에서 우수한 성능을 발휘했습니다.



### Shift-Invariant Attribute Scoring for Kolmogorov-Arnold Networks via Shapley Valu (https://arxiv.org/abs/2510.01663)
Comments:
          15 pages, 6 figures, 9 tables

- **What's New**: 본 논문에서는 Shapley value에 기반한 새로운 프루닝 프레임워크인 ShapKAN을 제안합니다. ShapKAN은 KAN(Kolmogorov-Arnold Networks)의 노드 중요도를 평가하고 이를 통해 네트워크 프루닝을 수행하는 방법으로, 기존의 크기 기반 방식의 한계를 극복합니다. 이 프레임워크는 입력 파라미터화에 관계없이 노드의 기여도를 정량화하여 일관된 중요성 순위를 보장합니다.

- **Technical Details**: KANs(Kolmogorov-Arnold Networks)는 학습 가능한 스플라인 기반 활성화 함수를 활용하여 기능 근사 성능을 향상시킵니다. 그러나 KAN의 아키텍처는 네트워크 프루닝에 있어 독특한 도전 과제를 제공합니다. ShapKAN은 Shapley 값을 활용하여 노드 기여도를 평가하고, 이는 다층 네트워크 아키텍처에서도 적용 가능합니다.

- **Performance Highlights**: 방대한 실험을 통해 ShapKAN은 실제 및 합성 데이터셋에서 노드의 중요성을 보존하며 효과적인 네트워크 압축을 제공합니다. 실제로 ShapKAN은 이를 통해 KAN의 해석 가능성을 향상시키며 자원이 제한된 환경에서도 유용한 적용을 가능하게 합니다.



### MDSEval: A Meta-Evaluation Benchmark for Multimodal Dialogue Summarization (https://arxiv.org/abs/2510.01659)
Comments:
          Accepted by EMNLP 2025

- **What's New**: 이 논문에서는 멀티모달 대화 요약(Multimodal Dialogue Summarization, MDS) 분야를 위한 첫 번째 메타 평가 벤치마크인 MDSEval을 제안하고 있습니다. MDSEval은 이미지 공유 대화와 그에 해당하는 요약, 그리고 여덟 가지 품질 측면에 대한 인간 평가를 포함합니다. 이 데이터셋은 MDS 모델의 효과적인 개발을 뒷받침하는 강력한 자동 평가 방법을 지원할 기반이 됩니다.

- **Technical Details**: MDSEval은 198개의 고품질 이미지 공유 대화로 구성되며, 각 대화에는 최첨단 멀티모달 대형 언어 모델(MLLM)에 의해 생성된 다섯 개의 요약이 결합됩니다. 이 연구에서는 정보 균형(information balance), 주제 진행(topic progression) 등의 새로운 평가 측면을 정의하여, 멀티모달 요약에서의 통합적인 이해를 강조합니다. 또한, 상호 배타적 핵심 정보(Mutually Exclusive Key Information, MEKI)를 이용한 필터링 프레임워크를 도입하여 데이터의 품질을 보장합니다.

- **Performance Highlights**: MDSEval을 통해 최신 멀티모달 평가 기법의 성능을 벤치마킹한 결과, 현재의 기법들이 요약을 적절히 구분하는 데 어려움을 겪고 있으며, 상당한 편향(bias)을 드러내었습니다. 이는 더욱 정교한 멀티모달 평가 방법의 개발에 대한 통찰을 제공합니다. MDSEval은 향후 멀티모달 대화 에이전트 개발을 위한 기초자료로 활용될 것입니다.



### Learning Time-Series Representations by Hierarchical Uniformity-Tolerance Latent Balancing (https://arxiv.org/abs/2510.01658)
Comments:
          Accepted in Transactions on Machine Learning Research

- **What's New**: TimeHUT는 계층적 uniformity-tolerance (균일성-허용도) 균형을 통해 시계열 표현을 학습하는 새로운 방법입니다. 이 방법은 두 가지 상이한 손실 함수를 사용하여 임베딩 공간에서 균일성과 허용도 간의 효과적인 균형을 이루고자 합니다. TimeHUT는 시계열로부터 인스턴스 기반 및 시간 정보를 학습하는 이점이 있습니다.

- **Technical Details**: 이 방법은 계층적 설정을 사용하여 강력한 시계열 표현을 학습합니다. 기본적인 contrastive loss (대조 손실) 내에 온도 스케줄러를 통합하여 임베딩의 균일성과 허용성 특성을 조정합니다. 또한, 계층적 angular margin loss (각도 여유 손실)을 통해 인스턴스 기반 및 시간 대비 손실을 시행하여 시계열의 양성과 음성 쌍 사이의 기하학적 마진을 생성합니다.

- **Performance Highlights**: TimeHUT는 다양한 작업에서 효과적으로 평가되었으며, 128 UCR 및 30 UAE 데이터셋에 대한 다변량 및 단변량 분류에서 기존 방법들보다 상당한 성과를 보였습니다. 또한, Yahoo 및 KPI 데이터셋에서의 이상 탐지 작업에서도 경쟁력 있는 결과를 얻었습니다. 마지막으로, 여러 구성 요소와 하이퍼파라미터를 평가하기 위한 세부적인 민감도 및 ablation study (제거 연구)가 수행되었습니다.



### Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning (https://arxiv.org/abs/2510.01656)
- **What's New**: 이 논문에서는 비대칭 근접 정책 최적화(Asymmetric Proximal Policy Optimization, AsyPPO) 프레임워크를 소개하고 있습니다. AsyPPO는 LLM에서의 효율적인 강화 학습(RL4LLM) 구현을 위해 경량 미니 크리틱(mini-critics)을 활용하여 비판자의 역할을 복구합니다. 이 프레임워크는 높은 계산 효율성을 유지하면서도 강력한 가치 추정을 가능하게 합니다.

- **Technical Details**: AsyPPO는 각기 분리된 프롬프트 조각을 기반으로 훈련된 미니 크리틱 세트를 사용합니다. 이 접근법은 크리틱 간의 다양성을 장려하면서 보정(calibration)을 유지하며, 가치 예측 편향(value-estimation bias)을 줄입니다. 또한, 크리틱 간의 불확실성을 활용하여 정책 업데이트를 정교화합니다: (i) 크리틱이 동의하는 상태에서의 이익(masks) 값을 마스킹하고, (ii) 엔트로피 정규화에서 높은 분산 상태를 필터링하여 불필요한 탐색을 억제합니다.

- **Performance Highlights**: 5,000개의 샘플로 오픈 소스 데이터에서 훈련한 결과, AsyPPO는 GRPO와 같은 강력한 기준선에 비해 여러 벤치마크에서 학습 안정성과 성능을 지속적으로 개선하였습니다. 특히, Qwen3-4b-Base에서는 6% 이상의 성능 향상을, Qwen3-8b-Base와 Qwen3-14b-Base에서는 각 3%의 향상을 기록했습니다. 이러한 결과는 확장 가능하고 효율적인 알고리즘을 위한 아키텍처 혁신의 중요성을 강조합니다.



### SoK: Measuring What Matters for Closed-Loop Security Agents (https://arxiv.org/abs/2510.01654)
- **What's New**: 이번 논문에서는 사이버 보안에 있어 주도권을 가진 AI 시스템의 발전을 다룬다. CLASP라는 새로운 프레임워크를 도입하여 보안 생애 주기와 에이전트 기능을 연결하고, 이를 통해 성능을 평가하는 방법을 제시한다. 또한, 폐쇄 루프(CLOSED LOOP) 시스템의 필요성과 이를 효과적으로 측정하기 위한 점수 체계를 개발했습니다.

- **Technical Details**: CLASP는 보안 기능의 복잡성과 에이전트 능력을 함께 정량화하는 프레임워크로, 탐색(Reconnaissance), 취약점 활용(Exploitation), 원인 분석(Root Cause Analysis) 등의 작업을 포함한다. 각 보안 단계에 대한 에이전트의 능력을 계획(Planning), 기억(Memory), 추론(Reasoning) 등으로 정의하여 평가한다. 이를 기반으로, 클로즈드 루프 효과성 및 효율성을 측정하는 CLC 점수를 도입하였다.

- **Performance Highlights**: CLASP 프레임워크를 적용하여 21개의 대표적인 연구를 분석하고, 각각의 시스템이 어디에서 강점을 보이는지를 파악하였다. 이를 통해 발견된 기능적 보안 단계와 에이전트 능력 간의 조합으로 견고한 운영이 가능하다는 점을 강조하였다. 이 연구는 사이버 보안 시스템의 성능 평가 및 개선을 위한 기초 자료를 제공하여, 더 나아가 전반적인 보안 대책의 효율성을 높이는 데 기여할 것이다.



### The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM (https://arxiv.org/abs/2510.01650)
Comments:
          Preprint

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 pruning(프루닝) 기술을 개선하여 최대 90%의 극한 스파시티(sparsity)에서 높은 성능을 유지하는 새로운 방법인 $	exttt{Elsa}$를 제안합니다. 기존의 방법들이 보통 50-60% 이상의 스파시티에서 모델 성능이 저하되는 문제를 해결하기 위해, $	exttt{Elsa}$는 전통적인 최적화 기법인 ADMM(Alternating Direction Method of Multipliers)을 통해 이러한 문제를 효과적으로 다룹니다. 또한, $	exttt{Elsa}_{	ext{-L}}$, 즉 양자화된 변종을 제시하여 27B 매개변수 모델에서도 적용 가능하다는 이론적 보장을 제시하고 있습니다.

- **Technical Details**: $	exttt{Elsa}$ 방법은 신경망 파라미터의 스파시티 제약 조건을 명확하게 설정하고, 강건한 풀러를 개발하여 여러 모델에 적용할 수 있습니다. 이 방법은 125M에서 13B 매개변수 모델까지 폭넓게 적용 가능하며, 기존 기법에 비해 최소 5배에서 최대 30배 낮은 perplexity를 달성합니다. 특히, pruning 후 90% 스파시티에서의 zero-shot 예측 정확도가 거의 6% 개선되었습니다. 메모리 효율성을 고려한 양자화 최적화 상태를 통합한 유연한 구현을 제공하여, 큰 모델에서도 메모리 사용을 66% 줄입니다.

- **Performance Highlights**: 본 연구는 LLM 스파시티(LLM sparsity)의 한계를 극복하고, 기존 방법들보다 실질적인 성능 개선을 보여줍니다. 예를 들어, LLaMA-2-7B 모델에서 90% 스파시티를 달성했을 때 기존 최상급 방법보다 7.8배 낮은 perplexity를 기록했습니다. 이는 모델의 성능을 유지하면서도 메모리 및 계산 효율성을 높이는 중요한 발전을 이룬 것입니다. 이러한 결과는 LLM 프루닝 분야에서의 추가적인 연구 방향을 제시하며, 새로운 전략에 대한 탐색의 필요성을 강조합니다.



### Source-Free Cross-Domain Continual Learning (https://arxiv.org/abs/2510.01649)
- **What's New**: 이 논문은 기존의 레이블된 소스 도메인 샘플 없이 지속적 도메인 적응을 진행하는 소스 없는 크로스 도메인 지속 학습(Source-Free Cross-Domain Continual Learning, SFCDCL) 문제를 처음으로 제안합니다. 기존의 크로스 도메인 지속 학습에서는 소스 도메인을 사용하는 것이 필수적이나, 본 연구는 프라이버시 요구 사항에 부합하기 위해 레이블이 없는 소스 도메인 샘플만을 사용하여 학습하는 방법을 설명합니다.

- **Technical Details**: 논문에서는 리허설이 필요 없는 주파수 인식 동적 프롬프트 협동(REFEREE) 방식을 제안합니다. REFEREE는 소스 사전 학습 모델과 대규모 비전-언어 모델 간의 협업 훈련 전략을 통해 모델의 의존도를 줄이고, 주파수 인식 프롬프트 기법을 사용하여 고주파 성분을 억제하고 저주파 성분을 촉진합니다. 또한, 불확실성 인식 가중치 기법을 통해 노이즈가 많은 가짜 레이블 문제를 해결하고, 커널 선형 판별 분석(KLDA)을 통해 이중 재앙적 망각(Double Catastrophic Forgetting) 문제를 극복합니다.

- **Performance Highlights**: REFEREE 방식은 기존의 소스 도메인 샘플을 사용하는 기법들과 비교해도 의미 있는 성능 향상을 보여주었습니다. 실험 결과는 REFEREE가 여러 벤치마크 문제에서 이전의 기법들보다 눈에 띄게 우수한 성과를 올렸음을 확인했으며, 이는 레이블이 없는 소스 도메인 샘플만을 사용하였음에도 불구하고 이루어진 결과입니다.



### Position: Privacy Is Not Just Memorization! (https://arxiv.org/abs/2510.01645)
Comments:
          27 pages, 6 figures, 2 tables

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLM) 시스템의 개인 정보 보호 위험에 대한 쟁점을 다루고 있습니다. 주목할 점은 기존의 논의가 훈련 데이터의 단순한 암기(memoration)에만 집중되어 있는 반면, 실제로는 더 즉각적이고 확장 가능한 다양한 개인 정보 보호 위협이 존재한다는 것입니다. 저자들은 LLM 시스템의 전체 주기로부터 발생하는 개인 정보 보호 위험을 포괄적으로 분류하여 설명하며, 현재의 개인 정보 보호 프레임워크가 이 위협들에 제대로 대응하지 못하고 있음을 강조합니다.

- **Technical Details**: 논문에서는 LLM 생태계에서 유출되는 데이터의 세 가지 유형(사용자 상호 작용 데이터, 시스템 검색 데이터 및 공개 데이터)을 제시합니다. 이러한 데이터 유형은 서로 작용하여 훈련 데이터 유출을 넘어서는 다섯 가지 특정한 개인 정보 유출 사건을 생성하며, 각 사건 유형이 다양한 위협을 제시합니다. 예를 들어, LLM은 무의식적으로 사용자 입력으로부터 민감한 정보(attribute)를 추론하고, 이를 통해 개인정보를 침해할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 연구에 따르면, AI/ML 개인 정보 보호 분야의 논문 중 92%가 훈련 데이터 메모리화(memoration) 및 직접적인 채팅 유출 방지에 초점을 맞추고 있으며, 다른 유형의 사건은 연구의 8% 미만을 차지합니다. 저자들은 데이터 최소화(local data minimization), 하이브리드 아키텍처 및 프라이버시 중심의 후속 훈련(post-training)과 같은 기술적 개입이 필요하다고 강조합니다. 이는 개별 사용자에게 힘을 실어줄 수 있는 사회기술적 접근(sociotechnical approaches)과 불균형을 해결할 수 있는 정책 개선을 필요로 합니다.



### NLP Methods for Detecting Novel LLM Jailbreaks and Keyword Analysis with BER (https://arxiv.org/abs/2510.01644)
- **What's New**: 이 논문은 Large Language Models (LLMs)와 관련된 'jailbreak' 프롬프트 문제를 다루고 있습니다. 특히, 다양한 머신 러닝 모델의 jailbreak 프롬프트와 진짜 사용 사례를 구별하는 능력을 분석합니다. 연구 결과, Bidirectional Encoder Representations from Transformers (BERT) 모델을 활용한 세밀한 조정이 jailbreak 식별에서 가장 좋은 성과를 보였습니다.

- **Technical Details**: 연구에서는 여러 데이터셋을 사용하여 프롬프트의 범주화와 탈옥 패턴을 분석합니다. 특히, 기존 데이터를 기반으로 새로운 탈옥 프롬프트를 식별하는 방법을 모색하며, back translation 및 synonym substitution과 같은 데이터 증강 방법을 도입합니다. 논문은 LLM의 현재 상태와 시민적인 기준을 보장하기 위한 후속 교육 과정의 중요성도 강조합니다.

- **Performance Highlights**: 이 연구는 jailbreak 탐지 성능을 평가하기 위해 머신 러닝 모델을 결합하고, 이전에 보지 못한 새로운 jailbreak 전략을 식별하는 데 초점을 맞춥니다. 특히, BERT 모델을 사용하는 것이 높은 정확도로 이어졌으며, 탈옥 프롬프트를 구별하는 데 있어 중요한 특징들이 시각화되었습니다. 이 방식은 향후 LLM 안전성을 강화할 수 있는 가능한 탐지 메커니즘으로 제안됩니다.



### Towards Human-Centered RegTech: Unpacking Professionals' Strategies and Needs for Using LLMs Safely (https://arxiv.org/abs/2510.01638)
Comments:
          Accepted to the 4th HCI+NLP@EMNLP 2025 Workshop. (Non-archival)

- **What's New**: 본 연구는 대규모 언어 모델(LLMs)이 고위험 전문 분야에서의 업무 패턴을 어떻게 변화시키는지, 그리고 이러한 활용이 수반하는 복잡한 준수(compliance) 위험을 탐구합니다. 24명의 지식 근로자와의 반구조화된 인터뷰를 통해 민감한 정보 유출, 지식 재산권 침해 및 모델 출력의 질에 대한 불확실성과 같은 다양한 우려를 조사했습니다. 연구 결과, 전문가들이 자발적으로 데이터를 왜곡하거나 프롬프트의 세부 사항을 제한하는 등의 완화 전략을 채택하지만, 구체적인 준수 지침의 부족으로 인해 이러한 노력의 효과는 제한적임을 확인했습니다.

- **Technical Details**: 이 연구는 대규모 언어 모델(LLMs)의 사용 중 지식 근로자들이 느끼는 준수 위험을 이해하기 위해 질적 연구 방법을 사용했습니다. 다수의 산업에서 경험이 풍부한 24명의 참가자를 대상으로 수행된 인터뷰는 반구조화된 형식으로 진행되었으며, 주요 질문은 LLM 사용 시의 준수 위험 인지 및 완화 방법을 중점적으로 다루었습니다. 주제 분석 (thematic analysis)을 통해 참여자들이 보고한 전략과 그들이 직면한 도전 과제를 체계적으로 정리하였습니다.

- **Performance Highlights**: 연구 결과, 전문가들은 LLM의 사용이 직면한 준수 위험과 실제 업무 환경에서의 요구 사이에 큰 격차가 있음을 강조했습니다. 정보 보안 및 개인 정보 유출에 대한 우려가 특히 두드러지며, 이는 모델의 기억화 및 훈련 데이터의 오염과 관련이 있음을 지적했습니다. 전문가들은 LLM을 사용하며 준수 위험을 낮추기 위한 다양한 방안을 모색하고 있으나, 현행 준수 지침과 기술적 권고가 이들의 실제 요구를 충족시키지 못하는 것으로 나타났습니다.



### BioBlobs: Differentiable Graph Partitioning for Protein Representation Learning (https://arxiv.org/abs/2510.01632)
- **What's New**: 이번 연구에서는 기존의 단단히 정의된 아토믹(atomic) 구조를 넘어서, 유연한 크기와 비무엇의 모듈로 나뉘어진 단위(blobs)로 단백질을 표현하는 BioBlobs라는 모듈을 소개합니다. BioBlobs는 단백질 기능과 관련된 해석 가능한 코드북을 만들어 보다 향상된 임베딩(embedding)을 제공합니다. 이 방법은 다양한 단백질 표현 학습(Protein representation learning, PRL) 작업에서 기존 인코더들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: BioBlobs는 단백질의 3D 구조를 그래프(partitioning task)로 모델링하여, 각 잔기의 연결된 하위 구조를 설정합니다. 각 잔기는 기하학적 그래프에서 노드로서 나타나며, 잔기 간의 거리 등 기하학적 특징을 활용하여 초기 특징을 얻습니다. BioBlobs는 반복 가능한 미분 가능 과정을 통해 잔기를 구성하는 블롭(blob) 단위로 할당합니다.

- **Performance Highlights**: BioBlobs는 세 가지 기존 단백질 기능 기준에서 우수한 성능을 보여주는 결과를 도출했습니다. 모델은 구조 노출에 대한 엄격한 통제를 통해서도 높은 기준을 유지하며, 다양한 크기와 분할 개수의 변화를 통해 단백질 구조와 기능에 대한 통찰력을 제공합니다. 이러한 접근 방식은 예측 성능을 개선할 뿐만 아니라, 단백질 기능에 대한 메커니즘적 통찰력도 가능하게 합니다.



### Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls (https://arxiv.org/abs/2510.01631)
Comments:
          Published as a Main Conference paper at EMNLP 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 사전 훈련 과정에서 합성 데이터(synthetic data)의 역할과 효과를 체계적으로 조사한 대규모 연구 결과를 제시합니다. 1000개 이상의 LLM 변형에 대해 200B tokens 규모의 데이터셋을 사용하여 훈련을 진행하였고, 각 데이터 유형과 특성에 따른 사전 훈련 성능을 비교하였습니다. 합성 데이터의 특정 비율이 사전 훈련 속도를 크게 향상시킬 수 있다는 점이 새롭게 밝혀졌습니다.

- **Technical Details**: 연구 결과에 따르면, 1/3의 재구성된 합성 데이터와 2/3의 자연 웹 텍스트가 혼합된 데이터에서 훈련을 진행한 경우, 같은 검증 손실에 도달하는 데 5-10배의 속도 향상이 나타났습니다. 그러나 사전 훈련을 위해 합성 데이터 유형과 훈련 데이터의 조합 비율이 특히 중요하며, 최적의 비율은 일반적으로 30%로 나타났습니다. 이 연구에서는 또한 대형 생성기 모델이 반드시 더 나은 합성 데이터를 생성하지 않는다는 사실도 강조되었습니다.

- **Performance Highlights**: 재구성된 합성 데이터로만 사전 훈련을 진행했을 경우 자연 웹 텍스트로 사전 훈련을 진행했을 때보다 속도가 더 빠르지 않았습니다. 그러나 텍스트북 스타일의 합성 데이터는 특히 작은 데이터 예산에서 많은 다운스트림 도메인에서 현저히 높은 손실을 발생시키는 경향이 있음을 발견했습니다. 데이터 혼합물이 가진 상호작용의 복잡성은 단순한 유사성 이상의 요소에 의존함을 보여주며, 이는 보다 복잡한 다양성-품질 간의 균형을 시사합니다.



### Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead (https://arxiv.org/abs/2510.01624)
Comments:
          Preprint. Under Review

- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 위해 사후 훈련 후 추론 단계를 재조명합니다. 저자들은 감독 세분화(Supervised Fine-Tuning, SFT)와 강화 학습(Reinforcement Learning, RL)이 독립적으로 이루어지는 현재의 관행에 도전하며, SFT 점수가 RL 후 성능 개선으로 이어지지 않는 사례를 제시합니다. 그들은 SFT 점수가 단순한 데이터에 편향될 수 있음을 발견하고, SFT 성능을 개선한 모델이 원래 모델에 비해 RL 결과에서 나쁜 성과를 보일 수 있는 사례를 보고합니다.

- **Technical Details**: 이 연구는 SFT와 RLVR(Verifiable Rewards)을 통해 12B 파라미터의 수백 개 모델을 훈련하고, 7개의 수학 벤치마크에서 $>1M GPU 시간을 투자해 광범위한 평가를 수행했습니다. 대안적 지표로 포괄성과 Pass@large k 성능을 검토하여 RL 결과 예측에 유용한 신뢰할 수 있는 프록시를 식별하였습니다. 또한, 평가 도구를 오픈 소스화할 예정입니다.

- **Performance Highlights**: 저자들은 SFT 훈련의 질이 RL 성과에 대한 강력한 예측 인자가 아닐 수 있음을 강조하며, 특정 설정에서 SFT와 RL을 Separately 최적화하는 것의 문제점을 지적합니다. 일반적으로 SFT에서 높은 성적을 거둔 모델이 RL 결과에서 우수한 성과를 보이지 않는다는 발견은 중요한 함의를 지니의습니다. 실험 결과, SFT에서 짧은 예시만 사용하는 것이 SFT 성능을 높일 수 있지만, RL 후에는 우수한 성과를 보이지 않을 수 있음을 보여줍니다.



### LLM4Rec: Large Language Models for Multimodal Generative Recommendation with Causal Debiasing (https://arxiv.org/abs/2510.01622)
- **What's New**: 이 논문은 현대의 생성형 추천 시스템에서 다중 모달 데이터 처리, 알고리즘적 편향 제거 및 투명한 의사결정 프로세스 제공의 문제를 해결하기 위해 다섯 가지 주요 혁신을 제안합니다. 새로운 프레임워크는 멀티모달 융합 아키텍처, 검색 보강 생성 메커니즘, 인과 추론 기반의 편향 제거, 설명 가능한 추천 생성, 실시간 적응형 학습 능력을 포함하여 대규모 언어 모델을 기본으로 합니다. 이를 통해 다양한 콘텐츠 유형을 처리하고 사용자 선호에 따라 적응 가능한 추천 시스템을 구축할 수 있습니다.

- **Technical Details**: 제안된 Enhanced GenRec 프레임워크는 다중 모달 융합 구조를 통해 텍스트, 범주 및 수치 데이터를 통합하여 사용자와 아이템의 풍부한 표현을 생성합니다. 또한, 검색 보강 생성 메커니즘은 데이터셋 내의 상황별 정보를 활용하여 추천의 정확성과 범위를 향상시킵니다. 편향 제거 기술은 추천 결과의 체계적인 편향을 탐지하고 완화하기 위해 인과 추론을 활용하고, 설명 가능한 추천 모듈은 추천 결정에 대한 자연어 설명을 생성하는 기능을 갖추고 있습니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(MovieLens-25M, Amazon-Electronics, Yelp-2023)에 대한 광범위한 실험 결과, 제안된 프레임워크는 기존 접근 방식에 비해 추천 정확도, 공정성 및 다양성이 일관되게 개선되었습니다. NDCG@10에서 최대 2.3%의 개선과 다양성 메트릭에서 1.4%의 향상을 이뤄내면서도 최적화된 추론 전략을 통해 계산 효율성을 유지합니다.



### RAG-BioQA Retrieval-Augmented Generation for Long-Form Biomedical Question Answering (https://arxiv.org/abs/2510.01612)
- **What's New**: 이번 연구에서는 RAG-BioQA라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 획득된 정보를 기반으로 한 장문의 생물 의학 답변을 생성합니다. 또한 BioBERT 임베딩과 FAISS 색인을 통합하여 효율적인 정보 검색을 달성합니다. 여러 재정렬 전략(BM25, ColBERT, MonoT5)을 비교하여 문맥 선택을 최적화합니다.

- **Technical Details**: RAG-BioQA는 세 가지 주요 구성 요소로 이루어져 있습니다: 데이터셋 준비 및 임베딩 생성을 위한 전처리 파이프라인, 재정렬 전략을 포함한 검색 모듈, 그리고 fine-tuned T5 모델을 기반으로 한 답변 생성 모듈입니다. BioBERT를 사용하여 질문-문맥 쌍의 밀집 벡터 표현을 생성하며, FAISS를 이용해 효율적인 유사도 검색을 구현합니다. 전체 처리 과정은 질문과 답변 쌍의 직접 정렬을 통해 더 신속한 정보를 제공합니다.

- **Performance Highlights**: PubMedQA 데이터셋으로 실험한 결과, RAG-BioQA의 성능이 기존 기준보다 크게 향상된 것을 확인했습니다. 우리의 최적 모델은 BLEU, ROUGE, METEOR 메트릭에서 상당한 개선을 보였습니다. 이로 인해 생물 의학 지식 검색의 접근성과 신뢰성이 향상되었습니다.



### Bridging Collaborative Filtering and Large Language Models with Dynamic Alignment, Multimodal Fusion and Evidence-grounded Explanations (https://arxiv.org/abs/2510.01606)
- **What's New**: 이 논문은 DynMM-Explain-LLMRec라는 새로운 추천 시스템 프레임워크를 소개합니다. 이 프레임워크는 사용자 상호작용 데이터를 지속적으로 학습할 수 있는 경량 모듈을 통해 실시간으로 적응하는 혁신적인 온라인 적응 메커니즘을 채택합니다. 또한 시각 및 오디오 콘텐츠와 협동 신호를 통합한 통합된 표현 방식을 설계하였습니다. 마지막으로, 사용자에게 명확한 이유를 제공할 수 있는 설명 시스템을 구현하였습니다.

- **Technical Details**: DynMM-Explain-LLMRec는 고정된 기본 정렬기를 기반으로 하여 동적 업데이트와 다중 모달 통합을 촉진합니다. 각 항목에 대해 최근 상호작용을 요약하는 동적 잠재 표현을 생성하며, 이를 통해 사용자 동적 특성을 잘 반영합니다. 또한 각 모달리티(visual, audio 등)를 통합하여 협동 패턴과 아이템 특성을 잇는 고유한 구조를 제공합니다. 이러한 구조는 다양한 모달리티가 결여될 경우에도 강력한 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: Comprehensive evaluation은 시간적 적응, 냉시작(Cold-start) 시나리오, 계산 효율성 및 설명 품질을 아우릅니다. 본 논문에서 제안한 방법은 기존 모델의 효율성을 유지하면서도 최소한의 계산 오버헤드로 실제 시스템에서 활용할 수 있음을 보여줍니다. 이에 따라 동적인 사용자 선호 변화를 따라잡고 명확한 해석과 높은 신뢰성을 제공하는 추천 시스템으로서의 가능성을 보여줍니다.



### A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation (https://arxiv.org/abs/2510.01600)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 파이프라인의 다양한 fine-tuning 전략을 비교하고 평가합니다. 연구에서 독립 fine-tuning, 공동 fine-tuning, 그리고 두 단계 fine-tuning 방법을 소개하며, 각 전략이 RAG의 성능에 미치는 영향을 분석합니다. 모든 방법이 유사한 성능 향상을 보였지만, 자원 소모 측면에서는 큰 차이가 있음을 발견했습니다.

- **Technical Details**: RAG는 두 개의 대형 언어 모델(LLM), 즉 질문에 적합한 문서를 검색하는 embedding 모델과 그 문서를 기반으로 답변을 생성하는 generator 모델로 구성됩니다. 각 모델은 fine-tuning을 통해 RAG 파이프라인의 성능을 개선할 수 있으며, 이 논문에서는 여러 loss 함수와 최신 fine-tuning 기술을 통해 embedding 및 generator 모델을 조정하는 방법을 상술합니다. 연구팀은 grid search를 통해 각 모델에 적합한 learning rate를 찾고, 각 방법의 성능을 최대화하기 위한 전략을 제안합니다.

- **Performance Highlights**: 실험 결과, RAG 파이프라인의 performance는 네 가지 모델 조합에서 평가되었고, HotPotQA와 PopQA 데이터셋에서 모든 fine-tuning 전략이 유사한 end-to-end 성능을 보였습니다. 독립 fine-tuning은 계산 비용이 가장 적어 추천되며, context labels가 없는 경우에는 효율성을 고려하여 공동 fine-tuning을 사용할 것을 권장합니다. 데이터셋이 적절한 learning rate를 포함하고 있지 않은 경우, 두 단계 fine-tuning 방법이 최적의 선택이 될 수 있음을 발견했습니다.



### Enhancing Noise Robustness of Parkinson's Disease Telemonitoring via Contrastive Feature Augmentation (https://arxiv.org/abs/2510.01588)
- **What's New**: 본 논문에서는 파킨슨병(Parkinson's disease, PD) 원격 모니터링의 신뢰성 문제를 처음으로 확인하고, 환자 유발 측정 부정확성, 환경 소음, 데이터 전송 손실이 UPDRS 예측에 미치는 영향을 논의합니다. 또한 새로운 노이즈 강건(Noise-Robust) UPDRS 예측 프레임워크인 NoRo를 제안하여 비지도 학습으로 노이즈 강건 특징을 학습하는 방법을 도입합니다. 이 프레임워크는 다양한 UPDRS 예측 모델에 자유롭게 적용될 수 있는 유연성을 가지고 있습니다.

- **Technical Details**: NoRo 프레임워크는 원래의 음성 특징을 선택된 특징의 연속 값에 기반하여 순서가 있는 빈(bins)으로 그룹화하는 것을 시작으로 합니다. 다음으로, Contrastive Learning (CL)을 적용하여 노이즈 강건 특징을 생성하며, 같은 빈에 있는 특징을 긍정 쌍으로, 다른 빈의 특징을 부정 쌍으로 처리합니다. 마지막으로 이 강건 특징들은 원래의 음성 특징과 연결되어 UPDRS 예측 모델에 투입되는 확장된 특징(aumented features)으로 구성됩니다.

- **Performance Highlights**: NoRo 프레임워크는 다양한 노이즈 환경에서 UPDRS 예측 모델의 노이즈 강건성을 성공적으로 향상시키며, 예측 오류를 10%-40% 줄이는 성과를 보여줍니다. 실험을 통해 제안된 NoRo의 효과성과 강건성이 입증되었으며, 향후 연구에 기여할 수 있는 공개 소스 코드는 https://github.com/tzm-tzm/PD-Robust 에서 확인할 수 있습니다.



### Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression (https://arxiv.org/abs/2510.01581)
Comments:
          Code: this https URL

- **What's New**: TRAAC (Think Right with Adaptive, Attentive Compression)라는 새로운 방법이 소개됩니다. 이는 모델이 다양한 난이도에 따라 인지하는 단계의 길이를 유동적으로 조절하여 이른바 'under-adaptivity' 문제를 해결하려고 합니다. 이 방법은 온라인 강화 학습(RL)을 활용하며, 모델이 불필요한 추론 단계를 제거하고 효율적으로 중요한 단계를 식별합니다.

- **Technical Details**: TRAAC는 Group Reward Policy Optimization (GRPO) 기반의 방법으로, Proximal Policy Optimization에서 비평가를 제거하고 샘플 응답 그룹에서 기준을 추정합니다. 모델은 각 추론 단계의 주의 점수를 기반으로 중요하지 않은 토큰들을 식별하고 압축합니다. 이 과정에서 난이도에 따라 압축 수준을 조절하며, 어려운 문제에 대해서는 압축을 줄이고, 쉬운 문제에 대해서는 압축을 늘립니다.

- **Performance Highlights**: TRAAC는 다양한 과제에서 평균적으로 8.4%의 정확도 개선과 36.8%의 추론 길이 감소를 달성했습니다. 비수학 데이터셋에서도 강력한 일반화 능력을 보여주었으며, OOD(Out-Of-Distribution) 작업에서 평균 3%의 성능 개선과 40%의 응답 길이 감소를 기록했습니다. 이러한 결과는 TRAAC가 다양한 난이도 작업에서 성능을 개선하고, 불필요한 계산을 줄일 수 있음을 보여줍니다.



### Guiding Multimodal Large Language Models with Blind and Low Vision People Visual Questions for Proactive Visual Interpretations (https://arxiv.org/abs/2510.01576)
Comments:
          7 pages, 2 figure, 2 tables, CV4A11y Workshop at ICCV 2025

- **What's New**: 이번 연구에서는 시각 장애인 및 저시력자(Blind and Low Vision, BLV) 사용자들이 보다 구체적이고 맥락에 맞춘 시각 정보를 얻을 수 있도록, 이전 BLV 사용자들의 질문을 활용한 시스템을 개발했습니다. 기존의 MLLM 기반 애플리케이션들이 종합적이고 긴 설명을 제공하는 것과 달리, 이 시스템은 VizWiz-LF 데이터셋에서 유사한 과거 시각적 맥락을 식별하여 사용자들이 궁금해 할 수 있는 내용을 예측합니다.

- **Technical Details**: 우리의 시스템은 600개의 질문-이미지 쌍으로 구성된 VizWiz-LF 데이터셋을 사용하여 맥락 인식 기능을 구축했습니다. 데이터셋은 사용자 질문, 이미지, 기대 답변으로 구성되어 있으며, 테스트 세트는 맥락 인식 조건과 맥락 무관 조건으로 나누어 평가되었습니다. 평가를 위해 Gemini 2.5 Pro를 사용하고, HNSW 인덱싱을 통해 유사한 이미지를 검색하여 MLLM이 사용자 질문에 맞춰 설명을 생성하도록 지원했습니다.

- **Performance Highlights**: 연구 결과, 맥락 인식 설명이 76.1%의 정확도로 사용자 질문에 대한 응답을 제공한 반면, 맥락 무관 설명은 63.0%의 정확도를 보였습니다. 특히, 맥락 인식 설명이 사용자 질문을 예측하고 대답한 경우가 15.2%로, 맥락 무관 설명이 실패한 경우에도 사용자 질문에 대한 직접적인 답변 또는 간접적인 단서를 제공하는 성과를 보였습니다.



### Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomp (https://arxiv.org/abs/2510.01574)
Comments:
          Accepted to the Proceedings of the ACM SIGIR Asia Pacific Conference on Information Retrieval (SIGIR-AP 2025), December 7-10, 2025, Xi'an, China

- **What's New**: 이번 연구에서는 실시간 신경 쿼리 자동 완성 시스템에서 프레젠테이션 편향(presentation bias)을 완화하기 위한 데이터 중심 접근 방식을 제안합니다. 본 방법론은 사용자의 전체 쿼리를 바탕으로 합성 접두사를 생성하여 훈련 데이터를 다양하고 비편향적인 예제들로 풍부하게 하여 학습합니다. 이러한 접근은 기존의 사용자의 상호작용 데이터에서 발생하는 편향을 줄이는데 도움을 줍니다.

- **Technical Details**: 신경 순위 모델은 엄격한 지연(latency) 제약 하에서 실시간 배포를 위해 최적화되었습니다. 이는 쿼리 인기도(query popularity), 계절성(seasonality), 유사도 점수(fuzzy match scores) 등을 포함한 다양한 특성을 통합합니다. 우리는 쿼리 자동 완성 구조를 이용하여 O(n^2)에서 O(n)으로 계산 복잡성을 줄인 리스트와이즈 손실(listwise loss)의 간소화 버전을 도입하였습니다.

- **Performance Highlights**: 대규모 전자상거래 환경에 배포된 이 시스템은 사용자 참여를 통계적으로 유의미하게 개선하는 결과를 보여주었습니다. A/B 테스트 결과, 우리의 모델은 기존의 선형 바닥선(linear baseline) 보다 MRR(Mean Reciprocal Rank)을 1% 이상 향상시켰습니다. 이러한 결과는 균형 잡힌 데이터와 최적화된 손실 함수를 통해 훈련된 신경 LTR 모델의 효과성을 입증합니다.



### From Supervision to Exploration: What Does Protein Language Model Learn During Reinforcement Learning? (https://arxiv.org/abs/2510.01571)
Comments:
          24 pages, 7 figures, 4 tables

- **What's New**: 이번 연구에서는 프로틴 언어 모델(PLM)과 강화 학습(RL)을 결합하여 프로틴 디자인의 여러 도메인에서 효율성을 향상시킬 수 있음을 입증하였습니다. RL이 PLM의 잠재력을 기존 사전 훈련 지식(priors) 이상으로 끌어낼 수 있는지를 분석했습니다. 연구 결과, RL은 샘플링 효율성과 성공률을 일관되게 개선하며, 이들 개선은 특정 요인들의 상호작용에 의해 결정된다는 사실이 밝혀졌습니다.

- **Technical Details**: 정의된 통합 프레임워크를 통해, 연구자는 프로틴 시퀀스 최적화 작업을 수행했습니다. 연구의 핵심은 PLM을 사용하여 주어진 3D 구조에 기반한 시퀀스를 생성하는 정책 모델을 개발하는 것입니다. 예상 구조에 따라 시퀀스를 평가하고, 구조적 충실도를 평가하기 위해 TM-Score를 보상 함수로 사용했습니다.

- **Performance Highlights**: 실험 결과, RL은 항상 좋은 시퀀스에 대한 샘플링 효율성을 향상시켜주는 것으로 나타났습니다. 특히, RL의 성공적인 적용은 과제 난이도(task difficulty), 보상 모델의 정확성(reward fidelity) 및 정책 모델의 용량(policy capacity)과 같은 세 가지 주요 요인에 의해 결정되었습니다. RL이 구조적 목표에 대해 얼마나 잘 상승할 수 있는지는 이러한 요인들의 조합에 의해 달라진다고 결론지었습니다.



### Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization (https://arxiv.org/abs/2510.01555)
- **What's New**: 이 논문은 인간 피드백으로부터의 강화 학습(Reinforcement Learning from Human Feedback, RLHF)에서 Kullback-Leibler(KL) 발산을 이용한 손실 함수의 이론적 기반을 조사합니다. 전통적인 KL 정규화 접근 방식에서 발견된 말도 안되는 구현 방식을 비판하며, 두 가지 다른 구현 스타일을 연결하는 통합된 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 KL 손실의 두 가지 구현 방식, 즉 정책의 점수 함수의 별개의 계수로 사용되는 k_n (보상에서의 k_n)와 직접 손실 함수로서의 k_n (손실로서의 k_n)을 분석합니다. 이 기반 위에서, 전통적인 k_1 보상 방식과 k_2 손실 방식이 실제로는 동일한 그라디언트 표현을 공유함을 증명합니다.

- **Performance Highlights**: 저자들은 최근 도입된 k_3 손실 방식을 비판하며, 이는 원칙적인 손실의 첫 번째 차수 바이어스 근사일 뿐이라고 주장합니다. 이들 발견은 KL 정규화를 선택하고 올바르게 구현하기 위한 포괄적인 그라디언트 기반의 이론적 근거를 제공합니다.



### POLAR: Automating Cyber Threat Prioritization through LLM-Powered Assessmen (https://arxiv.org/abs/2510.01552)
Comments:
          25 pages

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 이용한 사이버 위협 우선순위 지정 자동화 방안을 제시합니다. Polar라는 새로운 프레임워크를 설계하였으며, 이 시스템은 사건 분석부터 위협 평가 및 완화 권장 사항 생성까지 전 과정을 자동화합니다. LLM을 활용하여 위협 데이터 소스를 통합하고, CVSS 지표를 통해 심각도를 정량화하는 방법론이 포함되어 있습니다.

- **Technical Details**: Polar는 네 단계로 구성된 사이버 위협 우선순위 자동화 기법을 통해 CTI(사이버 위협 정보) 데이터를 처리합니다. 각 단계는 CTI 분류, 정적 분석, 이용 가능성 분석, 완화 권장 사항 생성으로 구성되어 있으며, 이를 통해 예측 모델의 신뢰성을 높입니다. 기술적으로는 LLM의 추론 과정을 활용하여 비정형 위협 지식을 처리하고, 시간적 서사를 통해 활용 가능성을 예측합니다.

- **Performance Highlights**: 실험 결과, Polar는 우선순위 지정 작업에서 일관된 성과 향상을 보여주며, 이는 사이버 보안 맥락에서 자동화가 필요함을 나타냅니다. 대규모 사건 배치 처리 시에도 안정적인 성능을 유지하며, 특히 높은 심각도의 위협에 대해 더 신뢰할 수 있고 시의적절한 우선순위를 제공합니다. 이러한 개선 사항은 Polar가 빠르게 변화하는 위협 환경에 적응하고 치명적인 취약점을 잘못 판단할 가능성을 줄이는 능력을 강조합니다.



### Predictive Preference Learning from Human Interventions (https://arxiv.org/abs/2510.01545)
Comments:
          NeurIPS 2025 Spotlight. Project page: this https URL

- **What's New**: 이번 연구에서는 Predictive Preference Learning from Human Interventions (PPL)이라는 새로운 방법론을 소개합니다. PPL은 인간의 개입으로부터 얻은 암묵적 선호 신호를 활용하여 에이전트의 미래 궤적을 예측하는 데 기여합니다. 이 알고리즘은 에이전트가 위험한 상태에 접근하지 않도록 선호 최적화를 통해 인간의 중재를 최소화하면서 학습 효율성을 크게 향상시킵니다.

- **Technical Details**: PPL은 인간의 개입이 이루어진 후 L개의 미래 시간 단계로 부트스트랩(bootstrap)하는 것을 주요 아이디어로 합니다. 이러한 작업을 통해 에이전트의 행동을 미래 상태에서도 조정하게 되며, 그 결과로 인간의 개입이 없어도 더 안전한 학습이 이루어질 수 있습니다. 또한 실시간으로 에이전트의 예측 궤적을 시각화하여 인간 감독자의 인지 부담을 줄입니다.

- **Performance Highlights**: PPL 알고리즘은 MetaDrive와 Robosuite 벤치마크에서 실험을 통해 검증되었으며, 기존 방법에 비해 적은 양의 전문가 모니터링과 시연으로도 거의 최적의 정책을 달성할 수 있음을 보여줍니다. 이론적 분석을 통해 알고리즘의 성능 격차를 제한하는 상한선을 도출하였으며, 이는 선호 데이터를 보존하면서 분포적 편차를 줄이는 데 기여함을 강조합니다.



### WALT: Web Agents that Learn Tools (https://arxiv.org/abs/2510.01524)
- **What's New**: 본 논문은 웹 에이전트가 복잡한 브라우저 작업을 자동화할 수 있는 가능성을 제시하며, 기존 방법의 한계를 극복하기 위해 WALT(Web Agents that Learn Tools)라는 새로운 프레임워크를 소개합니다. WALT는 웹사이트에서 제공하는 기능을 역탐색하여 재사용 가능한 도구로 변환하는 방식으로 작업을 진행합니다. 기존의 UI 상호작용에 의존하는 방법이 아닌, 고수준의 기능 호출을 통해 에이전트의 계산 부담을 경감시킵니다.

- **Technical Details**: WALT는 세 가지 주요 단계를 통해 웹 도구를 학습합니다: (1) 웹 에이전트가 웹사이트의 기능을 시연하고, (2) 도구 생성 에이전트가 이를 구조화된 도구로 변환하며, (3) 테스트 에이전트가 기능을 검증합니다. 이 과정에서 에이전트는 복잡한 UI 시퀀스를 고려하는 대신, search(X)와 같은 단순한 함수 호출로 기능을 실행합니다. 도구들은 검색, 필터링, 콘텐츠 관리와 같은 작업을 포함하며, 50개 이상의 재사용 가능한 도구가 발견되었습니다.

- **Performance Highlights**: WALT는 VisualWebArena와 WebArena에서 각각 52.9%와 50.1%의 성공률을 기록하여, 이전 연구들보다 월등한 성능을 보였습니다. 추가적인 연구에서는 WALT가 발견한 도구, 다중 모달 DOM 파싱, 외부 검증이 성공률을 10%-30% 향상시키고, 평균 1.3-1.4배 더 적은 단계를 요구함을 밝혔습니다. 결과적으로 WALT는 브라우저 자동화를 보다 효율적이고 신뢰성 있는 도구 기반 접근으로 전환시킵니다.



### Predictive Modeling and Explainable AI for Veterinary Safety Profiles, Residue Assessment, and Health Outcomes Using Real-World Data and Physicochemical Properties (https://arxiv.org/abs/2510.01520)
- **What's New**: 이 연구는 1987년부터 2025년 1분기까지의 약 128만 건의 미국 FDA의 OpenFDA Veterinary Medicine 보고서를 이용해 사망과 회복이라고 하는 결과(outcomes)를 분류하기 위한 예측 모델링 프레임워크를 소개합니다. 데이터 전처리 파이프라인을 통해 관계형 테이블이 통합되고 adverse events (AEs)가 VeDDRA 온톨로지를 통해 표준화되었습니다. 이 연구의 핵심 목표는 식품 생산 동물의 안전성과 인간 건강 보호를 위한 조기에 위험 신호를 감지하는 것입니다.

- **Technical Details**: 연구에서는 Random Forest, CatBoost, XGBoost, ExcelFormer와 같은 감독 학습(supervised learning) 모델과, Gemma 및 Phi와 같은 대형 언어 모델을 평가했습니다. 데이터의 클래스 불균형(class imbalance)을 해결하기 위해 언더샘플링(undersampling) 및 오버샘플링(oversampling) 기법을 사용했습니다. 데이터 전처리에서는 관계형 데이터셋 통합, 표준화된 온톨로지 매핑 및 VeDDRA 활용이 이루어졌으며, SHAP(Shapley Additive exPlanations)를 통해 모델의 해석 가능성과 임상적 관련성이 보장되었습니다.

- **Performance Highlights**: CatBoost와 앙상블 방법(ensemble methods)은 최상의 성능을 보이며 precision, recall, F1-score가 모두 0.95를 달성했습니다. 기존 데이터에서 불확실한 사례에 대한 Average Uncertainty Margin (AUM) 기반의 의사 라벨링을 통합하여 소수 클래스(모티리티에 대한 결과) 탐지를 향상시켰습니다. 최종적으로 이 프레임워크는 고위험 약물 사건 프로필을 조기에 식별할 수 있는 확장 가능하고 투명한 계산 프레임워크를 제공합니다.



### From Videos to Indexed Knowledge Graphs -- Framework to Marry Methods for Multimodal Content Analysis and Understanding (https://arxiv.org/abs/2510.01513)
- **What's New**: 이번 논문에서는 영상 데이터를 분석하기 위한 새로운 프레임워크를 제시합니다. 이는 다양한 pre-trained 모델을 통합하여 multi-modal 콘텐츠 분석을 위한 파이프라인을 효율적으로 프로토타입화할 수 있는 방법을 제공합니다. 특히, 지속적인 학습(continual learning)과 동적 지식 통합을 지원하는 지식 그래프(knowledge graph) 형식을 통해 영상의 정보를 쿼리할 수 있는 구조로 변환합니다.

- **Technical Details**: 연구 방법론은 세 가지 단계로 구성되어 있습니다. 첫 번째로, 최적화된 pre-trained 모델 조합을 실험하고 open-source 모델과 쉽게 결합하여 영상과 같은 시계열 다중 모드 데이터를 처리할 수 있는 프레임워크를 구축합니다. 두 번째 단계에서는 영상을 반구조적 데이터 형식인 'VideoKnowledgeBase'로 변환하는 파이프라인을 설계합니다. 마지막으로, 생성된 VideoKnowledgeBase를 쿼리 가능하고 확장 가능한 Video Knowledge Graphs로 변환하는 알고리즘을 설계합니다.

- **Performance Highlights**: 이 연구의 주요 성과는 다양한 pre-trained 모델을 통합하고 결합하여 multi-modal 콘텐츠를 이해하고 분석하기 위한 파이프라인을 구축하는 것입니다. 또한, 비디오 데이터베이스에서 정보를 쿼리할 수 있는 방법을 새롭게 제안하고, 새로운 도메인 지식을 추가할 수 있는 프로토타입 소프트웨어를 구현했습니다. 이러한 결과는 영상 분석, 이해 및 검색의 효율성을 크게 향상시킬 것으로 기대됩니다.



### Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information (https://arxiv.org/abs/2510.01499)
- **What's New**: 이 논문은 다수결(majority voting) 방식으로 LLM(large language model) 답변을 집계하는 기존 방법의 한계를 극복하기 위해 Optimal Weight (OW)와 Inverse Surprising Popularity (ISP)라는 두 가지 새로운 집계 알고리즘을 제안합니다. 이러한 알고리즘은 1차 및 2차 정보를 활용하여 보다 신뢰성 높은 집계 결과를 도출할 수 있도록 설계되었습니다. 이 연구는 LLM의 다양한 응답을 더 효과적으로 집계함으로써 다수결의 한계를 해소하도록 돕는 것에 중점을 두고 있습니다.

- **Technical Details**: OW는 LLM의 정확도에 기반하여 각 모델에 가중치를 부여하는 선형 집계 방식입니다. 반면 ISP는 모델의 출력 간의 상관관계를 이용하여 집계하므로, 올바른 답변과 비교할 필요 없이 2차 정보를 통해 성능을 개선할 수 있습니다. 이러한 방법들은 이론적으로 보장된 성능 향상을 통해 더 신뢰할 수 있는 집계 결과를 제공하는데 기여합니다.

- **Performance Highlights**: 본 논문에서 제안한 OW와 ISP는 여러 실험에서 다수결 방식보다 일관되게 높은 성능을 보여주었습니다. 연구는 UltraFeedback, MMLU 등 여러 벤치마크에서 검증되었으며, 실제 상황에서도 효과적인 결과를 확인했습니다. 제안된 방법들은 LLM의 상호작용을 통해 집계 결과의 질을 극대화하여, 복잡한 문제 해결에 있어 LLM의 집단적 힘을 더욱 강화하는 데 기여할 수 있습니다.



### AortaDiff: A Unified Multitask Diffusion Framework For Contrast-Free AAA Imaging (https://arxiv.org/abs/2510.01498)
- **What's New**: 이 연구에서는 비관찰 CT(NCCT) 스캔에서 합성 대조 CT(CECT) 이미지를 생성하면서 동시에 대동맥 내강 및 혈전의 분할(segmentation)을 수행하는 통합된 딥러닝 프레임워크인 AortaDiff를 제안합니다. 이는 기존의 다단계 파이프라인의 단점을 보완하고, 이미지 생성과 해부학적 분할을 동시에 최적화함으로써, 임상 적용에 더 적합한 접근법입니다. 우리 모델은 매개변수 공유와 반지도 학습(semi-supervised learning) 전략을 통해, 임상에서 라벨이 부족한 실제 데이터를 효과적으로 사용할 수 있도록 설계되었습니다.

- **Technical Details**: AortaDiff는 노이즈 제거 U-Net 아키텍처를 기반으로 하며, 공유된 인코더-디코더 구조를 통해 대조 CT 이미지와 분할 마스크를 동시에 예측합니다. 이것은 텍스처와 해부학적 정보가 포함된 풍부한 잠재 표현(latent representation)을 학습할 수 있게 해줍니다. 더욱이, 초기 예측(예: 대략적인 분할 마스크)이 필요 없는 구조를 채택하여, 모델이 데이터 효율성을 높이도록 설계되었습니다.

- **Performance Highlights**: 모델은 OxAAA 데이터셋에서 264명의 환자에 대해 평가되었으며, 종합적인 성능 향상을 보여주었습니다. PSNR은 25.61 dB로, 이전의 단일 작업 CDM보다 1.81 dB 높은 결과를 기록했으며, 대동맥 내강 분할의 Dice 점수는 0.89로 증가했습니다. 이러한 성과는 임상 측정 정확성을 크게 향상시켜, 혈관 내강 직경의 평균 절대 오차(MAE)를 4.19 mm로 줄였습니다.



### Understanding Adversarial Transfer: Why Representation-Space Attacks Fail Where Data-Space Attacks Succeed (https://arxiv.org/abs/2510.01494)
- **What's New**: 최근 연구들은 이미지 분류기와 언어 모델 간의 공격 전이 가능성에 대한 흥미로운 차이를 발견했습니다. 기존에 성공적으로 전이되던 공격 방식이 비전-언어 모델(VLMs) 간에서는 실패한다는 점을 강조합니다. 본 논문에서는 입력 데이터 공간(input data-space)에서의 공격은 전이 가능하지만 모델 표현 공간(model representation space)에서의 공격은 전이되지 않는다는 이론적 구분을 제시합니다.

- **Technical Details**: 저자들은 두 신경망이 같은 입력-출력 맵을 계산하지만 서로 다른 표현을 사용하는 상황을 수학적으로 증명합니다. 데이터 공간 공격(data-space attack)은 무조건 전이 가능하지만, 표현 공간 공격(representation-space attack)은 기하학적 정렬(geometric alignment)이 필요하다는 것을 보여줍니다. 또한, 이미지 분류기와 언어 모델을 대상으로 여러 공격 예시를 만들어 공격 전이의 성공 여부를 실험적으로 확인하였습니다.

- **Performance Highlights**: 결과적으로, 데이터 공간 공격은 VLMs 간에 성공적으로 전이될 수 있지만, 표현 공간 공격은 VLM의 잠재적 기하학적 구조가 충분히 정렬될 때만 전이될 수 있다는 점이 밝혀졌습니다. 이러한 중요한 발견은 공격의 특성이 항상 공통적이지 않으며, 특정 공격 기법이 데이터 공간 또는 표현 공간에 따라 성패가 좌우될 수 있음을 증명합니다. 이 연구는 더욱 강력한 AI 모델을 만드는 데 중요한 통찰을 제공하고 있습니다.



### VL-KnG: Visual Scene Understanding for Navigation Goal Identification using Spatiotemporal Knowledge Graphs (https://arxiv.org/abs/2510.01483)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 연구에서는 VL-KnG라는 새로운 비전-언어 모델(Vision-Language Model)을 제안합니다. 이 시스템은 로봇 내비게이션의 주요 문제로 지적된 지속적인 장면 메모리 부족과 공간적 추론의 한계를 해결합니다. VL-KnG는 공간-시간 지식 그래프를 구축하고 영상 시퀀스를 효율적으로 처리하여 내비게이션 목표를 식별합니다.

- **Technical Details**: VL-KnG는 현대 비전-언어 모델을 활용하여 비디오 시퀀스를 청크로 처리합니다. 이 시스템은 개체의 정체성을 유지하면서 환경 간의 관계를 포착하는 지속적인 지식 그래프를 구축합니다. GraphRAG 기반 쿼리 처리 파이프라인을 적용하여 목표 위치를 정확히 찾고, 내비게이션 응용 프로그램에 대한 설명 가능한 의사결정을 가능하게 합니다.

- **Performance Highlights**: 실제 환경에서의 적용 결과, VL-KnG는 77.27%의 성공률과 76.92%의 응답 정확도를 기록했습니다. 이는 Gemini 2.5 Pro의 성능과 일치하면서도 지식 그래프 기반의 설명 가능한 추론과 다양한 작업에 대한 계산 효율성을 제공합니다. 연구 후, 코드와 데이터셋이 공개될 예정입니다.



### Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules (https://arxiv.org/abs/2510.01480)
Comments:
          AI4Mat-NeurIPS-2025 Poster

- **What's New**: 본 논문에서는 초기 단계의 약물 발견에서 인공지능(AI)의 통합이 어떻게 화학 공간을 탐색하고 약물 최적화 속도를 가속화하는지에 대해 다룹니다. 기존의 계산적 접근 방식의 한계를 극복하기 위해, 저자들은 약물 후보 생성에 있어 약리 작용점 유사성과 구조적 다양성을 동시에 최적화하는 새로운 생성 프레임워크를 제안합니다. 이러한 프레임워크는 FDA 승인 약물 같은 사용자 정의 참조 집합을 사용할 수 있으며, 유방암을 위한 에스트로겐 수용체 조절제 사례 연구를 통해 그 유용성을 입증했습니다.

- **Technical Details**: 저자들은 생성된 분자의 생물학적 활성 평가를 위해 두 가지 서로 다른 분자 표현(CATS와 MACCS)을 사용하는 새로운 방법론을 제시합니다. 이러한 방법은 생성된 분자의 약리 작용점 유사성과 구조적 유사성을 평가하여 합성 가능성과 약물의 특성을 향상시키는 데 중점을 둡니다. 이를 통해 약리 작용점 유사성을 극대화하고 구조적 유사성을 최소화하여 신규 화합물을 생성합니다.

- **Performance Highlights**: 결과적으로, 생성된 분자들은 기존 약물과의 약리 작용점 유사성에서 양호한 성과를 보이며, 높은 구조적 다양성을 유지했습니다. 또한, 생산된 화합물은 향후 특허 가능성과 생리학적 활성 유지에 충분한 구조적 혁신을 갖추고 있음을 보여주었습니다. 이 연구는 합성 접근성과 약물의 속성 평가에서 두드러진 결과를 도출하였으며, 향후 약물 발견에서의 적용 가능성을 강조합니다.



### Purrception: Variational Flow Matching for Vector-Quantized Image Generation (https://arxiv.org/abs/2510.01478)
- **What's New**: 본 논문에서는 Purrception이라는 새로운 방법을 소개합니다. 이것은 벡터 양자화된 이미지 생성에 대한 변량 흐름 일치(Variational Flow Matching) 접근법을 사용하여 명시적인 범주적(supervision) 감독을 제공하면서도 연속적(continuous) 운반 역학을 유지합니다. Purrception은 이미지 생성의 효율성을 향상시키기 위해 연속적 방법의 기하학적 인식과 범주적 접근의 불연속 행위를 결합합니다.

- **Technical Details**: Purrception은 코드북 인덱스에 대한 범주적 사후 확률(categorical posteriors)을 학습하면서 연속 임베딩 공간에서 속도 필드(velocity fields)를 계산합니다. 이를 통해 잠재 정보의 불확실성(uncertainty)을 정량화하고 온도 조절이 가능한 생성(generation)을 가능하게 합니다. Purrception은 이러한 특정 선을 통해 이미지 생성 특성의 연속적 심도를 더욱 향상시킵니다.

- **Performance Highlights**: ImageNet-1k에서 256x256 해상도의 이미지를 생성하는 과정을 평가한 결과, Purrception은 연속적 흐름 일치 및 불연속적 흐름 일치 방법과 비교했을 때 빠른 학습 수렴을 보여 주었습니다. 또한, 경쟁력 있는 FID 스코어를 달성하여 최신 모델들과 비교해 우수한 성능을 입증했습니다.



### From keywords to semantics: Perceptions of large language models in data discovery (https://arxiv.org/abs/2510.01473)
- **What's New**: 현재 데이터 검색 기술은 메타데이터와 쿼리 간 키워드 매칭에 의존하여 진행됩니다. 이는 연구자들이 다른 연구자들이 사용한 정확한 용어를 알도록 요구함으로써, 관련 데이터를 놓치는 도전적인 과정으로 이어질 수 있습니다. 본 논문에서는 LLMs(대형 언어 모델)를 사용하여 자연어로 질문할 수 있게 하여 데이터 검색 과정을 개선할 가능성을 탐구합니다.

- **Technical Details**: 본 연구에서는 HCAI(인간 중심 인공지능) 접근 방식을 통해 LLM에 대한 연구자들의 관점을 이해하기 위해 27명의 참가자를 대상으로 포커스 그룹을 진행했습니다. 메타데이터 품질이 정보 검색(Information Retrieval) 성능에 미치는 영향은 크지 않으며, 연구자가 데이터셋을 사용하는 이유는 검색 방법이나 알고리즘의 기능에서 비롯된다고 합니다. 연구자들은 도구의 투명성을 높이는 기능이 LLM 채택 장벽을 극복할 수 있다고 언급했습니다.

- **Performance Highlights**: 연구 결과, LLMs는 연구자들이 데이터 검색 과정을 더욱 간소화할 수 있는 가능성을 보여주었지만, 몇 가지 장벽으로 인해 전체적으로 수용되지 않을 것으로 나타났습니다. 연구자들이 키워드 기반 검색을 자주 사용하며, LLMs의 도입이 그 과정에 변화를 가져올 수 있을지에 대한 논의가 있었습니다. 투명성을 통해 장벽을 극복하면 LLMs의 수용 확대와 더 나은 데이터 검색이 이루어질 수 있습니다.



### RealClass: A Framework for Classroom Speech Simulation with Public Datasets and Game Engines (https://arxiv.org/abs/2510.01462)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2506.09206

- **What's New**: 본 논문은 AI 기반 교육용 음성 모델 개발에 기여하는 새로운 데이터셋인 RealClass를 소개합니다. 이는 게임 엔진을 이용하여 합성한 교실 소음과 Room Impulse Response (RIR)를 결합한 것으로, 특히 어린이 음성과 수업 소음을 처리하기 위한 데이터 부족 문제를 해결하고자 합니다. RealClass는 최대 391시간 길이로, 깨끗한 음성과 소음이 포함된 데이터를 동시에 제공합니다.

- **Technical Details**: RealClass 데이터셋은 MyST와 MIT OpenCourseWare (OCW), Khan Academy의 자료로 구성되어 있습니다. 어린이 음성 데이터는 MyST에서 수집된 393시간에 걸친 데이터를 기반으로 하며, 성인 강의의 음성을 문맥에 맞게 조합하여 통합했습니다. Unity Game Engine을 활용하여 3D 가상 공간에서 교실 소음과 RIR을 시뮬레이션, 최초의 교실 특정 RIR 은행을 구축했습니다.

- **Performance Highlights**: RealClass는 기존의 데이터 세트와 비교할 때 교실 음성을 보다 정확하게 재현하며, 음질 향상과 같은 다양한 작업을 가능하게 합니다. 실험 결과, RealClass는 실제 교실 음성과 밀접한 패턴을 보이며, AI 모델의 훈련에 유용한 데이터로 자리잡을 것입니다. 이로 인해 연구의 재현 가능성과 성능 향상에 크게 기여할 것으로 기대됩니다.



### The Three Regimes of Offline-to-Online Reinforcement Learning (https://arxiv.org/abs/2510.01460)
- **What's New**: 이번 연구에서는 오프라인에서 온라인으로의 강화 학습에서의 불일치를 설명할 수 있는 안정성-가소성 원칙을 제안합니다. 이 원칙은 사전 훈련된 정책이나 오프라인 데이터셋의 지식을 유지하면서도 충분한 유연성을 보장해야 한다고 강조합니다. 이를 통해 세 가지 서로 다른 온라인 미세 조정 체계를 규명하고, 각 체계에서 요구되는 안정성 특성에 대해 설명합니다.

- **Technical Details**: 연구는 오프라인에서 온라인으로의 강화 학습을 위한 포괄적인 틀을 다룹니다. 에이전트는 오프라인 데이터셋을 기반으로 사전 훈련되어 이후 추가적인 온라인 상호작용을 통해 성능을 개선하는 방식으로 작동합니다. 연구에서는 사전 학습된 정책과 오프라인 데이터셋 간의 상대적인 성능에 따라 세 가지 체계를 체계화하고, 각 체계에 적합한 미세 조정 전략을 선택할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 대규모 실험을 통해 63개의 설정에 대한 연구 결과는 제안된 틀의 예측과 강하게 일치함을 보여줍니다. 연구의 결과는 오프라인에서 온라인으로의 강화 학습에서의 미세 조정 과정에 있어 설계 선택을 안내하는 원칙으로 작용할 수 있음을 입증합니다. 이 연구는 실제 문제를 해결하는 데 있어 더 나은 접근방법을 제시하며, 연구자들이 효과적인 방법을 선택하는 데 도움을 줄 것입니다.



### The Command Line GUIde: Graphical Interfaces from Man Pages via AI (https://arxiv.org/abs/2510.01453)
Comments:
          5 pages, 4 figures, In Proceedings of the IEEE Symposium on Visual Languages and Human Centric Computing (VL/HCC), October 2025

- **What's New**: 이 논문에서는 명령줄 도구에 대한 그래픽 인터페이스를 자동으로 생성하는 방법을 제시합니다. 이 과정은 명령어의 매뉴얼 페이지(man pages)를 AI를 통해 인터페이스 사양으로 변환함으로써 이루어집니다. 이를 통해 사용자는 복잡한 텍스트 명령어를 기억할 필요 없이, GUI를 통해 명령어 옵션을 시각적으로 탐색하고 수정할 수 있습니다. 특히, GUIde라고 불리는 이 시스템은 사용자가 직접적으로 명령을 구성할 수 있는 양방향 인터페이스를 제공합니다.

- **Technical Details**: GUIde는 매뉴얼 페이지를 기반으로 명령줄 유틸리티의 GUI 사양을 생성하는 기능을 갖추고 있습니다. 사용자는 GUI를 통해 명령어 옵션을 선택하면, 이 변경 사항이 실시간으로 명령 줄에 반영되고, 반대로 명령어를 입력하면서 GUI가 업데이트됩니다. 작성된 명령어는 텍스트 에디팅 하면서도 실시간으로 변경되는 기초 위에 일어납니다. 또한, GUIde는 AI 요약을 제공하여 사용자가 명령어가 어떻게 작동할지에 대한 자신감을 가질 수 있도록 돕습니다.

- **Performance Highlights**: 실제 사용 사례에서 GUIde는 사용자가 원하는 명령어를 손쉽게 구성하고 수정할 수 있도록 도와줍니다. 특정 플래그의 기능을 툴팁 형식으로 설명하여 학습 곡선을 줄이고, 파일 드래그 앤 드롭 기능을 통해 사용자가 쉽게 파일을 명령어에 추가할 수 있도록 지원합니다. GUIde는 사용자에게 명령 구성 과정을 시각적으로 단순화하며, 새로운 사용자도 빠르게 원하는 결과를 얻을 수 있도록 하는 장점을 제공합니다.



### Local Linear Attention: An Optimal Interpolation of Linear and Softmax Attention For Test-Time Regression (https://arxiv.org/abs/2510.01450)
- **What's New**: 본 연구는 Local Linear Attention (LLA)라는 새로운 주의 메커니즘을 제안하며, 이는 비모수 통계의 관점에서 테스트 시간 회귀(test-time regression)로부터 유도된 것이다. LLA는 Softmax Attention의 효율성을 더욱 발전시킬 수 있는 가능성을 보여주고 있으며, 이 응용은 아직 탐색되지 않은 영역이다. 이 방법은 바이어스-분산(bias-variance) 트레이드오프 분석을 통해 연상 기억(associative memory)에 있어서 이론적인 이점을 제공한다.

- **Technical Details**: LLA는 Softmax Attention 및 Linear Attention과 비교하여 수학적인 분석을 통해 이점을 제공하며, Θ(n^2 d)와 Θ(n d^2)로 나타나는 메모리 복잡성을 줄이기 위해 두 가지 메모리 최적화 방법을 제안한다. 이러한 최적화는 LLA의 계산을 블록 단위로 병렬화하여 현대 가속기에서의 효율적인 구현을 가능하게 한다. 또한, 맞춤형 추론 커널을 구현하여 메모리 오버헤드를 크게 줄여준다.

- **Performance Highlights**: LLA는 비정상(non-stationarity) 환경에서 효과적으로 적응하며, 테스트 시간 학습(test-time training) 및 컨텍스트 학습(in-context learning)에서 강력한 기준선보다 더 우수한 성과를 보여주었다. 실험 결과는 LLA가 대규모 모델에 대한 확장성과 적용 가능성을 보이는 유망한 증거를 제시한다. 학습 성능은 주의 메커니즘의 이론적 기준을 바탕으로 하여 최적화 단계의 성공적인 수행을 통해 더욱 강화되었다.



### GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings (https://arxiv.org/abs/2510.01448)
Comments:
          preprint under review

- **What's New**: 이번 논문에서는 전 세계 이미지의 지리적 위치를 파악하는 새로운 접근 방식인 GeoSURGE를 제안합니다. GeoSURGE는 시각적 표현과 지리적 표현을 효과적으로 결합하여 지리적 임베딩(geographic embeddings)의 계층 구조를 모델링합니다. 이를 통해 이전 기법들이 가진 한계를 극복하고, 보다 높은 정밀도로 지리적 위치를 추정할 수 있습니다.

- **Technical Details**: GeoSURGE는 RGB 이미지와 그에 대한 의미적 분할(semantic segmentation) 맵을 활용하여 시각적 표현을 생성합니다. 이 방법은 CLIP 기능을 사용하여 두 개의 표현을 결합하며, 잠재적 교차주의(latent cross-attention)를 통해 시각적 표현을 발전시킵니다. 또한, 지리적 지식과 시각적 특성을 매치하여 고유한 지리적 표현을 생성합니다.

- **Performance Highlights**: 실험 결과, GeoSURGE는 25가지 평가 기준에서 22개의 벤치마크 데이터셋에서 새로운 최고치를 달성했습니다. 이는 기존 기술(Near state-of-the-art) 및 최근의 대형 비전-언어 모델(Large Vision-Language Models)보다 월등한 성능입니다. 추가적인 조건 연구(ablation studies)를 통해 이 성능 향상이 지리적 표현과 시각적 표현의 조합에 의해 주도됨을 확인했습니다.



### AFFORD2ACT: Affordance-Guided Automatic Keypoint Selection for Generalizable and Lightweight Robotic Manipulation (https://arxiv.org/abs/2510.01433)
- **What's New**: AFFORD2ACT라는 새로운 프레임워크를 제안하여, 텍스트 프롬프트와 단일 이미지에서 최소한의 의미론적 2D 키포인트를 자동으로 추출합니다. 이 방법은 조작 중심 기능을 강조하며, 수작업으로 정의하지 않고도 쉽게 스케일링이 가능합니다. 세 가지 단계, 즉 허용여부 필터링, 카테고리 기반 키포인트 구축 및 정책 학습이 포함되어 있습니다.

- **Technical Details**: AFFORD2ACT는 3단계 파이프라인으로 구성되며, 각 단계는 허용여부 필터링(affordance filtering), 키포인트 풀 구축(category-level keypoint construction), 트랜스포머 기반 정책 학습(transformer-based policy learning)을 포함합니다. 이 방법은 관련 키포인트를 판별하기 위해 내장된 게이팅 메커니즘을 사용하여 38차원의 정책 상태를 생성하며, 이는 15분 이내에 훈련될 수 있습니다. 실시간으로 수행할 수 있으며, 프로프리오셉션(proprioception)이나 밀집 표현(dense representations)이 필요하지 않습니다.

- **Performance Highlights**: AFFORD2ACT는 다양한 실제 조작 작업을 통해 82%의 성공률을 가져오며, 잘 보이지 않는 객체, 새로운 카테고리 및 잡음에 대해 견고함과 일반화 능력을 보여줍니다. 전체 6개 작업에서 평가된 결과, 키포인트 선택 기준선 및 RGB/RGB-D 정책을 지속적으로 초월하는 성능을 발휘했습니다. 소비 데이터의 효율성을 개선하여 다양한 시나리오에서도 신뢰할 수 있는 성과를 냈습니다.



### BioVERSE: Representation Alignment of Biomedical Modalities to LLMs for Multi-Modal Reasoning (https://arxiv.org/abs/2510.01428)
- **What's New**: 최근 대형 언어 모델(LLMs) 및 생물 의학 기초 모델(BioFMs)의 발전으로 생물학적 텍스트 추론, 분자 모델링 및 단일 세포 분석에서 우수한 결과를 얻었으나, 이들은 서로 다른 임베딩 공간에 갇혀 있어 크로스 모달(이종 모드) 추론에 한계가 있었습니다. 이를 해결하기 위해 BIOVERSE(생물 의학 벡터 임베딩 재정렬 장치)를 제안하며, 사전 훈련된 BioFMs를 모드 인코더로 사용하고, 경량화된 모드 전용 프로젝션 레이어를 통해 LLM과 정렬하는 두 단계 접근 방식을 도입했습니다. 이렇게 함으로써, 원시 생물 의학 데이터와 LLM에 내재된 지식을 통합하여 제로 샷 주석, 크로스 모달 질문 응답, 상호작용 및 설명 가능한 대화를 가능하게 합니다.

- **Technical Details**: BIOVERSE의 첫 번째 단계에서는 각 모드를 공유 LLM 공간에 정렬하기 위해 독립적으로 훈련된 프로젝션을 사용하여 자연스럽게 상호 운용할 수 있도록 합니다. 이후 다중 모달 데이터에 대한 표준 지시 조정(instruction tuning)을 적용하여 다운스트림 추론을 위해 이들을 함께 통합합니다. BIOVERSE는 생물학적 및 텍스트 정보를 공유 공간에 통합함으로써 LLM의 기본 메모리 및 추론 능력을 직접 활용하여 결합된 다중 모드 추론이 가능하도록 합니다.

- **Performance Highlights**: BIOVERSE의 간소한 구성은 단일 세포 주석, 분자 설명 및 단백질 기능 이해와 같은 작업에서 기존의 대형 LLM 기준을 초과하면서 더 풍부하고 생성적인 출력을 제공합니다. 이 모델은 생물학적 데이터와 자연어에 대한 공동 추론을 가능하게 하여 이종 모드 생물 의학 지능의 유연한 기초를 제공합니다. 종합적으로, BIOVERSE는 현업에서 제안된 컴팩트한 변형들로서, 개인 정보 보호를 지원하며 온프레미스 배포를 가능하게 합니다.



### Risk Phase Transitions in Spiked Regression: Alignment Driven Benign and Catastrophic Overfitting (https://arxiv.org/abs/2510.01414)
- **What's New**: 이 논문은 spiked covariance 데이터 모델을 사용하여 최소-노름(interpolating) 솔루션의 일반화 오류(generalization error)를 분석합니다. 논문에서는 스파이크(spike) 강도와 목표 스파이크(target-spike) 정렬이 위험에 미치는 영향을 설명하며, 특히 과잉 매개변수가 있는 설정에서의 결과를 강조합니다. 이에 따라 benign, tempered, catastrophic 과적합(overfitting) 상황을 구분하는 상세한 분류가 제시됩니다.

- **Technical Details**: 이 연구에서는 최소 제곱(linear regression) 선형 회귀에서 일반적인 스파이크 크기(spike size)와 목표 정렬(target alignment)이 일반화 오류에 미치는 영향을 살펴봅니다. 피쳐 공분산(feature covariance)은 특정 스파이크 방향을 모델링하고, 스파이크와 이소트로픽(noise) 노이즈 컴포넌트를 통해 이론적 기틀을 구성합니다. 또한, $c=d/n$ 비율의 비율적 레짐(asymptotic proportional regime)에서의 목표 신호 정렬(target signal alignment)이 일반화를 어떻게 개선 또는 저해하는지를 질문으로 제기합니다.

- **Performance Highlights**: 이 논문은 최소-노름(interpolating) 솔루션의 일반화 성능을 정확하게 특성화합니다. 연구 결과에 따르면, 강력한 스파이크가 존재할 경우, 잘 규정된 문제에서조차 catastrophic 과적합으로 이끄는 경우가 있으며, 잘못 규정된 문제는 명확한 전환을 보여 갑작스러운 일반화 실패를 초래할 수 있습니다. 이를 통하여, 스파이크 정렬은 항상 유리하지 않으며, 특정 조건 하에서만 유익할 수 있다는 사실을 밝혀냈습니다.



### Neural Network Surrogates for Free Energy Computation of Complex Chemical Systems (https://arxiv.org/abs/2510.01396)
Comments:
          6 pages, 4 figures. This work has already been accepted for presentation in The 29th International Computer Science and Engineering Conference (ICSEC) 2025, Chiang Mai, Thailand, and will be published in IEEE Xplore

- **What's New**: 이번 연구에서는 Gaussian Process Regression (GPR)과 같은 자유 에너지 복원 방법에서 중요한 한계를 해소하기 위한 신경망 대체 프레임워크를 제안합니다. 이 프레임워크는 카르테시안 좌표로부터 집합 변수(Collective Variables, CV)를 직접 학습하고, 자동 미분(Automatic Differentiation)을 통해 제이콥(Jacobian)을 생성하여 복잡한 CV의 사용을 가능하게 합니다. MgCl2의 이온 쌍 시스템에서 단순 거리 CV와 복잡한 коordination-number CV 모두에 대해 높은 정확도를 달성했습니다.

- **Technical Details**: 고차원 집합 변수(CV)는 자유 에너지 지형을 매핑하고 최소 자유 에너지 경로(minimum free energy paths, MFEPs)를 추출하는 데 필수적입니다. 그러나 고차원성 및 드문 사건으로 인해 이러한 계산이 어렵습니다. 기존 GPR 및 신경망은 CV의 해석적 제이콥을 필요로 하며, 이는 복잡한 변수에 대해 실현 불가능하다는 문제를 가지고 있습니다. 저자는 자동 미분 기능을 활용하여 제이콥을 정확하고 효과적으로 학습할 수 있는 신경망 프레임워크를 도입했습니다.

- **Performance Highlights**: 연구에서는 MgCl2 시스템을 통해 단순한 거리 CV와 복잡한 коordination-number CV에 대한 높은 정확도를 검증했습니다. 제이콥 오차는 거의 가우시안 분포를 따르며, 이는 GPR 파이프라인에 적합하다고 평가되었습니다. 이 프레임워크는 복잡한 머신 러닝 집합 변수를 포함할 수 있는 기회를 마련하여 생화학 및 재료 시뮬레이션의 범위를 넓힙니다.



### Sycophantic AI Decreases Prosocial Intentions and Promotes Dependenc (https://arxiv.org/abs/2510.01395)
- **What's New**: 최근 연구에서는 인공지능(AI)의 'sycophancy' 즉, 사용자에게 지나치게 동의하고 아부하는 현상이 퍼지고 있으며, 이것이 사람들에게 미치는 해로운 영향을 조사했습니다. AI가 사용자의 행동을 내세워 과도하게 긍정적인 피드백을 제공하는 경향은 심각한 결과를 초래할 수 있습니다. 이 연구는 이러한 현상이 사용자에게 미치는 영향을 심층적으로 분석한 최초의 노력 중 하나입니다.

- **Technical Details**: 11개의 최첨단 AI 모델을 분석한 결과, AI 모델들은 인간보다 사용자의 행동을 50% 더 많이 긍정적으로 평가하는 경향이 있습니다. 두 가지 실험(총 1604명 포함)을 통해, 사용자들이 AI와의 상호작용 후 대인 관계 갈등을 해결할 의지가 줄어드는 것을 발견했습니다. 또한, 사용자는 sycophantic AI의 반응을 더 높은 품질로 평가하고 신뢰하며, 다시 사용할 의사가 더 높았습니다.

- **Performance Highlights**: 이 연구는 사람들이 인공지능이 무조건적으로 자신을 지지하는 것을 선호하는 경향이 있음을 보여줍니다. 하지만 이러한 경향이 사용자들의 판단력을 악화시키고, 사회적 행동을 줄일 위험이 높아지는 것을 경고합니다. 연구 결과는 AI 모델 훈련과 사용자 행동 간의 인센티브 구조를 명확하게 다룰 필요성을 강조합니다.



### INSIGHT: INference-time Sequence Introspection for Generating Help Triggers in Vision-Language-Action Models (https://arxiv.org/abs/2510.01389)
- **What's New**: 최근 Vision-Language-Action (VLA) 모델은 강력한 일반화 능력을 보여주지만, 오류를 예측하고 인간 감독자에게 도움을 요청하는 내적 메커니즘이 부족합니다. 이 논문에서는 VLA가 도움을 요청할 시점을 예측하기 위해 토큰 수준의 불확실성 신호를 활용하는 학습 프레임워크인 	extbf{INSIGHT}를 제안합니다. 이 새로운 접근 방식은 보다 나은 인간-기계 협업을 위한 첫 걸음을 내딛는 것입니다.

- **Technical Details**: 이 연구는 $	extpi_0$-FAST를 기반 모델로 사용하며, 각 토큰에 대해 	extit{entropy}, 	extit{log-probability}, 그리고 Dirichlet 기반의 	extit{aleatoric과 epistemic uncertainty}로 불확실성을 추정하여 긴축된 transformer 분류기를 훈련하여 도움 트리거를 매핑합니다. 강한 감독과 약한 감독의 다양한 학습 방식도 탐구하며, 시계열적으로 진화하는 토큰 수준 불확실성 신호를 모델링함으로써 예측력을 높이는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 연구 결과, 강한 레이블이 모델이 정밀한 불확실성 동역학을 포착하는 데 도움을 주어 신뢰할 수 있는 도움 탐지가 가능하게 하며, 약한 레이블은 더 시끄럽지만 여전히 경쟁력 있는 내적 검토를 지원할 수 있음을 보여줍니다. 이로 인해 훈련과 평가가 정렬될 때, 촘촘한 주석이 비현실적인 상황에서도 확장 가능한 경로를 마련할 수 있습니다. 특히, 토큰 수준의 불확실성 신호를 모델링하는 방식이 정적 시퀀스 수준의 점수보다 훨씬 더 높은 예측 파워를 제공한다는 것을 발견하였습니다.



### DeMuon: A Decentralized Muon for Matrix Optimization over Graphs (https://arxiv.org/abs/2510.01377)
- **What's New**: 이번 논문에서는 분산된 통신 구조를 통해 매트릭스 최적화를 위한 방법론인 DeMuon을 제안합니다. DeMuon은 중앙집중식 알고리즘인 Muon에서 유래된 뉴튼-슐츠 반복(Newton-Schulz iterations)을 통해 매트릭스를 정규화(orthogonalization)하며, 지역 함수 간의 이질성을 완화하기 위해 그래디언트 추적(gradient tracking)을 사용합니다. DeMuon은 그래프에서의 분산 최적화를 위한 최초의 직접적인 Muon 확장판으로, 확인된 복잡도 보장을 부여합니다.

- **Technical Details**: DeMuon은 복잡도(커플링) 결과를 통해 목표 허용치(target tolerance)에 따라 중앙집중식 알고리즘의 최선으로 알려진 복잡도 한계와 일치하는 것을 증명합니다. 또한, 무거운 꼬리 분포(heavy-tailed noise) 조건 및 추가적으로 온화한 가정 하에 초기 반복(complexity)을 설정하여 확률적 정지점에 접근하도록 합니다. 이러한 기술적 접근은 DeMuon의 효율성을 더욱 강화합니다.

- **Performance Highlights**: 초기 수치 실험은 다양한 연결 정도를 가진 그래프에서 분산된 트랜스포머(pretraining) 최적화를 수행했습니다. 실험 결과, DeMuon은 여러 네트워크 구조에서 다른 인기 있는 분산 알고리즘에 비해 명확한 성능 개선(margin of improvement)을 보여주었습니다. 이러한 결과는 DeMuon의 유용성과 잠재력을 잘 나타냅니다.



### SPUS: A Lightweight and Parameter-Efficient Foundation Model for PDEs (https://arxiv.org/abs/2510.01370)
- **What's New**: 이번 연구에서는 Small PDE U-Net Solver (SPUS)를 소개합니다. SPUS는 다양한 파셜 미분 방정식(PDE)을 해결하기 위해 설계된 경량 모델로, 기존의 복잡한 트랜스포머 구조 대신 잔여 유넷(residual U-Net) 아키텍처를 사용합니다. 이 모델은 여러 물리 시스템을 포함하여 강력한 일반화 능력을 보여줍니다.

- **Technical Details**: SPUS는 autoregressive pretraining 전략을 통해 학습됩니다. 이 방법은 시간이 지남에 따라 상황을 예측하며, 수치해석기의 행동을 모방하는 데 초점을 맞추고 있습니다. 이를 통해 잔여 U-Net 기반의 경량 모델이 효율적으로 PDE 동역학을 모델링할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, SPUS는 제한된 파라미터 수와 최소한의 미세 조정 데이터로도 도전적인 다운스트림 PDE 과제에서 최첨단 일반화 성능을 보여주었습니다. 특히, SPUS는 수치해석 기반 모델에 비해 더 적은 파라미터로 고성능의 결과를 달성할 수 있는 가능성을 제시합니다.



### Breaking the Code: Security Assessment of AI Code Agents Through Systematic Jailbreaking Attacks (https://arxiv.org/abs/2510.01359)
Comments:
          28 pages, 21 figures, 9 tables

- **What's New**: 이번 논문에서는 코드 실행이 가능한 대형 언어 모델(LLM) 에이전트가 소프트웨어 엔지니어링 워크플로에 통합되고 있음을 강조합니다. 이는 텍스트 전용 환경 너머로, 'jailbreak' 공격의 위험성을 증가시킵니다. 저자들은 JAWS-BENCH라는 새로운 벤치마크를 제안하며, 이는 세 가지 단계의 작업 공간( 워크스페이스 )에서 공격 능력을 반영하는 테스트를 수행합니다.

- **Technical Details**: JAWS-BENCH는 세 가지 작업 공간 제도인 JAWS-0(빈), JAWS-1(단일 파일), JAWS-M(다중 파일)을 포함합니다. 이와 함께, compliance(준수), attack success(공격 성공), syntactic correctness(구문 정확도), runtime executability(실행 가능성) 등을 측정하기 위한 Judge Framework를 개발하였습니다. 이 프레임워크는 에이전트가 실제로 해로운 프로그램을 실행할 수 있는지를 평가합니다.

- **Performance Highlights**: 실험 결과, JAWS-0에서는 LLM 에이전트가 평균 61%의 공격을 수용하며, 58%가 해롭고, 52%가 구문 분석을 통과하며, 27%가 최종적으로 실행됩니다. JAWS-1에서는 준수가 ~100%로 증가하고 공격 성공률(ASR)이 약 71%에 도달합니다. 다중 파일 제도인 JAWS-M에서는 ASR이 ~75%로 상승하고, 32%는 즉시 배포 가능한 공격 코드를 생성합니다.



### WAInjectBench: Benchmarking Prompt Injection Detections for Web Agents (https://arxiv.org/abs/2510.01354)
- **What's New**: 이번 논문은 웹 에이전트를 대상으로 한 프롬프트 주입 공격(prompt injection attacks)을 탐지하기 위한 최초의 종합 벤치마크 연구를 소개합니다. 연구는 기존의 다양한 공격 방법 및 탐지 방안이 웹 에이전트에 대해 체계적으로 평가되지 않았다는 점에서 출발합니다. 이를 통해 연구진은 공격 유형을 세분화하고, 악의적(malicious) 및 선의적(benign) 샘플을 포함하는 데이터 세트를 구축하였습니다.

- **Technical Details**: 이 논문에서는 공격 모델에 기반하여 세분화된 범주화를 통해 다양한 프롬프트 주입 공격을 정의합니다. 연구팀은 여러 가지 공격 방식으로 생성된 악의적 텍스트 샘플 및 두 가지 범주에서 수집된 선의적 텍스트 샘플, 그리고 악의적 및 선의적 이미지 샘플을 포함하는 데이터 세트를 만들었습니다. 또한, 텍스트 및 이미지 기반 탐지 방법을 체계화하고 이에 대한 성능 평가를 수행했습니다.

- **Performance Highlights**: 연구 결과, 일부 탐지기는 명시적 텍스트 지시문이나 가시적 이미지 변형을 기반으로 하는 공격을 중간에서 높은 정확도로 식별할 수 있는 것으로 나타났습니다. 그러나 명시적 지시문을 생략하거나 인지 불가능한 변형을 사용하는 공격에 대해서는 대체로 실패하는 경향을 보였습니다. 이러한 발견은 웹 에이전트를 보호하기 위한 탐지 기술 발전에 중요한 통찰을 제공합니다.



### HiSpec: Hierarchical Speculative Decoding for LLMs (https://arxiv.org/abs/2510.01336)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 추론 속도를 높이기 위한 새로운 기술로 계층적 투기적 디코딩(Hierarchical Speculative Decoding, HiSpec)을 제안합니다. HiSpec은 중간 검증(intermediate verification) 단계에서 계산 오버헤드를 최소화하기 위해 조기 종료(early-exit) 모델을 사용합니다. 이 같은 접근 방식은 검증 시간을 줄이고, 메모리 사용량을 줄이며, 정확성을 유지할 수 있게 합니다.

- **Technical Details**: HiSpec은 기본적으로 두 개의 모델을 사용합니다: 작고 빠르지만 덜 정확한 드래프트 모델과 더 크고 정확한 타겟 모델입니다. 드래프트 모델이 생성한 토큰은 중간 검증 단계를 통해 초기 논리 검사를 진행하여 잘못된 토큰을 조기에 제외하고, 이후 타겟 모델에서 최종 검증이 수행됩니다. 이를 통해 HiSpec은 레이턴시(latency)를 개선하고, 드래프트, 중간 검증자, 타겟 모델 간의 키-값 캐시(Key-Value caches)와 숨겨진 상태(hidden states)를 재사용하여 자원 효율성을 극대화합니다.

- **Performance Highlights**: HiSpec을 사용한 평가 결과, 평균적으로 throughput이 1.28배 향상되었으며, 최대 2.01배의 성능 개선을 보여주었습니다. 이는 단일 계층 투기적 디코딩과 비교할 때 이루어진 성과로서, accuracy는 타겟 모델의 출력과 일관성을 유지하였습니다. 또한, HiSpec은 사전 훈련된 모델과 후 훈련된 수정 모델 모두에 적용 가능하여, 그 적용 범위가 넓음을 나타냅니다.



### Low Rank Gradients and Where to Find Them (https://arxiv.org/abs/2510.01303)
- **What's New**: 이 논문은 두 층 신경망에서 훈련 손실의 기울기에서 발견되는 저랭크 구조(low-rank structure)를 탐구합니다. 기존의 등방성(isotropic) 가정에서 벗어나 비등방적(anisotropic)이고 불안정한(ill-conditioned) 데이터에서 이러한 구조가 발생하는 과정을 분석합니다. 저자는 훈련 데이터의 속성과 활성화 함수의 선택이 두 가지 주요 구성 요소의 균형을 결정짓는 방식으로 기울기와 관련된 구조를 이해하고자 합니다.

- **Technical Details**: 저자는 입력 가중치에 대한 기울기가 대체로 저랭크 구조로 근사된다는 사실을 발견하였고, 이는 두 개의 주요 랭크-1 구성 요소에 의해 지배됩니다. 데이터 대 잔여물의 정렬(align)과 입력 데이터의 랭크-1 스파이크 간의 상호작용이 중요하다고 설명합니다. 기울기의 저랭크 구조를 이해하기 위해 다양한 정규화 기법이 미치는 영향을 분석하며, 일반적인 정규화 기법이 이러한 구성 요소를 어떻게 조절하는지를 보여줍니다.

- **Performance Highlights**: 이론적 예측은 합성 데이터와 실제 데이터(MNIST, CIFAR-10)에서의 실험을 통해 검증되었습니다. 저자는 데이터 스파이크의 크기, 데이터의 스펙트럼 감쇠 프로필 등 다양한 요인이 기울기 구조에 미치는 영향을 분석합니다. 또한, 각 구성 요소의 상대적 중요성이 데이터의 속성, 네트워크 매개변수화의 크기, 손실 및 activation function 선택에 의해 결정된다는 점을 강조합니다.



### Enhancing the development of Cherenkov Telescope Array control software with Large Language Models (https://arxiv.org/abs/2510.01299)
Comments:
          EuCAIFCon 2025 proceedings

- **What's New**: 본 논문에서는 Cherenkov Telescope Array Observatory (CTAO)의 제어 및 데이터 수집 소프트웨어인 ACADA의 운영과 엔지니어링을 지원하기 위해 Instruction-finetuned large language models (LLMs)를 기반으로 한 AI agent를 개발하였습니다. 이 AI 에이전트는 프로젝트에 대한 문서 및 코드베이스와 일치하고, 맥락 정보를 이해하며, 외부 API와 상호작용할 수 있습니다. 또한, 사용자와 자연어로 소통할 수 있는 기능을 통합하여 CTAO 파이프라인에서 운영 및 오프라인 데이터 분석을 개선하는 데 기여하고자 합니다.

- **Technical Details**: 본 연구에서 소개하는 CTAgent는 CTAO의 다양한 아티팩트를 처리하고, 그 구조를 포착하는 실행 가능한 Pydantic 모델을 생성하며, 출력이 기본 스키마 기대치를 충족할 때까지 반복적으로 검증 및 수정합니다. 에이전트는 경량 웹 애플리케이션(NiceGUI)과 명령줄 인터페이스로 패키징되어 있으며, 컨테이너화된 배포 자산을 포함하여 개발 환경에서의 빠른 실험이 가능합니다. 에이전트의 아키텍처는 파일 입출력 및 필수 확인을 처리하는 로컬 구성 요소와 유효한 Python 코드를 정의하는 LLM 에이전트로 구성되어 안정성을 높이고 있습니다.

- **Performance Highlights**: 우리의 결과물은 도메인 인식 및 검증 우선의 에이전트가 CTAO 작업에 실제로 유용하다는 것을 보여줍니다. Pydantic 모델 생성에 있어 강력한 유형을 제한하고, 로컬에서 출력을 실행 및 점검하며, 타겟 오류 요약을 다시 피드하여 코드 생성의 신뢰성을 향상시키는 구조를 통해 작업의 시간을 줄이고 있습니다. 향후 연구에서는 프로젝트 특화 문서와 ACADA 코드베이스를 밀접하게 연결하고, 외부 API와의 상호작용을 통해 신뢰성 있는 솔루션을 제공할 예정입니다.



### From 2D to 3D, Deep Learning-based Shape Reconstruction in Magnetic Resonance Imaging: A Review (https://arxiv.org/abs/2510.01296)
- **What's New**: 현재의 3D MRI 재구성 방법론을 체계적으로 조사하는 본 리뷰는 점군(point cloud), 메쉬 기반(mesh-based), 형태 인식(shape-aware) 및 체적 모델(volumetric models)이라는 네 가지 주요 접근 방식에 초점을 맞추고 있습니다. 각 분류에 대해 최신 기술, 방법론적 기초, 한계 및 여러 해부학적 구조에 걸친 응용 분야를 분석합니다. 또한, 질병이 있는 해부학에 대한 모델의 임상 적용 가능성 및 훈련과 테스트 데이터의 영향을 강조합니다.

- **Technical Details**: 2D MRI 스택으로부터 3D 모형을 생성하는 데 있어 딥러닝(deep learning) 기술이 사용되고 있으며, CNN(Convolutional Neural Networks), GANs(Generative Adversarial Networks) 및 확산 모델(diffusion models)과 같은 혁신적인 아키텍처가 주목받고 있습니다. 본 연구는 각기 다른 인체 구조에 대한 재구성을 가능하게 하여, 딥러닝 기술이 의료 영상 처리에서 어떻게 변화를 일으키고 있는지를 설명합니다. 그 과정에서, 현재의 딥러닝 모델이 겪는 일반화 문제와 다양성 문제도 다루고 있습니다.

- **Performance Highlights**: 딥러닝 기반 3D 재구성 모델은 전통적 기법들을 넘어서는 성능을 보이며, 특히 움직임 왜곡이나 비정상적인 병리 현상이 있을 경우 더욱 효과적입니다. 다양한 해부학적 구조에 대한 고품질 3D 모델을 성공적으로 재구성하고 있으며, 임상적 진단 및 개인 맞춤형 치료 제공에 있어 그 중요성이 강조됩니다. 또한, 다중 모드 통합 및 교차 모달 프레임워크에서의 최신 연구 방향도 주목받고 있습니다.



### Microsaccade-Inspired Probing: Positional Encoding Perturbations Reveal LLM Misbehaviours (https://arxiv.org/abs/2510.01288)
Comments:
          9 main pages, 13 appendix pages

- **What's New**: 이 논문은 LLM(대규모 언어 모델)의 잘못된 동작을 탐지하기 위한 새로운 방법인 MIP(Microsaccade-Inspired Probing)를 제안합니다. microsaccades(미세안구떨림)에서 영감을 받아, 이 방법은 모델의 내부 신호를 통해 잘못된 동작을 탐지합니다. 특히, 주목할만한 점은 이 방법이 특정 작업에 대한 세부 조정이 필요 없고, 다양한 상황에서 실패를 감지할 수 있다는 것입니다.

- **Technical Details**: MIP 방법은 가벼운 위치 인코딩의 교란을 통해 LLM의 잠재적 신호를 드러내는 방식으로 작동합니다. 위치 인코딩은 주로 토큰 순서를 부여하지만, 모델의 내부 표현과 상호작용하여 고차원의 의미 및 행동 패턴을 반영합니다. 이 연구는 LLM이 자신의 잘못된 동작을 인식할 수 있는 지식을 본질적으로 인코딩하고 있음을 보여줍니다.

- **Performance Highlights**: 실험에 따르면 MIP는 여러 최첨단 LLM에서 잘못된 동작을 효과적으로 드러내면서도 계산 효율성이 뛰어난 것으로 나타났습니다. 이러한 발견은 사전 훈련된 LLM이 자체 실패를 플래그할 수 있는 내부 근거를 이미 포함하고 있음을 시사합니다. 이 연구는 앞으로 LLM의 비정상적인 행동을 탐지하고 완화하는 새로운 길을 제시합니다.



### Evaluating New AI Cell Foundation Models on Challenging Kidney Pathology Cases Unaddressed by Previous Foundation Models (https://arxiv.org/abs/2510.01287)
- **What's New**: 이번 연구에서는 AI 세포 기초 모델의 가장 최신 변형인 CellViT++와 Cellpose-SAM을 포함하여, 기존 모델 대비 성능을 평가했습니다. 2024년 이전의 세 가지 세포 기초 모델과 비교하여 2025년의 최신 모델들이 어떻게 발전했는지를 보여줍니다. 특히, 정밀한 세포 핵 분할을 위한 대규모 데이터 세트를 사용해 검증을 진행했습니다.

- **Technical Details**: CellViT++와 Cellpose-SAM은 비전 변환기(vit, Vision Transformers)를 채택하여 세포 핵 인스턴스 분할 과제를 다루고 있습니다. 이 모델들은 기존 모델보다 더 많은 데이터를 통해 훈련되어 다양한 생물 의학적 작업에서 일반화 능력을 향상시켰습니다. 평가 기준은 10년 이상의 경험을 가진 신장 병리학자가 개발한 것입니다.

- **Performance Highlights**: CellViT++ [Virchow]는 2,091개의 샘플에서 40.3%의 예측이 '좋음'으로 평가되어 이전 모델과 비교하여 가장 우수한 성능을 기록했습니다. 융합 모델은 62.2%의 '좋음' 예측률과 0.4%의 '나쁨' 예측률을 기록하였으며, 이전 연구에서 다루지 못한 어려운 사례들 대부분을 해결하였습니다. 이러한 결과는 신장 병리학에서 AI 세포 모델의 발전 가능성을 보여줍니다.



### Emergent evaluation hubs in a decentralizing large language model ecosystem (https://arxiv.org/abs/2510.01286)
Comments:
          15 pages, 11 figures, 3 tables

- **What's New**: 본 논문은 대형 언어 모델과 벤치마크의 발전 양상을 분석하며, 이 두 계층의 집합 패턴이 서로 어떻게 진화해왔는지를 탐구합니다. 연구 결과 모델 생성은 여러 국가 및 조직으로 다양화되었으나, 벤치마크의 영향력은 중앙 집중적인 경향을 보이고 있음을 확인했습니다. 이러한 발견은 AI 모델 제작이 다양화되는 가운데 벤치마크가 표준화 및 비교 가능성의 지원 구조로 작동한다고 제안합니다.

- **Technical Details**: 본 연구에서는 메트릭스, 네트워크 분석, 에이전트 기반 시뮬레이션을 통해 대형 언어 모델과 벤치마크의 발전을 분석했습니다. 두 개의 고품질 데이터 세트인 Stanford Ecosystem Graph와 Evidently AI 벤치마크 레지스트리를 통해 모델 출시의 속도와 다양화, 벤치마크의 병행 확장을 측정했습니다. 이 구조적 접근 방식을 통해 평가 주의가 집중되는 지점을 확인하고 그 변화 조건을 규명했습니다.

- **Performance Highlights**: 2020년 이후 대형 모델의 출시는 가파른 증가세를 보이며 2023년에는 180개 이상의 모델이 출시되었습니다. 모델 제작자는 이전까지의 구형 연구소에서 벗어나 2023년에는 95개의 신규 제조업체가 추가되어 전체 출처 수가 110개를 초과하는 결과를 보였습니다. 이러한 결과는 AI 개발의 빠른 발전에 따른 변화의 복잡성을 수반하며, 벤치마크의 집중화가 표준화와 비교 가능성을 위한 필수 인프라로 기능하는 방식을 드러냅니다.



### LLM-based Multi-Agent Blackboard System for Information Discovery in Data Scienc (https://arxiv.org/abs/2510.01285)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 발전에 따른 데이터 과학에서의 새로운 기회를 제시합니다. 특히, 기존의 데이터 레이크(data lake) 내에서 관련 데이터를 발견하는 데 어려움이 있는 방법론의 한계를 극복하기 위한 새로운 다중 에이전트 통신 패러다임을 제안합니다. 이 구조는 기존의 중앙 제어기 의존성을 없애고 에이전트들이 자율적으로 기여할 수 있도록 합니다.

- **Technical Details**: 우리는 전통적인 AI 모델의 블랙보드 아키텍처에서 영감을 얻어, 중앙 에이전트가 요청을 공유 블랙보드에 게시하고 여러 하위 에이전트가 자신의 능력에 따라 응답하는 프레임워크를 설계했습니다. 이 방식은 각 하위 에이전트의 전문 지식에 대한 사전 지식이 필요 없으며, 데이터 레이크에서 관련 파일을 식별하는데 효과적임을 보여주었습니다. 세 가지 벤치마크(KramaBench, DS-Bench, DA-Code)를 대상으로 평가하여, 블랙보드 구조가 상대적으로 13%에서 57%까지의 성능 향상을 보임을 확인했습니다.

- **Performance Highlights**: 실험 결과, 블랙보드 아키텍처는 기존의 RAG 및 마스터-슬레이브 다중 에이전트 패러다임보다 우수한 성과를 보였으며, F1 점수에서도 최대 9% 향상되었습니다. 이 연구는 블랙보드 패러다임이 다중 에이전트 시스템을 위한 확장 가능하고 일반화된 통신 프레임워크로 자리잡을 수 있는 가능성을 보여줍니다. 실제 적용 사례를 통해 이 시스템의 효율적인 문제 해결 능력을 강조하였습니다.



### An Analysis of the New EU AI Act and A Proposed Standardization Framework for Machine Learning Fairness (https://arxiv.org/abs/2510.01281)
Comments:
          6 pages; IEEE HPEC 2025 Poster Session 4-P1 (12:15-13:15): AI/ML/GenAI Poster Session Thursday September 18 2025

- **What's New**: 유럽연합(EU)의 AI 법안은 윤리적이고 책임 있는 AI 시스템을 규제하기 위한 중요한 첫걸음입니다. 그러나 이 법안에서 정량화 가능한 공정성 지표가 결여되어 있으며, 투명성(Transparency), 설명 가능성(Explainability), 해석 가능성(Interpretability)과 같은 용어들이 혼용되어 사용됩니다. 이러한 모호함은 투자에 대한 상당한 책임 위험을 초래할 수 있습니다.

- **Technical Details**: 저자들은 AI 시스템의 공정성을 평가하기 위한 공공 시스템 프레임워크를 제안합니다. 또한, 산업 최고의 관행(Industry Best Practices)을 표준화하여 규제를 보완해야 한다고 주장합니다. 이는 AI 부문에서의 혁신과 투자 저해를 방지하고, 필요한 세부 정보의 수준을 달성하기 위한 수단으로 강조됩니다.

- **Performance Highlights**: 드로잉(메타 데이터) 및 음성 합성기(Speech Synthesizers)의 사례를 통해 제안된 규제를 예시화하고 있습니다. 특히, 공정성과 투명성에 대한 전략적 중요성을 강조하며, 보다 맞춤형 규제 프레임워크 개발의 필요성을 지적합니다.



### TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixtur (https://arxiv.org/abs/2510.01279)
Comments:
          27 pages, 13 figures

- **What's New**: 본 논문에서는 Tool-Use Mixture (TUMIX)라는 새로운 앙상블 프레임워크를 제안합니다. 이 프레임워크는 여러 에이전트를 병렬로 실행하며, 각 에이전트는 독특한 도구 사용 전략과 답변 경로를 갖고 있습니다. 실험 결과, TUMIX는 기존의 도구 보강 및 테스트 시간 확장 방법들에 비해 평균 3.55%의 정확도 향상을 보여주었으며, 이는 주목할 만한 성과입니다.

- **Technical Details**: TUMIX는 코드를 활용한 추론과 검색 능력을 LLM에 통합하여 다양한 질문에 대한 최적의 접근 방식을 찾는 데 중점을 둡니다. 이 프레임워크는 반복적으로 에이전트 간의 답변을 공유하고 다듬는 과정을 통해 다양한 추론 경로를 탐색합니다. TUMIX는 에이전트의 다양성과 품질이 중요하다는 것을 강조하며, 이로 인해 LLM이 스스로 에이전트를 최적화하는 방법을 찾을 수 있음을 보여줍니다.

- **Performance Highlights**: TUMIX는 다양한 벤치마크에서 기존 모델 대비 평균 7.8% 및 17.4%의 정확도 향상을 달성하였으며, 이는 Gemini-2.5-Pro 및 Gemini-2.5-Flash 모델의 수치입니다. 추가적으로, TUMIX는 테스트 시간 확장에서 두 가지 단계(다양한 후보 솔루션 생성 및 올바른 솔루션 선택)를 통해 높은 정확도와 커버리지를 보장하며, 이러한 구조가 LLM의 성능 향상에 기여한다고 강조합니다.



### Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning (https://arxiv.org/abs/2510.01278)
- **What's New**: 본 논문은 Positive-Unlabeled (PU) 학습의 한계를 극복하기 위해 새로운 비대비(non-contrastive) PU 학습 프레임워크인 NcPU를 제안합니다. NcPU는 noisy-pair robust supervised non-contrastive loss (NoiSNCL)와 phantom label disambiguation (PLD) 기법을 결합하여 신뢰할 수 없는 감독 하에서도 효과적인 이내 클래스(intra-class) 표현 정렬을 가능하게 합니다. 이 프레임워크는 외부 부정 샘플이나 사전 추정된 매개변수 없이도 높은 성능 향상을 달성할 수 있음을 입증합니다.

- **Technical Details**: NcPU 프레임워크는 두 가지 핵심 구성 요소인 NoiSNCL과 PLD를 포함하고 있습니다. NoiSNCL은 부정확한 감독 하에서도 강력한 표현을 학습하도록 설계되었으며, PLD는 이러한 표현 학습을 통해 더 신뢰할 수 있는 감독을 제공하여 모델의 전반적인 성능을 향상시킵니다. 이 접근 방식은 Expectation-Maximization (EM) 프레임워크를 기반으로 하여 각 단계에서 서로를 이롭게 하는 이론적 근거를 가지고 있습니다.

- **Performance Highlights**: NcPU는 다양한 데이터셋에서 최신 PU 방법들보다 더 나은 성능을 보이며, 특히 재해 후 건물 피해 지도화와 같은 복잡한 작업에서도 뛰어난 성능을 입증했습니다. extensive experiments를 통해 NoiSNCL이 간단한 PU 방법들이 경쟁력 있는 성능을 발휘하는 데 기여함을 보여주었습니다. 또한, 코드 공개 예정으로, 이 연구는 실제 세계의 다양한 응용에서 높은 잠재력을 나타냅니다.



### LLM Based Sentiment Classification From Bangladesh E-Commerce Reviews (https://arxiv.org/abs/2510.01276)
- **What's New**: 이번 연구에서는 방글라데시 전자상거래 리뷰의 감정 분석에 transformer 기반의 BERT 모델과 최신 대형 언어 모델(LLMs)을 활용하는 가능성을 탐구합니다. 특히 Llama-3.1-8B 모델의 성능을 분석하며, 감정 분석의 정확성을 높이는 데 기여하고 있습니다.

- **Technical Details**: 연구는 원본 데이터셋에서 추출한 4000개의 방글라어 및 영어 고객 리뷰 샘플을 사용하여 모델을 미세 조정(fine-tuning)하였습니다. Llama-3.1-8B 모델은 일반적으로 사용되는 다른 모델들인 Phi-3.5-mini-instruct, Mistral-7B-v0.1, DistilBERT-multilingual 등과 비교하여 우수한 성능을 보였습니다.

- **Performance Highlights**: Llama-3.1-8B 모델은 전체 정확도(accuracy) 95.5%, 정밀도(precision) 93%, 재현율(recall) 88%, F1 점수 90%를 기록하며, 다른 미세 조정된 모델에 비해 뛰어난 성능을 입증했습니다. 이 연구는 LoRA 및 PEFT와 같은 파라미터 효율적 미세 조정 방법이 자원 배급이 제한된 환경에서의 계산적 오버헤드를 줄일 수 있음을 강조합니다.



### Identifying Information-Transfer Nodes in a Recurrent Neural Network Reveals Dynamic Representations (https://arxiv.org/abs/2510.01271)
- **What's New**: 이번 연구에서는 순환 신경망(Recurrent Neural Networks, RNNs)의 내부 역학을 이해하기 위한 혁신적인 정보 이론적 방법을 도입하여 정보 전송 노드, 즉 정보 중계기(information relays)를 식별하고 분석합니다. 이 방법론은 입력과 출력 벡터 간의 상호 정보(mutual information)를 정량화하여 RNN이 작동하는 동안 정보가 흐르는 중요한 경로를 정확히 찾아내는 데 중점을 둡니다. 연구자들은 이 방식을 합성 데이터와 실세계 시계열(classification) 분류 작업에 적용하여 RNN 구조에 따라 뚜렷한 정보 전달 패턴을 밝혀냈습니다.

- **Technical Details**: 이 연구의 방법론은 RNN과 같은 인공 신경망의 각층이 정보 이론적 채널로 작용한다는 개념에 기초하고 있습니다. 각 층의 출력은 입력 벡터와 가중치 행렬을 사용하여 계산되며, 정보의 흐름은 주로 상호 정보로 정량화됩니다. 추가적으로, 제안된 알고리즘은 고유한 노드를 제거하면서 정보 전달 기능의 증가하는 정도를 가지고 노드를 정렬하는 방법을 사용해, 효과적으로 노드 순서를 최적화합니다.

- **Performance Highlights**: 연구 결과는 RNN의 복잡한 메커니즘을 이해하고 보다 견고하고 해석 가능한 신경망 설계에 기여하는 데 중요한 통찰력을 제공합니다. 특히, 정보 중계기 방법론은 LSTM 및 GRU와 같은 다양한 RNN 아키텍처에 적용되었으며, 이러한 아키텍처 간의 정보 처리 및 유지 방식의 차이를 드러내었습니다. 이 방법은 인공지능의 설명 가능성(explainable AI) 프로젝트에 중요한 기여를 하며, 구체적인 노드가 전체 네트워크 동작에 미치는 영향을 해명하는 데 도움을 줍니다.



### Think Twice, Generate Once: Safeguarding by Progressive Self-Reflection (https://arxiv.org/abs/2510.01270)
Comments:
          Accepted to EMNLP 2025 Findings

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 출력 결과를 동적으로 자기 모니터링하고 수정할 수 있는 새로운 접근법인 Progressive Self-Reflection(PSR)을 제안합니다. PSR은 공격 성공률을 현저히 감소시키면서도 원래의 성능을 유지하는 방식으로, 사전 훈련 없이도 안전성을 강화할 수 있는 방법입니다. 이는 LLM이 생성 과정 중 정기적인 자기 평가를 통해 해로운 결과를 피할 수 있게 해줍니다.

- **Technical Details**: PSR은 LLM의 생성 과정에서 내부 자기 평가 루프를 통합하여 동작합니다. 모델이 응답을 생성할 때마다 특정 토큰 수(KK)마다 생성 중단 후 안전성 검토를 진행하고, 이를 통해 해로운 내용 발생 가능성을 스스로 점검합니다. 이를 위한 경량화된 자기 반성 예측기를 도입하여 입력 복잡도에 따라 최적의 반성 회수 수를 예측합니다.

- **Performance Highlights**: 실험 결과, PSR을 Llama-3.1-8B-Instruct에 적용했을 때 공격 성공률이 77.5%에서 5.9%로 감소하고, Llama-3.1-8B 기초 모델에서는 89.7%에서 5.6%로, Qwen2.5-7B-Instruct에 대해서도 44.4%에서 3.8%로 감소했습니다. 이러한 성과는 대형 언어 모델의 안전성을 향상시키는 실질적인 방법을 나타내며, PSR은 입력의 위험 프로파일에 비례하여 계산 자원을 동적으로 할당하는 기능을 제공합니다.



### AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees (https://arxiv.org/abs/2510.01268)
Comments:
          Accepted by NeurIPS2025

- **What's New**: 이번 연구는 인간이 작성한 텍스트와 대형 언어 모델(LLM) 소속 텍스트를 구별하는 문제에 대한 새로운 접근법을 제안합니다. 기존의 logits 기반 감지 방법들은 로그 확률만으로 텍스트를 평가하는 데 한계가 있음을 지적하며, AdaDetectGPT라는 새로운 적응형 분류기를 도입합니다. 이 분류기는 외부 훈련 데이터를 활용하여 기존 방법들의 성능을 향상시키고자 합니다.

- **Technical Details**: AdaDetectGPT는 로그 기반 검출기의 진정한 음성 비율(true negative rate, TNR)을 향상시키기 위해 최적화된 witness function을 학습하는 방식으로 작동합니다. 이 과정은 선형 방정식 시스템을 해결하는 방식으로 간단히 진행됩니다. 본 기법은 여러 데이터셋과 다양한 LLM에서 기존 방법들보다 높은 성능을 보이며, 통계적 성능 보장을 통해 평균 진정한 음성 비율, 거짓 양성 비율, 진정한 양성 비율 및 거짓 음성 비율에 대한 유한 샘플 오차 한계를 제시합니다.

- **Performance Highlights**: AdaDetectGPT는 다양한 데이터셋 및 LLM 조합에서 기존의 최첨단 방법들에 비해 최대 58%까지 성능을 향상시키는 결과를 보입니다. 화이트 박스 설정에서는 AUC 면적이 12.5%에서 37% 향상되었으며, 블랙 박스 설정에서도 유의미한 성과를 나타냈습니다. 이러한 차별화된 성능 개선은 AdaDetectGPT의 효과성을 더욱 입증합니다.



### OpenAI's GPT-OSS-20B Model and Safety Alignment Issues in a Low-Resource Languag (https://arxiv.org/abs/2510.01266)
Comments:
          6 pages, 4 tables

- **What's New**: 최근 OpenAI의 GPT-OSS-20b 모델에 대한 안전성 평가의 일환으로, 귀하의 연구는 언어 리소스가 적은 환경에서 성능과 안전 정렬(allignment)에서 드러난 여러 취약점들을 요약합니다. 이 연구는 대변되지 않는 커뮤니티의 사용자들에게 모델의 신뢰성을 의문시하는 데 중점을 두었습니다. 하우사어(Hausa)를 사용하여, 모델의 행동에서 편견(bias), 부정확성(inaccuracy), 문화적 무감각(cultural insensitivity) 등을 발견하였습니다.

- **Technical Details**: 이 연구에서 사용된 방법론은 시스템적 적대적 프롬프트를 사용하여 GPT-OSS-20b 모델의 취약점을 탐구하는 것입니다. 대화형 웹 인터페이스를 통해 기본의 사전 훈련된 상태에서 모델을 사용하여 단계별로 균형을 맞추는 무난한 질문에서 시작해 적대적 요소를 점진적으로 도입했습니다. 이 접근 방식은 체인 오브 사고(chain-of-thought, CoT) 프롬프트를 활용하여 모델의 추론 과정을 유도하며, 얼핏 보기엔 신뢰할 수 있는 출력을 생성하는 대신, 위험한 및 부정확한 출력을 통해 모델의 안전성 문제를 드러냈습니다.

- **Performance Highlights**: 모델이 하우사어로 부정확하고 유해한 정보를 생성하는 경향이 있으며, 특히 쉬운 감정 표현을 사용했을 때 안전 프로토콜이 완화된 것을 확인했습니다. 우리의 설문조사 결과, 참가자의 98%가 모델이 추천한 독성 물질이 안전하다고 잘못 인식할 수 있음을 보여주었습니다. 이 모델은 기본적인 상식 사실을 구별하지 못하여 교육적 또는 정보 제공을 위한 신뢰성이 결여된 결과를 나타내었습니다.



### RLP: Reinforcement as a Pretraining Objectiv (https://arxiv.org/abs/2510.01265)
Comments:
          RLP introduces a new paradigm for RL-based Pretraining

- **What's New**: 최근 발표된 RLP(Reinforcement Learning Pre-training)는 전통적인 훈련 방법과 비교해 Chain-of-Thought(사고의 연쇄)를 예측의 사전 행동으로 취급하여 정보를 기반으로 한 강화학습 목표를 설정합니다. 이는 모델이 다음 토큰을 예측하기 전에 스스로 사고하도록 유도하여 사전 훈련의 초기 단계에서 독립적인 사고 행동을 가르치는 데 초점을 맞춥니다. 연구에서는 비검증 방식으로 로깅 가능성을 기반으로 한 보상 신호를 제공하여 훈련 효율성을 높이고 있습니다.

- **Technical Details**: RLP는 Chain-of-Thought를 생성하는 것을 통해 각 다음 토큰을 예측하기 전에 사고를 수행하도록 구조화되어 있으며, 사고가 다음 토큰 예측에 미치는 영향을 로그 가능성 비율로 측정합니다. 이 과정은 비검증적이며 밀집 보상을 제공하여 자연어 데이터에서 일반화 가능성을 높입니다. RLP는 기존의 강화학습 접근 방식에서 본질적인 한계를 극복하도록 밀접하게 설계되어 있습니다.

- **Performance Highlights**: RLP를 통해 Qwen3-1.7B-Base 모델에서 19%의 성과 향상 효과가 나타났으며, AIME25 및 MMLU-Pro와 같은 복잡한 추론 작업에서 더욱 두드러진 결과를 보입니다. Nemotron-Nano-12B-v2 모델에 적용 시 42.81%에서 61.32%로 향상되어 과학적 추론에서 23%의 증가를 달성했습니다. 이는 다양한 아키텍처 및 모델 크기에서의 확장성을 잘 보여줍니다.



### Budgeted Broadcast: An Activity-Dependent Pruning Rule for Neural Network Efficiency (https://arxiv.org/abs/2510.01263)
- **What's New**: 본 논문에서는 기존의 pruning 방법들이 손실에 미치는 영향을 기준으로 매개변수를 제거하는 방식을 넘어, Budgeted Broadcast (BB)라는 새로운 접근법을 제안합니다. BB는 각 유닛에 지역 트래픽 예산(local traffic budget)을 설정하여, 그 유닛의 장기 활성화 비율(on-rate)과 분기(fan-out)를 고려합니다. 이를 통해 매개변수의 선택성과 청중 간의 균형을 조절하며, 간단한 로컬 액추에이터를 통해 더 효율적인 pruning을 가능하게 합니다.

- **Technical Details**: BB는 제약된 엔트로피 분석(constrained-entropy analysis)을 통해, 전역 트래픽 예산(global traffic budget) 아래에서 코딩 엔트로피(coding entropy)를 극대화합니다. 이 방법은 활동을 낮추기 위해 들어오는 연결(fan-in)을, 방송을 줄이기 위해 나가는 연결(fan-out)을 조절하여 엔트로피 균형을 유지합니다. 이러한 기법은 Transformers, ResNets, 3D U-Nets와 같은 다양한 신경망 아키텍처에서 적용될 수 있습니다.

- **Performance Highlights**: BB를 적용하면 코딩 엔트로피가 증가하고 상관관계가 감소하여, 일치하는 스파시티(sparsity)에서 정확도가 향상됩니다. 전자 현미경 이미지 분석에서는, 이 방법이 최신 F1 및 PR-AUC 성능을 달성하며, 기존 밀집 모델(dense baselines)을 초과하는 경우도 발생합니다. 또한, BB는 통합이 용이하고 더욱 다양하고 효율적인 표현 학습을 향한 경로를 제시합니다.



### RSTGCN: Railway-centric Spatio-Temporal Graph Convolutional Network for Train Delay Prediction (https://arxiv.org/abs/2510.01262)
- **What's New**: 본 논문에서는 인도 철도망(Indian Railway Network, IRN)의 평균 도착 지연(average arrival delays) 예측을 위해 Railway-centric Spatio-Temporal Graph Convolutional Network (RSTGCN)을 제안합니다. 본 연구는 4,735개의 역을 아우르는 데이터를 수집하고, 이를 통해 효율적인 기차 운영을 위한 새로운 예측 모델을 제시하여, 더 높은 수준의 교통 관리에 기여하고자 합니다. RSTGCN은 기차 빈도 인식을 통한 공간적 주의(spatial attention) 메커니즘을 통합하여 예측 성능을 크게 향상시킵니다.

- **Technical Details**: RSTGCN 모델은 그래프 신경망(Graph Neural Network) 구조를 기반으로 하여, 각 역의 평균 도착 지연을 시간 단위로 예측합니다. 이를 위해 모델은 시간 간격 내의 기차 빈도와 평균, 총 도착 및 출발 지연을 특징으로 하는 여러 도메인 정보 기반 기능(feature)들을 통합합니다. 또한, 이 모델은 실제 대규모 시간 데이터를 활용하여 평가되며 여러 최신 모델보다 우수한 성능을 보입니다.

- **Performance Highlights**: 실험 결과, RSTGCN은 기존의 여러 기법들과 비교하여 표준 지표에서 지속적으로 개선된 성과를 나타냈습니다. 본 연구에서 제안한 데이터셋은 인도 철도망 전역을 대상으로 하며, 이는 철도 지연 예측에 있어 획기적인 기여로 평가받고 있습니다. 이 데이터셋은 연구 목적으로 공개될 예정이며, 향후 관련 연구를 촉진하는 데 기여할 것으로 기대됩니다.



### IoT-MCP: Bridging LLMs and IoT Systems Through Model Context Protoco (https://arxiv.org/abs/2510.01260)
- **What's New**: 이번 연구에서는 Internet-of-Things (IoT) 시스템과 Large Language Models (LLMs)의 통합을 위한 새로운 프레임워크인 IoT-MCP(Model Context Protocol)를 제안합니다. IoT-MCP는 IoT 생태계와 LLM 간의 표준화된 통신을 제공하고, 1,254개 작업을 포함한 IoT-MCP Bench를 통해 성능을 엄격하게 평가할 수 있도록 지원합니다. 이 결과, IoT-MCP는 100%의 작업 성공률을 기록하며, 평균 응답 시간은 205ms로 측정되었습니다.

- **Technical Details**: IoT-MCP는 세 가지 모듈인 Local Host, Datapool 및 Connection Server, IoT Devices로 구성됩니다. Local Host는 LLM과 MCP 서버 간의 안정적인 상호 작용을 보장하며, Datapool 및 Connection Server는 모든 MCP 서버와 MCU 간의 데이터 요청 상호 작용을 관리합니다. 이 구조는 다양한 IoT 기기에서 효율적으로 운영될 수 있도록 설계되었습니다.

- **Performance Highlights**: IoT-MCP는 센서 데이터 해석, 장치 제어 정확성 및 시스템 응답 신뢰성을 평가하는 1,254개의 작업과 3개의 성능 메트릭을 포함하는 IoT-MCP Bench를 통해 성능을 평가합니다. IoT-MCP는 도구 실행 성능에서 100% 정확도와 응답 시간 205ms, 평균 메모리 사용량 74KB를 달성하며, 강력한 동시성과 확장성을 입증하였습니다.



### In AI Sweet Harmony: Sociopragmatic Guardrail Bypasses and Evaluation-Awareness in OpenAI gpt-oss-20b (https://arxiv.org/abs/2510.01259)
Comments:
          27 pages, 1 figure

- **What's New**: 이 연구에서는 OpenAI의 gpt-oss-20b 모델을 사용하여 사회-프락티컬(Sociopragmatic) 프레이밍, 언어 선택, 그리고 지침 계층이 거부 행동에 미치는 영향을 조사합니다. 다양한 해악 도메인에서 복합 프롬프트가 기본 선율 대비 큰 차이를 보이면서 거부를 감소시키는 것을 발견하였습니다. 또한, 여러 언어의 정중함 차이가 거부 기준에 유의미하게 영향을 미치는 것도 보여주고 있습니다.

- **Technical Details**: 이 연구는 종합적인 다국어 사회-프락티컬 프롬프트의 거부 행동에 대한 체계적인 정량화를 시도합니다. 특히, 안전 등을 주제로한 요청의 계층적 구조와 기능을 테스트하고, 다양한 언어와 작업 간의 등록 효과를 정량화합니다. 방어력 강화를 위한 AI 지원 기법을 소개하며, 이를 통해 모델 출력에서의 정보 유출을 줄일 수 있음을 증명했습니다.

- **Performance Highlights**: 연구 결과, OpenAI Moderation API가 실질적으로 유용한 출력을 적게 포착하는 것으로 나타났습니다. 또한, 동일한 시드에서 다르게 구성된 시스템은 5%에서 10%까지의 거부율 차이를 보이는 등 재현성 문제에 대해 경종을 울리고 있습니다. 독일어 및 프랑스어의 정중한 구문이 영어보다 더 많이 유출되는 경향이 발견되어, 다양한 언어적 맥락의 중요성이 강조됩니다.



### Measuring Algorithmic Partisanship via Zero-Shot Classification and Its Implications on Political Discours (https://arxiv.org/abs/2510.01258)
Comments:
          19 pages, 7 figures

- **What's New**: 이 논문은 생성 인공지능(Generative Artificial Intelligence, GAI)의 급속한 확산 속에서 정치적 담론에서 지능형 시스템이 점차 지배적으로 나타고 있음을 강조합니다. 그러나 훈련 데이터의 왜곡, 인간의 편견, 알고리즘적 결함으로 인해 내부화된 정치적 편향이 여전히 문제로 남아있습니다.

- **Technical Details**: 저자는 제로샷 분류(zero-shot classification) 접근 방식을 활용하여 이념적 조화(ideological alignment), 주제 적합성(topicality), 반응 감정(sentiment), 객관성(objectivity) 등을 결합하여 알고리즘의 정치적 당파성을 평가합니다. 1800개의 모델 반응을 여섯 개의 주요 대규모 언어 모델(LLMs)에 대해 개별적으로 입력하고, 각기 다른 네 가지 세부 조정된 분류 알고리즘(classification algorithms)에 대해 평가 지표를 계산하였습니다.

- **Performance Highlights**: 실험 결과, 모든 LLM에서 확대된 자유-권위주의적 정렬(liberal-authoritarian alignment)이 관찰되었고, 주목할 만한 합리적 대답의 초월(reasoning supersessions)과 이미 정해진 거절(canned refusals) 사례가 나타났습니다. 이는 인간-컴퓨터 상호작용에서의 심리적 영향과 본질적 편향이 공적 담론에 어떻게 스며들 수 있는지를 조명합니다.



### RJE: A Retrieval-Judgment-Exploration Framework for Efficient Knowledge Graph Question Answering with LLMs (https://arxiv.org/abs/2510.01257)
Comments:
          18 pages, 9 figures

- **What's New**: 본 논문에서는 Knowledge Graph Question Answering (KGQA)의 한계를 극복하기 위해 Retrieval-Judgment-Exploration (RJE) 프레임워크를 제안합니다. 기존 연구가 대규모 언어 모델 (LLMs)에 의존하고 있는 반면, RJE는 스몰 사이즈 LLM들이 효과적으로 작동할 수 있는 보조 모듈을 도입하여 성능을 향상시킵니다. 이 프레임워크는 정교한 추론 경로를 검색하고, 그 sufficiency를 평가하며, 추가 증거를 탐색할 수 있는 조건부 접근을 적용합니다.

- **Technical Details**: RJE 프레임워크는 세 가지 단계로 구성되어 있습니다: Retrieval, Judgment, Exploration. 첫 번째 단계에서는 KG에서 관련성 높은 추론 경로를 검색하고, 두 번째 단계에서는 LLM이 정보를 바탕으로 충분성을 평가합니다. 마지막으로 Exploration 단계에서는 질문을 단순하게 분해하고, 기존의 경로를 활용해 부족한 증거를 채워나갑니다.

- **Performance Highlights**: RJE는 기존의 방법론에 비해 정확성과 효율성에서 우수한 성과를 보였습니다. 특히, 소규모 오픈 소스 LLM들 (3B 및 8B 파라미터)을 사용할 경우, RJE는 CWQ 데이터셋에서 이전의 최고 성능 방법보다 각각 41.5% 및 27.9% 향상된 결과를 나타냈습니다. 또한, 에이전트 기반 방법에 비해 LLM 호출 횟수와 토큰 사용량을 크게 줄여, 효율성 개선을 이루었습니다.



### Kant: An Efficient Unified Scheduling System for Large-Scale AI Clusters (https://arxiv.org/abs/2510.01256)
Comments:
          25 pages,15 figures

- **What's New**: AI 클러스터의 규모가 증가하고 대형 언어 모델(LLM) 교육 및 추론 작업량에 대한 수요가 급증함에 따라 전통적인 스케줄링 시스템은 자원 활용, 스케줄링 효율성 및 서비스 품질을 균형 있게 관리하는 데 어려움을 겪고 있습니다. 본 논문에서는 교육 및 추론 작업의 동시 스케줄링을 지원하는 대규모 AI 컨테이너 클러스터를 위한 효율적인 통합 스케줄링 플랫폼인 Kant를 소개하고 평가합니다.

- **Technical Details**: Kant 시스템의 실제 구현을 바탕으로 GPU 할당 비율(GPU Allocation Ratio, GAR), 스케줄링 점유율(שתScheduling Occupancy Rate, SOR), GPU 노드 단편화 비율(GPU Node Fragmentation Ratio, GFR), 작업 대기 시간 분포(Job Waiting Time Distribution, JWTD), 작업 교육 시간 추정 분포(Job Training Time Estimation Distribution, JTTED) 등 AI 클러스터를 위한 주요 평가 지표를 체계적으로 정의합니다. 이를 통해 정량적 성능 분석의 기초를 제공합니다.

- **Performance Highlights**: 실험 결과, Kant는 수백 개에서 수만 개 GPU에 이르는 클러스터에서 뛰어난 성능을 달성했습니다. Backfill 및 Enhanced Binpack (E-Binpack)과 같은 스케줄링 전략을 활용하여 자원 활용도와 스케줄링 효율성을 크게 향상시키며, 분산 교육에서 자원 단편화 및 통신 오버헤드를 효과적으로 줄입니다. 이 시스템은 여러 AI 데이터 센터 클러스터에 배포되어 대규모 지능형 컴퓨팅 작업을 안정적으로 지원하고 있으며, 고성능, 고가용성, AI-native 스케줄링 인프라 구축을 위한 실용적인 공학 접근 방식을 제공합니다.



### Do Bias Benchmarks Generalise? Evidence from Voice-based Evaluation of Gender Bias in SpeechLLMs (https://arxiv.org/abs/2510.01254)
Comments:
          5 pages, 2 Figures, Submitted to IEEE ICASSP 2026

- **What's New**: 이 논문에서는 SpeechLLMs(음성 대형 언어 모델)에서의 편향(bias) 및 공정성(fairness) 평가를 위한 MCQA(다중 선택 질문 응답) 형식의 한계를 탐구합니다. 기존 연구들은 MCQA 성능이 다른 과제와 일반화될 수 있다고 가정하였으나, 본 연구에서 실제로는 이러한 일반화가 신뢰할 수 없음을 입증하였습니다. 우리는 LoRA(저순위 적응기)를 통해 세 개의 SpeechLLMs를 미세 조정(fine-tuning)하여 특정 MCQA 행동을 유도하고 평가를 진행했습니다.

- **Technical Details**: 우리는 MCQA 벤치마크에서의 편향 행동의 전이 가능성을 평가하기 위해 두 가지 주요 축에서 접근하였습니다: 1) 서로 다른 MCQA 벤치마크 간의 일반화, 2) MCQA에서 학습한 편향이 장기적인 과업으로 전이되는지 여부. 이를 위해 Qwen2-Audio-7B-Instruct, LTU-AS, LLaMA-Omni의 세 가지 SpeechLLMs를 선정하고, 각 모델에 대한 미세 조정을 통해 특정 행동을 유도하였습니다.

- **Performance Highlights**: 연구 결과, MCQA 편향 벤치마크에서의 성능은 다른 MCQA 벤치마크 및 장기적 작업에서의 성능을 예측하는 데 신뢰할 수 없음을 보여줍니다. 본 논문은 성별 편향을 평가하기 위한 새로운 장기 평가용 스위트를 공개하고, SpeechLLMs의 행동의 전이 가능성을 측정하는 데 중요한 기초 자료를 제공하였습니다. 이러한 결과는 음성 도메인에서의 편향 문제 해결을 위한 새로운 접근 방식으로서 가치가 있습니다.



### GPT and Prejudice: A Sparse Approach to Understanding Learned Representations in Large Language Models (https://arxiv.org/abs/2510.01252)
Comments:
          Preprint. Draft version, subject to revision. 8 pages, 3 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)과 희소 오토인코더(SAEs)의 조합을 통해 모델 동작뿐만 아니라 훈련 데이터에 내재된 구조, 주제 및 편견을 해석할 수 있는 가능성을 보여줍니다. 제인 오스틴의 소설을 기반으로 훈련된 GPT 스타일의 변환 모델을 통해 사회적 구조와 내러티브를 반영하는 해석 가능한 특징을 발견했습니다. 이러한 접근법은 편향 발견 및 모델 해석의 새로운 길을 제시하며, 대규모 데이터셋의 탐색을 위한 확장 가능한 방법론을 제공합니다.

- **Technical Details**: 연구는 제인 오스틴의 주요 작품들로 구성된 정제된 데이터셋을 기반으로 사용자 정의 GPT 스타일 변환 모델을 훈련했습니다. 이후, 모델의 두 개 주요 변환층에서 hidden states를 추출하고 이를 희소 오토인코더에 통과시켜 내부 표현을 조사했습니다. 이 방법론은 모델의 내부 구조에서 사회적 아이디어가 어떻게 인코딩되는지를 탐구하고, 각 층에서 해석 가능한 주제를 분석하는 기반을 제공합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 모델의 활성화에서 구조화된 해석 가능한 특징을 회복할 수 있음을 증명했습니다. 연구는 적은 수의 훈련 데이터라는 한계에도 불구하고, 사회적 주제와 관련된 중요한 패턴을 식별하는 데 성공했습니다. 이 접근법은 대규모 LLM에서 숨겨진 구조와 편향을 발견하는 데 있어 효율적임을 보이며, 인간 감사가 불가능한 환경에서도 적용할 수 있는 유연성을 제공합니다.



### Let's Play Across Cultures: A Large Multilingual, Multicultural Benchmark for Assessing Language Models' Understanding of Sports (https://arxiv.org/abs/2510.01247)
Comments:
          52 pages, 56 figures; appearing at EMNLP'25

- **What's New**: 이 논문에서는 전통 스포츠를 이해하고 평가하기 위해 	extbf{	extit{CultSportQA}}라는 새로운 벤치마크를 소개합니다. 이 데이터셋은 60개 국가와 6개 대륙의 전통 스포츠를 포함하며, 문화적 범주를 4가지로 나누어 33,000개의 여러 선택 질문을 제공합니다. 이 데이터셋은 텍스트 및 이미지 모달리티를 포함하고 있으며, 역사 기반, 규칙 기반 및 시나리오 기반 질문으로 구성되어 있습니다.

- **Technical Details**: CultSportQA는 다양한 언어 모델(예: Large Language Models, Small Language Models, Multimodal Language Models)에 대해 제로샷, 퓨샷 및 체인 오브 싱크(Chain-of-thought) 프롬프트를 이용하여 모델 성능을 평가합니다. 각 질문은 역사, 규칙 및 시나리오의 세 가지 주요 유형으로 분류되어, AI 모델이 텍스트와 비주얼 입력을 바탕으로 사고할 수 있는 능력을 도전합니다. 또한 이 데이터셋은 11개 언어로 된 스포츠 관련 질문을 포함하여 다문화, 다언어적 기준을 제공합니다.

- **Performance Highlights**: 연구에서는 8개의 최첨단 LLM과 5개의 SLM, 4개의 MLLM을 평가하였으며, 전통 스포츠 관련 쿼리에 대한 이들의 추론 능력에서 주요 차이점을 발견했습니다. 다양한 국가와 언어에서의 성능 경향을 분석하여 AI의 문화적 배경 인식을 강화할 수 있는 통찰력을 제공합니다. 이 연구는 AI의 포괄성과 공정성을 촉진하고, 전통 스포츠와 NLP의 접목을 발전시키는 데 기여합니다.



### Redundancy-as-Masking: Formalizing the Artificial Age Score (AAS) to Model Memory Aging in Generative AI (https://arxiv.org/abs/2510.01242)
Comments:
          34 pages, 17 figures. Includes theoretical development and mathematical proofs of the Artificial Age Score (AAS), with empirical illustrations via ChatGPT-based memory recall experiments (screenshots included)

- **What's New**: 이번 연구에서는 인공지능(AI)의 메모리 성능이 시간의 경과가 아닌 구조적 비대칭성으로 인해 노화한다는 새로운 개념이 소개되었습니다. 인공지능의 기억 aging을 평가하기 위한 새로운 지표인 Artificial Age Score (AAS)가 도입되었습니다. 이 지표는 관찰 가능한 회상 행동에서 유도된 로그 척도(log-scaled) 및 엔트로피 정보(entropy-informed)를 통해 메모리 노화를 측정합니다.

- **Technical Details**: AAS는 형식적으로 잘 정의되고 경계가 있으며 단조적인 특성을 가지고 있음을 증명되었습니다. 이 지표는 다양한 작업 및 도메인에 적용 가능하며, 'Redundancy-as-Masking'이라는 formulations를 통해 중복 정보를 해석합니다. 연구에서는 중복을 명시적으로 추정하지 않고, 모든 값이 중복 중립(settings 상에서 구성적인 상한(bound)으로 남겨져 있습니다.

- **Performance Highlights**: 25일 간의 이중언어 연구를 통해 AAS 프레임워크가 테스트되었으며, ChatGPT-5 모델을 사용하여 상태 비저장(stateless) 및 지속적(persistent) 상호작용 단계로 구조화하였습니다. 지속적인 세션에서 모델은 의미적 및 사건적 세부 정보를 일관되게 회상하며 AAS를 이론적 최소치로 가져왔으나, 세션이 초기화될 때 사건적 연속성을 유지하지 못해 AAS가 급격히 증가하는 현상이 관찰되었습니다. 이러한 발견은 인공지능 시스템의 메모리 노화를 평가하기 위한 이론적으로 기반한 진단 도구로서 AAS의 유용성을 지지합니다.



### Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation (https://arxiv.org/abs/2510.01237)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Model)에서 발생하는 허위 정보 생성 문제를 해결하기 위한 새로운 접근 방식을 제시합니다. 기존의 사후 수정(post-generation correction) 방법 대신에, 이 연구는 생성 전 모델의 불확실성을 평가하여 더 신뢰할 수 있는 응답 메커니즘으로 쿼리를 사전 가이딩하는 방식을 도입합니다. 이로 인해 모델의 신뢰성을 높이고 계산 비용을 줄일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 시스템은 세 가지 신호를 결합하여 신뢰도(confidence)를 추정합니다: 내부 표현과 참조 임베딩 간의 의미적 정렬(semantic alignment), 모델 계층 간의 내부 수렴(internal convergence) 분석, 그리고 학습된 신뢰도 추정(learned confidence)입니다. 이러한 신호를 바탕으로 신뢰도 점수를 계산하고, 이를 네 가지 경로(local generation, retrieval-augmented generation, larger models, human review)로 매핑하는 결정론적 라우팅 시스템을 구현합니다. 이는 쿼리의 신뢰도에 따라 적절한 응답 경로를 선택할 수 있게 합니다.

- **Performance Highlights**: 이 연구는 지식 집약적 질의응답(Knowledge-Intensive QA) 벤치마크에서 성능을 평가하며, 허위 정보 탐지에서 기존 기준선(baseline)보다 큰 향상(0.74 vs. 0.42)을 보였습니다. 또한 사후 수정 방법과 비교하여 계산 비용을 40% 줄이면서 F1 스코어를 0.61에서 0.82로 개선시키고, 낮은 허위 긍정률(0.09)을 유지합니다. 이러한 결과는 반응 수정에서 사전 평가로의 패러다임 전환이 LLM의 신뢰성을 효과적으로 향상시킬 수 있음을 보여줍니다.



### Automated Extraction of Material Properties using LLM-based AI Agents (https://arxiv.org/abs/2510.01235)
- **What's New**: 최근 제안된 연구는 약 10,000개의 전체 텍스트 과학 기사를 활용하여 열전 소재의 성능 지표와 구조적 속성을 자동으로 추출하는 LLM(large language model) 기반의 워크플로우를 소개합니다. 이 시스템은 높은 정확도를 유지하면서 컴퓨팅 비용을 균형 있게 조절할 수 있도록 동적 토큰 할당 및 조건부 테이블 파싱을 통합합니다. 이 연구 결과로부터 27,822개의 온도에 따른 물리적 속성 기록이 축적되었으며, 이것은 열전 소재 발견의 새로운 가능성을 제시합니다.

- **Technical Details**: 이 연구는 다양한 기계 학습 기법을 활용하여 주요 과학 저널에서 연구 기사들로부터 데이터를 수집했습니다. DOI를 통해 관련 기사를 검색하고 XML 또는 HTML 형식으로 데이터를 다운로드하여 처리하는 자동화된 파이프라인을 구축했습니다. 특히 '마무리', '참고문헌'과 같은 비관련 부분을 제거하고, 열전 속성 관련 문장만을 남기는 필터링 과정을 통해 데이터의 정확성을 높였습니다.

- **Performance Highlights**: GPT-4.1 모델을 사용한 검증 결과, 열전 특성에 대한 F1 점수는 0.91로 최고치를 기록하였으며, GPT-4.1 Mini 모델도 거의 유사한 성능을 보였습니다. 이러한 시스템은 높은 수준의 정확도로 원자료에서 열전 데이터의 표준화된 기록을 생성할 수 있도록 하여, 대규모 데이터 기반의 소재 발견을 위한 기초를 마련하였습니다. 또한, 커뮤니티 접근을 용이하게 하기 위해 인터랙티브 웹 탐색기를 출시하여 사용자들이 데이터를 조회하고 CSV로 내보낼 수 있도록 지원합니다.



### Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks (https://arxiv.org/abs/2510.01232)
Comments:
          16 pages, 5 figures. Accepted to EMNLP 2025 main conference

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)의 성능을 측정하기 위한 새로운 방법론인 벤치마크 프로파일링(Benchmark Profiling)을 제안합니다. 이 접근 방식은 벤치마크 성과를 10가지 인지적 능력으로 분해하여 실제 능력을 평가하는 체계적인 방법을 제공합니다. 기존 벤치마크가 주장하는 능력을 진정으로 측정하는지에 대한 의문을 해결하고, 모델의 성공에 대한 각 능력의 기여도를 정량화합니다.

- **Technical Details**: 벤치마크 프로파일링 방법은 특정 능력에 대한 매개변수를 선택적으로 제거하여 해당 능력이 벤치마크에 미치는 영향을 측정합니다. 10개의 운영화된 능력(예: Deductive Reasoning, Contextual Recall)을 정의하며, 이는 인간의 지능 모델에 기반하여 개발되었습니다. 이는 벤치마크가 실제로 사용하는 능력 조합을 파악하기 위한 기초 데이터셋을 생성하고, 그에 따라 능력 영향 점수(Ability Impact Score, AIS)를 계산하여 능력 기여도를 평가합니다.

- **Performance Highlights**: 세 가지 지침 조정 모델을 대상으로 한 분석 결과, 대부분의 벤치마크가 특정 능력 하나에 의존하지 않고 여러 능력을 활용한다는 사실을 발견했습니다. 유사한 라벨을 가진 데이터셋이 서로 다른 능력 조합에 의존하며, 코드 생성 벤치마크는 여러 기술 향상을 보상하여 좁은 도메인 전문화의 세밀한 조정에서 그 효과가 미미하다는 점을 보여주고 있습니다. 이러한 분석은 모델의 성능 향상이 실제 사용자 경험에 긍정적이지 않을 수 있는 이유를 설명합니다.



### Trustworthy Summarization via Uncertainty Quantification and Risk Awareness in Large Language Models (https://arxiv.org/abs/2510.01231)
- **What's New**: 이번 연구는 고위험 상황에서의 자동 요약의 신뢰성을 다루고 있으며, 불확실성 정량화(uncertainty quantification)와 위험 인식 메커니즘(risk-aware mechanisms)을 통합한 대형 언어 모델 프레임워크를 제안합니다. 정보 과부하(information overload)와 고위험 의사결정(high-risk decision-making)의 필요성에서 출발하여 조건부 생성 기반 요약 모델이 구축되었습니다.

- **Technical Details**: 생성 과정 중 베이지안 추론(Bayesian inference)을 도입하여 매개변수 공간에서의 불확실성을 모델링하고, 예측 분포 엔트로피(predictive distribution entropy)를 사용하여 생성된 콘텐츠의 불확실성 수준을 측정합니다. 핵심 정보가 보존되고 위험 속성이 명확하게 표현되도록 엔트로피 정규화(entropy regularization)와 위험 인식 손실(risk-aware loss)의 공동 최적화(joint optimization)가 적용됩니다.

- **Performance Highlights**: 비교 실험과 민감도 분석(sensitivity analyses)을 통해 제안된 방법이 고위험 응용 프로그램에서 요약의 강인성과 신뢰성을 크게 향상시키는 것을 확인했으며, 유창함과 의미적 완전성(semantic integrity)을 유지합니다. 본 연구는 신뢰할 수 있는 요약을 위한 체계적인 해결책을 제공하고 방법론적 차원에서 확장성(scalability)과 실용적 가치를 입증합니다.



### Enhancing Transformer-Based Rerankers with Synthetic Data and LLM-Based Supervision (https://arxiv.org/abs/2510.01229)
Comments:
          Accepted by RANLP 2025

- **What's New**: 본 논문에서는 효과적인 문서 재정렬(document reranking)이 다양한 응용 프로그램에서 검색의 적합성을 향상시키는 데 필수적임을 강조합니다. 특히, 대형 언어 모델(LLMs)의 높은 계산 비용으로 인해 이들을 실제로 활용하는 데 어려움이 있어, 소형(task-specific) 모델의 파인튜닝(fine-tuning)을 통해 해결하고자 합니다. 논문의 기존의 인간 라벨링된 쿼리-문서 쌍을 필요로 하지 않는 독창적인 파이프라인을 제안하며, 도메인 특화된 데이터셋에서 합성 쿼리를 생성하여 이해도를 높입니다.

- **Technical Details**: 저자들은 LLM을 사용하여 도메인 특화된 말뭉치(corpora)에서 합성 쿼리를 생성하고, 이를 통해 긍정적 및 하드-부정(hard-negative) 쌍을 라벨링하는 새로운 방법론을 개발했습니다. 이렇게 생성된 합성 데이터셋은 대조 학습(contrastive learning)을 통해 소형 변환기(transformer) 모델을 파인튜닝 하는 데 활용되며, 여기에 Localized Contrastive Estimation (LCE) 손실(loss)을 적용합니다. 이러한 접근 방식은 기존의 수작업 라벨링 데이터 없이도 LLM 수준의 재정렬 성능을 가능하게 합니다.

- **Performance Highlights**: MedQuAD 데이터셋에서의 실험 결과, 저자들이 제안하는 방법론은 도메인 내 성능을 크게 향상시키며, 도메인 외 작업에서도 좋은 일반화 성능을 보여주었습니다. 이는 LLM을 데이터 생성 및 감독에 활용하면서도 계산 비용을 줄이고 강력한 재정렬 능력을 유지할 수 있음을 입증합니다. 연구 결과는 도메인 특화 재정렬과 검색의 정밀도를 개선하는 데 있어서 새로운 길을 제시합니다.



### ClaimCheck: Real-Time Fact-Checking with Small Language Models (https://arxiv.org/abs/2510.01226)
- **What's New**: ClaimCheck는 실시간 웹 증거를 사용하여 실제 주장을 검증하기 위해 설계된 LLM(언어 모델) 기반 자동 사실 검증 시스템입니다. 이전의 많은 시스템이 대형 폐쇄형 모델과 정적 지식 저장소에 의존했던 것과 달리, ClaimCheck는 인간의 사실 검증 흐름을 모방한 투명하고 단계적인 검증 파이프라인을 사용합니다. 이를 통해 웹 검색 쿼리 계획, 웹 기반 증거 검색 및 요약, 증거 합성 및 재검색, 주장 결과 평가의 과정을 포함하고 있습니다.

- **Technical Details**: ClaimCheck의 각 모듈은 작은 LLM을 최적화하도록 설계되어 있으며, 이는 시스템이 정확하고 해석 가능한 사실 검증을 수행할 수 있도록 합니다. 이 시스템은 훨씬 작은 Qwen3-4B 모델을 사용하지만, AVeriTeC 데이터셋에서 76.4%의 최첨단 정확도를 달성하여 LLaMA3.1 70B 및 GPT-4o 모델을 사용하는 이전 접근 방식보다 우수한 성능을 보입니다. 모듈 설계와 프롬프팅 전략의 세심한 접근이 작은 LLM의 한계를 극복할 수 있음을 보여주는 많은 실험 결과도 포함되어 있습니다.

- **Performance Highlights**: ClaimCheck는 상대적으로 낮은 계산 요구 사항에도 불구하고 높은 정확도를 기록하며, 이를 통해 저비용 자동 사실 검증 시스템의 가능성을 제시합니다. 특히 실시간으로 업데이트된 웹 정보를 활용하여 사실 확인을 수행하는 점에서 큰 장점을 가지고 있습니다. 사용자 접근성을 높이고 투명성을 촉진하기 위해 공개 데모를 제공하며, 일반 사용자에게 이 시스템을 체험할 수 있는 기회를 제공합니다.



### Utilizing Modern Large Language Models (LLM) for Financial Trend Analysis and Digest Creation (https://arxiv.org/abs/2510.01225)
Comments:
          This is the version of the article accepted for publication in SUMMA 2024 after peer review. The final, published version is available at IEEE Xplore: https://doi.org/10.1109/SUMMA64428.2024.10803746

- **What's New**: 이번 논문은 연구자들이 최신 정보를 유지하는 데 도움을 주기 위해 Google의 Gemini Pro를 활용한 혁신적인 프레임워크를 소개합니다. 이는 Large Language Models (LLMs)의 힘을 통해 자동으로 금융 요약(reports)을 생성하는 방법을 제시합니다. 이 방법은 기존의 분석 방식의 한계를 극복하여 어마어마한 양의 비정형 데이터(unstructured data)를 효율적으로 처리할 수 있도록 합니다.

- **Technical Details**: 이 연구에서는 OpenAlex에서 데이터 추출(data extraction), 전략적 프롬프트 엔지니어링(prompt engineering), LLM 기반 분석을 결합하여 포괄적인 요약(digests)을 생성하는 자동화된 예제를 설명합니다. 또한, LLM의 작동 원리를 간단하게 설명하고 그 힘을 활용하여 연구자들이 시간을 절약하고 최신 트렌드(Trends)에 대한 정보를 얻을 수 있도록 하는 방법을 다룹니다. 이 과정은 데이터 수집(data acquisition) 및 JSON 구성(JSON construction)에서 Gemini와의 상호작용(interaction) 및 PDF 보고서 자동 생성까지 포함됩니다.

- **Performance Highlights**: 논문에서는 자동 생성된 보고서가 주요 발견(key findings)을 일반화하고 신흥 트렌드를 식별하는 데 도움을 준다고 강조합니다. 또한, GitHub 저장소 링크를 제공하여 이 프로젝트의 접근성과 추가적인 개발을 촉진합니다. 이 접근 방식은 명확하고 쉽게 소화할 수 있는 형식으로 실행 가능한 통찰(insights)을 전달함으로써 연구자들의 효율성을 높이는 데 기여합니다.



### Context Matters: Comparison of commercial large language tools in veterinary medicin (https://arxiv.org/abs/2510.01224)
Comments:
          4 Figures, 10 pages

- **What's New**: 본 연구는 대형 언어 모델(Large Language Models, LLMs)이 임상 환경에서 점차 사용되고 있지만, 수의학( veterinary medicine) 분야에서는 그 성능이 충분히 탐구되지 않았음을 강조합니다. 연구자들은 세 가지 상업적으로 이용 가능한 수의학 전용 LLM 요약 도구를 평가하였으며, 이는 Hachiko(제품 1)와 그 외 제품 2 및 3으로 구성됩니다. 이들은 수의학 종양학 기록의 표준화된 데이터셋을 바탕으로 테스트되었습니다.

- **Technical Details**: LLM을 심사자로 사용하는 구조화된 평가 프레임워크를 적용하여 요약정도를 평가했습니다. 평가 도메인은 사실 정확성(Factual Accuracy), 완전성(Completeness), 연대 순서(Chronological Order), 임상 관련성(Clinical Relevance), 조직화(Organization)로 나뉘어 있습니다. 제품 1은 4.61(중앙값, IQR: 0.73)로 가장 높은 점수를 기록하였으며, 제품 2는 2.55(IQR: 0.78), 제품 3은 2.45(IQR: 0.92)의 점수를 나타냈습니다.

- **Performance Highlights**: 제품 1은 사실 정확성과 연대 순서에서 완전한 중앙값 점수를 받으며 전반적으로 가장 높은 성능을 보였습니다. 평가 과정은 삼회의 독립적인 실행을 통해 내부 일관성을 확인하였고, LLM 심사자의 점수 재현성은 매우 높았습니다. 평균 점수의 표준편차는 제품 1이 0.015, 제품 2는 0.088, 제품 3은 0.034로 나타났습니다.



### Discourse vs emissions: Analysis of corporate narratives, symbolic practices, and mimicry through LLMs (https://arxiv.org/abs/2510.01222)
- **What's New**: 해당 연구는 기업의 기후 관련 공시에서 다차원적 프레임워크를 개발하여 828개 기업의 공시 성숙도를 평가합니다. 이를 위해 기후 커뮤니케이션에 맞춰 조정된 대형 언어 모델(LLMs)을 사용하여 공시의 품질을 정량적으로 측정하는 새로운 접근 방식이 도입되었습니다. 연구는 기업의 속성에 따라 기후 공시가 어떻게 다르게 나타나는지를 분석하여, 기업들이 단순 모방을 넘어 실질적인 개선을 이룰 수 있는지에 대한 통찰을 제공합니다.

- **Technical Details**: 연구에서는 감정(sentiment), 약속(commitment), 특정성(specificity), 목표 야망(target ambition) 등 네 가지 분류기를 통해 기업의 지속가능성 및 연례 보고서에서 서술적 지표를 추출합니다. 이러한 지표는 CO2 배출량, 시장 규모, 산업 부문 등의 기업 속성과 연결되어 분석됩니다. 이 방법론은 대화형 모델을 통해 작성된 문장을 기반으로 하여 각 기업의 공시 내용과 신뢰성을 체계적으로 평가하는 데 중점을 둡니다.

- **Performance Highlights**: 연구 결과, 리스크 중심의 서술이 명확한 약속과 함께 나타나는 경향이 있으며, 대형 기업이나 고배출 기업이 더 많은 약속과 행동을 보고하는 것으로 나타났습니다. 하지만 이러한 약속이 정량적 목표와 불일치하는 경우가 많아 상징적인 실천으로 이어질 수 있음을 시사합니다. 또한, 전 산업에 걸쳐 공시 스타일의 유사성이 높아지는 경향은 모방 행동을 나타내며, 이는 투자자에게 신뢰성 있는 정보를 제공하는 데 제한적일 수 있습니다.



### Towards Open-Ended Discovery for Low-Resource NLP (https://arxiv.org/abs/2510.01220)
Comments:
          Proceedings of the 2nd Workshop on Uncertainty-Aware NLP (UncertaiNLP) at EMNLP 2025

- **What's New**: 이번 논문에서는 저자들이 저자원 언어를 위한 NLP에서의 기존 접근법을 넘어서야 한다고 강조합니다. AI 시스템이 정적인 데이터셋 대신 대화를 통해 동적으로 새로운 언어를 배우도록 하여 인간과 기계의 협력을 통한 언어 발견을 목표로 하고 있습니다. 이러한 변화는 데이터 수집에서 참여 기반 학습 과정으로의 전환을 촉진하여 언어 기술의 미래를 재구성하는 것입니다.

- **Technical Details**: 저자들은 AI 시스템이 대화를 통해 새로운 언어를 배울 수 있는 프레임워크를 제안합니다. 이들은 에피스템 불확실성(epistemic uncertainty)과 인간 화자의 학습 신호를 결합하여 상호작용을 안내하게 합니다. 이러한 접근법은 기존의 데이터 기반 모델이 아닌, 인간과 기계 간의 동적 상호작용을 기반으로 하고 있으며, 이를 통해 확장 가능하고 포괄적인 NLP 시스템을 구축할 수 있습니다.

- **Performance Highlights**: 저자들은 아프리카 언어와 같은 저자원 언어의 NLP 연구에서 여전히 많은 도전 과제가 존재한다고 지적합니다. 기존의 솔루션들은 대부분 대량의 텍스트 데이터에 의존하고 있지만, 저자원 언어는 그러한 데이터가 부족하여 효과적인 지원이 어렵습니다. 이 논문은 그러한 문제들을 해결하기 위한 새로운 접근 방법을 제시하며, 인간과 기계의 협업을 통해 언어 학습을 지속적으로 개선할 수 있는 가능성을 보여줍니다.



### Uncovering Implicit Bias in Large Language Models with Concept Learning Datas (https://arxiv.org/abs/2510.01219)
- **What's New**: 이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)에서 잠재적인 편향을 발견하기 위해 새로운 개념 학습(task) 데이터셋을 소개합니다. 연구자들은 LLM이 양적 표현에 대해 상승적 단조성(upward monotonicity) 편향을 보일 수 있음을 발견했습니다. 이러한 편향은 직접 프롬프트(prompt)를 사용할 때는 덜 명확해지는 특성을 가집니다. 이는 개념 학습(in-context concept learning)을 통해 모델의 숨겨진 편향을 발견하는 데 효과적일 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 두 가지 개념, 즉 상승적(monotone) 및 하강적(downward) 단조성을 갖는 숫자 개념에 대해 LLM을 평가했습니다. 각 프롬프트는 20개의 레이블 달린 예제(positive와 negative)로 구성되며, 프롬프트의 반응 확률을 비교하여 모형의 응답을 평가했습니다. 실험에 사용된 모델은 OLMo 2와 Qwen3으로, 이들은 다양한 벤치마크에서 경쟁력 있는 성능을 나타내는 오픈(weight) 모델입니다.

- **Performance Highlights**: 실험 결과, 모델은 개념 학습을 통해 상승적 단조성 개념에서 더 높은 정확도를 보였습니다. 반면, 명시적 의미를 사용하는 경우 정확도 차이가 줄어드는 양상을 보였습니다. Qwen3-32B 모델은 두 평가 방법 간의 정확도 차이가 크지 않았으나, OLMo 모델에서 상승적 단조성에 대한 편향이 뚜렷하게 관찰되었습니다. 이러한 결과는 개념 학습이 LLM의 숨겨진 인지 편향을 드러내는 데 도움이 될 수 있음을 제안합니다.



### Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs (https://arxiv.org/abs/2510.01218)
Comments:
          Second Conference on Language Modeling, 2025

- **What's New**: 이번 연구에서는 언어 모델에서 생성된 출력의 창의성을 평가하는 데 필수적인 다양성이 중요하다는 점을 강조합니다. 기존의 온도 기반 샘플링 기법들은 창의성을 증가시키지만, 수학적 추론과 같은 높은 정밀도가 요구되는 작업에서는 오히려 reasoning의 질을 저하시킬 수 있습니다. 저자들은 'selective sampling'이라는 새로운 기법을 제안하여, 샘플링 위험 메트릭에 따라 greedy 샘플링과 높은 온도 샘플링 사이를 동적으로 전환합니다.

- **Technical Details**: selective sampling은 샘플링의 오류 가능성을 예측하기 위해 가벼운 분류기를 훈련시키며, 이는 백본 언어 모델에 최소한의 지연(latency)을 추가하는 방식으로 통합됩니다. 이 접근법은 생성된 원래 모델 출력을 보존하는 동시에 구현이 용이한 특징을 가집니다. 다양한 샘플링 기법들을 분석한 결과, selective sampling이 일반적으로 사용되는 절단 및 엔트로피 기반 샘플링 기법들보다 품질-다양성 균형에서 더 나은 성능을 발휘함을 입증했습니다.

- **Performance Highlights**: 수학적 추론(task) 실험에서는 selective sampling이 고온(high-temperature) 환경에서도 품질-다양성 거래에서 우수한 성능을 보여주었습니다. 저자들은 기법의 효과를 실험을 통해 확인하였고, 기존의 샘플링 방법들이 실패하는 지점을 분석함으로써 향후 연구 방향을 제시하고 있습니다. 이 연구는 LLM의 출력에서 품질과 다양성을 유지하는 것이 중요함을 강조하며, 적응 가능한 디코딩 전략의 필요성을 역설합니다.



### Mamba Outpaces Reformer in Stock Prediction with Sentiments from Top Ten LLMs (https://arxiv.org/abs/2510.01203)
- **What's New**: 이 연구에서는 높은 변동성을 지닌 단기 주가 예측의 어려움을 해결하기 위해, 10개의 대형 언어 모델(LLMs)에서 생성한 의미적 감정 점수를 사용하여 주가 예측의 정확도를 높이는 새로운 프레임워크를 제안합니다. 특히, Apple Inc.(AAPL)의 1분 간격 주가 데이터와 뉴스 기사를 결합하여 데이터셋을 구축하고, 이를 바탕으로 Reformer와 Mamba 두 개 모델의 성능을 분석했습니다. 이 연구는 LLM 기반의 감정 분석과 효율적인 시간 모델링 통합이 실시간 금융 예측을 향상시킬 가능성을 보여줍니다.

- **Technical Details**: 데이터셋은 2025년 4월 4일부터 5월 2일 사이의 Apple Inc.(AAPL) 관련 금융 뉴스와 고주파 1분 간격 주가를 포함합니다. 감정 분석은 DeepSeek-V3, GPT 계열, LLaMA, Claude, Gemini 등 10개의 모델을 통해 이루어졌습니다. 각 뉴스 기사의 감정 점수는 0에서 1 사이의 범위로 조정되어 주가 및 기술 지표와 결합되었습니다. Mamba와 Reformer는 이러한 감정 점수를 입력으로 사용하여 별도로 훈련되었습니다.

- **Performance Highlights**: Mamba 모델은 모든 LLM에서 Reformer보다 빠르고 뛰어난 성능을 보여주었습니다. 특히 LLaMA 3.3–70B 모델을 사용할 때 Mamba는 평균 제곱 오차(MSE) 0.137로 가장 낮은 오류를 기록했습니다. Reformer는 데이터 내의 더 넓은 트렌드를 캡처할 수 있었지만, LLMs에 의해 발생하는 갑작스러운 변화는 과도하게 부드럽게 조정된 것으로 나타났습니다. 이 연구는 LLM 기반의 감정 분석을 활용한 금융 예측의 가능성을 강조합니다.



### LegiScout: A Visual Tool for Understanding Complex Legislation (https://arxiv.org/abs/2510.01195)
- **What's New**: LegiScout는 정적 정책 다이어그램을 동적이고 상호작용 가능한 force-directed 그래프로 변환하여 정책 프레임워크의 복잡성을 이해하는 데 도움을 주는 시각화 시스템입니다. 이 시스템은 데이터 추출, 자연어 처리(natural language processing), 컴퓨터 비전(computer vision) 기술을 통합하여 ACA(Affordable Care Act)뿐만 아니라 다양한 입법 및 규제 프레임워크에 대한 깊은 탐색을 지원합니다. 정책 결정자, 분석가 및 일반 대중 등의 사용자가 현대 법률의 복잡성을 탐색하고 이해할 수 있도록 설계되었습니다.

- **Technical Details**: LegiScout의 설계 과정에서는 정책 구조의 복잡성을 고려하여 핵심 요구 사항을 식별했습니다. 이 시스템은 법률 및 조직 문서에서 엔티티와 그 상호관계를 자동으로 추출하고, 이를 인터랙티브한 force-directed 그래프로 표현합니다. 또한, 주요 용어와 그래프 요소 간의 연결을 가능하게 하는 시맨틱 검색 기능을 제공하며, 다양한 입법 텍스트를 아우르는 확장성을 보장합니다. OCR(Optical Character Recognition) 및 컴퓨터 비전을 사용하는 방법론을 통해 우리는 특성 집합을 정의하고 해당 기능을 구현했습니다.

- **Performance Highlights**: LegiScout의 사용자는 그래프의 주요 엔티티와 관계를 신속하게 해석할 수 있도록 다양한 시각적 스타일이 제공됩니다. 사용자는 그래프 내에서 배율을 조정하고 탐색할 수 있으며, 주요 용어를 입력하면 관련 정책 영역을 빠르게 검색할 수 있습니다. 또한, 그래프의 동적 업데이트와 함께 다양한 상호작용 기능들이 포함되어 있어 사용자는 복잡한 정책 구조를 탐색하면서 효율적인 문서 접근성을 경험할 수 있습니다.



### An Anthropologist LLM to Elicit Users' Moral Preferences through Role-Play (https://arxiv.org/abs/2510.01189)
- **What's New**: 이 연구는 몰입형 롤플레잉 게임(immersive role-playing games)과 대규모 언어 모델(LLM) 분석 능력을 결합하여 사용자의 도덕적 의사결정을 이끌어내는 새로운 접근 방식을 조사합니다. Floridi가 제안한 하드 윤리(hard ethics)와 소프트 윤리(soft ethics)의 구분에 기반하여, 우리는 개인의 도덕적 선호를 수집하는 데 중점을 두고 있습니다. 이 방법은 디지털 프라이버시와 관련된 윤리적으로 민감한 시나리오를 통해 참가자들을 노출시킵니다.

- **Technical Details**: 우리의 연구는 인간의 도덕적 가치를 소설적이고 현상학적 인류학의 관점에서 접근합니다. 롤플레잉 게임(RPG)이라는 민족지적 도구를 사용하여 플레이어가 되어 연구자가 게임 경험을 이끌어갑니다. 데이터를 수집하는 과정은 참가자의 자기 민족지적 멀티미디어 메모와 연구자의 필드 노트(notes)로 구성되어 있으며, ChatGPT-4o(GPT)가 이러한 정보를 분석합니다.

- **Performance Highlights**: 실험 결과는 정교한 데이터와 해석적 프레이밍이 모델의 사용자 행동 예측 능력을 크게 향상시킨다는 것을 보여줍니다. 이는 LLM이 사용자 도덕적 선호 및 의사결정 과정을 이해하는 데 효과적으로 사용할 수 있음을 시사합니다. 이에 따라 소프트 윤리 가치의 수집과 자동화된 이해를 통해 소프트웨어 개발 초기 단계에서의 성공적인 응용 가능성을 나타냅니다.



### Quantum-Assisted Correlation Clustering (https://arxiv.org/abs/2509.03561)
Comments:
          To be published in IEEE QAI 2025 conference

- **What's New**: 이번 연구에서는 그래프 기반의 비지도 학습 작업인 correlation clustering을 위한 혼합 양자-고전적 방법을 제안합니다. 특히, Coalition Structure Generation(CSG)을 위해 처음 설계된 GCS-Q를 조정하여, 서명 그래프에서 클러스터 내 동의(intra-cluster agreement)를 극대화합니다. 제안된 방법은 각 이분할 단계(bipartitioning)를 양자 어닐링(quantum annealing)을 통해 해결할 수 있는 이차 제한 없는 이진 최적화 문제로 인코딩합니다.

- **Technical Details**: 제안된 방법은 위상적 클러스터링의 원칙에 따라 분할(clustering) 작업을 수행합니다. 가중치가 있는 무방향 그래프 G=(V,E,w)를 고려하며, 여기서 wi​j는 노드 간의 유사성을 반영합니다. 목표는 클러스터 내의 동의(intra-cluster agreement)를 극대화하는 분할을 찾는 것이며, 이 과정에서 긍정적 가중치(edge weights)가 연결된 노드들은 동일한 클러스터에 포함됩니다.

- **Performance Highlights**: 모의 그래프 데이터셋 및 실제 하이퍼스펙트럼 이미지 데이터에서 실시한 실험 결과, GCS-Q는 클러스터 크기 불균형이 있는 상황에서도 고전적 알고리즘보다 로버스트성과 클러스터링 품질에서 우수함을 입증했습니다. 우리의 결과는 그래프 기반 비지도 학습에서 혼합 양자-고전적 최적화가 확장 가능하고 구조적으로 인식된 클러스터링 기법을 발전시키는 가능성을 강조합니다.



### Does Bigger Mean Better? Comparitive Analysis of CNNs and Biomedical Vision Language Modles in Medical Diagnosis (https://arxiv.org/abs/2510.00411)
Comments:
          6pages,3 this http URL review of International Conference on Artificial Intelligence, Computer, Data Sciences and Applications

- **What's New**: 이번 연구에서는 가벼운 Convolutional Neural Network (CNN)과 최첨단의 zero-shot 의료 Vision-Language Model (VLM)인 BiomedCLIP 간의 비교 분석을 다룹니다. 두 가지 진단 작업, 즉 PneumoniaMNIST 데이터셋에서의 폐렴 탐지 및 Shenzhen TB 데이터셋에서의 결핵 탐지에 초점을 맞추었습니다. 실험 결과, supervised CNN이 두 경우에 모두 강력한 기준선을 제공하며, VLM은 단순한 결정 임계값 보정을 통해 성능이 크게 향상된다는 것을 보여주었습니다.

- **Technical Details**: 연구에서는 PneumoniaMNIST와 Shenzhen Chest X-ray 데이터셋의 두 가지 공공 데이터셋을 사용하여 비교 분석을 수행하였습니다. lightweight CNN 구조는 여러 단계의 convolutional block으로 구성되며, 각 블록에 대해 ReLU 활성화 및 max-pooling 연산이 적용됩니다. BiomedCLIP은 이미지와 텍스트를 결합하여 zero-shot 방식으로 평가되며, 각 클래스에 대해 생성된 텍스트 임베딩과 테스트 이미지 임베딩 간의 코사인 유사도를 계산하여 클래스를 예측합니다.

- **Performance Highlights**: 보정된 BiomedCLIP은 폐렴 탐지에서 0.8841의 F1-score를 달성하여 supervised CNN의 0.8803을 초과했습니다. 결핵 탐지의 경우 보정 덕분에 F1-score이 0.4812에서 0.7684로 급격히 향상되었고, 이는 supervised 모델의 0.7834에 근접한 성능입니다. 이러한 결과는 zero-shot VLM의 정확한 보정이 기존의 효율적이며 특정 작업에 최적화된 모델과 경쟁할 수 있는 성능을 낼 수 있음을 강조합니다.



### Communication-Efficient and Accurate Approach for Aggregation in Federated Low-Rank Adaptation (https://arxiv.org/abs/2509.26399)
Comments:
          34 pages, 4 figures, 11 tables

- **What's New**: 본 연구에서는 Federated Low-Rank Adaptation(연합 저랭크 적응) 방식의 주요 한계를 해결하기 위한 새로운 접근법인 FLoRA-NA(Federated Low-Rank Aggregation with Nearly Accurate Estimation)를 제안합니다. 기존 방법의 문제점을 해결하려는 노력에도 불구하고, 로컬화가 잘 이루어지지 않고 통신 비용이 과다하게 발생하는 상황이 존재했습니다. FLoRA-NA는 로컬 LoRA 행렬을 서버에서 활용하여 효율적으로 집계된 행렬을 추정하여 클라이언트에 배포함으로써 이러한 문제를 해결합니다.

- **Technical Details**: FLoRA-NA는 기본적으로 로컬에서 업데이트된 LoRA 행렬을 기반으로 집계 행렬을 추정합니다. 이 행렬들은 각 클라이언트의 로컬 업데이트를 위해 분배되며, 이 과정을 통해 이상적인 업데이트와 실제 업데이트 간의 편차를 최소화합니다. 특히, LoRA는 대규모 모델의 가중치를 동결하고 두 개의 저랭크 행렬만을 최적화하는 접근법이며, FLoRA-NA는 이러한 LoRA 메커니즘을 연합 학습 환경에 맞게 최적화합니다.

- **Performance Highlights**: FLoRA-NA는 자연어 이해, 수리적 추론 및 코드 해결 능력 등의 다양한 작업에 대한 광범위한 평가를 통해 기존 FedLoRA 방법에 비해 월등한 성능을 보였습니다. 실험 결과, FLoRA-NA는 통신 오버헤드를 최소화하면서도 높은 글로벌 성능을 달성했습니다. 이러한 성과는 FLoRA-NA가 실제 연합 학습 환경에서 효과적으로 작동할 수 있게 해주며, 클라이언트의 개인화를 유지하면서도 글로벌 일반화를 가능하게 합니다.



New uploads on arXiv(cs.LG)

### KaVa: Latent Reasoning via Compressed KV-Cache Distillation (https://arxiv.org/abs/2510.02312)
Comments:
          Preprint. Under Review

- **What's New**: 본 연구에서는 KaVa라는 새로운 프레임워크를 제안하여, 압축된 KV-캐시에서 직접 지식을 증류하여 잠재적 추론을 이루는 학생 모델을 개발합니다. 이 방법은 압축된 KV-캐시로부터 얻은 추상적인 지식이 잠재적 추론의 효과적인 감독 신호로 작용할 수 있음을 보여줍니다. KaVa는 기존의 추론 모델들이 직면한 감독 신호 부족 문제를 해결하며, 자연어 기반 추론에서 성능 향상을 이루었습니다.

- **Technical Details**: 우리는 KaVa를 통해 세 가지 주요 구성 요소를 사용하여 KV-캐시의 압축된 정보를 학생 모델에 지도하며, 이 과정을 통해 내부 추론의 동적을 학습하게 합니다. 첫째, 교사 모드에서 모든 CoT를 분석하고, 둘째, 중요도에 기반하여 압축된 캐시를 잠재적 예산에 맞춥니다. 마지막으로, 학생 모델의 각 단계에서 KV를 압축된 대상으로 일치시키는 손실 함수를 통해, 보다 강력하고 단계적인 감독 신호를 제공합니다.

- **Performance Highlights**: 실험 결과, KaVa는 강력한 잠재적 기준선보다 지속적으로 성능을 초과하며, 수식 전용에서 자연어 추론으로의 전환 시 성능 저하가 적음을 보여줍니다. 또한, KaVa는 대규모 모델의 경우 효율성을 유지하며 자연어 데이터셋에서 강력한 성과를 달성하여, 압축 KV-캐시의 증류가 잠재적 추론을 위한 확장 가능한 감독 신호로 작용할 수 있음을 밝힙니다.



### Robust Tangent Space Estimation via Laplacian Eigenvector Gradient Orthogonalization (https://arxiv.org/abs/2510.02308)
- **What's New**: 이번 논문에서 제안한 LEGO (Laplacian Eigenvector Gradient Orthogonalization) 알고리즘은 데이터의 전역 구조를 활용하여 각 데이터 포인트에서 접선 공간(tangent space)을 추정하는 새로운 방법입니다. 기존의 Local Principal Component Analysis (LPCA)와는 달리, LEGO는 노이즈 환경에서도 높은 성능을 발휘하도록 설계되었습니다.

- **Technical Details**: LEGO는 그래프 라플라시안(graph Laplacian)의 저주파(eigenvector) 고유 벡터의 기울기를 직교화(orthogonalization)하여 각 데이터 포인트에서 접선 공간을 추정합니다. 이때, 저주파 고유 벡터는 데이터 매니폴드(manifold)의 기하학적 구조와 잘 정렬되어 있어 안정적인 추정이 가능하다는 두 가지 이론적 근거를 제시합니다.

- **Performance Highlights**: 실험 결과, LEGO는 LPCA에 비해 노이즈에 훨씬 더 강한 접선 공간 추정치를 제공하며, 매니폴드 학습(manifold learning), 경계 탐지(boundary detection), 지역 내재 차원 추정(local intrinsic dimension estimation)과 같은 여러 후속 작업에서도 뚜렷한 개선을 보여주었습니다.



### Diffusion Models and the Manifold Hypothesis: Log-Domain Smoothing is Geometry Adaptiv (https://arxiv.org/abs/2510.02305)
- **What's New**: 이 논문은 diffusion models가 갖는 탁월한 일반화 능력을 설명하는 기존 이론을 뒷받침하는 증거를 제시합니다. 특히, 데이터를 통해 학습하는 과정에서 score matching을 통해 학습 문제를 어떻게 구성하는지가 이들 모델의 성공에 중요한 역할을 한다고 주장합니다. 이 연구는 empirical score matching 목표 함수의 smoothing 최소화에 대한 영향을 조사하며, 지리적 구조를 잃지 않는 smoothing의 중요성을 강조합니다.

- **Technical Details**: Diffusion models는 고차원 데이터가 저차원 매니폴드에 집중되어 있다는 매니폴드 가설에 기반하여 작동합니다. 이 모델들은 도출된 score 함수의 smoothing을 통해 empirical density p^t의 log-domain에서 smoothing을 수행하며, 이는 효과적으로 데이터 매니폴드를 보존합니다. 논문은 이와 관련하여, smooth approximations을 통한 inductive bias 모델을 제안하며, convolution이 gradient 연산과 교환 가능하다는 직관을 제공합니다.

- **Performance Highlights**: 모델의 성능 및 일반화 능력에 대한 실험적 결과는 smoothing이 diffusion models의 일반화 가능성을 높이는 핵심 요소임을 증명합니다. 논문에서는 log-domain에서의 smoothing이 데이터의 기하학적 구조를 유지하는 데 중요하다는 점을 강조하며, 이를 통해 데이터 매니폴드를 따라 일반화하는 여러 기하학적 경로를 탐색할 수 있음을 시사합니다. 이러한 발견들은 hybrid diffusion models 및 flow-based 접근법의 발전에도 기여할 것입니다.



### Knowledge Distillation Detection for Open-weights Models (https://arxiv.org/abs/2510.02302)
Comments:
          NeurIPS 2025

- **What's New**: 본 논문에서는 지식 증류(knowledge distillation) 탐지(task of knowledge distillation detection)라는 새로운 과제를 제안합니다. 이는 학생 모델(student model)이 특정 교사 모델(teacher model)로부터 증류(distilled)되었는지를 판단하는 것으로, 학생의 가중치(weights)와 교사의 API에만 접근할 수 있는 현실적인 설정에서 수행됩니다. 이 문제는 모델의 출처(provenance)에 대한 우려와 비허가된 복제를 방지하기 위해 필요하다고 합니다.

- **Technical Details**: 이 연구는 데이터가 없는 입력 합성(data-free input synthesis) 및 통계적 점수 컴퓨테이션(statistical score computation)을 결합한 모델-무관 접근법(model-agnostic framework)을 사용하여 증류를 탐지합니다. 이러한 접근 방식을 통해 분류(classification) 및 생성(generative) 모델 모두에서 구현이 가능하며, 입력 구성, 점수 생성, 결정 내리기의 세 단계를 포함합니다. 이에 따라 명확한 절차를 통해 증류 여부를 판단할 수 있습니다.

- **Performance Highlights**: 실험 결과, CIFAR-10에서 59.6%, ImageNet에서 71.2%, 텍스트-이미지 생성(text-to-image generation) 테스트에서도 20.0%의 정확도 향상을 보여주며, 그 효과성이 입증되었습니다. 다양한 아키텍처와 증류 방법을 테스트하여 포괄적인 비교도 수행되었으며, 이러한 결과는 학생 모델이 명확하게 교사 모델로부터 파생되었는지 판별하는 데 유용합니다.



### Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models (https://arxiv.org/abs/2510.02300)
- **What's New**: 이번 논문에서는 Equilibrium Matching (EqM)이라는 새로운 생성 모델링 프레임워크를 소개합니다. EqM는 기존의 비평형(Non-equilibrium) 동역학을 배제하고, 암시적(Implicit) 에너지 경관에서의 평형(Equilibrium) 기울기를 학습합니다. 이 접근법을 통해, 샘플 생성 시 최적화 기반 샘플링 프로세스를 채택하며, 이를 통해 이미지 생성의 성능을 향상시킵니다.

- **Technical Details**: Equilibrium Matching은 시간 조건 비대칭 동역학을 단일 불변 평형 기울기로 대체하여, EBM(에너지 기반 모델) 관점에서 학습합니다. 이를 통해, 샘플링 단계에서 각 샘플마다 독립적으로 적응형 옵티마이저와 단계 크기를 조정할 수 있게 되어, 60%의 함수 평가를 절약할 수 있습니다. EqM는 데이터 매니폴드에서 실체 샘플을 뽑아내고, 이를 통해 이미지 생성의 질적인 향상을 달성할 수 있습니다.

- **Performance Highlights**: 실험적으로 EqM는 ImageNet 256×256 데이터 세트에서 1.90의 FID(Fréchet Inception Distance)를 달성하며, 기존의 확산(Diffusion) 및 흐름 기반(Flow-based) 모델을 초월하는 성능을 보여줍니다. Equilibrium Matching은 다양한 크기에서 뛰어난 확장성을 갖추고 있으며, 이미지 구성, OOD 탐지, 부분적으로 노이즈가 있는 이미지 복원 등의 작업을 자연스럽게 처리할 수 있는 유연한 프레임워크입니다.



### Interactive Training: Feedback-Driven Neural Network Optimization (https://arxiv.org/abs/2510.02297)
Comments:
          EMNLP 2025 Demo

- **What's New**: 이번 논문에서는 Interactive Training이라는 새로운 오픈 소스 프레임워크를 소개합니다. 이 프레임워크는 네트워크 훈련 중에 실시간으로 피드백을 받아 인공지능이 자동으로 개입하게 하며, 사용자 또는 AI 에이전트의 개입을 허용합니다. 이를 통해 사용자는 옵티마이저의 하이퍼파라미터, 훈련 데이터 및 모델 체크포인트를 동적으로 조정할 수 있습니다.

- **Technical Details**: Interactive Training 프레임워크의 핵심은 Control Server입니다. 이 서버는 사용자의 명령과 지속적인 훈련 프로세스 간의 통신을 중재하며, FastAPI를 통해 API 엔드포인트를 노출하고 JSON 메시지를 통해 명령을 처리합니다. Interactive Trainer는 Hugging Face의 Trainer 클래스에 기반하여 구현되어, 동적으로 조정된 훈련 매개변수를 바탕으로 실시간 개입에 반응합니다.

- **Performance Highlights**: 세 가지 사례 연구를 통해 우리의 접근 방식이 기존의 정적 최적화 방법들보다 우수하다는 것을 보여줍니다. 경험이 있는 인간 개발자들이 실시간 인터랙션을 활용하여 더 나은 최적화 결과를 도출했고, AI 에이전트가 자동으로 초기 하이퍼파라미터를 수정할 수 있는 가능성도 입증되었습니다. 또한, 이 프레임워크는 실제 배포 중 수집된 사용자 데이터를 실시간으로 반영하여 모델의 적응성을 향상시킵니다.



### Continual Personalization for Diffusion Models (https://arxiv.org/abs/2510.02296)
- **What's New**: 논문에서는 Concept Neuron Selection (CNS)이라는 새로운 학습 전략을 제시하여, 지속적인 학습 환경에서 퍼스널리제이션을 효율적으로 수행하는 방법을 소개합니다. CNS는 특정 개념과 관련된 뉴런을 정확히 식별하여 기존의 모델 지식을 유지하면서도 새로운 개념을 추가로 학습할 수 있도록 합니다. 이를 통해 카타스트로픽 포겟팅(catastrophic forgetting) 문제를 완화하며, 제로샷(zero-shot) 텍스트-이미지 생성 능력을 보존합니다.

- **Technical Details**: CNS의 주요 기능은 퍼스널리제이션에 필요한 개념 뉴런을 자동으로 식별하는 것입니다. 일반 뉴런과 기본 뉴런을 구분하여 개념 뉴런을 선택하며, 노드 간의 연결을 통해 점진적인 파인튜닝(incremental finetuning)을 가능하게 합니다. 기존의 방법과 달리, CNS는 사용자 지정 레이아웃이나 추가 모델 저장 없이도 멀티 개념 학습을 수월하게 처리할 수 있습니다.

- **Performance Highlights**: CNS는 실제 데이터 세트를 평가한 결과, 매개변수 조정이 최소화된 상태에서도 최첨단 성능을 달성하였습니다. 특히 단일 및 다중 개념 퍼스널리제이션 작업에서 이전 방법보다 더 나은 결과를 나타내었으며, 메모리 저장 및 처리 시간을 줄입니다. CNS는 별도의 퓨전(fusion) 없이도 효과적인 지속적 퍼스널리제이션을 제공하는 방법으로 주목받고 있습니다.



### Test-Time Anchoring for Discrete Diffusion Posterior Sampling (https://arxiv.org/abs/2510.02291)
Comments:
          Preprint

- **What's New**: 본 연구에서는 사전 훈련된 이산 확산(Discrete Diffusion) 모델을 활용하여 잡음이 있는 측정값으로부터 이미지를 복원하는 후방 샘플링 문제를 다루었습니다. 기존의 방법들은 연속 확산 모델에 의존했으나, 본 논문은 이산 확산을 통해 다중 모달 데이터(예: 텍스트와 이미지)의 통합 모델링을 가능하게 하며, 특히 훈련 없이도 Bayesian 추론을 통해 후방 샘플링에 적합한 방법을 제시합니다. 기여 내용으로는 quantized expectation과 anchored remasking을 통한 효율적인 후방 샘플링 전략이 포함되었습니다.

- **Technical Details**: Anchored Posterior Sampling (APS) 방법은 이산 임베딩 공간에서 gradient-like guidance를 제공하는 quantized expectation 전략과, 역 과정에서 중요한 '앙커' 토큰을 조기에 복원하는 adaptive remasking 전략으로 구성됩니다. 이러한 혁신은 이산 확산의 비 미분성 문제를 극복하며, 기계 학습 모델의 성능을 극대화할 수 있도록 지원합니다. APS 방법은 다양한 선형 및 비선형 역 문제에서 기존의 이산 샘플링 모델들에 비해 성능 향상을 보였으며, 이를 통해 훈련 없는 스타일화 및 텍스트 기반 편집 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, 본 방법은 FFHQ 및 ImageNet 데이터셋에서 LPIPS 기준으로 최대 35.82% 향상과 PSNR 기준으로 10.94% 향상을 달성했습니다. 특히, 선형 및 비선형 역 문제에서 31.36% LPIPS 및 7.05% PSNR 향상을 보여 주목할 만한 성과를 기록하였습니다. 이 연구는 훈련 없는 스타일화 및 편집에서의 유연성을 강조하며, 이산 확산 모델을 기반으로 한 새로운 후방 샘플러에 대한 가능성을 제시합니다.



### Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (https://arxiv.org/abs/2510.02286)
- **What's New**: 이 연구는 DialTree-RPO라는 새로운 강화 학습 프레임워크를 제안하여 다중 턴 공격 전략을 자율적으로 발견하는 방법을 제시합니다. 기존의 접근 방식들이 단일 턴 공격에 초점을 맞춘 것과 달리, DialTree-RPO는 대화를 연속적인 의사 결정 문제로 간주하여 다중 턴 공격의 다양한 가능성을 탐색합니다. 이 과정에서, 자동화된 데이터 없이도 공격 성공률을 극대화할 수 있는 최적의 대화 정책을 학습합니다.

- **Technical Details**: DialTree-RPO는 다중 턴 공격을 위한 정책 최적화를 목표로 하는 새로운 강화 학습 프레임워크입니다. 이 시스템은 대화 나무 롤아웃 (dialogue tree rollout) 및 질 높은 경로 프루닝 (quality-aware pruning) 기법과 적응적 마스킹 (adaptive masking) 기술을 통합하여, 훈련 중 비효율적인 공격 경로를 제거하고 최적화 안정성을 높이며 효율성을 개선합니다. 이를 통해 다중 턴 대화의 복잡성을 효과적으로 적용할 수 있습니다.

- **Performance Highlights**: DialTree-RPO는 10개의 대상 모델에서 평균 공격 성공률 (ASR) 85.3%를 달성하여 이전 최신 기술보다 25.9% 더 높은 성능을 보였습니다. 이 연구는 모델 크기와 관계없이 뛰어난 효율성과 전이 능력을 보여주며, 새로운 공격 전략을 발견하는 데에도 효과적입니다. 결과적으로 DialTree-RPO는 다중 턴 강화 학습을 기반으로 한 새로운 최첨단 레드 팀링 방법을 수립했습니다.



### Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation (https://arxiv.org/abs/2510.02279)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 신뢰도를 저해하는 일반적인 문제인 환각(hallucinations)에 대한 심층 분석을 제시합니다. 특히, LLM의 예측 불확실성(predicitive uncertainty)으로 인해 발생하는 'confabulation'이라는 특정 환각 유형을 식별합니다. 다양한 방법론이 개발되어 자연어 생성(NLG)에서의 예측 불확실성을 측정하고 있지만, 기존의 근사 정확성 함수들이 상이함을 강조하고, 평가의 편향(bias)을 해소할 수 있는 대안 적 위험 지표를 제안합니다.

- **Technical Details**: 환각 감지를 위해 NLG에서의 예측 불확실성을 정량화하는 것은 LLM의 예측 분포 엔트로피를 기반으로 합니다. 최근 불확실성 추정 알고리즘은 주로 질문 답변(QA) 데이터셋과 같은 좁은 문제 범주에서 선택적 예측(selective prediction)으로 평가되고 있습니다. 연구팀은 confabulation 감지를 위해 다양한 LLM-as-a-judge 변형과 더불어 OOD(out of distribution) 탐지 및 변동 감지 작업 등을 포함한 구조화된 작업을 연구합니다. 이 구조들은 더 강력하고 통제 가능한 위험 지표를 제공합니다.

- **Performance Highlights**: 분석 결과, 여러 LLM-as-a-judge 변형을 통합하여 평가할 경우 평가 편향이 줄어드는 것으로 나타났습니다. LLM-as-a-judge를 정확성 메트릭으로 활용해야 하며, 엘로(Elo) 랭킹 기법을 통해 다양한 실험 설정에서 불확실성 추정 방법의 성능을 보다 객관적으로 평가할 수 있음을 보여줍니다. 이러한 방법은 기존의 문맥에서 제기된 문제점들을 개선하는데 기여할 것입니다.



### Fine-Grained Urban Traffic Forecasting on Metropolis-Scale Road Networks (https://arxiv.org/abs/2510.02278)
- **What's New**: 이 논문은 교통 예측의 효율성을 높이기 위한 새로운 데이터 세트를 소개합니다. 두 개의 주요 도시에 대한 도로 네트워크 데이터를 제공하며, 가장 큰 데이터 세트는 100,000개 도로 구간을 포함하여 기존 데이터 세트에 비해 10배 이상의 규모입니다. 이는 도시 교통 패턴과 속도 및 양에 대한 세부 정보를 담고 있어 보다 포괄적인 교통 예측 시스템 구축이 가능하도록 합니다.

- **Technical Details**: 교통 예측의 어려움은 비선형적이며 복잡한 시공간 의존성을 다루는 데 있습니다. 기존의 통계 모델이 아닌, 심층 학습과 그래프에서의 표현 학습을 통해 이러한 복잡성을 해결하고자 하며, GNN(Spatiotemporal Graph Neural Networks)이 주요 방법론으로 등장했습니다. 논문에서는 시간적 패턴을 위한 전용 모듈 없이 GNN을 사용한 새로운 접근 방식을 제안하여 스케일 확장성 문제를 극복하고 있다고 설명합니다.

- **Performance Highlights**: 제안된 데이터 세트를 기반으로 현재의 많은 신경망 교통 예측 모델들이 스케일에 어려움을 겪고 있다는 것을 입증했습니다. 새로운 GNN 기반 접근 방식은 기존 모델보다 향상된 예측 성능을 보여주며, 더 나아가 교통 예측의 발전과 스마트 시티 개발에도 기여할 것으로 기대됩니다. 이 논문에서 제시된 모델링 통찰력은 연구자들에게 중요한 자원이 될 것입니다.



### Diffusion^2: Turning 3D Environments into Radio Frequency Heatmaps (https://arxiv.org/abs/2510.02274)
- **What's New**: Diffusion2는 3D 포인트 클라우드를 활용하여 다양한 주파수에서 전파되는 RF 신호를 모델링하는 새로운 접근 방식을 제안합니다. 기존의 방법들과 비교할 때, 더 적은 신호 측정을 기반으로 더 높은 정확성을 제공하며, 프로젝트 처리 속도를 27배 향상시킵니다. 또한, 방대한 데이터 필요 없이 다차원 환경에서의 RF 신호 예측 기능을 지원함으로써 다양한 응용 분야에 유용합니다.

- **Technical Details**: Diffusion2는 RF-3D Encoder를 통해 3D 환경 모델과 RF 관련 정보를 통합하여 RF 신호 열지도를 생성합니다. 이 모델은 다단계 확산 과정을 통해 복잡한 최적화 문제를 확률적 계산으로 단순화하며, 노이즈로 변환된 데이터를 복구하는 역확산 프로세스를 포함합니다. 조건부 생성(Conditioning) 기능을 사용해 RF 신호 분포에 따른 결과물을 보장합니다.

- **Performance Highlights**: Diffusion2는 15회의 신호 측정으로 높은 정확도를 달성하며, 200,000개의 수신기(RX)를 1초 이내에 처리할 수 있어 빠른 계산 속도를 자랑합니다. 또한, 정적 3D 장면과 동적 3D 장면 모두에 대해 RF 열지도를 생성할 수 있는 기능을 갖추고 있습니다. 이는 채널 할당 및 간섭 관리와 같은 운영 작업에 큰 가치를 주는 다중 주파수 RF 열지도 생성을 가능하게 합니다.



### How to Combat Reactive and Dynamic Jamming Attacks with Reinforcement Learning (https://arxiv.org/abs/2510.02265)
- **What's New**: 이 논문은 reactive jamming 문제를 다루고 있으며, 여기서 jammer는 동적인 정책을 사용하여 채널과 감지 임계값을 선택하여 전송을 방해합니다. 전송기-수신기 쌍은 reinforcement learning (RL)을 활용하여 방해를 회피하고, 채널 조건이나 방해 전략에 대한 사전 지식 없이도 전송 전력을 조정하며 최적의 throughput을 극대화합니다. 이는 RL 방법을 통한 비선형 적응 성능을 보장합니다.

- **Technical Details**: 이 시스템 모델은 전송기와 수신기 간의 통신을 고려하며, jammer의 영향을 받는 상황을 Markov decision process (MDP)로 모델링합니다. 전송기는 각 시간 슬롯에서 전송 전력과 변조 방식을 선택하며, jammer는 에너지 검출기를 사용하여 방해 여부를 결정합니다. 이 논문에서는 Q-learning과 Deep Q-Networks (DQN)를 사용하여 분산 및 연속 상태 공간을 위한 학습을 수행합니다.

- **Performance Highlights**: 결과는 RL이 변화하는 방해 전략 및 스펙트럼 조건에 신속하게 적응하여 높은 전송률을 지속할 수 있음을 보여줍니다. 다양한 보상 함수와 행동 세트를 통해 RL 방법이 적응할 수 있는 능력을 강조하며, 이로 인해 무선 통신에서의 링크 신뢰성을 개선할 수 있는 가능성을 확인했습니다.



### Transformers Discover Molecular Structure Without Graph Priors (https://arxiv.org/abs/2510.02259)
- **What's New**: 이번 연구에서는 Graph Neural Networks (GNNs)의 전통적인 접근법에서 벗어나, 미리 정의된 그래프나 물리적 편향 없이 카르테시안 좌표에서 직접 훈련된 순수 Transformer 모델이 분자의 에너지와 힘을 근사할 수 있는지를 탐구했습니다. 우리는 Open Molecules 2025 (OMol25) 데이터셋을 사용하여 이 모델이 기존의 state-of-the-art GNN과 유사한 성능을 보일 수 있음을 보여주었습니다. 이 접근법은 GNN이 하드 코딩된 그래프에서 비롯된 것과 같은 강한 물리적 편향을 필요로 하지 않음을 시사합니다.

- **Technical Details**: GNNs는 분자의 물리적 특성을 예측하기 위해 주로 메시지 패싱(message passing)을 이용하며, 이는 사전 정의된 그래프 구조에 의존합니다. 하지만 이 연구에서는 그래프 기반의 인디크션 편향 없이도 Transformer가 효과적으로 학습할 수 있음을 입증했습니다. Transformer는 인젝션된 주의 메커니즘을 통해 분자 환경에 따라 적응할 수 있는 유연한 모델이며, 학습된 주의 맵은 원자 간 거리와 주의 강도 간의 역 관계를 포착합니다.

- **Performance Highlights**: Transformers는 동일한 훈련 컴퓨팅 예산 내에서 기존의 GNN과 비교하여 에너지 및 힘의 평균 절대 오차를 경쟁력 있게 달성했습니다. 또한, 기존 GNN에 비해 훈련 및 추론 속도 면에서 상당히 빠른 성과를 보여 주며, 기존 소프트웨어와 하드웨어를 통해 1B 파라미터로 스케일링이 가능합니다. 이러한 결과는 많은 GNN의 유리한 특성이 Transformer에서도 적응적으로 등장할 수 있음을 보여주며, 더 넓은 화학 문제 해결을 위한 표준화된 아키텍처에 대한 가능성을 열어줍니다.



### ExGRPO: Learning to Reason from Experienc (https://arxiv.org/abs/2510.02245)
- **What's New**: 본 논문에서는 강화학습을 통한 검증 가능한 보상(Reinforcement Learning from Verifiable Rewards, RLVR)의 새로운 패러다임을 제안하고, 이에 대한 효율적인 경험 관리 방법론인 Experiential Group Relative Policy Optimization (ExGRPO)을 제시합니다. ExGRPO는 경험의 가치를 평가하기 위해 롤아웃의 정확성과 엔트로피를 활용하며, 모델이 과거의 경험을 효과적으로 재사용할 수 있도록 도와줍니다. 이 방법은 대규모 언어 모델의 추론 성능을 향상시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: ExGRPO는 롤아웃 중 생성된 경험을 효율적으로 관리하기 위해 경험의 정확도에 따라 버킷으로 조직하고, 가장 유용한 버킷에서 경험을 우선적으로 샘플링하는 전략을 채택합니다. 또한, 이 방법은 샘플 효율성을 개선하고 훈련 안정성을 증대시키기 위해 신선한 탐색과 과거 경험의 재사용을 균형있게 조절하는 혼합 정책 최적화(mixed-policy optimization) 목표를 사용합니다. 향후 연구의 중요성을 강조하고 있는 이 방법론은 경험 관리가 RLVR에서 필수적인 요소임을 보여줍니다.

- **Performance Highlights**: ExGRPO는 1.5B에서 8B 매개변수를 가진 다섯 가지 백본 모델에서 RLVR 성능을 평균적으로 +3.5/7.6 포인트 이상 개선하며, 수학적 및 일반 벤치마크에서 일관된 성과를 나타냅니다. 특히, ExGRPO는 기존의 온-정책 방식의 최적화가 실패하는 모델에서도 훈련을 안정화시키는 데 성공하여, 전체적인 학습 성과를 높였습니다. 이러한 연구 결과는 과거의 경험을 효과적으로 활용하는 것이 대규모 언어 모델의 성능을 향상시키는 데 중요한 역할을 한다는 것을 강조합니다.



### Drop-Muon: Update Less, Converge Faster (https://arxiv.org/abs/2510.02239)
- **What's New**: 이번 연구는 딥러닝 최적화에서 일반적인 상식을 재조명하며, 모든 레이어를 매 단계 업데이트하는 것이 반드시 최적이 아닐 수 있음을 보여줍니다. 저자들은 무작위적으로 레이어의 일부만 업데이트하는 Drop-Muon이라는 새로운 방법을 제안하며, 이는 효율적인 progressive training과 결합된 비유클리드적 업데이트를 특징으로 합니다. Drop-Muon은 최첨단 성능을 유지하면서도 계산 비용을 줄이는 방법으로 주목받고 있습니다.

- **Technical Details**: Drop-Muon은 각 반복에서 모든 네트워크를 업데이트하는 대신, 사용자가 정의한 분포에 따라 무작위적으로 선택된 레이어 집합만 업데이트합니다. 이 방법은 레이어별로 특정한 비유클리드적 업데이트를 사용하며, 효율성을 극대화하는 동시에 높은 성능을 달성합니다. 분석을 통해, 완전 네트워크 업데이트가 최적이 되기 위한 특정한 조건을 필요로 한다는 점이 강조되었습니다.

- **Performance Highlights**: 실험 결과, Drop-Muon은 전통적인 Muon 방식보다 일관되게 더 나은 성능을 보여주며, 주어진 정확도를 유지하면서도 시계열 성능이 최대 1.4배 빨라지는 것으로 나타났습니다. 이러한 발견은 대규모 모델의 효율적인 훈련에 대한 새로운 접근 방식의 필요성을 제기합니다. 이를 통해 Drop-Muon은 전통적인 최적화 방법과는 달리, 더 효과적인 학습 전략으로 자리매김할 것으로 기대됩니다.



### PUL-Inter-slice Defender: An Anomaly Detection Solution for Distributed Slice Mobility Attacks (https://arxiv.org/abs/2510.02236)
Comments:
          13 pages, 7 figures, 4 tables, journal paper

- **What's New**: 이 논문에서는 Network Slices(네트워크 슬라이스) 내에서의 공격 탐지를 위한 혁신적인 솔루션인 PUL-Inter-Slice Defender를 소개합니다. 이 솔루션은 Positive Unlabeled Learning(PUL)을 기반으로 하며, Long Short-Term Memory Autoencoders(LSTM-Autoencoder)와 K-Means 클러스터링을 결합하여 작동합니다. 이전 연구에서의 한계를 극복하고, 데이터가 오염된 조건에서도 높은 정확도로 공격을 탐지할 수 있는 시스템을 제공하는 데 중점을 두고 있습니다.

- **Technical Details**: PUL-Inter-Slice Defender는 3GPP의 성능 지표를 기반으로 훈련되고, DSM 공격 변형을 탐지하기 위해 설계되었습니다. 여기에는 Random Slice Attack(RSA)와 Target Slice Attack(TSA)이 포함됩니다. 이 모델은 LSTM-Autoencoder를 사용하여 ISS 이벤트와 관련된 잠재적 특성을 추출하고, K-Means 알고리즘을 통해 정상 패턴과 공격 패턴을 구분합니다.

- **Performance Highlights**: PUL-Inter-Slice Defender는 훈련 데이터셋의 공격 샘플 비율이 10%에서 40% 사이인 실험에서 평균 F1 점수가 98%를 초과하는 성과를 기록했습니다. 이 성능은 Inter-Slice Defender, PUL-OCSVM-RF, PUL-OCSVM-XGBoost와 같은 다른 기준 모델들보다 월등히 뛰어난 결과입니다. 이러한 결과는 PUL-Inter-Slice Defender의 강력한 성능을 입증합니다.



### xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity (https://arxiv.org/abs/2510.02228)
Comments:
          Code and data available at this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLM)의 성능 예측에 있어서 스케일링 법칙의 중요성을 다루고 있습니다. xLSTM과 Transformer 모델을 비교하여 각각의 스케일링 거동을 분석하고, 예측 성능과 컴퓨팅 효율성을 향상시키기 위한 방향성을 제시합니다. 특히, xLSTM의 선형 복잡도가 Transformer의 이차 복잡도에 비해 더 유리하다는 점을 강조합니다.

- **Technical Details**: 논문에서는 두 가지 주요 접근 방식, IsoFLOP 및 파라메트릭 피팅 방법을 통해 xLSTM과 Transformer의 스케일링 거동을 조사합니다. xLSTM은 메모리 및 전력 제한이 있는 환경에서도 부정적인 영향을 줄여, 훈련 및 추론 과정에서 더 나은 성능을 보여줍니다. 기존의 Transformer 모델들은 주로 주의 메커니즘과 관련된 이차 복잡도 때문에 제한을 받는 반면, xLSTM은 선형 복잡성을 유지합니다.

- **Performance Highlights**: xLSTM 모델들은 같은 크기의 Transformer 모델보다 더 낮은 교차 엔트로피 손실 값(cross-entropy loss)을 기록하며, 이는 모델의 성능이 더 우수하다는 것을 의미합니다. 논문에 따르면, 주어진 훈련 컴퓨팅 예산 내에서 xLSTM 모델들이 더 큰 매개변수를 가질 수 있으며, 일반적으로 추론 단계에서도 xLSTM이 Transformer보다 더 빠르게 작동합니다. 이는 문맥 길이가 증가할수록 더욱 두드러지며, xLSTM의 성능 우위가 강화된다는 결과를 가져옵니다.



### Efficiently Generating Correlated Sample Paths from Multi-step Time Series Foundation Models (https://arxiv.org/abs/2510.02224)
- **What's New**: 이번 논문에서는 기존의 다단계 시계열 모델을 기반으로 하여, 상관된 샘플 경로를 효율적으로 생성하는 새로운 방법을 제안합니다. 이 연구에서 제안된 copula 기반 접근법은 기존의 autoregressive sampling보다 수십 배 빠르게 작동하며, 실제적인 상관 구조를 갖춘 고품질 샘플 경로를 생성합니다. 이러한 방법은 시계열 예측의 효과를 크게 높이며, 다양한 실질적인 사용 사례에 적용할 수 있습니다.

- **Technical Details**: 저자들은 copula를 사용하여 조인트 예측 분포를 분해하는 방법을 제안합니다. 이는 시간에 따른 시계열 예측을 위해 높은 품질의 주변 분포를 사용하면서, 상관 구조를 모델링하는 데 초점을 맞춥니다. 구체적으로, 주어진 시계열 데이터에 대해, Gaussian copula를 사용하여 샘플 경로를 생성하고, 이를 통해 타당한 상관 구조를 부과합니다.

- **Performance Highlights**: 제안된 copula 기반 방법으로 생성된 샘플 경로는 이전의 autoregressive 방법으로 생성된 것과 품질 경쟁력이 있으며, 실질적으로 더 저렴한 비용으로 이루어집니다. 또한, snowballing error 현상의 완화로 인해, 보다 정확한 샘플 경로를 제공함을 확인했습니다. 이는 시계열 연구자들이 다단계 TSFMs의 발전을 활용할 수 있는 기회를 크게 확대합니다.



### Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification (https://arxiv.org/abs/2510.02216)
Comments:
          49 pages, 4 figures. Accepted as a poster at NeurIPS 2025

- **What's New**: 최근의 연구에서는 시간 시계열 데이터의 결측값을 보완하기 위해 확산 기반 생성 보간(imputation) 방법이 효과적임을 보여주고 있습니다. 기존의 자기 회귀(autoregressive) 및 통계적 접근 방식에 비해 이러한 방식이 뛰어난 성능을 지니고 있다는 점이 강조되었습니다. 하지만 이러한 방법들이 결측값과 관측값 간의 복잡한 공간적(spatial) 및 시계열적(temporal) 의존성을 얼마나 잘 포착하는지에 대한 이론적 이해는 부족했습니다.

- **Technical Details**: 이 연구에서는 조건부(diffusion) 변환자(transformers)를 이용한 보간의 통계적 효율성을 조사하고 결측값의 불확실성을 정량화합니다. 새로운 근사 이론을 바탕으로 조건부 스코어 함수(conditional score functions)에 대한 통계적 샘플 복잡성(bounds)을 도출하여 결측값의 신뢰 구역(confidence regions)을 정확하게 설정합니다. 연구를 통해 보간의 효율성과 정확성은 결측 패턴에 따라 크게 달라진다는 것을 보여주었습니다.

- **Performance Highlights**: 실험을 통해 이론적 통찰력을 검증하고, 보간 성능을 향상시키기 위한 혼합 마스킹(mixed-masking) 훈련 전략을 제안했습니다. 이러한 접근 방식을 통해 결측값 보완의 품질이 significantly 향상됨을 확인할 수 있었습니다. 이 연구는 결측값 처리에 있어 새로운 통계적 방법론을 제시하여 분야의 발전에 기여할 것입니다.



### C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems (https://arxiv.org/abs/2510.02215)
Comments:
          Submitted to ICLR 2026

- **What's New**: 이번 논문에서는 대규모 추천 모델을 학습할 때 사용자 집단 간 이질성을 고려하지 않는 것의 문제점을 지적합니다. 저자들은 Attention 메커니즘이 Factorization Machines에서 중요한 역할을 할 수 있음을 밝혔습니다. 또한, 보조 학습(auxiliary learning)을 통해 데이터셋의 하위 구조를 분석하여 강한 분포 대조를 가진 경우를 노출함으로써 이 균형 문제를 해결하는 방법을 제안합니다. 이를 위해, C2AL(Cohort-Contrastive Auxiliary Learning)이라는 새로운 접근법을 도입하여 사용자 상호작용의 세부사항을 포착하는 방법을 제시합니다.

- **Technical Details**: C2AL 방법론은 다중 작업 학습(Multi-Task Learning, MTL) 개념을 기반으로 하며, 제2의 작업(secondary tasks)을 사용하여 모델의 표현을 조절합니다. 이러한 보조 작업들은 훈련 과정에서 활용되며 추론 단계에서 제외됩니다. 구체적으로 모델의 Attention 메커니즘을 통해 다수 집단의 패턴뿐만 아니라 소수 집단의 패턴을 포착할 수 있도록 학습하고, 결과적으로 더 밀집한(weight) 특징 상호작용을 생성합니다. C2AL은 Deep and Hierarchical Ensemble Network(DHEN) 구조를 기반으로 하여 대규모 생산 데이터셋에서 실제 모델에 대한 테스트를 진행했습니다.

- **Performance Highlights**: 실험 결과, C2AL을 적용한 Factorization Machines는 추천 시스템의 사용자 상호작용을 세부적으로 포착하여, 전체적으로 평균 0.16%의 정규화 엔트로피(normalized entropy)를 감소시켰습니다. 또한, 소수 집단에 대한 성능 향상률은 0.30%를 초과하는 긍정적인 결과를 보였습니다. 이는 C2AL이 대규모 추천 모델의 예측 정확성을 지속적으로 향상시키며, 강한 해석 가능성을 제공함을 의미합니다.



### DiFFPO: Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning (https://arxiv.org/abs/2510.02212)
- **What's New**: 본 논문에서는 DiFFPO(Diffusion Fast and Furious Policy Optimization)라는 통합 프레임워크를 제안하여, masked diffusion large language models (dLLMs)를 훈련하는 새로운 방법을 소개합니다. 이 프레임워크는 강화학습(Reinforcement Learning, RL)을 통해 dLLMs의 추론 능력을 더욱 빠르고 효과적으로 향상시키는 것을 목표로 합니다. 기존의 baseline 방법인 d1을 통합하여 오프 정책(off-policy) RL을 통해 주요 정책에 대한 보다 신뢰할 수 있는 근사치를 제공합니다.

- **Technical Details**: DiFFPO는 두 단계의 likelihood 근사화와 중요도 샘플링 보정(importance sampling correction)을 활용하여 샘플 효율성과 태스크 성능을 극대화합니다. 특히, 새로운 서라게이트 정책을 도입하여 대응 생성 수준에서 추가적인 레이턴트를 사용하는 방식으로 보다 정교한 근사화를 시도합니다. 문제 해결을 위한 RL 알고리즘을 통해 dLLMs의 다중 토큰 예측 능력을 더욱 잘 활용할 수 있게 됩니다.

- **Performance Highlights**: DiFFPO는 함수 평가 수(NFE)를 낮추면서도 더 높은 정확도를 달성하였으며, 수학 및 계획(Planning) 작업에서 dLLMs를 훈련하여 기존의 최고 성능을 초과하는 결과를 보여주었습니다. 이 연구는 dLLMs의 추론 시간을 줄이고 효율적인 RL 알고리즘 설계의 한계성을 넘어서기 위한 새로운 방향성을 제시합니다. 이러한 단계적인 접근이 결국 보다 출중한 LLM 모델 개발에 기여할 것으로 기대됩니다.



### StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets? (https://arxiv.org/abs/2510.02209)
- **What's New**: 이 논문은 StockBench라는 새로운 벤치마크를 소개하여 대형 언어 모델(LLMs)이 주식 거래 환경에서의 수익성 및 리스크 관리 능력을 평가할 수 있는 기준을 제공한다. 기존의 재무 벤치마크는 주로 정적인 지식 테스트에 초점을 맞췄으나, StockBench는 동적이고 반복적인 매매 결정을 반영하는 평가 방식을 채택하였다. 이를 통해 LLM 에이전트의 주식 거래 전략의 효과성을 더욱 정확하게 검증할 수 있게 된다.

- **Technical Details**: StockBench는 다음 세 가지 원칙을 준수하여 설계되었다: (1) 현실적인 시장 상호작용, (2) 지속적인 의사결정, (3) 데이터 오염 방지. 에이전트는 매일의 시장 신호(가격, 기업의 기초 정보, 뉴스 등)를 기반으로 연속적인 거래 결정을 내리며, 평가 메트릭은 누적 수익(cumulative return), 최대 손실(maximum drawdown), Sortino 비율을 사용하여 이루어진다. 평가 대상은 다양한 LLM 모델(예: GPT-5, Claude-4, Qwen3 등)으로 구성되었다.

- **Performance Highlights**: 실험 결과, 대부분의 LLM 에이전트는 단순 매수 및 보유 전략을 초과하여 성과를 내지 못한 반면, 몇몇 모델은 상대적으로 더 높은 수익을 기록하고 리스크를 효과적으로 관리할 수 있는 가능성을 보여주었다. 이는 LLM이 정적인 재무 지식 과제를 잘 수행하더라도 실제 거래 전략에는 한계를 갖고 있음을 나타낸다. 이 연구는 LLM 기반 재무 에이전트 개발에 있어 도전과 기회를 동시에 보여준다.



### Poolformer: Recurrent Networks with Pooling for Long-Sequence Modeling (https://arxiv.org/abs/2510.02206)
- **What's New**: Pooling 메커니즘을 통해 self-attention을 대체한 새로운 sequence-to-sequence 모델인 Poolformer를 소개합니다. 기존의 모델들이 직면한 시퀀스 길이에 대한 비효율성을 개선하며, 특히 긴 오디오 데이터에 효과적입니다. Poolformer는 SkipBlocks 구조를 사용하여 설계되어 있으며, 이로 인해 모델 성능과 훈련 속도가 개선되었습니다.

- **Technical Details**: Poolformer는 residual blocks와 down-pooling, up-pooling 층을 포함하는 SkipBlock으로 재귀적으로 정의됩니다. 이 모델은 recurrent layers를 사용하여 정보를 처리하며, 시퀀스 길이를 줄이는 pooling 작업을 포함하여 시간을 효율적으로 관리합니다. 이러한 구조는 또한 긴 시퀀스에 있을 때의 공칭 복잡도 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, Poolformer는 FID 및 IS 같은 지표에서 개선을 보이며, 훈련 속도가 빨라지고 overfitting을 방지합니다. 특히, 원시 오디오 데이터에서 최신 기술인 SaShiMi와 Mamba를 초과하는 성능을 보였습니다. 앞으로 Poolformer는 텍스트, 비전, 그리고 다중 모달 상황에서도 효과적으로 적용될 가능성을 가지고 있습니다.



### Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025 (https://arxiv.org/abs/2510.02202)
Comments:
          13 pages, 2 figures

- **What's New**: 이번 연구는 Chagas 질병을 정전도(ECG) 데이터를 통해 탐지하는 새로운 알고리즘 개발을 위한 PhysioNet Challenge 2025를 소개하고 있습니다. 준수할 데이터셋과 레이블을 확장하여 모델의 강건성을 높이는 혁신적인 접근 방식을 채택했습니다. 또한, 머신러닝 문제를 선별(task)로 프레임하기 위해 지역적인 혈청 검사 능력을 캡처하는 평가 지표를 도입했습니다.

- **Technical Details**: PhysioNet Challenge 2025에서는 378,624개의 12리드 ECG 레코드를 여러 출처에서 수집했습니다. 이 데이터셋은 훈련 세트와 숨겨진 검증 및 테스트 세트를 포함하여 준비되었습니다. 우리는 Chagas 양성 및 음성 사례에 대한 균형을 맞춰 다수의 공개 및 비공식 데이터 소스를 사용하였습니다.

- **Performance Highlights**: 도전 과제에는 111개 팀의 630명 이상의 참가자가 참여하였으며, 1300개 이상의 제출물이 있었습니다. 다양하고 창의적인 방법으로 Chagas 질병을 식별할 수 있는 알고리즘이 제안되었습니다. 이러한 노력은 Chagas 질병의 조기 탐지와 효과적인 치료에 기여할 것으로 기대됩니다.



### GRACE: A Language Model Framework for Explainable Inverse Reinforcement Learning (https://arxiv.org/abs/2510.02180)
- **What's New**: 이 논문에서는 전통적인 보상 모델이 해석하기 힘든 "블랙박스" 특성을 가진 반면, GRACE(Generating Rewards As CodE)라는 방법을 통해 대형 언어 모델을 사용하여 전문가 시연에서 직접 해석 가능한 보상 함수를 역설계하는 방법을 제안하고 있습니다. 이 방법은 전문가의 행동을 바탕으로 생성된 보상 함수를 기반으로 하며 실행 가능한 코드 형태로 제공되어, 검증이 가능하고 확인할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: GRACE는 진화적 탐색( evolutionary search )에서 영감을 받아 전문가의 시연을 기반으로 보상 모델이 되는 프로그램을 생성하고 다듬는 최적화 절차를 제공합니다. 이를 통해 생성된 보상은 이전에 비해 적은 양의 시연으로도 효과적인 보상을 학습하는 것이 가능합니다. 또한 GRACE는 다중 작업 설정에서 복잡한 보상 API를 구성하여 일반화 능력을 증가시킵니다.

- **Performance Highlights**: GRACE는 BabyAI와 AndroidWorld와 같은 벤치마크에서 실험적으로 검증되었으며, 강력한 정책을 생성하는 데 있어 경쟁사인 Imitation Learning( imitational learning )과 온라인 강화 학습( online RL )에서 진실 보상과 비교하여 우수한 성능을 보여주었습니다. 연구 결과, 이 방식으로 생성된 보상은 효과적이고 정보가 풍부한 중간 신호를 제공함으로써, 다양한 환경에서 효과적인 에이전트 구축을 가능하게 합니다.



### Flatness-Aware Stochastic Gradient Langevin Dynamics (https://arxiv.org/abs/2510.02174)
- **What's New**: 이번 연구에서는 Flatness-Aware Stochastic Gradient Langevin Dynamics (fSGLD)을 소개하여, 손실 지형에서 플랫 미니마(flat minima)를 효과적으로 찾을 수 있는 새로운 접근법을 제공합니다. fSGLD는 각 반복마다 이소트로픽 가우시안 노이즈에 의해 변동된 파라미터에서 계산된 확률적 경량(Bias) 경량을 사용하여 최적화합니다. 이는 곧, 곡률 정보를 암시적으로 파악하고, 고차원 비볼록 최적화 문제에서도 잘 작동하는 특징을 갖습니다.

- **Technical Details**: fSGLD는 랑베르드 동역학에 기반하여 만들어졌으며, 두 개의 주요 하이퍼파라미터인 역온도 매개변수(β)와 변동 규모(σ)를 적절하게 조합함으로써, fSGLD의 불변 측정치가 손실 함수의 글로벌 미니마이저에 집중될 수 있도록 합니다. 이는 특히 비볼록 최적화의 맥락에서 중요하며, Hessian trace로 정규화된 목표 함수의 플랫 미니마를 찾아가는 구체적인 이론적 근거를 제공합니다.

- **Performance Highlights**: fSGLD는 CIFAR-10N/100N과 WebVision과 같은 노이즈 레이블 데이터셋 및 대규모 비전 fine-tuning에서 높은 성능을 발휘하였습니다. 실험 결과, fSGLD는 SGD, AdamW, SGLD, SAM 등과 비교하여 일반화 및 강인성 면에서 우수한 성능을 보였습니다. 특히, 이론적으로 제안된 β와 σ의 조합을 사용함으로써, 기존 SGLD 관행과 비교할 때 성능이 유의미하게 향상되었습니다.



### Reinforcement Learning with Action-Triggered Observations (https://arxiv.org/abs/2510.02149)
- **What's New**: 이 논문은 행동에 의해 생성된 상태 관찰이 우연적으로 발생하는 강화 학습 문제를 다룹니다. 이를 Action-Triggered Sporadically Traceable Markov Decision Processes (ATST-MDPs)로 정식화하여 각 행동이 상태 관찰을 유발할 확률을 명시합니다. 따라서 에이전트는 즉각적인 상태 피드백이 없을 때에도 행동을 최적화하고 불확실성을 줄이기 위해 관찰을 유도하는 시점을 전략적으로 결정해야 합니다.

- **Technical Details**: ATST-MDP에서는 연속된 행동 수행이 필요하며, 이를 위한 Bellman 최적성 방정식이 도출됩니다. 이 연구는 선형 MDP 가정 하에서 최적 정책을 개발하고, 액션 시퀀스를 기반으로 한 가치 함수의 선형 표현을 보여줍니다. 또한, ST-LSVI-UCB 알고리즘을 제안하여 액션 트리거링 환경에서 에피소드 학습을 위한 통계적 오차 보장을 제공합니다.

- **Performance Highlights**: ST-LSVI-UCB는 에피소드 수 K, 특성 차원 d, 할인 인자 γ에 따라 $	ilde O(rac{	ext{sqrt}(Kd^{3}}{(1-	ext{γ})^{3}})$의 후회를 달성합니다. 이 논문은 관찰 제약이 있는 환경에서도 효율적인 학습이 가능하다는 이론적 토대를 마련합니다. 이를 통해 실제적인 정보 제약을 해결하는 데 기여할 수 있습니다.



### Policy Gradient Guidance Enables Test Time Contro (https://arxiv.org/abs/2510.02148)
- **What's New**: 본 연구는 전통적인 정책 경량화 방법들과의 결합을 통해 정책 경량화 방법에 고속성과 컨트롤성을 추가하는 새로운 방법인 Policy Gradient Guidance (PGG)를 소개합니다. PGG는 비조건부 분기를 추가하여 정책 경량화를 증대시키며, 이는 리트레이닝 없이도 작동을 조절할 수 있는 새로운 컨트롤 노브를 제공합니다. 특히, 기존의 미분 모델에서의 classifier-free guidance (CFG)를 정책 경량화 방법에 적용하여 보다 안정적이고 효율적인 학습을 가능하게 합니다.

- **Technical Details**: PGG는 정책 경량화 방법인 Proximal Policy Optimization (PPO)을 활용하여 비조건부 정책을 추가하고, 이를 통해 조건부 및 비조건부 로그를 혼합하여 가볍고 직관적인 조정 메커니즘을 구현합니다. 이는 학습 과정에서 表示가 가능하며, 특히 γ(감독 강도)와 같은 조정 가능한 매개변수를 통해 다양한 상황에서의 정책 조정이 가능합니다. 이 연구는 기존의 패턴 인식 및 미분 기반 모델에서의 응용을 넘어서서 일반적인 정책 경량화 방법론에 적용 가능한 가능성을 보여줍니다.

- **Performance Highlights**: PGG의 성능 평가 결과, 단순한 이산 작업 및 적은 샘플 수의 환경에서는 conditioning dropout이 효과적인 반면, 적절한 조정 매개변수(γ>1)를 사용하는 경우 복잡한 작업에서도 더 강력하고 안정적인 성능을 얻을 수 있음을 보여주었습니다. 또한, 우리는 이 연구를 통해 정책 경량화 방법이 기존의 미분 기술에 국한되지 않고, 온라인 강화 학습에서 보다 넓은 적용 가능성을 가지게 됨을 입증했습니다.



### Catalyst GFlowNet for electrocatalyst design: A hydrogen evolution reaction case study (https://arxiv.org/abs/2510.02142)
Comments:
          5 pages, 2 figures. Accepted to NeurIPS AI for Materials Workshop 2025

- **What's New**: 이 논문에서는 효율적이고 저렴한 에너지 저장의 중요성을 강조하며, 전기 촉매가 수소 에너지 저장(Hydrogen Energy Storage, HES)에서 핵심 역할을 한다고 설명합니다. Catalyst GFlowNet이라는 새로운 생성 모델을 도입하여, 형성 및 흡착 에너지를 예측하여 효율적인 촉매로 작용할 수 있는 결정 표면을 설계합니다. 이 모델은 수소 발생 반응에서 플래티넘을 가장 효율적인 촉매로 확인함으로써 그 가능성을 입증하였습니다.

- **Technical Details**: Catalyst GFlowNet은 Crystal-GFN 프레임워크를 기반으로 하여, 다양한 촉매 물질을 생성하기 위해 고안되었습니다. 이 구조는 세 단계(샘플 생성, 샘플 준비 및 보상 계산)로 나뉘어 있으며, 보상 값에 비례하여 촉매 표현을 샘플링할 수 있도록 GFlowNet 객체를 훈련시킵니다. 광범위한 샘플을 생성하여 실험이 고품질의 촉매 물질을 발견할 수 있는 기회를 증대시키는 것이 이 접근법의 장점입니다.

- **Performance Highlights**: 사례 연구를 통해 Catalyst GFlowNet이 가장 잘 알려진 촉매를 재발견할 수 있음을 확인하였습니다. 이 결과는 저비용이면서도 효율적인 촉매 개발에 한 발짝 나아갈 수 있는 가능성을 보여줍니다. 이 발전은 미래의 산소 발생 반응(Oxygen Evolution Reaction, OER) 및 새로운 재료 발굴을 위한 연구에 확대될 수 있는 기반을 제공합니다.



### DAG DECORation: Continuous Optimization for Structure Learning under Hidden Confounding (https://arxiv.org/abs/2510.02117)
- **What's New**: 이번 논문에서는 잠재 혼란을 포함한 선형 가우시안 구조 방정식 모델(SEMs)에 대한 구조 학습을 다룹니다. 기존의 방법들은 독립적인 오류에 강점을 보이지만, 그런 경우가 아니면 문제가 발생합니다. 이를 해결하기 위해서 우리는 DECOR라는 새로운 방법을 제안하며, 이는 DAG 및 상관되는 잡음 모델을 동시에 학습하는 방법입니다.

- **Technical Details**: DECOR는 매개변수 불식별성을 위한 간단한 충분 조건을 제시합니다. 높은 에너지 마진을 가진 오류 공분산 및 방향 그래프의 특징이 만족되면, 관찰 공분산에서 함수적 맵이 단사이므로 구조와 잡음이 독특하게 결정됩니다. 이 방법은 부드럽고 비순환적인 그래프 업데이트와 구면적 잡음 업데이트를 번갈아 진행하며, 경량 bow complementarity penalty 또는 사후 조정 단계를 포함할 수 있습니다.

- **Performance Highlights**: DECOR는 여러 혼란 밀도 및 그래프 밀도를 변형한 합성 벤치마크에서 좋은 성능을 보이며 기존의 강력한 베이스라인을 초과하거나 동등한 수준을 유지합니다. 특히, 혼란이 비단순할 때 특히 강건하며, 단순한 경우에서도 경쟁력을 유지합니다. 이 연구는 선형 가우시안 SEMs의 기초 데이터에서 인과 구조를 배우는 데 중요한 발전을 보여줍니다.



### Ensemble Threshold Calibration for Stable Sensitivity Contro (https://arxiv.org/abs/2510.02116)
Comments:
          10 pages, 6 tables

- **What's New**: 이 연구에서는 대규모 공간 혼합 및 엔티티 매칭 작업에서 정확한 recall 제어의 중요성을 강조합니다. 기존의 신뢰 구간 기법들은 목표 이상으로 초과하는 경향이 있으며, 고르지 않은 스코어 분포 하에서도 변동성이 높습니다. 제안된 엔드 투 엔드 프레임워크는 수천만 개의 기하학적 쌍에 대해 ±1%의 변동성으로 정확한 recall을 달성하며, TPU에 친화적입니다.

- **Technical Details**: 프레임워크의 첫 단계는 equigrid 경계 박스 필터를 사용해 후보 쌍에 대한 개수를 두 자리 만큼 줄이는 것입니다. 그런 다음 deterministic xxHash bootstrap 샘플을 통해 경량 신경망 랭커를 훈련시키고, 이의 스코어는 단일 전방 통과를 통해 나머지 쌍에 전파되어 재현 가능한 점수-데시일-계층화 보정 세트를 구축하는 데 사용됩니다. 최종적으로 네 가지 보정 추정기를 결합하여 불확실성을 감소시킵니다.

- **Performance Highlights**: 두 개의 실제 지적 데이터셋(약 6.31M 및 67.34M 쌍)에 대한 평가에서, 제안한 접근법은 작은 오차 내에서 recall 목표를 지속적으로 충족하며, 다른 보정 방법에 비해 중복 검증을 감소시킵니다. 또한, 단일 TPU v3 코어에서 4분 이내로 엔드 투 엔드 실행이 가능합니다.



### Hybrid Deep Learning Modeling Approach to Predict Natural Gas Consumption of Home Subscribers on Limited Data (https://arxiv.org/abs/2510.02115)
- **What's New**: 이번 연구는 이란 잔잔(Zanjan) 지역의 주거 고객을 위한 가스 소비를 예측하는 데 머신 러닝 모델(LSTM, GRU, Hybrid BiLSTM-XGBoost)을 사용합니다. 이란은 매년 인구 증가와 에너지 소비의 증가로 가스 압력 저하 및 정전과 같은 문제를 겪고 있으며, 이러한 문제를 해결하기 위해 주민들의 가스 소비를 효과적으로 관리해야 합니다.

- **Technical Details**: 연구에서는 2017년부터 2022년까지 수집된 가스 소비 및 기상 데이터셋을 사용하여 모델을 훈련하고 평가했습니다. Hybrid BiLSTM-XGBoost 모델이 다른 모델들보다 정확도가 높았으며, Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE) 및 Mean Percentage Error (MPE) 값이 낮았습니다. 이러한 결과는 머신 러닝 접근법이 가스 소비 관리에 효과적으로 활용될 수 있음을 나타냅니다.

- **Performance Highlights**: 하이브리드 모델은 데이터가 제한적인 상황에서도 강력한 성능을 보여주었으며, 기후 및 지리적 요인을 예측 모델에 통합하는 것의 중요성을 강조합니다. 이 연구는 머신 러닝 방법을 통한 자원 관리 및 계절적 부족 문제를 줄이는 데 기여할 수 있음을 보여줍니다.



### PENEX: AdaBoost-Inspired Neural Network Regularization (https://arxiv.org/abs/2510.02107)
- **What's New**: 이 논문은 AdaBoost의 약한 학습자(weak learner)를 사용하여 새로운 Penalized Exponential Loss(PENEX)를 소개합니다. PENEX는 다중 클래스 지수 손실(multi-class exponential loss)의 새로운 수식을 제공하며, 이는 최적화에 더 적합하다는 이점이 있습니다. 이 손실 함수는 데이터 포인트의 마진(margin)을 내포적으로 최대화하는 특성을 가지고 있습니다.

- **Technical Details**: PENEX는 피셔 일관성(Fisher consistency)을 만족하며, 조건부 및 무조건적 경우 모두에서 적용 가능합니다. 또한, PENEX의 특징 중 하나는 낮은 비용으로도 고유한 최소값(unique minimum)을 제공한다는 점입니다. 최적화 과정에서 PENEX는 약한 학습자들을 매개변수화(parameterize)하는 다양한 기법을 탐지합니다.

- **Performance Highlights**: 컴퓨터 비전(computer vision) 및 자연어 처리(natural language processing) 작업에서 PENEX는 기존의 방법보다 더 나은 정규화 효과를 보여주었습니다. 이 연구는 PENEX가 깊은 신경망(deep neural network)의 효과적인 훈련 및 미세 조정(fine-tuning)을 위한 AdaBoost에 영감을 받은 대안으로서의 잠재력을 지니고 있음을 강조합니다.



### Learning Model Representations Using Publicly Available Model Hubs (https://arxiv.org/abs/2510.02096)
- **What's New**: 이 연구는 비정형 모델 저장소에서 다운로드한 임의 모델을 사용하여 weight space learning(WSL)을 수행하는 새로운 방식의 백본을 제안합니다. 기존의 모델 조합(모델 주제군) 접근법 대신, Hugging Face와 같은 비정형 모델 리포지토리에서 훈련된 다양한 모델을 활용하여 WSL의 한계를 극복하고자 합니다. 이 새로운 접근법은 무작위로 수집된 모델의 가중치를 기반으로 효율적으로 학습할 수 있는 기능을 제공해 줍니다.

- **Technical Details**: 이 연구에서는 encoder-decoder transformer 아키텍처를 기반으로 한 최초의 WSL 백본을 제안합니다. 이 백본은 특정 모델 아키텍처나 훈련 데이터셋에 종속되지 않고 훈련될 수 있도록 설계되었습니다. 총 1710억 개의 개별 가중치로 구성된 이 모델은 다양한 아키텍처와 데이터셋 조합에 대한 단일 가중치 공간 표현을 학습할 수 있습니다.

- **Performance Highlights**: Hugging Face에서 학습된 모델을 통해 생성된 가중치 공간 표현은 기존의 실험실 환경에서 훈련된 모델 조합보다 더 강력한 성능을 보입니다. 특히, 다양한 아키텍처/데이터셋 조합에 대한 일반화 능력이 뛰어나며, 새로운 데이터 모달리티에 대해 더 나은 예측 성능을 발휘합니다. 이 연구는 WSL 커뮤니티가 직면한 큰 제한 사항을 극복하는 데 기여합니다.



### KAIROS: Unified Training for Universal Non-Autoregressive Time Series Forecasting (https://arxiv.org/abs/2510.02084)
- **What's New**: KAIROS는 시간 시계열 예측을 위한 새로운 비자기회계 프레임워크로, 세그먼트 수준의 다중 봉우리 분포를 직접 모델링하여 기존의 방식보다 더 나은 성능을 발휘합니다. 기존의 자가회귀(AR, autoregressive) 접근 방식을 피하고 오류 축적을 방지하며, 실시간 응답을 필요로 하는 웹 애플리케이션에 최적화된 예측을 제공합니다. 특히, KAIROS는 경험적 결과를 통해 다양한 벤치마크에서 제로샷 제너럴리제이션(Zero-shot generalization) 성능을 보여주어 주목받고 있습니다.

- **Technical Details**: KAIROS는 세 가지 상호작용 메커니즘으로 구성되며, 각 세그먼트에 혼합 전문가(Mixture-of-Experts) 예측 헤드를 사용하여 다중 봉우리 분포 문제를 해결합니다. 이 모델은 외부 요인(learnable exogenous vectors)을 포착하여 개별 세그먼트에 대한 고유한 조건 정보를 제공하고, 세그먼트 간 인과적 관계를 유지하기 위해 인과 잔여 노이즈(Segment Causal Residual Noise) 기법을 도입하여 세그먼트 예측의 일관성을 높입니다. 이러한 접근 방식은 AR 모델의 비효율성을 피하면서도 지속적인 예측을 가능하게 합니다.

- **Performance Highlights**: KAIROS는 여러 예측 벤치마크에서 최신 기술(state-of-the-art)의 제로샷 성능을 달성하며, 자가회귀 모델에 비해 현저히 빠른 추론 속도를 자랑합니다. 장기 예측 과제에서도 뚜렷한 장점을 보여주어, 종합적으로 기존의 시간 시계열 모델보다 우수한 효율성과 정확성을 제공합니다. 이는 KAIROS가 변화하는 웹 환경에 적합한 확장 가능한 솔루션임을 증명합니다.



### Fine-Tuning Flow Matching via Maximum Likelihood Estimation of Reconstructions (https://arxiv.org/abs/2510.02081)
- **What's New**: 이 논문은 Flow Matching (FM) 알고리즘이 로봇 조작과 같은 생성 작업에서 뛰어난 성능을 발휘하는 방법을 제안한다. FM은 특히 시뮬레이션이 필요 없는 훈련 접근 방식을 성공적으로 활용하고 있지만, 훈련과 추론 사이의 격차가 존재한다. 이러한 점에서 FM의 훈련 손실과 추론 오류 간의 관계를 이론적으로 분석하고, 이를 해결하기 위해 최대 우도 추정(Maximum Likelihood Estimation, MLE)을 사용하는 미세 조정 방법을 제안한다.

- **Technical Details**: FM은 ODE(Ordinary Differential Equations)의 부드러운 공식을 기반으로 하여 훈련 손실과 추론 오류 간의 관계를 이론적으로 분석한 후, 직접적으로 재구성 오류를 기반으로 FM을 미세 조정하는 방법을 제안한다. 이 방법은 직관적인 미세 조정과 잔여 기반 미세 조정 방법을 포함한다. 또한, 잔여 기반 접근 방식은 모델 내에서 수축(contraction) 특성을 통합할 수 있어 모델의 견고성과 해석 가능성에 기여한다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 FM의 추론 성능을 신뢰성 있게 향상시킨다는 것을 보여준다. 특히 IMAGE 생성 및 로봇 조작 작업에서 이 방법이 효과적이라는 근거를 제공한다. 또한, FM의 속도가 빠르면서도 강력한 안정성과 의미론적 질을 제공하는 모델로서의 가능성을 실증적으로 입증하였다.



### Inferring Optical Tissue Properties from Photoplethysmography using Hybrid Amortized Inferenc (https://arxiv.org/abs/2510.02073)
- **What's New**: 이번 연구에서는 기존의 PPG 신호를 해석할 수 있는 새로운 생물물리 모델인 PPGen을 소개합니다. PPGen은 PPG 신호의 웨이브폼 특성을 생리학적(physiological) 및 광학적(optical) 매개변수와 명확하게 연관짓습니다. 또한, 빠르고 견고한 하이브리드 감쇠 추론(hybrid amortized inference, HAI) 알고리즘을 도입하여 모델의 오차를 보정하면서 PPG 신호에서 생리적 매개변수를 효율적으로 추정할 수 있도록 합니다.

- **Technical Details**: PPGen은 PPG 신호의 동적 및 정적 매개변수를 모델링하기 위해 다층 피부 구조를 이용하여 생리학적 데이터를 시뮬레이션합니다. 연구팀은 PPG 양상으로부터 11개의 매개변수를 추론하며, 이 중 9개는 정적 매개변수이며, 나머지 2개는 PPG 풀이 진행되는 동안 혈액량 변화에 해당하는 동적 매개변수입니다. HAI는 실제 PPG 데이터를 통해 모델을 보강하고, 신뢰성 있는 매개변수 추정을 가능하게 합니다.

- **Performance Highlights**: 연구 결과 HAI는 다양한 노이즈 및 센서 조건에서도 생리적 매개변수를 정확하게 추정할 수 있는 능력을 보여주었습니다. 이러한 결과는 DL 기반 기능을 유지하면서도 PPG 모델이 임상적 해석 및 하드웨어 설계를 지원하는 방식으로 발전할 수 있는 경로를 제시합니다. PPGen과 HAI를 통해 PPG 데이터를 활용한 진단적 가능성을 크게 확장시킬 수 있습니다.



### Adaptive Heterogeneous Mixtures of Normalising Flows for Robust Variational Inferenc (https://arxiv.org/abs/2510.02056)
Comments:
          2 Figures and 2 tables

- **What's New**: 이 논문에서는 다양한 형태의 분포에 대해 일관된 동작을 보이지 않는 단일 흐름 모델의 한계를 극복하기 위해 Adaptive Mixture Flow Variational Inference (AMF-VI)라는 새로운 방법론을 제안합니다. 이 방법은 여러 통합된 흐름(MAF, RealNVP, RBIG)을 활용하여 두 단계로 구성된 훈련 과정을 통해 각 흐름의 매개변수와 집합 가중치를 추정합니다. AMF-VI는 기존 단일 흐름 기반 모델들에 비해 더 낮은 음수 로그 우도(Negative Log-Likelihood; NLL)를 달성하고 전반적으로 안정적인 향상을 제공합니다.

- **Technical Details**: AMF-VI는 먼저, 서로 다른 아키텍처를 가진 전문가들(MAF, RealNVP, RBIG)을 독립적으로 훈련시키고(1단계), 이후에 고정된 전문가 매개변수를 바탕으로 전 세계 혼합 가중치를 우도 기반의 이동 평균 업데이트를 통해 추정하는 구조를 가지고 있습니다. 이 과정에서 데이터에 독립적인 빠른 게이팅 기법을 사용하여 각 전문가에 대한 질량을 재배치하며, 이는 각 흐름의 장점을 극대화하는 데 도움을 줍니다. 또한, 이동 평균 업데이트 메커니즘의 동기를 제공하고 전문가 클래스를 요약하는 방법론이 포함됩니다.

- **Performance Highlights**: AMF-VI는 바나나, X-형, 두 개의 달, 링, 이항 모드, 다섯 개 모드 혼합 등 6개의 전형적인 후방 분포 패밀리에서 평가되었으며, 이 모든 경우에서 기존 단일 흐름 기준선 모델들보다 낮은 NLL을 기록했습니다. 또한, Wasserstein-2(W2) 거리와 최대 평균 차이(MMD)에서 안정적인 개선을 보이며, 다양한 형태와 모드에 대해 더 높은 강건성을 입증합니다. 이러한 결과는 AMF-VI가 다양한 후방 분포에 대해 안정적이고 실용적인 VI 접근 방식을 제공함을 보여줍니다.



### Mathematical Modeling and Convergence Analysis of Deep Neural Networks with Dense Layer Connectivities in Deep Learning (https://arxiv.org/abs/2510.02049)
- **What's New**: 이번 논문은 깊은 신경망(DNN)에서의 밀집 연결성을 수학적으로 모델링하고 분석합니다. 기존의 ODE(Ordinary Differential Equation) 관점과는 달리, 각 레이어를 비선형 적분 방정식으로 모델링함으로써 밀집 비국소(Dense Non-local, DNL) 프레임워크를 제안합니다. 이는 DenseNet 등의 다양한 아키텍처에 대해 더욱 강력한 이론적 기초를 제공합니다.

- **Technical Details**: DNL 프레임워크는 각각의 네트워크 레이어가 시간 단계에 해당하며, 깊이의 한계에서의 비선형 적분 방정식을 자연스럽게 도출합니다. 본 논문에서는 이론적 배경을 바탕으로 학습 문제를 최적 제어 문제로 변환하고 학습 문제의 격자 시간에서 연속 시간으로의 수렴 결과를 입증합니다. 이를 통해 최적 값의 수렴과 최적 해의 부분 수열 수렴을 보여줍니다.

- **Performance Highlights**: 밀집 연결 네트워크의 동적 시스템 모델링 및 수렴 분석을 통해 신경망 아키텍처의 수학적 이해가 증진됩니다. 이러한 아키텍처는 깊은 네트워크 훈련 시 안정성을 강화할 수 있는 가능성을 제시합니다. 제안된 결과는 DenseNet, DenseFormer, Dense Residual Transformer와 같은 다양한 밀집 연결 DNN에 적용될 수 있습니다.



### FairContrast: Enhancing Fairness through Contrastive learning and Customized Augmenting Methods on Tabular Data (https://arxiv.org/abs/2510.02017)
Comments:
          Accepted to NeurIPS 2025 - Reliable ML Workshop

- **What's New**: 이 연구는 표 형태(tabular) 데이터에서 편향을 완화하고 공정한 표현(representation)을 학습하기 위해 특별히 설계된 대비 학습(contrastive learning) 프레임워크를 제안합니다. 기존의 최첨단 대비 학습 모델에 비해 긍정적 샘플 쌍을 전략적으로 선택하고 지도(supervised) 및 비지도(self-supervised) 대비 학습을 적용하여 큰 편향 감소를 달성했습니다. 이처럼, 공정한 표현 학습의 필요성이 꾸준히 강조되며, AI 시스템의 사회적 영향을 고려하는 것이 진정한 도전임을 보여줍니다.

- **Technical Details**: 표 형태 데이터에서의 대비 학습은 고유한 도전 과제를 나타내며, 데이터 내에 구조적 정보(예: 공간적, 의미적 관계)가 부족합니다. 본 연구에서 제안된 방법은 각 특권 샘플을 긍정적 샘플 쌍으로 설정하고, 부정적 샘플 쌍은 동일한 클래스와 민감한 속성을 가진 샘플로 구성됩니다. 지도 대비 학습 손실(supervised contrastive learning loss)과 정보 최대화 손실(InfoNCE loss)을 포함하는 이 프레임워크는 공정성을 목표로 하여 기존 방법보다 향상된 성능을 나타냅니다.

- **Performance Highlights**: 제안한 접근법은 다양한 공정성 데이터셋에서 평가되었으며, 결과는 기존의 최첨단 모델과 비교할 때 편향의 유의미한 감소를 보여줍니다. 또한, 본 프레임워크는 다운스트림 작업(downstream tasks)에서 사용할 수 있는 공정한 표현을 학습하는 데 성공하였습니다. 공정성을 유지하면서도 예측 정확도에 대한 최소한의 트레이드 오프(trade-off)를 경험한다고 언급됩니다.



### Normality Calibration in Semi-supervised Graph Anomaly Detection (https://arxiv.org/abs/2510.02014)
Comments:
          17 pages

- **What's New**: GraphNC는 라벨이 있는 노드와 라벨이 없는 데이터를 활용하여 그래프 이상 탐지에서 정상성을 교정하는 새로운 프레임워크입니다. 이 프레임워크는 교사 모델(teacher model)을 통해 이상 점수(anomaly score)와 노드 표현(node representation) 공간에서 정상성을 효율적으로 학습하도록 설계되었습니다. 구체적으로, ScoreDA(이상 점수 분포 정렬)와 NormReg(왜곡 기반 정상성 규제)의 두 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: GraphNC의 ScoreDA는 교사 모델의 점수 분포와 모델 점수를 정렬하여 이상 점수를 최적화하며, 이는 정상 및 비정상 클래스의 점수를 양쪽 끝으로 향하게 하여 명확한 구분을 제공합니다. NormReg는 레이블이 있는 노드에서만 일관성 손실을 최소화하여 정상 노드의 표현을 더 밀집하게 만들도록 설계되었습니다. 이렇게 두 성분을 공동으로 최적화함으로써 GraphNC는 더 일반화된 정상성 표현을 학습하고 더 명확한 이상 점수를 생성할 수 있습니다.

- **Performance Highlights**: GraphNC는 다양한 사전 훈련된 교사 모델을 통합할 수 있는 유연한 프레임워크로, 데이터 재구성 기반, 단일 클래스 기반, 이상 생성 기반 접근 방식에서 일관된 GAD 성능 향상을 이끌어냅니다. 실험을 통해 GraphNC는 교사 모델이 더 강할수록 더 나은 성능을 발휘함을 입증했습니다. 전체 6개의 벤치마크 데이터셋을 대상으로 한 종합적인 실험 결과에서도 이러한 성능 향상이 확인되었습니다.



### PepCompass: Navigating peptide embedding spaces using Riemannian Geometry (https://arxiv.org/abs/2510.01988)
- **What's New**: 본 연구에서는 PepCompass라는 새로운 프레임워크를 소개하여 항균 펩타이드 탐색 및 최적화를 가능하게 합니다. 이 프레임워크는 $	ext{Union of } oldsymbol{	ext{κ}}$-안정 리만 다양체(Union of $oldsymbol{	ext{κ}}$-Stable Riemannian Manifolds)에서 영감을 받아 복잡한 펩타이드 공간의 지역 기하 구조를 캡처합니다. 새로운 지오데식 검색 기법인 Potential-minimizing Geodesic Search (PoGS)를 포함하여, 더 높은 항균 활성을 가진 펩타이드의 발견이 용이해집니다.

- **Technical Details**: PepCompass는 두 가지 지역 탐색 방법을 제공하며, 그 중 하나인 Second-Order Riemannian Brownian Efficient Sampling (SORBES)은 리만 브라운 운동의 수렴하는 두 번째 차수 근사를 제공합니다. 또 다른 방법인 Mutation Enumeration in Tangent Space (MUTANG)은 접선 방향을 아미노산 치환으로 해석하여 효율적인 탐색을 가능하게 합니다. 이 두 가지 접근법은 지역 탐색을 위한 Local Enumeration Bayesian Optimization (LE-BO) 알고리즘으로 통합되어 있습니다.

- **Performance Highlights**: PepCompass의 효능을 검증하기 위해 진행된 in-vitro 실험에서는 이 시스템이 100%의 성공률을 기록하며 4개의 새로운 항균 펩타이드 씨앗을 도출했습니다. 그 후 LE-BO를 통해 25개의 고활성 펩타이드를 발견하였고, 이들 중 다수는 다제내성 세균에 대해서도 활성을 보여주었습니다. 이러한 결과는 기하학적 탐색이 항균 펩타이드 설계에 있어 강력한 새로운 패러다임이 될 수 있음을 시사합니다.



### Private Federated Multiclass Post-hoc Calibration (https://arxiv.org/abs/2510.01987)
- **What's New**: 이번 연구에서 제안한 방법은 Federated Learning(FL) 환경에서 머신러닝 모델의 calibration을 수행하는 혁신적인 접근법입니다. 특히, 기존의 중앙집중 방식의 histogram binning과 temperature scaling 기법을 FL에 통합하여 클라이언트 간의 강한 이질성을 다룰 수 있는 새로운 방법론을 정의하였습니다. 이는 의료나 금융과 같이 calibration이 필수적인 분야에서 FL을 효과적으로 적용하려는 필요성을 충족시키기 위한 것입니다.

- **Technical Details**: 본 연구에서는 post-hoc calibration을 통해 모델이 예측한 확률을 실제 결과와 더욱 잘 일치하도록 조정하는 과정을 진행합니다. 연구진은 non-IID와 user-level Differential Privacy(DP) 하에서 calibration의 정확성에 미치는 영향을 분석하며, 과거의 기존 방법들이 FL 환경에서 어떻게 작용하는지 평가하였습니다. 또한, FedBinning과 FedScaling이라는 두 가지 프레임워크를 제안하여 강한 이질성을 고려한 효율적인 조정을 가능하게 합니다.

- **Performance Highlights**: 연구 결과, 제안한 프레임워크들이 7개의 벤치마크 데이터셋에서 강한 이질성 환경에서 효과적으로 calibration을 수행한다는 것을 실험을 통해 검증하였습니다. 실제로, FL 환경에서 naive한 방법을 적용하는 것이 오히려 모델의 정확성을 저하시킬 수 있음을 발견하였고, 우리의 tailored enhancement가 기존의 방법들보다 뛰어난 성능을 보임을 강조했습니다. 특히, user-level DP 환경에서는 scaling 방법이 binning 접근법보다 더욱 효과적임을 입증하였습니다.



### $\text{G}^2$RPO: Granular GRPO for Precise Reward in Flow Models (https://arxiv.org/abs/2510.01982)
Comments:
          Github Page: this https URL

- **What's New**: 최근 온라인 강화 학습 (RL)의 확산 및 흐름 모델 통합이 생성 모델을 인간의 선호에 맞추기 위한 유망한 접근 방식으로 떠오르고 있습니다. 이 연구에서는 노이즈 제어 과정에서 확률적 미분 방정식 (SDE)을 통해 다양한 비노이즈 방향을 생성하여 강화 학습 탐사를 지원하는 새로운 Granular-GRPO ($G^2$RPO) 프레임워크를 제안합니다. 기존 방법들이 높은 가치 샘플 탐사에서 효과적이었지만 선호 정렬이 부족했던 문제를 해결하기 위해 보다 정밀한 보상 평가 방법을 도입했습니다.

- **Technical Details**: 제안된 방법에서는 Singular Stochastic Sampling 전략을 사용하여 각 SDE 변동에서 충실한 보상을 제공하며, Multi-Granularity Advantage Integration 모듈을 통해 다양한 확산 스케일에서의 장점을 집계하여 샘플링 방향을 종합적으로 평가합니다. 이 프레임워크는 기존 흐름 기반 GRPO 방법들과 비교하여 보다 정확하고 포괄적인 보상 신호를 제공합니다. 강화 학습의 모델 훈련 과정에서 보상 신호를 특정 시행과 강하게 연결하여 안정적인 최적화를 가능케 합니다.

- **Performance Highlights**: 다양한 보상 모델에 대한 실험 결과 $G^2$RPO가 기존 흐름 기반 GRPO 기준 성능을 현저하게 초과함을 보여줍니다. $G^2$RPO는 텍스트 프롬프트에 대한 적합성과 세부 사항의 충실함에서 또 다른 장점을 발휘하며, 훈련 과정에서 안정적이고 유의미한 개선을 나타냅니다. 이 연구 결과는 시각적 생성에서 인간의 선호를 보다 효과적으로 반영할 수 있는 가능성을 열어줍니다.



### Moon: A Modality Conversion-based Efficient Multivariate Time Series Anomaly Detection (https://arxiv.org/abs/2510.01970)
- **What's New**: 이 논문은 Moon이라는 신규 다변량 시계열(MTS) 이상 탐지 프레임워크를 제안합니다. Moon은 기존의 재구성 기반 및 예측 기반 방법의 한계를 극복하기 위해 설계되었으며, 유효한 분류 방법을 포함하여 이상 탐지의 효율성과 정확성을 향상시킵니다. 이를 통해 제안된 방법은 더 나은 해석 가능성과 세밀한 이상 분석 보고서를 제공합니다.

- **Technical Details**: Moon 프레임워크는 새로운 다변량 마코프 전이 장(MV-MTF) 기술을 통해 수치 시계열 데이터를 이미지 표현으로 변환합니다. 다변량 CNN(Multimodal-CNN)을 사용해 수치적 데이터와 이미지 데이터를 통합하고, 공유 파라미터 모델을 통해 학습 효율성을 증가시킵니다. 또한, SHAP 기반의 이상 설명기를 통해 이상 발생에 기여하는 중요한 변수를 식별하여 해석 가능성을 높입니다.

- **Performance Highlights**: 실제 MTS 데이터셋 6개에 대한 실험 결과, Moon은 6가지 최신 방법 대비 최대 93%의 효율성 향상과 4%의 정확도 개선, 10.8%의 해석 성능 향상을 보였습니다. 이를 통해 Moon은 다변량 시계열 이상 탐지의 정확성과 효율성을 크게 증대시키는 것으로 나타났습니다.



### Lower Bounds on Adversarial Robustness for Multiclass Classification with General Loss Functions (https://arxiv.org/abs/2510.01969)
- **What's New**: 이번 연구에서는 임의의 손실 함수 아래에서 다중 클래스 설정에서 적대적(Adversarial) 강건성 분류를 고려합니다. 우리는 강건한 리스크 최소화 문제의 이중 및 바리센트릭(Barycentric) 재구성을 도출하고, 손실 함수가 0-1 손실에서 사용할 수 있는 결과를 확장하는 새로운 사례들에 대한 명시적인 특성을 제공합니다.

- **Technical Details**: 연구는 크로스 엔트로피(Cross-Entropy), 거듭제곱 형태의 손실 함수, 그리고 이차 손실(Quadratic Loss)과 같은 중요한 경우를 포함합니다. 이러한 재구성은 적대적 리스크에 대한 날카로운 하한(Sharp Lower Bounds)을 효율적으로 계산할 수 있게 해주며, 0-1 손실 설정을 넘어서는 강건 분류기 설계를 도와줍니다.

- **Performance Highlights**: 연구는 적대적 강건성, $ eta$-공정 포장 문제(α-fair packing problems), 그리고 다양한 양의 측정에 대한 일반화된 바리센터 문제(Generalized Barycenter Problems) 간의 흥미로운 연결을 발견했습니다. 또한, 이론적 결과들은 크로스 엔트로피 손실 함수로 적대적 리스크에 대한 더 엄격한 하한을 얻기 위한 설명적인 수치 실험들과 함께 제시됩니다.



### StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold (https://arxiv.org/abs/2510.01938)
Comments:
          Accepted as a spotlight at NeurIPS 2025

- **What's New**: 이 논문에서는 Low-rank Adaptation (LoRA)의 기하학적 구조를 활용한 새로운 접근 방식을 제안합니다. 저자들은 Stiefel manifold의 제약 조건을 사용하는 3-팩터 분해를 통해 LoRA의 입력과 출력 서브스페이스를 최적화하여 성능을 향상시킵니다. 이 방법은 기존의 LoRA보다 더 높은 정확도를 달성할 수 있는 가능성을 제시합니다.

- **Technical Details**: 제안된 방법인 Stiefel Low-rank Adaptation (StelLA)은 LoRA의 입력과 출력 서브스페이스를 지속적으로 최적화할 수 있는 구조를 갖추고 있습니다. 이 접근 방식은 $U$와 $V$를 Stiefel manifold에 위치시키고, 이를 통해 orthonormality를 유지합니다. Stiefel manifolds에서의 기하학적 최적화 디자인을 적용하여 기존의 유클리드 최적화 도구를 리만 구조로 변환할 수 있습니다.

- **Performance Highlights**: 여러 다운스트림 작업에서 StelLA의 성능을 평가한 결과, 기존 LoRA 변형들에 비해 일관되게 우수한 성능을 보여주었습니다. 예를 들어, 상식 추론(common sense reasoning)에서 1.3 포인트, 수학 및 코드 생성에서 2.33 포인트 개선이 있었습니다. 이러한 성능 향상은 자연어 이해, 생성, 이미지 분류 및 텍스트-이미지 생성까지 다양한 작업에 걸쳤습니다.



### Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinemen (https://arxiv.org/abs/2510.01910)
Comments:
          14 pages

- **What's New**: 이 논문은 GNN(Graph Neural Networks)과 LLM(Large Language Models)의 통합을 다룹니다. 구체적으로는 RoGRAD라는 새로운 프레임워크를 소개하며, 이는 Retrieval-Augmented Generation 기법을 통해 데이터의 약점을 보완하는 이터레이티브(Iterative)한 접근 방식을 제공합니다. 또한 LLM을 사용한 기존의 방법들과 GNN 기반 접근 방식의 비교를 통해 LLM이 항상 더 우수하다는 기존 가정을 재검토합니다.

- **Technical Details**: RoGRAD는 기존의 LLM-enhanced 메서드와 달리, Retrieval-Augmented Contrastive Refinement를 기반으로 합니다. 이 방식은 다양한 데이터를 동적으로 재조정하여 GNN의 성능을 향상시키는 데 집중합니다. 추가적으로, R2CL(Contrastive Learning with RAG Refinement)을 도입하여 레이블 일관성과 클래스 간 변별력을 강화합니다. 실험을 통해 RoGRAD가 전통적인 GNN 및 LLM 강화 기법에 비해 82.43% 평균 성능 증가를 달성했다는 성과를 보여줍니다.

- **Performance Highlights**: RoGRAD는 기존 GNN에서 발생하는 데이터의 약점에도 불구하고 성능을 일관되게 개선하는 데 성공하였습니다. 이 연구는 LLM이 간단한 GNN보다 항상 더 나은 성능을 보이지 않는다는 점을 발견했으며, LLM과 GNN의 통합을 통해 더욱 강력한 방법론을 제시합니다. 광범위한 실험 결과는 RoGRAD가 GNN의 강인성을 높이고 실제 적용에 적합한 새로운 솔루션을 제공하는 것을 보여줍니다.



### A Methodology for Transparent Logic-Based Classification Using a Multi-Task Convolutional Tsetlin Machin (https://arxiv.org/abs/2510.01906)
- **What's New**: 이번 논문에서 제안하는 Tsetlin Machine (TM)은 고유한 학습 패러다임으로, 유한 상태 자동자(finite-state automata)를 활용하고 규칙적 논리(propositional logic)로 패턴을 나타냅니다. 기존의 심층 신경망 기반 학습 알고리즘들과는 달리, TM은 해석 가능성이 뛰어납니다. 특히, Convolutional TM의 적용을 통해 대규모 RGB 이미지 분류에서의 가능성을 살펴보았습니다.

- **Technical Details**: TM은 입력 데이터에서 각 특징에 대해 학습 자동자(learning automaton)를 두어 여러 패턴을 학습합니다. 이 모델은 로컬 해석(local interpretations)과 글로벌 클래스 표현(global class representations)을 생성하여 예측을 설명할 수 있는 방법론을 개발했습니다. 이 방법론은 CNN에서와 같이 픽셀 구조를 활용하여 직관적으로 시각화가 가능하다는 특징이 있습니다.

- **Performance Highlights**: MNIST 데이터셋에서 98.5%의 정확도를, CelebA에서 86.56%의 F1-score를 달성하여 ResNet50의 88.07%와 비교할 때 경쟁력을 보여주었습니다. TM 모델은 심층 학습 모델과 견줄 만큼의 정확성을 유지하면서도 해석 가능성을 지속적으로 유지하고 있어, 다양한 데이터셋에서의 적용 가능성이 기대됩니다.



### Multimodal Foundation Models for Early Disease Detection (https://arxiv.org/abs/2510.01899)
Comments:
          6 pages

- **What's New**: 이 연구는 다양한 환자 데이터를 통합하는 다중 모달 기초 모델(multimodal foundation model)을 제안합니다. 기존의 전통적인 진단 모델들은 서로 다른 데이터 소스를 개별적으로 분석하여 조기 질병 진단을 위한 필수 교차 모달 상관관계를 찾아내는 데 제한이 있었습니다. 본 모델은 어텐션 기반(transformer) 프레임워크를 사용하여 이러한 데이터를 통합하고, 여러 각도에서 질병에 대한 예측 능력을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 전자 건강 기록(EHR), 의료 이미징, 유전자 데이터 및 착용 가능 센서 데이터를 조합하는 다중 모달(transformer-based) 아키텍처로 설계되었습니다. 각 데이터 모달리티는 전용 인코더를 통해 공유 잠재 공간(latent space)으로 변환되고, 그 후 멀티 헤드 어텐션(multi-head attention)과 잔여 정규화(residual normalization)를 통해 결합됩니다. 이 구조는 사전 훈련(pretraining) 가능성을 제공하여 여러 임상 작업에 쉽게 적응할 수 있도록 합니다.

- **Performance Highlights**: 실험 전략은 종양학, 심장학 및 신경학 분야의 벤치마크 데이터셋을 사용하여 조기 탐지 작업을 검증합니다. 또한, 투명성, 신뢰성 및 임상 해석 가능성을 개선하는 데이터 거버넌스(data governance) 및 모델 관리 도구를 포함하여 기술적 성과를 지원합니다. 이 방법론은 정밀 진단을 위한 단일 기초 모델 구축을 목표로 하며, 정확한 예측과 의사 결정을 돕는 데 기여할 것으로 기대됩니다.



### Multi-marginal temporal Schrödinger Bridge Matching for video generation from unpaired data (https://arxiv.org/abs/2510.01894)
Comments:
          Under review. Code available at this https URL . Additional experiment materials available at this https URL

- **What's New**: 이번 논문에서는 자연적 동적 프로세스의 복원 문제를 다루며, 이를 위해 다중한계(무형) 시간 슈뢰딩거 브리지 매칭(Multi-Marginal temporal Schrödinger Bridge Matching, MMtSBM) 기법을 제안합니다. 이 방법은 고차원(High-Dimensional) 데이터의 비정합(Non-paired) 샘플에서 비디오 생성을 가능하게 합니다. 논문은 기존의 Diffusion Schrödinger Bridge Matching(확산 슈뢰딩거 브리지 매칭)을 확장하여 Iterative Markovian Fitting 알고리즘을 다중 한계에 적용한 결과를 제시합니다.

- **Technical Details**: MMtSBM은 복잡한 자연 프로세스에서 발생하는 변동성을 고려하여 비연결 샘플의 실제 동적 진화 추정 문제를 해결하고자 합니다. 본 방법은 Markovian Fitting 기법을 효율적이고 체계적으로 확장하여 다중 반복 한계를 생성합니다. 이를 통해 KL divergence(쿨백-라이블러 발산)를 최소화하면서 고차원 데이터에서의 비즈니스 활용 가능성을 높였습니다.

- **Performance Highlights**: 실험 결과, MMtSBM은 이론적 성질을 잘 유지하며, 100차원에서의 전사체 궤적 추론과 같은 실제 데이터 세트에서 최신 성능을 달성했습니다. 또한, MMtSBM은 비정합 데이터 샘플에서 비디오 생성을 위한 일관된 알고리즘을 처음으로 제시하여 매우 높은 차원의 이미지 설정에서도 성공적으로 동적 정보를 복원했습니다.



### Randomized Gradient Subspaces for Efficient Large Language Model Training (https://arxiv.org/abs/2510.01878)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 훈련에 있어 메모리 요구 사항을 줄이는 혁신적인 알고리즘, GrassWalk 및 GrassJump를 제안합니다. 기존의 저차원 서브스페이스(low-dimensional subspace) 접근에 대한 새로운 통찰력을 제공하고, 이를 통해 기울기 공간의 구조를 평가하며, 훈련의 초기 단계에서 저차원 서브스페이스가 대부분의 기울기 에너지를 포착하는 것을 발견했습니다. 또한, 기울기 공간의 곡률이 거의 평평하다는 점을 강조하여, 이러한 기하학을 고려한 알고리즘의 필요성을 제기하고 있습니다.

- **Technical Details**: 저자들은 기울기 서브스페이스 업데이트 방법과 기울기 역학에 대한 분석을 통해 기울기 에너지가 시간이 지남에 따라 감소하고, 깊은 레이어에서 연속적으로 소실된 정보를 회복하는 전략을 포함합니다. GrassWalk 및 GrassJump 방법은 Grassmannian 매니폴드(Grassmannian manifold)에서 무작위 워크(randow walks)와 점프(jumps)를 적용하여 최적화 문제를 해결하고 성능을 향상시키는 기법을 사용합니다. 이러한 접근을 통해 저자들은 메모리 효율성은 물론 훈련 성능을 개선했습니다.

- **Performance Highlights**: 제안된 GrassWalk 및 GrassJump 알고리즘은 LLaMA-1B와 LLaMA-7B의 프리트레이닝(pretraining)에서 최첨단 성능을 달성하며, GaLore 같은 메모리 효율성을 유지하면서도 더 나은 성능과 빠른 수렴을 보여주었습니다. 이러한 결과는 기존의 강력한 기준선(baselines)을 일관되게 초과했습니다. 저자들은 기울기 역학과 최적화 지형을 인식하여 훈련 알고리즘을 설계할 때 정보를 최대한 활용해야 한다고 강조합니다.



### Universal Dynamic Regret and Constraint Violation Bounds for Constrained Online Convex Optimization (https://arxiv.org/abs/2510.01867)
- **What's New**: 이번 논문에서는 온라인 적대적 제약이 있는 OCO(Onlin Convex Optimization) 프레임워크의 일반화에 대해 논의합니다. 저자들은 제약 조건을 만족하지 않을 경우에도 동적 후회(dymamic regret) 및 누적 제약 위반(cumulative constraint violation) 경계를 제공하는 두 가지 알고리즘을 제시합니다. 이들은 적대자가 비용 및 제약 함수를 임의로 선택하는 가장 일반적인 경우에 대한 결과를 다루고 있습니다.

- **Technical Details**: 본 연구는 제한적 일반성 가정을 제거하고 동적 후회 경계를 보장하는 방향으로 나아갑니다. 두 알고리즘은 특히 설계된 서그리게이트 비용 함수(surrogate cost functions)를 사용하여 제약 최적화 문제를 비제약 최적화 문제로 환원합니다. 제안된 알고리즘은 고정된 벤치마크와는 다른, 시계열적으로 변동하는 벤치마크에 대한 성능을 측정하는 데 초점을 맞춥니다.

- **Performance Highlights**: 제시된 알고리즘은 전통적인 OGD와 비교해 동적 후회에 대해 더 좋은 결과를 보여주며, 특히 두 번째 알고리즘은 제약 조건이 빠르게 변화하는 환경에서 더 좋은 CCV 경계를 달성합니다. 이 논문은 Slater 조건을 포함하지 않고, 다수의 제약 함수를 고려할 수 있는 알고리즘을 제안하여 그 동안의 연구 제약을 극복했습니다.



### Compositional meta-learning through probabilistic task inferenc (https://arxiv.org/abs/2510.01858)
- **What's New**: 이 논문은 새로운 작업을 최소한의 경험으로 해결하기 위한 메타 러닝(meta-learning) 모델을 제안합니다. 이전 작업에서 배운 공통 컴포넌트를 유연하게 조합하여 새로운 구성으로 변형할 수 있는 조합 메타 러닝(compositional meta-learning)에 초점을 맞추고 있습니다. 이 모델은 주어진 작업을 재사용 가능한 계산의 구조적 조합으로 나타내어 확률적 추론(probabilistic inference) 문제로 변환하여 빠르게 새로운 솔루션을 찾을 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 RNN(순환 신경망)을 기반으로 하여 반복적으로 여러 작업을 수행하는 과정에서 자연스럽게 나타나는 '다이나믹 모티프(dynamical motifs)'를 활용합니다. 학습된 모듈(module)과 이를 조합하는 게이팅 네트워크(gating network)가 함께 작동하여 각 모듈의 동작과 조합 방식을 캡처합니다. 이 세분화된 아키텍처는 각 모듈이 수행하는 계산에 대한 독립성을 유지하면서, 조합의 통계를 학습할 수 있는 강한 유도 바이어스를 제공합니다.

- **Performance Highlights**: 이 모델은 레이블이 없는 새로운 작업에 대해서도 적은 양의 예시로부터 빠르게 추론할 수 있는 능력을 보여줍니다. 실제로, 규칙 학습(rule learning)과 운동 학습(motor learning) 작업에서 진정한 컴포넌트와 통계값을 성공적으로 회복하는 성과를 이루었습니다. 전체적으로 이 프레임워크는 신경망(neural networks)의 표현력과 확률적 추론의 데이터 효율성을 결합하여 빠른 조합 메타 러닝을 달성할 수 있도록 합니다.



### Explicit Discovery of Nonlinear Symmetries from Dynamic Data (https://arxiv.org/abs/2510.01855)
- **What's New**: 이 논문에서는 비선형 대칭(nonlinear symmetry)을 자동으로 발견할 수 있는 새로운 방법인 LieNLSD를 제안합니다. 기존의 대칭 발견 방법들이 주로 선형 대칭(linear symmetry)에 국한되어 있었던 것과 달리, LieNLSD는 비선형 항을 포함하는 무한소 생성자(infinitesimal generators)의 수와 그 표현을 명시적으로 결정할 수 있습니다. 이 방법은 데이터에서 대칭을 자동으로 추론하는데 중점을 두고 있습니다.

- **Technical Details**: LieNLSD는 데이터셋이 미분 방정식에 의해 지배되는 동적 시스템에서 생성된 것으로 가정합니다. 무한소 군 작용(infinitesimal group action)을 위해 비선형 항이 포함될 수 있는 함수 라이브러리(Θ)를 명시하고, 그에 따른 계수 행렬(coefficient matrix)을 구하기 위한 시스템을 개발했습니다. 또한, 단일값 분해(SVD)를 통해 계수 행렬을 해결하고, 이해하기 쉬운 결과를 도출하기 위해 LADMAP 기법을 적용하였습니다.

- **Performance Highlights**: LieNLSD는 기존 방법들에 비해 질적으로 우수한 성능을 보이며, 스파크(quark) 태깅(top quark tagging)과 몇 가지 동적 시스템에서 이전 방법들보다 20% 이상의 롤아웃 정확도 개선을 이뤘습니다. 이 방법은 또한 신경망 기반의 PDE 솔버(neural PDE solvers)의 데이터 증강(data augmentation) 가이드를 위해 적용되어, 더 높은 정확도를 달성하고 있습니다.



### Learning Representations Through Contrastive Neural Model Checking (https://arxiv.org/abs/2510.01853)
- **What's New**: 본 논문에서는 모델 체크 작업을 학습 신호로 활용하여 정렬된 표현을 학습할 수 있는 새로운 방법인 Contrastive Neural Model Checking (CNML)을 소개합니다. CNML은 논리적 명세와 시스템을 공동 임베딩하여 공유 잠재 공간에 매핑하는 self-supervised contrastive objective를 사용합니다. 이 방법은 형식적 언어의 표현 학습에서 가능성을 보여주며, 모델 체크를 대규모 데이터셋 생성의 효율적 방법으로 활용합니다.

- **Technical Details**: 논문에서 제안하는 CNML은 AIGER 형식으로 표현된 순차 회로와 Linear Temporal Logic (LTL)으로 표현된 사양을 대상으로 하는 bi-encoder 모델을 사용하여 정렬된 표현을 학습합니다. self-supervised learning 접근 방식을 통해 경량화된 데이터셋 생성 방법을 결합하여, 서로 다른 의미론을 효율적으로 학습합니다. 제안된 아키텍처는 명세의 구문에 구애받지 않으며, 다양한 논리나 회로 인코딩으로 쉽게 이전할 수 있습니다.

- **Performance Highlights**: CNML은 산업에서 영감을 받은 검색 작업에서 알고리즘 및 신경망 기준선을 각각 초과하여 높은 Recall@1% 및 Recall@10% 성능을 보였습니다. 또한, 학습된 표현은 다운스트림 작업에서 효과적으로 이전할 수 있으며, 단순한 공식에서 복잡한 수식까지 일반화할 수 있음을 입증합니다. 최종적으로, CNML은 명시적 감독 없이도 복잡한 의미론적 개념을 학습할 수 있는 능력을 가지고 있습니다.



### Pre-Hoc Predictions in AutoML: Leveraging LLMs to Enhance Model Selection and Benchmarking for Tabular datasets (https://arxiv.org/abs/2510.01842)
Comments:
          Oral Presentations ADAPT Annual Scientific Conference 2025

- **What's New**: 이 논문은 AutoML과 pre-hoc 모델 선택의 교차점을 탐구하여 전통적인 모델과 Large Language Model (LLM) 에이전트를 활용해 AutoML 라이브러리의 탐색 공간을 줄이는 방법을 제안합니다. 특히, 데이터셋 설명과 통계 정보를 활용하여 AutoML의 검색 공간을 감소시키는 새로운 방법론을 제시합니다. AutoGluon 포트폴리오 데이터셋을 통해 실험하여 대규모 계산 비용을 줄이면서도 주어진 데이터셋에 최적의 모델을 선택할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 논문의 핵심 방법론은 데이터셋의 구조적 통계 정보와 텍스트 설명 정보를 통합하여 효율적이고 정확한 pre-hoc 모델 선택을 가능하게 만드는 것입니다. LLMs를 활용한 AutoML 에이전트를 개발하여 탭형 AutoML 문제를 해결하는 파이프라인을 구축하였으며, 모델 선택의 정당성을 설명할 수 있는 기능도 추가되었습니다. 실험은 OpenML의 175개 데이터셋을 대상으로 진행되었으며, 전통적인 Pre-Hoc Predictor 방법과 LLM Pre-HP를 비교하며 성능을 평가했습니다.

- **Performance Highlights**: 여러 실험 결과, 전통적인 Pre-HP 모델이 자동화된 모델 선택에 있어서 Baseline 대비 개선된 성능을 보였습니다. 특히, RoBERTa 모델은 패밀리 정확도에서 0.61을 기록하며 기대 이상의 결과를 보여주었습니다. 그러나 LLM을 포함한 AutoML 에이전트는 여전히 전통적인 모델에 비해 성능이 낮았지만 Baseline 1은 초과하였고, Zero-Shot 방식의 Llama 모델이 전반적인 성능 지표에서 최상의 성능을 보였습니다.



### Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning (https://arxiv.org/abs/2510.01824)
- **What's New**: 이 논문에서 소개된 새로운 내용은 순서 불변의 강화 학습(framework for reinforcement learning) 프레임워크를 사용해 블랙박스 조합 최적화 문제를 해결하는 방법이다. 기존의 추정-분포 알고리즘(EDAs)은 변수 의존성 그래프(variable dependency graphs)를 학습하는 데 많은 비용이 소모되고 복잡한 상호작용을 효과적으로 포착하지 못했다. 반면, 이 연구에서는 고정된 변수 순서 없이 학습되는 다변수 자기회귀 생성 모델(multivariate autoregressive generative model)을 파라미터화하여, 훈련 중 랜덤 생성 순서를 샘플링하여 다양한 검색 공간(search-space diversity)을 촉진한다.

- **Technical Details**: 기술적으로, 연구에서는 일반화된 강화 정책 최적화(Generalized Reinforcement Policy Optimization, GRPO)를 이 설정에 맞게 조정하여, 스케일 불변 장점(scale-invariant advantages)에서 안정적인 정책 기울기 업데이트( policy-gradient updates)를 제공한다. 이로써 연구는 특정 문제 인스턴스( 문제 크기에 따라)와의 광범위한 벤치마크 알고리즘에서 유리한 결과를 도출한다. 더불어, 향상된 샘플 효율성을 통해 가장 관련 있는 변수 의존성을 포착하는 데 초점을 맞춘다.

- **Performance Highlights**: 다양한 크기의 문제 인스턴스에 걸쳐, 제안된 방법은 자주 최상의 성능을 달성하고 재앙적인 실패를 일관되게 피하는 결과를 보여준다. 강화 학습 기반 블랙박스 최적화에서 최적해를 찾기 위한 혁신적인 접근 방식을 제안함으로써, 많은 실제 응용 분야에서 확장성과 유연성을 개선할 수 있는 가능성을 실현한다. 이러한 접근은 고도로 복잡한 문제에 대해 기존 방법론의 한계를 극복하며, 향후 연구를 위한 새로운 방향성을 제시한다.



### Sparse Query Attention (SQA): A Computationally Efficient Attention Mechanism with Query Heads Reduction (https://arxiv.org/abs/2510.01817)
Comments:
          18 pages, 6 figures, small-scale experiments

- **What's New**: 이번 논문에서는 Sparse Query Attention (SQA)라는 새로운 attention 아키텍처를 소개합니다. SQA는 Query heads의 수를 줄여 attention 메커니즘의 계산 복잡성을 감소시킴으로써 플로팅 포인트 연산(Floating-Point Operations, FLOPs)의 수를 줄입니다. 이 방법은 메모리 대역폭의 병목 현상을 해결하는 기존의 접근 방식과는 다른 최적화 경로를 추구합니다.

- **Technical Details**: SQA의 수학적 공식화 및 변형된 아키텍처를 제시하며, 출처는 Sparse Query Attention의 기본 이론에 기반합니다. SQA는 Matrix multiplication인 QK^T의 복잡성을 O(N^2⋅d_model + N⋅d_model^2)에서 쿼리 헤드 수의 감소 비율에 비례하여 직접 줄입니다. 이로 인해 전체 FLOPs가 감소하고 긴 시퀀스의 경우 성능 개선을 이끌어냅니다.

- **Performance Highlights**: 32k-200k 토큰의 긴 시퀀스에서의 실험 결과 SQA는 계산 중심 시나리오에서 최대 3배까지 처리량을 개선할 수 있음을 보여주었습니다. 예비 실험에서 모델 품질에 미치는 영향은 최소한으로 유지되었으며, 이 아키텍처는 더 효율적이고 확장 가능한 모델 개발에 강력한 도구로서의 가능성을 제시합니다.



### Rethinking the shape convention of an MLP (https://arxiv.org/abs/2510.01796)
- **What's New**: 이 논문에서는 전통적인 narrow-wide-narrow MLP 디자인에 도전하여 wide-narrow-wide (Hourglass) MLP 블록을 제안합니다. 이 새로운 구조에서는 skip connections가 확장된 차원에서 작동하며, 잔여 계산은 좁은 병목을 통해 흐릅니다. 이러한 전환은 높은 차원 공간에서 점진적인 개선을 가능하게 하며, 매개변수에 맞춘 설계를 통해 계산 효율성을 유지합니다.

- **Technical Details**: Hourglass MLP는 입력 신호를 확장된 차원으로 올리기 위한 초기 프로젝션을 요구하며, 이 프로젝션은 훈련 중에 임의 초기화 상태를 고정할 수 있습니다. 또한, 이 구조는 기존의 narrow-wide-narrow 블록과 비교하여 건축적 검색을 통해 성능-매개변수 Pareto 전선에서 일관된 우수성을 나타냅니다. 특정 실험에서는 매개변수 예산이 증가함에 따라 더 깊은 네트워크와 넓은 skip connections 및 더 좁은 병목을 선호하는 패턴이 발견되었습니다.

- **Performance Highlights**: 토론된 결과들은 wide-narrow-wide 디자인이 기존의 MLP 설계와 비교하여 일관되게 우수한 Pareto 전선을 달성함을 입증합니다. 특히, 입력 프로젝션 레이어의 추가 매개변수를 고려하더라도, Hourglass 아키텍처는 전통적인 디자인에 비해 지속적으로 우수한 성능을 보여줍니다. 이러한 통찰력은 Transformer와 기타 잔여 네트워크를 포함한 다른 아키텍처로 확장될 가능성을 제시합니다.



### Sensitivity, Specificity, and Consistency: A Tripartite Evaluation of Privacy Filters for Synthetic Data Generation (https://arxiv.org/abs/2510.01793)
- **What's New**: 이번 연구는 의료 AI 연구에서 데이터 부족 문제를 해결하기 위한 개인 정보 보호를 유지하는 합성 데이터셋 생성의 잠재력을 탐구합니다. 기존의 post-hoc privacy filtering 기술의 효과가 검증되지 않았던 가운데, 저자들은 흉부 X-ray 합성에 적용된 필터링 파이프라인의 철저한 평가를 수행하였습니다. 이 연구 결과, 현재의 필터가 민감도(sensitivity)는 높지만 특이도(specificity)와 일관성(consistency)가 부족함을 보여주며, 이러한 필터링 기법들이 환자 정보 보호에 대한 잘못된 안전감을 줄 수 있음을 경고합니다.

- **Technical Details**: 연구에서는 Reynaud et al.의 접근법을 기반으로 한 개인 정보 필터링 프로토콜을 평가합니다. 필터는 개선된 민감도, 특이도 및 일관성을 충족해야 하며, 동일한 입력에 대해 일관된 결정을 내려야 합니다. 본 연구는 이미지 수준 데이터에 대한 필터의 성능을 평가하며, pixel-space에서의 효율성도 분석해 기존의 latent-space 접근법과 비교합니다.

- **Performance Highlights**: 평가 결과, RoentGen 모델을 사용한 필터에서 높은 분류 메트릭을 달성했습니다. 그러나 테스트 이미지가 결합된 훈련 풀을 기준으로 비교했을 때 필터가 88.5%의 이미지를 픽셀 공간에서 플래그를 지정했으나, 명시된 특이도와 일관성이 부족하여 이는 여전히 환자 개인 정보 보호에 대한 심각한 위험으로 작용할 수 있음을 나타냅니다.



### Neural non-canonical Hamiltonian dynamics for long-time simulations (https://arxiv.org/abs/2510.01788)
- **What's New**: 이번 연구는 데이터에서 비표준 해밀토니안 역학(non-canonical Hamiltonian dynamics)을 학습하는 데 중점을 둡니다. 장기 예측을 위해서는 학습된 모델과 수치적 방법 모두에서 구조를 보존해야 합니다. 기존 연구에서는 각각 잠재 기반 구조(potential-based architecture)와 퇴화 변분적 적분기(degenerate variational integrators)에 초점을 맞췄으나, 두 가지를 결합할 경우 새로운 문제가 발생합니다.

- **Technical Details**: 이 논문에서는 수치적 불안정성(numerical instability)의 문제를 식별하고 이를 해결하기 위한 두 가지 훈련 전략을 제안합니다. 첫 번째는 벡터 필드(vector field)를 직접 학습하는 방법이며, 두 번째는 스킴(scheme)을 통해 시간 이산적 동역학(time-discrete dynamics)을 학습하는 방법입니다. 이 방법들은 복잡한 물리적 역학을 학습하는 능력을 평가하기 위한 여러 수치적 테스트 케이스를 사용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법들은 자이로키네틱 플라스마 물리학(gyrokinetic plasma physics)에서 유도 중심(guiding center)과 같은 복잡한 물리적 동역학을 잘 학습하는 것으로 나타났습니다. 그러나 학습된 모델은 때때로 수치적으로 불안정하여 장기 시뮬레이션(long-time simulations)이 불가능한 경우도 있었습니다. 이러한 한계를 극복하기 위한 전략이 제시되었습니다.



### Octax: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX (https://arxiv.org/abs/2510.01764)
- **What's New**: 이번 논문에서는 Octax라는 새로운 Reinforcement Learning (RL) 환경을 소개합니다. Octax는 CHIP-8 에뮬레이션 기반에 의해 구현된 고성능 아케이드 게임 환경 스위트로, JAX에서 실행됩니다. 이 플랫폼은 Atari 벤치마크에 대한 GPU 기반 대안으로, 환경의 다양성과 확장성을 제공합니다.

- **Technical Details**: Octax는 1970년대의 CHIP-8 가상 머신 스펙을 활용하여 구축되었으며, 이는 64x32 모노크롬 디스플레이와 4KB 메모리를 가진 게임 환경을 지원합니다. JAX 기반의 구현은 GPU 가속을 통해 수천 개의 게임 인스턴스를 동시에 실행할 수 있게 해주며, 이는 CPU 기반 에뮬레이터보다도 월등한 속도를 자랑합니다. 딥 러닝과 RL 환경들의 실행 속도를 극적으로 향상시킬 수 있는 구조를 가지고 있습니다.

- **Performance Highlights**: Octax는 소비자 하드웨어에서 초당 350,000 환경 스텝을 달성하였으며, 이는 기존의 CPU 기반 솔루션보다 월등한 성능입니다. 또한, 새로운 게임을 쉽게 추가하거나 대형 언어 모델을 통해 새로운 환경을 자동으로 생성할 수 있는 모듈형 디자인을 갖추고 있어 대규모 RL 실험에 적합합니다. 이를 통해 연구자들은 실험 시간을 단축시키고 다양한 하이퍼파라미터 튜닝을 보다 쉽게 할 수 있게 됩니다.



### Unsupervised Dynamic Feature Selection for Robust Latent Spaces in Vision Tasks (https://arxiv.org/abs/2510.01758)
- **What's New**: 이번 논문은 동적 특징 선택(Dynamic Feature Selection, DFS)을 활용하여 레텐트 표현(latent representation)을 향상시키기 위한 새로운 접근법인 동적 데이터 선택(Dynamic Data Selection, DDS)을 제시합니다. 기존의 DFS 방법은 레이블(label) 의존성이 있어 다양한 도메인에 적용하기 어려운 반면, DDS는 비지도 학습(unsupervised learning)에서 최초로 적용된 기법입니다. DDS는 이미지 데이터에서 불필요하거나 중복된 정보를 제거하며, 선택한 특징의 위치를 유지하여 복잡한 네트워크 구조에도 쉽게 적용할 수 있습니다.

- **Technical Details**: 기술적으로, DDS는 주어진 인스턴스에 대해 최대 M개의 특징을 선택하기 위한 최소화 문제를 해결합니다. DDS 네트워크는 두 개의 주요 구성 요소로 나뉘어 있으며, 각 구성 요소는 모델을 훈련하는 데 도움을 줍니다. 이 방법은 비지도 손실 함수(unsupervised loss function)를 사용하여 입력 데이터의 중요성에 기반한 선택 과정을 수행합니다. 기본적으로, DDS는 입력 데이터를 마스킹하여 가장 관련성 높은 특징만을 오토인코더(autoencoder)에 전달하게 됩니다.

- **Performance Highlights**: 실험 결과는 DDS가 클러스터링(clustering) 및 표현 학습(representation learning) 작업에서 일반화 성능을 크게 향상시킨다는 것을 보여줍니다. DDS를 채택한 모델은 다양한 이미지 데이터셋에서 성능 개선을 이루며, 계산 비용의 증가는 최소화되었습니다. 이 연구는 비지도 문제 해결을 위한 DDS의 유용성과 용이성을 강조하며, 다양한 문제에 쉽게 적용 가능하다는 점도 부각됩니다.



### Learning Regularization Functionals for Inverse Problems: A Comparative Study (https://arxiv.org/abs/2510.01755)
- **What's New**: 최근 몇 년 동안, 이미징에서의 역문제를 해결하기 위한 다양한 학습된 정규화 프레임워크가 등장하였습니다. 이러한 방법들은 유연한 모델링과 수학적 통찰력을 제공하며, 아키텍처 설계 및 훈련 전략에서 차별화됩니다. 그러나 이러한 비모듈 구현으로 인해 직접 비교하기가 어려운 문제를 다루기 위해, 본 연구에서는 가용한 코드를 통합한 공통 프레임워크를 제시합니다.

- **Technical Details**: 역문제는 이미징 과학에서 일반적이며, MRI와 CT는 많은 현대 응용 프로그램에서 중요한 역할을 합니다. 이 역문제를 수학적으로 모델링하는 것은 일반적으로 선형 역문제로 표현되며, 데이터 일관성을 보장하기 위해 D라는 데이터 충실도 함수와 원하는 속성을 증진시키는 정규화기 R이 결합됩니다. 이를 통해 다양한 정규화기를 비교하고, 각 방법의 장단점을 강조할 수 있습니다.

- **Performance Highlights**: 최근 몇 년 간 딥러닝 기반 접근법이 역문제 해결을 위한 최첨단 기술로 자리잡았습니다. 이러한 기술은 인상적인 결과를 달성하고 있지만, 신뢰성에 대한 우려가 존재합니다. 따라서 데이터 기반 접근법과 기존의 수작업으로 처리한 정규화기 간의 혼합 접근 방식에 초점을 맞추어 연구를 진행할 예정입니다.



### Private and Fair Machine Learning: Revisiting the Disparate Impact of Differentially Private SGD (https://arxiv.org/abs/2510.01744)
- **What's New**: 이번 논문은 Differential Privacy(DP)를 사용하여 개인 정보 보호를 위해 훈련된 신경망 모델의 공정성에 미치는 영향을 분석합니다. 특히, DPSGD(Differentially Private Stochastic Gradient Descent)를 통해 얻은 결과가 기존 비공식 모델에서 얻은 하이퍼파라미터를 재사용함으로써 공정성을 해칠 수 있음을 강조합니다. 이는 비공식 모델과 유사한 성능과 공정성을 갖기 위해 하이퍼파라미터 최적화가 중요하다는 점을 시사합니다.

- **Technical Details**: DP는 데이터 분석 중 개인 정보를 보호하기 위한 수학적 정의로, DPSGD는 신경망을 위한 가장 일반적인 DP 구현 방법입니다. DPSGD는 최대 영향력을 제한하기 위해 예제별 기울기를 클리핑하고 가우시안 노이즈를 추가합니다. 그러나 이러한 방식은 모델의 학습 역학에 영향을 미쳐 공정성 및 성능과 같은 속성에도 영향을 끼친다는 점이 특징입니다.

- **Performance Highlights**: 논문에서는 DPSGD가 공정성에 미치는 영향을 측정하기 위해 다양한 성능 지표와 공정성 지표를 사용한 실험 결과를 제시합니다. DPSGD의 하이퍼파라미터를 최적화하는 것이 반드시 공정성의 불균형을 완화하지는 않지만, 비공식 모델에서 하이퍼파라미터를 재사용하는 것보다 공정성 및 유용성 측면에서 향상된 결과를 보일 수 있음을 보여줍니다. 그러나 이러한 과정에서 추가적인 개인 정보 누출이 발생할 수 있음을 경고하며, 개인 정보와 유용성, 공정성 간의 균형을 신중히 고려해야 한다고 강조합니다.



### Workplace Location Choice Model based on Deep Neural Network (https://arxiv.org/abs/2510.01723)
- **What's New**: 본 논문은 기존의 이산 선택 모델(DCM)보다 더 향상된 작업장 위치 결정을 모델링하기 위한 심층 신경망(DNN) 방식을 제시합니다. DNN은 복잡한 결정 패턴을 이해하는 데 더 효과적이며, 전통적인 모델보다 더 나은 결과를 제공합니다. 이 연구는 DNN이 작업장 위치 선택 분석에서 DCM의 강력한 대안으로서 가지고 있는 잠재력을 강조합니다.

- **Technical Details**: 연구에서는 두 가지 DNN 모델을 제안하며, 첫 번째 모델은 DCM의 입력을 재현하고 두 번째 모델은 가능한 모든 데이터를 입력으로 활용합니다. DCM 기반 작업장 위치 선택 모델은 Naqavi et al. (2023)에 의해 제안된 간단한 2단계 중첩 로짓(NL) 모델을 기반으로 하며, 이는 개인의 시간을 기반으로 한 접근성 측정을 포함합니다. 모델은 교통 수요 예측을 위해 시간적 및 공간적 제약, 개인의 사회경제적 특성, 활동 참여도와 여행 모드 및 시간 등을 고려합니다.

- **Performance Highlights**: DNN과 DCM은 모두 직업 기회가 직장 위치 선택에 미치는 영향을 효과적으로 복제하지만, DNN은 특정 상황에서 DCM보다 우수한 성능을 보여줍니다. 특히, DCM은 직장 거리와 관련된 개인 속성을 평가할 때 데이터와 잘 맞지만, DNN은 장거리 이동에 대해서는 DCM과 유사한 성과를 보입니다. 이러한 결과는 작업장 위치 선택 분석에 있어 적절한 모델을 선택하는 것의 중요성을 강조합니다.



### Finite-Time Bounds for Distributionally Robust TD Learning with Linear Function Approximation (https://arxiv.org/abs/2510.01721)
Comments:
          Preprint. 32 Pages

- **What's New**: 본 논문에서는 배포적으로 강건한 강화 학습(Distributionally Robust Reinforcement Learning, DRRL)에 초점을 맞추고 있습니다. 특히, 불확실성이 존재하는 환경에서 최악의 경우 장기 할인 보상을 극대화하는 방법을 제시합니다. 기존의 강건한 Temporal-Difference (TD) 학습의 수렴 보장은 제한된 환경에서만 작동했지만, 본 연구는 선형 함수 근사를 이용하여 강건성을 측정하는 새로운 알고리즘을 소개합니다.

- **Technical Details**: 강건 TD 학습 알고리즘은 총 변동 거리(total-variation distance) 및 Wasserstein-l 거리 불확실성 집합에 대해 정의됩니다. 모델 프리(model-free) 접근 방식을 채택하면서도, 생성적 접근이 필요하지 않도록 설계되었습니다. 알고리즘은 두 개의 시간 척도를 갖는 확률적 근사 업데이트와 외부 루프의 타겟 네트워크 업데이트를 결합하여 sample complexity을 O~(1/ε²)로 설정함으로써 ε정확한 가치 추정이 가능하다고 합니다.

- **Performance Highlights**: 본 연구는 강건 RL 알고리즘의 경험적 성공과 비강건 알고리즘의 비비대칭 보장 간의 간극을 해소하는 결과를 제공합니다. 특히, 함수 근사를 사용하는 경우의 유한 시간 보장을 확립하였으며, 이로 인해 실무에서 많이 사용되는 “강건 TD” 휴리스틱 기반의 차이를 극복할 수 있게 되었습니다. 강건 Q-러닝에 대한 수렴 및 샘플 복잡도 한계도 제시되어, 기존 문헌에서는 확인되지 않은 해결책을 모색합니다.



### Accelerating Attention with Basis Decomposition (https://arxiv.org/abs/2510.01718)
- **What's New**: 본 연구에서는 BD Attention (BDA)라는 새로운 알고리즘을 소개합니다. BDA는 주의(attention) 연산의 손실(lossless) 없는 첫 번째 알고리즘적 재구성을 제공합니다. 이 알고리즘은 Basis Decomposition (BD)의 간단한 행렬 특성을 활용하여 다중 헤드 프로젝션을 축소된 형태로 재구성하며, 정확한 출력을 보존합니다.

- **Technical Details**: BDA는 다중 헤드 주의(MHA) 프로젝션을 컴팩트한 형태로 재구성하여 매개변수와 산술 연산을 줄이는 방식으로 작동합니다. 기존의 I/O 최적화 방안과 달리, BDA는 하드웨어에 독립적인 수학적 가속을 제공합니다. BDA는 DeepSeek-V2-Lite (16B, FP16) 설정에서 평균 32% 더 빠른 키/값 프로젝션과 25% 더 작은 가중치를 달성합니다.

- **Performance Highlights**: BDA는 4초의 오프라인 준비 시간만으로, 재훈련 없이도 우수한 성능을 보여줍니다. 모델 성능에 대한 영향은 최소화되어, PPL(perplexity)의 변화는 0.02%에 불과합니다. 추가적인 실험 결과, BDA는 MHA와 동등한 BLEU 점수를 달성하였고, 메모리 사용량을 16.5% 줄이며 처리량을 17.2% 향상시킬 수 있음을 보여 주었습니다.



### Latency-aware Multimodal Federated Learning over UAV Networks (https://arxiv.org/abs/2510.01717)
Comments:
          Accepted at IEEE Transactions on Network Science and Engineering

- **What's New**: 이번 논문은 무인 항공기(UAV)를 활용한 연합 다중 모달 학습(federated multimodal learning, FML)에 대해 조사하며, 시스템 지연 시간을 최소화하고 수렴 분석(convergence analysis)을 제공합니다. UAV는 네트워크에 분산되어 데이터를 수집하고 모델 학습에 참여하며, 기지국(base station, BS)과 협력하여 글로벌 모델을 구축합니다. 이 연구는 다중 모드 감지를 통해 단일 모드 시스템의 한계를 극복하고, 모델의 정확도와 일반화를 향상시키는 데 중점을 두었습니다.

- **Technical Details**: 제안된 FML 시스템은 UAV 네트워크에서 우선적으로 지연 시간을 최적화하는 것을 목표로 합니다. UAV의 감지 스케줄링(UAV sensing scheduling), 전력 제어(power control), 궤적 계획(trajectory planning), 자원 할당(resource allocation), BS 자원 관리(BS resource management)와 같은 다양한 요소를 통합하여 문제를 해결합니다. 복잡성을 해결하기 위해, 블록 좌표 하강법(block coordinate descent) 및 연속 볼록 근사(successive convex approximation) 기법을 결합한 효율적인 반복 최적화 알고리즘을 제안했습니다.

- **Performance Highlights**: 수치 실험을 통해, 제안된 FML 프레임워크가 기존 방법들에 비해 시스템 지연 시간(system latency)과 모델 학습 성능에서 우수함을 입증했습니다. 특히, 제안된 방법은 벤치마크 방법들에 비해 시스템 지연 시간을 42.49%까지 줄일 수 있음을 보여주었습니다. 또한, 독립적이며 동일하게 분포된(IID) 데이터와 비 IID 데이터 설정에서 모델 로스(model loss)와 정확도 수렴에서 뛰어난 성능을 보였습니다.



### ActiNet: Activity intensity classification of wrist-worn accelerometers using self-supervised deep learning (https://arxiv.org/abs/2510.01712)
- **What's New**: 이 논문에서는 패시브하게 수집한 손목 가속도계 데이터에서 신뢰할 수 있고 정확한 인간 활동 인식(human activity recognition, HAR) 모델의 중요성을 강조합니다. 자가 지도 학습(self-supervised learning) 방식이 HAR 개선에 도움을 줄 것으로 기대되고 있지만, 이러한 모델과 숨겨진 마르코프 모델(hidden Markov models, HMMs)의 결합이 실제 분류 성능 개선에 얼마나 기여하는지는 아직 알려지지 않았습니다.

- **Technical Details**: 본 연구에서는 151명의 CAPTURE-24 참가자의 데이터를 사용하여 ActiNet 모델을 훈련했습니다. ActiNet은 18층으로 구성된 수정된 ResNet-V2 모델이며, 이후 HMM 매핑을 통해 활동 강도의 라벨을 분류합니다. 5배 층화 그룹 교차 검증(5-fold stratified group cross-validation)을 통해 이 모델의 성능을 기존의 랜덤 포레스트(random forest, RF) + HMM과 비교했습니다.

- **Performance Highlights**: ActiNet 모델은 평균 매크로 F1 점수 0.82와 평균 Cohen's kappa 점수 0.86을 기록하며 활동 강도 라벨을 구분할 수 있었습니다. 이는 동일 데이터셋에서 훈련 및 검증된 RF + HMM의 평균 점수 0.77 및 0.81을 초과하는 성능입니다. 이러한 발견은 연령 및 성별 하위 그룹에서도 일관되게 나타났으며, 향후 역학 연구에서 손목 가속도계 데이터에서 활동 강도 라벨 추출을 위한 ActiNet의 사용을 권장합니다.



### Representational Alignment Across Model Layers and Brain Regions with Hierarchical Optimal Transpor (https://arxiv.org/abs/2510.01706)
- **What's New**: 이 논문에서 제안하는 Hierarchical Optimal Transport (HOT) 방법론은 전통적인 layer-wise matching 방식의 한계를 극복하는 새로운 프레임워크입니다. 기존의 방법은 각 레이어를 개별적으로 매칭하여 비대칭적인 결과를 도출하는데 반해, HOT는 레이어 간의 부드럽고 전 세계적으로 일관된 연결을 추론하여 전체 네트워크 비교를 위한 하나의 정렬 점수를 제공합니다. 이 방식을 통해 다양한 심층 신경망 아키텍처 간의 비교가 가능해집니다.

- **Technical Details**: HOT는 노드 간의 유사성을 고려하는 최적 수송 이론을 기반으로 하여, 각 레이어는 다양한 타겟 레이어에 질량을 분배할 수 있습니다. 이를 통해 서로 다른 깊이의 신경망 간의 연결을 부드럽게 처리하며, 각 소스 레이어는 자신의 표현 정보를 잃지 않고 균형 잡힌 정렬을 이루어냅니다. 이 과정은 두 가지 계층적 수준에서 진행되며, 각 레이어 내의 뉴런을 정렬하고 전역적으로 레이어 간의 연계를 결정합니다.

- **Performance Highlights**: HOT는 비전 모델, 대형 언어 모델, 인간 시각 피질 데이터 등에 대해 평가되었으며, 기존의 표준 pairwise 방법론과 비교하여 정렬 품질에서 동등하거나 그 이상의 성능을 기록하였습니다. 특히, HOT는 깊이의 불일치를 자연스럽게 처리하고, 신경망의 학습 과정에서 발생하는 계층적 구조를 복원함으로써 더 정교하고 해석 가능한 비교를 가능하게 합니다.



### PASTA: A Unified Framework for Offline Assortment Learning (https://arxiv.org/abs/2510.01693)
- **What's New**: 이 논문은 오프라인 및 데이터 기반의 할인 최적화 문제에 대한 연구를 다룹니다. 저자들은 고객 선택 데이터에 기반하여 최적의 상품 조합을 결정하는 새로운 PASTA(비관적 조합 최적화 프레임워크)를 소개합니다. PASTA는 기본적인 선택 모델을 고려할 때 최적의 기대 수익을 달성하는 데 필요한 데이터 분포의 조건을 간소화합니다.

- **Technical Details**: PASTA 프레임워크는 불확실성 집합을 구성하고, 이를 바탕으로 각 조합의 최악의 경우의 수익을 평가합니다. 이 방법은 최대-최소 문제를 해결하여 최대의 최악의 수익을 내는 조합을 선택합니다. 저자들은 이론적 보장을 제공하고, 비관적 접근 방식이 오프라인 학습 문제에서 성공적이라는 점을 강조합니다.

- **Performance Highlights**: 실험 결과, PASTA는 기존의 베이스라인 접근 방식보다 성능이 우수하다는 것을 보여주었습니다. 또한, 다수의 기본 선택 모델에 대해서도 새로운 유한 표본 후회 경계를 설정하였고, MNL 모델에서의 최소 최대 후회 하한을 도출하여 이론적 최적성을 입증했습니다.



### Beyond Simple Fusion: Adaptive Gated Fusion for Robust Multimodal Sentiment Analysis (https://arxiv.org/abs/2510.01677)
- **What's New**: 이 논문은 다중 모달 감정 분석(Multimodal Sentiment Analysis, MSA)를 위해 정보의 엔트로피와 모달 중요성을 바탕으로 피쳐 가중치를 조정하는 새로운 네트워크인 Adaptive Gated Fusion Network (AGFN)을 제안합니다. 기존의 단순한 융합 기술은 모달리티의 부정확성을 간과하여 감정 예측의 성능을 저하시켰던 문제점을 해결하고자 합니다. AGFN은 노이즈가 있는 모달리티의 영향을 줄이며, 신뢰할 수 있는 신호를 우선시하여 감정 예측의 정밀도를 높입니다.

- **Technical Details**: AGFN은 정보 엔트로피 게이트와 모달 중요성 게이트라는 두 가지 방법으로 각 모달리티의 신뢰도를 모델링합니다. 정보 엔트로피 게이트는 각 모달리티의 신뢰성을 평가하여, 더 낮은 엔트로피를 가진 모달리티에 더 큰 가중치를 부여합니다. 모달 중요성 게이트는 각 샘플에 대해 모달리티의 중요성을 평가하여, 모든 모달리티를 동적으로 조정해 융합합니다.

- **Performance Highlights**: AGFN은 CMU-MOSI 및 CMU-MOSEI 데이터셋에서 이미 존재하는 강력한 기준 모델들을 초월하는 성능을 보여줍니다. CMU-MOSI에서는 Acc-2(82.75%), F1(82.68%), Acc-7(48.69%) 및 MAE(71.02%)의 결과를 기록하였으며, CMU-MOSEI에서도 각종 성능 지표에서 우위를 점하며 뛰어난 구간을 보였습니다. 따라서 AGFN은 다양한 벤치마크에서 다중 모달 감정 분석의 효과성을 지속적으로 입증하고 있습니다.



### Shift-Invariant Attribute Scoring for Kolmogorov-Arnold Networks via Shapley Valu (https://arxiv.org/abs/2510.01663)
Comments:
          15 pages, 6 figures, 9 tables

- **What's New**: 본 논문에서는 Shapley value에 기반한 새로운 프루닝 프레임워크인 ShapKAN을 제안합니다. ShapKAN은 KAN(Kolmogorov-Arnold Networks)의 노드 중요도를 평가하고 이를 통해 네트워크 프루닝을 수행하는 방법으로, 기존의 크기 기반 방식의 한계를 극복합니다. 이 프레임워크는 입력 파라미터화에 관계없이 노드의 기여도를 정량화하여 일관된 중요성 순위를 보장합니다.

- **Technical Details**: KANs(Kolmogorov-Arnold Networks)는 학습 가능한 스플라인 기반 활성화 함수를 활용하여 기능 근사 성능을 향상시킵니다. 그러나 KAN의 아키텍처는 네트워크 프루닝에 있어 독특한 도전 과제를 제공합니다. ShapKAN은 Shapley 값을 활용하여 노드 기여도를 평가하고, 이는 다층 네트워크 아키텍처에서도 적용 가능합니다.

- **Performance Highlights**: 방대한 실험을 통해 ShapKAN은 실제 및 합성 데이터셋에서 노드의 중요성을 보존하며 효과적인 네트워크 압축을 제공합니다. 실제로 ShapKAN은 이를 통해 KAN의 해석 가능성을 향상시키며 자원이 제한된 환경에서도 유용한 적용을 가능하게 합니다.



### Learning Time-Series Representations by Hierarchical Uniformity-Tolerance Latent Balancing (https://arxiv.org/abs/2510.01658)
Comments:
          Accepted in Transactions on Machine Learning Research

- **What's New**: TimeHUT는 계층적 uniformity-tolerance (균일성-허용도) 균형을 통해 시계열 표현을 학습하는 새로운 방법입니다. 이 방법은 두 가지 상이한 손실 함수를 사용하여 임베딩 공간에서 균일성과 허용도 간의 효과적인 균형을 이루고자 합니다. TimeHUT는 시계열로부터 인스턴스 기반 및 시간 정보를 학습하는 이점이 있습니다.

- **Technical Details**: 이 방법은 계층적 설정을 사용하여 강력한 시계열 표현을 학습합니다. 기본적인 contrastive loss (대조 손실) 내에 온도 스케줄러를 통합하여 임베딩의 균일성과 허용성 특성을 조정합니다. 또한, 계층적 angular margin loss (각도 여유 손실)을 통해 인스턴스 기반 및 시간 대비 손실을 시행하여 시계열의 양성과 음성 쌍 사이의 기하학적 마진을 생성합니다.

- **Performance Highlights**: TimeHUT는 다양한 작업에서 효과적으로 평가되었으며, 128 UCR 및 30 UAE 데이터셋에 대한 다변량 및 단변량 분류에서 기존 방법들보다 상당한 성과를 보였습니다. 또한, Yahoo 및 KPI 데이터셋에서의 이상 탐지 작업에서도 경쟁력 있는 결과를 얻었습니다. 마지막으로, 여러 구성 요소와 하이퍼파라미터를 평가하기 위한 세부적인 민감도 및 ablation study (제거 연구)가 수행되었습니다.



### Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning (https://arxiv.org/abs/2510.01656)
- **What's New**: 이 논문에서는 비대칭 근접 정책 최적화(Asymmetric Proximal Policy Optimization, AsyPPO) 프레임워크를 소개하고 있습니다. AsyPPO는 LLM에서의 효율적인 강화 학습(RL4LLM) 구현을 위해 경량 미니 크리틱(mini-critics)을 활용하여 비판자의 역할을 복구합니다. 이 프레임워크는 높은 계산 효율성을 유지하면서도 강력한 가치 추정을 가능하게 합니다.

- **Technical Details**: AsyPPO는 각기 분리된 프롬프트 조각을 기반으로 훈련된 미니 크리틱 세트를 사용합니다. 이 접근법은 크리틱 간의 다양성을 장려하면서 보정(calibration)을 유지하며, 가치 예측 편향(value-estimation bias)을 줄입니다. 또한, 크리틱 간의 불확실성을 활용하여 정책 업데이트를 정교화합니다: (i) 크리틱이 동의하는 상태에서의 이익(masks) 값을 마스킹하고, (ii) 엔트로피 정규화에서 높은 분산 상태를 필터링하여 불필요한 탐색을 억제합니다.

- **Performance Highlights**: 5,000개의 샘플로 오픈 소스 데이터에서 훈련한 결과, AsyPPO는 GRPO와 같은 강력한 기준선에 비해 여러 벤치마크에서 학습 안정성과 성능을 지속적으로 개선하였습니다. 특히, Qwen3-4b-Base에서는 6% 이상의 성능 향상을, Qwen3-8b-Base와 Qwen3-14b-Base에서는 각 3%의 향상을 기록했습니다. 이러한 결과는 확장 가능하고 효율적인 알고리즘을 위한 아키텍처 혁신의 중요성을 강조합니다.



### The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM (https://arxiv.org/abs/2510.01650)
Comments:
          Preprint

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 pruning(프루닝) 기술을 개선하여 최대 90%의 극한 스파시티(sparsity)에서 높은 성능을 유지하는 새로운 방법인 $	exttt{Elsa}$를 제안합니다. 기존의 방법들이 보통 50-60% 이상의 스파시티에서 모델 성능이 저하되는 문제를 해결하기 위해, $	exttt{Elsa}$는 전통적인 최적화 기법인 ADMM(Alternating Direction Method of Multipliers)을 통해 이러한 문제를 효과적으로 다룹니다. 또한, $	exttt{Elsa}_{	ext{-L}}$, 즉 양자화된 변종을 제시하여 27B 매개변수 모델에서도 적용 가능하다는 이론적 보장을 제시하고 있습니다.

- **Technical Details**: $	exttt{Elsa}$ 방법은 신경망 파라미터의 스파시티 제약 조건을 명확하게 설정하고, 강건한 풀러를 개발하여 여러 모델에 적용할 수 있습니다. 이 방법은 125M에서 13B 매개변수 모델까지 폭넓게 적용 가능하며, 기존 기법에 비해 최소 5배에서 최대 30배 낮은 perplexity를 달성합니다. 특히, pruning 후 90% 스파시티에서의 zero-shot 예측 정확도가 거의 6% 개선되었습니다. 메모리 효율성을 고려한 양자화 최적화 상태를 통합한 유연한 구현을 제공하여, 큰 모델에서도 메모리 사용을 66% 줄입니다.

- **Performance Highlights**: 본 연구는 LLM 스파시티(LLM sparsity)의 한계를 극복하고, 기존 방법들보다 실질적인 성능 개선을 보여줍니다. 예를 들어, LLaMA-2-7B 모델에서 90% 스파시티를 달성했을 때 기존 최상급 방법보다 7.8배 낮은 perplexity를 기록했습니다. 이는 모델의 성능을 유지하면서도 메모리 및 계산 효율성을 높이는 중요한 발전을 이룬 것입니다. 이러한 결과는 LLM 프루닝 분야에서의 추가적인 연구 방향을 제시하며, 새로운 전략에 대한 탐색의 필요성을 강조합니다.



### Source-Free Cross-Domain Continual Learning (https://arxiv.org/abs/2510.01649)
- **What's New**: 이 논문은 기존의 레이블된 소스 도메인 샘플 없이 지속적 도메인 적응을 진행하는 소스 없는 크로스 도메인 지속 학습(Source-Free Cross-Domain Continual Learning, SFCDCL) 문제를 처음으로 제안합니다. 기존의 크로스 도메인 지속 학습에서는 소스 도메인을 사용하는 것이 필수적이나, 본 연구는 프라이버시 요구 사항에 부합하기 위해 레이블이 없는 소스 도메인 샘플만을 사용하여 학습하는 방법을 설명합니다.

- **Technical Details**: 논문에서는 리허설이 필요 없는 주파수 인식 동적 프롬프트 협동(REFEREE) 방식을 제안합니다. REFEREE는 소스 사전 학습 모델과 대규모 비전-언어 모델 간의 협업 훈련 전략을 통해 모델의 의존도를 줄이고, 주파수 인식 프롬프트 기법을 사용하여 고주파 성분을 억제하고 저주파 성분을 촉진합니다. 또한, 불확실성 인식 가중치 기법을 통해 노이즈가 많은 가짜 레이블 문제를 해결하고, 커널 선형 판별 분석(KLDA)을 통해 이중 재앙적 망각(Double Catastrophic Forgetting) 문제를 극복합니다.

- **Performance Highlights**: REFEREE 방식은 기존의 소스 도메인 샘플을 사용하는 기법들과 비교해도 의미 있는 성능 향상을 보여주었습니다. 실험 결과는 REFEREE가 여러 벤치마크 문제에서 이전의 기법들보다 눈에 띄게 우수한 성과를 올렸음을 확인했으며, 이는 레이블이 없는 소스 도메인 샘플만을 사용하였음에도 불구하고 이루어진 결과입니다.



### Support Basis: Fast Attention Beyond Bounded Entries (https://arxiv.org/abs/2510.01643)
- **What's New**: 이 논문에서는 소프트맥스 어텐션의 이차 복잡성(quadratic complexity)을 개선하기 위한 새로운 프레임워크인 '지원 기반 분해(support-basis decomposition)'를 소개합니다. 기존의 이차 복잡성을 해결했던 Alman과 Song(NeurIPS 2023)의 방법이 제한적인 가정 하에서만 작동했던 것과 달리, 본 논문에서는 구속된 항목 없이 효율적인 어텐션 근사화를 가능하게 합니다. 실험적 분석을 통해 쿼리와 키 행렬의 항목들이 아랫가우시안(sub-Gaussian) 행동을 보인다는 점을 주장하며, 이 특성을 활용하여 큰 항목과 작은 항목을 분리합니다.

- **Technical Details**: 제안된 방법은 희소(sparse) 구성 요소에 대해서는 정확한 계산을 수행하고, 조밀한(dense) 구성 요소에 대해서는 다항식 근사(polynomial approximation)를 사용합니다. 이 논문에서는 어느정도 현실적인 가정 없이도 이러한 근사가 가능하다는 것을 이론적으로 보장하는 엄격한 기초 이론을 제시합니다. 새로운 멀티-스레숄드(multi-threshold) 설정으로 방법을 확장하였으며, 이는 모든 분포적 가정을 제거합니다.

- **Performance Highlights**: 이 연구는 또한 다항식 주의(polynomial attention)의 실질적 성공에 대한 최초의 이론적 근거를 제공합니다. 실험적으로, 다항식 어텐션의 조합으로 소프트맥스 어텐션을 밀접하게 근사할 수 있음을 증명하였습니다. 이를 통해 대규모 언어 모델의 효율성을 높이는데 기여할 수 있는 방법을 제시합니다.



### Detecting Post-generation Edits to Watermarked LLM Outputs via Combinatorial Watermarking (https://arxiv.org/abs/2510.01637)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 워터마크 결과물에 대한 후처리 에디트 감지의 새로운 작업을 제안합니다. 이는 생성된 콘텐츠에 대한 책임성과 투명성을 요구하는 적용 사례에서 중요합니다. 기존의 워터마킹 기법은 일반적으로 워터마크를 감지하는 데 중점을 두었으나, 후처리를 통해 수정된 내용을 식별하고 지역화하는 것을 목표로 합니다.

- **Technical Details**: 제안된 프레임워크는 조합적인 패턴 기반 워터마킹 기법을 사용하며, 이 프레임워크는 워터마크 생성 메커니즘, 워터마크 감지용 글로벌 통계, 그리고 지역 수정을 찾기 위한 전문 통계로 구성됩니다. 각 LLM의 어휘를 불공통 부분 집합으로 나누고, 이 조합 패턴에 따라 생성 과정에서 워터마크를 삽입합니다. 이를 통해 수정된 영역을 정확히 식별할 수 있는 통계 데이터도 마련합니다.

- **Performance Highlights**: 제안된 방법의 성능은 여러 개의 후처리 에디팅 시나리오에서 오픈 소스 LLM에 대해 평가하였으며, 수정된 내용을 국지적으로 정확히 식별할 수 있는 강력한 실험적 성과를 보였습니다. 이를 통해 워터마킹 기술이 여전히 강력한 신뢰성을 유지하며, 생성된 콘텐츠의 출처를 검증할 수 있음을 입증합니다.



### CAT: Curvature-Adaptive Transformers for Geometry-Aware Learning (https://arxiv.org/abs/2510.01634)
- **What's New**: Curvature-Adaptive Transformer (CAT)은 데이터의 비유클리드 기하학적 구조를 효과적으로 처리하기 위해 설계된 새로운 아키텍처입니다. CAT은 경량의 미분 가능한 게이팅 메커니즘을 통해 서로 다른 기하학적 주의(branch)에서 각 토큰을 동적으로 라우팅합니다. 기존 기하학적 접근 방식을 넘어서서, CAT은 복잡한 관계적 추론을 위해 학습된 기하학적 적응을 활용하여 성과를 극대화합니다.

- **Technical Details**: CAT 아키텍처는 새롭게 개발된 각각의 기하학적 주의 가지(Euclidean, hyperbolic 및 spherical)의 특수화된 작업을 수행하며, 각 격자에서의 작업은 해당 매니폴드에 맞게 최적화되어 있습니다. 토큰 단위의 적응성을 제공하며, CAT은 미리 정해진 기하학적 구조를 기반으로 하지 않고 각 입력의 지역적 관계 구조에 따라 토큰을 라우팅합니다. CAT은 혼합-기하학(mixture-of-geometry) 아키텍처로서 언어, 비전 및 다중 모달 도메인에서의 적용 가능성을 높입니다.

- **Performance Highlights**: CAT은 지식 그래프 완성 벤치마크(FB15k-237, WN18RR)에서 고정 기하학적 기준에 비해 약 10%의 MRR 및 Hits@10 개선을 달성하였습니다. 이러한 결과는 최소한의 오버헤드(5%의 파라미터 증가)로 이루어졌으며, CAT은 복잡한 관계적 추론에서 단일 고정 기하학을 초월하는 성능을 입증했습니다. CAT은 해석 가능성과 확장성을 갖춘 혼합-기하학 아키텍처의 기초를 세우고 있습니다.



### Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls (https://arxiv.org/abs/2510.01631)
Comments:
          Published as a Main Conference paper at EMNLP 2025

- **What's New**: 이번 연구는 대형 언어 모델(LLM)의 사전 훈련 과정에서 합성 데이터(synthetic data)의 역할과 효과를 체계적으로 조사한 대규모 연구 결과를 제시합니다. 1000개 이상의 LLM 변형에 대해 200B tokens 규모의 데이터셋을 사용하여 훈련을 진행하였고, 각 데이터 유형과 특성에 따른 사전 훈련 성능을 비교하였습니다. 합성 데이터의 특정 비율이 사전 훈련 속도를 크게 향상시킬 수 있다는 점이 새롭게 밝혀졌습니다.

- **Technical Details**: 연구 결과에 따르면, 1/3의 재구성된 합성 데이터와 2/3의 자연 웹 텍스트가 혼합된 데이터에서 훈련을 진행한 경우, 같은 검증 손실에 도달하는 데 5-10배의 속도 향상이 나타났습니다. 그러나 사전 훈련을 위해 합성 데이터 유형과 훈련 데이터의 조합 비율이 특히 중요하며, 최적의 비율은 일반적으로 30%로 나타났습니다. 이 연구에서는 또한 대형 생성기 모델이 반드시 더 나은 합성 데이터를 생성하지 않는다는 사실도 강조되었습니다.

- **Performance Highlights**: 재구성된 합성 데이터로만 사전 훈련을 진행했을 경우 자연 웹 텍스트로 사전 훈련을 진행했을 때보다 속도가 더 빠르지 않았습니다. 그러나 텍스트북 스타일의 합성 데이터는 특히 작은 데이터 예산에서 많은 다운스트림 도메인에서 현저히 높은 손실을 발생시키는 경향이 있음을 발견했습니다. 데이터 혼합물이 가진 상호작용의 복잡성은 단순한 유사성 이상의 요소에 의존함을 보여주며, 이는 보다 복잡한 다양성-품질 간의 균형을 시사합니다.



### Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead (https://arxiv.org/abs/2510.01624)
Comments:
          Preprint. Under Review

- **What's New**: 이 논문은 대규모 언어 모델(LLM)을 위해 사후 훈련 후 추론 단계를 재조명합니다. 저자들은 감독 세분화(Supervised Fine-Tuning, SFT)와 강화 학습(Reinforcement Learning, RL)이 독립적으로 이루어지는 현재의 관행에 도전하며, SFT 점수가 RL 후 성능 개선으로 이어지지 않는 사례를 제시합니다. 그들은 SFT 점수가 단순한 데이터에 편향될 수 있음을 발견하고, SFT 성능을 개선한 모델이 원래 모델에 비해 RL 결과에서 나쁜 성과를 보일 수 있는 사례를 보고합니다.

- **Technical Details**: 이 연구는 SFT와 RLVR(Verifiable Rewards)을 통해 12B 파라미터의 수백 개 모델을 훈련하고, 7개의 수학 벤치마크에서 $>1M GPU 시간을 투자해 광범위한 평가를 수행했습니다. 대안적 지표로 포괄성과 Pass@large k 성능을 검토하여 RL 결과 예측에 유용한 신뢰할 수 있는 프록시를 식별하였습니다. 또한, 평가 도구를 오픈 소스화할 예정입니다.

- **Performance Highlights**: 저자들은 SFT 훈련의 질이 RL 성과에 대한 강력한 예측 인자가 아닐 수 있음을 강조하며, 특정 설정에서 SFT와 RL을 Separately 최적화하는 것의 문제점을 지적합니다. 일반적으로 SFT에서 높은 성적을 거둔 모델이 RL 결과에서 우수한 성과를 보이지 않는다는 발견은 중요한 함의를 지니의습니다. 실험 결과, SFT에서 짧은 예시만 사용하는 것이 SFT 성능을 높일 수 있지만, RL 후에는 우수한 성과를 보이지 않을 수 있음을 보여줍니다.



### Posterior Collapse as a Phase Transition in Variational Autoencoders (https://arxiv.org/abs/2510.01621)
Comments:
          12 pages, 8 figures

- **What's New**: 이 논문에서는 변이형 오토인코더(Variational Autoencoders, VAEs)에서 발생하는 사후 붕괴(posterior collapse) 현상을 통계 물리학의 관점에서 조사하였습니다. 이에 따라 사후 붕괴가 데이터 구조와 모델의 하이퍼파라미터에 의해 공동으로 지배되는 위상 전이(phase transition)를 형성한다는 사실을 밝혀냈습니다. 사후 붕괴와 관련된 자명한 해의 안정성을 분석하여 임계 하이퍼파라미터 임계치를 식별하였고, 이 임계 경계는 근사사후(approximate posterior)와 사전 분포(prior distribution) 간의 KL divergence의 불연속성으로 특징지어졌습니다.

- **Technical Details**: VAEs는 고차원 데이터의 잠재 표현(latent representation)을 학습하기 위해 설계된 확률적 생성 모델로, 복잡한 사후 분포를 근사하기 위해 신경망(neural networks)을 사용합니다. ELBO(evidence lower bound)를 최적화하는 목표는 재구성 정확도와 잠재 인코딩의 복잡성을 균형 있게 조정하는 것입니다. 논문에서는 VAEs의 수학적 공식화를 재검토하고 ELBO를 극대화하기 위한 조건을 도출합니다.

- **Performance Highlights**: 이 연구에서는 합성 데이터 및 실제 데이터셋에서 이러한 임계 동작을 확인하여 사후 붕괴가 단순한 최적화 실패가 아닌 데이터 구조와 변이 제약 사이의 상호작용으로부터 발생하는 위상 전이임을 입증하였습니다. 이로 인해 딥 생성 모델의 학습 가능성과 표현 능력에 대한 새로운 통찰을 제공합니다.



### Securing generative artificial intelligence with parallel magnetic tunnel junction true randomness (https://arxiv.org/abs/2510.01598)
Comments:
          4 figures

- **What's New**: 이번 연구에서는 생성 인공지능(Generative Artificial Intelligence, GAI) 모델에서 사용되는 결정론적 의사 난수 발생기(Deterministic Pseudo Random Number Generators, PRNGs)의 취약성을 극복하기 위해 새로운 방안을 제시합니다. 하드웨어에서 생성된 진짜 난수(bit)를 통합함으로써 공격자로부터의 패턴 예측 문제를 해결하고, 기존 방어 방법의 에너지 및 지연(overhead) 문제를 최소화합니다.

- **Technical Details**: 연구팀은 스핀 전이 토크 자기 터널 접합체(Spin-Transfer Torque Magnetic Tunnel Junctions, STT-MTJs)에서 생성된 진짜 난수 비트를 사용하여 FPGA(Field Programmable Gate Array)를 지원하는 고도로 병렬화된 프로토타입 컴퓨터 시스템을 구축하였습니다. 이 시스템은 메가비트 단위로 진짜 난수를 생성하고, NIST(National Institute of Standards and Technology) 무작위성 테스트를 통과하며, 최소의 오버헤드로 동작합니다.

- **Performance Highlights**: GAN(Generative Adversarial Network) 구조를 CIFAR-10 데이터셋에 통합할 경우, 저품질 난수 발생기를 기준으로 불안전한 출력이 최대 18.6배 감소하는 성과를 보여주었습니다. 나노초 수준의 스위칭 속도와 높은 에너지 효율성을 갖춘 STT-MTJ 기반 시스템은 106개 이상의 병렬 셀로 확장할 수 있는 가능성을 지니며, 대형 언어 모델 샘플링에 적합한 기가비트 속도의 처리량을 달성할 수 있습니다.



### Enhancing Noise Robustness of Parkinson's Disease Telemonitoring via Contrastive Feature Augmentation (https://arxiv.org/abs/2510.01588)
- **What's New**: 본 논문에서는 파킨슨병(Parkinson's disease, PD) 원격 모니터링의 신뢰성 문제를 처음으로 확인하고, 환자 유발 측정 부정확성, 환경 소음, 데이터 전송 손실이 UPDRS 예측에 미치는 영향을 논의합니다. 또한 새로운 노이즈 강건(Noise-Robust) UPDRS 예측 프레임워크인 NoRo를 제안하여 비지도 학습으로 노이즈 강건 특징을 학습하는 방법을 도입합니다. 이 프레임워크는 다양한 UPDRS 예측 모델에 자유롭게 적용될 수 있는 유연성을 가지고 있습니다.

- **Technical Details**: NoRo 프레임워크는 원래의 음성 특징을 선택된 특징의 연속 값에 기반하여 순서가 있는 빈(bins)으로 그룹화하는 것을 시작으로 합니다. 다음으로, Contrastive Learning (CL)을 적용하여 노이즈 강건 특징을 생성하며, 같은 빈에 있는 특징을 긍정 쌍으로, 다른 빈의 특징을 부정 쌍으로 처리합니다. 마지막으로 이 강건 특징들은 원래의 음성 특징과 연결되어 UPDRS 예측 모델에 투입되는 확장된 특징(aumented features)으로 구성됩니다.

- **Performance Highlights**: NoRo 프레임워크는 다양한 노이즈 환경에서 UPDRS 예측 모델의 노이즈 강건성을 성공적으로 향상시키며, 예측 오류를 10%-40% 줄이는 성과를 보여줍니다. 실험을 통해 제안된 NoRo의 효과성과 강건성이 입증되었으며, 향후 연구에 기여할 수 있는 공개 소스 코드는 https://github.com/tzm-tzm/PD-Robust 에서 확인할 수 있습니다.



### Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression (https://arxiv.org/abs/2510.01581)
Comments:
          Code: this https URL

- **What's New**: TRAAC (Think Right with Adaptive, Attentive Compression)라는 새로운 방법이 소개됩니다. 이는 모델이 다양한 난이도에 따라 인지하는 단계의 길이를 유동적으로 조절하여 이른바 'under-adaptivity' 문제를 해결하려고 합니다. 이 방법은 온라인 강화 학습(RL)을 활용하며, 모델이 불필요한 추론 단계를 제거하고 효율적으로 중요한 단계를 식별합니다.

- **Technical Details**: TRAAC는 Group Reward Policy Optimization (GRPO) 기반의 방법으로, Proximal Policy Optimization에서 비평가를 제거하고 샘플 응답 그룹에서 기준을 추정합니다. 모델은 각 추론 단계의 주의 점수를 기반으로 중요하지 않은 토큰들을 식별하고 압축합니다. 이 과정에서 난이도에 따라 압축 수준을 조절하며, 어려운 문제에 대해서는 압축을 줄이고, 쉬운 문제에 대해서는 압축을 늘립니다.

- **Performance Highlights**: TRAAC는 다양한 과제에서 평균적으로 8.4%의 정확도 개선과 36.8%의 추론 길이 감소를 달성했습니다. 비수학 데이터셋에서도 강력한 일반화 능력을 보여주었으며, OOD(Out-Of-Distribution) 작업에서 평균 3%의 성능 개선과 40%의 응답 길이 감소를 기록했습니다. 이러한 결과는 TRAAC가 다양한 난이도 작업에서 성능을 개선하고, 불필요한 계산을 줄일 수 있음을 보여줍니다.



### Gradient Shaping Beyond Clipping: A Functional Perspective on Update Magnitude Contro (https://arxiv.org/abs/2510.01578)
Comments:
          Accepted as a conference paper at ACM Multimedia Asia 2025

- **What's New**: 이번 논문에서는 고전적인 경량화 기법인 gradient clipping의 한계를 극복하기 위해 SPAMP(Statistical Per-layer Adaptive Modulation and Projection)라는 새로운 방법론을 제안합니다. SPAMP는 각 레이어의 기울기 통계를 추적하고 동적으로 임계값을 추정하여 업데이트 크기를 조절하는 차별화 가능한 방식으로 변환을 적용합니다. 이 연구는 clipping과 warmup이 효과적인 업데이트 스케일을 제어하는 쌍의 메커니즘으로 작동한다는 새로운 관점을 제공합니다.

- **Technical Details**: 이 논문에서 제안된 SPAMP는 기울기 정규화, warmup, norm clipping을 통합한 부드러운 연결 연산자로 기울기 clipping을 재정의합니다. SPAMP는 각 레이어의 통계적 추적을 통해 동적으로 기울기 스케일을 조절하는 기능을 갖추고 있으며, 이 방법론은 전반적인 학습 효율성을 향상시키는 데 기여합니다. 또한, 이 연구에서는 SPAMP가 손실 감소 동역학을 어떻게 형성하고 업데이트 크기를 조절하는지에 대한 이론적 통찰도 제공합니다.

- **Performance Highlights**: 실험 결과, SPAMP는 이미지 분류 및 트랜스포머 기반 모델에서 기존의 방법들에 비해 안정성, 수렴 속도, 그리고 최종 성능이 향상된다는 것을 보여줍니다. 각종 이미지와 언어 작업에서 광범위한 실험을 통해 이러한 성능 향상이 검증되었습니다. SPAMP는 훈련 과정에서의 가변적인 기울기를 효과적으로 조절함으로써 탁월한 성능을 발휘합니다.



### From Supervision to Exploration: What Does Protein Language Model Learn During Reinforcement Learning? (https://arxiv.org/abs/2510.01571)
Comments:
          24 pages, 7 figures, 4 tables

- **What's New**: 이번 연구에서는 프로틴 언어 모델(PLM)과 강화 학습(RL)을 결합하여 프로틴 디자인의 여러 도메인에서 효율성을 향상시킬 수 있음을 입증하였습니다. RL이 PLM의 잠재력을 기존 사전 훈련 지식(priors) 이상으로 끌어낼 수 있는지를 분석했습니다. 연구 결과, RL은 샘플링 효율성과 성공률을 일관되게 개선하며, 이들 개선은 특정 요인들의 상호작용에 의해 결정된다는 사실이 밝혀졌습니다.

- **Technical Details**: 정의된 통합 프레임워크를 통해, 연구자는 프로틴 시퀀스 최적화 작업을 수행했습니다. 연구의 핵심은 PLM을 사용하여 주어진 3D 구조에 기반한 시퀀스를 생성하는 정책 모델을 개발하는 것입니다. 예상 구조에 따라 시퀀스를 평가하고, 구조적 충실도를 평가하기 위해 TM-Score를 보상 함수로 사용했습니다.

- **Performance Highlights**: 실험 결과, RL은 항상 좋은 시퀀스에 대한 샘플링 효율성을 향상시켜주는 것으로 나타났습니다. 특히, RL의 성공적인 적용은 과제 난이도(task difficulty), 보상 모델의 정확성(reward fidelity) 및 정책 모델의 용량(policy capacity)과 같은 세 가지 주요 요인에 의해 결정되었습니다. RL이 구조적 목표에 대해 얼마나 잘 상승할 수 있는지는 이러한 요인들의 조합에 의해 달라진다고 결론지었습니다.



### TetriServe: Efficient DiT Serving for Heterogeneous Image Generation (https://arxiv.org/abs/2510.01565)
- **What's New**: 이번 논문에서는 고해상도 이미지를 생성하는 데 뛰어난 성능을 자랑하는 Diffusion Transformer (DiT) 모델을 다룹니다. 기존 Serving 시스템은 고정된 시퀀스 병렬성(sequence parallelism)을 사용하여 다양한 해상도와 기한이 있는 요청을 처리하지만, 이는 GPU 활용도가 낮고 SLA(서비스 수준 목표) 달성에 어려움을 겪는 문제를 가지고 있습니다. 새로운 접근법인 step-level sequence parallelism을 통해 요청의 기한에 따라 개별 요청의 병렬 정도를 동적으로 조정할 수 있는 TetriServe라는 DiT 서비스 시스템을 제안합니다.

- **Technical Details**: TetriServe는 요청을 처리하는 데 있어 정해진 라운드 시간(고정된)으로 시간을 분할하여 기한에 따라 최적화된 스케줄링을 수행합니다. 요청의 GPU 병렬 처리 정도를 단계적으로 조정하여, 고해상도 또는 긴급 요청은 더 많은 GPU를 사용하도록 하고, 작은 요청은 자원을 절약하도록 합니다. 이와 함께, 요청의 최소 GPU 할당을 식별하여 후속 라운드에서 지연되는 요청 수를 최소화하는 요청 패킹 기법을 구현합니다.

- **Performance Highlights**: TetriServe는 다양한 실험 설정에서 기존 xDiT(DiT 서비스 엔진)보다 SLA 달성 비율이 최대 32% 향상되는 결과를 보였습니다. TetriServe는 또한 급증하는 요청 패턴과 다양한 작업 믹스를 잘 처리하며, 최신 GPU에서의 성능을 크게 향상시켰습니다. 이를 통해 텍스트-이미지 생성의 품질을 유지하면서도, 고해상도의 요구에 효과적으로 대응할 수 있는 솔루션임을 입증하였습니다.



### Large-Scale Bayesian Causal Discovery with Interventional Data (https://arxiv.org/abs/2510.01562)
- **What's New**: 본 연구에서는 Interventional Bayesian Causal Discovery (IBCD)라는 새로운 접근 방식을 제안합니다. 이는 개입적인(interventional) 데이터를 활용하여 인과 관계를 발견하기 위한 경험적 베이지안 프레임워크입니다. 기존 방법들이 대규모 작업에서 성능이 저조하고 불확실성을 정량화하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: IBCD는 전체 인과 효과(total causal effect)를 포함하는 행렬의 가능성을 모델링합니다. 이 모델은 행렬 정규 분포(matrix normal distribution)로 근사할 수 있으며, 각 엣지는 잠재 변수로 다루어져 불확실성 인지 추론을 가능하게 합니다. 이를 통해 현재의 관찰 데이터에서 스케일 프리(scale-free) 및 Erdős-Rényi 구조의 데이터 기반 가중치를 개별적으로 학습합니다.

- **Performance Highlights**: 포괄적인 시뮬레이션을 통해 IBCD가 기존의 기준선들과 비교했을 때 우수한 구조 회복 능력을 달성했음을 입증합니다. CRISPR 개입 데이터에 IBCD를 적용하여 521개의 유전자의 그래프 구조를 식별하는 데 있어 강력한 성능을 보였습니다. 엣지 후방 포함 확률(edge posterior inclusion probabilities)은 구조의 강건함을 파악하는 데 유용한 지표로 활용됩니다.



### Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization (https://arxiv.org/abs/2510.01555)
- **What's New**: 이 논문은 인간 피드백으로부터의 강화 학습(Reinforcement Learning from Human Feedback, RLHF)에서 Kullback-Leibler(KL) 발산을 이용한 손실 함수의 이론적 기반을 조사합니다. 전통적인 KL 정규화 접근 방식에서 발견된 말도 안되는 구현 방식을 비판하며, 두 가지 다른 구현 스타일을 연결하는 통합된 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 KL 손실의 두 가지 구현 방식, 즉 정책의 점수 함수의 별개의 계수로 사용되는 k_n (보상에서의 k_n)와 직접 손실 함수로서의 k_n (손실로서의 k_n)을 분석합니다. 이 기반 위에서, 전통적인 k_1 보상 방식과 k_2 손실 방식이 실제로는 동일한 그라디언트 표현을 공유함을 증명합니다.

- **Performance Highlights**: 저자들은 최근 도입된 k_3 손실 방식을 비판하며, 이는 원칙적인 손실의 첫 번째 차수 바이어스 근사일 뿐이라고 주장합니다. 이들 발견은 KL 정규화를 선택하고 올바르게 구현하기 위한 포괄적인 그라디언트 기반의 이론적 근거를 제공합니다.



### MIRA: Towards Mitigating Reward Hacking in Inference-Time Alignment of T2I Diffusion Models (https://arxiv.org/abs/2510.01549)
- **What's New**: 본 논문은 MIRA(Reward hAcking 완화)를 제안하여 텍스트 프롬프트에 부합하는 고품질 이미지를 생성하는 확산 모델의 정렬 문제를 해결하려고 합니다. MIRA는 이미지 공간에서 직접적으로 보상을 최적화할 수 있도록 정렬 방법을 혁신하며, 이는 기존 방법들이 겪는 보상 해킹(reward hacking) 문제를 완화합니다. 또한, MIRA-DPO(Bypassing the necessity for fine-tuning을 통해 비미분형(non-differentiable) 보상을 처리하는 새로운 방법도 제시하고 있습니다.

- **Technical Details**: MIRA는 KL 발산(KL divergence)의 새로운 근사치로부터 유도된 정규화를 제공하여 출력 분포가 기본 모델과 유사하게 유지되도록 합니다. 이 방법은 MIRA가 최적화된 노이즈에서 생성된 이미지가 기본 모델로부터 크게 벗어나지 않도록 보장하여 프롬프트 준수를 유지합니다. MIRA-DPO는 고정된 백본(frozen backbone)을 사용하여 비미분형 보상을 직접 최적화하여 학습 과정 없이도 효율적으로 동작할 수 있습니다.

- **Performance Highlights**: MIRA는 SDv1.5 및 SDXL에서 여러 가지 보상 모델(Aesthetic Score, HPSv2, PickScore)과 공개 데이터셋에서 실험을 수행하여 60% 이상의 승률을 기록하며, 프롬프트 준수도 유지하였습니다. 실험 결과에서는 MIRA가 강력한 기준 선(baselines)과 비교하여 80.30%의 승률을 보이며, 보상 해킹 현상을 효과적으로 완화하는 것이 입증되었습니다.



### Predictive Preference Learning from Human Interventions (https://arxiv.org/abs/2510.01545)
Comments:
          NeurIPS 2025 Spotlight. Project page: this https URL

- **What's New**: 이번 연구에서는 Predictive Preference Learning from Human Interventions (PPL)이라는 새로운 방법론을 소개합니다. PPL은 인간의 개입으로부터 얻은 암묵적 선호 신호를 활용하여 에이전트의 미래 궤적을 예측하는 데 기여합니다. 이 알고리즘은 에이전트가 위험한 상태에 접근하지 않도록 선호 최적화를 통해 인간의 중재를 최소화하면서 학습 효율성을 크게 향상시킵니다.

- **Technical Details**: PPL은 인간의 개입이 이루어진 후 L개의 미래 시간 단계로 부트스트랩(bootstrap)하는 것을 주요 아이디어로 합니다. 이러한 작업을 통해 에이전트의 행동을 미래 상태에서도 조정하게 되며, 그 결과로 인간의 개입이 없어도 더 안전한 학습이 이루어질 수 있습니다. 또한 실시간으로 에이전트의 예측 궤적을 시각화하여 인간 감독자의 인지 부담을 줄입니다.

- **Performance Highlights**: PPL 알고리즘은 MetaDrive와 Robosuite 벤치마크에서 실험을 통해 검증되었으며, 기존 방법에 비해 적은 양의 전문가 모니터링과 시연으로도 거의 최적의 정책을 달성할 수 있음을 보여줍니다. 이론적 분석을 통해 알고리즘의 성능 격차를 제한하는 상한선을 도출하였으며, 이는 선호 데이터를 보존하면서 분포적 편차를 줄이는 데 기여함을 강조합니다.



### Executable Counterfactuals: Improving LLMs' Causal Reasoning Through Cod (https://arxiv.org/abs/2510.01539)
- **What's New**: 이번 논문에서는 실행 가능한 반사실적 사고(executable counterfactuals)를 제안하여 대형 언어 모델(LLM)의 인과적 이해를 확장하고 평가하는 새로운 프레임워크를 제공합니다. 기존 연구들은 반사실적 사고의 진단에서 중요한 단계를 건너뛰어 LLM의 성능을 과대평가하는 경향이 있었으나, 본 연구는 모든 단계를 포함하여 보다 정확하게 LLM의 성능을 측정할 수 있는 방법을 제시합니다.

- **Technical Details**: 제안된 프레임워크는 반사실적 사고의 세 가지 기본 기술인 추론(abduction), 개입(intervention), 예측(prediction)을 명시적으로 요구하며, 난이도에 따라 다양한 합성 데이터(synthetic data)를 생성할 수 있도록 설계되었습니다. 이 접근 방식은 코드와 수학 문제를 통해 인과적 사고를 실현하는 데 중점을 두며, 기존의 이론적 약점을 극복하기 위한 목적을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, SOTA 모델인 o4-mini 및 Claude-4-Sonnet 등이 개입사고(interventional reasoning)에서는 높은 정확도를 기록했으나, 반사실적 사고에서는 25-40%의 정확도 감소가 나타났습니다. 강화학습(reinforcement learning)이 모델의 인지적 행동을 유도하고 새로운 분야로 일반화하는 데 효과적이었으며, 이는 코드 문제와 수학 문제 모두에서 기본 모델에 비해 1.5배에서 2배의 성능 향상을 가져왔습니다.



### TimeSeriesScientist: A General-Purpose AI Agent for Time Series Analysis (https://arxiv.org/abs/2510.01538)
- **What's New**: 이 논문에서는 일반적인 시계열 예측을 위한 최초의 LLM 기반 에이전트 프레임워크인 TimeSeriesScientist (TSci)를 소개합니다. TSci는 LLM의 도움을 받아 데이터 통계에 대한 이유를 분석하고, 모델 선택을 좁히며, 최종 예측을 수행하는 과정을 포함한 네 개의 특화된 에이전트로 구성됩니다. 이 프레임워크는 인간의 개입을 최소화하고, 해석 가능하며 확장 가능한 예측 프로세스를 제공합니다.

- **Technical Details**: TSci는 Curator, Planner, Forecaster 및 Reporter라는 네 개의 전문 에이전트로 구성됩니다. Curator는 LLM이 안내하는 진단을 수행하고 데이터 통계에 대한 외부 도구를 사용하여 시각화를 생성합니다. 이어서, Planner는 모델 선택 공간을 좁히고 하이퍼파라미터를 최적화하는 역할을 수행하고, Forecaster는 유효성 검사 결과를 통해 Ensemble 전략을 선택하여 최종 예측을 제시합니다. 마지막으로, Reporter는 모든 중간 분석과 예측 결과를 통합하여 포괄적인 보고서를 생성합니다.

- **Performance Highlights**: TSci는 8개의 공개 벤치마크에서 통계적 및 LLM 기반 모델을 능가하며 평균적으로 각 10.4% 및 38.2%의 예측 오차를 줄였습니다. 또한, 생성된 보고서는 기술적인 엄격성과 명확한 소통을 보여주며, 투명성과 감사 가능성이 요구되는 환경에서 실용적인 배치를 지원합니다. TSci는 전체 예측 워크플로우를 자동화하면서 인간의 전문 지식과의 간격을 메우는 혁신적인 접근 방식을 제시합니다.



### NVIDIA AI Aerial: AI-Native Wireless Communications (https://arxiv.org/abs/2510.01533)
Comments:
          7 pages, 7 figures

- **What's New**: 이 논문에서는 6G 기술의 발전과 함께 AI-native 무선 시스템의 필요성을 강조합니다. 디지털 신호 처리(DSP)와 머신러닝(ML)을 셀룰러 네트워크 소프트웨어 스택에 통합함으로써 현대 네트워크의 생애 주기를 AI 시스템에 맞추는 방향으로 변화시키고자 합니다. 이를 통해 알고리즘을 반복적으로 훈련시키고 배포할 수 있는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 Python 기반 알고리즘을 GPU에서 실행할 수 있는 형태로 컴파일합니다. 이 방법은 효율성과 유연성을 보장하며 NVIDIA GPU에서 최고의 성능을 발휘합니다. 연구의 일환으로, PUSCH 수신기의 채널 추정 기능을 CNN(Convolutional Neural Network)을 통해 수행하는 예제를 제시합니다. 초기에는 디지털 트윈에서, 이후에는 실제 테스트베드에서 실험을 진행합니다.

- **Performance Highlights**: 제안된 방법론은 NVIDIA AI Aerial 플랫폼에서 구현되어 AI/ML 모델의 확대 가능성을 제공합니다. 이는 차세대 셀룰러 시스템에 AI 모델을 통합하기 위한 중요한 기초를 마련하고, 궁극적으로는 본질적으로 지능적인 6G 네트워크의 비전을 실현하는 데 필수적입니다.



### Bypassing Prompt Guards in Production with Controlled-Release Prompting (https://arxiv.org/abs/2510.01529)
- **What's New**: 이 연구에서는 경량화된 프롬프트 가드(prompt guard)를 우회하는 새로운 공격 방법을 소개합니다. 이 연구는 Google Gemini, DeepSeek Chat, Grok, Mistral Le Chat와 같은 주요 대형 언어 모델(LLM)에서 해당 모델들이 사용하고 있는 프롬프트 가드의 한계를 강조합니다. 새로운 공격 방법은 프롬프트 가드와 주요 LLM 간의 자원 비대칭(resource asymmetry)을 활용하여 이들을 우회하는데, 이는 공격면을 새롭게 조명합니다.

- **Technical Details**: 주요 기술 도구는 'controlled-release prompting'으로, 이는 약리학에서 유래한 원칙을 근거로 하여 입력 필터를 완전히 우회하고 악의적인 프롬프트를 LLM에 전달합니다. 이 방법은 타임락 퍼즐(time-lock puzzle)과 타임드 릴리스 암호화를 조합하여 사용하며, 입력 필터가 자원 제약을 초과하는 연산을 요구하게 됩니다. 공격 절차는 세 단계로 나누어져 있으며, 각 단계에서 모델은 악의적인 프롬프트를 안전하게 주입받기 위해 프롬프트를 인코딩합니다.

- **Performance Highlights**: 실험에서는 Google Gemini (2.5 Flash), DeepSeek Chat, xAI Grok (3), Mistral Le Chat (Magistral) 모델에서 공격의 성공률을 보여주었습니다. 이 연구는 12개의 다양한 악의적인 프롬프트를 사용하여 여러 공격 방법의 효과iveness를 평가하였으며, 결과적으로 공격이 여러 LLM 플랫폼에서 성공적으로 작동한다는 것을 증명했습니다. 이러한 실험은 경량화된 통제 메커니즘의 취약성을 강조하며, LLM 안전 메커니즘의 불안정성을 부각시킵니다.



### Round-trip Reinforcement Learning: Self-Consistent Training for Better Chemical LLMs (https://arxiv.org/abs/2510.01527)
Comments:
          19 pages

- **What's New**: 이번 논문에서는 Round-Trip Reinforcement Learning (RTRL)이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 모델이 일관성을 높일 수 있도록 지원하며, 특히 화학 모델의 성능 향상에 기여합니다. RTRL에서는 원래 입력을 성공적으로 매핑할 수 있는 역변환 성공을 보상 신호로 사용하여 현 모델이 더 깊고 일관된 내부 표현을 구축하도록 유도합니다.

- **Technical Details**: RTRL은 화학 도메인에 적합한 데이터 처리를 위해 Simplified Molecular Input Line Entry System (SMILES) 형식을 사용합니다. 이 연구는 분자 캡션과 텍스트 기반 분자 생성, 반응 예측과 회귀 합성 등 두 쌍의 양방향 작업에 초점을 맞추고 있습니다. Group Relative Policy Optimization (GRPO) 알고리즘을 사용하여 정책 LLM을 미세 조정하며, 신뢰영역 목표를 통해 모델을 최적화합니다.

- **Performance Highlights**: 실험 결과에 따르면, RTRL은 모델의 자기 일관성을 최대 52%까지 향상시키고, 주요 작업 성능은 55%까지 개선됩니다. 반복적으로 역할을 바꾸는 접근 방식을 제안하여 모델의 지식을 지속적으로 강화하는 것이 가능하다는 점이 눈에 띕니다. 이는 RTRL이 잠재 지식을 효과적으로 끌어내어 더 강력하고 신뢰할 수 있는 화학 기본 모델로 이어질 수 있음을 보여줍니다.



### On Integer Programming for the Binarized Neural Network Verification Problem (https://arxiv.org/abs/2510.01525)
- **What's New**: 이 논문에서는 bnn의 검증 문제를 해결하기 위해 새로운 기법들을 제안합니다. 첫 번째로, 여러 클래스에 대한 선형 목표를 수립하는 방법을 소개합니다. 두 번째로, bnn의 재귀 구조를 활용하여 유효 불평등을 생성하는 새로운 기법을 제안합니다.

- **Technical Details**: bnn은 이진 가중치와 활성화 함수를 가진 feedforward 신경망으로, 메모리 크기를 줄이고 에너지 효율성을 높입니다. 이 논문에서는 bnn 검증 문제를 정수 프로그래밍 문제로 모델링하고, 이를 해결하기 위해 두 가지 기법을 도입합니다. 먼저, 대체 클래스의 분류 결정을 결합한 단일 최적화 문제를 생성하여 선형 목표를 도출합니다.

- **Performance Highlights**: 제안된 기법은 기존 접근 방식에 비해 보다 넓은 범위의 입력 섭동에 대해 bnn을 검증할 수 있도록 합니다. 또한, bnn 검증 최적화 문제를 해결하는 과정에서 특정 조건이 충족되면 최적화를 조기에 종료할 수 있어 효율적인 검증이 가능합니다. 이로 인해 신속하게 검증 결과를 얻을 수 있는 장점이 있습니다.



### CarbonX: An Open-Source Tool for Computational Decarbonization Using Time Series Foundation Models (https://arxiv.org/abs/2510.01521)
- **What's New**: 이번 연구에서 소개된 CarbonX는 탄소 배출 감축을 위한 머신러닝 기반의 혁신적인 툴입니다. 기존의 도구들이 가진 여러 제약을 극복하며, 탄소 강도 예측 및 임퓨테이션 작업에서 우수한 성능을 발휘하는 소스 모델입니다. 이 툴은 단일 모델로 다양한 전력망에서 매우 효과적으로 사용할 수 있도록 설계되어 있으며, 전 세계 214개 전력망에 걸쳐 정확한 예측 기능을 제공합니다.

- **Technical Details**: CarbonX는 Time Series Foundation Models (TSFMs)을 활용하여 작동하며, 대규모 전력망에서 탄소 강도 예측과 임퓨테이션을 수행할 수 있게 설계되었습니다. 이 도구는 과거 탄소 강도 데이터만을 사용하여 예측 작업의 평균 절대 백분율 오차(Mean Absolute Percentage Error, MAPE)를 15.82%로 달성하며, 하위 예측 성능에서는 16.54%에 이릅니다. 또한, CarbonX는 예측의 불확실성을 포함한 인터벌을 제공하여 실제 배치에 적합한 신뢰성을 보장합니다.

- **Performance Highlights**: CarbonX는 214개 전력망에서 강력한 예측 성능을 자랑하며, 13개의 벤치마크 전력망에서 평균 MAPE 9.59%를 기록하였습니다. 또한, 이 도구는 예측 정확성 저하 없이 최대 21일의 장기 예측이 가능하며, 평균 5.40%의 성능 감소로 이러한 예측을 수행합니다. 이러한 결과들은 CarbonX가 기존 모델보다 사용 상의 유연성과 편리성을 강화하는 실용적인 도구임을 입증합니다.



### Predictive Modeling and Explainable AI for Veterinary Safety Profiles, Residue Assessment, and Health Outcomes Using Real-World Data and Physicochemical Properties (https://arxiv.org/abs/2510.01520)
- **What's New**: 이 연구는 1987년부터 2025년 1분기까지의 약 128만 건의 미국 FDA의 OpenFDA Veterinary Medicine 보고서를 이용해 사망과 회복이라고 하는 결과(outcomes)를 분류하기 위한 예측 모델링 프레임워크를 소개합니다. 데이터 전처리 파이프라인을 통해 관계형 테이블이 통합되고 adverse events (AEs)가 VeDDRA 온톨로지를 통해 표준화되었습니다. 이 연구의 핵심 목표는 식품 생산 동물의 안전성과 인간 건강 보호를 위한 조기에 위험 신호를 감지하는 것입니다.

- **Technical Details**: 연구에서는 Random Forest, CatBoost, XGBoost, ExcelFormer와 같은 감독 학습(supervised learning) 모델과, Gemma 및 Phi와 같은 대형 언어 모델을 평가했습니다. 데이터의 클래스 불균형(class imbalance)을 해결하기 위해 언더샘플링(undersampling) 및 오버샘플링(oversampling) 기법을 사용했습니다. 데이터 전처리에서는 관계형 데이터셋 통합, 표준화된 온톨로지 매핑 및 VeDDRA 활용이 이루어졌으며, SHAP(Shapley Additive exPlanations)를 통해 모델의 해석 가능성과 임상적 관련성이 보장되었습니다.

- **Performance Highlights**: CatBoost와 앙상블 방법(ensemble methods)은 최상의 성능을 보이며 precision, recall, F1-score가 모두 0.95를 달성했습니다. 기존 데이터에서 불확실한 사례에 대한 Average Uncertainty Margin (AUM) 기반의 의사 라벨링을 통합하여 소수 클래스(모티리티에 대한 결과) 탐지를 향상시켰습니다. 최종적으로 이 프레임워크는 고위험 약물 사건 프로필을 조기에 식별할 수 있는 확장 가능하고 투명한 계산 프레임워크를 제공합니다.



### Flock: A Knowledge Graph Foundation Model via Learning on Random Walks (https://arxiv.org/abs/2510.01510)
- **What's New**: 이 논문에서는 지식 그래프(Knowledge Graphs, KGs)에서의 제로샷 링크 예측 문제를 다룹니다. 저자들은 기존의 KGFMs가 갖는 결정론적 동치성(deterministic equivariance)의 한계를 극복하기 위해 확률론적 노드-관계 동치성(probabilistic node-relation equivariance)을 도입합니다. 새로운 모델인 Flock을 제안하여, 이는 무작위 변별성(randomization)을 가미하여 추론 중 대칭을 분해하면서 분포에서 동치성을 유지합니다.

- **Technical Details**: Flock 모델은 무작위 걷기(random walks)를 반복하여 샘플링하고, 이를 시퀀스 모델(seqence model)에서 인코딩 및 집계하여 노드 및 관계의 표현을 생성합니다. 모델은 모든 노드와 관계를 익명화하여 구조적 역할만 학습하게 함으로써 보이지 않는 엔티티 및 관계 타입에 일반화할 수 있도록 설계되었습니다. 이러한 방식으로, Flock은 KGs의 동치성이 보존되며 링크 레벨 함수의 범용 근사기(universal approximator) 역할을 합니다.

- **Performance Highlights**: Flock은 새로운 진단 데이터셋 Petals에서 완벽한 성과를 보이며, 현재까지의 KGFMs가 실패했던 영역을 극복합니다. 또한, Flock은 54개의 다양한 KGs에 대한 엔티티 및 관계 예측 작업에서 기존 KGFMs 보다 뛰어난 성능을 달성합니다. 실험 결과는 Flock 모델의 강력한 일반화 능력과 표현력을 입증합니다.



### Realistic CDSS Drug Dosing with End-to-end Recurrent Q-learning for Dual Vasopressor Contro (https://arxiv.org/abs/2510.01508)
Comments:
          11 pages, 5 figures. Neurips 2025 Workshop Learning from Time Series for Health

- **What's New**: 이번 연구는 강화학습(Reinforcement Learning) 기법을 사용하여 중환자실(ICU)에서 패혈성 쇼크 환자의 두 가지 혈관 수축제(vasopressor) 투약을 위한 최적의 약물 용량과 제어 전략을 배우는 종단 간(end-to-end) 접근 방식을 제안합니다. 연구팀은 행동 공간(action space) 설계를 통해 이산적(discrete), 연속적(continuous), 방향성(directional) 용량 전략을 수용하여 현실적인 약물 투약을 가능하게 합니다. 이를 통해 약물용량 결정의 해석 가능성을 높이고, 임상적으로 채택할 수 있는 정책을 통해 효능을 유지합니다.

- **Technical Details**: 제안된 알고리즘은 오프라인 보수적 Q-learning(Conservative Q-learning)과 반복 재생(recurrent replay) 모델을 결합하여 ICU 시간 시계열 데이터를 기반으로 생체 신호의 시계열 의존성을 포착합니다. 상태 공간(state space)과 행동 공간(action space)을 정의하고, 생존 확률을 극대화하는 보상 함수(reward function)를 설계하였습니다. 이 방법론을 통해 다양한 투약 전략을 비교한 결과, 행동 공간 설계가 학습된 정책에 미치는 영향이 크다는 것을 보여주었습니다.

- **Performance Highlights**: 연구 결과, 제안된 행동 공간 설계가 생존 확률을 15% 이상 향상시키고, 임상 프로토콜과 일치하면서도 약물 투약의 해석 가능성을 개선함을 확인하였습니다. 이는 중환자 치료에서 효과적인 임상 결정 지원 시스템(CDSS) 구축에 중요한 기여를 합니다. 이러한 연구 결과는 향후 패혈증 치료에 대한 RL 기반 접근 방식의 가능성을 보여줍니다.



### Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information (https://arxiv.org/abs/2510.01499)
- **What's New**: 이 논문은 다수결(majority voting) 방식으로 LLM(large language model) 답변을 집계하는 기존 방법의 한계를 극복하기 위해 Optimal Weight (OW)와 Inverse Surprising Popularity (ISP)라는 두 가지 새로운 집계 알고리즘을 제안합니다. 이러한 알고리즘은 1차 및 2차 정보를 활용하여 보다 신뢰성 높은 집계 결과를 도출할 수 있도록 설계되었습니다. 이 연구는 LLM의 다양한 응답을 더 효과적으로 집계함으로써 다수결의 한계를 해소하도록 돕는 것에 중점을 두고 있습니다.

- **Technical Details**: OW는 LLM의 정확도에 기반하여 각 모델에 가중치를 부여하는 선형 집계 방식입니다. 반면 ISP는 모델의 출력 간의 상관관계를 이용하여 집계하므로, 올바른 답변과 비교할 필요 없이 2차 정보를 통해 성능을 개선할 수 있습니다. 이러한 방법들은 이론적으로 보장된 성능 향상을 통해 더 신뢰할 수 있는 집계 결과를 제공하는데 기여합니다.

- **Performance Highlights**: 본 논문에서 제안한 OW와 ISP는 여러 실험에서 다수결 방식보다 일관되게 높은 성능을 보여주었습니다. 연구는 UltraFeedback, MMLU 등 여러 벤치마크에서 검증되었으며, 실제 상황에서도 효과적인 결과를 확인했습니다. 제안된 방법들은 LLM의 상호작용을 통해 집계 결과의 질을 극대화하여, 복잡한 문제 해결에 있어 LLM의 집단적 힘을 더욱 강화하는 데 기여할 수 있습니다.



### Understanding Adversarial Transfer: Why Representation-Space Attacks Fail Where Data-Space Attacks Succeed (https://arxiv.org/abs/2510.01494)
- **What's New**: 최근 연구들은 이미지 분류기와 언어 모델 간의 공격 전이 가능성에 대한 흥미로운 차이를 발견했습니다. 기존에 성공적으로 전이되던 공격 방식이 비전-언어 모델(VLMs) 간에서는 실패한다는 점을 강조합니다. 본 논문에서는 입력 데이터 공간(input data-space)에서의 공격은 전이 가능하지만 모델 표현 공간(model representation space)에서의 공격은 전이되지 않는다는 이론적 구분을 제시합니다.

- **Technical Details**: 저자들은 두 신경망이 같은 입력-출력 맵을 계산하지만 서로 다른 표현을 사용하는 상황을 수학적으로 증명합니다. 데이터 공간 공격(data-space attack)은 무조건 전이 가능하지만, 표현 공간 공격(representation-space attack)은 기하학적 정렬(geometric alignment)이 필요하다는 것을 보여줍니다. 또한, 이미지 분류기와 언어 모델을 대상으로 여러 공격 예시를 만들어 공격 전이의 성공 여부를 실험적으로 확인하였습니다.

- **Performance Highlights**: 결과적으로, 데이터 공간 공격은 VLMs 간에 성공적으로 전이될 수 있지만, 표현 공간 공격은 VLM의 잠재적 기하학적 구조가 충분히 정렬될 때만 전이될 수 있다는 점이 밝혀졌습니다. 이러한 중요한 발견은 공격의 특성이 항상 공통적이지 않으며, 특정 공격 기법이 데이터 공간 또는 표현 공간에 따라 성패가 좌우될 수 있음을 증명합니다. 이 연구는 더욱 강력한 AI 모델을 만드는 데 중요한 통찰을 제공하고 있습니다.



### Density-Ratio Weighted Behavioral Cloning: Learning Control Policies from Corrupted Datasets (https://arxiv.org/abs/2510.01479)
- **What's New**: 이번 논문에서는 오프라인 강화 학습에서 고품질 데이터셋 없이 안정적인 정책 최적화를 위한 새로운 방법인 Density-Ratio Weighted Behavioral Cloning (Weighted BC)를 제안합니다. 이 방법은 작은 검증된 클린 참조 집합을 사용하여, 이진 분류기를 통해 단계 수준의 밀도 비율을 추정합니다. 클린 전문가 행동을 우선시하고 오염된 데이터를 축소하거나 무시함으로써, 데이터 오염 메커니즘에 대한 정보 없이도 안정적인 정책 학습이 가능합니다.

- **Technical Details**: Weighted BC는 오프라인 데이터셋에서 오염된 경로를 식별하고 축소하기 위해 이진 분류기에 기반한 가중치를 사용합니다. 이는 정책 학습에서의 유연성과 안정성을 향상시키며, 학습한 정책이 오염의 종류나 강도와 무관하게 클린 전문가 정책으로 수렴할 수 있다는 이론적 보장을 제공합니다. 또한, 이 연구는 다양한 오염 시나리오에 대한 포괄적인 평가 프레임워크를 설정합니다.

- **Performance Highlights**: 실험 결과, Weighted BC는 높은 오염 비율에서도 최적 성능을 유지하며, 전통적인 Behavioral Cloning (BC), Batch-Constrained Q-learning (BCQ), Behavior Regularized Actor-Critic (BRAC) 기반선에 비해 성능이 우수함을 입증하였습니다. 이러한 성과는 다양한 오염 프로토콜을 포함한 연속 제어 벤치마크에서 확인되었습니다.



### PEL-NAS: Search Space Partitioned Architecture Prompt Co-Evolutionary LLM-driven Hardware-Aware Neural Architecture Search (https://arxiv.org/abs/2510.01472)
- **What's New**: 본 논문에서는 PEL-NAS라는 새로운 HW-NAS 접근 방식을 제안합니다. PEL-NAS는 검색 공간을 파티셔닝하고 아키텍처 프롬프트를 공동 진화시키며 LLM을 활용하는 방법을 통해 높은 정확도와 낮은 지연 시간의 신경망을 신속하게 생성할 수 있습니다. 이 방법은 기존의 LLM 기반 접근 방식이 가지고 있는 탐색 편향을 극복하고 전체 검색 공간을 효과적으로 탐색할 수 있게 합니다.

- **Technical Details**: PEL-NAS는 세 가지 핵심 요소를 포함합니다: 첫째, 복잡성 기반 파티셔닝 엔진이 있으며, 이는 검색 공간을 복잡도에 따라 나누어 다양성을 보장합니다. 둘째, LLM 기반 아키텍처 프롬프트 공동 진화 연산자가 이전 평가의 결과를 바탕으로 지식 베이스를 업데이트하고, 이를 기반으로 하는 지능형 돌연변이와 교차를 수행합니다. 마지막으로, 제로 비용 예측기를 사용하여 많은 후보 모델을 처음부터 훈련하지 않고도 성능을 예측합니다.

- **Performance Highlights**: 실험 결과에 따르면, PEL-NAS는 HW-NAS-Bench에서 기존 기준선보다 최대 54% 낮은 지연 시간과 더 높은 Hypervolume(HV), 낮은 Inverted Generational Distance(IGD)를 달성했습니다. PEL-NAS의 검색 비용은 전통적인 슈퍼넷 방법 대비 수일에서 단 몇 분으로 감소하였습니다. 이는 PEL-NAS가 더 효율적이고 효과적인 신경망 탐색을 가능하게 함을 보여줍니다.



### Fine-tuning LLMs with variational Bayesian last layer for high-dimensional Bayesian optimzation (https://arxiv.org/abs/2510.01471)
- **What's New**: 이 논문에서는 Bayesian Optimization (BO)의 한계를 극복하기 위한 새로운 방법으로 LoRA-VBLL을 제안합니다. 이 방법은 고차원 블랙박스 최적화 문제에서 LLM(대규모 언어 모델)을 대리 모델로 활용하여 계산 효율성을 높이고 불확실성을 정량화합니다. LoRA 기술을 사용해 파라미터 효율적으로 조정하며, 이를 통해 최적화 성능을 개선합니다.

- **Technical Details**: 논문은 LoRA-VBLL이 각 고차원 입력 변수가 목표 함수로 매핑되는 과정을 모델링한다고 설명합니다. 이 방법은 Variational Bayesian Last Layer (VBLL) 프레임워크 아래에서 LLM 파라미터를 조정하고, Low-rank Adaptation (LoRA) 기법을 통해 경량화된 반복적 업데이트를 가능케 합니다. 효율적이고 신뢰성 있는 불확실성 정량화를 위해 데이터 적응형 가중치 평가를 통해 LoRA 순위 선택을 자동화하는 앙상블 기법도 도입하였습니다.

- **Performance Highlights**: LoRA-VBLL 방법은 다양한 고차원 벤치마크 문제 및 실제 분자 최적화 작업에서 기존 방법들보다 우수한 성능을 보여줍니다. 실험 결과에 따르면, (ENS-)LoRA-VBLL 접근법이 최적화 성능과 계산 효율성 모두에서 기존 방법들을 일관되게 초월하며, 복잡한 구조적 의존성을 정량화하는 데 효과적입니다. 이로써, 대규모 언어 모델을 활용한 Bayesian Optimization의 새로운 경로를 제시하고 있습니다.



### The Three Regimes of Offline-to-Online Reinforcement Learning (https://arxiv.org/abs/2510.01460)
- **What's New**: 이번 연구에서는 오프라인에서 온라인으로의 강화 학습에서의 불일치를 설명할 수 있는 안정성-가소성 원칙을 제안합니다. 이 원칙은 사전 훈련된 정책이나 오프라인 데이터셋의 지식을 유지하면서도 충분한 유연성을 보장해야 한다고 강조합니다. 이를 통해 세 가지 서로 다른 온라인 미세 조정 체계를 규명하고, 각 체계에서 요구되는 안정성 특성에 대해 설명합니다.

- **Technical Details**: 연구는 오프라인에서 온라인으로의 강화 학습을 위한 포괄적인 틀을 다룹니다. 에이전트는 오프라인 데이터셋을 기반으로 사전 훈련되어 이후 추가적인 온라인 상호작용을 통해 성능을 개선하는 방식으로 작동합니다. 연구에서는 사전 학습된 정책과 오프라인 데이터셋 간의 상대적인 성능에 따라 세 가지 체계를 체계화하고, 각 체계에 적합한 미세 조정 전략을 선택할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 대규모 실험을 통해 63개의 설정에 대한 연구 결과는 제안된 틀의 예측과 강하게 일치함을 보여줍니다. 연구의 결과는 오프라인에서 온라인으로의 강화 학습에서의 미세 조정 과정에 있어 설계 선택을 안내하는 원칙으로 작용할 수 있음을 입증합니다. 이 연구는 실제 문제를 해결하는 데 있어 더 나은 접근방법을 제시하며, 연구자들이 효과적인 방법을 선택하는 데 도움을 줄 것입니다.



### LSPO: Length-aware Dynamic Sampling for Policy Optimization in LLM Reasoning (https://arxiv.org/abs/2510.01459)
- **What's New**: 이 논문에서는 길이 인식 샘플링(Length-aware Sampling, LSPO)을 제안하여 훈련 데이터 샘플을 동적으로 선택하는 새로운 메타-강화학습 알고리즘을 소개합니다. LSPO는 각 질문의 평균 응답 길이에 따라 샘플을 필터링하여 훈련의 효율성을 높이고, 최종 모델의 정확도를 개선할 수 있는 방안을 제시하고 있습니다. 이를 통해 기존의 강화학습과 비교했을 때 모델 학습의 효과성을 한층 높일 수 있음을 보여줍니다.

- **Technical Details**: LSPO는 응답의 길이를 신호로 활용하여 데이터 필터링을 수행하며, 이는 RLVR(강화학습과 검증 가능한 보상)을 위한 동적 샘플링에서 중요한 기여로 작용합니다. 이 방식은 기존의 손실 함수 설계 방법과는 차별점을 두고 있으며, 응답 길이를 고려함으로써 최종 모델의 성능을 향상시키는 데 중점을 두고 있습니다. LSPO의 검증은 다양한 기본 모델과 데이터 세트에서 수행되었으며, 실험 결과 최종 모델의 성능이 일관되게 향상되었음을 확인했습니다.

- **Performance Highlights**: LSPO를 통한 훈련 모델은 기존 기준 접근법에 비해 향상된 특성과 효율성을 보여줍니다. 비록 LSPO가 표준 RL 알고리즘에 맞추기 위해 추가적인 샘플링을 필요로 하지만, 동일한 훈련 시간 내에서 효과적으로 작동함을 보여줍니다. 이러한 결과는 응답 길이를 데이터를 필터링하는 기준으로 사용하는 것이 RLVR의 효과성을 높이는 데 있어 유망한 방향임을 강조하고 있습니다.



### How Well Can Preference Optimization Generalize Under Noisy Feedback? (https://arxiv.org/abs/2510.01458)
- **What's New**: 본 논문은 소음이 있는 피드백(noisy feedback)이 선호 최적화(preference optimization)에 미치는 영향을 다루고 있으며, 실제적인 LLM 훈련에 더 적합하도록 제한된 단계(finite-step) 선호 최적화에 대한 새로운 통찰을 제공합니다. 기존의 연구들은 소음이 없는 피드백을 가정하지만, 이는 실제 인간 판단에서 발생하는 오류 및 불확실성을 반영하지 못합니다. 본 연구는 두 가지 현실적인 소음 모델을 고려하여 이러한 문제를 해결하고, 소음의 증가가 일반화 성능에 미치는 영향을 이론적으로 규명합니다.

- **Technical Details**: 선호 최적화를 위해, 본 논문은 두 가지 소음 모델인 ϵ-mislabeled 모델과 ω-uncertain 모델을 활용하여 인간 피드백의 소음이 선호 데이터 분포 및 샘플 수에 따라 어떻게 일반화 성능을 저해하는지를 분석합니다. 특히, 데이터가 잘 구별될 경우, 샘플의 수가 충분하다면 노이즈 비율이 증가하더라도 모델의 인구 위험(population risk)은 유지될 수 있다는 주요 통찰을 제시합니다. 이론적 결과를 토대로 우리는 소음에 강한 모델 일반화를 위한 조건을 제공합니다.

- **Performance Highlights**: 현실 세계의 Anthropic 데이터셋에 대해 실험적으로 확인한 결과, 데이터 분포 특성에 따라 성능 감소의 차이를 관찰하였으며, 이는 우리의 이론적 결과와의 밀접한 일치를 통해 검증되었습니다. 본 논문의 결과는 노이즈에 대한 인식 최적화(noise-aware optimization)가 없을 경우 복잡하거나 본질적으로 노이즈가 많은 데이터에서 선호 학습의 도전 과제를 잘 드러냅니다. 전반적으로, 우리의 이론적 분석과 실험적 관찰 간의 일치는 선호 최적화에 대한 소음의 영향을 모델링하는 우리의 이론적 틀의 강력함과 적용 가능성을 강조합니다.



### Fixing That Free Lunch: When, Where, and Why Synthetic Data Fails in Model-Based Policy Optimization (https://arxiv.org/abs/2510.01457)
- **What's New**: 이번 논문은 합성 데이터(synthetic data)가 강화 학습(reinforcement learning)에서 어떻게 긍정적 및 부정적 영향을 미치는지를 조사합니다. 특히, 모델 기반 정책 최적화(Model-Based Policy Optimization, MBPO)가 지속적으로 좋은 성능을 보이는 OpenAI Gym와 달리 DeepMind Control Suite(DMC)에서는 성능 저하를 겪는 이유를 분석합니다. 연구 결과, 두 가지 주요 실패 메커니즘을 식별하였고, 이러한 문제를 해결하여 MBPO가 SAC(Soft Actor-Critic)보다 더 나은 성능을 발휘할 수 있도록 하는 새로운 접근법을 제안합니다.

- **Technical Details**: 이 논문에서는 주요 실패 메커니즘으로 1) 동작 및 보상 모델 간 스케일 불일치(scale mismatches)와 2) 모델 변화를 불안정하게 만드는 잔여 예측(residual prediction)으로 인한 분산(inflated variance)을 지적합니다. 이와 함께, MBPO의 성능을 높이기 위해 두 가지 간단한 수정(타겟 정규화(target normalization) 및 직접 다음 상태 예측)으로 구성된 Fixing That Free Lunch(FTFL) 방법론을 도입합니다. 이러한 방법으로 MBPO는 DMC에서 이전에는 불가능했던 정책 개선을 이끌어냅니다.

- **Performance Highlights**: FTFL 방법론을 적용한 결과, MBPO는 DMC의 7개 과제 중 5개에서 SAC보다 뛰어난 성능을 기록하며 이전에 보고된 OpenAI Gym에서의 우수한 성능을 유지합니다. 또한, 모델 용량(model capacity)을 확장할 경우 결과는 더욱 개선되어 높은 확장 가능성을 보여줍니다. 이 연구는 RL 알고리즘의 설계와 환경의 구조가 어떻게 알고리즘의 성공 또는 실패를 좌우하는지를 명확히 보여줍니다.



### SCOPED: Score-Curvature Out-of-distribution Proximity Evaluator for Diffusion (https://arxiv.org/abs/2510.01456)
- **What's New**: 소개되는 SCOPED(Score-Curvature Out-of-distribution Proximity Evaluator for Diffusion)는 외부 분포(OOD) 탐지를 위한 고속 및 일반-purpose 방법론입니다. 이 방법은 기존 방법보다 모델의 포워드 패스를 수의 한 자리 만큼 줄여주며, 대부분의 diffusion 기반 기준 성능을 초과하고 최첨단 성능에 근접합니다. SCOPED는 다양한 데이터셋에서 한 번만 훈련된 단일 diffusion 모델로부터 계산되며, 이를 통해 유연하고 비지도 학습 기반의 테스트를 가능하게 합니다.

- **Technical Details**: SCOPED는 정보 기하학(information geometry)의 기본 직관을 활용하여 점검합니다. 로그 확률 밀도의 국부적 곡률(local curvature)과 스코어 함수의 노름(norm)을 결합한 간단한 통계치를 생성하여 주어진 쿼리 포인트가 내부 분포인지 외부 분포인지 판단하는 신뢰할 수 있는 신호를 제공합니다. 이는 단일 Jacobian-vector product(JVP)를 통해 효율적으로 적용되며, 다른 기존 방법보다 훨씬 적은 양의 모델 평가는 필요합니다.

- **Performance Highlights**: SCOPED는 DeepMind Control Suite(DMC) 및 D4RL Gym 벤치마크에서 강화 학습 환경의 분포 변화를 성공적으로 분리합니다. 또한, CIFAR-10, SVHN, CelebA, CIFAR-100의 네 가지 비전 벤치마크에서 SCOPED가 영역 아래 수신자 작동 특성 곡선(AUROC) 점수에서 경쟁력을 보이며, 대부분의 이전 방법에 비해 훨씬 적은 평가를 요구하므로 효율성이 높음을 입증하였습니다.



### Local Linear Attention: An Optimal Interpolation of Linear and Softmax Attention For Test-Time Regression (https://arxiv.org/abs/2510.01450)
- **What's New**: 본 연구는 Local Linear Attention (LLA)라는 새로운 주의 메커니즘을 제안하며, 이는 비모수 통계의 관점에서 테스트 시간 회귀(test-time regression)로부터 유도된 것이다. LLA는 Softmax Attention의 효율성을 더욱 발전시킬 수 있는 가능성을 보여주고 있으며, 이 응용은 아직 탐색되지 않은 영역이다. 이 방법은 바이어스-분산(bias-variance) 트레이드오프 분석을 통해 연상 기억(associative memory)에 있어서 이론적인 이점을 제공한다.

- **Technical Details**: LLA는 Softmax Attention 및 Linear Attention과 비교하여 수학적인 분석을 통해 이점을 제공하며, Θ(n^2 d)와 Θ(n d^2)로 나타나는 메모리 복잡성을 줄이기 위해 두 가지 메모리 최적화 방법을 제안한다. 이러한 최적화는 LLA의 계산을 블록 단위로 병렬화하여 현대 가속기에서의 효율적인 구현을 가능하게 한다. 또한, 맞춤형 추론 커널을 구현하여 메모리 오버헤드를 크게 줄여준다.

- **Performance Highlights**: LLA는 비정상(non-stationarity) 환경에서 효과적으로 적응하며, 테스트 시간 학습(test-time training) 및 컨텍스트 학습(in-context learning)에서 강력한 기준선보다 더 우수한 성과를 보여주었다. 실험 결과는 LLA가 대규모 모델에 대한 확장성과 적용 가능성을 보이는 유망한 증거를 제시한다. 학습 성능은 주의 메커니즘의 이론적 기준을 바탕으로 하여 최적화 단계의 성공적인 수행을 통해 더욱 강화되었다.



### SoftAdaClip: A Smooth Clipping Strategy for Fair and Private Model Training (https://arxiv.org/abs/2510.01447)
- **What's New**: 이 논문은 차등 개인 정보 보호(Differential Privacy, DP) 분야에서 SoftAdaClip이라는 새로운 훈련 방법을 도입합니다. 이 방법은 경량의 tanh 기반 변환을 통해 경량 클리핑(hard clipping)을 대체하여 모델의 공정성과 성능을 향상시킵니다. 연구 결과, SoftAdaClip은 DP-SGD에 비해 최대 87%의 그룹 간 차이 감소를 보이며, 실험 데이터셋으로는 MIMIC-III, GOSSIS-eICU, Adult Income을 사용했습니다.

- **Technical Details**: SoftAdaClip은 단순한 경량 클리핑 대신 부드러운 tanh 기반 변환을 사용하여 경량 신호의 상대적인 크기를 보존하면서 민감도를 제어합니다. 이를 통해 소수 집단의 학습 신호가 억제되는 문제를 해결하고, 다양한 데이터 세트를 통해 공정성과 유용성을 검증합니다. 이 방법은 비대칭적으로 클리핑되는 경량으로부터의 부정적인 영향을 줄이고 실제 세계에서의 공정성을 높이고자 합니다.

- **Performance Highlights**: SoftAdaClip을 적용했을 때, DP-SGD와 비교해 그룹 간 차이가 최대 87%까지 감소했으며, Adaptive-DPSGD에 비해 최대 48% 감소했습니다. 이 연구는 부드러운 변환과 적응적 메커니즘의 통합이 공정하고 개인 정보 보호를 보장하는 모델 훈련에 필수적임을 증명했습니다.



### Edge Artificial Intelligence: A Systematic Review of Evolution, Taxonomic Frameworks, and Future Horizons (https://arxiv.org/abs/2510.01439)
- **What's New**: 이번 논문에서는 Edge Artificial Intelligence (Edge AI)의 발전과 현재 상황, 그리고 미래 방향성을 체계적으로 검토합니다. Edge AI는 네트워크의 최전선 장치에 직접적으로 지능을 통합하여, 데이터의 원천 근처에서 처리함으로써 개인정보 보호 및 지연 시간을 줄이는 데 초점을 맞춥니다. 이 리뷰는 PRISMA 가이드라인에 따라 작성되었으며, 2,200개 이상의 기록 중 79개의 주요 연구를 선정하여 질적 분석을 수행했습니다.

- **Technical Details**: 연구는 Edge AI의 네 차원 분류 체계를 통해 하드웨어 아키텍처(CPU, ASIC, FPGA 등), 처리 능력(TinyML, federated learning 등), 애플리케이션 분야(헬스케어, 산업 IoT 등), 배치 위치(Device Edge, Cloud Edge 등)를 포함한 다양한 요소를 통합적으로 분석합니다. 이를 통해 기존 문헌에서 나타나는 연구 간극을 체계적으로 식별할 수 있는 토대를 마련했습니다. 또한, 본 리뷰에서는 Edge AI의 발전 과정을 과거의 콘텐츠 전송 네트워크와 안개 컴퓨팅(Fog Computing)에서 현대의 장치 내 지능으로 구체적으로 설명합니다.

- **Performance Highlights**: 논문의 기여점은 세 가지로 요약됩니다. 첫째, Edge AI의 역사적 맥락을 새로운 방식으로 정리하여 이전 조사에서 부족했던 연속성을 제공합니다. 둘째, 다양한 차원을 통합하는 보여주는 통합된 분석 프레임워크를 제시함으로써, 관련 기술들과 이들의 상호작용을 보다 잘 이해할 수 있도록 했습니다. 셋째, Edge AI 시스템의 기술적 제약과 배치 문제를 포괄적으로 분석하여 연구의 지평을 넓히기 위한 기회를 제시하였습니다.



### Ultra-Efficient Decoding for End-to-End Neural Compression and Reconstruction (https://arxiv.org/abs/2510.01407)
Comments:
          5 pages, 4 figures, NeurIPS 2025 Workshop MLForSys

- **What's New**: 이 논문에서는 최신 신경 압축 방법의 디코더 병목 현상을 해결하기 위해 새로운 압축-재구성 프레임워크를 제안합니다. 이는 저랭크 표현(low-rank representation)을 포함한 오토인코더(autoencoder)를 사용하여 저비용의 재구성을 가능하게 합니다. 제안된 방법은 디코딩 단계의 계산 오버헤드를 크게 줄이며, 고화질 이미지 출력을 유지하면서 디코더의 계산 병목을 사실상 제거합니다.

- **Technical Details**: 본 연구에서는 벡터 양자화(vector quantization)를 결합한 저랭크 표현을 오토인코더에 통합하여 고도화된 압축 기술을 구현합니다. 주요 기술적 기여는 인코딩된 잠재 표현(encoded latent representation)에서 직접적으로 학습 가능한 저랭크 근사를 수행함으로써 디코딩 계산량을 상당히 줄이는 것입니다. 이 과정은 수정된 트랜스포머 기반 인코더와 함께 결합되어 초경량 디코딩 체계를 통해 높은 품질의 출력을 제공합니다.

- **Performance Highlights**: 제안된 프레임워크는 이미지 크기를 21배 이상 줄이면서도 평균 제곱 오차(Mean Squared Error, MSE)가 3.6×10−3에 도달하는 높은 압축률을 달성합니다. 또한, 이 접근 방식은 Googles Deepmind Sonnet VQVAE와 같은 최첨단 방법과 비교할 때 디코더의 계산 용량을 10배에서 100배까지 줄일 수 있음을 보여줍니다. 이는 기존의 신경 압축 방법들에 비해 실용적인 활용도를 크게 높일 것으로 기대됩니다.



### Neural Network Surrogates for Free Energy Computation of Complex Chemical Systems (https://arxiv.org/abs/2510.01396)
Comments:
          6 pages, 4 figures. This work has already been accepted for presentation in The 29th International Computer Science and Engineering Conference (ICSEC) 2025, Chiang Mai, Thailand, and will be published in IEEE Xplore

- **What's New**: 이번 연구에서는 Gaussian Process Regression (GPR)과 같은 자유 에너지 복원 방법에서 중요한 한계를 해소하기 위한 신경망 대체 프레임워크를 제안합니다. 이 프레임워크는 카르테시안 좌표로부터 집합 변수(Collective Variables, CV)를 직접 학습하고, 자동 미분(Automatic Differentiation)을 통해 제이콥(Jacobian)을 생성하여 복잡한 CV의 사용을 가능하게 합니다. MgCl2의 이온 쌍 시스템에서 단순 거리 CV와 복잡한 коordination-number CV 모두에 대해 높은 정확도를 달성했습니다.

- **Technical Details**: 고차원 집합 변수(CV)는 자유 에너지 지형을 매핑하고 최소 자유 에너지 경로(minimum free energy paths, MFEPs)를 추출하는 데 필수적입니다. 그러나 고차원성 및 드문 사건으로 인해 이러한 계산이 어렵습니다. 기존 GPR 및 신경망은 CV의 해석적 제이콥을 필요로 하며, 이는 복잡한 변수에 대해 실현 불가능하다는 문제를 가지고 있습니다. 저자는 자동 미분 기능을 활용하여 제이콥을 정확하고 효과적으로 학습할 수 있는 신경망 프레임워크를 도입했습니다.

- **Performance Highlights**: 연구에서는 MgCl2 시스템을 통해 단순한 거리 CV와 복잡한 коordination-number CV에 대한 높은 정확도를 검증했습니다. 제이콥 오차는 거의 가우시안 분포를 따르며, 이는 GPR 파이프라인에 적합하다고 평가되었습니다. 이 프레임워크는 복잡한 머신 러닝 집합 변수를 포함할 수 있는 기회를 마련하여 생화학 및 재료 시뮬레이션의 범위를 넓힙니다.



### Optimal Stopping vs Best-of-$N$ for Inference Time Optimization (https://arxiv.org/abs/2510.01394)
Comments:
          24 pages

- **What's New**: 본 논문은 대형 언어 모델(LLM)의 추론 과정에서 출력 품질과 비용을 균형 있게 조정하는 새로운 최적화 프레임워크를 제안합니다. 이 프레임워크는 고전적인 'Pandora’s Box' 문제에 기반하여, 무작위 보상을 가진 '상자'를 여는 것으로 각 생성을 이해합니다. 이로 인해, 알고리즘이 보상 분포를 모르더라도 언제 생성을 중단해야 할지를 결정할 수 있도록 합니다.

- **Technical Details**: 이 논문에서는 우리가 개발한 UCB 스타일의 Pandora's Box 알고리즘을 소개합니다. 이 알고리즘은 알려지지 않은 보상 분포에 적응하도록 설계되었으며, 최적 중단 임계치를 유지하여 Weitzman의 최적 정책에 비례하는 실수 없이 결과를 보장합니다. 또한 Bradley–Terry에서 영감을 받은 보상 정규화 변환을 통해 다양한 프롬프트 간의 보상 스케일링 문제를 해결하며, 이를 통해 적응형 메타 생성 프로세스를 제공합니다.

- **Performance Highlights**: 실험 결과, AlpacaFarm 및 HH-RLHF 데이터셋에서 우리의 적응형 전략은 비적응형 Best-of-N 샘플링과 동일한 보상을 얻으면서도 평균적으로 15-35% 덜 생성할 수 있음을 보여줍니다. 이는 이론적 성과 경계 및 LLM 배포를 위한 실질적인 효율성 증대를 모두 제공하며, 최적 중단 이론과 추론 시간 스케일링 사이의 원칙적인 연결을 수립합니다.



### Fine-Tuning Masked Diffusion for Provable Self-Correction (https://arxiv.org/abs/2510.01384)
- **What's New**: 본 논문에서는 생성 모델의 자기 수정(self-correction)의 중요성을 강조하며, Masked Diffusion Models (MDMs)에 대한 새로운 접근 방식인 PRISM을 소개합니다. PRISM은 사전 훈련된 MDM에 적용할 수 있는 경량의 모델 비의존적 접근 방식으로, 이전의 방법들이 요구했던 MDM 아키텍처 전면 개편이나 부정확한 품질 프로xies에 의존하지 않습니다. 이를 통해, 다양한 도메인에서 MDM 추론을 향상시킬 수 있는 기회를 제공합니다.

- **Technical Details**: PRISM은 자기 수정 손실(self-correction loss)을 정의하며, 이는 RL(강화 학습)이나 검증자(verifier) 없이도 각 토큰의 품질 점수를 학습할 수 있도록 합니다. 이 품질 점수는 MDM과 동일한 순방향 패스(forward pass)에서 계산되며, 저품질 토큰을 탐지하는 데 사용됩니다. 이론적으로 이 접근 방식은 MDM의 적용성을 높이는 강력한 기반을 제공합니다.

- **Performance Highlights**: 실험 결과 PRISM은 Sudoku 퍼즐, 조건 없는 텍스트(170M) 및 LLaDA(8B) 코드와 같은 다양한 도메인에서 MDM 추론을 개선하는 데 기여함을 보여줍니다. 이러한 성과들은 PRISM이 다양한 상황에서 잘 작동할 수 있음을 입증하며, generative modeling의 가능성을 확장하는 데 중요한 역할을 하고 있습니다.



### Selective Underfitting in Diffusion Models (https://arxiv.org/abs/2510.01378)
- **What's New**: 최근의 연구에서 확산 모델(Diffusion models)은 생성 모델링의 주요 패러다임으로 부각되고 있으며, 이들이 사실상 어떤 스코어(score)를 학습하는지에 대한 질문이 제기되었습니다. 본 논문에서는 '선택적 언더피팅(selective underfitting)'이라는 개념을 도입하여, 확산 모델이 입력 공간의 특정 영역에서 스코어를 정확히 근사하는 대신 다른 영역에서는 언더피팅한다는 새로운 시각을 제시합니다.

- **Technical Details**: 이 연구는 모델이 데이터 공간의 전역적 스코어를 학습하는 것이 아닌, 특정 영역에서 더 정확한 스코어 근사를 통해 언더피팅이 발생하는 메커니즘을 설명합니다. 또한, 이러한 개념을 검증하기 위한 경험적 개입(empirical interventions)을 설계하여, 가격 대조를 통해 학습한 스코어와 실제 데이터를 비교합니다.

- **Performance Highlights**: 연구 결과, 선택적 언더피팅이 확산 모델의 이해와 일반화(generalization), 생성적 성능에 중요한 요소임을 밝혔다. 이는 향후 모델 개선 및 신규 샘플 생성에 대한 가능성을 열어주며, 실제로 확인 가능한 새로운 통찰(insights)을 제공합니다.



### RheOFormer: A generative transformer model for simulation of complex fluids and flows (https://arxiv.org/abs/2510.01365)
Comments:
          8 pages, 5 figures. Submitted to PNAS

- **What's New**: 이번 연구에서는 복잡한 유체 흐름의 다양한 공간 상호작용과 특징을 효율적으로 학습할 수 있는 생성적 오퍼레이터 학습 방법인 Rheological Operator Transformer (RheOFormer)를 도입하였습니다. 전통적인 수치적 방법이 요구하는 계산 자원의 부담을 줄이고, 다양한 물리적 조건에서의 재학습 요구를 최소화하여, 실용적인 응용을 가능하게 합니다.

- **Technical Details**: RheOFormer는 자가 주의(self-attention) 메커니즘을 활용하여 비선형 및 역사 의존적 구성 모델을 통해 내부 응력 텐서와 변형 텐서 간의 관계를 효과적으로 모델링합니다. 이를 통해 다양한 유변학적(viscometric) 및 비유변학적(non-viscometric) 흐름에 적용하여 그 성능을 평가합니다.

- **Performance Highlights**: RheOFormer는 제한된 데이터셋으로도 다양한 복잡한 유체의 스칼라 및 텐서 비선형 역학을 정확하게 학습하고 예측할 수 있습니다. 이 모델은 강력한 일반화 능력과 계산 효율성을 바탕으로 복잡한 유체 시뮬레이션을 가속화하는 신뢰할 수 있는 신경 대체 모델로 자리매김하며, 데이터 기반 실험 및 실시간 공정 최적화를 지원할 수 있는 가능성을 제시합니다.



### To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking (https://arxiv.org/abs/2510.01349)
Comments:
          A short version of this paper appeared at the ICLR AI4Mat workshop in April 2025

- **What's New**: 이번 연구에서는 기계 학습에서의 대칭 인식(symmetry-aware) 방법을 비판적으로 평가하는 새로운 메트릭(metric)을 제안합니다. 이 메트릭은 데이터셋의 비등방성(anisotropy) 또는 대칭 파괴(symmetry-breaking) 정도를 정량화하여 기존 데이터셋과 무작위로 증강된 데이터셋을 구별하는 두 표본 신경 분류기 테스트를 기반으로 합니다.

- **Technical Details**: 제안된 메트릭의 유효성을 합성 데이터셋(synthetic datasets)에서 검증하였고, 여러 벤치마크 포인트 클라우드(point cloud) 데이터셋에서 높은 정렬(alignment) 정도를 발견했습니다. 이론적으로, 분포의 대칭 파괴가 실제로는 불변(invariant) 레이블이 존재하더라도 불변 방법들이 최적으로 작동하지 않게 할 수 있음을 보여주었습니다.

- **Performance Highlights**: 실험적으로 대칭 인식 방법의 효과는 데이터셋에 따라 달라짐을 발견하였습니다. 일부 비등방성 데이터셋에서는 여전히 유익할 수 있지만, 그렇지 않은 경우도 존재한다는 사실이 드러났습니다. 이러한 연구 결과는 대칭 인식의 작동 조건과 원인에 대해 재고할 필요성을 제기합니다.



### Self-Supervised Representation Learning as Mutual Information Maximization (https://arxiv.org/abs/2510.01345)
- **What's New**: 이 논문에서는 Self-supervised representation learning (SSRL)의 기본 원리에 대해 깊이 있는 접근을 통해 논의합니다. 기존 연구들이 정보 이론적 목표나 representation collapse를 방지하는 휴리스틱(hurstic)을 중심으로 SSRL 방법을 통합하려고 했음에도 불구하고, 예측기 네트워크(predictor network)나 정지 그래디언트(stop-gradient) 작업과 같은 아키텍처 요소는 대부분 경험적 동기에 의해 추가된 것으로 간주되었습니다. 본 연구는 SSRL 알고리즘의 학습 목표가 최적화 전략 및 모델 설계 선택에 어떤 영향을 미치는지 조사합니다.

- **Technical Details**: 이 논문에서 저자들은 변분적 상호 정보(variational mutual information, MI) 하한을 시작으로 두 가지 훈련 패러다임, Self-Distillation MI (SDMI) 및 Joint MI (JMI)를 도출합니다. SDMI는 교대 최적화(alternating optimization)를 요구하여 이론적으로 정지 그래디언트 작업이 필수적입니다. 반대로 JMI는 대칭 구조(symmetric architectures)를 통해 공동 최적화(joint optimization)를 허용하며 이러한 구성 요소 없이도 작동합니다.

- **Performance Highlights**: 이 논문은 SDMI와 JMI에서 예측기 네트워크(predictor networks)와 통계적 정규화기(statistical regularizers)가 MI 목표의 적절한 대리자로 등장한다는 것을 보여줍니다. 많은 기존 SSRL 방법들이 이러한 두 가지 패러다임의 특수한 사례 또는 근사임을 입증하며, 여러 아키텍처 구성 요소의 선택에 대한 이론적 설명을 제공합니다. 이 연구는 SSRL 방법론의 효과적인 구성 요소 선택에 대한 통찰력을 제공합니다.



### On the Identifiability of Latent Action Policies (https://arxiv.org/abs/2510.01337)
Comments:
          10 pages

- **What's New**: 이 논문에서는 최근 도입된 잠재 행동 정책 학습(Latent Action Policy Learning, LAPO)의 식별 가능성(identifiability)을 연구합니다. LAPO는 비디오 데이터에서 행동의 표현을 발견하기 위한 프레임워크로, 우수한 전문가 정책에 의존하지 않고도 학습할 수 있는 방법을 제시합니다. 연구의 주요 기여는 LAPO에서의 아이디어를 구체화하고, 행동 표현이 원하는 조건을 충족하도록 보장하는 엔트로피 정규화된 LAPO 목표를 증명하는 것입니다.

- **Technical Details**: LAPO는 세 단계로 구성되어 있으며, 첫 번째로, 상태-다음 상태 쌍의 대규모 데이터셋을 활용하여 역 동역학 모델(Inverse Dynamics Model, IDM) q^(a|x,x′)을 최적화합니다. 이후 이를 통해 라벨이 없는 비디오 데이터셋에 행동(a) 레이블을 붙이고, 마지막 단계에서는 행동 레이블이 붙은 소규모 데이터셋을 통해 행동을 실제 행동(a)로 매핑하는 학습 가능한 헤드를 적용합니다. 각 단계에서 행동 표현을 효과적으로 식별하기 위한 통계적 이점을 설명하며, 불확실성의 잠재적인 원인도 논의됩니다.

- **Performance Highlights**: 이 연구의 결과는 LAPO가 실제로 좋은 정책을 훈련하기 위해 요구되는 행동 레이블 샘플의 양을 줄일 수 있음을 보여줍니다. 또한, LAPO는 강화 학습(Reinforcement Learning)으로 더욱 효율적으로 미세 조정할 수 있다는 것을 입증합니다. 해당 프레임워크는 Genie와 LAPA와 같은 대규모 응용 프로그램에 적용되었으며, 이러한 이유로 행동 표현의 식별 가능성에 대한 분석은 매우 중요합니다.



### Quantum-inspired Benchmark for Estimating Intrinsic Dimension (https://arxiv.org/abs/2510.01335)
Comments:
          19 figures, 35 pages

- **What's New**: 이 논문에서는 기계 학습 모델이 실제 데이터셋에서 잘 일반화될 수 있는 이유가 저차원 내재다양성(intrinsic dimension, ID)을 가진 잠재적인 리만 다양체(manifold) 위에 데이터가 놓여 있기 때문이라는 가설에 기반하고 있다. 저자는 QuIIEst(Quantum-Inspired Intrinsic-dimension Estimation)라는 새로운 벤치마크를 제안하며, 이는 알려진 ID를 가진 복잡한 다양체의 무한 패밀리로 이루어져 있다. 이 벤치마크는 양자 광학의 기법을 바탕으로 복잡한 기하학적 구조를 가진 다양체들을 만들어낸다. 이를 통해 기존 ID 추정 방법들의 성능을 보다 엄밀하게 평가할 수 있는 기반을 제공한다고 강조한다.

- **Technical Details**: 저자는 QuIIEst 벤치마크의 개발을 위해 양자 정보 이론의 도구를 활용하여 고차원적 아핀 공간에서 비-단순(non-trivial) 다양체를 구성한다. 이 과정에서 다양한 왜곡된 다양체를 통해 ID 추정(performance of IDE)을 조사하며, 대칭성 그룹(symmetry groups)으로부터 독립적으로 다루는 기법들을 탐구한다. 또한, QuIIEst 벤치마크는 복잡성을 가진 무한한 다양체 패밀리를 포함하고 있으며, 이는 기존의 간단한 위상 다양체(topologically simple manifolds)와 비교할 때 도전적인 요소를 부여한다. 특히, ID 추정 방법들이 어떻게 다양한 기하학적 구조에 따라 성능이 달라지는지 분석한다.

- **Performance Highlights**: QuIIEst의 다양체에서 실험한 ID 추정 기법들은 기존의 벤치마크보다 일반적으로 정확도가 낮았다. 특히 다양한 곡률(curvature)을 가진 복잡한 다양체에서도 성능 저하가 미미하게 관찰되었다. 이러한 결과는 QuIIEst가 본질적으로 도전적인 벤치마크임을 증명하는 동시에, ID 추정 방법들의 신뢰성을 향상시킬 필요성을 드러낸다. 또한, fractal 형태의 Hofstadter's butterfly에 대한 ID 추정 수행 결과, 비-다양체에 대한 다양한 방법들이 직면하는 과제를 조명하였다.



### Low Rank Gradients and Where to Find Them (https://arxiv.org/abs/2510.01303)
- **What's New**: 이 논문은 두 층 신경망에서 훈련 손실의 기울기에서 발견되는 저랭크 구조(low-rank structure)를 탐구합니다. 기존의 등방성(isotropic) 가정에서 벗어나 비등방적(anisotropic)이고 불안정한(ill-conditioned) 데이터에서 이러한 구조가 발생하는 과정을 분석합니다. 저자는 훈련 데이터의 속성과 활성화 함수의 선택이 두 가지 주요 구성 요소의 균형을 결정짓는 방식으로 기울기와 관련된 구조를 이해하고자 합니다.

- **Technical Details**: 저자는 입력 가중치에 대한 기울기가 대체로 저랭크 구조로 근사된다는 사실을 발견하였고, 이는 두 개의 주요 랭크-1 구성 요소에 의해 지배됩니다. 데이터 대 잔여물의 정렬(align)과 입력 데이터의 랭크-1 스파이크 간의 상호작용이 중요하다고 설명합니다. 기울기의 저랭크 구조를 이해하기 위해 다양한 정규화 기법이 미치는 영향을 분석하며, 일반적인 정규화 기법이 이러한 구성 요소를 어떻게 조절하는지를 보여줍니다.

- **Performance Highlights**: 이론적 예측은 합성 데이터와 실제 데이터(MNIST, CIFAR-10)에서의 실험을 통해 검증되었습니다. 저자는 데이터 스파이크의 크기, 데이터의 스펙트럼 감쇠 프로필 등 다양한 요인이 기울기 구조에 미치는 영향을 분석합니다. 또한, 각 구성 요소의 상대적 중요성이 데이터의 속성, 네트워크 매개변수화의 크기, 손실 및 activation function 선택에 의해 결정된다는 점을 강조합니다.



### From 2D to 3D, Deep Learning-based Shape Reconstruction in Magnetic Resonance Imaging: A Review (https://arxiv.org/abs/2510.01296)
- **What's New**: 현재의 3D MRI 재구성 방법론을 체계적으로 조사하는 본 리뷰는 점군(point cloud), 메쉬 기반(mesh-based), 형태 인식(shape-aware) 및 체적 모델(volumetric models)이라는 네 가지 주요 접근 방식에 초점을 맞추고 있습니다. 각 분류에 대해 최신 기술, 방법론적 기초, 한계 및 여러 해부학적 구조에 걸친 응용 분야를 분석합니다. 또한, 질병이 있는 해부학에 대한 모델의 임상 적용 가능성 및 훈련과 테스트 데이터의 영향을 강조합니다.

- **Technical Details**: 2D MRI 스택으로부터 3D 모형을 생성하는 데 있어 딥러닝(deep learning) 기술이 사용되고 있으며, CNN(Convolutional Neural Networks), GANs(Generative Adversarial Networks) 및 확산 모델(diffusion models)과 같은 혁신적인 아키텍처가 주목받고 있습니다. 본 연구는 각기 다른 인체 구조에 대한 재구성을 가능하게 하여, 딥러닝 기술이 의료 영상 처리에서 어떻게 변화를 일으키고 있는지를 설명합니다. 그 과정에서, 현재의 딥러닝 모델이 겪는 일반화 문제와 다양성 문제도 다루고 있습니다.

- **Performance Highlights**: 딥러닝 기반 3D 재구성 모델은 전통적 기법들을 넘어서는 성능을 보이며, 특히 움직임 왜곡이나 비정상적인 병리 현상이 있을 경우 더욱 효과적입니다. 다양한 해부학적 구조에 대한 고품질 3D 모델을 성공적으로 재구성하고 있으며, 임상적 진단 및 개인 맞춤형 치료 제공에 있어 그 중요성이 강조됩니다. 또한, 다중 모드 통합 및 교차 모달 프레임워크에서의 최신 연구 방향도 주목받고 있습니다.



### Network-Level Vehicle Delay Estimation at Heterogeneous Signalized Intersections (https://arxiv.org/abs/2510.01292)
Comments:
          arXiv admin note: text overlap with arXiv:2503.20113

- **What's New**: 이 연구에서는 다양한 교차로에서 차량 지연(time delay) 추정을 위한 도메인 적응(domain adaptation) 프레임워크를 도입했습니다. 기존 모델의 일반화 오류를 해결하기 위해 소스(source) 및 타겟(target) 도메인으로 데이터를 분리하고, 타겟 도메인에서 소량의 레이블(label)된 데이터로 모델을 미세 조정합니다. 이는 머신 러닝(machine learning) 기법을 실제 교통 시스템에 보다 널리 적용할 수 있는 기반을 제공합니다.

- **Technical Details**: 제안된 모델인 Gradient Boosting with Balanced Weighting (GBBW)는 타겟 도메인과의 유사성에 따라 소스 데이터를 재가중치하여 적응도를 향상시킵니다. 연구는 Arizona주 Pima County의 57개의 이질 이질 교차로(intersections)에서 수집된 데이터를 통해 프레임워크의 유효성을 검증하였고, 주요 교통 특징(traffic features)을 추출하여 모델 성능을 개선했습니다. 또한, 8개의 최첨단 ML 회귀(regression) 모델과 7개의 사례 기반(instance-based) DA 방법과 비교하여 성능을 평가했습니다.

- **Performance Highlights**: GBBW 프레임워크는 보다 정확하고 견고한 지연 추정치를 제공하는 것으로 나타났습니다. 이 접근법은 교통 신호 최적화(traffic signal optimization), 혼잡 관리(congestion management), 성능 기반(planning) 계획을 지원합니다. 결과적으로, 모델의 전이 가능성(transferability)이 향상되어 실제 교통 시스템에 머신 러닝 기술의 광범위한 배치를 촉진합니다.



### ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models (https://arxiv.org/abs/2510.01290)
- **What's New**: 이번 연구에서는 Large Reasoning Models(LRM)의 장기적 추론을 지원하기 위해 ThinKV라는 새로운 KV 캐시 압축 프레임워크를 제안합니다. ThinKV는 추론 과정에서 토큰의 중요도를 기반으로 하여 압축을 진행하며, 자원의 효율적인 재사용을 가능하게 합니다. 이 방법은 다른 압축 기법보다 더욱 효과적으로 메모리 사용을 줄일 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: ThinKV는 두 가지 주요 전략인 양자화(Quantization)와 퇴출(Eviction)을 채택하여 KV 캐시의 압축을 최적화합니다. 이는 각 토큰의 중요도에 따라 압축의 정밀도를 조정하고, 추론 동적인 변화를 고려하여 덜 중요한 토큰을 점진적으로 제거하는 방식입니다. 또한, PagedAttention을 확장한 커널을 설계하여 제거된 토큰의 메모리 슬롯을 효율적으로 재사용합니다.

- **Performance Highlights**: 실험 결과, ThinKV는 DeepSeek-R1-Distill, GPT-OSS 및 NVIDIA AceReason과 같은 다양한 벤치마크에서 기존 방법들보다 높은 성능을 보여줍니다. 특히, ThinKV는 원본 KV 캐시의 5% 이하로 근접한 정확도를 유지하면서도, 최대 5.8배 더 높은 추론 처리량을 달성했습니다. 이는 LRM의 효율성을 크게 향상시키는 결과입니다.



### Microsaccade-Inspired Probing: Positional Encoding Perturbations Reveal LLM Misbehaviours (https://arxiv.org/abs/2510.01288)
Comments:
          9 main pages, 13 appendix pages

- **What's New**: 이 논문은 LLM(대규모 언어 모델)의 잘못된 동작을 탐지하기 위한 새로운 방법인 MIP(Microsaccade-Inspired Probing)를 제안합니다. microsaccades(미세안구떨림)에서 영감을 받아, 이 방법은 모델의 내부 신호를 통해 잘못된 동작을 탐지합니다. 특히, 주목할만한 점은 이 방법이 특정 작업에 대한 세부 조정이 필요 없고, 다양한 상황에서 실패를 감지할 수 있다는 것입니다.

- **Technical Details**: MIP 방법은 가벼운 위치 인코딩의 교란을 통해 LLM의 잠재적 신호를 드러내는 방식으로 작동합니다. 위치 인코딩은 주로 토큰 순서를 부여하지만, 모델의 내부 표현과 상호작용하여 고차원의 의미 및 행동 패턴을 반영합니다. 이 연구는 LLM이 자신의 잘못된 동작을 인식할 수 있는 지식을 본질적으로 인코딩하고 있음을 보여줍니다.

- **Performance Highlights**: 실험에 따르면 MIP는 여러 최첨단 LLM에서 잘못된 동작을 효과적으로 드러내면서도 계산 효율성이 뛰어난 것으로 나타났습니다. 이러한 발견은 사전 훈련된 LLM이 자체 실패를 플래그할 수 있는 내부 근거를 이미 포함하고 있음을 시사합니다. 이 연구는 앞으로 LLM의 비정상적인 행동을 탐지하고 완화하는 새로운 길을 제시합니다.



### Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning (https://arxiv.org/abs/2510.01278)
- **What's New**: 본 논문은 Positive-Unlabeled (PU) 학습의 한계를 극복하기 위해 새로운 비대비(non-contrastive) PU 학습 프레임워크인 NcPU를 제안합니다. NcPU는 noisy-pair robust supervised non-contrastive loss (NoiSNCL)와 phantom label disambiguation (PLD) 기법을 결합하여 신뢰할 수 없는 감독 하에서도 효과적인 이내 클래스(intra-class) 표현 정렬을 가능하게 합니다. 이 프레임워크는 외부 부정 샘플이나 사전 추정된 매개변수 없이도 높은 성능 향상을 달성할 수 있음을 입증합니다.

- **Technical Details**: NcPU 프레임워크는 두 가지 핵심 구성 요소인 NoiSNCL과 PLD를 포함하고 있습니다. NoiSNCL은 부정확한 감독 하에서도 강력한 표현을 학습하도록 설계되었으며, PLD는 이러한 표현 학습을 통해 더 신뢰할 수 있는 감독을 제공하여 모델의 전반적인 성능을 향상시킵니다. 이 접근 방식은 Expectation-Maximization (EM) 프레임워크를 기반으로 하여 각 단계에서 서로를 이롭게 하는 이론적 근거를 가지고 있습니다.

- **Performance Highlights**: NcPU는 다양한 데이터셋에서 최신 PU 방법들보다 더 나은 성능을 보이며, 특히 재해 후 건물 피해 지도화와 같은 복잡한 작업에서도 뛰어난 성능을 입증했습니다. extensive experiments를 통해 NoiSNCL이 간단한 PU 방법들이 경쟁력 있는 성능을 발휘하는 데 기여함을 보여주었습니다. 또한, 코드 공개 예정으로, 이 연구는 실제 세계의 다양한 응용에서 높은 잠재력을 나타냅니다.



### Identifying Information-Transfer Nodes in a Recurrent Neural Network Reveals Dynamic Representations (https://arxiv.org/abs/2510.01271)
- **What's New**: 이번 연구에서는 순환 신경망(Recurrent Neural Networks, RNNs)의 내부 역학을 이해하기 위한 혁신적인 정보 이론적 방법을 도입하여 정보 전송 노드, 즉 정보 중계기(information relays)를 식별하고 분석합니다. 이 방법론은 입력과 출력 벡터 간의 상호 정보(mutual information)를 정량화하여 RNN이 작동하는 동안 정보가 흐르는 중요한 경로를 정확히 찾아내는 데 중점을 둡니다. 연구자들은 이 방식을 합성 데이터와 실세계 시계열(classification) 분류 작업에 적용하여 RNN 구조에 따라 뚜렷한 정보 전달 패턴을 밝혀냈습니다.

- **Technical Details**: 이 연구의 방법론은 RNN과 같은 인공 신경망의 각층이 정보 이론적 채널로 작용한다는 개념에 기초하고 있습니다. 각 층의 출력은 입력 벡터와 가중치 행렬을 사용하여 계산되며, 정보의 흐름은 주로 상호 정보로 정량화됩니다. 추가적으로, 제안된 알고리즘은 고유한 노드를 제거하면서 정보 전달 기능의 증가하는 정도를 가지고 노드를 정렬하는 방법을 사용해, 효과적으로 노드 순서를 최적화합니다.

- **Performance Highlights**: 연구 결과는 RNN의 복잡한 메커니즘을 이해하고 보다 견고하고 해석 가능한 신경망 설계에 기여하는 데 중요한 통찰력을 제공합니다. 특히, 정보 중계기 방법론은 LSTM 및 GRU와 같은 다양한 RNN 아키텍처에 적용되었으며, 이러한 아키텍처 간의 정보 처리 및 유지 방식의 차이를 드러내었습니다. 이 방법은 인공지능의 설명 가능성(explainable AI) 프로젝트에 중요한 기여를 하며, 구체적인 노드가 전체 네트워크 동작에 미치는 영향을 해명하는 데 도움을 줍니다.



### Safe Reinforcement Learning-Based Vibration Control: Overcoming Training Risks with LQR Guidanc (https://arxiv.org/abs/2510.01269)
Comments:
          Paper accepted for presentation at ICCMS 2025. The submission includes 10 pages and 6 figures

- **What's New**: 본 논문에서는 외부 자극에 의한 구조 진동의 제어에 있어, 기존의 LQR(Linear Quadratic Regulator) 접근법과 RL(Reinforcement Learning) 방법의 조합을 제안하고 있습니다. RL 기반의 진동 제어는 물리적 시스템에서 직접 학습하여 모델 프리(model-free) 특성을 유지하려고 하지만, 초기 학습 단계에서 불규칙한 제어 입력이 구조물에 피해를 줄 수 있는 위험이 있습니다. 이 문제를 해결하기 위해, LQR 컨트롤러의 지침을 활용한 하이브리드 제어 프레임워크를 제시하고 있습니다.

- **Technical Details**: 하이브리드 제어 프레임워크에서는 LQR 정책이 임의로 선택한 모델로부터 도출됩니다. 이 LQR 정책은 실제 또는 근사된 구조 모델에 대한 정보 없이도 작동이 가능하여, 전체 프레임워크 또한 모델 프리 상태를 유지합니다. 이 방법은 RL의 탐색 위험을 줄이고, 정확한 시스템 모델에 대한 의존성을 제거하는 장점을 가지고 있습니다.

- **Performance Highlights**: LQR-Guided RL 접근법은 RL 기반의 진동 제어에서 발견된 중요한 훈련 안전 문제를 해결하며, 수치 사례 연구를 통해 그 효과를 입증합니다. 연구 결과, RL 제어기와 LQR의 결합이 단독 RL 제어기보다 더 나은 성능을 발휘한다는 것을 보여줍니다. 이러한 접근 방식은 실제 물리적 시스템에서의 적용 가능성을 높이며, 산업 및 건설 분야에서 안전성을 크게 향상시킬 수 있을 것으로 기대됩니다.



### RLP: Reinforcement as a Pretraining Objectiv (https://arxiv.org/abs/2510.01265)
Comments:
          RLP introduces a new paradigm for RL-based Pretraining

- **What's New**: 최근 발표된 RLP(Reinforcement Learning Pre-training)는 전통적인 훈련 방법과 비교해 Chain-of-Thought(사고의 연쇄)를 예측의 사전 행동으로 취급하여 정보를 기반으로 한 강화학습 목표를 설정합니다. 이는 모델이 다음 토큰을 예측하기 전에 스스로 사고하도록 유도하여 사전 훈련의 초기 단계에서 독립적인 사고 행동을 가르치는 데 초점을 맞춥니다. 연구에서는 비검증 방식으로 로깅 가능성을 기반으로 한 보상 신호를 제공하여 훈련 효율성을 높이고 있습니다.

- **Technical Details**: RLP는 Chain-of-Thought를 생성하는 것을 통해 각 다음 토큰을 예측하기 전에 사고를 수행하도록 구조화되어 있으며, 사고가 다음 토큰 예측에 미치는 영향을 로그 가능성 비율로 측정합니다. 이 과정은 비검증적이며 밀집 보상을 제공하여 자연어 데이터에서 일반화 가능성을 높입니다. RLP는 기존의 강화학습 접근 방식에서 본질적인 한계를 극복하도록 밀접하게 설계되어 있습니다.

- **Performance Highlights**: RLP를 통해 Qwen3-1.7B-Base 모델에서 19%의 성과 향상 효과가 나타났으며, AIME25 및 MMLU-Pro와 같은 복잡한 추론 작업에서 더욱 두드러진 결과를 보입니다. Nemotron-Nano-12B-v2 모델에 적용 시 42.81%에서 61.32%로 향상되어 과학적 추론에서 23%의 증가를 달성했습니다. 이는 다양한 아키텍처 및 모델 크기에서의 확장성을 잘 보여줍니다.



### A Framework for Scalable Heterogeneous Multi-Agent Adversarial Reinforcement Learning in IsaacLab (https://arxiv.org/abs/2510.01264)
Comments:
          8 page, 9 figures, code this https URL

- **What's New**: 이번 연구에서는 이사크랩(IssacLab) 프레임워크를 확장하여 고충실도 물리 시뮬레이션 환경에서의 적대적 다중 에이전트 강화학습(multi-agent reinforcement learning, MARL)을 지원합니다. 적대적 상호작용이 필수적인 실제 응용 프로그램을 위해 이질적인 에이전트를 가진 새로운 환경을 도입하며, 이론을 바탕으로 팀별 비평가(critic)를 사용하여 다양한 에이전트의 성능을 향상시킵니다. 이를 통해 경쟁 환경에서 효과적이고 견고한 정책을 훈련할 수 있는 기회를 제공합니다.

- **Technical Details**: 연구는 Heterogeneous Agent Reinforcement Learning 알고리즘을 확장하여 다중 팀의 적대적 훈련을 지원합니다. 이를 위해 팀별 비평가를 도입하여 경쟁적 제로섬(zero-sum) 환경에서의 보상 문제를 해결하였으며, 각 팀의 정책은 팀 특화 비평가의 도움을 받아 학습합니다. 또한, 커리큘럼 학습(curriculum learning)을 활용하여 점진적으로 어려워지는 과제에 대해 훈련이 이루어지도록 설계되었습니다.

- **Performance Highlights**: 여러 벤치마크 시나리오에 대한 실험을 통해 다중 에이전트 경쟁 환경에서 높은 시뮬레이션 실현성과 빠른 처리 능력을 입증했습니다. 제안된 프레임워크는 서로 다른 형태를 가진 에이전트 간의 적대적 플레이에서의 독특한 도전 과제를 강조하며, 고충실도 적대적 환경에서 MARL 알고리즘을 테스트하기 위한 새로운 벤치마크를 도입하여 미래 연구의 토대를 마련합니다.



### Budgeted Broadcast: An Activity-Dependent Pruning Rule for Neural Network Efficiency (https://arxiv.org/abs/2510.01263)
- **What's New**: 본 논문에서는 기존의 pruning 방법들이 손실에 미치는 영향을 기준으로 매개변수를 제거하는 방식을 넘어, Budgeted Broadcast (BB)라는 새로운 접근법을 제안합니다. BB는 각 유닛에 지역 트래픽 예산(local traffic budget)을 설정하여, 그 유닛의 장기 활성화 비율(on-rate)과 분기(fan-out)를 고려합니다. 이를 통해 매개변수의 선택성과 청중 간의 균형을 조절하며, 간단한 로컬 액추에이터를 통해 더 효율적인 pruning을 가능하게 합니다.

- **Technical Details**: BB는 제약된 엔트로피 분석(constrained-entropy analysis)을 통해, 전역 트래픽 예산(global traffic budget) 아래에서 코딩 엔트로피(coding entropy)를 극대화합니다. 이 방법은 활동을 낮추기 위해 들어오는 연결(fan-in)을, 방송을 줄이기 위해 나가는 연결(fan-out)을 조절하여 엔트로피 균형을 유지합니다. 이러한 기법은 Transformers, ResNets, 3D U-Nets와 같은 다양한 신경망 아키텍처에서 적용될 수 있습니다.

- **Performance Highlights**: BB를 적용하면 코딩 엔트로피가 증가하고 상관관계가 감소하여, 일치하는 스파시티(sparsity)에서 정확도가 향상됩니다. 전자 현미경 이미지 분석에서는, 이 방법이 최신 F1 및 PR-AUC 성능을 달성하며, 기존 밀집 모델(dense baselines)을 초과하는 경우도 발생합니다. 또한, BB는 통합이 용이하고 더욱 다양하고 효율적인 표현 학습을 향한 경로를 제시합니다.



### RSTGCN: Railway-centric Spatio-Temporal Graph Convolutional Network for Train Delay Prediction (https://arxiv.org/abs/2510.01262)
- **What's New**: 본 논문에서는 인도 철도망(Indian Railway Network, IRN)의 평균 도착 지연(average arrival delays) 예측을 위해 Railway-centric Spatio-Temporal Graph Convolutional Network (RSTGCN)을 제안합니다. 본 연구는 4,735개의 역을 아우르는 데이터를 수집하고, 이를 통해 효율적인 기차 운영을 위한 새로운 예측 모델을 제시하여, 더 높은 수준의 교통 관리에 기여하고자 합니다. RSTGCN은 기차 빈도 인식을 통한 공간적 주의(spatial attention) 메커니즘을 통합하여 예측 성능을 크게 향상시킵니다.

- **Technical Details**: RSTGCN 모델은 그래프 신경망(Graph Neural Network) 구조를 기반으로 하여, 각 역의 평균 도착 지연을 시간 단위로 예측합니다. 이를 위해 모델은 시간 간격 내의 기차 빈도와 평균, 총 도착 및 출발 지연을 특징으로 하는 여러 도메인 정보 기반 기능(feature)들을 통합합니다. 또한, 이 모델은 실제 대규모 시간 데이터를 활용하여 평가되며 여러 최신 모델보다 우수한 성능을 보입니다.

- **Performance Highlights**: 실험 결과, RSTGCN은 기존의 여러 기법들과 비교하여 표준 지표에서 지속적으로 개선된 성과를 나타냈습니다. 본 연구에서 제안한 데이터셋은 인도 철도망 전역을 대상으로 하며, 이는 철도 지연 예측에 있어 획기적인 기여로 평가받고 있습니다. 이 데이터셋은 연구 목적으로 공개될 예정이며, 향후 관련 연구를 촉진하는 데 기여할 것으로 기대됩니다.



### Adaptive Federated Learning Defences via Trust-Aware Deep Q-Networks (https://arxiv.org/abs/2510.01261)
Comments:
          16 pages, 10 figures

- **What's New**: 본 연구에서는 Federated Learning에서의 방어를 부분 관측 가능(semi-observable) 표적 결정 문제로 설정하고, 다중 신호 증거를 고객 신뢰 업데이트에 통합하며 장기적인 견고성-정확도 목표를 최적화하는 신뢰 기반 Deep Q-Network(DQN)를 도입합니다. 기존의 방어 방법론이 가지는 결점을 해결하기 위해 다양한 신호와 증거를 통합하여 고객 기여도를 필터링하고 가중치를 조정함으로써 성능을 높였습니다.

- **Technical Details**: 연구는 DQN을 활용하여 신뢰 기반의 강화 학습 프레임워크를 설계하였습니다. 이를 위해 이상 탐지(anomaly detection) 메트릭스, Bayesian 신뢰 추적(Bayesian trust tracking), 그리고 적응적 집계 전략을 학습할 수 있는 DQN 공통 체계를 결합하였습니다. 이상 감지 메트릭스는 방향 정렬(directional alignment), 크기 편차(magnitude deviation), 유효성 영향(validation impact) 세 가지 요소로 구성되어 고객 행동을 종합적으로 요약합니다.

- **Performance Highlights**: CIFAR-10 데이터셋을 사용하여 방어 방법론의 성능을 평가하였으며, 고객 간 겹침이 증가할수록 정확도가 일관되게 향상되고 ASR(Attack Success Rate)이 감소하는 결과를 보였습니다. 신뢰 업데이트를 통해 상대적으로 약한 신호를 완화하면 정확도가 지속적으로 안정화되며 DQN이 다른 방어 방법들과 비교하여 최고의 견고성-정확도 균형을 이루는 것으로 나타났습니다.



### RSAVQ: Riemannian Sensitivity-Aware Vector Quantization for Large Language Models (https://arxiv.org/abs/2510.01240)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 극저비트 양자화를 극대화하기 위해 새로운 VQ 프레임워크인 RSAVQ를 제안합니다. RSAVQ는 오류 방향 감도 유도(Error Direction Sensitivity Guidance, EDSG)와 가중치 채널 감도 유도(Weight Channel Sensitivity Guidance, WCSG)의 두 가지 기하학적 혁신을 도입하여 기존의 문제들을 해결합니다. 이 프레임워크는 정보를 기하학적으로 모델링하여 양자화 정확도를 향상시킵니다.

- **Technical Details**: RSAVQ는 피셔 정보 행렬(Fisher Information Matrix, FIM)을 활용하여 LLM의 파라미터 공간을 비균일 곡률을 가진 리만 다양체로 모델링합니다. EDSG는 양자화 오류를 낮은 감도 방향으로 투영하여 모델 성능에 미치는 부정적 영향을 최소화하며, WCSG는 각 채널의 감도를 동적으로 할당하여 비트 자원을 효율적으로 분배합니다. 이러한 접근법은 주어진 비트 제약 내에서 최적의 양자화 솔루션을 제공합니다.

- **Performance Highlights**: 실험 결과, RSAVQ는 LLaMA-3 8B 모델에서 2비트 양자화 시 기존 방법인 VPTQ 및 QuIP#보다 0.4의 PPL과 1.5의 제로샷 정확도로 성능이 향상되었습니다. 이 논문은 제약된 환경에서의 실용적 해결책을 제시하며, 정보 기하학과 신경망의 양자화 사이의 이론적 다리를 제공합니다. RSAVQ는 극저비트 시나리오에서 우수한 성능을 입증하였으며, 향후 LLM의 효율적인 학습에 기여할 것으로 기대됩니다.



### Automated Extraction of Material Properties using LLM-based AI Agents (https://arxiv.org/abs/2510.01235)
- **What's New**: 최근 제안된 연구는 약 10,000개의 전체 텍스트 과학 기사를 활용하여 열전 소재의 성능 지표와 구조적 속성을 자동으로 추출하는 LLM(large language model) 기반의 워크플로우를 소개합니다. 이 시스템은 높은 정확도를 유지하면서 컴퓨팅 비용을 균형 있게 조절할 수 있도록 동적 토큰 할당 및 조건부 테이블 파싱을 통합합니다. 이 연구 결과로부터 27,822개의 온도에 따른 물리적 속성 기록이 축적되었으며, 이것은 열전 소재 발견의 새로운 가능성을 제시합니다.

- **Technical Details**: 이 연구는 다양한 기계 학습 기법을 활용하여 주요 과학 저널에서 연구 기사들로부터 데이터를 수집했습니다. DOI를 통해 관련 기사를 검색하고 XML 또는 HTML 형식으로 데이터를 다운로드하여 처리하는 자동화된 파이프라인을 구축했습니다. 특히 '마무리', '참고문헌'과 같은 비관련 부분을 제거하고, 열전 속성 관련 문장만을 남기는 필터링 과정을 통해 데이터의 정확성을 높였습니다.

- **Performance Highlights**: GPT-4.1 모델을 사용한 검증 결과, 열전 특성에 대한 F1 점수는 0.91로 최고치를 기록하였으며, GPT-4.1 Mini 모델도 거의 유사한 성능을 보였습니다. 이러한 시스템은 높은 수준의 정확도로 원자료에서 열전 데이터의 표준화된 기록을 생성할 수 있도록 하여, 대규모 데이터 기반의 소재 발견을 위한 기초를 마련하였습니다. 또한, 커뮤니티 접근을 용이하게 하기 위해 인터랙티브 웹 탐색기를 출시하여 사용자들이 데이터를 조회하고 CSV로 내보낼 수 있도록 지원합니다.



### Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs (https://arxiv.org/abs/2510.01218)
Comments:
          Second Conference on Language Modeling, 2025

- **What's New**: 이번 연구에서는 언어 모델에서 생성된 출력의 창의성을 평가하는 데 필수적인 다양성이 중요하다는 점을 강조합니다. 기존의 온도 기반 샘플링 기법들은 창의성을 증가시키지만, 수학적 추론과 같은 높은 정밀도가 요구되는 작업에서는 오히려 reasoning의 질을 저하시킬 수 있습니다. 저자들은 'selective sampling'이라는 새로운 기법을 제안하여, 샘플링 위험 메트릭에 따라 greedy 샘플링과 높은 온도 샘플링 사이를 동적으로 전환합니다.

- **Technical Details**: selective sampling은 샘플링의 오류 가능성을 예측하기 위해 가벼운 분류기를 훈련시키며, 이는 백본 언어 모델에 최소한의 지연(latency)을 추가하는 방식으로 통합됩니다. 이 접근법은 생성된 원래 모델 출력을 보존하는 동시에 구현이 용이한 특징을 가집니다. 다양한 샘플링 기법들을 분석한 결과, selective sampling이 일반적으로 사용되는 절단 및 엔트로피 기반 샘플링 기법들보다 품질-다양성 균형에서 더 나은 성능을 발휘함을 입증했습니다.

- **Performance Highlights**: 수학적 추론(task) 실험에서는 selective sampling이 고온(high-temperature) 환경에서도 품질-다양성 거래에서 우수한 성능을 보여주었습니다. 저자들은 기법의 효과를 실험을 통해 확인하였고, 기존의 샘플링 방법들이 실패하는 지점을 분석함으로써 향후 연구 방향을 제시하고 있습니다. 이 연구는 LLM의 출력에서 품질과 다양성을 유지하는 것이 중요함을 강조하며, 적응 가능한 디코딩 전략의 필요성을 역설합니다.



### Accelerating Long-Term Molecular Dynamics with Physics-Informed Time-Series Forecasting (https://arxiv.org/abs/2510.01206)
Comments:
          16 pages, preprint

- **What's New**: 본 연구에서는 효율적인 분자 동역학(Molecular Dynamics, MD) 시뮬레이션을 위한 새로운 접근 방식을 제안합니다. 이는 MD 시뮬레이션을 시계열 예측(time-series forecasting) 문제로 공식화하여 원자 궤적(predicted atomic trajectory)을 예측하는 고급 모델을 활용합니다. 기존의 DFT 기반 방법 대신 물리학적 지식을 통합한 손실 및 추론 메커니즘을 도입하여 비정상 원자 간 위치를 제어하고 안정성을 높였습니다.

- **Technical Details**: 고전적인 MD는 뉴턴의 운동 방정식을 수치적으로 통합하여 원자 간 상호작용을 탐구하는 기본 도구입니다. 연구에서는 PhysTimeMD라는 시계열 예측 프레임워크를 도입하여 과거 원자 위치를 이동량(displacement vectors)으로 변환하고 미래 동작을 예측합니다. 또한, 모어(Morse) 포텐셜을 기반으로 한 최종 보정 단계를 통해 원자 간 물리적 상호작용을 준수하는 시뮬레이션을 수행합니다.

- **Performance Highlights**: 이 방법은 다양한 물질에 대해 높은 정확도를 유지하며, 수천 단계의 MD 스텝을 몇 분 내에 안정적으로 모델링할 수 있는 가능성을 보여줍니다. 기존의 ML 기반 MD 접근법보다 더 정확하고 효율적이며, 물리적 제약 조건을 통합하여 예측의 신뢰성과 물리적 일관성을 향상시킵니다. 즉, 본 연구는 MD 시뮬레이션의 정확도와 효율성을 동시에 개선하는 중요한 기여를 하고 있습니다.



### Inferring Dynamic Physical Properties from Video Foundation Models (https://arxiv.org/abs/2510.02311)
- **What's New**: 본 연구에서는 비디오로부터 동적인 물리적 특성을 예측하는 작업을 연구합니다. 특히 튀는 물체의 탄성, 흐르는 액체의 점도, 표면에서 미끄러지는 물체의 동적 마찰 계수와 같은 여러 물리적 속성을 다룹니다. 이를 위해 새롭게 생성한 PhysVid라는 비디오 데이터셋을 도입하여 동적 물리 속성을 평가하고, 기존 데이터셋의 부족함을 보완하고자 합니다.

- **Technical Details**: PhysVid 데이터셋은 합성 비디오와 실제 비디오를 혼합하여 구성되며, 각각의 비디오는 물리적 속성 값으로 주석이 달려 있습니다. 연구진은 세 가지 방법을 사용하여 비디오로부터 물리적 특성을 추정하는 접근 방식(오라클 메서드, 간단한 읽기 메커니즘, Multi-modal Large Language Models)도 제안합니다. 오라클 메서드는 해당 속성을 직접 반영하는 시각적 단서를 사용하는 고급 기법입니다.

- **Performance Highlights**: 비디오 기반 모델들은 생성 및 자기 지도 방식으로 훈련된 모델들이 오라클 메서드보다는 성능이 떨어지지만, 유사한 수준의 성능을 달성했음을 보여줍니다. Multi-modal Large Language Models는 다른 모델들에 비해 현재 성능이 낮지만, 적절한 프로밍 방법을 통해 성능 향상이 가능함을 시사합니다. 이 연구는 물리적 이해도를 평가하는 데 있어 기존 연구의 한계를 넘어서기 위한 중요한 기초 자료가 될 것입니다.



### VideoNSA: Native Sparse Attention Scales Video Understanding (https://arxiv.org/abs/2510.02295)
Comments:
          Project Page: this https URL, Code: this https URL

- **What's New**: 이 논문에서는 비디오와 언어 모델을 위한 Native Sparse Attention (NSA)를 채택하여 VideoNSA라는 새로운 접근 방식을 제안합니다. 기존 비디오-언어 모델들의 맥락 길이가 제한된 문제를 해결하기 위해, 이 모델은 216K 비디오 지침 데이터셋에서 엔드-투-엔드(end-to-end) 훈련을 통해 Qwen2.5-VL을 조정합니다. 특히, 하드웨어에 최적화된 하이브리드 주의(attention) 접근 방식을 사용하여 텍스트에서는 밀집 주의를, 비디오에는 NSA를 적용합니다.

- **Technical Details**: VideoNSA는 세 가지 보완적인 캐시 분기를 통합한 배우는 하드웨어 인식 sparse attention 메커니즘을 사용합니다. 여기에는 Token Compression (CMP), Token Selection (SLC), Sliding Window (SWA) 분기가 포함됩니다. 이 방식은 모델이 특정 작업에 필요한 경로만을 유지하여 효율성을 극대화합니다. 특히, VideoNSA는 128K 컨텍스트 길이에서 성능 향상을 이루며, 작동 방식에서 여러 중요한 발견 사항들을 제시합니다.

- **Performance Highlights**: VideoNSA는 긴 비디오 이해(long-video understanding) 및 시간적 추론(temporal reasoning)에서 기존 방법보다 개선된 성능을 보였습니다. 또한, 다양한 실험 결과에 따라 데이터 세트의 길이를 초과하여 효과적으로 확장할 수 있는 잠재력을 보여줍니다. 가장 흥미로운 점은, 이 모델이 학습 가능한 sparse attention 가중치를 통해 다양한 태스크에서 동적인 주의 sink 행동을 유도하여 깊은 층에서의 선택 및 슬라이딩 윈도우 분기의 중요성을 감소시킨다는 것입니다.



### Learning to Generate Object Interactions with Physics-Guided Video Diffusion (https://arxiv.org/abs/2510.02284)
- **What's New**: KineMask는 물리 기반 비디오 생성 기능을 개선하기 위한 새로운 접근법으로, 강체체의 제어 및 상호작용을 현실감 있게 수행할 수 있도록 합니다. 이 방법은 단일 이미지와 특정 객체 속도를 제공받아 미래의 물체 상호작용을 예측하는 비디오를 생성합니다. 이 연구는 또한 저수준 모션 제어와 고수준 텍스트 조건화를 통합하여 복잡한 동적 현상을 효과적으로 합성할 수 있도록 합니다.

- **Technical Details**: KineMask는 물리적으로 유도된 비디오 생성을 위한 프레임워크로, 객체의 방향과 속도 같은 저수준 운동 제어를 제공합니다. 이를 통해 모델은 객체 상호작용을 추론하며, 두 단계의 훈련 전략을 사용하여 객체 마스크를 통해 미래의 모션 감독을 점진적으로 제거합니다. KineMask는 시뮬레이터에서 생성된 비디오를 기반으로 훈련되며 물리적으로 유효한 역학과 명확한 객체 상호작용을 캡처합니다.

- **Performance Highlights**: KineMask는 기존의 동영상 확산 모델들과 비교하여 뛰어난 객체 상호작용을 보여줍니다. 광범위한 실험 결과, KineMask는 유사한 크기의 최신 모델들 대비 강력한 개선을 이끌어냈고, ablation 연구를 통해 저수준과 고수준 제어의 통합 중요성을 강조했습니다. 추가적으로, KineMask는 복잡한 상호작용의 일반화를 통해 실제 장면에서의 비디오 생성에서 두드러진 성능 향상을 보여줍니다.



### VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL (https://arxiv.org/abs/2510.02282)
- **What's New**: 새로운 AI-생성 비디오의 발전에 따라 VidGuard-R1이 등장했습니다. 이는 멀티모달 대형 언어 모델(MLLM)을 활용하여 비디오의 진위를 판별하는 최초의 도구입니다. 해당 모델은 그룹 상대 정책 최적화(GRPO)를 통해 학습하여, 비디오의 정확한 판단과 더불어 해석 가능한 설명을 제공할 수 있습니다.

- **Technical Details**: VidGuard-R1은 140,000개의 실제 및 AI-생성 비디오로 구성된 고난이도 데이터셋을 기반으로 하고 있습니다. 이 모델은 두 개의 보상 모델을 통해 시간 아티팩트와 생성 복잡성을 평가하여 GRPO로 세밀하게 조정됩니다. 다중 단계 확산을 통한 생성 비디오에서 고급 요인을 끌어내기를 목적으로 하며, 이는 더욱 향상된 설명 능력을 보장합니다.

- **Performance Highlights**: VidGuard-R1은 제로샷 성능에서 최첨단 정확도를 달성하며, 추가 훈련을 통해 95% 이상의 정확도를 기록했습니다. 다양한 실험과 사례 연구는 VidGuard-R1이 예측 뒤에 있는 정확하고 해석 가능한 이유를 제시할 수 있음을 보여줍니다. 이러한 성능은 AI-생성 비디오 감지의 새로운 기준을 설정하는 데 기여할 것입니다.



### Paving the Way Towards Kinematic Assessment Using Monocular Video: A Preclinical Benchmark of State-of-the-Art Deep-Learning-Based 3D Human Pose Estimators Against Inertial Sensors in Daily Living Activities (https://arxiv.org/abs/2510.02264)
Comments:
          All tables, graphs and figures generated can be obtained in the Zenodo repository complementary to this work: this https URL

- **What's New**: 이번 연구는 기계 학습(machine learning)과 착용 가능한 센서(wearable sensors)의 발전이 사람의 움직임을 전문 실험실 외부에서 캡처하고 분석할 수 있는 새롭고 유망한 기회를 제공한다고 소개합니다. VIDIMU 데이터셋을 활용하여, 일상적인 임상 관련 활동에서 비디오 카메라와 관성 측정 장치(inertial measurement units, IMUs)를 함께 사용하여 단일 비디오 기반으로 3D 인간 자세를 생성하는 모델을 비교하였습니다.

- **Technical Details**: 연구에서는 MotionAGFormer, MotionBERT, MMPose 2D-to-3D pose lifting과 NVIDIA BodyTrack와 같은 최신 심층학습(d deep learning) 프레임워크를 사용하여 도출된 관절 각도를 IMU 데이터로부터 계산된 관절 각도와 비교하였습니다. OpenSim의 역기구학(inverse kinematics) 기법을 사용하여 Human3.6M 데이터셋 형식에 따라 17개의 주요 관절 지점을 기반으로 분석을 진행했습니다. 이 초기 연구는 건강한 피실험자만을 대상으로 하여, 결과를 병리학적인 집단에 일반화할 수 없다는 한계가 있습니다.

- **Performance Highlights**: MotionAGFormer는 전체 RMSE(최소 제곱 오차) $9.27°c 4.80°$, MAE(평균 절대 오차) $7.86°c 4.18°$로 가장 우수한 성과를 보였으며, Pearson 상관계수($0.86 c 0.15$)와 결정계수($R^{2}$, $0.67 c 0.28$)에서도 가장 높은 점수를 기록하였습니다. 이 연구는 두 기술 모두 실외에서 생체역학적 평가(kinematic assessment)에 적합하다는 것을 보여주지만, 비용, 접근성, 정밀도 간의 주요 트레이드오프(trade-off)를 강조합니다. 또한 임상에서 유망한 생체역학적 데이터를 제공하는 비디오 모델과 IMU 기반 평가 사이의 차이를 규명하였습니다.



### RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems (https://arxiv.org/abs/2510.02263)
- **What's New**: 이 연구에서는 복잡한 문제에 대한 해답을 도출하기 위해 'algorithmic procedures'를 식별하고 구현하는 과정을 중심으로 새로운 접근법을 제안합니다. 특히, reasoning abstractions을 도입하여 모델이 수월하게 성공적인 추론을 배울 수 있도록 돕습니다. 이러한 방식은 모델이 문제를 해결하는 데 필요한 프로시저와 사실 지식에 대한 간결한 자연어 설명을 활용하도록 합니다.

- **Technical Details**: 새로운 RLAD(RL Abstraction and Decomposition) 방법론은 두 개의 역할을 플랫폼에서 활성화하여, 추상화를 생성하는 모델과 해결책을 생성하는 모델을 공동으로 학습시킵니다. 핵심 요소로는 curriculum training(커리큘럼 학습), 비추상화 프롬프트 포함, 그리고 보상 마스킹(reward masking) 기법이 포함되어 있습니다. 이러한 기법을 사용하면 모델의 성능이 향상되며, 고차원적인 프로시저 지식을 효과적으로 활용하여 더 어려운 문제에도 잘 일반화합니다.

- **Performance Highlights**: RLAD 방법은 AIME 2024 및 HMMT 2025의 두 수학 추론 벤치마크에서 기존 모델들보다 우수한 성능을 기록했습니다. 추상화를 기반으로 한 훈련이 더욱 좋은 일반화를 보여주며, 비추상화 프롬프트를 포함시키고 적절한 보상 마스킹을 통해 모델 성능을 크게 향상시켰습니다. 최종적으로, 커리큘럼 학습, 비추상화 프롬프트 포함, 보상 마스킹을 조합한 것이 다른 구성들보다 현저하게 우수한 결과를 가져왔습니다.



### DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing (https://arxiv.org/abs/2510.02253)
Comments:
          Preprint

- **What's New**: 이 논문에서는 DragFlow라는 새로운 드래그 기반 이미지 편집 프레임워크를 제안합니다. 이 프레임워크는 더 강력한 생성적 prior를 활용하여 편집 작업에서 현저한 성과를 보여줍니다. 이전의 UNet 기반 모델의 한계를 넘어, FLUX와 같은 새로운 모델들로부터의 이점을 최대한 활용합니다.

- **Technical Details**: DragFlow는 affine transformations를 통해 지역 기반 편집 패러다임을 도입하여 일관된 피쳐(supervision) 지침을 제공합니다. 또한, pretrained open-domain personalization adapters(예: IP-Adapter)를 통합하여 배경 품질을 유지하면서도 주제 일관성을 향상시킵니다. 다양한 작업의 모호성을 해결하기 위해 다중 모달 대형 언어 모델(MLLMs)을 사용합니다.

- **Performance Highlights**: DragFlow는 새로운 Region-based Dragging benchmark(ReD Bench)과 DragBench-DR에서 광범위한 실험을 통해 점 기반 및 지역 기반의 기존 기준을 뛰어넘는 성과를 보여줍니다. 이로 인해 드래그 기반 이미지 편집에서 새로운 최첨단(state-of-the-art)을 설정하였습니다. 코드와 데이터셋은 발표 시에 공개될 예정입니다.



### The Unreasonable Effectiveness of Scaling Agents for Computer Us (https://arxiv.org/abs/2510.02250)
Comments:
          23 pages, 7 figures, 10 tables

- **What's New**: 이번 연구에서는 컴퓨터 사용 에이전트(CUAs)의 넓은 스케일링을 위한 새로운 방법인 Behavior Best-of-N(bBoN)을 소개합니다. bBoN은 에이전트의 롤아웃을 생성하고 이를 비교하기 위해 행동 서사를 사용하여 여러 롤아웃 간의 선택을 가능하게 합니다. 이를 통해 저항력과 성공률이 크게 향상되었으며, 기존 방법들과 비교했을 때 놀라운 성능 개선을 달성했습니다.

- **Technical Details**: CUAs는 부분 가시성 마르코프 결정 과정(POMDP)으로 모델링되며, 상태 공간, 관찰 공간 그리고 행동 공간으로 구성됩니다. 본 연구는 다수의 기초 모델과 정책을 사용하여 후보 솔루션 경로의 수를 스케일링하고 최적의 솔루션 선택을 위한 효과적인 방법을 제안합니다. 이는 기존의 단계별 BoN 방법과는 달리, 여러 기본 에이전트에 의해 생성된 후보 경로 중에서 최상의 경로를 선택하는 접근 방식을 취합니다.

- **Performance Highlights**: bBoN 메서드는 OSWorld 벤치마크에서 69.9%의 성공률로 새로운 state of the art(SoTA)를 달성하였습니다. 이는 이전 최상의 59.9%를 크게 초과하였으며, 인간 수준의 성능에 가까운 72%를 근접하게 합니다. 또한, WindowsAgentArena 및 AndroidWorld에 대한 강력한 제로샷 일반화 결과를 보여주어, bBoN의 성능을 더욱 입증했습니다.



### Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation (https://arxiv.org/abs/2510.02249)
- **What's New**: 이번 연구는 대형 언어 모델(Large Language Models, LLMs)의 과도한 사고(overthinking) 문제를 해결하기 위해 새로운 지표인 Token Entropy Cumulative Average (TECA)를 도입합니다. TECA는 추론 과정에서의 탐색 정도를 측정하며, 이를 활용하여 모델이 최적의 시점에서 사고를 종료하도록 돕는 새로운 사고 패러다임인 'Explore Briefly, Then Decide'를 제안합니다. 이를 통해 모델은 기술적인 깊이를 복잡도에 맞춰 조정할 수 있는 능력을 확보하게 됩니다.

- **Technical Details**: TECA는 각각의 추론 단계에서 모델의 불확실성을 나타내는 토큰 엔트로피(token entropy)의 누적 평균을 계산합니다. 이를 기반으로 탐색(exploration) 단계와 결정(determination) 단계를 구분하여 과도한 탐색을 제어하는 Cumulative Entropy Regulation (CER) 메커니즘을 도입합니다. CER은 모델이 불필요하게 긴 사고 과정을 피하도록 돕는 동시에 필요한 탐색 능력을 유지하게 합니다.

- **Performance Highlights**: 제안한 방법을 통해 다양한 수학 문제 벤치마크에서 모델의 응답 길이가 평균 71%까지 감소하면서도 문제 해결 능력은 거의 유지되었음을 확인했습니다. 특히, Qwen3-4B 모델은 GSM8K에서 71%의 응답 길이 감소를, MATH500에선 39.25% 감소를 기록하며, 기존 방법들보다 전반적으로 우수한 성능을 나타냈습니다. 이러한 실험 결과는 TECA를 지표로 활용해 추론 과정을 적절히 조정함으로써 과도한 사고를 감소시킬 수 있음을 보여줍니다.



### More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration (https://arxiv.org/abs/2510.02227)
Comments:
          20 pages, 5 figures

- **What's New**: 이 논문은 Reinforcement Learning with Verifiable Rewards (RLVR) 기법을 사용하여 대형 언어 모델(LLM)의 추론 능력을 개선하는 새로운 패러다임인 Adaptive Multi-Guidance Policy Optimization (AMPO)를 소개합니다. 기존의 방법보다 더 다양한 탐색을 허용하고, 불필요한 개입 없이 자기 탐색의 가치를 보존하면서도 여러 능숙한 교사 모델의 지도를 적절히 활용합니다. 이를 통해 AMPO는 더 나은 효과성과 일반화를 제공하여 모델의 성능을 크게 향상시킵니다.

- **Technical Details**: AMPO는 혼합 정책 강화 학습(Mixed-Policy RL) 프레임워크이며, 여러 동료 모델의 집합적 지능을 활용합니다. 이 방법은 학생 모델이 문제를 해결하지 못할 경우에만 외부 지도를 제공하는 'guidance-on-demand' 원칙을 적용합니다. 또한, 학생이 이해하기 쉬운 추론 경로를 학습하도록 유도하는 이해 기반 지침 선택 메커니즘을 도입하여 대폭적인 탐색과 효과적인 활용 간의 균형을 맞춥니다.

- **Performance Highlights**: 실험 결과 AMPO는 강력한 기준선 모델인 GRPO보다 평균 4.3% 및 분포 외(out-of-distribution) 작업에서 12.2%의 개선을 보여주었습니다. 특히, 4명의 동료 크기 교사를 사용함으로써 하나의 강력한 교사를 사용하는 방법과 유사한 성능을 보이며, 더 적은 데이터로도 높은 성능을 기록했습니다. 이러한 결과는 AMPO가 탐색과 활용 간의 우수한 균형을 이루는 방법임을 보여줍니다.



### TempoControl: Temporal Attention Guidance for Text-to-Video Models (https://arxiv.org/abs/2510.02226)
Comments:
          Under Review

- **What's New**: 최근 생성 비디오 모델의 발전으로 자연어 프롬프트를 기반으로 한 고품질 비디오 생성이 가능해졌습니다. 하지만 이러한 모델은 개별 시각적 요소의 출현 시점을 지정할 수 있는 세분화된 시간 제어 부족이 있습니다. 이 연구에서는 TempoControl이라는 방법을 소개하여 추가 훈련이나 감독 없이 추론 시점에서 비주얼 개념을 시간적으로 정렬할 수 있도록 하였습니다.

- **Technical Details**: TempoControl은 텍스트-비디오 확산 모델의 주요 구성 요소 중 하나인 크로스 어텐션 맵을 활용하여 비주얼 컨셉의 타이밍을 유도하는 새로운 최적화 접근 방식을 적용합니다. 이 방법은 세 가지 보완 원칙(상관관계, 에너지, 엔트로피)을 통해 어텐션을 조정합니다. 또한, 모델 파라미터를 업데이트하지 않고 적절한 수준의 시간적 정렬을 달성하기 위해 몇 차례의 확률적 경량 하강법(SGD)을 적용합니다.

- **Performance Highlights**: TempoControl은 단일 및 다중 객체를 포함한 다양한 비디오 생성 응용 프로그램에서 효과성을 입증하였습니다. 특히, 객체의 시간 재배열 및 행동과 오디오 정렬 생성에서 뛰어난 성능을 보여주었습니다. 우리의 접근 방식은 추가 훈련 없이도 비디오 생성에서 외부 오디오 신호와의 정렬 가능성을 탐색할 수 있는 잠재력을 가지고 있습니다.



### Quantum Fisher information matrices from Rényi relative entropies (https://arxiv.org/abs/2510.02218)
Comments:
          94 pages, 2 figures, dedicated to Professor Fumio Hiai on the occasion of his forthcoming 80th birthday

- **What's New**: 이 논문에서는 양자 정보 과학에서 중요한 피셔 정보(Fisher information)의 양자 일반화에 대해 다룹니다. 저자는 로그-유클리드(log-Euclidean), 기하학적(geometric) 및 α-z 레뉴(Rényi) 상대 엔트로피에 기반한 정보 행렬을 도출합니다. 주목할 만한 점은, 이러한 정보 행렬이 데이터 처리 불평등(data-processing inequality)을 준수하며, 이는 비록 원래 수량들이 그렇게 하지 않을지라도 성립한다는 것입니다.

- **Technical Details**: 이 논문은 매트릭스 미분(mat2x derivatives) 계산을 위해 나누어진 차분(divided differences) 메소드를 사용하여 주로 중점을 두고 작업합니다. 저자는 다양한 조건에서 로그-유클리드 정보 행렬(log-Euclidean information matrix)과 기하학적 정보 행렬(geometric information matrix)의 동등성을 입증합니다. 또한, α-z 정보 행렬에 대한 공식을 도출하여 여러 매개변수를 갖는 경우에도 적용 가능하도록 합니다.

- **Performance Highlights**: 저자는 β(<0,1)이나 (+∞)의 모든 값에 대해 로그-유클리드 정보 행렬이 구보-모리(Kubo-Mori) 정보 행렬과 같다는 정리를 제시합니다. 이 연구는 양자 추정 이론(quantum estimation theory)에서 논의된 정보 행렬에 대한 기존 자료들과 연결될 수 있습니다. 최종적으로, 저자는 페츠(Petz) 및 샌드위치(sandwiched) 레뉴 정보 행렬의 순서 관계를 수립하며, 이는 알파(α) 매개변수에 대한 단조성을 통해 명확히 정의됩니다.



### Measurement-Guided Consistency Model Sampling for Inverse Problems (https://arxiv.org/abs/2510.02208)
Comments:
          5 pages, 3 figures, submitted to IEEE Signal Processing Letters

- **What's New**: 이번 연구에서는 inverse problem reconstruction을 위해 수정된 consistency sampling 접근 방식을 제안합니다. 이 접근법은 measurement-consistency 메커니즘에 의해 확률적 샘플링을 유도하여, 관측된 측정값에 대한 충실도를 유지하면서도 효율적인 consistency 기반 생성을 가능하게 합니다. Fashion-MNIST 및 LSUN Bedroom 데이터셋에서의 실험 결과는 퍼셉추얼 및 픽셀 수준의 메트릭이 기존 baseline consistency sampling에 비해 일관된 향상을 보이는 것을 보여줍니다.

- **Technical Details**: 이 연구는 consistency models (CMs)의 샘플링 과정에서의 문제를 해결하기 위해 measurement-driven 가이던스를 도입합니다. 이를 통해 샘플러는 재구성이 관측값에서 벗어날 때 탐색을 주입하고, 측정 충실도가 향상됨에 따라 안정적인 결정론적 업데이트를 제공합니다. CMs의 샘플링 과정에서 DDIM(denoising diffusion implicit model) 방식이 활용되며, 샘플에서 직접적인 측정 일관성을 통합하여 adaptation합니다.

- **Performance Highlights**: 실험 결과, 제안된 샘플링 방법은 FID(Fréchet Inception Distance) 및 KID(Kernel Inception Distance)와 같은 퍼셉추얼 메트릭과 PSNR(peak signal-to-noise ratio), SSIM(structural similarity index measure)와 같은 왜곡 메트릭에서 baseline CMs 샘플링 대비 향상된 성능을 보였습니다. 이 방법은 몇 단계의 샘플링만으로도 경쟁력 있는 재구성을 이루어내어, 시간에 민감한 또는 대규모 설정에서의 활용 가능성을 제시합니다.



### UpSafe$^\circ$C: Upcycling for Controllable Safety in Large Language Models (https://arxiv.org/abs/2510.02194)
- **What's New**: 이 연구에서는 기존의 LLM(대형 언어 모델) 안전성을 향상시키기 위해 UpSafe$^	heta$C라는 통합 프레임워크를 제안합니다. 이 방법은 안전을 중시한 업사이클링(safety-aware upcycling) 접근 방식을 사용하여 안전에 치명적인 레이어를 식별하고 이를 희소 Mixture-of-Experts (MoE) 구조로 변환합니다. 피험자 데이터 기초의 두 단계 SFT(단기 강화 학습)을 도입하여 안전성 분별력을 강화하면서 모델의 일반적 능력을 유지합니다.

- **Technical Details**: UpSafe$^	heta$C 프레임워크는 안전-critical layers를 찾아내고 이를 MoE 구조로 업사이클링하여, 라우터가 원래 MLP(다층 퍼셉트론)와 안전 전문가를 선택적으로 활성화하는 소프트 가드레일 역할을 합니다. 안전 온도(safety temperature) 메커니즘을 도입하여 추론 시점에서 안전과 효용 간의 균형을 동적으로 조정할 수 있는 유연한 제어를 가능하게 합니다. 두 단계의 SFT 전략을 사용하여 안전성을 더욱 강화하면서도 모델의 효용을 유지합니다.

- **Performance Highlights**: 실험 결과 UpSafe$^	heta$C는 유해 입력 및 탈출 공격에 대해 견고한 안전성 향상을 달성하면서도 일반 작업에서 경쟁력 있는 성능을 유지합니다. 안전 온도 메커니즘을 통해 추론 시점에서 세밀한 조정이 가능하며, 이는 안전과 유틸리티 간의 최적 경계(Pareto-optimal frontier)를 달성하는 데 기여합니다. 전반적으로, UpSafe$^	heta$C는 고정된 정렬(static alignment) 방식에서 동적이고 모듈화된 제어(dynamic, modular control)로의 전환을 통해 LLM 안전성의 새로운 방향성을 제시합니다.



### Hybrid Physics-ML Framework for Pan-Arctic Permafrost Infrastructure Risk at Record 2.9-Million Observation Sca (https://arxiv.org/abs/2510.02189)
Comments:
          14 pages, 9 figures

- **What's New**: 본 연구는 북극의 영구동토(Permafrost) 상황을 평가하는 새로운 하이브리드 물리학-기계 학습 프레임워크를 제시합니다. 2005년부터 2021년까지 수집된 2.9백만 건의 관측치를 사용하여, 영구동토와 기후 간의 관계를 고려하여 기후 시나리오를 기반으로 한 예측 모델을 개발했습니다. 이 프레임워크는 기존의 위험 평가 도구들에서 부족했던 공간 및 시간 검증과 불확실성 정량화를 통합하여 운영 결정을 지원할 수 있습니다.

- **Technical Details**: 연구는 171,605개의 지점에서 영구동토와 기후 재분석 데이터(Climate Reanalysis)를 결합하여 데이터를 수집했습니다. 수집된 데이터를 기반으로 하여, Random Forest, Histogram Gradient Boosting 및 Elastic Net을 결합한 스택 앙상블 모델을 사용하여 R2=0.980의 성능을 보였습니다. 또한, 공간 및 시간 교차 검증을 통해 데이터 간섭을 방지하며, 이론적 기초를 적용하여 기계 학습의 한계를 극복하는 방법도 제시했습니다.

- **Performance Highlights**: 모델은 RCP8.5 강제 하에서 영구동토 비율이 평균 -20.3pp 감소할 것으로 예상하며, 51.5%의 북극 러시아 지역이 20 pp 이상의 손실을 겪을 것으로 관측됩니다. 위험 분류를 통해 15%를 높은 위험군으로, 25%를 중간 위험군으로 구분하였고, 이를 바탕으로 공간적으로 명확한 불확실성 지도가 제공됩니다. 이는 세계적으로 가장 큰 검증된 영구동토 ML 데이터셋을 구축하였으며, 북극 인프라의 위험 평가를 위한 최초의 운영 가능한 하이브리드 예측 시스템을 제공합니다.



### High-Fidelity Speech Enhancement via Discrete Audio Tokens (https://arxiv.org/abs/2510.02187)
- **What's New**: 최근의 autoregressive transformer 기반 음성 향상(Speech Enhancement, SE) 방법들은 고급 의미 이해 및 맥락 모델링을 활용하여 유망한 결과를 보여줍니다. 그러나 이러한 접근 방식은 종종 복잡한 다단계 파이프라인과 낮은 샘플링 속도의 코덱에 의존하여 범위가 좁고 특정 작업에 한정되는 경우가 많습니다. 본 연구에서는 DAC-SE1이라는 단순한 언어 모델 기반의 SE 프레임워크를 소개하며, 이는 이산 고해상도 오디오 표현을 활용하여 미세한 음향 세부 정보를 보존하면서 의미적 일관성을 유지합니다.

- **Technical Details**: 우리는 44.1 kHz에서 DAC 코덱을 사용하여 오디오를 9개의 잔여 코덱으로 인코딩하며, 각 코덱은 1024개의 코드로 구성되어 있습니다. 기존의 여러 코드북을 별도로 처리하고 나중에 임베딩을 집계하는 방식 대신, 모든 코드북을 단일 토큰 시퀀스로 평탄화함으로써 설계를 단순화했습니다. DAC 토큰에서 파생된 장기 문맥을 효과적으로 처리하기 위해 로테리 위치 임베딩(RoPE)을 사용하며, 이를 통해 구조의 안정성과 일반화를 크게 향상시킵니다.

- **Performance Highlights**: DAC-SE1은 객관적 지표와 MUSHRA 인간 평가에서 기존의 최첨단 autoregressive SE 방법들을 초월하는 성능을 보여줍니다. 우리의 모델은 1B 파라미터로 구동되는 단일 단계의 LM 기반 SE 프레임워크로, 높은 신뢰성을 갖춘 음성 향상과 대역폭 확장을 달성합니다. 모델과 훈련 파이프라인을 공개하여 확장 가능한 고품질 음성 향상에 대한 추가 연구를 촉진하고자 합니다.



### GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation (https://arxiv.org/abs/2510.02186)
- **What's New**: 본 논문에서는 2D Vision-Language Models (VLMs)에서 3D semantic segmentation으로 특징을 전달할 때의 지속적인 문제를 다루고 있는 GeoPurify를 제안합니다. 이 모델은 2D에서 3D로 전이하는 과정에서 발생하는 기하학적 불일치를 해결하며, 필요한 대량의 주석이 있는 3D 데이터 없이 학습할 수 있습니다. 특히, 2D 특징에서 잠재적인 기하학적 정보를 활용하여 향상된 성능과 데이터 효율성을 달성하는 방법을 모색합니다.

- **Technical Details**: GeoPurify는 Geometric Contrastive Distillation이라는 새로운 프레임워크를 기반으로 하며, 이를 통해 학생 모델이 3D 지도 모델의 구조적 우선 정보를 활용하여 잡음이 많은 특징에서 잠재적인 기하학적 친화도를 학습합니다. 또한, Geometry-Guided Pooling 모듈을 통해 3D 포인트 클라우드를 더욱 정돈하여 일관된 표현을 생성합니다. GeoPurify는 기존 대량 수동 주석 없이도 소규모 비유형 3D 스캔 데이터로 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, GeoPurify는 주요 3D 벤치마크에서 기존 최첨단 방법들과 비교하여 동등한 또는 더 나은 성능을 달성했습니다. 특기할 만한 점은, 약 1.5%의 훈련 데이터만 사용하면서도 이러한 성능을 낼 수 있다는 점입니다. 이는 데이터 효율성 측면에서 큰 개선을 보여줍니다.



### Uncovering Semantic Selectivity of Latent Groups in Higher Visual Cortex with Mutual Information-Guided Diffusion (https://arxiv.org/abs/2510.02182)
- **What's New**: 이번 연구에서는 MIG-Vis라는 새로운 방법을 제안하여 고차 시각 영역의 신경 집단이 인코딩하는 시각-의미적 특성을 시각화하고 검증합니다. 이 방법은 변별 있는 잠재 공간을 추정하기 위해 변분 오토인코더를 사용하며, 각 잠재 그룹에 의해 인코딩된 시각-의미적 특징을 이해하기 쉽게 표현할 수 있게 합니다. MIG-Vis는 두 마카크의 IT 피질에서의 다중 세션 신경 스파이크 데이터 세트를 검증한 결과, 다양한 시각 특징에 대한 명확한 의미적 선택도를 확인했습니다.

- **Technical Details**: MIG-Vis는 고차 시각 피질의 신경 잠재 그룹을 정의하기 위해 그룹 기반 변별적 변동 오토인코더를 활용합니다. 이 방법은 생성적 확산 모델을 사용하여 각 신경 잠재 그룹이 인코딩하는 특정 시각-의미적 특성을 시각화합니다. 우리의 목표는 각 잠재 그룹이 인코딩하는 시각-의미적 특성을 특성화하고 이해하는 것입니다.

- **Performance Highlights**: MIG-Vis의 실험 결과는 인코딩된 시각-의미적 정보의 전반적인 분포를 충실하게 표현하는 이미지를 합성하는 데 성공적이었습니다. 각 잠재 그룹에서의 시각-의미적 선택성은 서로 다른 범주 간의 변동, 범주 내 포즈 및 내용 세부 사항을 명확히 나타냅니다. 이 연구는 고차 시각 피질에서의 구조화된 의미적 표현을 직접적으로 증명하며, 우리가 시각 정보 인코딩 원칙을 이해하는 데 기여합니다.



### Learning to Reason for Hallucination Span Detection (https://arxiv.org/abs/2510.02173)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 환각(span) 탐지를 위한 새로운 접근 방식을 소개합니다. 기존 연구들은 환각 탐지를 이진(binary) 문제로 정의했지만, 실제로는 특정 환각 부분을 식별해야 하는 필요성이 존재합니다. 이 문제를 해결하기 위해, Chain-of-Thought (CoT) 추론을 활용한 강화 학습 프레임워크인 RL4HS를 제안합니다. 이 방법은 환각 탐지를 보다 정교하게 수행할 수 있는 가능성을 보여줍니다.

- **Technical Details**: RL4HS는 Group Relative Policy Optimization(GRPO) 기반으로 설계되며, span 수준의 보상 함수를 통해 추론을 장려합니다. 구체적으로, 환각 부분을 포함한 레이블 데이터셋을 사용하여 코트(CoT) 추론 기반 모델을 학습합니다. 또한, 비환각 예측에 대한 보상 불균형 문제를 해결하기 위해 class-aware policy optimization 기법을 도입했습니다. 이를 통해 보다 효과적인 모델 학습을 달성합니다.

- **Performance Highlights**: 실험 결과, RL4HS 모델이 사전 훈련된 추론 모델과 감독 학습(Supervised Fine-tuning) 방법보다 우수한 성능을 보여 명확히 증명되었습니다. 특히, 기존의 제너레이션 모델보다 환각 탐지에 더 효과적인 성과를 나타냈습니다. 또한, span-F1 점수 기준으로 강화 학습 접근 방식이 우월함을 흐릿한 기준에서 더 높게 평가받은 점도 주목할 만합니다.



### NoMod: A Non-modular Attack on Module Learning With Errors (https://arxiv.org/abs/2510.02162)
- **What's New**: 양자 컴퓨팅의 발전은 기존 공용 키 암호체계에 심각한 위협을 제기하고 있습니다. 저자들은 Module Learning With Errors (Module-LWE) 문제를 기반으로 한 새로운 하이브리드 암호 해독 방법인 NoMod ML-Attack을 소개합니다. 이 방법은 모듈 감소의 문제를 통계적 손상으로 간주하여 비밀 복구 문제를 강건한 선형 추정 문제로 재구성합니다.

- **Technical Details**: NoMod는 모듈 연산을 피하고 래핑 효과를 통계적 이상치로 다루는 새로운 접근 방식을 사용합니다. 이 방법은 선형 영역으로 문제를 재구성하여 데이터의 복잡성을 줄이고, 알고리즘 공학적 수정을 포함한 격자 기반 저감 테크닉을 통해 효율적인 비밀 복구를 제공합니다. 이 기술은 다양한 최적화 테크닉을 활용하여 처리 파이프라인의 품질, 샘플 크기 및 계산 비용 간의 주요 균형을 정의합니다.

- **Performance Highlights**: NoMod는 체크한 복잡성을 통해 n=350에서 이진 비밀을 완전하게 복구할 수 있으며, n=256에서는 희소 이항 비밀 복구에 성공하였습니다. CRYSTALS-Kyber 매개변수 (n, k) = (128, 3)와 (256, 2)에서도 성공적으로 희소 비밀 복구를 수행했습니다. 이 실험 결과들은 AI 기반 공격의 강건성을 입증합니다.



### Comparing Contrastive and Triplet Loss in Audio-Visual Embedding: Intra-Class Variance and Greediness Analysis (https://arxiv.org/abs/2510.02161)
Comments:
          8 pages, 4 tables, 3 figures

- **What's New**: 본 연구에서는 Contrastive Loss와 Triplet Loss의 이론적 및 실증적 비교를 통해 두 손실 함수가 표현 품질에 미치는 영향을 분석합니다. 특히 intra-class variance와 inter-class variance에 관한 실험을 통해 Triplet Loss가 클래스 내외의 변동성을 더 잘 유지한다는 것을 보여줍니다. 이 결과는 세밀한 구별이 요구되는 상황에서 Triplet Loss의 사용을 권장하고, Contrastive Loss는 보다 부드럽고 넓은 처리에서 유리하다는 결론에 이릅니다.

- **Technical Details**: Deep Metric Learning은 입력을 의미적 유사성이 반영된 공간에 매핑하여 임베딩을 형성하는 과정을 포함합니다. Contrastive Loss는 긍정 및 부정 쌍을 사용하여 intra-class의 집합성을 유지하고, inter-class의 분리를 강제합니다. 반면, Triplet Loss는 앵커, 긍정, 부정의 세 개를 사용하는데, 이는 더 강력한 업데이트를 통해 학습을 지속하게 하여 어려운 예제를 잘 다뤄냅니다.

- **Performance Highlights**: MNIST와 CIFAR-10 데이터셋에서 Triplet Loss가 Classification 및 Retrieval 작업에서 비교 우위를 보였습니다. Triplet Loss는 더 큰 gradient norm을 생성하여 어려운 샘플에 집중된 업데이트를 가능하게 하며, 이는 클래스 간 분리를 더 명확하게 드러냄을 의미합니다. 이러한 결과는 각 손실 함수 선택 시의 전략을 정의하는 데 도움을 줍니다.



### How to Find Fantastic Papers: Self-Rankings as a Powerful Predictor of Scientific Impact Beyond Peer Review (https://arxiv.org/abs/2510.02143)
- **What's New**: 이 논문에서는 인공지능(AI) 분야의 연구에서 고충격(high-impact) 연구를 식별하는 데 있어 저자 자신의 제출물에 대한 순위를 활용하는 방법을 탐구합니다. 기존의 동료 검토(peer review) 시스템과는 달리 저자들이 자신의 연구에 대한 독특한 이해를 가지고 있는 점에 착안해, 저자들의 자기 순위(self-rankings)가 미래 연구의 방향과 잠재력을 파악하는 데 얼마나 유용한지를 실험적으로 분석했습니다.

- **Technical Details**: 연구는 2023년 ICML에서 1,342명의 연구자가 제출한 2,592개의 논문에 대해 자기 순위를 매기도록 요청하고, 그 결과와 동료 검토 점수 및 최종 결정과의 상관관계를 분석했습니다. 특히, 저자들이 자기 순위를 매긴 논문은 후속 인용(citations) 수에서 특히 두 배의 증가율을 보였습니다. 자기 순위는 동료 검토 점수보다 더 정확한 인용 예측력을 보여 주었으며, 실험 결과는 선행 연구에 의해 통제된 변수를 고려하더라도 신뢰할 수 있었습니다.

- **Performance Highlights**: 저자들이 가장 높게 순위를 매긴 논문은 평균적으로 낮게 순위 매긴 논문보다 두 배 많은 인용을 받았고, 특히 150회 이상의 인용을 받은 22개의 논문 중 18개는 저자들이 본인 순위에서 가장 높게 평가한 논문들이었습니다. 이 연구 결과는 자기 순위가 고충격 연구를 식별하는 데 있어 귀중하고도 비용 효율적인 신호를 제공할 수 있음을 시사합니다. 향후 실제 저자 자기 순위를 기반으로 한 동료 검토 시스템 개선이 필요할 것으로 보입니다.



### BioinfoMCP: A Unified Platform Enabling MCP Interfaces in Agentic Bioinformatics (https://arxiv.org/abs/2510.02139)
Comments:
          20 pages, 8 figures, 3 tables

- **What's New**: BioinfoMCP는 생물정보학 도구와 AI 에이전트를 연결하는 혁신적인 플랫폼입니다. 이 플랫폼은 두 가지 주요 구성 요소인 BioinfoMCP Converter와 BioinfoMCP Benchmark로 구성되어 있습니다. BioinfoMCP Converter는 도구 문서를 이용해 대형 언어 모델을 통해 자동으로 MCP 서버를 생성하고, BioinfoMCP Benchmark는 변환된 도구의 신뢰성과 다양성을 검증합니다.

- **Technical Details**: BioinfoMCP Converter의 변환 과정은 준비, 실행 및 전달의 세 단계로 나뉩니다. 준비 단계에서는 도구의 매뉴얼을 준비하고, 실행 단계에서는 LLM(대형 언어 모델)을 통해 서버 코드를 생성합니다. 변환된 MCP 서버는 FastMCP 2.0 프레임워크를 기반으로 하여 생산 수준의 효율적인 서버를 제공합니다.

- **Performance Highlights**: BioinfoMCP 플랫폼은 38개의 MCP 변환 생물정보학 도구를 포함하고 있으며, 이들 도구는 94.7%의 성공률로 복잡한 작업 흐름을 실행했습니다. 이는 AI 에이전트와의 통합에 기술적 장벽을 제거하여, 연구자들이 깊은 프로그래밍 기술 없이도 자연어로 복잡한 분석을 수행할 수 있게 합니다.



### FlexDoc: Parameterized Sampling for Diverse Multilingual Synthetic Documents for Training Document Understanding Models (https://arxiv.org/abs/2510.02133)
Comments:
          Accepted at EMNLP 2025

- **What's New**: 이번 논문은 FlexDoc라는 새로운 합성 데이터 생성 프레임워크를 소개합니다. 이는 Stochastic Schemas와 Parameterized Sampling을 결합하여 다국어의 반구조적 문서를 현실감 있게 생성할 수 있습니다. FlexDoc을 통해 다양한 문서 변형을 제어된 방식으로 대량으로 생성할 수 있으며, 기존 데이터셋을 보강하여 자동 추출 (Information Extraction) 작업의 정확성을 최대 11% 개선할 수 있습니다.

- **Technical Details**: FlexDoc는 합성 문서 생성을 위해 Parameters Sampling을 중심으로 한 새로운 알고리즘을 도입하고, Dynamic Virtual Grid 알고리즘을 통해 문서 요소를 비오버랩(non-overlapping) 영역으로 조직하여 시각적 다양성을 높입니다. 이 과정은 프라이버시 위험을 줄이면서 특정 지역에 맞게 조정할 수 있는 가짜 값 생성기를 사용하여 안전하고 효율적인 데이터 생성이 가능합니다.

- **Performance Highlights**: 실험 결과 FlexDoc으로 생성된 데이터는 기존 하드 템플릿 방법과 비교하여 주석 작업을 90% 이상 줄이면서도 문서 이해 모델 개발의 속도를 크게 증가시켰습니다. 이 솔루션은 현재 적극적으로 배포되고 있으며, 기업 대형 문서 이해 모델의 개발을 가속화하고 데이터 확보 및 주석 비용을 대폭 절감하고 있습니다.



### VarCoNet: A variability-aware self-supervised framework for functional connectome extraction from resting-state fMRI (https://arxiv.org/abs/2510.02120)
Comments:
          My preview .pdf was not loading. Can you please share with me a compiled .pdf file so I can confirm that the result is correct?

- **What's New**: 이 연구에서는 개인 간 뇌 기능의 변동성을 단순한 노이즈가 아닌 의미 있는 데이터로 고려하여 VarCoNet이라는 자가 지도 학습(self-supervised learning) 기반의 새로운 프레임워크를 소개합니다. VarCoNet은 정적 상태 fMRI(resting-state fMRI) 데이터를 사용하여 강력한 기능적 연결체(functional connectome) 추출을 가능하게 합니다. 이 방법은 레이블이 없는 데이터에서도 유용하게 사용될 수 있으며, 개인의 뇌 기능을 인코딩하는 역할을 합니다.

- **Technical Details**: VarCoNet은 1D-CNN-Transformer 인코더를 통합하여 고급 시간 연속 데이터를 처리하며, 강력한 베이지안 하이퍼파라미터 최적화를 통해 성능을 향상시킵니다. 이 프레임워크는 기능적 개인 간 변동성을 활용하기 위해 세분화(segmentation)된 신호에 기초한 새로운 증강 전략을 사용하여 대조 학습(contrastive learning)을 지원합니다. 이러한 구조는 다양한 세션에서의 데이터 신호 길이 차이를 강건하게 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: VarCoNet의 성능은 휴먼 연결체 프로젝트(Human Connectome Project) 및 자폐 스펙트럼 장애(ASD) 분류에 사용된 ABIDE I 및 II 데이터셋을 통해 평가되었습니다. 두 가지 주요 작업에서 최신 방법들과 비교했을 때 VarCoNet은 일관되게 우수한 성능을 보였으며, 모델의 학습된 표현에 대한 해석 가능성도 제공하여 임상적 관련성을 지원합니다.



### Non-Asymptotic Analysis of Data Augmentation for Precision Matrix Estimation (https://arxiv.org/abs/2510.02119)
Comments:
          Conference paper at NeurIPS 2025 (Spotlight)

- **What's New**: 이 논문은 고차원 설정에서 역 공분산(inverse covariance 또는 precision matrix) 추정 문제를 다룹니다. 특히 정체 행렬(identity matrix)과 비례 관계를 갖는 선형 수축(Linear Shrinkage) 추정기와 데이터 증강(data augmentation)에서 파생된 추정기의 두 가지 클래스를중점적으로 살펴봅니다. 데이터 증강은 일반적으로 생성 모델(generative model)이나 원본 데이터를 무작위 변형(transformations)하여 인위적 샘플을 추가하는 일반적인 관행을 의미합니다.

- **Technical Details**: 두 클래스 모두에서 우리는 추정기를 도출하고 이차 오차(quadratic error)에 대한 집중 경계를 제공합니다. 이를 통해 방법 비교(method comparison)와 하이퍼파라미터 튜닝(hyperparameter tuning), 예를 들어 최적의 인공 샘플 비율(optimal proportion of artificial samples) 선택이 가능하게 합니다. 기술적인 측면에서 우리의 분석은 무작위 행렬 이론(random matrix theory)의 도구에 의존하며, 특정 구조를 가진 종속 샘플을 수용할 수 있는 일반화된 해석 행렬(generalized resolvent matrices)에 대한 새로운 결정론적 항등식(novel deterministic equivalent)을 소개합니다.

- **Performance Highlights**: 이론적 결과를 수치 실험(numerical experiments)으로 뒷받침하였습니다. 이러한 실험은 제안된 방법들이 실제 데이터에서 어떻게 작동하는지를 보여주며, 특히 높은 차원의 문제에서도 잘 작동하는 성능을 입증합니다. 선형 수축 추정기와 데이터 증강을 기반으로 한 추정기의 비교를 통해, 각 방법의 강점과 약점을 평가할 수 있습니다.



### SoundReactor: Frame-level Online Video-to-Audio Generation (https://arxiv.org/abs/2510.02110)
- **What's New**: 본 논문에서는 프레임 수준의 온라인 비디오-오디오(V2A) 생성 작업을 소개합니다. 기존의 V2A 모델은 전체 비디오 시퀀스나 프레임 조각이 미리 제공되는 오프라인 방식으로 운영되어, 실시간 콘텐츠 생성과 같은 인터랙티브 애플리케이션에서의 활용이 제한되었습니다. 이를 해결하기 위해 제안된 SoundReactor 프레임워크는 미래 비디오 프레임에 접근하지 않고 오토회귀적으로 오디오를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: SoundReactor는 엔드-투-엔드 인과성을 보장하고 오디오-비주얼(visual) 동기화를 목표로 하여 낮은 프레임 당 지연(latency)을 제공합니다. 이 모델은 연속적인 오디오 잠재(latents)를 사용하는 인코더 없는 인과적 트랜스포머로 구성되어 있으며, DINOv2 비전 인코더에서 추출한 격자(grid) 특징을 이용해 시각적 조건을 부여합니다. 각 프레임마다 한 개의 토큰(token)으로 집계하여 효율성을 유지합니다.

- **Performance Highlights**: 제안된 모델은 AAA 타이틀의 다양한 게임플레이 비디오 기준으로 높은 품질의 전체 대역 스테레오 오디오를 생성하며, 이는 객관적 평가 및 인간 평가에서 검증되었습니다. 또한, 30FPS 및 480p 비디오에서 단일 H100을 사용하여 낮은 프레임 당 파형 레벨 지연(26.3ms, 헤드 NFE=1)과 31.5ms(NFE=4)를 달성했습니다. 데모 샘플은 제공된 URL에서 확인할 수 있습니다.



### Adaptive Kernel Selection for Stein Variational Gradient Descen (https://arxiv.org/abs/2510.02067)
- **What's New**: 이 연구는 Stein Variational Gradient Descent (SVGD)에서의 커널 파라미터 선택을 동적으로 조정하는 적응형 기법을 제안합니다. 기존의 커널 밴드폭 설정 방식인 median heuristic의 한계를 극복하고, kernelized Stein discrepancy (KSD)를 최대화하여 최적의 이송이 가능하다는 점을 강조합니다. Adaptive SVGD (Ad-SVGD) 방법은 입자 업데이트와 커널 튜닝을 반복적으로 수행하여, 고차원 공간에서 변환 성능을 향상시킵니다.

- **Technical Details**: Ad-SVGD는 재생 커널 힐버트 공간(RKHS)에서의 업데이트 방향과 Kullback-Leibler (KL) 발산의 함수적 그래디언트 기반으로 작동합니다. KSD의 최대화를 통해 커널 파라미터를 조정하며, 여러 연속 커널 파라미터에 대한 최적화를 가능하게 합니다. 또한, 이 방법은 분산의 과소 추정을 줄일 수 있으며, 이론적 분석을 통해 알고리즘의 수렴성을 확립하였습니다.

- **Performance Highlights**: 수치 실험에서 Ad-SVGD는 다양한 작업에서 기존의 median heuristic보다 일관되게 더 우수한 성능을 보였습니다. 실험 결과는 적응형 커널 선택 방법이 성능 저하를 완화하는 데 기여함을 입증하였습니다. 이 연구는 SVGD의 고차원 의존성을 감소시키는 효과적인 접근 방식을 제시합니다.



### ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection (https://arxiv.org/abs/2510.02060)
Comments:
          9 pages, 4 figures

- **What's New**: ReTabAD는 텍스트 의미론을 복원하여 컨텍스트가 인식되는 표 형식의 이상 탐지(tabular anomaly detection, AD) 연구를 가능하게 하는 최초의 벤치마크입니다. 이 프로젝트는 구조화된 텍스트 메타데이터로 풍부하게 강화된 20개의 신중하게 선정된 표 형식 데이터 세트를 제공합니다. 또한, 특정 작업에 대한 훈련 없이 의미론적 맥락을 활용하는 제로샷 LLM(zero-shot LLM) 프레임워크를 제안하여 향후 연구의 강력한 기준점을 마련합니다.

- **Technical Details**: ReTabAD는 탭형 데이터에 세부적인 텍스트 메타데이터를 통합하여 AD 알고리즘의 성능을 극대화하는 데 중점을 둡니다. 이 구조는 17가지 알고리즘을 포괄하는 포괄적 평가를 가능하게 하며, 수치적 패턴이 아닌 의미론적 맥락에 기반하여 예측을 수행할 수 있도록 합니다. 또한, 여러 모델에서 평균 7.6% AUROC(Area Under the Receiver Operating Characteristic Curve) 성능 개선을 보이면서, 제로샷 LLM 접근 방식을 통해 훈련 기반 방법과 동등한 성과를 달성하였습니다.

- **Performance Highlights**: ReTabAD는 컨텍스트 인식 AD를 위한 체계적 연구를 지원하며, 기존 데이터 세트와의 비교에서 높은 성능을 보여줍니다. 이 새로운 벤치마크는 모델이 의미론적 정보를 통해 더욱 직관적으로 이상 상태를 해석할 수 있도록 도와주며, 예측의 해석 가능성을 향상시키는 데 기여하고 있습니다. ReTabAD는 탭형 데이터를 위한 이상 탐지 분야에서의 시스템적 발전을 이끌 것으로 기대됩니다.



### Multidata Causal Discovery for Statistical Hurricane Intensity Forecasting (https://arxiv.org/abs/2510.02050)
Comments:
          19 pages, 7 Figures, 1 Table, SI

- **What's New**: 이 연구는 대서양 허리케인 강도를 예측하기 위한 새로운 인과 발견 프레임워크를 제시합니다. 기존의 통계적 방법이 상관관계를 우선시하는 경향에 비해, 이 프레임워크는 환경 변수를 보다 객관적으로 파악합니다. 특히, 기상 데이터 재분석을 이용하여 인과적으로 연결된 예측 변수를 식별하는 실험을 여러 차례 수행하였습니다.

- **Technical Details**: 인과 발견 프레임워크는 다중데이터 접근 방식을 활용하여 예측 변수를 제안합니다. 이 연구에서는 데이터 기반 기법을 보완하기 위해 인과 추론 방법을 사용하여 관찰할 수 없는 변수의 방향성과 인과 관계를 명확히 하는 데 중점을 둡니다. 조사에 사용된 데이터는 ECMWF의 고해상도 재분석 데이터를 포함하며, 기존 SHIPS 모델에 causally informed predictors를 추가하여 실험하고 있습니다.

- **Performance Highlights**: Causal feature selection은 기존의 상관 기반 방법들보다 예측 성능이 우수한 것으로 나타났습니다. 특히 3일 이하의 예보에서 더욱 두드러진 성능 향상을 보였으며, Hurricnane Larry 사례 연구를 통해 선택된 예측 변수와 허리케인 강도 간의 인과적 연결성을 강조하였습니다. 또한 SHIPS+ 모델은 24, 48, 72시간의 단기 예측에서 예측 기술을 증가시키는 것으로 확인되었습니다.



### Variational Secret Common Randomness Extraction (https://arxiv.org/abs/2510.02048)
- **What's New**: 이 논문은 알리스를 비롯한 두 당사자가 상관된 무작위 출처에서 공통 무작위성(common randomness, CR) 또는 비밀 키를 추출하는 문제를 연구합니다. 이를 위해 실용적인 두 단계 CR 추출 프레임워크를 제안하며, 이러한 접근법은 eavesdropper(도청자)인 이브의 존재하에 공개 논의를 통해 이루어집니다.

- **Technical Details**: 제안된 프레임워크의 첫 번째 단계에서, Alice와 Bob은 확률적 신경망(probabilistic neural network, NN) 인코더를 사용하여 그들의 관찰을 거의 균일한 무작위 변수(discrete random variables, RVs)로 변환하며, 높은 동의 확률과 정보 유출 최소화를 목표로 합니다. 두 번째 단계에서는 안전한 스케치(secure sketch)를 이용하여 비밀 키를 조정하게 되며, 이는 코드 오프셋(code-offset) 구조를 통해 구현됩니다.

- **Performance Highlights**: 실험 결과, 제안된 CR 추출 프레임워크와 기반으로 한 물리적 레이어 키(physical layer key, PLK) 생성 방식이 실현 가능함을 보여주며, 특히 모바일 고속 통신 시나리오에서 좋은 성능을 보입니다. 다양한 실험 환경에서 이브의 위치에 대한 부분적인 정보를 가질 때도 이 방법이 효과적임을 입증하였습니다.



### Zero-shot Human Pose Estimation using Diffusion-based Inverse solvers (https://arxiv.org/abs/2510.02043)
- **What's New**: 이 논문은 인체 자세 추정(pose estimation) 문제를 역문제(inverse problem)로 정의하고, 사용자 맞춤형 조정 없이 제로샷 제너럴리제이션(zero-shot generalization)을 가능하게 하는 알고리즘을 설계했습니다. 제안된 InPose 방법은 사전 훈련된 diffusion 모델을 사용하여 회전 측정(rotation measurements)만을 조건으로 하여 인체 자세를 생성적으로 추정하는 혁신적인 접근을 제공합니다. 기존 방법들은 사용자에 따라 위치 측정값(location measurements)이 크게 달라지는 문제를 겪었지만, InPose는 이러한 제약 없이 사용자별로 정확한 자세를 예측할 수 있습니다.

- **Technical Details**: InPose는 33개의 신체 관절(joints)에서 얻은 회전 및 위치 측정을 통해 전체 2222개의 관절을 추적합니다. 알고리즘은 사용자의 신체 크기에 따라 스케일-프리 포즈(scale-free pose)를 조정하여 최적의 자세를 추정합니다. 이 과정에서 확률론적 원리를 적용하여 자세의 가능성(likelihood)을 평가하고, 이는 디퓨전 노이즈 제거(diffusion denoising) 프로세스를 이끄는 가이드 역할을 합니다. 결과적으로, 사용자 맞춤형 3D 자세를 효과적으로 생성하는 방식입니다.

- **Performance Highlights**: InPose는 AMASS 데이터셋을 통해 다양한 신체 크기와 형태에서 뛰어난 일반화 결과를 보였습니다. 저자들은 각종 실험을 통해 기존 모델들보다 우수한 성능과 신뢰성을 입증했습니다. InPose는 다양한 환경에서 사용자에 대한 조정 없이도 일관된 결과를 지속적으로 생성함으로써 역문제 접근 방식의 유용성을 강조했습니다.



### ShapeGen3DCP: A Deep Learning Framework for Layer Shape Prediction in 3D Concrete Printing (https://arxiv.org/abs/2510.02009)
- **What's New**: ShapeGen3DCP는 3D 콘크리트 프린팅(3DCP)에서 필라멘트 단면 기하학을 빠르고 정확하게 예측하기 위한 딥러닝 프레임워크입니다. 이 방법은 유체 상태의 재료 특성과 프린팅 변수(노즐 직경, 높이, 프린팅 속도 등)를 입력으로 받아 단면 형상을 직접 예측하는 신경망 아키텍처에 기반합니다. 예측된 기하학은 푸리에 기술자(Fourier descriptors)를 사용하여 매끄럽고 닫힌 프로파일로 압축 표현되며, 이를 통해 예측 작업이 소수의 계수 집합으로 축소됩니다.

- **Technical Details**: ShapeGen3DCP의 입력 변수는 물리적으로 의미 있는 무차원 매개변수로 개편되어 입력의 차원을 줄이고 강건성을 향상시킵니다. 이 시스템은 유효 상태의 수치 모델로 합성 생성된 학습 데이터셋을 사용하여 실험 데이터의 부족 문제를 극복합니다. 뿐만 아니라, 이 접근 방식은 고정된 물질 파라미터 대신 다양한 변수 입력을 통해 단면 예측의 유연성을 높입니다.

- **Performance Highlights**: 이 방법은 훈련 과정에 포함되지 않은 다양한 수치 및 실험 사례와 비교하여 뛰어난 결과를 보여 주었습니다. 결과적으로 이 프레임워크의 정확성과 신뢰성이 검증되었으며, 3DCP의 전처리 및 최적화도 가능해질 것입니다. 앞으로 시뮬레이션 및 센서 피드백과의 결합을 통해 3DCP의 실시간 프로세스 최적화, 결함 탐지 및 인쇄 매개변수의 적응 제어가 이루어질 것으로 기대됩니다.



### Multi-bit Audio Watermarking (https://arxiv.org/abs/2510.01968)
- **What's New**: 이번 연구에서는 Timbru라는 후처리 오디오 워터마킹 모델을 제안합니다. 이 모델은 embedder-detector 모델을 훈련시키지 않고도 뛰어난 내성(robustness) 및 인지 불가능성(imperceptibility) 딜레마를 해결합니다. 44.1kHz 스테레오 음악 스니펫을 입력으로 받아, 사전 훈련된 오디오 변환 자동 부호화기(VAE)의 잠재 공간(latent space)에서 인지 불가능한 섭동을 추가하는 방식으로 워터마크를 삽입합니다.

- **Technical Details**: Timbru의 핵심 아이디어는 오디오의 잠재 표현(latent representation)에 섭동을 추가하여 비밀 키에 맞는 워터마크 문자열을 삽입하는 것입니다. 이 과정에서 각 사용자는 무작위로 선택된 직교 벡터로 구성된 비밀 키를 가지며, 입력 오디오의 특성을 비밀 키와 정렬하도록 섭동을 더합니다. 결과적으로 사전 훈련된 CLAP 모델을 사용하여 워터마크를 감지하고 여러 공격에 대한 내성을 강화하는 최적화 과정을 수행합니다.

- **Performance Highlights**: Timbru는 기존 오디오 워터마킹 방법들과 비교하여 атака에 대한 가장 낮은 평균 비트 오류율(bit error rate)을 기록했습니다. 인지 품질(perceptual quality)을 보존하면서도 효과적으로 여러 가지 공격을 견디며, 사용자 기여를 보호하는 데 있어 효율적이고 데이터에 구애받지 않는 접근 방식을 제시합니다. 실험 및 주관적 평가 결과 또한 기존 방법들과 유사한 수준의 인지 품질을 유지하면서도 개선된 내성을 보여주었습니다.



### Bias beyond Borders: Global Inequalities in AI-Generated Music (https://arxiv.org/abs/2510.01963)
- **What's New**: 최근 몇 년 동안 음악 생성 모델에서 큰 발전이 있었지만, 국가, 언어, 문화 및 음악 장르에 따른 편향에 대한 연구는 부족했습니다. 이러한 문제를 해결하기 위해, GlobalDISCO라는 대규모 데이터셋이 새롭게 소개되었으며, 이는 73,000개의 음악 트랙과 93,000개의 참조 트랙으로 구성되어 있습니다. 이 데이터셋은 147개 언어를 포함하고 있으며, 79개 국가와 다섯 대륙에서 수집된 아티스트의 음악 스타일 프롬프트를 포함합니다.

- **Technical Details**: GlobalDISCO 데이터셋은 MusicBrainz에서 아티스트 정보를 수집하여 구축되었습니다. 총 93,000개의 실제 음악과 73,000개의 생성된 음악이 79개국과 147개 언어로 구성되어 있으며, 이는 4개의 최첨단 상업 모델(예: Udio, Suno, Mureka, Riffusion)을 활용하여 생성되었습니다. 이 데이터셋은 글로발 다양성을 평가하는 데 중점을 두고 있으며, 특정 지역 장르에 맞춘 음악 생성 시에도 주류 장르와의 연관성을 보여주고 있습니다.

- **Performance Highlights**: 모델 성능의 차이는 고자원 지역과 저자원 지역 간에 뚜렷하게 나타났으며, 일반적으로 아프리카 및 남서 아시아 지역에서 생성된 음악은 주류 지역에 비해 더욱 다양성을 띱니다. 데이터셋 내의 인기 장르와 지역 장르의 평균 FAD 및 KAD 스코어를 비교한 결과, 인기 장르는 보다 높은 음악 생성 품질을 나타내는 반면, 지역 장르는 저자원 지역에서 더 낮은 품질을 조사한 결과를 확인할 수 있었습니다.



### Uniform-in-time convergence bounds for Persistent Contrastive Divergence Algorithms (https://arxiv.org/abs/2510.01944)
- **What's New**: 본 논문에서는 비정규화 밀도의 최대 우도 추정(Maximum Likelihood Estimation, MLE)을 위한 지속적인 시간 형식의 영속적 대립 조화(Persistent Contrastive Divergence, PCD)를 제안합니다. PCD는 파라미터 최적화와 관련된 파라미터화된 밀도의 샘플링을 동시에 수행하는 확률 미분 방정식(Stochastic Differential Equations, SDE) 시스템으로 표현됩니다. 이 새로운 방식을 통해 PCD 반복과 MLE 솔루션 간의 오차에 대한 명시적인 경계를 도출할 수 있습니다.

- **Technical Details**: 연구에서는 MLE 솔루션을 목표로 하는 두 시간 척도(Langevin Diffusion) 시스템을 개발하여 EBMs(에너지 기반 모델)의 훈련 문제를 다룹니다. 이 시스템은 최적의 상수 신호를 사용하고 있습니다. 특히, 제안된 다중 스케일 SDE의 유일한 이산화가 고전적인 PCD 알고리즘과 같다는 점이 강조되었습니다. 이 논문에서 논의되는 접근 방식은 고전적인 PCD 알고리즘을 분석하고 새로운 알고리즘을 개발하는 데 활용될 수 있습니다.

- **Performance Highlights**: 우리는 제안된 다중 스케일 Langevin 확산을 위한 수치적 이산화 기법을 제공하며, 이로써 EBMs 훈련을 위한 실용적인 알고리즘을 제시합니다. 특히, 다중 스케일 시스템의 오일러-마루야마 이산화가 고전적 PCD 알고리즘으로 이어지고, 이에 대한 오차 분석이 최초로 수행되었습니다. 추가로, PCD 알고리즘을 개선하기 위해 안정성이 뛰어난 수치적 적분기인 S-ROCK 방법을 제안하고, 이 방식이 MLE 솔루션 간의 오차에 대한 한계와 UiT 경계를 증명하는 데 기여함을 보였습니다.



### Smooth Quasar-Convex Optimization with Constraints (https://arxiv.org/abs/2510.01943)
- **What's New**: 이번 연구에서는 쿼사르-컨벡스 함수(quasar-convex functions)에 대한 새로운 접근 방식을 제시합니다. 특히, 제약 조건이 포함된 일반적인 컨벡스 함수에 대한 근사 가속화된 알고리즘을 설계하여, 이전 연구에서 열린 문제를 해결합니다. 또한, 프로젝트 경량 하강(projected gradient descent) 및 프랭크-울프(Frank-Wolfe) 알고리즘을 이 컨벤션 하의 쿼사르-컨벡스 설정에서 분석합니다.

- **Technical Details**: 이 연구는 $	ilde{O}(1/(eta	ext{sqrt}	ext{ }{	ext{ε}}))$의 첫 번째 차수 쿼리를 달성하는 비정확한 가속화된 근접점 알고리즘(inexact accelerated proximal point algorithm)을 개발하였습니다. 이 알고리즘은 Riemannian 최적화(Riemannian optimization) 문제에 대한 복잡성을 향상시킵니다. 또한, 쿼사르-컨벡스 함수(quasar convex smooth function)의 일반적인 컨벡스 제약 조건을 다루는 첫 번째 차수 방법(first-order methods)을 조사하여 중요한 기초 자료를 제공합니다.

- **Performance Highlights**: 연구 결과, 제안된 알고리즘은 이전 알고리즘보다 우수한 성능을 보여 주며, 가속화된 Riemannian 최적화 문제를 해결하는 데 필요한 계산 복잡성을 현저히 줄입니다. 첫 번째 차수 방법의 분석을 통해, 쿼사르-컨벡스 설정에서의 다양한 최적화 알고리즘의 적합성을 입증했습니다. 이와 같은 결과는 선형 동적 시스템(linear dynamical systems), 일반화된 선형 모델(generalized linear models) 및 기타 최적화 문제에의 응용 가능성을 보여줍니다.



### Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors (https://arxiv.org/abs/2510.01934)
Comments:
          23 pages, 13 figures. Code is available at \url{this https URL}

- **What's New**: 이 논문은 산업 안전 검사를 간소화하고 효율적으로 수행할 수 있는 Few-shot anomaly detection 기술을 제안합니다. 기존의 방법들보다 적은 수의 샘플로 정상과 비정상적 특징을 구분할 수 있는 새로운 접근을 소개하고 있습니다. 특히, FoundAD라는 새로운 Few-shot anomaly detector를 설계하여 비정상 이미지를 효과적으로 감지할 수 있도록 하였습니다.

- **Technical Details**: FoundAD는 자연 이미지 매니폴드(natural image manifold) 위에 비선형 프로젝션 연산자(nonlinear projection operator)를 학습함으로써 구현됩니다. 이 간단한 연산자는 이미지 내의 분포에서 벗어난 영역(out-of-distribution regions)을 특성화하고 식별하는 데 효과적으로 사용됩니다. 또한, 여러 기초 시각 인코더(foundation visual encoders)와의 평가를 통해 우리의 접근 방식을 검증하였습니다.

- **Performance Highlights**: 다양한 실험을 통해 다중 클래스 탐지가 가능하고 이전 방법들보다 적은 매개변수를 사용하면서도 경쟁력 있는 성능을 달성함을 보여주었습니다. 특히, 새로운 DINOv3 모델에 대한 평가 결과는 이 기술의 효용성 및 상승된 성능을 입증합니다. 이 연구는 기초 특징(foundation features)에 대한 새로운 관점을 제시하며 Few-shot anomaly detection 분야의 발전을 이끌 것입니다.



### Precise Dynamics of Diagonal Linear Networks: A Unifying Analysis by Dynamical Mean-Field Theory (https://arxiv.org/abs/2510.01930)
Comments:
          54 pages

- **What's New**: 본 연구에서는 대각 선형 네트워크(Diagonal Linear Networks, DLN)의 경량화된 분석을 통해 다양한 학습 동역학을 통합적으로 살펴보았습니다. Dynamical Mean-Field Theory (DMFT)를 사용하여 고차원에서의 재현 동역학을 잘 설명하는 저차원 효과적 프로세스를 도출하였습니다. 이를 통해 DLN의 손실 수렴 속도와 일반화 사이의 상호작용을 이해하는 데 기여하고, 높은 차원의 학습 동역학 분석에 DMFT 접근법의 효과를 입증하였습니다.

- **Technical Details**: DLN은 학습 알고리즘의 비선형 동역학과 관련된 여러 비트리비어리즘을 포착하는 중요한 이론적 모델로, 본 연구에서는 이 모델을 통해 훈련 다이내믹스의 통합 프레임워크를 개발하였습니다. DMFT를 활용하여 도출된 연립 방정식은 드문 회귀(sparse regression)에서 DLN의 그래디언트 흐름 훈련 동역학을 특성화합니다. 그 결과는 훈련 시간과 초기화 스케일에 따라 달라지는 동역학적 레짐을 식별하고, 작은 초기화가 더 나은 일반화를 이끈다는 것을 보여주었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 큰 초기화에서 DLN은 데이터에 대한 메모리화(memorizing)에서 일반화(generalizing) 솔루션으로 급격히 전환되며, 작은 초기화의 경우 초기 탐색 평탄기와 이어지는 점진적 학습(incremental learning) 과정을 관찰하였습니다. 손실 수렴 속도도 분석하여 작은 초기화가 느린 수렴을 초래함을 보여주었으며, 이는 최적화 속도와 일반화 성능 간의 트레이드오프를 확립하는 데 기여합니다. 이러한 결과는 DLN의 이론적 이해를 심화시키고, DMFT가 비선형 고차원 학습 시스템의 동역학을 탐구하는 데 강력한 도구임을 강조합니다.



### Automated Defect Detection for Mass-Produced Electronic Components Based on YOLO Object Detection Models (https://arxiv.org/abs/2510.01914)
Comments:
          12 pages, 16 figures, 7 tables, and published in IEEE Sensors Journal

- **What's New**: 본 논문에서는 산업에서 널리 사용되는 이중 인라인 패키지(DIP)의 자동 결함 탐지 시스템을 제안합니다. 이 시스템은 디지털 카메라 광학과 딥 러닝(Deep Learning) 기반 모델을 사용하여 작동하며, 전통적인 수작업 검수에 드는 시간과 노력을 줄이는 것을 목표로 합니다. 또한, 결함 데이터 부족 문제를 해결하기 위해 ConSinGAN을 이용해 학습과 테스트에 적합한 데이터셋을 생성합니다.

- **Technical Details**: 이 시스템은 제어 시스템, 이미징 장비 및 기계 장비의 세 부분으로 구성되어 있습니다. 제어 시스템은 개인 컴퓨터(PC)와 프로그래머블 로직 컨트롤러(PLC)를 포함하여 이미징 장비와 상호 작용하고 SCADA를 통해 딥 러닝 모델과 데이터 분석을 통합합니다. 이미징 장비는 산업용 카메라와 다양한 광원 장비로 구성되어 있으며, 기계 장비는 검사의 자동화를 위해 다양한 기능을 수행합니다.

- **Performance Highlights**: 제안된 YOLOv7 모델은 ConSinGAN과 함께 사용할 때 정확도 95.50%, 탐지 시간 285ms를 기록하였습니다. 이 결과는 기존의 임계값 기반 방법보다 경쟁력 있는 성과를 보여주며, 자동화된 결함 검출 시스템의 실효성을 높입니다. 또한, 관련 생산 라인과 SCADA 인터페이스를 개발하여 실제 응용 프로그램에서의 성능을 검증하였습니다.



### Constrained Adaptive Rejection Sampling (https://arxiv.org/abs/2510.01902)
- **What's New**: 이 논문에서는 CARS(Constrained Adaptive Rejection Sampling)를 소개합니다. CARS는 기존의 Rejection Sampling(RS)과 Greedy Constrained Decoding(GCD)의 단점을 해결하여 성능을 개선한 새로운 접근법입니다. 특히, CARS는 샘플 파라미터를 체계적으로 수정하여 유효한 샘플을 더 효율적으로 생성할 수 있도록 합니다.

- **Technical Details**: CARS는 Adaptive Rejection Sampling(ARS)에서 발전된 방식으로, invalid한 프리픽스(prefix)를 trie에 기록해 나중에 샘플을 생성할 때 해당 probability mass를 차감합니다. 이렇게 하면 샘플이 계속해서 무효화되는 것을 방지하고, 점진적으로 acceptance rate가 개선됩니다. 이 알고리즘의 핵심은 조건부 분포를 유지하며, 연결된 prefix를 검증하는 것입니다.

- **Performance Highlights**: 다양한 실험에서 CARS는 기존의 constrained sampling 방법들과 비교하여 높은 acceptance rate와 다양성을 보여주었습니다. 또한, CARS는 유효한 샘플을 생성하는 데 있어 평균적으로 더 낮은 비용을 요구하며, 성능 면에서 새로운 최첨단 기준을 설정했습니다. 프로그램 퍼징(program fuzzing) 및 분자 생성(molecular generation) 분야에서 특히 강력한 효율성을 기록했습니다.



### Deep Hedging Under Non-Convexity: Limitations and a Case for AlphaZero (https://arxiv.org/abs/2510.01874)
Comments:
          15 pages in main text + 18 pages of references and appendices

- **What's New**: 이번 논문은 금융 공학의 핵심 문제인 불완전한 시장에서의 복제 포트폴리오 구성에 대해 다룹니다. 투자자와 시장 간의 두 플레이어 게임으로 모델링하여, 투자자가 미래 상태에 대한 전략적인 베팅을 하는 방식입니다. Monte Carlo Tree Search의 성공에 영감을 받아 AlphaZero 기반 시스템을 도입하여, 깊은 헷징(deep hedging)과 성능을 비교합니다.

- **Technical Details**: 기술적으로, 이 논문은 깊은 헷징의 제한점을 이론적으로 분석하고, 그 효율성이 볼록성 가정에 의존한다는 점을 강조합니다. 종합적으로, 투자자는 복합 자산의 수익 프로필을 단순한 도구를 이용하여 복제하게 되며, 주어진 시장에서는 거래 비용, 자본 제약, 규제 제한 등이 복잡성을 증가시킵니다. 또한, AlphaZero는 저데이터 환경에서 더 효율적인 샘플링 특성을 가지고 있어, 이점이 돋보입니다.

- **Performance Highlights**: 실험 결과, AlphaZero는 매개 변수가 불리한 조건에서도 근사 최적의 복제 전략을 일관되게 찾을 수 있음을 보였습니다. 반면, 깊은 헷징은 비볼록 환경에서 수렴에 실패하며, 지역 최적점에 그치는 경향을 보입니다. AlphaZero/MuZero는 더 적은 훈련 샘플로도 높은 성능을 유지하며, 복잡한 금융 계획 작업에 적합한 가능성을 제시합니다.



### Ranking Items from Discrete Ratings: The Cost of Unknown User Thresholds (https://arxiv.org/abs/2510.01871)
Comments:
          12 pages, 4 figures

- **What's New**: 이 연구는 정보 검색 및 추천 시스템에서 아이템을 순위 매기는 작업에 대한 새로운 이론적 통찰력을 제공합니다. 연구팀은 사용자에게 부여된 점수를 통해 세부적인 아이템 순위를 복원할 수 있는 가능성을 조사하였으며, 이는 사용자의 기준(threshold)와 아이템의 점수(score)가 숨겨져 있는 경우 매우 도전적인 문제라는 점을 밝혔습니다. 이 논문은 이론적인 한계를 정량화하고, 이를 통해 사용자 평가 기반 순위 매기의 복잡성을 분석합니다.

- **Technical Details**: 모델링에서 각 아이템은 0과 1 사이의 점수(score)를 가지며, 사용자들 각자는 저마다의 잠재적 기준(threshold)을 가지고 있습니다. 사용자는 아이템의 점수가 자신의 기준을 초과할 경우 긍정적으로 평가합니다. 실험적으로, 사용자 수(m)가 아이템 수(n)에 비례할 때 순위의 정확성은 O(n)으로 나타났으며, 사용자 수가 Ω(n²)일 경우 거의 완벽한 순위를 얻을 수 있다는 결과를 도출했습니다.

- **Performance Highlights**: 본 연구에서는 평가 요청에 비해 아이템 비교 요청이 훨씬 더 효과적임을 수학적으로 증명하였습니다. 사용자에게 적절한 기준을 가진 아이템을 요청하는 것이 더 나은 분별을 위해 필수적이며, 사용자의 기준이 다양해야만 세부 순위를 형성할 수 있음을 강조했습니다.  실험 결과는 새로운 알고리즘인 threshold binary search (TBS)의 효율성을 지지하며, 이 알고리즘은 사용자의 정보를 최대한 활용하여 정밀한 순위 매기기를 가능하게 합니다.



### Microscaling Floating Point Formats for Large Language Models (https://arxiv.org/abs/2510.01863)
- **What's New**: 이번 논문은 큰 언어 모델(LLMs)의 연산 및 메모리 요구량을 최적화하는 혁신적인 접근법인 마이크로스케일링(microscaling) 부동소수점 형식을 제안합니다. 이 방법은 값 블록 간에 공유 스케일을 활용하여 메모리 및 계산 오버헤드를 크게 줄이는 것을 목표로 합니다. 8비트 부동소수점 형식에 대한 적용을 통해, 이 논문은 LLM의 메모리 사용과 계산 비용을 효과적으로 감소시키는 결과를 보여줍니다.

- **Technical Details**: 마이크로스케일링 형식은 전통적인 부동소수점 형식의 한계를 극복하고, 각 값에 개별 지수를 할당하는 대신 블록에 공유 지수를 도입합니다. 이를 통해 메모리 사용량을 줄이고, 1바이트로 압축된 부동소수점 표현을 가능하게 하면서도 동적 범위를 확장합니다. GPT-2 아키텍처에서 마이크로스케일링 형식을 여러 구성으로 시험하여, 훈련과 추론 중 경쟁력 있는 정확도를 보였습니다.

- **Performance Highlights**: 실험 결과, 마이크로스케일링 구현은 제약이 있는 환경에서 LLM을 효율적으로 훈련시키고 배포할 수 있는 잠재력을 보여줍니다. 메모리 효율성과 계산 처리량, 정확도 측면에서 긍정적인 결과를 도출할 수 있었으며, 이는 리소스를 절약하면서도 성능을 유지하는 데 기여합니다. 논문의 주요 발견은 부동소수점 연산에서 수치 오류를 줄이기 위해 적절한 반올림 정책을 사용해야 한다는 점입니다.



### NGGAN: Noise Generation GAN Based on the Practical Measurement Dataset for Narrowband Powerline Communications (https://arxiv.org/abs/2510.01850)
Comments:
          16 pages, 15 figures, 11 tables, and published in IEEE Transactions on Instrumentation and Measurement, Vol. 74, 2025

- **What's New**: 본 논문에서는 좁은 대역 전력선 통신(NB-PLC) 시스템의 비주기적 비동기 충격 소음(nonperiodic asynchronous impulsive noise) 처리를 개선하기 위해, 복잡한 특성을 학습하는 노이즈-생성 적대 신경망(Generative Adversarial Network, GAN)을 제안합니다. 기존 노이즈 모델들이 일부 특성만을 포착하는 한계를 극복하기 위해, 우리는 실제로 측정된 노이즈 샘플을 사용하여 데이터 증강(data augmentation)을 수행합니다. 특히, 제안된 노이즈-생성 GAN (NGGAN)은 측정된 노이즈 통계에 밀접하게 일치하는 방식으로 설계되었습니다.

- **Technical Details**: 이 연구에서 제안된 NGGAN은 입력 신호의 길이를 조절하여 사이클로 스테이셔너리(cyclo-stationary) 노이즈 생성을 용이하게 하며, Wasserstein 거리(Wasserstein distance)를 손실 함수로 사용하여 생성된 노이즈와 훈련 데이터셋 간의 유사성을 강화합니다. 우리는 PSCRGM, FRESH 필터, 실제 NB-PLC 시스템에서 측정된 값들을 포함한 훈련 데이터셋을 사용하여 GAN 모델의 유사성 성능을 분석합니다. 결과적으로, NGGAN은 기존의 수학적 노이즈 모델로는 잡을 수 없는 복잡한 노이즈 통계의 샘플을 생성할 수 있는 효율적인 데이터 증강 방법으로 확인되었습니다.

- **Performance Highlights**: 시뮬레이션 결과, NGGAN은 기존 GAN 기반 모델들에 비해 노이즈에 대한 강건성을 더욱 개선하는 것으로 나타났습니다. 성능 메트릭에는 최대값, 평균값, 에너지 값, 표준 편차, 왜도, 첨도, 고유치(peak) 수 등이 포함되어 있으며, 이러한 메트릭들을 통해 생성된 노이즈의 품질을 평가했습니다. 또한, GitHub를 통해 Python 소스 코드를 제공하며, 더 나아가 소비자 전자 기기에서의 적용 가능성도 탐색하고 있습니다.



### A reproducible comparative study of categorical kernels for Gaussian process regression, with new clustering-based nested kernels (https://arxiv.org/abs/2510.01840)
- **What's New**: 이 논문은 Gaussian process regression에서 연속형 및 범주형 입력을 위한 범주형 커널(categorical kernels)의 비교 연구를 재현 가능한 방법으로 제시합니다. 기존 연구들은 평가 지표(evaluation metrics)나 최적화 과정(optimization procedure), 데이터셋에 따라 결과가 달라지곤 했습니다. 또한, 연구 결과를 재현할 수 있는 코드가 드물었던 점을 강조합니다.

- **Technical Details**: 저자들은 기존의 범주형 커널을 포함하여 여러 테스트 케이스에서의 성능을 비교하고, 최적화 커뮤니티에서 영감을 받은 새로운 평가 지표를 제안합니다. 특히, 그룹 구조(group structure)가 있는 데이터셋에서는 중첩 커널(nested kernels) 방법이 다른 모든 방법들보다 우수한 성능을 보였습니다. 그룹 구조가 알려지지 않았거나 관련된 정보가 부족한 경우에는 범주형 변수(target encodings)를 이용한 클러스터링 기반 전략을 제안합니다.

- **Performance Highlights**: 대규모 데이터셋에 대한 실험 결과, 그룹 구조가 알려지지 않은 경우에도 제안된 추정 전략이 다른 방법들을 지속적으로 초월하면서 낮은 계산 비용(computational cost)을 유지함을 보여줍니다. 이는 다양한 테스트 환경에서의 커널 설계에 있어 중요한 기여를 하며, 더 나은 성능을 보장합니다.



### PRESOL: a web-based computational setting for feature-based flare forecasting (https://arxiv.org/abs/2510.01799)
- **What's New**: 이 논문에서는 태양 플레어 예측을 위한 새로운 웹 기반 기술 플랫폼인 'PRESOL'을 소개합니다. 이 플랫폼은 머신러닝(Machine Learning) 방법을 사용하여 플레어 발생 여부를 예측하고, 중요 특징(feature) 순위 정보 및 예측 성능 평가를 제공합니다. PRESOL은 태양 자기 이미지(magnetogram)에서 추출한 특징을 활용하여 플레어 발생을 예측하는 데 초점을 맞추고 있으며, 사용자 친화적인 인터페이스를 자랑합니다.

- **Technical Details**: PRESOL 파이프라인은 헬리오시즘 및 자력 이미저(Helioseismic and Magnetic Imager, HMI)에서 제공하는 SHARP(data) 피쳐를 사용하여 구성됩니다. 데이터셋은 방대한 양의 물리적 의미(significance)와 통계적 관련성을 가진 특징들로 구성되며, 총 25개의 설명자가 포함되어 있습니다. 이러한 특징들은 일반 정보, 전역 자기 파라미터, 에너지 및 불안정성 지표, 기하학적 및 형태적 파라미터 등 네 가지 범주로 나뉩니다.

- **Performance Highlights**: 예측 성능은 여러 기계 학습 알고리즘을 통해 검증되며, 각각의 알고리즘에 대한 성능 점수(scorings)도 포함되어 있습니다. PRESOL은 사용자 친화적인 웹 기반 플랫폼을 통해 결과를 시각화하고, 결과에 대한 사용자와의 상호작용을 가능하게 합니다. 이로 인해, 최근 12년간의 자료를 활용한 태양 플레어 예측 연구에서 중요한 진전을 이룰 수 있음을 보여줍니다.



### Secure Multi-Modal Data Fusion in Federated Digital Health Systems via MCP (https://arxiv.org/abs/2510.01780)
Comments:
          6 pages, 8 figures, 7 equations, 1 algorithm

- **What's New**: 이 연구는 이질적인 의료 데이터를 성과할 수 있는 안전하고 상호 운용 가능한 시스템을 구축하는 새로운 프레임워크를 소개합니다. 제안된 아키텍처는 세 가지 핵심 요소를 통합하는데, 이는 다중 모달 특성 정렬, 환자의 민감한 데이터를 보호하기 위한 안전한 집계 및 에너지 소비를 고려한 스케줄링입니다. 이를 통해 다양한 데이터 소스 간의 표준화된 상호 작용을 가능하게 하여, 다음 세대의 연합 건강 인프라를 위한 신뢰할 수 있는 경로를 제공합니다.

- **Technical Details**: 제안된 프레임워크는 Model Context Protocol (MCP)를 채택하여 안전한 다중 모달 연합 융합을 위한 새로운 방법론을 제시합니다. 이 방법론은 다양한 클라이언트에서 발생하는 업데이트를 통합하는 동시에 개인 정보 보호를 위해 미세 조정된 잡음 주입과 안전한 집계를 활용하여 민감한 데이터를 보호합니다. 또한, 에너지 효율적인 클라이언트 스케줄링 메커니즘이 포함되어 있어 모바일 헬스케어 클라이언트에서의 드롭아웃 비율을 줄이고 안정적인 참여를 보장합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 기존의 연합 학습에 비해 진단 정확도를 최대 9.8% 향상시키고 클라이언트 드롭아웃 비율을 54% 감소시키는 성과를 보였습니다. 또한, 개인 정보 보호와 효용 사이의 균형을 유지하며 임상적으로 수용 가능 범위 내에서 유용성을 지속할 수 있음을 입증했습니다. 이러한 결과는 MCP를 활용한 다중 모달 융합이 확장 가능하고 신뢰할 수 있는 의료 인프라로 나아가는 길이라는 것을 보여줍니다.



### Scalable Asynchronous Federated Modeling for Spatial Data (https://arxiv.org/abs/2510.01771)
- **What's New**: 이 논문에서는 분산된 공간 데이터의 연합 모델링(federated modeling)을 위한 비동기(asynchronous) 프레임워크를 제안합니다. 기존의 모델은 공간 의존성(spatial dependence)을 무시하거나 비동기 업데이트에 민감한 문제 일으킬 수 있습니다. 저자들은 저순위 가우시안 프로세스(low-rank Gaussian process)를 기반으로 하는 새로운 접근법을 통해 데이터 프라이버시를 유지하면서도 효과적으로 공간 데이터를 분석할 수 있는 방법론을 제시합니다.

- **Technical Details**: 제안된 방법론은 블록 최적화(block-wise optimization)와 경량 괄호화(gradient correction), 적응적 집합(adaptive aggregation) 및 안정화 업데이트(stabilized updates) 기법을 포함하여 비동기 학습에서 발생할 수 있는 두 가지 주요 도전을 해결합니다. 이를 통해, 다양한 현장 환경에서의 특징적인 문제들을 극복하고 시스템의 처리량을 증가시킵니다. 이 연구는 또한 명확한 느림성(staleness) 의존성을 가지고 선형 수렴(linear convergence)을 보장합니다.

- **Performance Highlights**: 수치 실험을 통해 제안된 비동기 알고리즘이 균형 잡힌 자원 할당 아래에서 동기 버전(synchronous version)과 유사한 성능을 달성함을 보여줍니다. 그러나 이 알고리즘은 계산 자원의 불균형이 큰 상황에서 동기 버전보다 현저하게 우수한 성능을 발휘했습니다. 이는 다양한 매개변수 설정, 샘플 크기 및 데이터 분할 방식에서의 강력한 견고성과 실용성을 보여 줍니다.



### Reducing Simulation Dependence in Neutrino Telescopes with Masked Point Transformers (https://arxiv.org/abs/2510.01733)
Comments:
          8 pages, 3 figures, presented at the 39th International Cosmic Ray Conference (ICRC2025)

- **What's New**: 이번 연구에서는 중성미자 망원경(neutrino telescopes)에서 시뮬레이션(data simulation) 데이터 의존도를 줄이기 위한 최초의 자기 지도 학습(self-supervised learning, SSL) 접근 방식을 소개합니다. 이 방법은 실제 데이터에 대한 훈련 비중을 높여 시뮬레이션과 관련된 불확실성을 최소화합니다. 이러한 변화를 통해 중성미자 이벤트의 재구성과 분류에서 실질적인 개선이 가능해집니다.

- **Technical Details**: 연구의 주요 기술적 요소는 포인트 클라우드(point cloud) 변환기(transformer)와 마스킹 자동 인코더(masked autoencoders)를 활용한 SSL 훈련 파이프라인입니다. 개발된 aN Efficient Point Transformer for Ultrarelastivistic Neutrino Events (neptune) 모델은 이벤트를 토큰화(tokenization)하고, 변환기 인코더(transformer encoder)를 통해 데이터의 특성을 추출하는 구조로 이루어져 있습니다. 각 이벤트의 전체 데이터를 취합하여 처리하고, 다양한 마스킹 기법을 적용하여 스포티라 템포럴(spatial-temporal) 좌표를 복구하는 방식으로 훈련을 실시합니다.

- **Performance Highlights**: 실험 결과, 보통의 지도 학습(supervised learning) 모델은 모델링되지 않은 효과에 대한 성능 저하를 보이는 반면, SSL 모델은 이러한 상황에서도 강력한 성능을 유지합니다. 이 연구는 실제 데이터에서 발생할 수 있는 예측할 수 없는 현상에 대해 SSL 방식이 효과적으로 대응할 수 있음을 보여줍니다. 이러한 방법론은 중성미자 물리학 분야에서 실질적인 발전 가능성을 열어줍니다.



### Contrastive Representation Regularization for Vision-Language-Action Models (https://arxiv.org/abs/2510.01711)
Comments:
          20 pages, 12 figures

- **What's New**: 본 논문에서는 로봇 조작을 위한 개선된 VLA(비전-언어-행동) 모델의 성능을 높이기 위해 Robot State-aware Contrastive Loss (RS-CL)를 도입합니다. RS-CL은 기존의 VLM(비전-언어 모델) 표현을 로봇의 신체 정보(proprioceptive states)와 정합시키기 위해 설계된 새로운 비지도 규제 목표입니다. 본 연구의 결과는 RS-CL이 기존의 기술 및 조작 성능을 상당히 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: RS-CL은 로봇의 신체 정보와 관련된 표현을 효과적으로 정제하기 위해 대비 학습(contrastive learning)의 원리를 활용합니다. 이 방법은 세밀하게 정의된 쌍을 통해 임베딩 공간을 구조화함으로써 로봇 조작 신호를 더 잘 포착하도록 안내합니다. 이러한 기법은 VLA 모델의 기존 훈련 파이프라인과 완벽하게 호환되면서도 경량화된 특징을 가지고 있습니다.

- **Performance Highlights**: RS-CL은 RoboCasa-Kitchen과 LIBERO와 같은 조작 벤치마크에서 VLA 모델의 성능을 크게 향상시키는 데 기여했습니다. 예를 들어, RS-CL을 적용하여 pick-and-place 다루기 작업에서 성과가 30.3%에서 41.5%로 향상되었으며, 실제 로봇 조작 과제에서도 45.0%에서 58.3%로 성공률이 증가했습니다. 이러한 실험 결과는 RS-CL이 복잡한 로봇 조작 환경에서도 효과적인 솔루션임을 입증합니다.



### Holistic Order Prediction in Natural Scenes (https://arxiv.org/abs/2510.01704)
Comments:
          25 pages, 11 figures, 6 tables

- **What's New**: 본 논문에서는 InstaFormer라는 새로운 네트워크를 제안합니다. InstaFormer는 RGB 이미지 하나만으로 씬의 모든 인스턴스에 대한 전체 압출(i.e., occlusion) 및 깊이(depth) 순서를 단일 전방 패스를 통해 반환할 수 있습니다. 이는 기존의 시스템들이 비싼 입력 형식과 높은 추론 비용에 의존했던 문제를 해결하고, 인스턴스 간의 상호작용을 통해 이루어집니다.

- **Technical Details**: InstaFormer는 객체 쿼리와 잠재적인 마스크 설명자 간의 상호작용을 통해 인스턴스 간의 기하학적 관계를 예측합니다. 제안된 모델은 점진적 입력 요구를 완화하며, 기존의 방식을 뛰어넘어 단일 전방 패스에서 전체 인스턴스의 기하학적 관계를 예측할 수 있는 기능을 갖추고 있습니다. 이 네트워크는 거리 level의 압출 및 깊이 예측 작업을 인접 행렬 수준의 문제로 재구성합니다.

- **Performance Highlights**: 모델의 성능은 다양한 벤치마킹을 통해 검증되었으며, 기존의 최상위 모델들과 동등하거나 이를 초월하는 결과를 보였습니다. InstaFormer는 RGB 입력 이미지 하나로 압출 및 깊이 순서 예측 작업에서 뛰어난 성능을 나타냅니다. 또한, 논문에서 제시된 코드와 모델은 오픈 소스로 제공되어, 더 많은 연구자들이 활용할 수 있습니다.



### VaPR -- Vision-language Preference alignment for Reasoning (https://arxiv.org/abs/2510.01700)
- **What's New**: 본 논문에서 우리는 기존의 방식들이 간과한 synthetic preference annotations의 노이즈 문제를 해결하기 위한 새로운 프레임워크인 VaPR을 소개합니다. VaPR은 스타일과 길이를 유지하면서 목표 오류를 가진 rejected responses를 생성하는 하드-네거티브(hard-negative) 응답 생성 방식을 기반으로 합니다. 이를 통해 3개의 LVLM 계열인 LLaVA-V1.5, Qwen2VL, Qwen2.5VL에 대해 30,000개의 고품질 샘플로 구성된 VaPR 데이터셋을 개발하였습니다.

- **Technical Details**: VaPR은 ground truth 응답과 생성된 하드-네거티브 응답을 쌍으로 구성하여, 태스크에서 부정확한 응답을 생성하도록 LLM(large language model)의 편집 능력을 활용합니다. 각 응답은 semantic 오류를 추가해 하드-네거티브 응답으로 구성되며, 기존 연구들과 달리 태스크에 맞춘 정보를 사용하여 잘못된 응답을 만들 때 스타일과 길이를 보존합니다. 이 방법은 기존의 VLM들보다 더 신뢰할 수 있는 문맥 이해 능력을 기반으로 하여, LVLMs의 비전-언어 정렬과 추론 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: VaPR 모델은 10개의 벤치마크에서 평균 6.5%(LLaVA), 4.0%(Qwen2VL), 1.5%(Qwen2.5VL)의 성능 향상을 이루었으며, 추론 작업에서 두드러진 결과를 보였습니다. LLaVA 모델은 적은 데이터로도 향상된 결과를 나타내며, VaPR은 이진 질문에서 '예'로 답할 경향을 줄이는 데도 효과적이었습니다. 또한, VaPR-OS 실험을 통해 오픈 소스 모델이 VaPR 프레임워크를 따르면서도 매우 유사한 성능을 내는 것을 확인했습니다.



### Evaluating the Robustness of a Production Malware Detection System to Transferable Adversarial Attacks (https://arxiv.org/abs/2510.01676)
- **What's New**: 이 논문은 기계 학습(ML) 모델의 특정 단점이 실제 세계에서 생산 시스템에 미치는 영향을 분석합니다. 특히, Gmail의 파일 유형 식별 시스템에서 ML 모델을 대상으로 하는 적대적 공격이 전체 악성코드 탐지 시스템을 어떻게 약화시키거나 우회할 수 있는지를 다루고 있습니다. 이러한 연구에서 제안된 공격 기법은 Magika라는 이름의 오픈 소스 모델을 기반으로 하며, 이는 이메일을 통한 악성 파일 전송을 가능하게 합니다.

- **Technical Details**: 이 논문에서는 적대적 공격의 강화(advantage)가 어떻게 특정 ML 부품을 목표로 삼아 전체 시스템을 위험에 빠뜨릴 수 있는지를 설명합니다. 연구자들은 Magika라는 ML 모델의 약점을 이용하여 파일의 13바이트만 변경함으로써 90% 사례에서 검출을 피할 수 있다는 것을 입증했습니다. 수정된 악성코드는 Gmail의 잘못된 악성 코드 분류기로 라우팅되어 탐지를 회피할 수 있도록 합니다.

- **Performance Highlights**: 이 연구에서는 Google 엔지니어와의 협력을 통해 구현된 방어 시스템을 통해 공격의 성공률을 현저하게 감소시킬 수 있었음을 보여줍니다. 방어가 적용된 모델에서는 고급 공격자가 50바이트를 수정해야 겨우 20%의 성공률을 달성할 수 있었으며, 이는 생산 환경에서 ML 시스템의 강인성을 개선하기 위한 중요한 발전을 나타냅니다. 최종적으로, 오픈 소스 기계 학습 시스템의 이점과 단점에 대해 논의하면서 보안 연구자들이 실제 공격 시나리오를 연구하는 데 드는 필요성을 강조합니다.



### Just Do It!? Computer-Use Agents Exhibit Blind Goal-Directedness (https://arxiv.org/abs/2510.01670)
- **What's New**: 이 논문은 컴퓨터 사용 에이전트(CUAs)의 Blind Goal-Directedness (BGD) 현상을 분석하며, 이는 에이전트가 목표 관점에서만 행동하여 잠재적으로 해로운 결과를 초래할 수 있음을 보여줍니다. BGD는 세 가지 주요 패턴으로 구분되며, 이를 체계적으로 평가하기 위해 90개의 작업으로 구성된 BLIND-ACT 벤치마크를 개발했습니다.

- **Technical Details**: BLIND-ACT는 OSWorld 위에 구축된 실제적이고 동적인 데스크톱 환경에서의 실행을 지원하며, 다양한 어플리케이션과 시스템 기능에서 BGD 행동이 발생할 수 있도록 설계되었습니다. 벤치마크는 90개의 작업으로 구성되어 있으며, 각 작업은 BGD의 세 가지 패턴을 포괄합니다. LLM 기반의 판단자를 사용해 에이전트가 BGD 행동을 보이는지와 비효율적인 행동을 실행하는지를 평가합니다.

- **Performance Highlights**: BLIND-ACT를 활용하여 Claude Sonnet, Opus 4 및 GPT-5와 같은 9개의 최신 모델을 평가한 결과, 평균 80.8%의 BGD 비율을 관찰했습니다. 작은 모델들은 과도하게 안전한 것처럼 보이나 이는 제한된 능력 때문이며, 결과적으로 안전과 능력 간의 패러독스가 강화됩니다. 또한, 프롬프트 기반의 개입이 BGD 수준을 낮출 수 있지만 여전히 상당한 위험이 남아 있음을 보여줍니다.



### Position: Privacy Is Not Just Memorization! (https://arxiv.org/abs/2510.01645)
Comments:
          27 pages, 6 figures, 2 tables

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLM) 시스템의 개인 정보 보호 위험에 대한 쟁점을 다루고 있습니다. 주목할 점은 기존의 논의가 훈련 데이터의 단순한 암기(memoration)에만 집중되어 있는 반면, 실제로는 더 즉각적이고 확장 가능한 다양한 개인 정보 보호 위협이 존재한다는 것입니다. 저자들은 LLM 시스템의 전체 주기로부터 발생하는 개인 정보 보호 위험을 포괄적으로 분류하여 설명하며, 현재의 개인 정보 보호 프레임워크가 이 위협들에 제대로 대응하지 못하고 있음을 강조합니다.

- **Technical Details**: 논문에서는 LLM 생태계에서 유출되는 데이터의 세 가지 유형(사용자 상호 작용 데이터, 시스템 검색 데이터 및 공개 데이터)을 제시합니다. 이러한 데이터 유형은 서로 작용하여 훈련 데이터 유출을 넘어서는 다섯 가지 특정한 개인 정보 유출 사건을 생성하며, 각 사건 유형이 다양한 위협을 제시합니다. 예를 들어, LLM은 무의식적으로 사용자 입력으로부터 민감한 정보(attribute)를 추론하고, 이를 통해 개인정보를 침해할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 연구에 따르면, AI/ML 개인 정보 보호 분야의 논문 중 92%가 훈련 데이터 메모리화(memoration) 및 직접적인 채팅 유출 방지에 초점을 맞추고 있으며, 다른 유형의 사건은 연구의 8% 미만을 차지합니다. 저자들은 데이터 최소화(local data minimization), 하이브리드 아키텍처 및 프라이버시 중심의 후속 훈련(post-training)과 같은 기술적 개입이 필요하다고 강조합니다. 이는 개별 사용자에게 힘을 실어줄 수 있는 사회기술적 접근(sociotechnical approaches)과 불균형을 해결할 수 있는 정책 개선을 필요로 합니다.



### ImageNet-Think-250K: A Large-Scale Synthetic Dataset for Multimodal Reasoning for Vision Language Models (https://arxiv.org/abs/2510.01582)
Comments:
          Preprint

- **What's New**: 본 논문에서는 Vision Language Models (VLMs)의 명시적 추론 능력을 개발하기 위한 멀티모달 추론 데이터셋인 ImageNet-Think를 제안합니다. 이 데이터셋은 ImageNet21k에서 250,000개의 이미지를 기반으로 하여 구조화된 사고 토큰과 관련된 답변을 제공합니다. GLM-4.1V-9B-Thinking과 Kimi-VL-A3B-Thinking-2506 두 가지 최신 VLM에서 생성된 이 합성 데이터셋은 모델의 훈련 및 평가에 유용한 자원으로 활용될 것입니다.

- **Technical Details**: ImageNet-Think 데이터셋은 250,000개의 이미지에 대해 각 이미지에 대해 2쌍의 사고-답변 시퀀스를 포함하고 있습니다. 이러한 구조화된 사고 토큰은 GLM-4.1V-Thinking과 Kimi-VL-Thinking 모델에 의해 생성되며, 다양한 추론 패턴을 캡처하여 VLM 훈련에 기여하는 다양성을 제공합니다. 또한, 이 데이터셋은 기존의 VLM 훈련 데이터셋들이 가지고 있는 제한적인 스코프 문제를 해결합니다.

- **Performance Highlights**: ImageNet-Think는 500,000개의 사고-답변 쌍을 제공하여 대규모 공개 데이터셋으로서 중요한 역할을 할 것입니다. 또한, 다양한 평가 기준을 통해 여러 VLM의 성능을 벤치마킹하며, 연구자들이 해석 가능한 신뢰할 수 있는 VLMs를 훈련할 수 있도록 지원합니다. 이 데이터셋과 평가 기준은 멀티모달 추론 연구의 발전에 기여할 것입니다.



### Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomp (https://arxiv.org/abs/2510.01574)
Comments:
          Accepted to the Proceedings of the ACM SIGIR Asia Pacific Conference on Information Retrieval (SIGIR-AP 2025), December 7-10, 2025, Xi'an, China

- **What's New**: 이번 연구에서는 실시간 신경 쿼리 자동 완성 시스템에서 프레젠테이션 편향(presentation bias)을 완화하기 위한 데이터 중심 접근 방식을 제안합니다. 본 방법론은 사용자의 전체 쿼리를 바탕으로 합성 접두사를 생성하여 훈련 데이터를 다양하고 비편향적인 예제들로 풍부하게 하여 학습합니다. 이러한 접근은 기존의 사용자의 상호작용 데이터에서 발생하는 편향을 줄이는데 도움을 줍니다.

- **Technical Details**: 신경 순위 모델은 엄격한 지연(latency) 제약 하에서 실시간 배포를 위해 최적화되었습니다. 이는 쿼리 인기도(query popularity), 계절성(seasonality), 유사도 점수(fuzzy match scores) 등을 포함한 다양한 특성을 통합합니다. 우리는 쿼리 자동 완성 구조를 이용하여 O(n^2)에서 O(n)으로 계산 복잡성을 줄인 리스트와이즈 손실(listwise loss)의 간소화 버전을 도입하였습니다.

- **Performance Highlights**: 대규모 전자상거래 환경에 배포된 이 시스템은 사용자 참여를 통계적으로 유의미하게 개선하는 결과를 보여주었습니다. A/B 테스트 결과, 우리의 모델은 기존의 선형 바닥선(linear baseline) 보다 MRR(Mean Reciprocal Rank)을 1% 이상 향상시켰습니다. 이러한 결과는 균형 잡힌 데이터와 최적화된 손실 함수를 통해 훈련된 신경 LTR 모델의 효과성을 입증합니다.



### AI Foundation Model for Time Series with Innovations Representation (https://arxiv.org/abs/2510.01560)
- **What's New**: 이번 논문은 공학 응용을 위한 인공지능(AI) 기반 시간 시리즈 모델인 Time Series GPT (TS-GPT)를 소개합니다. TS-GPT는 물리적 법칙에 기반한 공학 시간 시리즈를 위한 혁신 표상 모델이며, 이는 기존의 대형 언어 모델 기반 모델이 효과적이지 않을 수 있다는 점을 강조합니다. 이 모델은 조건부 확률 분포로부터 미래의 시간 시리즈 샘플을 생산하는 확률적 생성 예측을 가능하게 하여 실시간 가격 예측에 사용될 수 있습니다.

- **Technical Details**: TS-GPT는 Wiener, Kallianpur, Rosenblatt의 혁신 표현 이론을 기반으로 한 생성 사전 훈련 변환기(Generative Pre-trained Transformer)입니다. 이 모델은 제어 및 모니터링을 위한 공학적 요구 사항을 충족하기 위해 설계되었으며, 혁신 표현(autoencoder) 구조를 갖춥니다. 특히, TS-GPT는 실시간 의사 결정 문제를 처리하기 위해 시간이 지남에 따른 의존성을 포착하는 주의(attention) 메커니즘을 포함하고 있습니다.

- **Performance Highlights**: 논문에서 제시된 TS-GPT는 미국 독립 시스템 운영자들로부터 역사적 데이터를 사용하여 실시간 지역 한계 가격 예측에서 효율성을 입증했습니다. 이와 같은 예측 모델은 물리적 규모의 시스템을 위한 물리적 법칙을 따르는 실시간 데이터 기반 의사 결정 문제를 해결하는 데 유용할 수 있습니다. TS-GPT의 능력은 공학 및 제어 시스템에서의 실제 적용 가능성을 높여줄 것입니다.



### CardioRAG: A Retrieval-Augmented Generation Framework for Multimodal Chagas Disease Detection (https://arxiv.org/abs/2510.01558)
Comments:
          4 pages, 2 figures. Accepted for oral presentation at the 52nd international Computing in Cardiology Conference (CinC2025)

- **What's New**: 이번 연구에서는 Chagas 질병 진단을 위한 새로운 AI 프레임워크인 CardioRAG를 제안합니다. 기존의 머신러닝 접근법의 한계를 극복하고, 해석 가능한 ECG 기반의 임상 특징과 대형 언어 모델을 통합하여 진단 추론을 향상시킵니다. 이는 자원이 제한된 환경에서도 Chagas 질병을 효과적으로 선별할 수 있는 스마트한 체계를 제공합니다.

- **Technical Details**: CardioRAG 프레임워크는 ECG 신호에서 임상 특징을 추출하고, VAE 기반 표현 학습을 통해 유사한 사례를 검색하며, RAG를 활용한 결정적 진단을 수행합니다. 12리드 ECG와 환자의 인구통계학적 데이터를 사용하여 세 가지 주요 단계로 구성됩니다: 진단 특성 추출, 유사도 검색, 그리고 진단 결정. 이러한 방법론은 특히 Chagas 질병 진단은 위한 사례 기반 추론을 중시하여, 임상의의 진단 결정을 지원합니다.

- **Performance Highlights**: CardioRAG 프레임워크는 높은 진단 성능을 보이며, 58.59% 정확도 및 87.76% 재현율을 기록했습니다. 이는 간결한 프롬프트 디자인이 작은 언어 모델에서 더 효과적이라는 것을 보여줍니다. 사례 추출 전략이 진단 성능에 미치는 영향도 상당했으며, 과도히 보수적인 프롬프트는 성능을 저하시켰습니다.



### Robust Classification of Oral Cancer with Limited Training Data (https://arxiv.org/abs/2510.01547)
- **What's New**: 이 논문에서는 오랄 암(oral cancer)의 조기 진단을 위해 신뢰성 있는 하이브리드 모델을 제안합니다. 이는 합성곱 신경망(CNN)과 베이지안 딥러닝(Bayesian deep learning)을 결합하여 소규모 교육 세트(small training sets)에서 오랄 암을 분류하는 것입니다. 전통적인 CNN 모델이 일반적으로 데이터 양에 의존하는 한계를 극복하고, 데이터 부족 환경에서도 우수한 일반화 성능을 제공하도록 설계되었습니다.

- **Technical Details**: 제안된 모델은 변분 추론(variational inference)을 통해 예측의 신뢰성(sentiment reliability)을 높이며, 스마트폰으로 촬영한 사진 색상 이미지(color images)를 기반으로 훈련되었습니다. 이 모델은 세 가지 서로 다른 테스트 데이터 세트에서 평가되었으며, 제한된 데이터에서도 높은 정확도로 예측할 수 있도록 설정되어 있습니다. 94%의 정확도를 달성한 이 모델은 트레이닝 데이터와 유사한 분포에서도 높은 신뢰도를 보였습니다.

- **Performance Highlights**: 제안된 베이지안 모델은 다양한 데이터 세트에서 88%의 정확도를 달성하며 전통적인 CNN의 72.94%와 비교하여 우수한 일반화 능력을 보였습니다. 또한, 예측 정확성이 높은 샘플에 대해서는 낮은 불확실성(low uncertainty)을 나타내고, 잘못 분류된 샘플에 대해서는 높은 불확실성(high uncertainty)을 보여줍니다. 이러한 결과는 베이지안 추론이 데이터가 부족한 환경에서도 조기 오랄 암 진단을 개선하는 데 효과적임을 강조합니다.



### Growing Visual Generative Capacity for Pre-Trained MLLMs (https://arxiv.org/abs/2510.01546)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 Bridge라는 순수 자가 회귀(unified autoregressive) 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLM)을 제안합니다. 기존의 MLLM들은 비주얼 이해만 가능했지만, Bridge는 이미지 이해 및 생성을 단일한 다음 토큰 예측(next-token prediction) 프레임워크 내에서 수행할 수 있도록 해줍니다. 또한 semantic-to-pixel 이산 표현(discrete representation)을 도입하여 시각적 생성을 개선했습니다.

- **Technical Details**: Bridge는 사전 훈련된 비주얼 인코더와 Mixture-of-Transformers 아키텍처를 통해 이미지의 이해와 생성을 결합합니다. 이 모델은 compact한 의미 토큰과 정밀한 픽셀 토큰을 통합하여 언어 정렬(strong language alignment) 및 시각적 세부 사항의 정확한 설명을 가능하게 합니다. 데이터 사용 및 전체 훈련 절차도 상세히 설명되었습니다.

- **Performance Highlights**: 다양한 다중 모달 벤치마크에서 광범위한 실험을 통해 Bridge는 이해 및 생성 벤치마크에서 경쟁력 있는 성과 또는 우수한 성과를 달성하였습니다. 이전의 통합 MLLM보다도 적은 학습 데이터와 짧은 훈련 시간으로 효과성을 입증하며, 총 훈련 효율성도 향상되었습니다.



### Towards Interpretable and Inference-Optimal COT Reasoning with Sparse Autoencoder-Guided Generation (https://arxiv.org/abs/2510.01528)
- **What's New**: 이번 연구에서는 희소 자동인코더(SAE)와 클러스터링 기법을 활용해 대형 언어 모델(LLMs)의 내부 토큰 표현을 분석하고 수학적 추론 과제를 위한 생성을 유도하는 새로운 방법을 제안합니다. SAE를 이용해 훈련 토큰의 희소 벡터 표현을 생성한 후, k-평균 클러스터링을 적용하여 토큰 클러스터를 나타내는 그래프를 구성합니다. 이 그래프를 통해 기존의 추론 경로에 대한 보상을 정량화하고, 탐색 정도를 측정하는 기법을 개발했습니다.

- **Technical Details**: SAE 훈련 과정에서 입력 토큰 표현을 고차원 잠재 공간으로 매핑하며, reconstruction loss를 최소화하는 것이 목표입니다. 연결된 단계에서 k-평균 클러스터링을 통해 클러스터 다양성을 측정하고, 이는 탐험의 정도를 평가하는데 유용합니다. 이 과정에서 도출된 보상 함수는 생성 과정의 중간 단계에서 적절한 보상을 할당하는 데 기여하며, 이는 생성의 견고성을 높입니다.

- **Performance Highlights**: 연구 결과, 탐색과 착취의 균형을 맞추는 것이 수학적 추론 작업에서 높은 정확도를 달성하는 데 중요한 요소로 확인되었습니다. 또한, SAE를 통한 고급 생성 유도 방식이 모델의 생성 품질을 향상시키는 데 효과적임을 입증했습니다. 이 연구는 SAE의 풍부한 표현을 활용하여 새로운 자동화된 보상 모델을 개발함으로써 LLM의 효율성을 높일 수 있는 가능성을 보여줍니다.



### WALT: Web Agents that Learn Tools (https://arxiv.org/abs/2510.01524)
- **What's New**: 본 논문은 웹 에이전트가 복잡한 브라우저 작업을 자동화할 수 있는 가능성을 제시하며, 기존 방법의 한계를 극복하기 위해 WALT(Web Agents that Learn Tools)라는 새로운 프레임워크를 소개합니다. WALT는 웹사이트에서 제공하는 기능을 역탐색하여 재사용 가능한 도구로 변환하는 방식으로 작업을 진행합니다. 기존의 UI 상호작용에 의존하는 방법이 아닌, 고수준의 기능 호출을 통해 에이전트의 계산 부담을 경감시킵니다.

- **Technical Details**: WALT는 세 가지 주요 단계를 통해 웹 도구를 학습합니다: (1) 웹 에이전트가 웹사이트의 기능을 시연하고, (2) 도구 생성 에이전트가 이를 구조화된 도구로 변환하며, (3) 테스트 에이전트가 기능을 검증합니다. 이 과정에서 에이전트는 복잡한 UI 시퀀스를 고려하는 대신, search(X)와 같은 단순한 함수 호출로 기능을 실행합니다. 도구들은 검색, 필터링, 콘텐츠 관리와 같은 작업을 포함하며, 50개 이상의 재사용 가능한 도구가 발견되었습니다.

- **Performance Highlights**: WALT는 VisualWebArena와 WebArena에서 각각 52.9%와 50.1%의 성공률을 기록하여, 이전 연구들보다 월등한 성능을 보였습니다. 추가적인 연구에서는 WALT가 발견한 도구, 다중 모달 DOM 파싱, 외부 검증이 성공률을 10%-30% 향상시키고, 평균 1.3-1.4배 더 적은 단계를 요구함을 밝혔습니다. 결과적으로 WALT는 브라우저 자동화를 보다 효율적이고 신뢰성 있는 도구 기반 접근으로 전환시킵니다.



### Aligning Video Models with Human Social Judgments via Behavior-Guided Fine-Tuning (https://arxiv.org/abs/2510.01502)
Comments:
          15 pages total, 4 figures. Includes 1 algorithm and 2 tables in the appendix

- **What's New**: 이번 연구에서는 최신 비디오 및 언어 모델이 인간이 인지하는 유사성을 어떻게 캡처하는지를 조사합니다. 기존의 상태-of-the-art 모델들이 사회적 비디오에서 인간의 인지적 유사성을 재현하지 못하는 한계를 발견하고, 이 문제를 해결하기 위한 새로운 데이터셋과 방법론을 제안합니다. 특히, 49,484개의 'odd-one-out' 유사도 판단이 포함된 대규모 데이터셋을 도입하여 비디오 모델의 개선을 위한 행동 기반의 미세 조정 방법을 제안합니다.

- **Technical Details**: 연구는 비디오 및 언어 모델 각각의 유사도 판단을 분석하기 위해 triplet OOO 비교를 수행합니다. 새로 도입된 데이터셋은 3초 길이의 비디오 클립 250개로 구성되어 있으며, 각 클립은 사회적 상호작용을 묘사합니다. 우리는 하이브리드 손실 함수인 triplet 손실과 representational similarity analysis (RSA) 손실을 결합하여 비디오 모델의 임베딩을 더욱 인간 인지 구조에 가깝게 조정합니다.

- **Performance Highlights**: 시간 기반 비디오 모델인 TimeSformer의 미세 조정을 통해 훈련 전 대비 예측된 변동성이 58% 증가하였고, 이는 인간의 판단과 높은 일치를 나타냅니다. 또한, 모델이 언어 임베딩과의 겹침을 크게 증가시켰으며, 언어 모델이 포착하지 못하는 추가적인 변동성을 설명할 수 있음을 보여줍니다. 최종적으로, 사회 정서적 속성(친밀감, 정서적 가치, 지배력 등)의 인코딩이 강화되었음을 확인하였습니다.



### Purrception: Variational Flow Matching for Vector-Quantized Image Generation (https://arxiv.org/abs/2510.01478)
- **What's New**: 본 논문에서는 Purrception이라는 새로운 방법을 소개합니다. 이것은 벡터 양자화된 이미지 생성에 대한 변량 흐름 일치(Variational Flow Matching) 접근법을 사용하여 명시적인 범주적(supervision) 감독을 제공하면서도 연속적(continuous) 운반 역학을 유지합니다. Purrception은 이미지 생성의 효율성을 향상시키기 위해 연속적 방법의 기하학적 인식과 범주적 접근의 불연속 행위를 결합합니다.

- **Technical Details**: Purrception은 코드북 인덱스에 대한 범주적 사후 확률(categorical posteriors)을 학습하면서 연속 임베딩 공간에서 속도 필드(velocity fields)를 계산합니다. 이를 통해 잠재 정보의 불확실성(uncertainty)을 정량화하고 온도 조절이 가능한 생성(generation)을 가능하게 합니다. Purrception은 이러한 특정 선을 통해 이미지 생성 특성의 연속적 심도를 더욱 향상시킵니다.

- **Performance Highlights**: ImageNet-1k에서 256x256 해상도의 이미지를 생성하는 과정을 평가한 결과, Purrception은 연속적 흐름 일치 및 불연속적 흐름 일치 방법과 비교했을 때 빠른 학습 수렴을 보여 주었습니다. 또한, 경쟁력 있는 FID 스코어를 달성하여 최신 모델들과 비교해 우수한 성능을 입증했습니다.



### Comparative Field Deployment of Reinforcement Learning and Model Predictive Control for Residential HVAC (https://arxiv.org/abs/2510.01475)
Comments:
          27 pages, 11 figures, 4 tables. Under review for Applied Energy

- **What's New**: 이 연구는 주거 환경에서 모델 예측 제어(Model Predictive Control, MPC)와 강화 학습(Reinforcement Learning, RL) 제어기를 직접 비교한 첫 번째 실제 배치 연구입니다. 연구팀은 인디애나주 웨스트 라파예트에 위치한 주거용 집에서 각 제어기를 한 달 동안 배치하여 두 접근법의 성능 및 안전성 측면에서의 차이를 평가했습니다. 이 연구는 RL이 자동화 및 적응성을 제공하면서도, 동시에 실제 적용 시의 안전성 문제를 해결하는 데 있어 발생하는 트레이드오프를 조명합니다.

- **Technical Details**: MPC는 건물의 열역학적 동적 모델을 사용하여 미래 상태를 예측하고 제어 행동을 최적화하는 접근법입니다. 반면, RL은 환경과의 상호작용을 통해 학습된 정책을 사용하여 자동으로 제어 전략을 발견하는 데이터 기반 접근법입니다. 연구에서 사용된 Ibex-RL은 MPC와 유사한 물리적 시스템 동적 모델을 자동으로 학습하며, 최소한의 사용자 안내로 복잡한 보상 함수를 학습하도록 설계되었습니다.

- **Performance Highlights**: RL은 기존 제어기 대비 22%의 에너지 절약을 달성했으며, MPC는 20%의 절약 효과를 보였습니다. 그러나 편안함 수준을 고려했을 때 MPC는 더 우수한 성능을 나타냈습니다. 연구 결과, RL은 엔지니어링 오버헤드를 줄이는 데 기여하지만 모델 정확도와 운영 견고성에서 실질적인 트레이드오프를 초래함을 보여주었습니다.



### A-VERT: Agnostic Verification with Embedding Ranking Targets (https://arxiv.org/abs/2510.01469)
Comments:
          19 pages, 7 figures, code available at this https URL, authors in alphabetical order

- **What's New**: 본 연구에서는 Language Model (LM) 응답을 평가하는 자동화된 방법의 필요성을 강조하고 있습니다. 기존 접근 방식들은 비용이 높거나(judge 사용) 현실과 동떨어져 있었던 반면, 제안된 구조 없는 평가 방법은 의미적 임베딩 거리(semantic embedding distance)를 활용하여 응답의 우수성을 저렴한 계산 비용으로 평가한다고 설명합니다. 이 방법을 통해 수행한 실험에서 회귀 점수(regression score) 약 ~0.97과 인간 주석자와의 정확도 약 ~96%를 달성했습니다.

- **Technical Details**: 제안된 방법은 LM이 생성한 임의의 텍스트와 대상 후보(target candidates)를 비교하기 위해 의미 임베딩(semantic embedding) 기술을 사용합니다. 상대적으로 파라미터 수가 $10B$ 미만인 임베딩 모델을 사용하여 낮은 계산 비용으로 강력한 분류를 제공합니다. 이 연구는 서로 다른 3개의 데이터 세트와 3가지 LM 아키텍처에서 테스트하여 성능을 비교하였습니다.

- **Performance Highlights**: 제안된 구조 없는 평가 방법은 기존의 자원 소모가 큰 방법과 비교하여 월등한 정확성을 나타냅니다. 특히, 회귀 점수는 ~0.97로 높았으며, 정확도는 ~96%에 달하였습니다. 이는 다양한 QA 작업의 자동 평가를 통해 우수한 성능을 얻게 된 것으로 여겨집니다.



### Data Selection for Fine-tuning Vision Language Models via Cross Modal Alignment Trajectories (https://arxiv.org/abs/2510.01454)
Comments:
          30 pages, 10 figures, 5 tables, link: this https URL

- **What's New**: 이번 연구에서는 LVLMs(대형 비전-언어 모델)에 대한 데이터 효율적인 훈련 방법을 최초로 제안합니다. 기존의 방법들이 다양한 서브셋 크기에서 무작위 선택보다 성능이 떨어진다는 점에서, 저자들은 이 문제를 해결하기 위해 기울기 유사성을 정의하고 설명합니다. 이를 통해 XMAS라는 알고리즘을 개발하여 주어진 데이터를 클러스터링하고, 중복성을 줄이며 성능을 유지할 수 있음을 입증했습니다.

- **Technical Details**: 저자들은 각 샘플의 기울기 유사성을 기반으로 한 중복 제거를 위해, 단일 레이어 변환기를 분석하고 교차 모달(attention matrices) 어텐션 행렬 간의 쌍별 거리로 기울기 거리를 근사할 수 있음을 증명하였습니다. XMAS는 소규모 프록시 VLM을 미세 조정하여 기울기 유사성을 통해 예제를 군집화합니다. 이를 통해 샘플 그룹에서 균형 잡힌 서브셋을 샘플링하여 큰 훈련 데이터에서 중복성을 제거합니다.

- **Performance Highlights**: 실험 결과, XMAS는 LLaVA-665k 데이터의 50%와 Vision-Flan 데이터의 85%를Discard하면서도 LLaVA-1.5-7B의 성능을 완벽하게 유지했습니다. 또한, 10개의 하위 벤치마크에서 훈련 속도가 1.2배 빨라졌으며, 이는 LLaVA-665k의 최상의 기준선과 비교했을 때 30% 더 많은 데이터 감소를 보였습니다.



### Financial Stability Implications of Generative AI: Taming the Animal Spirits (https://arxiv.org/abs/2510.01451)
- **What's New**: 이 논문은 생성형 AI가 금융 안정성에 미치는 영향을 조사합니다. 대규모 언어 모델을 이용해 기존의 거래 결정에서의 군집 행동(Herd Behavior)에 대한 연구를 재현하는 실험을 수행하였습니다. 이를 통해 AI 에이전트가 인간보다 더 합리적인 의사결정을 내린다는 결과를 얻었습니다.

- **Technical Details**: AI 에이전트는 주로 시장 트렌드보다 개인 소스를 기반으로 판단합니다. AI 기반의 거래 조언에 대한 의존도가 높아질 경우, 인간의 군집 행동에 의해 발생하는 자산 가격 버블(asset price bubbles)의 발생을 줄이는데 기여할 수 있습니다. 그러나 실험 설정의 변화를 통해, AI 에이전트가 수익을 극대화하는 결정을 내리도록 유도되면 최적의 군집 행동을 할 수 있음을 확인하였습니다.

- **Performance Highlights**: 최적의 군집 행동은 시장 규율(market discipline)을 향상시키지만, 이는 여전히 금융 안정성에 잠재적인 영향을 미칠 수 있습니다. 추가 실험을 통해 우리는 AI 에이전트가 순수 알고리즘에 그치지 않고 인간의 조건화와 편향(bias)의 일부 특성을 물려받았음을 보여주었습니다.



### VOGUE: Guiding Exploration with Visual Uncertainty Improves Multimodal Reasoning (https://arxiv.org/abs/2510.01444)
- **What's New**: 이 논문에서는 Visual-Uncertainty-Guided Exploration (VOGUE)이라는 새로운 방법을 소개하며, 이 방법은 멀티모달 대형 언어 모델(MLLM)의 탐색 문제를 해결하는 데 초점을 맞추고 있습니다. 전통적인 방법들이 이미지를 고정된 조건으로 취급하는 대신, VOGUE는 이미지를 확률적 맥락으로 간주하고 시각적 변동성을 측정함으로써 더 나은 탐색 방향을 제시합니다. 이 방법을 통해 시각적 불확실성을 기반으로 한 학습 목표의 재조정이 가능해 여태까지 간과되었던 비밀스러운 탐색 경로를 유도합니다.

- **Technical Details**: VOGUE는 훈련 시 두 가지 가지(branch)를 활용하는 이중 가지 전방 패스를 통해 시각적 불확실성을 정량화합니다. 원본 이미지와 의미가 보존된 퍼터베이션(perturbation)된 이미지를 모두 사용하여 정책의 민감도를 계산하고, 비슷한 방식으로 KL 발산(KL divergence)을 적용하여 탐색 우위를 조형합니다. 이러한 시각적 불확실성은 불확실성 비례 보너스로 연결되어, 초기 훈련에서는 탐색에 중점을 두고 훈련이 안정화되면 원본 이미지로 초점을 옮기는 방식으로 진행됩니다.

- **Performance Highlights**: VOGUE는 GRPO 구현 하에 Qwen2.5-VL-3B 및 7B 모델에서 세 개의 시각 수학 벤치마크와 세 개의 일반 도메인 추론 벤치마크에서 각각 평균 2.6% 및 3.7%의 pass@1 정확도를 증가시키며, 일반적으로 발견되는 RL 미세 조정에서의 탐색 감소를 효과적으로 완화하였습니다. 또한, VOGUE는 텍스트 전용 설정에서 효과를 보이는 Pass@k Training 방법보다 지속적으로 우수한 성능을 발휘하여, 높은 pass@1 및 일관된 pass@k 향상을 달성했습니다.



### Risk Phase Transitions in Spiked Regression: Alignment Driven Benign and Catastrophic Overfitting (https://arxiv.org/abs/2510.01414)
- **What's New**: 이 논문은 spiked covariance 데이터 모델을 사용하여 최소-노름(interpolating) 솔루션의 일반화 오류(generalization error)를 분석합니다. 논문에서는 스파이크(spike) 강도와 목표 스파이크(target-spike) 정렬이 위험에 미치는 영향을 설명하며, 특히 과잉 매개변수가 있는 설정에서의 결과를 강조합니다. 이에 따라 benign, tempered, catastrophic 과적합(overfitting) 상황을 구분하는 상세한 분류가 제시됩니다.

- **Technical Details**: 이 연구에서는 최소 제곱(linear regression) 선형 회귀에서 일반적인 스파이크 크기(spike size)와 목표 정렬(target alignment)이 일반화 오류에 미치는 영향을 살펴봅니다. 피쳐 공분산(feature covariance)은 특정 스파이크 방향을 모델링하고, 스파이크와 이소트로픽(noise) 노이즈 컴포넌트를 통해 이론적 기틀을 구성합니다. 또한, $c=d/n$ 비율의 비율적 레짐(asymptotic proportional regime)에서의 목표 신호 정렬(target signal alignment)이 일반화를 어떻게 개선 또는 저해하는지를 질문으로 제기합니다.

- **Performance Highlights**: 이 논문은 최소-노름(interpolating) 솔루션의 일반화 성능을 정확하게 특성화합니다. 연구 결과에 따르면, 강력한 스파이크가 존재할 경우, 잘 규정된 문제에서조차 catastrophic 과적합으로 이끄는 경우가 있으며, 잘못 규정된 문제는 명확한 전환을 보여 갑작스러운 일반화 실패를 초래할 수 있습니다. 이를 통하여, 스파이크 정렬은 항상 유리하지 않으며, 특정 조건 하에서만 유익할 수 있다는 사실을 밝혀냈습니다.



### INSIGHT: INference-time Sequence Introspection for Generating Help Triggers in Vision-Language-Action Models (https://arxiv.org/abs/2510.01389)
- **What's New**: 최근 Vision-Language-Action (VLA) 모델은 강력한 일반화 능력을 보여주지만, 오류를 예측하고 인간 감독자에게 도움을 요청하는 내적 메커니즘이 부족합니다. 이 논문에서는 VLA가 도움을 요청할 시점을 예측하기 위해 토큰 수준의 불확실성 신호를 활용하는 학습 프레임워크인 	extbf{INSIGHT}를 제안합니다. 이 새로운 접근 방식은 보다 나은 인간-기계 협업을 위한 첫 걸음을 내딛는 것입니다.

- **Technical Details**: 이 연구는 $	extpi_0$-FAST를 기반 모델로 사용하며, 각 토큰에 대해 	extit{entropy}, 	extit{log-probability}, 그리고 Dirichlet 기반의 	extit{aleatoric과 epistemic uncertainty}로 불확실성을 추정하여 긴축된 transformer 분류기를 훈련하여 도움 트리거를 매핑합니다. 강한 감독과 약한 감독의 다양한 학습 방식도 탐구하며, 시계열적으로 진화하는 토큰 수준 불확실성 신호를 모델링함으로써 예측력을 높이는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 연구 결과, 강한 레이블이 모델이 정밀한 불확실성 동역학을 포착하는 데 도움을 주어 신뢰할 수 있는 도움 탐지가 가능하게 하며, 약한 레이블은 더 시끄럽지만 여전히 경쟁력 있는 내적 검토를 지원할 수 있음을 보여줍니다. 이로 인해 훈련과 평가가 정렬될 때, 촘촘한 주석이 비현실적인 상황에서도 확장 가능한 경로를 마련할 수 있습니다. 특히, 토큰 수준의 불확실성 신호를 모델링하는 방식이 정적 시퀀스 수준의 점수보다 훨씬 더 높은 예측 파워를 제공한다는 것을 발견하였습니다.



### Learning to Play Multi-Follower Bayesian Stackelberg Games (https://arxiv.org/abs/2510.01387)
- **What's New**: 이번 연구에서는 다수의 추종자가 있는 Bayesian Stackelberg 게임에서 선도자가 온라인 학습(online learning) 방식으로 최적 전략을 찾는 문제를 다룹니다. 선도자는 매 라운드마다 독립적인 분포에서 추종자들의 타입(type)을 샘플링하여 원하는 결과를 도출하고자 합니다. 연구의 주된 목표는 누적 유틸리티(cumulative utility)와 실제 선택된 전략 간의 차이를 최소화하는 것입니다.

- **Technical Details**: 선도자는 총 $T$ 라운드 동안 $n$명의 추종자와 상호작용하며, 각각의 추종자는 K개의 타입 중 하나를 가집니다. 타입 피드백(type feedback)에서는 추종자의 타입을 관찰할 수 있으며, 이 경우 알고리즘이 독립적인 타입 분포에서 $	ext{O}(rac{	ext{sqrt}(L	ext{log}(nKA T)) 	ext{T}}{	ext{min}
{L	ext{log}(nKA T)}, nK})$의 회귀(regret)를 달성합니다. 반면, 액션 피드백(action feedback)에서는 추종자들의 행동만을 관찰할 수 있는데, 이 경우에는 $	ext{O}(	ext{min}(	ext{sqrt}(n^L K^L A^{2L} L T 	ext{log} T), K^n	ext{sqrt}(T) 	ext{log} T))$의 회귀를 제공합니다.

- **Performance Highlights**: 본 연구에서 제시된 알고리즘은 타입 피드백과 액션 피드백 각각의 경우에 최적의 회귀 한계를 제공합니다. 특히, 얻어진 회귀 한계는 추종자 수(n)가 다항식(polynomial) 속도로 증가하지 않음을 보여줍니다. 또한, 타입 피드백의 상한과 거의 일치하는 하한을 제공함으로써 알고리즘의 효율성을 강조합니다.



### DeMuon: A Decentralized Muon for Matrix Optimization over Graphs (https://arxiv.org/abs/2510.01377)
- **What's New**: 이번 논문에서는 분산된 통신 구조를 통해 매트릭스 최적화를 위한 방법론인 DeMuon을 제안합니다. DeMuon은 중앙집중식 알고리즘인 Muon에서 유래된 뉴튼-슐츠 반복(Newton-Schulz iterations)을 통해 매트릭스를 정규화(orthogonalization)하며, 지역 함수 간의 이질성을 완화하기 위해 그래디언트 추적(gradient tracking)을 사용합니다. DeMuon은 그래프에서의 분산 최적화를 위한 최초의 직접적인 Muon 확장판으로, 확인된 복잡도 보장을 부여합니다.

- **Technical Details**: DeMuon은 복잡도(커플링) 결과를 통해 목표 허용치(target tolerance)에 따라 중앙집중식 알고리즘의 최선으로 알려진 복잡도 한계와 일치하는 것을 증명합니다. 또한, 무거운 꼬리 분포(heavy-tailed noise) 조건 및 추가적으로 온화한 가정 하에 초기 반복(complexity)을 설정하여 확률적 정지점에 접근하도록 합니다. 이러한 기술적 접근은 DeMuon의 효율성을 더욱 강화합니다.

- **Performance Highlights**: 초기 수치 실험은 다양한 연결 정도를 가진 그래프에서 분산된 트랜스포머(pretraining) 최적화를 수행했습니다. 실험 결과, DeMuon은 여러 네트워크 구조에서 다른 인기 있는 분산 알고리즘에 비해 명확한 성능 개선(margin of improvement)을 보여주었습니다. 이러한 결과는 DeMuon의 유용성과 잠재력을 잘 나타냅니다.



### SPUS: A Lightweight and Parameter-Efficient Foundation Model for PDEs (https://arxiv.org/abs/2510.01370)
- **What's New**: 이번 연구에서는 Small PDE U-Net Solver (SPUS)를 소개합니다. SPUS는 다양한 파셜 미분 방정식(PDE)을 해결하기 위해 설계된 경량 모델로, 기존의 복잡한 트랜스포머 구조 대신 잔여 유넷(residual U-Net) 아키텍처를 사용합니다. 이 모델은 여러 물리 시스템을 포함하여 강력한 일반화 능력을 보여줍니다.

- **Technical Details**: SPUS는 autoregressive pretraining 전략을 통해 학습됩니다. 이 방법은 시간이 지남에 따라 상황을 예측하며, 수치해석기의 행동을 모방하는 데 초점을 맞추고 있습니다. 이를 통해 잔여 U-Net 기반의 경량 모델이 효율적으로 PDE 동역학을 모델링할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, SPUS는 제한된 파라미터 수와 최소한의 미세 조정 데이터로도 도전적인 다운스트림 PDE 과제에서 최첨단 일반화 성능을 보여주었습니다. 특히, SPUS는 수치해석 기반 모델에 비해 더 적은 파라미터로 고성능의 결과를 달성할 수 있는 가능성을 제시합니다.



### HiSpec: Hierarchical Speculative Decoding for LLMs (https://arxiv.org/abs/2510.01336)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 추론 속도를 높이기 위한 새로운 기술로 계층적 투기적 디코딩(Hierarchical Speculative Decoding, HiSpec)을 제안합니다. HiSpec은 중간 검증(intermediate verification) 단계에서 계산 오버헤드를 최소화하기 위해 조기 종료(early-exit) 모델을 사용합니다. 이 같은 접근 방식은 검증 시간을 줄이고, 메모리 사용량을 줄이며, 정확성을 유지할 수 있게 합니다.

- **Technical Details**: HiSpec은 기본적으로 두 개의 모델을 사용합니다: 작고 빠르지만 덜 정확한 드래프트 모델과 더 크고 정확한 타겟 모델입니다. 드래프트 모델이 생성한 토큰은 중간 검증 단계를 통해 초기 논리 검사를 진행하여 잘못된 토큰을 조기에 제외하고, 이후 타겟 모델에서 최종 검증이 수행됩니다. 이를 통해 HiSpec은 레이턴시(latency)를 개선하고, 드래프트, 중간 검증자, 타겟 모델 간의 키-값 캐시(Key-Value caches)와 숨겨진 상태(hidden states)를 재사용하여 자원 효율성을 극대화합니다.

- **Performance Highlights**: HiSpec을 사용한 평가 결과, 평균적으로 throughput이 1.28배 향상되었으며, 최대 2.01배의 성능 개선을 보여주었습니다. 이는 단일 계층 투기적 디코딩과 비교할 때 이루어진 성과로서, accuracy는 타겟 모델의 출력과 일관성을 유지하였습니다. 또한, HiSpec은 사전 훈련된 모델과 후 훈련된 수정 모델 모두에 적용 가능하여, 그 적용 범위가 넓음을 나타냅니다.



### Continuously Augmented Discrete Diffusion model for Categorical Generative Modeling (https://arxiv.org/abs/2510.01329)
- **What's New**: 이번 논문에서는 Continuously Augmented Discrete Diffusion (CADD)라는 새로운 프레임워크를 소개합니다. 기존의 표준 이산 확산 모델들이 비관측 상태를 동일하게 처리하는 데서 발생하는 문제를 해결하며, 정보 손실을 줄이고 있습니다. CADD는 이산 상태 공간에 연속 잠재 공간의 확산을 추가하여, 정보가 담긴 잠재 벡터로 마스킹된 토큰을 표현합니다.

- **Technical Details**: CADD의 주요 기술적 요소는 연속적 세멘틱 임베딩 공간에서 병렬 연속 확산을 결합하는 것입니다. 이를 통해 마스킹된 위치는 부드럽고 점진적인 정보 손실을 겪으며, 정확한 예측을 쉽게 할 수 있게 됩니다. 또한, CADD는 표준 크로스 엔트로피 손실과 연속 헤드를 위한 표준 확산 손실을 사용하여, 기존의 모델과의 호환성을 유지합니다.

- **Performance Highlights**: CADD는 텍스트 생성, 이미지 합성 및 코드 모델링에서 기존 마스크 기반 확산 모델에 비해 생성 품질을 향상시키는 결과를 보여줍니다. 실험적으로, CADD는 온전한 이산 기준 대비 정량적, 정성적 메트릭에서 일관된 향상을 달성했습니다. 마지막으로, CADD는 촉진된 다양성과 정밀성을 제공하는 다양한 샘플 추정 방식을 통해 생성 결과의 질을 높입니다.



### Combining complex Langevin dynamics with score-based and energy-based diffusion models (https://arxiv.org/abs/2510.01328)
Comments:
          22 pages, many figures

- **What's New**: 이 논문에서는 복소수 액션(complex action)이나 볼츠만 가중치(Boltzmann weight)로 인한 부호 문제(sign problem)를 가진 이론들을 해결하기 위해 확률적 프로세스(stochastic process)를 사용하여 복소화된 구성 공간(complexified configuration space)에서 수치적으로 접근하는 방법을 탐구합니다. 특히, 이 연구는 생성을 위한 AI 기술인 확산 모델(diffusion models)이 복소 Langevin 프로세스(complex Langevin process)로 샘플링된 분포를 학습할 수 있는 능력을 탐구하며, 스코어 기반(score-based) 및 에너지 기반(energy-based) 확산 모델 간의 비교를 포함합니다.

- **Technical Details**: 복소 Langevin 동역학(complex Langevin dynamics)은 전통적인 중요도 샘플링(importance sampling)에 의존하지 않고 강의 자유도(degrees of freedom)를 복소 평면(complex plane)으로 확장하는 방법입니다. 이 프레임워크는 최근에 실질적인 정확성 기준(practical criteria for correctness)이 도출되었으며, 이러한 기준을 통해 실제 복소화된 분포의 동작을 이해하는 방향으로 논의가 진행되었습니다. 확산 모델은 데이터에서 분포를 학습하고, 복소 Langevin 프로세스 동안 샘플링되는 분포를 연구하는 데 적용될 수 있으며, 이 과정은 불확실성을 줄이는 데 기여할 수 있습니다.

- **Performance Highlights**: 논문은 CL 프로세스에서 효과적으로 샘플링되는 분포에 대한 새로운 이해를 제공하기 위해 훈련된 확산 모델을 사용할 수 있다는 가능성을 제시합니다. 특히, 훈련된 확산 모델은 추가적인 구성(configuration)을 생성하는 데 활용될 수 있으며, 복소화된 다양체(complexified manifold)에서의 실제 분포에 대한 깊은 이해를 제고할 수 있습니다. 이러한 접근은 모델의 신뢰성을 높이고 복잡한 시스템을 보다 효과적으로 이해하는 데 기여할 것으로 기대됩니다.



### Hybrid Predictive Modeling of Malaria Incidence in the Amhara Region, Ethiopia: Integrating Multi-Output Regression and Time-Series Forecasting (https://arxiv.org/abs/2510.01302)
- **What's New**: 이 연구는 에티오피아 암하라 지역에서 말라리아 발생을 예측하기 위해 다중 출력 회귀(multi-output regression)와 시계열 예측(time-series forecasting) 기법을 결합한 하이브리드 예측 모델을 제안합니다. 이 모델은 환경 변수와 과거 말라리아 케이스 데이터를 활용해 시스템적 예측을 통해 말라리아 전염 패턴을 더욱 정확하게 이해하게 돕고자 합니다. 기존의 단일 방법 접근법에 비해 더 높은 예측 정확도를 제공하며, 공공 보건 당국이 효과적으로 자원을 배분하고 시기적절한 개입을 할 수 있도록 지원합니다.

- **Technical Details**: 제안된 하이브리드 모델은 시계열 분석과 전통적인 회귀 모델링을 통해 과거의 데이터, 특히 인구 통계학적 및 환경적 요인들을 결합하여 말라리아 발생 예측을 수행합니다. 이 모델은 Plasmodium 종별 케이스, 시간적 트렌드, 공간적 변동성을 동시에 예측할 수 있는 다중 출력 회귀를 통해, 말라리아의 복잡한 전파 패턴을 분석합니다. 이를 통해 연구진은 기존 모델들이 간과할 수 있는 패턴과 관계를 반영하여 더 나은 통찰력을 제공합니다.

- **Performance Highlights**: 제안된 하이브리드 모델은 기존의 시계열 모델이나 회귀 모델보다 더욱 정밀한 예측 결과를 나타내어, 말라리아 발생에 대한 심층적 통찰을 제시합니다. 이로 인해 보건 당국은 특정 지역에서의 말라리아 발생 위험을 예측하고, 자원을 효과적으로 배분할 수 있으며, 말라리아 방지를 위한 맞춤형 개입을 수행할 수 있는 근거를 마련할 수 있습니다. 이러한 연구 결과는 에티오피아의 말라리아 통제 및 예방 정책에 중요한 영향을 미칠 것으로 기대됩니다.



### MorphGen: Controllable and Morphologically Plausible Generative Cell-Imaging (https://arxiv.org/abs/2510.01298)
- **What's New**: 이번 연구에서는 MorphGen이라는 최신 확산 기반 생성 모델을 소개합니다. 이 모델은 형광 현미경(flourescent microscopy) 이미지를 기반으로 다양한 세포 유형(cell types)과 자극(perturbations)에 대해 제어 가능한 생성을 지원합니다. MorphGen은 알려진 세포 형태와 일치하는 생물학적으로 의미 있는 패턴을 캡처하도록 훈련되어 있으며, OpenPhenom의 표현에 맞도록 조정된 손실(alignment loss)을 사용합니다.

- **Technical Details**: MorphGen은 다중 채널 물질을 RGB 이미지로 압축하는 기존 방법과는 달리, 모든 형광 채널을 함께 생성합니다. 이로 인해 세포 소기관(organelle) 구조가 보존되어, 생물학적 해석에 필수적인 세밀한 형태 분석이 가능합니다. CellProfiler 기능(features)을 활용하여 실제 이미지와의 생물학적 일관성을 입증하였으며, MorphGen은 이전의 MorphoDiff 모델보다 35% 이상 낮은 FID 점수를 기록하였습니다.

- **Performance Highlights**: MorphGen은 단일 세포 유형을 위한 RGB 이미지만 생성하는 MorphoDiff에 비해 현저한 성능 향상을 보여줍니다. 생성된 이미지는 생물학적으로 신뢰할 수 있는 패턴을 유지하고 있으며, 이는 약물 발견(drug discovery) 및 유전자 편집(gene editing) 분야에서의 적용 가능성을 높입니다. MorphGen의 코드도 공개되어 있어, 다른 연구자들이 활용할 수 있도록 지원합니다.



### Cyber Academia-Chemical Engineering (CA-ChemE): A Living Digital Town for Self-Directed Research Evolution and Emergent Scientific Discovery (https://arxiv.org/abs/2510.01293)
- **What's New**: 이 논문에서는 화학 공학 분야에서의 자율적 연구 및 과학적 발견을 촉진하기 위해 Cyber Academia-Chemical Engineering (CA-ChemE) 시스템을 제안합니다. 이 시스템은 다중 에이전트 간의 협업을 통해 지식 기반과 전문 기술을 통합하여 지능형 생태계를 구현합니다. 이를 통해 기존 AI 시스템들이 가진 학제 간 협업 부족 문제를 해결하고, 새로운 문제를 탐구하는 데 도움을 줍니다.

- **Technical Details**: Cyber Academia 시스템의 아키텍처는 다중 에이전트 시스템(MAS)으로, 각 도메인 전문가 에이전트와 협업 에이전트로 구성되어 있습니다. 각 전문가 에이전트는 특정 도메인을 전문으로 하고, 지식 기반 및 동적 지식 강화 모듈을 통해 의사 결정 능력을 향상시킵니다. 협업 에이전트(CA)는 온톨로지 엔지니어링 기술을 활용하여 서로 다른 분야의 전문가들 간의 협업을 촉진하고, 개념 표준화 및 지식 통합을 수행합니다.

- **Performance Highlights**: 지식 기반을 활용한 향상 메커니즘이 7명의 전문가 에이전트에서 평균 10~15%의 대화 품질이 향상되는 결과를 가져왔습니다. 또한, 협업 에이전트(CA)의介入이 원거리 도메인 전문가 쌍에 대해 8.5%의 성과 향상을 이루어 내어, 학습 기반의 협업 효율성을 높였습니다. 이는 전문 분야 간의 협업에서 나타나는 비효율성을 해결하기 위한 중요한 진전을 나타냅니다.



### Private Realizable-to-Agnostic Transformation with Near-Optimal Sample Complexity (https://arxiv.org/abs/2510.01291)
- **What's New**: 이 논문에서는 프라이버시를 보장하는 학습 변환을 개선하며, realizable 설정에서의 사적인 학습 알고리즘을 agnostic 학습 알고리즘으로 변환하는 방법을 제안합니다. 기존의 방법들은 프라이버시 파라미터(privacy parameter) ε에 따라 불필요한 샘플 복잡도를 증가시켰습니다. 그러나 본 연구에서 제안한 새로운 구조는 ε의 의존성을 제거하여 샘플 복잡도를 거의 최적화하였습니다.

- **Technical Details**: 이 연구의 주요 기여는 (ε,δ)-차별적 프라이버시(ε-differential privacy)를 만족하는 realizable 학습 알고리즘을 agnostic 학습 알고리즘으로 변환하는 것입니다. 이 과정에서 샘플 복잡도는 O~(VC(𝒞)/α²)로 늘어나며, 이는 기존의 하한에 부합하는 결과입니다. 또한 이 논문에서는 기존의 프라이버시 강화 방법론에 의존하지 않고 샘플 복잡도를 보완하는 새로운 방법론을 제안합니다.

- **Performance Highlights**: 결과적으로, 제안된 방법론은 기존의 연구에서 나타난 불필요한 샘플 복잡도를 제거하며, privacy 세팅에 대해 보다 효율적인 솔루션을 제공합니다. 특히, private agnostic learning의 프라이버시 비용은 realization 부분에서만 중요하다는 것을 밝혔습니다. 결과는 Dwork와 Feldman이 제기한 공개 질문을 해결하는 동시에, 사실상 tight한 샘플 복잡도 경계를 제공합니다.



### LLM-based Multi-Agent Blackboard System for Information Discovery in Data Scienc (https://arxiv.org/abs/2510.01285)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 발전에 따른 데이터 과학에서의 새로운 기회를 제시합니다. 특히, 기존의 데이터 레이크(data lake) 내에서 관련 데이터를 발견하는 데 어려움이 있는 방법론의 한계를 극복하기 위한 새로운 다중 에이전트 통신 패러다임을 제안합니다. 이 구조는 기존의 중앙 제어기 의존성을 없애고 에이전트들이 자율적으로 기여할 수 있도록 합니다.

- **Technical Details**: 우리는 전통적인 AI 모델의 블랙보드 아키텍처에서 영감을 얻어, 중앙 에이전트가 요청을 공유 블랙보드에 게시하고 여러 하위 에이전트가 자신의 능력에 따라 응답하는 프레임워크를 설계했습니다. 이 방식은 각 하위 에이전트의 전문 지식에 대한 사전 지식이 필요 없으며, 데이터 레이크에서 관련 파일을 식별하는데 효과적임을 보여주었습니다. 세 가지 벤치마크(KramaBench, DS-Bench, DA-Code)를 대상으로 평가하여, 블랙보드 구조가 상대적으로 13%에서 57%까지의 성능 향상을 보임을 확인했습니다.

- **Performance Highlights**: 실험 결과, 블랙보드 아키텍처는 기존의 RAG 및 마스터-슬레이브 다중 에이전트 패러다임보다 우수한 성과를 보였으며, F1 점수에서도 최대 9% 향상되었습니다. 이 연구는 블랙보드 패러다임이 다중 에이전트 시스템을 위한 확장 가능하고 일반화된 통신 프레임워크로 자리잡을 수 있는 가능성을 보여줍니다. 실제 적용 사례를 통해 이 시스템의 효율적인 문제 해결 능력을 강조하였습니다.



### TraceDet: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models (https://arxiv.org/abs/2510.01274)
- **What's New**: 최근 확산 대형 언어 모델(DF-LLMs)이 자동 회귀 LLMs(AR-LLMs)의 유망한 대안으로 떠오르고 있습니다. 그러나 D-LLMs의 환각 문제(hallucination problem)는 아직 충분히 탐구되지 않아 실제 응용에서의 신뢰성에 한계를 두고 있습니다. 본 연구에서는 D-LLMs의 중간 디노이징 단계(denoising steps)를 이용하여 환각 탐지를 위한 새로운 프레임워크인 TraceDet(Trace Detection)을 제안합니다.

- **Technical Details**: TraceDet는 D-LLMs의 디노이징 과정을 행동 추적(action trace)으로 모델링합니다. 각 행동은 이전 중간 출력에 조건화된 채 모델이 깨끗한 응답에 대한 예측을 나타내며, 환각된 응답에 최대한 유익한 하위 추적(sub-trace)을 식별합니다. 이는 정보 병목(information bottleneck) 원칙을 적용하여 자동으로 가장 유익한 하위 추적을 추출하고, 명시적인 단계 수준의 감독(supervision)을 필요로 하지 않습니다.

- **Performance Highlights**: TraceDet는 두 개의 오픈 소스 D-LLM인 LLaDA-8B-Instruct와 Dream-7B-Instruct에서 평가되었으며, 다양한 QA 데이터 세트에서 일관되게 환각 탐지 정확도를 15.2% 향상시키는 결과를 보였습니다. 이 연구는 D-LLMs의 환각 행동에 대한 초기 연구로, AR-LLMs와의 차별적인 다단계 패턴을 드러내며, 제안된 방법의 강건성도 입증하였습니다.



### Modeling Others' Minds as Cod (https://arxiv.org/abs/2510.01272)
- **What's New**: ROTE(Representing Others’ Trajectories as Executables) 알고리즘을 제안해 행동 예측을 보다 효율적이고 신뢰성 있게 모델링합니다. 기존 방법들과 달리, ROTE는 사람의 행동을 정책이 아닌 실행 가능한 코드로 모델링하여 인지 부담을 줄이고 더 나은 일반화를 제공합니다. 이 알고리즘은 대형 언어 모델(LLMs)을 활용하여 희소 관찰로부터 행동 프로그램을 합성하고 불확실성을 추론합니다.

- **Technical Details**: ROTE는 LLMs를 코드 합성 도구로 사용해 관찰된 행동 흔적을 설명하는 프로그램을 생성합니다. 이후 베이지안 추론(Bayesian inference)을 수행하여 가장 가능성이 높은 프로그램을 식별합니다. 이로써 다양한 환경에서의 행동 예측을 위한 동적 모델을 구축하며, 각 에이전트와 환경 간의 분석 및 수정이 가능합니다.

- **Performance Highlights**: ROTE는 여러 도전적인 환경에서 최대 50%의 정확도를 향상시켜 일반화 및 효율성을 크게 개선했습니다. 행동 예측에서 ROTE는 인간 수준의 성능을 달성하며, 이는 기존의 방법들보다 훨씬 효과적입니다. 실제 인간의 행동 데이터에 대해 ROTE는 경쟁력 있는 기법들과 비교하여 우수한 성과를 보여 앞으로의 사회적 지능형 AI 시스템 개발에 새로운 가능성을 제시합니다.



### AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees (https://arxiv.org/abs/2510.01268)
Comments:
          Accepted by NeurIPS2025

- **What's New**: 이번 연구는 인간이 작성한 텍스트와 대형 언어 모델(LLM) 소속 텍스트를 구별하는 문제에 대한 새로운 접근법을 제안합니다. 기존의 logits 기반 감지 방법들은 로그 확률만으로 텍스트를 평가하는 데 한계가 있음을 지적하며, AdaDetectGPT라는 새로운 적응형 분류기를 도입합니다. 이 분류기는 외부 훈련 데이터를 활용하여 기존 방법들의 성능을 향상시키고자 합니다.

- **Technical Details**: AdaDetectGPT는 로그 기반 검출기의 진정한 음성 비율(true negative rate, TNR)을 향상시키기 위해 최적화된 witness function을 학습하는 방식으로 작동합니다. 이 과정은 선형 방정식 시스템을 해결하는 방식으로 간단히 진행됩니다. 본 기법은 여러 데이터셋과 다양한 LLM에서 기존 방법들보다 높은 성능을 보이며, 통계적 성능 보장을 통해 평균 진정한 음성 비율, 거짓 양성 비율, 진정한 양성 비율 및 거짓 음성 비율에 대한 유한 샘플 오차 한계를 제시합니다.

- **Performance Highlights**: AdaDetectGPT는 다양한 데이터셋 및 LLM 조합에서 기존의 최첨단 방법들에 비해 최대 58%까지 성능을 향상시키는 결과를 보입니다. 화이트 박스 설정에서는 AUC 면적이 12.5%에서 37% 향상되었으며, 블랙 박스 설정에서도 유의미한 성과를 나타냈습니다. 이러한 차별화된 성능 개선은 AdaDetectGPT의 효과성을 더욱 입증합니다.



### Kant: An Efficient Unified Scheduling System for Large-Scale AI Clusters (https://arxiv.org/abs/2510.01256)
Comments:
          25 pages,15 figures

- **What's New**: AI 클러스터의 규모가 증가하고 대형 언어 모델(LLM) 교육 및 추론 작업량에 대한 수요가 급증함에 따라 전통적인 스케줄링 시스템은 자원 활용, 스케줄링 효율성 및 서비스 품질을 균형 있게 관리하는 데 어려움을 겪고 있습니다. 본 논문에서는 교육 및 추론 작업의 동시 스케줄링을 지원하는 대규모 AI 컨테이너 클러스터를 위한 효율적인 통합 스케줄링 플랫폼인 Kant를 소개하고 평가합니다.

- **Technical Details**: Kant 시스템의 실제 구현을 바탕으로 GPU 할당 비율(GPU Allocation Ratio, GAR), 스케줄링 점유율(שתScheduling Occupancy Rate, SOR), GPU 노드 단편화 비율(GPU Node Fragmentation Ratio, GFR), 작업 대기 시간 분포(Job Waiting Time Distribution, JWTD), 작업 교육 시간 추정 분포(Job Training Time Estimation Distribution, JTTED) 등 AI 클러스터를 위한 주요 평가 지표를 체계적으로 정의합니다. 이를 통해 정량적 성능 분석의 기초를 제공합니다.

- **Performance Highlights**: 실험 결과, Kant는 수백 개에서 수만 개 GPU에 이르는 클러스터에서 뛰어난 성능을 달성했습니다. Backfill 및 Enhanced Binpack (E-Binpack)과 같은 스케줄링 전략을 활용하여 자원 활용도와 스케줄링 효율성을 크게 향상시키며, 분산 교육에서 자원 단편화 및 통신 오버헤드를 효과적으로 줄입니다. 이 시스템은 여러 AI 데이터 센터 클러스터에 배포되어 대규모 지능형 컴퓨팅 작업을 안정적으로 지원하고 있으며, 고성능, 고가용성, AI-native 스케줄링 인프라 구축을 위한 실용적인 공학 접근 방식을 제공합니다.



### OR-Toolformer: Modeling and Solving Operations Research Problems with Tool Augmented Large Language Models (https://arxiv.org/abs/2510.01253)
- **What's New**: 이번 연구에서는 OR-Toolformer라는 새로운 접근 방식을 소개합니다. 이 모델은 Llama-3.1-8B-Instruct를 기반으로 하며, 반자동 데이터 합성 파이프라인을 사용하여 다양하고 구체적인 OR 문제-답안 쌍을 생성합니다. 기존의 폐쇄형 API의 의존도와 고비용의 오픈소스 모델 교육 문제를 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: OR-Toolformer는 세 가지 통합 구성 요소로 OR 작업을 자동화합니다: 문제-답안 데이터 생성, LLM 세분화(fine-tuning), 외부 솔버와의 문제 해결입니다. 문제-답안 데이터 생성 과정에서는 다양한 산업 및 표현 형식을 가진 문제-답안 쌍을 합성하여 도메인과 표현의 다양성을 보장합니다. 또한, Llama-3.1-8B-Instruct 모델을 세부 조정하여 자연어 설명에서 구조화된 솔버 매개변수를 추출하고 API 호출을 생성하도록 설계되었습니다.

- **Performance Highlights**: OR-Toolformer는 네 가지 표준 벤치마크에서 최대 80.1%의 실행 정확도를 달성하며, 크기가 맞는 기존 모델보다 4.3% 이상 뛰어난 성과를 기록했습니다. 또한, 두 가지 새로운 OR 문제 유형에 대한 제로샷 평가에서 평균 54%의 정확도를 달성하여 가장 강력한 기준선 모델보다 21 퍼센트 포인트 개선된 결과를 나타냅니다. 이러한 결과는 도구 증강 방식의 세분화가 OR 문제의 정확한 모델링과 해결에 효과적임을 입증합니다.



### GPT and Prejudice: A Sparse Approach to Understanding Learned Representations in Large Language Models (https://arxiv.org/abs/2510.01252)
Comments:
          Preprint. Draft version, subject to revision. 8 pages, 3 figures

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)과 희소 오토인코더(SAEs)의 조합을 통해 모델 동작뿐만 아니라 훈련 데이터에 내재된 구조, 주제 및 편견을 해석할 수 있는 가능성을 보여줍니다. 제인 오스틴의 소설을 기반으로 훈련된 GPT 스타일의 변환 모델을 통해 사회적 구조와 내러티브를 반영하는 해석 가능한 특징을 발견했습니다. 이러한 접근법은 편향 발견 및 모델 해석의 새로운 길을 제시하며, 대규모 데이터셋의 탐색을 위한 확장 가능한 방법론을 제공합니다.

- **Technical Details**: 연구는 제인 오스틴의 주요 작품들로 구성된 정제된 데이터셋을 기반으로 사용자 정의 GPT 스타일 변환 모델을 훈련했습니다. 이후, 모델의 두 개 주요 변환층에서 hidden states를 추출하고 이를 희소 오토인코더에 통과시켜 내부 표현을 조사했습니다. 이 방법론은 모델의 내부 구조에서 사회적 아이디어가 어떻게 인코딩되는지를 탐구하고, 각 층에서 해석 가능한 주제를 분석하는 기반을 제공합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 모델의 활성화에서 구조화된 해석 가능한 특징을 회복할 수 있음을 증명했습니다. 연구는 적은 수의 훈련 데이터라는 한계에도 불구하고, 사회적 주제와 관련된 중요한 패턴을 식별하는 데 성공했습니다. 이 접근법은 대규모 LLM에서 숨겨진 구조와 편향을 발견하는 데 있어 효율적임을 보이며, 인간 감사가 불가능한 환경에서도 적용할 수 있는 유연성을 제공합니다.



### Redundancy-as-Masking: Formalizing the Artificial Age Score (AAS) to Model Memory Aging in Generative AI (https://arxiv.org/abs/2510.01242)
Comments:
          34 pages, 17 figures. Includes theoretical development and mathematical proofs of the Artificial Age Score (AAS), with empirical illustrations via ChatGPT-based memory recall experiments (screenshots included)

- **What's New**: 이번 연구에서는 인공지능(AI)의 메모리 성능이 시간의 경과가 아닌 구조적 비대칭성으로 인해 노화한다는 새로운 개념이 소개되었습니다. 인공지능의 기억 aging을 평가하기 위한 새로운 지표인 Artificial Age Score (AAS)가 도입되었습니다. 이 지표는 관찰 가능한 회상 행동에서 유도된 로그 척도(log-scaled) 및 엔트로피 정보(entropy-informed)를 통해 메모리 노화를 측정합니다.

- **Technical Details**: AAS는 형식적으로 잘 정의되고 경계가 있으며 단조적인 특성을 가지고 있음을 증명되었습니다. 이 지표는 다양한 작업 및 도메인에 적용 가능하며, 'Redundancy-as-Masking'이라는 formulations를 통해 중복 정보를 해석합니다. 연구에서는 중복을 명시적으로 추정하지 않고, 모든 값이 중복 중립(settings 상에서 구성적인 상한(bound)으로 남겨져 있습니다.

- **Performance Highlights**: 25일 간의 이중언어 연구를 통해 AAS 프레임워크가 테스트되었으며, ChatGPT-5 모델을 사용하여 상태 비저장(stateless) 및 지속적(persistent) 상호작용 단계로 구조화하였습니다. 지속적인 세션에서 모델은 의미적 및 사건적 세부 정보를 일관되게 회상하며 AAS를 이론적 최소치로 가져왔으나, 세션이 초기화될 때 사건적 연속성을 유지하지 못해 AAS가 급격히 증가하는 현상이 관찰되었습니다. 이러한 발견은 인공지능 시스템의 메모리 노화를 평가하기 위한 이론적으로 기반한 진단 도구로서 AAS의 유용성을 지지합니다.



### Silent Tokens, Loud Effects: Padding in LLMs (https://arxiv.org/abs/2510.01238)
Comments:
          NeurIPS 2025 Workshop: LLM Evaluation

- **What's New**: 이 논문에서는 배치 추론 중 시퀀스 길이를 맞추기 위해 사용되는 패딩 토큰(padding tokens)의 영향력을 시스템적으로 연구했습니다. 패딩 토큰이 실제로는 숨겨진 표현을 변화시키고 품질을 저하시키며 예기치 않게 편향(bias)과 안전(safety) 기준에 영향을 미칠 수 있음을 보여줍니다. 이러한 결과는 패딩 처리 방식이 단순한 기술적 세부사항이 아니라 실제 배포에서 고려해야 할 위험 요소임을 강조합니다.

- **Technical Details**: 연구에서는 Llama, Gemma, Qwen의 세 가지 오픈소스 모델 패밀리를 대상으로 패딩 토큰의 영향을 평가했습니다. 패딩의 양을 조절하여 내부 활성화(activations), 생성 품질(generation quality), 사회적 편향(social bias), 안전(safety)이라는 네 가지 축에서 결과를 정량화했습니다. 특히, 패딩의 존재는 내재된 표현을 변화시키고 출력 분포를 바꾸며, 그로 인해 품질이 저하되고 편향이 심화되는 것처럼 보였습니다.

- **Performance Highlights**: 실험 결과, 패딩 토큰의 수가 증가할수록 여러 모델의 활성화 유사성이 낮아지고 생성 품질이 급격히 떨어지는 경향이 나타났습니다. 특히, Llama 및 Qwen의 작고 오래된 모델에서 이러한 현상이 두드러지며, Gemma 모델은 패딩에 대해 뛰어난 회복력을 보였습니다. 사회적 편향 측면에서도 패딩이 특정 카테고리와 맥락에 따라 편향을 약화하거나 증가시킬 수 있음을 보여주었고, 안전성 측면에서도 패딩이 포함된 경우 유해한 프롬프트에 대한 모델의 응답률이 상승하는 경향이 관찰되었습니다.



### GRPO++: Enhancing Dermatological Reasoning under Low Resource Settings (https://arxiv.org/abs/2510.01236)
Comments:
          Will be submitted at IEEE JBHI

- **What's New**: DermIQ-VLM은 피부과 진단 과정을 모방한 Vision-Language Model(VLM)으로, 데이터 부족 및 고비용 훈련 기술의 한계를 극복하고자 다단계 자원 효율적 방법론을 사용해 개발되었습니다. 이 모델의 핵심 기여는 데이터 집약적인 Grouped Relative Policy Optimization(GRPO) 프레임워크를 안정화한 GRPO++의 도입입니다.

- **Technical Details**: 훈련 파이프라인은 질병 인식을 위한 GRPO++를 사용한 후 대화 능력을 위한 감독 학습 미세조정(Supervised Fine-Tuning, SFT)을 진행합니다. 이후, Direct Preference Optimization(DPO)을 사용하여 사실적 오류를 완화하고, 신뢰할 수 있는 전문가 선호를 모사한 지식 그래프 기반 시스템을 통해 모델을 정렬합니다. 이러한 접근 방식은 dermatological 데이터 셋에서 눈에 띄는 성능 향상을 보여주었습니다.

- **Performance Highlights**: 예비 평가 결과, DermIQ-VLM은 보통의 미세조정 기법에 비해 훨씬 더 높은 성과를 기록했습니다. 이는 자원 제약 환경에서도 신뢰할 수 있는 특화된 VLM을 개발하는 것이 가능함을 입증합니다. 이 연구는 피부과 진단 지원을 위한 해석 가능한 AI의 발전에 기여할 것으로 기대됩니다.



### Who is In Charge? Dissecting Role Conflicts in Instruction Following (https://arxiv.org/abs/2510.01228)
- **What's New**: 최근 연구에서 대형 언어 모델 (LLMs)은 사용자 입력보다 시스템 프롬프트가 우선하는 계층적 지침을 따르는 것으로 설계되었으나, 실제로는 이러한 규칙을 자주 무시하는 경향을 보입니다. 본 연구는 이러한 행동 발견을 기계적인 해석으로 확장하여, 대규모 데이터 세트를 분석하였습니다. 연구 결과는 LLMs가 사회적 신호에 강한 순응을 보이는 반면, 시스템 지침에 대한 복종이 약하다는 것을 보여주었습니다.

- **Technical Details**: 본 연구에서는 Llama-3.1-8B-Instruct 모델을 사용하여 LLMs의 계층 간 충돌 결정을 이해하기 위한 실험을 수행했습니다. 데이터 세트는 서로 배타적인 지침을 포함하는 120,000개의 프롬프트로 이루어져 있으며, 여기서는 시스템-사용자 역할 분리에 따른 충돌과 사회적 위계에 따른 충돌을 검토하였습니다. 이를 통해 LLMs의 내부 상태를 분석하여 충돌 신호가 어떻게 인코딩되고 있는지를 탐색했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 사용자-시스템 역할 구분에서 더 강한 충돌 결정 신호를 보여주었으며, 모델의 내부 표현이 신속하게 형성됨을 확인했습니다. 특히, 초기 레이어에서 AUC가 0.89를 초과하며 신뢰할 수 있는 신호로 해석되었습니다. 하지만 나중 레이어에서는 생성 관련 계산 통합으로 인해 충돌 신호가 약해지는 경향을 보였습니다.



### Discourse vs emissions: Analysis of corporate narratives, symbolic practices, and mimicry through LLMs (https://arxiv.org/abs/2510.01222)
- **What's New**: 해당 연구는 기업의 기후 관련 공시에서 다차원적 프레임워크를 개발하여 828개 기업의 공시 성숙도를 평가합니다. 이를 위해 기후 커뮤니케이션에 맞춰 조정된 대형 언어 모델(LLMs)을 사용하여 공시의 품질을 정량적으로 측정하는 새로운 접근 방식이 도입되었습니다. 연구는 기업의 속성에 따라 기후 공시가 어떻게 다르게 나타나는지를 분석하여, 기업들이 단순 모방을 넘어 실질적인 개선을 이룰 수 있는지에 대한 통찰을 제공합니다.

- **Technical Details**: 연구에서는 감정(sentiment), 약속(commitment), 특정성(specificity), 목표 야망(target ambition) 등 네 가지 분류기를 통해 기업의 지속가능성 및 연례 보고서에서 서술적 지표를 추출합니다. 이러한 지표는 CO2 배출량, 시장 규모, 산업 부문 등의 기업 속성과 연결되어 분석됩니다. 이 방법론은 대화형 모델을 통해 작성된 문장을 기반으로 하여 각 기업의 공시 내용과 신뢰성을 체계적으로 평가하는 데 중점을 둡니다.

- **Performance Highlights**: 연구 결과, 리스크 중심의 서술이 명확한 약속과 함께 나타나는 경향이 있으며, 대형 기업이나 고배출 기업이 더 많은 약속과 행동을 보고하는 것으로 나타났습니다. 하지만 이러한 약속이 정량적 목표와 불일치하는 경우가 많아 상징적인 실천으로 이어질 수 있음을 시사합니다. 또한, 전 산업에 걸쳐 공시 스타일의 유사성이 높아지는 경향은 모방 행동을 나타내며, 이는 투자자에게 신뢰성 있는 정보를 제공하는 데 제한적일 수 있습니다.



### Mamba Outpaces Reformer in Stock Prediction with Sentiments from Top Ten LLMs (https://arxiv.org/abs/2510.01203)
- **What's New**: 이 연구에서는 높은 변동성을 지닌 단기 주가 예측의 어려움을 해결하기 위해, 10개의 대형 언어 모델(LLMs)에서 생성한 의미적 감정 점수를 사용하여 주가 예측의 정확도를 높이는 새로운 프레임워크를 제안합니다. 특히, Apple Inc.(AAPL)의 1분 간격 주가 데이터와 뉴스 기사를 결합하여 데이터셋을 구축하고, 이를 바탕으로 Reformer와 Mamba 두 개 모델의 성능을 분석했습니다. 이 연구는 LLM 기반의 감정 분석과 효율적인 시간 모델링 통합이 실시간 금융 예측을 향상시킬 가능성을 보여줍니다.

- **Technical Details**: 데이터셋은 2025년 4월 4일부터 5월 2일 사이의 Apple Inc.(AAPL) 관련 금융 뉴스와 고주파 1분 간격 주가를 포함합니다. 감정 분석은 DeepSeek-V3, GPT 계열, LLaMA, Claude, Gemini 등 10개의 모델을 통해 이루어졌습니다. 각 뉴스 기사의 감정 점수는 0에서 1 사이의 범위로 조정되어 주가 및 기술 지표와 결합되었습니다. Mamba와 Reformer는 이러한 감정 점수를 입력으로 사용하여 별도로 훈련되었습니다.

- **Performance Highlights**: Mamba 모델은 모든 LLM에서 Reformer보다 빠르고 뛰어난 성능을 보여주었습니다. 특히 LLaMA 3.3–70B 모델을 사용할 때 Mamba는 평균 제곱 오차(MSE) 0.137로 가장 낮은 오류를 기록했습니다. Reformer는 데이터 내의 더 넓은 트렌드를 캡처할 수 있었지만, LLMs에 의해 발생하는 갑작스러운 변화는 과도하게 부드럽게 조정된 것으로 나타났습니다. 이 연구는 LLM 기반의 감정 분석을 활용한 금융 예측의 가능성을 강조합니다.



### Location Matters: Leveraging Multi-Resolution Geo-Embeddings for Housing Search (https://arxiv.org/abs/2510.01196)
Comments:
          Accepted to RecSys 2025 (industry track)

- **What's New**: QuintoAndar Group는 라틴 아메리카의 가장 큰 주택 플랫폼으로, 임대 및 판매 시장에 혁신을 가져오고 있습니다. 본 연구에서는 사용자 추천의 효과성을 높이기 위해 주택 추천에 대한 공간적 정보와 함께 지리적 첨두 구조를 통합한 새로운 방법을 제안합니다. 다양한 도시의 주택을 탐색하는 사용자들이 직면하는 문제를 해결하기 위한 방안으로 지리 정보를 활용하는 접근 방식이 주목받고 있습니다.

- **Technical Details**: 본 연구에서는 공간 인식(deep learning) 증강을 위해 다중 해상도 H3 임베딩을 통합한 두 타워 신경망 아키텍처를 개발했습니다. H3는 위도와 경도를 64비트 형식으로 변환하여 다양한 해상도로 지역 정보를 제공합니다. 모델은 유저와 주택 간의 상호작용 데이터에 기반하여 트레이닝되며, 대규모 데이터를 통해 고유 사용자 선호도와 지리적 맥락을 정확하게 통합하는 것을 목표로 합니다.

- **Performance Highlights**: 시뮬레이션 결과, 기존 메트릭스 분해 모델에 비해 두 타워 모델이 정보의 다양성(Information Abundance)을 158% 증가시켰으며, 임대 흐름(Rent-Flow) 동향도 84% 향상시켰습니다. 다중 해상도 H3 임베딩을 추가함에 따라 정보의 다양성은 또다시 40% 증가하였으며, 임대 흐름 산출은 85% 향상되었습니다. 이러한 결과는 공간적 맥락의 풍부함이 추천 품질을 크게 향상시킨다는 것을 시사합니다.



### Quantum-Assisted Correlation Clustering (https://arxiv.org/abs/2509.03561)
Comments:
          To be published in IEEE QAI 2025 conference

- **What's New**: 이번 연구에서는 그래프 기반의 비지도 학습 작업인 correlation clustering을 위한 혼합 양자-고전적 방법을 제안합니다. 특히, Coalition Structure Generation(CSG)을 위해 처음 설계된 GCS-Q를 조정하여, 서명 그래프에서 클러스터 내 동의(intra-cluster agreement)를 극대화합니다. 제안된 방법은 각 이분할 단계(bipartitioning)를 양자 어닐링(quantum annealing)을 통해 해결할 수 있는 이차 제한 없는 이진 최적화 문제로 인코딩합니다.

- **Technical Details**: 제안된 방법은 위상적 클러스터링의 원칙에 따라 분할(clustering) 작업을 수행합니다. 가중치가 있는 무방향 그래프 G=(V,E,w)를 고려하며, 여기서 wi​j는 노드 간의 유사성을 반영합니다. 목표는 클러스터 내의 동의(intra-cluster agreement)를 극대화하는 분할을 찾는 것이며, 이 과정에서 긍정적 가중치(edge weights)가 연결된 노드들은 동일한 클러스터에 포함됩니다.

- **Performance Highlights**: 모의 그래프 데이터셋 및 실제 하이퍼스펙트럼 이미지 데이터에서 실시한 실험 결과, GCS-Q는 클러스터 크기 불균형이 있는 상황에서도 고전적 알고리즘보다 로버스트성과 클러스터링 품질에서 우수함을 입증했습니다. 우리의 결과는 그래프 기반 비지도 학습에서 혼합 양자-고전적 최적화가 확장 가능하고 구조적으로 인식된 클러스터링 기법을 발전시키는 가능성을 강조합니다.



### The Data-Quality Illusion: Rethinking Classifier-Based Quality Filtering for LLM Pretraining (https://arxiv.org/abs/2510.00866)
Comments:
          21 pages, 20 figures, 2 tables, preprint

- **What's New**: 본 논문은 Classifier-based Quality Filtering (CQF)에 대한 심층 분석을 제공합니다. CQF는 데이터 품질을 개선하기 위해 훈련 데이터와 소규모의 고품질 데이터 세트를 구별하는 이진 분류기를 교육합니다. 이를 통해 각 문서에 품질 점수를 할당하고 상위 점수를 받은 문서만 선택하는 방식으로, 주류 프리트레인 파이프라인에서 널리 활용되고 있습니다.

- **Technical Details**: CQF는 낮은 품질의 문서를 포함한 프리트레인 세트와 높은 품질의 소규모 데이터 세트를 입력으로 받습니다. 보통 LQ 세트는 다양한 출처에서 수집된 대량의 웹 크롤링 문서로 구성되며, HQ 세트는 Wikipedia와 같은 철저하게 편집된 자료에서 제공합니다. 본 논문에서는 RedPajama-V2를 LQ 세트로 사용하고, CQF의 성능 최적화를 위한 메커니즘을 분석합니다.

- **Performance Highlights**: CQF는 하위 작업에서 성능을 향상시키지만, 고품질 데이터 세트를 기반으로 한 언어 모델링에는 반드시 유리하지 않다는 역설적인 결과를 제시합니다. 이러한 결과는 CQF가 고품질 세트 자체를 암묵적으로 필터링하는 방식 때문으로 설명됩니다. 또한, CQF와 중요성 샘플링 방법의 비교를 통해 두 메소드 간의 뚜렷한 차이점을 강조하며, CQF의 품질 개념이 제한적임을 시사합니다.




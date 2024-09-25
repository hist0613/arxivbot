New uploads on arXiv(cs.CL)

### A fast and sound tagging method for discontinuous named-entity recognition (https://arxiv.org/abs/2409.16243)
Comments:
          EMNLP 2024

- **What's New**: 논문에서는 불연속(named entity) 개체 인식을 위한 새로운 태깅 방식(tagging scheme)을 제안합니다. 이 방식은 불연속적인 언급의 내부 구조에 대한 명시적 설명에 기반하며, 혼란 없는 태그 시퀀스 예측을 보장합니다.

- **Technical Details**: 제안된 방법은 가중 유한 상태 자동자(weighted finite state automaton)를 사용하여 마진(marginal) 및 최대 사후 확률 추론(maximum a posteriori inference)을 수행합니다. 이 방법의 시간 복잡도는 입력 길이에 대해 선형이며, GPU에서 효율적으로 구현할 수 있습니다.

- **Performance Highlights**: 생물의학(biomedical) 분야의 세 가지 영어 데이터셋에서 검증하였으며, 기존의 최첨단 결과와 유사한 성능을 보이면서 더 간단하고 빠른 모델을 사용하였습니다.



### EuroLLM: Multilingual Language Models for Europ (https://arxiv.org/abs/2409.16235)
- **What's New**: 이번 연구는 EuroLLM 프로젝트를 소개하고 있으며, 이는 유럽 연합의 모든 공식 언어 및 추가 언어들에 대해 이해하고 생성할 수 있는 다국어(open-weight) 대형 언어 모델(LLM) 스위트를 개발하는 것을 목표로 합니다.

- **Technical Details**: EuroLLM 모델은 다국어 토크나이저를 개발하고, 다양한 출처에서 수집한 데이터를 필터링하여 훈련 데이터셋을 구성하였습니다. 데이터는 웹 데이터, 평행 데이터, 코드/수학 데이터 및 고품질 데이터로 구분되며, 각 언어별로 적절한 데이터를 수집하였습니다. 또한, 머신러닝 성능 향상을 위한 하이퍼파라미터 설정 및 모델 사전 훈련, 후 훈련 과정을 통해 EuroLLM-1.7B 및 EuroLLM-1.7B-Instruct 모델을 개발하였습니다.

- **Performance Highlights**: EuroLLM 모델은 여러 다국어 일반 벤치마크 및 기계 번역 과제를 평가하였으며, 초기 모델인 EuroLLM-1.7B와 EuroLLM-1.7B-Instruct이 여러 언어에서 경쟁력 있는 성능을 보였습니다.



### HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models (https://arxiv.org/abs/2409.16191)
- **What's New**: 최근 대형 언어 모델(LLMs)의 긴 텍스트 생성 능력에 대한 포괄적인 벤치마크가 부족함을 인식하고, 이를 해결하기 위해 계층적 긴 텍스트 생성 벤치마크인 HelloBench를 제안합니다.

- **Technical Details**: HelloBench는 Bloom's Taxonomy에 기반하여 긴 텍스트 생성 작업을 오픈 엔디드 QA, 요약, 채팅, 텍스트 완성, 휴리스틱 텍스트 생성 등 5개 하위 작업으로 분류합니다. 또한 HelloEval이라는 인간-정렬(human-aligned) 평가 방법을 제안하여 인간 평가 시 소요되는 시간을 대폭 줄이면서도 인간 평가와의 상관성을 높입니다.

- **Performance Highlights**: 현재 LLM들은 4000단어 이상의 긴 텍스트 생성에 어려움을 겪고 있으며, 일부는 긴 텍스트 생성이 가능하지만 반복과 품질 저하 등의 문제가 큽니다. HelloEval은 기존 전통적인 지표들(ROUGE, BLEU 등)과 비교하여 인간 평가에 가장 높은 상관성을 보여줍니다.



### Controlling Risk of Retrieval-augmented Generation: A Counterfactual Prompting Framework (https://arxiv.org/abs/2409.16146)
- **What's New**: 재검색 강화 생성(RAG) 모델의 예측 불확실성을 다룬 연구로, 모델이 자신감이 낮은 질문에 대해서는 답변을 거부하는 프로세스 강화에 중점을 두고 있습니다.

- **Technical Details**: 두 가지 주요 요소, 즉 검색된 결과의 품질과 이 결과들이 활용되는 방식을 통해 RAG 모델의 신뢰도를 평가하는 새로운 접근방식을 제안합니다. 이 방법은 counterfactual prompting 프레임워크를 기반으로 하여, 모델이 이러한 요소를 변경할 수 있도록 유도합니다.

- **Performance Highlights**: 제안된 프레임워크는 Mistral과 ChatGPT를 이용한 RAG 실험에서 3개의 4개 설정에서 주의 깊음(carefulness)과 위험(risk) 측면에서 기존 기준선보다 높은 성능을 보여주었으며, 주의 깊음에서 최대 14.76% 개선 및 위험에서 평균 2.88% 감소를 달성하였습니다.



### Exploring Hint Generation Approaches in Open-Domain Question Answering (https://arxiv.org/abs/2409.16096)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이 논문에서는 기존의 정보 검색(retrieval) 및 생성(generation) 기반 방법 대신 자동 힌트 생성(Automatic Hint Generation, HG) 기술을 활용한 새로운 QA 시스템 구성 요소인 HINTQA를 제안합니다. HINTQA는 질문에 대한 답변 가능성을 제시하는 힌트를 생성하여 QA 시스템의 정확도를 높입니다.

- **Technical Details**: HINTQA는 질문 q와 후보 답변 집합 𝒜을 활용하여 다수의 힌트를 생성합니다. 각 힌트는 수렴 점수(convergence score)인 HICOS를 통해 힌트의 유용성을 측정하며, 이는 질문에 대한 잠재적 답변을 좁힙니다. 제안된 시스템은 TriviaQA, Natural Questions, Web Questions 데이터셋을 사용하여 세 가지 QA 데이터셋에서 힌트 생성의 효과를 실험했습니다.

- **Performance Highlights**: HINTQA 방식은 정보 검색 및 생성 기반 방법보다 우수한 성과를 보였습니다. 연구 결과에 따르면 힌트를 사용하는 것이 검색된 문서나 생성된 문맥보다 답변의 정확성을 높이는 데 더 효과적이라는 것을 증명했습니다.



### Unlocking Markets: A Multilingual Benchmark to Cross-Market Question Answering (https://arxiv.org/abs/2409.16025)
Comments:
          EMNLP 2024

- **What's New**: 본 논문은 다국어 및 다시장간 제품 기반 질문 응답(Multilingual Cross-market Product-based Question Answering, MCPQA)이라는 새로운 작업을 제안합니다. 특정 시장의 제품 관련 질문에 대해 더 많은 자원이 있는 보조 시장 정보를 활용하여 답변을 제공합니다.

- **Technical Details**: 우리는 17개 시장에서 700만 개 이상의 질문으로 구성된 데이터 세트를 도입하고 전자 상거래 분야의 질문을 맥락으로 자동 번역하여 McMarket라는 데이터 세트를 분석했습니다. 두 개의 하위 작업인 리뷰 기반 답변 생성(Answer Generation, AG)과 제품 관련 질문 순위 매기기(Question Ranking, QR)에 대한 실험이 수행되었습니다.

- **Performance Highlights**: 결과적으로, 크로스 마켓 정보의 통합이 두 작업 모두에서 성능을 크게 향상시켰으며, LLM 기반 접근 방식이 전통적인 모델보다 우수한 성능을 보여주었습니다.



### AI Can Be Cognitively Biased: An Exploratory Study on Threshold Priming in LLM-Based Batch Relevance Assessmen (https://arxiv.org/abs/2409.16022)
- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 정보 검색(Information Retrieval) 작업에서 문서의 관련성을 판단할 때 경험적으로 관찰된 threshold priming 효과에 영향을 받는지를 조사합니다. 이는 LLM의 인지 편향(cognitive bias)에 대한 연구의 일환으로, LLM이 사람의 판단과 유사한 방식으로 영향을 받을 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 TREC 2019 Deep Learning 구문 수집에서 10개의 주제를 대상으로 LLM의 문서 관련성 평가를 시험했습니다. 실험에는 GPT-3.5, GPT-4, LLaMa2-13B 및 LLaMa2-70B 모델이 사용되었습니다. 초기 문서의 관련성 레벨이 후속 문서의 평가에 미치는 영향을 분석하기 위해 다양한 프로로구(prologue) 및 에필로그(epilogue) 길이를 가지고 실험을 수행했습니다.

- **Performance Highlights**: 결과에 따르면, 초기 문서의 높은 관련성은 후속 문서의 낮은 점수를 유도하는 경향이 있었습니다. 반대로 초기 문서의 낮은 관련성은 후속 문서에 대한 높은 점수를 유발했습니다. LLaMa2-70B 모델은 일부 조건에서 다른 모델과 다른 경향을 보였으며, 다른 모델들은 threshold priming 효과의 성격을 유지했습니다.



### Bridging Speech and Text: Enhancing ASR with Pinyin-to-Character Pre-training in LLMs (https://arxiv.org/abs/2409.16005)
Comments:
          Accepted by ISCSLP2024-Special session-Speech Processing in LLM Era

- **What's New**: 이 논문은 대형 언어 모델(LLM)과 사전 훈련된 음성 모델을 통합하여 자동 음성 인식(ASR)에서의 성능을 개선하는 새로운 접근 방식을 제안합니다. 특히, LLM을 음성 특징을 나타내는 Pinyin 임베딩 시퀀스로 사전 훈련하여 중국어 문자를 생성하도록 훈련합니다.

- **Technical Details**: 본 연구에서는 두 단계의 훈련 방법을 제안하는데, 첫 번째 단계에서 LLM을 Pinyin 입력으로부터 중국어 문자를 예측하도록 미세 조정합니다. 두 번째 단계에서는 사전 훈련된 오디오 모델에서 오디오 특징을 추출하여 LLM에 전달합니다. LoRA(低秩适应)을 사용하여 매개변수를 업데이트하며, 이를 통해 모델이 음성 특징을 수용하고 해당 전사(prediction)를 예측할 수 있도록 합니다.

- **Performance Highlights**: 이 연구에서 제안한 접근 방식은 AISHELL-1 코퍼스에서 Pinyin-문자 사전 훈련 없이 기본 기준선에 비해 9.5%의 상대적인 성능 향상을 보여주었고, 추가적인 보조 텍스트 데이터를 포함하여 Pinyin-문자 사전 훈련을 진행함으로써 19.0%의 성능 향상을 기록했습니다.



### Finetuning LLMs for Comparative Assessment Tasks (https://arxiv.org/abs/2409.15979)
Comments:
          8 pages, 5 figures, 6 tables

- **What's New**: 이 논문은 instruction-tuned 대규모 언어 모델(LLM)을 활용한 비교 평가(comparative assessment) framework를 제안하여, 전통적인 평가 방법들보다 효율적인 방식으로 NLG 시스템의 텍스트 품질을 평가하는 방법을 다룹니다.

- **Technical Details**: 제안된 방법은 LLM을 비교 평가를 위한 soft probabilities로 fine-tuning(fine-tuning)하여, 두 텍스트 간의 비교 결과를 좀 더 정밀하게 모델링하는 접근법에 중점을 둡니다. 이를 통해 true inference time probabilities와 PoE(제품 전문가) 프레임워크 내에서의 가정된 분포를 일치시킬 수 있습니다.

- **Performance Highlights**: 이 논문에서 제안된 방법은 기존의 하드 이진 결정(hard binary decision) 훈련보다 효율적인 비교 수로 더 높은 성능을 달성할 수 있음을 보여주며, 이는 다양한 NLG 평가 벤치마크에서 입증되었습니다.



### Beats of Bias: Analyzing Lyrics with Topic Modeling and Gender Bias Measurements (https://arxiv.org/abs/2409.15949)
Comments:
          Accepted and presented at the 17th International Conference on Social Computing, Behavioral-Cultural Modeling, & Prediction and Behavior Representation in Modeling and Simulation (see this https URL )

- **What's New**: 이 논문은 주제 모델링(topic modeling)과 편견 측정(bias measurement) 기법을 사용하여 영어 노래 가사에서의 성별 편향을 분석하고자 합니다. 537,553개의 영어 노래를 BERTopic으로 클러스터링하여 주제를 구분하고, 시간이 지남에 따라 이들이 어떻게 변화하는지를 보여줍니다.

- **Technical Details**: 노래 가사를 분석하기 위해 자연어 처리(Natural Language Processing, NLP) 기술을 적용하였으며, SC-WEAT(Single Category Word Embedding Association Test)를 사용하여 각 장르별 편향 점수를 산출했습니다. 541개의 주제를 발견했으며, rap 장르의 가사에서 성적 불균형이 두드러진다는 점을 확인했습니다.

- **Performance Highlights**: 연구 결과, 가사의 주요 테마가 로맨스에서 여성의 성적 대상화로 변화하고 있으며, 특히 rap 장르에서 공격적인 표현과 여성혐오적 가사가 남용되고 있음을 발견했습니다. 남성과 관련된 단어는 전반적으로 남성 편향을 보이는 반면, 외모와 약함과 관련된 단어는 여성 편향을 나타냈습니다.



### Automated test generation to evaluate tool-augmented LLMs as conversational AI agents (https://arxiv.org/abs/2409.15934)
Comments:
          14 pages, 5 figures, Submitted to GenBench@EMNLP2024

- **What's New**: 본 논문에서는 Tool-augmented LLMs(대형 언어 모델)를 평가하기 위한 테스트 생성 파이프라인을 제시하고 있습니다. 기존의 평가 데이터셋이 단일 상호작용 및 함수 호출에만 집중했던 반면, 이 연구는 사용자 정의 절차에 기반한 다양한 테스트를 생성합니다.

- **Technical Details**: LLMs를 기반으로 한 추천된 파이프라인은 중간 그래프(intermediate graphs)를 활용하여 발생할 수 있는 비현실적인 내용生성을 제한하고, 대화의 가능성을 널리 포괄하는 고품질 데이터를 생성합니다. 이 연구에서는 고객 지원을 위한 AI 에이전트 평가를 위한 ALMITA(Automated benchmark of Language Models for Intelligent Tool-augmented Agents)라는 수작업으로 제작된 데이터셋을 개발했습니다.

- **Performance Highlights**: 기존 LLM들은 단일 메시지 정확도 및 올바른 함수 호출에 있어 높은 성능을 보였지만, 전체 대화에서의 정확도는 제한적임을 보여주었습니다. 이는 LLMs가 완전 자율 고객 지원 AI 에이전트로 배치될 경우의 성공 가능성에 의문을 제기합니다.



### SLIMER-IT: Zero-Shot NER on Italian Languag (https://arxiv.org/abs/2409.15933)
- **What's New**: 이 논문에서는 이탈리아어에 대한 제로샷 이름 개체 인식(Zero-Shot Named Entity Recognition, NER) 평가 프레임워크를 정의하고, 제로샷 NER을 위한 SLIMER의 이탈리아어 버전인 SLIMER-IT를 소개합니다. SLIMER-IT는 정의 및 가이드라인으로 향상된 프롬프트를 활용하여 다루지 않은 엔티티 태그를 식별하는 데 우수성을 보입니다.

- **Technical Details**: SLIMER-IT는 대형 언어 모델(LLM)을 기반으로 하며, 인스트럭션 튜닝(instruction tuning)을 통해 성능을 개선합니다. 이 접근법은 주어진 텍스트에서 각각의 엔티티 타입을 효과적으로 추출하기 위해 설계된 프롬프트를 사용하여, 모델이 각 엔티티 타입에 집중할 수 있도록 지원합니다. SLIMER는 네임드 엔티티(Named Entity)에 대한 정의와 가이드라인을 제공하여 모델의 라벨링을 최적화합니다.

- **Performance Highlights**: SLIMER-IT는 기존의 다른 최첨단 모델들과 비교했을 때 보지 못한 엔티티 태그(label)를 라벨링하는 데 있어 뛰어난 성능을 보여주었습니다. 실험 결과는 SLIMER-IT가 이탈리아어로 된 데이터셋에서 제로샷 NER을 수행하는 데 매우 효과적임을 입증하였습니다.



### Multilingual Transfer and Domain Adaptation for Low-Resource Languages of Spain (https://arxiv.org/abs/2409.15924)
Comments:
          6 pages,wmt24. arXiv admin note: substantial text overlap with arXiv:2409.14842; text overlap with arXiv:2409.14800

- **What's New**: 이번 논문은 Huawei Translation Service Center (HW-TSC)의 WMT 2024에서의 스페인 저자원 언어 번역 태스크 제출 상태를 소개합니다. 이 연구팀은 스페인어에서 아라곤어(es-arg), 아라니세어(es-arn), 아스투리안어(es-ast)로의 번역 작업에 참여했습니다.

- **Technical Details**: 우리는 다국어 전이(multi-language transfer), 정규화 드롭아웃(regularized dropout), 포워드 번역(forward translation), 백 번역(back translation), Labse denoising, 전이 집합 학습(transduction ensemble learning) 등의 훈련 전략을 사용하여 딥 트랜스포머 기반의 신경 기계 번역(NMT) 모델을 훈련했습니다.

- **Performance Highlights**: 이러한 개선 전략을 통해 우리 제출물은 최종 평가에서 경쟁력 있는 결과를 달성했습니다.



### Explaining word embeddings with perfect fidelity: Case study in research impact prediction (https://arxiv.org/abs/2409.15912)
- **What's New**: 이 논문은 논문 품질 예측을 위한 새로운 기능 중요성 방법인 Self-model Rated Entities (SMER)를 제안하며, SMER는 로지스틱 회귀 모델과 단어 임베딩을 기반으로 하여 정확하고 명확한 설명을 제공합니다.

- **Technical Details**: SMER는 예측의 평균이 특정 단어에 대한 개별 예측과 정확히 대응되도록 하여 로지스틱 회귀 모델에 대해 이론적으로 완벽한 정확성을 보장합니다. 본 연구는 5가지 다양한 실험을 통해 50,000개의 CORD-19 연구 논문에 대한 정량적 및 정성적 평가를 실시했습니다.

- **Performance Highlights**: AOPC 곡선 분석을 통해 SMER는 로지스틱 회귀에서 LIME보다 향상된 설명을 생성한다는 점을 실험적으로 입증하였습니다.



### A Modular-based Strategy for Mitigating Gradient Conflicts in Simultaneous Speech Translation (https://arxiv.org/abs/2409.15911)
- **What's New**: 이 논문에서는 Simultaneous Speech Translation (SimulST)에서 발생하는 최적화 충돌 문제를 해결하기 위해 Modular Gradient Conflict Mitigation (MGCM) 전략을 제안합니다. 기존의 모델 레벨에서 충돌을 해결하는 방법들은 비효율적이었으나, MGCM은 모듈 레벨에서 충돌을 탐지하고 해결함으로써, GPU 메모리 사용을 95% 이상 절감하며 SimulST 성능을 향상시켜줍니다.

- **Technical Details**: MGCM은 Multi-task Learning (MTL) 프레임워크 내의 SimulST 작업을 보다 세밀한 모듈 수준에서 충돌을 감지하고 완화하는데 초점을 맞추고 있습니다. 이는 특히 Simultaneous Automatic Speech Recognition (SimulASR)와 Simultaneous Machine Translation (SimulMT) 등의 보조 작업과 관련된 최적화 목표의 충돌로 발생하는 문제를 해결합니다. 기존의 PCGrad 방법의 한계를 극복하기 위해, MGCM은 모듈화를 통해 결과적이고 효율적인 충돌 해소가 가능하게 합니다.

- **Performance Highlights**: MGCM을 적용한 실험 결과, SimulST 성능은 중간 및 고지연 조건에서 크게 향상되었으며, 오프라인 작업에서 0.68의 BLEU 점수 개선을 달성하였습니다. 또한, GPU 메모리 소비를 95% 이상 줄여, SimulST 작업에 대한 효과적인 솔루션으로 자리잡게 되었습니다.



### Enhancing Text-to-SQL Capabilities of Large Language Models via Domain Database Knowledge Injection (https://arxiv.org/abs/2409.15907)
Comments:
          This paper has been accepted by ECAI 2024

- **What's New**: 이번 논문에서는 LLMs가 데이터베이스 스키마와 셀 값에 대한 도메인 지식을 효과적으로 이해하고 활용할 수 있도록 '지식 주입' (knowledge injection) 방법을 도입하였습니다. 이를 통해 Text-to-SQL 작업에서의 성능을 향상시키는 다양한 기술적 접근을 선보입니다.

- **Technical Details**: 제안된 방법은 특정 도메인 데이터베이스 지식을 기반으로 LLMs를 사전 훈련 (pre-training)하고, 하위 Text-to-SQL 작업에 맞춰 미세 조정 (fine-tuning)하는 것입니다. 이를 통해 Execution Match (EX) 및 Exact Match (EM) 지표에서 현저한 개선을 이루어내며, 컬럼 이름 생성 및 값 일치 오류를 줄입니다.

- **Performance Highlights**: 실험 결과, 제안한 지식 주입 방법이 여러 개의 오픈 소스 LLMs에서 실질적인 성능 향상을 보여주었으며, 이는 다양한 Text-to-SQL 작업에 광범위하게 적용 가능하다는 것을 검증하였습니다.



### Konstruktor: A Strong Baseline for Simple Knowledge Graph Question Answering (https://arxiv.org/abs/2409.15902)
Comments:
          18 pages, 2 figures, 7 tables

- **What's New**: 본 논문은 간단한 질문에 대한 답변을 제공하기 위한 신뢰할 수 있는 접근방식인 Konstruktor 를 소개합니다. 이 방법은 질문에서 엔티티(entities)를 추출하고 이를 구조화된 지식 그래프(knowledge graphs)와 결합하여 답변을 찾는 과정을 세 가지 단계로 나누어 체계적으로 접근합니다.

- **Technical Details**: Konstruktor는 (i) 엔티티 추출 및 링크, (ii) 관계 예측, (iii) 지식 그래프 쿼리의 세 가지 컴포넌트를 포함합니다. 이를 통해 자연어 처리의 강력한 언어 모델과 지식 그래프의 해석력을 활용합니다. 특히, 이 연구는 Wikidata와 관련된 다양한 방법 및 데이터셋을 활용하여 성능을 평가합니다.

- **Performance Highlights**: 이 연구는 Konstruktor가 기존의 여러 기법, 특히 비용이 많이 드는 end-to-end 신경망 방법들을 능가하는 성과를 보여주며, 네 가지 데이터셋에서 강력한 결과를 보고합니다. 또한, 엔티티 링크 및 관계 탐지에 대한 SOTA 기술과 비교하여 뛰어난 성능을 입증합니다.



### HLB: Benchmarking LLMs' Humanlikeness in Language Us (https://arxiv.org/abs/2409.15890)
- **What's New**: 이 논문에서는 20개의 대형 언어 모델(LLMs)의 인간 유사성을 평가하기 위한 포괄적인 인간 유사성 벤치마크(Humanlikeness Benchmark, HLB)를 제시합니다.

- **Technical Details**: 이 연구는 음성(sound), 단어(word), 구문(syntax), 의미(semantics), 담론(discourse) 등 핵심 언어적 측면을 탐구하기 위해 설계된 10개의 심리언어학적 실험을 사용하여 LLM을 평가합니다. 각 실험의 응답을 2000명 이상의 인간 참가자로부터 수집하고, LLMs의 응답과 비교하여 분포 유사성(distributional similarity)을 통해 인간 유사성을 정량화하였습니다.

- **Performance Highlights**: 결과는 LLM이 여러 언어적 수준에서 인간의 반응을 얼마나 잘 재현하는지에 대한 미세한 차이를 밝혀냅니다. 또한, 다른 성능 지표의 개선이 반드시 인간 유사성의 증가로 이어지지 않으며, 몇 가지 경우에는 감소를 초래할 수 있음을 보여줍니다. 이 연구는 LLM의 언어 사용에서 인간 유사성을 시스템적으로 평가할 수 있는 최초의 프레임워크를 제공합니다.



### Machine Translation Advancements of Low-Resource Indian Languages by Transfer Learning (https://arxiv.org/abs/2409.15879)
Comments:
          6 pages, wmt24. arXiv admin note: substantial text overlap with arXiv:2409.14800

- **What's New**: 이 논문은 Huawei Translation Center (HW-TSC)가 WMT24 인도 언어 기계 번역(MT) 공동 작업에 제출한 내용을 소개합니다. 본 연구는 리소스가 부족한 인도 언어에 대한 신뢰할 수 있는 기계 번역 시스템을 개발하기 위해 두 가지 별도의 knowledge transfer 전략을 적용했습니다.

- **Technical Details**: Assamese(as)와 Manipuri(mn)의 경우, 우리는 기존의 IndicTrans2 오픈소스 모델을 미세 조정하여 영어와 이들 언어 간의 쌍방향 번역을 가능하게 했습니다. Khasi (kh)와 Mizo (mz)의 경우, 네 언어 쌍의 이중 언어 데이터를 이용하여 다국어 모델을 훈련시켰고, 추가적으로 약 8천 쌍의 영어-벵골어 이중 언어 데이터를 사용했습니다. 이를 통해 데이터 부족 문제를 해결했습니다.

- **Performance Highlights**: 전달 학습 실험에서는 en-as에 대해 23.5 BLEU, en-mn에 대해 31.8 BLEU, as-en에 대해 36.2 BLEU, mn-en에 대해 47.9 BLEU의 성과를 거두었습니다. 다국어 모델의 전이 학습 실험 결과는 en-kh에서 19.7 BLEU, en-mz에서 32.8 BLEU, kh-en에서 16.1 BLEU, mz-en에서 33.9 BLEU를 기록하였습니다.



### Privacy Evaluation Benchmarks for NLP Models (https://arxiv.org/abs/2409.15868)
Comments:
          Accepted by the Findings of EMNLP 2024

- **What's New**: 이 논문은 NLP 모델 특성에 따른 개인정보 공격(Benchmarking) 및 방어 전략 평가 프레임워크를 제안합니다. 일반적인 소규모 모델 및 대형 언어 모델(LLM)을 포함한 여러 종류의 데이터와 프로토콜을 지원하여 종합적인 공격과 방어 전략의 평가를 가능하게 합니다.

- **Technical Details**: 논문에서는 Membership Inference Attack (MIA), Model Inversion Attack (MDIA), Attribute Inference Attack (AIA), Model Extraction Attack (MEA)와 같은 네 가지 주요 개인정보 공격 방식에 대해 연구하며, 다양한 방어 방법과 함께 Knowledge Distillation (KD) 방식을 통해 공격 성능을 개선할 수 있는 방법을 제안합니다. 또한, 공격을 체인 방식으로 연결하는 프레임워크를 제안하여 더욱 서버능한 공격 목표를 달성할 수 있도록 지원합니다.

- **Performance Highlights**: 연구 결과에 따르면, 다양한 도메인 데이터의 사용은 공격 성능에 미치는 영향이 있으며, 각 공격 방식 간의 상호작용과 관계를 분석하여 개선된 방어 전략 및 공격 방법들을 개발했습니다. 이 평가 벤치마크는 다양한 NLP 모델과 데이터셋을 지원하며, 전반적인 프라이버시 위험을 평가할 수 있는 목적으로 디자인되었습니다.



### A Zero-Shot Open-Vocabulary Pipeline for Dialogue Understanding (https://arxiv.org/abs/2409.15861)
- **What's New**: 본 연구에서 우리는 제로샷(zero-shot), 오픈 어휘(open-vocabulary) 시스템을 제안하며, 디지털 대화 이해를 위한 통합된 파이프라인을 구성합니다.

- **Technical Details**: 제안된 방법론은 도메인 분류(domain classification)부터 시작하여, 여러 방법으로 DST(대화 상태 추적)를 수행합니다. 특히 DST를 질문-답변(question-answering) 문제로 변환하는 'DST-as-QA' 방식과 자가 수정 프롬프트(self-refining prompt) 기법을 활용한 'DST-as-SRP'를 포함합니다. 이 시스템은 고정된 슬롯 값에 의존하지 않아 동적으로 적응할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 MultiWOZ 2.1 데이터셋에서 20% 향상된 Joint Goal Accuracy(JGA)를 달성하며, LLM API에 대한 요청 수를 최대 90% 줄입니다. 또한, 제로샷 및 오픈 어휘 설정에서 현재 SOTA 방법들을 초월하는 성능을 보였습니다.



### Unveiling Language Competence Neurons: A Psycholinguistic Approach to Model Interpretability (https://arxiv.org/abs/2409.15827)
- **What's New**: 이 연구는 대규모 언어 모델(GPT-2-XL)을 대상으로 심리언어학 실험을 통해 신경 수준에서의 언어 능력 표현을 탐구합니다. 특히 소리-형태 연관, 소리-성별 연관 및 암묵적 인과관계라는 세 가지 작업을 사용하여 모델의 언어 처리 능력을 분석합니다.

- **Technical Details**: 연구에서는 심리언어학적 Paradigms를 적용하여 신경 세포의 작용을 탐색하고, 특정 신경 세포의 활성화 조작 및 제거(ablation)를 통해 언어 능력과 관련된 신경 생리학적 관계를 규명합니다. 이를 통해 GPT-2-XL이 인간과 유사한 언어 처리 능력을 어떻게 나타내는지를 고찰합니다.

- **Performance Highlights**: GPT-2-XL은 소리-형태 작업에서 어려움을 겪었으나 소리-성별 연관 및 암묵적 인과관계 작업에서는 인간과 유사한 능력을 보였습니다. 이 연구는 언어 모델에서 신경 수준의 해석 가능성 및 내부 메커니즘에 대한 새로운 인사이트를 제공합니다.



### Empirical Insights on Fine-Tuning Large Language Models for Question-Answering (https://arxiv.org/abs/2409.15825)
- **What's New**: 본 연구는 질문-응답(QA) 작업을 위한 대형 언어 모델(LLM)의 전이 학습을 최적화할 수 있는 효과적인 세부 전략을 제시합니다. 기존 연구와 달리, 우리는 사전 훈련된 언어 모델의 메모리와 지식 수준에 따라 데이터를 체계적으로 분류하고, 실험 분석을 통해 세 가지 주요 질문에 대해 답변합니다.

- **Technical Details**: 이 연구에서는 다중 템플릿 보완 메커니즘(multi-template complementation mechanism)을 사용하여 LLM의 특정 지식 유형에 대한 기억 정도를 평가합니다. 또한 SFT(Supervised Fine-Tuning) 단계에서 소수의 데이터 포인트(최소 60개)로도 QA 작업을 성공적으로 수행할 수 있음을 확인했습니다.

- **Performance Highlights**: SFT 데이터의 메모리 수준이 모델 성능에 미치는 영향을 분석한 결과, 사전 훈련 단계에서 잘 기억된 데이터로 훈련할 경우 LLM의 성능이 유의미하게 향상되는 것으로 나타났습니다. 하지만 모델이 거의 기억하지 못한 데이터를 사용한 경우에는 성능이 크게 저하되었습니다.



### NER-Luxury: Named entity recognition for the fashion and luxury domain (https://arxiv.org/abs/2409.15804)
Comments:
          28 pages, 6 figures

- **What's New**: 이 연구에서는 패션과 럭셔리 산업을 위한 명명된 개체 인식(Named-Entity Recognition, NER) 모델 개발의 다양한 도전 과제를 다룹니다. 주된 도전 과제는 개체의 중의성 해소(entity disambiguation), 여러 하위 분야의 프랑스어 기술 전문 용어, ESG 방법론의 부족, 그리고 작고 중간 규모의 럭셔리 하우스부터 대기업에 이르는 산업 내 다양한 기업 구조입니다.

- **Technical Details**: 이 연구에서는 럭셔리 중심으로 주석(annotation)된 36가지 이상의 개체 유형의 분류법(taxonomy)을 도입하고, 명확한 계층적 분류(hierarchical classification)를 준수하는 40,000개 이상의 문장으로 구성된 데이터셋을 생성하였습니다. 또한 패션, 뷰티, 시계, 보석, 향수, 화장품 등 다양한 카테고리를 위한 다섯 개의 감독 학습(supervised) 미세 조정(fine-tuned) 모델(NER-Luxury)을 소개합니다. 이 모델은 미적 측면과 양적 측면 모두를 균형 있게 다룹니다.

- **Performance Highlights**: 추가 실험에서는 우리의 모델과 최신 오픈 소스 대형 언어 모델(state-of-the-art open-source large language models)의 NER 성능을 정량적으로 비교하였으며, 유망한 결과들을 보여주었습니다. 또한 기존 머신러닝 파이프라인에 맞춤형 NER 모델을 통합하는 것의 이점을 강조합니다.



### Small Language Models: Survey, Measurements, and Insights (https://arxiv.org/abs/2409.15790)
- **What's New**: 이 논문은 최근 몇 년 간의 모든 작은 언어 모델(SLM)들을 종합적으로 검토하고 이들의 기술 혁신 및 온디바이스(온기기) 비용을 벤치마킹하여 요약합니다. SLM의 매니페스트 데이터를 공개하여 앞으로의 연구에 기여할 수 있는 기반을 마련합니다.

- **Technical Details**: SLM은 100M에서 5B 파라미터 범위의 transformer 기반, decoder-only 아키텍처로 구성됩니다. 59개의 최첨단 오픈 소스 SLM을 분석하여 아키텍처, 훈련 데이터셋, 훈련 알고리즘의 세 가지 축을 중심으로 기술 혁신을 평가합니다.

- **Performance Highlights**: SLM의 성능을 평가하며, commonsense reasoning, in-context learning, mathematics, coding과 같은 다양한 분야에서 능력을 분석합니다. 또한 벤치마킹 데이터를 통해 디바이스에서의 런타임 비용에 대한 귀중한 통찰을 제공합니다.



### CHBench: A Chinese Dataset for Evaluating Health in Large Language Models (https://arxiv.org/abs/2409.15766)
Comments:
          11 pages

- **What's New**: 이 논문에서는 중국어로 된 LLMs(대형 언어 모델)의 건강 관련 질문 처리 능력을 평가하기 위한 첫 번째 포괄적인 벤치마크인 CHBench를 소개합니다. CHBench는 정신 건강과 신체 건강 각각에 대한 다양한 주제를 포함하는 9,492개의 항목을 포함하고 있습니다.

- **Technical Details**: CHBench는 중국어 LLM의 신체 및 정신 건강 지식 이해를 평가하기 위해 설계되었습니다. 데이터는 웹 게시물, 시험 및 기존 데이터셋에서 수집되며, 실제 시나리오 분석과 추론(task) 문제를 포함하고 있습니다. 데이터의 품질을 평가하기 위해 여러 지표가 사용되며, Ernie Bot을 통해 항목에 대한 응답도 생성합니다.

- **Performance Highlights**: 네 개의 인기 있는 중국어 LLM에 대한 실험 평가 결과, 건강 관련 정보에 대한 이해도를 개선할 여지가 상당히 많음을 보여주었습니다. 이 연구는 중국어 LLM이 건강 관련 시나리오에서 더 안전하고 신뢰할 수 있는 정보를 제공할 수 있도록 하는 데 기여할 것으로 기대됩니다.



### XTRUST: On the Multilingual Trustworthiness of Large Language Models (https://arxiv.org/abs/2409.15762)
Comments:
          21 pages

- **What's New**: 이번 연구에서는 XTRUST라는 최초의 다국어 신뢰성 벤치마크를 도입하였습니다. 이는 LLM의 신뢰성을 평가하는 데 있어 다양한 주제 및 언어를 포괄적으로 포함하고 있습니다.

- **Technical Details**: XTRUST는 10개 언어(아랍어, 중국어, 프랑스어, 독일어, 힌디어, 이탈리아어, 한국어, 포르투갈어, 러시아어, 스페인어)로 데이터를 제공하고 있으며, 23,590개의 샘플을 수집하여 불법 활동, 환각, OOD(Out-of-Distribution) 견고성, 정신 건강, 신체 건강, 독성, 공정성, 잘못된 정보 및 사생활 등 여러 카테고리에 걸쳐 있습니다.

- **Performance Highlights**: 연구 결과, GPT-4가 대부분의 신뢰성 차원에서 다른 모델들을 능가했으며, Text-Davinci-002는 독성 문제에서 최고 성능을 보였습니다. 하지만, 모든 모델은 환각, OOD 견고성, 신체 건강 등 특정 카테고리에서 평균 정확도가 70% 이하로 나타나, LLM 신뢰성 향상의 필요성이 강조되었습니다.



### Hypothesis Clustering and Merging: Novel MultiTalker Speech Recognition with Speaker Tokens (https://arxiv.org/abs/2409.15732)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이 연구에서는 다중 화자 간 overlapping speech recognition (중첩된 음성 인식)의 문제를 해결하기 위해 Hypothesis Clustering and Merging (HCM)이라는 새로운 방법을 제안합니다. 이 접근법은 특별히 설계된 speaker class tokens를 활용하며, inferring 과정에서 예측된 speaker cluster tokens에 기반한 여러 인식 가설을 선택합니다.

- **Technical Details**: HCM은 k-means clustering을 통해 speaker embeddings를 클러스터링하여 speaker cluster ID를 정의합니다. 이 토큰은 학습 중 전사 시작 부분에 추가되며, attention-based encoder-decoder (AED)를 사용하여 다중 전사 생성시 여러 speaker token을 가정합니다. 예측된 전사 간의 normalized edit distance를 기반으로 하여 Agglomerative Hierarchical Clustering (AHC)으로 클러스터링합니다.

- **Performance Highlights**: LibriMix 데이터셋에서의 실험 결과, 복잡한 3-mix 환경에서 제안한 방법은 기존의 serialized output training (SOT) 방법에 비해 청정 데이터에서 55%, 노이즈가 있는 데이터에서는 36%의 상대적 오류 감소를 달성했습니다.



### Lighter And Better: Towards Flexible Context Adaptation For Retrieval Augmented Generation (https://arxiv.org/abs/2409.15699)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 효율성 및 비용 문제를 해결하기 위한 새로운 방법인 FlexRAG를 소개합니다. FlexRAG는 검색된 컨텍스트를 압축된 형태로 변환하여 LLM(대형 언어 모델)의 인코딩 부담을 줄입니다.

- **Technical Details**: FlexRAG는 검색된 컨텍스트를 압축된 임베딩으로 변환하여 두 단계의 학습 워크플로우를 통해 RAG 성능을 최적화합니다. 첫 번째 단계에서는 일반 코퍼스에 대해 사전 학습을 실시하며, 두 번째 단계에서는 다양한 지침 튜닝 데이터셋을 이용해 과업 특화된 파인튜닝을 수행합니다. 이 과정에서 FlexRAG는 다양한 압축 비율을 지원하며 정보의 중요도에 따라 선택적으로 압축을 수행합니다.

- **Performance Highlights**: FlexRAG는 여러 질문-답변 데이터셋에서 실험을 통해 비용 효율성이 크고, 다양한 압축 비율을 효과적으로 지원하며, 일반적인 사용성 또한 우수함을 입증했습니다. 이러한 결과는 FlexRAG가 RAG 시스템의 효과적이고 경제적인 구성 요소임을 확인시켜 줍니다.



### A Survey of Stance Detection on Social Media: New Directions and Perspectives (https://arxiv.org/abs/2409.15690)
- **What's New**: 이번 논문은 소셜 미디어에서의 스탠스 감지(stance detection) 기법에 대한 포괄적인 설문조사를 제공합니다. 전통적인 모델과 최신 대형 언어 모델(LLM) 기반 기법들을 분석하며, 공공 여론과 감정을 이해하는 데 있어 스탠스 감지의 중요성을 강조합니다.

- **Technical Details**: 스탠스 감지는 사용자의 발언이 특정 주제에 대한 지지, 반대, 중립의 태도를 식별하는 과정입니다. 이 연구에서는 다중 타겟 스탠스 감지, 주장 기반 스탠스 감지, 대화 기반 스탠스 감지 등 다양한 하위 작업을 소개하고, 최신 LLM 기반 방법을 통해 전통적인 기법과 비교합니다.

- **Performance Highlights**: 최근 연구에서는 LLM을 활용한 스탠스 감지의 효과와 미래 방향성을 제안하고 있습니다. 복잡한 멀티모달 환경과 저자원 언어( low-resource languages )에 대한 대처 방안도 논의되고 있으며, 이러한 접근이 새로운 도전과제를 어떻게 해결할 수 있는지에 대해 강조하고 있습니다.



### Mitigating Semantic Leakage in Cross-lingual Embeddings via Orthogonality Constrain (https://arxiv.org/abs/2409.15664)
Comments:
          18 pages, 16 figures

- **What's New**: 이 논문에서는 크로스-링구얼(Cross-lingual) 문장 임베딩에서 의미와 언어를 분리하는 새로운 방법인 ORACLE(ORthogonAlity Constraint LEarning)를 제안합니다. 기존의 방법들이 의미 누수(semantic leakage) 문제로 고통받고 있음을 발견하였습니다.

- **Technical Details**: ORACLE은 두 가지 요소, 즉 intra-class clustering과 inter-class separation을 기반으로 합니다. 이는 의미 임베딩과 언어 임베딩 간의 직교성을 보장하여, 의미와 언어 정보의 분리를 효과적으로 지원합니다.

- **Performance Highlights**: 실험 결과, ORACLE을 사용한 훈련은 의미 누수를 줄이고 임베딩 공간 내에서 의미 정렬을 향상시키는 데 성공했습니다. 이는 크로스-링구얼 검색(cross-lingual retrieval) 및 의미 텍스트 유사성(semantic textual similarity) 작업에서 입증되었습니다.



### English offensive text detection using CNN based Bi-GRU mod (https://arxiv.org/abs/2409.15652)
Comments:
          6 pages and 6 figures

- **What's New**: 본 논문은 Bi-GRU와 CNN 모델을 결합한 새로운 텍스트 분류 모델을 제안하여, 소셜미디어에서의 공격적 언어 탐지 향상을 목표로 하고 있습니다. 기존 모델들을 능가하는 성능을 보이기 위해 31,962개의 트윗 데이터를 사용하여 실험을 수행하였습니다.

- **Technical Details**: 제안된 모델은 1D Convolutional Neural Network (CNN)과 Bi-directional Gated Recurrent Unit (Bi-GRU)을 결합하여 비정상적인 텍스트를 분류하는 구조를 가지고 있습니다. 모델은 입력 레이어, 임베딩 레이어, 여러 개의 컨볼루션 레이어, 비등방향 GRU 레이어, 밀집 레이어 등을 포함한 9개의 레이어로 구성됩니다.

- **Performance Highlights**: 제안한 Bi-GRU-CNN 모델은 기존 모델들과 비교했을 때 우수한 성능을 보이며, 특히 정확도, F1-score, 리콜, 정밀도와 같은 다양한 평가지표에서 양호한 결과를 얻었습니다.



### Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents (https://arxiv.org/abs/2409.15594)
Comments:
          EMNLP Main 2024

- **What's New**: 본 논문에서는 기존의 반이중(half-duplex) 대화 모델의 한계를 극복하고, 동기화된 대화 모델(Synchronous LLMs, SyncLLM)을 제안하여 말하는 대화 모델을 전이중(full-duplex)으로 개발하고자 합니다.

- **Technical Details**: SyncLLM은 Llama3-8b 모델에 시간 정보를 통합하여 실제 세계의 시계와 동기화되도록 설계되었습니다. 이 모델은 212k 시간의 합성된 말하기 대화 데이터와 2k 시간의 실제 대화 데이터를 사용하여 훈련되며, 의미론적 일관성을 유지하면서 자연스러운 대화를 생성합니다.

- **Performance Highlights**: SyncLLM은 대화의 의미론적 유의미성에서 최신 기술보다 +2.2 포인트 향상된 Mean Opinion Score (MOS)를 달성했으며, 자연스러운 턴 테이킹을 유지하면서 두 개의 다른 데이터셋에서 훈련된 모델 간의 전이중 대화를 시뮬레이션할 수 있는 능력을 보여주었습니다.



### Optimizing News Text Classification with Bi-LSTM and Attention Mechanism for Efficient Data Processing (https://arxiv.org/abs/2409.15576)
- **What's New**: 이 논문은 전통적인 수동 분류 방법의 비효율성을 극복하기 위해 딥 러닝을 기반으로 한 뉴스 텍스트 자동 분류 방안을 제안합니다.

- **Technical Details**: 제안하는 방법은 Bi-directional Long Short-Term Memory Network (Bi-LSTM)와 Attention Mechanism을 결합한 최적화 모델을 사용하여 뉴스 텍스트의 효율적인 분류와 관리를 달성합니다.

- **Performance Highlights**: 실험 결과, 이 솔루션은 분류의 정확성과 시의성을 크게 향상시키고 수동 개입의 필요성을 줄이며, 뉴스 산업의 정보 처리 능력을 향상시키고 정보 흐름의 속도를 가속화하는 데 중요한 실용적 의미가 있음을 보여줍니다.



### GEM-RAG: Graphical Eigen Memories For Retrieval Augmented Generation (https://arxiv.org/abs/2409.15566)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 Retrieval Augmented Generation (RAG) 방식을 개선하여, 메모리 조작을 통한 AI의 성능 향상을 목표로 하고 있습니다. 저자들은 Graphical Eigen Memories For Retrieval Augmented Generation (GEM-RAG)라는 새로운 방법론을 제안하며, 이는 각 텍스트 조각에 대한 '유틸리티(utility)' 질문을 생성하고, 이를 기반으로 메모리 그래프를 구축하여 더 높은 수준의 요약 노드를 생성하는 방법입니다.

- **Technical Details**: GEM-RAG는 주어진 텍스트 코퍼스를 조각으로 분할한 후, LLM을 이용해 관련된 유틸리티 질문을 생성합니다. 생성된 질문의 임베딩은 가중치 그래프를 형성하며, 이 그래프의 고유값 분해를 통해 텍스트의 주요 테마를 포착하는 'eigenthemes' 또는 요약 노드를 생성합니다. 이 방법론은 두 개의 QA 데이터셋, QuALITY와 Qasper에서 성능을 평가하며, 표준 RAG 절차 및 최신 방법인 RAPTOR와 비교하여 우수성을 입증했습니다.

- **Performance Highlights**: GEM-RAG는 두 개의 QA 태스크에서 다른 최신 RAG 방법들과 비교하여 더 나은 성능을 보였습니다. 실험 결과에 따르면, LLM에 의해 생성된 요약 노드와 유틸리티 질문의 수가 모델 성능에 미치는 영향을 정량적으로 분석하여, GEM의 효과성을 뒷받침하는 세부 실험을 수행하였습니다.



### Learning When to Retrieve, What to Rewrite, and How to Respond in Conversational QA (https://arxiv.org/abs/2409.15515)
Comments:
          Accepted in EMNLP (findings) 2024

- **What's New**: 이 논문에서는 대화 맥락에서의 정보 검색 필요성을 판단하여 retrieval을 수행하는 방법인 SELF-multi-RAG를 제안합니다. 이는 대화형 질문 답변(QA) 시스템의 맥락 이해 및 응답 생성의 질을 개선하기 위한 연구입니다.

- **Technical Details**: SELF-multi-RAG 모형은 기존의 SELF-RAG(Asai et al., 2023) 프레임워크를 기반으로 하여, 대화 중 필요한 경우에만 검색을 수행하고, 검색된 문서를 요약하여 유용한 응답을 생성하는 과정에서의 효과를 개선합니다. 이는 대화의 요약된 맥락을 사용하여 관련 문서를 검색하도록 설계되었습니다.

- **Performance Highlights**: SELF-multi-RAG는 실험을 통해 전통적인 SELF-RAG보다 약 13%의 응답 품질 향상을 보여주었으며, 검색 효과성(R@5) 또한 평균 13.5% 향상되었습니다. 이러한 결과는 human annotation에 의해 검증되었습니다.



### In-Context Learning May Not Elicit Trustworthy Reasoning: A-Not-B Errors in Pretrained Language Models (https://arxiv.org/abs/2409.15454)
Comments:
          Accepted at EMNLP 2024 Findings

- **What's New**: 대규모 언어 모델(LLMs)의 억제 제어(inhibitory control) 능력을 체계적으로 평가한 최초의 연구로, A-Not-B 오류라는 아동 인지 현상을 바탕으로 디자인한 멀티 초이스 QA 시나리오를 통해 LLMs가 얼마나 잘 고착된 답변 패턴을 억제할 수 있는지 테스트함.

- **Technical Details**: 본 연구에서는 유지된 답변 패턴을 억제하는 LLM의 능력을 평가하기 위해 A-Not-B 실험 설정을 텍스트 기반으로 변환하였습니다. LLM에게 같은 정답 선택을 반복적으로 제공하여 패턴을 형성한 뒤, 새로운 질문을 통해 패턴을 변경하는 방식으로 테스트를 진행하였으며, A-Not-B 프롬프트(A-Not-B prompting) 전략을 적용하였습니다.

- **Performance Highlights**: 최신 LLM(예: Llama3-8b)은 3-shot A-Not-B 프롬프트 사용시, 일부 추론 작업에서 정확도가 83.3% 감소하는 심각한 오류를 보였으며, 이는 이들이 초등학생보다도 낮은 인지 능력을 가지고 있음을 나타냅니다. LLM의 오류 발생 원인에 대한 분석 결과, 모델 크기와 데이터 품질이 중요한 요소로 작용하며, 후속 학습 단계에서 자기 설명(self-explanation) 전략이 어느 정도 효과를 보였음을 확인하였습니다.



### CUTE: Measuring LLMs' Understanding of Their Tokens (https://arxiv.org/abs/2409.15452)
Comments:
          Accepted to EMNLP 2024 main conference

- **What's New**: 새로운 벤치마크 CUTE(Character-level Understanding of Tokens Evaluation)를 제안하여 대형 언어 모델(LLM)의 철자(orthographic) 지식을 평가합니다. CUTE는 문자 수준의 작업으로 구성되어 있으며, LLM이 어떻게 텍스트를 조작할 수 있는지를 근본적으로 검사합니다.

- **Technical Details**: CUTE는 철자 정보와 의미적으로 유사한 것의 차이를 이해하는지, 그리고 문자 수준에서 텍스트 조작 능력을 포함한 여러 가지 과제를 포함합니다. 평가된 LLM은 7B부터 132B까지 다양한 매개변수를 갖습니다. 주요 질문으로는 LLM이 자신의 토큰을 구성하는 문자를 인식하는지, 철자와 해당 문자들의 관계를 이해하는지 등의 문제가 포함됩니다.

- **Performance Highlights**: 대부분의 LLM은 토큰의 철자에 대한 지식은 가지고 있지만, 이 정보를 적극적으로 사용하여 텍스트를 조작하는 데에는 실패하는 것으로 나타났습니다. LLM의 이해도가 얼마나 일반화 가능한지에 대한 의문이 제기되었습니다.



### Parse Trees Guided LLM Prompt Compression (https://arxiv.org/abs/2409.15395)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 입력 프롬프트를 압축하는 새로운 방법인 PartPrompt를 소개합니다. 기존의 압축 방법들은 주로 언어 모델을 사용하여 새로운 프롬프트를 생성하거나 중요한 부분을 선택하는 방식으로 이루어졌으나, PartPrompt는 구문 트리를 기반으로 하여 로컬 정보 엔트로피를 활용하고, 구문 간의 연결 패턴을 고려하여 압축을 수행합니다.

- **Technical Details**: PartPrompt는 각 문장의 파싱 트리를 생성하고, 각 노드에 대해 로컬 정보 엔트로피를 계산합니다. 이러한 지역 파싱 트리는 문장, 단락 및 섹션의 계층적 구조에 따라 전역 트리로 구성됩니다. 이후, 연구자들은 새로운 방식인 root-ward propagation 및 leaf-ward propagation을 이용하여 전역 트리의 노드 값을 조정합니다. 마지막으로, 조정된 노드 값을 기반으로 전역 트리를 가지치기하는 재귀 알고리즘이 개발되었습니다.

- **Performance Highlights**: 실험 결과, PartPrompt는 다양한 데이터 세트와 메트릭, 압축 비율 및 LLM의 타겟 모델에서 최신 성과를 기록했습니다. 특히, PartPrompt는 긴 프롬프트에서의 일관성(Corehence) 측면에서도 우수한 성능을 보여주었습니다. 연구진은 이 방법이 기존의 프롬프트 압축 방법들보다 효율적임을 증명했습니다.



### Adversarial Attacks on Parts of Speech: An Empirical Study in Text-to-Image Generation (https://arxiv.org/abs/2409.15381)
Comments:
          Findings of the EMNLP 2024

- **What's New**: 이번 연구는 텍스트-이미지(T2I) 모델에서 다양한 품사(POS) 태그에 대한 적대적 공격의 영향을 조사하며, 기존의 연구들이 주로 명사에 초점을 맞춘 것과 달리 다양한 품사를 다루는 첫 번째 데이터셋을 작성하였습니다.

- **Technical Details**: 고품질의 데이터셋을 통해 POS 태그의 실제 시나리오에서 토큰 교환을 수행하고, 기울기 기반 공격(gradient-based attacks)을 통해 T2I 모델이 잘못된 이미지를 생성하게 만드는 적대적 접미사(adversarial suffixes)를 찾아냈습니다. 연구 결과, 공격 성공률(ASR)은 명사, 고유명사, 형용사에 대해 가장 높았으며, 각각의 POS 태그 카테고리에 따른 공격 메커니즘의 차이를 설명하기 위해 실험을 수행했습니다.

- **Performance Highlights**: 명사와 형용사 공격 시 이미지 생성의 유효성이 높고, 예상하는 속성을 이미지에 포함하도록 유도할 수 있는 성공률이 높았습니다. 특히 동일한 적대적 접미사를 사용하여 다양한 입력 프롬프트에 대해 동일한 속성을 생성할 수 있는 전반적인 특성 또한 확인되었습니다.



### Kalahi: A handcrafted, grassroots cultural LLM evaluation suite for Filipino (https://arxiv.org/abs/2409.15380)
- **What's New**: 다국어 대형 언어 모델(LLMs)에 대한 연구가 진행되는 가운데, Kalahi라는 새로운 문화적 LLM 평가 도구가 필리핀 원주율 사용자를 대상으로 개발되었습니다. 이 도구는 필리핀 고유의 문화적 지식과 가치를 반영한 150개의 정교하게 수작업으로 작성된 프롬프트로 구성되어, LLM이 필리핀 사회에서 일어날 법한 상황에 대해 적절한 대답을 생성할 수 있는지 평가합니다.

- **Technical Details**: Kalahi는 LLM이 필리핀 문화에 기반한 질문에 얼마나 잘 응답할 수 있는지를 평가하기 위해 설계된 도구입니다. 이 평가는 다국어 및 필리핀어 지원 LLM에 대해 수행되었으며, 150개의 상황에 대한 프롬프트를 사용하여 LLM의 성능을 측정합니다. 또한, 프롬프트 작성자는 필리핀 원어민으로 이루어져 있으며, 다양한 사회적 배경을 가진 인물들이 참여하여 문화적 대표성을 보장합니다.

- **Performance Highlights**: 실험 결과, Kalahi는 필리핀 원주율 사용자에게는 쉬운 질문들이 포함되어 있지만, LLM에게는 도전적인 질문으로 나타났습니다. 가장 잘 수행된 LLM은 단지 46.0%의 질문에만 정확히 응답했으며, 필리핀 원주율 사용자의 정확도는 89.10%로 나타났습니다. 이러한 차이는 Kalahi가 LLM의 필리핀 문화 표현을 평가하는 데 있어 신뢰할 수 있는 도구임을 시사합니다.



### Prompting Large Language Models for Supporting the Differential Diagnosis of Anemia (https://arxiv.org/abs/2409.15377)
- **What's New**: 본 연구는 임상 지침에 영감을 받아 비슷한 방식의 진단 경로(Pathways)를 개발하고, 이를 통해 희귀 질환 진단에 대한 한계를 극복하고자 하였습니다.

- **Technical Details**: 연구에서는 Generative Pretrained Transformer 4 (GPT-4), Large Language Model Meta AI (LLaMA), Mistral이라는 세 가지 대형 언어 모델(LLMs)을 이용하여 합성된 데이터셋을 기반으로 빈혈(Anemia) 및 그 하위 유형의 진단을 진행하였습니다. 고급 프롬프트 기법(advanced prompting techniques)을 사용하여 의사결정 프로세스를 향상시켰습니다.

- **Performance Highlights**: 실험 결과, LLMs는 환자 데이터를 기반으로 한 임상 경로 발견에서 큰 잠재력을 보였으며, 모든 실험에서 GPT-4가 최고의 성능을 나타냈습니다.



### Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models (https://arxiv.org/abs/2409.15371)
- **What's New**: 본 논문에서는 Bone(Block Affine)이라는 새로운 PEFT(Parameter-Efficient Fine-Tuning) 방법을 제안하여 기존 LoRA 변형의 한계를 극복하고 전체 파라미터 학습을 초월하는 방법론을 소개합니다. 이 방법은 메모리 오버헤드를 줄이고, 가중치 간의 내부 연결을 강조하여 더 빠른 수렴과 더 나은 데이터 적합을 이끌어냅니다.

- **Technical Details**: Bone은 초기화가 복잡하지 않은 단일 학습 가능한 행렬을 활용하며, 이 행렬은 W𝑊와의 상호작용을 통해 Block Affine 연산을 수행합니다. 이 구조는 LoRA의 두 개의 구조 복잡성을 해소하고, 빠른 수렴 속도와 데이터를 적합시키는 능력을 보여줍니다. 실험은 서로 다른 LLM 아키텍처(LLaMA2, RWKV6)와 다양한 파라미터 스케일을 기반으로 수행되었습니다.

- **Performance Highlights**: 실험 결과, Bone 방식은 LLaMA2-7B 모델을 MetaMathQA 데이터셋으로 파인튜닝 시 49.36의 점수를 기록하며, PISSA보다 5.84% 향상된 성과를 보였습니다. 또한, Bone은 복잡한 초기화 없이 빠른 수렴과 우수한 데이터 적합성을 달성함으로써 성능을 입증하였습니다.



### MedCodER: A Generative AI Assistant for Medical Coding (https://arxiv.org/abs/2409.15368)
- **What's New**: 이번 연구에서는 MedCodER라는 새로운 Generative AI 프레임워크를 소개하며, 의료 코딩의 자동화를 위한 혁신적인 접근 방식을 제시합니다. 특히, 이 프레임워크는 추출(Extraction), 검색(Retrieval), 재정렬(Re-ranking) 기술을 핵심 요소로 활용하여 높은 정확도를 자랑합니다.

- **Technical Details**: MedCodER는 의료 기록에서 질병 진단과 지원 증거를 추출하고, ICD-10 코드를 후보 코드로 검색한 다음, 이를 종합하여 최종 코드를 예측합니다. 이 과정에서 LLM(대형 언어 모델)의 파라메트릭 지식을 보완하기 위해 검색 및 재정렬 기술을 통합하여 성능을 향상시킵니다.

- **Performance Highlights**: MedCodER는 ICD 코드 예측에서 0.60의 micro-F1 점수를 달성하여 현재 최고의 방법보다 유의미하게 향상된 성과를 보입니다. 또한, 제안된 데이터셋은 질병 진단과 ICD 코드, 그리고 이를 정당화하는 지원 증거 텍스트가 주석 처리되어 있어, 코드 선택의 신뢰성을 높이는 데 기여합니다.



### VERA: Validation and Enhancement for Retrieval Augmented systems (https://arxiv.org/abs/2409.15364)
- **What's New**: VERA는 Retrieval-Augmented Generation (RAG) 시스템을 위한 평가 및 개선 시스템으로, LLM의 응답精度를 향상시키기 위한 새로운 방법을 제공합니다. 또한, VERA는 외부 정보를 효과적으로 활용하도록 설계되었습니다.

- **Technical Details**: VERA는 수집된 컨텍스트의 적합성과 불필요한 정보를 제거하는 데 중점을 둡니다. 이 시스템은 평가자 및 향상 LLM을 사용하여 응답 생성 전에 컨텍스트를 평가하고, 응답 생성 후에는 응답을 분리하여 각 문장의 적합성을 점검합니다.

- **Performance Highlights**: 실험 결과, VERA는 소규모 공개 오픈 소스 모델에서 뿐만 아니라 대규모 최첨단 모델에서도 성능을 개선하는 데 뛰어난 효능을 나타냈습니다. VERA는 정보 생성에서 높은 정확성 및 신뢰성을 요구하는 응용 프로그램에 유용한 도구로 자리 잡을 잠재력을 보여주고 있습니다.



### Multitask Mayhem: Unveiling and Mitigating Safety Gaps in LLMs Fine-tuning (https://arxiv.org/abs/2409.15361)
Comments:
          19 pages, 11 figures

- **What's New**: 최근의 Large Language Models (LLMs)에 대한 연구는 다양한 다운스트림 작업에서 가벼운 튜닝이 안전성을 저해할 수 있음을 보여줍니다. 특히, 코드 생성 및 번역 작업에서 안전성 감소가 두드러지며, 새로운 멀티태스크 안전 데이터셋인 MultiTaskBench를 개발하여 이러한 문제를 해결하고자 했습니다.

- **Technical Details**: 이 연구는 네 가지 작업(요약, 코드 생성, 번역, 분류)에 대한 데이터셋을 사용하여 LLM의 튜닝 및 안전성 감소 현상을 분석합니다. 연구에서는 거짓 응답을 방지하기 위해 Reinforcement Learning from Human Feedback (RLHF) 방식을 활용하며, 선행 연구와의 차별성을 위해 안전하게 생성된 데이터셋을 적용합니다.

- **Performance Highlights**: 연구 결과, LLM은 번역 및 분류 작업에서 상대적으로 안전성 유지가 어려웠으며, 코드 데이터로 튜닝할 경우 가장 높은 안전성 저하를 보였습니다. 제안된 MultiTaskBench 데이터셋은 다양한 다운스트림 작업에서 공격 성공률을 효과적으로 감소시켰습니다.



### Watch Your Steps: Observable and Modular Chains of Though (https://arxiv.org/abs/2409.15359)
- **What's New**: 이 논문에서는 Program Trace Prompting (PTP)이라는 새로운 형태의 chain of thought (CoT) 프롬프트를 제안합니다. 이 방법은 CoT의 장점과 유연성을 유지하면서 설명 과정을 보다 관찰 가능하게 만들어 줍니다.

- **Technical Details**: PTP는 Python 기반의 형식을 사용하여 CoT 데모를 포장하고, 각 프롬프트에서는 단계 식별, 입력/출력 동작 정의, CoT 설명을 공식화된 단계로 대체하는 워크플로우를 제공합니다. 이 방법은 다양한 작업에 적용 가능하며, BIG-Bench Hard 벤치마크에서 23개의 작업에 대해 강력한 성능을 보여줍니다.

- **Performance Highlights**: PTP는 대부분의 작업에서 CoT 프롬프트와 비슷한 정확도를 달성하며, 생성된 트레이스는 99% 이상의 법적 단계로 파싱할 수 있습니다. 또한 PTP는 개별 단계 실행과 작업 전체 해결을 모두 가능하게 하며, 대부분의 단계에서 모듈성과 지역성을 평가할 수 있습니다. 실험 결과는 PTP의 유용성을 검증하고, 많은 비국소 오류가 잘못된 알고리즘 추정에서 유래함을 보여줍니다.



### Evaluating Large Language Models with Tests of Spanish as a Foreign Language: Pass or Fail? (https://arxiv.org/abs/2409.15334)
- **What's New**: 본 논문은 Large Language Models (LLMs)이 비영어권 사용자들을 위한 다른 언어, 특히 스페인어에 대한 이해도를 평가한 최초의 연구 중 하나입니다.

- **Technical Details**: TELEIA라는 새로운 벤치마크를 사용하여 LLM의 성능을 평가하였으며, 이 벤치마크는 외국인 학생을 위한 스페인어 시험과 유사한 질문들로 구성되어 있습니다. 평가 항목에는 읽기 이해(reading comprehension), 단어 형성(word formation), 의미(meaning) 및 구성 의미론(compositional semantics), 문법(grammar) 등이 포함됩니다.

- **Performance Highlights**: 결과적으로, LLMs는 스페인어 이해에서는 좋은 성능을 보였으나, 문법적 능력(grammatical competence) 면에서는 여전히 원어민 수준에는 미치지 않는 것으로 나타났습니다.



### Towards Enhancing Linked Data Retrieval in Conversational UIs using Large Language Models (https://arxiv.org/abs/2409.16220)
Comments:
          This paper has been accepted at the 25th International Web Information Systems Engineering Conference (WISE 2024)

- **What's New**: 이 논문은 기존 정보 시스템과 LLMs(대형 언어 모델)의 통합을 통해 Linked Data(LD) 및 RDF(Ressource Description Framework) 트리플스토어에서 데이터를 추출하고 탐색하는 방법을 탐구합니다. 특히, 모델 재훈련 없이도 더 정확한 SPARQL 쿼리를 생성할 수 있는 대화형 사용자 인터페이스(UI)의 강화를 강조합니다.

- **Technical Details**: 본 연구에서는 ForestQB라는 새로운 툴킷을 사용하여 관찰적 LD 데이터로부터 정보를 추출하고, 이 툴킷은 챗봇과 폼 기반 GUI를 통합하여 SPARQL 쿼리를 구성하고 실행합니다. 연구의 초점은 LLMs의 자연어 이해 능력을 활용하여 RDF 엔티티 추출의 정확성을 향상시키는 것입니다.

- **Performance Highlights**: 본 연구의 결과, 제안된 방법론을 통해 시스템의 표현력과 사용자 쿼리에 대한 응답 정확성이 크게 향상되었습니다. 평가 결과는 LLMs가 복잡한 데이터 환경에서 엔티티 추출 및 사용자 인터랙션을 개선시킬 수 있는 가능성을 제시하고 있습니다.



### Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering (https://arxiv.org/abs/2409.16167)
- **What's New**: 본 논문은 LoRA(저랭크 적응)을 조합하여 대형 언어 모델(LLM)의 성능을 극대화하는 새로운 접근 방식을 소개합니다. 기존의 LoRA 합성 방법이 주어진 특정 작업에 최적화되어 추가 훈련을 필요로 하는 반면, 본 연구에서는 LoRA 구성 요소를 독립적인 최소 의미 단위(MSU)로 분해 및 재조립하는 방식을 제안합니다.

- **Technical Details**: 제안된 LoRA-LEGO 프레임워크는 여러 LoRA에서 MSU를 클러스터링하여 새로운 LoRA를 구성하는 과정을 포함합니다. 이 과정은 세 가지 주요 단계로 나누어집니다: (1) 후보 LoRA로부터 MSU 풀(pool)을 만들기, (2) 이 MSU 풀을 k 클러스터로 그룹화하기, (3) 클러스터의 중심을 활용해 병합된 LoRA 구성하기. 이를 통해 파라미터 간섭을 해결하면서 다양한 랭크의 LoRA를 유연하게 조합할 수 있습니다.

- **Performance Highlights**: LoRA-LEGO는 다양한 벤치마크에서 기존의 LoRA 병합 방법보다 우수한 성능을 보였습니다. 실험 결과, LoRA-LEGO는 목표 랭크 k에 맞춘 병합 LoRA를 구성할 수 있을 뿐만 아니라, 개별 LoRA에 적용 시에도 파라미터 감소를 통해 원래 모델과 유사한 성능을 달성할 수 있음을 보여주었습니다.



### HA-FGOVD: Highlighting Fine-grained Attributes via Explicit Linear Composition for Open-Vocabulary Object Detection (https://arxiv.org/abs/2409.16136)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 논문은 Open-Vocabulary Object Detection (OVD) 모델에서 세부 속성을 강조하는 새로운 접근 방식을 제안하여 기존 모델의 성능을 향상시키는 방법을 소개합니다.

- **Technical Details**: 이 방법은 1) Attribute Word Extraction, 2) Attribute Feature Extraction, 3) Attribute Feature Enhancement의 세 가지 주요 프로세스로 구성됩니다. 강력한 LLM(대규모 언어 모델)을 이용해 입력 텍스트에서 속성 단어를 추출하고, 전략적으로 토큰 마스크를 조정하여 OVD 모델의 텍스트 인코더가 전역 텍스트와 속성 특정 피처를 추출합니다. 이 피처들은 선형 조합을 통해 새로운 속성 강조 피쳐로 통합됩니다.

- **Performance Highlights**: FG-OVD 데이터셋에서 실험한 결과, 제안된 방법이 다양한 OVD 모델의 세부 속성 인식 능력을 일관되게 향상시키며 새로운 최첨단 성능을 달성함을 입증하였습니다.



### Implicit assessment of language learning during practice as accurate as explicit testing (https://arxiv.org/abs/2409.16133)
- **What's New**: 이번 연구에서는 Intelligent Tutoring Systems (ITS)에서 학습자의 능력을 평가하기 위해 Item Response Theory (IRT)를 활용합니다. 기존의 포괄적인 테스트 방식 대신, 효율적이면서도 정확한 적응형 테스트(adaptive tests) 개발을 목표로 하고 있습니다.

- **Technical Details**: 연구는 학습자로부터 수집된 데이터를 바탕으로 IRT 모델을 훈련시키고, 이를 통해 적응형 테스트를 안내하는 방식을 사용합니다. 또한, 연습 세션(exercise sessions) 중에 수집된 데이터를 IRT 모델링에 적합한 형태로 변환하는 과정을 진행하며, 언어적 구성(linguistic constructs)을 '항목(items)'으로 연결하여 IRT 모델에 통합합니다.

- **Performance Highlights**: 대규모 연구 결과, 교사의 학습자 능력 평가를 '기준 진리(ground truth)'로 삼고, 테스트와 연습을 통해 얻은 능력 추정치를 비교한 결과, IRT 모델이 연습 기반의 능력 추정에서도 정확성을 발휘함을 확인했습니다.



### MOSS: Enabling Code-Driven Evolution and Context Management for AI Agents (https://arxiv.org/abs/2409.16120)
- **What's New**: MOSS (llM-oriented Operating System Simulation)이라는 새로운 프레임워크를 도입하여 코드 생성과 동적 컨텍스트 관리 시스템을 통합함으로써 AI 에이전트의 적응성과 일관성을 향상시킴.

- **Technical Details**: MOSS는 Python 실행 컨텍스트를 유지하고 지역 변수를 격리하여 여러 상호작용 간의 일관성을 보장하는 메커니즘을 사용합니다. Inversion of Control (IoC) 컨테이너와 데코레이터를 활용하여 가장 낮은 지식 원칙을 적용하며, 이로 인해 에이전트가 구체적인 구현보다는 추상 인터페이스에 집중할 수 있게 합니다.

- **Performance Highlights**: MOSS 프레임워크는 에이전트 개발의 효율성과 기능을 향상시키며, Turing-complete 에이전트를 생성할 수 있는 새로운 가능성을 보여줍니다. 다양한 실제 사례를 통해 에이전트가 코드 생성을 통해 스스로의 역량을 확장할 수 있는 것을 입증하였습니다.



### StyleSinger 2: Zero-Shot Singing Voice Synthesis with Style Transfer and Multi-Level Style Contro (https://arxiv.org/abs/2409.15977)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 이번 논문에서는 스타일 전송 및 스타일 제어를 위한 첫 번째 제로샷(Zero-shot) Singing Voice Synthesis (SVS) 모델인 StyleSinger 2를 소개합니다. 이 모델은 미지의 음색과 스타일을 가진 고품질의 노래 목소리를 생성하는 것을 목표로 하며, 이러한 과제를 해결하기 위해 다단계 스타일 제어를 가능하게 합니다.

- **Technical Details**: StyleSinger 2는 세 가지 주요 모듈로 구성됩니다: 1) Clustering Style Encoder는 스타일 정보를 안정적으로 집약하기 위해 클러스터링 벡터 양자화(clustering vector quantization) 모델을 사용합니다. 2) Style and Duration Language Model (S&D-LM)은 스타일 정보와 음소 지속 시간을 동시에 예측하여 서로의 성능을 향상시킵니다. 3) Style Adaptive Decoder는 독창적인 멜 스타일 적응 정규화(mel-style adaptive normalization) 방법을 통해 세부적인 노래 목소리를 생성합니다.

- **Performance Highlights**: 실험 결과, StyleSinger 2는 여러 작업에서 합성 품질, 가수 유사도, 스타일 제어 가능성 면에서 모든 기준 모델을 초월합니다. 이 모델은 제로샷 스타일 전송, 다단계 스타일 제어, 크로스-링골 스타일 전송, 음성을 노래로 전환하는 스타일 전송 작업에서도 뛰어난 성과를 보였습니다.



### BeSimulator: A Large Language Model Powered Text-based Behavior Simulator (https://arxiv.org/abs/2409.15865)
Comments:
          7 pages, 3 figures, 2 tables

- **What's New**: 본 논문에서는 기존의 로봇 시뮬레이터의 한계를 극복하기 위해 행동 시뮬레이션(Behavior Simulation)을 이론적으로 정의하고 새로운 프레임워크인 BeSimulator를 소개하였습니다. BeSimulator는 텍스트 기반의 가상 환경에서 로봇 행동 로직을 수립하여 시뮬레이션하고, 체계적인 사고 과정을 통해 행동의 실행 가능성과 상태 전이를 분석합니다.

- **Technical Details**: BeSimulator는 모듈화된 LLM(large language model) 기반의 프레임워크로, 행동 계획 솔루션(BPS)에 대한 단계별 시뮬레이션을 수행합니다. 이 프레임워크는 '사례 생성'(Case Generation), 'BPS 시뮬레이션'(BPS Simulation), 및 'BPS 평가'(BPS Evaluation)라는 세 가지 핵심 모듈을 포함하고 있습니다. 또한, Chain of Behavior Simulation(CBS) 접근법을 통해 행동의 실행 가능성과 상태 전이를 깊이 분석합니다.

- **Performance Highlights**: BTSIMBENCH라는 행동 트리 기반의 시뮬레이션 벤치마크를 통해 실험한 결과, BeSimulator는 기존 방법들에 비해 14.7%에서 26.6%까지 행동 시뮬레이션 성능이 향상되었습니다. 이는 BeSimulator가 특히 긴 기간의 복잡한 시뮬레이션에서 우수한 성능을 제공함을 입증합니다.



### iGAiVA: Integrated Generative AI and Visual Analytics in a Machine Learning Workflow for Text Classification (https://arxiv.org/abs/2409.15848)
- **What's New**: 이 논문에서는 텍스트 분류를 위한 기계 학습(ML) 모델 개발 시 데이터 부족 문제를 해결하기 위해 시각적 분석(Visual Analytics, VA)을 활용하여 합성 데이터를 생성하는 방법을 제안합니다. 이는 대규모 언어 모델을 활용하여 특정 데이터 부족 문제를 목표로 한 데이터 합성을 가능하게 합니다.

- **Technical Details**: 다양한 데이터 부족 유형을 논의하고, 이러한 부족을 식별하기 위한 다양한 VA 기법을 설명합니다. 또한, iGAiVA라는 소프트웨어 도구를 소개하여 4개의 ML 작업 그룹을 4개의 VA 뷰에 매핑하고, 생성적 AI와 VA를 ML 워크플로우에 통합했습니다.

- **Performance Highlights**: 대상 데이터 합성을 통해 모델 정확도를 향상시키는 효과를 입증했습니다. 이 연구는 ML 텍스트 분류 모델의 개발 및 개선을 위한 새로운 접근 방식을 제공합니다.



### Supervised Fine-Tuning: An Activation Pattern Optimization Process for Attention Heads (https://arxiv.org/abs/2409.15820)
Comments:
          in review

- **What's New**: 이번 논문에서는 LLM의 성능 향상을 위한 새로운 통찰력을 제시합니다. SFT(Supervised Fine-tuning) 과정에서의 주의 패턴(Attention Patterns)을 분석하여 LLM들이 복잡한 작업에 어떻게 적응하는지를 설명합니다.

- **Technical Details**: 저자들은 경량화된 gradient 기반 방법을 사용하여 SFT 과정에서 LLM의 주의 헤드(Attention Heads)가 어떻게 활용되는지를 밝혀냈습니다. 주요 발견은 LLM이 특정 작업에 대해 선택적으로 주의 헤드를 활성화한다는 것입니다.

- **Performance Highlights**: 논문에서 제안한 접근법을 통해 복잡한 작업을 수행할 수 있는 LLM의 성능이 현저히 향상될 수 있음을 보여줍니다. 특히, 주의 패턴을 분석하여 희소한 지침을 활용하면서도 효율성을 극대화할 수 있는 방법을 실험했습니다.



### AsthmaBot: Multi-modal, Multi-Lingual Retrieval Augmented Generation For Asthma Patient Suppor (https://arxiv.org/abs/2409.15815)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 다국적, 다중 모드 리트리벌 증강 생성(Retrieval-Augmented Generation, RAG) 시스템인 AsthmaBot을 소개합니다. AsthmaBot은 최신 정보와 관련된 문서, 비디오 및 이미지를 통합하여 천식 관련 질문에 대한 답변을 제공합니다.

- **Technical Details**: AsthmaBot은 리트리벌 알고리즘(retrievers)과 다중 모드 지원 기능이 있는 대형 언어 모델(LLM)로 구성되어 있습니다. 이 시스템은 질문-답변 쌍을 바탕으로 정보를 제공하며, 사용자에게 텍스트, 이미지, 비디오가 포함된 응답을 제공합니다. FAQ 데이터를 통해 평가되었습니다.

- **Performance Highlights**: 실험 결과, AsthmaBot은 RAG 기법을 사용하지 않는 기본 모델에 비해 다양한 언어와 모드에서 우수한 성능을 보였습니다. 이 시스템은 사용자 인터페이스를 통해 일반 대중이 손쉽게 접근할 수 있도록 설계되었습니다.



### Federated Large Language Models: Current Progress and Future Directions (https://arxiv.org/abs/2409.15723)
- **What's New**: 이번 논문은 Federated Learning (FL)과 Large Language Models (LLMs) 간의 상호작용을 심도 있게 분석하여 최신 발전 사항과 향후 방향성을 제시합니다. 특히, 데이터 프라이버시와 함께 LLMs의 학습 효율성을 개선하기 위해 각종 방법론과 과제를 살펴봤습니다.

- **Technical Details**: 논문은 LLMs의 Federated Learning(FedLLM)에서의 최신 기법을 요약하고, 세 가지 주요 측면인 데이터 이질성(data heterogeneity), 개인화(personalization), 및 보안(security) 문제를 다룹니다. 또한, FedPIT, FFA-LoRA, 그리고 FEDML-HE와 같은 특정 기법들을 소개하여 성능 및 프라이버시를 증대시키는 다양한 접근법을 제시합니다.

- **Performance Highlights**: 이 논문에서 제안된 다양한 방법론들은 LLMs의 훈련 효율을 극대화 하며, 데이터 통신 비용을 최소화합니다. 예를 들어, FedKSeed와 FedRDMA는 통신 요구량을 대폭 줄이며, 여러 가지 협업 학습 시나리오에서 효과적인 성과를 보여줍니다.



### Making Text Embedders Few-Shot Learners (https://arxiv.org/abs/2409.15700)
- **What's New**: 본 논문에서는 큰 언어 모델(LLM)의 인-context learning (ICL) 기능을 활용하여 텍스트 임베딩 생성 과정을 개선하는 새로운 모델 bge-en-icl을 제안합니다. 이 모델은 적은 수의 샘플을 사용하여 고품질의 텍스트 임베딩을 생성합니다.

- **Technical Details**: bge-en-icl 모델은 쿼리 프롬프트에 작업 관련 예제를 통합하여 LLM의 ICL 능력을 최대한 활용합니다. 이 연구에서는 다양한 attention 메커니즘과 pooling 방법을 평가하여 LLM을 임베딩 모델로 효과적으로 활용하는 방법을 조사했습니다.

- **Performance Highlights**: bge-en-icl 모델은 MTEB 및 AIR-Bench 벤치마크에서 새로운 최첨단(SOTA) 성능을 달성하였으며, 간단한 ICL 전략만으로도 뛰어난 성과를 거둘 수 있다는 것을 입증했습니다. 코드와 데이터셋은 자유롭게 제공됩니다.



### dnaGrinder: a lightweight and high-capacity genomic foundation mod (https://arxiv.org/abs/2409.15697)
- **What's New**: dnaGrinder는 유전자 서열 내의 복잡한 장기 종속성을 효과적으로 관리하면서도 계산 비용을 최소화하는 독창적이고 효율적인 유전체 모델로, 기존의 모델들보다 우수한 성능을 보여줍니다.

- **Technical Details**: dnaGrinder는 Byte Pair Encoding (BPE) 토크나이제이션을 사용하여 DNA 서열을 수치 표현으로 변환하고, Attention with Linear Bias (ALiBi) 기법을 사용하며, Flash Attention 2와 같은 고급 주의 메커니즘을 통합하여 성능과 효율성을 극대화합니다.

- **Performance Highlights**: dnaGrinder는 Nucleotide Transformer 및 DNABERT-2와 같은 최신 DNA 모델에 비해 성능이 동등하거나 우수하며, 단일 고성능 GPU에서 140,000 토큰 이상의 서열을 지원합니다.



### Language-based Audio Moment Retrieva (https://arxiv.org/abs/2409.15672)
- **What's New**: 이번 논문에서는 새로운 작업인 Audio Moment Retrieval (AMR)을 제안하고 설계하였습니다. AMR은 주어진 자연어 쿼리를 바탕으로 잘리지 않은 긴 오디오에서 관련 순간을 예측하는 것을 목표로 합니다. 기존의 언어 기반 오디오 검색 방법과는 달리, AMR은 특정 시간 구간을 추출하고자 합니다.

- **Technical Details**: AMR을 위한 데이터셋 Clotho-Moment를 구축하고, DETR 기반의 오디오 모멘트 모델인 Audio Moment DETR (AM-DETR)을 제안합니다. 이 모델은 오디오 특징 간의 시간적 의존성을 캡처하여 기존의 클립 수준 오디오 검색 방법과의 차별점을 보입니다. 또한, 수동으로 주석이 달린 데이터셋을 통해 실제 데이터에서 방법론의 효과를 측정할 수 있습니다.

- **Performance Highlights**: 실험 결과, Clotho-Moment로 훈련된 AM-DETR은 슬라이딩 윈도우 기반의 클립 수준 검색 방법을 사용하는 기초 모델을 모든 평가 지표에서 능가했습니다. 특히, Recall1@0.7에서 9.00 포인트 향상된 결과를 보였습니다.



### MMPT: Multimodal Prompt Tuning for Zero-shot Instruction Learning (https://arxiv.org/abs/2409.15657)
Comments:
          EMNLP 2024

- **What's New**: 본 연구에서는 Multimodal Prompt Tuning (MMPT) 접근 방식을 도입하여 Multimodal Large Language Models (MLLMs)의 효율적인 instruction tuning을 수행합니다. MMPT는 비주얼 및 텍스트 프롬프트를 효과적으로 통합하여 여러 모드 간의 피쳐 정렬을 촉진합니다.

- **Technical Details**: MMPT는 0.09%의 전체 파라미터를 조정하여 경쟁력 있는 성능을 발휘하고, 시각적 입력과 텍스트 입력에 각각 비주얼 프롬프트와 텍스츄얼 프롬프트를 추가하는 방식으로 작동합니다. 프롬프트 간의 상호 작용을 통해 두 모드 간의 피쳐 표현을 조화롭게 학습합니다.

- **Performance Highlights**: MMPT는 여러 멀티모달 평가 데이터셋에서 여러 최첨단 기준 모델들에 비해 우수한 성능을 보였으며, 파라미터 효율적인 fine-tuning 기법으로 그런 성과를 달성하였습니다.



### Asking an AI for salary negotiation advice is a matter of concern: Controlled experimental perturbation of ChatGPT for protected and non-protected group discrimination on a contextual task with no clear ground truth answers (https://arxiv.org/abs/2409.15567)
- **What's New**: 이 연구는 네 가지 버전의 ChatGPT를 대상으로 진행된 통제된 실험 편향 감사(bias audit)를 소개합니다. 연구진은 각각의 버전에게 새로운 채용자를 위한 초봉 제안을 요청했으며, 직원의 성별, 대학, 전공 등을 체계적으로 변화시켜 98,800개의 프롬프트를 제출했습니다. 이러한 실험을 통해 ChatGPT가 이러한 작업에 신뢰할 수 없다는 것을 발견했습니다.

- **Technical Details**: 대조군을 포함한 제어된 실험 방법을 사용하여 AI가 질문에 대해 차별적인 응답을 하는지 혹은 공정한 결과를 도출하는지를 평가했습니다. 특히 성별 변화에 따른 통계적으로 유의미한 초봉 제안 차이를 관찰했으며, 사기성 대학 및 전공에 대한 경험적 결과도 비교적 일관되지 않음을 확인하였습니다. 이 연구는 AI/ML의 공정성(fairness) 및 신뢰성(trustworthiness) 문헌에 기여합니다.

- **Performance Highlights**: ChatGPT의 네 가지 모델 버전 간의 초봉 차이가 상이하였으며, 성별, 대학 및 전공에 따라 상당한 격차가 관찰되었습니다. 특히, 고용주와 직원의 목소리에 따라 제안된 급여의 차이가 두드러졌고, 이는 ChatGPT 다중 모델 플랫폼의 일관성 및 신뢰성에 대한 우려를 제기합니다.



### Revise, Reason, and Recognize: LLM-Based Emotion Recognition via Emotion-Specific Prompts and ASR Error Correction (https://arxiv.org/abs/2409.15551)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)이 감정 인식에 효과적일 수 있음을 보여주었으나, 이들의 효용성에 대한 의문이 여전히 존재합니다. 본 논문에서는 음향학(acoustics), 언어학(linguistics), 심리학(psychology)에서의 감정별 지식을 통합한 새로운 프롬프트(prompts)를 제안하고, LLM 기반의 감정 인식의 정확성과 효과성을 실험을 통해 검증합니다.

- **Technical Details**: 우리는 Revise-Reason-Recognize (R3) 프롬프트 파이프라인을 제안하여, 부정확한 텍스트에 대한 감정 인식을 개선합니다. 이 파이프라인은 ASR(Automatic Speech Recognition) 오류를 수정하고, LLM이 감정 인식에 필요한 자율적 설명을 제공하는 방법으로 구성됩니다. 이 외에도, 문맥 인식 학습(context-aware learning)과 지시 조정(instruction tuning) 방법을 실험하여 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 감정별 지식을 통합한 프롬프트와 ASR 오류 교정이 LLM 기반 감정 인식에 효과적임을 확인하였습니다. R3 프롬프트는 ASR 전사 과정에서 발생하는 오류를 정확히 수정하고, 감정 인식의 정확도를 높이는 데 기여하였습니다. 또한, 훈련 스킴들은 LLM의 성능을 향상시키는 데 중요한 역할을 합니다.



### Rethinking Emotion Bias in Music via Frechet Audio Distanc (https://arxiv.org/abs/2409.15545)
- **What's New**: 이 연구 논문은 음악 감정 인식(MER)과 감정 음악 생성(EMG)에서 음악 감정 평가의 객관성을 높이기 위해 여러 오디오 인코더와 Frechet Audio Distance(FAD) 평가 척도를 사용했습니다. 단일 오디오 인코더에 의존할 때의 한계를 강조하며, 다양한 인코더를 통한 평가 방법을 제안합니다.

- **Technical Details**: 연구에서는 Emomusic, MMirex, EMOPIA 등 다양한 데이터셋을 활용하여 MER과 EMG의 객관적인 평가를 진행했습니다. FAD를 사용하여 MUSIC 감정의 객관적인 측정을 목표로 하며, 오디오 인코더의 종류에 따라 성능 차이를 분석합니다. 또한, EMG 접근 방식을 개선하여 생성된 음악 감정의 변동성과 현실성을 높이고자 합니다.

- **Performance Highlights**: 실험 결과는 MER과 EMG 모두 감정 편향 문제를 잘 보여주며, 다중 오디오 인코더와 FAD의 사용이 음악 감정 평가의 객관성을 높일 수 있는 가능성을 나타냅니다. 연구의 접근 방식은 기존의 방법들과 비교해 더 나은 결과를 나타내며, 현실 감정 표현의 향상을 목표로 합니다.



### RAM2C: A Liberal Arts Educational Chatbot based on Retrieval-augmented Multi-role Multi-expert Collaboration (https://arxiv.org/abs/2409.15461)
- **What's New**: 본 연구에서는 Retrieval-Augmented Multi-role Multi-expert Collaboration (RAM2C) 프레임워크를 제안하여 고품질의 자유 예술 교육 대화를 자동으로 생성하고, 이 데이터를 통해 LLM(대형 언어 모델)을 조정하는 방안을 소개합니다.

- **Technical Details**: RAM2C 프레임워크는 T-Group(중국어 교사), P-Group(교육 심리학자), E-Group(윤리적 안전 전문가)의 세 가지 전문가 그룹을 구성하여 다중 역할과 다중 전문가 협업을 통해 HTS(인간화된 소통, 교수 전문성, 안전-윤리) 기준에 부합하는 교육 대화를 생성합니다.

- **Performance Highlights**: RAM2C를 통해 생성된 LLM은 특히 문학 교육에서 높은 개인화된 응답과 윤리적으로 안전한 교육 반응을 제공하며, 실험 결과 미세 조정된 모델이 GLM-4와 유사한 성능을 보였습니다.



### The ParlaSpeech Collection of Automatically Generated Speech and Text Datasets from Parliamentary Proceedings (https://arxiv.org/abs/2409.15397)
Comments:
          Submitted to SPECOM 2024

- **What's New**: 이 논문에서는 자원이 부족한 언어의 대량 공개 음성과 텍스트가 정렬된 데이터셋을 구축하는 새로운 접근 방법을 제시합니다. 특히, 크로아티아어, 폴란드어, 세르비아어를 중심으로 한 의회 기록과 그에 따른 음성을 정렬시켜 고품질 데이터셋을 제공합니다.

- **Technical Details**: 이 연구는 ParlaMint 프로젝트의 전 과정을 활용하여, 의회에서의 발언을 기록한 음성과 해당 텍스트의 정렬 문제를 다룹니다. 이 과정에서 장기 오디오 캡쳐를 단어 수준 타임스탬프와 정렬하기 위한 현대적인 방법론을 제안하며, 기존의 복잡한 정렬 문제를 해결하기 위한 노력이 포함됩니다.

- **Performance Highlights**: 최종 결과로 5,000시간 이상의 연설과 텍스트 전사 데이터로 구성된 3개의 고품질 데이터셋이 생성되었습니다. 이 데이터셋은 대상 언어에 대한 음성 및 텍스트 데이터 접근성을 크게 향상시키며, 향후 비슷한 방식으로 더 많은 언어로 확장할 수 있는 잠재력을 지니고 있습니다.



### Toward Automated Clinical Transcriptions (https://arxiv.org/abs/2409.15378)
Comments:
          7 pages, 6 figures

- **What's New**: 본 논문은 최근의 speech-to-text (음성 인식) 및 speaker-labeling (화자 레이블링) 기술을 활용하여 환자-제공자 대화의 정확한 전사를 생성하고, 오류를 강조하여 신속한 인간 검증을 촉진하는 안전한 시스템을 소개합니다.

- **Technical Details**: 이 시스템은 40시간 이상의 시뮬레이션 대화에 적용되어 최적화 되었으며, 의료 문서화의 수작업 노력을 줄이기 위해 설계되었습니다. 특히, 불필요한 수작업을 최소화하여 임상 전사(Clinical Transcriptions)의 자동화를 위한 유망한 기초를 제공합니다.

- **Performance Highlights**: 이 시스템은 정확한 전사를 생성하는 데 있어 뛰어난 성능을 보이며, 의료 분야에서 피로도 증가 및 환자 관리 품질 저하와 같은 부정적인 결과를 완화하는 데 기여할 것으로 기대됩니다.



### ControlMath: Controllable Data Generation Promotes Math Generalist Models (https://arxiv.org/abs/2409.15376)
Comments:
          17 pages

- **What's New**: 본 연구에서는 데이터 증강(data augmentation)에 있어 대형 언어 모델(LLMs)의 제약 사항을 극복하기 위해 ControlMath라는 반복(iterative) 방법론을 소개합니다.

- **Technical Details**: ControlMath는 방정식 생성기(equation-generator) 모듈과 두 개의 LLM 기반 에이전트(agent)를 포함합니다. 방정식 생성 모듈은 다양한 방정식을 생성하고, Problem-Crafter 에이전트는 이를 수학적인 서술 문제로 변환합니다. Reverse-Agent는 'less is more' 원칙에 따라 고품질 데이터를 필터링하고 선택합니다.

- **Performance Highlights**: ControlMathQA는 190,000개의 수학 서술 문제(math word problems)를 포함하고 있으며, 이 데이터셋은 GSM8K와 같은 도메인 내(in-domain) 데이터셋과 결합함으로써 모델의 수학적 일반화(generalization) 능력을 향상시킵니다. 결과적으로 특정 도메인뿐만 아니라 그 너머에서도 성능이 개선되는 것을 보였습니다.



### Fine-Tuning a Time Series Foundation Model with Wasserstein Loss (https://arxiv.org/abs/2409.15367)
Comments:
          4 main pages; 2 figures

- **What's New**: 이번 연구는 시간 시계열 예측을 위한 기초 모델 개발에 있어 최근의 대형 언어 모델(LLM) 발전에 힘입어, cross-entropy loss 대신 Wasserstein loss를 사용하는 방법을 제안하고 있습니다.

- **Technical Details**: 연구진은 LLM 아키텍처를 토큰화된 시간 시계열 데이터로 교육하여 cross-entropy loss로 주어진 모델을 정밀 조정하였습니다. Wasserstein loss는 클래스 간 거리 정보를 반영하며, 성능 비교를 통해 예측 정확도를 개선하는 것이 입증되었습니다.

- **Performance Highlights**: 22개의 zero-shot 데이터셋에서 평가한 결과, cross-entropy loss에 비해 Wasserstein loss를 사용하는 것이 점 추정(point estimation) 성능을 유의미하게 향상시켰습니다.



### Reward-Robust RLHF in LLMs (https://arxiv.org/abs/2409.15360)
- **What's New**: 본 논문에서는 보상 모델의 불안정성과 오류를 해결하기 위한 보상 강건 RLHF(Reward-Robust Reinforcement Learning from Human Feedback) 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 Bayesian Reward Model Ensembles (BRME)를 통해 보상 함수의 불확실성 집합을 모델링하며, 성능과 강건성을 균형 있게 최적화하는 새로운 목표를 설정합니다.

- **Performance Highlights**: 제안하는 프레임워크는 16개의 다양한 벤치마크에서 전통적인 RLHF를 지속적으로 초월하여 평균 정확도가 약 4% 더 높게 나타났으며, 장기 훈련 과정에서도 더 강한 안정성과 향상된 성능을 보여주었습니다.



### A Joint Spectro-Temporal Relational Thinking Based Acoustic Modeling Framework (https://arxiv.org/abs/2409.15357)
- **What's New**: 본 논문은 인간의 언어 인식에서 중요한 역할을 하는 관계적 사고(relational thinking)를 기반으로 한 음성 인식 시스템을 제안합니다. 기존의 시스템들이 대부분 단순한 시간적 모델에 의존하고 있는 반면, 새로운 접근법은 스펙트로-시간적 관계를 모델링합니다.

- **Technical Details**: 제안된 프레임워크는 음성 세그먼트 간의 관계를 시간(time)과 주파수(frequency) 도메인 모두에서 모델링하여, 생성된 확률적 그래프를 통해 관계 정보를 집계하고 잠재 표현(latent representation)으로 변환합니다. 이는 음성 인식의 성능을 개선하는 데 기여합니다.

- **Performance Highlights**: 이 프레임워크를 통해 구축된 모델은 TIMIT 데이터셋에서 음소(phneme) 인식 작업에서 7.82%의 성능 향상을 달성하였고, 특히 모음(vowel) 인식 능력이 크게 개선되었음을 보여주었습니다.



### Block-Attention for Low-Latency RAG (https://arxiv.org/abs/2409.15355)
- **What's New**: 새로운 Block-Attention 메커니즘이 도입되었습니다. 이 방법은 Retrieval-Augmented Generation (RAG) 시나리오에서의 추론 지연을 줄이기 위해 고안되었습니다. 각 입력 시퀀스를 블록으로 나누어 최종 블록을 제외한 각 블록이 독립적으로 키-값(KV) 상태를 계산하게 합니다.

- **Technical Details**: Block-Attention은 입력 시퀀스를 여러 개의 블록으로 나누고, 각 블록은 다른 블록과 상관없이 KV 상태를 계산합니다. RAG 시나리오에서 각 패세지를 블록으로 정의함으로써 모든 패세지에 대한 KV 상태를 미리 계산하고 메모리에 캐시할 수 있습니다. 블록 세그먼트, 위치 인코딩 계산 및 LLM을 Block-Attention 메커니즘에 적응시키기 위한 미세 조정이 포함됩니다.

- **Performance Highlights**: Block-Attention 모델은 fine-tuning을 통해 Llama3에서 68.4%의 성능을 달성하며, 이는 기존의 self-attention 모델(67.9%)과 비슷합니다. 특히, Block-Attention은 TTFT를 평균 98.7% 줄여, 32K 길이의 입력 시퀀스에 대해 첫 번째 토큰을 출력하는 데 단 45ms가 소요됩니다.



### Contextualization of ASR with LLM using phonetic retrieval-based augmentation (https://arxiv.org/abs/2409.15353)
- **What's New**: 이 논문에서는 멀티모달(multi-modal) LLM(대형 언어 모델)을 개인 이름 엔티티를 올바르게 인식하기 위한 새로운 방법을 제안합니다. 구체적으로, 음성 인식 작업에서 LLM이 이름 엔티티를 감지한 후, 개인 데이터베이스에서 음성적으로 유사한 이름 엔티티를 검색하고 이를 LLM에 전달하여 컨텍스트 인식 ASR(자동 음성 인식)을 수행하는 방식을 사용합니다.

- **Technical Details**: 이 연구는 LLM과 오디오 인코더를 통합하고, 이를 통해 오디오 임베딩을 생성하여 ASR 디코딩을 실행합니다. 감지 단계에서 LLM은 음성 신호에서 개인 이름 엔티티를 감지하고, 그 후에 음성적으로 유사한 엔티티를 검색합니다. 검색된 엔티티를 LLM에 전달하여 ASR 디코딩을 통해 최종 결과를 생성합니다. 또한, NPD(정규화된 음향 거리)를 사용하여 음성의 발음 유사성을 측정합니다.

- **Performance Highlights**: 이 방법은 기존의 ASR 시스템과 비교하여 최대 30.2%의 단어 오류율 감소와 73.6%의 이름 엔티티 오류율 감소를 달성했습니다. 이 시스템은 전체 이름 엔티티 데이터베이스를 LLM에 제공하지 않기 때문에 높은 효율성과 대규모 이름 엔티티 데이터베이스에 적용 가능하다는 장점이 있습니다.



### A Large Dataset of Spontaneous Speech with the Accent Spoken in S\~ao Paulo for Automatic Speech Recognition Evaluation (https://arxiv.org/abs/2409.15350)
- **What's New**: 이 논문은 브라질 포르투갈어를 위한 자발적 대화 음성 코퍼스(NURC-SP Audio Corpus)를 새롭게 선보이며, 자동 음성 인식(ASR) 실험 결과를 보고합니다. 이 코퍼스는 포르투갈어의 대화체 발음을 포함하고 있으며, 401명의 화자(여성 204명, 남성 197명)가 참여하여 총 239.30시간의 음성 녹음이 포함되어 있습니다.

- **Technical Details**: NURC-SP Audio Corpus는 Wav2Vec2-XLSR-53 및 Distil-Whisper 모델을 활용하여 ASR 성능을 평가합니다. Wav2Vec2-XLSR-53는 53개 언어의 데이터로 미리 학습된 모델이며, Distil-Whisper는 Whisper 모델의 효율적인 단축 버전입니다. 본 연구에서는 두 모델을 우리의 데이터셋에 맞춰 조정하였습니다. 마지막으로 Distil-Whisper 모델은 NURC-SP Audio Corpus에서 24.22%의 단어 오류율(WER)을 기록했습니다.

- **Performance Highlights**: Distil-Whisper 모델은 24.22%의 WER로 가장 우수한 성과를 거두었으며, Wav2Vec2-XLSR-53 모델은 33.73%의 WER로 뒤를 따릅니다. 이 결과는 자발적 대화의 포르투갈어 음성을 인식할 때 NURC-SP Audio Corpus의 유용성을 입증합니다.



### Revisiting the Solution of Meta KDD Cup 2024: CRAG (https://arxiv.org/abs/2409.15337)
- **What's New**: 이 논문은 Meta KDD CUP 2024의 CRAG Comprehensive RAG Benchmark Challenge에서 팀 APEX의 솔루션을 소개합니다. CRAG 벤치마크는 Retrieval-Augmented Generation (RAG) 시스템의 다양하고 동적인 문제를 평가하는 데 있어 기존 QA 벤치마크의 한계를 극복할 수 있도록 설계되었습니다.

- **Technical Details**: 본 연구에서는 질문의 다양성과 동적인 특성에 맞춘routing 기반의 도메인 및 동적 적응 RAG 파이프라인을 제안합니다. 이 방법은 정보 검색(retrieval), 증대(augmentation), 생성(generation) 세 단계에서 모두 특별한 처리를 수행하며, CRAG에서 우수한 성과를 거두어 최종 경쟁 리더보드에서 작업 2와 3에서 2위를 기록했습니다.

- **Performance Highlights**: 우리의 방법은 CRAG에서 뛰어난 성과를 발휘했으며, 특히 웹 페이지 검색 및 Mock API를 활용해 정보 선택과 통합의 능력을 강조하였습니다. 각 과제는 이전 단계를 기반으로 하여, 참가자들이 더욱 정교한 end-to-end RAG 시스템을 개발하도록 유도합니다.



### Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Mod (https://arxiv.org/abs/2409.15298)
- **What's New**: 이 논문은 Neuromorphic hardware에 적합한 Transformer 기반의 스파이킹 언어 모델인 Sorbet를 소개합니다. Sorbet는 에너지 소모를 줄이면서도 경쟁력 있는 성능을 유지하는 혁신적인 Softmax와 정규화 방법을 도입하여, 기존의 복잡한 연산에 의존하지 않고 동작할 수 있습니다.

- **Technical Details**: Sorbet는 PTsoftmax라는 새로운 shifting 기반 Softmax와 bit-shifting를 이용한 BSBN이라는 파워 정규화 방법을 기반으로 합니다. 이러한 접근 방식을 통해 기존의 에너지를 많이 소모하는 연산을 대체하며, SNN 상에서의 효율적인 동작을 가능하게 합니다.

- **Performance Highlights**: GLUE 벤치마크를 통한 테스트에서 Sorbet는 낮은 에너지 비용으로 안정된 성능을 유지함을 증명했습니다. 이 모델은 또한 Knowledge Distillation 및 모델 양자화를 활용하여 고도로 압축된 바이너리 가중치 모델을 구현했고, 이는 에너지 효율적인 언어 모델 추론의 잠재력을 보여줍니다.



### The NGT200 Dataset: Geometric Multi-View Isolated Sign Recognition (https://arxiv.org/abs/2409.15284)
Comments:
          Proceedings of the Geometry-grounded Representation Learning and Generative Modeling Workshop (GRaM) at the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 251, 2024

- **What's New**: 본 연구에서는 Sign Language Processing (SLP)의 다중 관점 고립 기호 인식(MV-ISR)을 다루며, 3D 인식 및 기하학의 중요성을 강조합니다. 또한 새로운 spatio-temporal multi-view benchmark인 NGT200 데이터셋을 소개합니다.

- **Technical Details**: NGT200 데이터셋은 다중 관점에서 촬영된 고립 기호의 비디오 클립에서 추출한 2D 랜드마크를 포함합니다. 이 데이터셋은 3D-LEX 데이터셋과 함께 사용되어 각 기호에 대한 3D Ground Truth를 제공합니다. 이 연구는 SE(2) 등변 모델을 활용하여 MV-ISR의 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: MV-ISR은 SE(2) 등변 모델을 활용하여 성능이 기준선 대비 8%-22% 향상되었습니다. 이를 통해 기호 인식 시스템의 실용성을 높일 수 있는 방법을 제시합니다.



### A Preliminary Study of o1 in Medicine: Are We Closer to an AI Doctor? (https://arxiv.org/abs/2409.15277)
Comments:
          The first four authors contributed equally, project page available at this https URL

- **What's New**: 본 논문에서는 OpenAI의 새로운 모델 o1이 초기화된 Chain-of-Thought 기법을 사용하여 강력한 언어 모델(Large Language Models, LLMs)의 성능을 강화했음을 소개합니다. 특히, o1의 의료 분야에서의 적용 가능성과 성능을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: o1 모델은 37개의 의료 데이터셋을 바탕으로 6가지 과제를 평가하였으며, 두 개의 도전적인 질문-답변(QA) 데이터를 새로 작성하였습니다. 평가 항목으로는 이해(Understanding), 추론(Reasoning), 다국어 능력(Multilinguality)을 사용하였습니다.

- **Performance Highlights**: o1은 19개의 데이터셋과 두 개의 새로운 복잡한 QA 시나리오에서 GPT-4 보다 평균 6.2% 및 6.6% 더 정확한 성능을 보였습니다. 그러나 모델의 할루시네이션(Hallucination), 일관성 없는 다국어 능력, 평가 지표 간의 차이 등 여러 약점도 발견되었습니다.



### OmniBench: Towards The Future of Universal Omni-Language Models (https://arxiv.org/abs/2409.15272)
- **What's New**: OmniBench는 여러 모달리티(visual, acoustic, textual) 간의 상호작용을 평가하고 모델의 이해 및 추론 능력을 측정하는 새로운 벤치마크입니다. 이 벤치마크는 모든 모달리티 간의 통합된 이해를 요구하여 기존의 한계를 극복하고 있습니다.

- **Technical Details**: 오미벤치(OmniBench)는 미리 훈련된 초대형 언어 모델(MLLMs)의 tri-modal(3중 모달리티) 처리 능력을 테스트하기 위한 포괄적인 도구입니다. OLMs(omni-language models)는 이러한 능력을 갖춘 모델로 정의됩니다. OLMs는 high-quality human annotations에 의존하여 정확한 응답을 제공하는 데 필요한 모든 모달리티의 통합된 이해를 요구합니다.

- **Performance Highlights**: 대부분의 OLMs는 tri-modal 상황에서 지시 수행 및 추론 능력에 한계를 보이는 것으로 나타났습니다. 기존의 MLLM들은 이미지 또는 오디오와 함께 제공되었을 때 명확한 지시를 따르기 어려운 경우가 많으며, 대체 텍스트 표현 사용 시에도 정확도가 50% 미만으로 낮은 성능을 기록했습니다.



### Behavioral Bias of Vision-Language Models: A Behavioral Finance View (https://arxiv.org/abs/2409.15256)
Comments:
          ICML 2024 Workshop on Large Language Models and Cognition

- **What's New**: 이번 연구는 대형 비전-언어 모델(LVLM)의 행동 편향을 행동 재무학의 관점에서 분석한 최초의 연구로, LVLM이 인간과 유사한 결정의 합리성을 발휘하는지 혹은 인간과 유사한 판단 및 결정 편향에 영향을 받는지를 조사합니다.

- **Technical Details**: 연구는 S&P 500 기업의 주식 역사 및 분기별 주당순이익(EPS) 보고서를 포함하는 멀티모달 데이터셋인 DynoStock을 체계적으로 축적하고, recency bias(최근 편향) 및 authority bias(권위 편향)에 대한 프롬프트 템플릿을 설계하여 LVLM의 편향 영향을 평가하는 새로운 평가 프레임워크를 제안합니다.

- **Performance Highlights**: 연구 결과, LLaVA-NeXT, MobileVLM-V2, Mini-Gemini, MiniCPM-Llama3-V 2.5 및 Phi-3-vision-128k와 같은 최근 공개 소스 LVLM들은 두 가지 행동 편향에 심각한 영향을 받는 것으로 나타났습니다. 반면, 독점 모델인 GPT-4o는 편향의 영향에서 거의 영향을 받지 않았습니다.



### MemBench: Towards Real-world Evaluation of Memory-Augmented Dialogue Systems (https://arxiv.org/abs/2409.15240)
Comments:
          In progress

- **What's New**: 이 논문은 기존의 대화 시스템(DS) 평가 방식의 한계를 극복하기 위해 새로운 메모리 벤치마크인 MemBench를 제안합니다. 이는 인지 과학 및 심리학 이론에 기반하여 다양한 메모리 회상 패러다임을 포함하는 완전한 평가 방법론을 제공합니다.

- **Technical Details**: MemBench는 인지 과학의 두 단계 이론에 따라 구성된 두 가지 작업(메모리 회수 및 인지/주입)을 포함하고 있으며, 수동적 및 능동적 메모리 회상을 모두 고려합니다. 이 벤치마크는 새로운 점수 평가 방식을 도입하여 생성된 응답의 다양한 측면을 포괄적으로 측정합니다.

- **Performance Highlights**: 실험 결과, 현재의 대화 시스템이 메모리를 도입하여도 인간과의 대화에서 여전히 성능이 부족한 점이 드러났습니다. 특히, 메모리 주입이 감정 지원(ES) 능력과 친밀도에 긍정적인 연관성이 있음을 발견하였습니다.



### ASTE Transformer Modelling Dependencies in Aspect-Sentiment Triplet Extraction (https://arxiv.org/abs/2409.15202)
Comments:
          The 2024 Conference on Empirical Methods in Natural Language Processing, November 12-16, Miami, Florida 9 pages, appendix, diagrams

- **What's New**: 본 논문에서는 Aspect-Sentiment Triplet Extraction (ASTE)에서의 종속성을 모델링할 수 있는 새로운 접근 방식인 ASTE-Transformer를 제안합니다.

- **Technical Details**: ASTE-Transformer는 세 가지 유형의 transformer-inspired layers로 구성되며, 이는 (1) 표준 transformer layers, (2) aspect-opinion 쌍 생성 layer, (3) triple 생성 layer입니다. 이 구조는 두 개의 문장을 기반으로 aspect와 opinion을 추출하고, 그 종속성을 동시에 고려하여 sentiment polarity를 할당할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 실험 결과, ASTE-Transformer는 기존의 방법들보다 F1 성능이 향상되었으며, pre-training 기술이 모델 성능을 추가적으로 개선시켰습니다.



### Learning from Contrastive Prompts: Automated Optimization and Adaptation (https://arxiv.org/abs/2409.15199)
- **What's New**: LCP( Learning from Contrastive Prompts) 프레임워크는 기존의 프롬프트 최적화 방법의 한계를 극복하고자 하며, 여러 모델 버전과 언어에서 효과적으로 적응 감소를 제공한다.

- **Technical Details**: LCP는 입력 프롬프트의 패턴 분석을 통해 효과적인 프롬프트를 생성하기 위해 대조 학습(contrastive learning) 기법을 활용한다. 주안점은 좋은 프롬프트와 나쁜 프롬프트를 비교하면서 오류 사례를 학습하는 것이다.

- **Performance Highlights**: LCP는 Big-Bench Hard 데이터셋에서 기존 방법들보다 76% 이상의 승률을 기록하며, 특히 알고리즘적 및 단계별 산술 추론 작업에서 효율성을 보였다.



### PALLM: Evaluating and Enhancing PALLiative Care Conversations with Large Language Models (https://arxiv.org/abs/2409.15188)
Comments:
          Accepted by ACM Transactions on Computing for Healthcare, Special Issue on Large Language Models, Conversational Systems, and Generative AI in Health, pending minor revisions

- **What's New**: 본 연구는 대량 언어 모델(LLMs)을 평가자로 활용하여 통증을 포함한 중증 질환을 겪고 있는 환자들을 위한 완화 의료(Palliative Care) 커뮤니케이션의 품질을 평가하는 새로운 접근 방식을 제안합니다. 기존의 NLP 기술이 임상 커뮤니케이션의 뉘앙스를 포착하는 데 어려움을 겪는 반면, LLMs는 언어적, 상황별 학습 및 추론 능력을 활용하여 이를 극복할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구에서는 의료 전문가가 제작하고 라벨링한 8개의 시뮬레이션된 임상 커뮤니케이션 스크립트를 사용하여, GPT-4와 LLaMA2-13b와 같은 LLM을테스트하고 프로프팅 전략을 활용하여 이해, 공감, 감정, 존재감 및 명료성과 같은 커뮤니케이션 지표를 평가했습니다. 알고리즘의 성능 향상을 위해 GPT-4에 의해 생성된 합성 데이터를 활용하여 LLaMA 모델을 미세 조정(fine-tune)하였습니다.

- **Performance Highlights**: 우리의 연구 결과는 GPT 기반 모델이 임상 커뮤니케이션 지표 식별에서 90% 이상의 균형 잡힌 정확도와 LLaMA2-13b 모델이 80%의 정확도를 기록하며 비-LLM NLP 기준을 초과하여 LLM이 임상 커뮤니케이션 분석에 대한 높은 잠재력을 지니고 있음을 보여주었습니다. 이는 LLM 기반 시스템의 개발 가능성과 실제 응용 가능성을 강조합니다.



### Lessons Learned on Information Retrieval in Electronic Health Records: A Comparison of Embedding Models and Pooling Strategies (https://arxiv.org/abs/2409.15163)
- **What's New**: 이번 연구는 의료 분야에서 LLMs의 효과적인 정보 검색을 위한 다양한 embedding 모델과 pooling 방법의 영향을 분석한 것입니다. 특히, BGE라는 일반 도메인 모델이 의료 전용 모델보다 일관되게 우수한 성능을 보인 점이 주목할 만합니다.

- **Technical Details**: 연구는 MIMIC-III와 사설 전자 건강 기록(EHR) 데이터를 사용하는 세 가지 정보 검색 작업을 수행했습니다. 총 일곱 개 모델을 평가했으며, 각 모델에 대한 embedding pooling 전략이 독립적으로 어떻게 작용하는지도 분석했습니다.

- **Performance Highlights**: BGE 모델은 다른 의료 전용 모델보다 뛰어난 검색 성능을 보였으며, 데이터셋과 쿼리 내용에 따라 상당한 변동성이 있음을 발견했습니다. 최적의 pooling 방법을 제안하여 미래의 검색 시스템 디자인에 기여할 수 있는 통계적으로 검증된 권장 사항을 제공했습니다.



### Inferring Scientific Cross-Document Coreference and Hierarchy with Definition-Augmented Relational Reasoning (https://arxiv.org/abs/2409.15113)
- **What's New**: 이 연구에서는 과학 텍스트에서 문서 간 핵심 참조(coreference)와 계층(hierarchy)을 추론하는 새로운 방법을 제안합니다. 이는 지식 그래프 구축, 검색, 추천 및 발견에 중요한 응용 프로그램을 가지고 있습니다.

- **Technical Details**: 제안된 방법은 논문에서 개념 언급에 대한 문맥-의존적 정의를 생성하기 위해 전체 텍스트 문헌을 검색하며, 이러한 정의를 사용하여 문서 간 관계 감지를 향상시키고, 두 개념 언급의 관계를 설명하는 관계 정의(relational definitions)를 생성하는 것입니다. 또한, 두 단계 재정렬 접근방식을 설계하여 문서 간 링크를 추론할 때 조합 폭발(combinatorial explosion)을 방지합니다.

- **Performance Highlights**: SciCo 데이터셋(SciCo dataset)의 기초에서, 우리는 미세 조정(fine-tuning) 및 문맥 학습(in-context learning) 설정에서 모두 현저한 성능 향상을 달성했습니다. 특히, 미세 조정 설정에서는 기존 모델에 비해 CoNLL F1 점수에서 큰 개선을 보였으며, 계층 감지에서 특히 강력한 성과를 나타냈습니다.



### Using Similarity to Evaluate Factual Consistency in Summaries (https://arxiv.org/abs/2409.15090)
- **What's New**: 이 논문에서는 새로운 제로샷(Zero-shot) 사실성 평가 지표인 Sentence-BERT Score(SBERTScore)를 제안하며, 기존의 BERTScore보다 더 뛰어난 성능을 보임을 보여준다. 또한, 다양한 오류 유형 탐지에서 결합된 기술의 효과를 입증한다.

- **Technical Details**: SBERTScore는 생성된 요약과 원본 문서 간의 문장 임베딩을 비교하여 문장 수준에서 사실성을 평가한다. 이 과정에서 코사인 유사도(cosine similarity)를 계산하여 문장의 의미를 더 잘 표현할 수 있다. 실험 결과 SBERTScore는 n-그램(n-gram) 기반의 방법 및 BERTScore보다 우수하며, NLI 및 QA 기반의 사실성 지표와도 경쟁할 수 있다.

- **Performance Highlights**: SBERTScore는 제로샷 설정에서도 NLI 기반 메트릭보다 우수한 성능을 보였고, QA 기반 메트릭과 비슷한 성능을 보여줬다. 또한, 이 지표는 추가적인 학습 단계 없이 고품질의 사전 학습된 임베딩을 활용하여 계산 복잡성이 낮다.



### Depression Diagnosis Dialogue Simulation: Self-improving Psychiatrist with Tertiary Memory (https://arxiv.org/abs/2409.15084)
- **What's New**: 본 논문에서는 Agent Mental Clinic (AMC)라는 자가 개선형 대화형 에이전트 시스템을 소개하여 우울증 진단의 효율성을 높입니다. 이는 환자와 정신과 의사 에이전트 간의 시뮬레이션된 대화를 통해 이루어지며, 진단 정확도를 높이기 위해 정신과 의사 에이전트의 메모리 구조 및 대화 제어 플러그인을 설계하였습니다.

- **Technical Details**: AMC 시스템은 3개의 주요 부분으로 구성되어 있습니다: 1) 환자 에이전트: D4 데이터셋을 기반으로 생성된 다양한 환자들. 2) 정신과 의사 에이전트: 진단 대화를 통해 반영된 기술을 사용하는 에이전트. 3) 감독자 플러그인: 대화 과정을 제어하고 정신과 의사 에이전트의 반영을 촉진하는 불완전한 에이전트. 이러한 구조는 우울증 진단 및 대화 시뮬레이션의 최적화를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, AMC 시스템은 우울증 진단 정확도를 평균 6.05% 향상시켰으며, 자살 예측 정확도는 1.8% 증가했습니다. 이 시스템은 제한된 수의 레이블이 있는 경우에도 다른 특정 도메인에 적용 가능합니다.



### Enhancing Scientific Reproducibility Through Automated BioCompute Object Creation Using Retrieval-Augmented Generation from Publications (https://arxiv.org/abs/2409.15076)
Comments:
          21 pages, 8 figures

- **What's New**: 본 연구에서는 IEEE BioCompute Object (BCO) 표준을 따르는 문서 생성을 자동화하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 Retrieval-Augmented Generation (RAG) 및 Large Language Models (LLMs)을 사용하여 과학 논문에서 BCO를 자동으로 생성하는 도구인 BCO assistant를 개발합니다. 주요 기술적 도전 과제인 LLM 환각(data hallucination) 및 긴 맥락 이해(long-context understanding)에 대한 해결책을 설명하며, 두 단계 검색(two-pass retrieval) 및 재순위화(re-ranking) 과정을 포함한 최적화된 검색 프로세스를 구현하였습니다.

- **Performance Highlights**: BCO assistant는 생명정보학(bioinformatics) 연구의 문서화를 자동화하여 필요한 시간과 노력을 크게 줄일 수 있습니다. 이 접근 방식은 과학적 재현성(scientific reproducibility)을 높이고, AI 지원 과학 문서화 및 지식 추출 가능성을 열어줍니다.



### Brotherhood at WMT 2024: Leveraging LLM-Generated Contextual Conversations for Cross-Lingual Image Captioning (https://arxiv.org/abs/2409.15052)
Comments:
          Accepted at the Ninth Conference on Machine Translation (WMT24), co-located with EMNLP 2024

- **What's New**: 본 논문에서는 Brotherhood라는 팀 이름으로 영어에서 저해상도 다중 모달 번역 작업을 위한 시스템을 소개합니다. 영어-힌디어, 영어-하우사어, 영어-벵골어, 영어-말라얄람어 언어 쌍의 다중 모달 번역 작업에 참여하였습니다.

- **Technical Details**: 본 연구는 전통적인 훈련(fine-tuning) 없이 cross-lingual image captioning을 향상시키기 위해 다중 모달 대규모 언어 모델(multi-modal Large Language Models)인 GPT-4o와 Claude 3.5 Sonnet을 활용하는 방법을 제시합니다. Instruction-tuned prompting을 사용하여 잘라낸 이미지에 대한 풍부한 맥락 대화를 생성하며, 영어 캡션을 추가 맥락으로 사용합니다. 이러한 합성 대화는 타겟 언어로 번역됩니다. 마지막으로, 무게 조절 프롬프트(weighted prompting) 전략을 사용하여 원본 영어 캡션과 번역된 대화의 균형을 잡아 타겟 언어에서 캡션을 생성합니다.

- **Performance Highlights**: 본 방법은 영어-힌디어 챌린지 세트에서 37.90 BLEU 점수를 획득했으며, 영어-하우사어의 챌린지와 평가 리더보드에서 각각 1위와 2위에 랭크되었습니다. 또한, 250개의 이미지 하위 집합에 대한 추가 실험을 실시하여 다양한 가중치( weighting schemes) 조정 방식에서 BLEU 점수와 의미론적 유사성 사이의 trade-off를 탐색했습니다.



### Scaling Laws of Decoder-Only Models on the Multilingual Machine Translation Task (https://arxiv.org/abs/2409.15051)
- **What's New**: 최근 연구들은 인코더-디코더(encoder-decoder) 모델의 지배를 받던 기계 번역 분야에서 디코더 전용(decoder-only) 모델의 가능성을 탐구하고 있습니다. 본 연구는 다양한 언어와 도메인에서의 번역 작업을 위해 70M에서 7B 파라미터까지의 디코더 전용 모델을 훈련시키고, 이러한 모델의 스케일링 법칙을 조사하였습니다.

- **Technical Details**: 연구진은 디코더 전용 모델이 대형 언어 모델(LLM)에서 발견된 스케일링 법칙과 유사한 법칙을 따르지만, 모델의 크기가 너무 크거나 다른 데이터 분포에 일반화하는 데에는 어려움이 있다는 것을 발견했습니다. 또한, 모델의 깊이와 너비를 확장하는 방법이 유사한 테스트 손실 개선을 가져오지만, 후자는 모델의 효율성에 더 좋은 영향을 미친다고 합니다.

- **Performance Highlights**: 이번 연구에서 디코더 전용 모델은 이전의 인코더-디코더 모델보다 더 효율적인 훈련을 가능하게 하며, 특히 대량의 데이터를 처리하는 데 유리하다는 점이 강조되었습니다. 또한, 훈련 샘플의 배치(pack) 문제를 해결하기 위한 방법도 제안했습니다.



### Generative LLM Powered Conversational AI Application for Personalized Risk Assessment: A Case Study in COVID-19 (https://arxiv.org/abs/2409.15027)
- **What's New**: 이 연구는 전통적인 머신러닝 방법을 필요로 하지 않는 새로운 질병 위험 평가 방법을 제시하며, Generative LLM을 활용한 스티밍 인간-AI 대화를 통해 전방위적인 진단 가능성을 강조합니다.

- **Technical Details**: 본 연구는 Llama2-7b 및 Flan-T5-xl과 같은 사전 학습된 Generative LLM을 사용하여 COVID-19 심각도 위험 평가 사례 연구를 통해, 적은 수의 자연어 예제를 활용하여 모델을 고도화했습니다. 이를 통해 Logistic Regression, XGBoost, Random Forest와 같은 전통적인 분류기와 비교했습니다.

- **Performance Highlights**: 잘 조정된 LLM들이 전통적인 분류 방법보다 더 높은 AUC 점수를 달성하여, 제한된 데이터에서 사용할 수 있는 가능성을 입증했습니다. 이를 통해 Generative LLM이 의료 분야에서 일반적인 적재와 반응을 공정하게 처리할 수 있음을 강조했습니다.



### Inference-Friendly Models With MixAttention (https://arxiv.org/abs/2409.15012)
- **What's New**: 본 논문에서는 KV 캐시의 크기를 줄이기 위해 MixAttention 아키텍처를 제안하며, 이 아키텍처는 최근의 토큰 집합만을 저장하는 슬라이딩 윈도우 어텐션(Sliding Window Attention)과 레이어 간 KV 캐시 공유(KV Cache Sharing)를 결합한 방법입니다. 이를 통해 메모리 사용량을 줄이고 추론 속도를 개선할 수 있다는 점에서 주목할 만합니다.

- **Technical Details**: MixAttention은 모델의 메모리 소비를 줄이고, 긴 입력에 대한 추론 성능을 향상시키기 위해 슬라이딩 윈도우 어텐션과 KV 캐시 공유를 결합한 방법론을 사용합니다. 여러 가지 아키텍처 변형을 통해 평가하였고, 다양한 조합이 모델의 성능에 미치는 영향을 조사하였습니다. 이 결과, 짧은 및 긴 컨텍스트 작업 모두에서 모델 품질을 유지하면서도 자원 효율성을 최적화하는 구성을 발견하였습니다.

- **Performance Highlights**: MixAttention 구조는 추론 속도를 높이고 메모리 사용량을 줄이는 데 성공적으로 기여했으며, 대부분의 평가 지표에서 표준 트랜스포머 모델과 유사한 성능을 보여주었습니다. 특히, 레이어 간 KV 캐시 공유와 슬라이딩 윈도우 레이어 추가가 추론 성능을 향상시키고 메모리 사용을 줄이는 데 효과적입니다.



### Enhancing Aspect-based Sentiment Analysis in Tourism Using Large Language Models and Positional Information (https://arxiv.org/abs/2409.14997)
Comments:
          19 pages, 17 figures

- **What's New**: 이 논문에서는 관광 분야에서의 Aspect-Based Sentiment Analysis(ABSA)를 위한 새로운 모델 ACOS_LLM을 제안합니다. 이 모델은 Aspect-Category-Opinion-Sentiment Quadruple Extraction(ACOSQE)을 목표로 하며, 기존 전통적인 파이프라인 모델의 문제점을 해결하고자 합니다.

- **Technical Details**: ACOS_LLM 모델은 보조 지식 생성을 위한 Adalora와 모델 압축을 위한 Sparsegpt를 사용하여 전체 모델의 효율성을 높입니다. 이후 Positional 정보와 시퀀스 모델링을 통해 ACOSQE 작업을 수행합니다. 이를 위해 Bi-directional Long Short-Term Memory(BiLSTM)와 Bidirectional Gated Recurrent Unit(BiGRU)를 활용하여 감정 표현의 맥락을 이해하고, 다양한 측면의 감정 방향성을 구분합니다.

- **Performance Highlights**: 실험 결과, ACOS_LLM은 관광 데이터셋에서 다른 모델들에 비해 F1 점수가 7.49% 향상되었고, Rest15와 Rest16 데이터셋에서 각각 0.05% 및 1.06% 향상된 성능을 보여주었습니다.



### Beyond Fine-tuning: Unleashing the Potential of Continuous Pretraining for Clinical LLMs (https://arxiv.org/abs/2409.14988)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)을 임상 적용에 맞게 조정하기 위한 네 가지 기법(continuous pretraining, instruct fine-tuning, NEFTune, prompt engineering)의 효과를 검토했습니다.

- **Technical Details**: Mistral 7B 및 Mixtral 8x7B 모델을 활용하며, 500억 토큰의 대규모 임상 사전 학습 데이터셋과 5억 토큰의 instruct fine-tuning 데이터셋을 사용하였습니다. continuous pretraining은 모델의 기초를 다지며 instruct fine-tuning과 NEFTune 방식이 성능 향상에 기여하였습니다.

- **Performance Highlights**: continuous pretraining은 250억 토큰 이상에서 미미한 향상을 보였으나, instruct fine-tuning을 위한 튼튼한 기초를 마련하였고, NEFTune은 모델의 생성 품질을 향상시키는데 추가적인 이점을 가져왔습니다.



### Evaluating Theory of (an uncertain) Mind: Predicting the Uncertain Beliefs of Others in Conversation Forecasting (https://arxiv.org/abs/2409.14986)
- **What's New**: 이 연구는 Theory of Mind의 개념을 확장하여 언어 모델이 대화 중 다른 사람의 신념에 대한 불확실성을 예측하는 새로운 과제를 제안합니다. 기존의 신념 예측 과제가 이분법적으로 신념을 취급하는 반면, 이 논문은 대화자들의 신념이 더 유동적일 수 있으며, 불확실성의 정도까지 평가할 수 있음을 강조합니다.

- **Technical Details**: 저자들은 대화 예측(conversation forecasting) 기법을 통해 언어 모델의 불확실성 예측 능력을 평가합니다. 특히, 대화자가 직접적으로 대화 결과에 대한 불확실성을 예측하도록 모델을 훈련시키고, 이를 바탕으로 세 개의 대화 코퍼스(사회적, 협상, 작업 지향적)와 여덟 개의 모델을 가지고 실험을 수행했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 언어 모델은 다른 사람의 불확실성 변동의 최대 7.5%를 설명할 수 있지만, 이 과제가 여전히 어렵고 이는 향후 연구의 여지를 남깁니다. 저자들은 특히 이 연구의 방법론적 기여 외에도 상황에 따른 대화자의 목표와 맥락이 언어 모델의 ToM(Theory of Mind) 능력에 미치는 영향을 조사했습니다.



### Bilingual Rhetorical Structure Parsing with Large Parallel Annotations (https://arxiv.org/abs/2409.14969)
- **What's New**: 이 논문에서는 러시아어에 대한 병렬 주석을 포함한 대규모 영어 GUM RST 코퍼스를 제시합니다. 이를 통해 교차언어 RST 모델의 개발 및 평가를 가능하게 합니다.

- **Technical Details**: 이 연구는 언어 독립적인 고차 구조를 구성하고, 하위 수준에서 언어별 뉘앙스를 통합하는 최상위에서 하위로의 파서(adaptation) 접근 방식을 사용합니다. 또한, 제한적인 제2언어 주석을 가진 경우에도 파서 전이(transfer)를 효율적으로 수행할 수 있는 양을 탐구합니다.

- **Performance Highlights**: 개발된 end-to-end RST 파서는 영어 RST-DT에서 53.0%의 end-to-end Full F1 score을, 러시아어 RRT에서 45.3%의 F1 score을 기록하는 등 두 언어 모두에서 최첨단 성능을 달성했습니다.



### Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make your LLMs use External Data More Wisely (https://arxiv.org/abs/2409.14924)
- **What's New**: 이 연구에서는 외부 데이터를 활용한 대형 언어 모델(LLM)의 다양한 활용사례와 그 과정에서의 도전 과제를 제시합니다. 특히 Retrieval-Augmented Generation (RAG) 기법을 활용하여 사용자 쿼리를 네 가지 레벨로 분류하여 각 레벨에 맞는 기술적 접근 방식을 정리합니다.

- **Technical Details**: 본 논문에서는 쿼리를 명시적 사실 쿼리(Explicit Fact Queries), 암시적 사실 쿼리(Implicit Fact Queries), 해석 가능한 근거 쿼리(Interpretable Rationale Queries), 숨겨진 근거 쿼리(Hidden Rationale Queries)로 나누어, 각 쿼리에 필요한 외부 데이터의 유형과 작업의 주요 초점을 정의합니다. 또한 LLM에 외부 데이터를 통합하는 세 가지 주요 형태인 컨텍스트(Context), 소규모 모델(Small Model), 파인 튜닝(Fine-tuning)의 장단점도 살펴봅니다.

- **Performance Highlights**: 데이터 보강 LLM 애플리케이션은 전문성과 적시성을 향상시키며, 도메인 전문가와의 정렬, 모델 환각 감소 및 제어 가능성 및 설명 가능성을 개선하는 장점을 제공합니다. 그러나 여전히 많은 개발자들이 이 기술을 활용하기 위해 상당한 인간 노력을 투입해야 한다는 과제가 남아 있습니다.



### With Ears to See and Eyes to Hear: Sound Symbolism Experiments with Multimodal Large Language Models (https://arxiv.org/abs/2409.14917)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)과 비전 언어 모델(VLMs)이 사운드 상징(sound symbolism)을 어떻게 인지하는지를 분석하며, 시각적 정보와 텍스트 정보만으로 소리 기반 현상을 이해할 수 있는 능력을 조사합니다.

- **Technical Details**: VLMs와 LLMs의 소리 상징성을 평가하기 위해 Kiki-Bouba 및 Mil-Mal 심볼리즘 과제를 포함한 다양한 실험을 수행하였으며, 인간의 언어 아이코닉성(judgement of linguistic iconicity) 판단과 LLM의 판단을 비교하는 심층 분석을 진행했습니다.

- **Performance Highlights**: VLMs는 인간 레이블과 다양한 수준의 일치를 나타내며, Magnitude Symbolism을 이해하는 것이 Shape Symbolism보다 더 용이한 것으로 나타났습니다. 모델의 크기에 따라 언어 아이코닉성 이해도가 크게 달라지는 양상을 관찰했습니다.



### Towards a Realistic Long-Term Benchmark for Open-Web Research Agents (https://arxiv.org/abs/2409.14913)
- **What's New**: 새로운 벤치마크가 LLM 에이전트를 경제 가치가 있는 화이트칼라 작업에 대해 평가하기 위해 제안되었습니다. 이 연구에서는 금융 및 컨설팅 분야에서의 종합적이고 "어수선한(messy)" 작업을 평가하며, 이는 고객의 실제 사례에서 도출된 것입니다.

- **Technical Details**: 총 여덟 개의 실제 작업이 평가되며, 다양한 LLM 아키텍처(GPT-4o, Claude-3.5 Sonnet, Llama 3.1 (405b), GPT-4o-mini)가 시험되었습니다. 작업 수행 실패는 일반적인 문제(예: 웹사이트 분석 능력 부족)가 아니라 추론 및 계획의 실패로 간주되었습니다. ReAct 아키텍처가 보조작업을 하위 에이전트에 위임할 수 있는 능력 덕분에 가장 좋은 성과를 보였습니다.

- **Performance Highlights**: Claude-3.5 Sonnet를 사용한 LLM 에이전트들이 GPT-4o를 사용하는 에이전트들보다 크게 우수한 성과를 보였습니다. Llama 3.1 (405b) 및 GPT-4o-mini 기반의 에이전트는 상대적으로 성과가 낮았습니다. 이 벤치마크는 LLM 기반 에이전트의 경제 가치가 있는 작업에서 성능을 더 정확히 추정할 수 있게 합니다.



### Knowledge Planning in Large Language Models for Domain-Aligned Counseling Summarization (https://arxiv.org/abs/2409.14907)
Comments:
          Full paper accepted at EMNLP 2024 (main)

- **What's New**: 본 연구는 대규모 언어 모델(Large Language Models, LLMs)의 상담 요약능력을 강화하는 새로운 계획 엔진(planning engine)을 도입했습니다. 이 엔진은 우선 대화 구조를 보존하고 도메인별 지식을 통합하는 과정을 두 가지 주요 단계로 나누어 실행합니다.

- **Technical Details**: 제안하는 시스템은 PIECE로 이름 붙여졌으며, Llama-2 기반에서 작동합니다. PIECE는 지식 필터링과 구조적 요소(scaffolding)를 활용하여 도메인 지식을 캡슐화하며, sheaf convolution learning을 통해 대화의 구조적 미세함을 개선합니다. 다양한 자동 요약 평가 메트릭(ROUGE, Bleurt)을 통해 14개의 기준 방법 대비 성능을 비교했습니다.

- **Performance Highlights**: PIECE는 ROUGE 및 Bleurt 점수에서 유의미한 향상을 보여주었으며, Llama-2는 +2.72%, Mistral은 +2.04%, Zephyr는 +1.59%의 향상을 기록했습니다. 전문가 평가를 통해 생성 품질이 효과적이며, 때로는 금본위(gold standard)를 초과하는 결과를 보였습니다.



### DSG-KD: Knowledge Distillation from Domain-Specific to General Language Models (https://arxiv.org/abs/2409.14904)
Comments:
          IEEE ACCESS 2024

- **What's New**: 본 연구는 어린이 응급 치료 센터에서 얻은 전자 의료 기록(EMR) 데이터 기준으로 비상/비비상 분류 작업을 수행하며, 기존의 도메인 특화 언어 모델들이 일반 언어 모델에 비해 성능이 부족함을 보이고 있습니다. 이를 해결하기 위해 도메인 지식 전이 방법론을 제안하였습니다.

- **Technical Details**: 언어 모델은 교사 모델(teacher model)과 학생 모델(student model)로 정의됩니다. 의료 도메인 데이터로 사전 훈련된 모델(예: KM-BERT)을 교사 모델로, 일반 언어 모델(예: Ko-BERT)을 학생 모델로 삼아 지식을 전이하는 과정에서, 학생 모델이 교사 모델의 숨겨진 상태(hidden states) 및 주의 행렬(attention matrices)을 학습하도록 훈련됩니다. 이 방법에서는 Knowledge Distillation (KD) 기술을 활용하여 도메인 특정 지식을 일반 모델에 주입합니다.

- **Performance Highlights**: 제안된 방법은 한국 PED EMR 데이터에서 비상 및 비비상 사례 분류에서 높은 성능을 보여, 기존 모델들을 능가하였습니다. 또한 이 방법론은 다양한 전문 및 기술 분야에서도 폭넓은 적용 가능성을 제시합니다.



### End-to-End Graph Flattening Method for Large Language Models (https://arxiv.org/abs/2409.14880)
Comments:
          2024 1st International Conference on Computational Linguistics and Natural Language Processing (CLNLP 2024)

- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 그래프 처리 성능을 향상시키기 위한 새로운 방법인 End-to-End DAG-Path prompting (EEDP)를 제안합니다. EEDP는 기존의 그래프 평탄화 방법의 한계를 극복하여 장거리 시나리오에서의 추론 성능을 개선합니다.

- **Technical Details**: 기존의 그래프 평탄화 방법은 텍스트 형식으로 변환되어 LLMs에 사용되며, 이로 인해 장거리 의존성을 처리하는 데 한계를 보입니다. EEDP는 주 요약 경로(main backbone paths)를 활용하여 텍스트 설명을 생성하는 방법으로, 그래프의 시작 및 끝 노드만을 고려하여 최적의 표현을 도출합니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 실험 결과 EEDP는 장거리 및 단거리 시나리오에서 모두 뛰어난 성능을 보이며, 특히 장거리 시나리오에서의 LLM 성능 향상에 기여함을 보여주었습니다.



### Privacy Policy Analysis through Prompt Engineering for LLMs (https://arxiv.org/abs/2409.14879)
- **What's New**: PAPEL(Privacy Policy Analysis through Prompt Engineering for LLMs) 프레임워크는 대형 언어 모델(LLMs)의 성능을 활용하여 복잡한 개인정보 보호 정책을 자동으로 분석하는 혁신적인 접근 방식을 제시합니다.

- **Technical Details**: PAPEL는 제로샷(zero-shot), 원샷(one-shot), 소수샷(few-shot) 학습 방법 및 chain-of-thought prompting을 통합하여 LLMs가 개인정보 보호 정책을 효율적으로 분석하고 요약하도록 안내합니다. 추가적인 모델 훈련이 필요하지 않으며, 미리 정의된 프롬프트와 프롬프트 템플릿을 사용합니다.

- **Performance Highlights**: 실험 결과, LLaMA와 Chat GPT 모델을 포함한 LLMs는 개인정보 보호 정책 주석(annotation) 작업에서 F1 점수가 0.8 이상을 기록하여, 기존 자동 분석 접근 방식과 유사한 통찰을 제공하면서도 훈련 노력을 최소화하고 새로운 분석 요구에 대한 적응성을 높였습니다.



### Orthogonal Finetuning for Direct Preference Optimization (https://arxiv.org/abs/2409.14836)
- **What's New**: 본 논문에서는 기존의 Direct Preference Optimization (DPO) 알고리즘에서 발생하는 오버피팅 문제를 해결하기 위해 회전된 선호 최적화(weight-Rotated Preference Optimization, RoPO) 방법을 제안합니다. 이 방법은 신경망의 가중치 매개변수를 회전 및 크기 스트레칭 업데이트하여 초기 지식을 보존합니다.

- **Technical Details**: RoPO는 DPO에서 발생하는 오버피팅 현상을 완화하기 위하여 하이퍼스피어(hypersphere) 내에서의 에너지 변동을 활용하여 신경망의 뉴런 간 각도를 유지합니다. 이를 통해 모델의 표현 능력을 유지하면서도 사람의 선호에 잘 맞는 결과를 생성하도록 합니다. 특히, RoPO 방법은 단 0.0086%의 가중치 매개변수로도 우수한 성능을 발휘합니다.

- **Performance Highlights**: RoPO는 MT-Bench에서 DPO보다 최대 10점, AlpacaEval 2에서 최대 2.8점을 초과하는 성능을 보이며, 생성의 다양성도 평균 6점 증가시키는 결과를 보여줍니다.



### ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback (https://arxiv.org/abs/2409.14826)
- **What's New**: 최근 도구 보강 대형 언어 모델(LLMs)의 발전이 주목받고 있으며, 이들은 다양한 외부 도구와 상호작용하여 최종 답변을 제공합니다. 본 논문에서는 실제 사용자 시나리오를 반영하기 위해 새로운 데이터 세트 MGToolBench를 구축하고, LLM의 작업 완수 및 지침 이행 능력을 향상시키기 위한 ToolPlanner라는 두 단계 강화 학습 프레임워크를 제안합니다.

- **Technical Details**: MGToolBench는 사용자 행동을 반영하기 위해 다단계 사용자 지침 메커니즘을 채택하여 구성되었습니다. ToolPlanner는 첫 번째 단계에서 감독 하에 세밀화(Supervised Fine-Tuning, SFT) 모델을 사용하여 각각의 지침에 대한 해결책 트리를 생성하고, 두 번째 단계에서는 작업 완료 및 지침 이행이라는 두 가지 메트릭으로 생성된 솔루션을 평가합니다. 또한 솔루션 경로 계획 메커니즘을 사용하여 ToolPlanner의 다단계 추론 과정을 안내합니다.

- **Performance Highlights**: 실험 결과 ToolPlanner는 SOTA 모델 대비 Match Rate 26.8%, Pass Rate 20.2%, Win Rate 5.6% 향상되었음을 보여줍니다. 사람 평가에 따르면, 다중 세분화 지침이 사용자 사용 습관과 더 잘 일치하는 것으로 확인되었습니다.



### Past Meets Present: Creating Historical Analogy with Large Language Models (https://arxiv.org/abs/2409.14820)
- **What's New**: 이 논문은 역사적 유추(historical analogy) 획득(task) 연구에 초점을 맞추며, 대규모 언어 모델(Large Language Models, LLMs)을 활용한 유사한 역사적 사건의 검색 및 생성을 탐색합니다.

- **Technical Details**: 이 연구에서는 LLM을 기반으로 하여 역사적 유추를 획득하기 위한 검색(retrieval) 및 생성(generation) 방법을 제안하며, 생성 과정에서의 환각(hallucinations)과 고정관념(stereotypes)을 완화하기 위한 자가 반성(self-reflection) 방법도 제안합니다.

- **Performance Highlights**: 인간 평가 및 다차원 자동 평가를 통해 LLMs가 역사적 유추를 획득하는 데 일반적으로 좋은 잠재력을 가지고 있으며, 자가 반성 방법을 사용함으로써 모델의 성능을 더욱 개선할 수 있음을 보여줍니다.



### MobileVLM: A Vision-Language Model for Better Intra- and Inter-UI Understanding (https://arxiv.org/abs/2409.14818)
- **What's New**: 최근 VLM(비전-언어 모델)을 기반으로 한 모바일 AI 에이전트가 주목받고 있으며, 이를 위해 새로운 MobileVLM을 제안했습니다. MobileVLM은 모바일 도메인에 특화된 추가 전처리 단계를 포함하여 UI 이해력 향상을 목표로 합니다.

- **Technical Details**: MobileVLM은 내부 UI 이해와 외부 UI 이해를 위한 두 가지 추가 전처리 단계를 도입하였으며, 4개의 UI 기반 사전 훈련 작업을 정의했습니다. 이를 통해 모델은 세밀한 요소 인식 및 페이지 전환 행동을 보다 잘 인지할 수 있습니다. Mobile3M이라는 대규모 데이터세트를 구축하여 이 훈련을 지원하며, 3백만 개의 UI 페이지와 실제 사용자 행동 데이터로 구성된 방향 그래프 구조를 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, MobileVLM은 ScreenQA 및 다른 평가 데이터셋에서 기존의 SOTA VLM보다 각각 14.34%, 34.18% 향상된 성능을 보여주었습니다.



### MTP: A Dataset for Multi-Modal Turning Points in Casual Conversations (https://arxiv.org/abs/2409.14801)
Comments:
          Accepted by ACL 2024 main conference

- **What's New**: 본 연구는 대화 중 감정적 폭발이나 결정의 변화를 포함한 중요한 순간을 탐지하는 새로운 문제 설정을 제안합니다. 이 과정에서 고품질의 인간 주석이 포함된 멀티모달 데이터셋을 구축하였고, 각 전환점(turning points, TPs)의 정확한 타임스탬프, 설명 및 시각-텍스트 증거를 제시합니다.

- **Technical Details**: TPMaven이라는 프레임워크를 제안하며, 이는 최첨단 비전-언어 모델(vision-language models, VLMs)과 대형 언어 모델(large language models, LLMs)을 활용하여 비디오에서 내러티브를 구성하고 멀티모달 데이터셋에서 전환점을 분류 및 감지하기 위한 작업을 수행합니다. 이 연구는 Multi-modal Turning Point Classification (MTPC), Multi-modal Turning Point Detection (MTPD), Multi-modal Turning Point Reasoning (MTPR)와 같은 세 가지 작업을 통해 대화에서의 주요 전환점을 탐지합니다.

- **Performance Highlights**: TPMaven은 분류에서 0.88의 F1 점수를 달성하고, 탐지에서 0.61의 점수를 기록했습니다. 이러한 성과는 인간이 기대하는 설명과 잘 일치합니다.



### Towards Efficient and Robust VQA-NLE Data Generation with Large Vision-Language Models (https://arxiv.org/abs/2409.14785)
Comments:
          Preprint

- **What's New**: 본 연구는 대규모 비전-언어 모델(LVLMs)을 활용하여 효율적이고 고품질의 합성 VQA-NLE(비전 질문-응답 자연어 설명) 데이터셋을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: LVLMs의 생성 능력을 활용하여 복잡한 인간 주석 프로세스를 대체하고, 시각적 프롬프트(visual prompts)를 포함하여 데이터 생성의 정확성을 향상시켜 고품질의 설명을 생성하는 두 가지 접근 방식을 도입합니다. 그 과정에서 데이터 triplet (질문, 답변, 설명) 생성을 위한 다양한 프롬프트 파이프라인을 사용합니다.

- **Performance Highlights**: 합성된 VQA-NLE 데이터 생성 방식이 인간 주석에 비해 최대 20배 빠르며, 데이터 품질이 거의 동일하게 유지됨을 보여줍니다. 이 연구는 시각적 프롬프트를 포함하여 텍스트 생성의 관련성을 증가시켰습니다.



### Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method (https://arxiv.org/abs/2409.14781)
Comments:
          Accepted by EMNLP 2024 main

- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 훈련 데이터 검출 문제를 해결하기 위해 새로운 접근법인 DC-PDD(Divergence Calibration for Pretraining Data Detection)를 제안합니다. 기존의 Min-K% Prob 방식은 비훈련 예제를 정확히 구분하는 데 한계가 있었으나, DC-PDD는 토큰 확률 분포와 빈도 분포의 차이를 바탕으로 더 나은 성능을 보여줍니다.

- **Technical Details**: DC-PDD는 토큰 확률 분포와 토큰 빈도 분포 간의 교차 엔트로피를 계산하여 검출 점수를 산출합니다. 이 방법은 LLM이 블랙박스로 작동하여 내부 구조에 접근하지 않고도 텍스트의 출처를 확인할 수 있습니다. 또한, PatentMIA라는 새로운 벤치마크 데이터셋을 제안하여 중국어 텍스트에 대한 LLM의 검출 성능을 평가합니다.

- **Performance Highlights**: DC-PDD는 기존의 검출 방법에 비해 성능이 크게 향상되었습니다. 예를 들어, AUC와 TPR@5%FPR 지표에서 Min-K% Prob보다 각각 8.6% 및 13.3% 더 뛰어난 성능을 기록했습니다.



### OMPar: Automatic Parallelization with AI-Driven Source-to-Source Compilation (https://arxiv.org/abs/2409.14771)
- **What's New**: 이 논문에서는 OMPar라는 AI 기반 도구를 소개하여 C/C++ 코드의 병렬화를 자동화합니다. OMPar는 Loop 병렬화 가능성을 평가하는 OMPify와 OpenMP pragmas를 생성하는 MonoCoder-OMP라는 두 가지 주요 구성 요소를 통합합니다.

- **Technical Details**: OMPar는 대형 언어 모델(LLM)을 활용한 모듈형 접근 방식을 사용하여 루프 병렬화를 위한 OMPify와 코드 생성을 위한 MonoCoder-OMP를 결합합니다. 이 논문에서는 OMPar의 정확성을 HeCBench와 ParEval 벤치마크를 사용하여 평가하고, 전통적인 도구인 ICPC 및 AutoPar와의 성능을 비교합니다. OMPar는 루프 제안의 90% 이상이 컴파일 및 기능 테스트를 통과함을 보여줍니다.

- **Performance Highlights**: OMPar는 전통적인 자동 병렬화 방법보다 뛰어난 성능을 발휘하며, 부분 코드에서도 효과적으로 작동하여 유연성과 확장성을 강조합니다. 이 연구 결과는 LLM의 자동 병렬화 기술 혁신 가능성을 보여주며, 현대 소프트웨어 시스템에서의 병렬 처리의 효율성을 제고하는 데 기여할 것입니다.



### Language-Agnostic Analysis of Speech Depression Detection (https://arxiv.org/abs/2409.14769)
- **What's New**: 본 연구는 우울 장애(Major Depressive Disorder, MDD)를 가진 사람들의 음성에서 나타나는 멜로디 변화를 분석하여, 영어와 말라얄람어 두 가지 언어에서 음성 기반 우울증 감지 시스템을 개발했습니다. 음성이 감정 상태를 나타내는 중요한 지표임을 보여주며, 언어 독립적인 음성 기반 우울증 탐지 시스템의 가능성을 제시합니다.

- **Technical Details**: Convolutional Neural Networks (CNNs)를 이용해 음성 데이터를 분석하였으며, IViE corpus에서 수집한 데이터를 바탕으로 영어와 말라얄람어에서 특징을 추출하였습니다. 132명의 참가자들이 참여하여, 각 참가자는 44개의 문장을 레코딩했습니다. 음성 데이터는 잡음 처리와 피치 변화 등의 전처리 기법을 사용하여 준비되었습니다.

- **Performance Highlights**: 모델은 50 에폭에 걸쳐 64 샘플의 배치로 훈련되어, 테스트 데이터셋에서 76%의 정확도를 달성하였습니다. 이는 음성을 통해 감정 상태를 효과적으로 분류할 수 있는 능력을 보여줍니다.



### Do Large Language Models have Problem-Solving Capability under Incomplete Information Scenarios? (https://arxiv.org/abs/2409.14762)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 이 논문에서는 불완전한 정보 시나리오에서 대형 언어 모델(LLMs)의 문제 해결 능력을 평가하기 위해 BrainKing이라는 새로운 게임을 소개합니다. 이 게임은 'Who is undercover'와 'Twenty Questions'를 기반으로 하여 LLM이 제한된 예 또는 아니요 질문을 통해 목표 엔티티를 파악해야 합니다.

- **Technical Details**: BrainKing 게임은 세 가지 난이도(쉬움, 중간, 어려움)를 설정하여 LLM의 세계 지식, 역 발상 및 오류 탐지 능력을 평가합니다. 각 엔티티에 대해 최소 세 가지 개념을 포함하는 계층 개념 목록이 필요하며, LLM 참가자는 잘못된 정보 사이에서 정확한 답을 찾기 위해 최대 20개의 질문을 생성해야 합니다.

- **Performance Highlights**:  실험 결과, LLM은 BrainKing에서 불완전한 정보 시나리오에서의 정확도, 시작 난이도 및 잘못된 답변의 수가 LLM의 성능에 미치는 영향 등을 조사한 결과, LLM의 문제 해결 능력의 한계와 가능성을 확인할 수 있었습니다.



### LINKAGE: Listwise Ranking among Varied-Quality References for Non-Factoid QA Evaluation via LLMs (https://arxiv.org/abs/2409.14744)
Comments:
          Published as a conference paper at EMNLP Findings 2024

- **What's New**: 이 논문은 기존의 NFQA(비사실 질문 응답) 평가 방법의 한계를 극복하기 위해 새로운 listwise 평가 접근법인 LINKAGE를 제안합니다. 이 방법은 LLMs(대형 언어 모델)를 활용해 후보 답변을 다양한 품질의 참고 답변 목록에서 순위를 매기는 방식입니다.

- **Technical Details**: LINKAGE는 NFQA 평가에서 LLMs를 사용하여 품질이 내림차순으로 정렬된 참조 답변 목록에서 후보 답변의 순위를 매기는 접근법을 사용합니다. 또한, 다수의 금본(reference answer) 답변이 없을 때, LLMs의 컨텍스트 학습 능력을 활용하여 다양한 품질의 참조 답변 목록을 생성하여 listwise 평가를 지원합니다. 이 과정은 기존의 pointwise 및 pairwise 비교 방식보다 더 정확한 평가를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, LINKAGE는 세 가지 NFQA 데이터셋(ANTIQUE, TREC DL-NF, WebGLM)에서 자동 평가 점수 및 기존의 pointwise, pairwise 방법들보다 사람의 주석과의 상관관계가 유의미하게 높았습니다. 이는 NFQA 성능 개선을 위한 기반을 제공할 수 있습니다.



### ToxiCraft: A Novel Framework for Synthetic Generation of Harmful Information (https://arxiv.org/abs/2409.14740)
- **What's New**: 이번 논문에서는 Toxicraft라는 새로운 프레임워크를 제안하여 유해한 콘텐츠를 식별하는 데 있어 데이터 부족 문제와 일관되지 않은 정의의 문제를 해결하고자 합니다.

- **Technical Details**: Toxicraft는 적은 양의 seed data를 사용하여 다양한 합성(synthetic) 유해 정보 예시를 생성할 수 있는 프레임워크입니다. 이 프레임워크는 독특하면서도 매우 현실적인 예시를 만들어내며, 이를 통해 분류 모델을 강화합니다.

- **Performance Highlights**: 여러 데이터셋을 통해 이루어진 실험에서 탐지 모델의 강건성(robustness)과 적응성(adaptability)이 향상된 것을 보여주며, gold labels에 근접하거나 이를 초월하는 성능을 기록했습니다.



### ERABAL: Enhancing Role-Playing Agents through Boundary-Aware Learning (https://arxiv.org/abs/2409.14710)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2402.10618

- **What's New**: 새로운 연구에서 제안된 ERABAL 프레임워크는 역할을 맡은 언어 모델의 역할 연기 능력을 향상시키기 위한 경계 인식 학습(boundary-aware learning) 방법론을 소개합니다. 이는 역할 특화 대화 생성 파이프라인과 정렬 훈련 방법론을 포함합니다.

- **Technical Details**: ERABAL은 네 개의 모듈로 구성된 자동화된 데이터 생성 파이프라인을 도입하여 경계 인식 대화를 생성합니다. 이 모듈들은 대화 기획자(dialogue planner), 주제 관리자(topic manager), 경계 질의 생성기(boundary query generator), 응답 생성기(response generator)로 이루어져 있으며, 역할 속성에 기반한 특정 질문-응답 쌍을 생성합니다. 또, 경계 인식 선호 최적화(Boundary-aware preference optimization, BPO) 방법이 포함되어 있어 정교한 학습이 가능합니다.

- **Performance Highlights**: ERABAL은 일반적인 모델들과 비교하여 역할 일관성 평가에서 가장 우수한 성능을 보였습니다. 기존 역할 연기 벤치마크에 대한 실험을 통해, ERABAL은 10%의 훈련 대화만으로도 일반적 기준선 모델들에 비해 상당한 성능 향상을 이루어냈으며, WikiRoleEval, CharacterEval, MT-Bench 등에서 뛰어난 결과를 기록하였습니다.



### Target-Aware Language Modeling via Granular Data Sampling (https://arxiv.org/abs/2409.14705)
Comments:
          Accepted to EMNLP 2024 Main Conference, 9 pages, 6 figures, 3 tables

- **What's New**: 이 논문은 특정 도메인에서의 성능을 유지하면서도 데이터 샘플링의 최적화를 통해 언어 모델(ML) 사전 훈련의 효율성을 높이기 위한 새로운 접근 방식을 제안합니다. 특히, n-gram 기법을 활용하여 다중 그레인(멀티-그레인) 토큰으로 특징을 구성하는 중요 샘플링을 다시 다루고 있습니다.

- **Technical Details**: 저자들은 n-gram 토큰을 활용한 중요 샘플링 기법을 통해 데이터 선택성을 극대화하며, 이는 복잡한 모델이 아닌, 사전 훈련된 간단한 코어셋을 보여줍니다. 또한 다중 그레인 특성을 사용하는 새로운 알고리즘이 제안되었으며, 이를 통해 작업 지향적인 데이터 샘플링이 이루어지고 있습니다.

- **Performance Highlights**: 총 1%의 데이터로 학습된 모델이 전체 RefinedWeb 데이터에 맞먹는 성능을 발휘하는 결과를 보여주며, 랜덤 샘플링보다 뛰어난 성능을 나타냅니다. 다양한 모델 크기(125M부터 1.5B까지)에서 이과 같은 우수한 성능이 확인되었습니다.



### Instruction Tuning Vs. In-Context Learning: Revisiting Large Language Models in Few-Shot Computational Social Scienc (https://arxiv.org/abs/2409.14673)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 Instruction Tuning (IT)과 In-Context Learning (ICL)의 분류 성능을 비교하며, 실제적인 계산 사회과학(CSS) 작업에서의 중요성을 강조합니다. ICL이 대부분의 CSS 작업에서 IT보다 효과적이라는 실험 결과를 도출했습니다.

- **Technical Details**: 연구에서는 ICL과 IT를 사용하여 1-, 8-, 16-, 32-shot 설정에서 6개의 오픈 소스 LLM을 평가합니다. IT는 지도 방식으로 LLM의 파라미터를 업데이트하고, ICL은 특정 작업의 프롬프트(conditioning)를 이용하여 모델 가중치 업데이트 없이 작업을 수행하도록 합니다. 결과적으로 ICL이 IT보다 우수한 성능을 발휘할 뿐만 아니라, 샘플 수의 증가가 성능에 미치는 영향을 조사합니다.

- **Performance Highlights**: ICL은 5개의 CSS 작업에서 IT보다 평균적으로 더 높은 성능을 기록했습니다. 샘플 수를 단순히 증가시키는 것은 ICL이나 IT의 성능을 일관되게 향상시키지 못하며, 때때로 성능 저하를 초래할 수 있습니다. 또한, ICL 프롬프트가 제로샷(zero-shot) 및 Chain-of-Thought (CoT) 프롬프트보다 더 효과적임을 확인했습니다.



### Direct Judgement Preference Optimization (https://arxiv.org/abs/2409.14664)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 언어 모델을 평가하기 위한 새로운 접근 방식인 preference optimization을 통해 긍정적 및 부정적 데이터를 학습하는 방법을 제시합니다.

- **Technical Details**: 우리는 세 가지 접근 방식을 사용하여 다양한 용도에 맞는 preference pairs를 수집하고, 이러한 방식을 통해 생성적 평가 모델(generative judge)의 평가 능력을 향상시킵니다. 본 연구는 다양한 벤치마크에서 종합적인 연구를 통해 방법의 효과성을 입증합니다.

- **Performance Highlights**: 우리의 생성적 평가 모델은 13개 벤치마크 중 10개에서 최고 성능을 달성하였으며, GPT-4o 및 특화된 평가 모델과 같은 강력한 기준을 초월한 결과를 보여줍니다. 추가 분석 결과, 우리 모델은 내재된 편향(예: 위치 편향, 길이 편향)을 잘 극복하고, 평가 프로토콜에 유연하게 적응하며, 다운스트림 생성 모델의 개선을 위한 유용한 언어 피드백을 제공합니다.



### Building Tamil Treebanks (https://arxiv.org/abs/2409.14657)
Comments:
          10 pages

- **What's New**: 이번 논문에서는 타밀어 트리뱅크(Tamil treebanks)의 생성 방법을 세 가지 접근 방식을 통해 논의하고 있습니다: 수동 주석(manual annotation), 계산 문법(computational grammars), 그리고 기계 학습(machine learning) 기법입니다.

- **Technical Details**: 수동 주석은 높은 품질과 풍부한 구문 및 의미 정보를 보장하지만 시간이 많이 소요되고 언어학적 전문 지식이 필요합니다. Lexical Functional Grammar (LFG)와 같은 계산 심층 문법은 깊이 있는 언어 분석을 제공하지만 공식에 대한 상당한 지식이 요구됩니다. 기계 학습 접근 방식은 Stanza, UDpipe, UUParser와 같은 도구를 활용하여 대규모 데이터셋의 자동 주석을 가능하게 하지만, 품질이 높은 주석 데이터와 교차 언어 훈련 자료, 컴퓨팅 파워의 가용성에 의존합니다.

- **Performance Highlights**: 논문에서는 인터넷 데이터를 사용하는 것, 종합적인 언어 분석의 필요성, 숙련된 주석가를 찾는 어려움 등 타밀어 트리뱅크 구축 중 경험한 도전 과제를 다루고 있습니다. 이러한 도전에도 불구하고 타밀어 트리뱅크의 개발은 언어학 연구 진전 및 타밀어를 위한 NLP 도구 개선에 필수적입니다.



### Harmonising the Clinical Melody: Tuning Large Language Models for Hospital Course Summarisation in Clinical Coding (https://arxiv.org/abs/2409.14638)
Comments:
          20 pages, 4 figures

- **What's New**: 이번 연구에서는 병원 경과 요약(hospital course summarisation) 작업을 위해 Llama 3, BioMistral, Mistral Instruct v0.1의 세 가지 사전 훈련된 LLMs(large language models)를 조정했습니다.

- **Technical Details**: Quantized Low Rank Adaptation fine tuning 기법을 사용하여 MIMIC III 데이터셋에서 다양한 임상 노트를 결합하여 입력 임상 텍스트 형태의 자유 텍스트 임상 데이터셋을 생성했습니다. 모델 훈련을 위해 퇴원 요약(discharge summaries)에서 추출한 실제 Brief Hospital Course 섹션과 짝지었습니다. 모델의 효과성을 평가하기 위해 BERTScore와 ROUGE 메트릭스를 사용했습니다.

- **Performance Highlights**: 임상 도메인에 맞게 fine tuning된 LLM들이 병원 경과 요약 작업에서 성능을 크게 향상시켰으며, 임상 코딩을 위한 보조 도구로서의 잠재력을 시사했습니다. 향후 연구는 병원 경과 요약 작업에 적합한 데이터 커레이션(data curation) 방법을 개선하고, 독점 모델에 필적하는 더 발전된 오픈소스 LLM을 조정하는 데 초점을 맞춰야 합니다.



### Can a Neural Model Guide Fieldwork? A Case Study on Morphological Inflection (https://arxiv.org/abs/2409.14628)
- **What's New**: 이 논문에서는 언어학자와 화자 간의 상호작용을 고려한 새로운 모델을 제시하여 언어 데이터 수집 (data collection) 과정의 효율성을 높이고자 하였습니다. 특히, 기존 접근 방식과의 차별점을 두어, 상호작용의 두 가지 '원자적' 사례, 즉 언어학자가 화자를 만족시키는 정확한 추측을 하거나 더 많은 정보를 요청하는 경우를 명확히 구분하여 제안하였습니다.

- **Technical Details**: 제안된 모델은 필드워크 (fieldwork)에서 수집된 이전 데이터를 바탕으로 불완전한 데이터의 잠재적 격차를 식별하고, 다음 반복에서 수집해야 하는 정보의 우선순위를 설정합니다. 또한, 다양한 샘플링 전략의 효율성을 평가하고, 최신 신경 모델의 형태소 구조 일반화 (generalisation) 능력을 test합니다. 특히, morphology 데이터를 수집할 때의 대칭성을 활용하여 active learning (AL) 이론을 적용했습니다.

- **Performance Highlights**: 실험 결과, 주목할 만한 두 가지 전략이 언어 데이터 수집의 효율성을 개선하는 데 기여했습니다: (1) 패러다임 테이블의 셀을 균일하게 샘플링하여 주석 데이터의 다양성을 증가시킬 것과 (2) 모델 신뢰도를 사용하여 긍정적인 상호작용을 강화하고 데이터의 신뢰성 있는 예측을 제공하는 것입니다.



### Can pre-trained language models generate titles for research papers? (https://arxiv.org/abs/2409.14602)
- **What's New**: 이 연구에서는 연구 논문의 초록으로부터 제목을 자동 생성하기 위해 사전 훈련된 대형 언어 모델을 미세 조정하는 방법을 제안합니다. 특히, T5-base, BART-base, PEGASUS-large 모델을 서브셋 LREC-COLING-2024 데이터셋을 이용하여 훈련시켰습니다. 또한, ChatGPT-3.5를 제로샷(zero-shot) 설정에서 사용하여 초기 제목을 생성해보았습니다.

- **Technical Details**: 이 연구에서 활용된 핵심 기술은 딥 뉴럴 모델(deep neural models)로, 여러 사전 훈련된 트랜스포머 모델을 미세 조정하여 논문 제목을 생성하는 데 중점을 두었습니다. 이는 'abstractive text summarization'의 특별한 경우로 볼 수 있으며, 주요 평가 지표로는 ROUGE, METEOR, MoverScore, BERTScore 및 SciBERTScore가 사용되었습니다. 연구팀은 LREC-COLING-2024라는 새로운 데이터셋을 수집하였고, 이 데이터셋은 논문의 초록과 제목의 쌍을 포함하고 있습니다.

- **Performance Highlights**: PEGASUS-large 모델이 선택된 메트릭에서 다른 모델들보다 뛰어난 성능을 보여주었습니다. 특히, PEGASUS-large는 LLM(GPT-3.5, LLaMA)들에 비해 파라미터 수가 적음에도 불구하고 우수한 성능을 발휘했습니다. 연구에서는 또한 사용자에게 다양한 언어 모델을 선택할 수 있는 데모를 제공하였으며, Hugging Face에 미세 조정된 모델과 LREC-COLING-2024 데이터셋을 공개했습니다.



### EchoAtt: Attend, Copy, then Adjust for More Efficient Large Language Models (https://arxiv.org/abs/2409.14595)
- **What's New**: 본 논문에서는 내/외부 레이어 간의 유사한 attention 패턴을 분석하고 이를 활용하여 transformer 기반 모델의 효율성을 극대화하는 EchoAtt라는 새로운 프레임워크를 소개합니다. 이 방법은 LLMs에서 내 연결 레이어 간의 유사성을 활용하여 주의 매트릭스를 공유함으로써 계산 비용을 크게 줄이는 것을 목표로 합니다.

- **Technical Details**: EchoAtt는 지식 증류(knowledge distillation) 설정 내에서 작동하며, 사전 훈련된 teacher 모델이 계산 자원을 효율적으로 사용하는 동시에 모델 성능을 유지하도록 하는 student 모델의 훈련을 유도합니다. 특정 레이어 간 유사한 attention 매트릭스를 선택적으로 공유하여 모델의 복잡성과 필요 자원을 줄입니다.

- **Performance Highlights**: TinyLLaMA-1.1B 모델을 사용한 결과, EchoAtt는 추론 속도를 15% 증가시키고, 훈련 속도를 25% 향상시킴과 동시에 약 4%의 파라미터 수를 줄이는 성과를 나타냈습니다. 이러한 Compression이 이루어짐에도 불구하고 zero-shot 성능은 유지되었습니다.



### The X Types -- Mapping the Semantics of the Twitter Spher (https://arxiv.org/abs/2409.14584)
Comments:
          23 pages

- **What's New**: 본 연구는 약 200,000개의 인기 Twitter 계정을 포함하는 소셜 KB(Social Knowledge Base)를 구축하여, 그 계정에 대한 의미적 정보를 추출하고자 하였습니다. 특히, 이를 136개의 세부 의미 유형으로 구분하여, 각 계정이 정치인, 음악 아티스트 등 어떤 성격을 가지는지를 판별할 수 있는 모델을 개발하였습니다.

- **Technical Details**: 이 연구에서 우리는 Twitter의 대중 계정에 대한 의미적 유형 추정을 위해 DBpedia 및 Wikidata와의 매핑을 통해 레이블이 붙은 데이터를 생성하였습니다. 이 데이터를 이용하여 transformer 기반의 BERT 모델을 미세 조정(finetune)하고, 이를 활용하여 계정의 내용 기반으로 의미적 임베딩(embedding)을 생성하였습니다.

- **Performance Highlights**: 우리의 실험 결과, 레이블이 붙은 데이터셋에서 높은 예측 성능을 보였으며, 이 모델을 통해 Twitter의 모든 엔티티 계정에 대해 의미적 유형 예측을 수행했습니다. 이 정보는 사회적 엔티티의 유사성을 평가하는 주요 작업에서도 성능 향상을 가져오는 것으로 나타났습니다.



### Medical Concept Normalization in a Low-Resource Setting (https://arxiv.org/abs/2409.14579)
Comments:
          Master Thesis

- **What's New**: 이 논문에서는 생물의학 자연어 처리 분야에서의 의료 개념 정규화(Medical Concept Normalization)의 도전 과제를 다루고 있으며, 특히 독일어 비전문 텍스트에서의 한계에 대해 탐구합니다.

- **Technical Details**: 논문에서는 Unified Medical Language System으로 개념이 주석 처리된 독일 의료 온라인 포럼의 포스트로 구성된 데이터셋을 사용하여 실험을 진행했습니다. 다국어 Transformer 기반 모델들이 문자열 유사성 방법보다 우수한 성능을 보이는 결과를 나타냈습니다. 또한, 비전문 언급(normalization of lay mentions)을 개선하기 위한 맥락 정보의 활용도 실험했지만, 기대 이하의 결과를 초래했습니다.

- **Performance Highlights**: 최고 성능 모델의 결과를 기반으로 한 체계적인 오류 분석을 제시하고 있으며, 빈번한 오류를 완화하기 위한 잠재적인 개선 방안을 논의합니다.



### Evaluating the Performance and Robustness of LLMs in Materials Science Q&A and Property Predictions (https://arxiv.org/abs/2409.14572)
- **What's New**: 본 연구는 소재 과학 분야에서의 Large Language Models (LLMs)의 견고성과 신뢰성에 대한 포괄적인 평가와 분석을 수행합니다. 학부 소재 과학 강의에서의 객관식 질문 세트, 다양한 강재 조성과 항복 강도 데이터셋, 그리고 밴드 갭 값이 포함된 데이터셋 등 세 가지 독특한 데이터셋을 사용하여 도메인 특화된 Q&A와 소재 속성 예측을 분석합니다.

- **Technical Details**: 연구에서는 zero-shot chain-of-thought, expert prompting, few-shot in-context learning (ICL) 등의 다양한 프롬프트 전략을 사용하여 LLMs의 성능을 평가하였습니다. 또한, 텍스트 순서 변경과 같은 텍스트 섭동이 LLM의 추론에 미치는 영향을 조사하며, 현실적인 섭동부터 적대적 섭동까지 다양한 유형의 섭동을 테스트하였습니다. 밴드 갭 예측에 대한 세부 조사를 통해 일부 상황에서는 섭동이 모델의 예측 능력을 향상시키는 경향이 나타났습니다.

- **Performance Highlights**: LLMs의 성능 평가는 MSE-MCQs 데이터셋에서 수행되었고, gpt-4-0613이 모든 카테고리에서 가장 높은 점수를 기록했습니다. 또한, 전통적인 랜덤 포레스트 회귀 모델과 비교하여, gpt-3.5-turbo-0613이 few-shot ICL을 활용하여 강재 항복 강도 예측에서 유사한 성능을 보여주었습니다. 이 연구는 소재 과학 분야에서 LLMs의 신뢰성 있는 사용에 대한 정보에 기반한 회의적인 시각을 제시하고, 이를 통한 견고성과 신뢰성 향상을 위한 발전을 촉진하고자 합니다.



### Unleashing the Power of Emojis in Texts via Self-supervised Graph Pre-Training (https://arxiv.org/abs/2409.14552)
Comments:
          Accepted by EMNLP 2024 Main Conference

- **What's New**: 이 논문에서는 이모티콘과 텍스트 간의 관계를 개선하기 위해 포스트, 단어, 이모티콘으로 구성된 이종 그래프(heterogeneous graph)를 구축했습니다. 또한, 텍스트와 이모티콘의 공동 모델링을 위한 그래프 프리트레인 프레임워크(graph pre-train framework)를 제안하며, 이를 통해 텍스트와 이모티콘 사이의 상호작용을 더 잘 이해할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 자가 지도 그래프 사전 훈련 작업(self-supervised graph pre-training tasks)을 포함합니다: 1) 노드 수준 그래프 대비 학습(node-level graph contrastive learning), 2) 엣지 수준 링크 재구성 학습(edge-level link reconstruction learning). 이 방식을 통해 포스트, 이모티콘, 단어 간의 상호작용을 모델링하고, 이를 다양한 다운스트림 작업에서 활용 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Xiaohongshu 및 Twitter 데이터셋에서 다양한 다운스트림 작업(예: 인기 예측 및 감정 예측)에 대해 이전의 강력한 기준 모델보다 2%에서 10% 더 나은 성능을 보였습니다. 이 모델은 추가적으로 이모티콘 생성 작업에도 활용할 수 있습니다.



### Can AI writing be salvaged? Mitigating Idiosyncrasies and Improving Human-AI Alignment in the Writing Process through Edits (https://arxiv.org/abs/2409.14509)
Comments:
          NLP+HCI, Behavioral Science

- **What's New**: 이번 연구는 LLM(대형 언어 모델) 기반의 텍스트 생성이 인간의 글쓰기와 어떻게 다른지를 탐구하며, 전문가들이 LLM 생성 텍스트의 문제점을 수정하는 과정을 중심으로 진행되었습니다.

- **Technical Details**: 연구진은 LLM에 의해 생성된 텍스트의 단점을 일곱 가지 범주로 정리한 편집 분류법을 제안하였으며, 18명의 전문가 작가가 LLM 생성 문단을 편집하여 LAMP(Language model Authored, Manually Polished) 데이터셋을 구축했습니다. 이 데이터셋은 1,057개의 LLM 생성 문단으로 구성되어 있습니다.

- **Performance Highlights**:  LLM 모델(GPT4o, Claude-3.5-Sonnet, Llama-3.1-70b) 간의 글쓰기 질은 차이가 없음을 발견했으며, 전문가들에 의한 편집이 LLM 문서의 질을 개선하는 데 효과적임을 확인했습니다. 자동 편집 방법은 LLM 생성 텍스트와 인간 작성 텍스트 간의 정렬을 개선하는 데 유망한 성과를 보였습니다.



### A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders (https://arxiv.org/abs/2409.14507)
- **What's New**: 이번 연구에서는 Sparse Autoencoders (SAEs)를 활용하여 대형 언어 모델(LLMs)의 활성화를 인간이 해석할 수 있는 잠재 공간으로 분해하는 새로운 접근 방식에 대해 다룹니다. 특히, SAEs가 명확한 의미를 지닌 잠재 요소를 추출할 수 있는 정도와 sparsity 또는 SAE의 크기 변화가 명확한 의미성 및 해석 가능성에 미치는 영향을 탐구하였습니다.

- **Technical Details**: 연구는 첫 번째 문자 식별 작업을 통해 진행되었으며, 토큰에 대한 진리 레이블을 완전히 사용할 수 있는 환경에서 실험되었습니다. 이 과정에서 우리는 ‘feature absorption’이라 부르는 문제적 형태의 기능 분할을 확인하였고, 이는 잠재 요소가 인간이 해석할 수 있는 개념을 추적하는 것처럼 보이지만, 특정 토큰에서 예상대로 활성화되지 않는 현상입니다. 또한, SAE의 크기 및 sparsity를 변화시키는 것이 이 문제를 해결하기에 불충분하다는 점을 밝혔습니다.

- **Performance Highlights**: 실험 결과, 처음 문자 분류 작업에서 SAE latents의 정밀도 및 재현율이 선형 프로브보다 상당히 저조하다는 것을 발견하였습니다. 또한, 동일한 기능을 분류하는 것처럼 보이는 latents 간에 정밀도 및 재현율에 큰 차이가 존재하며, 이는 주로 sparsity와 SAE의 폭에 의해 매개된다는 것을 확인했습니다. 우리가 확인한 ‘feature absorption’ 현상은 SAE를 실제 애플리케이션에 활용하는 데 장애가 될 수 있으며, 이러한 잠재 요소들이 신뢰할 수 없는 분류기일 수 있음을 시사합니다.



### Thought-Path Contrastive Learning via Premise-Oriented Data Augmentation for Logical Reading Comprehension (https://arxiv.org/abs/2409.14495)
- **What's New**: 이번 논문에서는 Premise-Oriented Data Augmentation (PODA) 프레임워크를 제안하여 Chain-of-Thought (CoT) 합리화를 통해 올바른 답변 뿐만 아니라 잘못된 선택지에 대한 분석을 포함하고, 잘못된 후보 옵션으로부터 다양한 고품질의 반사실적 데이터(countersfactual context)를 자동으로 구축합니다.

- **Technical Details**: PODA는 올바른 및 잘못된 선택지에 대한 분석을 포함한 CoT 합리화를 생성하며, 각 선택지에 대한 요약 및 식별을 통해 반사실적 맥락을 구축합니다. Thought-Path Contrastive Learning (TPCL) 방법은 원본 및 반사실적 샘플 간의 사고 경로(thought-path)를 비교하여 모델의 논리적 추론 능력을 향상시키게 합니다. 구체적으로, 관계는 지지, 모순, 무관으로 분류되며, 이를 통해 다양한 반사실적 샘플을 생성합니다.

- **Performance Highlights**: 세 가지 대표적인 LLM(대형 언어 모델) 테스트에서 제안된 PODA와 TPCL 방법은 두 가지 논리적 MRC 벤치마크(ReClor 및 LogiQA 2.0)에서 기초 성능을 상당히 개선한 결과를 보여주었습니다.



### CPT-Boosted Wav2vec2.0: Towards Noise Robust Speech Recognition for Classroom Environments (https://arxiv.org/abs/2409.14494)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2405.13018

- **What's New**: 본 연구에서는 Wav2vec2.0을 교실 환경에 적합하도록 조정하기 위해 Continued Pretraining (CPT)의 효과를 분석하고, Wav2vec2.0 기반 모델의 Word Error Rate (WER)를 10% 이상 감소시킬 수 있음을 보여줍니다. 또한, CPT는 다양한 소음과 마이크, 그리고 교실 환경에 대한 모델의 강건성을 향상시키는 데 중요합니다.

- **Technical Details**: Wav2vec2.0은 Self-Supervised Learning (SSL) 기법을 사용하는 음성 표현 모델로, 비지도 학습된 데이터를 통해 언어 모델을 사전 학습한 후, 소규모 레이블 데이터로 파인튜닝(finetuning)합니다. CPT는 이미 사전 학습된 모델에 대해 추가적인 비지도 사전 학습을 수행하는 과정을 의미합니다. 본 연구는 CPT를 통해 교실의 noisy speech 인식에 적합하도록 Wav2vec2.0을 적응시키는 방법을 제안합니다.

- **Performance Highlights**: CPT를 통해 Wav2vec2.0은 다양한 소음 조건에 강건해지며, 기존의 최첨단(State Of The Art, SOTA) ASR 모델보다 소음에 강한 성능을 보여줍니다. 연구 결과, CPT를 사용한 모델은 다양한 마이크 구성과 인구통계적 요소에 대한 강건성을 향상시키는 것으로 나타났습니다.



### Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints (https://arxiv.org/abs/2409.14469)
Comments:
          Work in progress

- **What's New**: 이 논문은 LLMs (Large Language Models)에서의 시맨틱 파싱 (semantic parsing)의 효과를 조사하며, SENSE라는 새로운 프롬프트 접근 방식을 제안합니다. 기존의 방법과 달리, SENSE는 명시적인 파싱 결과를 주입하는 대신 시맨틱 힌트를 프롬프트에 포함시킵니다.

- **Technical Details**: SENSE는 LLMs가 내부의 시맨틱 파싱 능력을 활용하도록 유도하는 простратегия를 제공합니다. 연구에서는 GLUE (General Language Understanding Evaluation) 벤치마크의 여러 이해 과제와 기계 번역, 패러프레이징 및 단순화와 같은 생성 작업을 포함해 10가지 다양한 태스크에서 SENSE의 성능을 평가했습니다.

- **Performance Highlights**: SENSE는 GPT-4o-mini의 평균 성능을 79.43%에서 81.25%로 향상시켜 BERT의 83.2%에 매우 근접하게 개선했습니다. 또한 MRPC 및 MNLI와 같은 특정 태스크에서 유의미한 향상을 보여 주어, LLMs의 입력 문장 이해력을 강화하는 데 효과적임을 입증하였습니다.



### AggregHate: An Efficient Aggregative Approach for the Detection of Hatemongers on Social Platforms (https://arxiv.org/abs/2409.14464)
- **What's New**: 이 논문은 온라인 혐오 발언 감지의 새로운 접근 방식으로 사용자의 활동과 사용자 네트워크를 고려한 다중 모달(멀티모달) 집계 방식을 제안합니다. 이는 혐오 발언을 감지하기 위해 단순한 텍스트 기반 방법을 넘어서는 것입니다.

- **Technical Details**: 연구에서는 트위터(Twitter), 갭(Gab), 파를러(Parler)의 세 가지 고유 데이터 세트를 활용하여 사용자 레벨에서의 혐오 발언 탐지 기법을 평가하였으며, 다중 모달 집계 접근 방식과 관련된 세 가지 기본적 접근법을 탐구합니다: (i) 고정 임계값을 가진 이진 가중치 사용, (ii) 사회적 맥락에 조건화된 관계적 집계, (iii) 집계된 신뢰도 수준에 조건화된 분산 집계.

- **Performance Highlights**: 상대적으로 대규모 데이터 세트에서도 효율적이며, 사용자 레벨의 컨텍스트 정보를 종합적으로 고려함으로써 기존의 텍스트 및 그래프 기반 방법과 비교하여 혐오 발언 감지를 크게 개선할 수 있음을 보여줍니다.



### Exploring Multilingual Probing in Large Language Models: A Cross-Language Analysis (https://arxiv.org/abs/2409.14459)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 probing 기술을 다국어 환경으로 확장하여 다양한 언어에 대한 모델의 행동을 조사했습니다. 이는 대부분 영문 데이터에 초점을 맞추었으나 이제는 16개 언어에서 고급 언어와 저급 언어 간의 성능 차이를 분석합니다.

- **Technical Details**: 우리는 decoder-only LLMs를 사용하여 각 모델의 여러 레이어에서 hidden states를 추출하고 linear classifier probing을 통해 정보 인코딩 방식을 연구했습니다. 적용된 probing 기법은 다국어 컨텍스트에서 LLM의 사실적 지식 및 감성 분류 작업 수행 능력을 평가하는 데 중점을 두었습니다.

- **Performance Highlights**: 주요 발견으로는 고급 언어는 저급 언어에 비해 consistently higher probing accuracy를 나타내며, 고급 언어는 모델의 깊은 레이어에서 유의미한 정확도 향상을 보이는 반면, 저급 언어는 상대적으로 안정적인 성능을 유지했습니다. 또한, 고급 언어 간 probing vector의 유사성은 높은 반면, 저급 언어는 서로 및 고급 언어와의 유사성이 낮다는 점이 발견되었습니다.



### Automotive innovation landscaping using LLM (https://arxiv.org/abs/2409.14436)
Comments:
          9pages, 4Figures, 1 Flow chart

- **What's New**: 본 연구는 Prompt Engineering을 기반으로 한 특허 정보 추출 방법을 소개하며, 이 방법은 자동차 혁신의 경관을 조성하는 데 중요한 역할을 합니다. 기존의 수작업을 통한 방식에서 벗어나, 대형 언어 모델을 활용하여 보다 빠르고 효율적인 특허 분류 및 아이디어 추출을 가능하게 합니다.

- **Technical Details**: Prompt Engineering은 LLM(대형 언어 모델)과의 상호작용을 최적화하며, BERT와 같은 모델을 통해 여러 NLP(자연어 처리) 작업을 지원합니다. 이 연구에서는 OpenAI를 이용하여 TRIZ(Theory of Inventive Problem Solving) 모순을 추출하고, Transformer 기반 LLM을 활용해 특허에서 기술적 문제, 해결책, 이점 등을 식별하는 방법론을 다룹니다.

- **Performance Highlights**: 이 연구의 결과는 열린 특허 데이터셋을 사용하여 연료 전지 기술의 경관을 구성하는 방법을 보여줍니다. 이는 특허 문서의 복잡한 가독성 문제를 해결하고, 보다 빠르고 효율적인 정보 추출을 가능하게 하여 R&D 팀에 귀중한 통찰력을 제공합니다.



### Beyond Persuasion: Towards Conversational Recommender System with Credible Explanations (https://arxiv.org/abs/2409.14399)
Comments:
          Findings of EMNLP 2024

- **What's New**: 본 논문에서는 CRS(Conversational Recommender System)의 설명에서 신뢰성을 높이기 위한 새로운 접근법인 PC-CRS를 제시합니다. 이는 사용자의 수용력을 높이고 장기적인 신뢰를 구축하는 것을 목표로 합니다.

- **Technical Details**: PC-CRS는 전략 기반 설명 생성(Strategy-guided Explanation Generation)과 반복적인 설명 정제(Iterative Explanation Refinement)의 두 단계로 구성됩니다. 이는 Credibility-aware Persuasive Strategies를 활용하여 신뢰성 있는 정보를 포함한 설명을 생성하고, 이후 후보 설명을 수정하여 잘못된 정보를 제거합니다.

- **Performance Highlights**: 실험 결과에 따르면, PC-CRS는 기존 최적 기준선에 비해 평균 8.17%의 신뢰성 점수 향상과 5.07%의 설득력 점수 향상을 달성하였습니다. 신뢰성 있는 설명이 추천의 정확도를 개선하는데 기여한다는 추가 분석도 포함되어 있습니다.



### Predicting User Stances from Target-Agnostic Information using Large Language Models (https://arxiv.org/abs/2409.14395)
- **What's New**: 이 연구에서는 사용자의 대립 견해를 예측하는 방법으로 대규모 언어 모델(Large Language Models, LLMs)을 활용하는 가능성을 조사하고, 기존의 전통적인 기계 학습 모델보다 LLM이 더 효과적임을 보여주고 있습니다. 특히, LLM이 대상 관련이 없는(target-agnostic) 게시물로부터 안정적인 사용자 수준의 태도를 예측할 수 있다는 초기 증거를 제시했습니다.

- **Technical Details**: 연구는 1,000명의 Twitter 사용자로부터 수집된 Connected Behaviour (CB) 데이터셋을 사용하였으며, 이 데이터셋은 대상에 관계 없이 작성된 게시물과 특정 대상에 대한 게시물을 모두 포함합니다. 사용자 수준의 태도 예측(user-level stance prediction)이라는 새로운 작업을 정의하고, 사용자의 게시물이 특정 대상에 대한 언급이 없더라도 LLM을 통해 효과적으로 예측할 수 있다는 것을 입증하려고 하였습니다.

- **Performance Highlights**: LLM을 사용한 태도 예측은 초기에는 성능이 다소 낮을 수 있으나, 입력되는 대상 관련 없는 게시물이 증가함에 따라 성능이 급속도로 개선된다는 점이 관찰되었습니다. LLM은 대상 관련 데이터가 부족한 상황에서도 대중의 의견을 예측할 수 있는 유용한 방법임을 제시하며, 추가 연구를 통해 LLM의 성능과 효과를 향상시켜야 한다고 강조하고 있습니다.



### Investigating Layer Importance in Large Language Models (https://arxiv.org/abs/2409.14381)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델)의 개별 레이어의 중요성을 조사하여 이러한 모델의 이해를 심화시킵니다. 이를 위해 Shapley values라는 설명 기법을 활용하여 레이어의 중요성을 신뢰성 있게 평가하는 효율적인 샘플링 방법을 제안하였습니다.

- **Technical Details**: 연구에서는 Shapley value 프레임워크를 확장하여 LLM의 레이어에 대한 기여도를 정량화하였습니다. 또한, 레이어 절제(layer ablation) 실험을 통해 특정 레이어의 성능 저하를 평가하여, 초기 레이어 중 일부가 모델 성능에 미치는 중요한 역할을 탐구하였습니다.

- **Performance Highlights**: Cornerstone layer(주요 레이어)의 제거가 모델 성능의 급격한 하락으로 이어지는 반면, 비가건물 레이어의 제거는 성능에 미미한 변화를 주는 것으로 나타났습니다. 이 연구는 LLM에서 주요 레이어의 존재를 식별하고, 향후 연구에서 이들의 중요성을 강조합니다.



### J2N -- Nominal Adjective Identification and its Application (https://arxiv.org/abs/2409.14374)
Comments:
          7 pages, 5 figures

- **What's New**: 이 연구는 자연어 처리(NLP)에서 명사형 형용사(nominal adjectives, NAs)의 태깅 문제를 해결하기 위해 이들을 'JN'이라는 새로운 품사 태그로 분류하는 방안을 제안합니다. 이는 명사형 형용사를 명확히 식별하고 NLP 성능을 향상시킬 수 있는 가능성을 모색하는 것입니다.

- **Technical Details**: 연구에서는 Hidden Markov Models (HMMs), Maximum Entropy (MaxEnt) 모델 및 Spacy를 사용하여 명사형 형용사의 태깅 방법을 실험하였습니다. 명사형 형용사를 포함한 태깅의 직접적인 영향을 분석하고, BERT 모델을 통해 태그가 없는 텍스트에서도 명사형 형용사를 식별할 수 있도록 훈련하였습니다.

- **Performance Highlights**: 실험 결과, 명사형 형용사(JN)를 사용할 경우 전통적인 품사 태깅 모델에 비해 구문 분석(syntactic analysis) 및 구조적 이해(structural understanding)의 정확도가 개선됨을 보였습니다. 이는 NLP 시스템의 성능을 극대화하고 컴퓨터에서의 영어 문법 이해를 보다 세밀하게 할 수 있는 가능성을 제시합니다.



### The Ability of Large Language Models to Evaluate Constraint-satisfaction in Agent Responses to Open-ended Requests (https://arxiv.org/abs/2409.14371)
- **What's New**: 본 논문에서는 Generative AI 에이전트가 No One Right Answer (NORA) 요청에 대응할 수 있는 능력에 대해 다루고 있습니다. 특히, 입력된 요청의 제약 조건을 자동으로 평가할 수 있는 프레임워크의 필요성을 강조하며, 이를 위해 Arithmetic Constraint-Satisfaction (ACS) 데이터셋을 개발하고 공개합니다.

- **Technical Details**: ACS 데이터셋은 복잡한 사용자 요청과 해당 제약 조건, 에이전트의 응답, 제약 조건 만족도를 나타내는 인간의 레이블로 구성되어 있습니다. 이 데이터셋은 사용자가 요청한 제약 조건의 만족도를 평가하기 위해 응답 전체를 검토해야 하는 독특한 특성을 갖추고 있습니다. LLMs (Large Language Models)의 추론, 인-context 데이터 추출, 산술 계산 및 카운팅 능력을 평가하며, 자동 평가 프레임워크를 이용하여 제약 조건 만족도를 측정합니다.

- **Performance Highlights**: 연구 결과, 대부분의 LLM 모델이 여전히 개선 여지가 크며 주된 오류는 추론 문제에서 발생하는 것으로 나타났습니다. 또한, 적은 수의 프롬프트(few-shot prompting)를 활용할 경우 성능 저하가 발생하는 것이 관찰되었습니다.



### More Effective LLM Compressed Tokens with Uniformly Spread Position Identifiers and Compression Loss (https://arxiv.org/abs/2409.14364)
- **What's New**: 본 연구는 Transformer 입력을 압축된 토큰(compressed tokens)으로 변환하여 대규모 언어 모델(LLMs)의 속도와 비용 효율성을 개선하는 방법을 제시합니다. 특히, 기존의 ICAE 방법에 비해 약 15배에 달하는 압축 비율을 달성하며, 재구성 성능에서도 경쟁력을 유지합니다.

- **Technical Details**: ICA를 활용하여 압축된 토큰의 위치 식별자를 신중히 선택하고 새로운 압축 손실(compression loss)을 제안합니다. 본 연구에서 제시된 아키텍처는 기존의 AE 작업과 언어 모델링(LM) 작업을 결합하는 대신, 압축된 토큰에서 원본 문맥(segment)을 직접 디코딩하는 방법을 사용하여 효율성을 높입니다. 추가적으로, RoPE(position embedding) 기반의 위치 식별자를 고르게 분포시켜 압축 비율을 개선했습니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 15배의 압축 비율을 달성하며, 압축된 토큰을 통해 대부분의 정보를 유지할 수 있습니다. 이러한 접근 방식은 계산 시간과 비용을 줄이면서도, LLM의 성능을 유지하거나 향상시킬 수 있는 가능성을 보여줍니다.



### Using Natural Language Processing to find Indication for Burnout with Text Classification: From Online Data to Real-World Data (https://arxiv.org/abs/2409.14357)
- **What's New**: 이 논문은 독일어 텍스트에서의 번아웃 감지에 기여하며, 실시간 데이터를 활용한 연구 결과와 AI 모델의 해석 가능성에 대한 심층적인 통찰을 제공합니다.

- **Technical Details**: 번아웃(burnout)은 ICD-11에서 증상으로 분류되며, 만성 직장 스트레스에 기인합니다. 본 연구는 익명성을 유지한 실제 데이터셋을 수집하고, GermanBERT 기반 분류기의 한계를 보여주며, 실용적 응용에서 우수한 성능을 발휘하는 두 가지 BurnoutExpressions 데이터셋 버전을 제시합니다.

- **Performance Highlights**: 연구 결과는 AI 연구자와 임상 전문가 간의 협력이 번아웃 감지 모델 개선에 필수적임을 강조하며, 실제 환경에서 효과를 검증할 수 있는 더 많은 데이터의 필요성을 제기합니다.



### MQM-APE: Toward High-Quality Error Annotation Predictors with Automatic Post-Editing in LLM Translation Evaluators (https://arxiv.org/abs/2409.14335)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 활용하여 기계 번역(Machine Translation, MT)의 품질 평가를 위한 새로운 접근 방식인 MQM-APE를 제안합니다. 기존의 GEMBA-MQM 방법론이 제공하는 성능을 초월하여, 자동화된 오류 수정(Automatically Post-Editing, APE)을 통해 비영향적인 오류를 걸러내어 해석 가능성이 높은 오류 주석을 생성합니다.

- **Technical Details**: MQM-APE는 세 가지 주요 역할로 설정된 LLM을 통하여 작동합니다: 1) 오류 분석 평가자(evaluator)가 오류 주석을 제공하고, 2) 자동 포스트 에디터(post-editor)가 오류가 품질 개선에 영향을 미치는지를 판단하며, 3) 쌍별 품질 검증기(pairwise quality verifier)가 오류를 필터링합니다. 이 과정은 WMT22 테스트 세트를 포함하여 다양한 LLM을 사용하여 검증되었습니다.

- **Performance Highlights**: 실험 결과 MQM-APE는 GEMBA-MQM보다 더 높은 신뢰성과 품질을 보여주며, 인간의 주석과 잘 맞춰지는 해석 가능한 오류 범위를 제공합니다. 이 방법은 높은 자원과 낮은 자원 언어 모두에 효과적으로 적용되며, 번역별 평가자와의 상호 보완적인 관계를 형성합니다.



### Unveiling Narrative Reasoning Limits of Large Language Models with Trope in Movie Synopses (https://arxiv.org/abs/2409.14324)
Comments:
          EMNLP 2024 Findings. The first two authors contributed equally. Code: this https URL

- **What's New**: 이 연구는 캐릭터의 전형을 포함하는 영화의 줄거리를 사용하여 최신 대형 언어 모델(LLMs)의 추상적 추론 능력을 평가하였습니다. 특히, CoT(Chain-of-Thought) 프롬프트 방법을 사용할 때 내러티브 추론에서의 낮은 성능을 드러냈습니다. 이를 해결하기 위해 전형별 쿼리 방식을 도입하여 성능을 11.8포인트 향상시켰습니다.

- **Technical Details**: 이 연구는 영화 줄거리에 포함된 전형(trope)을 사용하는 최초의 LLM 분석입니다. CoT가 적용된 경우에도 GPT-4와 같은 모델이 전형 이해 작업에서 무작위 추측 수준의 성능을 보이는 반면, 전형별 쿼리 방식은 성능을 획기적으로 높였습니다. 또한, Adversarial Injection 기법을 통해 LLM이 전형 관련 텍스트 토큰에 대한 민감성을 가지게 될 수 있음을 발견했습니다.

- **Performance Highlights**: 이 연구에서 제시된 전형별 쿼리 방식은 기존 TiMoS(Trope in Movie Synopses) 데이터셋에서 성능을 11.8포인트 향상시켜 새로운 최첨단 성과를 초래하였습니다. CoT가 입력된 경우 LLM의 정확도가 현저히 감소하며, 적대적 입력(Adversarial Input)에 대한 높은 민감도를 보여줍니다.



### PretextTrans: Investigating Medical Factual Knowledge Mastery of LLMs with Predicate-text Dual Transformation (https://arxiv.org/abs/2409.14302)
Comments:
          17 pages, 10 figures

- **What's New**: 이번 연구에서는 현재 대규모 언어 모델(LLMs)의 의료 사실 지식 숙련도를 동적인 평가 스키마를 사용하여 조사하고, 각 의료 사실 지식 포인트에 대한 여러 테스트 샘플을 자동으로 생성하는 방법을 제안합니다. 이는 기존의 LLM이 사용했던 방식의 한계를 극복하기 위한 것입니다.

- **Technical Details**: 우리는 Predicate-text Dual Transformation (PretextTrans)라는 새로운 평가 방법을 제안합니다. 이 방법은 각 의료 지식 포인트를 술어 표현으로 변환하고, 이를 통해 생성된 변형을 바탕으로 다양한 텍스트 표현을 생성합니다. 이러한 방식은 사실적 신뢰성과 표현의 다양성을 동시에 보장합니다. 이 연구에서 12개의 잘 알려진 LLM의 의료 사실 지식 숙련도를 두 개의 의료 데이터 세트를 기반으로 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안된 PretextTrans 방법에 의해 생성된 다중 샘플 데이터 세트에서 LLM의 성능이 기존의 단일 샘플 데이터 세트에 비해 상당히 낮은 것으로 나타났습니다. 이는 현재 LLM들이 의료 사실 지식을 포괄적으로 습득하지 못하고 있음을 보여주며, 실제 의료 시나리오에서의 성능 부진의 원인을 설명합니다.



### ESPERANTO: Evaluating Synthesized Phrases to Enhance Robustness in AI Detection for Text Origination (https://arxiv.org/abs/2409.14285)
- **What's New**: 이 논문은 AI 생성 텍스트 탐지 시스템의 취약성과 현재 시스템의 내구성을 향상시킬 필요성을 강조합니다. 이를 해결하기 위해 새로운 기법인 back-translation을 소개하여 탐지를 우회하는 방법을 검토하면서, 이를 극복하기 위한 대응 방안도 제시합니다.

- **Technical Details**: 제안된 방법은 AI 생성 텍스트를 여러 언어로 번역한 후 영어로 다시 번역하는 과정으로 구성됩니다. 모델은 이러한 back-translated 텍스트를 결합하여 원본 AI 생성 텍스트의 변조 버전을 제작합니다. 이 기법은 720,000 개의 텍스트로 구성된 대규모 데이터셋에서의 실험을 통해 평가되었으며, 다양한 AI 탐지기를 대상으로 검증되었습니다.

- **Performance Highlights**: 변조된 텍스트는 원래 의미를 유지하면서, 기존 탐지 방법의 true positive rate (TPR)을 대폭 낮추는 효과를 보였습니다. 예를 들어, RADAR의 TPR은 질문-응답 데이터셋에서 52% 감소했습니다. 제안된 방법은 back-translation 변조에 노출되었을 때 TPR이 단지 1.85% 감소하는 것으로 나타났습니다.



### Instruction Following without Instruction Tuning (https://arxiv.org/abs/2409.14254)
- **What's New**: 이번 연구에서는 언어 모델을 명시적인 지침-응답 쌍으로 조정(finetuning)하는 기존의 instruction tuning보다 훨씬 간단한 implicit instruction tuning을 발견했다. 이 방식은 오직 응답만으로도 지침 따르기를 할 수 있으며, 교훈적으로 제공된 응답 분포 없이도 프로그래밍, 시 생성 등 다양한 작업을 수행할 수 있음을 보였다.

- **Technical Details**: Implicit instruction tuning은 두 가지 형식(응답 조정(response tuning) 및 단일 작업 조정(single-task finetuning))을 통해 이루어진다. 응답 조정은 오직 응답으로만 훈련되어도 지침 따르기가 가능하다는 것을 보여주었고, 단일 작업 조정은 좁은 도메인 데이터에서 훈련되더라도 넓은 범위의 지침 따르기 행동을 만들어낼 수 있다. 연구팀은 언어 모델의 분포에 대한 간단한 변경이 지침 따르기를 유도할 수 있다는 가설을 세웠고, 세 가지 규칙(시퀀스 종료 확률 증가, 15개 토큰의 확률 균일 수정, 토큰 반복에 대한 패널티)을 적용하여 이를 검증하였다.

- **Performance Highlights**: 응답 조정된 모델은 비슷한 instruction-tuned 모델에 비해 약 43%의 승률을 기록했고, 시 조정된 Llama-2-7B 모델은 AlpacaEval 2에서 instruction-tuned 모델에 비해 23.7%의 승률을 보였다. 이러한 결과는 일반적인 지침-응답 관계를가르치지 않고도 모델이 다양한 지침을 따를 수 있음을 나타낸다.



### Repairs in a Block World: A New Benchmark for Handling User Corrections with Multi-Modal Language Models (https://arxiv.org/abs/2409.14247)
Comments:
          Accepted to EMNLP'24 Main (Upcoming)

- **What's New**: 이 논문은 다층적인 지식 기반의 대화에서 발생하는 Third Position Repair (TPR) 시퀀스를 포함하는 BlockWorld-Repairs라는 새로운 데이터 세트를 수집하고 공개합니다. TPR은 대화 중 오해가 발생했을 때 이를 수정하는 일련의 과정을 설명하며, 대화형 AI 기술에서 중요한 역할을 담당합니다.

- **Technical Details**: TPR은 Addressee가 Speaker를 잘못 이해하고 잘못된 응답을 할 때 발생하는 일련의 반응입니다. 이 데이터 세트는 Vision and Language Models (VLM)의 성능을 평가하는 데 활용되며, 대화의 복잡성과 비대칭적인 참조를 포함합니다. 연구를 통해 VLM은 특정 토큰에 초점을 맞춘 손실 함수를 통해 미세 조정(fine-tuning) 시 보다 나은 성능을 발휘할 수 있음을 발견했습니다.

- **Performance Highlights**: 모든 모델이 인간과 비교할 때 TPR 처리가 미흡하여 성능 차이를 보였습니다. 이러한 결과는 특히 수리(Repair)가 잦은 다중 모달 협력 환경에서 이러한 모델들을 사용할 준비가 아직 안 되어 있음을 보여줍니다. 이에 따라 상호작용 학습을 위한 훈련 체계와 목표가 필요하다는 점을 강조합니다.



### Data-centric NLP Backdoor Defense from the Lens of Memorization (https://arxiv.org/abs/2409.14200)
- **What's New**: 이 논문에서는 DNN 기반 언어 모델의 백도어 공격에 대한 새로운 관점을 제시합니다. 언어 모델 메모리화의 정의를 샘플 단위에서 문장 요소 단위로 확장하고, 백도어는 요소 단위 메모리화의 일종이라는 점을 강조합니다.

- **Technical Details**: 기존의 연구에서는 메모리화를 샘플 단위로 분석했으나, 본 연구는 문장 내의 특정 요소(단어, 구, 구조 등)별 메모리화에 중점을 둡니다. 특히, 훈련 데이터에서 중복된 요소의 빈도가 백도어 공격의 성공률에 긍정적인 상관관계를 가진다는 것을 밝혔습니다. 새로운 데이터 중심 방어 방법인 BMC(Bad Memorization Cleanser)를 제안합니다.

- **Performance Highlights**: BMC는 다양한 NLP 백도어 공격에 대해 8.34배의 공격 성공률 감소를 달성하면서, 부정확도는 단 0.85% 증가시키는 성과를 보였습니다. 이는 기존 방어 방법보다 뛰어난 결과입니다.



### The Imperative of Conversation Analysis in the Era of LLMs: A Survey of Tasks, Techniques, and Trends (https://arxiv.org/abs/2409.14195)
Comments:
          21 pages, work in progress

- **What's New**: 이 논문은 대화 분석(Conversation Analysis, CA)의 체계적인 정의와 중요한 네 가지 절차를 정의하며, 특히 대규모 언어 모델(LLMs)을 활용하여 기존의 연구를 종합적으로 정리합니다.

- **Technical Details**: CA는 대화에서 중요한 정보(예: 참가자 프로필, 감정, 의도 등)를 식별하고, 이러한 요소의 배경을 분석하여 목표 달성을 위한 개선 방안을 제시하는 것을 목표로 합니다. CA 절차는 1) Scene Reconstruction, 2) Causality Analysis, 3) Skill Enhancement, 4) Conversation Generation의 네 가지 단계로 구성됩니다.

- **Performance Highlights**: 노력의 대부분이 여전히 표면적인 대화 요소 분석에 집중되고 있으며, 연구와 비즈니스 간의 큰 격차가 존재합니다. 그러나 LLMs의 도움으로 최근 연구는 원인 분석 및 전략적 작업과 같은 고급 주제로의 연구 경향을 보이고 있습니다.



### Knowledge in Triples for LLMs: Enhancing Table QA Accuracy with Semantic Extraction (https://arxiv.org/abs/2409.14192)
- **What's New**: 이 논문은 semi-structured (반구조화) 테이블에서 Triple을 추출하고 이를 Retrieval-augmented Generation (RAG) 모델과 결합하여 자연어 처리(NLP)에서 질문 응답 시스템(QA)의 정확도와 문맥적 풍부함을 향상시키는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법론은 RDFLib 라이브러리를 사용하여 테이블에서 Triple을 간단히 구성하는 과정을 포함합니다. Triple은 Subject(주어), Predicate(서술어), Object(목적어)로 구성되며, 이를 통해 테이블의 셀 간의 관계를 명확히 표현합니다. 이 Triple은 Fine-tuned GPT-3.5-turbo-0125 모델에 통합되어 응답 생성을 개선하는 데 사용됩니다.

- **Performance Highlights**: 제안된 접근 방식은 FeTaQA 데이터셋에서 기존 기법들보다 성능이 크게 향상되었으며, 특히 Sacre-BLEU 및 ROUGE 지표에서 우수한 성과를 나타냈습니다. 테이블에서 복잡한 정보를 효과적으로 식별하고 명확한 긴 형식의 답변을 생성하는 능력이 두드러집니다.



### QMOS: Enhancing LLMs for Telecommunication with Question Masked loss and Option Shuffling (https://arxiv.org/abs/2409.14175)
- **What's New**: 이 논문은 QMOS라는 혁신적인 접근 방식으로 통신 분야에서 다중 선택 질문에 대한 LLM(대규모 언어 모델)의 성능을 향상시키는 방법을 제시합니다. 기존의 수익 모델에 의존하지 않고 오픈소스의 작은 언어 모델인 Phi-2와 Falcon-7B를 사용하여 Retrieval Augmented Generation (RAG) 프레임워크 내에서 다양한 개선사항을 적용하여 성과를 올릴 수 있었습니다.

- **Technical Details**: QMOS는 질문 마스크 손실 함수(Question-Masked loss)와 옵션 셔플링(Option Shuffling) 기법을 활용하여 LLM을 다중 선택 질문에 효율적으로 적응시킵니다. 이 프로세스는 여러 임베딩 모델을 통해 관련 정보를 다각화하고, 약어 딕셔너리를 확장하며, 정교한 프롬프트 설계를 통해 LLM이 문서에서 답을 선택하도록 유도합니다. 또한 LoRA(저차원 적응, Low-Rank Adaptation) 기법을 통해 Phi-2 모델을 통신 도메인에 맞추어 효율적으로 조정합니다.

- **Performance Highlights**: Falcon-7B 모델은 기본선에서 24.70%에서 49.30%로 정확도를 개선했으며 Phi-2 모델은 42.07%에서 84.65% 사이의 성과를 달성했습니다. 이는 기존 모델에 비해 значительно 향상된 결과로, 효율적이고 비용 효과적인 QA 시스템 구축에 기여할 것으로 예상됩니다.



### Towards Building Efficient Sentence BERT Models using Layer Pruning (https://arxiv.org/abs/2409.14168)
- **What's New**: 이번 연구는 SBERT(문장 BERT) 모델의 효과적인 레이어 프루닝(layer pruning)에 대한 효과를 조사하며, 복잡성을 줄이면서도 강력한 임베딩 유사성을 유지할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 Muril 및 MahaBERT-v2 같은 BERT 모델을 프루닝 전후로 평가하고, MahaBERT-Small 및 MahaBERT-Smaller와 같은 작은 모델들과 비교합니다. NLI(자연어 추론) 및 STS(의미적 텍스트 유사성) 훈련을 포함한 2단계 SBERT 파인튜닝 과정에서 레이어 축소의 임팩트를 평가합니다.

- **Performance Highlights**: 프루닝된 모델이 레이어 수는 적지만 완전한 레이어 모델들과 경쟁력을 가지며, 유사한 크기의 스크래치 모델보다 일관되게 우수한 성능을 보입니다. 결과적으로 레이어 프루닝은 계산 수요를 줄이고 동시에 높은 품질의 임베딩을 유지하는 실용적인 접근법으로 자리잡았습니다.



### On Importance of Pruning and Distillation for Efficient Low Resource NLP (https://arxiv.org/abs/2409.14162)
- **What's New**: 본 연구는 저자들이 마라티어(Marathi)와 같은 자원이 부족한 언어에 대해 대형 트랜스포머 모델을 최적화하는 방법을 제안하는 최초의 연구 중 하나입니다. 마라티어에 대한 효율적인 언어 모델이 부족한 상황에서, 본 연구는 기존 모델의 최적화 기술을 통해 계산 시간과 메모리 사용량을 줄이는 것을 목표로 합니다.

- **Technical Details**: 연구에서는 marathi-topic-all-doc-v2 모델을 기반으로 여러 최적화 기술을 적용합니다. Block Movement Pruning, Knowledge Distillation, Mixed Precision 기법을 개별적으로 그리고 조합하여 적용하여 모델의 효율성을 높입니다. 6개의 실험을 통해 25%, 50%, 75% prunings 및 Knowledge Distillation을 포함한 조합을 테스트했습니다.

- **Performance Highlights**: 25% pruning과 Knowledge Distillation 조합을 통해 2.56배의 속도 향상을 달성하였으며, 기본 정확도를 유지했습니다. 모델의 파라미터 수는 223백만으로 유지되었고, 75% pruning과 Knowledge Distillation을 적용할 경우에는 195백만 파라미터를 갖고 2%의 정확도 감소가 있었습니다. 이러한 최적화는 환경적인 영향 감소에도 기여할 수 있습니다.



### Interpreting Arithmetic Mechanism in Large Language Models through Comparative Neuron Analysis (https://arxiv.org/abs/2409.14144)
Comments:
          Accepted by EMNLP 2024 main. Mechanistic interpretability for arithmetic tasks in large language models

- **What's New**: 이 논문에서는 산술 능력이 제한된 수의 attention heads에 존재하고, 각 head가 다른 작업에 특화되어 있다는 것을 발견하였습니다. Comparative Neuron Analysis (CNA) 방법을 도입하여 입력에서 예측까지의 내부 로직 체인을 네 가지 단계로 식별하였습니다.

- **Technical Details**: CNA 방법은 feature enhancing, feature transferring, feature predicting, prediction enhancing 의 네 가지 단계로 구성되어 있습니다. 이 연구에서 LoRA의 메커니즘을 분석하고, FFN (Feedforward Neural Network) neurons의 계수 점수를 증폭시킴으로써 최종 예측 확률을 높인다는 결론을 내렸습니다.

- **Performance Highlights**: 모델이 산술 작업을 위한 pruning 및 성별 편향을 감소시키기 위한 편집 방법을 설계하였으며, 특히 산술 능력에 영향을 미치는 산술 heads를 파악하는 데 중요한 기여를 하였습니다.



### Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm (https://arxiv.org/abs/2409.14119)
Comments:
          Under Review

- **What's New**: 본 연구에서는 Parameter-efficient fine-tuning (PEFT) 아키텍처에서 악의적인 백도어 공격에 대한 방어 방법인 Obliviate를 소개합니다. Obliviate는 PEFT 프로세스와 통합되어 작동하며, 이는 두 가지 주요 기술을 포함합니다: 1) PEFT 레이어 내에서 유해하지 않은 뉴런을 증폭하여 모델이 청정 학습 샘플에 더 집중하도록 유도합니다. 2) 트리거 토큰의 영향을 제한하기 위해 어텐션 점수를 정규화합니다.

- **Technical Details**: Obliviate는 PEFT 아키텍처, 특히 adapter, LoRA, prefix-tuning을 기반으로 하여 RoBERTa와 BERT 모델에 적용될 수 있습니다. PEFT에서 훈련 가능한 매개변수가 제한적이기 때문에 기존의 방어 방법을 적용하기 어려운 부분을 극복하여, 백도어 공격을 효과적으로 중화시키기 위한 두 개의 손실 항(term)을 추가합니다.

- **Performance Highlights**: 실험 결과 Obliviate는 최첨단의 task-agnostic 백도어 공격의 성공률(ASR)을 83.6% 감소시키는 효과를 보였으며, 청정 정확도(CACC)는 약간(0.78%) 하락하는 것으로 나타났습니다. 뿐만 아니라, Obliviate는 다양한 공격 전략에 대한 강력한 방어 능력을 보여줍니다.



### Routing in Sparsely-gated Language Models responds to Contex (https://arxiv.org/abs/2409.14107)
- **What's New**: 이 논문은 Mixture-of-Experts (MoE) 아키텍처에서의 토큰-전문가 할당의 문맥 민감도를 평가하고, 특정 문맥에서 단백의 할당을 조사하여 기존의 연구 결과를 발전시킵니다.

- **Technical Details**: 비지도 학습된 텍스트 쌍을 활용하여 다양한 모형 구성에 따른 라우팅의 특성을 분석하였고, 인코더 레이어에서는 (semantic) 연관성에 기반한 라우팅이 주요하나, 디코더 레이어에서는 문맥에 대해 변동성이 큼을 발견하였습니다. 또한, 전문가 수에 따라 문맥 민감도가 증가한다고 보고하였습니다.

- **Performance Highlights**: 모델의 인코더와 디코더 구성 모두에서 라우팅 결정이 문맥 정보에 반응하며, 유사한 문맥에서 단어가 동일한 전문가에 더 일관되게 배정되는 경향을 보였습니다.



### Probing Context Localization of Polysemous Words in Pre-trained Language Model Sub-Layers (https://arxiv.org/abs/2409.14097)
- **What's New**: 이 논문은 고성능의 대형 언어 모델(LLMs)에서의 문맥적 단어 표현의 중요성을 강조하며, 사전학습된 언어 모델(PLM)의 세부적인 서브-레이어 표현에서 문맥화의 강도를 실험적으로 조사합니다. 특히, BERT 모델의 Self-Attention, Feed-Forward Activation, Output 서브-레이어 간의 문맥화 정도를 비교 분석합니다.

- **Technical Details**: 이 연구에서는 선형 프로브(linear probe) 방법을 통해 다의어(polysemous word)의 의미를 식별하는 과제를 수행하며, 다른 문맥에서의 표현을 비교하여 PLM 서브-레이어에서의 문맥화 정도를 분석합니다. 다양한 유사성 지표를 사용해 BERT 서브-레이어의 문맥화 정도를 국소화하고, 서로 다른 단어 위치 및 문맥 길이가 문맥화 정보에 미치는 영향을 조사합니다.

- **Performance Highlights**: 주요 결과로 BERT는 특정 위치와 짧은 문맥에서 문맥화 정도가 높지만, 이는 모든 단어 위치와 문맥 크기에 체계적으로 일반화되지 않음을 보여줍니다. 실험 결과, 단어 위치와 문맥 길이에 따라 BERT 서브-레이어에서의 문맥화의 강도가 다르게 나타나는 것으로 확인되었습니다.



### PTD-SQL: Partitioning and Targeted Drilling with LLMs in Text-to-SQL (https://arxiv.org/abs/2409.14082)
Comments:
          EMNLP 2024 Main Conference. Revised by ARR April and ARR June. 32 pages, 7 figures and 30 tables

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 Text-to-SQL 작업에 있어 논리적 사고능력을 발휘할 수 있는 방법을 제안합니다. 특히, 문제 유형에 따라 쿼리 그룹 분할(query group partitioning)을 활용함으로써 LLM들이 특정 문제 유형에 대한 사고 과정을 더 잘 학습할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서 제안하는 PTD-SQL(Problem Type Differentiation SQL)은 LLM들이 다양한 난이도와 문제 카테고리에서 더 뛰어난 추론 능력을 가지도록 돕습니다. 이를 통해 LLM들이 전통적인 SQL 솔루션 접근 방법과의 차별성을 가지고, 특정 문제 유형에 대한 사고 과정을 심도 있게 학습할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, 여러 고급 LLM들이 PTD-SQL로 강화된 후 Spider 및 BIRD 데이터셋에서 이전의 최첨단(SOTA) 방법들을 능가하거나 동등한 성능을 발휘했습니다. 특히, 초기 성능이 다양한 모델들이 집중적인 훈련(targeted drilling)을 받은 후 큰 향상을 보였으며, 이는 인간의 능력 진전을 연상케 합니다.



### MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder (https://arxiv.org/abs/2409.14074)
Comments:
          Preprint

- **What's New**: 이번 연구는 Medical 분야의 다국어 자동 음성 인식(ASR) 시스템에 대해 소개하며, 5개 언어(베트남어, 영어, 독일어, 프랑스어, 중국어)로 구성된 MultiMed 데이터셋과 다양한 크기의 ASR 모델을 제공합니다. MultiMed는 지금까지의 의료 ASR 데이터셋 중 가장 크고 다양성 있는 데이터셋입니다.

- **Technical Details**: MultiMed는 150시간에 달하는 인적 주석이 달린 의료 분야 음성 데이터를 포함하고 있으며, 이는 다양한 질병, 녹음 조건, 화자 역할 및 고유한 의료 용어를 나타냅니다. 연구진은 End-to-End ASR 훈련을 위해 Layer-wise ablation study를 수행하고, 언어적 분석을 통해 다국어 의료 ASR에 대한 통찰력을 제공합니다.

- **Performance Highlights**: MultiMed는 5개 언어로 150시간의 데이터를 포함하는 세계 최대의 의료 ASR 데이터셋으로 평가받고 있습니다. 이 데이터셋은 ASR 시스템의 성능 향상을 위한 기초를 제공하며, 향후 다양한 의료 애플리케이션, 음성 번역, 그리고 음성 인식 기반의 비서 시스템 개발을 가능하게 합니다.



### Temporally Consistent Factuality Probing for Large Language Models (https://arxiv.org/abs/2409.14065)
- **What's New**: 이 연구에서는 TeCFaP라는 새로운 Temporally Consistent Factuality Probe 작업을 도입하고, TEMP-COFAC라는 고품질의 영어 쿼리 패러프레이즈 데이터셋을 제안합니다. 이를 통해 LLM의 일관된 사실성과 시간적 일관성을 평가하고 개선할 수 있습니다.

- **Technical Details**: TeCFaP는 (key_object, subject-relation, value_object) 형식의 쿼리 구조를 통해 시간적 연결성을 탐색하는 작업입니다. CoTSeLF(Consistent-Time-Sensitive Learning Framework)는 다중작업 지시 튜닝(MT-IT)과 시간 일관성 민감 강화학습(CTSRL)을 결합하여 LLM의 시간적 일관된 사실성을 개선합니다.

- **Performance Highlights**: 실험 결과, CoTSeLF는 temporal factuality, temporal consistency 및 temporally consistent factuality에서 이전 최선 모델보다 각각 12.7%, 10.9%, 90.4% 개선된 성능을 보였습니다.



### Co-occurrence is not Factual Association in Language Models (https://arxiv.org/abs/2409.14057)
- **What's New**: 이 연구에서는 언어 모델이 실제 사실 관계를 학습하는 데 어려움을 겪는 이유를 분석하고, 구체적인 전략을 제안하여 언어 모델이 사실 연관(factual associations)을 보다 효과적으로 학습할 수 있도록 돕는 방법을 제시합니다.

- **Technical Details**: 언어 모델에서 두 가지 형태의 지식 표현, 즉 단어 공변량 통계(co-occurrence statistics)와 실제 사실 연관(factual associations)을 구분했습니다. 연구 결과, 단어 공변량 통계는 트랜스포머 모델의 중간층에 주로 저장되며, 단순한 질문 응답을 넘어서는 추론 시나리오에 일반화되지 않는 반면, 실제 사실 연관은 모델의 하위층(lower layers)에 저장되어 다양한 추론 작업에서 자유롭게 활용할 수 있다는 사실을 발견했습니다.

- **Performance Highlights**: 제안된 두 가지 전략을 통해 새로 학습된 지식의 일반화가 유의미하게 향상되었습니다. 모형을 암묵적 사실 연관이 포함된 텍스트로 학습시킬 경우, 단순한 내러티브 텍스트를 사용할 때보다 새로운 사실을 더 잘 일반화할 수 있는 것으로 나타났습니다.



### GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion (https://arxiv.org/abs/2409.14051)
Comments:
          18 pages

- **What's New**: 이번 연구에서는 다수의 에이전트가 참여하는 논쟁 방식에서 토큰 비용을 대폭 줄이는 혁신적인 방법인 GroupDebate (GD)를 제안합니다. 기존의 방법들보다 더욱 효율적이면서도 성능을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: GroupDebate 방법은 모든 에이전트를 여러 논쟁 그룹으로 나누고, 그룹 내에서 내부 논쟁을 거친 후 중간 결과를 공유하는 방식을 채택합니다. 이렇게 함으로써 토큰 사용량을 약 51.7%까지 줄이고, 정확도 역시 최대 25%까지 향상시킬 수 있음을 실험을 통해 입증하였습니다.

- **Performance Highlights**: 실험 결과, GroupDebate는 기존 다수의 에이전트 논쟁 방법들에 비해 Arithmetic, GSM8K, MMLU, MATH 데이터셋에서 각각 45%에서 51.7%까지 토큰 비용을 절감하며, MMLU와 MATH 데이터셋에서는 최대 25%의 정확도 향상을 보여주었습니다.



### Can LLMs replace Neil deGrasse Tyson? Evaluating the Reliability of LLMs as Science Communicators (https://arxiv.org/abs/2409.14037)
- **What's New**: 이 논문에서는 현재의 Large Language Models(LLMs)를 과학 커뮤니케이터로서의 신뢰성을 평가하는 새로운 접근 방식을 소개합니다. 기존의 벤치마크와는 달리, LLM의 과학적 질문 응답 과제를 기반으로 LLM의 이해도를 평가하는 SCiPS-QA라는 새로운 데이터셋을 도입했습니다.

- **Technical Details**: SCiPS-QA 데이터셋은 복잡한 과학 개념에서의 742개의 Yes/No 질문으로 구성되며, 이를 통해 LLM의 정확성과 일관성을 다양한 기준으로 평가합니다. 실험에는 OpenAI의 GPT 시리즈와 Meta의 Llama 시리즈 및 Mistral 시리즈의 LLM이 포함됩니다. 고급 과학적 이해가 필요한 질문에 대한 LLM의 성능을 테스트하기 위해 다수의 평가 기준을 적용하였습니다.

- **Performance Highlights**: 대부분의 오픈 액세스 모델은 GPT-4 Turbo에 비해 떨어지지만, Llama-3-70B 모델은 다양한 평가에서 GPT-4 Turbo를 초과하는 경우가 있었습니다. 인간 평가자들이 GPT-4 Turbo의 잘못된 응답에 속아 넘어가는 경향도 관찰되었습니다.



### Uncovering Latent Chain of Thought Vectors in Language Models (https://arxiv.org/abs/2409.14026)
Comments:
          2 Pages, Intended for Tiny Papers 2025 Submission to ICLR

- **What's New**: 이 연구는 언어 모델(LM)의 행동을 선도하기 위해 'steering vector'라는 새로운 기법을 소개합니다. 이는 특정 작업에서 파생된 steering vector를 사용하여 Chain of Thought (CoT) Reasoning을 유도하며, 기존 자연어 프롬프트 없이도 이를 가능하게 합니다.

- **Technical Details**: 연구진은 Llama3 8b 및 Mistral 7b v0.2 모델에서 steering vector를 활용하여 CoT Reasoning을 진행하였습니다. 이들은 자연어 프롬프트 쌍을 대비시켜, 각 레이어의 활성화를 추출하고 이를 통해 최종 steering vector를 생성했습니다. PyTorch를 사용하여 추출된 레이어에 vector를 주입하는 방식으로 적용되었습니다.

- **Performance Highlights**: 이 접근 방식은 CoT 프롬프트를 사용한 모델들과 비교하여 경쟁력 있는 성능을 보였으며, 다양한 Reasoning benchmark(GSM8k, MMLU, ARC AI2)에서도 일관되게 CoT 응답을 유도하는 결과를 보여주었습니다. 또한, 전통적인 모델 미세 조정 방법보다 계산 비용이 절감된다는 장점이 있습니다.



### Graph Neural Network Framework for Sentiment Analysis Using Syntactic Featur (https://arxiv.org/abs/2409.14000)
- **What's New**: 소셜 미디어 플랫폼과 전자 상거래 생태계의 빠른 발전에 따라, 의견 분석(opinion mining) 분야가 자연어 처리(natural language processing)에서 중요한 연구 영역으로 떠오르고 있습니다. 본 연구는 텍스트 맥락 내 특정 요소에 대한 세밀한 평가를 추출하는 독특한 프레임워크를 제안합니다.

- **Technical Details**: 제안된 시스템은 구문 구조(syntactic structures)를 행렬(matrix) 형식으로 변환하고, 이 과정에서 그래프 내의 컨볼루션(convolutions) 및 어텐션(attention) 메커니즘을 활용하여 중요한 특징(salient characteristics)을 증류(distill)합니다. 설명자의 위치적 관련(positional relevance)을 어휘 항목(lexical items)과 연관시키는 방식은 입력의 순차적 무결성을 향상시킵니다.

- **Performance Highlights**: 실험(trials) 결과, 이 통합된 그래프 중심(graph-centric) 방안이 평가 범주화(evaluative categorization)의 효율성을 현저히 증대시키며 뛰어난 성능을 보여주었습니다.



### Contrastive Learning for Knowledge-Based Question Generation in Large Language Models (https://arxiv.org/abs/2409.13994)
Comments:
          5 pages, 2 figures

- **What's New**: 이번 연구는 고품질 질문 생성(question generation)을 지원하기 위한 새로운 방법론을 제안합니다. 특히, 지식 기반 질문 생성 기술에 초점을 맞추고 있으며, 이 과정에서 발생하는 착각(hallucination)과 지식 격차를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 contrastive learning을 통합하여 여러 모델이 도메인 지식을 공동으로 탐색하도록 하며, 생성 과정에서의 잡음과 착각을 줄이도록 유도합니다. 또한, contrasting examples를 포함한 프롬프트(prompt)를 설계하여 질문 생성 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 대조적인 지시 및 예제를 동시에 사용하는 경우 질문 생성의 품질이 크게 향상되었으며, 이는 높은 정확도를 이끌어냅니다. 제안된 방법은 대조적 맥락과 사고의 흐름(chain-of-thought) 프롬프트를 결합함으로써 질문 생성의 품질과 실용성을 효과적으로 개선할 수 있음을 보여주었습니다.



### SMART-RAG: Selection using Determinantal Matrices for Augmented Retrieva (https://arxiv.org/abs/2409.13992)
Comments:
          Under Review

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 기법을 발전시켜, 질문 답변(QA) 작업에서 기존의 문제를 해결하기 위한 새로운 방법인 SMART(Selection using Matrices for Augmented Retrieval)를 소개합니다. 특히, 이 방법은 불필요한 중복성 및 모순된 정보를 효과적으로 제거하는 데 중점을 두고 있습니다.

- **Technical Details**: SMART는 Determinantal Point Processes (DPPs)를 이용하여 검색된 문서 간의 관련성(relationship), 다양성(diversity), 모순(conflict) 관계를 동시에 모델링합니다. 이를 통해 보다 질 높은 맥락(context)을 선택하면서도 문서들 사이의 중복성과 충돌을 효과적으로 방지합니다. SMART는 Natural Language Inference (NLI) 모델을 통해 텍스트 간의 모순 관계를 평가하며, Cosine similarity를 이용해 관련성을 결정합니다.

- **Performance Highlights**: 여러 데이터셋에서 SMART를 적용한 결과, QA 성능이 유의미하게 향상되었으며, 이전의 비지도 학습 방법들보다 우수한 성능을 나타내었습니다. SMART는 RAG 분야에서 새로운 가능성을 제시하는 방법으로 각광받고 있습니다.



### ChemEval: A Comprehensive Multi-Level Chemical Evaluation for Large Language Models (https://arxiv.org/abs/2409.13989)
- **What's New**: 최근 화학 분야에서 LLMs(대형 언어 모델)가 수행하는 역할에 대한 관심이 증가하고 있으며, 이를 바탕으로 화학적 작업을 평가하기 위한 LLMs 벤치마크가 개발되고 있습니다.

- **Technical Details**: 본 연구에서는 ChemEval을 제안하며, 이는 화학 분야 내 다양한 작업에 대한 LLMs의 역량을 종합적으로 평가합니다. ChemEval은 4개의 주요 단계와 12개의 차원을 포함하여 42개의 화학 작업을 평가합니다. 이러한 작업은 오픈소스 데이터와 화학 전문가가 세심하게 구성한 데이터를 기반으로 합니다.

- **Performance Highlights**: 실험 결과, GPT-4와 Claude-3.5와 같은 일반 LLMs는 문헌 이해 및 지시 수행에서는 우수한 성능을 보이나, 고급 화학 지식이 필요한 작업에서는 부족한 성과를 보였습니다. 반면, 전문 LLMs는 향상된 화학 역량을 보여주나, 문학적인 이해력은 감소하는 경향이 있습니다. 이는 화학 분야의 복잡한 작업을 수행할 때 LLMs의 능력을 향상시킬 수 있는 잠재력을 시사합니다.



### Bias and Toxicity in Role-Play Reasoning (https://arxiv.org/abs/2409.13979)
Comments:
          14 pages, 9 figures, 9 tables

- **What's New**: 이번 논문에서는 Large Language Model (LLM)에서 역할 놀이(role-play)의 복잡한 역할과 관련된 잠재적인 위험을 체계적으로 평가하였습니다. 역할 놀이는 모델의 문맥 이해와 추론 개선을 위한 중요한 기술로 자리 잡고 있지만, 편향(bias)과 유해한 콘텐츠(harmful content) 생성의 가능성도 내포하고 있다는 점을 강조합니다.

- **Technical Details**: 우리는 다양한 벤치마크를 통해 LLM이 역할을 수행할 때의 성능 차이를 분석하였으며, 성별, 직업, 인종, 종교 등 여러 요인이 역할 놀이의 결과에 미치는 영향을 살펴보았습니다. 또한 서로 다른 LLM 간의 상호작용을 통한 실험도 진행하며 자동 역할 선택(auto-role selection)이 추론(capacity) 능력을 향상시키면서도 유해한 결과를 초래할 수 있음을 발견하였습니다.

- **Performance Highlights**: 각 벤치마크에서 모델에 따른 성능 변화를 비교한 결과, 역할에 따라 편향 및 유해성이 다르다는 것을 증명했습니다. 특정 역할을 수행할 때, LLM이 생성하는 응답의 유해성이 증가하는 경향을 보였으며, 이로 인해 역할 놀이에서의 사전 검증의 필요성이 대두되었습니다.



### Can Language Model Understand Word Semantics as A Chatbot? An Empirical Study of Language Model Internal External Mismatch (https://arxiv.org/abs/2409.13972)
Comments:
          10 pages, 1 figure, 5 tables

- **What's New**: 이 연구는 언어 모델이 단어 의미를 이해하는데 있어 내부 및 외부 표현 간의 불일치(discrepancy)를 조사합니다. 특히 Encoder-only, Decoder-only, Encoder-Decoder 모델 간의 차이를 살펴봅니다.

- **Technical Details**: 연구에서는 단어 유사성(word similarity), 구조적 예측(structured prediction), 유추(analogy)라는 세 가지 작업을 통해 언어 모델의 단어 의미 이해를 분석합니다. linear probing 방법을 사용하여 모델의 내부 상태와 외부 출력 간의 일치 여부를 평가합니다.

- **Performance Highlights**: 기존 연구와 달리, 본 연구에서는 queries와 probes 간의 유의미한 차이를 발견하였으며, 이는 단어 수준 의미를 포착하는 데 있어서 queries의 한계를 강조합니다.



### Exploring Automated Keyword Mnemonics Generation with Large Language Models via Overgenerate-and-Rank (https://arxiv.org/abs/2409.13952)
Comments:
          EMNLP 2024 findings

- **What's New**: 본 연구에서는 키워드 기억법(keyword mnemonics)이라는 언어 및 어휘 학습의 비교적 탐색이 덜 된 분야를 다루고 있습니다. 이 기술은 기억할 만한 연관성을 통해 어휘를 암기하는 방법입니다. 특히, 대규모 언어 모델(large language models, LLMs)를 활용하여 자동 생성된 기억법을 제안합니다.

- **Technical Details**: 우리는 먼저 LLM을 사용하여 음절 키워드의 집합을 생성한 다음, 해당 키워드와 관련된 언어적 단서를 생성하는 두 단계의 오버생성(overgenerate) 및 순위 매기기(rank) 방법을 제안합니다. 이 과정에서 심리언어학적(psycho-linguistic) 측정치와 사용자 연구 결과를 바탕으로 후보를 평가합니다.

- **Performance Highlights**: LLM이 생성한 기억법은 이미지 가능성(imageability), 일관성(coherence), 사용 유용성에서 인간이 작성한 기억법과 비슷하거나 더 나은 성능을 보였습니다. 그러나 언어 학습자의 배경 및 선호도에 따라 개선 여지가 아직 상당히 남아 있습니다.



### Mufu: Multilingual Fused Learning for Low-Resource Translation with LLM (https://arxiv.org/abs/2409.13949)
Comments:
          29 pages

- **What's New**: 새로운 연구에서 Mufu라는 다국어 지원 시스템을 소개하며, 이는 자동 생성된 다국어 후보를 선택하고 부정확한 번역을 수정하는 지침을 포함합니다. 이 접근 방식은 저자원 환경에서도 번역의 데이터 효율성을 극대화하여 LLM의 추론 기능을 활용합니다.

- **Technical Details**: Mufu는 번역 작업을 후편집(post-editing) 작업으로 전환시킬 수 있는 프롬프트를 사용하며, 이 프롬프트는 입력 품질을 평가하고 의미를 교차 언어적으로 맞추며 적절한 입력에서 내용을 복사하고 잘못된 경우를 무시하는 것을 포함합니다. 이 연구에서는 Flores-200 데이터셋을 기반으로 En-XX 번역에 대한 실험을 진행했습니다.

- **Performance Highlights**: Mufu 스타일의 프롬프트로 미세 조정된 LLM은 낮은 품질의 보조 번역 후보에 강인하게 나타났으며, 64%의 저자원 및 매우 저자원 언어 쌍에서 NLLB 1.3B distilled 모델보다 우수한 성능을 보였습니다. 또한, 이 모델의 증류(distillation)를 통해 추론 비용을 줄이면서도 저자원 번역에서 평균 3.1 chrF 개선을 달성하였습니다.



### Aligning Language Models Using Follow-up Likelihood as Reward Signa (https://arxiv.org/abs/2409.13948)
Comments:
          16 pages, reward model, LLM Alignment

- **What's New**: 본 논문은 사용자 피드백을 활용하여 사용자 요청에 대한 기계의 응답을 평가하는 방법을 제안합니다. 이 방법론에서는 Follow-up Likelihood as Reward (FLR) 메커니즘을 통해 응답의 선호도를 자동적으로 식별할 수 있습니다.

- **Technical Details**: FLR 메커니즘은 두 가지 주요 개념에 기반하여 설계되었습니다. 첫 번째는 사용자의 후속 발언을 보상 신호로 활용하여 응답의 유용성을 평가하는 것입니다. 두 번째는 기본 정책 모델의 온라인 세대에서 선호 데이터를 자동으로 추출하여 이 데이터를 DAP 기법을 통해 조정하는 것입니다. 이 과정에서 자연어 피드백 데이터로 언어 모델을 미세 조정하여 성능을 극대화합니다.

- **Performance Highlights**: FLR는 대규모 인간 주석 데이터 기반의 강력한 보상 모델과 성능에서 동등한 결과를 보였으며, Llama-3-8B-Instruct 모델의 Alpaca-Eval V2에서 길이 조절 승률을 4.45% 향상시켰습니다. 미세 조정을 통해 FLR는 보상 모델링 과제에서 실질적인 성능 향상을 달성했습니다.



### MirrorStories: Reflecting Diversity through Personalized Narrative Generation with Large Language Models (https://arxiv.org/abs/2409.13935)
Comments:
          5 pages (excluding references), accepted to EMNLP 2024 Main Conference

- **What's New**: 이 연구는 개인화된 '미러 스토리' 제작에 있는 대규모 언어 모델(LLMs)의 효과성을 탐구하며, 문학의 다양성이 부족한 점을 해결하려고 합니다.

- **Technical Details**: 미러 스토리는 이름, 성별, 연령, 민족, 독자 관심사 및 이야기 도덕성과 같은 요소들을 통합하여 생성된 1,500개의 개인화된 단편 이야기의 집합입니다. 이 연구는 26인의 다양한 인간 평과자를 통해 LLMs가 생성한 개인화된 이야기가 일반적인 인간 저술 및 LLM 저술의 이야기들에 비해 높은 점수를 기록함을 보여줍니다.

- **Performance Highlights**: 개인화된 LLM 생성 이야기는 참여도 모든 지표에서 평균 4.22 점(5점 만점)에 비해 3.37 점으로 일반적인 이야기들보다 뛰어난 성과를 보였습니다. 이러한 이야기는 텍스트의 다양성을 높이면서도 원래의 도덕성을 유지하는 결과를 가져옵니다.



### One Model is All You Need: ByT5-Sanskrit, a Unified Model for Sanskrit NLP Tasks (https://arxiv.org/abs/2409.13920)
- **What's New**: 이 논문에서는 형태학적으로 풍부한 언어인 산스크리트를 위한 새로운 프리트레인(pretrained) 언어 모델인 ByT5-Sanskrit를 소개합니다. 이 모델은 산스크리트어 단어 분할 작업에서 이전의 데이터 기반 접근 방식보다 더 나은 성과를 보여주며, 현재 최상의 어휘 기반 모델과 동일한 성능을 보입니다.

- **Technical Details**: ByT5-Sanskrit 모델은 문자 수준의 정보 처리를 기반으로 하며, 대규모 산스크리트 데이터에 대한 프리트레인 이후 여러 NLP Downstream 태스크에 대해 공동 미세 조정(fine-tuning)을 수행합니다. 이 모델은 2023년 최신 NLP 벤치마크에서 새로운 최고 성능(SOTA) 결과를 달성하였습니다.

- **Performance Highlights**: ByT5-Sanskrit는 Hackathon SWS 벤치마크에서 완벽 문장 일치 점수(PM)에서 8.8점의 성과 향상을 달성하며, SIGHUM 데이터셋에서도 현재 최고 성능 모델과 근접한 성과를 보입니다. 또한 Vedic 종속 구문 분석(UAS, LAS) 작업과 OCR 후 교정 작업에서도 기존 최고 방법을 초월한 성과를 냈습니다.



### Target word activity detector: An approach to obtain ASR word boundaries without lexicon (https://arxiv.org/abs/2409.13913)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 연구에서는 라이선스의 의존성을 배제하고 단어 경계(word boundaries)를 추정하는 새로운 방법을 제시합니다. 이 방법은 다국어 자동 음성 인식(multilingual ASR) 모델에서 구동될 수 있으며, 여러 언어에 대해 추가 비용 없이 확장 가능합니다.

- **Technical Details**: 제안된 Target Word Activity Detector (TWAD) 모델은 서브워드 토큰 유닛으로부터 단어 임베딩을 학습하고, 이 정보를 사용해 음성 신호 내에서 단어의 활동을 식별합니다. 훈련 중에는 단어 정렬 정보만 필요하여 이전의 방법들보다 더 쉽게 적용할 수 있습니다. TWAD 모델은 기존의 CTC 및 AED 모델들과의 조합을 통해 작동합니다.

- **Performance Highlights**: 이 접근법은 영어, 프랑스어, 스페인어, 이탈리아어, 독일어 총 5개 언어로 훈련된 다국어 ASR 모델을 이용해 검증되었습니다. 성능 비교에서 강력한 베이스라인에 비해 효과적으로 단어 시간 추정(word timing estimation) 성능을 개선했음을 입증하였습니다.



### Enhancing Large Language Models with Domain-specific Retrieval Augment Generation: A Case Study on Long-form Consumer Health Question Answering in Ophthalmology (https://arxiv.org/abs/2409.13902)
- **What's New**: 대규모 언어 모델(LLM)이 의료 분야에서의 잠재력에도 불구하고, 확증되지 않은 증거를 바탕으로 하거나 허구의 (hallucinated) 증거가 포함된 반응을 생성할 수 있다는 점에서 중요한 연구 결과가 제시되었습니다. 본 연구에서는 검색 강화 생성 (RAG, Retrieval Augment Generation) 방식이 의료 분야의 하위 도메인 특정 응용 프로그램에 적용된 몇 안 되는 사례 중 하나로, 70,000개의 안과 관련 문서를 활용한 RAG 파이프라인을 개발하였습니다.

- **Technical Details**: 연구는 장문의 소비자 건강 질문에 대한 사례 연구를 통해 진행되었습니다. 100개의 질문에 대해 RAG를 사용할 경우와 사용하지 않을 경우 LLM의 답변을 10명의 의료 전문가와 함께 체계적으로 평가하였습니다. 평가 항목은 증거의 사실성(factuality), 증거 선택 및 순위 (selection and ranking), 증거 귀속(attribution), 답변의 정확성(accuracy) 및 완전성(completeness)을 포함합니다.

- **Performance Highlights**: RAG가 없는 LLM은 총 252개의 참조 문헌을 제공했으나, 45.3%가 허구였고, 34.1%는 minor error가 있었으며, 20.6%만이 올바른 정보였습니다. 반면, RAG가 적용된 경우 정확성이 크게 향상되어 54.5%가 올바른 정보로 나타났고, 오류 비율은 18.8%로 줄어들었습니다. RAG에 의해 검색된 상위 10개 문서 중 62.5%가 LLM 반응에서 주요 참조로 선택되었으며, 평균 순위는 4.9에 달했습니다. RAG의 사용은 증거 귀속에서도 개선을 보여주었으나(5점 척도에서 1.85에서 2.49로 증가, P<0.001), 정확성은 약간 감소했습니다 (3.52에서 3.23로). RAG는 의료 분야에서의 하위 애플리케이션에 대한 우려를 해소하며, 허구 및 오류의 비율을 크게 감소시켰음을 제시합니다.



### LLM for Everyone: Representing the Underrepresented in Large Language Models (https://arxiv.org/abs/2409.13897)
Comments:
          PhD thesis

- **What's New**: 이 논문은 다국어 설정에서 특히 소외된 언어에 대한 대규모 언어 모델(LLMs)의 한계를 다루고 있습니다.

- **Technical Details**: 소외된 언어에서의 LLM 능력을 평가하기 위한 포괄적인 평가가 수행되었으며, 다국어 및 다문화 일반화의 과제가 드러났습니다. 제안된 방법론은 cross-lingual continual instruction tuning, retrieval-based cross-lingual in-context learning, 및 in-context query alignment을 포함합니다. 또한, 서로 다른 언어에서 작동하는 LLM 간의 문화적 가치 일치를 측정하기 위한 새로운 방법이 제안되었습니다.

- **Performance Highlights**: 이 연구는 소외된 언어에서도 효과적인 일반화를 가능하게 하여 다국어 및 다문화 조화성을 향상시키는 것을 목표로 하고 있습니다.



### Transfer Learning with Clinical Concept Embeddings from Large Language Models (https://arxiv.org/abs/2409.13893)
- **What's New**: 이 연구는 여러 임상 사이트에서 수집된 전자 건강 기록을 분석하여 대형 언어 모델(Large Language Models, LLMs)의 의미적 임베딩이 의료 분야의 지식 전이에 미치는 영향을 평가하였습니다.

- **Technical Details**: 연구는 Med-BERT와 같은 도메인 특화 LLM이 지역 및 직접 전이 시나리오에서 일관되게 우수한 성능을 보인다는 것을 보여주었습니다. 반면에 OpenAI 임베딩과 같은 일반 모델은 최적 성능을 위해 미세 조정(fine-tuning)이 필요합니다. 그러나 생의학 임베딩을 가진 모델의 과도한 조정은 효율성을 감소시킬 가능성이 있습니다.

- **Performance Highlights**: 도메인 특정 임베딩과 신중한 모델 조정의 중요성을 강조하며, 의료 분야에서 효과적인 지식 전이를 위해 이러한 접근 방식이 필요하다는 결론에 도달했습니다.



### A Multi-LLM Debiasing Framework (https://arxiv.org/abs/2409.13884)
- **What's New**: 본 연구에서는 다중 LLM(multi-LLM) 접근 방식을 제안하여 LLM의 편향(bias)을 효과적으로 감소시키고자 합니다. 특별히, 중앙 집중식(centralized)과 분산식(decentralized) 두 가지 방법을 도입하여 비교하였으며, 분산식 방법이 우수한 성능을 보임을 확인했습니다. 이러한 방식은 사회적 그룹 내의 편향을 효과적으로 줄이는 데 기여합니다.

- **Technical Details**: 다중 LLM 프레임워크를 통해 여러 모델이 대화 방식으로 상호작용하며, 편향을 줄이는 방법을 제안합니다. 여기서 중앙 집중식 방식은 하나의 LLM이 대화를 지원하며, 분산식 방식은 모든 모델이 직접 소통합니다. 제안된 BBQ-Hard 벤치마크를 통해 두 방법을 평가하였고, BBQ-Hard 데이터셋은 LLM의 편향을 더 효과적으로 테스트할 수 있는 어려운 문제가 포함되어 있습니다.

- **Performance Highlights**: 다중 LLM 방법은 여러 사회적 그룹에서 기존의 기준 방법(baseline)보다 일관되게 더 우수한 성능을 보였습니다. 연구 결과는 다중 LLM 기반의 편향 감소 프레임워크가 LLM의 출력에서 편향을 유의미하게 줄일 수 있음을 시사합니다.



### "I Never Said That": A dataset, taxonomy and baselines on response clarity classification (https://arxiv.org/abs/2409.13879)
Comments:
          Accepted at Findings of EMNLP 2024

- **What's New**: 본 연구에서는 정치 인터뷰에서 질문에 대한 응답의 명확성을 평가하는 새로운 작업, 즉 response clarity evaluation을 제안합니다. 이를 위해 새로운 구조화된 분류 체계를 도입하고, 관련된 정치 인터뷰에서의 질문-응답(QA) 쌍으로 이루어진 데이터셋을 구축하였습니다.

- **Technical Details**: 제안한 분류 체계는 고수준의 정보 제공과 저수준의 회피 기술을 포함한 두 단계로 구성되어 있습니다. 첫 번째 단계는 응답 명확성을 세 가지 해석 수에 따라 평가하며, 두 번째 단계는 정치 문헌에서의 일반적인 11가지 회피 기법에 대한 세부적인 분류를 제공합니다. 또한, ChatGPT와 인간 주석자를 결합하여 QA 쌍을 수집하고 검증하는 과정을 거쳤습니다.

- **Performance Highlights**: 다양한 모델 아키텍처 및 적응 방법을 실험하여 제안된 데이터셋과 작업에 대한 새로운 기준을 설정하였습니다. 단순한 프롬프트와 지침 조정 기법이 높은 성능을 제공하며, 두 단계 분류 전략을 통해 회피 레이블을 사용할 경우 명확성 분류 성능이 향상되었습니다.



### Instruct-Tuning Pretrained Causal Language Models for Ancient Greek Papyrology and Epigraphy (https://arxiv.org/abs/2409.13870)
Comments:
          7 pages, 1 table. Under review

- **What's New**: 이번 연구에서는 Meta의 Llama 3.1 8B Instruct 모델을 활용하여 고대 그리스 비문과 문서 파피루스의 연대 및 지리적 속성과 텍스트 복원 작업을 위한 미세 조정을 수행하였다. 이 모델은 기존 최고 기록을 초월하였으며, 특히 문서 복원에서 고전적 및 지리적 속성 부여에서 뛰어난 성능을 보였다.

- **Technical Details**: 연구진은 문서 비문과 파피루스를 위한 데이터 세트를 수집하고, 텍스트 복원, 지리적 속성 부여 및 연대 추정을 위한 전처리 과정을 거쳤다. 모델은 문자 에러율(CER) 22.5%와 지리적 속성 부여에서 top-1 정확도 75.0%를 달성하였다. 또한, 파피루스의 텍스트 복원에서는 CER 16.3%와 top-1 정확도 71.3%를 기록하였다.

- **Performance Highlights**: 미세 조정된 모델은 고대 비문의 텍스트 복원에서 평균 22.5%의 CER을 기록했으며, 지리적 속성 부여에서 75%의 top-1 정확도를 달성하였다. 또한 고대 그리스 문서 파피루스에 대한 새로운 기준을 수립하였으며, 연대 측정에서의 평균 편차는 26.2년으로, 기존에 비해 뛰어난 성능을 보였다.



### Unlocking Memorization in Large Language Models with Dynamic Soft Prompting (https://arxiv.org/abs/2409.13853)
- **What's New**: 본 논문은 LLM(대형 언어 모델)의 암기(memorization) 문제를 해결하기 위해 동적 소프트 프롬프트(dynamic soft prompts)를 사용하는 새로운 방법을 제안합니다. 이전 방법들은 입력의 변화에 반응하지 못하는 고정된 소프트 프롬프트만을 사용했으나, 본 방법은 입력 변화에 적응할 수 있는 프롬프트를 생성합니다.

- **Technical Details**: 제안된 방법은 transformer 기반 생성기(generator)를 활용하여 입력에 따라 동적으로 변경되는 소프트 프롬프트를 생성합니다. 이는 암기된 데이터를 보다 정확히 추출할 수 있게 해줍니다. 연구 결과, 본 방법은 기존 기술들과 비교하여 뛰어난 성능을 보였으며, 다양한 실험 환경에서 검증되었습니다.

- **Performance Highlights**: 본 방법은 텍스트 생성(task)과 코드 생성(task) 모두에서 vanilla 기준선 대비 각각 112.75% 및 32.26%의 최대 상대 개선을 달성했습니다. 이를 통해 동적 소프트 프롬프트의 효과성을 입증했습니다.



### Do language models practice what they preach? Examining language ideologies about gendered language reform encoded in LLMs (https://arxiv.org/abs/2409.13852)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 생성한 텍스트에서 언어 이데올로기(language ideologies)를 조사했으며, 특히 성별 구분이 있는 영어 표현의 개혁에 대한 사례 연구를 포함합니다. 이는 정치적 편향을 드러내고, LLM의 메타언어적 선호가 특정 정치 집단의 언어 이데올로기를 어떻게 암시적으로 전달하는지를 보여줍니다.

- **Technical Details**: 연구 결과, LLM은 '올바른(corresponding)' 또는 '자연스러운(natural)' 언어 사용에 대한 요청에 대해 보수적(conservative) 가치에 좀 더 유사한 언어를 생성하는 경향이 있음을 발견했습니다. 또한, LLM은 더 명확한 메타언어적(context) 맥락이 제공될 때 성 중립적(gender-neutral) 변형을 보다 자주 사용하는 내부 불일치(internal inconsistency)를 나타냈습니다.

- **Performance Highlights**: 이 연구는 LLM이 생성하는 텍스트에서 나타나는 언어 이데올로기가 사용자의 예상과 다를 수 있음을 강조하고, 이러한 결과가 가치 정렬(value alignment)과 관련된 더 넓은 함의를 갖고 있음을 논의합니다.



### STOP! Benchmarking Large Language Models with Sensitivity Testing on Offensive Progressions (https://arxiv.org/abs/2409.13843)
Comments:
          9 pages (excluding references), accepted to EMNLP 2024 Main Conference

- **What's New**: 본 연구에서는 Large Language Models (LLMs)에서의 명시적 및 암시적 편향을 평가하기 위한 새로운 접근 방식으로 Sensitivity Testing on Offensive Progressions (STOP) 데이터셋을 소개합니다. 이 데이터셋은 2,700개의 고유 문장을 포함하는 450개의 공격적 진행 상황을 제공하며, 다양한 심각도를 다룹니다.

- **Technical Details**: STOP 데이터셋은 9개의 인구 통계학적 그룹과 46개의 하위 인구 통계학적 그룹을 포괄하여 편향을 다양한 각도에서 평가할 수 있도록 설계되었습니다. 모델의 편향 인식 능력을 평가하기 위해 GPT-4, Mixtral, Llama 3와 같은 여러 주요 모델에 대한 실험이 수행되었습니다. 각 모델의 편향 인식 성공률은 19.3%에서 69.8%까지 다양했습니다.

- **Performance Highlights**: STOP 데이터셋을 활용하여 Llama 3-70b 모델을 파인 튜닝한 결과 BBQ, StereoSet 및 CrowS-Pairs 등에서 최대 191%의 높은 응답률을 달성하며 성능을 유지하거나 개선하는 성과를 보여주었습니다.



### Measuring Copyright Risks of Large Language Model via Partial Information Probing (https://arxiv.org/abs/2409.13831)
Comments:
          8 pages, 8 figures

- **What's New**: 본 논문은 Large Language Models (LLMs)가 저작권이 있는 콘텐츠를 생성할 수 있는 능력을 탐구합니다. 구체적으로, 저작권 내용의 일부 정보를 제공하고 생성된 콘텐츠와 원본 저작물 간의 겹침을 분석하는 방법을 사용하였습니다.

- **Technical Details**: 연구팀은 저작권이 있는 텍스트의 조각을 LLMs에 입력하고 이들이 이를 완성하도록 요청하는 실험을 진행하며, 여기에 Rouge-L Score를 사용하여 생성된 텍스트와 저작권 자료 간의 유사성을 평가하였습니다. 또한, 반복적인 프롬프트 기법을 사용하여 더 많은 저작권 침해 콘텐츠를 생성할 수 있는 가능성을 탐구했습니다.

- **Performance Highlights**: Llama, GPT-4-Turbo와 같은 대규모 매개변수를 가진 모델들이 특히 저작권 자료와 높은 유사성을 보였으며, 특정 유형의 콘텐츠에서는 성능 차이를 보였습니다. 예를 들어, GPT-4-Turbo는 노래 가사를 생성하는 데에서 더 높은 유사도를 나타냈습니다.



### Local Explanations and Self-Explanations for Assessing Faithfulness in black-box LLMs (https://arxiv.org/abs/2409.13764)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 충실성을 평가하는 새로운 작업을 도입했습니다. 이를 위해 로컬 섭동(local perturbations)과 자기 설명(self-explanations)을 활용한 효율적인 설명 가능성(explainability) 기술을 제안합니다.

- **Technical Details**: 이 연구에서는 LLM이 올바른 답변을 생성하는 데 필요한 충분하고 필수적인 부분을 식별하기 위해 일반적으로 사용되는 leave-one-out(LOO) 접근방식을 적용합니다. 우리는 Natural Questions 데이터셋을 사용하여 이 방법을 검증하며, LLM의 자기 설명이 실제로 모델 결정에 어떻게 기여하는지 평가하기 위해 고안된 메트릭을 제안합니다.

- **Performance Highlights**: 제안된 접근법은 모델의 결정 과정을 설명하고 충실성을 평가하는 데 매우 효과적임을 보여주었습니다. 특히 사용자의 질문에 대한 올바른 답변을 생성하는 데 중요한 키워드를 체계적으로 식별함으로써 LLM의 동작 방식에 대한 유의미한 통찰력을 제공합니다.



### Do Large Language Models Need a Content Delivery Network? (https://arxiv.org/abs/2409.13761)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)에서 새로운 지식을 유연하고 효율적으로 주입하는 방법으로 KV 캐시(KV caches)를 활용하는 방법에 대해 제안합니다. 기존의 fine-tuning과 in-context learning 방법에 비해 KV 캐시를 사용하는 것이 지식 주입의 모듈성(modularity) 및 효율성(efficiency)을 동시에 향상시킬 수 있다고 주장합니다. 이를 실현하기 위해 Knowledge Delivery Network(KDN)을 구상하였으며, 이는 LLM 서비스에서 KV 캐시를 동적으로 최적화하는 시스템 구성 요소입니다.

- **Technical Details**: KDN은 LLM에서 처리된 지식을 관리하는 백엔드 시스템으로, KV 캐시의 저장, 전송 및 조합을 최적화하는 기능을 갖추고 있습니다. KDN은 KV 캐시의 위계 구조와 압축을 활용하는 대규모 저장소, KV 캐시를 LLM 엔진 간에 빠르게 전송하는 스트리밍 시스템, 여러 조각의 지식을 결합하는 블렌딩 모듈을 포함합니다. 기존의 LLM 서비스 시스템과 달리 KV 캐시 관리와 LLM 서비스 엔진을 명확히 분리하여 모듈성과 효율성을 개선합니다.

- **Performance Highlights**: KDN 프로토타입 실험을 통해, KV 캐시 학습(KV-cache learning) 방법이 in-context learning과 fine-tuning 방법보다 더 나은 모듈성과 효율성을 보여줄 수 있음을 입증할 수 있는 초기 기술이 이미 연구되어 있다는 점을 강조합니다.



### Optimizing the Songwriting Process: Genre-Based Lyric Generation Using Deep Learning Models (https://arxiv.org/abs/2409.13758)
- **What's New**: 이 논문은 심층 학습 기술을 사용하여 전통적인 작사 과정을 단순화하는 방법을 제안합니다. 18,000곡의 Spotify 노래 데이터를 이용해 LSTM 기반 데이터 모델을 개발하여 장르별로 최적화된 가사를 생성하는 데 초점을 맞춰, 작사 과정을 가속화하는 목표를 세워 두 가지 모델을 비교했습니다.

- **Technical Details**: 본 연구에서는 seq2seq와 LSTM 모델을 사용하여 작업을 수행하였으며, T5 모델을 참조하여 사전 훈련된 모델과 독자적인 LSTM 모델 두 가지를 구축했습니다. 특별히, LSTM 모델의 입력으로 짧은 구문이나 단어를 받아 100자의 구문을 출력하도록 설정하고, 손실 함수는 Cross Entropy를 사용했습니다.

- **Performance Highlights**: 기본 모델이 ROUGE 메트릭에서 더 높은 리콜을 보인 반면, 두 모델 모두 BLEU 메트릭에서는 유사한 정밀도를 나타냈습니다. 생성된 가사 구문들이 특정 장르에 인식 가능하고 이해할 수 있는 수준에 이르렀음을 확인했습니다. 전체적으로, 가사 생성을 가속화하고 장르에 따른 가사를 효과적으로 생성할 수 있음을 보여주었습니다.



### Efficient Hybrid Inference for LLMs: Reward-Based Token Modelling with Selective Cloud Assistanc (https://arxiv.org/abs/2409.13757)
- **What's New**: 이번 논문에서는 고유한 하이브리드 추론(hybrid inference) 접근 방식을 제안합니다. 이 방법은 대형 언어 모델(LLMs)과 소형 언어 모델(SLMs)의 장점을 활용하면서도 비용이 많이 드는 클라우드 LLM의 의존도를 최소화합니다.

- **Technical Details**: 기존의 접근 방식은 전체 쿼리를 SLM이나 클라우드 LLM으로 라우팅하는 방식이나, 본 논문에서는 보상 기반 메커니즘(reward-based mechanism)을 도입하여 토큰 생성 중 클라우드 LLM의 참여 여부를 동적으로 결정합니다. SLM이 예측한 각 토큰은 보상 점수(reward score)로 평가되며, 이 점수가 기준치(threshold) 이하일 때만 클라우드 LLM의 도움을 요청합니다.

- **Performance Highlights**: 실험 결과, 본 기법은 클라우드 LLM의 사용을 획기적으로 줄였으며, 응답 품질에 미치는 영향은 최소한으로 유지되어 비용 효율적인 고성능 언어 모델 배포 솔루션을 제공합니다.



### Language Models Learn Metadata: Political Stance Detection Case Study (https://arxiv.org/abs/2409.13756)
- **What's New**: 이 연구에서는 정치적 입장 감지를 위한 메타데이터의 최적 통합 방법을 탐구하고, 단순한 베이지안 모델이 메타데이터만을 사용하여 기존의 모든 모델을 초월하는 성능을 보인다는 것을 보여줍니다.

- **Technical Details**: 이 논문은 ParlVote+ 데이터셋을 사용하여 정치적 발언 분석을 수행하며, 메타데이터(예: 정당 및 정책)를 포함하는 간단한 선행(prepending) 메커니즘이 더욱 효과적임을 입증합니다. MPNet 모델을 통해 텍스트 정보를 통합하고, 기존의 복잡한 모델링 방식을 비판합니다.

- **Performance Highlights**: 단순한 모델이기에도 불구하고, 이 연구는 큰 생성 언어 모델보다 소형 파인튜닝된 인코더 기반 언어 모델이 제로샷(Zero-shot) 설정에서 더 높은 성능을 나타낼 수 있다는 점을 강조합니다.



### Entity-Aware Self-Attention and Contextualized GCN for Enhanced Relation Extraction in Long Sentences (https://arxiv.org/abs/2409.13755)
- **What's New**: 본 논문에서는 기존의 의존성 기반 접근 방식의 한계를 극복하고자 Entity-aware Self-attention Contextualized GCN (ESC-GCN) 모델을 제안합니다. 이 모델은 입력 문장의 구문 구조와 의미적 문맥을 효과적으로 결합하여 관계 추출 성능을 향상시킵니다.

- **Technical Details**: ESC-GCN 모델은 상대 위치 self-attention을 통해 단어 위치와 관련된 전반적인 의미적 쌍상관성을 획득하고, 컨텍스트 그래프 컨볼루션 네트워크(Convolutional Networks)를 통해 단어 간의 복잡한 내부 문장 종속성을 포착합니다. 또한, entity-aware attention layer를 통해 최종 관계 예측을 위한 중요한 토큰을 동적으로 선택합니다.

- **Performance Highlights**: 다양한 작업에 대한 광범위한 실험에서 ESC-GCN 모델이 기존의 의존성 기반 및 시퀀스 기반 모델들에 비해 뛰어난 성능을 달성했음을 보여주었습니다. 특히 긴 문장에서의 엔티티 간 관계 추출에서 두드러진 성과를 보였습니다.



### Thinking Before Speaking: A Role-playing Model with Minds (https://arxiv.org/abs/2409.13752)
- **What's New**: 이번 논문에서는 Thinking Before Speaking (TBS) 모델을 제안하며, 역할 기반 대화에서 LLM의 성능을 개선하기 위한 새로운 접근법을 소개합니다. 이 모델은 캐릭터의 실제 시나리오를 바탕으로 데이터를 확장하고, 캐릭터의 사고 패턴을 반영하여 LLM이 더욱 사실적인 역할을 수행할 수 있도록 합니다.

- **Technical Details**: TBS 모델은 각 대화 쌍에 대해 캐릭터의 마음가짐을 보완하고, 특정 지식 이상의 질문을 포함한 일부 데이터를 추가하여 LLM을 미세 조정합니다. 이렇게 함으로써 LLM은 캐릭터의 사고 흐름과 논리를 채택하게 되며, 캐릭터의 지식 기반을 벗어나는 응답을 피할 수 있습니다. 이 연구는 새로운 데이터셋과 평가 지표를 마련하여 LLM의 능력을 시험합니다.

- **Performance Highlights**: 실험 결과, TBS 모델은 긴 대화 과정에서 톤, 지식, 마음가짐 측면에서 역할을 더 잘 모방할 수 있음을 보여주었으며, 이는 사용자 경험을 향상하는 데 기여합니다.



### KodeXv0.1: A Family of State-of-the-Art Financial Large Language Models (https://arxiv.org/abs/2409.13749)
Comments:
          11 pages, 8 figures

- **What's New**: 본 논문에서는 KodeXv0.1이라는 새로운 대형 언어 모델 패밀리를 소개합니다. 이 모델은 GPT-4를 초월하는 성능을 보이며, 주로 재무 분야에서 질문 답변을 수행하는 데 최적화되어 있습니다.

- **Technical Details**: KodeXv0.1은 Llama 3.1 8B 및 70B의 기본 변형을 사용하여 특수한 재무 영역을 위해 커스터마이즈된 교육 체계를 통해 발전되었습니다. 이를 위해 공개적으로 사용 가능한 재무 문서를 대량으로 수집하고 처리하여 Context-Question-Answer triplet 형태의 고품질 합성 데이터셋을 생성했습니다. 모델 튜닝은 RAG-aware 4bit LoRA 방법을 사용하여 수행되었습니다.

- **Performance Highlights**: 모델 평가 결과 KodeX-8Bv0.1은 동일한 매개변수 범위 내에서 최신 모델보다 최대 9.24% 더 신뢰성이 높은 결과를 보여주었고, GPT-4보다도 최대 7.07% 우수한 성능을 발휘했습니다. KodeX-70Bv0.1은 모든 테스트 벤치마크에서 GPT-4의 성능을 초과하는 개선을 나타냈습니다.



### TheraGen: Therapy for Every Generation (https://arxiv.org/abs/2409.13748)
Comments:
          12 pages, 11 figures

- **What's New**: 이번 논문에서는 LLaMA 2 7B 모델을 활용하여 개발한 고급 AI 기반 정신 건강 챗봇인 TheraGen을 소개합니다. TheraGen은 100만 개의 대화 입력 데이터를 이용하여 개인화되고 연민이 담긴 정신 건강 관리를 제공하며, 최근의 언어 모델과 트랜스포머 아키텍처의 발전을 기반으로 합니다.

- **Technical Details**: TheraGen은 transfer learning, fine-tuning 및 고급 훈련 기법을 활용하여 최적의 성능을 달성합니다. 클라우드 기반 아키텍처를 통해 고성능 및 가용성이 보장되고, 24시간 언제든지 접근할 수 있는 지원을 제공합니다. 이 시스템은 사용자 친화적인 인터페이스를 제공하여 공감적인 반응과 근거 기반 대처 전략을 제공합니다.

- **Performance Highlights**: 사용자 만족도 평가 결과에 따르면, 94%의 사용자들이 정신적 웰빙이 향상되었다고 보고했습니다. 또한, BLEU 점수는 0.67, ROUGE 점수는 0.62로 응답 정확성이 높음을 나타냅니다. 평균 응답 시간은 1395 밀리초로, 실시간으로 효율적인 지원을 보장합니다.



### Machine Translation with Large Language Models: Decoder Only vs. Encoder-Decoder (https://arxiv.org/abs/2409.13747)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)을 활용하여 인도 지역 언어인 텔루구어, 타밀어, 말라얄람어를 포함한 다국어 기계 번역 모델을 개발하는 데 중점을 두고 있습니다. 특히 Decoder-only 아키텍처와 Encoder-Decoder 아키텍처를 비교하여 번역 품질과 효율성을 최적화하고자 하며, 이는 다양한 언어 쌍 간의 정확하고 맥락에 적합한 번역을 지원합니다.

- **Technical Details**: 이 연구에서는 Encoder-Decoder와 Decoder-only 모델의 성능을 평가하기 위한 체계적인 방법론을 제안합니다. In-Context Learning(Few-Shot Learning)을 통해 기계 번역 쌍을 생성하고, 이를 평가하기 위해 BLEU 메트릭을 사용합니다. XGLM과 mT5 모델을 활용하였으며, XGLM은 500백만 개의 파라미터를 가진 Decoder-only 모델이고, mT5는 3억 개의 파라미터를 가진 Encoder-Decoder 모델로서 다국어 번역에 특화되어 있습니다.

- **Performance Highlights**: 실험 결과, mT5 모델이 인도 언어 쌍에 대해 보다 높은 BLEU 점수를 기록하며 우수한 번역 품질을 보였습니다. 연구는 XGLM과 mT5의 번역 성능을 비교하여 각 모델의 최적 사용 사례와 관련된 통찰력을 제공합니다.



### When Less Is Not More: Large Language Models Normalize Less-Frequent Terms with Lower Accuracy (https://arxiv.org/abs/2409.13746)
- **What's New**: 이 연구는 대규모 언어 모델(GPT-4o)이 인체 표현형 온톨로지(HPO)를 기반으로 한 용어 정규화에서 11,225개의 고유한 용어를 처리할 때 단 13.1%의 정확도를 달성했다는 점을 강조합니다. 이는 저주파 및 긴 용어가 더 많은 정규화 오류를 초래한다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 268,776개의 표현형 주석을 가지는 HPO 데이터셋을 활용하여, 용어의 빈도와 길이에 따라 정규화 정확도를 분석했습니다. 특히, SHAP 및 순열 방법을 사용하여 정규화 오류의 주요 예측 요인으로 낮은 용어 빈도를 확인했습니다.

- **Performance Highlights**: 정확도는 높은 빈도에서 시작하여, 주파수가 낮아질수록 급격히 감소했습니다. 특히 용어의 길이가 길어질수록 정확도가 떨어지는 경향이 있으며, ANOVA 분석을 통해 이 두 요인 간의 중요한 상호작용이 발견되었습니다.



### Context-Aware Membership Inference Attacks against Pre-trained Large Language Models (https://arxiv.org/abs/2409.13745)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 사전 훈련 과정에서 발생하는 맴버십 추론 공격(Membership Inference Attack, MIA)의 새로운 접근 방식을 제안합니다. 기존 MIA는 LLM의 순차적 텍스트 생성 과정을 고려하지 않았습니다.  본 연구에서는 MIA 통계 테스트를 데이터 포인트의 서브시퀀스(perplexity dynamics)에 적응시켜 효과적인 공격을 수행했습니다.

- **Technical Details**: CAMIA(상황 인식 맴버십 추론 공격) 프레임워크를 설계하여 LLM의 다음 토큰 예측 손실(sequence of next-token prediction losses)에서 조정된 맴버십 정보를 추출합니다. 이 방법은 프리픽스(prefix)의 길이, 다양성, perplexity 등의 맥락적 요소를 반영하여 조정된 테스트를 수행합니다. 손실의 기울기, 변동성, 이상치는 멤버와 비멤버를 구분하는 데 중요한 지표입니다. 이러한 동적 행동을 통해 모델의 다양한 상황에 따라 공격 결정을 조정합니다.

- **Performance Highlights**: CAMIA는 Pythia와 GPT-Neo 모델을 포함한 9999개의 사전 훈련된 LLM을 대상으로 평가했으며, 기존 MIA보다 3배 더 많은 멤버를 성공적으로 식별했습니다. 예를 들어, Pythia 모델에서 CAMIA는 1%의 특정 잘못 예측률(FPR)에서 3.35배 높은 진짜 긍정률(TPR)을 달성했습니다. GPT-Neo 모델에서도 TPR이 20% 증가했습니다. 이는 다양한 데이터 도메인에서 일관되게 높은 성능을 보였습니다.



### A Simplified Retriever to Improve Accuracy of Phenotype Normalizations by Large Language Models (https://arxiv.org/abs/2409.13744)
Comments:
          Submitted to Frontiers in Digital Health

- **What's New**: 이 연구는 생물 의학에서 주로 사용되는 딥 러닝 모델을 사용하여 표현형 용어 정규화의 정확성을 향상시키는 새로운 방법을 제안하고 있습니다. 특히 BioBERT를 사용한 문맥적 단어 임베딩에 기반하여 복잡한 정의 생성 없이 빠르고 효과적으로 후보 용어를 매칭시키는 간소화된 검색기를 도입했습니다.

- **Technical Details**: 제안된 방법은 검증된 1,820개의 표현형 용어에 대해 Human Phenotype Ontology (HPO)와의 코사인 유사성을 기준으로 키워드 후보를 선택하고, 이를 통해 LLM이 의미론적으로 가장 적당한 정규화를 선택하게 합니다. LLM은 GPT-4o를 사용하게 되며, 이 과정에서 BioBERT를 통한 두 가지 실험 조건과 LLM + Retriever 방식이 비교됩니다.

- **Performance Highlights**: 정규화 정확도는 기존 62.3%에서 90.3%로 향상되었습니다. 특히 LLM + Retriever 방법은 BioBERT 방법보다 높은 정확도를 보이며, 자동화된 정규화 솔루션의 필요성을 충족할 수 있는 잠재력을 보여줍니다.



### Knowing When to Ask -- Bridging Large Language Models and Data (https://arxiv.org/abs/2409.13741)
Comments:
          39 pages - 25 page paper, 14 page Appendix, 7 figures, 9 tables

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 정확성을 향상시키기 위한 새로운 접근 방식을 소개합니다. 데이터 커먼스(Data Commons)라는 신뢰할 수 있는 통계 데이터를 제공하는 오픈 소스 데이터 저장소와 통합하여 모델의 성능을 개선하고자 하였습니다.

- **Technical Details**: 이 논문에서는 두 가지 주요 방법론을 사용합니다: 1) Retrieval Interleaved Generation (RIG), 이 방법은 LLM이 데이터 커먼스로부터 정보를 검색하기 위해 자연어 쿼리를 생성하도록 훈련됩니다. 2) Retrieval Augmented Generation (RAG), 이 방법에서는 관련 데이터 테이블을 데이터 커먼스에서 가져오고 이를 사용하여 LLM의 프롬프트를 보강합니다.

- **Performance Highlights**: 제안된 방법은 다양한 쿼리에 대해 평가되었으며, LLM 출력의 사실적 정확성을 향상시키는 데 있어 효과적임을 입증했습니다. 이는 확인 가능한 통계 데이터에 기반하여 더 신뢰할 수 있는 LLM 구축을 위한 초기 단계로 자리잡고 있습니다.



### Language agents achieve superhuman synthesis of scientific knowledg (https://arxiv.org/abs/2409.13740)
- **What's New**: 이번 연구에서는 PaperQA2라는 고급 언어 모델을 개발하여 과학 문헌 검색 과제에서 인간 전문가와의 성능 비교를 수행했습니다. 결과적으로 PaperQA2는 정보 검색, 요약 및 모순 탐지 작업에서 주제 전문가를 초월할 수 있음을 입증하였습니다.

- **Technical Details**: 연구의 핵심은 PaperQA2를 활용하여 적용된 LitQA2 벤치마크를 통해 과학 문헌에 대한 정보 검색, 요약 및 모순 탐지의 성능을 평가한 점입니다. PaperQA2는 ‘retrieval-augmented generation (RAG)’ 접근 방식을 사용하여, 여러 단계의 작업을 통해 최종 응답을 생성합니다. 각 질문에 대해 평균 14.5개의 논문을 활용하였으며, 정확도는 66.0%로 나타났습니다.

- **Performance Highlights**: PaperQA2는 LitQA2 벤치마크에서 85.2%의 정밀도와 66.0%의 정확도를 기록하였으며, 생물학 논문에서 평균 2.34개의 모순을 발견하는 데 성공했습니다. 이 연구는 AI 모델이 특정 과학 문헌 작업에서 인간 전문가보다 뛰어난 성능을 보여줄 수 있음을 실증적으로 나타냈습니다.



### Table-to-Text Generation with Pretrained Diffusion Models (https://arxiv.org/abs/2409.13739)
Comments:
          IEEE Access

- **What's New**: 이 논문에서는 확산 모델(‘diffusion models’)의 새로운 적용 방식에 대해 다루고 있습니다. 특히, 표(table)에서 텍스트로의 변환(table-to-text) 문제에 대한 효과적인 솔루션을 모색하였고, DPM-Solver++라는 최신 확산 모델 가속기를 핵심 모델에 도입하여 샘플링 전략의 영향을 분석했습니다.

- **Technical Details**: 확산 모델은 여러 단계의 노이즈 제거 과정을 통해 높은 품질의 출력을 생성하는 반복 생성형 모델입니다. 본 연구에서는 GENIE 모델을 사용하여 ToTTo 도전 과제에 적용하였고, 생성 출력의 길이 제한, 예측 집계 방법(ROVER 및 Minimum Bayes-Risk)과 같은 여러 요소의 영향을 조사했습니다. 또한, 다양한 온도 설정에서 자동 회귀(text-to-text) 모델과의 비교를 통해 모델의 성능을 평가했습니다.

- **Performance Highlights**: 본 연구 결과, 확산 모델은 자동 회귀 모델에 비해 품질과 다양성 간의 균형을 유지하며 경쟁력 있는 결과를 도출할 수 있음을 발견했습니다. 특히, 가장 높은 품질을 위해서는 엄격한 길이 제약을 가진 일반 샘플러 사용 후 MBR을 통한 예측 집aggregating 방법이 바람직하며, 높은 수준의 다양성을 포기하고 처리 속도를 높이기를 원한다면 DPM-Solver++와 같은 빠른 샘플러를 활용할 수 있습니다.



### NLP4PBM: A Systematic Review on Process Extraction using Natural Language Processing with Rule-based, Machine and Deep Learning Methods (https://arxiv.org/abs/2409.13738)
- **What's New**: 이 문헌 리뷰에서는 텍스트 설명을 구조화된 프로세스로 변환하는 자동화된 프로세스 추출 분야를 연구하였습니다. 최근 Machine Learning (ML) 및 Deep Learning (DL) 방법들이 Natural Language Processing (NLP) 구성 요소에 점점 더 많이 사용되고 있으며, 이들은 전통적인 규칙 기반 방법보다 더 나은 성능을 보여주고 있습니다.

- **Technical Details**: 논문에서는 자동화된 프로세스 추출의 방법론으로 NLP와 프로세스 생성을 두 단계로 나누어 설명합니다. 첫 번째 단계인 NLP에서는 텍스트의 기본 구성 요소를 분류하고, 두 번째 단계인 프로세스 생성에서는 NLP 출력을 프로세스 모델로 변환하여 제어 흐름(control-flow)과 결정적 요소(decisional elements)를 캡처합니다. 최근에는 Transformers(예: BERT), Long Short-Term Memory (LSTM)와 같은 DL 모델이 많이 사용되고 있으며, Large Language Models (LLMs)의 출현으로 이 분야에 대한 관심이 증가하고 있습니다.

- **Performance Highlights**: 자동화된 프로세스 추출이 정확하고 신뢰할 수 있게 이루어진다면, 효율성을 크게 향상시킬 수 있습니다. 그러나 현재 금준 스탠다드, 스케일러블 주석 데이터셋의 부족은 객관적인 평가와 ML/DL 방법의 교육에 걸림돌이 되고 있습니다. 이 연구에서는 NLP 도구 및 ML/DL 기반의 최신 연구 결과를 반영하여 프로세스 추출에 대한 시스템 리뷰를 제공합니다.



### Analysis of Socially Unacceptable Discourse with Zero-shot Learning (https://arxiv.org/abs/2409.13735)
- **What's New**: 이번 연구는 Socially Unacceptable Discourse (SUD) 분석을 위한 새로운 접근법으로, Entailment 기반 (based) 제로샷 텍스트 분류 (text classification) 방법을 제안합니다. 이 방법은 사전 훈련된 (pre-trained) 변환기 모델 (transformer models)과 프롬프트 기법 (prompting techniques)을 활용하여 SUD 탐지 및 특성 분석에 효과적임을 보여줍니다.

- **Technical Details**: 연구에서는 사전 훈련된 변환기 모델을 바탕으로 SUD 분석을 수행하며, 레이블이 없는 데이터에서 효과적으로 작동하는 제로샷 텍스트 분류 방식을 사용합니다. 이를 통해 극단주의 서사 (extremist narratives)를 분석하고 특성화하기 위한 레이블이 있는 데이터셋 생성이 가능함을 강조합니다.

- **Performance Highlights**: 연구 결과는 모델이 이전에 보지 못한 데이터에 대해 우수한 일반화 능력 (generalization capabilities)을 갖추고 있음을 증명하였으며, SUD 연구 및 온라인 책임 있는 소통을 촉진하기 위한 강력한 도구 개발에 기여할 것으로 기대됩니다.



### Enhancing Kurdish Text-to-Speech with Native Corpus Training: A High-Quality WaveGlow Vocoder Approach (https://arxiv.org/abs/2409.13734)
- **What's New**: 이 논문은 중앙 쿠르드어(Central Kurdish, CKB)에 맞춘 새로운 TTS(텍스트-투-스피치) 시스템의 개발을 다룹니다. 기존의 영어 사전 훈련 모델 대신 21시간의 중앙 쿠르드어 음성 말뭉치(corpus)를 사용하여 커스텀 WaveGlow 보코더(vocoder)를 훈련시켰습니다.

- **Technical Details**: 기존의 Tacotron 기반 시스템을 개선하여, 중앙 쿠르드어에 특화된 WaveGlow 보코더를 훈련했습니다. 이는 쿠르드어의 음소(phonetic) 및 억양(prosodic) 변화를 정확하고 유창하게 적용하기 위해 필요합니다.

- **Performance Highlights**: 최종적으로, 제안된 어댑티브 WaveGlow 모델은 4.91의 MOS(mean opinion score)를 달성하며, 이는 쿠르드어 음성 합성의 새로운 기준을 제시합니다. 이 연구는 중앙 쿠르드어 TTS 시스템의 고급 기능을 강화하며, 다른 쿠르드어 방언과 관련 언어의 발전에도 기여할 수 있는 가능성을 열었습니다.



### RNR: Teaching Large Language Models to Follow Roles and Rules (https://arxiv.org/abs/2409.13733)
- **What's New**: 이 논문에서는 기존의 Instruction Fine-Tuning (IFT) 모델이 복잡한 역할과 규칙을 따르는 데 실패한다는 문제를 해결하기 위해 RoleNRules (RNR)라는 데이터 생성 파이프라인을 제안합니다. 이 파이프라인은 기존 IFT 지침으로부터 다양한 역할과 규칙을 자동으로 생성하여 LLM 모델의 성능을 향상시킵니다.

- **Technical Details**: RoleNRules는 고유한 (system prompt, instruction, response) 트리플을 생성하여 모델이 복잡한 시스템 prompt를 따르도록 훈련할 수 있도록 설계된 데이터 생성 파이프라인입니다. 이 과정에서 LLM을 통해 다양한 역할(description)과 규칙(rules)을 생성하고, 생성된 시스템 prompt와 원래 지침을 기반으로 응답을 생성합니다.

- **Performance Highlights**: RNR로 훈련된 모델은 Traditional Instruction Fine-Tuning에 비해 25% 이상 증가한 규칙 준수 pass-rate를 기록했으며, 일반적인 지침을 따르는 성능 저하 없이 복잡한 지침을 계속해서 따를 수 있는 능력을 유지했습니다.



### TopoChat: Enhancing Topological Materials Retrieval With Large Language Model and Multi-Source Knowledg (https://arxiv.org/abs/2409.13732)
- **What's New**: 대규모 언어 모델(LLMs)을 활용하여 특정 분야의 요구를 충족시키고 대규모 처리를 최적화하는 새로운 접근 방식을 제안합니다. 이를 위해 재료 지식 그래프(MaterialsKG)를 구축하고, 이를 문헌과 통합하여 토폴로지 재료를 위한 대화 시스템인 TopoChat을 개발했습니다.

- **Technical Details**: 이 시스템은 여러 출처의 데이터를 통합하고, 실험 및 이론 계산에서 생성된 방대한 양의 정보를 이용하여 효율적인 정보 검색을 가능하게 합니다. TopoChat은 복잡한 질문에 대해 더 나은 성능을 발휘하며, 재료 추천 및 관련 관계 추론 과제를 수행할 수 있습니다.

- **Performance Highlights**: TopoChat은 기초 LLM에 비해 구조 및 속성 질의, 재료 추천, 복잡한 관계 추론에서 우수한 성능을 보였습니다. 이는 효율적이고 정밀한 정보 검색을 가능하게 하여 응축 물질 분야의 발전을 촉진합니다.



### KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation (https://arxiv.org/abs/2409.13731)
Comments:
          33 pages

- **What's New**: Knowledge Augmented Generation (KAG)은 전문 도메인 지식 서비스를 위한 새로운 프레임워크로, 기존의 Retrieval-Augmented Generation (RAG)의 한계를 극복하기 위해 개발되었습니다. KAG는 Knowledge Graph (KG)와 벡터 검색의 장점을 결합하여 대규모 언어 모델(LLMs)과 KG의 상호 작용을 통해 생성 및 추론 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: KAG는 다섯 가지 핵심 요소를 통해 LLM 친화적인 지식 표현(LLMFriSPG), 지식 그래프와 원래 텍스트 청크 간의 상호 색인화, 논리적 형태 기반의 하이브리드 추론 엔진, 의미론적 추론에 따른 지식 정렬, KAG를 위한 모델 기능 향상을 설계하였습니다. 이 프레임워크는 복잡한 Q&A 데이터셋에서 성능을 평가하여 상당한 개선 결과를 나타냈습니다.

- **Performance Highlights**: KAG는 2wiki에서 19.6%, hotpotQA에서 33.5%의 F1 점수를 개선하며 기존의 RAG 방식보다 전문성이 크게 강화된 결과를 보여주었습니다. 또한, KAG는 Ant Group의 E-Government 및 E-Health Q&A 작업에 적용되어 전통적인 RAG 방법에 비해 높은 정확도를 기록하였습니다.



### MathGLM-Vision: Solving Mathematical Problems with Multi-Modal Large Language Mod (https://arxiv.org/abs/2409.13729)
Comments:
          30 pages,19 figures

- **What's New**: 이 논문에서는 다양한 시각적 요소가 포함된 수학 문제를 해결하기 위한 MathGLM-Vision이라는 다중 모달 대형 언어 모델(Multi-Modal Large Language Model, MLLM)을 소개합니다. 특히 이 모델은 MathVL이라는 세밀하게 조정된 데이터셋을 사용하여 수학 문제를 해결하는 데 필요한 시각적 정보를 통합하도록 설계되었습니다.

- **Technical Details**: MathGLM-Vision은 GLM-4V-9B, CogVLM2, CogVLM-32B를 기초 모델로 하여 Fine-Tuning(미세 조정)을 진행하였으며, 다양한 매개변수 크기로 구성됩니다. MathVL 데이터셋은 산술, 대수학, 기하학 등 다양한 수학적 주제를 포함하고, 각 문제에는 단계별 솔루션이 제공되어 모델의 문제 해결능력을 향상시킵니다.

- **Performance Highlights**: MathGLM-Vision은 MathVL-test에서 시각적 입력을 포함한 경우, 텍스트 입력만을 사용한 모델에 비해 더 우수한 성능을 보였습니다. 예를 들어, 기하학 문제 해결(minitest)에서는, MathGLM-Vision-9B가 GLM-4V-9B에 대해 39.68% 성능 개선을, MathGLM-Vision-19B가 CogVLM2에 대해 65.06% 성능 개선을 보였습니다.



### Rule Extrapolation in Language Models: A Study of Compositional Generalization on OOD Prompts (https://arxiv.org/abs/2409.13728)
- **What's New**: 이번 연구는 autoregressive LLMs(대량의 언어 모델)의 OOD(out-of-distribution) 행동을 이해하기 위해 새로운 개념인 'rule extrapolation'을 정의했습니다. 이 개념은 프롬프트가 최소한 하나의 규칙을 위반하는 OOD 시나리오를 설명합니다.

- **Technical Details**: 연구에서는 규칙의 교차점으로 정의된 형식 언어(formal languages)에서의 OOD 구성 일반화(compositional generalization)를 고려했습니다. 다양한 복잡성의 형식 언어를 통해 Transformer, 선형 및 순환 아키텍처, 상태 공간 모델(state space models)에서 규칙 외삽(rule extrapolation)을 평가하였습니다.

- **Performance Highlights**: 연구는 Solomonoff prior(솔로모프 우선)에서 영감을 받은 규칙 외삽의 규범 이론(n normative theory)의 기초를 세우며, LLMs의 아키텍처가 규칙 외삽에 미치는 영향을 이해하는 데 중요한 통찰을 제공합니다.



### Classification performance and reproducibility of GPT-4 omni for information extraction from veterinary electronic health records (https://arxiv.org/abs/2409.13727)
Comments:
          24 pages, 3 figures, 8 supplementary figures

- **What's New**: 이번 연구는 수의학 전자 건강 기록(EHRs)에서 정보 추출을 위한 대형 언어 모델(LLMs)의 성능 차이 및 환경 설정(temperature settings)의 영향을 평가한 최초의 연구입니다. 특히 GPT-4 omni(GPT-4o)와 GPT-3.5 Turbo 모델 간의 비교를 통해 LLM 오류와 인간 관찰자 간의 합의 관계를 조사하였습니다.

- **Technical Details**: 연구는 250개의 EHRs를 분석하여 GPT-4o와 GPT-3.5 Turbo의 성능을 비교하였고, 여섯 가지 임상 징후를 식별하는 작업을 수행했습니다. GPT-4o는 0 온도에서 96.9%의 민감도(sensitivity), 97.6%의 특이도(specificity), 80.7%의 양성 예측 값(positive predictive value) 등 매우 높은 성능을 보였습니다. 반면에 GPT-3.5 Turbo는 오직 81.7%의 민감도를 기록했습니다.

- **Performance Highlights**: GPT-4o는 온도의 영향을 받지 않았으며, 인간 쌍 대비 평균 Cohen's kappa가 0.98로 뛰어난 재현성을 보여주었습니다. 또한 GPT-4o의 오류 대부분은 인간 간의 의견 불일치에서 발생하였으며, 이는 EHR의 모호성(ambiguity)에 기인함을 시사합니다. 따라서, GPT-4o는 수의학 EHR 정보 추출 자동화에 있어 유망한 대안으로 제시됩니다.



### Multilingual Dyadic Interaction Corpus NoXi+J: Toward Understanding Asian-European Non-verbal Cultural Characteristics and their Influences on Engagemen (https://arxiv.org/abs/2409.13726)
Comments:
          8 pages. 6 figures. International Conference on Multimodal Interaction, November 4-8, 2024, San Jose, Costa Rica

- **What's New**: 이번 연구에서는 비언어적 행동이 문화에 따라 어떻게 차별화되는지를 분석하고, 이러한 차이가 대화의 참여도(engagement) 인식에 미치는 영향을 평가하기 위해 다국어(multi-lingual) 비언어적 기능을 COMPUTATIONAL 분석했습니다. 이를 위해 기존의 NoXi 데이터셋을 확장하여 일본어와 중국어로 이루어진 대화 데이터가 포함된 새로운 데이터셋 NoXi+J를 구성했습니다.

- **Technical Details**: 비언어적 특성(non-verbal features)에는 음성 음향(speech acoustics), 표정(facial expressions), 백채널(backchanneling), 제스처(gestures)가 포함됩니다. 다양한 패턴 인식 기법을 통해 이러한 특성을 추출하고, 통계적 분석을 통해 각 언어에서 문화적으로 의존적이고 독립적인 특징을 식별했습니다. 최종적으로 LSTM(Long Short-Term Memory) 모델을 훈련하여 대화의 참여도를 예측하고, SHAP(Shapley Additive Explanations) 분석을 통해 입력 특성과 문화적 특성 간의 상관관계를 분석했습니다.

- **Performance Highlights**: 여섯 개의 기계 학습 모델이 NoXi+J 데이터셋의 서로 다른 언어 화자(subsets)에서 훈련되어 성능을 평가했습니다. 모델의 성능 결과는 분석 결과와 상관관계를 나타내며, 다양한 언어의 비언어적 특성에 따라 성능이 달라지는 것을 확인했습니다. 이 연구는 문화 차이에 대한 인식뿐만 아니라, 비언어적 소통과 대화 참여도의 예측에 있어 기계 학습의 중요성을 강조합니다.



### Identity-related Speech Suppression in Generative AI Content Moderation (https://arxiv.org/abs/2409.13725)
- **What's New**: 이 논문은 자동화된 콘텐츠 조정 시스템에서 마이너리티(미소수) 집단과 관련된 발언 억압(speech suppression)을 정의하고 측정하기 위한 새로운 벤치마크를 제시합니다. 현재 사용되는 콘텐츠 조정 API가 특정 정체성 집단에 대해 어떻게 차별적으로 작용하는지를 분석합니다.

- **Technical Details**: 본 연구에서는 콘텐츠 조정 API의 성능을 평가하기 위해, 사용자 생성 데이터셋과 생성 AI에 특화된 데이터셋을 포함한 총 7개 벤치마크 데이터셋을 사용했습니다. 아울러, 9개 정체성 카테고리를 사용하여 텍스트 데이터를 태그하는 방법론을 공개했습니다. 연구는 활성화된 API에서의 잘못된 태깅과 관련된 문제를 다루면서, 특정 정체성 그룹의 발언이 더 자주 억압(inappropriately flagged)되는지를 확인했습니다.

- **Performance Highlights**: 연구 결과, 자동화된 콘텐츠 조정 시스템은 모든 정체성 그룹에서 관련 발언을 더 자주 억압하며, 기독교 및 이성애자(group) 외의 모든 그룹에 대해 이질적인 결과를 보였습니다. 여러 API 간의 차이점을 분석하여, 생성 AI 콘텐츠의 적절한 조정 능력에 대한 정보를 제공합니다.



### Logically Consistent Language Models via Neuro-Symbolic Integration (https://arxiv.org/abs/2409.13724)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 논리적 일관성을 향상시키기 위해 신경-기호적 추론(neuro-symbolic reasoning)을 기반으로 한 손실(loss) 개념을 도입했습니다. 이를 통해 LLM이 외부의 사실(facts)과 규칙에 논리적으로 일관되도록 학습할 수 있게 하여, LLM의 자가 일관성(self-consistency)을 개선했습니다.

- **Technical Details**: 연구에서는 LLM을 훈련시키는 동안, 강화된 샘플 모델의 확률을 최대화하는 원칙적 목적(objective)을 수립했습니다. 이 방법은 추론 레퍼토리를 잠재적으로 확장하면서, LLM이 제공된 논리적 제약조건에 따라 진실성을 유지하도록 합니다. 실험을 통해, 제한된 사실 세트에 대해 훈련 받은 LLM이 새로운 사실에 대한 진실 신념을 학습할 수 있음을 보였습니다.

- **Performance Highlights**: LoCo-LMs(논리적으로 일관된 LLMs)로 명명된 이번 모델은 외부 해결기(solvers)에 의존하지 않고도, 자가 일관성과 사실성을 향상시킨 것으로 나타났습니다. 제한된 데이터 환경에서도, 기존의 감독적 fine-tuning에 비해 더 우수한 성능을 보였습니다.



### LegiLM: A Fine-Tuned Legal Language Model for Data Complianc (https://arxiv.org/abs/2409.13721)
Comments:
          9 pages, 2 figures

- **What's New**: 이 논문에서는 데이터 보안 및 개인 정보 보호 규정을 준수하기 위해 특별히 설계된 법률 언어 모델인 LegiLM을 소개합니다. LegiLM은 GDPR 법안에 대한 사전 훈련을 기반으로 하여, 특정 행동이나 사건이 데이터 보안 및 개인 정보 보호 규정을 위반했는지를 자동으로 평가할 수 있습니다.

- **Technical Details**: LegiLM 모델은 영어권 법률 자료에서 수집된 전문 법률 데이터로 광범위하게 사전 훈련되었으며, GDPR 관련 데이터셋을 통해 세밀하게 미세 조정되었습니다. 이 모델은 고급 법적 추론 방법과 정보 검색 기술을 통합하여 실제 법률 상담 시의 정확성과 신뢰성을 향상시킵니다.

- **Performance Highlights**: 우리의 평가 결과, LegiLM은 데이터 규제 위반 탐지에서 뛰어난 성능을 보였으며, 이를 통해 적절한 법적 정당성을 제공하고 필요한 준수 수정 사항을 추천할 수 있음을 입증했습니다. 이로 인해 AI 기반의 법적 준수 솔루션의 새로운 기준을 세우게 되었습니다.



### DiVA-DocRE: A Discriminative and Voice-Aware Paradigm for Document-Level Relation Extraction (https://arxiv.org/abs/2409.13717)
- **What's New**: 이번 연구에서는 Document-level Relation Triplet Extraction (DocRTE)을 위한 혁신적인 접근 방식인 DiVA(Discriminative and Voice Aware Paradigm)를 소개합니다. DiVA는 문서 수준에서의 관계 추출을 단순화하여 관계를 식별하기 위해 단순히 문서를 입력하는 방식으로 작동합니다.

- **Technical Details**: DiVA는 두 가지 주요 단계로 구성됩니다: (1) 관계 추출을 위한 문서 수준의 차별화(paradigm) 방식, (2) 추출된 관계에 기반하여 주체와 객체 엔티티를 식별하는 음성 인식(voice-aware) 방식. 이러한 접근 방식을 통해 모델은 문서 전체의 맥락을 이해하고 관계의 방향성 및 음성의 영향을 감지할 수 있습니다.

- **Performance Highlights**: 실험 결과, Re-DocRED 및 DocRED 데이터셋에서 DiVA는 DocRTE 작업에 대해 최고의 성능(SOTA)을 기록했습니다. 기존 방법론들과 비교했을 때, 더 많은 관계 유형을 처리하고, 음성을 고려한 주체와 객체 구분에서 우수한 결과를 보여줍니다.



### Constrained Multi-Layer Contrastive Learning for Implicit Discourse Relationship Recognition (https://arxiv.org/abs/2409.13716)
- **What's New**: 이번 연구에서는 암시적 담화 관계 인식(IDRR)을 위한 새로운 접근 방식으로 감독 대조 학습(supervised contrastive learning, CL) 방법을 제안합니다. 특히, 레이블 및 인스턴스 중심의 대조 학습(label- and instance-centered CL)을 통해 표현 학습을 강화하고, 어휘적으로 제약된 다층 대조 학습(constrained multi-layer CL) 기법을 소개하여 상위 층의 대조 손실이 하위 층보다 작도록 제약을 둡니다.

- **Technical Details**: IDRR을 수행하기 위해 복잡한 다층 신경망(multi-layer neural networks)을 사용했던 기존 방법과 달리, 본 논문에서는 단순하면서도 효과적인 레이블 및 인스턴스 중심 대조 학습(LICL)을 각 레이어에 적용합니다. 이를 통해 여러 레이어에서 LICL을 적용할 때 발생할 수 있는 중복을 줄이기 위한 제약을 추가한 CMCL(Constrained Multi-layer Contrastive Learning) 방식을 개발하였습니다.

- **Performance Highlights**: PDTB 2.0 및 PDTB 3.0 데이터셋에서 실험 결과, 제안한 접근 방식이 다중 클래스 분류(multi-class classification)와 이진 분류(binary classification) 모두에서 성능을 크게 향상시켰음을 보여줍니다. 특히, 중간 레이어에 LICL을 적용하는 것 만으로도 기존 모델에 비해 상당한 개선을 이루었습니다.



### Introducing MeMo: A Multimodal Dataset for Memory Modelling in Multiparty Conversations (https://arxiv.org/abs/2409.13715)
- **What's New**: MeMo 코퍼스는 참가자의 기억 유지 보고서로 주석이 달린 최초의 대화형 데이터세트로, 컴퓨터 모델링을 위한 소중한 자원으로 제공됩니다.

- **Technical Details**: MeMo 코퍼스는 Covid-19 주제에 대한 31시간 분량의 소규모 그룹 토론을 포함하며, 행동 및 지각 측정이 검증된 데이터와 함께 오디오, 비디오, 다중 모달 주석을 통합하고 있습니다. 이 데이터셋은 대화 기억 및 그룹 역학 연구에서 유용한 자료가 됩니다.

- **Performance Highlights**: MeMo 코퍼스는 대화 기억 모델을 구축하는 데 활용 가능하며, 이를 통해 사용자의 기억 및 사회적 상호작용에 대한 이해를 진전시킬 수 있습니다.



### TracrBench: Generating Interpretability Testbeds with Large Language Models (https://arxiv.org/abs/2409.13714)
Comments:
          6 pages + appendix, 4 figures, ICML Mechanistic Interpretability Workshop

- **What's New**: 이 연구에서는 트랜스포머(tansformer) 기반 언어 모델의 해석 가능성(interpretability)을 평가하기 위한 새로운 접근법을 제시합니다. 특히 TracrBench라는 새로운 데이터셋을 소개하며, 이는 121개의 수작업으로 작성된 RASP 프로그램과 LLM(대형 언어 모델) 생성, 인간 검증의 결과로 구성되어 있습니다.

- **Technical Details**: Tracr는 RASP에서 본래의 사실 기반 매핑(ground truth mappings)을 가진 컴파일된 트랜스포머 모델을 생성하는 방법입니다. TracrBench는 해석 가능성 평가 방법들을 검증하기 위해 고안된 테스트베드(test bed)로, LLM을 활용하여 RASP 프로그램을 자동으로 생성하고자 했으나, 이 과정에서 많은 도전과제가 있음을 발견했습니다.

- **Performance Highlights**: 최신 LLM인 GPT-4-turbo는 20-shot 프롬프트와 best-of-5 샘플링을 사용하였으나, 총 101개 테스트 프로그램 중 57개만을 올바르게 구현했습니다. TracrBench는 이러한 과정을 통해 해석 가능성 방법의 평가 및 비교를 위한 가치 있는 테스트베드 역할을 목표로 하고 있습니다.



### Sentiment Informed Sentence BERT-Ensemble Algorithm for Depression Detection (https://arxiv.org/abs/2409.13713)
- **What's New**: 세계 보건 기구(WHO)는 약 2억8000만명이 우울증으로 고통받고 있다고 발표했습니다. 하지만 머신러닝(ML) 기법을 활용한 초기 우울증 탐지에 대한 기존 연구는 제한적입니다. 본 연구는 여러 ML 알고리즘의 성능을 검토하여 초기 우울증 탐지를 개선하는 데 중점을 두었습니다.

- **Technical Details**: 우리는 두 개의 기준 소셜 미디어 데이터셋(D1, D2)을 사용하여 ML 알고리즘의 성능을 분석했습니다. 추가적으로 감정 지표(sentiment indicators)를 통합하여 모델의 성능을 향상시켰습니다. 실험 결과, 문장 양방향 인코더 표현(SBERT)에서 추출한 숫자 벡터를 스태킹 앙상블(stacking ensemble) 모델에 적합시켜 D1 데이터셋에서 69%, D2 데이터셋에서 76%의 F1 점수를 달성했습니다.

- **Performance Highlights**: 감정 지표를 추가적인 특성으로 활용하는 것이 우울증 탐지 모델의 성능 향상에 기여함을 보여주었으며, 향후 연구를 위해 우울증 용어 말뭉치(depressive term corpus)의 개발을 추천합니다.



### Good Idea or Not, Representation of LLM Could (https://arxiv.org/abs/2409.13712)
- **What's New**: 이 논문은 과학적 아이디어를 정량적으로 평가하는 새로운 프레임워크를 제안하며, 대형 언어 모델(LLMs)로부터 얻은 표현을 활용하여 아이디어의 가치를 정량화하는 방법을 탐구합니다. 또한, 약 4,000개의 원고로 구성된 벤치마크 데이터 세트를 공개합니다.

- **Technical Details**: 우리는 LLM의 특정 계층에서 생산된 표현을 사용하여 아이디어의 가치를 정량화하는 프레임워크를 수립하였습니다. 이 연구는 LLM의 표현을 통해 텍스트의 의미론적 특징을 인코딩하고 이를 다양한 아이디어 평가 방법과 결합하여 학문적 질적 평가를 목표로 합니다.

- **Performance Highlights**: 실험 결과, LLM에서 생성된 표현은 인간의 판단과 높은 일관성을 보여주었으며, LLM의 중간 및 후위 계층에서 얻어진 표현이 아이디어 품질 평가에 더 적합하다는 것을 알았습니다. 이러한 접근법은 적은 양의 데이터로도 높은 정확도를 달성할 수 있음을 입증하였습니다.



### You can remove GPT2's LayerNorm by fine-tuning (https://arxiv.org/abs/2409.13710)
- **What's New**: 이 논문에서는 GPT 스타일의 transformer 모델에서 LayerNorm(LN) 레이어를 제거할 수 있음을 보여줍니다. 기존의 LN이 가지는 기계적 해석성의 방해 요소를 해결하기 위해, 사전 훈련된 GPT2-small 모델에서 일부 데이터로 미세 조정하여 LN-free 모델을 생성했습니다.

- **Technical Details**: LN 레이어는 대규모 언어 모델의 훈련을 안정화하는 데 필수적이지만, 그 비선형적 특성으로 인해 모델의 해석이 어려워졌습니다. 본 연구에서는 500M 토큰의 훈련 데이터를 사용하여 GPT2-small 모델에서 LN 레이어를 성공적으로 제거하였고, 해당 모델의 미세 조정 절차와 Hugging Face 리포지토리를 제공합니다.

- **Performance Highlights**: LN-free 모델은 OpenWebText 및 ThePile 데이터셋에서 기존 모델과 유사한 성능을 나타내며, 교차 엔트로피 손실에서는 -0.05, Hellaswag 벤치마크에서는 -0.5% 정확도를 기록했습니다.



### Column Vocabulary Association (CVA): semantic interpretation of dataless tables (https://arxiv.org/abs/2409.13709)
- **What's New**: 이번 논문에서는 Semantic Table Interpretation (STI) 분야에서 새로운 과제인 'Metadata to KG' 트랙을 소개하며, 기존의 데이터에 접근하지 않고 메타데이터 정보만으로 테이블 해석을 수행하는 방법을 검토하였습니다.

- **Technical Details**: 주요 내용은 Column Vocabulary Association (CVA)라는 새로운 개념을 도입하였고, 다양한 Large Language Models (LLMs)과 Retrieval Augmented Generation (RAG) 접근 방식을 통해 CVA 작업을 평가하였습니다. 실험에는 상업적 GPT 모델(예: gpt-3.5-turbo-0.125, gpt-4o, gpt-4-turbo)과 오픈 소스 모델(예: llama3-80b, llama3-7b) 총 7개 모델이 포함되었습니다.

- **Performance Highlights**: 결과적으로, LLM은 일반적으로 온도 설정이 1.0 이하일 때 우수한 성과를 내었고, 특정 사례에서는 100% 정확도를 달성하였습니다. 그러나 데이터의 특성에 따라 전통적인 방법이 LLM의 성과를 초월하는 경우도 있었습니다.



### Towards Safe Multilingual Frontier AI (https://arxiv.org/abs/2409.13708)
Comments:
          23 pages; 1 figure and 10 supplementary figures

- **What's New**: 이번 연구는 효과적으로 다언어 지원을 제공하는 LLMs(대형 언어 모델)의 개발을 강조하며, 특히 다국어 jailbreak(탈옥) 공격으로부터 모델의 안전성을 확보하기 위한 정책 제안을 제시합니다. 이를 통해 AI의 언어적 포용성을 증대시키고, EU의 법적 틀에 맞춘 정책 조치를 모색합니다.

- **Technical Details**: 5개의 선도적인 LLM을 대상으로 EU의 24개 공식 언어에 대한 다국어 jailbreak 공격의 취약성을 정량적으로 분석하였습니다. 다국어 기능과 취약성 간의 관계를 평가하기 위해,  언어 자원화 수준에 대한 새로운 가설을 제안하였습니다.

- **Performance Highlights**: 이 연구에서 제안된 정책은 AI 안전성을 개선하고, 기존의 기술적 공간과 정책적 요구 간의 간극을 줄이기 위해 설계되었습니다. 특히, 저자들은 다국어 AI 개발에 대한 국가적 지원과 다국어 기능의 의무 평가를 포함한 여러 가지 정책 권고안을 제시하고, 이러한 조치들이 AI의 효과성과 안전성을 향상시킬 수 있을 것이라고 강조합니다.



### Decolonising Data Systems: Using Jyutping or Pinyin as tonal representations of Chinese names for data linkag (https://arxiv.org/abs/2409.13706)
- **What's New**: 이 논문은 건강 연구 및 정책 결정에서 데이터 연결(data linkage)의 중요성을 강조하며, 이름 로마나이제이션(name romanisation) 문제를 해결하기 위한 표준화된 시스템의 사용을 제안합니다.

- **Technical Details**: 연구는 중국어 이름을 포함한 771개의 이름을 수집하고, Jyutping, Pinyin 및 홍콩 정부 로마나이제이션 시스템(HKG-romanisation)의 유용성을 비교했습니다. 분석 결과 Jyutping과 Pinyin이 HKG-romanisation 시스템에 비해 오류가 적음을 입증했습니다.

- **Performance Highlights**: 표준화된 로마나이제이션 시스템을 사용하는 것이 중국계 이민자의 데이터 연결률 및 정확성을 향상시킬 수 있으며, 이는 보다 포괄적인 연구 데이터 개발에 기여할 것으로 기대됩니다.



### Debiasing Text Safety Classifiers through a Fairness-Aware Ensemb (https://arxiv.org/abs/2409.13705)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 안전성과 공정성을 보장하기 위한 경량의 후처리(post-processing) 방법을 제시합니다. 기존의 데이터 불균형 문제를 해결하는 새로운 메트릭과 데이터 세트를 도입하여, 편향된 모델의 문제를 완화하고자 합니다.

- **Technical Details**: 편향을 완화하기 위해, 우리는 'Fair Data Reweighting (FDW)'라는 두 단계 방법을 적용하여 교육 세트를 재조정하고, 최종적으로 안전 분류기를 개선하는 앙상블(ensemble) 모델을 구축합니다. 또한, 두 가지 새로운 메트릭인 Average Counterfactual Variance (ACV)와 Sliced Averages (SA)를 도입하여 모델의 공정성을 평가합니다.

- **Performance Highlights**: 우리가 제안한 방법은 모델의 성능에 미치는 영향이 최소한인 상태에서 반사실적 공정성을 개선하는 것으로 나타났습니다. 또한, 새로운 Open AI 데이터 세트와 사용자 지정 프롬프트를 기반으로 한 LLM 생성 데이터 세트를 마련하여, 이들 데이터는 신원을 기반으로 균형이 잡힌 특징을 가집니다.



### Entity Extraction from High-Level Corruption Schemes via Large Language Models (https://arxiv.org/abs/2409.13704)
- **What's New**: 최근 몇 년간 증가한 금융 범죄를 해결하기 위한 전용 데이터셋의 부족을 해결하기 위해 새로운 마이크로 벤치마크 데이터셋을 제안.

- **Technical Details**: 이 논문은 뉴스 기사에서 개인 및 조직을 식별하도록 설계된 새로운 마이크로 벤치마크 데이터셋을 소개하며, 이를 기반으로 다양한 저조도 파라미터의 Large Language Models (LLMs)를 활용하여 금융 범죄 관련 기사에서 개인과 조직을 식별하는 방법론을 개발하였다. 데이터셋은 JSON 형식으로 제공되며, 애매한 엔티티 언급 문제를 해결하기 위한 LLM 기반의 비모호화 방법을 포함한다.

- **Performance Highlights**: 제안된 방법은 기존의 오픈 소스 베이스라인과 비교하여 성능의 우수성을 보였으며, 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수와 같은 표준 지표를 사용하여 평가되었다.



### Shaping the Future of Endangered and Low-Resource Languages -- Our Role in the Age of LLMs: A Keynote at ECIR 2024 (https://arxiv.org/abs/2409.13702)
- **What's New**: 이 논문은 언어가 문화적 및 사회적 정체성을 형성하는 데 중요한 역할을 한다는 점을 강조하며, 오늘날 7100개 이상의 언어 중 상당수가 멸종 위기에 처해 있음을 알립니다. 특히, Occitan 언어를 중심으로 기술과 전통 간의 협력 가능성을 탐구합니다.

- **Technical Details**: Large Language Model (LLM) 기술이 제공하는 번역 및 콘텐츠 생성의 가능성을 논의하며, 이는 멸종 위기 언어 보존과 재활성화의 중요한 요소가 될 수 있습니다. 그러나 이러한 기술은 또한 문화의 동질화 및 이미 취약한 언어의 추가적 소외를 초래할 위험이 있습니다.

- **Performance Highlights**: 인공지능(AI)과 인간의 전문성이 함께 작동하여 언어의 다양성을 보존할 수 있는 희망을 제공할 수 있음을 강조하며, 이를 위해서는 윤리적 및 실용적인 도전 과제를 해결해야 한다고 주장합니다.



### CA-BERT: Leveraging Context Awareness for Enhanced Multi-Turn Chat Interaction (https://arxiv.org/abs/2409.13701)
Comments:
          This paper has been accepted by ICBASE 2024

- **What's New**: 이 논문은 기존의 BERT 모델을 기반으로 다중 대화 상호작용에서의 맥락의 필요성을 감지하는 데 특화된 Context-Aware BERT (CA-BERT) 모델을 소개합니다. 이 모델은 맥락 필요성을 효과적으로 분석하여 대화의 정확성과 관련성을 높이는 데 기여합니다.

- **Technical Details**: CA-BERT는 BERT 아키텍처를 수정하여 다중 턴 대화의 맥락 필요성 분류를 위한 맞춤형 구조를 도입했습니다. 주요 개선 사항으로는 드롭아웃 레이어와 이진 분류기를 추가하여 '맥락 필요' 또는 '맥락 불필요'를 예측하는 기능을 강화했습니다. 이를 통해 효율성이 높아지고, 훈련 데이터에서 수집한 다중 대화 샘플을 활용하여 성과를 평가했습니다.

- **Performance Highlights**: CA-BERT는 기존의 BERT 모델 대비 높은 정확도와 효율성을 보여주었으며, 훈련 시간과 자원 사용량을 획기적으로 줄였습니다. 이번 연구는 네트워크의 맥락 인지 능력을 향상시킴으로써 자동화된 대화 시스템에서 사용자 경험과 상호작용 품질을 개선하는 데 기여할 것으로 기대됩니다.



### Lightweight Transducer Based on Frame-Level Criterion (https://arxiv.org/abs/2409.13698)
Comments:
          Accepted by Interspeech 2024, code repository: this https URL

- **What's New**: 본 논문에서는 메모리와 계산 요구 사항을 크게 줄이는 동시에 비슷한 성능을 유지하는 경량화된 transducer 모델을 제안합니다. 기존 sequence-level criterion 대신 frame-level criterion를 사용하여 성능을 개선하였습니다.

- **Technical Details**: 경량화된 transducer 모델은 CTC(CTC: Connectionist Temporal Classification) 강제 정렬 알고리즘의 결과를 기반으로 각 프레임의 레이블을 결정합니다. 인코더 출력은 디코더 출력과 해당 시간에 결합되며, 결과적으로 메모리 사용량이 O(N*T*U*V)에서 O(N*T*V)로 감소합니다. 또한, 우리는 cross-entropy loss를 사용하여 모델을 훈련시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존 transducer와 유사한 정확도를 달성하였으며, 블랭크 레이블의 비율로 인한 비대칭 분류 문제를 해결함으로써 성능 차이를 극복했습니다. AISHELL-1 데이터셋을 통해 검증하였습니다.



### Prompt Baking (https://arxiv.org/abs/2409.13697)
Comments:
          25 pages, 8 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 행동을 변경하기 위해 프롬프트(Prompt)를 가중치(Weight)에 '베이킹(Baking)'하는 새로운 기술을 제안합니다. 이를 통해 요청된 프롬프트에 따라 LLM이 행동하도록 만드는 방법을 제시합니다.

- **Technical Details**: '프롬프트 베이킹'은 프롬프트 $u$와 초기 가중치 $	heta$를 새로운 가중치 셋 $	heta_u$로 변환하여, 새로운 LLM이 원래의 프롬프트된 LLM처럼 행동하도록 하는 과정입니다. 이는 KL divergence를 최소화하는 방식으로 작동하며, 프롬프트를 가중치 업데이트로 변환하여 재사용성을 높입니다.

- **Performance Highlights**: 연구 결과, 프롬프트를 베이킹함으로써 여러 벤치마크(GSM8K, ASDiv, MBPP, ARC-Easy, ARC-Challenge, CommonsenseQA)에서 제로샷(zero-shot) 성능이 개선되었습니다. 또한, 뉴스 헤드라인을 베이킹함으로써 LLM의 지식을 직접 업데이트할 수 있으며, 장기적인 시퀀스에서는 '프롬프트 망각(prompt forgetting)'을 완화할 수 있습니다. 재프롬프트와 재베이킹을 통해 성능이 더욱 향상되며, 이를 반복적으로 수행하는 '프롬프트 추적(Prompt Pursuit)' 방식을 통해 인스트럭션 따라하기 성능에서 극적인 성능 향상을 보였습니다.



### You Only Use Reactive Attention Slice For Long Context Retrieva (https://arxiv.org/abs/2409.13695)
- **What's New**: 이 논문에서는 Attention을 기반으로 한 새로운 검색 기술인 You Only Use Reactive Attention slice (YOURA)를 제안합니다. 기존의 Retrieval Augmented Generation (RAG) 기법이 긴 맥락을 처리하는 데 한계가 있던 점을 개선하여, 모델이 긴 입력 맥락을 효과적으로 활용하도록 합니다.

- **Technical Details**: YOURA는 입력 문맥에서 문장의 관련성을 평가하기 위해 reaction score라는 새로운 검색 휴리스틱을 사용합니다. 각 토큰의 Attention 점수가 쿼리에 어떻게 "반응"하는지를 측정하여 가장 반응이 큰 문장을 검색합니다. 이 과정에서 Embedding-Agnostic Sentence Yield (EASY) 알고리즘을 활용하여 각 문장을 토큰 인덱스 벡터에 매핑합니다.

- **Performance Highlights**: YOURA는 LongBench QA 데이터셋에서 최대 30% 향상된 vLLM 추론 처리량을 달성하며, 질문 응답 품질을 10% 향상시킵니다. EASY 알고리즘은 문장-토큰 인덱스 매핑 정확도를 93% 이상 기록했습니다.



### A Knowledge-Centric Benchmarking Framework and Empirical Study for Retrieval-Augmented Generation (https://arxiv.org/abs/2409.13694)
Comments:
          14 pages, 11 figures; Mingyue Cheng is the corresponding author

- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 새로운 벤치마크를 제안하고, KDD Cup 2024 CRAG 대회의 데이터셋을 활용하여 RAG의 성능을 분석합니다. 여기서 HTML 형식의 웹 페이지를 Markdown 형식으로 변환하여 LLM이 정보를 효율적으로 활용할 수 있도록 하였습니다.

- **Technical Details**: RAG 모델은 내부 LLM 지식과 외부 지식 소스의 효과적인 융합을 목표로 합니다. 연구에서는 RAG의 전 과정(지식 소스 선택, 검색, 정리 및 추론)을 심도 있게 분석하고, 자동화된 지식 소스 선택 및 노이즈 청크의 영향 등을 조사했습니다. 또한, 하이퍼파라미터 설정에 따른 성능 변화를 분석하였습니다.

- **Performance Highlights**: RAG-X 프레임워크는 CRAG 기본 모델 및 LLM 전용 모델보다 일관되게 우수한 성과를 보였으며, 구조화된 Mock API의 데이터는 비구조화된 웹 출처에 비해 정확도를 향상시키고 환각(hallucination)률을 감소시켰습니다. 그러나 외부 지식 소스의 입력을 늘릴수록 정확도가 개선되지만 환각률도 소폭 증가하는 결과를 보였습니다.



### Archon: An Architecture Search Framework for Inference-Time Techniques (https://arxiv.org/abs/2409.15254)
- **What's New**: 최근의 조사에 따르면, Archon이라는 자동화된 프레임워크가 LLM과 추론 시간 기술들을 결합하여 성능을 향상시키는 데 효과적임을 입증하였습니다. Archon은 다양한 추론 시간 아키텍처를 설계하는 데 유용하며, 하이퍼파라미터 최적화를 통해 최적의 아키텍처를 도출합니다.

- **Technical Details**: Archon 프레임워크는 generation ensembling, multi-sampling, ranking, fusion, critiquing, verification, 및 unit testing과 같은 방법을 포함한 확장 가능한 디자인 공간을 정의합니다. 자동화된 Inference-Time Architecture Search (ITAS) 알고리즘을 통해 LLM과 추론 컴퓨팅 예산에 따라 최적화된 아키텍처를 출력합니다. 또한, Bayesian optimization을 사용하여 하이퍼파라미터 공간을 효과적이고 효율적으로 검색합니다.

- **Performance Highlights**: Archon 아키텍처는 MT-Bench, Arena-Hard-Auto, AlpacaEval 2.0, MixEval, MixEval Hard, MATH, 및 CodeContests와 같은 다양한 벤치마크에서 우수한 성능을 보였습니다. Archon이 설계한 아키텍처는 GPT-4o 및 Claude 3.5 Sonnet 모델보다 평균 14.1 포인트, 오픈 소스 모델과의 비교에서 평균 10.3 포인트의 성능 향상을 기록했습니다.



### Efficiently Dispatching Flash Attention For Partially Filled Attention Masks (https://arxiv.org/abs/2409.15097)
- **What's New**: 새로운 알고리즘 'Binary Block Masking (BinBlkMsk)'을 소개하며, 이 알고리즘은 기존의 Flash Attention을 개선하여 모든 종류의 attention mask를 지원합니다. 특히, 사용자 친화적이며 특정 마스크에 대한 사용자 조정 없이 사용할 수 있습니다.

- **Technical Details**: 이 방법은 attention 행렬의 관련 블록만을 처리하여 비효율적인 계산을 줄입니다. Binary Block Matrix (BinBlkMat)를 사용해 비영 상태의 항목만 선택적으로 처리하고, 긴성폭 넓이 기반으로 최적화된 방식을 도입하여 성능을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 기존 Flash Attention에 비해 최대 9배의 런타임 성능 개선을 실현했습니다. 이는 실제 시나리오에서 얻은 attention mask를 기반으로 한 실험을 통해 검증되었습니다.



### Evaluating the Usability of LLMs in Threat Intelligence Enrichmen (https://arxiv.org/abs/2409.15072)
- **What's New**: 이번 연구는 사이버 위협 정보(Cyber Threat Intelligence, CTI) 분야에서 다섯 개의 대규모 언어 모델(Large Language Models, LLM)인 ChatGPT, Gemini, Cohere, Copilot, Meta AI의 사용성을 포괄적으로 평가하였습니다. 이를 통해 보안 전문가들이 LLM을 효과적으로 활용할 수 있도록 사용자 인터페이스 및 기능 개선을 위한 구체적인 권장 사항을 제시합니다.

- **Technical Details**: 연구에서는 휴리스틱 워크스루(heuristic walkthrough) 및 사용자 연구(user study) 방법론을 이용하여 LLM의 사용자 인터페이스 설계, 오류 처리, 학습 곡선(learning curve), 성능, 기존 도구와의 통합 등을 평가했습니다. 이는 위협 데이터의 수집, 전처리, 분석 자동화를 지원하는 LLM의 다양한 기능을 탐구하는데 초점을 맞추었습니다.

- **Performance Highlights**: 결과적으로, 연구에서는 LLM 사용 시 발생할 수 있는 주요 사용성 문제들을 식별하고, 각 LLM의 향후 개선을 위한 실행 가능한 권장 사항을 제공합니다. 사용성이 향상되면 보안 전문가들이 이 도구들을 보다 효과적으로 활용하여 사이버 위협에 대응할 수 있는 기반이 마련됩니다.



### Can CLIP Count Stars? An Empirical Study on Quantity Bias in CLIP (https://arxiv.org/abs/2409.15035)
Comments:
          Short paper. Accepted by the Findings of EMNLP 2024

- **What's New**: CLIP 모델에서 수량 편향(quantity bias)을 조사하여 이미지 생성 작업에서 사용자 의도를 잘 이해하지 못하고 실제 출력과 요구되는 객체 수의 불일치를 보여줌.

- **Technical Details**: 이 연구에서는 텍스트, 이미지 및 교차 모달(cross-modal) 관점에서의 수량 편향을 다루며, 9개의 다양한 CLIP 모델을 평가하고, 수량 관련 명사를 포함하는 수작업 데이터셋을 제작하여 CLIP의 수량 이해력을 평가함. CLIP은 'fewer'와 'more'를 비교하는 데 효과적이지 않으며, 이미지 도메인에서는 서로 다른 원의 수를 가진 이미지 간에도 큰 차이를 구별하지 못함.

- **Performance Highlights**: 실험 결과, CLIP은 텍스트와 이미지의 수량 개념을 효과적으로 이해하지 못하며, 수량 단어 간의 유사성을 잘 구분하지 못하고, 이는 다운스트림(downstream) 작업의 신뢰성을 저하시킴.



### ViBERTgrid BiLSTM-CRF: Multimodal Key Information Extraction from Unstructured Financial Documents (https://arxiv.org/abs/2409.15004)
Comments:
          Accepted in MIDAS (The 8th Workshop on MIning DAta for financial applicationS) workshop of ECML PKDD 2023 conference

- **What's New**: 이 논문은 비정형 문서에서의 핵심 정보 추출(Information Extraction, KIE) 모델에 대한 새로운 접근 방식을 제안합니다. 특히, ViBERTgrid라는 다중 모드 트랜스포머를 비정형 금융 문서에 적응시키고 BiLSTM-CRF 레이어를 통합하여 성능을 크게 향상시킵니다.

- **Technical Details**: 제안된 ViBERTgrid BiLSTM-CRF 모델은 비정형 문서에서의 개체 인식(Named Entity Recognition, NER) 성능을 2%포인트 향상시키면서도 반구조 문서에서의 KIE 성능을 유지합니다. 이 모델은 두 가지 주요 아키텍처인 ViBERTgrid(트랜스포머 기반)와 BiLSTM-CRF(시퀀스 기반)를 결합하여 구문 및 장기 컨텍스트 인식을 제공합니다.

- **Performance Highlights**: 이 모델은 비정형 자금 이체 주문 데이터셋 및 반구조 영수증 데이터셋(SROIE)에서 평가되었으며, SROIE 데이터셋에 대한 토큰 수준 주석을 공개하여 다중 모드 시퀀스 레이블링 모델에서의 사용 가능성을 높였습니다.



### FineCops-Ref: A new Dataset and Task for Fine-Grained Compositional Referring Expression Comprehension (https://arxiv.org/abs/2409.14750)
Comments:
          19 pages, EMNLP 2024

- **What's New**: FineCops-Ref라는 새로운 REC 데이터셋을 제안했습니다. 이 데이터셋은 접근 가능한 난이도를 제어할 수 있으며, 기존 데이터셋에서는 간과되었던 부정적 샘플에 대한 모델의 저항력을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: FineCops-Ref 데이터셋은 객체 카테고리, 속성 및 다단계 관계에 대한 세밀한 추론을 요구합니다. 난이도 수준은 대상 객체 위치 파악에 필요한 속성과 관계의 수에 따라 분류됩니다. 샘플에는 부정적인 텍스트와 이미지가 포함되어 있어, 모델의 시각적 기초 능력을 직접 평가할 수 있습니다.

- **Performance Highlights**: 상태 최상 모델들 및 MLLMs을 종합적으로 평가한 결과, grounding 성능의 상당한 차이를 발견했습니다. 간단한 REC 작업에서는 전통적 비전-언어 모델이 우수한 성능을 보였고, 더 높은 난이도에서는 MLLMs가 더 나은 성과를 나타냈습니다. 이는 모델의 미세 조정을 통해 성능이 향상되었음을 보여줍니다.



### VLEU: a Method for Automatic Evaluation for Generalizability of Text-to-Image Models (https://arxiv.org/abs/2409.14704)
Comments:
          accepted by EMNLP2024(long paper,main conference)

- **What's New**: 텍스트-이미지(T2I) 모델의 평가 방법을 개선하기 위해 새로운 평가 지표인 VLEU(Visual Language Evaluation Understudy)를 소개합니다. 이 지표는 대규모 언어 모델을 사용하여 T2I 모델의 다양한 텍스트 프롬프트에 대한 일반화 능력을 정량적으로 평가할 수 있습니다.

- **Technical Details**: VLEU는 시각적 텍스트 도메인의 분포와 T2I 모델이 생성한 이미지의 조건부 분포 간의 Kullback-Leibler divergence를 계산하여 모델의 일반화 능력을 수치적으로 평가합니다. 이 지표는 다양한 텍스트 프롬프트에서 이미지의 생성 품질과 그 일치도를 평가하는 데 활용됩니다. LLM(대규모 언어 모델)과 CLIP 모델을 사용하여 텍스트와 이미지 간의 의미적 일치를 평가합니다.

- **Performance Highlights**: VLEU의 실험을 통해 다양한 T2I 모델의 일반화 능력을 효과적으로 평가할 수 있음을 입증하였습니다. 이 새로운 지표는 T2I 모델 개발에 필수적인 도구로 자리잡을 것으로 기대되며, 실제 사례 연구 또한 발표되어 그 유용성을 보여주었습니다.



### MemeCLIP: Leveraging CLIP Representations for Multimodal Meme Classification (https://arxiv.org/abs/2409.14703)
Comments:
          Accepted to EMNLP 2024 (Main)

- **What's New**: 본 연구는 LGBTQ+ 프라이드 운동과 관련된 5,063개의 텍스트 내장 이미지로 구성된 새로운 데이터셋 PrideMM을 소개합니다. 이전의 연구들이 단일 측면에 집착했던 것과 달리, 우리는 혐오 발언, 표적 탐지, 입장 분류 및 유머 탐지의 여러 측면을 포함하는 종합적인 데이터셋을 구축했습니다.

- **Technical Details**: PrideMM은 4개의 과제를 포함하고 있습니다: (A) 혐오 발언 탐지, (B) 혐오 발언의 대상 분류, (C) 주제 입장 분류, (D) 의도된 유머 탐지. 우리는 CLIP(Contrastive Language-Image Pre-Training) 모델의 지식을 활용하여 MemeCLIP이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 경량 Feature Adapters를 활용해 CLIP의 기존 지식을 보존하고, 데이터셋의 불균형을 고려한 코사인 분류기를 통합하여 더욱 견고한 성능을 발휘합니다.

- **Performance Highlights**: MemeCLIP은 두 개의 실제 데이터셋에 대한 실험에서 이전에 제안된 프레임워크보다 우수한 성능을 보여주었으며, 혐오 분류 작업에서는 zero-shot GPT-4와 성능을 비교했습니다. 최종적으로, 모델의 단점도 정 qualitatively 분석하여 문제점을 도출하였습니다.



### Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling (https://arxiv.org/abs/2409.14683)
- **What's New**: 이번 논문에서는 ColBERT와 같은 다중 벡터 검색 방법의 저장소 및 메모리 요구사항을 줄이기 위한 간단한 클러스터링 기반의 Token Pooling 방법을 도입했습니다. 이 방법은 저장해야 하는 벡터의 수를 획기적으로 줄여줍니다.

- **Technical Details**: Token Pooling 방법은 개별 벡터를 클러스터링하고 평균 풀링을 통해 하나의 벡터로 변환하는 2단계 시스템으로 작동합니다. 세 가지 풀링 방법이 제안되었으며, 특히 계층적 클러스터링이 가장 우수한 결과를 보여주었습니다.

- **Performance Highlights**: 이 방법은 ColBERT 인덱스를 평균적으로 50%의 저장 공간을 줄이면서도 성능 저하가 없음을 보여주었으며, 66%까지의 벡터 수 감소도 가능하였고, 이 경우 성능 저하는 3% 이하에 머물렀습니다.



### RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning (https://arxiv.org/abs/2409.14674)
Comments:
          Project Website: this https URL

- **What's New**: 본 논문은 로봇 조작을 위한 강력하고 교정 가능한 비주얼-모터(Visuomotor) 정책 개발의 어려움을 다룹니다. 실패 복구 메커니즘과 간단한 언어 지시의 한계를 극복하기 위해, 자동으로 전문가 시연을 실패 복구 궤적(failure recovery trajectories)과 세부 언어 주석으로 보강하는 데이터 생성 파이프라인을 제안합니다.

- **Technical Details**: 우리는 Rich languAge-guided failure reCovERy (RACER)라는 감독-행위자(supervisor-actor) 프레임워크를 소개하며, 이는 실패 복구 데이터를 풍부한 언어 설명과 결합하여 로봇 제어를 향상시킵니다. RACER는 온라인 감독으로 작동하는 비전-언어 모델(Vision-Language Model, VLM)과 다음 행동을 예측하는 언어 조건 비주얼-모터 정책을 포함합니다.

- **Performance Highlights**: 실험 결과, RACER는 RLbench의 다양한 평가 설정에서 기존 최첨단 모델인 Robotic View Transformer (RVT)를 초월하여 우수한 성능을 보여주었습니다. 이는 시뮬레이션 및 실제 환경 모두에서 탁월한 Robustness와 Adaptability를 입증합니다.



### Backtracking Improves Generation Safety (https://arxiv.org/abs/2409.14586)
- **What's New**: 이번 논문에서는 언어 모델 안전성을 위한 새로운 접근법인 'backtracking' 기법을 제안합니다. 이는 언어 모델이 올바르지 않은 생성 결과를 되돌리고 새로운 안전한 응답을 생성할 수 있도록 허용하는 기술입니다.

- **Technical Details**: 'backtracking'은 언어 모델이 생성 중에 특별한 [RESET] 토큰을 사용하여 이전의 안전하지 않은 생성 결과를 식별하고 이를 잊어버리면서 새로운 생성 작업을 시작하는 방식입니다. 본 연구는 SFT(Supervised Fine Tuning)와 DPO(Direct Preference Optimization) 방법론을 통해 훈련되었으며, 이를 통해 Gemma-2-2B와 Llama-3-8B 모델의 안전성을 크게 향상시켰습니다.

- **Performance Highlights**: Backtracking을 사용한 Llama-3-8B 모델은 기준 모델에 비해 안전성이 4배 증가했으며(6.1%에서 1.5%로), 유용성의 감소 없이도 이러한 안전성 향상이 이루어졌습니다. 추가로, 네 가지 적대적 공격에 대한 보호 기능도 제공되었습니다.



### What Are They Doing? Joint Audio-Speech Co-Reasoning (https://arxiv.org/abs/2409.14526)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 논문에서는 오디오와 음성을 동시에 처리할 수 있는 새로운 Joint Audio-Speech Co-Reasoning (JASCO) 태스크를 도입합니다. 이를 통해 오디오 및 음성 처리의 통합과 공동 추론을 필요로 하는 타 부문과의 비교를 범위로 합니다.

- **Technical Details**: JASCO는 오디오 클립과 텍스트 지시문을 입력으로 받아, 오디오 정보와 음성 정보를 결합하여 합리적인 응답을 생성하는 방식으로 설계되었습니다. 또한, 두 가지 정보의 의존성 평가를 통해 모델의 공동 추론 능력을 측정합니다.

- **Performance Highlights**: 제공된 데이터셋 'What Are They Doing'을 통해 여러 Auditory Large Language Models (ALLMs)의 공동 추론 능력을 벤치마크하여 평가하였습니다. 이 평가 방식은 모델이 특정 모달리티에 대한 의존성을 드러내는지를 측정합니다.



### Beyond Words: Evaluating Large Language Models in Transportation Planning (https://arxiv.org/abs/2409.14516)
- **What's New**: 2023년 Generative Artificial Intelligence (GenAI)의 급속한 발전이 도시 교통 및 물류 분야에 혁신적인 변화를 가져왔습니다. 본 연구는 GPT-4와 Phi-3-mini 같은 Large Language Models (LLMs)의 성능을 수송 계획에 적용하는 것을 탐구합니다.

- **Technical Details**: 이 연구는 교통 정보를 반영한 평가 프레임워크를 통해 LLM의 성능과 공간 이해력을 평가합니다. 평가 요소로는 일반적인 지리적 정보 시스템 (GIS) 기술, 교통 관련 도메인 지식 및 현실 세계의 교통 문제 해결 능력이 포함됩니다. 혼합 방법론을 활용하여 연구가 진행되었습니다.

- **Performance Highlights**: 연구 결과, GPT-4는 다양한 GIS 및 교통 관련 작업에서 Phi-3-mini보다 더 뛰어난 정확성과 신뢰성을 보였습니다. 그러나 Phi-3-mini는 특정 분석 시나리오에서 유용함을 나타내어 자원이 제한된 환경에서도 활용 가능성을 보여줍니다. 이 결과는 GenAI 기술이 도시 교통 계획에 미치는 혁신적인 잠재력을 강조합니다.



### A Large Language Model and Denoising Diffusion Framework for Targeted Design of Microstructures with Commands in Natural Languag (https://arxiv.org/abs/2409.14473)
Comments:
          29 pages, 15 figures

- **What's New**: 이번 연구에서는 자연어 처리(Natural Language Processing, NLP), 대규모 언어 모델(Large Language Models, LLMs), 및 Denoising Diffusion Probabilistic Models (DDPMs)를 통합하여 자연어 명령을 통한 미세구조 설계를 가능하게 하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 프레임워크는 두 가지 주요 구성 요소로 나뉘어 있으며, NLP 구성 요소와 미세구조 생성 구성 요소로 이루어집니다. NLP 구성 요소는 사전 훈련된 LLM을 활용하여 텍스트 설명자 데이터베이스를 생성하고, 사용자 제시 자연어 입력에서 관련 미세구조 설명자를 추출하는 재훈련된 Named Entity Recognition (NER) 모델을 사용합니다. 이후 DDPM을 이용해 특정 기계적 속성과 형태적 특성을 가진 미세구조를 생성합니다.

- **Performance Highlights**: 이 프레임워크는 비선형 하이퍼엘라스틱 미세구조 데이터베이스에서 시연되었으며, 직관적인 자연어 명령으로부터 접근 가능한 역설계를 위한 프로토타입으로 기능합니다. 이를 통해 고객의 입력에 적합한 미세구조 샘플을 효율적으로 생성하고, 광범위한 응용 분야에 대한 적용 가능성을 확대하는 데 기여할 것입니다.



### Opinion Mining on Offshore Wind Energy for Environmental Engineering (https://arxiv.org/abs/2409.14292)
- **What's New**: 이 논문에서는 소셜 미디어 데이터를 활용하여 해상 풍력 에너지에 대한 대중의 의견을 분석합니다. 세 가지 머신러닝 모델, 즉 TextBlob, VADER, SentiWordNet을 사용하여 각 모델이 제공하는 다양한 기능을 활용합니다.

- **Technical Details**: TextBlob은 주관성 분석(subjectivity analysis)과 극성 분류(polarity classification)를 제공하며, VADER는 누적 감정 점수(cumulative sentiment scores)를 산출합니다. SentiWordNet은 맥락(context)을 기준으로 감정을 고려하여 분류를 수행합니다. 자연어 처리(NLP) 기술을 통해 소셜 미디어의 텍스트 데이터에서 의미를 추출합니다.

- **Performance Highlights**: 데이터 시각화 도구를 적절히 사용하여 전체 결과를 표시하며, 이는 시민 과학(citizen science)과 스마트 거버넌스(smart governance)에 부합하여 대중의 의견이 의사 결정 지원(decision support)을 안내하는 역할을 합니다.



### Can-Do! A Dataset and Neuro-Symbolic Grounded Framework for Embodied Planning with Large Multimodal Models (https://arxiv.org/abs/2409.14277)
- **What's New**: 이 논문은 Can-Do라는 새로운 벤치마크 데이터 세트를 도입하여 대형 다중 모달 모델의 체화된 계획 능력을 평가합니다. 이 데이터 세트는 이전의 데이터 세트보다 더 다양한 복잡한 시나리오를 포함하고 있으며, 400개의 다중 모달 샘플로 구성되어 자연어 사용자 지침, 환경을 묘사하는 시각 이미지, 상태 변화 및 해당 동작 계획을 포함하고 있습니다.

- **Technical Details**: Can-Do 데이터 세트는 실제 환경을 묘사하기 위해 실제 장면 이미지와 합성 이미지를 모두 활용합니다. 세 가지 태스크 카테고리(물리적 이해, 상식, 안전)을 중심으로 설계되었으며, 각 샘플은 사용자 의도를 기반으로 시각 시나리오를 인식하고 단계를 생성하는 모델의 능력을 평가합니다. 연구에서는 또한 NeuroGround라는 신경 상징적 프레임워크를 제안하여 모델 생생 생성 과정이 환경의 초기 및 목표 상태에 명확하게 기반하도록 합니다.

- **Performance Highlights**: 실험 결과, NeuroGround 프레임워크는 기존의 강력한 기준선과 비교하여 상당한 이점을 보여주었습니다. 특히, 체화된 계획에서 기존 모델(GPT-4V 포함)의 병목 현상인 시각적 지각, 이해 및 추론 능력에서 개선된 성능을 입증했습니다.



### On Lexical Invariance on Multisets and Graphs (https://arxiv.org/abs/2409.14179)
- **What's New**: 이번 논문에서는 lexical invariance라는 새로운 문제를 다룹니다. 이는 입력의 특정 단어 기반 표현과 관계없이 문장의 의미가 변하지 않아야 한다는 것입니다. 예를 들어, '영화는 극도로 재미있었다'는 '영화는 매우 즐거웠다'와 동일한 의미를 가집니다.

- **Technical Details**: 우리는 multisets 및 그래프에 대해 가장 유표한 lexical invariant (lexical invariant) 함수에 대한 충분조건과 필요조건을 연구했습니다. multisets의 경우, 함수는 원래 multiset의 고유 요소의 개수(multiset of counts)만을 입력으로 받아야 합니다. 그래프의 경우, 함수는 adjacency matrix와 차이 행렬(difference matrix)을 입력으로 받아야 합니다.

- **Performance Highlights**: TU 데이터셋에 대한 합성 실험(synthetic experiments)을 통해 우리의 정리를 검증했습니다. 이러한 연구는 데이터 익명화(anonymization)와 관련하여 실제 애플리케이션에 이바지 할 수 있습니다.



### Will Large Language Models be a Panacea to Autonomous Driving? (https://arxiv.org/abs/2409.14165)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 자율주행(AD) 시스템에서 어떻게 활용될 수 있는지를 분석하며, LLM의 최적화 전략을 모듈화 및 엔드 투 엔드 접근법에서 탐색합니다.

- **Technical Details**: 자율주행 기술은 모듈화(modularization)와 엔드 투 엔드(end-to-end)로 나뉘며, 모듈화는 주행 작업을 감지(perception), 예측(prediction), 계획(planning), 제어(control) 모듈로 분해하여 각기 따로 훈련합니다. 반면, 엔드 투 엔드는 센서 데이터에서 제어 신호로 직접 매핑하는 단일 모델을 사용합니다. 두 접근법 모두 훈련 목표의 일관성 부족 문제와 복잡한 도시 교통 상황에서의 예측 불가능한 사건 처리에서 어려움을 겪고 있습니다.

- **Performance Highlights**: LLM들은 강력한 추론 능력과 방대한 지식을 바탕으로 AD 시스템의 이해도 및 의사결정 능력을 향상시킬 수 있는 잠재력을 지니고 있습니다. 하지만 LLM 기반 인공지능이 고급 AD 구현의 열쇠가 될 수 있을지, 그리고 AD 기술 개발을 촉진하는 데 있어서 LLM이 직면할 잠재적 한계와 도전과제를 논의합니다.



### PromptTA: Prompt-driven Text Adapter for Source-free Domain Generalization (https://arxiv.org/abs/2409.14163)
- **What's New**: 이 논문에서는 소스 도메인 데이터에 접근하지 않고도 미지의 타겟 도메인에 적응할 수 있도록 설계된 Prompt-Driven Text Adapter (PromptTA) 방법을 제안합니다. 이는 스타일 특징의 분포를 더 잘 포착하고 도메인 지식의 충분한 범위를 보장하기 위해 재샘플링을 활용합니다.

- **Technical Details**: PromptTA는 다양한 스타일 특징에서 정보를 학습하는 텍스트 기반 어댑터를 도입하여 도메인 정보를 저장합니다. 스타일 특징의 재샘플링 기법을 통해 포괄적인 도메인 지식을 효과적으로 담을 수 있도록 합니다. 이 방법은 CLIP와 같은 비전-언어 모델의 정렬된 이미지-텍스트 표현을 활용하며, 스타일 벡터의 학습 가능한 집합을 통해 다양한 도메인 정보를 표현합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터세트에서 실시된 실험을 통해 PromptTA가 최신 기술 수준의 성능을 달성한 것으로 나타났습니다. 이를 통해 SFDG 분야에서의 발전에 기여하고 있음을 보여줍니다.



### OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching (https://arxiv.org/abs/2409.14038)
Comments:
          4 pages, 1 figure

- **What's New**: 이번 연구에서는 LLM(hallucinations)의 발생이 Ontology Matching(OM) 작업에서 중요한 문제임을 제기하고, 이를 해결하기 위한 OAEI-LLM 데이터셋을 제안합니다. 이 데이터셋은 OM 작업에서 LLM의 환각 현상을 평가하기 위한 기준을 제공합니다.

- **Technical Details**: OAEI-LLM 데이터셋은 기존 OAEI 데이터셋의 확장으로, LLM의 OM 작업에서의 환각 유형을 분류하고 이를 바탕으로 LLM의 정답률을 평가합니다. 새로운 스키마 확장을 통해 LLM이 생성한 결과와 인간이 라벨링한 결과를 비교하고, 환각의 발생 정도를 측정합니다.

- **Performance Highlights**: LLM은 OM 작업에서 높은 성능을 보일 수 있지만, 환각 현상으로 인해 낮은 precision(정밀도) 및 recall(재현율) 문제를 초래할 수 있습니다. OAEI-LLM 데이터셋을 통해 LLM의 환각 현상에 대한 이해를 높이고, 향후 OM 개선 연구에 기여할 것으로 예상됩니다.



### On-device Collaborative Language Modeling via a Mixture of Generalists and Specialists (https://arxiv.org/abs/2409.13931)
- **What's New**: 본 연구에서는 Mixture of Experts (MoE) 아키텍처와 Low-Rank Adaptation (LoRA) 모듈을 활용하여 대형 언어 모델(LLMs)의 온디바이스(On-device) 협업 파인튜닝(collaborative fine-tuning)을 발표합니다. 구체적으로, 전문가의 역할을 일반화(generalists)와 전문화(specialists)로 다양화하는 CoMiGS(Collaborative Mixture of Generalists and Specialists) 접근 방식을 제안합니다.

- **Technical Details**: 우리의 연구에서 중심이 되는 것은 학습 가능한 라우팅 네트워크(routing network)로, 이는 토큰 수준에서 라우팅을 수행하여 협업과 개인화를 미세하게 조정합니다. MoE 아키텍처는 사용자가 다양한 수의 LoRA 모듈을 가질 수 있도록 하여 시스템의 이질성(system heterogeneity)을 해결합니다.

- **Performance Highlights**: 우리의 방법은 높은 데이터 이질성(data heterogeneity)을 가진 다양 한 데이터셋에서 일관되게 우수한 성능을 보입니다. 이를 통해 자원이 적은 사용자도 데이터 양이 많은 사용자에게서 혜택을 받을 수 있음을 보여줍니다.



### Eliciting Instruction-tuned Code Language Models' Capabilities to Utilize Auxiliary Function for Code Generation (https://arxiv.org/abs/2409.13928)
Comments:
          EMNLP 2024 Findings Short

- **What's New**: 본 논문에서는 코드를 생성하기 위해 instruction-tuned 모델(명령 조정 모델)이 보조 함수(auxiliary function)를 효과적으로 활용하는 방법을 탐구합니다. 기존 모델들은 보조 함수를 텍스트 프롬프트에 포함시키는 방식에 한계가 있었으나, 새로운 프롬프트 구조를 통해 성능을 개선했습니다.

- **Technical Details**: 연구자는 보조 함수 정보를 쿼리에 추가하거나 응답_prefix(구성 요소)를 제공하여 instruction-tuned 모델의 코드 생성 능력을 증진시키기 위해 여러 가지 프롬프트를 설계했습니다. 이러한 접근법은 모델이 실제로 코드를 이해하고 보조 함수를 활용하는 데 도움이 됩니다. 실험에서 사용된 모델은 최근의 경쟁력 있는 instruction-tuned 모델들로, Humanextension 벤치마크를 통해 성능 평가를 실시했습니다.

- **Performance Highlights**: 제안된 프롬프트 방식은 gpt-4o와 같은 강력한 상용 모델에 비해 오픈소스 모델의 성능을 초과하는 결과를 보여주었습니다. 특히, 보조 함수와 함께 제공된 쿼리 및 응답 구조에서 파생된 개선 성과가 두드러졌습니다. 결과적으로 instruction-tuned 모델은 일반적으로 기본 모델보다 우수한 성과를 기록했습니다.



### Generative AI Carries Non-Democratic Biases and Stereotypes: Representation of Women, Black Individuals, Age Groups, and People with Disability in AI-Generated Images across Occupations (https://arxiv.org/abs/2409.13869)
- **What's New**: AI 거버넌스(AI governance)와 AI 개발에서의 윤리(Ethics)가 중대한 문제로 떠오르며, 기술 기업, 정부, 연구자들 사이에서 AI가 우리의 민주주의에 미치는 잠재적 위험에 대한 활발한 논의가 이루어지고 있습니다. 이 논문은 생성적 AI(Generative AI)가 평등이 필요한 집단들을 어떻게 포함하거나 배제하는지를 조명합니다.

- **Technical Details**: 연구 결과는 생성적 AI가 성별(Gender), 인종(Race), 나이(Age), 그리고 가시적 장애(Visible Disability)에 관해 균등하게 포함되지 않음을 보여줍니다. 이는 AI 모델이 특정 집단에 대해 편향된 데이터를 학습함으로써 공정성을 결여하고 있음을 시사합니다.

- **Performance Highlights**: 이 연구의 주요 발견은 생성적 AI의 출력이 평등성(Equity)에 대한 고민 없이 설계되었음을 드러내며, 이는 AI 시스템의 설계와 데이터 수집이 더 포괄적이고 공정해야 함을 강조합니다.



### GTSinger: A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks (https://arxiv.org/abs/2409.13832)
Comments:
          under processing

- **What's New**: 새로운 GTSinger 데이터셋은 고품질의 다국적, 다기술적 노래 코퍼스로, 80.59시간의 노래 음성과 사실적인 음악 악보를 포함하고 있어 기존의 부족했던 노래 데이터셋의 한계를 극복합니다.

- **Technical Details**: GTSinger는 20명의 전문 가수가 참여하여 9개 언어로 다양한 음색과 스타일을 제공하며, 6가지 일반적인 노래 기법(혼합 음성, 플랫, 숨소리, 인두음, 비브라토, 글리산도)에 대한 조절 및 음소 수준 주석을 제공합니다. 데이터셋은 CC BY-NC-SA 4.0 라이센스로 사용 가능합니다.

- **Performance Highlights**: GTSinger의 사용 가능성과 품질을 검증하기 위해 기법 제어 노래 음성 합성, 기술 인식, 스타일 전이, 음성-노래 전환 등 4가지 벤치마크 실험이 수행되었습니다. 이를 통해 다양한 노래 임무에서 우수한 성능을 입증했습니다.



### Synergistic Simulations: Multi-Agent Problem Solving with Large Language Models (https://arxiv.org/abs/2409.13753)
Comments:
          15 pages, 5 figures, published in the MICS 2024 conference

- **What's New**: 이 논문은 Large Language Models (LLMs)을 활용하여 다중 에이전트 시스템을 개발하는 방법을 제시하고 있습니다. 특히, 시뮬레이션 환경에서 에이전트 간의 상호작용을 통합하여 인간 그룹의 문제 해결능력을 모델링하고자 합니다.

- **Technical Details**: 두 가지 시뮬레이션을 구현했습니다: 첫 번째는 두 명의 룸메이트가 있는 물리적 스튜디오 아파트 시뮬레이션, 두 번째는 에이전트들이 프로그래밍 과제를 협력하여 완수하는 시뮬레이션입니다. 이 논문에서는 멀티-에이전트 프레임워크에 대해 논의하고 각 시뮬레이션에서 에이전트의 성능을 분석합니다.

- **Performance Highlights**: 이 연구는 LLM이 인간 협력의 시너지를 어떻게 나타내는지를 보여주려 하며, 미래에 LLM의 응용 가능성을 높이는 데 기여할 수 있는 방향성을 모색합니다.



### VisScience: An Extensive Benchmark for Evaluating K12 Educational Multi-modal Scientific Reasoning (https://arxiv.org/abs/2409.13730)
Comments:
          89 pages, 70 figures

- **What's New**: 이번 논문에서는 다양한 과학 분야에서 다중 모달 대형 언어 모델(MLLMs)의 성능을 평가하기 위해 VisScience라는 새로운 벤치마크를 제시합니다. 이는 수학, 물리학, 화학 세 가지 과목을 아우르며, K12 교육을 기반으로 한 3,000개의 질문을 포함합니다.

- **Technical Details**: VisScience 벤치마크는 초등학교부터 고등학교까지의 21개 주제를 포함하여 각 과목마다 1,000개의 질문을 포함하고 있으며, 질문은 5개의 난이도 수준으로 구분됩니다. 이 연구에서는 25개의 대표적인 MLLMs의 과학적 추론 능력을 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면, 폐쇄형(Closed-source) MLLMs가 개방형(Open-source) 모델보다 일반적으로 뛰어난 성능을 보였습니다. Claude3.5-Sonnet 모델은 수학 53.4%의 정확도를 기록하며 가장 높은 성능을 나타냈고, GPT-4o는 물리학 38.2%, Gemini-1.5-Pro는 화학 47.0%의 정확도를 기록했습니다.



### Retrieval Augmented Generation-Based Incident Resolution Recommendation System for IT Suppor (https://arxiv.org/abs/2409.13707)
Comments:
          7 pages, 3 figures, 6 tables

- **What's New**: 이 연구는 IT 지원 도메인에서의 솔루션 추천 시스템을 위해 개발된 Retrieval Augmented Generation(RAG) 시스템을 소개합니다. 특히, IBM Slate 125m 모델을 사용하여 단일-턴과 다중-턴 IT 지원 사례를 분류하는 새로운 접근법과 성능을 보고합니다.

- **Technical Details**: 시스템은 네 가지 주요 구성 요소로 이루어져 있습니다: encoder-only transformer classifier, query generation system, retriever system, 그리고 answer generator system. 데이터 수집은 약 19,000개의 실제 지원 사례를 기반으로 하며 다양한 소프트웨어 제품에서 수집되었습니다. CNN(Convolutional Neural Networks) 및 cosine similarity를 활용하여 문서를 검색하고 재랭크합니다.

- **Performance Highlights**: 연구 결과, 작은 모델들이 RAG 사건 해결 사용 사례에서 매우 큰 모델들과 동등하거나 더 나은 성능을 보여주었다고 보고합니다. 최종적으로 F1 점수가 0.65에 이르고, 클래스 분류 정확도가 0.54, 재현율이 0.80으로 나타났습니다.



### Declarative Integration and Management of Large Language Models through Finite Automata: Application to Automation, Communication, and Ethics (https://arxiv.org/abs/2409.13693)
Comments:
          Submitted to IAAI-2025, Philadelphia, PA

- **What's New**: 이 논문에서는 공유 히스토리(shared histories)와 트리거(triggers)를 사용하여 주어진 작업에 가장 적합한 대형 언어 모델(Large Language Models, LLMs)을 선언적으로 결합할 수 있는 혁신적인 아키텍처를 제안합니다.

- **Technical Details**: 이 접근 방식은 유한 오토마타(finite automata)와 이벤트 관리 시스템(event management system)을 기반으로 하며, 프로그래밍 노력을 최소화하면서 LLM의 복잡한 통합을 지원합니다. 특히, 긍정 심리학(positive psychology) 방법을 AI와 통합하는 데 유용합니다. 아키텍처 설계 과정은 상태 정의, 트리거 우선순위 설정, LLM의 프롬프트 작성 등의 단계를 포함합니다.

- **Performance Highlights**: 이 프레임워크는 다양한 예제를 통해 그 유연성을 입증했으며, 기차 티켓 예약 자동화, 비폭력적 의사소통(non-violent communication) 계획, LLM의 윤리적 이슈 예방과 관련된 예제를 포함합니다. 이를 통해 복잡한 멀티모달 시스템에서도 효과적인 LLM 통합이 가능함을 보여주었습니다.



### Fine Tuning Large Language Models for Medicine: The Role and Importance of Direct Preference Optimization (https://arxiv.org/abs/2409.12741)
- **What's New**: 이번 연구는 의료 분야에서의 Large Language Model (LLM) 조정(fine tuning) 방법인 Supervised Fine Tuning (SFT)과 Direct Preference Optimization (DPO)의 성능을 비교합니다. 일반적으로 사용되는 조정 기술에 대한 명확한 지침이 부족한 상황에서 이 연구가 중요한 기초 자료를 제공합니다.

- **Technical Details**: 연구에서는 의료 분야의 다섯 가지 자연어 처리(Natural Language Processing) 과업인 텍스트 데이터 분류(Classification with text data), 숫자 데이터 분류(Classification with numeric data), 임상 추론(Clinical Reasoning), 요약(Summarization), 임상 분류(Clinical Triage)를 분석했습니다. SFT는 텍스트 데이터 분류에 적합하지만 DPO는 임상 추론, 요약, 임상 분류와 같은 복잡한 작업에서 성능 향상을 보였습니다.

- **Performance Highlights**: SFT는 텍스트 데이터 분류에서 충분한 성능을 발휘하며, DPO는 더 복잡한 임상 작업에서 성능을 향상시킵니다. 이 연구는 DPO 조정 기술의 필요성을 강조하며, 현재 소프트웨어에서의 공백을 지적하여 이 기술의 광범위한 배포에 대한 필요성을 환기시킵니다.



New uploads on arXiv(cs.IR)

### Towards Enhancing Linked Data Retrieval in Conversational UIs using Large Language Models (https://arxiv.org/abs/2409.16220)
Comments:
          This paper has been accepted at the 25th International Web Information Systems Engineering Conference (WISE 2024)

- **What's New**: 이 논문은 기존 정보 시스템과 LLMs(대형 언어 모델)의 통합을 통해 Linked Data(LD) 및 RDF(Ressource Description Framework) 트리플스토어에서 데이터를 추출하고 탐색하는 방법을 탐구합니다. 특히, 모델 재훈련 없이도 더 정확한 SPARQL 쿼리를 생성할 수 있는 대화형 사용자 인터페이스(UI)의 강화를 강조합니다.

- **Technical Details**: 본 연구에서는 ForestQB라는 새로운 툴킷을 사용하여 관찰적 LD 데이터로부터 정보를 추출하고, 이 툴킷은 챗봇과 폼 기반 GUI를 통합하여 SPARQL 쿼리를 구성하고 실행합니다. 연구의 초점은 LLMs의 자연어 이해 능력을 활용하여 RDF 엔티티 추출의 정확성을 향상시키는 것입니다.

- **Performance Highlights**: 본 연구의 결과, 제안된 방법론을 통해 시스템의 표현력과 사용자 쿼리에 대한 응답 정확성이 크게 향상되었습니다. 평가 결과는 LLMs가 복잡한 데이터 환경에서 엔티티 추출 및 사용자 인터랙션을 개선시킬 수 있는 가능성을 제시하고 있습니다.



### TiM4Rec: An Efficient Sequential Recommendation Model Based on Time-Aware Structured State Space Duality Mod (https://arxiv.org/abs/2409.16182)
- **What's New**: 본 연구에서는 TiM4Rec이라는 새로운 시퀀스 추천 백본 모델을 제안하며, low-dimensional 성능 감소 문제를 해결함과 동시에 Mamba 아키텍처의 계산 효율성을 유지합니다. 이는 SSD 아키텍처의 시간 인지 향상 기법을 통해 이루어집니다.

- **Technical Details**: TiM4Rec는 Mamba 아키텍처를 기반으로 하며, 먼저 SSD 아키텍처의 성능 한계를 극복하기 위한 시간 인지 향상 방법을 도입합니다. 이 모델은 low-dimensional 환경에서의 성능을 개선하면서도, linear computational complexity를 유지합니다. TiM4Rec는 기존 Transformer 구조의 SASRec 및 Mamba4Rec 모델보다 훈련 속도와 추론 속도에서 유리합니다.

- **Performance Highlights**: TiM4Rec는 세 가지 데이터 세트를 통해 실험을 진행하였으며, low-dimensional 환경에서 SSD4Rec보다 우수한 성능을 보여주고, high-dimensional 시나리오에서 SSD 아키텍처의 장점을 유지합니다.



### Ducho meets Elliot: Large-scale Benchmarks for Multimodal Recommendation (https://arxiv.org/abs/2409.15857)
- **What's New**: 이 논문은 multimodal recommender 시스템을 위한 대규모 벤치마킹을 제공하는 첫 번째 시도로, 특히 multimodal feature extractor(다중 모달 특성 추출기)에 중점을 두고 있습니다.

- **Technical Details**: 논문에서는 (i) 다중 모달 특성 추출, (ii) 추천 작업에 적합하도록 고수준 표현 개선, (iii) 다중 모달 특성 융합, (iv) 사용자-아이템 점수 예측을 포함하는 기존의 multmodal 추천 파이프라인 과정에서, 첫 번째 단계인 다중 모달 특성 추출에 대한 탐구가 부족하다는 점을 강조합니다.

- **Performance Highlights**: Ducho와 Elliot라는 두 가지 다중 모달 특성 추출 프레임워크를 활용하여, 다양한 하이퍼 파라미터 설정하에 수행된 실험 결과가 중요 통찰력을 제공하여, 차세대 다중 모달 추천 알고리즘을 훈련 및 조정하는 데 도움이 될 수 있습니다.



### IRSC: A Zero-shot Evaluation Benchmark for Information Retrieval through Semantic Comprehension in Retrieval-Augmented Generation Scenarios (https://arxiv.org/abs/2409.15763)
- **What's New**: 이 논문은 다국어 Retrieval-Augmented Generation (RAG) 작업에서 임베딩 모델의 성능을 평가하기 위한 IRSC 벤치마크를 소개합니다. 이 벤치마크는 쿼리 검색, 제목 검색, 단락의 일부 검색, 키워드 검색, 요약 검색의 cinco 가지 검색 작업을 포함합니다.

- **Technical Details**: IRSC 벤치마크는 embedding 모델의 성능을 다양한 검색 과제에서 평가하며, 새로운 메트릭인 Semantic Comprehension Index (SSCI) 및 Retrieval Capability Contest Index (RCCI)를 도입했습니다. 이 벤치마크는 여러 언어(영어, 중국어 및 혼합 언어 데이터셋)에서 모델을 평가합니다.

- **Performance Highlights**: 연구 결과에 따르면, IRSC 벤치마크는 실용적인 RAG 작업에서의 embedding 모델의 성능을 더 잘 이해하고 개발하는 데 기여할 수 있습니다. 이 연구는 embedding 모델의 언어 간 한계를 통찰하는 데 중요한 기여를 합니다.



### Making Text Embedders Few-Shot Learners (https://arxiv.org/abs/2409.15700)
- **What's New**: 본 논문에서는 큰 언어 모델(LLM)의 인-context learning (ICL) 기능을 활용하여 텍스트 임베딩 생성 과정을 개선하는 새로운 모델 bge-en-icl을 제안합니다. 이 모델은 적은 수의 샘플을 사용하여 고품질의 텍스트 임베딩을 생성합니다.

- **Technical Details**: bge-en-icl 모델은 쿼리 프롬프트에 작업 관련 예제를 통합하여 LLM의 ICL 능력을 최대한 활용합니다. 이 연구에서는 다양한 attention 메커니즘과 pooling 방법을 평가하여 LLM을 임베딩 모델로 효과적으로 활용하는 방법을 조사했습니다.

- **Performance Highlights**: bge-en-icl 모델은 MTEB 및 AIR-Bench 벤치마크에서 새로운 최첨단(SOTA) 성능을 달성하였으며, 간단한 ICL 전략만으로도 뛰어난 성과를 거둘 수 있다는 것을 입증했습니다. 코드와 데이터셋은 자유롭게 제공됩니다.



### Cross-Domain Latent Factors Sharing via Implicit Matrix Factorization (https://arxiv.org/abs/2409.15568)
- **What's New**: 이 논문에서는 데이터 희소성 문제를 해결하기 위해 Cross-Domain Implicit Matrix Factorization (CDIMF) 모델을 제안합니다. CDIMF는 기존의 implicit matrix factorization을 확장하여 cross-domain 시나리오에서 사용할 수 있도록 합니다.

- **Technical Details**: CDIMF 모델은 Alternating Direction Method of Multipliers (ADMM)를 활용하여 사용자 간 공유 잠재 요인을 학습하면서 상호작용 행렬을 분해합니다. 이 방법은 사용자의 개인 데이터를 보호하면서 다양한 도메인에서 정보를 교환할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, CDIMF는 여러 산업 데이터셋에서 cold-start와 warm-start 상황 모두에서 경쟁력을 보였으며, 최신 cross-domain 및 single-domain 모델보다 성능이 우수한 것으로 나타났습니다.



### GLARE: Guided LexRank for Advanced Retrieval in Legal Analysis (https://arxiv.org/abs/2409.15348)
Comments:
          26 pages, 8 figures, submitted to AI and Law

- **What's New**: 본 논문에서는 브라질의 특별 항소를 분류하기 위해 새로운 방법인 GLARE를 제안합니다. 이 방법은 비지도 기계 학습(unsupervised machine learning)을 기반으로 하며, 법적 주제를 탐색하는 데 필요한 기존 데이터의 수량적 요구를 줄입니다.

- **Technical Details**: GLARE 방법은 Graph 기반의 LexRank 알고리즘을 수정하여 'Guided LexRank'를 제안하고, BM25 알고리즘을 사용해 생성한 요약과 다양한 주제 간의 유사도를 평가합니다. 이 방법은 특별 항소의 내용을 요약한 후, 해당 요약과 기존 주제 간의 유사성을 평가하여 가장 적합한 주제를 순위 매깁니다.

- **Performance Highlights**: GLARE 방법은 TRF2의 기존 방법보다 훨씬 높은 정확도를 보였습니다. 특별 항소의 적합한 주제를 약 76%의 정확도로 추천할 수 있었으며, 이는 TRF2의 약 35%와 비교됩니다. 특히 데이터가 적은 특정 주제에서 비지도 학습 방법이 더 좋은 성능을 발휘했습니다.



### Big data searching using words (https://arxiv.org/abs/2409.15346)
- **What's New**: 이 논문에서는 빅데이터에서의 언어 검색의 이웃 구조에 대한 기본 개념을 소개하고, 이를 통해 빅데이터의 중요한 위상 구조를 형성하는 방법을 제시합니다.

- **Technical Details**: 논문은 빅데이터 검색에서 Jaccard 유사도(coefficient)를 사용하여 이웃 구조를 활용한 이상 탐지(anomaly detection) 방법을 논의합니다. 또한, 데이터 검색에서 빅데이터의 원시(primal) 개념을 도입합니다.

- **Performance Highlights**: 이 연구는 빅데이터 분석에서 TDA(Topological Data Analysis)와 같은 새로운 방법론의 필요성을 강조하며, 빅데이터의 이웃 구조를 탐색하는 데 있어 현재까지 발견되지 않은 위상적 특징들을 밝힐 가능성을 제안합니다.



### Advertiser Content Understanding via LLMs for Google Ads Safety (https://arxiv.org/abs/2409.15343)
- **What's New**: 이번 연구는 Google Ads 콘텐츠 정책의 일관성을 높이기 위해 광고주의 의도를 이해하는 방법을 제안합니다. 대규모 인공지능 모델(LLMs)을 활용하여 광고주의 콘텐츠 프로필을 생성하고 이를 바탕으로 정책 위반 가능성을 판단합니다.

- **Technical Details**: 연구에서는 LLM을 이용하여 광고주 정보를 수집한 뒤, 광고주 콘텐츠 프로필을 생성하고 이를 비즈니스, 도메인 신호 등 여러 데이터를 포함하여 분석합니다. 최종적으로 LLM이 광고주의 정책 위반 여부를 판단하는데 필요한 프롬프트를 조정합니다.

- **Performance Highlights**: 최소한의 프롬프트 조정을 통해 작은 테스트 세트에서 95%의 정확도에 도달하였습니다. LLM의 성능은 훈련 없이도 뛰어난 결과를 보여주었으며, 이는 향후 모델 개선 및 추가 프롬프트 조정으로 더욱 향상될 것으로 기대됩니다.



### Recall: Empowering Multimodal Embedding for Edge Devices (https://arxiv.org/abs/2409.15342)
- **What's New**: RECALL은 리소스가 제한된 모바일 환경을 위해 최적화된 최초의 on-device 멀티모달 임베딩 시스템이다. 이 시스템은 coarse-grained embedding을 생성하고 query 기반 필터링을 활용하여 높은 처리량과 정확한 검색을 달성한다.

- **Technical Details**: RECALL 시스템은 데이터 인식 pre-exit 예측기, Progressive LoRA healing, 그리고 Speculative fine-grained retrieval의 세 가지 하드웨어-알고리즘 공동 설계를 통해 동작한다. 이 시스템은 multi-layer transformer architecture를 통해 작동하며, coarse-grained embedding을 통해 모달리티 간 검색을 수행하고, 후속 쿼리 단계에서 최종 검색을 정제한다.

- **Performance Highlights**: RECALL은 평균 14.9배 처리량 향상과 13.1배 에너지 소비 감소를 달성하였다. 이 시스템은 배터리 소모를 최소화하면서도 높은 정확도를 유지하며, 전체 MEM에 비해 5% 미만의 상대적 정확도 손실을 초래한다.



### WISDOM: An AI-powered framework for emerging research detection using weak signal analysis and advanced topic modeling (https://arxiv.org/abs/2409.15340)
Comments:
          18 pages, 7 figures

- **What's New**: 이 연구는 WISDOM이라는 자동화된 인공지능(AI) 기반 프레임워크를 소개하여 복잡한 과학적 문제를 해결하고 새로운 연구 주제를 탐지하는 방법을 제시합니다. WISDOM은 고급 topic modeling과 weak signal 분석을 결합하여, 방대한 데이터를 신속하게 처리 및 분석하고, 숨겨진 교차학문적 패턴을 발견하며, 공정한 통찰력을 제공합니다.

- **Technical Details**: WISDOM은 여러 기술 영역에 적용 가능한 다층적이고 모듈형 접근 방식을 채택합니다. 이 프레임워크는 최신 기술인 BERTopic을 사용하여 연구 주제를 식별하고 그 진화를 추적합니다. 또한, weak signal 분석을 채택하여 기존 방법으로는 쉽게 발견하지 못하는 미세한 연구 동향을 탐지합니다. WISDOM은 2004년부터 2021년까지의 수중 감지 기술에 대한 과학 논문을 분석하여 성능을 평가합니다.

- **Performance Highlights**: WISDOM은 수중 감지 기술 분야에서 emerging research 주제를 식별하는 데 있어 높은 정확도와 신뢰성을 보여줍니다. 이 연구는 2004년부터 2021년까지의 데이터를 사용하여 시간에 따른 연구 주제의 진화를 파악하고, AI 기반 접근 방식이 주제 탐지 과정에서의 객관성을 높이며, 사람의 주관적 편향을 제거하는 데 기여한다고 강조합니다.



### Revisiting the Solution of Meta KDD Cup 2024: CRAG (https://arxiv.org/abs/2409.15337)
- **What's New**: 이 논문은 Meta KDD CUP 2024의 CRAG Comprehensive RAG Benchmark Challenge에서 팀 APEX의 솔루션을 소개합니다. CRAG 벤치마크는 Retrieval-Augmented Generation (RAG) 시스템의 다양하고 동적인 문제를 평가하는 데 있어 기존 QA 벤치마크의 한계를 극복할 수 있도록 설계되었습니다.

- **Technical Details**: 본 연구에서는 질문의 다양성과 동적인 특성에 맞춘routing 기반의 도메인 및 동적 적응 RAG 파이프라인을 제안합니다. 이 방법은 정보 검색(retrieval), 증대(augmentation), 생성(generation) 세 단계에서 모두 특별한 처리를 수행하며, CRAG에서 우수한 성과를 거두어 최종 경쟁 리더보드에서 작업 2와 3에서 2위를 기록했습니다.

- **Performance Highlights**: 우리의 방법은 CRAG에서 뛰어난 성과를 발휘했으며, 특히 웹 페이지 검색 및 Mock API를 활용해 정보 선택과 통합의 능력을 강조하였습니다. 각 과제는 이전 단계를 기반으로 하여, 참가자들이 더욱 정교한 end-to-end RAG 시스템을 개발하도록 유도합니다.



### Seeing Faces in Things: A Model and Dataset for Pareidolia (https://arxiv.org/abs/2409.16143)
- **What's New**: 본 연구에서는 인간과 머신 간의 face pareidolia (얼굴 패레이돌리아)에 대한 인식 차이를 조사하기 위해 새로운 데이터셋인 'Faces in Things'를 소개합니다. 이 데이터셋은 무작위로 생성된 이미지에서 인간이 인식한 얼굴 구조를 포함하고 있습니다.

- **Technical Details**: 이 연구는 5,000개의 웹 이미지로 구성된 'Faces in Things' 데이터셋을 사용하여 인간 얼굴 탐지 시스템의 성능을 분석합니다. 연구 결과는 최신 연구 모델인 RetinaFace를 사용하여 성과를 변별하며, 파리돌리아가 머신에서 어떻게 나타나는지를 탐구합니다.

- **Performance Highlights**: 최신 모델은 얼굴 패레이돌리아 탐지에서 인간의 성능에 비해 상당한 격차를 보였습니다. 연구는 이 격차의 약 절반이 동물 얼굴 탐지 모델을 미세 조정하는 것에서 개선될 수 있음을 보여줍니다. 또한, 'Goldilocks zone'이라고 불리는 조건들이 패레이돌리아를 유도할 수 있음을 실험으로 확인하였습니다.



### Exploring Hint Generation Approaches in Open-Domain Question Answering (https://arxiv.org/abs/2409.16096)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이 논문에서는 기존의 정보 검색(retrieval) 및 생성(generation) 기반 방법 대신 자동 힌트 생성(Automatic Hint Generation, HG) 기술을 활용한 새로운 QA 시스템 구성 요소인 HINTQA를 제안합니다. HINTQA는 질문에 대한 답변 가능성을 제시하는 힌트를 생성하여 QA 시스템의 정확도를 높입니다.

- **Technical Details**: HINTQA는 질문 q와 후보 답변 집합 𝒜을 활용하여 다수의 힌트를 생성합니다. 각 힌트는 수렴 점수(convergence score)인 HICOS를 통해 힌트의 유용성을 측정하며, 이는 질문에 대한 잠재적 답변을 좁힙니다. 제안된 시스템은 TriviaQA, Natural Questions, Web Questions 데이터셋을 사용하여 세 가지 QA 데이터셋에서 힌트 생성의 효과를 실험했습니다.

- **Performance Highlights**: HINTQA 방식은 정보 검색 및 생성 기반 방법보다 우수한 성과를 보였습니다. 연구 결과에 따르면 힌트를 사용하는 것이 검색된 문서나 생성된 문맥보다 답변의 정확성을 높이는 데 더 효과적이라는 것을 증명했습니다.



### SLIMER-IT: Zero-Shot NER on Italian Languag (https://arxiv.org/abs/2409.15933)
- **What's New**: 이 논문에서는 이탈리아어에 대한 제로샷 이름 개체 인식(Zero-Shot Named Entity Recognition, NER) 평가 프레임워크를 정의하고, 제로샷 NER을 위한 SLIMER의 이탈리아어 버전인 SLIMER-IT를 소개합니다. SLIMER-IT는 정의 및 가이드라인으로 향상된 프롬프트를 활용하여 다루지 않은 엔티티 태그를 식별하는 데 우수성을 보입니다.

- **Technical Details**: SLIMER-IT는 대형 언어 모델(LLM)을 기반으로 하며, 인스트럭션 튜닝(instruction tuning)을 통해 성능을 개선합니다. 이 접근법은 주어진 텍스트에서 각각의 엔티티 타입을 효과적으로 추출하기 위해 설계된 프롬프트를 사용하여, 모델이 각 엔티티 타입에 집중할 수 있도록 지원합니다. SLIMER는 네임드 엔티티(Named Entity)에 대한 정의와 가이드라인을 제공하여 모델의 라벨링을 최적화합니다.

- **Performance Highlights**: SLIMER-IT는 기존의 다른 최첨단 모델들과 비교했을 때 보지 못한 엔티티 태그(label)를 라벨링하는 데 있어 뛰어난 성능을 보여주었습니다. 실험 결과는 SLIMER-IT가 이탈리아어로 된 데이터셋에서 제로샷 NER을 수행하는 데 매우 효과적임을 입증하였습니다.



### Mitigating Digital Discrimination in Dating Apps - The Dutch Breeze cas (https://arxiv.org/abs/2409.15828)
- **What's New**: 2023년 9월, 네덜란드 인권 연구소는 네덜란드의 데이팅 앱인 Breeze가 비백인에 대해 차별적일 수 있다는 의심을 제기한 결정에 따라, Breeze가 인종에 기반한 차별을 방지해야 한다고 판단했습니다.

- **Technical Details**: 이 논문은 Breeze의 매칭 알고리즘에서 인종에 기반한 차별이 불법인지, 그리고 데이팅 앱들이 그들의 매칭 알고리즘에서 차별을 완화하거나 중단하기 위한 방법을 어떻게 개발할 수 있는지를 다룹니다. 또한, 컴퓨터 과학(computer science)과 법률(law)의 통찰을 결합하여 Breeze 결정의 법적 및 기술적 어려움을 분석합니다.

- **Performance Highlights**: Breeze 사건이 공정하고 비차별적인 머신 러닝(machine learning) 분야의 교육 및 실천에 미치는 영향에 대해 논의하며, 차별을 해결하기 위한 유망한 솔루션을 제시합니다.



### LLM-Cure: LLM-based Competitor User Review Analysis for Feature Enhancemen (https://arxiv.org/abs/2409.15724)
Comments:
          25 pages

- **What's New**: 이 연구에서는 사용자 리뷰를 통한 경쟁 앱 분석을 통해 모바일 앱 기능 향상을 자동으로 제안하는 LLM-Cure라는 접근법을 제안합니다.

- **Technical Details**: LLM-Cure는 사용자 리뷰에서 기능을 추출하고 분류하기 위해 Large Language Model (LLM)을 사용합니다. 앱 내 불만 사항이 제공되면, 경쟁 앱에서 4와 5점 리뷰를 큐레이션하여 타겟 앱에 대한 개선점을 제안합니다.

- **Performance Highlights**: LLM-Cure는 1,056,739개의 리뷰를 분석하여 기능 할당에서 13%의 F1-score, 16%의 recall, 11%의 precision 향상을 보였습니다. 또한, 제안된 개선 사항의 73%가 실제로 구현된 것으로 확인되었습니다.



### Optimizing News Text Classification with Bi-LSTM and Attention Mechanism for Efficient Data Processing (https://arxiv.org/abs/2409.15576)
- **What's New**: 이 논문은 전통적인 수동 분류 방법의 비효율성을 극복하기 위해 딥 러닝을 기반으로 한 뉴스 텍스트 자동 분류 방안을 제안합니다.

- **Technical Details**: 제안하는 방법은 Bi-directional Long Short-Term Memory Network (Bi-LSTM)와 Attention Mechanism을 결합한 최적화 모델을 사용하여 뉴스 텍스트의 효율적인 분류와 관리를 달성합니다.

- **Performance Highlights**: 실험 결과, 이 솔루션은 분류의 정확성과 시의성을 크게 향상시키고 수동 개입의 필요성을 줄이며, 뉴스 산업의 정보 처리 능력을 향상시키고 정보 흐름의 속도를 가속화하는 데 중요한 실용적 의미가 있음을 보여줍니다.



### Stalactite: Toolbox for Fast Prototyping of Vertical Federated Learning Systems (https://arxiv.org/abs/2409.15558)
- **What's New**: 이 논문에서는 다양한 데이터 소유자가 상대적으로 분산된 데이터를 활용하여 머신러닝 모델을 학습할 수 있도록 하는 Vertical Federated Learning (VFL)용 오픈 소스 프레임워크인 Stalactite를 소개합니다. 기존 프레임워크보다 연구자들이 알고리즘 개발에 집중할 수 있도록 UI가 개선되었으며, VFL의 다양한 알고리즘을 쉽게 구현할 수 있는 기능을 제공합니다.

- **Technical Details**: Stalactite는 수학적 개념과 메시지 교환 로직을 분리하는 설계를 통해 VFL 알고리즘을 쉽게 코드로 변환할 수 있도록 합니다. multi-thread, multi-process, distributed 실행 모드를 지원하며, 이들 모드 간의 전환이 간단합니다. 이 프레임워크는 데이터 전송, 훈련 메트릭 저장 등 다양한 기능을 포함하고 있습니다. 또한, Stalactite는 새로운 고유의 공개 데이터셋 SBOL을 제공하며, 레코드 ID를 매칭하여 훈련 집합을 형성합니다.

- **Performance Highlights**: Stalactite는 강화된 logging 기능을 통해 분산 실행 중의 payload, 교환 시간 및 머신러닝 메트릭을 기록할 수 있습니다. 이 프레임워크는 기존의 VFL 알고리즘을 지원하며, 보안성과 성능 면에서 기존 산업 도구의 한계를 극복하고 연구를 위한 실험 환경을 제공합니다.



### MedCodER: A Generative AI Assistant for Medical Coding (https://arxiv.org/abs/2409.15368)
- **What's New**: 이번 연구에서는 MedCodER라는 새로운 Generative AI 프레임워크를 소개하며, 의료 코딩의 자동화를 위한 혁신적인 접근 방식을 제시합니다. 특히, 이 프레임워크는 추출(Extraction), 검색(Retrieval), 재정렬(Re-ranking) 기술을 핵심 요소로 활용하여 높은 정확도를 자랑합니다.

- **Technical Details**: MedCodER는 의료 기록에서 질병 진단과 지원 증거를 추출하고, ICD-10 코드를 후보 코드로 검색한 다음, 이를 종합하여 최종 코드를 예측합니다. 이 과정에서 LLM(대형 언어 모델)의 파라메트릭 지식을 보완하기 위해 검색 및 재정렬 기술을 통합하여 성능을 향상시킵니다.

- **Performance Highlights**: MedCodER는 ICD 코드 예측에서 0.60의 micro-F1 점수를 달성하여 현재 최고의 방법보다 유의미하게 향상된 성과를 보입니다. 또한, 제안된 데이터셋은 질병 진단과 ICD 코드, 그리고 이를 정당화하는 지원 증거 텍스트가 주석 처리되어 있어, 코드 선택의 신뢰성을 높이는 데 기여합니다.



### VERA: Validation and Enhancement for Retrieval Augmented systems (https://arxiv.org/abs/2409.15364)
- **What's New**: VERA는 Retrieval-Augmented Generation (RAG) 시스템을 위한 평가 및 개선 시스템으로, LLM의 응답精度를 향상시키기 위한 새로운 방법을 제공합니다. 또한, VERA는 외부 정보를 효과적으로 활용하도록 설계되었습니다.

- **Technical Details**: VERA는 수집된 컨텍스트의 적합성과 불필요한 정보를 제거하는 데 중점을 둡니다. 이 시스템은 평가자 및 향상 LLM을 사용하여 응답 생성 전에 컨텍스트를 평가하고, 응답 생성 후에는 응답을 분리하여 각 문장의 적합성을 점검합니다.

- **Performance Highlights**: 실험 결과, VERA는 소규모 공개 오픈 소스 모델에서 뿐만 아니라 대규모 최첨단 모델에서도 성능을 개선하는 데 뛰어난 효능을 나타냈습니다. VERA는 정보 생성에서 높은 정확성 및 신뢰성을 요구하는 응용 프로그램에 유용한 도구로 자리 잡을 잠재력을 보여주고 있습니다.



### An Efficient Recommendation Model Based on Knowledge Graph Attention-Assisted Network (KGATAX) (https://arxiv.org/abs/2409.15315)
- **What's New**: 이번 연구에서는 다중 소스 정보를 효과적으로 활용하지 못하는 전통적인 추천 시스템의 한계를 극복하기 위해 'Knowledge Graph Attention-assisted Network (KGAT-AX)'라는 새로운 추천 모델을 제안합니다.

- **Technical Details**: KGAT-AX 모델은 추천 시스템에 지식 그래프(knowledge graph)를 통합하고 주의 메커니즘(attention mechanism)을 도입하여 더 높은 차원의 연결성을 탐구합니다. 다층 상호작용 정보 전파(multilayer interactive information propagation)를 통해 모델은 정보를 집계하여 일반화 능력을 향상시킵니다. 또한, 홀로그램 임베딩(holographic embeddings)을 통해 보조 정보(auxiliary information)를 엔티티에 통합하여, 인접 엔티티의 정보를 학습하여 더 나은 활용이 가능합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험을 통해 KGAT-AX 모델의 합리성과 효과성을 입증하였으며, 공공 데이터셋에서 다른 기준선 모델(baseline models)과 비교하여 KGAT-AX가 지식 정보 캡처 및 관계 학습 능력에서 우수함을 확인했습니다.



### Equivariance-based self-supervised learning for audio signal recovery from clipped measurements (https://arxiv.org/abs/2409.15283)
- **What's New**: 이번 연구에서는 비선형 역문제(non-linear inverse problem)인 잘린 측정값(clipped measurements)으로부터 오디오 신호를 복구하는 데 자가지도학습(self-supervised learning) 기법을 적용했습니다. 그동안 자가지도학습은 주로 선형 역문제(linear inverse problems)에 집중되었습니다.

- **Technical Details**: 연구에서는 에퀴바리언스 기반(self-supervised loss)을 제안하며, 이를 통해 잘린 측정값에서 오디오 신호를 복구하는 방법을 연구했습니다. 특히 잘린 측정값의 수준이 다양하고 조절된 조건에서 성능을 평가하였습니다.

- **Performance Highlights**: 제안된 에퀴바리언스 기반 자가지도 해득 전략은 완전지도 학습(fully supervised learning)과 비교했을 때 성능이 유사하게 나왔으며, 교육을 위해 오직 잘린 측정값만을 요구하는 장점을 가지고 있습니다.



### Recommendation with Generative Models (https://arxiv.org/abs/2409.15173)
Comments:
          This submission is a full-length book, expanding significantly on two chapters previously submitted (arXiv:2409.10993v1, arXiv:2408.10946v1). It includes additional chapters, context, analysis, and content, providing a comprehensive presentation of the subject. We have ensured it is appropriately presented as a new, distinct work. arXiv admin note: substantial text overlap with arXiv:2409.10993

- **What's New**: 이 논문은 생성 모델(generative models)과 그 응용 분야에 대한 포괄적인 이해를 제공하며, 특히 딥 생성 모델(deep generative models) 및 그 분류 방법에 중점을 두고 있습니다.

- **Technical Details**: 연구에서 제시된 분류 체계는 딥 생성 모델(DGMs)을 ID 기반 모델(ID-driven models), 대형 언어 모델(large language models, LLMs), 다중 모달 모델(multimodal models)의 세 가지 유형으로 나눴습니다. 각 유형은 고유한 기술적 및 구조적 진전을 다룹니다.

- **Performance Highlights**: Gen-RecSys(생성 권장 시스템)는 더 개인화된, 매력적이며 동적인 사용자 경험을 창출하여 추천의 정확성과 다양성을 개선하는 데 기여합니다. 이 연구는 대화형 AI(conversational AI) 및 다중 모달 콘텐츠 생성(multimodal content generation)과 같은 다양한 분야에서의 발전 방향을 탐색합니다.



### Don't Use LLMs to Make Relevance Judgments (https://arxiv.org/abs/2409.15133)
- **What's New**: 본 논문에서는 TREC 스타일 테스트 수집을 위한 관련성 판단을 생성할 때 대규모 언어 모델(LLM)을 사용하지 말라는 메시지를 전달합니다. LLM이 자연어 프롬프트에 반응하여 인간과 유사한 텍스트 출력을 생성하는 최근의 발전이 정보 검색(IR) 연구자들에게 관련성 판단 수집 과정에서 이 모델들을 어떻게 활용할 수 있을지를 고민하게 했습니다.

- **Technical Details**: 관련성 판단을 수집하는 과정은 복잡하고 비용이 많이 드는 작업입니다. 전통적으로 TREC(텍스트 검색 구성)에서는 2-4주 동안 6명의 계약자가 팀을 이루어 작업을 수행하며, 그 과정에서 소프트웨어를 개발하고 관련성 판단을 기록하는 것에 집중합니다. LLM4Eval 워크숍은 이러한 관련성 판단 생성을 위한 실험을 진행한 자리이며, 결과적으로 도출된 주장은 자동 평가의 가능성을 탐구했던 이전 연구들을 계승하였습니다.

- **Performance Highlights**: 전반적으로 LLM을 사용하여 TREC 스타일의 평가를 위한 관련성 판단을 생성하는 것이 바람직하지 않다는 결론을 도출하였습니다. 정보 검색의 불확실성과 예측의 오류는 성능 평가에 있어 중요한 요소임을 강조하였습니다. 실제 하드웨어와 소프트웨어 개발 없이도 관련성 판단의 효율성을 극대화하기 위한 다양한 접근 방법들이 모색되고 있습니다.



### EMERS: Energy Meter for Recommender Systems (https://arxiv.org/abs/2409.15060)
Comments:
          Accepted at the RecSoGood 2024 Workshop co-located with the 18th ACM Conference on Recommender Systems

- **What's New**: 최근 머신러닝 발전으로 인해 추천 시스템의 교육, 평가 및 배포 과정에서 에너지 소비가 증가하고 있지만, 연구 커뮤니티는 실험의 에너지 소비를 잘 보고하지 않습니다. 이를 해결하기 위해 에너지 소비를 측정, 모니터링 및 기록할 수 있는 EMERS라는 소프트웨어 라이브러리를 소개합니다.

- **Technical Details**: EMERS는 Python 기반의 오픈소스 라이브러리로, 스마트 플러그를 통해 추천 시스템 실험의 에너지를 측정하고 기록합니다. EMERS는 사용자 인터페이스를 제공하여 에너지 소비를 모니터링하고 분석하며, 자동화된 보고서를 생성해 커뮤니티와 공유할 수 있습니다. 주요 기능으로는 사용자 인터페이스, 표준화된 보고서 생성, API를 통한 에너지 소비 로그 기록 및 독립형 로그 기록이 있습니다.

- **Performance Highlights**: EMERS는 에너지 소비를 고주파로 측정할 수 있는 스마트 플러그와 연동되어 있으며, 다양한 하드웨어 시스템에서 호환성과 정확성을 보장합니다. EMERS는 사용자 컴퓨터의 성능에 미치는 영향을 최소화하면서도, 추가 비용이 발생하는 스마트 플러그를 통해 정확한 에너지 소비 측정을 지원합니다.



### FedSlate:A Federated Deep Reinforcement Learning Recommender System (https://arxiv.org/abs/2409.14872)
- **What's New**: FedSlate는 사용자의 행동 간 상호작용을 고려하여 멀티 플랫폼에서 추천 알고리즘의 효율성을 극대화하는 연합 강화 학습 (federated reinforcement learning) 기반의 새로운 접근 방식을 제안합니다.

- **Technical Details**: FedSlate 알고리즘은 SlateQ 알고리즘을 활용하여 추천 콘텐츠의 가치를 평가하고, 사용자 행동의 장기 패턴을 학습합니다. 이 알고리즘은 각 플랫폼에서 로컬 Q-값을 계산하고 이를 중앙 서버에 전달하여 글로벌 Q-값을 생성하는 구조를 가집니다. 이후 로컬 에이전트는 이 Q-값을 바탕으로 정책 결정을 내립니다.

- **Performance Highlights**: 실험 결과, FedSlate는 기존의 기준 방법들에 비해 다양한 환경 설정에서 우수한 성능을 보여주었으며, 기준 방법이 전혀 적용되지 않는 상황에서도 추천 전략 학습을 가능하게 했습니다.



### Pre-trained Language Model and Knowledge Distillation for Lightweight Sequential Recommendation (https://arxiv.org/abs/2409.14810)
Comments:
          in Chinese language

- **What's New**: 이 논문에서는 사전 훈련된 언어 모델(pre-trained language model)과 지식 증류(knowledge distillation)를 기반으로 한 새로운 연속 추천 알고리즘(sequential recommendation algorithm)을 제안합니다.

- **Technical Details**: 이 알고리즘은 두 단계로 작동합니다. 첫 번째 단계에서는 추천 데이터셋에서 사전 훈련된 언어 모델을 미세 조정(fine-tuning)하여 추천 작업에 사전 훈련된 지식을 전이합니다. 두 번째 단계에서는 훈련된 언어 모델을 경량 모델(lightweight model)로 변환하기 위해 지식을 증류(distill)합니다.

- **Performance Highlights**: 여러 공개 추천 데이터셋에서 광범위한 실험을 수행한 결과, 제안한 알고리즘이 추천 정확도(recommendation accuracy)를 향상시키고 적시 추천 서비스(timely recommendation services)를 제공함을 확인했습니다.



### EDGE-Rec: Efficient and Data-Guided Edge Diffusion For Recommender Systems Graphs (https://arxiv.org/abs/2409.14689)
Comments:
          6 pages, 13 figures

- **What's New**: 본 연구에서는 사용자-아이템 상호작용을 예측하기 위해 과거의 이진 데이터에만 의존하던 기존 추천 시스템의 한계를 극복하기 위해, Row-Column Separable Attention (RCSA)라는 새로운 주의 메커니즘을 제안합니다. 이 메커니즘은 실제 값의 상호작용 강도와 사용자 및 아이템의 특성을 직접 활용합니다.

- **Technical Details**: RCSA 메커니즘을 기반으로 한 새로운 Graph Diffusion Transformer (GDiT) 아키텍처를 통해 사용자-아이템 상호작용 그래프의 가중치가 있는 상호작용 행렬을 반복적으로 디노이즈하는 과정을 수행합니다. 이 과정에서는 사용자-아이템 평점 상호작용에서 파생된 에지 가중치와 이원적 구조의 상호작용 그래프를 이용하여 디노이징 과정에서 사용자 및 아이템의 특성을 조건으로 사용합니다.

- **Performance Highlights**: 제안된 방법은 추천 시스템의 사용자-아이템 상호작용 효율성을 높이는 데 기여하며, 예상치 못한 상호작용 강도를 근사하는 데 있어 효과적입니다. 이전 연구들과 비교했을 때, 사용자 및 아이템의 특성을 통합하여 사용자-아이템 상호작용 그래프에서 각 에지의 가중치를 직접 예측하는 최초의 접근 방식입니다.



### Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling (https://arxiv.org/abs/2409.14683)
- **What's New**: 이번 논문에서는 ColBERT와 같은 다중 벡터 검색 방법의 저장소 및 메모리 요구사항을 줄이기 위한 간단한 클러스터링 기반의 Token Pooling 방법을 도입했습니다. 이 방법은 저장해야 하는 벡터의 수를 획기적으로 줄여줍니다.

- **Technical Details**: Token Pooling 방법은 개별 벡터를 클러스터링하고 평균 풀링을 통해 하나의 벡터로 변환하는 2단계 시스템으로 작동합니다. 세 가지 풀링 방법이 제안되었으며, 특히 계층적 클러스터링이 가장 우수한 결과를 보여주었습니다.

- **Performance Highlights**: 이 방법은 ColBERT 인덱스를 평균적으로 50%의 저장 공간을 줄이면서도 성능 저하가 없음을 보여주었으며, 66%까지의 벡터 수 감소도 가능하였고, 이 경우 성능 저하는 3% 이하에 머물렀습니다.



### Robust Training Objectives Improve Embedding-based Retrieval in Industrial Recommendation Systems (https://arxiv.org/abs/2409.14682)
Comments:
          RobustRecSys workshop @ RecSys 2024

- **What's New**: 본 연구는 소셜 미디어 플랫폼에서 대규모 친구 추천 시스템과 관련된 SSMTL(self-supervised multitask learning)의 강건성(robustness)이 실제 환경에서도 개선될 수 있는지를 평가합니다. 연구를 통해 얻어진 통계적으로 유의미한 결과는 SSMTL 기반 EBR(embedding-based retrieval)의 성능 향상 가능성을 보여줍니다.

- **Technical Details**: 제안된 방법은 두 단계 프로세스에서 작동하며, 첫 번째 단계는 수백만 명의 후보 사용자를 대상으로 후보를 선택하는 검색(retrieval) 단계이며, 두 번째 단계는 선택된 후보 사용자 목록에서 최종 추천을 위한 정렬(ranking) 단계를 포함합니다. 연구에서는 link prediction을 사용하여 사용자 간 관계도에 의한 임베딩을 생성하는 방법을 사용하고, SSMTL을 통해 이 강건성을 대규모 산업 추천 시스템에 적용합니다.

- **Performance Highlights**: 실험 결과, SSMTL 접근 방식을 사용한 경우 친구 추천 시스템의 새로운 친구 추가율에서 최대 5.45%, 콜드 스타트 사용자에서 최대 1.91%의 유의미한 성과 향상이 관찰되었습니다.



### tabulapdf: An R Package to Extract Tables from PDF Documents (https://arxiv.org/abs/2409.14524)
Comments:
          10 pages, 1 figure

- **What's New**: 이 논문에서는 PDF 파일에서 테이블을 직접 R로 가져오는 데 사용되는 R 패키지인 tabulapdf를 소개합니다. 이 패키지는 Tabula Java 라이브러리를 활용하여 데이터 추출 과정을 간소화합니다.

- **Technical Details**: tabulapdf는 PDF 파일에서 테이블을 자동 및 수동으로 추출할 수 있는 기능을 제공하며, R의 Shiny 인터페이스와 통합되어 사용자에게 마우스를 이용한 영역 선택을 가능하게 합니다. 주요 함수인 extract_tables()는 PDF 파일의 모든 페이지에서 테이블을 추출하여 R의 tibble 형식으로 반환합니다.

- **Performance Highlights**: tabulapdf는 조사 저널리즘 분야에서 데이터 추출 시간을 줄여주는 유용한 도구로, 실제 사용 사례로는 COVID-19 치료에 관한 데이터 추출이 포함되어 있습니다. 이 패키지는 PDF의 각 페이지에서 테이블의 존재 여부를 판단하고, 두 가지 알고리즘(‘lattice’와 ‘stream’)을 통해 정확한 데이터 추출을 지원합니다.



### Sliding Window Training -- Utilizing Historical Recommender Systems Data for Foundation Models (https://arxiv.org/abs/2409.14517)
Comments:
          To be published In 18th ACM Conference on Recommender Systems (RecSys '24), October 14--18, 2024, Bari, Italy

- **What's New**: 이 논문에서는 sliding window training 기법을 도입하여 장기 사용자 상호작용 이력을 효과적으로 모델 학습에 활용하는 방법을 제안합니다. 이 방법은 모델 입력 차원을 증가시키지 않으면서도 사용자의 장기 선호를 학습할 수 있도록 지원합니다.

- **Technical Details**: 모델은 대규모 사용자 상호작용 데이터셋(약 2억 5천만 사용자와의 상호작용 포함)을 사용하여 훈련됩니다. 기존의 고정 창법(fixed window approach) 대신, sliding window를 적용하여 사용자 상호작용 이력의 모든 부분을 포함하여 훈련합니다. 이를 통해 모델은 사용자의 장기 관심사를 더 잘 학습할 수 있고, 각 훈련 시점에서 관찰되는 항목 수가 증가합니다.

- **Performance Highlights**: 슬라이딩 윈도우 방식을 사용한 학습이 기본 모델보다 모든 평가 지표에서 우수한 성능을 보였습니다. 이 연구의 결과는 RecSys FM이 사용자 상호작용을 최적화하여 장기적인 사용자 관심사를 이해하고, 전반적인 항목 표현 품질을 향상시키는 데 기여할 것임을 보여줍니다.



### Revisiting BPR: A Replicability Study of a Common Recommender System Baselin (https://arxiv.org/abs/2409.14217)
Comments:
          This paper is accepted at the Reproducibility track of the ACM RecSys '24 conference

- **What's New**: 이 논문에서는 Bayesian Personalized Ranking (BPR) 모델의 다양한 구현 세부사항을 분석하고, 이를 통해 모델의 성능에 미치는 영향을 조사했습니다. 뿐만 아니라, 오픈소스 구현 체계에 대한 일관성 문제를 확인하고, 일부 구현에서 성능이 50%까지 감소하는 문제를 발견했습니다.

- **Technical Details**: BPR 모델은 매트릭스 분해(matrix factorization) 기반의 협업 필터링(collaborative filtering) 방법론으로, 쌍별 순위 손실(pairwise ranking loss)을 도입하여 추천 성능을 향상시킵니다. 본 연구는 다양한 오픈소스 구현 체계와의 비교를 통해 BPR의 세부사항이 성능에 미치는 영향을 분석했습니다. 특정 하이퍼파라미터(hyperparameter)의 조정이 BPR 모델의 성능을 현대 기법들과 유사하게 끌어올릴 수 있음을 보여주었습니다.

- **Performance Highlights**: BPR 모델은 하이퍼파라미터 조정을 통해 최근의 SOTA(state-of-the-art) 방법들과 유사한 성능을 발휘할 수 있으며, 특정 데이터셋에서는 이를 초월한 성과를 남깁니다. 특히, Million Song Dataset에서는 BPR 모델이 NDCG@100에서 Mult-VAE 모델에 비해 10% 향상된 성능을 보였습니다.



### Data Generation via Latent Factor Simulation for Fairness-aware Re-ranking (https://arxiv.org/abs/2409.14078)
- **What's New**: 이 논문은 공정성 인식 추천 시스템(fairness-aware recommendation analysis) 분야에서 사용될 새로운 합성 데이터(synthetic data)를 제안합니다. 저자들은 기존 방식의 한계를 극복하고, 공정성을 고려한 재배치(re-ranking) 알고리즘을 연구하기 위한 합성 추천 시스템 출력(synthetic recommender system outputs)을 생성하는 방법론을 개발하였습니다.

- **Technical Details**: 저자들은 LAtent Factor Simulation (LAFS)이라는 새로운 방법론을 통해 합성 추천 목록을 생성합니다. 이 과정에서 행렬 분해(matrix factorization) 모델이 생성할 수 있는 잠재 요인(latent factors) 행렬을 시뮬레이션하고, 이를 바탕으로 표본 평가(sample ratings)를 생성하여 다양한 공정성 속성을 가진 추천 목록을 형성합니다.

- **Performance Highlights**: LAFS 방법을 통해 생성된 추천 목록은 실제 추천 시스템의 것이와 유사한 특성을 가지며, 다양한 공정성 관련 조건에 따라 데이터 세트의 특성을 조정할 수 있는 가능성을 보여줍니다. 이를 통해 공정성 인식 재배치 알고리즘의 효과를 다양하게 평가할 수 있는 장점이 있습니다.



### WebQuest: A Benchmark for Multimodal QA on Web Page Sequences (https://arxiv.org/abs/2409.13711)
- **What's New**: WebQuest는 다중 페이지 질문-답변(Question-Answering, QA) 데이터셋으로, 웹 상호작용에서의 정보 검색 및 추론을 동시에 요구하는 새로운 벤치마크를 제시합니다. 이 데이터셋은 단일 화면, 다중 화면 및 내비게이션 경로 기반의 질문을 포함하고 있어 기존의 다단계 웹 탐색 방식과는 차별화된 접근을 보여줍니다.

- **Technical Details**: WebQuest 데이터셋은 세 가지 질문 카테고리(단일 화면 QA, 다중 화면 QA, 내비게이션 경로 기반 질문)를 포함하여, 사용자 행동에 기반한 웹 상호작용 시퀀스를 반영하며, 다양한 멀티모달 모델(GPT-4V, Gemini Flash, Claude 3 등)을 평가합니다. 특히, Chain-of-Thought prompting 기법을 적용하여 다중 화면 추론 능력을 향상시키는 방법을 모색합니다.

- **Performance Highlights**: 모델 평가는 단일 페이지와 다중 페이지 추론 간의 성능 차이를 보여주며, WebQuest는 기존 QA 기반 콘텐츠 이해와 에이전트 모델 연구 간의 격차를 해소하는 새로운 QA 모드를 제공합니다. 또한, 데이터셋은 다양한 모델의 능력을 자세히 평가할 수 있는 3개의 데이터 하위 집합을 포함하고 있습니다.



### Retrieval Augmented Generation-Based Incident Resolution Recommendation System for IT Suppor (https://arxiv.org/abs/2409.13707)
Comments:
          7 pages, 3 figures, 6 tables

- **What's New**: 이 연구는 IT 지원 도메인에서의 솔루션 추천 시스템을 위해 개발된 Retrieval Augmented Generation(RAG) 시스템을 소개합니다. 특히, IBM Slate 125m 모델을 사용하여 단일-턴과 다중-턴 IT 지원 사례를 분류하는 새로운 접근법과 성능을 보고합니다.

- **Technical Details**: 시스템은 네 가지 주요 구성 요소로 이루어져 있습니다: encoder-only transformer classifier, query generation system, retriever system, 그리고 answer generator system. 데이터 수집은 약 19,000개의 실제 지원 사례를 기반으로 하며 다양한 소프트웨어 제품에서 수집되었습니다. CNN(Convolutional Neural Networks) 및 cosine similarity를 활용하여 문서를 검색하고 재랭크합니다.

- **Performance Highlights**: 연구 결과, 작은 모델들이 RAG 사건 해결 사용 사례에서 매우 큰 모델들과 동등하거나 더 나은 성능을 보여주었다고 보고합니다. 최종적으로 F1 점수가 0.65에 이르고, 클래스 분류 정확도가 0.54, 재현율이 0.80으로 나타났습니다.



### Zeroshot Listwise Learning to Rank Algorithm for Recommendation (https://arxiv.org/abs/2409.13703)
- **What's New**: 이번 연구는 순위 학습(Learning to rank) 기술의 저조한 채택을 극복하기 위해 제로샷(listwise learning to rank) 알고리즘을 제안합니다. 이는 정보 검색 분야에서 널리 사용되던 기존의 방법과 차별화됩니다.

- **Technical Details**: 연구에서는 순위 통계 근사(order statistic approximation)와 파워 법칙 분포(power law distribution)를 활용하여 제안한 알고리즘을 설계했습니다. 이 접근법은 추천 시스템에서 정확하고 공정한 결과를 제공함을 실험을 통해 입증하고 있습니다.

- **Performance Highlights**: 제안하는 알고리즘은 실험에서 정확성과 공정성을 모두 만족하는 것으로 나타났으며, 추천 시스템 분야에서의 효용성을 강조합니다.



### MAS4POI: a Multi-Agents Collaboration System for Next POI Recommendation (https://arxiv.org/abs/2409.13700)
Comments:
          14 pages, 4 figures

- **What's New**: 이 논문은 LLM(대형 언어 모델)을 기반으로 한 다중 에이전트 시스템(MAS4POI)을 제안하여 사용자 맞춤형 다음 POI(관심 지점) 추천의 성능을 향상시키고자 합니다. MAS4POI는 데이터 에이전트(DataAgent), 관리자(Manager), 분석가(Analyst), 반영기(Reflector), 사용자 에이전트(UserAgent), 탐색자(Searcher), 내비게이터(Navigator) 등 7개의 전문화된 에이전트를 포함하여 다각적인 협력 프로세스를 지원합니다.

- **Technical Details**: MAS4POI는 서로 다른 LLM을 통합하여 작업 워크플로우 및 리소스 관리, 사용자 데이터 분석, 외부 데이터 접근 등을 수행합니다. 각 에이전트는 POI 추천을 위한 상호작용을 통해 데이터 정교화 및 경로 계획을 지원하며, 시스템은 대규모 실세계 데이터셋을 통해 검증됩니다.

- **Performance Highlights**: MAS4POI는 다음 POI 추천의 정확성을 크게 향상시키며, 사용자 개인화된 서비스 제공 및 실시간 Q&A 기능을 통해 다양한 응용 분야에 쉽게 연결될 수 있습니다. 또한, 데이터 부족 문제를 완화하고 여러 대규모 데이터셋을 통해 시스템의 효과를 검증하였습니다.



### Vietnamese Legal Information Retrieval in Question-Answering System (https://arxiv.org/abs/2409.13699)
Comments:
          7 pages

- **What's New**: 본 연구는 Question Answering 시스템의 신뢰성을 증대시키기 위한 새로운 문서 검색 및 추천 방법을 제시합니다. 특히 베트남어를 대상으로 한 문제를 해결하기 위해 데이터 처리, 검색 순서 최적화 및 정보 재정렬 방법을 도입하고 있습니다.

- **Technical Details**: 제안된 시스템은 대형 언어 모델(LLM)과 BM25 검색, Dense Vector search를 결합하여 효율적이고 정확한 정보 검색을 가능하게 합니다. 또한 Reciprocal Rank Fusion 기법을 활용하여 키워드와 벡터 검색의 결과를 통합하고, Active Retrieval을 통해 소스 문서의 재정렬 과정을 수행합니다.

- **Performance Highlights**: 베트남 법률 정보를 다루는 QA 시스템에서 성능과 신뢰성이 크게 향상되었습니다. 최종적으로 약 1,293,347개의 법률 문서와 2,081개의 질문으로 구성된 데이터셋을 사용하여 실험을 진행하였으며, 정보 검색의 정확성과 사용자 경험이 개선되었습니다.



### Generative AI Is Not Ready for Clinical Use in Patient Education for Lower Back Pain Patients, Even With Retrieval-Augmented Generation (https://arxiv.org/abs/2409.15260)
- **What's New**: 본 연구에서는 Retrieval-Augmented Generation (RAG)과 few-shot learning을 활용하여 허리통증(LBP) 환자를 위한 맞춤형 교육 자료를 생성하는 새로운 접근법을 소개합니다.

- **Technical Details**: 이 연구에서는 대형 언어 모델(LLMs)이 RAG를 사용하여 생성된 교육 자료의 중복성, 정확성 및 완전성을 평가하기 위해 물리치료사가 수동으로 검토하였습니다. 또한 생성된 교육 자료의 가독성은 Flesch Reading Ease 점수를 통해 평가되었습니다.

- **Performance Highlights**: RAG 기반 LLMs는 전통적인 LLMs보다 더 정확하고 완전하며 적은 중복성을 가진 환자 교육 자료를 제공합니다. 그러나 생성된 자료는 아직 임상 실무에 사용하기에는 준비가 되어 있지 않으며, AI 모델의 임상적 관련성과 내용의 세분화를 보장하는 데에는 여전히 상당한 과제가 남아 있습니다.



### Lessons Learned on Information Retrieval in Electronic Health Records: A Comparison of Embedding Models and Pooling Strategies (https://arxiv.org/abs/2409.15163)
- **What's New**: 이번 연구는 의료 분야에서 LLMs의 효과적인 정보 검색을 위한 다양한 embedding 모델과 pooling 방법의 영향을 분석한 것입니다. 특히, BGE라는 일반 도메인 모델이 의료 전용 모델보다 일관되게 우수한 성능을 보인 점이 주목할 만합니다.

- **Technical Details**: 연구는 MIMIC-III와 사설 전자 건강 기록(EHR) 데이터를 사용하는 세 가지 정보 검색 작업을 수행했습니다. 총 일곱 개 모델을 평가했으며, 각 모델에 대한 embedding pooling 전략이 독립적으로 어떻게 작용하는지도 분석했습니다.

- **Performance Highlights**: BGE 모델은 다른 의료 전용 모델보다 뛰어난 검색 성능을 보였으며, 데이터셋과 쿼리 내용에 따라 상당한 변동성이 있음을 발견했습니다. 최적의 pooling 방법을 제안하여 미래의 검색 시스템 디자인에 기여할 수 있는 통계적으로 검증된 권장 사항을 제공했습니다.



### ViBERTgrid BiLSTM-CRF: Multimodal Key Information Extraction from Unstructured Financial Documents (https://arxiv.org/abs/2409.15004)
Comments:
          Accepted in MIDAS (The 8th Workshop on MIning DAta for financial applicationS) workshop of ECML PKDD 2023 conference

- **What's New**: 이 논문은 비정형 문서에서의 핵심 정보 추출(Information Extraction, KIE) 모델에 대한 새로운 접근 방식을 제안합니다. 특히, ViBERTgrid라는 다중 모드 트랜스포머를 비정형 금융 문서에 적응시키고 BiLSTM-CRF 레이어를 통합하여 성능을 크게 향상시킵니다.

- **Technical Details**: 제안된 ViBERTgrid BiLSTM-CRF 모델은 비정형 문서에서의 개체 인식(Named Entity Recognition, NER) 성능을 2%포인트 향상시키면서도 반구조 문서에서의 KIE 성능을 유지합니다. 이 모델은 두 가지 주요 아키텍처인 ViBERTgrid(트랜스포머 기반)와 BiLSTM-CRF(시퀀스 기반)를 결합하여 구문 및 장기 컨텍스트 인식을 제공합니다.

- **Performance Highlights**: 이 모델은 비정형 자금 이체 주문 데이터셋 및 반구조 영수증 데이터셋(SROIE)에서 평가되었으며, SROIE 데이터셋에 대한 토큰 수준 주석을 공개하여 다중 모드 시퀀스 레이블링 모델에서의 사용 가능성을 높였습니다.



### Adaptive Learning on User Segmentation: Universal to Specific Representation via Bipartite Neural Interaction (https://arxiv.org/abs/2409.14945)
- **What's New**: 본 연구에서는 클릭률(CTR) 및 전환률(CVR) 예측을 위한 사용자를 나타내는 새로운 학습 프레임워크를 제안합니다. 본 프레임워크는 정보 병목(information bottleneck)을 통해 먼저 일반적인 사용자 표현을 학습하고, 그 후 신경 상호 작용(neural interaction)을 통해 세분화(specific segmentation)에 맞는 표현을 통합하고 학습합니다.

- **Technical Details**: 제안된 학습 프레임워크는 가우시안 혼합(latent space)을 통해 사용자 특성을 다차원적으로 클러스터링하여 일반 사용자 표현을 학습합니다. 그런 다음, 비파르타이트 그래프(bipartite graph) 구조를 활용하여 특정 세분화에 대한 표현을 생성하는 상호 작용 과정을 설계합니다. 이 과정에서 모든 데이터로부터 메타 지식을 학습하여 다양한 사용자 그룹에 맞게 조정 가능한 모델을 생성합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 오픈소스 벤치마크와 두 개의 오프라인 비즈니스 데이터셋에서 벤치마크 테스트를 수행하였으며, 두 개의 온라인 마케팅 애플리케이션에서도 성공적으로 배포되었습니다. 결과적으로 제안된 방법은 기존 baseline 방법들보다 우수한 성능을 달성하여 CVR 예측에서 효과적으로 구현되었습니다.



### Nirjas: An open source framework for extracting metadata from the source cod (https://arxiv.org/abs/2409.14609)
Comments:
          2022 12th International Conference on Cloud Computing, Data Science & Engineering (Confluence)

- **What's New**: 이 논문에서는 소프트웨어 개발 과정에서 중요한 메타데이터(metadata)와 주석(comments)의 역할을 강조합니다. 저자들은 파이썬(Python) 기반의 오픈 소스 프레임워크인 Nirjas를 소개하며, 이를 통해 소스 코드에서 메타데이터를 구조적으로 추출할 수 있음을 설명합니다.

- **Technical Details**: Nirjas는 다양한 프로그래밍 언어의 소스 파일에 주석을 추가할 때 사용되는 여러 구문(syntax)과 타입(type), 널리 사용되는 관례(conventions)를 지원합니다. 이 프레임워크는 Regex를 사용하여 정밀하게 메타데이터를 추출하며, 비정규 표현식(non-Regex) 방법은 종종 정확성과 노이즈(separation noise) 처리에서 한계를 보입니다. Nirjas는 주석의 유형(type), 소스 코드 및 해당 주석에 대한 상세한 정보(예: 줄 번호(line number), 파일 이름(file name), 사용 언어(language used), 총 소스 코드 라인 수(SLOC) 등)를 분리하여 제공합니다.

- **Performance Highlights**: Nirjas는 독립형 파이썬 프레임워크/라이브러리로, 소스 또는 pip(파이썬 패키지 설치기)를 통해 쉽게 설치할 수 있습니다. 이 도구는 Google Summer of Code 프로젝트의 일환으로 처음 생성되었으며, 현재 FOSSology 조직에서 개발 및 유지 관리되고 있습니다.



### Beyond Words: Evaluating Large Language Models in Transportation Planning (https://arxiv.org/abs/2409.14516)
- **What's New**: 2023년 Generative Artificial Intelligence (GenAI)의 급속한 발전이 도시 교통 및 물류 분야에 혁신적인 변화를 가져왔습니다. 본 연구는 GPT-4와 Phi-3-mini 같은 Large Language Models (LLMs)의 성능을 수송 계획에 적용하는 것을 탐구합니다.

- **Technical Details**: 이 연구는 교통 정보를 반영한 평가 프레임워크를 통해 LLM의 성능과 공간 이해력을 평가합니다. 평가 요소로는 일반적인 지리적 정보 시스템 (GIS) 기술, 교통 관련 도메인 지식 및 현실 세계의 교통 문제 해결 능력이 포함됩니다. 혼합 방법론을 활용하여 연구가 진행되었습니다.

- **Performance Highlights**: 연구 결과, GPT-4는 다양한 GIS 및 교통 관련 작업에서 Phi-3-mini보다 더 뛰어난 정확성과 신뢰성을 보였습니다. 그러나 Phi-3-mini는 특정 분석 시나리오에서 유용함을 나타내어 자원이 제한된 환경에서도 활용 가능성을 보여줍니다. 이 결과는 GenAI 기술이 도시 교통 계획에 미치는 혁신적인 잠재력을 강조합니다.



### Knowledge in Triples for LLMs: Enhancing Table QA Accuracy with Semantic Extraction (https://arxiv.org/abs/2409.14192)
- **What's New**: 이 논문은 semi-structured (반구조화) 테이블에서 Triple을 추출하고 이를 Retrieval-augmented Generation (RAG) 모델과 결합하여 자연어 처리(NLP)에서 질문 응답 시스템(QA)의 정확도와 문맥적 풍부함을 향상시키는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법론은 RDFLib 라이브러리를 사용하여 테이블에서 Triple을 간단히 구성하는 과정을 포함합니다. Triple은 Subject(주어), Predicate(서술어), Object(목적어)로 구성되며, 이를 통해 테이블의 셀 간의 관계를 명확히 표현합니다. 이 Triple은 Fine-tuned GPT-3.5-turbo-0125 모델에 통합되어 응답 생성을 개선하는 데 사용됩니다.

- **Performance Highlights**: 제안된 접근 방식은 FeTaQA 데이터셋에서 기존 기법들보다 성능이 크게 향상되었으며, 특히 Sacre-BLEU 및 ROUGE 지표에서 우수한 성과를 나타냈습니다. 테이블에서 복잡한 정보를 효과적으로 식별하고 명확한 긴 형식의 답변을 생성하는 능력이 두드러집니다.



### OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching (https://arxiv.org/abs/2409.14038)
Comments:
          4 pages, 1 figure

- **What's New**: 이번 연구에서는 LLM(hallucinations)의 발생이 Ontology Matching(OM) 작업에서 중요한 문제임을 제기하고, 이를 해결하기 위한 OAEI-LLM 데이터셋을 제안합니다. 이 데이터셋은 OM 작업에서 LLM의 환각 현상을 평가하기 위한 기준을 제공합니다.

- **Technical Details**: OAEI-LLM 데이터셋은 기존 OAEI 데이터셋의 확장으로, LLM의 OM 작업에서의 환각 유형을 분류하고 이를 바탕으로 LLM의 정답률을 평가합니다. 새로운 스키마 확장을 통해 LLM이 생성한 결과와 인간이 라벨링한 결과를 비교하고, 환각의 발생 정도를 측정합니다.

- **Performance Highlights**: LLM은 OM 작업에서 높은 성능을 보일 수 있지만, 환각 현상으로 인해 낮은 precision(정밀도) 및 recall(재현율) 문제를 초래할 수 있습니다. OAEI-LLM 데이터셋을 통해 LLM의 환각 현상에 대한 이해를 높이고, 향후 OM 개선 연구에 기여할 것으로 예상됩니다.



### Cost-Effective Community-Hierarchy-Based Mutual Voting Approach for Influence Maximization in Complex Networks (https://arxiv.org/abs/2409.14034)
- **What's New**: 이번 연구는 Cost-Effective Community-Hierarchy-Based Mutual Voting이라는 새로운 접근 방식을 제안하여 복잡 네트워크에서 영향력 극대화를 해결합니다. 이 방법은 노드의 중요성을 측정하기 위한 새로운 개념인 Dual-Scale Community-Hierarchy Information을 기반으로 합니다.

- **Technical Details**: Dual-Scale Community-Hierarchy Information을 통해 노드의 계층 구조와 커뮤니티 구조 정보를 통합하여 중요성을 측정합니다. 새로운 Hierarchical-Community Entropy 개념을 통해 커뮤니티 구조 정보를 평가합니다. 또한, 저비용의 Mutual-Influence 기반 투표 메커니즘과 Lazy Score Updating Strategy를 사용하여 seed 노드를 선택하는 방법을 개발합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 16종의 최첨단 기술보다 시간 복잡성과 정확성 간의 균형에서 더 우수한 성능을 보였으며, 균형 지수에서 두 번째로 높은 방법과 비교했을 때 최대 9.29% 향상된 성능을 보여주었습니다.



### Causal Feature Selection Method for Contextual Multi-Armed Bandits in Recommender System (https://arxiv.org/abs/2409.13888)
- **What's New**: 이 논문에서는 기존 기계 학습 모델에 비해 맥락적 멀티암 밴딧 모델에서 중요한 특성을 선택하는 새로운 방법을 제안합니다. 제안된 방법은 이질적 causal 효과(heterogeneous causal effect)를 기반으로 하여 보상 분포에 대한 특성의 기여를 평가합니다.

- **Technical Details**: 제안된 방법은 모델에 특성을 포함시키지 않고, 특성이 보상에 미치는 기여를 기반으로 중요한 특성을 선정합니다. 이 방법은 비선형 관계를 고려하며, 연속형 및 범주형 특성 모두에 적응할 수 있습니다. 또한, 모델 오명세(model mis-specification) 문제를 방지하면서 빠른 계산 속도를 자랑합니다.

- **Performance Highlights**: 모의 데이터와 실제 온라인 실험 데이터를 기반으로 한 실험 평가 결과, 제안된 특성 선택 방법이 중요한 특성을 효과적으로 선택하고, 맥락적 MAB 성과를 향상시킨 것으로 나타났습니다. 더불어, 기존 모델 내장(mis-specification) 방법과 비교했을 때 계산 속도 및 구현 용이성에서 우수한 성과를 보였습니다.



### Segment Discovery: Enhancing E-commerce Targeting (https://arxiv.org/abs/2409.13847)
Comments:
          Accepted at the CONSEQUENCES'24 workshop, co-located with ACM RecSys'24

- **What's New**: 이 논문에서는 고객을 효과적으로 목표로 삼기 위하여 uplift 모델링과 제한 최적화(constrained optimization)를 기반으로 한 새로운 정책 프레임워크를 제안합니다. 이 접근법은 비즈니스 가치의 극대화를 위해 특정 사례에 맞는 개입(intervention)을 수행하는 고객을 식별함으로써, 기존의 무작위 표적 또는 성향에 기반한 접근 방식에 비해 개선점을 보여줍니다.

- **Technical Details**: 제안된 방법론은 두 단계의 접근 방식을 가지고 있습니다. 첫 번째 단계에서는 각 고객에 대한 처치(treatment)의 영향을 추정하고, 두 번째 단계에서는 이러한 추정을 기반으로 고객 집합을 최적화하여 주어진 제약 조건을 고려합니다. 고객의 성향 점수(threshold)와 결과 변수를 기반으로 하는 기존 방법들과 달리, 이 방법론은 불확실성을 감소시키고 보다 일반적인 고객 목표 문제에 적용할 수 있는 프레임워크를 제공합니다.

- **Performance Highlights**: 세 가지 주요 비즈니스 응용 사례를 통한 오프라인 정책 추정 기법과 대규모 온라인 A/B 테스트를 사용하여, 제안된 타겟팅 정책의 유효성을 입증하였습니다. 이 접근법은 기존의 성향 기반 방법에 비해 비즈니스 목표 달성에서 보다 높은 성과를 보여주었습니다.



### Language agents achieve superhuman synthesis of scientific knowledg (https://arxiv.org/abs/2409.13740)
- **What's New**: 이번 연구에서는 PaperQA2라는 고급 언어 모델을 개발하여 과학 문헌 검색 과제에서 인간 전문가와의 성능 비교를 수행했습니다. 결과적으로 PaperQA2는 정보 검색, 요약 및 모순 탐지 작업에서 주제 전문가를 초월할 수 있음을 입증하였습니다.

- **Technical Details**: 연구의 핵심은 PaperQA2를 활용하여 적용된 LitQA2 벤치마크를 통해 과학 문헌에 대한 정보 검색, 요약 및 모순 탐지의 성능을 평가한 점입니다. PaperQA2는 ‘retrieval-augmented generation (RAG)’ 접근 방식을 사용하여, 여러 단계의 작업을 통해 최종 응답을 생성합니다. 각 질문에 대해 평균 14.5개의 논문을 활용하였으며, 정확도는 66.0%로 나타났습니다.

- **Performance Highlights**: PaperQA2는 LitQA2 벤치마크에서 85.2%의 정밀도와 66.0%의 정확도를 기록하였으며, 생물학 논문에서 평균 2.34개의 모순을 발견하는 데 성공했습니다. 이 연구는 AI 모델이 특정 과학 문헌 작업에서 인간 전문가보다 뛰어난 성능을 보여줄 수 있음을 실증적으로 나타냈습니다.



### Shaping the Future of Endangered and Low-Resource Languages -- Our Role in the Age of LLMs: A Keynote at ECIR 2024 (https://arxiv.org/abs/2409.13702)
- **What's New**: 이 논문은 언어가 문화적 및 사회적 정체성을 형성하는 데 중요한 역할을 한다는 점을 강조하며, 오늘날 7100개 이상의 언어 중 상당수가 멸종 위기에 처해 있음을 알립니다. 특히, Occitan 언어를 중심으로 기술과 전통 간의 협력 가능성을 탐구합니다.

- **Technical Details**: Large Language Model (LLM) 기술이 제공하는 번역 및 콘텐츠 생성의 가능성을 논의하며, 이는 멸종 위기 언어 보존과 재활성화의 중요한 요소가 될 수 있습니다. 그러나 이러한 기술은 또한 문화의 동질화 및 이미 취약한 언어의 추가적 소외를 초래할 위험이 있습니다.

- **Performance Highlights**: 인공지능(AI)과 인간의 전문성이 함께 작동하여 언어의 다양성을 보존할 수 있는 희망을 제공할 수 있음을 강조하며, 이를 위해서는 윤리적 및 실용적인 도전 과제를 해결해야 한다고 주장합니다.



New uploads on arXiv(cs.CV)

### Self-Supervised Any-Point Tracking by Contrastive Random Walks (https://arxiv.org/abs/2409.16288)
Comments:
          ECCV 2024. Project link: this https URL . Code: this https URL

- **What's New**: 이번 연구에서는 Tracking Any Point (TAP) 문제에 대한 간단하고 효과적인 self-supervised (자기지도) 접근 방식이 제안되었습니다. 주목할 점은 global matching transformer를 사용해 cycle consistency (순환 일관성) 학습을 통해 관측한 데이터를 통해 모델을 훈련시키는 것입니다.

- **Technical Details**: 제안된 방법은 contrastive random walk (대조적 무작위 보행)를 이용하여 개체의 이동 궤적을 추적하는 것을 목표로 합니다. 이를 위해 space-time graph (공간-시간 그래프)의 transition matrix (전이 행렬)을 정의하여 모든 쌍의 비교를 수행할 수 있는 기법을 채택하였습니다. 이 방법은 데이터 증강 기법을 통해 모델이 shortcut solutions (지름길 해법)에 영향을 받지 않도록 설계되었습니다.

- **Performance Highlights**: TAP-Vid 벤치마크에서 강력한 성능을 입증하였으며, DIFT와 같은 이전의 self-supervised 추적 방법을 초월하였습니다. 실험 결과, TAP-Net과 유사한 경쟁력 있는 성능을 가지며 여러 supervised (지도학습) 방법과 겨룰 수 있는 수준에 도달했습니다.



### MonoFormer: One Transformer for Both Diffusion and Autoregression (https://arxiv.org/abs/2409.16280)
- **What's New**: 이 논문에서는 autoregression 기반의 텍스트 생성과 diffusion 기반의 시각적 생성 방법에 대해 단일 transformer를 공유하여 사용하자는 간단한 아이디어를 제안합니다. MonoFormer라는 이름의 접근 방식은 텍스트와 이미지 생성을 모두 수행할 수 있는 가능성을 보여줍니다.

- **Technical Details**: Transformer는 시각적 생성에 diffusion 모델을 적용하는 데 성공적으로 사용되며, autoregression과 diffusion을 위한 transformer 훈련이 유사하다는 점에서 feasibility가 있습니다. autoregressive transformer는 causal attention mask를 사용하고, diffusion transformer는 bidirectional attention mask를 사용하여 차이를 두고 있습니다. 단일 transformer를 통해 텍스트와 이미지 생성을 모두 학습하는 방법론이 논의됩니다.

- **Performance Highlights**: 실험 결과, MonoFormer는 현재 최첨단 방법에 대한 경쟁력 있는 이미지 생성 성능을 달성했으며, 텍스트 생성 능력 또한 유지합니다.



### Semantic Refocused Tuning for Open-Vocabulary Panoptic Segmentation (https://arxiv.org/abs/2409.16278)
Comments:
          9 pages, 6 figures

- **What's New**: 최근 열린 어휘(panoptic) 세분화(open-vocabulary panoptic segmentation) 기법의 발전이 이루어졌습니다. 새로운 방법인 Semantic Refocused Tuning (SMART)이 마스크 분류(mask classification)의 성능을 크게 향상시키며, 적은 학습 자원으로도 효과적인 방법론을 제안합니다.

- **Technical Details**: SMART는 Semantic-guided Mask Attention과 Query Projection Tuning이라는 두 가지 핵심 혁신을 통해 세분화를 개선합니다. 이를 통해 VLM(비전-언어 모델의 이미지 포커스 조정)에서 마스크 토큰의 새 분포에 적응할 수 있는 효율성을 제공합니다.

- **Performance Highlights**: SMART는 동종의 대표적인 벤치마크에서 최대 1.3 PQ 및 5.4 mIoU의 성능 향상을 보이며, 이전의 최상위 모델과 비교하여 학습 비용을 약 10배 감소시켰습니다.



### AIM 2024 Challenge on UHD Blind Photo Quality Assessmen (https://arxiv.org/abs/2409.16271)
Comments:
          ECCV 2024 - Advances in Image Manipulation (AIM). arXiv admin note: text overlap with arXiv:2401.10511 by other authors

- **What's New**: AIM 2024 UHD-IQA Challenge가 소개되었습니다. 이 대회는 최신 고해상도 사진에 대한 No-Reference Image Quality Assessment (NR-IQA)의 발전을 목표로 하고 있습니다. UHD-IQA Benchmark Database를 기반으로 하며, 뛰어난 기술적 품질을 지닌 6073개의 UHD-1 (4K) 이미지가 포함되어 있습니다.

- **Technical Details**: UHD-IQA는 현대 카메라로 촬영된 섬세한 열화가 있는 고해상도 이미지를 평가하기 위한 도전과제로 설정되었습니다. 대회 참가자들은 50G MACs의 계산 예산 내에서 높은 예측 성능을 목표로 새로운 아키텍처와 훈련 전략을 개발해야 합니다. 주요 평가 지표로는 Pearson Linear Correlation Coefficient (PLCC), Spearman Rank-order Correlation Coefficient (SRCC), Kendall Rank Correlation Coefficient (KRCC) 및 절대 오차 지표인 Mean Absolute Error (MAE)와 Root Mean Square Error (RMSE)가 사용됩니다.

- **Performance Highlights**: 주요 참가팀 중 SJTU는 전체 대회 우승을 차지하였고, SZU SongBai가 1위, CIPLAB이 2위로 뒤를 이었습니다. 성능 평가에서는 전반적으로 성능 지표의 감소가 나타났으며, 특히 CIPLAB은 특수 테스트 세트에서 2위를 기록했습니다.



### CDChat: A Large Multimodal Model for Remote Sensing Change Description (https://arxiv.org/abs/2409.16261)
- **What's New**: 이 논문에서는 리모트 센싱(REMOTE SENSING) 이미지의 변화를 설명할 수 있는 새로운 대화형 변환 모델인 CDChat을 제안합니다. CDChat은 기존의 GeoChat보다 향상된 성능을 보이며, bi-temporal RS 이미지 간의 변화 설명을 위한 새로운 데이터셋을 활용합니다.

- **Technical Details**: CDChat은 LLaVA-1.5와 같은 대형 언어 모델(LLM)을 기반으로 하며, 비디오 변환기(CLIP ViT-L-14)를 통해 bi-temporal 이미지를 처리합니다. 이 모델은 시암식 비전 인코더(Siamese vision encoder)를 활용하여 변경된 이미지의 특징을 별도로 추출하고 이를 결합하여 언어 공간에 투사합니다.

- **Performance Highlights**: CDChat은 기존의 LMM보다 개선된 성능을 보여주며, 특히 bi-temporal RS 이미지 간의 semantic 변화 탐지에 강점을 가지고 있습니다. 고해상도 이미지를 지원하여 작은 변화 지역에 대한 주의를 기울일 수 있는 능력을 강화했습니다.



### Fields of The World: A Machine Learning Benchmark Dataset For Global Agricultural Field Boundary Segmentation (https://arxiv.org/abs/2409.16252)
- **What's New**: 농업 모니터링 및 평가에서 중요한 역할을 하는 농작물 경계 데이터를 수집하는 비용을 절감하기 위해, 본 논문에서는 다양한 국가의 데이터를 포함한 새로운 기계 학습(Machine Learning, ML) 벤치마크 데이터셋인 'Fields of The World (FTW)'를 제안합니다.

- **Technical Details**: FTW 데이터셋은 24개국에서 수집된 70,462개의 샘플을 포함하며, 각 샘플은 다중 날짜, 다중 스펙트럴 Sentinel-2 위성 이미지와 함께 인스턴스 및 의미적 세분화 마스크가 쌍으로 제공됩니다. 이 데이터셋은 전 세계 농업 경관의 다양성을 반영하고 있으며 ML 모델의 성능을 향상시키기 위한 여러 기준 작업을 포함합니다.

- **Performance Highlights**: FTW 데이터셋으로 훈련된 모델은 다양한 국가에서 전이 학습 및 제로샷(Zero-shot) 성능이 우수하며, 실제 시나리오인 에티오피아의 Sentinel-2 장면에서 긍정적인 질적 성능을 보였습니다.



### Label-Augmented Dataset Distillation (https://arxiv.org/abs/2409.16239)
- **What's New**: 본 연구에서는 Label-Augmented Dataset Distillation (LADD)이라는 새로운 데이터셋 증류 프레임워크를 도입하였습니다. LADD는 라벨을 증강하여 데이터셋 증류를 개선하며, 이는 더 풍부한 의미를 포착하기 위해 각 합성 이미지에서 추가적인 밀집 라벨을 생성합니다.

- **Technical Details**: LADD는 두 가지 주요 단계로 이루어져 있습니다: 증류(distillation) 단계와 배포(deployment) 단계입니다. 증류 단계에서는 기존 증류 알고리즘을 사용하여 합성 이미지를 생성한 후 이미지 서브샘플링 알고리즘을 적용하여 각 합성 이미지에 대한 밀집 라벨을 생성합니다. 배포 단계에서는 글로벌 뷰 이미지와 원래 라벨, 그리고 로컬 뷰 이미지와 해당 밀집 라벨을 결합하여 다양한 학습 신호를 제공합니다.

- **Performance Highlights**: LADD는 기존 방법들보다 평균 14.9%의 정확도 향상을 달성했으며, 87% 적은 메모리를 사용하면서 5 IPC에서 6 IPC 기준을 지속적으로 초과했습니다. LADD는 또한 다양한 데이터셋과 모델 아키텍처에서 검증되었습니다.



### VideoPatchCore: An Effective Method to Memorize Normality for Video Anomaly Detection (https://arxiv.org/abs/2409.16225)
Comments:
          Accepted to ACCV 2024

- **What's New**: 영상 이상 탐지(Video Anomaly Detection, VAD) 분야에서 새로운 메모리 기반 접근법인 VideoPatchCore(VPC)를 제안합니다. VPC는 정상 프레임의 특징을 메모리에서 저장하고 재구성하여 비정상 프레임을 식별하는 기존 방법의 한계를 극복합니다.

- **Technical Details**: VPC는 두 개의 스트림(로컬 및 글로벌)과 세 가지 메모리 은행(공간적, 시간적, 고수준 의미론적)을 활용하여 영상 데이터의 시공간적 특성을 포착합니다. 이 방법은 PatchCore에서 영감을 얻었으며, CLIP의 비전 인코더를 활용하여 메모리를 최적화합니다. 메모리 최적화는 greedy coreset subsampling 방식을 통해 수행됩니다.

- **Performance Highlights**: VPC는 기존의 최첨단 방법과 비교해 훌륭한 성능을 보여주며, 다양한 형태의 이상 탐지가 가능합니다. 이 접근 방법은 추가 훈련 없이 구현이 용이하여 VAD 작업의 접근성을 높이고 있습니다.



### Deep Learning for Precision Agriculture: Post-Spraying Evaluation and Deposition Estimation (https://arxiv.org/abs/2409.16213)
- **What's New**: 본 논문은 정밀 농업에서의 정밀 스프레이 시스템을 평가하기 위한 자동화된 eXplainable Artificial Intelligence (XAI) 컴퓨터 비전 파이프라인을 제안합니다. 이 시스템은 전통적인 농업 방법 없이 포스트 스프레이 후 이미지를 평가할 수 있도록 설계되었습니다.

- **Technical Details**: 연구에서는 샘플로 사용하는 작물 및 잡초에 대해 의미론적 분할(semantic segmentation)을 수행하고, 각 작물이 스프레이되었는지를 식별할 수 있습니다. 이를 위해 Weakly Supervised Deposition Estimation (WSDE) 작업이 추가되어 클래스별 spray deposit 무게를 정확하게 정량화합니다. 데이터셋은 공개되어 있으며, 클래스 활성화 지도(Class Activation Mapping, CAM)를 사용하여 모델의 예측과 결합하여 스프레이 침착값을 도출합니다. 또한, Fully Convolutional Network와 EfficientNet-B0 백본 구조를 통해 성능을 최적화하고 의미론적 분할의 해석 가능성 또한 향상되었습니다.

- **Performance Highlights**: 시험 집합에서 세 클래스간에 스프레이 침착 값의 평균 절대 차이는 156.8 {\,}μL로 평가되었습니다. 이 연구에서는 AblationCAM과 ScoreCAM의 두 가지 다른 CAM 기법을 비교하여 각 기법의 효용과 해석 가능성을 평가했습니다.



### MaskBit: Embedding-free Image Generation via Bit Tokens (https://arxiv.org/abs/2409.16211)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 Masked transformer 모델을 활용한 이미지 생성 접근 방식의 발전을 다룬다. 주목할 만한 기여로는 기존 VQGAN 모델의 현대화와 비트 토큰(bit tokens)을 사용하는 새로운 생성 네트워크 MaskBit의 제안이 있다.

- **Technical Details**: VQGAN+라는 현대화된 VQGAN 모델은 성능을 매우 향상시켰으며, FID(Fréchet Inception Distance) 점수가 7.94에서 1.66으로 개선되었다. MaskBit은 비트 토큰을 직접 사용하여 이미지를 생성하는 새로운 방식을 도입하여, 305M 파라미터를 갖고도 ImageNet 256x256 벤치마크에서 FID 1.52를 달성하였다.

- **Performance Highlights**: 이 연구는 VQGAN+의 성능을 향상시킴으로써 기존의 최첨단 모델과 경쟁할 수 있는 성과를 달성했으며, MaskBit은 더욱 소형의 생성器 모델을 통해 가장 최신의 성과를 즉음으로 달성하였다.



### LLMCount: Enhancing Stationary mmWave Detection with Multimodal-LLM (https://arxiv.org/abs/2409.16209)
- **What's New**: LMMCount는 대형 언어 모델 (Large Language Models, LLM)을 활용하여 밀리미터파 감지기의 성능을 개선하는 최초의 시스템입니다. 이 시스템은 주변 군중을 비침습적이고 개인정보 보호 방식으로 감지할 수 있도록 개발되었습니다.

- **Technical Details**: LLMCount 시스템은 세 개의 모듈로 구성되어 있으며, 한 모듈은 로컬 처리 장치에서 실행되고 두 개의 모듈은 클라우드 기반 LLM으로 처리 에이전트로 배치됩니다. 기본적으로 IWR-1443 레이더 칩을 사용하여 mmWave 데이터를 수집하며, 이 데이터를 클라우드로 업로드하여 실시간으로 감지 결과를 제공합니다.

- **Performance Highlights**: LLMCount는 다양한 시나리오에서 높은 감지 정확성을 달성하였으며, 이전 방법들에 비해 낮은 전체 지연시간을 기록했습니다. 이는 다양한 환경에서의 적응성을 크게 향상시켰습니다.



### Segmentation Strategies in Deep Learning for Prostate Cancer Diagnosis: A Comparative Study of Mamba, SAM, and YOLO (https://arxiv.org/abs/2409.16205)
- **What's New**: 이 연구에서는 전립선 암의 조직병리학 이미지 세분화를 위한 3가지 딥러닝 기반 방법인 Mamba, SAM, YOLO의 비교 분석을 제시합니다.

- **Technical Details**: 연구에서는 Dice score, precision, recall 등의 메트릭을 사용하여 Gleason 2019와 SICAPv2 두 개의 포괄적인 데이터셋에서 이들 모델의 성능을 평가했습니다. H-vmunet 모델은 높은 차원의 시각 상태 공간과 2D 선택적 스캔 작업을 통합한 구조를 통해 다양한 규모의 병변 감지를 효율적이고 민감하게 수행할 수 있습니다.

- **Performance Highlights**: H-vmunet 모델이 모든 메트릭에서 가장 높은 점수를 달성하며, 전립선 암 진단과 치료 계획에서 중요한 역할을 할 수 있는 잠재력을 보여줍니다.



### Expert-level vision-language foundation model for real-world radiology and comprehensive evaluation (https://arxiv.org/abs/2409.16183)
- **What's New**: 이번 연구에서는 방사선학에 맞춘 대규모 오픈 소스 비전-언어(Vision-Language) 기반 모델인 RadFound를 소개합니다. RadFound는 810만 장 이상의 이미지와 25만 개의 이미지-텍스트 쌍으로 구성된 데이터셋을 사용하여 훈련되었습니다.

- **Technical Details**: RadFound는 방사선학에 특화된 고급 비전 인코더를 도입하여 이미지 내부의 로컬 특징과 이미지 간의 맥락 정보를 포착합니다. 또한, 방사선학에 맞춘 통합된 크로스 모달(Cross-modal) 학습 설계를 채택하고 있습니다. 이를 통해 의료 비전-언어 질문-응답, 캡셔닝(captioning), 리포트 생성과 같은 방사선 해석 작업을 포함한 기준을 설정했습니다.

- **Performance Highlights**: RadFound는 2D 이미지(흉부 X선), 다중 뷰 이미지(유방 촬영), 3D 이미지(갑상선 CT 스캔)와 같은 세 가지 대표적인 모달리티가 포함된 현실 세계 기준에서 평가받았으며, 다른 VL 기반 모델들에 비해 정량적 메트릭과 인간 평가 모두에서 현저한 성과를 보였습니다.



### SDFit: 3D Object Pose and Shape by Fitting a Morphable SDF to a Single Imag (https://arxiv.org/abs/2409.16178)
Comments:
          11 pages, 7 figures, 2 tables

- **What's New**: 이 논문에서는 단일 이미지를 통해 3D 객체의 자세(pose)와 형태(shape)를 복원하는 방법에 대한 새로운 접근법인 SDFit을 제시합니다. 이 방법은 기존의 한계점을 극복하려는 시도로, 모양과 자세를 동시에 추정할 수 있는 가능성을 보여줍니다.

- **Technical Details**: SDFit은 (1) 학습된 signed-distance-function (SDF) 모델을 기반으로 하여 강력한 변형 가능한 형태(morphable shape) 사전(prior)으로 작용하고, (2) 2D 이미지와 3D 형태를 공동 공간에 매핑할 수 있는 기초 모델(foundational models)을 활용하며, (3) 이미지로부터 풍부한 특징을 추론합니다. SDFit은 이미지로부터 3D 형태 가설을 생성하고, 해당 형태를 이미지와 비교하여 반복적으로 정교화하는 방식으로 작동합니다.

- **Performance Highlights**: SDFit은 Pix3D 및 Pascal3D+ 데이터셋에서 평가되었으며, 현대의 학습 기반 방법들과 유사한 성능을 보였습니다. 특히, 고유한 점은 SDFit이 미리 학습이 필요하지 않으며, 자연 이미지에 대한 일반화(generalization)에 강한 가능성을 보여주는 점입니다.



### Fine Tuning Text-to-Image Diffusion Models for Correcting Anomalous Images (https://arxiv.org/abs/2409.16174)
- **What's New**: 이 연구는 DreamBooth 기법을 사용하여 Stable Diffusion 3 모델을 미세 조정하는 방법을 제안합니다. 이 방법은 특정 프롬프트에 대해 생성된 비정상적인 이미지를 줄이는 것을 목표로 합니다.

- **Technical Details**: Stable Diffusion 3 모델은 DreamBooth 기법을 통해 추가 정보를 학습하여 이미지 생성을 개선합니다. LoRA(저랭크 적응) 기법을 이용하여 훈련 파라미터 수를 줄이면서 성능을 높일 수 있습니다. 실험에서는 SSIM(구조적 유사도 지수), PSNR(피크 신호 대 잡음 비율), FID(프레셰 관창 거리)와 같은 다양한 메트릭을 사용하여 성능을 평가하였습니다.

- **Performance Highlights**: 미세 조정된 Stable Diffusion 3 모델은 FID에서 266.5844를 기록하여 원래 모델의 366.9462보다 낮은 값을 보였습니다. SSIM에서는 0.2258로 원래 모델의 0.1387보다 개선되었습니다. PSNR은 23.2820 dB로 원래 모델 23.1765 dB보다 약간 더 높은 품질의 이미지를生成했습니다. 사용자 설문 조사에서는 미세 조정된 모델의 생성한 이미지가 보다 자연스러웠다는 의견이 대다수였습니다.



### MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling (https://arxiv.org/abs/2409.16160)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문에서는 사용자 입력에 의해 제어 가능한 속성(캐릭터, 동작 및 장면)을 가진 캐릭터 비디오를 합성할 수 있는 MIMO라는 새로운 프레임워크를 제안합니다. 이 방법은 다양한 캐릭터에 대해 고급 확장성 및 새로운 3D 동작의 일반성을 제공하며, 실시간 상호작용이 가능한 실제 장면에 적용됩니다.

- **Technical Details**: MIMO에서는 2D 비디오를 컴팩트한 공간 코드로 인코딩합니다. 구체적으로 단안 깊이 추정기를 사용하여 2D 프레임 픽셀을 3D로 변환 후, 동영상 클립을 3D 깊이를 기반으로 주요 인물, 하위 장면 및 부유하는 오클루전의 세 가지 공간 구성 요소로 분해합니다. 이 구성 요소들은 각각의 조정 신호로 사용되며, 이를 통해 복잡한 동작 표현과 사용자 제어를 가능하게 합니다.

- **Performance Highlights**: 실험 결과 MIMO는 다양한 속성을 제어할 수 있는 고품질 캐릭터 비디오 합성에서 효과성과 견고성을 입증했습니다. 이 방법은 또한 기존의 2D 방법들이 갖는 한계를 극복하고, 복잡한 3D 동작 및 물체 간 상호작용이 있는 실제 장면을 처리할 수 있는 가능성을 보여줍니다.



### ComiCap: A VLMs pipeline for dense captioning of Comic Panels (https://arxiv.org/abs/2409.16159)
Comments:
          Accepted at ECCV 2024 Workshop (AI for Visual Art), repo: this https URL

- **What's New**: 이 논문에서는 Vision-Language Models (VLMs)를 활용한 만화 패널의 밀집 캡션 생성 파이프라인을 제안합니다. 기존의 VLM을 추가 훈련 없이 활용하며, 중요한 속성을 고려하는 두 단계의 메트릭을 개발하여 모델의 성능을 평가합니다.

- **Technical Details**: 제안된 파이프라인은 자동 핵심 요소 추출 및 BERT-score 평가를 기반으로 한 두 단계의 메트릭을 사용하여 VLM이 제공하는 캡션의 속성을 검토합니다. 또한 1,500개의 패널에 대한 캡션과 속성 목록이 주석 처리되어 있으며, 이를 기반으로 하는 벤치마크 데이터를 사용하여 기존 오픈 소스 VLM의 성능을 평가합니다.

- **Performance Highlights**: 본 파이프라인을 통해 2백만 개 이상의 패널이 주석 처리되어, 고급 캡션이 제공됩니다. 밀집 캡션은 훈련된 특정 모델이 생성한 캡션보다 정량적 및 정성적으로 우수한 결과를 보여줍니다.



### MCTrack: A Unified 3D Multi-Object Tracking Framework for Autonomous Driving (https://arxiv.org/abs/2409.16149)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 논문에서는 MCTrack이라는 새로운 3D 다중 객체 추적 방법을 소개하며, KITTI, nuScenes, Waymo 데이터셋 전반에서 최신 성능(SOTA)을 달성하였습니다.

- **Technical Details**: MCTrack은 기존의 데이터셋 간 일반화의 격차를 해소하며, BaseVersion이라는 표준화된 결과 포맷을 제안하여 연구 진영에서 데이터 전처리 부담을 줄일 수 있게 합니다. 이 방법은 BEV(각도-위치) 평면에서의 1차 매칭과 이미지 평면에서의 2차 매칭을 포함한 두 단계의 매칭 전략을 사용하여 정확성을 높입니다.

- **Performance Highlights**: MCTrack은 KITTI 및 nuScenes 데이터셋에서 1위를, Waymo 데이터셋에서 2위를 차지하며, 특히 실제 엔지니어링 응용 프로그램을 염두에 두고 설계되었습니다. 또한, 움직임 관련 정보(속도, 가속도, 각속도)를 평가하는 새로운 메트릭스를 도입하여 다중 객체 추적 작업 후의 모션 정보를 적절히 전달하는 데 중점을 두었습니다.



### Gaussian D\'ej\`a-vu: Creating Controllable 3D Gaussian Head-Avatars with Enhanced Generalization and Personalization Abilities (https://arxiv.org/abs/2409.16147)
Comments:
          11 pages, Accepted by WACV 2025 in Round 1

- **What's New**: Gaussian Déjà-vu 프레임워크를 소개하여, 3D Gaussian Splatting을 기반으로 한 개인화된 3D 헤드 아바타 생성 시간을 단축합니다. 기존의 방법들은 수십 분에서 몇 시간까지 걸리던 과정을 수분으로 줄이는 것을 목표로 합니다.

- **Technical Details**: 이 프레임워크는 대규모 2D 이미지 데이터셋을 통해 훈련된 일반화된 모델을 이용하여 초기 3D Gaussian 헤드를 생성한 후, 단안을 통해 개인화하는 과정을 포함합니다. 개인화 단계에서는 학습 가능한 expression-aware rectification blendmaps를 통해 초기 3D Gaussian을 수정하고, 신경망을 사용하지 않고도 빠른 수렴을 이룹니다.

- **Performance Highlights**: 이 방법은 기존의 3D Gaussian 헤드 아바타와 비교하여 포토리얼리스틱 품질에서 우수한 성능을 보여주며, 훈련 시간을 최소한 4분의 1로 줄여 아바타를 몇 분 안에 생성할 수 있습니다.



### Learning to Localize Actions in Instructional Videos with LLM-Based Multi-Pathway Text-Video Alignmen (https://arxiv.org/abs/2409.16145)
Comments:
          Accepted to ECCV 2024

- **What's New**: 본 연구는 교육 비디오에서 절차 단계를 지역화하는 새로운 훈련 프레임워크를 제안합니다. 특히, Large Language Models (LLM)을 활용하여 작업에 관련 없는 정보를 필터링하고, LLM을 통해 요약된 단계 문장을 기반으로 신뢰할 수 있는 대응 관계를 생성하기 위한 Multi-Pathway Text-Video Alignment (MPTVA) 전략을 도입했습니다.

- **Technical Details**: 제안된 MPTVA 전략은 (1) 내레이션 타임스탬프를 이용한 단계-내레이션-비디오 정합, (2) 장기적인 의미 유사성 기반의 직접적인 단계-비디오 정합, (3) 다양한 비디오 도메인에서 학습된 짧은 텍스트-비디오 정합 모델을 통한 단계-비디오 정합을 포함하는 세 가지 경로에서 정합을 측정합니다. 이를 통해 LLM 단계와 비디오 간의 신뢰성 있는 가상 정합을 생성합니다.

- **Performance Highlights**: 제안된 접근법은 절차 단계 정렬, 단계 지역화 및 내레이션 정렬의 세 가지 하위 작업에서 기존 최첨단 기술들을 5.9%, 3.1%, 2.8% 각각 초과하는 성과를 보여주었습니다. LLM 단계로 훈련된 모델이 wikiHow 단계로 훈련된 모델에 비해 10.7% 더 나은 성능을 보인 점이 주목할 만합니다.



### Seeing Faces in Things: A Model and Dataset for Pareidolia (https://arxiv.org/abs/2409.16143)
- **What's New**: 본 연구에서는 인간과 머신 간의 face pareidolia (얼굴 패레이돌리아)에 대한 인식 차이를 조사하기 위해 새로운 데이터셋인 'Faces in Things'를 소개합니다. 이 데이터셋은 무작위로 생성된 이미지에서 인간이 인식한 얼굴 구조를 포함하고 있습니다.

- **Technical Details**: 이 연구는 5,000개의 웹 이미지로 구성된 'Faces in Things' 데이터셋을 사용하여 인간 얼굴 탐지 시스템의 성능을 분석합니다. 연구 결과는 최신 연구 모델인 RetinaFace를 사용하여 성과를 변별하며, 파리돌리아가 머신에서 어떻게 나타나는지를 탐구합니다.

- **Performance Highlights**: 최신 모델은 얼굴 패레이돌리아 탐지에서 인간의 성능에 비해 상당한 격차를 보였습니다. 연구는 이 격차의 약 절반이 동물 얼굴 탐지 모델을 미세 조정하는 것에서 개선될 수 있음을 보여줍니다. 또한, 'Goldilocks zone'이라고 불리는 조건들이 패레이돌리아를 유도할 수 있음을 실험으로 확인하였습니다.



### HA-FGOVD: Highlighting Fine-grained Attributes via Explicit Linear Composition for Open-Vocabulary Object Detection (https://arxiv.org/abs/2409.16136)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 논문은 Open-Vocabulary Object Detection (OVD) 모델에서 세부 속성을 강조하는 새로운 접근 방식을 제안하여 기존 모델의 성능을 향상시키는 방법을 소개합니다.

- **Technical Details**: 이 방법은 1) Attribute Word Extraction, 2) Attribute Feature Extraction, 3) Attribute Feature Enhancement의 세 가지 주요 프로세스로 구성됩니다. 강력한 LLM(대규모 언어 모델)을 이용해 입력 텍스트에서 속성 단어를 추출하고, 전략적으로 토큰 마스크를 조정하여 OVD 모델의 텍스트 인코더가 전역 텍스트와 속성 특정 피처를 추출합니다. 이 피처들은 선형 조합을 통해 새로운 속성 강조 피쳐로 통합됩니다.

- **Performance Highlights**: FG-OVD 데이터셋에서 실험한 결과, 제안된 방법이 다양한 OVD 모델의 세부 속성 인식 능력을 일관되게 향상시키며 새로운 최첨단 성능을 달성함을 입증하였습니다.



### VisioPhysioENet: Multimodal Engagement Detection using Visual and Physiological Signals (https://arxiv.org/abs/2409.16126)
Comments:
          5 Pages, 2 figures

- **What's New**: 이번 논문에서는 Learner engagement(학습자 참여)을 감지하기 위한 VisioPhysioENet이라는 새로운 멀티모달 시스템을 제안합니다. 이 시스템은 visual cues(시각적 신호)와 physiological signals(생리 신호)를 활용하여 참여도를 탐지하며, Dlib 라이브러리를 사용한 얼굴 랜드마크 추출 및 OpenCV 라이브러리를 통한 추가 평가를 통해 시각적 특성을 두 단계로 추출합니다.

- **Technical Details**: VisioPhysioENet는 facial landmark(얼굴 랜드마크) 추출을 위해 Dlib을 활용하고, Eye Aspect Ratio (EAR), Pitch(피치), Yaw(요), Roll(롤)과 같은 지표를 포함하는 비주얼 피쳐를 처리합니다. 또한, remote photoplethysmography (rPPG) 신호를 비디오 입력을 통해 캡처하여 심혈관 활동을 모니터링합니다. 이 시스템은 multi-output classifiers(다중 출력 분류기)와 late fusion techniques(후처리 융합 기법)을 활용하여 다양한 참여 수준을 탐지하는 데 있어 정확성을 높입니다.

- **Performance Highlights**: DAiSEE 데이터 세트에서 철저한 평가를 수행한 결과, VisioPhysioENet는 63.09%의 정확도를 달성하여 기존 방법론보다 다양한 참여 수준을 식별하는 데 있어 우수한 능력을 보였습니다.



### Neuromorphic Drone Detection: an Event-RGB Multimodal Approach (https://arxiv.org/abs/2409.16099)
Comments:
          Accepted at NeVi Workshop at ECCV24

- **What's New**: 이번 연구에서는 드론 감지를 위한 새로운 모델을 제안하며, Neuromorphic 데이터와 RGB 데이터를 효과적으로 융합하여 정확한 탐지를 위한 멀티모달 접근 방식을 탐구합니다. 또한, NeRDD(Neuromorphic-RGB Drone Detection)라는 새로운 데이터셋을 공개하여 3.5시간 이상의 주석이 달린 멀티모달 녹화를 제공합니다.

- **Technical Details**: Neuromorphic 카메라는 전통적인 RGB 카메라에 비해 높은 속도 및 변화하는 조명 조건에서 뛰어난 성능을 보여줍니다. 이 연구에서는 스파이킹 네트워크와 같은 다양한 신경망 아키텍처를 조합하여 드론 탐지 정확도를 향상시킵니다. 또한, 두 데이터 스트림을 융합하는 다양한 전략을 비교하여 성능 최적화를 도모합니다.

- **Performance Highlights**: 실험 결과에 따르면, Neuromorphic 카메라와 RGB 데이터의 조합은 각각 분리된 경우보다 드론 탐지율을 더욱 향상시킵니다. NeRDD 데이터셋의 사용으로 드론 탐지의 정확성이 크게 증가했음을 확인하였습니다.



### From Pixels to Words: Leveraging Explainability in Face Recognition through Interactive Natural Language Processing (https://arxiv.org/abs/2409.16089)
- **What's New**: 본 논문에서는 Face Recognition (FR) 모델의 해석 가능성(transformability)을 높이기 위해 모델 불가지론적 설명 가능한 인공지능(Explainable Artificial Intelligence, XAI)과 자연어 처리(Natural Language Processing, NLP) 기술을 결합한 상호작용형 프레임워크를 제안합니다. 이 프레임워크는 사용자와의 대화를 통해 다양한 질문에 정확하게 응답할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는, 프레임워크의 각 모듈에서 사용되는 기술의 세부사항을 포함하여 3개의 주요 모듈로 구성됩니다: (i) FR 시스템 및 신뢰도 추정, (ii) 설명 가능성 방법, (iii) NLP 기반의 사용자 친화적 질문-응답(QA) 인터페이스입니다. 이 시스템은 ArcFace 모델을 사용하여 얼굴 이미지를 비교하고, Probabilistic Interpretable Comparison (PIC) 스코어를 통해 유사성과 신뢰도를 평가합니다.

- **Performance Highlights**: 제안된 방법은 다양한 실험을 통해 FR 시스템 성능을 저하시키지 않으면서도 해석 가능성을 향상시키는 효과를 입증했습니다. 또한, 자가 분류에서의 사용자 질문을 통해 보다 정확한 정보를 제공하고, 민감한 애플리케이션에서의 의사 결정 투명성을 추가로 강화할 수 있습니다.



### MM-CamObj: A Comprehensive Multimodal Dataset for Camouflaged Object Scenarios (https://arxiv.org/abs/2409.16084)
Comments:
          9 pages, 5 figures. Work in progress

- **What's New**: 이 논문에서는 MM-CamObj 데이터셋을 새롭게 구축하였으며, 특히 위장된(camouflaged) 객체와 장면을 다루는 LVLM(CamObj-Llava)을 제안합니다.

- **Technical Details**: MM-CamObj 데이터셋은 두 개의 하위 집합으로 나뉘어 있으며, CamObj-Align는 VL 정렬(VL alignment)을 위한 11,363개의 이미지-텍스트 쌍을 포함하고, CamObj-Instruct는 LVLM을 위한 매뉴얼을 따르는 성능 개선을 목적으로 다양한 지시와 함께 68,849개의 대화를 포함합니다. 학습 전략으로는 curriculum learning이 도입되어, 난이도에 따라 자료를 제공하며, CamObj-Bench는 위장 JPEG 작업을 평가하는 기준으로 600개의 이미지와 7개의 작업을 포함합니다.

- **Performance Highlights**: CamObj-Llava는 CamObj-Bench에서 7개 작업의 4개에서 GPT-4o 대비 25.84%의 성능 향상을 보여주며, 위장 장면에 대한 이해, 인식, 위치 측정 및 개수를 정확히 수행하는 데 있어 뛰어난 성능을 입증합니다.



### GS-Net: Global Self-Attention Guided CNN for Multi-Stage Glaucoma Classification (https://arxiv.org/abs/2409.16082)
Comments:
          5 pages, 3 figures

- **What's New**: 본 논문은 다단계 녹내장(Glaucoma) 분류를 위한 새로운 네트워크인 GS-Net을 제안합니다. GS-Net은 글로벌 자기 주의(attention) 모듈인 GSAM을 도입하여 fundus 이미지에서 더 많은 구별 가능한 특징을 추출하도록 설계되었습니다.

- **Technical Details**: GS-Net은 백본 네트워크(backbone network), 글로벌 자기 주의 모듈(GSAM), 및 분류기로 구성됩니다. GSAM은 채널 주의 모듈(CAM)과 공간 주의 모듈(SAM) 두 개의 병렬 모듈로 이루어져 있으며, 이들은 각각 채널과 공간 차원에서의 글로벌 특징 종속성(global feature dependencies)을 학습합니다.

- **Performance Highlights**: 실험 결과 GS-Net은 기존의 최첨단 방법들보다 우수한 성능을 보이며, GSAM은 인기 있는 자기 주의 모듈들과 비교했을 때 경쟁력 있는 성능을 발휘합니다.



### Open-World Object Detection with Instance Representation Learning (https://arxiv.org/abs/2409.16073)
Comments:
          Our project website can be found at this https URL

- **What's New**: 본 논문에서는 Open World에서 객체 탐지(Object Detection) 문제를 해결하기 위해 Vision Foundation Models(VFM)의 지식을 활용한 새로운 방법을 제안합니다. 기존의 OWOD 방법들이 탐지된 객체 간의 세밀한 관계를 포착하지 못하는 문제를 해결하고자 합니다.

- **Technical Details**: 본 방법은 두 가지 모듈인 Unknown Box Refine Module과 Embedding Transfer Module을 통해 작동합니다. Unknown Box Refine Module은 Segment Anything Model(SAM)에서 생성된 세그멘테이션 마스크를 활용하여 미지의 객체의 바운딩 박스를 보정합니다. Embedding Transfer Module은 VFM의 픽셀 레벨 특징에서 얻은 인스턴스 간의 유사성을 탐지기의 인스턴스 임베딩에 전이하여 특징 공간을 향상시킵니다.

- **Performance Highlights**: 이 논문에서 제안한 방법은 OWOD 기반의 다른 방법들보다 뛰어난 성능을 보이며, 미지의 객체 탐지 성능과 임베딩된 특징 공간의 품질을 높이고 있습니다. 또한, 개방형 세계 추적(open-world tracking) 작업에서 세밀한 특징을 학습한 결과로 적용 가능성이 증대되었음을 보여줍니다.



### Machine learning approaches for automatic defect detection in photovoltaic systems (https://arxiv.org/abs/2409.16069)
Comments:
          31 pages, 14 figures

- **What's New**: 본 연구는 태양광(PV) 모듈의 결함 감지에 대한 최신 딥러닝 기반 컴퓨터 비전 기술을 검토하고 분석합니다. 드론 이미징을 통해 태양전지 패널의 결함을 실시간으로 분석할 수 있는 AI 모델 개발에 중점을 둡니다.

- **Technical Details**: 이 논문에서는 드론을 이용한 이미지 캡처 방법을 사용하여 IR 이미지, EL 이미지 및 RGB 이미지의 세 가지 유형을 기반으로 효율적으로 결함을 감지하는 다양한 접근법을 비교합니다. 또한, CNN을 포함한 딥러닝 아키텍처와 데이터 증강(data augmentation) 또는 생성적 적대 네트워크(generative adversarial networks) 기술을 결합하여 이루어진 기존의 다양한 방법론에 대해서도 논의합니다.

- **Performance Highlights**: 모델 해석 가능성 분석을 수행한 결과, 결함 분류를 위해 이미지의 어두운 영역에 초점을 맞추고 있음이 밝혀졌습니다. 결함 감지의 정확성을 높이기 위해 기하학적 딥러닝 기법을 기존 접근법과 통합하거나, 물리 법칙에 기반한 신경망을 활용하는 방법을 제안합니다.



### Benchmarking Robustness of Endoscopic Depth Estimation with Synthetically Corrupted Data (https://arxiv.org/abs/2409.16063)
Comments:
          To appear at the Simulation and Synthesis in Medical Imaging (SASHIMI) workshop at MICCAI 2024

- **What's New**: 이 연구에서 제안된 Depth Estimation Robustness Score (DERS)는 외과적 환경에서의 깊이 추정 모델의 강건성을 평가하기 위한 새로운 기준을 설정합니다. 이 평가는 수술 중 발생할 수 있는 이미지 왜곡을 반영하기 위한 포괄적인 데이터셋을 기반으로 합니다.

- **Technical Details**: 제안된 DERS는 에러 (Error), 정확도 (Accuracy) 및 강건성 (Robustness)을 결합하여 깊이 추정 기술의 성능을 평가하기 위한 복합 지표입니다. 연구는 조명 변화, 시각적 장애물, 센서 노이즈 등 모델의 성능에 영향을 미치는 다양한 왜곡 유형을 포함합니다.

- **Performance Highlights**: 두 개의 단안 (Monocular) 깊이 추정 모델을 통한 실험 결과는 실제 환경에서의 알고리즘 신뢰성을 강조하였으며, 데이터 저하 (Data Corruption)에 강한 알고리즘의 필요성을 부각시켰습니다. 연구는 수술 정밀도 및 환자 안전 개선에 기여하는 실질적인 결과를 도출하였습니다.



### Generative 3D Cardiac Shape Modelling for In-Silico Trials (https://arxiv.org/abs/2409.16058)
Comments:
          EFMI Special Topic Conference 2024

- **What's New**: 본 논문에서는 심층 학습(deep learning) 방법을 통해 인공 대동맥(aortic) 형태를 모델링하고 생성하는 기법을 제안합니다. 이 방법은 형태를 신경 서명 거리 필드(neural signed distance field)의 영(零) 수준 집합(zero-level set)으로 표현하고, 이 형태들의 기하학적 특징을 인코딩하는 학습 가능한 임베딩 벡터(embedding vectors)에 의해 조건화합니다.

- **Technical Details**: 네트워크는 CT 이미지로 재구성된 대동맥 뿌리(aortic root) 메쉬(mesh) 데이터셋을 기반으로 훈련되며, 신경 필드(neural field)가 샘플링된 표면 점(surface points)에서 사라지도록 하고, 공간 기울기(spatial gradient)가 단위 노름(unit norm)을 가지도록 강제합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 대동맥 형태를 높은 충실도(high fidelity)로 표현할 수 있으며, 학습된 임베딩 벡터에서 샘플링하여 실제 환자의 해부학을 닮은 새로운 형태를 생성할 수 있습니다. 이러한 형태는 인실리코 시험(in-silico trials)에 활용될 수 있습니다.



### Towards Robust Object Detection: Identifying and Removing Backdoors via Module Inconsistency Analysis (https://arxiv.org/abs/2409.16057)
- **What's New**: 본 논문은 두 단계 객체 탐지 모델에서의 백도어(Backdoor) 탐지 및 제거 문제를 다룬 최초의 접근법으로, 객체 탐지 모델의 독특한 특성에 맞춘 새로운 백도어 방어 프레임워크를 제안합니다.

- **Technical Details**: 제안된 백도어 탐지 방법은 객체 탐지 모델의 두 주요 구성 요소인 Region Proposal Network (RPN)과 Region Classification Network (R-CNN) 간의 예측 불일치를 정량화하고 분석하여 백도어의 존재를 확인합니다. 제안된 백도어 제거 전략은 특정 모듈에 대한 재초기화와 소량의 깨끗한 데이터에 대한 전체 모델의 미세 조정을 포함합니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, 제안된 방법은 백도어 제거율에서 기존 방법에 비해 약 90% 개선을 달성하였으며, 깨끗한 데이터에 대한 정확도 손실은 4% 미만으로 제한되었습니다.



### Adversarial Watermarking for Face Recognition (https://arxiv.org/abs/2409.16056)
- **What's New**: 본 연구는 얼굴 인식 시스템에서 워터마킹(watermarking)과 적대적 공격(adversarial attacks) 간의 상호작용을 탐구하며, 적대적 워터마킹 공격(adversarial watermarking attack)이라는 새로운 위협 모델을 소개합니다.

- **Technical Details**: 워터마킹은 디지털 이미지를 통해 소유권을 주장하고 무단 변경을 모니터링하는 데 필수적인 기술입니다. 얼굴 인식 시스템에서 워터마킹은 데이터 무결성과 보안을 보장하는 데 중요한 역할을 하지만, 공격자는 워터마킹 프로세스에 간섭하여 인식 성능을 심각하게 저하시킬 수 있습니다. 본 연구는 CASIA-WebFace 데이터셋을 통해 적대적 워터마킹 공격이 얼굴 매칭 정확성을 최대 67.2%까지 감소시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 적대적 워터마킹 공격의 적용으로 인해 워터마킹이 없는 상태에서는 이미지가 정확하게 인식되지만, 워터마킹이 적용된 후에는 인식 실패를 유발하는 중요한 취약점이 발견되었습니다.



### Unleashing the Potential of Synthetic Images: A Study on Histopathology Image Classification (https://arxiv.org/abs/2409.16002)
Comments:
          Accepted at ECCV 2024 - BioImage Computing Workshop

- **What's New**: 이 연구는 히스토패스올로지(Histopathology) 이미지를 위한 새로운 합성 데이터 생성 방법을 탐구합니다. 특히, Diffusion 모델을 사용하여 이미지 패치를 생성하고, 이를 통해 기존 데이터 세트를 보강하여 분류 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 연구에서는 Denoising Diffusion Probabilistic Models (DDPM)을 활용하여 클래스 라벨에 조건화된 합성 히스토패스올로지 이미지 패치를 생성합니다. 실험 결과에 따르면, Diffusion 모델은 전이 학습(Transfer Learning)에 효과적이며, GAN(Generative Adversarial Networks)으로 생성된 샘플은 데이터 증강(Augmentation)에 적합합니다. 또한, Transformer 기반의 생성 모델은 CNN(Convolutional Neural Networks) 기반 모델보다 이미지 필터링이 필요하지 않습니다.

- **Performance Highlights**: PCam 데이터 세트에 대한 실험 결과, 합성 이미지는 기존 데이터 세트를 효과적으로 보강하고, 히스토패스올로지 이미지 분류 작업의 성능을 향상시키는 데 기여합니다. 제안된 방법은 분석 모델과 최근 GAN 기반 합성 데이터 확장 방식 대비 분류 성능을 개선했습니다.



### Improvements to SDXL in NovelAI Diffusion V3 (https://arxiv.org/abs/2409.15997)
Comments:
          14 pages, 8 figures

- **What's New**: NovelAI Diffusion V3 모델은 SDXL을 기반으로 하며, 훈련 관행에서 여러 가지 향상을 이루었습니다. 특히 Zero Terminal SNR과 v-prediction 파라미터화를 도입했습니다.

- **Technical Details**: SDXL의 훈련 과정에서 𝜖-prediction에서 v-prediction으로 전환하였고, 노이즈 스케줄을 개선하여 더 높은 시그마 레벨까지 훈련하였습니다. 이를 통해 모델은 노이즈로부터 의미 있는 색상과 주파수를 예측하도록 학습하게 되었습니다.

- **Performance Highlights**: 모델의 훈련 진행을 통해 고해상도 이미지 생성에서 일관성을 회복하였으며, 실제 이미지의 품질 개선과 더불어 수렴 속도가 빨라졌습니다.



### Leveraging Unsupervised Learning for Cost-Effective Visual Anomaly Detection (https://arxiv.org/abs/2409.15980)
- **What's New**: 이 연구는 사전 훈련된 모델을 사용하고 저렴한 하드웨어로 시각적 이상 탐지 시스템을 개발하여 중소기업(SMEs)의 구매 부담을 줄이기 위한 새로운 접근 방식을 제안합니다. 이 시스템은 최소한의 데이터로 모델 훈련을 수행할 수 있으며, Raspberry Pi 4B에서 효율적으로 배포됩니다.

- **Technical Details**: 이 시스템은 Anomalib 라이브러리의 비지도 학습 모델을 활용하여 작동합니다. 10장의 정상 제품 이미지만을 사용하여 Raspberry Pi에서 90초 만에 이상 탐지 교육과 추론을 완료할 수 있으며, F1 매크로 점수는 0.95 이상을 기록합니다. PaDiM, PatchCore, CFlow-AD 및 FastFlow와 같은 여러 알고리즘이 적용되어 성능을 비교했습니다.

- **Performance Highlights**: 연구 결과, 이 저비용의 시각적 이상 탐지 시스템은 환경 변화에 약간 민감하지만, 중소 제조업체를 위한 공장 자동화 검사의 신속하고 경제적인 방법으로써의 가능성을 보여주었습니다. 시스템은 IoT(Internet of Things) 환경에서 효율적인 운영을 위한 샘플링과 같은 높은 성능을 유지합니다.



### Adversarial Backdoor Defense in CLIP (https://arxiv.org/abs/2409.15968)
- **What's New**: 이번 연구에서는 기존의 backdoor 공격 방어 방법들이 효과적이지 않은 이유를 조사하고, Adversarial Backdoor Defense (ABD)라는 새로운 데이터 증강(Data Augmentation) 방법을 제안합니다. ABD는 adversarial examples와 backdoor samples의 특징을 정교하게 맞추어 backdoor 공격을 효과적으로 무력화합니다.

- **Technical Details**: ABD는 기존의 데이터 증강 방법의 한계를 보완하기 위해, adversarial examples를 생성하고 이를 데이터 증강 과정에서 활용합니다. 이는 InfoNCE 손실 함수를 통해 모델의 피처 공간(feature space)에서 backdoor 특성과 유사한 주의를 생성합니다. 이러한 방식을 통해 모델은 악의적 입력에 더 잘 대응할 수 있도록 훈련됩니다.

- **Performance Highlights**: ABD는 BadNet에 대해 공격 성공률(Attack Success Rate, ASR)을 8.66%, Blended 공격을 10.52%, BadCLIP 공격에 대해 53.64%로 줄이는 성과를 보여주었습니다. 또한, clean accuracy의 평균 감소는 단 1.73%로 유지되었습니다.



### Semantics-Controlled Gaussian Splatting for Outdoor Scene Reconstruction and Rendering in Virtual Reality (https://arxiv.org/abs/2409.15959)
- **What's New**: 본 연구에서는 Semantics-Controlled GS (SCGS)라는 새로운 접근 방식을 제안하고, 이를 통해 비통제된 자연 환경에서 대규모 장면을 분리할 수 있게 되었습니다. 이 방법은 VR에서 장면 편집과 장면 파트 추출을 가능하게 합니다.

- **Technical Details**: SCGS는 Gaussian rasterization 과정을 수정하여 3D Gaussians를 2D 이미지 공간 및 3D Gaussian 공간에서 거의 동일한 품질로 분류합니다. 이를 통해 전체 클래스를 대규모로 제거할 수 있으며, SCGS는 기존의 'circling' 방식에 의존하지 않습니다. 또한, 새로운 야외 데이터셋을 도입하여 사용자 경험을 향상시킵니다.

- **Performance Highlights**: 당사는 제안된 방법이 기존 기술보다 시각 품질 및 3D-OVS 데이터셋에서의 세분화 품질 면에서 우수하다는 것을 입증했습니다. 사용자 연구 결과, 참가자들은 plain GS에 비해 SCGS를 선호하는 경향을 보였습니다.



### An ensemble framework approach of hybrid Quantum convolutional neural networks for classification of breast cancer images (https://arxiv.org/abs/2409.15958)
Comments:
          Accepted in the 3rd International Conference on Data Electronics and Computing

- **What's New**: 이번 연구에서는 Quantum Neural Network (QNN)와 혼합형 클래식-양자 신경망 아키텍처의 효과를 평가하여, 기존의 클래스 특징 추출과 결합한 새로운 하이브리드 모델을 제안하고 breast cancer histopathological dataset을 이용한 실험 결과를 제시했습니다.

- **Technical Details**: 연구에서는 세 가지의 하이브리드 Quantum Convolutional Neural Network (QCNN) 아키텍처를 개발하고, 각 아키텍처는 서로 다른 네트워크 구조를 가집니다. 이 모델들은 양자 회로를 통해 최종 클래스 확률을 계산하며, 전통적인 CNN을 통해 추출된 특성을 사용합니다. 연구 결과, 개별 모델의 정확도는 85.59%였으며, 앙상블 기법을 통해 86.72%로 개선되었습니다.

- **Performance Highlights**: 하이브리드 모델의 조합을 통한 앙상블 기법이 기존의 개별 QNN과 전통적인 신경망보다 우수한 성능을 보여주었습니다. 이는 양자-클래식 혼합 방식의 신경망이 의료 이미지 분류 작업에서 효과적으로 적용 가능하다는 것을 증명합니다.



### Mind the Prompt: A Novel Benchmark for Prompt-based Class-Agnostic Counting (https://arxiv.org/abs/2409.15953)
- **What's New**: 이 논문에서는 Class-Agnostic Counting (CAC)에 대한 새로운 평가 벤치마크인 Prompt-Aware Counting (PrACo)을 소개하여, 과거의 데이터세트와 메트릭스의 한계를 극복하고자 합니다. PrACo는 텍스트 프롬프트를 통해 수량을 카운팅하는 모델을 평가하기 위한 메트릭스를 제공합니다.

- **Technical Details**: PrACo 벤치마크는 두 가지 테스트로 구성되어 있습니다: (i) negative-label test, 이는 특정 클래스가 없는 이미지에서 프롬프트를 사용하여 테스트하고, (ii) mosaic test, 이는 두 개의 서로 다른 클래스가 포함된 인위적으로 만든 이미지에서 하나의 클래스만 설명하는 프롬프트를 사용하여 평가합니다. 이는 현재의 CAC 데이터셋의 부족함과 기존의 측정 지표의 한계를 보완하기 위해 설계되었습니다.

- **Performance Highlights**: 최신 SOTA(prompt-based CAC) 기술들을 평가한 결과, 일부 모델이 표준 클래스별 카운팅 메트릭에서 높은 성능을 보이지만, 프롬프트에서 설명하는 객체 클래스를 이해하는 데 있어 유의미한 결핍을 드러내어 더 세밀한 훈련 절차나 설계 수정이 필요함을 강조하였습니다.



### A Formalization of Image Vectorization by Region Merging (https://arxiv.org/abs/2409.15940)
- **What's New**: 이 논문에서는 이미지 벡터화(image vectorization)가 이미지 분할(image segmentation)이라는 점을 강조하며, 세분화에서 조합되는 방식으로 발전할 수 있음을 지적합니다. 또한, 기존 방식의 한계를 다루기 위해 영역 병합(region merging)과 곡선 평활화(curve smoothing)를 교대로 적용하는 새로운 벡터화 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 서로 다른 도메인 파라미터에 의해 유도된 이중 그래프(dual graph)와 원래 그래프(primal graph)에서의 교차 작업을 포함합니다. 이 방법은 또한 지역 정보의 갱신과 곡선 근사화를 분리하던 기존의 한계를 해결하고, Beaulieu-Goldberg 및 Mumford-Shah 함수와 같은 다양한 이득 함수(gain functionals)와 연관시켜 formalize합니다.

- **Performance Highlights**: 실험 결과, 제안된 벡터화 방법은 최신 소프트웨어와 비교해 동등하거나 우수한 충실도(fidelity)와 비용 효율을 보여주었습니다. 이 기술은 직관적인 몇 가지 파라미터에 의해서도 명확한 행동을 보이며, 비트맵에서 벡터 그래픽으로의 효율적인 변환을 가능하게 합니다.



### Self-supervised Shape Completion via Involution and Implicit Correspondences (https://arxiv.org/abs/2409.15939)
Comments:
          ECCV 2024

- **What's New**: 이 논문은 자가 감독(self-supervised) 학습 접근 방식을 사용하여 3D 형태 보완(shape completion) 작업을 수행하는 방법을 제안합니다. 특별한 점은 보완 함수(G)가 자기 역함수(involutory function)로 표현될 수 있어, 이는 G(G(X)) = X라는 제약을 의미합니다.

- **Technical Details**: 제안된 방법에서는 내용의 일관성(consistency measure)을 캔노니컬 스페이스(canonical space)에서 정의하여 보완 함수를 감독하고 'freeze and alternate' 전략을 활용하여 보완 및 대응 모듈을 최적화합니다. 또한, 불완전한(불완전한) 점 집합을 정의하기 위해 'Unsigned Distance Field (UDF)' 표현을 사용하여 점 정보의 밀도를 높이고, 부분 형태를 완성하는 데 필요한 자체 감독 손실을 도출합니다.

- **Performance Highlights**: 이 방법은 강체 형태(rigid shapes)와 비강체 형태(non-rigid shapes) 모두에서 우수한 성과를 보이며, 몇 가지 경우에 대해 감독 학습(supervised accuracy)에 근접하는 놀라운 정확도를 달성합니다.



### Automatic Registration of SHG and H&E Images with Feature-based Initial Alignment and Intensity-based Instance Optimization: Contribution to the COMULIS Challeng (https://arxiv.org/abs/2409.15931)
- **What's New**: 이 연구에서는 비침습적 두 번째 조화 생성 (second-harmonic generation, SHG) 이미지와 헤마톡 실린 (hematoxylin and eosin, H&E) 이미지를 자동으로 정합하는 새로운 방법을 제안합니다. 이 방법은 훈련 없이 자동 키포인트 매칭과 인스턴스 최적화를 통한 변형 정합(deformable registration)을 기반으로 합니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 구성됩니다: (i) 전처리, (ii) SuperPoint 및 SuperGlue를 이용한 특징 기반의 초기 정합, (iii) 인스턴스 최적화를 통한 조밀한 변형 정합. 이 과정에서 SHG와 H&E 이미지는 각기 다른 강도 분포를 가지고 있으므로 전처리를 통해 특징을 최대한 유사하게 만듭니다.

- **Performance Highlights**: 제안된 방법은 Learn2Reg 챌린지의 데이터셋을 사용하여 평가되었으며, 초기 정합에서 88%의 성공률과 평균 목표 등록 오류(average target registration error) 2.48을 기록했습니다. 이 방법의 소스 코드는 공개되어 DeeperHistReg 이미지 등록 프레임워크에 통합되어 사용될 수 있습니다.



### Facing Asymmetry - Uncovering the Causal Link between Facial Symmetry and Expression Classifiers using Synthetic Interventions (https://arxiv.org/abs/2409.15927)
Comments:
          45 pages; 26 figures; accepted at ACCV 2024

- **What's New**: 이 연구는 얼굴 비대칭이 표현 분류기의 출력 행동에 미치는 영향을 분석하기 위해 인과 추론 접근법을 사용합니다. 특히, 얼굴 비대칭이 표현 분류 모델의 예측 성능에 미치는 영향을 정량화하고, 이를 통해 블랙박스 모델의 내부 의사결정 프로세스를 이해하고자 합니다.

- **Technical Details**: 연구진은 구조적 인과 모델(structural causal model)을 기반으로 하는 합성 개입 프레임워크를 개발했습니다. 이 프레임워크는 얼굴 비대칭을 조작하여 출력 행동을 분석할 수 있게 해주며, 17개의 표현 분류기에 대해 비대칭 감소가 출력 활성화를 유의미하게 감소시킨다는 결과를 도출했습니다.

- **Performance Highlights**: 연구 결과, 모든 17개의 표현 분류기는 얼굴 비대칭이 줄어들 경우 출력 활성화가 현저히 낮아지는 것으로 나타났습니다. 이러한 결과는 실세계의 건강한 피험자 및 얼굴 마비 환자 데이터에서 관찰된 행동과 일치합니다.



### Learning Compact Channel Correlation Representation for LiDAR Place Recognition (https://arxiv.org/abs/2409.15919)
Comments:
          Submitted to ICRA 2025

- **What's New**: 본 논문은 LiDAR 장소 인식(LiDAR place recognition)을 위해 Compact Channel Correlation Representation (C3R)이라는 새로운 접근 방식을 제안합니다. C3R은 기존의 공분산 풀링(covariance pooling) 방법의 계산 부담과 차원 축소 문제를 해결하고자 하며, 피처 매트릭스를 소규모 그룹으로 나눈 후, 그룹별 공분산 행렬을 계산하고 이를 학습 가능한 집합 전략으로 집계합니다.

- **Technical Details**: C3R 방법은 피처 채널을 작은 그룹으로 나누고 각 그룹에 대해 공분산 행렬을 형성하여 로컬 피처 간의 상관관계를 효과적으로 캡처합니다. 이를 통해 데이터베이스 검색 속도를 높이고, 규범 행렬(normalization)을 적용하여 안정성을 보장합니다. 이론적 분석을 통해 순열 불변성(permutation invariance)과 고상호 정보(mutual information) 유지 능력을 보여줍니다.

- **Performance Highlights**: Oxford RobotCar, In-house, MulRan, WildPlaces 데이터셋에서 C3R 방법의 정확성과 강인성을 입증하기 위해 광범위한 실험을 수행하였으며, 기존의 첫 번째 차원 풀링 및 공분산 풀링 방법들과 비교하여 우수한 성능을 나타냈습니다. 코드 또한 논문 수락 후 공개될 예정입니다.



### Exploring the potential of collaborative UAV 3D mapping in Kenyan savanna for wildlife research (https://arxiv.org/abs/2409.15914)
Comments:
          accepted at IMAV 2024

- **What's New**: 본 논문은 UAV 기반의 협업 매핑 기술을 통해 야생 생물 보존을 지원하는 새로운 접근 방식을 탐구하고 있습니다. 특히 Visual Simultaneous Localization and Mapping (V-SLAM)과 Structure-from-Motion (SfM) 방법론의 실시간 성능 비교를 통해 이들의 장단점을 분석합니다.

- **Technical Details**: 이 연구에서는 V-SLAM의 전통적 제한 사항과 최근의 On-the-Fly (OtF) SfM의 가능성을 동시에 고려합니다. 협업 UAV의 데이터 수집 과정을 통해, 각각의 UAV가 실시간으로 자율적으로 위치를 파악하고, 맵을 재구성하는 구조를 갖추고 있습니다. OtF-SfM은 여러 UAV가 수집한 이미지 스트림을 실시간으로 처리하는 기능을 제안합니다.

- **Performance Highlights**: 연구에서는 두 가지 데이터 세트를 활용하여 협업 UAV의 성능을 평가했습니다. 최종 평가에서는 각 기법의 항공 경로 정확도를 Root Mean Square Error (RMSE) 지표로 측정하며, CNN으로 추출한 학습 기반의 tie points가 최종 정확도에 미치는 영향을 검토합니다.



### Unimotion: Unifying 3D Human Motion Synthesis and Understanding (https://arxiv.org/abs/2409.15904)
- **What's New**: Unimotion은 유연한 모션 제어 및 프레임 수준의 모션 이해가 가능한 최초의 통합 다중 작업 인간 모션 모델입니다. 기존의 모델은 글로벌 텍스트 또는 세밀한 프레임 스크립트 중 하나만 사용하여 아바타의 모션을 제어할 수 있었으나, Unimotion은 두 가지 모두를 동시에 수행할 수 있습니다.

- **Technical Details**: Unimotion은 글로벌 시퀀스 레벨 또는 로컬 프레임 레벨 텍스트 입력을 다룰 수 있으며, 각 포즈에 대한 세부적인 텍스트 설명이나 인간 모션 시퀀스를 출력합니다. 이 모델은 트랜스포머 아키텍처를 사용하고, 각각의 3D 포즈와 프레임 수준 텍스트 간의 시간적 정렬을 수행합니다. 지역 텍스트를 포즈와 함께 확산(diffusion)시키는 방법을 통해 모션 생성 및 이해의 유연성을 제공합니다.

- **Performance Highlights**: Unimotion은 HumanML3D 데이터셋의 프레임 수준 텍스트에서 모션으로의 전이 작업에서 최신 기술(State-of-the-art) 결과를 기록했습니다. 또한, 이 모델은 2D 비디오 주석, 4D 모캡 주석, 계층적 제어 및 모션 편집 등의 다양한 실제 응용 분야에 유용하게 사용될 수 있습니다.



### Unsupervised Attention Regularization Based Domain Adaptation for Oracle Character Recognition (https://arxiv.org/abs/2409.15893)
- **What's New**: 이 연구에서는 중국의 고대 문자인 oracle characters 인식 문제를 해결하기 위해 새로운 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 방법인 비지도 주의 정규화 네트워크(Unsung Attention Regularization Network, UARN)를 개발했습니다. UARN은 레이블이 있는 손으로 쓴 oracle 문자에서 레이블이 없는 스캔 데이터로 인식 지식을 전이하는 역할을 합니다.

- **Technical Details**: UARN은 주의 일관성(attention consistency) 및 주의 구분성(attention discriminability)을 고려하여 oracle 문자 인식을 개선합니다. 전통적인 UDA 방법은 주로 분포 불일치를 줄이는 데 중점을 두지만, 본 연구에서는 이미지 플리핑 (flipping)에 대한 강건성을 요구합니다. 이를 통해 맵 경계가 서로 일치하도록 하고 구분이 어려운 클래스 간의 시각적 혼란을 줄입니다.

- **Performance Highlights**: Oracle-241 데이터셋을 이용한 실험 결과, UARN은 기존의 구조-텍스처 분리 네트워크(Structure-Texture Separation Network)보다 8.5% 더 우수한 성능을 보였으며, 더 나은 해석 가능성과 정확도를 기록했습니다.



### CAD: Memory Efficient Convolutional Adapter for Segment Anything (https://arxiv.org/abs/2409.15889)
Comments:
          14 pages

- **What's New**: 이 논문은 이미지 분할(분할 의미하는 Segment Anything, SAM)에서의 메모리 효율성을 개선하기 위해 새로운 병렬 컨볼루션 어댑터 아키텍처를 제안합니다. 기존의 어댑터 방식이 GPU 메모리 소모 문제를 갖고 있다는 점을 강조하며, 이를 해결하기 위해 SAM의 이미지 인코더와 병렬로 연결되는 방안을 제시합니다.

- **Technical Details**: 새로운 아키텍처는 SAM 모델의 이미지 인코더와 병렬로 연결되어 훈련 중에 이미지 인코더의 활성화(activation) 및 기울기(gradient)를 저장할 필요가 없게 만듭니다. 이를 통해 GPU 메모리 사용량이 절반 이하로 줄어들어, 간단한 디코더 미세 조정(fine-tuning)에 대한 대안으로 가치를 부각시켰습니다. 이 연구에서는 Fast Fourier Transform(FFT)을 사용하여 입력 이미지에서 고주파수 성분(high-frequency components, HFC)을 추출하는 방식을 채택하였습니다.

- **Performance Highlights**: 우리의 제안된 구조는 SAM 어댑터와 SAM 디코더와 비교하여 두 가지 도전 과제(그림자 탐지 및 위장 객체 탐지)에서 경쟁력 있는 실험 결과를 보여주었으며, 하드웨어 제한에 따라 어댑터 기반 학습이 불가능할 경우 유용한 대안임을 입증하였습니다.



### Exploring VQ-VAE with Prosody Parameters for Speaker Anonymization (https://arxiv.org/abs/2409.15882)
- **What's New**: 이번 연구에서는 Vector-Quantized Variational Auto-Encoder (VQ-VAE)를 기반으로 한 새로운 화자 익명화 접근법이 소개됩니다. 이 방법은 음성의 프로소디(prosody), 언어적 콘텐츠, 화자 정체성을 분리하여 화자 정체성만 수정할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 아키텍처는 세 개의 별도 브랜치를 통해 콘텐츠, 프로소디 및 화자 정체성에 대한 임베딩(embedding)을 계산합니다. 합성(synthesis) 과정에서 이 임베딩을 기반으로, 디코더는 화자 및 프로소디 정보를 조건으로 하여 더 섬세한 감정 상태를 포착하고 화자 식별을 정밀 조정합니다.

- **Performance Highlights**: 이 방법은 감정 정보를 보존하는 데 있어 대부분의 기준 기술(baseline techniques)을 능가하는 성과를 보였습니다. 그러나 다른 음성 프라이버시(voice privacy) 작업에서는 더 제한된 성과를 보여 추가적인 개선이 필요하다는 점이 강조되었습니다.



### Zero-Shot Detection of AI-Generated Images (https://arxiv.org/abs/2409.15875)
- **What's New**: AI 생성 이미지 탐지는 전례 없는 현실감과 능력을 가진 새로운 생성 아키텍처들이 지속적으로 등장하면서 매우 어려운 도전이 되고 있습니다. 이 논문에서는 AI 생성 훈련 데이터가 필요 없고 생성 아키텍처에 대한 지식도 필요 없는 제로샷 엔트로피 기반 탐지기(ZED)를 제안합니다.

- **Technical Details**: ZED 탐지기는 머신 생성 텍스트 탐지에 대한 최근 연구에서 영감을 받아 분석 중인 이미지가 실제 이미지 모델에 비해 얼마나 놀라운지를 측정합니다. 이를 위해, 우리는 손실 없는 이미지 인코더를 사용하여 각 픽셀의 확률 분포를 추정합니다. 이 인코더는 다중 해상도 아키텍처를 가지고 있으며, 컨텍스트는 이미지의 저해상도 버전의 픽셀로 구성됩니다.

- **Performance Highlights**: 제안된 탐지기는 단일 판별 피처(discriminative feature)를 사용하여 최첨단의(SoTA) 성능을 달성하며, 다양한 생성 모델에 대해 평균 3% 이상의 정확도 향상을 이룹니다.



### Potential Field as Scene Affordance for Behavior Change-Based Visual Risk Object Identification (https://arxiv.org/abs/2409.15846)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문에서는 지능형 운전 시스템을 위해 잠재적 위험을 식별하는 새로운 비주얼 리스크 객체 식별 프레임워크인 Visual-ROI를 제안합니다. 기존 방법들은 공간적 정확성과 시간적 일관성에서 눈에 띄는 한계를 보였으며, 이는 장면의 affordance를 완전하게 이해하지 못한 데서 비롯됩니다.

- **Technical Details**: 새로운 프레임워크는 Bird's Eye View (BEV) 표현을 사용하여 위험 객체를 식별하는데 있어 잠재적 필드(potential fields)를 탐색합니다. 이는 도로 인프라 및 교통 참여자로부터 유도된 반발력(repulsive forces)과 목표 지점에서 유도되는 인력(attractive forces)을 포함합니다. BEV 시맨틱 분할 결과를 통해 잠재적 필드를 계산하여 공간적 및 시간적 일관성을 개선했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 RiskBench 데이터셋에서 20.3% 및 11.6% 향상을 보였으며, nuScenes 데이터셋에서도 각각 5.4%의 공간적 정확성 증가와 7.2%의 시간적 일관성 향상을 보여주었습니다. 또한, 계산 효율성이 88% 증가했습니다.



### FSF-Net: Enhance 4D Occupancy Forecasting with Coarse BEV Scene Flow for Autonomous Driving (https://arxiv.org/abs/2409.15841)
- **What's New**: 이번 연구는 차량 자율 주행을 위한 4D occupancy forecasting의 새로운 접근 방식을 제안합니다. 제안된 방법은 BEV(scene flow)에서의 coarse 예측을 활용하여 4D 점유도 맵의 경향성을 효과적으로 예측함으로써 자율 주행의 안전성을 향상시키는 데 기여하고자 합니다.

- **Technical Details**: FSF-Net은 coarse BEV scene flow에 기반한 구조를 가지며, 여러 단계의 네트워크로 구성되어 있습니다. 첫 단계에서는 coarse BEV scene flow를 기반으로 general occupancy forecasting architecture를 개발하고, 두 번째 단계에서는 VQ-Mamba network를 통해 spatial-temporal structural scene feature를 추출하고, 마지막으로 U-Net 기반의 quality fusion network를 통해 최종 예측 결과를 개선합니다.

- **Performance Highlights**: FSF-Net은 공공 Occ3D 데이터셋을 바탕으로 Extensive experiments을 수행한 결과, IoU 및 mIoU에서 각각 현존하는 최첨단 방법보다 9.56% 및 10.87% 향상된 성과를 기록했습니다. 이러한 성과는 자율 주행의 안전성을 크게 향상시킬 것으로 기대됩니다.



### Deep Learning Techniques for Automatic Lateral X-ray Cephalometric Landmark Detection: Is the Problem Solved? (https://arxiv.org/abs/2409.15834)
Comments:
          16 pages, 7 figures

- **What's New**: 본 논문에서는 craniofacial landmark (두개안면 중요 지점) 의 자동 탐지를 위한 'Cephalometric Landmark Detection (CL-Detection)' 데이터셋을 소개합니다. 이는 가장 크고 포괄적인 공개 데이터셋으로 600장의 Lateral X-ray 이미지와 38개의 랜드마크를 포함하고 있습니다.

- **Technical Details**: CL-Detection 데이터셋은 다수의 의료기관과 장비에서 수집된 이미지 데이터로 구성되어 있으며, 이 데이터셋을 기반으로 한 2023 MICCAI CL-Detection Challenge를 통해 여러 연구팀이 제안한 deep learning (딥 러닝) 방법을 평가했습니다. 참가자들은 자동 landmark detection 알고리즘을 개발하고 Docker 컨테이너를 제출하여 성능을 비교했습니다.

- **Performance Highlights**: 최고의 방법들은 전문가 분석을 근접하게 재현하며, 평균 탐지율은 75.719%, 평균 반경 오차는 1.518 mm에 도달했습니다. 그러나 아직 개선의 여지가 있으며 여러 가지 실패하는 시나리오가 존재함을 확인하였습니다.



### PseudoNeg-MAE: Self-Supervised Point Cloud Learning using Conditional Pseudo-Negative Embeddings (https://arxiv.org/abs/2409.15832)
Comments:
          Submitted to ICRA2025

- **What's New**: PseudoNeg-MAE는 점구름(point cloud) 마스크 오토인코더(mask autoencoder)의 전역 특징 표현을 향상시키는 새로운 자기 지도 학습(self-supervised learning) 프레임워크로, 변환에 민감하고 식별 가능(discriminative)한 표현을 모두 지원한다. 기존의 대비 학습(contrastive learning) 방법들은 불변성(invariance) 확보에 집중하여 변환과 관련된 중요한 정보를 손실할 수 있었으나, PseudoNeg-MAE는 COPE라는 파라메트릭 네트워크를 사용하여 원본과 변환된 데이터 포인트 간의 관계를 모델링한다.

- **Technical Details**: 본 프레임워크는 COnditional Pseudo Negatives Embedding 네트워크(COPE)를 사용하여 다양한 변환을 적용한 입력 점구름에 대해 의사 부정(pseudo-negative) 임베딩을 생성하고, 이 ผ่าน 새로운 손실 함수로 COPE를 정규화하여 불변 솔루션으로 수렴하는 것을 방지한다. 이를 통해 MAE의 글로벌 표현이 보다 식별 가능하고 변환에 민감해진다.

- **Performance Highlights**: PseudoNeg-MAE는 ModelNet40 및 ScanObjectNN 데이터셋에서의 모양 분류 및 상대 자세 추정(relative pose estimation) 작업에서 최고 성능(state-of-the-art performance)을 달성하며, 특히 상대 회전 추정에서 뛰어난 정확도를 보여준다. 이러한 결과는 PseudoNeg-MAE가 식별 가능하고 변환 민감한 표현을 학습하는 데 효과적임을 나타낸다.



### Layer-wise Model Merging for Unsupervised Domain Adaptation in Segmentation Tasks (https://arxiv.org/abs/2409.15813)
- **What's New**: 본 논문에서는 기존에 훈련된 모델들을 결합하여 비용 없이 모델 병합을 수행하는 새로운 아키텍처를 제안합니다. 이 방법은 레이어 단위로 모델을 통합하여 초기 레이어를 통합하면서 최종 레이어의 특수성을 유지합니다.

- **Technical Details**: 제안된 방법은 Unsupervised Domain Adaptation (UDA)의 맥락에서 다양한 태스크와 데이터셋에 대해 실험하였으며, 특히 Semantic과 Panoptic Segmentation 작업에 적합합니다. 이 방법은 모델 파라미터의 일관성을 유지하고 서로 다른 데이터셋 및 태스크에서의 모델 병합을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 서로 다른 아키텍처의 모델 병합 시 mIoU가 6.8% 향상되었고, Semantic과 Panoptic Segmentation 모델 병합을 통해 mPQ가 7% 증가하는 등 UDA의 성능이 크게 향상되었습니다.



### Hyperbolic Image-and-Pointcloud Contrastive Learning for 3D Classification (https://arxiv.org/abs/2409.15810)
Comments:
          Accepted at IROS2024

- **What's New**: 이번 연구에서는 3D 대비 표현 학습의 새로운 접근법인 하이퍼볼릭 이미지 및 포인트 클라우드 대비 학습(HyperIPC)을 제안합니다. 기존의 코사인 유사도의 대비 학습 방법이 다중 모달 데이터의 잠재적인 의미 계층을 충분히 탐색하지 못하는 문제를 해결하기 위해, 하이퍼볼릭 공간을 활용합니다. HyperIPC는 서로 다른 모달 간의 강한 의미 계층 상관관계를 확립하기 위해 이미지를 사용하는 동시에, 포인트 클라우드를 통해 불변의 특징을 포착합니다.

- **Technical Details**: HyperIPC의 두 가지 모드로, intra-modal(내부 모달)과 cross-modal(교차 모달) 대비 학습을 다룹니다. 내부 모달에서는 포인트 클라우드의 고유한 기하학적 구조를 기반으로 하이퍼볼릭 임베딩(embedding) 표현을 탐색하여 불변 특징을 캡처합니다. 교차 모달에서는 미리 훈련된 이미지 인코더를 사용하여 2D 정보를 추출하고 이를 하이퍼볼릭 공간으로 매핑하여 대비 학습을 수행합니다. 또한, Poincaré 원형 모델을 사용하여 수치 안정성을 유지하면서 노드를 최적화합니다.

- **Performance Highlights**: HyperIPC는 ScanObjectNN 데이터셋에서 객체 분류 성능을 2.8% 향상시키고, 몇 샷 분류 결과를 5.9% 개선하는 등 뛰어난 성과를 보였습니다. 또한, ablation study를 통해 HyperIPC의 파라미터 설정의 논리성과 서브모듈의 효과성을 검증하였습니다.



### A Computer Vision Approach for Autonomous Cars to Drive Safe at Construction Zon (https://arxiv.org/abs/2409.15809)
Comments:
          6 Pages, Double columns

- **What's New**: 이 논문은 건설 구역에서의 도로 장애물 검출을 위한 혁신적이고 정확한 모델을 제안하며, 다양한 데이터 드리프트 조건에서 작동하도록 설계되었습니다.

- **Technical Details**: 이 논문에서 제안하는 모델은 컴퓨터 비전 기술을 기반으로 하며, YOLO 프레임워크를 사용하여 건설 구역에서의 장애물(예: 비콘, 콘, 장애물)을 탐지합니다. CARLA 시뮬레이터를 사용하여 드리프트된 데이터 세트를 생성하고, 이 데이터 세트는 다양한 도시 맵, 도로 레이아웃, 조명 및 날씨 조건을 고려하여 구축되었습니다.

- **Performance Highlights**: 개발된 모델은 94% 이상의 평균 정밀도를 달성하였으며, 검증 데이터 세트에서의 추론 시간은 1.6 밀리초에 불과해 매우 신뢰할 만한 성능을 보여줍니다.



### 3D-JEPA: A Joint Embedding Predictive Architecture for 3D Self-Supervised Representation Learning (https://arxiv.org/abs/2409.15803)
- **What's New**: 3D-JEPA는 새로운 비생성적(非生成的) 3D self-supervised representation learning(SSRL) 프레임워크로, 다중 블록 샘플링 전략과 컨텍스트 인지 디코더를 도입하여 대상 블록의 재구성을 향상시킵니다. 이 접근법은 기존의 수작업 데이터 증강 기법을 사용하지 않고, 특징 공간에서 높은 수준의 의미론적 표현을 학습할 수 있도록 지원합니다. 

- **Technical Details**: 3D-JEPA는 입력 점 구름(Point Cloud)을 처리하기 위해 토큰 임베딩(module) 사용, 다중 블록 샘플링 전략을 통해 컨텍스트 블록과 다양한 대표 대상(target) 블록을 생성합니다. 컨텍스트 인지 디코더를 통해 컨텍스트 정보를 지속적으로 제공함으로써, encoder는 대상 블록에 대한 의미론적 모델링을 보다 잘 수행할 수 있습니다.

- **Performance Highlights**: 3D-JEPA는 다양한 3D 다운스트림 작업에서 높은 효율성과 효과성을 보여줍니다. 예를 들어, PB_T50_RS 데이터셋에서 150개의 프리트레이닝 에폭(epoch)으로 88.65%의 정확도를 달성하였으며, 기존 방법들에 비해 필요한 훈련 에폭 수를 절반으로 줄이는 성과를 올렸습니다.



### DIAL: Dense Image-text ALignment for Weakly Supervised Semantic Segmentation (https://arxiv.org/abs/2409.15801)
Comments:
          accepted by the European Conference on Computer Vision (ECCV), 2024

- **What's New**: 본 논문에서는 Dense Alignment Learning Network (DALNet)를 제안하여 약한 감독 시맨틱 세분화(Weakly Supervised Semantic Segmentation, WSSS)의 문제를 해결합니다. DALNet은 텍스트 임베딩을 활용하여 객체의 포괄적인 이해와 정확한 위치 지정 기능을 향상시킵니다.

- **Technical Details**: DALNet은 두 가지 정렬 전략을 사용합니다: (1) Global Implicit Alignment (GIA)로 클래스 토큰과 텍스트 임베딩 간의 유사성을 극대화하고 배경 임베딩과의 유사성을 최소화하며, (2) Local Explicit Alignment (LEA)는 패치 토큰의 공간 정보를 사용하여 객체의 위치 지정을 개선합니다. 또한 이미지와 텍스트 간의 전경 특징을 정렬하면서 배경과 분리하는 교차 대비 학습(cross-contrastive learning) 접근 방식을 제안합니다.

- **Performance Highlights**: PASCAL VOC 및 MS COCO 데이터셋에서의 실험을 통해 DALNet이 기존의 WSSS 방법들보다 우수한 성능을 보임을 입증하였으며, 특히 단일 스테이지 방법으로서 효율적인 종단 간 프로세스를 가능하게 합니다.



### Training Data Attribution: Was Your Model Secretly Trained On Data Created By Mine? (https://arxiv.org/abs/2409.15781)
- **What's New**: 이 연구에서는 상용 텍스트-이미지(Model) 모델을 사용하여 생성된 데이터를 활용하여 비인가 사용을 방지하는 새로운 접근법을 제안합니다. 특히 추가적인 수정 없이도 의심되는 모델의 학습 데이터를 추적할 수 있는 방법을 제공합니다.

- **Technical Details**: 이 논문에서는 데이터 생성 과정에서 발생하는 고유한 '기억' 특성을 활용하여, 의심되는 모델이 특정한 소스 모델로부터 학습된 데이터를 가지고 있는지를 규명하는 방법을 개발합니다. 이는 모델의 훈련 알고리즘이나 출력에 수정 없이 이루어집니다. 이를 통해 두 가지 주요 접근 방식을 취하는데, 하나는 개별 샘플 수준에서 '핵심 샘플'을 선택하는 방법이고, 다른 하나는 통계적 수준에서 여러 그림자 모델(shadow models)을 훈련하여 데이터를 판별하는 방법입니다.

- **Performance Highlights**: 본 연구의 방법론은 의심되는 모델의 학습 데이터가 약 30%만 포함되어 있는 경우에도 0.6 이상의 정확도를 기록하였으며, 통계적 수준의 방법은 전체적으로 85%가 넘는 정확도로 소스 모델의 데이터를 식별하는 데 성공했습니다.



### ManiNeg: Manifestation-guided Multimodal Pretraining for Mammography Classification (https://arxiv.org/abs/2409.15745)
- **What's New**: 이 논문에서는 유방암 스크리닝 및 분석을 위한 효과적인 방법으로 관심을 끌고 있는 대조 학습(Contrastive Learning)의 개선된 접근법인 ManiNeg을 소개합니다. 특히 유방 조직의 특성을 반영하여 강력한 부정 샘플을 선택하는 새로운 방법론이 등장했습니다.

- **Technical Details**: ManiNeg은 발생 현상(Manifestation)을 근거로 하여 부정 샘플을 효과적으로 선정하는 방법을 제안합니다. 발생 현상은 유병의 증상 및 징후를 의미하며, 이는 하드 부정 샘플 선택에 있어 지식 기반의 강력한 자료를 제공합니다. 이 방법은 모델 최적화에 대한 불변성을 갖고 있어 효율적인 샘플링을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 ManiNeg은 단일 모달과 다중 모달 환경 모두에서 표현 학습을 크게 향상시키는 것으로 나타났습니다. MVKL 데이터 세트에서 평가된 결과, ManiNeg은 다양한 데이터셋에서 일반화 능력을 보여주며 성능을 개선시키는 데 기여했습니다.



### ViKL: A Mammography Interpretation Framework via Multimodal Aggregation of Visual-knowledge-linguistic Features (https://arxiv.org/abs/2409.15744)
- **What's New**: 이 논문에서는 MVKL이라는 첫 번째 멀티모달(multi-modal) 유방 촬영술(mammography) 데이터셋을 소개하며, 시각적 정보 외에도 임상 보고서와 방사선학적 특성과 같은 다양한 특성을 통합하여 보다 해석 가능하고 일반화 가능한 표현 방식의 필요성을 강조하고 있습니다.

- **Technical Details**: MVKL 데이터셋은 다중 시점(multi-view) 이미지를 포함하며, ViKL(framework) 모델은 시각적(Visual), 지식 기반(Knowledge), 언어적(Linguistic) 특성을 조화롭게 통합하여 대조 학습(triple contrastive learning) 방식을 사용하여 양질의 표현 공간을 생성합니다. 이 모델은 병리학적 라벨 없이 쌍(pairing) 정보를 활용하며, 새로운 힘든 부정 샘플 선택 메커니즘인 ManiNeg를 제안하여 전통적인 방법의 한계를 극복합니다.

- **Performance Highlights**: ViKL은 시각적 사전 학습과 임상 보고서 및 상징적 정보를 통합하여 병리학적 분류를 현저히 개선하고, 다양한 데이터셋 간 전이 가능성을 보여주며, 미세 병변을 검출할 수 있는 능력을 갖추었습니다. 실험 결과는 MVKL 데이터셋을 통해 이루어져, 불균형 문제를 최소화하고 효율적인 다중 모달 구조를 갖춘 것을 확인했습니다.



### Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constrain (https://arxiv.org/abs/2409.15739)
Comments:
          Accepted by ECCV'2024

- **What's New**: 최근 악천후(Adverse Weather) 회복에 대한 발전이 잠재력을 보여주고 있지만, 실제 세계에서의 예측 불가능하고 다양한 날씨 열화 조합은 상당한 도전 과제가 되고 있습니다. 본 연구에서는 'T3-DiffWeather'라는 새로운 파이프라인을 소개하며, 이는 네트워크가 서브 프롬프트(sub-prompts)를 자율적으로 조합해 날씨 프롬프트(weather-prompts)를 구성하도록 하는 프롬프트 풀을 사용합니다.

- **Technical Details**: 우리는 또한 Depth-Anything 기능에 의해 제한된 일반 프롬프트를 통합하여 확산 과정(diffusion process)에 대한 장면별 조건(scene-specific condition)을 제공합니다. 또한, 대조 프롬프트 손실(contrastive prompt loss)을 도입하여 두 가지 유형의 프롬프트에 대한 뚜렷한 표현을 보장합니다. T3-DiffWeather는 다양한 합성 및 실제 데이터 세트에서 국지 상태를 통해 최신 악천후 제거 기술을 극복하며 최첨단 성능을 달성합니다.

- **Performance Highlights**: 우리는 T3-DiffWeather가 다양한 악천후 벤치마크에서 SOTA(State-of-the-Art) 성능을 달성했으며, 최신 WeatherDiffusion에 비해 샘플링 단계 수가 10배 줄어들어 계산 효율성(computational efficiency)에서 현저한 이점을 지닌다는 결과를 발표하였습니다.



### LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation (https://arxiv.org/abs/2409.15727)
Comments:
          Accepted by ECCV 2024

- **What's New**: LaPose는 RGB 기반의 객체 포즈 추정 기술에서 발생하는 깊이 정보 부재에 따른 문제를 해결하기 위해 제안된 새로운 프레임워크입니다. 이는 Laplacian mixture model을 사용하여 객체 모양의 불확실성을 명시적으로 정량화하고 다양한 객체 기하학적 특성을 포착합니다.

- **Technical Details**: LaPose는 두 가지 정보 스트림을 활용하여 객체 모양을 Laplacian mixture model로 모델링합니다. 이로 인해 각 점의 확률 분포를 정의하여 형태 불확실성을 제외하고, PnP 모듈을 통해 2D-3D 대응 관계를 설정하여 포즈를 해결합니다. 또한, 스케일 모호성을 해소하기 위해 스케일 불변(object size and translation) 표현을 도입하였습니다.

- **Performance Highlights**: LaPose는 NOCS 데이터셋에서 RGB 기반 객체 포즈 추정에서 최신 성능을 달성하였으며, 실험 결과는 제안된 디자인 선택의 효과성을 확인합니다.



### Disentangled Generation and Aggregation for Robust Radiance Fields (https://arxiv.org/abs/2409.15715)
Comments:
          27 pages, 11 figures, Accepted by ECCV'2024

- **What's New**: 이 연구에서는 트리플레인(triplane) 기반의 3D 표현을 사용하여 카메라 포즈(camera poses) 및 3D 장면을 효율적이고 고품질로 추정하는 방법을 제안합니다. 특히, Disentangled Triplane Generation 모듈과 Disentangled Plane Aggregation을 도입하여 카메라 포즈 업데이트 시 발생하는 문제를 완화하고, 새로운 두 단계의 Warm-start 훈련 전략을 통해 최적화를 개선합니다.

- **Technical Details**: 트리플레인 기반의 대표적인 접근 방식에서는 지역 업데이트(local updating)로 인해 최적화 과정에서 로컬 미니마(local minima)에 빠지는 문제가 있습니다. 이를 해결하기 위해, 본 연구의 Disentangled Plane Aggregation(DPA) 방식은 각 플레인의 피처(feature)를 독립적으로 분리하여 카메라 포즈와 장면의 대표성을 강건하고 모호하지 않게 최적화합니다. 이 연구는 또한 트리플레인 생성기를 통해 패러미터를 공유하여 전체 맥락을 참고할 수 있는 구조를 제공합니다.

- **Performance Highlights**: 제안된 방법은 LLFF 및 NeRF-synthetic 데이터셋에서 새로운 뷰 생성(novel view synthesis)과 포즈 추정(pose estimation) 모두에서 최신 성능(state-of-the-art performance)을 기록했습니다. 정량적 및 정성적 평가 결과를 통해, 카메라 포즈가 불확실하거나 노이즈가 있는 경우에도 우수한 성능을 보임을 입증했습니다.



### Plenoptic PNG: Real-Time Neural Radiance Fields in 150 KB (https://arxiv.org/abs/2409.15689)
- **What's New**: 이 논문의 목표는 2D 이미지로부터 3D 장면을 매우 압축된 표현으로 인코딩하고, 다양한 플랫폼에서 실시간으로 전송, 디코딩 및 렌더링할 수 있게 하는 것입니다. NeRF 및 Gaussian Splats의 발전에도 불구하고, 이들 방법의 큰 모델 크기와 특수 렌더러는 자유 시점 3D 콘텐츠를 이미지만큼 쉽게 배포하는 데에 도전 과제가 있습니다.

- **Technical Details**: 우리는 Plenoptic Portable Neural Graphics(줄여서 PPNG)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 장면의 플레놉틱 함수를 사인 곱함수로 인덱싱된 밀집 볼륨에 인코딩하는 새로운 3D 표현을 포함합니다. 이 방법은 공간의 여러 위치 간의 특징 공유를 가능하게 하여 전통적인 공간의 보컬보다 더 높은 압축성을 제공합니다. 또한, 새로운 경량 렌더링 파이프라인을 개발하여 Plenoptic PNG 표현을 표준 GL 텍스처와 프래그먼트 셰이더로 인스턴트 디코딩할 수 있습니다.

- **Performance Highlights**: 우리는 Plenoptic PNG가 이전 메모리 효율적 방법에 비해 100배 작은 154KB 모델 크기로 기초 라인보다 뛰어난 성능을 보였다고 보고합니다. 이 방법은 모든 실시간 웹 준비 NeRF 방법 중에서 훈련 속도, 렌더링 품질, 모델 크기 간의 최상의 균형을 보여 주며, 몰입형 3D 미디어를 위한 널리 접근 가능한 현실적이고 효율적인 상호 교환 파일 형식을 제공합니다.



### PDT: Uav Target Detection Dataset for Pests and Diseases Tr (https://arxiv.org/abs/2409.15679)
Comments:
          23 pages, 11 figures, European Conference on Computer Vision 2024

- **What's New**: 본 논문에서는 고해상도의 UAV 기반 데이터셋인 Pests and Diseases Tree 데이터셋(PDT dataset)과 Common Weed and Crop 데이터셋(CWC dataset)을 개발하여, 농업 드론의 잡초, 해충 및 질병 감지 모델의 한계를 극복하고자 합니다. 이러한 데이터셋들은 실세계 운영 환경에서 수집되어, 기존의 데이터셋으로 인해 발생하는 문제를 해결합니다.

- **Technical Details**: PDT 데이터셋은 다양한 고도에서 잡초 및 해충을 감지할 수 있는 고정밀 UAV 데이터셋으로, 이는 스마트 농업에서 필요한 실질적인 데이터 수요를 충족하기 위해 디자인되었습니다. 또한, YOLO-Dense Pest(YOLO-DP) 모델을 설계하고, 이를 통해 다수의 테스트 모델을 재평가하여 데이터셋의 완전성과 YOLO-DP 모델의 효과성을 증명합니다.

- **Performance Highlights**: PDT 및 CWC 데이터셋을 기반으로 한 YOLO-DP 모델은 높은 정확도로 잡초, 해충 및 질병 이미지를 탐지할 수 있는 능력을 보여줍니다. 기존의 검출 모델과 비교하여, YOLO-DP는 뛰어난 성능을 발휘하며, 농업 UAV의 효율성을 향상시킬 것으로 기대됩니다.



### ImPoster: Text and Frequency Guidance for Subject Driven Action Personalization using Diffusion Models (https://arxiv.org/abs/2409.15650)
- **What's New**: 이 논문에서는 ImPoster라는 새로운 알고리즘을 제안하여, '소스' 주체가 '드라이빙' 동작을 수행하는 목표 이미지를 생성합니다. 이는 기존의 방법들과 달리 비지도(unsupervised) 방식으로 동작하며, 추가적인 주석 없이 소스 이미지과 드라이빙 이미지, 그리고 두 이미지에 대한 텍스트 설명만으로 동작합니다.

- **Technical Details**: ImPoster는 사전 훈련된(text-to-image) Latent Diffusion Model을 기반으로 하며, 소스 이미지와 드라이빙 이미지의 특성을 학습하기 위해 모델을 소량의 반복을 통해 미세 조정(finetuning)합니다. 추론 시에는 단계적 텍스트 프롬프트를 사용하여 원하는 목표 이미지를 생성하며, 새로운 확산 가이드 구성을 통해 각 단계에서 소스 주체와 드라이빙 동작의 매니폴드(manifold)로 생성을 유도합니다. 주파수(guidance)는 이미지의 주파수 도메인 속성을 반영하여 생성 과정에서 유용하게 사용됩니다.

- **Performance Highlights**: ImPoster는 다양하고 광범위한 소스-드라이빙 이미지 쌍에서 검증되어 이전 방법들과 비교하여 성능 향상을 입증하였습니다. 120,120 개의 이미지 쌍을 다룬 데이터셋을 기반으로 하여, 예를 들어 코끼리가 책을 읽거나, 원숭이가 명상하고 푸시업을 하거나, 테디베어가 기타를 치는 등의 다양한 이미지를 생성할 수 있습니다. 또한, CLIP Score, SSCD, DINO 등과 같은 기존 대안들과의 정량적 및 정성적 비교를 통해 ImPoster의 효과성을 확인했습니다.



### KISS-Matcher: Fast and Robust Point Cloud Registration Revisited (https://arxiv.org/abs/2409.15615)
Comments:
          9 pages, 9 figures

- **What's New**: KISS-Matcher는 포인트 클라우드 등록을 위한 오픈 소스 C++ 라이브러리로, 기존의 포인트 피처 히스토그램(FPFH)보다 향상된 Faster-PFH 기능 감지기를 포함하여, 등록 파이프라인의 모든 구성 요소를 통합하여 전체적인 효율성을 높였습니다.

- **Technical Details**: KISS-Matcher는 4개의 주요 구성 요소로 이루어져 있습니다: 기하학적 억제(geometric suppression), Faster-PFH 기반의 피처 추출 및 초기 매칭, k-core 기반의 그래프 이론적 아웃라이어 제거, 그리고 변형 비선형(non-minimal) 해결사입니다. 이는 두 개의 불규칙한 복셀화된 포인트 클라우드 간의 정렬을 목표로 합니다.

- **Performance Highlights**: KISS-Matcher는 기존의 최첨단 아웃라이어 강건 등록 파이프라인보다 빠른 성능을 자랑하며, 스캔 수준 등록에서 최첨단 방법과 동등한 성능을 보여주고, 특히 서브맵 수준 및 맵 수준 등록에서 우수한 확장성과 적용성을 갖습니다.



### Assessment of Submillimeter Precision via Structure from Motion Technique in Close-Range Capture Environments (https://arxiv.org/abs/2409.15602)
Comments:
          This study comprises 23 pages, 15 figures, and 5 tables. It is part of an ongoing PhD thesis currently under development

- **What's New**: 이 연구는 구조물 실험을 위한 서브 밀리미터 품질의 3D 모델 생성을 위한 Structure from Motion (SfM) 기법의 가능성을 조사하며, 실험에서는 1미터 거리에서 다양한 품질 설정을 적용하였다.

- **Technical Details**: SfM 기법을 통해 캡처 과정에서 카메라 보정 모델, Scale Bars 분포, 겹치는 비율, 수직 및 경사진 이미지 사용 등을 고려하였다. 80%의 겹침률을 적용하여 RMSE 값 약 0.1mm를 달성하였다.

- **Performance Highlights**: 이 연구 결과는 실험실 환경에서 구조적 시험을 위한 서브 밀리미터 수준의 정밀도를 가진 3D 모델링이 가능함을 보여준다. 이를 통해 구조물 모니터링 및 분석 분야에서 SfM 기법의 활용 가능성이 더욱 확대될 것으로 기대된다.



### FACET: Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality (https://arxiv.org/abs/2409.15584)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문에서는 FACET(Fast and Accurate Event-based Eye Tracking)이라는 새로운 신경망 모델을 제안하여, 이벤트 데이터를 기반으로 눈동자의 타원 매개변수를 실시간으로 출력합니다. 이 모델은 XR(Extended Reality) 애플리케이션에 최적화되어 있으며, 기존의 프레임 기반 시스템이 가진 한계점을 극복하고자 합니다.

- **Technical Details**: FACET은 이벤트 기반의 경량화된 눈동자 검출기로, 기존의 이벤트 데이터에서 직접 타원을 예측하는 방식으로 동작합니다. 이를 위해 EV-Eye 데이터셋을 증강하고, 새로운 삼각법 손실 함수를 도입하여 타원 매개변수의 각도 불연속성 문제를 해결했습니다. 또한, 이벤트 볼륨의 수치화 방법을 설계하여 이벤트 표현값의 분포를 정규화했습니다.

- **Performance Highlights**: FACET은 향상된 EV-Eye 테스트 세트에서 평균 눈동자 중심 오류 0.20 픽셀과 0.53 ms의 추론 시간을 달성하여, 기존의 EV-Eye보다 픽셀 오류를 1.6배, 추론 시간을 1.8배 줄였습니다. 또한, 모델은 4.4배 적은 파라미터 수와 11.7배 적은 산술 연산으로 구현되었습니다.



### Clinical-grade Multi-Organ Pathology Report Generation for Multi-scale Whole Slide Images via a Semantically Guided Medical Text Foundation Mod (https://arxiv.org/abs/2409.15574)
- **What's New**: 이 논문은 Patient-level Multi-organ Pathology Report Generation (PMPRG) 모델을 제안하며, 이는 multi-scale WSI의 특징을 활용하여 정확한 병리 보고서를 생성하는 새로운 접근 방식을 모색합니다.

- **Technical Details**: PMPRG 모델은 multi-scale regional vision transformer (MR-ViT) 모델에서 나온 multi-scale WSI 특징을 이용하며, 실제 병리 보고서를 기반으로 VLM 훈련을 통해 보고서를 자동으로 생성합니다. 이 모델은 다양한 장기로부터 얻어진 여러 WSIs를 바탕으로 환자 수준의 보고서를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 모델은 METEOR 점수 0.68을 달성하며, 전반적인 접근 방식의 효과성을 입증합니다. 또한 이 모델은 많은 WSIs를 포함한 경우에도 효율적으로 병리 보고서를 생성할 수 있어 실제 임상 환경에 적합합니다.



### Critic Loss for Image Classification (https://arxiv.org/abs/2409.15565)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 이미지 분류를 위한 새로운 손실 함수인 Critic Loss for Image Classification (CrtCl)을 제안합니다. CrtCl는 분류Classifier와 평정Critic을 공동으로 학습시켜 모델의 성능을 향상시킵니다. 이 접근법은 레이블이 있는 데이터와 없는 데이터 모두에서 학습할 수 있도록 지원합니다.

- **Technical Details**: CrtCl은 이미지 분류를 생성기-비평가(Generator-Critic) 프레임워크로 설정하여, 기본 분류기가 생성기로서 클래스에 대한 확률 분포를 생성합니다. 평정가는 이 생성기 모델이 올바른 분류를 했는지를 예측하여 손실 값을 계산합니다. 이 구조는 레이블이 없는 데이터를 활용이 가능하며, 세미-슈퍼바이즈드(Secondary learning) 환경에서도 효과적입니다.

- **Performance Highlights**: CrtCl은 세 가지 이미지 분류 데이터셋에서 다양한 양의 레이블 데이터로 실험한 결과, 기존의 베이스라인 대비 분류기의 일반화 및 캘리브레이션(Calibration)을 향상시켰습니다. 또한, 액티브 러닝(Active Learning)에서도 높은 정확도와 캘리브레이션 결과를 보였습니다.



### QUB-PHEO: A Visual-Based Dyadic Multi-View Dataset for Intention Inference in Collaborative Assembly (https://arxiv.org/abs/2409.15560)
- **What's New**: QUB-PHEO는 조립 작업 및 의도 추론에서 인간-로봇 상호작용(HRI) 연구를 발전시킬 가능성을 지닌 시각 기반 쌍방 데이터셋을 소개합니다. 이 데이터셋은 '로봇 대리인' 역할을 하는 두 참가자 간의 풍부한 다중 모드 상호작용을 캡처하며, 36개의 고유한 하위 작업으로 further 세분화된 다양한 조립 작업을 포함합니다.

- **Technical Details**: QUB-PHEO는 70명의 참가자로부터 시각적 주석(visual annotations)인 얼굴 랜드마크(facial landmarks), 시선(gaze), 손 움직임(hand movements) 및 객체 위치(object localization) 등을 포함하여, 두 가지 버전의 데이터를 제공합니다: 50명의 참가자를 위한 전체 비디오 데이터와 70명 모두를 위한 시각적 신호(visual cues)입니다. 기존의 단일 뷰 데이터셋의 한계를 극복하기 위해 다중 뷰 데이터 수집 및 하위 작업 인코딩을 특화하여 개발되었습니다.

- **Performance Highlights**: 이 데이터셋은 다중 시각적 데이터와 복잡한 인간 행동 및 의도에 대한 세부 정보를 제공하여 HRI의 기존 문제를 해결합니다. QUB-PHEO는 인간-로봇 협업에서 향상된 알고리즘 개발을 위한 기본 자원으로 작용하여, 로봇이 인간 행동과 의도를 더 높은 정확도로 해석하고 예측할 수 있도록 돕습니다.



### Mixture of Efficient Diffusion Experts Through Automatic Interval and Sub-Network Selection (https://arxiv.org/abs/2409.15557)
Comments:
          Accepted to the 18th European Conference on Computer Vision, ECCV 2024

- **What's New**: 이 논문에서는 사전 훈련된 확산 모델을 효율적인 전문가의 혼합(Mixture of Efficient Experts)으로 가지치기(pruning)하는 새로운 방법인 DiffPruning을 제안합니다. 이는 각 시간대에 대해 전문화된 모델을 분리하여 사용하며, 이를 통해 샘플링 비용을 줄일 수 있습니다.

- **Technical Details**: Diffusion Probabilistic Models (DPMs)의 샘플링 과정은 많은 양의 denoising steps를 필요로 합니다. 이를 개선하기 위해, 논문에서는 denoising timesteps 간의 유사성을 분석하여 데이터 세트에 따라 자연스러운 클러스터링을 확인하고, 각 간격에 대해 전문가를 세분화하여 fine-tuning합니다. 또한, Expert Routing Agent (ERA)를 도입하여 전문가들 간의 compute resource를 자동으로 할당합니다.

- **Performance Highlights**: DiffPruning 방법은 LSUN-Church, LSUN-Beds, FFHQ, ImageNet 등의 다양한 데이터 세트에서 효과성을 입증하였습니다. 특히, 모델 성과를 유지하면서도 샘플링 속도를 크게 향상시켰습니다.



### SOFI: Multi-Scale Deformable Transformer for Camera Calibration with Enhanced Line Queries (https://arxiv.org/abs/2409.15553)
- **What's New**: 이 연구에서는 카메라 캘리브레이션(camera calibration)을 위한 새로운 모델인 SOFI (multi-Scale defOrmable transFormer for camera calibratIon with enhanced line queries)를 소개합니다. 기존의 transformer 기반 모델들이 교차 스케일 상호작용(cross-scale interaction) 부족 문제를 가지고 있었던 반면, SOFI는 선(line) 쿼리를 개선하여 이러한 문제를 해결합니다.

- **Technical Details**: SOFI는 선 내용을 기반으로 한 선 쿼리(line queries)를 개선해 카메라 캘리브레이션에서 더 나은 성능을 발휘합니다. 이 모델은 다중 스케일(deformable) 어텐션 메커니즘을 사용하여 백본(backbone)에서 생성된 피쳐 맵(feature maps) 간의 교차 스케일 상호작용을 촉진합니다. 새로 제안된 선 쿼리는 각 인코더 층에 선 분절(geometric features)을 입력으로 사용함으로써 쿼리 초기화를 새롭게 설정합니다.

- **Performance Highlights**: SOFI는 Google Street View, Horizon Line in the Wild, Holicity 등 다양한 데이터셋에서 기존 방법들을 능가하며, 경쟁력 있는 추론 속도(inference speed)를 유지하고 있습니다.



### VaLID: Verification as Late Integration of Detections for LiDAR-Camera Fusion (https://arxiv.org/abs/2409.15529)
- **What's New**: 이 논문에서는 LiDAR와 카메라 데이터를 활용한 차량 객체 탐지에 대한 새로운 모델 독립적인 late fusion 방법인 VaLID를 제안합니다. 이 방법은 각 예측된 bounding box가 허용 가능한지를 검증하며, LiDAR 탐지기의 잘못된 예측을 줄이는 데 중점을 둡니다.

- **Technical Details**: VaLID는 캘리퍼스 멀티-레이어 퍼셉트론(Multi-Layer Perceptron, MLP)을 사용하여 높은 재현율(bias)로 LiDAR 탐지기의 잘못된 예측을 줄이는 동시에 실제 예측을 유지합니다. KITTI 데이터셋에서 여러 조합의 LiDAR 및 카메라 탐지기를 평가하여 평균 63.9%의 거짓 긍정(false positive)을 줄였습니다.

- **Performance Highlights**: 우리의 방법은 특정 데이터셋에 특별히 훈련되지 않은 일반 카메라 탐지기를 사용할 때에도 최신 기술과 경쟁할 수 있는 성능을 나타냅니다. VaLID 방식은 차량 감지에 대한 전반적인 평균 정밀도(average precision)를 향상시키는 데 기여했습니다.



### SpaGBOL: Spatial-Graph-Based Orientated Localisation (https://arxiv.org/abs/2409.15514)
- **What's New**: 이번 연구에서는 Cross-View Geo-Localisation 문제를 해결하기 위해 첫 번째 그래프 구조의 데이터셋과 GNN(그래프 신경망)을 도입하였습니다. 새로운 접근 방식으로, 여러 개의 거리뷰 이미지를 통해 지도 노드의 일반화를 개선하고, 노드 근접성과 특성 유사성 간의 상관관계를 활용한 첫 시스템을 개발했습니다.

- **Technical Details**: 이 연구에서 제안하는 SpaGBOL 모델은 GNN 아키텍처를 사용하여 도시 지역에서의 Geo-Localisation 기술을 개선합니다. 데이터는 거리뷰 이미지 및 위성 이미지의 집합으로 구성되며, 이들은 그래프의 노드를 통해 연결되며 도로는 그래프의 엣지로 표현됩니다. GNN은 미지의 시퀀스를 생성하고, 이웃 도로 방향에 기초한 새로운 검색 필터링 기법을 제공합니다.

- **Performance Highlights**: SpaGBOL은 이전 기술 대비 11%의 Top-1 검색 정확도를 개선하였으며, SpaGBOL 데이터셋에 대한 Bearing Vector Matching으로 필터링 시 50%의 개선을 보여 주었습니다. 이로써 새로운 시퀀스를 인식하는 데 필요한 실제적인 적용 가능성을 높였습니다.



### PixelBytes: Catching Unified Embedding for Multimodal Generation (https://arxiv.org/abs/2409.15512)
- **What's New**: 이 보고서는 PixelBytes Embedding이라는 새로운 접근 방식을 소개합니다. 이는 통합된 멀티모달 표현 학습(unified multimodal representation learning)을 위한 방법으로, 다양한 입력을 단일하고 일관된 표현으로 캡처하여 텍스트 및 픽셀화된 이미지에 대한 멀티모달 시퀀스 생성의 우수한 특성을 조화롭게 구현할 수 있도록 합니다.

- **Technical Details**: PixelBytes는 최신 시퀀스 모델인 Image Transformers, PixelCNN 및 Mamba-Bytes에서 영감을 받아 서로 다른 데이터 타입을 통합하는 데에 어려움을 해결하는 것을 목표로 합니다. RNNs(순환신경망), SSMs(상태공간모델), 주의(Attention) 기반 모델 등 다양한 모델 아키텍처를 탐구하며, 양방향 처리(bidirectional processing)와 혁신적인 PxBy embedding 기술에 중점을 둡니다.

- **Performance Highlights**: PixelBytes Pok{é}mon 데이터셋을 사용한 실험에서, PxBy embedding과 합성곱(convolutional) 레이어를 사용한 양방향 시퀀스 모델이 일관된 멀티모달 시퀀스를 생성할 수 있는 가능성을 입증했습니다. 이 작업은 통합 AI 모델의 발전에 기여하며, 멀티모달 데이터를 이해하고 생성하는 데 있어 보다 통합된 방식을 가능하게 합니다.



### Analysis of Human Perception in Distinguishing Real and AI-Generated Faces: An Eye-Tracking Based Study (https://arxiv.org/abs/2409.15498)
- **What's New**: 최근 인공지능(AI) 분야의 발전으로 인해 사실감 있는 인간 얼굴 생성에서 현저한 향상을 이루었습니다. 본 연구에서는 인간이 실제 이미지와 가짜 이미지 간의 차이를 어떻게 인식하고 구별하는지를 조사하였습니다. 본 연구는 Eye-tracking 기술을 활용하여 AI가 생성한 얼굴 이미지와 실제 얼굴 이미지를 판별하는 인간의 인식 방식을 분석했습니다.

- **Technical Details**: 본 연구에서는 StyleGAN-3을 활용하여 생성된 이미지와 Flickr-Faces-HQ Dataset(FFHQ)에서 샘플링한 실제 얼굴 이미지를 사용하였습니다. 실험 대상자들은 7000개 이상의 이미지를 보고 그 결과를 기록하였으며, 이와 동시에 Eye-tracking을 통해 시선 정보를 수집했습니다. 연구를 통해 인식 정확도 76.80%로 실제와 가짜 얼굴을 구별할 수 있는 평균적인 능력을 확인하였습니다.

- **Performance Highlights**: 연구 결과, 참여자들은 가짜 이미지로 의심되는 경우 이미지를 더욱 면밀히 분석하는 경향이 있음을 발견하였습니다. Eye-tracking 방식으로 수집된 데이터 분석을 통해 인식 정확도, 반응 시간, 주목 지속 시간 등의 다양한 측면에서 심도 깊은 분석을 수행하여, 인간의 시각적 행동 패턴을 이해하는 중요한 통찰을 제공했습니다.



### VLMine: Long-Tail Data Mining with Vision Language Models (https://arxiv.org/abs/2409.15486)
- **What's New**: 본 연구에서는 사용되지 않은 라벨 데이터를 통해 드문(long-tail) 예제들을 식별하는 데이터 마이닝(data mining) 방법론을 제안합니다. 제안된 방법은 VLM(Vision Language Model)을 활용하여 이미지 내용을 키워드(keyword) 집합으로 요약하고, 키워드의 빈도를 기반으로 드문 예제를 식별합니다.

- **Technical Details**: 제안된 방법인 VLMine은 VLM에서 추출한 지식을 활용하여 드문 예제를 식별합니다. 기존 모델의 불확실성(uncertainty) 기반 방식보다 VLM의 키워드 빈도 분석이 더 효과적인 신호를 제공하는 것을 보여줍니다. VLMine은 특정 태스크와 무관하게 사용할 수 있는 모델-불가지론적(data-agnostic) 방법입니다.

- **Performance Highlights**: VLMine을 통해 2D 이미지 분류 및 3D 객체 탐지 태스크에서 10%에서 50%의 성능 향상을 달성하였으며, ImageNet-LT, Places-LT, Waymo Open Dataset에서의 벤치마크 시험에서 기존 방법들에 비해 일관된 개선 결과를 보여주었습니다.



### MediConfusion: Can you trust your AI radiologist? Probing the reliability of multimodal medical foundation models (https://arxiv.org/abs/2409.15477)
Comments:
          17 Pages, 5 figures

- **What's New**: 새로운 벤치마크인 MediConfusion이 등장하여, 의료 분야에서 Multimodal Large Language Models (MLLMs)의 실패 유형을 시각적 관점에서 평가하는 데 중점을 두고 있습니다. 이 벤치마크는 기존 모델들의 한계를 드러내고, 신뢰할 수 있는 의료 AI 솔루션 개발에 중요한 기초 자료를 제공합니다.

- **Technical Details**: MediConfusion은 시각적으로 분명히 다른 이미지 쌍을 사용해 MLLMs가 혼란을 겪는 경향을 조사합니다. 이 벤치마크는 LLM(대규모 언어 모델) 프롬프트를 통해 생성된 다수의 선택 문제를 포함하고 있으며, 의료 영상 데이터셋인 ROCO를 활용하여 수집한 이미지 쌍 간의 특성을 분석해 신뢰성 문제를 탐구합니다.

- **Performance Highlights**: 현재 모든 모델(오픈 소스 및 독점 모델 포함)이 MediConfusion에서 무작위 추측보다 낮은 성능을 보이고 있습니다. 이는 의료 MLLMs의 신뢰성에 대한 심각한 우려를 나타내며, 연구자들은 보다 신뢰할 수 있는 모델 디자인을 위한 공통적인 실패 패턴을 파악하기 위해 노력하고 있습니다.



### Mat\'ern Kernels for Tunable Implicit Surface Reconstruction (https://arxiv.org/abs/2409.15466)
Comments:
          18 pages, 8 figures

- **What's New**: 이번 연구에서는 Matérn 커널(Matern kernels) 가족을 활용하여 조정 가능한 implicit surface reconstruction 방법을 제안하고 있습니다. 최근 3D 방향 포인트 클라우드 재구성에서 커널 방법의 성공을 기반으로 개발되었습니다.

- **Technical Details**: Matérn 커널은 정규화된 실수 공간을 기반으로 하며, 강력한 조정 가능성을 제공합니다. 이 커널들은 특히 arc-cosine 커널을 기반으로 한 최신 기술을 초월하는 성능을 보이며, 구현이 쉽고 계산 속도가 빠르며 확장성이 뛰어납니다. 저자들은 Matérn 커널의 스펙트럼을 Fourier feature 매핑과 유사한 방식으로 조정할 수 있음을 이론적으로 분석했습니다. 또한, SIREN 네트워크와의 관련성을 탐구하며 arc-cosine 커널과의 관계도 분석하였습니다.

- **Performance Highlights**: 특히 Laplace 커널은 Matérn 커널 가족의 일원으로서, 노이즈가 없는 경우 최신 방법들과 거의 동등한 성능을 보이며, 훈련 시간도 5배 이상 단축되어 효율적입니다.



### Revealing an Unattractivity Bias in Mental Reconstruction of Occluded Faces using Generative Image Models (https://arxiv.org/abs/2409.15443)
Comments:
          This paper and a corresponding poster were presented at the Cognitive Computational Neuroscience conference in 2024

- **What's New**: 이번 연구는 얼굴의 부분적 가림이 얼굴 매력을 증가시킨다는 기존의 가설에 도전합니다. 실험 결과, 가려진 얼굴의 모습이 오히려 비매력적인 특성을 가진 것으로 재구성된다는 것을 밝혀냈습니다.

- **Technical Details**: 두 가지 온라인 실험을 통해 관찰자들에게 가려진 얼굴 부위의 매력을 평가하도록 하였으며, 최신의 diffusion-based 이미지 생성기를 사용하여 비매력적, 중립적, 매력적인 얼굴 부분 이미지를 생성하였습니다. 실험 방법으로는 delayed matching-to-sample (DMTS) 과제를 활용하였습니다.

- **Performance Highlights**: 실험 결과, 일반적인 얼굴 매력 평가 과제에서는 매력도 편향이 나타났지만, DMTS 과제에서는 비매력적 이미지가 더 많이 선택되는 경향을 보여 비매력적 편향이 재구성 과정에서 발생함을 밝혀내었습니다.



### Ultrafast vision perception by neuromorphic optical flow (https://arxiv.org/abs/2409.15345)
Comments:
          17 pages, 4 figures

- **What's New**: 이번 연구는 기존의 2D 방법의 한계를 극복하기 위한 3D neuromorphic optical flow (광학 흐름) 방법을 제시합니다. 이는 움직임 속성을 직접 하드웨어에 적용하여 정확한 모션 신호를 생성함으로써 비디오 데이터의 처리 시간을 단축시킵니다.

- **Technical Details**: 이 연구에서는 memristors를 활용하여 외부 움직임 특성을 직접 하드웨어에 통합하는 새로운 접근법이 소개되었습니다. 이는 모션 속도를 처리하는 속도를 크게 향상시키고, 다중 영역의 세부적인 움직임 분석을 가능하게 합니다.

- **Performance Highlights**: 이 접근 방식은 평균 0.3초의 비디오 데이터 처리 시간을 단축하면서도 모션 예측, 객체 추적, 객체 분할의 정확도를 유지 또는 향상시킵니다. UAV 시나리오에서 첫 번째로 interframe visual processing (프레임 간 시각 처리)를 달성했습니다.



### Video-Driven Graph Network-Based Simulators (https://arxiv.org/abs/2409.15344)
- **What's New**: 본 논문에서는 짧은 비디오를 통해 물리적 속성을 유추할 수 있는 새로운 방법을 제안합니다. 이 방식은 명시적인 파라미터 입력 없이도 시스템의 동작을 시뮬레이션할 수 있게 해줍니다.

- **Technical Details**: 제안된 방법은 Video-Driven Graph Network-based Simulator (VDGNS)로, 짧은 비디오 세퀀스를 통해 물리적 인코딩 P를 추정하고, 기존의 Graph Neural Network (GNS) 프레임워크를 활용하여 다양한 물리 시스템의 동작을 예측합니다. 이 과정은 비디오 인코더와 GNS의 두 가지 주요 구성 요소로 이루어져 있습니다.

- **Performance Highlights**: 실험 결과, 비디오에서 추출한 인코딩이 효과적으로 시스템의 물리적 속성을 포착하며, 특정 인코딩과 시스템의 운동 사이의 선형적인 관계가 나타났음을 보여줍니다.



### StyleReiser: Stylizing Video With Reinforced Structure Guid (https://arxiv.org/abs/2409.15341)
- **What's New**: 본 논문에서는 StyleReiser라는 예제 기반 비디오 스타일화 방법을 소개합니다. 이 방법은 주어진 키프레임의 스타일을 전체 비디오 시퀀스에 전달하면서 시각적 일관성을 유지합니다. 이전의 키프레임 기반 방법들과 달리, StyleReiser는 지정된 스타일과의 일관성을 고려하고 비디오 시퀀스에 나타나는 새로운 구조적 요소에 대한 충실도를 유지합니다.

- **Technical Details**: StyleReiser는 예제 기반 비디오 스타일화에서 기존 방법들이 구조 보존을 명시적으로 고려하지 않았음을 지적하고, 스타일 가이드를 완화하여 입력 비디오의 새로운 구조적 요소에 대한 충실도를 강조합니다. 이 접근법은 구조 변화에 대한 저항력을 상당히 증가시켜 추가 키프레임을 지정할 필요성을 없앱니다. 특히, 이 기술은 실시간으로 추론을 수행할 수 있어 비디오 통화 같은 인터랙티브한 상황에서도 유용하게 사용할 수 있습니다.

- **Performance Highlights**: StyleReiser는 비디오 스타일화 품질을 크게 향상시킬 수 있으며, 텍스트 기반 비디오 스타일화 방법의 결과를 개선하는 데 도움을 줍니다. 특히, 새로운 구조 요소가 등장할 때 발생할 수 있는 불안정을 억제함으로써 사용자에게 생성된 키프레임을 통해 맞춤 편집을 수행할 수 있게 해줍니다.



### Electrooptical Image Synthesis from SAR Imagery Using Generative Adversarial Networks (https://arxiv.org/abs/2409.15331)
- **What's New**: 본 연구는 Synthetic Aperture Radar (SAR) 이미지를 electro-optical (EO) 이미지로 변환하는 데 사용되는 최첨단 Generative Adversarial Networks (GAN) 모델을 비교하고 평가했습니다. 특히, 개선된 시각적 해석 가능성을 제공하는 새로운 dual-generator architecture를 도입하였습니다.

- **Technical Details**: 연구에서 사용된 GAN 모델에는 Pix2Pix, CycleGan, S-CycleGan 및 partial convolutions를 활용한 새로운 dual-generator GAN이 포함됩니다. 이 모델들은 SAR 이미지를 EO 이미지로 변환하는 과정에서 점진적으로 사실감을 개선하도록 설계되었으며, transformers 아키텍처를 이용하여 더욱 정교한 변환을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 생성된 EO 이미지들은 실제 EO 이미지와 비교할 때 시각적 충실성과 특징 보존 측면에서 상당한 개선을 보여 SAR 데이터의 해석 가능성을 높였습니다. 또한, 이 기술은 환경 모니터링, 도시 계획 및 군사 정찰과 같은 다양한 응용 분야에서 SAR 데이터의 신속하고 정확한 해석에 기여할 수 있는 잠재력이 있습니다.



### Texture Discrimination via Hilbert Curve Path Based Information Quantifiers (https://arxiv.org/abs/2409.15327)
- **What's New**: 이 논문은 Hilbert curve를 사용하여 이미지로부터 데이터를 추출하고 새로운 texture classification 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 두 차원 이미지를 Hilbert curve를 통해 일 차원 시간 시리즈로 변환하고, 두 번째 단계에서는 Bandt & Pompe 기호화를 사용하여 세 가지 정보 이론 quantifiers인 permutation entropy, permutation complexity 및 Fisher 정보 측정을 계산합니다.

- **Performance Highlights**: 이 방법은 Brodatz 이미지 데이터베이스와 같은 일반적으로 사용되는 이미지 데이터셋에서 우수한 성능을 보여주며, 회전, 대칭, 색상 변형에 강한 강건성을 입증하였습니다.



### Deep Transfer Learning for Breast Cancer Classification (https://arxiv.org/abs/2409.15313)
- **What's New**: 이 논문에서는 유방암 이미지를 분류하기 위해 VGG, Vision Transformers (ViT), ResNet 모델을 활용하고, 이 알고리즘들의 성능을 비교 분석하였습니다. 특히, Invasive Ductal Carcinoma (IDC) 암 이미지를 분류하는 데 있어 ResNet-34가 90.40%의 정확도를 기록하며 우수한 성능을 보였습니다.

- **Technical Details**: 본 연구는 Convolutional Neural Networks (CNN) 아키텍처인 VGG, ResNet 및 새로운 Vision Transformers를 사용하여 유방암 데이터를 분석했습니다. 데이터셋은 162개의 전체 슬라이드 이미지로 구성되어 있으며, 이를 50x50 픽셀 크기의 패치로 나누어 학습에 사용했습니다. 이미지 전처리 및 증강(image augmentation) 기술이 적용되어 모델 성능을 향상시켰습니다.

- **Performance Highlights**: ResNet-34는 90.40%의 높은 정확도를 기록하였고, Pretrained VGG-16은 적은 파라미터를 업데이트하기 때문에 더 높은 F1-score를 달성하였습니다. 이러한 결과는 Deep Transfer Learning이 유방암 진단 분야에 큰 도움이 될 것임을 시사합니다.



### Enhancing coastal water body segmentation with Landsat Irish Coastal Segmentation (LICS) datas (https://arxiv.org/abs/2409.15311)
- **What's New**: 아일랜드 해안선의 중대한 변화 감지를 위해 Landsat Irish Coastal Segmentation (LICS) 데이터셋이 새롭게 발표되었습니다. 이 데이터셋은 해양 물체 세분화에 적용할 수 있는 딥러닝 방법 개발을 지원합니다.

- **Technical Details**: LICS 데이터셋은 아일랜드의 다양한 환경 조건과 기후 특정성을 반영하여, 해안선 세분화를 위한 최초의 딥러닝용 세분화 데이터셋입니다. 연구에서는 U-NET 알고리즘이 95.0%의 최고 정확도를 기록하였으나, Normalised Difference Water Index (NDWI)가 평균 97.2%의 정확도로 더 우수한 성능을 보였습니다.

- **Performance Highlights**: 딥러닝을 활용한 해안선 세분화 성능이 기대보다 우수했으나, 더 정확한 훈련 데이터와 침식 측정 방안을 고려하여 성능 개선이 가능할 것이라고 제안합니다. 데이터셋과 코드가 무료로 제공되며, 이는 재현 가능한 연구와 해안 모니터링의 발전을 지원합니다.



### The NGT200 Dataset: Geometric Multi-View Isolated Sign Recognition (https://arxiv.org/abs/2409.15284)
Comments:
          Proceedings of the Geometry-grounded Representation Learning and Generative Modeling Workshop (GRaM) at the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 251, 2024

- **What's New**: 본 연구에서는 Sign Language Processing (SLP)의 다중 관점 고립 기호 인식(MV-ISR)을 다루며, 3D 인식 및 기하학의 중요성을 강조합니다. 또한 새로운 spatio-temporal multi-view benchmark인 NGT200 데이터셋을 소개합니다.

- **Technical Details**: NGT200 데이터셋은 다중 관점에서 촬영된 고립 기호의 비디오 클립에서 추출한 2D 랜드마크를 포함합니다. 이 데이터셋은 3D-LEX 데이터셋과 함께 사용되어 각 기호에 대한 3D Ground Truth를 제공합니다. 이 연구는 SE(2) 등변 모델을 활용하여 MV-ISR의 성능을 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: MV-ISR은 SE(2) 등변 모델을 활용하여 성능이 기준선 대비 8%-22% 향상되었습니다. 이를 통해 기호 인식 시스템의 실용성을 높일 수 있는 방법을 제시합니다.



### Gen2Act: Human Video Generation in Novel Scenarios enables Generalizable Robot Manipulation (https://arxiv.org/abs/2409.16283)
Comments:
          Preprint. Under Review

- **What's New**: 이 논문은 웹 데이터에서 생성한 인간 비디오를 통해 로봇 조작 정책이 새로운 물체 및 모션에 일반화될 수 있도록 하는 방법을 제시합니다. Gen2Act라는 접근 방식을 통해 로봇 정책을 비디오 생성 모델에 조건화하여 무작위 작업을 수행하는 프레임워크를 사용합니다.

- **Technical Details**: Gen2Act는 언어 조건조작을 제로샷(Zero-shot) 인간 비디오 생성과 로봇 행동으로의 변환을 포함하는 폐쇄 루프(policy) 형태로 구현되어 있습니다. 이를 위해 비디오 예측 모델을 사용하여 주어진 작업 설명에 따라 인간 비디오를 생성하고, 생성된 비디오에 조건화된 로봇 동작을 추론하는 모델을 학습시킵니다.

- **Performance Highlights**: Gen2Act는 로봇 상호작용 데이터가 부족한 상황에서도 새로운 물체 유형과 모션 유형을 Manipulation 하는 데 평균 약 30% 높은 성공률을 보여주었습니다. 또한 연속적인 작업인 커피 만들기와 같은 긴 활동 수행에도 적용 가능합니다.



### Compressed Depth Map Super-Resolution and Restoration: AIM 2024 Challenge Results (https://arxiv.org/abs/2409.16277)
Comments:
          ECCV 2024 - Advances in Image Manipulation (AIM)

- **What's New**: 이 논문은 증강 현실(AR) 및 가상 현실(VR) 응용 프로그램을 위한 깊이 정보의 효율적 처리를 강조합니다. 특히, 압축된 데이터로부터 고품질 깊이 맵을 reconstruction하는 혁신적인 depth upsampling 기술 개발에 중점을 두고 있습니다.

- **Technical Details**: 깊이 맵은 객체 인식(object recognition), 장면 이해(scene understanding), 제스처 추적(gesture tracking) 등에 필수적인 요소입니다. 다루는 기술적 측면으로는 depth compression, depth completion 및 depth densification이 있습니다. 복잡한 degradations 문제를 해결하기 위해 U-Net 기반 아키텍처를 활용한 단순하고 효과적인 네트워크를 제안합니다.

- **Performance Highlights**: 전통적인 Bicubic interpolation이 기대한 성능을 보이지 않으며, 최근의 상위 3개 방법(UM-IT, DAS-Depth, DINOv2 + ControlNet)은 인코더-디코더 구조를 사용하여 HR RGB 이미지 및 LR 깊이 맵을 조건화하여 HR 깊이 맵을 예측합니다.



### Fine-Tuning is Fine, if Calibrated (https://arxiv.org/abs/2409.16223)
Comments:
          The first three authors contribute equally

- **What's New**: 본 연구에서는 사전 학습된 모델(Foundation Model)에서 세부 클래스를 대상으로 한 파인튜닝(fine-tuning) 중 발생하는 문제를 체계적으로 분석합니다. 연구 결과, 파인튜닝된 모델이 정확도가 하락하는 주된 원인이 로짓 스케일의 불일치임을 밝혀내고, 이러한 문제를 해결하기 위해 단순한 후처리(calibration) 방법을 제안합니다.

- **Technical Details**: 사전 학습된 분류기를 세부 클래스에 맞춰 파인튜닝하면 일반적인 정확도가 손상되지만, 이 논문에서는 파인튜닝 후에도 특징 추출기(feature extractor)의 성능 수치가 개선됨을 확인하였습니다. NCM(Nearest Class Mean) 분류기를 통해 특징 품질을 평가한 결과, 파인튜닝 동안 실종된 클래스에 대한 특징 분리가 향상되었습니다. 로짓 스케일의 불일치가 피해를 주며, 이 문제는 후처리 방법으로 보완할 수 있습니다.

- **Performance Highlights**: 다수의 벤치마크(예: ImageNet)에서 후처리 보정(calibration)을 통해 파인튜닝된 모델의 성능이 현저히 향상되었으며, 이는 강력한 기준선 모델조차 능가했습니다. 연구 결과는 간단한 파라미터 조정만으로도 유의미한 성과를 달성할 수 있음을 보여줍니다.



### Tiny Robotics Dataset and Benchmark for Continual Object Detection (https://arxiv.org/abs/2409.16215)
Comments:
          Paper under review

- **What's New**: 본 논문에서는 Tiny Robotics 분야에서 시스템의 지속적 학습(continual learning) 능력을 평가하기 위한 새로운 벤치마크와 Tiny Robotics Object Detection (TiROD)라는 데이터 세트를 소개합니다. 이 데이터 세트는 소형 모바일 로봇을 이용해 수집되었으며, 다양한 도메인과 객체 클래스에서 객체 탐지기의 적응력을 테스트하도록 설계되었습니다.

- **Technical Details**: 제안된 TiROD 데이터 세트는 5개의 서로 다른 환경(실내 및 실외)에서 13개의 객체 클래스를 캡처하여 다양한 객체 탐지기의 성능을 평가하는 데 사용됩니다. 본 연구는 Nanodet과 Yolov8라는 두 대의 최첨단 객체 탐지기를 지속적 학습 전략과 결합하여 평가하며, 그들의 강점과 한계에 대한 통찰을 제공합니다.

- **Performance Highlights**: TiROD의 벤치마크 결과는 Tiny Robotics에서 강력하고 효율적인 객체 탐지 시스템의 발전을 위해 해결해야 할 주요 도전과제를 나타냅니다. 이 논문은 또한 이 분야의 지속적인 발전을 촉진하기 위해 모든 접근 방식의 소스 코드를 공개합니다.



### Upper-body free-breathing Magnetic Resonance Fingerprinting applied to the quantification of water T1 and fat fraction (https://arxiv.org/abs/2409.16200)
Comments:
          19 pages, 9 figures, 3 tables

- **What's New**: 이 연구는 모션 보정된 (MoCo) MRF T1-FF 접근법을 제안하여 폐부위에서 발생하는 호흡 모션 효과를 줄이고, 이를 통해 정확한 MRI 매개변수 맵을 재구성하는 새로운 방법을 보여줍니다.

- **Technical Details**: 제안된 접근법은 최적화된 초기 모션 스캔을 사용하여 모션 필드를 추정하고, 이를 기반으로 MRF 데이터를 수정하여 FF 및 T1이 보정된 파라메트릭 맵을 재구성합니다. 이 방법은 호흡 근육과 같은 희귀한 부위에서도 적용 가능성을 높이며, 기존 MRF의 한계를 극복합니다.

- **Performance Highlights**: 검증 결과, 최소한의 모션 영향을 받은 지역에서 MoCo 재구성이 비보정 재구성과 큰 차이를 보이지 않았으며, 호흡 근육 같은 모션의 영향을 많이 받는 영역에서는 모션 블러링과 아티팩트를 유의미하게 감소시켰습니다. 또한, 횡격막이 모션 보정 후에도 선명하게 나타났습니다.



### Efficient Motion Prediction: A Lightweight & Accurate Trajectory Prediction Model With Fast Training and Inference Speed (https://arxiv.org/abs/2409.16154)
Comments:
          Accepted to IROS 2024

- **What's New**: 본 논문에서는 자율 주행을 위한 새로운 효율적인 모션 예측 모델(EMP)을 제안하며, 이는 단일 GPU에서 몇 시간의 훈련으로도 경쟁력 있는 결과를 도출해냅니다.

- **Technical Details**: EMP는 표준 transformer 블록을 기반으로 하여 에이전트의 이력, 도로 형태 및 장면 정보를 인코딩하고, 미래의 경로 및 신뢰도 점수를 디코딩하기 위해 단순한 다층 퍼셉트론 기반 디코더와 복잡한 transformer 기반 방법을 비교합니다.

- **Performance Highlights**: 이 모델은 AV2 데이터셋에서 우수한 성능을 보이며, 훈련 속도가 약 100% 빨라졌으며, 매우 효율적인 추론 속도를 제공합니다.



### CloudTrack: Scalable UAV Tracking with Cloud Semantics (https://arxiv.org/abs/2409.16111)
Comments:
          7 pages, 3 figures

- **What's New**: 본 논문에서는 UAV(무인 항공기) 하드웨어의 한계를 극복하기 위해 특별히 설계된 의미적으로 조건화된 개방 어휘(open vocabulary) 객체 추적 방법을 제안합니다.

- **Technical Details**: 제안된 방법은Missing person의 언어적 설명, 예를 들면 셔츠의 색상 등을 기반으로 작동할 수 있으며, 특정 훈련 없이 미션을 수행할 수 있습니다. 또한, 잠재적으로 움직이는 대상을 효율적으로 추적할 수 있는 장점이 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법의 다재다능함과 효과성을 입증합니다.



### Multi-Model Ensemble Approach for Accurate Bi-Atrial Segmentation in LGE-MRI of Atrial Fibrillation Patients (https://arxiv.org/abs/2409.16083)
- **What's New**: 이 논문은 심방 세동(Atrial Fibrillation, AF)에 대한 이해를 높이고, AF 환자들의 절제술(ablation) 성공 가능성을 예측하는 데 필요한 심방(fibrosis 및 scarring) 이미징의 중요성을 강조합니다. 특히, 다중 센터의 3D LGE-MRI를 이용한 다중 클래스 심방 분할(Multi-class Bi-Atrial Segmentation, MBAS) 문제를 다룹니다.

- **Technical Details**: 이 연구는 Unet, ResNet, EfficientNet, VGG와 같은 여러 머신 러닝 모델을 통합하여 LGE-MRI 데이터에서 자동 심방 분할을 수행하는 앙상블 모델을 제안합니다. 모델은 Dice Similarity Coefficient (DSC) 및 95% Hausdorff distance (HD95)을 사용하여 평가되었습니다.

- **Performance Highlights**: 내부 테스트 데이터셋에서 모델은 DSC 88.41%, 98.48%, 98.45%와 HD95 1.07, 0.95, 0.64라는 성과를 달성하여 분할 정확도가 향상되었음을 보여줍니다. 이는 AF에 대한 이해를 지지하고 보다 타겟팅된 절제 전략 개발에 기여합니다.



### Enhanced Unsupervised Image-to-Image Translation Using Contrastive Learning and Histogram of Oriented Gradients (https://arxiv.org/abs/2409.16042)
Comments:
          10pages,4 figures

- **What's New**: 이 논문은 이미지-투-이미지 전환 분야에서, 대칭이 아닌(contrastive) 데이터 쌍을 초월하여 이미지의 핵심 콘텐츠와 구조를 유지하면서 변환하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Contrastive Unpaired Translation (CUT) 모델을 기반으로 하며, Histogram of Oriented Gradients (HOG) 특징을 통합하여 세미틱 레이블 없이도 이미지의 세미틱 구조를 보존할 수 있도록 합니다. HOG 특징 간의 손실(loss)을 최소화하여 이미지 품질을 향상시킵니다.

- **Performance Highlights**: GTA5 데이터셋의 합성 게임 환경을 도시 풍경(cityscapes) 데이터셋의 현실적인 장면으로 변환하는 실험에서, 환상(hallucinations)을 줄이고 이미지 품질을 향상시키는 데 있어 중요한 개선을 보였습니다.



### Deep chroma compression of tone-mapped images (https://arxiv.org/abs/2409.16032)
- **What's New**: 본 논문에서는 HDR(High Dynamic Range) 톤 매핑 이미지의 색상 압축을 위한 생성적 적대 신경망(GAN)을 제안합니다. 이는 정확한 색상 표현을 위해 이미지의 색조 속성을 고려한 손실 함수를 설계하였습니다.

- **Technical Details**: 제안된 모델은 널리 사용되는 모든 톤 매핑 연산자(TMO)와 호환되며, 색상 정확도를 향상시키기 위해 GAN 손실과 L1 손실, 색조 기반 손실을 결합한 새로운 손실 함수를 사용합니다.

- **Performance Highlights**: 모델은 기존의 색상 압축 방법에 비해 색상 정확도에서 뛰어난 성능을 보이며, 실시간 성능을 달성하여 제한된 계산 자원을 가진 장치에 적합합니다.



### VascX Models: Model Ensembles for Retinal Vascular Analysis from Color Fundus Images (https://arxiv.org/abs/2409.16016)
- **What's New**: 본 논문에서는 색 망막 사진(Color Fundus Images, CFI)에서 망막 혈관 구조를 분석하기 위한 VascX 모델 세트를 소개합니다. 이 모델들은 다양한 장치로부터의 이미지 품질과 다양한 병리학적 조건에서 강력한 성능을 보여줍니다.

- **Technical Details**: VascX 모델은 혈관(vessel), 동맥-정맥(artery-vein), 디스크(segmentation에 대한) 세분화와 중심와(fovea) 위치 지정에 대한 모델을 제공합니다. 이 모델들은 Rotterdam Study에서 수집된 공공 데이터세트와 전문가 주석을 통해 구축되었습니다.

- **Performance Highlights**: 우리의 모델은 다양한 품질의 CFI에서 기존 시스템에 비해 우수한 세분화(segmentation) 성능을 보여주었으며, 특히 중간 품질의 이미지에서 동맥-정맥 및 디스크 세분화 성능의 현저한 향상을 보였습니다. VascX 모델은 인력의 판단보다 높은 정밀도로 혈관을 세분화할 수 있습니다.



### DepMamba: Progressive Fusion Mamba for Multimodal Depression Detection (https://arxiv.org/abs/2409.15936)
- **What's New**: DepMamba라는 새로운 모델을 제안하여, 우울증 검출에 필요한 효율적이고 진보적인 오디오-비주얼 융합 방식으로 접근합니다.

- **Technical Details**: Hierarchy를 기반으로 한 문맥 모델링과 진보적인 다중모달 융합을 특징으로 하며, CNN과 Mamba 블록을 이용하여 긴 시퀀스에서 로컬부터 글로벌 특징을 추출합니다. 또한, 다중모달 협력 State Space Model (SSM)을 이용하여 각 모달리티의 상호 및 개별 정보를 추출합니다.

- **Performance Highlights**: 실험 결과, DepMamba는 기존 최첨단 모델들에 비해 정확성과 효율성에서 우수한 성능을 보였습니다.



### FedRepOpt: Gradient Re-parametrized Optimizers in Federated Learning (https://arxiv.org/abs/2409.15898)
- **What's New**: 이 논문에서는 Federated Learning(FL) 환경에서 머신 러닝 모델의 훈련을 위한 새로운 방법론인 FedRepOpt를 제안합니다. FedRepOpt는 복잡한 모델과 유사한 성능을 내는 간단한 로컬 모델을 학습할 수 있도록 하는 gradient re-parameterization 기법을 활용합니다.

- **Technical Details**: FedRepOpt는 VGG 스타일과 Ghost 스타일 모델에서의 FL 환경에 적합한 optimizer로, 복잡한 모델에서 추출된 모델 특화 하이퍼파라미터에 따라 optimizer의 gradient를 수정합니다. 이를 통해 FedRepOpt 기반 모델이 RepGhost 스타일 및 RepVGG 스타일 네트워크에 비해 성능이 각각 16.7% 및 11.4% 향상되었음을 실험적으로 입증하였습니다.

- **Performance Highlights**: FedRepOpt를 사용하는 모델은 복잡한 구조에 비해 11.7% 및 57.4%의 빠른 수렴 시간을 달성하며, FL 환경에서 높은 성능과 효율성을 보여줍니다. 또한, FedRepOpt는 저사양 및 고사양 기기에서 유효한 성능을 유지하면서도 비슷한 학습 패턴을 보입니다.



### Investigating Gender Bias in Lymph-node Segmentation with Anatomical Priors (https://arxiv.org/abs/2409.15888)
- **What's New**: 이번 연구는 방사선 치료에서의 임상 목표 부피(CTV) 세분화를 개선하기 위해 인체 해부학적 사전 지식(Anatomical Prior, AP)을 활용한 새로운 접근 방식을 소개합니다. 특히 여성 환자에서의 성별 편향(gender bias)을 완화하는 효과를 연구하였습니다.

- **Technical Details**: 연구에서는 45개의 전신 CT 3D 볼륨 및 여러 구조(예: OARs)를 세분화하고 이를 여러 가지 입력 방식을 통해 세분화 모델에 적용하였습니다. nnU-Net 프레임워크를 사용하여 데이터의 특성에 따라 모델을 조정했습니다. 평가 지표로는 Dice Score(DSC)와 Hausdorff Distance(HD)를 사용했습니다.

- **Performance Highlights**: 해부학적 사전 지식을 사용하여 CTV 세분화의 품질이 향상되었으며, 특히 복부 지역에서 성별 편향이 감소하였습니다. 이러한 접근은 자동 세분화에서의 성별 편향을 줄이는 가능성을 보여주었습니다.



### Unsupervised dMRI Artifact Detection via Angular Resolution Enhancement and Cycle Consistency Learning (https://arxiv.org/abs/2409.15883)
Comments:
          Accepted to AJCAI2024, dMRI, Unsupervised artifact detection, Angular resolution enhancement, Cycle consistency

- **What's New**: 이 연구에서는 노코멘트 학습(unsupervised learning) 기반의 새로운 dMRI 아티팩트 검출 도구 UdAD-AC를 제안하였습니다. 이 도구는 앵글 해상도 향상(angular resolution enhancement)과 사이클 일관성 학습(cycle consistency learning)을 활용하여 아티팩트 없는 dMRI 데이터를 학습하며, 아티팩트를 포함한 데이터를 자동으로 검출할 수 있습니다.

- **Technical Details**: UdAD-AC는 아티팩트 없는 dMRI 볼륨을 앵글 해상도 향상된 분수 이방성(FA) 맵으로 변환하고, 사이클 일관성 학습을 통해 아티팩트 검출 시 일관성을 유지합니다. 테스트 단계에서는 아티팩트가 포함된 dMRI 데이터에 대해 설계된 신뢰도 점수(confidence score)를 사용하여 아티팩트를 검출합니다.

- **Performance Highlights**: 실험 결과, UdAD-AC는 광범위한 공공 데이터셋에서 경쟁 방법들에 비해 우수한 성능을 보여주었으며, 아티팩트 검출 정확도가 가장 높았습니다. 이는 dMRI 연구의 신뢰성과 생산성을 높이는 데 큰 기여를 할 것입니다.



### Aided design of bridge aesthetics based on Stable Diffusion fine-tuning (https://arxiv.org/abs/2409.15812)
Comments:
          10 pages, 13 figures

- **What's New**: 이번 논문에서는 Stable Diffusion의 세부 조정(fine-tuning) 기법을 활용하여 새로운 교량(bridge) 디자인 혁신을 돕는 방법을 제시합니다.

- **Technical Details**: 교량의 실제 사진 데이터셋을 구축하고, Textual Inversion, Dreambooth, Hypernetwork, Lora의 네 가지 방법을 사용하여 Stable Diffusion을 세부 조정합니다. 이 기술들은 데이터셋 이미지의 주요 특성을 포착하여 Stable Diffusion의 개인화(customization)를 실현합니다.

- **Performance Highlights**: 세부 조정된 모델은 다양한 혁신적인 새로운 교량 유형을 생성할 수 있으며, 이는 인간 디자이너에게 풍부한 영감을 제공합니다. 이 기술은 인간 디자이너의 창의력을 촉진하는 엔진(engine of creativity)으로 작용할 수 있습니다.



### Real-Time Pedestrian Detection on IoT Edge Devices: A Lightweight Deep Learning Approach (https://arxiv.org/abs/2409.15740)
Comments:
          10 pages, 3 tables, 12 figures, article submitted to IEEE for possible publication

- **What's New**: 이번 연구는 경량화된 딥러닝(Deep Learning) 모델을 인공지능 사물인터넷(Artificial Intelligence of Things, AIoT) 에지 기기에 구현하여 보행자 탐지를 실시간으로 수행하는 방안을 제안합니다. 이를 위해 최적화된 유 아운리 룩 원(You Only Look Once, YOLO) 기반의 딥러닝 모델을 개발하였습니다.

- **Technical Details**: 연구에서는 에지 서버(edge server)의 한계인 제한된 처리 능력을 극복하기 위해 압축된 심층 신경망(Deep Neural Network, DNN) 모델을 활용합니다. 해당 경량화된 모델을 Nvidia Jetson Nano에 배포하여 실시간 보행자 탐지 테스트를 실시하였으며, 결과적으로 147 밀리세컨드의 빠른 추론 속도와 78%의 정확도를 달성했습니다.

- **Performance Highlights**: 최적화된 YOLO 모델은 2.3 프레임 매 초에 78%의 정확도로 실시간 보행자 탐지를 수행하였으며, 이는 기존 모델보다 상당한 개선을 나타냅니다.



### Autonomous Hiking Trail Navigation via Semantic Segmentation and Geometric Analysis (https://arxiv.org/abs/2409.15671)
- **What's New**: 논문에서는 자율 하이킹 경로 내비게이션을 위한 새로운 접근 방식을 소개하며, 이 방식은 경로 준수와 필요 시 비경로 이동을 조화롭게 할 수 있는 기술을 개발하였습니다. Traversability Analysis 모듈은 카메라 이미지의 의미론적 데이터와 LiDAR의 기하학적 정보를 통합하여 주변 지형에 대한 포괄적인 이해를 제공합니다.

- **Technical Details**: 이 연구에서는 하이킹 경로의 탐색성을 평가하기 위해 두 가지 주요 모듈을 개발하였습니다. 첫 번째는 기하학적 및 의미적 정보를 결합하는 Traversability 분석 모듈이고, 두 번째는 경량 로봇이 경로를 안전하게 탐색할 수 있도록 중간 목표를 선택하는 Waypoint selection 모듈입니다. 이 시스템은 LiDAR 및 스테레오 카메라로부터 수집한 데이터를 기반으로 실시간으로 경로의 탐색 가능성을 평가합니다.

- **Performance Highlights**: West Virginia University Core Arboretum에서 실험을 통해 방법론의 효과가 검증되었으며, 시뮬레이션을 통한 다양한 가중치 테스트를 통해 의미적 및 기하학적 정보 간의 균형을 평가하였습니다. 이 연구는 자율 로봇이 하이킹 경로에서 안전하게 탐색할 수 있도록 하는 가능성을 보여주었습니다.



### Personalized Federated Learning via Backbone Self-Distillation (https://arxiv.org/abs/2409.15636)
Comments:
          Pubished in ACM MMAsia 2023

- **What's New**: 이 논문에서는 개인화된 연합 학습을 촉진하기 위해 백본(Backbone) 자기 증류(Self-Distillation) 접근 방식을 제안합니다. 각 클라이언트가 로컬 모델을 훈련하고 오직 백본 가중치만 서버에 전송하는 방식을 사용합니다.

- **Technical Details**: 각 클라이언트 모델은 공유 백본과 개인 헤드(Head)로 나뉘어 있으며, 서버는 백본 가중치만 집계하여 글로벌 백본을 구축합니다. 그런 다음 각 클라이언트는 글로벌 백본을 교사(Teacher)로 사용하여 로컬 백본을 업데이트 합니다.

- **Performance Highlights**: 12개의 최신 접근 방식과 비교한 실험 결과, 제안된 방법이 성능을 크게 향상시키며, 글로벌 지식 전이(Global Knowledge Transfer)를 통해 클라이언트 모델의 개인화를 효과적으로 지원함을 보여주었습니다.



### MapEx: Indoor Structure Exploration with Probabilistic Information Gain from Global Map Predictions (https://arxiv.org/abs/2409.15590)
Comments:
          7 pages

- **What's New**: 본 연구에서는 구조화된 실내 환경을 탐색하는 로봇을 위한 새로운 탐색 프레임워크인 MapEx를 제안합니다. MapEx는 예측된 지도를 사용해 확률적 센서 모델을 형성하여 정보 이득(metrics of information gain)을 추정합니다. 특히, 다양한 예측된 지도를 생성하고 계산된 변동성과 추정된 가시 영역을 고려하여 탐색 계획을 수립합니다.

- **Technical Details**: MapEx는 관찰된 정보를 기반으로 여러 개의 예측된 지도를 생성하며, 여기서 평균 및 분산 지도를 계산합니다. 이 프레임워크는 여러 예측된 지도의 변동성과 가시 영역을 고려하여 주어진 관점(viewpoint)의 정보 이득을 계산합니다. 또한, MapEx는 LiDAR 센서를 사용하여 환경을 탐색하며, 이전의 탐사 방식과 비교하여 더 효율적인 정보를 얻을 수 있습니다.

- **Performance Highlights**: KTH 데이터셋을 이용한 실험에서, MapEx는 기존의 지도 예측 기반 탐색 방법보다 평균 12.4% 개선된 성능을 보였고, 최근접 전선(frontier) 접근법보다 25.4% 향상된 결과를 달성했습니다.



### Mixing Data-driven and Geometric Models for Satellite Docking Port State Estimation using an RGB or Event Camera (https://arxiv.org/abs/2409.15581)
Comments:
          Submitted to IEEE ICRA 2025

- **What's New**: 이 논문은 자동화된 위성 도킹 포트 탐지 및 상태 추정을 위한 파이프라인을 제안하며, 이는 모노큘러 비전 데이터(standard RGB sensing 또는 event camera)를 활용하여 비용을 절감하고 궤도 쓰레기를 줄이는 데 기여합니다.

- **Technical Details**: 제안된 파이프라인은 Lockheed Martin Mission Augmentation Port (LM-MAP)를 목표로 하며, 단순 기하학적 모델과 데이터 기반 기법을 조합하여 데이터 효율적인 경량 추정을 실현합니다. 이 방법은 RANndom SAmple Consensus (RANSAC) 접근 방식을 사용하여 포트의 6-자유도(DoF) 위치 추정을 가능하게 합니다.

- **Performance Highlights**: 실험을 통해 RGB와 event camera 데이터 모두에 대해 알고리즘의 성능을 비교하였으며, 두 가지 데이터 소스를 모두 독립적으로 처리할 수 있는 유연성을 갖추고 있습니다. 이는 위성 유지 관리 및 궤도 클러터 방지에 기여할 것으로 기대됩니다.



### A Novel Framework for the Automated Characterization of Gram-Stained Blood Culture Slides Using a Large-Scale Vision Transformer (https://arxiv.org/abs/2409.15546)
- **What's New**: 본 연구에서는 Gram 염색을 이용한 전체 슬라이드 이미지 (WSI) 분석을 위한 인공지능 보조 특성화 프레임워크를 새롭게 소개합니다. 이는 혈류 감염 진단을 위한 중요한 초기 데이터를 제공합니다.

- **Technical Details**: 이 모델은 transformer 기반으로 개발되어 이전의 convolutional neural network (CNN) 방법에 비해 대규모 데이터 세트에 대해 더욱 확장 가능합니다. 수동으로 패치 레벨 주석을 필요로 하지 않기 때문입니다. 본 연구에서는 Dartmouth-Hitchcock Medical Center에서 수집한 대규모 Gram 염색 데이터 세트를 사용하여 다섯 가지 주요 Gram 염색 WSI 카테고리를 분류했습니다.

- **Performance Highlights**: 모델의 분류 정확도는 0.858 (95% CI: 0.805, 0.905)이며, AUC는 0.952 (95% CI: 0.922, 0.976)로, 475개 슬라이드를 사용한 다섯 겹 중첩 교차 검증을 통해 확인되었습니다. 또한, 추가적인 파인튜닝 없이도 외부 데이터 세트에서도 강력한 성능을 달성하였습니다.



### Speech2rtMRI: Speech-Guided Diffusion Model for Real-time MRI Video of the Vocal Tract during Speech (https://arxiv.org/abs/2409.15525)
Comments:
          4 pages

- **What's New**: 이 연구는 Magnetic Resonance Imaging (MRI) 비디오를 기반으로 말하는 동안의 외부 기관(articulator) 모션을 시각적으로 표현하기 위한 데이터 기반 방법을 소개하고 있습니다. 이 방법은 사전 학습된 스피치 모델을 활용하여 일반화된 시각적 도메인을 구축하고, Speech-to-video diffusion model을 통해 새로운 데이터에 적용할 수 있습니다.

- **Technical Details**: 연구는 Speech-Conditioned Diffusion 모델인 Speech2rtMRI를 제안하여, 실시간 MRI 비디오에서 발음 운동을 합성합니다. 모델은 원시 오디오와 기관 운동 비디오 세트를 동기화하여 학습하며, Pre-trained speech model(WavLM 등)을 사용하여 비디오 생성을 위한 임베딩을 보다 효과적으로 제공합니다. 훈련과 샘플링 두 단계로 이루어진 이 접근법은 3D Diffusion U-Net을 통해 시각적 데이터를 생성합니다.

- **Performance Highlights**: 실험 결과, WavLM 모델이 전체적으로 가장 높은 일반화 점수를 기록했으며, 생성된 비디오에서 자주 발생하는 부자연스러운 혀 운동과 비디오 왜곡 현상이 있는 것으로 나타났습니다. 인간 평가에서는 생성된 MRI 비디오의 관련성, 진정성, 그리고 제한 사항이 평가되었습니다.



### MATCH POLICY: A Simple Pipeline from Point Cloud Registration to Manipulation Policies (https://arxiv.org/abs/2409.15517)
Comments:
          project url: this https URL

- **What's New**: MATCH POLICY는 로봇의 다양한 조작 작업에서 높은 정확도를 요구하는 픽-플레이스(pick and place) 작업을 해결하는 새로운 파이프라인을 제안합니다. 기존 방법과 달리 MATCH POLICY는 작동 예시를 기반으로 목표를 등록하는 점에 초점을 맞추어 학습 없이 조작 정책을 실현할 수 있는 간단한 방법입니다.

- **Technical Details**: MATCH POLICY는 피크 및 플레이스(pick and place) 대상으로 지정된 점 구름(point cloud) 등록(registration) 작업으로 로봇 조작 정책 학습을전환하는 것을 특징으로 합니다. 이 방법은 RANSAC 및 ICP와 같은 최적화 기반 방법을 활용하여 이전 작업에서 수집한 데모를 활용하여 픽-플레이스 정책을 즉시 생성합니다.

- **Performance Highlights**: 이 방법은 RLBench 벤치마크에서 여러 강력한 기준선과 비교하여 다양한 작업에서 최첨단 성능을 보이며, 오직 하나의 데모로도 신뢰할 수 있는 성능을 달성할 수 있습니다. 또한 다양한 카메라 설정에서도 높은 적응력을 보이며, 장기 작업 및 관절 객체에 대해서도 효과적으로 작동합니다.



### Bayesian computation with generative diffusion models by Multilevel Monte Carlo (https://arxiv.org/abs/2409.15511)
Comments:
          13 images

- **What's New**: 본 논문에서는 Bayesian computation에서 diffusion models (확산 모델)의 계산 비용을 크게 줄이는 Multilevel Monte Carlo (MLMC) 전략을 제안하고 있습니다. 이는 다양한 정확도를 가진 모델을 결합하여 비용-정확도 트레이드오프를 활용함으로써 달성됩니다.

- **Technical Details**: MLMC 접근법은 Bayesian 통계 프레임워크 내에서 고차원 양 x∈ℝ^n을 관측 데이터 y∈ℝ^m로부터 추정함으로써, posterior distribution (후방 분포)을 생성하는 데 사용됩니다. 이는 관측 모델의 결정론적 측면을 인코딩하는 선형 Gaussian 모델을 포함하며, 본 연구는 다양한 기계 학습 기술을 통해 이러한 모델의 효율성을 향상시키는 방법을 설명합니다.

- **Performance Highlights**: 이 제안된 MLMC 접근법은 세 가지 전형적인 계산 영상 문제에서 검증되었으며, 기존의 Monte Carlo 평균 방식에 비해 계산 비용이 4배에서 8배까지 감소하는 결과를 보였습니다.



### Adenocarcinoma Segmentation Using Pre-trained Swin-UNet with Parallel Cross-Attention for Multi-Domain Imaging (https://arxiv.org/abs/2409.15501)
Comments:
          6 pages 2 figures

- **What's New**: 이 논문에서는 다양한 장기와 스캐너 간의 선종성(adenocarcinoma) 세분화를 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 Swin-UNet 아키텍처와 병렬 교차 주의 모듈을 결합하여 도메인 변동과 형태적 변화를 고려합니다.

- **Technical Details**: 이 프레임워크는 사전 훈련된 인코더(encoder)와 Swin-UNet 구조로 구성되어 있으며, 다중 스케일 특징(feature)을 추출할 수 있는 장점을 가지고 있습니다. 인코더는 각각 다른 장기에서의 세분화 병목을 방지하기 위해 이미지넷(ImageNet) 데이터셋에서 사전 훈련되었습니다. 프레임워크는 두 개의 주요 작업, 즉 Cross-Organ and Cross-Scanner Adenocarcinoma Segmentation에 대해 평가되었습니다.

- **Performance Highlights**: 제안된 모델은 Cross-Organ 및 Cross-Scanner 세분화 작업에서 각각 0.7469 및 0.7597의 점수를 기록하여 다양한 조건에서도 일관된 성능을 보여주었습니다. 특히, 모델의 세분화 결과는 해부학적 변동성이 있는 상태에서도 종양 경계를 정확히 예측하는 데 강점을 보였으며, 이는 강력한 특성 추출 및 전이 학습(transfer learning) 능력 덕분으로 분석됩니다.



### Autonomous Exploration and Semantic Updating of Large-Scale Indoor Environments with Mobile Robots (https://arxiv.org/abs/2409.15493)
Comments:
          7 pages, 7 figures. Project page is available at this https URL

- **What's New**: 본 연구에서는 모바일 로봇이 자율적으로 미지의 환경을 탐색하고, 해당 환경의 의미 있는 지도를 생성 및 업데이트할 수 있는 새로운 로봇 시스템을 소개합니다. 이 시스템은 LiDAR 스캐너와 RGB-D 카메라를 활용하여 2D occupancy grid mapping과 객체 인식을 수행합니다.

- **Technical Details**: 우리는 2D occupancy grid map과 객체 의미 정보를 결합한 새로운 의미적 맵 표현 방식을 도입하였습니다. 이 표현 방식을 통해 topological map의 노드를 추가하거나 삭제하여 쉽게 의미를 업데이트할 수 있습니다. 또한, 로봇은 

- LiDAR를 통해 환경의 2D occupancy grid map을 생성하고
- 탐색 경로를 계획하여 따라가며
- 실시간으로 RGB-D 카메라로 객체를 감지해 2D occupancy grid에 추가하는 방식으로 semantic map을 생성합니다.

- **Performance Highlights**: 우리는 이 시스템을 텍사스 대학교의 ECSS 빌딩 4층에서 테스트하였으며, 로봇은 93m x 90m의 공간을 탐색하여 효율적으로 semantic map을 만들고, 환경 내에서 객체가 이동된 후에도 의미 지도를 성공적으로 업데이트하는 성능을 보였습니다.



### Adapting Segment Anything Model for Unseen Object Instance Segmentation (https://arxiv.org/abs/2409.15481)
Comments:
          Submitted to ICRA 2025

- **What's New**: 이번 논문에서는 Unseen Object Instance Segmentation (UOIS) 작업을 위한 데이터 효율적인 솔루션인 UOIS-SAM을 제안합니다. 이 모델은 Segment Anything Model (SAM)의 높은 정확도와 강력한 일반화 능력을 활용합니다.

- **Technical Details**: UOIS-SAM은 Heatmap-based Prompt Generator (HPG)와 Hierarchical Discrimination Network (HDNet) 두 가지 주요 구성 요소를 통합합니다. HPG는 클래스에 구애받지 않는 포인트 프롬프트를 생성하며, HDNet은 SAM의 마스크 디코더를 대체하여 배경 혼동과 과분할(over-segmentation) 문제를 완화합니다.

- **Performance Highlights**: UOIS-SAM은 이전 방법들에 비해 훈련 샘플의 10%만을 사용하여 OCID, OSD와 같은 여러 데이터셋에서 최첨단 성능을 달성하였으며, 복잡한 테이블 환경에서의 효과성과 강건성을 강조합니다.



### Tag Map: A Text-Based Map for Spatial Reasoning and Navigation with Large Language Models (https://arxiv.org/abs/2409.15451)
- **What's New**: 본 연구에서는 대형 이미지 인식 모델을 활용하여 수천 개의 의미 클래스를 명시적으로 표현할 수 있는 텍스트 기반 맵을 제안합니다. 이 맵은 대형 언어 모델(LLM)과 쉽게 통합되며, 로봇이 사용자 작업을 해결하기 위해 PLANS (작업 계획)를 생성하는 데 필요한 장면 정보를 제공합니다.

- **Technical Details**: 제안된 태그 맵은 이미지 태깅 모델이 인식한 고유 엔티티(태그)를 저장하고, 각 태그가 인식된 시점(뷰포인트)과 연관됩니다. 이 맵은 메모리의 효율성을 극대화하기 위해 비구조적인 데이터베이스로 구현됩니다. 또한, 3D 로컬라이제이션을 통해 태그와 관련된 지역을 생성하는 과정을 설명합니다.

- **Performance Highlights**: 정량적 실험을 통해 제안된 태그 맵의 로컬라이제이션 성능이 최신 개방형 어휘 맵과 비교하여 정확도와 재현성을 유지하면서도 사용하는 메모리를 몇 배나 줄일 수 있음을 보여줍니다. 실제 로봇 실험에서도 태그 맵이 LLM을 기반으로 사용자 요청을 처리하고 실행 가능한 내비게이션 계획을 생성하는 데 효과적임을 입증했습니다.



### BurstM: Deep Burst Multi-scale SR using Fourier Space with Optical Flow (https://arxiv.org/abs/2409.15384)
Comments:
          12 pages

- **What's New**: 새로운 접근 방식인 Deep Burst Multi-scale SR(BurstM)을 소개합니다. 이 방법은 Optical Flow를 활용하여 정확한 프레임 정렬을 가능하게 하고, 각 프레임의 연속적인 Fourier 계수를 예측하여 고주파 텍스처를 표현하는 데 중점을 둡니다.

- **Technical Details**: BurstM은 Optical Flow를 이용하여 프레임 간의 상관된 오프셋을 제공하며, Well-aligned 정보를 통해 고주파 텍스처를 잘 표현할 수 있도록 Fourier 정보를 추정합니다. 이 방법은 고정된 SR 스케일 문제를 해결하고, 다양한 스케일의 슈퍼 해상도(SR) 처리를 지원하여 계산적 효율성을 갖춥니다.

- **Performance Highlights**: BurstM은 기존 MFSR 방법에 비해 이미지 품질과 계산 효율성에서 우수한 성능을 보이며, 광범위한 스케일에서의 유연성을 제공합니다.



### DS2TA: Denoising Spiking Transformer with Attenuated Spatiotemporal Attention (https://arxiv.org/abs/2409.15375)
Comments:
          arXiv admin note: text overlap with arXiv:2311.09376

- **What's New**: 본 논문에서는 비전을 위한 새로운 아키텍처인 DS2TA(Denoising Spiking transformer with Attenuated SpatioTemporal Attention)를 소개하며, 이는 템포럴 차원에서 조정된 스페이쇼템포럴 어텐션(SpatioTemporal Attention) 메커니즘을 도입하여 기존 스파이킹 트랜스포머의 한계를 뛰어넘습니다.

- **Technical Details**: DS2TA는 입력 발화의 시간 및 공간에서 발생하는 상관관계를 고려하여, spiking 쿼리, 키, 값 및 최종 출력을 계산하는 TASA(Temporally Attenuated Spatiotemporal Attention)를 구현함으로써, 스파이킹 뉴런의 계산 능력을 최대한 활용합니다. 또한, 비선형 스파이킹 어텐션 디노이저(nonlinear spiking attention denoisers)를 사용하여 주의 맵의 강인성과 표현력을 향상시킵니다.

- **Performance Highlights**: DS2TA는 CIFAR10에서 94.92%, CIFAR100에서 77.47%, CIFAR10-DVS에서 79.1%, DVS-Gesture에서 94.44%의 top-1 정확도로 여러 정적 이미지와 동적 신경형 하드웨어 데이터셋에서 최첨단 성능을 입증하였습니다.



### Explainable AI for Autism Diagnosis: Identifying Critical Brain Regions Using fMRI Data (https://arxiv.org/abs/2409.15374)
- **What's New**: 이 연구는 자폐 스펙트럼 장애(ASD)의 조기 진단과 개입을 위한 새로운 접근 방식을 제시하며, 기존의 진단 모델의 해석 가능성을 높이는 데 초점을 맞추고 있습니다. 이 연구는 ASD를 정확히 분류할 뿐만 아니라, 그 작동 방식에 대한 설명 가능한 통찰력을 제공하는 딥 러닝(DL) 모델을 개발하려고 합니다.

- **Technical Details**: 사용된 데이터셋은 884개의 샘플로 구성된 자폐 뇌 영상 데이터 교환(ABIDE)의 전처리된 버전입니다. 이 연구는 resting-state functional Magnetic Resonance Imaging (fMRI) 데이터 분석을 통해 ASD의 잠재 바이오마커를 식별하고, Remove And Retrain (ROAR) 기술을 사용하여 해석 가능성 방법을 벤치마킹합니다.

- **Performance Highlights**: 모델은 ASD를 정확하게 분류할 수 있으며, ASD와 일반인 집단(Typical Controls) 간의 비정상적인 뇌 영역을 강조합니다. 이러한 발견은 다양한 데이터셋과 방법론에서의 선행 연구들에 의해 검증되었으며, 향후 ASD의 조기 진단 및 신경 기초 이해에 중요한 의미를 갖습니다.



### Damage detection in an uncertain nonlinear beam based on stochastic Volterra series (https://arxiv.org/abs/2409.15349)
- **What's New**: 본 논문은 비선형 행동을 보이는 기계 시스템에서 손상 감지를 위한 확률적(Probabilistic)Volterra 시리즈 접근 방식을 제안합니다. 이를 통해 시스템의 비선형 특성과 데이터 변동성을 동시에 설명하는 새로운 모델을 개발하였습니다.

- **Technical Details**: Volterra 시리즈는 선형 컨볼루션 개념의 일반화로, 시스템의 선형 및 비선형 기여를 분리하는 데 유리합니다. 본 연구에서는 Kautz 함수와 함께 확률적 Volterra 시리즈를 활용하여 비선형 동작을 분석하고, 시스템의 불확실성으로 인해 발생하는 문제를 극복하고자 하였습니다. 특히, 비선형 강성(linear stiffness)과 감쇠 계수(damping coefficient)의 변화를 통해 시스템의 불확실성을 시뮬레이션합니다.

- **Performance Highlights**: 실험 결과, 높은 차수의 Volterra 커널을 고려하여 비선형 분석을 수행했을 때, 작고 확률적인 신뢰도(probability confidence)를 가진 균열을 감지할 수 있음을 보여주었습니다. 이는 불확실성이 존재하는 경우에도 손상 감지 절차를 효과적으로 진행할 수 있도록 하며, 기존의 SHM 기술을 보완하는 중요한 접근을 제시합니다.



### A Lightweight GAN-Based Image Fusion Algorithm for Visible and Infrared Images (https://arxiv.org/abs/2409.15332)
- **What's New**: 이 논문은 성능과 효율성의 균형을 강조하여 가시광선(visible light) 이미지와 적외선(infrared) 이미지를 병합하기 위한 경량화된 이미지 융합(image fusion) 알고리즘을 제시합니다.

- **Technical Details**: 제안된 방법은 Generative Adversarial Network (GAN)에서 생성기(generator)를 개선하기 위해 Convolutional Block Attention Module (CBAM)을 통합하고, Depthwise Separable Convolution (DSConv)을 사용하여 계산의 효율성을 높입니다. 이러한 혁신은 모델의 계산 비용을 크게 줄이며, 파라미터 수와 추론 지연(inference latency)을 낮춥니다.

- **Performance Highlights**: M3FD 데이터셋을 이용한 비교 실험을 통해 제안된 알고리즘이 융합 품질(fusion quality) 면에서 유사한 이미지 융합 방법들을 능가하며, 임베디드 장치에서 배포하기에 더 자원 효율적인 솔루션을 제공함을 보여주었습니다. 경량화된 설계의 효과는 광범위한 ablation 연구를 통해 검증되었습니다.



### Visual Prompting in Multimodal Large Language Models: A Survey (https://arxiv.org/abs/2409.15310)
Comments:
          10 pages

- **What's New**: 본 논문은 시각적 프롬프트(visual prompting) 방법에 대한 첫 포괄적 설문조사로, 다중 모달 대규모 언어 모델(MLLMs)의 발전을 다룬다. 이를 통해 기존의 텍스트 기반 프롬프트보다 미세 조정된 비주얼 인스트럭션을 제공하여 MLLMs의 시각적 이해 및 추론 능력을 향상하는 다양한 접근 방식을 소개한다.

- **Technical Details**: MLLMs는 사전 훈련된 대형 언어 모델(LLMs)과 시각적 능력을 결합한 모델로, 단어뿐만 아니라 이미지에 대한 비주얼 프롬프트를 생성하여 더 높은 수준의 상황 이해를 가능하게 한다. 논문은 비주얼 프롬프트의 분류, 자동 프롬프트 주석 생성(generative methods), 시각적 부합(alignment) 방안 및 오브젝트 참조(object referring)에 대한 내용이 포함되어 있다.

- **Performance Highlights**: 강화된 시각적 프롬프트 방법들은 MLLM의 시각적 감각을 더욱 정교하게 하여, 보다 정확한 시각적 기초(visual grounding), 객체 참조 및 조합적 추론(compositional reasoning) 기능을 제공한다고 제안된다.



### PixWizard: Versatile Image-to-Image Visual Assistant with Open-Language Instructions (https://arxiv.org/abs/2409.15278)
Comments:
          Code is released at this https URL

- **What's New**: 이번 논문에서는 PixWizard라는 다재다능한 이미지-투-이미지 비주얼 어시스턴트를 제시합니다. 이 모델은 자연어 명령에 기반하여 이미지 생성, 조작 및 변환을 수행할 수 있으며, 다양한 비전 작업을 통합한 이미지-텍스트-투-이미지 생성 프레임워크를 구축했습니다.

- **Technical Details**: PixWizard는 Diffusion Transformers (DiT)를 기반 모델로 사용하고 있으며, 다양한 해상도의 이미지를 처리할 수 있는 유연한 메커니즘을 도입했습니다. 이를 통해 입력의 종횡비에 따라 동적으로 이미지를 처리할 수 있으며, 구조 인식 (structure-aware) 및 의미 인식 (semantic-aware) 가이드를 포함하여 효과적인 정보 융합을 지원합니다.

- **Performance Highlights**: 실험 결과, PixWizard는 다양한 해상도에서 인상적인 생성 및 이해 능력을 보여주었으며, 훈련 중 접하지 않았던 작업에 대해서도 뛰어난 일반화 능력을 보였습니다. 이는 PixWizard가 강력한 인터랙티브 이미지-투-이미지 비주얼 어시스턴트로서의 위상을 높여줍니다.



### MaterialFusion: Enhancing Inverse Rendering with Material Diffusion Priors (https://arxiv.org/abs/2409.15273)
Comments:
          Project Page: this https URL

- **What's New**: 본 연구는 MaterialFusion이라는 새로운 3D 역 렌더링 파이프라인을 소개하며, 텍스처와 재료 특성에 대한 2D prior를 결합하여 다중 조명 환경에서도 신뢰할 수 있는 복원력을 가지고 있다고 주장합니다.

- **Technical Details**: 이 연구는 안정적인 2D diffusion 모델인 StableMaterial을 활용하여 주어진 입력 외관으로부터 가장 가능성 높은 albedo(반사율)와 재료를 추정합니다. 또한, score distillation sampling (SDS)을 사용하여 albedo 및 재료의 최적화를 유도함으로써 이전 연구에 비해 재조명(relighting) 성능을 향상시킵니다.

- **Performance Highlights**: MaterialFusion은 NeRF Synthetic, NeRFactor 데이터셋, BlenderVault 데이터셋 및 Stanford-ORB 데이터셋에서 검증되어, 새로운 조명 조건 아래에서 복원된 객체의 외관을 크게 개선함을 보여줍니다.



### ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild (https://arxiv.org/abs/2409.15269)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 ReLoo라는 새로운 방법을 소개합니다. 이 방법은 느슨한 의류를 착용한 인물의 3D 모델을 높은 품질로 재구성할 수 있는 기술을 제공합니다. 기존의 방법들이 타이트한 의류에만 적합했으나, ReLoo는 다양한 의류의 비정형적인 변형을 효과적으로 처리할 수 있습니다.

- **Technical Details**: ReLoo는 layered neural human representation을 기반으로 하여, 인체의 내부와 외부 의류를 분리하여 표현합니다. 또한 non-hierarchical virtual bone deformation module을 통해 의류의 자연스러운 움직임을 표현할 수 있습니다. 이 모델은 다계층 차별적 볼륨 렌더링(multi-layer differentiable volume rendering)을 활용하여 인체와 의류의 형상, 외관 및 변형을 최적화합니다.

- **Performance Highlights**: ReLoo는 MonoLoose라는 새로운 데이터셋을 사용하여 느슨한 의류를 착용한 인물의 3D 재구성에서 기존의 방법들보다 우월한 성능을 보여주었습니다. 실험 결과는 인체 모양과 의류의 변화를 정확하게 캡처하며, 이전 기술 대비 높은 품질의 시각적 결과를 제공합니다.



### S$^2$AG-Vid: Enhancing Multi-Motion Alignment in Video Diffusion Models via Spatial and Syntactic Attention-Based Guidanc (https://arxiv.org/abs/2409.15259)
- **What's New**: 본 논문에서는 다중 객체가 포함된 텍스트-비디오(T2V) 생성에서의 객체와 동작 정렬 문제를 해결하기 위해 S$^2$AG-Vid라는 새로운 추론 단계 최적화 방법을 제안합니다. 이 방법은 훈련 없이 적용될 수 있으며, 다양한 객체가 특정 동작과 더 잘 정렬되도록 돕습니다.

- **Technical Details**: S$^2$AG-Vid는 첫째, 초기 노이즈 제거 과정에서 공간 위치 기반 교차 주의(cross-attention, CA) 제약 조건을 적용해 여러 개의 명사가 올바른 주제 영역에 주의를 기울일 수 있도록 합니다. 둘째, 동작-주제 결합을 강화하기 위해 문법 안내 대조 제약(syntax-guided contrastive constraint)을 실시하며, 이는 동사 CA 맵과 해당 명사 CA 맵 간의 상관관계를 개선하는 것을 목표로 합니다.

- **Performance Highlights**: 정성적 및 정량적 평가에서 S$^2$AG-Vid는 기존 모델에 비해 매우 높은 품질의 비디오를 생성하며, 객체와 동작 간의 일관성을 크게 향상시키는 데 성공하였습니다.



### ReVLA: Reverting Visual Domain Limitation of Robotic Foundation Models (https://arxiv.org/abs/2409.15250)
- **What's New**: 최근 대규모 언어 모델과 대규모 로봇 데이터셋의 발전이 로봇 모델의 패러다임 전환을 이끌며, 다양한 작업, 장면 및 로봇 양식에 적응할 수 있는 일반화 능력을 가진 모델로 변화했습니다. 특히 Open Vision Language Action 모델은 다양한 작업에서 강력한 성능을 보여줍니다.

- **Technical Details**: 이 연구에서는 3가지 기존 로봇 기초 모델의 시각적 일반화 능력을 연구하고, 이에 대한 평가 프레임워크를 제안합니다. 기존 모델들은 시각적 도메인 아웃오브도메인(Out-of-Domain, OOD) 시나리오에 대한 강건성을 보이지 않으며, 이는 훈련 데이터의 변동성 부족 및 기억상실(catastrophic forgetting) 때문일 수 있습니다. OpenVLA 모델이 이러한 문제를 겪는 것을 보여주고, 모델 병합(model merging)을 기반으로 한 점진적 백본 리버설(grand backbone reversal) 접근법을 제안하여 재학습 후 시각적 일반화 능력을 복원합니다.

- **Performance Highlights**: ReVLA 모델은 OpenVLA에 비해 시각적 OOD 작업에서 그립(grasping) 및 리프팅(lifting) 성능에서 각각 77%와 66%의 향상을 달성하였습니다.



### Enhancing Pedestrian Trajectory Prediction with Crowd Trip Information (https://arxiv.org/abs/2409.15224)
- **What's New**: 본 논문은 보행자 궤적 예측을 위한 새로운 접근 방식으로, 군중의 이동 정보를 새로운 모달리티로 도입한 RNTransformer 모델을 제안합니다. 이 모델은 사회적 상호작용 및 도로 환경을 고려하여 보행자의 행동을 보다 정확하게 예측할 수 있도록 설계되었습니다.

- **Technical Details**: RNTransformer는 군중 이동 정보를 활용하여 사회적 상호작용에 대한 글로벌 정보를 캡처하는 일반 모델입니다. 이 모델은 여러 소셜 인지 지역 보행자 궤적 예측 모델과 결합되어 성능을 입증하였으며, Social-LSTM에서 ADE/FDE 지표에서 각각 1.3/2.2%, Social-STGCNN에서 6.5/28.4%, S-Implicit에서 8.6/4.3% 향상을 보여주었습니다. 이를 통해 다양한 데이터세트에서 보행자 궤적 예측의 정확도가 크게 향상되었음을 확인하였습니다.

- **Performance Highlights**: RNTransformer는 다양한 기저 보행자 궤적 예측 모델에 대한 정확도를 개선하는 데 성공했으며, 기본 모델에 비해 보행자 목표를 보다 정확하게 샘플링하는 데 기여했습니다. 본 연구에서 개발된 모델은 다양한 데이터세트에 대해 폭넓은 실험을 통해 검증되었습니다.



### HydroVision: LiDAR-Guided Hydrometric Prediction with Vision Transformers and Hybrid Graph Learning (https://arxiv.org/abs/2409.15213)
- **What's New**: 본 연구는 수문 예측에서 수면 상승 예측을 위해 LiDAR 데이터와 Vision Transformer (ViT)를 사용하여 지형 고도가 수역의 흐름과 연결성에 미치는 영향을 통합합니다.

- **Technical Details**: 연구는 GRU 블록을 그래프 합성과 결합하여 시계열 데이터 내의 공간적 의존성을 모델링합니다. 정적 그래프는 LiDAR 데이터에서 파생되며, 동적 그래프는 시간에 따른 변화를 고려합니다. 이 하이브리드 그래프 학습 구조는 수문 시스템 내의 복잡한 상호작용을 포착합니다.

- **Performance Highlights**: 퀘벡의 여러 수위 관측소에서 실험한 결과, 제안된 방법은 예측 오류를 평균 10% 감소시켰고, 예측 기간이 길어질수록 개선 효과가 더욱 두드러졌습니다.



### HOTVCOM: Generating Buzzworthy Comments for Videos (https://arxiv.org/abs/2409.15196)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 중국어 비디오 핫 댓글 생성을 위한 새로운 데이터셋 HotVCom을 구축하였으며, 94,000개의 다양한 비디오와 1억 3천7백만 개의 댓글을 포함하고 있습니다. 또한, 비디오 핫 댓글 생성 프레임워크 ComHeat를 소개합니다.

- **Technical Details**: ComHeat는 시각적, 청각적, 텍스트 데이터를 통합하여 중국 비디오 데이터셋에서 영향력 있는 핫 댓글을 생성합니다. 이 프레임워크는 Supervised Fine-Tuning 기법을 통해 초기 댓글을 생성하고, 강화 학습을 통해 개선합니다. 또한 Tree-of-Thought 접근법을 활용하여 댓글의 품질을 높입니다.

- **Performance Highlights**: ComHeat 프레임워크는 새로 구축된 HotVCom 데이터셋과 기존 데이터셋에서 다른 기준 모델들을 초월하는 성능을 보여줍니다. 새로운 종합 평가 지표를 통해 댓글의 유용성, 관련성, 창의성 및 사용자 참여도를 강조합니다.



### Interpretability-Guided Test-Time Adversarial Defens (https://arxiv.org/abs/2409.15190)
Comments:
          ECCV 2024. Project Page: this https URL

- **What's New**: 본 논문에서는 해석 가능성(interpretablity) 기반의 뉴런 중요도 순위(neuron importance ranking) 방법을 이용하여 출력 클래스에 중요한 뉴런을 식별하는 혁신적이고 저비용의 테스트 타임 적대적 방어(test-time adversarial defense) 방식을 제안합니다. 우리의 방법은 훈련 없이도 실행되어 로버스트-정확도 무역비율을 상당히 개선하며, 최소한의 계산 오버헤드를 맞습니다.

- **Technical Details**: 우리의 IG-Defense(Interpretability-Guided Defense) 방법은 뉴런의 중요도 순위를 기반으로 클래스별 뉴런의 중요도를 계산하여 얻은 인사이트를 활용합니다. 주요 아이디어는 중요한 뉴런이 아닌 뉴런의 활동을 마스킹(masking)하여 인식된 활성화 이동을 제한하는 것입니다.

- **Performance Highlights**: IG-Defense는 CIFAR10, CIFAR100, ImageNet-1k에서 각각 평균 2.6%, 4.9%, 2.8%의 성능 향상을 보였으며, 강력한 적응형 공격(adaptive attacks) 하에서도 1.5%의 성능 향상을 기록하였습니다. IG-Defense는 가장 효율적인 테스트 타임 방어 방법 중 하나이며, 기존 방법보다 4배 빠른 성능을 보여줍니다.



### MIMAFace: Face Animation via Motion-Identity Modulated Appearance Feature Learning (https://arxiv.org/abs/2409.15179)
- **What's New**: 이 논문에서는 현재 확산 기반 얼굴 애니메이션 방법의 한계를 살펴보고, 모션과 정체성 수준에서 CLIP(Contrastive Language–Image Pre-training) 특징을 조절하는 새로운 모듈을 제안합니다. 이를 통해 더 높은 품질의 애니메이션 비디오를 생성할 수 있게 됩니다.

- **Technical Details**: Motion-Identity Modulated Appearance Learning Module (MIA)와 Inter-clip Affinity Learning Module (ICA)를 도입하여 모션과 정체성을 동시에 조절하고, 클립 간의 시간적 관계를 모델링합니다. MIA는 CLIP 특징을 조절하여 고해상도 얼굴 텍스처(예: 주름, 근육 수축)를 생성하고, ICA는 훈련 데이터와 생성된 프레임 간의 의미/색상 불연속성을 해소합니다.

- **Performance Highlights**: 이 방법을 사용하여 정밀한 얼굴 모션 제어(예: 표정과 시선)를 달성하고, 신뢰할 수 있는 정체성 보존과 함께 클립 내 및 클립 간 시간적 일관성을 유지하는 애니메이션 비디오를 생성할 수 있습니다. 실험 결과, 제안하는 방법이 기존의 다른 방법들보다 뛰어난 성능을 보임을 보여줍니다.



### SpikeGS: Learning 3D Gaussian Fields from Continuous Spike Stream (https://arxiv.org/abs/2409.15176)
Comments:
          Accepted by ACCV 2024. Project page: this https URL

- **What's New**: 이번 논문에서는 스파이크 카메라(spike camera)의 스파이크 스트림(spike stream)만을 사용하여 3D 가우시안 필드(3D Gaussian fields)를 학습하는 최초의 방법인 SpikeGS를 소개합니다. SpikeGS는 고품질의 실시간 렌더링(real-time rendering)을 달성하며, 노이즈가 많은 저조도(low-light) 환경에서도 높인 강인함(robustness)을 보여줍니다.

- **Technical Details**: SpikeGS는 3DGS(3D Gaussian Splatting) 기법을 기반으로 한 미분 가능(diff differentiable) 스파이크 스트림 렌더링 프레임워크를 설계하였습니다. 이 프레임워크는 다중 뷰 일관성(multi-view consistency)과 타일 기반의 멀티스레드 병렬 렌더링(tile-based multi-threaded parallel rendering) 메커니즘을 활용하여, 높은 품질의 렌더링 결과를 생성합니다. 추가로, 다양한 조명 조건에서 일반화될 수 있는 스파이크 렌더링 손실 함수(spike rendering loss function)를 제안하였습니다.

- **Performance Highlights**: 실험 결과, 본 방법은 실제 및 합성 데이터셋(synthetic datasets) 모두에서 기존의 최신 기술 대비 렌더링 품질과 속도에서 뛰어난 성능을 나타냈습니다.



### FusionRF: High-Fidelity Satellite Neural Radiance Fields from Multispectral and Panchromatic Acquisitions (https://arxiv.org/abs/2409.15132)
- **What's New**: FusionRF는 위성 이미지에서 광학적으로 비가공된 데이터를 사용하여 지형 재구성을 수행하는 새로운 신경 렌더링 방법입니다. 기존의 팬샤프닝(pansharpening) 방법과는 달리 사전 지식이 필요 없이 복잡한 외부 처리 없이 이미지를 융합할 수 있도록 합니다.

- **Technical Details**: FusionRF는 멀티스펙트럴(multispectral) 이미지와 팬크로매틱(panchromatic) 이미지를 직접 최적화하여 고해상도 이미지를 생성합니다. 이를 통해 공간 해상도 손실을 모델링하는 새로운 블러 커널(sparse blur kernel)을 내장하였습니다. 또한, 모달 임베딩(modal embedding)을 도입하여 다양한 이미지 특성을 효과적으로 인코딩하고, 불확실성 학습을 통해 동적 객체와 고정 객체를 구분합니다.

- **Performance Highlights**: 실험 결과, FusionRF는 Depth Reconstruction(깊이 재구성) 및 새로운 보기에서의 선명도에서 기존의 최신 방법(State-of-the-Art)을 초월하는 성능을 보였으며, 멀티스펙트럴 정보를 잘 유지합니다.



### Detect, Describe, Discriminate: Moving Beyond VQA for MLLM Evaluation (https://arxiv.org/abs/2409.15125)
Comments:
          ECCV 2024 Workshop EVAL-FoMo; Project Page: this https URL

- **What's New**: 본 논문은 Multimodal Large Language Models (MLLMs)의 비주얼 개념 이해 능력을 평가하기 위한 새로운 벤치마크인 D3 벤치마크를 소개합니다. D3에서는 매우 유사한 이미지 쌍을 비교하여 집합된 시각적 차이를 정확하게 감지하고 이를 기반으로 자연어로 묘사하도록 요구합니다.

- **Technical Details**: D3 벤치마크는 247개의 고도로 유사한 이미지 쌍으로 구성되며, 각 쌍은 특별한 시각적 차이점(POD, Point of Difference)을 가지고 있습니다. 모델은 해당 차이를 감지하고 타겟 이미지를 독특하게 설명하도록 유도되며, 자가 검색(self-retrieval) 방법을 통해 효과성을 평가합니다.

- **Performance Highlights**: 일반적으로 현재의 MLLM 모델들은 D3 벤치마크에서 39.7%의 성과를 보이며, 이는 무작위 추측보다 낮습니다. 반면 MMVP 벤치마크에서는 동일 개념 네트워크인 Gemini-1.5-Pro가 87.3%의 성과를 기록해 학생들은 D3에서 보다 도전적인 작업으로 인식하고 있음을 나타냅니다.



### Diffusion-based RGB-D Semantic Segmentation with Deformable Attention Transformer (https://arxiv.org/abs/2409.15117)
- **What's New**: 본 연구에서는 RGB-D semantic segmentation 문제를 해결하기 위해 diffusion 기반의 프레임워크를 제안합니다. 또한 Depth 이미지에서 특징을 추출하기 위해 Deformable Attention Transformer를 사용하면 무효 영역의 특성을 효과적으로 포착할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 방법은 RGB-D 이미지를 모형화하는 능력이 뛰어나며, 훈련 시간이 크게 단축된 상태에서 State-of-the-Art 성능을 달성합니다. 실험 결과, NYUv2와 SUN-RGBD 데이터 세트에서 뛰어난 성능을 보였습니다.

- **Performance Highlights**: RGB-D semantic segmentation에서의 성능 향상을 위해 제안된 방법은 전통적인 discriminative 방법과 비교하여 훈련 시간은 짧으면서도 성능은 우수합니다. 실험 결과, 특히 어려운 이미지 데이터에서도 State-of-the-Art 성능을 달성하였습니다.



### The BRAVO Semantic Segmentation Challenge Results in UNCV2024 (https://arxiv.org/abs/2409.15107)
Comments:
          ECCV 2024 proceeding paper of the BRAVO challenge 2024, see this https URL

- **What's New**: BRAVO Challenge는 의미적 분할 모델의 신뢰성을 평가하기 위해 설계되었으며, 실제 왜곡과未知分布(unknown out-of-distribution) 시나리오에서 모델의 성능을 벤치마킹합니다. 대회는 100개 이상의 국제 팀이 참여하여 ML(머신러닝) 모델의 개발에 유용한 통찰을 제공합니다.

- **Technical Details**: BRAVO Challenge는 두 가지 신뢰성 범주를 정의합니다: (1) semantic reliability와 (2) OOD (out-of-distribution) reliability. 각 참가자는 Cityscapes 데이터셋을 사용한 단일 도메인 훈련 또는 여러 데이터셋을 혼합한 다중 도메인 훈련 중 하나를 선택하여 모델을 훈련해야 합니다.

- **Performance Highlights**: 제출된 모델들은 'BRAVO Index'에 의해 순위가 매겨졌으며, 결과적으로 다양한 기본 모델에서 큰 개선을 보여주었습니다. 특히 두 트랙 모두 Vision Foundation Models(VFMs)을 활용하여 전반적으로 안정적인 성능을 나타냈습니다.



### M2OST: Many-to-one Regression for Predicting Spatial Transcriptomics from Digital Pathology Images (https://arxiv.org/abs/2409.15092)
- **What's New**: M2OST는 디지털 병리 이미지의 다중 레벨 데이터 구조를 활용하여 유전자 발현을 정확하게 예측해주는 새로운 회귀 Transformer 모델입니다. 기존의 방법들과 달리 M2OST는 다수의 이미지를 함께 사용하여 단일 출력으로 유전자 발현을 예측합니다.

- **Technical Details**: M2OST는 회귀 Transformer로 다수의 WSI(Whole Slide Images)에서 각각의 이미지 레벨을 통합하여 ST(Spatial Transcriptomics) 맵을 예측합니다. 모델의 구조는 Intra-Level Token Mixing Module, Cross-Level Token Mixing Module 및 Cross-Level Channel Mixing Module을 포함하여 다중 스케일 특징 추출 과정을 효율적으로 분리하고 상호 작용하게 구성되어 있습니다.

- **Performance Highlights**: M2OST는 공개된 세 가지 ST 데이터셋에서 테스트되었으며, 적은 수의 매개 변수와 부동 소수점 연산(FLOPs)으로도 최첨단 성능을 달성하는 것으로 나타났습니다.



### TSCLIP: Robust CLIP Fine-Tuning for Worldwide Cross-Regional Traffic Sign Recognition (https://arxiv.org/abs/2409.15077)
- **What's New**: 본 논문에서 제안한 TSCLIP은 교차 지역(개별 지역) 교통 표지 인식의 성능을 향상시키기 위해 개발된 강력한 미세 조정(fine-tuning) 접근법입니다. TSCLIP은 CLIP(contrastive language-image pre-training) 모델을 기반으로 하여, 데이터 분포의 변화를 효과적으로 처리할 수 있습니다.

- **Technical Details**: TSCLIP은 10개의 다양한 출처에서 데이터를 결합하여 만든 교차 지역 교통 표지 벤치마크 데이터셋을 활용합니다. 또한 교통 표지의 특성에 맞춘 특정 장면 설명 및 규칙을 포함하는 프롬프트 엔지니어링(prompt engineering) 기법을 도입하여 모델 훈련 과정을 최적화합니다. 동적 가중치 앙상블(ADWE)을 사용하여 각 훈련 반복의 결과를 통합하며, 제로샷(zero-shot) CLIP 모델의 일반화 능력을 보존합니다.

- **Performance Highlights**: 본 연구에서 제안한 TSCLIP은 기존 분류 벤치마크 모델을 초월하는 성능을 발휘하였으며, 기존 CLIP 미세 조정 기술에 비해 최첨단(state-of-the-art) 성능을 달성하였습니다. TSCLIP은 교차 지역 교통 표지 인식 작업에 처음으로 사용된 CLIP 모델이라는 점에서 중요한 의의를 갖습니다.



### FisheyeDepth: A Real Scale Self-Supervised Depth Estimation Model for Fisheye Camera (https://arxiv.org/abs/2409.15054)
- **What's New**: FisheyeDepth는 피쉬아이 카메라에 최적화된 자기지도 학습(self-supervised) 깊이 추정 모델로, 이미지 왜곡을 처리하기 위해 특수한 기하학적 프로젝션 방법을 통합했습니다. 이 모델은 실제 스케일 포즈 정보를 사용하여 로봇 시나리오에서의 적용 가능성을 높였습니다.

- **Technical Details**: 모델은 Monodepth2 네트워크를 기반으로 하며, 피쉬아이 카메라 모델을 도입하여 왜곡을 제거하고 안정적인 학습을 지원합니다. 또한, 다중 채널 출력 모듈을 계획하여 다양한 스케일의 특징을 융합해 깊이 추정을 더욱 강화합니다.

- **Performance Highlights**: FisheyeDepth는 공개 데이터셋 및 실제 환경에서 평가를 통해 기존 자기지도 학습 모델에 비해 우수한 성능과 강인성을 보여주었습니다.



### AIM 2024 Sparse Neural Rendering Challenge: Methods and Results (https://arxiv.org/abs/2409.15045)
Comments:
          Part of Advances in Image Manipulation workshop at ECCV 2024

- **What's New**: 이번 논문은 Sparse Neural Rendering에 관한 AIM 2024 워크숍의 도전 과제를 리뷰하며, 대회 설정, 제안된 방법 및 각각의 결과에 중점을 둡니다. 참가자들은 Sparse 이미지 관찰을 기반으로 다양한 장면의 새로운 카메라 시점 합성을 목표로 합니다.

- **Technical Details**: 이 도전 과제는 두 개의 트랙으로 구성되며, 첫 번째 트랙은 3개의 뷰(매우 희소)로, 두 번째 트랙은 9개의 뷰(희소)로 구분됩니다. 참가자들은 Peak Signal-to-Noise Ratio (PSNR)를 통해 실제 이미지에 대한 객관적인 충실도를 최적화해야 하며, 새로운 Sparse Rendering (SpaRe) 데이터셋과 DTU MVS 데이터셋을 사용합니다.

- **Performance Highlights**: 이 대회에서는 5개 팀이 Track 1에 최종 결과를 제출하고, 4개 팀이 Track 2에 최종 결과를 제출했습니다. 제출된 모델들은 다양하며 현재 sparse neural rendering의 최첨단 경계를 넓히고 있습니다.



### AIM 2024 Sparse Neural Rendering Challenge: Dataset and Benchmark (https://arxiv.org/abs/2409.15041)
Comments:
          Part of Advances in Image Manipulation workshop at ECCV 2024. Available at: this https URL

- **What's New**: 이 논문에서는 Sparse Rendering (SpaRe) 데이터셋과 벤치마크를 제안합니다. 이는 희소 뷰 네이블 렌더링의 최신 상태를 평가하고 발전시키기 위해 특별히 설계된 새로운 데이터셋입니다.

- **Technical Details**: SpaRe 데이터셋은 97개의 새로운 장면으로 구성되어 있으며, 각 장면은 최대 64개의 카메라 뷰와 7개의 조명 구성을 가지고 있습니다. 해상도는 1600x1200이며, 3개 또는 9개의 입력 이미지를 사용하는 두 가지 희소 구성으로 제공됩니다.

- **Performance Highlights**: SpaRe는 각 장면에서 고해상도의 정확한 ground-truth 이미지를 제공하며, 연구자들에게 재현 가능한 평가를 위한 강력하고 편리한 툴을 제공합니다. 데이터셋은 AIM 2024 Sparse Neural Rendering Challenge의 핵심 데이터셋으로 사용되어, 여러 선도 알고리즘의 성능을 평가합니다.



### Can CLIP Count Stars? An Empirical Study on Quantity Bias in CLIP (https://arxiv.org/abs/2409.15035)
Comments:
          Short paper. Accepted by the Findings of EMNLP 2024

- **What's New**: CLIP 모델에서 수량 편향(quantity bias)을 조사하여 이미지 생성 작업에서 사용자 의도를 잘 이해하지 못하고 실제 출력과 요구되는 객체 수의 불일치를 보여줌.

- **Technical Details**: 이 연구에서는 텍스트, 이미지 및 교차 모달(cross-modal) 관점에서의 수량 편향을 다루며, 9개의 다양한 CLIP 모델을 평가하고, 수량 관련 명사를 포함하는 수작업 데이터셋을 제작하여 CLIP의 수량 이해력을 평가함. CLIP은 'fewer'와 'more'를 비교하는 데 효과적이지 않으며, 이미지 도메인에서는 서로 다른 원의 수를 가진 이미지 간에도 큰 차이를 구별하지 못함.

- **Performance Highlights**: 실험 결과, CLIP은 텍스트와 이미지의 수량 개념을 효과적으로 이해하지 못하며, 수량 단어 간의 유사성을 잘 구분하지 못하고, 이는 다운스트림(downstream) 작업의 신뢰성을 저하시킴.



### Region Mixup (https://arxiv.org/abs/2409.15028)
Comments:
          Published as a Tiny Paper at ICLR 2024

- **What's New**: 이번 연구는 시각적 인식 작업에서 일반화(generalization)를 개선하기 위한 mixup (Zhang et al., 2018) 데이터 증강 방법의 간단한 확장을 소개합니다. 기존 mixup 방법이 전체 이미지를 혼합하는 것과 달리, 제안된 Region Mixup은 여러 이미지의 특정 영역을 결합하는 데 중점을 둡니다.

- **Technical Details**: Region Mixup의 핵심은 이미지의 지역(region)을 선택하여 새로운 학습 샘플(𝑥~,𝑦~)을 생성하는 것입니다. 각 이미지를 연결된 타일(tile)로 나누고, 이 타일들의 조합을 통해 새로운 이미지를 생성합니다. 이 과정에서 binary mask를 사용하여 섞을 부분을 지정하고, element-wise multiplication을 통해 두 이미지의 특정 영역을 결합합니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100, Tiny ImageNet 데이터셋에 대한 이미지 분류 실험을 통해 Region Mixup의 효과성을 입증했습니다. 실험 결과, Region Mixup 방법이 기존의 Mixup 및 CutMix보다 더 향상된 성능을 보여주었습니다. 이 연구는 Region Mixup이 심층 학습의 귀중한 정규화 도구가 될 가능성이 높음을 시사합니다.



### Cross Branch Feature Fusion Decoder for Consistency Regularization-based Semi-Supervised Change Detection (https://arxiv.org/abs/2409.15021)
Comments:
          5 pages, 4 figures, accepted by ICASSP 2024

- **What's New**: 변화 탐지 분야에서 생성된 새로운 준지도 변화 탐지(Semi-supervised Change Detection, SSCD) 방법은 'Cross Branch Feature Fusion (CBFF)' 디코더를 도입하여 기존의 transformer 기반 접근 방식의 한계를 극복합니다. 이 디코더는 지역 합성곱(convolution)과 전역 transformer의 장점을 결합합니다.

- **Technical Details**: 이 논문에서는 CBFF라는 새로운 디코더를 도입하여 합성곱(convolutional)과 transformer 구조의 강점을 효과적으로 결합하는 SSCD 모델을 제안합니다. 이 모델은 강-약 일관성(strong-to-weak consistency) 전략을 기반으로 구축되었으며, 세 개의 cross-branch feature fusion 모듈과 두 개의 예측 헤드로 구성됩니다. 또한, ResNet50 기반의 difference feature generator를 사용하여 변화 특징을 생성합니다.

- **Performance Highlights**: WHU-CD 및 LEVIR-CD 데이터셋을 사용한 포괄적인 실험 결과, 제안된 방법은 현존하는 일곱 가지 SSCD 방법보다 우수한 성능을 보였습니다. 특히, 5%의 레이블이 지정된 데이터와 95%의 레이블이 없는 데이터를 사용하는 경우, 합성곱 기반 모델이 transformer 기반 모델보다 더 좋은 결과를 나타냈습니다.



### DepthART: Monocular Depth Estimation as Autoregressive Refinement Task (https://arxiv.org/abs/2409.15010)
- **What's New**: 이 논문은 Visual AutoRegressive modeling(시각적 자기회귀 모델링)을 기반으로 한 첫 번째 자기회귀(depth estimation) 깊이 추정 모델인 DepthART를 소개합니다. 이 모델은 기존의 VAR 훈련 절차에서 정적 타겟(static targets) 대신 동적 타겟(dynamic targets)을 사용하여 신뢰성을 높이고, 훈련 중 다중 모드 가이드를 포함합니다.

- **Technical Details**: DepthART는 Depth Autoregressive Refinement Task로 구성된 새로운 훈련 방법을 사용합니다. 본 방법은 실제 목표 점(token maps) 대신 모델의 예측을 입력으로 활용하며, 잔여 최소화(residual minimization)라는 목표로 구성되어 훈련과 추론(inference) 단계 간의 차이를 극복합니다.

- **Performance Highlights**: DepthART로 훈련된 Visual Autoregressive Transformer는 다른 생성적 및 비차별적(baselines) 기본 모델 안에서도 뛰어난 성능을 보이며, unseen benchmark에 대해 우수한 결과를 도출했습니다.



### Generalizing monocular colonoscopy image depth estimation by uncertainty-based global and local fusion network (https://arxiv.org/abs/2409.15006)
- **What's New**: 이번 연구에서는 깊이 추정(depth estimation)을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 CNN(Convolutional Neural Network)과 Transformer의 조합을 통해 지역(local) 및 글로벌(global) 정보의 상호 보완을 극대화하며, 의료 영상에서의 심도 추정을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 CNN을 이용한 지역 특성 추출과 Transformer를 통한 글로벌 정보 캡처로 구성되어 있습니다. 여기에는 불확실성(uncertainty)을 기반으로 한 융합 모듈(fusion module)이 포함되어 있어, CNN과 Transformer의 예측을 효과적으로 결합합니다. 이 네트워크는 시뮬레이션 데이터로 훈련될 수 있으며, 어떠한 파인튜닝(fine-tuning) 없이도 실제 임상 데이터에 직결되는 일반화(generalization)를 이루어냅니다.

- **Performance Highlights**: 다양한 데이터셋에서 검증을 통해 우수한 일반화 능력을 보여주며, 실제 임상 환경에서도 강력한 성능을 입증했습니다. 이는 복잡한 조건 속에서도 endoscopic 영상에서의 심도 맵(depth map)을 신뢰성 있게 추정할 수 있는 기반을 마련하여, 자동 내시경 내비게이션 및 기타 임상 작업, 예를 들어 폴립 탐지 및 분할(segmentation) 등에 기여할 것입니다.



### Sparse-to-Dense LiDAR Point Generation by LiDAR-Camera Fusion for 3D Object Detection (https://arxiv.org/abs/2409.14985)
Comments:
          7 pages

- **What's New**: 본 논문에서는 LiDAR 센서의 한계를 보완하기 위해 LiDAR-Camera Augmentation Network (LCANet)라는 새로운 프레임워크를 제안합니다. 이는 2D 이미지 특징을 활용하여 LiDAR 포인트 클라우드 데이터를 재구성하며, 탐지 정확도를 개선하기 위해 추가 포인트를 생성합니다.

- **Technical Details**: LCANet은 LiDAR 센서와 카메라로부터 수집된 데이터를 융합하여 3D 공간에 이미지 특징을 투사합니다. 이 과정에서 생성된 3D 피처는 의미론적(semantic) 및 공간적(spatial) 정보를 포함합니다. 네트워크는 2D 이미지 특징을 적용하여 부족한 포인트를 보충하며, 이는 자율주행 차량이나 로봇에서 중요한 안전성을 확보하는 데 기여합니다.

- **Performance Highlights**: KITTI 및 Waymo 데이터셋에서의 광범위한 실험 결과, LCANet은 기존 모델에 비해 특히 희소하고 먼 거리의 물체 탐지에서 매우 우수한 성능을 보였습니다. 이는 LiDAR만 사용하는 방법과 비교하여 더 정밀한 포인트 생성과 더 적은 허위 긍정을 달성했습니다.



### SocialCircle+: Learning the Angle-based Conditioned Interaction Representation for Pedestrian Trajectory Prediction (https://arxiv.org/abs/2409.14984)
- **What's New**: 이 연구는 SocialCircle+라는 새로운 방법론을 통해 보행자의 사회적 상호작용과 환경 조건을 기반으로 한 경로 예측을 설명 가능하게 만드는 데 초점을 맞추었습니다.

- **Technical Details**: SocialCircle+는 물속에서 다른 동료들과 환경을 인식하는 해양 동물의 울림(ekolocation)에서 영감을 받아 개발되었습니다. 이 방법론은 사회적 브랜치(social branch)와 조건적 브랜치(conditional branch)를 사용하여 보행자들이 어떻게 각 예측 장면에서 사회적 및 물리적으로 위치해 있는지를 각도 기반 순환 시퀀스(angle-based cyclic sequence) 형태로 설명합니다. 비즈니스(procedural) 적 융합(adaptive fusion)을 통해 최종 상호작용 표현(interaction representation)을 학습합니다.

- **Performance Highlights**: 실험 결과, SocialCircle+는 서로 다른 경로 예측(backbones)에서의 성능 우수성을 입증했습니다. 또한 반사실적 개입(counterfactual interventions)을 통해 상호작용 변수들 간의 인과관계(causalities) 모델링 능력과 조건화(capacity)를 동시에 검증하였습니다.



### Dynamic Integration of Task-Specific Adapters for Class Incremental Learning (https://arxiv.org/abs/2409.14983)
- **What's New**: 본 논문은 Non-Exemplar Class Incremental Learning (NECIL) 문제를 해결하기 위해 Dynamic Integration of task-specific Adapters (DIA)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Task-Specific Adapter Integration (TSAI)와 Patch-Level Model Alignment의 두 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: DIA 프레임워크는 패치 레벨 어댑터 통합 전략을 사용하여 컴포지셔널리티(compositionality)를 향상시키고, 두 가지 특수 메커니즘인 Patch-Level Distillation Loss (PDL)와 Patch-Level Feature Reconstruction (PFR)를 통해 특성 일관성(feature consistency)과 정확한 결정 경계(decision boundary)를 유지합니다.

- **Performance Highlights**: 광범위한 실험 결과, DIA는 NECIL 환경에서 벤치마크 데이터셋에서 눈에 띄는 성능 개선을 보였으며, 계산 복잡도를 최대 90% 감소시키면서도 최첨단(state-of-the-art, SOTA) 성능을 유지합니다.



### A new baseline for edge detection: Make Encoder-Decoder great again (https://arxiv.org/abs/2409.14976)
- **What's New**: 이번 논문에서는 복잡한 훈련 전략 없이도 깊은 학습 기반 엣지 감지기가 인간보다 우수한 성능을 발휘할 수 있도록 Vanilla Encoder-Decoder 기반의 엣지 감지기가 제안되었습니다. 바일라터럴 인코더를 사용하여 위치 특징과 의미론적 특징을 분리하고, 이를 통해 보다 컴팩트한 모델을 구현하였습니다.

- **Technical Details**: 저자는 위치 특징과 의미론적 특징의 추출 과정을 분리하는 바일라터럴 인코더를 설계하였습니다. 의미론적 특징에서 파생된 정제된 위치 특징만을 사용하여 엣지 맵을 생성하며, 이는 원본 위치 특징 및 의미론적 특징과의 불필요한 접촉을 피함으로써 생성된 엣지 맵의 품질을 개선합니다.

- **Performance Highlights**: 제안된 New Baseline for Edge Detection (NBED)은 여러 엣지 감지 벤치마크에서 일관되게 우수한 성능을 나타내며, BSDS500에서 ODS는 0.838로 최신 기술 수준의 성과를 달성하였습니다. 복잡한 훈련 전략 없이도 경쟁력 있는 결과를 얻을 수 있음을 보여줍니다.



### Exploring Fine-grained Retail Product Discrimination with Zero-shot Object Classification Using Vision-Language Models (https://arxiv.org/abs/2409.14963)
Comments:
          Accepted at 2024 IEEE 8th Forum on Research and Technologies for Society and Industry Innovation (RTSI) conference

- **What's New**: 이 논문은 MIMEX 데이터셋을 소개하며, 28개의 다양한 제품 카테고리에 걸쳐 세밀한 제품 분류를 목표로 하는 제로샷(Zero-shot) 객체 분류 방법을 제안합니다. 이 데이터셋은 기존의 데이터셋들과는 달리 소매 환경에서의 실제 소비자 상호작용을 반영하여 제품 이미지를 수집했습니다.

- **Technical Details**: 제안하는 방법은 CLIP과 DINOv2로부터의 임베딩을 결합하고 차원 축소 기법을 활용하여 분류 성능을 향상시키는 앙상블(Ensemble) 접근법을 채택합니다. 또한, 희귀한 라벨 데이터 환경에서 적은 샘플로 비주얼 프로토타입을 이용한 클래스 적응(Class Adaptation) 방법도 도입합니다.

- **Performance Highlights**: 본 연구의 앙상블 모델은 기존 모델 대비 15%의 분류 정확성 향상을 보여주었으며, 특히 다변화된 스마트 소매 환경에서 실제 적용 가능성을 보여주고 있습니다.



### Improving Adversarial Robustness for 3D Point Cloud Recognition at Test-Time through Purified Self-Training (https://arxiv.org/abs/2409.14940)
- **What's New**: 이번 논문에서는 3D 포인트 클라우드 인식이 적대적 공격(adversarial attacks)에 취약하다는 문제를 다루고 있으며, 이를 극복하기 위한 새로운 방법인 Purified Self-Training(PST)을 제안합니다. PST는 테스트 시 데이터 스트림에서 발생하는 공격에 적응하기 위해 모델을 동적으로 업데이트하는 접근 방식을 사용합니다.

- **Technical Details**: 제안된 PST 방법은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, self-training(ST) 프로세스를 통해 레이블 없는 테스트 샘플에 대한 예측을 수행하고, 높은 신뢰성을 가진 예측(위조 레이블)을 사용하여 모델을 재훈련합니다. 둘째, adaptive thresholding과 feature distribution alignment 정규화를 통해 self-training 과정의 강인성을 향상시킵니다. 이는 테스트 데이터가 스트리밍 방식으로 계속해서 제공되는 현실적인 시나리오를 모사합니다.

- **Performance Highlights**: 다양한 적대적 공격에 대한 extensive 결과를 바탕으로 PST 방법이 기존의 정화(purification) 방식과 상호 보완적이며, 지속적으로 변화하는 적대적 공격을 처리하는 데에서 우수한 성능을 보인다는 것을 보여줍니다. 제안된 방법은 현실적인 공격 시나리오에서 모델의 강인성을 크게 향상시킵니다.



### Deep Cost Ray Fusion for Sparse Depth Video Completion (https://arxiv.org/abs/2409.14935)
Comments:
          19 pages, accepted to ECCV 2024

- **What's New**: 본 논문에서는 희소(depth) 깊이 비디오 보완을 위한 학습 기반 프레임워크인 RayFusion을 제안합니다. 이 프레임워크는 여러 관점에서의 깊이 정보를 효과적으로 융합하기 위해 주목(attention) 메커니즘을 활용합니다.

- **Technical Details**: RayFusion은 깊이 가설 평면을 기반으로 구축된 비용 볼륨(cost volume)을 생성하고, 이전 및 현재의 볼륨을 crossover 함으로써 깊이 보완을 수행합니다. 이 방법은 각 ray를 최소 단위로 사용하여 메모리 사용량을 줄이며, self-attention과 cross-attention을 통해 최적화합니다.

- **Performance Highlights**: RayFusion은 KITTI, VOID, ScanNetV2 데이터셋에서 기존 최첨단(depth completion) 방법들보다 우수한 성능을 보여주며, 사용하는 네트워크 파라미터 수가 94.5% 적습니다. 특히 단일 뷰 정보만을 사용하면서도 SOTA 성능을 달성했습니다.



### DanceCamAnimator: Keyframe-Based Controllable 3D Dance Camera Synthesis (https://arxiv.org/abs/2409.14925)
Comments:
          Accepted by ACM Multimedia 2024

- **What's New**: 이번 논문에서는 음악과 춤을 기반으로 한 카메라 움직임 생성의 어려움을 극복하기 위해, 애니메이터의 춤 영화 촬영 지식을 통합하여 DanceCamAnimator라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: DanceCamAnimator는 세 단계의 과정을 통해 카메라 무브먼트를 합성합니다: 키프레임 탐지(Camera Keyframe Detection), 키프레임 합성(Camera Keyframe Synthesis), 트윈 함수 예측(Tween Function Prediction). 이 모델은 각 프레임이 키프레임인지 여부를 판단하고, 키프레임에서의 카메라 매개변수를 추론한 후, 카메라 매개변수의 변화 속도를 제어하기 위해 트윈 함수값을 예측합니다.

- **Performance Highlights**: 상당한 실험을 통해 DCM 데이터셋에서 기존 기법들보다 정량적 및 정성적으로 뛰어난 결과를 보여주었습니다. DanceCamAnimator는 키프레임 수준의 제어력을 제공하며 카메라 무브먼트의 부드러움과 안정성을 개선하여 사용자 경험을 향상시킵니다.



### Advancing Video Quality Assessment for AIGC (https://arxiv.org/abs/2409.14888)
Comments:
          5 pages, 1 figure

- **What's New**: 최근 AI 생성 모델은 텍스트 생성, 이미지 생성, 비디오 생성 등 여러 분야에서 놀라운 발전을 이루었습니다. 본 논문에서는 텍스트-비디오 생성 분야에서 평가 방법이 underveloped 되었음을 지적하며, Frame Consistency Loss (FCL)라는 새로운 손실 함수를 제안합니다. 이를 통해 생성된 비디오 프레임 간의 질적 일관성을 개선할 수 있습니다.

- **Technical Details**: 제안된 FCL은 평균 절대 오차(mean absolute error) 손실과 이진 교차 엔트로피(binary cross-entropy) 손실을 조합하여 inter-frame quality inconsistencies(프레임 간 품질 불일치)를 완화합니다. 또한, S2CNet을 활용한 content-aware cropping 기법을 도입하여 필수 내용을 유지합니다. 이는 adversarial training(적대적 학습)을 적용하여 모델 일반화 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 본 방법이 AIGC Video 데이터셋에서 기존 VQA 기법보다 3.1% 더 나은 PLCC(피어슨 선형 상관 계수)를 달성하여 최신 기술 수준을 초과하는 성능을 입증했습니다.



### Probabilistically Aligned View-unaligned Clustering with Adaptive Template Selection (https://arxiv.org/abs/2409.14882)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문에서는 View-unaligned clustering (VuC) 문제를 해결하기 위해 새로운 접근법인 Probabilistically Aligned View-unaligned Clustering with Adaptive Template Selection (PAVuC-ATS)를 제안합니다. 이 방법은 bipartite graph를 기반으로 하여, 서로 다른 뷰에서의 동일한 객체의 cross-view correspondence (CVC)를 복원하는 데 중점을 둡니다.

- **Technical Details**: PAVuC-ATS는 두 개의 잠재 표현 간의 정렬을 Markov chain의 2단계 전환으로 재설계하여 unaligned 그래프에 적용될 수 있는 permutations를 도출합니다. 이 과정에서 adaptive template selection이 포함되어 probabilistic alignment를 달성합니다.

- **Performance Highlights**: 여섯 개의 벤치마크 데이터셋에서 진행된 광범위한 실험 결과, 제안된 PAVuC-ATS가 기존 baseline 방법들보다 우수한 성능을 보임을 입증하였습니다.



### Mammo-Clustering:A Weakly Supervised Multi-view Global-Local Context Clustering Network for Detection and Classification in Mammography (https://arxiv.org/abs/2409.14876)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구에서는 맥락 클러스터링에 기반한 약한 감독 멀티 뷰 유방촬영 조기 스크리닝 모델을 제안하여 유방암의 조기 발견을 위한 새로운 접근 방식을 제공합니다. 이 모델은 CNN 또는 Transformer에 의존하지 않고 정보를 보완하기 위한 멀티 뷰 학습을 결합하여 사람들에게 유용한 솔루션을 제공합니다.

- **Technical Details**: 본 모델은 약한 감독 학습(w weak supervision) 전략을 적용하여 데이터 제한 문제를 해결하고, 다수의 이미지 시각을 활용하여 특성 변형을 통해 향상된 성능을 이루어냅니다. 기존 전통적 방법과는 달리, 이 모델은 두 개의 공개 데이터 세트에서 각각 AUC 0.828 및 0.805를 기록하며, 이로써 실제 환경에서의 적용 가능성을 높입니다.

- **Performance Highlights**: 이 모델은 기존의 방법론에 비해 더 적은 파라미터로 첨단 성능을 달성하며, 특히 덜 발전된 지역에서의 유방암 스크리닝의 용이성을 높이고 의사의 부담을 줄일 수 있는 잠재력을 보여줍니다.



### FUSED-Net: Enhancing Few-Shot Traffic Sign Detection with Unfrozen Parameters, Pseudo-Support Sets, Embedding Normalization, and Domain Adaptation (https://arxiv.org/abs/2409.14852)
Comments:
          17 pages, 6 figures, 3 tables, submitted to IEEE Access for review

- **What's New**: 이 논문에서는 FUSED-Net이라는 새로운 트래픽 신호 인식 네트워크를 제안합니다. 이 네트워크는 Faster RCNN을 기반으로 하며, Unfrozen Parameters, Pseudo-Support Sets, Embedding Normalization, 그리고 Domain Adaptation을 활용하여 적은 데이터로도 우수한 성능을 발휘할 수 있도록 설계되었습니다.

- **Technical Details**: FUSED-Net의 주요 기술적 특징으로는 모든 파라미터를 학습 가능한 상태로 유지하는 방식(Unfrozen Parameters), 데이터 증강을 통해 생성되는 Pseudo-Support Sets, 인터 클래스 변동성을 줄이는 Embedding Normalization, 그리고 다양한 트래픽 신호 데이터셋을 통한 사전 학습을 이용한 Domain Adaptation이 있습니다. 이러한 요소들은 FUSED-Net이 제한된 샘플에서 더욱 효율적으로 학습하도록 합니다.

- **Performance Highlights**: BDTSD 데이터셋에서 FUSED-Net은 기존의 최첨단 Few-Shot Object Detection(FSOD) 모델 대비 1-shot, 3-shot, 5-shot, 10-shot 시나리오에서 각각 2.4배, 2.2배, 1.5배, 1.3배 향상된 mAP(mean Average Precision)를 달성하였습니다. 이 결과는 서로 다른 도메인에서의 성능 평가에서도 우수함을 보여줍니다.



### Disentanglement with Factor Quantized Variational Autoencoders (https://arxiv.org/abs/2409.14851)
Comments:
          Preprint submitted to Pattern Recognition

- **What's New**: 이번 연구에서는 FactorQVAE라는 새로운 모델을 제안하여, 단일의 전역 코드북(global codebook)과 추가적인 유도 편향(inductive bias)을 사용하여 disentanglement 성능을 향상시키고자 하였다.

- **Technical Details**: FactorQVAE는 discrete variational autoencoder (VAE) 기반의 모델로, 스칼라 값의 전역 코드북을 이용하여 잠재 변수를 양자화하고, 최적화 과정에서 total correlation 항을 추가하는 방식으로 작동한다. 이 모델은 단일 코드북을 사용하여 각 생성적 요인에 고유한 파라미터의 할당을 가능하게 한다.

- **Performance Highlights**: FactorQVAE는 기존의 분리(disentanglement) 방법들과 비교하여 DCI와 InfoMEC 두 가지 지표에서 우수한 성능을 보였으며, 모델의 재구성 성능 또한 개선되었다.



### GroCo: Ground Constraint for Metric Self-Supervised Monocular Depth (https://arxiv.org/abs/2409.14850)
- **What's New**: 본 논문에서는 자가 감독 학습(self-supervised learning) 환경에서의 깊이 추정을 개선하기 위해 새롭고 혁신적인 접근법을 제안합니다. 특히, we propose a new constraint on ground areas를 통해 깊이 예측의 정확성과 일반화 능력을 크게 향상시킵니다.

- **Technical Details**: 우리는 ground plane prior를 통합하기 위한 새로운 손실 함수(loss function)를 도입하여, 깊이 추정 모델이 다양한 카메라 구성을 고려할 수 있는 힘을 부여합니다. 이 접근법은 깊이 예측과 ground prior 간의 일관성을 보장하여, 보다 정확한 스케일 복구(scale recovery)를 가능하게 합니다.

- **Performance Highlights**: 우리의 실험 결과는 KITTI 벤치마크에서 기존의 스케일 복구 기법들보다 뛰어난 성능을 보여주며, 다양한 카메라 회전(camera rotations)과 이전에 보지 못한 운전 데이터셋에 대한 제로샷(zero-shot) 조건에서도 모델의 일반화 능력을 크게 향상시키는 것으로 나타났습니다.



### Revisiting Video Quality Assessment from the Perspective of Generalization (https://arxiv.org/abs/2409.14847)
Comments:
          13 pages, 4 figures

- **What's New**: 최근 YouTube Shorts, TikTok, Kwai와 같은 짧은 비디오 플랫폼의 인기로 사용자 생성 콘텐츠(User-Generated Content, UGC)가 급증하면서 비디오 품질 평가(Video Quality Assessment, VQA) 작업의 일반화 성능에 상당한 도전 과제가 발생하고 있습니다. 본 연구에서는 VQA 작업을 일반화 관점에서 재조명하며, weight loss landscape와 일반화 성능 간의 상관관계를 확인하고, weight loss landscape를 정규화하는 다양한 기법을 탐색합니다.

- **Technical Details**: VQA 모델의 weight loss landscape 분석을 통해, 이는 일반화 격차(generalization gap)와 강한 상관관계를 보임을 확인했습니다. 우리는 adversarial weight perturbations를 통해 loss landscape를 부드럽게 하여 VQA 모델의 일반화 성능 향상을 달성했습니다. 실험 결과, cross-dataset generalization에서 최대 1.8%, fine-tuning 성능에서 최대 3% 향상되었습니다.

- **Performance Highlights**: 여러 VQA 방법 및 데이터셋에 대한 광범위한 실험을 통해, adversarial weight perturbations가 VQA 모델의 일반화 성능을 크게 향상시킨다는 것을 확인하였습니다. 또한, 이미지 품질 평가(Image Quality Assessment, IQA) 작업에서도 최첨단 성능을 달성했습니다.



### Two Deep Learning Solutions for Automatic Blurring of Faces in Videos (https://arxiv.org/abs/2409.14828)
- **What's New**: 이 논문은 공공장소에서 개인의 얼굴을 모자이크 처리하는 두 가지 깊이 학습 방법을 제안합니다. 첫 번째는 YOLO(You Only Look Once) 모델을 이용한 직접 접근 방식이며, 두 번째는 Unet에 기반한 세그멘테이션 네트워크를 활용한 간접 접근 방식입니다.

- **Technical Details**: 첫 번째 접근법에서는 YOLOv5Face 모델을 사용하여 얼굴을 탐지한 후, 탐지된 얼굴에 블러를 적용합니다. YOLOv5Face는 빠르고 정확한 얼굴 탐지를 위해 설계된 컨볼루셔널 신경망(CNN)입니다. 두 번째 접근법에서는 DeOldify와 같은 Unet 아키텍처를 통해 이미지에서 직접 얼굴을 블러링하는 방법을 탐구합니다. 두 접근 방식은 각각의 방법론을 이용하여 실험적으로 비교되었습니다.

- **Performance Highlights**: 실험 결과, YOLOv5Face 모델을 사용한 블러링의 정밀도와 속도가 높은 반면, Unet 아키텍처는 세그멘테이션을 통해 빠르고 효과적인 블러링 결과를 제공했습니다. 두 방법 모두 같은 데이터를 기반으로 실험하였으며, 최종적으로 각 방법의 효과를 종합하여 비교했습니다.



### AIM 2024 Challenge on Video Saliency Prediction: Methods and Results (https://arxiv.org/abs/2409.14827)
Comments:
          ECCVW 2024

- **What's New**: 이번 논문은 AIM 2024에서의 Video Saliency Prediction Challenge에 대해 리뷰합니다. 참가자들은 제공된 비디오 시퀀스에 대한 정확한 saliency maps를 예측하는 방법을 개발하는 것을 목표로 했습니다. 새로운 대규모 오디오-비주얼 마우스 saliency (AViMoS) 데이터셋이 수집되어, 1500개의 비디오에 대해 70명 이상의 관찰자가 참여했습니다.

- **Technical Details**: 이번 챌린지를 위해, 246개의 비디오는 YouTube-UGC 데이터셋에서, 1254개의 비디오는 www.vimeo.com에서 크롤링된 고비트레이트 오픈 소스 비디오를 통해 수집되었습니다. 데이터 수집 방법론은 기존 연구에 기반하여 crowdsourcing을 사용해 saliency 데이터를 모았습니다. 눈 추적 대신 마우스 추적을 통해 수집된 데이터는 높은 일치를 보였습니다.

- **Performance Highlights**: 30개 이상의 팀이 참가하였으며, 최종 단계에서는 7개 팀이 결과를 제출했습니다. 평가 결과는 사적 테스트 하위 집합의 일반적인 품질 지표로 테스트 및 순위화되었습니다. 최종 평가의 AUC-Judd, CC, SIM 등의 메트릭에서 상당히 높은 성과를 달성하였으며, 자동화된 방법론보다 높은 결과를 보였습니다.



### Advancing Depression Detection on Social Media Platforms Through Fine-Tuned Large Language Models (https://arxiv.org/abs/2409.14794)
Comments:
          16 pages

- **What's New**: 본 연구는 사용자 소셜 미디어 데이터를 기반으로 우울증 탐지를 개선하기 위해 Fine-tuned Large Language Models (LLMs)를 사용하는 방법을 조사합니다. GPT 3.5 Turbo 1106과 LLaMA2-7B 모델을 활용하여 상당한 정확도(96.0%)로 우울한 콘텐츠를 식별했습니다.

- **Technical Details**: 이 연구는 Fine-tuned LLMs의 활용과 관련하여 세부적인 접근법을 설명하며, 사용된 매개변수와 Fine-tuning 절차를 다룹니다. 구체적으로, GPT-3.5 Turbo 1106 모델과 LLaMA2-7B 모델을 Fine-tuning하여 우울증 데이터셋을 통한 탐지 성능을 강화하였습니다.

- **Performance Highlights**: Fine-tuned LLM을 사용함으로써 기존의 최첨단 시스템들에 비해 개선된 성능을 달성했으며, 테스트 데이터에서 96.0%의 정확도를 기록했습니다. 이는 LLM 기반의 정교한 시스템이 우울증 탐지에 potential한 도구가 될 수 있음을 나타냅니다.



### Human Hair Reconstruction with Strand-Aligned 3D Gaussians (https://arxiv.org/abs/2409.14778)
- **What's New**: 이 논문에서는 다중 뷰 데이터에서 정확하고 사실적인 strand 기반 헤어 복원을 생성하는 새로운 헤어 모델링 방법인 Gaussian Haircut을 소개합니다. 이 방법은 기존의 비정형 Gaussian을 사용하는 접근 방식과는 달리, 3D polylines(폴리라인)을 사용하여 헤어를 복원합니다.

- **Technical Details**: Gaussian Haircut은 3D Gaussians(3D 가우시안)와 클래식 헤어 strand(헤어 스트랜드)의 이중 표현을 활용합니다. 이 접근 방식은 헤어의 내부 구조를 사실적으로 모델링하기 위해 strand 기반의 헤어 priors(프라이어)를 결합하여 사용합니다. 이 모델은 다중 뷰 이미지를 입력으로 받아 현실적인 3D strands를 출력합니다.

- **Performance Highlights**: 이 방법은 합성 및 실제 장면에서 평가되어 strand 기반 헤어 복원 작업에서 최첨단 성능을 보여줍니다. 또한, 경량 최적화 방법을 통해 복원 속도와 품질 모두 향상되었습니다.



### CFVNet: An End-to-End Cancelable Finger Vein Network for Recognition (https://arxiv.org/abs/2409.14774)
- **What's New**: 이 논문은 손 정맥 인식 시스템을 위한 일체형의 보안 솔루션을 제안합니다. 기존의 수많은 시스템들과 달리, 사전 처리 및 템플릿 보호 과정을 통합한 딥 러닝 모델을 통해 안정적이고 안전한 손 정맥 인식 기술을 구현하였습니다.

- **Technical Details**: 제안된 CFVNet는 BWR-ROIAlign 모듈을 포함하며, 이는 네 가지 구성 요소로 나뉩니다: (1) Localization - 손 정맥의 특성 지역을 자동으로 찾아냅니다. (2) Compression - 공간적 불필요 정보를 손실 없이 제거합니다. (3) Transformation - BWR 방법을 통해 비가역성, 연계 불가능성 및 회수 가능성을 도입합니다. 이 모듈은 DCNN 기반의 인식 시스템에 직접 연결하여 이러한 특성을 추가합니다.

- **Performance Highlights**: 제안된 시스템은 네 개의 공개 데이터셋에서 평균 정확도 99.82%, EER 0.01%, Dsys 0.025를 기록하며, 최신 기술과 매우 경쟁력 있는 성능을 보여줍니다.



### Robust and Flexible Omnidirectional Depth Estimation with Multiple 360{\deg} Cameras (https://arxiv.org/abs/2409.14766)
- **What's New**: 본 논문에서는 카메라 오염과 다양한 카메라 배치의 문제를 해결하기 위해 다수의 360도 카메라로부터 기하학적 제약과 중복 정보를 활용하여 강건하고 유연한 다중 시점의 전방위 깊이 추정을 수행하는 두 가지 알고리즘을 제안합니다.

- **Technical Details**: 제안된 방법 중 첫 번째는 Pairwise Stereo MODE (PSMODE)로, 두 단계 접근 방식을 사용하여 다중 시점의 전방위 깊이 맵을 추정합니다. 첫 번째 단계에서는 여러 카메라 쌍을 선택하여 쌍별 스테레오 매칭을 통해 깊이 맵을 얻고, 두 번째 단계에서 이 깊이 맵을 융합하여 최종 깊이를 추정합니다. 두 번째 방법인 SSMODE는 가상의 깊이를 기반으로 구면 스위핑(spherical sweeping)을 활용하여 다중 카메라 이미지의 일관된 구면 매칭 비용을 구축합니다. 또한, Generalized Epipolar Equirectangular (GEER) 투영을 도입하여 구면 에피폴라 제약을 단순화합니다.

- **Performance Highlights**: 실험 결과, 두 가지 알고리즘 모두 여러 장면에서 신뢰할 수 있는 깊이 맵을 생성하는 데 성공했으며, 특히 오염된 파노라마 입력에 대해서도 최첨단 성능을 달성했습니다. 또한, 카메라 설정과 카메라 수에 따라 실험적으로 알고리즘의 유연성을 입증했습니다.



### VLM's Eye Examination: Instruct and Inspect Visual Competency of Vision Language Models (https://arxiv.org/abs/2409.14759)
- **What's New**: 본 논문은 VLM(Visual Language Models)의 시각 인식 능력을 측정하기 위한 eye examination 프로세스를 제안합니다. 기존의 VLM들이 다양한 벤치마크에서 향상된 성능을 보였으나, 이들이 이미지를 인식하는 방식에 대한 이해는 부족했습니다. 이를 위해 LENS라는 데이터셋을 도입하여 VLM의 시각 인식 절차에 따라 준비 상태를 점검하고 검사합니다.

- **Technical Details**: LENS 데이터셋은 색상, 형태, 의미론적 요소의 세 가지 기본 요소로 구성되어 있으며, 각 요소에 대한 질문과 진단 과제를 포함합니다. 모델은 해당 요소의 ready check를 거쳐 시험을 진행하며, 색상 민감도(Sensitivity Area of Color, SAC) 및 형태 민감도(Sensitivity Area of Shape, SAS) 등의 지표를 통해 평가됩니다. 검사 과정에서는 세 가지 시험이 포함되며, 각 단계에서 VLM의 성능이 점검됩니다.

- **Performance Highlights**: 검사 결과, VLM들은 서로 다른 색깔에 대해 다르게 반응하고, 특히 모든 VLM에서 초록색에 대한 민감도가 낮음을 확인했습니다. 형태에 대해서도 다양한 VLM 모델의 능력에 따라 차이가 나타났습니다. 이러한 결과는 시각 인식 성능 향상을 위한 VLM의 설계 및 입력 전처리에 도움을 줄 수 있습니다.



### BranchPoseNet: Characterizing tree branching with a deep learning-based pose estimation approach (https://arxiv.org/abs/2409.14755)
- **What's New**: 이 논문은 태형(whorl) 탐지를 위한 자동화된 파이프라인을 제시합니다. 이 시스템은 포즈 추정(pose-estimation) 심층 학습(deep learning) 모델을 사용하여 레이저 스캐닝 데이터에서 나무의 태형을 감지합니다.

- **Technical Details**: 점 구름 데이터(point cloud data)를 처리하여 단면 이미지를 생성하며, 이후 나무의 태형과 줄기(stem)를 따라 있는 가지(branches)를 나타내는 키포인트(keypoints)를 식별하는 데 사용됩니다. 방법은 파괴적으로 샘플링된 개별 나무의 데이터셋에서 테스트되었습니다.

- **Performance Highlights**: 결과는 나무 태형의 정확한 식별과 주요 구조적 메트릭(metrics)의 정밀 계산을 통해 강력한 가능성을 보여주며, 개별 나무의 점 구름에서 새로운 통찰(insight)과 깊은 수준의 정보를 제공합니다.



### UniBEVFusion: Unified Radar-Vision BEVFusion for 3D Object Detection (https://arxiv.org/abs/2409.14751)
Comments:
          6 pages, 4 figues, conference

- **What's New**: 본 논문에서는 4D 밀리미터-파(mmWave) 레이더와 비전 데이터를 통합한 Radar Depth Lift-Splat-Shoot (RDL) 모듈을 제안하여 깊이 예측 과정에 레이더 특유의 데이터를 포함시키고 시각적 Bird-Eye View (BEV) 특징의 품질을 개선했습니다. 또한 다양한 모드에서 BEV 특징을 추출하는 Unified Feature Fusion (UFF) 접근법을 소개했습니다.

- **Technical Details**: RDL 모듈은 Radar Cross-Section (RCS) 데이터를 깊이 예측 모듈에 추가하여 레이더 특유의 정보를 최대한 활용합니다. 이 연구는 시각적 모드 실패를 시뮬레이션하기 위해 Gaussian noise를 주입하는 새로운 Failure Test (FT) 실험을 개발하였으며, 이를 통해 다중 모달 모델의 강인성을 평가했습니다. 실험 데이터로는 View-of-Delft (VoD)와 TJ4D 데이터셋을 사용했습니다.

- **Performance Highlights**: 제안된 UniBEVFusion 네트워크는 TJ4D 데이터셋에서 기존의 최첨단 모델보다 3D 객체 탐지 정확도에서 1.44의 개선과 BEV 객체 탐지 정확도에서 1.72의 개선을 보였습니다. 이 결과는 레이더-비전 융합 모델의 성능 향상을 위한 새로운 접근법의 가능성을 보여줍니다.



### FineCops-Ref: A new Dataset and Task for Fine-Grained Compositional Referring Expression Comprehension (https://arxiv.org/abs/2409.14750)
Comments:
          19 pages, EMNLP 2024

- **What's New**: FineCops-Ref라는 새로운 REC 데이터셋을 제안했습니다. 이 데이터셋은 접근 가능한 난이도를 제어할 수 있으며, 기존 데이터셋에서는 간과되었던 부정적 샘플에 대한 모델의 저항력을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: FineCops-Ref 데이터셋은 객체 카테고리, 속성 및 다단계 관계에 대한 세밀한 추론을 요구합니다. 난이도 수준은 대상 객체 위치 파악에 필요한 속성과 관계의 수에 따라 분류됩니다. 샘플에는 부정적인 텍스트와 이미지가 포함되어 있어, 모델의 시각적 기초 능력을 직접 평가할 수 있습니다.

- **Performance Highlights**: 상태 최상 모델들 및 MLLMs을 종합적으로 평가한 결과, grounding 성능의 상당한 차이를 발견했습니다. 간단한 REC 작업에서는 전통적 비전-언어 모델이 우수한 성능을 보였고, 더 높은 난이도에서는 MLLMs가 더 나은 성과를 나타냈습니다. 이는 모델의 미세 조정을 통해 성능이 향상되었음을 보여줍니다.



### Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting (https://arxiv.org/abs/2409.14747)
Comments:
          10 pages, 6 figures

- **What's New**: 본 연구에서는 'correlation collapse'라는 새로운 문제를 제기하고, 이를 해결하기 위한 새로운 방법인 Distribution-Level Feature Distancing (DLFD)을 제안합니다. 이 방법을 통해 특정 이미지를 효과적으로 잊으면서도 모델의 전반적인 성능을 유지할 수 있습니다.

- **Technical Details**: DLFD는 최적 운송 문제(Optimal Transport)를 활용하여, 잊어야 할 이미지 분포로부터 보존해야 할 이미지의 특징 분포를 이동시킵니다. 이 과정에서 생성된 데이터는 잊어야 할 데이터와 다른 분포를 가지도록 하여, 효율적인 이미지 잊기를 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해, 우리의 방법이 기존의 최신 기계 유학 방법들보다 우수하다는 것을 입증하였습니다. DLFD는 특히 얼굴 인식 데이터셋에서 뛰어난 성과를 보여주었습니다.



### Less yet robust: crucial region selection for scene recognition (https://arxiv.org/abs/2409.14741)
- **What's New**: 이 논문은 저품질 이미지에서 효과적인 장면 인식을 위한 새로운 적응형 선택 메커니즘을 제안합니다. 이를 통해 고수준의 의미 있는 특징을 가진 결정을 더욱 강조하여 성능 향상을 도모합니다.

- **Technical Details**: 제안된 방법은 CNN(Convolutional Neural Network)을 기반으로 하여, 중요한 고수준의 의미 특징이 있는 지역을 식별합니다. 학습 가능한 마스크를 네트워크에 구현하여 특징 행렬의 서로 다른 지역에 가중치를 부여하며, 중요 정규화 항을 추가하여 핵심 고수준 특징 지역의 중요성을 증가시킵니다. 이는 명확한 결정에 도움이 되는 저품질 이미지를 효과적으로 처리하는 방법입니다.

- **Performance Highlights**: 우리는 수중 지질 장면 분류 데이터셋을 구성하고, 제안된 방법이 최신 기술에 비해 우수성과 강건성을 입증하는 다양한 실험 결과를 보여줍니다.



### EDSNet: Efficient-DSNet for Video Summarization (https://arxiv.org/abs/2409.14724)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문은 비디오 요약(video summarization) 방법의 비효율성을 해결하기 위해 Direct-to-Summarize Network (DSNet)를 개선하여 더 자원 효율적인 토큰 혼합(token mixing) 메커니즘을 도입했습니다. 전통적인 어텐션(attention) 기법을 Fourier 변환(Fourier), 웨이브렛 변환(Wavelet transforms), Nyströmformer로 대체함으로써 효율성과 성능을 향상시켰습니다. 또한 ROI 풀링(ROI pooling), Fast Fourier Transform 풀링, 평면 풀링(flat pooling)과 같은 다양한 풀링(pooling) 전략을 탐구했습니다.

- **Technical Details**: 본 연구에서는 비디오 프레임에서 주요 세그먼트를 식별하고 요약을 수행하기 위해 Temporal Region Proposal Network을 사용하며, DSNet 아키텍처를 수정하여 특징 추출(feature extraction) 및 영역 제안(region proposal) 네트워크의 효율성을 향상시킵니다. 각 프레임은 관심 제안으로 분류되며, Nyströmformer와 FNet 블록을 사용하여 자원 소모를 줄이고 대량의 비디오 데이터를 효율적으로 처리할 수 있습니다. 본 시스템은 나중에 정밀한 요약 생성을 위해 회귀(regression) 모델을 추가로 사용합니다.

- **Performance Highlights**: TVSum과 SumMe 데이터셋에 대한 실험 결과, 이 수정사항들은 계산 비용을 상당히 줄이면서도 경쟁력 있는 요약 성능을 유지함을 보여줍니다. 이는 특히 대용량 비디오 데이터 처리에 더욱 확장 가능한 솔루션을 제공하는 것으로 평가됩니다.



### ControlEdit: A MultiModal Local Clothing Image Editing Method (https://arxiv.org/abs/2409.14720)
- **What's New**: 이 논문에서 우리는 ControlEdit이라는 새로운 의류 이미지 편집 방법을 제안했습니다. 이 방법은 의류 이미지 편집을 멀티모달(예: 텍스트 설명 및 시각적 이미지) 기반의 로컬 인페인팅(local inpainting)으로 전환하여 디자이너의 작업 효율성을 높이고 사용자 디자인의 접근성을 낮춰줍니다.

- **Technical Details**: ControlEdit는 자가 지도 학습(self-supervised learning) 접근 방식을 활용하여 현실 이미지 데이터셋 수집의 어려움을 극복하며, 특징 추출 네트워크의 채널을 확장하고 역(latent) 손실 함수(inverse latent loss function)를 설계하여 편집되지 않은 영역의 콘텐츠에 대한 부드러운 제어를 구현합니다. 또한 Blended Latent Diffusion를 채택하여 편집 경계가 자연스럽게 전환되도록 하여 미편집 영역 콘텐츠의 일관성을 강화합니다.

- **Performance Highlights**: 광범위한 실험 결과 ControlEdit가 정성적 및 정량적 평가에서 기존의 기준 알고리즘을 초월함을 보였습니다.



### Phantom of Latent for Large Language and Vision Models (https://arxiv.org/abs/2409.14713)
Comments:
          Code is available in this https URL

- **What's New**: 이번 논문에서는 0.5B, 1.8B, 3.8B, 7B 파라미터를 가진 새로운 효율적인 LLVM(large language and vision models) 패밀리인 Phantom을 소개합니다. 이 모델은 작지만 성능은 더 큰 모델들과 유사한 수준을 유지합니다.

- **Technical Details**: Phantom은 멀티-헤드 자기-주의(MHSA) 중에 잠재 숨겨진 차원을 일시적으로 증가시켜 비전-언어(vision-language) 지식을 더 잘 이해할 수 있도록 합니다. 이를 통해 물리적인 모델 크기를 크게 증가시키지 않으면서도 학습 능력을 강화합니다. 또한, Phantom 최적화(Phantom Optimization, PO)를 도입하여 자가 회귀(supervised fine-tuning, SFT)와 직접적 선호 최적화(direct preference optimization, DPO) 개념을 결합하여 정확한 답변을 효과적으로 따릅니다.

- **Performance Highlights**: Phantom은 여러 대형 오픈 및 클로즈드 소스 LLVM 모델들보다 뛰어난 성능을 보여줍니다. 이는 효율적인 LLVM의 선도적인 솔루션으로 자리매김하게 만듭니다.



### VLEU: a Method for Automatic Evaluation for Generalizability of Text-to-Image Models (https://arxiv.org/abs/2409.14704)
Comments:
          accepted by EMNLP2024(long paper,main conference)

- **What's New**: 텍스트-이미지(T2I) 모델의 평가 방법을 개선하기 위해 새로운 평가 지표인 VLEU(Visual Language Evaluation Understudy)를 소개합니다. 이 지표는 대규모 언어 모델을 사용하여 T2I 모델의 다양한 텍스트 프롬프트에 대한 일반화 능력을 정량적으로 평가할 수 있습니다.

- **Technical Details**: VLEU는 시각적 텍스트 도메인의 분포와 T2I 모델이 생성한 이미지의 조건부 분포 간의 Kullback-Leibler divergence를 계산하여 모델의 일반화 능력을 수치적으로 평가합니다. 이 지표는 다양한 텍스트 프롬프트에서 이미지의 생성 품질과 그 일치도를 평가하는 데 활용됩니다. LLM(대규모 언어 모델)과 CLIP 모델을 사용하여 텍스트와 이미지 간의 의미적 일치를 평가합니다.

- **Performance Highlights**: VLEU의 실험을 통해 다양한 T2I 모델의 일반화 능력을 효과적으로 평가할 수 있음을 입증하였습니다. 이 새로운 지표는 T2I 모델 개발에 필수적인 도구로 자리잡을 것으로 기대되며, 실제 사례 연구 또한 발표되어 그 유용성을 보여주었습니다.



### Dynamic Realms: 4D Content Analysis, Recovery and Generation with Geometric, Topological and Physical Priors (https://arxiv.org/abs/2409.14692)
Comments:
          Research Summary - DC

- **What's New**: 이 연구는 4D 콘텐츠의 분석, 복구 및 생성을 다루며, 여기서 4D는 세 가지 공간 차원(x, y, z)과 시간 차원(t)을 포함합니다. 연구는 정적인 객체뿐만 아니라 시간에 따른 동적인 변화까지 포괄합니다.

- **Technical Details**: 연구는 기하학적(geometric), 위상적(topological), 물리적(physical) 선행 정보(priors)를 통합하여 4D 콘텐츠 생성을 더 효율적이고 접근 가능하며 고품질로 만드는 것을 목표로 하고 있습니다. 또한, 이러한 선행 정보를 활용하여 4D 콘텐츠의 복구 및 분석을 위한 효과적인 방법을 개발하는 데 집중합니다.

- **Performance Highlights**: 이 기술은 AR/VR, 체화된 AI(embodied AI), 로보틱스와 같은 다양한 응용 프로그램에서 중요하며, 4D 콘텐츠의 효율적인 생성 및 품질 향상에 기여할 것으로 기대됩니다.



### Quantifying Context Bias in Domain Adaptation for Object Detection (https://arxiv.org/abs/2409.14679)
Comments:
          Under review

- **What's New**: 본 연구는 객체 탐지에서의 도메인 적응(Domain Adaptation for Object Detection, DAOD)에서 컨텍스트 바이어스(context bias)를 분석하고, 이를 다양한 도메인에서 어떻게 활용할 수 있는지를 제안합니다. 특히 배경 특징의 변화가 적응 과정에서 미치는 영향을 분석하여, DAOD 접근법을 개선하기 위한 새로운 아이디어를 도출합니다.

- **Technical Details**: 저자들은 CARLA 데이터셋과 Cityscapes 데이터셋을 사용하여 배경 마스킹 및 모델의 다양한 층에서의 활성화 값을 변화시키는 실험을 통해 컨텍스트 바이어스를 정량화합니다. 다양한 계량지표인 최대 평균 불일치(Maximum Mean Discrepancy, MMD)와 최대 분산 불일치(Maximum Variance Discrepancy, MVD)를 사용하여, 서로 다른 도메인에서 조작된 배경 영역에 대하여 전경의 조건부 확률 추정을 시행합니다.

- **Performance Highlights**:  저자들은 YOLOv4 및 YOLOv8 모델을 사용한 실험을 통해, 배경의 변화가 차량 탐지에 미치는 영향을 분석하였으며, 배경과 전경의 강한 연관성을 드러내어, 다른 배경에서 차량을 탐지하는 데 필요한 시각적 정보가 결여되었음을 보였습니다. 이러한 결과는 DAOD 접근법에서 컨텍스트 바이어스를 고려하는 것이 모델의 일반화 및 견고성을 향상시키는데 필수적임을 강조합니다.



### Reflecting Reality: Enabling Diffusion Models to Produce Faithful Mirror Reflections (https://arxiv.org/abs/2409.14677)
Comments:
          Project Page: this https URL

- **What's New**: 본 논문은 분산 기반 생성 모델을 활용하여 매우 현실적이고 신뢰할 수 있는 거울 반사를 생성하는 문제를 해결합니다. 이를 위해 Image Inpainting(이미지 인페인팅) 태스크로 문제를 정의하고, 'SynMirror'라는 대규모 데이터셋을 제작했습니다. 이 데이터셋은 다양한 합성 장면을 포함하고 있으며, 'MirrorFusion'이라는 새로운 깊이 조건 인페인팅 방법을 제안합니다.

- **Technical Details**: 'SynMirror'는 약 198K개의 샘플과 66K개의 고유 3D 객체로 구성된 대규모 데이터셋으로, 깊이는 맵, 노말 맵 및 인스턴스 세분화 마스크를 포함하여 장면의 기하학적 특성을 캡처합니다. 'MirrorFusion'은 입력 이미지와 거울 영역을 나타내는 마스크를 사용하여 고품질의 기하학적으로 일관된 포토리얼리스틱 반사를 생성하는 깊이 조건 인페인팅 방법입니다.

- **Performance Highlights**: 'MirrorFusion'은 'SynMirror'에서 최신의 방법들과 비교하여 탁월한 성능을 보이며, 정량적 및 정성적 분석을 통해 그 효과가 입증되었습니다. 본 연구는 분산 모델을 사용하여 제어 가능하고 신뢰할 수 있는 거울 반사를 생성하는 문제를 최초로 해결하였습니다.



### AEANet: Affinity Enhanced Attentional Networks for Arbitrary Style Transfer (https://arxiv.org/abs/2409.14652)
Comments:
          10 pages, 5 figures,1 table

- **What's New**: 이 연구에서는 기존 스타일 전이 방식의 한계를 극복하기 위해 콘텐츠 이미지와 스타일 이미지 간의 정보 유사성을 극대화하는 접근 방식인 &#39;affinity-enhanced attentional network (AEANet)&#39;을 제안합니다. 이 네트워크는 콘텐츠와 스타일 정보를 보다 세밀하게 조정하는 세 가지 모듈(CAEA, SAEA, HA)을 포함하고 있습니다.

- **Technical Details**: AEANet은 콘텐츠 친화성 강화 주의 모듈(CAEA), 스타일 친화성 강화 주의 모듈(SAEA), 하이브리드 주의 모듈(HA)을 포함합니다. CAEA와 SAEA 모듈은 주의 메커니즘을 사용하여 콘텐츠와 스타일 표현을 강화하고, 세부 강화 모듈(DE)을 통해 세부 특징을 강화합니다. 또한, 지역 간 비유사성 손실(local dissimilarity loss)을 도입하여 콘텐츠 이미지와 스타일 이미지 간의 연관성을 보다 효과적으로 유지합니다.

- **Performance Highlights**: 실험 결과, AEANet은 기존의 최첨단(style transfer) 방식보다 더 우수한 결과를 보여주었으며, 감정적인 스타일과 더불어 합리적인 세부 사항을 효과적으로 유지함을 입증했습니다.



### EQ-CBM: A Probabilistic Concept Bottleneck with Energy-based Models and Quantized Vectors (https://arxiv.org/abs/2409.14630)
Comments:
          Accepted by ACCV 2024

- **What's New**: 최신 논문에서는 해석 가능한 AI 시스템에 대한 수요가 급증하면서, 인간이 이해할 수 있는 개념을 활용하여 해석 가능성을 향상시키는 개념 병목 모델(Concept Bottleneck Models, CBMs)에 대한 새로운 접근법인 EQ-CBM을 제안합니다.

- **Technical Details**: EQ-CBM은 확률적 개념 인코딩을 개선하기 위해 에너지 기반 모델(Energy-based Models, EBMs)과 양자화된 개념 활성화 벡터(Quantized Concept Activation Vectors, qCAVs)를 활용합니다. 이 방법은 확률적 개념을 캡처하여 예측의 신뢰성과 정확성을 향상시킵니다. 특히, EQ-CBM은 동질적인 벡터를 선택하여 개념을 인코딩함으로써 인간의 개입을 더 수월하게 만들어 주며, 다양한 이미지를 통해 작업 성능을 높입니다.

- **Performance Highlights**: 다양한 벤치마크 데이터세트를 사용한 실험 결과, EQ-CBM은 기존의 CBM 접근법보다 개념 및 작업 정확도에서 우수한 성능을 보여줍니다. 이를 통해 EQ-CBM은 EBMs와 qCAVs를 통한 확률적 접근법이 해석 가능성과 정확성을 모두 향상시킬 수 있음을 입증했습니다.



### SOS: Segment Object System for Open-World Instance Segmentation With Object Priors (https://arxiv.org/abs/2409.14627)
Comments:
          Accepted at ECCV 2024. Code available at this https URL

- **What's New**: 이번 논문에서는 Open-World Instance Segmentation (OWIS) 문제를 해결하기 위한 Segment Object System (SOS)을 제안합니다. SOS는 제한된 주석 클래스에서의 일반화를 통해 이미지 내 임의의 미지 객체를 세분화하는 방법을 탐구합니다.

- **Technical Details**: SOS는 고품질의 의사 주석(pseudo annotations)을 생성하기 위해 foundation model인 Segment Anything Model (SAM)을 활용합니다. Self-supervised Vision Transformers (ViTs)로부터 얻은 self-attention map을 사용하여 SAM을 객체에 집중시킵니다. 확인된 객체와 함께 의사 주석을 결합하여 표준 instance segmentation 시스템이 훈련되도록 합니다.

- **Performance Highlights**: 이 방법은 COCO, LVIS, ADE20k 데이터셋에서 이전 최첨단 시스템들보다 최대 81.6% 향상된 정밀도를 보여줍니다. SOS는 높은 품질의 의사 주석으로 인해 OWIS에서의 성능을 크게 개선합니다.



### Secrets of Edge-Informed Contrast Maximization for Event-Based Vision (https://arxiv.org/abs/2409.14611)
Comments:
          To be published in the 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)

- **What's New**: 이번 연구에서는 이벤트와 림에 기반한 데이터의 두 가지 양식을 결합한 새로운 하이브리드 접근 방식을 제안합니다. 이 방법은 이벤트와 엣지를 모두 활용하여 이벤트 기반 광학 흐름 추정을 크게 향상시키는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 이벤트 카메라로 수집된 비동기 이벤트의 대비 및 상관관계를 동시에 극대화하여 최적의 모션 트라젝토리를 추정합니다. 이를 위해 다중 스케일 및 다중 참조 기술을 적용했습니다.

- **Performance Highlights**: 새로운 하이브리드 프레임워크는 MVSEC, DSEC 및 ECD 데이터 세트에서 최첨단(event optical flow) 성능을 기록했습니다. 이 방법은 이미지에서 엣지가 포함된 샤프한 구조를 생성하는 데 있어 뛰어난 성능을 보여줍니다.



### Patch Ranking: Efficient CLIP by Learning to Rank Local Patches (https://arxiv.org/abs/2409.14607)
- **What's New**: 이 논문에서는 CLIP 모델의 Vision Transformer (ViT) 백본에서 패치 토큰을 가지런히 정리하는 새로운 방법을 제안합니다. 이 방법은 'Golden Ranking'이라는 최적 토큰 순위를 확립하고, 이 순위를 근사하는 경량 예측기를 통합합니다. 이를 통해 CLIP 모델의 성능을 유지하면서 패치 토큰의 40%를 줄였습니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 나뉘며, (1) 각 토큰의 유용성을 평가하여 'Golden Ranking'을 수립하고, (2) 이 순위를 근사하는 경량 예측기를 교육하며, (3) 패치 토큰이 제거된 후 발생할 수 있는 성능 저하를 보완하기 위해 모델을 최적화하는 과정을 포함합니다. 이 과정에서 학습 가능한 비주얼 토큰을 통합하여 성능 저하를 방지하고 모델의 정확성을 향상시킵니다.

- **Performance Highlights**: 이 방법을 통해 CLIP의 ViT에서 40%의 패치 토큰을 성공적으로 줄였음에도 불구하고 7개 데이터셋에서 평균 정확도는 0.3% 감소에 그쳤습니다. 이는 계산 효율성과 성능 간의 균형을 이루는 접근 방식의 유용성을 입증합니다.



### URSimulator: Human-Perception-Driven Prompt Tuning for Enhanced Virtual Urban Renewal via Diffusion Models (https://arxiv.org/abs/2409.14589)
- **What's New**: 이 논문은 Urban Physical Disorder (도시 물리적 불균형) 문제를 해결하기 위해, 안정적인 확산(stable diffusion)을 이용하여 도시 환경 개선을 시뮬레이션하는 새로운 프레임워크를 제안합니다. 또한, 사람의 인지 피드백을 결합하여 도시 재생에 대한 주관적 판단을 보완합니다.

- **Technical Details**: 본 연구에서는 prompt tuning 접근법을 개발하여 텍스트 기반 Stable Diffusion과 사람의 인지 피드백을 통합합니다. 이를 통해 거리 풍경 이미지의 로컬 지역을 반복적으로 편집하여 아름다움, 생동감, 안전 기준에 더 잘 맞도록 조정합니다. 실험 결과, 이 프레임워크는 도시 환경의 인식을 17.60% 안전성, 31.15% 아름다움, 28.82% 생동감 향상시키는 데 성공하였습니다.

- **Performance Highlights**: 기존의 DiffEdit와 같은 고급 방법들은 각각 2.31%, 11.87%, 15.84%의 개선만을 보였으나, 제안된 프레임워크는 다양한 가상 시나리오에서 뛰어난 성과를 나타냈습니다. 이 프레임워크는 이웃 개선, 건물 재개발, 녹지 공간 확대, 커뮤니티 정원 조성 등 다양한 도시 재생 시뮬레이션에 효과적으로 적용되었으며, 도시 계획 및 정책 입안에 유용한 통찰력을 제공합니다.



### Space evaluation based on pitch control using drone video in Ultima (https://arxiv.org/abs/2409.14588)
Comments:
          2 pages, 1 figure. Presented at Cascadia Symposium on Statistics in Sport (CASSIS) 2024

- **What's New**: 이 연구는 Ultimate 스포츠에서 공격 시 공간을 평가하는 새로운 지표인 Ultimate Scoring Opportunity (USO)를 제안합니다. 이 모델은 축구의 Pitch Control Model을 기반으로 하여 Ultimate의 규칙에 맞게 조정되었습니다.

- **Technical Details**: 연구에서는 드론을 이용해 필드 영상을 촬영하고, 포지션 데이터를 추출하기 위해 영상의 각도 보정이 이루어졌습니다. Ultimate의 점수 기회를 평가하기 위해, USO는 두 가지 요소인 공간의 가치를 나타내는 가중치와 패스 거리의 가중치를 결합하여 계산됩니다.

- **Performance Highlights**: 3대3 형태의 게임에서 USO 점수가 마지막 패스 직전에 증가했으며, 이는 공간을 창출한 다음 정확한 패스를 통해 점수를 올리는 데 기여한다는 것을 보여주었습니다. 반면, 스테이지의 턴오버 상황에서는 USO 점수의 증가가 관찰되지 않았습니다.



### Deep Learning Techniques for Atmospheric Turbulence Removal: A Review (https://arxiv.org/abs/2409.14587)
Comments:
          36 Pages, 8 figures

- **What's New**: 이 논문은 대기 난류(Atmospheric Turbulence)가 이미지에 미치는 영향을 분석하고, 다양한 최첨단 딥 러닝(Deep Learning) 네트워크의 성능을 비교하여 왜곡을 완화하는 방법을 제시합니다.

- **Technical Details**: 대기 난류는 영상 인식 및 추적의 성능을 저하시킵니다. 딥러닝 기반 접근법을 통해 기존의 복잡한 모델 기반 접근법에 비해 데이터 처리 속도가 빨라지고, 메모리 요구 사항이 적습니다. 이 연구에서는 Transformers, SWIN, Mamba와 같은 다양한 아키텍처의 성능을 평가합니다.

- **Performance Highlights**: 딥 러닝 방법으로 대기 난류의 왜곡을 완화할 수 있으며, 특히 TMT 방식의 개선된 버전과 SWIN3D 및 Mamba3D 모델이 효과적임을 보여줍니다. 이러한 방법들은 잡음 제거(Denoising), 선명화(Deblurring), 초해상도(Super-resolution) 작업에 적용됩니다.



### AR Overlay: Training Image Pose Estimation on Curved Surface in a Synthetic Way (https://arxiv.org/abs/2409.14577)
Comments:
          12th International Conference on Signal, Image Processing and Pattern Recognition (SIPP 2024)

- **What's New**: 이 연구는 Augmented Reality (AR) 분야에서 여러 로고 이미지를 동시에 탐지할 수 있는 새로운 파이프라인을 제안합니다. 기존 알고리즘이 특정 곡률 측정을 요구하는 한계를 극복하고, 구조적 제약을 이용하여 원본 이미지만을 입력으로 사용합니다.

- **Technical Details**: 제안된 알고리즘은 YOLOv8을 이용한 로고 탐지, Convolutional Neural Network (CNN)을 통한 직경 추정, SIFT를 활용한 특징 추출, 그리고 6D pose estimation을 위한 Perspective-n-Point (PnP) 문제 해결을 포함합니다. 또한, 20,000개의 이미지를 포함하는 합성 데이터셋을 Blender를 사용하여 생성하였습니다.

- **Performance Highlights**: 이 연구는 고급 로고 탐지 및 곡면 이미지 추정을 통한 실시간 곡선 이미지 추적에서의 개선을 보이며, 다양한 실제 시나리오에서의 유용성과 효율성을 크게 향상시킵니다.



### Event-ECC: Asynchronous Tracking of Events with Continuous Optimization (https://arxiv.org/abs/2409.14564)
- **What's New**: 이 논문에서는 새로운 event-based tracker가 제안되었다. 독립적으로 처리된 개별 이벤트에 대한 비동기성 프로세스를 이용하여 시간에 따라 공간 분포를 정렬하는 직접 매칭 알고리즘이 개발되었다. Enhanced Correlation Coefficient (ECC) 기준을 채택하여, 이벤트 하나당 2D motion warp를 계산하는 tracking 알고리즘이 제안되었다.

- **Technical Details**: 제안된 알고리즘은 개별 이벤트를 처리하며, 이벤트 기반 비동기 트래킹을 위한 새로운 방법론을 도입하였다. 각 이벤트에 대해 한 번의 최적화 스텝을 사용하여 non-discrete motion warp를 추정한다. 경량화된 버전으로는 점진적 처리와 업데이트 방법을 활용하여, 계산의 부담을 줄였다.

- **Performance Highlights**: 공식적으로 사용 가능한 데이터 세트에서 실험을 수행하였고, 기존의 최첨단 event-based 비동기 트래커보다 향상된 트래킹 정확도와 기능 수명을 보고하였다.



### GlamTry: Advancing Virtual Try-On for High-End Accessories (https://arxiv.org/abs/2409.14553)
- **What's New**: 이 연구는 주얼리 및 시계와 같은 액세서리에 대한 사진 현실적인 가상 착용 모델 부족 문제를 해결하는 것을 목표로 하고 있습니다. 기존의 가상 착용 모델이 의류 아이템에 주로 초점을 맞추고 있는 반면, 액세서리에 대한 시장의 공백을 메우고자 합니다.

- **Technical Details**: 본 연구에서는 의류를 위한 2D 가상 착용 모델에서의 기술과 MediaPipe Hand Landmarker와 같은 컴퓨터 비전 모델을 통합하여 액세서리 전용 데이터를 사용하는 맞춤형 모델을 재훈련합니다. 또한, 고해상도 이미지 데이터셋을 활용하여 액세서리에 대한 가상 착용 기술의 가능성을 평가합니다. 이를 위해 Dense-pose와 Human Parsing 등 다양한 전처리 절차를 포함하여 최종 쿼리 데이터로 변환하게 됩니다.

- **Performance Highlights**: 작은 데이터셋에도 불구하고 기존 의류 모델과 비교하여 위치 예측 성능이 향상되었습니다. 약 10,000장을 초과하는 대규모 데이터셋을 통해 이 모델의 가능성을 더욱 확장할 수 있음을 보여줍니다.



### TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps (https://arxiv.org/abs/2409.14543)
Comments:
          Research report

- **What's New**: 이 논문에서는 TrackNet 계열의 최신 모델인 TrackNetV4를 소개합니다. 이는 고급 시각적 특징과 학습 가능한 모션 주의 맵(motion attention maps)을 융합한 모션 인식 융합 메커니즘을 통해 향상되었습니다.

- **Technical Details**: TrackNetV4는 프레임 차분 맵(frame differencing maps)과 모션 프롬프트 레이어(motion prompt layer)를 활용하여 이동하는 공의 위치를 강조하고, 이로 인해 추적 성능이 개선됩니다. 이 과정에서 모델은 시각적 특징에만 의존하지 않고, 모션 정보를 명시적으로 통합하여 정밀한 추적 및 궤적 예측(trajectory prediction)에 유리합니다.

- **Performance Highlights**: 테니스 공과 셔틀콕 데이터셋을 기반으로 한 실험 결과, TrackNetV4는 TrackNetV2와 V3의 추적 성능을 개선하는 것으로 나타났습니다.



### Towards Model-Agnostic Dataset Condensation by Heterogeneous Models (https://arxiv.org/abs/2409.14538)
Comments:
          ECCV 2024, 17 pages, 3 figures, 4 tables in main paper

- **What's New**: 이번 연구에서는 Heterogeneous Model Dataset Condensation (HMDC)라는 새로운 방법을 도입하여 모델 간의 상호작용을 통해 보편 적용 가능한 응축 이미지(condensed image)를 생성하는 방법을 제안합니다.

- **Technical Details**: HMDC는 Gradient Balance Module (GBM)과 Mutual Distillation (MD)을 활용하여 이질적인 모델 간의 그래디언트 크기 차이와 의미적 거리 문제를 해결합니다. GBM은 각 최적화 목표의 그래디언트 크기를 수집하여 손실 크기를 제어하며, MD는 공간-의미 분해 방법을 통해 두 모델 간의 특징 매칭 과정을 수행합니다. 이를 통해 모든 모델 독립적으로 합성 이미지의 일관된 업데이트를 가능합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법은 얕은 모델부터 널리 사용되는 대형 모델까지 지속적으로 뛰어난 성능을 보였습니다. 기존의 모델 특정 응축 이미지 문제를 해결하며, 보다 범용적인 적용 가능성을 높였습니다.



### Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding (https://arxiv.org/abs/2409.14485)
- **What's New**: Video-XL은 80GB GPU에서 단일 모델로 1024 프레임을 처리할 수 있으며, Needle-in-a-Haystack 평가에서 100%에 가까운 정확도를 달성합니다. 이는 긴 비디오 이해를 위한 효율적인 솔루션을 제공하며, 기존 모델에서 겪었던 최장 비디오 처리 시의 성능 저하 문제를 극복합니다.

- **Technical Details**: Video-XL은 LLM(대형 언어 모델), Vision Encoder, Cross-modality Projector의 세 가지 주요 모듈로 구성되어 있으며, Visual Context Latent Summarization 기법을 사용하여 긴 비디오의 시각적 컨텍스트를 압축합니다. 또한, CLIP-ViT-L을 사용하여 비디오와 이미지를 인코딩하며, 각 비디오 프레임을 336x336 해상도로 처리합니다.

- **Performance Highlights**: Video-XL은 MLVU, Video-MME, LongVideoBench, VNBench와 같은 긴 비디오 이해 벤치마크에서 우수한 성과를 보입니다. 또한, 이 모델은 감시 비디오에서 이상 징후를 탐지하고, 긴 비디오의 광고 삽입을 식별하는 등의 작업에서 탁월한 잠재력을 보여줍니다.



### Effectively Enhancing Vision Language Large Models by Prompt Augmentation and Caption Utilization (https://arxiv.org/abs/2409.14484)
- **What's New**: 본 논문에서는 Vision Language Large Models (VLLMs)의 출력 결과가 입력 이미지와 관련되지 않는 "hallucination phenomenon" 문제를 해결하기 위한 새로운 프레임워크인 Prompt Augmentation and Caption Utilization (PACU)를 제안합니다.

- **Technical Details**: PACU는 자동으로 다양한 프롬프트를 증강 및 평가하는 모듈과 이미지 캡션을 결합하여 응답 생성을 돕는 캡션 활용 생성 메커니즘을 포함합니다. 이 프레임워크는 기존의 VLLMs와 호환되며, 입력 및 응답 생성의 두 측면에서 VLLM의 성능을 향상시킵니다.

- **Performance Highlights**: PACU는 다양한 테스트 결과에서 기존 기법과 결합하여 VLLM 성능을 효과적으로 향상시킬 수 있음을 보여줍니다. VLLM의 프롬프트 처리 능력을 개선하며, 기존의 anti-hallucination 솔루션과 함께 사용할 수 있는 장점이 있습니다.



### One Model for Two Tasks: Cooperatively Recognizing and Recovering Low-Resolution Scene Text Images by Iterative Mutual Guidanc (https://arxiv.org/abs/2409.14483)
- **What's New**: 본 논문에서는 LR(Low-Resolution) 이미지를 효과적으로 인식하고 복원하는 새로운 방법인 IMAGE(Iterative MutuAl GuidancE)를 제안합니다. 기존의 STR(Scene Text Recognition) 및 STISR(Scene Text Image Super-Resolution) 모델의 한계를 극복하고 두 가지 모델을 별도로 최적화하여 성능 저하를 방지합니다.

- **Technical Details**: IMAGE는 인식 전용 STR 모델과 LR 이미지를 복원하는 STISR 모델로 구성되어 있습니다. 두 모델은 서로 협력하여 최적의 성능을 이끌며, STR 모델은 STISR 모델에 고수준의 의미 정보를 제공하고, STISR 모델은 STR 모델에 저수준의 픽셀 정보를 제공합니다. 이러한 상호 안내 메커니즘을 통해 두 모델이 협력적으로 목표를 달성할 수 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 제안한 IMAGE 방법은 두 개의 LR 데이터셋에서 기존 기술들보다 인식 성능과 초해상도(Super-Resolution) 충실도(fidelity) 모두에서 우수한 성능을 보였습니다.



### SynBench: A Synthetic Benchmark for Non-rigid 3D Point Cloud Registration (https://arxiv.org/abs/2409.14474)
- **What's New**: 본 논문에서는 SynBench라는 새로운 비강체(Non-rigid) 포인트 클라우드 등록 데이터셋을 소개합니다. 이 데이터셋은 Soft body 시뮬레이션을 위한 도구인 SimTool을 사용하여 생성되었으며, 다양한 도전 과제를 포함하여 포인트 클라우드 등록 방법을 공정하게 평가할 수 있는 기준점을 제공합니다.

- **Technical Details**: SynBench는 다양한 변형(deformation) 수준, 노이즈(noise), 아웃라이어(outlier), 및 불완전성(incompleteness)을 포함한 여러 가지 도전 과제를 제공합니다. 각 데이터 샘플은 변형 전후의 대응하는 포인트에 대한 Ground Truth 정보를 제공하여 등록 방법의 정확성을 평가하는 데 유용합니다. 데이터셋은 30개의 원시 객체(primitive objects)와 그에 따른 5개의 주요 카테고리로 구성되어 있습니다.

- **Performance Highlights**: SynBench는 기존 데이터셋에 비해 세 가지 특성을 가지고 있습니다: (1) 비강체 포인트 클라우드 등록을 위한 다양한 도전 과제를 제공하는 최초의 벤치마크, (2) 다양한 난이도의 도전 과제를 포함, (3) 변형 전후의 대응 포인트에 대한 Ground Truth를 포함. 이를 통해 향후 비강체 포인트 클라우드 등록 방법의 성능을 공정하게 비교할 수 있습니다.



### Low-Light Enhancement Effect on Classification and Detection: An Empirical Study (https://arxiv.org/abs/2409.14461)
Comments:
          8 pages,8 figures

- **What's New**: 본 논문은 저조도(低照度) 이미지 향상(Low-Light Image Enhancement, LLIE) 기술이 고수준 비전 태스크(예: 이미지 분류 및 객체 탐지)에 미치는 영향을 포괄적으로 평가하여, 사람의 시각적 해석 향상에는 기여하지만, 컴퓨터 비전 태스크에서는 일관되지 않거나 해로운 영향을 미칠 수 있음을 발견하였습니다.

- **Technical Details**: LLIE 방법은 전통적인 방식과 심층 학습 기반 방식으로 나뉩니다. 전통적인 방법은 히스토그램 평활화(histogram equalization) 및 레티넥스 이론(retinex theory)과 같은 기술을 포함하고, 심층 학습 방법은 데이터를 통해 특징과 패턴을 자동으로 학습하여 더 강력하고 효과적인 결과를 도출합니다. PSNR과 SSIM은 이미지 품질 평가에 자주 사용되는 지표입니다.

- **Performance Highlights**: 실험 결과, LLIE 방법이 저조도 환경에서 고수준 태스크의 성능을 눈에 띄게 향상시키지 못하며, 오히려 특정 응용에서 이러한 태스크의 효과성을 감소시킬 수 있다는 반증을 제시하였습니다.



### Fake It till You Make It: Curricular Dynamic Forgery Augmentations towards General Deepfake Detection (https://arxiv.org/abs/2409.14444)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이번 연구에서는 Curricular Dynamic Forgery Augmentation (CDFA)라는 새로운 딥페이크 탐지 방법을 제시합니다. 이는 딥페이크 감지기와 위조 증강 정책 네트워크를 함께 훈련시킵니다. CDFA는 훈련 과정에서 점진적으로 위조 증강을 적용하며, 각 이미지에 대한 적절한 위조 증강 작업을 선택하는 동적 위조 검색 전략을 포함하여 일반화를 최적화합니다.

- **Technical Details**: CDFA는 monotonic curriculum (MC) 전략을 통해 p-fake 샘플의 비율을 점진적으로 증가시키고 o-fake 샘플의 비율을 줄이는 방식으로 훈련합니다. 또한, Self-shifted Blending Image (SSBI)라는 새로운 위조 증강 방법을 제안하여 시간적 불일치를 모방하는 간단한 방법을 제공합니다. 이 방법은 기존의 위조 증강들이 모방하지 못한 시간적 아티팩트를 도입합니다.

- **Performance Highlights**: 실험 결과, CDFA는 다양한 naive 딥페이크 탐지기의 일반화 성능을 크게 향상시키며, 여러 벤치마크 데이터셋에서 기존의 여러 방법들보다 우수한 성능을 달성하는 것을 보여줍니다. CDFA는 플러그 앤 플레이 방식으로 적용될 수 있으며, 다양한 크로스-데이터셋 및 크로스-조작 벤치마크에서 탁월한 성과를 기록했습니다.



### EM-DARTS: Hierarchical Differentiable Architecture Search for Eye Movement Recognition (https://arxiv.org/abs/2409.14432)
Comments:
          Submited to IEEE Transactions on Information Forensics and Security

- **What's New**: 본 논문에서는 EM-DARTS라는 계층적 차별화 아키텍처 검색 알고리즘을 제안하여 눈 움직임 인식을 위한 최적의 딥러닝 아키텍처를 자동으로 설계하는 방법을 소개합니다. EM-DARTS는 기존 DARTS 방식의 한계를 극복하고, 각 계층의 셀에 대해 최적화된 아키텍처를 정의할 수 있습니다.

- **Technical Details**: EM-DARTS는 글로벌 및 로컬 차별화 신경망 아키텍처 검색 방법을 사용하여 최적 아키텍처를 검색합니다. 로컬 검색 전략은 각 셀에 최적의 아키텍처를 찾는 데 중점을 두고, 글로벌 검색 전략은 목표 네트워크의 아키텍처를 최적화합니다. 또한, 각 층의 정보량을 계산하기 위해 전이엔트로피(transfer entropy)를 사용하며, 이를 통해 중복성을 줄입니다.

- **Performance Highlights**: 세 개의 공용 데이터베이스에서 수행한 실험 결과, EM-DARTS는 눈 움직임 인식 분야에서 최첨단 인식 성능을 달성하는 최적의 아키텍처를 생성할 수 있음을 입증했습니다.



### Pomo3D: 3D-Aware Portrait Accessorizing and Mor (https://arxiv.org/abs/2409.14430)
- **What's New**: 본 논문에서는 Pomo3D라는 3D 초상화 조작 프레임워크를 제안한다. 이 프레임워크는 초상화와 액세서리를 분해하고 다시 조합하여 자유로운 액세서리 추가를 가능하게 하며, 다양한 액세서리를 동시에 착용한 모습을 제공한다. 기존의 방법들이 제시하는 제약을 뛰어넘어 보다 명확한 조작을 가능케 한다.

- **Technical Details**: Pomo3D의 주요 특징은 두 개의 별도 장면 표현을 도입한 것이다. 하나는 초상화용이고, 다른 하나는 덜 일반적인 액세서리용이다. 이러한 구조를 통해 High-resolution RGB 이미지를 생성하며, 'Scribble2Accessories' 모듈을 통해 사용자 그리기 스케치를 바탕으로 3D 액세서리를 생성할 수 있다. 또한, 편향을 완화하기 위해 'bias-conscious mapper'를 설계하여 실제 데이터셋에서 발견되는 편향된 연관성을 줄인다.

- **Performance Highlights**: Pomo3D는 액세서리 조작에서 제공하는 자유도와 포괄적인 초상화 편집 옵션으로 기존의 3D 생성 모델 중 가장 높은 수준의 편집 가능성을 가지고 있다. 사용자 그리기 스케치로부터 3D 액세서리를 생성할 수 있으며, 액세서리와 초상화를 별개로 모델링하여 조합할 수 있어 다양한 스타일의 초상화를 생성할 수 있다.



### Prior Knowledge Distillation Network for Face Super-Resolution (https://arxiv.org/abs/2409.14385)
- **What's New**: 본 연구는 prior knowledge distillation network (PKDN)을 통해 얼굴 초해상도(Face Super-Resolution, FSR) 문제를 해결하는 새로운 접근 방식을 제안합니다. 기존의 FSR 방법들이 저해상도 이미지에서 얻은 facial priors의 정확도 문제로 어려움을 겪었으나, PKDN은 teacher network와 student network 간의 지식 전이를 통해 이 문제를 극복합니다.

- **Technical Details**: PKDN은 teacher network가 고해상도(HR) 이미지에서 추출한 facial parsing maps을 사용하여 저해상도(LR) 이미지를 학습하는 구조로 되어 있습니다. 이때, parsing map fusion block(PFB)과 feature fusion block(FFB)을 통하여 prior 정보를 효과적으로 활용하고, multi-scale 특성을 유지하여 reconstruction 과정에서 정보 손실을 방지합니다. 또한 이 과정에서 ℒ1 loss 함수가 사용됩니다.

- **Performance Highlights**: CelebA와 Helen 데이터셋을 활용한 실험 결과, PKDN 방법은 기존 FSR 기법들보다 우수한 품질의 고해상도 얼굴 이미지를 생성하는 것으로 나타났습니다. 이는 facial prior 정보를 통해 FSR 성능을 크게 향상시켰음을 의미합니다.



### GroupDiff: Diffusion-based Group Portrait Editing (https://arxiv.org/abs/2409.14379)
Comments:
          ECCV 2024

- **What's New**: 이번 연구에서는 GroupDiff라는 새로운 그룹 초상화 편집 도구를 소개합니다. GroupDiff는 사람 추가, 삭제 및 조정을 통해 기존 개인의 모습을 유지하면서 자연스러운 상호작용을 생성하는 독창적인 접근 방식을 제안합니다.

- **Technical Details**: 우리의 방법은 1) 데이터 엔진을 통한 훈련 데이터의 생성, 2) 외형 보존을 위한 인식 모듈, 3) 각 개인의 위치를 정교하게 조정하는 경계 상자를 통해 작업을 수행합니다. 특히, 인간 상호작용을 반영하는 합리적인 편집을 위해 
재료들이 인식 모듈 및 뼈대 정보를 통해 서로의 특징을 기반으로 결합됩니다.

- **Performance Highlights**: 실험 결과, GroupDiff는 기존 방법들보다 우수한 성능을 보이며, 사용자가 요구하는 상호작용을 정확하게 다양하게 조정할 수 있는 유연성을 제공합니다.



### Memory Matching is not Enough: Jointly Improving Memory Matching and Decoding for Video Object Segmentation (https://arxiv.org/abs/2409.14343)
Comments:
          Accepted to ICPR2024

- **What's New**: 이번 논문에서는 비디오 객체 분할(Video Object Segmentation, VOS)에서 발생하는 잘못된 매칭(false matching) 문제와 중요한 정보 손실을 해결하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안하는 방법은 단기 메모리(short-term memory)와 장기 메모리(long-term memory) 관리를 위한 cost-aware matching 및 cross-scale matching 메커니즘을 포함합니다. 또한, 읽어내기 디코딩(readout decoding) 단계에서 정보 복구를 위한 보상 디코딩(compensatory decoding) 메커니즘을 구현하여 초기 메모리 읽어내기에서 손실된 중요한 정보를 보완합니다.

- **Performance Highlights**: 이 방법은 DAVIS 2016&2017 Val에서 각각 92.4%와 88.1%, DAVIS 2017 Test에서 83.9%, YouTubeVOS 2018&2019 Val에서 각각 84.8%와 84.6%의 성능을 기록하며 여러 인기 벤치마크에서 최첨단 성능을 달성합니다.



### Self-Supervised Audio-Visual Soundscape Stylization (https://arxiv.org/abs/2409.14340)
Comments:
          ECCV 2024

- **What's New**: 본 논문에서는 입력된 음성이 다른 장면에서 녹음된 것처럼 들리도록 변형하는 방법을 제안합니다. 이를 위해 음향-영상 조건 예시를 활용하며, 자연 비디오에서 반복적인 소리 이벤트와 텍스처를 학습하여 음성을 맥락에 맞게 조정합니다.

- **Technical Details**: 모델은 자가 지도 학습(self-supervision)을 사용하며, 잠재적 확산 모델(latent diffusion model)을 기반으로 합니다. 주어진 조건 예시에서 소리 속성을 추출하고, 이를 입력 음성과 결합하여 음향 특성과 환경 소음이 반영된 음성을 생성합니다. 이 과정에서 비디오 내의 두 개의 인접한 음향-영상 클립을 무작위로 샘플링하고, 하나에서 장면 특정 속성을 제거한 후, 다른 클립을 조건 힌트로 사용하여 원래 음성을 복구하는 방법을 학습합니다.

- **Performance Highlights**: 모델은 라벨이 없는 실제 동영상 데이터를 활용하여 성공적으로 훈련되며, 추가적인 시각적 신호가 음향 예측 능력을 향상시킬 수 있음을 보여줍니다. 실험 결과, 음향-영상 조건 예시와의 조합을 통해 입력된 음성을 다양한 환경에 맞게 효과적으로 스타일화(stylize)할 수 있습니다.



### Zero-Shot Skeleton-based Action Recognition with Dual Visual-Text Alignmen (https://arxiv.org/abs/2409.14336)
- **What's New**: 본 논문에서는 Dual Visual-Text Alignment (DVTA)라는 새로운 접근 방식을 제안하여 스켈레톤 기반 제로샷 액션 인식을 위한 기술을 발전시키고 있습니다. 이 방법은 Direct Alignment (DA)와 Augmented Alignment (AA)라는 두 개의 정렬 모듈을 포함하여, 보강된 의미 설명(Semantic Description Enhancement, SDE)을 통해 텍스트와 스켈레톤 특징 간의 격차를 줄이고 있습니다.

- **Technical Details**: DVTA는 두 가지의 주요 구성 요소를 포함합니다. 첫 번째 구성 요소인 Direct Alignment (DA) 모듈은 시각적 프로젝터를 사용하여 스켈레톤 특징을 의미 공간으로 매핑하고, 그 후 크로스 주의 기반의 SDE를 통해 스켈레톤과 텍스트 간의 연결을 향상시킵니다. 두 번째 구성 요소인 Augmented Alignment (AA) 모듈은 깊이 있는 거리 학습(deep metric learning)을 이용하여 스켈레톤과 텍스트 간의 유사성을 학습합니다. 이 모든 과정은 KL 다이버전스(kullback-leibler divergence) 손실을 통해 최적화됩니다.

- **Performance Highlights**: 제안된 DVTA는 NTU RGB+D, NTU RGB+D 120, PKU-MMD와 같은 여러 인기 있는 제로샷 스켈레톤 기반 액션 인식 벤치마크에서 최첨단 성능(state-of-the-art performance)을 달성하였습니다.



### PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects (https://arxiv.org/abs/2409.14331)
Comments:
          Accepted to ECCV 2024

- **What's New**: PISR(Polarimetric Neural Implicit Surface Reconstruction)라는 혁신적인 방법을 소개하며, 복잡한 radiance modeling을 통해 높은 정확도로 textureless 및 specular 표면을 재구성 가능하게 함.

- **Technical Details**: 이 방법은 polarimetric loss를 사용하여 외관과 무관하게 형태를 정제하고, hash-grid 기반의 neural signed distance function을 활용하여 재구성을 가속화합니다. 이 방법은 L1 Chamfer distance가 0.5 mm, F-score가 99.5%로 측정되며, 기존 방법에 비해 4~30배 빠른 수렴 속도를 자랑합니다.

- **Performance Highlights**: PISR는 textureless 및 specular 물체에 대한 표면 재구성의 정확성을 높이며, 다양한 실험적 결과를 통해 높은 강건함과 정확성을 입증합니다.



### Scene-Text Grounding for Text-Based Video Question Answering (https://arxiv.org/abs/2409.14319)
- **What's New**: 이번 논문은 'Grounded TextVideoQA'에 대한 연구를 제안하며, 모델이 질문에 답변하고 관련된 scene-text 영역을 시공간적으로 위치 지정(location)하도록 강제합니다. 이를 통해 QA를 scene-text 인식과 분리하여 해석 가능한 QA 연구를 촉진합니다.

- **Technical Details**: 우리는 'T2S-QA' 모델을 제안하여 약한 감독 하에 scene-text grounding 및 grounded TextVideoQA를 위한 분리된 시간-공간 대비 학습 전략을 강조합니다. 이 모델은 질문에 적합한 scene texts의 시간적 구분과 비주얼적인 시각화를 동시에 다룹니다.

- **Performance Highlights**: T2S-QA는 기존 기술들보다 우수한 결과를 달성했지만, 인간 성능과의 큰 성능 차이는 여전히 개선할 여지가 많음을 나타냅니다. 구체적으로, 인간은 질문에 대한 답을 77% 정확도로 적절히 grounding할 수 있는 반면, T2S-QA는 28%에 불과합니다.



### MVPGS: Excavating Multi-view Priors for Gaussian Splatting from Sparse Input Views (https://arxiv.org/abs/2409.14316)
Comments:
          Accepted by ECCV 2024, Project page: this https URL

- **What's New**: 이번 논문에서는 3D Gaussian Splatting(3DGS)를 기반으로 하는 새로운 몇 장의 이미지로 이루어진 Novel View Synthesis(MVPGS) 방법을 제안합니다. 과거 연구들이 NeRF(Neural Radiance Field)와 같은 기존 방법에서 고질적인 과적합(overfitting) 문제를 해결하려고 시도했지만, 여전히 충분한 입력이 없는 경우 성능이 저하되는 문제를 겪고 있었습니다. MVPGS는 이러한 문제를 해결하기 위해 최근 학습 기반의 Multi-view Stereo(MVS)를 활용하여 3DGS의 기하학적 초기화를 향상시킵니다.

- **Technical Details**: MVPGS는 MVS를 통해 얻어진 기하학 정보를 사용하여 3D Gaussian Splatting 초기화에서 나오는 잠재적 외형 정보를 활용합니다. 이를 위해 forward-warping 방법을 도입하여 추가적인 appearance constraints를 생성하고, Gaussian 파라미터에 대해 view-consistent geometry constraint를 적용하여 최적화의 수렴을 돕습니다. 또한, monocular depth regularization을 활용하여 정확한 기하학적 구조를 보장합니다.

- **Performance Highlights**: 제안된 방법은 LLFF, DTU, NVS-RGBD, Tanks and Temples 데이터셋에서 실험을 진행한 결과, 실시간 렌더링 속도에서 최첨단 성능을 달성하며, 적은 이미지를 사용한 상황에서도 높은 품질을 유지했습니다.



### Anisotropic Diffusion Probabilistic Model for Imbalanced Image Classification (https://arxiv.org/abs/2409.14313)
- **What's New**: 본 논문에서는 비정상 확산 확률 모델(Anisotropic Diffusion Probabilistic Model, ADPM)을 제안하여 불균형 이미지 분류 문제를 해결합니다.

- **Technical Details**: ADPM은 다양한 클래스 샘플의 확산 속도를 제어하여 분류 정확성을 향상시킵니다. 본 모델은 노이즈 수준을 선택하기 위한 이론적 전략을 제공하며, 전이 과정에서 이미지의 전역 및 지역 정보를 통합합니다.

- **Performance Highlights**: 제안된 방법은 다수의 의학 데이터셋에서 실험을 통해 귀중한 클래스의 분류 정확성을 개선하였습니다. PAD-UFES와 HAM10000 데이터셋에서 기존 모델보다 각각 4% 및 3% 향상된 F1 점수를 달성했습니다.



### DilateQuant: Accurate and Efficient Diffusion Quantization via Weight Dilation (https://arxiv.org/abs/2409.14307)
Comments:
          Code: this http URL

- **What's New**: 본 논문에서는 DilateQuant라는 새로운 양자화 프레임워크를 제안하여, 기존의 양자화 기술보다 정확도와 효율성을 동시에 향상시킵니다. 이 프레임워크는 특히 diffusion models(확산 모델)의 고유한 문제인 wide-range (넓은 범위)와 time-varying (시간 변화) activation(활성화)을 해결하는 데 중점을 두었습니다.

- **Technical Details**: DilateQuant는 unsaturated in-channel (비포화 채널) weights(가중치)를 활용하여 Weight Dilation (WD) 기법을 통해 가중치 범위를 유지한 채로 활성화 범위를 줄입니다. 이로 인해 activation quantization(활성화 양자화)가 용이해지고, 모델이 학습 단계에서 수렴하는 데 도움이 됩니다. 또한, Temporal Parallel Quantizer (TPQ)를 설계하여 서로 다른 시간 단계에 대해 병렬 양자화를 지원, 성능을 크게 향상시키며, Block-wise Knowledge Distillation (BKD)를 통해 전체 모델을 재학습할 필요 없이 효율적으로 성능을 개선합니다.

- **Performance Highlights**: DilateQuant는 기존 방법들과 비교하여 낮은 양자화 설정(6-bit, 4-bit)에서 더 뛰어난 성능을 보여줍니다. 다양한 모델(DDPM, LDM-4, LDM-8, Stable-Diffusion)과 데이터셋(CIFAR-10, LSUN-Bedroom, LSUN-Church, ImageNet, MS-COCO)에서 실험을 통해 그 우수성을 입증했습니다.



### Deep Learning Technology for Face Forgery Detection: A Survey (https://arxiv.org/abs/2409.14289)
- **What's New**: 이 논문은 최신 딥러닝 기반의 얼굴 변조 탐지 방법들에 대한 종합적인 조사 연구를 제공합니다.

- **Technical Details**: 현재 딥페이크(Deepfake) 기술의 진전과 다양한 데이터셋의 특성을 분석하여, 딥러닝 기반 얼굴 변조 탐지의 주요 도전 과제를 진단합니다. 각기 다른 카테고리의 변조 탐지 기법과 최첨단 탐지 방법들을 체계적으로 검토하였습니다.

- **Performance Highlights**: 기존 탐지 방법의 한계를 분석하고, 성능 향상과 일반화 문제를 해결하기 위한 미래 연구 방향을 제시합니다.



### Lidar Panoptic Segmentation in an Open World (https://arxiv.org/abs/2409.14273)
Comments:
          Pre-print. Accepted in the International Journal of Computer Vision, 19 Sept 2024. Code available at this https URL

- **What's New**: 이 논문에서는 Lidar Panoptic Segmentation (LPS)를 오픈 월드 환경에서 다루는 새로운 접근 방식을 소개합니다. LiPSOW라는 새로운 평가 프로토콜을 통해 기존의 한정된 클래스 어휘를 넘어서는 새로운 객체 인스턴스를 인식하고 분할하는 방법을 제안합니다.

- **Technical Details**: LiPSOW는 SemanticKITTI 데이터셋을 사용하여 사전 정의된 클래스의 지식을 바탕으로 훈련하고, KITTI360 데이터셋에서 새로운 클래스 인스턴스를 평가합니다. 연구에서는 객체 인식을 위해 사전 지식을 최대한 활용하면서도, 클래스 비의존적인 하향식 그룹화 방법을 통해 기존 클래스와 새로운 클래스에서 신뢰할 수 있는 성능을 달성하는 전략을 사용합니다.

- **Performance Highlights**: 제안된 방법은 알려진 클래스에 대해 학습된 점 그룹화 방법보다 성능이 뛰어나며, 알려진 클래스뿐만 아니라 미지의 클래스에 대해서도 높은 정확도로 분할을 수행합니다. 이는 LPS에서 알려지지 않은 객체의 인식을 위한 강력한 대안을 제공합니다.



### Combining Absolute and Semi-Generalized Relative Poses for Visual Localization (https://arxiv.org/abs/2409.14269)
- **What's New**: 본 논문은 구조 기반(strcture-based)과 구조가 없는(structure-less) 포즈 추정 기법의 조합이 실질적으로 성능 향상에 기여할 수 있는지를 탐구합니다. 특히, 2D-2D 및 2D-3D 매칭에 따라 포즈를 선택하는 방법을 제안합니다.

- **Technical Details**: 제안하는 방법은 RANSAC 루프 내에서 P3P 솔버와 E5+1 솔버를 사용하며, 각 이터레이션에서 2D-3D 및 2D-2D 매칭으로부터 포즈 추정을 수행합니다. 실험 결과, 선택 전략이 포즈 정확성에 미치는 영향을 분석하며, 구조 기반과 구조가 없는 기법의 조합이 효과적임을 보입니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험 결과, 구조 기반 및 구조가 없는 포즈 추정 전략의 선택이 포즈 정확성에 상당한 영향을 미친다는 것을 보여주었습니다. 적은 데이터 이미지 또는 부정확한 3D 기하학이 제공되는 경우에도 성능 향상에 기여할 수 있습니다.



### End to End Face Reconstruction via Differentiable PnP (https://arxiv.org/abs/2409.14249)
Comments:
          Accepted by ECCV2022 workshop

- **What's New**: ECCV 2022 WCPA Challenge에서 제안된 얼굴 복원(Face Reconstruction) 및 얼굴 랜드마크 탐지(Face Landmark Detection) 문제를 해결하기 위해 이중 가지 네트워크를 설계하여 3D 얼굴 좌표와 2D 랜드마크를 동시에 예측합니다.

- **Technical Details**: 이 방법은 Perspective-n-Points (PnP) 레이어를 통해 3D 메쉬와 2D 랜드마크를 결합하여 더욱 정확한 6DoF 포즈 추정을 제공합니다. 이중 가지 네트워크는 각각 3D 얼굴 메쉬와 랜드마크를 예측하며, Gaussian Negative Log Loss (GNLL)를 손실 함수로 사용해 토대를 둡니다.

- **Performance Highlights**: MVP-Human 데이터셋에서 경쟁력 있는 성능을 달성하며 대회에서 3위를 차지했습니다.



### Cloud Adversarial Example Generation for Remote Sensing Image Classification (https://arxiv.org/abs/2409.14240)
- **What's New**: 본 논문은 원거리 감지 이미지의 인공적 공격에서 새로운 접근법으로, 일반적인 적대적 공격 방법이 생성하는 비자연스러운 수정 대신, 인간의 인식에 더 잘 맞도록 구름을 생성하는 방식의 효과를 제안합니다.

- **Technical Details**: 우리는 Perlin noise 기반의 구름 생성 공격 방법을 제안합니다. 이 방법은 Perlin Gradient Generator Network (PGGN)를 설계하여, 그래디언트 파라미터 벡터를 입력으로 받아 다양한 스케일에서의 Perlin noise의 그래디언트 벡터 격자를 출력합니다. 생성된 구름 마스크는 혼합 계수 벡터와 스케일링 팩터에 따라 가중치가 주어지고 합산되어 최종 구름 마스크를 생성합니다. 이 과정은 구름 생성을 블랙박스 최적화 문제로 변환합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 강력한 공격 능력을 가지며 높은 쿼리 효율성을 달성합니다. 또한 생성된 적대적 예제의 전이 가능성 및 적대적 방어 시나리오에서의 강건성도 분석하였습니다.



### Masks and Boxes: Combining the Best of Both Worlds for Multi-Object Tracking (https://arxiv.org/abs/2409.14220)
- **What's New**: 본 논문에서는 Multi-object tracking (MOT) 분야에서 시간적으로 전파된 segmentation mask를 활용하여 tracking-by-detection 방법론을 개선하는 McByte라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: McByte는 bounding box와 segmentation mask 정보를 결합하여 객체 간의 연관성을 강화하고, 개별 비디오 시퀀스에 대한 튜닝 없이 강한 association cue를 제공합니다. 이 접근법은 네 가지 벤치마크 데이터셋 (DanceTrack, MOT17, SoccerNet-tracking 2022, KITTI-tracking)에서 테스트되어 모든 경우에서 성능 향상을 보였습니다.

- **Performance Highlights**: McByte는 기존의 mask 기반 방법들보다 우수한 성능을 보이며, 특히 DanceTrack, SoccerNet-tracking 2022, KITTI-tracking 데이터셋에서 tracking-by-detection 알고리즘 중 가장 높은 성능을 기록했습니다.



### @Bench: Benchmarking Vision-Language Models for Human-centered Assistive Technology (https://arxiv.org/abs/2409.14215)
Comments:
          Accepted by WACV 2025, project page: this https URL

- **What's New**: 이번 연구에서는 시각 장애인을 위한 Assistive Technology (AT)를 평가하기 위한 새로운 비전-언어(Vision-Language) 벤치마크인 @Bench를 제안했습니다. 이 벤치마크는 시각 장애인의 요구를 반영하여 다섯 가지 주요 비전-언어 과제를 포함합니다.

- **Technical Details**: @Bench는 Panoptic Segmentation (PS), Depth Estimation (DE), Optical Character Recognition (OCR), Image Captioning (IC), Visual Question Answering (VQA) 등 시각 장애인 관련 5가지 비전-언어 과제를 포함하며, 사용자 중심의 설계를 통해 효율성과 성능을 평가하는 프레임워크를 제공합니다. 또한, 새로운 다기능 AT 모델(@Model)을 제안하여 이러한 과제를 동시에 해결합니다.

- **Performance Highlights**: 제안된 @Model은 모든 과제를 한 번에 처리할 수 있으며, 일반적인 방법들과 비교해 경쟁력 있는 성능을 보여줍니다. 이 모델은 매개변수를 크게 줄일 수 있어, 휴대용 장치에 배포하기에 적합합니다.



### Egocentric zone-aware action recognition across environments (https://arxiv.org/abs/2409.14205)
Comments:
          Project webpage: this https URL

- **What's New**: 이 논문은 활동 중심 지역(activity-centric zones)와 제안된 모델인 EgoZAR의 개념을 도입하여, 환경의 외관에 관계없이 행동 인식을 개선할 수 있는 방법을 제시합니다.

- **Technical Details**: Egocentric Action Recognition (EAR) 시스템은 도메인 독립적인 표현을 사용하여 다양한 환경에서 활동 중심 지역의 활용 가능성을 개선합니다. 이는 RGB 모델이 도메인 특정 편향을 제거할 수 있도록 합니다. 논문에서는 EPIC-Kitchens-100과 Argo1M 데이터셋을 통해 실험 결과를 검증합니다.

- **Performance Highlights**: EgoZAR 모델은 EK100에서 최신 도메인 일반화(Domain Generalization) 결과를 달성하였으며, Argo1M에서도 경쟁적인 성능을 보여주었습니다.



### LATTE: Improving Latex Recognition for Tables and Formulae with Iterative Refinemen (https://arxiv.org/abs/2409.14201)
- **What's New**: 이 논문은 처음으로 LaTeX 인식(LaTeX Recognition)에 대한 반복적 개선(iterative refinement) 접근 방식을 제안하는 LATTE 프레임워크를 소개합니다.

- **Technical Details**: LATTE는 delta-view라는 피드백을 활용하여 잘못 인식된 LaTeX 소스 코드의 불일치 부분을 정확히 찾아내고 이를 수정하는 모델로 구성됩니다. 구체적으로, 이미지 간의 차이를 식별하는 ImageEdit 알고리즘을 사용하여 올바른 LaTeX 코드를 생성합니다. 이 과정은 LaTeX 공식과 테이블을 포함하여 PDF에서 LaTeX 소스를 효과적으로 추출하는 데 중점을 두고 있습니다.

- **Performance Highlights**: LATTE는 LaTeX 공식과 테이블의 소스 추출 정확도를 기존 기술보다 최소 7.07% 개선하였으며, GPT-4V와 비교하여 46.08%의 성공적인 개선율을 달성하였습니다. 또한, LATTE의 전체 결함 위치 탐지 정확도는 56.90%에서 60.53% 사이입니다.



### Content-aware Tile Generation using Exterior Boundary Inpainting (https://arxiv.org/abs/2409.14184)
- **What's New**: 이 논문은 타일 이미지를 생성하기 위한 새로운 학습 기반 방법을 제안합니다. 이 방법은 간단한 자기 타일링(self-tiling)을 넘어, 서로 타일링 가능한 이미지 세트를 지원하여 높은 다양성을 보여줍니다.

- **Technical Details**: 제안된 방법은 자연 이미지와 텍스처에 대한 사전 지식을 활용하여 'diffusion models'를 사용하여 타일 생성을 안내합니다. 외부 경계 조건을 설계하고 선택하는 과정을 통해 타일 생성 과정을 'inpainting' 문제로 재구성하여, 기존의 'diffusion-based inpainting models'를 직접 사용하게 됩니다.

- **Performance Highlights**: 이 콘텐츠 인식 타일 생성 방법은 Wang 타일과 같은 다양한 타일링 방식에서 유연성과 효율성을 보여주며, 특히 새로운 Dual Wang 타일 형식을 소개하여 기존 Wang 타일 변종보다 더 나은 텍스처 연속성과 다양성을 제공합니다.



### LFP: Efficient and Accurate End-to-End Lane-Level Planning via Camera-LiDAR Fusion (https://arxiv.org/abs/2409.14170)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 자율주행에 있어 LiDAR(라이다)와 카메라 데이터를 효과적으로 융합하기 위한 새로운 방법론, Lane-level camera-LiDAR Fusion Planning (LFP)을 제안합니다. 이 방법은 겹치는 정보를 줄이고 운전 관련 요소에 집중하여 LiDAR의 핵심 정보를 유지하면서도 효율성을 높이는 데 초점을 맞추고 있습니다.

- **Technical Details**: LFP는 카메라와 LiDAR 간의 상호작용을 강화하여 Lane level(차선 수준)에서 데이터를 융합합니다. 이 방법은 크게 세 가지 모듈로 구성됩니다: 1) image-guided coarse lane prior generation 모듈, 2) Lane-aware LiDAR feature extraction 모듈, 3) lane-level cross-modal query integration and feature enhancement 모듈입니다. 이러한 모듈들은 각각 효율성과 성능 향상을 위해 설계되었습니다.

- **Performance Highlights**: Carla benchmark에서의 실험 결과, LFP는 드라이빙 점수와 위반 점수 측면에서 기존 방식보다 최대 15%와 14%의 성능 향상을 보였으며, 19.27 FPS의 높은 프레임 속도를 유지하였습니다. 이러한 성과는 LiDAR와 카메라의 상호작용을 통한 주요 기능 추출과 융합의 결과입니다.



### PromptTA: Prompt-driven Text Adapter for Source-free Domain Generalization (https://arxiv.org/abs/2409.14163)
- **What's New**: 이 논문에서는 소스 도메인 데이터에 접근하지 않고도 미지의 타겟 도메인에 적응할 수 있도록 설계된 Prompt-Driven Text Adapter (PromptTA) 방법을 제안합니다. 이는 스타일 특징의 분포를 더 잘 포착하고 도메인 지식의 충분한 범위를 보장하기 위해 재샘플링을 활용합니다.

- **Technical Details**: PromptTA는 다양한 스타일 특징에서 정보를 학습하는 텍스트 기반 어댑터를 도입하여 도메인 정보를 저장합니다. 스타일 특징의 재샘플링 기법을 통해 포괄적인 도메인 지식을 효과적으로 담을 수 있도록 합니다. 이 방법은 CLIP와 같은 비전-언어 모델의 정렬된 이미지-텍스트 표현을 활용하며, 스타일 벡터의 학습 가능한 집합을 통해 다양한 도메인 정보를 표현합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터세트에서 실시된 실험을 통해 PromptTA가 최신 기술 수준의 성능을 달성한 것으로 나타났습니다. 이를 통해 SFDG 분야에서의 발전에 기여하고 있음을 보여줍니다.



### MSSDA: Multi-Sub-Source Adaptation for Diabetic Foot Neuropathy Recognition (https://arxiv.org/abs/2409.14154)
- **What's New**: 이번 연구에서는 당뇨병성 족부 신경병증(Diabetic Foot Neuropathy, DFN) 인식을 위한 새로운 연속적인 plantar pressure 데이터셋을 수집하였고, 이를 기반으로 효과적인 도메인 적응(domain adaptation) 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 three-stage alignment framework을 통해 구성됩니다. 첫 단계에서는 contrastive learning을 사용하여 모든 샘플을 가능한 한 잘 분리하고, 두 번째 단계에서는 원본 소스 데이터셋을 convolutional feature statistics에 따라 K개의 하위 소스(domain)로 나누어, pseudo domain labels를 할당합니다. 세 번째 단계에서는 적절한 domain labels을 가진 소스 샘플을 선택하고, 여러 feature 공간에서 소스 및 타겟 도메인 간의 분포를 정렬합니다.

- **Performance Highlights**: 새로 제안한 DFN 인식 데이터셋과 기존 데이터셋에 대해 포괄적인 실험을 수행하였으며, 제안된 모델의 효과성을 검증하는 실험 결과를 얻었습니다.



### JVID: Joint Video-Image Diffusion for Visual-Quality and Temporal-Consistency in Video Generation (https://arxiv.org/abs/2409.14149)
- **What's New**: 새로운 Joint Video-Image Diffusion 모델(JVID)을 소개하며, 이미지 품질을 높이고 시간적 일관성을 보장하는 두 개의 모델(LIDM, LVDM)을 통합하여 고품질 비디오를 생성하는 방법을 제안합니다.

- **Technical Details**: LIDM(잠재 이미지 확산 모델)과 LVDM(잠재 비디오 확산 모델)을 결합하여 비디오 생성을 위한 역확산 과정(reversed diffusion process)을 사용할 수 있으며, 이는 각 단계에서 한 모델 또는 다른 모델로 샘플을 디노이즈(denoise)하는 방법입니다. 모델들은 동일한 퍼터베이션 프로세스와 노이즈 스케줄을 사용하여 호환성을 유지합니다.

- **Performance Highlights**: JVID는 비디오 생성에서 더 나은 시각적 품질과 시간적 일관성을 보여주며, 기존의 GAN 및 autoregressive 모델보다 우수한 성능을 달성합니다. 이 방법은 계산 비용과 샘플링 시간을 크게 줄여줍니다.



### A Feature Generator for Few-Shot Learning (https://arxiv.org/abs/2409.14141)
Comments:
          17 pages, Accepted to ACCV 2024

- **What's New**: 이 논문에서는 Few-shot learning (FSL) 분야에서 feature generator를 활용하여 클래스 수준의 텍스트 설명으로부터 시각적 특징을 생성함으로써 임베딩 프로세스를 향상시키는 새로운 접근 방식을 제안합니다. 이는 지원 클래스의 특징 표현을 강화하여 정확한 임베딩을 보장하기 위한 것입니다.

- **Technical Details**: 제안된 접근 방식은 class-level semantic features를 활용하여 synthetic visual features를 생성하는 것입니다. 이를 통해 n𝑛n-shot learning 시나리오를 2⁢n2𝑛2n2로 변환할 수 있으며, 각 특징 생성 단계에서 classifier loss, discriminator loss, cosine distance loss를 포함하는 복합 손실 함수를 최소화하여 생성된 특징의 품질을 높입니다.

- **Performance Highlights**: 실험 결과, miniImageNet 및 tieredImageNet 데이터셋을 사용한 결과, baseline 모델에 비해 1-shot 접근에서 약 10%, 5-shot 접근에서 약 5%의 정확도 향상을 확인하였습니다. 이 방법은 기존의 몇몇 모델에 모듈형태로 추가하여 성능을 향상시킬 수 있습니다.



### Present and Future Generalization of Synthetic Image Detectors (https://arxiv.org/abs/2409.14128)
Comments:
          16 pages, 6 figures

- **What's New**: 본 연구는 이미지 생성 모델의 발전에 직면하여 합성 이미지 감지기(Detector)의 일반화 능력을 고찰합니다. 실험 결과, 현재 평가된 모든 감지기는 보편적이지 않지만, 앙상블 방법이 보편적일 수 있다는 가능성을 제시합니다. 특히 야외에서 수집된 데이터에 대한 실험이 대규모 데이터셋에서 정의된 작업보다 더 도전적임을 보여주며, 생성기와 감지기 간의 균형 효과를 관찰합니다.

- **Technical Details**: 이 연구는 합성 이미지 감지를 위한 강력한 감지기 구축을 위한 다양한 훈련 조건의 영향을 분석합니다. ResNet 구조(ResNet-18)를 고정하여 실험을 수행하며, 다양한 합성 이미지 데이터셋과 AI 이미지 생성기를 사용합니다. 또한, 이미지 패치 기반 접근 방식을 채택하여 감지 성능을 향상시키기 위한 전략을 사용합니다.

- **Performance Highlights**: 현재 감지기는 단독으로 사용되었을 때 합성 내용 탐지에 불충분함을 나타냅니다. 감지기 앙상블 사용 및 생성기 특정 감지기 훈련이 권장되며, 후자의 접근법이 다른 데이터 소스에 대해 일반화 가능성을 보여줍니다. 두 가지 새로운 데이터셋과 실험에 사용된 소프트웨어 라이브러리가 공개되었습니다.



### Local Patterns Generalize Better for Novel Anomalies (https://arxiv.org/abs/2409.14109)
- **What's New**: 이 논문은 Video Anomaly Detection (VAD)에서 새로운 액션이나 이벤트를 식별하는 데 있어, 기존의 기술들이 글로벌 패턴에만 집중하는 것과 달리, 공간적 지역 패턴(spatial local patterns)을 식별하여 새로운 샘플에 일반화할 수 있는 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 주요 부분으로 나뉘어 있습니다. 첫 번째는 Image-Text Alignment Module (ITAM)을 통해 이미지와 텍스트의 대조 학습을 이용하여 지역 패턴을 추출하는 공간적 부분입니다. 두 번째는 State Machine Module (SMM)을 통해 지역 패턴의 동적 변화를 모델링하고, 이들의 시간적 변화를 동작 구성 요소(motion components)로 분해하는 시간적 부분입니다. 다양한 동적 변화를 표현하기 위해 고정된 동작 구성 요소의 가중치 합계를 사용합니다.

- **Performance Highlights**: 이 논문에서 제안한 방법은 인기 있는 벤치마크 데이터셋에서 기존의 최신 기술들에 비해 우수한 성능을 달성하였음을 입증하는 광범위한 실험 결과를 보이고 있습니다.



### ExFMan: Rendering 3D Dynamic Humans with Hybrid Monocular Blurry Frames and Events (https://arxiv.org/abs/2409.14103)
- **What's New**: 최근 연구에서 ExFMan이란 새로운 신경 렌더링 프레임워크가 소개되었습니다. 이 프레임워크는 빠른 동작 중에도 고품질 인체 렌더링이 가능하다는 가능성을 보여줍니다. 특히, 모노큘러 비디오에서 발생하는 움직임 블러를 효과적으로 처리하는 방법을 제공합니다.

- **Technical Details**: ExFMan은 하이브리드 프레임 기반 RGB와 바이오 영감을 받은 이벤트 카메라를 활용하여 성능을 극대화하는 네트워크입니다. 이 프레임워크는 인체의 속도 정보를 반영하여 블러가 발생하는 신체 부위를 식별하고, 속도 인식 포토메트릭 손실 및 속도 상대 이벤트 손실 이라는 두 가지 새로운 손실 함수를 도입하여 이미지를 최적화합니다.

- **Performance Highlights**: 광범위한 합성 및 실제 데이터 세트에서 ExFMan은 더 선명하고 높은 품질의 인체 재구성을 가능하게 하였으며, 기존 방법들보다 민감하고 고도의 품질을 유지하는 성능을 보여주었습니다.



### PoseAugment: Generative Human Pose Data Augmentation with Physical Plausibility for IMU-based Motion Captur (https://arxiv.org/abs/2409.14101)
Comments:
          Accepted to ECCV 2024. Code: this https URL

- **What's New**: 이 논문에서는 IMU 기반 인간 동작 캡처의 데이터 부족 문제를 해결하기 위해 새로운 데이터 증강 기법인 PoseAugment를 제시합니다. 이는 VAE 기반의 포즈 생성 및 물리적 최적화를 포함하는 혁신적인 파이프라인입니다.

- **Technical Details**: PoseAugment는 주어진 포즈 시퀀스를 입력받아 VAE(Variational Autoencoder) 모듈을 통해 무한한 고충실도(high fidelity) 및 다양성을 가진 포즈를 생성합니다. 생성된 포즈는 물리적 제약을 만족하도록 최적화되며, 최소한의 운동 제약으로 동작을 제한합니다. 마지막으로 고품질 IMU 데이터를 합성하여 동작 캡처 모델 훈련에 활용합니다.

- **Performance Highlights**: 실험 결과 PoseAugment는 기존 데이터 증강 및 포즈 생성 방법들보다 동작 캡처 정확도에서 우수한 성능을 나타내어 IMU 기반 동작 캡처 및 관련 작업의 데이터 수집 부담을 줄일 수 있는 강력한 잠재력을 보여주었습니다.



### Foundation Models for Amodal Video Instance Segmentation in Automated Driving (https://arxiv.org/abs/2409.14095)
Comments:
          accepted at ECCV VCAD Workshop 2024

- **What's New**: 이번 연구에서는 자동 운전을 위한 amodal 비디오 인스턴스 분할(video instance segmentation, VIS) 방법을 제안합니다. 기존의 연구들은 완전히 레이블이 된 비디오 데이터를 기반으로 amodal VIS를 수행했으며, 이는 데이터 수집이 어렵고 비용이 많이 들었습니다. 새로운 방법론으로, Segment Anything Model (SAM)을 활용하여 amodal 인스턴스 분할 작업에 맞춰 조정하였습니다.

- **Technical Details**: S-AModal 방법은 SAM을 미세 조정하여 amodal 인스턴스 분할을 수행합니다. 초기 비디오 인스턴스 분할에서 샘플링된 포인트를 SAM에 프롬프트로 제공하고, 포인트 메모리를 통해 관찰된 인스턴스의 포인트를 저장합니다. 예측되지 않은 인스턴스를 추적하기 위해 포인트 트래킹 방법을 사용하여 현재 프레임으로 이동시키며, 이에 따라 amodal 인스턴스 마스크를 조정합니다.

- **Performance Highlights**: S-AModal 방법은 amodal 비디오 인스턴스 분할에서 이전 기술보다 높은 성능을 보여줍니다. 특히, amodal 비디오 기반 레이블에 의존하지 않고도 우수한 결과를 달성하였으며, 이미지 수준의 amodal 분할에서도 최상위 성능을 기록하였습니다.



### BRep Boundary and Junction Detection for CAD Reverse Engineering (https://arxiv.org/abs/2409.14087)
Comments:
          6 pages, 5 figures

- **What's New**: 이번 논문은 3D 스캔을 기반으로 한 CAD 모델링 과정에서의 경계 표현(BRep) 탐지 네트워크인 BRepDetNet을 제안합니다. 이는 기존의 3D 스캔과 CAD 모델 간의 변환 문제를 해결하는 데 중점을 두었습니다.

- **Technical Details**: BRepDetNet은 CC3D 및 ABC 데이터셋의 3D 스캔에서 BRep 경계와 접합점을 탐지하는 데 최적화된 모델입니다. 50,000 개 및 45,000 개의 스캔을 주석 처리하고, non-maximal suppression(NMS) 기법을 훈련 과정에 통합하여 정확한 경계 및 접합점 탐지를 목표로 합니다. 이 모델은 Scan-to-BRep 문제를 해결하는 데 중요한 역할을 하며 기본 CAD 모델링 기능에 직접적으로 연결됩니다.

- **Performance Highlights**: BRepDetNet은 NMS-Loss를 사용하여 기존 방법들에 비해 향상된 경계 및 접합점 탐지 성능을 보여주며, 실험 결과에서 인상적인 성능을 입증하였습니다. 특히, 정확하고 효율적인 CAD 모델링에 기여할 수 있는 가능성을 열어줍니다.



### SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information (https://arxiv.org/abs/2409.14083)
Comments:
          19 pages, 9 tables, 11 figures

- **What's New**: 이번 연구에서는 대형 비전-언어 모델(LVLMs)의 Retrieval-Augmented Generation(RAG) 기능을 효과적으로 활용하기 위한 자기 개선(self-refinement) 프레임워크를 제안합니다. 기존 LVLMs가 신뢰성 없는 정보에 민감하게 반응하는 문제를 해결하고, 선택적으로 검색된 정보를 활용하도록 학습합니다.

- **Technical Details**: 제안된 프레임워크는 LVLMs가 잘못 답변한 시각적 질문에 대해 긍정적인 참조(positive references)와 부정적인 참조(negative references)를 이용하여 모델을 미세 조정합니다. 이 과정에서 이미지-캡션 이미지 쌍을 활용하며, RAG 지침 데이터셋을 구축하여 모델이 RAG 작업에서 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: 세 가지 작업(VQA, 이미지 캡셔닝, 이미지 분류)에 대해 7개 데이터셋을 활용한 실험에서, 제안된 방법은 LVLMs의 멀티모달 참조 활용 능력을 크게 향상시키며, 불관련한 정보에 대한 강건성을 개선하는 데 중요한 효과를 보였습니다.



### Dynamic 2D Gaussians: Geometrically accurate radiance fields for dynamic objects (https://arxiv.org/abs/2409.14072)
- **What's New**: 본 논문에서는 Dynamic 2D Gaussians (D-2DGS)라는 새로운 표현 방식을 제안하여 스파스 이미지 입력으로부터 정확한 메쉬를 재구성할 수 있는 방법을 제시합니다. 이 방법은 2D Gaussians의 기본 기하학적 표현을 채택하고 스파스 제어 점을 사용하여 2D Gaussian의 변형을 캡처합니다.

- **Technical Details**: D-2DGS는 스파스 컨트롤 포인트를 활용하여 2D Gaussians의 변형을 안내하고, 렌더링된 고품질 이미지에서 객체 마스크를 추출하여 깊이 맵을 마스킹합니다. 그리고 이 과정을 통해 정확한 동적 메쉬 시퀀스를 추출합니다.

- **Performance Highlights**: 실험 결과, D-2DGS는 스파스 입력으로부터 높은 품질의 메쉬를 재구성하는 데 있어 최신 기술(State-of-the-Art)을 달성하였으며, 이는 다른 고급 표현 방식들과 비교했을 때 두드러진 결과를 보여줍니다.



### SplatLoc: 3D Gaussian Splatting-based Visual Localization for Augmented Reality (https://arxiv.org/abs/2409.14067)
- **What's New**: 본 논문에서는 효율적인 3D Gaussian primitive 기반의 시각적 로컬라이제이션(viz localization) 방법을 제안합니다. 이 방법은 적은 파라미터로 고품질 렌더링을 가능하게 하며, 2D-3D 대응을 정확하게 보장하기 위해 블론 인프라로부터 독립적인 3D 장면 특정 디스크립터 디코더를 개발하였습니다.

- **Technical Details**: 제안된 방법은 3D Gaussian primitive를 시각적 로컬라이제이션을 위해 사용합니다. 이 시스템은 특히 Gaussian primitive의 정규화, salient(중요한) 3D 랜드마크 선택 알고리즘 및 고차원 특성의 압축을 통해 성능을 향상시킵니다. 또한, 2D-3D 대응을 위한 포즈 예측을 정확히 수행하도록 지연 설명 및 최적화 목표를 설정합니다.

- **Performance Highlights**: 두 개의 널리 사용되는 데이터셋에 대한 광범위한 실험 결과, 제안된 방법은 최신 암묵 기반 시각적 로컬라이제이션 접근법에 비해 우수하거나 동등한 성능의 렌더링 및 로컬라이제이션을 달성했습니다.



### Soft Segmented Randomization: Enhancing Domain Generalization in SAR-ATR for Synthetic-to-Measured (https://arxiv.org/abs/2409.14060)
Comments:
          19 pages, 13 figures

- **What's New**: 본 연구에서는 소프트 세그먼트 랜덤화(soft segmented randomization, SSR) 프레임워크를 제안하여 SAR(합성 개구 레이더) 자동 목표 인식 모형의 도메인 불일치를 줄이고 일반화 능력을 향상시키고자 하였습니다. 이는 합성 데이터의 통계적 특성을 실제 데이터와 더 긴밀하게 정렬시키기 위한 높은 확률 밀도 함수를 기반으로 하고 있습니다.

- **Technical Details**: SSR 프레임워크는 Gaussian mixture model(GMM)을 사용하여 타겟과 클러터 영역을 부드럽게 세분화하고, 각 영역에 대해 랜덤화된 변화를 도입하여 클러터의 반사도 및 스페클 노이즈의 차이를 완화합니다. SSR은 클러터 영역에서 평균 및 분산 값을 조정하여 반사도와 노이즈를 변화시키며, 타겟 영역에서는 도메인에 구애받지 않는 속성을 최대한 보존합니다.

- **Performance Highlights**: 실험 결과, SSR 프레임워크는 측정된 SAR 데이터에서 모델 성능을 유의미하게 향상시키며, 제한적 또는 측정된 데이터가 없는 시나리오에서의 강력한 자동 목표 인식에 대한 가능성을 제시합니다.



### BrainDreamer: Reasoning-Coherent and Controllable Image Generation from EEG Brain Signals via Language Guidanc (https://arxiv.org/abs/2409.14021)
- **What's New**: 신경과학적 뇌 신호를 기반으로 언어를 통해 인간의 사고를 모방하는 BrainDreamer라는 새로운 생성 프레임워크를 소개합니다. 이는 비침습적인 EEG 데이터로부터 고품질 이미지를 생성하는 능력을 가지고 있습니다.

- **Technical Details**: BrainDreamer는 두 가지 주요 학습 단계를 포함합니다: 1) modality alignment와 2) 이미지 생성. 첫 번째 단계에서는 새로운 마스크 기반의 세 가지 대비 학습 전략을 사용하여 EEG, 텍스트, 이미지 임베딩을 효과적으로 정렬하고 통합된 표현을 학습합니다. 두 번째 단계에서는 스테이블 디퓨전 모델에 EEG 임베딩을 주입하여 고품질 이미지를 생성합니다.

- **Performance Highlights**: 이 방법은 기존 기술보다 이미지 생성의 품질과 정량적인 성능에서 크게 향상된 결과를 보여 줍니다. 실험 결과 BrainDreamer는 더 소음이 제거되고 정밀한 EEG-이미지 매핑을 달성하여 기존 방법들을 초월하는 성능을 나타냅니다.



### MOSE: Monocular Semantic Reconstruction Using NeRF-Lifted Noisy Priors (https://arxiv.org/abs/2409.14019)
Comments:
          8 pages, 10 figures

- **What's New**: 본 논문에서는 MOSE라는 개념을 제안하여, 단일 이미지만을 사용하여 3D 메쉬를 재구성하는 데 있어 고품질의 기하학과 세밀한 의미 레이블링을 동시에 달성하고자 합니다. 이는 불완전한 2D 우선순위를 기반으로 하여 이루어집니다.

- **Technical Details**: MOSE는 임펄스(impulse) 신경 네트워크를 사용하여 2D 이미지와 노이즈가 포함된 2D 장면 우선 순위를 입력으로 받으며, 이를 기반으로 3D 의미 맵을 재구성합니다. 이 연구에서는 클래스 무관 이미지 세그먼트(masks)를 활용하여 훈련 중 지역 일관성을 촉진하고, 텍스처가 없는 영역에서 더 나은 기하학적 품질을 위해 매끄러움 정규화를 적용합니다.

- **Performance Highlights**: ScanNet 데이터 세트를 통한 실험 결과, MOSE는 3D 의미 분할, 2D 의미 분할 및 3D 표면 재구성 작업에서 모든 메트릭에서 이전 기법들보다 뛰어난 성능을 보여주었습니다.



### Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors (https://arxiv.org/abs/2409.14011)
- **What's New**: 본 연구는 비선형 시야(Non-Line-of-Sight, NLOS) 이미징을 위한 새로운 학습 기반 접근 방식을 제안합니다. 이 방식은 기존의 경험적 물리 우선 (physical priors) 에 대한 의존성을 극복하여 일반화 능력을 향상시킵니다.

- **Technical Details**: 주요 기술 설계는 Learnable Path Compensation (LPC)과 Adaptive Phasor Field (APF)입니다. LPC는 장면 내의 다양한 물체에 맞춰 조정된 경로 보상 계수를 적용하여 특히 먼 지역에서 빛의 파동 감쇠를 효과적으로 줄입니다. APF는 조명 함수의 가우시안 윈도우에 대한 정밀한 표준 편차를 학습하며, 순간 측정의 관련 스펙트럼 밴드를 동적으로 선택합니다.

- **Performance Highlights**: 이 방법은 합성 데이터로 훈련되었고, 다양한 저 SNR(real-world datasets) 데이터셋에서도 원활하게 일반화할 수 있는 능력을 입증했습니다. 빠른 데이터 수집 시간과 낮은 SNR 조건에서도 일관되게 경쟁자들을 초월하는 성능을 보였습니다.



### Multiple-Exit Tuning: Towards Inference-Efficient Adaptation for Vision Transformer (https://arxiv.org/abs/2409.13999)
Comments:
          13 pages,13 figures,6 tables

- **What's New**: 본 논문에서는 parameter-efficient transfer learning (PETL) 접근법에서 발생하는 비효율적인 추론을 개선하기 위한 새로운 방법, multiple-exit tuning (MET)을 소개합니다. MET는 사전 학습된 ViT(backbone)에 여러 개의 exit를 통합하여 쉽게 분류될 수 있는 샘플이 조기 exit을 통해 빠르게 처리될 수 있도록 합니다.

- **Technical Details**: MET 방법은 exit-specific adapters (E-adapters)와 그래프 정규화(graph regularization)를 포함합니다. E-adapters는 서로 다른 exit을 위한 적합한 표현을 추출하도록 설계되었으며, 모든 E-adapters는 효율적인 저장을 위해 동일한 down-projection과 up-projection 행렬을 공유합니다. 또한, 그래프 정규화를 사용하여 초기 exit에서의 데이터 포인트 간의 intra-class compactness 및 inter-class separability를 향상시킵니다.

- **Performance Highlights**: 28개의 다운스트림(task) 과제를 통해 MET는 최신 기술들과 비교했을 때 명확한 정확성 및 추론 효율성 우위를 보였습니다. 이 방법은 기존 방법에 비해 계산 비용을 절감하고, 쉽게 인식되는 샘플을 효과적으로 처리하여 전체적인 추론 과정을 가속화하는 데 기여합니다.



### GAInS: Gradient Anomaly-aware Biomedical Instance Segmentation (https://arxiv.org/abs/2409.13988)
Comments:
          Accepted by BIBM2024

- **What's New**: 본 연구에서는 Gradient Anomaly-aware Biomedical Instance Segmentation 접근법(GAInS)을 제안하여, 서로 겹치거나 닿은 생물학적 객체들을 보다 정교하게 식별하는 방법을 다룹니다. 기존 방식들과 달리, GAInS는 인스턴스의 gradient 정보를 활용하여 차별화된 세분화(Segmentation)를 가능케 합니다.

- **Technical Details**: GAInS는 두 가지 주요 모듈인 Gradient Anomaly Mapping Module (GAMM)과 Adaptive Local Refinement Module (ALRM)을 포함합니다. GAMM은 인스턴스의 radial field를 통해 gradient anomaly 맵을 생성하며, ALRM은 gradient anomaly-aware loss function을 사용하여 인스턴스 간의 경계와 지역을 정교하게 다듬습니다.

- **Performance Highlights**: GAInS는 세 가지 생물학적 시나리오에서 기존의 SOTA(instance segmentation) 방법들과 비교했을 때 뛰어난 성능을 보였습니다. 구체적으로, 제안된 방법은 교차(touching), 겹침(overlapping), 및 경계 구분에서 개선된 성과를 입증했습니다.



### Holistic and Historical Instance Comparison for Cervical Cell Detection (https://arxiv.org/abs/2409.13987)
Comments:
          Accepted by BIBM2024

- **What's New**: 본 논문에서는 자궁 경부 세포 탐지를 위한 포괄적이고 역사적인 사례 비교 접근법을 제안합니다. 이는 세포 분류의 모호성을 해소하고 미세한 클래스 불균형 문제를 해결하기 위한 방법입니다.

- **Technical Details**: 제안된 방법은 RoI(Region of Interest) 수준 및 클래스 수준의 세포 구별을 강제하는 전체적인 사례 비교 방식을 개발합니다. 이 방식은 각 RoI 후보의 특징을 추출한 후, 현재 RoI를 금전적 참조 샘플과 대조하여 학습하게 됩니다. 역사적 사례 비교는 과거의 임베딩과 현재 임베딩을 비교하여 더 나은 세포 인스턴스 구별을 도모합니다.

- **Performance Highlights**: 42,592 및 114,513개의 자궁 경부 세포의 대규모 데이터셋에서 실험을 진행한 결과, 제안된 방법이 기존 SOTA(State Of The Art) 방법보다 우수한 성능을 보임을 입증했습니다.



### Cycle-Consistency Uncertainty Estimation for Visual Prompting based One-Shot Defect Segmentation (https://arxiv.org/abs/2409.13984)
Comments:
          ECCV 2024 VISION workshop Most Innovative Prize

- **What's New**: 본 연구에서는 산업 결함 검출을 위해 새로운 비디오 프롬프트 기법을 도입하였으며, Cycle-consistency를 통한 불확실성 추정 방법을 제안하여 시각적 프롬프트 프로세스의 신뢰성을 높였습니다.

- **Technical Details**: DINOv 기반의 시각적 프롬프트 방법을 사용하여, 프롬프트 마스크와 원래 제공된 프롬프트 마스크 사이의 평균 IoU(Intersection over Union)를 측정하여 모델의 성능을 신뢰할 수 있는지 평가합니다.

- **Performance Highlights**: VISION24 원샷 산업 챌린지에서 0.9175의 높은 수율을 달성하며, 특히 기존의 복잡한 설계나 앙상블 기법을 사용하지 않고도 우수한 성능을 보였습니다.



### Enhanced Semantic Segmentation for Large-Scale and Imbalanced Point Clouds (https://arxiv.org/abs/2409.13983)
- **What's New**: 이번 연구에서는 대규모 및 샘플 불균형(point cloud) 데이터의 의미적 분할(semantic segmentation)을 위한 멀티라테랄 캐스케이딩 네트워크(MCNet)을 제안합니다.

- **Technical Details**: MCNet은 작은 객체의 빈도를 높이기 위해 의미적 가중 샘플링 모듈을 도입하며, 다수의 단계에서 복잡한 지역 특성을 학습하기 위해 멀티라테랄 캐스케이딩 주의 증강(MCAE) 모듈을 포함하고 있습니다. 또한 글로벌 및 지역 특성을 통합하여 유용한 특성 정보를 여러 규모에서 최적화하는 Point Cross Stage Partial (P-CSP) 모듈이 설계되었습니다. 마지막으로 이웃 투표 모듈을 통해 출력 레이어에서 결과를 통합합니다.

- **Performance Highlights**: 제안된 방법은 S3DIS, Toronto3D 및 SensatUrban의 세 가지 널리 인정된 벤치마크 데이터셋에서 각각 mIoU 점수 74.0%, 82.9%, 64.5%로 최신 기술들과 비교하였을 때 경쟁력 있는 성능 또는 우수한 성능을 보여주었습니다. 특히, 잦은 표본화(sample imbalance)로 인해 하위 샘플에 대한 인식 성능이 향상되었습니다.



### CUS3D :CLIP-based Unsupervised 3D Segmentation via Object-level Denois (https://arxiv.org/abs/2409.13982)
Comments:
          6 pages,3 figures

- **What's New**: 이 논문에서는 3D 데이터에서 주석 레이블 획득의 어려움을 해결하기 위해, 2D CLIP (Contrastive Language-Image Pretraining) 기반의 비지도 및 개방 어휘(unsupervised and open-vocabulary) 의미 분할을 활용하는 새로운 distillation learning (증류 학습) 프레임워크인 CUS3D를 제안합니다. 기존 연구에서 2D에서 3D로의 특징 투영 과정에서 발생하는 'noise'를 무시한 점을 반영하여, 객체 수준의 denoising projection 모듈을 설계하였습니다.

- **Technical Details**: 본 연구에서는 비지도 및 개방 어휘 의미 분할을 위해 두 가지 주요 모듈을 제안합니다. 첫 번째는 Object-level Denoising Projection (ODP) 모듈로, 2D 단계와 3D 단계에서 발생하는 'noise'를 필터링하여 보다 정확한 3D 특징을 획득합니다. 두 번째는 3D Multimodal Distillation Learning (MDL) 모듈로, 객체 중심 제약을 통해 2D 및 3D 의미 공간을 정렬합니다. 이를 통해 CLIP의 의미 공간과 3D 특징 공간 간의 보다 정밀한 정렬을 달성하고 있습니다.

- **Performance Highlights**: 실험 결과, 우리 모델은 비지도 및 개방 어휘 분할에서 우수한 성능을 보였습니다. 특히, 정확한 3D 특징을 활용하여 segmentation (분할) 작업의 향상된 결과를 얻었으며, 이는 다양한 시나리오에서 의미 분할 작업의 정확성을 높이는 데 기여합니다.



### Enhancing Advanced Visual Reasoning Ability of Large Language Models (https://arxiv.org/abs/2409.13980)
Comments:
          EMNLP 2024 Main

- **What's New**: 본 논문에서는 Complex Visual Reasoning Large Language Models (CVR-LLM)를 제안하여, Vision-Language Models (VLMs)와 Large Language Models (LLMs)의 장점을 통합하여 복잡한 시각적 추론을 수행하는 방법을 소개합니다.

- **Technical Details**: CVR-LLM은 LLM의 방대한 텍스트 지식을 활용하여 이미지-텍스트 쌍 없이도 정확한 예측을 가능하게 하며, 이미지에서 문맥 인식 이미지 설명(context-aware image descriptions, CaID)을 생성하기 위해 이터러티브 자기 정제 과정(iterative self-refinement loop)을 사용합니다. 이 과정에서 LLM의 피드백을 반영하여 복잡한 시각적 추론 과제에 단순화된 단일 모달 문제로 변환합니다. 또한, Chain-of-Comparison (CoC) 기술을 도입하여 예측 결과의 다양한 측면을 단계적으로 비교합니다.

- **Performance Highlights**: CVR-LLM은 WinoGAViL, Winoground, Whoops, VCR, NYCCC와 같은 복잡한 시각적 추론 작업에서 SOTA(performance state-of-the-art) 성능을 달성하며, 기존의 접근법과 비교하여 우수성을 입증합니다.



### FracGM: A Fast Fractional Programming Technique for Geman-McClure Robust Estimator (https://arxiv.org/abs/2409.13978)
Comments:
          8 pages, 6 figures

- **What's New**: 본 연구에서는 Geman-McClure(지멘-맥클루어) 강건 추정의 새로운 알고리즘인 FracGM을 제안하며, 이는 분수 프로그래밍(fractional programming) 기술을 활용하여 기존의 비볼록(non-convex) 문제를 볼록(convex) 쌍대 문제와 선형 방정식 시스템으로 변환합니다.

- **Technical Details**: FracGM은 교차 최적화 패턴을 따르며 주어진 조건 하에 전역 최적성(global optimality)을 보장할 수 있습니다. 기존의 Geman-McClure 강건 함수가 가지는 수렴성과 전역 최적성을 이론적으로 분석했습니다. 또한, 비-볼록성이 문제인 3-D 회전 문제에서의 강건성을 입증하고, 최적화 과정에서의 효율성을 강조합니다.

- **Performance Highlights**: FracGM은 인공 데이터셋에서 회전 및 변환 오차가 각각 53%와 88% 더 적게 증가하는 결과를 나타냈습니다. 실제 환경에서도 18개의 결과 중 13개에서 가장 우수한 성능을 보였으며, 연산 시간도 19.43% 개선되었습니다.



### Improving 3D Semi-supervised Learning by Effectively Utilizing All Unlabelled Data (https://arxiv.org/abs/2409.13977)
Comments:
          Accepted at the European Conference on Computer Vision, ECCV 2024

- **What's New**: 이번 연구에서는 모든 비지도 샘플을 효과적으로 활용할 수 있는 새로운 반지도 학습(Semi-Supervised Learning, SSL) 기반의 3D 분류 프레임워크인 AllMatch를 제안합니다. AllMatch는 기존 SSL 접근 방식이 비효율적으로 샘플을 사용해 한정된 성능에 그쳤음을 지적하고, 이를 개선하기 위해 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: AllMatch는 (1) 상대적으로 높은 신뢰도를 가진 비지도 샘플에 대해 비교적 강력한 변환을 적용하는 적응형 하드 증강 모듈, (2) 학습할 필요 없었던 것을 학습함으로써 비지도 데이터를 더 효과적으로 활용하는 역학습 모듈, (3) 감독 및 비감독 환경 모두에서 모든 샘플로부터 학습을 보장하는 대조 학습 모듈로 구성되어 있습니다. 이 프레임워크는 ModelNet40 및 ScanObjectNN 데이터셋에서 실험을 통해 성능 향상을 입증했습니다.

- **Performance Highlights**: AllMatch는 적은 양의 라벨 데이터(1%)로 기존 최첨단 모델(SOTA)을 초과하여 최대 11.2%의 성능 향상을 보였습니다. 단지 10%의 라벨 데이터만으로도 기존 방식의 거의 동일한 성능을 달성할 수 있음을 보여주었으며, 학습 과정에서 250 에폭을 소요하여 이전 SOTA 모델보다 더 효율적이라고 평가되었습니다.



### Detecting Inpainted Video with Frequency Domain Insights (https://arxiv.org/abs/2409.13976)
Comments:
          submit to ICASSP2025

- **What's New**: 이번 논문에서는 Frequency Domain Insights Network (FDIN)이라는 새로운 비디오 인페인팅 탐지 모델을 소개합니다. FDIN은 주파수 영역의 통찰력을 포함하여 탐지 정확성을 획기적으로 향상시키는 방법을 제안합니다.

- **Technical Details**: FDIN은 Adaptive Band Selective Response (ABSR) 모듈을 통해 다양한 인페인팅 기술에 특정한 주파수 특징을 구별합니다. 또한 Fast Fourier Convolution 기반의 Attention (FFCA) 모듈을 사용해 인페인팅된 영역에서 주기적인 아티팩트를 식별합니다. 이 구조는 3D ResBlock을 활용하여 시공간(spatiotemporal) 분석을 실현합니다.

- **Performance Highlights**: 실험 결과, FDIN은 여러 공개 데이터셋에서 기존의 방법들보다 우수한 성능을 발휘하며 비디오 인페인팅 탐지에 대한 새로운 기준을 세웠습니다.



### Monocular Event-Inertial Odometry with Adaptive decay-based Time Surface and Polarity-aware Tracking (https://arxiv.org/abs/2409.13971)
Comments:
          Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024

- **What's New**: 이번 논문에서는 폴라리티-인식 추적 기능이 통합된 적응형 감소 커널 기반의 타임 서페이스를 사용하는 단일 이벤트 관성 오도메트리를 제안합니다. 이를 통해 환경 텍스처의 표현력을 향상시키며, 폴라리티 변화로 인한 문제를 해결할 수 있습니다.

- **Technical Details**: 제안하는 시스템은 모노큘러 이벤트 카메라와 IMU 데이터의 융합을 통해 Multi-State Constraint Kalman Filter (MSCKF) 프레임워크를 기반으로 하여 정확한 자세 추정을 수행합니다. 적응형 감소 기반의 타임 서페이스는 이벤트 스트림의 동적 특성에 맞춰 텍스처 정보를 추출하고, 추가적인 폴라리티-반전 타임 서페이스를 이용하여 특징 추적의 안정성을 높입니다.

- **Performance Highlights**: 비교 분석 결과, 제안한 방법은 다양한 데이터셋에서 최신 기술들을 초월하는 경쟁력을 보였으며, 정확도 면에서도 시각-관성 오도메트리 및 이벤트-관성 오도메트리 방법들과 비교해 우수한 성능을 나타냈습니다.



### Deep learning for fast segmentation and critical dimension metrology & characterization enabling AR/VR design and fabrication (https://arxiv.org/abs/2409.13951)
- **What's New**: 이번 연구에서는 전자 현미경 이미지의 다양한 데이터셋을 사용하여 사전 학습된 Segment Anything Model (SAM)을 미세 조정했습니다.

- **Technical Details**: 우리는 ROI(Region of Interest) 추출의 정확성을 높이기 위해 저순위 적응(low-rank adaptation, LoRA) 등의 방법을 활용하고, 모델의 일반화 능력을 통해 제로샷 학습(zero-shot learning)을 지원합니다. 또한, 세그먼트된 ROI에서 중요 치수(Critical Dimensions, CDs)를 정확하게 추출하는 모델을 개발했습니다.

- **Performance Highlights**: 우리는 Surface Relief Gratings (SRGs)와 프레넬 렌즈의 단일 및 다중 클래스 모드에서 단면 이미지를 정확하게 이진 이미지로 추출하는 데 성공했습니다. 이 이진 이미지는 전환 지점을 식별하는 데 사용되어 관련 CDs의 추출을 돕습니다.



### TalkMosaic: Interactive PhotoMosaic with Multi-modal LLM Q&A Interactions (https://arxiv.org/abs/2409.13941)
Comments:
          6 pages, 5 figures

- **What's New**: 이 논문은 다양한 자동차 이미지를 사용하여 환경 보호 주제를 중심으로 새롭게 구성된 포토모자이크 이미지에서 동물 이미지(예: 새, 사자)를 생성하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: 우리는 'click and display'라는 간단한 작업을 통해 포토모자이크 이미지에서 타일 이미지와 원래의 자동차 이미지 간의 인터랙티브한 전환을 시연하는 방법을 제시하며، 사용자가 자동차 이미지를 TalkMosaic에 업로드한 후 관련 질문을 효율적으로 할 수 있게 돕습니다. 또한, 희소 주의(Sparse Attention)와 양자화(Quantization) 기술을 활용하여 다중 모달 LLM(대형 언어 모델)의 추론 속도를 높이는 방법을 심층적으로 분석합니다.

- **Performance Highlights**: 제안된 프로토타입은 PrFlashAttention(확률적 플래시 주의)와 SAQ(계단식 적응 양자화) 방법을 통해 효율성과 효과성을 입증하며, 사용자는 고환경 기준을 만족하는 타이어를 자동차 이미지 관련하여 어디서 구매할 수 있는지 등에 대한 질문에 대한 신속한 답변을 받을 수 있습니다.



### Data Pruning via Separability, Integrity, and Model Uncertainty-Aware Importance Sampling (https://arxiv.org/abs/2409.13915)
- **What's New**: 이 논문은 이미지 분류를 위한 데이터 프루닝(data pruning) 방법을 개선하기 위해 새로운 프루닝 메트릭(metric)과 중요도 샘플링(importance sampling) 기반의 절차를 도입합니다. 제안된 메트릭은 데이터의 분리 가능성(separability), 데이터 무결성(integrity), 모델 불확실성(model uncertainty)을 명시적으로 고려하고, 샘플링 방식이 프루닝 비율에 적응하며, 클래스 간 및 클래스 내 분리를 더욱 향상시킵니다.

- **Technical Details**: 이 연구는 새로운 데이터 프루닝 메트릭인 SIM (Separability, Integrity, Model uncertainty)을 도입하여 데이터 유용성의 여러 요소를 포착합니다. 또한, 중요도 샘플링 절차를 통해 흔히 사용되는 프루닝 메트릭과 결합하여 프루닝된 데이터의 효과를 향상시킬 수 있습니다. 네 개의 벤치마크 데이터셋을 통해 본 방법의 스케일링(scale), 교차 모델의 일반화능력(cross-model generalization), 그리고 프루닝 메트릭 계산에 소요되는 시간을 줄이는 성능을 입증하였습니다.

- **Performance Highlights**: 제안된 SIMS 배합 방법은 높은 프루닝 비율에서도 더 나은 스케일링과 모델 간 일반화를 보이며, 기존 방법보다 프루닝 메트릭 계산에 필요한 시간이 줄어드는 점이 실험을 통해 확인되었습니다.



### OneBEV: Using One Panoramic Image for Bird's-Eye-View Semantic Mapping (https://arxiv.org/abs/2409.13912)
Comments:
          Accepted by ACCV 2024. Project code at: this https URL

- **What's New**: 이 논문에서는 단일 팬오라마 이미지를 사용하여 Bird's-Eye-View (BEV) 시맨틱 매핑을 단순화하고 계산 복잡성을 줄이는 새로운 방법인 OneBEV를 소개합니다. 기존의 고전적인 BEV 방법과 달리, OneBEV는 복잡한 포즈 추정 없이도 BEV 기능으로 변환할 수 있는 Mamba View Transformation (MVT) 모듈을 사용하여 공간 왜곡 문제를 해결합니다.

- **Technical Details**: OneBEV는 360° 이미지를 단일 입력으로 요구하며, 세 가지 주요 구성 요소로 이루어져 있습니다: 특징 인코더, 뷰 변환 모듈, 시맨틱 디코더. MVT 모듈은 기존 주의 메커니즘을 사용하지 않고 특징을 BEV 공간으로 변환하여 효율적인 BEV 변환을 달성합니다. 또한, nuScenes-360과 DeepAccident-360 두 개의 데이터셋을 새롭게 도입하여 OneBEV 작업에 최적화된 환경을 제공합니다.

- **Performance Highlights**: OneBEV는 nuScenes-360과 DeepAccident-360 데이터셋에서 각각 51.1%와 36.1%의 mIoU를 달성하며, 기본 모델에 비해 1.8백만 개의 매개변수를 줄였습니다. 이는 동적 외부 환경에서의 시맨틱 장면 이해를 위한 실질적인 솔루션을 제공하며, 더 효율적이고 신뢰할 수 있는 자율 주행 시스템의 개발을 위한 중요한 진전을 이룹니다.



### Brain-Cognition Fingerprinting via Graph-GCCA with Contrastive Learning (https://arxiv.org/abs/2409.13887)
- **What's New**: 이번 연구에서는 뇌 기능과 인지 간의 관계를 정확하게 인코딩하기 위해 
 비지도 학습 모델인 Contrastive Learning-based Graph Generalized Canonical Correlation Analysis (CoGraCa)를 제안합니다. 이 모델은 Graph Attention Networks와 일반화된 Canonical Correlation Analysis를 사용하여 개인의 신경 및 인지 특성을 반영한 '브레인-인지 지문'을 생성합니다.

- **Technical Details**: CoGraCa는 개인 맞춤형 뇌-인지 지문을 만들기 위해 개인화된 대조 학습 기법과 다중 모드 대조 학습을 활용하여 각 개인의 독특한 연결 패턴과 동적 진화를 보존합니다. 각 방문 시 촬영된 resting-state functional MRI와 인지 데이터의 상관 관계를 인코딩하는데, GAT와 GCCA의 조합을 사용하여 뇌 기능 네트워크와 인지 데이터를 일치시킵니다.

- **Performance Highlights**: CoGraCa는 총 57명의 참가자로 구성된 데이터셋을 사용하여 검증되었으며, 생성된 '브레인-인지' 지문은 높은 개별 차별성을 보여주었습니다. 또한, 성별 및 연령 분류 작업에서 최신 단일 모드 및 CCA 기반 다중 모드 방법과 비교하여 더 높은 정확도를 달성함으로써 인지 데이터와 뇌 연결성을 통합하는 데 효과적임을 입증하였습니다.



### SSE: Multimodal Semantic Data Selection and Enrichment for Industrial-scale Data Assimilation (https://arxiv.org/abs/2409.13860)
- **What's New**: 최근 인공지능(AI) 모델의 발전으로 인해 데이터의 양이 기하급수적으로 증가하고 있습니다. 특히 자율주행차와 같은 산업 응용 프로그램에서는 데이터 양이 너무 많아 모델 성능이 포화 상태에 이르고 있습니다. 이에 따라, 본 논문에서는 가장 의미 있는 데이터 세트를 선택하고, 대규모 비주석 데이터 풀에서 의미 있는 새로운 데이터를 발견하여 데이터를 풍부하게 만드는 프레임워크를 제안합니다.

- **Technical Details**: 논문에서 제안하는 Semantic Selection and Enrichment (SSE) 프레임워크는 데이터 선택 및 풍부화 과정을 동시에 수행합니다. SSE는 MLLMs(다중 모드 언어 모델)를 활용하여 데이터 포인트에 대한 설명을 생성하고, 이를 통해 각 데이터의 의미를 파악하며, 의미적으로 다양한 데이터를 선택합니다. 선택 후, 비주석 데이터 풀에서 의미적으로 유의미한 새로운 데이터를 추가하여 데이터 세트를 확장합니다.

- **Performance Highlights**: SSE 프레임워크는 더 적은 양의 고품질 데이터로 모델 성능을 유지하며, 원래 데이터 세트 크기를 초과하지 않고도 모델 성능을 향상시키는 것으로 나타났습니다. 이를 통해 의미적 다양성이 최적의 데이터 선택 및 모델 성능에 필수적이라는 것을 보여주었습니다.



### Multi-Modality Conditioned Variational U-Net for Field-of-View Extension in Brain Diffusion MRI (https://arxiv.org/abs/2409.13846)
Comments:
          20 pages; 8 figures

- **What's New**: 본 연구에서는 diffusion magnetic resonance imaging (dMRI)의 불완전한 field-of-view (FOV) 문제를 해결하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 dMRI 스캔의 불완전한 부분을 보완하기 위해 획득된 FOV에서 학습된 확산 특성과 전체 뇌의 해부학적 구조를 통합합니다.

- **Technical Details**: 제안된 프레임워크는 두 개의 서로 다른 사이트에서 총 96명의 피험자를 대상으로 테스트되었으며, T1w와 dMRI 스캔 정보를 동등하게 처리하는 기준 임퓨테이션 방법과 비교되었습니다.

- **Performance Highlights**: 제안된 프레임워크는 각도 상관 계수(angular correlation coefficient)에서 p < 1E-5의 유의미한 개선을 보였고, 다운스트림 트랙토그래피 정확도는 Dice score에서 p < 0.01의 결과로 입증되었습니다. 이는 dMRI 스캔의 임퓨테이션 성능을 향상시켜 신경퇴행과 관련된 번들 분석 시 불확실성을 줄이는 데 기여함을 나타냅니다.



### ViTGuard: Attention-aware Detection against Adversarial Examples for Vision Transformer (https://arxiv.org/abs/2409.13828)
Comments:
          To appear in the Annual Computer Security Applications Conference (ACSAC) 2024

- **What's New**: 이 논문에서는 Vision Transformer (ViT) 모델을 위한 새로운 탐지 방법인 ViTGuard를 제안합니다. ViTGuard는 적대적인 공격으로부터 ViT 모델을 방어하기 위해 주의 맵(attention maps)과 분류 토큰(classification token) 표현을 활용합니다.

- **Technical Details**: ViTGuard는 Masked Autoencoder (MAE) 모델을 사용하여 무작위로 마스킹된 패치(patch)를 복구하고, 입력 이미지의 원본과 복원된 이미지 간의 주의 맵과 CLS 표현을 비교합니다. 이 과정에서 L2 거리(L2 distance)를 계산하여 입력이 정상인지 적대적인지를 판별합니다. 디텍터는 훈련 중 적대적 샘플을 사용하지 않아서 보이지 않는 공격에 대해서도 효과적입니다.

- **Performance Highlights**: ViTGuard는 세 가지 데이터셋에서 아홉 가지 공격을 대상으로 기존의 일곱 개 탐지 방법과 비교하였을 때, 기존 탐지기들에 비해 우수한 성능을 보였습니다. 또한, 탐지 회피 가능성을 고려하여, 적대적인 공격에 대한 강인함도 입증되었습니다.



### Intrinsic Single-Image HDR Reconstruction (https://arxiv.org/abs/2409.13803)
Comments:
          Accepted for ECCV 2024

- **What's New**: 이 논문은 일반 카메라의 낮은 동적 범위(LDR)가 자연 장면의 풍부한 대비를 포착하지 못하며, 이로 인해 색상과 세부정보가 손실된다는 문제를 다루고 있습니다. 단일 LDR 사진에서 장면의 고동적 범위(HDR)를 복원하는 새로운 접근법을 제시하고 있습니다.

- **Technical Details**: 논문에서는 HDR 복원 문제를 본질적인 영역(intrinsic domain)으로 리모델링하는 물리적으로 영감을 받은 방법을 제안합니다. 본질 모델은 각각의 네트워크를 훈련시켜 음영(domain)에서 동적 범위를 확장하고 알베도(domain)에서 손실된 색상 세부정보를 회복할 수 있도록 합니다. 문제를 두 개의 간단한 하위 작업으로 나누는 것이 성능을 향상시키는 데 기여한다고 합니다.

- **Performance Highlights**: 제시된 방법은 다양한 사진에서 성능이 향상되는 것을 보여주며, 데이터 기반 알고리즘이 보다 정확하고 해상도가 높은 결과를 생성하는 데 도움이 됩니다.



### OmniBench: Towards The Future of Universal Omni-Language Models (https://arxiv.org/abs/2409.15272)
- **What's New**: OmniBench는 여러 모달리티(visual, acoustic, textual) 간의 상호작용을 평가하고 모델의 이해 및 추론 능력을 측정하는 새로운 벤치마크입니다. 이 벤치마크는 모든 모달리티 간의 통합된 이해를 요구하여 기존의 한계를 극복하고 있습니다.

- **Technical Details**: 오미벤치(OmniBench)는 미리 훈련된 초대형 언어 모델(MLLMs)의 tri-modal(3중 모달리티) 처리 능력을 테스트하기 위한 포괄적인 도구입니다. OLMs(omni-language models)는 이러한 능력을 갖춘 모델로 정의됩니다. OLMs는 high-quality human annotations에 의존하여 정확한 응답을 제공하는 데 필요한 모든 모달리티의 통합된 이해를 요구합니다.

- **Performance Highlights**: 대부분의 OLMs는 tri-modal 상황에서 지시 수행 및 추론 능력에 한계를 보이는 것으로 나타났습니다. 기존의 MLLM들은 이미지 또는 오디오와 함께 제공되었을 때 명확한 지시를 따르기 어려운 경우가 많으며, 대체 텍스트 표현 사용 시에도 정확도가 50% 미만으로 낮은 성능을 기록했습니다.



### UDA-Bench: Revisiting Common Assumptions in Unsupervised Domain Adaptation Using a Standardized Framework (https://arxiv.org/abs/2409.15264)
Comments:
          ECCV 2024 Camera-ready version

- **What's New**: 이 논문에서는 현대의 비지도 도메인 적응(Unsupservised Domain Adaptation, UDA) 방법의 효과성을 결정짓는 다양한 요소에 대해 심도 있게 분석했습니다. 이를 위해 UDA-Bench라는 새로운 PyTorch 프레임워크를 개발하여 교육 및 평가를 표준화하고, 여러 UDA 방법 간의 공정한 비교를 가능하게 하였습니다.

- **Technical Details**: 주요 요소로는 백본 아키텍처(backbone architecture)의 선택, 비라벨 데이터(unlabeled data)의 양, 그리고 프리트레이닝 데이터(pre-training data)의 성질이 있습니다. UDA-Bench를 사용하여 여러 UDA 방법의 성능을 평가하며, 다양한 아키텍처와 비라벨 데이터 양이 성능에 미치는 영향을 분석했습니다. 또한 프리트레이닝이 이미지 분류 작업에서 효과적이라는 점을 입증했습니다.

- **Performance Highlights**: 백본 아키텍처의 발전이 UDA 방법의 이점을 감소시킨다는 사실이 발견되었습니다. 비라벨 데이터의 양을 최대 75% 줄여도 목표 정확도는 단 1%만 감소했습니다. 프리트레이닝 데이터 역시 다운스트림 적응에 중요한 영향을 미치며, supervised 환경에서는 유사한 데이터로 프리트레이닝 하는 것이 효과적입니다. 이러한 결과들은 UDA 연구의 이론과 실제 사이의 불일치를 드러냅니다.



### ZeroSCD: Zero-Shot Street Scene Change Detection (https://arxiv.org/abs/2409.15255)
- **What's New**: ZeroSCD는 데이터 주석 없이 장면 변화 감지를 가능하게 하는 새로운 제로샷(change detection) 프레임워크입니다. 이 방법은 기존 모델의 기능을 활용하여 구조적 변화를 정밀하게 탐지합니다.

- **Technical Details**: ZeroSCD는 Visual Place Recognition(VPR) 모델인 PlaceFormer를 활용하여 스타일 변화에 향상된 저항력을 가지고 있습니다. 또한, Semantic Segmentation 모델의 결과와 결합하여 변화를 정확하게 구분합니다.

- **Performance Highlights**: ZeroSCD는 여러 변화 감지 데이터셋에서 상태-of-the-art 성능을 달성했으며, 이들 데이터셋에 대해 훈련을 요구하지 않아 다양한 시나리오에서의 효과성과 적응성을 입증했습니다.



### Investigating Robot Dogs for Construction Monitoring: A Comparative Analysis of Specifications and On-site Requirements (https://arxiv.org/abs/2409.15253)
Comments:
          8 pages, 3 figures, 2 Tables, Forum Bauinformatik

- **What's New**: 본 논문에서는 건설 현장에서 현재 사용 가능한 로봇 개들의 유용성을 조사하고, 자동화된 지원을 통해 수작업 노력을 줄일 수 있는 가능성을 보여줍니다. 특히 기술 발전에 따라 로봇 개들이 복잡한 건설 환경을 모니터링하는데 중요한 자산이 될 수 있음을 강조합니다.

- **Technical Details**: 로봇 개는 4개의 기계다리를 가진 네 발 동물의 구조와 움직임을 기반으로 설계되었습니다. 이 논문은 유럽 시장에서의 다양한 다리가 있는 로봇들을 분석하고, LiDAR, 카메라 및 관성 측정 장치(IMU) 센서를 통합한 독립형 매핑 시스템을 개발했습니다. 이 시스템은 포터블하나 로봇과 함께 사용할 수 있습니다.

- **Performance Highlights**: Go1 로봇 개는 건설 현장에서 12,500 제곱미터의 면적을 커버하여 3D 포인트 클라우드를 생성하였습니다. 로봇 개는 고층 및 복잡한 지형에서 어려움을 겪을 수 있지만, BIM 모델을 활용해 자율 항법이 가능하며, 3D 데이터는 정확한 좌표 시스템으로 정렬, 교정 및 분석될 수 있습니다.



### Semantic Inference-Based Deep Learning and Modeling for Earth Observation: Cognitive Semantic Augmentation Satellite Networks (https://arxiv.org/abs/2409.15246)
Comments:
          18 pages, 10 figures, magazine

- **What's New**: 본 논문은 Earth Observation (EO) 위성 네트워크에서의 의미 기반 통신(semantic communication) 프레임워크를 제안하며, 대량의 EO 데이터와 의미 있는 데이터를 융합하여 전송 효율성과 시스템 성능을 개선하는 방법을 소개합니다.

- **Technical Details**: 제안된 시스템은 Discrete-Task-Oriented Source-Channel Coding (DT-JSCC)와 Semantic Data Augmentation (SA)을 활용하여, 관련 정보를 강조하며 통신 오버헤드를 최소화합니다. 또한, Cognitive Semantic Augmentation (CSA)를 도입하여 위성이 의미 있는 정보를 처리하고 전송할 수 있도록 하여 변화하는 환경과 응용 요구에 잘 적응할 수 있도록 합니다.

- **Performance Highlights**: 이 연구에서 제안된 end-to-end 아키텍처는 차세대 위성 네트워크(예: 6G 지원)를 위해 설계되었으며, 효율성 및 정확성 면에서 상당한 개선을 보여줍니다. 이는 객체 탐지, 패턴 인식 및 실시간 의사결정 능력을 향상시키는 데 기여합니다.



### FLeNS: Federated Learning with Enhanced Nesterov-Newton Sketch (https://arxiv.org/abs/2409.15216)
Comments:
          10 pages, 3 figures, 2 Tables

- **What's New**: FLeNS (Federated Learning with Enhanced Nesterov-Newton Sketch)는 Nesterov 방법의 가속화 능력과 Hessian 스케치의 차원 축소 이점을 결합하여 연합 학습의 통신 효율성과 수렴 속도를 개선하는 새로운 최적화 알고리즘을 제안합니다.

- **Technical Details**: FLeNS는 Nesterov 가속화와 적응형 Hessian 스케칭을 통해 중앙집중식 Newton 방법을 근사화하는 방식으로, 정확한 Hessian에 의존하지 않고 통신 오버헤드를 대폭 줄입니다. 이론적 분석 결과 FLeNS는 통신 라운드에서 초선형 수렴 속도를 달성하여 연합 최적화에서 중요한 발전을 이루었습니다.

- **Performance Highlights**: FLeNS는 통신 요구 사항을 줄이면서도 우수한 성능을 발휘하며, 특히 개인정보 보호가 중요한 경우와 엣지 컴퓨팅 시나리오에서 유리합니다. 실험적 평가를 통해 이론적 발견을 검증하며, 실질적인 연합 환경에서의 확장성을 보여줍니다.



### MAR-DTN: Metal Artifact Reduction using Domain Transformation Network for Radiotherapy Planning (https://arxiv.org/abs/2409.15155)
Comments:
          Accepted in 27th International Conference on Pattern Recognition (ICPR). Mubashara Rehman and Belén Serrano-Antón, both co-first authors of the manuscript

- **What's New**: 본 연구는 기계적 이식물(지르코니아 또는 금속 재료)로 인해 왜곡된 kVCT 이미지를 MVCT 이미지로 변환하여 아티팩트(artifact) 없는 이미지를 생성하는 새로운 심층 학습 기반 접근 방식을 제안합니다.

- **Technical Details**: 이 방법은 UNet 구조를 기반으로 하여 kVCT 이미지를 MVCT 이미지로 변환하는 과정을 체계적으로 수행합니다. 모델은 512x512 픽셀 이미지를 처리하며, 385838 슬라이스의 kVCT 이미지를 사용하여 학습되었습니다. PSNR과 SSIM 계산은 배경을 제외한 관심 영역만을 고려하여 수행됩니다.

- **Performance Highlights**: 제안된 방법은 전체 환자 볼륨에서 PSNR 30.02 dB, 아티팩트가 영향을 미친 영역에서는 27.47 dB를 달성하였습니다. 이는 의미 있는 개선을 나타내며, 방사선 종양학자들이 kVCT 만으로도 MVCT의 통찰력을 얻을 수 있게 해줍니다.



### Towards Accountable AI-Assisted Eye Disease Diagnosis: Workflow Design, External Validation, and Continual Learning (https://arxiv.org/abs/2409.15087)
- **What's New**: 이번 연구는 나이 관련 황반 변성(Age-related Macular Degeneration, AMD) 진단 및 심각도 분류를 위한 AI 보조 진단 작업 흐름을 설계하고 구현하여, AI 도움의 효과를 입증했습니다.

- **Technical Details**: 연구팀은 12개 기관의 24명의 임상 의사를 대상으로 실제 환자 데이터를 활용하여 AI 지원과 비지원 진단 성능을 비교하였습니다. 또한, 약 40,000개의 추가 의료 이미지를 포함한 AREDS2 데이터세트를 통해 기존 AI 모델을 지속적으로 개선하였습니다.

- **Performance Highlights**: AI 지원이 24명의 의사 중 23명의 진단 정확도와 분류 성능을 현저히 향상시켰고, F1-score는 평균 20% 증가하여 37.71에서 45.52로 상승했습니다(P-value < 0.0001). AI 지원은 19명의 임상 의사 중 17명에서 진단 시간을 최대 40% 단축시켰습니다. 지속 학습이 포함된 모델은 세 개의 독립 데이터세트에서 29%의 정확도 증가를 기록하며, 싱가포르 인구에서 F1-score를 42에서 54로 향상시켰습니다.



### ViBERTgrid BiLSTM-CRF: Multimodal Key Information Extraction from Unstructured Financial Documents (https://arxiv.org/abs/2409.15004)
Comments:
          Accepted in MIDAS (The 8th Workshop on MIning DAta for financial applicationS) workshop of ECML PKDD 2023 conference

- **What's New**: 이 논문은 비정형 문서에서의 핵심 정보 추출(Information Extraction, KIE) 모델에 대한 새로운 접근 방식을 제안합니다. 특히, ViBERTgrid라는 다중 모드 트랜스포머를 비정형 금융 문서에 적응시키고 BiLSTM-CRF 레이어를 통합하여 성능을 크게 향상시킵니다.

- **Technical Details**: 제안된 ViBERTgrid BiLSTM-CRF 모델은 비정형 문서에서의 개체 인식(Named Entity Recognition, NER) 성능을 2%포인트 향상시키면서도 반구조 문서에서의 KIE 성능을 유지합니다. 이 모델은 두 가지 주요 아키텍처인 ViBERTgrid(트랜스포머 기반)와 BiLSTM-CRF(시퀀스 기반)를 결합하여 구문 및 장기 컨텍스트 인식을 제공합니다.

- **Performance Highlights**: 이 모델은 비정형 자금 이체 주문 데이터셋 및 반구조 영수증 데이터셋(SROIE)에서 평가되었으며, SROIE 데이터셋에 대한 토큰 수준 주석을 공개하여 다중 모드 시퀀스 레이블링 모델에서의 사용 가능성을 높였습니다.



### Multi-Modal Generative AI: Multi-modal LLM, Diffusion and Beyond (https://arxiv.org/abs/2409.14993)
- **What's New**: 이 논문에서는 다중 모달 제네레이티브 AI에 대한 체계적인 리뷰와 더불어, 통합 모델의 가능성 및 설계 방안에 대해 논의합니다. 특히, 다중 모달 대형 언어 모델(MLLM)과 확산 모델(diffusion model)의 특징을 비교하며, 이해와 생성을 동시에 수행할 수 있는 통합 모델의 필요성에 대해 다룹니다.

- **Technical Details**: 논문에서는 MLLM과 확산 모델의 확률적 모델링 기법, 아키텍처 디자인, 그리고 영상 및 비디오 생성에 대한 응용을 포함한 공통적인 구조를 분석합니다. 두 가지 중요한 질문을 제시하며, 모델이 자가 회귀(auto-regressive) 또는 확산(probabilistic) 모델링을 채택해야 하는지와 밀집 아키텍처(dense architecture) 또는 전문가 혼합(Mixure of Experts, MoE) 아키텍처를 어떻게 활용할 것인지에 대한 전략도 제안합니다.

- **Performance Highlights**: MLLM은 비주얼 이해 분야에서 두각을 나타내며, 다양한 비주얼-언어 프리트레이닝 방법과 비주얼 토크나이저의 발전을 통해 강화된 성능을 보이고 있습니다. BLIP 모델은 이미지-텍스트 이해 및 생성에서 중요한 역할을 하며, 새로운 데이터셋 생성 방법을 통해 모델의 학습 효율성을 높이고 있습니다.



### CON: Continual Object Navigation via Data-Free Inter-Agent Knowledge Transfer in Unseen and Unfamiliar Places (https://arxiv.org/abs/2409.14899)
Comments:
          6 pages, 3 figures, workshop paper's draft version

- **What's New**: 본 연구는 새로운 로봇 객체 목표 탐색(Object Goal Navigation, ON) 문제 해결을 위한 잠재적인 초고속 상호간 지식 전달(Knowledge Transfer, KT)을 탐구합니다. 여행자 로봇(학생)이 지역 로봇(교사)과 간단한 상호작용을 통해 ON 지식을 획득하는 프레임워크를 제안하며, 데이터 없는 지속 학습(Continual Learning, CL) 문제로 구성하여 기존 블랙박스 모델에서 새로운 모델로 지식을 전송하는 방식을 다룹니다.

- **Technical Details**: 이 연구는 비협조적인 블랙박스 교사 로봇을 위한 경량 KT 모듈을 개발하며, 모든 교사 로봇이 시각 및 이동 능력을 가지고 있다고 가정합니다. 상태-행동 이력을 기본 지식으로 정의하며, 이를 통해 동적으로 타겟 객체 위치를 표현하는 쿼리 기반 점유 맵을 개발합니다. 이 맵은 정교하고 통신 친화적인 지식 표현으로 사용됩니다.

- **Performance Highlights**: 이 방법은 Habitat 환경에서 실험을 통해 검증되었으며, 기존 방법들에 비해 보다 효율적이고 안전한 객체 탐색을 가능하게 함을 보입니다. 경량의 점유 맵은 원래 데이터 세트에 비해 상당히 가벼워 통신 비용을 줄이며, 또한 스팸한 그리드 맵 구조로 더욱 압축이 가능합니다.



### Observe Then Act: Asynchronous Active Vision-Action Model for Robotic Manipulation (https://arxiv.org/abs/2409.14891)
- **What's New**: 이 논문에서는 제한된 시각적 관찰 하에서 로봇 조작 문제를 다루고, 작업 지향적인 비동기 능동 시각-행동 모델을 제안합니다. 이 모델은 카메라의 Next-Best-View(NBV) 정책과 그리퍼의 Next-Best-Pose(NBP) 정책을 직렬로 연결하여 센서-모터 조정 프레임워크 내에서 Few-Shot 강화 학습을 통해 훈련됩니다.

- **Technical Details**: 제안된 모델은 두 개의 에이전트, 즉 최적 카메라 뷰를 추론하는 NBV 에이전트와 NBV에서 추론된 관찰에 따라 그리퍼의 행동을 결정하는 NBP 에이전트로 구성됩니다. 각 에피소드에서 센서와 모터 행동 추론을 번갈아 하여 환경에 대한 능동적 인식을 가능하게 합니다.

- **Performance Highlights**: RLBench에서의 8개의 제한된 시점 과제에 대한 훈련 및 평가 결과, 본 모델은 기준 알고리즘들을 일관되게 초월하며 조작 작업에서 시각적 제약을 처리하는 데 있어 효과적임을 보여주었습니다.



### Towards Ground-truth-free Evaluation of Any Segmentation in Medical Images (https://arxiv.org/abs/2409.14874)
Comments:
          17 pages, 15 figures

- **What's New**: 이번 연구에서는 의학 영상에서 Segment Anything Model (SAM)과 그 변형들이 생성하는 분할(segmentation) 품질을 평가할 수 있는 ground-truth-free(기준 데이터 없음) 평가 모델을 제안했습니다. 이 모델은 입력 이미지와 해당하는 분할 예측 간의 일관성과 응집력을 분석하여 품질 점수를 추정합니다.

- **Technical Details**: 이 평가 모델은 지도 학습(supervised learning) 프레임워크 내의 회귀 문제로 구성되며, Dice 수치 등과 평균 제곱 오차(mean squared error)를 사용하여 학습 손실을 계산합니다. EvanySeg라는 이름의 이 모델은 ResNet 및 ViT 등 다양한 컨볼루션 모델(convolution-based models)과 변환기 모델(transformer-based models)을 활용하였으며, ViT 모델이 더 나은 성능을 보였습니다.

- **Performance Highlights**: EvanySeg는 저품질 분할 샘플을 발견하고, 기준 데이터 없이 분할 모델을 벤치마킹하며, 인간-인공지능 협업 중 저품질 분할 예측에 대해 전문가에게 경고하고, 여러 개의 분할 모델이 있을 때 각 테스트 샘플에 대해 최상의 분할 예측을 선택할 수 있는 다양한 과제에 활용될 수 있습니다. 코드와 모델은 공개될 예정입니다.



### A-VL: Adaptive Attention for Large Vision-Language Models (https://arxiv.org/abs/2409.14846)
- **What's New**: 본 연구에서는 LVLM(Inference)에서의 메모리 및 계산 부하를 감소시키기 위해 A-VL이라는 적응형 어텐션 메커니즘을 제안합니다. 새로운 어텐션 패턴 분석을 통해 각 모달리티에 대해 별도의 어텐션 처리가 필요함을 밝히고, 이를 적절히 조정하여 성능 저하 없이 효율성을 극대화했습니다.

- **Technical Details**: A-VL은 시각 입력 및 언어 입력의 어텐션 패턴 차이를 이해하여 각 모달리티를 별도로 관리합니다. 시각 입력의 경우 중요한 정보를 캐시하여 가장 중대한 부분만 계산하고, 언어 입력은 지역 정보를 중시하여 필요한 원거리 텍스트 캐시만을 유지합니다. 이 연구는 각각의 입력 모드에 대한 메모리 사용 및 계산 부하를 줄이기 위해 KV 캐시의 중대한 부분만 선택하는 방법을 사용합니다.

- **Performance Highlights**: A-VL은 세 가지 비전-언어 작업 및 다섯 가지 데이터 세트에서의 평가를 통해 기존의 적응형 어텐션 방법들에 비해 메모리와 계산 효율성이 높다는 것을 입증했습니다. 따라서 LVLM의 실용성을 크게 향상시킵니다.



### RoWSFormer: A Robust Watermarking Framework with Swin Transformer for Enhanced Geometric Attack Resilienc (https://arxiv.org/abs/2409.14829)
- **What's New**: RoWSFormer는 Swin Transformer를 기반으로 한 새로운 이미지 워터마킹 프레임워크로, 기존 CNN 기반 방법들을 초월하는 성능을 목표로 합니다.

- **Technical Details**: RoWSFormer는 Locally-Channel Enhanced Swin Transformer Block (LCESTB)와 Frequency-Enhanced Transformer Block (FETB)를 핵심 구성 요소로 사용하여 전역 및 장거리 정보를 효과적으로 캡처하고, 빈도 영역의 특성을 추출하여 워터마킹의 강인성을 강화합니다.

- **Performance Highlights**: RoWSFormer는 대부분의 비기하학적 공격에 대해 PSNR을 3 dB 개선하며, 기하학적 공격에 대한 경우 PSNR을 6 dB 이상 향상시키고, 추출 정확도는 97%를 초과하는 성과를 보입니다.



### Towards Efficient and Robust VQA-NLE Data Generation with Large Vision-Language Models (https://arxiv.org/abs/2409.14785)
Comments:
          Preprint

- **What's New**: 본 연구는 대규모 비전-언어 모델(LVLMs)을 활용하여 효율적이고 고품질의 합성 VQA-NLE(비전 질문-응답 자연어 설명) 데이터셋을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: LVLMs의 생성 능력을 활용하여 복잡한 인간 주석 프로세스를 대체하고, 시각적 프롬프트(visual prompts)를 포함하여 데이터 생성의 정확성을 향상시켜 고품질의 설명을 생성하는 두 가지 접근 방식을 도입합니다. 그 과정에서 데이터 triplet (질문, 답변, 설명) 생성을 위한 다양한 프롬프트 파이프라인을 사용합니다.

- **Performance Highlights**: 합성된 VQA-NLE 데이터 생성 방식이 인간 주석에 비해 최대 20배 빠르며, 데이터 품질이 거의 동일하게 유지됨을 보여줍니다. 이 연구는 시각적 프롬프트를 포함하여 텍스트 생성의 관련성을 증가시켰습니다.



### TransUKAN:Computing-Efficient Hybrid KAN-Transformer for Enhanced Medical Image Segmentation (https://arxiv.org/abs/2409.14676)
- **What's New**: 이 논문에서는 TransUKAN이라는 새로운 네트워크 아키텍처를 제안합니다. TransUKAN은 U-Net, Transformer 및 Kolmogorov–Arnold Networks (KAN)를 결합하여 의료 이미지 분할에서 지역적 비선형 관계를 캡처하면서 글로벌 정보를 모델링 할 수 있게 합니다.

- **Technical Details**: TransUKAN은 U-Net과 Transformer의 장점을 통합하고 KAN을 활용하여 새로운 구조를 형성합니다. KAN의 개선 버전을 Transformer에 도입함으로써 지역 세부 정보를 모델링하는 성능을 높이고, 비선형 관계를 모델링하는 데 필요한 추가 매개변수를 최소화합니다. EfficientKAN은 메모리 사용량과 계산 량을 줄이는 구조입니다.

- **Performance Highlights**: TransUKAN은 다양한 의료 이미지 분할 작업에서 우수한 성능을 보이며, 기존의 최첨단 방법과 비교할 때 상당히 적은 수의 매개변수로도 동일한 성과를 달성할 수 있음을 실험적으로 입증했습니다.



### RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning (https://arxiv.org/abs/2409.14674)
Comments:
          Project Website: this https URL

- **What's New**: 본 논문은 로봇 조작을 위한 강력하고 교정 가능한 비주얼-모터(Visuomotor) 정책 개발의 어려움을 다룹니다. 실패 복구 메커니즘과 간단한 언어 지시의 한계를 극복하기 위해, 자동으로 전문가 시연을 실패 복구 궤적(failure recovery trajectories)과 세부 언어 주석으로 보강하는 데이터 생성 파이프라인을 제안합니다.

- **Technical Details**: 우리는 Rich languAge-guided failure reCovERy (RACER)라는 감독-행위자(supervisor-actor) 프레임워크를 소개하며, 이는 실패 복구 데이터를 풍부한 언어 설명과 결합하여 로봇 제어를 향상시킵니다. RACER는 온라인 감독으로 작동하는 비전-언어 모델(Vision-Language Model, VLM)과 다음 행동을 예측하는 언어 조건 비주얼-모터 정책을 포함합니다.

- **Performance Highlights**: 실험 결과, RACER는 RLbench의 다양한 평가 설정에서 기존 최첨단 모델인 Robotic View Transformer (RVT)를 초월하여 우수한 성능을 보여주었습니다. 이는 시뮬레이션 및 실제 환경 모두에서 탁월한 Robustness와 Adaptability를 입증합니다.



### FedGCA: Global Consistent Augmentation Based Single-Source Federated Domain Generalization (https://arxiv.org/abs/2409.14671)
Comments:
          6 pages, 7 figures, conference

- **What's New**: 본 논문은 단일 출처 연합 도메인 일반화(single-source FedDG) 문제를 다루며, 이를 해결하기 위한 새로운 방법인 연합 글로벌 일관성 증강(FedGCA)을 제안합니다. 이 방법은 다양한 도메인 스타일로 데이터를 증강하기 위해 스타일 보강 모듈을 통합합니다.

- **Technical Details**: FedGCA는 제한된 도메인 스타일의 데이터 샘플을 증강하기 위해 스타일 보강 모듈을 사용합니다. 이 과정에서 FedGCA는 글로벌 가이드 의미 일관성(global guided semantic consistency)과 클래스 일관성(class consistency) 손실을 통해 각 클라이언트와 클래스 간의 의미 불일치를 완화합니다.

- **Performance Highlights**: 다양한 데이터셋에서 실시된 실험 결과, FedGCA는 기존의 벤치마크 모델보다 우수한 성능을 보이며, 이를 통해 sFedDG 문제를 효과적으로 해결함을 입증합니다.



### RobotFingerPrint: Unified Gripper Coordinate Space for Multi-Gripper Grasp Synthesis (https://arxiv.org/abs/2409.14519)
Comments:
          7 pages, 8 figures, 2 tables. Project page available at this https URL

- **What's New**: 이번 논문에서는 여러 그리퍼를 위한 그립 합성을 위한 통합 그리퍼 좌표 공간(UGCS)이라는 새로운 표현 방식을 소개합니다. UGCS는 모든 로봇 그리퍼를 공유하는 3D 구의 2D 면이며, 경도 및 위도를 좌표로 사용합니다.

- **Technical Details**: UGCS는 로봇 그리퍼의 내부 표면 포인트를 구의 표면으로 매핑하여 표현하는 좌표 공간입니다. 이 좌표 공간은 다양한 그리퍼에 공통적으로 사용되며, 내부 표면에 대한 밀집 표현을 제공합니다. 연구진은 조건부 변량 오토인코더(Conditional Variational Autoencoder, CVAE)를 사용하여 입력 객체의 포인트 클라우드에 대한 UGCS 좌표를 예측합니다.

- **Performance Highlights**: 제안된 방법인 RobotFingerPrint는 MultiDex 데이터셋을 사용하여 실험을 수행하였으며, 그립 성공률과 다양성 면에서 GenDexGrasp 및 GeoMatch보다 우수함을 보여주었습니다.



### SPAQ-DL-SLAM: Towards Optimizing Deep Learning-based SLAM for Resource-Constrained Embedded Platforms (https://arxiv.org/abs/2409.14515)
Comments:
          To appear at the 18th International Conference on Control, Automation, Robotics and Vision (ICARCV), December 2024, Dubai, UAE

- **What's New**: 본 논문에서는 자원 및 에너지 효율성을 모두 고려한 DL-SLAM 알고리즘 최적화 프레임워크인 SPAQ-DL-SLAM을 제안합니다.

- **Technical Details**: SPAQ-DL-SLAM은 DROID-SLAM 아키텍처에 Structured Pruning과 Quantization을 적용하여 20%의 구조적 프루닝과 8비트 PTQ를 이용하여 FLOPs를 18.9% 감소시키고 모델 크기를 79.8% 줄였습니다. 이 과정에서 레이어 별 감도 분석을 기반으로 세부 조정을 실시했습니다.

- **Performance Highlights**: TUM-RGBD 벤치마크에서 SPAQ-DROID-SLAM은 DROID-SLAM보다 절대 궤적 오류(ATE)에서 평균 10.5% 향상되었으며, ETH3D SLAM 벤치마크에서는 더 높은 AUC 점수를 기록해 우수한 일반화 능력을 입증했습니다.



### Lesion Segmentation in Whole-Body Multi-Tracer PET-CT Images; a Contribution to AutoPET 2024 Challeng (https://arxiv.org/abs/2409.14475)
Comments:
          7 pages, 4 tables, 1 figure, AutoPET MICCAI 24

- **What's New**: 이 연구는 전체 몸 PET-CT( Positron Emission Tomography-Computed Tomography) 볼륨 내 병리학적 영역을 자동으로 분할하는 방법을 제안하며, AutoPET MICCAI 2024 챌린지에 기여합니다.

- **Technical Details**: 제안된 워크플로우는 이미지 전처리(image preprocessing), 트레이서 분류(tracer classification), 병변 분할(lesion segmentation) 단계로 구성되어 있습니다.

- **Performance Highlights**: 모델의 분할 정확도(segmentation accuracy)는 1611명의 훈련 피험자를 대상으로 평균 Dice 점수가 0.548로 개선되었으며, 분류된 FDG 및 PSMA 피험자는 각각 0.631과 0.559의 점수를 기록했습니다. 예비 테스트 데이터셋에서는 0.792의 성과를 보였습니다.



### Detection of pulmonary pathologies using convolutional neural networks, Data Augmentation, ResNet50 and Vision Transformers (https://arxiv.org/abs/2409.14446)
Comments:
          10 pages

- **What's New**: 이 논문에서는 폐 질환 진단을 위한 새로운 방법을 제안합니다. 이 방법은 Convolutional Neural Networks (CNN)와 Data Augmentation, ResNet50 및 Vision Transformers (ViT)를 기반으로 하여 의료 이미지를 분석합니다.

- **Technical Details**: 제안된 방법은 다양한 폐 질환을 가진 환자의 X-ray 및 CT 스캔 이미지를 포함하는 데이터셋을 사용합니다. 성능 평가에는 Accuracy, Sensitivity, Specificity 및 ROC curve 아래 면적과 같은 평가 지표가 포함됩니다.

- **Performance Highlights**: 제안된 방법은 모든 성능 지표에서 기존 방법들보다 우수한 성능을 보였으며, 정확도는 98%, ROC curve 아래 면적은 99%에 달합니다. 이를 통해 제안된 방법이 의료 이미지를 이용한 폐 질환 진단에 효과적이고 유망한 도구임을 결론짓습니다.



### Dormant: Defending against Pose-driven Human Image Animation (https://arxiv.org/abs/2409.14424)
- **What's New**: 본 논문에서는 pose-driven human image animation 기술에 대한 새로운 방어 방법인 Dormant를 제안하여, 개인의 초상권과 프라이버시를 보호하는 효과적인 방어 메커니즘을 제공합니다.

- **Technical Details**: Dormant는 입력된 이미지를 보호하기 위해 protective perturbation을 적용하여 시각적인 유사성을 유지하면서도 저화질 비디오 생성을 유도합니다. 이 perturbation은 appearance feature의 비정상적인 추출을 초래하고 생성된 비디오 프레임 간의 일관성을 깨뜨리도록 최적화됩니다. 연구진은 8가지 animation 방법과 4개의 데이터셋을 포함하여 광범위한 평가를 수행하였습니다.

- **Performance Highlights**: Dormant는 6개의 기존 보호 방법에 비해 우수한 성능을 보이며, 생성된 비디오에서의 정체성 불일치, 시각적 왜곡, 눈에 띄는 아티팩트 및 일관성 결여를 초래합니다. 또한 Dormant는 6가지 현실 세계 상업 서비스에서도 효과적으로 작동하여, 다양한 생성 방법에 대한 방어 능력을 보여줍니다.



### GraspMamba: A Mamba-based Language-driven Grasp Detection Framework with Hierarchical Feature Learning (https://arxiv.org/abs/2409.14403)
Comments:
          8 pages. Project page: this https URL

- **What's New**: GraspMamba는 Mamba 비전을 이용한 새로운 언어 기반 grasp 감지 방법으로, 효율적인 다계층(feature fusion) 통합을 통해 복잡한 이미지를 처리하고 빠른 추론 속도를 구현합니다.

- **Technical Details**: GraspMamba는 Mamba 기반 백본을 활용하여 텍스트 정보를 통해 시각적 특성을 다계층으로 통합하는 기법을 제안합니다. 이 접근 방식은 텍스트와 비주얼 특징을 공동 공간에서 정렬하여 다중 스케일에서 모달 표현을 결합합니다. 모델은 실질적으로 현재의 언어 기반 grasp 감지 기술의 한계를 극복하도록 설계되었습니다.

- **Performance Highlights**: 대규모 언어 기반 grasping 데이터세트에서 실행된 실험 결과, GraspMamba는 최신 기술 대비 정확도와 추론 속도에서 명백하게 우수한 성능을 보여줍니다. 또한, 제로샷 학습(zero-shot learning) 지원으로 실세계 로봇용 손잡이 애플리케이션으로 일반화될 수 있습니다.



### Frequency-regularized Neural Representation Method for Sparse-view Tomographic Reconstruction (https://arxiv.org/abs/2409.14394)
Comments:
          6 pages,5 figures,Accepted to ICME 2024

- **What's New**: 이 논문에서는 Sparse-view 토모그래픽 재구성을 위한 Frequency Regularized Neural Attenuation/Activity Field (Freq-NAF)를 소개합니다. 이 방법은 고주파 정보와 저주파 정보를 균형 있게 조절하여 과적합(overfitting) 문제를 완화합니다.

- **Technical Details**: Freq-NAF는 입력 이미지의 주파수 정보를 정규화하는 주파수 정규화 항을 도입하여 NAF (Neural Attenuation Field) 모델에서 발생하는 과적합 문제를 해결합니다. 이 방법은 CBCT와 SPECT 데이터셋에서 수행된 수치 실험을 통해 효과성을 입증했습니다.

- **Performance Highlights**: Freq-NAF는 기존의 최첨단 방법들보다 더 높은 정확도를 보여주었으며, CT 복부 단면 이미지의 아티팩트를 줄이고, SPECT 이미지의 노이즈를 감소시키며, 재구성된 이미지의 공간 해상도(spatial resolution)와 대비(contrast)를 향상시켰습니다.



### Thinking in Granularity: Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues (https://arxiv.org/abs/2409.14330)
- **What's New**: 본 논문에서는 이미지 초해상도(SR) 분야에서 Dynamic quantization을 적용한 Granular-DQ라는 새로운 방법을 제안합니다. 이 방법은 이전의 레이어 감도에 대한 고려 없이 이미지의 고유한 특성을 활용하여 각 패치에 대한 비트 할당을 개선합니다.

- **Technical Details**: Granular-DQ는 다중 세분성(multi-granularity) 분석을 진행하며, 이를 통해 패치의 정보 밀도를 탐색하여 동적 양자화(dynamically quantization)를 수행합니다. 이 과정에서 Granularity-Bit Controller (GBC)를 개발하고, 엔트로피-비트(Entropy-to-Bit, E2B) 메커니즘을 통해 높은 비트를 가진 패치에 대해 세밀하게 비트를 조정합니다.

- **Performance Highlights**: Granular-DQ는 다양한 SR 모델들을 대상으로 한 실험에서 최근의 최첨단 방법들에 비해 정확도와 양자화 효율 간의 균형에서 우수성을 입증했습니다.



### Can-Do! A Dataset and Neuro-Symbolic Grounded Framework for Embodied Planning with Large Multimodal Models (https://arxiv.org/abs/2409.14277)
- **What's New**: 이 논문은 Can-Do라는 새로운 벤치마크 데이터 세트를 도입하여 대형 다중 모달 모델의 체화된 계획 능력을 평가합니다. 이 데이터 세트는 이전의 데이터 세트보다 더 다양한 복잡한 시나리오를 포함하고 있으며, 400개의 다중 모달 샘플로 구성되어 자연어 사용자 지침, 환경을 묘사하는 시각 이미지, 상태 변화 및 해당 동작 계획을 포함하고 있습니다.

- **Technical Details**: Can-Do 데이터 세트는 실제 환경을 묘사하기 위해 실제 장면 이미지와 합성 이미지를 모두 활용합니다. 세 가지 태스크 카테고리(물리적 이해, 상식, 안전)을 중심으로 설계되었으며, 각 샘플은 사용자 의도를 기반으로 시각 시나리오를 인식하고 단계를 생성하는 모델의 능력을 평가합니다. 연구에서는 또한 NeuroGround라는 신경 상징적 프레임워크를 제안하여 모델 생생 생성 과정이 환경의 초기 및 목표 상태에 명확하게 기반하도록 합니다.

- **Performance Highlights**: 실험 결과, NeuroGround 프레임워크는 기존의 강력한 기준선과 비교하여 상당한 이점을 보여주었습니다. 특히, 체화된 계획에서 기존 모델(GPT-4V 포함)의 병목 현상인 시각적 지각, 이해 및 추론 능력에서 개선된 성능을 입증했습니다.



### FeDETR: a Federated Approach for Stenosis Detection in Coronary Angiography (https://arxiv.org/abs/2409.14268)
Comments:
          9 pages, 9 figures, Image Analysis and Processing - ICIAP 2023 Workshops. ICIAP 2023. Lecture Notes in Computer Science, vol 14366. Springer, Cham

- **What's New**: 본 연구에서는 심장 혈관 조영술(Coronary Angiography)에서 좁아진 혈관(stenosis)의 심각성을 평가하기 위한 최초의 연합 학습(federated learning) 기반 검출 변환기(detection transformer) 접근 방식인 FeDETR을 제안합니다.

- **Technical Details**: FeDETR은 각 노드가 로컬 데이터셋에서 검출 변환기(DETR)를 훈련시키고, 중앙 서버가 네트워크의 백본(backbone) 부분을 연합하여 처리하는 방식을 사용합니다. 이 방법은 FFR/ iFR 값을 기반으로 좁아진 혈관의 중대한 정도를 평가하는 데 주안점을 둡니다.

- **Performance Highlights**: 제안된 방법은 5개의 병원에서 수집된 총 1001개의 혈관 조영 검사로 만들어진 데이터셋으로 훈련 및 평가되었으며, 최신 연합 학습 방법(FedAvg, FedBN)과 성능을 비교하여 유효성을 확인하였습니다.



### R-AIF: Solving Sparse-Reward Robotic Tasks from Pixels with Active Inference and World Models (https://arxiv.org/abs/2409.14216)
Comments:
          20 pages, 2 algorithms, 2 tables, 5 figures, submitted to ICRA 2025

- **What's New**: 이번 연구는 부분 관측 마르코프 의사결정 프로세스(POMDP)에서의 능동 추론(active inference, AIF) 모델을 다루며, 특히 sparse reward 신호가 있는 연속 액션 공간을 가진 POMDP 제어 문제를 해결하기 위한 독창적인 접근 방식을 제안합니다.

- **Technical Details**: 이 논문에서는 CRSPP(Contrastive Recurrent State Prior Preference) 모델을 통해 에이전트가 온라인으로 환경 상태에 대한 선호를 학습하도록 하고, 강화학습(actor-critic) 방법을 사용하여 예상 자유 에너지를 최적화하여 액션 플래너의 안정성을 높입니다. 또한, R-AIF(Robust Active Inference) 에이전트를 도입하여 스스로 수정하는 메커니즘을 사용해 sparse-reward 작업에서 모델의 수렴 속도를 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 R-AIF 에이전트가 기존의 최첨단 모델(DreamerV3) 및 다른 AIF 기준 모델보다 누적 보상, 상대적 안정성, 성공률 측면에서 향상된 성능을 보였습니다.



### UniMo: Universal Motion Correction For Medical Images without Network Retraining (https://arxiv.org/abs/2409.14204)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 다양한 이미징 모드에서 모션 교정의 문제를 해결하기 위해 딥 뉴럴 네트워크를 활용한 보편적인 모션 교정 프레임워크인 UniMo를 소개합니다. 기존 모델이 새로운 이미지 모드에 대해 반복적인 추측이나 재훈련을 요구하는 한계를 극복하며, 단일 모드에서의 1회 훈련으로 다수의 보지 못한 이미지 모드에서의 고 안정성과 적응성을 유지합니다.

- **Technical Details**: UniMo는 모양(shape)과 이미지(image)의 다중모드 지식을 통합하는 공동 학습 프레임워크를 개발하여, 이미지 외관 변화에도 불구하고 모션 교정 정확도를 개선합니다. 또한, 기하학적 변형 증강기(geometric deformation augmenter)를 통해 대규모 모션 교정의 견고성을 향상시키고, 지역 왜곡(local deformations) 및 객체 변형(object deformations)으로 인한 문제를 해결합니다.

- **Performance Highlights**: 여러 데이터 세트에서 진행된 실험 결과, UniMo는 정확도 측면에서 기존의 모션 교정 방법을 초과했음을 보여줍니다. UniMo는 재훈련 없이 어떤 이미지 모드에서든 모션 교정을 수행할 수 있는 첫 번째 방법이며, 고도로 정확한 이미지 쌍 등록을 위한 실시간 추론(real-time inference)이 가능합니다.



### A Sinkhorn Regularized Adversarial Network for Image Guided DEM Super-resolution using Frequency Selective Hybrid Graph Transformer (https://arxiv.org/abs/2409.14198)
Comments:
          25 pages, 19 figures. arXiv admin note: substantial text overlap with arXiv:2311.16490

- **What's New**: 본 연구에서는 고해상도 (HR) 다중 스펙트럼 (MX) 위성 이미지를 기반으로 고해상도 디지털 고도 모델 (DEM)을 생성하기 위해 새로운 하이브리드 변환기 모델을 제안합니다. 이 모델은 Densely connected Multi-Residual Block (DMRB) 및 multi-headed Frequency Selective Graph Attention (M-FSGA)으로 구성되어 있습니다.

- **Technical Details**: 하이브리드 변환기 블록은 조정된 판별 공간 맵 (discriminator spatial maps)을 통해 HR MX 이미지를 가이드로 활용합니다. Sinkhorn 거리 최적화와 관련된 새로운 적대적 (adversarial) 목표도 제시하고, 경량화를 위해 이론적 및 경험적 근거를 제공합니다.

- **Performance Highlights**: 실험 결과, 본 모델은 4개의 서로 다른 DEM 데이터셋을 기반으로 기존 기준 방법들과의 질적, 양적 비교에서 더 우수한 성능을 보였으며, 날카로운 세부 묘사와 최소화된 오류를 통해 뛰어난 결과를 나타냈습니다.



### Accelerated Multi-Contrast MRI Reconstruction via Frequency and Spatial Mutual Learning (https://arxiv.org/abs/2409.14113)
Comments:
          Accepted as a poster by Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024

- **What's New**: 이번 논문에서는 Frequency and Spatial Mutual Learning Network (FSMNet)을 제안합니다. FSMNet은 다양한 모달리티에서 글로벌 종속성을 효율적으로 탐색하고, 보조 모달리티의 보완 정보를 활용하여 고품질의 다중 대조 자기 공명(Magnetic Resonance, MR) 이미지를 재구성하는 새로운 접근법입니다.

- **Technical Details**: FSMNet은 Frequency-Spatial Feature Extraction (FSFE) 모듈을 통해 각 모달리티의 특징을 추출하며, 주파수 분기와 공간 분기로 구성됩니다. 주파수 분기를 통해 이미지 크기 수용 영역을 활용하여 글로벌 종속성을 캡처하고, 공간 분기를 통해 지역적 특징을 추출합니다. Cross-Modal Selective fusion (CMS-fusion) 모듈은 보조 모달리티로부터 주파수와 공간 특징을 선택적으로 통합하여 목표 모달리티의 해당 분기를 강화합니다. 이후 FS-fusion 모듈을 통해 주파수 분기에서 강화된 글로벌 특징과 공간 분기에서 강화된 지역적 특징을 통합하여 포괄적인 특징 표현을 생성합니다.

- **Performance Highlights**: BraTS 및 fastMRI 데이터셋에 대한 광범위한 실험을 통해 FSMNet은 다양한 가속화 요인에 대해 기존 MCMR 방법들보다 우수한 성능을 달성했습니다.



### Window-based Channel Attention for Wavelet-enhanced Learned Image Compression (https://arxiv.org/abs/2409.14090)
Comments:
          ACCV2024 accepted; reviewed version

- **What's New**: 이번 연구에서는 Learned Image Compression (LIC) 모델에서 Swin-Transformer 기반 접근법의 한계를 극복하기 위해 공간-채널 주의 메커니즘을 도입하여 보다 넓은 수용 영역을 확보하고, 이로 인해 이미지 내 큰 객체를 모델링하는 능력을 향상시켰습니다.

- **Technical Details**: 연구에서는 채널 주의 및 윈도우 파티션을 혼합하여 큰 수용 영역을 획득하고, 내용의 글로벌 정보를 포착하는 방법을 채택했습니다. 또한, 공간-채널 하이브리드 (SCH) 프레임워크에 이산 웨이블릿 변환 (Discrete Wavelet Transform)을 적용하여 주파수 의존적인 다운 샘플링을 효율적으로 수행하고 수용 영역을 더욱 확장했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 VTM-23.1에 비해 BD-rate를 각각 18.54%, 23.98%, 22.33%, 24.71% 감소시키며 최첨단 성능을 달성하였습니다.



### Recovering Global Data Distribution Locally in Federated Learning (https://arxiv.org/abs/2409.14063)
Comments:
          Accepted by BMVC 2024

- **What's New**: 본 논문은 Federated Learning(FL)에서 발생하는 레이블 불균형 문제를 해결하기 위한 새로운 접근 방식인 ReGL(Recovering Global data distribution Locally)를 제안합니다. 이 방법은 각 클라이언트가 생성 모델을 사용하여 마이너리티 및 누락된 클래스를 보완하는 이미지를 합성함으로써 레이블 불균형을 완화합니다.

- **Technical Details**: ReGL은 클라이언트 측에서 기초 생성 모델을 활용하여 마이너리티 및 누락된 클래스를 위한 합성 이미지를 생성합니다. 그런 다음, 클라이언트는 실제 데이터와 합성 데이터를 결합하여 로컬 모델을 훈련합니다. 중요한 것은 생성 및 파인튜닝(Adaptive Fine-tuning) 과정이 클라이언트 측에서 수행되며 데이터 프라이버시가 보호된다는 점입니다.

- **Performance Highlights**: 우리는 다양한 이미지 분류 데이터셋에 대한 포괄적인 실험을 통해 ReGL이 기존 FL 알고리즘들보다 평균 30%의 성능 향상을 보여주었음을 입증하였습니다. 이 방식은 레이블 분포 편향 문제를 효과적으로 해결하며, 전반적인 일반화와 로컬 개인화 모두에서 놀라운 우수성을 발휘합니다.



### ECHO: Environmental Sound Classification with Hierarchical Ontology-guided Semi-Supervised Learning (https://arxiv.org/abs/2409.14043)
Comments:
          IEEE CONECCT 2024, Signal Processing and Pattern Recognition, Environmental Sound Classification, ESC

- **What's New**: 이 논문에서는 Environmental Sound Classification (ESC) 분야에서 새로운 반지도 학습 프레임워크인 ECHO (Environmental Sound Classification with Hierarchical Ontology-guided semi-supervised Learning)를 제안합니다. 이 프레임워크는 레이블 온톨로지 기반 계층 구조를 활용하여 새로운 프리텍스트 작업을 정의하고, 이는 대량의 비지도 데이터 없이 모델의 성능을 향상시킵니다.

- **Technical Details**: ECHO 프레임워크는 Large Language Model (LLM)을 활용하여 생성된 새로운 레이블을 예측하는 프리텍스트 작업에서 시작합니다. 모델은 주어진 레이블 온톨로지를 기반으로 코스 레이블을 예측하며, 이후 실제 작업을 위한 지도 학습 단계에서 미세 조정됩니다. 이러한 과정에서 세 가지 데이터셋 (UrbanSound8K, ESC-10, ESC-50)에 대해 1%에서 8%까지의 정확도 향상을 보였습니다.

- **Performance Highlights**: ECHO는 기존 시스템에 비해 UrbanSound8K, ESC-10 및 ESC-50 데이터셋에서 성능을 1%에서 8% 향상시키는 성과를 달성했습니다. 이는 데이터가 부족한 환경에서도 우수한 정확도를 제공할 수 있음을 나타냅니다.



### MSDet: Receptive Field Enhanced Multiscale Detection for Tiny Pulmonary Nodu (https://arxiv.org/abs/2409.14028)
- **What's New**: 본 논문에서는 폐 결절 감지의 새로운 모델인 MSDet을 제안하였습니다. MSDet은 다중 스케일 주의 메커니즘과 향상된 수용 영역을 통합하여 높은 위양성율과 낮은 감지 정확도를 극복하도록 설계되었습니다.

- **Technical Details**: 제안된 MSDet 모델은 확장 수용 영역(ERD) 전략을 통해 결절의 문맥 정보를 풍부하게 포착하여 결절 차단으로 발생하는 위양성을 줄입니다. 그리고 위치 채널 주의 메커니즘(PCAM)을 설계하여 특징 학습을 최적화하고 다중 스케일 감지 오류를 줄입니다. Tiny Object Detection Block (TODB)을 사용하여 작은 결절 감지의 정확도를 높입니다.

- **Performance Highlights**: LUNA16 데이터셋에서의 실험 결과, 본 모델은 기존의 최첨단 방법인 YOLOv8에 비해 8.8%의 mAP(Mean Average Precision) 향상을 이루어냈습니다. 이는 조기 폐암 진단을 위한 정확성과 신뢰성을 크게 높이는 솔루션으로 자리잡을 수 있습니다.



### Simple Unsupervised Knowledge Distillation With Space Similarity (https://arxiv.org/abs/2409.13939)
- **What's New**: 이 논문에서는 Self-supervised learning (SSL)이 소형 네트워크에 쉽게 확장되지 않는 문제를 해결하기 위해, 기존의 필수적인 sample 간의 관계를 수작업으로 설정하는 대신, 교사의 embedding manifold을 직접 모델링하도록 학생을 유도하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 	extbf{space similarity}라는 손실 구성 요소를 이용하여 학생의 각 feature 공간의 차원이 교사의 해당 차원과 유사하도록 유도합니다. 이로 인해 교사의 embedding manifold과의 정렬을 통해 필수적인 정보 보존을 달성합니다.

- **Performance Highlights**: 다양한 UKD 벤치마크에서 제안된 접근 방식이 우수한 성능을 보여주며, 기존 방식에 비해 state of the art 결과를 보고합니다.



### RN-SDEs: Limited-Angle CT Reconstruction with Residual Null-Space Diffusion Stochastic Differential Equations (https://arxiv.org/abs/2409.13930)
- **What's New**: 이 논문에서는 Computed Tomography (CT) 이미지에서 Limited Angle Computed Tomography (LACT) 재구성 문제를 다루기 위해 Residual Null-Space Diffusion Stochastic Differential Equations (RN-SDEs) 모델을 제안합니다. 이는 평균 회귀(mean-reverting) 확률 미분 방정식으로 확산 과정을 특징짓는 확산 모델의 변형입니다.

- **Technical Details**: RN-SDE는 양호한 데이터 일관성을 유지하기 위해 Range-Null Space Decomposition (RNSD) 기반의 수정(rectification)을 강조하며, ChromSTEM 및 C4KC-KiTS라는 두 가지 LACT 데이터 세트에서 실험을 통해 일반화 가능성을 입증합니다. 또한, RN-SDEs는 고품질 이미지를 복원하는 과정에서 고도로 왜곡된 이미지를 처리할 수 있습니다.

- **Performance Highlights**: RN-SDEs를 사용하여 높은 품질의 이미지를 복원하고 대부분의 LACT 작업에서 최신 성능(SOTA)을 달성함을 보여줍니다. 또한 계산 복잡성과 실행 효율성에 대한 정량적 비교를 통해 제안된 접근 방식의 우수성을 강조합니다.



### Learning to Play Video Games with Intuitive Physics Priors (https://arxiv.org/abs/2409.13886)
Comments:
          7 pages, Accepted in Proceedings of the Annual Meeting of the Cognitive Science Society, Volume 46

- **What's New**: 이 논문에서는 비디오 게임 학습을 위한 객체 기반 입력 표현(object-based input representations)을 설계하여 여러 게임에서 잘 일반화할 수 있는 방법을 제시합니다. 이를 통해 인공지능 에이전트가 아기와 유사하게 제한된 경험을 바탕으로 게임을 배우는 방식을 연구합니다.

- **Technical Details**: 연구에서는 Q-learning 알고리즘을 사용하여 객체 범주 표현을 통해 게임의 상태 공간을 구성하며, 이러한 표현이 DQN 모델 대비 어떻게 학습 및 일반화 되는지를 비교 분석합니다. 또한, 'affordances'라는 개념을 도입해 물체의 상호작용을 혁신적으로 학습하는 방안을 모색합니다. 이 연구는 또한 인간의 공통적 직관 물리학(inductive biases) 지식을 활용해 게임 플레이를 배우는 접근 방법을 다룹니다.

- **Performance Highlights**: 제안된 방법론은 인간과 유사한 객체 상호작용을 통해 여러 비디오 게임을 효과적으로 학습할 수 있는 능력을 보여주었으며, 특히 낯선 객체에 대해 뛰어난 일반화 성능을 나타냈습니다. 이러한 연구 결과는 기계가 인간 중심으로 학습할 수 있는 새로운 가능성을 제시합니다.



### Deep Learning-Based Channel Squeeze U-Structure for Lung Nodule Detection and Segmentation (https://arxiv.org/abs/2409.13868)
- **What's New**: 이번 논문은 폐 결절의 자동 감지 및 분할을 위한 새로운 딥러닝(deep-learning) 방법을 제안하며, 이를 통해 조기 폐암 진단의 정확성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 제안된 접근법은 독특한 'Channel Squeeze U-Structure'를 활용하여 네트워크의 여러 의미 수준에서 기능 추출(feature extraction) 및 정보 통합(information integration)을 최적화합니다. 이 아키텍처는 얕은 정보 처리(shallow information processing), 채널 잔여 구조(channel residual structure), 채널 압축 통합(channel squeeze integration)이라는 세 가지 주요 모듈로 구성되어 있습니다.

- **Performance Highlights**: 이 방법은 민감도(sensitivity), Dice 유사도 계수(Dice similarity coefficient), 정밀도(precision) 및 평균 교차 점유율(mean Intersection over Union, IoU) 측면에서 뛰어난 성능을 보여줍니다. LIDC(Lung Image Database Consortium) 데이터셋을 사용한 다섯 번 교차 검증(five-fold cross-validation) 실험에서 안정성과 강인성이 우수하다는 결과를 보였습니다.



### AutoPET III Challenge: Tumor Lesion Segmentation using ResEnc-Model Ensemb (https://arxiv.org/abs/2409.13779)
- **What's New**: 이번 연구에서는 다양한 암 진단을 위한 Positron Emission Tomography (PET)와 Computed Tomography (CT) 영상을 활용하여, 다중 트레이서(multi-tracer) 멀티 센터 환경에서의 종양 병변(segmentation) 분할을 위한 신뢰할 수 있는 딥러닝 모델 개발을 목표로 하였습니다.

- **Technical Details**: 3D Residual encoder U-Net을 사용한 autoPET III 도전 과제가 진행되었으며, Total Segmentator를 통한 데이터 전처리 및 데이터 증강(data augmentation) 기법을 적용하여, 훈련 데이터의 품질을 향상시켰습니다. nnU-Net ResEnc XL 아키텍처를 활용하여 1,611개의 이미지를 대상으로 훈련하였고, Dice 점수로 성능 평가를 하였습니다.

- **Performance Highlights**: 최종 평가에서, 2D 단일 모델이 0.9627의 Dice 점수를 기록하며, 5-fold 앙상블 모델 역시 0.9602를 기록하였습니다. 3D 모델은 0.7530의 Dice 점수를 달성하여, 2D 모델들에 비해 성능이 떨어졌으나, 3D 모델 앙상블을 통한 접근이 유효함을 보였습니다.



### Efficient Classification of Histopathology Images (https://arxiv.org/abs/2409.13720)
Comments:
          12 pages, 2 figures, Accepted paper for the 27th International Conference on Pattern Recognition (ICPR) 2024

- **What's New**: 이 연구는 기가픽셀(gigapixel) 전체 슬라이드 이미지를 효과적으로 분류하는 방법을 다룹니다. 특히, 암 진단을 위한 이미지 레벨 주석에서 도전적인 조직병리학 이미지를 처리하는 데 집중합니다.

- **Technical Details**: 본 연구에서는 암 슬라이드에서 양성(tumor) 및 악성(benign) 패치를 분류하기 위한 세 가지 하위 문제를 정의합니다. 정보 이론적 클러스터 기반 샘플링을 통해 적극적인 패치를 샘플링하고, 이를 활용하여 새로운 깊은 모델을 구축하여 최종 결정을 내립니다. 또한, z-score 기반의 층화 샘플링 방법을 통해 데이터의 다양성을 극대화합니다.

- **Performance Highlights**: 제안된 방법은 전체 슬라이드 이미지에서 사용 가능한 패치의 아주 작은 비율을 사용하여 경쟁력 있는 성능을 보여줍니다.



### A Stochastic Geo-spatiotemporal Bipartite Network to Optimize GCOOS Sensor Placement Strategies (https://arxiv.org/abs/2404.14357)
Comments:
          7 pages, 6 figures, 2022 IEEE International Conference on Big Data (Big Data)

- **What's New**: 이 논문은 공간 이분 네트워크 모델에서 적용 가능한 새로운 두 가지 측정치인 coverage와 coverage robustness를 제안합니다. 이 네트워크는 관찰자 노드와 관찰 가능한 노드, 그리고 이들을 연결하는 엣지로 구성되어 있습니다. 이 측정치는 관찰자 노드 배치의 효과성을 평가하는 데 유용합니다.

- **Technical Details**: 논문에서는 Gulf of Mexico의 확률론적 및 동적 환경에서 Geo-SpatioTemporal Bipartite Network (GSTBN)를 구성합니다. GSTBN은 GCOOS 센서 노드와 HYCOM Region of Interest (RoI) 이벤트 노드로 구성되어 있으며, GCOOS의 확장을 통해 HYCOM 해양 예측 모델의 예측 결과를 개선하는 최적 배치를 식별하는 것이 목표입니다.

- **Performance Highlights**: 이 연구는 현재의 센서 배열을 가장 잘 보완할 수 있는 새로운 센서의 최적 배치를 식별하고자 하며, 이를 위해 Monte Carlo 시뮬레이션을 통해 GCOOS 노드 추가의 효과성을 평가합니다. 이 방법으로 GCOOS의 저커버리지 문제를 해결하는 것이 기대됩니다.



### Boosting Federated Domain Generalization: Understanding the Role of Advanced Pre-Trained Architectures (https://arxiv.org/abs/2409.13527)
- **What's New**: 이 연구에서는 Vision Transformers (ViT), ConvNeXt, Swin Transformers와 같은 고급 사전 훈련 아키텍처의 효능을 탐구하여 Federated Domain Generalization (FDG)을 향상시키고자 하였습니다. 이러한 아키텍처는 전역 컨텍스트 특성을 포착하고 장거리 종속성을 모델링하여 크로스 도메인 일반화를 개선할 수 있는 유망한 후보로 평가받고 있습니다.

- **Technical Details**: 이 연구에서는 ImageNet-1K, ImageNet-21K, JFT-300M, ImageNet-22K와 같은 광범위한 사전 훈련 데이터셋을 사용하여 다양한 아키텍처의 변종을 체계적으로 평가하였습니다. 자가 감독(Self-supervised)과 감독(Supervised) 사전 훈련 전략을 비교하여 FDG 성능에 미치는 영향을 분석하였습니다. 연구 결과, 마스크된 이미지 패치 재구성을 중점적으로 다루는 자가 감독 기술이 감독 대안보다 더 나은 성능을 보임을 발견하였습니다.

- **Performance Highlights**: Office-Home 및 PACS 데이터셋에 대한 종합 평가 결과, 큰 데이터셋에서 사전 훈련된 고급 아키텍처를 채택한 결과 각각 84.46% 및 92.55%의 평균 정확도를 달성하여 새로운 벤치마크를 수립했습니다. 또한 몇 가지 고급 모델의 변종이 적은 매개변수를 가지면서도 더 큰 ResNet 모델보다 뛰어난 성능을 보였으며, 효율성과 모델 용량이 중요한 상황에서 FDG 성능을 향상시키기 위한 복잡한 아키텍처 및 다양한 사전 훈련 전략의 중요성을 강조했습니다.



### Exploiting Minority Pseudo-Labels for Semi-Supervised Semantic Segmentation in Autonomous Driving (https://arxiv.org/abs/2409.12680)
Comments:
          17 pages, 8 figures

- **What's New**: 본 논문은 자율주행 기술의 발전과 관련하여 시맨틱 세그멘테이션에서의 클래스 불균형 문제를 해결하기 위한 새로운 Synergistic Training framework (협력 훈련 프레임워크)를 제안합니다. 이 프레임워크는 전문 훈련 모듈과 일반 훈련 모듈로 구성되어 있으며, 소수 클래스의 학습을 향상시키고 불확실한 pseudo-label의 영향을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: Synergistic Training framework (STPG)는 두 가지 훈련 모듈로 이루어져 있습니다. 전문 훈련 모듈(professional training module)은 예측이 일관되거나 크게 불일치하는 pseudo-label을 선택하여 소수 클래스 정보를 학습합니다. 일반 훈련 모듈(general training module)은 전문 교사(professional teacher)로부터 모든 pseudo-label을 받아 이를 통해 더 포괄적인 의미 정보를 학습합니다. 또한, dual contrastive learning을 채택하여 서로 다른 클래스 간의 결정 경계를 강조합니다.

- **Performance Highlights**: 제안된 프레임워크는 벤치마크 데이터셋에서 최신 기술들(state-of-the-art)보다 우수한 성능을 보였습니다. 이는 시맨틱 세그멘테이션 분야에서 모델 성능을 크게 향상할 수 있는 가능성을 제시합니다.



New uploads on arXiv(cs.AI)

### LLM Echo Chamber: personalized and automated disinformation (https://arxiv.org/abs/2409.16241)
Comments:
          42 pages

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)인 GPT-4와 Llama2가 잘못된 정보를 대량으로 전파할 수 있는 잠재적 위험성을 중점적으로 분석했습니다.

- **Technical Details**: LLM Echo Chamber라는 제어된 디지털 환경을 구축하여 LLM이 잘못된 정보의 진위를 주장하는 과정을 실험하였으며, Microsoft의 phi2 모델을 커스터마이징하여 정보 전파 경로를 연구했습니다.

- **Performance Highlights**: 실험 결과, LLM이 생성하는 잘못된 정보의 설득력과 악영향을 분석하였으며, LLM의 안전성을 강화하고 윤리적 기준을 수립할 필요성을 강조했습니다.



### Efficiently Learning Probabilistic Logical Models by Cheaply Ranking Mined Rules (https://arxiv.org/abs/2409.16238)
Comments:
          21 pages

- **What's New**: 이 논문에서는 관계형 데이터로부터 논리 모델을 학습하기 위한 새로운 프레임워크인 SPECTRUM을 소개합니다. 이 프레임워크는 비용 효율적인 규칙 유용성(rule utility) 측정을 통해 논리 모델의 예측력을 평가할 수 있게 합니다.

- **Technical Details**: SPECTRUM은 반복적인 데이터 구조를 검색하는 선형 시간(linear-time) 알고리즘과 더불어, 유용성 측정치를 활용하여 규칙을 효율적으로 정렬하는 두 번째 알고리즘을 사용합니다. 또한, 논리 모델의 유용성에 대한 이론적 보장을 제공합니다. 논문에서는 저비용 유용성 측정, 선형 시간 패턴 마이닝 및 이차 시간 최적화(quadratic-time optimization) 알고리즘을 통해 스케일러빌리티(scalability) 문제를 해결합니다.

- **Performance Highlights**: SPECTRUM은 이전 방법들과 비교하여 실제 데이터셋에서 정확한 논리 모델을 훨씬 빠른 속도로 학습하며 최대 19%의 정확도 향상을 보여주었습니다. 이전 최신 기술에 비해 실행 시간을 1% 이하로 줄였습니다.



### CJEval: A Benchmark for Assessing Large Language Models Using Chinese Junior High School Exam Data (https://arxiv.org/abs/2409.16202)
- **What's New**: 본 논문에서는 중국 중학교 시험 평가(CJEval)를 기반으로 한 새로운 벤치마크를 소개하고 있습니다. 이 벤치마크는 26,136개의 샘플이 포함되어 있으며, 4가지 교육적 응용 과제와 10개 과목을 아우릅니다. 또한, 문제 유형, 난이도, 지식 개념 및 답변 설명과 같은 세부 주석 정보도 제공합니다.

- **Technical Details**: CJEval 벤치마크는 시험 질문을 중심으로 구성된 데이터 세트로, 지식 개념 태그 지정(Knowledge Concept Tagging), 질문 난이도 예측(Question Difficulty Prediction), 질문 응답(Question Answering), 질문 생성(Question Generation)과 같은 4개의 핵심 작업을 포함합니다. 다양한 교육 과제에 대해 LLM(대형 언어 모델)을 미세 조정을 통해 평가했습니다.

- **Performance Highlights**: 광범위한 실험을 통해 LLM의 교육 분야에서의 잠재적 응용과 한계를 파악하였고, 이를 통해 교육 LLM의 적용 가능성과 도전 과제를 논의했습니다. 특히, 문제 유형이 다양한 평가는 단순한 객관식 문제에 의존하는 기존 벤치마크와 차별화됩니다.



### Leveraging Estimated Transferability Over Human Intuition for Model Selection in Text Ranking (https://arxiv.org/abs/2409.16198)
Comments:
          Accepted by EMNLP 2024 main conference

- **What's New**: 이 논문은 Transferability Estimation (TE) 방법을 텍스트 순위 (text ranking) 작업에 적합하게 조정하여 모델 선택 문제를 해결하는 새로운 접근 방식을 제안합니다. 특히, Adaptive Ranking Transferability (AiRTran)라는 방법을 통해 예상 순위를 계산하여 모델의 순위 성능을 명확하게 반영합니다.

- **Technical Details**: 제안된 AiRTran 방법은 isotropic한 문장 임베딩을 적응적으로 스케일링하여 훈련 동역학을 통합합니다. 이를 통해 더 정확한 예상 순위 점수를 얻고 모델의 진정한 전이 가능성을 반영할 수 있습니다. 방법론에서는 적응형 비등방성 (Adaptive Isotropization, AdaIso)을 활용하여 문서의 순위를 향상시키고 있습니다.

- **Performance Highlights**: AiRTran은 다양한 텍스트 순위 데이터셋에서 두 가지 모델 후보 풀(작고 큰 후보 PLMs)로 평가되었으며, 브루트 포스 방법이나 기존의 TE 방법, 인지적 판단과 ChatGPT보다 현저한 개선을 보여줍니다. 이 방법은 텍스트 순위 작업에 대한 모델 선택에서 매우 효율적이며 뛰어난 성능을 발휘합니다.



### EnIGMA: Enhanced Interactive Generative Model Agent for CTF Challenges (https://arxiv.org/abs/2409.16165)
- **What's New**: 이 논문은 EnIGMA라는 새로운 언어 모델(Language Model, LM) 에이전트를 소개합니다. EnIGMA는 자율적으로 Capture The Flag (CTF) 도전과제를 해결할 수 있는 능력을 개발하였습니다.

- **Technical Details**: EnIGMA는 Agent-Computer Interfaces (ACIs)라는 새로운 개념을 도입하여 CTF 과제를 해결하는 성공률을 향상시킵니다. 이 논문의 핵심은 Interactive Agent Tool 개념을 설정하여 LM이 이 도전과제에 필수적인 대화형 명령줄 유틸리티(command-line utilities)를 실행할 수 있게 합니다.

- **Performance Highlights**: EnIGMA는 세 가지 서로 다른 벤치마크에서 350개 이상의 CTF 도전과제를 실험한 결과, 새로운 도구 세트를 제공하고 그 사용법을 시연함으로써 복잡한 문제를 해결하는 데 도움을 주며, NYU CTF 및 Intercode-CTF 벤치마크에서 최첨단(results) 성과를 달성했습니다.



### Implicit assessment of language learning during practice as accurate as explicit testing (https://arxiv.org/abs/2409.16133)
- **What's New**: 이번 연구에서는 Intelligent Tutoring Systems (ITS)에서 학습자의 능력을 평가하기 위해 Item Response Theory (IRT)를 활용합니다. 기존의 포괄적인 테스트 방식 대신, 효율적이면서도 정확한 적응형 테스트(adaptive tests) 개발을 목표로 하고 있습니다.

- **Technical Details**: 연구는 학습자로부터 수집된 데이터를 바탕으로 IRT 모델을 훈련시키고, 이를 통해 적응형 테스트를 안내하는 방식을 사용합니다. 또한, 연습 세션(exercise sessions) 중에 수집된 데이터를 IRT 모델링에 적합한 형태로 변환하는 과정을 진행하며, 언어적 구성(linguistic constructs)을 '항목(items)'으로 연결하여 IRT 모델에 통합합니다.

- **Performance Highlights**: 대규모 연구 결과, 교사의 학습자 능력 평가를 '기준 진리(ground truth)'로 삼고, 테스트와 연습을 통해 얻은 능력 추정치를 비교한 결과, IRT 모델이 연습 기반의 능력 추정에서도 정확성을 발휘함을 확인했습니다.



### Analyzing Probabilistic Methods for Evaluating Agent Capabilities (https://arxiv.org/abs/2409.16125)
- **What's New**: AI 시스템의 위험을 완화하기 위해, Phuong et al.은 두 가지 방법을 제안했습니다. 첫 번째 방법은 task를 subtasks로 나누어 성공률 추정치를 향상시키는 milestone method이며, 두 번째 방법은 인간의 지침을 통해 모델의 성능을 추정하는 expert best-of-N method입니다.

- **Technical Details**: 이 연구에서는 두 방법을 Monte Carlo estimator 관점에서 분석하였습니다. 두 방법 모두 naive Monte Carlo sampling에 비해 variance를 효과적으로 줄였지만 bias를 도입한다는 결과를 보였습니다. milestone method는 여러 실제 작업에 대한 진정한 해결률을 과소평가하는 경향이 있으며, expert best-of-N method는 모든 작업에서 더 심각한 과소평가를 보입니다.

- **Performance Highlights**: 이 방법들은 바람직하게도 variance를 줄였지만, 실제 작업에서는 성공 확률을 과소평가하여 실용성을 크게 제한합니다. 따라서 향후 연구에서는 Monte Carlo estimator 문헌을 활용하여 AI 에이전트의 성공률 추정 방법을 개발할 필요성이 있습니다.



### LTNtorch: PyTorch Implementation of Logic Tensor Networks (https://arxiv.org/abs/2409.16045)
Comments:
          5 pages, 2 figures

- **What's New**: Logic Tensor Networks (LTN)는 딥러닝과 논리적 추론(logical reasoning)을 효과적으로 통합하는 Neuro-Symbolic 프레임워크입니다. LTN은 논리적 지식 베이스를 정의하고 이를 신경망 모델의 목표로 사용하여 논리적 추론에 의한 학습을 가능하게 합니다.

- **Technical Details**: LTN은 특정 일차(logic) 언어인 Real Logic을 사용하여 지식 베이스를 정의합니다. 이 프레임워크는 텐서(tensor)로의 매핑(mapping) 및 퍼지 논리(fuzzy logic) 의미론을 적용하여 학습을 최적화합니다. LTNtorch는 LTN의 PyTorch 구현으로, 논리적 손실 함수(logical loss functions)를 명시하고 최소화하는 과정을 포함합니다.

- **Performance Highlights**: LTN은 훈련 데이터를 기반으로 공리(axiom)를 평가하고 손실 함수를 계산한 후, 신경망의 파라미터를 조정하여 지식 베이스가 최대한 만족되도록 하는 과정으로 학습합니다. 이 논문에서는 LTN의 공식화와 LTNtorch의 구현 방법을 제시하며, 기본 이진 분류(binary classification) 예제를 제공합니다.



### Bridging Environments and Language with Rendering Functions and Vision-Language Models (https://arxiv.org/abs/2409.16024)
- **What's New**: 이 논문에서는 VLM (Vision-Language Models)을 활용하여 LCAs (Language-Conditioned Agents)의 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다. 특히, MTRL (Multi-Task Reinforcement Learning) 한계를 극복하기 위해 환경 구성을 먼저 찾고 그에 맞는 목표 지향 정책(goal-conditioned policy)을 사용하는 방법을 소개합니다.

- **Technical Details**: 논문에서는 LCAs를 구축하는 과정을 두 단계로 나누어 다룹니다. 첫 번째 단계에서는 작업을 설명하는 텍스트에 대해 높은 VLM 점수를 가지는 환경 구성(configuration)을 찾고, 두 번째 단계에서는 사전 훈련된 목표 지향 정책을 사용하여 해당 구성을 달성하는 방식으로 진행합니다. 또한, 모델의 속도와 품질을 향상시키기 위한 여러 가지 기법, 특히 distilled models의 활용과 다양한 관점에서 구성의 평가를 수행하여 2D 뷰의 모호함을 해결하는 방법을 탐구합니다.

- **Performance Highlights**: Humanoid 환경에서 이 방법을 적용한 결과, MTRL의 기준 모델(baseline) 대비 제로샷 제너럴리제이션(zero-shot generalization)에서 더 우수한 성능을 달성했습니다. 이 과정에서 텍스트 작업 설명이나 환경 특정 주석(annotation)이 전혀 필요하지 않았습니다.



### Artificial Human Intelligence: The role of Humans in the Development of Next Generation AI (https://arxiv.org/abs/2409.16001)
Comments:
          34 pages, 5 figures, submitted to IEEE Trans. on Artificial Intelligence

- **What's New**: 인간 지능과 기계 지능 간의 상호작용이 급증함에 따라, 이를 둘러싼 윤리적이고 책임감 있는 지능 시스템 개발의 중요성이 강조되고 있습니다. 또한, 인공지능(AI)의 다음 세대 발전 방향에 대한 인간 중심의 관점을 제안하며, 인간과 기계의 상호작용을 기반으로 한 ‘인공지능 인간 지능’(artificial human intelligence)의 개념을 도입합니다.

- **Technical Details**: 논문은 인간의 뇌 발달과 신경과학(neuroscience)의 메커니즘을 참고하여 AI 시스템의 진화 과정에서 인간의 역할을 강조합니다. 이를 통해 인공지능의 정의 및 다양한 형태의 지능 학습의 방향성을 모색하고, 현재 AI 시스템이 인간의 가치 및 비판적 사고를 어떻게 반영하는지를 탐구합니다.

- **Performance Highlights**: 기계 지능이 다양한 복잡한 업무를 수행할 수 있다는 점은 명백하지만, 인간의 참여와 개입이 다음 세대 AI 시스템의 궤적을 결정짓는 데 필수적이라는 것입니다. AI의 지속적인 발전과 함께 인간과 기계가 협조하여 생성할 수 있는 새로운 인지 기술 및 발명이 기대됩니다.



### DataGpt-SQL-7B: An Open-Source Language Model for Text-to-SQL (https://arxiv.org/abs/2409.15985)
- **What's New**: 본 논문에서는 자연어 질의를 SQL 명령어로 변환하는 문제의 중요성을 강조하며, 비전문가도 데이터에 접근하고 분석할 수 있도록 돕기 위한 compact하고 fine-tuned한 모델과 self-refine 메커니즘을 제안합니다. 데이터 접근에 있어 closed-source Large Language Models(LLM)의 위험을 완화하기 위한 접근법도 포함되어 있습니다.

- **Technical Details**: 우리는 20,000개 이상의 Text-to-SQL 샘플로 구성된 데이터셋을 구축하고, 코드의 유효성을 보장하기 위해 코드 수정을 통합한 DataGpt-sql 시스템을 개발했습니다. 또한, cross-DataBase 및 Inner-DataBase 방법을 통해 올바른 스키마와 열의 정보를 식별하는 능력을 개선했습니다. Direct Preference Optimization(DPO)를 활용하여 모델을 추가로 fine-tuning 하였습니다.

- **Performance Highlights**: DataGpt-sql 시스템은 spider-dev 벤치마크에서 각각 87.2%(EX) 및 83.5%(TS)의 정확도를 달성하며, 기존의 pure model은 84.8%(EX) 및 81.5%(TS)의 정확도를 보였습니다. 이는 text-to-SQL 변환 작업에서 우리의 솔루션의 효과성을 입증합니다.



### TSFeatLIME: An Online User Study in Enhancing Explainability in Univariate Time Series Forecasting (https://arxiv.org/abs/2409.15950)
- **What's New**: 본 논문에서는 시계열 예측을 위한 새로운 프레임워크인 TSFeatLIME을 제안합니다. 이 프레임워크는 기존의 TSLIME을 확장한 것으로, 단변량 시계열 예측을 위한 설명 가능성(Explainability)을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: TSFeatLIME은 대체 특성을 보조 모형에 통합하고, 쿼리된 시계열과 생성된 샘플 간의 쌍별 유클리드 거리(Euclidean distance)를 고려하여 대체 모델의 신뢰도를 향상시킵니다. 이 방법론은 LIME(Local Interpretable Model-agnostic Explanation)에 기초하며, 특히 퍼뮤테이션(perturbation) 과정을 통해 시계열 데이터의 해석 가능성을 높입니다.

- **Performance Highlights**: 사용자 연구 결과, TSFeatLIME 프레임워크는 쿼리된 시계열 데이터에 대한 보다 나은 모의(simulation) 능력을 보여주었으며, 이러한 설명은 컴퓨터 과학 배경이 없는 참여자들에게 특히 효과적이었습니다. 이 연구는 160명의 참여자를 대상으로 두 개의 인터페이스를 통해 시각적인 결과를 측정하였으며, 결과적으로 신뢰도(trust)와 만족도(satisfaction)가 높아지는 경향을 보였습니다.



### Planning in the Dark: LLM-Symbolic Planning Pipeline without Experts (https://arxiv.org/abs/2409.15915)
Comments:
          8 main body pages, 10 appendix pages

- **What's New**: 본 논문은 자연어로 설명된 계획(Task) 작업을 해결하기 위한 새로운 접근 방식을 제안합니다. 특히, 기존의 LLM(symbolic planning) 방법론이 요구하는 전문가의 개입을 최소화하여, 완전히 자동화된 end-to-end LLM-기호 플래너를 구현하는 데 중점을 둡니다.

- **Technical Details**: 우리는 다양한 자연어 설명의 해석을 고려하여 여러 후보를 생성하는 동작 스키마(action schema) 라이브러리를 구축합니다. 또한, 생성된 스키마를 자동으로 필터링하고 순위를 매기는 의미적 검증(semantic validation) 및 랭킹 모듈을 도입하여 전문가 개입 없이도 신뢰할 수 있는 계획을 생성합니다.

- **Performance Highlights**: 실험 결과, 우리의 파이프라인은 전문가 개입 없이도 직접 LLM 기반 계획 생성 방식보다 나은 계획 성능을 유지하며, 모호한 자연어 설명에서 발생하는 다양한 해석을 보존할 수 있는 여러 스키마 집합과 계획 후보를 제공합니다.



### Enhancing IoT based Plant Health Monitoring through Advanced Human Plant Interaction using Large Language Models and Mobile Applications (https://arxiv.org/abs/2409.15910)
Comments:
          Pre-print Version. Submitted to conference

- **What's New**: 새로운 식물 소통 애플리케이션이 개발되어 실시간 센서 데이터를 바탕으로 식물이 인간과 소통할 수 있게 되었습니다. 이 시스템은 지면의 수분, 온도, 영양 수준을 모니터링하는 토양 센서를 활용하여 Gemini API를 통해 데이터를 처리하고 식물의 건강 상태와 '기분'에 대한 자연어 인사이트로 변환합니다.

- **Technical Details**: 애플리케이션은 Flutter, Firebase, ThingSpeak를 사용하여 개발되었습니다. Flutter는 다중 플랫폼 기능을 제공하며, Firebase는 데이터 저장 및 실시간 업데이트를 지원합니다. ThingSpeak는 IoT 센서 데이터의 수집 및 전송을 처리합니다. 이 시스템은 실시간 데이터 처리를 통해 사용자와의 상호 작용을 가능하게 하며, 사용자가 식물 건강을 추적하고 '기분'을 이해할 수 있도록 합니다.

- **Performance Highlights**: 이 앱은 사용자에게 직관적인 피드백을 제공하여 식물 관리 관행을 향상시키고 지속 가능성(promotes sustainability)을 촉진합니다. AI와 IoT 기술을 활용하여 개인적 및 농업적 맥락에서 혁신적인 애플리케이션을 도입하였으며, 농작물 관리 및 수확량 향상에도 기여할 수 있는 가능성을 보여줍니다.



### Five questions and answers about artificial intelligenc (https://arxiv.org/abs/2409.15903)
Comments:
          17 pages, 0 figures, Scientific and technological popularization article

- **What's New**: 이번 논문은 인공지능(AI)의 빠른 발전이 사회에서 회의와 논란을 일으키고 있다는 점을 다루고 있습니다. 특히 과학적 근거 없이 이루어지는 이 논란에 대한 해결책으로 R.W. 에머슨(R.W. Emerson)의 지식을 통한 두려움의 해소를 제안합니다.

- **Technical Details**: 논문은 AI의 기원(origin), 미래의 진화(possible future evolution), 감정을 표현할 수 있는 능력(ability to show feelings), 그리고 관련된 위협(threats)과 위험(dangers)에 대해 탐구합니다. 또한, AI의 특이점(singularity) 개념에 대해서도 성찰합니다.

- **Performance Highlights**: AI 기술에 대한 소양을 증가시키고자 하며, 사회적 두려움을 감소시키기 위해 보다 많은 지식을 제공하려는 의도를 갖고 있습니다.



### Symmetries and Expressive Requirements for Learning General Policies (https://arxiv.org/abs/2409.15892)
Comments:
          Accepted at the 21st International Conference on Principles of Knowledge Representation and Reasoning (KR2024) in the Reasoning, Learning, and Decision Making track

- **What's New**: 본 논문에서는 플래닝(planning)과 일반화된 플래닝(generalized planning)에서의 상태 대칭(symmetries) 탐지 문제를 다루고 있습니다. 상태 대칭 탐지는 검색 공간의 크기를 줄이는 데 중요한 역할을 하며, 이를 통해 학습의 효율성을 향상시킬 수 있습니다. 또한, 비대칭(non-symmetric) 상태를 구분하는 것이 중요하다고 강조하고 있습니다.

- **Technical Details**: 이 연구에서는 플래닝 상태를 평범한 그래프(plain graphs)로 매핑한 후, 목표에 대해 두 상태가 동형(isomorphic)인지 확인하기 위해 기존 그래프 알고리즘(off-the-shelf algorithms)을 사용합니다. 또한, 상태 비대칭성을 구분하기 위해 색칠 알고리즘(coloring algorithms)을 활용하여 C_2 특징이 비동형(non-isomorphic) 상태를 구분할 수 있는지를 평가합니다.

- **Performance Highlights**: 대칭 탐지의 결과는 학습의 효율성을 높이며, 비대칭을 탐지하지 못할 경우 특정 도메인에서 일반 정책(general policies)을 전혀 학습할 수 없음을 보여줍니다. 따라서, 다양한 플래닝 도메인에서 일반 정책 학습을 위한 표현 요구 사항(expressive requirements)을 평가하며, 실험적인 성과를 통해 향상된 학습 성과를 확인했습니다.



### In-Context Ensemble Improves Video-Language Models for Low-Level Workflow Understanding from Human Demonstrations (https://arxiv.org/abs/2409.15867)
Comments:
          multimodal in-context ensemble learning; video-language models; SOP generation; pseudo-labels

- **What's New**: 이 논문은 비디오-언어 모델을 사용하여 표준 운영 절차(Standard Operating Procedure, SOP) 생성을 자동화하는 방법을 탐구하고 있습니다. 특히, in-context learning을 활용한 SOP 생성의 가능성을 조사하며, 인-context ensemble learning을 제안하여 모델의 성능을 더 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 표준 운영 절차(SOP) 생성을 위한 연구에서 최신 비디오-언어 모델을 평가하는 내용이 포함되어 있으며, in-context learning이 포함된 멀티모달 in-context ensemble 학습 접근 방식이 소개됩니다. 이 방법은 비디오 입력과 텍스트 기반의 pseudo labels를 결합하여 모델이 더 많은 예제에서 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, GPT-4o-mini는 Gemini-1.5-flash와 Phi-3.5를 포함하여 다른 모델들에 비해 전반적으로 우수한 성능을 보였습니다. in-context learning은 모델의 단계 순서 예측 능력을 일관되게 향상시켰으며, 제안된 ICE 방법은 특히 Gemini-1.5-flash에서 recall을 9.22% 향상시키는 성과를 보여주었습니다.



### From Passive Watching to Active Learning: Empowering Proactive Participation in Digital Classrooms with AI Video Assistan (https://arxiv.org/abs/2409.15843)
- **What's New**: SAM (Study with AI Mentor)는 교육 동영상과 대화형 학습 환경을 통합한 혁신적인 플랫폼으로, 학습자가 실시간으로 질문을 던지고 불명확한 개념을 탐구할 수 있도록 지원합니다. 이 플랫폼은 대규모 언어 모델을 활용하여 개인화된, 맥락에 기반한 지원을 제공합니다.

- **Technical Details**: SAM은 비디오를 보면서 실시간으로 AI 멘토와 상호작용할 수 있게 설계되어 있으며, 사용자가 동영상과 관련된 질문을 할 수 있는 기능을 가지고 있습니다. 또한, SAM은 LaTeX를 사용하여 복잡한 수학 공식을 정확하게 표현할 수 있어 수학과 같은 과목에서 특히 유용합니다.

- **Performance Highlights**: 140명의 참가자가 참여한 사용성 연구에서 SAM 사용자는 96.8%의 답변 정확도로 지식 향상을 보였습니다. 참가자들은 SAM의 사용성과 효과성에 대해 긍정적인 피드백을 제공했으며, SAM은 학생들이 자신의 교육 경험에 주도권을 가질 수 있도록 돕는 것으로 평가되었습니다.



### SwiftDossier: Tailored Automatic Dossier for Drug Discovery with LLMs and Agents (https://arxiv.org/abs/2409.15817)
Comments:
          10 pages, 7 figures, 2 tables

- **What's New**: 이번 연구에서는 LLMs(대규모 언어 모델)가 약물 발견(drug discovery) 과정에서 보다 정확한 정보 생성을 지원할 수 있도록 고급 RAG(기억 증진 생성) 시스템을 통합하는 방법을 제시합니다.

- **Technical Details**: 고급 RAG 시스템을 LLM과 결합하여 약물 발견 관련 질문에 대한 정확한 답변 생성을 가능하게 합니다. 또한 LLM을 활용하여 외부 도구를 이용해 복잡한 작업을 수행하며 자동 타겟 도서(target dossier)를 생성하는 방법도 설명합니다.

- **Performance Highlights**: RAG 시스템을 통해 생성된 답변은 RAG가 없는 모델에서 생성된 답변의 품질을 초과하며, 최종적으로 수집된 정보를 PDF 및 PowerPoint 발표자료로 요약한 생산 준비 완료(target dossier)를 제공합니다.



### AsthmaBot: Multi-modal, Multi-Lingual Retrieval Augmented Generation For Asthma Patient Suppor (https://arxiv.org/abs/2409.15815)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 다국적, 다중 모드 리트리벌 증강 생성(Retrieval-Augmented Generation, RAG) 시스템인 AsthmaBot을 소개합니다. AsthmaBot은 최신 정보와 관련된 문서, 비디오 및 이미지를 통합하여 천식 관련 질문에 대한 답변을 제공합니다.

- **Technical Details**: AsthmaBot은 리트리벌 알고리즘(retrievers)과 다중 모드 지원 기능이 있는 대형 언어 모델(LLM)로 구성되어 있습니다. 이 시스템은 질문-답변 쌍을 바탕으로 정보를 제공하며, 사용자에게 텍스트, 이미지, 비디오가 포함된 응답을 제공합니다. FAQ 데이터를 통해 평가되었습니다.

- **Performance Highlights**: 실험 결과, AsthmaBot은 RAG 기법을 사용하지 않는 기본 모델에 비해 다양한 언어와 모드에서 우수한 성능을 보였습니다. 이 시스템은 사용자 인터페이스를 통해 일반 대중이 손쉽게 접근할 수 있도록 설계되었습니다.



### CLSP: High-Fidelity Contrastive Language-State Pre-training for Agent State Representation (https://arxiv.org/abs/2409.15806)
- **What's New**: 이번 연구에서는 인공지능의 빠른 발전에 따라 점점 중요해지고 있는 다중 모달 학습(multimodal learning) 분야에서, 상태(state) 표현의 발전이 필요하다는 점을 강조하고 있습니다. 특히, High-Fidelity Contrastive Language-State Pre-training (CLSP) 방법을 제안하여 강화 학습(reinforcement learning)과 다중 모달 대형 언어 모델(multimodal large language models) 모두에 활용할 수 있는 정보를 정확하게 인코딩합니다.

- **Technical Details**: CLSP는 두 단계로 구성됩니다. 첫 번째 단계에서는 다중 클래스 분류(supervised multiclass classification) 방법을 이용해 인코더를 미리 학습시키고, 이로부터 coarse-grained 정보를 확보합니다. 두 번째 단계에서는 대조 학습(contrastive learning)을 통해 CLSP 인코더와 텍스트 인코더 간의 정렬을 학습하여 더욱 정밀한 상태 정보를 표현합니다. 또한, Random Fourier Features (RFF) 방법을 통해 숫자 정보의 표현을 향상시킵니다.

- **Performance Highlights**: 체계적인 실험을 통해 CLSP의 우수한 정밀성과 일반화 능력이 입증되었습니다. 텍스트-상태 검색(task), 강화 학습 내비게이션(navigation) 작업, 다중 모달 대형 언어 모델 이해에서 향상된 성능을 달성했습니다. 결과적으로 CLSP는 RL 학습 속도를 증가시키고, 최종 수렴 값을 높이며, 다중 모달 LLM과의 스칼라 생성 오류를 줄이는 데 기여했습니다.



### Automated Assessment of Multimodal Answer Sheets in the STEM domain (https://arxiv.org/abs/2409.15749)
- **What's New**: 이 연구의 주요 업데이트는 STEM(Science, Technology, Engineering, Mathematics) 분야에서의 지속적인 과제가 되는 평가 자동화 방안을 제공하는 데 있습니다. 특히, 자동화된 평가 기술을 통한 효율적이고 신뢰할 수 있는 채점 방법의 개발에 중점을 두고 있습니다.

- **Technical Details**: 이 연구에서는 자연어 처리(Natural Language Processing) 기법과 고급 알고리즘을 활용하여 텍스트 응답 및 다이어그램 평가에 대한 효율적인 시스템을 개발했습니다. CRAFT 모델을 사용하여 텍스트 추출을 수행하고, YoloV5를 활용한 객체 감지와 Mistral-7B와 같은 대형 언어 모델(LLM)을 통한 텍스트 평가 시스템을 통합합니다. 또한, 흐름도를 텍스트 표현으로 변환하여 더 세밀한 평가를 수행합니다.

- **Performance Highlights**: 제안된 시스템은 학생 응답의 미묘한 차이를 효과적으로 분석하여 높은 정확도로 채점할 수 있습니다. 연구 결과, 자동 채점 시스템이 학생들의 이해도를 객관적으로 측정할 수 있도록 지원하며, 특히 수작업 조정의 필요성을 최소화하여 채점의 효율성을 높입니다.



### Real-Time Pedestrian Detection on IoT Edge Devices: A Lightweight Deep Learning Approach (https://arxiv.org/abs/2409.15740)
Comments:
          10 pages, 3 tables, 12 figures, article submitted to IEEE for possible publication

- **What's New**: 이번 연구는 경량화된 딥러닝(Deep Learning) 모델을 인공지능 사물인터넷(Artificial Intelligence of Things, AIoT) 에지 기기에 구현하여 보행자 탐지를 실시간으로 수행하는 방안을 제안합니다. 이를 위해 최적화된 유 아운리 룩 원(You Only Look Once, YOLO) 기반의 딥러닝 모델을 개발하였습니다.

- **Technical Details**: 연구에서는 에지 서버(edge server)의 한계인 제한된 처리 능력을 극복하기 위해 압축된 심층 신경망(Deep Neural Network, DNN) 모델을 활용합니다. 해당 경량화된 모델을 Nvidia Jetson Nano에 배포하여 실시간 보행자 탐지 테스트를 실시하였으며, 결과적으로 147 밀리세컨드의 빠른 추론 속도와 78%의 정확도를 달성했습니다.

- **Performance Highlights**: 최적화된 YOLO 모델은 2.3 프레임 매 초에 78%의 정확도로 실시간 보행자 탐지를 수행하였으며, 이는 기존 모델보다 상당한 개선을 나타냅니다.



### A Comprehensive Evaluation of Large Language Models on Mental Illnesses (https://arxiv.org/abs/2409.15687)
- **What's New**: 이번 연구에서는 소셜 미디어 데이터를 활용하여 정신 건강 관련 다양한 과제에서의 LLM(대형 언어 모델)의 성능을 종합적으로 평가하였습니다. 특히 Zero-Shot(ZS) 및 Few-Shot(FS) 학습의 가능성을 고찰하였습니다.

- **Technical Details**: GPT-4, Llama 3, Gemini 등의 다양한 LLM 모델을 33개 시험하고, 9개의 주요 프롬프트 템플릿을 활용하여 이진 장애 탐지, 장애의 심각도 평가, 정신과 지식 평가 등의 과제를 수행하였습니다. 프롬프트 엔지니어링이 모델 성능 향상에 중요한 역할을 했으며, Mixtral 8x22b 모델과 Gemma 7b는 각각 20% 이상의 성능 향상을 보였습니다.

- **Performance Highlights**: GPT-4 및 Llama 3은 이진 장애 탐지에서 85%의 정확도로 우수한 성능을 보였으며, FS 학습은 모델의 정확성을 상당히 향상시켰습니다. Phi-3-mini 모델은 ZS에서 FS 학습으로 이동할 때 균형 있는 정확도가 6.80% 이상 개선되었고, 평균 평균 오류는 1.3 가량 줄어들었습니다. Llama 3.1 405b는 정신과 지식 평가에서 91.2%의 정확도로 최신 모델들이 구형 모델들보다 일반적으로 우수한 성능을 보여주었습니다.



### MMPT: Multimodal Prompt Tuning for Zero-shot Instruction Learning (https://arxiv.org/abs/2409.15657)
Comments:
          EMNLP 2024

- **What's New**: 본 연구에서는 Multimodal Prompt Tuning (MMPT) 접근 방식을 도입하여 Multimodal Large Language Models (MLLMs)의 효율적인 instruction tuning을 수행합니다. MMPT는 비주얼 및 텍스트 프롬프트를 효과적으로 통합하여 여러 모드 간의 피쳐 정렬을 촉진합니다.

- **Technical Details**: MMPT는 0.09%의 전체 파라미터를 조정하여 경쟁력 있는 성능을 발휘하고, 시각적 입력과 텍스트 입력에 각각 비주얼 프롬프트와 텍스츄얼 프롬프트를 추가하는 방식으로 작동합니다. 프롬프트 간의 상호 작용을 통해 두 모드 간의 피쳐 표현을 조화롭게 학습합니다.

- **Performance Highlights**: MMPT는 여러 멀티모달 평가 데이터셋에서 여러 최첨단 기준 모델들에 비해 우수한 성능을 보였으며, 파라미터 효율적인 fine-tuning 기법으로 그런 성과를 달성하였습니다.



### Synatra: Turning Indirect Knowledge into Direct Demonstrations for Digital Agents at Sca (https://arxiv.org/abs/2409.15637)
- **What's New**: 이 논문에서는 Synatra라는 접근 방식을 소개하며, 이는 간접 지식을 대규모로 직접 감독(supervision)으로 변환하는 방법입니다. 자동화된 데이터 수집의 한계를 극복하고, 인간이 만든 온라인 튜토리얼과 같은 간접 지식을 사용하는 방법에 대해 설명합니다.

- **Technical Details**: 연구진은 간접 지식의 다양한 유형을 정의하고, 이를 수집하기 위한 자원과 직접 시연(demonstrations)의 구조를 인코딩(encoding)하는 방법, 그리고 간접 지식을 직접 시연으로 변환하는 방법을 연구합니다. 100,000개의 합성 시연(synthetic demonstrations)을 사용하여 7B CodeLlama 모델을 파인튜닝(finetuning)했습니다.

- **Performance Highlights**: 새로운 에이전트는 Mind2Web, MiniWoB++, WebArena 세 가지 웹 기반 작업 벤치마크에서 비슷한 크기의 모든 모델을 초과했으며, WebArena와 Mind2Web에서 GPT-3.5를 초과하는 성능을 보였습니다. 또한 합성 시연의 비용은 인간 시연의 3%에 불과하지만, 제한된 도메인에서 수집된 동일 수의 인간 시연보다 더 효과적임을 입증했습니다.



### Physics Enhanced Residual Policy Learning (PERPL) for safety cruising in mixed traffic platooning under actuator and communication delay (https://arxiv.org/abs/2409.15595)
- **What's New**: 이 논문에서는 물리적 정보(physics-informed)를 활용한 제어 전략을 통해 강화 학습(RL) 기반의 컨트롤러를 개발하였으며, 이는 전통적인 선형 모델 및 RL 모델의 장점을 모두 포괄하고자 합니다.

- **Technical Details**: 제안된 Physics-Enhanced Residual Policy Learning (PERPL) 프레임워크는 물리적 요소가 모델 해석 가능성 및 안정성을 제공하며, 학습 기반의 Residual Policy가 환경 변화에 적응하여 물리 모델의 결정을 개선합니다.

- **Performance Highlights**: PERPL 기법을 적용한 실험 결과, 인공적으로 극단적인 조건 및 실제 선행 차량 궤적을 사용하는 상황에서 선형 모델이나 단독 RL보다 작은 헤드웨이 오류(headway errors)와 더 나은 진동 감쇠(oscillation dampening)를 달성하였습니다. 또한, CAVs의 PERPL 스킴 침투율이 증가할수록 전체 교통 진동도 감소했습니다.



### SEAL: Suite for Evaluating API-use of LLMs (https://arxiv.org/abs/2409.15523)
- **What's New**: 이번 논문에서는 LLM(대규모 언어 모델)의 API 사용 기능을 평가하기 위해 새로운 테스트베드인 SEAL을 소개합니다. SEAL은 기존 벤치마크를 표준화하고, API 검색 및 계획을 위한 에이전트 시스템을 통합하며, 실시간 API의 불안정성을 해결하기 위해 GPT-4 기반의 API 시뮬레이터를 도입합니다.

- **Technical Details**: SEAL은 API 검색, API 호출 및 최종 응답을 포함하는 종합 평가 파이프라인을 제공하고, 지속적인 벤치마크 업데이트를 통해 새로운 테스트 환경에 적응합니다. 이 테스트베드는 엔드 투 엔드 방식으로 LLM의 실제 API 사용을 평가하는 것을 목표로 합니다.

- **Performance Highlights**: SEAL은 다양한 실제 시나리오에서 LLM의 성능을 신뢰성 있게 비교할 수 있는 구조적 프레임워크를 제공하며, 비결정적 환경에서도 안정적인 평가를 가능하게 합니다.



### From Text to Treatment Effects: A Meta-Learning Approach to Handling Text-Based Confounding (https://arxiv.org/abs/2409.15503)
- **What's New**: 이 논문은 관찰 데이터에서 이질적인 치료 효과를 정확하게 추정하는 메타 학습(meta-learning)의 성능을 검사합니다. 특히, 혼란 변수(confounding variables)가 텍스트에 담겨 있을 때의 메타 학습자의 성능 개선을 보여줍니다.

- **Technical Details**: 본 연구에서는 T-learner, RA-learner, DR-learner, R-learner와 같은 네 가지 메타 학습자를 고려하며, 각 메타 학습자는 대신 구성된 매개변수(nuisance parameters)인 η^⁢(X)={μ^0⁢(X),μ^1⁢(X),μ^⁢(X),π^⁢(X)}를 기반으로 작동합니다. 혼란 변수를 텍스트 표현(pre-trained text representations)으로 표시하여 CATE(Conditional Average Treatment Effect) 추정의 효율성을 분석합니다.

- **Performance Highlights**: 실험 결과, 사전 훈련된 텍스트 표현을 사용하는 학습자는 표 형태의 변수만 사용하는 경우보다 특히 데이터가 충분할 때 CATE 추정치를 개선하였습니다. 그러나, 이러한 텍스트 임베딩(embeddings)의 얽힘(entangled) 특성으로 인해 이 모델들은 완벽한 혼란 변수 지식을 갖춘 메타 학습자와 동일한 성능을 보이지는 않았습니다.



### RAM2C: A Liberal Arts Educational Chatbot based on Retrieval-augmented Multi-role Multi-expert Collaboration (https://arxiv.org/abs/2409.15461)
- **What's New**: 본 연구에서는 Retrieval-Augmented Multi-role Multi-expert Collaboration (RAM2C) 프레임워크를 제안하여 고품질의 자유 예술 교육 대화를 자동으로 생성하고, 이 데이터를 통해 LLM(대형 언어 모델)을 조정하는 방안을 소개합니다.

- **Technical Details**: RAM2C 프레임워크는 T-Group(중국어 교사), P-Group(교육 심리학자), E-Group(윤리적 안전 전문가)의 세 가지 전문가 그룹을 구성하여 다중 역할과 다중 전문가 협업을 통해 HTS(인간화된 소통, 교수 전문성, 안전-윤리) 기준에 부합하는 교육 대화를 생성합니다.

- **Performance Highlights**: RAM2C를 통해 생성된 LLM은 특히 문학 교육에서 높은 개인화된 응답과 윤리적으로 안전한 교육 반응을 제공하며, 실험 결과 미세 조정된 모델이 GLM-4와 유사한 성능을 보였습니다.



### Steward: Natural Language Web Automation (https://arxiv.org/abs/2409.15441)
- **What's New**: Steward는 LLM(대형 언어 모델)을 활용한 웹 자동화 도구로서, 자연어 지시어를 기반으로 하고 웹 사이트와 상호작용을 수행하는 시스템입니다. 기존의 브라우저 자동화 도구와는 달리, Steward는 LLM의 자연어 처리 능력을 통합하여 사용자의 지시를 해석하고 웹 상에서 자연스럽게 행동합니다.

- **Technical Details**: Steward는 사용자로부터 자연어로 지시를 받고 웹사이트에서 작업을 수행하는 고속, 신뢰할 수 있으며 비용 효율적인 LLM 기반의 웹 자동화 시스템입니다. 이 시스템은 Playwright와의 통합을 통해 설계되었으며, 최소한의 사용자 입력으로 웹사이트에서 복잡한 작업을 자동으로 수행할 수 있습니다. 캐싱 메커니즘을 통한 성능 최적화로 실행 시간을 4.8초까지 단축시킬 수 있습니다.

- **Performance Highlights**: Steward는 평균 8.52초에서 10.14초의 실행 시간을 기록하며, 작업당 비용은 $0.028입니다. 캐싱 메커니즘을 통해 시간과 비용이 더욱 줄어들어 각각 4.8초와 $0.022로 감소합니다. 작업 완료 성공률은 40%이며, 사용자 지시 없이도 81.44%의 정확도로 행동을 수행할 수 있습니다.



### Fuzzy Rule based Intelligent Cardiovascular Disease Prediction using Complex Event Processing (https://arxiv.org/abs/2409.15372)
- **What's New**: 본 연구에서는 심혈관 질환(CVDs)에 대한 실시간 의사결정 지원을 제공하기 위해 퍼지 규칙 기반 시스템(fuzzy rule-based system)을 제안합니다. 이 시스템은 임상 데이터 모니터링을 통해 건강 매개변수를 효과적으로 분석합니다.

- **Technical Details**: 연구에서는 Apache Kafka와 Spark를 사용하여 데이터 스트리밍(data streaming)을 수행하고, Siddhi CEP 엔진(Complex Event Processing engine)을 통해 이벤트 처리(event processing)를 진행합니다. 퍼지 규칙은 WHO 기준과 임상 표준을 바탕으로 설계하여 예측의 정확성을 보장합니다.

- **Performance Highlights**: 검증 결과에 따르면, 1000개의 합성 데이터 샘플을 기반으로 성과를 평가한 결과, 샘플의 20%가 '매우 낮은 위험(Very Low Risk)', 15-45%가 '낮은 위험(Low Risk)', 35-65%가 '중간 위험(Medium Risk)', 55-85%가 '높은 위험(High Risk)', 그리고 75%가 '매우 높은 위험(Very High Risk)'으로 분류되었습니다.



### Cognitive phantoms in LLMs through the lens of latent variables (https://arxiv.org/abs/2409.15324)
- **What's New**: 이 연구는 대규모 언어 모델(LLMs)의 심리적 특성을 평가하는 기존 방법들의 유효성 문제를 다룹니다. 기존 인간을 기반으로 한 측정 도구의 적합성을 검토하며, LLM의 행동 이해에 필요한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구에서는 두 개의 검증된 성격 질문지를 사용하여 인간과 세 가지 LLM 간의 잠재 구조(personality latent structures)를 비교합니다. 연구 결과, 인간을 위해 설계된 질문지는 LLMs에서 유사한 구성 요소를 유효하게 측정하지 못하며, LLMs에 존재하지 않을 수 있는 특성을 강조합니다.

- **Performance Highlights**: 이 연구의 발견은 LLM에 대한 기존 간접 측정 및 심리적 특성 평가가 LLM의 실제 행동 및 특성을 정확히 반영하지 않을 수 있음을 나타냅니다. 이는 LLM에 대한 심리적 분석의 필요성과 함께 사적 성격 패턴들이 유해한 반응으로 이어질 수 있음을 지적합니다.



### Articulated Object Manipulation using Online Axis Estimation with SAM2-Based Tracking (https://arxiv.org/abs/2409.16287)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구는 동적 딥러닝 환경에서 상호작용 인식을 접목한 온라인 축 추정(closed-loop axis estimation) 파이프라인을 제안합니다. 기존의 오픈 루프(open-loop) 접근 방식의 한계를 극복하여 로봇의 조작 작업의 정밀성과 효율성을 크게 향상시킵니다.

- **Technical Details**: 제안된 방법은 세분화된 3D 포인트 클라우드(point clouds)에서 온라인 축 추정을 통해 상호작용 인식을 통합합니다. 특히, RGBManip 기법을 사용하여 물체의 경미한 움직임을 유도하고, Segment Anything Model 2 (SAM2)를 이용하여 동적 장면의 포인트 클라우드를 세분화합니다. 이를 통해 이동하는 물체의 부위를 마스킹하여 정확한 축 추정을 수행합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 전통적인 오픈 루프 방법에 비해 조작 정확도와 일반화 능력을 크게 향상시켰습니다. 특히, 문 열기 및 서랍 열기 작업과 같은 정밀 축 기반 조작이 필요한 작업에서 기존의 기법들을 능가하는 성능을 보여주었습니다.



### Fields of The World: A Machine Learning Benchmark Dataset For Global Agricultural Field Boundary Segmentation (https://arxiv.org/abs/2409.16252)
- **What's New**: 농업 모니터링 및 평가에서 중요한 역할을 하는 농작물 경계 데이터를 수집하는 비용을 절감하기 위해, 본 논문에서는 다양한 국가의 데이터를 포함한 새로운 기계 학습(Machine Learning, ML) 벤치마크 데이터셋인 'Fields of The World (FTW)'를 제안합니다.

- **Technical Details**: FTW 데이터셋은 24개국에서 수집된 70,462개의 샘플을 포함하며, 각 샘플은 다중 날짜, 다중 스펙트럴 Sentinel-2 위성 이미지와 함께 인스턴스 및 의미적 세분화 마스크가 쌍으로 제공됩니다. 이 데이터셋은 전 세계 농업 경관의 다양성을 반영하고 있으며 ML 모델의 성능을 향상시키기 위한 여러 기준 작업을 포함합니다.

- **Performance Highlights**: FTW 데이터셋으로 훈련된 모델은 다양한 국가에서 전이 학습 및 제로샷(Zero-shot) 성능이 우수하며, 실제 시나리오인 에티오피아의 Sentinel-2 장면에서 긍정적인 질적 성능을 보였습니다.



### Label-Augmented Dataset Distillation (https://arxiv.org/abs/2409.16239)
- **What's New**: 본 연구에서는 Label-Augmented Dataset Distillation (LADD)이라는 새로운 데이터셋 증류 프레임워크를 도입하였습니다. LADD는 라벨을 증강하여 데이터셋 증류를 개선하며, 이는 더 풍부한 의미를 포착하기 위해 각 합성 이미지에서 추가적인 밀집 라벨을 생성합니다.

- **Technical Details**: LADD는 두 가지 주요 단계로 이루어져 있습니다: 증류(distillation) 단계와 배포(deployment) 단계입니다. 증류 단계에서는 기존 증류 알고리즘을 사용하여 합성 이미지를 생성한 후 이미지 서브샘플링 알고리즘을 적용하여 각 합성 이미지에 대한 밀집 라벨을 생성합니다. 배포 단계에서는 글로벌 뷰 이미지와 원래 라벨, 그리고 로컬 뷰 이미지와 해당 밀집 라벨을 결합하여 다양한 학습 신호를 제공합니다.

- **Performance Highlights**: LADD는 기존 방법들보다 평균 14.9%의 정확도 향상을 달성했으며, 87% 적은 메모리를 사용하면서 5 IPC에서 6 IPC 기준을 지속적으로 초과했습니다. LADD는 또한 다양한 데이터셋과 모델 아키텍처에서 검증되었습니다.



### Predicting Deterioration in Mild Cognitive Impairment with Survival Transformers, Extreme Gradient Boosting and Cox Proportional Hazard Modelling (https://arxiv.org/abs/2409.16231)
Comments:
          Accepted to ICANN 2024

- **What's New**: 본 논문은 ADNI 코호트에서 경증 인지장애(MCI) 환자의 인지 저하를 예측하기 위해 생체대사학 데이터와 생존 트랜스포머(survival transformer) 및 극단 그래디언트 부스팅(Extreme Gradient Boosting) 모델을 결합한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 기계학습(machine learning) 및 트랜스포머 기반 기술을 활용하여 생존 분석(survival analysis)에서의 조기 탐지 및 개입의 정확도를 개선하는 가능성을 강조하고 있으며, 100회의 몬테 카를로 시뮬레이션을 통해 생존 기계 학습 모델이 기존의 Cox 비례 위험 모델보다 더 높은 평균 C-인덱스(performance C-index)를 달성했음을 보여줍니다.

- **Performance Highlights**: 제안된 생존 기계 학습 모델은 C-인덱스에서 각각 0.85 및 0.8의 평균 성과를 기록하며, 기존 모델보다 더 안정적이고 정확한 결과를 제공합니다.



### Fine-Tuning is Fine, if Calibrated (https://arxiv.org/abs/2409.16223)
Comments:
          The first three authors contribute equally

- **What's New**: 본 연구에서는 사전 학습된 모델(Foundation Model)에서 세부 클래스를 대상으로 한 파인튜닝(fine-tuning) 중 발생하는 문제를 체계적으로 분석합니다. 연구 결과, 파인튜닝된 모델이 정확도가 하락하는 주된 원인이 로짓 스케일의 불일치임을 밝혀내고, 이러한 문제를 해결하기 위해 단순한 후처리(calibration) 방법을 제안합니다.

- **Technical Details**: 사전 학습된 분류기를 세부 클래스에 맞춰 파인튜닝하면 일반적인 정확도가 손상되지만, 이 논문에서는 파인튜닝 후에도 특징 추출기(feature extractor)의 성능 수치가 개선됨을 확인하였습니다. NCM(Nearest Class Mean) 분류기를 통해 특징 품질을 평가한 결과, 파인튜닝 동안 실종된 클래스에 대한 특징 분리가 향상되었습니다. 로짓 스케일의 불일치가 피해를 주며, 이 문제는 후처리 방법으로 보완할 수 있습니다.

- **Performance Highlights**: 다수의 벤치마크(예: ImageNet)에서 후처리 보정(calibration)을 통해 파인튜닝된 모델의 성능이 현저히 향상되었으며, 이는 강력한 기준선 모델조차 능가했습니다. 연구 결과는 간단한 파라미터 조정만으로도 유의미한 성과를 달성할 수 있음을 보여줍니다.



### Towards Enhancing Linked Data Retrieval in Conversational UIs using Large Language Models (https://arxiv.org/abs/2409.16220)
Comments:
          This paper has been accepted at the 25th International Web Information Systems Engineering Conference (WISE 2024)

- **What's New**: 이 논문은 기존 정보 시스템과 LLMs(대형 언어 모델)의 통합을 통해 Linked Data(LD) 및 RDF(Ressource Description Framework) 트리플스토어에서 데이터를 추출하고 탐색하는 방법을 탐구합니다. 특히, 모델 재훈련 없이도 더 정확한 SPARQL 쿼리를 생성할 수 있는 대화형 사용자 인터페이스(UI)의 강화를 강조합니다.

- **Technical Details**: 본 연구에서는 ForestQB라는 새로운 툴킷을 사용하여 관찰적 LD 데이터로부터 정보를 추출하고, 이 툴킷은 챗봇과 폼 기반 GUI를 통합하여 SPARQL 쿼리를 구성하고 실행합니다. 연구의 초점은 LLMs의 자연어 이해 능력을 활용하여 RDF 엔티티 추출의 정확성을 향상시키는 것입니다.

- **Performance Highlights**: 본 연구의 결과, 제안된 방법론을 통해 시스템의 표현력과 사용자 쿼리에 대한 응답 정확성이 크게 향상되었습니다. 평가 결과는 LLMs가 복잡한 데이터 환경에서 엔티티 추출 및 사용자 인터랙션을 개선시킬 수 있는 가능성을 제시하고 있습니다.



### Problem-oriented AutoML in Clustering (https://arxiv.org/abs/2409.16218)
- **What's New**: 문제 지향적 오토ML(Problem-oriented AutoML, PoAC) 프레임워크는 전통적인 오토ML 솔루션의 단점을 해결하며 클러스터링 작업을 자동화하는 새로운 유연한 접근법을 소개합니다. PoAC는 클러스터링 문제, 내부 클러스터링 유효성 지수(Clustering Validity Indexes, CVIs) 및 메타 피처(meta-features) 간의 동적인 연결을 수립하여 사용자가 이러한 구성 요소를 특정 맥락 및 목표에 맞게 맞춤 설정할 수 있도록 합니다.

- **Technical Details**: PoAC의 핵심은 방대한 메타 지식 기반(meta-knowledge base)의 클러스터링 데이터셋과 솔루션으로 훈련된 대체 모델(surrogate model)을 사용하여 새로운 클러스터링 파이프라인의 질을 추론하는 것입니다. PoAC는 알고리즘에 구애받지 않으며, 추가 데이터나 재훈련 없이 다양한 클러스터링 문제에 원활히 적응할 수 있습니다.

- **Performance Highlights**: 실험 결과, PoAC는 다양한 데이터셋에서 최신 기술의 프레임워크보다 우수한 성능을 달성했으며, 데이터 시각화와 관련된 CVIs에서도 더 나은 결과를 보였습니다. 또한, PoAC는 데이터셋의 복잡성과 정의된 문제에 따라 전처리 단계를 추가하거나 제거하여 파이프라인 구성을 동적으로 조정하는 능력을 보여주었습니다.



### Facial Expression-Enhanced TTS: Combining Face Representation and Emotion Intensity for Adaptive Speech (https://arxiv.org/abs/2409.16203)
Comments:
          13 pages, 3 figures, accepted to ECCV Workshop ABAW(Affective Behavior Analysis in-the-wild)7 (to be appear)

- **What's New**: FEIM-TTS는 감정적 언어 표현을 합성하는 혁신적인 제로샷 텍스트-투-스피치(TTS) 모델로, 얼굴 이미지와 감정 강도에 맞춰 조정됩니다.

- **Technical Details**: FEIM-TTS는 Classifier-Free Diffusion Guidance를 사용하여 조건부 및 비조건부 훈련을 수행합니다. 이 모델은 LRS3, CREMA-D 및 MELD 데이터셋을 활용하여 훈련됐으며, 얼굴 표정과 감정 양을 조절함으로써 고품질의 스피치를 생성하도록 설계되었습니다.

- **Performance Highlights**: FEIM-TTS는 감정 모듈레이션에서의 능력을 입증하며, 시각적 장애인들이 웹코믹을 더욱 몰입해서 즐길 수 있도록 지원합니다. 이 모델은 사용자에게 더 역동적이고 매력적인 청각 경험을 제공할 수 있는 가능성을 보여줍니다.



### Second Order Bounds for Contextual Bandits with Function Approximation (https://arxiv.org/abs/2409.16197)
Comments:
          12 pages main, 33 pages total

- **What's New**: 본 연구는 Contextual Bandits (맥락 기반 밴딧) 문제에서 함수 근사(Function Approximation)를 사용할 때, 기존의 알고리즘보다 더 나은 후회의 한계를 제공하는 새로운 알고리즘을 개발했습니다. 특히, 이 알고리즘은 후회(Regret)가 시간 지평(Time Horizon)의 제곱근이 아닌, 측정 오차의 분산(Variance)의 합의 제곱근으로 감소하도록 개선되었습니다.

- **Technical Details**: 연구팀은 각 시간에 따른 보상의 측정 노이즈의 분산이 변화하고 매우 작더라도, Optimistic Least Squares 알고리즘의 후회가 시간 지평의 제곱근에 비례하여 증가하는 문제를 해결했습니다. 이들은 측정 분산이 알려지지 않았을 때의 Contextual Bandits 설정에서 후회 한계를 도출하는 알고리즘을 제안하였습니다.

- **Performance Highlights**: 제안된 알고리즘은 기존의 OFUL(Optimistic Least Squares)과 SquareCB 알고리즘보다 후회 한계가 더욱 개선된 결과를 보여주며, 통계적 복잡성 측정에 기반한 새로운 알고리즘 설계로 Contextual Bandits 문제의 실제 적용 가능성을 높였습니다.



### Cyber Knowledge Completion Using Large Language Models (https://arxiv.org/abs/2409.16176)
Comments:
          7 pages, 2 figures. Submitted to 2024 IEEE International Conference on Big Data

- **What's New**: 이 논문에서는 사이버-물리 시스템(CPSs) 내 IoT 통합으로 인해 증가한 새로운 사이버 공격의 위협에 대응하기 위한 방법론을 제시합니다. 특히 감정 및 요약 능력을 활용한 대규모 언어 모델(LLMs)을 사용하여 사이버 공격 지식 완성을 위한 자동화된 매핑 프로세스를 발전시키고 있습니다.

- **Technical Details**: 저자들은 CAPEC 공격 패턴과 MITRE ATT&CK ICS 기술 사이의 관계를 이해하고 모델링하기 위해 새로운 접근 방식을 개발하였습니다. 이 과정에서 수학적 임베딩 모델을 사용하여 비정형 텍스트를 벡터로 인코딩하고, 이를 활용해 머신 러닝 알고리즘으로 매핑을 생성합니다. 또한, Retrieval-Augmented Generation (RAG) 방법론을 적용하여 다양한 위협 패턴의 분류 체계 간의 구조화된 매핑을 생성합니다.

- **Performance Highlights**: 저자들은 제안된 RAG 기반 접근 방식이 전통적인 이진 분류 모델과 비교하여 더욱 정확한 매핑을 생성함을 보여주었습니다. 또한, 손으로 라벨링된 작은 데이터셋을 공개하여 CAPEC 공격 패턴과 ATT&CK ICS 기술 간의 관계를 검증할 수 있는 중요한 자원을 제공합니다.



### Merging LoRAs like Playing LEGO: Pushing the Modularity of LoRA to Extremes Through Rank-Wise Clustering (https://arxiv.org/abs/2409.16167)
- **What's New**: 본 논문은 LoRA(저랭크 적응)을 조합하여 대형 언어 모델(LLM)의 성능을 극대화하는 새로운 접근 방식을 소개합니다. 기존의 LoRA 합성 방법이 주어진 특정 작업에 최적화되어 추가 훈련을 필요로 하는 반면, 본 연구에서는 LoRA 구성 요소를 독립적인 최소 의미 단위(MSU)로 분해 및 재조립하는 방식을 제안합니다.

- **Technical Details**: 제안된 LoRA-LEGO 프레임워크는 여러 LoRA에서 MSU를 클러스터링하여 새로운 LoRA를 구성하는 과정을 포함합니다. 이 과정은 세 가지 주요 단계로 나누어집니다: (1) 후보 LoRA로부터 MSU 풀(pool)을 만들기, (2) 이 MSU 풀을 k 클러스터로 그룹화하기, (3) 클러스터의 중심을 활용해 병합된 LoRA 구성하기. 이를 통해 파라미터 간섭을 해결하면서 다양한 랭크의 LoRA를 유연하게 조합할 수 있습니다.

- **Performance Highlights**: LoRA-LEGO는 다양한 벤치마크에서 기존의 LoRA 병합 방법보다 우수한 성능을 보였습니다. 실험 결과, LoRA-LEGO는 목표 랭크 k에 맞춘 병합 LoRA를 구성할 수 있을 뿐만 아니라, 개별 LoRA에 적용 시에도 파라미터 감소를 통해 원래 모델과 유사한 성능을 달성할 수 있음을 보여주었습니다.



### Seeing Faces in Things: A Model and Dataset for Pareidolia (https://arxiv.org/abs/2409.16143)
- **What's New**: 본 연구에서는 인간과 머신 간의 face pareidolia (얼굴 패레이돌리아)에 대한 인식 차이를 조사하기 위해 새로운 데이터셋인 'Faces in Things'를 소개합니다. 이 데이터셋은 무작위로 생성된 이미지에서 인간이 인식한 얼굴 구조를 포함하고 있습니다.

- **Technical Details**: 이 연구는 5,000개의 웹 이미지로 구성된 'Faces in Things' 데이터셋을 사용하여 인간 얼굴 탐지 시스템의 성능을 분석합니다. 연구 결과는 최신 연구 모델인 RetinaFace를 사용하여 성과를 변별하며, 파리돌리아가 머신에서 어떻게 나타나는지를 탐구합니다.

- **Performance Highlights**: 최신 모델은 얼굴 패레이돌리아 탐지에서 인간의 성능에 비해 상당한 격차를 보였습니다. 연구는 이 격차의 약 절반이 동물 얼굴 탐지 모델을 미세 조정하는 것에서 개선될 수 있음을 보여줍니다. 또한, 'Goldilocks zone'이라고 불리는 조건들이 패레이돌리아를 유도할 수 있음을 실험으로 확인하였습니다.



### HA-FGOVD: Highlighting Fine-grained Attributes via Explicit Linear Composition for Open-Vocabulary Object Detection (https://arxiv.org/abs/2409.16136)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 본 논문은 Open-Vocabulary Object Detection (OVD) 모델에서 세부 속성을 강조하는 새로운 접근 방식을 제안하여 기존 모델의 성능을 향상시키는 방법을 소개합니다.

- **Technical Details**: 이 방법은 1) Attribute Word Extraction, 2) Attribute Feature Extraction, 3) Attribute Feature Enhancement의 세 가지 주요 프로세스로 구성됩니다. 강력한 LLM(대규모 언어 모델)을 이용해 입력 텍스트에서 속성 단어를 추출하고, 전략적으로 토큰 마스크를 조정하여 OVD 모델의 텍스트 인코더가 전역 텍스트와 속성 특정 피처를 추출합니다. 이 피처들은 선형 조합을 통해 새로운 속성 강조 피쳐로 통합됩니다.

- **Performance Highlights**: FG-OVD 데이터셋에서 실험한 결과, 제안된 방법이 다양한 OVD 모델의 세부 속성 인식 능력을 일관되게 향상시키며 새로운 최첨단 성능을 달성함을 입증하였습니다.



### MOSS: Enabling Code-Driven Evolution and Context Management for AI Agents (https://arxiv.org/abs/2409.16120)
- **What's New**: MOSS (llM-oriented Operating System Simulation)이라는 새로운 프레임워크를 도입하여 코드 생성과 동적 컨텍스트 관리 시스템을 통합함으로써 AI 에이전트의 적응성과 일관성을 향상시킴.

- **Technical Details**: MOSS는 Python 실행 컨텍스트를 유지하고 지역 변수를 격리하여 여러 상호작용 간의 일관성을 보장하는 메커니즘을 사용합니다. Inversion of Control (IoC) 컨테이너와 데코레이터를 활용하여 가장 낮은 지식 원칙을 적용하며, 이로 인해 에이전트가 구체적인 구현보다는 추상 인터페이스에 집중할 수 있게 합니다.

- **Performance Highlights**: MOSS 프레임워크는 에이전트 개발의 효율성과 기능을 향상시키며, Turing-complete 에이전트를 생성할 수 있는 새로운 가능성을 보여줍니다. 다양한 실제 사례를 통해 에이전트가 코드 생성을 통해 스스로의 역량을 확장할 수 있는 것을 입증하였습니다.



### Scenario of Use Scheme: Threat Model Specification for Speaker Privacy Protection in the Medical Domain (https://arxiv.org/abs/2409.16106)
Comments:
          Accepted and published at SPSC Symposium 2024 4th Symposium on Security and Privacy in Speech Communication. Interspeech 2024

- **What's New**: 이 논문에서는 질병 감지 및 모니터링에 사용되는 음성 녹음의 프라이버시 문제를 다루기 위해 새로운 접근 방식인 Scenario of Use Scheme를 제안합니다.

- **Technical Details**: 제안된 접근 방식은 Attacker Model과 Protector Model을 포함하며, 이는 음성 기밀성을 방어하기 위해 필요한 가정과 의료 전문가의 요구 사항을 명확히 하고 체계적으로 규정합니다.

- **Performance Highlights**: 구체적인 예로, 이 연구는 성별 추정 공격(gender inference attacks)에 대한 음성 데이터 보호 실험을 수행하였으며, 파킨슨병 검출( Parkinson's detection)의 유용성을 유지하는 방법을 제시합니다.



### Neuromorphic Drone Detection: an Event-RGB Multimodal Approach (https://arxiv.org/abs/2409.16099)
Comments:
          Accepted at NeVi Workshop at ECCV24

- **What's New**: 이번 연구에서는 드론 감지를 위한 새로운 모델을 제안하며, Neuromorphic 데이터와 RGB 데이터를 효과적으로 융합하여 정확한 탐지를 위한 멀티모달 접근 방식을 탐구합니다. 또한, NeRDD(Neuromorphic-RGB Drone Detection)라는 새로운 데이터셋을 공개하여 3.5시간 이상의 주석이 달린 멀티모달 녹화를 제공합니다.

- **Technical Details**: Neuromorphic 카메라는 전통적인 RGB 카메라에 비해 높은 속도 및 변화하는 조명 조건에서 뛰어난 성능을 보여줍니다. 이 연구에서는 스파이킹 네트워크와 같은 다양한 신경망 아키텍처를 조합하여 드론 탐지 정확도를 향상시킵니다. 또한, 두 데이터 스트림을 융합하는 다양한 전략을 비교하여 성능 최적화를 도모합니다.

- **Performance Highlights**: 실험 결과에 따르면, Neuromorphic 카메라와 RGB 데이터의 조합은 각각 분리된 경우보다 드론 탐지율을 더욱 향상시킵니다. NeRDD 데이터셋의 사용으로 드론 탐지의 정확성이 크게 증가했음을 확인하였습니다.



### The Digital Transformation in Health: How AI Can Improve the Performance of Health Systems (https://arxiv.org/abs/2409.16098)
Comments:
          This article has been accepted for publication in Health Systems & Reform, published by Taylor & Francis

- **What's New**: 이 논문에서는 인공지능(Artificial Intelligence, AI)과 강화학습(Reinforcement Learning, RL)을 활용한 새로운 디지털 건강 플랫폼을 소개하며, 해당 플랫폼이 수집한 데이터를 기반으로 개인화된 추천과 개입을 제공하고, 건강 시스템을 효율적으로 개선할 수 있는 방법을 제시합니다.

- **Technical Details**: 이 플랫폼은 다양한 디지털 건강 애플리케이션과 연결 가능하며, 실시간 모니터링 및 실험을 통해 사용자 맞춤형 반응을 제공하는 능력을 갖추고 있습니다. AI는 데이터를 기반으로 한 예측 분석을 통해 질병의 발병 예측 및 자원 배분을 최적화할 수 있습니다. 이는 다수의 데이터 소스와 디지털 도구를 통합하여 복합적인 건강 상태를 평가 및 관리할 수 있게 해줍니다.

- **Performance Highlights**: 자원 부족이 우려되는 저소득 국가(Low- and Middle-Income Countries, LMICs)에서 이 접근 방식이 건강 결과에 미치는 영향이 더욱 결정적일 수 있으며, 이는 고소득 국가(High-Income Countries, HICs)에서도 유사하게 효과를 볼 수 있습니다. 이 플랫폼은 건강 관리의 효율성을 높이고 궁극적으로 공공 건강 결과를 개선하는 데 기여할 것으로 기대됩니다.



### From Pixels to Words: Leveraging Explainability in Face Recognition through Interactive Natural Language Processing (https://arxiv.org/abs/2409.16089)
- **What's New**: 본 논문에서는 Face Recognition (FR) 모델의 해석 가능성(transformability)을 높이기 위해 모델 불가지론적 설명 가능한 인공지능(Explainable Artificial Intelligence, XAI)과 자연어 처리(Natural Language Processing, NLP) 기술을 결합한 상호작용형 프레임워크를 제안합니다. 이 프레임워크는 사용자와의 대화를 통해 다양한 질문에 정확하게 응답할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는, 프레임워크의 각 모듈에서 사용되는 기술의 세부사항을 포함하여 3개의 주요 모듈로 구성됩니다: (i) FR 시스템 및 신뢰도 추정, (ii) 설명 가능성 방법, (iii) NLP 기반의 사용자 친화적 질문-응답(QA) 인터페이스입니다. 이 시스템은 ArcFace 모델을 사용하여 얼굴 이미지를 비교하고, Probabilistic Interpretable Comparison (PIC) 스코어를 통해 유사성과 신뢰도를 평가합니다.

- **Performance Highlights**: 제안된 방법은 다양한 실험을 통해 FR 시스템 성능을 저하시키지 않으면서도 해석 가능성을 향상시키는 효과를 입증했습니다. 또한, 자가 분류에서의 사용자 질문을 통해 보다 정확한 정보를 제공하고, 민감한 애플리케이션에서의 의사 결정 투명성을 추가로 강화할 수 있습니다.



### Assessing Simplification Levels in Neural Networks: The Impact of Hyperparameter Configurations on Complexity and Sensitivity (https://arxiv.org/abs/2409.16086)
- **What's New**: 이 연구는 신경망의 다양한 하이퍼파라미터 구성에서의 단순화 특성을 이해하기 위한 실험적 연구이다. 특히 Lempel-Ziv 복잡성과 민감도에 미치는 영향을 조사하였다.

- **Technical Details**: 하이퍼파라미터로는 활성화 함수, 은닉층 수, 학습률을 조정했으며, MNIST 데이터셋을 활용하여 네트워크 출력의 복잡성과 입력 섭동에 대한 민감도를 평가하였다.

- **Performance Highlights**: 실험 결과, ReLU와 LeakyReLU 활성화 함수를 사용하는 네트워크는 높은 민감도를 보여주었고, Sigmoid와 Tanh를 사용하는 네트워크는 낮은 민감도를 보였다. 또한, 높은 학습률이 모델의 학습을 실패하게 하고 Lempel-Ziv 복잡도가 낮아지는 결과를 초래하는 것으로 나타났다.



### Online Multi-level Contrastive Representation Distillation for Cross-Subject fNIRS Emotion Recognition (https://arxiv.org/abs/2409.16081)
Comments:
          Accepted in ACMMM-2024 Workshop BCI. Codes are available at this https URL

- **What's New**: 이 논문에서는 기능적 근적외선 분광법(*fNIRS*) 신호를 사용한 감정 인식을 위한 새로운 방법인 *Online Multi-level Contrastive Representation Distillation*(OMCRD) 프레임워크를 제안합니다. 이 방법은 경량 모델을 요구하는 휴대용 장치의 필요에 대응하며, 다양한 주체 간의 생리적, 심리적 차이로 인한 감정 인식의 어려움을 극복하기 위해 설계되었습니다.

- **Technical Details**: OMCRD는 다수의 경량 네트워크들 간의 상호 학습을 촉진하여 복잡한 '교사' 모델의 의존성을 줄입니다. 또한, *Inter-Subject Interaction Contrastive Representation*(IS-ICR) 손실 함수를 사용하여 비슷한 자극을 받는 서로 다른 주체들로부터 학습한 지식을 활성화하여 상호 작용을 증진합니다. OMCRD는 다중 수준 (*multi-level*)의 *fNIRS* 특징 추출기를 활용하여 여러 뷰의 감정 특징을 추출합니다.

- **Performance Highlights**: 실험 결과 OMCRD는 감정 인식과 감정적 이미징 작업에서 최첨단 성능을 달성했습니다. 제안된 방법은 공개된 *fNIRS* 데이터셋을 바탕으로 그 효과성과 내구성을 입증하며, 경량화된 학생 모델의 최선의 성능을 보장합니다.



### Leveraging Mixture of Experts for Improved Speech Deepfake Detection (https://arxiv.org/abs/2409.16077)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 논문에서는 Mixture of Experts (MoE) 아키텍처를 활용하여 음성 딥페이크 탐지 성능을 향상시키는 새로운 접근 방식을 제안합니다. 이 방법은 여러 데이터셋에서 일반화 능력을 높이고, 모델의 적응성을 개선합니다.

- **Technical Details**: 제안된 MoE 기반 탐지기는 다양한 음성 딥페이크 데이터셋에 대한 전문성을 가집니다. 각 전문가는 고유의 데이터셋에서 훈련되고, 게이팅 네트워크는 동적으로 입력에 대한 전문가 가중치를 할당하여 탐지 성능을 최적화합니다. 두 가지 아키텍처인 표준 MoE와 향상된 MoE를 통해 각각의 입력에 대해 모든 전문가가 쿼리됩니다.

- **Performance Highlights**: 여러 실험 결과에서 제안된 MoE 접근 방식이 기존 단일 모델이나 앙상블 방법들에 비해 뛰어난 일반화 및 데이터 적응성을 보여주는 것을 입증했습니다.



### Towards Robust Object Detection: Identifying and Removing Backdoors via Module Inconsistency Analysis (https://arxiv.org/abs/2409.16057)
- **What's New**: 본 논문은 두 단계 객체 탐지 모델에서의 백도어(Backdoor) 탐지 및 제거 문제를 다룬 최초의 접근법으로, 객체 탐지 모델의 독특한 특성에 맞춘 새로운 백도어 방어 프레임워크를 제안합니다.

- **Technical Details**: 제안된 백도어 탐지 방법은 객체 탐지 모델의 두 주요 구성 요소인 Region Proposal Network (RPN)과 Region Classification Network (R-CNN) 간의 예측 불일치를 정량화하고 분석하여 백도어의 존재를 확인합니다. 제안된 백도어 제거 전략은 특정 모듈에 대한 재초기화와 소량의 깨끗한 데이터에 대한 전체 모델의 미세 조정을 포함합니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, 제안된 방법은 백도어 제거율에서 기존 방법에 비해 약 90% 개선을 달성하였으며, 깨끗한 데이터에 대한 정확도 손실은 4% 미만으로 제한되었습니다.



### Adversarial Watermarking for Face Recognition (https://arxiv.org/abs/2409.16056)
- **What's New**: 본 연구는 얼굴 인식 시스템에서 워터마킹(watermarking)과 적대적 공격(adversarial attacks) 간의 상호작용을 탐구하며, 적대적 워터마킹 공격(adversarial watermarking attack)이라는 새로운 위협 모델을 소개합니다.

- **Technical Details**: 워터마킹은 디지털 이미지를 통해 소유권을 주장하고 무단 변경을 모니터링하는 데 필수적인 기술입니다. 얼굴 인식 시스템에서 워터마킹은 데이터 무결성과 보안을 보장하는 데 중요한 역할을 하지만, 공격자는 워터마킹 프로세스에 간섭하여 인식 성능을 심각하게 저하시킬 수 있습니다. 본 연구는 CASIA-WebFace 데이터셋을 통해 적대적 워터마킹 공격이 얼굴 매칭 정확성을 최대 67.2%까지 감소시킬 수 있음을 보여줍니다.

- **Performance Highlights**: 적대적 워터마킹 공격의 적용으로 인해 워터마킹이 없는 상태에서는 이미지가 정확하게 인식되지만, 워터마킹이 적용된 후에는 인식 실패를 유발하는 중요한 취약점이 발견되었습니다.



### Whole-body end-effector pose tracking (https://arxiv.org/abs/2409.16048)
- **What's New**: 본 연구는 레그드 로봇의 팔과 이동성을 결합하여 복잡한 환경에서의 조작 능력을 향상시키기 위한 새로운 전체 몸체 강화 학습(RL) 프레임워크를 제안합니다. 기존 방법들이 다루지 못한 대규모 작업 공간과 거친 지형에서 최종 효과기(end-effector)의 포즈 추적의 한계를 극복했습니다.

- **Technical Details**: 제안된 방법은 지형 인지 샘플링 전략을 통해 로봇의 초기 구성과 최종 효과기 명령을 관리하며, 게임 기반 커리큘럼을 통해 로봇의 운영 범위를 확장하는 방식으로 설계되었습니다. 이 연구에서는 ANymal 사족 보행 로봇과 6 DoF 로봇 팔을 사용하여 실험하였습니다.

- **Performance Highlights**: 실험 결과, 학습된 컨트롤러는 2.64cm의 포즈 추적 오류와 3.64도의 방향 추적 오류를 기록하며 기존 모델 기반 접근법 및 경쟁 있는 강화 학습 방법들과 비교하여 우수한 추적 정확성을 보여주었습니다. 또한 다양한 지형에서도 적절히 적응하는 능력을 입증하였습니다.



### Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts (https://arxiv.org/abs/2409.16040)
Comments:
          29 pages, 10 figures, 13 tables

- **What's New**: 이 논문에서는 Time-MoE라는 새로운 아키텍처를 소개하여, 더 크고 강력한 시계열 예측 모델을 사전에 훈련할 수 있게 하며, 추론 비용을 줄이고자 합니다.

- **Technical Details**: Time-MoE는 희소 혼합 전문가(mixture-of-experts, MoE) 디자인을 이용해 예측마다 모델의 서브셋만 활성화하여 계산 효율성을 높입니다. 이 모델은 오토 회귀(auto-regressive) 방식으로 작동하며, 문맥 길이가 다양하고 예측 지평선(forecasting horizons)을 지원하는 디코더 전용(transformer) 모델로 구성되어 있습니다. 또한, Time-300B는 9개의 도메인에 걸쳐 3000억 개 이상의 시간 포인트를 포함한 대규모 데이터입니다.

- **Performance Highlights**: Time-MoE는 24억 개의 파라미터로 스케일링되어 기존 모델들에 비해 평균 23% 및 25%씩 예측 오류를 줄이는 성과를 보였습니다. 기존 밀집 모델(dense models)과 비교하여 예측 정확도가 상당히 향상되었습니다.



### Grounded Computation & Consciousness: A Framework for Exploring Consciousness in Machines & Other Organisms (https://arxiv.org/abs/2409.16036)
- **What's New**: 이 논문은 의식(Consciousness)의 존재를 이해하기 위해서는 단순한 계산 모델링(computational modeling)만으로는 부족하다는 주장을 제기합니다. 이를 위해 의식의 형이상학적(ontological) 기초를 제시하고, 계산적 설명을 형이상학적 기초(layer)에 기반을 둔 형식적(framework) 틀을 도입합니다.

- **Technical Details**: 이 논문에서는 여러 계산적 이론에 대한 검토를 통해 '계산적 논제(The Computational Thesis)'를 제안합니다. 이 논제에 따르면 알고리즘 X에 의해 설명될 수 있는 모든 시스템은 의식을 가진다고 합니다. 논문에서는 Gödel, Escher, Bach, Global Workspace Theory, Attention Schema Theory, Integrated Information Theory(ÉIT) 등 여러 이론을 검토합니다.

- **Performance Highlights**: 본 연구는 기존의 다양한 계산적 이론이 의식 현상(conscious phenomena)에 대한 포괄적인 설명을 제공하지는 못하지만, 각각의 시스템에서 배울 점이 있다고 결론짓습니다. 특히, 형이상학적 기초의 중요성을 강조하며 기존 이론에서 의식을 설명하는데 있어 필수적인 요소로 작용함을 제시합니다.



### Deep chroma compression of tone-mapped images (https://arxiv.org/abs/2409.16032)
- **What's New**: 본 논문에서는 HDR(High Dynamic Range) 톤 매핑 이미지의 색상 압축을 위한 생성적 적대 신경망(GAN)을 제안합니다. 이는 정확한 색상 표현을 위해 이미지의 색조 속성을 고려한 손실 함수를 설계하였습니다.

- **Technical Details**: 제안된 모델은 널리 사용되는 모든 톤 매핑 연산자(TMO)와 호환되며, 색상 정확도를 향상시키기 위해 GAN 손실과 L1 손실, 색조 기반 손실을 결합한 새로운 손실 함수를 사용합니다.

- **Performance Highlights**: 모델은 기존의 색상 압축 방법에 비해 색상 정확도에서 뛰어난 성능을 보이며, 실시간 성능을 달성하여 제한된 계산 자원을 가진 장치에 적합합니다.



### AI Can Be Cognitively Biased: An Exploratory Study on Threshold Priming in LLM-Based Batch Relevance Assessmen (https://arxiv.org/abs/2409.16022)
- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 정보 검색(Information Retrieval) 작업에서 문서의 관련성을 판단할 때 경험적으로 관찰된 threshold priming 효과에 영향을 받는지를 조사합니다. 이는 LLM의 인지 편향(cognitive bias)에 대한 연구의 일환으로, LLM이 사람의 판단과 유사한 방식으로 영향을 받을 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 TREC 2019 Deep Learning 구문 수집에서 10개의 주제를 대상으로 LLM의 문서 관련성 평가를 시험했습니다. 실험에는 GPT-3.5, GPT-4, LLaMa2-13B 및 LLaMa2-70B 모델이 사용되었습니다. 초기 문서의 관련성 레벨이 후속 문서의 평가에 미치는 영향을 분석하기 위해 다양한 프로로구(prologue) 및 에필로그(epilogue) 길이를 가지고 실험을 수행했습니다.

- **Performance Highlights**: 결과에 따르면, 초기 문서의 높은 관련성은 후속 문서의 낮은 점수를 유도하는 경향이 있었습니다. 반대로 초기 문서의 낮은 관련성은 후속 문서에 대한 높은 점수를 유발했습니다. LLaMa2-70B 모델은 일부 조건에서 다른 모델과 다른 경향을 보였으며, 다른 모델들은 threshold priming 효과의 성격을 유지했습니다.



### Improvements to SDXL in NovelAI Diffusion V3 (https://arxiv.org/abs/2409.15997)
Comments:
          14 pages, 8 figures

- **What's New**: NovelAI Diffusion V3 모델은 SDXL을 기반으로 하며, 훈련 관행에서 여러 가지 향상을 이루었습니다. 특히 Zero Terminal SNR과 v-prediction 파라미터화를 도입했습니다.

- **Technical Details**: SDXL의 훈련 과정에서 𝜖-prediction에서 v-prediction으로 전환하였고, 노이즈 스케줄을 개선하여 더 높은 시그마 레벨까지 훈련하였습니다. 이를 통해 모델은 노이즈로부터 의미 있는 색상과 주파수를 예측하도록 학습하게 되었습니다.

- **Performance Highlights**: 모델의 훈련 진행을 통해 고해상도 이미지 생성에서 일관성을 회복하였으며, 실제 이미지의 품질 개선과 더불어 수렴 속도가 빨라졌습니다.



### Leveraging Unsupervised Learning for Cost-Effective Visual Anomaly Detection (https://arxiv.org/abs/2409.15980)
- **What's New**: 이 연구는 사전 훈련된 모델을 사용하고 저렴한 하드웨어로 시각적 이상 탐지 시스템을 개발하여 중소기업(SMEs)의 구매 부담을 줄이기 위한 새로운 접근 방식을 제안합니다. 이 시스템은 최소한의 데이터로 모델 훈련을 수행할 수 있으며, Raspberry Pi 4B에서 효율적으로 배포됩니다.

- **Technical Details**: 이 시스템은 Anomalib 라이브러리의 비지도 학습 모델을 활용하여 작동합니다. 10장의 정상 제품 이미지만을 사용하여 Raspberry Pi에서 90초 만에 이상 탐지 교육과 추론을 완료할 수 있으며, F1 매크로 점수는 0.95 이상을 기록합니다. PaDiM, PatchCore, CFlow-AD 및 FastFlow와 같은 여러 알고리즘이 적용되어 성능을 비교했습니다.

- **Performance Highlights**: 연구 결과, 이 저비용의 시각적 이상 탐지 시스템은 환경 변화에 약간 민감하지만, 중소 제조업체를 위한 공장 자동화 검사의 신속하고 경제적인 방법으로써의 가능성을 보여주었습니다. 시스템은 IoT(Internet of Things) 환경에서 효율적인 운영을 위한 샘플링과 같은 높은 성능을 유지합니다.



### Disentangling Age and Identity with a Mutual Information Minimization Approach for Cross-Age Speaker Verification (https://arxiv.org/abs/2409.15974)
Comments:
          Interspeech 2024

- **What's New**: 본 논문은 Cross-Age Speaker Verification (CASV)에서 나이와 관련된 정보와 정체성 정보를 효과적으로 분리하는 새로운 방법인 상호 정보(minimization of mutual information, MI) 기반의 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 두 개의 모듈로 구성되어 있습니다: 백본 모델(backbone model)과 MI 추정기(MI estimator). 백본 모델은 초기 임베딩(initial embedding)을 추출하고, 이를 나이 관련 임베딩(age embedding)과 나이 불변의 정체성 임베딩(age-invariant identity embedding)으로 분리합니다. MI 추정기는 두 임베딩 간의 상호 정보를 측정하고 이를 최소화하는 방향으로 백본 모델을 유도하여 나이 불변의 음성 임베딩을 만듭니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Vox-CA 데이터 세트의 다양한 Cross-Age 테스트 세트에서 이전의 최첨단 방법(state-of-the-art)보다 EER에서 1.53%, minDCF에서 4.79% 향상된 성능을 보여줍니다.



### Edge-device Collaborative Computing for Multi-view Classification (https://arxiv.org/abs/2409.15973)
- **What's New**: 이 논문은 딥러닝 계산을 클라우드 대신 네트워크의 엣지로 이전하여 응답 속도를 높이고 대역폭 소비를 줄이며 개인 정보 보호 문제를 해결하는 방안을 제안합니다. 특히, 협업 추론(collaborative inference)을 통해 데이터를 공유하고 컴퓨팅 부담을 분산하는 다양한 방법을 탐구합니다.

- **Technical Details**: 본 연구는 다중 뷰 분류(multi-view classification) 태스크에 중점을 두고, 엣지 서버와 최종 장치 간의 협업 방식을 조사합니다. 기존의 중앙 집중식(centralized) 또는 분산(distributed) 방식 외에도, 데이터 중복을 줄여 대역폭 소비를 감소시키는 선택적(selective) 방식도 제안합니다. 각 엣지 노드는 그들의 데이터를 기초로 하여 추론 작업에 기여할지를 결정합니다.

- **Performance Highlights**: 실험 결과, 제안된 선택적 협업 방식이 통신 절약을 18%에서 74%까지 달성할 수 있고, 여전히 90% 이상의 정확도를 유지할 수 있음을 보여줍니다. 심지어, 시각적 뷰를 일부 생략하더라도 평균 정확도는 71.92%에서 83.75%의 범위를 유지하며, 중앙 집중식 추론에 비해 대역폭 소비를 몇 배 감소시킬 수 있음을 강조합니다.



### Creating Healthy Friction: Determining Stakeholder Requirements of Job Recommendation Explanations (https://arxiv.org/abs/2409.15971)
Comments:
          14 pages, 3 figures, to be published in ACM RecSys in HR '24: 4th Workshop on Recommender Systems for Human Resources

- **What's New**: 이 연구는 구직에 있어 신뢰성 있는 Job Recommender System (JRS)의 필요성을 강조하며, 이 시스템이 이해 가능하고 투명하게 작동해야 한다고 주장합니다.

- **Technical Details**: 혼합 설계 사용자 연구(n=30)를 통해 여러 이해관계자가 추천 시스템의 설명을 바탕으로 의사결정을 하도록 평가했습니다. 사용된 지표는 객관적 지표인 정확도(correctness)와 효율성(efficiency)과 주관적 지표인 신뢰(trust), 투명성(transparency), 유용성(usefulness)을 포함했습니다.

- **Performance Highlights**: 모든 이해관계자는 설명이 포함된 JRS를 유용하다고 느꼈으나, 설명이 의사결정 속도와 정확도를 크게 개선시키지 못했습니다. 또한 이해관계자들은 주로 자신의 지식과 직관에 의존하며, 텍스트 기반 설명을 선호하는 경향이 있었습니다.



### Provably Efficient Exploration in Inverse Constrained Reinforcement Learning (https://arxiv.org/abs/2409.15963)
- **What's New**: 이번 논문에서는 Inverse Constrained Reinforcement Learning (ICRL)을 통해 복잡한 환경에서 최적 제약 조건을 도출하기 위한 새로운 전략적 탐색 프레임워크를 제안합니다. 기존의 ICRL 알고리즘은 상호작용 환경에서 학습 샘플을 수집하지만, 이러한 샘플링 전략의 효율성과 효과성은 불분명했습니다.

- **Technical Details**: 제안된 프레임워크에서는 ICRL 문제를 위한 실현 가능 제약 조건 집합을 정의하고, 전문가 정책(expert policy)과 환경 역학(environmental dynamics)이 제약 조건의 최적성에 어떻게 영향을 미치는지 조사합니다. 두 가지 탐색 알고리즘을 제안하여 1) 비용 추정의 한정된 집합 오류(bounded aggregate error)를 동적으로 줄이고, 2) 탐색 정책(exploration policy)을 전략적으로 제약합니다.

- **Performance Highlights**: 제안된 알고리즘의 성능은 다양한 환경에서 실증적으로 검증되었으며, 이론적으로 타당한 샘플 복잡도가 기반이 됩니다.



### ASD-Diffusion: Anomalous Sound Detection with Diffusion Models (https://arxiv.org/abs/2409.15957)
Comments:
          This paper will appear at ICPR 2024

- **What's New**: 본 논문은 공장 환경에서 비감독형 이상 음성 감지(ASD) 위한 새로운 방법인 ASD-Diffusion을 제안합니다. 이는 정상 소리만으로 이상을 감지하는 일반화 가능한 방법을 개발하는 것을 목표로 합니다.

- **Technical Details**: ASD-Diffusion은 노이즈가 섞인 음성 특징을 사용하여 정상 패턴으로 재구성한 후, 재구성 과정에서 원래 입력과 큰 차이를 보이는 이상을 필터링하는 알고리즘을 도입합니다. 또한, Denoising Diffusion Implicit Model (DDIM)을 적용하여 샘플링 속도를 향상시켰습니다.

- **Performance Highlights**: DCASE 2023 챌린지의 실험 결과, 제안된 방법은 기준 모델 대비 7.75% 성능 향상을 보여 효과성을 입증했습니다.



### Historical Trajectory Assisted Zeroth-Order Federated Optimization (https://arxiv.org/abs/2409.15955)
Comments:
          28 pages with theoretical proof

- **What's New**: 이번 연구에서는 Federated Learning에서 기울기 정보가 없는 상황에서도 기울기 추정을 개선하기 위해 비등방성 샘플링(non-isotropic sampling) 방법을 제안합니다. 이는 과거의 솔루션 궤적을 기반으로 하는 방법으로 유망한 영역을 탐색하는 것을 장려합니다.

- **Technical Details**: 제안된 방법은 비등방성 가우시안 분포(non-isotropic Gaussian distribution)를 사용하여 기울기를 추정하며, 이는 두 개의 부분으로 구성된 공분산 행렬(covariance matrix)을 사용하여 구현됩니다. 첫 번째 부분은 최근의 훈련 궤적에 의해 형성된 부분공간을 포함하며, 두 번째 부분은 아이덴티티 행렬(identity matrix)로, 이는 전역적인 탐색(global exploration)을 보장합니다. 기울기 추정에는 수렴 속도가 기존 방법들이 유지되면서 통신 오버헤드가 거의 없는 장점이 있습니다.

- **Performance Highlights**: 여러 수치 실험을 통해 제안된 방법의 효과가 검증되었으며, 기존에 과대 평가된 기울기 추정 방법에 비해 개선된 성능을 보였습니다. 이 방법은 특히 로컬 데이터 샘플만을 사용해도 강력한 결과를 도출할 수 있습니다.



### Automated test generation to evaluate tool-augmented LLMs as conversational AI agents (https://arxiv.org/abs/2409.15934)
Comments:
          14 pages, 5 figures, Submitted to GenBench@EMNLP2024

- **What's New**: 본 논문에서는 Tool-augmented LLMs(대형 언어 모델)를 평가하기 위한 테스트 생성 파이프라인을 제시하고 있습니다. 기존의 평가 데이터셋이 단일 상호작용 및 함수 호출에만 집중했던 반면, 이 연구는 사용자 정의 절차에 기반한 다양한 테스트를 생성합니다.

- **Technical Details**: LLMs를 기반으로 한 추천된 파이프라인은 중간 그래프(intermediate graphs)를 활용하여 발생할 수 있는 비현실적인 내용生성을 제한하고, 대화의 가능성을 널리 포괄하는 고품질 데이터를 생성합니다. 이 연구에서는 고객 지원을 위한 AI 에이전트 평가를 위한 ALMITA(Automated benchmark of Language Models for Intelligent Tool-augmented Agents)라는 수작업으로 제작된 데이터셋을 개발했습니다.

- **Performance Highlights**: 기존 LLM들은 단일 메시지 정확도 및 올바른 함수 호출에 있어 높은 성능을 보였지만, 전체 대화에서의 정확도는 제한적임을 보여주었습니다. 이는 LLMs가 완전 자율 고객 지원 AI 에이전트로 배치될 경우의 성공 가능성에 의문을 제기합니다.



### Multilingual Transfer and Domain Adaptation for Low-Resource Languages of Spain (https://arxiv.org/abs/2409.15924)
Comments:
          6 pages,wmt24. arXiv admin note: substantial text overlap with arXiv:2409.14842; text overlap with arXiv:2409.14800

- **What's New**: 이번 논문은 Huawei Translation Service Center (HW-TSC)의 WMT 2024에서의 스페인 저자원 언어 번역 태스크 제출 상태를 소개합니다. 이 연구팀은 스페인어에서 아라곤어(es-arg), 아라니세어(es-arn), 아스투리안어(es-ast)로의 번역 작업에 참여했습니다.

- **Technical Details**: 우리는 다국어 전이(multi-language transfer), 정규화 드롭아웃(regularized dropout), 포워드 번역(forward translation), 백 번역(back translation), Labse denoising, 전이 집합 학습(transduction ensemble learning) 등의 훈련 전략을 사용하여 딥 트랜스포머 기반의 신경 기계 번역(NMT) 모델을 훈련했습니다.

- **Performance Highlights**: 이러한 개선 전략을 통해 우리 제출물은 최종 평가에서 경쟁력 있는 결과를 달성했습니다.



### Enhancing Text-to-SQL Capabilities of Large Language Models via Domain Database Knowledge Injection (https://arxiv.org/abs/2409.15907)
Comments:
          This paper has been accepted by ECAI 2024

- **What's New**: 이번 논문에서는 LLMs가 데이터베이스 스키마와 셀 값에 대한 도메인 지식을 효과적으로 이해하고 활용할 수 있도록 '지식 주입' (knowledge injection) 방법을 도입하였습니다. 이를 통해 Text-to-SQL 작업에서의 성능을 향상시키는 다양한 기술적 접근을 선보입니다.

- **Technical Details**: 제안된 방법은 특정 도메인 데이터베이스 지식을 기반으로 LLMs를 사전 훈련 (pre-training)하고, 하위 Text-to-SQL 작업에 맞춰 미세 조정 (fine-tuning)하는 것입니다. 이를 통해 Execution Match (EX) 및 Exact Match (EM) 지표에서 현저한 개선을 이루어내며, 컬럼 이름 생성 및 값 일치 오류를 줄입니다.

- **Performance Highlights**: 실험 결과, 제안한 지식 주입 방법이 여러 개의 오픈 소스 LLMs에서 실질적인 성능 향상을 보여주었으며, 이는 다양한 Text-to-SQL 작업에 광범위하게 적용 가능하다는 것을 검증하였습니다.



### Boosting Code-Switching ASR with Mixture of Experts Enhanced Speech-Conditioned LLM (https://arxiv.org/abs/2409.15905)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 본 논문에서는 자동 음성 인식(Automatic Speech Recognition, ASR)의 코드 스위칭(Code-Switching, CS) 문제를 해결하기 위해 대화식 음성 모델(Speech-Conditioned Large Language Model, SC-LLM)과 전문가 혼합(Mixture of Experts, MoE) 기반 커넥터를 통합한 새로운 접근 방식을 소개합니다.

- **Technical Details**: IDIT(Insertion and Deletion of Interruption Token) 메커니즘을 통해 LLM의 텍스트 생성 능력을 음성 인식 과업에 효과적으로 이전하는 방법을 제안합니다. 두 단계로 진행되는 훈련 전략이 포함되어 있으며, 첫 번째 단계에서 언어 전문화 전문가(Language-Specialized Experts, LSE)와 함께 음성 표현을 텍스트로 매핑합니다. 두 번째 단계에서는 IDIT 메커니즘을 활용하여 모든 전문가가 일반 표현을 학습하도록 합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 ASRU-2019 Mandarin-English 코드 스위칭 데이터셋에서 기존 모델들에 비해 10% 이상의 성능 향상을 보여주며, SC-LLM의 잠재력을 입증했습니다.



### Machine Translation Advancements of Low-Resource Indian Languages by Transfer Learning (https://arxiv.org/abs/2409.15879)
Comments:
          6 pages, wmt24. arXiv admin note: substantial text overlap with arXiv:2409.14800

- **What's New**: 이 논문은 Huawei Translation Center (HW-TSC)가 WMT24 인도 언어 기계 번역(MT) 공동 작업에 제출한 내용을 소개합니다. 본 연구는 리소스가 부족한 인도 언어에 대한 신뢰할 수 있는 기계 번역 시스템을 개발하기 위해 두 가지 별도의 knowledge transfer 전략을 적용했습니다.

- **Technical Details**: Assamese(as)와 Manipuri(mn)의 경우, 우리는 기존의 IndicTrans2 오픈소스 모델을 미세 조정하여 영어와 이들 언어 간의 쌍방향 번역을 가능하게 했습니다. Khasi (kh)와 Mizo (mz)의 경우, 네 언어 쌍의 이중 언어 데이터를 이용하여 다국어 모델을 훈련시켰고, 추가적으로 약 8천 쌍의 영어-벵골어 이중 언어 데이터를 사용했습니다. 이를 통해 데이터 부족 문제를 해결했습니다.

- **Performance Highlights**: 전달 학습 실험에서는 en-as에 대해 23.5 BLEU, en-mn에 대해 31.8 BLEU, as-en에 대해 36.2 BLEU, mn-en에 대해 47.9 BLEU의 성과를 거두었습니다. 다국어 모델의 전이 학습 실험 결과는 en-kh에서 19.7 BLEU, en-mz에서 32.8 BLEU, kh-en에서 16.1 BLEU, mz-en에서 33.9 BLEU를 기록하였습니다.



### Whisper in Medusa's Ear: Multi-head Efficient Decoding for Transformer-based ASR (https://arxiv.org/abs/2409.15869)
Comments:
          Under Review

- **What's New**: Whisper-Medusa는 대규모 Transformer 모델을 위한 新한 음성 인식(Speech Recognition) 방법론으로, 단일 단계에서 여러 개의 토큰을 예측하여 처리 속도를 개선합니다. 기존 Whisper 모델의 구조를 확장하여 속도를 극대화하고 Word Error Rate (WER)에 미치는 영향을 최소화하는 방식입니다.

- **Technical Details**: Whisper-Medusa는 Speculative Decoding 기법을 활용하여 K+1 토큰을 동시에 예측하는 방식을 채택합니다. 전반적인 구조는 인코더-디코더 transformer 모델로, 입력 오디오를 처리하고 다차원 임베딩으로 변환한 후, 이를 기반으로 여러 개의 토큰을 동시에 생성합니다.

- **Performance Highlights**: 이 모델은 다양한 다국어 데이터셋에서 올바른 인식 성능을 유지하면서 50%의 레이턴시 감소를 보여줍니다. Whisper-Medusa의 효율성을 강조하기 위해 여러 학습 설정 및 데이터셋에서의 효과를 평가했습니다.



### BeSimulator: A Large Language Model Powered Text-based Behavior Simulator (https://arxiv.org/abs/2409.15865)
Comments:
          7 pages, 3 figures, 2 tables

- **What's New**: 본 논문에서는 기존의 로봇 시뮬레이터의 한계를 극복하기 위해 행동 시뮬레이션(Behavior Simulation)을 이론적으로 정의하고 새로운 프레임워크인 BeSimulator를 소개하였습니다. BeSimulator는 텍스트 기반의 가상 환경에서 로봇 행동 로직을 수립하여 시뮬레이션하고, 체계적인 사고 과정을 통해 행동의 실행 가능성과 상태 전이를 분석합니다.

- **Technical Details**: BeSimulator는 모듈화된 LLM(large language model) 기반의 프레임워크로, 행동 계획 솔루션(BPS)에 대한 단계별 시뮬레이션을 수행합니다. 이 프레임워크는 '사례 생성'(Case Generation), 'BPS 시뮬레이션'(BPS Simulation), 및 'BPS 평가'(BPS Evaluation)라는 세 가지 핵심 모듈을 포함하고 있습니다. 또한, Chain of Behavior Simulation(CBS) 접근법을 통해 행동의 실행 가능성과 상태 전이를 깊이 분석합니다.

- **Performance Highlights**: BTSIMBENCH라는 행동 트리 기반의 시뮬레이션 벤치마크를 통해 실험한 결과, BeSimulator는 기존 방법들에 비해 14.7%에서 26.6%까지 행동 시뮬레이션 성능이 향상되었습니다. 이는 BeSimulator가 특히 긴 기간의 복잡한 시뮬레이션에서 우수한 성능을 제공함을 입증합니다.



### A Zero-Shot Open-Vocabulary Pipeline for Dialogue Understanding (https://arxiv.org/abs/2409.15861)
- **What's New**: 본 연구에서 우리는 제로샷(zero-shot), 오픈 어휘(open-vocabulary) 시스템을 제안하며, 디지털 대화 이해를 위한 통합된 파이프라인을 구성합니다.

- **Technical Details**: 제안된 방법론은 도메인 분류(domain classification)부터 시작하여, 여러 방법으로 DST(대화 상태 추적)를 수행합니다. 특히 DST를 질문-답변(question-answering) 문제로 변환하는 'DST-as-QA' 방식과 자가 수정 프롬프트(self-refining prompt) 기법을 활용한 'DST-as-SRP'를 포함합니다. 이 시스템은 고정된 슬롯 값에 의존하지 않아 동적으로 적응할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 MultiWOZ 2.1 데이터셋에서 20% 향상된 Joint Goal Accuracy(JGA)를 달성하며, LLM API에 대한 요청 수를 최대 90% 줄입니다. 또한, 제로샷 및 오픈 어휘 설정에서 현재 SOTA 방법들을 초월하는 성능을 보였습니다.



### Identification For Control Based on Neural Networks: Approximately Linearizable Models (https://arxiv.org/abs/2409.15858)
Comments:
          15 pages, 3 figures, 6 tables, accepted as a poster in SysDO 2024, Stuttgart, Germany

- **What's New**: 이번 연구는 비선형 시스템의 효율적인 제어 설계 및 안정성 분석을 위한 제어 지향 식별 방법을 제시합니다. 신경망(Neural Networks)을 사용하여 비선형 시스템의 시간 영역 내 입력-출력 동작을 근사하는 이산 시간 비선형 상태-공간 모델을 식별합니다.

- **Technical Details**: 제안된 방법은 모델을 피드백에 의해 근사적으로 선형화할 수 있도록 구성되어, 제어 법칙이 학습 단계에서 자명하게 따르도록 합니다. 상태공간 모델 내 신경망 구조를 요구하며, 비선형 함수 f는 신경망을 통해 근사화됩니다. 선형 제어 이론이 제어기 설계 및 폐루프 시스템의 안정성 분석에 활용됩니다.

- **Performance Highlights**: 연구된 방법론은 시스템 식별을 위한 일반적인 벤치마크를 사용하여 효과성을 입증되었으며, 비선형 시스템의 안정성 분석 및 제어 설계 과정을 간소화하는 데 기여합니다.



### Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection (https://arxiv.org/abs/2409.15844)
- **What's New**: 새로운 기법인 adaptive learn-then-test (aLTT)를 소개합니다. aLTT는 AI 모델의 모집단 리스크에 대해 유한 샘플 통계적 보증을 제공하는 효율적인 하이퍼파라미터 선택 절차입니다. 기존의 learn-then-test (LTT) 방식과는 달리, aLTT는 e-processes를 활용한 데이터 의존적 방식으로 순차적으로 다중 가설 검정을 수행하며, 테스트 라운드를 줄이는 데 적합합니다.

- **Technical Details**: aLTT는 p-value 기반의 다중 가설 검정(MHT)에 의존하는 기존의 LTT 기술과는 달리, e-process를 통한 데이터 의존적 MHT 방식을 구현합니다. 이러므로 테스트 라운드 수를 줄일 수 있으며, 특히 테스트가 비용이 많이 들거나 안전 위험이 있는 상황에서 효과적입니다. aLTT는 FWER(가족간 오류 비율) 및 FDR(허위 발견율)에 대한 엄격한 통제를 보장합니다.

- **Performance Highlights**: aLTT는 오프라인 강화 학습을 위한 온라인 정책 선택 및 무선 공학을 위한 자원 할당과 같은 두 가지 실제 시나리오에서 하이퍼파라미터를 효과적으로 선택할 수 있는 능력을 보여주었습니다. LTT가 필요한 테스트 라운드의 일부만을 사용하여 신뢰할 수 있는 하이퍼파라미터를 제공합니다.



### Empirical Insights on Fine-Tuning Large Language Models for Question-Answering (https://arxiv.org/abs/2409.15825)
- **What's New**: 본 연구는 질문-응답(QA) 작업을 위한 대형 언어 모델(LLM)의 전이 학습을 최적화할 수 있는 효과적인 세부 전략을 제시합니다. 기존 연구와 달리, 우리는 사전 훈련된 언어 모델의 메모리와 지식 수준에 따라 데이터를 체계적으로 분류하고, 실험 분석을 통해 세 가지 주요 질문에 대해 답변합니다.

- **Technical Details**: 이 연구에서는 다중 템플릿 보완 메커니즘(multi-template complementation mechanism)을 사용하여 LLM의 특정 지식 유형에 대한 기억 정도를 평가합니다. 또한 SFT(Supervised Fine-Tuning) 단계에서 소수의 데이터 포인트(최소 60개)로도 QA 작업을 성공적으로 수행할 수 있음을 확인했습니다.

- **Performance Highlights**: SFT 데이터의 메모리 수준이 모델 성능에 미치는 영향을 분석한 결과, 사전 훈련 단계에서 잘 기억된 데이터로 훈련할 경우 LLM의 성능이 유의미하게 향상되는 것으로 나타났습니다. 하지만 모델이 거의 기억하지 못한 데이터를 사용한 경우에는 성능이 크게 저하되었습니다.



### Interactive Example-based Explanations to Improve Health Professionals' Onboarding with AI for Human-AI Collaborative Decision Making (https://arxiv.org/abs/2409.15814)
- **What's New**: 이 논문에서는 AI-assisted decision-making에서 건강 전문가들이 AI의 결정을 더 잘 신뢰할 수 있도록 도와주는 interactive example-based explanations를 제안합니다. 이를 통해 사용자가 AI 모델에 대한 신뢰를 향상시킬 수 있는 방법을 모색합니다.

- **Technical Details**: AI 기반의 의사결정 지원 시스템을 구현하였으며, 신경망(neural network)을 사용하여 뇌졸중 생존자의 운동 질을 평가합니다. 이 시스템은 사용자가 입력한 새로운 운동 데이터에 가까운 k-neighbourhoods를 시각화하여 AI의 출력과 실제 레이블을 함께 보여줍니다. 이를 통해 사용자는 데이터가 어떻게 표현되었고 AI 모델이 어떻게 작동하는지를 이해할 수 있습니다.

- **Performance Highlights**: interactive example-based explanations를 제공한 결과, 건강 전문가들은 AI에 대한 신뢰도가 개선되었으며, '올바른' 결정을 내릴 확률이 높아지고 '잘못된' 결정의 확률은 낮아졌습니다. 이러한 결과는 사용자가 onboarding 과정에서 AI를 더 효과적으로 사용할 수 있도록 도와줍니다.



### Layer-wise Model Merging for Unsupervised Domain Adaptation in Segmentation Tasks (https://arxiv.org/abs/2409.15813)
- **What's New**: 본 논문에서는 기존에 훈련된 모델들을 결합하여 비용 없이 모델 병합을 수행하는 새로운 아키텍처를 제안합니다. 이 방법은 레이어 단위로 모델을 통합하여 초기 레이어를 통합하면서 최종 레이어의 특수성을 유지합니다.

- **Technical Details**: 제안된 방법은 Unsupervised Domain Adaptation (UDA)의 맥락에서 다양한 태스크와 데이터셋에 대해 실험하였으며, 특히 Semantic과 Panoptic Segmentation 작업에 적합합니다. 이 방법은 모델 파라미터의 일관성을 유지하고 서로 다른 데이터셋 및 태스크에서의 모델 병합을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 서로 다른 아키텍처의 모델 병합 시 mIoU가 6.8% 향상되었고, Semantic과 Panoptic Segmentation 모델 병합을 통해 mPQ가 7% 증가하는 등 UDA의 성능이 크게 향상되었습니다.



### Towards Universal Large-Scale Foundational Model for Natural Gas Demand Forecasting (https://arxiv.org/abs/2409.15794)
- **What's New**: 본 연구는 자연가스 수요 예측을 위해 특별히 설계된 첫 번째 기초 모델(Foundation Model)을 제안합니다. 기존 전통적 방법들이 현업에서의 복잡한 수요 패턴 예측에 한계를 보이는 반면, 기초 모델은 이 문제를 해결하기 위한 견고한 솔루션을 제공합니다.

- **Technical Details**: 기반 모델은 대조 학습(contrastive learning)을 활용하여 역사적 소비 데이터의 노이즈 및 유사 샘플의 잘못된 분류 문제가 예측 정확도에 미치는 영향을 개선했습니다. 새로운 노이즈 필터링 기법을 통합하여 학습된 표현의 질을 향상시키고, 산업별 특성을 잘 포착할 수 있도록 사전 훈련(prior tuning) 과정을 수행했습니다.

- **Performance Highlights**: 우리의 모델은 ENN 그룹에서 다수의 산업 및 상업적 고객 데이터(MSE에서 3.68%, MASE에서 6.15% 개선)를 사용한 실험에서 기존의 최첨단 방법을 초월하는 성능을 나타냈습니다.



### Small Language Models: Survey, Measurements, and Insights (https://arxiv.org/abs/2409.15790)
- **What's New**: 이 논문은 최근 몇 년 간의 모든 작은 언어 모델(SLM)들을 종합적으로 검토하고 이들의 기술 혁신 및 온디바이스(온기기) 비용을 벤치마킹하여 요약합니다. SLM의 매니페스트 데이터를 공개하여 앞으로의 연구에 기여할 수 있는 기반을 마련합니다.

- **Technical Details**: SLM은 100M에서 5B 파라미터 범위의 transformer 기반, decoder-only 아키텍처로 구성됩니다. 59개의 최첨단 오픈 소스 SLM을 분석하여 아키텍처, 훈련 데이터셋, 훈련 알고리즘의 세 가지 축을 중심으로 기술 혁신을 평가합니다.

- **Performance Highlights**: SLM의 성능을 평가하며, commonsense reasoning, in-context learning, mathematics, coding과 같은 다양한 분야에서 능력을 분석합니다. 또한 벤치마킹 데이터를 통해 디바이스에서의 런타임 비용에 대한 귀중한 통찰을 제공합니다.



### Spatial-Temporal Mixture-of-Graph-Experts for Multi-Type Crime Prediction (https://arxiv.org/abs/2409.15764)
- **What's New**: ST-MoGE는 범죄 예측의 공간-시간 이질성을 해결하려고 고안된 혁신적인 프레임워크입니다. 이는 범죄 카테고리에 특화된 Mixture-of-Experts (MoE) 아키텍처를 활용하여 다양한 범죄 패턴을 통합적으로 잡아냅니다.

- **Technical Details**: ST-MoGE는 Attentive-gated Mixture-of-Graph-Experts (MGEs) 모듈을 도입하여 다양한 범죄 카테고리의 공간-시간 의존성을 포착합니다. 또한 Cross-Expert Contrastive Learning (CECL)을 통해 각 전문가가 특정 패턴 모델링에 집중하도록 하여 혼합 및 중복을 줄입니다. Hierarchical Adaptive Loss Re-weighting (HALR) 기법을 통해 불균형적인 공간 분포 문제를 해결합니다.

- **Performance Highlights**: ST-MoGE는 뉴욕시와 시카고 두 개의 실제 범죄 데이터셋을 활용하여 12개의 기존 방법들과 비교하여 탁월한 성과를 보였습니다. 실험 결과, 제안된 방법의 우수성이 입증되었습니다.



### IRSC: A Zero-shot Evaluation Benchmark for Information Retrieval through Semantic Comprehension in Retrieval-Augmented Generation Scenarios (https://arxiv.org/abs/2409.15763)
- **What's New**: 이 논문은 다국어 Retrieval-Augmented Generation (RAG) 작업에서 임베딩 모델의 성능을 평가하기 위한 IRSC 벤치마크를 소개합니다. 이 벤치마크는 쿼리 검색, 제목 검색, 단락의 일부 검색, 키워드 검색, 요약 검색의 cinco 가지 검색 작업을 포함합니다.

- **Technical Details**: IRSC 벤치마크는 embedding 모델의 성능을 다양한 검색 과제에서 평가하며, 새로운 메트릭인 Semantic Comprehension Index (SSCI) 및 Retrieval Capability Contest Index (RCCI)를 도입했습니다. 이 벤치마크는 여러 언어(영어, 중국어 및 혼합 언어 데이터셋)에서 모델을 평가합니다.

- **Performance Highlights**: 연구 결과에 따르면, IRSC 벤치마크는 실용적인 RAG 작업에서의 embedding 모델의 성능을 더 잘 이해하고 개발하는 데 기여할 수 있습니다. 이 연구는 embedding 모델의 언어 간 한계를 통찰하는 데 중요한 기여를 합니다.



### TFG: Unified Training-Free Guidance for Diffusion Models (https://arxiv.org/abs/2409.15761)
- **What's New**: 본 논문은 교육 없이도 원하는 목표 속성을 가진 샘플을 생성할 수 있는 새로운 알고리즘 프레임워크인 Training Free Guidance (TFG)를 소개합니다. 기존 방식들의 이론적 기초와 방대한 벤치마크 테스트가 부족했던 점을 해결하고, TFG에서 기존 방법들이 특별한 하이퍼파라미터 서브스페이스에 해당함을 입증합니다.

- **Technical Details**: TFG의 하이퍼파라미터 검색 전략은 다양한 다운스트림 작업에 쉽게 적용할 수 있도록 설계되었습니다. 이 연구에서는 7개의 확산 모델(difussion models)을 이용해 16개의 작업과 40개의 목표에 대해 체계적으로 벤치마크를 수행했습니다. TFG는_all_datasets에서 평균 8.5% 향상된 성능을 기록하였습니다.

- **Performance Highlights**: TFG는 다양한 복잡도의 목표와 데이터셋에서 사용자 요구에 맞는 샘플을 생성하는 데 뛰어난 성능을 보였습니다. TFG 방법론은 기존 방법들과 비교하여 전반적으로 우수함을 입증하였고, 교육이 필요 없는 조건 생성 알고리즘 관련 연구의 강력한 기반을 제시합니다.



### Stage-Wise Reward Shaping for Acrobatic Robots: A Constrained Multi-Objective Reinforcement Learning Approach (https://arxiv.org/abs/2409.15755)
Comments:
          7 pages

- **What's New**: 본 논문에서는 강화 학습(reinforcement learning, RL)의 보상 기능 정의를 단순화하기 위한 새로운 방법을 제안합니다. 제안된 방법은 제약된 다목적 RL(constrained multi-objective RL, CMORL) 프레임워크를 활용하여, 여러 개의 보상 및 비용 함수를 단계별로 정의함으로써 보상 설계 과정을 간소화하는 것을 목표로 합니다.

- **Technical Details**: CMORL 프레임워크를 사용하여 복잡한 동작을 필요로 하는 여러 단계의 작업을 정의합니다. 각 단계에 대해 독립적인 보상 및 비용 함수를 설정하며, 이를 통해 각 동작의 특성에 맞는 보상을 제공합니다. 또한, 제안된 방식은 여러 가지 acrobatic tasks에서의 성공적인 수행을 목표로 하며, proximal policy optimization (PPO) 알고리즘의 변형인 constrained multi-objective PPO (CoMOPPO)를 통해 정책 업데이트를 수행합니다.

- **Performance Highlights**: 제안된 방법은 쿼드복잡(rigorous quadrupedal) 및 휴머노이드 로봇에서 다양한 acrobatic tasks(예: back-flips, two-hand walks 등)를 성공적으로 수행하는 것을 보여주었으며, 기존의 RL 및 제약 RL 알고리즘에 비해 우수한 결과를 보였습니다. 실제 환경에서의 테스트에서도 우수한 성능을 입증하였습니다.



### Development and Validation of Heparin Dosing Policies Using an Offline Reinforcement Learning Algorithm (https://arxiv.org/abs/2409.15753)
- **What's New**: 본 연구는 강화학습(Artificial Intelligence, AI) 기반의 개인화된 헤파린(heparin) 투여 정책을 제안하여 집중치료실(ICU)에서의 복잡한 약물 투여 문제를 해결합니다. 이는 정교한 의사결정 지원 도구의 개발을 위한 선례를 마련합니다.

- **Technical Details**: 이 연구는 향상된 기계 학습 기술과 대규모 임상 데이터(Medical Information Mart for Intensive Care III, MIMIC-III)를 활용하여 헤파린 투여의 정확성을 높이는 방법을 제시합니다. 배치 제약 강화 학습(batch-constrained RL) 접근 방식을 통해 비원주율 오류를 최소화하고, 가중 중요 샘플링(weighted importance sampling) 기술을 사용하여 정책의 효과성을 평가합니다.

- **Performance Highlights**: 제안된 배치 제약 Q 학습(Batch-Constrained Q-learning, BCQ) 알고리즘은 기존의 딥 RL(ddeep Reinforcement Learning) 방식보다 성능이 우수하며, t-SNE 기법을 통해 정책의 효율성을 시각적으로 분석하고 헤파린 투여의 효과를 극대화하는 데 기여합니다.



### The Roles of Generative Artificial Intelligence in Internet of Electric Vehicles (https://arxiv.org/abs/2409.15750)
Comments:
          25 Pages

- **What's New**: 이번 논문은 생성형 인공지능(GenAI) 모델의 발전과 전기차(\acEV) 및 전기차 인터넷(\acIoEV) 애플리케이션에서의 활용을 조사합니다. 특히, GenAI를 통해 IoEV의 다양한 계층에서 전기차의 배터리, 개별 전기차, 스마트 그리드, 그리고 보안 레이어에 대한 구체적인 기술을 분류하고 소개합니다.

- **Technical Details**: 논문에서는 IoEV의 네 가지 주요 레이어를 소개합니다: 배터리 레이어, 개별 EV 레이어, EV가 포함된 스마트 그리드 레이어, 그리고 보안 레이어. 각 레이어에 적용된 GenAI 기술로는 Transformer, \acGAN, \acAE, \acVAE, 그리고 \acGDM 등이 포함됩니다. 또한, 각 레이어의 GenAI 모델 학습을 위한 공개 데이터셋 요약이 제공됩니다.

- **Performance Highlights**: 이 논문은 GenAI 기술을 통해 IoEV 시스템을 더욱 강력하고 효율적으로 발전시키기 위한 미래 연구 방향을 제시합니다. GenAI의 활용은 데이터 부족 문제 해결, 이상 탐지, EV 충전 로드 예측, 시나리오 생성을 통해 IoEV 애플리케이션의 품질을 향상시킬 것으로 기대됩니다.



### Training Neural Networks for Modularity aids Interpretability (https://arxiv.org/abs/2409.15747)
Comments:
          4 pages, preprint

- **What's New**: 본 논문은 신경망의 해석 가능성을 높이기 위해 모델을 분리된 클러스터로 나누는 새로운 접근법을 제시합니다. 이를 통해 모델의 각 기능을 독립적으로 연구할 수 있습니다. 또한, 'enmeshment loss'라는 새로운 손실 함수를 도입하여 훈련 동안 비상호작용 클러스터를 형성할 수 있도록 유도합니다.

- **Technical Details**: 기존의 클러스터링 방법은 신경망의 해석에 비효율적이라는 점을 강조하며, 'Bipartite Spectral Graph Clustering (BSGC)' 알고리즘을 제안합니다. 이 알고리즘은 가중치 기반의 유사도 행렬을 사용하여 신경망의 레이어를 bipartite 클러스터로 나누며, 이 과정에서 'enmeshment loss'가 클러스터의 모듈화를 촉진합니다. 이 손실 항은 훈련 중에 클러스터 간의 간섭을 최소화하는 데 도움을 줍니다.

- **Performance Highlights**: CIFAR-10 데이터셋에서 실험을 통해 클러스터링된 모델은 95% 이상의 정확도를 유지하며, 효과적인 회로 크기(ECS)가 줄어들어 해석 가능성이 향상되었습니다. 연구 결과, 클러스터링된 모델은 비클러스터 모델에 비해 평균적으로 61.25% 더 적은 파라미터를 갖는 효과적인 회로를 생성하여, 독립적으로 클러스터가 기여하는 정도를 보여줍니다.



### EvoFA: Evolvable Fast Adaptation for EEG Emotion Recognition (https://arxiv.org/abs/2409.15733)
- **What's New**: EEG 기반 감정 인식에서의 모델 재사용 시 성능 저하 문제를 해결하기 위해, 본 논문에서는 Evolvable Fast Adaptation (EvoFA)라는 온라인 적응 프레임워크를 제안했습니다. 이 프레임워크는 Few-Shot Learning (FSL)과 Domain Adaptation (DA)을 통합하여 빠른 적응과 분포 일치를 이룹니다.

- **Technical Details**: EvoFA는 두 단계 일반화 과정을 통해 FSL의 빠른 적응성과 DA의 분포 일치를 조화롭게 결합합니다. 학습 단계에서는 강력한 일반화 능력을 지닌 메타 학습 모델을 구축하고, 테스팅 단계에서는 진화하는 소스 데이터를 기반으로 타겟 데이터의 주변 분포를 반복적으로 정렬하여 온라인 테스트 성능을 향상시킵니다.

- **Performance Highlights**: EvoFA는 기존 FSL 방법 및 이전의 온라인 방법들과 비교해 상당한 성능 향상을 달성했습니다. 본 연구는 실제 상황에서 EEG 기반 감정 인식의 보다 폭넓은 사용을 가능하게 합니다.



### Learning Multiple Probabilistic Decisions from Latent World Model in Autonomous Driving (https://arxiv.org/abs/2409.15730)
- **What's New**: 이번 논문에서 제안하는 LatentDriver는 오토리그레시브(world model) 세계 모델을 활용하여 불확실성 모델링을 개선하고 자기오도문제를 해결하여 더 나은 의사결정을 이끌어내는 방법을 제시합니다.

- **Technical Details**: LatentDriver는 환경의 다음 상태와 차량의 가능한 행동을 혼합(미xture) 분포로 모델링합니다. 이를 통해 결정의 확률적 성격을 캡쳐하며, 여러 확률적 가설을 세워 행동을 예측합니다. 또한, 행동을 미리 샘플링하여 자기오도 문제를 완화합니다. 미들 레이어에서 샘플링된 행동을 사용하여 최종 결정을 내리는 방식입니다.

- **Performance Highlights**: 실험 결과, LatentDriver는 최신 강화 학습(reinforcement learning) 및 모방 학습(imitation learning) 방법을 초월하며, 전문가 수준의 성능을 보였습니다. Waymax 벤치마크에서 평가할 때, 비반응형 및 반응형 에이전트에 대한 평가에서 두각을 나타냈습니다.



### Sequential Learning in the Dense Associative Memory (https://arxiv.org/abs/2409.15729)
- **What's New**: Dense Associative Memory (현대 Hopfield 네트워크)을 통한 연속 학습의 성능을 조사합니다. 이 연구는 Hopfield 네트워크의 생물학적 영감을 바탕으로 하는 다양한 기법을 조명합니다.

- **Technical Details**: Dense Associative Memory는 Hopfield 네트워크의 일반화 모델로, 상호작용 정점 (interaction vertex)을 조정하여 더 높은 용량과 프로토타입 학습 능력을 제공합니다. 이 네트워크는 기존의 Hopfield 네트워크와 비교하여, 학습 시간이 개선되고 더 흥미로운 메모리 구조를 보유합니다.

- **Performance Highlights**: 기존의 연속 학습 방법들이 Dense Associative Memory에 적용되어 성능 향상 결과를 보여줍니다. 특히, 낮은 및 높은 상호작용 정점에서의 행동 차이 및 다양한 최신 연속 학습 기법의 효과를 분석합니다.



### LLM-Cure: LLM-based Competitor User Review Analysis for Feature Enhancemen (https://arxiv.org/abs/2409.15724)
Comments:
          25 pages

- **What's New**: 이 연구에서는 사용자 리뷰를 통한 경쟁 앱 분석을 통해 모바일 앱 기능 향상을 자동으로 제안하는 LLM-Cure라는 접근법을 제안합니다.

- **Technical Details**: LLM-Cure는 사용자 리뷰에서 기능을 추출하고 분류하기 위해 Large Language Model (LLM)을 사용합니다. 앱 내 불만 사항이 제공되면, 경쟁 앱에서 4와 5점 리뷰를 큐레이션하여 타겟 앱에 대한 개선점을 제안합니다.

- **Performance Highlights**: LLM-Cure는 1,056,739개의 리뷰를 분석하여 기능 할당에서 13%의 F1-score, 16%의 recall, 11%의 precision 향상을 보였습니다. 또한, 제안된 개선 사항의 73%가 실제로 구현된 것으로 확인되었습니다.



### Adversarial Federated Consensus Learning for Surface Defect Classification Under Data Heterogeneity in IIo (https://arxiv.org/abs/2409.15711)
- **What's New**: 이 논문에서는 데이터 이질성과 개인 정보 보호 문제를 해결하기 위한 새로운 개인화된 연합 학습 방법론인 Adversarial Federated Consensus Learning (AFedCL)을 제안합니다. 이 방법은 여러 클라이언트 간 협업 학습을 통해, 각 클라이언트의 특정 데이터 분포에 최적화된 개인화된 모델을 개발합니다.

- **Technical Details**: AFedCL은 adversarial training을 활용하여 클라이언트 간 데이터 분포를 정렬하고, 글로벌 지식 망각 문제를 완화합니다. 이 접근 방식은 동적 합의 구조 전략과 합의 인식 집계 메커니즘(aggregation mechanism)을 결합하여 클라이언트의 글로벌 지식 학습 효율성을 기반으로 집계 가중치를 부여합니다. 또한, 글로벌 및 로컬 기능을 최적의 비율로 활용하기 위해 적응형 특징 융합 모듈을 설계하였습니다.

- **Performance Highlights**: AFedCL 방식은 세 가지 스트립 스틸 SDC 데이터셋에서 기존의 최신 방법인 FedALA에 비해 최대 5.67%의 정확도 증가를 달성했습니다.



### Autotuning Bipedal Locomotion MPC with GRFM-Net for Efficient Sim-to-Real Transfer (https://arxiv.org/abs/2409.15710)
- **What's New**: 이번 연구는 인간형 로봇의 2족 보행을 위한 파라미터 선택의 어려움을 해결하기 위해 DiffTune이라는 모델 기반 자동 튜닝 방법을 사용합니다. 이는 효율적인 파라미터 학습을 위해 미분 프로그래밍(differential programming)을 활용합니다.

- **Technical Details**: DiffTune은 낮은 충실도의 동역학 모델을 활용하여 미분 가능성을 확보하고, Ground Reaction Force-and-Moment Network (GRFM-Net)를 통해 MPC 명령과 실제 제어 효과 간의 차이를 포착합니다. 이 연구에서는 HECTOR V2 로봇을 사용하여 실험이 진행되었습니다.

- **Performance Highlights**: DiffTune을 통해 학습된 파라미터는 실제 하드웨어 실험에서 최적성을 보여주며, 전문가가 조정한 파라미터에 비해 총 손실을 최대 40.5% 감소시켰습니다. GRFM-Net은 시뮬레이션 학습된 파라미터의 현실적 적용 가능성을 개선하는 데 큰 역할을 했습니다.



### Improving Emotional Support Delivery in Text-Based Community Safety Reporting Using Large Language Models (https://arxiv.org/abs/2409.15706)
- **What's New**: 이 연구는 텍스트 기반 안전 보고 시스템에서의 정서적 지원의 전달 방식에 대한 새로운 실증적 통찰을 제공합니다. 특히, dispatcherLLM이라는 정밀 조정된 대형 언어 모델(LLM)을 개발하여 응급 서비스의 정서적 지원 품질을 향상시키기 위한 기초 자료를 제공합니다.

- **Technical Details**: 본 연구는 130개의 고등 교육 기관에서 2년간 수집된 57,114개의 메시지를 포함한 대화 로그를 분석하였습니다. 연구 결과, 사건 유형과 서비스 시간에 따라 정서적 지원의 제공이 크게 달라지며, 시간이 흐를수록 지원이 감소하는 경향이 발견되었습니다. 이를 통해 dispatcherLLM이 정서적 지원을 일관되고 효과적으로 제공 할 수 있는 가능성이 도출되었습니다.

- **Performance Highlights**: dispatcherLLM은 현업의 인력 및 기존의 LLM(도메인 특화 훈련을 받지 않은)과 비교하였을 때 다양한 사건에 대해 더 일관되고 효과적인 정서적 지원을 제공함을 보여주었습니다. 사용자 평가는 dispatcherLLM의 지원이 보다 효율적이라는 것을 입증하였습니다.



### dnaGrinder: a lightweight and high-capacity genomic foundation mod (https://arxiv.org/abs/2409.15697)
- **What's New**: dnaGrinder는 유전자 서열 내의 복잡한 장기 종속성을 효과적으로 관리하면서도 계산 비용을 최소화하는 독창적이고 효율적인 유전체 모델로, 기존의 모델들보다 우수한 성능을 보여줍니다.

- **Technical Details**: dnaGrinder는 Byte Pair Encoding (BPE) 토크나이제이션을 사용하여 DNA 서열을 수치 표현으로 변환하고, Attention with Linear Bias (ALiBi) 기법을 사용하며, Flash Attention 2와 같은 고급 주의 메커니즘을 통합하여 성능과 효율성을 극대화합니다.

- **Performance Highlights**: dnaGrinder는 Nucleotide Transformer 및 DNABERT-2와 같은 최신 DNA 모델에 비해 성능이 동등하거나 우수하며, 단일 고성능 GPU에서 140,000 토큰 이상의 서열을 지원합니다.



### Toward Mixture-of-Experts Enabled Trustworthy Semantic Communication for 6G Networks (https://arxiv.org/abs/2409.15695)
Comments:
          8 pages, 3 figures

- **What's New**: 본 논문에서는 Mixture-of-Experts (MoE) 기반의 시맨틱 커뮤니케이션(SemCom) 시스템을 도입하여 6G 네트워크의 보안과 신뢰성을 높이고자 합니다. 이 시스템은 다양한 보안 도전에 특화된 여러 전문가를 통해 동시 이종 공격에 대응할 수 있는 능력을 제공합니다.

- **Technical Details**: MoE 모델은 게이팅 네트워크(gating network)와 여러 전문가(experts)로 구성되며, 게이팅 네트워크는 사용자가 정의한 보안 요구 사항에 따라 적절한 전문가를 선택합니다. 이러한 동적 선택 과정은 SemCom 시스템의 보안 요구 사항에 맞춰 시맨틱 인코딩과 방어 전략을 조정할 수 있게 합니다.

- **Performance Highlights**: 모의 실험(case study) 결과는 제안된 MoE 기반 SemCom 시스템이 여러 이종 공격에 효과적으로 대응할 수 있으며, 하위 작업의 정확성에 미치는 영향이 최소화된다는 것을 보여줍니다.



### Safe Navigation for Robotic Digestive Endoscopy via Human Intervention-based Reinforcement Learning (https://arxiv.org/abs/2409.15688)
- **What's New**: 최근에 발표된 논문에서는 자동화된 로봇 소화 내시경(Robotic Digestive Endoscopy, RDE)의 안전성과 효율성을 극대화하기 위한 새로운 프레임워크인 HI-PPO(Human Intervention-based Proximal Policy Optimization)를 제안하고 있습니다. 이 프레임워크는 전문가의 지식을 통합하여 RDE의 안전성을 향상시키고, 탐색 효율성을 높이기 위한 Enhanced Exploration Mechanism (EEM)과 초기 개입 시 안전하지 않은 행동을 처벌하는 보상-벌칙 조정(Reward-Penalty Adjustment, RPA) 기법을 도입하고 있습니다.

- **Technical Details**: HI-PPO 프레임워크는 강화 학습(Reinforcement Learning, RL)을 기반으로 하고 있으며, 전통적인 Proximal Policy Optimization (PPO)의 탐색 효율성을 높이기 위해 EEM을 적용합니다. 또한, RPA를 사용하여 초기 개입 동안 안전하지 않은 행동에 대해 벌칙을 부과하여 보다 안전한 정책 학습을 촉진하고, Behavior Cloning Similarity (BCS)를 통해 에이전트가 전문가의 행동을 모사하도록 보조 목표를 설정합니다. 이 모든 기법들은 로봇 소화 내시경의 안전한 내비게이션을 보장하는 데 기여합니다.

- **Performance Highlights**: 다양한 해부학적 대장 구간에서의 시뮬레이션 실험 결과, HI-PPO 모델은 RDE를 안전하고 효과적으로 안내할 수 있음을 입증하였습니다. 이는 기존의 자동화된 RL 내비게이션 알고리즘이 가진 위험한 충돌 문제를 해결하는 데 중요한 성과입니다.



### Mitigating Semantic Leakage in Cross-lingual Embeddings via Orthogonality Constrain (https://arxiv.org/abs/2409.15664)
Comments:
          18 pages, 16 figures

- **What's New**: 이 논문에서는 크로스-링구얼(Cross-lingual) 문장 임베딩에서 의미와 언어를 분리하는 새로운 방법인 ORACLE(ORthogonAlity Constraint LEarning)를 제안합니다. 기존의 방법들이 의미 누수(semantic leakage) 문제로 고통받고 있음을 발견하였습니다.

- **Technical Details**: ORACLE은 두 가지 요소, 즉 intra-class clustering과 inter-class separation을 기반으로 합니다. 이는 의미 임베딩과 언어 임베딩 간의 직교성을 보장하여, 의미와 언어 정보의 분리를 효과적으로 지원합니다.

- **Performance Highlights**: 실험 결과, ORACLE을 사용한 훈련은 의미 누수를 줄이고 임베딩 공간 내에서 의미 정렬을 향상시키는 데 성공했습니다. 이는 크로스-링구얼 검색(cross-lingual retrieval) 및 의미 텍스트 유사성(semantic textual similarity) 작업에서 입증되었습니다.



### Double-Path Adaptive-correlation Spatial-Temporal Inverted Transformer for Stock Time Series Forecasting (https://arxiv.org/abs/2409.15662)
- **What's New**: DPA-STIFormer라는 새로운 Spatial-Temporal Transformer 모델이 주식 예측을 위한 혁신적인 방법으로 제안되었습니다. 이 모델은 시간 단계를 토큰으로 사용하지 않고, 특징 변화에 따라 각 노드를 모델링함으로써 더 포괄적으로 동적인 공간 정보를 추출합니다.

- **Technical Details**: DPA-STIFormer는 Double-Path Adaptive-correlation Inverted Encoder와 Decomposed Fitting을 포함한 단일 Decoder Block으로 구성됩니다. 이 모델은 Softmax Attention 메커니즘을 사용하여 중요 가중치를 포함한 토큰 간의 상관관계를 학습하고, Double-Path 방식으로 서로 다른 두 종류의 상관관계를 통합합니다.

- **Performance Highlights**: 4개의 주식 데이터셋에 대한 실험 결과, DPA-STIFormer는 주식 시장의 공간-시간 상관관계를 더 효과적으로 모델링할 수 있는 능력을 보여주며, 기존 방법들보다 뛰어난 성능을 달성했습니다.



### ReLEP: A Novel Framework for Real-world Long-horizon Embodied Planning (https://arxiv.org/abs/2409.15658)
- **What's New**: ReLEP는 실제 세계의 장기적인 인체 계획을 위한 새로운 프레임워크로, 다양한 일상 작업을 수행할 수 있는 능력을 가지고 있다. 이 프레임워크는 세분화된 기술 기능의 순서로 계획을 수립하며, 입력 지시 및 장면 이미지에 따라 조정된다.

- **Technical Details**: ReLEP의 핵심은 세밀하게 조정된 대형 비전 언어 모델로, 이 모델은 계획을 기술 기능의 연속으로 구성한다. 또한 메모리 모듈과 로봇 구성 모듈을 갖추어 다양한 로봇 유형에 대한 유연성을 지원하며, 반자동 데이터 생성 파이프라인을 통해 데이터 부족 문제를 해결한다.

- **Performance Highlights**: 오프라인 실험에서 ReLEP는 8개의 일상적인 장기 작업을 수행하며 기존의 최첨단 방법들과 비교해 모든 방식에서 우수한 성능을 보였고, 제안된 프레임워크의 효과를 입증하였다.



### Personalized Federated Learning via Backbone Self-Distillation (https://arxiv.org/abs/2409.15636)
Comments:
          Pubished in ACM MMAsia 2023

- **What's New**: 이 논문에서는 개인화된 연합 학습을 촉진하기 위해 백본(Backbone) 자기 증류(Self-Distillation) 접근 방식을 제안합니다. 각 클라이언트가 로컬 모델을 훈련하고 오직 백본 가중치만 서버에 전송하는 방식을 사용합니다.

- **Technical Details**: 각 클라이언트 모델은 공유 백본과 개인 헤드(Head)로 나뉘어 있으며, 서버는 백본 가중치만 집계하여 글로벌 백본을 구축합니다. 그런 다음 각 클라이언트는 글로벌 백본을 교사(Teacher)로 사용하여 로컬 백본을 업데이트 합니다.

- **Performance Highlights**: 12개의 최신 접근 방식과 비교한 실험 결과, 제안된 방법이 성능을 크게 향상시키며, 글로벌 지식 전이(Global Knowledge Transfer)를 통해 클라이언트 모델의 개인화를 효과적으로 지원함을 보여주었습니다.



### Data Augmentation for Sparse Multidimensional Learning Performance Data Using Generative AI (https://arxiv.org/abs/2409.15631)
- **What's New**: 이 논문에서는 학습 성과 데이터의 희소성(data sparsity) 문제를 해결하기 위해 학습 데이터 보강을 위한 체계적인 프레임워크를 제안하고 있습니다. 주목할 점은 텐서 분해(tensor factorization) 기법을 활용하여 결측값을 보간하고 생성적 인공지능(generative AI) 모델을 통해 다양한 학습 성과 패턴을 생성하는 것입니다.

- **Technical Details**: 연구는 학습자의 질문, 답변 및 시도를 포함하는 3차원 텐서로 학습 성과를 표현합니다. 결측값 보간을 위해 텐서 분해 기법을 사용하며, 이는 지식 추적(knowledge tracing) 작업에 기반하여 실측값을 바탕으로 결측 성과 값을 예측할 수 있도록 합니다. 생성적 모델로는 Generative Adversarial Networks (GANs)와 Generative Pre-Trained Transformers (GPT)를 비교하여 다양한 학습 데이터 클러스터의 데이터를 생성합니다.

- **Performance Highlights**: 연구 결과 텐서 분해를 통한 데이터 보강이 지식 마스터리 추적 및 예측 성과를 향상시키며, GAN 기반의 시뮬레이션이 GPT와 비교해 통계적 편향이 적고 전반적인 안정성이 뛰어난 것으로 나타났습니다. 이는 저자의 실험적으로 개발된 성인 읽기 이해(AutoTutor) 데이터셋에서 확인되었습니다.



### Safe Guard: an LLM-agent for Real-time Voice-based Hate Speech Detection in Social Virtual Reality (https://arxiv.org/abs/2409.15623)
- **What's New**: 이 논문에서는 VRChat과 같은 사회적 VR 환경에서 음성 기반 상호작용에서 증오 발언을 감지하기 위한 LLM-agent인 Safe Guard를 제안합니다. 이 시스템은 Open AI GPT와 오디오 피처 추출 (audio feature extraction)을 활용하여 실시간 음성 상호작용을 처리합니다.

- **Technical Details**: Safe Guard는 LLM 기반의 에이전트로, 음성 기반의 증오 발언을 신속하게 감지할 수 있도록 설계되었습니다. 기존의 방법들과 비교하여 높은 정확도로 결과를 도출하며, 허위 양성 사례 (false positives)를 줄이는 데 성공했습니다.

- **Performance Highlights**: 연구 결과, LLM 기반의 에이전트가 안전한 가상 환경을 조성하는 데 잠재력을 가지고 있으며, 향후 LLM-driven moderation 접근방법의 발전을 위한 기초를 제공한다고 밝혔습니다.



### Revolutionizing Biomarker Discovery: Leveraging Generative AI for Bio-Knowledge-Embedded Continuous Space Exploration (https://arxiv.org/abs/2409.15612)
- **What's New**: 본 논문은 개인 맞춤형 의학에서 중요한 바이오마커(​biomarker) 식별을 자동화하는 새로운 프레임워크를 제안합니다. 이는 복잡한 생물학적 시스템의 분석 수고를 줄이고, 기계 학습을 통해 보다 효율적이고 효과적인 예측 결과를 도출하는 것을 목표로 합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 주요 모듈로 구성됩니다: 1) 훈련 데이터 준비, 2) 임베딩(embedding) 최적화-생성(generation). 첫 번째 모듈은 다중 에이전트 시스템을 사용하여 바이오마커 서브셋과 그에 대한 예측 정확성의 쌍을 자동으로 수집하여 훈련 데이터를 생성합니다. 두 번째 모듈은 변환기(transformer) 기반의 구조를 바탕으로 임베딩 공간을 학습하며, 예측 정확성을 향상시키기 위한 최적화를 수행합니다.

- **Performance Highlights**: 이 연구는 세 개의 실제 데이터셋에 대한 광범위한 실험을 통해 제안된 방법의 효율성, 강건성(robustness), 효과성을 입증하였습니다. 자동화된 바이오마커 식별을 통해, 더 적은 노동력으로 더욱 정확한 예측 결과를 도출할 수 있는 가능성을 보여주었습니다.



### TFT-multi: simultaneous forecasting of vital sign trajectories in the ICU (https://arxiv.org/abs/2409.15586)
- **What's New**: 본 논문에서는 다중 생체 징후(정맥압, 맥박, SpO2, 체온 및 호흡수)를 동시에 예측할 수 있는 새로운 프레임워크인 TFT-multi를 제안합니다. 이는 기존의 단일 생체 징후 예측 방식보다 더 현실적인 임상 환경에 적합한 모델입니다.

- **Technical Details**: TFT-multi는 기존 temporal fusion transformer (TFT)의 확장으로, 멀티 변수 예측을 지원하기 위해 입력-출력 구조와 손실 함수를 수정하였습니다. 이를 통해 15분 간격으로 생체 징후를 예측할 수 있으며, 학습 과정에서 마스킹 기법을 사용하여 결측치에 대한 편향을 줄입니다.

- **Performance Highlights**: 이 모델은 MIMIC 데이터셋과 독립적인 기관 데이터셋에서 성능 검증을 통과했으며, 기존의 Prophet, TFT 및 벡터 자기 회귀 모델보다 높은 예측력을 보였습니다. 특히, 예측력 향상은 상관성이 높은 생체 징후를 동시에 예측함으로써 이루어졌습니다.



### FACET: Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality (https://arxiv.org/abs/2409.15584)
Comments:
          8 pages, 5 figures

- **What's New**: 본 논문에서는 FACET(Fast and Accurate Event-based Eye Tracking)이라는 새로운 신경망 모델을 제안하여, 이벤트 데이터를 기반으로 눈동자의 타원 매개변수를 실시간으로 출력합니다. 이 모델은 XR(Extended Reality) 애플리케이션에 최적화되어 있으며, 기존의 프레임 기반 시스템이 가진 한계점을 극복하고자 합니다.

- **Technical Details**: FACET은 이벤트 기반의 경량화된 눈동자 검출기로, 기존의 이벤트 데이터에서 직접 타원을 예측하는 방식으로 동작합니다. 이를 위해 EV-Eye 데이터셋을 증강하고, 새로운 삼각법 손실 함수를 도입하여 타원 매개변수의 각도 불연속성 문제를 해결했습니다. 또한, 이벤트 볼륨의 수치화 방법을 설계하여 이벤트 표현값의 분포를 정규화했습니다.

- **Performance Highlights**: FACET은 향상된 EV-Eye 테스트 세트에서 평균 눈동자 중심 오류 0.20 픽셀과 0.53 ms의 추론 시간을 달성하여, 기존의 EV-Eye보다 픽셀 오류를 1.6배, 추론 시간을 1.8배 줄였습니다. 또한, 모델은 4.4배 적은 파라미터 수와 11.7배 적은 산술 연산으로 구현되었습니다.



### Asking an AI for salary negotiation advice is a matter of concern: Controlled experimental perturbation of ChatGPT for protected and non-protected group discrimination on a contextual task with no clear ground truth answers (https://arxiv.org/abs/2409.15567)
- **What's New**: 이 연구는 네 가지 버전의 ChatGPT를 대상으로 진행된 통제된 실험 편향 감사(bias audit)를 소개합니다. 연구진은 각각의 버전에게 새로운 채용자를 위한 초봉 제안을 요청했으며, 직원의 성별, 대학, 전공 등을 체계적으로 변화시켜 98,800개의 프롬프트를 제출했습니다. 이러한 실험을 통해 ChatGPT가 이러한 작업에 신뢰할 수 없다는 것을 발견했습니다.

- **Technical Details**: 대조군을 포함한 제어된 실험 방법을 사용하여 AI가 질문에 대해 차별적인 응답을 하는지 혹은 공정한 결과를 도출하는지를 평가했습니다. 특히 성별 변화에 따른 통계적으로 유의미한 초봉 제안 차이를 관찰했으며, 사기성 대학 및 전공에 대한 경험적 결과도 비교적 일관되지 않음을 확인하였습니다. 이 연구는 AI/ML의 공정성(fairness) 및 신뢰성(trustworthiness) 문헌에 기여합니다.

- **Performance Highlights**: ChatGPT의 네 가지 모델 버전 간의 초봉 차이가 상이하였으며, 성별, 대학 및 전공에 따라 상당한 격차가 관찰되었습니다. 특히, 고용주와 직원의 목소리에 따라 제안된 급여의 차이가 두드러졌고, 이는 ChatGPT 다중 모델 플랫폼의 일관성 및 신뢰성에 대한 우려를 제기합니다.



### GEM-RAG: Graphical Eigen Memories For Retrieval Augmented Generation (https://arxiv.org/abs/2409.15566)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 Retrieval Augmented Generation (RAG) 방식을 개선하여, 메모리 조작을 통한 AI의 성능 향상을 목표로 하고 있습니다. 저자들은 Graphical Eigen Memories For Retrieval Augmented Generation (GEM-RAG)라는 새로운 방법론을 제안하며, 이는 각 텍스트 조각에 대한 '유틸리티(utility)' 질문을 생성하고, 이를 기반으로 메모리 그래프를 구축하여 더 높은 수준의 요약 노드를 생성하는 방법입니다.

- **Technical Details**: GEM-RAG는 주어진 텍스트 코퍼스를 조각으로 분할한 후, LLM을 이용해 관련된 유틸리티 질문을 생성합니다. 생성된 질문의 임베딩은 가중치 그래프를 형성하며, 이 그래프의 고유값 분해를 통해 텍스트의 주요 테마를 포착하는 'eigenthemes' 또는 요약 노드를 생성합니다. 이 방법론은 두 개의 QA 데이터셋, QuALITY와 Qasper에서 성능을 평가하며, 표준 RAG 절차 및 최신 방법인 RAPTOR와 비교하여 우수성을 입증했습니다.

- **Performance Highlights**: GEM-RAG는 두 개의 QA 태스크에서 다른 최신 RAG 방법들과 비교하여 더 나은 성능을 보였습니다. 실험 결과에 따르면, LLM에 의해 생성된 요약 노드와 유틸리티 질문의 수가 모델 성능에 미치는 영향을 정량적으로 분석하여, GEM의 효과성을 뒷받침하는 세부 실험을 수행하였습니다.



### Revise, Reason, and Recognize: LLM-Based Emotion Recognition via Emotion-Specific Prompts and ASR Error Correction (https://arxiv.org/abs/2409.15551)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)이 감정 인식에 효과적일 수 있음을 보여주었으나, 이들의 효용성에 대한 의문이 여전히 존재합니다. 본 논문에서는 음향학(acoustics), 언어학(linguistics), 심리학(psychology)에서의 감정별 지식을 통합한 새로운 프롬프트(prompts)를 제안하고, LLM 기반의 감정 인식의 정확성과 효과성을 실험을 통해 검증합니다.

- **Technical Details**: 우리는 Revise-Reason-Recognize (R3) 프롬프트 파이프라인을 제안하여, 부정확한 텍스트에 대한 감정 인식을 개선합니다. 이 파이프라인은 ASR(Automatic Speech Recognition) 오류를 수정하고, LLM이 감정 인식에 필요한 자율적 설명을 제공하는 방법으로 구성됩니다. 이 외에도, 문맥 인식 학습(context-aware learning)과 지시 조정(instruction tuning) 방법을 실험하여 성능을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 감정별 지식을 통합한 프롬프트와 ASR 오류 교정이 LLM 기반 감정 인식에 효과적임을 확인하였습니다. R3 프롬프트는 ASR 전사 과정에서 발생하는 오류를 정확히 수정하고, 감정 인식의 정확도를 높이는 데 기여하였습니다. 또한, 훈련 스킴들은 LLM의 성능을 향상시키는 데 중요한 역할을 합니다.



### CANDERE-COACH: Reinforcement Learning from Noisy Feedback (https://arxiv.org/abs/2409.15521)
- **What's New**: 본 논문에서는 비정상적인 피드백(noisy feedback)으로부터 학습할 수 있는 CANDERE-COACH 알고리즘을 제안합니다. 이 알고리즘은 불완전한 교사(teacher)로부터의 피드백을 처리하여 RL 에이전트가 성공적으로 학습할 수 있도록 해줍니다.

- **Technical Details**: 제안된 CANDERE-COACH 알고리즘은 노이즈 필터링 메커니즘을 포함하여 이미 40%의 잘못된 교사 피드백을 포함한 상황에서도 작동할 수 있습니다. 알고리즘은 피드백을 긍정적(positive) 및 부정적(negative)으로 분류하여 RL 에이전트에게 필요한 정보를 제공합니다. 또한, noise detecting module을 통해 시각적 데이터에서 노이즈를 감지하고 필터링합니다.

- **Performance Highlights**: 세 가지 일반 도메인에서의 실험 결과, 제안된 방법이 기초 선행 연구(baselines)보다 비약적으로 뛰어난 성능을 보여 주었으며, 노이즈가 포함된 피드백에서도 RL 에이전트가 효과적으로 학습할 수 있다는 것을 입증했습니다.



### Learning When to Retrieve, What to Rewrite, and How to Respond in Conversational QA (https://arxiv.org/abs/2409.15515)
Comments:
          Accepted in EMNLP (findings) 2024

- **What's New**: 이 논문에서는 대화 맥락에서의 정보 검색 필요성을 판단하여 retrieval을 수행하는 방법인 SELF-multi-RAG를 제안합니다. 이는 대화형 질문 답변(QA) 시스템의 맥락 이해 및 응답 생성의 질을 개선하기 위한 연구입니다.

- **Technical Details**: SELF-multi-RAG 모형은 기존의 SELF-RAG(Asai et al., 2023) 프레임워크를 기반으로 하여, 대화 중 필요한 경우에만 검색을 수행하고, 검색된 문서를 요약하여 유용한 응답을 생성하는 과정에서의 효과를 개선합니다. 이는 대화의 요약된 맥락을 사용하여 관련 문서를 검색하도록 설계되었습니다.

- **Performance Highlights**: SELF-multi-RAG는 실험을 통해 전통적인 SELF-RAG보다 약 13%의 응답 품질 향상을 보여주었으며, 검색 효과성(R@5) 또한 평균 13.5% 향상되었습니다. 이러한 결과는 human annotation에 의해 검증되었습니다.



### PixelBytes: Catching Unified Embedding for Multimodal Generation (https://arxiv.org/abs/2409.15512)
- **What's New**: 이 보고서는 PixelBytes Embedding이라는 새로운 접근 방식을 소개합니다. 이는 통합된 멀티모달 표현 학습(unified multimodal representation learning)을 위한 방법으로, 다양한 입력을 단일하고 일관된 표현으로 캡처하여 텍스트 및 픽셀화된 이미지에 대한 멀티모달 시퀀스 생성의 우수한 특성을 조화롭게 구현할 수 있도록 합니다.

- **Technical Details**: PixelBytes는 최신 시퀀스 모델인 Image Transformers, PixelCNN 및 Mamba-Bytes에서 영감을 받아 서로 다른 데이터 타입을 통합하는 데에 어려움을 해결하는 것을 목표로 합니다. RNNs(순환신경망), SSMs(상태공간모델), 주의(Attention) 기반 모델 등 다양한 모델 아키텍처를 탐구하며, 양방향 처리(bidirectional processing)와 혁신적인 PxBy embedding 기술에 중점을 둡니다.

- **Performance Highlights**: PixelBytes Pok{é}mon 데이터셋을 사용한 실험에서, PxBy embedding과 합성곱(convolutional) 레이어를 사용한 양방향 시퀀스 모델이 일관된 멀티모달 시퀀스를 생성할 수 있는 가능성을 입증했습니다. 이 작업은 통합 AI 모델의 발전에 기여하며, 멀티모달 데이터를 이해하고 생성하는 데 있어 보다 통합된 방식을 가능하게 합니다.



### Computational Pathology for Accurate Prediction of Breast Cancer Recurrence: Development and Validation of a Deep Learning-based Too (https://arxiv.org/abs/2409.15491)
- **What's New**: Breast cancer의 재발 위험 예측을 위한 새로운 딥 러닝 기반 방법론인 Deep-BCR-Auto를 소개합니다. 이 방법은 H&E 염색된 전체 슬라이드 이미지(WSI)에서 정보를 추출하여 재발 위험을 예측합니다.

- **Technical Details**: Deep-BCR-Auto는 TCGA-BRCA 데이터셋과 오하이오 주립대학교(OSU) 내부 데이터셋을 사용하여 검증되었습니다. 이 모델은 환자를 저위험 및 고위험 범주로 효과적으로 분류할 수 있으며, AUROC 값이 각각 0.827(TCGA-BRCA)와 0.832(OSU)로 나타났습니다.

- **Performance Highlights**: Deep-BCR-Auto는 82.0%의 정확도, 85.0%의 특이도, 67.7%의 민감도를 보여주며, 기존 약한 지도 학습 모델에 비해 유의미한 성과를 보였습니다(p=0.041). 이는 저비용으로 개인화된 치료 전략에 대한 접근성을 높일 수 있는 가능성을 나타냅니다.



### VLMine: Long-Tail Data Mining with Vision Language Models (https://arxiv.org/abs/2409.15486)
- **What's New**: 본 연구에서는 사용되지 않은 라벨 데이터를 통해 드문(long-tail) 예제들을 식별하는 데이터 마이닝(data mining) 방법론을 제안합니다. 제안된 방법은 VLM(Vision Language Model)을 활용하여 이미지 내용을 키워드(keyword) 집합으로 요약하고, 키워드의 빈도를 기반으로 드문 예제를 식별합니다.

- **Technical Details**: 제안된 방법인 VLMine은 VLM에서 추출한 지식을 활용하여 드문 예제를 식별합니다. 기존 모델의 불확실성(uncertainty) 기반 방식보다 VLM의 키워드 빈도 분석이 더 효과적인 신호를 제공하는 것을 보여줍니다. VLMine은 특정 태스크와 무관하게 사용할 수 있는 모델-불가지론적(data-agnostic) 방법입니다.

- **Performance Highlights**: VLMine을 통해 2D 이미지 분류 및 3D 객체 탐지 태스크에서 10%에서 50%의 성능 향상을 달성하였으며, ImageNet-LT, Places-LT, Waymo Open Dataset에서의 벤치마크 시험에서 기존 방법들에 비해 일관된 개선 결과를 보여주었습니다.



### In-Context Learning May Not Elicit Trustworthy Reasoning: A-Not-B Errors in Pretrained Language Models (https://arxiv.org/abs/2409.15454)
Comments:
          Accepted at EMNLP 2024 Findings

- **What's New**: 대규모 언어 모델(LLMs)의 억제 제어(inhibitory control) 능력을 체계적으로 평가한 최초의 연구로, A-Not-B 오류라는 아동 인지 현상을 바탕으로 디자인한 멀티 초이스 QA 시나리오를 통해 LLMs가 얼마나 잘 고착된 답변 패턴을 억제할 수 있는지 테스트함.

- **Technical Details**: 본 연구에서는 유지된 답변 패턴을 억제하는 LLM의 능력을 평가하기 위해 A-Not-B 실험 설정을 텍스트 기반으로 변환하였습니다. LLM에게 같은 정답 선택을 반복적으로 제공하여 패턴을 형성한 뒤, 새로운 질문을 통해 패턴을 변경하는 방식으로 테스트를 진행하였으며, A-Not-B 프롬프트(A-Not-B prompting) 전략을 적용하였습니다.

- **Performance Highlights**: 최신 LLM(예: Llama3-8b)은 3-shot A-Not-B 프롬프트 사용시, 일부 추론 작업에서 정확도가 83.3% 감소하는 심각한 오류를 보였으며, 이는 이들이 초등학생보다도 낮은 인지 능력을 가지고 있음을 나타냅니다. LLM의 오류 발생 원인에 대한 분석 결과, 모델 크기와 데이터 품질이 중요한 요소로 작용하며, 후속 학습 단계에서 자기 설명(self-explanation) 전략이 어느 정도 효과를 보였음을 확인하였습니다.



### Tag Map: A Text-Based Map for Spatial Reasoning and Navigation with Large Language Models (https://arxiv.org/abs/2409.15451)
- **What's New**: 본 연구에서는 대형 이미지 인식 모델을 활용하여 수천 개의 의미 클래스를 명시적으로 표현할 수 있는 텍스트 기반 맵을 제안합니다. 이 맵은 대형 언어 모델(LLM)과 쉽게 통합되며, 로봇이 사용자 작업을 해결하기 위해 PLANS (작업 계획)를 생성하는 데 필요한 장면 정보를 제공합니다.

- **Technical Details**: 제안된 태그 맵은 이미지 태깅 모델이 인식한 고유 엔티티(태그)를 저장하고, 각 태그가 인식된 시점(뷰포인트)과 연관됩니다. 이 맵은 메모리의 효율성을 극대화하기 위해 비구조적인 데이터베이스로 구현됩니다. 또한, 3D 로컬라이제이션을 통해 태그와 관련된 지역을 생성하는 과정을 설명합니다.

- **Performance Highlights**: 정량적 실험을 통해 제안된 태그 맵의 로컬라이제이션 성능이 최신 개방형 어휘 맵과 비교하여 정확도와 재현성을 유지하면서도 사용하는 메모리를 몇 배나 줄일 수 있음을 보여줍니다. 실제 로봇 실험에서도 태그 맵이 LLM을 기반으로 사용자 요청을 처리하고 실행 가능한 내비게이션 계획을 생성하는 데 효과적임을 입증했습니다.



### Attack Atlas: A Practitioner's Perspective on Challenges and Pitfalls in Red Teaming GenAI (https://arxiv.org/abs/2409.15398)
- **What's New**: 생성적 AI(Generative AI)와 대형 언어 모델(LLMs)의 보안에 대한 새로운 접근 방식으로, 레드 팀(Red Team)과 블루 팀(Blue Team)의 전략을 결합해 실제 레퍼런스를 제공하고자 합니다.

- **Technical Details**: 레드 팀은 생성적 AI 시스템의 취약점을 능동적으로 탐색하는 역할을 하며, 블루 팀은 이러한 공격으로부터 시스템을 보호하는 역할을 합니다. 이 연구는 Prompt Injection, Jailbreak Attack과 같은 공격 기법에도 중점을 둡니다.

- **Performance Highlights**: 이 연구는 'Attack Atlas'라는 직관적인 프레임워크를 제시하여, 단일 입력 공격 벡터를 분석하는 실용적인 접근 방식을 제공합니다.



### Parse Trees Guided LLM Prompt Compression (https://arxiv.org/abs/2409.15395)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 입력 프롬프트를 압축하는 새로운 방법인 PartPrompt를 소개합니다. 기존의 압축 방법들은 주로 언어 모델을 사용하여 새로운 프롬프트를 생성하거나 중요한 부분을 선택하는 방식으로 이루어졌으나, PartPrompt는 구문 트리를 기반으로 하여 로컬 정보 엔트로피를 활용하고, 구문 간의 연결 패턴을 고려하여 압축을 수행합니다.

- **Technical Details**: PartPrompt는 각 문장의 파싱 트리를 생성하고, 각 노드에 대해 로컬 정보 엔트로피를 계산합니다. 이러한 지역 파싱 트리는 문장, 단락 및 섹션의 계층적 구조에 따라 전역 트리로 구성됩니다. 이후, 연구자들은 새로운 방식인 root-ward propagation 및 leaf-ward propagation을 이용하여 전역 트리의 노드 값을 조정합니다. 마지막으로, 조정된 노드 값을 기반으로 전역 트리를 가지치기하는 재귀 알고리즘이 개발되었습니다.

- **Performance Highlights**: 실험 결과, PartPrompt는 다양한 데이터 세트와 메트릭, 압축 비율 및 LLM의 타겟 모델에서 최신 성과를 기록했습니다. 특히, PartPrompt는 긴 프롬프트에서의 일관성(Corehence) 측면에서도 우수한 성능을 보여주었습니다. 연구진은 이 방법이 기존의 프롬프트 압축 방법들보다 효율적임을 증명했습니다.



### Neural Control Variates with Automatic Integration (https://arxiv.org/abs/2409.15394)
- **What's New**: 이 논문은 임의의 신경망 아키텍처를 활용하여 control variates를 구축하는 방법을 제안합니다. 기존의 접근법이 충분히 표현력이 없는 함수에 의존했던 반면, 본 연구는 learnable parametric model을 이용하여 이 문제를 해결하려고 합니다.

- **Technical Details**: 우리는 신경망을 사용해 적분 함수의 anti-derivative를 근사화하여 control variate 함수를 구성합니다. 이 방법을 Walk-on-sphere 알고리즘에 적용하여 편미분 방정식을 해결하였습니다. 자동 미분(automatic differentiation)을 통해 적분이 이루어질 수 있도록 하는 것이 핵심입니다.

- **Performance Highlights**: 우리의 결과는 다양한 네트워크 아키텍처를 사용하여 기존의 control variate 방법보다 낮은 분산(variance)을 달성함을 보여줍니다. 이 연구는 Monte Carlo integration의 variance를 줄이는 데 있어 중요한 첫 걸음을 내딛었습니다.



### Approximated Orthogonal Projection Unit: Stabilizing Regression Network Training Using Natural Gradien (https://arxiv.org/abs/2409.15393)
- **What's New**: AOPU(Approximated Orthogonal Projection Unit)는 신경망(Neural Network)의 훈련 안정성을 향상시키고 최소 분산 추정(MVE)에 도달할 수 있도록 설계된 새로운 네트워크입니다. AOPU는 이중 매개변수로 그래디언트 역전파를 잘라내는 방식을 제안하며, 이를 통해 신뢰할 수 있는 훈련 안정성을 제공합니다.

- **Technical Details**: AOPU는 주요 두 개의 매개변수를 도입하여 각각의 매개변수가 추적 가능한지 여부를 구분합니다. 이중 매개변수는 추적 가능한 매개변수의 그래디언트 업데이트를 최적화하며, 일부는 자연 그래디언트(NG)에 근사화됩니다. Rank Ratio(RR)라는 해석 가능성 지표를 통해 네트워크의 동역학을 심층적으로 분석할 수 있으며, RR은 샘플의 독립성 비율을 측정하여 모델의 성능 예측에 도움을 줍니다.

- **Performance Highlights**: AOPU는 두 개의 화학 프로세스 데이터셋에서 실험 결과, 기존의 다른 모델들보다 우수한 안정적 수렴성을 보여주어 소프트 센서 분야에서 중요한 발전으로 기록되었습니다.



### Adversarial Attacks on Parts of Speech: An Empirical Study in Text-to-Image Generation (https://arxiv.org/abs/2409.15381)
Comments:
          Findings of the EMNLP 2024

- **What's New**: 이번 연구는 텍스트-이미지(T2I) 모델에서 다양한 품사(POS) 태그에 대한 적대적 공격의 영향을 조사하며, 기존의 연구들이 주로 명사에 초점을 맞춘 것과 달리 다양한 품사를 다루는 첫 번째 데이터셋을 작성하였습니다.

- **Technical Details**: 고품질의 데이터셋을 통해 POS 태그의 실제 시나리오에서 토큰 교환을 수행하고, 기울기 기반 공격(gradient-based attacks)을 통해 T2I 모델이 잘못된 이미지를 생성하게 만드는 적대적 접미사(adversarial suffixes)를 찾아냈습니다. 연구 결과, 공격 성공률(ASR)은 명사, 고유명사, 형용사에 대해 가장 높았으며, 각각의 POS 태그 카테고리에 따른 공격 메커니즘의 차이를 설명하기 위해 실험을 수행했습니다.

- **Performance Highlights**: 명사와 형용사 공격 시 이미지 생성의 유효성이 높고, 예상하는 속성을 이미지에 포함하도록 유도할 수 있는 성공률이 높았습니다. 특히 동일한 적대적 접미사를 사용하여 다양한 입력 프롬프트에 대해 동일한 속성을 생성할 수 있는 전반적인 특성 또한 확인되었습니다.



### Kalahi: A handcrafted, grassroots cultural LLM evaluation suite for Filipino (https://arxiv.org/abs/2409.15380)
- **What's New**: 다국어 대형 언어 모델(LLMs)에 대한 연구가 진행되는 가운데, Kalahi라는 새로운 문화적 LLM 평가 도구가 필리핀 원주율 사용자를 대상으로 개발되었습니다. 이 도구는 필리핀 고유의 문화적 지식과 가치를 반영한 150개의 정교하게 수작업으로 작성된 프롬프트로 구성되어, LLM이 필리핀 사회에서 일어날 법한 상황에 대해 적절한 대답을 생성할 수 있는지 평가합니다.

- **Technical Details**: Kalahi는 LLM이 필리핀 문화에 기반한 질문에 얼마나 잘 응답할 수 있는지를 평가하기 위해 설계된 도구입니다. 이 평가는 다국어 및 필리핀어 지원 LLM에 대해 수행되었으며, 150개의 상황에 대한 프롬프트를 사용하여 LLM의 성능을 측정합니다. 또한, 프롬프트 작성자는 필리핀 원어민으로 이루어져 있으며, 다양한 사회적 배경을 가진 인물들이 참여하여 문화적 대표성을 보장합니다.

- **Performance Highlights**: 실험 결과, Kalahi는 필리핀 원주율 사용자에게는 쉬운 질문들이 포함되어 있지만, LLM에게는 도전적인 질문으로 나타났습니다. 가장 잘 수행된 LLM은 단지 46.0%의 질문에만 정확히 응답했으며, 필리핀 원주율 사용자의 정확도는 89.10%로 나타났습니다. 이러한 차이는 Kalahi가 LLM의 필리핀 문화 표현을 평가하는 데 있어 신뢰할 수 있는 도구임을 시사합니다.



### Toward Automated Clinical Transcriptions (https://arxiv.org/abs/2409.15378)
Comments:
          7 pages, 6 figures

- **What's New**: 본 논문은 최근의 speech-to-text (음성 인식) 및 speaker-labeling (화자 레이블링) 기술을 활용하여 환자-제공자 대화의 정확한 전사를 생성하고, 오류를 강조하여 신속한 인간 검증을 촉진하는 안전한 시스템을 소개합니다.

- **Technical Details**: 이 시스템은 40시간 이상의 시뮬레이션 대화에 적용되어 최적화 되었으며, 의료 문서화의 수작업 노력을 줄이기 위해 설계되었습니다. 특히, 불필요한 수작업을 최소화하여 임상 전사(Clinical Transcriptions)의 자동화를 위한 유망한 기초를 제공합니다.

- **Performance Highlights**: 이 시스템은 정확한 전사를 생성하는 데 있어 뛰어난 성능을 보이며, 의료 분야에서 피로도 증가 및 환자 관리 품질 저하와 같은 부정적인 결과를 완화하는 데 기여할 것으로 기대됩니다.



### Prompting Large Language Models for Supporting the Differential Diagnosis of Anemia (https://arxiv.org/abs/2409.15377)
- **What's New**: 본 연구는 임상 지침에 영감을 받아 비슷한 방식의 진단 경로(Pathways)를 개발하고, 이를 통해 희귀 질환 진단에 대한 한계를 극복하고자 하였습니다.

- **Technical Details**: 연구에서는 Generative Pretrained Transformer 4 (GPT-4), Large Language Model Meta AI (LLaMA), Mistral이라는 세 가지 대형 언어 모델(LLMs)을 이용하여 합성된 데이터셋을 기반으로 빈혈(Anemia) 및 그 하위 유형의 진단을 진행하였습니다. 고급 프롬프트 기법(advanced prompting techniques)을 사용하여 의사결정 프로세스를 향상시켰습니다.

- **Performance Highlights**: 실험 결과, LLMs는 환자 데이터를 기반으로 한 임상 경로 발견에서 큰 잠재력을 보였으며, 모든 실험에서 GPT-4가 최고의 성능을 나타냈습니다.



### ControlMath: Controllable Data Generation Promotes Math Generalist Models (https://arxiv.org/abs/2409.15376)
Comments:
          17 pages

- **What's New**: 본 연구에서는 데이터 증강(data augmentation)에 있어 대형 언어 모델(LLMs)의 제약 사항을 극복하기 위해 ControlMath라는 반복(iterative) 방법론을 소개합니다.

- **Technical Details**: ControlMath는 방정식 생성기(equation-generator) 모듈과 두 개의 LLM 기반 에이전트(agent)를 포함합니다. 방정식 생성 모듈은 다양한 방정식을 생성하고, Problem-Crafter 에이전트는 이를 수학적인 서술 문제로 변환합니다. Reverse-Agent는 'less is more' 원칙에 따라 고품질 데이터를 필터링하고 선택합니다.

- **Performance Highlights**: ControlMathQA는 190,000개의 수학 서술 문제(math word problems)를 포함하고 있으며, 이 데이터셋은 GSM8K와 같은 도메인 내(in-domain) 데이터셋과 결합함으로써 모델의 수학적 일반화(generalization) 능력을 향상시킵니다. 결과적으로 특정 도메인뿐만 아니라 그 너머에서도 성능이 개선되는 것을 보였습니다.



### DS2TA: Denoising Spiking Transformer with Attenuated Spatiotemporal Attention (https://arxiv.org/abs/2409.15375)
Comments:
          arXiv admin note: text overlap with arXiv:2311.09376

- **What's New**: 본 논문에서는 비전을 위한 새로운 아키텍처인 DS2TA(Denoising Spiking transformer with Attenuated SpatioTemporal Attention)를 소개하며, 이는 템포럴 차원에서 조정된 스페이쇼템포럴 어텐션(SpatioTemporal Attention) 메커니즘을 도입하여 기존 스파이킹 트랜스포머의 한계를 뛰어넘습니다.

- **Technical Details**: DS2TA는 입력 발화의 시간 및 공간에서 발생하는 상관관계를 고려하여, spiking 쿼리, 키, 값 및 최종 출력을 계산하는 TASA(Temporally Attenuated Spatiotemporal Attention)를 구현함으로써, 스파이킹 뉴런의 계산 능력을 최대한 활용합니다. 또한, 비선형 스파이킹 어텐션 디노이저(nonlinear spiking attention denoisers)를 사용하여 주의 맵의 강인성과 표현력을 향상시킵니다.

- **Performance Highlights**: DS2TA는 CIFAR10에서 94.92%, CIFAR100에서 77.47%, CIFAR10-DVS에서 79.1%, DVS-Gesture에서 94.44%의 top-1 정확도로 여러 정적 이미지와 동적 신경형 하드웨어 데이터셋에서 최첨단 성능을 입증하였습니다.



### Explainable AI for Autism Diagnosis: Identifying Critical Brain Regions Using fMRI Data (https://arxiv.org/abs/2409.15374)
- **What's New**: 이 연구는 자폐 스펙트럼 장애(ASD)의 조기 진단과 개입을 위한 새로운 접근 방식을 제시하며, 기존의 진단 모델의 해석 가능성을 높이는 데 초점을 맞추고 있습니다. 이 연구는 ASD를 정확히 분류할 뿐만 아니라, 그 작동 방식에 대한 설명 가능한 통찰력을 제공하는 딥 러닝(DL) 모델을 개발하려고 합니다.

- **Technical Details**: 사용된 데이터셋은 884개의 샘플로 구성된 자폐 뇌 영상 데이터 교환(ABIDE)의 전처리된 버전입니다. 이 연구는 resting-state functional Magnetic Resonance Imaging (fMRI) 데이터 분석을 통해 ASD의 잠재 바이오마커를 식별하고, Remove And Retrain (ROAR) 기술을 사용하여 해석 가능성 방법을 벤치마킹합니다.

- **Performance Highlights**: 모델은 ASD를 정확하게 분류할 수 있으며, ASD와 일반인 집단(Typical Controls) 간의 비정상적인 뇌 영역을 강조합니다. 이러한 발견은 다양한 데이터셋과 방법론에서의 선행 연구들에 의해 검증되었으며, 향후 ASD의 조기 진단 및 신경 기초 이해에 중요한 의미를 갖습니다.



### Enhancing Performance and Scalability of Large-Scale Recommendation Systems with Jagged Flash Attention (https://arxiv.org/abs/2409.15373)
Comments:
          3 pages, 2 figures

- **What's New**: 이 논문은 최신 추천 시스템에서 하드웨어 가속기를 활용하여 복잡한 랭킹 패러다임을 탐색하는 새로운 접근 방식을 제시합니다. 특히, GPU 기반의 계산 비용 문제를 해결하기 위한 Jagged Feature Interaction Kernels와 Jagged Flash Attention을 도입하여 성능을 개선하고 메모리 사용을 줄였습니다.

- **Technical Details**: Jagged Feature Interaction Kernels는 긴 범주형 특징에서 세밀한 인사이트를 추출하기 위한 혁신적인 방법으로, 패딩 없이 동적으로 크기가 조정되는 텐서를 효율적으로 처리합니다. Jagged Flash Attention은 기존의 dense attention에 비해 최대 9배의 속도 향상과 22배의 메모리 절약을 실현합니다. 이를 통해 메모리 사용이 선형적으로 증가하며, 53% 더 메모리 효율성을 보여 줍니다.

- **Performance Highlights**: 생산 모델에서 약 10%의 QPS(Queries Per Second) 향상과 18%의 메모리 절약을 관찰 하였으며, 이는 더 긴 특징과 복잡한 모델 아키텍처를 수용하는 추천 시스템의 확장을 가능하게 합니다.



### Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models (https://arxiv.org/abs/2409.15371)
- **What's New**: 본 논문에서는 Bone(Block Affine)이라는 새로운 PEFT(Parameter-Efficient Fine-Tuning) 방법을 제안하여 기존 LoRA 변형의 한계를 극복하고 전체 파라미터 학습을 초월하는 방법론을 소개합니다. 이 방법은 메모리 오버헤드를 줄이고, 가중치 간의 내부 연결을 강조하여 더 빠른 수렴과 더 나은 데이터 적합을 이끌어냅니다.

- **Technical Details**: Bone은 초기화가 복잡하지 않은 단일 학습 가능한 행렬을 활용하며, 이 행렬은 W𝑊와의 상호작용을 통해 Block Affine 연산을 수행합니다. 이 구조는 LoRA의 두 개의 구조 복잡성을 해소하고, 빠른 수렴 속도와 데이터를 적합시키는 능력을 보여줍니다. 실험은 서로 다른 LLM 아키텍처(LLaMA2, RWKV6)와 다양한 파라미터 스케일을 기반으로 수행되었습니다.

- **Performance Highlights**: 실험 결과, Bone 방식은 LLaMA2-7B 모델을 MetaMathQA 데이터셋으로 파인튜닝 시 49.36의 점수를 기록하며, PISSA보다 5.84% 향상된 성과를 보였습니다. 또한, Bone은 복잡한 초기화 없이 빠른 수렴과 우수한 데이터 적합성을 달성함으로써 성능을 입증하였습니다.



### Smirk: An Atomically Complete Tokenizer for Molecular Foundation Models (https://arxiv.org/abs/2409.15370)
Comments:
          26 pages, 6 figures

- **What's New**: 이번 연구에서는 분자 기반 모델(Molecular Foundation Models)의 발전을 다루며, 기존의 폐쇄 어휘(tokenizer) 방식에 의해 발생하는 한계를 극복하기 위한 두 가지 새로운 tokenizer(s)인 smirk와 smirk-gpe를 소개합니다. 이들은 OpenSMILES 규격을 전부 표현할 수 있도록 설계되었습니다.

- **Technical Details**: 연구팀은 13개의 화학 전용 tokenizer를 체계적으로 평가하며 SMILES 언어의 커버리지를 분석하였습니다. 이를 통해 다양한 화학 구조를 포괄하지 못하는 기존 모델의 정보 손실 문제를 조명했습니다. 특히, 화학 기원 모형에서 open-vocabulary 모델링의 중요성을 강조하고, 이를 위한 화학적으로 다양한 벤치마크의 필요성을 제기했습니다.

- **Performance Highlights**: 새로 소개된 smirk 및 smirk-gpe tokenizer는 SMILES 문자열의 전 범위를 처리하면서 기존의 방법들에서 겪었던 여러 문제를 회피하는데 성공하였으며, 이는 화학 분자 예측 및 디자인에서의 성능 향상에 기여할 것으로 기대됩니다.



### Geometric Relational Embeddings (https://arxiv.org/abs/2409.15369)
Comments:
          Doctoral Dissertation, 177 pages

- **What's New**: 본 논문은 관계형 데이터의 복잡한 구조적 특성과 기호적 속성을 효과적으로 포착할 수 있는 기하학적 관계 임베딩(geometric relational embeddings) 모델을 제안합니다. 기존의 벡터 임베딩(vector embeddings) 방법론의 한계를 뛰어넘어 기하학적 임베딩을 도입하여 더 정교한 관계 구조 및 이산적 의미를 표현합니다.

- **Technical Details**: 제안된 기하학적 관계 임베딩 모델은 네트워크와 지식 그래프에서의 계층(hierarchies) 및 사이클(cycles)과 같은 복잡한 구조 패턴, 지식 그래프에서의 관계 및 논리적 패턴, 온톨로지에서의 논리적 구조(logical structures), 그리고 엔티티(entity)와 관계(relation) 간의 고차(high-order) 복잡한 관계를 포착하는 데 유효합니다. 이러한 기하학적 임베딩은 하이퍼볼릭 공간(hyperbolic space)과 유클리드 벡터 공간(Euclidean vector space)에서의 기하학적 요소로 데이터 객체를 매핑합니다.

- **Performance Highlights**: 벤치마크 및 실제 데이터셋을 통한 결과는 기하학적 관계 임베딩이 관계 데이터의 이산적, 기호적 및 구조적 특성을 효과적으로 포착함을 보여줍니다. 이로 인해 다양한 관계 추론(task)에서 성능 향상이 발생했습니다.



### MedCodER: A Generative AI Assistant for Medical Coding (https://arxiv.org/abs/2409.15368)
- **What's New**: 이번 연구에서는 MedCodER라는 새로운 Generative AI 프레임워크를 소개하며, 의료 코딩의 자동화를 위한 혁신적인 접근 방식을 제시합니다. 특히, 이 프레임워크는 추출(Extraction), 검색(Retrieval), 재정렬(Re-ranking) 기술을 핵심 요소로 활용하여 높은 정확도를 자랑합니다.

- **Technical Details**: MedCodER는 의료 기록에서 질병 진단과 지원 증거를 추출하고, ICD-10 코드를 후보 코드로 검색한 다음, 이를 종합하여 최종 코드를 예측합니다. 이 과정에서 LLM(대형 언어 모델)의 파라메트릭 지식을 보완하기 위해 검색 및 재정렬 기술을 통합하여 성능을 향상시킵니다.

- **Performance Highlights**: MedCodER는 ICD 코드 예측에서 0.60의 micro-F1 점수를 달성하여 현재 최고의 방법보다 유의미하게 향상된 성과를 보입니다. 또한, 제안된 데이터셋은 질병 진단과 ICD 코드, 그리고 이를 정당화하는 지원 증거 텍스트가 주석 처리되어 있어, 코드 선택의 신뢰성을 높이는 데 기여합니다.



### Fine-Tuning a Time Series Foundation Model with Wasserstein Loss (https://arxiv.org/abs/2409.15367)
Comments:
          4 main pages; 2 figures

- **What's New**: 이번 연구는 시간 시계열 예측을 위한 기초 모델 개발에 있어 최근의 대형 언어 모델(LLM) 발전에 힘입어, cross-entropy loss 대신 Wasserstein loss를 사용하는 방법을 제안하고 있습니다.

- **Technical Details**: 연구진은 LLM 아키텍처를 토큰화된 시간 시계열 데이터로 교육하여 cross-entropy loss로 주어진 모델을 정밀 조정하였습니다. Wasserstein loss는 클래스 간 거리 정보를 반영하며, 성능 비교를 통해 예측 정확도를 개선하는 것이 입증되었습니다.

- **Performance Highlights**: 22개의 zero-shot 데이터셋에서 평가한 결과, cross-entropy loss에 비해 Wasserstein loss를 사용하는 것이 점 추정(point estimation) 성능을 유의미하게 향상시켰습니다.



### Trajectory Anomaly Detection with Language Models (https://arxiv.org/abs/2409.15366)
- **What's New**: 본 논문은 LM-TAD라는 자가 회귀 인과 주의 학습 모델을 사용하여 궤적(anomaly) 이상 탐지에 대한 새로운 접근을 제안합니다. 이 방법은 궤적과 언어 진술 사이의 유사성을 활용하여 구조적 연관성을 기반으로 궤적의 확률 분포를 학습합니다.

- **Technical Details**: 이 모델은 궤적을 일종의 토큰(토큰화된 GPS 좌표 등) 시퀀스로 취급하여 각 위치를 생성할 확률을 계산하고, 이를 바탕으로 비정상적인 위치를 높은 정밀도로 식별합니다. 사용자의 행동 패턴을 반영하기 위해 사용자 특정 토큰을 통합하였고, perplexity와 surprisal rate 메트릭을 도입하여 궤적 내 특정한 비정상 위치를 탐지합니다.

- **Performance Highlights**: LM-TAD는 Pattern of Life (PoL) 데이터 세트에서 사용자 맥락을 고려한 비정상 궤적을 성공적으로 탐지하여 기존 방법들보다 우수한 성능을 보였고, Porto 택시 데이터 세트에서도 경쟁력 있는 결과를 나타냈습니다. 또한, 이 방법은 온라인 이상 탐지에 적합하여 계산 지연을 대폭 줄일 수 있습니다.



### Novel Saliency Analysis for the Forward Forward Algorithm (https://arxiv.org/abs/2409.15365)
Comments:
          2nd International Conference on Artificial Intelligence, Blockchain, and Internet of Things, (AIBThings)

- **What's New**: 이 논문에서는 신경망 학습에 Forward Forward (FF) 알고리즘을 도입하여 전통적인 백프로파게이션(Backpropagation, BP) 방법에서 벗어난 새로운 접근 방식을 제안합니다. FF 알고리즘은 실제 데이터와 합성 데이터로 두 번의 전방 패스를 실행하여 학습 과정을 단순하고 효율적으로 만듭니다.

- **Technical Details**: FF 알고리즘은 신경망에서 식별되지 않은 비선형성 문제를 해결하는 새로운 방식으로, 긍정적 데이터와 부정적 데이터에 대한 각각의 목표를 달성하는 두 개의 전방 패스를 사용합니다. 이 알고리즘은 기존의 BP 방식 없이도 신경망이 데이터를 직접 학습할 수 있게 합니다.

- **Performance Highlights**: MNIST 및 Fashion MNIST 데이터셋을 활용한 실험 결과, FF 알고리즘은 전통적인 다층 퍼셉트론(Multi-Layer Perceptron, MLP) 아키텍처와 동등한 성능을 보이며, 해석 가능성을 크게 향상시킵니다.



### VERA: Validation and Enhancement for Retrieval Augmented systems (https://arxiv.org/abs/2409.15364)
- **What's New**: VERA는 Retrieval-Augmented Generation (RAG) 시스템을 위한 평가 및 개선 시스템으로, LLM의 응답精度를 향상시키기 위한 새로운 방법을 제공합니다. 또한, VERA는 외부 정보를 효과적으로 활용하도록 설계되었습니다.

- **Technical Details**: VERA는 수집된 컨텍스트의 적합성과 불필요한 정보를 제거하는 데 중점을 둡니다. 이 시스템은 평가자 및 향상 LLM을 사용하여 응답 생성 전에 컨텍스트를 평가하고, 응답 생성 후에는 응답을 분리하여 각 문장의 적합성을 점검합니다.

- **Performance Highlights**: 실험 결과, VERA는 소규모 공개 오픈 소스 모델에서 뿐만 아니라 대규모 최첨단 모델에서도 성능을 개선하는 데 뛰어난 효능을 나타냈습니다. VERA는 정보 생성에서 높은 정확성 및 신뢰성을 요구하는 응용 프로그램에 유용한 도구로 자리 잡을 잠재력을 보여주고 있습니다.



### Multitask Mayhem: Unveiling and Mitigating Safety Gaps in LLMs Fine-tuning (https://arxiv.org/abs/2409.15361)
Comments:
          19 pages, 11 figures

- **What's New**: 최근의 Large Language Models (LLMs)에 대한 연구는 다양한 다운스트림 작업에서 가벼운 튜닝이 안전성을 저해할 수 있음을 보여줍니다. 특히, 코드 생성 및 번역 작업에서 안전성 감소가 두드러지며, 새로운 멀티태스크 안전 데이터셋인 MultiTaskBench를 개발하여 이러한 문제를 해결하고자 했습니다.

- **Technical Details**: 이 연구는 네 가지 작업(요약, 코드 생성, 번역, 분류)에 대한 데이터셋을 사용하여 LLM의 튜닝 및 안전성 감소 현상을 분석합니다. 연구에서는 거짓 응답을 방지하기 위해 Reinforcement Learning from Human Feedback (RLHF) 방식을 활용하며, 선행 연구와의 차별성을 위해 안전하게 생성된 데이터셋을 적용합니다.

- **Performance Highlights**: 연구 결과, LLM은 번역 및 분류 작업에서 상대적으로 안전성 유지가 어려웠으며, 코드 데이터로 튜닝할 경우 가장 높은 안전성 저하를 보였습니다. 제안된 MultiTaskBench 데이터셋은 다양한 다운스트림 작업에서 공격 성공률을 효과적으로 감소시켰습니다.



### Reward-Robust RLHF in LLMs (https://arxiv.org/abs/2409.15360)
- **What's New**: 본 논문에서는 보상 모델의 불안정성과 오류를 해결하기 위한 보상 강건 RLHF(Reward-Robust Reinforcement Learning from Human Feedback) 프레임워크를 제안합니다.

- **Technical Details**: 이 프레임워크는 Bayesian Reward Model Ensembles (BRME)를 통해 보상 함수의 불확실성 집합을 모델링하며, 성능과 강건성을 균형 있게 최적화하는 새로운 목표를 설정합니다.

- **Performance Highlights**: 제안하는 프레임워크는 16개의 다양한 벤치마크에서 전통적인 RLHF를 지속적으로 초월하여 평균 정확도가 약 4% 더 높게 나타났으며, 장기 훈련 과정에서도 더 강한 안정성과 향상된 성능을 보여주었습니다.



### Watch Your Steps: Observable and Modular Chains of Though (https://arxiv.org/abs/2409.15359)
- **What's New**: 이 논문에서는 Program Trace Prompting (PTP)이라는 새로운 형태의 chain of thought (CoT) 프롬프트를 제안합니다. 이 방법은 CoT의 장점과 유연성을 유지하면서 설명 과정을 보다 관찰 가능하게 만들어 줍니다.

- **Technical Details**: PTP는 Python 기반의 형식을 사용하여 CoT 데모를 포장하고, 각 프롬프트에서는 단계 식별, 입력/출력 동작 정의, CoT 설명을 공식화된 단계로 대체하는 워크플로우를 제공합니다. 이 방법은 다양한 작업에 적용 가능하며, BIG-Bench Hard 벤치마크에서 23개의 작업에 대해 강력한 성능을 보여줍니다.

- **Performance Highlights**: PTP는 대부분의 작업에서 CoT 프롬프트와 비슷한 정확도를 달성하며, 생성된 트레이스는 99% 이상의 법적 단계로 파싱할 수 있습니다. 또한 PTP는 개별 단계 실행과 작업 전체 해결을 모두 가능하게 하며, 대부분의 단계에서 모듈성과 지역성을 평가할 수 있습니다. 실험 결과는 PTP의 유용성을 검증하고, 많은 비국소 오류가 잘못된 알고리즘 추정에서 유래함을 보여줍니다.



### Block-Attention for Low-Latency RAG (https://arxiv.org/abs/2409.15355)
- **What's New**: 새로운 Block-Attention 메커니즘이 도입되었습니다. 이 방법은 Retrieval-Augmented Generation (RAG) 시나리오에서의 추론 지연을 줄이기 위해 고안되었습니다. 각 입력 시퀀스를 블록으로 나누어 최종 블록을 제외한 각 블록이 독립적으로 키-값(KV) 상태를 계산하게 합니다.

- **Technical Details**: Block-Attention은 입력 시퀀스를 여러 개의 블록으로 나누고, 각 블록은 다른 블록과 상관없이 KV 상태를 계산합니다. RAG 시나리오에서 각 패세지를 블록으로 정의함으로써 모든 패세지에 대한 KV 상태를 미리 계산하고 메모리에 캐시할 수 있습니다. 블록 세그먼트, 위치 인코딩 계산 및 LLM을 Block-Attention 메커니즘에 적응시키기 위한 미세 조정이 포함됩니다.

- **Performance Highlights**: Block-Attention 모델은 fine-tuning을 통해 Llama3에서 68.4%의 성능을 달성하며, 이는 기존의 self-attention 모델(67.9%)과 비슷합니다. 특히, Block-Attention은 TTFT를 평균 98.7% 줄여, 32K 길이의 입력 시퀀스에 대해 첫 번째 토큰을 출력하는 데 단 45ms가 소요됩니다.



### Recall: Empowering Multimodal Embedding for Edge Devices (https://arxiv.org/abs/2409.15342)
- **What's New**: RECALL은 리소스가 제한된 모바일 환경을 위해 최적화된 최초의 on-device 멀티모달 임베딩 시스템이다. 이 시스템은 coarse-grained embedding을 생성하고 query 기반 필터링을 활용하여 높은 처리량과 정확한 검색을 달성한다.

- **Technical Details**: RECALL 시스템은 데이터 인식 pre-exit 예측기, Progressive LoRA healing, 그리고 Speculative fine-grained retrieval의 세 가지 하드웨어-알고리즘 공동 설계를 통해 동작한다. 이 시스템은 multi-layer transformer architecture를 통해 작동하며, coarse-grained embedding을 통해 모달리티 간 검색을 수행하고, 후속 쿼리 단계에서 최종 검색을 정제한다.

- **Performance Highlights**: RECALL은 평균 14.9배 처리량 향상과 13.1배 에너지 소비 감소를 달성하였다. 이 시스템은 배터리 소모를 최소화하면서도 높은 정확도를 유지하며, 전체 MEM에 비해 5% 미만의 상대적 정확도 손실을 초래한다.



### Explainable AI: Definition and attributes of a good explanation for health AI (https://arxiv.org/abs/2409.15338)
Comments:
          21 pages

- **What's New**: 본 논문은 건강 관련 인공지능(health-AI)에서 좋은 설명(explanation)이 무엇인지와 그 속성을 정의하기 위한 연구 결과를 제시합니다.

- **Technical Details**: 연구는 두 가지 주요 질문에 초점을 맞췄습니다: (1) 건강-AI에서 설명이란 무엇인가? (2) 건강-AI에서 좋은 설명의 속성은 무엇인가? 이를 위해, 기존 문헌을 검토하고 전문가의 의견을 수집하기 위해 두 차례의 델파이 연구(Delphi study)를 진행했습니다.

- **Performance Highlights**: 연구 결과는 건강-AI에서 설명의 정의와 좋은 설명을 특징짓는 속성의 포괄적인 목록을 포함하고 있으며, 이는 안전 중심의 AI 애플리케이션에서 중요한 기초 자료로 사용될 수 있습니다.



### Revisiting the Solution of Meta KDD Cup 2024: CRAG (https://arxiv.org/abs/2409.15337)
- **What's New**: 이 논문은 Meta KDD CUP 2024의 CRAG Comprehensive RAG Benchmark Challenge에서 팀 APEX의 솔루션을 소개합니다. CRAG 벤치마크는 Retrieval-Augmented Generation (RAG) 시스템의 다양하고 동적인 문제를 평가하는 데 있어 기존 QA 벤치마크의 한계를 극복할 수 있도록 설계되었습니다.

- **Technical Details**: 본 연구에서는 질문의 다양성과 동적인 특성에 맞춘routing 기반의 도메인 및 동적 적응 RAG 파이프라인을 제안합니다. 이 방법은 정보 검색(retrieval), 증대(augmentation), 생성(generation) 세 단계에서 모두 특별한 처리를 수행하며, CRAG에서 우수한 성과를 거두어 최종 경쟁 리더보드에서 작업 2와 3에서 2위를 기록했습니다.

- **Performance Highlights**: 우리의 방법은 CRAG에서 뛰어난 성과를 발휘했으며, 특히 웹 페이지 검색 및 Mock API를 활용해 정보 선택과 통합의 능력을 강조하였습니다. 각 과제는 이전 단계를 기반으로 하여, 참가자들이 더욱 정교한 end-to-end RAG 시스템을 개발하도록 유도합니다.



### Causality-Driven Reinforcement Learning for Joint Communication and Sensing (https://arxiv.org/abs/2409.15329)
Comments:
          18 pages, 9 figures, 4 tables, 1 algorithm

- **What's New**: 본 논문에서는 mMIMO 기반 Joint Communication and Sensing (JCAS) 환경에서 인과 관계를 고려하는 강화 학습 (Reinforcement Learning, RL) 접근 방식을 소개합니다. 이는 비효율적인 beamforming을 개선하고 학습 효율성을 높이기 위해 설계되었습니다.

- **Technical Details**: State-Wise Action-Refined Temporal Difference Learning (TD3-INVASE) 알고리즘을 사용하여 mMIMO 시스템의 action space에서 유용한 행동을 탐색하고 보상과의 인과 관계를 발견합니다. 이 프레임워크는 대규모의 action 및 상태 공간에서도 샘플 효율성을 제공합니다.

- **Performance Highlights**: 다양한 시나리오에서 평가한 결과, 제안된 프레임워크는 기존의 RL 기반 beamforming 방법과 비교하여 높은 beamforming gain을 보였으며, 이는 JCAS 전용의 beamforming에 있어 샘플 효율적인 훈련을 가능하게 하였습니다.



### Evaluating the Impact of a Specialized LLM on Physician Experience in Clinical Decision Support: A Comparison of Ask Avo and ChatGPT-4 (https://arxiv.org/abs/2409.15326)
Comments:
          8 pages, 1 figure

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 이용한 임상 결정 지원 시스템의 발전을 논의합니다. 특히, AvoMD의 Ask Avo 소프트웨어와 그것의 LMAR(언어 모델 증강 검색) 시스템이 ChatGPT-4에 비해 임상 환경에서 어떤 성능을 보이는지 평가했습니다.

- **Technical Details**: Ask Avo는 의료 가이드라인 문서에서 파생된 임상 질문을 62명의 참가자로부터 수집하여, 각 모델(Ask Avo와 ChatGPT-4)이 생성한 답변을 신뢰성(trustworthiness), 실행 가능성(actionability), 관련성(relevancy), 포괄성(comprehensiveness), 친근한 형식(friendly format)으로 평가했습니다. 연구에서는 1부터 5까지의 점수를 주었고, Ask Avo는 모든 기준에서 ChatGPT-4를 초과하는 성능을 나타냈습니다.

- **Performance Highlights**: Ask Avo는 신뢰성에서 4.52, 실행 가능성에서 4.41, 관련성에서 4.55, 포괄성에서 4.50, 친근한 형식에서 4.52의 점수를 기록했습니다. 이와 비교하여 ChatGPT-4의 점수는 각각 3.34, 3.19, 3.49, 3.37, 3.60으로 낮았으며, 모든 비교에서 유의미한 차이가 있었습니다(p<0.001). 이러한 결과는 임상 요구에 맞춰 설계된 특화된 LLM이 전반적인 사용자 경험에서 현저한 개선을 가능하게 함을 시사합니다.



### Introducing ELLIPS: An Ethics-Centered Approach to Research on LLM-Based Inference of Psychiatric Conditions (https://arxiv.org/abs/2409.15323)
- **What's New**: 본 논문은 언어 기반의 정신병리 인식을 위한 모델 개발 시 고려해야 할 윤리적 원칙들을 제시하고, 연구자들이 이를 활용할 수 있도록 ELLIPS라는 윤리적 도구 킷을 개발하여 실제 적용 가능성을 높이고자 합니다.

- **Technical Details**: 연구에서는 정신병리의 모델 개발과 배포에 영향을 미치는 네 가지 주요 질문을 다룹니다: (1) 타겟 변수: 어떤 종류의 특성을 예측할 것인지, (2) 훈련 데이터: 어떤 종류의 언어적 행동 데이터를 사용할 것인지, (3) 모델 아키텍처: 어떤 모델 구조와 크기를 사용할 것인지, (4) 평가 프로토콜: 시스템의 평가에 사용할 데이터는 무엇인지.

- **Performance Highlights**: 정신질환의 진단을 위한 언어적 마커에 대한 기존 연구에서 성능이 큰 차이를 보였으며, 임상 결정 과정에서의 적용 가능성을 높이기 위해서는 더욱 실질적인 예측 모델링이 필요합니다.



### On the Complexity of Neural Computation in Superposition (https://arxiv.org/abs/2409.15318)
Comments:
          43 pages, 8 figures

- **What's New**: 이번 논문은 superposition(중첩)에서의 계산에 대한 이론적 기초를 탐구하며, 명시적이고 증명 가능한 알고리즘과 그 효율성을 강조합니다. 또한 이 연구는 단일 뉴런이 여러 특성을 동시에 표현할 수 있는 능력이 대규모 네트워크의 계산 효율성의 핵심 메커니즘임을 제시합니다.

- **Technical Details**: 우리는 $orall$의 일부 문제들, 특히 permutation(순열) 및 pairwise logical operations(쌍별 논리 연산)에 대해 최소한 $	ext{Ω}(m' 	ext{log} m')$의 매개변수와 $	ext{Ω}(	ext{√}(m' 	ext{log} m'))$ 뉴런이 필요하다는 첫 번째 하한을 제시합니다. 반대로, pairwise AND와 같은 논리 연산을 $	ext{O}(	ext{√}(m') 	ext{log} m')$ 뉴런과 $	ext{O}(m' 	ext{log}^2 m')$ 매개변수로 수행할 수 있음을 보여줍니다.

- **Performance Highlights**: 본 연구의 결과에 따르면, 중첩에서 계산을 수행할 때 특성의 표현을 대략 출력 특성의 제곱근 정도로 압축할 수 있으며, 이는 추후 뉴럴 네트워크의 설계 및 복잡성에 대한 흥미로운 질문들을 제기합니다.



### Shared Autonomy with IDA: Interventional Diffusion Assistanc (https://arxiv.org/abs/2409.15317)
Comments:
          10 pages, 4 main figures, 2 appendix figures

- **What's New**: 이번 연구에서는 Shared Autonomy (SA) 시스템 안에서 인간의 자율성을 유지하면서도 필요한 경우에만 AI 공동 조종사(copilot)가 개입할 수 있는 목표 비의존적 개입 지원(intervention assistance, IA) 방식을 개발했습니다. 이는 인간 조종사의 행동과 AI의 행동을 비교하여 인간의 행동이 비효율적일 때에만 AI가 개입하도록 설계된 것입니다. 이를 통해 인간의 자율성과 성능을 동시에 향상시킬 수 있음을 보여주었습니다.

- **Technical Details**: 연구진은 Diffusion Copilot을 기반으로 한 IA 시스템(ID)을 구현하였으며, 전문가의 시연에 기초하여 훈련을 진행하였습니다. 이 과정에서 IA의 성능을 줄어드는 하한값을 증명하였고, 시뮬레이션된 인간 조종사들과의 실험을 통해, 기존의 파일럿 전용 시스템 및 전통적인 SA 제어보다 더 높은 성능을 달성하였음을 입증하였습니다.

- **Performance Highlights**: Lunar Lander 및 Reacher 환경에서 AI 협조하의 제어 시스템이 파일럿 전용 및 기존 SA 제어 방식보다 뛰어난 성능을 나타냈으며, 실험 참가자들은 IA를 통해 더 큰 자율성을 경험하고, IA를 기존 방법보다 선호한다고 보고하였습니다.



### An Efficient Recommendation Model Based on Knowledge Graph Attention-Assisted Network (KGATAX) (https://arxiv.org/abs/2409.15315)
- **What's New**: 이번 연구에서는 다중 소스 정보를 효과적으로 활용하지 못하는 전통적인 추천 시스템의 한계를 극복하기 위해 'Knowledge Graph Attention-assisted Network (KGAT-AX)'라는 새로운 추천 모델을 제안합니다.

- **Technical Details**: KGAT-AX 모델은 추천 시스템에 지식 그래프(knowledge graph)를 통합하고 주의 메커니즘(attention mechanism)을 도입하여 더 높은 차원의 연결성을 탐구합니다. 다층 상호작용 정보 전파(multilayer interactive information propagation)를 통해 모델은 정보를 집계하여 일반화 능력을 향상시킵니다. 또한, 홀로그램 임베딩(holographic embeddings)을 통해 보조 정보(auxiliary information)를 엔티티에 통합하여, 인접 엔티티의 정보를 학습하여 더 나은 활용이 가능합니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험을 통해 KGAT-AX 모델의 합리성과 효과성을 입증하였으며, 공공 데이터셋에서 다른 기준선 모델(baseline models)과 비교하여 KGAT-AX가 지식 정보 캡처 및 관계 학습 능력에서 우수함을 확인했습니다.



### Irrelevant Alternatives Bias Large Language Model Hiring Decisions (https://arxiv.org/abs/2409.15299)
- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 인사 결정에서 인간의 인지 편향인 매력 효과(attraction effect)를 나타내는지를 조사합니다. 매력 효과는 열등한 후보의 존재가 우수한 후보를 더 매력적으로 만들어 선택될 가능성을 증가시키는 현상입니다. 연구 결과, GPT-3.5와 GPT-4 모두에서 매력 효과가 존재함을 발견했습니다.

- **Technical Details**: 본 연구에서는 LLM이 리크루터 역할을 맡아 후보자를 선택하는 실험을 진행했습니다. 또한, 탈락 후보의 성별과 같은 무관한 속성이 관찰된 편향을 더욱 증대시키는 것으로 나타났습니다. 실험은 Huber et al. (1982)와 같은 고전적인 디자인을 따르며, 두 가지 조건인 통제 조건과 처치 조건을 통해 진행되었습니다.

- **Performance Highlights**: GPT-4는 GPT-3.5에 비해 더 큰 편향 변동을 보였으며, 경고를 포함하더라도 매력 효과가 강하게 유지되었습니다. 이 연구 결과는 LLM이 인사 결정에서도 불필요한 대안에 의해 편향될 수 있음을 시사합니다.



### SketcherX: AI-Driven Interactive Robotic drawing with Diffusion model and Vectorization Techniques (https://arxiv.org/abs/2409.15292)
Comments:
          10 pages, 10 figures

- **What's New**: 이번 논문은 SketcherX라는 새로운 로봇 시스템을 소개합니다. 이 시스템은 사용자와 상호작용하여 개인화된 초상화를 그리는 기능을 가지고 있으며, 기존의 아날로그 인쇄 기법에 의존하지 않고 얼굴 이미지를 캡처하여 독특한 인간과 유사한 예술 스타일로 벡터화된 그림을 생성합니다.

- **Technical Details**: SketcherX는 두 개의 6축 로봇 팔로 구성되어 있으며, 하나는 의사소통 기능을 갖춘 얼굴 로봇이고 다른 하나는 스타일리시한 그림을 그리는 로봇입니다. 이 로봇들은 각각 Large Language Model (LLM), Stable Diffusion, ControlNet 및 Vision-Language 모델을 사용하여 사용자와의 상호작용과 다이나믹한 그림 그리기를 수행합니다. 특히, Vector Low Rank Adaptation (LoRA) 모델을 통해 다양한 예술 스타일에 원활하게 적응할 수 있는 커스터마이징된 모델을 개발했습니다.

- **Performance Highlights**: SketcherX는 2분 내에 고품질의 개인화된 초상화를 생성할 수 있는 능력을 보여주었으며, 로봇의 창의적 작업 과정이 사용자에게 실시간으로 전달되는 새로운 패러다임을 제시합니다. 연구 결과는 이 시스템이 다양한 예술 스타일을 학습하고 표현할 수 있어, 로봇이 상호작용하는 새로운 매체로 자리 잡을 가능성을 보여줍니다.



### Broadening Access to Simulations for End-Users via Large Language Models: Challenges and Opportunities (https://arxiv.org/abs/2409.15290)
Comments:
          To appear in proceedings of the 2024 Winter Simulation Conference

- **What's New**: 이 논문은 Large Language Models (LLMs)를 사용하여 비전문가들이 시뮬레이션과 상호작용할 수 있는 새로운 방안을 제시하고 있습니다. 특히, LLMs가 사용자가 일상 언어로 'what-if' 질문을 할 수 있도록 도와줌으로써 시뮬레이션에 대한 접근성을 높일 수 있는 가능성을 탐구합니다.

- **Technical Details**: 논문은 시스템 설계를 세 가지 주요 단계로 나누어 설명합니다. 첫째, 사용자 질문을 가장 적절한 시뮬레이션 모델에 매핑합니다. 둘째, 매핑이 불가능할 경우 질문을 자동으로 재구성하고 추가적으로 명확화 질문을 생성합니다. 마지막으로, 시뮬레이션 결과를 생성하고 의사 결정을 위해 맥락을 제공합니다. 자연어 처리(NLP) 기술을 통해 사용자의 질문에서 주요 개념을 인식하고 이를 시뮬레이션 모델의 파라미터와 연결하는 과정을 포함합니다.

- **Performance Highlights**: 논문에서는 LLMs가 시뮬레이션 소프트웨어와의 통합을 통해 사용자 경험을 향상시키고, 비전문가들도 쉽게 시뮬레이션을 사용할 수 있도록 지원하는 방법을 다루고 있습니다. 이러한 접근 방식은 커뮤니티의 기존 연구에 새로운 기회를 제공하며, 공공 정책, 건강 관리 등 다양한 분야에서 혁신적인 의사 결정 지원 시스템 개발을 촉진할 것으로 기대됩니다.



### Generative AI Is Not Ready for Clinical Use in Patient Education for Lower Back Pain Patients, Even With Retrieval-Augmented Generation (https://arxiv.org/abs/2409.15260)
- **What's New**: 본 연구에서는 Retrieval-Augmented Generation (RAG)과 few-shot learning을 활용하여 허리통증(LBP) 환자를 위한 맞춤형 교육 자료를 생성하는 새로운 접근법을 소개합니다.

- **Technical Details**: 이 연구에서는 대형 언어 모델(LLMs)이 RAG를 사용하여 생성된 교육 자료의 중복성, 정확성 및 완전성을 평가하기 위해 물리치료사가 수동으로 검토하였습니다. 또한 생성된 교육 자료의 가독성은 Flesch Reading Ease 점수를 통해 평가되었습니다.

- **Performance Highlights**: RAG 기반 LLMs는 전통적인 LLMs보다 더 정확하고 완전하며 적은 중복성을 가진 환자 교육 자료를 제공합니다. 그러나 생성된 자료는 아직 임상 실무에 사용하기에는 준비가 되어 있지 않으며, AI 모델의 임상적 관련성과 내용의 세분화를 보장하는 데에는 여전히 상당한 과제가 남아 있습니다.



### MACeIP: A Multimodal Ambient Context-enriched Intelligence Platform in Smart Cities (https://arxiv.org/abs/2409.15243)
Comments:
          4 pages, 6 figures, IEEE/IEIE ICCE-Asia 2024

- **What's New**: 본 논문에서는 Smart Cities을 위한 Multimodal Ambient Context-enriched Intelligence Platform (MACeIP)을 제안하며, 도시 관리 및 시민 참여를 향상시키기 위해 다양한 IoT 센서, Edge computing 및 Cloud computing을 통합한 포괄적인 시스템입니다.

- **Technical Details**: MACeIP은 시민 상호작용을 위한 Interactive Hubs, IoT 센서 네트워크, 지능형 공공 자산 관리 시스템, 보행자 모니터링 시스템, City Planning Portal (CPP) 및 Multimodal AI를 활용한 Cloud Computing System (CCS)의 주요 구성 요소로 이루어져 있습니다. 이 시스템은 시간적 시리즈 및 비전 모델, 대규모 언어 모델 (Large Language Models, LLMs), 설명 가능한 AI (Explainable AI, XAI) 기술을 포함하여 사용자 중심의 지능적이고 효율적인 의사결정을 지원합니다.

- **Performance Highlights**: 여러 도시에서 MACeIP 프로토타입을 시연하여 시민과의 상호작용을 증대시키고 있으며, 특히 New Brunswick의 Fredericton에서 추진된 프로젝트가 주목받고 있습니다. 이 플랫폼은 실시간 데이터 수집 및 처리, 공공 안전 강화를 위한 보행자 모니터링 시스템을 통한 도시 관리 효율성을 높이고 있으며, 사용자 친화적인 대시보드를 통해 도시 관리 및 자산 관리를 구현하고 있습니다.



### Chattronics: using GPTs to assist in the design of data acquisition systems (https://arxiv.org/abs/2409.15183)
Comments:
          8 pages

- **What's New**: 대규모 언어 모델(LLM)을 사용하여 데이터 수집 시스템의 설계 단계를 지원하는 새로운 접근 방식을 제시하고 있습니다. 이 도구는 사용자와의 대화적 인터페이스를 통해 프로젝트 요구 사항을 이해하고 시스템 아키텍처 다이어그램 및 세부 사양을 생성하는 기능을 갖추고 있습니다.

- **Technical Details**: 이 애플리케이션은 Go 언어로 개발되었으며, OpenAI의 GPT-4-Turbo 모델을 활용합니다. 사용자는 응답할 내용을 입력하고, 모델은 최대 5개의 질문을 통해 프로젝트에 대한 추가 정보를 요청합니다. 최종적으로 모델은 .DOT 문자열 형태의 아키텍처를 생성하여 시각적으로 제시합니다.

- **Performance Highlights**: 160회의 테스트 결과, LLM이 데이터 수집 시스템의 설계 및 합성 도구로 활용 가능성을 보여주지만, 여러 요구 사항을 동시에 고려하는 데 어려움을 겪어 이론적인 오류를 범하는 경우도 발생했습니다. 성능은 일관된 아키텍처와 토폴로지를 보여주었습니다.



### Goal-based Neural Physics Vehicle Trajectory Prediction Mod (https://arxiv.org/abs/2409.15182)
- **What's New**: 이 논문은 차량의 목표를 고려하여 차량 궤적 예측을 두 단계의 프로세스로 단순화하는 GNP(Goal-based Neural Physics) 모델을 제안합니다. GNP 모델은 목표 예측과 궤적 예측의 두 가지 하위 모듈로 구성되어 있어 장기 예측 정확성을 극대화합니다.

- **Technical Details**: GNP 모델은 multi-head attention 메커니즘을 활용하여 차량의 목표를 정확히 예측하고, 물리 기반 사회력 모델을 통합한 심층 학습 모델을 통해 예측된 목표를 사용하여 전체 궤적을 점진적으로 예측합니다. 이를 통해 다중 모드와 본질적인 특성을 적절히 시각화할 수 있습니다.

- **Performance Highlights**: GNP 모델은 네 개의 기준 모델과 비교하여 최첨단의 장기 예측 정확성을 입증하였으며, 주요 디자인의 유효성을 검증하기 위한 ablation 연구를 진행하여 모델의 효과를 강조했습니다.



### Automatic Feature Learning for Essence: a Case Study on Car Sequencing (https://arxiv.org/abs/2409.15158)
- **What's New**: 이 논문은 고급 표현을 사용하여 조합 문제의 인스턴스 특징을 자동으로 학습하는 방법을 제안합니다. Essence 모델링 언어를 활용하여 머신러닝 모델이 최적의 조합을 선택할 수 있도록 합니다.

- **Technical Details**: 핵심 아이디어는 고급 모델링 언어를 통해 문제 인스턴스에서 필드를 직접 추출하는 것입니다. 기존의 low-level 표현인 FlatZinc와 달리, 고급 표현으로부터 자동으로 정보를 추출합니다. 이 접근법은 모델과 솔버의 조합을 선택할 때 더 많은 정보와 빠른 계산을 제공합니다.

- **Performance Highlights**: 제안된 모델은 car sequencing 문제 케이스 스터디를 통해 평가되었습니다. 자동으로 학습한 인스턴스 특징들은 기존의 fzn2feat 접근법보다 더 효과적이고, 컴퓨팅 비용이 적게 들며, 다양한 인스턴스에 대해 일관된 성능 향상을 보여주었습니다.



### Boosting Healthcare LLMs Through Retrieved Contex (https://arxiv.org/abs/2409.15127)
Comments:
          9 pages, 5 figures, 12 tables

- **What's New**: 이 연구는 의료 분야에서의 맥락 검색(context retrieval) 방법의 경계를 탐구하며, 이들의 성능을 최적화하고 공공 및 민간 모델 간의 성능을 비교합니다. 특히, 최적화된 검색 시스템을 통해 오픈 LLM(Open LLM)이 민간 솔루션과 유사한 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구에서는 Retrieval Augmented Generation (RAG) 시스템과 함께 데이터베이스(DB), 임베딩 모형(embedding model), 리랭킹 모형(reranking model) 등 다양한 컴포넌트를 최적화하여 오픈 LLM의 성능 향상 방법을 분석합니다. OpenMedPrompt라는 파이프라인을 제안하여 개방형 질문에 대한 신뢰할 수 있는 답변 생성을 개선하고 있습니다.

- **Performance Highlights**: 최적화된 시스템을 통해 MedQA, MedMCQA, CareQA와 MMLU와 같은 의료 관련 MCQA 데이터셋에서의 성능을 평가하였으며, 오픈 LLM은 설정된 벤치마크에서 강력한 성능을 달성했습니다. 이에 따라, 실제 응용 가능성을 한층 높였습니다.



### Log-normal Mutations and their Use in Detecting Surreptitious Fake Images (https://arxiv.org/abs/2409.15119)
Comments:
          log-normal mutations and their use in detecting surreptitious fake images

- **What's New**: 이 논문에서는 전통적인 적대적 공격(adversarial attack)에 대한 새로운 접근 방식을 제안합니다. 특히, log-normal 알고리즘을 활용하여 기존의 탐지 시스템을 우회하는 효과적인 공격 방법을 개발하고 있습니다.

- **Technical Details**: log-normal mutation 기법을 사용하여 공격을 수행하며, 이는 기존의 black-box optimization의 기술에서 영감을 얻은 것입니다. 논문은 다양한 알고리즘과의 비교를 포함하여 실험을 통해 log-normal mutation의 특성을 보이고 있습니다. 또한, 이 기술을 가짜 감지기(fake detector 공격)에 적용하여 가짜 이미지를 탐지되지 않도록 생성하는 방법을 제안합니다.

- **Performance Highlights**: log-normal mutation은 특히 어려운 multi-modal 및 highly-deceptive 문제에서 뛰어난 성능을 보이며, 이는 기존의 gfake 감지기에서 탐지되지 않는 가짜 이미지를 생성하는데 성공했습니다. 또한, 이러한 공격을 방어하기 위해 새로운 탐지기를 결합하여 그 강건성도 연구하고 있습니다.



### Evaluating ML Robustness in GNSS Interference Classification, Characterization \& Localization (https://arxiv.org/abs/2409.15114)
- **What's New**: 본 연구는 다양한 생성된 간섭을 포함한 저주파 안테나에서 수집된 스냅샷 데이터셋을 통하여, GNSS(Global Navigation Satellite System) 신호에 대한 방해를 감지하고, 마모 모델(ML models)의 강인성을 평가하는 시스템을 제안합니다. 이를 통해 환경 변화에 대응하는 ML 모델의 적응성을 분석합니다.

- **Technical Details**: 우리의 연구에서는 총 6 종류의 간섭 신호(Chirp, FreqHopper, Modulated, Multitone, Pulsed, Noise)와 그에 대한 다양한 특성(BW, StN ratio)을 수집합니다. 제시된 데이터는 산업적 환경에서 수집되었으며, 다차원 환경 측정을 위한 구조화된 데이터 수집 시스템을 사용하였습니다. 신호 생성 위해 MXG 벡터 신호 발생기를 사용하였으며, 각 스냅샷은 100MHz에서 10μs의 지속시간으로 수집되었습니다. 이를 통해 방해 신호의 분류 및 장치 소스의 국지화 액세스 가능성을 높였습니다.

- **Performance Highlights**: 모델은 주목할 만한 일반화 능력을 보이며, 다양한 환경 변수가 주어졌을 때에도 안정적인 예측이 가능하다는 것을 보여주었습니다. 이 연구의 결과는 실제 GNSS 적용에 있어 방해 신호의 감지 및 분류에서 ML 모델의 강인성을 강조하고 있으며, 이를 통해 향후 보안 프로토콜 향상에 기여할 수 있는 가능성을 제시합니다.



### ChatGPT as a Solver and Grader of Programming Exams written in Spanish (https://arxiv.org/abs/2409.15112)
- **What's New**: 본 연구에서는 ChatGPT가 스페인어로 작성된 실제 프로그램 시험 문제를 해결하고 평가하는 능력을 시험합니다. 이 연구는 복잡한 프로그래밍 및 알고리즘 문제를 해결하는 ChatGPT의 능력을 평가하며, AI가 학생의 해결책을 평가하는 데 얼마나 효과적인지를 분석합니다.

- **Technical Details**: 연구에서는 ChatGPT를 gpt-3.5-turbo 버전을 통해 90명의 컴퓨터 과학 학부생이 응시한 프로그램 시험에서 문제를 해결하도록 요청했습니다. 시험은 기본적인 코딩 기술뿐만 아니라 알고리즘 및 데이터 구조 개념을 테스트하며, 7개의 질문으로 구성되었습니다. 과제 및 해답 채점에 대한 AI의 처리 능력을 조사했습니다.

- **Performance Highlights**: ChatGPT는 기본적인 프로그래밍 테스트에서는 평균 학생의 성적과 유사한 결과를 보였으나, 복잡한 문제에 대해서는 30% 이상의 오류율을 보였습니다. 복잡한 프롬프트가 성능 향상에 기여하지 않았고, 특히 ADT 정의 및 파라다이스 알고리즘 문제에서 큰 실패를 보였습니다. 그러나 2-6번 질문에서는 괜찮은 성과를 보였습니다.



### SPformer: A Transformer Based DRL Decision Making Method for Connected Automated Vehicles (https://arxiv.org/abs/2409.15105)
- **What's New**: 이번 연구는 Connected Automated Vehicles (CAVs)를 기반으로 심층 강화 학습(deep reinforcement learning, DRL)과 변환기(transformer) 아키텍처를 결합한 다중 차량 협업 의사 결정 프레임워크인 SPformer를 소개합니다. 이 프레임워크는 차량 간의 상호작용을 모델링하고, 고급 의사 결정 능력을 가능하게 하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: SPformer는 고유한 학습 가능한 정책 토큰을 사용하여 다중 차량의 공동 정책을 생성하며, 이를 통해 모든 차량의 상태를 능동적으로 인지하여 상호작용적 특징을 추출할 수 있습니다. 또한, 직관적인 물리적 위치 인코딩을 설계하여 네트워크 성능을 최적화합니다. 이 연구에서는 on-ramp 시나리오를 통한 시뮬레이션을 활용하여 제안된 방법의 유효성을 검증하였습니다.

- **Performance Highlights**: SPformer는 모델이 모든 차량의 상태 정보를 효과적으로 활용하여 안전성과 효율성을 충족하는 고품질 주행 결정을 내릴 수 있도록 합니다. 기존의 DRL 기반 다중 차량 협동 의사 결정 알고리즘과 비교하여 성능이 크게 향상된 결과를 보였습니다.



### Acting for the Right Reasons: Creating Reason-Sensitive Artificial Moral Agents (https://arxiv.org/abs/2409.15014)
Comments:
          8 pages, 2 figures, Workshop paper accepted to FEAR24 (IFM Workshop)

- **What's New**: 이번 논문에서는 강화 학습(Reinforcement Learning) 에이전트가 규범적(normative) 이유를 기반으로 도덕적(moral) 결정을 내릴 수 있도록 하는 강화 학습 아키텍처의 확장을 제안합니다. 이를 위해 이유 기반(reason-based) 차단기(generator)를 포함하여, 에이전트가 도덕적으로 정당화된 행동만을 수행하도록 제한합니다.

- **Technical Details**: 제안된 시스템에서는 도덕적 의무를 유도하는 모듈이 포함됩니다. 이 모듈은 에이전트가 올바른 행동을 선택할 수 있도록 서로 간의 규범적 이유(normative reasons)의 우선순위를 설정합니다. 또한, 에이전트는 도덕적 판단에 대한 피드백을 통해 이유 이론(reason theory)을 반복적으로 개선할 수 있으며, 이를 위한 '도덕적 판단자(moral judge)'의 역할을 통해 피드백을 받을 수 있습니다.

- **Performance Highlights**: 이 접근법은 기존 보상 신호에 의존하기보다는 에이전트가 도덕적으로 허용된 행동만을 수행하도록 필터링하는 방법을 사용하여, 도덕적으로 정당화된 행동의 이행 가능성을 높일 것으로 기대됩니다. 따라서 에이전트는 보다 명확한 이유와 함께 도덕적 결정을 내릴 수 있는 능력을 강화할 것입니다.



### Analogous Alignments: Digital "Formally" meets Analog (https://arxiv.org/abs/2409.15013)
Comments:
          Accepted for publication at the Design and Verification Conference and Exhibition (DVCon) Europe, Munich, Germany, 2024

- **What's New**: 이 논문은 혼합 신호(Analog Mixed Signal, AMS) 지적 재산(Intellectual Property, IP)의 포괄적인 공식 검증(pragmatic formal verification) 방법론에 대해 다룹니다. 특히, 아날로그 행동 모델(analog behavioral model)을 공식 검증 설정(formal verification setup)에 통합하는 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 Digital 및 Analog Mixed-Signal (AMS) 설계의 통합을 다루며, 형식적 검증(f Formal Verification) 방법 중 FPV(Fully Property Verification), CSR(Connectivity and Robustness verification) 검증, 연결 검사를 활용합니다. 특성들은 메타모델링 프레임워크를 통해 자동 생성되어, 공정한 검증 과정을 지원합니다.

- **Performance Highlights**: 연구 결과, 저희 검증 접근 방식을 통해 설계를 합리적인 시간 내에 포괄적으로 검증할 수 있었으며, 초기 단계에서 여러 버그를 발견하였습니다. 이로 인해 검증 과정이 반복적이고 효율적이게 되었습니다.



### ViBERTgrid BiLSTM-CRF: Multimodal Key Information Extraction from Unstructured Financial Documents (https://arxiv.org/abs/2409.15004)
Comments:
          Accepted in MIDAS (The 8th Workshop on MIning DAta for financial applicationS) workshop of ECML PKDD 2023 conference

- **What's New**: 이 논문은 비정형 문서에서의 핵심 정보 추출(Information Extraction, KIE) 모델에 대한 새로운 접근 방식을 제안합니다. 특히, ViBERTgrid라는 다중 모드 트랜스포머를 비정형 금융 문서에 적응시키고 BiLSTM-CRF 레이어를 통합하여 성능을 크게 향상시킵니다.

- **Technical Details**: 제안된 ViBERTgrid BiLSTM-CRF 모델은 비정형 문서에서의 개체 인식(Named Entity Recognition, NER) 성능을 2%포인트 향상시키면서도 반구조 문서에서의 KIE 성능을 유지합니다. 이 모델은 두 가지 주요 아키텍처인 ViBERTgrid(트랜스포머 기반)와 BiLSTM-CRF(시퀀스 기반)를 결합하여 구문 및 장기 컨텍스트 인식을 제공합니다.

- **Performance Highlights**: 이 모델은 비정형 자금 이체 주문 데이터셋 및 반구조 영수증 데이터셋(SROIE)에서 평가되었으며, SROIE 데이터셋에 대한 토큰 수준 주석을 공개하여 다중 모드 시퀀스 레이블링 모델에서의 사용 가능성을 높였습니다.



### Multi-Modal Generative AI: Multi-modal LLM, Diffusion and Beyond (https://arxiv.org/abs/2409.14993)
- **What's New**: 이 논문에서는 다중 모달 제네레이티브 AI에 대한 체계적인 리뷰와 더불어, 통합 모델의 가능성 및 설계 방안에 대해 논의합니다. 특히, 다중 모달 대형 언어 모델(MLLM)과 확산 모델(diffusion model)의 특징을 비교하며, 이해와 생성을 동시에 수행할 수 있는 통합 모델의 필요성에 대해 다룹니다.

- **Technical Details**: 논문에서는 MLLM과 확산 모델의 확률적 모델링 기법, 아키텍처 디자인, 그리고 영상 및 비디오 생성에 대한 응용을 포함한 공통적인 구조를 분석합니다. 두 가지 중요한 질문을 제시하며, 모델이 자가 회귀(auto-regressive) 또는 확산(probabilistic) 모델링을 채택해야 하는지와 밀집 아키텍처(dense architecture) 또는 전문가 혼합(Mixure of Experts, MoE) 아키텍처를 어떻게 활용할 것인지에 대한 전략도 제안합니다.

- **Performance Highlights**: MLLM은 비주얼 이해 분야에서 두각을 나타내며, 다양한 비주얼-언어 프리트레이닝 방법과 비주얼 토크나이저의 발전을 통해 강화된 성능을 보이고 있습니다. BLIP 모델은 이미지-텍스트 이해 및 생성에서 중요한 역할을 하며, 새로운 데이터셋 생성 방법을 통해 모델의 학습 효율성을 높이고 있습니다.



### TS-TCD: Triplet-Level Cross-Modal Distillation for Time-Series Forecasting Using Large Language Models (https://arxiv.org/abs/2409.14978)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 최근에 제안된 TS-TCD 프레임워크는 시간 대 시리즈(시간-모달)의 정교한 지식 증류 구조를 사용하여 성능을 향상시키는 혁신적인 접근법입니다.

- **Technical Details**: TS-TCD는 동적 적응 게이팅(Dynamic Adaptive Gating)을 통해 입력 인코딩과 정렬을 수행하고, 레이어별 대조 학습(Layer-Wise Contrastive Learning)으로 중간 표현을 정렬하며, 최적 수송 기반 출력 정렬(Optimal Transport-Driven Output Alignment)을 적용하는 세 가지 단계로 구성됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 TS-TCD가 기존 방법들보다 뛰어난 정확도와 강인성을 보여주며, 여러 시간 대 시리즈 데이터셋에서 최신 기술 수준의 결과를 달성했습니다.



### A-VL: Adaptive Attention for Large Vision-Language Models (https://arxiv.org/abs/2409.14846)
- **What's New**: 본 연구에서는 LVLM(Inference)에서의 메모리 및 계산 부하를 감소시키기 위해 A-VL이라는 적응형 어텐션 메커니즘을 제안합니다. 새로운 어텐션 패턴 분석을 통해 각 모달리티에 대해 별도의 어텐션 처리가 필요함을 밝히고, 이를 적절히 조정하여 성능 저하 없이 효율성을 극대화했습니다.

- **Technical Details**: A-VL은 시각 입력 및 언어 입력의 어텐션 패턴 차이를 이해하여 각 모달리티를 별도로 관리합니다. 시각 입력의 경우 중요한 정보를 캐시하여 가장 중대한 부분만 계산하고, 언어 입력은 지역 정보를 중시하여 필요한 원거리 텍스트 캐시만을 유지합니다. 이 연구는 각각의 입력 모드에 대한 메모리 사용 및 계산 부하를 줄이기 위해 KV 캐시의 중대한 부분만 선택하는 방법을 사용합니다.

- **Performance Highlights**: A-VL은 세 가지 비전-언어 작업 및 다섯 가지 데이터 세트에서의 평가를 통해 기존의 적응형 어텐션 방법들에 비해 메모리와 계산 효율성이 높다는 것을 입증했습니다. 따라서 LVLM의 실용성을 크게 향상시킵니다.



### HW-TSC's Submission to the CCMT 2024 Machine Translation Tasks (https://arxiv.org/abs/2409.14842)
Comments:
          14 pages, 2 figures, 6 Tables, CCMT2024

- **What's New**: 화웨이 번역 서비스 센터(HW-TSC)가 제안한 이 논문은 2024년 제20회 중국 기계 번역 회의(CCMT 2024)의 기계 번역 작업에 대한 제출 내용을 소개합니다. 우리는 이중 언어 기계 번역 작업과 다중 도메인 기계 번역 작업에 참여하며, 다양한 학습 전략을 사용하여 신경망 기계 번역(NMT) 모델을 훈련합니다.

- **Technical Details**: 우리는 Regularized Dropout, Bidirectional Training, Data Diversification, Forward Translation, Back Translation, Alternated Training, Curriculum Learning, Transductive Ensemble Learning과 같은 훈련 전략을 사용하여 Deep Transformer-Big 아키텍처 기반의 NMT 모델을 개발했습니다. 또한, 대규모 언어 모델(LLM)인 llama2-13b를 Supervised Fine-Tuning(SFT)을 통해 훈련하여 자동 포스트 편집(APE) 모델로 활용하였습니다.

- **Performance Highlights**: 이러한 다양한 전략을 활용하여 우리 제출물은 최종 평가에서 경쟁력 있는 결과를 달성하였습니다.



### Explainable and Human-Grounded AI for Decision Support Systems: The Theory of Epistemic Quasi-Partnerships (https://arxiv.org/abs/2409.14839)
Comments:
          20 pages

- **What's New**: 본 논문에서는 AI 의사결정 지원 시스템(AI-DSS)에서 윤리적이고 설명 가능한 AI(XAI)의 요구를 충족시키기 위해, 인간 의사결정자에게 세 가지 유형의 인간 기반 설명을 제공하는 RCC 접근법을 제안합니다.

- **Technical Details**: 현재의 XAI 문헌을 검토하며, LIME, SHAP, Anchors와 같은 모델 설명 생성 방법과 모델의 신뢰성, 최종 사용자 정확도 간의 관계를 조사합니다. 이 연구는 좋은 인간 기반 이유를 구성하는 것에 대한 기존 이론들이 증거를 충분히 설명하지 못하거나 개발을 위한 윤리적 조언을 제공하지 못한다는 점을 지적합니다. 따라서 우리는 인식적 준 파트너십 이론(EQP)을 제안합니다.

- **Performance Highlights**: EQP 이론을 채택함으로써 경험적 증거를 설명하고, 윤리적 조언을 제공할 수 있음을 입증하며 RCC 접근법의 채택 필요성을 강조합니다.



### MICSim: A Modular Simulator for Mixed-signal Compute-in-Memory based AI Accelerator (https://arxiv.org/abs/2409.14838)
Comments:
          The 30th Asia and South Pacific Design Automation Conference (ASP-DAC 2025)

- **What's New**: 이번 연구에서는 칩 수준의 소프트웨어 성능 및 하드웨어 오버헤드를 평가하기 위해 설계된 오픈 소스 프리 서킷 시뮬레이터인 MICSim을 소개합니다. MICSim은 모듈화된 설계를 특징으로 하여 복합 신호 컴퓨트 인 메모리(CIM) 가속기에서의 디자인 스페이스 탐색이 용이합니다.

- **Technical Details**: MICSim은 NeuroSim이라는 최신 CIM 시뮬레이터에서 모듈화된 구조를 바탕으로 하며, 여러 양자화 알고리즘 및 다양한 회로/아키텍처 설계를 지원할 수 있는 고도로 구성 가능한 시뮬레이션 프레임워크를 제공합니다. 이를 통해 MICSim은 CNN 및 Transformer의 소프트웨어와 하드웨어 성능 평가를 지원합니다.

- **Performance Highlights**: MICSim은 NeuralSim 대비 9배에서 32배의 속도 향상을 달성하며, 최적화 전략과 결합되어 디자인 스페이스 탐색을 수행하는 데 사용될 수 있습니다.



### Benchmarking Edge AI Platforms for High-Performance ML Inferenc (https://arxiv.org/abs/2409.14803)
- **What's New**: 이 논문은 edge computing 환경의 요구에 따라 CPU, GPU, NPU를 통합한 이종(System-on-Chip, SoC) 플랫폼에서의 신경망 추론과 기본 선형 대수 작업의 성능을 비교 분석합니다.

- **Technical Details**: 연구에서는 매트릭스-벡터 곱셈에서 NPU가 GPU보다 58.6% 빠르고, 비디오 분류 및 대형 언어 모델에서는 3.2배 더 빠른 성능을 보이는 반면, 매트릭스 곱셈에서는 GPU가 22.6% 더 빠르며 LSTM 네트워크에서는 2.7배 더 우월함을 발견했습니다. CPU는 점 곱과 같은 저병렬 연산에서 가장 낮은 대기 시간을 기록했습니다.

- **Performance Highlights**: NPU에 기반한 추론은 더 낮은 전력 소비로 지연 시간과 처리량의 균형을 제공합니다. 반면, GPU는 대규모 차원과 배치 크기에서 최고의 성능을 나타내지만 에너지 소모가 더 큽니다. 이 연구는 다양한 계산 유닛을 전략적으로 활용하여 edge AI에서의 정확하고 실시간 추론을 증진할 수 있는 이종 컴퓨팅 솔루션의 잠재력을 강조합니다.



### Choose the Final Translation from NMT and LLM hypotheses Using MBR Decoding: HW-TSC's Submission to the WMT24 General MT Shared Task (https://arxiv.org/abs/2409.14800)
Comments:
          10 pages, 4 figures, 2 Tables, EMNLP2024

- **What's New**: 이 논문은 Huawei Translate Services Center (HW-TSC)의 WMT24 일반 기계 번역(MT) 공유 작업 참가 결과를 제시합니다. 영어에서 중국어(en to zh) 언어 쌍에 대한 최신 차별화된 접근 방식을 통합하여 모델을 훈련시키고 경쟁력 있는 성과를 기록했습니다.

- **Technical Details**: 본 연구에서는 Transformer-big 아키텍처를 기반으로 하는 신경망 기계 번역(NMT) 모델을 훈련시키기 위해 정규화된 드롭아웃(regularized dropout), 양방향 훈련(bidirectional training), 데이터 다각화(data diversification), 전방 번역(forward translation), 후방 번역(back translation) 등 다양한 훈련 전략을 사용했습니다. 또한 큰 언어 모델(LLM)에 대해 계속 사전 훈련(continue pre-training), 감독 세부 조정(supervised fine-tuning), 대조 선호 최적화(contrastive preference optimization)를 추가하여 MT 모델을 훈련했습니다. 최소 베이지안 리스크(Minimum Bayesian Risk, MBR) 디코딩을 사용하여 여러 가설 중 최종 번역을 선택합니다.

- **Performance Highlights**: 우리는 MBR 디코딩을 사용하여 NMT 및 LLM 기반 MT 모델에 대해 경쟁력 있는 결과를 기록했습니다. 이러한 접근 방식은 대조적 선호 최적화를 통해 번역 성능을 향상시키는 데 효과적임을 보여주었습니다.



### SAMEdge: An Edge-cloud Video Analytics Architecture for the Segment Anything Mod (https://arxiv.org/abs/2409.14784)
- **What's New**: AI가 다양한 비디오 분석 작업을 단일 대형 모델로 처리할 수 있는 능력이 향상되고 있으며, Segment Anything Model(SAM)이 이에 중요한 역할을 합니다. 논문에서는 이를 위해 새로운 엣지-클라우드 컴퓨팅 아키텍처인 SAMEdge를 제안합니다.

- **Technical Details**: SAMEdge는 엣지 사용자에 대한 SAM 계산을 지원하기 위해 새로운 모듈들을 통합하여analytics accuracy를 최적화합니다. 또한 visual prompt transformation 및 이미지 인코딩을 위한 효율적인 작업 분할 알고리즘을 개발했습니다.

- **Performance Highlights**: SAMEdge는 기존 SAM 접근 방식에 비해 비디오 분석 정확도를 최대 4.64배 및 1.81배 향상시키는 성능을 보여주었습니다. 이를 통해 네트워크 대역폭에 따라 다양한 프롬프트 입력에서도 성능을 유지할 수 있음을 입증했습니다.



### Speechworthy Instruction-tuned Language Models (https://arxiv.org/abs/2409.14672)
Comments:
          EMNLP2024

- **What's New**: 이번 연구는 텍스트 기반의 선호 데이터를 사용하는 기존의 instruction-tuned language models (ITLMs)에 대안으로, 20,000개의 샘플로 구성된 음성 기반 선호 데이터를 활용하여 모델을 음성 영역에 맞게 조정하는 방법을 제안합니다.

- **Technical Details**: 두 가지 주요 접근법인 (i) 프롬프트 엔지니어링 (prompt engineering)과 (ii) 선호 학습 (preference learning)을 사용하여 ITLM의 음성 적합성을 개선합니다. 연구는 연관 데이터인 SpeechPref를 첨부하여 실제 음성을 통해 피어 평가하며, 모델은 proximal policy optimization (PPO) 및 direct preference optimization (DPO)을 통해 조정됩니다.

- **Performance Highlights**: 연구 결과, 프롬프트와 선호 학습을 합친 방식이 평균 76.2%의 선호 비율을 보이며, Falcon 7B Instruct 및 OLMo 7B Instruct 모델에서 기존 모델보다 유의미한 향상이 있음을 입증했습니다. 또한, 다른 모델들에 비해 최대 88.3%의 선호를 기록하였습니다.



### FedGCA: Global Consistent Augmentation Based Single-Source Federated Domain Generalization (https://arxiv.org/abs/2409.14671)
Comments:
          6 pages, 7 figures, conference

- **What's New**: 본 논문은 단일 출처 연합 도메인 일반화(single-source FedDG) 문제를 다루며, 이를 해결하기 위한 새로운 방법인 연합 글로벌 일관성 증강(FedGCA)을 제안합니다. 이 방법은 다양한 도메인 스타일로 데이터를 증강하기 위해 스타일 보강 모듈을 통합합니다.

- **Technical Details**: FedGCA는 제한된 도메인 스타일의 데이터 샘플을 증강하기 위해 스타일 보강 모듈을 사용합니다. 이 과정에서 FedGCA는 글로벌 가이드 의미 일관성(global guided semantic consistency)과 클래스 일관성(class consistency) 손실을 통해 각 클라이언트와 클래스 간의 의미 불일치를 완화합니다.

- **Performance Highlights**: 다양한 데이터셋에서 실시된 실험 결과, FedGCA는 기존의 벤치마크 모델보다 우수한 성능을 보이며, 이를 통해 sFedDG 문제를 효과적으로 해결함을 입증합니다.



### Semi-supervised Learning For Robust Speech Evaluation (https://arxiv.org/abs/2409.14666)
Comments:
          6 pages

- **What's New**: 이 논문에서는 학습자의 구술 능력을 평가하기 위한 자동 모델의 Speech evaluation (스피치 평가) 방법에 대해 논의합니다. 한정된 스코어 데이터와 불균형적인 스코어 분포로 인해 발생하는 문제를 해결하기 위해 반도체습식 사전 학습(semi-supervised pre-training)과 객관적 규제(objective regularization)를 이용한 새로운 접근 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 normalized mutual information을 사용하여 학습자의 발화 특징을 양적화하고, pseudo labels를 활용해 발음의 올바름을 예측하는 anchor model을 훈련합니다. 또한, 발음 평가 모델과 앵커 모델 사이의 두 확률 분포 간의 다르기를 최소화하는 보간 손실(interpolated loss function)을 제안하여 정확도를 높입니다.

- **Performance Highlights**: 제안된 방법은 공개 데이터 세트에서 기존 최첨단 방법들에 비해 전체 테스트 세트를 평가할 때 높은 성능을 달성하며, 각 능숙도 수준 간의 예측 오류를 고르게 분포시킵니다. 또한, 실제 배포 상황에서 발생할 수 있는 out-of-distribution 데이터에 대한 모델 정확도 또한 경쟁적인 기준선과 비교하여 우수한 결과를 보였습니다.



### Brain Surgery: Ensuring GDPR Compliance in Large Language Models via Concept Erasur (https://arxiv.org/abs/2409.14603)
- **What's New**: 최근 대규모 AI 시스템의 확산에 따라 개인정보 보호법인 GDPR(General Data Protection Regulation) 준수가 필수적이라는 점이 강조되고 있습니다. 이 논문에서는 모든 로컬 AI 모델을 GDPR에 맞게 준비할 수 있도록 돕는 'Brain Surgery' 방법론을 소개합니다. 'Brain Surgery'는 향상된 기술인 Embedding-Corrupted Prompts (ECO Prompts), 블록체인 기반의 개인정보 관리 및 개인정보 인식 지속 학습을 결합하여 실시간 개인정보 관리를 가능하게 합니다.

- **Technical Details**: 'Brain Surgery'는 로컬 AI 모델에서 원치 않는 데이터를 제거하면서 모델의 전반적인 성능을 유지하는 타겟팅된 유학(unlearning) 및 동적 개인정보 관리의 조합을 통해 GDPR 준수를 가능하게 합니다. ECO Prompts는 특정 개념과 관련된 임베딩 공간에 교란을 적용하여 원치 않는 정보를 효과적으로 '잊게' 하고, 실시간 충돌 모니터링을 통해 모델의 정확성을 유지합니다.

- **Performance Highlights**: 'Brain Surgery' 방법론은 개인정보 보호 제약 사항을 실시간으로 반영하여 학습 목표를 동적으로 조정하는 지속 학습 메커니즘을 도입했습니다. 블록체인 기반의 개인정보 관리 레이어는 '잊혀질 권리(right to be forgotten)' 요청을 로그로 남겨 투명성을 제공합니다. 이 시스템은 GDPR 요구 사항을 충족할 뿐만 아니라 사용자 요청에 따라 개인정보 보유 기간을 설정하거나 훈련과 추론 과정에서 제외할 정보를 지정할 수 있는 기능을 제공합니다.



### Evaluating Gender, Racial, and Age Biases in Large Language Models: A Comparative Analysis of Occupational and Crime Scenarios (https://arxiv.org/abs/2409.14583)
Comments:
          10 pages, 17 figures

- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 발전이 두드러지지만, 기업에서의 광범위한 채택은 여러 제한사항으로 인해 제한적입니다. 본 논문은 LLMs에서의 편향(bias) 문제를 다루며, 이는 사용성, 신뢰성 및 공정성에 영향을 미치는 중요한 이슈입니다.

- **Technical Details**: 연구자들은 편향 완화(debiasing) 층, Winogender와 Winobias와 같은 전문화된 참고 데이터셋, 인간 피드백이 포함된 강화 학습(Reinforcement Learning with Human Feedback, RLHF) 등 다양한 전략을 개발하고 있습니다. 이러한 기법들은 2024년에 출시된 네 가지 주요 LLM(Gemini 1.5 Pro, Llama 3 70B, Claude 3 Opus, GPT-4o)에 통합되어 평가되었습니다.

- **Performance Highlights**: 우리의 연구 결과, LLM들은 다양한 직업에서 여성 캐릭터를 남성보다 37% 더 자주 묘사하고 있으며, 범죄 시나리오에서 성별 편향은 54%, 인종 편향은 28%, 나이 편향은 17%의 편차를 보였습니다. 성별 및 인종 편향을 줄이기 위한 노력은 종종 특정 하위 클래스에 대한 과도한 지수를 초래하여 문제를 악화시킬 가능성이 있습니다. 이러한 결과는 현재의 편향 완화 기법의 한계를 강조하며, 보다 효과적인 접근 방식의 필요성을 보여줍니다.



### Encoder with the Empirical Mode Decomposition (EMD) to remove muscle artefacts from EEG signa (https://arxiv.org/abs/2409.14571)
- **What's New**: 이 논문은 EEG 신호에서 아티팩트를 효과적으로 제거하기 위한 새로운 방법을 제시합니다. 이 방법은 Empirical Mode Decomposition (EMD) 기법과 머신러닝 아키텍처를 결합하여 기존 아티팩트 제거 기술의 한계를 극복합니다.

- **Technical Details**: 제안된 방법은 EMD 기법을 보강하기 위해 신호의 상단 및 하단의 보간(interpolation)을 통해 아티팩트를 제거합니다. 기존 아티팩트 제거 방법에서는 EMD 기법이 일반적으로 사용되지만, 신호의 중요한 주파수 성분을 유지하면서 누락된 성분을 정확하게 보간하는 것이 도전 과제가 됩니다. 이 논문에서는 머신러닝 기술을 통합하여 데이터에 직접적으로 개입하지 않고 보간 과정을 정교하게 처리합니다.

- **Performance Highlights**: 우리 접근법의 주요 장점은 아티팩트 제거 과정에서 EEG 신호의 자연적인 특성을 유지하는 것입니다. 머신러닝을 활용하여 보간을 수행함으로써 EMD 방법을 통해 얻어진 평균 성분이 원래 신호의 중요한 주파수 성분을 유지할 수 있게 됩니다. 이 보존은 EEG 데이터의 무결성과 충실도를 유지하는 데 필수적이며, 정확한 분석 및 해석을 가능하게 합니다. 평가 결과는 우리의 접근법의 효과를 검증하고 EEG 신호 처리 및 분석의 추가 발전을 위한 길을 열어줍니다.



### Why Is Anything Conscious? (https://arxiv.org/abs/2409.14545)
- **What's New**: 이 연구는 생물학적 시스템이 비표지 감각 정보를 계층적으로 해석하는 방법을 수학적으로 설명하며, 자아, 세계, 다른 자아를 모델링하는 능력이 없으면 인간 수준의 접근 의식이 불가능하다는 주장을 합니다.

- **Technical Details**: 연구는 'pancomputationalism'과 'enactivism'의 개념에 기반하여, 하위 및 상위 수준의 정보 처리 이론을 통합한 수학적 기초를 제공합니다. 이는 각 상태 간의 인과 관계를 정의하고, 정보 처리가 한 상태에서 다른 상태로의 전이를 초래하는 메커니즘을 설명합니다.

- **Performance Highlights**: 이 연구는 인간 수준의 사고와 기능을 이해하기 위해 현상적 의식(phenomenal consciousness)이 필수적임을 강조하며, 자아와 외부 세계의 정보 처리가 어떻게 연계되는지를 밝힙니다.



### Beyond Words: Evaluating Large Language Models in Transportation Planning (https://arxiv.org/abs/2409.14516)
- **What's New**: 2023년 Generative Artificial Intelligence (GenAI)의 급속한 발전이 도시 교통 및 물류 분야에 혁신적인 변화를 가져왔습니다. 본 연구는 GPT-4와 Phi-3-mini 같은 Large Language Models (LLMs)의 성능을 수송 계획에 적용하는 것을 탐구합니다.

- **Technical Details**: 이 연구는 교통 정보를 반영한 평가 프레임워크를 통해 LLM의 성능과 공간 이해력을 평가합니다. 평가 요소로는 일반적인 지리적 정보 시스템 (GIS) 기술, 교통 관련 도메인 지식 및 현실 세계의 교통 문제 해결 능력이 포함됩니다. 혼합 방법론을 활용하여 연구가 진행되었습니다.

- **Performance Highlights**: 연구 결과, GPT-4는 다양한 GIS 및 교통 관련 작업에서 Phi-3-mini보다 더 뛰어난 정확성과 신뢰성을 보였습니다. 그러나 Phi-3-mini는 특정 분석 시나리오에서 유용함을 나타내어 자원이 제한된 환경에서도 활용 가능성을 보여줍니다. 이 결과는 GenAI 기술이 도시 교통 계획에 미치는 혁신적인 잠재력을 강조합니다.



### On a measure of intelligenc (https://arxiv.org/abs/2409.14496)
- **What's New**: 이번 논문에서는 François Chollet의 'On the measure of intelligence'라는 글을 바탕으로 지능, 지능 측정, 그리고 관련된 쟁점들에 대해 논의하고 있습니다.

- **Technical Details**: Chollet는 Algorithmic Information Theory(AIT)를 기반으로 지능의 새로운 형식적 정의를 제시하며, 지능을 스킬 습득 효율(skill-acquisition efficiency)로 설명합니다. 이 정의는 범위(scope), 일반화 난이도(generalization difficulty), 프라이어(priors), 경험(experience) 등을 지능 체계의 특성으로 강조합니다.

- **Performance Highlights**: Chollet의 접근 방식은 인간과 유사한 일반 지능을 양적으로 측정하기 위한 첫 걸음으로, 지능을 스킬 습득의 효율성으로 재정의한 점이 주목할 만합니다. 그는 스킬 습득이 지능의 과정(process)이며, 기존의 IQ 점수가 단순한 지능의 측정 방식이라는 점에서 비판적인 논의를 제기합니다.



### Can Large Language Models Logically Predict Myocardial Infarction? Evaluation based on UK Biobank Cohor (https://arxiv.org/abs/2409.14478)
- **What's New**: 이번 연구는 최신 LLM(대형 언어 모델)인 ChatGPT와 GPT-4가 실제 의료 데이터를 기반으로 심근경색(MI)의 발생 위험을 예측할 수 있는지를 정량적으로 평가한 것입니다.

- **Technical Details**: 연구에서는 2006년부터 2010년까지 모집된 482,310명의 UK Biobank 데이터베이스 참가자들을 대상으로 하였고, 최종적으로 690명으로 재샘플링하여 분석했습니다. MI 위험 요인에 대한 표 형식 데이터를 ChatGPT가 인식할 수 있는 표준화된 텍스트 설명으로 변환하였고, 0에서 10까지의 점수를 선택하도록 요청하여 위험도를 평가했습니다. Chain of Thought (CoT) 질문 기법을 사용하여 LLM의 예측 논리성을 평가했습니다.

- **Performance Highlights**: ChatGPT의 예측 성능은 기존 의료 지표, 전통적인 머신러닝 모델 및 다른 LLM과 비교되었으며, 연구 결과 현재 LLMs는 임상 의학 분야에 적용하기에는 준비가 미비하다는 결론을 도출했습니다.



### On logic and generative AI (https://arxiv.org/abs/2409.14465)
- **What's New**: 이 논문에서는 AI 혁명이 우리가 전에 알지 못했던 심오한 기초 문제들을 제기하고 있으며, 특히 젊은 논리학자들이 이러한 문제에 관심을 가져야 함을 강조하고 있습니다.

- **Technical Details**: AI의 발전이 뇌과학(neuroscience), 철학(philosophy), 컴퓨터 과학(computer science), 논리(logic) 등 여러 분야에 영향을 미치고 있습니다. 특히 생성형 AI(generative AI)의 성능과 지능(intelligence)에 대한 논의가 활발합니다. System 1(빠른 사고)와 System 2(느린 사고)의 구분을 통해 인간의 사고 구조 또한 논의되었습니다.

- **Performance Highlights**: 현재 LLMs(대형 언어 모델)는 여러 면에서 유용하지만, 여전히 인간 수준의 지능에 도달하기에는 필수적인 능력인 이해(understanding), 지속적인 기억(persistent memory), 추론(reasoning), 계획(planning) 등이 결여되어 있습니다.



### Large Model Agents: State-of-the-Art, Cooperation Paradigms, Security and Privacy, and Future Trends (https://arxiv.org/abs/2409.14457)
Comments:
          35 pages, 23 figures, 9 tables

- **What's New**: 이 논문은 대형 모델(Large Model, LM) 에이전트에 대한 포괄적인 조사로, 이러한 에이전트가 인공지능의 일반 지능(Artificial General Intelligence, AGI) 달성을 위한 중요한 진전을 보여준다고 강조하고 있습니다. 특히, LM 에이전트의 자율성, 구현, 그리고 연결성의 주요 특성을 설명하며, 물리적, 가상 및 혼합 현실 환경에서 원활하게 작업할 수 있도록 합니다.

- **Technical Details**: LM 에이전트는 기획(planning), 행동(action), 기억(memory), 상호작용(interaction) 등 네 가지 주요 구성 요소로 구성됩니다. 그들의 능력은 인공지능-인간 상호작용(Human-Machine Interaction, HMI), 복잡한 패턴 인식, 지식 유지, 추론, 장기 계획, 일반화 및 적응성을 포함합니다. 특히, Chain-of-Thought (CoT), Tree-of-Thought (ToT) 및 회상(reflection)과 같은 고급 추론 및 몇 가지/제로 샷 계획 기술을 통해 복잡하고 다면적인 작업을 효과적으로 해결할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: LM 에이전트는 웹 검색, 추천 시스템, 가상 비서, 메타버스 게임, 로봇 공학, 자율 주행 차량 등 다양한 분야에 걸쳐 널리 응용되고 있으며, 시장에서의 가치는 2023년 480억 달러에 달하며, 2028년까지 285억 달러에 이를 것으로 예상됩니다. 그러나 보안 및 개인 정보 보호 문제는 이러한 에이전트의 광범위한 채택에 있어 여전히 큰 장벽으로 남아 있습니다.



### Scoring rule nets: beyond mean target prediction in multivariate regression (https://arxiv.org/abs/2409.14456)
- **What's New**: 이번 논문에서는 다변량 (multivariate) 데이터에서 최대 우도 추정 (Maximum Likelihood Estimation, MLE)의 한계를 해결하기 위해 Conditional CRPS라는 새로운 평가 지표를 제안합니다.

- **Technical Details**: Conditional CRPS는 CRPS(Continuous Ranked Probability Score)를 확장한 것으로, 다변량 환경에서 감도 (sensitivity) 분석이 가능하고, 주요 분포에 대한 폐쇄형 식 (closed-form expressions)을 제공합니다.

- **Performance Highlights**: 실험 결과, Conditional CRPS는 MLE보다 더 나은 성능을 보이며, Distributional Random Forest (DRF)와 같은 최신 비모수 (non-parametric) 모델과 비교했을 때도 유사한 결과를 나타냅니다.



### OStr-DARTS: Differentiable Neural Architecture Search based on Operation Strength (https://arxiv.org/abs/2409.14433)
- **What's New**: 이 논문에서는 기존의 크기 기반 선택(Magnitude-based Selection) 방법을 중단하고, 최종 손실에 미치는 효과를 통해 연산의 중요성을 평가하는 새롭고 혁신적인 기준인 Operation Strength를 제안합니다. 이 방법은 Supernet 최적화 과정을 수정하지 않고도 DARTS의 퇴화 문제를 효과적으로 해결할 수 있음을 보여줍니다.

- **Technical Details**: DARTS는 두 단계로 이루어진 네트워크 구조 검색(NAS) 방법입니다: 첫 번째는 Mixed Operations로 구성된 Supernet의 최적화를 위한 Gradient Descent이며, 두 번째는 Supernet에서 가장 큰 기여를 하는 연산을 선택하여 최종 아키텍처를 구축하는 과정입니다. 새로운 선택 기준인 Operation Strength는 연산의 손실에 미치는 영향을 통해 연산의 중요성을 평가합니다. 이 방법을 통해 퇴화 문제를 해결하는 OStr-DARTS가 제안되었습니다.

- **Performance Highlights**: NAS-Bench-201 및 DARTS 검색 공간에 대한 실험 결과, OStr-DARTS가 DARTS의 안정성 문제를 효과적으로 해결하며, 궁극적으로 더 나은 검색 성능을 달성할 수 있음을 입증하였습니다.



### MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting (https://arxiv.org/abs/2409.14393)
Comments:
          ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2024) Project page: this https URL

- **What's New**: MaskedMimic은 다양한 제어 모달리티를 지원하는 단일 물리 기반 캐릭터 제어 모델을 구현하는 새로운 접근 방식으로, 부분적인 동작 설명(예: 마스크된 키프레임, 객체, 텍스트 설명 등)으로부터 동작을 합성하는 문제로 정의된다.

- **Technical Details**: 이 연구에서는 한 가지 통합 모델을 훈련시켜 부분적인 동작 서술에서 전체 동작을 복원하는 'motion inpainting' 기법을 채택하였다. 이를 위해 모션 트래킹 데이터와 다양한 모션 설명을 효과적으로 활용할 수 있는 확장 가능한 훈련 방법을 설계하였다.

- **Performance Highlights**: MaskedMimic은 이전의 특정 작업에 특화된 방법보다 다양한 작업에서 더 나은 성능을 발휘하며, 특히 VR 입력에서 전체 신체 동작을 생성하고 불규칙한 지형과 새로운 객체에 대한 일반화 측면에서 우수하다.



### To Err Is AI! Debugging as an Intervention to Facilitate Appropriate Reliance on AI Systems (https://arxiv.org/abs/2409.14377)
Comments:
          Paper accepted at HT'24 as late-break. This is an expanded version of HT'24 paper, providing more details and experimental analysis

- **What's New**: 이 논문은 AI 시스템에 대한 인간의 적절한 의존성을 촉진하기 위해 디버깅(dibugging)이라는 개입(intervention)의 사용을 제안합니다. 기존의 연구들은 AI 조언의 신뢰성을 정확하게 추정하는 것이 매우 어렵다는 점을 강조했습니다. 이 연구는 디버깅 개입이 AI 시스템의 신뢰성을 사용자에게 효과적으로 전달하는데 도움이 될 수 있는지를 탐구했습니다.

- **Technical Details**: 연구는 234명의 참가자를 대상으로 한 정량적(empirical) 실험을 통해, 디버깅 개입이 AI 시스템에 대한 사용자 신뢰도 및 의존성을 어떻게 변화시키는지를 분석했습니다. 특히, AI 시스템의 약점을 일찍 노출했을 때 사용자의 의존성이 오히려 감소하는 경향을 관찰했습니다. 이를 통해 사용자 신뢰도(user confidence)와 AI 신뢰성(trustworthiness) 추정과의 관계성을 분석했습니다.

- **Performance Highlights**: 연구 결과, 디버깅 개입이 사용자들에게 AI 성능에 대한 신뢰를 잘 조정하지 못했으며, 오히려 의존도가 낮아졌다는 점이 강조되었습니다. 이러한 결과는 사용자들이 AI의 약점을 미리 인지하게 되었을 때, 스스로의 능력을 과대평가하여 부적절한 의존 패턴이 발생할 수 있음을 시사합니다.



### MANTA -- Model Adapter Native generations that's Affordab (https://arxiv.org/abs/2409.14363)
- **What's New**: 본 연구에서는 하드웨어와 비용 제약을 고려하여 과거의 연구를 일반화한 모델-어댑터 조합 문제를 제안하며, 이를 해결하기 위한 새로운 접근 방식인 MANTA를 소개합니다.

- **Technical Details**: MANTA 시스템은 COCO 2014 검증 데이터셋에서 실험을 통해 이미지 작업의 다양성과 품질에서 기존 시스템보다 우수한 성능을 보여주지만, 정렬성에서는 약간의 감소를 보입니다. 본 연구는 모델 체크포인트와 추가 어댑터의 조합을 통해 이미지 생성 워크플로우를 구성하는 문제를 다루고 있으며, 사용자로 하여금 적절한 모델-어댑터 조합을 찾는 방법론을 제시합니다.

- **Performance Highlights**: MANTA 시스템은 작업 다양성에서 94%의 승률, 작업 품질에서 80%의 승률을 기록하여 강력한 성능을 입증하였으며, 합성 데이터 생성과 창의적인 예술 분야에서 직접 활용될 가능성을 보여줍니다.



### LLMs are One-Shot URL Classifiers and Explainers (https://arxiv.org/abs/2409.14306)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)을 활용해 악성 URL 분류 문제를 해결하는 새로운 방법을 제안합니다. 특히, Chain-of-Thought (CoT) 추론 방식을 적용하여 주어진 URL이 benign인지 phishing인지 예측하는 LLM 기반의 one-shot learning 프레임워크를 도입하였습니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 URL 데이터 세트와 다섯 가지 최신 LLM을 사용하여 평가되었으며, LLM의 one-shot 학습이 감독 모델과 유사한 성능을 제공함을 시연합니다. GPT-4 Turbo 모델이 가장 뛰어난 성능을 보였고, 이어서 Claude 3 Opus가 뒤를 잇습니다. 또한, LLM이 제공하는 설명은 사용자가 সহজ하게 이해할 수 있도록 높은 가독성, 일관성, 정보성을 갖추고 있음을 보여줍니다.

- **Performance Highlights**: 제안된 LLM 기반 접근 방식은 감독 학습 환경과의 비교에서 F1 점수 0.05에서 0.12 이내의 예측 성능을 달성했습니다. GPT-4 Turbo는 one-shot 설정에서 평균 F1 점수 0.92를 기록하였으며, 이는 완전 감독 설정인 0.99에 비해 단 0.07포인트 낮습니다.



### UU-Mamba: Uncertainty-aware U-Mamba for Cardiovascular Segmentation (https://arxiv.org/abs/2409.14305)
- **What's New**: 본 논문에서는 심장 및 혈관 segmentation을 위한 UU-Mamba 모델을 소개하며, 이전의 U-Mamba 아키텍처의 확장 버전으로, 데이터셋의 부족 문제를 해결하는 데 중점을 두고 있습니다. 특히, Sharpness-Aware Minimization (SAM) 기법을 활용하여 loss landscape에서 더 넓고 편평한 최소값을 타겟으로 하여 모델의 일반화 및 안정성을 개선했습니다.

- **Technical Details**: UU-Mamba 모델은 불확실성 인식 손실 함수(unity-aware loss function)를 통합하여 지역 기반, 분포 기반 및 픽셀 기반 구성 요소를 결합함으로써 정확한 segmentation을 달성합니다. 이 모델은 자동 학습 가능한 가중치(auto-learnable weights)를 사용하여 각 예측의 불확실성에 따라 손실 구성 요소의 기여도를 동적으로 조정할 수 있습니다. 또한, 모델은 SAM optimizer를 통해 과적합(overfitting)의 위험을 줄이고 일반화 능력을 향상시킵니다.

- **Performance Highlights**: UU-Mamba 모델은 TransUNet, Swin-Unet, nnUNet 및 nnFormer와 같은 선도하는 모델들과 비교하여 뛰어난 성능을 보였으며, ImageCAS 및 Aorta 데이터셋에서의 새로운 평가 결과를 통해 다양한 복잡성의 segmentation 문제에 대한 적응성과 강인성을 입증했습니다.



### HM3D-OVON: A Dataset and Benchmark for Open-Vocabulary Object Goal Navigation (https://arxiv.org/abs/2409.14296)
- **What's New**: 이 논문에서는 Habitat-Matterport 3D Open Vocabulary Object Goal Navigation dataset (HM3D-OVON)을 소개합니다. 이전 Object Goal Navigation (ObjectNav) 벤치마크보다 더 넓고 다양한 의미의 범위를 수용하는 대규모 벤치마크로, 379개의 물체 카테고리에서 15,000개 이상의 주석이 달린 가정용 물체 인스턴스를 포함합니다.

- **Technical Details**: HM3D-OVON은 HM3DSem 데이터셋을 활용하여 자유형 언어로 정의된 개방형 목표에 대해 에이전트를 훈련 및 평가할 수 있는 데이터셋입니다. 기존의 ObjectNav 데이터셋은 목표 개체를 6-20개의 고정된 카테고리로 제한했으나, HM3D-OVON은 언어에 기반한 탐색을 통해 변수가 있는 카테고리로의 일반화를 연구합니다.

- **Performance Highlights**: HM3D-OVON을 사용한 연구 결과, open-vocabulary ObjectNav 에이전트가 기존의 최첨단 ObjectNav 방식보다 더 높은 성능과 로컬리제이션(localization) 및 작동 노이즈에 대한 강인성을 보입니다. DAgRL라는 모듈 방법이 최상의 성능을 기록하고, VLFM과 같은 다른 접근법은 개방형 목표 범주에서 일관된 성능을 보입니다.



### Can-Do! A Dataset and Neuro-Symbolic Grounded Framework for Embodied Planning with Large Multimodal Models (https://arxiv.org/abs/2409.14277)
- **What's New**: 이 논문은 Can-Do라는 새로운 벤치마크 데이터 세트를 도입하여 대형 다중 모달 모델의 체화된 계획 능력을 평가합니다. 이 데이터 세트는 이전의 데이터 세트보다 더 다양한 복잡한 시나리오를 포함하고 있으며, 400개의 다중 모달 샘플로 구성되어 자연어 사용자 지침, 환경을 묘사하는 시각 이미지, 상태 변화 및 해당 동작 계획을 포함하고 있습니다.

- **Technical Details**: Can-Do 데이터 세트는 실제 환경을 묘사하기 위해 실제 장면 이미지와 합성 이미지를 모두 활용합니다. 세 가지 태스크 카테고리(물리적 이해, 상식, 안전)을 중심으로 설계되었으며, 각 샘플은 사용자 의도를 기반으로 시각 시나리오를 인식하고 단계를 생성하는 모델의 능력을 평가합니다. 연구에서는 또한 NeuroGround라는 신경 상징적 프레임워크를 제안하여 모델 생생 생성 과정이 환경의 초기 및 목표 상태에 명확하게 기반하도록 합니다.

- **Performance Highlights**: 실험 결과, NeuroGround 프레임워크는 기존의 강력한 기준선과 비교하여 상당한 이점을 보여주었습니다. 특히, 체화된 계획에서 기존 모델(GPT-4V 포함)의 병목 현상인 시각적 지각, 이해 및 추론 능력에서 개선된 성능을 입증했습니다.



### Predicting Coronary Heart Disease Using a Suite of Machine Learning Models (https://arxiv.org/abs/2409.14231)
Comments:
          14 pages, 3 figures, 2 tables

- **What's New**: 이 연구는 Coronary Heart Disease (CHD) 예측을 위한 여러 가지 머신러닝 알고리즘을 비교하여 가장 정확한 방법을 찾아냄으로써, 기존 연구의 한계를 극복하려고 시도했습니다. 특히 Random Forest 모델이 84%라는 높은 정확도로 가장 뛰어난 성능을 보였습니다.

- **Technical Details**: 연구는 여러 머신러닝 기법에 대한 문헌을 리뷰하고, CRISP-DM 프레임워크를 사용하여 데이터 전처리 및 모델링을 수행했습니다. 여러 머신러닝 알고리즘을 사용하여 데이터를 분석하고, 결정 트리, K-최근접 이웃, SVM, Naive Bayes, 그리고 하이브리드 모델 등 다양한 방법론과 수학적 접근법을 제시했습니다. 이러한 방법들은 초기에 레이블된 데이터에 기반하여 훈련되고, 비슷한 맥락에서 비레이블 데이터를 활용하여 모델 성능을 높이고자 했습니다.

- **Performance Highlights**: Random Forest 모델은 예측 정확도가 84%에 달하며, 여러 비교 대상 중 가장 높은 성능을 기록했습니다. 이 연구에서 사용된 모든 모델은 재현성이 높고, 간단한 비술어적 코드로 쉽게 구현할 수 있다는 점이 강조되었습니다.



### AI Assistants for Spaceflight Procedures: Combining Generative Pre-Trained Transformer and Retrieval-Augmented Generation on Knowledge Graphs With Augmented Reality Cues (https://arxiv.org/abs/2409.14206)
Comments:
          Accepted for the ESA SPAICE Conference 2024: AI in and for Space

- **What's New**: 이 논문은 우주 비행 중 우주 비행사들을 지원하기 위해 설계된 새로운 지능형 개인 비서(IPA)인 CORE(체크리스트 조직기)를 소개합니다. 특히, CORE는 Knowledge Graphs (KGs), Retrieval-Augmented Generation (RAG), Generative Pre-Trained Transformer (GPT), Augmented Reality (AR) 요소를 결합하여 절차 단계를 직관적으로 이해할 수 있도록 합니다.

- **Technical Details**: CORE의 설계는 모듈식 접근 방식을 기반으로 하며, 오프라인 배치가 가능한 오픈 소스 구성 요소를 통합할 수 있습니다. 이 구성 요소들은 GPT, RAG, KGs, 음성 인식, 음성 합성 기술을 포함하여 실제 대화처럼 인간과의 상호작용을 모방하도록 설계되었습니다. AR 요소는 헤드업 디스플레이(HUD)에 표시되어 우주 비행사들에게 시각적 지원을 제공합니다.

- **Performance Highlights**: CORE는 신뢰성, 유연성, 오프라인 가용성, 시각적 3D 정보 통합의 네 가지 주요 기준을 만족시키며, 이는 현대 우주 비행에서 매우 중요한 요소입니다. 이러한 요소들은 우주 비행사들이 의사 결정을 할 때 필수적인 정보에 쉽고 빠르게 접근할 수 있도록 도와줍니다.



### Loop-Residual Neural Networks for Iterative Refinemen (https://arxiv.org/abs/2409.14199)
- **What's New**: 이 논문에서는 새로운 Loop-Residual Neural Network (회로-잔여 신경망)을 소개하여 언어 모델의 성능을 향상시키는 방법을 제안합니다. 이는 모델 크기를 증가시키지 않고도 더 긴 계산 시간을 활용하여 예측을 반복적으로 개선하는 방식입니다.

- **Technical Details**: Loop-Residual 모델은 잔여 연결 (residual connections)을 사용하여 입력을 여러 번 반복하고, 각 반복에서 현재 상태와 원하는 상태 간의 잔여를 예측합니다. 이 접근 방식은 변환기 블록을 반복적으로 통과하면서 예측 정확도를 높입니다.

- **Performance Highlights**: 기존 GPT-2 모델과 비교했을 때, Loop-Residual GPT-2-81M 모델은 OpenWebText 데이터셋에서 3.11의 검증 손실(validation loss)을 달성하여 124M 파라미터를 가진 GPT-2-124M 모델의 3.12와 유사한 성능을 보였습니다. 이는 모델 크기를 증가시키지 않고 더 나은 성능을 달성한 것을 의미합니다.



### Addressing and Visualizing Misalignments in Human Task-Solving Trajectories (https://arxiv.org/abs/2409.14191)
- **What's New**: 이 연구는 AI 모델 훈련의 기초가 되는 경로 데이터가 인간의 의도와 크게 일치하지 않음을 설명하고, 이를 해결하기 위한 시각화 도구와 휴리스틱 알고리즘을 제안합니다.

- **Technical Details**: 연구에서는 O2ARC에서 수집된 경로 데이터의 세부 구조를 분석하며, 각 경로는 상태-행동(sequence of state-action) 형식으로 정의됩니다. 연구팀은 인간의 의도와 경로 데이터 간의 불일치를 체계적으로 탐지하고 분류하는 방법을 제안합니다.

- **Performance Highlights**: 이 연구의 결과는 경로 데이터와 인간의 의도 간의 불일치를 해결함으로써 AI 모델의 훈련효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Democratising Artificial Intelligence for Pandemic Preparedness and Global Governance in Latin American and Caribbean Countries (https://arxiv.org/abs/2409.14181)
- **What's New**: 본 논문에서는 Global South의 16개국에서 시행되는 AI4PEP 이니셔티브를 소개하며, 감염병 예방과 대응을 위한 공정하고 책임 있는 AI 솔루션을 강화하는 것을 목표로 하고 있습니다.

- **Technical Details**: AI4PEP 네트워크는 LAC(라틴 아메리카 및 카리브해) 내에서 AI 거버넌스와 생명공학의 관계를 논의하며, 감염병 퇴치에 필요한 AI 기술의 폭넓은 활용 가능성을 강조합니다.

- **Performance Highlights**: 저소득 및 중간소득 국가의 공공 건강 시스템을 개선하고, 감염병 발생에 대한 준비와 대응 능력을 향상시키기 위해 AI 기술의 활용을 촉진하는 기회를 창출합니다.



### An Evolutionary Algorithm For the Vehicle Routing Problem with Drones with Interceptions (https://arxiv.org/abs/2409.14173)
- **What's New**: 이번 논문에서는 트럭과 드론을 활용하여 마지막 단계(last-mile) 배송 문제를 해결하는 새로운 접근 방식을 제안합니다. 특히 드론이 배송 중 트럭을 가로채는 상황을 포함한 차량 경로 최적화 문제인 VRPDi(Vehicle Routing Problem with Drones with Interception)를 다룹니다.

- **Technical Details**: 이 연구에서는 진화 알고리즘(evolutionary algorithm)을 기반으로 VRPDi 문제를 해결하는 방법을 제시합니다. 이 문제는 여러 쌍의 트럭과 드론을 스케줄링하며, 드론은 고객 위치에서 트럭을 만나거나 배송 후 트럭과 재회하는 방식으로 최적의 경로를 찾습니다. 알고리즘은 Bouman et al. (2015)의 Traveling Salesman Problem with Drones (TSPD) 데이터셋에서 실행되었습니다.

- **Performance Highlights**: 알고리즘의 성능은 VRPDi의 결과를 기존 VRP 결과와 비교하여 총 배송 시간이 39%에서 60% 향상된 것으로 나타났습니다. 알고리즘은 50개 및 100개 노드 문제를 합리적인 시간 내에 효과적으로 해결했고, Dillon et al. (2023) 및 Ernst (2024)의 알고리즘보다 우수한 결과를 도출했습니다.



### Will Large Language Models be a Panacea to Autonomous Driving? (https://arxiv.org/abs/2409.14165)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 자율주행(AD) 시스템에서 어떻게 활용될 수 있는지를 분석하며, LLM의 최적화 전략을 모듈화 및 엔드 투 엔드 접근법에서 탐색합니다.

- **Technical Details**: 자율주행 기술은 모듈화(modularization)와 엔드 투 엔드(end-to-end)로 나뉘며, 모듈화는 주행 작업을 감지(perception), 예측(prediction), 계획(planning), 제어(control) 모듈로 분해하여 각기 따로 훈련합니다. 반면, 엔드 투 엔드는 센서 데이터에서 제어 신호로 직접 매핑하는 단일 모델을 사용합니다. 두 접근법 모두 훈련 목표의 일관성 부족 문제와 복잡한 도시 교통 상황에서의 예측 불가능한 사건 처리에서 어려움을 겪고 있습니다.

- **Performance Highlights**: LLM들은 강력한 추론 능력과 방대한 지식을 바탕으로 AD 시스템의 이해도 및 의사결정 능력을 향상시킬 수 있는 잠재력을 지니고 있습니다. 하지만 LLM 기반 인공지능이 고급 AD 구현의 열쇠가 될 수 있을지, 그리고 AD 기술 개발을 촉진하는 데 있어서 LLM이 직면할 잠재적 한계와 도전과제를 논의합니다.



### FineMolTex: Towards Fine-grained Molecular Graph-Text Pre-training (https://arxiv.org/abs/2409.14106)
- **What's New**: 본 연구에서는 FineMolTex라는 새로운 Fine-grained Molecular graph-Text 사전 학습 프레임워크를 제안하여 전체 분자 그래프가 아닌 세부적인 motif(모티프) 수준의 지식을 학습하며, 이를 통해 분자의 특성을 보다 잘 이해할 수 있도록 한다.

- **Technical Details**: FineMolTex는 두 가지 사전 학습 작업으로 구성된다: coarse-grained(코르스 그레인) 매칭을 위한 contrastive alignment task(대조적 정렬 과제)와 masked multi-modal modeling task(마스크 다중 모달 모델링 과제)로, 후자는 서로 다른 모달리티의 정보를 활용하여 마스크 처리된 motifs(모티프)와 단어의 레이블을 예측한다.

- **Performance Highlights**: FineMolTex는 230% 향상을 이루며, 약물을 편집하는 텍스트 기반 작업에서 탁월한 성과를 낸다. 이 모델은 제약 발견 및 촉매 설계와 같은 분야에서 세부적인 지식을 효과적으로 포착할 수 있다.



### Normalized Narrow Jump To Conclusions: Normalized Narrow Shortcuts for Parameter Efficient Early Exit Transformer Prediction (https://arxiv.org/abs/2409.14091)
- **What's New**: 이 논문에서는 Narrow Jump to Conclusions (NJTC)와 Normalized Narrow Jump to Conclusions (N-NJTC)라는 두 가지 새로운 방법을 제안하여, 대규모 Transformer 모델에서의 shortcut을 보다 파라미터 효율적으로 구현할 수 있음을 보여줍니다.

- **Technical Details**: NJTC와 N-NJTC는 이전의 JTC (Jump-To-Conclusions) 방법에 비해 97% 이상의 파라미터 수 감소를 달성합니다. 이들은 두 개의 선형 신경망 매트릭스 A와 B를 사용하여 초기 블록의 출력으로부터 또 다른 블록의 출력을 근사화합니다. 이 최적화 과정은 경량화된 파라미터 도출을 통해 실제 출력에 대한 신뢰도를 신속하게 평가할 수 있도록 돕습니다.

- **Performance Highlights**: N-NJTC는 Identity shortcut보다 초기 단계에서 신뢰성 있게 더 나은 성능을 보이며, GPT-2-XL, Phi3-Mini, Llama2-7B와 같은 여러 Transformer 모델의 모든 블록 레벨에서도 안정적인 정확도를 제공합니다.



### The use of GPT-4o and Other Large Language Models for the Improvement and Design of Self-Assessment Scales for Measurement of Interpersonal Communication Skills (https://arxiv.org/abs/2409.14050)
Comments:
          41 pages

- **What's New**: OpenAI의 ChatGPT (GPT-4 및 GPT-4o)를 포함한 여러 대형 언어 모델(LLMs)이 과학 연구의 여러 단계에서 효과적으로 사용될 수 있다는 새로운 통찰을 제시합니다. 이 모델들은 인간의 평균 수준에 가까운 또는 그 이상의 성과를 보이고 있으며, 인간 심리학 및 의사소통에 대한 정보 처리 능력이 주목받고 있습니다.

- **Technical Details**: 이 논문은 LLMs가 대인 의사소통 능력 측정을 위한 자기평가 척도의 설계에서 각 항목의 선택 및 개선, 내용 타당성 평가와 같은 전형적인 작업에 어떻게 활용될 수 있는지를 보여줍니다. 또한, 자동 항목 생성(automated item generation) 가능성도 언급됩니다.

- **Performance Highlights**: LLMs의 사용으로 대인 의사소통 능력 자기평가 척도의 평가, 설계 및 개선 과정에서의 잠재적 이점이 강조되며, 사례 연구와 함께 유용한 LLM 프롬프트(prompts)가 제시됩니다.



### OAEI-LLM: A Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching (https://arxiv.org/abs/2409.14038)
Comments:
          4 pages, 1 figure

- **What's New**: 이번 연구에서는 LLM(hallucinations)의 발생이 Ontology Matching(OM) 작업에서 중요한 문제임을 제기하고, 이를 해결하기 위한 OAEI-LLM 데이터셋을 제안합니다. 이 데이터셋은 OM 작업에서 LLM의 환각 현상을 평가하기 위한 기준을 제공합니다.

- **Technical Details**: OAEI-LLM 데이터셋은 기존 OAEI 데이터셋의 확장으로, LLM의 OM 작업에서의 환각 유형을 분류하고 이를 바탕으로 LLM의 정답률을 평가합니다. 새로운 스키마 확장을 통해 LLM이 생성한 결과와 인간이 라벨링한 결과를 비교하고, 환각의 발생 정도를 측정합니다.

- **Performance Highlights**: LLM은 OM 작업에서 높은 성능을 보일 수 있지만, 환각 현상으로 인해 낮은 precision(정밀도) 및 recall(재현율) 문제를 초래할 수 있습니다. OAEI-LLM 데이터셋을 통해 LLM의 환각 현상에 대한 이해를 높이고, 향후 OM 개선 연구에 기여할 것으로 예상됩니다.



### Drift to Remember (https://arxiv.org/abs/2409.13997)
- **What's New**: 이 연구에서는 DriftNet이라는 네트워크를 도입하여 평생 학습(lifelong learning) 문제를 해결하고, 생물학적 뇌의 학습 능력을 모방하려고 합니다. DriftNet은 신경망의 가중치를 지속적으로 변화시켜 다양한 로컬 미니마(local minima)를 탐색합니다.

- **Technical Details**: DriftNet은 탐색, 인코딩, 검색의 세 단계로 구성됩니다. 외부 노이즈(예: 배치 샘플링, 드롭아웃, 그래디언트 노이즈)로 인해 가중치가 드리프트(drift)하며, 이로 인해 손실 경관(loss landscape)에서 다양한 로컬 미니마를 발견합니다. 이러한 미니마는 작업별 그룹으로 조직되고, 새로운 작업과 관련된 미니마는 기존 그룹에 추가됩니다.

- **Performance Highlights**: 이미지 분류 및 자연어 처리(NLP) 작업에서 DriftNet은 기존 모델들에 비해 우수한 성능을 보여줍니다. 예를 들어, CIFAR-10 데이터셋에서 DriftNet은 80.19%의 정확도를 기록하며, 이는 Stable baseline의 19.18%에 비해 상당히 높은 성과입니다. 또한, NLP에서는 DriftNet이 70.37%의 정확도를 달성하였고, 이는 18.29%인 Stable baseline을 크게 초월하는 결과입니다.



### PureDiffusion: Using Backdoor to Counter Backdoor in Generative Diffusion Models (https://arxiv.org/abs/2409.13945)
- **What's New**: 본 논문에서는 PureDiffusion이라는 새로운 백도어 방어 프레임워크를 소개하며, 이것이 백도어 공격을 효율적으로 탐지하고 역전할 수 있는 방법을 제시합니다. 이는 기존의 방어 방법보다 월등한 성능을 보입니다.

- **Technical Details**: PureDiffusion은 다단계 트리거 역전 방법을 기반으로 하여, 각 디노이징(denoising) 단계에서의 트리거 관련 분포 변화를 추정하는 첫 번째 솔루션을 제공합니다. 이를 통해 우리는 gradient descent를 활용하여 여러 타임스텝을 통해 트리거를 학습할 수 있습니다.

- **Performance Highlights**: 다양한 트리거-타겟 쌍에 대해 실시한 전반적인 실험 결과, PureDiffusion은 기존 방어 방법들에 비해 트리거 충실도(fidelity)와 백도어 성공률(backdoor success rate)에서 큰 격차를 보였습니다. 특히, 특정 경우에서는 PureDiffusion에 의해 역전된 트리거가 원래 트리거보다 더 높은 공격 성공률을 달성했습니다.



### Simple Unsupervised Knowledge Distillation With Space Similarity (https://arxiv.org/abs/2409.13939)
- **What's New**: 이 논문에서는 Self-supervised learning (SSL)이 소형 네트워크에 쉽게 확장되지 않는 문제를 해결하기 위해, 기존의 필수적인 sample 간의 관계를 수작업으로 설정하는 대신, 교사의 embedding manifold을 직접 모델링하도록 학생을 유도하는 새로운 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 	extbf{space similarity}라는 손실 구성 요소를 이용하여 학생의 각 feature 공간의 차원이 교사의 해당 차원과 유사하도록 유도합니다. 이로 인해 교사의 embedding manifold과의 정렬을 통해 필수적인 정보 보존을 달성합니다.

- **Performance Highlights**: 다양한 UKD 벤치마크에서 제안된 접근 방식이 우수한 성능을 보여주며, 기존 방식에 비해 state of the art 결과를 보고합니다.



### Failures in Perspective-taking of Multimodal AI Systems (https://arxiv.org/abs/2409.13929)
- **What's New**: 이 연구는 다중 모달 AI 시스템에서의 공간 표현에 대한 이전 연구를 확장합니다. 인공지능 모델들이 이미지에서 공간 정보를 이해하는 데는 개선이 있지만, 인간의 공간 인지에서 사용하는 아날로그 표현과는 다르게 명제적 표현에 뿌리를 두고 있다는 점을 강조합니다.

- **Technical Details**: 연구는 GPT-4o 의 시점 취득(perceptive-taking) 능력을 평가하기 위해 인지 및 발달 과학의 기술을 적용했습니다. 시점 취득은 인간 공간 추론의 중요한 요소로, AI 시스템의 일상적 보조자로서 기능하기 위해 강력한 시점 취득 능력이 필요합니다. 연구에서는 Level 1과 Level 2 시점 취득을 구분하여 다중 모달 모델의 능력을 평가하는 작은 벤치마크를 제안합니다.

- **Performance Highlights**: GPT-4o는 8개의 이미지 각도 중 6개에서 거의 완벽한 정확도로 수행했으나, 특정 각도(0° 및 315°)에서 성과가 저조했습니다. 수사적 프롬프트(chain-of-thought prompting)를 사용할 때 특정 각도(180°)에서 성능이 개선되었으나, 90°와 180° 사이의 중간 회전에서는 성능 향상이 없었습니다.



### SpaceBlender: Creating Context-Rich Collaborative Spaces Through Generative 3D Scene Blending (https://arxiv.org/abs/2409.13926)
- **What's New**: 이 논문에서는 사용자 정의 3D 가상 공간을 생성하기 위한 새로운 파이프라인인 SpaceBlender를 소개합니다. 이 파이프라인은 사용자의 물리적 환경을 통합하여 VR 텔레프레즌스(VR telepresence)를 지원하는 환경을 생성하는 데 중점을 두고 있습니다.

- **Technical Details**: SpaceBlender는 사용자가 제공한 2D 이미지를 깊이 추정(depth estimation), 메시 정합(mesh alignment), 그리고 기하학적 우선 사항과 적응형 텍스트 프롬프트에 의해 안내된 확산 기반(space completion) 프로세스를 통해 컨텍스트가 풍부한 3D 환경으로 변환합니다. 이 과정은 사용자가 제공한 여러 개의 환경을 결합하고 최적화하여 VR 환경에서의 사용성을 높입니다.

- **Performance Highlights**: 초기 사용자 연구에서는 SpaceBlender가 생성한 환경이 기존 저비용 저해상도 방과 텍스트로 생성된 방보다 사용자가 더욱 익숙함을 느끼게 했습니다. 그러나 사용자는 생성된 환경의 복잡성이 작업 집중에 방해가 될 수 있다고 언급하였으며, 이후 시나리오의 시각적 품질 및 리얼리즘 향상이 필요하다는 피드백을 주었습니다.



### Measuring Error Alignment for Decision-Making Systems (https://arxiv.org/abs/2409.13919)
- **What's New**: 이번 연구에서는 AI 시스템과 인간의 의사결정 방식을 비교하기 위해 두 가지 새로운 행동 정렬(metric) 지표인 Misclassification Agreement (MA)와 Class-Level Error Similarity (CLES)를 제안합니다.

- **Technical Details**: MA는 두 시스템이 동일한 인스턴스에 대해 발생시키는 오류 간의 유사성을 측정하며, CLES는 각 클래스별 오류의 분포를 비교합니다. 이러한 메트릭들은 기존의 Representational Alignment (RA) 방법과 상호 보완적인 정보를 제공합니다.

- **Performance Highlights**: 실험 결과, MA는 Error Consistency (EC)와 비교하여 더 다양한 정보를 포착하며, CLES는 데이터 접근의 제한을 덜 받는 방식으로 널리 적용 가능함을 보여주었습니다.



### Nonlinear Inverse Design of Mechanical Multi-Material Metamaterials Enabled by Video Denoising Diffusion and Structure Identifier (https://arxiv.org/abs/2409.13908)
Comments:
          26 pages, 15 figures

- **What's New**: 본 논문에서는 비디오 확산 모델(video diffusion model)을 활용한 비선형 다중 재료 설계를 위한 새로운 프레임워크를 제안합니다. 이를 통해 기존의 전통적인 역설계 접근 방식의 한계를 극복하고, 다양한 구조적 구성에 대한 비선형 물질 거동을 매핑할 수 있게 됩니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 구성 요소로 이루어져 있습니다: (1) 비디오 확산 모델을 이용한 필드 생성기와 (2) 다수의 UNet 모델을 활용한 구조 식별기입니다. 이 방식은 다중 재료, 소성(plasticity) 및 대변형을 포함하여 물질의 비선형 기계적 거동을 제어하고, 복잡한 격자 구조를 설계하는 데 유용합니다.

- **Performance Highlights**: 이 전략은 실제 응용 사례에서 관찰되는 고도의 비선형 기계적 거동에 대한 조절을 향상시키는 것을 가능하게 하여, 새롭게 조정된 기계적 특성을 지닌 차세대 메타물질을 생성할 수 있는 유망한 해결책을 제공합니다.



### CI-Bench: Benchmarking Contextual Integrity of AI Assistants on Synthetic Data (https://arxiv.org/abs/2409.13903)
- **What's New**: 본 논문은 AI 보조기가 개인 정보를 보호하는 능력을 평가하기 위한 종합적인 합성 벤치마크인 CI-Bench를 소개합니다. 기존의 AI 모델 평가 체계는 일반적인 개인 정보 유출 문제를 다루지 않았기에, 본 벤치마크는 다양한 상황적 맥락에서 정보 흐름을 체계적으로 평가할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: CI-Bench는 Contextual Integrity(정보의 적정성) 프레임워크를 활용하여 AI 보조기의 정보 흐름을 평가합니다. 이 벤치마크에는 대화 및 이메일을 포함하여 현실적인 자연 통신을 생성하는 다단계 합성 데이터 파이프라인이 포함됩니다. 총 8개의 도메인에서 44,000개의 테스트 샘플을 생성하였으며, AI 보조기가 정보 흐름의 적절성을 판단할 수 있는 능력을 상세히 평가합니다.

- **Performance Highlights**: 실험 결과, 현재의 AI 모델들은 적절성 판단 작업에서 개선이 필요함을 보여줍니다. 작은 모델들은 맥락 이해에서 어려움을 겪고 있으며, 잘 정의된 데이터 흐름 규칙 및 지침은 정보 교환 평가의 정확성을 크게 향상시킵니다. CI-Bench는 개인 데이터 전송을 평가하기 위한 가장 종합적인 벤치마크로, 향후 모델 개발 및 시스템 설계에 기여할 수 있습니다.



### Generative AI Carries Non-Democratic Biases and Stereotypes: Representation of Women, Black Individuals, Age Groups, and People with Disability in AI-Generated Images across Occupations (https://arxiv.org/abs/2409.13869)
- **What's New**: AI 거버넌스(AI governance)와 AI 개발에서의 윤리(Ethics)가 중대한 문제로 떠오르며, 기술 기업, 정부, 연구자들 사이에서 AI가 우리의 민주주의에 미치는 잠재적 위험에 대한 활발한 논의가 이루어지고 있습니다. 이 논문은 생성적 AI(Generative AI)가 평등이 필요한 집단들을 어떻게 포함하거나 배제하는지를 조명합니다.

- **Technical Details**: 연구 결과는 생성적 AI가 성별(Gender), 인종(Race), 나이(Age), 그리고 가시적 장애(Visible Disability)에 관해 균등하게 포함되지 않음을 보여줍니다. 이는 AI 모델이 특정 집단에 대해 편향된 데이터를 학습함으로써 공정성을 결여하고 있음을 시사합니다.

- **Performance Highlights**: 이 연구의 주요 발견은 생성적 AI의 출력이 평등성(Equity)에 대한 고민 없이 설계되었음을 드러내며, 이는 AI 시스템의 설계와 데이터 수집이 더 포괄적이고 공정해야 함을 강조합니다.



### A Personalised 3D+t Mesh Generative Model for Unveiling Normal Heart Dynamics (https://arxiv.org/abs/2409.13825)
- **What's New**: 이 논문은 심장 형태와 운동 패턴을 이해하기 위해 개발된 새로운 조건부 생성 모델인 MeshHeart를 소개합니다. 이 모델은 임상 요인(예: 나이, 성별, 체중 및 신장)을 고려하여 3D+t 심장 메쉬 시퀀스를 생성할 수 있습니다.

- **Technical Details**: MeshHeart는 심장 메쉬를 잠재 공간(latent space)에 나타내기 위해 기하학적 인코더(geometric encoder)를 사용하며, 잠재 표현의 운동 역학을 모델링하기 위해 시간 변환기(temporal Transformer)를 사용합니다. 이 모델은 고차원 및 복잡한 시공간(spatio-temporal) 메쉬 데이터를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: MeshHeart는 38,309명의 대규모 데이터셋을 활용하여 심장 메쉬 시퀀스 재구성 및 생성에서 높은 성능을 나타내었습니다. Latent space에서 정의된 특징은 심장 질병 분류를 위한 높은 판별력을 가지며, latent delta는 임상 표현형과 강한 상관관계를 나타냅니다.



### Simulaci\'on de la distribuci\'on de alimento en el cultivo de camar\'on (https://arxiv.org/abs/2409.13759)
- **What's New**: 이 연구는 새우 양식의 4가지 사료 분배 사례에 대한 실험을 소개합니다. 자동 급이기(location of the automatic feeders)의 위치에 따라 조정된 사료 분배가 주요 초점입니다.

- **Technical Details**: 첫 번째 단계에서는 3가지 사료 분배 사례의 시뮬레이션(simulation)이 실제 진행 상황에 성공적으로 조정되었습니다. 두 번째 단계에서는 사료의 양, 생물량 밀도(biomass density), 그리고 사료 분배(distribution) 방법을 기반으로 16개의 구성(configuration)에서 실험했습니다. 유전자 알고리즘(genetic algorithms)과 퍼지 로직(fuzzy logic)을 사용하여 의사 결정(decision-making)을 위한 평가 기법(agent evaluation technique)이 도입되었습니다.

- **Performance Highlights**: 총 양식 시간(total culture time)이 22주에서 14주로 감소하는 결과를 도출했습니다.



### Increasing the Value of Information During Planning in Uncertain Environments (https://arxiv.org/abs/2409.13754)
Comments:
          Honors thesis submitted to Computer Science Department at Oberlin College. this https URL

- **What's New**: 이 연구에서는 POMDP(Partially Observable Markov Decision Process) 문제를 해결하기 위해 정보 수집 행동의 가치를 새롭게 모델링함으로써 기존의 온라인 계획 알고리즘의 한계를 극복하는 새로운 알고리즘을 제안합니다.

- **Technical Details**: 제안한 방법은 POMCP(Partially Observable Monte Carlo Planning) 알고리즘의 UCB1(Upper Confidence Bound 1) 휴리스틱에 엔트로피(Entropy)를 추가하여 정보 수집 행동을 강조합니다. 이 방법은 기존 솔루션의 anytime 성격을 유지하면서 계산적으로도 비용이 적게 드는 방안으로 성과를 도출합니다.

- **Performance Highlights**: 이 새로운 알고리즘은 hallway 문제에서 POMCP보다 유의미하게 향상된 성능을 보이며, 정보 수집 행동의 가치를 보다 잘 반영하여 최적 정책을 강화하는 데 기여합니다.



### VisScience: An Extensive Benchmark for Evaluating K12 Educational Multi-modal Scientific Reasoning (https://arxiv.org/abs/2409.13730)
Comments:
          89 pages, 70 figures

- **What's New**: 이번 논문에서는 다양한 과학 분야에서 다중 모달 대형 언어 모델(MLLMs)의 성능을 평가하기 위해 VisScience라는 새로운 벤치마크를 제시합니다. 이는 수학, 물리학, 화학 세 가지 과목을 아우르며, K12 교육을 기반으로 한 3,000개의 질문을 포함합니다.

- **Technical Details**: VisScience 벤치마크는 초등학교부터 고등학교까지의 21개 주제를 포함하여 각 과목마다 1,000개의 질문을 포함하고 있으며, 질문은 5개의 난이도 수준으로 구분됩니다. 이 연구에서는 25개의 대표적인 MLLMs의 과학적 추론 능력을 평가합니다.

- **Performance Highlights**: 실험 결과에 따르면, 폐쇄형(Closed-source) MLLMs가 개방형(Open-source) 모델보다 일반적으로 뛰어난 성능을 보였습니다. Claude3.5-Sonnet 모델은 수학 53.4%의 정확도를 기록하며 가장 높은 성능을 나타냈고, GPT-4o는 물리학 38.2%, Gemini-1.5-Pro는 화학 47.0%의 정확도를 기록했습니다.



### A Preliminary Study of o1 in Medicine: Are We Closer to an AI Doctor? (https://arxiv.org/abs/2409.15277)
Comments:
          The first four authors contributed equally, project page available at this https URL

- **What's New**: 본 논문에서는 OpenAI의 새로운 모델 o1이 초기화된 Chain-of-Thought 기법을 사용하여 강력한 언어 모델(Large Language Models, LLMs)의 성능을 강화했음을 소개합니다. 특히, o1의 의료 분야에서의 적용 가능성과 성능을 평가하는 데 중점을 두고 있습니다.

- **Technical Details**: o1 모델은 37개의 의료 데이터셋을 바탕으로 6가지 과제를 평가하였으며, 두 개의 도전적인 질문-답변(QA) 데이터를 새로 작성하였습니다. 평가 항목으로는 이해(Understanding), 추론(Reasoning), 다국어 능력(Multilinguality)을 사용하였습니다.

- **Performance Highlights**: o1은 19개의 데이터셋과 두 개의 새로운 복잡한 QA 시나리오에서 GPT-4 보다 평균 6.2% 및 6.6% 더 정확한 성능을 보였습니다. 그러나 모델의 할루시네이션(Hallucination), 일관성 없는 다국어 능력, 평가 지표 간의 차이 등 여러 약점도 발견되었습니다.



### OmniBench: Towards The Future of Universal Omni-Language Models (https://arxiv.org/abs/2409.15272)
- **What's New**: OmniBench는 여러 모달리티(visual, acoustic, textual) 간의 상호작용을 평가하고 모델의 이해 및 추론 능력을 측정하는 새로운 벤치마크입니다. 이 벤치마크는 모든 모달리티 간의 통합된 이해를 요구하여 기존의 한계를 극복하고 있습니다.

- **Technical Details**: 오미벤치(OmniBench)는 미리 훈련된 초대형 언어 모델(MLLMs)의 tri-modal(3중 모달리티) 처리 능력을 테스트하기 위한 포괄적인 도구입니다. OLMs(omni-language models)는 이러한 능력을 갖춘 모델로 정의됩니다. OLMs는 high-quality human annotations에 의존하여 정확한 응답을 제공하는 데 필요한 모든 모달리티의 통합된 이해를 요구합니다.

- **Performance Highlights**: 대부분의 OLMs는 tri-modal 상황에서 지시 수행 및 추론 능력에 한계를 보이는 것으로 나타났습니다. 기존의 MLLM들은 이미지 또는 오디오와 함께 제공되었을 때 명확한 지시를 따르기 어려운 경우가 많으며, 대체 텍스트 표현 사용 시에도 정확도가 50% 미만으로 낮은 성능을 기록했습니다.



### Style over Substance: Failure Modes of LLM Judges in Alignment Benchmarking (https://arxiv.org/abs/2409.15268)
- **What's New**: 본 연구에서는 LLM-judge(대형 언어 모델 심사자)의 선호도가 다른 구체적인 정렬 메트릭에 어떻게 적용되는지에 대한 질문을 탐구합니다. 이를 통해 현재 널리 사용되고 있는 LLM-judge 벤치마크의 신뢰성과 그 한계를 조명합니다.

- **Technical Details**: 우리는 SOS-Bench라는 새로운 정렬 벤치마크를 도입하고, LLM-judge가 사실성과 안전성 같은 중요한 정렬 요소보다 스타일적 선호를 우선시하는 편향을 가지고 있다는 점을 발견했습니다. 또한, LLM-judge 평가 방식의 주요 요소를 체계적으로 분석할 수 있는 프레임워크를 구축하였습니다.

- **Performance Highlights**: 연구 결과, 대형 언어 모델 심사자에 의한 평가가 공공의 안전, 세계 지식 및 지시 수행률과 실질적으로 상관관계가 없음을 확인했습니다. SFT(지도 학습 미세 조정) 단계가 PO(선호 최적화) 단계보다 정렬에 더 큰 영향을 미치며, 데이터 스케일링과 프롬프트 다양성이 가장 중요한 예측 변수임을 밝혀냈습니다.



### The Palomar twilight survey of 'Ayl\'o'chaxnim, Atiras, and comets (https://arxiv.org/abs/2409.15263)
Comments:
          26 pages, 13 figures, 4 tables, accepted for publication in Icarus

- **What's New**: 이 연구는 Palomar 48인치 망원경(P48)과 Zwicky Transient Facility(ZTF) 카메라를 이용하여 2019년 9월 20일부터 2022년 3월 7일까지 저녁과 아침 천문 황혼에서 46,000회 이상의 관측을 수행한 결과를 제공합니다. 주요 발견으로는 'Ayló'chaxnim 소행성과 4개의 Atira 소행성(2020 OV1, 2021 BS1, 2021 PB2, 2021 VR3) 및 6개의 장주기 혜성과 2개의 단주기 혜성이 포함됩니다.

- **Technical Details**: 이 논문에서 설명하는 관측 결과는 저녁 및 아침 천문 황혼 동안 31도에서 66도 사이의 태양 한계 magnitude r-band 18.1에서 20.9 사이로 진행되었습니다. 이 연구는 발생하는 계절성과 한계 magnitude의 경미한 변화를 관찰하였으며, 여름 시즌에서 약간의 개선이 있음을 보고합니다. 또한 딥 러닝(Deep Learning) 기반 혜성 탐지 파이프라인을 사용하여 혜성을 탐지했습니다.

- **Performance Highlights**: 이번 조사 결과는 총 11개의 이미 알려진 Atira 소행성과 1개의 Aylo, 3개의 단주기 혜성, 2개의 장주기 혜성 및 1개의 성간 물체를 회복했습니다. Vera Rubin Observatory는 앞으로 1년의 운영 시작과 함께 새로운 황혼 조사도 실시할 예정이며, 이는 지구와 금성의 궤도 내에서 소행성을 발견할 기회를 제공할 것입니다.



### Identification and Localization of Cometary Activity in Solar System Objects with Machine Learning (https://arxiv.org/abs/2409.15261)
Comments:
          25 pages, 9 figures, accepted chapter in Machine Learning for Small Bodies in the Solar System, Valerio Carruba, Evgeny Smirnov, and Dagmara Oszkiewicz, Elsevier, 2024, p. 209-227

- **What's New**: 본 장에서는 지상 및 우주 기반의 광범위한 전천후 관측에서 태양계 객체의 혜성 활동을 식별하고 국소화하기 위해 머신러닝(Machine Learning) 방법의 사용에 대해 논의합니다. 특히, 알려진 및 알려지지 않은 활동적인 태양계 객체를 별과 같은 출처의 존재 속에서 식별하는 데 있어 고전적인 사전 머신러닝 기법의 한계와 머신러닝 기법의 구현을 다루게 됩니다.

- **Technical Details**: 혜성 탐지 및 식별을 위한 고전적 방법과 최근 진전을 이룬 머신러닝 기술의 적용을 다룹니다. 특히, Deep Learning 기술이 혜성을 확장된 출처로 인식하는 데 도움을 주며, Vera C. Rubin Observatory의 Legacy Survey of Space and Time(LSST) 같은 미래 관측에 대한 기술을 논의합니다.

- **Performance Highlights**: 현재 및 차세대 전천후 관측은 태양계의 소행성과 혜성을 발견할 수 있는 전례 없는 기회를 제공합니다. 그러나 ZTF 조사와 같은 특정 조사에서는 확장된 객체의 감지가 제한되어 있으며, 이를 해결하기 위한 새로운 머신러닝 기법의 필요성이 강조됩니다.



### S$^2$AG-Vid: Enhancing Multi-Motion Alignment in Video Diffusion Models via Spatial and Syntactic Attention-Based Guidanc (https://arxiv.org/abs/2409.15259)
- **What's New**: 본 논문에서는 다중 객체가 포함된 텍스트-비디오(T2V) 생성에서의 객체와 동작 정렬 문제를 해결하기 위해 S$^2$AG-Vid라는 새로운 추론 단계 최적화 방법을 제안합니다. 이 방법은 훈련 없이 적용될 수 있으며, 다양한 객체가 특정 동작과 더 잘 정렬되도록 돕습니다.

- **Technical Details**: S$^2$AG-Vid는 첫째, 초기 노이즈 제거 과정에서 공간 위치 기반 교차 주의(cross-attention, CA) 제약 조건을 적용해 여러 개의 명사가 올바른 주제 영역에 주의를 기울일 수 있도록 합니다. 둘째, 동작-주제 결합을 강화하기 위해 문법 안내 대조 제약(syntax-guided contrastive constraint)을 실시하며, 이는 동사 CA 맵과 해당 명사 CA 맵 간의 상관관계를 개선하는 것을 목표로 합니다.

- **Performance Highlights**: 정성적 및 정량적 평가에서 S$^2$AG-Vid는 기존 모델에 비해 매우 높은 품질의 비디오를 생성하며, 객체와 동작 간의 일관성을 크게 향상시키는 데 성공하였습니다.



### Behavioral Bias of Vision-Language Models: A Behavioral Finance View (https://arxiv.org/abs/2409.15256)
Comments:
          ICML 2024 Workshop on Large Language Models and Cognition

- **What's New**: 이번 연구는 대형 비전-언어 모델(LVLM)의 행동 편향을 행동 재무학의 관점에서 분석한 최초의 연구로, LVLM이 인간과 유사한 결정의 합리성을 발휘하는지 혹은 인간과 유사한 판단 및 결정 편향에 영향을 받는지를 조사합니다.

- **Technical Details**: 연구는 S&P 500 기업의 주식 역사 및 분기별 주당순이익(EPS) 보고서를 포함하는 멀티모달 데이터셋인 DynoStock을 체계적으로 축적하고, recency bias(최근 편향) 및 authority bias(권위 편향)에 대한 프롬프트 템플릿을 설계하여 LVLM의 편향 영향을 평가하는 새로운 평가 프레임워크를 제안합니다.

- **Performance Highlights**: 연구 결과, LLaVA-NeXT, MobileVLM-V2, Mini-Gemini, MiniCPM-Llama3-V 2.5 및 Phi-3-vision-128k와 같은 최근 공개 소스 LVLM들은 두 가지 행동 편향에 심각한 영향을 받는 것으로 나타났습니다. 반면, 독점 모델인 GPT-4o는 편향의 영향에서 거의 영향을 받지 않았습니다.



### Archon: An Architecture Search Framework for Inference-Time Techniques (https://arxiv.org/abs/2409.15254)
- **What's New**: 최근의 조사에 따르면, Archon이라는 자동화된 프레임워크가 LLM과 추론 시간 기술들을 결합하여 성능을 향상시키는 데 효과적임을 입증하였습니다. Archon은 다양한 추론 시간 아키텍처를 설계하는 데 유용하며, 하이퍼파라미터 최적화를 통해 최적의 아키텍처를 도출합니다.

- **Technical Details**: Archon 프레임워크는 generation ensembling, multi-sampling, ranking, fusion, critiquing, verification, 및 unit testing과 같은 방법을 포함한 확장 가능한 디자인 공간을 정의합니다. 자동화된 Inference-Time Architecture Search (ITAS) 알고리즘을 통해 LLM과 추론 컴퓨팅 예산에 따라 최적화된 아키텍처를 출력합니다. 또한, Bayesian optimization을 사용하여 하이퍼파라미터 공간을 효과적이고 효율적으로 검색합니다.

- **Performance Highlights**: Archon 아키텍처는 MT-Bench, Arena-Hard-Auto, AlpacaEval 2.0, MixEval, MixEval Hard, MATH, 및 CodeContests와 같은 다양한 벤치마크에서 우수한 성능을 보였습니다. Archon이 설계한 아키텍처는 GPT-4o 및 Claude 3.5 Sonnet 모델보다 평균 14.1 포인트, 오픈 소스 모델과의 비교에서 평균 10.3 포인트의 성능 향상을 기록했습니다.



### Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping (https://arxiv.org/abs/2409.15241)
Comments:
          12 pages

- **What's New**: 최근 Generative AI의 인기로 인해 Large Language Models (LLMs) 훈련에서 통신 오버헤드가 중요한 문제로 부각되고 있습니다. 이 문제를 해결하기 위해, 'Domino'라는 새로운 접근 방식을 제안하며, 이는 통신을 계산 뒤로 숨기고, 데이터 의존성을 분해하여 독립적인 작은 단위로 훈련하는 방식을 사용합니다.

- **Technical Details**: Domino는 모델 훈련의 데이터 의존성을 독립적인 조각으로 분해하고, 이를 파이프라인 방식으로 훈련하여 통신과 계산의 중첩을 제공합니다. 이 접근법은 전통적인 Tensor Parallelism (TP)에서의 통신 오버헤드를 줄이며, 단일 노드와 다중 노드 환경 모두에서 효과적으로 작동합니다. Domino는 이전의 GeMM+NCCL 융합 솔루션보다 넓은 범위의 계산 및 통신 중첩을 제공합니다.

- **Performance Highlights**: Nvidia DGX-H100 하드웨어에서의 광범위한 벤치마크 결과에 따르면, Domino는 Megatron-LM과 비교하여 최대 1.3배의 속도 향상을 달성했습니다. 이 결과는 단일 노드와 다중 노드 환경 모두에서 유효합니다. Domino는 앞으로 오픈 소스 프로젝트로도 제공될 예정입니다.



### MemBench: Towards Real-world Evaluation of Memory-Augmented Dialogue Systems (https://arxiv.org/abs/2409.15240)
Comments:
          In progress

- **What's New**: 이 논문은 기존의 대화 시스템(DS) 평가 방식의 한계를 극복하기 위해 새로운 메모리 벤치마크인 MemBench를 제안합니다. 이는 인지 과학 및 심리학 이론에 기반하여 다양한 메모리 회상 패러다임을 포함하는 완전한 평가 방법론을 제공합니다.

- **Technical Details**: MemBench는 인지 과학의 두 단계 이론에 따라 구성된 두 가지 작업(메모리 회수 및 인지/주입)을 포함하고 있으며, 수동적 및 능동적 메모리 회상을 모두 고려합니다. 이 벤치마크는 새로운 점수 평가 방식을 도입하여 생성된 응답의 다양한 측면을 포괄적으로 측정합니다.

- **Performance Highlights**: 실험 결과, 현재의 대화 시스템이 메모리를 도입하여도 인간과의 대화에서 여전히 성능이 부족한 점이 드러났습니다. 특히, 메모리 주입이 감정 지원(ES) 능력과 친밀도에 긍정적인 연관성이 있음을 발견하였습니다.



### A Comprehensive Framework for Evaluating API-oriented Code Generation in Large Language Models (https://arxiv.org/abs/2409.15228)
- **What's New**: 이 논문은 API 지향 코드 생성을 위한 새로운 평가 프레임워크인 AutoAPIEval을 제안합니다. 이 프레임워크는 LLM(대형 언어 모델)의 API 호출 코드 생성 능력을 자동으로 평가하는 데 초점을 맞추고 있습니다.

- **Technical Details**: AutoAPIEval은 API 문서를 제공하는 어떤 라이브러리와도 작동하며, 두 가지 주요 단위 작업인 API 추천 및 코드 예제 생성을 다룹니다. 또한, 각 작업에 대한 평가 메트릭스를 제공합니다: Task 1은 잘못된 API 추천 비율을 평가하며, Task 2는 호출된 API가 없거나 컴파일되지 않거나 실행 불가능한 코드 예제 비율을 평가합니다.

- **Performance Highlights**: 사례 연구에서 ChatGPT는 MagiCoder 및 DeepSeek Coder보다 지침을 잘 따르며 발생하는 오류의 비율이 낮은 것으로 나타났습니다. API 추천에서 추천된 API의 58.1%에서 84.1%는 확인된 라이브러리에 존재하지 않았습니다. 코드 예제 생성에서는 39.4%에서 54.4%가 오류를 포함하며, 그 중 5.4%에서 20.7%는 지정된 API를 생략하고 나머지는 컴파일 또는 실행 실패가 발생했습니다.



### Enhancing Pedestrian Trajectory Prediction with Crowd Trip Information (https://arxiv.org/abs/2409.15224)
- **What's New**: 본 논문은 보행자 궤적 예측을 위한 새로운 접근 방식으로, 군중의 이동 정보를 새로운 모달리티로 도입한 RNTransformer 모델을 제안합니다. 이 모델은 사회적 상호작용 및 도로 환경을 고려하여 보행자의 행동을 보다 정확하게 예측할 수 있도록 설계되었습니다.

- **Technical Details**: RNTransformer는 군중 이동 정보를 활용하여 사회적 상호작용에 대한 글로벌 정보를 캡처하는 일반 모델입니다. 이 모델은 여러 소셜 인지 지역 보행자 궤적 예측 모델과 결합되어 성능을 입증하였으며, Social-LSTM에서 ADE/FDE 지표에서 각각 1.3/2.2%, Social-STGCNN에서 6.5/28.4%, S-Implicit에서 8.6/4.3% 향상을 보여주었습니다. 이를 통해 다양한 데이터세트에서 보행자 궤적 예측의 정확도가 크게 향상되었음을 확인하였습니다.

- **Performance Highlights**: RNTransformer는 다양한 기저 보행자 궤적 예측 모델에 대한 정확도를 개선하는 데 성공했으며, 기본 모델에 비해 보행자 목표를 보다 정확하게 샘플링하는 데 기여했습니다. 본 연구에서 개발된 모델은 다양한 데이터세트에 대해 폭넓은 실험을 통해 검증되었습니다.



### ASTE Transformer Modelling Dependencies in Aspect-Sentiment Triplet Extraction (https://arxiv.org/abs/2409.15202)
Comments:
          The 2024 Conference on Empirical Methods in Natural Language Processing, November 12-16, Miami, Florida 9 pages, appendix, diagrams

- **What's New**: 본 논문에서는 Aspect-Sentiment Triplet Extraction (ASTE)에서의 종속성을 모델링할 수 있는 새로운 접근 방식인 ASTE-Transformer를 제안합니다.

- **Technical Details**: ASTE-Transformer는 세 가지 유형의 transformer-inspired layers로 구성되며, 이는 (1) 표준 transformer layers, (2) aspect-opinion 쌍 생성 layer, (3) triple 생성 layer입니다. 이 구조는 두 개의 문장을 기반으로 aspect와 opinion을 추출하고, 그 종속성을 동시에 고려하여 sentiment polarity를 할당할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 실험 결과, ASTE-Transformer는 기존의 방법들보다 F1 성능이 향상되었으며, pre-training 기술이 모델 성능을 추가적으로 개선시켰습니다.



### Learning from Contrastive Prompts: Automated Optimization and Adaptation (https://arxiv.org/abs/2409.15199)
- **What's New**: LCP( Learning from Contrastive Prompts) 프레임워크는 기존의 프롬프트 최적화 방법의 한계를 극복하고자 하며, 여러 모델 버전과 언어에서 효과적으로 적응 감소를 제공한다.

- **Technical Details**: LCP는 입력 프롬프트의 패턴 분석을 통해 효과적인 프롬프트를 생성하기 위해 대조 학습(contrastive learning) 기법을 활용한다. 주안점은 좋은 프롬프트와 나쁜 프롬프트를 비교하면서 오류 사례를 학습하는 것이다.

- **Performance Highlights**: LCP는 Big-Bench Hard 데이터셋에서 기존 방법들보다 76% 이상의 승률을 기록하며, 특히 알고리즘적 및 단계별 산술 추론 작업에서 효율성을 보였다.



### HOTVCOM: Generating Buzzworthy Comments for Videos (https://arxiv.org/abs/2409.15196)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 중국어 비디오 핫 댓글 생성을 위한 새로운 데이터셋 HotVCom을 구축하였으며, 94,000개의 다양한 비디오와 1억 3천7백만 개의 댓글을 포함하고 있습니다. 또한, 비디오 핫 댓글 생성 프레임워크 ComHeat를 소개합니다.

- **Technical Details**: ComHeat는 시각적, 청각적, 텍스트 데이터를 통합하여 중국 비디오 데이터셋에서 영향력 있는 핫 댓글을 생성합니다. 이 프레임워크는 Supervised Fine-Tuning 기법을 통해 초기 댓글을 생성하고, 강화 학습을 통해 개선합니다. 또한 Tree-of-Thought 접근법을 활용하여 댓글의 품질을 높입니다.

- **Performance Highlights**: ComHeat 프레임워크는 새로 구축된 HotVCom 데이터셋과 기존 데이터셋에서 다른 기준 모델들을 초월하는 성능을 보여줍니다. 새로운 종합 평가 지표를 통해 댓글의 유용성, 관련성, 창의성 및 사용자 참여도를 강조합니다.



### Location is Key: Leveraging Large Language Model for Functional Bug Localization in Verilog (https://arxiv.org/abs/2409.15186)
- **What's New**: 본 논문에서는 Verilog 코드에서 기능 오류를 효율적으로 찾아내기 위한 새로운 오픈소스 LLM 솔루션인 Location-is-Key(LiK)를 제안합니다. 기존의 LLM들은 소프트웨어 언어의 버그 찾기에서 유용하게 사용되어 왔지만, Verilog 코드에서는 이러한 접근이 부족했습니다. LiK는 주어진 코드 조각과 설계 사양만으로도 오류를 높은 정확도로 찾아낼 수 있는 가능성을 보여줍니다.

- **Technical Details**: LiK는 두 단계의 지속적인 사전 학습(continual pre-training), 감독된 미세 조정(supervised fine-tuning), 그리고 사용자 피드백에 의한 강화 학습(reinforcement learning with human feedback)을 통해 훈련되었습니다. 이 모델은 RTLLM 기반의 테스트 세트에서 93.3%의 정확도로 기능 오류를 찾아내었으며, GPT-4의 77.9%를 초월하고 Claude-3.5의 90.8%와 유사한 성능을 보였습니다.

- **Performance Highlights**: LiK는 기능적 버그 찾기에서 93.3%의 정확도를 기록하며, 이 결과는 GPT-3.5의 버그 수정 효율을 크게 향상시켰습니다(기능 패스율이 40.39%에서 58.92%로 증가). 또한, LiK는 다른 기존 방법들과 달리 테스트 벤치나 어설션이 필요하지 않아 보다 간편한 디버깅을 가능하게 합니다.



### Skills Made to Order: Efficient Acquisition of Robot Cooking Skills Guided by Multiple Forms of Internet Data (https://arxiv.org/abs/2409.15172)
Comments:
          6 pages, 5 figures

- **What's New**: 이 연구는 로봇 기술을 인터넷 데이터 소스를 활용하여 다양한 템플릿 로봇 행동 중에서 선택하는 방법을 탐구합니다. 특히 도구 사용과 관련된 접촉이 많은 기술을 익히는 데 있어 도전 과제를 다룹니다.

- **Technical Details**: 연구에서는 세 가지 템플릿 선택 방법을 모색합니다: 대규모 언어 모델(LLMs)을 활용하여 템플릿을 쿼리하는 방법, 사전 훈련된 비디오 인코더의 피처를 사용하여 로봇 실행 비디오와 인류 비디오를 비교하는 방법, 그리고 인터넷 데이터에서 훈련된 옵틱 플로우 인코더를 사용하는 방법입니다. 총 33개의 로봇 행동 템플릿을 사용하며, 각 템플릿은 파라미터화된 궤적 및 힘 수준으로 구성됩니다.

- **Performance Highlights**: 최종적으로 로봇 시스템은 조리 기술을 수행하는 데 79%의 성공률을 기록했습니다. LLM은 비주얼 정보가 부족함에도 불구하고 놀라운 템플릿 선택 능력을 보였으며, 옵틱 플로우 인코딩이 더 많은 데이터를 사용해 훈련된 비디오 인코더보다 현저하게 우수한 성능을 발휘했습니다.



### DeepCloth-ROB$^2_{\text{QS}}$P&P: Towards a Robust Robot Deployment for Quasi-Static Pick-and-Place Cloth-Shaping Neural Controllers (https://arxiv.org/abs/2409.15159)
Comments:
          8 pages main texts, 3 figures, and 3 tables. It is submitted to the 2025 IEEE International Conference on Robotics & Automation (ICRA)

- **What's New**: 본 논문은 시뮬레이션에서 훈련된 비전 기반 데이터 중심의 천 조작 신경 제어기와 실제 세계 간의 성능 차이를 줄이기 위한 새로운 접근 방식을 제안합니다. 특히, Towel-Sim2Real라는 전략을 통해 다양한 직물 유형과 로봇 플랫폼에서의 일반화 가능성을 보여주고 있습니다.

- **Technical Details**: DeepCloth-ROB$^2_{QS}$P&P를 사용하여 시뮬레이션에서 현실로의 전이 전략을 포함하고 있으며, 다층 그립 및 잘못된 그립과 같은 잡기 오류를 완화하는 프로토콜을 수립합니다. 이를 통해 다양한 직물의 모양을 바꾸기 위한 신경 제어기를 비교하는 데 성공했습니다.

- **Performance Highlights**: Franka Emika Panda 및 Universal Robots UR3e와 같은 다양한 로봇 플랫폼에서 시스템의 호환성을 보여주며, 시뮬레이션에서 훈련된 정책을 실제 환경에서 효과적으로 비교할 수 있는 기반을 마련했습니다.



### MAR-DTN: Metal Artifact Reduction using Domain Transformation Network for Radiotherapy Planning (https://arxiv.org/abs/2409.15155)
Comments:
          Accepted in 27th International Conference on Pattern Recognition (ICPR). Mubashara Rehman and Belén Serrano-Antón, both co-first authors of the manuscript

- **What's New**: 본 연구는 기계적 이식물(지르코니아 또는 금속 재료)로 인해 왜곡된 kVCT 이미지를 MVCT 이미지로 변환하여 아티팩트(artifact) 없는 이미지를 생성하는 새로운 심층 학습 기반 접근 방식을 제안합니다.

- **Technical Details**: 이 방법은 UNet 구조를 기반으로 하여 kVCT 이미지를 MVCT 이미지로 변환하는 과정을 체계적으로 수행합니다. 모델은 512x512 픽셀 이미지를 처리하며, 385838 슬라이스의 kVCT 이미지를 사용하여 학습되었습니다. PSNR과 SSIM 계산은 배경을 제외한 관심 영역만을 고려하여 수행됩니다.

- **Performance Highlights**: 제안된 방법은 전체 환자 볼륨에서 PSNR 30.02 dB, 아티팩트가 영향을 미친 영역에서는 27.47 dB를 달성하였습니다. 이는 의미 있는 개선을 나타내며, 방사선 종양학자들이 kVCT 만으로도 MVCT의 통찰력을 얻을 수 있게 해줍니다.



### RMCBench: Benchmarking Large Language Models' Resistance to Malicious Cod (https://arxiv.org/abs/2409.15154)
Comments:
          12 pages, 6 figures, 5 tables, 39th IEEE/ACM International Conference on Automated Software Engineering (ASE '24)

- **What's New**: 이번 논문에서는 RMCBench라는 새로운 벤치마크를 제안하며, 이는 LLMs가 악성 코드 생성을 저항하는 능력을 평가하기 위해 설계된 473개의 프롬프트로 구성되어 있습니다. 기존 연구와는 달리, 이 연구는 LLMs가 악성 코드 생성을 저항하는 능력을 평가하는 데 초점이 맞춰져 있습니다.

- **Technical Details**: RMCBench는 두 가지 시나리오를 사용합니다: 1) text-to-code 시나리오, 2) code-to-code 시나리오. 각 시나리오는 악성코드 생성 관련 프롬프트를 사용하여 LLMs의 저항력을 평가합니다. 텍스트 설명에서 코드를 생성하거나 기존 악성 코드를 번역하거나 완성하는 방식입니다.

- **Performance Highlights**: 11개의 LLMs를 평가한 결과, text-to-code 시나리오에서 평균 거부율이 40.36%였고, code-to-code 시나리오에서는 11.52%로 나타났습니다. ChatGPT-4의 경우 거부율은 35.73%였습니다. LLMs의 저항력은 모델의 파라미터, 유형, 악성 코드 유형 및 입력 컨텍스트 길이에 따라 달라졌습니다.



### COHERENT: Collaboration of Heterogeneous Multi-Robot System with Large Language Models (https://arxiv.org/abs/2409.15146)
Comments:
          7 pages, 5 figures. Submitted to IEEE International Conference on Robotics and Automation (ICRA), 2025

- **What's New**: 본 논문에서는 수많은 이질적인 로봇 시스템(quadrotors, robotic dogs, robotic arms 등) 간의 협업을 위한 새로운 LLM 기반 작업 계획 프레임워크인 COHERENT를 제안합니다.

- **Technical Details**: 이 시스템은 Proposal-Execution-Feedback-Adjustment (PEFA) 메커니즘을 설계하여 개별 로봇의 작업을 분해 및 할당합니다. 중앙 집중식 작업 할당기가 복잡한 작업을 하위 작업으로 분해하고, 각 로봇 실행자는 할당된 하위 작업을 실행하기 위한 실행 가능한 작업을 선택합니다. 작업 실행 후, 피드백을 통해 계획을 조정합니다. 이 과정은 작업이 완료될 때까지 반복됩니다.

- **Performance Highlights**: 실험 결과, COHERENT는 성공률과 실행 효율성 면에서 이전 방법들을 큰 폭으로 초월하는 성과를 보였습니다. 총 100개의 복잡한 장기 작업을 포함하는 이질적인 다중 로봇 작업 계획 벤치마크를 새롭게 생성하였습니다.



### CAMAL: Optimizing LSM-trees via Active Learning (https://arxiv.org/abs/2409.15130)
Comments:
          SIGMOD 2025

- **What's New**: 이번 논문에서는 LSM-tree 구조 최적화를 위해 기계 학습(machine learning, ML)을 활용한 Camal이라는 새로운 접근법을 소개합니다. 이 방법은 LSM-tree 기반 핵-값 저장소의 파라미터를 효율적으로 튜닝하기 위해 활성 학습(active learning)을 처음으로 적용했습니다.

- **Technical Details**: Camal은 다음과 같은 여러 기능을 제공합니다: (1) ML 지원: ML을 통해 LSM-tree의 다양한 파라미터를 효과적으로 조정하며, 비용 모델과 결합하여 훈련 프로세스를 개선합니다. (2) 분리된 활성 학습: 각 파라미터를 개별적으로 조정하여 학습 프로세스를 가속화합니다. (3) 용이한 외삽: 데이터 크기가 추가됨에 따라 모델을 점진적으로 업데이트할 수 있는 메커니즘을 채택합니다. (4) 동적 모드: 변화하는 작업 부하에 따라 LSM-tree를 온라인으로 조정할 수 있습니다.

- **Performance Highlights**: Camal을 RocksDB 시스템에 통합하였을 때, 평균 28%의 성능 향상을 보여주었으며, 최신 RocksDB 디자인 대비 최대 8배까지 개선된 성능을 달성하였습니다.



### The BRAVO Semantic Segmentation Challenge Results in UNCV2024 (https://arxiv.org/abs/2409.15107)
Comments:
          ECCV 2024 proceeding paper of the BRAVO challenge 2024, see this https URL

- **What's New**: BRAVO Challenge는 의미적 분할 모델의 신뢰성을 평가하기 위해 설계되었으며, 실제 왜곡과未知分布(unknown out-of-distribution) 시나리오에서 모델의 성능을 벤치마킹합니다. 대회는 100개 이상의 국제 팀이 참여하여 ML(머신러닝) 모델의 개발에 유용한 통찰을 제공합니다.

- **Technical Details**: BRAVO Challenge는 두 가지 신뢰성 범주를 정의합니다: (1) semantic reliability와 (2) OOD (out-of-distribution) reliability. 각 참가자는 Cityscapes 데이터셋을 사용한 단일 도메인 훈련 또는 여러 데이터셋을 혼합한 다중 도메인 훈련 중 하나를 선택하여 모델을 훈련해야 합니다.

- **Performance Highlights**: 제출된 모델들은 'BRAVO Index'에 의해 순위가 매겨졌으며, 결과적으로 다양한 기본 모델에서 큰 개선을 보여주었습니다. 특히 두 트랙 모두 Vision Foundation Models(VFMs)을 활용하여 전반적으로 안정적인 성능을 나타냈습니다.



### Robust Federated Learning Over the Air: Combating Heavy-Tailed Noise with Median Anchored Clipping (https://arxiv.org/abs/2409.15100)
- **What's New**: 이 논문은 연속 송신(over-the-air) 계산을 활용하여 연합 엣지 학습(federated edge learning)에서 모델 집계의 효율성을 높이는 새로운 방안을 제안합니다. 특히, 전파 채널의 중첩(superposition) 특성을 이용하여 통신과 계산의 통합 설계를 통해 시스템의 프라이버시를 향상시키고 구현 비용을 줄입니다.

- **Technical Details**: 제안된 Median Anchored Clipping (MAC) 기법은 무선 채널에서의 강한 noise의 악영향을 줄이는 새로운 그래디언트 클리핑 방법입니다. MAC은 전송 후 수신된 신호의 크기를 제한하고, 그래디언트 간의 비례 관계를 조정하여 그래디언트의 보존도를 극대화하고, 무선 전송에서의 heavy-tailed interference의 영향을 완화합니다. 또한, MAC 아래에서의 모델 학습 수렴 속도(convergence rate)를 분석적으로 도출하였습니다.

- **Performance Highlights**: 실험 결과, MAC 알고리즘은 heavy-tailed noise의 영향을 효과적으로 완화하여 시스템의 강인함을 크게 향상시킵니다. 이러한 접근 방식은 연합 학습 시스템에서의 신뢰성을 높이는데 기여하며, 다양한 실험을 통해 그 효과성을 입증하였습니다.



### Efficiently Dispatching Flash Attention For Partially Filled Attention Masks (https://arxiv.org/abs/2409.15097)
- **What's New**: 새로운 알고리즘 'Binary Block Masking (BinBlkMsk)'을 소개하며, 이 알고리즘은 기존의 Flash Attention을 개선하여 모든 종류의 attention mask를 지원합니다. 특히, 사용자 친화적이며 특정 마스크에 대한 사용자 조정 없이 사용할 수 있습니다.

- **Technical Details**: 이 방법은 attention 행렬의 관련 블록만을 처리하여 비효율적인 계산을 줄입니다. Binary Block Matrix (BinBlkMat)를 사용해 비영 상태의 항목만 선택적으로 처리하고, 긴성폭 넓이 기반으로 최적화된 방식을 도입하여 성능을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 기존 Flash Attention에 비해 최대 9배의 런타임 성능 개선을 실현했습니다. 이는 실제 시나리오에서 얻은 attention mask를 기반으로 한 실험을 통해 검증되었습니다.



### Zero-Cost Whole-Body Teleoperation for Mobile Manipulation (https://arxiv.org/abs/2409.15095)
Comments:
          Project Website: this http URL

- **What's New**: 모바일 조작자에 대한 혁신적인 원격 조작 방법론인 MoMa-Teleop을 소개합니다. 이 방법론은 추가 비용이나 장비 없이도 표준 인터페이스(예: 조이스틱, 핸드 가이드)를 통해 전체적인 조작이 가능합니다.

- **Technical Details**: MoMa-Teleop은 강화 학습(reinforcement learning) 에이전트에 기반 모션 생성을 위임하고, 조작자가 주로 작업 관련 끝단 제어(end-effector motions)에 집중할 수 있게 합니다. 이 시스템은 사용자가 6D 신호를 생성할 수 있는 어떤 인터페이스를 사용하더라도 작동 가능하며, 생성된 신호는 짧은 시간 내의 모션 계획으로 변환됩니다. 이 에이전트는 노이즈가 있는 신호도 잘 처리하는 내구성을 갖추고 있습니다.

- **Performance Highlights**: 본 접근 방식을 통해 다양한 로봇 및 작업에서 작업 완료 시간을 크게 단축할 수 있음을 입증했습니다. MoMa-Teleop은 한번의 다섯 개 시연만으로도 기존 환경에서 새로운 장애물이나 변경된 물체 위치로의 일반화(generalization)가 가능하다는 점에서도 두드러집니다.



### M2OST: Many-to-one Regression for Predicting Spatial Transcriptomics from Digital Pathology Images (https://arxiv.org/abs/2409.15092)
- **What's New**: M2OST는 디지털 병리 이미지의 다중 레벨 데이터 구조를 활용하여 유전자 발현을 정확하게 예측해주는 새로운 회귀 Transformer 모델입니다. 기존의 방법들과 달리 M2OST는 다수의 이미지를 함께 사용하여 단일 출력으로 유전자 발현을 예측합니다.

- **Technical Details**: M2OST는 회귀 Transformer로 다수의 WSI(Whole Slide Images)에서 각각의 이미지 레벨을 통합하여 ST(Spatial Transcriptomics) 맵을 예측합니다. 모델의 구조는 Intra-Level Token Mixing Module, Cross-Level Token Mixing Module 및 Cross-Level Channel Mixing Module을 포함하여 다중 스케일 특징 추출 과정을 효율적으로 분리하고 상호 작용하게 구성되어 있습니다.

- **Performance Highlights**: M2OST는 공개된 세 가지 ST 데이터셋에서 테스트되었으며, 적은 수의 매개 변수와 부동 소수점 연산(FLOPs)으로도 최첨단 성능을 달성하는 것으로 나타났습니다.



### Depression Diagnosis Dialogue Simulation: Self-improving Psychiatrist with Tertiary Memory (https://arxiv.org/abs/2409.15084)
- **What's New**: 본 논문에서는 Agent Mental Clinic (AMC)라는 자가 개선형 대화형 에이전트 시스템을 소개하여 우울증 진단의 효율성을 높입니다. 이는 환자와 정신과 의사 에이전트 간의 시뮬레이션된 대화를 통해 이루어지며, 진단 정확도를 높이기 위해 정신과 의사 에이전트의 메모리 구조 및 대화 제어 플러그인을 설계하였습니다.

- **Technical Details**: AMC 시스템은 3개의 주요 부분으로 구성되어 있습니다: 1) 환자 에이전트: D4 데이터셋을 기반으로 생성된 다양한 환자들. 2) 정신과 의사 에이전트: 진단 대화를 통해 반영된 기술을 사용하는 에이전트. 3) 감독자 플러그인: 대화 과정을 제어하고 정신과 의사 에이전트의 반영을 촉진하는 불완전한 에이전트. 이러한 구조는 우울증 진단 및 대화 시뮬레이션의 최적화를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, AMC 시스템은 우울증 진단 정확도를 평균 6.05% 향상시켰으며, 자살 예측 정확도는 1.8% 증가했습니다. 이 시스템은 제한된 수의 레이블이 있는 경우에도 다른 특정 도메인에 적용 가능합니다.



### Enhancing Scientific Reproducibility Through Automated BioCompute Object Creation Using Retrieval-Augmented Generation from Publications (https://arxiv.org/abs/2409.15076)
Comments:
          21 pages, 8 figures

- **What's New**: 본 연구에서는 IEEE BioCompute Object (BCO) 표준을 따르는 문서 생성을 자동화하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 Retrieval-Augmented Generation (RAG) 및 Large Language Models (LLMs)을 사용하여 과학 논문에서 BCO를 자동으로 생성하는 도구인 BCO assistant를 개발합니다. 주요 기술적 도전 과제인 LLM 환각(data hallucination) 및 긴 맥락 이해(long-context understanding)에 대한 해결책을 설명하며, 두 단계 검색(two-pass retrieval) 및 재순위화(re-ranking) 과정을 포함한 최적화된 검색 프로세스를 구현하였습니다.

- **Performance Highlights**: BCO assistant는 생명정보학(bioinformatics) 연구의 문서화를 자동화하여 필요한 시간과 노력을 크게 줄일 수 있습니다. 이 접근 방식은 과학적 재현성(scientific reproducibility)을 높이고, AI 지원 과학 문서화 및 지식 추출 가능성을 열어줍니다.



### Brotherhood at WMT 2024: Leveraging LLM-Generated Contextual Conversations for Cross-Lingual Image Captioning (https://arxiv.org/abs/2409.15052)
Comments:
          Accepted at the Ninth Conference on Machine Translation (WMT24), co-located with EMNLP 2024

- **What's New**: 본 논문에서는 Brotherhood라는 팀 이름으로 영어에서 저해상도 다중 모달 번역 작업을 위한 시스템을 소개합니다. 영어-힌디어, 영어-하우사어, 영어-벵골어, 영어-말라얄람어 언어 쌍의 다중 모달 번역 작업에 참여하였습니다.

- **Technical Details**: 본 연구는 전통적인 훈련(fine-tuning) 없이 cross-lingual image captioning을 향상시키기 위해 다중 모달 대규모 언어 모델(multi-modal Large Language Models)인 GPT-4o와 Claude 3.5 Sonnet을 활용하는 방법을 제시합니다. Instruction-tuned prompting을 사용하여 잘라낸 이미지에 대한 풍부한 맥락 대화를 생성하며, 영어 캡션을 추가 맥락으로 사용합니다. 이러한 합성 대화는 타겟 언어로 번역됩니다. 마지막으로, 무게 조절 프롬프트(weighted prompting) 전략을 사용하여 원본 영어 캡션과 번역된 대화의 균형을 잡아 타겟 언어에서 캡션을 생성합니다.

- **Performance Highlights**: 본 방법은 영어-힌디어 챌린지 세트에서 37.90 BLEU 점수를 획득했으며, 영어-하우사어의 챌린지와 평가 리더보드에서 각각 1위와 2위에 랭크되었습니다. 또한, 250개의 이미지 하위 집합에 대한 추가 실험을 실시하여 다양한 가중치( weighting schemes) 조정 방식에서 BLEU 점수와 의미론적 유사성 사이의 trade-off를 탐색했습니다.



### Scaling Laws of Decoder-Only Models on the Multilingual Machine Translation Task (https://arxiv.org/abs/2409.15051)
- **What's New**: 최근 연구들은 인코더-디코더(encoder-decoder) 모델의 지배를 받던 기계 번역 분야에서 디코더 전용(decoder-only) 모델의 가능성을 탐구하고 있습니다. 본 연구는 다양한 언어와 도메인에서의 번역 작업을 위해 70M에서 7B 파라미터까지의 디코더 전용 모델을 훈련시키고, 이러한 모델의 스케일링 법칙을 조사하였습니다.

- **Technical Details**: 연구진은 디코더 전용 모델이 대형 언어 모델(LLM)에서 발견된 스케일링 법칙과 유사한 법칙을 따르지만, 모델의 크기가 너무 크거나 다른 데이터 분포에 일반화하는 데에는 어려움이 있다는 것을 발견했습니다. 또한, 모델의 깊이와 너비를 확장하는 방법이 유사한 테스트 손실 개선을 가져오지만, 후자는 모델의 효율성에 더 좋은 영향을 미친다고 합니다.

- **Performance Highlights**: 이번 연구에서 디코더 전용 모델은 이전의 인코더-디코더 모델보다 더 효율적인 훈련을 가능하게 하며, 특히 대량의 데이터를 처리하는 데 유리하다는 점이 강조되었습니다. 또한, 훈련 샘플의 배치(pack) 문제를 해결하기 위한 방법도 제안했습니다.



### AlphaZip: Neural Network-Enhanced Lossless Text Compression (https://arxiv.org/abs/2409.15046)
- **What's New**: 본 논문은 대규모 언어 모델(LLM)을 사용한 무손실 텍스트 압축 접근법을 제안합니다. 이 방법은 두 가지 주요 단계로 구성됩니다: 1) 밀집 신경망 아키텍처를 사용한 순위 예측, 2) 예측된 순위를 표준 압축 알고리즘으로 압축.

- **Technical Details**: 신경망 기반의 예측 압축 방식은 신경망의 예측 기능을 통해 추가적인 중복성을 도입하고, 표준 알고리즘으로 이를 압축합니다. 이 연구에서는 GZIP 압축 알고리즘을 기준으로 57% 향상된 압축 비율을 달성했습니다.

- **Performance Highlights**: LLM을 활용한 압축 성능 분석 결과, 더 작은 모델에서도 효과적인 텍스트 압축이 가능하며, 모델 크기 및 매개변수 수를 다양하게 조정하여 성능을 비교하였습니다. LLM의 예측 정확성이 압축 비율에 밀접한 관련이 있음을 보여주었습니다.



### Region Mixup (https://arxiv.org/abs/2409.15028)
Comments:
          Published as a Tiny Paper at ICLR 2024

- **What's New**: 이번 연구는 시각적 인식 작업에서 일반화(generalization)를 개선하기 위한 mixup (Zhang et al., 2018) 데이터 증강 방법의 간단한 확장을 소개합니다. 기존 mixup 방법이 전체 이미지를 혼합하는 것과 달리, 제안된 Region Mixup은 여러 이미지의 특정 영역을 결합하는 데 중점을 둡니다.

- **Technical Details**: Region Mixup의 핵심은 이미지의 지역(region)을 선택하여 새로운 학습 샘플(𝑥~,𝑦~)을 생성하는 것입니다. 각 이미지를 연결된 타일(tile)로 나누고, 이 타일들의 조합을 통해 새로운 이미지를 생성합니다. 이 과정에서 binary mask를 사용하여 섞을 부분을 지정하고, element-wise multiplication을 통해 두 이미지의 특정 영역을 결합합니다.

- **Performance Highlights**: CIFAR-10, CIFAR-100, Tiny ImageNet 데이터셋에 대한 이미지 분류 실험을 통해 Region Mixup의 효과성을 입증했습니다. 실험 결과, Region Mixup 방법이 기존의 Mixup 및 CutMix보다 더 향상된 성능을 보여주었습니다. 이 연구는 Region Mixup이 심층 학습의 귀중한 정규화 도구가 될 가능성이 높음을 시사합니다.



### Generative LLM Powered Conversational AI Application for Personalized Risk Assessment: A Case Study in COVID-19 (https://arxiv.org/abs/2409.15027)
- **What's New**: 이 연구는 전통적인 머신러닝 방법을 필요로 하지 않는 새로운 질병 위험 평가 방법을 제시하며, Generative LLM을 활용한 스티밍 인간-AI 대화를 통해 전방위적인 진단 가능성을 강조합니다.

- **Technical Details**: 본 연구는 Llama2-7b 및 Flan-T5-xl과 같은 사전 학습된 Generative LLM을 사용하여 COVID-19 심각도 위험 평가 사례 연구를 통해, 적은 수의 자연어 예제를 활용하여 모델을 고도화했습니다. 이를 통해 Logistic Regression, XGBoost, Random Forest와 같은 전통적인 분류기와 비교했습니다.

- **Performance Highlights**: 잘 조정된 LLM들이 전통적인 분류 방법보다 더 높은 AUC 점수를 달성하여, 제한된 데이터에서 사용할 수 있는 가능성을 입증했습니다. 이를 통해 Generative LLM이 의료 분야에서 일반적인 적재와 반응을 공정하게 처리할 수 있음을 강조했습니다.



### A Diagonal Structured State Space Model on Loihi 2 for Efficient Streaming Sequence Processing (https://arxiv.org/abs/2409.15022)
Comments:
          6 pages, 2 figures

- **What's New**: 이번 연구에서는 Deep State-Space Models (SSM)의 새로운 효율적인 token-by-token 처리 방법을 Intel의 Loihi 2 신경형 컴퓨터 프로세서에서 구현했습니다.

- **Technical Details**: SSM S4D의 neuromorphic (신경형) 하드웨어 구현을 통해 기존의 Jetson Orin Nano에서 수행된 recurrent (재귀적) 및 convolutional (합성곱) 방식과 비교하였습니다. Loihi 2는 token-by-token 처리에서 효과적으로 작동하며, 기존 재귀 버전보다 에너지를 1000배 덜 소모하고, 지연시간이 75배 낮으며, 데이터 처리량이 75배 더 높습니다.

- **Performance Highlights**: Loihi 2는 sample-by-sample (샘플 단위) 처리를 제외한 token-by-token 기반 처리에서 우수한 성능을 발휘합니다. 이는 SSM의 실시간 스트리밍 응용 프로그램을 위한 새로운 가능성을 열어줍니다.



### Inference-Friendly Models With MixAttention (https://arxiv.org/abs/2409.15012)
- **What's New**: 본 논문에서는 KV 캐시의 크기를 줄이기 위해 MixAttention 아키텍처를 제안하며, 이 아키텍처는 최근의 토큰 집합만을 저장하는 슬라이딩 윈도우 어텐션(Sliding Window Attention)과 레이어 간 KV 캐시 공유(KV Cache Sharing)를 결합한 방법입니다. 이를 통해 메모리 사용량을 줄이고 추론 속도를 개선할 수 있다는 점에서 주목할 만합니다.

- **Technical Details**: MixAttention은 모델의 메모리 소비를 줄이고, 긴 입력에 대한 추론 성능을 향상시키기 위해 슬라이딩 윈도우 어텐션과 KV 캐시 공유를 결합한 방법론을 사용합니다. 여러 가지 아키텍처 변형을 통해 평가하였고, 다양한 조합이 모델의 성능에 미치는 영향을 조사하였습니다. 이 결과, 짧은 및 긴 컨텍스트 작업 모두에서 모델 품질을 유지하면서도 자원 효율성을 최적화하는 구성을 발견하였습니다.

- **Performance Highlights**: MixAttention 구조는 추론 속도를 높이고 메모리 사용량을 줄이는 데 성공적으로 기여했으며, 대부분의 평가 지표에서 표준 트랜스포머 모델과 유사한 성능을 보여주었습니다. 특히, 레이어 간 KV 캐시 공유와 슬라이딩 윈도우 레이어 추가가 추론 성능을 향상시키고 메모리 사용을 줄이는 데 효과적입니다.



### Generalizing monocular colonoscopy image depth estimation by uncertainty-based global and local fusion network (https://arxiv.org/abs/2409.15006)
- **What's New**: 이번 연구에서는 깊이 추정(depth estimation)을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 CNN(Convolutional Neural Network)과 Transformer의 조합을 통해 지역(local) 및 글로벌(global) 정보의 상호 보완을 극대화하며, 의료 영상에서의 심도 추정을 향상시킵니다.

- **Technical Details**: 제안된 프레임워크는 CNN을 이용한 지역 특성 추출과 Transformer를 통한 글로벌 정보 캡처로 구성되어 있습니다. 여기에는 불확실성(uncertainty)을 기반으로 한 융합 모듈(fusion module)이 포함되어 있어, CNN과 Transformer의 예측을 효과적으로 결합합니다. 이 네트워크는 시뮬레이션 데이터로 훈련될 수 있으며, 어떠한 파인튜닝(fine-tuning) 없이도 실제 임상 데이터에 직결되는 일반화(generalization)를 이루어냅니다.

- **Performance Highlights**: 다양한 데이터셋에서 검증을 통해 우수한 일반화 능력을 보여주며, 실제 임상 환경에서도 강력한 성능을 입증했습니다. 이는 복잡한 조건 속에서도 endoscopic 영상에서의 심도 맵(depth map)을 신뢰성 있게 추정할 수 있는 기반을 마련하여, 자동 내시경 내비게이션 및 기타 임상 작업, 예를 들어 폴립 탐지 및 분할(segmentation) 등에 기여할 것입니다.



### Method of Equal Shares with Bounded Overspending (https://arxiv.org/abs/2409.15005)
- **What's New**: 이번 논문에서는 참여 예산편성(participatory budgeting, PB)에서의 비율(proportionality)의 중요성을 강조하면서, 전통적인 비율 규칙들이 때로는 비효율적인 결과를 초래할 수 있음을 지적합니다. 우리는 비율과 효율성(efficiency)을 균형있게 유지하는 새로운 방법인 Bounded Overspending(BOS Equal Shares)을 소개합니다.

- **Technical Details**: BOS Equal Shares는 기존의 Equal Shares 방법에 대한 강력한 변형으로, 엄격한 비율 보장을 통해 발생하는 비효율성을 해결합니다. 이 방법은 여전히 원래의 Equal Shares 방법과 유사한 좋은 비율을 제공합니다. 또한, 프로젝트의 부분 자원 지원(partial funding)을 가능하게 하는 방법의 분수적(fractional) 변형도 논의합니다.

- **Performance Highlights**: BOS Equal Shares 방법은 참여 예산편성에서 비율과 효율성을 동시에 만족시키는 새로운 접근 방식을 제공하며, 이는 다양한 투표 그룹의 공정한 대우를 보장하면서도 실질적인 예산 배분을 최적화하는 데 기여합니다.



### Evaluating Theory of (an uncertain) Mind: Predicting the Uncertain Beliefs of Others in Conversation Forecasting (https://arxiv.org/abs/2409.14986)
- **What's New**: 이 연구는 Theory of Mind의 개념을 확장하여 언어 모델이 대화 중 다른 사람의 신념에 대한 불확실성을 예측하는 새로운 과제를 제안합니다. 기존의 신념 예측 과제가 이분법적으로 신념을 취급하는 반면, 이 논문은 대화자들의 신념이 더 유동적일 수 있으며, 불확실성의 정도까지 평가할 수 있음을 강조합니다.

- **Technical Details**: 저자들은 대화 예측(conversation forecasting) 기법을 통해 언어 모델의 불확실성 예측 능력을 평가합니다. 특히, 대화자가 직접적으로 대화 결과에 대한 불확실성을 예측하도록 모델을 훈련시키고, 이를 바탕으로 세 개의 대화 코퍼스(사회적, 협상, 작업 지향적)와 여덟 개의 모델을 가지고 실험을 수행했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 언어 모델은 다른 사람의 불확실성 변동의 최대 7.5%를 설명할 수 있지만, 이 과제가 여전히 어렵고 이는 향후 연구의 여지를 남깁니다. 저자들은 특히 이 연구의 방법론적 기여 외에도 상황에 따른 대화자의 목표와 맥락이 언어 모델의 ToM(Theory of Mind) 능력에 미치는 영향을 조사했습니다.



### Sparse-to-Dense LiDAR Point Generation by LiDAR-Camera Fusion for 3D Object Detection (https://arxiv.org/abs/2409.14985)
Comments:
          7 pages

- **What's New**: 본 논문에서는 LiDAR 센서의 한계를 보완하기 위해 LiDAR-Camera Augmentation Network (LCANet)라는 새로운 프레임워크를 제안합니다. 이는 2D 이미지 특징을 활용하여 LiDAR 포인트 클라우드 데이터를 재구성하며, 탐지 정확도를 개선하기 위해 추가 포인트를 생성합니다.

- **Technical Details**: LCANet은 LiDAR 센서와 카메라로부터 수집된 데이터를 융합하여 3D 공간에 이미지 특징을 투사합니다. 이 과정에서 생성된 3D 피처는 의미론적(semantic) 및 공간적(spatial) 정보를 포함합니다. 네트워크는 2D 이미지 특징을 적용하여 부족한 포인트를 보충하며, 이는 자율주행 차량이나 로봇에서 중요한 안전성을 확보하는 데 기여합니다.

- **Performance Highlights**: KITTI 및 Waymo 데이터셋에서의 광범위한 실험 결과, LCANet은 기존 모델에 비해 특히 희소하고 먼 거리의 물체 탐지에서 매우 우수한 성능을 보였습니다. 이는 LiDAR만 사용하는 방법과 비교하여 더 정밀한 포인트 생성과 더 적은 허위 긍정을 달성했습니다.



### Dynamic Integration of Task-Specific Adapters for Class Incremental Learning (https://arxiv.org/abs/2409.14983)
- **What's New**: 본 논문은 Non-Exemplar Class Incremental Learning (NECIL) 문제를 해결하기 위해 Dynamic Integration of task-specific Adapters (DIA)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Task-Specific Adapter Integration (TSAI)와 Patch-Level Model Alignment의 두 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: DIA 프레임워크는 패치 레벨 어댑터 통합 전략을 사용하여 컴포지셔널리티(compositionality)를 향상시키고, 두 가지 특수 메커니즘인 Patch-Level Distillation Loss (PDL)와 Patch-Level Feature Reconstruction (PFR)를 통해 특성 일관성(feature consistency)과 정확한 결정 경계(decision boundary)를 유지합니다.

- **Performance Highlights**: 광범위한 실험 결과, DIA는 NECIL 환경에서 벤치마크 데이터셋에서 눈에 띄는 성능 개선을 보였으며, 계산 복잡도를 최대 90% 감소시키면서도 최첨단(state-of-the-art, SOTA) 성능을 유지합니다.



### On The Specialization of Neural Modules (https://arxiv.org/abs/2409.14981)
Comments:
          The Eleventh International Conference on Learning Representations 2023

- **What's New**: 이번 연구는 신경 모듈이 데이터셋의 구조에 특화될 수 있는 가능성을 이론적으로 연구하고, 체계적인 일반화를 위한 형식을 제공합니다.

- **Technical Details**: 본 연구에서는 컴포지셔널 구조(compositional structure)와 체계성(systematicity) 간의 차이를 명확히 하고, 데이터셋의 하위 구조(sub-structure)를 탐색하여 일반화를 향상시킬 수 있는 모듈성을 목표로 합니다. 깊은 선형 신경망 모듈(deep linear network modules)의 훈련 역학(training dynamics)을 데이터셋 매개변수에 따라 분석하며, 비모듈 네트워크(non-modular networks)와 모듈 네트워크(modular networks)의 일반화 능력을 다룹니다.

- **Performance Highlights**: 연구 결과는 모듈 네트워크 아키텍처가 체계적인 매핑을 배울 수 있는 조건을 밝히며, 데이터의 구조와 아키텍처의 편향이 체계적인 일반화를 촉진하는 데 매우 중요함을 보여줍니다.



### Deep Reinforcement Learning-based Obstacle Avoidance for Robot Movement in Warehouse Environments (https://arxiv.org/abs/2409.14972)
- **What's New**: 이 논문은 창고 환경에서 모바일 로봇의 장애물 회피를 효율적이고 친숙하게 수행하기 위한 심층 강화 학습(deep reinforcement learning) 기반 알고리즘을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 보행자 상호작용을 기반으로 가치 함수 네트워크(value function network)를 개선하며, 보행자 각도를 격자 형태로 나누어 상호작용 정보를 추출합니다. 또한, 주의 메커니즘(attention mechanism)을 통해 보행자의 개별적인 시간적 특성을 추출하고, 현재 상태와 역사적 궤적 상태의 상대적 중요성을 학습하여 로봇의 장애물 회피 전략에 대한 공동 영향을 분석합니다. 보상 함수(reward function)는 보행자의 공간 행동을 기반으로 설계되며, 각도가 과도하게 변하는 상태에 대해 로봇에 패널티를 부여하여 편안한 장애물 회피를 구현합니다.

- **Performance Highlights**: 시뮬레이션 실험을 통해 제안된 심층 강화 학습 기반 모바일 로봇 장애물 회피 알고리즘이 복잡한 창고 환경에서도 실행 가능성과 효율성을 입증하였습니다.



### Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make your LLMs use External Data More Wisely (https://arxiv.org/abs/2409.14924)
- **What's New**: 이 연구에서는 외부 데이터를 활용한 대형 언어 모델(LLM)의 다양한 활용사례와 그 과정에서의 도전 과제를 제시합니다. 특히 Retrieval-Augmented Generation (RAG) 기법을 활용하여 사용자 쿼리를 네 가지 레벨로 분류하여 각 레벨에 맞는 기술적 접근 방식을 정리합니다.

- **Technical Details**: 본 논문에서는 쿼리를 명시적 사실 쿼리(Explicit Fact Queries), 암시적 사실 쿼리(Implicit Fact Queries), 해석 가능한 근거 쿼리(Interpretable Rationale Queries), 숨겨진 근거 쿼리(Hidden Rationale Queries)로 나누어, 각 쿼리에 필요한 외부 데이터의 유형과 작업의 주요 초점을 정의합니다. 또한 LLM에 외부 데이터를 통합하는 세 가지 주요 형태인 컨텍스트(Context), 소규모 모델(Small Model), 파인 튜닝(Fine-tuning)의 장단점도 살펴봅니다.

- **Performance Highlights**: 데이터 보강 LLM 애플리케이션은 전문성과 적시성을 향상시키며, 도메인 전문가와의 정렬, 모델 환각 감소 및 제어 가능성 및 설명 가능성을 개선하는 장점을 제공합니다. 그러나 여전히 많은 개발자들이 이 기술을 활용하기 위해 상당한 인간 노력을 투입해야 한다는 과제가 남아 있습니다.



### KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems (https://arxiv.org/abs/2409.14908)
- **What's New**: KARMA라는 혁신적인 메모리 시스템을 도입하여, 긴 순서의 가사 작업을 수행하는 Embodied AI 에이전트의 인문맥 기억(in-context memory) 문제를 해결합니다.

- **Technical Details**: KARMA는 장기 기억(long-term memory)과 단기 기억(short-term memory) 모듈을 통합하여, 환경의 종합적인 3D 장면 그래프(scene graphs)와 객체의 위치 및 상태의 변화를 기록합니다. 이 이중 기억 구조는 에이전트가 관련 과거 장면 경험을 검색할 수 있도록 하여, 작업 계획의 정확성과 효율성을 향상시킵니다.

- **Performance Highlights**: KARMA를 갖춘 메모리 강화 Embodied AI 에이전트는 AI2-THOR 시뮬레이터 내에서 Composite Tasks에서 1.3배, Complex Tasks에서 2.3배 성공률을 개선하였으며, 작업 실행 효율성을 각각 3.4배 및 62.7배 향상시킵니다.



### DSG-KD: Knowledge Distillation from Domain-Specific to General Language Models (https://arxiv.org/abs/2409.14904)
Comments:
          IEEE ACCESS 2024

- **What's New**: 본 연구는 어린이 응급 치료 센터에서 얻은 전자 의료 기록(EMR) 데이터 기준으로 비상/비비상 분류 작업을 수행하며, 기존의 도메인 특화 언어 모델들이 일반 언어 모델에 비해 성능이 부족함을 보이고 있습니다. 이를 해결하기 위해 도메인 지식 전이 방법론을 제안하였습니다.

- **Technical Details**: 언어 모델은 교사 모델(teacher model)과 학생 모델(student model)로 정의됩니다. 의료 도메인 데이터로 사전 훈련된 모델(예: KM-BERT)을 교사 모델로, 일반 언어 모델(예: Ko-BERT)을 학생 모델로 삼아 지식을 전이하는 과정에서, 학생 모델이 교사 모델의 숨겨진 상태(hidden states) 및 주의 행렬(attention matrices)을 학습하도록 훈련됩니다. 이 방법에서는 Knowledge Distillation (KD) 기술을 활용하여 도메인 특정 지식을 일반 모델에 주입합니다.

- **Performance Highlights**: 제안된 방법은 한국 PED EMR 데이터에서 비상 및 비비상 사례 분류에서 높은 성능을 보여, 기존 모델들을 능가하였습니다. 또한 이 방법론은 다양한 전문 및 기술 분야에서도 폭넓은 적용 가능성을 제시합니다.



### Deploying Open-Source Large Language Models: A performance Analysis (https://arxiv.org/abs/2409.14887)
- **What's New**: 이번 연구는 ChatGPT가 출시된 이후 대형 언어 모델(LLMs)의 성능과 배포 관련 정보를 제공하도록 설계된 여러 가지 테스트 결과를 제시합니다. Mistral과 LLaMa와 같은 다양한 크기의 모델들을 비교하며, 이를 통해 LLMs를 대량으로 사용할 수 있는 환경을 조성하고자 합니다.

- **Technical Details**: 실험은 Plafrim 컴퓨팅 서버에서 진행되었으며, NVIDIA V100 16GB 및 A100 40GB GPU를 사용했습니다. vLLM이라는 Python 라이브러리를 통해 모델 추론을 최적화하고, 여러 요청을 동시에 처리할 수 있는 능력으로 효율성을 증대시켰습니다. 모델들은 AWQ, GPTQ, GGUF와 같은 다양한 정량화(quantification) 방법을 통해 메모리 사용을 최소화하며, 최대 70억 개의 파라미터를 가진 모델에서도 정량화를 통한 성능 유지를 보여주었습니다.

- **Performance Highlights**: Mistral 및 LLaMa 모델의 실험 결과, 더 큰 입력 컨텍스트 크기가 모델의 속도에 영향을 미치는 것으로 나타났습니다. 요청 수가 증가할 때 성능 저하가 비선형적으로 발생하며, 두 개의 A100 40GB GPU로도 LLaMa-3-70B와 Mixtral 8x7B 모델을 원활하게 실행할 수 있음을 보여주었습니다.



### End-to-End Graph Flattening Method for Large Language Models (https://arxiv.org/abs/2409.14880)
Comments:
          2024 1st International Conference on Computational Linguistics and Natural Language Processing (CLNLP 2024)

- **What's New**: 본 연구에서는 Large Language Models (LLMs)의 그래프 처리 성능을 향상시키기 위한 새로운 방법인 End-to-End DAG-Path prompting (EEDP)를 제안합니다. EEDP는 기존의 그래프 평탄화 방법의 한계를 극복하여 장거리 시나리오에서의 추론 성능을 개선합니다.

- **Technical Details**: 기존의 그래프 평탄화 방법은 텍스트 형식으로 변환되어 LLMs에 사용되며, 이로 인해 장거리 의존성을 처리하는 데 한계를 보입니다. EEDP는 주 요약 경로(main backbone paths)를 활용하여 텍스트 설명을 생성하는 방법으로, 그래프의 시작 및 끝 노드만을 고려하여 최적의 표현을 도출합니다.

- **Performance Highlights**: 실제 데이터셋을 기반으로 한 실험 결과 EEDP는 장거리 및 단거리 시나리오에서 모두 뛰어난 성능을 보이며, 특히 장거리 시나리오에서의 LLM 성능 향상에 기여함을 보여주었습니다.



### Mammo-Clustering:A Weakly Supervised Multi-view Global-Local Context Clustering Network for Detection and Classification in Mammography (https://arxiv.org/abs/2409.14876)
Comments:
          10 pages, 5 figures

- **What's New**: 이 연구에서는 맥락 클러스터링에 기반한 약한 감독 멀티 뷰 유방촬영 조기 스크리닝 모델을 제안하여 유방암의 조기 발견을 위한 새로운 접근 방식을 제공합니다. 이 모델은 CNN 또는 Transformer에 의존하지 않고 정보를 보완하기 위한 멀티 뷰 학습을 결합하여 사람들에게 유용한 솔루션을 제공합니다.

- **Technical Details**: 본 모델은 약한 감독 학습(w weak supervision) 전략을 적용하여 데이터 제한 문제를 해결하고, 다수의 이미지 시각을 활용하여 특성 변형을 통해 향상된 성능을 이루어냅니다. 기존 전통적 방법과는 달리, 이 모델은 두 개의 공개 데이터 세트에서 각각 AUC 0.828 및 0.805를 기록하며, 이로써 실제 환경에서의 적용 가능성을 높입니다.

- **Performance Highlights**: 이 모델은 기존의 방법론에 비해 더 적은 파라미터로 첨단 성능을 달성하며, 특히 덜 발전된 지역에서의 유방암 스크리닝의 용이성을 높이고 의사의 부담을 줄일 수 있는 잠재력을 보여줍니다.



### Towards Ground-truth-free Evaluation of Any Segmentation in Medical Images (https://arxiv.org/abs/2409.14874)
Comments:
          17 pages, 15 figures

- **What's New**: 이번 연구에서는 의학 영상에서 Segment Anything Model (SAM)과 그 변형들이 생성하는 분할(segmentation) 품질을 평가할 수 있는 ground-truth-free(기준 데이터 없음) 평가 모델을 제안했습니다. 이 모델은 입력 이미지와 해당하는 분할 예측 간의 일관성과 응집력을 분석하여 품질 점수를 추정합니다.

- **Technical Details**: 이 평가 모델은 지도 학습(supervised learning) 프레임워크 내의 회귀 문제로 구성되며, Dice 수치 등과 평균 제곱 오차(mean squared error)를 사용하여 학습 손실을 계산합니다. EvanySeg라는 이름의 이 모델은 ResNet 및 ViT 등 다양한 컨볼루션 모델(convolution-based models)과 변환기 모델(transformer-based models)을 활용하였으며, ViT 모델이 더 나은 성능을 보였습니다.

- **Performance Highlights**: EvanySeg는 저품질 분할 샘플을 발견하고, 기준 데이터 없이 분할 모델을 벤치마킹하며, 인간-인공지능 협업 중 저품질 분할 예측에 대해 전문가에게 경고하고, 여러 개의 분할 모델이 있을 때 각 테스트 샘플에 대해 최상의 분할 예측을 선택할 수 있는 다양한 과제에 활용될 수 있습니다. 코드와 모델은 공개될 예정입니다.



### FedSlate:A Federated Deep Reinforcement Learning Recommender System (https://arxiv.org/abs/2409.14872)
- **What's New**: FedSlate는 사용자의 행동 간 상호작용을 고려하여 멀티 플랫폼에서 추천 알고리즘의 효율성을 극대화하는 연합 강화 학습 (federated reinforcement learning) 기반의 새로운 접근 방식을 제안합니다.

- **Technical Details**: FedSlate 알고리즘은 SlateQ 알고리즘을 활용하여 추천 콘텐츠의 가치를 평가하고, 사용자 행동의 장기 패턴을 학습합니다. 이 알고리즘은 각 플랫폼에서 로컬 Q-값을 계산하고 이를 중앙 서버에 전달하여 글로벌 Q-값을 생성하는 구조를 가집니다. 이후 로컬 에이전트는 이 Q-값을 바탕으로 정책 결정을 내립니다.

- **Performance Highlights**: 실험 결과, FedSlate는 기존의 기준 방법들에 비해 다양한 환경 설정에서 우수한 성능을 보여주었으며, 기준 방법이 전혀 적용되지 않는 상황에서도 추천 전략 학습을 가능하게 했습니다.



### A novel agent with formal goal-reaching guarantees: an experimental study with a mobile robo (https://arxiv.org/abs/2409.14867)
- **What's New**: 이번 연구는 Critic As Lyapunov Function (CALF)이라는 새로운 안전 모델-프리 Reinforcement Learning (RL) 에이전트를 제안하며, 로봇 제어의 기반성을 향상시키면서도 안전한 목표 도달을 보장하는 방법을 보여준다.

- **Technical Details**: CALF는 모든 상태-행동 쌍이 탐색 가능하도록 하면서도 원하는 목표 상태에 도달하는 것을 수학적으로 보장한다. 본 연구에서는 TURTLEBOT3 Burger와 같은 비유한 휠 모바일 로봇에 대한 수치 실험을 통해 CALF의 우수성을 확인하였다.

- **Performance Highlights**: CALF는 Proximal Policy Optimization (PPO) 및 수정된 SARSA와 비교했을 때 적은 에피소드(episode) 환경에서 얻어진 총 비용 측면에서 우수한 성능을 보였다.



### Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs (https://arxiv.org/abs/2409.14866)
- **What's New**: 본 논문은 새로운 자동화된 jailbreaking 공격 프레임워크를 제안하여 기존의 수동적이고 길이가 긴 템플릿에서 벗어나, 짧고 의미 있는 프롬프트를 생성할 수 있는 방법을 소개합니다.

- **Technical Details**: 이 프레임워크는 black-box fuzz testing 접근 방식을 기반으로 하며, 빈 시드 풀(empty seed pool)로 시작하여 템플릿에 대한 의존성을 없앱니다. LLM 헬퍼를 활용하여 의미적으로 일관된 질문 종속 변이 전략(question-dependent mutation strategies)을 개발하고, 두 단계의 검사 모듈을 통해 성공적인 jailbreak 여부를 판단합니다.

- **Performance Highlights**: 7개의 대표 LLM을 대상으로 평가한 결과, 본 방법은 GPT-3.5 turbo, GPT-4, Gemini-Pro에 대해 각각 90%, 80%, 74%의 공격 성공률을 달성하며, 기존 방법보다 60% 이상 우수한 성능을 나타냈습니다.



### Embedding Knowledge Graph in Function Spaces (https://arxiv.org/abs/2409.14857)
- **What's New**: 본 연구에서는 기존의 knowledge graph embedding (KGE) 접근 방식에서 벗어나, 정해진 차원의 함수 공간(function space) 내에서 작동하는 새로운 embedding 방법을 소개합니다. 이 방법은 폴리노미얼 함수(polynomial functions)를 사용하여 embedding을 계산하며, 점차 복잡한 레이어를 가진 신경망(neural networks)으로 발전합니다.

- **Technical Details**: 제안된 방법론은 FMultn, FMultni, FMult 세 가지 기능적 embedding 기법을 포함합니다. FMultn은 폴리노미얼을 사용하여 엔티티와 관계 사이의 비선형(non-linear) 관계를 모델링하며, FMultni는 삼각함수를 사용하여 KG 내의 주기적 패턴을 통합합니다. FMult는 신경망을 활용하여 다양한 맥락에 적응 가능한 복잡한 고차원 embedding을 학습합니다.

- **Performance Highlights**: 기능적 embedding 접근 방법은 기존의 정적(static) embedding 기법보다 향상된 성능을 보여주며, knowledge graph의 복합적인 동적(dynamics)을 포착하는 데 효과적임을 강조합니다.



### FUSED-Net: Enhancing Few-Shot Traffic Sign Detection with Unfrozen Parameters, Pseudo-Support Sets, Embedding Normalization, and Domain Adaptation (https://arxiv.org/abs/2409.14852)
Comments:
          17 pages, 6 figures, 3 tables, submitted to IEEE Access for review

- **What's New**: 이 논문에서는 FUSED-Net이라는 새로운 트래픽 신호 인식 네트워크를 제안합니다. 이 네트워크는 Faster RCNN을 기반으로 하며, Unfrozen Parameters, Pseudo-Support Sets, Embedding Normalization, 그리고 Domain Adaptation을 활용하여 적은 데이터로도 우수한 성능을 발휘할 수 있도록 설계되었습니다.

- **Technical Details**: FUSED-Net의 주요 기술적 특징으로는 모든 파라미터를 학습 가능한 상태로 유지하는 방식(Unfrozen Parameters), 데이터 증강을 통해 생성되는 Pseudo-Support Sets, 인터 클래스 변동성을 줄이는 Embedding Normalization, 그리고 다양한 트래픽 신호 데이터셋을 통한 사전 학습을 이용한 Domain Adaptation이 있습니다. 이러한 요소들은 FUSED-Net이 제한된 샘플에서 더욱 효율적으로 학습하도록 합니다.

- **Performance Highlights**: BDTSD 데이터셋에서 FUSED-Net은 기존의 최첨단 Few-Shot Object Detection(FSOD) 모델 대비 1-shot, 3-shot, 5-shot, 10-shot 시나리오에서 각각 2.4배, 2.2배, 1.5배, 1.3배 향상된 mAP(mean Average Precision)를 달성하였습니다. 이 결과는 서로 다른 도메인에서의 성능 평가에서도 우수함을 보여줍니다.



### GroCo: Ground Constraint for Metric Self-Supervised Monocular Depth (https://arxiv.org/abs/2409.14850)
- **What's New**: 본 논문에서는 자가 감독 학습(self-supervised learning) 환경에서의 깊이 추정을 개선하기 위해 새롭고 혁신적인 접근법을 제안합니다. 특히, we propose a new constraint on ground areas를 통해 깊이 예측의 정확성과 일반화 능력을 크게 향상시킵니다.

- **Technical Details**: 우리는 ground plane prior를 통합하기 위한 새로운 손실 함수(loss function)를 도입하여, 깊이 추정 모델이 다양한 카메라 구성을 고려할 수 있는 힘을 부여합니다. 이 접근법은 깊이 예측과 ground prior 간의 일관성을 보장하여, 보다 정확한 스케일 복구(scale recovery)를 가능하게 합니다.

- **Performance Highlights**: 우리의 실험 결과는 KITTI 벤치마크에서 기존의 스케일 복구 기법들보다 뛰어난 성능을 보여주며, 다양한 카메라 회전(camera rotations)과 이전에 보지 못한 운전 데이터셋에 대한 제로샷(zero-shot) 조건에서도 모델의 일반화 능력을 크게 향상시키는 것으로 나타났습니다.



### Orthogonal Finetuning for Direct Preference Optimization (https://arxiv.org/abs/2409.14836)
- **What's New**: 본 논문에서는 기존의 Direct Preference Optimization (DPO) 알고리즘에서 발생하는 오버피팅 문제를 해결하기 위해 회전된 선호 최적화(weight-Rotated Preference Optimization, RoPO) 방법을 제안합니다. 이 방법은 신경망의 가중치 매개변수를 회전 및 크기 스트레칭 업데이트하여 초기 지식을 보존합니다.

- **Technical Details**: RoPO는 DPO에서 발생하는 오버피팅 현상을 완화하기 위하여 하이퍼스피어(hypersphere) 내에서의 에너지 변동을 활용하여 신경망의 뉴런 간 각도를 유지합니다. 이를 통해 모델의 표현 능력을 유지하면서도 사람의 선호에 잘 맞는 결과를 생성하도록 합니다. 특히, RoPO 방법은 단 0.0086%의 가중치 매개변수로도 우수한 성능을 발휘합니다.

- **Performance Highlights**: RoPO는 MT-Bench에서 DPO보다 최대 10점, AlpacaEval 2에서 최대 2.8점을 초과하는 성능을 보이며, 생성의 다양성도 평균 6점 증가시키는 결과를 보여줍니다.



### Identify As A Human Does: A Pathfinder of Next-Generation Anti-Cheat Framework for First-Person Shooter Games (https://arxiv.org/abs/2409.14830)
- **What's New**: 게임 산업의 성장은 눈부시지만, 온라인 게임에서의 부정행위는 게임 경험의 무결성에 중대한 위협이 되고 있습니다. HAWK라는 서버 측 FPS(First-Person Shooter)용 안티치트 프레임워크가 제안되어, 기존 솔루션의 한계를 극복하고자 합니다.

- **Technical Details**: HAWK는 기계 학습(machine learning) 기술을 활용하여 인간 전문가의 식별 과정을 모방하며, 여러 관점의 기능 및 잘 정의된 워크플로우를 포함합니다. 이는 플레이어의 관점, 통계적 성과 및 게임 감각과 성취 일관성을 분석하는 세 가지 주요 측면에 중점을 두고 있습니다. HAWK는 LSTM(Long Short-Term Memory) 및 주의 메커니즘(attention mechanism), 앙상블 학습(ensemble learning) 및 딥 러닝 네트워크(deep learning networks)를 사용하여 인간 식별 과정을 재현합니다.

- **Performance Highlights**: HAWK는 CS:GO에서 최대 84%의 재현율(recall)과 80%의 정확도(accuracy)를 기록하며, 기존의 안티치트 솔루션보다 뛰어난 성능을 보여줍니다. HAWK는 또한 기존 검사에서 회피한 치터를 식별할 수 있는 능력을 가지고 있습니다.



### ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback (https://arxiv.org/abs/2409.14826)
- **What's New**: 최근 도구 보강 대형 언어 모델(LLMs)의 발전이 주목받고 있으며, 이들은 다양한 외부 도구와 상호작용하여 최종 답변을 제공합니다. 본 논문에서는 실제 사용자 시나리오를 반영하기 위해 새로운 데이터 세트 MGToolBench를 구축하고, LLM의 작업 완수 및 지침 이행 능력을 향상시키기 위한 ToolPlanner라는 두 단계 강화 학습 프레임워크를 제안합니다.

- **Technical Details**: MGToolBench는 사용자 행동을 반영하기 위해 다단계 사용자 지침 메커니즘을 채택하여 구성되었습니다. ToolPlanner는 첫 번째 단계에서 감독 하에 세밀화(Supervised Fine-Tuning, SFT) 모델을 사용하여 각각의 지침에 대한 해결책 트리를 생성하고, 두 번째 단계에서는 작업 완료 및 지침 이행이라는 두 가지 메트릭으로 생성된 솔루션을 평가합니다. 또한 솔루션 경로 계획 메커니즘을 사용하여 ToolPlanner의 다단계 추론 과정을 안내합니다.

- **Performance Highlights**: 실험 결과 ToolPlanner는 SOTA 모델 대비 Match Rate 26.8%, Pass Rate 20.2%, Win Rate 5.6% 향상되었음을 보여줍니다. 사람 평가에 따르면, 다중 세분화 지침이 사용자 사용 습관과 더 잘 일치하는 것으로 확인되었습니다.



### Towards Real-world Deployment of NILM Systems: Challenges and Practices (https://arxiv.org/abs/2409.14821)
- **What's New**: 이번 논문에서는 에지(Edge)와 클라우드(Cloud)의 협업을 통해 비침습적 전력 모니터링(NILM)의 실제 적용 가능성을 향상시키기 위한 3단계 프레임워크를 제안합니다. 이는 기존 클라우드 전용 NILM 알고리즘의 높은 계산 비용과 서비스 지연 문제를 해결하려는 시도의 일환으로, 경량화된 NILM 모델과 딥러닝(deep learning) 기반 모델을 각각 에지와 클라우드에 구현합니다.

- **Technical Details**: 제안된 NILM 프레임워크는 데이터 수집, 모델 훈련, 시스템 배포의 전체 과정을 포함하는 실험을 통해 검증되었습니다. 논문에서는 Gunicorn과 NGINX를 통합한 NILM 전용 배포 계획을 설계하여 응답 시간과 부하 균형 문제를 해결했습니다. 또한, 경량 모델을 현지에 배포하여 데이터 전송 지연을 최소화하고 데이터 유출 위험을 줄이는 의의를 강조합니다.

- **Performance Highlights**: 제안된 프레임워크는 높은 분해 정확도를 달성할 수 있으며, 클라우드의 작업 부하와 통신 오버헤드를 크게 줄일 수 있음을 실험을 통해 입증하였습니다. 에지-클라우드 협업을 통해 정확한 성능과 저지연 응답을 동시에 실현할 수 있음을 발견하였고, NGINX와 Gunicorn의 조합이 Flask 프레임워크 단독 사용시보다 동시성 성능을 크게 향상시킨다는 결과를 도출했습니다.



### Past Meets Present: Creating Historical Analogy with Large Language Models (https://arxiv.org/abs/2409.14820)
- **What's New**: 이 논문은 역사적 유추(historical analogy) 획득(task) 연구에 초점을 맞추며, 대규모 언어 모델(Large Language Models, LLMs)을 활용한 유사한 역사적 사건의 검색 및 생성을 탐색합니다.

- **Technical Details**: 이 연구에서는 LLM을 기반으로 하여 역사적 유추를 획득하기 위한 검색(retrieval) 및 생성(generation) 방법을 제안하며, 생성 과정에서의 환각(hallucinations)과 고정관념(stereotypes)을 완화하기 위한 자가 반성(self-reflection) 방법도 제안합니다.

- **Performance Highlights**: 인간 평가 및 다차원 자동 평가를 통해 LLMs가 역사적 유추를 획득하는 데 일반적으로 좋은 잠재력을 가지고 있으며, 자가 반성 방법을 사용함으로써 모델의 성능을 더욱 개선할 수 있음을 보여줍니다.



### MobileVLM: A Vision-Language Model for Better Intra- and Inter-UI Understanding (https://arxiv.org/abs/2409.14818)
- **What's New**: 최근 VLM(비전-언어 모델)을 기반으로 한 모바일 AI 에이전트가 주목받고 있으며, 이를 위해 새로운 MobileVLM을 제안했습니다. MobileVLM은 모바일 도메인에 특화된 추가 전처리 단계를 포함하여 UI 이해력 향상을 목표로 합니다.

- **Technical Details**: MobileVLM은 내부 UI 이해와 외부 UI 이해를 위한 두 가지 추가 전처리 단계를 도입하였으며, 4개의 UI 기반 사전 훈련 작업을 정의했습니다. 이를 통해 모델은 세밀한 요소 인식 및 페이지 전환 행동을 보다 잘 인지할 수 있습니다. Mobile3M이라는 대규모 데이터세트를 구축하여 이 훈련을 지원하며, 3백만 개의 UI 페이지와 실제 사용자 행동 데이터로 구성된 방향 그래프 구조를 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, MobileVLM은 ScreenQA 및 다른 평가 데이터셋에서 기존의 SOTA VLM보다 각각 14.34%, 34.18% 향상된 성능을 보여주었습니다.



### VARADE: a Variational-based AutoRegressive model for Anomaly Detection on the Edg (https://arxiv.org/abs/2409.14816)
- **What's New**: 이 논문은 VARADE라는 새로운 경량 자가회귀 모델을 소개합니다. 이 모델은 산업 4.0의 복잡한 이상 탐지를 실시간으로 수행할 수 있도록 설계되었습니다.

- **Technical Details**: VARADE는 variational inference를 기반으로 하는 경량 자가회귀 프레임워크를 구현하여 수신 데이터 스트리밍 처리 시 지연을 최소화합니다. 이 모델은 convolutional layers와 ReLU 활성화 함수를 활용하여 미래의 시간 단계를 예측합니다.

- **Performance Highlights**: VARADE는 두 가지 다른 엣지 플랫폼에서 수행된 실험에서 다른 최신 경량 MTSAD 솔루션들과 비교하여 이상 탐지 정확도, 전력 소비 및 추론 빈도에서 최상의 트레이드오프를 달성했습니다.



### Research on Dynamic Data Flow Anomaly Detection based on Machine Learning (https://arxiv.org/abs/2409.14796)
- **What's New**: 현대 사이버 공격의 정교함과 다양성이 증가함에 따라 기존의 프록시, 게이트웨이, 방화벽 및 암호화된 터널을 단독 방어 전략으로 사용하는 것은 불충분하다는 점이 강조됩니다. 이 연구에서는 데이터 이상을 사전적으로 식별하는 방법에 대한 연구가 강조되며, 특히 불균형 데이터의 경우 최적의 탐지 효과를 나타내지 않던 기존 연구의 한계를 극복하고자 합니다.

- **Technical Details**: 이 연구에서는 동적 데이터 흐름에서 이상치를 식별하기 위해 비지도 학습 방법이 사용됩니다. 실시간 데이터에서 다차원 특성을 추출하고, 클러스터링 알고리즘(clustering algorithm)을 이용하여 데이터 패턴을 분석합니다. 이 과정을 통해 잠재적 아웃라이어(potential outliers)를 자동으로 식별할 수 있습니다. 유사한 데이터를 클러스터링함으로써 레이블이 없는 데이터에서도 정상 트래픽에서 크게 이탈하는 데이터 행동을 탐지할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 시나리오에서 이상 탐지의 높은 정확도를 보여줍니다. 특히 불균형 데이터의 맥락에서 강력하고 적응력이 뛰어난 성능을 발휘합니다.



### Do Large Language Models have Problem-Solving Capability under Incomplete Information Scenarios? (https://arxiv.org/abs/2409.14762)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 이 논문에서는 불완전한 정보 시나리오에서 대형 언어 모델(LLMs)의 문제 해결 능력을 평가하기 위해 BrainKing이라는 새로운 게임을 소개합니다. 이 게임은 'Who is undercover'와 'Twenty Questions'를 기반으로 하여 LLM이 제한된 예 또는 아니요 질문을 통해 목표 엔티티를 파악해야 합니다.

- **Technical Details**: BrainKing 게임은 세 가지 난이도(쉬움, 중간, 어려움)를 설정하여 LLM의 세계 지식, 역 발상 및 오류 탐지 능력을 평가합니다. 각 엔티티에 대해 최소 세 가지 개념을 포함하는 계층 개념 목록이 필요하며, LLM 참가자는 잘못된 정보 사이에서 정확한 답을 찾기 위해 최대 20개의 질문을 생성해야 합니다.

- **Performance Highlights**:  실험 결과, LLM은 BrainKing에서 불완전한 정보 시나리오에서의 정확도, 시작 난이도 및 잘못된 답변의 수가 LLM의 성능에 미치는 영향 등을 조사한 결과, LLM의 문제 해결 능력의 한계와 가능성을 확인할 수 있었습니다.



### VLM's Eye Examination: Instruct and Inspect Visual Competency of Vision Language Models (https://arxiv.org/abs/2409.14759)
- **What's New**: 본 논문은 VLM(Visual Language Models)의 시각 인식 능력을 측정하기 위한 eye examination 프로세스를 제안합니다. 기존의 VLM들이 다양한 벤치마크에서 향상된 성능을 보였으나, 이들이 이미지를 인식하는 방식에 대한 이해는 부족했습니다. 이를 위해 LENS라는 데이터셋을 도입하여 VLM의 시각 인식 절차에 따라 준비 상태를 점검하고 검사합니다.

- **Technical Details**: LENS 데이터셋은 색상, 형태, 의미론적 요소의 세 가지 기본 요소로 구성되어 있으며, 각 요소에 대한 질문과 진단 과제를 포함합니다. 모델은 해당 요소의 ready check를 거쳐 시험을 진행하며, 색상 민감도(Sensitivity Area of Color, SAC) 및 형태 민감도(Sensitivity Area of Shape, SAS) 등의 지표를 통해 평가됩니다. 검사 과정에서는 세 가지 시험이 포함되며, 각 단계에서 VLM의 성능이 점검됩니다.

- **Performance Highlights**: 검사 결과, VLM들은 서로 다른 색깔에 대해 다르게 반응하고, 특히 모든 VLM에서 초록색에 대한 민감도가 낮음을 확인했습니다. 형태에 대해서도 다양한 VLM 모델의 능력에 따라 차이가 나타났습니다. 이러한 결과는 시각 인식 성능 향상을 위한 VLM의 설계 및 입력 전처리에 도움을 줄 수 있습니다.



### UniBEVFusion: Unified Radar-Vision BEVFusion for 3D Object Detection (https://arxiv.org/abs/2409.14751)
Comments:
          6 pages, 4 figues, conference

- **What's New**: 본 논문에서는 4D 밀리미터-파(mmWave) 레이더와 비전 데이터를 통합한 Radar Depth Lift-Splat-Shoot (RDL) 모듈을 제안하여 깊이 예측 과정에 레이더 특유의 데이터를 포함시키고 시각적 Bird-Eye View (BEV) 특징의 품질을 개선했습니다. 또한 다양한 모드에서 BEV 특징을 추출하는 Unified Feature Fusion (UFF) 접근법을 소개했습니다.

- **Technical Details**: RDL 모듈은 Radar Cross-Section (RCS) 데이터를 깊이 예측 모듈에 추가하여 레이더 특유의 정보를 최대한 활용합니다. 이 연구는 시각적 모드 실패를 시뮬레이션하기 위해 Gaussian noise를 주입하는 새로운 Failure Test (FT) 실험을 개발하였으며, 이를 통해 다중 모달 모델의 강인성을 평가했습니다. 실험 데이터로는 View-of-Delft (VoD)와 TJ4D 데이터셋을 사용했습니다.

- **Performance Highlights**: 제안된 UniBEVFusion 네트워크는 TJ4D 데이터셋에서 기존의 최첨단 모델보다 3D 객체 탐지 정확도에서 1.44의 개선과 BEV 객체 탐지 정확도에서 1.72의 개선을 보였습니다. 이 결과는 레이더-비전 융합 모델의 성능 향상을 위한 새로운 접근법의 가능성을 보여줍니다.



### Distribution-Level Feature Distancing for Machine Unlearning: Towards a Better Trade-off Between Model Utility and Forgetting (https://arxiv.org/abs/2409.14747)
Comments:
          10 pages, 6 figures

- **What's New**: 본 연구에서는 'correlation collapse'라는 새로운 문제를 제기하고, 이를 해결하기 위한 새로운 방법인 Distribution-Level Feature Distancing (DLFD)을 제안합니다. 이 방법을 통해 특정 이미지를 효과적으로 잊으면서도 모델의 전반적인 성능을 유지할 수 있습니다.

- **Technical Details**: DLFD는 최적 운송 문제(Optimal Transport)를 활용하여, 잊어야 할 이미지 분포로부터 보존해야 할 이미지의 특징 분포를 이동시킵니다. 이 과정에서 생성된 데이터는 잊어야 할 데이터와 다른 분포를 가지도록 하여, 효율적인 이미지 잊기를 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해, 우리의 방법이 기존의 최신 기계 유학 방법들보다 우수하다는 것을 입증하였습니다. DLFD는 특히 얼굴 인식 데이터셋에서 뛰어난 성과를 보여주었습니다.



### Less yet robust: crucial region selection for scene recognition (https://arxiv.org/abs/2409.14741)
- **What's New**: 이 논문은 저품질 이미지에서 효과적인 장면 인식을 위한 새로운 적응형 선택 메커니즘을 제안합니다. 이를 통해 고수준의 의미 있는 특징을 가진 결정을 더욱 강조하여 성능 향상을 도모합니다.

- **Technical Details**: 제안된 방법은 CNN(Convolutional Neural Network)을 기반으로 하여, 중요한 고수준의 의미 특징이 있는 지역을 식별합니다. 학습 가능한 마스크를 네트워크에 구현하여 특징 행렬의 서로 다른 지역에 가중치를 부여하며, 중요 정규화 항을 추가하여 핵심 고수준 특징 지역의 중요성을 증가시킵니다. 이는 명확한 결정에 도움이 되는 저품질 이미지를 효과적으로 처리하는 방법입니다.

- **Performance Highlights**: 우리는 수중 지질 장면 분류 데이터셋을 구성하고, 제안된 방법이 최신 기술에 비해 우수성과 강건성을 입증하는 다양한 실험 결과를 보여줍니다.



### ToxiCraft: A Novel Framework for Synthetic Generation of Harmful Information (https://arxiv.org/abs/2409.14740)
- **What's New**: 이번 논문에서는 Toxicraft라는 새로운 프레임워크를 제안하여 유해한 콘텐츠를 식별하는 데 있어 데이터 부족 문제와 일관되지 않은 정의의 문제를 해결하고자 합니다.

- **Technical Details**: Toxicraft는 적은 양의 seed data를 사용하여 다양한 합성(synthetic) 유해 정보 예시를 생성할 수 있는 프레임워크입니다. 이 프레임워크는 독특하면서도 매우 현실적인 예시를 만들어내며, 이를 통해 분류 모델을 강화합니다.

- **Performance Highlights**: 여러 데이터셋을 통해 이루어진 실험에서 탐지 모델의 강건성(robustness)과 적응성(adaptability)이 향상된 것을 보여주며, gold labels에 근접하거나 이를 초월하는 성능을 기록했습니다.



### PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs (https://arxiv.org/abs/2409.14729)
- **What's New**: 본 논문에서는 PROMPTFUZZ라는 새로운 테스트 프레임워크를 제안하여, 대규모 언어 모델(LLM)이 Prompt Injection 공격에 대한 강건성을 체계적으로 평가할 수 있도록 하였습니다.

- **Technical Details**: PROMPTFUZZ는 소프트웨어 퍼징(fuzzing) 기법에서 영감을 받아, 유망한 seed prompt를 선택하고 다양한 Prompt Injection을 생성하여 LLM의 회복력을 평가합니다. PROMPTFUZZ는 준비 단계(prepare phase)와 집중 단계(focus phase)로 나뉘며, 준비 단계에서는 초기 seed와 few-shot 예제를 수집하고, 집중 단계에서는 수집된 예제를 사용해 다양한 고품질의 prompt injections를 생성합니다.

- **Performance Highlights**: PROMPTFUZZ를 사용하여 강력한 방어 prompt를 가진 LLM에서도 더 많은 취약점을 발견할 수 있었으며, 실제 대회에서 4000명 이상의 참가자 중 7위(상위 0.14%)를 기록하는 성과를 달성했습니다. 또한, Prompt Injection 공격에 대한 강건성을 향상시키기 위해 LLM을 미세 조정(fine-tuning)하기 위한 데이터셋을 구축하였습니다.



### EDSNet: Efficient-DSNet for Video Summarization (https://arxiv.org/abs/2409.14724)
Comments:
          10 pages, 5 figures

- **What's New**: 이 논문은 비디오 요약(video summarization) 방법의 비효율성을 해결하기 위해 Direct-to-Summarize Network (DSNet)를 개선하여 더 자원 효율적인 토큰 혼합(token mixing) 메커니즘을 도입했습니다. 전통적인 어텐션(attention) 기법을 Fourier 변환(Fourier), 웨이브렛 변환(Wavelet transforms), Nyströmformer로 대체함으로써 효율성과 성능을 향상시켰습니다. 또한 ROI 풀링(ROI pooling), Fast Fourier Transform 풀링, 평면 풀링(flat pooling)과 같은 다양한 풀링(pooling) 전략을 탐구했습니다.

- **Technical Details**: 본 연구에서는 비디오 프레임에서 주요 세그먼트를 식별하고 요약을 수행하기 위해 Temporal Region Proposal Network을 사용하며, DSNet 아키텍처를 수정하여 특징 추출(feature extraction) 및 영역 제안(region proposal) 네트워크의 효율성을 향상시킵니다. 각 프레임은 관심 제안으로 분류되며, Nyströmformer와 FNet 블록을 사용하여 자원 소모를 줄이고 대량의 비디오 데이터를 효율적으로 처리할 수 있습니다. 본 시스템은 나중에 정밀한 요약 생성을 위해 회귀(regression) 모델을 추가로 사용합니다.

- **Performance Highlights**: TVSum과 SumMe 데이터셋에 대한 실험 결과, 이 수정사항들은 계산 비용을 상당히 줄이면서도 경쟁력 있는 요약 성능을 유지함을 보여줍니다. 이는 특히 대용량 비디오 데이터 처리에 더욱 확장 가능한 솔루션을 제공하는 것으로 평가됩니다.



### ERABAL: Enhancing Role-Playing Agents through Boundary-Aware Learning (https://arxiv.org/abs/2409.14710)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2402.10618

- **What's New**: 새로운 연구에서 제안된 ERABAL 프레임워크는 역할을 맡은 언어 모델의 역할 연기 능력을 향상시키기 위한 경계 인식 학습(boundary-aware learning) 방법론을 소개합니다. 이는 역할 특화 대화 생성 파이프라인과 정렬 훈련 방법론을 포함합니다.

- **Technical Details**: ERABAL은 네 개의 모듈로 구성된 자동화된 데이터 생성 파이프라인을 도입하여 경계 인식 대화를 생성합니다. 이 모듈들은 대화 기획자(dialogue planner), 주제 관리자(topic manager), 경계 질의 생성기(boundary query generator), 응답 생성기(response generator)로 이루어져 있으며, 역할 속성에 기반한 특정 질문-응답 쌍을 생성합니다. 또, 경계 인식 선호 최적화(Boundary-aware preference optimization, BPO) 방법이 포함되어 있어 정교한 학습이 가능합니다.

- **Performance Highlights**: ERABAL은 일반적인 모델들과 비교하여 역할 일관성 평가에서 가장 우수한 성능을 보였습니다. 기존 역할 연기 벤치마크에 대한 실험을 통해, ERABAL은 10%의 훈련 대화만으로도 일반적 기준선 모델들에 비해 상당한 성능 향상을 이루어냈으며, WikiRoleEval, CharacterEval, MT-Bench 등에서 뛰어난 결과를 기록하였습니다.



### Target-Aware Language Modeling via Granular Data Sampling (https://arxiv.org/abs/2409.14705)
Comments:
          Accepted to EMNLP 2024 Main Conference, 9 pages, 6 figures, 3 tables

- **What's New**: 이 논문은 특정 도메인에서의 성능을 유지하면서도 데이터 샘플링의 최적화를 통해 언어 모델(ML) 사전 훈련의 효율성을 높이기 위한 새로운 접근 방식을 제안합니다. 특히, n-gram 기법을 활용하여 다중 그레인(멀티-그레인) 토큰으로 특징을 구성하는 중요 샘플링을 다시 다루고 있습니다.

- **Technical Details**: 저자들은 n-gram 토큰을 활용한 중요 샘플링 기법을 통해 데이터 선택성을 극대화하며, 이는 복잡한 모델이 아닌, 사전 훈련된 간단한 코어셋을 보여줍니다. 또한 다중 그레인 특성을 사용하는 새로운 알고리즘이 제안되었으며, 이를 통해 작업 지향적인 데이터 샘플링이 이루어지고 있습니다.

- **Performance Highlights**: 총 1%의 데이터로 학습된 모델이 전체 RefinedWeb 데이터에 맞먹는 성능을 발휘하는 결과를 보여주며, 랜덤 샘플링보다 뛰어난 성능을 나타냅니다. 다양한 모델 크기(125M부터 1.5B까지)에서 이과 같은 우수한 성능이 확인되었습니다.



### VLEU: a Method for Automatic Evaluation for Generalizability of Text-to-Image Models (https://arxiv.org/abs/2409.14704)
Comments:
          accepted by EMNLP2024(long paper,main conference)

- **What's New**: 텍스트-이미지(T2I) 모델의 평가 방법을 개선하기 위해 새로운 평가 지표인 VLEU(Visual Language Evaluation Understudy)를 소개합니다. 이 지표는 대규모 언어 모델을 사용하여 T2I 모델의 다양한 텍스트 프롬프트에 대한 일반화 능력을 정량적으로 평가할 수 있습니다.

- **Technical Details**: VLEU는 시각적 텍스트 도메인의 분포와 T2I 모델이 생성한 이미지의 조건부 분포 간의 Kullback-Leibler divergence를 계산하여 모델의 일반화 능력을 수치적으로 평가합니다. 이 지표는 다양한 텍스트 프롬프트에서 이미지의 생성 품질과 그 일치도를 평가하는 데 활용됩니다. LLM(대규모 언어 모델)과 CLIP 모델을 사용하여 텍스트와 이미지 간의 의미적 일치를 평가합니다.

- **Performance Highlights**: VLEU의 실험을 통해 다양한 T2I 모델의 일반화 능력을 효과적으로 평가할 수 있음을 입증하였습니다. 이 새로운 지표는 T2I 모델 개발에 필수적인 도구로 자리잡을 것으로 기대되며, 실제 사례 연구 또한 발표되어 그 유용성을 보여주었습니다.



### Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling (https://arxiv.org/abs/2409.14683)
- **What's New**: 이번 논문에서는 ColBERT와 같은 다중 벡터 검색 방법의 저장소 및 메모리 요구사항을 줄이기 위한 간단한 클러스터링 기반의 Token Pooling 방법을 도입했습니다. 이 방법은 저장해야 하는 벡터의 수를 획기적으로 줄여줍니다.

- **Technical Details**: Token Pooling 방법은 개별 벡터를 클러스터링하고 평균 풀링을 통해 하나의 벡터로 변환하는 2단계 시스템으로 작동합니다. 세 가지 풀링 방법이 제안되었으며, 특히 계층적 클러스터링이 가장 우수한 결과를 보여주었습니다.

- **Performance Highlights**: 이 방법은 ColBERT 인덱스를 평균적으로 50%의 저장 공간을 줄이면서도 성능 저하가 없음을 보여주었으며, 66%까지의 벡터 수 감소도 가능하였고, 이 경우 성능 저하는 3% 이하에 머물렀습니다.



### Quantifying Context Bias in Domain Adaptation for Object Detection (https://arxiv.org/abs/2409.14679)
Comments:
          Under review

- **What's New**: 본 연구는 객체 탐지에서의 도메인 적응(Domain Adaptation for Object Detection, DAOD)에서 컨텍스트 바이어스(context bias)를 분석하고, 이를 다양한 도메인에서 어떻게 활용할 수 있는지를 제안합니다. 특히 배경 특징의 변화가 적응 과정에서 미치는 영향을 분석하여, DAOD 접근법을 개선하기 위한 새로운 아이디어를 도출합니다.

- **Technical Details**: 저자들은 CARLA 데이터셋과 Cityscapes 데이터셋을 사용하여 배경 마스킹 및 모델의 다양한 층에서의 활성화 값을 변화시키는 실험을 통해 컨텍스트 바이어스를 정량화합니다. 다양한 계량지표인 최대 평균 불일치(Maximum Mean Discrepancy, MMD)와 최대 분산 불일치(Maximum Variance Discrepancy, MVD)를 사용하여, 서로 다른 도메인에서 조작된 배경 영역에 대하여 전경의 조건부 확률 추정을 시행합니다.

- **Performance Highlights**:  저자들은 YOLOv4 및 YOLOv8 모델을 사용한 실험을 통해, 배경의 변화가 차량 탐지에 미치는 영향을 분석하였으며, 배경과 전경의 강한 연관성을 드러내어, 다른 배경에서 차량을 탐지하는 데 필요한 시각적 정보가 결여되었음을 보였습니다. 이러한 결과는 DAOD 접근법에서 컨텍스트 바이어스를 고려하는 것이 모델의 일반화 및 견고성을 향상시키는데 필수적임을 강조합니다.



### Instruction Tuning Vs. In-Context Learning: Revisiting Large Language Models in Few-Shot Computational Social Scienc (https://arxiv.org/abs/2409.14673)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM)의 Instruction Tuning (IT)과 In-Context Learning (ICL)의 분류 성능을 비교하며, 실제적인 계산 사회과학(CSS) 작업에서의 중요성을 강조합니다. ICL이 대부분의 CSS 작업에서 IT보다 효과적이라는 실험 결과를 도출했습니다.

- **Technical Details**: 연구에서는 ICL과 IT를 사용하여 1-, 8-, 16-, 32-shot 설정에서 6개의 오픈 소스 LLM을 평가합니다. IT는 지도 방식으로 LLM의 파라미터를 업데이트하고, ICL은 특정 작업의 프롬프트(conditioning)를 이용하여 모델 가중치 업데이트 없이 작업을 수행하도록 합니다. 결과적으로 ICL이 IT보다 우수한 성능을 발휘할 뿐만 아니라, 샘플 수의 증가가 성능에 미치는 영향을 조사합니다.

- **Performance Highlights**: ICL은 5개의 CSS 작업에서 IT보다 평균적으로 더 높은 성능을 기록했습니다. 샘플 수를 단순히 증가시키는 것은 ICL이나 IT의 성능을 일관되게 향상시키지 못하며, 때때로 성능 저하를 초래할 수 있습니다. 또한, ICL 프롬프트가 제로샷(zero-shot) 및 Chain-of-Thought (CoT) 프롬프트보다 더 효과적임을 확인했습니다.



### zsLLMCode: An Effective Approach for Functional Code Embedding via LLM with Zero-Shot Learning (https://arxiv.org/abs/2409.14644)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)을 활용하여 기능적 코드 임베딩(functional code embeddings)을 생성하는 새로운 접근 방식인 zsLLMCode를 제안합니다. 이 방법은 LLMs를 사용하여 소스 코드를 간결한 요약으로 변환하고, 이를 전문 임베딩 모델을 통해 기능적 코드 임베딩으로 변환합니다.

- **Technical Details**: zsLLMCode는 LLMs의 제로샷 학습(zero-shot learning) 기능을 활용하며, 학습이 필요 없는 비지도 학습 방식입니다. 이 접근법은 복잡한 다운스트림 작업을 수행할 때 LLMs에서 발생할 수 있는 환각(hallucinations) 문제를 해결합니다. 또한 LLMs의 제약된 컨텍스트 길이가 더 큰 입력을 처리하는 데 어려움을 겪는 문제를 완화합니다.

- **Performance Highlights**: 실험 결과, zsLLMCode는 최신 비지도 학습 방법들보다 더 효과적이며 우수한 성능을 보여주었습니다.



### Not Only the Last-Layer Features for Spurious Correlations: All Layer Deep Feature Reweighting (https://arxiv.org/abs/2409.14637)
- **What's New**: 본 논문에서는 spurious correlation (불필요한 상관관계) 문제를 해결하기 위해 모든 레이어에서 특징을 추출하여 분류기를 재학습하는 새로운 방법인 Head2Toe를 제안합니다. 이전 연구에서 마지막 레이어만을 재훈련하는 방법보다 더 효과적으로 편향되지 않은 특징을 선택하여 성능 향상을 보여줍니다.

- **Technical Details**: Head2Toe는 신경망의 모든 레이어에서 특징을 활용하여 더 이식 가능한 특징을 발견하는 단순하면서도 효율적인 Transfer Learning(전이 학습) 방법입니다. 이 방법은 Deep Feature Reweighting (DFR)와 통합되어 불안정한 특징의 영향을 줄이며 최적의 성능을 달성하도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 여러 표준 벤치마크에서 worst-group accuracy(최악의 그룹 정확도) 향상을 보여주었으며, 이는 비슷한 접근 방식보다 더 높은 성과를 달성함을 의미합니다.



### Scideator: Human-LLM Scientific Idea Generation Grounded in Research-Paper Facet Recombination (https://arxiv.org/abs/2409.14634)
- **What's New**: Scideator는 기존 논문의 주요 요소를 혼합하여 새로운 아이디어를 생성하는데 도움을 주는 혁신적인 도구입니다. 사용자가 제공한 논문 세트에서 목적(purpose), 메커니즘(mechanisms) 및 평가(evaluations) 등 중요한 측면을 추출하여 사용자가 아이디어 스페이스(idea space)를 탐색할 수 있도록 지원합니다.

- **Technical Details**: Scideator는 네 가지 LLM(Blueprint) 기반 RAG(retrieval-augmented generation) 모듈을 도입합니다: Analogous Paper Facet Finder, Faceted Idea Generator, Idea Novelty Checker, Idea Novelty Iterator. 이 모듈들은 사용자에게 기존 문헌과의 겹침을 조사하고 자동화된 아이디어 참신성(novelty) 평가 및 설명을 제공하여 새로운 아이디어 생성 과정을 지원합니다.

- **Performance Highlights**: 사용자 연구에서 19명의 컴퓨터 과학 연구자들은 Scideator를 사용하여 강력한 기준선과 비교할 때 훨씬 더 흥미로운 아이디어를 찾아냈습니다.



### Hierarchical end-to-end autonomous navigation through few-shot waypoint detection (https://arxiv.org/abs/2409.14633)
Comments:
          Appeared at the 40th Anniversary of the IEEE International Conference on Robotics and Automation (ICRA@40), 23-26 September, 2024, Rotterdam, The Netherlands. 9 pages, 5 figures

- **What's New**: 이번 논문은 인간의 탐색 능력에 영감을 받아 모바일 로봇이 미리 알려지지 않은 환경에서 간단한 시각적 데이터와 높은 수준의 내비게이션 액션을 통해 탐색할 수 있도록 하는 새로운 계층적 메타 러닝 방식을 제안합니다. 이는 내비게이션 프로세스를 단순화하고 새로운 환경에 쉽게 적응할 수 있게 합니다.

- **Technical Details**: 제안된 시스템은 Description-based Navigation System (DNS)이라고 불리며, 고수준 내비게이션 액션을 지정하는 경우에만 필요한 제한된 시각적 데이터 결과를 결합하여 저수준 조작을 분리하는 계층적 엔드투엔드 내비게이션 시스템입니다. 이 시스템은 few-shot learning 기술을 적용하여 탐지 효과를 극대화하며, 웨이포인트(waypoint) 검출 시 로우 레벨 조작을 관리하는 멀티태스크 컨트롤러와 상호작용합니다.

- **Performance Highlights**: 작은 규모의 자율 차량에 대한 실험을 통해, 제안된 방안이 다양한 이전에 보지 못한 환경에서 효과적으로 작동함을 입증했습니다. 그 결과, 복잡한 로케이션 센서에 의존하지 않으면서도 안정적인 탐색 효과를 달성했습니다.



### EQ-CBM: A Probabilistic Concept Bottleneck with Energy-based Models and Quantized Vectors (https://arxiv.org/abs/2409.14630)
Comments:
          Accepted by ACCV 2024

- **What's New**: 최신 논문에서는 해석 가능한 AI 시스템에 대한 수요가 급증하면서, 인간이 이해할 수 있는 개념을 활용하여 해석 가능성을 향상시키는 개념 병목 모델(Concept Bottleneck Models, CBMs)에 대한 새로운 접근법인 EQ-CBM을 제안합니다.

- **Technical Details**: EQ-CBM은 확률적 개념 인코딩을 개선하기 위해 에너지 기반 모델(Energy-based Models, EBMs)과 양자화된 개념 활성화 벡터(Quantized Concept Activation Vectors, qCAVs)를 활용합니다. 이 방법은 확률적 개념을 캡처하여 예측의 신뢰성과 정확성을 향상시킵니다. 특히, EQ-CBM은 동질적인 벡터를 선택하여 개념을 인코딩함으로써 인간의 개입을 더 수월하게 만들어 주며, 다양한 이미지를 통해 작업 성능을 높입니다.

- **Performance Highlights**: 다양한 벤치마크 데이터세트를 사용한 실험 결과, EQ-CBM은 기존의 CBM 접근법보다 개념 및 작업 정확도에서 우수한 성능을 보여줍니다. 이를 통해 EQ-CBM은 EBMs와 qCAVs를 통한 확률적 접근법이 해석 가능성과 정확성을 모두 향상시킬 수 있음을 입증했습니다.



### LatentQGAN: A Hybrid QGAN with Classical Convolutional Autoencoder (https://arxiv.org/abs/2409.14622)
Comments:
          This paper was accepted for publication on the 10th IEEE World Forum on Internet of Things (IEEE WFIoT2024), in the session SS - QIoT-1: Special Session - Quantum Internet of Things (QIoT)-1, November 10th, from 14:00 to 15:30 EST

- **What's New**: 이 논문은 LatentQGAN이라는 새로운 양자 모델을 제안하여 합성 데이터 생성에서의 확장성과 모드 붕괴 문제를 해결하려고 합니다. LatentQGAN은 하이브리드 양자-고전적 GAN과 오토인코더를 결합하여 개발되었습니다.

- **Technical Details**: LatentQGAN은 이미지를 생성하기 위해 설계되었지만, 다양한 데이터 생성 작업에 대한 응용 가능성을 가지고 있습니다. 오토인코더를 통해 원본 데이터셋의 압축 표현을 학습함으로써 양자 회로의 효율성을 극대화하고 양자 리소스의 활용을 최소화합니다. 이를 통해 NISQ 컴퓨터의 한계를 극복할 수 있습니다.

- **Performance Highlights**: 실험 결과, LatentQGAN은 기존의 QGAN과 같은 수의 매개변수를 가진 고전적 모델들보다 우수한 성능을 보였으며, MNIST 데이터셋에서 훈련되었습니다.



### Can pre-trained language models generate titles for research papers? (https://arxiv.org/abs/2409.14602)
- **What's New**: 이 연구에서는 연구 논문의 초록으로부터 제목을 자동 생성하기 위해 사전 훈련된 대형 언어 모델을 미세 조정하는 방법을 제안합니다. 특히, T5-base, BART-base, PEGASUS-large 모델을 서브셋 LREC-COLING-2024 데이터셋을 이용하여 훈련시켰습니다. 또한, ChatGPT-3.5를 제로샷(zero-shot) 설정에서 사용하여 초기 제목을 생성해보았습니다.

- **Technical Details**: 이 연구에서 활용된 핵심 기술은 딥 뉴럴 모델(deep neural models)로, 여러 사전 훈련된 트랜스포머 모델을 미세 조정하여 논문 제목을 생성하는 데 중점을 두었습니다. 이는 'abstractive text summarization'의 특별한 경우로 볼 수 있으며, 주요 평가 지표로는 ROUGE, METEOR, MoverScore, BERTScore 및 SciBERTScore가 사용되었습니다. 연구팀은 LREC-COLING-2024라는 새로운 데이터셋을 수집하였고, 이 데이터셋은 논문의 초록과 제목의 쌍을 포함하고 있습니다.

- **Performance Highlights**: PEGASUS-large 모델이 선택된 메트릭에서 다른 모델들보다 뛰어난 성능을 보여주었습니다. 특히, PEGASUS-large는 LLM(GPT-3.5, LLaMA)들에 비해 파라미터 수가 적음에도 불구하고 우수한 성능을 발휘했습니다. 연구에서는 또한 사용자에게 다양한 언어 모델을 선택할 수 있는 데모를 제공하였으며, Hugging Face에 미세 조정된 모델과 LREC-COLING-2024 데이터셋을 공개했습니다.



### Testing Causal Models with Hidden Variables in Polynomial Delay via Conditional Independencies (https://arxiv.org/abs/2409.14593)
Comments:
          34 total pages, 14 figures

- **What's New**: 본 논문에서는 숨겨진 변수(hidden variables)를 가진 인과 그래프(causal graphs)에 대한 새로운 테스트 방법을 소개합니다. 특히, C-LMP(c-component local Markov property)를 도입하여 기존 방법보다 빠르게 조건 독립성(conditional independence, CI)을 테스트할 수 있는 알고리즘을 개발하였습니다.

- **Technical Details**: C-LMP는 숨겨진 변수가 포함된 인과 모델에서 조건 독립성 관계를 효율적으로 검증하기 위한 방법입니다. 기존에는 이들 관계를 확인하기 위해 지수적인 시간 복잡도를 가진 알고리즘이 필요했지만, 본 논문에서는 다항 시간(polynomial time) 내에 알맞은 CI를 나열할 수 있는 알고리즘을 제안합니다.

- **Performance Highlights**: 실제 데이터와 합성 데이터(synthetic data)에서의 실험 결과는 제안한 알고리즘의 실제 적용 가능성을 보여줍니다. C-LMP를 이용한 CI 테스트가 클 경우에도 효율적으로 수행될 수 있음을 입증하였습니다.



### Explainable AI needs formal notions of explanation correctness (https://arxiv.org/abs/2409.14590)
- **What's New**: 이 논문은 '설명 가능한 인공지능'(XAI) 분야가 기계 학습(ML) 분야의 품질 보증에 적합하지 않으며, 현재의 XAI 방법이 신뢰할 수 있는 동일한 예측 목표를 가지고 있지 않다는 점을 강조합니다. 연구자들은 문제를 명확하게 정의한 후 이를 해결하기 위해 적절한 방법을 설계해야 한다고 주장합니다.

- **Technical Details**: 현재의 XAI 방법들은 ML 모델, 훈련 데이터, 또는 특정 테스트 입력에 대한 중요한 질문에 신뢰성 있게 대답하지 못합니다. 이 논문에서는 두 가지 예시를 통해 XAI 방법이 예측 목표와 독립적인 입력 특성에 중요도를 잘못 할당하는 문제를 보여줍니다. 연구자들은 XAI 방법이 잘 정의된 문제를 다루어야 한다고 주장합니다.

- **Performance Highlights**: 현재 XAI 방법들은 모델 및 데이터의 유효성 검증, 모델 개선, 과학적 발견과 같은 중요한 목적을 수행하는 데 실패하고 있으며, 이로 인해 ML의 품질 보증 기능이 제한적입니다.



### Backtracking Improves Generation Safety (https://arxiv.org/abs/2409.14586)
- **What's New**: 이번 논문에서는 언어 모델 안전성을 위한 새로운 접근법인 'backtracking' 기법을 제안합니다. 이는 언어 모델이 올바르지 않은 생성 결과를 되돌리고 새로운 안전한 응답을 생성할 수 있도록 허용하는 기술입니다.

- **Technical Details**: 'backtracking'은 언어 모델이 생성 중에 특별한 [RESET] 토큰을 사용하여 이전의 안전하지 않은 생성 결과를 식별하고 이를 잊어버리면서 새로운 생성 작업을 시작하는 방식입니다. 본 연구는 SFT(Supervised Fine Tuning)와 DPO(Direct Preference Optimization) 방법론을 통해 훈련되었으며, 이를 통해 Gemma-2-2B와 Llama-3-8B 모델의 안전성을 크게 향상시켰습니다.

- **Performance Highlights**: Backtracking을 사용한 Llama-3-8B 모델은 기준 모델에 비해 안전성이 4배 증가했으며(6.1%에서 1.5%로), 유용성의 감소 없이도 이러한 안전성 향상이 이루어졌습니다. 추가로, 네 가지 적대적 공격에 대한 보호 기능도 제공되었습니다.



### Evaluating the Performance and Robustness of LLMs in Materials Science Q&A and Property Predictions (https://arxiv.org/abs/2409.14572)
- **What's New**: 본 연구는 소재 과학 분야에서의 Large Language Models (LLMs)의 견고성과 신뢰성에 대한 포괄적인 평가와 분석을 수행합니다. 학부 소재 과학 강의에서의 객관식 질문 세트, 다양한 강재 조성과 항복 강도 데이터셋, 그리고 밴드 갭 값이 포함된 데이터셋 등 세 가지 독특한 데이터셋을 사용하여 도메인 특화된 Q&A와 소재 속성 예측을 분석합니다.

- **Technical Details**: 연구에서는 zero-shot chain-of-thought, expert prompting, few-shot in-context learning (ICL) 등의 다양한 프롬프트 전략을 사용하여 LLMs의 성능을 평가하였습니다. 또한, 텍스트 순서 변경과 같은 텍스트 섭동이 LLM의 추론에 미치는 영향을 조사하며, 현실적인 섭동부터 적대적 섭동까지 다양한 유형의 섭동을 테스트하였습니다. 밴드 갭 예측에 대한 세부 조사를 통해 일부 상황에서는 섭동이 모델의 예측 능력을 향상시키는 경향이 나타났습니다.

- **Performance Highlights**: LLMs의 성능 평가는 MSE-MCQs 데이터셋에서 수행되었고, gpt-4-0613이 모든 카테고리에서 가장 높은 점수를 기록했습니다. 또한, 전통적인 랜덤 포레스트 회귀 모델과 비교하여, gpt-3.5-turbo-0613이 few-shot ICL을 활용하여 강재 항복 강도 예측에서 유사한 성능을 보여주었습니다. 이 연구는 소재 과학 분야에서 LLMs의 신뢰성 있는 사용에 대한 정보에 기반한 회의적인 시각을 제시하고, 이를 통한 견고성과 신뢰성 향상을 위한 발전을 촉진하고자 합니다.



### Combating Spatial Disorientation in a Dynamic Self-Stabilization Task Using AI Assistants (https://arxiv.org/abs/2409.14565)
Comments:
          10 pages, To be published in the International Conference on Human-Agent Interaction (HAI '24) proceedings

- **What's New**: 이번 연구는 조종사가 공간 착시 상태에서 균형을 유지하도록 돕기 위한 AI 에이전트의 가능성을 탐구합니다. AI 에이전트는 조종사가 시뮬레이션된 우주 비행 환경에서 물리적 피드백을 통해 스스로 균형을 맞추도록 안내합니다.

- **Technical Details**: 연구에서는 multi-axis rotation system (MARS)를 통해 실험 데이터를 수집하고, 이를 기반으로 다양한 reinforcement learning 및 deep learning 모델을 훈련시켜 '디지털 트윈'을 만들었습니다. 이 모델들은 조종사의 균형 회복을 위한 교정 cue를 제공하는 역할을 합니다.

- **Performance Highlights**: AI 어시스턴트는 조종사의 성과를 향상시키는 데 기여했으며, reinforcement learning 기반의 어시스턴트가 더 객관적으로 효과적이었지만 인간 사용자로부터는 상대적으로 낮은 신뢰를 받았습니다.



### RACOON: An LLM-based Framework for Retrieval-Augmented Column Type Annotation with a Knowledge Graph (https://arxiv.org/abs/2409.14556)
- **What's New**: 이 논문에서는 Column Type Annotation (CTA)에 대한 새로운 접근 방식을 제시합니다. LLM(대규모 언어 모델)을 활용한 CTA 기법을 개선하기 위해 Knowledge Graph (KG)를 사용하여 LLM에 제공되는 맥락 정보를 보강하는 방법을 보여줍니다.

- **Technical Details**: 제안된 방법은 RACOON이라는 프레임워크로, 사전 훈련된 파라메트릭 및 비파라메트릭 지식을 결합하여 LLM의 성능을 개선합니다. 이 과정에서는 다음과 같은 세 가지 단계가 포함됩니다: (1) 테이블에서 엔티티 언급을 확인하고 관련 정보를 KG에서 검색, (2) 검색된 내용을 처리하여 노이즈를 줄이기, (3) 최종적으로 LLM에 대한 프롬프트를 보강하기 위해 압축된 맥락을 직렬화합니다.

- **Performance Highlights**: RACOON 프레임워크는 기존 LLM 추론에 비해 최대 0.21 마이크로 F1 개선을 보였습니다. 실험 결과 RACOON이 다양한 시나리오와 검색 방법에서 일관되게 기존 LLM 추론을 능가하는 성능을 보여주었음을 확인할 수 있었습니다.



### Unleashing the Power of Emojis in Texts via Self-supervised Graph Pre-Training (https://arxiv.org/abs/2409.14552)
Comments:
          Accepted by EMNLP 2024 Main Conference

- **What's New**: 이 논문에서는 이모티콘과 텍스트 간의 관계를 개선하기 위해 포스트, 단어, 이모티콘으로 구성된 이종 그래프(heterogeneous graph)를 구축했습니다. 또한, 텍스트와 이모티콘의 공동 모델링을 위한 그래프 프리트레인 프레임워크(graph pre-train framework)를 제안하며, 이를 통해 텍스트와 이모티콘 사이의 상호작용을 더 잘 이해할 수 있도록 합니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 자가 지도 그래프 사전 훈련 작업(self-supervised graph pre-training tasks)을 포함합니다: 1) 노드 수준 그래프 대비 학습(node-level graph contrastive learning), 2) 엣지 수준 링크 재구성 학습(edge-level link reconstruction learning). 이 방식을 통해 포스트, 이모티콘, 단어 간의 상호작용을 모델링하고, 이를 다양한 다운스트림 작업에서 활용 가능합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Xiaohongshu 및 Twitter 데이터셋에서 다양한 다운스트림 작업(예: 인기 예측 및 감정 예측)에 대해 이전의 강력한 기준 모델보다 2%에서 10% 더 나은 성능을 보였습니다. 이 모델은 추가적으로 이모티콘 생성 작업에도 활용할 수 있습니다.



### TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps (https://arxiv.org/abs/2409.14543)
Comments:
          Research report

- **What's New**: 이 논문에서는 TrackNet 계열의 최신 모델인 TrackNetV4를 소개합니다. 이는 고급 시각적 특징과 학습 가능한 모션 주의 맵(motion attention maps)을 융합한 모션 인식 융합 메커니즘을 통해 향상되었습니다.

- **Technical Details**: TrackNetV4는 프레임 차분 맵(frame differencing maps)과 모션 프롬프트 레이어(motion prompt layer)를 활용하여 이동하는 공의 위치를 강조하고, 이로 인해 추적 성능이 개선됩니다. 이 과정에서 모델은 시각적 특징에만 의존하지 않고, 모션 정보를 명시적으로 통합하여 정밀한 추적 및 궤적 예측(trajectory prediction)에 유리합니다.

- **Performance Highlights**: 테니스 공과 셔틀콕 데이터셋을 기반으로 한 실험 결과, TrackNetV4는 TrackNetV2와 V3의 추적 성능을 개선하는 것으로 나타났습니다.



### A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders (https://arxiv.org/abs/2409.14507)
- **What's New**: 이번 연구에서는 Sparse Autoencoders (SAEs)를 활용하여 대형 언어 모델(LLMs)의 활성화를 인간이 해석할 수 있는 잠재 공간으로 분해하는 새로운 접근 방식에 대해 다룹니다. 특히, SAEs가 명확한 의미를 지닌 잠재 요소를 추출할 수 있는 정도와 sparsity 또는 SAE의 크기 변화가 명확한 의미성 및 해석 가능성에 미치는 영향을 탐구하였습니다.

- **Technical Details**: 연구는 첫 번째 문자 식별 작업을 통해 진행되었으며, 토큰에 대한 진리 레이블을 완전히 사용할 수 있는 환경에서 실험되었습니다. 이 과정에서 우리는 ‘feature absorption’이라 부르는 문제적 형태의 기능 분할을 확인하였고, 이는 잠재 요소가 인간이 해석할 수 있는 개념을 추적하는 것처럼 보이지만, 특정 토큰에서 예상대로 활성화되지 않는 현상입니다. 또한, SAE의 크기 및 sparsity를 변화시키는 것이 이 문제를 해결하기에 불충분하다는 점을 밝혔습니다.

- **Performance Highlights**: 실험 결과, 처음 문자 분류 작업에서 SAE latents의 정밀도 및 재현율이 선형 프로브보다 상당히 저조하다는 것을 발견하였습니다. 또한, 동일한 기능을 분류하는 것처럼 보이는 latents 간에 정밀도 및 재현율에 큰 차이가 존재하며, 이는 주로 sparsity와 SAE의 폭에 의해 매개된다는 것을 확인했습니다. 우리가 확인한 ‘feature absorption’ 현상은 SAE를 실제 애플리케이션에 활용하는 데 장애가 될 수 있으며, 이러한 잠재 요소들이 신뢰할 수 없는 분류기일 수 있음을 시사합니다.



### TabGraphs: A Benchmark and Strong Baselines for Learning on Graphs with Tabular Features (https://arxiv.org/abs/2409.14500)
- **What's New**: 이 연구에서는 탭형 데이터(tabular data)에서 이종 노드 특성을 가진 다양한 그래프의 벤치마크인 TabGraphs를 제안하고, 그래프 정보를 활용하여 예측 성능을 향상시키는 방법을 탐구합니다.

- **Technical Details**: TabGraphs 벤치마크에는 유사한 행동을 보이는 사용자들 간의 연결, 웹사이트 간의 트래픽 및 제품의 자주 함께 구매되는 관계와 같은 외부 정보를 기반으로 한 그래프 구조가 포함되어 있습니다. 이를 통해 다양한 데이터 도메인과 관계 유형을涵盖하며, 여러 머신러닝 모델을 테스트합니다.

- **Performance Highlights**: 그래프 신경망(GNNs)은 탭형 데이터에 대한 예측 성능이 향상되지만, 표준 탭형 모델도 그래프 데이터와 결합하여 사용하면 경쟁력을 가질 수 있습니다. 특히, 그래프 정보로 보강된 표준 GNN은 최근 제안된 이종 노드 특성을 가진 그래프 전용 모델보다 뛰어난 성능을 보였습니다.



### Thought-Path Contrastive Learning via Premise-Oriented Data Augmentation for Logical Reading Comprehension (https://arxiv.org/abs/2409.14495)
- **What's New**: 이번 논문에서는 Premise-Oriented Data Augmentation (PODA) 프레임워크를 제안하여 Chain-of-Thought (CoT) 합리화를 통해 올바른 답변 뿐만 아니라 잘못된 선택지에 대한 분석을 포함하고, 잘못된 후보 옵션으로부터 다양한 고품질의 반사실적 데이터(countersfactual context)를 자동으로 구축합니다.

- **Technical Details**: PODA는 올바른 및 잘못된 선택지에 대한 분석을 포함한 CoT 합리화를 생성하며, 각 선택지에 대한 요약 및 식별을 통해 반사실적 맥락을 구축합니다. Thought-Path Contrastive Learning (TPCL) 방법은 원본 및 반사실적 샘플 간의 사고 경로(thought-path)를 비교하여 모델의 논리적 추론 능력을 향상시키게 합니다. 구체적으로, 관계는 지지, 모순, 무관으로 분류되며, 이를 통해 다양한 반사실적 샘플을 생성합니다.

- **Performance Highlights**: 세 가지 대표적인 LLM(대형 언어 모델) 테스트에서 제안된 PODA와 TPCL 방법은 두 가지 논리적 MRC 벤치마크(ReClor 및 LogiQA 2.0)에서 기초 성능을 상당히 개선한 결과를 보여주었습니다.



### Enhancing LLM-based Autonomous Driving Agents to Mitigate Perception Attacks (https://arxiv.org/abs/2409.14488)
- **What's New**: 최근에 Autonomous Driving (AD) 시스템과 Large Language Models (LLMs)의 통합에 대한 관심이 증가하고 있습니다. 하지만 이들 AD 시스템은 object detection and tracking (ODT) 기능에 대한 공격에 취약합니다. 이 연구는 LLM Agents가 ODT 공격에 얼마나 취약한지를 평가하고, 이를 해결하기 위한 새로운 에이전트 Hudson을 소개합니다.

- **Technical Details**: Hudson은 AD 소프트웨어를 수정하여 드라이빙 장면의 실시간 인식 결과와 맥락 정보를 수집합니다. 이 데이터는 domain-specific language (DSL)로 형식화되고, 안전한 제어 결정을 내리기 위해 LLM에 안내됩니다. Hudson은 DSL을 자연어로 변환하고, 공격 탐지 지침 목록과 함께 제공합니다.

- **Performance Highlights**: Hudson은 GPT-4 및 두 개의 오픈 소스 LLM (Llama, Gemma)에 대해 평가되었고, 공격 탐지 정확도는 각각 83.3%, 63.6%, 73.6%를 기록했습니다. 그 결과, 안전한 제어 결정을 내린 비율은 각각 86.4%, 73.9%, 80%로 나타났습니다. LLM의 강점을 확인하고, ODT 공격 탐지 및 완화 가능성을 강조합니다.



### SynBench: A Synthetic Benchmark for Non-rigid 3D Point Cloud Registration (https://arxiv.org/abs/2409.14474)
- **What's New**: 본 논문에서는 SynBench라는 새로운 비강체(Non-rigid) 포인트 클라우드 등록 데이터셋을 소개합니다. 이 데이터셋은 Soft body 시뮬레이션을 위한 도구인 SimTool을 사용하여 생성되었으며, 다양한 도전 과제를 포함하여 포인트 클라우드 등록 방법을 공정하게 평가할 수 있는 기준점을 제공합니다.

- **Technical Details**: SynBench는 다양한 변형(deformation) 수준, 노이즈(noise), 아웃라이어(outlier), 및 불완전성(incompleteness)을 포함한 여러 가지 도전 과제를 제공합니다. 각 데이터 샘플은 변형 전후의 대응하는 포인트에 대한 Ground Truth 정보를 제공하여 등록 방법의 정확성을 평가하는 데 유용합니다. 데이터셋은 30개의 원시 객체(primitive objects)와 그에 따른 5개의 주요 카테고리로 구성되어 있습니다.

- **Performance Highlights**: SynBench는 기존 데이터셋에 비해 세 가지 특성을 가지고 있습니다: (1) 비강체 포인트 클라우드 등록을 위한 다양한 도전 과제를 제공하는 최초의 벤치마크, (2) 다양한 난이도의 도전 과제를 포함, (3) 변형 전후의 대응 포인트에 대한 Ground Truth를 포함. 이를 통해 향후 비강체 포인트 클라우드 등록 방법의 성능을 공정하게 비교할 수 있습니다.



### Exploring Multilingual Probing in Large Language Models: A Cross-Language Analysis (https://arxiv.org/abs/2409.14459)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 probing 기술을 다국어 환경으로 확장하여 다양한 언어에 대한 모델의 행동을 조사했습니다. 이는 대부분 영문 데이터에 초점을 맞추었으나 이제는 16개 언어에서 고급 언어와 저급 언어 간의 성능 차이를 분석합니다.

- **Technical Details**: 우리는 decoder-only LLMs를 사용하여 각 모델의 여러 레이어에서 hidden states를 추출하고 linear classifier probing을 통해 정보 인코딩 방식을 연구했습니다. 적용된 probing 기법은 다국어 컨텍스트에서 LLM의 사실적 지식 및 감성 분류 작업 수행 능력을 평가하는 데 중점을 두었습니다.

- **Performance Highlights**: 주요 발견으로는 고급 언어는 저급 언어에 비해 consistently higher probing accuracy를 나타내며, 고급 언어는 모델의 깊은 레이어에서 유의미한 정확도 향상을 보이는 반면, 저급 언어는 상대적으로 안정적인 성능을 유지했습니다. 또한, 고급 언어 간 probing vector의 유사성은 높은 반면, 저급 언어는 서로 및 고급 언어와의 유사성이 낮다는 점이 발견되었습니다.



### Detection of pulmonary pathologies using convolutional neural networks, Data Augmentation, ResNet50 and Vision Transformers (https://arxiv.org/abs/2409.14446)
Comments:
          10 pages

- **What's New**: 이 논문에서는 폐 질환 진단을 위한 새로운 방법을 제안합니다. 이 방법은 Convolutional Neural Networks (CNN)와 Data Augmentation, ResNet50 및 Vision Transformers (ViT)를 기반으로 하여 의료 이미지를 분석합니다.

- **Technical Details**: 제안된 방법은 다양한 폐 질환을 가진 환자의 X-ray 및 CT 스캔 이미지를 포함하는 데이터셋을 사용합니다. 성능 평가에는 Accuracy, Sensitivity, Specificity 및 ROC curve 아래 면적과 같은 평가 지표가 포함됩니다.

- **Performance Highlights**: 제안된 방법은 모든 성능 지표에서 기존 방법들보다 우수한 성능을 보였으며, 정확도는 98%, ROC curve 아래 면적은 99%에 달합니다. 이를 통해 제안된 방법이 의료 이미지를 이용한 폐 질환 진단에 효과적이고 유망한 도구임을 결론짓습니다.



### A Visualized Malware Detection Framework with CNN and Conditional GAN (https://arxiv.org/abs/2409.14439)
Comments:
          7 pages, 2022 IEEE International Conference on Big Data (Big Data), 2022

- **What's New**: 이번 연구에서는 Malware(멀웨어) 탐지를 위한 통합된 프레임워크를 제안합니다. 이는 Pictorial Representation System(PRS)을 포함하며, Conditional Generative Adversarial Network(cGAN) 기반의 이미지 증강 모델과 Convolutional Neural Networks(CNN) 기반의 탐지 모델로 구성되어 있습니다. 이러한 접근은 다양한 시나리오에서 효과적으로 작용합니다.

- **Technical Details**: 제안된 접근법은 4단계로 이루어져 있습니다: 1) 소수 클래스(악성)의 예제를 생성하기 위해 synthetic minority oversampling technique(SMOTE) 방식으로 표 형식 데이터를 준비하고 증가시키기, 2) 이진 이미지를 통해 훈련 및 테스트 데이터셋을 PRS를 통해 변환하기, 3) CNN 모델을 구축하여 이미지 분류 및 탐지 성능을 균형 있게 비교하기, 4) cGAN 모델을 설계하여 데이터 불균형 문제를 해결하기 위한 인위적인 이미지를 생성하는 것입니다. 이 과정은 TensorFlow와 Keras를 사용하여 실용적으로 평가되었습니다.

- **Performance Highlights**: 제안된 모델은 원본 및 인공 데이터셋에서 각각 98.51% 및 97.26%의 높은 정확도를 기록하며, 높은 컴퓨팅 성능을 기반으로 한 효율적인 Malware 탐지 시스템과 성공적인 시각화 시스템 및 이미지 증강 방법을 보여주었습니다.



### Automotive innovation landscaping using LLM (https://arxiv.org/abs/2409.14436)
Comments:
          9pages, 4Figures, 1 Flow chart

- **What's New**: 본 연구는 Prompt Engineering을 기반으로 한 특허 정보 추출 방법을 소개하며, 이 방법은 자동차 혁신의 경관을 조성하는 데 중요한 역할을 합니다. 기존의 수작업을 통한 방식에서 벗어나, 대형 언어 모델을 활용하여 보다 빠르고 효율적인 특허 분류 및 아이디어 추출을 가능하게 합니다.

- **Technical Details**: Prompt Engineering은 LLM(대형 언어 모델)과의 상호작용을 최적화하며, BERT와 같은 모델을 통해 여러 NLP(자연어 처리) 작업을 지원합니다. 이 연구에서는 OpenAI를 이용하여 TRIZ(Theory of Inventive Problem Solving) 모순을 추출하고, Transformer 기반 LLM을 활용해 특허에서 기술적 문제, 해결책, 이점 등을 식별하는 방법론을 다룹니다.

- **Performance Highlights**: 이 연구의 결과는 열린 특허 데이터셋을 사용하여 연료 전지 기술의 경관을 구성하는 방법을 보여줍니다. 이는 특허 문서의 복잡한 가독성 문제를 해결하고, 보다 빠르고 효율적인 정보 추출을 가능하게 하여 R&D 팀에 귀중한 통찰력을 제공합니다.



### Pomo3D: 3D-Aware Portrait Accessorizing and Mor (https://arxiv.org/abs/2409.14430)
- **What's New**: 본 논문에서는 Pomo3D라는 3D 초상화 조작 프레임워크를 제안한다. 이 프레임워크는 초상화와 액세서리를 분해하고 다시 조합하여 자유로운 액세서리 추가를 가능하게 하며, 다양한 액세서리를 동시에 착용한 모습을 제공한다. 기존의 방법들이 제시하는 제약을 뛰어넘어 보다 명확한 조작을 가능케 한다.

- **Technical Details**: Pomo3D의 주요 특징은 두 개의 별도 장면 표현을 도입한 것이다. 하나는 초상화용이고, 다른 하나는 덜 일반적인 액세서리용이다. 이러한 구조를 통해 High-resolution RGB 이미지를 생성하며, 'Scribble2Accessories' 모듈을 통해 사용자 그리기 스케치를 바탕으로 3D 액세서리를 생성할 수 있다. 또한, 편향을 완화하기 위해 'bias-conscious mapper'를 설계하여 실제 데이터셋에서 발견되는 편향된 연관성을 줄인다.

- **Performance Highlights**: Pomo3D는 액세서리 조작에서 제공하는 자유도와 포괄적인 초상화 편집 옵션으로 기존의 3D 생성 모델 중 가장 높은 수준의 편집 가능성을 가지고 있다. 사용자 그리기 스케치로부터 3D 액세서리를 생성할 수 있으며, 액세서리와 초상화를 별개로 모델링하여 조합할 수 있어 다양한 스타일의 초상화를 생성할 수 있다.



### Challenging the Performance-Interpretability Trade-off: An Evaluation of Interpretable Machine Learning Models (https://arxiv.org/abs/2409.14429)
Comments:
          Accepted for publication in Business & Information Systems Engineering (2024)

- **What's New**: 본 연구에서 제안된 새로운 세대의 일반화된 가법 모델(Generalized Additive Models, GAMs)은 복잡하고 비선형 패턴을 포착하면서도 완전한 해석 가능성을 유지하는 특성을 보여줍니다. 이 모델들은 전통적으로 예측 성능이 우수하다고 여겨지는 블랙박스 모델들과 비교할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구에서는 20개의 표 형식 벤치마크 데이터셋을 기반으로 7개의 다른 GAM과 7개의 일반적으로 사용되는 기계 학습 모델을 비교하였습니다. 이를 위해, 68,500번의 모델 실행을 포함한 광범위한 하이퍼파라미터 검색과 교차 검증이 수행되었습니다. 또한 모델의 해석 가능성을 평가하기 위해 모델의 시각적 출력을 정성적으로 분석하였습니다.

- **Performance Highlights**: 연구 결과, 모델의 예측 성능과 해석 가능성 간에는 엄격한 상충 관계가 없음을 보여주었습니다. 즉, 블랙박스 모델만이 높은 정확성을 달성할 수 있다는 오해를 불식시켰습니다. 또한, 정보 시스템 분야에서 GAMs의 중요성에 대해 논의하고, 사회 기술적 관점에서 향후 연구를 위한 시사점을 도출하였습니다.



### Dormant: Defending against Pose-driven Human Image Animation (https://arxiv.org/abs/2409.14424)
- **What's New**: 본 논문에서는 pose-driven human image animation 기술에 대한 새로운 방어 방법인 Dormant를 제안하여, 개인의 초상권과 프라이버시를 보호하는 효과적인 방어 메커니즘을 제공합니다.

- **Technical Details**: Dormant는 입력된 이미지를 보호하기 위해 protective perturbation을 적용하여 시각적인 유사성을 유지하면서도 저화질 비디오 생성을 유도합니다. 이 perturbation은 appearance feature의 비정상적인 추출을 초래하고 생성된 비디오 프레임 간의 일관성을 깨뜨리도록 최적화됩니다. 연구진은 8가지 animation 방법과 4개의 데이터셋을 포함하여 광범위한 평가를 수행하였습니다.

- **Performance Highlights**: Dormant는 6개의 기존 보호 방법에 비해 우수한 성능을 보이며, 생성된 비디오에서의 정체성 불일치, 시각적 왜곡, 눈에 띄는 아티팩트 및 일관성 결여를 초래합니다. 또한 Dormant는 6가지 현실 세계 상업 서비스에서도 효과적으로 작동하여, 다양한 생성 방법에 대한 방어 능력을 보여줍니다.



### COSBO: Conservative Offline Simulation-Based Policy Optimization (https://arxiv.org/abs/2409.14412)
- **What's New**: 새로운 연구에서는 Offline Reinforcement Learning (Offline RL) 모델 학습을 위해 불완전한 시뮬레이션 환경과 목표 환경의 데이터를 결합하는 새로운 방법을 제안합니다. 이 방법은 시뮬레이터로부터 얻어진 데이터를 효과적으로 활용하여 Offline RL의 성과를 향상시킵니다.

- **Technical Details**: COSBO라는 새로운 Simulation-based Offline RL 알고리즘은 다양한 동적 환경에서 롤아웃(rollout)을 통해 생성된 out-of-support state-action 튜플에 대해 가치 함수(value function)를 정규화하여 보수적인 가치 함수 추정을 수행합니다. 이전 방법들과 달리 COSBO는 모델 학습이 필요하지 않으며, 주어진 데이터의 분포 밖에서도 보다 일반화된 가치 함수 추정을 가능하게 합니다.

- **Performance Highlights**: COSBO는 CQL, MOPO, COMBO와 같은 최신 기술들과 비교했을 때 특히 다양한 동적 시나리오에서 우수한 성능을 발휘했습니다. 실험 결과, COSBO는 D4RL 벤치마크와 실제 로봇 환경에서 높은 보상을 주는 정책을 학습하는 데 성공적이었습니다.



### Beyond Persuasion: Towards Conversational Recommender System with Credible Explanations (https://arxiv.org/abs/2409.14399)
Comments:
          Findings of EMNLP 2024

- **What's New**: 본 논문에서는 CRS(Conversational Recommender System)의 설명에서 신뢰성을 높이기 위한 새로운 접근법인 PC-CRS를 제시합니다. 이는 사용자의 수용력을 높이고 장기적인 신뢰를 구축하는 것을 목표로 합니다.

- **Technical Details**: PC-CRS는 전략 기반 설명 생성(Strategy-guided Explanation Generation)과 반복적인 설명 정제(Iterative Explanation Refinement)의 두 단계로 구성됩니다. 이는 Credibility-aware Persuasive Strategies를 활용하여 신뢰성 있는 정보를 포함한 설명을 생성하고, 이후 후보 설명을 수정하여 잘못된 정보를 제거합니다.

- **Performance Highlights**: 실험 결과에 따르면, PC-CRS는 기존 최적 기준선에 비해 평균 8.17%의 신뢰성 점수 향상과 5.07%의 설득력 점수 향상을 달성하였습니다. 신뢰성 있는 설명이 추천의 정확도를 개선하는데 기여한다는 추가 분석도 포함되어 있습니다.



### Sparse Low-Ranked Self-Attention Transformer for Remaining Useful Lifetime Prediction of Optical Fiber Amplifiers (https://arxiv.org/abs/2409.14378)
Comments:
          9 pages, 7 figures, submitted to IEEE Transactions on Machine Learning in Communications and Networking (TMLCN)

- **What's New**: 이 논문에서는 Sparse Low-ranked self-Attention Transformer (SLAT)을 이용하여 광섬유 증폭기의 잔여 유효 수명(remaining useful lifetime, RUL) 예측을 위한 새로운 방법을 제안합니다. 이 방법은 고급 데이터 기반 예측 기법으로, 시스템 고장을 조기에 예측할 수 있도록 돕습니다.

- **Technical Details**: SLAT는 인코더-디코더 아키텍처를 기반으로 하며, 두 개의 병렬 인코더가 센서 및 시간 단계에 대한 피처를 추출합니다. 자기 주의(self-attention) 메커니즘을 활용하여 긴 시퀀스에서 장기 종속성을 학습할 수 있으며, 주의(Attention) 행렬의 희소성(sparsity) 구현과 저순위(parametrization) 파라미터화(low-rank)로 과적합(overfitting)을 줄이고 일반화 능력을 증가시킵니다.

- **Performance Highlights**: SLAT는 EDFA(Erbium-Doped Fiber Amplifier)를 포함한 광섬유 증폭기에 대한 실험적 응용에서 기존의 최첨단 방법들을 초월하여 우수한 성능을 보였습니다.



### Evaluating the Quality of Code Comments Generated by Large Language Models for Novice Programmers (https://arxiv.org/abs/2409.14368)
- **What's New**: 본 연구는 GPT-4, GPT-3.5-Turbo, Llama2와 같은 대형 언어 모델(LLM)이 생성한 코드 주석의 교육적 효과를 평가했습니다. LLM들이 생성한 코드 주석의 품질은 초보 프로그래머에게 적합하여, 전문가가 작성한 주석과 비교할 수 있습니다. 특히 GPT-4는 초보자에게 보다 지원적인 수단으로 평가되었습니다.

- **Technical Details**: 연구는 LeetCode에서 수집한 '쉬움' 수준의 Java 솔루션 데이터 세트를 사용하여, 각 LLM의 생성 주석이 초보 프로그래머의 이해를 돕는지 평가했습니다. 주석 품질 평가는 명확성, 초보자 친화성, 개념 설명 및 단계별 안내와 같은 여덟 가지 기준을 사용한 코드북에 기반하여 이루어졌습니다.

- **Performance Highlights**: 분석 결과 GPT-4는 전문가가 작성한 주석과 비교해도 유사한 품질을 보여주었으며, 복잡성에 대한 논의에서 Llama2를 능가했습니다. 또한, GPT-4는 GPT-3.5와 Llama2에 비해 초보자에게 더욱 지원적이라는 통계적 차이를 보였습니다.



### Data-Driven Spatiotemporal Feature Representation and Mining in Multidimensional Time Series (https://arxiv.org/abs/2409.14327)
- **What's New**: 이번 논문에서는 다차원 시계열 데이터 분석을 위한 새로운 방법론을 제시합니다. 이 방법은 기존의 전통적인 데이터 마이닝 기법의 한계를 극복하여 다차원 시계열 데이터를 효과적으로 처리하는 것을 목표로 합니다.

- **Technical Details**: 논문에서는 다차원 시계열(Multidimensional Time Series, MTS)을 일차원 사건 시퀀스로 변환하는 새로운 시공간(feature representation) 표현 방법을 소개합니다. 이를 통해 공간적으로 변화하는 사건을 변환하고, 사건 기호를 사용하여 시퀀스 내의 다차원 연결(Spatial Structural Information)을 나타냅니다. 또한, 변동 길이 튜플 마이닝(variable-length tuple mining) 방법을 도입하여 사건 시퀀스에서 비중복(non-redundant) 키 사건 하위 시퀀스를 추출합니다.

- **Performance Highlights**: STEM 모델의 우수한 성능은 다양한 모션 시퀀스에 대한 패턴 분류 실험을 통해 입증되었으며, 이 연구 결과는 인간 행동 패턴의 이해와 예측을 위한 중요한 이론적 기초 및 기술적 지원을 제공합니다.



### Unveiling Narrative Reasoning Limits of Large Language Models with Trope in Movie Synopses (https://arxiv.org/abs/2409.14324)
Comments:
          EMNLP 2024 Findings. The first two authors contributed equally. Code: this https URL

- **What's New**: 이 연구는 캐릭터의 전형을 포함하는 영화의 줄거리를 사용하여 최신 대형 언어 모델(LLMs)의 추상적 추론 능력을 평가하였습니다. 특히, CoT(Chain-of-Thought) 프롬프트 방법을 사용할 때 내러티브 추론에서의 낮은 성능을 드러냈습니다. 이를 해결하기 위해 전형별 쿼리 방식을 도입하여 성능을 11.8포인트 향상시켰습니다.

- **Technical Details**: 이 연구는 영화 줄거리에 포함된 전형(trope)을 사용하는 최초의 LLM 분석입니다. CoT가 적용된 경우에도 GPT-4와 같은 모델이 전형 이해 작업에서 무작위 추측 수준의 성능을 보이는 반면, 전형별 쿼리 방식은 성능을 획기적으로 높였습니다. 또한, Adversarial Injection 기법을 통해 LLM이 전형 관련 텍스트 토큰에 대한 민감성을 가지게 될 수 있음을 발견했습니다.

- **Performance Highlights**: 이 연구에서 제시된 전형별 쿼리 방식은 기존 TiMoS(Trope in Movie Synopses) 데이터셋에서 성능을 11.8포인트 향상시켜 새로운 최첨단 성과를 초래하였습니다. CoT가 입력된 경우 LLM의 정확도가 현저히 감소하며, 적대적 입력(Adversarial Input)에 대한 높은 민감도를 보여줍니다.



### DilateQuant: Accurate and Efficient Diffusion Quantization via Weight Dilation (https://arxiv.org/abs/2409.14307)
Comments:
          Code: this http URL

- **What's New**: 본 논문에서는 DilateQuant라는 새로운 양자화 프레임워크를 제안하여, 기존의 양자화 기술보다 정확도와 효율성을 동시에 향상시킵니다. 이 프레임워크는 특히 diffusion models(확산 모델)의 고유한 문제인 wide-range (넓은 범위)와 time-varying (시간 변화) activation(활성화)을 해결하는 데 중점을 두었습니다.

- **Technical Details**: DilateQuant는 unsaturated in-channel (비포화 채널) weights(가중치)를 활용하여 Weight Dilation (WD) 기법을 통해 가중치 범위를 유지한 채로 활성화 범위를 줄입니다. 이로 인해 activation quantization(활성화 양자화)가 용이해지고, 모델이 학습 단계에서 수렴하는 데 도움이 됩니다. 또한, Temporal Parallel Quantizer (TPQ)를 설계하여 서로 다른 시간 단계에 대해 병렬 양자화를 지원, 성능을 크게 향상시키며, Block-wise Knowledge Distillation (BKD)를 통해 전체 모델을 재학습할 필요 없이 효율적으로 성능을 개선합니다.

- **Performance Highlights**: DilateQuant는 기존 방법들과 비교하여 낮은 양자화 설정(6-bit, 4-bit)에서 더 뛰어난 성능을 보여줍니다. 다양한 모델(DDPM, LDM-4, LDM-8, Stable-Diffusion)과 데이터셋(CIFAR-10, LSUN-Bedroom, LSUN-Church, ImageNet, MS-COCO)에서 실험을 통해 그 우수성을 입증했습니다.



### PretextTrans: Investigating Medical Factual Knowledge Mastery of LLMs with Predicate-text Dual Transformation (https://arxiv.org/abs/2409.14302)
Comments:
          17 pages, 10 figures

- **What's New**: 이번 연구에서는 현재 대규모 언어 모델(LLMs)의 의료 사실 지식 숙련도를 동적인 평가 스키마를 사용하여 조사하고, 각 의료 사실 지식 포인트에 대한 여러 테스트 샘플을 자동으로 생성하는 방법을 제안합니다. 이는 기존의 LLM이 사용했던 방식의 한계를 극복하기 위한 것입니다.

- **Technical Details**: 우리는 Predicate-text Dual Transformation (PretextTrans)라는 새로운 평가 방법을 제안합니다. 이 방법은 각 의료 지식 포인트를 술어 표현으로 변환하고, 이를 통해 생성된 변형을 바탕으로 다양한 텍스트 표현을 생성합니다. 이러한 방식은 사실적 신뢰성과 표현의 다양성을 동시에 보장합니다. 이 연구에서 12개의 잘 알려진 LLM의 의료 사실 지식 숙련도를 두 개의 의료 데이터 세트를 기반으로 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안된 PretextTrans 방법에 의해 생성된 다중 샘플 데이터 세트에서 LLM의 성능이 기존의 단일 샘플 데이터 세트에 비해 상당히 낮은 것으로 나타났습니다. 이는 현재 LLM들이 의료 사실 지식을 포괄적으로 습득하지 못하고 있음을 보여주며, 실제 의료 시나리오에서의 성능 부진의 원인을 설명합니다.



### Opinion Mining on Offshore Wind Energy for Environmental Engineering (https://arxiv.org/abs/2409.14292)
- **What's New**: 이 논문에서는 소셜 미디어 데이터를 활용하여 해상 풍력 에너지에 대한 대중의 의견을 분석합니다. 세 가지 머신러닝 모델, 즉 TextBlob, VADER, SentiWordNet을 사용하여 각 모델이 제공하는 다양한 기능을 활용합니다.

- **Technical Details**: TextBlob은 주관성 분석(subjectivity analysis)과 극성 분류(polarity classification)를 제공하며, VADER는 누적 감정 점수(cumulative sentiment scores)를 산출합니다. SentiWordNet은 맥락(context)을 기준으로 감정을 고려하여 분류를 수행합니다. 자연어 처리(NLP) 기술을 통해 소셜 미디어의 텍스트 데이터에서 의미를 추출합니다.

- **Performance Highlights**: 데이터 시각화 도구를 적절히 사용하여 전체 결과를 표시하며, 이는 시민 과학(citizen science)과 스마트 거버넌스(smart governance)에 부합하여 대중의 의견이 의사 결정 지원(decision support)을 안내하는 역할을 합니다.



### ESPERANTO: Evaluating Synthesized Phrases to Enhance Robustness in AI Detection for Text Origination (https://arxiv.org/abs/2409.14285)
- **What's New**: 이 논문은 AI 생성 텍스트 탐지 시스템의 취약성과 현재 시스템의 내구성을 향상시킬 필요성을 강조합니다. 이를 해결하기 위해 새로운 기법인 back-translation을 소개하여 탐지를 우회하는 방법을 검토하면서, 이를 극복하기 위한 대응 방안도 제시합니다.

- **Technical Details**: 제안된 방법은 AI 생성 텍스트를 여러 언어로 번역한 후 영어로 다시 번역하는 과정으로 구성됩니다. 모델은 이러한 back-translated 텍스트를 결합하여 원본 AI 생성 텍스트의 변조 버전을 제작합니다. 이 기법은 720,000 개의 텍스트로 구성된 대규모 데이터셋에서의 실험을 통해 평가되었으며, 다양한 AI 탐지기를 대상으로 검증되었습니다.

- **Performance Highlights**: 변조된 텍스트는 원래 의미를 유지하면서, 기존 탐지 방법의 true positive rate (TPR)을 대폭 낮추는 효과를 보였습니다. 예를 들어, RADAR의 TPR은 질문-응답 데이터셋에서 52% 감소했습니다. 제안된 방법은 back-translation 변조에 노출되었을 때 TPR이 단지 1.85% 감소하는 것으로 나타났습니다.



### Proof Automation with Large Language Models (https://arxiv.org/abs/2409.14274)
Comments:
          12 pages, 15 figures, Accepted to ASE 2024

- **What's New**: 새로운 연구에서 PALM을 제안하여 대형 언어 모델(LLMs)을 기반으로 한 공식 증명 생성의 오류를 분석하고 수정하는 혁신적인 접근 방식을 소개합니다.

- **Technical Details**: PALM은 LLM을 사용하여 초기 증명을 생성한 후, 심볼릭 방법(symbolic methods)을 통해 저수준 문제를 반복적으로 수정하는 generate-then-repair 접근 방식을 채택합니다. 이는 기존 LLM들이 고수준 구조는 잘 잡지만 세부 사항에서 오류를 범하는 문제를 해결하기 위함입니다.

- **Performance Highlights**: PALM은 10,000개 이상의 정리에 대해 평가했으며, 기존 최첨단 방법들보다 76.6%에서 180.4% 더 많은 정리를 성공적으로 증명했습니다. 또한, PALM은 기존 접근법으로는 증명할 수 없는 1,270개의 정리도 증명하였습니다.



### Higher-order-ReLU-KANs (HRKANs) for solving physics-informed neural networks (PINNs) more accurately, robustly and faster (https://arxiv.org/abs/2409.14248)
- **What's New**: 본 연구에서는 Higher-order-ReLU (HR)라는 새로운 활성화 기반 함수(basis function)를 제안하여 기존의 Kolmogorov-Arnold Networks (KANs) 및 ReLU-KANs보다 더 간단하고 효율적인 KAN 수학적 연산(matrix operations)을 가능하게 합니다.

- **Technical Details**: 제안된 HRKANs는 KANs와 ReLU-KANs의 단점을 극복하여 높은 정확도(fitting accuracy) 및 강한 훈련 견고성(training robustness)을 제공하고, 효율적인 GPU 병렬 컴퓨팅(parallel computing)을 지원합니다. 기존의 B-spline과 'square of ReLU' 기반 함수에 비해 연속적인 고차 미분(higher-order derivatives)을 가지고 있어, physics-informed 문제에 적합합니다.

- **Performance Highlights**: 우리는 linear Poisson 방정식과 비선형 Burgers 방정식에 대해 HRKANs를 평가한 결과, KANs 및 ReLU-KANs에 비해 가장 높은 적합 정확도(fitting accuracy), 가장 강한 훈련 견고성(training robustness), 그리고 가장 빠른 수렴 속도를 보여주었습니다.



### An Instance-based Plus Ensemble Learning Method for Classification of Scientific Papers (https://arxiv.org/abs/2409.14237)
- **What's New**: 이 논문에서는 과학 논문의 효과적이고 효율적인 분류를 위한 새로운 접근 방식을 소개합니다. 인스턴스 기반 학습(instance-based learning)과 앙상블 학습(ensemble learning) 기법을 결합하여 연구 분야에 적합한 논문을 분류하는 시스템을 구축합니다.

- **Technical Details**: 이 접근법은 연구 분야 그룹에 대해 수동으로 할당된 수많은 전형적인 시드 논문(seed papers)을 기준으로 합니다. 각각의 분류가 필요한 논문에 대해 모든 전형 시드 논문과 비교하며, 내용(content)과 인용(citations)은 각각 따로 고려합니다. 마지막으로 앙상블 기반 방법을 사용하여 최종 결정을 내립니다.

- **Performance Highlights**: DBLP 데이터셋을 사용한 실험 결과, 제안된 분류 방법이 다양한 연구 분야에 논문을 효과적이고 효율적으로 분류할 수 있음을 입증했습니다. 또한 내용과 인용 특성이 과학 논문 분류에 유용하게 작용한다는 것을 발견했습니다.



### MEGA-PT: A Meta-Game Framework for Agile Penetration Testing (https://arxiv.org/abs/2409.14219)
- **What's New**: MEGA-PT라는 메타-게임 기반의 자동화된 침투 테스트 프레임워크를 제안하여, 정량적이고 효율적인 테스트를 수행함으로써 보안 취약성을 신속하게 발견하고 방어 전략을 제공할 수 있습니다.

- **Technical Details**: MEGA-PT는 두 가지 차원에서 모델링됩니다: 로컬 노드 간 상호작용을 위한 마이크로 전술 게임(micro tactic games)과 네트워크 전체 공격 체인을 모델링하는 매크로 전략 프로세스(macro strategy process). 이를 통해 분산된, 적응 가능한, 협업적인 침투 테스트가 가능하도록 하며, MITRE ATT&CK 프레임워크를 따릅니다.

- **Performance Highlights**: 실험 결과, MEGA-PT 모델은 방어 전략의 향상과 네트워크 및 로컬 레벨에서의 변화에 대한 적응성을 보여 주었습니다. 이로 인해 자동화된 침투 테스트의 효율성과 효과성을 크게 개선할 수 있음을 입증하였습니다.



### R-AIF: Solving Sparse-Reward Robotic Tasks from Pixels with Active Inference and World Models (https://arxiv.org/abs/2409.14216)
Comments:
          20 pages, 2 algorithms, 2 tables, 5 figures, submitted to ICRA 2025

- **What's New**: 이번 연구는 부분 관측 마르코프 의사결정 프로세스(POMDP)에서의 능동 추론(active inference, AIF) 모델을 다루며, 특히 sparse reward 신호가 있는 연속 액션 공간을 가진 POMDP 제어 문제를 해결하기 위한 독창적인 접근 방식을 제안합니다.

- **Technical Details**: 이 논문에서는 CRSPP(Contrastive Recurrent State Prior Preference) 모델을 통해 에이전트가 온라인으로 환경 상태에 대한 선호를 학습하도록 하고, 강화학습(actor-critic) 방법을 사용하여 예상 자유 에너지를 최적화하여 액션 플래너의 안정성을 높입니다. 또한, R-AIF(Robust Active Inference) 에이전트를 도입하여 스스로 수정하는 메커니즘을 사용해 sparse-reward 작업에서 모델의 수렴 속도를 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 R-AIF 에이전트가 기존의 최첨단 모델(DreamerV3) 및 다른 AIF 기준 모델보다 누적 보상, 상대적 안정성, 성공률 측면에서 향상된 성능을 보였습니다.



### Data-Driven Approach to assess and identify gaps in healthcare set up in South Asia (https://arxiv.org/abs/2409.14194)
- **What's New**: 이번 연구에서는 남아시아 국가들이 WHO의 보건 시스템 프레임워크에 따라 건강 불균형을 해결하고, 보편적인 건강 접근성을 보장하기 위한 정책을 어떻게 개선하고 있는지를 다룹니다. 특히 최근의 Earth-observation (EO) 기술의 발전을 활용하여 정확하고 구체적인 데이터를 생성하여 공정한 자원 분배를 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 연구에서는 EO 기술을 통해 수집된 데이터가 기본 건강 관리 시스템의 격차를 분석하고, 지역 사회의 다양한 건강 위험 요소를 평가하는 데 어떻게 사용될 수 있는지를 탐구합니다. 이는 자동화된 데이터 수집 및 분석을 통해 인간의 오류를 줄이고, 관측 능력을 강화하는 데 기여합니다.

- **Performance Highlights**: 기술적 기여를 통해 원거리 지역 및 의료 서비스가 부족한 지역의 건강 격차를 동적으로 판단하고, 맞춤형 개입을 통해 공정한 의료 접근성을 향상시킬 수 있는 가능성을 보여줍니다. 또한, 데이터 기반 의사 결정 체계를 통해 보건 정책과 중재를 개선할 수 있는 도구를 제공합니다.



### PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach (https://arxiv.org/abs/2409.14177)
- **What's New**: 본 논문에서는 LLM 보안 취약점을 극복하기 위한 새로운 블랙박스 jailbreak 방법인 PathSeeker를 제안합니다. 이는 기존의 jailbreak 공격 방식과 다르게 내부 모델 정보에 의존하지 않으며, 다수의 소형 모델들이 협력하여 메인 LLM의 돌연변이 작업(mutational operations)을 유도합니다.

- **Technical Details**: PathSeeker는 라트를 미로에서 탈출시키는 게임에서 영감을 받은 멀티 에이전트 강화 학습(multi-agent reinforcement learning) 기법에 기반합니다. 공격자들은 모델의 피드백을 통해 입력을 점진적으로 수정하며, 이는 목표 LLM의 유해한 반응을 유도하는 방식입니다.

- **Performance Highlights**: 이 방법은 13개의 상용 및 오픈소스 LLM에 대해 테스트된 5개의 최신 공격 기법을 능가하며, 특히 GPT-4o-mini, Claude-3.5와 같은 강력한 안전 정렬이 이루어진 상용 모델에서 높은 공격 성공률을 기록하였습니다.



### QMOS: Enhancing LLMs for Telecommunication with Question Masked loss and Option Shuffling (https://arxiv.org/abs/2409.14175)
- **What's New**: 이 논문은 QMOS라는 혁신적인 접근 방식으로 통신 분야에서 다중 선택 질문에 대한 LLM(대규모 언어 모델)의 성능을 향상시키는 방법을 제시합니다. 기존의 수익 모델에 의존하지 않고 오픈소스의 작은 언어 모델인 Phi-2와 Falcon-7B를 사용하여 Retrieval Augmented Generation (RAG) 프레임워크 내에서 다양한 개선사항을 적용하여 성과를 올릴 수 있었습니다.

- **Technical Details**: QMOS는 질문 마스크 손실 함수(Question-Masked loss)와 옵션 셔플링(Option Shuffling) 기법을 활용하여 LLM을 다중 선택 질문에 효율적으로 적응시킵니다. 이 프로세스는 여러 임베딩 모델을 통해 관련 정보를 다각화하고, 약어 딕셔너리를 확장하며, 정교한 프롬프트 설계를 통해 LLM이 문서에서 답을 선택하도록 유도합니다. 또한 LoRA(저차원 적응, Low-Rank Adaptation) 기법을 통해 Phi-2 모델을 통신 도메인에 맞추어 효율적으로 조정합니다.

- **Performance Highlights**: Falcon-7B 모델은 기본선에서 24.70%에서 49.30%로 정확도를 개선했으며 Phi-2 모델은 42.07%에서 84.65% 사이의 성과를 달성했습니다. 이는 기존 모델에 비해 значительно 향상된 결과로, 효율적이고 비용 효과적인 QA 시스템 구축에 기여할 것으로 예상됩니다.



### MSSDA: Multi-Sub-Source Adaptation for Diabetic Foot Neuropathy Recognition (https://arxiv.org/abs/2409.14154)
- **What's New**: 이번 연구에서는 당뇨병성 족부 신경병증(Diabetic Foot Neuropathy, DFN) 인식을 위한 새로운 연속적인 plantar pressure 데이터셋을 수집하였고, 이를 기반으로 효과적인 도메인 적응(domain adaptation) 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 three-stage alignment framework을 통해 구성됩니다. 첫 단계에서는 contrastive learning을 사용하여 모든 샘플을 가능한 한 잘 분리하고, 두 번째 단계에서는 원본 소스 데이터셋을 convolutional feature statistics에 따라 K개의 하위 소스(domain)로 나누어, pseudo domain labels를 할당합니다. 세 번째 단계에서는 적절한 domain labels을 가진 소스 샘플을 선택하고, 여러 feature 공간에서 소스 및 타겟 도메인 간의 분포를 정렬합니다.

- **Performance Highlights**: 새로 제안한 DFN 인식 데이터셋과 기존 데이터셋에 대해 포괄적인 실험을 수행하였으며, 제안된 모델의 효과성을 검증하는 실험 결과를 얻었습니다.



### Present and Future Generalization of Synthetic Image Detectors (https://arxiv.org/abs/2409.14128)
Comments:
          16 pages, 6 figures

- **What's New**: 본 연구는 이미지 생성 모델의 발전에 직면하여 합성 이미지 감지기(Detector)의 일반화 능력을 고찰합니다. 실험 결과, 현재 평가된 모든 감지기는 보편적이지 않지만, 앙상블 방법이 보편적일 수 있다는 가능성을 제시합니다. 특히 야외에서 수집된 데이터에 대한 실험이 대규모 데이터셋에서 정의된 작업보다 더 도전적임을 보여주며, 생성기와 감지기 간의 균형 효과를 관찰합니다.

- **Technical Details**: 이 연구는 합성 이미지 감지를 위한 강력한 감지기 구축을 위한 다양한 훈련 조건의 영향을 분석합니다. ResNet 구조(ResNet-18)를 고정하여 실험을 수행하며, 다양한 합성 이미지 데이터셋과 AI 이미지 생성기를 사용합니다. 또한, 이미지 패치 기반 접근 방식을 채택하여 감지 성능을 향상시키기 위한 전략을 사용합니다.

- **Performance Highlights**: 현재 감지기는 단독으로 사용되었을 때 합성 내용 탐지에 불충분함을 나타냅니다. 감지기 앙상블 사용 및 생성기 특정 감지기 훈련이 권장되며, 후자의 접근법이 다른 데이터 소스에 대해 일반화 가능성을 보여줍니다. 두 가지 새로운 데이터셋과 실험에 사용된 소프트웨어 라이브러리가 공개되었습니다.



### Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm (https://arxiv.org/abs/2409.14119)
Comments:
          Under Review

- **What's New**: 본 연구에서는 Parameter-efficient fine-tuning (PEFT) 아키텍처에서 악의적인 백도어 공격에 대한 방어 방법인 Obliviate를 소개합니다. Obliviate는 PEFT 프로세스와 통합되어 작동하며, 이는 두 가지 주요 기술을 포함합니다: 1) PEFT 레이어 내에서 유해하지 않은 뉴런을 증폭하여 모델이 청정 학습 샘플에 더 집중하도록 유도합니다. 2) 트리거 토큰의 영향을 제한하기 위해 어텐션 점수를 정규화합니다.

- **Technical Details**: Obliviate는 PEFT 아키텍처, 특히 adapter, LoRA, prefix-tuning을 기반으로 하여 RoBERTa와 BERT 모델에 적용될 수 있습니다. PEFT에서 훈련 가능한 매개변수가 제한적이기 때문에 기존의 방어 방법을 적용하기 어려운 부분을 극복하여, 백도어 공격을 효과적으로 중화시키기 위한 두 개의 손실 항(term)을 추가합니다.

- **Performance Highlights**: 실험 결과 Obliviate는 최첨단의 task-agnostic 백도어 공격의 성공률(ASR)을 83.6% 감소시키는 효과를 보였으며, 청정 정확도(CACC)는 약간(0.78%) 하락하는 것으로 나타났습니다. 뿐만 아니라, Obliviate는 다양한 공격 전략에 대한 강력한 방어 능력을 보여줍니다.



### One-shot World Models Using a Transformer Trained on a Synthetic Prior (https://arxiv.org/abs/2409.14084)
- **What's New**: 이번 논문에서는 One-Shot World Model (OSWM)이라고 불리는 새로운 transformer 기반의 월드 모델을 제안합니다. 이 모델은 순전히 합성 데이터로부터 학습하며, 다양한 환경에 대한 빠른 적응을 목표로 합니다.

- **Technical Details**: OSWM은 여러 개의 무작위 초기화된 신경망으로 구성된 합성 prior를 사용하여 환경 동역학을 모델링합니다. 이 방법은 Prior-Fitted Networks의 감독 학습 절차를 따르며, 다음 상태와 보상을 무작위 컨텍스트 위치에서 마스킹하여 OSWM이 나머지 상태 전이 컨텍스트를 바탕으로 확률적 예측을 할 수 있도록 합니다. 이 과정에서 OSWM은 1,000개의 무작위 전이 샘플을 컨텍스트로 제공받아 새로운 환경의 동역학에 적응합니다.

- **Performance Highlights**: OSWM은 간단한 GridWorld, CartPole gym 및 커스텀 제어 환경에서 환경을 해결하는 에이전트 정책을 성공적으로 훈련할 수 있음을 보여주었습니다. 그러나 복잡한 환경으로의 전환에 여전히 과제가 남아 있습니다.



### PTD-SQL: Partitioning and Targeted Drilling with LLMs in Text-to-SQL (https://arxiv.org/abs/2409.14082)
Comments:
          EMNLP 2024 Main Conference. Revised by ARR April and ARR June. 32 pages, 7 figures and 30 tables

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 Text-to-SQL 작업에 있어 논리적 사고능력을 발휘할 수 있는 방법을 제안합니다. 특히, 문제 유형에 따라 쿼리 그룹 분할(query group partitioning)을 활용함으로써 LLM들이 특정 문제 유형에 대한 사고 과정을 더 잘 학습할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서 제안하는 PTD-SQL(Problem Type Differentiation SQL)은 LLM들이 다양한 난이도와 문제 카테고리에서 더 뛰어난 추론 능력을 가지도록 돕습니다. 이를 통해 LLM들이 전통적인 SQL 솔루션 접근 방법과의 차별성을 가지고, 특정 문제 유형에 대한 사고 과정을 심도 있게 학습할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, 여러 고급 LLM들이 PTD-SQL로 강화된 후 Spider 및 BIRD 데이터셋에서 이전의 최첨단(SOTA) 방법들을 능가하거나 동등한 성능을 발휘했습니다. 특히, 초기 성능이 다양한 모델들이 집중적인 훈련(targeted drilling)을 받은 후 큰 향상을 보였으며, 이는 인간의 능력 진전을 연상케 합니다.



### N-Version Assessment and Enhancement of Generative AI (https://arxiv.org/abs/2409.14071)
Comments:
          This work has been accepted for publication in an upcoming issue of IEEE Software. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 소프트웨어 공학에서 생성적 인공지능(Generative AI, GAI)의 신뢰성을 높이기 위해 다중 버전을 활용한 코드 및 테스트 생성 방법을 제안합니다. 특히, '차별적 GAI'(Differential GAI, D-GAI) 접근 방식을 통해 코드의 품질 평가를 보다 신뢰할 수 있게 합니다.

- **Technical Details**: D-GAI는 여러 코드 버전과 테스트를 생성하여 각 버전을 비교 분석하는 방법입니다. LASSO(대규모 소프트웨어 관찰소, Large-Scale Software Observatorium) 플랫폼을 도입하여 코드 추천 요청을 여러 번 수행하고, 자동 생성된 테스트를 통해 결과를 평가합니다. 이 과정에서 코드 모델과의 다양한 호출을 통해 N개의 버전을 생성하며, 동적 및 정적 코드 메트릭을 활용하여 결과를 검증합니다.

- **Performance Highlights**: 기존 코드 모델의 추천 품질을 향상시키며, 테스트 주도 개발(test-driven development) 및 지속적 통합(Continuous Integration) 환경에서 비동기적으로 활용될 수 있습니다. D-GAI 엔진을 사용한 최근 실험에서는 1,000개의 코드 모듈 생성에 약 4분의 평균 응답 시간이 기록되었습니다. 이 접근법은 기술 부채를 줄이고 재작업 및 유지 보수 비용을 절감할 수 있습니다.



### KALIE: Fine-Tuning Vision-Language Models for Open-World Manipulation without Robot Data (https://arxiv.org/abs/2409.14066)
Comments:
          8 pages, 7 figures

- **What's New**: 이번 연구에서는 일반화 가능한 로봇 시스템의 개발을 위한 새로운 접근 방법으로 KALIE(Keypoint Affordance Learning from Imagined Environments)를 제안합니다. 이 방법은 로봇 제어를 위해 학습된 Vision Language Model(VLM)을 활용하여 자동으로 다양한 고품질 훈련 데이터를 생성합니다.

- **Technical Details**: KALIE는 자연어 지시사항과 장면의 시각적 관찰을 기반으로 포인트 기반의 affordance 표현을 예측하여 로봇을 제어합니다. 인간이 라벨링한 affordance 데이터로 교육된 VLM은 제어 데이터가 아닌 2D 이미지를 사용하여 훈련되며, 이를 통해 새로운 조작 작업을 수행할 수 있습니다.

- **Performance Highlights**: KALIE는 불특정 객체에 대한 조작 작업에서 50개의 예제 데이터만으로도 견고하게 문제를 해결할 수 있으며, 기존의 사전 학습된 VLM을 사용한 방법들에 비해 일관되게 우수한 성능을 보여줍니다.



### GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion (https://arxiv.org/abs/2409.14051)
Comments:
          18 pages

- **What's New**: 이번 연구에서는 다수의 에이전트가 참여하는 논쟁 방식에서 토큰 비용을 대폭 줄이는 혁신적인 방법인 GroupDebate (GD)를 제안합니다. 기존의 방법들보다 더욱 효율적이면서도 성능을 개선하는 데 초점을 맞추고 있습니다.

- **Technical Details**: GroupDebate 방법은 모든 에이전트를 여러 논쟁 그룹으로 나누고, 그룹 내에서 내부 논쟁을 거친 후 중간 결과를 공유하는 방식을 채택합니다. 이렇게 함으로써 토큰 사용량을 약 51.7%까지 줄이고, 정확도 역시 최대 25%까지 향상시킬 수 있음을 실험을 통해 입증하였습니다.

- **Performance Highlights**: 실험 결과, GroupDebate는 기존 다수의 에이전트 논쟁 방법들에 비해 Arithmetic, GSM8K, MMLU, MATH 데이터셋에서 각각 45%에서 51.7%까지 토큰 비용을 절감하며, MMLU와 MATH 데이터셋에서는 최대 25%의 정확도 향상을 보여주었습니다.



### PepINVENT: Generative peptide design beyond the natural amino acids (https://arxiv.org/abs/2409.14040)
- **What's New**: 이번 연구에서는 PepINVENT라는 새로운 generative AI 기반 도구를 소개하며, 이것은 기존의 peptide 설계 도구인 REINVENT의 연장선으로서 작용합니다. PepINVENT는 자연 및 비자연 아미노산의 방대한 공간을 탐색하며 혁신적이고 다양한 peptide 디자인을 제안합니다.

- **Technical Details**: PepINVENT는 특정한 속성이나 구조를 가진 peptide에 대한 훈련 없이 peptide와 아미노산의 미세한 특성을 이해하도록 훈련된 모델입니다. 이 도구는 강화 학습(reinforcement learning)과 결합되어, 화학적 정보에 기반한 generative 능력을 통해 목표 지향적 peptide 설계를 가능하게 합니다.

- **Performance Highlights**: PepINVENT는 독창적인 디자인으로 peptide 공간을 탐색하는 능력과 치료적으로 중요한 peptide의 특성 최적화 능력을 보여줍니다. 이 도구는 다중 매개변수 학습 목표, peptidomimetics, 리드 최적화(lead optimization) 등 다양한 peptide 관련 작업에 활용될 수 있습니다.



### Can LLMs replace Neil deGrasse Tyson? Evaluating the Reliability of LLMs as Science Communicators (https://arxiv.org/abs/2409.14037)
- **What's New**: 이 논문에서는 현재의 Large Language Models(LLMs)를 과학 커뮤니케이터로서의 신뢰성을 평가하는 새로운 접근 방식을 소개합니다. 기존의 벤치마크와는 달리, LLM의 과학적 질문 응답 과제를 기반으로 LLM의 이해도를 평가하는 SCiPS-QA라는 새로운 데이터셋을 도입했습니다.

- **Technical Details**: SCiPS-QA 데이터셋은 복잡한 과학 개념에서의 742개의 Yes/No 질문으로 구성되며, 이를 통해 LLM의 정확성과 일관성을 다양한 기준으로 평가합니다. 실험에는 OpenAI의 GPT 시리즈와 Meta의 Llama 시리즈 및 Mistral 시리즈의 LLM이 포함됩니다. 고급 과학적 이해가 필요한 질문에 대한 LLM의 성능을 테스트하기 위해 다수의 평가 기준을 적용하였습니다.

- **Performance Highlights**: 대부분의 오픈 액세스 모델은 GPT-4 Turbo에 비해 떨어지지만, Llama-3-70B 모델은 다양한 평가에서 GPT-4 Turbo를 초과하는 경우가 있었습니다. 인간 평가자들이 GPT-4 Turbo의 잘못된 응답에 속아 넘어가는 경향도 관찰되었습니다.



### Uncovering Latent Chain of Thought Vectors in Language Models (https://arxiv.org/abs/2409.14026)
Comments:
          2 Pages, Intended for Tiny Papers 2025 Submission to ICLR

- **What's New**: 이 연구는 언어 모델(LM)의 행동을 선도하기 위해 'steering vector'라는 새로운 기법을 소개합니다. 이는 특정 작업에서 파생된 steering vector를 사용하여 Chain of Thought (CoT) Reasoning을 유도하며, 기존 자연어 프롬프트 없이도 이를 가능하게 합니다.

- **Technical Details**: 연구진은 Llama3 8b 및 Mistral 7b v0.2 모델에서 steering vector를 활용하여 CoT Reasoning을 진행하였습니다. 이들은 자연어 프롬프트 쌍을 대비시켜, 각 레이어의 활성화를 추출하고 이를 통해 최종 steering vector를 생성했습니다. PyTorch를 사용하여 추출된 레이어에 vector를 주입하는 방식으로 적용되었습니다.

- **Performance Highlights**: 이 접근 방식은 CoT 프롬프트를 사용한 모델들과 비교하여 경쟁력 있는 성능을 보였으며, 다양한 Reasoning benchmark(GSM8k, MMLU, ARC AI2)에서도 일관되게 CoT 응답을 유도하는 결과를 보여주었습니다. 또한, 전통적인 모델 미세 조정 방법보다 계산 비용이 절감된다는 장점이 있습니다.



### FAMOUS: Flexible Accelerator for the Attention Mechanism of Transformer on UltraScale+ FPGAs (https://arxiv.org/abs/2409.14023)
- **What's New**: 이 논문은 Transformer Neural Networks (TNNs) 용으로 설계된 궤도 가변형 하드웨어 가속기인 'FAMOUS'를 제안하며, 이는 FPGA 플랫폼에서 밀집형 다중헤드 어텐션(MHA) 계산을 최적화하여 높은 프로세싱 성능을 구현합니다.

- **Technical Details**: FAMOUS는 프로세싱 요소(Processing Element)와 온칩 메모리(On-chip Memory)의 높은 활용도를 극대화하여 병렬 처리와 대기시간(Latency)를 줄일 수 있도록 설계되었습니다. 실험은 Xilinx Alveo U55C와 U200 데이터 센터 카드에서 수행되었으며, 효율적인 매트릭스 타일링(Tiling)을 통해 다양한 FPGA 플랫폼에서 메모리와 컴퓨팅 자원을 고르게 분배합니다. 또한, HLS(High-Level Synthesis) 기반 코드를 사용하여 DSP(Digital Signal Processing)와 BRAM(Block RAM)의 병렬 활용을 극대화했습니다.

- **Performance Highlights**: FAMOUS는 U55C에서 최대 328 GOPS(giga operations/second), 8개의 병렬 어텐션 헤드, 768의 임베딩 차원 및 64의 타일 크기를 달성하였으며, Intel Xeon Gold 5220R CPU와 NVIDIA V100 GPU보다 각각 3.28배 및 2.6배 빠릅니다. 또한, 최신 FPGA 기반 가속기보다도 1.3배 더 빠른 성능을 제공합니다.



### BrainDreamer: Reasoning-Coherent and Controllable Image Generation from EEG Brain Signals via Language Guidanc (https://arxiv.org/abs/2409.14021)
- **What's New**: 신경과학적 뇌 신호를 기반으로 언어를 통해 인간의 사고를 모방하는 BrainDreamer라는 새로운 생성 프레임워크를 소개합니다. 이는 비침습적인 EEG 데이터로부터 고품질 이미지를 생성하는 능력을 가지고 있습니다.

- **Technical Details**: BrainDreamer는 두 가지 주요 학습 단계를 포함합니다: 1) modality alignment와 2) 이미지 생성. 첫 번째 단계에서는 새로운 마스크 기반의 세 가지 대비 학습 전략을 사용하여 EEG, 텍스트, 이미지 임베딩을 효과적으로 정렬하고 통합된 표현을 학습합니다. 두 번째 단계에서는 스테이블 디퓨전 모델에 EEG 임베딩을 주입하여 고품질 이미지를 생성합니다.

- **Performance Highlights**: 이 방법은 기존 기술보다 이미지 생성의 품질과 정량적인 성능에서 크게 향상된 결과를 보여 줍니다. 실험 결과 BrainDreamer는 더 소음이 제거되고 정밀한 EEG-이미지 매핑을 달성하여 기존 방법들을 초월하는 성능을 나타냅니다.



### MOSE: Monocular Semantic Reconstruction Using NeRF-Lifted Noisy Priors (https://arxiv.org/abs/2409.14019)
Comments:
          8 pages, 10 figures

- **What's New**: 본 논문에서는 MOSE라는 개념을 제안하여, 단일 이미지만을 사용하여 3D 메쉬를 재구성하는 데 있어 고품질의 기하학과 세밀한 의미 레이블링을 동시에 달성하고자 합니다. 이는 불완전한 2D 우선순위를 기반으로 하여 이루어집니다.

- **Technical Details**: MOSE는 임펄스(impulse) 신경 네트워크를 사용하여 2D 이미지와 노이즈가 포함된 2D 장면 우선 순위를 입력으로 받으며, 이를 기반으로 3D 의미 맵을 재구성합니다. 이 연구에서는 클래스 무관 이미지 세그먼트(masks)를 활용하여 훈련 중 지역 일관성을 촉진하고, 텍스처가 없는 영역에서 더 나은 기하학적 품질을 위해 매끄러움 정규화를 적용합니다.

- **Performance Highlights**: ScanNet 데이터 세트를 통한 실험 결과, MOSE는 3D 의미 분할, 2D 의미 분할 및 3D 표면 재구성 작업에서 모든 메트릭에서 이전 기법들보다 뛰어난 성능을 보여주었습니다.



### Enhancing Multivariate Time Series-based Solar Flare Prediction with Multifaceted Preprocessing and Contrastive Learning (https://arxiv.org/abs/2409.14016)
Comments:
          This work has been accepted at ICMLA 2024 on September 7, 2024, as a regular paper for an oral presentation

- **What's New**: 본 연구는 다변량 시계열(multivariate time series) 기반의 데이터셋에 대한 고급 데이터 전처리 및 분류 방법을 활용하여 태양 플레어 예측의 정확성을 향상시키는 데 중점을 두었습니다.

- **Technical Details**: 우리는 결측값 보간(missing value imputation), 정규화(normalization), 균형 샘플링(balanced sampling), 근접 결정 경계 샘플 제거(near decision boundary sample removal), 피처 선택(feature selection) 등의 단계로 구성된 새로운 전처리 파이프라인을 적용하였습니다. 또한, GRU 회귀 모델과 대조 학습(contrastive learning)을 통합하여 ContReg라는 새로운 분류기를 개발했습니다.

- **Performance Highlights**: 이 연구의 결과는 TSS(True Skill Statistic) 점수가 뛰어나며, 이전 방법들을 초월하여 태양 플레어 예측의 데이터 전처리 및 분류기 개발의 중요성을 강조하고 있습니다.



### Mitigating Exposure Bias in Score-Based Generation of Molecular Conformations (https://arxiv.org/abs/2409.14014)
Comments:
          SMC 2024

- **What's New**: 본 연구에서는 Score-Based Generative Models (SGMs)에서 발생하는 exposure bias를 측정하는 방법을 제안하였으며, 이는 기존의 Diffusion Probabilistic Models (DPMs)에서 발생하는 healing 문제를 해결하기 위한 중요한 단서를 제공합니다. 특히, Torsional Diffusion 모델과 ConfGF 모델에서 이 biais의 존재를 확인하고 그 값을 측정했습니다.

- **Technical Details**: Exposure bias는 모델 훈련 시 실제 샘플에 의존하는 것과 생성 시 예측 샘플에 의존하는 것 간의 일관성 부족에서 발생합니다. 제안된 Input Perturbation (IP) 알고리즘은 DPMs를 위해 개발된 방법으로, SGM에서도 적용되었습니다. GEOM-Drugs 및 GEOM-QM9 데이터셋에서 Torsional Diffusion 모델 기반의 새로운 최첨단 성능을 달성했습니다.

- **Performance Highlights**: Torsional Diffusion 모델을 활용해 GEOM-Drugs 데이터셋에서 새로운 최첨단 성능을 기록하였고, GEOM-QM9에서 기존 최고 성능과 대등한 성능을 보였습니다. Input Perturbation 방법을 통해 생성된 구조의 정확성과 다양성이 크게 향상되었습니다.



### ChronoGAN: Supervised and Embedded Generative Adversarial Networks for Time Series Generation (https://arxiv.org/abs/2409.14013)
Comments:
          This work has been accepted at ICMLA 2024 on September 7, 2024, as a regular paper for an oral presentation

- **What's New**: ChronoGAN은 Generative Adversarial Networks (GANs)를 활용한 새로운 프레임워크로, 시간 기반 손실 함수와 감독 네트워크를 통합하여 시계열 데이터 생성을 개선합니다.

- **Technical Details**: 이 프레임워크는 5개의 신경망으로 구성되며, 잠재 공간(latent space)에서 데이터를 생성하는 생성기(generator)와 특성 공간(feature space)에서 피드백을 제공하는 판별기(discriminator)를 포함합니다. 또한, 시간에 따라 데이터의 동적 변화를 학습하는 감독 네트워크(supervisor)를 활용합니다.

- **Performance Highlights**: ChronoGAN은 TimeGAN과 같은 기존 모델들에 비해 뛰어난 성능을 보이며, 다양한 실세계 및 합성 데이터셋에서 고품질의 시간 시계열 데이터를 생성합니다.



### Test Time Learning for Time Series Forecasting (https://arxiv.org/abs/2409.14012)
- **What's New**: 이 논문은 Test-Time Training (TTT) 모듈을 병렬 아키텍처에 도입하여 Long-Time Series Forecasting (LTSF) 성능을 개선하는 방법을 제안합니다. TTT 모듈이 기존의 최첨단 모델들을 지속적으로 능가함을 보여줍니다.

- **Technical Details**: TTT 모듈은 긴 시퀀스와 예측 길이에 대한 성능이 우수하며, Conv Stack 5와 같은 아키텍처가 Mamba 기반 모델보다 긴 종속성(long-range dependencies)을 더 효과적으로 포착합니다. TTT 모듈은 효율적인 추론을 위한 선형 RNN 아키텍처를 사용합니다.

- **Performance Highlights**: TTT 기반 모델은 Electricity, Traffic, Weather와 같은 큰 데이터셋에서 특히 뛰어난 성능을 발휘하며, 평균 제곱 오차(Mean Squared Error, MSE)와 평균 절대 오차(Mean Absolute Error, MAE)에서 두드러진 개선을 보여줍니다. TTT는 예측 길이가 긴 다양한 시나리오에서도 뛰어난 결과를 기록하였습니다.



### Boolean Product Graph Neural Networks (https://arxiv.org/abs/2409.14001)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2407.10688

- **What's New**: 본 논문은 GNNs(그래프 신경망)에서의 잠재 그래프 구조 학습 중 변동성을 완화하기 위한 새로운 방법으로 Boolean product 기반의 그래프 잔여 연결을 제안합니다.

- **Technical Details**: 제안된 방법은 각 레이어에서 잠재 그래프와 원본 그래프 간의 Boolean product를 계산하여 학습 과정을 수정하는 것으로, 이를 통해 두 그래프의 인접 행렬 간의 삼각형 감지를 수행합니다.

- **Performance Highlights**: 여러 벤치마크 데이터 세트에서 실험을 통해 제안된 Boolean product 기반 그래프 신경망이 GNNs의 성능과 강건성을 향상시키는 능력을 입증했습니다.



### Graph Neural Network Framework for Sentiment Analysis Using Syntactic Featur (https://arxiv.org/abs/2409.14000)
- **What's New**: 소셜 미디어 플랫폼과 전자 상거래 생태계의 빠른 발전에 따라, 의견 분석(opinion mining) 분야가 자연어 처리(natural language processing)에서 중요한 연구 영역으로 떠오르고 있습니다. 본 연구는 텍스트 맥락 내 특정 요소에 대한 세밀한 평가를 추출하는 독특한 프레임워크를 제안합니다.

- **Technical Details**: 제안된 시스템은 구문 구조(syntactic structures)를 행렬(matrix) 형식으로 변환하고, 이 과정에서 그래프 내의 컨볼루션(convolutions) 및 어텐션(attention) 메커니즘을 활용하여 중요한 특징(salient characteristics)을 증류(distill)합니다. 설명자의 위치적 관련(positional relevance)을 어휘 항목(lexical items)과 연관시키는 방식은 입력의 순차적 무결성을 향상시킵니다.

- **Performance Highlights**: 실험(trials) 결과, 이 통합된 그래프 중심(graph-centric) 방안이 평가 범주화(evaluative categorization)의 효율성을 현저히 증대시키며 뛰어난 성능을 보여주었습니다.



### Relevance-driven Decision Making for Safer and More Efficient Human Robot Collaboration (https://arxiv.org/abs/2409.13998)
- **What's New**: 이 연구에서는 인간-로봇 협업(Human-Robot Collaboration, HRC)에서의 관련성(relevance) 개념을 도입하여 새로운 두 개의 루프 프레임워크를 개발하고, 이를 통해 HRC의 안전성과 효율성을 높이는 방법을 제시합니다. 이 프레임워크는 실시간(real-time)과 비동기(asynchronous) 처리를 통합하여 관련성을 정량화하고 안전한 결정 메커니즘을 가능하게 합니다.

- **Technical Details**: 연구는 LLMs(Large Language Models)를 활용하여 비동기 루프에서 인간의 목표와 관련성을 예측하고, 실시간 루프에서 장면 이해(scene understanding), 인간 의도 예측(human intent prediction), 의사결정을 실행합니다. 의사결정 모듈에서는 인간의 경로 예측을 고려한 새로운 작업 할당(method for task allocation) 방법과 모션 생성(motion generation) 및 충돌 회피(collision avoidance) 방법론을 제안하였습니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 제안된 방법론은 목표 예측(objective prediction)에서 평균 0.90의 정확도를 달성하며, 관련성 예측에서 0.96에 이릅니다. 또한 Motion Generation 방법은 최신 충돌 회피(state-of-the-art collision avoidance) 방법과 비교했을 때 충돌 사례를 63.76% 감소시키고, 충돌 프레임을 44.74% 줄이는 성과를 보였습니다.



### Contrastive Learning for Knowledge-Based Question Generation in Large Language Models (https://arxiv.org/abs/2409.13994)
Comments:
          5 pages, 2 figures

- **What's New**: 이번 연구는 고품질 질문 생성(question generation)을 지원하기 위한 새로운 방법론을 제안합니다. 특히, 지식 기반 질문 생성 기술에 초점을 맞추고 있으며, 이 과정에서 발생하는 착각(hallucination)과 지식 격차를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 contrastive learning을 통합하여 여러 모델이 도메인 지식을 공동으로 탐색하도록 하며, 생성 과정에서의 잡음과 착각을 줄이도록 유도합니다. 또한, contrasting examples를 포함한 프롬프트(prompt)를 설계하여 질문 생성 성능을 향상시킵니다.

- **Performance Highlights**: 실험 결과, 대조적인 지시 및 예제를 동시에 사용하는 경우 질문 생성의 품질이 크게 향상되었으며, 이는 높은 정확도를 이끌어냅니다. 제안된 방법은 대조적 맥락과 사고의 흐름(chain-of-thought) 프롬프트를 결합함으로써 질문 생성의 품질과 실용성을 효과적으로 개선할 수 있음을 보여주었습니다.



### ChemEval: A Comprehensive Multi-Level Chemical Evaluation for Large Language Models (https://arxiv.org/abs/2409.13989)
- **What's New**: 최근 화학 분야에서 LLMs(대형 언어 모델)가 수행하는 역할에 대한 관심이 증가하고 있으며, 이를 바탕으로 화학적 작업을 평가하기 위한 LLMs 벤치마크가 개발되고 있습니다.

- **Technical Details**: 본 연구에서는 ChemEval을 제안하며, 이는 화학 분야 내 다양한 작업에 대한 LLMs의 역량을 종합적으로 평가합니다. ChemEval은 4개의 주요 단계와 12개의 차원을 포함하여 42개의 화학 작업을 평가합니다. 이러한 작업은 오픈소스 데이터와 화학 전문가가 세심하게 구성한 데이터를 기반으로 합니다.

- **Performance Highlights**: 실험 결과, GPT-4와 Claude-3.5와 같은 일반 LLMs는 문헌 이해 및 지시 수행에서는 우수한 성능을 보이나, 고급 화학 지식이 필요한 작업에서는 부족한 성과를 보였습니다. 반면, 전문 LLMs는 향상된 화학 역량을 보여주나, 문학적인 이해력은 감소하는 경향이 있습니다. 이는 화학 분야의 복잡한 작업을 수행할 때 LLMs의 능력을 향상시킬 수 있는 잠재력을 시사합니다.



### Enhancing Advanced Visual Reasoning Ability of Large Language Models (https://arxiv.org/abs/2409.13980)
Comments:
          EMNLP 2024 Main

- **What's New**: 본 논문에서는 Complex Visual Reasoning Large Language Models (CVR-LLM)를 제안하여, Vision-Language Models (VLMs)와 Large Language Models (LLMs)의 장점을 통합하여 복잡한 시각적 추론을 수행하는 방법을 소개합니다.

- **Technical Details**: CVR-LLM은 LLM의 방대한 텍스트 지식을 활용하여 이미지-텍스트 쌍 없이도 정확한 예측을 가능하게 하며, 이미지에서 문맥 인식 이미지 설명(context-aware image descriptions, CaID)을 생성하기 위해 이터러티브 자기 정제 과정(iterative self-refinement loop)을 사용합니다. 이 과정에서 LLM의 피드백을 반영하여 복잡한 시각적 추론 과제에 단순화된 단일 모달 문제로 변환합니다. 또한, Chain-of-Comparison (CoC) 기술을 도입하여 예측 결과의 다양한 측면을 단계적으로 비교합니다.

- **Performance Highlights**: CVR-LLM은 WinoGAViL, Winoground, Whoops, VCR, NYCCC와 같은 복잡한 시각적 추론 작업에서 SOTA(performance state-of-the-art) 성능을 달성하며, 기존의 접근법과 비교하여 우수성을 입증합니다.



### Detecting Inpainted Video with Frequency Domain Insights (https://arxiv.org/abs/2409.13976)
Comments:
          submit to ICASSP2025

- **What's New**: 이번 논문에서는 Frequency Domain Insights Network (FDIN)이라는 새로운 비디오 인페인팅 탐지 모델을 소개합니다. FDIN은 주파수 영역의 통찰력을 포함하여 탐지 정확성을 획기적으로 향상시키는 방법을 제안합니다.

- **Technical Details**: FDIN은 Adaptive Band Selective Response (ABSR) 모듈을 통해 다양한 인페인팅 기술에 특정한 주파수 특징을 구별합니다. 또한 Fast Fourier Convolution 기반의 Attention (FFCA) 모듈을 사용해 인페인팅된 영역에서 주기적인 아티팩트를 식별합니다. 이 구조는 3D ResBlock을 활용하여 시공간(spatiotemporal) 분석을 실현합니다.

- **Performance Highlights**: 실험 결과, FDIN은 여러 공개 데이터셋에서 기존의 방법들보다 우수한 성능을 발휘하며 비디오 인페인팅 탐지에 대한 새로운 기준을 세웠습니다.



### ProTEA: Programmable Transformer Encoder Acceleration on FPGA (https://arxiv.org/abs/2409.13975)
- **What's New**: 본 논문에서는 최신 트랜스포머 인코더를 위한 런타임 프로그래밍 가능 가속기인 ProTEA를 소개합니다. ProTEA는 밀집 연산에 최적화되어 있으며, 각종 트랜스포머 네트워크의 실행 속도를 극대화하는 설계를 구현하였습니다.

- **Technical Details**: ProTEA는 FPAG 내 여러 하드웨어 구성 요소에 메모리와 컴퓨팅 자원을 분배하는 효율적인 매트릭스 타일링(tiling) 방식을 도입하여 지연(latency)을 줄이고 병렬성(parallelism)을 극대화합니다. 특히 8개의 병렬 주의 머리(parallel attention heads), 12개의 레이어(layer), 768의 임베딩 차원으로 설정 시 멀티 헤드 셀프 어텐션 블록(multi-head self-attention block)에서 타일 크기(tile size)를 64로 설정할 때 가장 높은 성능을 발휘함을 보였습니다.

- **Performance Highlights**: ProTEA는 Xilinx Alveo U55C 고성능 데이터 센터 가속기 카드에서 NVIDIA Titan XP GPU보다 2.5배 빠른 성능을 나타냈습니다. 또한, 현재 최신 맞춤형 FPGA 가속기와 비교했을 때 1.3배에서 2.8배의 속도 향상을 달성했습니다.



### One Model, Any Conjunctive Query: Graph Neural Networks for Answering Complex Queries over Knowledge Graphs (https://arxiv.org/abs/2409.13959)
- **What's New**: 전통적인 지식 그래프 기반 질의 응답 시스템의 한계를 넘어, AnyCQ라는 새로운 그래프 신경망 모델을 제안합니다. 이 모델은 모든 관계형 데이터에서 질의에 대한 답변을 예측할 수 있도록 설계되었습니다.

- **Technical Details**: AnyCQ 모델은 강화 학습( reinforcement learning) 목표를 사용하여 이진(Boolean) 질의를 처리합니다. 이 모델은 질의와 가능한 답변 세트를 입력으로 받아, 이러한 답변이 완전한 지식 그래프와 비교하여 참(true) 또는 거짓(false)으로 분류합니다. 또한 질의 응답 검색(query answer retrieval) 문제를 다루어, 주어진 질의에 대해 적합한 답변을 찾아내거나 올바른 해결책이 없다고 결정합니다.

- **Performance Highlights**: AnyCQ는 간단한 사례에 대해 학습한 후, 큰 구조의 복잡한 질의에 대한 답변을 신뢰성 있게 분류하고 검색할 수 있습니다. 기존 접근 방식이 실패한 샘플에서도 잘 작동하며,  새로운 도전적인 벤치마크에서 경험적으로 검증되었습니다. 또한, 관련 링크 예측기가 장착되었을 때 AnyCQ 모델은 분포 외 지식 그래프(out-of-distribution knowledge graphs)에도 효과적으로 전이 가능하다는 점이 강조되었습니다.



### TalkMosaic: Interactive PhotoMosaic with Multi-modal LLM Q&A Interactions (https://arxiv.org/abs/2409.13941)
Comments:
          6 pages, 5 figures

- **What's New**: 이 논문은 다양한 자동차 이미지를 사용하여 환경 보호 주제를 중심으로 새롭게 구성된 포토모자이크 이미지에서 동물 이미지(예: 새, 사자)를 생성하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: 우리는 'click and display'라는 간단한 작업을 통해 포토모자이크 이미지에서 타일 이미지와 원래의 자동차 이미지 간의 인터랙티브한 전환을 시연하는 방법을 제시하며، 사용자가 자동차 이미지를 TalkMosaic에 업로드한 후 관련 질문을 효율적으로 할 수 있게 돕습니다. 또한, 희소 주의(Sparse Attention)와 양자화(Quantization) 기술을 활용하여 다중 모달 LLM(대형 언어 모델)의 추론 속도를 높이는 방법을 심층적으로 분석합니다.

- **Performance Highlights**: 제안된 프로토타입은 PrFlashAttention(확률적 플래시 주의)와 SAQ(계단식 적응 양자화) 방법을 통해 효율성과 효과성을 입증하며, 사용자는 고환경 기준을 만족하는 타이어를 자동차 이미지 관련하여 어디서 구매할 수 있는지 등에 대한 질문에 대한 신속한 답변을 받을 수 있습니다.



### Learning Recourse Costs from Pairwise Feature Comparisons (https://arxiv.org/abs/2409.13940)
Comments:
          "Recourse for Humans", paper 49 from the Participatory Approaches to Machine Learning workshop at the International Conference on Machine Learning (ICML) 2020. For workshop website, see this https URL

- **What's New**: 본 논문은 사용자의 입력을 기계학습 모델의 선호도 학습 및 추론에 통합하는 새로운 기법을 제시합니다. 사용자가 각 개별 feature의 수정 용이성에 대한 개인적인 선호를 반영할 수 있도록 돕는 recourse 찾기 알고리즘을 개발했습니다.

- **Technical Details**: Bradley-Terry 모델을 사용하여 비포괄적인 인간 비교 설문을 통해 feature-wise 비용을 자동으로 유추합니다. 사용자는 전체 recourse를 비교하고, 각 feature 수정을 위한 명시적인 비용을 정량화하는 대신 어떤 recourse가 더 구현하기 쉬운지를 결정합니다. MAP 추정을 사용하여 개별 feature 비용을 효과적으로 학습하는 방법을 보여줍니다.

- **Performance Highlights**: 이 연구에서는 비포괄적 인간 설문조사가 모든 feature 쌍 비교에 대한 데이터를 포함하지 않더라도 exhaustive feature 비용 셋 학습에 충분하다는 것을 입증합니다. 사용자가 전체 recourse를 비교하여 수집한 데이터만으로도 각 feature의 '수정 용이성' 비용을 추출할 수 있음을 시뮬레이션을 통해 demonstrated합니다.



### MirrorStories: Reflecting Diversity through Personalized Narrative Generation with Large Language Models (https://arxiv.org/abs/2409.13935)
Comments:
          5 pages (excluding references), accepted to EMNLP 2024 Main Conference

- **What's New**: 이 연구는 개인화된 '미러 스토리' 제작에 있는 대규모 언어 모델(LLMs)의 효과성을 탐구하며, 문학의 다양성이 부족한 점을 해결하려고 합니다.

- **Technical Details**: 미러 스토리는 이름, 성별, 연령, 민족, 독자 관심사 및 이야기 도덕성과 같은 요소들을 통합하여 생성된 1,500개의 개인화된 단편 이야기의 집합입니다. 이 연구는 26인의 다양한 인간 평과자를 통해 LLMs가 생성한 개인화된 이야기가 일반적인 인간 저술 및 LLM 저술의 이야기들에 비해 높은 점수를 기록함을 보여줍니다.

- **Performance Highlights**: 개인화된 LLM 생성 이야기는 참여도 모든 지표에서 평균 4.22 점(5점 만점)에 비해 3.37 점으로 일반적인 이야기들보다 뛰어난 성과를 보였습니다. 이러한 이야기는 텍스트의 다양성을 높이면서도 원래의 도덕성을 유지하는 결과를 가져옵니다.



### Eliciting Instruction-tuned Code Language Models' Capabilities to Utilize Auxiliary Function for Code Generation (https://arxiv.org/abs/2409.13928)
Comments:
          EMNLP 2024 Findings Short

- **What's New**: 본 논문에서는 코드를 생성하기 위해 instruction-tuned 모델(명령 조정 모델)이 보조 함수(auxiliary function)를 효과적으로 활용하는 방법을 탐구합니다. 기존 모델들은 보조 함수를 텍스트 프롬프트에 포함시키는 방식에 한계가 있었으나, 새로운 프롬프트 구조를 통해 성능을 개선했습니다.

- **Technical Details**: 연구자는 보조 함수 정보를 쿼리에 추가하거나 응답_prefix(구성 요소)를 제공하여 instruction-tuned 모델의 코드 생성 능력을 증진시키기 위해 여러 가지 프롬프트를 설계했습니다. 이러한 접근법은 모델이 실제로 코드를 이해하고 보조 함수를 활용하는 데 도움이 됩니다. 실험에서 사용된 모델은 최근의 경쟁력 있는 instruction-tuned 모델들로, Humanextension 벤치마크를 통해 성능 평가를 실시했습니다.

- **Performance Highlights**: 제안된 프롬프트 방식은 gpt-4o와 같은 강력한 상용 모델에 비해 오픈소스 모델의 성능을 초과하는 결과를 보여주었습니다. 특히, 보조 함수와 함께 제공된 쿼리 및 응답 구조에서 파생된 개선 성과가 두드러졌습니다. 결과적으로 instruction-tuned 모델은 일반적으로 기본 모델보다 우수한 성과를 기록했습니다.



### Enhancing Large Language Models with Domain-specific Retrieval Augment Generation: A Case Study on Long-form Consumer Health Question Answering in Ophthalmology (https://arxiv.org/abs/2409.13902)
- **What's New**: 대규모 언어 모델(LLM)이 의료 분야에서의 잠재력에도 불구하고, 확증되지 않은 증거를 바탕으로 하거나 허구의 (hallucinated) 증거가 포함된 반응을 생성할 수 있다는 점에서 중요한 연구 결과가 제시되었습니다. 본 연구에서는 검색 강화 생성 (RAG, Retrieval Augment Generation) 방식이 의료 분야의 하위 도메인 특정 응용 프로그램에 적용된 몇 안 되는 사례 중 하나로, 70,000개의 안과 관련 문서를 활용한 RAG 파이프라인을 개발하였습니다.

- **Technical Details**: 연구는 장문의 소비자 건강 질문에 대한 사례 연구를 통해 진행되었습니다. 100개의 질문에 대해 RAG를 사용할 경우와 사용하지 않을 경우 LLM의 답변을 10명의 의료 전문가와 함께 체계적으로 평가하였습니다. 평가 항목은 증거의 사실성(factuality), 증거 선택 및 순위 (selection and ranking), 증거 귀속(attribution), 답변의 정확성(accuracy) 및 완전성(completeness)을 포함합니다.

- **Performance Highlights**: RAG가 없는 LLM은 총 252개의 참조 문헌을 제공했으나, 45.3%가 허구였고, 34.1%는 minor error가 있었으며, 20.6%만이 올바른 정보였습니다. 반면, RAG가 적용된 경우 정확성이 크게 향상되어 54.5%가 올바른 정보로 나타났고, 오류 비율은 18.8%로 줄어들었습니다. RAG에 의해 검색된 상위 10개 문서 중 62.5%가 LLM 반응에서 주요 참조로 선택되었으며, 평균 순위는 4.9에 달했습니다. RAG의 사용은 증거 귀속에서도 개선을 보여주었으나(5점 척도에서 1.85에서 2.49로 증가, P<0.001), 정확성은 약간 감소했습니다 (3.52에서 3.23로). RAG는 의료 분야에서의 하위 애플리케이션에 대한 우려를 해소하며, 허구 및 오류의 비율을 크게 감소시켰음을 제시합니다.



### LLM for Everyone: Representing the Underrepresented in Large Language Models (https://arxiv.org/abs/2409.13897)
Comments:
          PhD thesis

- **What's New**: 이 논문은 다국어 설정에서 특히 소외된 언어에 대한 대규모 언어 모델(LLMs)의 한계를 다루고 있습니다.

- **Technical Details**: 소외된 언어에서의 LLM 능력을 평가하기 위한 포괄적인 평가가 수행되었으며, 다국어 및 다문화 일반화의 과제가 드러났습니다. 제안된 방법론은 cross-lingual continual instruction tuning, retrieval-based cross-lingual in-context learning, 및 in-context query alignment을 포함합니다. 또한, 서로 다른 언어에서 작동하는 LLM 간의 문화적 가치 일치를 측정하기 위한 새로운 방법이 제안되었습니다.

- **Performance Highlights**: 이 연구는 소외된 언어에서도 효과적인 일반화를 가능하게 하여 다국어 및 다문화 조화성을 향상시키는 것을 목표로 하고 있습니다.



### Learning to Play Video Games with Intuitive Physics Priors (https://arxiv.org/abs/2409.13886)
Comments:
          7 pages, Accepted in Proceedings of the Annual Meeting of the Cognitive Science Society, Volume 46

- **What's New**: 이 논문에서는 비디오 게임 학습을 위한 객체 기반 입력 표현(object-based input representations)을 설계하여 여러 게임에서 잘 일반화할 수 있는 방법을 제시합니다. 이를 통해 인공지능 에이전트가 아기와 유사하게 제한된 경험을 바탕으로 게임을 배우는 방식을 연구합니다.

- **Technical Details**: 연구에서는 Q-learning 알고리즘을 사용하여 객체 범주 표현을 통해 게임의 상태 공간을 구성하며, 이러한 표현이 DQN 모델 대비 어떻게 학습 및 일반화 되는지를 비교 분석합니다. 또한, 'affordances'라는 개념을 도입해 물체의 상호작용을 혁신적으로 학습하는 방안을 모색합니다. 이 연구는 또한 인간의 공통적 직관 물리학(inductive biases) 지식을 활용해 게임 플레이를 배우는 접근 방법을 다룹니다.

- **Performance Highlights**: 제안된 방법론은 인간과 유사한 객체 상호작용을 통해 여러 비디오 게임을 효과적으로 학습할 수 있는 능력을 보여주었으며, 특히 낯선 객체에 대해 뛰어난 일반화 성능을 나타냈습니다. 이러한 연구 결과는 기계가 인간 중심으로 학습할 수 있는 새로운 가능성을 제시합니다.



### A Multi-LLM Debiasing Framework (https://arxiv.org/abs/2409.13884)
- **What's New**: 본 연구에서는 다중 LLM(multi-LLM) 접근 방식을 제안하여 LLM의 편향(bias)을 효과적으로 감소시키고자 합니다. 특별히, 중앙 집중식(centralized)과 분산식(decentralized) 두 가지 방법을 도입하여 비교하였으며, 분산식 방법이 우수한 성능을 보임을 확인했습니다. 이러한 방식은 사회적 그룹 내의 편향을 효과적으로 줄이는 데 기여합니다.

- **Technical Details**: 다중 LLM 프레임워크를 통해 여러 모델이 대화 방식으로 상호작용하며, 편향을 줄이는 방법을 제안합니다. 여기서 중앙 집중식 방식은 하나의 LLM이 대화를 지원하며, 분산식 방식은 모든 모델이 직접 소통합니다. 제안된 BBQ-Hard 벤치마크를 통해 두 방법을 평가하였고, BBQ-Hard 데이터셋은 LLM의 편향을 더 효과적으로 테스트할 수 있는 어려운 문제가 포함되어 있습니다.

- **Performance Highlights**: 다중 LLM 방법은 여러 사회적 그룹에서 기존의 기준 방법(baseline)보다 일관되게 더 우수한 성능을 보였습니다. 연구 결과는 다중 LLM 기반의 편향 감소 프레임워크가 LLM의 출력에서 편향을 유의미하게 줄일 수 있음을 시사합니다.



### Tabular Data Generation using Binary Diffusion (https://arxiv.org/abs/2409.13882)
- **What's New**: 이 논문에서는 기계 학습에서 중요하게 여겨지는 합성 테이블 데이터 생성에 대한 새로운 접근법을 소개합니다. 이 새로운 방법은 테이블 데이터를 고정 크기의 이진 표현으로 변환하는 무손실(binary transformation) 기법과 이진 데이터를 위해 설계된 새로운 생성 모델인 Binary Diffusion입니다.

- **Technical Details**: Binary Diffusion은 XOR 연산을 활용하여 노이즈를 추가하고 제거하는 방식을 채택합니다. 이 모델은 이진 교차 엔트로피(binary cross-entropy) 손실을 사용하여 훈련되며, 복잡한 전처리 과정이나 대규모 사전 훈련이 필요하지 않습니다. 데이터 변환 과정에서 연속형 데이터는 min-max 정규화를 거쳐 이진 표현으로 변환되며, 범주형 데이터는 바이너리 인코딩을 사용합니다.

- **Performance Highlights**: Binary Diffusion 모델은 Travel, Adult Income, Diabetes 데이터셋에서 기존 최첨단 모델보다 더 나은 성능을 보이며, 모델 크기도 상당히 작습니다. 또한, 대규모 데이터셋에 대한 사전 훈련이 필요 없기 때문에 처리 속도가 빠르고 효율적입니다.



### Instruct-Tuning Pretrained Causal Language Models for Ancient Greek Papyrology and Epigraphy (https://arxiv.org/abs/2409.13870)
Comments:
          7 pages, 1 table. Under review

- **What's New**: 이번 연구에서는 Meta의 Llama 3.1 8B Instruct 모델을 활용하여 고대 그리스 비문과 문서 파피루스의 연대 및 지리적 속성과 텍스트 복원 작업을 위한 미세 조정을 수행하였다. 이 모델은 기존 최고 기록을 초월하였으며, 특히 문서 복원에서 고전적 및 지리적 속성 부여에서 뛰어난 성능을 보였다.

- **Technical Details**: 연구진은 문서 비문과 파피루스를 위한 데이터 세트를 수집하고, 텍스트 복원, 지리적 속성 부여 및 연대 추정을 위한 전처리 과정을 거쳤다. 모델은 문자 에러율(CER) 22.5%와 지리적 속성 부여에서 top-1 정확도 75.0%를 달성하였다. 또한, 파피루스의 텍스트 복원에서는 CER 16.3%와 top-1 정확도 71.3%를 기록하였다.

- **Performance Highlights**: 미세 조정된 모델은 고대 비문의 텍스트 복원에서 평균 22.5%의 CER을 기록했으며, 지리적 속성 부여에서 75%의 top-1 정확도를 달성하였다. 또한 고대 그리스 문서 파피루스에 대한 새로운 기준을 수립하였으며, 연대 측정에서의 평균 편차는 26.2년으로, 기존에 비해 뛰어난 성능을 보였다.



### MAGICS: Adversarial RL with Minimax Actors Guided by Implicit Critic Stackelberg for Convergent Neural Synthesis of Robot Safety (https://arxiv.org/abs/2409.13867)
Comments:
          Algorithmic Foundations of Robotics (WAFR) XVI

- **What's New**: 본 논문에서는 Minimax Actors Guided by Implicit Critic Stackelberg (MAGICS)라는 새로운 적대적 강화 학습(Adversarial Reinforcement Learning, RL) 알고리즘을 소개합니다. 이 알고리즘은 최소최대 평형(minimax equilibrium) 솔루션으로의 지역적 수렴(local convergence)을 보장합니다.

- **Technical Details**: MAGICS는 로봇 안전 보장의 딥 RL 기반 알고리즘을 위한 지역적 수렴 보장을 제공합니다. 이 알고리즘은 시뮬레이션(OpenAI Gym)과 36차원 사족 보행 로봇 하드웨어 실험에서 검증되었습니다. MAGICS는 경쟁적 설정에서의 자동 안정성 문제를 해결하며, 제로섬(zero-sum) 설정에서도 수렴을 보장합니다.

- **Performance Highlights**: 실험 결과, MAGICS는 기존의 최신 신경 안전 합성(neural safety synthesis) 방법들보다 일관되게 더 우수한 로봇 제어 정책을 산출함을 보여줍니다.



### Wormhole: Concept-Aware Deep Representation Learning for Co-Evolving Sequences (https://arxiv.org/abs/2409.13857)
- **What's New**: 이 논문은 Wormhole이라는 새로운 딥 표현 학습 프레임워크를 소개하며, 이는 동시 발전하는 시계열 데이터에 대해 개념을 인식할 수 있도록 설계되었습니다. Wormhole은 동적 개념 및 그 전환을 효과적으로 파악할 수 있는 자가 표현 층과 시간의 부드러움 제약을 포함합니다.

- **Technical Details**: Wormhole 프레임워크는 멀티베리어트 시계열 데이터를 슬라이딩 윈도우 방식으로 여러 세그먼트로 나누고, 각 세그먼트는 모델의 입력으로 처리됩니다. 또한, 자가 표현 층이 서로 다른 시퀀스 내에서 본질적인 관계를 포착하도록 돕습니다. 모델은 인코더, 자가 표현 층, 디코더로 구성됩니다.

- **Performance Highlights**: 실험 결과, Wormhole은 기존의 배치 처리 방법에 비해 계산 효율성과 동시 발전하는 시퀀스 처리 능력에서 더욱 뛰어난 성능을 보였습니다. 또한, 개념 변화를 감지하는 데 있어 значительные 개선이 있음을 보여주었고, 복잡한 시간 패턴 분석에 유용한 도구로 자리잡을 가능성을 제시합니다.



### More Consideration for the Perceptron (https://arxiv.org/abs/2409.13854)
Comments:
          15 pages, 11 figures

- **What's New**: 본 논문에서는 기존의 perceptron을 개선한 gated perceptron을 소개합니다. 이는 기존 입력의 곱으로 계산된 추가 입력을 포함하여 특징 간의 비선형 상호작용을 캡처할 수 있게 하여 복잡한 데이터 세트에서 분류 및 회귀 능력을 크게 향상시킵니다.

- **Technical Details**: gated perceptron은 기존의 perceptron 구조에 AND 게이트를 추가하여 입력을 곱으로 계산한 새로운 입력을 도입합니다. 이를 통해 비선형 데이터를 더 효과적으로 처리할 수 있으며, XOR와 같은 비선형 문제를 해결할 수 있는 능력을 가지고 있습니다. 논문에서는 Iris 데이터 세트와 PIMA Indian 데이터 세트 및 Breast Cancer Wisconsin 데이터 세트를 사용하여 성능을 평가합니다.

- **Performance Highlights**: gated perceptron은 전통적인 perceptron에 비해 더 뚜렷한 결정 영역을 생성할 수 있으며, 선형 및 비선형 데이터에 대한 분류 성능이 크게 개선됩니다. 또한, 간단한 구조를 유지하면서 최첨단 분류기와 경쟁할 수 있는 성능을 보여줍니다.



### Unlocking Memorization in Large Language Models with Dynamic Soft Prompting (https://arxiv.org/abs/2409.13853)
- **What's New**: 본 논문은 LLM(대형 언어 모델)의 암기(memorization) 문제를 해결하기 위해 동적 소프트 프롬프트(dynamic soft prompts)를 사용하는 새로운 방법을 제안합니다. 이전 방법들은 입력의 변화에 반응하지 못하는 고정된 소프트 프롬프트만을 사용했으나, 본 방법은 입력 변화에 적응할 수 있는 프롬프트를 생성합니다.

- **Technical Details**: 제안된 방법은 transformer 기반 생성기(generator)를 활용하여 입력에 따라 동적으로 변경되는 소프트 프롬프트를 생성합니다. 이는 암기된 데이터를 보다 정확히 추출할 수 있게 해줍니다. 연구 결과, 본 방법은 기존 기술들과 비교하여 뛰어난 성능을 보였으며, 다양한 실험 환경에서 검증되었습니다.

- **Performance Highlights**: 본 방법은 텍스트 생성(task)과 코드 생성(task) 모두에서 vanilla 기준선 대비 각각 112.75% 및 32.26%의 최대 상대 개선을 달성했습니다. 이를 통해 동적 소프트 프롬프트의 효과성을 입증했습니다.



### Do language models practice what they preach? Examining language ideologies about gendered language reform encoded in LLMs (https://arxiv.org/abs/2409.13852)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)이 생성한 텍스트에서 언어 이데올로기(language ideologies)를 조사했으며, 특히 성별 구분이 있는 영어 표현의 개혁에 대한 사례 연구를 포함합니다. 이는 정치적 편향을 드러내고, LLM의 메타언어적 선호가 특정 정치 집단의 언어 이데올로기를 어떻게 암시적으로 전달하는지를 보여줍니다.

- **Technical Details**: 연구 결과, LLM은 '올바른(corresponding)' 또는 '자연스러운(natural)' 언어 사용에 대한 요청에 대해 보수적(conservative) 가치에 좀 더 유사한 언어를 생성하는 경향이 있음을 발견했습니다. 또한, LLM은 더 명확한 메타언어적(context) 맥락이 제공될 때 성 중립적(gender-neutral) 변형을 보다 자주 사용하는 내부 불일치(internal inconsistency)를 나타냈습니다.

- **Performance Highlights**: 이 연구는 LLM이 생성하는 텍스트에서 나타나는 언어 이데올로기가 사용자의 예상과 다를 수 있음을 강조하고, 이러한 결과가 가치 정렬(value alignment)과 관련된 더 넓은 함의를 갖고 있음을 논의합니다.



### STOP! Benchmarking Large Language Models with Sensitivity Testing on Offensive Progressions (https://arxiv.org/abs/2409.13843)
Comments:
          9 pages (excluding references), accepted to EMNLP 2024 Main Conference

- **What's New**: 본 연구에서는 Large Language Models (LLMs)에서의 명시적 및 암시적 편향을 평가하기 위한 새로운 접근 방식으로 Sensitivity Testing on Offensive Progressions (STOP) 데이터셋을 소개합니다. 이 데이터셋은 2,700개의 고유 문장을 포함하는 450개의 공격적 진행 상황을 제공하며, 다양한 심각도를 다룹니다.

- **Technical Details**: STOP 데이터셋은 9개의 인구 통계학적 그룹과 46개의 하위 인구 통계학적 그룹을 포괄하여 편향을 다양한 각도에서 평가할 수 있도록 설계되었습니다. 모델의 편향 인식 능력을 평가하기 위해 GPT-4, Mixtral, Llama 3와 같은 여러 주요 모델에 대한 실험이 수행되었습니다. 각 모델의 편향 인식 성공률은 19.3%에서 69.8%까지 다양했습니다.

- **Performance Highlights**: STOP 데이터셋을 활용하여 Llama 3-70b 모델을 파인 튜닝한 결과 BBQ, StereoSet 및 CrowS-Pairs 등에서 최대 191%의 높은 응답률을 달성하며 성능을 유지하거나 개선하는 성과를 보여주었습니다.



### Measuring Copyright Risks of Large Language Model via Partial Information Probing (https://arxiv.org/abs/2409.13831)
Comments:
          8 pages, 8 figures

- **What's New**: 본 논문은 Large Language Models (LLMs)가 저작권이 있는 콘텐츠를 생성할 수 있는 능력을 탐구합니다. 구체적으로, 저작권 내용의 일부 정보를 제공하고 생성된 콘텐츠와 원본 저작물 간의 겹침을 분석하는 방법을 사용하였습니다.

- **Technical Details**: 연구팀은 저작권이 있는 텍스트의 조각을 LLMs에 입력하고 이들이 이를 완성하도록 요청하는 실험을 진행하며, 여기에 Rouge-L Score를 사용하여 생성된 텍스트와 저작권 자료 간의 유사성을 평가하였습니다. 또한, 반복적인 프롬프트 기법을 사용하여 더 많은 저작권 침해 콘텐츠를 생성할 수 있는 가능성을 탐구했습니다.

- **Performance Highlights**: Llama, GPT-4-Turbo와 같은 대규모 매개변수를 가진 모델들이 특히 저작권 자료와 높은 유사성을 보였으며, 특정 유형의 콘텐츠에서는 성능 차이를 보였습니다. 예를 들어, GPT-4-Turbo는 노래 가사를 생성하는 데에서 더 높은 유사도를 나타냈습니다.



### On the Feasibility of Fully AI-automated Vishing Attacks (https://arxiv.org/abs/2409.13793)
- **What's New**: 이번 연구에서는 AI 기술의 발전으로 인해 vishing 공격의 심각성이 증가할 수 있는 가능성을 다룹니다. 특히, ViKing이라는 AI 기반 vishing 시스템을 소개하며, 이를 통해 공격자들이 피해자와의 전화 통화를 자동으로 진행하며 민감한 정보를 추출하는 방법을 제시합니다.

- **Technical Details**: ViKing 시스템은 대화 유도에 최적화된 Large Language Model (LLM)을 핵심 처리기로 사용하며, 음성 텍스트 변환 및 텍스트 음성 변환 모듈을 포함하여 완전 자동화된 vishing 봇을 구현합니다. 연구에서는 240명의 참가자를 대상으로 한 통제된 사회 실험을 통해 효율성을 검증하였습니다.

- **Performance Highlights**: 실험 결과, ViKing 봇이 52%의 참가자로부터 민감한 정보를 성공적으로 추출했으며, 위험을 알리지 않은 경우에는 이 비율이 77%까지 상승했습니다. 참가자의 46.25%가 ViKing을 신뢰할 수 있는 시스템으로 간주하였으며, 68.33%는 ViKing과의 상호작용이 현실적이라고 평가했습니다.



### Continual Learning for Multimodal Data Fusion of a Soft Gripper (https://arxiv.org/abs/2409.13792)
Comments:
          15 pages, 10 figures

- **What's New**: 이 논문에서는 계속적인 학습(Continual Learning, CL) 알고리즘을 통해 다양한 데이터 모달리티를 점진적으로 학습할 수 있는 방법을 제안합니다. 기존의 방법들은 새로운 도메인을 만날 때마다 모델을 처음부터 재학습하는 것을 요구하는 반면, 이 알고리즘은 클래스와 도메인 증가 학습(Class-Incremental and Domain-Incremental Learning) 시나리오를 활용하여 효율적으로 학습합니다.

- **Technical Details**: 제안하는 알고리즘은 비라벨 데이터가 풍부한 인공 환경에서 두 가지 모달리티(촉각 및 시각 데이터)를 조합하여 학습합니다. 이 알고리즘은 각 클래스에 대한 프로토타입을 저장하는 것만을 요구하며, FeCAM(Feature Covariance-Aware Metric) 알고리즘을 확장한 exFeCAM 알고리즘을 도입하여 incremental online semi-supervised learning을 활용합니다.

- **Performance Highlights**: 알고리즘의 효율성을 높이기 위해 새로운 다중 모달 비라벨 데이터셋과 Core50 데이터셋에서 실험을 진행하였으며, 다양한 알고리즘 구성 요소의 기여도를 검증하는 ablation study를 실시했습니다. 또한, 실제 환경에서의 객체 분류를 위한 실시간 실험을 통해 알고리즘의 견고성을 입증했습니다.



### Multi-omics data integration for early diagnosis of hepatocellular carcinoma (HCC) using machine learning (https://arxiv.org/abs/2409.13791)
Comments:
          21 pages, 5 figures

- **What's New**: 이번 연구에서는 다중 모달(multi-modal) 및 다중 오믹스(multi-omics) 데이터를 통합하여 환자의 질병 상태를 더욱 정확하게 모델링하는 방법을 제시하고 있습니다. 특히, 다양한 앙상블 머신러닝(ensemble machine learning) 알고리즘의 성능을 비교하여 서로 다른 모달리티(modality)의 데이터를 통합하는 데 초점을 맞추었습니다.

- **Technical Details**: 연구에서 테스트된 앙상블 방법은 다음과 같습니다: i) 하드(voting ensemble with hard vote) 및 소프트(voting ensemble with soft vote) 투표를 사용하는 투표 앙상블, ii) 메타 러너(meta learner), iii) 각각의 부스팅(boosting) 라운드에서 모달리티를 통합하는 하드/소프트 투표 및 메타 러너를 사용하는 다중 모달 Adaboost 모델, iv) PB-MVBoost 모델, v) 전문가 혼합 모델(mixture of experts model). 이러한 방법들은 단순한 연결(concatenation) 방식과 비교되었습니다.

- **Performance Highlights**: 수신자 작동 특성(receiver operating characteristic, ROC) 곡선 아래 면적을 성능 측정 지표로 사용하여, 0.85의 성능 값을 달성하는 모델을 개발하였습니다. PB-MVBoost 와 소프트 투표를 사용하는 Adaboost 두 가지 방법이 전체적으로 가장 뛰어난 성능을 보였으며, 선택된 특징(feature)의 안정성과 임상 서명(clinical signature) 크기 또한 분석하였습니다. 마지막으로 다중 모달 다중 클래스 데이터 통합에 대한 권장 사항을 제공합니다.



### Revisiting Synthetic Human Trajectories: Imitative Generation and Benchmarks Beyond Datasaurus (https://arxiv.org/abs/2409.13790)
- **What's New**: 본 논문에서는 MIRAGE라는 새로운 인간 모방 궤적 생성 모델을 제안합니다. 이 모델은 사용자의 의사결정 과정을 모방하여 궤적 생성을 수행하며, 기존의 통계적 분포를 맞추는 접근법의 한계를 극복합니다.

- **Technical Details**: MIRAGE는 신경망 기반의 Temporal Point Process (TPP)로 설계되어 연속 시간에서의 확률적 사건을 모델링합니다. 사용자의 탐색 및 선호 귀환(Exploration and Preferential Return) 결정을 모방하여 궤적을 생성합니다. 또한, 사용자 기반의 Variational Autoencoder를 통해 개인의 선호도를 반영합니다.

- **Performance Highlights**: MIRAGE는 세 가지 실제 사용자 궤적 데이터셋에서 비교 평가하였으며, 기존 최상의 벤치마크 대비 통계적 및 분포적 유사성을 59.0-71.5% 개선하고, 작업 기반 평가에서 성능을 10.9-33.4% 향상시켰습니다.



### Learning to Generalize Unseen Domains via Multi-Source Meta Learning for Text Classification (https://arxiv.org/abs/2409.13787)
- **What's New**: 이번 논문에서는 텍스트 분류의 multi-source Domain Generalization(DG) 문제를 다루며, 여러 개의 보이는 도메인을 활용하여 보이지 않는 도메인에서 높은 정확도를 달성할 수 있는 메타-러닝 프레임워크를 제안합니다. 이를 통해 도메인 관련 특징을 충분히 추출하고자 합니다.

- **Technical Details**: 제안된 프레임워크는 'Memory module'과 'Jury mechanism'을 포함하여 도메인 고유 및 도메인 불변 특징을 학습할 수 있도록 설계되었습니다. 이 프레임워크는 보이는 도메인에서 라벨링된 데이터만을 사용하여 훈련하고, 보이지 않는 도메인에서 테스트를 수행합니다. 메타-러닝 접근 방식을 통해 모델은 도메인을 구분하고, 보이지 않는 도메인에서의 분류 방식도 학습하게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 메타-러닝 프레임워크가 보이지 않는 도메인으로 일반화하는 능력을 효과적으로 향상시킬 수 있으며, 다중 소스 텍스트 분류 데이터셋에서 최신 기술(State-of-the-art) 방법들을 초월하는 성능을 보였습니다.



### A Value Based Parallel Update MCTS Method for Multi-Agent Cooperative Decision Making of Connected and Automated Vehicles (https://arxiv.org/abs/2409.13783)
Comments:
          arXiv admin note: text overlap with arXiv:2408.04295 by other authors

- **What's New**: 본 연구는 연결된 자동화 차량(CAVs)의 다중 차량 협업 운전을 위한 측면 및 장기적 합동 의사결정 문제를 해결하기 위해, 제한된 수명과 시간 할인 설정을 가진 다중 에이전트 마르코프 게임에서 병렬 업데이드를 포함한 몬테 카를로 트리 검색(MCTS) 방법을 제안합니다. 이 방법은 다중 차량의 조인트 액션 공간에서 발생하는 병렬 작용을 분석하여 잠재적 위험한 행동을 신속하게 배제함으로써 탐색 깊이를 향상시킵니다.

- **Technical Details**: MCTS 알고리즘은 네 단계(선택, 확장, 시뮬레이션, 반환)를 통해 작동하며, 각 단계에서 가장 유망한 행동을 선택하여 검색 트리를 구축하고 탐색합니다. 제안된 가치 기반 MCTS 방법은 다중 차량이 협력적으로 의사결정할 수 있도록 도와주며, 기존의 SOTA 강화 학습 알고리즘과 규칙 기반 방법보다 우수한 성능을 보였습니다. 목적에 맞춘 병렬 업데이트 방법은 특정한 환경에서 동적 문제가 어떻게 해결될 수 있는지를 보여줍니다.

- **Performance Highlights**: 다양한 실험 결과, 제안된 알고리즘은 일반적인 인류 운전자의 합리성을 초과하는 수준의 협력적인 주행 행위를 보였으며, 이는 교통 조건 개선에 기여했습니다. 이 알고리즘은 복잡한 교통 흐름에서도 높은 안정성을 보여주었으며, 트래픽 효율성과 안전성을 동시에 강화하여, CAVs의 협동 영역에서 인상적인 성과를 나타냈습니다.



### AutoPET III Challenge: Tumor Lesion Segmentation using ResEnc-Model Ensemb (https://arxiv.org/abs/2409.13779)
- **What's New**: 이번 연구에서는 다양한 암 진단을 위한 Positron Emission Tomography (PET)와 Computed Tomography (CT) 영상을 활용하여, 다중 트레이서(multi-tracer) 멀티 센터 환경에서의 종양 병변(segmentation) 분할을 위한 신뢰할 수 있는 딥러닝 모델 개발을 목표로 하였습니다.

- **Technical Details**: 3D Residual encoder U-Net을 사용한 autoPET III 도전 과제가 진행되었으며, Total Segmentator를 통한 데이터 전처리 및 데이터 증강(data augmentation) 기법을 적용하여, 훈련 데이터의 품질을 향상시켰습니다. nnU-Net ResEnc XL 아키텍처를 활용하여 1,611개의 이미지를 대상으로 훈련하였고, Dice 점수로 성능 평가를 하였습니다.

- **Performance Highlights**: 최종 평가에서, 2D 단일 모델이 0.9627의 Dice 점수를 기록하며, 5-fold 앙상블 모델 역시 0.9602를 기록하였습니다. 3D 모델은 0.7530의 Dice 점수를 달성하여, 2D 모델들에 비해 성능이 떨어졌으나, 3D 모델 앙상블을 통한 접근이 유효함을 보였습니다.



### Trustworthy Intrusion Detection: Confidence Estimation Using Latent Spac (https://arxiv.org/abs/2409.13774)
Comments:
          7 pages

- **What's New**: 본 연구는 Variational Autoencoder (VAE) 아키텍처를 활용하여 Intrusion Detection Systems (IDS)에서 이상 탐지의 신뢰성을 향상시키는 새로운 방법을 소개합니다. 잠재 공간 표현에서 파생된 신뢰성 메트릭을 개발하여 사이버 공격에 대한 IDS 예측의 신뢰성을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 본 연구에서는 NSL-KDD 데이터셋을 기반으로 방법론을 적용하며, VAE의 인코더와 디코더 구조를 활용하여 입력 데이터를 잠재 공간으로 인코딩하고 그 공간에서 새로운 데이터를 생성합니다. 이 과정에서 매할로비스 거리(Mahalanobis distance)를 사용하여 신뢰성 메트릭을 도출하고, 이를 통해 이상 탐지 결과의 신뢰도를 평가합니다.

- **Performance Highlights**: 이 연구의 방법론은 이상 탐지에서 0.45의 상관관계를 보여줍니다. 이는 본 접근법이 기존의 IDS 시스템보다 더 정확하고 신뢰할 수 있는 예측을 가능하게 함을 입증합니다. 이로 인해 사이버 위협을 식별하고 완화하는 데 있어 더 강력하고 효과적인 IDS 시스템 개발에 기여할 수 있습니다.



### A Case Study of Web App Coding with OpenAI Reasoning Models (https://arxiv.org/abs/2409.13773)
- **What's New**: 본 논문은 OpenAI의 최신 추론 모델(o1-preview 및 o1-mini)의 코딩 작업에 대한 사례 연구를 제공합니다. 이러한 모델들은 새로운 벤치마크인 WebApp1K-Duo의 도입과 함께, 기존의 단일 작업 벤치마크(WebApp1K)에서 SOTA(SOTA) 성능을 달성했습니다.

- **Technical Details**: WebApp1K-Duo 벤치마크는 과제가 두 배로 늘어나고, 테스트 케이스가 더 다양해져 o1 모델의 성능 저하를 초래했습니다. o1 모델은 비정상적이지만 올바른 테스트 케이스에서 지속적으로 실패하며, 이는 지침(comprehension) 이해 부족으로 인해 성능 변동성을 초래한다고 가정합니다. 이러한 문제는 특정 기대가 누락될 때 발생합니다.

- **Performance Highlights**: 단일 작업 평가에서 o1 모델은 SOTA를 7% 향상시켰으며 이전 비추론 모델들이 해결하지 못한 16개의 새로운 문제를 해결했습니다. 그러나 듀오 작업(deu-task) 평가에서는 o1 모델이 Claude 3.5보다 낮은 성능을 보였고 특정 테스트 형식에서 일관되게 실패했습니다.



### Magika: AI-Powered Content-Type Detection (https://arxiv.org/abs/2409.13768)
- **What's New**: Magika는 AI 기반의 새로운 콘텐츠 유형 감지 도구로, 단일 CPU에서 1MB의 메모리만으로도 실행할 수 있는 딥러닝 모델을 사용합니다. 이 도구는 99%의 평균 F1 점수를 달성하여 기존의 모든 콘텐츠 유형 감지 도구보다 뛰어난 성능을 보입니다.

- **Technical Details**: Magika는 파일의 시작, 중간, 끝에서 각각 512바이트를 입력으로 받아 콘텐츠 유형을 자동으로 식별합니다. 24M개의 파일 샘플을 이용해 113개의 표준 콘텐츠 유형을 학습했으며, 1.2M 샘플의 테스트 데이터셋에서 99%의 F1 점수를 기록했습니다. 또한 1MB 메모리로 단일 CPU에서 성능을 발휘하며, 1파일 당 5.77ms의 처리 속도를 기록했습니다.

- **Performance Highlights**: Magika는 기존의 도구(file, exiftool, trid, guesslang)와 비교 시 모든 도구보다 높은 성능을 보였으며, 이진 콘텐츠 유형에서는 4% F1 증가, 텍스트 콘텐츠 유형에서는 22% F1 증가, 전체적으로는 12%의 F1 증가를 기록했습니다. 현재 Gmail과 VirusTotal에 통합되어 있으며, GitHub에서 Apache 2 라이선스로 오픈소스화 되어 있습니다.



### Local Explanations and Self-Explanations for Assessing Faithfulness in black-box LLMs (https://arxiv.org/abs/2409.13764)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 충실성을 평가하는 새로운 작업을 도입했습니다. 이를 위해 로컬 섭동(local perturbations)과 자기 설명(self-explanations)을 활용한 효율적인 설명 가능성(explainability) 기술을 제안합니다.

- **Technical Details**: 이 연구에서는 LLM이 올바른 답변을 생성하는 데 필요한 충분하고 필수적인 부분을 식별하기 위해 일반적으로 사용되는 leave-one-out(LOO) 접근방식을 적용합니다. 우리는 Natural Questions 데이터셋을 사용하여 이 방법을 검증하며, LLM의 자기 설명이 실제로 모델 결정에 어떻게 기여하는지 평가하기 위해 고안된 메트릭을 제안합니다.

- **Performance Highlights**: 제안된 접근법은 모델의 결정 과정을 설명하고 충실성을 평가하는 데 매우 효과적임을 보여주었습니다. 특히 사용자의 질문에 대한 올바른 답변을 생성하는 데 중요한 키워드를 체계적으로 식별함으로써 LLM의 동작 방식에 대한 유의미한 통찰력을 제공합니다.



### Do Large Language Models Need a Content Delivery Network? (https://arxiv.org/abs/2409.13761)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)에서 새로운 지식을 유연하고 효율적으로 주입하는 방법으로 KV 캐시(KV caches)를 활용하는 방법에 대해 제안합니다. 기존의 fine-tuning과 in-context learning 방법에 비해 KV 캐시를 사용하는 것이 지식 주입의 모듈성(modularity) 및 효율성(efficiency)을 동시에 향상시킬 수 있다고 주장합니다. 이를 실현하기 위해 Knowledge Delivery Network(KDN)을 구상하였으며, 이는 LLM 서비스에서 KV 캐시를 동적으로 최적화하는 시스템 구성 요소입니다.

- **Technical Details**: KDN은 LLM에서 처리된 지식을 관리하는 백엔드 시스템으로, KV 캐시의 저장, 전송 및 조합을 최적화하는 기능을 갖추고 있습니다. KDN은 KV 캐시의 위계 구조와 압축을 활용하는 대규모 저장소, KV 캐시를 LLM 엔진 간에 빠르게 전송하는 스트리밍 시스템, 여러 조각의 지식을 결합하는 블렌딩 모듈을 포함합니다. 기존의 LLM 서비스 시스템과 달리 KV 캐시 관리와 LLM 서비스 엔진을 명확히 분리하여 모듈성과 효율성을 개선합니다.

- **Performance Highlights**: KDN 프로토타입 실험을 통해, KV 캐시 학습(KV-cache learning) 방법이 in-context learning과 fine-tuning 방법보다 더 나은 모듈성과 효율성을 보여줄 수 있음을 입증할 수 있는 초기 기술이 이미 연구되어 있다는 점을 강조합니다.



### Optimizing the Songwriting Process: Genre-Based Lyric Generation Using Deep Learning Models (https://arxiv.org/abs/2409.13758)
- **What's New**: 이 논문은 심층 학습 기술을 사용하여 전통적인 작사 과정을 단순화하는 방법을 제안합니다. 18,000곡의 Spotify 노래 데이터를 이용해 LSTM 기반 데이터 모델을 개발하여 장르별로 최적화된 가사를 생성하는 데 초점을 맞춰, 작사 과정을 가속화하는 목표를 세워 두 가지 모델을 비교했습니다.

- **Technical Details**: 본 연구에서는 seq2seq와 LSTM 모델을 사용하여 작업을 수행하였으며, T5 모델을 참조하여 사전 훈련된 모델과 독자적인 LSTM 모델 두 가지를 구축했습니다. 특별히, LSTM 모델의 입력으로 짧은 구문이나 단어를 받아 100자의 구문을 출력하도록 설정하고, 손실 함수는 Cross Entropy를 사용했습니다.

- **Performance Highlights**: 기본 모델이 ROUGE 메트릭에서 더 높은 리콜을 보인 반면, 두 모델 모두 BLEU 메트릭에서는 유사한 정밀도를 나타냈습니다. 생성된 가사 구문들이 특정 장르에 인식 가능하고 이해할 수 있는 수준에 이르렀음을 확인했습니다. 전체적으로, 가사 생성을 가속화하고 장르에 따른 가사를 효과적으로 생성할 수 있음을 보여주었습니다.



### Entity-Aware Self-Attention and Contextualized GCN for Enhanced Relation Extraction in Long Sentences (https://arxiv.org/abs/2409.13755)
- **What's New**: 본 논문에서는 기존의 의존성 기반 접근 방식의 한계를 극복하고자 Entity-aware Self-attention Contextualized GCN (ESC-GCN) 모델을 제안합니다. 이 모델은 입력 문장의 구문 구조와 의미적 문맥을 효과적으로 결합하여 관계 추출 성능을 향상시킵니다.

- **Technical Details**: ESC-GCN 모델은 상대 위치 self-attention을 통해 단어 위치와 관련된 전반적인 의미적 쌍상관성을 획득하고, 컨텍스트 그래프 컨볼루션 네트워크(Convolutional Networks)를 통해 단어 간의 복잡한 내부 문장 종속성을 포착합니다. 또한, entity-aware attention layer를 통해 최종 관계 예측을 위한 중요한 토큰을 동적으로 선택합니다.

- **Performance Highlights**: 다양한 작업에 대한 광범위한 실험에서 ESC-GCN 모델이 기존의 의존성 기반 및 시퀀스 기반 모델들에 비해 뛰어난 성능을 달성했음을 보여주었습니다. 특히 긴 문장에서의 엔티티 간 관계 추출에서 두드러진 성과를 보였습니다.



### Synergistic Simulations: Multi-Agent Problem Solving with Large Language Models (https://arxiv.org/abs/2409.13753)
Comments:
          15 pages, 5 figures, published in the MICS 2024 conference

- **What's New**: 이 논문은 Large Language Models (LLMs)을 활용하여 다중 에이전트 시스템을 개발하는 방법을 제시하고 있습니다. 특히, 시뮬레이션 환경에서 에이전트 간의 상호작용을 통합하여 인간 그룹의 문제 해결능력을 모델링하고자 합니다.

- **Technical Details**: 두 가지 시뮬레이션을 구현했습니다: 첫 번째는 두 명의 룸메이트가 있는 물리적 스튜디오 아파트 시뮬레이션, 두 번째는 에이전트들이 프로그래밍 과제를 협력하여 완수하는 시뮬레이션입니다. 이 논문에서는 멀티-에이전트 프레임워크에 대해 논의하고 각 시뮬레이션에서 에이전트의 성능을 분석합니다.

- **Performance Highlights**: 이 연구는 LLM이 인간 협력의 시너지를 어떻게 나타내는지를 보여주려 하며, 미래에 LLM의 응용 가능성을 높이는 데 기여할 수 있는 방향성을 모색합니다.



### Thinking Before Speaking: A Role-playing Model with Minds (https://arxiv.org/abs/2409.13752)
- **What's New**: 이번 논문에서는 Thinking Before Speaking (TBS) 모델을 제안하며, 역할 기반 대화에서 LLM의 성능을 개선하기 위한 새로운 접근법을 소개합니다. 이 모델은 캐릭터의 실제 시나리오를 바탕으로 데이터를 확장하고, 캐릭터의 사고 패턴을 반영하여 LLM이 더욱 사실적인 역할을 수행할 수 있도록 합니다.

- **Technical Details**: TBS 모델은 각 대화 쌍에 대해 캐릭터의 마음가짐을 보완하고, 특정 지식 이상의 질문을 포함한 일부 데이터를 추가하여 LLM을 미세 조정합니다. 이렇게 함으로써 LLM은 캐릭터의 사고 흐름과 논리를 채택하게 되며, 캐릭터의 지식 기반을 벗어나는 응답을 피할 수 있습니다. 이 연구는 새로운 데이터셋과 평가 지표를 마련하여 LLM의 능력을 시험합니다.

- **Performance Highlights**: 실험 결과, TBS 모델은 긴 대화 과정에서 톤, 지식, 마음가짐 측면에서 역할을 더 잘 모방할 수 있음을 보여주었으며, 이는 사용자 경험을 향상하는 데 기여합니다.



### KodeXv0.1: A Family of State-of-the-Art Financial Large Language Models (https://arxiv.org/abs/2409.13749)
Comments:
          11 pages, 8 figures

- **What's New**: 본 논문에서는 KodeXv0.1이라는 새로운 대형 언어 모델 패밀리를 소개합니다. 이 모델은 GPT-4를 초월하는 성능을 보이며, 주로 재무 분야에서 질문 답변을 수행하는 데 최적화되어 있습니다.

- **Technical Details**: KodeXv0.1은 Llama 3.1 8B 및 70B의 기본 변형을 사용하여 특수한 재무 영역을 위해 커스터마이즈된 교육 체계를 통해 발전되었습니다. 이를 위해 공개적으로 사용 가능한 재무 문서를 대량으로 수집하고 처리하여 Context-Question-Answer triplet 형태의 고품질 합성 데이터셋을 생성했습니다. 모델 튜닝은 RAG-aware 4bit LoRA 방법을 사용하여 수행되었습니다.

- **Performance Highlights**: 모델 평가 결과 KodeX-8Bv0.1은 동일한 매개변수 범위 내에서 최신 모델보다 최대 9.24% 더 신뢰성이 높은 결과를 보여주었고, GPT-4보다도 최대 7.07% 우수한 성능을 발휘했습니다. KodeX-70Bv0.1은 모든 테스트 벤치마크에서 GPT-4의 성능을 초과하는 개선을 나타냈습니다.



### TheraGen: Therapy for Every Generation (https://arxiv.org/abs/2409.13748)
Comments:
          12 pages, 11 figures

- **What's New**: 이번 논문에서는 LLaMA 2 7B 모델을 활용하여 개발한 고급 AI 기반 정신 건강 챗봇인 TheraGen을 소개합니다. TheraGen은 100만 개의 대화 입력 데이터를 이용하여 개인화되고 연민이 담긴 정신 건강 관리를 제공하며, 최근의 언어 모델과 트랜스포머 아키텍처의 발전을 기반으로 합니다.

- **Technical Details**: TheraGen은 transfer learning, fine-tuning 및 고급 훈련 기법을 활용하여 최적의 성능을 달성합니다. 클라우드 기반 아키텍처를 통해 고성능 및 가용성이 보장되고, 24시간 언제든지 접근할 수 있는 지원을 제공합니다. 이 시스템은 사용자 친화적인 인터페이스를 제공하여 공감적인 반응과 근거 기반 대처 전략을 제공합니다.

- **Performance Highlights**: 사용자 만족도 평가 결과에 따르면, 94%의 사용자들이 정신적 웰빙이 향상되었다고 보고했습니다. 또한, BLEU 점수는 0.67, ROUGE 점수는 0.62로 응답 정확성이 높음을 나타냅니다. 평균 응답 시간은 1395 밀리초로, 실시간으로 효율적인 지원을 보장합니다.



### When Less Is Not More: Large Language Models Normalize Less-Frequent Terms with Lower Accuracy (https://arxiv.org/abs/2409.13746)
- **What's New**: 이 연구는 대규모 언어 모델(GPT-4o)이 인체 표현형 온톨로지(HPO)를 기반으로 한 용어 정규화에서 11,225개의 고유한 용어를 처리할 때 단 13.1%의 정확도를 달성했다는 점을 강조합니다. 이는 저주파 및 긴 용어가 더 많은 정규화 오류를 초래한다는 것을 보여줍니다.

- **Technical Details**: 연구에서는 268,776개의 표현형 주석을 가지는 HPO 데이터셋을 활용하여, 용어의 빈도와 길이에 따라 정규화 정확도를 분석했습니다. 특히, SHAP 및 순열 방법을 사용하여 정규화 오류의 주요 예측 요인으로 낮은 용어 빈도를 확인했습니다.

- **Performance Highlights**: 정확도는 높은 빈도에서 시작하여, 주파수가 낮아질수록 급격히 감소했습니다. 특히 용어의 길이가 길어질수록 정확도가 떨어지는 경향이 있으며, ANOVA 분석을 통해 이 두 요인 간의 중요한 상호작용이 발견되었습니다.



### Context-Aware Membership Inference Attacks against Pre-trained Large Language Models (https://arxiv.org/abs/2409.13745)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 사전 훈련 과정에서 발생하는 맴버십 추론 공격(Membership Inference Attack, MIA)의 새로운 접근 방식을 제안합니다. 기존 MIA는 LLM의 순차적 텍스트 생성 과정을 고려하지 않았습니다.  본 연구에서는 MIA 통계 테스트를 데이터 포인트의 서브시퀀스(perplexity dynamics)에 적응시켜 효과적인 공격을 수행했습니다.

- **Technical Details**: CAMIA(상황 인식 맴버십 추론 공격) 프레임워크를 설계하여 LLM의 다음 토큰 예측 손실(sequence of next-token prediction losses)에서 조정된 맴버십 정보를 추출합니다. 이 방법은 프리픽스(prefix)의 길이, 다양성, perplexity 등의 맥락적 요소를 반영하여 조정된 테스트를 수행합니다. 손실의 기울기, 변동성, 이상치는 멤버와 비멤버를 구분하는 데 중요한 지표입니다. 이러한 동적 행동을 통해 모델의 다양한 상황에 따라 공격 결정을 조정합니다.

- **Performance Highlights**: CAMIA는 Pythia와 GPT-Neo 모델을 포함한 9999개의 사전 훈련된 LLM을 대상으로 평가했으며, 기존 MIA보다 3배 더 많은 멤버를 성공적으로 식별했습니다. 예를 들어, Pythia 모델에서 CAMIA는 1%의 특정 잘못 예측률(FPR)에서 3.35배 높은 진짜 긍정률(TPR)을 달성했습니다. GPT-Neo 모델에서도 TPR이 20% 증가했습니다. 이는 다양한 데이터 도메인에서 일관되게 높은 성능을 보였습니다.



### A Simplified Retriever to Improve Accuracy of Phenotype Normalizations by Large Language Models (https://arxiv.org/abs/2409.13744)
Comments:
          Submitted to Frontiers in Digital Health

- **What's New**: 이 연구는 생물 의학에서 주로 사용되는 딥 러닝 모델을 사용하여 표현형 용어 정규화의 정확성을 향상시키는 새로운 방법을 제안하고 있습니다. 특히 BioBERT를 사용한 문맥적 단어 임베딩에 기반하여 복잡한 정의 생성 없이 빠르고 효과적으로 후보 용어를 매칭시키는 간소화된 검색기를 도입했습니다.

- **Technical Details**: 제안된 방법은 검증된 1,820개의 표현형 용어에 대해 Human Phenotype Ontology (HPO)와의 코사인 유사성을 기준으로 키워드 후보를 선택하고, 이를 통해 LLM이 의미론적으로 가장 적당한 정규화를 선택하게 합니다. LLM은 GPT-4o를 사용하게 되며, 이 과정에서 BioBERT를 통한 두 가지 실험 조건과 LLM + Retriever 방식이 비교됩니다.

- **Performance Highlights**: 정규화 정확도는 기존 62.3%에서 90.3%로 향상되었습니다. 특히 LLM + Retriever 방법은 BioBERT 방법보다 높은 정확도를 보이며, 자동화된 정규화 솔루션의 필요성을 충족할 수 있는 잠재력을 보여줍니다.



### Language agents achieve superhuman synthesis of scientific knowledg (https://arxiv.org/abs/2409.13740)
- **What's New**: 이번 연구에서는 PaperQA2라는 고급 언어 모델을 개발하여 과학 문헌 검색 과제에서 인간 전문가와의 성능 비교를 수행했습니다. 결과적으로 PaperQA2는 정보 검색, 요약 및 모순 탐지 작업에서 주제 전문가를 초월할 수 있음을 입증하였습니다.

- **Technical Details**: 연구의 핵심은 PaperQA2를 활용하여 적용된 LitQA2 벤치마크를 통해 과학 문헌에 대한 정보 검색, 요약 및 모순 탐지의 성능을 평가한 점입니다. PaperQA2는 ‘retrieval-augmented generation (RAG)’ 접근 방식을 사용하여, 여러 단계의 작업을 통해 최종 응답을 생성합니다. 각 질문에 대해 평균 14.5개의 논문을 활용하였으며, 정확도는 66.0%로 나타났습니다.

- **Performance Highlights**: PaperQA2는 LitQA2 벤치마크에서 85.2%의 정밀도와 66.0%의 정확도를 기록하였으며, 생물학 논문에서 평균 2.34개의 모순을 발견하는 데 성공했습니다. 이 연구는 AI 모델이 특정 과학 문헌 작업에서 인간 전문가보다 뛰어난 성능을 보여줄 수 있음을 실증적으로 나타냈습니다.



### RNR: Teaching Large Language Models to Follow Roles and Rules (https://arxiv.org/abs/2409.13733)
- **What's New**: 이 논문에서는 기존의 Instruction Fine-Tuning (IFT) 모델이 복잡한 역할과 규칙을 따르는 데 실패한다는 문제를 해결하기 위해 RoleNRules (RNR)라는 데이터 생성 파이프라인을 제안합니다. 이 파이프라인은 기존 IFT 지침으로부터 다양한 역할과 규칙을 자동으로 생성하여 LLM 모델의 성능을 향상시킵니다.

- **Technical Details**: RoleNRules는 고유한 (system prompt, instruction, response) 트리플을 생성하여 모델이 복잡한 시스템 prompt를 따르도록 훈련할 수 있도록 설계된 데이터 생성 파이프라인입니다. 이 과정에서 LLM을 통해 다양한 역할(description)과 규칙(rules)을 생성하고, 생성된 시스템 prompt와 원래 지침을 기반으로 응답을 생성합니다.

- **Performance Highlights**: RNR로 훈련된 모델은 Traditional Instruction Fine-Tuning에 비해 25% 이상 증가한 규칙 준수 pass-rate를 기록했으며, 일반적인 지침을 따르는 성능 저하 없이 복잡한 지침을 계속해서 따를 수 있는 능력을 유지했습니다.



### KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation (https://arxiv.org/abs/2409.13731)
Comments:
          33 pages

- **What's New**: Knowledge Augmented Generation (KAG)은 전문 도메인 지식 서비스를 위한 새로운 프레임워크로, 기존의 Retrieval-Augmented Generation (RAG)의 한계를 극복하기 위해 개발되었습니다. KAG는 Knowledge Graph (KG)와 벡터 검색의 장점을 결합하여 대규모 언어 모델(LLMs)과 KG의 상호 작용을 통해 생성 및 추론 성능을 향상시키는 것을 목표로 합니다.

- **Technical Details**: KAG는 다섯 가지 핵심 요소를 통해 LLM 친화적인 지식 표현(LLMFriSPG), 지식 그래프와 원래 텍스트 청크 간의 상호 색인화, 논리적 형태 기반의 하이브리드 추론 엔진, 의미론적 추론에 따른 지식 정렬, KAG를 위한 모델 기능 향상을 설계하였습니다. 이 프레임워크는 복잡한 Q&A 데이터셋에서 성능을 평가하여 상당한 개선 결과를 나타냈습니다.

- **Performance Highlights**: KAG는 2wiki에서 19.6%, hotpotQA에서 33.5%의 F1 점수를 개선하며 기존의 RAG 방식보다 전문성이 크게 강화된 결과를 보여주었습니다. 또한, KAG는 Ant Group의 E-Government 및 E-Health Q&A 작업에 적용되어 전통적인 RAG 방법에 비해 높은 정확도를 기록하였습니다.



### MathGLM-Vision: Solving Mathematical Problems with Multi-Modal Large Language Mod (https://arxiv.org/abs/2409.13729)
Comments:
          30 pages,19 figures

- **What's New**: 이 논문에서는 다양한 시각적 요소가 포함된 수학 문제를 해결하기 위한 MathGLM-Vision이라는 다중 모달 대형 언어 모델(Multi-Modal Large Language Model, MLLM)을 소개합니다. 특히 이 모델은 MathVL이라는 세밀하게 조정된 데이터셋을 사용하여 수학 문제를 해결하는 데 필요한 시각적 정보를 통합하도록 설계되었습니다.

- **Technical Details**: MathGLM-Vision은 GLM-4V-9B, CogVLM2, CogVLM-32B를 기초 모델로 하여 Fine-Tuning(미세 조정)을 진행하였으며, 다양한 매개변수 크기로 구성됩니다. MathVL 데이터셋은 산술, 대수학, 기하학 등 다양한 수학적 주제를 포함하고, 각 문제에는 단계별 솔루션이 제공되어 모델의 문제 해결능력을 향상시킵니다.

- **Performance Highlights**: MathGLM-Vision은 MathVL-test에서 시각적 입력을 포함한 경우, 텍스트 입력만을 사용한 모델에 비해 더 우수한 성능을 보였습니다. 예를 들어, 기하학 문제 해결(minitest)에서는, MathGLM-Vision-9B가 GLM-4V-9B에 대해 39.68% 성능 개선을, MathGLM-Vision-19B가 CogVLM2에 대해 65.06% 성능 개선을 보였습니다.



### Multilingual Dyadic Interaction Corpus NoXi+J: Toward Understanding Asian-European Non-verbal Cultural Characteristics and their Influences on Engagemen (https://arxiv.org/abs/2409.13726)
Comments:
          8 pages. 6 figures. International Conference on Multimodal Interaction, November 4-8, 2024, San Jose, Costa Rica

- **What's New**: 이번 연구에서는 비언어적 행동이 문화에 따라 어떻게 차별화되는지를 분석하고, 이러한 차이가 대화의 참여도(engagement) 인식에 미치는 영향을 평가하기 위해 다국어(multi-lingual) 비언어적 기능을 COMPUTATIONAL 분석했습니다. 이를 위해 기존의 NoXi 데이터셋을 확장하여 일본어와 중국어로 이루어진 대화 데이터가 포함된 새로운 데이터셋 NoXi+J를 구성했습니다.

- **Technical Details**: 비언어적 특성(non-verbal features)에는 음성 음향(speech acoustics), 표정(facial expressions), 백채널(backchanneling), 제스처(gestures)가 포함됩니다. 다양한 패턴 인식 기법을 통해 이러한 특성을 추출하고, 통계적 분석을 통해 각 언어에서 문화적으로 의존적이고 독립적인 특징을 식별했습니다. 최종적으로 LSTM(Long Short-Term Memory) 모델을 훈련하여 대화의 참여도를 예측하고, SHAP(Shapley Additive Explanations) 분석을 통해 입력 특성과 문화적 특성 간의 상관관계를 분석했습니다.

- **Performance Highlights**: 여섯 개의 기계 학습 모델이 NoXi+J 데이터셋의 서로 다른 언어 화자(subsets)에서 훈련되어 성능을 평가했습니다. 모델의 성능 결과는 분석 결과와 상관관계를 나타내며, 다양한 언어의 비언어적 특성에 따라 성능이 달라지는 것을 확인했습니다. 이 연구는 문화 차이에 대한 인식뿐만 아니라, 비언어적 소통과 대화 참여도의 예측에 있어 기계 학습의 중요성을 강조합니다.



### Logically Consistent Language Models via Neuro-Symbolic Integration (https://arxiv.org/abs/2409.13724)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM)의 논리적 일관성을 향상시키기 위해 신경-기호적 추론(neuro-symbolic reasoning)을 기반으로 한 손실(loss) 개념을 도입했습니다. 이를 통해 LLM이 외부의 사실(facts)과 규칙에 논리적으로 일관되도록 학습할 수 있게 하여, LLM의 자가 일관성(self-consistency)을 개선했습니다.

- **Technical Details**: 연구에서는 LLM을 훈련시키는 동안, 강화된 샘플 모델의 확률을 최대화하는 원칙적 목적(objective)을 수립했습니다. 이 방법은 추론 레퍼토리를 잠재적으로 확장하면서, LLM이 제공된 논리적 제약조건에 따라 진실성을 유지하도록 합니다. 실험을 통해, 제한된 사실 세트에 대해 훈련 받은 LLM이 새로운 사실에 대한 진실 신념을 학습할 수 있음을 보였습니다.

- **Performance Highlights**: LoCo-LMs(논리적으로 일관된 LLMs)로 명명된 이번 모델은 외부 해결기(solvers)에 의존하지 않고도, 자가 일관성과 사실성을 향상시킨 것으로 나타났습니다. 제한된 데이터 환경에서도, 기존의 감독적 fine-tuning에 비해 더 우수한 성능을 보였습니다.



### Explainable Malware Analysis: Concepts, Approaches and Challenges (https://arxiv.org/abs/2409.13723)
- **What's New**: 이번 논문에서는 Explainable AI (XAI)의 발전이 악성 코드 탐지에서의 중요성을 강조하며, 기존의 기계 학습(ML) 기반 악성 코드 탐지 기법과 설명 가능한 AI 접근법을 포괄적으로 검토하고 있습니다. 이 설문조사는 XAI 응용 프로그램에 대해 연구자들이 관심을 가질 수 있도록 안내하는 역할을 합니다.

- **Technical Details**: 하이브리드 분석, 정적 분석(static analysis), 동적 분석(dynamic analysis)과 같은 다양한 악성 코드 탐지 기술을 소개하며, XAI가 이러한 기법의 투명성을 어떻게 향상시키는지를 다룹니다. 또한, ML 모델의 결정 과정이 불투명한 문제에 대한 해결 방안으로 XAI의 필요성을 논의합니다.

- **Performance Highlights**: 이 논문은 악성 코드 탐지 분야에서의 XAI 모델 및 기술을 광범위하게 조사하여 이 분야의 최신 동향과 발전을 제시함으로써, 기존 연구의 주요 한계와 도전 과제를 명확히 지적하고 메소드를 설명할 수 있는 새로운 분류 체계를 도입합니다.



### DiVA-DocRE: A Discriminative and Voice-Aware Paradigm for Document-Level Relation Extraction (https://arxiv.org/abs/2409.13717)
- **What's New**: 이번 연구에서는 Document-level Relation Triplet Extraction (DocRTE)을 위한 혁신적인 접근 방식인 DiVA(Discriminative and Voice Aware Paradigm)를 소개합니다. DiVA는 문서 수준에서의 관계 추출을 단순화하여 관계를 식별하기 위해 단순히 문서를 입력하는 방식으로 작동합니다.

- **Technical Details**: DiVA는 두 가지 주요 단계로 구성됩니다: (1) 관계 추출을 위한 문서 수준의 차별화(paradigm) 방식, (2) 추출된 관계에 기반하여 주체와 객체 엔티티를 식별하는 음성 인식(voice-aware) 방식. 이러한 접근 방식을 통해 모델은 문서 전체의 맥락을 이해하고 관계의 방향성 및 음성의 영향을 감지할 수 있습니다.

- **Performance Highlights**: 실험 결과, Re-DocRED 및 DocRED 데이터셋에서 DiVA는 DocRTE 작업에 대해 최고의 성능(SOTA)을 기록했습니다. 기존 방법론들과 비교했을 때, 더 많은 관계 유형을 처리하고, 음성을 고려한 주체와 객체 구분에서 우수한 결과를 보여줍니다.



### Introducing MeMo: A Multimodal Dataset for Memory Modelling in Multiparty Conversations (https://arxiv.org/abs/2409.13715)
- **What's New**: MeMo 코퍼스는 참가자의 기억 유지 보고서로 주석이 달린 최초의 대화형 데이터세트로, 컴퓨터 모델링을 위한 소중한 자원으로 제공됩니다.

- **Technical Details**: MeMo 코퍼스는 Covid-19 주제에 대한 31시간 분량의 소규모 그룹 토론을 포함하며, 행동 및 지각 측정이 검증된 데이터와 함께 오디오, 비디오, 다중 모달 주석을 통합하고 있습니다. 이 데이터셋은 대화 기억 및 그룹 역학 연구에서 유용한 자료가 됩니다.

- **Performance Highlights**: MeMo 코퍼스는 대화 기억 모델을 구축하는 데 활용 가능하며, 이를 통해 사용자의 기억 및 사회적 상호작용에 대한 이해를 진전시킬 수 있습니다.



### TracrBench: Generating Interpretability Testbeds with Large Language Models (https://arxiv.org/abs/2409.13714)
Comments:
          6 pages + appendix, 4 figures, ICML Mechanistic Interpretability Workshop

- **What's New**: 이 연구에서는 트랜스포머(tansformer) 기반 언어 모델의 해석 가능성(interpretability)을 평가하기 위한 새로운 접근법을 제시합니다. 특히 TracrBench라는 새로운 데이터셋을 소개하며, 이는 121개의 수작업으로 작성된 RASP 프로그램과 LLM(대형 언어 모델) 생성, 인간 검증의 결과로 구성되어 있습니다.

- **Technical Details**: Tracr는 RASP에서 본래의 사실 기반 매핑(ground truth mappings)을 가진 컴파일된 트랜스포머 모델을 생성하는 방법입니다. TracrBench는 해석 가능성 평가 방법들을 검증하기 위해 고안된 테스트베드(test bed)로, LLM을 활용하여 RASP 프로그램을 자동으로 생성하고자 했으나, 이 과정에서 많은 도전과제가 있음을 발견했습니다.

- **Performance Highlights**: 최신 LLM인 GPT-4-turbo는 20-shot 프롬프트와 best-of-5 샘플링을 사용하였으나, 총 101개 테스트 프로그램 중 57개만을 올바르게 구현했습니다. TracrBench는 이러한 과정을 통해 해석 가능성 방법의 평가 및 비교를 위한 가치 있는 테스트베드 역할을 목표로 하고 있습니다.



### Good Idea or Not, Representation of LLM Could (https://arxiv.org/abs/2409.13712)
- **What's New**: 이 논문은 과학적 아이디어를 정량적으로 평가하는 새로운 프레임워크를 제안하며, 대형 언어 모델(LLMs)로부터 얻은 표현을 활용하여 아이디어의 가치를 정량화하는 방법을 탐구합니다. 또한, 약 4,000개의 원고로 구성된 벤치마크 데이터 세트를 공개합니다.

- **Technical Details**: 우리는 LLM의 특정 계층에서 생산된 표현을 사용하여 아이디어의 가치를 정량화하는 프레임워크를 수립하였습니다. 이 연구는 LLM의 표현을 통해 텍스트의 의미론적 특징을 인코딩하고 이를 다양한 아이디어 평가 방법과 결합하여 학문적 질적 평가를 목표로 합니다.

- **Performance Highlights**: 실험 결과, LLM에서 생성된 표현은 인간의 판단과 높은 일관성을 보여주었으며, LLM의 중간 및 후위 계층에서 얻어진 표현이 아이디어 품질 평가에 더 적합하다는 것을 알았습니다. 이러한 접근법은 적은 양의 데이터로도 높은 정확도를 달성할 수 있음을 입증하였습니다.



### WebQuest: A Benchmark for Multimodal QA on Web Page Sequences (https://arxiv.org/abs/2409.13711)
- **What's New**: WebQuest는 다중 페이지 질문-답변(Question-Answering, QA) 데이터셋으로, 웹 상호작용에서의 정보 검색 및 추론을 동시에 요구하는 새로운 벤치마크를 제시합니다. 이 데이터셋은 단일 화면, 다중 화면 및 내비게이션 경로 기반의 질문을 포함하고 있어 기존의 다단계 웹 탐색 방식과는 차별화된 접근을 보여줍니다.

- **Technical Details**: WebQuest 데이터셋은 세 가지 질문 카테고리(단일 화면 QA, 다중 화면 QA, 내비게이션 경로 기반 질문)를 포함하여, 사용자 행동에 기반한 웹 상호작용 시퀀스를 반영하며, 다양한 멀티모달 모델(GPT-4V, Gemini Flash, Claude 3 등)을 평가합니다. 특히, Chain-of-Thought prompting 기법을 적용하여 다중 화면 추론 능력을 향상시키는 방법을 모색합니다.

- **Performance Highlights**: 모델 평가는 단일 페이지와 다중 페이지 추론 간의 성능 차이를 보여주며, WebQuest는 기존 QA 기반 콘텐츠 이해와 에이전트 모델 연구 간의 격차를 해소하는 새로운 QA 모드를 제공합니다. 또한, 데이터셋은 다양한 모델의 능력을 자세히 평가할 수 있는 3개의 데이터 하위 집합을 포함하고 있습니다.



### Column Vocabulary Association (CVA): semantic interpretation of dataless tables (https://arxiv.org/abs/2409.13709)
- **What's New**: 이번 논문에서는 Semantic Table Interpretation (STI) 분야에서 새로운 과제인 'Metadata to KG' 트랙을 소개하며, 기존의 데이터에 접근하지 않고 메타데이터 정보만으로 테이블 해석을 수행하는 방법을 검토하였습니다.

- **Technical Details**: 주요 내용은 Column Vocabulary Association (CVA)라는 새로운 개념을 도입하였고, 다양한 Large Language Models (LLMs)과 Retrieval Augmented Generation (RAG) 접근 방식을 통해 CVA 작업을 평가하였습니다. 실험에는 상업적 GPT 모델(예: gpt-3.5-turbo-0.125, gpt-4o, gpt-4-turbo)과 오픈 소스 모델(예: llama3-80b, llama3-7b) 총 7개 모델이 포함되었습니다.

- **Performance Highlights**: 결과적으로, LLM은 일반적으로 온도 설정이 1.0 이하일 때 우수한 성과를 내었고, 특정 사례에서는 100% 정확도를 달성하였습니다. 그러나 데이터의 특성에 따라 전통적인 방법이 LLM의 성과를 초월하는 경우도 있었습니다.



### Towards Safe Multilingual Frontier AI (https://arxiv.org/abs/2409.13708)
Comments:
          23 pages; 1 figure and 10 supplementary figures

- **What's New**: 이번 연구는 효과적으로 다언어 지원을 제공하는 LLMs(대형 언어 모델)의 개발을 강조하며, 특히 다국어 jailbreak(탈옥) 공격으로부터 모델의 안전성을 확보하기 위한 정책 제안을 제시합니다. 이를 통해 AI의 언어적 포용성을 증대시키고, EU의 법적 틀에 맞춘 정책 조치를 모색합니다.

- **Technical Details**: 5개의 선도적인 LLM을 대상으로 EU의 24개 공식 언어에 대한 다국어 jailbreak 공격의 취약성을 정량적으로 분석하였습니다. 다국어 기능과 취약성 간의 관계를 평가하기 위해,  언어 자원화 수준에 대한 새로운 가설을 제안하였습니다.

- **Performance Highlights**: 이 연구에서 제안된 정책은 AI 안전성을 개선하고, 기존의 기술적 공간과 정책적 요구 간의 간극을 줄이기 위해 설계되었습니다. 특히, 저자들은 다국어 AI 개발에 대한 국가적 지원과 다국어 기능의 의무 평가를 포함한 여러 가지 정책 권고안을 제시하고, 이러한 조치들이 AI의 효과성과 안전성을 향상시킬 수 있을 것이라고 강조합니다.



### Retrieval Augmented Generation-Based Incident Resolution Recommendation System for IT Suppor (https://arxiv.org/abs/2409.13707)
Comments:
          7 pages, 3 figures, 6 tables

- **What's New**: 이 연구는 IT 지원 도메인에서의 솔루션 추천 시스템을 위해 개발된 Retrieval Augmented Generation(RAG) 시스템을 소개합니다. 특히, IBM Slate 125m 모델을 사용하여 단일-턴과 다중-턴 IT 지원 사례를 분류하는 새로운 접근법과 성능을 보고합니다.

- **Technical Details**: 시스템은 네 가지 주요 구성 요소로 이루어져 있습니다: encoder-only transformer classifier, query generation system, retriever system, 그리고 answer generator system. 데이터 수집은 약 19,000개의 실제 지원 사례를 기반으로 하며 다양한 소프트웨어 제품에서 수집되었습니다. CNN(Convolutional Neural Networks) 및 cosine similarity를 활용하여 문서를 검색하고 재랭크합니다.

- **Performance Highlights**: 연구 결과, 작은 모델들이 RAG 사건 해결 사용 사례에서 매우 큰 모델들과 동등하거나 더 나은 성능을 보여주었다고 보고합니다. 최종적으로 F1 점수가 0.65에 이르고, 클래스 분류 정확도가 0.54, 재현율이 0.80으로 나타났습니다.



### Debiasing Text Safety Classifiers through a Fairness-Aware Ensemb (https://arxiv.org/abs/2409.13705)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 안전성과 공정성을 보장하기 위한 경량의 후처리(post-processing) 방법을 제시합니다. 기존의 데이터 불균형 문제를 해결하는 새로운 메트릭과 데이터 세트를 도입하여, 편향된 모델의 문제를 완화하고자 합니다.

- **Technical Details**: 편향을 완화하기 위해, 우리는 'Fair Data Reweighting (FDW)'라는 두 단계 방법을 적용하여 교육 세트를 재조정하고, 최종적으로 안전 분류기를 개선하는 앙상블(ensemble) 모델을 구축합니다. 또한, 두 가지 새로운 메트릭인 Average Counterfactual Variance (ACV)와 Sliced Averages (SA)를 도입하여 모델의 공정성을 평가합니다.

- **Performance Highlights**: 우리가 제안한 방법은 모델의 성능에 미치는 영향이 최소한인 상태에서 반사실적 공정성을 개선하는 것으로 나타났습니다. 또한, 새로운 Open AI 데이터 세트와 사용자 지정 프롬프트를 기반으로 한 LLM 생성 데이터 세트를 마련하여, 이들 데이터는 신원을 기반으로 균형이 잡힌 특징을 가집니다.



### CA-BERT: Leveraging Context Awareness for Enhanced Multi-Turn Chat Interaction (https://arxiv.org/abs/2409.13701)
Comments:
          This paper has been accepted by ICBASE 2024

- **What's New**: 이 논문은 기존의 BERT 모델을 기반으로 다중 대화 상호작용에서의 맥락의 필요성을 감지하는 데 특화된 Context-Aware BERT (CA-BERT) 모델을 소개합니다. 이 모델은 맥락 필요성을 효과적으로 분석하여 대화의 정확성과 관련성을 높이는 데 기여합니다.

- **Technical Details**: CA-BERT는 BERT 아키텍처를 수정하여 다중 턴 대화의 맥락 필요성 분류를 위한 맞춤형 구조를 도입했습니다. 주요 개선 사항으로는 드롭아웃 레이어와 이진 분류기를 추가하여 '맥락 필요' 또는 '맥락 불필요'를 예측하는 기능을 강화했습니다. 이를 통해 효율성이 높아지고, 훈련 데이터에서 수집한 다중 대화 샘플을 활용하여 성과를 평가했습니다.

- **Performance Highlights**: CA-BERT는 기존의 BERT 모델 대비 높은 정확도와 효율성을 보여주었으며, 훈련 시간과 자원 사용량을 획기적으로 줄였습니다. 이번 연구는 네트워크의 맥락 인지 능력을 향상시킴으로써 자동화된 대화 시스템에서 사용자 경험과 상호작용 품질을 개선하는 데 기여할 것으로 기대됩니다.



### MAS4POI: a Multi-Agents Collaboration System for Next POI Recommendation (https://arxiv.org/abs/2409.13700)
Comments:
          14 pages, 4 figures

- **What's New**: 이 논문은 LLM(대형 언어 모델)을 기반으로 한 다중 에이전트 시스템(MAS4POI)을 제안하여 사용자 맞춤형 다음 POI(관심 지점) 추천의 성능을 향상시키고자 합니다. MAS4POI는 데이터 에이전트(DataAgent), 관리자(Manager), 분석가(Analyst), 반영기(Reflector), 사용자 에이전트(UserAgent), 탐색자(Searcher), 내비게이터(Navigator) 등 7개의 전문화된 에이전트를 포함하여 다각적인 협력 프로세스를 지원합니다.

- **Technical Details**: MAS4POI는 서로 다른 LLM을 통합하여 작업 워크플로우 및 리소스 관리, 사용자 데이터 분석, 외부 데이터 접근 등을 수행합니다. 각 에이전트는 POI 추천을 위한 상호작용을 통해 데이터 정교화 및 경로 계획을 지원하며, 시스템은 대규모 실세계 데이터셋을 통해 검증됩니다.

- **Performance Highlights**: MAS4POI는 다음 POI 추천의 정확성을 크게 향상시키며, 사용자 개인화된 서비스 제공 및 실시간 Q&A 기능을 통해 다양한 응용 분야에 쉽게 연결될 수 있습니다. 또한, 데이터 부족 문제를 완화하고 여러 대규모 데이터셋을 통해 시스템의 효과를 검증하였습니다.



### Prompt Baking (https://arxiv.org/abs/2409.13697)
Comments:
          25 pages, 8 figures

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)의 행동을 변경하기 위해 프롬프트(Prompt)를 가중치(Weight)에 '베이킹(Baking)'하는 새로운 기술을 제안합니다. 이를 통해 요청된 프롬프트에 따라 LLM이 행동하도록 만드는 방법을 제시합니다.

- **Technical Details**: '프롬프트 베이킹'은 프롬프트 $u$와 초기 가중치 $	heta$를 새로운 가중치 셋 $	heta_u$로 변환하여, 새로운 LLM이 원래의 프롬프트된 LLM처럼 행동하도록 하는 과정입니다. 이는 KL divergence를 최소화하는 방식으로 작동하며, 프롬프트를 가중치 업데이트로 변환하여 재사용성을 높입니다.

- **Performance Highlights**: 연구 결과, 프롬프트를 베이킹함으로써 여러 벤치마크(GSM8K, ASDiv, MBPP, ARC-Easy, ARC-Challenge, CommonsenseQA)에서 제로샷(zero-shot) 성능이 개선되었습니다. 또한, 뉴스 헤드라인을 베이킹함으로써 LLM의 지식을 직접 업데이트할 수 있으며, 장기적인 시퀀스에서는 '프롬프트 망각(prompt forgetting)'을 완화할 수 있습니다. 재프롬프트와 재베이킹을 통해 성능이 더욱 향상되며, 이를 반복적으로 수행하는 '프롬프트 추적(Prompt Pursuit)' 방식을 통해 인스트럭션 따라하기 성능에서 극적인 성능 향상을 보였습니다.



### You Only Use Reactive Attention Slice For Long Context Retrieva (https://arxiv.org/abs/2409.13695)
- **What's New**: 이 논문에서는 Attention을 기반으로 한 새로운 검색 기술인 You Only Use Reactive Attention slice (YOURA)를 제안합니다. 기존의 Retrieval Augmented Generation (RAG) 기법이 긴 맥락을 처리하는 데 한계가 있던 점을 개선하여, 모델이 긴 입력 맥락을 효과적으로 활용하도록 합니다.

- **Technical Details**: YOURA는 입력 문맥에서 문장의 관련성을 평가하기 위해 reaction score라는 새로운 검색 휴리스틱을 사용합니다. 각 토큰의 Attention 점수가 쿼리에 어떻게 "반응"하는지를 측정하여 가장 반응이 큰 문장을 검색합니다. 이 과정에서 Embedding-Agnostic Sentence Yield (EASY) 알고리즘을 활용하여 각 문장을 토큰 인덱스 벡터에 매핑합니다.

- **Performance Highlights**: YOURA는 LongBench QA 데이터셋에서 최대 30% 향상된 vLLM 추론 처리량을 달성하며, 질문 응답 품질을 10% 향상시킵니다. EASY 알고리즘은 문장-토큰 인덱스 매핑 정확도를 93% 이상 기록했습니다.



### A Knowledge-Centric Benchmarking Framework and Empirical Study for Retrieval-Augmented Generation (https://arxiv.org/abs/2409.13694)
Comments:
          14 pages, 11 figures; Mingyue Cheng is the corresponding author

- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 새로운 벤치마크를 제안하고, KDD Cup 2024 CRAG 대회의 데이터셋을 활용하여 RAG의 성능을 분석합니다. 여기서 HTML 형식의 웹 페이지를 Markdown 형식으로 변환하여 LLM이 정보를 효율적으로 활용할 수 있도록 하였습니다.

- **Technical Details**: RAG 모델은 내부 LLM 지식과 외부 지식 소스의 효과적인 융합을 목표로 합니다. 연구에서는 RAG의 전 과정(지식 소스 선택, 검색, 정리 및 추론)을 심도 있게 분석하고, 자동화된 지식 소스 선택 및 노이즈 청크의 영향 등을 조사했습니다. 또한, 하이퍼파라미터 설정에 따른 성능 변화를 분석하였습니다.

- **Performance Highlights**: RAG-X 프레임워크는 CRAG 기본 모델 및 LLM 전용 모델보다 일관되게 우수한 성과를 보였으며, 구조화된 Mock API의 데이터는 비구조화된 웹 출처에 비해 정확도를 향상시키고 환각(hallucination)률을 감소시켰습니다. 그러나 외부 지식 소스의 입력을 늘릴수록 정확도가 개선되지만 환각률도 소폭 증가하는 결과를 보였습니다.



### Declarative Integration and Management of Large Language Models through Finite Automata: Application to Automation, Communication, and Ethics (https://arxiv.org/abs/2409.13693)
Comments:
          Submitted to IAAI-2025, Philadelphia, PA

- **What's New**: 이 논문에서는 공유 히스토리(shared histories)와 트리거(triggers)를 사용하여 주어진 작업에 가장 적합한 대형 언어 모델(Large Language Models, LLMs)을 선언적으로 결합할 수 있는 혁신적인 아키텍처를 제안합니다.

- **Technical Details**: 이 접근 방식은 유한 오토마타(finite automata)와 이벤트 관리 시스템(event management system)을 기반으로 하며, 프로그래밍 노력을 최소화하면서 LLM의 복잡한 통합을 지원합니다. 특히, 긍정 심리학(positive psychology) 방법을 AI와 통합하는 데 유용합니다. 아키텍처 설계 과정은 상태 정의, 트리거 우선순위 설정, LLM의 프롬프트 작성 등의 단계를 포함합니다.

- **Performance Highlights**: 이 프레임워크는 다양한 예제를 통해 그 유연성을 입증했으며, 기차 티켓 예약 자동화, 비폭력적 의사소통(non-violent communication) 계획, LLM의 윤리적 이슈 예방과 관련된 예제를 포함합니다. 이를 통해 복잡한 멀티모달 시스템에서도 효과적인 LLM 통합이 가능함을 보여주었습니다.




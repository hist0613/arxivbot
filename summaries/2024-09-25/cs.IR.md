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



